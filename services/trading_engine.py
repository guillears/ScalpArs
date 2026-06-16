"""
SCALPARS Trading Platform - Trading Engine
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from sqlalchemy import select, update, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import Order, Transaction, BotState, PairData, BnbSwapLog, PhantomFlip
from database import AsyncSessionLocal
import config
from config import save_trading_config, TradingConfig
from services.binance_service import binance_service, _leverage_blocked_pairs
from services.indicators import calculate_indicators, get_signal, check_exit_conditions, calculate_pnl, determine_macro_regime, is_signal_direction_active, gap_expand_marginal
from services.regime import classify_btc_regime
from services.websocket_tracker import websocket_tracker

logger = logging.getLogger(__name__)

OHLCV_BATCH_SIZE = 10
OHLCV_BATCH_DELAY = 5.0

# Cache for open orders to enable fast real-time stop loss checks
_open_orders_cache: Dict[str, List[Dict]] = {}  # pair -> list of order info
_cache_lock = asyncio.Lock()
_close_lock = asyncio.Lock()

# Orders whose exit failed but whose position still exists on Binance.
# Maps order_id -> attempt count. Retried each monitor cycle until success,
# EXTERNAL_CLOSE, or max retries reached.
_exit_retry_queue: Dict[int, int] = {}
_EXIT_RETRY_MAX = 30

# Current BTC macro regime, updated by scan_and_trade, read by update_open_positions
_current_btc_regime: str = "NEUTRAL"


def _calculate_quality_score(direction: str, entry_rsi, entry_adx, entry_gap,
                              bull_pct, bear_pct, btc_adx, pair_ema20_slope) -> int:
    """Calculate entry quality score (0-6). Higher = more favorable conditions aligned."""
    score = 0
    if direction == "LONG":
        if entry_rsi is not None and 55 <= entry_rsi < 60: score += 1
        if entry_adx is not None and 20 <= entry_adx < 25: score += 1
        if entry_gap is not None and 0.25 <= entry_gap <= 0.50: score += 1
        if bull_pct is not None and bull_pct > 50: score += 1
        if btc_adx is not None and 20 <= btc_adx < 25: score += 1
        if pair_ema20_slope is not None and abs(pair_ema20_slope) > 0.12: score += 1
    else:  # SHORT
        if entry_rsi is not None and 30 <= entry_rsi < 35: score += 1
        if entry_adx is not None and 25 <= entry_adx < 30: score += 1
        if entry_gap is not None and 0.20 <= entry_gap <= 0.40: score += 1
        if bear_pct is not None and bear_pct > 65: score += 1
        if btc_adx is not None and 30 <= btc_adx < 35: score += 1
        if pair_ema20_slope is not None and abs(pair_ema20_slope) > 0.10: score += 1
    return score

# Global volume ratio: sum(current volumes) / sum(20-bar avg volumes) across top pairs.
# Computed at end of each scan, used by next scan cycle as a market regime gate.
_global_volume_ratio: float = 1.0
_btc_ema20_slope_pct: float = 0.0
# BTC Trend Filter state (May 5) — EMA20 vs EMA50 medium-term trend.
# Updated in BTC scan loop. Surfaced in /api/engine/state for header badge.
_current_btc_ema20: Optional[float] = None
_current_btc_ema13: Optional[float] = None  # May 6 — BTC Trend Filter switched to EMA13/EMA50
_current_btc_price: Optional[float] = None  # May 14 — BTC price for BTC Market Extension dimension
_current_btc_1h_slope: Optional[float] = None  # May 14 — BTC 1h EMA20 slope (higher-TF macro context)
_current_btc_ema50: Optional[float] = None
_current_btc_trend_gap_pct: Optional[float] = None  # As of May 6: (EMA13 - EMA50) / EMA50; was EMA20-based before
# Module-level BTC indicators for regime classification at exit time
_current_btc_adx: float = None
_current_btc_rsi: float = None
_market_bull_pct: float = 0.0
_market_bear_pct: float = 0.0
_breadth_n_bull: int = 0
_breadth_n_bear: int = 0
_breadth_n_neutral: int = 0
_breadth_n_total: int = 0

# Phantom Tick Momentum shadow configs: (label, windows, delta_or_deltas)
# delta_or_deltas: float = uniform delta for all windows, list = per-window deltas
_SHADOW_TICK_CONFIGS = [
    ('a', [15, 30, 45], 0.15),
    ('b', [30, 45, 60], 0.12),
    ('c', [30, 45, 60], 0.15),
    ('d', [30, 60, 90], 0.12),
    ('e', [30, 60, 90], 0.15),
    ('f', [30, 60, 90], [0.08, 0.12, 0.18]),
    ('g', [60, 90, 120], 0.15),
]


def _check_tick_momentum_fade(tick_buf, now, windows, per_window_deltas, direction):
    """Check if all windows confirm momentum fading. Returns True if confirmed."""
    min_window = min(windows) if windows else 15
    if len(tick_buf) < 5 or (now - tick_buf[0][0]) < min_window:
        return False

    smooth_cutoff = now - 5.0
    smooth_prices = [p for t, p in tick_buf if t >= smooth_cutoff]
    smoothed = sum(smooth_prices) / len(smooth_prices) if smooth_prices else tick_buf[-1][1]

    for w, delta in zip(windows, per_window_deltas):
        target_time = now - w
        best_tick = None
        best_diff = float('inf')
        for t, p in tick_buf:
            diff = abs(t - target_time)
            if diff < best_diff:
                best_diff = diff
                best_tick = p
        if best_tick is None or best_diff > w * 0.5:
            return False
        price_change_pct = ((smoothed - best_tick) / best_tick) * 100
        if direction == "LONG" and price_change_pct > -delta:
            return False
        elif direction == "SHORT" and price_change_pct < delta:
            return False
    return True


# ===================== LEASH SHADOW START (May 30, 2026 — observation-only) =====================
# Virtual trailing leashes run alongside the real exit to measure the true net of a
# runner-tuned exit on the high-stretch LONG profile (separates XLM-clean-capture from
# NEAR-trap-mirage that coarse snapshots can't). Each leash respects the SAME live exits
# (hard SL, EMA13 cross, signal-lost) and only swaps the trailing width. NEVER affects
# live trading — all logic is wrapped in try/except and isolated in this module dict.
# TO REMOVE: delete every fenced "LEASH SHADOW" block (grep "LEASH SHADOW") + all
# shadow_* columns in models.py/database.py + the report block in main.py + the UI block
# in templates/index.html. See CLAUDE.md May 30 / May 31 entries.
import time as _leash_time
_LEASH_STATE = {}  # order_id -> {'rmax', 'ts', 'exits': {name: (pnl, reason)}}
# spec: (name, kind, tight_width, wide_width, switch_threshold)
_LEASH_SPECS = [
    ('tight', 'flat', 0.25, 0.25, 0.0),   # SANITY — should land ~= actual close
    ('wide',  'flat', 0.60, 0.60, 0.0),   # "just loosen it" contrast
    ('tierA', 'tier', 0.25, 0.80, 1.0),   # runner design, conservative
    ('tierB', 'tier', 0.30, 1.00, 1.0),   # runner design, the params that flipped the rough sim
]
_LEASH_ACT = 0.45    # trailing activation (matches live tp_min=0.45 V_S/S_B; was 0.5 — stale, missed 0.45-0.50 peakers the live trailing armed, e.g. 1000PEPE peak 0.485)
_LEASH_SL = -0.7    # hard SL floor (matches live)
# Stretch-exit variants (May 30 ext): exit on EXTENSION fade, not price pullback.
# Live stretch = signed %-distance of price from EMA5 (positive = favorable extension).
#   strpk*  = exit when live stretch retraces to <= Kx PEAK stretch (stretch-trail from peak).
#             K bracket (May 31): 0.5 (strpk) / 0.4 (strpk04) / 0.3 (strpk03) — LOWER K = looser
#             trail = holds the runner longer (more on runners, more giveback on reversers).
#             The cohort settles K the same way tierA/tierB bracket the price-trail params.
#   stren = exit when live stretch falls back to <= ENTRY stretch (extension collapsed to entry)
_STRPK_K = {'strpk': 0.5, 'strpk04': 0.4, 'strpk03': 0.3}
# Jun 1: strpk_signed = hold the runner FULLY while price stays above EMA5, exit
# only when favorable stretch is lost (price crosses back below EMA5, signed ≤ 0).
# Looser than strpk(K=0.5) on partial pullbacks — the candidate to catch Type-B
# monsters (HOME: pulled back to +0.77 but stayed near EMA5, then re-ran +7.58).
_STRETCH_NAMES = ('strpk', 'strpk04', 'strpk03', 'stren', 'strpk_signed')

# ===== PHANTOM FLIP TRACKER (Jun 13, observation-only) =====
# When an entry is BLOCKED by fan-ratio / ATR×gap / pair-trend, simulate the OPPOSITE
# ("fade") position with a real entry/SL/trailing exit on live ws prices — to measure
# whether the reversion the block implies actually pays or just whipsaws. NEVER affects
# live trading; all logic is fail-silent and isolated. Seeds are DE-DUPED (the filters
# fire every scan cycle a pair sits in the zone) — one phantom per pair|source per
# cooldown window. TO REMOVE: grep "PHANTOM_FLIP" / "_seed_phantom_flip" + the model +
# the main.py perf block + the UI block.
_PHANTOM_FLIP_STATE = {}      # key "pair|source|ts" -> live state
_PFLIP_COOLDOWN = {}          # "pair|source" -> last seed epoch (dedupe distinct episodes)
_PFLIP_ACT = 0.45             # trailing arm (raw price-move %, matches live tp_min)
_PFLIP_SL = -0.70             # base hard SL (fresh hypothetical → base, not signal-active wide)
_PFLIP_PB = 0.25              # trailing pullback
_PFLIP_MAX_MIN = 45           # max tracking horizon
_PFLIP_COOLDOWN_MIN = 30      # min minutes between phantoms for the same pair|source


def reset_phantom_flip_state():
    """Clear the in-memory phantom-flip tracking (open virtual fades + per-pair|source
    cooldowns). Called on a full data reset so a fresh batch starts with no carryover —
    the persisted phantom_flips rows are deleted separately in the /api/reset handler.
    Mutates the existing dicts in place (no rebinding) so all module references stay live."""
    _PHANTOM_FLIP_STATE.clear()
    _PFLIP_COOLDOWN.clear()


def _seed_phantom_flip(pair, entry_price, blocked_direction, source, cohort=None, entry_fields=None):
    """Seed a virtual opposite-direction position when an entry is blocked. Fail-silent.
    De-duped: skips if an active phantom exists for pair|source or one was seeded within
    the cooldown (the block filters re-fire every scan cycle the pair stays in the zone).
    cohort: for LONG_UNMATCHED_ONLY only — "C+W"/"C"/"W" pattern family of the blocked
    long, so the fade can be sub-divided downstream (None for other sources).
    entry_fields (Jun 15): the _flip_entry_fields() dict so the persisted PhantomFlip row
    carries full entry context (RSI/ATR/fan-ratio/regime) for cross-batch analysis."""
    try:
        if not entry_price or entry_price <= 0 or blocked_direction not in ("LONG", "SHORT"):
            return
        # Build the entry-context dict (+ regime from globals, as open_position does).
        _ef = dict(entry_fields or {})
        _g = globals()
        if 'entry_macro_trend' not in _ef:
            _ef['entry_macro_trend'] = _g.get('_current_btc_regime')
        if 'entry_btc_regime' not in _ef:
            try:
                _ef['entry_btc_regime'] = classify_btc_regime(
                    _g.get('_current_btc_adx'), _g.get('_current_btc_rsi'), _g.get('_btc_ema20_slope_pct'))
            except Exception:
                pass
        ck = f"{pair}|{source}"
        _now = _leash_time.time()
        _last = _PFLIP_COOLDOWN.get(ck, 0)
        if _now - _last < _PFLIP_COOLDOWN_MIN * 60:
            return
        if any(v.get('pair') == pair and v.get('source') == source for v in _PHANTOM_FLIP_STATE.values()):
            return
        # bounded self-clean
        if len(_PHANTOM_FLIP_STATE) > 200:
            _cut = _now - 3 * 3600
            for k in [k for k, v in _PHANTOM_FLIP_STATE.items() if v.get('open_ts', 0) < _cut]:
                _PHANTOM_FLIP_STATE.pop(k, None)
        _PFLIP_COOLDOWN[ck] = _now
        _PHANTOM_FLIP_STATE[f"{ck}|{_now:.0f}"] = {
            'pair': pair, 'source': source, 'blocked_dir': blocked_direction,
            'flip_dir': "SHORT" if blocked_direction == "LONG" else "LONG",
            'entry': entry_price, 'open_ts': _now, 'cohort': cohort, '_ef': _ef,
            'peak': 0.0, 'trough': 0.0, 'armed': False, '_last_pnl': 0.0,
        }
    except Exception:
        pass


# ===== FLIP ENTRY SLEEVE (Jun 14) =====
# Promote a proven Phantom-Flip cell to a LIVE naked mean-reversion entry: when a
# listed filter blocks an entry, open the OPPOSITE direction with the SAME exit model
# the phantom measured the edge under (SL/arm/trail/horizon, reusing the _PFLIP_*
# constants above). All flip code is fail-silent + isolated so it can NEVER break the
# momentum bot. Registry config: thresholds.flip_entry_sources = "SOURCE:size_mult,...".
# TO REMOVE: grep "FLIP ENTRY" / "flip_source" / "_flip_" / "FLIP_" + the entry_strategy
# column + the main.py Flip Entry perf block + the UI block.
def _flip_registry():
    """Parse the flip-entry registry into {SOURCE: (size_mult, lev_mult)}. A source listed
    here is LIVE for BOTH directions. Master kill-switch = thresholds.flip_entry_enabled.
    Format: SOURCE:size_mult:lev_mult (lev optional → 1.0; bare SOURCE → 1.0/1.0).
    Fail-silent → empty dict (sleeve off)."""
    try:
        th = config.trading_config.thresholds
        if not getattr(th, 'flip_entry_enabled', False):
            return {}
        out = {}
        for part in (getattr(th, 'flip_entry_sources', '') or '').split(','):
            part = part.strip()
            if not part:
                continue
            bits = [b.strip() for b in part.split(':')]
            name = bits[0]
            if not name:
                continue
            def _pf(x, d=1.0):
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return d
            size = _pf(bits[1]) if len(bits) > 1 and bits[1] else 1.0
            lev = _pf(bits[2]) if len(bits) > 2 and bits[2] else 1.0
            out[name] = (size, lev)
        return out
    except Exception:
        return {}

def _flip_active(source):
    return source in _flip_registry()

def _flip_size_mult(source):
    return _flip_registry().get(source, (1.0, 1.0))[0]

def _flip_lev_mult(source):
    return _flip_registry().get(source, (1.0, 1.0))[1]

def _flip_filters(source, ind):
    """Source-namespaced flip filter layer (Jun 16). Given the flip's `source` and the
    blocked entry's `indicators`, decide whether to VETO the flip, how much to SIZE it, and
    which EXIT mode to use. Each source is an INDEPENDENT branch with its own config namespace
    and its own filter TYPES (FAN uses stretch+regime+strpk+mult; future LONG_UNMATCHED /
    PAIR_RSI_OB branches define their own). Fully fail-open: any error → (False, None, 1.0,
    None) so a filter bug can never block a flip or break the scan.
    Returns (blocked: bool, reason: str|None, size_mult: float, lev_mult: float, exit_mode: str|None)."""
    try:
        th = config.trading_config.thresholds
        if source == "FAN_RATIO_GATE":
            stretch = ind.get('ema5_stretch')
            brsi = ind.get('btc_rsi'); badx = ind.get('btc_adx')
            # 1) thin-fuel block
            smin = float(getattr(th, 'flip_fan_stretch_min', 0.0) or 0.0)
            if smin > 0 and stretch is not None and stretch < smin:
                return (True, "FLIP_FAN_STRETCH", 1.0, 1.0, None)
            # 2) regime block — fade into a strong, un-exhausted bull
            rmin = float(getattr(th, 'flip_fan_block_btc_rsi', 0.0) or 0.0)
            amin = float(getattr(th, 'flip_fan_block_btc_adx', 0.0) or 0.0)
            if rmin > 0 and amin > 0 and brsi is not None and badx is not None and brsi >= rmin and badx >= amin:
                return (True, "FLIP_FAN_REGIME", 1.0, 1.0, None)
            # 3) size/lev multiplier cells. Format matches the other multiplier cells:
            #    btc_rsi_lo-hi : btc_adx_lo-hi : size_mult [: lev_mult]  (lev optional, defaults 1.0)
            size = 1.0; lev = 1.0
            rule = (getattr(th, 'flip_fan_mult_rule', '') or '').strip()
            if rule and brsi is not None and badx is not None:
                for cellspec in rule.split(','):
                    try:
                        parts = [p.strip() for p in cellspec.strip().split(':')]
                        if len(parts) < 3:
                            continue
                        rlo, rhi = map(float, parts[0].split('-')); alo, ahi = map(float, parts[1].split('-'))
                        if rlo <= brsi < rhi and alo <= badx < ahi:
                            size = float(parts[2])
                            lev = float(parts[3]) if len(parts) >= 4 and parts[3] else 1.0
                            break
                    except (ValueError, TypeError):
                        continue
            # 4) exit mode
            exitm = "strpk" if getattr(th, 'flip_fan_runner_strpk', False) else None
            return (False, None, size, lev, exitm)
        # LONG_UNMATCHED_ONLY / PAIR_RSI_OB: no entry filters yet (their own data pending), but
        # Jun 16 they DO share the SHORT runner stretch-trail exit via flip_runner_strpk_shorts.
        return (False, None, 1.0, 1.0, ("strpk" if getattr(th, 'flip_runner_strpk_shorts', False) else None))
    except Exception:
        return (False, None, 1.0, 1.0, None)

def _leash_update(order_id, pnl_pct, peak_hint=None, ema13_crossed=False, signal_lost=False,
                  stretch=None, entry_stretch=None):
    """Update virtual leashes for one order on a price tick. Observation-only; fail-silent."""
    try:
        if order_id is None or pnl_pct is None:
            return
        st = _LEASH_STATE.get(order_id)
        if st is None:
            if len(_LEASH_STATE) > 100:  # bounded self-cleaning
                _cut = _leash_time.time() - 3600
                for _k in [k for k, v in _LEASH_STATE.items() if v.get('ts', 0) < _cut]:
                    _LEASH_STATE.pop(_k, None)
            st = {'rmax': pnl_pct, 'ts': _leash_time.time(), 'open_ts': _leash_time.time(),
                  'exits': {n: None for n, _, _, _, _ in _LEASH_SPECS},
                  'exit_mins': {n: None for n, _, _, _, _ in _LEASH_SPECS},
                  'sexits': {n: None for n in _STRETCH_NAMES},
                  'sexit_mins': {n: None for n in _STRETCH_NAMES},
                  'pstretch': None, 'estretch': entry_stretch}
            _LEASH_STATE[order_id] = st
        st['ts'] = _leash_time.time()
        if st.get('estretch') is None and entry_stretch is not None:
            st['estretch'] = entry_stretch
        if peak_hint is not None and peak_hint > st['rmax']:
            st['rmax'] = peak_hint
        if pnl_pct > st['rmax']:
            st['rmax'] = pnl_pct
        rmax = st['rmax']
        if rmax < _LEASH_ACT:
            return  # not armed yet — leash inactive, other exits own the trade
        # track peak favorable stretch once armed
        if stretch is not None and (st['pstretch'] is None or stretch > st['pstretch']):
            st['pstretch'] = stretch
        # ---- price-leash exits ----
        for name, kind, tight, wide, switch in _LEASH_SPECS:
            if st['exits'][name] is not None:
                continue  # already exited
            if pnl_pct <= _LEASH_SL:
                st['exits'][name] = (_LEASH_SL, 'hard_sl'); continue
            if ema13_crossed:
                st['exits'][name] = (round(pnl_pct, 4), 'ema13'); continue
            if signal_lost:
                st['exits'][name] = (round(pnl_pct, 4), 'signal_lost'); continue
            width = wide if (kind == 'tier' and rmax >= switch) else tight
            if pnl_pct <= rmax - width:
                st['exits'][name] = (round(rmax - width, 4), 'trailing')
        # ---- stretch-exits (fire at current P&L when extension fades; same backstops) ----
        if stretch is not None:
            for sname in _STRETCH_NAMES:
                if st['sexits'][sname] is not None:
                    continue
                if pnl_pct <= _LEASH_SL:
                    st['sexits'][sname] = (_LEASH_SL, 'hard_sl'); continue
                if ema13_crossed:
                    st['sexits'][sname] = (round(pnl_pct, 4), 'ema13'); continue
                if signal_lost:
                    st['sexits'][sname] = (round(pnl_pct, 4), 'signal_lost'); continue
                if sname in _STRPK_K:
                    pk = st.get('pstretch')
                    if pk is not None and pk > 0 and stretch <= pk * _STRPK_K[sname]:
                        st['sexits'][sname] = (round(pnl_pct, 4), 'stretch')
                elif sname == 'strpk_signed':
                    # exit only when favorable extension is fully lost (EMA5 cross-back)
                    pk = st.get('pstretch')
                    if pk is not None and pk > 0 and stretch <= 0:
                        st['sexits'][sname] = (round(pnl_pct, 4), 'ema5_cross')
                elif sname == 'stren':
                    es = st.get('estretch')
                    if es is not None and stretch <= es:
                        st['sexits'][sname] = (round(pnl_pct, 4), 'stretch')
        # ---- stamp fire-minute (from open) on whichever leash just fired this tick ----
        _emin = round((st['ts'] - st['open_ts']) / 60.0, 2)
        for _n in st['exits']:
            if st['exits'][_n] is not None and st['exit_mins'].get(_n) is None:
                st['exit_mins'][_n] = _emin
        for _sn in st['sexits']:
            if st['sexits'][_sn] is not None and st['sexit_mins'].get(_sn) is None:
                st['sexit_mins'][_sn] = _emin
    except Exception:
        pass  # observation-only: a shadow error must NEVER affect trading

def _leash_finalize(order_id, fallback_pnl):
    """Pop leash state -> {name:(pnl,reason)} (price + stretch variants) + '_peak_stretch'. Unfired -> 'window'."""
    out = {}
    try:
        st = _LEASH_STATE.pop(order_id, None)
        for name, _, _, _, _ in _LEASH_SPECS:
            if st and st['exits'].get(name) is not None:
                out[name] = st['exits'][name]
                out[name + '_min'] = st.get('exit_mins', {}).get(name)
            else:
                out[name] = (round(fallback_pnl, 4) if fallback_pnl is not None else None, 'window')
                out[name + '_min'] = None  # unfired (held to window end)
        for sname in _STRETCH_NAMES:
            if st and st.get('sexits', {}).get(sname) is not None:
                out[sname] = st['sexits'][sname]
                out[sname + '_min'] = st.get('sexit_mins', {}).get(sname)
            else:
                out[sname] = (round(fallback_pnl, 4) if fallback_pnl is not None else None, 'window')
                out[sname + '_min'] = None
        out['_peak_stretch'] = round(st['pstretch'], 4) if (st and st.get('pstretch') is not None) else None
    except Exception:
        pass
    return out
# ====================== LEASH SHADOW END ======================


def _compute_pattern_c_match(direction, rng_pos, pair_gap, adx_delta,
                             btc_rsi, btc_rsi_prev, btc_adx, btc_adx_prev,
                             btc_gap, stretch, pair_adx, btc_atr,
                             ema20_slope=None, ema50_slope=None):
    """Pattern C Tracker (May 19-20, 2026 — observation-only).

    Evaluates 9 candidate Pattern C precursor signatures at entry. Returns
    (c1, c2, c3, c4, c5, c6, c7, c8, c9, c_any) — all booleans or None if
    tracker disabled.

    Pattern C = trade peaks <+0.10% (never positive). Multiple structural
    causes are tested simultaneously:
      C1: Capitulation/Climax chase — extreme RngPos + extreme Pair Gap + fast ADXΔ
      C2: Macro counter-trend — BTC RSI/ADX reversing against trade direction
      C3: Stretch exhaustion — high EMA5 stretch + strong Pair ADX + extreme RngPos
      C4: Low-vol chop — low BTC ATR + low BTC ADX + low Pair ADX (no momentum)
      C5: Slow Climber Death — weak Pair ADX + low ADXΔ + flat EMA20 slope (May 19)
      C6: Macro over-extended same direction — BTC RSI/ADX/gap all aligned WITH
          trade direction at climactic strength (BTC about to revert) (May 19)
      C7: Pair Countertrend Bounce — pair deeply against 4hr trend + EMA50 slope
          confirming + mid-range entry (dead-cat LONG / failed-breakdown SHORT) (May 20)
      C8: Oversold/Overbought Chop — range extreme + sharp ADXΔ + NO clear pair
          trend (|gap|≤0.20) + low BTC vol. (May 20-late)
      C9: Low-vol Countertrend Chop — C4 base + MILD countertrend pair_gap
          (LONG: pair_gap ≤ -0.10%; SHORT: pair_gap ≥ +0.10%). The "tight
          C4-LOSS" sub-pattern derived from May 20 C4 LONG deep-dive (EDEN
          losers signature). Different from C7 which needs deep countertrend
          (≤-0.50%) + slope confirmation. (May 20-latest)

    No behavior change. Pure capture for cross-batch validation at N≥30 per
    pattern. See CLAUDE.md May 19-20 entries for promotion gates.
    """
    import config as _cfg
    th = _cfg.trading_config.thresholds
    if not getattr(th, 'pattern_c_tracker_enabled', True):
        return (None, None, None, None, None, None, None, None, None, None)
    if direction not in ('LONG', 'SHORT'):
        return (None, None, None, None, None, None, None, None, None, None)

    # Helper to safely evaluate AND of optional conditions
    def _safe_and(*conds):
        """All conds must evaluate to True. None values fail the AND (return False)."""
        return all(c is True for c in conds)

    if direction == 'SHORT':
        c1 = _safe_and(
            rng_pos is not None and rng_pos <= getattr(th, 'pc_short_c1_rngpos_max', 15.0),
            pair_gap is not None and pair_gap <= getattr(th, 'pc_short_c1_pair_gap_max', -0.50),
            adx_delta is not None and adx_delta >= getattr(th, 'pc_short_c1_adxd_min', 1.0),
        )
        c2 = _safe_and(
            btc_rsi is not None and btc_rsi_prev is not None and btc_rsi > btc_rsi_prev,
            btc_adx is not None and btc_adx_prev is not None and btc_adx < btc_adx_prev,
            btc_gap is not None and btc_gap > getattr(th, 'pc_short_c2_btc_gap_min', -0.05),
        )
        c3 = _safe_and(
            stretch is not None and stretch >= getattr(th, 'pc_short_c3_stretch_min', 0.40),
            pair_adx is not None and pair_adx >= getattr(th, 'pc_short_c3_pair_adx_min', 30.0),
            rng_pos is not None and rng_pos <= getattr(th, 'pc_short_c3_rngpos_max', 15.0),
        )
        c4 = _safe_and(
            btc_atr is not None and btc_atr < getattr(th, 'pc_short_c4_btc_atr_max', 0.15),
            btc_adx is not None and btc_adx < getattr(th, 'pc_short_c4_btc_adx_max', 22.0),
            pair_adx is not None and pair_adx < getattr(th, 'pc_short_c4_pair_adx_max', 25.0),
        )
        # C5 — Slow Climber Death (SHORT mirror): weak ADX + slow accel + flat/weak slope
        c5 = _safe_and(
            pair_adx is not None and pair_adx <= getattr(th, 'pc_short_c5_pair_adx_max', 22.0),
            adx_delta is not None and adx_delta <= getattr(th, 'pc_short_c5_adxd_max', 0.3),
            ema20_slope is not None and ema20_slope >= getattr(th, 'pc_short_c5_ema20_slope_min', -0.05),
        )
        # C6 — Macro over-extended same direction (BTC late-bottom)
        c6 = _safe_and(
            btc_rsi is not None and btc_rsi <= getattr(th, 'pc_short_c6_btc_rsi_max', 35.0),
            btc_adx is not None and btc_adx >= getattr(th, 'pc_short_c6_btc_adx_min', 28.0),
            btc_gap is not None and btc_gap <= getattr(th, 'pc_short_c6_btc_gap_max', -0.15),
        )
        # C7 — Pair Countertrend Bounce (SHORT mirror): pair stretched ABOVE 4hr trend,
        # slope rising, bot shorting mid-range pullback in uptrend = failed-breakdown SHORT
        c7 = _safe_and(
            pair_gap is not None and pair_gap >= getattr(th, 'pc_short_c7_pair_gap_min', 0.50),
            ema50_slope is not None and ema50_slope >= getattr(th, 'pc_short_c7_ema50_slope_min', 0.05),
            rng_pos is not None and rng_pos <= getattr(th, 'pc_short_c7_rngpos_max', 60.0),
        )
        # C8 — Oversold Chop SHORT: at range bottom + sharp ADXΔ + pair has NO clear
        # direction (|gap|≤0.20) + low BTC vol regime. Bot SHORTs deep RSI but chop
        # kills momentum continuation, leading to squeeze instead of breakdown.
        c8 = _safe_and(
            rng_pos is not None and rng_pos <= getattr(th, 'pc_short_c8_rngpos_max', 25.0),
            adx_delta is not None and adx_delta >= getattr(th, 'pc_short_c8_adx_delta_min', 1.0),
            pair_gap is not None and abs(pair_gap) <= getattr(th, 'pc_short_c8_pair_gap_abs_max', 0.20),
            btc_atr is not None and btc_atr <= getattr(th, 'pc_short_c8_btc_atr_max', 0.15),
        )
        # C9 — Low-vol Countertrend Chop SHORT: C4 base (low BTC vol + low BTC ADX
        # + low Pair ADX) PLUS pair_gap ≥ +0.10% (pair is mildly UP-trending against
        # the SHORT direction). The "tight C4-LOSS" sub-pattern for SHORT — bot is
        # SHORTing a pair drifting up in low-vol regime, no momentum either way.
        c9 = _safe_and(
            btc_atr is not None and btc_atr <= getattr(th, 'pc_short_c9_btc_atr_max', 0.15),
            btc_adx is not None and btc_adx <= getattr(th, 'pc_short_c9_btc_adx_max', 22.0),
            pair_adx is not None and pair_adx <= getattr(th, 'pc_short_c9_pair_adx_max', 25.0),
            pair_gap is not None and pair_gap >= getattr(th, 'pc_short_c9_pair_gap_min', 0.10),
        )
    else:  # LONG
        c1 = _safe_and(
            rng_pos is not None and rng_pos >= getattr(th, 'pc_long_c1_rngpos_min', 85.0),
            pair_gap is not None and pair_gap >= getattr(th, 'pc_long_c1_pair_gap_min', 0.50),
            adx_delta is not None and adx_delta >= getattr(th, 'pc_long_c1_adxd_min', 1.0),
        )
        c2 = _safe_and(
            btc_rsi is not None and btc_rsi_prev is not None and btc_rsi < btc_rsi_prev,
            btc_adx is not None and btc_adx_prev is not None and btc_adx < btc_adx_prev,
            btc_gap is not None and btc_gap < getattr(th, 'pc_long_c2_btc_gap_max', 0.05),
        )
        c3 = _safe_and(
            stretch is not None and stretch >= getattr(th, 'pc_long_c3_stretch_min', 0.40),
            pair_adx is not None and pair_adx >= getattr(th, 'pc_long_c3_pair_adx_min', 30.0),
            rng_pos is not None and rng_pos >= getattr(th, 'pc_long_c3_rngpos_min', 85.0),
        )
        c4 = _safe_and(
            btc_atr is not None and btc_atr < getattr(th, 'pc_long_c4_btc_atr_max', 0.15),
            btc_adx is not None and btc_adx < getattr(th, 'pc_long_c4_btc_adx_max', 22.0),
            pair_adx is not None and pair_adx < getattr(th, 'pc_long_c4_pair_adx_max', 25.0),
        )
        # C5 — Slow Climber Death (LONG): weak ADX + slow accel + flat/weak slope
        c5 = _safe_and(
            pair_adx is not None and pair_adx <= getattr(th, 'pc_long_c5_pair_adx_max', 22.0),
            adx_delta is not None and adx_delta <= getattr(th, 'pc_long_c5_adxd_max', 0.3),
            ema20_slope is not None and ema20_slope <= getattr(th, 'pc_long_c5_ema20_slope_max', 0.05),
        )
        # C6 — Macro over-extended same direction (BTC late-top)
        c6 = _safe_and(
            btc_rsi is not None and btc_rsi >= getattr(th, 'pc_long_c6_btc_rsi_min', 65.0),
            btc_adx is not None and btc_adx >= getattr(th, 'pc_long_c6_btc_adx_min', 28.0),
            btc_gap is not None and btc_gap >= getattr(th, 'pc_long_c6_btc_gap_min', 0.15),
        )
        # C7 — Pair Countertrend Bounce (LONG): pair deeply BELOW 4hr trend,
        # slope declining, bot longing mid-range bounce = dead-cat bounce LONG
        c7 = _safe_and(
            pair_gap is not None and pair_gap <= getattr(th, 'pc_long_c7_pair_gap_max', -0.50),
            ema50_slope is not None and ema50_slope <= getattr(th, 'pc_long_c7_ema50_slope_max', -0.05),
            rng_pos is not None and rng_pos >= getattr(th, 'pc_long_c7_rngpos_min', 40.0),
        )
        # C8 — Overbought Chop LONG (mirror): at range top + sharp ADXΔ + pair has NO
        # clear direction (|gap|≤0.20) + low BTC vol regime. Bot LONGs overbought RSI
        # but chop kills follow-through, leading to fade instead of breakout.
        c8 = _safe_and(
            rng_pos is not None and rng_pos >= getattr(th, 'pc_long_c8_rngpos_min', 75.0),
            adx_delta is not None and adx_delta >= getattr(th, 'pc_long_c8_adx_delta_min', 1.0),
            pair_gap is not None and abs(pair_gap) <= getattr(th, 'pc_long_c8_pair_gap_abs_max', 0.20),
            btc_atr is not None and btc_atr <= getattr(th, 'pc_long_c8_btc_atr_max', 0.15),
        )
        # C9 — Low-vol Countertrend Chop LONG (tight C4-LOSS sub-pattern):
        # C4 base + pair_gap ≤ -0.10% (pair mildly DOWN-trending against LONG).
        # Captures EDEN-style "large-cap LONG into countertrend pair + chop" losses
        # that C7 misses because EDEN's slope wasn't ≤ -0.05%. Mild countertrend
        # + low-vol regime = no follow-through, bot rides to SL.
        c9 = _safe_and(
            btc_atr is not None and btc_atr <= getattr(th, 'pc_long_c9_btc_atr_max', 0.15),
            btc_adx is not None and btc_adx <= getattr(th, 'pc_long_c9_btc_adx_max', 22.0),
            pair_adx is not None and pair_adx <= getattr(th, 'pc_long_c9_pair_adx_max', 25.0),
            pair_gap is not None and pair_gap <= getattr(th, 'pc_long_c9_pair_gap_max', -0.10),
        )

    return (c1, c2, c3, c4, c5, c6, c7, c8, c9,
            c1 or c2 or c3 or c4 or c5 or c6 or c7 or c8 or c9)


def _compute_pattern_w_match(direction, rsi, adx, adx_delta, stretch,
                              rng_pos, pair_gap, btc_rsi, btc_adx,
                              btc_atr, btc_gap, pair_vol_ratio):
    """Pattern W (winner tracker) — May 21, 2026: lifted to ENTRY-TIME computation
    from main.py's report-time helper, mirroring Pattern C's pattern.

    Returns (w1, w2, w3, w4, w5, w6, w_any) booleans (or None tuple if direction
    invalid). Direction-aware: LONG and SHORT use mirrored thresholds.

    Signatures (designed from cross-batch winner analysis, May 20-21):
      W1: HighConv trend continuation — strong ADX + accel + stretch
      W2: Macro tailwind — BTC RSI sweet spot + BTC ADX committed + gap aligned
      W3: Energetic volatility breakout — BTC ATR high + above-avg pair vol + stretch
      W4: Pullback entry aligned — mid-range + pair gap aligned + ADX not declining
      W5: Confluence — multiple sweet-spot cells true simultaneously
      W6 (LONG): Healthy BTC Tailwind — BTC ADX 22-26 + Pair Gap ≤ +0.20% (May 21 — 100% WR cross-batch N=14)
      W6 (SHORT): Mature BTC Bear — BTC ADX ≥ 32 (May 21 — 100% WR cross-batch N=25)

    Captured at entry to support live multiplier rules (CLAUDE.md May 21 ship).
    Matches the post-hoc helper in main.py::_compute_pattern_w_match — when
    both fire they MUST produce identical results for the same trade. The
    main.py version reads from the persisted columns; this version computes
    fresh at entry time before the columns exist.
    """
    if direction not in ('LONG', 'SHORT'):
        return (None, None, None, None, None, None, None)

    def _and(*conds):
        return all(c is True for c in conds)

    if direction == 'LONG':
        w1 = _and(
            adx is not None and adx >= 22,
            adx_delta is not None and adx_delta >= 0.5,
            stretch is not None and stretch >= 0.16,
        )
        w2 = _and(
            btc_rsi is not None and 50 <= btc_rsi <= 65,
            btc_adx is not None and btc_adx >= 22,
            btc_gap is not None and btc_gap >= 0.10,
        )
        w3 = _and(
            btc_atr is not None and btc_atr >= 0.20,
            pair_vol_ratio is not None and pair_vol_ratio >= 1.20,
            stretch is not None and stretch >= 0.20,
        )
        w4 = _and(
            rng_pos is not None and 40 <= rng_pos <= 75,
            pair_gap is not None and pair_gap >= 0.10,
            adx_delta is not None and adx_delta >= 0,
        )
        w5 = _and(
            btc_adx is not None and 22 <= btc_adx <= 30,
            btc_rsi is not None and 55 <= btc_rsi <= 65,
            adx is not None and 22 <= adx <= 30,
            stretch is not None and 0.16 <= stretch <= 0.25,
        )
        # W6 LONG — Healthy BTC Tailwind (May 21): BTC ADX in moderate-strong
        # zone (22-26) AND pair NOT extended (gap ≤ +0.20%). Captures "macro
        # doing the work, pair just along for the ride" — different from W2
        # which requires BTC RSI sweet spot AND gap ≥ +0.10%.
        w6 = _and(
            btc_adx is not None and 22 <= btc_adx < 26,
            pair_gap is not None and pair_gap < 0.20,
        )
    else:  # SHORT
        w1 = _and(
            adx is not None and adx >= 22,
            adx_delta is not None and adx_delta >= 0.5,
            stretch is not None and stretch >= 0.20,
        )
        w2 = _and(
            btc_rsi is not None and 30 <= btc_rsi <= 45,
            btc_adx is not None and btc_adx >= 22,
            btc_gap is not None and btc_gap <= -0.10,
        )
        w3 = _and(
            btc_atr is not None and btc_atr >= 0.20,
            pair_vol_ratio is not None and pair_vol_ratio >= 1.20,
            stretch is not None and stretch >= 0.25,
        )
        w4 = _and(
            rng_pos is not None and 25 <= rng_pos <= 60,
            pair_gap is not None and pair_gap <= -0.10,
            adx_delta is not None and adx_delta >= 0,
        )
        w5 = _and(
            btc_adx is not None and 22 <= btc_adx <= 30,
            btc_rsi is not None and 30 <= btc_rsi <= 40,
            adx is not None and 22 <= adx <= 30,
            stretch is not None and 0.20 <= stretch <= 0.30,
        )
        # W6 SHORT — Mature BTC Bear (May 21): single-axis BTC ADX ≥ 32
        # captures the late-stage committed-bearish-move zone where the trend
        # is established and SHORTs ride continuation. W1/W2/W5 cap at BTC ADX
        # 22 / 22 / 22-30 — this extreme zone was a blind spot.
        w6 = _and(
            btc_adx is not None and btc_adx >= 32,
        )

    return (w1, w2, w3, w4, w5, w6,
            w1 or w2 or w3 or w4 or w5 or w6)


class TradingEngine:
    """Main trading engine that manages positions and executes trades"""
    
    def __init__(self):
        self.is_running = False
        self.is_paper_mode = True
        self.paper_balance = config.trading_config.paper_balance
        self.started_at: Optional[datetime] = None
        self.total_runtime_seconds = 0
        self._task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_scan_time: float = 0
        self._initialized = False
        self._post_exit_tracking: Dict[int, dict] = {}
        self._rsi3_history: Dict[int, list] = {}  # per-order RSI history for 3-drop detection
        # Signal re-validation tracking (Amendment #7 / Apr 18)
        # Tracks entries aborted after maker timeout because the signal went stale.
        self.signal_expired_reasons: Dict[str, int] = {}  # reason_code -> count
        self.signal_expired_log_recent = []  # recent expirations for debugging (bounded)
        self._signal_expired_log_max = 200
        # BNB fee management
        self.paper_bnb_balance_usd: float = config.trading_config.paper_bnb_initial_usd
        self._bnb_emergency_threshold: float = 0.0
        self._bnb_projected_need: float = 0.0
        self._bnb_burn_rate: float = 0.0
        # May 25 — "data mature" flag gates AUTO-SWAP decisions (not display).
        # True only when oldest closed trade in 24h window is ≥2h old. Below
        # that threshold, burn rate still updates with every closed order
        # (display + UI accurate), but scheduled/emergency swaps suppressed
        # to avoid extrapolating from a narrow window.
        self._bnb_data_mature: bool = False
        self._last_bnb_check: Optional[datetime] = None
        # Filter Block Counters (May 5) — in-memory tally of pre-entry filter
        # rejections, surfaced via /api/engine/state and the dashboard.
        # Key: (filter_name, direction) → count. Reset on bot start.
        # See CLAUDE.md May 5 entry on BTC Trend Filter for context.
        self._filter_block_counts: Dict[tuple, int] = {}

        # Per-pair last block reason (May 26) — keyed by pair → filter tag.
        # Updated at every _record_filter_block call site. Read by main.py
        # /api/pairs to show Block Reason column without re-enumerating
        # 40+ filters in UI code (single source of truth).
        self._last_pair_block_reason: Dict[str, str] = {}
        # Jun 3: BTC-acceleration-chase filter state (stateful evolution filter).
        # Tracks the BTC EMA20 slope at the most recent LONG that actually opened.
        self._last_long_open_ts: Optional[datetime] = None
        self._last_long_open_btc_ema20_slope: Optional[float] = None

    async def initialize(self, db: AsyncSession):
        """Initialize engine state from database (only on first call).

        Mode resolution:
        - If BotState row exists (normal case): load is_paper_mode from DB.
          This preserves any UI toggle the user has set.
        - If BotState row does NOT exist (cold start on empty DB — the Apr 11
          scenario): default to config.trading_config.paper_trading instead
          of a hardcoded True.  Previous behaviour silently flipped the bot
          to paper mode on any DB loss, which orphaned live positions for
          8 hours on Apr 11.  Now the config file is the cold-start source
          of truth, controllable by the user.

        A loud [MODE] log is emitted on every init so any mode transition
        is immediately visible in CloudWatch and post-mortem logs.
        """
        if self._initialized:
            return

        result = await db.execute(select(BotState).limit(1))
        state = result.scalar_one_or_none()

        if state:
            self.is_running = state.is_running
            self.is_paper_mode = state.is_paper_mode
            self.paper_balance = state.paper_balance
            self.paper_bnb_balance_usd = getattr(state, 'paper_bnb_balance_usd', None) or config.trading_config.paper_bnb_initial_usd
            self.total_runtime_seconds = state.total_runtime_seconds
            # Backfill runtime_initial_total_usd if NULL (column added May 5).
            # One-time backfill: set to current paper_balance + paper_bnb_balance_usd
            # so the baseline reflects "wherever we are now" for existing runs that
            # predate the column. New cold starts use the proper init in the else branch.
            if getattr(state, 'runtime_initial_total_usd', None) is None:
                _backfill_initial = (state.paper_balance or 0) + (self.paper_bnb_balance_usd or 0)
                state.runtime_initial_total_usd = _backfill_initial
                await db.commit()
                logger.warning(
                    f"[BOTSTATE] Backfilled runtime_initial_total_usd=${_backfill_initial:.2f} "
                    f"for existing BotState row. This is a one-time migration — Return Multiple "
                    f"will use this as the immutable baseline going forward."
                )
            if state.is_running and state.started_at:
                self.started_at = state.started_at
            # Restore filter block counters persisted from previous session
            _fb_json = getattr(state, 'filter_block_counts_json', None)
            if _fb_json:
                try:
                    _fb_raw = json.loads(_fb_json)
                    # Format: "filter|direction|room_state" (3 parts) or legacy "filter|direction" (2)
                    restored = {}
                    for k, v in _fb_raw.items():
                        parts = k.split("|")
                        if len(parts) == 3:
                            restored[(parts[0], parts[1], parts[2])] = v
                        elif len(parts) == 2:
                            # Legacy: assume had_room=True
                            restored[(parts[0], parts[1], "ROOM")] = v
                    self._filter_block_counts = restored
                    logger.info(f"[FILTER_BLOCKS] Restored {len(self._filter_block_counts)} counters from DB")
                except Exception as _e:
                    logger.warning(f"[FILTER_BLOCKS] Failed to restore counters: {_e}")
            # Restore last BNB check timestamp so the interval is respected
            # across restarts (May 7 fix).
            _last_bnb_check_db = getattr(state, 'last_bnb_check_at', None)
            if _last_bnb_check_db:
                self._last_bnb_check = _last_bnb_check_db
                logger.info(f"[BNB_CHECK] Restored last_bnb_check from DB: {_last_bnb_check_db.isoformat()}")
            logger.info(
                f"[MODE] Loaded from BotState DB: is_paper_mode={self.is_paper_mode}, "
                f"is_running={self.is_running} — runtime mode recovered from previous session."
            )
        else:
            # Cold start: no BotState row. Read mode default from config file
            # (config.trading_config.paper_trading) rather than a hardcoded True.
            # See docstring above + CLAUDE.md Apr 11 incident for context.
            _default_is_paper = bool(getattr(config.trading_config, 'paper_trading', True))
            logger.critical(
                f"[MODE] COLD START — no BotState row found in DB. "
                f"Defaulting to config.trading_config.paper_trading={_default_is_paper}. "
                f"If this is unexpected (DB wipe / instance replacement / migration), "
                f"investigate immediately — live positions may be orphaned on Binance."
            )
            # Immutable starting capital baseline = paper_balance + paper_bnb_initial_usd
            # Set ONCE here, never updated on config edits. See CLAUDE.md May 5 entry.
            _initial_total = (
                config.trading_config.paper_balance
                + config.trading_config.paper_bnb_initial_usd
            )
            state = BotState(
                is_running=False,  # Never auto-start on cold boot (Apr 11 defense)
                is_paper_mode=_default_is_paper,
                paper_balance=config.trading_config.paper_balance,
                paper_bnb_balance_usd=config.trading_config.paper_bnb_initial_usd,
                runtime_initial_total_usd=_initial_total,
                total_runtime_seconds=0
            )
            db.add(state)
            await db.commit()
            self.is_running = False
            self.is_paper_mode = _default_is_paper
            self.paper_balance = config.trading_config.paper_balance
            self.paper_bnb_balance_usd = config.trading_config.paper_bnb_initial_usd

        # Recalculate paper_balance from orders to self-heal any accumulated drift
        if self.is_paper_mode:
            await self._recalculate_paper_balance(db)
            await self.save_state(db)

        await self._recover_post_exit_tracking(db)
        self._initialized = True

    async def _recover_post_exit_tracking(self, db: AsyncSession):
        """Re-register recently-closed orders for post-exit tracking that was interrupted by a restart.

        On restart, _post_exit_tracking (in-memory) is wiped. Orders whose 45-min window
        spans the restart never get their post_exit_peak_pnl written. This method finds
        those orders and re-registers them for whatever time remains in their window.
        Orders whose window has fully expired (closed_at + tracking_minutes < now) are
        skipped — their data is permanently lost for this run.
        """
        tc = config.trading_config
        if not getattr(tc, 'post_exit_tracking_enabled', False):
            return
        minutes = getattr(tc, 'post_exit_tracking_minutes', 45)
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=minutes)

        try:
            result = await db.execute(
                select(Order).where(
                    Order.status == 'CLOSED',
                    Order.post_exit_peak_pnl.is_(None),
                    Order.closed_at >= cutoff,
                )
            )
            candidates = result.scalars().all()
        except Exception as e:
            logger.warning(f"[POST_EXIT_RECOVER] DB query failed: {e}")
            return

        recovered = 0
        for order in candidates:
            if not order.close_reason or not order.closed_at:
                continue
            # Jun 14: strip FLIP_ (then FL_) so flip exits resolve to base reason.
            _reason_base = order.close_reason
            if _reason_base.startswith("FLIP_"):
                _reason_base = _reason_base[5:]
            if _reason_base.startswith("FL_"):
                _reason_base = _reason_base[3:]
            # May 7: added EMA13_CROSS_EXIT and EMA_STACK_CROSS_EXIT to recovery
            # whitelist. Without them, EMA13/EMA_STACK trades that spanned a
            # bot restart never got post_exit_peak_pnl written → silently
            # missing from Post-Exit Regret Deep Dive table. Live registration
            # whitelist (line ~3171) already had these; recovery had drifted.
            # May 19: same drift caught for FAST_EXIT. Live whitelist (line ~3171)
            # had FAST_EXIT; this recovery path didn't. Added now to align.
            # May 21: added PATTERN_FIXED_TP/SL to recovery whitelist (matches live
            # whitelist at ~line 3663). Without this, Pattern Cell Ship rule trades
            # that close + span a bot restart wouldn't get post_exit_peak_pnl
            # tracked → silently missing from Post-Exit Regret Deep Dive.
            if not (_reason_base.startswith("BREAKEVEN_EXIT") or _reason_base.startswith("SIGNAL_LOST") or
                    _reason_base.startswith("TICK_MOMENTUM_EXIT") or _reason_base.startswith("RSI_MOMENTUM_EXIT") or
                    _reason_base.startswith("RSI_HANDOFF_EXIT") or _reason_base.startswith("EMA13_CROSS_EXIT") or
                    _reason_base.startswith("EMA_STACK_CROSS_EXIT") or _reason_base.startswith("STOP_LOSS") or
                    _reason_base.startswith("REGIME_CHANGE") or _reason_base.startswith("TRAILING_STOP") or
                    _reason_base.startswith("RUNNER_TRAIL") or
                    _reason_base.startswith("MOMENTUM_EXIT") or _reason_base.startswith("SLOPE_EXIT") or
                    _reason_base.startswith("NO_EXPANSION") or _reason_base.startswith("RECOVERED") or
                    _reason_base.startswith("DEEP_STOP") or _reason_base.startswith("EMERGENCY_SL") or
                    _reason_base.startswith("FAST_EXIT") or
                    _reason_base.startswith("ATR_FIXED_TP") or
                    _reason_base.startswith("PATTERN_FIXED_TP") or _reason_base.startswith("PATTERN_FIXED_SL") or
                    # Jun 14: Flip Entry exits — keep post-exit tracking alive across restart
                    _reason_base.startswith("FLIP_")):
                continue

            closed_utc = order.closed_at if order.closed_at.tzinfo else order.closed_at.replace(tzinfo=None)
            tracking_until = closed_utc + timedelta(minutes=minutes)

            tracker = websocket_tracker.get_tracker(order.pair)
            initial_price = tracker.last_price if tracker else order.exit_price

            _pe_notional = order.entry_price * order.quantity if order.quantity else 1
            _pe_fee_drag = (((order.entry_fee or 0) + _pe_notional * getattr(tc, 'taker_fee', tc.trading_fee)) / _pe_notional) * 100

            # May 8: resume tracking from saved running state (survives restart).
            # If running state exists, use it; else fall back to current price + now.
            _resumed = (order.post_exit_running_high is not None or
                        order.post_exit_running_low is not None)
            _post_high = order.post_exit_running_high if order.post_exit_running_high is not None else (initial_price or order.exit_price)
            _post_low = order.post_exit_running_low if order.post_exit_running_low is not None else (initial_price or order.exit_price)
            _peak_at = order.post_exit_running_peak_at if order.post_exit_running_peak_at is not None else now
            _trough_at = order.post_exit_running_trough_at if order.post_exit_running_trough_at is not None else now

            self._post_exit_tracking[order.id] = {
                "order_id": order.id,
                "pair": order.pair,
                "entry_price": order.entry_price,
                "direction": order.direction,
                "fee_drag_pct": _pe_fee_drag,
                "exit_time": order.closed_at,
                "tracking_until": tracking_until,
                "post_high": _post_high,
                "post_low": _post_low,
                "peak_at": _peak_at,
                "trough_at": _trough_at,
                "signal_lost_at": None,
                "pnl_at_signal_lost": None,
                "peak_before_signal_lost": 0.0,
                "rsi_exit_at": None,
                "rsi_exit_pnl": None,
                "rsi3_exit_at": None,
                "rsi3_exit_pnl": None,
                "rsi_history": [],
                "ema13_cross_at": None,
                "ema13_cross_pnl": None,
                # May 23: post-exit regime-flip tracker. entry_regime is the
                # regime at trade open; we watch for first opposite-or-neutral
                # transition during post-exit window.
                "entry_regime": order.entry_btc_regime,
                "regime_flip_at": order.post_exit_regime_flip_at,
                "regime_flip_pnl": order.post_exit_regime_flip_pnl_pct,
                "signal_regained_at": None,
                "pnl_at_signal_regained": None,
                "running_min_pnl": None,
                "floor_before_signal_regain": None,
                "close_reason": order.close_reason,
                "tick_prices": [],
                # May 12 LATE PM: time-bucketed P&L snapshots (1/2/5/15/30 min)
                # Resume from DB if already captured pre-restart
                "pnl_at_1min": order.post_exit_pnl_at_1min,
                "pnl_at_2min": order.post_exit_pnl_at_2min,
                "pnl_at_5min": order.post_exit_pnl_at_5min,
                "pnl_at_15min": order.post_exit_pnl_at_15min,
                "pnl_at_30min": order.post_exit_pnl_at_30min,
            }
            recovered += 1
            _resumed_tag = " (resumed running state)" if _resumed else " (fresh)"
            logger.info(f"[POST_EXIT_RECOVER] Re-registered {order.pair} order {order.id} ({order.close_reason}){_resumed_tag} — "
                        f"tracking_until={tracking_until.strftime('%H:%M:%S')}")

        if recovered:
            logger.info(f"[POST_EXIT_RECOVER] Recovered {recovered} orders for post-exit tracking after restart")

    async def save_state(self, db: AsyncSession):
        """Save engine state to database"""
        result = await db.execute(select(BotState).limit(1))
        state = result.scalar_one_or_none()
        
        # Format: "filter|direction|room_state" (3 parts). Legacy 2-part keys
        # restored on load default to room_state="ROOM" (assumes had_room=True).
        _fb_json = json.dumps({
            "|".join(str(p) for p in (k if len(k) == 3 else (*k, "ROOM"))): v
            for k, v in self._filter_block_counts.items()
        }) if self._filter_block_counts else None

        if state:
            state.is_running = self.is_running
            state.is_paper_mode = self.is_paper_mode
            state.paper_balance = self.paper_balance
            state.paper_bnb_balance_usd = self.paper_bnb_balance_usd
            state.total_runtime_seconds = self.total_runtime_seconds
            state.started_at = self.started_at
            state.updated_at = datetime.utcnow()
            state.filter_block_counts_json = _fb_json
            state.last_bnb_check_at = self._last_bnb_check
        else:
            state = BotState(
                is_running=self.is_running,
                is_paper_mode=self.is_paper_mode,
                paper_balance=self.paper_balance,
                paper_bnb_balance_usd=self.paper_bnb_balance_usd,
                total_runtime_seconds=self.total_runtime_seconds,
                started_at=self.started_at,
                filter_block_counts_json=_fb_json,
                last_bnb_check_at=self._last_bnb_check,
            )
            db.add(state)

        await db.commit()
    
    async def start(self, db: AsyncSession):
        """Start the trading bot"""
        self.is_running = True
        self.started_at = datetime.utcnow()
        await self.save_state(db)
        return {"status": "running", "message": "Bot started"}
    
    async def pause(self, db: AsyncSession):
        """Pause the trading bot (can still close positions)"""
        if self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            self.total_runtime_seconds += int(elapsed)
        
        self.is_running = False
        self.started_at = None
        await self.save_state(db)
        return {"status": "paused", "message": "Bot paused - will still close open positions"}
    
    async def set_paper_mode(self, enabled: bool, db: AsyncSession):
        """Toggle paper trading mode"""
        self.is_paper_mode = enabled
        if enabled:
            self.paper_balance = config.trading_config.paper_balance
        await self.save_state(db)
        return {"paper_mode": enabled}
    
    def get_runtime_seconds(self) -> int:
        """Get total runtime in seconds"""
        if self.is_running and self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            return self.total_runtime_seconds + int(elapsed)
        return self.total_runtime_seconds
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        runtime = self.get_runtime_seconds()
        hours = runtime // 3600
        minutes = (runtime % 3600) // 60
        seconds = runtime % 60
        
        return {
            "is_running": self.is_running,
            "is_paper_mode": self.is_paper_mode,
            "paper_balance": self.paper_balance,
            "paper_bnb_balance_usd": round(self.paper_bnb_balance_usd, 2),
            "bnb_burn_rate": round(self._bnb_burn_rate, 2),
            "bnb_emergency_threshold": round(self._bnb_emergency_threshold, 2),
            "bnb_data_mature": self._bnb_data_mature,
            # May 7 — emit TZ-aware ISO so JS unambiguously interprets as UTC
            "bnb_last_check": (self._last_bnb_check.replace(tzinfo=timezone.utc).isoformat()
                               if self._last_bnb_check else None),
            "runtime_seconds": runtime,
            "runtime_formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "global_volume_ratio": round(_global_volume_ratio, 4),
            "btc_ema20_slope_pct": round(_btc_ema20_slope_pct, 4),
            # BTC Trend Filter state (May 5) — EMA20 vs EMA50 medium-term trend.
            # Header badge uses these to show macro trend + filter state.
            "btc_ema20": round(_current_btc_ema20, 2) if _current_btc_ema20 else None,
            "btc_ema13": round(_current_btc_ema13, 2) if _current_btc_ema13 else None,
            "btc_ema50": round(_current_btc_ema50, 2) if _current_btc_ema50 else None,
            "btc_trend_gap_pct": round(_current_btc_trend_gap_pct, 4) if _current_btc_trend_gap_pct is not None else None,
            "btc_trend_filter_enabled": bool(getattr(config.trading_config.thresholds, 'btc_trend_filter_enabled', False)),
            "market_bull_pct": round(_market_bull_pct, 1),
            "market_bear_pct": round(_market_bear_pct, 1),
            "breadth_n_bull": _breadth_n_bull,
            "breadth_n_bear": _breadth_n_bear,
            "breadth_n_neutral": _breadth_n_neutral,
            "breadth_n_total": _breadth_n_total,
            "filter_block_counts": self._get_filter_block_summary()
        }

    def _record_filter_block(self, filter_name: str, direction: str, had_room: bool = True) -> None:
        """Increment a counter for a pre-entry filter block.

        Called from each filter site in the scan loop right before
        ``signal = "NO_TRADE"``.  In-memory only; resets on bot restart.
        Surfaced via /api/engine/state for the dashboard.

        Args:
            filter_name: Stable identifier matching the log tag, e.g.
                "BTC_TREND_FILTER", "BTC_RSI_ADX_CROSS", "BTC_ADX_GATE".
            direction: "LONG" or "SHORT" (or "ANY" for filters that don't
                differentiate).
            had_room: True if the bot had open-position headroom at filter
                fire time (i.e. could have actually opened a new position).
                False if at max_open_positions (the filter block is "free"
                — no trade was prevented).  Defaults to True for legacy
                callers that haven't been updated.

        May 7 — Regime-aligned gating: skip recording when the trade
        direction is countertrend to the current BTC regime. In a clear
        BEARISH regime, LONG signals are structurally not the desired
        trade — counting their pair-level filter rejections (RSI range,
        EMA20 filter, ADX max, etc.) just adds noise that masks the
        actionable filter pressure on the regime-aligned direction.
        Same logic for SHORT in BULLISH. NEUTRAL regime records both
        (no clear directional preference).
        """
        if not filter_name:
            return
        # Regime-aligned gating: skip countertrend blocks during clear regimes
        try:
            _regime = _current_btc_regime
            if _regime == "BEARISH" and direction == "LONG":
                return
            if _regime == "BULLISH" and direction == "SHORT":
                return
        except NameError:
            pass  # Regime global not yet set (cold start) — fail open, record both
        room_state = "ROOM" if had_room else "FULL"
        key = (filter_name, direction or "ANY", room_state)
        self._filter_block_counts[key] = self._filter_block_counts.get(key, 0) + 1

    def _get_filter_block_summary(self) -> Dict:
        """Return filter block counts grouped per-filter with direction + room split.

        Output shape (sorted by total descending):
            [
                {"filter": "BTC_TREND_FILTER",
                 "long": 3, "short": 12, "any": 0, "total": 15,
                 "long_room": 1, "short_room": 8, "any_room": 0,
                 "long_full": 2, "short_full": 4, "any_full": 0,
                 "total_room": 9, "total_full": 6},
                ...
            ]
        Plus aggregate "totals".  Empty list when no blocks recorded.
        """
        per_filter: Dict[str, Dict[str, int]] = {}
        for k, count in self._filter_block_counts.items():
            # Backward compat: old 2-tuple keys (filter, direction)
            if len(k) == 2:
                filter_name, direction = k
                room_state = "ROOM"  # legacy entries assumed had_room=True
            else:
                filter_name, direction, room_state = k
            dir_key = direction.lower() if direction in ("LONG", "SHORT") else "any"
            row = per_filter.setdefault(filter_name, {
                "long": 0, "short": 0, "any": 0,
                "long_room": 0, "short_room": 0, "any_room": 0,
                "long_full": 0, "short_full": 0, "any_full": 0,
            })
            row[dir_key] += count
            suffix = "_room" if room_state == "ROOM" else "_full"
            row[dir_key + suffix] += count

        rows = []
        total_long = total_short = total_any = 0
        total_room = total_full = 0
        for filter_name, splits in per_filter.items():
            t = splits["long"] + splits["short"] + splits["any"]
            t_room = splits["long_room"] + splits["short_room"] + splits["any_room"]
            t_full = splits["long_full"] + splits["short_full"] + splits["any_full"]
            rows.append({
                "filter": filter_name,
                "long": splits["long"],
                "short": splits["short"],
                "any": splits["any"],
                "total": t,
                "long_room": splits["long_room"],
                "short_room": splits["short_room"],
                "any_room": splits["any_room"],
                "long_full": splits["long_full"],
                "short_full": splits["short_full"],
                "any_full": splits["any_full"],
                "total_room": t_room,
                "total_full": t_full,
            })
            total_long += splits["long"]
            total_short += splits["short"]
            total_any += splits["any"]
            total_room += t_room
            total_full += t_full

        rows.sort(key=lambda r: r["total"], reverse=True)
        return {
            "rows": rows,
            "total_long": total_long,
            "total_short": total_short,
            "total_any": total_any,
            "total": total_long + total_short + total_any,
            "total_room": total_room,
            "total_full": total_full,
        }

    async def _get_exit_btc_trend_gap(self) -> float:
        """Capture BTC EMA13-EMA50 gap at close time (May 6).

        Returns the BTC gap pulled from the global updated each scan.
        May 7: pair-side removed — observation-only column dropped per
        CLAUDE.md cleanup (BTCTrend(exit) is the analog that matters).
        """
        global _current_btc_trend_gap_pct
        return _current_btc_trend_gap_pct

    async def _recalculate_paper_balance(self, db: AsyncSession) -> float:
        """Recalculate paper_balance from DB as source of truth.
        
        Formula: initial + closed_pnl + closed_fees - open_margin - bnb_swaps
        
        Fees are paid from BNB, not USDT. Since Order.pnl is net of fees,
        we add back closed fees to avoid double-counting (fees already tracked
        in the BNB balance via _deduct_fee_from_bnb).
        """
        initial = config.trading_config.paper_balance
        closed_pnl_result = await db.execute(
            select(func.coalesce(func.sum(Order.pnl), 0)).where(
                and_(Order.status == "CLOSED", Order.is_paper == True)
            )
        )
        total_closed_pnl = closed_pnl_result.scalar() or 0
        closed_fees_result = await db.execute(
            select(func.coalesce(func.sum(Order.total_fee), 0)).where(
                and_(Order.status == "CLOSED", Order.is_paper == True)
            )
        )
        total_closed_fees = closed_fees_result.scalar() or 0
        open_margin_result = await db.execute(
            select(func.coalesce(func.sum(Order.investment), 0)).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        total_open_margin = open_margin_result.scalar() or 0
        bnb_swap_result = await db.execute(
            select(func.coalesce(func.sum(BnbSwapLog.amount_usdt), 0)).where(
                BnbSwapLog.is_paper == True
            )
        )
        total_bnb_swaps = bnb_swap_result.scalar() or 0
        correct_balance = initial + total_closed_pnl + total_closed_fees - total_open_margin - total_bnb_swaps
        if abs(correct_balance - self.paper_balance) > 0.01:
            logger.warning(
                f"[BALANCE_SYNC] Correcting drift: "
                f"in_memory={self.paper_balance:.2f}, db_correct={correct_balance:.2f}, "
                f"diff={self.paper_balance - correct_balance:.2f}"
            )
        self.paper_balance = correct_balance
        return correct_balance

    async def _recalculate_paper_bnb(self, db: AsyncSession) -> float:
        """Recalculate paper BNB balance from DB as source of truth.
        
        Formula: initial_bnb + sum(swap inflows) - sum(all fees from paper orders)
        """
        initial = config.trading_config.paper_bnb_initial_usd
        swap_result = await db.execute(
            select(func.coalesce(func.sum(BnbSwapLog.amount_usdt), 0)).where(
                BnbSwapLog.is_paper == True
            )
        )
        total_swaps = swap_result.scalar() or 0
        fee_result = await db.execute(
            select(func.coalesce(func.sum(Order.total_fee), 0)).where(
                and_(Order.status == "CLOSED", Order.is_paper == True)
            )
        )
        total_closed_fees = fee_result.scalar() or 0
        open_entry_fee_result = await db.execute(
            select(func.coalesce(func.sum(Order.entry_fee), 0)).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        total_open_entry_fees = open_entry_fee_result.scalar() or 0
        correct = initial + total_swaps - total_closed_fees - total_open_entry_fees
        self.paper_bnb_balance_usd = correct
        return correct

    async def _deduct_fee_from_bnb(self, fee_usd: float, db: AsyncSession):
        """Check BNB reserve after a fee is paid; trigger emergency swap if low.

        Paper mode: decrements in-memory paper BNB balance and checks threshold.
        Live mode: queries actual Binance BNB balance and checks threshold.
        Without this live-mode path, emergency swaps only happened every 6h
        via bnb_scheduled_check, causing silent fee increases when BNB ran out
        between checks.
        """
        tc = config.trading_config
        if fee_usd <= 0 or not tc.bnb_swap_enabled:
            return

        if self.is_paper_mode:
            self.paper_bnb_balance_usd -= fee_usd
            if self.paper_bnb_balance_usd < 0:
                self.paper_bnb_balance_usd = 0
            current_bnb = self.paper_bnb_balance_usd
        else:
            # Live: query actual wallet BNB balance
            try:
                balance = await binance_service.get_balance()
                bnb_price = await binance_service.get_bnb_price()
                if bnb_price <= 0 or not balance:
                    return
                current_bnb = balance.get('bnb_total', 0) * bnb_price
            except Exception as e:
                logger.warning(f"[BNB_EMERGENCY] Failed to query live BNB balance: {e}")
                return

        # Fallback emergency threshold for cold-start (before first scheduled check).
        # Uses 10% of initial BNB target as a conservative floor.
        emergency_threshold = self._bnb_emergency_threshold
        if emergency_threshold <= 0:
            emergency_threshold = tc.paper_bnb_initial_usd * 0.1

        if current_bnb < emergency_threshold:
            # May 25 v2 — burn-rate-derived emergency threshold can fire from
            # an extrapolation off <2h of data. Gate auto-swap on data maturity.
            # Fall-back fixed threshold (10% of initial) is still allowed to
            # fire — that's a genuine "BNB nearly empty" signal, not extrapolation.
            using_extrapolated = self._bnb_emergency_threshold > 0 and emergency_threshold == self._bnb_emergency_threshold
            if using_extrapolated and not self._bnb_data_mature:
                logger.info(
                    f"[BNB_EMERGENCY] {'Paper' if self.is_paper_mode else 'Live'} BNB ${current_bnb:.2f} "
                    f"< extrapolated threshold ${emergency_threshold:.2f}, but data window <2h — "
                    f"suppressing emergency swap. Real BNB floor (10% of initial = ${tc.paper_bnb_initial_usd * 0.1:.2f}) not breached."
                )
                return
            logger.warning(
                f"[BNB_EMERGENCY] {'Paper' if self.is_paper_mode else 'Live'} BNB ${current_bnb:.2f} "
                f"< emergency threshold ${emergency_threshold:.2f} — triggering swap"
            )
            await self._execute_bnb_swap(db, swap_type="emergency")

    async def _execute_bnb_swap(self, db: AsyncSession, swap_type: str = "scheduled"):
        """Execute a USDT→BNB swap (paper or live)."""
        tc = config.trading_config
        if not tc.bnb_swap_enabled:
            return
        
        target = self._bnb_projected_need if self._bnb_projected_need > 0 else tc.paper_bnb_initial_usd * 0.4
        
        if self.is_paper_mode:
            current_bnb = self.paper_bnb_balance_usd
            if current_bnb >= target:
                return
            shortfall = target - current_bnb
            available_usdt = await self._recalculate_paper_balance(db)
            min_investment = tc.investment.min_investment_size
            if available_usdt - shortfall < min_investment:
                shortfall = max(0, available_usdt - min_investment)
            if shortfall <= 0:
                logger.warning(f"[BNB_SWAP] Cannot swap: insufficient USDT (available={available_usdt:.2f})")
                return
            # May 7: mirror live mode's $5 min-shortfall guard. Avoids tiny
            # rebalance swaps (e.g., $3.94) on rapid redeploys when BNB is
            # already close to target.
            if shortfall <= 5:
                logger.info(f"[BNB_SWAP] Skipped: shortfall ${shortfall:.2f} below $5 min threshold")
                return
            
            bnb_price = await binance_service.get_bnb_price()
            if bnb_price <= 0:
                bnb_price = 600.0  # fallback for paper mode
            
            pre_usdt = self.paper_balance
            pre_bnb = self.paper_bnb_balance_usd
            self.paper_bnb_balance_usd += shortfall
            
            swap_log = BnbSwapLog(
                swap_type=swap_type,
                amount_usdt=shortfall,
                bnb_price=bnb_price,
                amount_bnb=round(shortfall / bnb_price, 6),
                pre_bnb_usd=pre_bnb,
                post_bnb_usd=self.paper_bnb_balance_usd,
                pre_usdt=pre_usdt,
                post_usdt=pre_usdt - shortfall,
                burn_rate=self._bnb_burn_rate,
                is_paper=True
            )
            db.add(swap_log)
            await db.commit()
            
            await self._recalculate_paper_balance(db)
            await self.save_state(db)
            logger.info(
                f"[BNB_SWAP] Paper {swap_type}: swapped ${shortfall:.2f} USDT → "
                f"{shortfall/bnb_price:.4f} BNB @ ${bnb_price:.2f}. "
                f"BNB: ${pre_bnb:.2f} → ${self.paper_bnb_balance_usd:.2f}"
            )
        else:
            balance = await binance_service.get_balance()
            bnb_price = await binance_service.get_bnb_price()
            if bnb_price <= 0:
                return
            current_bnb_usd = balance['bnb_total'] * bnb_price
            if current_bnb_usd >= target:
                return
            shortfall = target - current_bnb_usd
            available_usdt = balance['usdt_free']
            min_investment = tc.investment.min_investment_size
            if available_usdt - shortfall < min_investment:
                shortfall = max(0, available_usdt - min_investment)
            if shortfall <= 5:
                return
            
            result = await binance_service.buy_bnb(shortfall)
            if not result:
                return
            
            new_balance = await binance_service.get_balance()
            swap_log = BnbSwapLog(
                swap_type=swap_type,
                amount_usdt=result['cost_usdt'],
                bnb_price=result['price'],
                amount_bnb=result['bnb_amount'],
                pre_bnb_usd=current_bnb_usd,
                post_bnb_usd=new_balance['bnb_total'] * result['price'],
                pre_usdt=available_usdt,
                post_usdt=new_balance['usdt_free'],
                burn_rate=self._bnb_burn_rate,
                is_paper=False
            )
            db.add(swap_log)
            await db.commit()
            logger.info(
                f"[BNB_SWAP] Live {swap_type}: bought {result['bnb_amount']:.4f} BNB "
                f"for ${result['cost_usdt']:.2f} @ ${result['price']:.2f}"
            )

    async def _recompute_bnb_burn_rate(self, db: AsyncSession) -> float:
        """Recompute self._bnb_burn_rate, _bnb_projected_need, _bnb_emergency_threshold
        from CLOSED orders in DB. Returns fees_24h.

        May 11: extracted from bnb_scheduled_check so the burn-rate metric can be
        refreshed every scan cycle WITHOUT firing the gated swap action. Cheap
        (a few SQL aggregates); the runway display in the UI depends on this
        being current. Previously the metric stayed at 0 for up to 6h after a
        bot restart because the swap-gate also blocked the recompute.
        """
        tc = config.trading_config
        now = datetime.utcnow()
        cutoff_24h = now - timedelta(hours=24)
        cutoff_12h = now - timedelta(hours=12)

        result_24h = await db.execute(
            select(
                func.coalesce(func.sum(Order.total_fee), 0),
                func.count(Order.id)
            ).where(
                and_(Order.status == "CLOSED", Order.closed_at >= cutoff_24h)
            )
        )
        row_24h = result_24h.one()
        fees_24h = float(row_24h[0] or 0)
        count_24h = int(row_24h[1] or 0)

        result_12h = await db.execute(
            select(
                func.coalesce(func.sum(Order.total_fee), 0),
                func.count(Order.id),
                func.min(Order.closed_at),
            ).where(
                and_(Order.status == "CLOSED", Order.closed_at >= cutoff_12h)
            )
        )
        row_12h = result_12h.one()
        fees_12h = float(row_12h[0] or 0)
        count_12h = int(row_12h[1] or 0)
        oldest_12h = row_12h[2]

        # Oldest closed trade inside the 24h window — used to measure the TRUE
        # time span of the fee data, not bot runtime. The DB persists across
        # bot restarts, so dividing fees_24h by runtime grossly overestimates
        # burn rate immediately after a restart.
        result_oldest_24h = await db.execute(
            select(func.min(Order.closed_at)).where(
                and_(Order.status == "CLOSED", Order.closed_at >= cutoff_24h)
            )
        )
        oldest_24h = result_oldest_24h.scalar()

        # May 25 BUGFIX (v4): denominator = CUMULATIVE bot runtime (across
        # restarts), not per-session started_at. v3 used `started_at` which
        # resets on every deploy/restart, so after 3 deploys it was minutes
        # ago — `min(started_at, oldest_close)` still collapsed to oldest_close
        # (~1.23h), reproducing the v2 bug.
        #
        # Correct semantic: "for how long has the bot been running (and
        # eligible to accumulate fees) within the 24h window?" That's
        # cumulative runtime (`total_runtime_seconds + current_session`),
        # accessed via `get_runtime_seconds()`. It persists across restarts.
        #
        # Cap at 24h (the window size). If bot has been alive cumulatively
        # >24h → full 24h denominator. Otherwise → cumulative uptime.
        MIN_DATA_MATURE_HOURS = 2.0
        runtime_h = self.get_runtime_seconds() / 3600.0
        if count_24h > 0 and oldest_24h:
            # Denominator: cumulative runtime, capped at the 24h window.
            span_24h_hours = min(24.0, runtime_h) if runtime_h > 0 else 0
            # Safety: never let span be less than the actual trade span
            # (trades older than runtime imply pre-tracked sessions or
            # clock drift; use whichever gives a larger denominator).
            trade_span_h = (now - oldest_24h).total_seconds() / 3600.0
            if trade_span_h > span_24h_hours:
                span_24h_hours = min(24.0, trade_span_h)
            self._bnb_burn_rate = fees_24h / span_24h_hours if span_24h_hours > 0 else 0
            self._bnb_projected_need = self._bnb_burn_rate * tc.bnb_runway_hours
            self._bnb_data_mature = span_24h_hours >= MIN_DATA_MATURE_HOURS
        else:
            span_24h_hours = 0
            self._bnb_burn_rate = 0
            self._bnb_projected_need = 0
            self._bnb_data_mature = False

        # 12h emergency threshold — same logic, capped at 12h
        if count_12h > 0 and oldest_12h:
            span_12h_hours = min(12.0, runtime_h) if runtime_h > 0 else 0
            trade_span_12h = (now - oldest_12h).total_seconds() / 3600.0
            if trade_span_12h > span_12h_hours:
                span_12h_hours = min(12.0, trade_span_12h)
            burn_rate_12h = fees_12h / span_12h_hours if span_12h_hours > 0 else 0
        else:
            burn_rate_12h = 0
        self._bnb_emergency_threshold = burn_rate_12h * 12.0

        return fees_24h

    async def bnb_scheduled_check(self, db: AsyncSession, force: bool = False):
        """Scheduled BNB balance check: compute burn rate, project needs, swap if necessary.

        May 7: respects bnb_check_interval_hours across restarts. Without this
        gate, every redeploy triggered a fresh check ~60s after startup, causing
        repeated tiny rebalance swaps when the operator deployed multiple times
        in a short window. Pass force=True to override (e.g., manual UI trigger).

        May 11: burn-rate recompute is now decoupled from the swap-gate. The
        metric is refreshed every call (cheap SQL aggregates), but the swap
        action remains gated. Previously the runway display stayed empty for
        up to 6h after a bot restart because the gate blocked the recompute.
        """
        tc = config.trading_config
        if not tc.bnb_swap_enabled:
            return

        # Always recompute the burn-rate metric (cheap, drives UI runway display).
        # Only the swap action is gated below.
        fees_24h = await self._recompute_bnb_burn_rate(db)

        # Interval gate (skip swap ACTION if last check was within bnb_check_interval_hours).
        if not force and self._last_bnb_check is not None:
            interval_hours = max(1, int(tc.bnb_check_interval_hours or 6))
            elapsed = (datetime.utcnow() - self._last_bnb_check).total_seconds()
            if elapsed < interval_hours * 3600:
                logger.info(
                    f"[BNB_CHECK] Swap action skipped: last check {elapsed/3600:.2f}h ago "
                    f"(interval={interval_hours}h). Next swap eligible in "
                    f"{(interval_hours * 3600 - elapsed)/3600:.2f}h. "
                    f"Burn rate refreshed: ${self._bnb_burn_rate:.2f}/hr."
                )
                return

        self._last_bnb_check = datetime.utcnow()
        # Persist last-check timestamp immediately so restarts within the
        # interval window correctly skip until the interval elapses.
        try:
            await self.save_state(db)
        except Exception as _e:
            logger.debug(f"[BNB_CHECK] Failed to persist last_bnb_check: {_e}")

        # Safety rail: burn_rate (in $/hr) can never exceed total fees when the
        # window is >= 1h. If this ever trips, the span calculation is broken
        # and we refuse to swap rather than over-spend.
        if self._bnb_burn_rate > fees_24h and fees_24h > 0:
            logger.error(
                f"[BNB_CHECK] Burn rate sanity check failed: "
                f"${self._bnb_burn_rate:.2f}/hr > ${fees_24h:.2f} total 24h fees. "
                f"Refusing to swap."
            )
            return
        
        if self._bnb_projected_need <= 0:
            logger.info("[BNB_CHECK] No fee history yet, skipping swap check")
            return

        # May 25 v2 — projected_need is derived from burn rate. If data window
        # is <2h (data not mature), the projected need can be wildly inflated
        # from a narrow burst of trades. Display the rate but don't act on it.
        if not self._bnb_data_mature:
            logger.info(
                f"[BNB_CHECK] Data window <2h — burn rate ${self._bnb_burn_rate:.2f}/hr "
                f"published for display but scheduled swap suppressed (need ≥2h of history)."
            )
            return

        if self.is_paper_mode:
            await self._recalculate_paper_bnb(db)
            current_bnb = self.paper_bnb_balance_usd
        else:
            balance = await binance_service.get_balance()
            bnb_price = await binance_service.get_bnb_price()
            current_bnb = balance['bnb_total'] * bnb_price if bnb_price > 0 else 0
        
        logger.info(
            f"[BNB_CHECK] Burn rate: ${self._bnb_burn_rate:.2f}/hr | "
            f"Projected need ({tc.bnb_runway_hours}h): ${self._bnb_projected_need:.2f} | "
            f"Emergency threshold (12h fees): ${self._bnb_emergency_threshold:.2f} | "
            f"Current BNB: ${current_bnb:.2f}"
        )
        
        if current_bnb < self._bnb_projected_need:
            await self._execute_bnb_swap(db, swap_type="scheduled")

    async def get_available_balance(self, db: AsyncSession) -> float:
        """Get available balance for trading.
        
        For paper trading: always recalculate from DB to prevent drift.
        """
        if self.is_paper_mode:
            return await self._recalculate_paper_balance(db)
        else:
            balance = await binance_service.get_balance()
            return balance['usdt_free']
    
    def calculate_position_size(
        self, available_balance: float, confidence: str, total_portfolio: float = None,
        cell_multiplier: float = 1.0, cell_lev_multiplier: float = 1.0,
        multiplier_target: str = "investment",
    ) -> Tuple[float, float, bool]:
        """
        Calculate position size and leverage based on config.

        Premium Multiplier (May 4, 2026 — per CLAUDE.md May 3 design; extended May 21):
        - cell_multiplier (1.0 = no boost) is the INVESTMENT-side multiplier.
        - cell_lev_multiplier (1.0 = no boost) is the LEVERAGE-side multiplier (May 21).
        - Each is applied AFTER confidence-level multiplier and BEFORE the tradeable cap.
        - When investment cap kicks in, the trade still proceeds at the available amount —
          capital cap is the natural ceiling (no abort).
        - multiplier_target =
            "investment" → only cell_multiplier applies (cell_lev_multiplier treated as 1.0)
            "leverage"   → only cell_lev_multiplier applies (cell_multiplier treated as 1.0)
            "both"       → BOTH apply (compounding — effective notional ≈ inv_mult × lev_mult × base)

        Returns:
            Tuple of (investment_amount, leverage, capped_by_balance)
            where capped_by_balance=True if the cell multiplier wanted more
            than tradeable allowed (logged via [CELL_MULT_CAPPED] in caller).
        """
        tc = config.trading_config
        conf_level = tc.confidence_levels.get(confidence)

        if not conf_level or not conf_level.enabled:
            return 0, 0, False

        # Calculate safe reserve
        if tc.investment.reserve_mode == "percentage":
            reserve = available_balance * (tc.investment.reserve_percentage / 100)
        else:
            reserve = tc.investment.reserve_fixed

        # Available after reserve
        tradeable = max(0, available_balance - reserve)

        # Calculate base investment
        if tc.investment.mode == "percentage":
            investment = tradeable * (tc.investment.percentage / 100)
        elif tc.investment.mode == "equal_split":
            max_pos = tc.investment.max_open_positions or 5
            base = total_portfolio if total_portfolio else available_balance
            if tc.investment.reserve_mode == "percentage":
                reserve_from_total = base * (tc.investment.reserve_percentage / 100)
            else:
                reserve_from_total = tc.investment.reserve_fixed
            investment = max(0, base - reserve_from_total) / max_pos
        else:
            investment = min(tc.investment.fixed_amount, tradeable)

        # Apply investment multiplier for higher confidence levels
        conf_multiplier = getattr(conf_level, 'investment_multiplier', 1.0)
        investment = investment * conf_multiplier

        # === Premium Multiplier: investment-side path (active in "investment" or "both") ===
        # Track desired-vs-actual to surface capital-cap fallback to the caller.
        capped_by_balance = False
        apply_inv = multiplier_target in ("investment", "both")
        apply_lev = multiplier_target in ("leverage", "both")
        if apply_inv and cell_multiplier and cell_multiplier != 1.0:
            target_investment = investment * cell_multiplier
            if target_investment > tradeable + 0.01:
                capped_by_balance = True
            investment = target_investment

        # Ensure investment doesn't exceed tradeable balance.  When the cell
        # multiplier wanted more than tradeable (capped_by_balance flag set
        # above), this min() is what executes the fallback: invest all available.
        investment = min(investment, tradeable)

        # Clamp investment to min/max size limits
        investment = max(investment, tc.investment.min_investment_size)
        investment = min(investment, tc.investment.max_investment_size)

        # If clamped min exceeds available tradeable balance, skip the trade
        if investment > tradeable:
            logger.warning(f"Min investment size ({tc.investment.min_investment_size}) exceeds tradeable balance ({tradeable:.2f}), skipping")
            return 0, 0, False

        # Get leverage from config
        leverage = conf_level.leverage

        # === Premium Multiplier: leverage-side path (active in "leverage" or "both") ===
        if apply_lev and cell_lev_multiplier and cell_lev_multiplier != 1.0:
            leverage = max(1, int(round(leverage * cell_lev_multiplier)))

        return investment, leverage, capped_by_balance

    def _lookup_rsi_adx_multiplier(
        self, rsi_val: Optional[float], adx_val: Optional[float],
        rule_string: str, source_prefix: str,
    ) -> Tuple[float, float, Optional[str]]:
        """
        Premium Multiplier (May 4, 2026 → extended May 21) — parse RSI×ADX multiplier rule
        and return (invest_multiplier, leverage_multiplier, source_label).

        Rule string format (May 21+ extended, 4-part):
          "<RSI_min>-<RSI_max>:<ADX_min>-<ADX_max>:<invest_mult>:<lev_mult>,..."
        Backward-compat (May 4 → May 20, 3-part):
          "<RSI_min>-<RSI_max>:<ADX_min>-<ADX_max>:<invest_mult>,..."
          → leverage_multiplier defaults to 1.0 (lev side inert under old configs)

        Both ranges are half-open [min, max).
        Returns (1.0, 1.0, None) if no rule matches or inputs are missing.
        Malformed rules are silently skipped (logged at WARNING level).

        source_prefix is "PAIR" or "BTC" — embedded in the returned source_label
        so the tracking table can attribute which rule fired (e.g., "PAIR_55-60_22-25").
        """
        if rsi_val is None or adx_val is None or not rule_string:
            return 1.0, 1.0, None
        for rule in rule_string.split(','):
            rule = rule.strip()
            if not rule:
                continue
            try:
                parts = rule.split(':')
                if len(parts) not in (3, 4):
                    logger.warning(f"[CELL_MULT] Malformed rule '{rule}' (expected 3 or 4 parts), skipping")
                    continue
                rsi_part = parts[0]
                adx_part = parts[1]
                inv_mult = float(parts[2])
                lev_mult = float(parts[3]) if len(parts) == 4 else 1.0
                rsi_min, rsi_max = map(float, rsi_part.split('-'))
                adx_min, adx_max = map(float, adx_part.split('-'))
                if rsi_min <= rsi_val < rsi_max and adx_min <= adx_val < adx_max:
                    label = f"{source_prefix}_{rsi_part}_{adx_part}"
                    return inv_mult, lev_mult, label
            except (ValueError, TypeError) as e:
                logger.warning(f"[CELL_MULT] Failed to parse rule '{rule}': {e}, skipping")
                continue
        return 1.0, 1.0, None

    # _lookup_stretch_multiplier removed May 15 PM — stretch-based multiplier
    # source retired (no longer surfaced in UI / no rule strings active in JSON).
    # Historical trades with cell_multiplier_source starting "STRETCH_..." retain
    # their attribution in the Multiplier Cell Performance table.

    def _lookup_extension_multiplier(
        self,
        direction: str,
        ext_pct: Optional[float],
        pair_vol_ratio: Optional[float],
        adx_delta: Optional[float],
    ) -> Tuple[float, float, Optional[str]]:
        """Extension Multiplier (May 24, 2026) — Pair Distance from EMA13 multiplier dimension.

        Walks `extension_multiplier_rules` config and returns
        (invest_multiplier, leverage_multiplier, source_label) for the matching cell.

        Rule structure (see config.py):
          {name, direction, ext_min, ext_max, pair_vol_max?, adx_delta_max?, inv_mult, lev_mult}

        Matching logic:
          - direction must match
          - ext_pct must be in [ext_min, ext_max)
          - if pair_vol_max present, pair_vol_ratio must be < pair_vol_max
          - if adx_delta_max present, adx_delta must be < adx_delta_max

        Conflict resolution: HIGHER inv_mult wins across multiple matching rules
        (when several rules match the same trade). Source labels for combined matches
        are joined as "EXT_{name1}+{name2}" — but the active inv/lev pair returned
        is the single highest-inv-mult rule.

        Returns (1.0, 1.0, None) on no match or missing required inputs.
        """
        try:
            rules = getattr(config.trading_config.thresholds, 'extension_multiplier_rules', []) or []
        except Exception:
            return 1.0, 1.0, None
        if not rules or ext_pct is None:
            return 1.0, 1.0, None

        matches = []
        for r in rules:
            try:
                if r.get('direction') != direction:
                    continue
                ext_min = float(r.get('ext_min', -999))
                ext_max = float(r.get('ext_max', 999))
                if not (ext_min <= ext_pct < ext_max):
                    continue
                pv_max = r.get('pair_vol_max')
                if pv_max is not None:
                    if pair_vol_ratio is None or pair_vol_ratio >= float(pv_max):
                        continue
                ad_max = r.get('adx_delta_max')
                if ad_max is not None:
                    if adx_delta is None or adx_delta >= float(ad_max):
                        continue
                matches.append(r)
            except (ValueError, TypeError) as e:
                logger.warning(f"[EXT_MULT] Failed to parse rule {r}: {e}, skipping")
                continue

        if not matches:
            return 1.0, 1.0, None

        # HIGHER inv_mult wins for the active inv/lev pair; combined names in label.
        best = max(matches, key=lambda r: float(r.get('inv_mult', 1.0)))
        inv = float(best.get('inv_mult', 1.0))
        lev = float(best.get('lev_mult', 1.0))
        names = '+'.join(r.get('name', '?') for r in matches)
        label = f"EXT_{names}"
        return inv, lev, label

    def _lookup_btc_1h_slope_btc_adx_multiplier(
        self,
        direction: str,
        btc_1h_slope: Optional[float],
        btc_adx: Optional[float],
    ) -> Tuple[float, float, Optional[str]]:
        """BTC 1h Slope × BTC ADX Multiplier (May 24 evening, 2026) — NEW dimension.

        Walks `btc_1h_slope_btc_adx_multiplier_rules` config and returns
        (invest_multiplier, leverage_multiplier, source_label).

        Rule struct (JSON-list, see config.py):
          {name, direction, slope_min, slope_max, adx_min, adx_max, inv_mult, lev_mult}

        Matching: direction must match, btc_1h_slope in [slope_min, slope_max),
        btc_adx in [adx_min, adx_max). HIGHER inv_mult wins on multi-match.
        Source label: "BTC1H_{name}" (e.g., "BTC1H_M3" for LONG, "BTC1H_M2" for SHORT).

        Returns (1.0, 1.0, None) on no match or missing inputs.
        """
        try:
            rules = getattr(config.trading_config.thresholds,
                            'btc_1h_slope_btc_adx_multiplier_rules', []) or []
        except Exception:
            return 1.0, 1.0, None
        if not rules or btc_1h_slope is None or btc_adx is None:
            return 1.0, 1.0, None

        matches = []
        for r in rules:
            try:
                if r.get('direction') != direction:
                    continue
                slope_min = float(r.get('slope_min', -999))
                slope_max = float(r.get('slope_max', 999))
                adx_min = float(r.get('adx_min', -1))
                adx_max = float(r.get('adx_max', 999))
                if not (slope_min <= btc_1h_slope < slope_max):
                    continue
                if not (adx_min <= btc_adx < adx_max):
                    continue
                matches.append(r)
            except (ValueError, TypeError) as e:
                logger.warning(f"[BTC1H_MULT] Failed to parse rule {r}: {e}, skipping")
                continue

        if not matches:
            return 1.0, 1.0, None

        best = max(matches, key=lambda r: float(r.get('inv_mult', 1.0)))
        inv = float(best.get('inv_mult', 1.0))
        lev = float(best.get('lev_mult', 1.0))
        names = '+'.join(r.get('name', '?') for r in matches)
        label = f"BTC1H_{names}"
        return inv, lev, label

    def _lookup_pattern_cell_rule(
        self, direction: str, c_flags: dict, w_flags: dict,
    ) -> Tuple[float, float, Optional[str], Optional[float], Optional[float], bool]:
        """Pattern Cell Ship Rules — May 21, NEW dimension per CLAUDE.md May 21 ship plan.

        Walks pattern_cell_rules config, collects rules matching this trade's
        direction + matched C/W patterns, applies Option D conflict resolution
        (May 23 strict-C-blocks-W refinement):
          - If ANY C-pattern matches AND a C rule fires → apply C rule
          - If ANY C-pattern matches but NO C rule fires → return BASELINE
            (1.0, 1.0, None, None, None). DO NOT fall through to W.
            Rationale: a C-signature match means "loser-shape" — defang any
            co-matched W multipliers. Operator can explicitly opt-in to
            multiplier on a C cell by configuring its rule (e.g., C1 SHORT
            at 2.0× — see CLAUDE.md May 21 treatment-decoupling).
          - Else if ANY W-pattern matches → W-side rules apply
          - Else if no C and no W → UNMATCHED rules apply
          - Else no rule fires → returns (1.0, 1.0, None, None, None)

        Within the active side, ANY rule can carry ANY treatment (May 21 late ship —
        de-coupled rule pattern code from treatment type). Example: a C-rule can carry
        an inv_mult > 1.0 (e.g., C1 SHORT @ 2.0× because cross-batch shows 78% WR);
        a W-rule can carry fixed_tp_pct + fixed_sl_pct (e.g., W1 LONG with caps because
        cross-batch shows 20% WR). Pattern code is the SIGNATURE; treatment is in fields.

        For multiple matching rules within the active side:
          - inv_mult / lev_mult: HIGHER-wins (max — not multiplied)
          - fixed_tp_pct / fixed_sl_pct: most aggressive (lowest TP, tightest SL)

        Returns (inv_mult, lev_mult, source_label, fixed_tp_pct, fixed_sl_pct, block)
        where source_label is comma-joined matched patterns (e.g., "C4+C8" or "W1+W2")
        and block=True if any matching rule carries block:true (entry should be skipped).
        Jun 8: pattern may be a single code, "UNMATCHED", or an AND-combo ("C1+C6").
        Jun 10: a part may carry a '!' prefix to negate it ("W6+!W1" = W6 AND NOT W1).
        """
        rules = getattr(config.trading_config.thresholds, 'pattern_cell_rules', []) or []
        if not rules:
            return 1.0, 1.0, None, None, None

        # Determine candidate sides in priority order (May 23 Option D — strict
        # C-blocks-W with explicit-opt-in for C multipliers).
        #
        # Evolution of this logic:
        #   May 21 first ship: strict C-blocks-W. Broke FILUSDT (C1+W2, no C1 rule).
        #   May 21 bug fix:    fall through to W when C has no rule. Broke MTLUSDT
        #                      (id=28, May 23): C2 matched + W1+W6 mults applied at
        #                      2.0× → loss doubled to -$91.37.
        #   May 23 Option D:   restore strict C-blocks-W with surgery — if C matches
        #                      but no C rule fires, return BASELINE (don't fall to W).
        #                      Operator opts into C multipliers explicitly via rule
        #                      config (e.g., C1 SHORT at 2.0× already shipped).
        #
        # Why this is structurally correct:
        # A C-signature firing means "trade has loser-shape signature." If no rule
        # is explicitly configured for that C cell, the conservative default is
        # baseline sizing (1.0×) and default exit chain — NOT amplification via
        # co-matched W rules. This preserves the May 21 treatment-decoupling lesson
        # (pattern code is the signature, treatment is in rule fields) while making
        # the default safe.
        matched_c = [k for k, v in c_flags.items() if v is True]
        matched_w = [k for k, v in w_flags.items() if v is True]
        sides_to_try: List[Tuple[str, set]] = []
        if matched_c:
            sides_to_try.append(('C', set(matched_c)))
        if matched_w:
            sides_to_try.append(('W', set(matched_w)))
        sides_to_try.append(('UNMATCHED', {'UNMATCHED'}))

        # Jun 8: generalized signature matching — single code, UNMATCHED, or combo (AND).
        _mc = set(matched_c)
        _mw = set(matched_w)

        def _rule_side_and_match(p):
            """Map a rule pattern to (side, matched_bool). 'UNMATCHED' = no C and no W.
            Combo 'C1+C6' = AND of all component codes. A '!' prefix negates a part
            (Jun 10: 'W6+!W1' = W6 matched AND W1 NOT matched — lets a rule target
            e.g. macro-tag-only shorts without pair-momentum confirmation). A mixed
            C+W combo resolves to the 'C' side (C-blocks-W priority); side comes
            from the positive parts. Single code = combo of one part."""
            if not p:
                return None, False
            if p == 'UNMATCHED':
                return 'UNMATCHED', (not _mc and not _mw)
            parts = [x.strip() for x in str(p).split('+') if x.strip()]
            if not parts:
                return None, False
            pos = [x for x in parts if not x.startswith('!')]
            neg = [x[1:].strip() for x in parts if x.startswith('!')]
            if not pos:
                # All-negated pattern has no anchor cohort — refuse rather than
                # silently matching everything outside the negated codes.
                return None, False
            side = 'C' if any(x.startswith('C') for x in pos) else 'W'
            for x in pos:
                if x.startswith('C') and x not in _mc:
                    return side, False
                if x.startswith('W') and x not in _mw:
                    return side, False
            for x in neg:
                if x.startswith('C') and x in _mc:
                    return side, False
                if x.startswith('W') and x in _mw:
                    return side, False
            return side, True

        def _walk_side(active_side: str, matched_patterns: set):
            """Walk rules for one active side. Returns (applied_inv, applied_lev,
            applied_sources, applied_tp, applied_sl, applied_block)."""
            applied_sources = []
            applied_inv = 1.0
            applied_lev = 1.0
            applied_tp = None
            applied_sl = None
            applied_block = False
            for rule in rules:
                try:
                    if rule.get('direction') != direction:
                        continue
                    p = rule.get('pattern')
                    side, is_match = _rule_side_and_match(p)
                    if not is_match or side != active_side:
                        continue
                    applied_sources.append(p)
                    if bool(rule.get('block', False)):
                        applied_block = True
                    r_inv = float(rule.get('inv_mult', 1.0) or 1.0)
                    r_lev = float(rule.get('lev_mult', 1.0) or 1.0)
                    if r_inv > applied_inv:
                        applied_inv = r_inv
                    if r_lev > applied_lev:
                        applied_lev = r_lev
                    r_tp = rule.get('fixed_tp_pct')
                    r_sl = rule.get('fixed_sl_pct')
                    if r_tp is not None:
                        r_tp = float(r_tp)
                        if applied_tp is None or r_tp < applied_tp:
                            applied_tp = r_tp
                    if r_sl is not None:
                        r_sl = float(r_sl)
                        if applied_sl is None or r_sl > applied_sl:
                            applied_sl = r_sl
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"[PATTERN_CELL] Malformed rule {rule}: {e}, skipping")
                    continue
            return applied_inv, applied_lev, applied_sources, applied_tp, applied_sl, applied_block

        for active_side, matched_patterns in sides_to_try:
            applied_inv, applied_lev, applied_sources, applied_tp, applied_sl, applied_block = _walk_side(
                active_side, matched_patterns
            )
            if applied_sources:
                source_label = '+'.join(sorted(applied_sources))
                return applied_inv, applied_lev, source_label, applied_tp, applied_sl, applied_block
            # May 23 Option D: strict C-blocks-W. If C matched but no C rule fired,
            # return baseline immediately — DON'T fall through to W (which would
            # apply co-matched W multipliers and amplify a loser-shape trade).
            # Operator opts into C multiplier by explicitly configuring a C rule.
            if active_side == 'C':
                return 1.0, 1.0, None, None, None, False

        return 1.0, 1.0, None, None, None, False

    async def _revalidate_entry_signal(
        self, symbol: str, pair: str, original_direction: str, original_confidence: str
    ) -> Tuple[bool, str]:
        """Re-evaluate whether the original entry signal is still valid after maker timeout.

        Amendment #7 (Apr 18): prevents the taker fallback from entering on stale signals
        that have expired during the maker wait window. Re-fetches fresh indicators
        and re-runs the core signal check + key BTC-level filters.

        Returns (is_valid, reason):
          - is_valid=True: signal still valid, proceed to taker fallback
          - is_valid=False: signal expired, abort entry. reason describes why.

        FAILS OPEN: if re-fetch fails, defer to taker (don't block on infra errors).
        """
        try:
            ohlcv = await binance_service.get_ohlcv(symbol, '5m', 100)
            if not ohlcv:
                return True, 'fetch_failed_defer'

            tc = config.trading_config
            pair_vol_bars = getattr(tc.thresholds, 'pair_volume_lookback_bars', 20)
            global_vol_bars = getattr(tc.thresholds, 'global_volume_lookback_bars', 48)
            indicators = calculate_indicators(
                ohlcv, pair_volume_bars=pair_vol_bars, global_volume_bars=global_vol_bars
            )
            if not indicators:
                return True, 'indicators_failed_defer'

            # Re-run the core signal check
            new_signal, new_confidence = get_signal(
                ema5=indicators.get('ema5'),
                ema8=indicators.get('ema8'),
                ema13=indicators.get('ema13'),
                ema20=indicators.get('ema20'),
                rsi=indicators.get('rsi'),
                adx=indicators.get('adx'),
                volume=indicators.get('volume'),
                avg_volume=indicators.get('avg_volume'),
                price=indicators.get('price'),
                ema20_prev3=indicators.get('ema20_prev3'),
                ema50=indicators.get('ema50'),
                ema50_prev12=indicators.get('ema50_prev12'),
                rsi_prev3=indicators.get('rsi_prev3'),
                rsi_prev2=indicators.get('rsi_prev2'),
                ema5_prev1=indicators.get('ema5_prev1'),
                ema8_prev1=indicators.get('ema8_prev1'),
                ema5_prev2=indicators.get('ema5_prev2'),
                ema8_prev2=indicators.get('ema8_prev2'),
                ema13_prev1=indicators.get('ema13_prev1'),
                ema13_prev2=indicators.get('ema13_prev2'),
                adx_prev1=indicators.get('adx_prev1'),
                high_20=indicators.get('high_20'),
                low_20=indicators.get('low_20'),
            )

            if new_signal != original_direction:
                return False, f'signal_flipped_{original_direction}_to_{new_signal}'
            if new_confidence is None or new_confidence == "NO_TRADE":
                return False, 'confidence_lost'

            # Check BTC-level filters (refetch BTC)
            btc_ohlcv = await binance_service.get_ohlcv('BTC/USDT:USDT', '5m', 100)
            if btc_ohlcv:
                btc_ind = calculate_indicators(btc_ohlcv)
                if btc_ind:
                    new_btc_adx = btc_ind.get('adx')
                    new_btc_adx_prev = btc_ind.get('adx_prev1')
                    new_btc_rsi = btc_ind.get('rsi')

                    th = tc.thresholds
                    # BTC ADX direction filter (independent per Option B refactor)
                    adx_dir_cfg = getattr(th, f'btc_adx_dir_{original_direction.lower()}', 'both')
                    if new_btc_adx is not None and new_btc_adx_prev is not None:
                        if adx_dir_cfg == 'rising' and new_btc_adx <= new_btc_adx_prev:
                            return False, 'btc_adx_direction_not_rising'
                        if adx_dir_cfg == 'falling' and new_btc_adx >= new_btc_adx_prev:
                            return False, 'btc_adx_direction_not_falling'

                    # BTC ADX range
                    if original_direction == 'LONG':
                        btc_adx_min = getattr(th, 'btc_adx_min_long', 0)
                        btc_adx_max = getattr(th, 'btc_adx_max_long', 100)
                    else:
                        btc_adx_min = getattr(th, 'btc_adx_min_short', 0)
                        btc_adx_max = getattr(th, 'btc_adx_max_short', 100)
                    if new_btc_adx is not None and (new_btc_adx < btc_adx_min or new_btc_adx > btc_adx_max):
                        return False, f'btc_adx_out_of_range_{round(new_btc_adx, 1)}'

                    # BTC RSI range — ONLY checked when BTC Global is enabled.
                    # Apr 30 bug fix: this previously ran unconditionally, while at
                    # entry time (services/trading_engine.py ~line 3439) the BTC RSI
                    # check is gated inside `if btc_global_enabled:`. The mismatch
                    # caused legitimate entries to be blocked from taker fallback by
                    # a filter that didn't actually apply at entry. The Phase 2 plan
                    # is to move BTC RSI into "BTC Independent Filters" alongside
                    # BTC ADX, but until that ships, re-validation must mirror entry
                    # behaviour exactly.
                    btc_global = getattr(th, 'btc_global_filter_enabled', False)
                    if btc_global:
                        if original_direction == 'LONG':
                            btc_rsi_min = getattr(th, 'btc_rsi_min_long', 0)
                            btc_rsi_max = getattr(th, 'btc_rsi_max_long', 100)
                        else:
                            btc_rsi_min = getattr(th, 'btc_rsi_min_short', 0)
                            btc_rsi_max = getattr(th, 'btc_rsi_max_short', 100)
                        if new_btc_rsi is not None and (new_btc_rsi < btc_rsi_min or new_btc_rsi > btc_rsi_max):
                            return False, f'btc_rsi_out_of_range_{round(new_btc_rsi, 1)}'

            return True, 'ok'
        except Exception as e:
            logger.error(f"[REVALIDATE] {pair}: Error during signal re-validation: {e}")
            return True, 'error_defer'  # FAIL OPEN

    def _record_signal_expired(self, pair: str, direction: str, confidence: str, reason: str):
        """Record a signal-expiration event for in-memory tracking (Amendment #7)."""
        self.signal_expired_reasons[reason] = self.signal_expired_reasons.get(reason, 0) + 1
        entry = {
            'pair': pair,
            'direction': direction,
            'confidence': confidence,
            'reason': reason,
            'time': datetime.utcnow().isoformat(),
        }
        self.signal_expired_log_recent.append(entry)
        if len(self.signal_expired_log_recent) > self._signal_expired_log_max:
            self.signal_expired_log_recent.pop(0)
        logger.warning(
            f"[SIGNAL_EXPIRED] {pair} {direction} {confidence}: {reason} — taker fallback aborted"
        )

    async def _record_signal_expired_order(
        self, db: AsyncSession, pair: str, direction: str, confidence: str,
        reason: str, entry_price: float,
        # Wait-time capture (May 2 enrichment) — actual maker-wait elapsed before
        # re-validation killed the entry. opened_at is back-dated so closed_at -
        # opened_at == real wait. None means "wait time not tracked" (legacy path).
        wait_seconds: Optional[float] = None,
        # Entry-indicator capture (May 2 enrichment) — same fields as a CLOSED
        # Order. All optional; missing fields stay NULL in DB. Available in scope
        # at open_position call sites because they're already function params.
        entry_gap: Optional[float] = None,
        entry_ema_gap_5_8: Optional[float] = None,
        entry_ema_gap_8_13: Optional[float] = None,
        entry_ema5_stretch: Optional[float] = None,
        entry_rsi: Optional[float] = None,
        entry_rsi_prev: Optional[float] = None,
        entry_adx: Optional[float] = None,
        entry_adx_prev: Optional[float] = None,
        entry_ema20_slope: Optional[float] = None,
        entry_btc_ema20_slope: Optional[float] = None,
        entry_btc_adx: Optional[float] = None,
        entry_btc_adx_prev: Optional[float] = None,
        entry_btc_rsi: Optional[float] = None,
        entry_btc_rsi_prev: Optional[float] = None,
        entry_btc_rsi_prev6: Optional[float] = None,
        entry_btc_atr_pct: Optional[float] = None,
        entry_btc_rsi_1h: Optional[float] = None,
        entry_btc_rsi_1h_prev: Optional[float] = None,
        entry_price_vs_ema5_pct: Optional[float] = None,
        entry_global_volume_ratio: Optional[float] = None,
        entry_pair_volume_ratio: Optional[float] = None,
        entry_bull_pct: Optional[float] = None,
        entry_bear_pct: Optional[float] = None,
        entry_range_position: Optional[float] = None,
        entry_adx_delta: Optional[float] = None,
        entry_quality_score: Optional[int] = None,
        entry_btc_regime: Optional[str] = None,
        entry_pos_di: Optional[float] = None,
        entry_neg_di: Optional[float] = None,
        entry_atr_pct: Optional[float] = None,
        entry_ema50_slope: Optional[float] = None,
        entry_funding_rate: Optional[float] = None,
        entry_pair_ema20_ema50_gap_pct: Optional[float] = None,
        entry_dist_from_ema13_pct: Optional[float] = None,
        entry_btc_dist_from_ema13_pct: Optional[float] = None,
        entry_btc_1h_slope: Optional[float] = None,
    ):
        """Persist a signal-expired entry attempt as a minimal Order row for reporting.

        Amendment #7 (Apr 18) shipped this with status='SIGNAL_EXPIRED' so the
        operator could see the rate of aborted entries via Entry Type Performance.
        status='SIGNAL_EXPIRED' keeps these rows out of PnL/WR aggregations
        (which filter on 'CLOSED'/'SIGNAL_EXPIRED' separately).

        May 2 enrichment: now also persists entry-indicator values + wait_seconds
        so aborted entries can be compared against Winners L / Losers L on the
        same dimensions (Entry Conditions by Outcome). Without this we could not
        tell whether re-validation was correctly self-protecting (aborts match
        loser profile) or murdering good trades (aborts match winner profile).
        Historical SIGNAL_EXPIRED rows persisted before this change have NULL
        indicator values forever — only post-deploy aborts are analyzable.
        """
        try:
            now = datetime.utcnow()
            opened_at = (now - timedelta(seconds=wait_seconds)) if wait_seconds is not None else now
            _pc1, _pc2, _pc3, _pc4, _pc5, _pc6, _pc7, _pc8, _pc9, _pc_any = _compute_pattern_c_match(
                direction=direction,
                rng_pos=entry_range_position,
                pair_gap=entry_pair_ema20_ema50_gap_pct,
                adx_delta=entry_adx_delta,
                btc_rsi=entry_btc_rsi,
                btc_rsi_prev=entry_btc_rsi_prev,
                btc_adx=entry_btc_adx,
                btc_adx_prev=entry_btc_adx_prev,
                btc_gap=globals().get('_current_btc_trend_gap_pct'),
                stretch=entry_ema5_stretch,
                pair_adx=entry_adx,
                btc_atr=entry_btc_atr_pct,
                ema20_slope=entry_ema20_slope,
                ema50_slope=entry_ema50_slope,
            )
            # Pattern W tracker (May 21 — lifted to entry, observation flags here too)
            _pw1, _pw2, _pw3, _pw4, _pw5, _pw6, _pw_any = _compute_pattern_w_match(
                direction=direction,
                rsi=entry_rsi,
                adx=entry_adx,
                adx_delta=entry_adx_delta,
                stretch=entry_ema5_stretch,
                rng_pos=entry_range_position,
                pair_gap=entry_pair_ema20_ema50_gap_pct,
                btc_rsi=entry_btc_rsi,
                btc_adx=entry_btc_adx,
                btc_atr=entry_btc_atr_pct,
                btc_gap=globals().get('_current_btc_trend_gap_pct'),
                pair_vol_ratio=None,  # not captured in scope here; pair_vol_ratio is local to live entry only
            )
            order = Order(
                pair=pair,
                direction=direction,
                status="SIGNAL_EXPIRED",
                entry_price=entry_price,
                current_price=entry_price,
                exit_price=entry_price,
                investment=0.0,
                leverage=1,
                notional_value=0.0,
                quantity=0.0,
                confidence=confidence,
                entry_fee=0.0,
                exit_fee=0.0,
                total_fee=0.0,
                pnl=0.0,
                pnl_percentage=0.0,
                peak_pnl=0.0,
                trough_pnl=0.0,
                entry_order_type="SIGNAL_EXPIRED",
                exit_order_type=None,
                close_reason=f"SIGNAL_EXPIRED:{reason}",
                opened_at=opened_at,
                closed_at=now,
                is_paper=self.is_paper_mode,
                # Entry indicators (May 2)
                entry_gap=entry_gap,
                entry_ema_gap_5_8=entry_ema_gap_5_8,
                entry_ema_gap_8_13=entry_ema_gap_8_13,
                entry_ema5_stretch=entry_ema5_stretch,
                entry_rsi=entry_rsi,
                entry_rsi_prev=entry_rsi_prev,
                entry_adx=entry_adx,
                entry_adx_prev=entry_adx_prev,
                entry_ema20_slope=entry_ema20_slope,
                entry_btc_ema20_slope=entry_btc_ema20_slope,
                entry_btc_adx=entry_btc_adx,
                entry_btc_adx_prev=entry_btc_adx_prev,
                entry_btc_rsi=entry_btc_rsi,
                entry_btc_rsi_prev=entry_btc_rsi_prev,
                entry_btc_rsi_prev6=entry_btc_rsi_prev6,
                entry_btc_atr_pct=entry_btc_atr_pct,
                entry_btc_rsi_1h=entry_btc_rsi_1h,
                entry_btc_rsi_1h_prev=entry_btc_rsi_1h_prev,
                entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
                entry_global_volume_ratio=entry_global_volume_ratio,
                entry_pair_volume_ratio=entry_pair_volume_ratio,
                entry_bull_pct=entry_bull_pct,
                entry_bear_pct=entry_bear_pct,
                entry_range_position=entry_range_position,
                entry_adx_delta=entry_adx_delta,
                entry_quality_score=entry_quality_score,
                entry_btc_regime=entry_btc_regime,
                entry_btc_trend_gap_pct=globals().get('_current_btc_trend_gap_pct'),
                entry_pos_di=entry_pos_di,
                entry_neg_di=entry_neg_di,
                entry_atr_pct=entry_atr_pct,
                entry_ema50_slope=entry_ema50_slope,
                entry_funding_rate=entry_funding_rate,
                entry_pair_ema20_ema50_gap_pct=entry_pair_ema20_ema50_gap_pct,
                entry_dist_from_ema13_pct=entry_dist_from_ema13_pct,
                entry_btc_dist_from_ema13_pct=entry_btc_dist_from_ema13_pct,
                entry_btc_1h_slope=entry_btc_1h_slope,
                entry_pattern_c1_match=_pc1,
                entry_pattern_c2_match=_pc2,
                entry_pattern_c3_match=_pc3,
                entry_pattern_c4_match=_pc4,
                entry_pattern_c5_match=_pc5,
                entry_pattern_c6_match=_pc6,
                entry_pattern_c7_match=_pc7,
                entry_pattern_c8_match=_pc8,
                entry_pattern_c9_match=_pc9,
                entry_pattern_c_any_match=_pc_any,
                # Pattern W (May 21 — lifted to entry)
                entry_pattern_w1_match=_pw1,
                entry_pattern_w2_match=_pw2,
                entry_pattern_w3_match=_pw3,
                entry_pattern_w4_match=_pw4,
                entry_pattern_w5_match=_pw5,
                entry_pattern_w6_match=_pw6,
                entry_pattern_w_any_match=_pw_any,
            )
            db.add(order)
            await db.commit()
        except Exception as e:
            logger.error(f"[SIGNAL_EXPIRED] {pair}: Failed to persist aborted-entry row: {e}")
            try:
                await db.rollback()
            except Exception:
                pass

    async def _try_maker_entry(
        self, symbol: str, side: str, amount: float, leverage: int,
        direction: str, pair: str, notional_value: float,
        maker_fee_rate: float, taker_fee_rate: float,
        confidence: Optional[str] = None,
    ) -> Optional[Dict]:
        """Attempt a maker (limit) entry, falling back to taker (market) on timeout.

        Amendment #7 (Apr 18): if `confidence` is provided, re-validates the entry
        signal at timeout before placing the taker fallback. Returns
        `{'entry_order_type': 'SIGNAL_EXPIRED', 'skipped': True, 'reason': ...}`
        when re-validation fails — caller should create a SIGNAL_EXPIRED Order row
        for tracking but NOT open a position.
        """
        tc = config.trading_config
        timeout = getattr(tc, 'maker_timeout_seconds', 15)
        offset_ticks = getattr(tc, 'maker_offset_ticks', 2)

        ob = await binance_service.fetch_orderbook(symbol)
        if not ob:
            logger.warning(f"[MAKER_ENTRY] {pair}: Orderbook unavailable, falling back to taker")
            result = await binance_service.create_market_order(symbol, side, amount, leverage)
            if not result:
                return None
            fill_amount = result.get('amount', amount)
            fill_price = result['price']
            return {
                'id': result['id'], 'price': fill_price,
                'amount': fill_amount,
                'entry_fee': fill_amount * fill_price * taker_fee_rate,
                'entry_order_type': 'TAKER_FALLBACK',
            }

        tick_size = await binance_service.get_tick_size(symbol)
        if direction == 'LONG':
            limit_price = ob['best_bid'] - (offset_ticks * tick_size)
        else:
            limit_price = ob['best_ask'] + (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_ENTRY] {pair}: Placing limit {side} @ {limit_price} "
                     f"(bid={ob['best_bid']}, ask={ob['best_ask']}, offset={offset_ticks} ticks)")

        limit_result = await binance_service.create_limit_order(
            symbol=symbol, side=side, amount=amount, price=limit_price, leverage=leverage
        )
        if not limit_result:
            logger.warning(f"[MAKER_ENTRY] {pair}: Limit order failed, falling back to taker")
            result = await binance_service.create_market_order(symbol, side, amount, leverage)
            if not result:
                return None
            fill_amount = result.get('amount', amount)
            fill_price = result['price']
            return {
                'id': result['id'], 'price': fill_price,
                'amount': fill_amount,
                'entry_fee': fill_amount * fill_price * taker_fee_rate,
                'entry_order_type': 'TAKER_FALLBACK',
            }

        order_id = limit_result['id']
        polls = max(1, timeout // 2)
        filled = False

        for i in range(polls):
            await asyncio.sleep(2)
            status = await binance_service.fetch_order_status(symbol, order_id)
            if not status:
                continue
            if status['status'] == 'closed':
                filled = True
                fill_price = status['average'] or limit_price
                fill_amount = status['filled'] or amount
                fill_fee = fill_amount * fill_price * maker_fee_rate
                logger.info(f"[MAKER_ENTRY] {pair}: Limit FILLED @ {fill_price} after {(i+1)*2}s")
                return {
                    'id': order_id, 'price': fill_price,
                    'amount': fill_amount, 'entry_fee': fill_fee,
                    'entry_order_type': 'MAKER',
                }

        # Timeout -- cancel and check for partial fill
        logger.info(f"[MAKER_ENTRY] {pair}: Timeout after {timeout}s, cancelling limit order")
        await binance_service.cancel_order(symbol, order_id)
        await asyncio.sleep(0.5)

        final_status = await binance_service.fetch_order_status(symbol, order_id)
        filled_qty = final_status['filled'] if final_status else 0

        if filled_qty and filled_qty > 0:
            fill_price = final_status['average'] or limit_price
            fill_fee = filled_qty * fill_price * maker_fee_rate
            logger.info(f"[MAKER_ENTRY] {pair}: Partial fill {filled_qty}/{amount} @ {fill_price}")
            return {
                'id': order_id, 'price': fill_price,
                'amount': filled_qty, 'entry_fee': fill_fee,
                'entry_order_type': 'MAKER',
            }

        # No fill at all -- re-validate signal before taker fallback (Amendment #7).
        # Toggle (May 4, 2026): if `revalidate_on_taker_fallback` is False, skip
        # re-validation and fall back to taker immediately (pre-Apr-18 behaviour).
        wait_seconds_elapsed = float(timeout)
        revalidate_enabled = getattr(config.trading_config, 'revalidate_on_taker_fallback', True)
        if revalidate_enabled and confidence is not None:
            is_valid, revalidate_reason = await self._revalidate_entry_signal(
                symbol, pair, direction, confidence
            )
            if not is_valid:
                self._record_signal_expired(pair, direction, confidence, revalidate_reason)
                return {
                    'entry_order_type': 'SIGNAL_EXPIRED',
                    'skipped': True,
                    'reason': revalidate_reason,
                    'wait_seconds': wait_seconds_elapsed,
                }
            logger.info(f"[MAKER_ENTRY] {pair}: No fill, signal re-validated, falling back to market order")
        else:
            logger.info(f"[MAKER_ENTRY] {pair}: No fill, re-validation disabled, falling back to market order")
        result = await binance_service.create_market_order(symbol, side, amount, leverage)
        if not result:
            return None
        fill_amount = result.get('amount', amount)
        fill_price = result['price']
        return {
            'id': result['id'], 'price': fill_price,
            'amount': fill_amount,
            'entry_fee': fill_amount * fill_price * taker_fee_rate,
            'entry_order_type': 'TAKER_FALLBACK',
        }

    async def _simulate_maker_entry_paper(
        self, pair: str, direction: str, current_price: float,
        notional_value: float, maker_fee_rate: float, taker_fee_rate: float,
        confidence: Optional[str] = None,
    ) -> Dict:
        """Simulate maker entry for paper trading using WebSocket prices.

        Amendment #7 (Apr 18): if `confidence` is provided, re-validates the entry
        signal at timeout before falling back to taker. Returns
        `{'entry_order_type': 'SIGNAL_EXPIRED', 'skipped': True, 'reason': ...}`
        when re-validation fails.
        """
        tc = config.trading_config
        timeout = getattr(tc, 'maker_timeout_seconds', 15)
        offset_ticks = getattr(tc, 'maker_offset_ticks', 2)

        # Estimate tick size from price magnitude
        if current_price >= 10000:
            tick_size = 0.10
        elif current_price >= 100:
            tick_size = 0.01
        elif current_price >= 1:
            tick_size = 0.001
        else:
            tick_size = 0.0001

        if direction == 'LONG':
            limit_price = current_price - (offset_ticks * tick_size)
        else:
            limit_price = current_price + (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_PAPER] {pair}: Simulating limit {direction} @ {limit_price} "
                     f"(current={current_price}, offset={offset_ticks} ticks)")

        # Monitor WebSocket prices for the timeout window
        polls = max(1, timeout // 2)
        for i in range(polls):
            await asyncio.sleep(2)
            tracker = websocket_tracker.get_tracker(pair)
            if not tracker or not tracker.last_price:
                continue

            ws_price = tracker.last_price
            if direction == 'LONG' and ws_price <= limit_price:
                logger.info(f"[MAKER_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price,
                    'entry_fee': notional_value * maker_fee_rate,
                    'entry_order_type': 'MAKER',
                }
            elif direction == 'SHORT' and ws_price >= limit_price:
                logger.info(f"[MAKER_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price,
                    'entry_fee': notional_value * maker_fee_rate,
                    'entry_order_type': 'MAKER',
                }

        # No fill -- re-validate signal before taker fallback (Amendment #7).
        # Toggle (May 4, 2026): if `revalidate_on_taker_fallback` is False, skip
        # re-validation and fall back to taker immediately (pre-Apr-18 behaviour).
        wait_seconds_elapsed = float(timeout)
        revalidate_enabled = getattr(config.trading_config, 'revalidate_on_taker_fallback', True)
        if revalidate_enabled and confidence is not None:
            symbol_ccxt = pair.replace('USDT', '/USDT:USDT')
            is_valid, revalidate_reason = await self._revalidate_entry_signal(
                symbol_ccxt, pair, direction, confidence
            )
            if not is_valid:
                self._record_signal_expired(pair, direction, confidence, revalidate_reason)
                return {
                    'entry_order_type': 'SIGNAL_EXPIRED',
                    'skipped': True,
                    'reason': revalidate_reason,
                    'price': current_price,
                    'entry_fee': 0.0,
                    'wait_seconds': wait_seconds_elapsed,
                }

        tracker = websocket_tracker.get_tracker(pair)
        fallback_price = tracker.last_price if tracker and tracker.last_price else current_price
        logger.info(f"[MAKER_PAPER] {pair}: No fill after {timeout}s, signal re-validated, taker fallback @ {fallback_price}")
        return {
            'price': fallback_price,
            'entry_fee': notional_value * taker_fee_rate,
            'entry_order_type': 'TAKER_FALLBACK',
        }

    async def _fetch_actual_fill_price(self, order, fallback_price: float) -> float:
        """Fetch the actual fill price from Binance trade history for an externally closed order."""
        symbol = order.pair.replace('USDT', '/USDT:USDT')
        try:
            trades = await binance_service.fetch_my_trades(symbol, limit=10)
            if trades:
                close_side = 'sell' if order.direction == 'LONG' else 'buy'
                relevant = [t for t in trades if t['side'] == close_side]
                if relevant:
                    latest = relevant[-1]
                    logger.info(
                        f"[FILL_PRICE] {order.pair}: Found actual fill @ {latest['price']} "
                        f"(side={latest['side']}, time={latest['datetime']})"
                    )
                    return latest['price']
        except Exception as e:
            logger.warning(f"[FILL_PRICE] {order.pair}: Could not fetch trade history: {e}")

        logger.warning(f"[FILL_PRICE] {order.pair}: Using fallback price {fallback_price}")
        return fallback_price

    async def _try_maker_exit(
        self, symbol: str, side: str, amount: float,
        pair: str, direction: str, current_price: float
    ) -> Dict:
        """Attempt a maker (limit) exit, falling back to taker (market) on timeout.
        For LONG exits: sell at best_ask + offset (higher = better).
        For SHORT exits: buy at best_bid - offset (lower = better)."""
        tc = config.trading_config
        timeout = getattr(tc, 'maker_exit_timeout_seconds', 10)
        offset_ticks = getattr(tc, 'maker_exit_offset_ticks', 2)
        maker_fee_rate = getattr(tc, 'maker_fee', 0.00018)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)
        close_side = 'sell' if direction == 'LONG' else 'buy'

        ob = await binance_service.fetch_orderbook(symbol)
        if not ob:
            logger.warning(f"[MAKER_EXIT] {pair}: Orderbook unavailable, falling back to taker")
            try:
                result = await binance_service.close_position(symbol, direction, amount)
                if not result:
                    logger.error(f"[MAKER_EXIT] {pair}: Taker fallback ALSO failed — position NOT closed on Binance")
                    return None
                return {
                    'price': result['price'], 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK',
                    'decision_price': current_price,
                }
            except Exception as e:
                logger.critical(
                    f"[MAKER_EXIT] {pair}: Taker fallback CRASHED (orderbook unavailable path): {e}. "
                    f"Returning fallback with current_price={current_price}."
                )
                return {
                    'price': current_price, 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK_RECOVERED',
                    'decision_price': current_price,
                }

        tick_size = await binance_service.get_tick_size(symbol)
        if direction == 'LONG':
            limit_price = ob['best_ask'] + (offset_ticks * tick_size)
        else:
            limit_price = ob['best_bid'] - (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_EXIT] {pair}: Placing limit {close_side} @ {limit_price} "
                     f"(bid={ob['best_bid']}, ask={ob['best_ask']}, offset={offset_ticks} ticks)")

        limit_result = await binance_service.create_limit_order(
            symbol=symbol, side=close_side, amount=amount, price=limit_price, leverage=1, is_close=True
        )
        if not limit_result:
            logger.warning(f"[MAKER_EXIT] {pair}: Limit order failed, falling back to taker")
            try:
                result = await binance_service.close_position(symbol, direction, amount)
                if not result:
                    logger.error(f"[MAKER_EXIT] {pair}: Taker fallback ALSO failed — position NOT closed on Binance")
                    return None
                return {
                    'price': result['price'], 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK',
                    'decision_price': current_price,
                }
            except Exception as e:
                logger.critical(
                    f"[MAKER_EXIT] {pair}: Taker fallback CRASHED (limit order failed path): {e}. "
                    f"Returning fallback with current_price={current_price}."
                )
                return {
                    'price': current_price, 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK_RECOVERED',
                    'decision_price': current_price,
                }

        order_id = limit_result['id']
        polls = max(1, timeout // 2)

        for i in range(polls):
            await asyncio.sleep(2)
            status = await binance_service.fetch_order_status(symbol, order_id)
            if not status:
                continue
            if status['status'] == 'closed':
                fill_price = status['average'] or limit_price
                logger.info(f"[MAKER_EXIT] {pair}: Limit FILLED @ {fill_price} after {(i+1)*2}s")
                return {
                    'price': fill_price, 'fee_rate': maker_fee_rate,
                    'exit_order_type': 'MAKER',
                    'decision_price': current_price,
                }

        logger.info(f"[MAKER_EXIT] {pair}: Timeout after {timeout}s, cancelling limit order")
        await binance_service.cancel_order(symbol, order_id)
        await asyncio.sleep(0.5)

        final_status = await binance_service.fetch_order_status(symbol, order_id)
        filled_qty = final_status['filled'] if final_status else 0

        if filled_qty and filled_qty > 0:
            fill_price = final_status['average'] or limit_price
            logger.info(f"[MAKER_EXIT] {pair}: Partial fill {filled_qty}/{amount} @ {fill_price}, market closing remainder")
            remainder = amount - filled_qty
            if remainder > 0:
                await binance_service.close_position(symbol, direction, remainder)
            return {
                'price': fill_price, 'fee_rate': maker_fee_rate,
                'exit_order_type': 'MAKER',
                'decision_price': current_price,
            }

        logger.info(f"[MAKER_EXIT] {pair}: No fill, falling back to market order")
        try:
            result = await binance_service.close_position(symbol, direction, amount)
            if not result:
                logger.error(f"[MAKER_EXIT] {pair}: Taker fallback ALSO failed — position NOT closed on Binance")
                return None
            return {
                'price': result['price'], 'fee_rate': taker_fee_rate,
                'exit_order_type': 'TAKER_FALLBACK',
                'decision_price': current_price,
            }
        except Exception as e:
            logger.critical(
                f"[MAKER_EXIT] {pair}: Taker fallback CRASHED after market order likely executed on Binance: {e}. "
                f"Returning fallback result with current_price={current_price} to allow DB closure."
            )
            return {
                'price': current_price, 'fee_rate': taker_fee_rate,
                'exit_order_type': 'TAKER_FALLBACK_RECOVERED',
                'decision_price': current_price,
            }

    async def _simulate_maker_exit_paper(
        self, pair: str, direction: str, current_price: float
    ) -> Dict:
        """Simulate maker exit for paper trading using WebSocket prices."""
        tc = config.trading_config
        timeout = getattr(tc, 'maker_exit_timeout_seconds', 10)
        offset_ticks = getattr(tc, 'maker_exit_offset_ticks', 2)
        maker_fee_rate = getattr(tc, 'maker_fee', 0.00018)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)

        if current_price >= 10000:
            tick_size = 0.10
        elif current_price >= 100:
            tick_size = 0.01
        elif current_price >= 1:
            tick_size = 0.001
        else:
            tick_size = 0.0001

        if direction == 'LONG':
            limit_price = current_price + (offset_ticks * tick_size)
        else:
            limit_price = current_price - (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_EXIT_PAPER] {pair}: Simulating limit exit {direction} @ {limit_price} "
                     f"(current={current_price}, offset={offset_ticks} ticks)")

        polls = max(1, timeout // 2)
        for i in range(polls):
            await asyncio.sleep(2)
            tracker = websocket_tracker.get_tracker(pair)
            if not tracker or not tracker.last_price:
                continue

            ws_price = tracker.last_price
            if direction == 'LONG' and ws_price >= limit_price:
                logger.info(f"[MAKER_EXIT_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price, 'fee_rate': maker_fee_rate,
                    'exit_order_type': 'MAKER',
                }
            elif direction == 'SHORT' and ws_price <= limit_price:
                logger.info(f"[MAKER_EXIT_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price, 'fee_rate': maker_fee_rate,
                    'exit_order_type': 'MAKER',
                }

        tracker = websocket_tracker.get_tracker(pair)
        fallback_price = tracker.last_price if tracker and tracker.last_price else current_price
        logger.info(f"[MAKER_EXIT_PAPER] {pair}: No fill after {timeout}s, taker fallback @ {fallback_price}")
        return {
            'price': fallback_price, 'fee_rate': taker_fee_rate,
            'exit_order_type': 'TAKER_FALLBACK',
        }

    def _flip_scan_ctx(self, L):
        """Jun 15: pull the scan-state market context (volume / breadth / rank) for a flip.
        Caller passes `locals()`. Pair-specific values are scan_and_trade locals; the
        market-wide ones (_market_bull_pct/_market_bear_pct are module GLOBALS, declared
        `global` in scan_and_trade; _global_volume_ratio is a local) are read local-then-
        global. Uses .get() so a not-yet-assigned name never raises. Keys = entry_* columns.
        Whole body wrapped fail-silent → {} so a flip helper can NEVER break the scan loop."""
        try:
            g = globals()
            def pick(k):
                v = L.get(k)
                return v if v is not None else g.get(k)
            return {
                'entry_global_volume_ratio': pick('_global_volume_ratio'),
                'entry_pair_volume_ratio': L.get('_pair_volume_ratio'),
                'entry_bull_pct': pick('_market_bull_pct'),
                'entry_bear_pct': pick('_market_bear_pct'),
                'entry_pair_volume_24h_usd': L.get('volume_24h'),
                'entry_pair_rank': L.get('_pair_rank'),
            }
        except Exception:
            return {}

    def _flip_entry_fields(self, indicators, flip_dir=None, scan=None):
        """Jun 15: build the FULL entry-indicator kwarg set for a flip Order from the raw
        `indicators` dict (+ BTC globals + scan-state), mirroring the momentum entry path so
        flip trades carry the SAME analytics columns as normal trades — gaps (5-20 / 5-8 /
        8-13), fan-ratio, EMA5 stretch, range-position, dist-EMA13, ADX-delta, ±DI, pair
        EMA20/EMA50 slopes, BTC adx/rsi/slope, volume ratios, breadth, rank, quality score.
        flip_dir → compute the quality score for the fade's direction. scan → market context
        (volume/breadth/rank) from _flip_scan_ctx. Fail-silent PER FIELD: a missing input
        drops that one field, never the flip. Fields with NO source at the flip's firing
        point (funding rate, BTC RSI/ADX history, gap-expand A/B tag — all computed later in
        the entry pipeline a blocked fade never reaches; none drive a 'Performance by' table)
        stay NULL by design."""
        ind = indicators or {}
        g = globals()
        out = {}
        def put(k, fn):
            try:
                v = fn()
                if v is not None:
                    out[k] = v
            except Exception:
                pass
        # Whole computation wrapped fail-silent → returns whatever was accumulated so a
        # flip helper can NEVER raise into scan_and_trade / open_position (the #1 invariant).
        try:
            px = ind.get('price')
            e5, e8, e13, e50, e20 = ind.get('ema5'), ind.get('ema8'), ind.get('ema13'), ind.get('ema50'), ind.get('ema20')
            e20p3, e50p12 = ind.get('ema20_prev3'), ind.get('ema50_prev12')
            # ── pair fields (recomputed exactly as the momentum path does) ──
            put('entry_gap', lambda: round(abs((e5 - e20) / px * 100), 4))   # EMA5-EMA20 gap (Entry Gap 5-20 table)
            put('entry_rsi', lambda: round(ind['rsi'], 2))
            put('entry_rsi_prev', lambda: round(ind['rsi_prev2'], 2))
            put('entry_adx', lambda: round(ind['adx'], 4))
            put('entry_adx_prev', lambda: round(ind['adx_prev1'], 4))
            put('entry_adx_delta', lambda: round(ind['adx'] - ind['adx_prev1'], 4))
            put('entry_pos_di', lambda: ind['pos_di'])
            put('entry_neg_di', lambda: ind['neg_di'])
            put('entry_ema_gap_5_8', lambda: round(abs((e5 - e8) / e8 * 100), 4))
            put('entry_ema_gap_8_13', lambda: round(abs((e8 - e13) / e13 * 100), 4))
            put('entry_ema5_stretch', lambda: round(abs(px - e5) / px * 100, 4))
            put('entry_price_vs_ema5_pct', lambda: round((px - e5) / e5 * 100, 4))
            put('entry_atr_pct', lambda: round(ind['atr'] / px * 100, 4))
            put('entry_pair_ema20_ema50_gap_pct', lambda: round((e13 - e50) / e50 * 100, 4))
            put('entry_dist_from_ema13_pct', lambda: round((px - e13) / e13 * 100, 4))
            put('entry_range_position', lambda: round((px - ind['low_20']) / (ind['high_20'] - ind['low_20']) * 100, 1))
            put('entry_ema20_slope', lambda: round((e20 - e20p3) / e20p3 * 100, 4))    # pair EMA20 slope (Pair EMA20 Slope table)
            put('entry_ema50_slope', lambda: round((e50 - e50p12) / e50p12 * 100, 4))  # pair EMA50 slope
            # ── BTC fields from live module globals ──
            put('entry_btc_adx', lambda: round(g.get('_current_btc_adx'), 4))
            put('entry_btc_rsi', lambda: round(g.get('_current_btc_rsi'), 1))
            put('entry_btc_ema20_slope', lambda: g.get('_btc_ema20_slope_pct'))
            put('entry_btc_1h_slope', lambda: g.get('_current_btc_1h_slope'))
            put('entry_btc_dist_from_ema13_pct', lambda: round((g['_current_btc_price'] - g['_current_btc_ema13']) / g['_current_btc_ema13'] * 100, 4))
            # BTC prev/higher-TF COMPANIONS (Jun 15) — the "vs prev candle / vs 6-ago / 1h"
            # values the "Performance by BTC ... Direction / Volatility / 1h RSI" tables compare
            # against. Without these a flip is invisible to every one of those tables.
            put('entry_btc_adx_prev', lambda: round(g.get('_current_btc_adx_prev'), 4))
            put('entry_btc_rsi_prev', lambda: round(g.get('_current_btc_rsi_prev'), 1))
            put('entry_btc_rsi_prev6', lambda: round(g.get('_current_btc_rsi_prev6'), 1))
            put('entry_btc_atr_pct', lambda: g.get('_current_btc_atr_pct'))
            put('entry_btc_rsi_1h', lambda: g.get('_current_btc_rsi_1h'))
            put('entry_btc_rsi_1h_prev', lambda: g.get('_current_btc_rsi_1h_prev'))
            # ── scan-state market context (volume / breadth / rank) ──
            if scan:
                for k, v in scan.items():
                    if v is not None:
                        out[k] = v
            # ── quality score (0-6) for the fade's direction, from the assembled inputs ──
            if flip_dir:
                put('entry_quality_score', lambda: _calculate_quality_score(
                    flip_dir, out.get('entry_rsi'), out.get('entry_adx'), out.get('entry_gap'),
                    out.get('entry_bull_pct'), out.get('entry_bear_pct'), out.get('entry_btc_adx'),
                    out.get('entry_ema20_slope')))
        except Exception:
            pass
        return out

    async def _maybe_open_flip(self, db, pair, blocked_signal, source, indicators, isolate=False, entry_fields=None):
        """Flip Entry trigger — when `source` blocks `blocked_signal`, open the OPPOSITE
        direction as a NAKED mean-reversion entry (its own FLIP exit model). Fail-silent
        so a flip-path bug can NEVER break the caller. All risk controls (max-open,
        existing-position, cooldown, liquidity caps) are enforced inside open_position.
        isolate=True opens the flip in a FRESH DB session — required when called
        re-entrantly from INSIDE open_position (e.g. the LONG_UNMATCHED_ONLY block) so it
        can't disturb the outer transaction. entry_fields (Jun 15) = a ready dict of entry_*
        analytics kwargs so the flip Order carries the same columns as a normal trade; the
        caller supplies it (recomputed from indicators at the FAN site, or forwarded from
        open_position's own params at the LONG_UNMATCHED site)."""
        try:
            if not _flip_active(source):
                return
            flip_dir = "SHORT" if blocked_signal == "LONG" else "LONG"
            price = indicators.get('price') if indicators else None
            if not price or price <= 0:
                return
            _ef = dict(entry_fields or {})
            # Jun 16: source-namespaced flip filter layer — veto / size / exit-mode.
            _g = globals(); _ind = indicators or {}
            _ff_in = {
                'ema5_stretch': _ef.get('entry_ema5_stretch') if _ef.get('entry_ema5_stretch') is not None else _ind.get('ema5_stretch'),
                'btc_rsi': _ef.get('entry_btc_rsi') if _ef.get('entry_btc_rsi') is not None else _g.get('_current_btc_rsi'),
                'btc_adx': _ef.get('entry_btc_adx') if _ef.get('entry_btc_adx') is not None else _g.get('_current_btc_adx'),
            }
            _blocked, _reason, _flip_cell_mult, _flip_cell_lev_mult, _flip_exit_mode = _flip_filters(source, _ff_in)
            if _blocked:
                try: self._record_filter_block(_reason, flip_dir)
                except Exception: pass
                logger.info(f"[FLIP_FILTER] {pair}: {source} flip vetoed by {_reason} "
                            f"(stretch={_ff_in.get('ema5_stretch')}, btcRSI={_ff_in.get('btc_rsi')}, btcADX={_ff_in.get('btc_adx')})")
                return
            async def _open(_db):
                return await self.open_position(
                    db=_db, pair=pair, direction=flip_dir, confidence="STRONG_BUY",
                    current_price=price,
                    entry_rsi=_ef.pop('entry_rsi', None) or indicators.get('rsi'),
                    entry_adx=_ef.pop('entry_adx', None) or indicators.get('adx'),
                    entry_atr_pct=_ef.pop('entry_atr_pct', None) or indicators.get('atr_pct'),
                    flip_source=source, flip_cell_mult=_flip_cell_mult, flip_cell_lev_mult=_flip_cell_lev_mult, flip_exit_mode=_flip_exit_mode,
                    **_ef,
                )
            if isolate:
                async with AsyncSessionLocal() as _fdb:
                    order = await _open(_fdb)
            else:
                order = await _open(db)
            if order:
                logger.info(f"[FLIP_ENTRY] {pair}: {source} blocked {blocked_signal} -> opened {flip_dir} flip (id={order.id})")
        except Exception as e:
            logger.error(f"[FLIP_ENTRY] {pair}: flip open failed for {source}/{blocked_signal}: {e}")

    async def open_position(
        self,
        db: AsyncSession,
        pair: str,
        direction: str,
        confidence: str,
        current_price: float,
        entry_gap: float = None,
        entry_ema_gap_5_8: float = None,
        entry_ema_gap_8_13: float = None,
        entry_ema5_stretch: float = None,
        entry_rsi: float = None,
        entry_rsi_prev: float = None,
        entry_adx: float = None,
        entry_adx_prev: float = None,
        entry_macro_trend: str = None,
        entry_ema20_slope: float = None,
        entry_btc_ema20_slope: float = None,
        entry_btc_adx: float = None,
        entry_btc_adx_prev: float = None,
        entry_btc_rsi: float = None,
        entry_btc_rsi_prev: float = None,
        entry_btc_rsi_prev6: float = None,
        # May 15 PM: BTC Volatility Regime + BTC 1h RSI Direction (observation-only)
        entry_btc_atr_pct: float = None,
        entry_btc_rsi_1h: float = None,
        entry_btc_rsi_1h_prev: float = None,
        entry_price_vs_ema5_pct: float = None,
        entry_global_volume_ratio: float = None,
        entry_pair_volume_ratio: float = None,
        entry_bull_pct: float = None,
        entry_bear_pct: float = None,
        entry_range_position: float = None,
        entry_adx_delta: float = None,
        entry_quality_score: int = None,
        entry_btc_regime: str = None,
        # Exploration Analytics (Apr 28, observation-only)
        entry_pos_di: float = None,
        entry_neg_di: float = None,
        entry_atr_pct: float = None,
        entry_ema50_slope: float = None,
        entry_funding_rate: float = None,
        entry_pair_ema20_ema50_gap_pct: float = None,
        # May 13 PM: Entry Distance from EMA13 (Late Entry Risk dimension)
        entry_dist_from_ema13_pct: float = None,
        # May 14: BTC Market Extension / BTC Late Regime Risk dimension
        entry_btc_dist_from_ema13_pct: float = None,
        # May 14: BTC 1h EMA20 slope at entry (higher-TF macro context)
        entry_btc_1h_slope: float = None,
        # May 10: capture absolute pair 24h USD volume at entry for size-bucket analysis
        entry_pair_volume_24h_usd: float = None,
        entry_pair_rank: int = None,
        # Jun 8: gap-expanding relaxation A/B tag (prev2_only-admitted MARGINAL cohort)
        entry_gap_expand_marginal: bool = None,
        # Jun 14: Flip Entry sleeve — when set, this is a NAKED fade-the-block entry
        # (bypasses pattern/multiplier logic, base×registry sizing, FLIP exit model).
        flip_source: str = None,
        flip_cell_mult: float = 1.0,
        flip_cell_lev_mult: float = 1.0,
        flip_exit_mode: str = None,
    ) -> Optional[Order]:
        """Open a new position"""
        if not self.is_running:
            logger.warning(f"[SKIP] {pair}: Bot not running")
            return None

        # Jun 15: Flip Entry — the flip trigger fires mid-filter-chain, before the
        # momentum path computes entry_regime, so flips arrived with empty macro_trend /
        # btc_regime → they classified as NEUTRAL and vanished from the BULLISH/BEARISH
        # report sections (all-flips batch => empty tables). Populate both from the live
        # BTC globals, exactly as the momentum path does.
        if flip_source:
            if not entry_macro_trend:
                entry_macro_trend = globals().get('_current_btc_regime') or 'NEUTRAL'
            if not entry_btc_regime:
                try:
                    entry_btc_regime = classify_btc_regime(
                        globals().get('_current_btc_adx'),
                        globals().get('_current_btc_rsi'),
                        globals().get('_btc_ema20_slope_pct'))
                except Exception:
                    entry_btc_regime = None

        # Check if confidence level is enabled
        conf_config = config.trading_config.confidence_levels.get(confidence)
        if not conf_config or not conf_config.enabled:
            logger.warning(f"[SKIP] {pair}: {confidence} confidence not enabled")
            return None
        
        # Check max open positions limit
        total_open = await db.execute(
            select(func.count(Order.id)).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        # Jun 2: when redeploy_leftover is on, the count limit rises to the hard
        # ceiling and the gross-notional cap (below) + tradeable margin become the
        # real limiters. Default (redeploy off) keeps the plain max_open_positions.
        _inv_cfg = config.trading_config.investment
        _eff_max_pos = _inv_cfg.max_open_positions
        _redeploy_on = getattr(_inv_cfg, 'redeploy_leftover_enabled', False)
        if _redeploy_on:
            _eff_max_pos = max(_eff_max_pos, getattr(_inv_cfg, 'max_open_positions_hard', _eff_max_pos))
        _open_count_now = total_open.scalar()
        if _open_count_now >= _eff_max_pos:
            logger.warning(f"[SKIP] {pair}: Max open positions ({_eff_max_pos}) reached")
            return None
        # Jun 2: this open sits in the "redeploy band" if it's beyond the normal
        # max_open_positions — only reachable because redeploy raised the ceiling.
        # Recorded as REDEPLOY_OPEN after the Order commits (positive-event counter).
        _is_redeploy_open = _redeploy_on and _open_count_now >= _inv_cfg.max_open_positions
        
        # Check if we already have a position for this pair
        result = await db.execute(
            select(Order).where(
                and_(
                    Order.pair == pair,
                    Order.status == "OPEN",
                    Order.is_paper == self.is_paper_mode
                )
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info(f"[SKIP] {pair}: Already have open position")
            return None  # Already have position
        
        # Check cooldown - don't re-enter same pair too quickly after ANY close (win or loss)
        # CLAUDE.md May 26: cross-batch evidence on 919-trade pool — 84 same-pair re-entries
        # within 5min after a WINNING trade had 61.9% WR but -$731 net (2.71:1 R:R loss asymmetry).
        cooldown_minutes = config.trading_config.investment.cooldown_after_loss_minutes
        if cooldown_minutes > 0:
            cooldown_threshold = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
            result = await db.execute(
                select(Order).where(
                    and_(
                        Order.pair == pair,
                        Order.status == "CLOSED",
                        Order.is_paper == self.is_paper_mode,
                        Order.closed_at >= cooldown_threshold,
                    )
                ).order_by(desc(Order.closed_at)).limit(1)
            )
            recent_close = result.scalar_one_or_none()
            if recent_close:
                time_since_close = (datetime.utcnow() - recent_close.closed_at).total_seconds() / 60
                outcome = "loss" if recent_close.pnl < 0 else "win"
                logger.info(f"[COOLDOWN] {pair}: Recent {outcome} {time_since_close:.1f} mins ago (pnl={recent_close.pnl:.2f}), waiting {cooldown_minutes} mins")
                return None
        
        # Calculate position size
        available = await self.get_available_balance(db)
        open_margin_result = await db.execute(
            select(func.coalesce(func.sum(Order.investment), 0)).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        total_portfolio = available + (open_margin_result.scalar() or 0)

        # === Pattern Cell Ship Rules (May 21, NEW dimension) ===
        # Compute Pattern C + Pattern W matches at entry, look up active rules from
        # pattern_cell_rules config, apply Option C conflict resolution (C presence
        # blocks W multipliers). C-rules contribute fixed TP/SL; W-rules contribute
        # multiplier. Pattern rules take PRIORITY over RSI×ADX multipliers below
        # (when both fire on a single trade, pattern wins — Pattern is more specific).
        _btc_gap_for_pc = globals().get('_current_btc_trend_gap_pct')
        _pc1_e, _pc2_e, _pc3_e, _pc4_e, _pc5_e, _pc6_e, _pc7_e, _pc8_e, _pc9_e, _pc_any_e = _compute_pattern_c_match(
            direction=direction,
            rng_pos=entry_range_position,
            pair_gap=entry_pair_ema20_ema50_gap_pct,
            adx_delta=entry_adx_delta,
            btc_rsi=entry_btc_rsi,
            btc_rsi_prev=entry_btc_rsi_prev,
            btc_adx=entry_btc_adx,
            btc_adx_prev=entry_btc_adx_prev,
            btc_gap=_btc_gap_for_pc,
            stretch=entry_ema5_stretch,
            pair_adx=entry_adx,
            btc_atr=entry_btc_atr_pct,
            ema20_slope=entry_ema20_slope,
            ema50_slope=entry_ema50_slope,
        )
        _pw1_e, _pw2_e, _pw3_e, _pw4_e, _pw5_e, _pw6_e, _pw_any_e = _compute_pattern_w_match(
            direction=direction,
            rsi=entry_rsi,
            adx=entry_adx,
            adx_delta=entry_adx_delta,
            stretch=entry_ema5_stretch,
            rng_pos=entry_range_position,
            pair_gap=entry_pair_ema20_ema50_gap_pct,
            btc_rsi=entry_btc_rsi,
            btc_adx=entry_btc_adx,
            btc_atr=entry_btc_atr_pct,
            btc_gap=_btc_gap_for_pc,
            pair_vol_ratio=entry_pair_volume_ratio,
        )
        # Jun 9: "keep only unmatched longs" — the LONG pattern library selects for losers
        # (every C/W pattern net-negative); the edge is the no-pattern runner cohort (85% WR).
        # Block any LONG that matches ANY C or W pattern. Counter LONG_UNMATCHED_ONLY.
        if direction == "LONG" and not flip_source and getattr(config.trading_config.thresholds, 'long_unmatched_only', False) and (_pc_any_e or _pw_any_e):
            logger.info(f"[LONG_UNMATCHED_ONLY] {pair}: LONG blocked — matched a pattern (c_any={_pc_any_e}, w_any={_pw_any_e})")
            try:
                self._record_filter_block("LONG_UNMATCHED_ONLY", "LONG")
            except Exception:
                pass
            # Jun 13: phantom flip — matched longs are countertrend/exhaustion signatures
            # (C7 dead-cat bounce, W6 top) that fail as longs → fade to SHORT. Strongest
            # flip candidate (historical N=271, +0.142pp/trade proxy; C7 sub-cell +0.259).
            # Measures REALIZED matched-long→short P&L. Blocked dir LONG → flip SHORT.
            # Jun 14: tag the C/W family so the fade can be sub-divided (C+W / C / W).
            _um_cohort = "C+W" if (_pc_any_e and _pw_any_e) else ("C" if _pc_any_e else "W")
            # Jun 15: forward THIS blocked-long's full entry context (open_position's own
            # derived params) so BOTH the phantom row AND the SHORT fade Order carry the same
            # analytics columns as a normal trade — ATR, fan-ratio gaps, stretch, range-pos,
            # dist-EMA13, BTC fields. Built once, shared by the seed + the live flip below.
            _um_ef = {k: v for k, v in {
                'entry_gap': entry_gap,
                'entry_ema_gap_5_8': entry_ema_gap_5_8, 'entry_ema_gap_8_13': entry_ema_gap_8_13,
                'entry_ema5_stretch': entry_ema5_stretch, 'entry_price_vs_ema5_pct': entry_price_vs_ema5_pct,
                'entry_rsi': entry_rsi, 'entry_rsi_prev': entry_rsi_prev,
                'entry_adx': entry_adx, 'entry_adx_prev': entry_adx_prev, 'entry_adx_delta': entry_adx_delta,
                'entry_pos_di': entry_pos_di, 'entry_neg_di': entry_neg_di, 'entry_atr_pct': entry_atr_pct,
                'entry_range_position': entry_range_position, 'entry_dist_from_ema13_pct': entry_dist_from_ema13_pct,
                'entry_pair_ema20_ema50_gap_pct': entry_pair_ema20_ema50_gap_pct,
                'entry_ema20_slope': entry_ema20_slope, 'entry_ema50_slope': entry_ema50_slope,
                'entry_btc_adx': entry_btc_adx, 'entry_btc_rsi': entry_btc_rsi,
                'entry_btc_ema20_slope': entry_btc_ema20_slope,
                'entry_btc_1h_slope': entry_btc_1h_slope, 'entry_btc_dist_from_ema13_pct': entry_btc_dist_from_ema13_pct,
                # Jun 15 — BTC prev/higher-TF companions (parity with the FAN flip path) so the
                # "by BTC ... Direction / Volatility / 1h RSI" tables also see LONG_UNMATCHED flips.
                'entry_btc_adx_prev': entry_btc_adx_prev, 'entry_btc_rsi_prev': entry_btc_rsi_prev,
                'entry_btc_rsi_prev6': entry_btc_rsi_prev6, 'entry_btc_atr_pct': entry_btc_atr_pct,
                'entry_btc_rsi_1h': entry_btc_rsi_1h, 'entry_btc_rsi_1h_prev': entry_btc_rsi_1h_prev,
                'entry_global_volume_ratio': entry_global_volume_ratio, 'entry_pair_volume_ratio': entry_pair_volume_ratio,
                'entry_bull_pct': entry_bull_pct, 'entry_bear_pct': entry_bear_pct,
                'entry_pair_volume_24h_usd': entry_pair_volume_24h_usd, 'entry_pair_rank': entry_pair_rank,
                'entry_quality_score': entry_quality_score,
            }.items() if v is not None}
            _seed_phantom_flip(pair, current_price, "LONG", "LONG_UNMATCHED_ONLY", cohort=_um_cohort, entry_fields=_um_ef)
            # Jun 15: LIVE flip — fade the matched long to a SHORT (registry-gated). This
            # block runs INSIDE open_position, so the flip opens in an ISOLATED session
            # (isolate=True) to keep the outer (blocked-long) transaction clean. The flip
            # SHORT can't re-enter this block (it's direction=="LONG" + not flip_source).
            await self._maybe_open_flip(
                db, pair, "LONG", "LONG_UNMATCHED_ONLY",
                {'price': current_price, 'rsi': entry_rsi, 'adx': entry_adx, 'atr_pct': entry_atr_pct},
                isolate=True, entry_fields=_um_ef,
            )
            return None
        _pcell_inv, _pcell_lev, _pcell_src, _pcell_fixed_tp, _pcell_fixed_sl, _pcell_block = self._lookup_pattern_cell_rule(
            direction=direction,
            c_flags={'C1': _pc1_e, 'C2': _pc2_e, 'C3': _pc3_e, 'C4': _pc4_e, 'C5': _pc5_e,
                     'C6': _pc6_e, 'C7': _pc7_e, 'C8': _pc8_e, 'C9': _pc9_e},
            w_flags={'W1': _pw1_e, 'W2': _pw2_e, 'W3': _pw3_e, 'W4': _pw4_e, 'W5': _pw5_e, 'W6': _pw6_e},
        )
        # Jun 8: pattern-cell BLOCK action — skip the entry entirely (no order, no exchange
        # call; we're before position sizing / Order creation). Counter PATTERN_CELL_BLOCK.
        if _pcell_block and not flip_source:
            logger.info(f"[PATTERN_CELL_BLOCK] {pair} {direction}: entry blocked by pattern-cell rule (signature={_pcell_src})")
            try:
                self._record_filter_block("PATTERN_CELL_BLOCK", direction)
            except Exception:
                pass
            return None

        # === Premium Multiplier (May 4, 2026 — Phase 3 Position Multiplier per CLAUDE.md May 3) ===
        # Look up cell multiplier from BOTH pair-level (Pair RSI × Pair ADX) and BTC-level
        # (BTC RSI × BTC ADX) rule strings.  When both match, take HIGHER (max) — not multiply
        # — to prevent compounding past the hard cap.  Hard cap is UI-configurable, default 2.0×.
        # Capital cap fallback: if cell-multiplied investment exceeds tradeable balance,
        # the existing min(investment, tradeable) inside calculate_position_size invests
        # all available; capped_by_balance flag tells us so we can log + persist.
        _th = config.trading_config.thresholds
        _mult_target = getattr(_th, 'rsi_adx_multiplier_target', 'investment')
        _inv_cap = getattr(_th, 'rsi_adx_multiplier_hard_cap', 2.0)
        _lev_cap = getattr(_th, 'rsi_adx_multiplier_lev_hard_cap', 2.0)
        _pair_rules = getattr(_th, f'rsi_adx_multiplier_{direction.lower()}', '')
        _btc_rules = getattr(_th, f'btc_rsi_adx_multiplier_{direction.lower()}', '')
        _pair_inv, _pair_lev, _pair_src = self._lookup_rsi_adx_multiplier(entry_rsi, entry_adx, _pair_rules, 'PAIR')
        _btc_inv, _btc_lev, _btc_src = self._lookup_rsi_adx_multiplier(entry_btc_rsi, entry_btc_adx, _btc_rules, 'BTC')
        # Extension multiplier (May 24) — Pair Distance from EMA13 dimension.
        _ext_inv, _ext_lev, _ext_src = self._lookup_extension_multiplier(
            direction,
            entry_dist_from_ema13_pct,
            entry_pair_volume_ratio,
            entry_adx_delta,
        )
        # BTC 1h Slope × BTC ADX multiplier (May 24 evening) — NEW dimension.
        _btc1h_inv, _btc1h_lev, _btc1h_src = self._lookup_btc_1h_slope_btc_adx_multiplier(
            direction,
            _current_btc_1h_slope,
            entry_btc_adx,
        )

        # Conflict resolution (May 21 — extended for "both" mode):
        # When pair-level AND BTC-level cells both match, the HIGHER candidate wins.
        # "Higher" is measured by the metric that ACTUALLY affects the position under
        # current target mode:
        #   "investment" → compare inv_mult alone (lev ignored)
        #   "leverage"   → compare lev_mult alone
        #   "both"       → compare effective notional product (inv × lev)
        # This way the winning cell is the one producing the largest actual position
        # effect under the current mode, not an abstract sum.
        # Stretch-based multiplier retired May 15 PM (was a candidate).
        # Score-based multiplier retired May 21 (cross-batch no-edge / decay).
        def _score_candidate(inv, lev):
            if _mult_target == "leverage":
                return lev
            if _mult_target == "both":
                return inv * lev
            return inv  # "investment" mode (default)

        # Pattern Cell rule takes PRIORITY over RSI×ADX (May 21 — Pattern is more specific).
        # If a pattern rule fires (ANY pcell_src — including baseline 1.0× defensive cells
        # like C4 LONG and UNMATCHED), it BLOCKS all other dimensional multipliers
        # (RSI×ADX pair/BTC, Extension, BTC 1h Slope×ADX). The pattern match IS the
        # conviction signal — co-firing EXT/RSI×ADX boost on a known-loser-shape signature
        # is structurally wrong (CLAUDE.md May 26 RENDERUSDT bug: C4 LONG matched but
        # EXT_Ext0.4-0.6_L still fired at 2.0×, doubling -$87 loss into -$173).
        if _pcell_src is not None:
            cell_mult, cell_lev_mult, cell_src = _pcell_inv, _pcell_lev, _pcell_src
            logger.info(f"[PATTERN_CELL] {pair} {direction}: rule fired ({_pcell_src}) inv={_pcell_inv}x lev={_pcell_lev}x — overrides RSI×ADX/EXT/BTC1H")
        else:
            _candidates = [
                (_pair_inv, _pair_lev, _pair_src),
                (_btc_inv, _btc_lev, _btc_src),
                (_ext_inv, _ext_lev, _ext_src),
                (_btc1h_inv, _btc1h_lev, _btc1h_src),
            ]
            _winner = max(_candidates, key=lambda c: _score_candidate(c[0], c[1]))
            cell_mult, cell_lev_mult, cell_src = _winner

        # Hard cap clamps — applied independently to each side.
        # In "both" mode, max effective notional = inv_cap × lev_cap (operator
        # accepts compounding for high-conviction setups; documented in CLAUDE.md).
        if cell_mult > _inv_cap:
            logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested inv={cell_mult}x, hard-capped to inv={_inv_cap}x")
            cell_mult = _inv_cap
        if cell_lev_mult > _lev_cap:
            logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested lev={cell_lev_mult}x, hard-capped to lev={_lev_cap}x")
            cell_lev_mult = _lev_cap
        cell_mult = max(0.5, cell_mult)  # safety floor
        cell_lev_mult = max(0.5, cell_lev_mult)

        # Jun 14: Flip Entry sleeve overrides ALL momentum multipliers — a naked fade
        # sizes at base × registry size_mult × registry lev_mult (no pattern/RSI×ADX boost).
        # Jun 15: registry now carries a per-source leverage multiplier too (3-part format);
        # force multiplier_target="both" so size AND lev apply. Hard caps below still clamp.
        if flip_source:
            cell_mult, cell_lev_mult, cell_src = _flip_size_mult(flip_source), _flip_lev_mult(flip_source), f"FLIP:{flip_source}"
            # Jun 16: per-source CONDITIONAL cell multiplier from _flip_filters (e.g. FAN's
            # strong-bear BTC RSI40-45×ADX≥35 cell @2× size / 1× lev). Multiplies the registry
            # size AND leverage and carries a distinct cell_src so it groups as its OWN row in
            # Multiplier Cell Performance (the flip multiplier table). Hard caps below still clamp.
            if (flip_cell_mult and flip_cell_mult != 1.0) or (flip_cell_lev_mult and flip_cell_lev_mult != 1.0):
                cell_mult = cell_mult * (flip_cell_mult or 1.0)
                cell_lev_mult = cell_lev_mult * (flip_cell_lev_mult or 1.0)
                _tag = f"×{flip_cell_mult:g}" if flip_cell_mult and flip_cell_mult != 1.0 else ""
                _tag += f"L{flip_cell_lev_mult:g}" if flip_cell_lev_mult and flip_cell_lev_mult != 1.0 else ""
                cell_src = f"FLIP:{flip_source}{_tag}"
            _mult_target = "both"
            # Re-apply the hard caps (the clamp above ran before this override).
            if cell_mult > _inv_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested inv={cell_mult}x, hard-capped to inv={_inv_cap}x")
                cell_mult = _inv_cap
            if cell_lev_mult > _lev_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested lev={cell_lev_mult}x, hard-capped to lev={_lev_cap}x")
                cell_lev_mult = _lev_cap
            cell_mult, cell_lev_mult = max(0.5, cell_mult), max(0.5, cell_lev_mult)

        investment, leverage, cell_capped = self.calculate_position_size(
            available, confidence, total_portfolio=total_portfolio,
            cell_multiplier=cell_mult, cell_lev_multiplier=cell_lev_mult,
            multiplier_target=_mult_target,
        )
        if cell_capped:
            logger.info(
                f"[CELL_MULT_CAPPED] {pair} {direction}: target multiplier inv={cell_mult}x lev={cell_lev_mult}x via {cell_src} "
                f"({_mult_target} target), capped by available balance — proceeded at ${investment:.2f}"
            )
        if (cell_mult != 1.0 or cell_lev_mult != 1.0) and not cell_capped:
            logger.info(f"[CELL_MULT] {pair} {direction}: applied inv={cell_mult}x lev={cell_lev_mult}x via {cell_src} ({_mult_target} target)")

        # ── Liquidity-aware sizing caps (Jun 2 — see CLAUDE.md) ──────────────────
        # ① per-pair liquidity cap: throttle this order's NOTIONAL to a small slice
        #    of the pair's 24h volume (slippage protection — the order stays
        #    absorbable). ② gross-notional cap: keep Σ(open notional) under
        #    balance × max_gross_leverage (correlated-dump / liquidation guard).
        # Both operate on NOTIONAL (what hits the book); margin is backed out as
        # notional / leverage. Throttling below min_investment_size → skip the trade.
        _liq_capped = False
        _desired_notional = None  # observability: pre-cap notional (stays None when caps off)
        _liq_cap = None           # observability: ① per-pair cap value (stays None when caps off)
        _inv_cfg = config.trading_config.investment
        _liq_pct = getattr(_inv_cfg, 'max_notional_pct_of_pair_volume', 0.0) or 0.0
        _liq_ceiling = getattr(_inv_cfg, 'max_notional_hard_ceiling', 0.0) or 0.0
        _gross_lev = getattr(_inv_cfg, 'max_gross_leverage', 0.0) or 0.0
        if investment > 0 and leverage > 0 and (_liq_pct > 0 or _liq_ceiling > 0 or _gross_lev > 0):
            _desired_notional = investment * leverage
            _final_notional = _desired_notional
            _cap_reason = None
            # ① per-pair liquidity cap
            _liq_cap = None
            if _liq_pct > 0 and entry_pair_volume_24h_usd and entry_pair_volume_24h_usd > 0:
                _liq_cap = (_liq_pct / 100.0) * entry_pair_volume_24h_usd
            if _liq_ceiling > 0:
                _liq_cap = _liq_ceiling if _liq_cap is None else min(_liq_cap, _liq_ceiling)
            if _liq_cap is not None and _final_notional > _liq_cap:
                _final_notional = _liq_cap
                _cap_reason = 'LIQ'
            # ② gross-notional cap
            if _gross_lev > 0:
                _bal_for_gross = total_portfolio if total_portfolio else available
                try:
                    _gross_q = await db.execute(
                        select(func.coalesce(func.sum(Order.notional_value), 0.0)).where(
                            and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
                        )
                    )
                    _open_notional = float(_gross_q.scalar() or 0.0)
                except Exception:
                    _open_notional = 0.0
                _gross_budget = (_bal_for_gross or 0.0) * _gross_lev
                _gross_room = max(0.0, _gross_budget - _open_notional)
                if _gross_room <= 0:
                    logger.warning(
                        f"[GROSS_CAP] {pair}: open notional ${_open_notional:,.0f} >= budget "
                        f"${_gross_budget:,.0f} (balance ${(_bal_for_gross or 0):,.0f} x {_gross_lev:g}x) — skip"
                    )
                    self._record_filter_block('GROSS_CAP_SKIP', direction)  # Jun 2: surface gross-full rejections in Filter Blocks
                    return None
                if _final_notional > _gross_room:
                    _final_notional = _gross_room
                    _cap_reason = 'GROSS' if _cap_reason is None else 'LIQ+GROSS'
            # apply throttle: shrink margin to fit the capped notional
            if _final_notional < _desired_notional - 0.01:
                _new_investment = _final_notional / leverage
                if _new_investment < _inv_cfg.min_investment_size:
                    logger.warning(
                        f"[LIQ_CAP] {pair} {direction}: {_cap_reason} cap -> ${_new_investment:.2f} margin "
                        f"< min ${_inv_cfg.min_investment_size:.0f} (pair too thin / no gross room) — skip"
                    )
                    self._record_filter_block('LIQ_CAP_SKIP', direction)  # Jun 2: surface liquidity-below-min rejections in Filter Blocks
                    return None
                logger.info(
                    f"[LIQ_CAP] {pair} {direction}: {_cap_reason} notional ${_desired_notional:,.0f}->${_final_notional:,.0f} "
                    f"(investment ${investment:.2f}->${_new_investment:.2f}, lev {leverage}x)"
                )
                investment = _new_investment
                _liq_capped = True

        logger.info(f"[TRADE] {pair}: {direction} {confidence} - Investment: ${investment:.2f}, Leverage: {leverage}x")
        
        if investment <= 0:
            return None
        
        # Calculate notional and quantity
        notional_value = investment * leverage
        quantity = notional_value / current_price
        
        # Determine fee rate and entry type
        tc = config.trading_config
        maker_enabled = tc.maker_entry_enabled
        maker_fee_rate = getattr(tc, 'maker_fee', tc.trading_fee)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)

        entry_order_type = "TAKER"
        entry_fee = notional_value * taker_fee_rate
        
        # Execute trade
        binance_order_id = None
        actual_price = current_price
        
        if not self.is_paper_mode:
            symbol = pair.replace('USDT', '/USDT:USDT')
            side = 'buy' if direction == 'LONG' else 'sell'

            if maker_enabled:
                # --- Maker entry flow ---
                result = await self._try_maker_entry(
                    symbol=symbol, side=side, amount=quantity,
                    leverage=int(leverage), direction=direction, pair=pair,
                    notional_value=notional_value,
                    maker_fee_rate=maker_fee_rate, taker_fee_rate=taker_fee_rate,
                    confidence=confidence,
                )
                if result and result.get('skipped'):
                    # Amendment #7: signal expired during maker wait → record + abort entry.
                    # May 2: forward all entry indicators + wait_seconds so aborted entries
                    # land in Entry Conditions by Outcome with full attribution data.
                    await self._record_signal_expired_order(
                        db=db, pair=pair, direction=direction, confidence=confidence,
                        reason=result.get('reason', 'unknown'),
                        entry_price=current_price,
                        wait_seconds=result.get('wait_seconds'),
                        entry_gap=entry_gap,
                        entry_ema_gap_5_8=entry_ema_gap_5_8,
                        entry_ema_gap_8_13=entry_ema_gap_8_13,
                        entry_ema5_stretch=entry_ema5_stretch,
                        entry_rsi=entry_rsi,
                        entry_rsi_prev=entry_rsi_prev,
                        entry_adx=entry_adx,
                        entry_adx_prev=entry_adx_prev,
                        entry_ema20_slope=entry_ema20_slope,
                        entry_btc_ema20_slope=entry_btc_ema20_slope,
                        entry_btc_adx=entry_btc_adx,
                        entry_btc_adx_prev=entry_btc_adx_prev,
                        entry_btc_rsi=entry_btc_rsi,
                        entry_btc_rsi_prev=entry_btc_rsi_prev,
                        entry_btc_rsi_prev6=entry_btc_rsi_prev6,
                        entry_btc_atr_pct=entry_btc_atr_pct,
                        entry_btc_rsi_1h=entry_btc_rsi_1h,
                        entry_btc_rsi_1h_prev=entry_btc_rsi_1h_prev,
                        entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
                        entry_global_volume_ratio=entry_global_volume_ratio,
                        entry_pair_volume_ratio=entry_pair_volume_ratio,
                        entry_bull_pct=entry_bull_pct,
                        entry_bear_pct=entry_bear_pct,
                        entry_range_position=entry_range_position,
                        entry_adx_delta=entry_adx_delta,
                        entry_quality_score=entry_quality_score,
                        entry_btc_regime=entry_btc_regime,
                        entry_btc_trend_gap_pct=globals().get('_current_btc_trend_gap_pct'),
                        entry_pos_di=entry_pos_di,
                        entry_neg_di=entry_neg_di,
                        entry_atr_pct=entry_atr_pct,
                        entry_ema50_slope=entry_ema50_slope,
                        entry_funding_rate=entry_funding_rate,
                        entry_pair_ema20_ema50_gap_pct=entry_pair_ema20_ema50_gap_pct,
                        entry_dist_from_ema13_pct=entry_dist_from_ema13_pct,
                        entry_btc_dist_from_ema13_pct=entry_btc_dist_from_ema13_pct,
                        entry_btc_1h_slope=entry_btc_1h_slope,
                    )
                    return None
                if result:
                    binance_order_id = result['id']
                    actual_price = result['price']
                    entry_fee = result['entry_fee']
                    entry_order_type = result['entry_order_type']
                    quantity = result.get('amount', quantity)
                else:
                    logger.error(f"[MAKER_ENTRY] {pair}: Both maker and fallback failed")
                    return None
            else:
                result = await binance_service.create_market_order(
                    symbol=symbol, side=side, amount=quantity, leverage=int(leverage)
                )
                if result:
                    binance_order_id = result['id']
                    actual_price = result['price']
                    quantity = result.get('amount', quantity)
                    entry_fee = actual_price * quantity * taker_fee_rate
                    entry_order_type = "TAKER"
                else:
                    logger.error(f"[TRADE] {pair}: Market order failed (leverage mismatch or Binance error)")
                    return None
        else:
            # Paper trade -- simulate maker fill if enabled
            if maker_enabled:
                result = await self._simulate_maker_entry_paper(
                    pair=pair, direction=direction, current_price=current_price,
                    notional_value=notional_value,
                    maker_fee_rate=maker_fee_rate, taker_fee_rate=taker_fee_rate,
                    confidence=confidence,
                )
                if result.get('skipped'):
                    # Amendment #7: signal expired during maker wait → record + abort entry.
                    # May 2: forward all entry indicators + wait_seconds so aborted entries
                    # land in Entry Conditions by Outcome with full attribution data.
                    await self._record_signal_expired_order(
                        db=db, pair=pair, direction=direction, confidence=confidence,
                        reason=result.get('reason', 'unknown'),
                        entry_price=current_price,
                        wait_seconds=result.get('wait_seconds'),
                        entry_gap=entry_gap,
                        entry_ema_gap_5_8=entry_ema_gap_5_8,
                        entry_ema_gap_8_13=entry_ema_gap_8_13,
                        entry_ema5_stretch=entry_ema5_stretch,
                        entry_rsi=entry_rsi,
                        entry_rsi_prev=entry_rsi_prev,
                        entry_adx=entry_adx,
                        entry_adx_prev=entry_adx_prev,
                        entry_ema20_slope=entry_ema20_slope,
                        entry_btc_ema20_slope=entry_btc_ema20_slope,
                        entry_btc_adx=entry_btc_adx,
                        entry_btc_adx_prev=entry_btc_adx_prev,
                        entry_btc_rsi=entry_btc_rsi,
                        entry_btc_rsi_prev=entry_btc_rsi_prev,
                        entry_btc_rsi_prev6=entry_btc_rsi_prev6,
                        entry_btc_atr_pct=entry_btc_atr_pct,
                        entry_btc_rsi_1h=entry_btc_rsi_1h,
                        entry_btc_rsi_1h_prev=entry_btc_rsi_1h_prev,
                        entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
                        entry_global_volume_ratio=entry_global_volume_ratio,
                        entry_pair_volume_ratio=entry_pair_volume_ratio,
                        entry_bull_pct=entry_bull_pct,
                        entry_bear_pct=entry_bear_pct,
                        entry_range_position=entry_range_position,
                        entry_adx_delta=entry_adx_delta,
                        entry_quality_score=entry_quality_score,
                        entry_btc_regime=entry_btc_regime,
                        entry_btc_trend_gap_pct=globals().get('_current_btc_trend_gap_pct'),
                        entry_pos_di=entry_pos_di,
                        entry_neg_di=entry_neg_di,
                        entry_atr_pct=entry_atr_pct,
                        entry_ema50_slope=entry_ema50_slope,
                        entry_funding_rate=entry_funding_rate,
                        entry_pair_ema20_ema50_gap_pct=entry_pair_ema20_ema50_gap_pct,
                        entry_dist_from_ema13_pct=entry_dist_from_ema13_pct,
                        entry_btc_dist_from_ema13_pct=entry_btc_dist_from_ema13_pct,
                        entry_btc_1h_slope=entry_btc_1h_slope,
                    )
                    return None
                actual_price = result['price']
                entry_fee = result['entry_fee']
                entry_order_type = result['entry_order_type']
                quantity = notional_value / actual_price
            else:
                entry_order_type = "TAKER"
        
        # Pattern C tracker (May 19, 2026 — observation-only signature flags)
        # Reuse the values already computed above for Pattern Cell rule lookup
        # (deduplicating the helper call; same inputs would produce identical values).
        _pc1_m, _pc2_m, _pc3_m, _pc4_m, _pc5_m, _pc6_m, _pc7_m, _pc8_m, _pc9_m, _pc_any_m = (
            _pc1_e, _pc2_e, _pc3_e, _pc4_e, _pc5_e, _pc6_e, _pc7_e, _pc8_e, _pc9_e, _pc_any_e
        )
        # Pattern W tracker (May 21, 2026 — observation-only signature flags, now ALSO at entry)
        _pw1_m, _pw2_m, _pw3_m, _pw4_m, _pw5_m, _pw6_m, _pw_any_m = (
            _pw1_e, _pw2_e, _pw3_e, _pw4_e, _pw5_e, _pw6_e, _pw_any_e
        )
        # Jun 2: entry-fill slippage (signed, positive = filled WORSE than the decision price).
        # ~0 in paper (sim fills at signal price); meaningful live. Gives ① a slippage verdict.
        _entry_slippage_pct = None
        try:
            if current_price and current_price > 0 and actual_price and actual_price > 0:
                if direction == "LONG":
                    _entry_slippage_pct = round((actual_price - current_price) / current_price * 100, 4)
                else:
                    _entry_slippage_pct = round((current_price - actual_price) / current_price * 100, 4)
        except Exception:
            _entry_slippage_pct = None

        # Create order record
        order = Order(
            binance_order_id=binance_order_id,
            pair=pair,
            direction=direction,
            status="OPEN",
            entry_price=actual_price,
            current_price=actual_price,
            investment=investment,
            leverage=leverage,
            notional_value=notional_value,
            quantity=quantity,
            confidence=confidence,
            entry_gap=entry_gap,
            entry_ema_gap_5_8=entry_ema_gap_5_8,
            entry_ema_gap_8_13=entry_ema_gap_8_13,
            entry_ema5_stretch=entry_ema5_stretch,
            entry_rsi=entry_rsi,
            entry_rsi_prev=entry_rsi_prev,
            entry_adx=entry_adx,
            entry_adx_prev=entry_adx_prev,
            entry_macro_trend=entry_macro_trend,
            entry_ema20_slope=entry_ema20_slope,
            entry_btc_ema20_slope=entry_btc_ema20_slope,
            entry_btc_adx=entry_btc_adx,
            entry_btc_adx_prev=entry_btc_adx_prev,
            entry_btc_rsi=entry_btc_rsi,
            entry_btc_rsi_prev=entry_btc_rsi_prev,
            entry_btc_rsi_prev6=entry_btc_rsi_prev6,
            entry_btc_atr_pct=entry_btc_atr_pct,
            entry_btc_rsi_1h=entry_btc_rsi_1h,
            entry_btc_rsi_1h_prev=entry_btc_rsi_1h_prev,
            entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
            entry_global_volume_ratio=entry_global_volume_ratio,
            entry_pair_volume_ratio=entry_pair_volume_ratio,
            entry_bull_pct=entry_bull_pct,
            entry_bear_pct=entry_bear_pct,
            entry_range_position=entry_range_position,
            entry_adx_delta=entry_adx_delta,
            entry_quality_score=entry_quality_score,
            entry_btc_regime=entry_btc_regime,
            entry_btc_trend_gap_pct=globals().get('_current_btc_trend_gap_pct'),
            exit_btc_regime=entry_btc_regime,  # Initialize to entry; updated on close
            # Exploration Analytics (Apr 28, observation-only)
            entry_pos_di=entry_pos_di,
            entry_neg_di=entry_neg_di,
            entry_atr_pct=entry_atr_pct,
            entry_ema50_slope=entry_ema50_slope,
            entry_funding_rate=entry_funding_rate,
            entry_pair_ema20_ema50_gap_pct=entry_pair_ema20_ema50_gap_pct,
            entry_dist_from_ema13_pct=entry_dist_from_ema13_pct,
            entry_btc_dist_from_ema13_pct=entry_btc_dist_from_ema13_pct,
            entry_btc_1h_slope=entry_btc_1h_slope,
            # May 10: absolute pair 24h USD volume at entry (size-bucket analytics)
            entry_pair_volume_24h_usd=entry_pair_volume_24h_usd,
            entry_pair_rank=entry_pair_rank,
            # Jun 8: gap-expanding relaxation A/B cohort tag
            entry_gap_expand_marginal=entry_gap_expand_marginal,
            # Jun 2: liquidity-aware sizing observability (final notional = notional_value above)
            entry_desired_notional=_desired_notional,
            entry_liquidity_cap_notional=_liq_cap,
            liquidity_capped=_liq_capped,
            entry_slippage_pct=_entry_slippage_pct,
            entry_fee=entry_fee,
            entry_order_type=entry_order_type,
            peak_pnl=0.0,
            trough_pnl=0.0,
            high_price_since_entry=actual_price if direction == "LONG" else None,
            low_price_since_entry=actual_price if direction == "SHORT" else None,
            is_paper=self.is_paper_mode,
            # Premium Multiplier (May 4, 2026 → extended May 21) — track which RSI×ADX cell rule fired.
            # cell_multiplier = INVESTMENT-side multiplier; cell_lev_multiplier = LEVERAGE-side (May 21).
            cell_multiplier=cell_mult,
            cell_lev_multiplier=cell_lev_mult,
            cell_multiplier_source=cell_src,
            cell_multiplier_capped=cell_capped,
            # Jun 14: Flip Entry sleeve strategy tag (segregates flip P&L from momentum)
            entry_strategy=(f"FLIP:{flip_source}" if flip_source else "MOMENTUM"),
            # Initialize dynamic TP tracking
            current_tp_level=1,
            dynamic_tp_target=conf_config.tp_min,
            # Pattern C tracker flags (May 19, observation-only)
            entry_pattern_c1_match=_pc1_m,
            entry_pattern_c2_match=_pc2_m,
            entry_pattern_c3_match=_pc3_m,
            entry_pattern_c4_match=_pc4_m,
            entry_pattern_c5_match=_pc5_m,
            entry_pattern_c6_match=_pc6_m,
            entry_pattern_c7_match=_pc7_m,
            entry_pattern_c8_match=_pc8_m,
            entry_pattern_c9_match=_pc9_m,
            entry_pattern_c_any_match=_pc_any_m,
            # Pattern W tracker flags (May 21 — lifted to entry for live multiplier ship)
            entry_pattern_w1_match=_pw1_m,
            entry_pattern_w2_match=_pw2_m,
            entry_pattern_w3_match=_pw3_m,
            entry_pattern_w4_match=_pw4_m,
            entry_pattern_w5_match=_pw5_m,
            entry_pattern_w6_match=_pw6_m,
            entry_pattern_w_any_match=_pw_any_m,
            # Pattern Cell Ship rule attribution (May 21)
            pattern_cell_source=_pcell_src,
            pattern_fixed_tp_pct=_pcell_fixed_tp,
            pattern_fixed_sl_pct=_pcell_fixed_sl,
        )
        db.add(order)
        await db.flush()  # Flush to get the order ID
        
        # Create transaction record
        transaction = Transaction(
            order_id=order.id,
            binance_order_id=binance_order_id,
            pair=pair,
            action=f"OPEN_{direction}",
            price=actual_price,
            quantity=quantity,
            investment=investment,
            leverage=leverage,
            notional_value=notional_value,
            fee=entry_fee,
            order_type="MAKER" if entry_order_type == "MAKER" else "TAKER",
            is_paper=self.is_paper_mode
        )
        db.add(transaction)

        await db.commit()
        await db.refresh(order)

        # Jun 2: count a redeploy-band open (position beyond normal max_open_positions,
        # only reachable because ① throttling freed margin + redeploy raised the ceiling).
        if _is_redeploy_open:
            self._record_filter_block('REDEPLOY_OPEN', direction)

        # Broker-side protective stops feature REMOVED Apr 17 after 4 failed
        # hotfix attempts — Binance repeatedly rejected with -4120 "Order type
        # not supported for this endpoint" on the standard /fapi/v1/order for
        # this account/CCXT combo, and the Portfolio Margin routing path
        # returned -2015 because the account is not PM-enrolled.  Root cause
        # remains unidentified.  Bot relies exclusively on internal in-process
        # exits (SL, trailing, FL2, FL_EMERGENCY_SL, regime_change_exit) for
        # risk management.  See CLAUDE.md "Broker-side Protective Stops
        # removal" section for the forensic trail and what to investigate
        # before any future attempt.

        # Recalculate paper balance from DB (source of truth) and save
        if self.is_paper_mode:
            pre_usdt = self.paper_balance
            pre_bnb = self.paper_bnb_balance_usd

            await self._recalculate_paper_balance(db)
            await self._deduct_fee_from_bnb(entry_fee, db)
            await self.save_state(db)

            _snap = await db.execute(
                select(func.coalesce(func.sum(Order.investment), 0)).where(
                    and_(Order.status == "OPEN", Order.is_paper == True)
                )
            )
            post_margin = _snap.scalar() or 0
            pre_margin = post_margin - investment
            pre_total = pre_usdt + pre_margin + pre_bnb
            post_total = self.paper_balance + post_margin + self.paper_bnb_balance_usd
            delta = post_total - pre_total
            logger.info(
                f"[PORTFOLIO_OPEN] {pair} {direction} | "
                f"Investment={investment:.2f} EntryFee={entry_fee:.4f} | "
                f"PRE: USDT={pre_usdt:.2f} Margin={pre_margin:.2f} "
                f"BNB={pre_bnb:.2f} Total={pre_total:.2f} | "
                f"POST: USDT={self.paper_balance:.2f} Margin={post_margin:.2f} "
                f"BNB={self.paper_bnb_balance_usd:.2f} Total={post_total:.2f} | "
                f"Delta={delta:+.2f} (expected: -{entry_fee:.4f})"
            )
        elif not self.is_paper_mode:
            try:
                bal = await binance_service.get_balance()
                bnb_price = await binance_service.get_bnb_price()
                bnb_usd = bal['bnb_total'] * bnb_price if bnb_price > 0 else 0
                total = bal['usdt_total'] + bnb_usd
                logger.info(
                    f"[PORTFOLIO_OPEN] {pair} {direction} | "
                    f"Investment={investment:.2f} EntryFee={entry_fee:.4f} | "
                    f"USDT_total={bal['usdt_total']:.2f} USDT_free={bal['usdt_free']:.2f} "
                    f"BNB={bal['bnb_total']:.6f} BNB_price={bnb_price:.2f} BNB_usd={bnb_usd:.2f} | "
                    f"Total={total:.2f}"
                )
            except Exception as e:
                logger.warning(f"[PORTFOLIO_OPEN] Failed to log live balance: {e}")

        # Force reset WebSocket tracking for new order (fresh start from entry price)
        # This ensures we track high/low from the actual entry, not from previous orders
        websocket_tracker.force_reset_tracking(pair, actual_price)
        await websocket_tracker.subscribe_pair(pair, actual_price)
        
        # Fetch current EMA5/13/20 data so the WebSocket tick loop can capture
        # peak EMA5 metrics + price-vs-EMA cross shadow (May 6 Phase 1) immediately
        # before update_orders_cache runs.
        _pair_data_row = await db.execute(
            select(PairData.ema5, PairData.ema5_prev3, PairData.ema8, PairData.ema13, PairData.ema20).where(PairData.pair == pair)
        )
        _pair_data = _pair_data_row.first()
        _cached_ema5 = _pair_data.ema5 if _pair_data else None
        _cached_ema5_prev3 = _pair_data.ema5_prev3 if _pair_data else None
        _cached_ema8 = _pair_data.ema8 if _pair_data else None
        _cached_ema13 = _pair_data.ema13 if _pair_data else None
        _cached_ema20 = _pair_data.ema20 if _pair_data else None

        # Immediately add to real-time cache so the WebSocket SL callback can
        # protect this order right away (without waiting for update_orders_cache).
        async with _cache_lock:
            order_cache_entry = {
                'id': order.id,
                'direction': direction,
                'entry_strategy': (f"FLIP:{flip_source}" if flip_source else "MOMENTUM"),  # Jun 15: flips now exit via the realtime stack (entry_strategy gates _is_flip)
                'entry_ema5_stretch': entry_ema5_stretch,  # LEASH SHADOW (May 30) — stretch-exit entry anchor
                'entry_price': actual_price,
                'quantity': quantity,
                'entry_fee': entry_fee,
                'confidence': confidence,
                'stop_loss': conf_config.stop_loss,
                'current_tp_level': 1,
                'peak_pnl': 0.0,
                'trough_pnl': 0.0,
                # May 17: post-arm-min tracking (BE-floor counterfactual support).
                # Set to True the first time peak_pnl crosses be_level1_trigger.
                # post_arm_min_pnl tracks the running minimum of pnl_pct from that
                # moment until close. Captures pre-global-peak dips after BE armed.
                'be_armed': False,
                'post_arm_min_pnl': None,
                'post_arm_min_at': None,
                'be_levels_enabled': getattr(conf_config, 'be_levels_enabled', True),
                'be_level1_trigger': conf_config.be_level1_trigger,
                'be_level1_offset': conf_config.be_level1_offset,
                'be_level2_trigger': conf_config.be_level2_trigger,
                'be_level2_offset': conf_config.be_level2_offset,
                'be_level3_trigger': conf_config.be_level3_trigger,
                'be_level3_offset': conf_config.be_level3_offset,
                'be_level4_trigger': conf_config.be_level4_trigger,
                'be_level4_offset': conf_config.be_level4_offset,
                'be_level5_trigger': conf_config.be_level5_trigger,
                'be_level5_offset': conf_config.be_level5_offset,
                'high_price': actual_price,
                'low_price': actual_price,
                'pullback_trigger': conf_config.pullback_trigger,
                'tp_trailing_enabled': conf_config.tp_trailing_enabled,
                'entry_atr_pct': entry_atr_pct,  # May 7 Phase 1: ATR-normalized trailing in realtime path
                'tp_min': conf_config.tp_min,    # May 7 Phase 2: needed for early-arm zone check
                'cached_ema5': _cached_ema5,
                'cached_ema5_prev3': _cached_ema5_prev3,
                'cached_ema8': _cached_ema8,
                'cached_ema13': _cached_ema13,
                'cached_ema20': _cached_ema20,
                # Phase 1 shadow tracking — counterfactual exit at price-vs-EMA cross.
                # Brand-new order: no prior crosses recorded.
                'first_cross_ema13_at': None,
                'first_cross_ema13_pnl_pct': None,
                'confirmed_cross_ema13_at': None,
                'confirmed_cross_ema13_pnl_pct': None,
                'first_cross_ema20_at': None,
                'first_cross_ema20_pnl_pct': None,
                'confirmed_cross_ema20_at': None,
                'confirmed_cross_ema20_pnl_pct': None,
                'pending_cross_ema13_started_at': None,
                'pending_cross_ema20_started_at': None,
                'peak_ema5_dist_pct': None,
                'peak_ema5_slope_pct': None,
                'peak_reached_at': None,
                'trough_reached_at': None,
                'trough_ema5_dist_pct': None,
                'ema5_ever_negative': False,
                'signal_lost_flagged': False,
                'signal_lost_flag_pnl': None,
                'signal_lost_flagged_at': None,
                'tick_prices': [],
                'phantom_be_l1_triggered': False,
                'phantom_be_l1_triggered_at': None,
                'phantom_be_l1_would_exit_pnl': None,
                'phantom_be_l2_triggered': False,
                'phantom_be_l2_triggered_at': None,
                'phantom_be_l2_would_exit_pnl': None,
                # May 14 — Aggressive phantom BE @ 0.20/0.10 (observation-only)
                'phantom_be_aggr_triggered': False,
                'phantom_be_aggr_triggered_at': None,
                'phantom_be_aggr_would_exit_pnl': None,
                'phantom_regime_change_triggered': False,
                'phantom_regime_change_exit_triggered_at': None,
                'phantom_regime_change_exit_pnl': None,
                'phantom_tick_a_triggered': False,
                'phantom_tick_a_triggered_at': None,
                'phantom_tick_a_pnl': None,
                'phantom_tick_b_triggered': False,
                'phantom_tick_b_triggered_at': None,
                'phantom_tick_b_pnl': None,
                'phantom_tick_c_triggered': False,
                'phantom_tick_c_triggered_at': None,
                'phantom_tick_c_pnl': None,
                'phantom_tick_d_triggered': False,
                'phantom_tick_d_triggered_at': None,
                'phantom_tick_d_pnl': None,
                'phantom_tick_e_triggered': False,
                'phantom_tick_e_triggered_at': None,
                'phantom_tick_e_pnl': None,
                'phantom_tick_f_triggered': False,
                'phantom_tick_f_triggered_at': None,
                'phantom_tick_f_pnl': None,
                'phantom_tick_g_triggered': False,
                'phantom_tick_g_triggered_at': None,
                'phantom_tick_g_pnl': None,
                'regime_neutral_hit': False,
                'regime_neutral_hit_at': None,
                'regime_neutral_pnl': None,
                'regime_comeback_at': None,
                'regime_comeback_pnl': None,
                'regime_opposite_at': None,
                'regime_opposite_pnl': None,
                # Pattern Cell Ship rule overrides (May 21)
                'pattern_cell_source': _pcell_src,
                'pattern_fixed_tp_pct': _pcell_fixed_tp,
                'pattern_fixed_sl_pct': _pcell_fixed_sl,
            }
            if pair not in _open_orders_cache:
                _open_orders_cache[pair] = []
            _open_orders_cache[pair].append(order_cache_entry)
        
        logger.info(f"[ORDER CREATED] {pair}: {direction} {confidence} - ID={order.id}, Investment=${investment:.2f}")

        # Jun 3: update the BTC-acceleration-chase reference on every LONG that actually
        # opens (blocked LONGs never reach here, so the reference stays the last REAL
        # entry). Stores the same global the filter reads, for an apples-to-apples compare.
        if direction == "LONG":
            self._last_long_open_ts = datetime.utcnow()
            self._last_long_open_btc_ema20_slope = _btc_ema20_slope_pct

        return order
    
    async def close_position(
        self,
        db: AsyncSession,
        order: Order,
        current_price: float,
        reason: str = "MANUAL"
    ) -> Optional[Order]:
        """Close an existing position"""
        # Jun 14: Flip Entry sleeve — flips use the SAME exit stack as normal trades
        # (only EMA13-cross is disabled for them), but every exit reason is FLIP_-prefixed
        # here (the single close funnel) so flip exits are distinguishable everywhere:
        # FLIP_STOP_LOSS L1 / FLIP_TRAILING_STOP L1 / FLIP_RUNNER_TRAIL / etc. Report +
        # post-exit-whitelist matchers strip the FLIP_ prefix to recover the base reason.
        if reason and (order.entry_strategy or "").startswith("FLIP:") and not reason.startswith("FLIP_"):
            reason = "FLIP_" + reason
        async with _close_lock:
            return await self._close_position_locked(db, order, current_price, reason)

    async def _mark_close_in_progress(self, db: AsyncSession, order_id: int) -> bool:
        """Publish intent-to-close for this order so the monitor reconciler can
        tell an in-flight bot close apart from a truly external close.

        Writes closing_in_progress=True + close_initiated_at=NOW() and commits
        immediately (separate transaction from the later status=CLOSED commit)
        so the reconciler — which runs in its own AsyncSession — can observe
        the flag.  Fails open: if the flag commit fails after retries the
        close proceeds anyway.  Without the flag the reconciler race is still
        bounded by the existing duplicate-close guard and SELECT ... WHERE
        status='OPEN' filter, just not race-free.

        Returns True on successful commit, False otherwise.
        """
        # 5 attempts with short progressive backoff: 0.1, 0.2, 0.3, 0.4s.
        # Total worst case ~1s added to the close path under heavy SQLite
        # contention — acceptable given the protection it provides.
        for attempt in range(1, 6):
            try:
                await db.execute(
                    update(Order)
                    .where(and_(Order.id == order_id, Order.status == "OPEN"))
                    .values(
                        closing_in_progress=True,
                        close_initiated_at=datetime.utcnow(),
                    )
                )
                await db.commit()
                return True
            except Exception as _e:
                try:
                    await db.rollback()
                except Exception:
                    pass
                if attempt < 5:
                    await asyncio.sleep(0.1 * attempt)
                else:
                    logger.warning(
                        f"[CLOSE_INTENT_FAIL] order_id={order_id}: could not publish "
                        f"close-intent after {attempt} attempts ({str(_e)[:80]}); "
                        f"proceeding with close — reconciler race guard disabled for this close"
                    )
                    return False
        return False

    async def _close_position_locked(
        self,
        db: AsyncSession,
        order: Order,
        current_price: float,
        reason: str = "MANUAL"
    ) -> Optional[Order]:
        """Internal close logic, must be called under _close_lock."""
        if order.status != "OPEN":
            return None
        
        # Re-verify from DB to prevent race between polling loop and real-time monitor
        fresh_check = await db.execute(
            select(Order.status).where(Order.id == order.id)
        )
        db_status = fresh_check.scalar_one_or_none()
        if db_status != "OPEN":
            logger.warning(f"[CLOSE_RACE_PREVENTED] {order.pair}: Order {order.id} already {db_status}, skipping duplicate close (reason={reason})")
            return None
        
        # CRITICAL: Never close with invalid price - this would cause -100% P&L
        if current_price is None or current_price <= 0:
            logger.error(f"[CLOSE_BLOCKED] {order.pair}: Attempted to close with invalid price={current_price}, reason={reason}")
            return None

        # Publish intent-to-close BEFORE sending the Binance order so the
        # monitor reconciler (main._reconcile_open_orders) can recognise this
        # as a bot-initiated close in flight and skip it.  Live mode only —
        # paper mode never hits the reconciler.  See CLAUDE.md "SUIUSDT
        # reconciler race (Apr 16)" for the original incident.
        if not self.is_paper_mode:
            await self._mark_close_in_progress(db, order.id)

        # Attempt maker exit if enabled, otherwise use taker
        tc = config.trading_config
        maker_exit_enabled = getattr(tc, 'maker_exit_enabled', False)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)
        exit_order_type = 'TAKER'
        actual_exit_price = current_price

        if not self.is_paper_mode:
            # --- Live mode: exit with bounded retry ---
            max_exit_retries = 3
            exit_result = None

            _urgent_exit = any(reason.startswith(p) for p in (
                "STOP_LOSS", "BREAKEVEN_EXIT", "FL_SIGNAL_LOST", "FL_REGIME_CHANGE", "FL_TICK_MOMENTUM", "FL_EMERGENCY_SL", "FL_DEEP_STOP", "FL_RECOVERED",
            ))

            for attempt in range(1, max_exit_retries + 1):
                if maker_exit_enabled and reason != "MANUAL" and not _urgent_exit:
                    symbol = order.pair.replace('USDT', '/USDT:USDT')
                    exit_result = await self._try_maker_exit(
                        symbol=symbol, side=order.direction, amount=order.quantity,
                        pair=order.pair, direction=order.direction, current_price=current_price
                    )
                else:
                    symbol = order.pair.replace('USDT', '/USDT:USDT')
                    result = await binance_service.close_position(
                        symbol=symbol, side=order.direction, amount=order.quantity
                    )
                    if result:
                        # Use actual Binance fill price, fall back to WebSocket price only if unavailable
                        binance_fill_price = result.get('price', 0)
                        if binance_fill_price and binance_fill_price > 0:
                            _exit_price = binance_fill_price
                        else:
                            _exit_price = current_price
                            logger.warning(f"[EXIT_FILL_PRICE] {order.pair}: Binance returned no fill price, using WebSocket price {current_price}")
                        exit_result = {
                            'price': _exit_price,
                            'fee_rate': taker_fee_rate,
                            'exit_order_type': 'TAKER',
                            'decision_price': current_price,  # WebSocket price at decision time for slippage calc
                        }
                    else:
                        exit_result = None

                if exit_result is not None:
                    break

                if attempt < max_exit_retries:
                    logger.warning(
                        f"[EXIT_RETRY] {order.pair}: Attempt {attempt}/{max_exit_retries} failed, retrying in 2s..."
                    )
                    await asyncio.sleep(2)

                    # Before retrying, check if position is already gone (close succeeded but response was lost)
                    try:
                        _check_pos = await binance_service.get_position_for_symbol(symbol)
                        if _check_pos is None:
                            # Position gone — close succeeded on Binance, fetch actual fill price
                            actual_price = await self._fetch_actual_fill_price(order, current_price)
                            logger.info(
                                f"[EXIT_ALREADY_CLOSED] {order.pair}: Position gone after attempt {attempt}, "
                                f"close succeeded on Binance (fill={actual_price}). Skipping retries."
                            )
                            _decision_price_for_slip = current_price
                            exit_result = {
                                'price': actual_price,
                                'fee_rate': taker_fee_rate,
                                'exit_order_type': 'TAKER',
                                'decision_price': _decision_price_for_slip,
                            }
                            break
                    except Exception as _check_err:
                        logger.warning(f"[EXIT_RETRY] {order.pair}: Position check failed ({_check_err}), continuing retry")

                    _retry_tracker = websocket_tracker.get_tracker(order.pair)
                    fresh_price = _retry_tracker.last_price if _retry_tracker else None
                    if fresh_price and fresh_price > 0:
                        current_price = fresh_price

            if exit_result is None:
                try:
                    positions = await binance_service.get_open_positions()
                    if positions is None:
                        raise RuntimeError("Binance API error — cannot determine position state")
                    binance_pairs = {p['symbol'].replace('/USDT:USDT', 'USDT') for p in positions}
                    if order.pair not in binance_pairs:
                        actual_price = await self._fetch_actual_fill_price(order, current_price)
                        logger.warning(f"[CLOSE_FALLBACK] {order.pair}: position gone from Binance — closing in DB @ {actual_price} (reason={reason})")
                        order.status = "CLOSED"
                        order.close_reason = reason
                        order.closed_at = datetime.utcnow()
                        order.exit_price = actual_price
                        order.exit_btc_regime = classify_btc_regime(_current_btc_adx, _current_btc_rsi, _btc_ema20_slope_pct)
                        # Exit BTC trend gap at close (May 6, simplified May 7)
                        try:
                            order.exit_btc_trend_gap_pct = await self._get_exit_btc_trend_gap()
                        except Exception as _e:
                            logger.debug(f"[EXIT_GAPS] {order.pair}: capture failed: {_e}")
                        taker_fee = getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                        notional_at_close = order.quantity * actual_price
                        exit_fee = notional_at_close * taker_fee
                        if order.direction == "LONG":
                            raw_pnl = (actual_price - order.entry_price) * order.quantity
                        else:
                            raw_pnl = (order.entry_price - actual_price) * order.quantity
                        order.pnl = round(raw_pnl - (order.entry_fee or 0) - exit_fee, 4)
                        _notional = order.entry_price * order.quantity if order.quantity else 1
                        order.pnl_percentage = round(((raw_pnl - (order.entry_fee or 0) - exit_fee) / _notional) * 100, 4)
                        order.exit_fee = round(exit_fee, 4)
                        order.total_fee = round((order.entry_fee or 0) + exit_fee, 4)
                        order.exit_order_type = "EXTERNAL"
                        # Slippage for external close: compare WebSocket decision price vs actual fill
                        if current_price and current_price > 0 and actual_price > 0:
                            if order.direction == "LONG":
                                order.exit_slippage_pct = round((current_price - actual_price) / current_price * 100, 4)
                            else:
                                order.exit_slippage_pct = round((actual_price - current_price) / current_price * 100, 4)
                            logger.info(
                                f"[EXIT_SLIPPAGE] {order.pair} {order.direction} (EXTERNAL): "
                                f"decision={current_price:.6f}, fill={actual_price:.6f}, "
                                f"slippage={order.exit_slippage_pct:+.4f}%"
                            )
                        tx = Transaction(
                            order_id=order.id, pair=order.pair,
                            action=f"CLOSE_{order.direction}", price=actual_price,
                            quantity=order.quantity, investment=order.investment,
                            leverage=order.leverage, notional_value=order.notional_value,
                            fee=order.exit_fee, order_type="EXTERNAL",
                            is_paper=False
                        )
                        db.add(tx)
                        _exit_retry_queue.pop(order.id, None)
                        await db.commit()
                        await db.refresh(order)
                        async with _cache_lock:
                            _open_orders_cache[order.pair] = [
                                o for o in _open_orders_cache.get(order.pair, []) if o['id'] != order.id
                            ]
                        return order
                except Exception as e:
                    logger.error(f"[EXTERNAL_CLOSE] {order.pair}: reconcile check failed: {e}")
                _exit_retry_queue.setdefault(order.id, 0)
                logger.critical(
                    f"[EXIT_FAILED] {order.pair}: All {max_exit_retries} exit attempts failed — "
                    f"added to retry queue (attempt {_exit_retry_queue[order.id]}/{_EXIT_RETRY_MAX})"
                )
                return None

            actual_exit_price = exit_result['price']
            exit_fee_rate = exit_result['fee_rate']
            exit_order_type = exit_result['exit_order_type']
            notional_at_close = order.quantity * actual_exit_price
            exit_fee = notional_at_close * exit_fee_rate

            # SLIPPAGE TRACKING: compare decision price (WebSocket) vs actual Binance fill
            _decision_price = exit_result.get('decision_price', current_price)
            if _decision_price and _decision_price > 0 and actual_exit_price > 0:
                if order.direction == "LONG":
                    # Closing a LONG = selling. Worse fill = lower price. Slippage = (decision - actual) / decision * 100
                    _slippage_pct = round((_decision_price - actual_exit_price) / _decision_price * 100, 4)
                else:
                    # Closing a SHORT = buying. Worse fill = higher price. Slippage = (actual - decision) / decision * 100
                    _slippage_pct = round((actual_exit_price - _decision_price) / _decision_price * 100, 4)
                _slippage_dollar = abs(actual_exit_price - _decision_price) * order.quantity
                _direction_label = "WORSE" if _slippage_pct > 0 else "BETTER" if _slippage_pct < 0 else "EXACT"
                logger.info(
                    f"[EXIT_SLIPPAGE] {order.pair} {order.direction}: "
                    f"decision={_decision_price:.6f}, fill={actual_exit_price:.6f}, "
                    f"slippage={_slippage_pct:+.4f}% (${_slippage_dollar:.2f}) [{_direction_label}] "
                    f"type={exit_order_type}"
                )
            else:
                _slippage_pct = None

            # POST-CLOSE VERIFICATION: confirm position is actually gone from Binance
            try:
                await asyncio.sleep(1)  # brief delay for Binance to process
                positions = await binance_service.get_open_positions()
                if positions is not None:
                    binance_pairs = {p['symbol'].replace('/USDT:USDT', 'USDT') for p in positions}
                    if order.pair in binance_pairs:
                        logger.critical(f"[CLOSE_VERIFY_FAIL] {order.pair}: Position still open on Binance after close — will be caught by retry queue or next reconciliation")
                    else:
                        logger.info(f"[CLOSE_VERIFY_OK] {order.pair}: Position confirmed closed on Binance")
            except Exception as e:
                logger.warning(f"[CLOSE_VERIFY] {order.pair}: Verification check failed: {e}")
        else:
            # --- Paper mode: no retry needed, no slippage ---
            _slippage_pct = None
            _urgent_exit_paper = any(reason.startswith(p) for p in (
                "STOP_LOSS", "BREAKEVEN_EXIT", "FL_SIGNAL_LOST", "FL_REGIME_CHANGE", "FL_TICK_MOMENTUM", "FL_EMERGENCY_SL", "FL_DEEP_STOP", "FL_RECOVERED",
            ))
            if maker_exit_enabled and reason != "MANUAL" and not _urgent_exit_paper:
                exit_result = await self._simulate_maker_exit_paper(
                    pair=order.pair, direction=order.direction, current_price=current_price
                )
                actual_exit_price = exit_result['price']
                exit_fee_rate = exit_result['fee_rate']
                exit_order_type = exit_result['exit_order_type']
                notional_at_close = order.quantity * actual_exit_price
                exit_fee = notional_at_close * exit_fee_rate
            else:
                notional_at_close = order.quantity * current_price
                exit_fee = notional_at_close * taker_fee_rate

        total_fee = (order.entry_fee or 0) + exit_fee

        # Apply FL_ prefix if trade was flagged (check cache before it can be wiped)
        # This ensures ALL close reasons get the flag, not just those in Phase 2
        if not reason.startswith("FL_"):
            for _fl_cached in _open_orders_cache.get(order.pair, []):
                if _fl_cached['id'] == order.id and _fl_cached.get('signal_lost_flagged'):
                    reason = f"FL_{reason}"
                    break

        # Calculate P&L
        pnl_data = calculate_pnl(
            direction=order.direction,
            entry_price=order.entry_price,
            current_price=actual_exit_price,
            quantity=order.quantity,
            leverage=order.leverage,
            entry_fee=order.entry_fee or 0,
            exit_fee=exit_fee
        )

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Essential close — commit to DB immediately so the
        # order is never left as a zombie if optional metadata fails.
        # Uses retry loop on the SAME session to handle SQLite
        # "database is locked" errors from scan loop contention.
        # ═══════════════════════════════════════════════════════════════
        _close_time = datetime.utcnow()
        _db_commit_success = False
        _max_db_retries = 5
        # Cache every field needed to rebuild state during a retry.
        # After a rollback SQLAlchemy expires ORM instances — any subsequent
        # sync attribute read (order.pair, order.quantity, ...) triggers a
        # lazy-load which in async context raises
        # "greenlet_spawn has not been called; can't call await_only()".
        # Reading primitives from local variables avoids that entirely.
        order_pair = order.pair
        order_id = order.id
        _tx_binance_order_id = order.binance_order_id
        _tx_direction = order.direction
        _tx_quantity = order.quantity
        _tx_investment = order.investment
        _tx_leverage = order.leverage
        _tx_is_paper = order.is_paper
        _db_attempt = 0

        # Track total time spent waiting on the DB across all retry attempts so we can
        # measure real-world lock contention in CloudWatch later.  Each attempt also
        # records its own elapsed time individually in the [DB_LOCKED] / [DB_COMMIT_OK]
        # log lines.  busy_timeout is 5s per attempt so a healthy commit is <100ms; any
        # elapsed value close to 5s indicates the attempt hit the timeout ceiling.
        _db_total_wait_start = time.monotonic()

        for _db_attempt in range(1, _max_db_retries + 1):
            _db_attempt_start = time.monotonic()
            try:
                # On retry, re-fetch the order so we operate on a fresh attached instance
                # instead of the one expired by the previous rollback.  No autoflush risk
                # because Transaction is added later in this try block, not in the except.
                if _db_attempt > 1:
                    _fresh = await db.execute(
                        select(Order).where(Order.id == order_id)
                    )
                    order = _fresh.scalar_one_or_none()
                    if order is None:
                        logger.error(f"[DB_RETRY] {order_pair}: order {order_id} disappeared between attempts, aborting")
                        break

                # Set all fields on each attempt (rollback resets dirty state)
                order.status = "CLOSED"
                order.exit_price = actual_exit_price
                order.exit_fee = exit_fee
                order.total_fee = total_fee
                order.exit_order_type = exit_order_type
                order.pnl = pnl_data['pnl']
                order.pnl_percentage = pnl_data['pnl_percentage']

                # ─────────────────────────────────────────────────────────────
                # May 7 — Sync realtime cache → DB BEFORE invariant guard.
                # Without this, realtime-triggered closes (trailing, EMA13/Stack
                # cross, RSI Handoff, etc.) persist STALE peak/low values from
                # the last monitor-loop write, even when the cache has fresher
                # values that the realtime trigger itself just used.
                # The invariant guard below would only enforce peak >= close,
                # but the actual intra-trade peak (per cache) could be higher.
                # Pull from cache here so DB matches realtime cache state.
                # ─────────────────────────────────────────────────────────────
                try:
                    for _cached in _open_orders_cache.get(order_pair, []):
                        if _cached['id'] == order_id:
                            _cache_low = _cached.get('low_price')
                            _cache_high = _cached.get('high_price')
                            _cache_peak = _cached.get('peak_pnl')
                            _cache_trough = _cached.get('trough_pnl')
                            if _cache_low is not None and _cache_low > 0:
                                if order.low_price_since_entry is None or _cache_low < order.low_price_since_entry:
                                    order.low_price_since_entry = _cache_low
                            if _cache_high is not None and _cache_high > 0:
                                if order.high_price_since_entry is None or _cache_high > order.high_price_since_entry:
                                    order.high_price_since_entry = _cache_high
                            if _cache_peak is not None:
                                if order.peak_pnl is None or _cache_peak > order.peak_pnl:
                                    order.peak_pnl = _cache_peak
                            if _cache_trough is not None:
                                if order.trough_pnl is None or _cache_trough < order.trough_pnl:
                                    order.trough_pnl = _cache_trough
                            # May 17: persist post-arm-min for BE-floor counterfactual.
                            # Only set if BE armed during the trade (peak crossed BE trigger).
                            _cache_pam = _cached.get('post_arm_min_pnl')
                            _cache_pam_at = _cached.get('post_arm_min_at')
                            if _cache_pam is not None and _cached.get('be_armed'):
                                if order.post_arm_min_pnl_pct is None or _cache_pam < order.post_arm_min_pnl_pct:
                                    order.post_arm_min_pnl_pct = _cache_pam
                                    order.post_arm_min_pnl_at = _cache_pam_at
                            break
                except Exception as _sync_e:
                    logger.debug(f"[CACHE_SYNC_PRE_CLOSE] {order_pair}: cache sync skipped: {_sync_e}")

                # Enforce invariant: peak P&L must be ≥ close P&L, trough P&L must be ≤ close P&L.
                # The realtime callback can miss intra-tick spikes (WS tick stream isn't continuous),
                # so the cached peak/trough can lag. The actual exit price is always a real point
                # the trade reached, so peak/trough must bracket it. Without this fix, reports show
                # impossible cells like "peak +0.03% / close +0.35%" (Apr 29 closed-orders bug).
                # Log every activation so we can quantify how often the cache lag is happening
                # — frequent activations indicate an upstream realtime-callback issue worth
                # investigating beyond this symptom-level guard.
                _close_pct = pnl_data['pnl_percentage']
                _old_peak = order.peak_pnl
                _old_trough = order.trough_pnl
                if order.peak_pnl is None or order.peak_pnl < _close_pct:
                    order.peak_pnl = _close_pct
                    if _old_peak is not None:
                        logger.warning(
                            f"[PEAK_INVARIANT_FIX] {order_pair} {_tx_direction}: "
                            f"peak_pnl was {_old_peak:+.4f}% but close was {_close_pct:+.4f}% — "
                            f"corrected (likely realtime-callback cache lag, reason={reason})"
                        )
                if order.trough_pnl is None or order.trough_pnl > _close_pct:
                    order.trough_pnl = _close_pct
                    if _old_trough is not None and _close_pct < 0:
                        logger.warning(
                            f"[TROUGH_INVARIANT_FIX] {order_pair} {_tx_direction}: "
                            f"trough_pnl was {_old_trough:+.4f}% but close was {_close_pct:+.4f}% — "
                            f"corrected (likely realtime-callback cache lag, reason={reason})"
                        )
                order.closed_at = _close_time
                order.close_reason = reason
                order.exit_slippage_pct = _slippage_pct
                order.exit_btc_regime = classify_btc_regime(_current_btc_adx, _current_btc_rsi, _btc_ema20_slope_pct)
                # Exit BTC trend gap at close (May 6, simplified May 7)
                try:
                    order.exit_btc_trend_gap_pct = await self._get_exit_btc_trend_gap()
                except Exception as _e:
                    logger.debug(f"[EXIT_GAPS] {order.pair}: capture failed: {_e}")

                # Create and add the Transaction on EVERY attempt, right before commit.
                # Rollback on the previous iteration removed any prior pending Transaction,
                # so the session only holds one at a time.  Keeping db.add(...) inside the
                # try block (and not in the except) ensures the re-fetch select on the
                # next iteration has NOTHING to autoflush — this is what caused the
                # "Query-invoked autoflush" cascade (and the 2-minute DOTUSDT stall).
                transaction = Transaction(
                    order_id=order_id,
                    binance_order_id=_tx_binance_order_id,
                    pair=order_pair,
                    action=f"CLOSE_{_tx_direction}",
                    price=actual_exit_price,
                    quantity=_tx_quantity,
                    investment=_tx_investment,
                    leverage=_tx_leverage,
                    notional_value=notional_at_close,
                    fee=exit_fee,
                    order_type=exit_order_type,
                    is_paper=_tx_is_paper
                )
                db.add(transaction)

                await db.commit()
                await db.refresh(order)
                _db_commit_success = True

                _attempt_elapsed = time.monotonic() - _db_attempt_start
                _total_elapsed = time.monotonic() - _db_total_wait_start

                _slip_str = f", slippage={_slippage_pct:+.4f}%" if _slippage_pct is not None else ""
                if _db_attempt > 1:
                    logger.info(
                        f"[DB_RETRY_OK] {order_pair} {_tx_direction}: DB commit succeeded on attempt {_db_attempt} "
                        f"(attempt_waited={_attempt_elapsed:.2f}s, total_waited={_total_elapsed:.2f}s, "
                        f"reason={reason}, pnl=${pnl_data['pnl']:.4f}, exit={actual_exit_price:.6f}{_slip_str})"
                    )
                else:
                    # Only emit DB_COMMIT_SLOW if the first attempt took >1s — this flags
                    # low-grade contention that didn't fully fail but was still slow.
                    if _attempt_elapsed > 1.0:
                        logger.warning(
                            f"[DB_COMMIT_SLOW] {order_pair}: first-attempt commit took {_attempt_elapsed:.2f}s "
                            f"(below 5s timeout — lock contention is present but not starving)"
                        )
                    logger.info(
                        f"[CLOSE_COMMITTED] {order_pair} {_tx_direction}: essential close saved "
                        f"(waited={_attempt_elapsed:.2f}s, reason={reason}, pnl=${pnl_data['pnl']:.4f}, "
                        f"exit={actual_exit_price:.6f}{_slip_str})"
                    )
                break

            except Exception as _db_err:
                _attempt_elapsed = time.monotonic() - _db_attempt_start
                _err_str = str(_db_err)
                if _db_attempt < _max_db_retries:
                    logger.warning(
                        f"[DB_LOCKED] {order_pair}: DB commit attempt {_db_attempt}/{_max_db_retries} failed "
                        f"after waited={_attempt_elapsed:.2f}s ({_err_str[:80]}), "
                        f"retrying in {_db_attempt}s... (Binance close already succeeded)"
                    )
                    try:
                        await db.rollback()
                    except Exception:
                        pass
                    await asyncio.sleep(_db_attempt)  # progressive backoff: 1s, 2s, 3s, 4s
                    # Do NOT add a Transaction here — it will be created fresh in the
                    # next iteration's try block, right before commit.  Adding it here
                    # leaves it pending during the select re-fetch, which triggers an
                    # autoflush cascade that was stalling closes for 2+ minutes.
                else:
                    _total_elapsed = time.monotonic() - _db_total_wait_start
                    logger.error(
                        f"[DB_COMMIT_FAILED] {order_pair}: All {_db_attempt} DB commit attempts failed "
                        f"(final_attempt_waited={_attempt_elapsed:.2f}s, total_waited={_total_elapsed:.2f}s): "
                        f"{_err_str[:120]}"
                    )

        if not _db_commit_success:
            _total_elapsed = time.monotonic() - _db_total_wait_start
            logger.critical(
                f"[DB_COMMIT_FAILED] {order_pair}: Could not save close to DB after {_max_db_retries} attempts "
                f"(total_waited={_total_elapsed:.2f}s). "
                f"Position IS closed on Binance (exit={actual_exit_price}). "
                f"Will be caught by next reconciliation cycle."
            )
            return None

        # Broker-side protective stop cancellation REMOVED Apr 17 with the
        # feature itself.  See place site (around open_position commit) for
        # the forensic context.

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Optional metadata — failures here must NEVER revert
        # the close above.  A second commit persists the extras.
        # ═══════════════════════════════════════════════════════════════
        try:
            # Persist phantom shadow data, peak EMA5 metrics, and signal-lost flag from real-time cache
            for cached in _open_orders_cache.get(order_pair, []):
                if cached['id'] == order_id:
                    if cached.get('signal_lost_flagged'):
                        order.signal_lost_flagged = True
                        order.signal_lost_flag_pnl = cached.get('signal_lost_flag_pnl')
                        order.signal_lost_flagged_at = cached.get('signal_lost_flagged_at')
                        if not reason.startswith("FL_"):
                            order.close_reason = f"FL_{reason}"
                    order.phantom_be_l1_triggered_at = cached.get('phantom_be_l1_triggered_at')
                    order.phantom_be_l1_would_exit_pnl = cached.get('phantom_be_l1_would_exit_pnl')
                    order.phantom_be_l2_triggered_at = cached.get('phantom_be_l2_triggered_at')
                    order.phantom_be_l2_would_exit_pnl = cached.get('phantom_be_l2_would_exit_pnl')
                    # May 14 — aggressive phantom BE @ 0.20/0.10
                    order.phantom_be_aggr_triggered_at = cached.get('phantom_be_aggr_triggered_at')
                    order.phantom_be_aggr_would_exit_pnl = cached.get('phantom_be_aggr_would_exit_pnl')
                    order.phantom_regime_change_exit_triggered_at = cached.get('phantom_regime_change_exit_triggered_at')
                    order.phantom_regime_change_exit_pnl = cached.get('phantom_regime_change_exit_pnl')
                    for _lbl in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                        setattr(order, f'phantom_tick_{_lbl}_triggered_at', cached.get(f'phantom_tick_{_lbl}_triggered_at'))
                        setattr(order, f'phantom_tick_{_lbl}_pnl', cached.get(f'phantom_tick_{_lbl}_pnl'))
                    if cached.get('peak_ema5_dist_pct') is not None:
                        order.peak_ema5_dist_pct = cached['peak_ema5_dist_pct']
                    if cached.get('peak_ema5_slope_pct') is not None:
                        order.peak_ema5_slope_pct = cached['peak_ema5_slope_pct']
                    if cached.get('peak_reached_at') is not None:
                        order.peak_reached_at = cached['peak_reached_at']
                    if cached.get('trough_reached_at') is not None:
                        order.trough_reached_at = cached['trough_reached_at']
                    if cached.get('trough_ema5_dist_pct') is not None:
                        order.trough_ema5_dist_pct = cached['trough_ema5_dist_pct']
                    order.regime_neutral_hit_at = cached.get('regime_neutral_hit_at')
                    order.regime_neutral_pnl = cached.get('regime_neutral_pnl')
                    order.regime_comeback_at = cached.get('regime_comeback_at')
                    order.regime_comeback_pnl = cached.get('regime_comeback_pnl')
                    order.regime_opposite_at = cached.get('regime_opposite_at')
                    order.regime_opposite_pnl = cached.get('regime_opposite_pnl')
                    order._ema5_ever_negative = cached.get('ema5_ever_negative', False)
                    # Phase 1 shadow tracking — persist price-vs-EMA cross moments + counterfactual P&L
                    if cached.get('first_cross_ema13_at') is not None:
                        order.first_cross_ema13_at = cached['first_cross_ema13_at']
                        order.first_cross_ema13_pnl_pct = cached.get('first_cross_ema13_pnl_pct')
                    if cached.get('confirmed_cross_ema13_at') is not None:
                        order.confirmed_cross_ema13_at = cached['confirmed_cross_ema13_at']
                        order.confirmed_cross_ema13_pnl_pct = cached.get('confirmed_cross_ema13_pnl_pct')
                    if cached.get('first_cross_ema20_at') is not None:
                        order.first_cross_ema20_at = cached['first_cross_ema20_at']
                        order.first_cross_ema20_pnl_pct = cached.get('first_cross_ema20_pnl_pct')
                    if cached.get('confirmed_cross_ema20_at') is not None:
                        order.confirmed_cross_ema20_at = cached['confirmed_cross_ema20_at']
                        order.confirmed_cross_ema20_pnl_pct = cached.get('confirmed_cross_ema20_pnl_pct')
                    break

            pd = None
            try:
                pair_data_result = await db.execute(
                    select(PairData).where(PairData.pair == order.pair)
                )
                pd = pair_data_result.scalar_one_or_none()
                if pd:
                    order.signal_active_at_close = is_signal_direction_active(
                        order.direction, pd.ema5, pd.ema8, pd.ema20, pd.price
                    )
                else:
                    order.signal_active_at_close = None
            except Exception:
                order.signal_active_at_close = None

            if pd and pd.ema5 and actual_exit_price:
                if order.direction == "LONG":
                    order.exit_price_vs_ema5_pct = round((actual_exit_price - pd.ema5) / actual_exit_price * 100, 4)
                else:
                    order.exit_price_vs_ema5_pct = round((pd.ema5 - actual_exit_price) / actual_exit_price * 100, 4)
                if pd.ema5_prev3 and pd.ema5:
                    order.exit_ema5_slope_pct = round((pd.ema5 - pd.ema5_prev3) / pd.ema5 * 100, 4)

            try:
                ohlcv_data = await binance_service.get_ohlcv(order.pair, '5m', 100)
                if ohlcv_data and len(ohlcv_data) >= 51 and order.opened_at:
                    import pandas as _pd_lib
                    from ta.trend import EMAIndicator as _EMA
                    df = _pd_lib.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['close'] = df['close'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['timestamp'] = _pd_lib.to_datetime(df['timestamp'], unit='ms')
                    ema5_series = _EMA(close=df['close'], window=5).ema_indicator()
                    entry_ts = order.opened_at
                    if entry_ts.tzinfo:
                        entry_ts = entry_ts.replace(tzinfo=None)
                    mask = df['timestamp'] >= _pd_lib.Timestamp(entry_ts)
                    crossed = False
                    for idx in df.index[mask]:
                        e5 = ema5_series.iloc[idx]
                        if _pd_lib.isna(e5):
                            continue
                        if order.direction == "LONG" and df.at[idx, 'low'] <= e5:
                            crossed = True
                            break
                        elif order.direction == "SHORT" and df.at[idx, 'high'] >= e5:
                            crossed = True
                            break
                    order.exit_ema5_crossed = crossed
            except Exception:
                pass

            ema5_ever_neg = getattr(order, '_ema5_ever_negative', False)
            if not ema5_ever_neg:
                order.ema5_went_negative = "NEVER"
            elif order.exit_price_vs_ema5_pct is not None and order.exit_price_vs_ema5_pct >= 0:
                order.ema5_went_negative = "RECOVERED"
            else:
                order.ema5_went_negative = "ENDED_NEG"

            await db.commit()
        except Exception as _meta_err:
            logger.warning(f"[CLOSE_METADATA] {order.pair}: Optional metadata failed (order already closed safely): {_meta_err}")
            try:
                await db.rollback()
            except Exception:
                pass
        
        # Recalculate paper balance from DB (source of truth) and save
        if self.is_paper_mode:
            pre_usdt = self.paper_balance
            pre_bnb = self.paper_bnb_balance_usd

            await self._recalculate_paper_balance(db)
            await self._deduct_fee_from_bnb(exit_fee, db)
            await self.save_state(db)

            _snap = await db.execute(
                select(func.coalesce(func.sum(Order.investment), 0)).where(
                    and_(Order.status == "OPEN", Order.is_paper == True)
                )
            )
            post_margin = _snap.scalar() or 0
            pre_margin = post_margin + order.investment
            pre_total = pre_usdt + pre_margin + pre_bnb
            post_total = self.paper_balance + post_margin + self.paper_bnb_balance_usd
            delta = post_total - pre_total
            logger.info(
                f"[PORTFOLIO_CLOSE] {order.pair} {order.direction} | "
                f"PnL={pnl_data['pnl']:+.4f} (raw={pnl_data['raw_pnl']:+.4f} "
                f"entry_fee={order.entry_fee or 0:.4f} exit_fee={exit_fee:.4f}) | "
                f"PRE: USDT={pre_usdt:.2f} Margin={pre_margin:.2f} "
                f"BNB={pre_bnb:.2f} Total={pre_total:.2f} | "
                f"POST: USDT={self.paper_balance:.2f} Margin={post_margin:.2f} "
                f"BNB={self.paper_bnb_balance_usd:.2f} Total={post_total:.2f} | "
                f"Delta={delta:+.2f} vs NetPnL={pnl_data['pnl']:+.4f}"
            )
        elif not self.is_paper_mode:
            try:
                bal = await binance_service.get_balance()
                bnb_price = await binance_service.get_bnb_price()
                bnb_usd = bal['bnb_total'] * bnb_price if bnb_price > 0 else 0
                total = bal['usdt_total'] + bnb_usd
                logger.info(
                    f"[PORTFOLIO_CLOSE] {order.pair} {order.direction} | "
                    f"PnL={pnl_data['pnl']:+.4f} (raw={pnl_data['raw_pnl']:+.4f} "
                    f"entry_fee={order.entry_fee or 0:.4f} exit_fee={exit_fee:.4f}) | "
                    f"USDT_total={bal['usdt_total']:.2f} USDT_free={bal['usdt_free']:.2f} "
                    f"BNB={bal['bnb_total']:.6f} BNB_price={bnb_price:.2f} BNB_usd={bnb_usd:.2f} | "
                    f"Total={total:.2f}"
                )
            except Exception as e:
                logger.warning(f"[PORTFOLIO_CLOSE] Failed to log live balance: {e}")

        self._register_post_exit_tracking(order, reason)
        self._rsi3_history.pop(order.id, None)

        return order

    def _register_post_exit_tracking(self, order: Order, reason: str):
        """Register a BE or Signal Lost exit trade for post-exit price tracking (regret metric)."""
        tc = config.trading_config
        if not getattr(tc, 'post_exit_tracking_enabled', False):
            return
        # Jun 14: strip the FLIP_ prefix (then any FL_) so flip exits resolve to their
        # base reason and get post-exit (regret) tracking like the normal exit.
        _reason_base = reason
        if _reason_base.startswith("FLIP_"):
            _reason_base = _reason_base[5:]
        if _reason_base.startswith("FL_"):
            _reason_base = _reason_base[3:]
        if not (_reason_base.startswith("BREAKEVEN_EXIT") or _reason_base.startswith("SIGNAL_LOST") or _reason_base.startswith("TICK_MOMENTUM_EXIT") or _reason_base.startswith("RSI_MOMENTUM_EXIT") or _reason_base.startswith("RSI_HANDOFF_EXIT") or _reason_base.startswith("EMA13_CROSS_EXIT") or _reason_base.startswith("EMA_STACK_CROSS_EXIT") or _reason_base.startswith("STOP_LOSS") or _reason_base.startswith("REGIME_CHANGE") or _reason_base.startswith("TRAILING_STOP") or _reason_base.startswith("RUNNER_TRAIL") or _reason_base.startswith("MOMENTUM_EXIT") or _reason_base.startswith("SLOPE_EXIT") or _reason_base.startswith("NO_EXPANSION") or _reason_base.startswith("RECOVERED") or _reason_base.startswith("DEEP_STOP") or _reason_base.startswith("EMERGENCY_SL") or _reason_base.startswith("FAST_EXIT") or _reason_base.startswith("ATR_FIXED_TP") or _reason_base.startswith("PATTERN_FIXED_TP") or _reason_base.startswith("PATTERN_FIXED_SL")):
            return
        minutes = getattr(tc, 'post_exit_tracking_minutes', 45)
        tracker = websocket_tracker.get_tracker(order.pair)
        initial_price = tracker.last_price if tracker else order.exit_price
        now = datetime.utcnow()
        # Carry tick buffer and phantom tick states from cache
        cached_tick_buf = []
        phantom_tick_states = {}
        for cached in _open_orders_cache.get(order.pair, []):
            if cached['id'] == order.id:
                cached_tick_buf = cached.get('tick_prices', [])
                for _lbl in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                    phantom_tick_states[f'phantom_tick_{_lbl}_triggered'] = cached.get(f'phantom_tick_{_lbl}_triggered', False)
                    phantom_tick_states[f'phantom_tick_{_lbl}_triggered_at'] = cached.get(f'phantom_tick_{_lbl}_triggered_at')
                    phantom_tick_states[f'phantom_tick_{_lbl}_pnl'] = cached.get(f'phantom_tick_{_lbl}_pnl')
                break

        _pe_notional = order.entry_price * order.quantity if order.quantity else 1
        _pe_fee_drag = (((order.entry_fee or 0) + _pe_notional * getattr(tc, 'taker_fee', tc.trading_fee)) / _pe_notional) * 100

        self._post_exit_tracking[order.id] = {
            "order_id": order.id,
            "pair": order.pair,
            "entry_price": order.entry_price,
            "direction": order.direction,
            "fee_drag_pct": _pe_fee_drag,
            "exit_time": now,
            "tracking_until": now + timedelta(minutes=minutes),
            "post_high": initial_price or order.exit_price,
            "post_low": initial_price or order.exit_price,
            "peak_at": now,
            "trough_at": now,
            "signal_lost_at": None,
            "pnl_at_signal_lost": None,
            "peak_before_signal_lost": 0.0,
            "rsi_exit_at": None,
            "rsi_exit_pnl": None,
            "rsi3_exit_at": None,
            "rsi3_exit_pnl": None,
            "rsi_history": [],
            "ema13_cross_at": None,
            "ema13_cross_pnl": None,
            # May 23: post-exit regime-flip tracker (fresh registration path)
            "entry_regime": order.entry_btc_regime,
            "regime_flip_at": None,
            "regime_flip_pnl": None,
            "signal_regained_at": None,
            "pnl_at_signal_regained": None,
            "running_min_pnl": None,
            "floor_before_signal_regain": None,
            "close_reason": reason,
            "tick_prices": cached_tick_buf,
            # May 12 LATE PM: time-bucketed P&L snapshots (1/2/5/15/30 min after exit)
            "pnl_at_1min": None,
            "pnl_at_2min": None,
            "pnl_at_5min": None,
            "pnl_at_15min": None,
            "pnl_at_30min": None,
            **phantom_tick_states,
        }
        logger.info(f"[POST_EXIT] Registered {order.pair} order {order.id} ({reason}) for {minutes}min tracking")

    async def update_phantom_flips(self, db: AsyncSession):
        """Monitor-tick (1s) update of virtual flip positions (Jun 13, observation-only).
        For each blocked-entry phantom, read the live ws price, compute the flip's raw
        price-move %, apply the SL/trailing exit model, and persist on exit/timeout.
        Fully fail-silent — NEVER affects live trading. Uses an isolated DB session per
        persist so a failure can't poison the monitor-loop session."""
        if not _PHANTOM_FLIP_STATE:
            return
        try:
            now = _leash_time.time()
            done = []  # (key, state, exit_pnl, reason)
            for key, st in list(_PHANTOM_FLIP_STATE.items()):
                try:
                    aged = (now - st['open_ts']) / 60.0
                    tracker = websocket_tracker.get_tracker(st['pair'])
                    price = tracker.last_price if tracker else None
                    if not price or price <= 0:
                        if aged >= _PFLIP_MAX_MIN:
                            done.append((key, st, st.get('_last_pnl', 0.0), 'horizon'))
                        continue
                    if st['flip_dir'] == "LONG":
                        pnl = (price - st['entry']) / st['entry'] * 100.0
                    else:
                        pnl = (st['entry'] - price) / st['entry'] * 100.0
                    st['_last_pnl'] = pnl
                    if pnl > st['peak']:
                        st['peak'] = pnl
                    if pnl < st['trough']:
                        st['trough'] = pnl
                    if not st['armed'] and st['peak'] >= _PFLIP_ACT:
                        st['armed'] = True
                    if pnl <= _PFLIP_SL:
                        done.append((key, st, _PFLIP_SL, 'sl'))
                    elif st['armed'] and pnl <= st['peak'] - _PFLIP_PB:
                        done.append((key, st, round(st['peak'] - _PFLIP_PB, 4), 'trail'))
                    elif aged >= _PFLIP_MAX_MIN:
                        done.append((key, st, round(pnl, 4), 'horizon'))
                except Exception:
                    continue
            for key, st, exit_pnl, reason in done:
                try:
                    async with AsyncSessionLocal() as _pdb:
                        _pdb.add(PhantomFlip(
                            pair=st['pair'], source_filter=st['source'],
                            blocked_direction=st['blocked_dir'], flip_direction=st['flip_dir'],
                            entry_price=st['entry'], pnl_pct=round(exit_pnl, 4),
                            peak_pct=round(st['peak'], 4), trough_pct=round(st['trough'], 4),
                            exit_reason=reason, is_paper=self.is_paper_mode,
                            entry_cohort=st.get('cohort'),
                            entry_at=datetime.utcfromtimestamp(st['open_ts']),
                            exit_at=datetime.utcnow(),
                            # Jun 15: full entry context (RSI/ATR/fan-ratio/regime) for analysis.
                            # Filter to REAL PhantomFlip columns — _ef is the Order-shaped field
                            # set and may carry keys the phantom table lacks; an unknown kwarg
                            # would raise into the swallowing try/except and silently kill the row.
                            **{k: v for k, v in (st.get('_ef') or {}).items()
                               if v is not None and k in PhantomFlip.__table__.columns},
                        ))
                        await _pdb.commit()
                except Exception:
                    pass
                _PHANTOM_FLIP_STATE.pop(key, None)
        except Exception:
            pass

    async def update_post_exit_tracking(self, db: AsyncSession):
        """Check prices for recently closed BE trades and update peak/trough/timing. Called from monitor loop.

        Uses isolated DB sessions for all queries and writes so that failures
        never corrupt the shared monitor-loop session / connection pool.
        """
        if not self._post_exit_tracking:
            return

        now = datetime.utcnow()
        completed = []

        for order_id in list(self._post_exit_tracking.keys()):
            info = self._post_exit_tracking[order_id]
            tracker = websocket_tracker.get_tracker(info["pair"])
            if not tracker or not tracker.last_price or tracker.last_price <= 0:
                continue

            price = tracker.last_price
            entry = info["entry_price"]
            direction = info["direction"]

            _new_high = price > info["post_high"]
            _new_low = price < info["post_low"]
            if _new_high:
                info["post_high"] = price
                info["peak_at"] = now
            if _new_low:
                info["post_low"] = price
                info["trough_at"] = now

            # May 8: persist running state to DB whenever a new extreme is observed.
            # Survives bot restart — _recover_post_exit_tracking reads these to
            # resume tracking instead of resetting peak/trough to current price.
            # Throttled to actual new highs/lows; no per-tick writes.
            if _new_high or _new_low:
                try:
                    async with AsyncSessionLocal() as _pe_state_db:
                        await _pe_state_db.execute(
                            update(Order)
                            .where(Order.id == order_id)
                            .values(
                                post_exit_running_high=info["post_high"],
                                post_exit_running_low=info["post_low"],
                                post_exit_running_peak_at=info["peak_at"],
                                post_exit_running_trough_at=info["trough_at"],
                            )
                        )
                        await _pe_state_db.commit()
                except Exception as _pe_state_exc:
                    logger.debug(f"[POST_EXIT_RUNNING] Failed to persist running state for {info['pair']}: {_pe_state_exc}")

            # Current P&L for tracking calculations (net of fees, consistent with pnl_percentage)
            if direction == "LONG":
                current_pnl = ((price - entry) / entry) * 100 - info["fee_drag_pct"]
            else:
                current_pnl = ((entry - price) / entry) * 100 - info["fee_drag_pct"]

            # Track running minimum P&L (from entry) for floor-before-recovery analysis
            if info["running_min_pnl"] is None or current_pnl < info["running_min_pnl"]:
                info["running_min_pnl"] = current_pnl

            # May 12 LATE PM: time-bucketed P&L snapshots (1/2/5/15/30 min after exit).
            # Captures the answer to "if we held N min more, what would close% be?"
            # Each snapshot is recorded only once, the first time elapsed crosses
            # the threshold. NULL means tracking ended before reaching the threshold
            # — interpret as "this counterfactual is invalid".
            _elapsed_sec = (now - info["exit_time"]).total_seconds()
            _snap_thresholds = [(60, "pnl_at_1min"), (120, "pnl_at_2min"),
                                (300, "pnl_at_5min"), (900, "pnl_at_15min"),
                                (1800, "pnl_at_30min")]
            _snap_fired = []
            for _thr_sec, _key in _snap_thresholds:
                if _elapsed_sec >= _thr_sec and info.get(_key) is None:
                    info[_key] = round(current_pnl, 4)
                    _snap_fired.append(_key)
            if _snap_fired:
                # Persist new snapshots in a single update
                try:
                    _values = {
                        f"post_exit_{k}": info[k] for k in _snap_fired
                    }
                    async with AsyncSessionLocal() as _pe_snap_db:
                        await _pe_snap_db.execute(
                            update(Order)
                            .where(Order.id == order_id)
                            .values(**_values)
                        )
                        await _pe_snap_db.commit()
                except Exception as _pe_snap_exc:
                    logger.debug(f"[POST_EXIT_SNAP] Failed to persist time snapshot for {info['pair']}: {_pe_snap_exc}")

            # Read pair_data for signal-lost, signal-regained, RSI momentum, and EMA13 cross checks (isolated session)
            pair_data = None
            if info["signal_lost_at"] is None or info["signal_regained_at"] is None or info["rsi_exit_at"] is None or info["ema13_cross_at"] is None:
                try:
                    async with AsyncSessionLocal() as pe_read_db:
                        pd_result = await pe_read_db.execute(
                            select(PairData).where(PairData.pair == info["pair"])
                        )
                        pair_data = pd_result.scalar_one_or_none()
                except Exception:
                    pass

            # Signal-lost detection
            if info["signal_lost_at"] is None and pair_data:
                if not is_signal_direction_active(
                    direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
                ):
                    info["signal_lost_at"] = now
                    info["pnl_at_signal_lost"] = current_pnl

            # Signal-regained detection (for SIGNAL_LOST exits: did the signal come back?)
            _cr = info.get("close_reason", "")
            if info["signal_regained_at"] is None and pair_data and ("SIGNAL_LOST" in _cr):
                if is_signal_direction_active(
                    direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
                ):
                    info["signal_regained_at"] = now
                    info["pnl_at_signal_regained"] = current_pnl
                    info["floor_before_signal_regain"] = info["running_min_pnl"]

            # RSI momentum exit simulation (2-drop and 3-drop)
            if pair_data and pair_data.rsi is not None:
                _rsi = pair_data.rsi
                _rsi1 = pair_data.rsi_prev1
                _rsi2 = pair_data.rsi_prev2

                # 2-drop check
                if info["rsi_exit_at"] is None and _rsi1 is not None and _rsi2 is not None:
                    rsi_triggered = False
                    if direction == "LONG" and _rsi < _rsi1 < _rsi2:
                        rsi_triggered = True
                    elif direction == "SHORT" and _rsi > _rsi1 > _rsi2:
                        rsi_triggered = True
                    if rsi_triggered:
                        info["rsi_exit_at"] = now
                        info["rsi_exit_pnl"] = current_pnl

                # 3-drop check: maintain RSI history buffer
                history = info["rsi_history"]
                if not history or history[-1] != _rsi:
                    history.append(_rsi)
                    if len(history) > 4:
                        history.pop(0)
                if info["rsi3_exit_at"] is None and len(history) >= 4:
                    if direction == "LONG" and history[-1] < history[-2] < history[-3] < history[-4]:
                        info["rsi3_exit_at"] = now
                        info["rsi3_exit_pnl"] = current_pnl
                    elif direction == "SHORT" and history[-1] > history[-2] > history[-3] > history[-4]:
                        info["rsi3_exit_at"] = now
                        info["rsi3_exit_pnl"] = current_pnl

            # May 16: EMA13 cross counterfactual — would the EMA13_CROSS_EXIT
            # mechanism have fired during the post-exit window?
            # LONG cross-against: price < EMA13; SHORT cross-against: price > EMA13.
            # Mirrors live detection in the realtime loop (around line 5443+),
            # including strict-mode (require EMA5/EMA8 stack flip).
            # Records the FIRST moment the condition would have fired.
            if info["ema13_cross_at"] is None and pair_data and pair_data.ema13 is not None and pair_data.ema13 > 0:
                _ema13 = pair_data.ema13
                if direction == "LONG":
                    _cross_fires = price < _ema13
                else:
                    _cross_fires = price > _ema13
                if _cross_fires:
                    # Apply strict-mode gate if configured (same as live path)
                    _strict = getattr(config.trading_config.thresholds, 'ema13_cross_requires_stack_flip', False)
                    _stack_confirms = True
                    if _strict:
                        _es5 = pair_data.ema5
                        _es8 = pair_data.ema8
                        if _es5 is None or _es8 is None or _es5 <= 0 or _es8 <= 0:
                            _stack_confirms = False  # fail-closed, matches live
                        elif direction == "LONG":
                            _stack_confirms = _es5 < _es8
                        else:
                            _stack_confirms = _es5 > _es8
                    if _stack_confirms:
                        info["ema13_cross_at"] = now
                        info["ema13_cross_pnl"] = current_pnl

            # May 23: post-exit regime-flip detection. Compare live BTC regime
            # against entry_regime (captured at trade open). First transition
            # to OPPOSITE-of-direction or NEUTRAL captures the moment +
            # post-exit running P&L. Answers: "would holding past current
            # exit until regime flipped have been better?"
            if info["regime_flip_at"] is None and info.get("entry_regime"):
                _entry_reg = info["entry_regime"]
                _live_reg = _current_btc_regime
                # Define "supporting regime" by trade direction:
                #   LONG supportive: regime contains "BULL" (BULLISH, STRONG_BULL, HEALTHY_BULL, BULL_EXHAUSTED)
                #   SHORT supportive: regime contains "BEAR" (BEARISH, STRONG_BEAR, HEALTHY_BEAR, BEAR_EXHAUSTED)
                # Flip = entry was supportive AND live is NOT supportive (= NEUTRAL/CHOPPY or opposite-direction)
                if direction == "LONG":
                    _entry_supportive = "BULL" in (_entry_reg or "")
                    _live_supportive = "BULL" in (_live_reg or "")
                else:
                    _entry_supportive = "BEAR" in (_entry_reg or "")
                    _live_supportive = "BEAR" in (_live_reg or "")
                if _entry_supportive and not _live_supportive:
                    info["regime_flip_at"] = now
                    info["regime_flip_pnl"] = current_pnl
                    logger.info(f"[POST_EXIT_REGIME_FLIP] {order.pair} {order.direction}: regime flipped {_entry_reg} → {_live_reg} after exit, captured pnl={current_pnl:.4f}%")

            # Track reachable peak (best P&L while signal still active)
            if info["signal_lost_at"] is None:
                if current_pnl > info["peak_before_signal_lost"]:
                    info["peak_before_signal_lost"] = current_pnl

            # ===== LEASH SHADOW START — post-exit continuation (observation-only) =====
            # Wide leashes that didn't fire in-trade keep holding past the real exit;
            # continue them, respecting EMA13-cross and signal-lost as live backstops.
            _pe_ema5 = pair_data.ema5 if pair_data else None
            _pe_stretch = None
            if _pe_ema5 and _pe_ema5 > 0 and price > 0:
                _pe_stretch = ((price - _pe_ema5) / price * 100) if direction == 'LONG' \
                    else ((_pe_ema5 - price) / price * 100)
            _leash_update(order_id, current_pnl, peak_hint=None,
                          ema13_crossed=(info.get("ema13_cross_at") is not None),
                          signal_lost=(info.get("signal_lost_at") is not None),
                          stretch=_pe_stretch)
            # ===== LEASH SHADOW END =====

            # Post-exit phantom tick momentum checks
            tick_exit_min_profit = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_profit', 0.05)
            pe_tick_buf = info.get("tick_prices")
            if pe_tick_buf is not None:
                now_ts = time.time()
                pe_tick_buf.append((now_ts, price))
                pe_tick_buf[:] = [(t, p) for t, p in pe_tick_buf if t >= now_ts - 125]
                if current_pnl > tick_exit_min_profit:
                    for _lbl, _swin, _sdelta in _SHADOW_TICK_CONFIGS:
                        _tk = f'phantom_tick_{_lbl}_triggered'
                        if not info.get(_tk):
                            _sdeltas = _sdelta if isinstance(_sdelta, list) else [_sdelta] * len(_swin)
                            if _check_tick_momentum_fade(pe_tick_buf, now_ts, _swin, _sdeltas, direction):
                                info[_tk] = True
                                info[f'phantom_tick_{_lbl}_triggered_at'] = now
                                info[f'phantom_tick_{_lbl}_pnl'] = current_pnl

            if now >= info["tracking_until"]:
                _fd = info["fee_drag_pct"]
                if direction == "LONG":
                    peak_pnl = ((info["post_high"] - entry) / entry) * 100 - _fd
                    trough_pnl = ((info["post_low"] - entry) / entry) * 100 - _fd
                    final_pnl = ((price - entry) / entry) * 100 - _fd
                else:
                    peak_pnl = ((entry - info["post_low"]) / entry) * 100 - _fd
                    trough_pnl = ((entry - info["post_high"]) / entry) * 100 - _fd
                    final_pnl = ((entry - price) / entry) * 100 - _fd

                exit_time = info["exit_time"]
                peak_minutes = (info["peak_at"] - exit_time).total_seconds() / 60.0
                trough_minutes = (info["trough_at"] - exit_time).total_seconds() / 60.0
                sig_lost_minutes = None
                if info["signal_lost_at"]:
                    sig_lost_minutes = (info["signal_lost_at"] - exit_time).total_seconds() / 60.0
                rsi_exit_minutes = None
                if info["rsi_exit_at"]:
                    rsi_exit_minutes = (info["rsi_exit_at"] - exit_time).total_seconds() / 60.0
                rsi3_exit_minutes = None
                if info["rsi3_exit_at"]:
                    rsi3_exit_minutes = (info["rsi3_exit_at"] - exit_time).total_seconds() / 60.0
                ema13_cross_minutes = None
                if info["ema13_cross_at"]:
                    ema13_cross_minutes = (info["ema13_cross_at"] - exit_time).total_seconds() / 60.0
                sig_regained_minutes = None
                if info["signal_regained_at"]:
                    sig_regained_minutes = (info["signal_regained_at"] - exit_time).total_seconds() / 60.0

                # ===== LEASH SHADOW START — finalize at post-exit window end =====
                _leash_exits = _leash_finalize(order_id, final_pnl)
                # ===== LEASH SHADOW END =====

                try:
                    async with AsyncSessionLocal() as pe_write_db:
                        await pe_write_db.execute(
                            update(Order)
                            .where(Order.id == order_id)
                            .values(
                                # ===== LEASH SHADOW START =====
                                shadow_tight_pnl=_leash_exits.get('tight', (None, None))[0],
                                shadow_tight_reason=_leash_exits.get('tight', (None, None))[1],
                                shadow_tight_min=_leash_exits.get('tight_min'),
                                shadow_wide_pnl=_leash_exits.get('wide', (None, None))[0],
                                shadow_wide_reason=_leash_exits.get('wide', (None, None))[1],
                                shadow_wide_min=_leash_exits.get('wide_min'),
                                shadow_tierA_pnl=_leash_exits.get('tierA', (None, None))[0],
                                shadow_tierA_reason=_leash_exits.get('tierA', (None, None))[1],
                                shadow_tierA_min=_leash_exits.get('tierA_min'),
                                shadow_tierB_pnl=_leash_exits.get('tierB', (None, None))[0],
                                shadow_tierB_reason=_leash_exits.get('tierB', (None, None))[1],
                                shadow_tierB_min=_leash_exits.get('tierB_min'),
                                shadow_strpk_pnl=_leash_exits.get('strpk', (None, None))[0],
                                shadow_strpk_reason=_leash_exits.get('strpk', (None, None))[1],
                                shadow_strpk_min=_leash_exits.get('strpk_min'),
                                shadow_strpk04_pnl=_leash_exits.get('strpk04', (None, None))[0],
                                shadow_strpk04_reason=_leash_exits.get('strpk04', (None, None))[1],
                                shadow_strpk04_min=_leash_exits.get('strpk04_min'),
                                shadow_strpk03_pnl=_leash_exits.get('strpk03', (None, None))[0],
                                shadow_strpk03_reason=_leash_exits.get('strpk03', (None, None))[1],
                                shadow_strpk03_min=_leash_exits.get('strpk03_min'),
                                shadow_stren_pnl=_leash_exits.get('stren', (None, None))[0],
                                shadow_stren_reason=_leash_exits.get('stren', (None, None))[1],
                                shadow_stren_min=_leash_exits.get('stren_min'),
                                shadow_strpk_signed_pnl=_leash_exits.get('strpk_signed', (None, None))[0],
                                shadow_strpk_signed_reason=_leash_exits.get('strpk_signed', (None, None))[1],
                                shadow_strpk_signed_min=_leash_exits.get('strpk_signed_min'),
                                shadow_peak_stretch=_leash_exits.get('_peak_stretch'),
                                # ===== LEASH SHADOW END =====
                                post_exit_peak_pnl=round(peak_pnl, 4),
                                post_exit_trough_pnl=round(trough_pnl, 4),
                                post_exit_peak_minutes=round(peak_minutes, 2),
                                post_exit_trough_minutes=round(trough_minutes, 2),
                                post_exit_signal_lost_minutes=round(sig_lost_minutes, 2) if sig_lost_minutes is not None else None,
                                post_exit_pnl_at_signal_lost=round(info["pnl_at_signal_lost"], 4) if info["pnl_at_signal_lost"] is not None else None,
                                post_exit_final_pnl=round(final_pnl, 4),
                                post_exit_peak_before_signal_lost=round(info["peak_before_signal_lost"], 4) if info["signal_lost_at"] is not None else None,
                                post_exit_rsi_exit_minutes=round(rsi_exit_minutes, 2) if rsi_exit_minutes is not None else None,
                                post_exit_rsi_exit_pnl=round(info["rsi_exit_pnl"], 4) if info["rsi_exit_pnl"] is not None else None,
                                post_exit_rsi3_exit_minutes=round(rsi3_exit_minutes, 2) if rsi3_exit_minutes is not None else None,
                                post_exit_rsi3_exit_pnl=round(info["rsi3_exit_pnl"], 4) if info["rsi3_exit_pnl"] is not None else None,
                                post_exit_ema13_cross_minutes=round(ema13_cross_minutes, 2) if ema13_cross_minutes is not None else None,
                                post_exit_ema13_cross_pnl=round(info["ema13_cross_pnl"], 4) if info["ema13_cross_pnl"] is not None else None,
                                # May 23: post-exit regime flip
                                post_exit_regime_flip_at=info["regime_flip_at"],
                                post_exit_regime_flip_pnl_pct=round(info["regime_flip_pnl"], 4) if info["regime_flip_pnl"] is not None else None,
                                post_exit_signal_regained_minutes=round(sig_regained_minutes, 2) if sig_regained_minutes is not None else None,
                                post_exit_pnl_at_signal_regained=round(info["pnl_at_signal_regained"], 4) if info["pnl_at_signal_regained"] is not None else None,
                                post_exit_floor_before_signal_regain=round(info["floor_before_signal_regain"], 4) if info["floor_before_signal_regain"] is not None else None,
                                phantom_tick_a_triggered_at=info.get("phantom_tick_a_triggered_at"),
                                phantom_tick_a_pnl=round(info["phantom_tick_a_pnl"], 4) if info.get("phantom_tick_a_pnl") is not None else None,
                                phantom_tick_b_triggered_at=info.get("phantom_tick_b_triggered_at"),
                                phantom_tick_b_pnl=round(info["phantom_tick_b_pnl"], 4) if info.get("phantom_tick_b_pnl") is not None else None,
                                phantom_tick_c_triggered_at=info.get("phantom_tick_c_triggered_at"),
                                phantom_tick_c_pnl=round(info["phantom_tick_c_pnl"], 4) if info.get("phantom_tick_c_pnl") is not None else None,
                                phantom_tick_d_triggered_at=info.get("phantom_tick_d_triggered_at"),
                                phantom_tick_d_pnl=round(info["phantom_tick_d_pnl"], 4) if info.get("phantom_tick_d_pnl") is not None else None,
                                phantom_tick_e_triggered_at=info.get("phantom_tick_e_triggered_at"),
                                phantom_tick_e_pnl=round(info["phantom_tick_e_pnl"], 4) if info.get("phantom_tick_e_pnl") is not None else None,
                                phantom_tick_f_triggered_at=info.get("phantom_tick_f_triggered_at"),
                                phantom_tick_f_pnl=round(info["phantom_tick_f_pnl"], 4) if info.get("phantom_tick_f_pnl") is not None else None,
                                phantom_tick_g_triggered_at=info.get("phantom_tick_g_triggered_at"),
                                phantom_tick_g_pnl=round(info["phantom_tick_g_pnl"], 4) if info.get("phantom_tick_g_pnl") is not None else None,
                            )
                        )
                        await pe_write_db.commit()
                    sig_info = f", sig_lost={sig_lost_minutes:.1f}min" if sig_lost_minutes is not None else ""
                    rsi_info = f", rsi_exit={rsi_exit_minutes:.1f}min@{info['rsi_exit_pnl']:.4f}%" if rsi_exit_minutes is not None else ""
                    rsi3_info = f", rsi3_exit={rsi3_exit_minutes:.1f}min@{info['rsi3_exit_pnl']:.4f}%" if rsi3_exit_minutes is not None else ""
                    ema13_info = f", ema13_cross={ema13_cross_minutes:.1f}min@{info['ema13_cross_pnl']:.4f}%" if ema13_cross_minutes is not None else ""
                    logger.info(
                        f"[POST_EXIT] {info['pair']} order {order_id}: "
                        f"peak={peak_pnl:.4f}%@{peak_minutes:.1f}min trough={trough_pnl:.4f}%@{trough_minutes:.1f}min "
                        f"final={final_pnl:.4f}%{sig_info}{rsi_info}{rsi3_info}{ema13_info}"
                    )
                except Exception as e:
                    logger.error(f"[POST_EXIT] Error saving order {order_id}: {e}")

                completed.append(order_id)

        for order_id in completed:
            del self._post_exit_tracking[order_id]

    async def update_open_positions(self, db: AsyncSession) -> List[Dict]:
        """Update all open positions with current prices and check exit conditions"""
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        open_orders = result.scalars().all()
        
        updates = []
        
        for order in open_orders:
            # Use WebSocket price only -- no REST fallback to avoid rate-limit bans.
            # Open orders are always subscribed to WebSocket; if no price yet
            # (e.g. first seconds after startup), just skip and retry next cycle.
            tracker = websocket_tracker.get_tracker(order.pair)
            current_price = tracker.last_price if tracker else None

            if not current_price or current_price <= 0:
                continue
            
            order.current_price = current_price
            
            ws_high, ws_low = websocket_tracker.get_high_low(order.pair)
            
            websocket_tracker.update_price(order.pair, current_price)
            
            # Use the best of WebSocket tracking and order tracking
            if order.direction == "LONG":
                # For LONG, track highest price
                old_high = order.high_price_since_entry
                
                # DEFENSIVE: If high_price is 0, None, or invalid, initialize to entry price
                # This fixes corrupted orders from race conditions during creation
                if order.high_price_since_entry is None or order.high_price_since_entry <= 0:
                    order.high_price_since_entry = order.entry_price
                    logger.warning(f"[TRACKING_FIX] {order.pair} LONG: Initialized high_price from {old_high} to entry {order.entry_price}")
                    old_high = order.high_price_since_entry  # Update old_high for comparison
                
                # Apply normal tracking logic - only update if new price is HIGHER
                if ws_high is not None and ws_high > 0:
                    if ws_high > order.high_price_since_entry:
                        order.high_price_since_entry = ws_high
                if current_price > 0 and current_price > order.high_price_since_entry:
                    order.high_price_since_entry = current_price
                    
                # Log if high_price was updated
                if order.current_tp_level and order.current_tp_level >= 2 and old_high != order.high_price_since_entry:
                    logger.info(f"[TRACKING] {order.pair} LONG L{order.current_tp_level}: HIGH updated {old_high} -> {order.high_price_since_entry} (ws_high={ws_high})")
            else:
                # For SHORT, track lowest price
                old_low = order.low_price_since_entry
                
                # DEFENSIVE: If low_price is 0, None, or invalid, initialize to entry price
                # This fixes corrupted orders from race conditions during creation
                if order.low_price_since_entry is None or order.low_price_since_entry <= 0:
                    order.low_price_since_entry = order.entry_price
                    logger.warning(f"[TRACKING_FIX] {order.pair} SHORT: Initialized low_price from {old_low} to entry {order.entry_price}")
                    old_low = order.low_price_since_entry  # Update old_low for comparison
                
                # Apply normal tracking logic - only update if new price is LOWER
                if ws_low is not None and ws_low > 0:
                    if ws_low < order.low_price_since_entry:
                        order.low_price_since_entry = ws_low
                if current_price > 0 and current_price < order.low_price_since_entry:
                    order.low_price_since_entry = current_price
                    
                # Log if low_price was updated
                if order.current_tp_level and order.current_tp_level >= 2 and old_low != order.low_price_since_entry:
                    logger.info(f"[TRACKING] {order.pair} SHORT L{order.current_tp_level}: LOW updated {old_low} -> {order.low_price_since_entry} (ws_low={ws_low})")

            # Jun 16: flips no longer use the flip-specific 45min horizon. They flow through
            # the SAME monitor-loop timeouts as normal trades — MAX_HOLD + NO_EXPANSION below
            # (180min + BE-peak gate + signal-active reset) — then skip the momentum exit STACK
            # (the flip-skip guard just before the momentum exits). SL + ATR trailing + the flip
            # min-profit gate remain handled realtime in check_realtime_stop_loss (EMA13/runner
            # disabled). close_position still FLIP_-prefixes the reason (-> FLIP_NO_EXPANSION).

            # Get cached indicator data for this pair
            pair_result = await db.execute(
                select(PairData).where(PairData.pair == order.pair)
            )
            pair_data = pair_result.scalar_one_or_none()
            
            # Extract EMA values for trend check
            ema5 = pair_data.ema5 if pair_data else None
            ema8 = pair_data.ema8 if pair_data else None
            ema13 = pair_data.ema13 if pair_data else None
            ema20 = pair_data.ema20 if pair_data else None
            
            # Check max holding time
            max_hold = config.trading_config.investment.max_holding_time_minutes
            if max_hold > 0 and order.opened_at:
                from datetime import timezone
                opened = order.opened_at.replace(tzinfo=timezone.utc) if order.opened_at.tzinfo is None else order.opened_at
                age_minutes = (datetime.now(timezone.utc) - opened).total_seconds() / 60
                if age_minutes >= max_hold:
                    logger.info(f"[MAX_HOLD_TIME] {order.pair} {order.direction}: held {age_minutes:.0f}min >= limit {max_hold}min, force closing")
                    closed_order = await self.close_position(db, order, current_price, "MAX_HOLD_TIME")
                    if closed_order:
                        updates.append({
                            "order_id": closed_order.id,
                            "pair": closed_order.pair,
                            "action": "CLOSED",
                            "reason": "MAX_HOLD_TIME",
                            "pnl": closed_order.pnl,
                            "tp_level": order.current_tp_level or 1
                        })
                    continue
            
            # Merge realtime peak/trough from cache (may differ from DB if a
            # price spike occurred between polling cycles)
            realtime_peak = order.peak_pnl or 0
            realtime_trough = order.trough_pnl or 0
            realtime_peak_ema5_gap = order.peak_ema5_gap or 0
            cached = None  # guard: may stay None for newly-opened trades not yet cached
            async with _cache_lock:
                for _cached_iter in _open_orders_cache.get(order.pair, []):
                    if _cached_iter['id'] == order.id:
                        cached = _cached_iter
                        realtime_peak = max(realtime_peak, cached.get('peak_pnl', 0))
                        realtime_trough = min(realtime_trough, cached.get('trough_pnl', 0))
                        realtime_peak_ema5_gap = max(realtime_peak_ema5_gap, cached.get('peak_ema5_gap', 0))
                        break
            
            # Check NO_EXPANSION: close stale trades that never expanded
            no_exp_minutes = config.trading_config.investment.no_expansion_minutes
            if no_exp_minutes > 0 and order.opened_at:
                from datetime import timezone
                # Use last reset time if available, otherwise use opened_at
                ref_time = order.no_expansion_last_check or order.opened_at
                ref_time = ref_time.replace(tzinfo=timezone.utc) if ref_time.tzinfo is None else ref_time
                age_minutes = (datetime.now(timezone.utc) - ref_time).total_seconds() / 60
                if age_minutes >= no_exp_minutes:
                    conf_config = config.trading_config.confidence_levels.get(order.confidence)
                    if conf_config:
                        be_l1_trigger = conf_config.be_level1_trigger
                        be_l1_offset = conf_config.be_level1_offset
                        if order.direction == "LONG":
                            raw_pnl = (current_price - order.entry_price) * order.quantity
                        else:
                            raw_pnl = (order.entry_price - current_price) * order.quantity
                        est_exit_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                        net_pnl = raw_pnl - (order.entry_fee or 0) - est_exit_fee
                        entry_notional = order.entry_price * order.quantity if order.quantity > 0 else 1
                        cur_pnl_pct = (net_pnl / entry_notional) * 100
                        if realtime_peak < be_l1_trigger and cur_pnl_pct < be_l1_offset:
                            # Re-check if buy signal is still active before closing
                            if pair_data and pair_data.signal == order.direction:
                                order.no_expansion_last_check = datetime.now(timezone.utc)
                                logger.info(f"[NO_EXPANSION_RESET] {order.pair} {order.direction}: signal still {order.direction}, resetting timer (was {age_minutes:.0f}min)")
                                continue
                            logger.info(f"[NO_EXPANSION] {order.pair} {order.direction}: {age_minutes:.0f}min, peak={realtime_peak:.4f}% < BE_L1={be_l1_trigger}%, cur={cur_pnl_pct:.4f}% < BE_L1_off={be_l1_offset}%")
                            closed_order = await self.close_position(db, order, current_price, "NO_EXPANSION")
                            if closed_order:
                                updates.append({
                                    "order_id": closed_order.id,
                                    "pair": closed_order.pair,
                                    "action": "CLOSED",
                                    "reason": "NO_EXPANSION",
                                    "pnl": closed_order.pnl,
                                    "tp_level": order.current_tp_level or 1
                                })
                            continue

            # Flips skip the momentum exit STACK below (base −0.70 SL, ATR trailing, EMA13,
            # signal-lost, etc.) — those are handled realtime in check_realtime_stop_loss with
            # EMA13 + short-runner disabled. Flips DID just run the shared MAX_HOLD + NO_EXPANSION
            # above, so a stale flip closes on the SAME no-expansion as a normal trade (Jun 16).
            if (order.entry_strategy or "").startswith("FLIP:"):
                # Jun 16: per-source flip exit. FAN flips with runner_strpk exit via the SHORT
                # runner stretch-trail (same engine/params as normal shorts) — fired HERE in the
                # monitor where ema5 is fresh (the realtime path lacks it). Armed once peak ≥ arm;
                # close when live stretch retraces to ≤ K × peak stretch. The realtime tight-trail
                # is suppressed for armed FAN-strpk flips (see :8823) so this can run. Fail-open.
                try:
                    _fsrc = (order.entry_strategy or "")[5:]
                    _rt_th = config.trading_config.thresholds
                    # Jun 16: strpk now covers ALL flip shorts — FAN via flip_fan_runner_strpk,
                    # the other sleeves (PAIR_RSI_OB / LONG_UNMATCHED_ONLY) via flip_runner_strpk_shorts.
                    _strpk_on = ((_fsrc == "FAN_RATIO_GATE" and getattr(_rt_th, 'flip_fan_runner_strpk', False))
                                 or (_fsrc != "FAN_RATIO_GATE" and getattr(_rt_th, 'flip_runner_strpk_shorts', False)))
                    if (_strpk_on
                            and order.direction == "SHORT" and ema5 and ema5 > 0 and current_price > 0):
                        _fl_stretch = ((ema5 - current_price) / current_price) * 100.0
                        if getattr(order, 'runner_peak_stretch', None) is None or _fl_stretch > order.runner_peak_stretch:
                            order.runner_peak_stretch = _fl_stretch
                        _fl_arm = float(getattr(_rt_th, 'runner_trail_short_arm_peak', 0.45) or 0.45)
                        _fl_k = float(getattr(_rt_th, 'runner_trail_short_k', 0.5) or 0.5)
                        _fl_pk = order.runner_peak_stretch or 0.0
                        if realtime_peak >= _fl_arm and _fl_pk > 0 and _fl_stretch <= _fl_pk * _fl_k:
                            logger.info(f"[FLIP_RUNNER_TRAIL] {order.pair} SHORT: stretch {_fl_stretch:.3f} <= {_fl_k}x peak {_fl_pk:.3f} (armed peak={realtime_peak:.2f}%) -> close")
                            closed_order = await self.close_position(db, order, current_price, "RUNNER_TRAIL")
                            if closed_order:
                                updates.append({"order_id": closed_order.id, "pair": closed_order.pair,
                                                "action": "CLOSED", "reason": closed_order.close_reason,
                                                "pnl": closed_order.pnl, "tp_level": order.current_tp_level or 1})
                            continue
                except Exception as _fl_rt_err:
                    logger.error(f"[FLIP_RUNNER_TRAIL] {order.pair}: {_fl_rt_err}")
                await db.commit()
                continue

            # Compute current P&L % for exit checks
            if order.direction == "LONG":
                _raw_pnl = (current_price - order.entry_price) * order.quantity
            else:
                _raw_pnl = (order.entry_price - current_price) * order.quantity
            _est_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
            _net_pnl = _raw_pnl - (order.entry_fee or 0) - _est_fee
            _notional = order.entry_price * order.quantity if order.quantity > 0 else 1
            pnl_pct = (_net_pnl / _notional) * 100

            # In-trade RSI pattern tracking (first occurrence, no P&L threshold)
            if pair_data and pair_data.rsi is not None:
                _trk_rsi = pair_data.rsi
                _trk_rsi1 = pair_data.rsi_prev1
                _trk_rsi2 = pair_data.rsi_prev2
                from datetime import timezone as _tz
                _trk_opened = order.opened_at.replace(tzinfo=_tz.utc) if order.opened_at and order.opened_at.tzinfo is None else order.opened_at
                _trk_age = (datetime.now(_tz.utc) - _trk_opened).total_seconds() / 60 if _trk_opened else 0

                # 2-drop detection
                if order.first_rsi2_pnl is None and _trk_rsi1 is not None and _trk_rsi2 is not None:
                    rsi2_fired = False
                    if order.direction == "LONG" and _trk_rsi < _trk_rsi1 < _trk_rsi2:
                        rsi2_fired = True
                    elif order.direction == "SHORT" and _trk_rsi > _trk_rsi1 > _trk_rsi2:
                        rsi2_fired = True
                    if rsi2_fired:
                        order.first_rsi2_pnl = round(pnl_pct, 4)
                        order.first_rsi2_minutes = round(_trk_age, 2)

                # 3-drop detection via rolling history buffer
                oid = order.id
                if oid not in self._rsi3_history:
                    self._rsi3_history[oid] = []
                hist = self._rsi3_history[oid]
                if not hist or hist[-1] != _trk_rsi:
                    hist.append(_trk_rsi)
                    if len(hist) > 4:
                        hist.pop(0)
                if order.first_rsi3_pnl is None and len(hist) >= 4:
                    if order.direction == "LONG" and hist[-1] < hist[-2] < hist[-3] < hist[-4]:
                        order.first_rsi3_pnl = round(pnl_pct, 4)
                        order.first_rsi3_minutes = round(_trk_age, 2)
                    elif order.direction == "SHORT" and hist[-1] > hist[-2] > hist[-3] > hist[-4]:
                        order.first_rsi3_pnl = round(pnl_pct, 4)
                        order.first_rsi3_minutes = round(_trk_age, 2)

            # Regime Neutral tracking: record when regime goes NEUTRAL, comes back, or goes opposite
            _favorable_regime = "BULLISH" if order.direction == "LONG" else "BEARISH"
            _opposite_regime = "BEARISH" if order.direction == "LONG" else "BULLISH"
            if cached is not None:
                if _current_btc_regime == "NEUTRAL" and not cached.get('regime_neutral_hit'):
                    cached['regime_neutral_hit'] = True
                    cached['regime_neutral_hit_at'] = datetime.utcnow()
                    cached['regime_neutral_pnl'] = round(pnl_pct, 4)
                    logger.info(f"[REGIME_NEUTRAL] {order.pair} {order.direction}: regime went NEUTRAL (pnl={pnl_pct:.4f}%)")
                elif cached.get('regime_neutral_hit'):
                    if _current_btc_regime == _favorable_regime and not cached.get('regime_comeback_at'):
                        cached['regime_comeback_at'] = datetime.utcnow()
                        cached['regime_comeback_pnl'] = round(pnl_pct, 4)
                        logger.info(f"[REGIME_COMEBACK] {order.pair} {order.direction}: regime back to {_favorable_regime} (pnl={pnl_pct:.4f}%)")
                    elif _current_btc_regime == _opposite_regime and not cached.get('regime_opposite_at'):
                        cached['regime_opposite_at'] = datetime.utcnow()
                        cached['regime_opposite_pnl'] = round(pnl_pct, 4)
                        logger.info(f"[REGIME_OPPOSITE] {order.pair} {order.direction}: regime went {_opposite_regime} (pnl={pnl_pct:.4f}%)")

            # Phantom Regime Change Exit shadow tracking (added May 11 UTC-3):
            # Capture the FIRST cycle where BTC regime flips opposite to trade direction,
            # regardless of whether the real exit is enabled. Enables counterfactual
            # evaluation of regime_change_exit_enabled before flipping it on.
            # May 20 fix: guard against cached=None (newly-opened trade not yet in cache,
            # or post-restart before update_orders_cache runs). Previously crashed silently
            # if regime was BULLISH/BEARISH and cached was None, contributing to the
            # 1/278 capture rate.
            if cached is not None and _current_btc_regime != "NEUTRAL" and not cached.get('phantom_regime_change_triggered'):
                _phantom_regime_conflict = (
                    (order.direction == "LONG" and _current_btc_regime == "BEARISH") or
                    (order.direction == "SHORT" and _current_btc_regime == "BULLISH")
                )
                if _phantom_regime_conflict:
                    cached['phantom_regime_change_triggered'] = True
                    cached['phantom_regime_change_exit_triggered_at'] = datetime.utcnow()
                    cached['phantom_regime_change_exit_pnl'] = round(pnl_pct, 4)
                    logger.info(f"[PHANTOM_REGIME_CHANGE] {order.pair} {order.direction}: regime flipped to {_current_btc_regime}, captured pnl={pnl_pct:.4f}% for counterfactual")

            # REGIME_CHANGE: close when BTC macro regime flips against trade direction
            regime_exit_enabled = getattr(config.trading_config.thresholds, 'regime_change_exit_enabled', True)
            if regime_exit_enabled and _current_btc_regime != "NEUTRAL":
                regime_conflicts = (
                    (order.direction == "LONG" and _current_btc_regime == "BEARISH") or
                    (order.direction == "SHORT" and _current_btc_regime == "BULLISH")
                )
                if regime_conflicts:
                    tp_level = order.current_tp_level or 1
                    logger.info(f"[REGIME_CHANGE] {order.pair} {order.direction} L{tp_level}: BTC regime now {_current_btc_regime}, closing (pnl={pnl_pct:.4f}%)")
                    closed_order = await self.close_position(db, order, current_price, f"REGIME_CHANGE L{tp_level}")
                    if closed_order:
                        updates.append({
                            "order_id": closed_order.id,
                            "pair": closed_order.pair,
                            "action": "CLOSED",
                            "reason": f"REGIME_CHANGE L{tp_level}",
                            "pnl": closed_order.pnl,
                            "tp_level": tp_level
                        })
                    continue

            # Phase 1d-ExitTest (May 2): RSI Handoff exit — fires when:
            #   - rsi_handoff_active=True (master toggle, default OFF)
            #   - current_tp_level >= rsi_handoff_level (default L3)
            #   - 2-drop RSI sequence confirmed (any P&L, including profitable)
            # This is the WINNER-EXIT counterpart to rsi_momentum_exit (which is
            # the LOSS-CUTTING tool). Hypothesis: past the handoff level, RSI
            # exhaustion is a better exit signal than trailing pullback.
            handoff_active = getattr(config.trading_config.thresholds, 'rsi_handoff_active', False)
            handoff_level = getattr(config.trading_config.thresholds, 'rsi_handoff_level', 3)
            if handoff_active and (order.current_tp_level or 1) >= handoff_level and pair_data:
                _rsi_h = pair_data.rsi
                _rsi_h1 = pair_data.rsi_prev1
                _rsi_h2 = pair_data.rsi_prev2
                if _rsi_h is not None and _rsi_h1 is not None and _rsi_h2 is not None:
                    handoff_fading = False
                    if order.direction == "LONG" and _rsi_h < _rsi_h1 < _rsi_h2:
                        handoff_fading = True
                    elif order.direction == "SHORT" and _rsi_h > _rsi_h1 > _rsi_h2:
                        handoff_fading = True
                    if handoff_fading:
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[RSI_HANDOFF_EXIT] {order.pair} {order.direction} L{tp_level}: RSI fading ({_rsi_h2:.1f}->{_rsi_h1:.1f}->{_rsi_h:.1f}), pnl={pnl_pct:.4f}% (handoff_level={handoff_level})")
                        closed_order = await self.close_position(db, order, current_price, f"RSI_HANDOFF_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"RSI_HANDOFF_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # EMA Stack Cross Exit (May 6) — closes when EMA5 crosses EMA8 against
            # trade direction past `ema_stack_cross_exit_level`. Mirrors RSI Handoff
            # but uses the entry-signal-inverted condition. Faster than RSI 2-drop
            # (~5min vs ~15min). Suppresses trailing past level (Option A).
            es_active = getattr(config.trading_config.thresholds, 'ema_stack_cross_exit_enabled', False)
            es_level = getattr(config.trading_config.thresholds, 'ema_stack_cross_exit_level', 2)
            if es_active and (order.current_tp_level or 1) >= es_level and pair_data:
                _es5 = pair_data.ema5
                _es8 = pair_data.ema8
                if _es5 is not None and _es8 is not None and _es5 > 0 and _es8 > 0:
                    es_inverted = False
                    if order.direction == "LONG" and _es5 < _es8:
                        es_inverted = True
                    elif order.direction == "SHORT" and _es5 > _es8:
                        es_inverted = True
                    if es_inverted:
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[EMA_STACK_CROSS_EXIT] {order.pair} {order.direction} L{tp_level}: stack inverted (ema5={_es5:.6f} {'<' if order.direction == 'LONG' else '>'} ema8={_es8:.6f}), pnl={pnl_pct:.4f}% (level={es_level})")
                        closed_order = await self.close_position(db, order, current_price, f"EMA_STACK_CROSS_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"EMA_STACK_CROSS_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # RSI Momentum Exit: two consecutive RSI drops (LONG) or rises (SHORT) within P&L range
            rsi_exit_enabled = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_enabled', False)
            rsi_exit_min_profit = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_min_profit', 0.05)
            rsi_exit_max_profit = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_max_profit', 999.0)
            if rsi_exit_enabled and pair_data and pnl_pct > rsi_exit_min_profit and pnl_pct < rsi_exit_max_profit:
                _rsi = pair_data.rsi
                _rsi1 = pair_data.rsi_prev1
                _rsi2 = pair_data.rsi_prev2
                if _rsi is not None and _rsi1 is not None and _rsi2 is not None:
                    rsi_fading = False
                    if order.direction == "LONG" and _rsi < _rsi1 < _rsi2:
                        rsi_fading = True
                    elif order.direction == "SHORT" and _rsi > _rsi1 > _rsi2:
                        rsi_fading = True
                    if rsi_fading:
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[RSI_MOMENTUM_EXIT] {order.pair} {order.direction} L{tp_level}: RSI fading ({_rsi2:.1f}->{_rsi1:.1f}->{_rsi:.1f}), pnl={pnl_pct:.4f}% (range {rsi_exit_min_profit}% to {rsi_exit_max_profit}%)")
                        closed_order = await self.close_position(db, order, current_price, f"RSI_MOMENTUM_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"RSI_MOMENTUM_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # P&L trailing stop: only MOMENTUM_EXIT (signal lost). Skipped when signal active + RSI exit enabled.
            pnl_trigger = getattr(config.trading_config.thresholds, 'pnl_trailing_trigger', 0.0)
            pnl_ratio = getattr(config.trading_config.thresholds, 'pnl_trailing_ratio', 0.0)
            if pnl_trigger > 0 and pnl_ratio > 0 and realtime_peak >= pnl_trigger:
                signal_active = pair_data and is_signal_direction_active(
                    order.direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
                )
                if signal_active and rsi_exit_enabled:
                    pass  # RSI momentum exit handles signal-active exits
                else:
                    pnl_exit_level = realtime_peak * pnl_ratio
                    if pnl_pct <= pnl_exit_level:
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[MOMENTUM_EXIT] {order.pair} {order.direction} L{tp_level}: pnl={pnl_pct:.4f}% <= peak={realtime_peak:.4f}%*{pnl_ratio}(no-signal)={pnl_exit_level:.4f}%")
                        closed_order = await self.close_position(db, order, current_price, f"MOMENTUM_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"MOMENTUM_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # SLOPE_EXIT: EMA5 slope reversal
            ema5_slope_enabled = getattr(config.trading_config.thresholds, 'ema5_slope_exit_enabled', False)
            if ema5_slope_enabled and pair_data and pair_data.ema5 is not None:
                if pair_data.ema5_prev3 is not None and pair_data.ema5_prev3 != 0:
                    ema5_slope_pct = ((pair_data.ema5 - pair_data.ema5_prev3) / pair_data.ema5_prev3) * 100
                    slope_threshold = getattr(config.trading_config.thresholds, 'ema5_slope_threshold', 0.0)
                    if (order.direction == "LONG" and ema5_slope_pct <= slope_threshold) or \
                       (order.direction == "SHORT" and ema5_slope_pct >= -slope_threshold):
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[SLOPE_EXIT] {order.pair} {order.direction} L{tp_level}: slope={ema5_slope_pct:.4f}% (threshold={slope_threshold}%)")
                        closed_order = await self.close_position(db, order, current_price, f"SLOPE_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"SLOPE_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # SIGNAL_LOST: full signal no longer matches entry direction
            # Flag system: instead of exiting in primary range, flag the trade and let it run.
            # Security gap at [-0.9, -0.7] is the hard exit for flagged trades.
            signal_lost_enabled = getattr(config.trading_config.thresholds, 'signal_lost_exit_enabled', True)
            signal_dir_active = pair_data and is_signal_direction_active(
                order.direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
            )
            if signal_lost_enabled and pair_data and not signal_dir_active:
                signal_lost_min = getattr(config.trading_config.thresholds, 'signal_lost_min_profit', 0.03)
                signal_lost_max = getattr(config.trading_config.thresholds, 'signal_lost_max_profit', 999.0)
                if order.direction == "LONG":
                    sl_raw_pnl = (current_price - order.entry_price) * order.quantity
                else:
                    sl_raw_pnl = (order.entry_price - current_price) * order.quantity
                sl_exit_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                sl_net_pnl = sl_raw_pnl - (order.entry_fee or 0) - sl_exit_fee
                sl_notional = order.entry_price * order.quantity if order.quantity > 0 else 1
                sl_pnl_pct = (sl_net_pnl / sl_notional) * 100
                conf_config = config.trading_config.confidence_levels.get(order.confidence)
                sl_tp_target = order.dynamic_tp_target if order.dynamic_tp_target is not None else (conf_config.tp_min if conf_config else 0.2)

                _flag_enabled = getattr(config.trading_config.thresholds, 'signal_lost_flag_enabled', True)

                # Check if trade is already flagged (from cache)
                _is_flagged = False
                async with _cache_lock:
                    for _ci in _open_orders_cache.get(order.pair, []):
                        if _ci['id'] == order.id:
                            _is_flagged = _ci.get('signal_lost_flagged', False)
                            break

                if sl_pnl_pct >= signal_lost_min and sl_pnl_pct <= signal_lost_max and sl_pnl_pct < sl_tp_target:
                    if _flag_enabled and not _is_flagged:
                        # Flag system ON: flag the trade instead of exiting
                        tp_level = order.current_tp_level or 1
                        flag_time = datetime.utcnow()
                        async with _cache_lock:
                            for _ci in _open_orders_cache.get(order.pair, []):
                                if _ci['id'] == order.id:
                                    _ci['signal_lost_flagged'] = True
                                    _ci['signal_lost_flag_pnl'] = round(sl_pnl_pct, 4)
                                    _ci['signal_lost_flagged_at'] = flag_time
                                    _ci['fl1_origin'] = "SIGNAL_LOST"
                                    break
                        order.signal_lost_flagged = True
                        order.signal_lost_flag_pnl = round(sl_pnl_pct, 4)
                        order.signal_lost_flagged_at = flag_time
                        order.fl1_origin = "SIGNAL_LOST"
                        await db.commit()
                        logger.info(f"[SIGNAL_LOST_FLAG] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% — FLAGGED[SIGNAL_LOST] (persisted to DB), signal='{pair_data.signal}'")
                        continue
                    elif not _flag_enabled:
                        # Flag system OFF: original behavior — exit immediately
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[SIGNAL_LOST] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% >= min {signal_lost_min}%, signal now '{pair_data.signal}' != '{order.direction}'")
                        closed_order = await self.close_position(db, order, current_price, f"SIGNAL_LOST L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"SIGNAL_LOST L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

                # Security gap: flagged trade with signal still lost
                # FL1[WIDE_SL] trades bypass the security gap entirely — they get flagged AT -0.9%,
                # so the gap [-0.9, -0.8] would trigger immediately and collapse them into FL2 on the
                # next tick. WIDE_SL trades should run to emergency backstop (-1.2%), trailing recovery,
                # signal regain, or max hold time — nothing else.
                _fl1_origin_check = None
                async with _cache_lock:
                    for _ci in _open_orders_cache.get(order.pair, []):
                        if _ci['id'] == order.id:
                            _fl1_origin_check = _ci.get('fl1_origin') or getattr(order, 'fl1_origin', None)
                            break
                if _fl1_origin_check is None:
                    _fl1_origin_check = getattr(order, 'fl1_origin', None)
                security_gap_min = getattr(config.trading_config.thresholds, 'signal_lost_flag_security_min', -0.9)
                security_gap_max = getattr(config.trading_config.thresholds, 'signal_lost_flag_security_max', -0.7)
                if _is_flagged and _fl1_origin_check != "WIDE_SL" and sl_pnl_pct >= security_gap_min and sl_pnl_pct <= security_gap_max:
                    tp_level = order.current_tp_level or 1
                    _fl2_enabled = getattr(config.trading_config.thresholds, 'fl2_enabled', True)
                    # Check if already FL2-flagged (from cache)
                    _is_fl2 = False
                    async with _cache_lock:
                        for _ci in _open_orders_cache.get(order.pair, []):
                            if _ci['id'] == order.id:
                                _is_fl2 = _ci.get('fl2_flagged', False)
                                break
                    if _fl2_enabled and not _is_fl2:
                        # Promote to FL2 instead of closing — let it try to recover
                        fl2_time = datetime.utcnow()
                        async with _cache_lock:
                            for _ci in _open_orders_cache.get(order.pair, []):
                                if _ci['id'] == order.id:
                                    _ci['fl2_flagged'] = True
                                    _ci['fl2_flagged_at'] = fl2_time
                                    _ci['fl2_flag_pnl'] = round(sl_pnl_pct, 4)
                                    break
                        order.fl2_flagged = True
                        order.fl2_flagged_at = fl2_time
                        order.fl2_flag_pnl = round(sl_pnl_pct, 4)
                        await db.commit()
                        logger.info(f"[FL2_FLAG] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% — promoted to FL2 (origin={order.fl1_origin or 'SIGNAL_LOST'}), recovery_target={getattr(config.trading_config.thresholds, 'fl2_recovery_target', -0.4)}%, deep_stop={getattr(config.trading_config.thresholds, 'fl2_deep_stop', -1.0)}%")
                        continue
                    if _is_fl2:
                        # Already FL2-flagged (likely promoted by realtime in the last few ms).
                        # Do NOT close as FL_SIGNAL_LOST — let the FL2 monitor handle recovery/deep_stop.
                        logger.debug(f"[FL2_HOLD] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% — already FL2-flagged, deferring to FL2 monitor")
                        continue
                    # FL2 disabled — original behavior: close here
                    logger.info(f"[FL_SIGNAL_LOST] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% hit security gap [{security_gap_min}, {security_gap_max}]")
                    closed_order = await self.close_position(db, order, current_price, f"FL_SIGNAL_LOST L{tp_level}")
                    if closed_order:
                        updates.append({
                            "order_id": closed_order.id,
                            "pair": closed_order.pair,
                            "action": "CLOSED",
                            "reason": f"FL_SIGNAL_LOST L{tp_level}",
                            "pnl": closed_order.pnl,
                            "tp_level": tp_level
                        })
                    continue

            # Check exit conditions (including fees for accurate SL/TP)
            is_signal_active = (pair_data and is_signal_direction_active(
                order.direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
            )) if pair_data else False
            exit_conf_config = config.trading_config.confidence_levels.get(order.confidence)
            # Jun 1: runner stretch-trail — track live |price−EMA5| stretch + peak.
            # Only meaningful for LONG (the runner trail is LONG-scoped); cheap to
            # always compute. Peak persisted on the Order so it survives restart.
            _rt_stretch = None
            try:
                if ema5 and ema5 > 0 and current_price and current_price > 0:
                    # SIGNED stretch, EXACT match to the shadow strpk formula
                    # (/current_price denominator; + = favorable extension).
                    _rt_stretch = (((current_price - ema5) / current_price) * 100.0
                                   if order.direction == "LONG"
                                   else ((ema5 - current_price) / current_price) * 100.0)
                    _rt_prev_peak = getattr(order, 'runner_peak_stretch', None)
                    if _rt_prev_peak is None or _rt_stretch > _rt_prev_peak:
                        order.runner_peak_stretch = _rt_stretch
            except Exception:
                _rt_stretch = None
            exit_result = check_exit_conditions(
                direction=order.direction,
                entry_price=order.entry_price,
                current_price=current_price,
                leverage=order.leverage,
                confidence=order.confidence,
                peak_pnl=realtime_peak,
                trough_pnl=realtime_trough,
                quantity=order.quantity,
                entry_fee=order.entry_fee,
                investment=order.investment,
                high_price=order.high_price_since_entry,
                low_price=order.low_price_since_entry,
                # Pass indicators for dynamic TP
                ema5=ema5,
                ema8=ema8,
                ema13=ema13,
                ema20=ema20,
                current_tp_level=order.current_tp_level or 1,
                dynamic_tp_target=order.dynamic_tp_target,
                signal_active=is_signal_active,
                tp_trailing_enabled=exit_conf_config.tp_trailing_enabled if exit_conf_config else True,
                entry_atr_pct=getattr(order, 'entry_atr_pct', None),  # May 7 Phase 1: ATR-normalized trailing
                current_stretch=_rt_stretch,  # Jun 1: runner stretch-trail
                peak_stretch=getattr(order, 'runner_peak_stretch', None),  # Jun 1: runner stretch-trail
                is_flip=(order.entry_strategy or "").startswith("FLIP:"),  # Jun 14: runner-trail off for flips → normal trailing
            )

            order.peak_pnl = exit_result.get("peak_pnl", order.peak_pnl)
            order.trough_pnl = exit_result.get("trough_pnl", order.trough_pnl)
            # Jun 8: trailing min-profit gate — record the would-have-cut pnl the FIRST
            # time the gate suppresses a trailing fire (phantom CF: cut vs held-to-exit).
            _ts_supp = exit_result.get("trail_suppressed_pnl")
            if _ts_supp is not None and getattr(order, 'phantom_trail_suppress_pnl', None) is None:
                order.phantom_trail_suppress_pnl = float(_ts_supp)
                order.phantom_trail_suppress_at = datetime.utcnow()
            # May 14 — sync DB peak/trough updates back to realtime cache.
            # Without this, the realtime callback's phantom BE / BE / FL checks
            # use a stale cached peak/trough and miss extremes that monitor saw
            # between WebSocket ticks. Caused Phantom BE 0.20/0.05 to never arm
            # on trades where peak was reached between realtime ticks.
            if cached is not None:
                async with _cache_lock:
                    _db_peak = order.peak_pnl
                    _db_trough = order.trough_pnl
                    if _db_peak is not None and _db_peak > cached.get('peak_pnl', 0):
                        cached['peak_pnl'] = _db_peak
                    if _db_trough is not None and _db_trough < cached.get('trough_pnl', 0):
                        cached['trough_pnl'] = _db_trough
            reason = exit_result.get("reason")

            # ─── FL1[WIDE_SL] interception: convert STOP_LOSS_WIDE into a flag instead of closing ───
            _fl1_wide_enabled = getattr(config.trading_config.thresholds, 'fl1_for_wide_sl_enabled', True)
            if (exit_result.get("should_close")
                    and isinstance(reason, str)
                    and reason.startswith("STOP_LOSS_WIDE")
                    and _fl1_wide_enabled
                    and not order.signal_lost_flagged):
                tp_level = exit_result.get("tp_level", order.current_tp_level or 1)
                # Compute actual P&L % at this moment (with fees) for the flag record
                _entry_notional_w = order.entry_price * order.quantity
                _exit_fee_w = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                if order.direction == "LONG":
                    _pnl_w = (current_price - order.entry_price) * order.quantity - (order.entry_fee or 0) - _exit_fee_w
                else:
                    _pnl_w = (order.entry_price - current_price) * order.quantity - (order.entry_fee or 0) - _exit_fee_w
                _pnl_pct_w = round((_pnl_w / _entry_notional_w) * 100, 4) if _entry_notional_w else 0.0
                flag_time_w = datetime.utcnow()
                async with _cache_lock:
                    for _ci in _open_orders_cache.get(order.pair, []):
                        if _ci['id'] == order.id:
                            _ci['signal_lost_flagged'] = True
                            _ci['signal_lost_flag_pnl'] = _pnl_pct_w
                            _ci['signal_lost_flagged_at'] = flag_time_w
                            _ci['fl1_origin'] = "WIDE_SL"
                            break
                order.signal_lost_flagged = True
                order.signal_lost_flag_pnl = _pnl_pct_w
                order.signal_lost_flagged_at = flag_time_w
                order.fl1_origin = "WIDE_SL"
                await db.commit()
                logger.info(f"[FL1_WIDE_SL] {order.pair} {order.direction} L{tp_level}: pnl={_pnl_pct_w:.4f}% — flagged from STOP_LOSS_WIDE (origin=WIDE_SL), backstop={getattr(config.trading_config.thresholds, 'fl1_wide_sl_backstop', -1.2)}%")
                continue

            # ─── FL1[WIDE_SL] emergency backstop + FL2 recovery/deep_stop monitors ───
            if order.signal_lost_flagged:
                # Compute current P&L % with fees
                _entry_notional_m = order.entry_price * order.quantity
                _exit_fee_m = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                if order.direction == "LONG":
                    _pnl_m = (current_price - order.entry_price) * order.quantity - (order.entry_fee or 0) - _exit_fee_m
                else:
                    _pnl_m = (order.entry_price - current_price) * order.quantity - (order.entry_fee or 0) - _exit_fee_m
                _pnl_pct_m = (_pnl_m / _entry_notional_m) * 100 if _entry_notional_m else 0.0
                _tp_level_m = order.current_tp_level or 1

                if order.fl2_flagged:
                    # FL2 monitor: recover → FL_RECOVERED, fall → FL_DEEP_STOP
                    _fl2_recovery = getattr(config.trading_config.thresholds, 'fl2_recovery_target', -0.4)
                    _fl2_deep = getattr(config.trading_config.thresholds, 'fl2_deep_stop', -1.0)
                    if _pnl_pct_m >= _fl2_recovery:
                        logger.info(f"[FL_RECOVERED] {order.pair} {order.direction} L{_tp_level_m}: pnl={_pnl_pct_m:.4f}% >= fl2_recovery_target={_fl2_recovery}%")
                        closed_order = await self.close_position(db, order, current_price, f"FL_RECOVERED L{_tp_level_m}")
                        if closed_order:
                            updates.append({"order_id": closed_order.id, "pair": closed_order.pair, "action": "CLOSED", "reason": f"FL_RECOVERED L{_tp_level_m}", "pnl": closed_order.pnl, "tp_level": _tp_level_m})
                        continue
                    if _pnl_pct_m <= _fl2_deep:
                        logger.info(f"[FL_DEEP_STOP] {order.pair} {order.direction} L{_tp_level_m}: pnl={_pnl_pct_m:.4f}% <= fl2_deep_stop={_fl2_deep}%")
                        closed_order = await self.close_position(db, order, current_price, f"FL_DEEP_STOP L{_tp_level_m}")
                        if closed_order:
                            updates.append({"order_id": closed_order.id, "pair": closed_order.pair, "action": "CLOSED", "reason": f"FL_DEEP_STOP L{_tp_level_m}", "pnl": closed_order.pnl, "tp_level": _tp_level_m})
                        continue
                    # FL2 middle zone (between recovery and deep stop) — suppress any STOP_LOSS(_WIDE) close.
                    # Only FL_RECOVERED, FL_DEEP_STOP, or max hold time should exit a FL2 trade.
                    if exit_result.get("should_close") and isinstance(reason, str) and reason.startswith("STOP_LOSS"):
                        logger.debug(f"[FL2_HOLD] {order.pair} {order.direction} L{_tp_level_m}: pnl={_pnl_pct_m:.4f}% — suppressing {reason}, FL2 monitor holds to recovery/deep_stop")
                        await db.commit()
                        continue
                elif (order.fl1_origin or "") == "WIDE_SL":
                    # FL1[WIDE_SL] emergency backstop — fires at fl1_wide_sl_backstop (-1.2%)
                    _fl1_backstop = getattr(config.trading_config.thresholds, 'fl1_wide_sl_backstop', -1.2)
                    if _pnl_pct_m <= _fl1_backstop:
                        logger.info(f"[FL_EMERGENCY_SL] {order.pair} {order.direction} L{_tp_level_m}: pnl={_pnl_pct_m:.4f}% <= fl1_wide_sl_backstop={_fl1_backstop}%")
                        closed_order = await self.close_position(db, order, current_price, f"FL_EMERGENCY_SL L{_tp_level_m}")
                        if closed_order:
                            updates.append({"order_id": closed_order.id, "pair": closed_order.pair, "action": "CLOSED", "reason": f"FL_EMERGENCY_SL L{_tp_level_m}", "pnl": closed_order.pnl, "tp_level": _tp_level_m})
                        continue
                    # WIDE_SL flagged but not at backstop yet — suppress any STOP_LOSS(_WIDE) close from check_exit_conditions.
                    # The trade should only exit via backstop, trailing recovery, signal regain + trailing, or max hold time.
                    if exit_result.get("should_close") and isinstance(reason, str) and reason.startswith("STOP_LOSS"):
                        logger.debug(f"[FL1_WIDE_SL_HOLD] {order.pair} {order.direction} L{_tp_level_m}: pnl={_pnl_pct_m:.4f}% — suppressing {reason}, runway to backstop={_fl1_backstop}%")
                        await db.commit()
                        continue

            if exit_result.get("should_close"):
                closed_order = await self.close_position(db, order, current_price, reason)
                if closed_order:
                    updates.append({
                        "order_id": closed_order.id,
                        "pair": closed_order.pair,
                        "action": "CLOSED",
                        "reason": reason,
                        "pnl": closed_order.pnl,
                        "tp_level": exit_result.get("tp_level", 1)
                    })
            elif reason == "EXTEND_TP":
                # Extend TP target - update order fields
                new_tp_level = exit_result.get("new_tp_level", order.current_tp_level + 1)
                new_tp_target = exit_result.get("new_tp_target")

                logger.info(f"[EXTEND_TP] {order.pair} {order.direction}: L{order.current_tp_level} -> L{new_tp_level} (target: {new_tp_target:.4f}%)")

                order.current_tp_level = new_tp_level
                order.dynamic_tp_target = new_tp_target

                # NOTE: Do NOT reset high/low tracking when extending TP!
                # We want to keep the best price ever seen for trailing stop calculation.
                # Otherwise, if price reverses after extension, we lose the profit reference.

                # Sync cache so real-time WebSocket exits use the correct level
                async with _cache_lock:
                    for cached_order in _open_orders_cache.get(order.pair, []):
                        if cached_order['id'] == order.id:
                            cached_order['current_tp_level'] = new_tp_level
                            break

                await db.commit()

                updates.append({
                    "order_id": order.id,
                    "pair": order.pair,
                    "action": "EXTEND_TP",
                    "new_level": new_tp_level,
                    "new_target": new_tp_target
                })
            else:
                # Per-order commit for routine bookkeeping (peak/trough/high/low).
                # Keeping this commit short is critical: it releases the SQLite
                # write lock so close_position can acquire it.  An earlier
                # "optimization" batched these into a single commit at the end
                # of the loop, but autoflush on the next iteration's SELECT
                # held the write lock continuously, starving close_position's
                # retry loop for 2+ minutes.
                await db.commit()

        # --- Process exit retry queue (live mode only) ---
        if not self.is_paper_mode and _exit_retry_queue:
            retry_ids = list(_exit_retry_queue.keys())
            for order_id in retry_ids:
                attempt = _exit_retry_queue.get(order_id, 0) + 1
                _exit_retry_queue[order_id] = attempt

                if attempt > _EXIT_RETRY_MAX:
                    logger.critical(
                        f"[EXIT_RETRY_EXHAUSTED] Order {order_id}: Gave up after {_EXIT_RETRY_MAX} retries"
                    )
                    del _exit_retry_queue[order_id]
                    continue

                retry_result = await db.execute(
                    select(Order).where(Order.id == order_id)
                )
                retry_order = retry_result.scalar_one_or_none()
                if not retry_order or retry_order.status != "OPEN":
                    _exit_retry_queue.pop(order_id, None)
                    continue

                tracker = websocket_tracker.get_tracker(retry_order.pair)
                price = tracker.last_price if tracker else None
                if not price or price <= 0:
                    continue

                logger.info(
                    f"[EXIT_RETRY_QUEUE] {retry_order.pair}: Retry {attempt}/{_EXIT_RETRY_MAX}"
                )
                async with _close_lock:
                    closed = await self._close_position_locked(
                        db, retry_order, price, reason=retry_order.close_reason or "EXIT_RETRY"
                    )
                if closed:
                    _exit_retry_queue.pop(order_id, None)
                    logger.info(f"[EXIT_RETRY_QUEUE] {retry_order.pair}: Successfully closed on retry {attempt}")

        return updates
    
    async def scan_and_trade(self, db: AsyncSession) -> List[Dict]:
        """Scan top pairs and open positions based on signals"""
        if not self.is_running:
            return []
        
        import time
        now = time.time()
        if now - self._last_scan_time < 30:
            return []
        
        logger.info(f"[SCAN] Starting scan_and_trade cycle...")
        global _global_volume_ratio
        actions = []
        _scan_vol_sum = 0.0
        _scan_avg_vol_sum = 0.0
        
        # Get top pairs based on config limit.
        # New-listing and Alpha-subtype filters both run inside
        # get_top_futures_pairs BEFORE the top-N cut, so the returned list
        # is always "top N of eligible pairs" after both pre-filters apply.
        pairs_limit = config.trading_config.trading_pairs_limit
        _new_listing_days = getattr(config.trading_config, 'new_listing_filter_days', 0)
        _alpha_filter = getattr(config.trading_config, 'alpha_subtype_filter_enabled', True)
        top_pairs = await binance_service.get_top_futures_pairs(
            pairs_limit,
            new_listing_filter_days=_new_listing_days,
            alpha_subtype_filter_enabled=_alpha_filter,
        )
        # Jun 12: stamp the eligible-universe volume rank (1 = highest 24h vol)
        # BEFORE blacklist removal, so ranks stay comparable across config changes.
        # Persisted per-trade as entry_pair_rank (read gate for the 50->75 expansion).
        for _rank_i, _rank_p in enumerate(top_pairs):
            _rank_p['rank'] = _rank_i + 1
        _blacklist_str = getattr(config.trading_config, 'pair_blacklist', '')
        _blacklist = set(p.strip() for p in _blacklist_str.split(',') if p.strip())
        if _blacklist:
            top_pairs = [p for p in top_pairs if p['pair'] not in _blacklist]
            logger.info(f"[SCAN] Blacklist active: excluded {len(_blacklist)} pairs ({', '.join(sorted(_blacklist))})")
        logger.info(f"[SCAN] Fetched {len(top_pairs)} pairs from Binance (limit={pairs_limit})")
        
        if not top_pairs:
            logger.warning("[SCAN] No pairs returned from Binance - skipping scan cycle")
            self._last_scan_time = time.time()
            return []
        
        # Subscribe all top pairs to WebSocket in a single batch (one reconnection)
        await websocket_tracker.subscribe_pairs_batch([p['pair'] for p in top_pairs])
        
        # BTC global regime filter: fetch BTC data once before processing all pairs
        btc_global_enabled = getattr(config.trading_config.thresholds, 'btc_global_filter_enabled', False)
        btc_ema20 = None
        btc_ema20_prev3 = None
        btc_ema50 = None
        btc_regime = "NEUTRAL"
        btc_ema20_slope_pct = None
        btc_adx = None
        btc_adx_prev = None
        btc_rsi = None
        btc_rsi_prev = None
        btc_rsi_prev6 = None
        btc_atr_pct = None
        btc_rsi_1h = None
        btc_rsi_1h_prev = None
        # Always fetch BTC data for regime/slope display; the toggle only gates entry filters
        btc_ohlcv = await binance_service.get_ohlcv('BTC/USDT:USDT', '5m', 100)
        if btc_ohlcv:
            btc_indicators = calculate_indicators(btc_ohlcv)
            if btc_indicators:
                btc_ema20 = btc_indicators.get('ema20')
                btc_ema13 = btc_indicators.get('ema13')  # May 6 — used for BTC Trend Filter (EMA13/EMA50)
                btc_ema20_prev3 = btc_indicators.get('ema20_prev3')
                btc_ema50 = btc_indicators.get('ema50')
                btc_adx = btc_indicators.get('adx')
                btc_adx_prev = btc_indicators.get('adx_prev1')
                btc_rsi = btc_indicators.get('rsi')
                btc_rsi_prev = btc_indicators.get('rsi_prev1')
                btc_rsi_prev6 = btc_indicators.get('rsi_prev6')  # May 15: 30min sustained-momentum window
                # May 15 PM: BTC Volatility Regime (ATR / price × 100)
                _btc_atr_raw = btc_indicators.get('atr')
                _btc_price_now = btc_indicators.get('price')
                if _btc_atr_raw is not None and _btc_price_now is not None and _btc_price_now > 0:
                    btc_atr_pct = round((_btc_atr_raw / _btc_price_now) * 100, 4)
                else:
                    btc_atr_pct = None
                _flat_th_long = getattr(config.trading_config.thresholds, 'macro_trend_flat_threshold_long',
                                       config.trading_config.thresholds.macro_trend_flat_threshold)
                _flat_th_short = getattr(config.trading_config.thresholds, 'macro_trend_flat_threshold_short',
                                        config.trading_config.thresholds.macro_trend_flat_threshold)
                # Use the lower threshold so BTC regime stays directional for both sides;
                # direction-specific re-evaluation happens at signal level
                flat_th = min(_flat_th_long, _flat_th_short)
                btc_regime = determine_macro_regime(btc_ema20, btc_ema20_prev3, flat_th)
                if btc_ema20 and btc_ema20_prev3 and btc_ema20_prev3 != 0:
                    btc_ema20_slope_pct = round(((btc_ema20 - btc_ema20_prev3) / btc_ema20_prev3) * 100, 4)
        global _current_btc_regime, _btc_ema20_slope_pct, _current_btc_adx, _current_btc_rsi
        global _current_btc_ema20, _current_btc_ema13, _current_btc_ema50, _current_btc_trend_gap_pct, _current_btc_price
        _current_btc_regime = btc_regime
        _btc_ema20_slope_pct = btc_ema20_slope_pct if btc_ema20_slope_pct is not None else 0.0
        _current_btc_adx = btc_adx
        _current_btc_rsi = btc_rsi
        # BTC Trend Filter state (May 5; switched from EMA20→EMA13 on May 6 for faster reversal detection)
        _current_btc_ema20 = btc_ema20
        _current_btc_ema13 = btc_ema13
        _current_btc_ema50 = btc_ema50
        # May 14 — BTC price for BTC Market Extension dimension (price vs EMA13).
        _current_btc_price = btc_indicators.get('price') if btc_indicators else None
        # May 14 — BTC 1h EMA20 slope: fetch 1h OHLCV and compute slope.
        # Cached at the same cadence as the 5m scan (every cycle). 1h slope changes
        # slowly so this is mildly redundant but keeps the pipeline simple.
        try:
            global _current_btc_1h_slope
            btc_1h_ohlcv = await binance_service.get_ohlcv('BTC/USDT:USDT', '1h', 100)
            if btc_1h_ohlcv:
                btc_1h_ind = calculate_indicators(btc_1h_ohlcv)
                if btc_1h_ind:
                    _ema20_1h = btc_1h_ind.get('ema20')
                    _ema20_1h_prev3 = btc_1h_ind.get('ema20_prev3')
                    if _ema20_1h is not None and _ema20_1h_prev3 is not None and _ema20_1h_prev3 != 0:
                        _current_btc_1h_slope = round(((_ema20_1h - _ema20_1h_prev3) / _ema20_1h_prev3) * 100, 4)
                    # May 15 PM: BTC 1h RSI Direction. rsi_prev1 on 1h timeframe = 1h ago.
                    _rsi_1h_now = btc_1h_ind.get('rsi')
                    _rsi_1h_prev = btc_1h_ind.get('rsi_prev1')
                    if _rsi_1h_now is not None:
                        btc_rsi_1h = round(_rsi_1h_now, 1)
                    if _rsi_1h_prev is not None:
                        btc_rsi_1h_prev = round(_rsi_1h_prev, 1)
        except Exception as _e:
            logger.debug(f'[BTC_1H_SLOPE] fetch/compute failed: {_e}')
        if btc_ema13 is not None and btc_ema50 is not None and btc_ema50 != 0:
            # Trend gap = (EMA13 - EMA50) / EMA50 × 100. EMA13 spans ~65 min on 5m chart;
            # EMA50 spans ~250 min (~4 hours). Gap > 0 = BTC in 4hr uptrend, gap < 0 = downtrend.
            _current_btc_trend_gap_pct = round(((btc_ema13 - btc_ema50) / btc_ema50) * 100, 4)
        else:
            _current_btc_trend_gap_pct = None
        # Jun 15 — mirror the BTC prev/higher-TF COMPANION values to module globals too, so
        # _flip_entry_fields (which reads globals) can stamp them on flip Orders. The normal
        # entry path stamps these from the scan-locals below; flips fire mid-scan and read
        # globals, so without this mirror flips carried entry_btc_adx but NOT entry_btc_adx_prev
        # → every "by BTC ... Direction / Volatility / 1h" perf table (which compares cur vs
        # prev) silently dropped all flips. BTC is computed once per scan = scan-wide, so a
        # global mirror is correct (same pattern as _current_btc_adx above).
        global _current_btc_adx_prev, _current_btc_rsi_prev, _current_btc_rsi_prev6
        global _current_btc_atr_pct, _current_btc_rsi_1h, _current_btc_rsi_1h_prev
        _current_btc_adx_prev = btc_adx_prev
        _current_btc_rsi_prev = btc_rsi_prev
        _current_btc_rsi_prev6 = btc_rsi_prev6
        _current_btc_atr_pct = btc_atr_pct
        _current_btc_rsi_1h = btc_rsi_1h
        _current_btc_rsi_1h_prev = btc_rsi_1h_prev
        logger.info(f"[SCAN] BTC regime={btc_regime} slope={_btc_ema20_slope_pct}% (ema20={btc_ema20}, prev3={btc_ema20_prev3}, adx={btc_adx}) global_filter={'ON' if btc_global_enabled else 'OFF'}")
        
        # ── Phase 1: Collect indicators, signals, and pair regimes for ALL pairs ──
        _collected = []
        _breadth_flat_th = getattr(config.trading_config.thresholds, 'market_breadth_flat_threshold', 0.03)

        # Phase B observability (May 6) — snapshot open positions count for the
        # had_room flag used by get_signal's block_recorder. Approximation: the
        # value at scan start. Doesn't account for positions opened mid-scan,
        # but for observability of "did we generate the signal at all" this is
        # sufficient.
        try:
            _scan_start_open_count_q = await db.execute(
                select(func.count(Order.id)).where(
                    and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
                )
            )
            _scan_start_open_count = _scan_start_open_count_q.scalar() or 0
        except Exception:
            _scan_start_open_count = 0
        _scan_max_positions = config.trading_config.investment.max_open_positions or 5
        _scan_had_room_snapshot = _scan_start_open_count < _scan_max_positions

        # ── BTC macro veto pre-compute (May 8) ───────────────────────────────
        # Pair-level filter block counts were inflated because the chain runs
        # pair filters (in get_signal) BEFORE the BTC-level filters in this
        # function. So pair-level counters recorded blocks for signals that
        # would have been killed downstream by BTC anyway, making the Filter
        # Blocks table misleading ("the dominant blocker" was an artifact of
        # ordering, not reality).
        #
        # Fix: compute, scan-wide, which directions BTC-level filters would
        # veto using pair-agnostic BTC indicators (btc_ema13/50, btc_adx,
        # btc_adx_prev, btc_rsi, btc_ema20_slope_pct + thresholds). Then
        # suppress pair-level block recording for vetoed directions. Result:
        #   - Pair-level counters only count blocks where BTC was OK (= the
        #     pair-level filter was the decisive last gate).
        #   - BTC-level counters (recorded later in the loop, post-get_signal)
        #     continue to count blocks on signals that actually got generated.
        #   - Total block count drops, but each block reflects a real veto.
        _th_pre = config.trading_config.thresholds
        _btc_macro_blocks_long: Optional[str] = None
        _btc_macro_blocks_short: Optional[str] = None

        # 1) BTC Trend Filter (EMA13 vs EMA50)
        _btc_trend_enabled_pre = getattr(_th_pre, 'btc_trend_filter_enabled', False)
        if _btc_trend_enabled_pre and btc_ema13 is not None and btc_ema50 is not None:
            if btc_ema13 < btc_ema50:
                _btc_macro_blocks_long = _btc_macro_blocks_long or "BTC_TREND_FILTER"
            elif btc_ema13 > btc_ema50:
                _btc_macro_blocks_short = _btc_macro_blocks_short or "BTC_TREND_FILTER"

        # 2) BTC ADX range
        if btc_adx is not None:
            _l_lo = getattr(_th_pre, 'btc_adx_min_long', 0)
            _l_hi = getattr(_th_pre, 'btc_adx_max_long', 100)
            if (_l_lo > 0 and btc_adx < _l_lo) or (_l_hi < 100 and btc_adx > _l_hi):
                _btc_macro_blocks_long = _btc_macro_blocks_long or (
                    "BTC_ADX_GATE_LOW" if (_l_lo > 0 and btc_adx < _l_lo) else "BTC_ADX_GATE_HIGH"
                )
            _s_lo = getattr(_th_pre, 'btc_adx_min_short', 0)
            _s_hi = getattr(_th_pre, 'btc_adx_max_short', 100)
            if (_s_lo > 0 and btc_adx < _s_lo) or (_s_hi < 100 and btc_adx > _s_hi):
                _btc_macro_blocks_short = _btc_macro_blocks_short or (
                    "BTC_ADX_GATE_LOW" if (_s_lo > 0 and btc_adx < _s_lo) else "BTC_ADX_GATE_HIGH"
                )

        # 3) BTC ADX Direction
        if btc_adx is not None and btc_adx_prev is not None:
            _l_dir = getattr(_th_pre, 'btc_adx_dir_long', 'both')
            if (_l_dir == 'rising' and btc_adx <= btc_adx_prev) or (_l_dir == 'falling' and btc_adx >= btc_adx_prev):
                _btc_macro_blocks_long = _btc_macro_blocks_long or "BTC_ADX_DIR"
            _s_dir = getattr(_th_pre, 'btc_adx_dir_short', 'both')
            if (_s_dir == 'rising' and btc_adx <= btc_adx_prev) or (_s_dir == 'falling' and btc_adx >= btc_adx_prev):
                _btc_macro_blocks_short = _btc_macro_blocks_short or "BTC_ADX_DIR"

        # 4) BTC RSI x BTC ADX cross-filter
        if btc_rsi is not None and btc_adx is not None:
            for _dir_name, _slot_setter in (("LONG", "long"), ("SHORT", "short")):
                _cf_str = getattr(_th_pre, f'btc_rsi_adx_filter_{_slot_setter}', '') or ''
                if not _cf_str.strip():
                    continue
                for _cf_rule in _cf_str.split(','):
                    _cf_rule = _cf_rule.strip()
                    if not _cf_rule or ':' not in _cf_rule:
                        continue
                    try:
                        _r_part, _a_part = _cf_rule.split(':')
                        _r_lo, _r_hi = map(float, _r_part.split('-'))
                        _ab = _a_part.split('-')
                        if len(_ab) == 1:
                            _a_lo, _a_hi = float(_ab[0]), float('inf')
                        elif len(_ab) == 2:
                            _a_lo, _a_hi = float(_ab[0]), float(_ab[1])
                        else:
                            continue
                        if _r_lo <= btc_rsi < _r_hi and (btc_adx < _a_lo or btc_adx > _a_hi):
                            if _dir_name == "LONG":
                                _btc_macro_blocks_long = _btc_macro_blocks_long or "BTC_RSI_ADX_CROSS"
                            else:
                                _btc_macro_blocks_short = _btc_macro_blocks_short or "BTC_RSI_ADX_CROSS"
                            break
                    except (ValueError, TypeError):
                        continue

        # 5) BTC slope directional gate + slope max guard
        if btc_ema20_slope_pct is not None:
            _l_flat = getattr(_th_pre, 'macro_trend_flat_threshold_long',
                              getattr(_th_pre, 'macro_trend_flat_threshold', 0))
            if _l_flat > 0 and btc_ema20_slope_pct < _l_flat:
                _btc_macro_blocks_long = _btc_macro_blocks_long or "BTC_SLOPE_GATE"
            _s_flat = getattr(_th_pre, 'macro_trend_flat_threshold_short',
                              getattr(_th_pre, 'macro_trend_flat_threshold', 0))
            if _s_flat > 0 and btc_ema20_slope_pct > -_s_flat:
                _btc_macro_blocks_short = _btc_macro_blocks_short or "BTC_SLOPE_GATE"
            _l_smax = getattr(_th_pre, 'btc_ema20_slope_max_long', 0)
            if _l_smax and _l_smax > 0 and abs(btc_ema20_slope_pct) > _l_smax:
                _btc_macro_blocks_long = _btc_macro_blocks_long or "BTC_SLOPE_MAX_GATE"
            _s_smax = getattr(_th_pre, 'btc_ema20_slope_max_short', 0)
            if _s_smax and _s_smax > 0 and abs(btc_ema20_slope_pct) > _s_smax:
                _btc_macro_blocks_short = _btc_macro_blocks_short or "BTC_SLOPE_MAX_GATE"

        if _btc_macro_blocks_long or _btc_macro_blocks_short:
            logger.info(
                f"[FILTER_BLOCK_ATTRIB] BTC macro veto active this scan — "
                f"LONG={_btc_macro_blocks_long or 'OK'} SHORT={_btc_macro_blocks_short or 'OK'}; "
                f"pair-level blocks for vetoed directions will be suppressed from counters"
            )

        # Container holding the pair being evaluated. Set inside the per-pair
        # loop below so the recorder closure can stamp _last_pair_block_reason.
        _current_pair_holder = {'pair': None}

        def _signal_block_recorder(filter_name: str, direction: str):
            # Suppress pair-level block recording for directions that BTC-level
            # filters would have vetoed anyway. This makes Filter Blocks counts
            # reflect the *decisive* last gate, not artifacts of evaluation order.
            if direction == "LONG" and _btc_macro_blocks_long is not None:
                return
            if direction == "SHORT" and _btc_macro_blocks_short is not None:
                return
            self._record_filter_block(filter_name, direction, had_room=_scan_had_room_snapshot)
            _p = _current_pair_holder.get('pair')
            if _p:
                self._last_pair_block_reason[_p] = filter_name
            # Jun 15: phantom — fade the OVERBOUGHT-long RSI block to a SHORT (dedup-pool
            # NP=28% for RSI>65 longs, the data's top fade candidate). Seed ONLY the
            # overbought case (rsi > long_rsi_max), NOT the oversold-long block (those
            # bounce up). Observation-only; source label "Pair RSI >65".
            if filter_name == "PAIR_RSI_RANGE" and direction == "LONG":
                try:
                    _rsi = _current_pair_holder.get('rsi')
                    _px = _current_pair_holder.get('price')
                    _rmax = getattr(config.trading_config.thresholds, 'momentum_long_rsi_max', 65)
                    if _rsi is not None and _px and _rsi > _rmax:
                        _seed_phantom_flip(_p, _px, "LONG", "Pair RSI >65",
                                           entry_fields=self._flip_entry_fields(_current_pair_holder, flip_dir="SHORT"))
                        # Jun 16: also mark this pair so Phase 3 can open the LIVE flip
                        # (PAIR_RSI_OB source). The block fires inside get_signal (signal is
                        # already NO_TRADE by Phase 3), so the marker rides the _collected row.
                        _current_pair_holder['rsi_ob_flip'] = True
                except Exception:
                    pass
            # Jun 16: LONG-fade phantom trackers — fade a BLOCKED SHORT to a LONG (block→fade,
            # observation-only, mirror of the short-side fades). These two short-blocks fire
            # inside get_signal; seed here off the decisive-gate recorder. PAIR_ADX_MAX = the
            # down-move was too extended to short → bounce-long fade; PAIR_RSI_ADX_CROSS = the
            # pair RSI×ADX cross gate (pair mirror of the BTC cross).
            if direction == "SHORT" and filter_name in ("PAIR_ADX_MAX", "PAIR_RSI_ADX_CROSS"):
                try:
                    _px2 = _current_pair_holder.get('price')
                    if _px2:
                        _seed_phantom_flip(_p, _px2, "SHORT", filter_name,
                                           entry_fields=self._flip_entry_fields(_current_pair_holder, flip_dir="LONG"))
                except Exception:
                    pass

        for batch_start in range(0, len(top_pairs), OHLCV_BATCH_SIZE):
            batch = top_pairs[batch_start:batch_start + OHLCV_BATCH_SIZE]
            batch_num = batch_start // OHLCV_BATCH_SIZE + 1
            total_batches = (len(top_pairs) + OHLCV_BATCH_SIZE - 1) // OHLCV_BATCH_SIZE
            logger.info(f"[SCAN] Processing batch {batch_num}/{total_batches} ({len(batch)} pairs)")

            for pair_info in batch:
                pair = pair_info['pair']
                symbol = pair_info['symbol']
                volume_24h = pair_info['volume_24h']

                # Stash current pair so the block recorder closure can stamp
                # _last_pair_block_reason for the UI's Block Reason column.
                _current_pair_holder['pair'] = pair
                # Pre-stamp a default "no setup" reason. Most top-50 pairs at any
                # moment have no EMA stack alignment → get_signal returns NOTHING
                # without calling _record(). Default placeholder is overwritten
                # the moment any filter actually fires.
                self._last_pair_block_reason[pair] = "No EMA Stack"

                ohlcv = await binance_service.get_ohlcv(symbol, '5m', 100)
                if not ohlcv:
                    continue

                _pair_vol_bars = getattr(config.trading_config.thresholds, 'pair_volume_lookback_bars', 20)
                _global_vol_bars = getattr(config.trading_config.thresholds, 'global_volume_lookback_bars', 48)
                indicators = calculate_indicators(ohlcv, pair_volume_bars=_pair_vol_bars, global_volume_bars=_global_vol_bars)
                if not indicators:
                    continue

                rsi_val = indicators.get('rsi')
                adx_val = indicators.get('adx')
                if rsi_val is not None and (rsi_val >= 99.9 or rsi_val <= 0.1):
                    logger.debug(f"[SKIP] {pair}: Degenerate RSI={rsi_val:.1f} (no price variation)")
                    continue
                if adx_val is None:
                    logger.debug(f"[SKIP] {pair}: ADX is null (insufficient price data)")
                    continue

                _pair_vol = indicators.get('volume') or 0
                _pair_avg_vol = indicators.get('avg_volume') or 0
                _pair_volume_ratio = round(_pair_vol / _pair_avg_vol, 4) if _pair_avg_vol > 0 else 1.0
                _pair_avg_vol_global = indicators.get('avg_volume_global') or 0
                _scan_vol_sum += _pair_vol
                _scan_avg_vol_sum += _pair_avg_vol_global if _pair_avg_vol_global > 0 else _pair_avg_vol

                # Jun 15: stash rsi/price so the block recorder can seed the overbought-RSI
                # phantom (the PAIR_RSI_RANGE LONG block fires inside get_signal).
                _current_pair_holder['rsi'] = indicators.get('rsi')
                _current_pair_holder['price'] = indicators.get('price')
                _current_pair_holder['rsi_ob_flip'] = False  # Jun 16: reset live-flip marker per pair

                signal, confidence = get_signal(
                    ema5=indicators.get('ema5'),
                    ema8=indicators.get('ema8'),
                    ema13=indicators.get('ema13'),
                    ema20=indicators.get('ema20'),
                    rsi=indicators.get('rsi'),
                    adx=indicators.get('adx'),
                    volume=indicators.get('volume'),
                    avg_volume=indicators.get('avg_volume'),
                    price=indicators.get('price'),
                    ema20_prev3=indicators.get('ema20_prev3'),
                    ema50=indicators.get('ema50'),
                    ema50_prev12=indicators.get('ema50_prev12'),
                    rsi_prev3=indicators.get('rsi_prev3'),
                    ema5_prev1=indicators.get('ema5_prev1'),
                    ema8_prev1=indicators.get('ema8_prev1'),
                    ema5_prev2=indicators.get('ema5_prev2'),
                    ema8_prev2=indicators.get('ema8_prev2'),
                    ema13_prev1=indicators.get('ema13_prev1'),
                    ema13_prev2=indicators.get('ema13_prev2'),
                    adx_prev1=indicators.get('adx_prev1'),
                    high_20=indicators.get('high_20'),
                    low_20=indicators.get('low_20'),
                    block_recorder=_signal_block_recorder,
                )

                if signal in ["LONG", "SHORT"]:
                    logger.info(f"[SIGNAL-FOUND] {pair}: {signal} {confidence} - RSI={indicators.get('rsi'):.1f}, ADX={indicators.get('adx')}")

                breadth_regime = determine_macro_regime(
                    indicators.get('ema20'), indicators.get('ema20_prev3'), _breadth_flat_th
                )

                _collected.append({
                    'pair': pair, 'symbol': symbol, 'volume_24h': volume_24h,
                    'indicators': indicators, 'signal': signal, 'confidence': confidence,
                    'pair_volume_ratio': _pair_volume_ratio, 'breadth_regime': breadth_regime,
                    'rank': pair_info.get('rank'),
                    'rsi_ob_flip': _current_pair_holder.get('rsi_ob_flip', False),  # Jun 16: overbought-RSI live flip
                })

            if batch_start + OHLCV_BATCH_SIZE < len(top_pairs):
                await asyncio.sleep(OHLCV_BATCH_DELAY)

        # ── Phase 2: Compute global volume ratio and market breadth ──
        if _scan_avg_vol_sum > 0:
            _global_volume_ratio = round(_scan_vol_sum / _scan_avg_vol_sum, 4)
            logger.info(f"[GLOBAL_VOL] ratio={_global_volume_ratio:.4f} (sum_vol={_scan_vol_sum:.0f}, sum_avg={_scan_avg_vol_sum:.0f})")

        global _market_bull_pct, _market_bear_pct, _breadth_n_bull, _breadth_n_bear, _breadth_n_neutral, _breadth_n_total
        _breadth_n_bull = sum(1 for r in _collected if r['breadth_regime'] == "BULLISH")
        _breadth_n_bear = sum(1 for r in _collected if r['breadth_regime'] == "BEARISH")
        _breadth_n_total = len(_collected)
        _breadth_n_neutral = _breadth_n_total - _breadth_n_bull - _breadth_n_bear
        if _breadth_n_total > 0:
            _market_bull_pct = round(_breadth_n_bull / _breadth_n_total * 100, 1)
            _market_bear_pct = round(_breadth_n_bear / _breadth_n_total * 100, 1)
        else:
            _market_bull_pct = 0.0
            _market_bear_pct = 0.0
        logger.info(f"[BREADTH] Bull={_market_bull_pct:.1f}% ({_breadth_n_bull}/{_breadth_n_total}) Bear={_market_bear_pct:.1f}% ({_breadth_n_bear}/{_breadth_n_total}) threshold={_breadth_flat_th}%")

        # ── Phase 3: Apply gates (BTC, volume, breadth) and enter trades ──
        _breadth_enabled = getattr(config.trading_config.thresholds, 'market_breadth_filter_enabled', True)
        _breadth_bull_th = getattr(config.trading_config.thresholds, 'market_breadth_bull_threshold_long', 50.0)
        _breadth_bear_th = getattr(config.trading_config.thresholds, 'market_breadth_bear_threshold_short', 65.0)

        # Track had_room state for filter blocks: count open positions at scan
        # start; increment when open_position succeeds in this loop. Filter
        # blocks recorded with had_room=False (FULL) didn't actually prevent a
        # trade — bot was already at max_open_positions when the block fired.
        try:
            _open_count_q = await db.execute(
                select(func.count(Order.id)).where(
                    and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
                )
            )
            _open_positions_in_scan = _open_count_q.scalar() or 0
        except Exception:
            _open_positions_in_scan = 0
        _max_positions = config.trading_config.investment.max_open_positions or 5

        for _cr in _collected:
            _had_room = _open_positions_in_scan < _max_positions
            pair = _cr['pair']
            indicators = _cr['indicators']
            signal = _cr['signal']
            confidence = _cr['confidence']
            volume_24h = _cr['volume_24h']
            _pair_volume_ratio = _cr['pair_volume_ratio']
            _pair_rank = _cr.get('rank')

            # Jun 16: PAIR_RSI_OB live flip — fade an overbought-long (rsi>65) block to SHORT.
            # The block fired inside get_signal (Phase 1, signal already NO_TRADE here), so we
            # act on the marker carried on the row. Source key "PAIR_RSI_OB" (the phantom cell
            # is "Pair RSI >65"). Below-evidence operator override @1x — _maybe_open_flip is a
            # no-op unless the source is in flip_entry_sources; all risk caps live in
            # open_position. Fail-silent so a flip bug can't break the scan.
            if _cr.get('rsi_ob_flip'):
                try:
                    await self._maybe_open_flip(
                        db, pair, "LONG", "PAIR_RSI_OB", indicators,
                        entry_fields=self._flip_entry_fields(indicators, flip_dir="SHORT",
                                                             scan=self._flip_scan_ctx(locals())))
                except Exception as _ob_flip_err:
                    logger.error(f"[FLIP_ENTRY] {pair}: PAIR_RSI_OB flip failed: {_ob_flip_err}")

            # Jun 3: NO-TRADE pairs — stay in the top-pair/volume universe (subscribed,
            # scanned, displayed) but entries are blocked. Distinct from pair_blacklist
            # (which removes the pair from the universe entirely). Used for BTCUSDT: visible
            # for reference, never opens a position.
            if signal in ["LONG", "SHORT"]:
                _nt_str = getattr(config.trading_config, 'no_trade_pairs', '') or ''
                _nt = set(p.strip() for p in _nt_str.split(',') if p.strip())
                if pair in _nt:
                    logger.info(f"[PAIR_NO_TRADE] {pair}: {signal} blocked — pair is track-only (no_trade_pairs)")
                    self._record_filter_block("PAIR_NO_TRADE", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "PAIR_NO_TRADE"
                    signal = "NO_TRADE"

            if signal in ["LONG", "SHORT"] and not self.is_paper_mode:
                _symbol_check = pair.replace('USDT', '/USDT:USDT')
                if _symbol_check in _leverage_blocked_pairs:
                    logger.debug(f"[LEVERAGE_BLOCKED] {pair}: Skipping — leverage mismatch previously detected")
                    signal = "NO_TRADE"

            if signal in ["LONG", "SHORT"] and btc_global_enabled:
                _th_cfg = config.trading_config.thresholds
                if signal == "LONG":
                    flat_th = getattr(_th_cfg, 'macro_trend_flat_threshold_long', _th_cfg.macro_trend_flat_threshold)
                else:
                    flat_th = getattr(_th_cfg, 'macro_trend_flat_threshold_short', _th_cfg.macro_trend_flat_threshold)
                pair_regime = determine_macro_regime(
                    indicators.get('ema20'), indicators.get('ema20_prev3'), flat_th
                )
                neutral_mode = getattr(config.trading_config.thresholds, 'macro_trend_neutral_mode', 'both')
                btc_blocks = False
                if btc_regime == "NEUTRAL" and neutral_mode != "both":
                    btc_blocks = True
                elif btc_regime == "BULLISH" and signal != "LONG":
                    btc_blocks = True
                elif btc_regime == "BEARISH" and signal != "SHORT":
                    btc_blocks = True

                pair_blocks = (pair_regime != btc_regime)

                _th = config.trading_config.thresholds
                btc_rsi_blocks = False
                if btc_rsi is not None:
                    if signal == "LONG":
                        _rsi_lo = getattr(_th, 'btc_rsi_min_long', 0)
                        _rsi_hi = getattr(_th, 'btc_rsi_max_long', 100)
                    else:
                        _rsi_lo = getattr(_th, 'btc_rsi_min_short', 0)
                        _rsi_hi = getattr(_th, 'btc_rsi_max_short', 100)
                    if (_rsi_lo > 0 and btc_rsi < _rsi_lo) or (_rsi_hi < 100 and btc_rsi > _rsi_hi):
                        btc_rsi_blocks = True

                # BTC RSI x BTC ADX cross-filter moved outside btc_global gate (May 5 fix —
                # was dead code when btc_global_filter_enabled=false, the current default).
                # BTC ADX range check moved outside btc_global gate (runs independently).
                # BTC ADX Direction check also moved outside — see independent block below
                # (Phase 1c Option B refactor, Apr 17 — 3-sample confirmed structural signal
                # for shorts: Rising BTC ADX > Falling BTC ADX across Apr 6, Apr 13, Apr 17).
                # BTC RSI min/max stays GATED by btc_global_enabled per user direction May 5.

                pair_adx_dir_blocks = False
                _pair_adx = indicators.get('adx')
                _pair_adx_prev = indicators.get('adx_prev1')
                if _pair_adx is not None and _pair_adx_prev is not None:
                    _pair_adx_dir_cfg = getattr(_th, f'adx_dir_{signal.lower()}', 'both')
                    if _pair_adx_dir_cfg == 'rising' and _pair_adx <= _pair_adx_prev:
                        pair_adx_dir_blocks = True
                    elif _pair_adx_dir_cfg == 'falling' and _pair_adx >= _pair_adx_prev:
                        pair_adx_dir_blocks = True

                if btc_blocks or pair_blocks or btc_rsi_blocks or pair_adx_dir_blocks:
                    if pair_adx_dir_blocks:
                        _pd_label = "Rising" if _pair_adx > _pair_adx_prev else "Falling"
                        _pd_want = getattr(_th, f'adx_dir_{signal.lower()}', 'both')
                        reason = f"Pair ADX {_pd_label} ({_pair_adx:.1f} vs prev {_pair_adx_prev:.1f}), {signal} requires {_pd_want}"
                    elif btc_rsi_blocks:
                        reason = f"BTC RSI {btc_rsi:.1f} out of {signal} range [{_rsi_lo}-{_rsi_hi}]"
                    elif btc_blocks:
                        reason = f"BTC={btc_regime}"
                    else:
                        reason = f"pair={pair_regime} vs BTC={btc_regime}"
                    logger.info(f"[BTC-GATE] {pair}: {signal} blocked — {reason}")
                    self._record_filter_block("BTC_REGIME", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_REGIME"
                    signal = "NO_TRADE"

            # Pair ADX Direction check — runs independently of BTC global filter
            if signal in ["LONG", "SHORT"]:
                _th = config.trading_config.thresholds
                _pair_adx = indicators.get('adx')
                _pair_adx_prev = indicators.get('adx_prev1')
                if _pair_adx is not None and _pair_adx_prev is not None:
                    _pair_adx_dir_cfg = getattr(_th, f'adx_dir_{signal.lower()}', 'both')
                    if _pair_adx_dir_cfg == 'rising' and _pair_adx <= _pair_adx_prev:
                        _pd_label = "Rising" if _pair_adx > _pair_adx_prev else "Falling"
                        logger.info(f"[PAIR_ADX_DIR] {pair}: {signal} blocked — Pair ADX {_pd_label} ({_pair_adx:.4f} vs prev {_pair_adx_prev:.4f}), requires {_pair_adx_dir_cfg}")
                        self._record_filter_block("PAIR_ADX_DIR", signal, had_room=_had_room)
                        self._last_pair_block_reason[pair] = "PAIR_ADX_DIR"
                        signal = "NO_TRADE"
                    elif _pair_adx_dir_cfg == 'falling' and _pair_adx >= _pair_adx_prev:
                        _pd_label = "Rising" if _pair_adx > _pair_adx_prev else "Falling"
                        logger.info(f"[PAIR_ADX_DIR] {pair}: {signal} blocked — Pair ADX {_pd_label} ({_pair_adx:.4f} vs prev {_pair_adx_prev:.4f}), requires {_pair_adx_dir_cfg}")
                        self._record_filter_block("PAIR_ADX_DIR", signal, had_room=_had_room)
                        self._last_pair_block_reason[pair] = "PAIR_ADX_DIR"
                        signal = "NO_TRADE"

            # BTC ADX range check — runs independently of BTC global filter.
            # When btc_adx_min > 0 or btc_adx_max < 100, the check is active
            # regardless of whether the Macro Trend toggle is on.
            if signal in ["LONG", "SHORT"] and btc_adx is not None:
                _th = config.trading_config.thresholds
                if signal == "LONG":
                    _btc_adx_lo = getattr(_th, 'btc_adx_min_long', 0)
                    _btc_adx_hi = getattr(_th, 'btc_adx_max_long', 100)
                else:
                    _btc_adx_lo = getattr(_th, 'btc_adx_min_short', 0)
                    _btc_adx_hi = getattr(_th, 'btc_adx_max_short', 100)
                _btc_adx_too_low = _btc_adx_lo > 0 and btc_adx < _btc_adx_lo
                _btc_adx_too_high = _btc_adx_hi < 100 and btc_adx > _btc_adx_hi
                if _btc_adx_too_low or _btc_adx_too_high:
                    _gate_subtype = "BTC_ADX_GATE_LOW" if _btc_adx_too_low else "BTC_ADX_GATE_HIGH"
                    _bound_label = f"<{_btc_adx_lo}" if _btc_adx_too_low else f">{_btc_adx_hi}"
                    logger.info(f"[{_gate_subtype}] {pair}: {signal} blocked — BTC ADX {btc_adx:.1f} {_bound_label} (range [{_btc_adx_lo}-{_btc_adx_hi}])")
                    self._record_filter_block(_gate_subtype, signal, had_room=_had_room)
                    signal = "NO_TRADE"

            # BTC RSI BAND × BTC ATR conditional block — May 27, 2026 A3 ship.
            # Replaces the broad "65-70:99-100" BTC RSI 65-70 LONG block (over-restrictive)
            # with a surgical "BTC RSI in band AND BTC ATR condition" filter.
            # Cross-batch (965-trade pool): broad block had 1.91:1 save:cut ratio while
            # the A3 conditional (BTC ATR <0.10) has 3.99:1 — preserves NEAR +$197 / GMT +$86
            # / TIA +$57 winners that hit BTC RSI 65-70 in healthy-volatility regimes.
            # Format per rule: "RSI_LO-RSI_HI:OP" where OP is "<X", ">X", or "X-Y".
            # OP semantics:
            #   "<X" → block when BTC ATR < X
            #   ">X" → block when BTC ATR > X
            #   "X-Y" → block when X <= BTC ATR < Y
            # Multi-rule via comma. Empty string = inactive.
            if signal in ["LONG", "SHORT"] and btc_rsi is not None and btc_atr_pct is not None:
                _th_atr = config.trading_config.thresholds
                _atr_key = 'btc_rsi_band_atr_block_long' if signal == 'LONG' else 'btc_rsi_band_atr_block_short'
                _atr_rules_str = getattr(_th_atr, _atr_key, '') or ''
                if _atr_rules_str.strip():
                    for _atr_rule in _atr_rules_str.split(','):
                        _atr_rule = _atr_rule.strip()
                        if not _atr_rule or ':' not in _atr_rule:
                            continue
                        try:
                            _atr_rsi_part, _atr_op_part = _atr_rule.split(':', 1)
                            _atr_rsi_lo, _atr_rsi_hi = map(float, _atr_rsi_part.split('-'))
                            if not (_atr_rsi_lo <= btc_rsi < _atr_rsi_hi):
                                continue
                            _atr_op = _atr_op_part.strip()
                            _blocked = False
                            _label = ""
                            if _atr_op.startswith('<'):
                                _thr = float(_atr_op[1:])
                                if btc_atr_pct < _thr:
                                    _blocked = True
                                    _label = f"BTC ATR {btc_atr_pct:.3f} < {_thr}"
                            elif _atr_op.startswith('>'):
                                _thr = float(_atr_op[1:])
                                if btc_atr_pct > _thr:
                                    _blocked = True
                                    _label = f"BTC ATR {btc_atr_pct:.3f} > {_thr}"
                            elif '-' in _atr_op:
                                _thr_lo, _thr_hi = map(float, _atr_op.split('-'))
                                if _thr_lo <= btc_atr_pct < _thr_hi:
                                    _blocked = True
                                    _label = f"BTC ATR {btc_atr_pct:.3f} in [{_thr_lo}, {_thr_hi})"
                            if _blocked:
                                logger.info(
                                    f"[BTC_RSI_ATR_COND] {pair}: {signal} blocked — "
                                    f"BTC RSI {btc_rsi:.1f} in [{_atr_rsi_lo}, {_atr_rsi_hi}) AND {_label}"
                                )
                                self._record_filter_block("BTC_RSI_ATR_COND", signal, had_room=_had_room)
                                self._last_pair_block_reason[pair] = "BTC_RSI_ATR_COND"
                                signal = "NO_TRADE"
                                break
                        except (ValueError, TypeError):
                            continue

            # SHORT-only BTC ADX BLOCK RANGE — May 27, 2026 (see CLAUDE.md).
            # Blocks SHORT entries when BTC ADX falls inside a "kill zone" range, even though
            # min/max gate above would allow it. Cross-batch evidence (965-trade pool, full
            # STRONG_BUY SHORT cohort): BTC ADX 24-30 = 100 trades / 49% WR / -$1,607 / -$16/tr.
            # Both 0 = disabled. Default config 24/30 (block ≥24 AND <30 within the SHORT
            # min/max window). VERY_STRONG SHORT in the same zone: 38 trades / 60.5% WR /
            # +$2.07/tr — borderline; this filter cuts that cohort too (acceptable trade-off).
            if signal == "SHORT" and btc_adx is not None:
                _th2 = config.trading_config.thresholds
                _block_lo = getattr(_th2, 'btc_adx_block_min_short', 0.0)
                _block_hi = getattr(_th2, 'btc_adx_block_max_short', 0.0)
                if _block_lo > 0 and _block_hi > _block_lo and _block_lo <= btc_adx < _block_hi:
                    logger.info(
                        f"[BTC_ADX_BLOCK_SHORT] {pair}: SHORT blocked — BTC ADX {btc_adx:.1f} "
                        f"in kill range [{_block_lo}, {_block_hi})"
                    )
                    self._record_filter_block("BTC_ADX_BLOCK_SHORT", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_ADX_BLOCK_SHORT"
                    # Jun 16: LONG-fade phantom — short killed by a strong-bull BTC regime →
                    # fading LONG is macro-ALIGNED (the one robust short-side lesson). Obs-only.
                    try:
                        _seed_phantom_flip(pair, indicators.get('price'), "SHORT", "BTC_ADX_BLOCK_SHORT",
                                           entry_fields=self._flip_entry_fields(indicators, flip_dir="LONG", scan=self._flip_scan_ctx(locals())))
                    except Exception:
                        pass
                    signal = "NO_TRADE"

            # BTC ADX Direction check — runs independently of BTC global filter
            # (Phase 1c Option B refactor, Apr 17).  Pre-refactor this lived inside
            # the `if btc_global_enabled:` block, so turning off Macro Trend
            # silently disabled the directional filter.  Moved here so
            # btc_adx_dir_long / btc_adx_dir_short works standalone.
            # Structural basis: 3-sample confirmation across Apr 6, Apr 13, Apr 17
            # that SHORTS in Rising BTC ADX materially outperform SHORTS in
            # Falling BTC ADX (exhausting downtrend = bounce risk).  "both" = no
            # filter active.  "rising"/"falling" gates the entry.
            if signal in ["LONG", "SHORT"] and btc_adx is not None and btc_adx_prev is not None:
                _th = config.trading_config.thresholds
                _adx_dir_cfg = getattr(_th, f'btc_adx_dir_{signal.lower()}', 'both')
                _dir_blocks = False
                if _adx_dir_cfg == 'rising' and btc_adx <= btc_adx_prev:
                    _dir_blocks = True
                elif _adx_dir_cfg == 'falling' and btc_adx >= btc_adx_prev:
                    _dir_blocks = True
                if _dir_blocks:
                    _dir_label = "Rising" if btc_adx > btc_adx_prev else ("Falling" if btc_adx < btc_adx_prev else "Flat")
                    logger.info(
                        f"[BTC_ADX_DIR] {pair}: {signal} blocked — BTC ADX {_dir_label} "
                        f"({btc_adx:.2f} vs prev {btc_adx_prev:.2f}), requires {_adx_dir_cfg}"
                    )
                    self._record_filter_block("BTC_ADX_DIR", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_ADX_DIR"
                    signal = "NO_TRADE"

            # BTC RSI x BTC ADX Cross-Filter — runs independently of BTC global filter (May 5 fix).
            # Pre-fix this lived inside the `if btc_global_enabled:` block, so the cross-filter
            # rules in btc_rsi_adx_filter_long/short were dead code when Macro Trend was off
            # (current default).  Discovered May 5 when a BTC RSI 76.2 x BTC ADX 32.5 LONG fired
            # despite the "70-100:35" rule.  Same Apr 17 Option B refactor pattern as BTC ADX
            # direction/range moved out before.
            # Cross-filter rule formats supported (backward compatible):
            #   "RSI_LO-RSI_HI:MIN_ADX"          → require ADX >= MIN_ADX (existing)
            #   "RSI_LO-RSI_HI:MIN_ADX-MAX_ADX"  → require MIN_ADX <= ADX <= MAX_ADX (new May 5)
            # The new range form lets us express "block when BTC ADX > X" by setting
            # MIN_ADX low (e.g. 0).  Example: "65-70:0-34" blocks BTC RSI 65-70 entries
            # when BTC ADX > 34 (i.e., the over-extended high-ADX edge of that band).
            if signal in ["LONG", "SHORT"] and btc_rsi is not None and btc_adx is not None:
                _th = config.trading_config.thresholds
                _cf_key = 'btc_rsi_adx_filter_long' if signal == 'LONG' else 'btc_rsi_adx_filter_short'
                _cf_str = getattr(_th, _cf_key, '')
                if _cf_str and _cf_str.strip():
                    for _cf_rule in _cf_str.split(','):
                        _cf_rule = _cf_rule.strip()
                        if not _cf_rule or ':' not in _cf_rule:
                            continue
                        try:
                            _cf_rsi_part, _cf_adx_part = _cf_rule.split(':')
                            _cf_rsi_min, _cf_rsi_max = map(float, _cf_rsi_part.split('-'))
                            _adx_bounds = _cf_adx_part.split('-')
                            if len(_adx_bounds) == 1:
                                _cf_min_adx = float(_adx_bounds[0])
                                _cf_max_adx = float('inf')
                                _cf_label = f"requires ADX>={_cf_min_adx}"
                            elif len(_adx_bounds) == 2:
                                _cf_min_adx = float(_adx_bounds[0])
                                _cf_max_adx = float(_adx_bounds[1])
                                _cf_label = f"requires {_cf_min_adx}<=ADX<={_cf_max_adx}"
                            else:
                                continue
                            if _cf_rsi_min <= btc_rsi < _cf_rsi_max:
                                if btc_adx < _cf_min_adx or btc_adx > _cf_max_adx:
                                    logger.info(
                                        f"[BTC_RSI_ADX_CROSS] {pair}: {signal} blocked — "
                                        f"BTC RSI {btc_rsi:.1f} in [{_cf_rsi_min}-{_cf_rsi_max}) "
                                        f"{_cf_label}, got {btc_adx:.1f}"
                                    )
                                    self._record_filter_block("BTC_RSI_ADX_CROSS", signal, had_room=_had_room)
                                    self._last_pair_block_reason[pair] = "BTC_RSI_ADX_CROSS"
                                    # Jun 13: phantom flip — EXTREMES ONLY. Only BTC RSI extremes carry
                                    # mean-reversion logic: overbought (≥70) LONG-block → fade SHORT;
                                    # oversold (≤35) SHORT-block → fade LONG (the cleaner one). Mid-RSI
                                    # cells are directionless — skipped. Macro/correlated: read separately.
                                    if (signal == "LONG" and btc_rsi >= 70) or (signal == "SHORT" and btc_rsi <= 35):
                                        _seed_phantom_flip(pair, indicators.get('price'), signal, "BTC_RSI_ADX_CROSS",
                                                           entry_fields=self._flip_entry_fields(indicators, flip_dir=('SHORT' if signal == 'LONG' else 'LONG'), scan=self._flip_scan_ctx(locals())))
                                    signal = "NO_TRADE"
                                break
                        except (ValueError, TypeError):
                            continue

            # ADX Delta x BTC ADX Cross-Filter (May 11, 2026 — see CLAUDE.md May 11
            # deep review).  Pooled data across May 4 → tonight (288 LONGs, 6 batches)
            # shows that when pair ADX is spiking fast (delta 1.0-2.0) AND BTC ADX is
            # in the mid-strength zone (18-25), entries are catastrophic losers:
            # N=49, 31% WR, -$267.  Same pair-delta in stronger BTC regimes (25-30 →
            # +$98 / 30-35 → +$98) is profitable, so the signal is conditional on
            # macro confirmation, not pair-level alone.
            # Rule format: "deltaLo-deltaHi:btcAdxLo-btcAdxHi" — block when ADX Delta
            # in [deltaLo, deltaHi) AND BTC ADX in [btcAdxLo, btcAdxHi).
            # Multiple rules separated by commas.  Empty = filter inactive.
            # May 18: gated by adx_delta_btc_adx_filter_enabled master toggle.
            _adx_df_enabled = getattr(config.trading_config.thresholds,
                                      'adx_delta_btc_adx_filter_enabled', True)
            if _adx_df_enabled and signal in ["LONG", "SHORT"] and btc_adx is not None:
                _pair_adx_now = indicators.get('adx')
                _pair_adx_pre = indicators.get('adx_prev1')
                if _pair_adx_now is not None and _pair_adx_pre is not None:
                    _adx_delta_val = _pair_adx_now - _pair_adx_pre
                    _th = config.trading_config.thresholds
                    _df_key = 'adx_delta_btc_adx_filter_long' if signal == 'LONG' else 'adx_delta_btc_adx_filter_short'
                    _df_str = getattr(_th, _df_key, '')
                    if _df_str and _df_str.strip():
                        for _df_rule in _df_str.split(','):
                            _df_rule = _df_rule.strip()
                            if not _df_rule or ':' not in _df_rule:
                                continue
                            try:
                                _df_d_part, _df_a_part = _df_rule.split(':')
                                _df_d_lo, _df_d_hi = map(float, _df_d_part.split('-'))
                                _df_a_lo, _df_a_hi = map(float, _df_a_part.split('-'))
                                if (_df_d_lo <= _adx_delta_val < _df_d_hi and
                                        _df_a_lo <= btc_adx < _df_a_hi):
                                    logger.info(
                                        f"[ADX_DELTA_BTC_ADX_CROSS] {pair}: {signal} blocked — "
                                        f"ADXΔ {_adx_delta_val:.2f} in [{_df_d_lo}-{_df_d_hi}) "
                                        f"AND BTC ADX {btc_adx:.1f} in [{_df_a_lo}-{_df_a_hi})"
                                    )
                                    self._record_filter_block("ADX_DELTA_BTC_ADX_CROSS", signal, had_room=_had_room)
                                    self._last_pair_block_reason[pair] = "ADX_DELTA_BTC_ADX_CROSS"
                                    signal = "NO_TRADE"
                                    break
                            except (ValueError, TypeError):
                                continue

            # RngPos × ADX Δ 2D Cross-Filter (May 18 PM).
            # Catches "bottom/top-fishing into momentum acceleration" — the
            # pattern that killed 4 SHORTs in the May 18 cluster (RngPos 8-9,
            # ADX Δ 1.27-1.77, BTC RSI low). Cross-batch evidence: N=10, 30%
            # WR, -$359 in the SHORT 5-10 × 1.0-2.0 cell. Existing filters
            # don't catch this — range_position_min_short blocks only <2%
            # and min_adx_delta_short blocks only the LOW delta side.
            # Rule format: "<rngLo>-<rngHi>:<adxdLo>-<adxdHi>" — block when
            # range_position in [rngLo, rngHi] AND ADX Δ in [adxdLo, adxdHi).
            # Multiple rules separated by commas. Empty = filter inactive.
            _rpad_enabled = getattr(config.trading_config.thresholds,
                                    'rngpos_adx_delta_filter_enabled', True)
            if _rpad_enabled and signal in ["LONG", "SHORT"]:
                _pair_adx_now = indicators.get('adx')
                _pair_adx_pre = indicators.get('adx_prev1')
                _price = indicators.get('price')
                _hi20 = indicators.get('high_20')
                _lo20 = indicators.get('low_20')
                _rngpos_val = None
                if (_price is not None and _hi20 is not None and _lo20 is not None
                        and _hi20 != _lo20):
                    _rngpos_val = (_price - _lo20) / (_hi20 - _lo20) * 100
                if (_pair_adx_now is not None and _pair_adx_pre is not None
                        and _rngpos_val is not None):
                    _adx_delta_val2 = _pair_adx_now - _pair_adx_pre
                    _th2 = config.trading_config.thresholds
                    _rpad_key = ('rngpos_adx_delta_filter_long' if signal == 'LONG'
                                 else 'rngpos_adx_delta_filter_short')
                    _rpad_str = getattr(_th2, _rpad_key, '')
                    if _rpad_str and _rpad_str.strip():
                        for _rpad_rule in _rpad_str.split(','):
                            _rpad_rule = _rpad_rule.strip()
                            if not _rpad_rule or ':' not in _rpad_rule:
                                continue
                            try:
                                _rp_part, _ad_part = _rpad_rule.split(':')
                                _rp_lo, _rp_hi = map(float, _rp_part.split('-'))
                                _ad_lo, _ad_hi = map(float, _ad_part.split('-'))
                                if (_rp_lo <= _rngpos_val <= _rp_hi and
                                        _ad_lo <= _adx_delta_val2 < _ad_hi):
                                    logger.info(
                                        f"[RNGPOS_ADX_DELTA_CROSS] {pair}: {signal} blocked — "
                                        f"RngPos {_rngpos_val:.1f} in [{_rp_lo}-{_rp_hi}] "
                                        f"AND ADXΔ {_adx_delta_val2:.2f} in [{_ad_lo}-{_ad_hi})"
                                    )
                                    self._record_filter_block("RNGPOS_ADX_DELTA_CROSS", signal, had_room=_had_room)
                                    self._last_pair_block_reason[pair] = "RNGPOS_ADX_DELTA_CROSS"
                                    signal = "NO_TRADE"
                                    break
                            except (ValueError, TypeError):
                                continue

            # EMA Fan Acceleration (fan_ratio) dead-zone filter (May 29, 2026).
            # fan_ratio = abs(EMA5-EMA8 gap%) / abs(EMA8-EMA13 gap%). The MID-fan band
            # is a clean loser dead-zone (mature/fully-developed trend = entering late,
            # no edge). SHORT active [1.02,1.65); LONG observation-only (rule empty).
            # Block when fan_ratio in any configured [lo, hi) band. UNVALIDATED cross-
            # batch (ema_gap_8_13 only exists May-27+) — validate on next post-May-27 batch.
            _fan_enabled = getattr(config.trading_config.thresholds,
                                   'fan_ratio_filter_enabled', True)
            if _fan_enabled and signal in ["LONG", "SHORT"]:
                _e5 = indicators.get('ema5')
                _e8 = indicators.get('ema8')
                _e13 = indicators.get('ema13')
                _fan_val = None
                if (_e5 is not None and _e8 is not None and _e13 is not None
                        and _e8 != 0 and _e13 != 0):
                    _g58 = abs((_e5 - _e8) / _e8 * 100)
                    _g813 = abs((_e8 - _e13) / _e13 * 100)
                    if _g813 > 0:
                        _fan_val = _g58 / _g813
                if _fan_val is not None:
                    _th3 = config.trading_config.thresholds
                    _fan_key = ('fan_ratio_block_long' if signal == 'LONG'
                                else 'fan_ratio_block_short')
                    _fan_str = getattr(_th3, _fan_key, '')
                    if _fan_str and _fan_str.strip():
                        for _fan_rule in _fan_str.split(','):
                            _fan_rule = _fan_rule.strip()
                            if not _fan_rule or '-' not in _fan_rule:
                                continue
                            try:
                                _fl, _fh = map(float, _fan_rule.split('-'))
                                if _fl <= _fan_val < _fh:
                                    logger.info(
                                        f"[FAN_RATIO_GATE] {pair}: {signal} blocked — "
                                        f"fan_ratio {_fan_val:.2f} in [{_fl}-{_fh})"
                                    )
                                    self._record_filter_block("FAN_RATIO_GATE", signal, had_room=_had_room)
                                    self._last_pair_block_reason[pair] = "FAN_RATIO_GATE"
                                    _fan_dir = 'SHORT' if signal == 'LONG' else 'LONG'
                                    _fan_ef = self._flip_entry_fields(indicators, flip_dir=_fan_dir, scan=self._flip_scan_ctx(locals()))
                                    _seed_phantom_flip(pair, indicators.get('price'), signal, "FAN_RATIO_GATE",
                                                       entry_fields=_fan_ef)
                                    # Jun 14: Flip Entry — fade the block live (both sides)
                                    await self._maybe_open_flip(db, pair, signal, "FAN_RATIO_GATE", indicators,
                                                                entry_fields=_fan_ef)
                                    signal = "NO_TRADE"
                                    break
                            except (ValueError, TypeError):
                                continue
                        else:
                            # FAN_CONTROL (Jun 15) — the for-loop completed with NO block, i.e.
                            # the fan ratio is OUTSIDE every dead-zone band → this entry PASSES
                            # the fan filter (a "clean"/accelerating move the bot would trade).
                            # Seed a phantom fade here too (observation-only, NO live flip) as the
                            # A/B control vs FAN_RATIO_GATE: bucketed by fan ratio × regime it
                            # answers "does the dead-zone band actually select better fades, or is
                            # the edge just regime-alignment?". Forward-only; fail-silent.
                            try:
                                _ctrl_dir = 'SHORT' if signal == 'LONG' else 'LONG'
                                _seed_phantom_flip(pair, indicators.get('price'), signal, "FAN_CONTROL",
                                                   entry_fields=self._flip_entry_fields(indicators, flip_dir=_ctrl_dir, scan=self._flip_scan_ctx(locals())))
                            except Exception:
                                pass

            # Pair ATR minimum filter (June 1, 2026). Block entries when pair
            # ATR% < min — the dead-tape / no-fuel fade zone (mirror of the
            # high-ATR runner finding). LONG <0.25%: 5-batch 12% WR / -$230
            # (sharpest clean loser sub-band), zero overlap with fan>5 and
            # BTC-RSI-50-55 (both this-batch <0.25 LONGs were unique). Spares
            # the low-ATR winners that a <0.40 cut would clip (LTC at 0.29).
            _patr_enabled = getattr(config.trading_config.thresholds,
                                    'pair_atr_filter_enabled', True)
            if _patr_enabled and signal in ["LONG", "SHORT"]:
                _patr_min = getattr(
                    config.trading_config.thresholds,
                    'pair_atr_min_long' if signal == 'LONG' else 'pair_atr_min_short',
                    0.0) or 0.0
                if _patr_min > 0:
                    _patr_atr = indicators.get('atr')
                    _patr_price = indicators.get('price')
                    if (_patr_atr is not None and _patr_price
                            and _patr_price > 0):
                        _patr_pct = (_patr_atr / _patr_price) * 100
                        if _patr_pct < _patr_min:
                            logger.info(
                                f"[PAIR_ATR_MIN] {pair}: {signal} blocked — "
                                f"pair ATR {_patr_pct:.3f}% < min {_patr_min}%"
                            )
                            self._record_filter_block("PAIR_ATR_MIN", signal, had_room=_had_room)
                            self._last_pair_block_reason[pair] = "PAIR_ATR_MIN"
                            signal = "NO_TRADE"
            # Jun 10 — pair ATR CEILING (LONG): distribution guard. Historic max
            # unmatched-long winner = ATR 2.49 (HOME); ESPORTS at 4.68 (p100 outlier
            # meme) was a -$220 DOA. Blocks only out-of-distribution pairs. 0 = off.
            # Jun 10 review fix: stands ALONE (not under pair_atr_filter_enabled) — the
            # master toggle governs the MIN filter; this ceiling must survive it being off.
            if signal == "LONG":
                _patr_max = getattr(config.trading_config.thresholds, 'pair_atr_max_long', 0.0) or 0.0
                if _patr_max > 0:
                    _patr_atr2 = indicators.get('atr'); _patr_price2 = indicators.get('price')
                    if _patr_atr2 is not None and _patr_price2 and _patr_price2 > 0:
                        _patr_pct2 = (_patr_atr2 / _patr_price2) * 100
                        if _patr_pct2 >= _patr_max:
                            logger.info(f"[PAIR_ATR_MAX] {pair}: LONG blocked — pair ATR {_patr_pct2:.3f}% >= max {_patr_max}% (out-of-distribution volatility)")
                            self._record_filter_block("PAIR_ATR_MAX", "LONG", had_room=_had_room)
                            self._last_pair_block_reason[pair] = "PAIR_ATR_MAX"
                            signal = "NO_TRADE"

            # Jun 13 — ATR×GAP LONG block (volatile-and-already-extended quadrant).
            # High-ATR pair that has ALREADY run far above its 4hr trend = buying the
            # exhaustion top → mean-reverts (ENJ -$253/57s). Unmatched longs ATR>=1.0 &
            # gap>=0.5: 31% WR -$611 demux; same high-ATR with gap<0.5 = 64-75% WR
            # POSITIVE (the genuine runner — preserved). gap = (EMA13-EMA50)/EMA50*100,
            # matching the entry_pair_ema20_ema50_gap_pct field. Counter ATR_GAP_LONG.
            if signal == "LONG" and getattr(config.trading_config.thresholds, 'atr_gap_block_long_enabled', False):
                _ag_atr_min = getattr(config.trading_config.thresholds, 'atr_gap_block_atr_min_long', 1.0) or 0.0
                _ag_gap_min = getattr(config.trading_config.thresholds, 'atr_gap_block_gap_min_long', 0.5)
                if _ag_atr_min > 0:
                    _ag_atr = indicators.get('atr'); _ag_price = indicators.get('price')
                    _ag_e13 = indicators.get('ema13'); _ag_e50 = indicators.get('ema50')
                    if (_ag_atr is not None and _ag_price and _ag_price > 0
                            and _ag_e13 is not None and _ag_e50 is not None and _ag_e50 != 0):
                        _ag_atr_pct = (_ag_atr / _ag_price) * 100
                        _ag_gap_pct = (_ag_e13 - _ag_e50) / _ag_e50 * 100
                        if _ag_atr_pct >= _ag_atr_min and _ag_gap_pct >= _ag_gap_min:
                            logger.info(f"[ATR_GAP_LONG] {pair}: LONG blocked — ATR {_ag_atr_pct:.2f}% >= {_ag_atr_min}% AND pair-gap {_ag_gap_pct:.2f}% >= {_ag_gap_min}% (volatile + already-extended → reverts)")
                            self._record_filter_block("ATR_GAP_LONG", "LONG", had_room=_had_room)
                            self._last_pair_block_reason[pair] = "ATR_GAP_LONG"
                            # Jun 15: phantom seed removed — ATR_GAP_LONG fade was ✗ whipsaws
                            # (N=6, -0.057% avg, 50% SL); that zone is chop, neither side pays.
                            signal = "NO_TRADE"

            # Jun 10 — RSI-SPIKE GUARD (LONG): block when the pair's RSI one candle ago was
            # below the floor = RSI teleported from neutral into the entry zone in a single
            # candle = first-candle pump chase (VVV 44.6->65, PIPPIN 45.5->58.3). Complements
            # the fan-window block (fan sees candles 2-5 of a spike; this sees candle 1).
            # 0 = disabled. GATE: drop if it blocks >=3 would-be winners w/ no loser saves.
            if signal == "LONG":
                _rsiprev_min = getattr(config.trading_config.thresholds, 'rsi_prev_min_long', 0.0) or 0.0
                _spike_min_jump = getattr(config.trading_config.thresholds, 'rsi_spike_min_jump_long', 0.0) or 0.0
                _rsi_prev1 = indicators.get('rsi_prev1')
                _rsi_now = indicators.get('rsi')
                if _rsiprev_min > 0 and _rsi_prev1 is not None and _rsi_prev1 < _rsiprev_min:
                    # Jun 10 refinement: require a real 1-candle JUMP too (>= min_jump), so a
                    # 49.8->51 non-spike passes. 0 = jump condition off (pure floor).
                    _jump = (_rsi_now - _rsi_prev1) if _rsi_now is not None else None
                    if _spike_min_jump <= 0 or (_jump is not None and _jump >= _spike_min_jump):
                        logger.info(f"[RSI_SPIKE_GUARD] {pair}: LONG blocked — RSI {_rsi_prev1:.1f}->{(_rsi_now if _rsi_now is not None else 0):.1f} (jump {(_jump if _jump is not None else 0):+.1f}) from below {_rsiprev_min} = single-candle pump chase")
                        self._record_filter_block("RSI_SPIKE_GUARD", "LONG", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "RSI_SPIKE_GUARD"
                        signal = "NO_TRADE"

            # BTC 1h × BTC 5m RSI Direction Cross-Filter (May 26, 2026 PM).
            # Block entry when both BTC RSI timeframes are in specified
            # directions (Rising/Falling). Rule encoded as 2-char codes:
            # "RR" "RF" "FR" "FF" where first=1h dir, second=5m dir.
            # R=Rising (curr > prev), F=Falling (curr <= prev, matches existing
            # BTC 1h × 5m RSI cross-tab convention).
            # Default SHORT="RR" — blocks double-countertrend setup (BTC rising
            # on both timeframes while SHORT signal fires). Cross-batch N=5
            # combined, 60% WR, -$182. 11th locked-discipline override
            # acknowledged per CLAUDE.md May 26 PM watchlist.
            _rsi_dir_enabled = getattr(config.trading_config.thresholds,
                                       'btc_1h_5m_rsi_dir_filter_enabled', True)
            if (_rsi_dir_enabled and signal in ["LONG", "SHORT"]
                    and btc_rsi is not None and btc_rsi_prev is not None
                    and btc_rsi_1h is not None and btc_rsi_1h_prev is not None):
                _th_rsi = config.trading_config.thresholds
                _rsi_dir_key = ('btc_1h_5m_rsi_dir_filter_long' if signal == 'LONG'
                                else 'btc_1h_5m_rsi_dir_filter_short')
                _rsi_dir_str = (getattr(_th_rsi, _rsi_dir_key, '') or '').strip()
                if _rsi_dir_str:
                    _dir_1h = 'R' if btc_rsi_1h > btc_rsi_1h_prev else 'F'
                    _dir_5m = 'R' if btc_rsi > btc_rsi_prev else 'F'
                    _trade_key = f"{_dir_1h}{_dir_5m}"
                    _rules = [r.strip().upper() for r in _rsi_dir_str.split(',') if r.strip()]
                    if _trade_key in _rules:
                        _dir_full = lambda c: 'Rising' if c == 'R' else 'Falling'
                        logger.info(
                            f"[BTC_1H_5M_RSI_DIR_GATE] {pair}: {signal} blocked — "
                            f"1h {_dir_full(_dir_1h)} × 5m {_dir_full(_dir_5m)} "
                            f"(1h RSI {btc_rsi_1h:.1f} vs prev {btc_rsi_1h_prev:.1f}, "
                            f"5m RSI {btc_rsi:.1f} vs prev {btc_rsi_prev:.1f}) "
                            f"matches rule {_trade_key}"
                        )
                        self._record_filter_block("BTC_1H_5M_RSI_DIR_GATE", signal, had_room=_had_room)
                        self._last_pair_block_reason[pair] = "BTC_1H_5M_RSI_DIR_GATE"
                        signal = "NO_TRADE"

            # BTC EMA13-EMA50 Gap × BTC ADX 2D Cross-Filter (May 19, 2026).
            # Catches the "BTC mid-extension + low/climax trend conviction" LONG
            # loser zone that single-axis filters can't express. Cross-batch
            # evidence inside Gap [+0.10, +0.20%]:
            #   - ADX <22: N=31, 39% WR, -$1,022 (5 of 6 dates losing) — block
            #   - ADX 22-25: N=10, 90% WR, +$177 — RESCUE, preserved (open)
            #   - ADX 25-28: N=9, 22% WR, -$415 — block (N=9 override, all 4 dates negative)
            # Rule format: "<gapLo>-<gapHi>:<adxLo>-<adxHi>" — block when BTC
            # EMA13-EMA50 gap in [gapLo, gapHi) AND BTC ADX in [adxLo, adxHi).
            # Multiple rules comma-separated. Empty = inactive.
            _bgad_enabled = getattr(config.trading_config.thresholds,
                                    'btc_gap_btc_adx_filter_enabled', True)
            if (_bgad_enabled and signal in ["LONG", "SHORT"]
                    and btc_ema13 is not None and btc_ema50 is not None
                    and btc_ema50 != 0 and btc_adx is not None):
                _btc_gap_val = ((btc_ema13 - btc_ema50) / btc_ema50) * 100
                _th3 = config.trading_config.thresholds
                _bgad_key = ('btc_gap_btc_adx_filter_long' if signal == 'LONG'
                             else 'btc_gap_btc_adx_filter_short')
                _bgad_str = getattr(_th3, _bgad_key, '')
                if _bgad_str and _bgad_str.strip():
                    for _bgad_rule in _bgad_str.split(','):
                        _bgad_rule = _bgad_rule.strip()
                        if not _bgad_rule or ':' not in _bgad_rule:
                            continue
                        try:
                            _g_part, _a_part = _bgad_rule.split(':')
                            _g_lo, _g_hi = map(float, _g_part.split('-'))
                            _a_lo, _a_hi = map(float, _a_part.split('-'))
                            if (_g_lo <= _btc_gap_val < _g_hi and
                                    _a_lo <= btc_adx < _a_hi):
                                logger.info(
                                    f"[BTC_GAP_BTC_ADX_CROSS] {pair}: {signal} blocked — "
                                    f"BTC Gap {_btc_gap_val:+.3f}% in [{_g_lo}-{_g_hi}) "
                                    f"AND BTC ADX {btc_adx:.1f} in [{_a_lo}-{_a_hi})"
                                )
                                self._record_filter_block("BTC_GAP_BTC_ADX_CROSS", signal, had_room=_had_room)
                                self._last_pair_block_reason[pair] = "BTC_GAP_BTC_ADX_CROSS"
                                signal = "NO_TRADE"
                                break
                        except (ValueError, TypeError):
                            continue

            # BTC ATR × BTC ADX 2D Cross-Filter (May 22, 2026).
            # Cross-batch evidence — SHORT at strong BTC trend (ADX≥30) needs
            # volatility; dead-quiet BTC = exhausted + squeeze ammo. See CLAUDE.md
            # May 22 entry. LONG mirror shows OPPOSITE pattern → asymmetric filter.
            # Default ships SHORT-only rule "0.0-0.10:30-999".
            _batr_enabled = getattr(config.trading_config.thresholds,
                                    'btc_atr_btc_adx_filter_enabled', True)
            _btc_atr_val = btc_atr_pct
            if (_batr_enabled and signal in ["LONG", "SHORT"]
                    and _btc_atr_val is not None and btc_adx is not None):
                _th4 = config.trading_config.thresholds
                _batr_key = ('btc_atr_btc_adx_filter_long' if signal == 'LONG'
                             else 'btc_atr_btc_adx_filter_short')
                _batr_str = getattr(_th4, _batr_key, '')
                if _batr_str and _batr_str.strip():
                    for _batr_rule in _batr_str.split(','):
                        _batr_rule = _batr_rule.strip()
                        if not _batr_rule or ':' not in _batr_rule:
                            continue
                        try:
                            _at_part, _ax_part = _batr_rule.split(':')
                            _at_lo, _at_hi = map(float, _at_part.split('-'))
                            _ax_lo, _ax_hi = map(float, _ax_part.split('-'))
                            if (_at_lo <= _btc_atr_val < _at_hi and
                                    _ax_lo <= btc_adx < _ax_hi):
                                logger.info(
                                    f"[BTC_ATR_BTC_ADX_CROSS] {pair}: {signal} blocked — "
                                    f"BTC ATR {_btc_atr_val:.3f}% in [{_at_lo}-{_at_hi}) "
                                    f"AND BTC ADX {btc_adx:.1f} in [{_ax_lo}-{_ax_hi})"
                                )
                                self._record_filter_block("BTC_ATR_BTC_ADX_CROSS", signal, had_room=_had_room)
                                self._last_pair_block_reason[pair] = "BTC_ATR_BTC_ADX_CROSS"
                                signal = "NO_TRADE"
                                break
                        except (ValueError, TypeError):
                            continue

            # BTC Trend Filter — runs independently of Macro Trend toggle (May 5).
            # Compares BTC EMA13 vs BTC EMA50 on the 5m chart (May 6 — switched from
            # EMA20 to EMA13 for faster reversal detection; EMA13 spans ~65 min vs EMA20's
            # 100 min, EMA50 spans ~250 min ~4 hours).
            # Blocks countertrend entries:
            #   EMA13 > EMA50 → BTC in medium-term uptrend → block SHORTs
            #   EMA13 < EMA50 → BTC in medium-term downtrend → block LONGs
            # Addresses the case where short-horizon (15min) BTC slope flips
            # bearish during a brief pullback within a multi-hour bullish trend
            # (and vice versa). See CLAUDE.md May 5 entry on BTC Trend Filter.
            _btc_trend_enabled = getattr(config.trading_config.thresholds, 'btc_trend_filter_enabled', False)
            if signal in ["LONG", "SHORT"]:
                _gap_pct_dbg = (((btc_ema13 - btc_ema50) / btc_ema50) * 100) if (btc_ema13 and btc_ema50) else None
                _gap_str_dbg = f"{_gap_pct_dbg:.4f}%" if _gap_pct_dbg is not None else "N/A"
                logger.info(
                    f"[DEBUG_TREND] {pair} {signal} {confidence}: filter_enabled={_btc_trend_enabled} "
                    f"btc_ema13={btc_ema13} btc_ema50={btc_ema50} gap={_gap_str_dbg}"
                )

            if (signal in ["LONG", "SHORT"]
                    and _btc_trend_enabled
                    and btc_ema13 is not None and btc_ema50 is not None):
                if signal == "LONG" and btc_ema13 < btc_ema50:
                    logger.info(
                        f"[BTC_TREND_FILTER] {pair}: LONG blocked — BTC EMA13 {btc_ema13:.2f} < EMA50 {btc_ema50:.2f} "
                        f"(macro downtrend, countertrend LONG blocked)"
                    )
                    self._record_filter_block("BTC_TREND_FILTER", "LONG", had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_TREND_FILTER"
                    signal = "NO_TRADE"
                elif signal == "SHORT" and btc_ema13 > btc_ema50:
                    logger.info(
                        f"[BTC_TREND_FILTER] {pair}: SHORT blocked — BTC EMA13 {btc_ema13:.2f} > EMA50 {btc_ema50:.2f} "
                        f"(macro uptrend, countertrend SHORT blocked)"
                    )
                    self._record_filter_block("BTC_TREND_FILTER", "SHORT", had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_TREND_FILTER"
                    signal = "NO_TRADE"
                else:
                    logger.info(f"[BTC_TREND_FILTER_PASS] {pair} {signal}: btc_ema20={btc_ema20:.2f} btc_ema50={btc_ema50:.2f} (passed)")

            # BTC Slope directional check — runs independently of Macro Trend toggle.
            # For LONG: require BTC slope >= +flat_threshold_long (BTC rising meaningfully)
            # For SHORT: require BTC slope <= -flat_threshold_short (BTC falling meaningfully)
            # When the threshold is 0, the check is a no-op (allows any slope including flat/opposite).
            if signal in ["LONG", "SHORT"] and btc_ema20_slope_pct is not None:
                _th = config.trading_config.thresholds
                if signal == "LONG":
                    _flat_th = getattr(_th, 'macro_trend_flat_threshold_long',
                                       getattr(_th, 'macro_trend_flat_threshold', 0))
                    if _flat_th > 0 and btc_ema20_slope_pct < _flat_th:
                        logger.info(f"[BTC_SLOPE_GATE] {pair}: LONG blocked — BTC slope {btc_ema20_slope_pct:+.4f}% < min +{_flat_th}%")
                        self._record_filter_block("BTC_SLOPE_GATE", "LONG", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "BTC_SLOPE_GATE"
                        signal = "NO_TRADE"
                else:  # SHORT
                    _flat_th = getattr(_th, 'macro_trend_flat_threshold_short',
                                       getattr(_th, 'macro_trend_flat_threshold', 0))
                    if _flat_th > 0 and btc_ema20_slope_pct > -_flat_th:
                        logger.info(f"[BTC_SLOPE_GATE] {pair}: SHORT blocked — BTC slope {btc_ema20_slope_pct:+.4f}% > max -{_flat_th}%")
                        self._record_filter_block("BTC_SLOPE_GATE", "SHORT", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "BTC_SLOPE_GATE"
                        signal = "NO_TRADE"

            # May 2: BTC EMA20 slope MAX guard. Block over-extended BTC trends
            # (late-cycle entries when BTC has already run too far). 0 = disabled.
            if signal in ["LONG", "SHORT"] and btc_ema20_slope_pct is not None:
                _th = config.trading_config.thresholds
                _btc_max = getattr(_th, f'btc_ema20_slope_max_{signal.lower()}', 0)
                if _btc_max and _btc_max > 0 and abs(btc_ema20_slope_pct) > _btc_max:
                    logger.info(f"[BTC_SLOPE_MAX_GATE] {pair}: {signal} blocked — abs(BTC slope) {abs(btc_ema20_slope_pct):.4f}% > max {_btc_max}%")
                    self._record_filter_block("BTC_SLOPE_MAX_GATE", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_SLOPE_MAX_GATE"
                    signal = "NO_TRADE"

            # May 24: BTC 1h Slope MAX guard. Block late-stage steep-rising BTC LONG
            # entries (and symmetric SHORT if configured). Uses the 1h-timeframe slope
            # captured globally by the BTC scan loop. 0 = disabled.
            # LONG @ slope > +0.15%: 26 trades / 30.8% WR / -$837 today, 14 trades in
            # the 0.15-0.20 cliff at 21.4% WR. Mechanism: BTC in late-stage rising
            # trend → mean reversion → countertrend LONG bounces fail.
            if signal in ["LONG", "SHORT"] and _current_btc_1h_slope is not None:
                _th = config.trading_config.thresholds
                _btc_1h_max = getattr(_th, f'btc_1h_slope_max_{signal.lower()}', 0)
                # May 24 (evening) — SHORT semantics REVERSED to block COUNTERTREND SHORTs
                # (SHORTs into rising BTC). Cross-batch evidence: SHORTs at BTC 1h slope > +0.10
                # are catastrophic (N=6, 1W only +$12, others NP/loser, -$236 total).
                # Both LONG and SHORT now block when slope > max (intuitive: max is the
                # upper bound on BTC strength a same-direction entry tolerates).
                if _btc_1h_max and _btc_1h_max > 0 and _current_btc_1h_slope > _btc_1h_max:
                    if signal == "LONG":
                        logger.info(f"[BTC_1H_SLOPE_MAX_GATE] {pair}: LONG blocked — BTC 1h slope {_current_btc_1h_slope:+.4f}% > max +{_btc_1h_max}% (late-stage rising trend)")
                        self._record_filter_block("BTC_1H_SLOPE_MAX_GATE", "LONG", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "BTC_1H_SLOPE_MAX_GATE"
                        signal = "NO_TRADE"
                    elif signal == "SHORT":
                        logger.info(f"[BTC_1H_SLOPE_MAX_GATE] {pair}: SHORT blocked — BTC 1h slope {_current_btc_1h_slope:+.4f}% > max +{_btc_1h_max}% (countertrend SHORT in rising BTC)")
                        self._record_filter_block("BTC_1H_SLOPE_MAX_GATE", "SHORT", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "BTC_1H_SLOPE_MAX_GATE"
                        signal = "NO_TRADE"

                # Jun 3 — BTC 1h Slope MIN floor (higher-TF macro). Blocks entries when the
                # 1h slope is too steeply NEGATIVE = entering into a steep 1h crash =
                # exhaustion/mean-reversion bounce. 0 = disabled; a negative value activates.
                # SHORT cross-batch: 1h slope < -0.60 = 0W/4L (SEI, XRP, BTC, JTO). LONG off.
                _btc_1h_min = getattr(_th, f'btc_1h_slope_min_{signal.lower()}', 0) if signal in ["LONG", "SHORT"] else 0
                if _btc_1h_min and _current_btc_1h_slope < _btc_1h_min:
                    logger.info(f"[BTC_1H_SLOPE_MIN_GATE] {pair}: {signal} blocked — BTC 1h slope {_current_btc_1h_slope:+.4f}% < min {_btc_1h_min}% (exhaustion: entering steep 1h crash)")
                    self._record_filter_block("BTC_1H_SLOPE_MIN_GATE", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_1H_SLOPE_MIN_GATE"
                    signal = "NO_TRADE"

            # Jun 10 — BTC 1h RSI FLOOR (SHORT). Block shorting when BTC's HOURLY RSI is
            # already deep-oversold = shorting into the hourly bounce zone (the 1h twin of
            # the 5m climax-oversold cross-filter block). Cross-batch matched shorts:
            # 1hRSI<30 = -$940 · 30-35 = -$382 · 35-40 = +$651 (monotonic). 0 = disabled.
            if signal == "SHORT" and btc_rsi_1h is not None:
                _rsi1h_min = getattr(config.trading_config.thresholds, 'btc_rsi_1h_min_short', 0) or 0
                if _rsi1h_min > 0 and btc_rsi_1h < _rsi1h_min:
                    logger.info(f"[BTC_1H_RSI_MIN_GATE] {pair}: SHORT blocked — BTC 1h RSI {btc_rsi_1h:.1f} < min {_rsi1h_min} (hourly oversold: bounce risk)")
                    self._record_filter_block("BTC_1H_RSI_MIN_GATE", "SHORT", had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_1H_RSI_MIN_GATE"
                    signal = "NO_TRADE"

            # Jun 3 — BTC-ACCELERATION CHASE filter (STATEFUL, evolution vs last entry).
            # Block a LONG when live BTC EMA20 slope is HIGHER than at the most recent
            # LONG that opened within the window = BTC accelerated since the last entry
            # = chasing a maturing move. Cross-batch (7-batch, 30min): 30.8% WR block
            # cohort. LONG only (SHORT side untested). Reference auto-expires after window.
            _th_evo = config.trading_config.thresholds
            if signal == "LONG" and getattr(_th_evo, 'evo_chase_filter_long_enabled', False):
                _evo_win = getattr(_th_evo, 'evo_chase_window_min', 30)
                _last_ts = self._last_long_open_ts
                _last_slp = self._last_long_open_btc_ema20_slope
                if (_last_ts is not None and _last_slp is not None
                        and (datetime.utcnow() - _last_ts).total_seconds() <= _evo_win * 60
                        and _btc_ema20_slope_pct > _last_slp):
                    logger.info(f"[BTC_ACCEL_CHASE_LONG] {pair}: LONG blocked — BTC EMA20 slope {_btc_ema20_slope_pct:.4f} > last-LONG {_last_slp:.4f} (chasing accelerating BTC)")
                    self._record_filter_block("BTC_ACCEL_CHASE_LONG", "LONG", had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BTC_ACCEL_CHASE_LONG"
                    signal = "NO_TRADE"

            # May 2: per-pair EMA20 slope MAX guard. Block over-extended pair trends.
            # Computes the slope locally from indicators (pair_ema20_slope_pct is
            # only computed later in the entry-payload section). Matches the
            # formula used by momentum_ema20_slope_min_* check in services/indicators.py.
            # 0 = disabled.
            if signal in ["LONG", "SHORT"]:
                _th = config.trading_config.thresholds
                _pair_max = getattr(_th, f'momentum_ema20_slope_max_{signal.lower()}', 0)
                if _pair_max and _pair_max > 0:
                    _ema20_now = indicators.get('ema20')
                    _ema20_p3 = indicators.get('ema20_prev3')
                    if _ema20_now is not None and _ema20_p3 is not None and _ema20_p3 != 0:
                        _pair_slope_abs = abs((_ema20_now - _ema20_p3) / _ema20_p3 * 100)
                        if _pair_slope_abs > _pair_max:
                            logger.info(f"[PAIR_SLOPE_MAX_GATE] {pair}: {signal} blocked — abs(pair slope) {_pair_slope_abs:.4f}% > max {_pair_max}%")
                            self._record_filter_block("PAIR_SLOPE_MAX_GATE", signal, had_room=_had_room)
                            self._last_pair_block_reason[pair] = "PAIR_SLOPE_MAX_GATE"
                            signal = "NO_TRADE"

            # May 7: Pair Trend Filter (pair-level analog of BTC Trend Filter).
            # Compares pair EMA13 vs EMA50 to block countertrend entries:
            #   LONG with pair_ema13 < pair_ema50 → pair in 4hr downtrend
            #   SHORT with pair_ema13 > pair_ema50 → pair in 4hr uptrend
            # 6-trade cross-sample evidence: May 5 SHORTs vs uptrend (4 lost) +
            # May 7 LONGs vs downtrend (2 lost) = 0/6. Defensive ship default ON.
            # Pair Trend Filter — pair EMA13 vs EMA50, Jun 13: per-direction split.
            # LONG: block when EMA13 < EMA50 (countertrend long). Currently OFF
            #   (gap<0 unmatched longs are ~breakeven, 58% WR — not worth blocking).
            # SHORT: block when pair gap >= short_gap_max (default 0 = EMA13>EMA50 =
            #   shorting before the breakdown confirms → bounces). Counter PAIR_TREND_FILTER.
            if signal in ["LONG", "SHORT"]:
                _th_pt = config.trading_config.thresholds
                _pt_long_en = getattr(_th_pt, 'pair_trend_filter_long_enabled', False)
                _pt_short_en = getattr(_th_pt, 'pair_trend_filter_short_enabled', True)
                _pt_short_gap_max = getattr(_th_pt, 'pair_trend_short_gap_max', 0.0)
                if (signal == "LONG" and _pt_long_en) or (signal == "SHORT" and _pt_short_en):
                    _pair_ema13 = indicators.get('ema13')
                    _pair_ema50 = indicators.get('ema50')
                    if _pair_ema13 is not None and _pair_ema50 is not None and _pair_ema50 != 0:
                        _pair_gap_pct = (_pair_ema13 - _pair_ema50) / _pair_ema50 * 100
                        if signal == "LONG" and _pair_ema13 < _pair_ema50:
                            logger.info(
                                f"[PAIR_TREND_FILTER] {pair}: LONG blocked — pair EMA13 {_pair_ema13:.6f} < EMA50 {_pair_ema50:.6f} "
                                f"(gap {_pair_gap_pct:.4f}% — pair in 4hr downtrend, countertrend LONG blocked)"
                            )
                            self._record_filter_block("PAIR_TREND_FILTER", "LONG", had_room=_had_room)
                            self._last_pair_block_reason[pair] = "PAIR_TREND_FILTER"
                            signal = "NO_TRADE"
                        elif signal == "SHORT" and _pair_gap_pct >= _pt_short_gap_max:
                            logger.info(
                                f"[PAIR_TREND_FILTER] {pair}: SHORT blocked — pair gap {_pair_gap_pct:.4f}% >= {_pt_short_gap_max}% "
                                f"(pair not yet below its 4hr trend → shorting before breakdown confirms → bounces)"
                            )
                            self._record_filter_block("PAIR_TREND_FILTER", "SHORT", had_room=_had_room)
                            self._last_pair_block_reason[pair] = "PAIR_TREND_FILTER"
                            _seed_phantom_flip(pair, indicators.get('price'), "SHORT", "PAIR_TREND_FILTER",
                                               entry_fields=self._flip_entry_fields(indicators, flip_dir="LONG", scan=self._flip_scan_ctx(locals())))
                            signal = "NO_TRADE"

            if signal in ["LONG", "SHORT"]:
                _th = config.trading_config.thresholds
                global_vol_blocks = False
                if getattr(_th, 'global_volume_filter_enabled', False):
                    _gv_thresh = getattr(_th, f'global_volume_threshold_{signal.lower()}', 1.05)
                    if _global_volume_ratio < _gv_thresh:
                        # May 10 evening: intersection-style rescue. If pair's
                        # absolute 24h USD volume is ≥ rescue threshold, the pair
                        # is large enough to sustain its own momentum even in a
                        # quiet global market — let it through. 0 = no rescue.
                        _pair_vol_rescue = getattr(_th, f'pair_volume_usd_rescue_{signal.lower()}', 0.0)
                        # May 25: rescue MAX ceiling. Rescue only fires when
                        # GVol < this value. Above ceiling but below threshold
                        # = block (no rescue). 0 = no ceiling. Cross-batch
                        # evidence: GVol 0.60-0.70 LONG rescue zone = N=36,
                        # 47% WR, -$717 (structural loser). GVol <0.60 = +$62
                        # winner. Default 0.60 LONG isolates the loser zone.
                        _rescue_max = getattr(_th, f'global_volume_rescue_max_{signal.lower()}', 0.0)
                        _rescue_zone_ok = (_rescue_max <= 0) or (_global_volume_ratio < _rescue_max)
                        if _pair_vol_rescue > 0 and volume_24h >= _pair_vol_rescue and _rescue_zone_ok:
                            logger.info(f"[VOL_GATE_RESCUE] {pair}: {signal} GlobalVol {_global_volume_ratio:.2f}<{_gv_thresh} BUT PairVol ${volume_24h/1e6:.0f}M ≥ ${_pair_vol_rescue/1e6:.0f}M (rescue_max={_rescue_max:.2f}) — rescued")
                        else:
                            global_vol_blocks = True

                pair_vol_blocks = False
                if getattr(_th, 'pair_volume_filter_enabled', False):
                    _pv_thresh = getattr(_th, f'pair_volume_threshold_{signal.lower()}', 1.10)
                    if _pair_volume_ratio < _pv_thresh:
                        pair_vol_blocks = True

                # SHORT-only MAX-side GlobalVol cap with BTC CAPITULATION OVERRIDE
                # (May 11, 2026 — multi-axis filter per CLAUDE.md SHORT capitulation finding).
                # Block SHORTs at high GlobalVol UNLESS BTC is in capitulation state
                # (deep oversold + falling = selling climax = SHORT-friendly cascade).
                # Pool evidence: 47 SHORTs at GlobalVol >1.05 across 5 batches split as:
                #   - Capitulation (BTC RSI <30 AND slope <0): N=19, 63% WR, +$157 ★ (preserve)
                #   - Non-capitulation: N=28, 29% WR, -$243 ✗ (block)
                # Runs independently of global_volume_filter_enabled toggle (additive).
                global_vol_max_blocks = False
                _gv_max_thresh = None
                _capitulation_override = False
                if signal == "SHORT":
                    _gv_max_thresh = getattr(_th, 'global_volume_max_short', 0.0)
                    if _gv_max_thresh > 0 and _global_volume_ratio > _gv_max_thresh:
                        # Check capitulation override: BTC RSI < threshold AND BTC slope < threshold
                        _cap_rsi_thresh = getattr(_th, 'global_volume_max_short_capitulation_rsi', 30.0)
                        _cap_slope_thresh = getattr(_th, 'global_volume_max_short_capitulation_slope', 0.0)
                        # May 27 2026: GV CAP on the capitulation override.
                        # Extreme GV (e.g. TON 5/27 at GV 5.24 + capitulation = -$232) blows past
                        # the override's protective rationale. If gv_cap > 0, override only fires
                        # when GlobalVol ≤ gv_cap. SHORT blocked when GV > gv_cap regardless of capitulation.
                        _cap_gv_cap = getattr(_th, 'global_volume_max_short_capitulation_gv_cap', 0.0)
                        # Jun 5 2026: master toggle — when disabled, the override never fires
                        # (high-GV SHORTs always blocked, no capitulation rescue).
                        _cap_override_enabled = getattr(_th, 'global_volume_max_short_capitulation_override_enabled', True)
                        _cap_match = (_cap_override_enabled and btc_rsi is not None and btc_ema20_slope_pct is not None
                                      and btc_rsi < _cap_rsi_thresh and btc_ema20_slope_pct < _cap_slope_thresh)
                        _gv_cap_exceeded = (_cap_gv_cap > 0 and _global_volume_ratio > _cap_gv_cap)
                        if _cap_match and not _gv_cap_exceeded:
                            _capitulation_override = True
                            logger.info(
                                f"[VOL_GATE_MAX_OVERRIDE] {pair}: SHORT allowed despite "
                                f"GlobalVol {_global_volume_ratio:.2f} > {_gv_max_thresh} — "
                                f"BTC capitulation (RSI {btc_rsi:.1f} < {_cap_rsi_thresh}, "
                                f"slope {btc_ema20_slope_pct:+.3f} < {_cap_slope_thresh})"
                            )
                        else:
                            global_vol_max_blocks = True
                            if _cap_match and _gv_cap_exceeded:
                                logger.info(
                                    f"[VOL_GATE_MAX_CAP_OVERRIDE_CAPPED] {pair}: SHORT blocked — "
                                    f"BTC capitulation met BUT GlobalVol {_global_volume_ratio:.2f} > "
                                    f"GV cap {_cap_gv_cap} (override capped)"
                                )

                if global_vol_blocks or pair_vol_blocks or global_vol_max_blocks:
                    if global_vol_max_blocks:
                        reason = (
                            f"GlobalVol {_global_volume_ratio:.2f} > {_gv_max_thresh} (SHORT max cap) "
                            f"and NOT in BTC capitulation"
                        )
                        logger.info(f"[VOL_GATE_MAX_SHORT] {pair}: SHORT blocked — {reason}")
                        self._record_filter_block("VOL_GATE_MAX_SHORT", signal, had_room=_had_room)
                        self._last_pair_block_reason[pair] = "VOL_GATE_MAX_SHORT"
                    else:
                        if global_vol_blocks:
                            reason = f"Global Vol {_global_volume_ratio:.2f} < {_gv_thresh} for {signal}"
                        else:
                            reason = f"Pair Vol {_pair_volume_ratio:.2f} < {_pv_thresh} for {signal}"
                        logger.info(f"[VOL-GATE] {pair}: {signal} blocked — {reason}")
                        self._record_filter_block("VOL_GATE", signal, had_room=_had_room)
                        self._last_pair_block_reason[pair] = "VOL_GATE"
                    signal = "NO_TRADE"

            if signal in ["LONG", "SHORT"] and _breadth_enabled:
                if signal == "LONG" and _market_bull_pct < _breadth_bull_th:
                    logger.info(f"[BREADTH_GATE] {pair}: LONG blocked — Bull% {_market_bull_pct:.1f}% < {_breadth_bull_th}%")
                    self._record_filter_block("BREADTH_GATE", "LONG", had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BREADTH_GATE"
                    signal = "NO_TRADE"
                elif signal == "SHORT" and _market_bear_pct < _breadth_bear_th:
                    logger.info(f"[BREADTH_GATE] {pair}: SHORT blocked — Bear% {_market_bear_pct:.1f}% < {_breadth_bear_th}%")
                    self._record_filter_block("BREADTH_GATE", "SHORT", had_room=_had_room)
                    self._last_pair_block_reason[pair] = "BREADTH_GATE"
                    signal = "NO_TRADE"

            await self.update_pair_data(db, pair, indicators, signal, confidence, volume_24h, _pair_volume_ratio)
            if signal in ["LONG", "SHORT"]:
                logger.info(f"[DEBUG_AFTER_PAIRDATA] {pair} {signal} {confidence}: signal still valid after PairData write")

            # --- SPIKE GUARD: block entries during abnormal candles ---
            _sg = config.trading_config.thresholds
            _spike_guard_on = getattr(_sg, 'spike_guard_enabled', False)
            if signal in ["LONG", "SHORT"] and _spike_guard_on:
                _sg_blocked = False
                _sg_reason = ""
                _candle_vol = indicators.get('candle_volume_raw')
                _candle_avg = indicators.get('candle_avg_volume_20')
                _candle_open = indicators.get('candle_open')
                _candle_close = indicators.get('price')
                _ema20 = indicators.get('ema20')
                _sg_vol_mult = getattr(_sg, 'spike_guard_volume_multiplier', 3.0)
                _sg_price_pct = getattr(_sg, 'spike_guard_price_move_pct', 1.5)

                # Volume spike + price move on current candle
                if _candle_vol and _candle_avg and _candle_avg > 0 and _candle_open and _candle_close and _candle_open > 0:
                    _vol_ratio = _candle_vol / _candle_avg
                    _price_move = abs((_candle_close - _candle_open) / _candle_open) * 100
                    if _vol_ratio >= _sg_vol_mult and _price_move >= _sg_price_pct:
                        _sg_blocked = True
                        _sg_reason = f"Volume spike {_vol_ratio:.1f}x + price move {_price_move:.2f}%"

                if _sg_blocked:
                    logger.info(f"[SPIKE_GUARD] {pair}: {signal} blocked — {_sg_reason}")
                    self._record_filter_block("SPIKE_GUARD", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "SPIKE_GUARD"
                    signal = "NO_TRADE"

            if signal in ["LONG", "SHORT"] and confidence and confidence != "NO_TRADE":
                logger.info(f"[DEBUG_REACHED_OPEN] {pair} {signal} {confidence}: about to call open_position()")
                logger.info(f"[SIGNAL] {pair}: {signal} with {confidence} confidence - Opening position...")
                entry_gap = None
                if indicators.get('ema5') and indicators.get('ema20') and indicators['price'] > 0:
                    entry_gap = round(abs((indicators['ema5'] - indicators['ema20']) / indicators['price'] * 100), 4)
                entry_ema_gap_5_8 = None
                if indicators.get('ema5') and indicators.get('ema8') and indicators['ema8'] > 0:
                    entry_ema_gap_5_8 = round(abs((indicators['ema5'] - indicators['ema8']) / indicators['ema8'] * 100), 4)
                entry_ema_gap_8_13 = None
                if indicators.get('ema8') and indicators.get('ema13') and indicators['ema13'] > 0:
                    entry_ema_gap_8_13 = round(abs((indicators['ema8'] - indicators['ema13']) / indicators['ema13'] * 100), 4)
                entry_ema5_stretch = None
                entry_price_vs_ema5_pct = None
                if indicators.get('ema5') and indicators['price'] > 0:
                    entry_ema5_stretch = round(abs(indicators['price'] - indicators['ema5']) / indicators['price'] * 100, 4)
                    entry_price_vs_ema5_pct = round((indicators['price'] - indicators['ema5']) / indicators['ema5'] * 100, 4)
                entry_rsi = indicators.get('rsi')
                # May 15: pair RSI direction = compare to rsi_prev2 (matches RSI Momentum Filter logic
                # which gates on rsi vs rsi_prev2). Stored as entry_rsi_prev but represents prev2.
                entry_rsi_prev = indicators.get('rsi_prev2')
                entry_adx = indicators.get('adx')
                entry_adx_prev = indicators.get('adx_prev1')
                if btc_global_enabled:
                    entry_regime = btc_regime
                else:
                    _th_cfg = config.trading_config.thresholds
                    if signal == "LONG":
                        flat_th = getattr(_th_cfg, 'macro_trend_flat_threshold_long', _th_cfg.macro_trend_flat_threshold)
                    else:
                        flat_th = getattr(_th_cfg, 'macro_trend_flat_threshold_short', _th_cfg.macro_trend_flat_threshold)
                    entry_regime = determine_macro_regime(
                        indicators.get('ema20'), indicators.get('ema20_prev3'), flat_th
                    )
                pair_ema20_slope_pct = None
                pair_ema20 = indicators.get('ema20')
                pair_ema20_prev3 = indicators.get('ema20_prev3')
                if pair_ema20 and pair_ema20_prev3 and pair_ema20_prev3 != 0:
                    pair_ema20_slope_pct = round(((pair_ema20 - pair_ema20_prev3) / pair_ema20_prev3) * 100, 4)
                entry_quality_score = _calculate_quality_score(
                    signal, entry_rsi, entry_adx, entry_gap,
                    _market_bull_pct, _market_bear_pct, btc_adx, pair_ema20_slope_pct
                )
                # Entry Quality Score block filter (May 15 PM) — opt-in.
                # Toggle + threshold. When enabled, blocks entries with
                # entry_quality_score <= block_max. Threshold matches table
                # semantics: block_max=1 → blocks Score 0 AND Score 1.
                # Cross-sample evidence (CLAUDE.md May 15 watchlist): Score ≤ 1
                # across 10 archived samples + today = N=95, 34.7% WR, −$684,
                # direction-consistent loser.
                _qs_enabled = getattr(config.trading_config.thresholds, 'entry_quality_score_filter_enabled', False)
                _qs_block_max = getattr(config.trading_config.thresholds, 'entry_quality_score_block_max', 1)
                if _qs_enabled and entry_quality_score <= _qs_block_max:
                    logger.info(
                        f"[QUALITY_SCORE_GATE] {pair}: {signal} blocked — entry_quality_score={entry_quality_score} <= block_max={_qs_block_max}"
                    )
                    self._record_filter_block("ENTRY_QUALITY_SCORE", signal, had_room=_scan_had_room_snapshot)
                    self._last_pair_block_reason[pair] = "ENTRY_QUALITY_SCORE"
                    continue
                entry_btc_regime = classify_btc_regime(btc_adx, btc_rsi, btc_ema20_slope_pct)

                # Exploration Analytics (Apr 28) — observation-only fields
                _entry_pos_di = indicators.get('pos_di')
                _entry_neg_di = indicators.get('neg_di')
                _entry_atr_pct = None
                _atr = indicators.get('atr')
                if _atr is not None and indicators.get('price') and indicators['price'] > 0:
                    _entry_atr_pct = round((_atr / indicators['price']) * 100, 4)
                _entry_ema50_slope = None
                _ema50 = indicators.get('ema50')
                _ema50_prev12 = indicators.get('ema50_prev12')
                if _ema50 is not None and _ema50_prev12 is not None and _ema50_prev12 != 0:
                    _entry_ema50_slope = round(((_ema50 - _ema50_prev12) / _ema50_prev12) * 100, 4)
                # Pair EMA13 vs EMA50 gap (observation-only; May 6 — switched from EMA20→EMA13
                # for consistency with BTC Trend Filter switch). Field name kept for storage compat;
                # values stored before May 6 deploy use EMA20/EMA50, after use EMA13/EMA50.
                _entry_pair_ema20_ema50_gap_pct = None
                _ema13_val = indicators.get('ema13')
                if _ema13_val is not None and _ema50 is not None and _ema50 != 0:
                    _entry_pair_ema20_ema50_gap_pct = round((_ema13_val - _ema50) / _ema50 * 100, 4)
                # May 13 PM: Entry Distance from EMA13 (Late Entry Risk dimension).
                # Signed: positive = price above EMA13 (LONG chasing), negative = below (SHORT late).
                _entry_dist_from_ema13_pct = None
                _entry_price = indicators.get('price')
                if _ema13_val is not None and _entry_price is not None and _ema13_val != 0:
                    _entry_dist_from_ema13_pct = round((_entry_price - _ema13_val) / _ema13_val * 100, 4)
                # May 14: BTC Market Extension / BTC Late Regime Risk dimension.
                # Signed: positive = BTC price above EMA13 (LONG-risk: chasing market top),
                # negative = BTC below EMA13 (SHORT-risk: late after capitulation).
                _entry_btc_dist_from_ema13_pct = None
                if _current_btc_ema13 is not None and _current_btc_price is not None and _current_btc_ema13 != 0:
                    _entry_btc_dist_from_ema13_pct = round((_current_btc_price - _current_btc_ema13) / _current_btc_ema13 * 100, 4)
                _entry_funding_rate = None
                try:
                    _funding = await binance_service.fetch_funding_rate(symbol)
                    if _funding is not None:
                        _entry_funding_rate = round(_funding, 6)
                except Exception:
                    pass

                order = await self.open_position(
                    db=db,
                    pair=pair,
                    direction=signal,
                    confidence=confidence,
                    current_price=indicators['price'],
                    entry_gap=entry_gap,
                    entry_ema_gap_5_8=entry_ema_gap_5_8,
                    entry_ema_gap_8_13=entry_ema_gap_8_13,
                    entry_ema5_stretch=entry_ema5_stretch,
                    entry_rsi=round(entry_rsi, 2) if entry_rsi is not None else None,
                    entry_rsi_prev=round(entry_rsi_prev, 2) if entry_rsi_prev is not None else None,
                    entry_adx=round(entry_adx, 4) if entry_adx is not None else None,
                    entry_adx_prev=round(entry_adx_prev, 4) if entry_adx_prev is not None else None,
                    entry_macro_trend=entry_regime,
                    entry_ema20_slope=pair_ema20_slope_pct,
                    entry_btc_ema20_slope=btc_ema20_slope_pct,
                    entry_btc_adx=round(btc_adx, 4) if btc_adx is not None else None,
                    entry_btc_adx_prev=round(btc_adx_prev, 4) if btc_adx_prev is not None else None,
                    entry_btc_rsi=round(btc_rsi, 1) if btc_rsi is not None else None,
                    entry_btc_rsi_prev=round(btc_rsi_prev, 1) if btc_rsi_prev is not None else None,
                    entry_btc_rsi_prev6=round(btc_rsi_prev6, 1) if btc_rsi_prev6 is not None else None,
                    entry_btc_atr_pct=btc_atr_pct,
                    entry_btc_rsi_1h=btc_rsi_1h,
                    entry_btc_rsi_1h_prev=btc_rsi_1h_prev,
                    entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
                    entry_global_volume_ratio=round(_global_volume_ratio, 4),
                    entry_pair_volume_ratio=round(_pair_volume_ratio, 4),
                    entry_bull_pct=_market_bull_pct,
                    entry_bear_pct=_market_bear_pct,
                    entry_range_position=round(((indicators['price'] - indicators['low_20']) / (indicators['high_20'] - indicators['low_20'])) * 100, 1) if indicators.get('high_20') and indicators.get('low_20') and indicators['high_20'] != indicators['low_20'] else None,
                    entry_adx_delta=round(entry_adx - entry_adx_prev, 4) if entry_adx is not None and entry_adx_prev is not None else None,
                    entry_quality_score=entry_quality_score,
                    entry_btc_regime=entry_btc_regime,
                    # entry_btc_trend_gap_pct is handled inside open_position via globals lookup
                    # (see line ~1840 — Order() constructor reads _current_btc_trend_gap_pct directly).
                    # Passing it as a kwarg was a bug — open_position's signature doesn't accept it,
                    # which TypeError'd every scan loop and prevented ALL position openings (May 5).
                    entry_pos_di=_entry_pos_di,
                    entry_neg_di=_entry_neg_di,
                    entry_atr_pct=_entry_atr_pct,
                    entry_ema50_slope=_entry_ema50_slope,
                    entry_funding_rate=_entry_funding_rate,
                    entry_pair_ema20_ema50_gap_pct=_entry_pair_ema20_ema50_gap_pct,
                    entry_dist_from_ema13_pct=_entry_dist_from_ema13_pct,
                    entry_btc_dist_from_ema13_pct=_entry_btc_dist_from_ema13_pct,
                    entry_btc_1h_slope=_current_btc_1h_slope,
                    # May 10: absolute pair 24h USD volume — sourced from binance scan
                    entry_pair_volume_24h_usd=volume_24h,
                    # Jun 12: eligible-universe volume rank at entry (50->75 read gate)
                    entry_pair_rank=_pair_rank,
                    # Jun 8: gap-expanding relaxation A/B tag — True if this entry was admitted
                    # by prev2_only but would have failed the strict prev1 check (MARGINAL cohort).
                    entry_gap_expand_marginal=gap_expand_marginal(indicators, signal),
                )

                if order:
                    logger.info(f"[DEBUG_OPENED] {pair} {signal} {confidence}: open_position returned order id={order.id}")
                    actions.append({
                        "pair": pair,
                        "action": f"OPENED_{signal}",
                        "confidence": confidence,
                        "price": indicators['price']
                    })
                    # Track newly-opened position for had_room state on subsequent
                    # filter checks within this same scan iteration.
                    _open_positions_in_scan += 1
                else:
                    logger.warning(f"[DEBUG_OPEN_FAILED] {pair} {signal} {confidence}: open_position returned None — check upstream logs in open_position for the real reason")

        self._last_scan_time = time.time()
        elapsed = self._last_scan_time - now
        logger.info(f"[SCAN] Completed in {elapsed:.1f}s - {len(top_pairs)} pairs processed, {len(actions)} positions opened")
        return actions
    
    async def update_pair_data(
        self,
        db: AsyncSession,
        pair: str,
        indicators: Dict,
        signal: str,
        confidence: Optional[str],
        volume_24h: Optional[float] = None,
        volume_ratio: Optional[float] = None
    ):
        """Update pair data cache in database.

        Commits per pair intentionally.  An earlier "optimization" batched
        the commit to once per scan cycle, but autoflush was still emitting
        UPDATEs on every subsequent SELECT inside the loop — which
        ACQUIRED the SQLite write lock and held it until the final commit.
        That made close_position unable to acquire the lock for 60+ seconds
        at a time.  Per-pair commits keep each write transaction short so
        the write lock is released between pairs, giving other writers
        (close_position, monitor_loop, realtime callbacks) windows to
        sneak in.
        """
        result = await db.execute(
            select(PairData).where(PairData.pair == pair)
        )
        pair_data = result.scalar_one_or_none()

        # Use provided 24h volume, or fall back to candle volume
        actual_volume_24h = volume_24h if volume_24h is not None else indicators.get('volume', 0)

        _th_cfg = config.trading_config.thresholds
        _flat_l = getattr(_th_cfg, 'macro_trend_flat_threshold_long', _th_cfg.macro_trend_flat_threshold)
        _flat_s = getattr(_th_cfg, 'macro_trend_flat_threshold_short', _th_cfg.macro_trend_flat_threshold)
        flat_th = min(_flat_l, _flat_s)
        regime = determine_macro_regime(
            indicators.get('ema20'), indicators.get('ema20_prev3'), flat_th
        )

        if pair_data:
            pair_data.price = indicators.get('price', 0)
            pair_data.ema5 = indicators.get('ema5')
            pair_data.ema5_prev3 = indicators.get('ema5_prev3')
            pair_data.ema8 = indicators.get('ema8')
            pair_data.ema13 = indicators.get('ema13')
            pair_data.ema20 = indicators.get('ema20')
            pair_data.ema20_prev3 = indicators.get('ema20_prev3')
            pair_data.rsi = indicators.get('rsi')
            pair_data.rsi_prev1 = indicators.get('rsi_prev1')
            pair_data.rsi_prev2 = indicators.get('rsi_prev2')
            pair_data.adx = indicators.get('adx')
            pair_data.volume_24h = actual_volume_24h
            pair_data.avg_volume = indicators.get('avg_volume')
            pair_data.signal = signal
            pair_data.confidence = confidence
            pair_data.macro_regime = regime
            pair_data.volume_ratio = volume_ratio
            pair_data.updated_at = datetime.utcnow()
        else:
            pair_data = PairData(
                pair=pair,
                price=indicators.get('price', 0),
                ema5=indicators.get('ema5'),
                ema5_prev3=indicators.get('ema5_prev3'),
                ema8=indicators.get('ema8'),
                ema13=indicators.get('ema13'),
                ema20=indicators.get('ema20'),
                ema20_prev3=indicators.get('ema20_prev3'),
                rsi=indicators.get('rsi'),
                rsi_prev1=indicators.get('rsi_prev1'),
                rsi_prev2=indicators.get('rsi_prev2'),
                adx=indicators.get('adx'),
                volume_24h=actual_volume_24h,
                avg_volume=indicators.get('avg_volume'),
                signal=signal,
                confidence=confidence,
                macro_regime=regime,
                volume_ratio=volume_ratio
            )
            db.add(pair_data)

        await db.commit()
    
    async def check_realtime_stop_loss(self, pair: str, current_price: float):
        """
        Real-time stop loss AND trailing stop check called by WebSocket on each price update.
        This provides instant protection instead of waiting for polling cycles.
        - Stop loss / break-even SL: triggers when P&L drops below threshold.
        - Trailing stop: triggers when price pulls back X% from high/low (only in post-TP zone).
        """
        global _open_orders_cache
        
        # CRITICAL: Never process invalid prices
        if current_price is None or current_price <= 0:
            return
        
        # Check cache first for fast lookup
        async with _cache_lock:
            cached_orders = _open_orders_cache.get(pair, [])
        
        if not cached_orders:
            return  # No open orders for this pair
        
        # Check each cached order
        for order_info in cached_orders:
            # Skip entirely if a close is already in progress for this order.
            # Prevents warning spam and duplicate close attempts when a close
            # has been initiated but the cache hasn't been refreshed yet
            # (e.g. DB commit failed, Binance fill succeeded but we haven't cleaned up).
            # The flag resets on the next update_orders_cache cycle.
            if order_info.get('_closing_in_progress'):
                continue
            # Jun 15 (operator request): flips now exit via the NORMAL realtime stack —
            # same SL (base −0.70 → ATR-widen ×1.5 → floor −1.20) and same ATR trailing +
            # min-profit gate as momentum trades. `_is_flip` gates ONLY the two exceptions:
            # EMA13 cross OFF (see :7963) and the short-specific runner-trail OFF (see :8755),
            # so a flip SHORT trails like a LONG. All other RT exits (fast-exit / tick /
            # rsi-momentum / signal-lost) are config-disabled, so flips get exactly SL +
            # trailing here; the monitor loop only enforces the 45min flip max-hold.
            _is_flip = (order_info.get('entry_strategy') or "").startswith("FLIP:")
            order_id = order_info['id']
            direction = order_info['direction']
            entry_price = order_info['entry_price']
            stop_loss_pct = order_info['stop_loss']
            quantity = order_info['quantity']
            entry_fee = order_info['entry_fee']
            cached_peak_pnl = order_info.get('peak_pnl', 0.0)
            cached_trough_pnl = order_info.get('trough_pnl', 0.0)
            be_l1_trigger = order_info.get('be_level1_trigger', 999)
            be_l1_offset = order_info.get('be_level1_offset', 0.0)
            be_l2_trigger = order_info.get('be_level2_trigger', 999)
            be_l2_offset = order_info.get('be_level2_offset', 0.0)
            be_l3_trigger = order_info.get('be_level3_trigger', 999)
            be_l3_offset = order_info.get('be_level3_offset', 0.0)
            be_l4_trigger = order_info.get('be_level4_trigger', 999)
            be_l4_offset = order_info.get('be_level4_offset', 0.0)
            be_l5_trigger = order_info.get('be_level5_trigger', 999)
            be_l5_offset = order_info.get('be_level5_offset', 0.0)
            pullback_trigger = order_info.get('pullback_trigger', 0.04)
            
            # Skip if entry data is invalid
            if entry_price <= 0 or quantity <= 0:
                logger.warning(f"[REALTIME_SL] {pair}: Invalid cache data - entry_price={entry_price}, quantity={quantity}")
                continue
            
            # Track high/low prices in real-time (updated on every tick)
            if direction == "LONG":
                cached_high = order_info.get('high_price', entry_price)
                if current_price > cached_high:
                    order_info['high_price'] = current_price
                high_price = order_info.get('high_price', entry_price)
            else:
                cached_low = order_info.get('low_price', entry_price)
                if current_price < cached_low:
                    order_info['low_price'] = current_price
                low_price = order_info.get('low_price', entry_price)
            
            # Calculate current P&L with fees
            entry_notional = entry_price * quantity
            current_notional = current_price * quantity
            exit_fee = current_notional * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
            total_fees = entry_fee + exit_fee
            
            if direction == "LONG":
                pnl = (current_price - entry_price) * quantity - total_fees
            else:
                pnl = (entry_price - current_price) * quantity - total_fees
            
            pnl_pct = (pnl / entry_notional) * 100

            # ════════════════════════════════════════════════════════════════
            # Pattern Fixed TP/SL (May 21, Pattern Cell Ship rules) — fires
            # BEFORE Fast Exit because TP at e.g. +0.10% needs to lock before
            # Fast Exit's higher threshold could engage. Only applies when the
            # trade was opened with a C-side pattern rule that set fixed_tp_pct
            # or fixed_sl_pct (stored in cache as 'pattern_fixed_tp_pct' /
            # 'pattern_fixed_sl_pct'). Trades without a pattern rule fall
            # through to standard exits.
            # ════════════════════════════════════════════════════════════════
            _ptn_tp = order_info.get('pattern_fixed_tp_pct')
            _ptn_sl = order_info.get('pattern_fixed_sl_pct')
            if (_ptn_tp is not None or _ptn_sl is not None) and not order_info.get('_closing_in_progress'):
                _ptn_close_reason = None
                if _ptn_tp is not None and pnl_pct >= float(_ptn_tp):
                    _ptn_close_reason = "PATTERN_FIXED_TP L1"
                    logger.warning(
                        f"[PATTERN_FIXED_TP] {pair} {direction}: pnl={pnl_pct:.4f}% >= rule_tp={_ptn_tp}% "
                        f"(source={order_info.get('pattern_cell_source')}) — CLOSING NOW!"
                    )
                elif _ptn_sl is not None and pnl_pct <= float(_ptn_sl):
                    _ptn_close_reason = "PATTERN_FIXED_SL L1"
                    logger.warning(
                        f"[PATTERN_FIXED_SL] {pair} {direction}: pnl={pnl_pct:.4f}% <= rule_sl={_ptn_sl}% "
                        f"(source={order_info.get('pattern_cell_source')}) — CLOSING NOW!"
                    )
                if _ptn_close_reason is not None:
                    order_info['_closing_in_progress'] = True
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                closed = await self.close_position(db, order, current_price, _ptn_close_reason)
                                if closed:
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                    except Exception as e:
                        logger.error(f"[PATTERN_FIXED_EXIT] Error closing {pair}: {e}")
                    continue  # Trade closed; skip remaining checks

            # ════════════════════════════════════════════════════════════════
            # ATR-LOW Fixed TP (Jun 5, 2026) — LONG "pop-and-fade" cohort lock.
            # When enabled, a LONG opened on a low-ATR pair (entry_atr_pct <
            # atr_low_fixed_tp_atr_max) exits the moment pnl_pct ≥ tp_pct. This is
            # a profit-LOCK only — it can never fire on a losing/DOA trade (those
            # ride to their stop). Low-ATR longs have no runners (batch 6-05 autopsy),
            # so we lock the pop and forgo the (non-existent) tail. Fires before
            # Fast Exit / trailing / EMA13. Close reason "ATR_FIXED_TP L1".
            # ════════════════════════════════════════════════════════════════
            _atr_tp_enabled = getattr(config.trading_config.thresholds, 'atr_low_fixed_tp_long_enabled', False)
            if (_atr_tp_enabled and direction == "LONG"
                    and not order_info.get('_closing_in_progress')):
                _atr_e = order_info.get('entry_atr_pct')
                _atr_tp_max = float(getattr(config.trading_config.thresholds, 'atr_low_fixed_tp_atr_max', 1.1))
                _atr_tp_pct = float(getattr(config.trading_config.thresholds, 'atr_low_fixed_tp_pct', 0.25))
                if _atr_e is not None and _atr_e < _atr_tp_max and pnl_pct >= _atr_tp_pct:
                    logger.warning(
                        f"[ATR_FIXED_TP] {pair} LONG: pnl={pnl_pct:.4f}% >= tp={_atr_tp_pct}% "
                        f"(entry_atr={_atr_e:.3f} < {_atr_tp_max}) — CLOSING NOW!"
                    )
                    order_info['_closing_in_progress'] = True
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                closed = await self.close_position(db, order, current_price, "ATR_FIXED_TP L1")
                                if closed:
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                    except Exception as e:
                        logger.error(f"[ATR_FIXED_TP] Error closing {pair}: {e}")
                    continue  # Trade closed; skip remaining checks

            # ════════════════════════════════════════════════════════════════
            # Fast Exit (May 15 PM, opt-in) — quick-profit lock for trades
            # that hit a threshold within a small window after entry. Fires
            # FIRST in the exit-check chain so it wins against EMA13_CROSS /
            # trailing / etc. Closes immediately as "FAST_EXIT L1".
            # Mirrors the Fast-Exit Counterfactual mechanic but fires LIVE on
            # first qualifying tick (vs. peak-time proxy in counterfactual).
            # ════════════════════════════════════════════════════════════════
            _fe_enabled = getattr(config.trading_config.thresholds, 'fast_exit_enabled', False)
            if _fe_enabled and not order_info.get('_closing_in_progress'):
                _fe_thr = getattr(config.trading_config.thresholds, 'fast_exit_threshold_pct', 0.20)
                _fe_window_min = getattr(config.trading_config.thresholds, 'fast_exit_window_minutes', 2)
                # May 25 — ATR-normalized FE L1 floor (mirror of trailing_atr_multiplier).
                # threshold = max(fast_exit_threshold_pct, entry_atr_pct × multiplier).
                # Prevents FE from firing on sub-noise moves on high-ATR pairs.
                # May 25 evening — added floor cap. On extreme-ATR pairs (e.g., XAN
                # at 1.6%), uncapped floor drove eff threshold to 0.84% — trade
                # peak never reached it, FE never fired, rode to SL. Cap bounds:
                # effective = min(cap, max(fixed, ATR × mult)).
                _fe_atr_mult = float(getattr(config.trading_config.thresholds, 'fast_exit_l1_atr_multiplier', 0.0) or 0.0)
                _fe_atr_pct = order_info.get('entry_atr_pct')
                if _fe_atr_mult > 0 and _fe_atr_pct is not None and _fe_atr_pct > 0:
                    _fe_atr_floor = _fe_atr_pct * _fe_atr_mult
                    _fe_atr_cap = float(getattr(config.trading_config.thresholds, 'fast_exit_l1_atr_floor_cap_pct', 0.0) or 0.0)
                    if _fe_atr_cap > 0 and _fe_atr_floor > _fe_atr_cap:
                        _fe_atr_floor = _fe_atr_cap
                    if _fe_atr_floor > _fe_thr:
                        _fe_thr = _fe_atr_floor
                _fe_opened_at = order_info.get('opened_at')
                if _fe_opened_at is not None and pnl_pct >= _fe_thr:
                    _fe_opened_naive = _fe_opened_at.replace(tzinfo=None) if _fe_opened_at.tzinfo is not None else _fe_opened_at
                    _fe_elapsed_min = (datetime.utcnow() - _fe_opened_naive).total_seconds() / 60.0
                    if _fe_elapsed_min <= _fe_window_min:
                        logger.warning(
                            f"[REALTIME_FAST_EXIT] {pair} {direction}: pnl={pnl_pct:.4f}% >= threshold={_fe_thr}%, "
                            f"elapsed={_fe_elapsed_min:.2f}min <= window={_fe_window_min}min - CLOSING NOW!"
                        )
                        order_info['_closing_in_progress'] = True
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    closed = await self.close_position(db, order, current_price, "FAST_EXIT L1")
                                    if closed:
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                        except Exception as e:
                            logger.error(f"[REALTIME_FAST_EXIT] Error closing {pair}: {e}")
                        continue  # Trade is closed; skip remaining checks for this order

            # ════════════════════════════════════════════════════════════════
            # Fast Exit L2 (May 19) — "slow climber" tier between L1 and trailing.
            # L1 catches fast bursts (peak ≥0.20% within 2min). Trailing arms at
            # peak ≥0.50%. L2 fills the gap: trades that build to 0.40% over
            # 2-5min then would die without ever hitting trailing's threshold.
            # Runs only if L1 did NOT fire (the `continue` above skips L2 if
            # L1 closed). Close reason: "FAST_EXIT L2".
            # ════════════════════════════════════════════════════════════════
            _fe2_enabled = getattr(config.trading_config.thresholds, 'fast_exit_l2_enabled', False)
            if _fe2_enabled and not order_info.get('_closing_in_progress'):
                _fe2_thr = getattr(config.trading_config.thresholds, 'fast_exit_l2_threshold_pct', 0.40)
                _fe2_window_min = getattr(config.trading_config.thresholds, 'fast_exit_l2_window_minutes', 5)
                # May 25 — ATR-normalized FE L2 floor (mirror of L1 + trailing_atr_multiplier).
                # Floor cap (May 25 evening): differentiated per tier — L2 cap is
                # higher than L1 cap to preserve slow-climber semantics.
                _fe2_atr_mult = float(getattr(config.trading_config.thresholds, 'fast_exit_l2_atr_multiplier', 0.0) or 0.0)
                _fe2_atr_pct = order_info.get('entry_atr_pct')
                if _fe2_atr_mult > 0 and _fe2_atr_pct is not None and _fe2_atr_pct > 0:
                    _fe2_atr_floor = _fe2_atr_pct * _fe2_atr_mult
                    _fe2_atr_cap = float(getattr(config.trading_config.thresholds, 'fast_exit_l2_atr_floor_cap_pct', 0.0) or 0.0)
                    if _fe2_atr_cap > 0 and _fe2_atr_floor > _fe2_atr_cap:
                        _fe2_atr_floor = _fe2_atr_cap
                    if _fe2_atr_floor > _fe2_thr:
                        _fe2_thr = _fe2_atr_floor
                _fe2_opened_at = order_info.get('opened_at')
                if _fe2_opened_at is not None and pnl_pct >= _fe2_thr:
                    _fe2_opened_naive = _fe2_opened_at.replace(tzinfo=None) if _fe2_opened_at.tzinfo is not None else _fe2_opened_at
                    _fe2_elapsed_min = (datetime.utcnow() - _fe2_opened_naive).total_seconds() / 60.0
                    if _fe2_elapsed_min <= _fe2_window_min:
                        logger.warning(
                            f"[REALTIME_FAST_EXIT_L2] {pair} {direction}: pnl={pnl_pct:.4f}% >= threshold={_fe2_thr}%, "
                            f"elapsed={_fe2_elapsed_min:.2f}min <= window={_fe2_window_min}min - CLOSING NOW!"
                        )
                        order_info['_closing_in_progress'] = True
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    closed = await self.close_position(db, order, current_price, "FAST_EXIT L2")
                                    if closed:
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                        except Exception as e:
                            logger.error(f"[REALTIME_FAST_EXIT_L2] Error closing {pair}: {e}")
                        continue  # Trade is closed; skip remaining checks for this order

            # ════════════════════════════════════════════════════════════════
            # Phase 1 shadow tracking (May 6) — counterfactual exit at first
            # price-vs-EMA cross against trade direction. Observation only:
            # records the moment + counterfactual close P&L if we had exited
            # at that point. Both NAIVE (first-tick cross) and CONFIRMED
            # (cross sustained ≥5min, filtering single-candle wicks) per
            # EMA13 and EMA20. Once recorded, never overwritten.
            # ════════════════════════════════════════════════════════════════
            _now_for_cross = datetime.utcnow()
            for _ema_label, _ema_val in (
                ('ema13', order_info.get('cached_ema13')),
                ('ema20', order_info.get('cached_ema20')),
            ):
                if _ema_val is None or _ema_val <= 0:
                    continue
                # "Wrong side" = price has reversed past the EMA against trade direction
                if direction == "LONG":
                    _is_wrong_side = current_price < _ema_val
                else:  # SHORT
                    _is_wrong_side = current_price > _ema_val
                _first_at_key = f'first_cross_{_ema_label}_at'
                _first_pnl_key = f'first_cross_{_ema_label}_pnl_pct'
                _conf_at_key = f'confirmed_cross_{_ema_label}_at'
                _conf_pnl_key = f'confirmed_cross_{_ema_label}_pnl_pct'
                _pending_key = f'pending_cross_{_ema_label}_started_at'
                if _is_wrong_side:
                    # NAIVE: record first-ever cross moment
                    if order_info.get(_first_at_key) is None:
                        order_info[_first_at_key] = _now_for_cross
                        order_info[_first_pnl_key] = round(pnl_pct, 4)
                    # CONFIRMED: track sustained cross (≥5min = ~1 candle persistence)
                    if order_info.get(_conf_at_key) is None:
                        _pending_at = order_info.get(_pending_key)
                        if _pending_at is None:
                            order_info[_pending_key] = _now_for_cross
                        else:
                            _elapsed_sec = (_now_for_cross - _pending_at).total_seconds()
                            if _elapsed_sec >= 300:  # 5min sustained = confirmed
                                order_info[_conf_at_key] = _now_for_cross
                                order_info[_conf_pnl_key] = round(pnl_pct, 4)
                                order_info[_pending_key] = None  # clear, one-shot done
                else:
                    # Price flipped back to right side before confirmation — whipsaw, reset pending
                    if order_info.get(_pending_key) is not None:
                        order_info[_pending_key] = None

            # ════════════════════════════════════════════════════════════════
            # EMA13 Cross Exit (May 6) — live exit when toggle is ON.
            # Fires on every tick where price is on wrong side of EMA13 (LONG:
            # price < EMA13, SHORT: price > EMA13). First-tick mode (no
            # confirmation window). Runs in PARALLEL to FL flags, RSI Handoff,
            # trailing stop — first-to-fire wins. No P&L filter (any state).
            # Cascade-close behavior: when toggle activates, any open trade
            # currently on wrong side of EMA13 closes on next tick.
            # ════════════════════════════════════════════════════════════════
            if not _is_flip and getattr(config.trading_config.thresholds, 'ema13_cross_exit_enabled', False):
                _ema13_for_exit = order_info.get('cached_ema13')
                if _ema13_for_exit is not None and _ema13_for_exit > 0:
                    if direction == "LONG":
                        _ema13_cross_fire = current_price < _ema13_for_exit
                    else:  # SHORT
                        _ema13_cross_fire = current_price > _ema13_for_exit
                    if _ema13_cross_fire and not order_info.get('_closing_in_progress'):
                        # May 8: optional AND-gate with EMA5/EMA8 stack flip.
                        # When ema13_cross_requires_stack_flip is True, EMA13 cross
                        # alone is not enough — also require the EMA5/EMA8 stack to
                        # have flipped against trade direction. Filters single-candle
                        # price wicks from firing the exit. Fail-closed on missing data.
                        # Jun 7: per-direction gate. When this side is disabled, the
                        # EMA13 cross records a PHANTOM (would-have-exited pnl) instead
                        # of closing — the trade rides to its real exit.
                        _e13c_th = config.trading_config.thresholds
                        _e13_dir_enabled = (getattr(_e13c_th, 'ema13_cross_exit_long_enabled', True)
                                            if direction == "LONG"
                                            else getattr(_e13c_th, 'ema13_cross_exit_short_enabled', True))
                        # Jun 12: SHORT runner stretch-trail handoff — once armed
                        # (peak >= arm, ATR gate if configured), the EMA13 cross must
                        # NOT close the trade (the measured shadow-strpk uplift comes
                        # from riding through the first cross). Records a phantom via
                        # the same path as a disabled direction; RUNNER_TRAIL/hard SL
                        # own the exit from here.
                        if _e13_dir_enabled and direction == "SHORT":
                            try:
                                if getattr(_e13c_th, 'runner_trail_short_enabled', False):
                                    _e13rt_amin = float(getattr(_e13c_th, 'runner_trail_short_atr_min', 0.0) or 0.0)
                                    _e13rt_arm = float(getattr(_e13c_th, 'runner_trail_short_arm_peak', 0.45) or 0.45)
                                    _e13rt_atr = order_info.get('entry_atr_pct')
                                    _e13rt_peak = order_info.get('peak_pnl', 0.0) or 0.0
                                    if (_e13rt_peak >= _e13rt_arm
                                            and (_e13rt_amin <= 0
                                                 or (_e13rt_atr is not None and _e13rt_atr >= _e13rt_amin))):
                                        _e13_dir_enabled = False  # phantom path below
                                        logger.info(f"[EMA13_RUNNER_SUPPRESS] {pair} SHORT: cross fired but runner armed (peak={_e13rt_peak:.2f}>= {_e13rt_arm}) — phantom + ride")
                            except Exception:
                                pass
                        _e13_strict = getattr(config.trading_config.thresholds, 'ema13_cross_requires_stack_flip', False)
                        _e13_stack_confirms = True  # default: not required
                        if _e13_strict:
                            _e13_es5 = order_info.get('cached_ema5')
                            _e13_es8 = order_info.get('cached_ema8')
                            if _e13_es5 is None or _e13_es8 is None or _e13_es5 <= 0 or _e13_es8 <= 0:
                                _e13_stack_confirms = False  # fail-closed
                            elif direction == "LONG":
                                _e13_stack_confirms = _e13_es5 < _e13_es8
                            else:
                                _e13_stack_confirms = _e13_es5 > _e13_es8
                            if not _e13_stack_confirms:
                                logger.info(
                                    f"[EMA13_CROSS_EXIT_HOLD] {pair} {direction}: price crossed EMA13 "
                                    f"({current_price:.6f} vs {_ema13_for_exit:.6f}) but stack intact "
                                    f"(ema5={_e13_es5}, ema8={_e13_es8}) — strict mode, holding"
                                )
                                # Capture pnl_pct at FIRST hold for tracking. Subsequent holds
                                # don't overwrite — we want the would-have-been-EMA13-exit P&L
                                # to compare against the eventual close.
                                if not order_info.get('_ema13_strict_held_recorded'):
                                    order_info['_ema13_strict_held_recorded'] = True
                                    try:
                                        async with AsyncSessionLocal() as _hdb:
                                            _h_result = await _hdb.execute(
                                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                            )
                                            _h_order = _h_result.scalar_one_or_none()
                                            if _h_order is not None and _h_order.ema13_strict_held_pnl_pct is None:
                                                _h_order.ema13_strict_held_pnl_pct = float(pnl_pct)
                                                await _hdb.commit()
                                                logger.info(
                                                    f"[EMA13_STRICT_FIRST_HOLD] {pair} order_id={order_id}: "
                                                    f"recorded held_pnl_pct={pnl_pct:.4f}%"
                                                )
                                    except Exception as _hexc:
                                        logger.warning(f"[EMA13_STRICT_FIRST_HOLD] Failed to persist for {pair}: {_hexc}")
                        if _e13_stack_confirms and not _e13_dir_enabled:
                            # PHANTOM: EMA13 cross is OFF for this direction — record the
                            # would-have-exited pnl at the FIRST fire (don't close).
                            if not order_info.get('_phantom_ema13_recorded'):
                                order_info['_phantom_ema13_recorded'] = True
                                logger.info(f"[PHANTOM_EMA13_CROSS] {pair} {direction}: EMA13 cross fired but disabled for {direction} — phantom pnl={pnl_pct:.4f}% (holding)")
                                try:
                                    async with AsyncSessionLocal() as _pdb:
                                        _p_result = await _pdb.execute(
                                            select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                        )
                                        _p_order = _p_result.scalar_one_or_none()
                                        if _p_order is not None and _p_order.phantom_ema13_cross_pnl is None:
                                            _p_order.phantom_ema13_cross_pnl = float(pnl_pct)
                                            _p_order.phantom_ema13_cross_at = datetime.utcnow()
                                            await _pdb.commit()
                                except Exception as _pexc:
                                    logger.warning(f"[PHANTOM_EMA13_CROSS] persist failed for {pair}: {_pexc}")
                            # fall through to other exit checks (no close)
                        elif _e13_stack_confirms and _e13_dir_enabled:
                            _tp_lvl_for_exit = order_info.get('current_tp_level', 1) or 1
                            _close_reason_e13 = f"EMA13_CROSS_EXIT L{_tp_lvl_for_exit}"
                            logger.warning(
                                f"[REALTIME_EMA13_CROSS_EXIT] {pair} {direction} L{_tp_lvl_for_exit}: "
                                f"price={current_price:.6f} {('<' if direction == 'LONG' else '>')}"
                                f" EMA13={_ema13_for_exit:.6f}, pnl={pnl_pct:.4f}% (peak={cached_peak_pnl:.4f}%) - CLOSING NOW!"
                            )
                            order_info['_closing_in_progress'] = True
                            try:
                                async with AsyncSessionLocal() as db:
                                    result = await db.execute(
                                        select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                    )
                                    order = result.scalar_one_or_none()
                                    if order:
                                        closed = await self.close_position(db, order, current_price, _close_reason_e13)
                                        if closed:
                                            async with _cache_lock:
                                                _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                            except Exception as e:
                                logger.error(f"[REALTIME_EMA13_CROSS_EXIT] Error closing {pair}: {e}")
                            continue  # Trade is closed; skip remaining checks for this order
                        # else: stack didn't confirm — fall through to other exit checks below

            # ════════════════════════════════════════════════════════════════
            # EMA Stack Cross Exit (May 6) — closes trade when EMA5 crosses EMA8
            # against trade direction past `ema_stack_cross_exit_level`.
            # LONG: ema5 < ema8 (bearish stack forming, entry signal inverted)
            # SHORT: ema5 > ema8 (bullish stack forming, entry signal inverted)
            # ARCHITECTURE: mirrors RSI Handoff (Option A — suppression active).
            # When current_tp_level >= level, this exit is the exclusive natural
            # exit and trailing pullback is suppressed (separate guard below in
            # the trailing block).  Cascade-close on activation: any open trade
            # currently with inverted EMA stack closes on next tick.
            # ════════════════════════════════════════════════════════════════
            _ema_stack_enabled = getattr(config.trading_config.thresholds, 'ema_stack_cross_exit_enabled', False)
            _ema_stack_level = getattr(config.trading_config.thresholds, 'ema_stack_cross_exit_level', 2)
            if _ema_stack_enabled and order_info.get('current_tp_level', 1) >= _ema_stack_level:
                _es_ema5 = order_info.get('cached_ema5')
                _es_ema8 = order_info.get('cached_ema8')
                if _es_ema5 is not None and _es_ema8 is not None and _es_ema5 > 0 and _es_ema8 > 0:
                    if direction == "LONG":
                        _stack_inverted = _es_ema5 < _es_ema8
                    else:  # SHORT
                        _stack_inverted = _es_ema5 > _es_ema8
                    if _stack_inverted and not order_info.get('_closing_in_progress'):
                        _tp_lvl_es = order_info.get('current_tp_level', 1) or 1
                        _close_reason_es = f"EMA_STACK_CROSS_EXIT L{_tp_lvl_es}"
                        logger.warning(
                            f"[REALTIME_EMA_STACK_CROSS_EXIT] {pair} {direction} L{_tp_lvl_es}: "
                            f"ema5={_es_ema5:.6f} {('<' if direction == 'LONG' else '>')} ema8={_es_ema8:.6f}, "
                            f"pnl={pnl_pct:.4f}% (peak={cached_peak_pnl:.4f}%) - CLOSING NOW!"
                        )
                        order_info['_closing_in_progress'] = True
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    closed = await self.close_position(db, order, current_price, _close_reason_es)
                                    if closed:
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                        except Exception as e:
                            logger.error(f"[REALTIME_EMA_STACK_CROSS_EXIT] Error closing {pair}: {e}")
                        continue  # Trade is closed; skip remaining checks for this order

            # Track peak P&L in real-time for break-even decisions
            current_peak = max(cached_peak_pnl, pnl_pct) if pnl_pct > 0 else cached_peak_pnl
            if pnl_pct > cached_peak_pnl and pnl_pct > 0:
                order_info['peak_reached_at'] = datetime.utcnow()
                _ema5_val = order_info.get('cached_ema5')
                if _ema5_val and _ema5_val > 0:
                    if direction == 'LONG':
                        order_info['peak_ema5_dist_pct'] = round((current_price - _ema5_val) / current_price * 100, 4)
                    else:
                        order_info['peak_ema5_dist_pct'] = round((_ema5_val - current_price) / current_price * 100, 4)
                    _ema5_prev3 = order_info.get('cached_ema5_prev3')
                    if _ema5_prev3 and _ema5_prev3 > 0:
                        order_info['peak_ema5_slope_pct'] = round((_ema5_val - _ema5_prev3) / _ema5_val * 100, 4)
            order_info['peak_pnl'] = current_peak

            # ===== LEASH SHADOW START — in-trade tick (observation-only) =====
            _ls_ema5 = order_info.get('cached_ema5')
            _ls_stretch = None
            if _ls_ema5 and _ls_ema5 > 0 and current_price > 0:
                _ls_stretch = ((current_price - _ls_ema5) / current_price * 100) if direction == 'LONG' \
                    else ((_ls_ema5 - current_price) / current_price * 100)
            _leash_update(order_info.get('id'), pnl_pct, peak_hint=current_peak,
                          stretch=_ls_stretch, entry_stretch=order_info.get('entry_ema5_stretch'))
            # ===== LEASH SHADOW END =====

            # May 17: post-arm-min tracking for BE-floor counterfactual analysis.
            # Once peak crosses BE trigger, start tracking the minimum P&L from
            # that moment onward (covers pre-global-peak dips AND post-peak retraces).
            _be_trigger_post_arm = order_info.get('be_level1_trigger', 0.20)
            if current_peak >= _be_trigger_post_arm:
                if not order_info.get('be_armed'):
                    order_info['be_armed'] = True
                    order_info['post_arm_min_pnl'] = pnl_pct
                    order_info['post_arm_min_at'] = datetime.utcnow()
                else:
                    _cur_min = order_info.get('post_arm_min_pnl')
                    if _cur_min is None or pnl_pct < _cur_min:
                        order_info['post_arm_min_pnl'] = pnl_pct
                        order_info['post_arm_min_at'] = datetime.utcnow()

            current_trough = min(cached_trough_pnl, pnl_pct) if pnl_pct < 0 else cached_trough_pnl
            if pnl_pct < cached_trough_pnl and pnl_pct < 0:
                order_info['trough_reached_at'] = datetime.utcnow()
                _ema5_val_t = order_info.get('cached_ema5')
                if _ema5_val_t and _ema5_val_t > 0:
                    if direction == 'LONG':
                        order_info['trough_ema5_dist_pct'] = round((current_price - _ema5_val_t) / current_price * 100, 4)
                    else:
                        order_info['trough_ema5_dist_pct'] = round((_ema5_val_t - current_price) / current_price * 100, 4)
            order_info['trough_pnl'] = current_trough

            # Track if EMA5 distance ever went unfavorable
            if not order_info.get('ema5_ever_negative'):
                _ema5_neg = order_info.get('cached_ema5')
                if _ema5_neg and _ema5_neg > 0:
                    if direction == 'LONG' and current_price < _ema5_neg:
                        order_info['ema5_ever_negative'] = True
                    elif direction == 'SHORT' and current_price > _ema5_neg:
                        order_info['ema5_ever_negative'] = True
            
            # Shadow BE tracking: record phantom triggers using original L1/L2 values
            _SHADOW_BE = [(1, 0.50, 0.20), (2, 1.00, 0.50)]
            for _sl, _strig, _soff in _SHADOW_BE:
                _tk = f'phantom_be_l{_sl}_triggered'
                _ek = f'phantom_be_l{_sl}_would_exit_pnl'
                _ak = f'phantom_be_l{_sl}_triggered_at'
                if not order_info.get(_tk) and current_peak >= _strig:
                    order_info[_tk] = True
                    order_info[_ak] = datetime.utcnow()
                if order_info.get(_tk) and order_info.get(_ek) is None and pnl_pct <= _soff:
                    order_info[_ek] = pnl_pct
            # May 14: Aggressive phantom BE @ 0.20/0.10 — observation-only counterfactual
            # for the BE design (May 19: floor raised from 0.05 to 0.10 per user request,
            # matches the live BE level1_offset under the new exit stack). Arms when
            # peak ≥ +0.20%, fires (records would_exit_pnl) when P&L retraces to ≤ +0.10%
            # after arming.
            # NOTE on mixed provenance: trades persisted before May 19 captured P&L at
            # the ≤+0.05% retrace point (lower than +0.10%). Those values still reflect
            # "BE would have fired" but at a deeper retrace than 0.10 would catch.
            # Going forward, captured values reflect the ≤+0.10% retrace point.
            if not order_info.get('phantom_be_aggr_triggered') and current_peak >= 0.20:
                order_info['phantom_be_aggr_triggered'] = True
                order_info['phantom_be_aggr_triggered_at'] = datetime.utcnow()
            if (order_info.get('phantom_be_aggr_triggered')
                    and order_info.get('phantom_be_aggr_would_exit_pnl') is None
                    and pnl_pct <= 0.10):
                order_info['phantom_be_aggr_would_exit_pnl'] = pnl_pct

            # Get TP target to determine if trailing stop would be active
            tp_level = order_info.get('current_tp_level', 1)
            conf = config.trading_config.confidence_levels.get(
                order_info.get('confidence', 'LOW'))
            tp_min = conf.tp_min if conf else 0.1
            # Jun 15 (operator request): FLIP per-level advance on PROFIT MILESTONES, decoupled
            # from `trend_continues` (which never fires for a fade — the pair's EMA stack is against
            # the short, so the momentum path pins flips at L1). Ratchet current_tp_level to
            # 1 + floor(peak / tp_min) (tp_min 0.45 → L2 @0.45%, L3 @0.90%, …), capped at 5, so the
            # per-level trailing widening applies to flips like normal trades (tight at L1, wider at
            # L2+ to let big reversals run). Ratchet-up only; persist to cache + DB so the UI shows
            # the level and the realtime trailing widening below uses it.
            if _is_flip and tp_min > 0 and current_peak > 0:
                _flip_lvl = min(5, 1 + int(current_peak / tp_min))
                if _flip_lvl > (tp_level or 1):
                    tp_level = _flip_lvl
                    order_info['current_tp_level'] = tp_level
                    try:
                        async with AsyncSessionLocal() as _lvl_db:
                            await _lvl_db.execute(update(Order).where(Order.id == order_id).values(current_tp_level=tp_level))
                            await _lvl_db.commit()
                    except Exception:
                        pass
            effective_tp_target = tp_level * tp_min if tp_level > 1 else tp_min
            
            # Trailing stop activates once peak reaches TP target or at L2+.
            # 0.005pp tolerance (May 6 — bug fix): floating-point rounding can leave
            # a peak at e.g. +0.4998% when tp_min is 0.50% — operationally identical
            # but strict >= would never arm trailing. Tolerance is well below any
            # configurable pullback_trigger so it doesn't affect intended behavior.
            # May 7 Phase 2: ALSO activate in the early-arm zone (peak between
            # trailing_early_arm_threshold and tp_min) to lock in moderate-momentum
            # gains that would otherwise reverse before reaching L1.
            try:
                _early_arm_thr_rt = float(getattr(config.trading_config.thresholds, 'trailing_early_arm_threshold', 0.0) or 0.0)
            except Exception:
                _early_arm_thr_rt = 0.0
            _in_early_arm_rt = (
                _early_arm_thr_rt > 0
                and current_peak >= _early_arm_thr_rt
                and current_peak < (tp_min - 0.005)
                and tp_level <= 1
            )
            trailing_stop_would_be_active = (
                current_peak >= (effective_tp_target - 0.005)
                or tp_level >= 2
                or _in_early_arm_rt
            )
            
            # Apply 3-level break-even logic (highest level wins)
            effective_sl = stop_loss_pct
            signal_still_active = order_info.get('signal_active', False)
            breakeven_active = False
            be_level = 0
            be_enabled = order_info.get('be_levels_enabled', True)

            if be_enabled and current_peak >= be_l5_trigger:
                breakeven_active = True
                be_level = 5
                effective_sl = be_l5_offset
            elif be_enabled and current_peak >= be_l4_trigger:
                breakeven_active = True
                be_level = 4
                effective_sl = be_l4_offset
            elif be_enabled and current_peak >= be_l3_trigger:
                breakeven_active = True
                be_level = 3
                effective_sl = be_l3_offset
            elif be_enabled and current_peak >= be_l2_trigger:
                breakeven_active = True
                be_level = 2
                effective_sl = be_l2_offset
            elif be_enabled and current_peak >= be_l1_trigger:
                breakeven_active = True
                be_level = 1
                effective_sl = be_l1_offset
            elif signal_still_active:
                effective_sl = order_info.get('signal_active_sl', stop_loss_pct)

            # May 22: ATR-adjusted SL widening for high-volatility pairs. Mirrors
            # trailing_atr_multiplier on the pullback side. Only WIDENS — if ATR-SL
            # is tighter than current effective_sl, keep current (no tightening).
            # Skipped when BE is active (BE floor overrides).
            if not breakeven_active:
                try:
                    _sl_atr_mult = float(getattr(config.trading_config.thresholds, 'sl_atr_multiplier', 0.0) or 0.0)
                except Exception:
                    _sl_atr_mult = 0.0
                _entry_atr_pct = order_info.get('entry_atr_pct')
                if _sl_atr_mult > 0 and _entry_atr_pct is not None and _entry_atr_pct > 0:
                    _atr_sl = -(_entry_atr_pct * _sl_atr_mult)
                    if _atr_sl < effective_sl:  # more negative = wider
                        effective_sl = _atr_sl
                # May 23: cap ATR widening at floor. Prevents extreme-ATR
                # pairs (e.g., ATR 2.3% → -3.47% SL) from effectively
                # disabling the SL. See CLAUDE.md May 23 entry.
                try:
                    _sl_floor = float(getattr(config.trading_config.thresholds, 'sl_atr_widen_floor_pct', 0.0) or 0.0)
                except Exception:
                    _sl_floor = 0.0
                if _sl_floor < 0 and effective_sl < _sl_floor:
                    effective_sl = _sl_floor

            # Check if stop loss triggered (epsilon 0.01% to avoid boundary precision issues)
            if pnl_pct <= effective_sl + 0.01:
                if breakeven_active:
                    close_reason = f"BREAKEVEN_EXIT_L{be_level}"
                elif signal_still_active:
                    close_reason = f"STOP_LOSS_WIDE L{tp_level}"
                else:
                    close_reason = f"STOP_LOSS L{tp_level}"

                _is_flagged_sl = order_info.get('signal_lost_flagged', False)

                # ─── FL1[WIDE_SL] interception: convert STOP_LOSS_WIDE into a flag instead of closing ───
                _fl1_wide_enabled_rt = getattr(config.trading_config.thresholds, 'fl1_for_wide_sl_enabled', True)
                if (close_reason.startswith("STOP_LOSS_WIDE")
                        and _fl1_wide_enabled_rt
                        and not _is_flagged_sl):
                    flag_time_rt = datetime.utcnow()
                    order_info['signal_lost_flagged'] = True
                    order_info['signal_lost_flag_pnl'] = round(pnl_pct, 4)
                    order_info['signal_lost_flagged_at'] = flag_time_rt
                    order_info['fl1_origin'] = "WIDE_SL"
                    logger.warning(f"[REALTIME_FL1_WIDE_SL] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% — flagged from STOP_LOSS_WIDE (origin=WIDE_SL)")
                    # Persist flag to DB
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                            )
                            order_db = result.scalar_one_or_none()
                            if order_db:
                                order_db.signal_lost_flagged = True
                                order_db.signal_lost_flag_pnl = round(pnl_pct, 4)
                                order_db.signal_lost_flagged_at = flag_time_rt
                                order_db.fl1_origin = "WIDE_SL"
                                await db.commit()
                    except Exception as e:
                        logger.error(f"[REALTIME_FL1_WIDE_SL] Error persisting flag for {pair}: {e}")
                    continue  # Don't close — let the trade run to backstop or recover

                # ─── FL1[WIDE_SL] emergency backstop: flagged WIDE_SL trade hit deep loss ───
                if _is_flagged_sl and order_info.get('fl1_origin') == "WIDE_SL" and not order_info.get('fl2_flagged'):
                    _fl1_backstop_rt = getattr(config.trading_config.thresholds, 'fl1_wide_sl_backstop', -1.2)
                    if pnl_pct <= _fl1_backstop_rt + 0.01:
                        close_reason = f"FL_EMERGENCY_SL L{tp_level}"
                        logger.warning(f"[REALTIME_FL_EMERGENCY_SL] {pair} {direction}: pnl={pnl_pct:.4f}% <= backstop={_fl1_backstop_rt}% (peak={current_peak:.4f}%) - CLOSING NOW!")
                        if order_info.get('_closing_in_progress'):
                            continue
                        order_info['_closing_in_progress'] = True
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    closed = await self.close_position(db, order, current_price, close_reason)
                                    if closed:
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                        except Exception as e:
                            logger.error(f"[REALTIME_FL_EMERGENCY_SL] Error closing {pair}: {e}")
                        continue
                    # WIDE_SL flagged but not at backstop yet — do NOT fire any normal SL close.
                    # The trade should only exit via backstop, trailing recovery, signal regain, or max hold time.
                    logger.debug(f"[REALTIME_FL1_WIDE_SL_HOLD] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% — holding to backstop={_fl1_backstop_rt}%, suppressing {close_reason}")
                    continue

                # ─── FL2 suppression: FL2-flagged trades only exit via recovery, deep_stop, trailing, or max hold ───
                if _is_flagged_sl and order_info.get('fl2_flagged'):
                    logger.debug(f"[REALTIME_FL2_HOLD] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% — suppressing {close_reason}, FL2 monitor handles recovery/deep_stop")
                    continue

                # Apply FL_ prefix if trade was flagged (signal lost at some point)
                if _is_flagged_sl and not close_reason.startswith("FL_"):
                    close_reason = f"FL_{close_reason}"

                logger.warning(f"[REALTIME_{close_reason}] {pair} {direction}: pnl={pnl_pct:.4f}% <= effective_sl={effective_sl}% (original_sl={stop_loss_pct}%, peak={current_peak:.4f}%) - CLOSING NOW!")

                # Prevent duplicate close attempts from consecutive monitor cycles
                if order_info.get('_closing_in_progress'):
                    continue
                order_info['_closing_in_progress'] = True

                # Close the order immediately using a new database session
                try:
                    async with AsyncSessionLocal() as db:
                        # Re-fetch the order to ensure it's still open
                        result = await db.execute(
                            select(Order).where(
                                and_(Order.id == order_id, Order.status == "OPEN")
                            )
                        )
                        order = result.scalar_one_or_none()
                        
                        if order:
                            closed = await self.close_position(
                                db, order, current_price, 
                                close_reason
                            )
                            if closed:
                                logger.info(f"[REALTIME_{close_reason}] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                async with _cache_lock:
                                    _open_orders_cache[pair] = [
                                        o for o in _open_orders_cache.get(pair, []) 
                                        if o['id'] != order_id
                                    ]
                            else:
                                logger.warning(f"[REALTIME_{close_reason}] {pair}: close_position returned None — will retry next cycle")
                except Exception as e:
                    logger.error(f"[REALTIME_SL] Error closing {pair}: {e}")
                continue  # Already handled, skip trailing stop check

            # Real-time Security Gap Exit: flagged trades within security gap range
            _is_flagged_rt = order_info.get('signal_lost_flagged', False)

            # ─── FL2 monitors: fire BEFORE the security gap check for already-FL2-flagged trades ───
            if _is_flagged_rt and order_info.get('fl2_flagged'):
                _fl2_recovery_rt = getattr(config.trading_config.thresholds, 'fl2_recovery_target', -0.4)
                _fl2_deep_rt = getattr(config.trading_config.thresholds, 'fl2_deep_stop', -1.0)
                tp_level = order_info.get('current_tp_level', 1)
                _fl2_close_reason = None
                if pnl_pct >= _fl2_recovery_rt:
                    _fl2_close_reason = f"FL_RECOVERED L{tp_level}"
                    logger.warning(f"[REALTIME_FL_RECOVERED] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% >= fl2_recovery={_fl2_recovery_rt}% - CLOSING NOW!")
                elif pnl_pct <= _fl2_deep_rt + 0.01:
                    _fl2_close_reason = f"FL_DEEP_STOP L{tp_level}"
                    logger.warning(f"[REALTIME_FL_DEEP_STOP] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% <= fl2_deep_stop={_fl2_deep_rt}% - CLOSING NOW!")
                if _fl2_close_reason:
                    if order_info.get('_closing_in_progress'):
                        continue
                    order_info['_closing_in_progress'] = True
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                closed = await self.close_position(db, order, current_price, _fl2_close_reason)
                                if closed:
                                    logger.info(f"[REALTIME_{_fl2_close_reason.split()[0]}] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                    except Exception as e:
                        logger.error(f"[REALTIME_FL2_MONITOR] Error closing {pair}: {e}")
                    continue

            # FL1[WIDE_SL] trades bypass the security gap entirely. See scan-loop comment.
            if (_is_flagged_rt
                    and not order_info.get('fl2_flagged')
                    and order_info.get('fl1_origin') != "WIDE_SL"):
                _sg_min = getattr(config.trading_config.thresholds, 'signal_lost_flag_security_min', -0.9)
                _sg_max = getattr(config.trading_config.thresholds, 'signal_lost_flag_security_max', -0.7)
                if pnl_pct >= _sg_min and pnl_pct <= _sg_max:
                    tp_level = order_info.get('current_tp_level', 1)
                    _fl2_enabled_rt = getattr(config.trading_config.thresholds, 'fl2_enabled', True)

                    # ─── FL2 promotion: flag the trade for recovery/deep_stop monitoring instead of closing ───
                    if _fl2_enabled_rt:
                        fl2_time_rt = datetime.utcnow()
                        order_info['fl2_flagged'] = True
                        order_info['fl2_flagged_at'] = fl2_time_rt
                        order_info['fl2_flag_pnl'] = round(pnl_pct, 4)
                        _fl2_recovery_target = getattr(config.trading_config.thresholds, 'fl2_recovery_target', -0.4)
                        _fl2_deep_stop = getattr(config.trading_config.thresholds, 'fl2_deep_stop', -1.0)
                        logger.warning(f"[REALTIME_FL2_FLAG] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% hit security gap — promoted to FL2 (origin={order_info.get('fl1_origin') or 'SIGNAL_LOST'}, recovery={_fl2_recovery_target}%, deep_stop={_fl2_deep_stop}%)")
                        # Persist FL2 flag to DB
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                order_db = result.scalar_one_or_none()
                                if order_db:
                                    order_db.fl2_flagged = True
                                    order_db.fl2_flagged_at = fl2_time_rt
                                    order_db.fl2_flag_pnl = round(pnl_pct, 4)
                                    await db.commit()
                        except Exception as e:
                            logger.error(f"[REALTIME_FL2_FLAG] Error persisting FL2 for {pair}: {e}")
                        continue

                    # FL2 disabled — original behavior: close here as FL_SIGNAL_LOST
                    logger.warning(f"[REALTIME_FL_SIGNAL_LOST] {pair} {direction} L{tp_level}: flagged trade hit security gap pnl={pnl_pct:.4f}% in [{_sg_min}, {_sg_max}] - CLOSING NOW!")

                    if order_info.get('_closing_in_progress'):
                        continue
                    order_info['_closing_in_progress'] = True

                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(
                                    and_(Order.id == order_id, Order.status == "OPEN")
                                )
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                fl_reason = f"FL_SIGNAL_LOST L{tp_level}"
                                closed = await self.close_position(db, order, current_price, fl_reason)
                                if closed:
                                    logger.info(f"[REALTIME_FL_SIGNAL_LOST] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [
                                            o for o in _open_orders_cache.get(pair, [])
                                            if o['id'] != order_id
                                        ]
                                else:
                                    logger.warning(f"[REALTIME_FL_SIGNAL_LOST] {pair}: close_position returned None — will retry next cycle")
                    except Exception as e:
                        logger.error(f"[REALTIME_FL_SIGNAL_LOST] Error closing {pair}: {e}")
                    continue

            # Real-time RSI Handoff Exit (May 6 — bug fix: was missing realtime path).
            # Mirrors the monitor-loop RSI Handoff at line ~3424 so it can fire sub-second
            # when the cached RSI sequence flips against direction past the handoff TP level.
            # Without this, RSI Handoff waited up to one full monitor cycle (~5min) longer
            # than RSI Momentum to fire — operationally inconsistent.
            rt_handoff_active = getattr(config.trading_config.thresholds, 'rsi_handoff_active', False)
            rt_handoff_level = getattr(config.trading_config.thresholds, 'rsi_handoff_level', 3)
            if rt_handoff_active and order_info.get('current_tp_level', 1) >= rt_handoff_level:
                _rt_h = order_info.get('rsi')
                _rt_h1 = order_info.get('rsi_prev1')
                _rt_h2 = order_info.get('rsi_prev2')
                if _rt_h is not None and _rt_h1 is not None and _rt_h2 is not None:
                    rt_handoff_fading = False
                    if direction == "LONG" and _rt_h < _rt_h1 < _rt_h2:
                        rt_handoff_fading = True
                    elif direction == "SHORT" and _rt_h > _rt_h1 > _rt_h2:
                        rt_handoff_fading = True
                    if rt_handoff_fading:
                        if order_info.get('_closing_in_progress'):
                            continue
                        order_info['_closing_in_progress'] = True
                        tp_level = order_info.get('current_tp_level', 1)
                        logger.warning(f"[REALTIME_RSI_HANDOFF_EXIT] {pair} {direction} L{tp_level}: RSI fading ({_rt_h2:.1f}->{_rt_h1:.1f}->{_rt_h:.1f}), pnl={pnl_pct:.4f}% (handoff_level={rt_handoff_level}) - CLOSING NOW!")
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    handoff_reason = f"RSI_HANDOFF_EXIT L{tp_level}"
                                    closed = await self.close_position(db, order, current_price, handoff_reason)
                                    if closed:
                                        logger.info(f"[REALTIME_RSI_HANDOFF_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                                    else:
                                        logger.warning(f"[REALTIME_RSI_HANDOFF_EXIT] {pair}: close_position returned None — will retry next cycle")
                        except Exception as e:
                            logger.error(f"[REALTIME_RSI_HANDOFF_EXIT] Error closing {pair}: {e}")
                        continue

            # Real-time RSI Momentum Exit: two consecutive RSI drops/rises within P&L range
            rt_rsi_exit_enabled = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_enabled', False)
            rt_rsi_exit_min = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_min_profit', 0.05)
            rt_rsi_exit_max = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_max_profit', 999.0)
            if rt_rsi_exit_enabled and pnl_pct > rt_rsi_exit_min and pnl_pct < rt_rsi_exit_max:
                _rt_rsi = order_info.get('rsi')
                _rt_rsi1 = order_info.get('rsi_prev1')
                _rt_rsi2 = order_info.get('rsi_prev2')
                if _rt_rsi is not None and _rt_rsi1 is not None and _rt_rsi2 is not None:
                    rt_rsi_fading = False
                    if direction == "LONG" and _rt_rsi < _rt_rsi1 < _rt_rsi2:
                        rt_rsi_fading = True
                    elif direction == "SHORT" and _rt_rsi > _rt_rsi1 > _rt_rsi2:
                        rt_rsi_fading = True
                    if rt_rsi_fading:
                        # Prevent duplicate close attempts from consecutive monitor cycles
                        if order_info.get('_closing_in_progress'):
                            continue
                        order_info['_closing_in_progress'] = True

                        tp_level = order_info.get('current_tp_level', 1)
                        logger.warning(f"[REALTIME_RSI_MOMENTUM_EXIT] {pair} {direction} L{tp_level}: RSI fading ({_rt_rsi2:.1f}->{_rt_rsi1:.1f}->{_rt_rsi:.1f}), pnl={pnl_pct:.4f}% (range {rt_rsi_exit_min}% to {rt_rsi_exit_max}%) - CLOSING NOW!")
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(
                                        and_(Order.id == order_id, Order.status == "OPEN")
                                    )
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    rsi_reason = f"RSI_MOMENTUM_EXIT L{tp_level}"
                                    closed = await self.close_position(
                                        db, order, current_price, rsi_reason
                                    )
                                    if closed:
                                        logger.info(f"[REALTIME_RSI_MOMENTUM_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [
                                                o for o in _open_orders_cache.get(pair, [])
                                                if o['id'] != order_id
                                            ]
                                    else:
                                        logger.warning(f"[REALTIME_RSI_MOMENTUM_EXIT] {pair}: close_position returned None — will retry next cycle")
                        except Exception as e:
                            logger.error(f"[REALTIME_RSI_MOMENTUM_EXIT] Error closing {pair}: {e}")
                        continue

            # Real-time Tick Momentum Exit: multi-window price velocity check
            tick_exit_enabled = getattr(config.trading_config.thresholds, 'tick_momentum_exit_enabled', False)
            _is_trade_flagged = order_info.get('signal_lost_flagged', False)
            tick_exit_min_profit = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_profit', 0.05)
            if _is_trade_flagged:
                tick_exit_min_profit = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_profit_flagged', -0.10)
            now = time.time()
            tick_buf = order_info.get('tick_prices', [])
            tick_buf.append((now, current_price))
            cutoff = now - 125
            tick_buf[:] = [(t, p) for t, p in tick_buf if t >= cutoff]
            order_info['tick_prices'] = tick_buf

            if tick_exit_enabled and pnl_pct > tick_exit_min_profit:
                tick_min_delta_fallback = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_delta', 0.05)
                deltas_str = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_deltas', '')
                windows_str = getattr(config.trading_config.thresholds, 'tick_momentum_exit_windows', '15,30,60')
                try:
                    windows = [int(w.strip()) for w in windows_str.split(',') if w.strip()]
                except (ValueError, AttributeError):
                    windows = [15, 30, 60]

                per_window_deltas = None
                if deltas_str and deltas_str.strip():
                    try:
                        parsed = [float(d.strip()) for d in deltas_str.split(',') if d.strip()]
                        if len(parsed) == len(windows):
                            per_window_deltas = parsed
                    except (ValueError, AttributeError):
                        pass
                if per_window_deltas is None:
                    per_window_deltas = [tick_min_delta_fallback] * len(windows)

                all_windows_confirm = _check_tick_momentum_fade(tick_buf, now, windows, per_window_deltas, direction)

                # Shadow tick momentum: check phantom configs
                for _lbl, _swin, _sdelta in _SHADOW_TICK_CONFIGS:
                    _tk = f'phantom_tick_{_lbl}_triggered'
                    if not order_info.get(_tk):
                        _sdeltas = _sdelta if isinstance(_sdelta, list) else [_sdelta] * len(_swin)
                        if _check_tick_momentum_fade(tick_buf, now, _swin, _sdeltas, direction):
                            order_info[_tk] = True
                            order_info[f'phantom_tick_{_lbl}_triggered_at'] = datetime.utcnow()
                            order_info[f'phantom_tick_{_lbl}_pnl'] = pnl_pct

                if all_windows_confirm:
                    # Prevent duplicate close attempts from consecutive monitor cycles
                    if order_info.get('_closing_in_progress'):
                        continue
                    order_info['_closing_in_progress'] = True

                    tp_level = order_info.get('current_tp_level', 1)
                    deltas_info = '/'.join(f"{d:.2f}" for d in per_window_deltas)
                    logger.warning(f"[REALTIME_TICK_MOMENTUM_EXIT] {pair} {direction} L{tp_level}: tick momentum fading across {windows}s windows (deltas={deltas_info}%), pnl={pnl_pct:.4f}% > min={tick_exit_min_profit}% - CLOSING NOW!")
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(
                                    and_(Order.id == order_id, Order.status == "OPEN")
                                )
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                tick_reason = f"TICK_MOMENTUM_EXIT L{tp_level}"
                                closed = await self.close_position(
                                    db, order, current_price, tick_reason
                                )
                                if closed:
                                    logger.info(f"[REALTIME_TICK_MOMENTUM_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [
                                            o for o in _open_orders_cache.get(pair, [])
                                            if o['id'] != order_id
                                        ]
                                else:
                                    logger.warning(f"[REALTIME_TICK_MOMENTUM_EXIT] {pair}: close_position returned None — will retry next cycle")
                    except Exception as e:
                        logger.error(f"[REALTIME_TICK_MOMENTUM_EXIT] Error closing {pair}: {e}")
                    continue

            # Real-time P&L trailing: only MOMENTUM_EXIT (signal lost). Skipped when signal active + RSI exit enabled.
            pnl_trigger = getattr(config.trading_config.thresholds, 'pnl_trailing_trigger', 0.0)
            pnl_ratio = getattr(config.trading_config.thresholds, 'pnl_trailing_ratio', 0.0)
            if pnl_trigger > 0 and pnl_ratio > 0 and cached_peak_pnl >= pnl_trigger:
                rt_signal_active = order_info.get('signal_active', False)
                if rt_signal_active and rt_rsi_exit_enabled:
                    pass  # RSI momentum exit handles signal-active exits
                else:
                    pnl_exit_level = cached_peak_pnl * pnl_ratio
                    if pnl_pct <= pnl_exit_level:
                        tp_level = order_info.get('current_tp_level', 1)
                        logger.warning(f"[REALTIME_MOMENTUM_EXIT] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% <= peak={cached_peak_pnl:.4f}%*{pnl_ratio}(no-signal)={pnl_exit_level:.4f}%, price={current_price:.6f} - CLOSING NOW!")
                        try:
                            async with AsyncSessionLocal() as db:
                                result = await db.execute(
                                    select(Order).where(
                                        and_(Order.id == order_id, Order.status == "OPEN")
                                    )
                                )
                                order = result.scalar_one_or_none()
                                if order:
                                    mom_reason = f"MOMENTUM_EXIT L{tp_level}"
                                    closed = await self.close_position(
                                        db, order, current_price, mom_reason
                                    )
                                    if closed:
                                        logger.info(f"[REALTIME_MOMENTUM_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%, peak was {cached_peak_pnl:.4f}%")
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [
                                                o for o in _open_orders_cache.get(pair, [])
                                                if o['id'] != order_id
                                            ]
                                    else:
                                        logger.warning(f"[REALTIME_MOMENTUM_EXIT] {pair}: close_position returned None — will retry next cycle")
                        except Exception as e:
                            logger.error(f"[REALTIME_MOMENTUM_EXIT] Error closing {pair}: {e}")
                        continue
            
            # Real-time trailing stop check (only when trailing stop is active and TP/trailing enabled).
            # Phase 1d-ExitTest (May 2): suppress trailing when RSI Handoff is active and trade is past level.
            # May 6: also suppress when EMA Stack Cross Exit is active and trade is past level.
            # The respective handoff exit fires through its own handler — this guard prevents trailing from racing it.
            _handoff_suppress = False
            try:
                if getattr(config.trading_config.thresholds, 'rsi_handoff_active', False):
                    _hl = getattr(config.trading_config.thresholds, 'rsi_handoff_level', 3)
                    if order_info.get('current_tp_level', 1) >= _hl:
                        _handoff_suppress = True
                if not _handoff_suppress and getattr(config.trading_config.thresholds, 'ema_stack_cross_exit_enabled', False):
                    _esl = getattr(config.trading_config.thresholds, 'ema_stack_cross_exit_level', 2)
                    if order_info.get('current_tp_level', 1) >= _esl:
                        _handoff_suppress = True
                # Jun 1: runner stretch-trail — suppress the realtime tight-trailing
                # for runner-armed high-ATR LONGs so the trade rides; the actual
                # RUNNER_TRAIL exit fires from check_exit_conditions in the monitor
                # loop. Backstops (hard SL / EMA13) below are NOT suppressed.
                if not _handoff_suppress:
                    _rt_th = config.trading_config.thresholds
                    if direction == "LONG":
                        _rt_en = getattr(_rt_th, 'runner_trail_enabled', False)
                        _rt_amin = float(getattr(_rt_th, 'runner_trail_atr_min', 1.0) or 0.0)
                        _rt_arm = float(getattr(_rt_th, 'runner_trail_arm_peak', 0.70) or 0.70)
                    else:  # Jun 12: SHORT runner trail (no ATR gate, arm 0.45)
                        _rt_en = getattr(_rt_th, 'runner_trail_short_enabled', False)
                        _rt_amin = float(getattr(_rt_th, 'runner_trail_short_atr_min', 0.0) or 0.0)
                        _rt_arm = float(getattr(_rt_th, 'runner_trail_short_arm_peak', 0.45) or 0.45)
                    _rt_atr = order_info.get('entry_atr_pct')
                    _rt_peak = order_info.get('peak_pnl', 0.0) or 0.0
                    # Jun 14: runner-trail disabled for flips by default. Jun 16: EXCEPT FAN flips
                    # with flip_fan_runner_strpk — they get the SHORT runner stretch-trail (the
                    # actual RUNNER_TRAIL exit fires in the monitor loop where ema5 is fresh); here
                    # we suppress the realtime tight-trail once armed so it can't close first.
                    # Jun 16: suppress the realtime tight-trail for ANY armed strpk flip short
                    # (FAN via flip_fan_runner_strpk, others via flip_runner_strpk_shorts) so the
                    # monitor's RUNNER_TRAIL handles the exit.
                    _strpk_src = (order_info.get('entry_strategy') or "")[5:]
                    _flip_strpk_ok = (_is_flip and direction == "SHORT" and (
                        (_strpk_src == "FAN_RATIO_GATE" and getattr(_rt_th, 'flip_fan_runner_strpk', False))
                        or (_strpk_src != "FAN_RATIO_GATE" and getattr(_rt_th, 'flip_runner_strpk_shorts', False))))
                    if (_rt_en and (not _is_flip or _flip_strpk_ok) and _rt_peak >= _rt_arm
                            and (_rt_amin <= 0 or (_rt_atr is not None and _rt_atr >= _rt_amin))):
                        _handoff_suppress = True
            except Exception:
                pass

            if trailing_stop_would_be_active and order_info.get('tp_trailing_enabled', False) and not _handoff_suppress:
                should_close_trailing = False
                tp_level = order_info.get('current_tp_level', 1)

                # May 7 — apply BOTH widening (realtime mirror) AND ATR floor + early-arm.
                _th = config.trading_config.thresholds
                try:
                    _widening = float(getattr(_th, 'pullback_widening_per_level', 0.0) or 0.0)
                except Exception:
                    _widening = 0.0
                # May 7 Phase 2: detect early-arm zone using cached tp_min and current peak.
                _entry_atr = order_info.get('entry_atr_pct')
                _tp_min = order_info.get('tp_min', 0.50)
                _cur_peak = order_info.get('peak_pnl', 0.0) or 0.0
                try:
                    _early_arm_thr = float(getattr(_th, 'trailing_early_arm_threshold', 0.0) or 0.0)
                    _early_arm_pb = float(getattr(_th, 'trailing_early_arm_pullback', 0.10) or 0.10)
                except Exception:
                    _early_arm_thr = 0.0
                    _early_arm_pb = 0.10
                _in_early_arm = (
                    _early_arm_thr > 0
                    and _cur_peak >= _early_arm_thr
                    and _cur_peak < (_tp_min - 0.005)
                    and tp_level <= 1
                )
                if _in_early_arm:
                    _effective_pullback = _early_arm_pb
                else:
                    _effective_pullback = pullback_trigger + _widening * max(0, tp_level - 1)
                # May 7 Phase 1: ATR floor
                try:
                    _atr_mult = float(getattr(_th, 'trailing_atr_multiplier', 0.0) or 0.0)
                except Exception:
                    _atr_mult = 0.0
                if _atr_mult > 0 and _entry_atr is not None and _entry_atr > 0:
                    _atr_floor = _entry_atr * _atr_mult
                    if _atr_floor > _effective_pullback:
                        _effective_pullback = _atr_floor

                # Determine if pullback threshold is currently crossed
                _pullback_threshold_crossed = False
                if direction == "LONG" and high_price and high_price > 0:
                    price_drop_pct = ((high_price - current_price) / high_price) * 100
                    _pullback_threshold_crossed = price_drop_pct >= _effective_pullback
                elif direction == "SHORT" and low_price and low_price > 0:
                    price_rise_pct = ((current_price - low_price) / low_price) * 100
                    _pullback_threshold_crossed = price_rise_pct >= _effective_pullback

                # May 9: Confirmation timer. Catch single-tick noise wicks (e.g.
                # SAHARAUSDT 1.34s wick on 1.87% ATR pair) by requiring sustained
                # pullback for N seconds. 0 = disabled (pre-May-9 immediate fire).
                try:
                    _confirm_secs = int(getattr(_th, 'trailing_pullback_confirmation_seconds', 15) or 0)
                except (ValueError, TypeError):
                    _confirm_secs = 15
                _now = datetime.utcnow()

                if _pullback_threshold_crossed:
                    if order_info.get('_trailing_pullback_first_at') is None:
                        # First moment threshold crossed — start timer, record counterfactual P&L
                        order_info['_trailing_pullback_first_at'] = _now
                        order_info['_trailing_pullback_first_pnl_pct'] = float(pnl_pct)
                        # Persist counterfactual to DB (one-time record per trade)
                        try:
                            async with AsyncSessionLocal() as _tp_db:
                                await _tp_db.execute(
                                    update(Order).where(Order.id == order_id).values(
                                        trailing_first_pullback_pnl_pct=float(pnl_pct)
                                    )
                                )
                                await _tp_db.commit()
                        except Exception:
                            pass
                        if _confirm_secs > 0:
                            logger.info(f"[TRAILING_CONFIRM] {pair} {direction} L{tp_level}: pullback threshold crossed at pnl={pnl_pct:.4f}% — confirmation timer started ({_confirm_secs}s)")
                            should_close_trailing = False
                        else:
                            should_close_trailing = True
                            logger.warning(f"[REALTIME_TRAILING] {pair} {direction} L{tp_level}: confirmation disabled — CLOSING NOW! pnl={pnl_pct:.4f}%")
                    else:
                        # Timer already running, check elapsed
                        _elapsed = (_now - order_info['_trailing_pullback_first_at']).total_seconds()
                        if _elapsed >= _confirm_secs:
                            should_close_trailing = True
                            logger.warning(f"[REALTIME_TRAILING] {pair} {direction} L{tp_level}: pullback CONFIRMED after {_elapsed:.1f}s — CLOSING NOW! pnl={pnl_pct:.4f}% (vs first_pullback={order_info.get('_trailing_pullback_first_pnl_pct'):.4f}%)")
                            # Persist confirmed_at
                            try:
                                async with AsyncSessionLocal() as _tp_db2:
                                    await _tp_db2.execute(
                                        update(Order).where(Order.id == order_id).values(
                                            trailing_confirmed_at=_now
                                        )
                                    )
                                    await _tp_db2.commit()
                            except Exception:
                                pass
                        # else: still waiting for confirmation, no close
                else:
                    # Pullback condition NOT met — if timer was running, reset it
                    if order_info.get('_trailing_pullback_first_at') is not None:
                        _resets = order_info.get('_trailing_pullback_resets', 0) + 1
                        order_info['_trailing_pullback_resets'] = _resets
                        order_info['_trailing_pullback_first_at'] = None
                        logger.info(f"[TRAILING_CONFIRM] {pair} {direction} L{tp_level}: price recovered — timer reset (#{_resets} for this trade)")
                        # Persist reset count
                        try:
                            async with AsyncSessionLocal() as _tp_db3:
                                await _tp_db3.execute(
                                    update(Order).where(Order.id == order_id).values(
                                        trailing_pullback_resets=_resets
                                    )
                                )
                                await _tp_db3.commit()
                        except Exception:
                            pass
                
                # Jun 15: trailing MIN-PROFIT GATE for flips only (operator-requested "never
                # trail into a loss"). The realtime trailing path has no such gate — the momentum
                # min-profit gate lives in the monitor's check_exit_conditions, which flips skip.
                # Suppress a flip trailing-exit that would close below trailing_min_profit_to_fire;
                # the ATR-widened SL still bounds the downside. Momentum trades are untouched.
                if should_close_trailing and _is_flip:
                    try:
                        _flip_trail_min = float(getattr(_th, 'trailing_min_profit_to_fire', 0.0) or 0.0)
                    except (ValueError, TypeError):
                        _flip_trail_min = 0.0
                    if pnl_pct < _flip_trail_min:
                        should_close_trailing = False

                if should_close_trailing:
                    # Prevent duplicate close attempts from consecutive monitor cycles
                    if order_info.get('_closing_in_progress'):
                        continue
                    order_info['_closing_in_progress'] = True

                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(
                                    and_(Order.id == order_id, Order.status == "OPEN")
                                )
                            )
                            order = result.scalar_one_or_none()

                            if order:
                                trail_reason = f"TRAILING_STOP L{order.current_tp_level}"
                                # Apply FL_ prefix if trade was flagged
                                if order_info.get('signal_lost_flagged') and not trail_reason.startswith("FL_"):
                                    trail_reason = f"FL_{trail_reason}"
                                closed = await self.close_position(
                                    db, order, current_price, trail_reason
                                )
                                if closed:
                                    logger.info(f"[REALTIME_TRAILING] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [
                                            o for o in _open_orders_cache.get(pair, [])
                                            if o['id'] != order_id
                                        ]
                                else:
                                    logger.warning(f"[REALTIME_TRAILING] {pair}: close_position returned None — will retry next cycle")
                    except Exception as e:
                        logger.error(f"[REALTIME_TRAILING] Error closing {pair}: {e}")
    
    async def update_orders_cache(self, db: AsyncSession):
        """Update the open orders cache for real-time stop loss checking.
        Includes peak_pnl and breakeven config for break-even SL logic."""
        global _open_orders_cache
        
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        orders = result.scalars().all()
        
        # Fetch current EMA values for each pair with open orders
        pair_names = list({o.pair for o in orders})
        pair_emas: Dict[str, Dict] = {}
        pair_ema5s: Dict[str, float] = {}
        if pair_names:
            sig_result = await db.execute(
                select(PairData.pair, PairData.ema5, PairData.ema8, PairData.ema13,
                       PairData.ema20, PairData.price,
                       PairData.rsi, PairData.rsi_prev1, PairData.rsi_prev2,
                       PairData.ema5_prev3).where(PairData.pair.in_(pair_names))
            )
            for row in sig_result:
                pair_emas[row.pair] = {
                    'ema5': row.ema5, 'ema8': row.ema8,
                    'ema13': row.ema13,
                    'ema20': row.ema20, 'price': row.price,
                    'rsi': row.rsi, 'rsi_prev1': row.rsi_prev1, 'rsi_prev2': row.rsi_prev2,
                    'ema5_prev3': row.ema5_prev3,
                }
                if row.ema5 is not None:
                    pair_ema5s[row.pair] = row.ema5
        
        # Build new cache
        new_cache: Dict[str, List[Dict]] = {}
        for order in orders:
            # Get config for this order's confidence level
            conf_config = config.trading_config.confidence_levels.get(order.confidence)
            if not conf_config:
                continue
            
            order_info = {
                'id': order.id,
                'direction': order.direction,
                'entry_strategy': (order.entry_strategy or "MOMENTUM"),  # Jun 15: flips now exit via the realtime stack (entry_strategy gates _is_flip)
                'entry_price': order.entry_price,
                'quantity': order.quantity,
                'entry_fee': order.entry_fee,
                'confidence': order.confidence,
                # May 15 PM: required by FAST_EXIT (Fast Exit) realtime check —
                # computes elapsed-minutes from open against fast_exit_window_minutes.
                'opened_at': order.opened_at,
                'stop_loss': conf_config.stop_loss,
                'signal_active_sl': conf_config.signal_active_sl,
                'signal_active': is_signal_direction_active(
                    order.direction,
                    pair_emas.get(order.pair, {}).get('ema5'),
                    pair_emas.get(order.pair, {}).get('ema8'),
                    pair_emas.get(order.pair, {}).get('ema20'),
                    pair_emas.get(order.pair, {}).get('price')
                ),
                'current_tp_level': order.current_tp_level,
                'peak_pnl': order.peak_pnl or 0.0,
                'trough_pnl': order.trough_pnl or 0.0,
                # May 17: post-arm-min tracking (resumed if already populated)
                'be_armed': order.post_arm_min_pnl_pct is not None,
                'post_arm_min_pnl': order.post_arm_min_pnl_pct,
                'post_arm_min_at': order.post_arm_min_pnl_at,
                'be_levels_enabled': getattr(conf_config, 'be_levels_enabled', True),
                'be_level1_trigger': conf_config.be_level1_trigger,
                'be_level1_offset': conf_config.be_level1_offset,
                'be_level2_trigger': conf_config.be_level2_trigger,
                'be_level2_offset': conf_config.be_level2_offset,
                'be_level3_trigger': conf_config.be_level3_trigger,
                'be_level3_offset': conf_config.be_level3_offset,
                'be_level4_trigger': conf_config.be_level4_trigger,
                'be_level4_offset': conf_config.be_level4_offset,
                'be_level5_trigger': conf_config.be_level5_trigger,
                'be_level5_offset': conf_config.be_level5_offset,
                'high_price': order.high_price_since_entry or order.entry_price,
                'low_price': order.low_price_since_entry or order.entry_price,
                'pullback_trigger': conf_config.pullback_trigger,
                'tp_trailing_enabled': conf_config.tp_trailing_enabled,
                'entry_atr_pct': getattr(order, 'entry_atr_pct', None),  # May 7 Phase 1: ATR-normalized trailing
                'tp_min': conf_config.tp_min,                            # May 7 Phase 2: early-arm zone check
                'cached_ema5': pair_ema5s.get(order.pair),
                'cached_ema5_prev3': pair_emas.get(order.pair, {}).get('ema5_prev3'),
                'cached_ema8': pair_emas.get(order.pair, {}).get('ema8'),
                'cached_ema13': pair_emas.get(order.pair, {}).get('ema13'),
                'cached_ema20': pair_emas.get(order.pair, {}).get('ema20'),
                # Phase 1 shadow tracking (May 6) — counterfactual exit at price-vs-EMA cross.
                # Restored from DB so a bot restart preserves prior cross records.
                'first_cross_ema13_at': order.first_cross_ema13_at,
                'first_cross_ema13_pnl_pct': order.first_cross_ema13_pnl_pct,
                'confirmed_cross_ema13_at': order.confirmed_cross_ema13_at,
                'confirmed_cross_ema13_pnl_pct': order.confirmed_cross_ema13_pnl_pct,
                'first_cross_ema20_at': order.first_cross_ema20_at,
                'first_cross_ema20_pnl_pct': order.first_cross_ema20_pnl_pct,
                'confirmed_cross_ema20_at': order.confirmed_cross_ema20_at,
                'confirmed_cross_ema20_pnl_pct': order.confirmed_cross_ema20_pnl_pct,
                'pending_cross_ema13_started_at': None,
                'pending_cross_ema20_started_at': None,
                'peak_ema5_gap': order.peak_ema5_gap or 0.0,
                'peak_ema5_dist_pct': order.peak_ema5_dist_pct,
                'peak_ema5_slope_pct': order.peak_ema5_slope_pct,
                'peak_reached_at': order.peak_reached_at,
                'trough_reached_at': order.trough_reached_at,
                'trough_ema5_dist_pct': order.trough_ema5_dist_pct,
                'ema5_ever_negative': order.ema5_went_negative in ("RECOVERED", "ENDED_NEG") if order.ema5_went_negative else False,
                'signal_lost_flagged': bool(order.signal_lost_flagged) if order.signal_lost_flagged else False,
                'signal_lost_flag_pnl': order.signal_lost_flag_pnl,
                'signal_lost_flagged_at': order.signal_lost_flagged_at,
                'fl1_origin': order.fl1_origin,
                'fl2_flagged': bool(order.fl2_flagged) if order.fl2_flagged else False,
                'fl2_flagged_at': order.fl2_flagged_at,
                'fl2_flag_pnl': order.fl2_flag_pnl,
                'rsi': pair_emas.get(order.pair, {}).get('rsi'),
                'rsi_prev1': pair_emas.get(order.pair, {}).get('rsi_prev1'),
                'rsi_prev2': pair_emas.get(order.pair, {}).get('rsi_prev2'),
                'tick_prices': [],
                'phantom_be_l1_triggered': order.phantom_be_l1_triggered_at is not None,
                'phantom_be_l1_triggered_at': order.phantom_be_l1_triggered_at,
                'phantom_be_l1_would_exit_pnl': order.phantom_be_l1_would_exit_pnl,
                'phantom_be_l2_triggered': order.phantom_be_l2_triggered_at is not None,
                'phantom_be_l2_triggered_at': order.phantom_be_l2_triggered_at,
                'phantom_be_l2_would_exit_pnl': order.phantom_be_l2_would_exit_pnl,
                # May 14 — aggressive phantom BE @ 0.20/0.10 (observation-only)
                'phantom_be_aggr_triggered': order.phantom_be_aggr_triggered_at is not None,
                'phantom_be_aggr_triggered_at': order.phantom_be_aggr_triggered_at,
                'phantom_be_aggr_would_exit_pnl': order.phantom_be_aggr_would_exit_pnl,
                # May 11 — phantom regime change exit (observation-only).
                # May 20 fix: bootstrap from persisted Order columns so bot restart preserves
                # any prior capture. Was missing from update_orders_cache: result = 1/278 trades
                # had the data (0.4%), should have been ~3-5% based on regime_opposite_at rate.
                'phantom_regime_change_triggered': order.phantom_regime_change_exit_triggered_at is not None,
                'phantom_regime_change_exit_triggered_at': order.phantom_regime_change_exit_triggered_at,
                'phantom_regime_change_exit_pnl': order.phantom_regime_change_exit_pnl,
                'phantom_tick_a_triggered': order.phantom_tick_a_triggered_at is not None,
                'phantom_tick_a_triggered_at': order.phantom_tick_a_triggered_at,
                'phantom_tick_a_pnl': order.phantom_tick_a_pnl,
                'phantom_tick_b_triggered': order.phantom_tick_b_triggered_at is not None,
                'phantom_tick_b_triggered_at': order.phantom_tick_b_triggered_at,
                'phantom_tick_b_pnl': order.phantom_tick_b_pnl,
                'phantom_tick_c_triggered': order.phantom_tick_c_triggered_at is not None,
                'phantom_tick_c_triggered_at': order.phantom_tick_c_triggered_at,
                'phantom_tick_c_pnl': order.phantom_tick_c_pnl,
                'phantom_tick_d_triggered': order.phantom_tick_d_triggered_at is not None,
                'phantom_tick_d_triggered_at': order.phantom_tick_d_triggered_at,
                'phantom_tick_d_pnl': order.phantom_tick_d_pnl,
                'phantom_tick_e_triggered': order.phantom_tick_e_triggered_at is not None,
                'phantom_tick_e_triggered_at': order.phantom_tick_e_triggered_at,
                'phantom_tick_e_pnl': order.phantom_tick_e_pnl,
                'phantom_tick_f_triggered': order.phantom_tick_f_triggered_at is not None,
                'phantom_tick_f_triggered_at': order.phantom_tick_f_triggered_at,
                'phantom_tick_f_pnl': order.phantom_tick_f_pnl,
                'phantom_tick_g_triggered': order.phantom_tick_g_triggered_at is not None,
                'phantom_tick_g_triggered_at': order.phantom_tick_g_triggered_at,
                'phantom_tick_g_pnl': order.phantom_tick_g_pnl,
                'regime_neutral_hit': order.regime_neutral_hit_at is not None,
                'regime_neutral_hit_at': order.regime_neutral_hit_at,
                'regime_neutral_pnl': order.regime_neutral_pnl,
                'regime_comeback_at': order.regime_comeback_at,
                'regime_comeback_pnl': order.regime_comeback_pnl,
                'regime_opposite_at': order.regime_opposite_at,
                'regime_opposite_pnl': order.regime_opposite_pnl,
                # Pattern Cell Ship rule overrides (May 21) — restored from DB on recovery.
                # Trades opened pre-May-21 have these NULL → fall through to default exit ladder.
                'pattern_cell_source': getattr(order, 'pattern_cell_source', None),
                'pattern_fixed_tp_pct': getattr(order, 'pattern_fixed_tp_pct', None),
                'pattern_fixed_sl_pct': getattr(order, 'pattern_fixed_sl_pct', None),
            }

            if order.pair not in new_cache:
                new_cache[order.pair] = []
            new_cache[order.pair].append(order_info)
        
        async with _cache_lock:
            # Preserve realtime-tracked peaks that the DB may not have yet.
            # The realtime callback updates peak_pnl/high_price/low_price in
            # the cache between polling cycles; a naive overwrite would lose them.
            for pair, new_orders in new_cache.items():
                old_orders = _open_orders_cache.get(pair, [])
                for new_info in new_orders:
                    for old_info in old_orders:
                        if old_info['id'] == new_info['id']:
                            new_info['peak_pnl'] = max(new_info['peak_pnl'], old_info.get('peak_pnl', 0))
                            new_info['trough_pnl'] = min(new_info['trough_pnl'], old_info.get('trough_pnl', 0))
                            new_info['peak_ema5_gap'] = max(new_info['peak_ema5_gap'], old_info.get('peak_ema5_gap', 0))
                            if old_info.get('peak_pnl', 0) >= new_info.get('peak_pnl', 0):
                                new_info['peak_ema5_dist_pct'] = old_info.get('peak_ema5_dist_pct')
                                new_info['peak_ema5_slope_pct'] = old_info.get('peak_ema5_slope_pct')
                                new_info['peak_reached_at'] = old_info.get('peak_reached_at')
                            if old_info.get('trough_pnl', 0) <= new_info.get('trough_pnl', 0):
                                new_info['trough_reached_at'] = old_info.get('trough_reached_at')
                                new_info['trough_ema5_dist_pct'] = old_info.get('trough_ema5_dist_pct')
                            if old_info.get('ema5_ever_negative'):
                                new_info['ema5_ever_negative'] = True
                            if old_info.get('signal_lost_flagged'):
                                new_info['signal_lost_flagged'] = True
                                new_info['signal_lost_flag_pnl'] = old_info.get('signal_lost_flag_pnl')
                                new_info['signal_lost_flagged_at'] = old_info.get('signal_lost_flagged_at')
                                if old_info.get('fl1_origin'):
                                    new_info['fl1_origin'] = old_info.get('fl1_origin')
                            if old_info.get('fl2_flagged'):
                                new_info['fl2_flagged'] = True
                                new_info['fl2_flagged_at'] = old_info.get('fl2_flagged_at')
                                new_info['fl2_flag_pnl'] = old_info.get('fl2_flag_pnl')
                            # Phase 1 shadow tracking — preserve cross records + pending state
                            for _xkey in (
                                'first_cross_ema13_at', 'first_cross_ema13_pnl_pct',
                                'confirmed_cross_ema13_at', 'confirmed_cross_ema13_pnl_pct',
                                'first_cross_ema20_at', 'first_cross_ema20_pnl_pct',
                                'confirmed_cross_ema20_at', 'confirmed_cross_ema20_pnl_pct',
                                'pending_cross_ema13_started_at', 'pending_cross_ema20_started_at',
                            ):
                                if old_info.get(_xkey) is not None:
                                    new_info[_xkey] = old_info[_xkey]
                            if new_info['direction'] == 'LONG':
                                new_info['high_price'] = max(new_info['high_price'], old_info.get('high_price', 0))
                            else:
                                new_info['low_price'] = min(new_info['low_price'], old_info.get('low_price', float('inf')))
                            new_info['tick_prices'] = old_info.get('tick_prices', [])
                            for _lvl in [1, 2]:
                                for _key in [f'phantom_be_l{_lvl}_triggered', f'phantom_be_l{_lvl}_triggered_at', f'phantom_be_l{_lvl}_would_exit_pnl']:
                                    if old_info.get(_key) is not None:
                                        new_info[_key] = old_info[_key]
                            # May 15 PM bug fix: phantom_be_aggr_* (added May 14) was omitted from the
                            # preservation loop above. Result: monitor cache rebuild silently reset arm
                            # flags to False between realtime ticks; if an exit (e.g., EMA13_CROSS) fired
                            # before the next tick could re-arm, the counterfactual recorded nothing.
                            for _key in ('phantom_be_aggr_triggered', 'phantom_be_aggr_triggered_at', 'phantom_be_aggr_would_exit_pnl'):
                                if old_info.get(_key) is not None:
                                    new_info[_key] = old_info[_key]
                            # May 20 bug fix: same omission for phantom_regime_change_* (added May 11).
                            # Cache rebuilds were silently wiping the captured regime-flip moment.
                            # Result before fix: 1 of 278 closed trades had phantom data (0.4%),
                            # vs 9-15 trades where regime did flip (regime_opposite_at populated).
                            # See CLAUDE.md May 20 entry for diagnosis.
                            for _key in ('phantom_regime_change_triggered', 'phantom_regime_change_exit_triggered_at', 'phantom_regime_change_exit_pnl'):
                                if old_info.get(_key) is not None:
                                    new_info[_key] = old_info[_key]
                            for _lbl in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                                for _key in [f'phantom_tick_{_lbl}_triggered', f'phantom_tick_{_lbl}_triggered_at', f'phantom_tick_{_lbl}_pnl']:
                                    if old_info.get(_key) is not None:
                                        new_info[_key] = old_info[_key]
                            if old_info.get('regime_neutral_hit'):
                                new_info['regime_neutral_hit'] = True
                                new_info['regime_neutral_hit_at'] = old_info.get('regime_neutral_hit_at')
                                new_info['regime_neutral_pnl'] = old_info.get('regime_neutral_pnl')
                            if old_info.get('regime_comeback_at') is not None:
                                new_info['regime_comeback_at'] = old_info['regime_comeback_at']
                                new_info['regime_comeback_pnl'] = old_info.get('regime_comeback_pnl')
                            if old_info.get('regime_opposite_at') is not None:
                                new_info['regime_opposite_at'] = old_info['regime_opposite_at']
                                new_info['regime_opposite_pnl'] = old_info.get('regime_opposite_pnl')
                            break
            _open_orders_cache.clear()
            _open_orders_cache.update(new_cache)
        
        logger.debug(f"[CACHE] Updated orders cache: {len(orders)} orders across {len(new_cache)} pairs")


# Global trading engine instance
trading_engine = TradingEngine()


async def realtime_stop_loss_callback(pair: str, price: float):
    """Callback function for WebSocket price updates to check stop loss in real-time"""
    await trading_engine.check_realtime_stop_loss(pair, price)
