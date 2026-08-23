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

from models import Order, Transaction, BotState, PairData, BnbSwapLog, PhantomFlip, MonitorPeriod
from database import AsyncSessionLocal
import config
from config import save_trading_config, TradingConfig
from services.binance_service import binance_service, _leverage_blocked_pairs
from services.indicators import calculate_indicators, get_signal, check_exit_conditions, calculate_pnl, determine_macro_regime, is_signal_direction_active, gap_expand_marginal, gap_expand_flat, gap_min_band, _rsi_adx_block_rule, rsiceil_band, adxmax_band, adxmax2_band, gminflat_band
from services.regime import classify_btc_regime
from services.hard_tp_ladder import parse_hard_tp_ladder, hard_tp_ladder_floor, DEFAULT_LADDER_RUNGS


def _strip_reason_prefixes(reason):
    """Strip FLIP_ then FL_ then BR_ prefixes from a close reason (the whitelist convention)."""
    r = reason or ""
    if r.startswith("FLIP_"):
        r = r[5:]
    if r.startswith("FL_"):
        r = r[3:]
    if r.startswith("BR_"):  # Aug-21 gate 57: bull-run sleeve exits (BR_STOP_LOSS etc.)
        r = r[3:]
    return r
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
_current_btc_adx_prev1: Optional[float] = None  # Aug-22: previous closed bar — arrow = sign(adx - adx_prev1)
# Module-level BTC indicators for regime classification at exit time
_current_btc_adx: float = None
_current_btc_rsi: float = None
_market_bull_pct: float = 0.0
_market_bear_pct: float = 0.0
# 🌊 Aug-21 gate 57: Bull-Run Monitor state (in-memory; flips also logged for post-restart
# forensics). state ∈ DARK/AMBER/GREEN. Schmitt band + crash-latch computed in
# _update_bullrun_monitor (throttled). Entries additionally require updated_at fresh
# (≤30 min) — a stale monitor is NOT GREEN (fail-safe).
_bullrun_monitor: Dict = {
    'state': 'DARK', 'green': False, 'amber': False, 'latch': False,
    'r72': None, 'above': None, 'eff': None, 'r6': None, 'r24': None, 'off24h': None,
    'green_since': None, 'updated_at': 0.0, 'flips': [],
}
_br_dip_state: Dict[str, Dict] = {}      # pair -> {'dipped': bool, 'last_bar_ts': int}
_br_e50h_cache: Dict[str, tuple] = {}    # pair -> (fetched_at_epoch, ema50_1h)
_br_alt_stats: dict = {}  # Aug-23 (20): per-pair {'r6h','above','ts'} stamped by the sleeve hook (rank≤N, non-blacklisted) for the RE-ARM door
_br_last_fire: Dict[str, float] = {}     # pair -> epoch of last sleeve entry OR exit (spacing)
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
# Jul 23 SHADOW-SLOT REVIEW (operator: "simplify what we track"): wide/tierA/tierB price-leash
# variants RETIRED — same slot principle as phantoms: a shadow earns per-tick compute only
# while an armed gate consumes it (wide sanity + tierA/tierB questions resolved May-31/Jun-26).
# 'tight' KEPT (review catch): the Jun-12/16 short-runner revert gates keyed on it read
# FIRED-BUT-SUPERSEDED at the Jul-23 audit (full-size armed shorts cum actual−tight −4.13pp
# N=41 honest-capped, but −3.71 of it is the June strpk era; July current-stack −0.42/10) —
# a FRESH tight-vs-actual gate on the current stack is re-armed in CURRENT_STATE and consumes
# this shadow. Columns stay in models/DB; reopen = re-add the spec line BEFORE the read.
_LEASH_SPECS = [
    ('tight', 'flat', 0.25, 0.25, 0.0),  # 0.25 flat trail — the fresh short-gate benchmark
]
_LEASH_ACT = 0.40    # trailing activation (Aug-14: matches live runner arm + tp_min = 0.40; was 0.45 — stale after 32a8847, missed 0.40-0.45 peakers; before that 0.5 missed the 0.485 peaker 1000PEPE)
_LEASH_SL = -0.7    # hard SL floor (matches live)
# Stretch-exit variants (May 30 ext): exit on EXTENSION fade, not price pullback.
# Live stretch = signed %-distance of price from EMA5 (positive = favorable extension).
#   strpk*  = exit when live stretch retraces to <= Kx PEAK stretch (stretch-trail from peak).
#             K bracket (May 31): 0.5 (strpk) / 0.4 (strpk04) / 0.3 (strpk03) — LOWER K = looser
#             trail = holds the runner longer (more on runners, more giveback on reversers).
#             The cohort settles K the same way tierA/tierB bracket the price-trail params.
#   stren = exit when live stretch falls back to <= ENTRY stretch (extension collapsed to entry)
_STRPK_K = {'strpk': 0.5}
# Jul 23 SHADOW-SLOT REVIEW: strpk04/strpk03/strpk_signed RETIRED (stretch-trail K-bracket
# refuted; no armed consumer). strpk + stren KEPT — NOT for the leash table but because they
# feed the Post-Exit Regret recoverable-regret band (Strpk%/Stren% columns, May-31 wiring).
_STRETCH_NAMES = ('strpk', 'stren')
# Jun 16: ATR-floored give-back trail (chandelier) shadows — exit when P&L retraces
# > N×entry_atr_pct from peak. Tests which N the live runner_trail_short_atr_mult should be.
def _ind_atr_pct(ind):
    """ATR% from a raw indicators dict. The dict carries 'atr' (ABSOLUTE) — there is no
    'atr_pct' key (Jul 10: the misnamed-key class that silenced the weakcap filter for
    12 days; the old `... or indicators.get('atr_pct')` fallbacks were dead code)."""
    try:
        _a, _p = ind.get('atr'), ind.get('price')
        return (_a / _p * 100) if (_a is not None and _p) else ind.get('atr_pct')
    except Exception:
        return None


def _pop_or(ef, key, fallback):
    """ef.pop(key) unless None, else fallback — `is not None` coalesce (a plain `or`
    would eat a legitimate 0.0)."""
    v = ef.pop(key, None)
    return v if v is not None else fallback


def _quiet_sl_for(direction, entry_strategy, entry_atr_pct):
    """🛡 Aug 19 gate 53: quiet-pair conditional SL eligibility. Returns the widened
    SL pct for eligible fills (momentum LONG whose entry ATR% < threshold — the −0.70
    stop is a ≥2.3-ATR flash move on those and mean-reverts 75% of the time), else
    None. Scope: momentum LONGs ONLY (flips/spikes/bull/bounce excluded — flips are
    hot-pair stops by construction, wide SL refuted 5/7; fades already carry −1.5).
    threshold 0 or pct 0 = feature off (instant revert). Fail-safe: any error → None
    (base SL applies)."""
    try:
        th = config.trading_config.thresholds
        thr = float(getattr(th, 'momentum_long_sl_atr_threshold', 0.0) or 0.0)
        qsl = float(getattr(th, 'momentum_long_sl_quiet_pct', 0.0) or 0.0)
        if thr <= 0 or qsl >= 0:
            return None
        if direction != 'LONG':
            return None
        if (entry_strategy or 'MOMENTUM') not in ('MOMENTUM', ''):
            return None
        if entry_atr_pct is None or entry_atr_pct <= 0 or entry_atr_pct >= thr:
            return None
        return qsl
    except Exception:
        return None


_BR_LADDER_CACHE = {}
def _bullrun_ladder_floor(peak, ladder_str):
    """Highest rung floor whose peak threshold ≤ peak, from "peak:floor, ..." (parsed once per string).
    Malformed → None (fail-safe: no ladder, trail unchanged)."""
    try:
        rungs = _BR_LADDER_CACHE.get(ladder_str)
        if rungs is None:
            rungs = []
            try:
                for part in (ladder_str or '').split(','):
                    part = part.strip()
                    if not part or ':' not in part:
                        continue
                    a, b = part.split(':', 1)
                    thr, fl = float(a.strip()), float(b.strip())
                    if fl >= thr:
                        logger.warning(f"[BULLRUN_LADDER] rung '{part}' ignored — floor must be BELOW its peak threshold (else it is a take-profit, not a lock)")
                        continue
                    rungs.append((thr, fl))
                rungs.sort()
                if not rungs and (ladder_str or '').strip():
                    logger.warning(f"[BULLRUN_LADDER] bullrun_ladder '{ladder_str}' yields no valid rungs — ladder OFF, plain 2×ATR trail applies")
            except Exception as _lp_err:
                logger.warning(f"[BULLRUN_LADDER] malformed bullrun_ladder '{ladder_str}' ({_lp_err}) — ladder OFF, plain 2×ATR trail applies")
                rungs = []
            _BR_LADDER_CACHE[ladder_str] = rungs  # cached even when empty → warnings fire once per string, not per tick
        out = None
        for thr, fl in rungs:
            if peak >= thr:
                out = fl
        return out
    except Exception:
        return None


def _bullrun_exit_for(pnl, peak_pnl, entry_atr_pct):
    """🌊 Aug-21 gate 57: dedicated BULLRUN_LONG exit check — the ONLY exit logic sleeve
    trades run (both paths intercept BEFORE the alt exit machinery, so FAST_EXIT / tick /
    RSI / signal-lost / gate-53 quiet-SL never touch them; MAX_HOLD + manual still apply
    via their own paths). Exactly the replay-validated stack: SL = min(base, -(ATR×mult))
    floored; BE arm at peak ≥ arm → floor lock; trail = peak − trail_mult×ATR once armed.
    pnl/peak are the CALLER's net-pnl-% convention (both live paths use net-of-fees price %).
    Trail width uses ENTRY ATR% (live paths carry no rolling pair ATR — known approximation
    vs the sim's rolling ATR, on record in DECISION_LOG 2026-08-21(4)).
    Returns (close: bool, reason: str|None, effective_stop_pct: float). Fail-safe: on any
    error returns the plain base-SL check so a position is never left stop-less."""
    try:
        th = config.trading_config.thresholds
        base = float(getattr(th, 'bullrun_base_sl_pct', -0.7) or -0.7)
        atr = float(entry_atr_pct or 0.0)
        mult = float(getattr(th, 'sl_atr_multiplier', 0.0) or 0.0)
        floor_cap = float(getattr(th, 'sl_atr_widen_floor_pct', 0.0) or 0.0)
        sl = base
        if atr > 0 and mult > 0:
            widened = -(atr * mult)
            if floor_cap < 0:
                widened = max(widened, floor_cap)
            sl = min(base, widened)
        arm = float(getattr(th, 'bullrun_be_arm_pct', 1.0) or 1.0)
        lock = float(getattr(th, 'bullrun_be_lock_pct', 0.2) or 0.2)
        trail_mult = float(getattr(th, 'bullrun_trail_atr_mult', 2.0) or 2.0)
        pk = float(peak_pnl or 0.0)
        if pk >= arm:
            trail_line = (pk - trail_mult * atr) if atr > 0 else lock
            stop_line = max(lock, trail_line)
            reason = "TRAILING_STOP" if trail_line > lock else "BREAKEVEN_EXIT"
            # Aug-21 (15) high-rung profit lock: floor = max(trail line, highest rung ≤ peak)
            _lf = _bullrun_ladder_floor(pk, getattr(th, 'bullrun_ladder', '') or '')
            if _lf is not None and _lf > stop_line:
                stop_line = _lf; reason = "LADDER_FLOOR"
            if pnl <= stop_line:
                if reason == "LADDER_FLOOR":
                    # revert-gate instrument (DECISION_LOG (15)): the trail counterfactual this lock pre-empted
                    logger.info(f"[BULLRUN_LADDER] floor {stop_line:.2f} fired at peak {pk:.2f} (trail line would have been {max(lock, trail_line):.2f})")
                return True, reason, stop_line
            return False, None, stop_line
        if pnl <= sl:
            return True, "STOP_LOSS", sl
        return False, None, sl
    except Exception:
        try:
            return (pnl <= -0.7), ("STOP_LOSS" if pnl <= -0.7 else None), -0.7
        except Exception:
            return False, None, -0.7


# Jul 23 SHADOW-SLOT REVIEW: atr15 RETIRED (refuted Jun-29 — 1.5 over-widens; no consumer).
# atr05 + atr10 are the ONLY leash shadows with armed gates: runner_trail_atr_mult revert
# ("atr05≥atr10 over N≥20 fresh armed longs") + BE-ratchet revert (actual vs lockless-atr10).
_ATR_N = {'atr05': 0.5, 'atr10': 1.0}
# Jun 17 PM: give-back-CAP shadows — ATR-floor at the LIVE N + lock, but give_back capped at
# frac×peak. Varies frac (0.25/0.35/0.50) to tune runner_trail_short_giveback_frac from data
# (which frac captures most without noise-stopping), parallel to how _ATR_N tuned N.
# Jul 23 SHADOW-SLOT REVIEW: cap-frac shadows RETIRED — the flat-cap question was answered
# by the HARD_TP ladder ship (Jul-22), which has its own dedicated mechanism shadow.
_CAP_FRAC = {}
# Jul 6: ARM-LEVEL shadows — the recurring "lower the arm 0.45→0.35/0.40?" question, answered
# with data instead of path-blind CSV counterfactuals. Simulates arming the 0.25-trail at a
# LOWER peak threshold; tracked on EVERY trade from the first tick (unlike the armed-only
# leashes above) so both sides of the trade-off are measured: rescues on 0.35-0.45 peakers
# that died (RPL/AAVE-flip class) AND early-chop on runners that the live 0.45 arm rode.
# Decision offline at N≥30 from the orders CSV (columns ride free).
# Jul 23 SHADOW-SLOT REVIEW: arm-level shadows RETIRED — no armed gate ever registered for
# the 0.35/0.40 arm question; collected columns remain in the CSV history if it reopens.
_ARM_VAR = {}
_ARM_TRAIL = 0.25  # same trail width as the live/flip replica trail

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
_PFLIP_ACT = 0.40             # trailing arm (raw price-move %, matches live tp_min; 0.45→0.40 Aug-14 with 32a8847)
_PFLIP_SL = -0.70             # base hard SL (fresh hypothetical → base, not signal-active wide)
_PFLIP_PB = 0.25              # trailing pullback
_PFLIP_MAX_MIN = 45           # max tracking horizon
_PFLIP_COOLDOWN_MIN = 30      # min minutes between phantoms for the same pair|source

# ── SPIKE-REVERSION phantom (Jul 5, operator-directed; observation-only) ──
# Fade acute BTC velocity extremes: |BTC 15-min move| >= 0.5% → seed a virtual FADE on
# BTCUSDT (spike up → phantom SHORT, spike down → phantom LONG), flat replica exit like
# every phantom. Distinct mechanism from the dead PAIR_RSI_OB sleeve (that faded pair-level
# overbought TRENDS; this fades BTC-level 15m VELOCITY) — never measured before. Pre-committed
# bar (CURRENT_STATE): discussion-worthy only at N>=30 & WR>=60% & avg>=+0.15%; analyst
# prediction ON RECORD: fails the bar (the falsifiable version of the PAIR_RSI_OB tombstone).
# Cooldown/dedup ride _seed_phantom_flip (30min per direction via source key).
_SPIKE_REV_MOVE_PCT = 0.5     # trigger: |move| over the window
_SPIKE_REV_WINDOW_MIN = 15    # lookback window
_BTC_SPIKE_HIST = []          # [(ts, price)] ~5s samples, pruned to window+5min

def _maybe_seed_spike_rev():
    """Jul 30 — RETIRED no-op (phantom retirement; SPIKE_REV_BTC research superseded by
    the live 🚀 spike program)."""
    return


def reset_phantom_flip_state():
    """Clear the in-memory phantom-flip tracking (open virtual fades + per-pair|source
    cooldowns). Called on a full data reset so a fresh batch starts with no carryover —
    the persisted phantom_flips rows are deleted separately in the /api/reset handler.
    Mutates the existing dicts in place (no rebinding) so all module references stay live."""
    _PHANTOM_FLIP_STATE.clear()
    _PFLIP_COOLDOWN.clear()


# Jul 30, 2026 — phantom seeding allowlist REMOVED with the tracker retirement (was the
# Jul-1 keep-sources mechanism; models.PHANTOM_KEEP_SOURCES stays in models.py as schema
# history for the frozen phantom_flips table).


def _seed_phantom_flip(pair, entry_price, blocked_direction, source, cohort=None, entry_fields=None, mode='FADE'):
    """Jul 30 — RETIRED no-op (operator-directed phantom retirement). Virtual phantom
    seeding is off permanently: the phantom->probe pipeline matured (final DEEPGAP read
    N=17 · 71% · Σ+1.85% graduated to DEEPGAP_PROBE the same day) and probes measure the
    same hypotheses with real fills. Call sites left inert on purpose — each one documents
    WHERE a blocked-cohort shadow used to be collected; revisit triggers now live on the
    Funnel v2 Sole counters (the TRENDGAP precedent). FLIP_SHORT_BTC1H_SLOPE's revert
    surface migrated to its Sole count (see CURRENT_STATE)."""
    return


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

def _fan_qs_cell_match(th, qs, bear, rng):
    """FAN flip-SHORT 'winner cell' (Jun 26). Spec `flip_fan_qs_cell` =
    qs_min : bear_min : range_lo-range_hi : size [: lev]. Returns (size, lev, tag) when the
    entry's quality_score ≥ qs_min AND entry_bear_pct ≥ bear_min AND
    range_lo ≤ entry_range_position ≤ range_hi (both ends inclusive, matching the +$671
    winner-cell analysis); else (1.0, 1.0, None). Tag is returned even at size/lev 1.0 so the
    cohort gets its OWN row in Multiplier Cell Performance for tracking. Fail-open on any error."""
    try:
        spec = (getattr(th, 'flip_fan_qs_cell', '') or '').strip()
        if not spec or qs is None or bear is None or rng is None:
            return (1.0, 1.0, None)
        parts = [p.strip() for p in spec.split(':')]
        if len(parts) < 4:
            return (1.0, 1.0, None)
        qs_min = float(parts[0]); bear_min = float(parts[1])
        rlo, rhi = map(float, parts[2].split('-'))
        if qs >= qs_min and bear >= bear_min and rlo <= rng <= rhi:
            size = float(parts[3])
            lev = float(parts[4]) if len(parts) >= 5 and parts[4] else 1.0
            tag = "[QS≥%g×BEAR≥%g×RNG%g-%g]" % (qs_min, bear_min, rlo, rhi)
            return (size, lev, tag)
        return (1.0, 1.0, None)
    except (ValueError, TypeError):
        return (1.0, 1.0, None)

def _lookup_leverage_schedule(schedule, equity):
    """Balance→leverage CEILING (Jun 26). schedule = "bal0:lev0, bal1:lev1, ..." ascending
    balance tiers; returns the max leverage for the highest balance threshold ≤ equity, or
    None if the schedule is empty/unparseable/equity below the lowest tier. Fail-open (None →
    no cap) so a malformed schedule can never block trading or shrink leverage unexpectedly."""
    try:
        s = (schedule or '').strip()
        if not s or equity is None:
            return None
        tiers = []
        for part in s.split(','):
            part = part.strip()
            if not part:
                continue
            bal, lev = part.split(':')
            tiers.append((float(bal.strip()), float(lev.strip())))
        if not tiers:
            return None
        tiers.sort()
        cap = None
        for bal, lev in tiers:
            if equity >= bal:
                cap = lev
            else:
                break
        return cap
    except (ValueError, TypeError):
        return None

def _flip_filters(source, ind):
    """Source-namespaced flip filter layer (Jun 16). Given the flip's `source` and the
    blocked entry's `indicators`, decide whether to VETO the flip, how much to SIZE it, and
    which EXIT mode to use. Each source is an INDEPENDENT branch with its own config namespace
    and its own filter TYPES (FAN uses stretch+regime+strpk+mult; future LONG_UNMATCHED /
    PAIR_RSI_OB branches define their own). Fully fail-open: any error → (False, None, 1.0,
    None) so a filter bug can never block a flip or break the scan.
    Jul 17 FUNNEL v2: the chain now evaluates EVERY gate and collects the full fail list
    instead of stopping at the first block (needed for Sole/Episode/AllF accounting — with
    ~12 conjunctive gates, first-fail-wins made the marginal blocker unmeasurable). Behavior
    is unchanged: blocked iff fails is non-empty, and reason = fails[0] (the same first-fail
    name the legacy counters/logs/phantom-seeds always saw).
    Returns (blocked: bool, reason: str|None, size_mult: float, lev_mult: float,
    exit_mode: str|None, fails: list[str])."""
    try:
        th = config.trading_config.thresholds
        _fails = []
        # PAIR_RSI_OB source (Jun 19): the overbought-fade short (a LONG blocked at pair RSI>65 → fade
        # SHORT). Its OWN per-source filter — fires ONLY in the validated regime allow-list (STRONG_BULL).
        # Cross-batch (Jun18/19, independent windows): S.BULL 76-80% WR / +0.20..+0.32 vs H.BULL 29-47% /
        # negative — a clean regime split. Validated under the FLAT flip exit (= the live flip exit), so the
        # phantom→live haircut is just fees/slippage. RETURNS EARLY so it does NOT inherit FAN's flip-short
        # gates (FAN blocks STRONG_BULL — the exact OPPOSITE of what this source needs). Fail-open: empty
        # allow-list → block (off). TIGHT REVERT: at live N≥15, restrict regimes→'' (off) if WR≤65% OR
        # avg≤+0.05% OR the S.BULL losers gap the SL beyond ~−0.85% (counter-trend tail check).
        if source == 'PAIR_RSI_OB':
            if ind.get('flip_dir') == 'SHORT':
                _ob_set = {s.strip().upper() for s in (getattr(th, 'flip_pair_rsi_ob_short_regimes', '') or '').split(',') if s.strip()}
                _ob_reg = (ind.get('btc_regime') or '').upper()
                if not _ob_set or _ob_reg not in _ob_set:
                    _fails.append("FLIP_PAIR_RSI_OB_REGIME")
                # Pair-ADX floor (Jun 20, N=9 DISCIPLINE-OVERRIDE): the overbought-fade pays ONLY on a
                # blow-off in a STRONGLY-trending pair — live pair-ADX bucket 33+ = 9/89% WR/+$698; every
                # bucket <33 net-negative (18-22 −$75, 22-25 −$138, 25-28 −$163, 28-30 −$76, 30-33 −$91).
                # N=9 << the N≥30 gate → shipped with a TIGHT REVERT (flip_pair_rsi_ob_adx_min→0). Fail-open:
                # missing adx or min=0 → no block. Counter FLIP_PAIR_RSI_OB_ADX.
                _ob_amin = float(getattr(th, 'flip_pair_rsi_ob_adx_min', 0.0) or 0.0)
                if _ob_amin > 0:
                    _ob_adx = ind.get('adx')
                    if _ob_adx is not None and _ob_adx < _ob_amin:
                        _fails.append("FLIP_PAIR_RSI_OB_ADX")
                # Jun 22 — pair EMA13-EMA50 gap ceiling (parabola guard). PAIR_RSI_OB returns early below so it
                # never inherited the universal flip_short_pair_gap_max block; replicate it here on its OWN field.
                # Don't fade a pair already steeply extended above its 4h trend (gap≥max) — it never arms and the
                # 20× gaps the SL. N=22 in-sample but mirrors the cross-batch FAN gap filter. Counter
                # FLIP_PAIR_RSI_OB_GAP. Fail-open: missing gap or max=0 → no block.
                _ob_pgmax = float(getattr(th, 'flip_pair_rsi_ob_pair_gap_max', 0.0) or 0.0)
                if _ob_pgmax > 0:
                    _ob_pgap = ind.get('pair_gap')
                    if _ob_pgap is not None and _ob_pgap >= _ob_pgmax:
                        _fails.append("FLIP_PAIR_RSI_OB_GAP")
                # Jun 21: the BTC-ADX>40 cohort is PROMOTED from de-risked 1x to full 20x after its first
                # live batch — at the raised pADX>=45 floor it was 17/82%WR/+0.20%avg with BE-compat 67%
                # (≥60% of losers armed, only 1/17 gapped). De-risk removed → all pADX>=45 STRONG_BULL fires
                # at full lev regardless of BTC ADX. N=17/one-batch DISCIPLINE-OVERRIDE; revert = set
                # flip_pair_rsi_ob_btc_adx_high_mode→off (stops the >40 seed) and/or floor→40.
            if _fails:
                return (True, _fails[0], 1.0, 1.0, None, _fails)
            return (False, None, 1.0, 1.0, None, _fails)
        # Universal fan-SPIKE block (ALL sources, Jun 17): refuse a flip that fades a violently-
        # accelerating parabolic fan (ratio >= flip_fan_spike_max) — it never arms, runs to SL.
        # Cross-batch N=3 / 0% WR (ASTER/VELVET/ALLO). Mirrors the live long-side fan>=5 block.
        _fan = ind.get('fan_ratio')
        _fmax = float(getattr(th, 'flip_fan_spike_max', 0.0) or 0.0)
        if _fmax > 0 and _fan is not None and _fan >= _fmax:
            _fails.append("FLIP_FAN_SPIKE")
        # Universal 2D regime×ADXΔ block for flip-SHORTS (ALL sources, Jun 17). Block a short flip
        # when entry ADXΔ < adxd_max AND BTC regime ∈ blocked set. Cross-batch BULL/CHOP ∧ ADXΔ<0 =
        # N=38/40%WR/-$1070; 96% of losers peak <0.45 arm so the give-back cap can't save them →
        # entry block. Fail-open: empty regimes or missing data → no block.
        # B1 (Jun 17): anti-parabola — block flip-SHORT when EMA5 stretch ≥ max. Shorting a vertical
        # blow-off that keeps ripping (ESPORTS 10.47% stretch → −2.25% gapped stop in 0s). Pool stretch≥2
        # = N=2/0%WR (ASTER+ESPORTS), 0 winners removed (1–2% band 67%WR preserved). Regime-agnostic
        # catastrophe guard. Fail-open: missing stretch or max=0 → no block.
        _smax = float(getattr(th, 'flip_short_stretch_block_max', 0.0) or 0.0)
        if ind.get('flip_dir') == 'SHORT' and _smax > 0:
            _sstr = ind.get('ema5_stretch')
            if _sstr is not None and _sstr >= _smax:
                _fails.append("FLIP_SHORT_HISTRETCH")
        _regs = (getattr(th, 'flip_short_regime_block_regimes', '') or '').strip()
        _anyregs = (getattr(th, 'flip_short_regime_block_any_adxd_regimes', '') or '').strip()
        if ind.get('flip_dir') == 'SHORT' and (_regs or _anyregs):
            _adxd = ind.get('adx_delta'); _reg = ind.get('btc_regime')
            _amax = float(getattr(th, 'flip_short_regime_block_adxd_max', 0.0) or 0.0)
            _regset = {s.strip() for s in _regs.split(',') if s.strip()}
            _anyset = {s.strip() for s in _anyregs.split(',') if s.strip()}
            # B2 (Jun 17): block any-ADXΔ in these regimes (STRONG_BULL loses both ADXΔ halves).
            if _reg and _reg in _anyset:
                _fails.append("FLIP_SHORT_REGIME")
            if _adxd is not None and _reg and _adxd < _amax and _reg in _regset:
                _fails.append("FLIP_SHORT_REGIME")
        # BTC 30m-RSI-rising block for flip-SHORTS (Jun 18): the cleanest cross-batch differentiator. FAN
        # flip-shorts LOSE when BTC 30m RSI is rising (macro bouncing → the faded pump squeezes with it) and
        # PAY when falling. 2-batch consistent (rising −$1031 vs falling +$811; today −$965/−$998 was rising).
        # Block SHORT when (btc_rsi − btc_rsi_prev6) > min. Fail-open: missing data or min≥99 → no block.
        _b30raw = getattr(th, 'flip_short_btc30_rise_block_min', 99.0)
        _b30min = 99.0 if _b30raw is None else float(_b30raw)  # None -> 99 (off, fail-open); NOT `or 0.0` which fail-CLOSED (Jul 3 review)
        if ind.get('flip_dir') == 'SHORT' and _b30min < 99:
            _br, _br6 = ind.get('btc_rsi'), ind.get('btc_rsi_prev6')
            if _br is not None and _br6 is not None and (_br - _br6) > _b30min:
                _fails.append("FLIP_SHORT_BTC30_RISE")
        # BTC 1h-slope regime gate for flip-SHORTS (Jul 3): don't fade an alt pump while BTC's HOURLY
        # trend is rising — in a recovery the pump is real and squeezes the short; when the hour falls
        # it's exhaustion and the fade pays. The ONLY flip separator (of 8 tested) direction-consistent
        # across periods: baseline slope>0 = 17/65%WR/−$73 vs ≤0 = 29/76%/+$774; fresh Jun30-Jul3
        # slope>0 = 9/33%/−$405 vs ≤0 = 7/71%/+$51. Parity fix: momentum shorts already carry
        # btc_1h_slope_max(+0.1) — flips bypassed it. Fail-open: missing slope or max≥99 → no block.
        _b1hraw = getattr(th, 'flip_short_btc_1h_slope_max', 99.0)
        _b1hmax = 99.0 if _b1hraw is None else float(_b1hraw)  # NOT `or` — the ship value 0.0 is falsy
        if ind.get('flip_dir') == 'SHORT' and _b1hmax < 99:
            _b1h = ind.get('btc_1h_slope')
            if _b1h is not None and _b1h > _b1hmax:
                # Jul 6 — gate revert fired (blocked phantoms 18·78% ≥ locked 60%/N≥10) → graduated
                # response: admit LIVE with cell mult capped at flip_short_btc_1h_slope_admit_mult
                # (ship 1.0 = 1× while base flips also 1×; cap survives any future base re-mux).
                # Tag applied at the open site (B1H_SLOPEUP) so the cohort tracks as its own cell.
                # admit_mult 0/None = legacy hard block. Downstream gates still run (net-admissible
                # by construction — the phantom surface's known overcount is bypassed entirely).
                _adm_raw = getattr(th, 'flip_short_btc_1h_slope_admit_mult', 0.0)
                _adm = 0.0 if _adm_raw is None else float(_adm_raw)
                if _adm <= 0:
                    _fails.append("FLIP_SHORT_BTC1H_SLOPE")
                # fall through un-blocked; sizing cap + tag handled in _maybe_open_flip
        # Aug-23 (21): FAN_RATIO_GATE shorts need a BEARISH BTC — block while BTC sits at/above its 5m EMA13
        # (distance > max, default −0.08%). Fail-open on missing distance; None/blank config = off.
        _fe13_raw = getattr(th, 'flip_fan_btc_ema13_max', None)
        if source == "FAN_RATIO_GATE" and ind.get('flip_dir') == 'SHORT' and _fe13_raw is not None and str(_fe13_raw) != '':
            _bd13 = ind.get('btc_dist_ema13')
            if _bd13 is not None and float(_bd13) > float(_fe13_raw):
                _fails.append("FLIP_FAN_BTC_EMA13")
        # BTC trend-gap DEPTH gate for flip-SHORTS (Jul 8): don't fade an alt pump while BTC sits
        # DEEP below its own trend (EMA13-50 gap ≤ min) — oversold tape means the pump is a
        # market-wide relief bounce that keeps squeezing, not idiosyncratic exhaustion. Found by the
        # full 31-dimension winner/loser sweep (the ONLY ship-grade survivor): baseline MONOTONE
        # across 5 buckets (≤−0.30 = 25% WR → −0.10..0 = 100% WR), direction-consistent in the fresh
        # 07-06..08 window; at −0.22 blocked = 16 (12 base + 4 fresh) · ~44% WR · −$417 over 10+
        # dates, diffuse pairs (top 28%); kept baseline 27·85%·+$881. COMPLEMENTARY to the 1h slope
        # gate (overlap 2/65; corr +0.49 — depth vs direction; cuts WITHIN regimes: 11/18 S.BEAR vs
        # 3/17 H.BEAR). ⚠ N=16<30 near-gate ship (operator-directed, same evidence pattern as the
        # Jul-3 1h gate at N=26) → TIGHT REVERT via PASS phantoms (see CURRENT_STATE). Sentinel
        # 0 = off (active when < 0). Fail-open: missing gap → no block.
        _tgmin_raw = getattr(th, 'flip_short_btc_trend_gap_min', 0.0)
        _tgmin = 0.0 if _tgmin_raw is None else float(_tgmin_raw)
        if ind.get('flip_dir') == 'SHORT' and _tgmin < 0:
            _btg = ind.get('btc_trend_gap')
            if _btg is not None and _btg <= _tgmin:
                _fails.append("FLIP_SHORT_BTC_TRENDGAP")
        # High-ATR bear block (Jun 17): the REGIME-INVERTED hole in FLIP_SHORT_REGIME's bear exemption.
        # A high-ATR parabolic pump in a strong bear is a counter-trend short-SQUEEZE that keeps ripping →
        # the short never arms and the high ATR gaps the −0.70 SL to ~−1.2 (ESPORTS 4.0, HUSDT 3.0 = 0%WR/
        # −$245). The CUT IS HIGH (≥3, NOT ≥2): below ~2.5 bear shorts are net-positive (PORTAL 1.5 +$322,
        # BR 2.0 +$272, STG 1.5 +$40) — cutting at 2 would kill proven winners. Same high-ATR pair WINS in
        # bull (regime inversion) so the block is bear-only. Operator-directed N=2/one-window → TIGHT REVERT.
        # Fail-open: missing atr/regime or min=0 → no block.
        _haregs = (getattr(th, 'flip_short_atr_block_regimes', '') or '').strip()
        if ind.get('flip_dir') == 'SHORT' and _haregs:
            _hamin = float(getattr(th, 'flip_short_atr_block_min', 0.0) or 0.0)
            _hatr = ind.get('atr_pct'); _hareg = ind.get('btc_regime')
            _haset = {s.strip() for s in _haregs.split(',') if s.strip()}
            if _hamin > 0 and _hatr is not None and _hatr >= _hamin and _hareg and _hareg in _haset:
                _fails.append("FLIP_SHORT_HIATR")
        # Pair-RSI floor for flip-SHORTS (Jun 19): fade quality scales with how overbought the blocked
        # long was. Cross-batch (Jun17/18/19 deduped) RSI<55 = N=21/57%WR/-0.094%/Σ-1.98 (only consistently
        # negative zone); RSI>=55 = N=78/65%WR/+0.056%/Σ+4.33 (carries ~all the edge); 60-65 = 71%WR/+0.187.
        # Block SHORT when pair RSI < min. Operator-directed, N below filter gate → TIGHT REVERT.
        # Fail-open: missing rsi or min=0 → no block.
        _rmin = float(getattr(th, 'flip_short_rsi_min', 0.0) or 0.0)
        if ind.get('flip_dir') == 'SHORT' and _rmin > 0:
            _prsi = ind.get('pair_rsi')
            if _prsi is not None and _prsi < _rmin:
                _fails.append("FLIP_SHORT_RSI_MIN")
        # Quality-score floor for flip-SHORTS (Jun 25): the global Entry-Quality-Score filter
        # blocks score ≤1 for NORMAL entries but flips BYPASS it. Cross-batch FAN flip-short is
        # monotonic in quality score; score ≤1 is the only negative band (N=18/56%WR/−2.98%/8 dates,
        # loss diffuse across 16 pairs). Block flip-SHORT when score < min (min=2 → blocks ≤1).
        # Fail-open: missing score or min=0 → no block. Counter FLIP_SHORT_QUALITY.
        _qmin = float(getattr(th, 'flip_short_quality_min', 0.0) or 0.0)
        if ind.get('flip_dir') == 'SHORT' and _qmin > 0:
            _qsc = ind.get('quality_score')
            if _qsc is not None and _qsc < _qmin:
                _fails.append("FLIP_SHORT_QUALITY")
        # Market-breadth FLOOR for flip-SHORTS (2026-06-29): a fade-short needs the broad market falling
        # to follow through; when breadth is bullish/neutral the fade has no tailwind → DOA grind → 20× SL
        # gap. Fine bear-band split (COMBINED in-sample + 06-29 forward) localises the loss ENTIRELY to
        # bear<20 (1W/4L/−$314, NO high-ATR confound); bear 30-40 WINS (+$146) and 50-80 are the edge, so
        # only <20 is a clean cut (a <40 floor would forfeit winners). Block flip-SHORT when entry bear% <
        # min (=20). Counter FLIP_SHORT_BEAR_MIN (auto-recorded by caller). Fail-open: missing bear% or
        # min=0 → no block. N=5/DISCIPLINE-OVERRIDE → tight revert (see config flip_short_bear_min).
        _bmin = float(getattr(th, 'flip_short_bear_min', 0.0) or 0.0)
        if ind.get('flip_dir') == 'SHORT' and _bmin > 0:
            _bp = ind.get('bear_pct')
            if _bp is not None and _bp < _bmin:
                _fails.append("FLIP_SHORT_BEAR_MIN")
        # Universal collapsing-pair-ADX block for flip-SHORTS (Jun 28): flip shorts BYPASS the momentum-
        # short `Pair ADX Dir S: rising` filter, so a flip-short can fire into a pair whose ADX is
        # COLLAPSING (ADXΔ << 0 = the trend that justified the fade is dying → no follow-through →
        # never arms → 20× gaps the SL). Strongest flip-short loser-separator cross-batch; 06-28 BEL
        # (STRONG_BEAR, ADXΔ −1.02) = −$195 confirms (cut blocks BEL only, 0 winners, +$195). Block
        # SHORT when ADXΔ < min. SENTINEL −99 = OFF. Counter FLIP_SHORT_ADXD (auto-recorded by the
        # caller). Fail-open: missing ADXΔ or min≤−99 → no block.
        _admin_raw = getattr(th, 'flip_adx_delta_min', -99.0)
        _admin = -99.0 if _admin_raw is None else float(_admin_raw)  # NOT `or` — 0.0 ('block ADXd<0') is falsy and would silently disable (Jul 3 review)
        if ind.get('flip_dir') == 'SHORT' and _admin > -99:
            _fad = ind.get('adx_delta')
            if _fad is not None and _fad < _admin:
                _fails.append("FLIP_SHORT_ADXD")
        # Pair EMA13-EMA50 gap ceiling for flip-SHORTS (Jun 21): refuse to fade a pair already steeply
        # extended above its OWN 4h trend — a parabola that keeps ripping → the 20× short gaps the SL
        # to ~-1.2%. Cross-batch FAN survivors (Jun17-21 deduped) gap≥1.0 = N=16/44%WR/-0.359%/Σ-$461,
        # net-negative every batch; the 0-1.0 band is the fade sweet spot (19/87%WR/+0.45%). ONE-SIDED:
        # a big NEGATIVE gap is with-trend momentum that WINS (≤-1.5 = +0.79%) → only the positive
        # (counter-trend) tail is blocked. The flip-side of the live non-flip pair_trend_short filter,
        # tuned for flips. N=16/DISCIPLINE-OVERRIDE → TIGHT REVERT. Fail-open: missing gap or max=0.
        _pgmax = float(getattr(th, 'flip_short_pair_gap_max', 0.0) or 0.0)
        if ind.get('flip_dir') == 'SHORT' and _pgmax > 0:
            _pgap = ind.get('pair_gap')
            if _pgap is not None and _pgap >= _pgmax:
                _fails.append("FLIP_SHORT_PAIR_GAP")
        # Mirror for flip-LONGS (Jun 17): block a long flip when BTC regime ∈ bear set. A flip-LONG
        # fades a blocked SHORT -> goes LONG; in a STRONG_BEAR that's long-into-the-trend (AAVE/TAO
        # this batch: 2/0%WR/-$220, both straight to SL). UNLIKE the short gate, the observed long
        # losers were ADXΔ-AGNOSTIC (ADXΔ +1.5 — regime was the killer, not falling ADX) → default
        # Flip-LONG HARD DISABLE (Jun 27): fade blocked LONGs→SHORT only; never fade a blocked
        # SHORT→LONG. Flip-LONG is a net-negative micro-sleeve (N=8/−$297; fresh H.BULL countertrend
        # losers DYDX −$115 + XPL −$164 = 0/2). Discipline-override (N=2 < N≥10 gate). Counter
        # FLIP_LONG_DISABLED. Fail-open: default enabled. Sits ABOVE the regime block (supersedes it).
        if ind.get('flip_dir') == 'LONG' and not getattr(th, 'flip_long_enabled', True):
            _fails.append("FLIP_LONG_DISABLED")
        # adxd_max high (99) so the gate is regime-dominant; operator can lower it to add an ADXΔ
        # cut later if a long ADXΔ cell proves out. Fail-open: empty regimes or missing regime → no block.
        _lregs = (getattr(th, 'flip_long_regime_block_regimes', '') or '').strip()
        if ind.get('flip_dir') == 'LONG' and _lregs:
            _ladxd = ind.get('adx_delta'); _lreg = ind.get('btc_regime')
            _lamax_raw = getattr(th, 'flip_long_regime_block_adxd_max', 99.0)
            _lamax = 99.0 if _lamax_raw is None else float(_lamax_raw)  # NOT `or` — 0.0 is a valid threshold (Jul 3 review)
            _lregset = {s.strip() for s in _lregs.split(',') if s.strip()}
            if _lreg and _lreg in _lregset and (_ladxd is None or _ladxd < _lamax):
                _fails.append("FLIP_LONG_REGIME")
        if source == "FAN_RATIO_GATE":
            stretch = ind.get('ema5_stretch')
            brsi = ind.get('btc_rsi'); badx = ind.get('btc_adx')
            size = 1.0; lev = 1.0
            # Jun 17 — the FAN entry filters (thin-fuel, regime-block) AND the size/lev multiplier
            # cell are ALL derived & validated ONLY on SHORT fades (the strong-bear mirror). A FAN
            # flip-LONG (blocked SHORT -> LONG) must NOT inherit them — it would size 2x into a
            # bearish BTC (the cell's BTC RSI 40-45 x ADX>=35 fires on macro state regardless of
            # direction) and the short-reasoning regime-block is backwards for a long. Gate them all
            # to flip_dir=='SHORT'; a flip-LONG falls through at 1x with no FAN entry veto.
            # (Operator-confirmed on the BCHUSDT flip-LONG that wrongly carried FAN_RATIO_GATE x2.)
            if ind.get('flip_dir') == 'SHORT':
                # U3 (Jun 20) — block FAN flip-short when BTC ATR% < min (weekend thin-liquidity
                # regime). FAN-ONLY: the sub-0.10 losses are 20× gap-through-SL fat tails clustered
                # in bear/chop; PAIR_RSI_OB (S.BULL, pADX>=40) is a different setup handled by its own
                # floor, so it returns early above and never reaches here. Cross-batch context: weekday
                # batches (Jun17 Wed / Jun18 Thu) never dipped <0.109 → the regime is weekend-only;
                # the Jun20 Saturday cell = N=14/36%WR/-$775, every loss a straight-to-SL gap. N=14 /
                # ONE weekend DISCIPLINE-OVERRIDE → TIGHT REVERT (set min→0 if next weekend's sub-0.10
                # phantom-fade ≥45% WR on N≥6). Fail-open: missing btc_atr or min=0 → no block.
                _loatr = float(getattr(th, 'flip_fan_btc_atr_min', 0.0) or 0.0)
                if _loatr > 0:
                    _batr = ind.get('btc_atr_pct')
                    if _batr is not None and _batr < _loatr:
                        _fails.append("FLIP_FAN_LOATR")
                # 1) thin-fuel block
                smin = float(getattr(th, 'flip_fan_stretch_min', 0.0) or 0.0)
                if smin > 0 and stretch is not None and stretch < smin:
                    _fails.append("FLIP_FAN_STRETCH")
                # 2) regime block — fade into a strong, un-exhausted bull
                rmin = float(getattr(th, 'flip_fan_block_btc_rsi', 0.0) or 0.0)
                amin = float(getattr(th, 'flip_fan_block_btc_adx', 0.0) or 0.0)
                if rmin > 0 and amin > 0 and brsi is not None and badx is not None and brsi >= rmin and badx >= amin:
                    _fails.append("FLIP_FAN_REGIME")
                # 2b) pair-ADX floor (Jun 23) — FAN flips BYPASS the momentum short system's pair-ADX
                # requirement (Pair ADX Dir rising + ADX-Strong>20), so they fire weak-trend fades
                # (pADX 15-19) with no follow-through that chop/gap back. Restore the floor. Cross-batch
                # (3 batches J20/J22a/J22b, deduped N=89): pADX>=20 = 42/71%WR/+$482 vs <20 = 47/51%WR/-$850
                # (the whole drain); KEEP>BLOCK + WR up every batch; loss diffuse (top-2 pairs 28% →
                # dimension not blacklist). Fail-open: missing adx or min=0.
                _padxmin = float(getattr(th, 'flip_fan_pair_adx_min', 0.0) or 0.0)
                # 2026-06-30: regime exemption — the pADX<min floor LOSES in HEALTHY_BEAR (block correct)
                # but WINS in STRONG_BEAR (78%WR/+$310, ex-top-2 +$58), so don't fire it there. Empty set
                # = universal (no exemption). N=9/one-window DISCIPLINE-OVERRIDE; revert = clear the regimes.
                _padx_exempt = {s.strip() for s in (getattr(th, 'flip_fan_pair_adx_exempt_regimes', '') or '').split(',') if s.strip()}
                if _padxmin > 0 and (ind.get('btc_regime') or '') not in _padx_exempt:
                    _padx = ind.get('adx')
                    if _padx is not None and _padx < _padxmin:
                        _fails.append("FLIP_FAN_PAIR_ADX")
                # 3) size/lev multiplier — the FAN flip-SHORT "winner cell" (qs/bear/range) is
                #    applied DOWNSTREAM in _maybe_open_flip via _fan_qs_cell_match (so it can carry
                #    a distinct cell_multiplier_source tag even at 1× for tracking). This branch
                #    leaves size/lev at 1.0; the override happens at the open site.
            # 4) exit mode (short runner stretch-trail; short-only at execution, harmless for longs)
            exitm = "strpk" if getattr(th, 'flip_fan_runner_strpk', False) else None
            if _fails:
                return (True, _fails[0], 1.0, 1.0, None, _fails)
            return (False, None, size, lev, exitm, _fails)
        # LONG_UNMATCHED_ONLY / PAIR_RSI_OB: no entry filters yet (their own data pending), but
        # Jun 16 they DO share the SHORT runner stretch-trail exit via flip_runner_strpk_shorts.
        if _fails:
            return (True, _fails[0], 1.0, 1.0, None, _fails)
        return (False, None, 1.0, 1.0, ("strpk" if getattr(th, 'flip_runner_strpk_shorts', False) else None), _fails)
    except Exception:
        return (False, None, 1.0, 1.0, None, [])

def _leash_update(order_id, pnl_pct, peak_hint=None, ema13_crossed=False, signal_lost=False,
                  stretch=None, entry_stretch=None, atr=None):
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
                  'aexits': {n: None for n in _ATR_N},
                  'aexit_mins': {n: None for n in _ATR_N},
                  'cexits': {n: None for n in _CAP_FRAC},
                  'cexit_mins': {n: None for n in _CAP_FRAC},
                  'xexits': {n: None for n in _ARM_VAR},
                  'xexit_mins': {n: None for n in _ARM_VAR},
                  'atr': atr,
                  'pstretch': None, 'estretch': entry_stretch}
            _LEASH_STATE[order_id] = st
        if st.get('atr') is None and atr is not None:
            st['atr'] = atr
        st['ts'] = _leash_time.time()
        if st.get('estretch') is None and entry_stretch is not None:
            st['estretch'] = entry_stretch
        if peak_hint is not None and peak_hint > st['rmax']:
            st['rmax'] = peak_hint
        if pnl_pct > st['rmax']:
            st['rmax'] = pnl_pct
        rmax = st['rmax']
        # ---- Jul 6: ARM-LEVEL shadows — run BEFORE the 0.45 arm guard (their whole point is the
        # sub-0.45 zone). Once running peak ≥ variant threshold: floor = peak − 0.25; exit at floor.
        # Same backstops as the other leashes. Unfired → finalize falls back to actual ('window').
        _xmin_now = round((_leash_time.time() - st['open_ts']) / 60.0, 2)
        for _xn, _xarm in _ARM_VAR.items():
            if st['xexits'][_xn] is not None:
                continue
            if pnl_pct <= _LEASH_SL:
                st['xexits'][_xn] = (_LEASH_SL, 'hard_sl'); st['xexit_mins'][_xn] = _xmin_now; continue
            if rmax < _xarm:
                continue  # this variant not armed yet
            if ema13_crossed:
                st['xexits'][_xn] = (round(pnl_pct, 4), 'ema13'); st['xexit_mins'][_xn] = _xmin_now; continue
            if signal_lost:
                st['xexits'][_xn] = (round(pnl_pct, 4), 'signal_lost'); st['xexit_mins'][_xn] = _xmin_now; continue
            if pnl_pct <= rmax - _ARM_TRAIL:
                st['xexits'][_xn] = (round(rmax - _ARM_TRAIL, 4), 'trailing'); st['xexit_mins'][_xn] = _xmin_now
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
        # ---- ATR-floored give-back exits (chandelier): exit when P&L retraces > N×ATR from peak.
        # Jun 17 PM: now LOCK-AWARE — floor = max(peak − N×ATR, lock), mirroring the then-LIVE
        # policy (cap-off Jun 17, BE-ratchet/lock ON). Aug-14 (8108a60): live shorts now run
        # cap=0.35 with ratchet OFF, so atr05 is no longer the live replica — it is the
        # UNCAPPED N=0.5 counterfactual the 46c revert gate compares against (cap-bound exits
        # vs these shadows). Gate 46's calm-band read must re-interpret atr05 accordingly.
        # atr10/atr15 remain the N=1.0/1.5 candidates under the SAME lock → the clean decision
        # surface for runner_trail_short_atr_mult. (Pre-Jun-17 these were lockless = the cap-era
        # chandelier; the lock now backstops the low-peak give-back that N alone can't.)
        # _aatr = the pair's entry ATR% captured at first tick. ----
        _aatr = st.get('atr')
        if _aatr and _aatr > 0:
            try:
                _alock = float(getattr(config.trading_config.thresholds, 'runner_trail_short_be_lock_pct', 0.10) or 0.10)
                _aratchet = bool(getattr(config.trading_config.thresholds, 'runner_trail_short_be_ratchet_enabled', True))
            except Exception:
                _alock, _aratchet = 0.10, True
            for aname, _aN in _ATR_N.items():
                if st['aexits'][aname] is not None:
                    continue
                if pnl_pct <= _LEASH_SL:
                    st['aexits'][aname] = (_LEASH_SL, 'hard_sl'); continue
                if ema13_crossed:
                    st['aexits'][aname] = (round(pnl_pct, 4), 'ema13'); continue
                if signal_lost:
                    st['aexits'][aname] = (round(pnl_pct, 4), 'signal_lost'); continue
                _araw = rmax - _aN * _aatr
                _afloor = max(_araw, _alock) if _aratchet else _araw
                if pnl_pct <= _afloor:
                    st['aexits'][aname] = (round(pnl_pct, 4), 'lock' if (_aratchet and _alock > _araw) else 'atr')
        # ---- give-back-CAP shadows (Jun 17 PM): mirror the LIVE exit (ATR-floor at live N + lock)
        # but cap the give-back at frac×peak; vary frac to tune runner_trail_short_giveback_frac. ----
        if _aatr and _aatr > 0:
            try:
                _cN = float(getattr(config.trading_config.thresholds, 'runner_trail_short_atr_mult', 0.5) or 0.5)
                _clock = float(getattr(config.trading_config.thresholds, 'runner_trail_short_be_lock_pct', 0.10) or 0.10)
            except Exception:
                _cN, _clock = 0.5, 0.10
            for cname, _cfrac in _CAP_FRAC.items():
                if st['cexits'][cname] is not None:
                    continue
                if pnl_pct <= _LEASH_SL:
                    st['cexits'][cname] = (_LEASH_SL, 'hard_sl'); continue
                if ema13_crossed:
                    st['cexits'][cname] = (round(pnl_pct, 4), 'ema13'); continue
                if signal_lost:
                    st['cexits'][cname] = (round(pnl_pct, 4), 'signal_lost'); continue
                _cfloor = max(rmax - min(_cN * _aatr, _cfrac * rmax), _clock)
                if pnl_pct <= _cfloor:
                    st['cexits'][cname] = (round(pnl_pct, 4), 'cap')
        # ---- stamp fire-minute (from open) on whichever leash just fired this tick ----
        _emin = round((st['ts'] - st['open_ts']) / 60.0, 2)
        for _n in st['exits']:
            if st['exits'][_n] is not None and st['exit_mins'].get(_n) is None:
                st['exit_mins'][_n] = _emin
        for _sn in st['sexits']:
            if st['sexits'][_sn] is not None and st['sexit_mins'].get(_sn) is None:
                st['sexit_mins'][_sn] = _emin
        for _an in st['aexits']:
            if st['aexits'][_an] is not None and st['aexit_mins'].get(_an) is None:
                st['aexit_mins'][_an] = _emin
        for _cn in st['cexits']:
            if st['cexits'][_cn] is not None and st['cexit_mins'].get(_cn) is None:
                st['cexit_mins'][_cn] = _emin
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
        for aname in _ATR_N:
            if st and st.get('aexits', {}).get(aname) is not None:
                out[aname] = st['aexits'][aname]
                out[aname + '_min'] = st.get('aexit_mins', {}).get(aname)
            else:
                out[aname] = (round(fallback_pnl, 4) if fallback_pnl is not None else None, 'window')
                out[aname + '_min'] = None
        for cname in _CAP_FRAC:
            if st and st.get('cexits', {}).get(cname) is not None:
                out[cname] = st['cexits'][cname]
                out[cname + '_min'] = st.get('cexit_mins', {}).get(cname)
            else:
                out[cname] = (round(fallback_pnl, 4) if fallback_pnl is not None else None, 'window')
                out[cname + '_min'] = None
        for xname in _ARM_VAR:
            if st and st.get('xexits', {}).get(xname) is not None:
                out[xname] = st['xexits'][xname]
                out[xname + '_min'] = st.get('xexit_mins', {}).get(xname)
            else:
                out[xname] = (round(fallback_pnl, 4) if fallback_pnl is not None else None, 'window')
                out[xname + '_min'] = None
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
        # Jul 14 FUNNEL v2 (in-memory, not persisted): honest per-filter accounting from the
        # momentum ladder's evaluate-all pass. all = filter failed (regardless of order);
        # sole = filter was the ONLY fail (its true marginal cost in trades); episodes =
        # edge-triggered per (pair, dir, filter) — counts blocked EPISODES, not scan ticks
        # (kills the ~100x re-evaluation inflation). Keys: (filter, direction).
        self._filter_all_counts: Dict[tuple, int] = {}
        self._filter_sole_counts: Dict[tuple, int] = {}
        self._filter_episode_counts: Dict[tuple, int] = {}
        self._filter_blocked_state: Dict[tuple, int] = {}  # (pair, dir, filter) -> last scan seq
        self._funnel_scan_seq: int = 0
        # Jun 18: REAL cap-cost — fully-qualified signals turned away by the position cap (open_position
        # max-pos gate), split normal vs flip. Distinct from the filter-blocks-while-full "blocked_at_max"
        # (which counts filter rejections during full scans, not trades the cap actually prevented). In-memory.
        self._cap_skip_counts: Dict[str, int] = {"normal": 0, "flip": 0}

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
            # Restore Funnel v2 counters (Jul 14) — Sole/AllF/Episode share the same
            # persisted lifetime as the legacy Total columns above. Keys "filter|dir".
            _fv2_json = getattr(state, 'filter_funnel_v2_json', None)
            if _fv2_json:
                try:
                    _fv2_raw = json.loads(_fv2_json)

                    def _fv2_load(name):
                        return {
                            (parts[0], parts[1]): v
                            for k, v in (_fv2_raw.get(name) or {}).items()
                            if len(parts := k.split("|")) == 2
                        }

                    self._filter_all_counts = _fv2_load("all")
                    self._filter_sole_counts = _fv2_load("sole")
                    self._filter_episode_counts = _fv2_load("episode")
                    logger.info(
                        f"[FILTER_BLOCKS] Restored Funnel v2 counters from DB "
                        f"(all={len(self._filter_all_counts)}, sole={len(self._filter_sole_counts)}, "
                        f"episode={len(self._filter_episode_counts)})"
                    )
                except Exception as _e:
                    logger.warning(f"[FILTER_BLOCKS] Failed to restore Funnel v2 counters: {_e}")
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
            if _reason_base.startswith("BR_"):  # Aug 21 gate 57: bull-run sleeve reasons
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
                    _reason_base.startswith("REGIME_CHANGE") or _reason_base.startswith("TRAILING_STOP") or _reason_base.startswith("LADDER_FLOOR") or
                    _reason_base.startswith("RUNNER_TRAIL") or
                    _reason_base.startswith("MOMENTUM_EXIT") or _reason_base.startswith("SLOPE_EXIT") or
                    _reason_base.startswith("NO_EXPANSION") or _reason_base.startswith("RECOVERED") or
                    _reason_base.startswith("DEEP_STOP") or _reason_base.startswith("EMERGENCY_SL") or
                    _reason_base.startswith("FAST_EXIT") or
                    _reason_base.startswith("ATR_FIXED_TP") or
                    _reason_base.startswith("HARD_TP") or  # Jul 20: hard TP cap — regret rows are its revert-gate data
                    _reason_base.startswith("SPIKE_") or  # Jul 27: spike option-D reasons (SL/FLOOR/RSI_COOL) — post-exit rows are the BANANA-watch + fade-looseness read instruments

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
                # Jul 22: HARD_TP mechanism shadow — recovery path (drift-bug lesson: every
                # tracked field must ride BOTH seed dicts). In-memory shadow peaks are lost
                # on restart; resume conservatively from close pnl (if the leash already
                # fired pre-restart the columns are only written at horizon, so re-simulate).
                "htp_shadow": _strip_reason_prefixes(order.close_reason).startswith("HARD_TP"),
                "htp_A_peak": max(float(order.pnl_percentage or 1.0), float(order.peak_pnl or 0.0)),
                "htp_A_exit": None,
                "htp_B_peak": max(float(order.pnl_percentage or 1.0), float(order.peak_pnl or 0.0)),
                "htp_B_exit": None,
                "htp_B_rungs": (parse_hard_tp_ladder(getattr(tc.thresholds,
                    'hard_tp_ladder_long' if order.direction == "LONG" else 'hard_tp_ladder_short', ''))
                    or DEFAULT_LADDER_RUNGS),
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

        # Funnel v2 counters (Jul 14) — persisted alongside the legacy counters so
        # Sole/AllF/Episode survive redeploys. _filter_blocked_state is scan-transient
        # (edge detector vs _funnel_scan_seq, which restarts at 0) and is NOT saved.
        def _fv2_dump(d):
            return {f"{k[0]}|{k[1]}": v for k, v in d.items()}
        _fv2_json = json.dumps({
            "all": _fv2_dump(self._filter_all_counts),
            "sole": _fv2_dump(self._filter_sole_counts),
            "episode": _fv2_dump(self._filter_episode_counts),
        }) if (self._filter_all_counts or self._filter_sole_counts or self._filter_episode_counts) else None

        if state:
            state.is_running = self.is_running
            state.is_paper_mode = self.is_paper_mode
            state.paper_balance = self.paper_balance
            state.paper_bnb_balance_usd = self.paper_bnb_balance_usd
            state.total_runtime_seconds = self.total_runtime_seconds
            state.started_at = self.started_at
            state.updated_at = datetime.utcnow()
            state.filter_block_counts_json = _fb_json
            state.filter_funnel_v2_json = _fv2_json
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
                filter_funnel_v2_json=_fv2_json,
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
            # Aug-22 (operator): persistent ADX direction arrow — closed-bar ADX vs previous closed bar
            "btc_adx": round(_current_btc_adx, 2) if _current_btc_adx is not None else None,
            "btc_adx_prev1": round(_current_btc_adx_prev1, 2) if _current_btc_adx_prev1 is not None else None,
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
        # Jul 15 VISIBILITY FIX (audit finding S1): countertrend blocks during clear
        # regimes were silently DROPPED here since May 7 — which let gates whose firing
        # condition correlates with the suppressing regime (e.g. shorts-blockers in a
        # bull) block heavily while their counters barely moved (the BTC_SLOPE_GATE
        # anatomy: +5 counted vs 112 actual kills/day). They are now recorded under
        # room_state="SUPP": the "Real" column stays DECISIVE (regime-aligned, historical
        # continuity), the "All" column becomes the TRUE count (Real + suppressed).
        room_state = "ROOM" if had_room else "FULL"
        try:
            _regime = _current_btc_regime
            if had_room and ((_regime == "BEARISH" and direction == "LONG") or
                             (_regime == "BULLISH" and direction == "SHORT")):
                room_state = "SUPP"  # FULL keeps precedence — at-cap info is real either way
        except NameError:
            pass  # Regime global not yet set (cold start) — record as decisive
        key = (filter_name, direction or "ANY", room_state)
        self._filter_block_counts[key] = self._filter_block_counts.get(key, 0) + 1
        self._last_filter_block_ts = time.time()  # Aug-12: lets the open-failed logger tell 'counted filter block' from 'real failure'
        self._last_filter_block_name = filter_name  # Aug-12 review fix: name travels with the stamp so the logger can attribute

    def _sanitize_open_kwargs(self, ef: dict, sleeve: str, direction: str) -> dict:
        """Aug-11 hardening — the unexpected-kwarg SILENT SPECIES-KILL class (3rd occurrence:
        May-5 momentum, Jul-8 FLIP dead 34 days, Aug-11 BULL/BOUNCE_LONG found pre-armed).
        The `_flip_entry_fields` dict is **-splatted into open_position(), which has no
        **kwargs — one key added without a matching param = TypeError on EVERY open,
        swallowed by the sleeve's log-only except, and the sleeve dies invisibly (zero
        fires reads as quiet tape). This guard intersects the splat against the LIVE
        signature: unknown keys are DROPPED (the trade opens) + logged ERROR + counted
        on the visible Filter Blocks surface (OPEN_KWARG_DROPPED) so the defect is a
        dashboard number the same day, not a month of log archaeology."""
        allowed = getattr(self, '_open_position_params', None)
        if allowed is None:
            import inspect
            allowed = set(inspect.signature(self.open_position).parameters)
            self._open_position_params = allowed
        unknown = [k for k in ef if k not in allowed]
        if not unknown:
            return ef
        for k in unknown:
            logger.error(f"[OPEN_KWARG_DROPPED] {sleeve}/{direction}: '{k}' is not an open_position param — "
                         f"dropped to keep the sleeve ALIVE (fix the caller; this key would have TypeError-killed every {sleeve} open)")
            self._record_filter_block("OPEN_KWARG_DROPPED", direction)
        return {k: v for k, v in ef.items() if k in allowed}

    def _record_filter_multi(self, fails, direction: str, pair: str) -> None:
        """Jul 14 FUNNEL v2: honest accounting for one momentum-ladder candidate evaluation.

        fails = the FULL list of filters that failed (legacy elif order). Increments:
        all-counts for every fail; sole-count when exactly ONE filter blocked (= remove it
        and the trade happens — the true marginal cost); episode-counts edge-triggered per
        (pair, dir, filter): +1 only when the pair NEWLY enters that blocked state (was not
        blocked by it on the previous scan cycle). Same regime-aligned suppression as the
        legacy counter so the surfaces stay comparable. In-memory only; resets on restart.
        """
        if not fails:
            return
        # Jul 15 VISIBILITY FIX: the regime-aligned suppression is REMOVED from the
        # Funnel-v2 counters — a suppressed sole block is still a real lost trade (the
        # regime "suppression" was cosmetic noise-reduction, not an actual trade veto),
        # and censoring it made countertrend filters look free. The honest table now
        # counts every direction in every regime.
        try:
            _regime = _current_btc_regime  # kept for parity of the try structure
        except NameError:
            pass
        _seq = self._funnel_scan_seq
        for f in fails:
            k = (f, direction)
            self._filter_all_counts[k] = self._filter_all_counts.get(k, 0) + 1
            _sk = (pair, direction, f)
            if self._filter_blocked_state.get(_sk) != _seq - 1:
                self._filter_episode_counts[k] = self._filter_episode_counts.get(k, 0) + 1
            self._filter_blocked_state[_sk] = _seq
        if len(fails) == 1:
            k = (fails[0], direction)
            self._filter_sole_counts[k] = self._filter_sole_counts.get(k, 0) + 1

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
                "long_supp": 0, "short_supp": 0, "any_supp": 0,
            })
            row[dir_key] += count
            # Jul 15 hotfix (review I1): SUPP gets its OWN bucket — folding it into
            # _full made regime-suppressed blocks display as "Blocked at max-5".
            suffix = "_room" if room_state == "ROOM" else ("_supp" if room_state == "SUPP" else "_full")
            row[dir_key + suffix] += count

        # Jul 14 FUNNEL v2: honest per-filter metrics from the ladder's evaluate-all pass.
        # sole = candidate would have TRADED but for this one filter (true marginal cost);
        # allf = filter failed regardless of order; episodes = edge-triggered blocked
        # episodes per (pair, dir) — de-inflated of scan-tick re-evaluation. Only momentum-
        # ladder filters populate these; engine-level gates show 0 (first-fail only).
        # Jul 14 sub-rule breakdown: v2 keys may carry a variant suffix, e.g.
        # "PAIR_RSI_ADX_CROSS[35-50:30]" — the exact composite sub-rule that
        # blocked. Parent rows aggregate across variants; ↳ child rows split them.
        def _parent(fname):
            return fname.split('[', 1)[0]

        def _dir_sum(store, fname):
            l = sum(v for (f, d), v in store.items() if d == "LONG" and _parent(f) == fname)
            s = sum(v for (f, d), v in store.items() if d == "SHORT" and _parent(f) == fname)
            return (l, s)

        def _dir_sum_exact(store, fname):
            return (store.get((fname, "LONG"), 0), store.get((fname, "SHORT"), 0))

        # Include filters that only appear in the v2 stores (a filter that always
        # fails together with an earlier ladder filter never becomes fails[0], so
        # it has no legacy row — but its AllF/Sole columns are still meaningful).
        for _store in (self._filter_all_counts, self._filter_sole_counts, self._filter_episode_counts):
            for (_fname, _d) in _store.keys():
                per_filter.setdefault(_parent(_fname), {
                    "long": 0, "short": 0, "any": 0,
                    "long_room": 0, "short_room": 0, "any_room": 0,
                    "long_full": 0, "short_full": 0, "any_full": 0,
                    "long_supp": 0, "short_supp": 0, "any_supp": 0,
                })

        rows = []
        total_long = total_short = total_any = 0
        total_long_room = total_short_room = 0
        total_room = total_full = total_supp = 0
        for filter_name, splits in per_filter.items():
            t = splits["long"] + splits["short"] + splits["any"]
            t_room = splits["long_room"] + splits["short_room"] + splits["any_room"]
            t_full = splits["long_full"] + splits["short_full"] + splits["any_full"]
            t_supp = splits.get("long_supp", 0) + splits.get("short_supp", 0) + splits.get("any_supp", 0)
            _sole_l, _sole_s = _dir_sum(self._filter_sole_counts, filter_name)
            _all_l, _all_s = _dir_sum(self._filter_all_counts, filter_name)
            _ep_l, _ep_s = _dir_sum(self._filter_episode_counts, filter_name)
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
                "total_supp": t_supp,
                "sole_long": _sole_l, "sole_short": _sole_s, "sole": _sole_l + _sole_s,
                "allf_long": _all_l, "allf_short": _all_s, "allf": _all_l + _all_s,
                "episodes_long": _ep_l, "episodes_short": _ep_s, "episodes": _ep_l + _ep_s,
            })
            total_long += splits["long"]
            total_short += splits["short"]
            total_any += splits["any"]
            total_long_room += splits["long_room"]
            total_short_room += splits["short_room"]
            total_room += t_room
            total_full += t_full
            total_supp += t_supp

        # Jul 14: rank by SOLE (true marginal cost — the decision column), tiebreak
        # by legacy Total so the zero-Sole majority keeps a stable, familiar order.
        rows.sort(key=lambda r: (r["sole"], r["total"]), reverse=True)

        # Splice ↳ variant child rows under their parent (v2 columns only; legacy
        # columns stay on the parent — the legacy counter never sees suffixes).
        # Child rows carry is_variant so footer totals / percentage sums skip them.
        _variants: Dict[str, set] = {}
        for _store in (self._filter_all_counts, self._filter_sole_counts, self._filter_episode_counts):
            for (_fname, _d) in _store.keys():
                if '[' in _fname:
                    _variants.setdefault(_parent(_fname), set()).add(_fname)
        if _variants:
            _spliced = []
            for r in rows:
                _spliced.append(r)
                for _vname in sorted(_variants.get(r["filter"], ()),
                                     key=lambda v: -(sum(_dir_sum_exact(self._filter_sole_counts, v)))):
                    _vsl, _vss = _dir_sum_exact(self._filter_sole_counts, _vname)
                    _val, _vas = _dir_sum_exact(self._filter_all_counts, _vname)
                    _vel, _ves = _dir_sum_exact(self._filter_episode_counts, _vname)
                    _spliced.append({
                        "filter": "↳ " + _vname[_vname.index('['):],  # e.g. "↳ [35-50:30]"
                        "is_variant": True,
                        "long": 0, "short": 0, "any": 0, "total": 0,
                        "long_room": 0, "short_room": 0, "any_room": 0,
                        "long_full": 0, "short_full": 0, "any_full": 0,
                        "total_room": 0, "total_full": 0,
                        "sole_long": _vsl, "sole_short": _vss, "sole": _vsl + _vss,
                        "allf_long": _val, "allf_short": _vas, "allf": _val + _vas,
                        "episodes_long": _vel, "episodes_short": _ves, "episodes": _vel + _ves,
                    })
            rows = _spliced
        return {
            "rows": rows,
            "total_long": total_long,
            "total_short": total_short,
            "total_any": total_any,
            "total": total_long + total_short + total_any,
            "total_room": total_room,
            "total_full": total_full,
            "total_supp": total_supp,
            "total_long_room": total_long_room,
            "total_short_room": total_short_room,
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

    async def _execute_bnb_sell(self, db: AsyncSession, target_usd: float, swap_type: str = "auto_sell"):
        """Sell EXCESS BNB→USDT down to target_usd (paper or live). Symmetric counterpart of
        _execute_bnb_swap (Jun 22). Reuses the proven manual-sell mechanics: logs a NEGATIVE
        amount_usdt so the reverse-derived paper balance INCREASES by the proceeds, and the
        live path calls binance_service.sell_bnb (the same call the manual-sell endpoint uses).
        Fully guarded: respects bnb_min_sell_usd, never sells below target_usd, fail-safe on
        price/API errors. Caller (bnb_scheduled_check) gates the trigger; this only executes."""
        tc = config.trading_config
        if not tc.bnb_swap_enabled or not tc.bnb_auto_sell_enabled:
            return
        min_sell = max(float(tc.bnb_min_sell_usd or 0), 5.0)  # floor mirrors the $5 buy guard

        if self.is_paper_mode:
            current_bnb = self.paper_bnb_balance_usd
            excess = current_bnb - target_usd
            if excess < min_sell:
                logger.info(f"[BNB_SELL] Skipped: excess ${excess:.2f} below ${min_sell:.2f} min")
                return

            bnb_price = await binance_service.get_bnb_price()
            if bnb_price <= 0:
                bnb_price = 600.0  # fallback for paper mode

            pre_bnb = self.paper_bnb_balance_usd
            pre_usdt = self.paper_balance
            self.paper_bnb_balance_usd -= excess
            if self.paper_bnb_balance_usd < 0:
                self.paper_bnb_balance_usd = 0

            swap_log = BnbSwapLog(
                swap_type=swap_type,
                amount_usdt=-excess,  # negative = USDT inflow (BNB → USDT)
                bnb_price=bnb_price,
                amount_bnb=round(excess / bnb_price, 6),
                pre_bnb_usd=pre_bnb,
                post_bnb_usd=self.paper_bnb_balance_usd,
                pre_usdt=pre_usdt,
                post_usdt=pre_usdt + excess,
                burn_rate=self._bnb_burn_rate,
                is_paper=True,
            )
            db.add(swap_log)
            await db.commit()

            await self._recalculate_paper_balance(db)
            await self.save_state(db)
            logger.info(
                f"[BNB_SELL] Paper {swap_type}: sold ${excess:.2f} of BNB → USDT @ ${bnb_price:.2f}. "
                f"BNB: ${pre_bnb:.2f} → ${self.paper_bnb_balance_usd:.2f}"
            )
        else:
            balance = await binance_service.get_balance()
            bnb_price = await binance_service.get_bnb_price()
            if bnb_price <= 0:
                return
            current_bnb_usd = balance['bnb_total'] * bnb_price
            excess = current_bnb_usd - target_usd
            if excess < min_sell:
                return

            result = await binance_service.sell_bnb(excess)
            if not result:
                logger.error(f"[BNB_SELL] Live {swap_type} failed — check Binance API logs")
                return

            new_balance = await binance_service.get_balance()
            swap_log = BnbSwapLog(
                swap_type=swap_type,
                amount_usdt=-result['proceeds_usdt'],  # negative = USDT inflow
                bnb_price=result['price'],
                amount_bnb=result['bnb_amount'],
                pre_bnb_usd=current_bnb_usd,
                post_bnb_usd=new_balance['bnb_total'] * result['price'],
                pre_usdt=balance['usdt_free'],
                post_usdt=new_balance['usdt_free'],
                burn_rate=self._bnb_burn_rate,
                is_paper=False,
            )
            db.add(swap_log)
            await db.commit()
            logger.info(
                f"[BNB_SELL] Live {swap_type}: sold {result['bnb_amount']:.4f} BNB "
                f"for ${result['proceeds_usdt']:.2f} @ ${result['price']:.2f}"
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
            # Jul 14: HARD FLOOR (operator, $50) — trailing burn is a rear-view mirror (observed
            # 0.32→3.16 $/hr = 10x swing): an idle stretch collapses the runway target to a few
            # dollars, then a trading burst can drain the reserve between 6h checks. The floor
            # covers the burst case; runway targeting still sizes the steady state above it.
            _bnb_floor = float(getattr(tc, 'bnb_min_balance_usd', 50.0) or 0.0)
            self._bnb_projected_need = max(self._bnb_burn_rate * tc.bnb_runway_hours, _bnb_floor)
            self._bnb_data_mature = span_24h_hours >= MIN_DATA_MATURE_HOURS
        else:
            span_24h_hours = 0
            self._bnb_burn_rate = 0
            self._bnb_projected_need = float(getattr(tc, 'bnb_min_balance_usd', 50.0) or 0.0)
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
        # Jul 14: emergency threshold floored at half the min-balance floor (can no longer
        # collapse to cents during idle stretches — it sat at $0.24 when this shipped).
        self._bnb_emergency_threshold = max(burn_rate_12h * 12.0,
                                            float(getattr(tc, 'bnb_min_balance_usd', 50.0) or 0.0) * 0.5)

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
        elif tc.bnb_auto_sell_enabled:
            # Symmetric rebalance (Jun 22): the reserve can drift ABOVE need when the 24h
            # burn rate decays after an activity slowdown (runway inflates), locking USDT out
            # of trading. Sell the excess back. CONSERVATIVE burn = max(24h, 12h) so a recent
            # pickup (rising 12h fees) keeps MORE BNB — never sell into rising activity. Ceiling
            # 48h >> the 24h buy floor and we sell DOWN TO 36h (a buffer, not the floor); the
            # wide band + 6h interval prevent buy/sell churn, and runway can't double inside one
            # 6h window so this can't fire right after a buy. Mutually exclusive with the buy (elif).
            burn_12h = (self._bnb_emergency_threshold / 12.0) if self._bnb_emergency_threshold > 0 else 0.0
            sell_burn = max(self._bnb_burn_rate, burn_12h)
            sell_runway_h = float(tc.bnb_sell_runway_hours or 0)
            if sell_burn > 0 and sell_runway_h > 0:
                sell_ceiling = sell_burn * sell_runway_h
                if current_bnb > sell_ceiling:
                    # Jul 14: auto-sell may never drain below the min-balance floor.
                    target_usd = max(sell_burn * float(tc.bnb_sell_target_hours or 0),
                                     float(getattr(tc, 'bnb_min_balance_usd', 50.0) or 0.0))
                    logger.info(
                        f"[BNB_CHECK] Reserve over-funded: BNB ${current_bnb:.2f} > ceiling "
                        f"${sell_ceiling:.2f} ({sell_runway_h:.0f}h @ ${sell_burn:.2f}/hr). "
                        f"Auto-selling down to ${target_usd:.2f} ({tc.bnb_sell_target_hours:.0f}h)."
                    )
                    await self._execute_bnb_sell(db, target_usd, swap_type="auto_sell")

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
        if tc.investment.reserve_mode == "schedule":
            # Jul 2, 2026 (operator-directed) — AUTOMATIC balance→tradeable schedule: the engine
            # walks the v3 capital-scaling table by itself (no manual milestone flips). Tier is
            # keyed on TOTAL EQUITY (free+margin, same basis as the lev schedule and the operator's
            # table) — keying on free balance would read low with margin deployed and over-reserve.
            # reserve = equity − target, so tradeable(free) = target − deployed_margin; the outer
            # max(0, free − reserve) clamps to 0 when already at/over target. Below the first tier
            # (or empty/malformed schedule) → no reserve = full balance tradeable. Fail-open.
            _sched_eq = total_portfolio if total_portfolio else available_balance
            _tgt = _lookup_leverage_schedule(
                getattr(tc.investment, 'reserve_schedule', ''), _sched_eq)
            reserve = max(0.0, _sched_eq - _tgt) if (_tgt is not None and _tgt > 0) else 0.0
        elif tc.investment.reserve_mode == "working_capital":
            # Jul 2, 2026 — capital-scaling PRIMARY knob: fix the working capital, reserve absorbs
            # ALL balance growth (tradeable = min(available, target); reserve auto-grows). Clamps
            # the max correlated-cluster loss to a fixed $ no matter how big the account gets.
            # target<=0 = inert (behaves like reserve 0). See CURRENT_STATE capital-scaling v3.
            _wct = float(getattr(tc.investment, 'working_capital_target', 0.0) or 0.0)
            reserve = max(0.0, available_balance - _wct) if _wct > 0 else 0.0
        elif tc.investment.reserve_mode == "percentage":
            reserve = available_balance * (tc.investment.reserve_percentage / 100)
        else:
            reserve = tc.investment.reserve_fixed

        # Aug 21 (operator): fee-reserve FLOOR — USDT sizing never deploys, so the BNB auto-swap can
        # always refuel (sub-$10k books have no schedule tier → went 100% into margin). Applies on
        # top of every reserve mode. 0 = off.
        _fee_res = max(0.0, float(getattr(tc.investment, 'fee_reserve_usd', 0.0) or 0.0))
        reserve += _fee_res

        # Available after reserve
        tradeable = max(0, available_balance - reserve)

        # Calculate base investment
        if tc.investment.mode == "percentage":
            investment = tradeable * (tc.investment.percentage / 100)
        elif tc.investment.mode == "equal_split":
            max_pos = tc.investment.max_open_positions or 5
            base = total_portfolio if total_portfolio else available_balance
            if tc.investment.reserve_mode == "schedule":
                # Jul 2 fix (code-review C1): this branch previously only knew percentage/fixed and
                # silently fell back to reserve_fixed in schedule mode → per-position base ignored
                # the working-capital target ((equity-500)/5 ≈ $30k instead of target/5 = $10k at
                # the $150k tier). Same target lookup as the safe-reserve calc above.
                _es_tgt = _lookup_leverage_schedule(
                    getattr(tc.investment, 'reserve_schedule', ''), base)
                reserve_from_total = max(0.0, base - _es_tgt) if (_es_tgt is not None and _es_tgt > 0) else 0.0
            elif tc.investment.reserve_mode == "working_capital":
                _es_wct = float(getattr(tc.investment, 'working_capital_target', 0.0) or 0.0)
                reserve_from_total = max(0.0, base - _es_wct) if _es_wct > 0 else 0.0
            elif tc.investment.reserve_mode == "percentage":
                reserve_from_total = base * (tc.investment.reserve_percentage / 100)
            else:
                reserve_from_total = tc.investment.reserve_fixed
            reserve_from_total += _fee_res  # fee-reserve floor rides the equal-split base too
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

        # ④ Balance→leverage schedule (Jun 26): final leverage CEILING based on current equity
        # (de-lever-as-you-grow risk knob). Applied AFTER the cell lev-mult so it bounds the
        # effective leverage. Empty schedule → None → no cap (current behavior). Fail-open.
        _sched_equity = total_portfolio if total_portfolio else available_balance
        _sched_cap = _lookup_leverage_schedule(
            getattr(tc.investment, 'leverage_balance_schedule', ''), _sched_equity)
        if _sched_cap is not None and _sched_cap >= 1 and leverage > _sched_cap:
            logger.info(f"[LEV_SCHEDULE] leverage {leverage}x -> {int(round(_sched_cap))}x "
                        f"(equity ${_sched_equity:,.0f}, schedule cap)")
            leverage = max(1, int(round(_sched_cap)))

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
            # Jul 27 review fix: was returning 5 values against the 6-tuple contract every
            # other path honors — clearing pattern_cell_rules (a documented revert action)
            # would ValueError at the caller's unpack and fail-silently block ALL opens.
            return 1.0, 1.0, None, None, None, False

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
            put('entry_btc_trend_gap_pct', lambda: g.get('_current_btc_trend_gap_pct'))  # Jul 8 — flip depth gate (flip_short_btc_trend_gap_min) surface
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
            # Jul 30 HOTFIX (review-caught containment leak): the MAJORS probe fall-through
            # keeps a BTC/ETH signal alive to the fan gate, whose block branch lands HERE —
            # and this path has no no-trade check, so a track-only major could open a
            # FULL-SIZE flip. No-trade pairs never flip, probe or not (the MAJORS contract
            # is probe-sized momentum ONLY). Belt-and-suspenders with the open_position invariant.
            _nt_flip = set(p.strip() for p in (getattr(config.trading_config, 'no_trade_pairs', '') or '').split(',') if p.strip())
            if pair in _nt_flip:
                return
            flip_dir = "SHORT" if blocked_signal == "LONG" else "LONG"
            price = indicators.get('price') if indicators else None
            if not price or price <= 0:
                return
            _ef = dict(entry_fields or {})
            # Jun 16: source-namespaced flip filter layer — veto / size / exit-mode.
            _g = globals(); _ind = indicators or {}
            # fan ratio = |EMA5-8 gap| / |EMA8-13 gap| (for the fan-spike block, all sources)
            _g58 = _ef.get('entry_ema_gap_5_8'); _g813 = _ef.get('entry_ema_gap_8_13')
            if _g58 is None and _ind.get('ema5') and _ind.get('ema8'):
                _g58 = abs((_ind['ema5'] - _ind['ema8']) / _ind['ema8'] * 100)
            if _g813 is None and _ind.get('ema8') and _ind.get('ema13'):
                _g813 = abs((_ind['ema8'] - _ind['ema13']) / _ind['ema13'] * 100)
            # BTC regime for the 2D flip-short block: prefer the recorded entry regime, else
            # classify from live globals (mirrors _seed_phantom_flip).
            _ff_reg = _ef.get('entry_btc_regime')
            if _ff_reg is None:
                try:
                    _ff_reg = classify_btc_regime(_g.get('_current_btc_adx'), _g.get('_current_btc_rsi'), _g.get('_btc_ema20_slope_pct'))
                except Exception:
                    _ff_reg = None
            _ff_in = {
                'ema5_stretch': _ef.get('entry_ema5_stretch') if _ef.get('entry_ema5_stretch') is not None else _ind.get('ema5_stretch'),
                'btc_rsi': _ef.get('entry_btc_rsi') if _ef.get('entry_btc_rsi') is not None else _g.get('_current_btc_rsi'),
                'btc_rsi_prev6': _ef.get('entry_btc_rsi_prev6') if _ef.get('entry_btc_rsi_prev6') is not None else _g.get('_current_btc_rsi_prev6'),
                'btc_adx': _ef.get('entry_btc_adx') if _ef.get('entry_btc_adx') is not None else _g.get('_current_btc_adx'),
                'fan_ratio': (abs(_g58 / _g813) if (_g58 is not None and _g813) else None),
                'pair_rsi': _ef.get('entry_rsi') if _ef.get('entry_rsi') is not None else _ind.get('rsi'),
                'flip_dir': flip_dir,
                'adx_delta': _ef.get('entry_adx_delta') if _ef.get('entry_adx_delta') is not None else _ind.get('adx_delta'),
                # pair ADX — required by the PAIR_RSI_OB ADX floor (flip_pair_rsi_ob_adx_min); was
                # missing, so ind.get('adx') returned None and the floor never fired (Jun 20 fix).
                'adx': _ef.get('entry_adx') if _ef.get('entry_adx') is not None else _ind.get('adx'),
                # BTC ATR% — required by U3 (flip_fan_btc_atr_min), the FAN sub-0.10 weekend block.
                'btc_atr_pct': _ef.get('entry_btc_atr_pct') if _ef.get('entry_btc_atr_pct') is not None else _g.get('_current_btc_atr_pct'),
                'btc_regime': _ff_reg,
                'atr_pct': (_ef.get('entry_atr_pct') if _ef.get('entry_atr_pct') is not None
                            else (round(_ind['atr'] / price * 100, 4) if (_ind.get('atr') and price) else _ind.get('atr_pct'))),
                # pair EMA13-EMA50 gap% — required by the flip-SHORT parabola block (flip_short_pair_gap_max).
                # Same (EMA13-EMA50)/EMA50 the entry feature stamps; fall back to ind EMAs (Jun 21).
                'pair_gap': (_ef.get('entry_pair_ema20_ema50_gap_pct') if _ef.get('entry_pair_ema20_ema50_gap_pct') is not None
                             else (round((_ind['ema13'] - _ind['ema50']) / _ind['ema50'] * 100, 4)
                                   if (_ind.get('ema13') is not None and _ind.get('ema50')) else None)),
                # entry quality score — required by the flip-SHORT quality floor (flip_short_quality_min). Jun 25.
                'quality_score': _ef.get('entry_quality_score'),
                # pair −DI (downward directional movement) — required by the NEGDI15 sellers-present
                # multiplier cell (flip_short_negdi_mult). Jul 10.
                'neg_di': _ef.get('entry_neg_di') if _ef.get('entry_neg_di') is not None else _ind.get('neg_di'),
                # market breadth + range position — required by the FAN flip-SHORT winner cell
                # (flip_fan_qs_cell), applied below. Jun 26.
                'bear_pct': _ef.get('entry_bear_pct'),
                'range_position': _ef.get('entry_range_position'),
                # BTC 1h EMA20 slope — required by the flip-SHORT regime gate (flip_short_btc_1h_slope_max). Jul 3.
                'btc_1h_slope': (_ef.get('entry_btc_1h_slope') if _ef.get('entry_btc_1h_slope') is not None
                                 else _g.get('_current_btc_1h_slope')),
                # BTC EMA13-50 trend gap — required by the flip-SHORT depth gate (flip_short_btc_trend_gap_min). Jul 8.
                'btc_trend_gap': (_ef.get('entry_btc_trend_gap_pct') if _ef.get('entry_btc_trend_gap_pct') is not None
                                  else _g.get('_current_btc_trend_gap_pct')),
                # BTC distance from its 5m EMA13 — required by the FAN-gate bearish-BTC filter (flip_fan_btc_ema13_max). Aug 23.
                'btc_dist_ema13': (_ef.get('entry_btc_dist_from_ema13_pct') if _ef.get('entry_btc_dist_from_ema13_pct') is not None
                                   else ((float(_g.get('_current_btc_price')) - float(_g.get('_current_btc_ema13'))) / float(_g.get('_current_btc_ema13')) * 100.0
                                         if (_g.get('_current_btc_price') and _g.get('_current_btc_ema13')) else None)),
            }
            _blocked, _reason, _flip_cell_mult, _flip_cell_lev_mult, _flip_exit_mode, _flip_fails = _flip_filters(source, _ff_in)
            # Jul 17 FUNNEL v2 for the flip chain: feed the FULL fail list to the Sole/Epis/AllF
            # recorder (same stores as the momentum ladder → same table rows on every surface).
            # Sole = the one gate whose removal alone admits the flip (the marginal blocker).
            try: self._record_filter_multi(_flip_fails, flip_dir, pair)
            except Exception: pass
            # Jul 29 FLIPGATE PROBES (operator-directed after the down-window forensics): flip
            # candidates DO exist inside BTC down-windows (313 in-window vetoes/7d) but die on
            # three June-fitted secondary floors (QUALITY 85 / RSI_MIN 72 / TRENDGAP 54 = 67%
            # of in-window kills; TRENDGAP's registered Sole-growth reopen trigger has fired).
            # A SHORT candidate sole-blocked by exactly ONE probe-listed gate opens at gap-probe
            # sizing (~1× eff) under its own FGP_* cell tag — per-gate cohorts, per-gate locked
            # verdicts at N≥30 (CURRENT_STATE). Fail-silent: any error → normal block path.
            _fg_admit_tag = None
            if _blocked:
                try:
                    _fg_th = config.trading_config.thresholds
                    if (getattr(_fg_th, 'flipgate_probe_enabled', False) and flip_dir == 'SHORT'
                            and isinstance(_flip_fails, list) and len(_flip_fails) == 1):
                        _fg_map = {'FLIP_SHORT_QUALITY': 'FGP_QS', 'FLIP_SHORT_RSI_MIN': 'FGP_RSI',
                                   'FLIP_SHORT_BTC_TRENDGAP': 'FGP_TG'}
                        _fg_gates = {s.strip() for s in (getattr(_fg_th, 'flipgate_probe_gates', '') or '').split(',') if s.strip()}
                        if _flip_fails[0] in _fg_gates and _flip_fails[0] in _fg_map and db is not None:
                            _fg_q = await db.execute(
                                select(func.count(Order.id)).where(and_(
                                    Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                                    Order.cell_multiplier_source.like('%FGP_%'))))
                            if (_fg_q.scalar() or 0) < int(getattr(_fg_th, 'flipgate_probe_max_open', 3) or 0):
                                _fg_admit_tag = _fg_map[_flip_fails[0]]
                except Exception as _fg_e:
                    # Jul 30 (zero-fire trace): the admit check stays FAIL-OPEN (normal block
                    # path), but the error is no longer swallowed silently — 3 TRENDGAP soles
                    # accrued post-deploy with 0 FGP fires and the logs couldn't say why.
                    logger.warning(f"[FLIPGATE_PROBE] {pair}: admit check errored (fail-open to normal block): {_fg_e}")
                    _fg_admit_tag = None
            if _blocked and _fg_admit_tag is None:
                try: self._record_filter_block(_reason, flip_dir)
                except Exception: pass
                # Jul 5: same-direction PASS phantom for the decision-gated flip-SHORT blocker.
                # BTC1H_SLOPE = the Jul-3 gate's locked revert surface (≥60% WR on N≥10 blocked → gate
                # off) — the gate shipped without it, so the revert could never fire. The phantom runs
                # the exact flip replica exit, so its WR is directly gate-comparable.
                # Jul 23 (phantom-slot review): REGIME and TRENDGAP seeding RETIRED — a phantom source
                # only holds a slot while an armed gate consumes it. REGIME's relaxation discussion
                # closed Jul-13 (0/130 net-admissible) and its flow died with the bull regime;
                # TRENDGAP's revert fired-but-MOOT (Funnel v2 Sole=0 — the per-veto phantom WR is a
                # first-block mirage; the revisit trigger is the funnel's Sole count, not phantoms).
                if _reason == "FLIP_SHORT_BTC1H_SLOPE":
                    _seed_phantom_flip(pair, price, flip_dir, f"PASS:{_reason}",
                                       entry_fields=_ef, mode='PASS')
                logger.info(f"[FLIP_FILTER] {pair}: {source} flip vetoed by {_reason} "
                            f"(stretch={_ff_in.get('ema5_stretch')}, btcRSI={_ff_in.get('btc_rsi')}, btcADX={_ff_in.get('btc_adx')})")
                return
            if _fg_admit_tag is not None:
                logger.info(f"[FLIPGATE_PROBE] {pair}: sole-blocked by {_reason} → probe-admitted as {_fg_admit_tag} (gap-probe sizing)")
            # FAN flip-SHORT winner cell (Jun 26): qs≥3 × bear≥70 × range 60-90. SHORT flips only;
            # applies size/lev (default 1×=inert) AND a distinct cell tag so the cohort tracks as its
            # own row in Multiplier Cell Performance. Non-FAN sources / flip-LONGs leave it untouched.
            _flip_cell_tag = None
            if source == "FAN_RATIO_GATE" and flip_dir == "SHORT":
                _cs, _cl, _ct = _fan_qs_cell_match(config.trading_config.thresholds, _ff_in.get('quality_score'),
                                                   _ff_in.get('bear_pct'), _ff_in.get('range_position'))
                if _ct:
                    _flip_cell_mult, _flip_cell_lev_mult, _flip_cell_tag = _cs, _cl, _ct
            # Jul 8 — TG_SHALLOW multiplier cell (operator-directed 2× ship, same session as the
            # trend-gap depth gate): BTC EMA13-50 gap in [shallow_min, 0) = the flip sleeve's edge
            # core (baseline 8·100%·+0.53·+$455 + fresh 2·100% = 10/10 combined; the monotone
            # gradient's top bucket). Invest mult only (lev stays 1×). ⚠ DOUBLE OVERRIDE
            # acknowledged: N=10 ≪ 30 promotion gate AND skips the locked 1.5×-first staging
            # (operator: sleeve must scale; testing the confluence now). 🔒 TIGHT REVERT (verdict
            # machinery): ✗ HARMFUL (net-negative on N≥5 fresh fires) → mult 1.0; ⚠ DRAG → 1.5×.
            try:
                _tgm_raw = getattr(config.trading_config.thresholds, 'flip_short_tg_shallow_mult', 0.0)
                _tgm = 0.0 if _tgm_raw is None else float(_tgm_raw)
                _tgz_raw = getattr(config.trading_config.thresholds, 'flip_short_tg_shallow_min', -0.10)
                _tgz = -0.10 if _tgz_raw is None else float(_tgz_raw)
                _tgmax_raw = getattr(config.trading_config.thresholds, 'flip_short_tg_shallow_max', 0.0)
                _tgmax = 0.0 if _tgmax_raw is None else float(_tgmax_raw)
                _tglev_raw = getattr(config.trading_config.thresholds, 'flip_short_tg_shallow_lev_mult', 1.0)
                _tglev = 1.0 if _tglev_raw is None else float(_tglev_raw)
                _btg2 = _ff_in.get('btc_trend_gap')
                if (flip_dir == 'SHORT' and _tgm > 1.0 and _btg2 is not None
                        and _tgz <= _btg2 < _tgmax):
                    _flip_cell_mult = max(_flip_cell_mult or 1.0, _tgm)
                    if _tglev > 1.0:
                        _flip_cell_lev_mult = max(_flip_cell_lev_mult or 1.0, _tglev)
                    _flip_cell_tag = (_flip_cell_tag + "+[TG_SHALLOW]") if _flip_cell_tag else "[TG_SHALLOW]"
            except Exception:
                pass
            # Jul 10 — NEGDI15 "sellers-present" multiplier cell (operator-directed 2× ship): pair
            # −DI ≥ min at entry = sellers already active in the faded pair → the fade has fuel.
            # Baseline cell: 17·100%WR·+$971 (~+0.4%/trade) over 13 dates / 15 pairs, era-consistent
            # (12/12 pre-06-30, 5/5 post); all 6 sleeve losers sit below −DI 15. NOT a filter — the
            # <min flank is a 57%-WR mixed cohort (locked rule: multiply winners, don't block them).
            # ⚠ DOUBLE OVERRIDE acknowledged: N=17 < 30 W-gate AND skips the locked 1.5×-first
            # staging (operator call; TG_SHALLOW precedent). 🔒 TIGHT REVERT (verdict machinery):
            # ✗ HARMFUL (net-negative on N≥5 fresh fires) → 1.0× · ⚠ DRAG (Δ$ vs BL <−$1) → 1.5×.
            try:
                _ndm_raw = getattr(config.trading_config.thresholds, 'flip_short_negdi_mult', 0.0)
                _ndm = 0.0 if _ndm_raw is None else float(_ndm_raw)
                _ndmin_raw = getattr(config.trading_config.thresholds, 'flip_short_negdi_min', 15.0)
                _ndmin = 15.0 if _ndmin_raw is None else float(_ndmin_raw)
                _ndlev_raw = getattr(config.trading_config.thresholds, 'flip_short_negdi_lev_mult', 1.0)
                _ndlev = 1.0 if _ndlev_raw is None else float(_ndlev_raw)
                _nd2 = _ff_in.get('neg_di')
                if (flip_dir == 'SHORT' and _ndm > 1.0 and _nd2 is not None and _nd2 >= _ndmin):
                    _flip_cell_mult = max(_flip_cell_mult or 1.0, _ndm)
                    if _ndlev > 1.0:
                        _flip_cell_lev_mult = max(_flip_cell_lev_mult or 1.0, _ndlev)
                    _flip_cell_tag = (_flip_cell_tag + "+[NEGDI15]") if _flip_cell_tag else "[NEGDI15]"
            except Exception:
                pass
            # Jul 6 — B1H_SLOPEUP admit cohort (the fired revert gate's graduated live test):
            # slope>0 flip-shorts trade at cell mult CAPPED at admit_mult (ship 1.0) under their
            # own tag so Multiplier Cell Perf + Flip×Regime track them as a distinct cell.
            # 🔒 RE-BLOCK (admit_mult→0) if ≤50% WR or net-negative on N≥10; PROMOTE at ≥65%/avg≥+0.15/N≥20.
            try:
                _b1x = getattr(config.trading_config.thresholds, 'flip_short_btc_1h_slope_max', 99.0)
                _b1x = 99.0 if _b1x is None else float(_b1x)
                _adm2 = getattr(config.trading_config.thresholds, 'flip_short_btc_1h_slope_admit_mult', 0.0)
                _adm2 = 0.0 if _adm2 is None else float(_adm2)
                _s1h2 = _ff_in.get('btc_1h_slope')
                if (flip_dir == 'SHORT' and _b1x < 99 and _adm2 > 0
                        and _s1h2 is not None and _s1h2 > _b1x):
                    _flip_cell_mult = min(_flip_cell_mult or 1.0, _adm2)
                    _flip_cell_tag = (_flip_cell_tag + "+B1H_SLOPEUP") if _flip_cell_tag else "B1H_SLOPEUP"
            except Exception:
                pass
            # Jul 29 FLIPGATE probe sizing — ABSOLUTE assign, placed AFTER the multiplier-cell
            # blocks (they use max() and would re-inflate an observation-sized probe back to
            # 1-2×). Probe rides the normal flip exit stack; only size/lev/tag are overridden.
            if _fg_admit_tag is not None:
                _fg_th2 = config.trading_config.thresholds
                _flip_cell_mult = float(getattr(_fg_th2, 'gap_probe_invest_mult', 0.5) or 0.5)
                _flip_cell_lev_mult = float(getattr(_fg_th2, 'gap_probe_lev_mult', 0.05) or 0.05)
                _flip_cell_tag = "+" + _fg_admit_tag
            async def _open(_db):
                # Aug-11 CRITICAL FIX: open_position has NO entry_btc_trend_gap_pct param (it
                # stamps the Order internally from the same global) — the Jul-8 depth-gate ship
                # added the key to _ef, and the **_ef splat made EVERY flip open raise
                # TypeError since Jul-8 10:40 (last flip ever: Jul-8 07:41 — the sleeve was
                # silently dead through all of B1+B2). Pop it before the splat; the key must
                # STAY in _ef (the flip-filters consumer at ~3800 reads it).
                _ef.pop('entry_btc_trend_gap_pct', None)
                return await self.open_position(
                    db=_db, pair=pair, direction=flip_dir, confidence="STRONG_BUY",
                    current_price=price,
                    entry_rsi=_ef.pop('entry_rsi', None) or indicators.get('rsi'),
                    entry_adx=_ef.pop('entry_adx', None) or indicators.get('adx'),
                    entry_atr_pct=_pop_or(_ef, 'entry_atr_pct', _ind_atr_pct(indicators)),
                    flip_source=source, flip_cell_mult=_flip_cell_mult, flip_cell_lev_mult=_flip_cell_lev_mult, flip_exit_mode=_flip_exit_mode,
                    flip_cell_tag=_flip_cell_tag,
                    **self._sanitize_open_kwargs(_ef, 'FLIP', flip_dir),
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
            self._record_filter_block("OPEN_FAILED_FLIP", flip_dir if 'flip_dir' in dir() else "ANY")  # Aug-11: dashboard-visible sleeve-death counter

    async def _maybe_open_bull_long(self, db, pair, indicators, isolate=False, entry_fields=None):
        """Bull-Long Entry trigger (Jun 18) — the BUILD-side twin of the flip sleeve. When a
        LONG PASSES the fan gate (low fan ratio) in an allowed bull regime, open the SAME
        direction as a REAL momentum long (NOT a fade) and let it ride the NORMAL long exit
        stack. Tagged entry_strategy="BULL_LONG" via open_position(bull_long=True); it is NOT
        _is_flip so it flows through per-level trailing / ATR-widened SL like any long. Sizes at
        base × bull_long_size_mult × bull_long_lev_mult (both default 1.0 = no amplification,
        normal leverage). Fail-silent + isolatable so a bull-long bug can NEVER break the scan
        or the monitor loop (mirrors _maybe_open_flip). De-duped per pair/30min via the same
        _PFLIP_COOLDOWN infra (key "pair|BULL_LONG") so a pair sitting in the zone across scan
        cycles opens at most one bull-long per cooldown window. All hard risk controls
        (max-open, existing-position, cooldown, liquidity caps) are enforced inside
        open_position. TO REMOVE: grep "BULL_LONG" / "bull_long" / "_maybe_open_bull_long"."""
        try:
            _th = config.trading_config.thresholds
            if not getattr(_th, 'bull_long_enabled', False):
                return
            # Jul 30 HOTFIX (same containment class as the flip guard): no-trade pairs never
            # open bull-longs — latent today (sleeve disabled) but the MAJORS fall-through
            # makes this path reachable for BTC/ETH when re-enabled.
            _nt_bl = set(p.strip() for p in (getattr(config.trading_config, 'no_trade_pairs', '') or '').split(',') if p.strip())
            if pair in _nt_bl:
                return
            price = indicators.get('price') if indicators else None
            if not price or price <= 0:
                return
            _ef = dict(entry_fields or {})
            _ind = indicators or {}
            # fan ratio = |EMA5-8 gap| / |EMA8-13 gap| (mirror _maybe_open_flip's computation)
            _g58 = _ef.get('entry_ema_gap_5_8'); _g813 = _ef.get('entry_ema_gap_8_13')
            if _g58 is None and _ind.get('ema5') and _ind.get('ema8'):
                _g58 = abs((_ind['ema5'] - _ind['ema8']) / _ind['ema8'] * 100)
            if _g813 is None and _ind.get('ema8') and _ind.get('ema13'):
                _g813 = abs((_ind['ema8'] - _ind['ema13']) / _ind['ema13'] * 100)
            _fan_ratio = (abs(_g58 / _g813) if (_g58 is not None and _g813) else None)
            if _fan_ratio is None:
                return
            # BTC regime FIRST (the per-regime fan window needs it) — prefer the recorded entry regime,
            # else classify from live globals.
            _g = globals()
            _reg = _ef.get('entry_btc_regime')
            if _reg is None:
                try:
                    _reg = classify_btc_regime(_g.get('_current_btc_adx'), _g.get('_current_btc_rsi'), _g.get('_btc_ema20_slope_pct'))
                except Exception:
                    _reg = None
            _allowed = {r.strip().upper() for r in (getattr(_th, 'bull_long_regimes', '') or '').split(',') if r.strip()}
            _rkey = (_reg or '').upper()
            if not _allowed or _rkey not in _allowed:
                return
            # Stamp the GATED regime onto the order so it records exactly what the sleeve admitted on
            # (else open_position re-classifies and can land on null/NEUTRAL → the bull-long drops out of
            # the regime tables, the live gate surface). Jun 18 bugfix.
            _ef['entry_btc_regime'] = _reg
            # Fan window: PER-REGIME override (bull_long_fan_by_regime = "REGIME:min-max,...") if the gated
            # regime is mapped, else fall back to the global bull_long_fan_min/max. Low-fan (flat/decelerating)
            # build-side longs bleed; the winning band is regime-specific (Jun 24 cross-batch: S.BULL 1.35-2.0,
            # H.BULL 2.0-3.0). 0 = that bound disabled.
            _fan_min = getattr(_th, 'bull_long_fan_min', 0.0) or 0.0
            _fan_max = getattr(_th, 'bull_long_fan_max', 0.0) or 0.0
            _by_reg = getattr(_th, 'bull_long_fan_by_regime', '') or ''
            if _by_reg.strip():
                for _tok in _by_reg.split(','):
                    _tok = _tok.strip()
                    if not _tok or ':' not in _tok:
                        continue
                    _rk, _, _rv = _tok.partition(':')
                    if _rk.strip().upper() == _rkey:
                        try:
                            _lo, _, _hi = _rv.strip().partition('-')
                            _fan_min, _fan_max = float(_lo), float(_hi)
                        except Exception:
                            pass
                        break
            if _fan_min > 0 and _fan_ratio < _fan_min:
                return
            if _fan_max > 0 and _fan_ratio >= _fan_max:
                return
            # De-dupe per pair / cooldown window (reuse the phantom-flip cooldown infra).
            _ck = f"{pair}|BULL_LONG"
            _now = _leash_time.time()
            if _now - _PFLIP_COOLDOWN.get(_ck, 0) < _PFLIP_COOLDOWN_MIN * 60:
                return
            # Per-sleeve concurrency cap (Jun 23): reserve book slots for higher-conviction
            # sleeves (MOMENTUM / UNMATCHED longs). BULL_LONG is a 1×-obs sleeve — a bull cluster
            # must not monopolize the max-open book and crowd out proven longs. 0 = uncapped.
            _bl_cap = int(getattr(_th, 'bull_long_max_concurrent', 0) or 0)
            if _bl_cap > 0:
                _bl_open = (await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN",
                        Order.is_paper == self.is_paper_mode,
                        Order.entry_strategy == "BULL_LONG",
                    ))
                )).scalar() or 0
                if _bl_open >= _bl_cap:
                    self._record_filter_block("BULL_LONG_MAX", "LONG")
                    return
            _size_mult = getattr(_th, 'bull_long_size_mult', 1.0) or 1.0
            _lev_mult = getattr(_th, 'bull_long_lev_mult', 1.0) or 1.0
            async def _open(_db):
                _ef.pop('entry_btc_trend_gap_pct', None)  # Aug-11: not an open_position param (stamped internally from globals) — same poison that killed FLIP for 34 days; sanitizer below catches any future stragglers
                return await self.open_position(
                    db=_db, pair=pair, direction="LONG", confidence="STRONG_BUY",
                    current_price=price,
                    entry_rsi=_ef.pop('entry_rsi', None) or indicators.get('rsi'),
                    entry_adx=_ef.pop('entry_adx', None) or indicators.get('adx'),
                    entry_atr_pct=_pop_or(_ef, 'entry_atr_pct', _ind_atr_pct(indicators)),
                    bull_long=True, bull_long_size_mult=_size_mult, bull_long_lev_mult=_lev_mult,
                    **self._sanitize_open_kwargs(_ef, 'BULL_LONG', 'LONG'),
                )
            if isolate:
                async with AsyncSessionLocal() as _bdb:
                    order = await _open(_bdb)
            else:
                order = await _open(db)
            if order:
                # only stamp the cooldown on a REAL open (so a max-open skip can retry next cycle)
                _PFLIP_COOLDOWN[_ck] = _now
                logger.info(f"[BULL_LONG] {pair}: opened bull-long (fan={_fan_ratio:.2f} regime={_reg} id={order.id})")
            return order  # caller uses this to PRE-EMPT the flip-fade (don't open opposite positions)
        except Exception as e:
            logger.error(f"[BULL_LONG] {pair}: bull-long open failed: {e}")
            self._record_filter_block("OPEN_FAILED_BULL_LONG", "LONG")  # Aug-11: dashboard-visible sleeve-death counter
            return None

    async def _bullrun_persist_period(self, db, new_state, rd):
        """🌊 Periods ledger writer (gate 57 episode table). Restart-proof: adopts the open DB
        row when its state matches the recomputed one (no fake period); closes it with
        ended_by='restart→X' otherwise. Running fields (r72/above/eff_end, btc_end, peaks,
        r6_min, blocked counters) refresh every compute so the OPEN row is always current.
        Sets _bullrun_monitor['resumed']=True on adoption so the caller skips the flip log.
        Never raises — persistence failure must not touch the trading monitor."""
        if db is None:
            return
        try:
            now = datetime.utcnow()
            pid = _bullrun_monitor.get('period_id')
            cur = None
            if pid:
                cur = (await db.execute(select(MonitorPeriod).where(MonitorPeriod.id == pid))).scalar_one_or_none()
            if cur is None:
                cur = (await db.execute(
                    select(MonitorPeriod).where(MonitorPeriod.ended_at.is_(None)).order_by(MonitorPeriod.started_at.desc()).limit(1)
                )).scalar_one_or_none()
                if cur is not None:
                    # legacy rows (pre-hotfix, last_update NULL) are adopted — never split an episode on the deploy that adds the column
                    _stale_min = 0.0 if cur.last_update is None else (now - cur.last_update).total_seconds() / 60
                    if cur.state == new_state and _stale_min <= 60:
                        _bullrun_monitor['period_id'] = cur.id
                        _bullrun_monitor['resumed'] = True
                        _bullrun_monitor['green_since'] = cur.started_at.strftime('%Y-%m-%d %H:%M') if new_state == 'GREEN' else None
                        for _k, _col in (('blk_spacing', 'blocked_spacing'), ('blk_slots', 'blocked_slots'), ('blk_ema50', 'blocked_ema50'), ('blk_off24h', 'blocked_off24h'), ('blk_ema13', 'blocked_ema13'), ('blk_1h', 'blocked_1h')):
                            _bullrun_monitor[_k] = int(getattr(cur, _col, 0) or 0)
                        logger.info(f"[BULLRUN_MONITOR] resumed open {cur.state} period #{cur.id} (since {cur.started_at:%Y-%m-%d %H:%M} UTC) after restart")
                    else:
                        # stale (>60 min without a running update = bot was down) or state changed across the restart
                        cur.ended_at = (cur.last_update or now) if _stale_min > 60 else now
                        if cur.state == 'GREEN':  # Aug-23 (24) review: the ledger close is the authoritative GREEN-end stamp (covers live, restart, downtime)
                            _bullrun_monitor['green_end_ts'] = cur.ended_at.replace(tzinfo=timezone.utc).timestamp()
                        cur.ended_by = f"restart→{new_state}" if _stale_min <= 60 else f"downtime→{new_state}"
                        cur.r72_end, cur.above_end, cur.eff_end, cur.btc_end = rd['r72'], rd['above'], rd['eff'], rd['px']
                        cur = None
            if cur is not None and cur.state == new_state:
                # Bootstrap honesty (idempotent): the ledger's OLDEST row, if GREEN, starts no later than the
                # first sleeve fill (a sleeve can only fill while GREEN). Ship day: the table was born 18:29 UTC
                # mid-episode while the first fills were 17:06 → row #1 is corrected once; never fires again.
                try:
                    _min_id = (await db.execute(select(func.min(MonitorPeriod.id)))).scalar()
                    if cur.state == 'GREEN' and _min_id is not None and cur.id == _min_id:
                        _first_fill = (await db.execute(
                            select(func.min(Order.opened_at)).where(and_(Order.entry_strategy == 'BULLRUN_LONG', Order.is_paper == self.is_paper_mode))
                        )).scalar()
                        if _first_fill is not None and _first_fill < cur.started_at:
                            logger.info(f"[BULLRUN_MONITOR] period #{cur.id} started_at {cur.started_at:%H:%M} → backdated to first sleeve fill {_first_fill:%Y-%m-%d %H:%M} UTC (one-time bootstrap correction)")
                            cur.started_at = _first_fill
                            _bullrun_monitor['green_since'] = _first_fill.strftime('%Y-%m-%d %H:%M')
                except Exception as _bd2_err:
                    logger.debug(f"[BULLRUN_MONITOR] oldest-row backdate check skipped: {_bd2_err}")
                cur.r72_end, cur.above_end, cur.eff_end, cur.btc_end = rd['r72'], rd['above'], rd['eff'], rd['px']
                cur.eff_peak = max(cur.eff_peak if cur.eff_peak is not None else rd['eff'], rd['eff'])
                cur.r72_peak = max(cur.r72_peak if cur.r72_peak is not None else rd['r72'], rd['r72'])
                cur.r6_min = min(cur.r6_min if cur.r6_min is not None else rd['r6'], rd['r6'])
                cur.blocked_spacing = int(_bullrun_monitor.get('blk_spacing', 0) or 0)
                cur.blocked_slots = int(_bullrun_monitor.get('blk_slots', 0) or 0)
                cur.blocked_ema50 = int(_bullrun_monitor.get('blk_ema50', 0) or 0)
                cur.blocked_off24h = int(_bullrun_monitor.get('blk_off24h', 0) or 0)
                cur.blocked_ema13 = int(_bullrun_monitor.get('blk_ema13', 0) or 0)
                cur.blocked_1h = int(_bullrun_monitor.get('blk_1h', 0) or 0)
                cur.last_update = now
                # breadth backfill: the first compute after boot can precede the breadth scan (0/0)
                _gb = globals()
                if not cur.bull_pct_start and not cur.bear_pct_start and (_gb.get('_market_bull_pct') or _gb.get('_market_bear_pct')):
                    cur.bull_pct_start = _gb.get('_market_bull_pct'); cur.bear_pct_start = _gb.get('_market_bear_pct')
                _bullrun_monitor['period_id'] = cur.id
            else:
                amber_lead = None
                if cur is not None:
                    cur.ended_at = now
                    if cur.state == 'GREEN':  # Aug-23 (24) review: the ledger close is the authoritative GREEN-end stamp (covers live, restart, downtime)
                        _bullrun_monitor['green_end_ts'] = cur.ended_at.replace(tzinfo=timezone.utc).timestamp()
                    cur.r72_end, cur.above_end, cur.eff_end, cur.btc_end = rd['r72'], rd['above'], rd['eff'], rd['px']
                    cur.ended_by = 'latch' if rd['latch'] else ('stay-band' if cur.state == 'GREEN' else f"→{new_state}")
                    cur.blocked_spacing = int(_bullrun_monitor.get('blk_spacing', 0) or 0)
                    cur.blocked_slots = int(_bullrun_monitor.get('blk_slots', 0) or 0)
                    cur.blocked_ema50 = int(_bullrun_monitor.get('blk_ema50', 0) or 0)
                    cur.blocked_off24h = int(_bullrun_monitor.get('blk_off24h', 0) or 0)
                    cur.blocked_ema13 = int(_bullrun_monitor.get('blk_ema13', 0) or 0)
                    cur.blocked_1h = int(_bullrun_monitor.get('blk_1h', 0) or 0)
                    if cur.state == 'AMBER' and new_state == 'GREEN':
                        amber_lead = int((now - cur.started_at).total_seconds() / 60)
                _g = globals()
                _start = now
                # Aug-23 (16) review: with the efficiency band at 0.10 = 0.10 a GREEN can flap on a 10-min tick.
                # A GREEN row that closed ≤30 min ago on the stay band is REOPENED (counters kept) instead of
                # inserting a fragment — keeps the episode a single ledger unit (WINDOW-UNITS rule).
                if new_state == 'GREEN':
                    try:
                        _prev = (await db.execute(
                            select(MonitorPeriod).where(and_(
                                MonitorPeriod.state == 'GREEN', MonitorPeriod.ended_at.isnot(None),
                                MonitorPeriod.ended_at >= now - timedelta(minutes=30),
                            )).order_by(MonitorPeriod.ended_at.desc()).limit(1)
                        )).scalars().first()
                        if _prev is not None and 'stay' in str(_prev.ended_by or ''):
                            _prev.ended_at = None; _prev.ended_by = None; _prev.last_update = now
                            _prev.r72_end, _prev.above_end, _prev.eff_end, _prev.btc_end = rd['r72'], rd['above'], rd['eff'], rd['px']
                            _bullrun_monitor['period_id'] = _prev.id
                            for _k, _col in (('blk_spacing', 'blocked_spacing'), ('blk_slots', 'blocked_slots'), ('blk_ema50', 'blocked_ema50'), ('blk_off24h', 'blocked_off24h'), ('blk_ema13', 'blocked_ema13'), ('blk_1h', 'blocked_1h')):
                                _bullrun_monitor[_k] = int(getattr(_prev, _col, 0) or 0)
                            logger.info(f"[BULLRUN_MONITOR] GREEN re-armed within 30 min of a stay-band drop — period #{_prev.id} reopened (flap, not a new episode)")
                            await db.commit()
                            return
                    except Exception as _rm_err:
                        logger.warning(f"[BULLRUN_MONITOR] re-arm merge check failed ({_rm_err}) — opening a new period")
                if cur is None and new_state == 'GREEN':
                    # Bootstrap honesty: with NO period history at all, a GREEN first row starts at the
                    # earliest sleeve fill still in the DB (a sleeve can only fill while GREEN) — the
                    # ledger shipped mid-episode (Aug-21: fire 17:05 UTC, ledger deploy 18:29 UTC).
                    try:
                        _any_p = (await db.execute(select(func.count(MonitorPeriod.id)))).scalar() or 0
                        if _any_p == 0:
                            _first_fill = (await db.execute(
                                select(func.min(Order.opened_at)).where(and_(Order.entry_strategy == 'BULLRUN_LONG', Order.is_paper == self.is_paper_mode))
                            )).scalar()
                            if _first_fill is not None and _first_fill < now:
                                _start = _first_fill
                                logger.info(f"[BULLRUN_MONITOR] first-ever GREEN period backdated to earliest sleeve fill {_first_fill:%Y-%m-%d %H:%M} UTC")
                    except Exception as _bd_err:
                        logger.warning(f"[BULLRUN_MONITOR] backdate check failed: {_bd_err}")
                newp = MonitorPeriod(
                    state=new_state, started_at=_start, last_update=now,
                    r72_start=rd['r72'], above_start=rd['above'], eff_start=rd['eff'], r6_start=rd['r6'],
                    r72_end=rd['r72'], above_end=rd['above'], eff_end=rd['eff'],
                    r72_peak=rd['r72'], eff_peak=rd['eff'], r6_min=rd['r6'],
                    btc_start=rd['px'], btc_end=rd['px'],
                    bull_pct_start=_g.get('_market_bull_pct'), bear_pct_start=_g.get('_market_bear_pct'),
                    amber_lead_min=amber_lead, blocked_spacing=0, blocked_slots=0, blocked_ema50=0, blocked_off24h=0, blocked_ema13=0, blocked_1h=0,
                )
                db.add(newp)
                await db.flush()
                _bullrun_monitor['period_id'] = newp.id
                for _k in ('blk_spacing', 'blk_slots', 'blk_ema50', 'blk_off24h', 'blk_ema13', 'blk_1h'):
                    _bullrun_monitor[_k] = 0
            await db.commit()
        except Exception as e:
            logger.error(f"[BULLRUN_MONITOR] period persistence failed: {e}")
            try:
                await db.rollback()
            except Exception:
                pass

    async def _update_bullrun_monitor(self, db=None):
        """🌊 Aug-21 gate 57: Bull-Run Monitor — hourly-class composite on BTC 5m (Schmitt band
        + crash-latch + AMBER). Throttled to one recompute per 10 min; updated_at is stamped
        ONLY on success so a failed fetch retries next cycle and the 30-min staleness gate in
        the sleeve entry fails safe (stale ≠ GREEN). Periods ledger persisted via
        _bullrun_persist_period (restart-proof). Evidence: config.py bullrun_* block."""
        global _bullrun_monitor
        th = config.trading_config.thresholds
        _now = _leash_time.time()
        if _now - (_bullrun_monitor.get('updated_at') or 0) < 600:
            return
        try:
            k5 = await binance_service.get_ohlcv('BTC/USDT:USDT', '5m', 1000)
            if not k5 or len(k5) < 940:
                logger.warning(f"[BULLRUN_MONITOR] short BTC 5m fetch ({len(k5) if k5 else 0} bars) — skipping update")
                return
            closes = [float(r[4]) for r in k5[:-1]]  # closed bars only
            W = 864
            win = closes[-W:]
            r72 = (win[-1] / win[0] - 1) * 100.0
            ema = closes[0]; _k = 2.0 / 21.0; above_flags = []
            for c in closes:
                ema = c * _k + ema * (1 - _k)
                above_flags.append(c > ema)
            above = 100.0 * sum(above_flags[-W:]) / W
            diffs = sum(abs(win[i] - win[i - 1]) for i in range(1, W))
            eff = (abs(win[-1] - win[0]) / diffs) if diffs > 0 else 0.0
            r6 = (win[-1] / win[-72] - 1) * 100.0
            off24h = (win[-1] / max(float(r[2]) for r in k5[-289:-1]) - 1) * 100.0  # % below the 24h HIGH (pullback-phase gate)
            W24 = 288
            w24 = closes[-W24:]
            r24 = (w24[-1] / w24[0] - 1) * 100.0
            d24 = sum(abs(w24[i] - w24[i - 1]) for i in range(1, W24))
            eff24 = (abs(w24[-1] - w24[0]) / d24) if d24 > 0 else 0.0
            above24 = 100.0 * sum(above_flags[-W24:]) / W24
            e50h = None
            try:
                k1h = await binance_service.get_ohlcv('BTC/USDT:USDT', '1h', 150)
                if k1h and len(k1h) >= 60:
                    hc = [float(r[4]) for r in k1h[:-1]]
                    _e = hc[0]; _kk = 2.0 / 51.0
                    for c in hc:
                        _e = c * _kk + _e * (1 - _kk)
                    e50h = _e
            except Exception as _e50_err:
                logger.warning(f"[BULLRUN_MONITOR] 1h EMA50 fetch failed ({_e50_err}) — latch runs on r6h only this cycle")
            price = closes[-1]
            latch = (r6 <= float(getattr(th, 'bullrun_latch_r6h', -3.0) or -3.0)) or (e50h is not None and price < e50h)
            # Review C1: on the FIRST compute after a boot, seed the hysteresis state from the
            # open DB period — memory starts DARK, so without this a GREEN sitting in the
            # stay-band (below the turn-on bar) would be evaluated against turn-on thresholds,
            # fail, and be closed as 'restart→DARK' (a fake episode end + sleeve silently off).
            if not _bullrun_monitor.get('updated_at'):
                try:
                    async with AsyncSessionLocal() as _sdb:
                        _open_p = (await _sdb.execute(
                            select(MonitorPeriod).where(MonitorPeriod.ended_at.is_(None)).order_by(MonitorPeriod.started_at.desc()).limit(1)
                        )).scalar_one_or_none()
                        _age_min = (0.0 if _open_p.last_update is None else (datetime.utcnow() - _open_p.last_update).total_seconds() / 60) if _open_p is not None else None
                        if _open_p is not None and _age_min is not None and _age_min > 60:
                            logger.info(f"[BULLRUN_MONITOR] boot seed: open period #{_open_p.id} ({_open_p.state}) is {_age_min:.0f} min stale — NOT adopted; evaluating with turn-on thresholds (review I3 age bound)")
                        elif _open_p is not None:
                            _bullrun_monitor['green'] = (_open_p.state == 'GREEN')
                            _bullrun_monitor['state'] = _open_p.state
                            # Aug-23 (20) review: seed REARM too, else a deploy mid-REARM after a deploy mid-REARM the sleeve would silently disarm (the C1 bug class).
                            _bullrun_monitor['rearm'] = (_open_p.state == 'REARM')
                            if _open_p.state == 'REARM' and _open_p.started_at is not None:
                                _bullrun_monitor['rearm_t0'] = _open_p.started_at.replace(tzinfo=timezone.utc).timestamp()
                            logger.info(f"[BULLRUN_MONITOR] boot seed: open period #{_open_p.id} is {_open_p.state} — hysteresis seeded from DB")
                except Exception as _seed_err:
                    logger.warning(f"[BULLRUN_MONITOR] boot seed failed ({_seed_err}) — evaluating with turn-on thresholds")
            was_green = bool(_bullrun_monitor.get('green'))
            # `or default` guards: a blanked UI field saves 0, which must FAIL-DARK (defaults
            # restore), never fail-open into a trivially-satisfied GREEN (review minor).
            if latch:
                green = False
            elif was_green:
                green = (r72 >= float(th.bullrun_green_r72_off or 4.0) and above >= float(th.bullrun_green_above_off or 53.0)
                         and eff >= float(th.bullrun_green_eff_off or 0.10))
            else:
                green = (r72 >= float(th.bullrun_green_r72_on or 5.0) and above >= float(th.bullrun_green_above_on or 56.0)
                         and eff >= float(th.bullrun_green_eff_on or 0.10))
            amber = (not green) and (r24 >= float(th.bullrun_amber_r24 or 6.0) and above24 >= float(th.bullrun_amber_above or 65.0)
                                     and eff24 >= float(th.bullrun_amber_eff or 0.12))
            # Aug-23 (24): remember when the last GREEN episode ended (restart-proof: seeded from the periods ledger)
            if was_green and not green:
                _bullrun_monitor['green_end_ts'] = _now
            if _bullrun_monitor.get('green_end_ts') is None and not _bullrun_monitor.get('_gend_seeded'):
                _bullrun_monitor['_gend_seeded'] = True
                try:
                    async with AsyncSessionLocal() as _gdb:
                        _lg = (await _gdb.execute(
                            select(MonitorPeriod).where(and_(MonitorPeriod.state == 'GREEN', MonitorPeriod.ended_at.isnot(None)))
                            .order_by(MonitorPeriod.ended_at.desc()).limit(1))).scalars().first()
                        if _lg is not None:
                            _bullrun_monitor['green_end_ts'] = _lg.ended_at.replace(tzinfo=timezone.utc).timestamp()
                except Exception as _ge:
                    logger.warning(f"[BULLRUN_MONITOR] last-GREEN-end seed failed ({_ge})")
            # Aug-23 (25): RE-ARM door (composite OFF only). The rising test is computed BAR-EXACT from the
            # monitor's own closed 5m bars (Wilder ADX-14 series, now vs 6 bars ago) — restart-immune (the
            # in-memory adx_hist deque needed 25-35 min of warm-up after every deploy) and identical to the
            # 8.7-month backtest's definition.
            rearm = False
            try:
                _adx_now = None; _adx_prev6 = None
                try:
                    _kb = k5[:-1][-420:]
                    _bh = [float(r[2]) for r in _kb]; _bl = [float(r[3]) for r in _kb]; _bc = [float(r[4]) for r in _kb]
                    if len(_bc) >= 60:
                        _n = 14; _trs = []; _pdms = []; _ndms = []
                        for _i in range(1, len(_bc)):
                            _up = _bh[_i] - _bh[_i - 1]; _dn = _bl[_i - 1] - _bl[_i]
                            _pdms.append(_up if (_up > _dn and _up > 0) else 0.0)
                            _ndms.append(_dn if (_dn > _up and _dn > 0) else 0.0)
                            _trs.append(max(_bh[_i] - _bl[_i], abs(_bh[_i] - _bc[_i - 1]), abs(_bl[_i] - _bc[_i - 1])))
                        _atr = _trs[0]; _pd = _pdms[0]; _nd = _ndms[0]; _dxs = []
                        for _i in range(1, len(_trs)):
                            _atr = (_atr * (_n - 1) + _trs[_i]) / _n
                            _pd = (_pd * (_n - 1) + _pdms[_i]) / _n
                            _nd = (_nd * (_n - 1) + _ndms[_i]) / _n
                            _pdi = 100.0 * _pd / _atr if _atr > 0 else 0.0
                            _ndi = 100.0 * _nd / _atr if _atr > 0 else 0.0
                            _dxs.append(100.0 * abs(_pdi - _ndi) / (_pdi + _ndi) if (_pdi + _ndi) > 0 else 0.0)
                        _adx = _dxs[0]; _adx_series = []
                        for _i in range(1, len(_dxs)):
                            _adx = (_adx * (_n - 1) + _dxs[_i]) / _n
                            _adx_series.append(_adx)
                        if len(_adx_series) >= 7:
                            _adx_now = _adx_series[-1]; _adx_prev6 = _adx_series[-7]
                except Exception as _adx_err:
                    logger.warning(f"[BULLRUN_MONITOR] bar-exact ADX failed ({_adx_err}) — REARM off this tick")
                if bool(getattr(th, 'bullrun_rearm_enabled', False)) and not green and not latch:
                    _was_rearm = bool(_bullrun_monitor.get('rearm'))
                    _adx_min = float(getattr(th, 'bullrun_rearm_adx_min', 40.0) or 40.0)
                    _adx_off = float(getattr(th, 'bullrun_rearm_adx_off', 30.0) or 30.0)
                    _r6_min = float(getattr(th, 'bullrun_rearm_alt_r6h_min', 1.0) or 1.0)
                    _ab_min = float(getattr(th, 'bullrun_rearm_alt_above_pct', 80.0) or 80.0)
                    _max_h = float(getattr(th, 'bullrun_rearm_max_hours', 24.0) or 24.0)
                    _pv = getattr(th, 'bullrun_rearm_after_green_hours', 48.0)
                    _post_h = 48.0 if _pv is None else float(_pv)   # 0 = any time (explicit); None → default 48 (fail-closed)
                    _gend = _bullrun_monitor.get('green_end_ts')
                    _post_ok = (_post_h <= 0) or (_gend is not None and (_now - float(_gend)) / 3600.0 <= _post_h)
                    # EMA fan on the same closed BTC 5m closes the composite used
                    def _ema(seq, n):
                        k_ = 2.0 / (n + 1.0); e_ = seq[0]
                        for c_ in seq[1:]:
                            e_ = c_ * k_ + e_ * (1 - k_)
                        return e_
                    _e13, _e20, _e50 = _ema(closes, 13), _ema(closes, 20), _ema(closes, 50)
                    _fan = _e13 > _e20 > _e50
                    _above_1h = (e50h is not None and price > e50h)
                    _rising = (_adx_now is not None and _adx_prev6 is not None and float(_adx_now) > float(_adx_prev6))
                    _alts = [v for v in _br_alt_stats.values() if _now - (v.get('ts') or 0) <= 1200]
                    _alt_ok = False; _alt_med = None; _alt_ab = None
                    if len(_alts) >= 3:
                        _r6s = sorted(v['r6h'] for v in _alts); _alt_med = _r6s[len(_r6s) // 2]
                        _alt_ab = 100.0 * sum(1 for v in _alts if v.get('above')) / len(_alts)
                        _alt_ok = (_alt_med > _r6_min and _alt_ab >= _ab_min)
                    if not _was_rearm:
                        rearm = bool(_post_ok and _adx_now is not None and float(_adx_now) >= _adx_min and _rising and _alt_ok and _fan and _above_1h)
                        if rearm:
                            _bullrun_monitor['rearm_t0'] = _now
                            logger.critical(f"[BULLRUN_MONITOR] RE-ARM ON: ADX {float(_adx_now):.1f} rising, alts med r6h {_alt_med:+.2f}% / {_alt_ab:.0f}% above 1h EMA50, fan={_fan}, BTC>1hEMA50={_above_1h}")
                    else:
                        _age_h = (_now - float(_bullrun_monitor.get('rearm_t0') or _now)) / 3600.0
                        rearm = bool(_above_1h and _adx_now is not None and float(_adx_now) >= _adx_off and _age_h <= _max_h)
                        if not rearm:
                            logger.critical(f"[BULLRUN_MONITOR] RE-ARM OFF: ADX {float(_adx_now) if _adx_now is not None else float('nan'):.1f}, BTC>1hEMA50={_above_1h}, age {_age_h:.1f}h")
                    _bullrun_monitor['rearm_alt_med'] = _alt_med; _bullrun_monitor['rearm_alt_above'] = _alt_ab
            except Exception as _re_err:
                logger.warning(f"[BULLRUN_MONITOR] re-arm evaluation failed ({_re_err}) — REARM off this tick")
                rearm = False
            if _bullrun_monitor.get('rearm') and not rearm:
                logger.info(f"[BULLRUN_MONITOR] RE-ARM → {'GREEN' if green else ('latch' if latch else 'off')}")
            state = 'GREEN' if green else ('REARM' if rearm else ('AMBER' if amber else 'DARK'))
            prev_state = _bullrun_monitor.get('state')
            # Periods ledger (restart-proof adoption happens inside; sets 'resumed' on adoption)
            # Review I1: persistence runs on its OWN session — never commit/rollback the scan loop's
            # shared session from inside the monitor (today nothing is pending there; this keeps
            # it true if anyone ever writes above the call site).
            try:
                async with AsyncSessionLocal() as _pdb:
                    await self._bullrun_persist_period(_pdb, state, {'r72': r72, 'above': above, 'eff': eff, 'r6': r6, 'px': price, 'latch': latch})
            except Exception as _pers_err:
                logger.error(f"[BULLRUN_MONITOR] period session failed: {_pers_err}")
            resumed = bool(_bullrun_monitor.pop('resumed', False))
            if state != prev_state and not resumed:
                flip = {'ts': datetime.utcnow().strftime('%Y-%m-%d %H:%M'), 'state': state,
                        'r72': round(r72, 2), 'above': round(above, 1), 'eff': round(eff, 3), 'r6': round(r6, 2)}
                _bullrun_monitor['flips'] = (_bullrun_monitor.get('flips') or [])[-59:] + [flip]
                logger.critical(f"[BULLRUN_MONITOR] state {prev_state} → {state} | "
                                f"r72={r72:+.2f}% above={above:.1f}% eff={eff:.3f} r6={r6:+.2f}% latch={latch}")
            if green and not was_green and not resumed:
                _bullrun_monitor['green_since'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
            elif not green:
                _bullrun_monitor['green_since'] = None
            if rearm and not _bullrun_monitor.get('rearm'):
                _bullrun_monitor['rearm_since'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M')
            elif not rearm:
                _bullrun_monitor['rearm_since'] = None
            _bullrun_monitor.update({
                'state': state, 'green': green, 'rearm': rearm, 'amber': amber, 'latch': latch,
                'r72': round(r72, 2), 'above': round(above, 1), 'eff': round(eff, 3),
                'r6': round(r6, 2), 'r24': round(r24, 2), 'off24h': round(off24h, 2), 'updated_at': _now,
            })
        except Exception as e:
            logger.error(f"[BULLRUN_MONITOR] update failed: {e}")

    async def _maybe_open_bullrun_long(self, db, pair_info, ohlcv, indicators):
        """🌊 Aug-21 gate 57: BULLRUN_LONG sleeve entry — GREEN-gated dip-reclaim on scan-rank
        ≤ N COIN pairs (universe already COIN-only via coin_underlying_only). Entry: dip ≥
        dip_atr_mult×ATR(14,5m) below 5m EMA20 (flag persists), then a closed 5m bar reclaims
        EMA20, pair above its own 1h EMA50 (lazy 30-min cache), ≥ spacing_hours per pair, ≤
        max_slots concurrent. Sized via open_position(bullrun_long=True) → 1×/1× cells; the
        no-trade list (BTC/ETH) is bypassed ONLY for this tagged path (containment invariant
        amended). Alt entry filters do NOT run — the monitor replaces them at regime level.
        TO REMOVE: grep "BULLRUN" / "bullrun" / "_maybe_open_bullrun_long"."""
        pair = None
        try:
            th = config.trading_config.thresholds
            if not getattr(th, 'bullrun_sleeve_enabled', False):
                return
            _now = _leash_time.time()
            _fresh = (_now - (_bullrun_monitor.get('updated_at') or 0) <= 1800)
            _armed = bool(_fresh and (_bullrun_monitor.get('green') or _bullrun_monitor.get('rearm')))
            pair = pair_info.get('pair') or pair_info.get('symbol')
            _br_bl = {p.strip().upper() for p in str(getattr(th, 'bullrun_pair_blacklist', '') or '').split(',') if p.strip()}
            if _br_bl and str(pair).upper() in _br_bl:
                if _armed:
                    self._record_filter_block('BR_PAIR_BLACKLIST', 'LONG')
                return
            # Aug-22: rank among TRADEABLE pairs (br_rank, blacklists skipped) so top-N stays N pairs
            rank = pair_info.get('br_rank') if pair_info.get('br_rank') is not None else pair_info.get('rank')
            if not rank or int(rank) > int(getattr(th, 'bullrun_universe_size', 10) or 10):
                return
            if not ohlcv or len(ohlcv) < 40:
                return
            bars = ohlcv[:-1]  # closed bars only (last row is forming)
            closes = [float(b[4]) for b in bars]
            # Aug-23 (20): alt-participation stamp for the RE-ARM door (runs even while DARK; zero extra requests).
            # above = pair close vs its 1h EMA50 (sleeve cache when present, else the scan's 5m EMA50 as proxy).
            try:
                _r6h_p = (closes[-1] / closes[-73] - 1.0) * 100.0 if len(closes) >= 73 else None
                _c50 = _br_e50h_cache.get(pair)  # (ts, value) tuple written by the sleeve's 1h fetch
                _e50_ref = _c50[1] if (isinstance(_c50, tuple) and len(_c50) == 2 and _now - float(_c50[0] or 0) <= 3600) else None
                if _e50_ref is None:
                    _e50_ref = (indicators or {}).get('ema50')  # 5m EMA50 proxy while the 1h cache is cold
                if _r6h_p is not None and _e50_ref:
                    _br_alt_stats[pair] = {'r6h': _r6h_p, 'above': closes[-1] > float(_e50_ref), 'ts': _now}
            except Exception:
                pass
            if not _armed:
                return
            highs = [float(b[2]) for b in bars]
            lows = [float(b[3]) for b in bars]
            ema = closes[0]; _k = 2.0 / 21.0; emas = []
            for c in closes:
                ema = c * _k + ema * (1 - _k)
                emas.append(ema)
            trs = [max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
                   for i in range(1, len(bars))]
            if len(trs) < 20:
                return
            atr = sum(trs[:14]) / 14.0
            for t in trs[14:]:
                atr = (atr * 13 + t) / 14.0
            dipm = float(getattr(th, 'bullrun_dip_atr_mult', 0.3) or 0.3)
            st = _br_dip_state.setdefault(pair, {'dipped': False, 'last_bar_ts': 0})
            last_ts = st.get('last_bar_ts') or 0
            for i in range(max(1, len(bars) - 24), len(bars)):
                if int(bars[i][0]) <= last_ts:
                    continue
                if lows[i] <= emas[i] - dipm * atr:
                    st['dipped'] = True
                    st['dip_ts'] = int(bars[i][0])
            st['last_bar_ts'] = int(bars[-1][0])
            # review I3: dips EXPIRE after 6h — without this the flag survives GREEN→DARK→GREEN
            # and a stale dip from a prior window authorizes an instant entry on re-fire.
            if st.get('dipped') and int(bars[-1][0]) - (st.get('dip_ts') or 0) > 6 * 3600 * 1000:
                st['dipped'] = False
            if not st['dipped']:
                return
            if closes[-1] <= emas[-1]:
                return  # dip marked, reclaim not yet — wait
            # Aug-21 (11) PULLBACK-PHASE GATE (placed AFTER dip→reclaim so it counts real candidates,
            # before any DB work): no entry while BTC sits more than N% below its 24h high. The dip
            # is KEPT ALIVE while refused (6h expiry still applies) so the entry fires on the first
            # scan where BTC is back within range and the pair still holds above its EMA20 —
            # replay-tested vs consuming the dip: consumption cost −$925 on the founding window
            # (Aug-19 +$110 → −$865; strong pairs don't re-dip after BTC recovers, they run).
            # DECISION_LOG 2026-08-21 (13). 0 = off.
            _off_max = float(getattr(th, 'bullrun_btc_off24h_max', 0.0) or 0.0)
            if _off_max > 0:
                # review hygiene: a sign slip in the JSON (+2 meaning "2% below") must NOT silently disable
                # the gate — normalize to the negative form (0 is the only OFF value).
                if not _bullrun_monitor.get('_off_sign_warned'):
                    logger.warning(f"[BULLRUN_LONG] bullrun_btc_off24h_max={_off_max} is positive — interpreting as {-_off_max} (% below the 24h high); set 0 to disable")
                    _bullrun_monitor['_off_sign_warned'] = True
                _off_max = -_off_max
            _off_now = _bullrun_monitor.get('off24h')
            if _bullrun_monitor.get('green') and _off_max < 0 and _off_now is not None and float(_off_now) <= _off_max:  # bypassed in REARM (bounce from a ≥2% low)
                if st.get('blk_24h_ts') != st.get('dip_ts'):  # own key (review: shared blk_key double-counted an oscillating dip)
                    st['blk_24h_ts'] = st.get('dip_ts')
                    _bullrun_monitor['blk_off24h'] = _bullrun_monitor.get('blk_off24h', 0) + 1
                    self._record_filter_block("BULLRUN_BTC_OFF24H", "LONG")
                    logger.info(f"[BULLRUN_LONG] {pair}: refused by BTC off-24h-high gate (BTC {float(_off_now):+.2f}% vs max {_off_max}%, dip_ts={st.get('dip_ts')}) — dip kept alive, re-evaluated next scan")
                return
            # Aug-22 (3): BTC-leader gate — BTC must be at/above its own 5m EMA13 (module globals stamped by the scan).
            # Fail-open on missing data (parity with the other BTC gates). Counter per (rule, dip) like off-24h.
            if bool(getattr(th, 'bullrun_btc_ema13_required', True)):
                _g = globals(); _bpx = _g.get('_current_btc_price'); _be13 = _g.get('_current_btc_ema13')
                if _bpx is not None and _be13 is not None and float(_be13) > 0 and float(_bpx) < float(_be13):
                    if st.get('blk_e13_ts') != st.get('dip_ts'):
                        st['blk_e13_ts'] = st.get('dip_ts')
                        _bullrun_monitor['blk_ema13'] = _bullrun_monitor.get('blk_ema13', 0) + 1
                        self._record_filter_block("BULLRUN_BTC_EMA13", "LONG")
                        logger.info(f"[BULLRUN_LONG] {pair}: refused by BTC-leader gate (BTC {float(_bpx):.2f} below EMA13 {float(_be13):.2f}, {((float(_bpx)/float(_be13))-1)*100:+.3f}%) — dip stays alive")
                    return
            # Aug-23 (18): BTC 1h-EMA20 slope gate (None/blank = off). Fail-open on missing slope. Once per (rule, dip).
            _s_min = getattr(th, 'bullrun_btc_1h_slope_min', None)
            if _s_min is not None and str(_s_min) != '':
                _s_now = globals().get('_current_btc_1h_slope')
                if _s_now is not None and float(_s_now) <= float(_s_min):
                    if st.get('blk_1h_ts') != st.get('dip_ts'):
                        st['blk_1h_ts'] = st.get('dip_ts')
                        _bullrun_monitor['blk_1h'] = _bullrun_monitor.get('blk_1h', 0) + 1
                        self._record_filter_block("BULLRUN_BTC_1H_SLOPE", "LONG")
                        logger.info(f"[BULLRUN_LONG] {pair}: refused by BTC 1h-slope gate ({float(_s_now):+.3f} ≤ {float(_s_min):+.3f}) — dip stays alive")
                    return
            sp = float(getattr(th, 'bullrun_pair_spacing_hours', 2.0) or 2.0) * 3600.0
            # Restart-proof spacing (Aug-21 live catch: ONG re-entered 11 min after its stop because a
            # deploy wiped the in-memory stamp) — the DB is the source of truth: latest BULLRUN_LONG
            # order on this pair, opened_at OR closed_at, whichever is later.
            if pair not in _br_last_fire:  # one DB lookup per pair per process life (open+close both write the stamp afterwards)
              try:
                _last_o = (await db.execute(
                    select(Order).where(and_(Order.pair == pair, Order.entry_strategy == 'BULLRUN_LONG', Order.is_paper == self.is_paper_mode))
                    .order_by(Order.opened_at.desc()).limit(1)
                )).scalar_one_or_none()
                if _last_o is not None:
                    # DB datetimes are naive UTC (engine utcnow / SQLite func.now) — epoch math assumes that
                    _ts = max([t for t in (_last_o.opened_at, _last_o.closed_at) if t is not None]).replace(tzinfo=None)
                    _br_last_fire[pair] = (_ts - datetime(1970, 1, 1)).total_seconds()
                else:
                    _br_last_fire[pair] = 0.0  # no history: sentinel so we don't re-query every candidate
              except Exception as _sp_err:
                logger.warning(f"[BULLRUN_LONG] {pair}: DB spacing lookup failed ({_sp_err}) — using in-memory stamp")
            if _now - _br_last_fire.get(pair, 0) < sp:
                if st.get('blk_key') != ('sp', st.get('dip_ts')):  # review: count CANDIDATES, not scan cycles
                    st['blk_key'] = ('sp', st.get('dip_ts'))
                    _bullrun_monitor['blk_spacing'] = _bullrun_monitor.get('blk_spacing', 0) + 1
                return
            slots = int(getattr(th, 'bullrun_max_slots', 4) or 4)
            _open_ct = (await db.execute(
                select(func.count(Order.id)).where(and_(
                    Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                    Order.entry_strategy == "BULLRUN_LONG",
                ))
            )).scalar() or 0
            if _open_ct >= slots:
                self._record_filter_block("BULLRUN_MAX_SLOTS", "LONG")
                if st.get('blk_key') != ('sl', st.get('dip_ts')):
                    st['blk_key'] = ('sl', st.get('dip_ts'))
                    _bullrun_monitor['blk_slots'] = _bullrun_monitor.get('blk_slots', 0) + 1
                return
            price = float(indicators.get('price') or closes[-1])
            cached = _br_e50h_cache.get(pair)
            if not cached or _now - cached[0] > 1800:
                try:
                    k1h = await binance_service.get_ohlcv(pair_info.get('symbol') or pair, '1h', 100)
                    hc = [float(r[4]) for r in k1h[:-1]] if k1h else []
                    if len(hc) < 55:
                        return
                    _e = hc[0]; _kk = 2.0 / 51.0
                    for c in hc:
                        _e = c * _kk + _e * (1 - _kk)
                    cached = (_now, _e)
                    _br_e50h_cache[pair] = cached
                except Exception:
                    return  # fail-safe: no 1h context = no entry
            if price <= cached[1]:
                self._record_filter_block("BULLRUN_BELOW_1H_EMA50", "LONG")
                if st.get('blk_key') != ('ema', st.get('dip_ts')):
                    st['blk_key'] = ('ema', st.get('dip_ts'))
                    _bullrun_monitor['blk_ema50'] = _bullrun_monitor.get('blk_ema50', 0) + 1
                return
            atr_pct = (atr / closes[-1] * 100.0) if closes[-1] else None
            # Full entry-column stamping (operator, Aug-21): sleeve fills carry the SAME
            # analytics columns as every other trade (ADX/RSI + prevs, EMA gaps, stretch,
            # range-position, breadth, BTC-side state, funding, rank, quality score) via the
            # shared builder — W/L review must never be blind on a dimension (BULL_LONG pattern).
            _ef = dict(self._flip_entry_fields(indicators, flip_dir='LONG') or {})
            _ef.pop('entry_btc_trend_gap_pct', None)  # not an open_position param (stamped internally)
            for _k_ovr in ('entry_rsi', 'entry_adx', 'entry_atr_pct', 'entry_pair_volume_24h_usd', 'entry_pair_rank',
                           'entry_bull_pct', 'entry_bear_pct', 'entry_global_volume_ratio', 'entry_pair_volume_ratio'):
                _ef.pop(_k_ovr, None)
            # post-ship review I2: breadth + volume ratios come from scan ctx the hook doesn't
            # have — stamp from the live globals / indicators explicitly (WINDOW-UNITS review
            # of a market-wide-gated sleeve NEEDS breadth). Funding stays NULL for sleeve
            # fills (no source at the hook; on record in DECISION_LOG).
            _g_now = globals()
            _br_bull = _g_now.get('_market_bull_pct'); _br_bear = _g_now.get('_market_bear_pct')
            _br_gvr = _g_now.get('_global_volume_ratio')
            _br_pvr = ((indicators.get('volume') or 0) / indicators['avg_volume']) if indicators.get('avg_volume') else None
            order = await self.open_position(
                db=db, pair=pair, direction="LONG", confidence="STRONG_BUY", current_price=price,
                entry_rsi=indicators.get('rsi'), entry_adx=indicators.get('adx'),
                entry_atr_pct=atr_pct,
                entry_pair_volume_24h_usd=pair_info.get('volume_24h'),
                entry_pair_rank=rank,
                entry_br_r72=_bullrun_monitor.get('r72'),
                entry_br_above=_bullrun_monitor.get('above'),
                entry_br_eff=_bullrun_monitor.get('eff'),
                entry_br_off24h=_bullrun_monitor.get('off24h'),
                entry_br_door=('GREEN' if _bullrun_monitor.get('green') else 'REARM'),
                entry_bull_pct=_br_bull, entry_bear_pct=_br_bear,
                entry_global_volume_ratio=_br_gvr, entry_pair_volume_ratio=_br_pvr,
                bullrun_long=True,
                **self._sanitize_open_kwargs(_ef, 'BULLRUN_LONG', 'LONG'),
            )
            if order:
                st['dipped'] = False
                _br_last_fire[pair] = _now
                logger.info(f"[BULLRUN_LONG] {pair}: GREEN dip-reclaim opened (rank={rank} atr={atr_pct:.2f}% id={order.id})")
        except Exception as e:
            logger.error(f"[BULLRUN_LONG] {pair}: open failed: {e}")
            self._record_filter_block("OPEN_FAILED_BULLRUN", "LONG")

    async def _maybe_open_bounce_long(self, db, pair, indicators, isolate=False, entry_fields=None):
        """Bounce-Long Entry trigger (Jun 19) — the oversold-WASHOUT-bounce sleeve. In an allowed
        BEAR regime, when a SHORT is blocked by BTC_RSI_ADX_CROSS because BTC is washed out (the
        validated BTC RSI × BTC ADX cells), open a REAL LONG (NOT a fade) to catch the dead-cat
        bounce and let it ride the NORMAL long exit stack. Tagged entry_strategy="BOUNCE_LONG" via
        open_position(bounce_long=True); NOT _is_flip → normal per-level trailing / ATR-widened SL,
        exactly like BULL_LONG. Sizes base × bounce_long_size_mult × bounce_long_lev_mult (1.0 / 0.05
        = 1× observation). Fail-silent + isolatable. De-duped per pair/30min (key "pair|BOUNCE_LONG").
        Validated (phantom BTC_RSI_ADX_CROSS LONG): N=21, 95% WR, 0% SL, ALL H.BEAR; TIGHT cells only
        25-30:20-25 (89%) + 30-35:15-20 (100%). Its OWN sleeve — never calls _flip_filters, so the
        flip-long bear veto does NOT apply (that gate is correct for the FAN fade; this is the opposite
        thesis). TO REMOVE: grep "BOUNCE_LONG" / "bounce_long" / "_maybe_open_bounce_long"."""
        try:
            _th = config.trading_config.thresholds
            if not getattr(_th, 'bounce_long_enabled', False):
                return
            price = indicators.get('price') if indicators else None
            if not price or price <= 0:
                return
            _ef = dict(entry_fields or {})
            _ind = indicators or {}
            _g = globals()
            # BTC RSI / ADX — prefer the recorded entry fields, else live globals.
            _brsi = _ef.get('entry_btc_rsi'); _badx = _ef.get('entry_btc_adx')
            if _brsi is None:
                _brsi = _g.get('_current_btc_rsi')
            if _badx is None:
                _badx = _g.get('_current_btc_adx')
            if _brsi is None or _badx is None:
                return
            # TIGHT (BTC RSI × BTC ADX) cell gate — fire ONLY inside a validated washout cell.
            # Format "rsi_lo-rsi_hi:adx_lo-adx_hi,..." (mirrors btc_rsi_adx_multiplier_*). Empty = OFF.
            _cells = (getattr(_th, 'bounce_long_btc_cells', '') or '').strip()
            if not _cells:
                return
            _hit = False
            for _cell in _cells.split(','):
                _cell = _cell.strip()
                if not _cell or ':' not in _cell:
                    continue
                try:
                    _rb, _ab = _cell.split(':')
                    _rlo, _rhi = (float(x) for x in _rb.split('-'))
                    _alo, _ahi = (float(x) for x in _ab.split('-'))
                except (ValueError, TypeError):
                    continue
                if _rlo <= _brsi < _rhi and _alo <= _badx < _ahi:
                    _hit = True
                    break
            if not _hit:
                return
            # BTC regime — prefer recorded entry regime, else classify from live globals.
            _reg = _ef.get('entry_btc_regime')
            if _reg is None:
                try:
                    _reg = classify_btc_regime(_g.get('_current_btc_adx'), _g.get('_current_btc_rsi'), _g.get('_btc_ema20_slope_pct'))
                except Exception:
                    _reg = None
            _allowed = {r.strip().upper() for r in (getattr(_th, 'bounce_long_regimes', '') or '').split(',') if r.strip()}
            if not _allowed or (_reg or '').upper() not in _allowed:
                return
            # Stamp the GATED regime onto the order (BULL_LONG bugfix mirror — else open_position
            # re-classifies and can land on null/NEUTRAL, dropping the bounce-long out of the regime tables).
            _ef['entry_btc_regime'] = _reg
            _ck = f"{pair}|BOUNCE_LONG"
            _now = _leash_time.time()
            if _now - _PFLIP_COOLDOWN.get(_ck, 0) < _PFLIP_COOLDOWN_MIN * 60:
                return
            _size_mult = getattr(_th, 'bounce_long_size_mult', 1.0) or 1.0
            _lev_mult = getattr(_th, 'bounce_long_lev_mult', 1.0) or 1.0
            async def _open(_db):
                _ef.pop('entry_btc_trend_gap_pct', None)  # Aug-11: not an open_position param (stamped internally from globals) — same poison that killed FLIP for 34 days; sanitizer below catches any future stragglers
                return await self.open_position(
                    db=_db, pair=pair, direction="LONG", confidence="STRONG_BUY",
                    current_price=price,
                    entry_rsi=_ef.pop('entry_rsi', None) or indicators.get('rsi'),
                    entry_adx=_ef.pop('entry_adx', None) or indicators.get('adx'),
                    entry_atr_pct=_pop_or(_ef, 'entry_atr_pct', _ind_atr_pct(indicators)),
                    bounce_long=True, bounce_long_size_mult=_size_mult, bounce_long_lev_mult=_lev_mult,
                    **self._sanitize_open_kwargs(_ef, 'BOUNCE_LONG', 'LONG'),
                )
            if isolate:
                async with AsyncSessionLocal() as _bdb:
                    order = await _open(_bdb)
            else:
                order = await _open(db)
            if order:
                _PFLIP_COOLDOWN[_ck] = _now
                logger.info(f"[BOUNCE_LONG] {pair}: opened bounce-long (btcRSI={_brsi:.1f} btcADX={_badx:.1f} regime={_reg} id={order.id})")
            return order
        except Exception as e:
            logger.error(f"[BOUNCE_LONG] {pair}: bounce-long open failed: {e}")
            self._record_filter_block("OPEN_FAILED_BOUNCE_LONG", "LONG")  # Aug-11: dashboard-visible sleeve-death counter
            return None

    # ===== SPIKE SCANNER START (Jul 24) — helper + cycle =====
    @staticmethod
    def _spike_rsi12(closes):
        """Wilder RSI(12) from a close list -> (rsi, rsi_prev1). Lightweight (no pandas)."""
        try:
            if len(closes) < 20:
                return None, None
            au = ad = 0.0
            rsis = []
            for i in range(1, len(closes)):
                ch = closes[i] - closes[i - 1]
                u = ch if ch > 0 else 0.0
                d = -ch if ch < 0 else 0.0
                if i == 1:
                    au, ad = u, d
                else:
                    au = (au * 11 + u) / 12.0
                    ad = (ad * 11 + d) / 12.0
                if i >= 12:
                    rsis.append(100.0 - 100.0 / (1 + (au / ad)) if ad > 0 else 100.0)
            if len(rsis) < 2:
                return None, None
            return rsis[-1], rsis[-2]
        except Exception:
            return None, None

    async def _spike_scanner_cycle(self, db: AsyncSession, top_set: set):
        """Full-universe SPIKE_CHASE feeder (Jul 24). Scans eligible USDT perps BEYOND the
        top-50 with ONLY the 2-candle RSI-jump trigger (never the ladder); a fire routes into
        the standard SPIKE_CHASE probe (same tag / caps / sizing / gates). Fail-silent;
        piggybacks the scan loop; one fire per pair per candle. ZERO-RISK REVERT =
        spike_scanner_enabled=false."""
        th = config.trading_config.thresholds
        # Jul 31 🏀: the scanner now feeds TWO species — pump (chase/fade) and bounce.
        # It runs if either is on; each trigger branch is gated by its own toggle so
        # disabling one species can never silently kill the other's 400-pair surface.
        _pump_on = bool(getattr(th, 'spike_chase_probe_enabled', False))
        _sb_on = bool(getattr(th, 'spike_bounce_enabled', False))
        if not getattr(th, 'spike_scanner_enabled', False) or not (_pump_on or _sb_on):
            return
        if not self.is_running:
            return
        _jump = float(getattr(th, 'spike_chase_probe_rsi_jump', 25.0) or 25.0)
        _prev_max = float(getattr(th, 'spike_chase_probe_rsi_prev_max', 55.0) or 55.0)
        _min_chg = float(getattr(th, 'spike_chase_probe_min_candle_pct', 0.5) or 0.5)
        _prev_min = float(getattr(th, 'spike_chase_probe_rsi_prev_min', 35.0) or 35.0)
        _min_vr = float(getattr(th, 'spike_chase_probe_min_vol_ratio', 5.0) or 5.0)
        # Jul 31 🏀 SPIKE_BOUNCE (third species): mirrored dump-trigger params.
        _sb_crash = float(getattr(th, 'spike_bounce_rsi_crash', 25.0) or 25.0)
        _sb_pmin = float(getattr(th, 'spike_bounce_rsi_prev_min', 45.0) or 45.0)
        _sb_pmax = float(getattr(th, 'spike_bounce_rsi_prev_max', 65.0) or 65.0)
        _sb_chg = float(getattr(th, 'spike_bounce_min_candle_pct', 0.5) or 0.5)
        _sb_maxdump = float(getattr(th, 'spike_bounce_max_dump_pct', 3.0) or 3.0)
        _sb_vr = float(getattr(th, 'spike_bounce_min_vol_ratio', 5.0) or 5.0)
        _vol_floor = float(getattr(th, 'spike_scanner_min_vol_usd', 1000000.0) or 1000000.0)
        _max_pairs = int(getattr(th, 'spike_scanner_max_pairs', 400) or 400)
        # Same protective screens as the trading universe (new-listing/Alpha/coin-only).
        universe = await binance_service.get_top_futures_pairs(
            _max_pairs,
            new_listing_filter_days=getattr(config.trading_config, 'new_listing_filter_days', 0),
            alpha_subtype_filter_enabled=getattr(config.trading_config, 'alpha_subtype_filter_enabled', True),
            coin_underlying_only=getattr(config.trading_config, 'coin_underlying_only', True),
        )
        # Jul 24 (review fix I2): stamp the eligible-universe volume rank (1 = highest)
        # BEFORE filtering, mirroring the main scan's convention, so entry_pair_rank is
        # populated on scanner fires (rank-dimension analysis at the read).
        for _ui, _up in enumerate(universe):
            _up['rank'] = _ui + 1
        _bl = set(x.strip() for x in (getattr(config.trading_config, 'pair_blacklist', '') or '').split(',') if x.strip())
        _nt = set(x.strip() for x in (getattr(config.trading_config, 'no_trade_pairs', '') or '').split(',') if x.strip())
        cands = [p for p in universe
                 if p['pair'] not in top_set and p['pair'] not in _bl and p['pair'] not in _nt
                 and (p.get('volume_24h') or 0) >= _vol_floor]
        if not cands:
            return
        seen = getattr(self, '_spike_scan_seen', None)
        if seen is None:
            seen = {}
            self._spike_scan_seen = seen
        _fired = 0
        _checked = 0
        _B = 8  # concurrency batch — gentle on rate limits (~350 klines calls / cycle)
        for i in range(0, len(cands), _B):
            batch = cands[i:i + _B]
            results = await asyncio.gather(
                *[binance_service.get_ohlcv(p['symbol'], '5m', 100) for p in batch],
                return_exceptions=True)
            for p, ohlcv in zip(batch, results):
                if isinstance(ohlcv, Exception) or not ohlcv or len(ohlcv) < 20:
                    continue
                _checked += 1
                try:
                    closes = [float(c[4]) for c in ohlcv]
                    rsi, rsi_prev = self._spike_rsi12(closes)
                    if rsi is None or rsi_prev is None:
                        continue
                    _pump_trig = (_pump_on and _prev_min <= rsi_prev <= _prev_max and (rsi - rsi_prev) >= _jump)
                    # Jul 31 🏀 BOUNCE trigger (mirror): RSI crash from a resting band. Only
                    # evaluated when the pump didn't fire (a candle can't be both).
                    _bounce_trig = (not _pump_trig and _sb_on
                                    and _sb_pmin <= rsi_prev <= _sb_pmax
                                    and (rsi_prev - rsi) >= _sb_crash)
                    if not (_pump_trig or _bounce_trig):
                        continue
                    # Jul 24 PM: price-magnitude leg — RSI is scale-free, so a stablecoin's
                    # +0.01% wiggle can print a +36-pt "jump" (USDCUSDT #175). Real discovery
                    # candles move price (MIRA +1.84%); require the candle itself >= _min_chg %.
                    if closes[-2] <= 0:
                        continue
                    _chg_now = (closes[-1] / closes[-2] - 1.0) * 100.0
                    if _pump_trig and _chg_now < _min_chg:
                        continue
                    if _bounce_trig and _chg_now > -_sb_chg:
                        continue  # dump candle too shallow
                    # Jul 24 PM leg 5: attention — discovery-candle volume >= _min_vr x prior-20 avg
                    # (MIRA 59.6x / chop max 5.7x / USDCUSDT fire 2.39x). Free from these klines.
                    _vols = [float(c[5]) for c in ohlcv]
                    _av20 = sum(_vols[-21:-1]) / 20.0
                    if _av20 <= 0 or (_vols[-1] / _av20) < (_sb_vr if _bounce_trig else _min_vr):
                        continue
                    _candle_ts = ohlcv[-1][0]  # one fire per pair per candle
                    if seen.get(p['pair']) == _candle_ts:
                        continue
                    seen[p['pair']] = _candle_ts
                    if len(seen) > 600:
                        seen.clear()
                    if _bounce_trig and _chg_now < -_sb_maxdump:
                        # 🏀 guard ①: news/delist/hack class — no bounce edge, catastrophic
                        # tail. Placed AFTER the vol leg + dedup (review fix): the counter
                        # must count full triggers ONCE per candle, matching the ladder hook.
                        self._record_filter_block("SPIKE_BOUNCE_DUMPCAP", "LONG")
                        logger.info(f"[SPIKE_BOUNCE_DUMPCAP] {p['pair']}: dump {_chg_now:+.2f}% deeper than -{_sb_maxdump}% — news-class, no bounce")
                        continue
                    ind = calculate_indicators(ohlcv)
                    if not ind or not ind.get('price'):
                        continue
                    # Jul 27 LEG 6 router (same rule as the top-50 hook): pair ADX decides
                    # the direction — <=max CHASE LONG, >max FADE SHORT (or skip if fade off).
                    _sp_adx = ind.get('adx')
                    _sp_max_adx = float(getattr(th, 'spike_chase_max_adx', 30.0) or 30.0)
                    _sp_dir = "LONG"
                    _sp_is_fade = False
                    _sp_is_bounce = False
                    if _bounce_trig:
                        # Jul 31 🏀 BOUNCE guards — a dump routes to LONG directly (no
                        # ADX/regime router: the router splits PUMPS into chase/fade; a
                        # dump has exactly one long thesis). Each guard earned its place
                        # with side-specific evidence (design rule: no symmetry-by-aesthetics).
                        _gl = globals()
                        # guard ② bRSI >= floor — true mirror of the fade's <=50 ceiling:
                        # fade shorts idiosyncratic exhaustion (needs calm BTC); bounce longs
                        # idiosyncratic panic (needs non-bearish BTC). Fail-open on missing.
                        _sb_brsi = _gl.get('_current_btc_rsi')
                        _sb_brsi_min = float(getattr(th, 'spike_bounce_min_btc_rsi', 0.0) or 0.0)
                        if _sb_brsi_min > 0 and _sb_brsi is not None and _sb_brsi < _sb_brsi_min:
                            self._record_filter_block("SPIKE_BOUNCE_BRSI", "LONG")
                            logger.info(f"[SPIKE_BOUNCE_BRSI] {p['pair']}: bounce blocked — BTC RSI {_sb_brsi:.1f} < {_sb_brsi_min} (market-wide risk-off, dump is beta not panic) | entry_px={ind.get('price')} pair_rsi={ind.get('rsi')}")
                            continue
                        # guard ③ crashed-pair exclusion (DEEPGAP evidence): pair already in
                        # a downtrend (EMA13-50 gap <= min) = knife in trend, not a flash panic.
                        _sb_gap_min = float(getattr(th, 'spike_bounce_min_pair_gap', 0.0) or 0.0)
                        _sb_e13 = ind.get('ema13'); _sb_e50 = ind.get('ema50')
                        if _sb_e13 is not None and _sb_e50:
                            _sb_gap = (_sb_e13 - _sb_e50) / _sb_e50 * 100.0
                            if _sb_gap_min != 0 and _sb_gap <= _sb_gap_min:
                                self._record_filter_block("SPIKE_BOUNCE_CRASHED", "LONG")
                                logger.info(f"[SPIKE_BOUNCE_CRASHED] {p['pair']}: bounce blocked — EMA13-50 gap {_sb_gap:+.2f}% <= {_sb_gap_min}% (already-crashed pair, DEEPGAP class)")
                                continue
                            # guard ③b healthy-pair exclusion (Aug-5 pgap window): a violent dump
                            # out of a flat/uptrending pair is news, not a stop cascade — no snapback.
                            # Independent of the DEEPGAP floor (still runs if min-gap is set to 0/off).
                            _sb_gap_max = float(getattr(th, 'spike_bounce_max_pair_gap', 99.0) if getattr(th, 'spike_bounce_max_pair_gap', None) is not None else 99.0)
                            if _sb_gap_max < 99 and _sb_gap > _sb_gap_max:
                                self._record_filter_block("SPIKE_BOUNCE_PGAP", "LONG")
                                logger.info(f"[SPIKE_BOUNCE_PGAP] {p['pair']}: bounce blocked — EMA13-50 gap {_sb_gap:+.2f}% > {_sb_gap_max}% (healthy-pair dump = news class) | entry_px={ind.get('price')} pair_rsi={ind.get('rsi')} — re-sim revert row")
                                continue
                        # guard ④ regime block: no knife-catching in confirmed bear tape
                        # (BOUNCE_LONG's Jun-23 graves were HEALTHY_BEAR cells).
                        _sb_blocked = set(x.strip() for x in (getattr(th, 'spike_bounce_blocked_regimes', '') or '').split(',') if x.strip())
                        if _sb_blocked:
                            try:
                                _sb_reg = classify_btc_regime(
                                    _gl.get('_current_btc_adx'), _gl.get('_current_btc_rsi'), _gl.get('_btc_ema20_slope_pct'))
                            except Exception:
                                _sb_reg = None
                            if _sb_reg is None or _sb_reg == 'UNKNOWN':
                                # coarse fallback: treat BEARISH macro as blocked
                                if (_gl.get('_current_btc_regime') or 'NEUTRAL') == 'BEARISH':
                                    _sb_reg = 'STRONG_BEAR'
                            if _sb_reg in _sb_blocked:
                                self._record_filter_block("SPIKE_BOUNCE_REGIME", "LONG")
                                logger.info(f"[SPIKE_BOUNCE_REGIME] {p['pair']}: bounce blocked — regime {_sb_reg} in blocked set {sorted(_sb_blocked)}")
                                continue
                        _sp_is_bounce = True
                    # Jul 28 REGIME ROUTER — regime decides FIRST (non-bull spikes are
                    # squeeze-wicks: 12/13 never reached +0.45 honest). Chase regimes ->
                    # ADX leg as before; anything else -> FADE. Fallback on classify
                    # failure = coarse macro (BULLISH -> chase logic, else fade).
                    _sp_regime_fade = False
                    _gl = globals()
                    if not _sp_is_bounce and getattr(th, 'spike_regime_router_enabled', False):
                        try:
                            _sp_reg_now = classify_btc_regime(
                                _gl.get('_current_btc_adx'), _gl.get('_current_btc_rsi'), _gl.get('_btc_ema20_slope_pct'))
                        except Exception:
                            _sp_reg_now = None
                        _sp_chase_regs = set(x.strip() for x in (getattr(th, 'spike_chase_regimes', '') or '').split(',') if x.strip())
                        if _sp_reg_now is not None and _sp_reg_now != 'UNKNOWN':
                            _sp_regime_fade = _sp_reg_now not in _sp_chase_regs
                        else:
                            # classify returns UNKNOWN on missing inputs (never raises) —
                            # fall back to the coarse macro global (review M-3).
                            _sp_regime_fade = (_gl.get('_current_btc_regime') or 'NEUTRAL') != 'BULLISH'
                    if not _sp_is_bounce and (_sp_regime_fade or (_sp_adx is not None and _sp_adx > _sp_max_adx)):
                        if not getattr(th, 'spike_fade_enabled', False):
                            logger.info(f"[SPIKE_ROUTER_BLOCK] {p['pair']}: trigger fired but routed to FADE ({'regime' if _sp_regime_fade else 'ADX'}) and fade disabled — no trade")
                            continue
                        # Jul 30 PM — fade bRSI ceiling (scanner parity with the top-50 hook):
                        # don't fade while BTC's own momentum is hot. Fail-open on missing bRSI.
                        # Log carries the re-sim revert surface (price + bRSI at block time).
                        _sp_brsi = _gl.get('_current_btc_rsi')
                        _sp_brsi_max = float(getattr(th, 'spike_fade_max_btc_rsi', 0.0) or 0.0)
                        if _sp_brsi_max > 0 and _sp_brsi is not None and _sp_brsi > _sp_brsi_max:
                            self._record_filter_block("SPIKE_FADE_BRSI", "SHORT")
                            logger.info(f"[SPIKE_FADE_BRSI] {p['pair']}: scanner fade blocked — BTC RSI {_sp_brsi:.1f} > {_sp_brsi_max} (squeeze-against-gravity guard) | entry_px={ind.get('price')} pair_rsi={ind.get('rsi')} — re-sim revert row")
                            continue
                        # Aug-4 FADE BTC-DIST13 GATE (scanner parity; see config comment):
                        # don't short vs BTC above its 5m mean. 0W/5L lifetime above the line.
                        _sp_bd13_max = float(getattr(th, 'spike_fade_max_btc_dist13', 99.0) if getattr(th, 'spike_fade_max_btc_dist13', 99.0) is not None else 99.0)
                        _sp_bpx = _gl.get('_current_btc_price'); _sp_be13 = _gl.get('_current_btc_ema13')
                        _sp_bd13 = ((_sp_bpx - _sp_be13) / _sp_be13 * 100.0) if (_sp_bpx and _sp_be13) else None
                        if _sp_bd13_max < 99 and _sp_bd13 is not None and _sp_bd13 > _sp_bd13_max:
                            self._record_filter_block("SPIKE_FADE_BD13", "SHORT")
                            logger.info(f"[SPIKE_FADE_BD13] {p['pair']}: scanner fade blocked — BTC dist-EMA13 {_sp_bd13:+.3f}% > {_sp_bd13_max} (beta-tailwind guard) | entry_px={ind.get('price')} pair_rsi={ind.get('rsi')} — re-sim revert row")
                            continue
                        # Aug-10 FRESH-BREAKOUT GUARD (operator "ship B"; zone leg = watchlist):
                        # never fade a spike from a low-RSI base on a non-crashed pair — that is
                        # the START of a move (breakout / squeeze ignition), not an exhaustion.
                        # Stack-screened: blocked 20·45%·−$344 combined; VANRY class kept.
                        _sp_fb_rmin = float(getattr(th, 'spike_fade_fb_rsi_prev_min', 0.0) or 0.0)
                        _sp_fb_gmin = float(getattr(th, 'spike_fade_fb_pgap_min', -0.40) if getattr(th, 'spike_fade_fb_pgap_min', None) is not None else -0.40)
                        if _sp_fb_rmin > 0 and rsi_prev is not None and rsi_prev < _sp_fb_rmin:
                            _sp_e13 = ind.get('ema13'); _sp_e50 = ind.get('ema50')
                            _sp_pg = ((_sp_e13 - _sp_e50) / _sp_e50 * 100.0) if (_sp_e13 is not None and _sp_e50) else None
                            if _sp_pg is not None and _sp_pg > _sp_fb_gmin:
                                self._record_filter_block("SPIKE_FADE_FRESHBREAK", "SHORT")
                                logger.info(f"[SPIKE_FADE_FRESHBREAK] {p['pair']}: scanner fade blocked — base RSI {rsi_prev:.1f} < {_sp_fb_rmin} on non-crashed pair (pgap {_sp_pg:+.2f} > {_sp_fb_gmin}) = fresh breakout, not exhaustion | entry_px={ind.get('price')} — re-sim revert row")
                                continue
                        _sp_dir, _sp_is_fade = "SHORT", True
                        if _sp_regime_fade:
                            logger.info(f"[SPIKE_REGIME_FADE] {p['pair']}: non-chase regime — routing trigger to FADE short")
                    if not _sp_is_fade and not _sp_is_bounce:
                        # 🛑 Aug-21 dedicated CHASE kill-switch (operator-directed; mirrors
                        # spike_fade_enabled — the species had no own flag since graduation).
                        if not getattr(th, 'spike_chase_enabled', True):
                            self._record_filter_block("SPIKE_CHASE_DISABLED", "LONG")
                            logger.info(f"[SPIKE_ROUTER_BLOCK] {p['pair']}: trigger fired, routed to CHASE and chase disabled — no trade")
                            continue
                        # Jul 30 EXTENSION GUARD (chase-only, scanner parity with the top-50
                        # hook): block chase when stretch > mult x ATR%. Fail-open on missing.
                        _sp_str_mult = float(getattr(th, 'spike_chase_max_stretch_atr', 1.5) or 0.0)
                        _sp_atrp = _ind_atr_pct(ind)
                        _sp_stretch = None
                        try:
                            if ind.get('ema5') and ind.get('price'):
                                _sp_stretch = (ind['price'] - ind['ema5']) / ind['ema5'] * 100.0
                        except Exception:
                            _sp_stretch = None
                        if (_sp_str_mult > 0 and _sp_stretch is not None and _sp_atrp
                                and _sp_stretch > _sp_str_mult * _sp_atrp):
                            self._record_filter_block("SPIKE_CHASE_STRETCH", "LONG")
                            logger.info(f"[SPIKE_CHASE_STRETCH] {p['pair']}: scanner chase blocked — stretch {_sp_stretch:+.2f}% > {_sp_str_mult}x ATR {_sp_atrp:.2f}% (ratio {(_sp_stretch/_sp_atrp):.1f})")
                            continue
                    logger.info(f"[SPIKE_SCANNER] {p['pair']}: RSI {'crash' if _sp_is_bounce else 'jump'} {rsi_prev:.1f}->{rsi:.1f} "
                                f"({rsi - rsi_prev:+.1f}) candle {(closes[-1]/closes[-2]-1.0)*100.0:+.2f}% vol {_vols[-1]/_av20:.1f}x ADX {(_sp_adx if _sp_adx is not None else -1):.1f} vol24h=${(p.get('volume_24h') or 0)/1e6:.1f}M — {'SPIKE_BOUNCE long' if _sp_is_bounce else ('SPIKE_FADE short' if _sp_is_fade else 'SPIKE_CHASE long')} entry")
                    _g = globals()
                    _px = ind['price']
                    def _r(v, n=4):
                        return round(v, n) if v is not None else None
                    _atr = ind.get('atr')
                    _ema50 = ind.get('ema50'); _ema50_p12 = ind.get('ema50_prev12')
                    _ema13v = ind.get('ema13')
                    try:
                        _sp_fine_regime = classify_btc_regime(
                            _g.get('_current_btc_adx'), _g.get('_current_btc_rsi'), _g.get('_btc_ema20_slope_pct'))
                    except Exception:
                        _sp_fine_regime = _g.get('_current_btc_regime')
                    # Jul 27 night: quality score for spike fires too (operator) — same formulas
                    # as the top-50 ladder (gap = |EMA5-EMA20|/price, slope = EMA20 3-bar).
                    # Analytics-only: QS never gates the spike class (spike is ladder-exempt).
                    try:
                        _sp_e20 = ind.get('ema20'); _sp_e20p3 = ind.get('ema20_prev3')
                        _sp_gap = round(abs((ind['ema5'] - _sp_e20) / _px * 100), 4) if ind.get('ema5') is not None and _sp_e20 else None
                        _sp_slope = round(((_sp_e20 - _sp_e20p3) / _sp_e20p3) * 100, 4) if _sp_e20 and _sp_e20p3 else None
                        _sp_qs = _calculate_quality_score(
                            _sp_dir, ind.get('rsi'), ind.get('adx'), _sp_gap,
                            _g.get('_market_bull_pct'), _g.get('_market_bear_pct'),
                            _g.get('_current_btc_adx'), _sp_slope)
                    except Exception:
                        _sp_qs = None; _sp_slope = None
                    order = await self.open_position(
                        db=db, pair=p['pair'], direction=_sp_dir, confidence="STRONG_BUY",
                        current_price=_px,
                        entry_gap=_r(abs((ind['ema5'] - ind['ema20']) / _px * 100)) if ind.get('ema5') and ind.get('ema20') else None,
                        entry_ema_gap_5_8=_r(abs((ind['ema5'] - ind['ema8']) / ind['ema8'] * 100)) if ind.get('ema5') and ind.get('ema8') else None,
                        entry_ema_gap_8_13=_r(abs((ind['ema8'] - ind['ema13']) / ind['ema13'] * 100)) if ind.get('ema8') and ind.get('ema13') else None,
                        entry_ema5_stretch=_r(abs(_px - ind['ema5']) / _px * 100) if ind.get('ema5') else None,
                        entry_price_vs_ema5_pct=_r((_px - ind['ema5']) / ind['ema5'] * 100) if ind.get('ema5') else None,
                        entry_rsi=_r(ind.get('rsi'), 2), entry_rsi_prev=_r(ind.get('rsi_prev2'), 2),
                        entry_adx=_r(ind.get('adx')), entry_adx_prev=_r(ind.get('adx_prev1')),
                        entry_adx_delta=_r((ind.get('adx') - ind.get('adx_prev1'))) if ind.get('adx') is not None and ind.get('adx_prev1') is not None else None,
                        entry_pos_di=ind.get('pos_di'), entry_neg_di=ind.get('neg_di'),
                        entry_atr_pct=_r((_atr / _px) * 100) if _atr is not None else None,
                        entry_ema50_slope=_r(((_ema50 - _ema50_p12) / _ema50_p12) * 100) if _ema50 is not None and _ema50_p12 else None,
                        entry_pair_ema20_ema50_gap_pct=_r((_ema13v - _ema50) / _ema50 * 100) if _ema13v is not None and _ema50 else None,
                        entry_dist_from_ema13_pct=_r((_px - _ema13v) / _ema13v * 100) if _ema13v else None,
                        entry_range_position=_r(((_px - ind['low_20']) / (ind['high_20'] - ind['low_20'])) * 100, 1) if ind.get('high_20') and ind.get('low_20') and ind['high_20'] != ind['low_20'] else None,
                        entry_pair_volume_ratio=_r((ind.get('volume') or 0) / ind['avg_volume']) if ind.get('avg_volume') else None,
                        entry_pair_volume_24h_usd=p.get('volume_24h'),
                        entry_pair_rank=p.get('rank'),
                        entry_pair_age_days=p.get('age_days'),
                        entry_btc_adx=_g.get('_current_btc_adx'), entry_btc_rsi=_g.get('_current_btc_rsi'),
                        entry_btc_adx_prev=_g.get('_current_btc_adx_prev'), entry_btc_rsi_prev=_g.get('_current_btc_rsi_prev'),
                        entry_btc_rsi_prev6=_g.get('_current_btc_rsi_prev6'),
                        entry_btc_atr_pct=_g.get('_current_btc_atr_pct'),
                        entry_btc_rsi_1h=_g.get('_current_btc_rsi_1h'), entry_btc_rsi_1h_prev=_g.get('_current_btc_rsi_1h_prev'),
                        entry_btc_dist_from_ema13_pct=_r((_g.get('_current_btc_price') - _g.get('_current_btc_ema13')) / _g.get('_current_btc_ema13') * 100) if _g.get('_current_btc_price') and _g.get('_current_btc_ema13') else None,
                        entry_bull_pct=_g.get('_market_bull_pct'), entry_bear_pct=_g.get('_market_bear_pct'),
                        # Jul 27 night fix: was stamping the COARSE macro-trend global into
                        # entry_btc_regime (BULLISH/NEUTRAL/BEARISH) — the fine 6-way regime
                        # was never recorded on scanner-path spike fires, capping the regime
                        # read at bull-vs-non-bull granularity. Stamp both, like momentum does.
                        entry_btc_regime=_sp_fine_regime,
                        entry_macro_trend=_g.get('_current_btc_regime'),
                        entry_quality_score=_sp_qs, entry_ema20_slope=_sp_slope,
                        spike_chase_probe=(not _sp_is_fade and not _sp_is_bounce),
                        spike_fade=_sp_is_fade,
                        spike_bounce=_sp_is_bounce,
                    )
                    if order:
                        _fired += 1
                except Exception as _sp_err:
                    logger.error(f"[SPIKE_SCANNER] {p.get('pair')}: trigger/open failed (fail-silent): {_sp_err}")
                    self._record_filter_block("OPEN_FAILED_SPIKE_SCANNER", "ANY")  # Aug-11: dashboard-visible sleeve-death counter
            if i + _B < len(cands):
                # Jul 24 (review I3): ccxt's enableRateLimit throttler already serializes/paces
                # the calls; keep only a token yield so the loop stays cooperative (~20-30s/cycle).
                await asyncio.sleep(0.1)
        if _fired or _checked:
            logger.info(f"[SPIKE_SCANNER] cycle done: {_checked} extended-universe pairs checked, {_fired} probe fires")
    # ===== SPIKE SCANNER END =====

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
        entry_pair_age_days: float = None,  # Jul 13: listing age at entry (180->90 read gate)
        # Aug 21 gate 57: Bull-Run Monitor readings at entry (BULLRUN_LONG fills only)
        entry_br_r72: float = None,
        entry_br_above: float = None,
        entry_br_eff: float = None,
        entry_br_off24h: float = None,
        entry_br_door: str = None,   # Aug-23 (20): 'GREEN' (composite) or 'REARM' (re-arm door)
        # Jun 8: gap-expanding relaxation A/B tag (prev2_only-admitted MARGINAL cohort)
        entry_gap_expand_marginal: bool = None,
        # Jun 14: Flip Entry sleeve — when set, this is a NAKED fade-the-block entry
        # (bypasses pattern/multiplier logic, base×registry sizing, FLIP exit model).
        flip_source: str = None,
        flip_cell_mult: float = 1.0,
        flip_cell_lev_mult: float = 1.0,
        flip_cell_tag: str = None,
        flip_exit_mode: str = None,
        # Jun 18: Bull-Long Entry sleeve — when set, this is a REAL build-side LONG (NOT a
        # fade). Tagged entry_strategy="BULL_LONG"; bypasses the long-unmatched + pattern-cell
        # entry blocks; sizes base × size_mult × lev_mult. NOT _is_flip → normal long exit.
        bull_long: bool = False,
        bull_long_size_mult: float = 1.0,
        bull_long_lev_mult: float = 1.0,
        # Jun 19: Bounce-Long sleeve — oversold-washout dead-cat bounce LONG (fades the BTC_RSI_ADX_CROSS
        # oversold short-block). Tagged entry_strategy="BOUNCE_LONG"; same bypasses + NOT _is_flip →
        # normal long exit, exactly like bull_long. Sizes base × size_mult × lev_mult (1.0 / 0.05 = 1×).
        bounce_long: bool = False,
        bounce_long_size_mult: float = 1.0,
        bounce_long_lev_mult: float = 1.0,
        # 🌊 Aug-21 gate 57: Bull-Run Continuation sleeve — GREEN-gated dip-reclaim LONG on
        # top-N COIN pairs. Tagged entry_strategy="BULLRUN_LONG"; bypasses pattern/multiplier
        # cells (sized bullrun_invest_mult × bullrun_lev_mult, default 1×/1×) and the no-trade
        # list (scoped — the ONLY non-MAJORS_PROBE bypass); NOT _is_flip. Exits via the
        # dedicated _bullrun_exit_for stack (BR_-prefixed reasons), never the alt exit chain.
        bullrun_long: bool = False,
        # Jul 13: GAPFLAT probe — this LONG failed ONLY the gap-expanding check (passed the whole
        # rest of the ladder). Opens as a REAL order at ~1x effective leverage (invest_mult x
        # lev_mult from gap_probe_* config), tagged cell_src=GAPFLAT_PROBE (own analytics row;
        # excluded from screen anchors). Capped: only when book is light, 1 concurrent, N/day.
        gap_probe: bool = False,
        # Jul 13 PM: GAPMIN probe — sibling cohort: EMA5-8 gap in [floor, threshold), accelerating
        # but young. Same 1x sizing + guards, tagged GAPMIN_PROBE. Mutually exclusive with gap_probe.
        gapmin_probe: bool = False,
        # Jul 14 SLOPEGATE probe: signal-found candidate killed ONLY by the BTC 5m slope
        # dead-band (macro_trend_flat_threshold_*, April N=4 calibration; measured 133
        # kills/43 opportunities per day). Opens 1x tagged SLOPEGATE_PROBE.
        slopegate_probe: bool = False,
        # Jul 15 RSIADX probe (#4): candidate whose ONLY ladder fail was the Mar-27
        # RSI×ADX cross-filter (840 sole shorts / 66% of short soles in 2 uncensored
        # days; April evidence ~$6, contested). Opens 1x tagged RSIADX_PROBE.
        rsiadx_probe: bool = False,
        deadband_probe: bool = False,
        rsiceil_probe: bool = False,
        gminflat_probe: bool = False,
        adxmax_probe: bool = False,
        dbdown_probe: bool = False,
        # Jul 30 DEEPGAP probe (#13, SHORT-only): momentum-SHORT killed ONLY by the Jul-6
        # deep-gap floor (pair >=1% below 4h trend). Graduated from the retired
        # PASS:MOMENTUM_SHORT_DEEPGAP phantom (N=17 · 71% · Σ+1.85%). Tagged DEEPGAP_PROBE.
        deepgap_probe: bool = False,
        # Jul 30 MAJORS probe (#14, BOTH directions): BTC/ETH candidate that survived the
        # FULL ladder and was blocked only by no_trade_pairs. Strategic scaling experiment
        # (the liquidity cap makes majors the only home for roadmap capital). Tagged MAJORS_PROBE.
        majors_probe: bool = False,
        # Jul 21 ADXMAX2 probe (#10, LONG-only): second rung of the LONG pair-ADX
        # ladder, band (35, 40]. Parallel cohort to ADXMAX (30, 35] — disjoint bands,
        # independent verdicts.
        adxmax2_probe: bool = False,
        # Jul 24 SPIKE_CHASE probe (#11) — PROMOTED TO FULL SIZE Jul 27 (operator one-ship):
        # single-candle RSI-explosion chase, a NEW ENTRY CLASS that bypasses the signal
        # ladder by design (fires only when the ladder produced no signal). Now opens at
        # spike_invest_mult/spike_lev_mult, strategy=SPIKE_CHASE, option-D exit stack.
        spike_chase_probe: bool = False,
        # Jul 27 SPIKE_FADE (operator full ship): the trigger's inverse — legs 1-5 pass but
        # pair ADX > spike_chase_max_adx (mature blowoff, 0/4 rides lifetime) -> SHORT at
        # spike_fade mults, fixed SL spike_fade_sl_pct (NO ATR widening), standard short
        # exit stack, strategy=SPIKE_FADE. Kill: spike_fade_enabled + auto-tripwire.
        spike_fade: bool = False,
        # Jul 31 🏀 SPIKE_BOUNCE (operator full ship, freeze carve-out): single-candle RSI
        # CRASH -> LONG the violent idiosyncratic dump. Full size 1x/1x, fade-mirrored
        # exits (fixed SL spike_bounce_sl_pct NO widening), strategy=SPIKE_BOUNCE.
        # Kill: spike_bounce_enabled + auto-tripwire (spike_bounce_tripwire_pct).
        spike_bounce: bool = False,
        # Jul 27 PM PROMOTION: NONEXP_CALM3D admission — gap-flat/flat+small LONG admitted
        # full-size (Inv 2x/Lev 1x) when SBULL ∧ BTC-ATR <= 0.147 (engine router decided).
        # Bypasses keep-only-unmatched + pattern treatment per the Jul-20 projection spec.
        nonexp_calm3d: bool = False,
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
        if flip_source or bullrun_long:  # Aug 21 gate 57 (post-ship review): sleeve fills were NULL-regime → 100% filed under NEUTRAL, invisible to every BULLISH-filtered view (4th recurrence of the Jun-15 flip bug class)
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
            # Jun 18: the REAL cap-cost — a fully-qualified signal we couldn't open because full.
            try: self._cap_skip_counts["flip" if flip_source else "normal"] += 1
            except Exception: pass
            return None
        # Jun 2: this open sits in the "redeploy band" if it's beyond the normal
        # max_open_positions — only reachable because redeploy raised the ceiling.
        # Recorded as REDEPLOY_OPEN after the Order commits (positive-event counter).
        _is_redeploy_open = _redeploy_on and _open_count_now >= _inv_cfg.max_open_positions

        # Jul 13: GAPFLAT probe caps — a probe must NEVER crowd out the core edge. It opens only
        # when ① the book is light (open count <= max_open_positions - 2 at open time — i.e. a
        # probe only fires with >=2 free slots), and ② concurrent probes < gap_probe_max_open.
        # Jul 13 PM (operator): NO daily budget — at ~1x-lev/$4-a-trade sizing a per-day cap only
        # slows the N>=30 clock; the slot guard + concurrency cap are the real protections.
        # A cap rejection is recorded as PAIR_EMA_GAP_NOT_EXPANDING — identical funnel semantics
        # to the probe being off.
        # Jul 20 (code-review fix): resolve the candidate's FINAL probe tag first (newest
        # wins — same precedence as cell_src below) and run ONLY that probe's cap block.
        # Without this, a dual-flag candidate (e.g. every GMINFLAT is also gap-flat =>
        # gap_probe=True) is capped/counted by the OLDER probe's block — starving the
        # newer cohort's N-clock and mis-attributing the funnel counter.
        # Jul 27: SPIKE_CHASE left the probe fleet (full-size species with its own sizing
        # block below) — no longer part of probe tag precedence or probe caps.
        # Review I1: a spike fire must NEVER be claimed by a co-matching probe band
        # (the top-50 hook passes all probe kwargs computed from indicators) — a probe
        # cap block would silently drop a full-size fire as "book too full".
        _probe_final_tag = None if (spike_chase_probe or spike_fade or spike_bounce or nonexp_calm3d) else (("MAJORS_PROBE" if majors_probe else
                            "DEEPGAP_PROBE" if deepgap_probe else
                            "ADXMAX2_PROBE" if adxmax2_probe else
                            "DBDOWN_PROBE" if dbdown_probe else
                            ("ADXMAX_PROBE" if adxmax_probe else
                             ("GMINFLAT_PROBE" if gminflat_probe else
                              ("RSICEIL_PROBE" if rsiceil_probe else
                               ("DEADBAND_PROBE" if deadband_probe else
                                ("RSIADX_PROBE" if rsiadx_probe else
                                 ("SLOPEGATE_PROBE" if slopegate_probe else
                                  ("GAPFLAT_PROBE" if gap_probe else
                                   ("GAPMIN_PROBE" if gapmin_probe else None))))))))))
        # Jul 30 HOTFIX — STRUCTURAL no-trade invariant (review recommendation after the flip
        # containment leak): a no_trade_pairs order may open ONLY as a MAJORS_PROBE. This is
        # the single choke-point every path funnels through (flips carry flip_source and no
        # probe tag; bull/bounce-longs carry their flags; spikes suppress the probe tag) — so
        # containment no longer depends on each call site remembering the list.
        _nt_open = set(p.strip() for p in (getattr(config.trading_config, 'no_trade_pairs', '') or '').split(',') if p.strip())
        if pair in _nt_open and _probe_final_tag != "MAJORS_PROBE" and not bullrun_long:
            logger.warning(f"[PAIR_NO_TRADE] {pair} {direction}: open_position invariant blocked a non-MAJORS_PROBE order on a track-only pair (source={flip_source or ('BULL_LONG' if bull_long else 'ladder')})")
            try:
                self._record_filter_block("PAIR_NO_TRADE", direction)
            except Exception:
                pass
            return None

        # Jul 30: MAJORS probe caps — mirror of the fleet blocks (same slot guard, own
        # concurrency cap, shared 0.5x/0.05x sizing). Cap rejection records
        # PAIR_NO_TRADE = probe-off semantics.
        if _probe_final_tag == "MAJORS_PROBE" and direction in ("LONG", "SHORT") and not flip_source and not bull_long and not bounce_long:
            _th_mj = config.trading_config.thresholds
            _mj_reason = None
            if not getattr(_th_mj, 'majors_probe_enabled', False):
                _mj_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _mj_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _mj_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "MAJORS_PROBE")))
                if (_mj_open_q.scalar() or 0) >= int(getattr(_th_mj, 'majors_probe_max_open', 2) or 2):
                    _mj_reason = "max concurrent probes open"
            if _mj_reason:
                logger.info(f"[MAJORS_PROBE] {pair} {direction} skipped: {_mj_reason}")
                try:
                    self._record_filter_block("PAIR_NO_TRADE", direction)
                    self._last_pair_block_reason[pair] = "PAIR_NO_TRADE"  # review fix: keep the per-pair surface honest
                except Exception:
                    pass
                return None

        # Jul 30: DEEPGAP probe caps — mirror of the fleet blocks (same slot guard, own
        # concurrency cap, shared 0.5x/0.05x sizing). Cap rejection records
        # MOMENTUM_SHORT_DEEPGAP = probe-off semantics.
        if _probe_final_tag == "DEEPGAP_PROBE" and direction == "SHORT" and not flip_source and not bull_long and not bounce_long:
            _th_dg = config.trading_config.thresholds
            _dgp_reason = None
            if not getattr(_th_dg, 'deepgap_probe_enabled', False):
                _dgp_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _dgp_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _dgp_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "DEEPGAP_PROBE")))
                if (_dgp_open_q.scalar() or 0) >= int(getattr(_th_dg, 'deepgap_probe_max_open', 3) or 3):
                    _dgp_reason = "max concurrent probes open"
            if _dgp_reason:
                logger.info(f"[DEEPGAP_PROBE] {pair} {direction} skipped: {_dgp_reason}")
                try:
                    self._record_filter_block("MOMENTUM_SHORT_DEEPGAP", direction)
                except Exception:
                    pass
                return None

        if _probe_final_tag == "GAPFLAT_PROBE" and direction in ("LONG", "SHORT") and not flip_source and not bull_long and not bounce_long:
            _th_gp = config.trading_config.thresholds
            _gp_reason = None
            if not getattr(_th_gp, 'gap_probe_enabled', False):
                _gp_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _gp_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _gp_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "GAPFLAT_PROBE")))
                if (_gp_open_q.scalar() or 0) >= int(getattr(_th_gp, 'gap_probe_max_open', 3) or 3):
                    _gp_reason = "max concurrent probes open"
            if _gp_reason:
                logger.info(f"[GAPFLAT_PROBE] {pair} {direction} skipped: {_gp_reason}")
                try:
                    self._record_filter_block("PAIR_EMA_GAP_NOT_EXPANDING", direction)
                except Exception:
                    pass
                return None

        # Jul 13 PM: GAPMIN probe caps — mirror of the GAPFLAT block (same slot guard, own
        # concurrency cap shared across BOTH directions, cap rejection recorded as
        # PAIR_EMA_GAP_MIN = probe-off semantics). SHORTs included (operator "both ways").
        if _probe_final_tag == "GAPMIN_PROBE" and direction in ("LONG", "SHORT") and not flip_source and not bull_long and not bounce_long:
            _th_gm = config.trading_config.thresholds
            _gm_reason = None
            if not getattr(_th_gm, 'gapmin_probe_enabled', False):
                _gm_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _gm_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _gm_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "GAPMIN_PROBE")))
                if (_gm_open_q.scalar() or 0) >= int(getattr(_th_gm, 'gapmin_probe_max_open', 3) or 3):
                    _gm_reason = "max concurrent probes open"
            if _gm_reason:
                logger.info(f"[GAPMIN_PROBE] {pair} {direction} skipped: {_gm_reason}")
                try:
                    self._record_filter_block("PAIR_EMA_GAP_MIN", direction)
                except Exception:
                    pass
                return None

        # Jul 14: SLOPEGATE probe caps — mirror of the GAPFLAT/GAPMIN blocks (same slot
        # guard, own concurrency cap shared across BOTH directions, shared 0.5x/0.05x
        # sizing). Cap rejection records BTC_SLOPE_GATE = probe-off semantics.
        if _probe_final_tag == "SLOPEGATE_PROBE" and direction in ("LONG", "SHORT") and not flip_source and not bull_long and not bounce_long:
            _th_sg = config.trading_config.thresholds
            _sg_reason = None
            if not getattr(_th_sg, 'slopegate_probe_enabled', False):
                _sg_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _sg_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _sg_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "SLOPEGATE_PROBE")))
                if (_sg_open_q.scalar() or 0) >= int(getattr(_th_sg, 'slopegate_probe_max_open', 3) or 3):
                    _sg_reason = "max concurrent probes open"
            if _sg_reason:
                logger.info(f"[SLOPEGATE_PROBE] {pair} {direction} skipped: {_sg_reason}")
                try:
                    self._record_filter_block("BTC_SLOPE_GATE", direction)
                except Exception:
                    pass
                return None

        # Jul 15: RSIADX probe caps — mirror of the other probe blocks (slot guard, own
        # concurrency cap shared across BOTH directions, shared 0.5x/0.05x sizing).
        # Cap rejection records PAIR_RSI_ADX_CROSS = probe-off semantics.
        if _probe_final_tag == "RSIADX_PROBE" and direction in ("LONG", "SHORT") and not flip_source and not bull_long and not bounce_long:
            _th_rx = config.trading_config.thresholds
            _rx_reason = None
            if not getattr(_th_rx, 'rsiadx_probe_enabled', False):
                _rx_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _rx_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _rx_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "RSIADX_PROBE")))
                if (_rx_open_q.scalar() or 0) >= int(getattr(_th_rx, 'rsiadx_probe_max_open', 3) or 3):
                    _rx_reason = "max concurrent probes open"
            if _rx_reason:
                logger.info(f"[RSIADX_PROBE] {pair} {direction} skipped: {_rx_reason}")
                try:
                    self._record_filter_block("PAIR_RSI_ADX_CROSS", direction)
                except Exception:
                    pass
                return None

        # Jul 15: DEADBAND probe caps (probe #5, LONG-only by construction) — mirror of the
        # other probe blocks. Cap rejection records LONG_BTC1H_DEADBAND = probe-off semantics.
        if _probe_final_tag == "DEADBAND_PROBE" and direction == "LONG" and not flip_source and not bull_long and not bounce_long:
            _th_db = config.trading_config.thresholds
            _db_reason = None
            if not getattr(_th_db, 'deadband_probe_enabled', False):
                _db_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _db_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _db_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "DEADBAND_PROBE")))
                if (_db_open_q.scalar() or 0) >= int(getattr(_th_db, 'deadband_probe_max_open', 3) or 3):
                    _db_reason = "max concurrent probes open"
            if _db_reason:
                logger.info(f"[DEADBAND_PROBE] {pair} {direction} skipped: {_db_reason}")
                try:
                    self._record_filter_block("LONG_BTC1H_DEADBAND", direction)
                except Exception:
                    pass
                return None

        # Jul 15: RSICEIL probe caps (probe #6, LONG-only by construction) — mirror block.
        # Cap rejection records PAIR_RSI_RANGE = probe-off semantics.
        if _probe_final_tag == "RSICEIL_PROBE" and direction == "LONG" and not flip_source and not bull_long and not bounce_long:
            _th_rc = config.trading_config.thresholds
            _rc_reason = None
            if not getattr(_th_rc, 'rsiceil_probe_enabled', False):
                _rc_reason = "probe disabled"
            elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                _rc_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
            else:
                _rc_open_q = await db.execute(
                    select(func.count(Order.id)).where(and_(
                        Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                        Order.cell_multiplier_source == "RSICEIL_PROBE")))
                if (_rc_open_q.scalar() or 0) >= int(getattr(_th_rc, 'rsiceil_probe_max_open', 3) or 3):
                    _rc_reason = "max concurrent probes open"
            if _rc_reason:
                logger.info(f"[RSICEIL_PROBE] {pair} {direction} skipped: {_rc_reason}")
                try:
                    self._record_filter_block("PAIR_RSI_RANGE", direction)
                except Exception:
                    pass
                return None

        # Jul 20: caps for probes #7-#9 (GMINFLAT / ADXMAX / DBDOWN) — same mirror block;
        # cap rejection records probe-off semantics (the filter that would have blocked).
        for _p_flag, _p_tag, _p_en, _p_max, _p_ctr, _p_dirs in (
            # (Jul 27: SPIKE_CHASE removed from probe caps — full-size species now competes
            # for the normal max-5 like any trade; kill switches live at the fire sites.)
            (gminflat_probe, "GMINFLAT_PROBE", 'gminflat_probe_enabled', 'gminflat_probe_max_open', "PAIR_EMA_GAP_MIN", ("LONG", "SHORT")),
            (adxmax_probe, "ADXMAX_PROBE", 'adxmax_probe_enabled', 'adxmax_probe_max_open', "PAIR_ADX_MAX", ("LONG", "SHORT")),
            (dbdown_probe, "DBDOWN_PROBE", 'dbdown_probe_enabled', 'dbdown_probe_max_open', "LONG_BTC1H_DEADBAND", ("LONG",)),
            # Jul 21 probe #10: second LONG ADX rung (35, 40]
            (adxmax2_probe, "ADXMAX2_PROBE", 'adxmax2_probe_enabled', 'adxmax2_probe_max_open', "PAIR_ADX_MAX", ("LONG",)),
        ):
            # (cap rejection for DBDOWN records the counter but does NOT seed the PASS
            # phantom the probe-off path would have — accepted: seeding is retiring anyway.)
            if _probe_final_tag == _p_tag and direction in _p_dirs and not flip_source and not bull_long and not bounce_long:
                _th_p = config.trading_config.thresholds
                _p_reason = None
                if not getattr(_th_p, _p_en, False):
                    _p_reason = "probe disabled"
                elif _open_count_now > max(0, _inv_cfg.max_open_positions - 2):
                    _p_reason = f"book too full ({_open_count_now} open, last 2 slots reserved)"
                else:
                    _p_open_q = await db.execute(
                        select(func.count(Order.id)).where(and_(
                            Order.status == "OPEN", Order.is_paper == self.is_paper_mode,
                            Order.cell_multiplier_source == _p_tag)))
                    if (_p_open_q.scalar() or 0) >= int(getattr(_th_p, _p_max, 3) or 3):
                        _p_reason = "max concurrent probes open"
                if _p_reason:
                    logger.info(f"[{_p_tag}] {pair} {direction} skipped: {_p_reason}")
                    try:
                        self._record_filter_block(_p_ctr, direction)
                    except Exception:
                        pass
                    return None

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
            try:
                # Jul 15 visibility fix (audit B7): the one-position-per-pair rule was a
                # zero-counter black hole — every repeat signal on a held pair vanished.
                self._record_filter_block("PAIR_HELD", direction)
            except Exception:
                pass
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
                try:
                    # Jul 15 visibility fix (audit B8): re-entry suppression, previously uncounted.
                    self._record_filter_block("COOLDOWN", direction)
                except Exception:
                    pass
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
        # Jul 6 — W2 RE-ENABLE (1h-rising conditioned): admit a W2-matched long (NO C co-match)
        # when BTC 1h slope ≥ long_w2_reenable_1h_min. First matched cell back since Jun-9.
        # History split: W2 × 1h-rising 29·72%WR vs × pullback 14·14%·−0.55 (conditioned cell
        # not refuted); phantoms ≈10·90%·+0.39. Trades at the W2 cell's 1× (not UNMATCHED 2×);
        # tracking rides the Pattern Cell Ship table. Missing 1h → NO admit (fail-closed: the
        # block is the safe state). 🔒 REVERT →99 if live cohort ≤50% WR or net-neg on N≥8.
        _w2_admit = False
        if direction == "LONG" and not flip_source and not bull_long and not bounce_long and _pw2_e and not _pc_any_e:
            try:
                _w2r_raw = getattr(config.trading_config.thresholds, 'long_w2_reenable_1h_min', 99.0)
                _w2r = 99.0 if _w2r_raw is None else float(_w2r_raw)
                _w2_1h = globals().get('_current_btc_1h_slope')
                if _w2r < 99 and _w2_1h is not None and _w2_1h >= _w2r:
                    _w2_admit = True
                    logger.info(f"[W2_REENABLE] {pair}: W2-matched LONG ADMITTED — BTC 1h slope {_w2_1h:+.4f}% >= {_w2r}% (rising flank; cell 1x)")
            except Exception:
                _w2_admit = False
        # Jul 6 — W6 RE-ENABLE (2D: pullback AND thrust). W6 = laggard catch-up; the ONLY
        # era-consistent W6 cell is 1h ≤ −0.05 (BTC dip = the discount) AND stretch ≥ 0.31
        # (the laggard is actually moving): 23·78%WR both-era vs pullback-alone FAILING the
        # era test (eraB 46%). Cell 1× via the W6 pattern cell; fail-closed on missing data.
        # 🔒 REVERT →99 (off) if live cohort ≤50% WR or net-negative on N≥8.
        _w6_admit = False
        if direction == "LONG" and not flip_source and not bull_long and not bounce_long and _pw6_e and not _pc_any_e and not _w2_admit:
            try:
                _w6r_raw = getattr(config.trading_config.thresholds, 'long_w6_reenable_1h_max', 99.0)
                _w6r = 99.0 if _w6r_raw is None else float(_w6r_raw)
                _w6s_raw = getattr(config.trading_config.thresholds, 'long_w6_reenable_stretch_min', 0.31)
                _w6s = 0.31 if _w6s_raw is None else float(_w6s_raw)
                _w6_1h = globals().get('_current_btc_1h_slope')
                if (_w6r < 99 and _w6_1h is not None and _w6_1h <= _w6r
                        and entry_ema5_stretch is not None and entry_ema5_stretch >= _w6s):
                    _w6_admit = True
                    logger.info(f"[W6_REENABLE] {pair}: W6-matched LONG ADMITTED — BTC 1h {_w6_1h:+.4f}% <= {_w6r}% AND stretch {entry_ema5_stretch:.3f} >= {_w6s} (dip+thrust; cell 1x)")
            except Exception:
                _w6_admit = False
        if direction == "LONG" and not flip_source and not bull_long and not bounce_long and not spike_chase_probe and not spike_fade and not spike_bounce and not nonexp_calm3d and getattr(config.trading_config.thresholds, 'long_unmatched_only', False) and (_pc_any_e or _pw_any_e) and not _w2_admit and not _w6_admit:
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
            # Jun 29: refine to the SPECIFIC matched patterns (e.g. "W6", "C6+W6") so high-value
            # flip candidates break out for OOS phantom tracking. Lead candidate W6→flip-short
            # (BTC bear-tailwind long fails → short with the bear): screen +0.199%/65%WR/N=26 but
            # all in ONE bear window (May21-Jun5) → phantom-validate across ≥2 bear episodes before
            # any capital. Family (C/W/C+W) still derivable from the codes; nothing keys off the
            # old exact values (verified). Query PhantomFlip rows by cohort LIKE '%W6%' to read out.
            _um_pats = "+".join(p for p, m in (
                ("C1", _pc1_e), ("C2", _pc2_e), ("C3", _pc3_e), ("C4", _pc4_e), ("C5", _pc5_e),
                ("C6", _pc6_e), ("C7", _pc7_e), ("C8", _pc8_e), ("C9", _pc9_e),
                ("W1", _pw1_e), ("W2", _pw2_e), ("W3", _pw3_e), ("W4", _pw4_e), ("W5", _pw5_e), ("W6", _pw6_e)) if m)
            _um_cohort = _um_pats or ("C+W" if (_pc_any_e and _pw_any_e) else ("C" if _pc_any_e else "W"))
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
            # Jul 4 (operator) — SAME-direction phantom of the blocked matched long itself
            # (mode='PASS'): measures whether the Jun-9 LONG_UNMATCHED_ONLY block is over-
            # restrictive in a bull tape. The Jun-9 evidence predates the current exit stack
            # (which turned unmatched longs -0.10% -> +0.31%), so matched longs were never
            # tested under today's bot. Re-enable gates per pattern cell in CURRENT_STATE.
            _seed_phantom_flip(pair, current_price, "LONG", "PASS:LONG_UNMATCHED_ONLY", cohort=_um_cohort, entry_fields=_um_ef, mode='PASS')
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
        # Jul 27 (review C1/I4): SPIKE species take NO pattern-cell treatment — the
        # UNMATCHED-SHORT block would silently strangle every fade (new signature =
        # unmatched by construction), and any pattern fixed-TP/SL would pre-empt the
        # option-D stack (a +0.10 pattern TP amputates a +17 rider). Router owns spikes.
        # Aug 21 gate 57 (review I1): BULLRUN same exemption — a pattern rule must never block a
        # sleeve fill or stamp fixed TP/SL that would pre-empt the dedicated BR_ exit stack.
        if spike_chase_probe or spike_fade or spike_bounce or nonexp_calm3d or bullrun_long:
            _pcell_inv, _pcell_lev, _pcell_src = None, None, None
            _pcell_fixed_tp, _pcell_fixed_sl, _pcell_block = None, None, False
        # C1 SHORT breadth-scoped de-mux (Jun 28): the C1 capitulation-chase 2× only earns its
        # multiplier in the 70–85 bear-breadth band (cross-pool 73–76% WR / +avg; both tails 50–60%
        # WR / −avg where the 2× merely amplifies fat-tail DOA losers). When a C1 SHORT cell would
        # size >1×, KEEP it only inside [lo, hi), else DE-MUX to 1× (sizing only — entry NOT blocked;
        # a 50%-WR cohort must be de-amplified, not blocked). 06-28: AAVE/HYPE −$242/−$186 → +$214.
        if (direction == "SHORT" and _pcell_src and 'C1' in str(_pcell_src) and (_pcell_inv or 1.0) > 1.0
                and getattr(config.trading_config.thresholds, 'c1_short_demux_breadth_enabled', False)):
            _clo = float(getattr(config.trading_config.thresholds, 'c1_short_demux_breadth_lo', 70.0) or 0.0)
            _chi = float(getattr(config.trading_config.thresholds, 'c1_short_demux_breadth_hi', 85.0) or 0.0)
            if entry_bear_pct is None or not (_clo <= entry_bear_pct < _chi):
                logger.info(f"[C1_DEMUX_BREADTH] {pair} SHORT: bear%={entry_bear_pct} outside [{_clo},{_chi}) "
                            f"→ de-mux C1 {_pcell_inv}x/{_pcell_lev}x → 1x ({_pcell_src})")
                _pcell_inv, _pcell_lev = 1.0, 1.0
        # UNMATCHED LONG crowded-entry de-mux (Jul 10): the UNMATCHED 2× only earns its multiplier
        # below pair_volume_ratio 0.90 — the ≥0.90 zone is a ✗ HARMFUL sub-cell (pool 10 trades,
        # 60% WR, net-NEGATIVE at both sizings; below 0.90 the sleeve ran 29W/3L). Mechanism:
        # PVR ≥ 0.90 = volume burst already happened = buying someone's exit (crowded entry —
        # LDO/HYPE/ME/PYTH class). Sizing only, entry NOT blocked (a 60%-WR cohort is de-amplified,
        # never blocked). 0 = off. Fail-open on missing PVR.
        if (direction == "LONG" and _pcell_src and 'UNMATCHED' in str(_pcell_src) and (_pcell_inv or 1.0) > 1.0):
            _upv_max = float(getattr(config.trading_config.thresholds, 'long_unmatched_mult_pvr_max', 0.0) or 0.0)
            # Aug-10 CROWD-SPRINT DE-MUX (operator ship after the 3-era "Rest" hunt): crowded
            # global tape ∧ BTC sprinting = the FOMO window — the fan-expansion trigger fires
            # everywhere at once (beta, not idiosyncratic flow), so the unmatched THESIS is
            # unverifiable there; conviction sizing withdrawn, entry kept (never block what
            # baseline tape can turn into ACT +$263). 3-era: window 19·58%·−$714 pooled
            # (B1 6·17%·−$823 · BANK Aug-10 out-of-sample catch · BASE cost = ACT alone).
            # TAKES PRECEDENCE over the quiet-PVR boost (quiet book under a loud market is
            # still an unreadable backdrop). Either threshold 0 = leg off; fail-open on None.
            _us_gvr_min = float(getattr(config.trading_config.thresholds, 'long_unmatched_sprint_demux_gvr_min', 0.0) or 0.0)
            _us_slp_min = float(getattr(config.trading_config.thresholds, 'long_unmatched_sprint_demux_b20slope_min', 0.0) or 0.0)
            _us_slp = globals().get('_btc_ema20_slope_pct')
            if (_us_gvr_min > 0 and _us_slp_min > 0 and entry_global_volume_ratio is not None
                    and _us_slp is not None and entry_global_volume_ratio > _us_gvr_min
                    and float(_us_slp) > _us_slp_min):
                logger.info(f"[UNMATCHED_SPRINT_DEMUX] {pair} LONG: global-vol {entry_global_volume_ratio:.2f} > {_us_gvr_min} "
                            f"∧ BTC-slope {float(_us_slp):+.3f} > {_us_slp_min} (crowd-sprint window, thesis unverifiable) "
                            f"→ de-mux UNMATCHED {_pcell_inv}x/{_pcell_lev}x → 1.0x/1.0x — re-sim read row")
                _pcell_inv, _pcell_lev = 1.0, 1.0
            elif _upv_max > 0 and entry_pair_volume_ratio is not None and entry_pair_volume_ratio >= _upv_max:
                # Jul 26 (operator patron fix): configurable de-mux targets (default 1.0/1.0 =
                # the original full de-mux). <=0 coerced to 1.0 — a zero would zero the position.
                _dm_inv = float(getattr(config.trading_config.thresholds, 'long_unmatched_demux_inv_mult', 1.0) or 1.0)
                _dm_lev = float(getattr(config.trading_config.thresholds, 'long_unmatched_demux_lev_mult', 1.0) or 1.0)
                if _dm_inv <= 0: _dm_inv = 1.0
                if _dm_lev <= 0: _dm_lev = 1.0
                logger.info(f"[UNMATCHED_DEMUX_PVR] {pair} LONG: pair-vol ratio {entry_pair_volume_ratio:.2f} >= {_upv_max} "
                            f"(crowded entry) → de-mux UNMATCHED {_pcell_inv}x/{_pcell_lev}x → {_dm_inv}x/{_dm_lev}x")
                _pcell_inv, _pcell_lev = _dm_inv, _dm_lev
            else:
                # Jul 26 QUIET BOOST (the opposite end of the same PVR ladder; discipline-override
                # at 19-0, tight revert in config.py). PVR < quiet threshold → invest mult up
                # (take-the-max, replaces the 2x; LEVERAGE UNTOUCHED — BE-compat gate). Fail-open
                # on missing PVR (stays at the cell's 2x).
                _uq_max = float(getattr(config.trading_config.thresholds, 'long_unmatched_quiet_pvr_max', 0.0) or 0.0)
                _uq_mult = float(getattr(config.trading_config.thresholds, 'long_unmatched_quiet_mult', 0.0) or 0.0)
                _uq_lev = float(getattr(config.trading_config.thresholds, 'long_unmatched_quiet_lev_mult', 1.0) or 1.0)
                if (_uq_max > 0 and _uq_mult > 0 and entry_pair_volume_ratio is not None
                        and entry_pair_volume_ratio < _uq_max and _uq_mult > (_pcell_inv or 1.0)):
                    # Review fix (Jul 26): lev is take-the-max too — the quiet field may only
                    # RAISE leverage (BE-compat-gated, default 1.0 = untouched); an unconditional
                    # overwrite could silently DOWNGRADE the best cohort if the cell's lev is
                    # ever raised above the quiet field.
                    _uq_lev_eff = max(_pcell_lev or 1.0, _uq_lev)
                    logger.info(f"[UNMATCHED_QUIET_BOOST] {pair} LONG: pair-vol ratio {entry_pair_volume_ratio:.2f} < {_uq_max} "
                                f"(quiet book) → UNMATCHED invest {_pcell_inv}x → {_uq_mult}x"
                                + (f", lev {_pcell_lev}x → {_uq_lev_eff}x" if _uq_lev_eff != (_pcell_lev or 1.0) else ""))
                    _pcell_inv = _uq_mult
                    _pcell_lev = _uq_lev_eff
        # Jun 8: pattern-cell BLOCK action — skip the entry entirely (no order, no exchange
        # call; we're before position sizing / Order creation). Counter PATTERN_CELL_BLOCK.
        if _pcell_block and not flip_source and not bull_long and not bounce_long:
            logger.info(f"[PATTERN_CELL_BLOCK] {pair} {direction}: entry blocked by pattern-cell rule (signature={_pcell_src})")
            try:
                self._record_filter_block("PATTERN_CELL_BLOCK", direction)
            except Exception:
                pass
            return None

        # MOMENTUM-SHORT W1 regime block — Jun 30, 2026 (see config.py momentum_short_w1_block_regimes).
        # W1 ("HighConv trend") fired as a SHORT drains specifically in HEALTHY_BEAR. Evidence (2
        # direction-consistent windows): SCREENED_BASELINE (06-16→28) + 06-29/30 batch → W1 mom-short
        # HEALTHY_BEAR N=20 / 40%WR / -$650 / avg -0.265%, while the non-W1 mom-short CONTROL in the
        # SAME regime is breakeven+ (N=7 / +$24 / +0.014%) → the discriminator is W1, not the regime
        # (confound check passed). Loss is diffuse (no pair ≥60%). STRONG_BEAR W1 WINS (N=4/75%/+$229)
        # → EXEMPT: only regimes listed here block. Momentum-shorts reach this path; flips bypass it
        # (gated upstream by _flip_filters). DISCIPLINE-OVERRIDE ship (N=20 < 30 promotion gate) →
        # tracked via the phantom below. TIGHT REVERT: clear the regime list (→'') if this block's
        # phantom (LONG fade) goes net-NEGATIVE on N≥10 fresh (= the blocked shorts would have WON).
        # Empty list = filter off. Fail-open: missing regime → no block.
        # Jul 27 (review-2 I-1): spike_fade exempt — a fade is not a momentum short;
        # the router owns it (and this block's phantom would seed a LONG flip into a pump).
        if direction == "SHORT" and not flip_source and not bull_long and not bounce_long and not spike_fade and _pw1_e:
            _w1blk = {s.strip() for s in (getattr(config.trading_config.thresholds, 'momentum_short_w1_block_regimes', '') or '').split(',') if s.strip()}
            if entry_btc_regime in _w1blk:
                logger.info(f"[MOMENTUM_SHORT_W1_REGIME] {pair}: W1 momentum SHORT blocked — regime {entry_btc_regime} in block-list {sorted(_w1blk)}")
                try:
                    self._record_filter_block("MOMENTUM_SHORT_W1_REGIME", "SHORT")
                except Exception:
                    pass
                # Phantom fade (blocked SHORT → LONG) so the block stays observable for the revert gate:
                # a net-NEGATIVE LONG fade means the blocked short would have won → revert signal.
                try:
                    _w1_ef = {k: v for k, v in {
                        'entry_gap': entry_gap, 'entry_ema_gap_5_8': entry_ema_gap_5_8, 'entry_ema_gap_8_13': entry_ema_gap_8_13,
                        'entry_ema5_stretch': entry_ema5_stretch, 'entry_rsi': entry_rsi, 'entry_adx': entry_adx,
                        'entry_adx_delta': entry_adx_delta, 'entry_atr_pct': entry_atr_pct, 'entry_range_position': entry_range_position,
                        'entry_pair_ema20_ema50_gap_pct': entry_pair_ema20_ema50_gap_pct, 'entry_dist_from_ema13_pct': entry_dist_from_ema13_pct,
                        'entry_btc_adx': entry_btc_adx, 'entry_btc_rsi': entry_btc_rsi, 'entry_btc_atr_pct': entry_btc_atr_pct,
                        'entry_quality_score': entry_quality_score, 'entry_bear_pct': entry_bear_pct,
                    }.items() if v is not None}
                    _seed_phantom_flip(pair, current_price, "SHORT", "MOMENTUM_SHORT_W1_REGIME", entry_fields=_w1_ef)
                except Exception:
                    pass
                return None

        # MOMENTUM-SHORT high-pair-volume block — Jun 30, 2026 (see config.py momentum_short_pair_vol_max).
        # Block a momentum SHORT when entry pair-volume ratio >= the threshold. Mechanism: shorting into HIGH
        # pair volume = climactic/exhaustive move that bounces (no follow-through); LOW pair volume = orderly
        # continuation that follows through. The ONLY entry separator that does NOT invert across periods:
        # pair_vol<1.0 wins in BOTH (69%/+$392 recent, 64%/+$449 ≤06-13); pair_vol>=1.0 loses recent / net-neg
        # old → blocking >=1.0 is +EV in both windows. Replaces the reverted W1-regime block. Momentum-only
        # (flips bypass via _flip_filters). MAX semantics (block at/above); the legacy pair_volume_threshold_short
        # is a MIN and stays OFF. Counter MOMENTUM_SHORT_PAIRVOL. 0 = off. Fail-open: missing pair-vol → no block.
        # Jul 27 (review-2 C-1): spike_fade MUST be exempt — the spike trigger's leg 5
        # REQUIRES candle vol >= 5x avg20, so every fade arrives with PVR >= 5 >= this
        # block's 1.0 max; without the exemption the entire fade species is unreachable.
        # (The fade deliberately shorts the climactic move this filter avoids — that IS
        # its thesis, protected by the fixed -0.70 stop + tripwire, not by this block.)
        if direction == "SHORT" and not flip_source and not bull_long and not bounce_long and not spike_fade:
            _pvmax = float(getattr(config.trading_config.thresholds, 'momentum_short_pair_vol_max', 0.0) or 0.0)
            if _pvmax > 0 and entry_pair_volume_ratio is not None and entry_pair_volume_ratio >= _pvmax:
                logger.info(f"[MOMENTUM_SHORT_PAIRVOL] {pair}: momentum SHORT blocked — pair-vol {entry_pair_volume_ratio:.2f} >= {_pvmax} (climactic/exhaustion)")
                try:
                    self._record_filter_block("MOMENTUM_SHORT_PAIRVOL", "SHORT")
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
            # Jun 16: per-source CONDITIONAL cell multiplier (e.g. FAN's qs/bear/range winner cell).
            # Multiplies the registry size AND leverage and carries a distinct cell_src so it groups
            # as its OWN row in Multiplier Cell Performance. Jun 26: a flip_cell_tag (FAN winner cell)
            # forces the distinct source EVEN at 1× so the inert track-only cell still gets its own
            # row; the ×/L size tag is appended only when the multiplier actually differs from 1.0.
            if flip_cell_tag or (flip_cell_mult and flip_cell_mult != 1.0) or (flip_cell_lev_mult and flip_cell_lev_mult != 1.0):
                cell_mult = cell_mult * (flip_cell_mult or 1.0)
                cell_lev_mult = cell_lev_mult * (flip_cell_lev_mult or 1.0)
                _tag = f"×{flip_cell_mult:g}" if flip_cell_mult and flip_cell_mult != 1.0 else ""
                _tag += f"L{flip_cell_lev_mult:g}" if flip_cell_lev_mult and flip_cell_lev_mult != 1.0 else ""
                cell_src = f"FLIP:{flip_source}{flip_cell_tag or ''}{_tag}"
            _mult_target = "both"
            # Re-apply the hard caps (the clamp above ran before this override).
            if cell_mult > _inv_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested inv={cell_mult}x, hard-capped to inv={_inv_cap}x")
                cell_mult = _inv_cap
            if cell_lev_mult > _lev_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested lev={cell_lev_mult}x, hard-capped to lev={_lev_cap}x")
                cell_lev_mult = _lev_cap
            # Jun 19: lev floor scoped to 0.05 (was 0.5) so a flip source can DE-lever to 1× observation
            # (0.05 × 20× base = 1×, e.g. PAIR_RSI_OB:1.0:0.05). Size floor stays 0.5. A source at lev 1.0
            # (FAN) is unaffected (max(0.05,1.0)=1.0). Mirrors the bull/bounce-long branches.
            cell_mult, cell_lev_mult = max(0.5, cell_mult), max(0.05, cell_lev_mult)

        # Jun 18: Bull-Long sleeve overrides ALL momentum multipliers — a real build-side long
        # sizes at base × bull_long_size_mult × bull_long_lev_mult (no pattern/RSI×ADX boost).
        # Defaults are 1.0/1.0 (no amplification, normal leverage). Carries a distinct cell_src
        # so it groups as its OWN row in Multiplier Cell Performance. Hard caps below still clamp.
        if bull_long:
            cell_mult = bull_long_size_mult or 1.0
            cell_lev_mult = bull_long_lev_mult or 1.0
            cell_src = "BULL_LONG"
            _mult_target = "both"
            # Jun 23: flat fixed SL for the bull-long revival TEST. Re-sim showed the flat -0.70
            # SL beat the live ATR-widened -1.20 exit for this cohort (-0.151 → -0.078). Stamp it
            # onto the order so the existing PATTERN_FIXED_SL exit path enforces a flat -0.70
            # (fires before the normal -1.20 can engage). Negative = active; 0 = off (normal exit).
            _bl_sl = getattr(config.trading_config.thresholds, 'bull_long_fixed_sl', 0.0) or 0.0
            if _bl_sl < 0:
                _pcell_fixed_sl = _bl_sl
            if cell_mult > _inv_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested inv={cell_mult}x, hard-capped to inv={_inv_cap}x")
                cell_mult = _inv_cap
            if cell_lev_mult > _lev_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested lev={cell_lev_mult}x, hard-capped to lev={_lev_cap}x")
                cell_lev_mult = _lev_cap
            # Jun 18: bull-long is an OBSERVATION sleeve — allow it to DE-lever well below
            # the 0.5 floor (0.05 × 20× base = 1× live) so it keeps collecting WR / range-pos
            # data at minimal $ risk while we hunt the clean 2nd variable. Size floor stays 0.5.
            cell_mult, cell_lev_mult = max(0.5, cell_mult), max(0.05, cell_lev_mult)

        # Jun 19: Bounce-Long sleeve — same override as bull-long (own cell_src row, de-lever to 1×
        # for observation). Size floor stays 0.5; lev floor scoped to 0.05 (0.05 × 20× = 1× live).
        if bounce_long:
            cell_mult = bounce_long_size_mult or 1.0
            cell_lev_mult = bounce_long_lev_mult or 1.0
            cell_src = "BOUNCE_LONG"
            _mult_target = "both"
            if cell_mult > _inv_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested inv={cell_mult}x, hard-capped to inv={_inv_cap}x")
                cell_mult = _inv_cap
            if cell_lev_mult > _lev_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested lev={cell_lev_mult}x, hard-capped to lev={_lev_cap}x")
                cell_lev_mult = _lev_cap
            cell_mult, cell_lev_mult = max(0.5, cell_mult), max(0.05, cell_lev_mult)

        # 🌊 Aug-21 gate 57: Bull-Run sleeve sizing — same absolute-assign override pattern as
        # BULL_LONG/BOUNCE_LONG (a sleeve fill must never be re-multiplied by UNMATCHED/pattern
        # cells). Inv/Lev mults are UI fields (bullrun_invest_mult/bullrun_lev_mult, default
        # 1×/1× — 🔒 frozen until ≥2 profitable episodes per gate 57).
        if bullrun_long:
            _th_br = config.trading_config.thresholds
            cell_mult = max(0.1, float(getattr(_th_br, 'bullrun_invest_mult', 1.0) or 1.0))
            cell_lev_mult = max(0.05, float(getattr(_th_br, 'bullrun_lev_mult', 1.0) or 1.0))
            cell_src = "BULLRUN"
            _mult_target = "both"
            if cell_mult > _inv_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested inv={cell_mult}x, hard-capped to inv={_inv_cap}x")
                cell_mult = _inv_cap
            if cell_lev_mult > _lev_cap:
                logger.info(f"[CELL_MULT_CAPPED_HARD] {pair} {direction}: {cell_src} requested lev={cell_lev_mult}x, hard-capped to lev={_lev_cap}x")
                cell_lev_mult = _lev_cap

        # Jul 13: GAPFLAT probe sizing — overrides ALL multiplier cells (a probe must never be
        # 2x'd by the UNMATCHED cell); own cell_src row (rides Multiplier Cell Performance + CSV
        # for free); de-levers to ~1x effective (invest 0.5x, lev 0.05x x 20x base = 1x live).
        # Same observation-sleeve pattern as BULL_LONG / BOUNCE_LONG.
        if ((gap_probe or gapmin_probe or slopegate_probe or rsiadx_probe or deadband_probe or rsiceil_probe
             or gminflat_probe or adxmax_probe or dbdown_probe or adxmax2_probe or deepgap_probe or majors_probe) and direction in ("LONG", "SHORT")
                and not flip_source and not bull_long and not bounce_long and not bullrun_long and not spike_chase_probe and not spike_fade and not spike_bounce and not nonexp_calm3d):
            _th_gp2 = config.trading_config.thresholds
            cell_mult = min(1.0, max(0.1, float(getattr(_th_gp2, 'gap_probe_invest_mult', 0.5) or 0.5)))
            cell_lev_mult = min(1.0, max(0.05, float(getattr(_th_gp2, 'gap_probe_lev_mult', 0.05) or 0.05)))
            # Tag precedence = NEWEST probe wins (RSICEIL Jul15(3) > DEADBAND Jul15(2) >
            # RSIADX Jul15 > SLOPEGATE Jul14 > GAPFLAT/GAPMIN Jul13). Rationale: a dual-flag candidate was still blocked under
            # yesterday's stack (the older probe's fall-through alone didn't free it), so it
            # exists only because the newest probe opened its last blocker. Tagging it with
            # the newest probe keeps every OLDER running A/B cohort's admission criteria
            # frozen mid-experiment. NOT chronological ladder order (gap kills before RSIADX
            # kills before the engine slope gate) — verdict-time rule: slice each cohort on
            # the other probes' dimensions (gap band / slope band / RSI x ADX) before shipping.
            # Jul 20: probes #7-#9 prepend (newest wins); single source of truth with the
            # cap blocks above (code-review fix).
            cell_src = _probe_final_tag or "GAPMIN_PROBE"
            _mult_target = "both"
            logger.info(f"[{cell_src}] {pair} {direction}: opening probe at inv={cell_mult}x lev={cell_lev_mult}x (~1x effective)")

        # ── Jul 27: 🚀 SPIKE full-size sizing (operator one-ship) — overrides ALL multiplier
        # cells (same absolute-assign pattern as the probe block; a spike must never be
        # re-multiplied by UNMATCHED/quiet-boost). CHASE = Inv 2x/Lev 1x; FADE = Inv 1x/Lev 1x.
        # The 0.1% liquidity cap below is the TRUE governor (binds on every fire).
        if spike_chase_probe or spike_fade or spike_bounce or nonexp_calm3d:
            _th_sp = config.trading_config.thresholds
            if nonexp_calm3d:
                # Jul 27 PM promotion: NONEXP_CALM3D cell (operator Inv 2x / Lev 1x)
                cell_mult = max(0.1, float(getattr(_th_sp, 'nonexp_calm3d_invest_mult', 2.0) or 2.0))
                cell_lev_mult = max(0.05, float(getattr(_th_sp, 'nonexp_calm3d_lev_mult', 1.0) or 1.0))
                cell_src = "NONEXP_CALM3D"
            elif spike_bounce:
                # Jul 31 🏀 bounce = full size Inv 1x/Lev 1x (operator: normal trade, no probe cap)
                cell_mult = max(0.1, float(getattr(_th_sp, 'spike_bounce_invest_mult', 1.0) or 1.0))
                cell_lev_mult = max(0.05, float(getattr(_th_sp, 'spike_bounce_lev_mult', 1.0) or 1.0))
                cell_src = "SPIKE_BOUNCE"
            elif spike_fade:
                cell_mult = max(0.1, float(getattr(_th_sp, 'spike_fade_invest_mult', 1.0) or 1.0))
                cell_lev_mult = max(0.05, float(getattr(_th_sp, 'spike_fade_lev_mult', 1.0) or 1.0))
                cell_src = "SPIKE_FADE"
            else:
                cell_mult = max(0.1, float(getattr(_th_sp, 'spike_invest_mult', 2.0) or 2.0))
                cell_lev_mult = max(0.05, float(getattr(_th_sp, 'spike_lev_mult', 1.0) or 1.0))
                cell_src = "SPIKE_CHASE"
            _mult_target = "both"
            logger.info(f"[{cell_src}] {pair} {direction}: full-size open at inv={cell_mult}x lev={cell_lev_mult}x (liquidity cap governs)")

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
            # 🔒 Aug-3 SPIKE LOW-VOL CAP RAISE (operator override; 0.1→0.2 Aug-3, 0.2→0.3
            # Aug-18 gate 50 — see config comment: increment UNVERIFIABLE UNTIL LIVE):
            # spike species on pairs below the frozen $10M slice boundary use the raised
            # pct (config value, currently 0.3%); everything else (momentum/flips/spikes
            # >= $10M) keeps the global pct. Raised-cap throttles stamp LIQ2 (Liquidity
            # Sizing table + CSV = the revert-gate instrument).
            _liq_pct_eff = _liq_pct
            _liq_reason_tag = 'LIQ'
            _sp_lowvol_pct = float(getattr(_inv_cfg, 'spike_lowvol_liq_cap_pct', 0.0) or 0.0)
            _sp_lowvol_thr = float(getattr(_inv_cfg, 'spike_lowvol_threshold_usd', 0.0) or 0.0)
            if ((spike_fade or spike_bounce or spike_chase_probe)
                    and _sp_lowvol_pct > 0 and _sp_lowvol_thr > 0
                    and entry_pair_volume_24h_usd and 0 < entry_pair_volume_24h_usd < _sp_lowvol_thr):
                _liq_pct_eff = _sp_lowvol_pct
                _liq_reason_tag = 'LIQ2'
            _liq_cap = None
            if _liq_pct_eff > 0 and entry_pair_volume_24h_usd and entry_pair_volume_24h_usd > 0:
                _liq_cap = (_liq_pct_eff / 100.0) * entry_pair_volume_24h_usd
            if _liq_ceiling > 0:
                _liq_cap = _liq_ceiling if _liq_cap is None else min(_liq_cap, _liq_ceiling)
            if _liq_cap is not None and _final_notional > _liq_cap:
                _final_notional = _liq_cap
                _cap_reason = _liq_reason_tag
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
                    _cap_reason = 'GROSS' if _cap_reason is None else f'{_liq_reason_tag}+GROSS'
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
            # Jul 15 visibility fix (audit B15): a balance-starved bot previously opened
            # nothing SILENTLY — indistinguishable from strict filters in every report.
            logger.warning(f"[NO_BALANCE] {pair} {direction}: computed investment ${investment:.2f} <= 0 — balance exhausted, entry dropped")
            try:
                self._record_filter_block("NO_BALANCE", direction)
            except Exception:
                pass
            return None
        
        # Calculate notional and quantity
        notional_value = investment * leverage
        quantity = notional_value / current_price
        
        # Determine fee rate and entry type
        tc = config.trading_config
        maker_enabled = tc.maker_entry_enabled
        # Jul 30 (operator-directed): SPIKE species BYPASS the maker entry — direct taker.
        # Evidence: 18/18 lifetime spike fires were TAKER_FALLBACK (0 maker fills — the limit
        # posts at the passive side while the spike runs away from it by definition), so every
        # fire paid up to the full 20s timeout in ADVERSE DRIFT and captured zero fee savings.
        # On this species latency is the whole game: ERA's entire trade lasted 19s; a fade
        # sells a decaying top (20s late = systematically worse fill); a chase 20s late enters
        # closer to/past the 1.5xATR stretch guard. Fee cost of the fix: taker 0.045 vs maker
        # 0.018 = +0.027%/entry — noise vs 20s of drift on a >=0.5%/5min candle. Momentum/flip
        # entries keep maker (slow-forming signals; book-wide maker record 87 fills/+$80 saved).
        if maker_enabled and (spike_chase_probe or spike_fade or spike_bounce):
            maker_enabled = False
            logger.info(f"[SPIKE_TAKER] {pair} {direction}: maker entry bypassed for spike species — direct taker (latency > fee)")
        maker_fee_rate = getattr(tc, 'maker_fee', tc.trading_fee)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)

        entry_order_type = "TAKER"
        entry_fee = notional_value * taker_fee_rate
        
        # Execute trade
        binance_order_id = None
        actual_price = current_price
        _bk_algo_id = None  # Aug-11 🛡 broker backstop algoId (live only)
        
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
        # Aug-11 🛡 BROKER BACKSTOP (live only): resting exchange-side STOP_MARKET via the
        # Algo Order API — dead-man's brake for deploy/crash/WS-starve/ban windows. WIDE by
        # design (broker_backstop_pct from the ACTUAL fill px) so it never races the software
        # stops; fires only when the bot cannot act. Failure = position still opens on
        # software stops, but counted + CRITICAL (never silent — the flip-bug lesson).
        if (not self.is_paper_mode and actual_price and actual_price > 0
                and bool(getattr(config.trading_config.thresholds, 'broker_backstop_enabled', False))):
            try:
                _bk_pct = float(getattr(config.trading_config.thresholds, 'broker_backstop_pct', 2.5) or 2.5)
                _bk_trigger = actual_price * (1 - _bk_pct / 100.0) if direction == "LONG" else actual_price * (1 + _bk_pct / 100.0)
                _bk_algo_id = await binance_service.place_backstop_stop(pair, direction, _bk_trigger)
                if _bk_algo_id:
                    logger.info(f"[BACKSTOP_PLACED] {pair} {direction}: algoId={_bk_algo_id} trigger={_bk_trigger:.6g} ({_bk_pct}% from fill {actual_price})")
                else:
                    self._record_filter_block("BACKSTOP_PLACE_FAILED", direction)
                    logger.critical(f"[BACKSTOP_PLACE_FAILED] {pair} {direction}: position runs on SOFTWARE stops only — investigate SAME DAY")
            except Exception as _bk_err:
                self._record_filter_block("BACKSTOP_PLACE_FAILED", direction)
                logger.critical(f"[BACKSTOP_PLACE_FAILED] {pair} {direction}: {_bk_err}")
        order = Order(
            binance_order_id=binance_order_id,
            backstop_algo_id=_bk_algo_id,
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
            entry_pair_age_days=entry_pair_age_days,
            entry_br_r72=entry_br_r72,
            entry_br_above=entry_br_above,
            entry_br_eff=entry_br_eff,
            entry_br_off24h=entry_br_off24h,
            entry_br_door=entry_br_door,
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
            # Jun 18: BULL_LONG tag for the build-side sleeve (real long, normal exit; NOT _is_flip)
            entry_strategy=("BULLRUN_LONG" if bullrun_long else ("SPIKE_BOUNCE" if spike_bounce else ("SPIKE_FADE" if spike_fade else ("SPIKE_CHASE" if spike_chase_probe else ("BOUNCE_LONG" if bounce_long else ("BULL_LONG" if bull_long else (f"FLIP:{flip_source}" if flip_source else "MOMENTUM"))))))),
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
                'opened_at': order.opened_at,          # Jul 28 review M-5: spike stale-kill/trail live from t0
                'entry_atr_pct': entry_atr_pct,        # (both were previously added only at the first cache refresh)
                'entry_strategy': ("BULLRUN_LONG" if bullrun_long else ("SPIKE_BOUNCE" if spike_bounce else ("SPIKE_FADE" if spike_fade else ("SPIKE_CHASE" if spike_chase_probe else ("BOUNCE_LONG" if bounce_long else ("BULL_LONG" if bull_long else (f"FLIP:{flip_source}" if flip_source else "MOMENTUM"))))))),  # Jun 15: flips exit via realtime stack; Jul 27: SPIKE_* gate option-D / fixed-SL branches; Aug 21: BULLRUN_LONG (gate 57, dedicated BR_ exits)
                'entry_ema5_stretch': entry_ema5_stretch,  # LEASH SHADOW (May 30) — stretch-exit entry anchor
                'entry_price': actual_price,
                'quantity': quantity,
                'entry_fee': entry_fee,
                'confidence': confidence,
                # Jul 27 spike ship: fixed SLs — CHASE spike_sl_pct (−1.2, winner-breath
                # margin), FADE spike_fade_sl_pct (−0.70, squeeze bound). ATR widening is
                # skipped for both in the check paths (entry_strategy-gated).
                'stop_loss': (float(getattr(config.trading_config.thresholds, 'spike_sl_pct', -1.2) or -1.2) if spike_chase_probe
                              else float(getattr(config.trading_config.thresholds, 'spike_fade_sl_pct', -1.50) or -1.50) if spike_fade
                              else float(getattr(config.trading_config.thresholds, 'spike_bounce_sl_pct', -0.70) or -0.70) if spike_bounce
                              else conf_config.stop_loss),
                'current_tp_level': 1,
                'peak_pnl': 0.0,
                'trough_pnl': 0.0,
                # Jul 27 option-D state (SPIKE_CHASE only): armed = 5m RSI(12) >= arm
                # threshold seen since entry; rsi_max = running maximum (L2 exit anchor).
                'spike_armed': False,
                'spike_rsi_max': None,
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
            if nonexp_calm3d:
                # Aug-4 re-entry cooldown tracker (in-memory; see config comment)
                if not hasattr(self, '_calm3d_last_fire'):
                    self._calm3d_last_fire = {}
                self._calm3d_last_fire[pair] = datetime.utcnow()
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
        # Aug 21 gate 57: same funnel-prefix convention for the bull-run sleeve — BR_STOP_LOSS /
        # BR_BREAKEVEN_EXIT / BR_TRAILING_STOP / BR_MAX_HOLD_TIME. Whitelist matchers strip BR_.
        if reason and (order.entry_strategy or "") == "BULLRUN_LONG" and not reason.startswith("BR_"):
            reason = "BR_" + reason
        # Aug 21 gate 57: stamp per-pair spacing on sleeve CLOSES too (entry stamps on open) —
        # the replay's 2h spacing ran exit-to-entry.
        if (order.entry_strategy or "") == "BULLRUN_LONG":
            try:
                _br_last_fire[order.pair] = _leash_time.time()
            except Exception:
                pass
        # Jun 16: snapshot the shadow's peak stretch AT THIS CLOSE INSTANT (before post-exit
        # tracking grows it). Diagnostic — vs runner_peak_stretch (live peak at exit): if ≈ equal
        # the live strpk was NOT under-sampling (the whole shadow gap is post-exit → Fix B, not A).
        try:
            _sps_close = _LEASH_STATE.get(order.id, {}).get('pstretch')
            if _sps_close is not None:
                order.shadow_peak_stretch_at_close = round(_sps_close, 4)
        except Exception:
            pass
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
            # Aug-11 🛡: cancel the resting backstop BEFORE our own close order goes out —
            # the dual-close race guard (never let the exchange stop + the bot's close race).
            if getattr(order, 'backstop_algo_id', None):
                try:
                    _bkc_ok = await binance_service.cancel_backstop_stop(order.pair, order.backstop_algo_id)
                    logger.info(f"[BACKSTOP_CANCELED] {order.pair}: algoId={order.backstop_algo_id} ok={_bkc_ok} (pre-close)")
                    if _bkc_ok:
                        # review fix: null + commit so a failed close leaves a NULL id the
                        # sweep can heal (a dead algoId would read as 'protected' while naked)
                        order.backstop_algo_id = None
                        await db.commit()
                    else:
                        self._record_filter_block("BACKSTOP_CANCEL_FAILED", order.direction)
                except Exception as _bkc_err:
                    self._record_filter_block("BACKSTOP_CANCEL_FAILED", order.direction)
                    logger.error(f"[BACKSTOP_CANCEL_FAILED] {order.pair}: {_bkc_err}")

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
                "STOP_LOSS", "BREAKEVEN_EXIT", "FL_SIGNAL_LOST", "FL_REGIME_CHANGE", "FL_TICK_MOMENTUM", "FL_EMERGENCY_SL", "FL_DEEP_STOP", "FL_RECOVERED", "BR_",  # Aug 21 gate 57: all bull-run sleeve exits are stop-class/urgent
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
                # Aug-11 🛡: close failed → RE-PLACE the backstop immediately so the position
                # is never naked while the retry queue grinds (we canceled it pre-close).
                if bool(getattr(config.trading_config.thresholds, 'broker_backstop_enabled', False)) and not self.is_paper_mode:
                    try:
                        _bk_pct = float(getattr(config.trading_config.thresholds, 'broker_backstop_pct', 2.5) or 2.5)
                        _bk_tr = order.entry_price * (1 - _bk_pct / 100.0) if order.direction == "LONG" else order.entry_price * (1 + _bk_pct / 100.0)
                        _bk_new = await binance_service.place_backstop_stop(order.pair, order.direction, _bk_tr)
                        if _bk_new:
                            order.backstop_algo_id = _bk_new
                            await db.commit()  # review fix: persist NOW — reconciler/next close must see the LIVE id, not the canceled one
                            logger.warning(f"[BACKSTOP_REPLACED] {order.pair}: close failed → fresh algoId={_bk_new} guarding the retry window")
                        else:
                            self._record_filter_block("BACKSTOP_PLACE_FAILED", order.direction)
                    except Exception as _bkr_err:
                        logger.error(f"[BACKSTOP_REPLACE_FAILED] {order.pair}: {_bkr_err}")
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
                "STOP_LOSS", "BREAKEVEN_EXIT", "FL_SIGNAL_LOST", "FL_REGIME_CHANGE", "FL_TICK_MOMENTUM", "FL_EMERGENCY_SL", "FL_DEEP_STOP", "FL_RECOVERED", "BR_",  # Aug 21 gate 57: all bull-run sleeve exits are stop-class/urgent
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
                # 🚀 Jul 27 SPIKE_FADE AUTO-TRIPWIRE: a fade closing <= tripwire
                # means the price GAPPED THROUGH the monitored −0.70 stop (squeeze
                # signature / stop-failure — should NEVER happen on clean paper
                # fills). If spike_tripwire_autodisable: engine self-disables the fade species; else (default, Aug-5) CRITICAL alert-only;
                # squeezes cluster faster than human reaction at full size.
                # Re-enable = manual UI toggle after investigation.
                # ─────────────────────────────────────────────────────────────
                try:
                    if ((order.entry_strategy or "") == "SPIKE_FADE"
                            and order.pnl_percentage is not None
                            and getattr(config.trading_config.thresholds, 'spike_fade_enabled', False)
                            and order.pnl_percentage <= float(getattr(config.trading_config.thresholds, 'spike_fade_tripwire_pct', -1.5) or -1.5)):
                        # Aug-5 operator directive: tripwire is ALERT-ONLY unless
                        # spike_tripwire_autodisable is re-enabled (see config comment).
                        if getattr(config.trading_config.thresholds, 'spike_tripwire_autodisable', False):
                            config.trading_config.thresholds.spike_fade_enabled = False
                            from config import save_trading_config as _sp_save_cfg
                            _sp_save_cfg(config.trading_config)
                            logger.critical(
                                f"[SPIKE_FADE_TRIPWIRE] {order.pair}: fade closed {order.pnl_percentage:.2f}% <= "
                                f"tripwire — price gapped through the fixed stop (squeeze). "
                                f"SPIKE_FADE AUTO-DISABLED; re-enable manually after investigation.")
                        else:
                            logger.critical(
                                f"[SPIKE_FADE_TRIPWIRE] {order.pair}: fade closed {order.pnl_percentage:.2f}% <= "
                                f"tripwire — price gapped through the fixed stop (squeeze). "
                                f"ALERT-ONLY (auto-disable OFF per operator Aug-5); species stays enabled.")
                except Exception as _sp_trip_err:
                    logger.error(f"[SPIKE_FADE_TRIPWIRE] flip failed (fade stays enabled — investigate): {_sp_trip_err}")

                # 🏀 Jul 31 SPIKE_BOUNCE AUTO-TRIPWIRE (mirror): a bounce closing <=
                # tripwire means price gapped through the fixed −0.70 stop — the
                # falling-knife / liquidation-cascade signature the Jun post-mortem
                # documented (−5/−6% continuation = ruin class). Self-disable only if spike_tripwire_autodisable (default OFF Aug-5 = alert-only); manual
                # re-enable after investigation. Retained deliberately even though the
                # operator removed the N-based kill gate — this is a stop-FAILURE
                # detector, not a performance verdict.
                try:
                    if ((order.entry_strategy or "") == "SPIKE_BOUNCE"
                            and order.pnl_percentage is not None
                            and getattr(config.trading_config.thresholds, 'spike_bounce_enabled', False)
                            and order.pnl_percentage <= float(getattr(config.trading_config.thresholds, 'spike_bounce_tripwire_pct', -1.5) or -1.5)):
                        if getattr(config.trading_config.thresholds, 'spike_tripwire_autodisable', False):
                            config.trading_config.thresholds.spike_bounce_enabled = False
                            from config import save_trading_config as _sb_save_cfg
                            _sb_save_cfg(config.trading_config)
                            logger.critical(
                                f"[SPIKE_BOUNCE_TRIPWIRE] {order.pair}: bounce closed {order.pnl_percentage:.2f}% <= "
                                f"tripwire — price gapped through the fixed stop (cascade continuation). "
                                f"SPIKE_BOUNCE AUTO-DISABLED; re-enable manually after investigation.")
                        else:
                            logger.critical(
                                f"[SPIKE_BOUNCE_TRIPWIRE] {order.pair}: bounce closed {order.pnl_percentage:.2f}% <= "
                                f"tripwire — price gapped through the fixed stop (cascade continuation). "
                                f"ALERT-ONLY (auto-disable OFF per operator Aug-5); species stays enabled.")
                except Exception as _sb_trip_err:
                    logger.error(f"[SPIKE_BOUNCE_TRIPWIRE] flip failed (bounce stays enabled — investigate): {_sb_trip_err}")

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
                            # Jul 28 BE-LOCK SHADOW: stamp first-touch minute + post-touch
                            # trough per arm threshold (tainted/spike trades leave NULLs).
                            _cache_bl = _cached.get('_belock')
                            if _cache_bl and not _cached.get('_belock_taint'):
                                for _bl_x, _bl_cols in ((15, ('belock_t15_min', 'belock_tr15')),
                                                        (20, ('belock_t20_min', 'belock_tr20')),
                                                        (30, ('belock_t30_min', 'belock_tr30'))):
                                    _bl_state = _cache_bl.get(_bl_x)
                                    if _bl_state and _bl_state[0] is not None:
                                        setattr(order, _bl_cols[0], _bl_state[0])
                                        setattr(order, _bl_cols[1], _bl_state[1])
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
        if _reason_base.startswith("BR_"):  # Aug 21 gate 57: bull-run sleeve reasons (BR_STOP_LOSS → STOP_LOSS etc.)
            _reason_base = _reason_base[3:]
        if not (_reason_base.startswith("BREAKEVEN_EXIT") or _reason_base.startswith("SIGNAL_LOST") or _reason_base.startswith("TICK_MOMENTUM_EXIT") or _reason_base.startswith("RSI_MOMENTUM_EXIT") or _reason_base.startswith("RSI_HANDOFF_EXIT") or _reason_base.startswith("EMA13_CROSS_EXIT") or _reason_base.startswith("EMA_STACK_CROSS_EXIT") or _reason_base.startswith("STOP_LOSS") or _reason_base.startswith("REGIME_CHANGE") or _reason_base.startswith("TRAILING_STOP") or _reason_base.startswith("LADDER_FLOOR") or _reason_base.startswith("RUNNER_TRAIL") or _reason_base.startswith("MOMENTUM_EXIT") or _reason_base.startswith("SLOPE_EXIT") or _reason_base.startswith("NO_EXPANSION") or _reason_base.startswith("RECOVERED") or _reason_base.startswith("DEEP_STOP") or _reason_base.startswith("EMERGENCY_SL") or _reason_base.startswith("FAST_EXIT") or _reason_base.startswith("ATR_FIXED_TP") or _reason_base.startswith("HARD_TP") or _reason_base.startswith("SPIKE_") or _reason_base.startswith("PATTERN_FIXED_TP") or _reason_base.startswith("PATTERN_FIXED_SL") or _reason_base.startswith("BACKSTOP_STOP")):
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
            # Jul 22: HARD_TP mechanism shadow (leash A / ladder B). The cap closed the
            # trade, so the post-exit stream IS the counterfactual in-trade path. Peaks
            # start at the realized close pnl; exits freeze when the variant's pullback
            # threshold is crossed. Unfired at horizon => censored (final pnl, fired=False).
            "htp_shadow": _reason_base.startswith("HARD_TP"),
            # Review fix (Jul 23): peaks seed from the IN-TRADE peak (a ladder fire exits at
            # its floor, below the trade's real peak — seeding from close alone would forget
            # already-locked floors). Rungs = the live per-side ladder, parsed once.
            "htp_A_peak": max(float(order.pnl_percentage or 1.0), float(order.peak_pnl or 0.0)),
            "htp_A_exit": None,
            "htp_B_peak": max(float(order.pnl_percentage or 1.0), float(order.peak_pnl or 0.0)),
            "htp_B_exit": None,
            "htp_B_rungs": (parse_hard_tp_ladder(getattr(tc.thresholds,
                'hard_tp_ladder_long' if order.direction == "LONG" else 'hard_tp_ladder_short', ''))
                or DEFAULT_LADDER_RUNGS),
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

    # Jul 30 — PHANTOM-FLIP TRACKER RETIRED (operator-directed): update_phantom_flips removed.
    # The phantom->probe pipeline matured (DEEPGAP graduated as probe #13 the same day); probes
    # are the live instrument (real fills/fees). Final phantom report archived in reports/.

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

            # Jul 22: HARD_TP mechanism shadow — advance leash (A) and ladder-replica (B) per tick.
            # Review fixes (Jul 23): ① the floor is IN the trigger condition (recording a
            # floored exit the condition never enforced inflated the scorekeeper once ladder
            # fires seed peaks below 1.0); ② variant B now replicates the LIVE ladder
            # semantics exactly (trigger-locked floors via the shared helper, per-side live
            # rungs stored at seed) instead of a peak-trailing leash that flattered it.
            if info.get("htp_shadow"):
                try:
                    # Variant A: single leash — trail 0.25 behind the running peak, hard
                    # floor 0.75. Effective stop = max(peak - 0.25, 0.75); fires when pnl
                    # falls TO the stop; records the stop.
                    if info["htp_A_exit"] is None:
                        if current_pnl > info["htp_A_peak"]:
                            info["htp_A_peak"] = current_pnl
                        else:
                            _a_stop = max(info["htp_A_peak"] - 0.25, 0.75)
                            if current_pnl <= _a_stop:
                                info["htp_A_exit"] = round(_a_stop, 4)
                    # Variant B: LIVE-ladder replica — same helper as the live exit.
                    if info["htp_B_exit"] is None:
                        if current_pnl > info["htp_B_peak"]:
                            info["htp_B_peak"] = current_pnl
                        else:
                            _b_floor, _ = hard_tp_ladder_floor(
                                info.get("htp_B_rungs") or DEFAULT_LADDER_RUNGS, info["htp_B_peak"])
                            if _b_floor is not None and current_pnl <= _b_floor:
                                info["htp_B_exit"] = round(_b_floor, 4)
                except Exception:
                    pass

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
                    logger.info(f"[POST_EXIT_REGIME_FLIP] {info['pair']} {direction}: regime flipped {_entry_reg} → {_live_reg} after exit, captured pnl={current_pnl:.4f}%")

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
                          stretch=_pe_stretch, atr=info.get('entry_atr_pct'))
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
                # peak_at is stamped on a new HIGH, trough_at on a new LOW (see ~L5048/5051) —
                # that's LONG-centric. For a SHORT the favorable "peak" is the LOW price, so the
                # minutes must flip to MATCH the direction-aware peak_pnl/trough_pnl above. Without
                # this, every SHORT's peak_minutes reports the brief post-exit high-tick (~0.0m)
                # instead of when the real favorable move landed. (Jun 18 bugfix.)
                if direction == "LONG":
                    peak_minutes = (info["peak_at"] - exit_time).total_seconds() / 60.0
                    trough_minutes = (info["trough_at"] - exit_time).total_seconds() / 60.0
                else:
                    peak_minutes = (info["trough_at"] - exit_time).total_seconds() / 60.0
                    trough_minutes = (info["peak_at"] - exit_time).total_seconds() / 60.0
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
                                # Jun 16: ATR-floored give-back shadows (N=0.5/1.0/1.5) — tune runner_trail_short_atr_mult
                                shadow_atr05_pnl=_leash_exits.get('atr05', (None, None))[0],
                                shadow_atr05_min=_leash_exits.get('atr05_min'),
                                shadow_atr10_pnl=_leash_exits.get('atr10', (None, None))[0],
                                shadow_atr10_min=_leash_exits.get('atr10_min'),
                                shadow_atr15_pnl=_leash_exits.get('atr15', (None, None))[0],
                                shadow_atr15_min=_leash_exits.get('atr15_min'),
                                # Jun 17 PM: give-back-cap shadows (frac 0.25/0.35/0.50) — tune runner_trail_short_giveback_frac
                                shadow_cap025_pnl=_leash_exits.get('cap025', (None, None))[0],
                                shadow_cap025_min=_leash_exits.get('cap025_min'),
                                shadow_cap035_pnl=_leash_exits.get('cap035', (None, None))[0],
                                shadow_cap035_min=_leash_exits.get('cap035_min'),
                                shadow_cap050_pnl=_leash_exits.get('cap050', (None, None))[0],
                                shadow_cap050_min=_leash_exits.get('cap050_min'),
                                # Jul 6: arm-level shadows (arm 0.35/0.40, trail 0.25, tracked pre-0.45) — the arm-lowering question
                                shadow_arm035_pnl=_leash_exits.get('arm035', (None, None))[0],
                                shadow_arm035_min=_leash_exits.get('arm035_min'),
                                shadow_arm040_pnl=_leash_exits.get('arm040', (None, None))[0],
                                shadow_arm040_min=_leash_exits.get('arm040_min'),
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
                                # Jul 22: HARD_TP mechanism shadow finalization. Unfired at
                                # horizon => censored: record final observed pnl, fired=False.
                                hard_tp_shadow_leash_pnl=(round(info["htp_A_exit"], 4) if info.get("htp_A_exit") is not None
                                                          else (round(final_pnl, 4) if info.get("htp_shadow") else None)),
                                hard_tp_shadow_leash_fired=((info.get("htp_A_exit") is not None) if info.get("htp_shadow") else None),
                                hard_tp_shadow_ladder_pnl=(round(info["htp_B_exit"], 4) if info.get("htp_B_exit") is not None
                                                           else (round(final_pnl, 4) if info.get("htp_shadow") else None)),
                                hard_tp_shadow_ladder_fired=((info.get("htp_B_exit") is not None) if info.get("htp_shadow") else None),
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
            # WebSocket price is primary. [WS_WATCHDOG] Jul-27 FLOCK/EVAA/HBAR
            # incidents: an open order can go price-blind (pair missing from
            # the live stream / dead stream) — frozen at entry with realtime
            # SL starved. If the pair's REAL-tick clock is silent >90s,
            # REST-fetch the price so exits keep working (self-limited to
            # ~1 call/90s/pair because the tick=True update refreshes
            # last_tick). Force a WS reconnect ONLY when the pair is missing
            # from the live stream (a quiet-but-healthy thin pair trades
            # <1/90s legitimately — reconnecting the whole stream would blip
            # every position blind for nothing); globally rate-limited 120s.
            tracker = websocket_tracker.get_tracker(order.pair)
            current_price = tracker.last_price if tracker else None

            _silence = websocket_tracker.pair_silence_seconds(order.pair)
            if _silence is None or _silence > 90:
                _rest_price = await binance_service.get_current_price(order.pair)
                if _rest_price and _rest_price > 0:
                    # tick=True: a REST price is a real observation — feeds the
                    # silence clock so this fallback self-limits to ~1/90s/pair
                    websocket_tracker.update_price(order.pair, _rest_price, tick=True)
                    logger.warning(
                        f"[WS_WATCHDOG] {order.pair} silent on WS for "
                        f"{'never-ticked' if _silence is None else f'{int(_silence)}s'} — "
                        f"REST price fallback {_rest_price}"
                    )
                    current_price = _rest_price
                _now_ts = datetime.utcnow().timestamp()
                if (not websocket_tracker.is_pair_streamed(order.pair)
                        and _now_ts - getattr(self, '_ws_watchdog_last_reconnect', 0) > 120):
                    self._ws_watchdog_last_reconnect = _now_ts
                    await websocket_tracker.force_reconnect(
                        f"open order {order.pair} missing from live WS stream"
                    )

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
            # Jul 24 (review fix I1): stale-guard — >10-min-old PairData (unscanned
            # extended-universe pair) treated as absent so EMA-driven exits skip cleanly
            # instead of firing on days-old EMAs. See the cache-build twin guard.
            if pair_data is not None and getattr(pair_data, 'updated_at', None) is not None:
                from datetime import timezone as _tz_pd2
                _pdts2 = pair_data.updated_at
                _pdts2 = _pdts2.replace(tzinfo=_tz_pd2.utc) if _pdts2.tzinfo is None else _pdts2
                if (datetime.now(_tz_pd2.utc) - _pdts2).total_seconds() > 600:
                    pair_data = None

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

            # 🌊 Aug-21 gate 57: BULLRUN_LONG dedicated exit path — sleeve trades run ONLY
            # _bullrun_exit_for (+ MAX_HOLD above + manual). Intercept BEFORE NO_EXPANSION /
            # FL / momentum-exit stack / check_exit_conditions so none of the alt exit
            # machinery ever touches them. `continue` sits OUTSIDE the try so a sleeve order
            # can never fall through into the alt chain on an error.
            if (order.entry_strategy or "") == "BULLRUN_LONG":
                try:
                    if order.direction == "LONG":
                        _br_raw = (current_price - order.entry_price) * order.quantity
                    else:
                        _br_raw = (order.entry_price - current_price) * order.quantity
                    _br_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                    _br_notional = order.entry_price * order.quantity if order.quantity > 0 else 1
                    _br_pnl = ((_br_raw - (order.entry_fee or 0) - _br_fee) / _br_notional) * 100
                    _br_peak = max(realtime_peak, _br_pnl)
                    order.peak_pnl = _br_peak
                    order.trough_pnl = min(realtime_trough, _br_pnl)
                    _br_close, _br_reason, _br_stop = _bullrun_exit_for(_br_pnl, _br_peak, getattr(order, 'entry_atr_pct', None))
                    if _br_close:
                        logger.info(f"[BULLRUN_EXIT] {order.pair}: {_br_reason} fire pnl={_br_pnl:.2f}% peak={_br_peak:.2f}% stop_line={_br_stop:.2f}%")
                        closed_order = await self.close_position(db, order, current_price, _br_reason)
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id, "pair": closed_order.pair,
                                "action": "CLOSED", "reason": closed_order.close_reason,
                                "pnl": closed_order.pnl, "tp_level": order.current_tp_level or 1,
                            })
                except Exception as _br_ex_err:
                    logger.error(f"[BULLRUN_EXIT] {order.pair}: monitor-path check failed: {_br_ex_err}")
                try:
                    await db.commit()  # review I2: persist peak/trough on the no-close path (FLIP-skip precedent) — else a restart reseeds from a stale peak and BE/trail state is lost
                except Exception:
                    pass
                continue

            # Check NO_EXPANSION: close stale trades that never expanded
            # Jul 27: SPIKE_CHASE exemption WHILE ARMED only — a confirmed pump (RSI>=arm
            # seen) must never be clock-killed (ZEREBRO armed +66min, 3h clock closed a
            # +3.47 ride at −0.09); UNARMED zombie spikes KEEP the sweep.
            _sp_noexp_exempt = (
                (order.entry_strategy or "") == "SPIKE_CHASE"
                and bool(getattr(order, 'spike_armed', False))
                and getattr(config.trading_config.thresholds, 'spike_no_expansion_exempt_armed', True)
            )
            no_exp_minutes = config.trading_config.investment.no_expansion_minutes
            if no_exp_minutes > 0 and order.opened_at and not _sp_noexp_exempt:
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
                # Flips skip the momentum exit STACK below. Their SHORT runner stretch-trail
                # (strpk → FLIP_RUNNER_TRAIL) now fires in the REALTIME tick path (Jun 16 Fix A:
                # check_realtime_stop_loss), at WS-tick resolution instead of this 1s loop — so it
                # tracks peak-stretch and checks the trail like the leash-shadow does, rather than
                # under-tracking the peak and trailing out on a 1-second bounce. The shared
                # MAX_HOLD + NO_EXPANSION above already ran; just commit and skip the stack.
                await db.commit()
                continue

            # ════════════════════════════════════════════════════════════════
            # 🚀 Jul 27 SPIKE_CHASE monitor layer — option-D L2 (RSI-cooling, the
            # MAIN exit) + armed-state maintenance. Spike longs then SKIP the
            # momentum stack below (their L1 SL + L3 floors run realtime; this
            # loop owns only the RSI layer + the shared NO_EXPANSION above).
            # 5m RSI source: PairData for top-50 pairs; scanner-class pairs get
            # a rate-limited klines fetch (>=60s/order — 1-2 open spikes max).
            # SPIKE_FADE shorts deliberately fall through: they ride the normal
            # short stack (trail + EMA13) with their fixed −0.70 stop.
            # ════════════════════════════════════════════════════════════════
            if (order.entry_strategy or "") == "SPIKE_CHASE":
                try:
                    _th_sp_mon = config.trading_config.thresholds
                    _sp_arm_th = float(getattr(_th_sp_mon, 'spike_rsi_cool_arm', 75.0) or 75.0)
                    _sp_drop = float(getattr(_th_sp_mon, 'spike_rsi_cool_drop', 10.0) or 10.0)
                    # ── Review I2: monitor-resolution SL/floor BACKSTOP. The realtime
                    # block is tick-driven; if the WS stream starves (the Jul-27
                    # incident class), the WS_WATCHDOG REST fallback refreshes prices
                    # WITHOUT invoking the tick callback — so enforce L1/L3 here too
                    # (~1s resolution) from the same price the watchdog keeps fresh.
                    _sp_raw = ((current_price - order.entry_price) * order.quantity if order.direction == "LONG"
                               else (order.entry_price - current_price) * order.quantity)
                    _sp_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                    _sp_notional = order.entry_price * order.quantity if order.quantity > 0 else 1
                    _sp_pnl_pct = ((_sp_raw - (order.entry_fee or 0) - _sp_fee) / _sp_notional) * 100
                    _sp_bk_reason = None
                    _sp_sl_mon = float(getattr(_th_sp_mon, 'spike_sl_pct', -1.2) or -1.2)
                    _sp_peak_mon = max(realtime_peak or 0.0, _sp_pnl_pct)
                    # 🔒 Aug-3 SPIKE PROFIT LOCK — monitor-backstop mirror of the realtime
                    # leg (WS-starve lesson: every realtime exit needs a monitor twin).
                    _lk_en_mon = bool(getattr(_th_sp_mon, 'spike_lock_enabled', True))
                    _lk_arm_mon = float(getattr(_th_sp_mon, 'spike_lock_arm_pct', 0.20) or 0.0)
                    _lk_sl_mon = float(getattr(_th_sp_mon, 'spike_lock_sl_pct', -0.15) or -0.15)
                    if (_lk_en_mon and _lk_arm_mon > 0 and _sp_peak_mon >= _lk_arm_mon
                            and _sp_pnl_pct <= _lk_sl_mon + 0.01):
                        _sp_bk_reason = "SPIKE_LOCK L1"
                    elif _sp_pnl_pct <= _sp_sl_mon + 0.01:
                        _sp_bk_reason = "SPIKE_SL L1"
                    else:
                        _sp_rungs_mon = parse_hard_tp_ladder(getattr(
                            _th_sp_mon, 'spike_ladder_armed' if getattr(order, 'spike_armed', False) else 'spike_ladder_unarmed', '') or '')
                        if _sp_rungs_mon:
                            _sp_floor_mon, _sp_lvl_mon = hard_tp_ladder_floor(_sp_rungs_mon, _sp_peak_mon)
                            if _sp_floor_mon is not None and _sp_pnl_pct <= _sp_floor_mon:
                                _sp_bk_reason = f"SPIKE_FLOOR L{_sp_lvl_mon}"
                        # ── Jul 28 EXIT PATCH backstop parity (WS-starve lesson: every realtime
                        # exit needs a monitor mirror). Unarmed only, same formulas.
                        if _sp_bk_reason is None and not getattr(order, 'spike_armed', False):
                            _sp_ta_mon = float(getattr(_th_sp_mon, 'spike_trail_arm_pct', 0.45) or 0.0)
                            if _sp_ta_mon > 0 and _sp_peak_mon >= _sp_ta_mon:
                                _sp_atr_mon = float(getattr(order, 'entry_atr_pct', None) or 0.0)
                                _sp_gb_mon = (float(getattr(_th_sp_mon, 'runner_trail_atr_mult', 1.0) or 1.0) * _sp_atr_mon) if _sp_atr_mon > 0 else 0.45
                                _sp_te_mon = max(_sp_peak_mon - _sp_gb_mon,
                                                 float(getattr(_th_sp_mon, 'runner_trail_be_lock_pct', 0.10) or 0.10))
                                if _sp_pnl_pct <= _sp_te_mon:
                                    _sp_bk_reason = "SPIKE_TRAIL"
                            if _sp_bk_reason is None:
                                _sp_sk_mon = float(getattr(_th_sp_mon, 'spike_stale_kill_min', 30.0) or 0.0)
                                if _sp_sk_mon > 0 and order.opened_at is not None and _sp_peak_mon < 0.2:
                                    from datetime import timezone as _sp_tz2
                                    _sp_ref2 = order.opened_at.replace(tzinfo=_sp_tz2.utc) if order.opened_at.tzinfo is None else order.opened_at
                                    if (datetime.now(_sp_tz2.utc) - _sp_ref2).total_seconds() >= _sp_sk_mon * 60:
                                        _sp_bk_reason = "SPIKE_STALE"
                    if _sp_bk_reason is not None:
                        logger.warning(f"[SPIKE_MONITOR_BACKSTOP] {order.pair}: {_sp_bk_reason} at pnl={_sp_pnl_pct:.4f}% (monitor-resolution enforcement)")
                        closed_order = await self.close_position(db, order, current_price, _sp_bk_reason)
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id, "pair": closed_order.pair,
                                "action": "CLOSED", "reason": _sp_bk_reason,
                                "pnl": closed_order.pnl, "tp_level": order.current_tp_level or 1,
                            })
                        continue
                    _sp_rsi_now = pair_data.rsi if (pair_data and pair_data.rsi is not None) else None
                    if _sp_rsi_now is None:
                        # extended-universe pair: fetch 5m klines at most every 60s
                        if not hasattr(self, '_spike_rsi_fetch'):
                            self._spike_rsi_fetch = {}
                        if len(self._spike_rsi_fetch) > 50:  # review M2: bound the dict (ids of long-closed orders)
                            self._spike_rsi_fetch.clear()
                        _sp_last = self._spike_rsi_fetch.get(order.id, 0.0)
                        _sp_now_ts = datetime.utcnow().timestamp()
                        if _sp_now_ts - _sp_last >= 60.0:
                            self._spike_rsi_fetch[order.id] = _sp_now_ts
                            _sp_ohlcv = await binance_service.get_ohlcv(order.pair, '5m', limit=100)
                            if _sp_ohlcv and len(_sp_ohlcv) >= 20:
                                _sp_ind = calculate_indicators(_sp_ohlcv)
                                _sp_rsi_now = _sp_ind.get('rsi') if _sp_ind else None
                    if _sp_rsi_now is not None:
                        _sp_max_prev = getattr(order, 'spike_rsi_max', None)
                        if _sp_max_prev is None or _sp_rsi_now > _sp_max_prev:
                            order.spike_rsi_max = round(float(_sp_rsi_now), 2)
                        if not getattr(order, 'spike_armed', False) and _sp_rsi_now >= _sp_arm_th:
                            order.spike_armed = True
                            logger.info(f"[SPIKE_ARMED] {order.pair}: 5m RSI {_sp_rsi_now:.1f} >= {_sp_arm_th:.0f} — L2 armed, floors switch to the wide envelope")
                        # mirror state to the realtime cache (floors read it per tick)
                        if cached is not None:
                            cached['spike_armed'] = bool(order.spike_armed)
                            cached['spike_rsi_max'] = order.spike_rsi_max
                        if (getattr(order, 'spike_armed', False)
                                and order.spike_rsi_max is not None
                                and _sp_rsi_now <= float(order.spike_rsi_max) - _sp_drop):
                            logger.warning(
                                f"[SPIKE_RSI_COOL] {order.pair}: RSI {_sp_rsi_now:.1f} <= max {order.spike_rsi_max:.1f} − {_sp_drop:.0f} "
                                f"— momentum death, exiting at market")
                            closed_order = await self.close_position(db, order, current_price, "SPIKE_RSI_COOL")
                            if closed_order:
                                updates.append({
                                    "order_id": closed_order.id, "pair": closed_order.pair,
                                    "action": "CLOSED", "reason": "SPIKE_RSI_COOL",
                                    "pnl": closed_order.pnl, "tp_level": order.current_tp_level or 1,
                                })
                            continue
                except Exception as _sp_mon_err:
                    logger.error(f"[SPIKE_MONITOR] {order.pair}: L2 layer error (fail-open, SL/floors still live): {_sp_mon_err}")
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

            # 🔒 Aug-3 review fix I2: FADE/BOUNCE SPIKE-LOCK monitor twin (WS-starve
            # parity — every realtime exit needs a monitor mirror; the chase leg got one
            # at ship, this is the fade/bounce mirror). LOCK ONLY: the fixed −0.70 keeps
            # its historical realtime-only enforcement (inherited scope, unchanged).
            if (order.entry_strategy or "") in ("SPIKE_FADE", "SPIKE_BOUNCE"):
                try:
                    _th_fb_mon = config.trading_config.thresholds
                    _fb_lk_en = bool(getattr(_th_fb_mon, 'spike_lock_enabled', True))
                    # Aug-10 PM: fade exemption — monitor twin of the realtime leg (WS-starve rule)
                    if (_fb_lk_en and bool(getattr(_th_fb_mon, 'spike_lock_exempt_fade', True))
                            and (order.entry_strategy or "") == "SPIKE_FADE"):
                        _fb_lk_en = False
                    _fb_lk_arm = float(getattr(_th_fb_mon, 'spike_lock_arm_pct', 0.20) or 0.0)
                    _fb_lk_sl = float(getattr(_th_fb_mon, 'spike_lock_sl_pct', -0.15) or -0.15)
                    _fb_peak_mon = max(realtime_peak or 0.0, pnl_pct, float(order.peak_pnl or 0.0))
                    if (_fb_lk_en and _fb_lk_arm > 0 and _fb_peak_mon >= _fb_lk_arm
                            and pnl_pct <= _fb_lk_sl + 0.01):
                        logger.warning(f"[SPIKE_MONITOR_BACKSTOP] {order.pair}: SPIKE_LOCK L1 at "
                                       f"pnl={pnl_pct:.4f}% (peak {_fb_peak_mon:.2f}%, monitor-resolution enforcement)")
                        closed_order = await self.close_position(db, order, current_price, "SPIKE_LOCK L1")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id, "pair": closed_order.pair,
                                "action": "CLOSED", "reason": "SPIKE_LOCK L1",
                                "pnl": closed_order.pnl, "tp_level": order.current_tp_level or 1,
                            })
                            continue
                except Exception as _fb_lk_err:
                    logger.error(f"[SPIKE_LOCK_MONITOR] {order.pair}: fail-open: {_fb_lk_err}")

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
                # Jul 31 🏀 SPIKE_BOUNCE: strategy-scoped trail giveback (0.5×ATR frozen)
                runner_atr_mult_override=(float(getattr(config.trading_config.thresholds, 'spike_bounce_trail_atr_mult', 0.5) or 0.5)
                                          if (order.entry_strategy or '') == 'SPIKE_BOUNCE' else None),
                # Aug 11 ROSE bug fix: spike species carry FIXED SLs — this monitor checker
                # was racing the realtime path with the momentum conf SL (−0.70) and won on
                # ROSE (stopped −0.75 vs the shipped fade −1.5). Pass the species SL so the
                # monitor enforces the SAME stop as realtime (chase included for safety —
                # its option-D block normally exits it before reaching here).
                fixed_sl_override=(float(getattr(config.trading_config.thresholds, 'spike_fade_sl_pct', -1.50) or -1.50)
                                   if (order.entry_strategy or '') == 'SPIKE_FADE'
                                   else float(getattr(config.trading_config.thresholds, 'spike_bounce_sl_pct', -0.70) or -0.70)
                                   if (order.entry_strategy or '') == 'SPIKE_BOUNCE'
                                   else float(getattr(config.trading_config.thresholds, 'spike_sl_pct', -1.2) or -1.2)
                                   if (order.entry_strategy or '') == 'SPIKE_CHASE'
                                   else None),
                # 🛡 Aug 19 gate 53: quiet-pair conditional SL — same value in BOTH SL
                # paths (this monitor + realtime), same lesson as the ROSE fix above.
                quiet_sl_pct=_quiet_sl_for(order.direction, order.entry_strategy,
                                           getattr(order, 'entry_atr_pct', None)),
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

        # Aug-11 🛡 BACKSTOP SWEEP (live only): heal missing resting stops every scan —
        # covers restart/deploy gaps (in-flight positions re-armed) and earlier placement
        # failures. The DOT-orphan class dies here.
        if (not self.is_paper_mode
                and bool(getattr(config.trading_config.thresholds, 'broker_backstop_enabled', False))):
            try:
                async with AsyncSessionLocal() as _bk_db:
                    _bk_rows = (await _bk_db.execute(
                        select(Order).where(and_(Order.status == "OPEN", Order.is_paper == False,
                                                 Order.backstop_algo_id.is_(None))))).scalars().all()
                    for _bko in _bk_rows:
                        _bk_pct = float(getattr(config.trading_config.thresholds, 'broker_backstop_pct', 2.5) or 2.5)
                        _bk_tr = _bko.entry_price * (1 - _bk_pct / 100.0) if _bko.direction == "LONG" else _bko.entry_price * (1 + _bk_pct / 100.0)
                        _bk_new = await binance_service.place_backstop_stop(_bko.pair, _bko.direction, _bk_tr)
                        if _bk_new:
                            _bko.backstop_algo_id = _bk_new
                            logger.warning(f"[BACKSTOP_SWEEP] {_bko.pair}: healed missing backstop algoId={_bk_new}")
                        else:
                            self._record_filter_block("BACKSTOP_PLACE_FAILED", _bko.direction)
                    if _bk_rows:
                        await _bk_db.commit()
            except Exception as _bks_err:
                logger.error(f"[BACKSTOP_SWEEP] failed: {_bks_err}")

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
        # Jul 14 FUNNEL v2: scan sequence for edge-triggered episode counting.
        self._funnel_scan_seq = getattr(self, '_funnel_scan_seq', 0) + 1
        _new_listing_days = getattr(config.trading_config, 'new_listing_filter_days', 0)
        _alpha_filter = getattr(config.trading_config, 'alpha_subtype_filter_enabled', True)
        _coin_only = getattr(config.trading_config, 'coin_underlying_only', True)
        top_pairs = await binance_service.get_top_futures_pairs(
            pairs_limit,
            new_listing_filter_days=_new_listing_days,
            alpha_subtype_filter_enabled=_alpha_filter,
            coin_underlying_only=_coin_only,
        )
        # Jun 12: stamp the eligible-universe volume rank (1 = highest 24h vol)
        # BEFORE blacklist removal, so ranks stay comparable across config changes.
        # Persisted per-trade as entry_pair_rank (read gate for the 50->75 expansion).
        for _rank_i, _rank_p in enumerate(top_pairs):
            _rank_p['rank'] = _rank_i + 1
        # Aug-22 (operator-caught): the 🌊 sleeve's "top-N" must be N TRADEABLE pairs — blacklisted pairs
        # (global list AND bullrun_pair_blacklist) must not occupy rank slots, else ONG+ETH+BNB turn
        # top-10 into top-7. br_rank = rank among non-blacklisted pairs (what the replay models).
        try:
            _br_skip = set(x.strip().upper() for x in (getattr(config.trading_config, 'pair_blacklist', '') or '').split(',') if x.strip())
            _br_skip |= set(x.strip().upper() for x in (getattr(getattr(config.trading_config, 'thresholds', None), 'bullrun_pair_blacklist', '') or '').split(',') if x.strip())
            _br_n = 0
            for _rank_p in top_pairs:
                _sym = str(_rank_p.get('pair') or _rank_p.get('symbol') or '').upper()
                if _sym in _br_skip:
                    _rank_p['br_rank'] = None
                else:
                    _br_n += 1; _rank_p['br_rank'] = _br_n
        except Exception as _bre:
            logger.warning(f"[BULLRUN_LONG] br_rank stamping failed ({_bre}) — falling back to raw rank")
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
        global _current_btc_ema20, _current_btc_ema13, _current_btc_ema50, _current_btc_trend_gap_pct, _current_btc_price, _current_btc_adx_prev1
        _current_btc_regime = btc_regime
        _btc_ema20_slope_pct = btc_ema20_slope_pct if btc_ema20_slope_pct is not None else 0.0
        _current_btc_adx = btc_adx
        _current_btc_adx_prev1 = btc_adx_prev  # Aug-22: header arrow = sign(adx - adx_prev1) on closed bars
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

        # 🌊 Aug-21 gate 57: refresh the Bull-Run Monitor (self-throttled to 10 min; never raises)
        await self._update_bullrun_monitor(db)

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

        def _signal_multi_recorder(fails, direction: str):
            # Jul 14 FUNNEL v2: full fail-list accounting (All-fails / SOLE / Episodes).
            # Mirrors the legacy recorder's BTC-macro suppression so surfaces stay comparable.
            if direction == "LONG" and _btc_macro_blocks_long is not None:
                return
            if direction == "SHORT" and _btc_macro_blocks_short is not None:
                return
            try:
                self._record_filter_multi(fails, direction, _current_pair_holder.get('pair') or '?')
            except Exception:
                pass

        def _signal_block_recorder(filter_name: str, direction: str):
            # Suppress pair-level block recording for directions that BTC-level
            # filters would have vetoed anyway. This makes Filter Blocks counts
            # reflect the *decisive* last gate, not artifacts of evaluation order.
            # U3-followup (Jun 20): the PAIR_RSI_OB overbought-fade is a SHORT in STRONG_BULL — the
            # long's BTC-ADX ceiling is irrelevant to it, and overbought pairs are richest exactly when
            # BTC trends hardest (ADX>40 — the band the long veto used to choke the seed). When the long's
            # SOLE macro veto is BTC_ADX_GATE_HIGH and the mode is on, seed the fade THROUGH the veto
            # (phantom always; live only if mode==live). Its own gates (STRONG_BULL + pADX>=40) + the
            # ADX>40 de-risk (lev 0.05 in _flip_filters) still apply downstream.
            _rsiob_mode = (getattr(config.trading_config.thresholds, 'flip_pair_rsi_ob_btc_adx_high_mode', 'off') or 'off').lower()
            _seed_through = (direction == "LONG" and _btc_macro_blocks_long == "BTC_ADX_GATE_HIGH" and _rsiob_mode != 'off')
            if direction == "LONG" and _btc_macro_blocks_long is not None and not _seed_through:
                return
            if direction == "SHORT" and _btc_macro_blocks_short is not None:
                return
            _p = _current_pair_holder.get('pair')
            # When seeding THROUGH a macro veto, suppress the redundant pair-block count + decisive-reason
            # stamp (keeps Filter Blocks = the decisive BTC gate, not an evaluation-order artifact).
            if not _seed_through:
                self._record_filter_block(filter_name, direction, had_room=_scan_had_room_snapshot)
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
                        # Jun 16: mark for the LIVE flip (Phase 3). Through a BTC-ADX-high veto, go live
                        # only when mode==live (phantom mode observes the ADX>40 cohort without trading).
                        if not _seed_through or _rsiob_mode == 'live':
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
            # Jun 18: SHORT-fade phantom trackers — fade a BLOCKED LONG to a SHORT for the two
            # biggest UNTRACKED long-blockers. PAIR_RANGE_POSITION_MAX = long blocked at the TOP of
            # its range → short the range top (textbook mean-reversion). PAIR_ADX_MAX = long blocked
            # for an over-extended/exhausted trend → fade short (same logic as the fan dead-zone).
            # Observation-only; ~270 blocks/batch each so they accrue N fast. Routes into the
            # Source×Regime tracker as "<src> SHORT" rows (distinct from the existing SHORT-block
            # "<src> LONG" fade rows). Mirror of the PAIR_RSI_RANGE>65 LONG→SHORT seed above.
            if direction == "LONG" and filter_name in ("PAIR_RANGE_POSITION_MAX", "PAIR_ADX_MAX"):
                try:
                    _px3 = _current_pair_holder.get('price')
                    if _px3:
                        _seed_phantom_flip(_p, _px3, "LONG", filter_name,
                                           entry_fields=self._flip_entry_fields(_current_pair_holder, flip_dir="SHORT"))
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

                # 🌊 Aug-21 gate 57: Bull-Run sleeve hook — independent of the alt signal ladder
                # (the monitor replaces the entry filters at regime level). Self-gates on
                # enabled/GREEN/rank≤N inside; own try/except so it can never break the scan.
                try:
                    await self._maybe_open_bullrun_long(db, pair_info, ohlcv, indicators)
                except Exception as _br_err:
                    logger.error(f"[BULLRUN_LONG] {pair}: hook failed: {_br_err}")

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
                    multi_block_recorder=_signal_multi_recorder,
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
                    'age_days': pair_info.get('age_days'),  # Jul 13: listing age (180->90 step-down read gate)
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
            _pair_age_days = _cr.get('age_days')

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
            # Jul 30 MAJORS probe (#14): with the probe on, a no-trade major (BTC/ETH) is NOT
            # blocked here — it stays alive (flag set) and must still pass EVERY downstream
            # gate; survivors open as MAJORS_PROBE at gap-probe sizing in open_position
            # (sole-block cohort purity by construction — SLOPEGATE pattern). Probe off →
            # legacy track-only hard block, byte-identical behavior.
            _majors_probe_hit = False
            if signal in ["LONG", "SHORT"]:
                _nt_str = getattr(config.trading_config, 'no_trade_pairs', '') or ''
                _nt = set(p.strip() for p in _nt_str.split(',') if p.strip())
                if pair in _nt:
                    if getattr(config.trading_config.thresholds, 'majors_probe_enabled', False):
                        _majors_probe_hit = True
                        logger.info(f"[MAJORS_PROBE] {pair}: {signal} candidate (track-only pair) — probing instead of blocking")
                    else:
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
                    # Jun 17 passthrough-long (un-block hunt): low-ADX macro gate is the prime
                    # over-blocker in a real bull trend. Seed a SAME-direction virtual LONG so the
                    # Source×Regime cross-tab shows whether these blocked longs win in bull.
                    if signal == "LONG" and _btc_adx_too_low:
                        try:
                            _seed_phantom_flip(pair, indicators.get('price'), "LONG", "PASS:BTC_ADX_GATE_LOW",
                                               entry_fields=self._flip_entry_fields(indicators, flip_dir='LONG', scan=self._flip_scan_ctx(locals())), mode='PASS')
                        except Exception:
                            pass
                    signal = "NO_TRADE"

            # 🛡 FAKE_BULL_GUARD — Aug-14 2026 operator-override ship (blocked cohort N=10,
            # below locked gates — acknowledged; DECISION_LOG 2026-08-14). Blocks a momentum
            # LONG when every bull-confirmation axis fails at once: regime HEALTHY_BULL
            # (weakest bull tier) ∧ confidence STRONG_BUY (weakest tier) ∧ breadth
            # bull_pct <= 71 ∧ BTC EMA13-50 trend gap <= +0.01 (0 + measurement tolerance).
            # 4-pool blocked cohort 10 · 3W/7L · −$818; fires 0× in B2 real bull; 8/10 losers
            # DOA (peak <0.10 — no exit rule reaches them). Same-direction phantom seeds the
            # 🔒 revert read: first 6 blocked phantoms >=50% winners or net>0 → revert.
            # Fail-open on any error.
            if (signal == "LONG" and confidence == "STRONG_BUY"
                    and getattr(config.trading_config.thresholds, 'fake_bull_guard_enabled', False)):
                try:
                    _fbg_th = config.trading_config.thresholds
                    _fbg_bull_raw = getattr(_fbg_th, 'fake_bull_guard_bull_pct_max', 71.0)
                    _fbg_bull_max = float(_fbg_bull_raw) if _fbg_bull_raw is not None else 71.0
                    _fbg_tg_raw = getattr(_fbg_th, 'fake_bull_guard_tg_max', 0.01)
                    _fbg_tg_max = float(_fbg_tg_raw) if _fbg_tg_raw is not None else 0.01
                    _fbg_bull = globals().get('_market_bull_pct')
                    _fbg_tg = globals().get('_current_btc_trend_gap_pct')
                    try:
                        _fbg_reg = classify_btc_regime(globals().get('_current_btc_adx'),
                                                       globals().get('_current_btc_rsi'),
                                                       globals().get('_btc_ema20_slope_pct'))
                    except Exception:
                        _fbg_reg = None
                    if (_fbg_reg == 'HEALTHY_BULL' and _fbg_bull is not None and _fbg_tg is not None
                            and _fbg_bull <= _fbg_bull_max and _fbg_tg <= _fbg_tg_max):
                        # px= stamp IS the gate-47 revert surface (review fix: _seed_phantom_flip
                        # has been a retired no-op since Jul-30 — the CALM3D "blocks log entry px"
                        # pattern replaces it). At the first-6-blocks checkpoint, each block is
                        # replayed from (pair, ts, px) via klines under the current long exit
                        # stack; >=50% would-be winners or would-be net>0 -> disable. Logs rotate:
                        # pull the read within EB retention / at each batch review.
                        logger.info(f"[FAKE_BULL_GUARD] {pair}: LONG blocked px={indicators.get('price')} — "
                                    f"HEALTHY_BULL×STRONG_BUY, bull_pct {_fbg_bull:.1f}<={_fbg_bull_max:.0f} AND "
                                    f"btc_trend_gap {_fbg_tg:+.4f}<={_fbg_tg_max:+g} (bull label unconfirmed)")
                        self._record_filter_block("FAKE_BULL_GUARD", "LONG", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "FAKE_BULL_GUARD"
                        signal = "NO_TRADE"
                except Exception as _fbg_e:
                    logger.warning(f"[FAKE_BULL_GUARD] {pair}: check errored ({_fbg_e}) — fail-open, no block")

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

            # MOMENTUM-SHORT dead-tape block — Jun 28, 2026 (see config.py momentum_short_btc_atr_min).
            # Momentum SHORTs reach this normal-entry path (flips bypass it via _flip_filters), so a
            # SHORT here IS a momentum short. Block when BTC ATR% < threshold — the dead-BTC band where
            # momentum-shorts are 0% WR / 100% DOA cross-data and ZERO winners ever fell (winners all
            # ATR≥0.132). 0 = off. Fail-open: missing btc_atr_pct → no block.
            if signal == "SHORT":
                _msatr_min = float(getattr(config.trading_config.thresholds, 'momentum_short_btc_atr_min', 0.0) or 0.0)
                if _msatr_min > 0 and btc_atr_pct is not None and btc_atr_pct < _msatr_min:
                    logger.info(
                        f"[MOMENTUM_SHORT_LOATR] {pair}: momentum SHORT blocked — BTC ATR "
                        f"{btc_atr_pct:.3f} < {_msatr_min} (dead-tape DOA band)"
                    )
                    self._record_filter_block("MOMENTUM_SHORT_LOATR", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "MOMENTUM_SHORT_LOATR"
                    signal = "NO_TRADE"

            # MOMENTUM-SHORT weak-capitulation block — Jun 28, 2026 (see config.py momentum_short_weakcap_*).
            # Block momentum SHORT when ALL three hold: range_position < range_max (capitulation near low) AND
            # pair ATR% < atr_max (low vol) AND pair ADX < padx_max (weak trend) = triple-weak DOA short with
            # no follow-through. Blocks by BEHAVIOR (a low-pADX C1 like XLM is caught; trend-backed C1 pADX≥28
            # still fires). Momentum-shorts reach this path; flips bypass. Fail-open on any missing input.
            if signal == "SHORT" and getattr(config.trading_config.thresholds, 'momentum_short_weakcap_enabled', False):
                _wc_th = config.trading_config.thresholds
                _wc_rmax = float(getattr(_wc_th, 'momentum_short_weakcap_range_max', 0.0) or 0.0)
                _wc_amax = float(getattr(_wc_th, 'momentum_short_weakcap_atr_max', 0.0) or 0.0)
                _wc_pmax = float(getattr(_wc_th, 'momentum_short_weakcap_padx_max', 0.0) or 0.0)
                _wc_adx = indicators.get('adx')
                # Jul 10 BUGFIX: the indicators dict carries 'atr' (absolute), not 'atr_pct' — the
                # old .get('atr_pct') returned None every tick, so this filter NEVER fired since the
                # Jun-28 ship (fail-open). NEARUSDT 07-08 (-$73 triple-weak DOA) leaked through here.
                _wc_atr_abs = indicators.get('atr')
                _wc_price0 = indicators.get('price')
                _wc_atr = (_wc_atr_abs / _wc_price0 * 100) if (_wc_atr_abs is not None and _wc_price0) else None
                _wc_px = indicators.get('price'); _wc_hi = indicators.get('high_20'); _wc_lo = indicators.get('low_20')
                _wc_rng = ((_wc_px - _wc_lo) / (_wc_hi - _wc_lo) * 100) if (_wc_px is not None and _wc_hi is not None and _wc_lo is not None and _wc_hi != _wc_lo) else None
                if (_wc_rmax > 0 and _wc_amax > 0 and _wc_pmax > 0
                        and _wc_rng is not None and _wc_rng < _wc_rmax
                        and _wc_atr is not None and _wc_atr < _wc_amax
                        and _wc_adx is not None and _wc_adx < _wc_pmax):
                    logger.info(
                        f"[MOMENTUM_SHORT_WEAKCAP] {pair}: momentum SHORT blocked — triple-weak DOA "
                        f"(range {_wc_rng:.1f}<{_wc_rmax}, pATR {_wc_atr:.2f}<{_wc_amax}, pADX {_wc_adx:.1f}<{_wc_pmax})"
                    )
                    self._record_filter_block("MOMENTUM_SHORT_WEAKCAP", signal, had_room=_had_room)
                    self._last_pair_block_reason[pair] = "MOMENTUM_SHORT_WEAKCAP"
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
                                    # Jun 19 LIVE bounce-long — fade the OVERSOLD short-block into a real LONG
                                    # (washout dead-cat bounce). Self-gates: enabled / tight BTC RSI×ADX cells /
                                    # regime ∈ bounce_long_regimes. Phantom above keeps running as the proxy.
                                    if signal == "SHORT" and btc_rsi <= 35:
                                        try:
                                            await self._maybe_open_bounce_long(
                                                db, pair, indicators,
                                                entry_fields=self._flip_entry_fields(indicators, flip_dir='LONG', scan=self._flip_scan_ctx(locals())))
                                        except Exception:
                                            pass
                                    # Jun 17 passthrough-long (un-block hunt): ALL long blocks (not just
                                    # RSI extremes) — does this macro cross over-block good bull longs?
                                    if signal == "LONG":
                                        try:
                                            _seed_phantom_flip(pair, indicators.get('price'), "LONG", "PASS:BTC_RSI_ADX_CROSS",
                                                               entry_fields=self._flip_entry_fields(indicators, flip_dir='LONG', scan=self._flip_scan_ctx(locals())), mode='PASS')
                                        except Exception:
                                            pass
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
                                    # Jun 17 passthrough-long phantom (un-block hunt) + Jun 18 LIVE bull-long.
                                    _bl_opened = False
                                    if signal == "LONG":
                                        # PASS:FAN_RATIO_GATE phantom — the BLOCKED long tracked as a virtual LONG
                                        # (fan ≥ 0.85). The validated bull-long population (H.BULL 94% WR). Feeds the
                                        # Bull-Long Curve by fan × regime.
                                        try:
                                            _seed_phantom_flip(pair, indicators.get('price'), "LONG", "PASS:FAN_RATIO_GATE",
                                                               entry_fields=self._flip_entry_fields(indicators, flip_dir='LONG', scan=self._flip_scan_ctx(locals())), mode='PASS')
                                        except Exception:
                                            pass
                                        # Jun 18 (CORRECTED) LIVE bull-long — un-block the long. THIS is the validated
                                        # population (blocked long, fan ≥ 0.85). Self-gates: enabled / fan < bull_long_fan_max
                                        # (5.0) / regime ∈ bull_long_regimes. PRE-EMPTS the flip-short fade — in its regime
                                        # (H.BULL) the long should be UN-BLOCKED, not faded short (flip-shorts lose in H.BULL
                                        # anyway), and opening both would race for the one-per-pair slot.
                                        try:
                                            _bl_o = await self._maybe_open_bull_long(
                                                db, pair, indicators,
                                                entry_fields=self._flip_entry_fields(indicators, flip_dir='LONG', scan=self._flip_scan_ctx(locals())))
                                            _bl_opened = bool(_bl_o)
                                        except Exception:
                                            pass
                                    # Flip Entry — fade the block live (both sides), UNLESS a bull-long already
                                    # un-blocked this long (avoid opposite positions on the same pair).
                                    if not _bl_opened:
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
                            # Jun 18 (CORRECTED): the BULL-LONG live open + PASS phantom were wrongly placed
                            # here (fan-PASSED branch, fan < 0.85 — a rare, UNVALIDATED population). The validated
                            # bull-long is the BLOCKED-long passthrough (fan ≥ 0.85) — both now live in the block
                            # branch above. This else-branch keeps only the FAN_CONTROL short-fade control.

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
            # Jul 14 SLOPEGATE PROBE: this gate was measured killing ~133 signal-found
            # candidates/day (43 distinct opportunities) while its calibration rests on an
            # April N=4 sample. With slopegate_probe_enabled, a candidate hit ONLY by this
            # gate stays alive (tagged) and must still pass every OTHER engine gate below;
            # survivors open as 1x SLOPEGATE_PROBE in open_position. The gate still blocks
            # normally when the probe is off. Legacy counter freezes while probing (probe
            # fires ARE the signal — same convention as GAPFLAT/PAIR_EMA_GAP_NOT_EXPANDING).
            _slopegate_probe_hit = False
            _deadband_probe_hit = False
            _dbdown_probe_hit = False
            _deepgap_probe_hit = False  # Jul 30 probe #13 (SHORT-only, set at the deep-gap floor below)
            if signal in ["LONG", "SHORT"] and btc_ema20_slope_pct is not None:
                _th = config.trading_config.thresholds
                _sg_probe_on = getattr(_th, 'slopegate_probe_enabled', False)
                if signal == "LONG":
                    _flat_th = getattr(_th, 'macro_trend_flat_threshold_long',
                                       getattr(_th, 'macro_trend_flat_threshold', 0))
                    if _flat_th > 0 and btc_ema20_slope_pct < _flat_th:
                        # Jul 20: LONG probe verdict ✗ VINDICATED at 27/30 (avg arm locked) —
                        # per-side kill switch; the gate resumes blocking LONGs normally.
                        if _sg_probe_on and getattr(_th, 'slopegate_probe_long_enabled', True):
                            _slopegate_probe_hit = True
                            logger.info(f"[SLOPEGATE_PROBE] {pair}: LONG candidate (BTC slope {btc_ema20_slope_pct:+.4f}% < +{_flat_th}%) — probing instead of blocking")
                        else:
                            logger.info(f"[BTC_SLOPE_GATE] {pair}: LONG blocked — BTC slope {btc_ema20_slope_pct:+.4f}% < min +{_flat_th}%")
                            self._record_filter_block("BTC_SLOPE_GATE", "LONG", had_room=_had_room)
                            self._last_pair_block_reason[pair] = "BTC_SLOPE_GATE"
                            signal = "NO_TRADE"
                else:  # SHORT
                    _flat_th = getattr(_th, 'macro_trend_flat_threshold_short',
                                       getattr(_th, 'macro_trend_flat_threshold', 0))
                    if _flat_th > 0 and btc_ema20_slope_pct > -_flat_th:
                        if _sg_probe_on:
                            _slopegate_probe_hit = True
                            logger.info(f"[SLOPEGATE_PROBE] {pair}: SHORT candidate (BTC slope {btc_ema20_slope_pct:+.4f}% > -{_flat_th}%) — probing instead of blocking")
                        else:
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

                # Jul 5 — BTC 1h slope DEAD-BAND (LONG only). Flat hourly slope = no macro
                # carry: the alt breakout has no sponsor and dies on arrival (baseline flat
                # zone 43% WR vs 92-93% on both flanks; fresh losers RPL/ETHFI both in-zone,
                # peaks 0.00-0.44 = DOA fingerprint). Blocks the middle the max/min gates
                # leave open → allowed region becomes two trend windows. None-check (or-falsy
                # lesson: 0.0 config = deliberate off, None = unset → off).
                _dbraw = getattr(_th, 'long_btc_1h_deadband', 0.0)
                _db = 0.0 if _dbraw is None else float(_dbraw)
                # Jul 27 PROMOTION ③b (operator-ratified asymmetric split): the POSITIVE side
                # narrows to long_btc_1h_deadband_pos (0.025) — the proven upper half
                # [pos, _db) becomes NORMAL full-size flow (8·75%·+0.248 at ship). The
                # NEGATIVE side keeps the full _db width (flat-down unproven; DBDOWN probe
                # keeps collecting it). pos<=0 → legacy symmetric behavior.
                _db_pos_raw = getattr(_th, 'long_btc_1h_deadband_pos', 0.0)
                _db_pos = _db if (_db_pos_raw is None or float(_db_pos_raw) <= 0) else min(_db, float(_db_pos_raw))
                _in_deadband = (_current_btc_1h_slope >= 0 and _current_btc_1h_slope < _db_pos) or \
                               (_current_btc_1h_slope < 0 and _current_btc_1h_slope > -_db)
                if signal == "LONG" and _db > 0 and _in_deadband:
                    # Jul 15 DEADBAND_PROBE (probe #5, operator-directed): HALF-OPEN test of the
                    # flat-UP side only. Phantom raw revert gate technically fired (24·63%) but
                    # the split says the halves differ: flat-up 14·79%·+0.284% (single-day,
                    # ~4-5 episodes, 5 rows equity-perps now untradeable) vs flat-down
                    # 10·40%·−0.05%; historical pool REFUTES flat-up (6·50%·−0.08%) — so the
                    # case must be proven by fresh fills: slope in [0, +db) proceeds as a 1x
                    # DEADBAND_PROBE (still faces every other gate below); flat-down (−db, 0)
                    # keeps blocking + seeding PASS phantoms. Probe off → full band blocks.
                    if getattr(_th, 'deadband_probe_enabled', False) and _current_btc_1h_slope >= 0:
                        _deadband_probe_hit = True
                        logger.info(f"[DEADBAND_PROBE] {pair}: LONG candidate (BTC 1h slope {_current_btc_1h_slope:+.4f}% in flat-up [0,+{_db}%)) — probing instead of blocking")
                    elif getattr(_th, 'dbdown_probe_enabled', False) and _current_btc_1h_slope < 0:
                        # Jul 20 DBDOWN PROBE (probe #9): flat-DOWN half opens — graduated
                        # execution of the FIRED phantom revert gate (95·60.0%·+0.100; fresh
                        # flat-down 51·65%·+0.154 meets both arms; halves invert vs flat-up).
                        _dbdown_probe_hit = True
                        logger.info(f"[DBDOWN_PROBE] {pair}: LONG candidate (BTC 1h slope {_current_btc_1h_slope:+.4f}% in flat-down (−{_db}%,0)) — probing instead of blocking")
                    else:
                        logger.info(f"[LONG_BTC1H_DEADBAND] {pair}: LONG blocked — BTC 1h slope {_current_btc_1h_slope:+.4f}% in dead-band (−{_db}%, +{_db_pos}%) (flat hourly: no carry, DOA zone)")
                        self._record_filter_block("LONG_BTC1H_DEADBAND", "LONG", had_room=_had_room)
                        self._last_pair_block_reason[pair] = "LONG_BTC1H_DEADBAND"
                        # revert surface: same-direction PASS phantom of the blocked LONG
                        # (gate: re-open at >=60% WR on N>=10 fresh — the Jul-3 lesson, day-one wired)
                        _seed_phantom_flip(pair, indicators.get('price'), "LONG", "PASS:LONG_BTC1H_DEADBAND",
                                           entry_fields=self._flip_entry_fields(indicators, flip_dir="LONG"), mode='PASS')
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
                        else:
                            # Jul 6 — DEEP-GAP FLOOR (GIGGLE post-mortem): block a momentum-SHORT when the
                            # pair is ALREADY ≥ this far below its 4h trend — selling after the crash →
                            # bounce kills it at 20× (pair-level twin of btc_1h_slope_min_short −0.60,
                            # shipped N=4 same mechanism). Baseline: gap ≤−1.0 shorts 3·33%·−$224 (all
                            # ≤−1.5; band −1.5..−1.0 EMPTY → threshold in clean space; mild pullback
                            # −1.0..−0.6 = 2·100%·+$164 untouched). ⚠ N=3 operator-directed override —
                            # PASS phantom = day-one revert surface. None-check (0 = off sentinel).
                            _dg_raw = getattr(_th_pt, 'momentum_short_pair_gap_min', 0.0)
                            _dg = 0.0 if _dg_raw is None else float(_dg_raw)
                            if signal == "SHORT" and _dg < 0 and _pair_gap_pct <= _dg:
                                # Jul 30 DEEPGAP probe (#13): the PASS phantom graduated (final read
                                # N=17 · 71% · Σ+1.85%) — with the probe on, the candidate stays alive
                                # (ALL regimes; the regime read is the verdict's job) and must still
                                # pass every OTHER gate below; survivors open as DEEPGAP_PROBE at
                                # gap-probe sizing in open_position. Probe off → legacy hard block.
                                if getattr(_th_pt, 'deepgap_probe_enabled', False):
                                    _deepgap_probe_hit = True
                                    logger.info(f"[DEEPGAP_PROBE] {pair}: SHORT candidate (pair gap {_pair_gap_pct:.4f}% <= {_dg}%) — probing instead of blocking")
                                else:
                                    logger.info(f"[MOMENTUM_SHORT_DEEPGAP] {pair}: SHORT blocked — pair gap {_pair_gap_pct:.4f}% <= {_dg}% (already crashed below 4h trend: late short, bounce risk)")
                                    self._record_filter_block("MOMENTUM_SHORT_DEEPGAP", "SHORT", had_room=_had_room)
                                    self._last_pair_block_reason[pair] = "MOMENTUM_SHORT_DEEPGAP"
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

            # ── Jul 24 SPIKE_CHASE probe (#11): single-candle 5m RSI explosion chase. Fires
            # ONLY when the ladder produced NO signal (normal full-size entries keep priority);
            # bypasses the ladder BY DESIGN — this entry class has no fan yet and RSI>65 blocks
            # every later candle (MIRA Jul-22 00:00 anatomy: RSI 49->82 on the discovery candle).
            # Caps/sizing/tagging enforced inside open_position (max_open, last-2-slots guard).
            _spike_chase_hit = False
            _spike_fade_hit = False
            try:
                _sc_th = config.trading_config.thresholds
                if (signal not in ("LONG", "SHORT")
                        and getattr(_sc_th, 'spike_chase_probe_enabled', False)):
                    _sc_rsi = indicators.get('rsi'); _sc_prev = indicators.get('rsi_prev1')
                    # Jul 24 PM: third trigger leg — the discovery candle must move PRICE, not
                    # just RSI (scale-free RSI explodes on stablecoin/flatline noise: USDCUSDT
                    # fired on +0.01%; MIRA's real discovery candle was +1.84%).
                    _sc_chg = None
                    _sc_vr = None
                    try:
                        if ohlcv and len(ohlcv) >= 22 and float(ohlcv[-2][4]) > 0:
                            _sc_chg = (float(ohlcv[-1][4]) / float(ohlcv[-2][4]) - 1.0) * 100.0
                            # Jul 24 PM leg 5: attention — discovery-candle volume vs prior-20 avg
                            # (MIRA 59.6x; chop max 5.7x; USDC 2.39x). Klines col 5 = base volume.
                            _sc_av = sum(float(c[5]) for c in ohlcv[-21:-1]) / 20.0
                            _sc_vr = (float(ohlcv[-1][5]) / _sc_av) if _sc_av > 0 else None
                    except Exception:
                        _sc_chg = None
                        _sc_vr = None
                    if (_sc_rsi is not None and _sc_prev is not None and _sc_chg is not None
                            and _sc_vr is not None
                            and _sc_vr >= float(getattr(_sc_th, 'spike_chase_probe_min_vol_ratio', 5.0) or 5.0)
                            and _sc_chg >= float(getattr(_sc_th, 'spike_chase_probe_min_candle_pct', 0.5) or 0.5)
                            # Jul 24 PM leg 4: quiet FLOOR — FHE fired from RSI ~12 (markdown bounce
                            # = knife-catch, not discovery); resting band is [prev_min, prev_max].
                            and _sc_prev >= float(getattr(_sc_th, 'spike_chase_probe_rsi_prev_min', 35.0) or 35.0)
                            and _sc_prev <= float(getattr(_sc_th, 'spike_chase_probe_rsi_prev_max', 55.0) or 55.0)
                            and (_sc_rsi - _sc_prev) >= float(getattr(_sc_th, 'spike_chase_probe_rsi_jump', 25.0) or 25.0)):
                        _nt_sc = set(x.strip() for x in (getattr(config.trading_config, 'no_trade_pairs', '') or '').split(',') if x.strip())
                        if pair not in _nt_sc:
                            # Jul 27 LEG 6 — pair ADX ROUTES the direction (10/10 lifetime:
                            # riders all <=20.2, duds all >=29.6). <=max -> CHASE LONG;
                            # >max -> FADE SHORT (mature blowoff). ADX None fails open to LONG.
                            _sc_adx = indicators.get('adx')
                            _sc_max_adx = float(getattr(_sc_th, 'spike_chase_max_adx', 30.0) or 30.0)
                            # Jul 28 REGIME ROUTER (parity with scanner): regime decides first.
                            _sc_regime_fade = False
                            if getattr(_sc_th, 'spike_regime_router_enabled', False):
                                try:
                                    _sc_reg_now = classify_btc_regime(
                                        globals().get('_current_btc_adx'), globals().get('_current_btc_rsi'), globals().get('_btc_ema20_slope_pct'))
                                except Exception:
                                    _sc_reg_now = None
                                _sc_chase_regs = set(x.strip() for x in (getattr(_sc_th, 'spike_chase_regimes', '') or '').split(',') if x.strip())
                                if _sc_reg_now is not None and _sc_reg_now != 'UNKNOWN':
                                    _sc_regime_fade = _sc_reg_now not in _sc_chase_regs
                                else:
                                    _sc_regime_fade = (globals().get('_current_btc_regime') or 'NEUTRAL') != 'BULLISH'
                                if _sc_regime_fade:
                                    logger.info(f"[SPIKE_REGIME_FADE] {pair}: non-chase regime ({_sc_reg_now}) — routing trigger to FADE short")
                            if _sc_regime_fade or (_sc_adx is not None and _sc_adx > _sc_max_adx):
                                # Jul 30 PM — fade bRSI ceiling: don't fade an alt spike while
                                # BTC's own momentum is hot (bRSI > max = market-wide beta ->
                                # the short gets squeezed; calm BTC = idiosyncratic exhaustion,
                                # the fade pays). Fail-open on missing bRSI. Block log = the
                                # kline re-sim revert surface (see config comment).
                                _sc_brsi = globals().get('_current_btc_rsi')
                                _sc_brsi_max = float(getattr(_sc_th, 'spike_fade_max_btc_rsi', 0.0) or 0.0)
                                if (_sc_brsi_max > 0 and _sc_brsi is not None and _sc_brsi > _sc_brsi_max
                                        and getattr(_sc_th, 'spike_fade_enabled', False)):
                                    self._record_filter_block("SPIKE_FADE_BRSI", "SHORT")
                                    logger.info(f"[SPIKE_FADE_BRSI] {pair}: fade blocked — BTC RSI {_sc_brsi:.1f} > {_sc_brsi_max} (squeeze-against-gravity guard) | entry_px={indicators.get('price')} pair_rsi={_sc_rsi:.1f} — re-sim revert row")
                                elif (lambda _bmax, _bpx, _b13: _bmax < 99 and _bpx and _b13 and ((_bpx - _b13) / _b13 * 100.0) > _bmax)(
                                        float(getattr(_sc_th, 'spike_fade_max_btc_dist13', 99.0) if getattr(_sc_th, 'spike_fade_max_btc_dist13', 99.0) is not None else 99.0),
                                        globals().get('_current_btc_price'), globals().get('_current_btc_ema13')) and getattr(_sc_th, 'spike_fade_enabled', False):
                                    # Aug-4 FADE BTC-DIST13 GATE (hook parity; see config comment)
                                    self._record_filter_block("SPIKE_FADE_BD13", "SHORT")
                                    _sc_bd13v = (globals().get('_current_btc_price') - globals().get('_current_btc_ema13')) / globals().get('_current_btc_ema13') * 100.0
                                    logger.info(f"[SPIKE_FADE_BD13] {pair}: fade blocked — BTC dist-EMA13 {_sc_bd13v:+.3f}% > {getattr(_sc_th, 'spike_fade_max_btc_dist13', 0.0)} (beta-tailwind guard) | entry_px={indicators.get('price')} pair_rsi={_sc_rsi:.1f} — re-sim revert row")
                                elif (lambda _rm, _gm, _rp, _e13, _e50: _rm > 0 and _rp is not None and _rp < _rm and _e13 is not None and _e50 and ((_e13 - _e50) / _e50 * 100.0) > _gm)(
                                        float(getattr(_sc_th, 'spike_fade_fb_rsi_prev_min', 0.0) or 0.0),
                                        float(getattr(_sc_th, 'spike_fade_fb_pgap_min', -0.40) if getattr(_sc_th, 'spike_fade_fb_pgap_min', None) is not None else -0.40),
                                        _sc_prev, indicators.get('ema13'), indicators.get('ema50')) and getattr(_sc_th, 'spike_fade_enabled', False):
                                    # Aug-10 FRESH-BREAKOUT GUARD (hook parity; see config comment)
                                    self._record_filter_block("SPIKE_FADE_FRESHBREAK", "SHORT")
                                    logger.info(f"[SPIKE_FADE_FRESHBREAK] {pair}: fade blocked — base RSI {_sc_prev:.1f} < {getattr(_sc_th, 'spike_fade_fb_rsi_prev_min', 0)} on non-crashed pair = fresh breakout, not exhaustion | entry_px={indicators.get('price')} — re-sim revert row")
                                elif getattr(_sc_th, 'spike_fade_enabled', False):
                                    signal, confidence = "SHORT", "STRONG_BUY"
                                    _spike_fade_hit = True
                                    logger.info(f"[SPIKE_FADE] {pair}: RSI jump {_sc_prev:.1f}->{_sc_rsi:.1f} (+{_sc_rsi - _sc_prev:.1f}) ADX {(_sc_adx if _sc_adx is not None else -1):.1f} route={'regime' if _sc_regime_fade else 'ADX>' + str(int(_sc_max_adx))} — fading (SHORT)")
                                else:
                                    logger.info(f"[SPIKE_ROUTER_BLOCK] {pair}: trigger fired, routed FADE ({'regime' if _sc_regime_fade else 'ADX'}, ADX {(_sc_adx if _sc_adx is not None else -1):.1f}) but fade disabled — no trade")
                            else:
                                # Jul 30 EXTENSION GUARD (chase-only): by the time all trigger
                                # legs confirm, price can be the completed spike top — all 8
                                # lifetime chases entered >=2.0xATR above EMA5, 0 wins (ERA:
                                # top tick, -1.2% in 19s). Block chase when stretch > mult x
                                # ATR%; fade untouched (it profits from the same extension).
                                # Fail-open on missing data (parity with the ADX router).
                                _sc_str_mult = float(getattr(_sc_th, 'spike_chase_max_stretch_atr', 1.5) or 0.0)
                                _sc_atrp = _ind_atr_pct(indicators)
                                _sc_stretch = None
                                try:
                                    if indicators.get('ema5') and indicators.get('price'):
                                        _sc_stretch = (indicators['price'] - indicators['ema5']) / indicators['ema5'] * 100.0
                                except Exception:
                                    _sc_stretch = None
                                if not getattr(_sc_th, 'spike_chase_enabled', True):
                                    # 🛑 Aug-21 dedicated CHASE kill-switch (hook parity)
                                    self._record_filter_block("SPIKE_CHASE_DISABLED", "LONG")
                                    logger.info(f"[SPIKE_ROUTER_BLOCK] {pair}: trigger fired, routed CHASE and chase disabled — no trade")
                                elif (_sc_str_mult > 0 and _sc_stretch is not None and _sc_atrp
                                        and _sc_stretch > _sc_str_mult * _sc_atrp):
                                    self._record_filter_block("SPIKE_CHASE_STRETCH", "LONG")
                                    logger.info(f"[SPIKE_CHASE_STRETCH] {pair}: chase blocked — stretch {_sc_stretch:+.2f}% > {_sc_str_mult}x ATR {_sc_atrp:.2f}% (ratio {(_sc_stretch/_sc_atrp):.1f})")
                                else:
                                    signal, confidence = "LONG", "STRONG_BUY"
                                    _spike_chase_hit = True
                                    logger.info(f"[SPIKE_CHASE] {pair}: RSI jump {_sc_prev:.1f}->{_sc_rsi:.1f} (+{_sc_rsi - _sc_prev:.1f}) candle {_sc_chg:+.2f}% vol {_sc_vr:.1f}x ADX {(_sc_adx if _sc_adx is not None else -1):.1f} stretch {(_sc_stretch if _sc_stretch is not None else 0):+.2f}%/{(_sc_atrp or 0):.2f}ATR — chase entry (LONG)")
            except Exception as _sph_err:
                # Review M5: clear signal too — an exception AFTER the router set
                # signal="SHORT"/"LONG" must not leak a plain MOMENTUM open at
                # normal sizing/exits.
                # Aug-17 audit (Important-4): the swallow was SILENT — the flip-kill
                # anatomy in miniature for the spike species (a recurring exception here
                # would read as "quiet tape"). Now logged loudly.
                logger.error(f"[SPIKE_HOOK_ERROR] {pair}: chase/fade hook raised — signal discarded (fail-safe): {_sph_err}")
                signal = None
                _spike_chase_hit = False
                _spike_fade_hit = False

            # ── Jul 31 🏀 SPIKE_BOUNCE (third species): single-candle 5m RSI CRASH →
            # LONG the violent idiosyncratic dump (mirror of the pump trigger). Fires
            # only when the ladder produced no signal and no pump-spike fired. Guards
            # each carry side-specific evidence (config.py comment block). Full size
            # 1×/1×, fade-mirrored exits, tripwire −1.5 — see open_position wiring.
            _spike_bounce_hit = False
            try:
                _sb_th = config.trading_config.thresholds
                if (signal not in ("LONG", "SHORT")
                        and not _spike_chase_hit and not _spike_fade_hit
                        and getattr(_sb_th, 'spike_bounce_enabled', False)):
                    _sb_rsi = indicators.get('rsi'); _sb_prev = indicators.get('rsi_prev1')
                    _sb_chg = None; _sb_vr = None
                    try:
                        if ohlcv and len(ohlcv) >= 22 and float(ohlcv[-2][4]) > 0:
                            _sb_chg = (float(ohlcv[-1][4]) / float(ohlcv[-2][4]) - 1.0) * 100.0
                            _sb_av = sum(float(c[5]) for c in ohlcv[-21:-1]) / 20.0
                            _sb_vr = (float(ohlcv[-1][5]) / _sb_av) if _sb_av > 0 else None
                    except Exception:
                        _sb_chg = None; _sb_vr = None
                    _sb_crash_min = float(getattr(_sb_th, 'spike_bounce_rsi_crash', 25.0) or 25.0)
                    _sb_pmin = float(getattr(_sb_th, 'spike_bounce_rsi_prev_min', 45.0) or 45.0)
                    _sb_pmax = float(getattr(_sb_th, 'spike_bounce_rsi_prev_max', 65.0) or 65.0)
                    _sb_min_chg = float(getattr(_sb_th, 'spike_bounce_min_candle_pct', 0.5) or 0.5)
                    _sb_max_dump = float(getattr(_sb_th, 'spike_bounce_max_dump_pct', 3.0) or 3.0)
                    _sb_min_vr = float(getattr(_sb_th, 'spike_bounce_min_vol_ratio', 5.0) or 5.0)
                    if (_sb_rsi is not None and _sb_prev is not None and _sb_chg is not None
                            and _sb_vr is not None and _sb_vr >= _sb_min_vr
                            and _sb_pmin <= _sb_prev <= _sb_pmax
                            and (_sb_prev - _sb_rsi) >= _sb_crash_min
                            and _sb_chg <= -_sb_min_chg):
                        _nt_sb = set(x.strip() for x in (getattr(config.trading_config, 'no_trade_pairs', '') or '').split(',') if x.strip())
                        if pair not in _nt_sb:
                            if _sb_chg < -_sb_max_dump:
                                # guard ①: news/delist/hack class — no bounce edge
                                self._record_filter_block("SPIKE_BOUNCE_DUMPCAP", "LONG")
                                logger.info(f"[SPIKE_BOUNCE_DUMPCAP] {pair}: dump {_sb_chg:+.2f}% deeper than -{_sb_max_dump}% — news-class, no bounce")
                            else:
                                _sb_ok = True
                                # guard ② bRSI floor (true mirror of the fade ceiling):
                                # bounce longs idiosyncratic panic — needs non-bearish BTC.
                                _sb_brsi = globals().get('_current_btc_rsi')
                                _sb_brsi_min = float(getattr(_sb_th, 'spike_bounce_min_btc_rsi', 0.0) or 0.0)
                                if _sb_brsi_min > 0 and _sb_brsi is not None and _sb_brsi < _sb_brsi_min:
                                    self._record_filter_block("SPIKE_BOUNCE_BRSI", "LONG")
                                    logger.info(f"[SPIKE_BOUNCE_BRSI] {pair}: bounce blocked — BTC RSI {_sb_brsi:.1f} < {_sb_brsi_min} (market-wide risk-off, dump is beta not panic) | entry_px={indicators.get('price')} pair_rsi={_sb_rsi:.1f}")
                                    _sb_ok = False
                                # guard ③ crashed-pair exclusion (DEEPGAP class)
                                if _sb_ok:
                                    _sb_gap_min = float(getattr(_sb_th, 'spike_bounce_min_pair_gap', 0.0) or 0.0)
                                    _sb_e13 = indicators.get('ema13'); _sb_e50 = indicators.get('ema50')
                                    if _sb_e13 is not None and _sb_e50:
                                        _sb_gap = (_sb_e13 - _sb_e50) / _sb_e50 * 100.0
                                        if _sb_gap_min != 0 and _sb_gap <= _sb_gap_min:
                                            self._record_filter_block("SPIKE_BOUNCE_CRASHED", "LONG")
                                            logger.info(f"[SPIKE_BOUNCE_CRASHED] {pair}: bounce blocked — EMA13-50 gap {_sb_gap:+.2f}% <= {_sb_gap_min}% (already-crashed pair, DEEPGAP class)")
                                            _sb_ok = False
                                        # guard ③b healthy-pair exclusion (Aug-5 pgap window) —
                                        # independent of the DEEPGAP floor (runs even if min-gap = 0/off)
                                        _sb_gap_max = float(getattr(_sb_th, 'spike_bounce_max_pair_gap', 99.0) if getattr(_sb_th, 'spike_bounce_max_pair_gap', None) is not None else 99.0)
                                        if _sb_ok and _sb_gap_max < 99 and _sb_gap > _sb_gap_max:
                                            self._record_filter_block("SPIKE_BOUNCE_PGAP", "LONG")
                                            logger.info(f"[SPIKE_BOUNCE_PGAP] {pair}: bounce blocked — EMA13-50 gap {_sb_gap:+.2f}% > {_sb_gap_max}% (healthy-pair dump = news class) | entry_px={indicators.get('price')} pair_rsi={_sb_rsi:.1f} — re-sim revert row")
                                            _sb_ok = False
                                # guard ④ regime block (BOUNCE_LONG's Jun-23 graves were bear cells)
                                if _sb_ok:
                                    _sb_blocked = set(x.strip() for x in (getattr(_sb_th, 'spike_bounce_blocked_regimes', '') or '').split(',') if x.strip())
                                    if _sb_blocked:
                                        try:
                                            _sb_reg = classify_btc_regime(
                                                globals().get('_current_btc_adx'), globals().get('_current_btc_rsi'), globals().get('_btc_ema20_slope_pct'))
                                        except Exception:
                                            _sb_reg = None
                                        if _sb_reg is None or _sb_reg == 'UNKNOWN':
                                            if (globals().get('_current_btc_regime') or 'NEUTRAL') == 'BEARISH':
                                                _sb_reg = 'STRONG_BEAR'
                                        if _sb_reg in _sb_blocked:
                                            self._record_filter_block("SPIKE_BOUNCE_REGIME", "LONG")
                                            logger.info(f"[SPIKE_BOUNCE_REGIME] {pair}: bounce blocked — regime {_sb_reg} in blocked set {sorted(_sb_blocked)}")
                                            _sb_ok = False
                                if _sb_ok:
                                    signal, confidence = "LONG", "STRONG_BUY"
                                    _spike_bounce_hit = True
                                    logger.info(f"[SPIKE_BOUNCE] {pair}: RSI crash {_sb_prev:.1f}->{_sb_rsi:.1f} ({_sb_rsi - _sb_prev:+.1f}) candle {_sb_chg:+.2f}% vol {_sb_vr:.1f}x — bounce entry (LONG)")
            except Exception as _sbh_err:
                # Same M5 rule as the pump block: an exception after signal was set
                # must not leak a plain MOMENTUM open at normal sizing/exits.
                # Aug-17 audit (Important-4): silent swallow now logged (flip-kill class).
                logger.error(f"[SPIKE_HOOK_ERROR] {pair}: bounce hook raised — signal discarded (fail-safe): {_sbh_err}")
                signal = None
                _spike_bounce_hit = False

            if signal in ["LONG", "SHORT"] and confidence and confidence != "NO_TRADE":
                # Aug-17 audit (Important-1): these two lines fire BEFORE the quality gate,
                # the promo router, and every in-open_position rejection — reworded so log
                # audits stop counting them as opens ([PORTFOLIO_OPEN] is the only true open).
                logger.info(f"[DEBUG_REACHED_OPEN] {pair} {signal} {confidence}: survived entry ladder — evaluating open (quality gate → router → open_position)")
                logger.info(f"[SIGNAL] {pair}: {signal} with {confidence} confidence - candidate accepted, not yet opened")
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
                if _qs_enabled and entry_quality_score <= _qs_block_max and not _spike_chase_hit and not _spike_fade_hit and not _spike_bounce_hit:
                    logger.info(
                        f"[QUALITY_SCORE_GATE] {pair}: {signal} blocked — entry_quality_score={entry_quality_score} <= block_max={_qs_block_max}"
                    )
                    self._record_filter_block("ENTRY_QUALITY_SCORE", signal, had_room=_scan_had_room_snapshot)
                    self._last_pair_block_reason[pair] = "ENTRY_QUALITY_SCORE"
                    continue
                entry_btc_regime = classify_btc_regime(btc_adx, btc_rsi, btc_ema20_slope_pct)

                # ══ Jul 27 PROMOTION PACKAGE — engine-side admission router (LONG only).
                # The ladder now falls gap-flat / flat+small / rsiadx-sole candidates through
                # (probes OFF); this router either ADMITS them (full size) or RE-BLOCKS with
                # the exact legacy counters. Fail-CLOSED: a router error blocks (old behavior)
                # rather than admitting unchecked flow at full size.
                _nonexp_calm3d_hit = False
                if signal == "LONG" and not _spike_chase_hit and not _spike_fade_hit and not _spike_bounce_hit:
                    _pp_block = None
                    try:
                        _th_pp = config.trading_config.thresholds
                        _pp_gapflat = bool(gap_expand_flat(indicators, signal, _th_pp))
                        _pp_gminflat = gminflat_band(indicators, signal, _th_pp) is True
                        # Review I-2 + pass-2 IMPORTANT-1: per-CLASS probe gating, MOST-SPECIFIC
                        # FIRST — gminflat ⊂ gapflat, so a flat+small candidate must be judged
                        # by the GMINFLAT toggle alone (a re-enabled GMINFLAT probe owns its
                        # class; the router must neither steal it at 2× nor starve it), and a
                        # pure gap-flat candidate by the GAPFLAT toggle alone.
                        if _pp_gminflat:
                            _pp_route = not getattr(_th_pp, 'gminflat_probe_enabled', False)
                        elif _pp_gapflat:
                            _pp_route = not getattr(_th_pp, 'gap_probe_enabled', False)
                        else:
                            _pp_route = False
                        if _pp_route:
                            if getattr(_th_pp, 'nonexp_calm3d_enabled', False):
                                _pp_regs = {x.strip() for x in (getattr(_th_pp, 'nonexp_calm3d_regimes', 'STRONG_BULL') or '').split(',') if x.strip()}
                                _pp_atr_max = float(getattr(_th_pp, 'nonexp_calm3d_btc_atr_max', 0.147) or 0.147)
                                _pp_batr = globals().get('_current_btc_atr_pct')
                                # Jul-30 COILED-PAIR leg (#24b pooled read, Option A vs the fired
                                # revert gate): calm tape only carries UNSTRETCHED entries —
                                # pooled N=19: stretch<=0.06 11W/1L +0.33 vs stretched 2W/5L
                                # -0.35 (door fires perfectly monotonic). 0=off; stretch None
                                # fails open (admit) for parity with the other legs.
                                _pp_smax = float(getattr(_th_pp, 'nonexp_calm3d_max_stretch', 0.06) or 0.0)
                                _pp_coiled = (_pp_smax <= 0) or (entry_ema5_stretch is None) or (entry_ema5_stretch <= _pp_smax)
                                # Jul-31 RISING-HOUR leg (operator-directed, #24b pooled read):
                                # the door's third identity condition — calm BTC ∧ coiled pair ∧
                                # RUNNING 1h engine. Coiled cohort split: b1h>0 = 10·90%·+$288
                                # (every big winner) vs b1h<=0 = 5·60%·−$65 (cluster-inflated).
                                # Sign boundary (not fitted). Sentinel <= -98 = leg off; b1h
                                # None fails open (admit) for parity with the other legs.
                                _pp_b1h_min = float(getattr(_th_pp, 'nonexp_calm3d_b1h_min', 0.0) if getattr(_th_pp, 'nonexp_calm3d_b1h_min', 0.0) is not None else 0.0)
                                _pp_b1h = globals().get('_current_btc_1h_slope')
                                _pp_hour_up = (_pp_b1h_min <= -98) or (_pp_b1h is None) or (float(_pp_b1h) > _pp_b1h_min)
                                # Aug-4 SAME-PAIR RE-ENTRY COOLDOWN (4th leg, see config comment):
                                # the door's only losers are <=57min same-pair re-fires — the coil
                                # already released; the re-entry buys the spent spring. 0 = off.
                                # Aug-10 DMI THRUST leg (5th leg, operator "ship A"): the door
                                # verifies the spring (calm tape / coiled pair / rising hour) but
                                # never asks WHO is winding it. +DI(14) = the pair's upward-push
                                # share; pADX = its smoothed strength. A coil without directional
                                # sponsorship is a one-candle breakout that dies on arrival (the
                                # DOA loser class). CUR door: keep 8·100%·+$874 / blocked 12·33%·
                                # −$1,383; B1 keep 2·100%·+$120 (3 old-2-leg-door winners
                                # sacrificed — discount on record). 0 = leg off; None fails open.
                                _pp_di_min = float(getattr(_th_pp, 'nonexp_calm3d_min_pos_di', 0.0) or 0.0)
                                _pp_padx_min = float(getattr(_th_pp, 'nonexp_calm3d_min_pair_adx', 0.0) or 0.0)
                                _pp_di = indicators.get('pos_di'); _pp_padx = indicators.get('adx')
                                _pp_dmi_ok = ((_pp_di_min <= 0 or _pp_di is None or float(_pp_di) >= _pp_di_min)
                                              and (_pp_padx_min <= 0 or _pp_padx is None or float(_pp_padx) >= _pp_padx_min))
                                _pp_cool_min = float(getattr(_th_pp, 'nonexp_calm3d_reentry_cooldown_min', 90.0) or 0.0)
                                _pp_cool_ok = True
                                if _pp_cool_min > 0:
                                    _pp_last = getattr(self, '_calm3d_last_fire', {}).get(pair)
                                    if _pp_last is not None and (datetime.utcnow() - _pp_last).total_seconds() < _pp_cool_min * 60:
                                        _pp_cool_ok = False
                                if (entry_btc_regime in _pp_regs and _pp_batr is not None
                                        and float(_pp_batr) <= _pp_atr_max and _pp_coiled and _pp_hour_up
                                        and not _pp_cool_ok):
                                    self._record_filter_block("CALM3D_REENTRY", "LONG", had_room=_scan_had_room_snapshot)
                                    logger.info(f"[CALM3D_REENTRY] {pair}: door candidate REJECTED — same-pair CALM3D fire "
                                                f"{(datetime.utcnow() - _pp_last).total_seconds()/60:.0f}min ago < cooldown {_pp_cool_min:.0f}min (re-entry buys the spent coil)")
                                if (entry_btc_regime in _pp_regs and _pp_batr is not None
                                        and float(_pp_batr) <= _pp_atr_max and _pp_coiled and _pp_hour_up
                                        and _pp_cool_ok and not _pp_dmi_ok):
                                    self._record_filter_block("CALM3D_DMI", "LONG", had_room=_scan_had_room_snapshot)
                                    logger.info(f"[CALM3D_DMI] {pair}: coiled candidate REJECTED — +DI {(float(_pp_di) if _pp_di is not None else -1):.1f} < {_pp_di_min} or pADX {(float(_pp_padx) if _pp_padx is not None else -1):.1f} < {_pp_padx_min} (thrust leg: coil without sponsorship) | entry_px={indicators.get('price')} — re-sim revert row")
                                if (entry_btc_regime in _pp_regs and _pp_batr is not None
                                        and float(_pp_batr) <= _pp_atr_max and _pp_coiled and _pp_hour_up
                                        and _pp_cool_ok and _pp_dmi_ok):
                                    _nonexp_calm3d_hit = True
                                    logger.info(f"[NONEXP_CALM3D] {pair}: gap-{'flat' if _pp_gapflat else 'min[flat]'} LONG ADMITTED full-size — {entry_btc_regime} ∧ BTC-ATR {float(_pp_batr):.3f} <= {_pp_atr_max} ∧ stretch {(entry_ema5_stretch if entry_ema5_stretch is not None else -1):.2f} <= {_pp_smax} ∧ b1h {(float(_pp_b1h) if _pp_b1h is not None else -1):+.3f} > {_pp_b1h_min}")
                                else:
                                    if (not _pp_coiled and entry_btc_regime in _pp_regs and _pp_batr is not None
                                            and float(_pp_batr) <= _pp_atr_max):
                                        logger.info(f"[NONEXP_CALM3D_STRETCH] {pair}: calm-tape candidate REJECTED — stretch {entry_ema5_stretch:.2f} > {_pp_smax} (coiled-pair leg)")
                                    elif (_pp_coiled and not _pp_hour_up and entry_btc_regime in _pp_regs and _pp_batr is not None
                                            and float(_pp_batr) <= _pp_atr_max):
                                        logger.info(f"[NONEXP_CALM3D_B1H] {pair}: coiled calm-tape candidate REJECTED — BTC 1h slope {(float(_pp_b1h) if _pp_b1h is not None else -1):+.3f} <= {_pp_b1h_min} (rising-hour leg)")
                                    _pp_block = "PAIR_EMA_GAP_NOT_EXPANDING" if _pp_gapflat else "PAIR_EMA_GAP_MIN"
                            else:
                                _pp_block = "PAIR_EMA_GAP_NOT_EXPANDING" if _pp_gapflat else "PAIR_EMA_GAP_MIN"
                        # ② RSIADX breadth release — runs INDEPENDENTLY (a dual-class candidate
                        # must pass BOTH admissions; calm3d passing must not skip this check).
                        if (_pp_block is None
                                and _rsi_adx_block_rule(signal, indicators.get('rsi'), indicators.get('adx'), _th_pp) is not None
                                and not getattr(_th_pp, 'rsiadx_probe_enabled', False)):
                            _pp_adm = float(getattr(_th_pp, 'rsiadx_breadth_admit_max', 0.0) or 0.0)
                            if _pp_adm > 0 and _market_bull_pct is not None and _market_bull_pct <= _pp_adm:
                                logger.info(f"[RSIADX_BREADTH_ADMIT] {pair}: cross-filter released — breadth {_market_bull_pct:.1f}% <= {_pp_adm} (inherits normal stack)")
                            else:
                                _pp_block = "PAIR_RSI_ADX_CROSS"
                    except Exception as _pp_err:
                        logger.error(f"[PROMO_ROUTER] {pair}: router error — fail-closed block: {_pp_err}")
                        _pp_block = "PROMO_ROUTER_ERROR"  # review M-1: own counter, not a legacy mislabel
                    if _pp_block is not None:
                        # Aug-17 (operator "no LONGs" audit): this re-block was COUNTED but never
                        # LOGGED — reached-open candidates vanished from the logs (the exact
                        # observability hole class that hid the flip kill 34 days). Named log line
                        # makes every router re-block visible between DEBUG_REACHED_OPEN and the
                        # scan summary.
                        logger.info(f"[PROMO_ROUTER_REBLOCK] {pair}: LONG fall-through candidate re-blocked as {_pp_block} (door legs not met — counted, not opened)")
                        self._record_filter_block(_pp_block, "LONG", had_room=_scan_had_room_snapshot)
                        self._last_pair_block_reason[pair] = _pp_block
                        continue

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

                # Aug-17 audit (Important-2): an exception escaping open_position previously
                # rode to scan_loop as a pair-UNNAMED error and ABORTED the whole cycle
                # (all later pairs unscanned). Now: pair-named, counted, cycle continues.
                try:
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
                        entry_pair_age_days=_pair_age_days,
                        # Jun 8: gap-expanding relaxation A/B tag — True if this entry was admitted
                        # by prev2_only but would have failed the strict prev1 check (MARGINAL cohort).
                        entry_gap_expand_marginal=gap_expand_marginal(indicators, signal),
                        # Jul 13 GAPFLAT probe: True iff this LONG failed the gap-expanding check but
                        # was let through the ladder by gap_probe_enabled (get_signal only admits a
                        # gap-flat candidate when the probe is on, so this tag is exact). open_position
                        # then applies the probe caps + 1x sizing, or records the block if capped.
                        gap_probe=bool(
                            signal in ("LONG", "SHORT")
                            and getattr(config.trading_config.thresholds, 'gap_probe_enabled', False)
                            and gap_expand_flat(indicators, signal, config.trading_config.thresholds)
                        ),
                        # Jul 13 PM GAPMIN probe (BOTH directions): gap in [floor, per-side threshold),
                        # NOT gap-flat on longs (band helper enforces cohort purity itself); only tagged
                        # when the probe is on — otherwise such candidates never reach here.
                        gapmin_probe=bool(
                            signal in ("LONG", "SHORT")
                            and getattr(config.trading_config.thresholds, 'gapmin_probe_enabled', False)
                            and gap_min_band(indicators, signal, config.trading_config.thresholds)
                        ),
                        # Jul 14 SLOPEGATE probe: candidate was hit by the BTC 5m flat dead-band
                        # (Phase-1 fall-through above) and survived every OTHER engine gate.
                        slopegate_probe=bool(_slopegate_probe_hit),
                        # Jul 15 RSIADX probe (#4): the ladder admitted this candidate via the
                        # cross-filter fall-through — recompute the exact block condition here
                        # (same helper the ladder uses; identical inputs => identical output).
                        rsiadx_probe=bool(
                            signal in ("LONG", "SHORT")
                            and getattr(config.trading_config.thresholds, 'rsiadx_probe_enabled', False)
                            and (signal != "SHORT"
                                 or getattr(config.trading_config.thresholds, 'rsiadx_probe_short_enabled', True))
                            and _rsi_adx_block_rule(signal, indicators.get('rsi'), indicators.get('adx'),
                                                    config.trading_config.thresholds) is not None
                        ),
                        # Jul 15 DEADBAND probe (#5): candidate sat in the 1h flat-UP half-band
                        # (fall-through above) and survived every OTHER engine gate.
                        deadband_probe=bool(_deadband_probe_hit),
                        # Jul 15 RSICEIL probe (#6): the ladder admitted this LONG via the
                        # (max, ceiling] RSI band suppression — recompute the exact band here.
                        rsiceil_probe=bool(
                            signal == "LONG"
                            and getattr(config.trading_config.thresholds, 'rsiceil_probe_enabled', False)
                            and rsiceil_band(indicators, signal, config.trading_config.thresholds) is True
                        ),
                        # Jul 20 GMINFLAT probe (#7): flat+small cohort-purity class admitted via
                        # the [flat] sub-rule suppression — recompute the exact band here.
                        gminflat_probe=bool(
                            signal in ("LONG", "SHORT")
                            and getattr(config.trading_config.thresholds, 'gminflat_probe_enabled', False)
                            and gminflat_band(indicators, signal, config.trading_config.thresholds) is True
                        ),
                        # Jul 20 ADXMAX probe (#8): (per-side ADX max, probe ceiling] band.
                        adxmax_probe=bool(
                            signal in ("LONG", "SHORT")
                            and getattr(config.trading_config.thresholds, 'adxmax_probe_enabled', False)
                            and adxmax_band(indicators, signal, config.trading_config.thresholds) is True
                        ),
                        # Jul 20 DBDOWN probe (#9): 1h flat-DOWN half-band fall-through above.
                        dbdown_probe=bool(_dbdown_probe_hit),
                        # Jul 30 DEEPGAP probe (#13, SHORT-only): deep-gap floor fall-through above.
                        deepgap_probe=bool(_deepgap_probe_hit),
                        # Jul 30 MAJORS probe (#14): no-trade major fall-through above (full ladder passed).
                        majors_probe=bool(_majors_probe_hit),
                        # Jul 21 ADXMAX2 probe (#10, LONG-only): second rung (35, 40].
                        adxmax2_probe=bool(
                            signal == "LONG"
                            and getattr(config.trading_config.thresholds, 'adxmax2_probe_enabled', False)
                            and adxmax2_band(indicators, signal, config.trading_config.thresholds) is True
                        ),
                        # Jul 24 probe #11 → Jul 27 full ship: ladder-bypass chase entry (flag set
                        # in the hook above); fade = the trigger's ADX>30 SHORT branch.
                        spike_chase_probe=bool(_spike_chase_hit),
                        spike_fade=bool(_spike_fade_hit),
                        spike_bounce=bool(_spike_bounce_hit),
                        # Jul 27 PM promotion: NONEXP_CALM3D admission (engine router above)
                        nonexp_calm3d=bool(_nonexp_calm3d_hit),
                    )
                except Exception as _op_err:
                    logger.error(f"[OPEN_EXCEPTION] {pair} {signal} {confidence}: open_position raised — cycle continues: {_op_err}", exc_info=True)
                    self._record_filter_block("OPEN_EXCEPTION", signal if signal in ("LONG", "SHORT") else "ANY")
                    order = None

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
                    # Aug-12 (operator): a None from open_position is USUALLY a counted entry
                    # filter (LONG_UNMATCHED_ONLY, already-open skip, ...) that logged its own
                    # line one instant earlier — warning about it read as a malfunction during
                    # log reviews. Only WARN when NO filter block was recorded in the last 2s
                    # (that residue = the real M4 failure class: rejections, precision, etc.).
                    if time.time() - getattr(self, '_last_filter_block_ts', 0) < 2.0:
                        # review fix: name the filter in the line itself — the 2s window is engine-global,
                        # so cross-pair masking is possible on busy scans (OPEN_FAILED_MOMENTUM may
                        # undercount; the named filter lets a log review catch a mismatch by eye)
                        logger.info(f"[OPEN_FILTERED] {pair} {signal} {confidence}: declined by a counted entry filter (last block: {getattr(self, '_last_filter_block_name', '?')})")
                    else:
                        logger.warning(f"[OPEN_FAILED_UNATTRIBUTED] {pair} {signal} {confidence}: open_position returned None with NO filter block recorded — real failure class (rejection/precision/balance), investigate")
                        self._record_filter_block("OPEN_FAILED_MOMENTUM", signal if signal in ("LONG", "SHORT") else "ANY")

        # ===== SPIKE SCANNER START (Jul 24 — full-universe SPIKE_CHASE feeder) =====
        # REVERT = spike_scanner_enabled=false (UI toggle): this is the ONLY call site, the
        # cycle is fail-silent, and the probe class has its own kill (spike_chase_probe_enabled).
        # TO REMOVE: delete this fenced call + the _spike_scanner_cycle method + _spike_rsi12
        # helper + the 3 config fields + UI block (grep "SPIKE SCANNER").
        try:
            await self._spike_scanner_cycle(db, set(p['pair'] for p in top_pairs))
        except Exception as _ss_err:
            logger.error(f"[SPIKE_SCANNER] cycle failed (fail-silent): {_ss_err}")
        # ===== SPIKE SCANNER END =====

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
            # Jul 27: SPIKE_FADE shorts ride the normal short stack but with a FIXED
            # stop (spike_fade_sl_pct, stamped in the cache) — no ATR widening, no
            # signal-active widening (squeeze tail must stay bounded). Jul 31: 🏀
            # SPIKE_BOUNCE longs get the same fixed-SL treatment (spike_bounce_sl_pct
            # — entry ATR is inflated by the dump candle itself; widening off it
            # would re-open the falling-knife tail the fixed stop exists to bound).
            _is_spike_fade = (order_info.get('entry_strategy') or "") in ("SPIKE_FADE", "SPIKE_BOUNCE")
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

            # 🌊 Aug-21 gate 57: BULLRUN_LONG dedicated realtime exit — FIRST in the chain, so
            # NO alt close mechanism (PATTERN_FIXED / HARD_TP ladder / ATR_FIXED_TP / FAST_EXIT /
            # EMA13 / EMA_STACK / BE / SL / trailing / tick) can ever touch a sleeve trade
            # (review C2: the original intercept sat after HARD_TP, which would have floor-locked
            # every winner at +1.25% and nullified the replay-validated 2×ATR trail). Peak/trough
            # are updated inline here (the shared tracking below is skipped for sleeve orders —
            # phantom/shadow columns stay NULL for them, on record). `continue` sits OUTSIDE the
            # try so a sleeve order can never fall through into the alt chain on an error.
            if (order_info.get('entry_strategy') or '') == 'BULLRUN_LONG':
                try:
                    _br_peak_rt = max(order_info.get('peak_pnl', 0) or 0, pnl_pct)
                    order_info['peak_pnl'] = _br_peak_rt
                    if pnl_pct < (order_info.get('trough_pnl', 0) or 0):
                        order_info['trough_pnl'] = pnl_pct
                    _br_close, _br_reason, _br_stop = _bullrun_exit_for(pnl_pct, _br_peak_rt, order_info.get('entry_atr_pct'))
                    if _br_close and not order_info.get('_closing_in_progress'):
                        order_info['_closing_in_progress'] = True
                        logger.warning(f"[REALTIME_BULLRUN_EXIT] {pair} {direction}: {_br_reason} pnl={pnl_pct:.4f}% peak={_br_peak_rt:.4f}% stop_line={_br_stop:.2f}% - CLOSING NOW!")
                        try:
                            async with AsyncSessionLocal() as _br_db:
                                _br_res = await _br_db.execute(
                                    select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                                )
                                _br_order = _br_res.scalar_one_or_none()
                                if _br_order:
                                    _br_closed = await self.close_position(_br_db, _br_order, current_price, _br_reason)
                                    if _br_closed:
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                        except Exception as _br_e:
                            logger.error(f"[REALTIME_BULLRUN_EXIT] Error closing {pair}: {_br_e}")
                            order_info['_closing_in_progress'] = False
                except Exception as _br_e2:
                    logger.error(f"[REALTIME_BULLRUN_EXIT] {pair}: check failed: {_br_e2}")
                continue

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
            # 🚀 Jul 27 SPIKE_CHASE option-D realtime layers (L1 fixed SL + L3
            # state-dependent floors). REPLACES the entire normal long exit
            # stack for spike longs (hard-tp / trailing / runner / EMA13 / fast
            # exits never run — `continue` below). L2 (RSI-cool) runs in the
            # monitor loop (needs 5m RSI, not ticks). Armed state is written by
            # the monitor and read here; floors ratchet by construction (derived
            # from the running peak via hard_tp_ladder_floor).
            # ════════════════════════════════════════════════════════════════
            _rt_strategy = order_info.get('entry_strategy') or ''
            if _rt_strategy == 'SPIKE_CHASE' and not order_info.get('_closing_in_progress'):
                _th_sp_rt = config.trading_config.thresholds
                # keep the shared peak fresh for the monitor / floors
                if pnl_pct > (order_info.get('peak_pnl') or 0.0):
                    order_info['peak_pnl'] = pnl_pct
                _sp_peak_rt = order_info.get('peak_pnl') or 0.0
                _sp_close_reason = None
                _sp_sl_rt = float(getattr(_th_sp_rt, 'spike_sl_pct', -1.2) or -1.2)
                # 🔒 Aug-3 SPIKE PROFIT LOCK (#24b ② verdict, fade-capture N=13 S$87/K$0):
                # once peak touches the arm, the fixed species SL tightens to the lock
                # level (−1.2 → −0.15 on this path). Checked BEFORE the species SL so an
                # armed collapse stamps SPIKE_LOCK, not SPIKE_SL. SPIKE_ prefix rides
                # both post-exit whitelists. Chase leg = mechanism-transfer (own tally).
                _sp_lk_en = bool(getattr(_th_sp_rt, 'spike_lock_enabled', True))
                _sp_lk_arm = float(getattr(_th_sp_rt, 'spike_lock_arm_pct', 0.20) or 0.0)
                _sp_lk_sl = float(getattr(_th_sp_rt, 'spike_lock_sl_pct', -0.15) or -0.15)
                # Review fix I1 (Aug-3): persist the peak ONCE when it first crosses the
                # lock arm — chase `continue`s out of the monitor stack that persists
                # peak_pnl for other strategies, so without this the armed lock (and the
                # floor-ladder peak) died on redeploy (cache reseeds from Order.peak_pnl).
                # Bounded: 1 write/trade. Same pattern as the Jul-23 HARD_TP review fix.
                if (_sp_lk_en and _sp_lk_arm > 0 and _sp_peak_rt >= _sp_lk_arm
                        and not order_info.get('_sp_lock_peak_persisted')):
                    order_info['_sp_lock_peak_persisted'] = True
                    try:
                        async with AsyncSessionLocal() as _sp_pk_db:
                            await _sp_pk_db.execute(
                                update(Order).where(Order.id == order_id).values(peak_pnl=_sp_peak_rt)
                            )
                            await _sp_pk_db.commit()
                    except Exception:
                        pass
                if (_sp_lk_en and _sp_lk_arm > 0 and _sp_peak_rt >= _sp_lk_arm
                        and pnl_pct <= _sp_lk_sl + 0.01):
                    _sp_close_reason = "SPIKE_LOCK L1"
                    logger.warning(f"[SPIKE_LOCK] {pair} LONG: pnl={pnl_pct:.4f}% <= lock {_sp_lk_sl}% "
                                   f"(peak {_sp_peak_rt:.2f}% >= arm {_sp_lk_arm}) — CLOSING NOW!")
                elif pnl_pct <= _sp_sl_rt + 0.01:
                    _sp_close_reason = "SPIKE_SL L1"
                    logger.warning(f"[SPIKE_SL] {pair} LONG: pnl={pnl_pct:.4f}% <= {_sp_sl_rt}% (fixed, no ATR widen) — CLOSING NOW!")
                else:
                    _sp_armed_rt = bool(order_info.get('spike_armed'))
                    _sp_rungs = parse_hard_tp_ladder(getattr(
                        _th_sp_rt, 'spike_ladder_armed' if _sp_armed_rt else 'spike_ladder_unarmed', '') or '')
                    if _sp_rungs:
                        _sp_floor, _sp_lvl = hard_tp_ladder_floor(_sp_rungs, _sp_peak_rt)
                        if _sp_floor is not None and pnl_pct <= _sp_floor:
                            _sp_close_reason = f"SPIKE_FLOOR L{_sp_lvl}"
                            logger.warning(
                                f"[SPIKE_FLOOR] {pair} LONG ({'armed' if _sp_armed_rt else 'unarmed'}): "
                                f"pnl={pnl_pct:.4f}% <= floor={_sp_floor:.2f}% (peak {_sp_peak_rt:.2f}%) — CLOSING NOW!")
                    # ── Jul 28 EXIT PATCH (unarmed only — armed rides keep the wide envelope) ──
                    if _sp_close_reason is None and not _sp_armed_rt:
                        # ① mid-zone trail: peaks in [arm, first rung) previously had NO
                        #   protection (AKT +0.83 → −0.45). Standard runner-trail params.
                        _sp_trail_arm = float(getattr(_th_sp_rt, 'spike_trail_arm_pct', 0.45) or 0.0)
                        if _sp_trail_arm > 0 and _sp_peak_rt >= _sp_trail_arm:
                            _sp_atr_rt = float(order_info.get('entry_atr_pct') or 0.0)
                            # Review fix: ATR None/0 would make giveback 0 = instant TP at the
                            # arm. Fallback to a generic 0.45 leash when ATR is unavailable.
                            _sp_gb = (float(getattr(_th_sp_rt, 'runner_trail_atr_mult', 1.0) or 1.0) * _sp_atr_rt) if _sp_atr_rt > 0 else 0.45
                            _sp_trail_exit = max(_sp_peak_rt - _sp_gb,
                                                 float(getattr(_th_sp_rt, 'runner_trail_be_lock_pct', 0.10) or 0.10))
                            if pnl_pct <= _sp_trail_exit:
                                _sp_close_reason = "SPIKE_TRAIL"
                                logger.warning(
                                    f"[SPIKE_TRAIL] {pair} LONG (unarmed mid-zone): pnl={pnl_pct:.4f}% <= "
                                    f"trail={_sp_trail_exit:.2f}% (peak {_sp_peak_rt:.2f}%, ATR {_sp_atr_rt:.2f}) — CLOSING NOW!")
                        # ② stale-spike kill: no follow-through = failed explosion. Unarmed AND
                        #   peak < +0.2 after N minutes → exit (QTUM/KAS zombie class, 3h → 30min).
                        if _sp_close_reason is None:
                            _sp_stale_min = float(getattr(_th_sp_rt, 'spike_stale_kill_min', 30.0) or 0.0)
                            _sp_opened_rt = order_info.get('opened_at')
                            if _sp_stale_min > 0 and _sp_opened_rt is not None and _sp_peak_rt < 0.2:
                                from datetime import timezone as _sp_tz
                                _sp_ref = _sp_opened_rt.replace(tzinfo=_sp_tz.utc) if _sp_opened_rt.tzinfo is None else _sp_opened_rt
                                if (datetime.now(_sp_tz.utc) - _sp_ref).total_seconds() >= _sp_stale_min * 60:
                                    _sp_close_reason = "SPIKE_STALE"
                                    logger.warning(
                                        f"[SPIKE_STALE] {pair} LONG: {_sp_stale_min:.0f}min elapsed, peak "
                                        f"{_sp_peak_rt:.2f}% < 0.2 (unarmed) — failed explosion, CLOSING NOW!")
                if _sp_close_reason is not None:
                    order_info['_closing_in_progress'] = True
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                closed = await self.close_position(db, order, current_price, _sp_close_reason)
                                if closed:
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                    except Exception as e:
                        logger.error(f"[SPIKE_RT_EXIT] Error closing {pair}: {e}")
                        order_info['_closing_in_progress'] = False
                continue  # option-D owns spike-long exits; normal stack never runs

            # ════════════════════════════════════════════════════════════════
            # HARD TP (Jul 20, 2026) — flat profit cap at +1.0%, BOTH directions.
            # CF evidence (peak-based, norm ruler): baseline +$519 / Jul-20 batch
            # +$338; robust plateau 0.8-1.3% both eras (0.7% flips negative =
            # floor). Harvests the wick round-trip class the condition-based
            # runner trail structurally can't monetize (DEXE +3.64% peak →
            # +0.43% close anatomy); tail forfeit is thin (worst single runner
            # SXT −$176 vs +$659 saves). ATR-scaled variants REFUTED (0.5×ATR
            # direction-inconsistent −$2,014 BL / +$1,118 batch); atr05-leash
            # combo REFUTED (mechanisms cannibalize, −$257 BL). Profit-lock
            # only — can never fire on a losing trade. Fires after PATTERN_FIXED
            # (deliberate sub-1.0% loser-cohort locks win) but before
            # ATR_FIXED_TP / Fast Exit / trailing / EMA13. Reason "HARD_TP L1"
            # (added to BOTH post-exit tracking whitelists → Post-Exit Regret
            # rows give the revert-gate data: post-exit continuation = cut
            # runner; post-exit reversal = save).
            # 🔒 Revert gate: on N≥15 fires, revert if avg(post_exit_peak_pnl)
            # exceeds saves (runners dominate) or realized Δ vs pre-TP stack
            # goes negative.
            # ════════════════════════════════════════════════════════════════
            _hard_tp_enabled = getattr(config.trading_config.thresholds, 'hard_tp_enabled', False)
            if _hard_tp_enabled and not order_info.get('_closing_in_progress'):
                # Jul 22 (operator-directed mechanism swap): LADDER mode replaces the flat cap
                # when a per-side rung list is configured. Rungs = "trigger:offset,..." — once
                # the trade's PEAK crosses a trigger, a profit FLOOR locks at trigger-offset
                # (monotone: floors only rise). Exit fires when pnl falls TO the floor; there
                # is NO upper cap — a MIRA-class pump rides the runner trail untouched.
                # Empty/invalid ladder string -> legacy flat hard_tp_pct behavior (= revert path).
                # Reason "HARD_TP_LADDER L{n}" — startswith("HARD_TP") inherits BOTH post-exit
                # whitelists, the mechanism shadow, and its own Post-Exit Regret rows.
                _ladder_raw = getattr(config.trading_config.thresholds,
                                      'hard_tp_ladder_long' if direction == "LONG" else 'hard_tp_ladder_short',
                                      '') or ''
                _rungs = parse_hard_tp_ladder(_ladder_raw)
                _htp_fire_reason = None
                if _rungs:
                    # Note: peak_pnl is the PREVIOUS tick's peak (updated later in this
                    # iteration) — harmless: a new-peak tick is always above any floor it
                    # would lock, so the floor merely arms one tick later.
                    _htp_peak = order_info.get('peak_pnl', 0.0) or 0.0
                    _floor, _lvl = hard_tp_ladder_floor(_rungs, _htp_peak)
                    # Review fix (Jul 23): persist the in-trade peak whenever a NEW rung is
                    # crossed (bounded: <= len(rungs) writes/trade) so locked floors survive
                    # an engine restart (cache reseeds peak_pnl from the Order row).
                    if _lvl > order_info.get('_htp_persisted_lvl', 0):
                        order_info['_htp_persisted_lvl'] = _lvl
                        try:
                            async with AsyncSessionLocal() as _htp_pk_db:
                                await _htp_pk_db.execute(
                                    update(Order).where(Order.id == order_id).values(peak_pnl=_htp_peak)
                                )
                                await _htp_pk_db.commit()
                        except Exception:
                            pass
                    if _floor is not None and pnl_pct <= _floor:
                        _htp_fire_reason = f"HARD_TP_LADDER L{_lvl}"
                        logger.warning(
                            f"[HARD_TP_LADDER] {pair} {direction}: pnl={pnl_pct:.4f}% <= floor={_floor:.2f}% "
                            f"(peak={_htp_peak:.4f}%, rung L{_lvl}) — CLOSING NOW!"
                        )
                else:
                    # Legacy flat cap. Review fix (Jul 20): no `or 1.0` coalescing — 0 must
                    # reach the >0 guard so "0 = disabled" works.
                    _hard_tp_raw = getattr(config.trading_config.thresholds, 'hard_tp_pct', 1.0)
                    _hard_tp_pct = float(_hard_tp_raw) if _hard_tp_raw is not None else 1.0
                    if _hard_tp_pct > 0 and pnl_pct >= _hard_tp_pct:
                        _htp_fire_reason = "HARD_TP L1"
                        logger.warning(
                            f"[HARD_TP] {pair} {direction}: pnl={pnl_pct:.4f}% >= tp={_hard_tp_pct}% — CLOSING NOW!"
                        )
                if _htp_fire_reason:
                    order_info['_closing_in_progress'] = True
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(and_(Order.id == order_id, Order.status == "OPEN"))
                            )
                            order = result.scalar_one_or_none()
                            if order:
                                closed = await self.close_position(db, order, current_price, _htp_fire_reason)
                                if closed:
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                    except Exception as e:
                        logger.error(f"[HARD_TP] Error closing {pair}: {e}")
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

            # ===== BE-LOCK SHADOW — in-trade tick (Jul 28, observation-only) =====
            # Records, per arm threshold X, the FIRST minute P&L touched +X% and the
            # minimum P&L AFTER that touch. Feeds the time-boxed BE-lock counterfactual
            # (arm BE only if peak >= X by minute Y). Momentum book + (Jul 31) SPIKE_FADE —
            # the fade quick-lock question (touch 0.20 -> lock -0.15, pre-registered
            # frozen in #24b) needs post-touch trough ordering that peak/trough columns
            # can't resolve; capture is exit-neutral. SPIKE_CHASE stays excluded (own
            # option-D ecosystem, no lock candidate). The ⏱ shadow TABLE still filters
            # spikes out (main.py) — the momentum gate stays uncontaminated; the fade
            # cohort is read from these columns separately at N>=12.
            # Restart-tainted trades ('_belock_taint', cache reseeded with peak already
            # >= 0.15) are skipped so sequencing stays honest.
            _bl_strat = (order_info.get('entry_strategy') or '')
            if not order_info.get('_belock_taint') \
                    and (not _bl_strat.startswith('SPIKE') or _bl_strat in ('SPIKE_FADE', 'SPIKE_BOUNCE')):
                try:
                    _bl = order_info.get('_belock')
                    if _bl is None:
                        _bl = {15: [None, None], 20: [None, None], 30: [None, None]}
                        order_info['_belock'] = _bl
                    _bl_opened = order_info.get('opened_at')
                    for _bl_x, _bl_state in _bl.items():
                        if _bl_state[0] is None:
                            if pnl_pct >= _bl_x / 100.0 and _bl_opened is not None:
                                _bl_now = datetime.utcnow()
                                _bl_ref = _bl_opened if _bl_opened.tzinfo is None else _bl_opened.replace(tzinfo=None)
                                _bl_state[0] = round(max(0.0, (_bl_now - _bl_ref).total_seconds() / 60.0), 2)
                                _bl_state[1] = round(pnl_pct, 4)
                        elif _bl_state[1] is None or pnl_pct < _bl_state[1]:
                            _bl_state[1] = round(pnl_pct, 4)
                except Exception:
                    pass  # observation-only: never let shadow tracking touch the exit path

            # ===== LEASH SHADOW START — in-trade tick (observation-only) =====
            _ls_ema5 = order_info.get('cached_ema5')
            _ls_stretch = None
            if _ls_ema5 and _ls_ema5 > 0 and current_price > 0:
                _ls_stretch = ((current_price - _ls_ema5) / current_price * 100) if direction == 'LONG' \
                    else ((_ls_ema5 - current_price) / current_price * 100)
            _leash_update(order_info.get('id'), pnl_pct, peak_hint=current_peak,
                          stretch=_ls_stretch, entry_stretch=order_info.get('entry_ema5_stretch'),
                          atr=order_info.get('entry_atr_pct'))
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
            elif signal_still_active and not _is_spike_fade:
                effective_sl = order_info.get('signal_active_sl', stop_loss_pct)

            # May 22: ATR-adjusted SL widening for high-volatility pairs. Mirrors
            # trailing_atr_multiplier on the pullback side. Only WIDENS — if ATR-SL
            # is tighter than current effective_sl, keep current (no tightening).
            # Skipped when BE is active (BE floor overrides). Jul 27: skipped for
            # SPIKE_FADE (fixed −0.70 — high-ATR pumpers would widen to −1.2 and
            # double the squeeze exposure).
            if not breakeven_active and not _is_spike_fade:
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
                # 🛡 Aug 19 gate 53: quiet-pair conditional SL — applied AFTER the
                # −1.2 widen cap (the quiet width deliberately exceeds it). Same
                # eligibility as the monitor path; BE floors already returned above.
                _q_sl = _quiet_sl_for(direction, order_info.get('entry_strategy'),
                                      _entry_atr_pct)
                if _q_sl is not None and _q_sl < effective_sl:
                    effective_sl = _q_sl

            # 🔒 Aug-3 SPIKE PROFIT LOCK (#24b ② verdict, fade-capture N=13 S$87/K$0):
            # FADE/BOUNCE leg (_is_spike_fade covers both) — once peak touches the arm,
            # the fixed −0.70 tightens to the lock level. Winners' post-touch dips
            # bottom at −0.09 → −0.15 sits below the band. Close reason SPIKE_LOCK
            # (SPIKE_ prefix rides both post-exit whitelists). Chase gets the same leg
            # inside its option-D chain + monitor backstop.
            _spike_lock_active_rt = False
            if _is_spike_fade:
                _lk_en_rt = bool(getattr(config.trading_config.thresholds, 'spike_lock_enabled', True))
                # Aug-10 PM: FADES EXEMPT from the lock (spike_lock_exempt_fade) — under the
                # −1.5 SL the lock is 0 saves / 4 kills on covered fades (multi-wave pumps
                # arm on wave-1 pullback, eject before the wave-2 wick the wide SL rides).
                # BOUNCE keeps the lock (species off; phantom semantics preserved).
                if (_lk_en_rt and bool(getattr(config.trading_config.thresholds, 'spike_lock_exempt_fade', True))
                        and (order_info.get('entry_strategy') or "") == "SPIKE_FADE"):
                    _lk_en_rt = False
                _lk_arm_rt = float(getattr(config.trading_config.thresholds, 'spike_lock_arm_pct', 0.20) or 0.0)
                _lk_sl_rt = float(getattr(config.trading_config.thresholds, 'spike_lock_sl_pct', -0.15) or -0.15)
                if (_lk_en_rt and _lk_arm_rt > 0 and current_peak >= _lk_arm_rt
                        and _lk_sl_rt > effective_sl):
                    effective_sl = _lk_sl_rt
                    _spike_lock_active_rt = True

            # Check if stop loss triggered (epsilon 0.01% to avoid boundary precision issues)
            if pnl_pct <= effective_sl + 0.01:
                if _spike_lock_active_rt:
                    close_reason = f"SPIKE_LOCK L{tp_level}"
                elif breakeven_active:
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
            
            # ─── FLIP STRPK RUNNER-TRAIL — realtime tick exit (Jun 16, Fix A) ───
            # The SHORT runner stretch-trail for flip shorts fires HERE (every WS tick) instead
            # of the 1s monitor, so peak-stretch is tracked and the trail checked at TICK
            # resolution (matching the leash-shadow) — not under-tracking the peak and trailing
            # out on a 1-second bounce. Same arm/K, same cached_ema5 the shadow uses; hard SL
            # above kept priority. Fail-open (a strpk error must never break the realtime loop).
            if (_is_flip and direction == "SHORT" and _ls_stretch is not None
                    and not order_info.get('_closing_in_progress')):
                try:
                    _sp_th = config.trading_config.thresholds
                    _sp_src = (order_info.get('entry_strategy') or "")[5:]
                    _sp_on = ((_sp_src == "FAN_RATIO_GATE" and getattr(_sp_th, 'flip_fan_runner_strpk', False))
                              or (_sp_src != "FAN_RATIO_GATE" and getattr(_sp_th, 'flip_runner_strpk_shorts', False)))
                    if _sp_on:
                        _sp_pk = order_info.get('runner_peak_stretch')
                        if _sp_pk is None or _ls_stretch > _sp_pk:
                            _sp_pk = _ls_stretch
                            order_info['runner_peak_stretch'] = _sp_pk
                        _sp_arm = float(getattr(_sp_th, 'runner_trail_short_arm_peak', 0.45) or 0.45)
                        # Jun 16: ATR-floored give-back (chandelier) is the primary trail — exit
                        # when P&L retraces > N×ATR% from peak (a normal bounce <1 ATR can't trip
                        # it; only a real reversal does). Fixes the ratio-trail collapsing on a
                        # tiny freshly-armed peak. Fallback to K×peak_stretch when use_atr=false.
                        _sp_use_atr = getattr(_sp_th, 'runner_trail_short_use_atr', True)
                        _sp_atr = order_info.get('entry_atr_pct')
                        _sp_fire = False; _sp_why = ""; _sp_bound = None
                        # Jun 18: 0.005pp float-tolerance on the arm, matching the standard trailing
                        # path (line ~8567, effective_tp_target − 0.005). Without it the flip arm was
                        # STRICTER than the UI/standard arm → a peak of +0.4471% showed armed in the UI
                        # but the BE-ratchet never engaged (JTO −1.04 SL that should have locked +0.10).
                        if current_peak >= _sp_arm - 0.005:
                            if _sp_use_atr and _sp_atr and _sp_atr > 0:
                                _sp_n = float(getattr(_sp_th, 'runner_trail_short_atr_mult', 0.5) or 0.5)
                                # Jun 17 PM give-back CAP: never give back more than frac×peak, so a high-ATR
                                # pair can't surrender the whole runner to the lock (AGT +2.42->+0.10). The floor
                                # then RISES with the peak. Jun 17 BE-ratchet (lock) still backstops round-trips.
                                _sp_giveback = _sp_n * _sp_atr
                                _sp_frac = float(getattr(_sp_th, 'runner_trail_short_giveback_frac', 0.0) or 0.0)
                                _sp_capped = (_sp_frac > 0 and current_peak > 0 and _sp_frac * current_peak < _sp_giveback)
                                if _sp_capped:
                                    _sp_giveback = _sp_frac * current_peak
                                _sp_atr_floor = current_peak - _sp_giveback
                                _sp_lock = float(getattr(_sp_th, 'runner_trail_short_be_lock_pct', 0.10) or 0.10)
                                _sp_ratchet = getattr(_sp_th, 'runner_trail_short_be_ratchet_enabled', True)
                                _sp_floor = max(_sp_atr_floor, _sp_lock) if _sp_ratchet else _sp_atr_floor
                                # Aug-12 🛞 NEGFLOOR RIDE (operator ship, IDOL/BICO class — DECISION_LOG 2026-08-12 (3)):
                                # a NEGATIVE floor (giveback > peak on a wild-ATR pair) means the trail is
                                # definitionally broken — firing it exits an ARMED WINNER red. Suppress the
                                # trail instead; the hard SL bounds the tail; the trail resumes automatically
                                # once the peak outgrows the giveback. Both lifetime cases re-ran the short's
                                # way post-red-exit (IDOL +1.37 / BICO +1.25); May-25 sibling precedent
                                # (suppress+ride +$1,506 vs +$190). REVERT: 2 ridden trades to full SL
                                # without re-arming before the 4th positive tally -> clamp-at-BE instead.
                                if _sp_floor < 0:
                                    if not order_info.get('_negfloor_ride_logged'):
                                        order_info['_negfloor_ride_logged'] = True
                                        logger.warning(f"[RUNNER_NEGFLOOR_RIDE] {pair} SHORT(flip): floor {_sp_floor:.3f}% < 0 (peak {current_peak:.2f} < giveback {_sp_giveback:.2f}) — trail SUPPRESSED, hard SL governs until peak outgrows giveback")
                                elif pnl_pct <= _sp_floor:
                                    _sp_fire = True
                                    _sp_bound = ("lock" if (_sp_ratchet and _sp_lock > _sp_atr_floor)
                                                 else "cap" if _sp_capped else "atr")
                                    _sp_why = f"pnl {pnl_pct:.3f}% <= floor {_sp_floor:.3f}% [{_sp_bound}] (peak={current_peak:.2f}%, giveback={_sp_giveback:.3f}%)"
                            else:
                                _sp_k = float(getattr(_sp_th, 'runner_trail_short_k', 0.5) or 0.5)
                                if _sp_pk > 0 and _ls_stretch <= _sp_pk * _sp_k:
                                    _sp_fire = True
                                    _sp_bound = "stretch"
                                    _sp_why = f"stretch {_ls_stretch:.3f} <= {_sp_k}x peak {_sp_pk:.3f} (peak={current_peak:.2f}%)"
                        if _sp_fire:
                            logger.info(f"[REALTIME_RUNNER_TRAIL] {pair} SHORT: {_sp_why} -> close")
                            order_info['_closing_in_progress'] = True
                            async with AsyncSessionLocal() as db:
                                _spr = await db.execute(select(Order).where(and_(Order.id == order_id, Order.status == "OPEN")))
                                _sp_order = _spr.scalar_one_or_none()
                                if _sp_order:
                                    _sp_order.runner_peak_stretch = _sp_pk  # persist for the CSV/report
                                    _sp_order.runner_trail_bound = _sp_bound  # Jun 17: lock/atr/stretch — which mechanism bound this exit
                                    _sp_closed = await self.close_position(db, _sp_order, current_price, "RUNNER_TRAIL")
                                    if _sp_closed:
                                        logger.info(f"[REALTIME_RUNNER_TRAIL] {pair} closed at {current_price} pnl={pnl_pct:.4f}%")
                                        async with _cache_lock:
                                            _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                                    else:
                                        order_info['_closing_in_progress'] = False
                            continue
                except Exception as _sp_err:
                    logger.error(f"[REALTIME_RUNNER_TRAIL] {pair}: {_sp_err}")
                    order_info['_closing_in_progress'] = False
            # ─── FLIP STRPK RUNNER-TRAIL END ───

            # ─── Jul 6: REALTIME LONG RUNNER ATR-FLOOR (Fix A extended to momentum longs). Once a
            # LONG arms (peak ≥ runner_trail_arm_peak), the runner handoff SUPPRESSES the tight 1s
            # trailing — but its ATR-floor was only checked by the slow monitor loop (~3-4 min for
            # 50 pairs), leaving a blind window on fast movers: PUMP closed 0.18pp under its floor
            # (−$40), USELESS 0.07pp (−$15), AAVE ~0. Same formula as indicators.check_exit_conditions
            # (live: N=1.0, ratchet OFF → floor CAN be negative on high-ATR pairs — by design; the
            # lock-ON counterfactual is the lock-aware atr10 shadow), evaluated on every tick; the
            # monitor path stays as backstop. Fail-open. ───
            if (not _is_flip and direction == "LONG" and not order_info.get('_closing_in_progress')):
                try:
                    _rl_th = config.trading_config.thresholds
                    if getattr(_rl_th, 'runner_trail_enabled', False) and getattr(_rl_th, 'runner_trail_use_atr', True):
                        _rl_arm = float(getattr(_rl_th, 'runner_trail_arm_peak', 0.45) or 0.45)
                        _rl_amin = float(getattr(_rl_th, 'runner_trail_atr_min', 0.0) or 0.0)
                        _rl_atr = order_info.get('entry_atr_pct')
                        if (current_peak >= _rl_arm - 0.005 and _rl_atr and _rl_atr > 0
                                and (_rl_amin <= 0 or _rl_atr >= _rl_amin)):
                            _rl_n = float(getattr(_rl_th, 'runner_trail_atr_mult', 0.5) or 0.5)
                            if (order_info.get('entry_strategy') or '') == 'SPIKE_BOUNCE':
                                # Jul 31 🏀 strategy-scoped trail N (dump-inflated entry ATR)
                                _rl_n = float(getattr(_rl_th, 'spike_bounce_trail_atr_mult', 0.5) or 0.5)
                            _rl_gb = _rl_n * _rl_atr
                            _rl_frac = float(getattr(_rl_th, 'runner_trail_giveback_frac', 0.0) or 0.0)
                            _rl_capped = (_rl_frac > 0 and current_peak > 0 and _rl_frac * current_peak < _rl_gb)
                            if _rl_capped:
                                _rl_gb = _rl_frac * current_peak
                            _rl_raw_floor = current_peak - _rl_gb
                            _rl_floor = _rl_raw_floor
                            if getattr(_rl_th, 'runner_trail_be_ratchet_enabled', True):
                                _rl_floor = max(_rl_floor, float(getattr(_rl_th, 'runner_trail_be_lock_pct', 0.10) or 0.10))
                            if pnl_pct <= _rl_floor:
                                logger.info(f"[REALTIME_RUNNER_TRAIL] {pair} LONG: pnl {pnl_pct:.3f}% <= floor {_rl_floor:.3f}% (peak={current_peak:.2f}%, giveback={_rl_gb:.3f}%) -> close")
                                order_info['_closing_in_progress'] = True
                                async with AsyncSessionLocal() as db:
                                    _rlr = await db.execute(select(Order).where(and_(Order.id == order_id, Order.status == "OPEN")))
                                    _rl_order = _rlr.scalar_one_or_none()
                                    if _rl_order:
                                        _rl_order.runner_trail_bound = ("lock" if _rl_floor > _rl_raw_floor
                                                                        else "cap" if _rl_capped else "atr")
                                        _rl_closed = await self.close_position(db, _rl_order, current_price, "RUNNER_TRAIL")
                                        if _rl_closed:
                                            logger.info(f"[REALTIME_RUNNER_TRAIL] {pair} LONG closed at {current_price} pnl={pnl_pct:.4f}%")
                                            async with _cache_lock:
                                                _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                                        else:
                                            order_info['_closing_in_progress'] = False
                                    else:
                                        order_info['_closing_in_progress'] = False
                                continue
                except Exception as _rl_err:
                    logger.error(f"[REALTIME_RUNNER_TRAIL] {pair} LONG: {_rl_err}")
                    order_info['_closing_in_progress'] = False
            # ─── REALTIME LONG RUNNER ATR-FLOOR END ───

            # ─── Jul 29: REALTIME NON-FLIP SHORT RUNNER ATR-FLOOR (3rd appearance of the
            # monitor-latency bug class; direction mirror of the Jul-6 LONG block). Non-flip
            # SHORTS (momentum shorts + SPIKE_FADE) had their runner floor evaluated ONLY in
            # the slow monitor loop — the realtime mirrors covered LONGs (Jul-6) and FLIP
            # shorts (Jun-16 Fix A) but this class predated the fade program and was starved:
            # PROM fade Jul-29 peaked +1.89 with a 0.5×ATR floor ≈ +1.58, fell through it
            # between monitor passes, ladder backstop caught it at +0.52 (−1.0pp latency).
            # Same formula as the monitor path (runner_trail_short_*: N=0.5, arm 0.45,
            # ratchet per config), evaluated on every tick; monitor stays as backstop.
            # Fail-open. Zero config changes. ───
            if (not _is_flip and direction == "SHORT" and not order_info.get('_closing_in_progress')):
                try:
                    _rs_th = config.trading_config.thresholds
                    if getattr(_rs_th, 'runner_trail_short_enabled', False) and getattr(_rs_th, 'runner_trail_short_use_atr', True):
                        _rs_arm = float(getattr(_rs_th, 'runner_trail_short_arm_peak', 0.45) or 0.45)
                        _rs_amin = float(getattr(_rs_th, 'runner_trail_short_atr_min', 0.0) or 0.0)
                        _rs_atr = order_info.get('entry_atr_pct')
                        if (current_peak >= _rs_arm - 0.005 and _rs_atr and _rs_atr > 0
                                and (_rs_amin <= 0 or _rs_atr >= _rs_amin)):
                            _rs_n = float(getattr(_rs_th, 'runner_trail_short_atr_mult', 0.5) or 0.5)
                            _rs_gb = _rs_n * _rs_atr
                            _rs_frac = float(getattr(_rs_th, 'runner_trail_short_giveback_frac', 0.0) or 0.0)
                            _rs_capped = (_rs_frac > 0 and current_peak > 0 and _rs_frac * current_peak < _rs_gb)
                            if _rs_capped:
                                _rs_gb = _rs_frac * current_peak
                            _rs_raw_floor = current_peak - _rs_gb
                            _rs_floor = _rs_raw_floor
                            if getattr(_rs_th, 'runner_trail_short_be_ratchet_enabled', False):
                                _rs_floor = max(_rs_floor, float(getattr(_rs_th, 'runner_trail_short_be_lock_pct', 0.10) or 0.10))
                            # Aug-12 🛞 NEGFLOOR RIDE — see the flip-site comment (same class, same revert)
                            if _rs_floor < 0:
                                if not order_info.get('_negfloor_ride_logged'):
                                    order_info['_negfloor_ride_logged'] = True
                                    logger.warning(f"[RUNNER_NEGFLOOR_RIDE] {pair} SHORT: floor {_rs_floor:.3f}% < 0 (peak {current_peak:.2f} < giveback {_rs_gb:.2f}) — trail SUPPRESSED, hard SL governs until peak outgrows giveback")
                            elif pnl_pct <= _rs_floor:
                                logger.info(f"[REALTIME_RUNNER_TRAIL] {pair} SHORT(non-flip): pnl {pnl_pct:.3f}% <= floor {_rs_floor:.3f}% (peak={current_peak:.2f}%, giveback={_rs_gb:.3f}%) -> close")
                                order_info['_closing_in_progress'] = True
                                async with AsyncSessionLocal() as db:
                                    _rsr = await db.execute(select(Order).where(and_(Order.id == order_id, Order.status == "OPEN")))
                                    _rs_order = _rsr.scalar_one_or_none()
                                    if _rs_order:
                                        _rs_order.runner_trail_bound = ("lock" if _rs_floor > _rs_raw_floor
                                                                        else "cap" if _rs_capped else "atr")
                                        _rs_closed = await self.close_position(db, _rs_order, current_price, "RUNNER_TRAIL")
                                        if _rs_closed:
                                            logger.info(f"[REALTIME_RUNNER_TRAIL] {pair} SHORT(non-flip) closed at {current_price} pnl={pnl_pct:.4f}%")
                                            async with _cache_lock:
                                                _open_orders_cache[pair] = [o for o in _open_orders_cache.get(pair, []) if o['id'] != order_id]
                                        else:
                                            order_info['_closing_in_progress'] = False
                                    else:
                                        order_info['_closing_in_progress'] = False
                                continue
                except Exception as _rs_err:
                    logger.error(f"[REALTIME_RUNNER_TRAIL] {pair} SHORT(non-flip): {_rs_err}")
                    order_info['_closing_in_progress'] = False
            # ─── REALTIME NON-FLIP SHORT RUNNER ATR-FLOOR END ───

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
                    # Jul 22 BUG FIX (operator-identified): suppress the legacy tight-trail for ALL
                    # runner-eligible MOMENTUM trades, not only armed ones. The old `peak >= arm`
                    # condition left the tight-trail alive in the sub-arm zone, creating a kill zone
                    # just under the arm — both lifetime TRAILING_STOP closes peaked 0.4489/0.4494 vs
                    # arm 0.45 (MIRA closed +0.45%, ran +19.1% in the next 30min). Jun-24 design
                    # intent: the runner trail IS the exit; sub-arm exits = EMA13-strict / hard SL /
                    # HARD_TP / NO_EXPANSION backstops. If runner_trail_atr_min > 0, low-ATR trades
                    # remain tight-trail-managed (they are not runner-eligible). Flips keep the
                    # armed-only suppression (their strpk exit design assumed it; flip-LONGs deferred).
                    _rt_eligible = _rt_en and (_rt_amin <= 0 or (_rt_atr is not None and _rt_atr >= _rt_amin))
                    if _rt_eligible:
                        if not _is_flip:
                            _handoff_suppress = True
                        elif _flip_strpk_ok and _rt_peak >= _rt_arm - 0.005:
                            # Aug-12: −0.005 tolerance MATCHES the tier-activation tolerance, so the
                            # strpk runner always wins the arm boundary — closes the 0.005pp window
                            # where a flip-short tier stop could fire (MIRA class) now that tp_min
                            # and the runner arm share the 0.40 boundary (DECISION_LOG 2026-08-12 (5))
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
                       PairData.ema5_prev3, PairData.updated_at).where(PairData.pair.in_(pair_names))
            )
            # Jul 24 (review fix I1): extended-universe (spike-scanner) pairs are outside the
            # top-50 scan, so their PairData row is absent or STALE (refreshed only for scanned
            # pairs). Stale EMAs must not feed the realtime EMA13/stack exits — skip rows older
            # than 10 min (cache emas stay None -> those exits skip cleanly; price-based exits
            # SL/BE/trailing/ladder are websocket-fed and unaffected).
            from datetime import timezone as _tz_pd
            _pd_now = datetime.now(_tz_pd.utc)
            for row in sig_result:
                _pd_ts = getattr(row, 'updated_at', None)
                if _pd_ts is not None:
                    _pd_ts = _pd_ts.replace(tzinfo=_tz_pd.utc) if _pd_ts.tzinfo is None else _pd_ts
                    if (_pd_now - _pd_ts).total_seconds() > 600:
                        continue
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
                # Jul 27 spike ship: fixed SLs survive cache rebuilds/restarts via entry_strategy
                'stop_loss': (float(getattr(config.trading_config.thresholds, 'spike_sl_pct', -1.2) or -1.2) if (order.entry_strategy or '') == 'SPIKE_CHASE'
                              else float(getattr(config.trading_config.thresholds, 'spike_fade_sl_pct', -1.50) or -1.50) if (order.entry_strategy or '') == 'SPIKE_FADE'
                              else float(getattr(config.trading_config.thresholds, 'spike_bounce_sl_pct', -0.70) or -0.70) if (order.entry_strategy or '') == 'SPIKE_BOUNCE'
                              else conf_config.stop_loss),
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
                # Jul 27 option-D state resumed across restarts (SPIKE_CHASE)
                'spike_armed': bool(getattr(order, 'spike_armed', False)),
                'spike_rsi_max': getattr(order, 'spike_rsi_max', None),
                # Jul 28 BE-LOCK SHADOW: a trade whose cache reseeds mid-flight with peak
                # already >= 0.15 may have touched an arm threshold BEFORE the restart —
                # its first-touch time is unrecoverable, so taint it (all six columns
                # stay NULL, the CF excludes it). Fresh trades track from tick 1.
                '_belock_taint': bool((order.peak_pnl or 0.0) >= 0.15),
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
                            # Jul 28 BE-LOCK SHADOW fix: carry the tracking dict + taint flag
                            # across PERIODIC cache rebuilds. Without this, the reseed default
                            # ('taint if DB peak >= 0.15') re-fired on every routine refresh and
                            # wiped every armed trade's state — the taint guard is meant for
                            # RESTARTS only (no old_info generation to inherit from).
                            new_info['_belock'] = old_info.get('_belock')
                            new_info['_belock_taint'] = bool(old_info.get('_belock_taint'))
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
