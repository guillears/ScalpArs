#!/usr/bin/env python3
"""
build_master_pool.py — the MASTER cross-era pool with the current stack applied (Aug-10 2026).

Merges the three canonical era pools into ONE file with stack columns, so every
cross-era analysis starts from the same screened source instead of re-deriving
the gate logic inline (the error class behind the d6 double-count incident).

  era          BASE (screened Jun-17→Jul-10) / B1 (Jul-11→31 anchor) / B2 (Jul-31→Aug-10) /
               B3 (Aug-11→24) / B4 (Aug-24→26 first live) / B5 (Aug-26→27) / B<n> auto-discovered
               from reports/BASELINE<n>_*.csv for n ≥ 6 (archive each batch there at its pre-reset export)
  is_probe     *_PROBE cell fires (headline tables exclude these — full-size-only rule)
  is_door      NONEXP_CALM3D fires
  stack_keep   would TODAY'S entry stack admit this trade?
  stack_block_reason  first gate that catches it (overlap audits: group by this)
  stack_pnl    CF-adjusted P&L for kept trades (arm re-exit, sprint de-mux) — APPROXIMATION
  stack_pct    P&L % under today's rules for kept trades (NaN when not kept): path CFs + FRENZY +TP move it, size
               re-prices never do — read pct from here, never from a stack_pnl / pnl ratio
  stack_version  regenerate after EVERY filter ship: ./venv/bin/python scripts/build_master_pool.py

Raw pools stay untouched (ground truth). stack_keep is exact (entry gates);
stack_pnl layers mechanism counterfactuals — analyses must say which they used.
"""
import warnings; warnings.filterwarnings('ignore')
import re
import pandas as pd, numpy as np
from datetime import datetime

STACK_VERSION = "2026-10-08a"  # 10-08a (DECISION_LOG 250, operator declared overrides): FRENZY sleeve fills (LONG / WIDE / LITE, every era) priced at TODAY's FIXED +3 / −3 exit — a fill whose recorded peak_pnl (net, the TP's own basis) reached +3 books +3 (ASSUMPTION: the +3 TP fills first, at the line; a fill live-stopped at −3 before touching +3 never has peak ≥ 3 and stays as traded; 12 h-cap / other closes below +3 keep their actual) — replaces the 10-06d lock pricing · WIDE hold-green ATR cap 2.5 → 3.0 (frenzy_max_atr_pct 3.0; the builder replays no FRENZY_LONG ATR gate — every master FRENZY_LONG fill had ATR ≤ 2.5, and refusals 2.5–3.0 are not in the pool) · FRENZY_BEARISH_DAY: FRENZY / WIDE / LITE fills with entry_btc_1d_ret_pct < 0 ∧ entry_btc_trend_gap_pct < 0 refused (services.frenzy.frenzy_bearish_day; unreadable = kept, fail-open like the engine; judged after hold-green like the engine's last gate, before the chop-burst pass). Prior 10-06d — 10-06d: B17 archived (Oct-3→Oct-6) · FRENZY sleeve fills opened before the live lock (DECISION_LOG 205, deploy 2026-10-05 15:49) priced with the lock — peak ≥ +3 books max(min(+2, +3), peak − 2) — instead of the retired fixed +3 TP (later fills keep their real pct) · FRENZY_WIDE hold-green rule (DECISION_LOG 231) replayed with the engine's own function before the chop-burst pass (streak: stamp entry_frenzy_above_streak, else reports/WIDE_STREAK_REBUILD.csv, else refused = fail-closed) (DECISION_LOG 234). Prior 10-06c — 10-06c: TODAY's cell sizing = ONE pure rule, today_size_scale() (DECISION_LOG 233), shared with scripts/screen_pool.py pnl_current — closes the two gaps vs the engine: W2+W1 momentum shorts 2× → 1× (cell de-muxed 2026-07-30; every SHORT pattern cell is 1× today) and UNMATCHED longs at pair-vol ratio ≥ 0.90 → 1× (the Jul-10 crowded-entry de-mux; was 1.5× here, the "pre-Jul-10 gap"). The rule also strips the cell LEVERAGE multiplier (SOL 06-18 W2+W1 short traded 2× at 30× → 1× at 20×) and applies to the unrounded P&L (single rounding: 21 rows move ±$0.01). The crowd-sprint de-mux moved from the first pass to the final re-price (an ARM040 row in a sprint window now de-muxes too; stack_pct untouched; a re-sized path-CF row folds the factor into stack_ticket_scale so stack_pnl / scale / notional stays its pct). Non-probe kept $7,583.59 → $7,458.50. Prior 10-06b — 10-06b: NONEXP_CALM3D door 2× → 1× (its cell verdict fired: fresh 2× fires since Sep-23 net-negative; operator, DECISION_LOG 225): kept CALM3D rows above 1× re-priced stack_pnl ÷ cell multiplier (size-only; stack_pct unchanged). Prior 10-06a — 10-06a: FLIP short 2× cells NEGDI15 + TG_SHALLOW → 1× (NEGDI15's own revert gate fired, TG_SHALLOW ✗ HARMFUL on real fills; operator, DECISION_LOG 220): every kept FLIP row above 1× (the tagged cells + one pre-tag June ×2 BASE row) re-priced stack_pnl ÷ cell multiplier — no flip cell sizes above 1× today. Prior 10-05b — 10-05b: LONG_HEAT_BLOCK re-scope REVERTED to the Sep-18 3-leg rule (BTC slope ≥ 0.07 ∧ BTC RSI prev ≥ 64 ∧ bull ≥ 80, washed-out exempt; DECISION_LOG 208 — the re-scope's pre-committed gate fired). Prior 10-05a — 10-05a: UNMATCHED momentum-long cell 2× → 1.5× (and the quiet boost 2 → 1.5; operator, DECISION_LOG 206): kept UNMATCHED longs above 1.5× re-priced × 1.5 / cell multiplier (CALM3D door unchanged at 2×; rows the sprint de-mux already set to 1× are left at 1×; known pre-Jul-10 gap: PVR ≥ 0.90 rows are not de-muxed to 1×). The FRENZY lock-then-trail exit (205) is path-dependent → NOT re-priced (kept at the +3 TP pricing; forward read = scout FRENZY_LOCK gate). Prior 10-04b — 10-04b: LONG_CHOP_BURST — MOMENTUM longs refused when BTC eff72 ≤ 0.007 (live stamp entry_btc_eff72, else the validated 864-bar rebuild from the k5m_full BTC cache) AND another kept non-probe non-MANUAL bot fill of the same era opened 0…120 s earlier (sub-line A; operator ARMED override, DECISION_LOG 201; rule = engine long_chop_burst_block). Prior 10-04a — 10-04a: SURGE_SHORT_OFF (surge_short_enabled false) + sleeve fills (FRENZY / WIDE / SURGE_LONG / BEARRUN) re-priced at today's size and the FRENZY +3 TP (DECISION_LOG 200). Prior 10-03a — 10-03a: FRENZY_LONG / FRENZY_WIDE tagged FRENZY_SLEEVE (own-sleeve observation, never a momentum row; before this a FRENZY fill fell into the momentum-long gates). Prior 10-01b — 10-01b: EVERY kept capped fade re-priced at the 0.5 % ticket (+ stack_ticket_scale column; DECISION_LOG 168). 10-01a: CF_FADE_CAP05 — kept fades on pairs ≥ $10M throttled by the old 0.1 % cap re-priced at min(desired, 0.5 % × vol) (DECISION_LOG 165/167; 3 rows). Prior 09-29b — b: FADE_LAGGARD — SPIKE_FADE refused when pair daily Wilder −DI(14) > 16.1 AND BTC 4h EMA50/EMA200 gap > 0 (laggard squeeze; operator ARMED override at master N=10 on 295 backtest fills, DECISION_LOG 128; rule = indicators.fade_laggard_block; inputs = stamps entry_pair_1d_ndi / entry_btc_4h_ema50_200_gap_pct, else the closed-bar feature factory from k5m_full). Prior: "2026-09-29a"  # a: LONG_RSI_MOM_LOADX — momentum longs (unmatched + doors) refused when RSI(12) < RSI two candles ago AND pair ADX < long_rsi_momentum_adx_max (21; declared override at master N=10, DECISION_LOG 126; rule = indicators.rsi_mom_loadx_block; stamps entry_rsi / entry_rsi_prev (= rsi_prev2) / entry_adx). Prior: "2026-09-27b"  # b: CALM3D_BTC_ATR_MIN — NONEXP_CALM3D door longs refused at BTC 5m ATR% < 0.08 (dead tape; operator ARMED override, DECISION_LOG 118; rule = engine calm3d_btc_atr_floor_block). Prior: "2026-09-27a"  # a: FADE_FRESHBREAK stamp proxy applied to PRE-SHIP fills only (opened < 2026-08-10T13:48:19 UTC, commit a10a879) — the live gate reads RSI(12) rsi_prev1 at trigger, the stamp entry_rsi_prev is rsi_prev2, so post-ship fills (which already passed the live gate) were wrongly removed (6 winners, Sep-27 audit). Prior: 2026-09-25c # c: LONG_HEAT_BLOCK re-scoped to bull breadth ≥85 only (BTC slope/RSI legs off, washed-out exemption kept; declared override, DECISION_LOG 116). Prior: 2026-09-25b # b: MOM_SHORT_C1_REGIME — C1 momentum shorts refused when BTC is STRONG_BEAR (operator ARMED override, DECISION_LOG 114; rule = engine mom_short_c1_regime_block). Prior: 2026-09-25a # a: FLIP_FAN_WEAK_BOUNCE — FAN flip-shorts refused when pair EMA13−EMA50 gap < 0 AND EMA20 slope < 0.15 (operator ARMED override at N=8, DECISION_LOG 113; rule = engine flip_fan_weak_bounce); B12 snapshot as-of 09-25. Prior: 2026-09-24b # b: FADE_BRSI 45→50 — the Aug-5 ceiling's own pre-committed revert fired (DECISION_LOG 112); label FADE_BRSI45→FADE_BRSI50. Prior: 2026-09-24a # a: CF_FADE_LATE_ARM — SPIKE_FADE never armed, open past 15 min, peak after 15 in [0.30,0.40) → re-priced to the late trail floor (stamps-only, optimistic: exposed late winners not repriced; DECISION_LOG 111). Prior: 2026-09-23a # a: LONG_MEGACAP_BLOCK — momentum longs (unmatched + doors) refused at raw eligible-universe rank ≤ 10 (operator override at N=10, DECISION_LOG 110; rule = engine long_megacap_block). Prior: 2026-09-18b # b: LONG_HEAT_BLOCK — momentum longs (unmatched + doors) refused at BTC slope≥0.07 ∧ BTC RSI prev≥64 ∧ bull≥80 unless BTC ≤−10% vs its 30d high (DECISION_LOG Sep-18 (70); rule = engine long_heat_eval, 30d reading = stamped column else reports/btc_off30d_hourly.csv); era B8 (Sep 16-18) added. Prior: 2026-09-18a # a: MOM_SHORT_PAIRVOL — momentum shorts blocked at pair-vol ratio ≥ 0.86 (ceiling tightened 1.0→0.86, DECISION_LOG Sep-18 (68)). Prior: 2026-09-16a # a: FLIP_FAN_BTC_EMA13 — FAN_RATIO_GATE shorts blocked when BTC dist-EMA13 > -0.08 (Aug-23 live gate, builder gap caught Sep-16). Prior: 2026-09-15a # a: gate 60 BEARRUN_SHORT — 1× probe fills PROBE_EXEMPT, armed fills own-sleeve label (never MOM-short). Prior: 2026-09-14a # a: FADE_MAXVOL — SPIKE_FADE blocked at 24h vol ≥ $20M (Sep-14 operator override, DECISION_LOG 55); engine tests it FIRST among the fade gates. Prior: 2026-08-16a # a: FAKE_BULL_GUARD gate REMOVED (guard reverted by locked gate 47 after forward refutation — 12-block replay 6W/6L). Restores the 2026-08-10c keep-set. NOTE: cap35 (8108a60) is EXIT-side and path-dependent — stack_pnl deliberately NOT re-priced for it (floor-bound CF is optimistic; forward accounting = bound='cap' tallies).
WIDE_HG_STREAK_FROZEN, WIDE_HG_MAX_ATR_FROZEN = 12.0, 3.0   # frenzy_wide_hold_green_streak (231) / frenzy_max_atr_pct — 2.5 at the 231 ship, 3.0 since 10-08a (DECISION_LOG 250; pinned by tests)
WIDE_HG_MAX_ATR_PRE_1008 = 2.5                               # 10-08a (deep review): the cap in force BEFORE the raise
FRENZY_ATR30_FROM = "2026-10-09T00:00:00"                    # 10-08a: fills opened from here are judged at 3.0. CONSERVATIVE (≥ the Oct-8 deploy, push
#   pending): a WIDE fill in between is judged at 2.5. Pin it to the deploy (push + 10 min) at the next archive. (No master fill is that late today.)
FRENZY_TP_FROZEN = 3.0                                       # 10-08a: the live fixed take profit (frenzy_tp_pct 3, frenzy_lock_arm_pct 0)
FRENZY_LOCK_LIVE_FROM = "2026-10-05T15:49"   # RETIRED 10-08a (fixed TP back, every era priced the same) — kept for the record. 10-06c: the live lock (DECISION_LOG 205) deploy 8743989 — fills opened later already carry its exit
G = 'entry_pair_ema20_ema50_gap_pct'   # holds EMA13-50 (known misnomer — do not rename)

# 🧯 FADE_FRESHBREAK stamp-proxy scope (Sep-27). The live gate reads the RSI(12) of the candle BEFORE the trigger
# (rsi_prev1 — the base the spike launched from: scanner = hand-rolled Wilder `_spike_rsi12`, top-50 hook =
# indicators['rsi_prev1']); the order stamp `entry_rsi_prev` is ta-RSI rsi_prev2, one bar earlier. pgap has the same
# source in both (EMA13/EMA50 of the trigger-cycle indicators). A fill opened after the gate shipped already passed the
# real gate, so re-screening it on the wrong bar can only remove a live-admitted fade (6 winners, Sep-27 audit).
# Pre-ship fills have no rsi_prev1 stamp → the rsi_prev2 proxy is the best available (approximate, kept as-is).
# ⚠ Valid only while the live thresholds equal the shipped ones — a retune means fills opened under the old values need
# re-screening (tests/test_fade_freshbreak_scope.py pins these to trading_config.json, so a retune fails the suite).
FADE_FB_SHIP_UTC = "2026-08-10T13:48:19"   # commit a10a879 (10:48:19 −03). Last pre-ship fade 08-10 10:04, none until the
                                           # B3 reset → ≥3h buffer, so commit-vs-deploy lag / tz cannot misplace a fill
FADE_FB_RSI_PREV_MIN, FADE_FB_PGAP_MIN = 44.0, -0.40   # live spike_fade_fb_rsi_prev_min / spike_fade_fb_pgap_min
FADE_LAG_SHIP_UTC = "2026-09-29T20:30:00"   # 🪤 fade laggard gate ship (stamps exist on every fade fill opened after this)


FRENZY3 = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")


def wide_hg_cap_at(opened_at):
    """⬆ 10-08a (deep review): the WIDE hold-green ATR cap for a fill — the cap IN FORCE when it opened (2.5 before FRENZY_ATR30_FROM, 3.0 from
    it). A pre-raise WIDE fill with ATR in (2.5, 3.0] on a red / flat candle was FRENZY_LONG territory under today's 3.0 — never re-coded
    GREEN_BAR and kept as WIDE: at 2.5 it codes ATR_HIGH → FRENZY_WIDE_ATR_HIGH (refused as a WIDE row; the pool cannot show the FRENZY_LONG
    fill today's rules would have opened instead)."""
    return WIDE_HG_MAX_ATR_PRE_1008 if str(opened_at)[:19].replace(" ", "T") < FRENZY_ATR30_FROM else WIDE_HG_MAX_ATR_FROZEN


def wide_hg_code(opened_at, atr):
    """the FRENZY_LONG refusal code a WIDE fill carried, as the engine judges it (ATR before colour) with the era's cap."""
    a = pd.to_numeric(atr, errors="coerce")
    return "FRENZY_ATR_HIGH" if (pd.isna(a) or float(a) > wide_hg_cap_at(opened_at)) else "FRENZY_GREEN_BAR"


def frenzy_fixed_pct(strat, peak, pct, tp=FRENZY_TP_FROZEN):
    """🎯 10-08a (DECISION_LOG 250): a FRENZY sleeve fill's pct under TODAY's fixed exit — a recorded net peak ≥ +tp books +tp (the TP fills
    first, at the line); otherwise the as-traded pct (a −3 stop before +3, a 12 h cap, any other close). Non-FRENZY / unreadable → pct."""
    try:
        if str(strat) in FRENZY3 and tp > 0 and peak is not None and np.isfinite(float(peak)) and float(peak) >= tp:
            return float(tp)
    except (TypeError, ValueError):
        pass
    return pct


def frenzy_bearish_stack_block(strat, ret1d, gap):
    """🐻 10-08a (DECISION_LOG 250): True when today's stack refuses a FRENZY / WIDE / LITE fill for a bearish day — the engine's pure rule
    (services.frenzy.frenzy_bearish_day) on the fill's own stamps; undecidable → False (kept, fail-open like the engine)."""
    from services.frenzy import frenzy_bearish_day
    if str(strat) not in FRENZY3:
        return False
    return frenzy_bearish_day(ret1d, gap) is True


def fade_cap05_scale(desired, vol, notional, capped, ceiling=500_000.0):
    """🐳 Oct-1 (DECISION_LOG 165/167/168): ticket scale for a kept fade that an older liquidity cap (0.1 / 0.2 / 0.3 %) throttled —
    today's size = min(desired, 0.5 % × 24h volume, hard ceiling) / lived notional, on EVERY pair (operator 10-01: "re-price all
    of them at 0.5%"). 1.0 (no-op) when not capped, any input missing / non-positive, or the new ticket is not > 0.1 % bigger
    (fills already at 0.5 %). Same pct on a bigger ticket — OPTIMISTIC (no market-impact haircut; least credible on $2–5M pairs)."""
    try:
        d, v, nv = float(desired), float(vol), float(notional)
    except (TypeError, ValueError):
        return 1.0
    if str(capped).lower() != 'true' or not all(np.isfinite(x) and x > 0 for x in (d, v, nv)):
        return 1.0
    new = min(d, 0.005 * v, ceiling)
    return new / nv if new > nv * 1.001 else 1.0


# 📏 Oct-6 10-06c (DECISION_LOG 233): TODAY's cell sizing, FROZEN with STACK_VERSION (never the live JSON — a later settings change must
# not silently re-price history; a sizing ship = new STACK_VERSION + these values). tests/test_today_size_scale.py pins them against
# trading_config.json. ONE source of truth: this builder's stack_pnl AND scripts/screen_pool.py pnl_current (SCREENED_BASELINE).
UNMATCHED_LONG_INV_FROZEN = 1.5             # 10-05a (DECISION_LOG 206): UNMATCHED long cell 2× → 1.5× (quiet boost 1.5 too)
UNMATCHED_SPRINT_GVR_MIN_FROZEN = 0.74      # Aug-10 crowd-sprint de-mux: global vol ratio > 0.74 ∧ BTC 5m EMA20 slope > 0.07 → 1×
UNMATCHED_SPRINT_B20SLOPE_MIN_FROZEN = 0.07
UNMATCHED_PVR_MAX_FROZEN = 0.90             # Jul-10 crowded-entry de-mux: pair-vol ratio ≥ 0.90 → 1×
UNMATCHED_LONG_LEV_FROZEN = 1.0             # every re-priced cell's leverage multiplier is 1× today (UNMATCHED LONG rule lev_mult 1.0)


_PATH_CF_RE = re.compile(r"ARM040|LATE_ARM|FADE_SL")   # path counterfactuals: their stack_pct is re-derived from stack_pnl


def _fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def today_size_rule(strat, direction, src, cm, gvr=None, b20slope=None, pvr=None, lev=None):
    """(factor, tag): re-prices a MOMENTUM / FLIP fill's $ from its as-traded cell size — invest multiplier `cm` × leverage multiplier
    `lev` (cell_lev_multiplier; e.g. SOL 06-18 W2+W1 short traded 2× at 30× lev) — to TODAY's size (size-only — the P&L % does not
    change). Every re-priced cell runs at leverage multiplier 1× today. Engine order = trading_engine open_position pattern-cell block:
      · FLIP (any cell) → 1×                         (10-06a, DECISION_LOG 220: no flip cell sizes above 1×)
      · NONEXP_CALM3D door → 1×                      (10-06b, 225)
      · momentum SHORT (any cell) → 1×               (C1 2026-06-29, W2+W1 2026-07-30: every SHORT pattern cell is 1× today)
      · UNMATCHED momentum LONG: crowd-sprint → 1×, else pair-vol ≥ 0.90 → 1×, else min(cm, 1.5)   (a missing stamp fails that leg open)
    tag '' = not a sized cell (sleeves, fades, chase, a fill already at ≤ 1× invest and lev; probes 0.5×/0.05×) → factor 1.0."""
    cm, lv = _fnum(cm) or 1.0, _fnum(lev) or 1.0
    if cm <= 1.0 and lv <= 1.0:
        return 1.0, ''
    s, src = str(strat or ''), str(src or '')
    if s.startswith('FLIP') or src.startswith('FLIP'):
        return 1.0 / (cm * lv), 'FLIP_1X'
    if s not in ('', 'nan', 'None', 'MOMENTUM'):
        return 1.0, ''
    if 'CALM3D' in src:
        return 1.0 / (cm * lv), 'CALM3D_1X'
    if str(direction) == 'SHORT':
        return 1.0 / (cm * lv), 'SHORT_1X'
    if str(direction) == 'LONG' and src == 'UNMATCHED':
        g, sl, pv = _fnum(gvr), _fnum(b20slope), _fnum(pvr)
        if g is not None and sl is not None and g > UNMATCHED_SPRINT_GVR_MIN_FROZEN and sl > UNMATCHED_SPRINT_B20SLOPE_MIN_FROZEN:
            return 1.0 / (cm * lv), 'SPRINT_DEMUX'
        if pv is not None and pv >= UNMATCHED_PVR_MAX_FROZEN:
            return 1.0 / (cm * lv), 'PVR_DEMUX'
        return min(cm, UNMATCHED_LONG_INV_FROZEN) / cm * UNMATCHED_LONG_LEV_FROZEN / lv, 'UNMATCHED_INV'
    return 1.0, ''


def today_size_scale(strat, direction, src, cm, gvr=None, b20slope=None, pvr=None, lev=None):
    return today_size_rule(strat, direction, src, cm, gvr, b20slope, pvr, lev)[0]


def fade_laggard_inputs(df):
    """🪤 Sep-29 FADE_LAGGARD inputs per row: (pair 1d −DI, BTC 4h EMA50/200 gap) — stamped columns first, feature factory for the
    SPIKE_FADE rows that lack them (pre-ship fills). Rows the factory cannot price stay NaN → the rule fails open (never blocks)."""
    ndi = pd.to_numeric(df['entry_pair_1d_ndi'], errors='coerce') if 'entry_pair_1d_ndi' in df else pd.Series(np.nan, index=df.index)
    gap = pd.to_numeric(df['entry_btc_4h_ema50_200_gap_pct'], errors='coerce') if 'entry_btc_4h_ema50_200_gap_pct' in df else pd.Series(np.nan, index=df.index)
    # post-ship fills carry the stamps; a post-ship NaN means live FAILED OPEN on that fill (kept it) — never rebuild it here,
    # else builder/ledger would block a fill the live gate admitted (caveman review). Pre-ship fills: factory (closed-bar klines).
    pre_ship = df.opened_at.astype(str).str[:19].str.replace(' ', 'T') < FADE_LAG_SHIP_UTC
    need = (df.entry_strategy.astype(str) == 'SPIKE_FADE') & (ndi.isna() | gap.isna()) & pre_ship
    if need.any():
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
            import entry_feature_factory as EF
            X = EF.features(df.loc[need])
            ndi.loc[need] = ndi.loc[need].fillna(pd.to_numeric(X['PAIR_1d_ndi'], errors='coerce'))
            gap.loc[need] = gap.loc[need].fillna(pd.to_numeric(X['BTC_4h_gap_ema50_200'], errors='coerce'))
        except Exception as e:                                        # noqa: BLE001
            print(f"WARNING: fade laggard inputs could not be rebuilt ({e}) — FADE_LAGGARD fails open on {int(need.sum())} unstamped fade(s)")
    return ndi, gap


def fade_freshbreak_stamp_block(opened_at, rsi_prev, pgap):
    """Builder FADE_FRESHBREAK: stamp proxy for pre-ship fills; post-ship fills defer to the live gate (never blocked)."""
    # opened_at is part of the locked dedup key, so never NaN here (a NaN would read "nan" ≥ cutoff → fail-open)
    if str(opened_at)[:19].replace(' ', 'T') >= FADE_FB_SHIP_UTC:
        return False
    return (pd.notna(rsi_prev) and rsi_prev < FADE_FB_RSI_PREV_MIN
            and pd.notna(pgap) and pgap > FADE_FB_PGAP_MIN)

BTC5M_CACHE = "reports/backtest_cache/k5m_full/BTCUSDT.csv"


def btc_eff72_rebuild(open_ms, T, C):
    """The engine's bull-run monitor eff (864 closed 5m bars, |net| / path, truncated to 3 dp) at a fill time — the SAME formula as
    scripts/ml_watchlist_adjudicate.py / ml_regime_observe_read.py eff72_rebuild (validated vs live stamps, corr 0.999). NaN when the
    history is short or the last closed bar is > 15 min old (a fill past the cache end keeps only its live stamp → fail-open)."""
    if not len(T):
        return np.nan
    i = np.searchsorted(T, open_ms, side="right") - 1
    if i < 0:
        return np.nan
    lc = i - 1 if T[i] + 300_000 > open_ms else i
    if lc < 999 or open_ms - T[lc] > 15 * 60_000:
        return np.nan
    w = C[lc - 863: lc + 1]; d = np.abs(np.diff(w)).sum()
    return int(abs(w[-1] - w[0]) / d * 1000) / 1000.0 if d > 0 else 0.0


def chop_eff72(df):
    """🌀 Oct-4 eff72 per row: live stamp entry_btc_eff72 first, else the rebuild from the BTC 5m cache (deterministic — no network).
    Returns (Series, n_rebuilt)."""
    eff = pd.to_numeric(df['entry_btc_eff72'], errors='coerce') if 'entry_btc_eff72' in df else pd.Series(np.nan, index=df.index)
    need = eff.isna()
    if not need.any():
        return eff, 0
    try:
        k = pd.read_csv(BTC5M_CACHE, usecols=["open_time", "c"]).drop_duplicates("open_time").set_index("open_time").sort_index()
        T, C = k.index.values.astype("int64"), k.c.values
        om = ((pd.to_datetime(df.loc[need, 'opened_at'].astype(str).str[:19].str.replace('T', ' '), errors='coerce')
               - pd.Timestamp(0)).dt.total_seconds() * 1000)
        rb = pd.Series([btc_eff72_rebuild(int(x), T, C) if pd.notna(x) else np.nan for x in om], index=om.index)
        eff.loc[need] = rb
        return eff, int(rb.notna().sum())
    except Exception as e:                                        # noqa: BLE001
        print(f"WARNING: BTC 5m cache unreadable ({e}) — LONG_CHOP_BURST fails open on {int(need.sum())} unstamped row(s)")
        return eff, 0


def chop_burst_pass(df, keep, eff, th, rule, window_s=120.0):
    """🌀👥 Oct-4 LONG_CHOP_BURST over the first-pass keep list (engine order: the last momentum-long gate). Per era, in opened_at
    order: a kept non-probe MOMENTUM LONG is refused when long_chop_burst_block(th, eff72, gap) holds, gap = seconds since the most
    recent OTHER kept, non-probe, non-MANUAL fill of the same era (0…window_s; a refused fill is no longer a neighbour — live never
    opened it). Returns the set of refused row indices."""
    ts = pd.to_datetime(df.opened_at.astype(str).str[:19].str.replace('T', ' '), errors='coerce')
    kept = pd.Series(keep, index=df.index).astype(bool)
    nb_ok = kept & ~df.is_probe.astype(bool) & (df.entry_strategy.astype(str) != 'MANUAL') & ts.notna()
    slv = df.screen_sleeve.astype(str) if 'screen_sleeve' in df else pd.Series('', index=df.index)
    ml = (nb_ok & (df.entry_strategy.astype(str).str.startswith('MOMENTUM') | slv.str.startswith('MOM'))
          & (df.direction.astype(str) == 'LONG'))
    blocked = set()
    for era in df.era.unique():
        idx = df.index[(df.era == era) & nb_ok]
        order = sorted(idx, key=lambda j: (ts[j], j))
        for j in order:
            if not ml[j]:
                continue
            gaps = [(ts[j] - ts[m]).total_seconds() for m in order
                    if m != j and m not in blocked and 0 <= (ts[j] - ts[m]).total_seconds() <= window_s]
            if gaps and rule(th, eff.get(j), min(gaps)):
                blocked.add(j)
    return blocked


# Era registry (Sep-11: B3/B4/B5 were previously stacked by a one-off — the builder only knew
# BASE/B1/B2, so the committed MASTER_POOL had B3/B4 rows with NULL stack columns and no B5).
# Fixed eras are listed explicitly; every later batch is auto-discovered from the archive naming
# convention reports/BASELINE<n>_*.csv (n ≥ 6 → era B<n>). Archive each batch under that name
# at its pre-reset export and the pool picks it up on the next regen.
# (era, path, min_opened_at, lenient_status) — lenient_status=True keeps NaN-status rows as CLOSED
# (the B1 anchor file's legacy quirk); every other era is strict `status == "CLOSED"`.
BASE_ERA_END = "2026-07-11"   # B1/ANCHOR starts here; see the BASE cap in load()

FIXED_ERAS = [
    ('BASE', "reports/SCREENED_BASELINE.csv", None, False),
    ('B1',   "reports/BASELINE2_ANCHOR_batch0711-31_current_stack.csv", None, True),
    ('B2',   "reports/BASELINE2_batch0731-0810_orders_prereset.csv", "2026-07-31", False),
    ('B3',   "reports/BASELINE3_batch0811-0824_orders_prereset.csv", None, False),
    ('B4',   "reports/BASELINE4_batch0824-0826_first_live_orders_prereset.csv", None, False),
    ('B5',   "reports/BASELINE5_batch0826-0903_orders.csv", None, False),
]

def discover_eras():
    """FIXED_ERAS + reports/BASELINE<n>_*.csv for n ≥ 6 (one file per n; ambiguity is fatal)."""
    import glob, re
    eras = list(FIXED_ERAS)
    found = {}
    for f in sorted(glob.glob("reports/BASELINE[0-9]*_*.csv")):
        m = re.match(r"reports/BASELINE(\d+)_.*\.csv$", f)
        if not m or int(m.group(1)) < 6 or '_split_report' in f or 'ANCHOR' in f:
            continue
        n = int(m.group(1))
        if n in found:
            raise SystemExit(f"FATAL: two archive files for BASELINE{n}: {found[n]} and {f} — keep one")
        found[n] = f
    for n in sorted(found):
        eras.append((f'B{n}', found[n], None, False))
    return eras

def load():
    frames = []
    for era, path, min_open, lenient in discover_eras():
        d = pd.read_csv(path, low_memory=False)
        if era == 'BASE':
            # 🔒 BASE ERA CAP (2026-09-21 fix): SCREENED_BASELINE.csv is the SCREENED COMBINED pool,
            # and every new batch gets appended to COMBINED at its review (v17 appended B9's three
            # momentum longs on Sep-19). Those rows then appear BOTH here and in their own
            # BASELINE<n> archive → the locked dedup key collides and the build dies. BASE is an
            # ERA (Jun-17 → Jul-10, before B1/ANCHOR starts Jul-11), so cap it at that boundary and
            # let each later era come from its own archive. Restores BASE to its historical 83 rows.
            d = d[d.opened_at < BASE_ERA_END]
        else:  # BASE is already the screened CLOSED set
            st = d.status.fillna("CLOSED") if lenient else d.status
            d = d[st == "CLOSED"]
            if min_open:
                d = d[d.opened_at >= min_open]
        d = d.copy(); d['era'] = era
        frames.append(d)
    n_eras = len(frames)
    # Schema: the gate-load-bearing `required` columns must be present in EVERY era (strict
    # intersection, fails loudly on drift); everything else is UNIONED (NaN where an era predates
    # the column) so later-era shadow/BE-lock columns survive the regen (review Sep-11: a strict
    # intersection silently dropped 23 B3/B4 columns).
    # ONE exception on record: the stack's MOM fallback reads `screen_sleeve` (union-only, BASE-populated) —
    # verified Sep-11 it never disagrees with the required `entry_strategy`, so the union is stack-neutral.
    inter = set.intersection(*[set(d.columns) for d in frames])
    cols = set.union(*[set(d.columns) for d in frames])
    # gate-load-bearing columns must survive the N-way intersection — a schema
    # drift in ONE era file would otherwise silently shrink the analysis surface
    required = ['opened_at', 'pair', 'direction', 'pnl', 'pnl_percentage', 'peak_pnl',
                'entry_strategy', 'entry_rsi_prev', 'entry_pos_di', 'entry_adx',
                'entry_btc_rsi', 'entry_btc_dist_from_ema13_pct', G,
                'entry_pair_volume_24h_usd', 'entry_atr_pct', 'entry_ema5_stretch',
                'entry_btc_regime', 'entry_global_volume_ratio', 'entry_btc_ema20_slope',
                # 🪃 Sep-25 FLIP_FAN_WEAK_BOUNCE inputs (review fix: a missing era column must fail loudly, not fail open)
                'entry_ema20_slope', 'entry_pair_ema20_ema50_gap_pct',
                # 🧊 Sep-25 MOM_SHORT_C1_REGIME inputs (a missing era column must fail loudly)
                'entry_pattern_c1_match',
                # FAKE_BULL_GUARD gate columns (2026-08-14a) — schema drift must fail loudly
                'confidence', 'entry_bull_pct', 'entry_btc_trend_gap_pct',
                'cell_multiplier', 'cell_multiplier_source', 'pattern_cell_source', 'status',
                # fade-SL/lock CF load-bearing (review fix: schema drift here must fail loudly,
                # else every fade loser silently re-prices to the full stop)
                'close_reason', 'entry_price', 'post_exit_running_high', 'post_exit_final_pnl']
    missing = [c for c in required if c not in inter]
    if missing:
        raise SystemExit(f"FATAL: gate columns missing from the {n_eras}-way intersection: {missing}")
    cols.discard('id')  # locked rule: NEVER use `id` (resets on paper reset) — keep it out of the pool
    cols = sorted(cols)  # deterministic column order (review fix: set order reshuffled every regen)
    df = pd.concat([d.reindex(columns=cols) for d in frames], ignore_index=True, sort=False)
    dup = df.duplicated(subset=['opened_at', 'pair', 'direction'])
    if dup.any():
        raise SystemExit(f"FATAL: {dup.sum()} duplicate rows on the locked dedup key "
                         f"(opened_at, pair, direction) — era boundaries overlap")
    return df

def main():
    df = load()
    src = df.cell_multiplier_source.fillna('') + df.pattern_cell_source.fillna('')
    df['is_probe'] = src.str.upper().str.contains('PROBE')
    # Sep-15 gate 60: BEARRUN_SHORT probe fills (lev mult < 1 = 1× effective) are PROBE-EXEMPT like every other 1× probe
    # (full-size-only rule); armed sleeve fills keep their own entry_strategy and must never be read as MOM-short.
    if 'cell_lev_multiplier' in df.columns:
        _lev = pd.to_numeric(df.cell_lev_multiplier, errors='coerce')
        df['is_probe'] = df['is_probe'] | ((df.entry_strategy.astype(str) == 'BEARRUN_SHORT') & (_lev < 1.0))
    df['is_door'] = src.str.contains('CALM3D')
    vol = df.entry_pair_volume_24h_usd
    # 🔥 Sep-18 long heat block parity — the engine's own pure rule with the SHIPPED thresholds pinned here
    # (builder convention: the stack version freezes its numbers). 30d reading: stamped column when the
    # fill has it, else the hourly series from scripts/build_btc_off30d.py; missing → fail-open like live.
    import os, sys
    from types import SimpleNamespace
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from services.trading_engine import long_heat_eval, long_megacap_block, long_chop_burst_block, fade_late_arm_cf, flip_fan_weak_bounce, mom_short_c1_regime_block, calm3d_btc_atr_floor_block
    from services.indicators import rsi_mom_loadx_block, fade_laggard_block
    # ⏱ Sep-24 frozen fade exit constants for the late-arm CF (live values at ship; builder pins a STACK_VERSION)
    _FADE_TH = SimpleNamespace(spike_fade_late_arm_after_min=15.0, spike_fade_late_arm_peak=0.30,
                               runner_trail_short_arm_peak=0.40, runner_trail_short_atr_mult=0.5,
                               runner_trail_short_giveback_frac=0.35, runner_trail_short_be_ratchet_enabled=False,
                               runner_trail_short_be_lock_pct=0.10)
    # frozen stack constants (the builder pins a STACK_VERSION, it does not read hot config) — Sep-23: mega-cap rank ≤10
    # Oct-5 (DECISION_LOG 208): REVERTED to the 3-leg rule 0.07 / 64 / 80 (values below). History — Sep-25 (DECISION_LOG 116): heat block re-scoped to breadth only — BTC legs 0 (off), bull 80 → 85, washed-out exemption kept
    _HEAT_TH = SimpleNamespace(long_heat_block_enabled=True, long_heat_btc_slope_min=0.07, long_heat_btc_rsi_prev_min=64.0,
                               long_heat_bull_pct_min=80.0, long_heat_exempt_off30d_max=-10.0,
                               long_megacap_rank_max=10)
    # 🪃 Sep-25 frozen fan-flip weak-bounce constants (live values at ship; DECISION_LOG 113)
    _FLIP_TH = SimpleNamespace(flip_fan_weak_bounce_enabled=True, flip_fan_weak_bounce_gap_max=0.0,
                               flip_fan_weak_bounce_slope_max=0.15)
    # 🧊 Sep-25 frozen C1 momentum-short regime block (live value at ship; DECISION_LOG 114)
    _C1_TH = SimpleNamespace(momentum_short_c1_block_regimes='STRONG_BEAR')
    # 🌫 Sep-27 frozen CALM3D BTC-ATR floor (live value at ship; DECISION_LOG 118)
    _CALM3D_TH = SimpleNamespace(nonexp_calm3d_btc_atr_min=0.08)
    _LOADX_TH = SimpleNamespace(long_rsi_momentum_adx_max=21.0)   # 🧭 Sep-29 freeze (DECISION_LOG 126) — ledger warns if live drifts
    # 🪤 Sep-29 fade laggard gate (DECISION_LOG 128): frozen thresholds; inputs = the order stamps when present (post-ship fills),
    # else rebuilt from the closed-bar 5m kline cache by the research feature factory (the exact columns the rule was found on).
    _LAG_TH = SimpleNamespace(spike_fade_lag_ndi_min=16.1, spike_fade_lag_btc_gap_min=0.0)
    _lag_ndi, _lag_gap = fade_laggard_inputs(df)
    # 🌀👥 Oct-4 frozen chop ∧ burst block (live values at ship; DECISION_LOG 201) — ledger/test pin these to trading_config.json
    _CB_TH = SimpleNamespace(long_chop_burst_block_enabled=True, long_chop_burst_eff72_max=0.007, long_chop_burst_window_s=120.0)
    _off30 = {}
    if os.path.exists("reports/btc_off30d_hourly.csv"):
        _o = pd.read_csv("reports/btc_off30d_hourly.csv")
        _off30 = dict(zip(_o.hour_utc, _o.btc_off30d_high_pct))
    else:
        print("WARNING: reports/btc_off30d_hourly.csv missing — LONG_HEAT_BLOCK fails open on unstamped rows")
    def _row_off30d(r):
        v = r.get('entry_btc_off30d_high_pct')
        if v is not None and pd.notna(v):
            return float(v)
        return _off30.get(str(r.opened_at)[:13].replace('T', ' ') + ':00')
    keep, reason, spnl, scale = [], [], [], []
    # door same-pair <=90min re-fire detection (cooldown), computed per era
    df['_ts'] = pd.to_datetime(df.opened_at.str[:19], errors='coerce')
    cooldown_idx = set()
    for era in df.era.unique():
        last = {}
        d = df[(df.era == era) & df.is_door].sort_values('_ts')
        for i, r in d.iterrows():
            if r.pair in last and (r._ts - last[r.pair]).total_seconds() <= 90 * 60:
                cooldown_idx.add(i)
            last[r.pair] = r._ts
    for i, r in df.iterrows():
        p = r.pnl; strat = str(r.entry_strategy); slv = str(r.get('screen_sleeve') or '')
        v = vol[i] if pd.notna(vol[i]) else None
        k, why = True, ''
        if strat == 'MANUAL':
            k, why = False, 'MANUAL_EXEMPT'   # 🖐 Sep-29: operator-opened — never a stack trade, never in the ledger
        elif r.is_probe:
            why = 'PROBE_EXEMPT'
        if not r.is_probe and strat != 'MANUAL':
            if strat == 'BEARRUN_SHORT':
                why = 'BEARRUN_SLEEVE'   # kept, but its OWN sleeve — MOM-short reads must filter entry_strategy == 'MOMENTUM'
            elif strat == 'SURGE_SHORT':
                k, why = False, 'SURGE_SHORT_OFF'   # 🐻⚡ Oct-4 (DECISION_LOG 200): surge_short_enabled = false → today's stack never opens it
            elif strat == 'SURGE_LONG':
                why = 'SURGE_SLEEVE'     # ⚡ Sep-30: kept as its OWN sleeve (BTC spike trigger) — never a momentum row; Oct-4: probe size (lev 0.05)
            elif strat in ('FRENZY_LONG', 'FRENZY_WIDE', 'FRENZY_LITE'):
                why = 'FRENZY_SLEEVE'    # 🔥 Oct-3: OBSERVATION — the volume-frenzy sleeve's own row (40-fill review), never a momentum row · 🪶 Oct-7 LITE (243): same (no LITE fill in the master yet → nothing re-prices, STACK_VERSION stays)
            elif strat == 'SPIKE_FADE':
                if v is not None and v >= 20e6: k, why = False, 'FADE_MAXVOL'   # Sep-14 ceiling — engine order: first fade gate
                elif r.entry_btc_rsi > 50: k, why = False, 'FADE_BRSI50'  # engine uses strict > (50.0 passes); Sep-24 45→50 (DECISION_LOG 112)
                elif pd.notna(r.entry_btc_dist_from_ema13_pct) and r.entry_btc_dist_from_ema13_pct > 0: k, why = False, 'FADE_BD13'
                elif v is not None and v < 2e6: k, why = False, 'FLOOR_2M'
                elif fade_freshbreak_stamp_block(r.opened_at, r.entry_rsi_prev, r[G]): k, why = False, 'FADE_FRESHBREAK'
                # 🪤 Sep-29: laggard squeeze — pair daily −DI > 16.1 while BTC 4h EMA50 > EMA200 (indicators.fade_laggard_block; engine
                # tests it LAST among the fade gates; fail-open on a missing reading). DECISION_LOG 128.
                elif fade_laggard_block(_LAG_TH, _lag_ndi.get(i), _lag_gap.get(i)): k, why = False, 'FADE_LAGGARD'
            elif strat == 'SPIKE_CHASE':
                sa = (r.entry_ema5_stretch / r.entry_atr_pct) if (pd.notna(r.entry_atr_pct) and r.entry_atr_pct) else None
                if v is not None and v < 2e6: k, why = False, 'FLOOR_2M'
                elif sa is not None and sa > 1.5: k, why = False, 'CHASE_STRETCH15'
            elif strat == 'SPIKE_BOUNCE':
                pg = r[G] if pd.notna(r[G]) else None
                if v is not None and v < 2e6: k, why = False, 'FLOOR_2M'
                elif pg is not None and not (-1.0 < pg <= -0.125): k, why = False, 'BOUNCE_PGAP'
                elif pd.notna(r.entry_btc_rsi) and r.entry_btc_rsi < 50: k, why = False, 'BOUNCE_BRSI'
                elif any(x in str(r.entry_btc_regime) for x in ('STRONG_BEAR', 'HEALTHY_BEAR')): k, why = False, 'BOUNCE_REGIME'
            elif strat.startswith('FLIP:FAN') and str(r.direction) == 'SHORT':
                # Aug-23 FAN-gate bearish-BTC filter (flip_fan_btc_ema13_max=-0.08): a FAN_RATIO_GATE short is
                # refused while BTC sits above -0.08% vs its 5m EMA13. Engine: strict >, fail-open on a missing
                # distance. Added Sep-16 (builder never carried the flip gates; GIGGLE/ONDO/TRB read as kept).
                bd13 = r.entry_btc_dist_from_ema13_pct
                if pd.notna(bd13) and float(bd13) > -0.08: k, why = False, 'FLIP_FAN_BTC_EMA13'
                # Sep-25: 🪃 weak-bounce block — pair below its trend (EMA13−EMA50 gap < 0) AND gentle EMA20 slope (< 0.15).
                # Rule = engine flip_fan_weak_bounce (fail-open on unstamped fields). Live parity, DECISION_LOG 113.
                elif flip_fan_weak_bounce(_FLIP_TH, r.get('entry_pair_ema20_ema50_gap_pct'), r.get('entry_ema20_slope')):
                    k, why = False, 'FLIP_FAN_WEAK_BOUNCE'
            elif strat.startswith('MOMENTUM') or slv.startswith('MOM'):
                if i in cooldown_idx: k, why = False, 'CALM3D_REENTRY'
                elif r.is_door and pd.notna(r.entry_pos_di) and r.entry_pos_di < 28: k, why = False, 'CALM3D_DMI_DI'
                elif r.is_door and pd.notna(r.entry_adx) and r.entry_adx < 21: k, why = False, 'CALM3D_DMI_ADX'
                # Sep-27: 🌫 CALM3D BTC-ATR floor — a door long on a DEAD tape (BTC ATR < 0.08) is refused (engine
                # calm3d_btc_atr_floor_block; fail-open on a missing stamp). Live parity, DECISION_LOG 118.
                elif r.is_door and calm3d_btc_atr_floor_block(_CALM3D_TH, r.get('entry_btc_atr_pct')): k, why = False, 'CALM3D_BTC_ATR_MIN'
                elif (str(r.direction) == 'LONG'
                      and long_heat_eval(_HEAT_TH, r.entry_btc_ema20_slope, r.get('entry_btc_rsi_prev'), r.entry_bull_pct, _row_off30d(r))[1]):
                    k, why = False, 'LONG_HEAT_BLOCK'
                # Sep-23: 🏦 mega-cap exclusion — momentum longs refused at RAW eligible-universe rank ≤ long_megacap_rank_max
                # (engine long_megacap_block; fail-open on an unstamped rank). Live parity, DECISION_LOG 110.
                elif str(r.direction) == 'LONG' and long_megacap_block(_HEAT_TH, r.get('entry_pair_rank')):
                    k, why = False, 'LONG_MEGACAP_BLOCK'
                # Sep-29: 🧭 low-ADX RSI-momentum leg — momentum longs refused when RSI(12) < RSI two candles ago AND pair ADX <
                # long_rsi_momentum_adx_max (indicators.rsi_mom_loadx_block; stamp entry_rsi_prev IS rsi_prev2; fail-open). DECISION_LOG 126.
                elif str(r.direction) == 'LONG' and rsi_mom_loadx_block(_LOADX_TH, r.get('entry_rsi'), r.get('entry_rsi_prev'), r.get('entry_adx')):
                    k, why = False, 'LONG_RSI_MOM_LOADX'
                # Sep-25: 🧊 C1 momentum shorts refused in STRONG_BEAR (engine mom_short_c1_regime_block, checked BEFORE the
                # pair-vol ceiling like the engine; fail-open on missing flag/regime). Live parity, DECISION_LOG 114.
                elif (str(r.direction) == 'SHORT'
                      and mom_short_c1_regime_block(_C1_TH, r.get('entry_pattern_c1_match'), r.get('entry_btc_regime'))):
                    k, why = False, 'MOM_SHORT_C1_REGIME'
                # Sep-18: momentum_short_pair_vol_max 1.0 → 0.86 (live parity; engine: strict >=, fail-open on missing PVR).
                elif (str(r.direction) == 'SHORT' and pd.notna(r.entry_pair_volume_ratio)
                      and float(r.entry_pair_volume_ratio) >= 0.86): k, why = False, 'MOM_SHORT_PAIRVOL'
                # 🛡 FAKE_BULL_GUARD gate REMOVED 2026-08-16 (guard reverted by its own locked
                # gate 47: forward 12-block replay 6W/6L·net+4.65pp — see DECISION_LOG). The
                # stack no longer blocks this cohort; columns stay load-bearing for the record.
        # CF P&L for kept trades
        sp = p
        # fade SL -1.5 (41e): SL-stopped fade losers re-priced — survive if worst-ever
        # adverse < 1.5% (post-exit running high vs entry, short side), outcome = held
        # trajectory endpoint; else full stop at -1.5. Non-SL exits (EMA13/SPIKE_LOCK) untouched.
        if (k and not r.is_probe and strat == 'SPIKE_FADE' and p < 0
                and str(r.get('close_reason') or '').startswith(('STOP_LOSS', 'SPIKE_LOCK'))
                and pd.notna(r.pnl_percentage) and r.pnl_percentage != 0):
            dpp = abs(p / r.pnl_percentage)
            worst = None
            if pd.notna(r.get('post_exit_running_high')) and pd.notna(r.entry_price) and r.entry_price:
                worst = (r.post_exit_running_high / r.entry_price - 1) * 100  # adverse % for the short
            if worst is not None and worst < 1.5 and pd.notna(r.get('post_exit_final_pnl')):
                sp = r.post_exit_final_pnl * dpp
            else:
                sp = -1.5 * dpp
            why = why or 'CF_FADE_SL15'
        # ⏱ Sep-24 FADE LATE-ARM (DECISION_LOG 111): stamps-only CF — a never-armed fade still open past 15 min whose
        # (max) peak came AFTER minute 15 inside [0.30, 0.40) is re-priced to the late trail floor. It supersedes the
        # SL15 reprice (the late trail fires at the post-15 peak, before any later stop). Exposed late winners (armed
        # normally after 15) stay as lived — their post-15 path is not stamped, so this CF is OPTIMISTIC by design.
        if k and not r.is_probe and strat == 'SPIKE_FADE' and pd.notna(r.pnl_percentage) and r.pnl_percentage != 0:
            _o = pd.to_datetime(r.opened_at, errors='coerce')
            _pmin = (pd.to_datetime(r.get('peak_reached_at'), errors='coerce') - _o).total_seconds() / 60 if pd.notna(r.get('peak_reached_at')) else None
            _dmin = (pd.to_datetime(r.closed_at, errors='coerce') - _o).total_seconds() / 60 if pd.notna(r.get('closed_at')) else None
            _cf = fade_late_arm_cf(_FADE_TH, r.pnl_percentage, r.peak_pnl, _pmin, _dmin, r.entry_atr_pct)
            if _cf is not None:
                sp = _cf * abs(p / r.pnl_percentage); why = 'CF_FADE_LATE_ARM'
        # 🐳 Oct-1 FADE CAP 0.5 % ON ALL PAIRS (DECISION_LOG 165/167/168): every kept fade an older cap throttled is re-priced at
        # today's ticket = min(desired, 0.5 % × 24h volume, ceiling) — same pct, bigger ticket, applied AFTER the SL15 / late-arm
        # CFs (losers scale too). The scale is stored in stack_ticket_scale: a pct derived from stack_pnl MUST divide by it
        # (stack_pnl / stack_ticket_scale / notional_value). Untagged rows get the CF_FADE_CAP05 tag; CF-tagged rows keep theirs.
        # current_stack_ledger --batch rows do not mirror this (a fresh batch fade already sits at the live 0.5 % ticket).
        tsc = 1.0
        if k and not r.is_probe and strat == 'SPIKE_FADE':
            tsc = fade_cap05_scale(r.get('entry_desired_notional'), r.get('entry_pair_volume_24h_usd'), r.get('notional_value'), r.get('liquidity_capped'))
            if tsc != 1.0:
                sp = sp * tsc; why = why or 'CF_FADE_CAP05'
        scale.append(tsc)
        if k and not r.is_probe and (strat.startswith('MOMENTUM') or slv.startswith('MOM')) and str(r.direction) == 'LONG':
            pk, atr = r.peak_pnl, (r.entry_atr_pct if pd.notna(r.entry_atr_pct) else 99)
            if pd.notna(pk) and 0.40 <= pk < 0.45 and pd.notna(r.pnl_percentage) and r.pnl_percentage < max(pk - atr, 0.10) and r.pnl_percentage != 0:
                sp = max(pk - atr, 0.10) / 100 * abs(p / (r.pnl_percentage / 100)); why = why or 'CF_ARM040'
            # (the crowd-sprint de-mux lives in today_size_rule since 10-06c — applied in the final re-price, after stack_pct)
        keep.append(k); reason.append(why); spnl.append(sp if k else 0.0)
    # 🌀👥 Oct-4 LONG_CHOP_BURST (operator ARMED override, DECISION_LOG 201): second pass — needs every row's first-pass keep (the
    # neighbours are fills of ANY sleeve). Rule = engine long_chop_burst_block with the frozen 0.007 / 120 s.
    # 🟢 10-06d (DECISION_LOG 231/234): today FRENZY_WIDE opens only hold-green setups — replay the engine's own rule on the stamps, BEFORE
    # the chop-burst pass (review: a WIDE fill refused today must not count as a burst neighbour). Streak: stamp entry_frenzy_above_streak
    # (B18+), else reports/WIDE_STREAK_REBUILD.csv (frenzy_walk on 1499 public 5m bars), else refused = fail-closed like the engine.
    from services.frenzy import frenzy_wide_hold_green_block
    _hg_th = SimpleNamespace(frenzy_wide_hold_green_streak=WIDE_HG_STREAK_FROZEN)
    try:
        _w = pd.read_csv("reports/WIDE_STREAK_REBUILD.csv")
        _wsr = {(str(a)[:19].replace(" ", "T"), p, d_): (s_, c_) for a, p, d_, s_, c_ in zip(_w.opened_at, _w.pair, _w.direction, _w.above_streak, _w.code)}
    except FileNotFoundError:
        _wsr = {}
    _pos0 = {ix: n for n, ix in enumerate(df.index)}
    _hg_src = {"stamp": 0, "rebuild": 0, "none (fail-closed)": 0}
    for i, r in df[df.entry_strategy.astype(str) == "FRENZY_WIDE"].iterrows():
        n = _pos0[i]
        if not keep[n]:
            continue
        _code = wide_hg_code(r.opened_at, r.get("entry_atr_pct"))   # the engine judges ATR before colour · 10-08a: the era's cap (2.5 pre-raise)
        _st = pd.to_numeric(r.get("entry_frenzy_above_streak"), errors="coerce")
        if pd.notna(_st):
            _hg_src["stamp"] += 1
        else:
            _hit = _wsr.get((str(r.opened_at)[:19].replace(" ", "T"), r.pair, str(r.direction)))
            if _hit is not None:
                _st = _hit[0]; _hg_src["rebuild"] += 1
                assert _hit[1] == _code, f"WIDE_STREAK_REBUILD code {_hit[1]} != stamp-derived {_code} for {r.pair} {r.opened_at}"
            else:
                _st = None; _hg_src["none (fail-closed)"] += 1
        _ret = pd.to_numeric(r.get("entry_frenzy_bar_ret_pct"), errors="coerce")
        _blk = frenzy_wide_hold_green_block({"above_streak": (None if _st is None or pd.isna(_st) else int(_st)),
                                             "bar_ret_pct": (None if pd.isna(_ret) else float(_ret))}, _code, _hg_th)
        if _blk:
            keep[n] = False; reason[n] = _blk; spnl[n] = 0.0
    print(f"WIDE hold-green replay: streak from {_hg_src}")
    # 🐻 10-08a (DECISION_LOG 250): FRENZY / WIDE / LITE refused on a bearish day — the engine's own pure rule on the two stamps the fill
    # carries (the very values the live gate reads); undecidable (a stamp missing / NaN with the other not ≥ 0) = kept (fail-open).
    _bd = {"blocked": 0, "kept": 0}
    for i, r in df[df.entry_strategy.astype(str).isin(list(FRENZY3))].iterrows():
        n = _pos0[i]
        if not keep[n]:
            continue
        if frenzy_bearish_stack_block(r.entry_strategy, pd.to_numeric(r.get("entry_btc_1d_ret_pct"), errors="coerce"),
                                      pd.to_numeric(r.get("entry_btc_trend_gap_pct"), errors="coerce")):
            keep[n] = False; reason[n] = "FRENZY_BEARISH_DAY"; spnl[n] = 0.0; _bd["blocked"] += 1
        else:
            _bd["kept"] += 1
    print(f"FRENZY_BEARISH_DAY: {_bd}")
    _cb_eff, _cb_nrb = chop_eff72(df)
    _cb_blk = chop_burst_pass(df, keep, _cb_eff, _CB_TH, long_chop_burst_block, _CB_TH.long_chop_burst_window_s)
    _pos = {ix: n for n, ix in enumerate(df.index)}
    for j in _cb_blk:
        n = _pos[j]; keep[n] = False; reason[n] = 'LONG_CHOP_BURST'; spnl[n] = 0.0
    print(f"LONG_CHOP_BURST: eff72 = live stamp on {int(pd.to_numeric(df.get('entry_btc_eff72'), errors='coerce').notna().sum())} rows, "
          f"rebuilt on {_cb_nrb}; refused {len(_cb_blk)}: " + ", ".join(f"{df.at[j, 'pair']}@{str(df.at[j, 'opened_at'])[:19]}({df.at[j, 'era']})" for j in sorted(_cb_blk)))
    df['stack_keep'] = keep; df['stack_block_reason'] = reason
    _spnl_raw = dict(zip(df.index, spnl))   # 10-06c review: the today's-size factor applies to the UNROUNDED P&L (no double rounding)
    df['stack_pnl'] = np.round(spnl, 2); df['stack_ticket_scale'] = np.round(scale, 6); df['stack_version'] = STACK_VERSION
    # 📐 stack_pct = the P&L % under today's rules. Size-only re-prices (ticket CAP05, sprint de-mux, UNMATCHED 1.5×, flip cells → 1×,
    # sleeve sizing) change stack_pnl but NOT the pct, so a pct must never be read off a stack_pnl / pnl ratio. Path CFs (ARM040 /
    # LATE_ARM / FADE_SL) re-price the pct (same convention as validate_against_master M1); the FRENZY +TP re-price is set below.
    _cf = df.stack_keep & df.stack_block_reason.fillna("").astype(str).str.contains(_PATH_CF_RE.pattern)
    df['stack_pct'] = np.where(_cf, df.stack_pnl / df.stack_ticket_scale.fillna(1) / pd.to_numeric(df.notional_value, errors="coerce") * 100,
                               pd.to_numeric(df.pnl_percentage, errors="coerce"))
    df.loc[~df.stack_keep.astype(bool), 'stack_pct'] = np.nan   # blocked trades have no today's-rules pct (stack_pnl is 0 there)
    df = df.drop(columns=['_ts'])
    # 🔥 Oct-4 (DECISION_LOG 193/197/199/200): sleeve fills re-priced at TODAY's size (pct is size-free; $ = pct × today's notional).
    # Margin today = as-traded margin ÷ its cell multiplier × today's invest mult; leverage today = max(1, round(20 × today's lev mult)) —
    # FRENZY_LONG 1 × 0.32 (6×), or 0.5 (10×) when its ADX-rising ∧ +DI-above stamps say so; FRENZY_WIDE 1 × 0.2 (4×);
    # BEARRUN_SHORT 1 × 0.05 (1×); SURGE_LONG 1 × 1.0 (20×) since Oct-4 option B (DECISION_LOG 202 — the master holds no SURGE_LONG fill, so
    # nothing re-prices and STACK_VERSION stays). 10-06d priced the lock (retired 10-08a); 10-08a: every FRENZY fill with a recorded peak ≥ +3 books the fixed +3.
    # FROZEN with STACK_VERSION (review: never read the live JSON — a later settings change must not silently re-price history; a sizing
    # change = a new STACK_VERSION with these values updated; tests/test_long_chop_burst.py pins them against trading_config.json)
    _th = dict(frenzy_long_invest_mult=1.0, frenzy_long_lev_mult=0.32, frenzy_long_lev_mult_strong=0.5, frenzy_wide_invest_mult=1.0,
               frenzy_wide_lev_mult=0.2, frenzy_lite_invest_mult=1.0, frenzy_lite_lev_mult=0.32, surge_long_invest_mult=1.0, surge_long_lev_mult=1.0, bearrun_invest_mult=1.0, bearrun_lev_mult=0.05,
               frenzy_tp_pct=FRENZY_TP_FROZEN, frenzy_lock_arm_pct=0.0, frenzy_lock_floor_pct=2.0, frenzy_lock_trail_pct=2.0,   # 10-08a: fixed +3 back, lock off (DECISION_LOG 250)
               frenzy_wide_hold_green_streak=WIDE_HG_STREAK_FROZEN, frenzy_max_atr_pct=WIDE_HG_MAX_ATR_FROZEN)        # WIDE hold-green (231) · ATR 3.0 (250)
    SLEEVE_SIZE_FROZEN = _th
    def _today(strat, r):
        if strat == "FRENZY_LONG":
            strong = (pd.notna(r.get("entry_frenzy_adx_delta")) and pd.notna(r.get("entry_frenzy_di_spread"))
                      and float(r["entry_frenzy_adx_delta"]) > 0 and float(r["entry_frenzy_di_spread"]) > 0 and float(_th.get("frenzy_long_lev_mult_strong", 0) or 0) > 0)
            return float(_th.get("frenzy_long_invest_mult", 1.0)), float(_th.get("frenzy_long_lev_mult_strong") if strong else _th.get("frenzy_long_lev_mult", 1.0))
        if strat == "FRENZY_WIDE":
            return float(_th.get("frenzy_wide_invest_mult", 1.0)), float(_th.get("frenzy_wide_lev_mult", 1.0))
        if strat == "FRENZY_LITE":   # 🪶 Oct-7 (243): 1 × 0.2 (4×) → 0.32 (247, operator override; the master holds no LITE fill yet), never the strong bump
            return float(_th.get("frenzy_lite_invest_mult", 1.0)), float(_th.get("frenzy_lite_lev_mult", 1.0))
        if strat == "SURGE_LONG":
            return float(_th.get("surge_long_invest_mult", 1.0)), float(_th.get("surge_long_lev_mult", 1.0))
        if strat == "BEARRUN_SHORT":
            return float(_th.get("bearrun_invest_mult", 1.0)), float(_th.get("bearrun_lev_mult", 1.0))
        return None
    _tp = float(_th["frenzy_tp_pct"])
    for i, r in df.iterrows():
        strat = str(r.entry_strategy)
        if bool(r.stack_keep) and not bool(r.is_probe) and pd.notna(r.get("stack_pnl")):
            # 📏 10-06c: TODAY's cell size (CALM3D / FLIP / momentum SHORT → 1×, UNMATCHED long sprint / PVR → 1× else 1.5×; lev 1×) — one
            # rule, shared with screen_pool.pnl_current. Size-only: stack_pct (computed above) is not touched.
            _f, _tag = today_size_rule(strat, r.direction, r.get("cell_multiplier_source"), r.get("cell_multiplier"),
                                       r.get("entry_global_volume_ratio"), r.get("entry_btc_ema20_slope"), r.get("entry_pair_volume_ratio"),
                                       r.get("cell_lev_multiplier"))
            if _tag:
                _rsn = r.get("stack_block_reason")
                _rsn = "" if pd.isna(_rsn) else str(_rsn).strip()
                if _f != 1.0:
                    df.at[i, "stack_pnl"] = round(float(_spnl_raw[i]) * _f, 2)
                    if _PATH_CF_RE.search(_rsn):
                        # a path-CF row's pct is read back as stack_pnl / stack_ticket_scale / notional (validate_against_master M1 and
                        # ~12 analysis scripts) — fold the size factor into the ticket scale so that formula stays exact (0 rows at 10-06c)
                        df.at[i, "stack_ticket_scale"] = round(float(r.stack_ticket_scale or 1.0) * _f, 6)
                if _tag in ("SPRINT_DEMUX", "PVR_DEMUX") and not _rsn:
                    df.at[i, "stack_block_reason"] = "CF_" + _tag   # label kept from the old first-pass de-mux (CF_SPRINT_DEMUX) + its PVR twin
                continue
        t = _today(strat, r) if bool(r.stack_keep) else None
        if t is None or pd.isna(r.get("investment")) or pd.isna(r.get("pnl_percentage")):
            continue
        cm = float(r.get("cell_multiplier") or 1.0) or 1.0
        lev = max(1, int(round(20 * t[1])))
        pct = float(r.pnl_percentage)
        _fp = frenzy_fixed_pct(strat, pd.to_numeric(r.get("peak_pnl"), errors="coerce"), pct, _tp)
        if _fp != pct:
            # 10-08a (DECISION_LOG 250): TODAY's fixed take profit — a recorded net peak ≥ +3 books +3 (frenzy_exit_for closes at once on a
            # peak ≥ tp); every era (the lock / trail fills that rode past +3 included). Peak < +3 → the as-traded pct (−3 stop / cap / other).
            pct = _fp
            df.at[i, "stack_pct"] = pct
        df.at[i, "stack_pnl"] = round(pct / 100.0 * float(r.investment) / cm * t[0] * lev, 2)
    out = "reports/MASTER_POOL_stacked.csv"
    df.to_csv(out, index=False)
    print(f"MASTER_POOL_stacked.csv written — {len(df)} rows, stack v{STACK_VERSION}\n")
    fs = df[~df.is_probe]
    for era in [e[0] for e in discover_eras()]:
        d = fs[fs.era == era]; k = d[d.stack_keep]
        print(f"  {era:4s} full-size raw {len(d):3d}·{100*(d.pnl>0).mean():4.1f}%·${d.pnl.sum():+9.2f}"
              f"  |  stack-kept {len(k):3d}·{100*(k.pnl>0).mean():4.1f}%·${k.stack_pnl.sum():+9.2f}")
    print("\nblock reasons (full-size):")
    for rz, g in fs[~fs.stack_keep].groupby('stack_block_reason'):
        print(f"  {rz:18s} {len(g):3d} · ${g.pnl.sum():+8.2f}")

if __name__ == '__main__':
    main()
