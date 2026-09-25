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
  stack_version  regenerate after EVERY filter ship: ./venv/bin/python scripts/build_master_pool.py

Raw pools stay untouched (ground truth). stack_keep is exact (entry gates);
stack_pnl layers mechanism counterfactuals — analyses must say which they used.
"""
import warnings; warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
from datetime import datetime

STACK_VERSION = "2026-09-25b"  # b: MOM_SHORT_C1_REGIME — C1 momentum shorts refused when BTC is STRONG_BEAR (operator ARMED override, DECISION_LOG 114; rule = engine mom_short_c1_regime_block). Prior: 2026-09-25a # a: FLIP_FAN_WEAK_BOUNCE — FAN flip-shorts refused when pair EMA13−EMA50 gap < 0 AND EMA20 slope < 0.15 (operator ARMED override at N=8, DECISION_LOG 113; rule = engine flip_fan_weak_bounce); B12 snapshot as-of 09-25. Prior: 2026-09-24b # b: FADE_BRSI 45→50 — the Aug-5 ceiling's own pre-committed revert fired (DECISION_LOG 112); label FADE_BRSI45→FADE_BRSI50. Prior: 2026-09-24a # a: CF_FADE_LATE_ARM — SPIKE_FADE never armed, open past 15 min, peak after 15 in [0.30,0.40) → re-priced to the late trail floor (stamps-only, optimistic: exposed late winners not repriced; DECISION_LOG 111). Prior: 2026-09-23a # a: LONG_MEGACAP_BLOCK — momentum longs (unmatched + doors) refused at raw eligible-universe rank ≤ 10 (operator override at N=10, DECISION_LOG 110; rule = engine long_megacap_block). Prior: 2026-09-18b # b: LONG_HEAT_BLOCK — momentum longs (unmatched + doors) refused at BTC slope≥0.07 ∧ BTC RSI prev≥64 ∧ bull≥80 unless BTC ≤−10% vs its 30d high (DECISION_LOG Sep-18 (70); rule = engine long_heat_eval, 30d reading = stamped column else reports/btc_off30d_hourly.csv); era B8 (Sep 16-18) added. Prior: 2026-09-18a # a: MOM_SHORT_PAIRVOL — momentum shorts blocked at pair-vol ratio ≥ 0.86 (ceiling tightened 1.0→0.86, DECISION_LOG Sep-18 (68)). Prior: 2026-09-16a # a: FLIP_FAN_BTC_EMA13 — FAN_RATIO_GATE shorts blocked when BTC dist-EMA13 > -0.08 (Aug-23 live gate, builder gap caught Sep-16). Prior: 2026-09-15a # a: gate 60 BEARRUN_SHORT — 1× probe fills PROBE_EXEMPT, armed fills own-sleeve label (never MOM-short). Prior: 2026-09-14a # a: FADE_MAXVOL — SPIKE_FADE blocked at 24h vol ≥ $20M (Sep-14 operator override, DECISION_LOG 55); engine tests it FIRST among the fade gates. Prior: 2026-08-16a # a: FAKE_BULL_GUARD gate REMOVED (guard reverted by locked gate 47 after forward refutation — 12-block replay 6W/6L). Restores the 2026-08-10c keep-set. NOTE: cap35 (8108a60) is EXIT-side and path-dependent — stack_pnl deliberately NOT re-priced for it (floor-bound CF is optimistic; forward accounting = bound='cap' tallies).
G = 'entry_pair_ema20_ema50_gap_pct'   # holds EMA13-50 (known misnomer — do not rename)

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
    from services.trading_engine import long_heat_eval, long_megacap_block, fade_late_arm_cf, flip_fan_weak_bounce, mom_short_c1_regime_block
    # ⏱ Sep-24 frozen fade exit constants for the late-arm CF (live values at ship; builder pins a STACK_VERSION)
    _FADE_TH = SimpleNamespace(spike_fade_late_arm_after_min=15.0, spike_fade_late_arm_peak=0.30,
                               runner_trail_short_arm_peak=0.40, runner_trail_short_atr_mult=0.5,
                               runner_trail_short_giveback_frac=0.35, runner_trail_short_be_ratchet_enabled=False,
                               runner_trail_short_be_lock_pct=0.10)
    # frozen stack constants (the builder pins a STACK_VERSION, it does not read hot config) — Sep-23: mega-cap rank ≤10
    _HEAT_TH = SimpleNamespace(long_heat_block_enabled=True, long_heat_btc_slope_min=0.07, long_heat_btc_rsi_prev_min=64.0,
                               long_heat_bull_pct_min=80.0, long_heat_exempt_off30d_max=-10.0,
                               long_megacap_rank_max=10)
    # 🪃 Sep-25 frozen fan-flip weak-bounce constants (live values at ship; DECISION_LOG 113)
    _FLIP_TH = SimpleNamespace(flip_fan_weak_bounce_enabled=True, flip_fan_weak_bounce_gap_max=0.0,
                               flip_fan_weak_bounce_slope_max=0.15)
    # 🧊 Sep-25 frozen C1 momentum-short regime block (live value at ship; DECISION_LOG 114)
    _C1_TH = SimpleNamespace(momentum_short_c1_block_regimes='STRONG_BEAR')
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
    keep, reason, spnl = [], [], []
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
        if r.is_probe:
            why = 'PROBE_EXEMPT'
        if not r.is_probe:
            if strat == 'BEARRUN_SHORT':
                why = 'BEARRUN_SLEEVE'   # kept, but its OWN sleeve — MOM-short reads must filter entry_strategy == 'MOMENTUM'
            elif strat == 'SPIKE_FADE':
                if v is not None and v >= 20e6: k, why = False, 'FADE_MAXVOL'   # Sep-14 ceiling — engine order: first fade gate
                elif r.entry_btc_rsi > 50: k, why = False, 'FADE_BRSI50'  # engine uses strict > (50.0 passes); Sep-24 45→50 (DECISION_LOG 112)
                elif pd.notna(r.entry_btc_dist_from_ema13_pct) and r.entry_btc_dist_from_ema13_pct > 0: k, why = False, 'FADE_BD13'
                elif v is not None and v < 2e6: k, why = False, 'FLOOR_2M'
                elif (pd.notna(r.entry_rsi_prev) and r.entry_rsi_prev < 44
                      and pd.notna(r[G]) and r[G] > -0.40): k, why = False, 'FADE_FRESHBREAK'
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
                elif (str(r.direction) == 'LONG'
                      and long_heat_eval(_HEAT_TH, r.entry_btc_ema20_slope, r.get('entry_btc_rsi_prev'), r.entry_bull_pct, _row_off30d(r))[1]):
                    k, why = False, 'LONG_HEAT_BLOCK'
                # Sep-23: 🏦 mega-cap exclusion — momentum longs refused at RAW eligible-universe rank ≤ long_megacap_rank_max
                # (engine long_megacap_block; fail-open on an unstamped rank). Live parity, DECISION_LOG 110.
                elif str(r.direction) == 'LONG' and long_megacap_block(_HEAT_TH, r.get('entry_pair_rank')):
                    k, why = False, 'LONG_MEGACAP_BLOCK'
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
        if k and not r.is_probe and (strat.startswith('MOMENTUM') or slv.startswith('MOM')) and str(r.direction) == 'LONG':
            pk, atr = r.peak_pnl, (r.entry_atr_pct if pd.notna(r.entry_atr_pct) else 99)
            if pd.notna(pk) and 0.40 <= pk < 0.45 and pd.notna(r.pnl_percentage) and r.pnl_percentage < max(pk - atr, 0.10) and r.pnl_percentage != 0:
                sp = max(pk - atr, 0.10) / 100 * abs(p / (r.pnl_percentage / 100)); why = why or 'CF_ARM040'
            elif (not r.is_door and pd.notna(r.entry_global_volume_ratio) and pd.notna(r.entry_btc_ema20_slope)
                  and r.entry_global_volume_ratio > 0.74 and r.entry_btc_ema20_slope > 0.07):
                # engine de-mux only strips a >1x boost on UNMATCHED cells — a trade that
                # actually sized 1x is untouched (no 2.0 fallback: that halved real 1x P&L)
                m = pd.to_numeric(r.cell_multiplier, errors='coerce')
                if pd.notna(m) and m > 1 and 'UNMATCHED' in str(r.cell_multiplier_source or '').upper():
                    sp = p / m; why = why or 'CF_SPRINT_DEMUX'
        keep.append(k); reason.append(why); spnl.append(sp if k else 0.0)
    df['stack_keep'] = keep; df['stack_block_reason'] = reason
    df['stack_pnl'] = np.round(spnl, 2); df['stack_version'] = STACK_VERSION
    df = df.drop(columns=['_ts'])
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
