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

STACK_VERSION = "2026-08-16a"  # a: FAKE_BULL_GUARD gate REMOVED (guard reverted by locked gate 47 after forward refutation — 12-block replay 6W/6L). Restores the 2026-08-10c keep-set. NOTE: cap35 (8108a60) is EXIT-side and path-dependent — stack_pnl deliberately NOT re-priced for it (floor-bound CF is optimistic; forward accounting = bound='cap' tallies).
G = 'entry_pair_ema20_ema50_gap_pct'   # holds EMA13-50 (known misnomer — do not rename)

# Era registry (Sep-11: B3/B4/B5 were previously stacked by a one-off — the builder only knew
# BASE/B1/B2, so the committed MASTER_POOL had B3/B4 rows with NULL stack columns and no B5).
# Fixed eras are listed explicitly; every later batch is auto-discovered from the archive naming
# convention reports/BASELINE<n>_*.csv (n ≥ 6 → era B<n>). Archive each batch under that name
# at its pre-reset export and the pool picks it up on the next regen.
# (era, path, min_opened_at, lenient_status) — lenient_status=True keeps NaN-status rows as CLOSED
# (the B1 anchor file's legacy quirk); every other era is strict `status == "CLOSED"`.
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
        if era != 'BASE':  # BASE is already the screened CLOSED set
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
    df['is_door'] = src.str.contains('CALM3D')
    vol = df.entry_pair_volume_24h_usd
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
            if strat == 'SPIKE_FADE':
                if r.entry_btc_rsi > 45: k, why = False, 'FADE_BRSI45'  # engine uses strict > (45.0 passes)
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
            elif strat.startswith('MOMENTUM') or slv.startswith('MOM'):
                if i in cooldown_idx: k, why = False, 'CALM3D_REENTRY'
                elif r.is_door and pd.notna(r.entry_pos_di) and r.entry_pos_di < 28: k, why = False, 'CALM3D_DMI_DI'
                elif r.is_door and pd.notna(r.entry_adx) and r.entry_adx < 21: k, why = False, 'CALM3D_DMI_ADX'
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
