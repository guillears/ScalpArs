#!/usr/bin/env python3
"""
gate51_cross.py — the MASTER-POOL cross for gate 51 (the docket's "master-pool cross is MANDATORY"
read, Sep-11). The dashboard's 🔓 Gate 51 tables are reset-censored (live DB only); this script
reproduces BOTH tables (band rows with their locked bars + the 6-cell partition + dedup total) from
reports/MASTER_POOL_stacked.csv plus any extra order CSVs you point it at (the current batch's
dashboard export, or a not-yet-stacked archive).

    ./venv/bin/python scripts/gate51_cross.py                       # master pool only
    ./venv/bin/python scripts/gate51_cross.py ~/Downloads/scalpars_orders_paper_*.csv   # + current batch
    ./venv/bin/python scripts/gate51_cross.py --list --history ~/Downloads/<batch>.csv     # + founding-era per-cell record

Cohort = the dashboard's exactly: momentum LONGs, CLOSED, non-probe, opened ≥ the gate-51 deploy
(2026-08-18 15:30 UTC). Extra CSVs are de-duplicated against the pool on the locked key
(opened_at, pair, direction) — NEVER `id`. Band predicates and the cell classifier are imported
from main.py (single source of truth — the dashboard and this script cannot drift apart).
"""
import argparse, os, sys
import warnings; warnings.filterwarnings('ignore')
import pandas as pd

import logging; logging.disable(logging.CRITICAL)  # main.py logs a CRITICAL auth line at import — noise here
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import _g51_cell, _G51_CELL_ORDER, _G51_CELL_LABEL  # noqa: E402
logging.disable(logging.NOTSET)
RAW_PRE_POOL = "reports/dedupe_pool.csv"   # May-4 → Jun-25 raw pool: the founding evidence the old gates were built on

G51_TS = "2026-08-18T15:30:00"
BANDS = [  # label, predicate(rsi, adx), gate_n, wr_bar, revert text — mirrors main.py gate51_bands
    ("① RSI 50-55 band (was TOTAL block)",
     lambda r, a: pd.notna(r) and 50 <= r < 55, 10, 45, "restore 50-55:99-100"),
    ("② BTC ADX 15-18 floor cohort",
     lambda r, a: pd.notna(a) and 15 <= a < 18, 8, 45, "btc_adx_min_long back to 18"),
    ("③ 55-60 window new zone [15,20)∪(25,30]",
     lambda r, a: pd.notna(r) and pd.notna(a) and 55 <= r < 60 and (15 <= a < 20 or 25 < a <= 30),
     10, 45, "window back to 20-25"),
]
NUM = ['entry_btc_rsi', 'entry_btc_adx', 'entry_atr_pct', 'pnl', 'pnl_percentage', 'peak_pnl',
       'cell_multiplier', 'entry_btc_1h_slope']


def load(extra_paths, since):
    pool = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    pool['src'] = pool['era']
    frames = [pool]
    for p in extra_paths:
        d = pd.read_csv(p, low_memory=False); d['src'] = os.path.basename(p)[:28]
        frames.append(d)
    a = pd.concat(frames, ignore_index=True, sort=False)
    for c in NUM:
        a[c] = pd.to_numeric(a.get(c), errors='coerce')
    a['key'] = a.opened_at.astype(str).str[:19] + '|' + a.pair.astype(str) + '|' + a.direction.astype(str)
    before = len(a); a = a.drop_duplicates('key', keep='first'); dropped = before - len(a)
    probe = (a.get('cell_multiplier_source', pd.Series('', index=a.index)).astype(str).str.endswith('_PROBE')
             | a.get('is_probe', pd.Series(False, index=a.index)).fillna(False).astype(bool))
    ml = a[(a.status == 'CLOSED') & (a.direction == 'LONG')  # strict, like the builder (only B1 is lenient)
           & (a.entry_strategy.fillna('MOMENTUM').isin(['MOMENTUM', ''])) & ~probe
           & (a.opened_at.astype(str) >= since)].copy()
    # NO stack_keep filter — the dashboard applies none, and this script mirrors the dashboard
    # exactly (review Sep-11). The count of stack-blocked rows is disclosed in the header instead.
    stack_blocked = int((ml.stack_keep.astype(str).str.lower() == 'false').sum()) if 'stack_keep' in ml else 0
    ml['cell'] = [_g51_cell(r, x) for r, x in zip(ml.entry_btc_rsi, ml.entry_btc_adx)]
    return ml, dropped, stack_blocked


def load_pre():
    """Founding-era raw pool (May-4 → Jun-25, old exit stack; Jun-9→25 ran keep-only-unmatched):
    REFUTE-ONLY under the locked rule. The pool predates probe cells, sleeves and multipliers — it has no
    cell_multiplier / cell_multiplier_source / entry_strategy columns, so the probe/strategy guards below
    are no-ops on it and exist only so a richer file dropped in its place is handled the same way."""
    d = pd.read_csv(RAW_PRE_POOL, low_memory=False)
    for c in NUM:
        d[c] = pd.to_numeric(d.get(c), errors='coerce')
    # (dedupe_pool.csv has no cell_multiplier → the NUM loop manufactures an all-NaN column → no probes, 1× = as-sized)
    probe = d.get('cell_multiplier_source', pd.Series('', index=d.index)).astype(str).str.endswith('_PROBE') \
        | (d.cell_multiplier.fillna(1) < 1)
    if 'entry_strategy' in d:
        d = d[d.entry_strategy.fillna('MOMENTUM').isin(['MOMENTUM', ''])]
    d = d[(d.status.fillna('CLOSED') == 'CLOSED') & (d.direction == 'LONG') & ~probe
          & (d.opened_at.astype(str) < G51_TS)].copy()
    d['cell'] = [_g51_cell(r, x) for r, x in zip(d.entry_btc_rsi, d.entry_btc_adx)]
    return d


def stats(d):
    if len(d) == 0:
        return dict(n=0, wr=None, avg=None, tot=0.0, one_x=0.0, pk=None, dates=0, worst=None)
    pp = d.pnl_percentage.fillna(0.0)  # dashboard parity: _cohort_stats uses pnl_percentage for WR and avg
    return dict(n=len(d), wr=100 * (pp > 0).mean(), avg=pp.mean(), tot=d.pnl.sum(),
                one_x=(d.pnl / d.cell_multiplier.fillna(1).replace(0, 1)).sum(),
                pk=d.peak_pnl.mean(), dates=d.opened_at.astype(str).str[:10].nunique(),
                worst=d.pnl_percentage.min())


def verdict(st, gate_n, wr_bar, txt):  # mirrors main.py _door
    n, wr, tot = st['n'], st['wr'], st['tot']
    bad = (wr is not None and wr <= wr_bar) or tot < 0
    if n >= gate_n and bad:
        return f"✗ REVERT: N={n}>={gate_n} WR {wr:.0f}% / Σ${tot:+.0f} — {txt}"
    if n >= max(4, gate_n - 3) and bad:
        return f"⚠ approaching revert ({n}/{gate_n})"
    if n == 0:
        return "⏳ no fires yet"
    return f"⏳ building ({n}/{gate_n})" + (" ✓ healthy" if tot >= 0 else "")


def fmt(label, st, note, w=52):
    f = lambda v, s: ('-' if v is None else s % v)
    return (f"{label:<{w}} {st['n']:>4} {f(st['wr'], '%5.1f'):>6} {f(st['avg'], '%+.3f'):>8} "
            f"{('$%+.0f' % st['tot']):>9} {('$%+.0f' % st['one_x']):>8} {f(st['pk'], '%+.2f'):>7} {st['dates']:>5}  {note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('extra', nargs='*', help='extra order CSVs (current batch export, unstacked archives)')
    ap.add_argument('--since', default=G51_TS, help='cohort floor (default = gate-51 deploy)')
    ap.add_argument('--list', action='store_true', help='print every band fill with its cell')
    ap.add_argument('--history', action='store_true',
                    help='append the founding-era (May-Jun raw pool) per-cell record — REFUTE-ONLY')
    args = ap.parse_args()
    ml, dropped, stack_blocked = load(args.extra, args.since)
    hdr = f"{'':<52} {'N':>4} {'WR%':>6} {'avg%':>8} {'Σ$':>9} {'1×$':>8} {'avgPk':>7} {'Dates':>5}  Gate verdict"
    print(f"🔓 GATE 51 — master-pool cross · momentum LONGs · CLOSED · non-probe · opened ≥ {args.since}")
    print(f"   sources: MASTER_POOL_stacked.csv" + (f" + {len(args.extra)} extra" if args.extra else '')
          + f" · {dropped} duplicate rows dropped on (opened_at, pair, direction) · cohort N={len(ml)} "
          f"Σ${ml.pnl.sum():+.0f} WR {100 * (ml.pnl > 0).mean() if len(ml) else 0:.0f}%"
          + (f" · ⚠ {stack_blocked} of these are stack_keep=False (kept — dashboard parity)" if stack_blocked else '') + "\n")
    print("BANDS (overlap by design — each row answers its own locked revert)"); print(hdr)
    masks = {}
    for label, pred, gn, wb, txt in BANDS:
        m = ml.apply(lambda o: bool(pred(o.entry_btc_rsi, o.entry_btc_adx)), axis=1) if len(ml) else pd.Series(dtype=bool)
        masks[label] = m
        print(fmt(label, stats(ml[m]), verdict(stats(ml[m]), gn, wb, txt)))
    print("\nCELLS (partition of the bands — attribution only, no locked bar)"); print(hdr.replace('Gate verdict', 'Note'))
    for ck in _G51_CELL_ORDER:
        d = ml[ml.cell == ck]
        print(fmt(_G51_CELL_LABEL[ck], stats(d), '— attribution only' if len(d) else '⏳ no fires yet'))
    anyb = pd.concat(masks.values(), axis=1).any(axis=1) if masks and len(ml) else pd.Series(False, index=ml.index)
    uniq = ml[anyb]; uncl = int((uniq.cell.isna()).sum())
    print(fmt("Σ unique band fills (bands overlap; cells do not)", stats(uniq),
              '— dedup total' + (f' · ⚠ {uncl} band fill(s) not in any cell' if uncl else '')))
    print(fmt("non-band momentum longs (reference)", stats(ml[~anyb]), '— outside all three bands'))
    if args.list and len(uniq):
        print("\nBAND FILLS")
        cols = ['src', 'opened_at', 'pair', 'entry_btc_rsi', 'entry_btc_adx', 'entry_atr_pct',
                'entry_btc_1h_slope', 'cell_multiplier', 'pnl', 'pnl_percentage', 'close_reason', 'cell']
        print(uniq.sort_values('opened_at')[cols].to_string(index=False))
    if args.history:
        pre = load_pre()
        print(f"\nHISTORY — founding-era UNSCREENED raw pool (incl. now-blacklisted pairs; {RAW_PRE_POOL}, {str(pre.opened_at.min())[:10]} → {str(pre.opened_at.max())[:10]}, "
              f"old exit stack, Jun-9→25 keep-only-unmatched): REFUTE-ONLY — may argue against a change, never for one")
        print(f"{'cell':<52} {'PRE N':>5} {'WR%':>6} {'avg%':>8} {'months':<16} {'POST N':>6} {'WR%':>6} {'avg%':>8} {'Σ$':>9}   (PRE $ omitted — sizing differed by batch; compare avg%)")
        for ck in _G51_CELL_ORDER:
            a_, b_ = stats(pre[pre.cell == ck]), stats(ml[ml.cell == ck])
            f = lambda v, s: ('-' if v is None else s % v)
            mo = ','.join(sorted(pre[pre.cell == ck].opened_at.astype(str).str[:7].unique()))
            print(f"{_G51_CELL_LABEL[ck]:<52} {a_['n']:>5} {f(a_['wr'], '%5.1f'):>6} {f(a_['avg'], '%+.3f'):>8} {mo:<16} "
                  f"{b_['n']:>6} {f(b_['wr'], '%5.1f'):>6} {f(b_['avg'], '%+.3f'):>8} {('$%+.0f' % b_['tot']):>9}")
    print("\n⚠ falling-BTC tripwire (first −2% BTC day with the bot running): that day's band fills ≤30% WR ∨ ≤−$250 → ALL THREE revert — check the BTC daily tape by hand.")


if __name__ == '__main__':
    main()
