#!/usr/bin/env python3
"""build_pool_FULL.py — Full-column unified pool builder.

Sister to build_unified_pool.py but:
  - UNION columns across all CSVs (not LCD intersection)
  - Preserves newer columns (entry_btc_trend_gap_pct, entry_btc_1h_slope,
    entry_btc_atr_pct, entry_pattern_*, entry_dist_from_ema13_pct,
    cell_multiplier, cell_lev_multiplier, etc.) where present
  - Includes archived CSVs from reports/ AND fresh ones from ~/Downloads/

Output: reports/dedupe_pool_FULL.csv
Dedup key: (opened_at, pair, direction) — same as build_unified_pool.py.

Usage:
    python3 scripts/build_pool_FULL.py
    python3 scripts/build_pool_FULL.py --quiet
    python3 scripts/build_pool_FULL.py --from-date 2026-05-04

Why this matters: many analytical surfaces (BTC Gap × BTC ADX cross-tab,
BTC 1h Slope analytics, Pattern C/W trackers, Extension multiplier) rely
on columns added AFTER the original capture. The LCD-intersection pool
silently drops these columns, hiding cells like:
  - LONG RngPos 85-95 × ADXΔ 0-0.3 (N=47, -$537 cross-batch, never blocked)
  - SHORT Ext -0.40 to -0.20 × PairVol<0.95 (N=19, 95% WR, +$203)
This builder preserves all columns so structural cell analysis works.
"""
import csv, glob, os, sys
from datetime import datetime


def build_pool(reports_dir, downloads_dir, from_date=None, quiet=False):
    """Build full-column dedup pool from archived + fresh CSVs."""
    files = sorted(glob.glob(os.path.join(reports_dir, 'orders_*.csv')))
    fresh = sorted(glob.glob(os.path.join(downloads_dir, 'scalpars_orders_paper_*.csv')))
    files.extend(fresh)
    if not files:
        print(f"No order CSVs found in {reports_dir} or {downloads_dir}", file=sys.stderr)
        sys.exit(1)

    all_cols = set()
    rows = []
    per_file_counts = {}
    for fname in files:
        added = 0
        with open(fname) as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                all_cols.update(r.keys())
                rows.append(r)
                added += 1
        per_file_counts[fname] = added

    # Dedup by (opened_at, pair, direction) — match build_unified_pool.py
    seen = set()
    deduped = []
    file_counts = {}
    for r in rows:
        if r.get('status') != 'CLOSED':
            continue
        try:
            o = datetime.fromisoformat(r['opened_at'].replace('Z', '').split('+')[0])
        except (KeyError, ValueError):
            continue
        if from_date:
            try:
                cutoff = datetime.fromisoformat(from_date)
                if o < cutoff:
                    continue
            except ValueError:
                pass
        key = (r['opened_at'], r['pair'], r['direction'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # Write
    output = os.path.join(reports_dir, 'dedupe_pool_FULL.csv')
    fieldnames = sorted(all_cols)
    with open(output, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in deduped:
            writer.writerow({k: r.get(k, '') for k in fieldnames})

    if not quiet:
        print(f"=== Full-column Unified Pool (dedup by opened_at+pair+direction) ===\n")
        if from_date:
            print(f"Filter: opened_at >= {from_date}")
        long_n = sum(1 for r in deduped if r['direction'] == 'LONG')
        short_n = sum(1 for r in deduped if r['direction'] == 'SHORT')
        dates = sorted(set(r['opened_at'][:10] for r in deduped))
        print(f"Total: {len(deduped)} closed trades ({long_n}L / {short_n}S)")
        print(f"Columns preserved: {len(all_cols)} (union across {len(files)} files)")
        print(f"Date range: {dates[0]} → {dates[-1]} ({len(dates)} distinct dates)")
        print(f"Output: {output}")

        # Coverage of key columns
        def f(v): return v not in ('', 'None', None)
        keycols = [
            ('entry_btc_trend_gap_pct', 'BTC Gap × ADX cross-tab'),
            ('entry_btc_1h_slope', 'BTC 1h Slope analytics'),
            ('entry_btc_atr_pct', 'BTC Volatility regime'),
            ('entry_pattern_c1_match', 'Pattern C trackers'),
            ('entry_pattern_w6_match', 'Pattern W trackers'),
            ('entry_dist_from_ema13_pct', 'Extension dim'),
            ('entry_pair_volume_ratio', 'Pair Vol cross-tabs'),
            ('cell_multiplier', 'Multiplier source attribution'),
            ('cell_lev_multiplier', 'Lev multiplier (May 21+)'),
        ]
        print(f"\nColumn coverage (% of trades with value populated):")
        for col, desc in keycols:
            if col not in all_cols:
                print(f"  {col:32s} NOT IN ANY FILE — {desc}")
                continue
            n = sum(1 for r in deduped if f(r.get(col)))
            print(f"  {col:32s} {n:>4d}/{len(deduped):<4d} ({n*100/len(deduped):>3.0f}%) — {desc}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--reports-dir', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports'))
    p.add_argument('--downloads-dir', default=os.path.expanduser('~/Downloads'))
    p.add_argument('--from-date', default=None, help='Cutoff like 2026-05-04 (inclusive)')
    p.add_argument('--quiet', action='store_true')
    args = p.parse_args()
    build_pool(args.reports_dir, args.downloads_dir, args.from_date, args.quiet)
