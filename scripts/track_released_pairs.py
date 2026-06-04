#!/usr/bin/env python3
"""
Released-pair fresh-data tracker (2026-06-04).

Resolves the revert gate locked on 2026-06-03 when the pair_blacklist was trimmed
23->10 and 12 thin-N pairs were RELEASED for forward re-test.

LOCKED REVERT GATE: re-blacklist any released pair at <=35% WR on N>=10 fresh
(current-stack) trades. "Fresh" = opened_at >= CUTOFF (the release date), so only
trades taken AFTER the release under the current filter/exit stack count.

Usage:
  python3 scripts/track_released_pairs.py [extra_order_csv ...]
Scans reports/*.csv by default, plus any CSV paths passed as args (e.g. a fresh
batch export in ~/Downloads). Dedups by (opened_at, pair, direction); CLOSED only.
"""
import csv, glob, sys, os, collections

RELEASED = ['ADAUSDT','ASTERUSDT','BCHUSDT','DOGEUSDT','ERAUSDT','HYPEUSDT',
            'ICPUSDT','LABUSDT','LINKUSDT','PUMPUSDT','SKYAIUSDT','WLFIUSDT']
CUTOFF = '2026-06-03'          # release date; only opened_at >= this counts as fresh
N_GATE = 10                    # minimum fresh trades to resolve the gate
WR_REVERT = 35.0               # WR% at/below which (on N>=N_GATE) -> re-blacklist

def fnum(v):
    try: return float(v)
    except: return float('nan')

def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(here, 'reports', '*.csv'))) + sys.argv[1:]
    rows, seen = [], set()
    for p in paths:
        try: f = open(p)
        except OSError: continue
        for d in csv.DictReader(f):
            if d.get('status') != 'CLOSED': continue
            if d.get('pair') not in RELEASED: continue
            oa = (d.get('opened_at') or '')
            if oa[:10] < CUTOFF: continue            # fresh only
            k = (oa, d.get('pair'), d.get('direction'))
            if k in seen: continue
            seen.add(k); rows.append(d)

    by = collections.defaultdict(list)
    for d in rows: by[d['pair']].append(d)

    print(f"RELEASED-PAIR FRESH TRACKER  (cutoff opened_at >= {CUTOFF}; gate: re-blacklist at <={WR_REVERT:.0f}% WR on N>={N_GATE})")
    print(f"scanned {len(paths)} files, {len(rows)} fresh released-pair trades\n")
    print(f"{'pair':11s} {'N':>3s} {'W/L':>6s} {'WR%':>5s} {'avgP&L%':>8s} {'Tot$':>9s}  status")
    print('-'*72)
    tot_n = 0
    for p in RELEASED:
        r = by.get(p, [])
        n = len(r); tot_n += n
        if n == 0:
            print(f"{p:11s} {0:3d} {'-':>6s} {'-':>5s} {'-':>8s} {'-':>9s}  collecting (0/{N_GATE})")
            continue
        w = sum(1 for d in r if fnum(d.get('pnl_percentage')) > 0); l = n - w
        wr = 100*w/n
        avg = sum(fnum(d.get('pnl_percentage')) for d in r)/n
        tot = sum(fnum(d.get('pnl')) for d in r)
        if n < N_GATE:
            status = f"collecting ({n}/{N_GATE})"
        elif wr <= WR_REVERT:
            status = f"⚠ REVERT — re-blacklist (WR {wr:.0f}% <= {WR_REVERT:.0f}%)"
        else:
            status = f"✓ HOLD — validated (WR {wr:.0f}%)"
        print(f"{p:11s} {n:3d} {w:>3d}/{l:<2d} {wr:5.0f} {avg:+8.3f} {tot:+9.1f}  {status}")
    print('-'*72)
    print(f"total fresh released-pair trades: {tot_n}")

if __name__ == '__main__':
    main()
