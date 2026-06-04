#!/usr/bin/env python3
"""structural_cell_analysis.py — Full-pool structural cell analysis (CLAUDE.md May 24).

Identifies:
  - LOSER cells (filter ship candidates) — should-be-blocked structural losers
  - WINNER cells (multiplier ship candidates) — passing the BE-floor gate
  - BE-rescue-dependent cells (NOT shippable until BE is permanently re-enabled)

Methodology (CLAUDE.md May 24 lock — full-pool structural analysis):

1. Use the FULL-column unified pool (build_pool_FULL.py) — not the LCD pool.
   Newer dimensions (BTC Gap, BTC 1h Slope, Pattern C/W, Extension) only
   captured post-May-13. The full pool keeps these columns where present.

2. Use Avg P&L % (leverage-invariant) NOT raw $ for cross-batch comparison.
   Per CLAUDE.md core operating principle: "When comparing results across
   different reports or batches, always use Avg P&L % instead of absolute P&L."

3. De-multiply historical $ for $-aggregations:
       pnl_1x = pnl / max(cell_multiplier × cell_lev_multiplier, 1)
   So $-totals reflect what the pool would have earned at 1× sizing.
   Comparing cells at consistent sizing exposes true edge.

4. LOSER ship gate (block candidate):
     N ≥ 15
     WR ≤ 45%
     Avg P&L % ≤ -0.10% (clearly losing, not just BE noise)
     dates ≥ 3 (multi-batch direction-consistency)
     pool $-impact ≤ -$30 (1×-equivalent)
     Direction-consistent across dates (preferred)

5. WINNER ship gate (multiplier candidate) — THE STRICT VERSION:
     N ≥ 15
     WR ≥ 70%
     Avg P&L % ≥ +0.10% (BE-floor gate — CLAUDE.md May 21 latest+5)
     dates ≥ 3
     pool $-impact ≥ +$30 (1×-equivalent)
     Median win % ≥ +0.12% (filters out BE-floor capture clusters)
     BE-rescue rate ≤ 30% of wins (most wins from TRAILING/FAST_EXIT —
       config-stable mechanisms, not just BE rescue)

6. BE-rescue-dependent FLAGGING — cells that LOOK profitable due to BE only:
     WR ≥ 65% but Avg P&L % < +0.10%
     AND BE-rescue rate > 30%
   These cannot be shipped as multipliers — under BE-OFF, the BE-rescued wins
   become losers and the cell collapses. Watchlist only.

Cell intersections analyzed (all 2D, single-axis trivial):
  - BTC RSI × BTC ADX (LONG / SHORT)
  - BTC Trend Gap × BTC ADX (LONG / SHORT)
  - BTC 1h Slope × BTC ADX (LONG / SHORT)
  - Pair RSI × Pair ADX (LONG / SHORT)
  - Range Position × ADX Δ (LONG / SHORT)
  - Pair Extension × PairVol Ratio (LONG / SHORT)

Output: structured findings table (stdout) + optional save to reports/.

Usage:
    python3 scripts/structural_cell_analysis.py
    python3 scripts/structural_cell_analysis.py --save reports/findings.txt
    python3 scripts/structural_cell_analysis.py --min-n 20 --min-dates 5
"""
import csv, os, sys, argparse
from collections import Counter

POOL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports', 'dedupe_pool_FULL.csv')


def f(v, d=None):
    """Safe float."""
    try:
        return float(v) if v not in ('', 'None', None) else d
    except (ValueError, TypeError):
        return d


def pnl_1x(t):
    """De-multiply: convert historical $ to 1× equivalent."""
    inv = f(t.get('cell_multiplier'), 1) or 1
    lev = f(t.get('cell_lev_multiplier'), 1) or 1
    eff = max(inv * lev, 1)
    return f(t['pnl'], 0) / eff


def analyze(arr, label, side, pred, col_required=None):
    """Run cell analysis. Returns dict or None if N < 10."""
    if col_required:
        m = [t for t in arr if t.get(col_required) not in ('', 'None', None) and pred(t)]
    else:
        m = [t for t in arr if pred(t)]
    if len(m) < 10:
        return None
    n = len(m)
    wins = [t for t in m if f(t['pnl'], 0) > 0]
    w = len(wins)
    tot_1x = sum(pnl_1x(t) for t in m)
    avg = sum(f(t['pnl_percentage'], 0) for t in m) / n
    np = sum(1 for t in m if (f(t['peak_pnl'], 0) or 0) < 0.05)
    dates = len(set(t['opened_at'][:10] for t in m))
    # BE-rescue diagnostic
    reasons = Counter(t.get('close_reason', '') for t in wins)
    be_rescue = sum(c for r, c in reasons.items() if 'BREAKEVEN' in r or 'BE_LEVEL' in r)
    trailing = sum(c for r, c in reasons.items() if 'TRAILING' in r)
    fast_exit = sum(c for r, c in reasons.items() if 'FAST_EXIT' in r)
    other_w = w - be_rescue - trailing - fast_exit
    # Median win %
    wins_pct = sorted([f(t['pnl_percentage'], 0) for t in wins])
    med_win = wins_pct[len(wins_pct) // 2] if wins_pct else 0
    return {
        'label': label, 'side': side, 'n': n, 'wr': w * 100 / n, 'avg': avg,
        'tot_1x': tot_1x, 'np': np, 'dates': dates, 'wins': w,
        'be': be_rescue, 'trail': trailing, 'fe': fast_exit, 'other_w': other_w,
        'med_win': med_win,
        'be_pct': be_rescue * 100 / w if w else 0,
    }


def cell_verdict(r, loser_gates, winner_gates):
    """Apply CLAUDE.md gates to classify cell."""
    # LOSER gates
    if (r['n'] >= loser_gates['n'] and r['wr'] <= loser_gates['wr']
            and r['avg'] <= loser_gates['avg'] and r['dates'] >= loser_gates['dates']
            and r['tot_1x'] <= loser_gates['tot1x']):
        return 'LOSER_BLOCK', '✗ SHIP BLOCK'
    # WINNER gates (strict)
    if (r['n'] >= winner_gates['n'] and r['wr'] >= winner_gates['wr']
            and r['avg'] >= winner_gates['avg'] and r['dates'] >= winner_gates['dates']
            and r['tot_1x'] >= winner_gates['tot1x']
            and r['med_win'] >= winner_gates['med_win']
            and r['be_pct'] <= winner_gates['be_pct_max']):
        return 'WINNER_MULT', '★ SHIP MULT'
    # BE-rescue suspect
    if r['wr'] >= 65 and r['avg'] < 0.10 and r['be_pct'] > 30:
        return 'BE_SUSPECT', '⚠ BE-rescue suspect — watchlist'
    # Marginal winner
    if r['wr'] >= 60 and r['avg'] >= 0.05:
        return 'MARGINAL_WIN', '? Marginal — needs more N'
    # Marginal loser
    if r['wr'] <= 50 and r['avg'] <= -0.05 and r['tot_1x'] <= 0:
        return 'MARGINAL_LOSS', '? Loser-leaning — needs more N'
    return 'NEUTRAL', '— neutral'


def build_cells():
    """Define all cell intersections to analyze."""
    cells = []
    # 1. BTC RSI × BTC ADX
    for side, rsi_buckets in [
        ('LONG', [(40, 50), (50, 55), (55, 60), (60, 65), (65, 70), (70, 100)]),
        ('SHORT', [(15, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 55)])
    ]:
        for rlo, rhi in rsi_buckets:
            for alo, ahi in [(15, 18), (18, 22), (22, 25), (25, 28), (28, 30), (30, 35), (35, 40)]:
                cells.append(('BTC_RSI_ADX', side,
                              f"BTC RSI {rlo}-{rhi} × BTC ADX {alo}-{ahi}",
                              lambda t, rl=rlo, rh=rhi, al=alo, ah=ahi:
                                  rl <= (f(t['entry_btc_rsi'], -1) or -1) < rh
                                  and al <= (f(t['entry_btc_adx'], -1) or -1) < ah,
                              'entry_btc_rsi'))
    # 2. BTC Trend Gap × BTC ADX (post-May-13)
    for side in ['LONG', 'SHORT']:
        for glo, ghi, glbl in [(-0.5, -0.25, '-0.5to-0.25'), (-0.25, -0.10, '-0.25to-0.10'),
                                (-0.10, 0, '-0.10to0'), (0, 0.10, '0to+0.10'),
                                (0.10, 0.20, '+0.10to+0.20'), (0.20, 0.50, '+0.20to+0.50')]:
            for alo, ahi in [(18, 22), (22, 25), (25, 30), (30, 35), (35, 40)]:
                cells.append(('BTC_GAP_ADX', side,
                              f"BTC Gap {glbl} × BTC ADX {alo}-{ahi}",
                              lambda t, gl=glo, gh=ghi, al=alo, ah=ahi:
                                  gl <= (f(t['entry_btc_trend_gap_pct'], -9) or -9) < gh
                                  and al <= (f(t['entry_btc_adx'], -1) or -1) < ah,
                              'entry_btc_trend_gap_pct'))
    # 3. BTC 1h Slope × BTC ADX
    for side in ['LONG', 'SHORT']:
        for slo, shi, slbl in [(-0.5, -0.20, '<-0.20'), (-0.20, -0.10, '-0.20to-0.10'),
                                (-0.10, 0, '-0.10to0'), (0, 0.10, '0to+0.10'),
                                (0.10, 0.20, '+0.10to+0.20'), (0.20, 0.50, '>+0.20')]:
            for alo, ahi in [(18, 25), (25, 30), (30, 35), (35, 40)]:
                cells.append(('BTC_1H_ADX', side,
                              f"BTC 1h Slope {slbl} × BTC ADX {alo}-{ahi}",
                              lambda t, sl=slo, sh=shi, al=alo, ah=ahi:
                                  sl <= (f(t['entry_btc_1h_slope'], -9) or -9) < sh
                                  and al <= (f(t['entry_btc_adx'], -1) or -1) < ah,
                              'entry_btc_1h_slope'))
    # 4. Pair RSI × Pair ADX
    for side, rsi_buckets in [
        ('LONG', [(40, 50), (50, 55), (55, 60), (60, 65), (65, 70)]),
        ('SHORT', [(20, 30), (30, 35), (35, 40), (40, 50)])
    ]:
        for rlo, rhi in rsi_buckets:
            for alo, ahi in [(15, 18), (18, 22), (22, 25), (25, 30), (30, 35)]:
                cells.append(('PAIR_RSI_ADX', side,
                              f"Pair RSI {rlo}-{rhi} × Pair ADX {alo}-{ahi}",
                              lambda t, rl=rlo, rh=rhi, al=alo, ah=ahi:
                                  rl <= (f(t['entry_rsi'], -1) or -1) < rh
                                  and al <= (f(t['entry_adx'], -1) or -1) < ah,
                              None))
    # 5. Range Position × ADX Δ
    for side in ['LONG', 'SHORT']:
        for rlo, rhi, rlbl in [(0, 5, '0-5'), (5, 10, '5-10'), (10, 15, '10-15'), (15, 25, '15-25'),
                                (75, 85, '75-85'), (85, 95, '85-95'), (95, 100, '95-100')]:
            for dlo, dhi in [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 2.0), (2.0, 5.0)]:
                cells.append(('RNGPOS_ADXD', side,
                              f"RngPos {rlbl} × ADXΔ {dlo}-{dhi}",
                              lambda t, rl=rlo, rh=rhi, dl=dlo, dh=dhi:
                                  rl <= (f(t['entry_range_position'], -1) or -1) < rh
                                  and dl <= (f(t['entry_adx_delta'], -9) or -9) < dh,
                              None))
    # 6. Pair Extension × PairVol (post-May-13)
    for side, exts in [
        ('LONG', [(-0.20, 0, '-0.20to0'), (0, 0.20, '0to+0.20'),
                  (0.20, 0.40, '+0.20to+0.40'), (0.40, 0.60, '+0.40to+0.60'),
                  (0.60, 1.0, '+0.60to+1.0')]),
        ('SHORT', [(-1.0, -0.60, '-1.0to-0.60'), (-0.60, -0.40, '-0.60to-0.40'),
                   (-0.40, -0.20, '-0.40to-0.20'), (-0.20, 0, '-0.20to0'),
                   (0, 0.20, '0to+0.20')])
    ]:
        for elo, ehi, elbl in exts:
            for pvlo, pvhi, pvlbl in [(0, 0.95, '<0.95'), (0.95, 1.10, '0.95-1.10'),
                                       (1.10, 1.50, '1.10-1.50'), (1.50, 3.0, '>1.50')]:
                cells.append(('EXT_PVOL', side,
                              f"Ext {elbl}% × PairVol {pvlbl}",
                              lambda t, el=elo, eh=ehi, pvl=pvlo, pvh=pvhi:
                                  el <= (f(t['entry_dist_from_ema13_pct'], -9) or -9) < eh
                                  and pvl <= (f(t['entry_pair_volume_ratio'], -1) or -1) < pvh,
                              'entry_dist_from_ema13_pct'))
    return cells


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pool', default=POOL_PATH)
    p.add_argument('--save', default=None, help='Save findings to this file')
    p.add_argument('--min-n', type=int, default=15)
    p.add_argument('--min-dates', type=int, default=3)
    args = p.parse_args()

    if not os.path.exists(args.pool):
        print(f"Pool not found at {args.pool}. Run scripts/build_pool_FULL.py first.", file=sys.stderr)
        sys.exit(1)

    trades = list(csv.DictReader(open(args.pool)))
    LONG = [t for t in trades if t['direction'] == 'LONG']
    SHORT = [t for t in trades if t['direction'] == 'SHORT']
    print(f"Pool: {len(LONG)}L + {len(SHORT)}S = {len(trades)} trades")
    dates = sorted(set(t['opened_at'][:10] for t in trades))
    print(f"Range: {dates[0]} → {dates[-1]} ({len(dates)} dates)\n")

    # Locked gates (CLAUDE.md May 24 methodology)
    loser_gates = {'n': args.min_n, 'wr': 45, 'avg': -0.10,
                   'dates': args.min_dates, 'tot1x': -30}
    winner_gates = {'n': args.min_n, 'wr': 70, 'avg': 0.10,
                    'dates': args.min_dates, 'tot1x': 30,
                    'med_win': 0.12, 'be_pct_max': 30}

    cells_def = build_cells()
    findings = []
    for cat, side, label, pred, col in cells_def:
        arr = LONG if side == 'LONG' else SHORT
        r = analyze(arr, label, side, pred, col_required=col)
        if r:
            r['cat'] = cat
            r['verdict_code'], r['verdict_text'] = cell_verdict(r, loser_gates, winner_gates)
            findings.append(r)

    losers = [r for r in findings if r['verdict_code'] == 'LOSER_BLOCK']
    winners = [r for r in findings if r['verdict_code'] == 'WINNER_MULT']
    be_suspect = [r for r in findings if r['verdict_code'] == 'BE_SUSPECT']

    losers.sort(key=lambda r: r['tot_1x'])
    winners.sort(key=lambda r: -r['tot_1x'])
    be_suspect.sort(key=lambda r: -r['tot_1x'])

    out = []

    def emit(line=''):
        out.append(line)
        print(line)

    emit('=' * 110)
    emit(f"FULL-POOL STRUCTURAL CELL ANALYSIS — locked CLAUDE.md May 24 methodology")
    emit('=' * 110)
    emit(f"Pool: {len(trades)} trades ({len(LONG)}L / {len(SHORT)}S), {dates[0]} → {dates[-1]} ({len(dates)} dates)")
    emit(f"Gates: LOSER (block) — N≥{loser_gates['n']}, WR≤{loser_gates['wr']}%, Avg≤{loser_gates['avg']}%, ≥{loser_gates['dates']} dates")
    emit(f"       WINNER (mult) — N≥{winner_gates['n']}, WR≥{winner_gates['wr']}%, Avg≥+{winner_gates['avg']}%, MedWin≥+{winner_gates['med_win']}%, BE-rescue≤{winner_gates['be_pct_max']}%")
    emit(f"\n$ values shown as 1×-equivalent (de-multiplied: pnl / max(cell_multiplier × cell_lev_multiplier, 1))")
    emit('')

    emit('=' * 110)
    emit(f"STRUCTURAL LOSERS — {len(losers)} ship candidates (sorted by $-impact)")
    emit('=' * 110)
    emit(f"{'Category':<14} {'Cell':<46} {'Side':<5} {'N':>3} {'WR':>4} {'Avg%':>8} {'MedW':>7} {'$ 1×':>8} {'NP':>5} {'D':>2}")
    emit('-' * 110)
    for r in losers:
        emit(f"{r['cat']:<14} {r['label'][:45]:<46} {r['side']:<5} {r['n']:>3} {r['wr']:>3.0f}% {r['avg']:>+7.3f}% {r['med_win']:>+6.3f}% ${r['tot_1x']:>+6.2f} {r['np']:>2}/{r['n']:<2} {r['dates']:>2}")

    emit('')
    emit('=' * 110)
    emit(f"STRUCTURAL WINNERS (passes BE-floor gate) — {len(winners)} multiplier candidates")
    emit('=' * 110)
    emit(f"{'Category':<14} {'Cell':<46} {'Side':<5} {'N':>3} {'WR':>4} {'Avg%':>8} {'MedW':>7} {'$ 1×':>8} {'BE%':>4} {'D':>2}")
    emit('-' * 110)
    for r in winners:
        emit(f"{r['cat']:<14} {r['label'][:45]:<46} {r['side']:<5} {r['n']:>3} {r['wr']:>3.0f}% {r['avg']:>+7.3f}% {r['med_win']:>+6.3f}% ${r['tot_1x']:>+6.2f} {r['be_pct']:>3.0f}% {r['dates']:>2}")

    emit('')
    emit('=' * 110)
    emit(f"BE-RESCUE-DEPENDENT CELLS — {len(be_suspect)} (NOT shippable as multipliers under BE-OFF)")
    emit('=' * 110)
    emit(f"{'Category':<14} {'Cell':<46} {'Side':<5} {'N':>3} {'WR':>4} {'Avg%':>8} {'$ 1×':>8} {'BE%':>4} {'D':>2}")
    emit('-' * 110)
    for r in be_suspect:
        emit(f"{r['cat']:<14} {r['label'][:45]:<46} {r['side']:<5} {r['n']:>3} {r['wr']:>3.0f}% {r['avg']:>+7.3f}% ${r['tot_1x']:>+6.2f} {r['be_pct']:>3.0f}% {r['dates']:>2}")

    emit('')
    emit(f"Total cells examined: {len(findings)}")
    emit(f"  ★ Multiplier candidates: {len(winners)}")
    emit(f"  ✗ Filter (block) candidates: {len(losers)}")
    emit(f"  ⚠ BE-rescue dependent (watchlist): {len(be_suspect)}")

    if args.save:
        with open(args.save, 'w') as fh:
            fh.write('\n'.join(out))
        print(f"\nSaved to {args.save}", file=sys.stderr)


if __name__ == '__main__':
    main()
