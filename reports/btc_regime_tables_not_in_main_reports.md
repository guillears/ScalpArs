# BTC Regime Tables (Performance by BTC Regime at Entry)

The detailed 7-bucket BTC Regime breakdown (CHOPPY_WEAK / CHOPPY_FLAT / HEALTHY_BULL /
STRONG_BULL / BULL_EXHAUSTED / HEALTHY_BEAR / STRONG_BEAR / BEAR_EXHAUSTED) was NOT
saved in the main split report text files. This file preserves screenshots the user
shared so the historical patterns are not lost.

Classifier logic lives in `services/regime.py`:
- BTC ADX < 18 → CHOPPY_WEAK
- BTC ADX ≥ 18 AND |slope| < 0.02% → CHOPPY_FLAT
- BTC ADX ≥ 28, slope up, RSI ≥ 70 → BULL_EXHAUSTED
- BTC ADX ≥ 28, slope up, RSI < 70 → STRONG_BULL
- BTC ADX 18-27, slope up → HEALTHY_BULL
- BTC ADX ≥ 28, slope down, RSI ≤ 30 → BEAR_EXHAUSTED
- BTC ADX ≥ 28, slope down, RSI > 30 → STRONG_BEAR
- BTC ADX 18-27, slope down → HEALTHY_BEAR

================================================================================
## Apr 13 report — SHORTS Performance by BTC Regime (from screenshot, 51 shorts)
================================================================================

Source: Apr 13 run, 51 shorts total (0 longs in this breakdown since this is shorts-only)

Regime          # Trades  Direction  Win Rate  Total P&L  Avg P&L$  Avg P&L%  Avg Peak%  Confidence
CHOPPY_WEAK     7         SHORT      43%       $-2.54     $-0.36    -0.14%   +0.29%     S:5 V:2
CHOPPY_FLAT     4         SHORT      25%       $-2.69     $-0.67    -0.30%   +0.28%     S:3 V:1
HEALTHY_BULL    1         SHORT      100%      $1.10      $1.10     +0.49%   +0.61%     S:1
HEALTHY_BEAR    18        SHORT      83%       $14.17     $0.79     +0.36%   +0.67%     S:14 V:4
STRONG_BEAR     20        SHORT      50%       $-10.52    $-0.53    -0.24%   +0.36%     V:11 S:9
BEAR_EXHAUSTED  1         SHORT      100%      $0.81      $0.81     +0.35%   +0.42%     V:1

================================================================================
## Apr 15 partial report — SHORTS Performance by BTC Regime (18 shorts)
================================================================================

Source: Apr 15 run (current), 18 shorts + 1 long (the long was STRONG_BULL)

Regime          # Trades  Direction  Win Rate  Total P&L  Avg P&L$  Avg P&L%  Avg Peak%  Confidence
STRONG_BULL     1         LONG       0%        $-0.70     $-0.70    -0.42%   +0.01%     S:1
HEALTHY_BEAR    9         SHORT      78%       $5.76      $0.64     +0.36%   +0.68%     S:7 V:2
STRONG_BEAR     7         SHORT      57%       $-0.19     $-0.03    -0.02%   +0.44%     S:4 V:3
BEAR_EXHAUSTED  2         SHORT      0%        $-1.07     $-0.53    -0.28%   +0.14%     S:1 V:1

================================================================================
## Combined Apr 13 + Apr 15 — SHORTS Performance by BTC Regime (69 shorts)
================================================================================

Regime          # Trades  Combined WR  Combined $  Avg %     Pattern
CHOPPY_WEAK     7         43%          -$2.54      -0.14%   Losing (Apr 13 only)
CHOPPY_FLAT     4         25%          -$2.69      -0.30%   Losing (Apr 13 only)
HEALTHY_BULL    1         100%         +$1.10      +0.49%   Tiny N (Apr 13 only)
HEALTHY_BEAR    27        81%          +$19.93     +0.36%   ★ BEST — 2-sample confirmed winning zone
STRONG_BEAR     27        52%          -$10.71     -0.13%   ★ LOSER — 2-sample confirmed, SAME sample size as HEALTHY_BEAR
BEAR_EXHAUSTED  3         33%          -$0.26      +0.04%   Small N, directionally negative

================================================================================
## Key insight

27 HEALTHY_BEAR trades won big. 27 STRONG_BEAR trades (same sample size) lost.
The difference is BTC ADX cutoff: HEALTHY_BEAR = ADX 18-27, STRONG_BEAR = ADX ≥28.

This is equivalent to (and cleaner than) the BTC RSI <30 × BTC ADX cross-tab
hypothesis. The regime classifier already encodes the "moderate vs strong
downtrend" distinction. Using regime labels directly would be simpler than
conditional cross-tab filters.

================================================================================
## Cross-reference — BTC ADX bucket view (same data, different slice)

Apr 13 shorts by BTC ADX:
- 20-25:  16 trades, 75% WR, +$9.31  (mostly HEALTHY_BEAR territory)
- 25-30:  6 trades, 83% WR, +$2.39   (borderline)
- 30-35:  8 trades, 38% WR, -$4.79   (STRONG_BEAR)
- 40+:    8 trades, 38% WR, -$7.44   (STRONG_BEAR / exhaustion)

Apr 15 shorts by BTC ADX:
- 20-25:  5 trades, 80% WR, +$4.07   (HEALTHY_BEAR zone)
- 25-30:  6 trades, 50% WR, -$0.55   (transitional)
- 30-35:  2 trades, 50% WR, +$0.37
- 35-40:  5 trades, 60% WR, +$0.62

The ADX 20-25 bucket wins in both samples, ≥30 loses (mostly). Matches the
HEALTHY_BEAR vs STRONG_BEAR regime story.
