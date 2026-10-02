# 📉 OFF-PEAK SHORT — short the break only when the entry is 15–40 % below the run's peak (frozen before reading)

% of position at 1×, 1-minute bars, gap-aware stop + 0.10 % slippage, after 0.11 % and funding. One position per pair.

## PRIMARY · EMA50, run +50–200 % (never seen by the cut) — 1561 of 3511 triggers are in the band (242 pairs, 244 days)

| Exit | trades | won | stopped | per trade | Jan–Apr / May–Sep | 95 % by day | 95 % by pair | worst with one month removed | top 3 pairs' share | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop 2 · trail 5/3 | 1520 | 31% | 69% | -0.16 | +0.08 / -0.35 | [-0.34, +0.02] | [-0.34, +0.03] | -0.24 (−2026-04) | n/a (total ≤ 0) | 18 | — |
| stop 2 · trail 3/2 | 1536 | 41% | 59% | -0.17 | -0.04 / -0.26 | [-0.32, -0.02] | [-0.31, -0.03] | -0.23 (−2026-04) | n/a (total ≤ 0) | 13 | — |
| stop 3 · trail 5/3 | 1485 | 41% | 59% | -0.07 | -0.09 / -0.06 | [-0.28, +0.14] | [-0.28, +0.15] | -0.13 (−2026-09) | n/a (total ≤ 0) | 11 | — |
| stop 3 · trail 3/2 | 1509 | 51% | 48% | -0.13 | -0.22 / -0.07 | [-0.30, +0.05] | [-0.30, +0.04] | -0.20 (−2026-09) | n/a (total ≤ 0) | 11 | — |

**Dose — every band of distance below the peak, proposed exit (trades · per trade · halves)**

| Entry vs peak | result |
|---|---|
| -100 to -40 % | 301 · -0.02 (-0.13 / +0.07) · 77 pairs |
| -40 to -25 % | 602 · -0.10 (+0.23 / -0.38) · 180 pairs |
| -25 to -15 % | 919 · -0.20 (-0.02 / -0.33) · 216 pairs |
| -15 to -8 % | 1134 · -0.38 (-0.38 / -0.38) · 225 pairs |
| -8 to 0.01 % | 496 · -0.21 (-0.08 / -0.30) · 127 pairs |

## SECONDARY · EMA200, all runs (never seen by the cut) — 1491 of 2274 triggers are in the band (246 pairs, 249 days)

| Exit | trades | won | stopped | per trade | Jan–Apr / May–Sep | 95 % by day | 95 % by pair | worst with one month removed | top 3 pairs' share | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop 2 · trail 5/3 | 1483 | 30% | 70% | -0.14 | +0.05 / -0.27 | [-0.33, +0.05] | [-0.32, +0.05] | -0.18 (−2026-02) | n/a (total ≤ 0) | 17 | — |
| stop 2 · trail 3/2 | 1486 | 40% | 60% | -0.25 | -0.12 / -0.34 | [-0.40, -0.10] | [-0.39, -0.11] | -0.28 (−2026-02) | n/a (total ≤ 0) | 12 | — |
| stop 3 · trail 5/3 | 1463 | 39% | 60% | -0.09 | +0.14 / -0.24 | [-0.32, +0.15] | [-0.31, +0.16] | -0.17 (−2026-02) | n/a (total ≤ 0) | 11 | — |
| stop 3 · trail 3/2 | 1468 | 50% | 49% | -0.23 | -0.08 / -0.32 | [-0.39, -0.05] | [-0.39, -0.05] | -0.28 (−2026-02) | n/a (total ≤ 0) | 9 | — |

**Dose — every band of distance below the peak, proposed exit (trades · per trade · halves)**

| Entry vs peak | result |
|---|---|
| -100 to -40 % | 236 · -0.39 (-0.36 / -0.42) · 68 pairs |
| -40 to -25 % | 564 · -0.03 (+0.08 / -0.12) · 163 pairs |
| -25 to -15 % | 919 · -0.20 (+0.03 / -0.35) · 211 pairs |
| -15 to -8 % | 492 · -0.57 (-0.46 / -0.67) · 138 pairs |
| -8 to 0.01 % | 50 · -0.45 (+0.34 / -0.83) · 22 pairs |

## SECONDARY · EMA200, run > +200 % — 156 of 283 triggers are in the band (46 pairs, 75 days)

| Exit | trades | won | stopped | per trade | Jan–Apr / May–Sep | 95 % by day | 95 % by pair | worst with one month removed | top 3 pairs' share | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop 2 · trail 5/3 | 156 | 27% | 72% | -0.00 | +0.99 / -0.44 | [-0.69, +0.77] | [-0.60, +0.70] | -0.22 (−2026-06) | n/a (total ≤ 0) | 25 | — |
| stop 2 · trail 3/2 | 156 | 38% | 61% | -0.17 | +0.60 / -0.52 | [-0.71, +0.43] | [-0.58, +0.27] | -0.36 (−2026-04) | n/a (total ≤ 0) | 15 | — |
| stop 3 · trail 5/3 | 154 | 34% | 66% | -0.07 | +1.43 / -0.76 | [-0.88, +0.81] | [-0.81, +0.77] | -0.34 (−2026-02) | n/a (total ≤ 0) | 15 | — |
| stop 3 · trail 3/2 | 154 | 45% | 53% | -0.29 | +0.71 / -0.73 | [-0.88, +0.37] | [-0.74, +0.23] | -0.43 (−2026-04) | n/a (total ≤ 0) | 7 | — |

**Dose — every band of distance below the peak, proposed exit (trades · per trade · halves)**

| Entry vs peak | result |
|---|---|
| -100 to -40 % | 105 · -0.13 (-0.26 / +0.05) · 28 pairs |
| -40 to -25 % | 66 · -0.01 (+2.08 / -0.52) · 30 pairs |
| -25 to -15 % | 90 · +0.00 (+0.59 / -0.37) · 30 pairs |
| -15 to -8 % | 22 · -0.59 (+0.66 / -1.06) · 9 pairs |
| -8 to 0.01 % | 0 · – |

## REFERENCE (in-sample) · EMA50, run > +200 % — 201 of 499 triggers are in the band (52 pairs, 97 days)

| Exit | trades | won | stopped | per trade | Jan–Apr / May–Sep | 95 % by day | 95 % by pair | worst with one month removed | top 3 pairs' share | longest losing run | PASS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stop 2 · trail 5/3 | 201 | 36% | 64% | +0.58 | +0.52 / +0.62 | [-0.00, +1.17] | [+0.08, +1.16] | +0.40 (−2026-08) | 53% | 12 | — |
| stop 2 · trail 3/2 | 201 | 46% | 53% | +0.27 | +0.12 / +0.36 | [-0.14, +0.68] | [-0.08, +0.63] | +0.08 (−2026-08) | 76% | 6 | — |
| stop 3 · trail 5/3 | 201 | 45% | 55% | +0.65 | +0.61 / +0.68 | [-0.03, +1.33] | [-0.00, +1.34] | +0.45 (−2026-08) | 54% | 12 | — |
| stop 3 · trail 3/2 | 201 | 57% | 42% | +0.41 | +0.40 / +0.42 | [-0.05, +0.94] | [-0.03, +0.88] | +0.25 (−2026-08) | 60% | 4 | — |

**Dose — every band of distance below the peak, proposed exit (trades · per trade · halves)**

| Entry vs peak | result |
|---|---|
| -100 to -40 % | 189 · +0.20 (+0.22 / +0.17) · 29 pairs |
| -40 to -25 % | 81 · +0.63 (+0.53 / +0.72) · 33 pairs |
| -25 to -15 % | 120 · +0.55 (+0.51 / +0.57) · 37 pairs |
| -15 to -8 % | 94 · -0.61 (-0.43 / -0.72) · 32 pairs |
| -8 to 0.01 % | 14 · – |

## Verdict (proposed exit: stop 2 %, trail from +5 % giving back 3 %)

- PRIMARY · EMA50, run +50–200 %: 1520 trades · -0.16 per trade (+0.08 / -0.35) · by day [-0.34, +0.02] → no pass
- SECONDARY · EMA200, all runs: 1483 trades · -0.14 per trade (+0.05 / -0.27) · by day [-0.33, +0.05] → no pass
- SECONDARY · EMA200, run > +200 %: 156 trades · -0.00 per trade (+0.99 / -0.44) · by day [-0.69, +0.77] → no pass
- REFERENCE: 201 trades · +0.58 per trade (+0.52 / +0.62) · by day [-0.00, +1.17] → no pass

## NOT tested

- No fresh time period exists: the year is fully used, so 'unseen' means other run sizes and the other line, not later dates.
- Delisted pairs, exchange position limits on small pairs, slippage beyond 0.10 %.
- Triggers on one pair in one episode are not independent; the by-pair interval is the stricter read.
