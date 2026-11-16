# Theory-first filter audit — 2026-11-16
Operator: "the theory of the filters makes sense … theory first, then a re-check of all filters — you will find the rules are not fitted."
Question per filter: does it still separate on trades it NEVER saw? Read-only; nothing in code / config / CLAUDE changed.

## Method
- **Unseen data A — forward journal (Oct 14 → Nov 16, B22–B27):** every momentum / FAN signal the gate ladder refused is in the decisions journal (`FAILS` rows with the full fail list). A **sole-blocked** signal = this gate was the ONLY failing gate (removing it would have let the signal through the ladder). Deduped one per pair · gate · hour, random 60 per gate (seed 7), priced on its own 1m path. Every momentum / FAN gate here shipped May–Sep (before Oct 14) → this month is unseen by all of them (later threshold tweaks: caveat).
- **Same ruler both sides:** entry = next 1m open; TP +0.5 / SL −0.7 within 180 min (else close), −0.10 costs, adverse-first inside a bar. Kept = the real momentum / FAN fills of the same window on the same ruler (+ their booked %). It is not the live trail exit — it is a uniform yardstick for blocked vs kept (on kept fills it reads 0.09 pt more pessimistic than booked).
- **Unseen data B — pre-master backtest (2025-01 → 2026-06-16):** accuracy check first (real master fills reproduced by yr6, ±15 min, per seed): FRENZY_LITE recall 88 % · P&L corr 0.94 · LONG 70 % · 1.00 ✅; WILLY 93 % but corr 0.54 / bias −0.61 (old exit) ❌; MOMENTUM long 28 % · short 33 % · FAN 43 % · fade 24 % ❌ → backtest used ONLY for LITE / FRENZY_LONG filters (`reports/FILTER_THEORY_AUDIT_2026-11-16_gates.csv` + this file).
- **Unseen data C — DL 527+ blocked-signal pricer since 11-11** (`scripts/scout_blocked_filters.py`, live exits, 0.10 slip).
- Verdict: **REAL** = blocked worse than kept on unseen data, day-bootstrap P(blocked worse) ≥ 0.75 (≥ 0.9 strong) · **FITTED** = no separation or reversed (P ≤ 0.6) · **PENDING** = < 15 unseen blocked.

## Theory first (written before the numbers)
- **Momentum long:** buys the continuation of a fresh, expanding 5m uptrend while the market gives a tailwind; must not chase an exhausted / overextended move. Predicts: trend-definition gates (EMA gap / slope / ADX confidence), exhaustion gates (ADX max, RSI range, stretch, range position, gap max), tailwind gates (BTC slope / BTC ADX low). Data-found: BTC RSI×ADX cross, RSI-momentum-low-ADX, BTC ADX high, RSI-ceiling door.
- **Momentum short:** the mirror — sells a fresh expanding downtrend with a bearish tailwind; must not short an exhausted dump (RSI too low, range bottom). Crosses are data-found.
- **FAN flip:** a long refused for an over-spread EMA fan (an exhausted rally) is faded short; it should pay only when BTC is not rising and the coin is not in a strong uptrend regime (BTC 30m / 1h / trend-gap, regime, BTC-EMA13 — predicted); pair-ADX, quality, low-ATR — data-found.
- **FRENZY_LITE / LONG:** second leg of a pump on a defended base above the spike VWAP; needs a market that is not falling (BTC EMA20 down — predicted), a defended base (weak hold — predicted), not an overheated BTC (RSI hot — predicted); LONG_WEAK_FRENZY (low volume multiple) — data-found.
- **WILLY / HT:** buy the first red candle after a fresh volume spike; a failed spike (price back below VWAP) or a flat / over-extended EMA stack should fail (predicted).

## Momentum LONG — kept fills Oct 14 → Nov 16: 81 · WR 56 % · ruler −0.13 (CI −0.25 … +0.02) · booked −0.03
| Gate | Theory? | Sole-blocked unseen: N · WR · avg | P(blocked worse) | Verdict |
|---|---|---|---|---|
| PAIR_EMA20_FILTER | predicted (trend) | 60 · 33 % · −0.39 | 0.98 | **REAL** |
| PAIR_ADX_MAX | predicted (exhaustion) | 60 · 43 % · −0.26 | 0.93 | **REAL** |
| RSICEIL_DOOR_ADXMIN | data-found | 60 · 40 % · −0.29 | 0.93 | **REAL** |
| PAIR_RSI_MOMENTUM_LOADX | data-found (+theory) | 60 · 48 % · −0.20 | 0.82 | REAL (weak) |
| PAIR_EMA_GAP_5_20 | predicted | 60 · 48 % · −0.21 | 0.80 | REAL (weak) |
| PAIR_EMA_GAP_MAX | predicted | 60 · 48 % · −0.22 | 0.79 | REAL (weak) |
| PAIR_RSI_RANGE | predicted | 60 · 47 % · −0.21 | 0.77 | REAL (weak) |
| BTC_ADX_GATE_LOW | predicted (tailwind) | 60 · 52 % · −0.17 | 0.70 | FITTED (no sep.) |
| BTC_SLOPE_GATE | predicted (tailwind) | 60 · 52 % · −0.16 | 0.66 | FITTED |
| BTC_RSI_ADX_CROSS | data-found | 60 · 50 % · −0.17 | 0.62 | FITTED |
| BTC_ADX_GATE_HIGH | data-found | 60 · 53 % · −0.16 | 0.60 | FITTED |
| PAIR_EMA_GAP_MIN | predicted | 60 · 50 % · −0.12 | 0.43 | FITTED |
| PAIR_ADX_CONFIDENCE | predicted | 60 · 57 % · −0.10 | 0.40 | FITTED |
| PAIR_RANGE_POSITION_MAX | predicted | 60 · 58 % · −0.09 | 0.32 | FITTED (reversed) |
| EMA5_STRETCH | predicted | 60 · 63 % · −0.03 | 0.09 | FITTED (reversed) |
| PAIR_EMA20_SLOPE | predicted | 60 · 72 % · +0.07 | 0.07 | FITTED (reversed) |
| ML_MKT_HOT / BULL90 / WEAK_RESCUE / WEAK_MARKET / FADING_QUIET (DL 527–559) | mixed | ≤ 6 priced since ship | — | PENDING |

## Momentum SHORT — kept 14 · WR 64 % · ruler −0.03 (CI −0.32 … +0.31) · booked +0.03
| Gate | Theory? | Sole-blocked unseen | P(worse) | Verdict |
|---|---|---|---|---|
| PAIR_RSI_ADX_CROSS | data-found | 60 · 53 % · −0.15 | 0.74 | weak / borderline |
| PAIR_EMA_GAP_NOT_EXPANDING | predicted | 60 · 55 % · −0.13 | 0.74 | weak / borderline |
| PAIR_EMA20_SLOPE_MIN | predicted | 15 · 53 % · −0.11 | 0.65 | FITTED |
| EMA_GAP_MIN / 5_20 / MAX · PAIR_ADX_MAX · RSI_RANGE · BTC_RSI_ADX_CROSS | mixed | 60 each · 58–65 % · −0.07 … −0.02 | 0.50–0.62 | FITTED |
| BTC_ADX_GATE_LOW / HIGH · BTC_SLOPE · RANGE_POSITION_MIN · ADX_CONFIDENCE | mixed | 60 each · 68–77 % · +0.03 … +0.12 | 0.27–0.42 | FITTED (blocked BETTER than kept) |
| MS_REGIME_ENDS (DL 561) | partly | 0 since ship | — | PENDING |

## FAN flip — kept 22 · WR 73 % · ruler +0.07 (CI −0.24 … +0.30) · booked −0.04
| Gate | Theory? | Sole-blocked unseen | P(worse) | Verdict |
|---|---|---|---|---|
| FLIP_FAN_BTC_EMA13 | predicted | 9 · 56 % · −0.13 | 0.87 | PENDING (direction right) |
| FLIP_FAN_PAIR_ADX | data-found | 14 · 57 % · −0.11 | 0.87 | PENDING (direction right) |
| BTC_TRENDGAP · BTC1H_SLOPE · QUALITY | predicted / data | 5–8 · 60–71 % · −0.08 … +0.06 | 0.50–0.62 | PENDING (no sep. so far) |
| BTC30_RISE · RSI_MIN · WEAK_BOUNCE · STRETCH · ADXD · PAIR_GAP · LOATR | mixed | 1–4 each | — | PENDING |

## Frenzy — pre-master backtest (2025-01 → 2026-06-16; recall check passed for LITE / LONG) + forward pricer since 11-11
| Filter | Theory? | Backtest blocked vs kept, seeds 1/2/3 (avg %, P(worse)) | Forward since 11-11 | Verdict |
|---|---|---|---|---|
| LITE_BTC_EMA20_DOWN | predicted | −0.23/−0.23/−0.22 vs +0.09/+0.07/+0.09 · P 0.97/0.97/0.95 | 4 · +1.50 vs kept +1.50 | **REAL** |
| LITE_WEAK_HOLD | predicted | −0.13/−0.16/−0.12 vs +0.05/+0.05/+0.04 · P 0.84/0.87/0.81 | 2 · +3.00 (small) | **REAL (weak)** |
| LITE_BTC_RSI_HOT | predicted | +0.25/+0.03/+0.10 vs −0.06/−0.05/−0.04 · P 0.15/0.37/0.34 | 0 | **FITTED (reversed)** |
| LONG_WEAK_FRENZY | data-found | −0.07/−0.09/−0.10 vs −0.11/−0.13/−0.13 · P ≈ 0.45 | 2 · +1.50 vs −0.75 | **FITTED** |
| WILLY_GAP | predicted | backtest unusable (exit check failed) | 5 · 40 % · −2.09 vs kept +0.23 | PENDING (direction right) |
| WILLY_FAILED_SPIKE (WILLY / HT) | predicted | unusable | 6 · −1.56 / HT 11 · −0.91 vs kept +0.23 / +0.25 | PENDING (direction right, HT 11/15) |
| RETRY_BTC1H | predicted | not in the backtest | 3 · +3.00 (reversed) | PENDING |
| FADE_LATE_WEAK (fade) | data-found | unusable | 3 · +0.26 vs kept −0.52 (reversed) | PENDING |
| Flatbase higher-high (DL 540) | predicted | not in the backtest | none priced | PENDING |

## Per-sleeve answer: is there a positive edge WITH the real filters on unseen data?
- **Momentum long:** the REAL gates are already in the live ladder; the kept fills of the unseen month (= raw sleeve + every gate incl. the real ones) are 81 · 56 % · booked −0.03 / ruler −0.13 (CI −0.25 … +0.02). The FITTED gates block trades that are about as bad (−0.10 … −0.17), so removing them would not create an edge either. **No positive edge with or without.**
- **Momentum short:** kept 14 · booked +0.03 (CI spans ±0.3); most gates block signals no worse — several block BETTER ones. **No edge shown.**
- **FAN flip:** kept 22 · 73 % · booked −0.04; two gates point the right way on small N. **No edge shown.**
- **FRENZY_LITE:** two theory gates are REAL out of sample (EMA20 down strong, weak hold weak), one is not (RSI hot). The pre-master backtest LITE kept by the real gates is ≈ +0.05 … +0.09 %/fill — **small positive, not significant on its own**.
- **FRENZY_LONG:** its only filter is FITTED; the sleeve is ≈ −0.1 %/fill pre-master either way → no edge.

## Multiple testing
Momentum long carries ~16 ladder gates plus 5 watch lines, each picked from sweeps of ~2,600 tests (DL 398–403) and many single-gate studies; at 5 %, ~1 in 20 of the gates tested would look "real" by chance — here 7 of 16 pass, 3 strongly (P ≥ 0.93), more than chance, so several ARE real. Momentum short: 2 of 15 borderline — about what chance gives.

## Conclusion on the operator's hypothesis
**Partly right.** Theory-predicted filters are NOT all fitted: on data they never saw, PAIR_EMA20_FILTER, PAIR_ADX_MAX, RSICEIL (data-found), LITE_BTC_EMA20_DOWN and (weaker) LITE_WEAK_HOLD, RSI_RANGE, EMA_GAP_MAX / 5_20, LOADX keep separating. But **about half of the theory-predicted gates do not** (BTC tailwind gates, EMA5_STRETCH, EMA20_SLOPE, RANGE_POSITION_MAX, LITE_BTC_RSI_HOT), and having a theory did not predict which half — theory-predicted and data-found gates passed at similar rates. Most importantly, **with the real filters in place the momentum long / short / FAN sleeves still show no positive edge on the unseen month** — the filters are real but they sort bad from worse, not losers from winners. FRENZY_LITE is the one sleeve where real filters leave a small positive pre-master edge.

**Caveats:** one unseen month (Oct 14 → Nov 16, ~22 trading days, mostly a BTC uptrend); the ruler is a fixed TP / SL, not the live trail; sole-blocked signals may still have been refused later (heat / megacap / watch lines / slots); thresholds of some old gates were re-tuned after their first ship; Frenzy backtest results lean on the Oct-13 replay (passed the recall check for LITE / LONG only).
