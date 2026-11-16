# FAN flip exit study: today's trail vs fixed TP (2026-11-16)
Operator questions: "The fan flip exit strategy is the best we could have? or better a fix TP?" and "check master batch too".

## Today's live FAN flip exit (code and config)
- **Stop:** −0.70 %, widened to 1.5 × ATR, at most −1.20 %.
- **Runner trail:** arms once the trade has been up +0.40 %. It then gives back the smaller of 0.5 × ATR and 35 % of the best gain. There is no breakeven lock; a negative floor is ridden.
- **Short HARD_TP ladder:** "1.0:0.25 … 4.0:0.80".
- **Time:** shared NO_EXPANSION at 180 min and MAX_HOLD.
- **Code path:** flip_exit_mode "strpk" → `REALTIME_RUNNER_TRAIL` (services/trading_engine.py ~25950), booked as FLIP_RUNNER_TRAIL.
- **Live examples:** PEPE armed at +0.40 and trailed to +0.23. INIT peaked +0.71, then fell through its floor within a minute and booked −0.10.

## Data and method
- **Fills:** master MASTER_POOL_stacked FLIP:FAN_RATIO_GATE, stack_keep, full-size: 54 (BASE 20 · Aug–Oct 15 · B25 13 · B26 6), plus live B27: 2. Total 56 fills over 32 days.
- **Walker:** scripts/flip_exit_replica.py logic, validated before on ticks, fed with Binance 1m bars. For a short each bar is read open → high → low → close, so a stop wins any tie. The first bar starts at the entry price.
  - Agreement with booked results: mean |Δ| 0.17 pp, 80 % within 0.25. 1m bars run about 0.05 pp more conservative than ticks.
  - **Every variant is compared with the same walker's "today"**, never with booked results.
- **Fixed TP:** books exactly at the line, keeps today's stop, has no trail, and closes at 180 min (NO_EXPANSION); 12 h shown as a sensitivity check.
- **Hybrid:** today's trail plus a cap at +X.
- **Statistics:** one-size DCR at X = 2.05 (live FAN notional ÷ account); day-block bootstrap 95 % CI on Δ per fill; LOO = drop the best fill.

## Decomposition (booked)
| Path | N | avg % | Σ % |
|---|---|---|---|
| CAPTURED / other | 41 | +0.39 | +15.8 |
| NEVER_WORKED (peak < +0.10) | 9 | −0.94 | **−8.5** |
| LEFT_ON_TABLE (≥ 0.5 given back) | 4 | +0.20 | +0.8 |
| GAVE_BACK (armed, lost) | 2 | −0.11 | −0.2 |

- **The loss side is the ENTRY.** The 9 never-worked stops cost more than any exit variant gains.
- **Two-sided check:** the 11 fills stopped under today's exit book the same −1.04 average under every TP variant (Δ = 0).

## Results (avg % per fill · Δ vs today · day-boot 95 % CI · DCR %/day)
| Exit | ALL 56 (32 d) | Aug → live 36 (22 d) | B25 + B26 + live 21 (11 d) |
|---|---|---|---|
| Today (trail) | +0.066 · DCR +0.05 | −0.044 · DCR −0.04 | −0.039 · DCR −0.08 |
| TP +0.3 | −0.03 (−0.09 … +0.02) | −0.02 | +0.06 (+0.02 … +0.13) |
| TP +0.5 | +0.09 (−0.01 … +0.16) · DCR +0.11 | +0.05 (−0.10 … +0.16) | +0.19 (+0.13 … +0.29) · DCR +0.25 |
| TP +0.7 | +0.17 (+0.03 … +0.30) · DCR +0.17 | +0.08 (−0.12 … +0.26) | +0.25 (+0.02 … +0.44) · DCR +0.35 |
| TP +1.0 | +0.18 (−0.04 … +0.38) | +0.04 (−0.22 … +0.28) | +0.06 (−0.36 … +0.42) |
| TP +2.0 | +0.28 (−0.11 … +0.65) | +0.10 (−0.28 … +0.58) | +0.23 (−0.29 … +1.01) |
| TP +3.0 | +0.30 (−0.15 … +0.78) | −0.16 | −0.24 |
| Trail + cap +0.7 | +0.08 (+0.03 … +0.14) | +0.04 (−0.01 … +0.11) | +0.07 (0.00 … +0.19) |
| Trail + cap +1.0 | +0.04 (+0.01 … +0.08) | +0.02 | 0.00 |

**Per era group (avg %: today → TP +0.5 · TP +1.0 · TP +2.0):**
| Era | N | Today | TP +0.5 | TP +1.0 | TP +2.0 |
|---|---|---|---|---|---|
| Jun–Jul (BASE) | 20 | +0.27 | +0.41 | +0.69 | +0.87 |
| Aug–Oct | 15 | −0.05 | **−0.18** | −0.04 | −0.12 |
| B25 | 13 | −0.19 | −0.04 | −0.20 | +0.18 |
| B26 | 6 | +0.28 | +0.42 | +0.43 | 0.00 |
| Live B27 | 2 | +0.01 | +0.50 | +0.27 | +0.77 |

## Backtest cross-check (yr6, 2 years, secondary)
- Approximation: TP x = x when the recorded peak ≥ x, else as booked (upper bound).
- The sleeve is negative on today's exit (−0.10 … −0.15 per fill). Fixed TP moves it only +0.00 … +0.04 (TP +0.5: +0.023 / +0.029 / +0.007; TP +1.0 tie or worse in seed 3). The sign does not change.

## Verdict
- **Today's trail is not clearly the best.** A low fixed TP (+0.5 / +0.7) beats it on the master overall (+0.09 / +0.17 per fill) and clearly on B25 → live (+0.19 / +0.25). INIT live would have booked +0.50 instead of −0.10, and PEPE +0.50 instead of +0.23.
- **It is not a ship under the locked rules:**
  1. The Aug–Oct era goes the other way (TP +0.5 −0.18 vs −0.05).
  2. Aug → live the CI crosses 0.
  3. The best cut was picked from 16 variants.
  4. It is in-sample: after the 30–50 % haircut, TP +0.5 is about +0.05 … +0.09 per fill overall.
  5. The 2-year backtest gives only +0.01 … +0.03.
- **High TPs (+2 / +3)** live on the Jun–Jul washed-out window (the same finding as DL 411 / 431) and are already tracked forward in EXIT_SHADOWS (TP1.5 / 2 / 3 / TP2_SL3).
- **Recommendation:** keep today's exit. Add TP +0.5 and TP +0.7 (180 min, today's stop) as forward observe lines next to the existing EXIT_SHADOWS flip lines, under the same pre-committed bar (≥ 20 flips on ≥ 8 days, Δ ≥ +0.10 per fill, day-boot P ≥ 1 − 0.10/m). The bigger lever is the entry: 9 never-worked stops cost −8.5 pts, more than any exit variant gains.
