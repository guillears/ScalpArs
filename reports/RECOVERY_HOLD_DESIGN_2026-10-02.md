# 🩹 RECOVERY HOLD — exit design on the live master batch (in-sample, 13 stops)

🩹 RECOVERY HOLD — full exit design for the operator's "we exited wrong" case (2026-10-02), read on the LIVE master batch only.
TRIGGER   a momentum LONG reaches its stop  ∧  BTC RSI(14, closed 5m) ≥ its value at entry  ∧  60 ≤ that RSI ≤ 66
          (band chosen AFTER seeing the master stops → every number here is in-sample; 13 qualifying stops)
ON TRIGGER the position is NOT closed; it enters recovery mode:
  hard stop     0.5 % below the original stop price (the most the hold can add to the loss)
  premise exit  at any 5m close where BTC RSI < 60 or < its entry value → close at market (the reason to hold is gone)
  time exit     30 min after the trigger, if the trade is still below entry → close at market
  V1 BREAKEVEN  close when price is back at the entry price
  V2 PREMISE    V1 + the premise exit
  V3 RESUME     hard stop + premise exit + time exit; no break-even close — once back above entry the normal runner logic resumes:
                arm at +0.40 %, then floor = max(peak − 1×ATR, +0.10 %); 240 min cap
1m futures bars from the minute after the stop; inside a bar the hard stop is checked first. Δ = result − the actual stop (gross; same single exit fee).

| Batch | Opened | Pair | BTC RSI entry→stop | Stop | V1 Δ (how) | V2 Δ (how) | V3 Δ (how) |
|---|---|---|---|---|---|---|---|
| B1 | 2026-07-17T16:52 | XLM | 64→66 | -0.60% | +0.12 (time 30m) | -0.02 (premise gone 4m) | -0.02 (premise gone 4m) |
| B1 | 2026-07-20T15:05 | NEAR | 56→60 | -0.66% | +0.66 (break-even 8m) | +0.66 (break-even 8m) | +0.99 (runner trail 50m) |
| B1 | 2026-07-22T15:43 | NEAR | 57→61 | -0.63% | +0.58 (time 30m) | +0.11 (premise gone 1m) | +0.11 (premise gone 1m) |
| B1 | 2026-07-22T15:43 | TAO | 57→63 | -0.63% | +0.63 (break-even 22m) | +0.01 (premise gone 3m) | +0.01 (premise gone 3m) |
| B1 | 2026-07-23T09:34 | UNI | 61→64 | -0.65% | -0.08 (time 30m) | -0.08 (premise gone 3m) | -0.08 (premise gone 3m) |
| B1 | 2026-07-27T17:18 | UNI | 57→61 | -0.62% | -0.13 (time 30m) | +0.03 (premise gone 1m) | +0.03 (premise gone 1m) |
| B1 | 2026-07-29T00:55 | LIT | 61→61 | -0.76% | -0.50 (hard stop 17m) | +0.16 (premise gone 5m) | +0.16 (premise gone 5m) |
| B2 | 2026-08-05T14:00 | KAITO | 59→61 | -0.81% | -0.50 (hard stop 2m) | -0.50 (hard stop 2m) | -0.50 (hard stop 2m) |
| B3 | 2026-08-12T20:48 | AVAX | 61→63 | -0.61% | -0.50 (hard stop 21m) | +0.00 (premise gone 4m) | +0.00 (premise gone 4m) |
| B3 | 2026-08-23T09:02 | PENGU | 61→61 | -0.72% | +0.72 (break-even 18m) | +0.16 (premise gone 1m) | +0.16 (premise gone 1m) |
| B9 | 2026-09-18T19:14 | 1000PEPE | 58→61 | -0.60% | -0.14 (time 30m) | -0.06 (premise gone 3m) | -0.06 (premise gone 3m) |
| B14 | 2026-09-29T10:05 | 0G | 64→64 | -1.13% | -0.49 (hard stop 1m) | -0.49 (hard stop 1m) | -0.49 (hard stop 1m) |
| B16 | 2026-10-02T01:57 | ZRO | 60→64 | -1.04% | +1.04 (break-even 5m) | +1.04 (break-even 5m) | +1.44 (runner trail 13m) |

| Batch | qualifying stops | of all ML stops in the batch | V1 Δ % / $ | V2 Δ % / $ | V3 Δ % / $ |
|---|---|---|---|---|---|
| B1 | 7 | 85 | +1.28 / +4 | +0.87 / +2 | +1.20 / +3 |
| B2 | 1 | 14 | -0.50 / -2 | -0.50 / -2 | -0.50 / -2 |
| B3 | 2 | 6 | +0.22 / -37 | +0.16 / +25 | +0.16 / +25 |
| B9 | 1 | 2 | -0.14 / -19 | -0.06 / -9 | -0.06 / -9 |
| B14 | 1 | 5 | -0.49 / -57 | -0.49 / -57 | -0.49 / -57 |
| B16 | 1 | 2 | +1.04 / +296 | +1.04 / +296 | +1.44 / +411 |
| **TOTAL** | 13 | 135 | **+1.41 / +185** | **+1.02 / +256** | **+1.75 / +372** |

Better than the stop: V1 6 of 13 (worse 7) · V2 7 of 13 (worse 5) · V3 7 of 13 (worse 5)
