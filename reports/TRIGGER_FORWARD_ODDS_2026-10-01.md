# Do the triggers raise the odds of a big move? (exit-free, 5m bars, day units)

Gross of fees. `target share of resolved` = among events that hit EITHER +5 % (target) or −3 % (stop) first, the share that hit the target first (volatility-neutral; same for +10 / −5; 5m bars, stop-first on a shared bar). Baseline = the same universe sampled every 6 h on every pair with ≥ 30 days of history, no trigger.

| Cohort | events | days | 4 h return % [95 %] | 24 h return % [95 %] | +5 vs −3: target share of resolved | +10 vs −5: target share of resolved | median best / worst 24 h |
|---|---|---|---|---|---|---|---|
| BASELINE TOP50 LONG | 46417 | 242 | -0.05 [-0.15,+0.05] | -0.29 [-0.64,+0.06] | 35% | 28% | +3.6 / -4.0 |
| BASELINE TOP50 SHORT | 46417 | 242 | +0.05 [-0.05,+0.15] | +0.29 [-0.06,+0.64] | 35% | 24% | +4.0 / -3.6 |
| BASELINE NEXT50 LONG | 46786 | 242 | +0.02 [-0.06,+0.10] | -0.01 [-0.30,+0.28] | 35% | 27% | +3.4 / -3.6 |
| BASELINE NEXT50 SHORT | 46786 | 242 | -0.02 [-0.10,+0.06] | +0.01 [-0.28,+0.30] | 35% | 20% | +3.6 / -3.4 |
| RUNAWAY LONG 8.0 | 2173 | 241 | -0.55 [-1.26,+0.15] | -1.86 [-3.23,-0.50] | 38% | 35% | +10.9 / -12.0 |
| RUNAWAY LONG 12.0 | 1548 | 237 | -0.89 [-1.75,-0.03] | -2.70 [-4.43,-0.97] | 38% | 35% | +13.0 / -14.4 |
| RUNAWAY SHORT 8.0 | 822 | 227 | +0.13 [-1.17,+1.42] | +0.24 [-1.94,+2.42] | 39% | 36% | +11.8 / -11.3 |
| RUNAWAY SHORT 12.0 | 527 | 209 | -0.68 [-2.75,+1.39] | -2.24 [-6.47,+1.99] | 38% | 36% | +14.5 / -14.2 |
| LIFTOFF LONG EMA200_LIFT·NEXT50 | 4798 | 240 | -0.23 [-0.41,-0.05] | -0.51 [-0.95,-0.07] | 36% | 30% | +3.6 / -4.0 |
| LIFTOFF LONG EMA200_LIFT·TOP50 | 3208 | 238 | -0.40 [-0.64,-0.16] | -0.47 [-1.11,+0.17] | 39% | 33% | +3.5 / -3.2 |
| LIFTOFF LONG VOL_LEAD·NEXT50 | 2567 | 241 | -0.64 [-0.98,-0.30] | -0.79 [-1.60,+0.03] | 36% | 30% | +6.4 / -7.5 |
| LIFTOFF LONG VOL_LEAD·TOP50 | 1304 | 238 | -0.49 [-1.25,+0.26] | -2.39 [-3.85,-0.92] | 34% | 29% | +7.1 / -9.6 |
| LIFTOFF SHORT EMA200_LIFT·NEXT50 | 4380 | 241 | -0.43 [-0.59,-0.26] | -0.85 [-1.26,-0.43] | 33% | 17% | +3.2 / -3.4 |
| LIFTOFF SHORT EMA200_LIFT·TOP50 | 3363 | 237 | -0.37 [-0.58,-0.15] | -0.45 [-0.98,+0.07] | 34% | 21% | +3.1 / -2.9 |
| LIFTOFF SHORT VOL_LEAD·NEXT50 | 737 | 219 | -0.26 [-1.02,+0.51] | -0.38 [-1.61,+0.84] | 37% | 23% | +4.3 / -4.2 |
| LIFTOFF SHORT VOL_LEAD·TOP50 | 557 | 192 | -0.02 [-1.07,+1.04] | +0.54 [-1.09,+2.17] | 38% | 29% | +4.9 / -4.1 |
| COMBO VOL_LEAD ∧ EMA200_LIFT (≤2 h) TOP50 LONG | 224 | 120 | -1.07 [-2.10,-0.03] | -1.36 [-4.27,+1.55] | 36% | 27% | +3.5 / -3.5 |
| COMBO VOL_LEAD ∧ EMA200_LIFT (≤2 h) TOP50 SHORT | 207 | 81 | +0.22 [-0.72,+1.16] | +1.01 [-0.73,+2.74] | 38% | 25% | +2.7 / -2.4 |
| COMBO VOL_LEAD ∧ EMA200_LIFT (≤2 h) NEXT50 LONG | 645 | 218 | -0.32 [-1.15,+0.51] | -0.10 [-1.54,+1.34] | 41% | 33% | +5.6 / -5.6 |
| COMBO VOL_LEAD ∧ EMA200_LIFT (≤2 h) NEXT50 SHORT | 302 | 124 | -0.55 [-1.33,+0.22] | -0.46 [-1.97,+1.04] | 34% | 23% | +3.0 / -3.2 |
