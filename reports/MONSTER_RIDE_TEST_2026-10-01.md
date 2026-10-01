# 🐉 Ride-the-monster test — small, wide-stop, multi-day longs on runaway / volume-surge triggers

Unlevered %, after 0.19 % costs, funding ignored. One position per pair at a time, ≥ 30 days of pair history. Only pairs still listed today (flatters longs). Ranges are 95 % t-intervals on ENTRY-WEEK means (holds are 3–7 days, so entry days overlap). Bad-tick guard: 5 bars in the whole cache had a high > 1.5× the body top and were clipped there. The four exits are not run on identical trades (the one-per-pair rule and the data-end cut depend on the exit).

## Exit A: stop −20 %, hold 7 d

| Cohort | trades | weeks | avg per trade % | week mean % [95 %] | winners | ended by initial stop / trail / TP / time % | reached +50 % / +100 % | total % (concentration) |
|---|---|---|---|---|---|---|---|---|
| RUNAWAY LONG +8% | 1213 | 34 | -3.49 | -3.54 [-6.21, -0.87] | 23% | 55 / 0 / 0 / 45 | 17.6% / 8.2% | -4228 (best 5: +1818; without them -6046) |
| RUNAWAY LONG +12% | 940 | 34 | -4.92 | -4.66 [-7.66, -1.66] | 21% | 65 / 0 / 0 / 35 | 20.5% / 9.7% | -4628 (best 5: +1641; without them -6270) |
| VOLUME LEADS PRICE LONG TOP50 | 1006 | 34 | -3.45 | -3.92 [-6.65, -1.18] | 26% | 45 / 0 / 0 / 55 | 15.3% / 6.2% | -3472 (best 5: +1222; without them -4694) |
| VOLUME LEADS PRICE LONG NEXT50 | 2049 | 34 | -1.48 | -1.88 [-4.04, +0.29] | 31% | 30 / 0 / 0 / 70 | 12.7% / 4.8% | -3042 (best 5: +2173; without them -5215) |
| BASELINE TOP50 (every 6 h) | 3652 | 35 | -1.70 | -1.38 [-3.43, +0.66] | 33% | 34 / 0 / 0 / 66 | 10.5% / 4.1% | -6191 (best 5: +2389; without them -8580) |
| BASELINE NEXT50 (every 6 h) | 4866 | 35 | -0.61 | -0.02 [-2.25, +2.20] | 38% | 23 / 0 / 0 / 77 | 8.9% / 3.0% | -2988 (best 5: +1613; without them -4602) |

## Exit B: stop −20 %, 25 % trail after +30 %, hold 7 d

| Cohort | trades | weeks | avg per trade % | week mean % [95 %] | winners | ended by initial stop / trail / TP / time % | reached +50 % / +100 % | total % (concentration) |
|---|---|---|---|---|---|---|---|---|
| RUNAWAY LONG +8% | 1379 | 34 | -0.41 | -0.91 [-3.02, +1.20] | 36% | 45 / 27 / 0 / 27 | 16.1% / 5.2% | -571 (best 5: +1675; without them -2247) |
| RUNAWAY LONG +12% | 1084 | 34 | -1.12 | -1.26 [-3.48, +0.97] | 35% | 52 / 31 / 0 / 17 | 17.8% / 6.2% | -1215 (best 5: +1226; without them -2441) |
| VOLUME LEADS PRICE LONG TOP50 | 1049 | 34 | -1.09 | -1.26 [-3.53, +1.00] | 35% | 37 / 21 / 0 / 42 | 14.0% / 3.9% | -1138 (best 5: +1106; without them -2245) |
| VOLUME LEADS PRICE LONG NEXT50 | 2084 | 34 | -0.35 | -0.84 [-2.34, +0.66] | 37% | 26 / 17 / 0 / 56 | 11.6% / 3.3% | -738 (best 5: +1265; without them -2003) |
| BASELINE TOP50 (every 6 h) | 4427 | 35 | -0.95 | -0.98 [-2.70, +0.75] | 37% | 34 / 18 / 0 / 49 | 11.0% / 3.3% | -4188 (best 5: +1765; without them -5953) |
| BASELINE NEXT50 (every 6 h) | 5117 | 35 | -0.12 | +0.24 [-1.73, +2.21] | 41% | 22 / 12 / 0 / 67 | 8.3% / 2.2% | -589 (best 5: +1553; without them -2142) |

## Exit C: stop −10 %, 20 % trail after +20 %, hold 3 d

| Cohort | trades | weeks | avg per trade % | week mean % [95 %] | winners | ended by initial stop / trail / TP / time % | reached +50 % / +100 % | total % (concentration) |
|---|---|---|---|---|---|---|---|---|
| RUNAWAY LONG +8% | 1740 | 35 | -0.40 | -0.54 [-1.67, +0.60] | 29% | 62 / 24 / 0 / 14 | 8.4% / 2.0% | -693 (best 5: +871; without them -1563) |
| RUNAWAY LONG +12% | 1338 | 35 | -0.37 | -0.39 [-1.55, +0.76] | 28% | 64 / 28 / 0 / 7 | 9.3% / 2.2% | -496 (best 5: +843; without them -1339) |
| VOLUME LEADS PRICE LONG TOP50 | 1203 | 35 | -0.84 | -0.99 [-2.16, +0.19] | 29% | 55 / 19 / 0 / 26 | 7.1% / 1.6% | -1013 (best 5: +814; without them -1827) |
| VOLUME LEADS PRICE LONG NEXT50 | 2363 | 35 | -0.85 | -0.99 [-1.71, -0.28] | 31% | 50 / 17 / 0 / 33 | 6.1% / 1.2% | -2010 (best 5: +864; without them -2874) |
| BASELINE TOP50 (every 6 h) | 8500 | 35 | -0.77 | -0.77 [-1.57, +0.03] | 34% | 46 / 14 / 0 / 40 | 5.0% / 1.0% | -6543 (best 5: +1166; without them -7709) |
| BASELINE NEXT50 (every 6 h) | 8857 | 35 | -0.46 | -0.45 [-1.24, +0.34] | 38% | 36 / 8 / 0 / 56 | 3.1% / 0.6% | -4044 (best 5: +1290; without them -5334) |

## Exit D: stop −30 %, TP +100 %, hold 7 d

| Cohort | trades | weeks | avg per trade % | week mean % [95 %] | winners | ended by initial stop / trail / TP / time % | reached +50 % / +100 % | total % (concentration) |
|---|---|---|---|---|---|---|---|---|
| RUNAWAY LONG +8% | 1214 | 34 | -1.37 | -1.57 [-3.99, +0.86] | 29% | 31 / 0 / 10 / 60 | 21.0% / 9.6% | -1658 (best 5: +499; without them -2157) |
| RUNAWAY LONG +12% | 958 | 34 | -1.53 | -1.26 [-4.33, +1.82] | 28% | 37 / 0 / 12 / 51 | 24.6% / 11.7% | -1469 (best 5: +499; without them -1968) |
| VOLUME LEADS PRICE LONG TOP50 | 1000 | 34 | -2.24 | -2.93 [-5.95, +0.08] | 30% | 25 / 0 / 7 / 68 | 17.5% / 7.1% | -2243 (best 5: +499; without them -2742) |
| VOLUME LEADS PRICE LONG NEXT50 | 2012 | 34 | -0.23 | -0.85 [-2.97, +1.27] | 34% | 11 / 0 / 5 / 84 | 13.2% / 5.0% | -458 (best 5: +499; without them -957) |
| BASELINE TOP50 (every 6 h) | 3508 | 35 | -0.84 | -0.94 [-3.10, +1.22] | 35% | 18 / 0 / 5 / 77 | 12.7% / 5.3% | -2947 (best 5: +499; without them -3446) |
| BASELINE NEXT50 (every 6 h) | 4601 | 35 | +0.23 | +0.93 [-1.77, +3.62] | 40% | 8 / 0 / 3 / 89 | 9.3% / 3.2% | +1076 (best 5: +499; without them +577) |

## Reading (computed from the tables above)

- 16 of 16 trigger × exit cells have a negative average per trade; in 4 of them the week range excludes zero. No cell has a range above zero.
- 3 of 16 trigger cells have a better per-trade average than the no-trigger baseline of their universe under the same exit (runaway triggers are compared with TOP50). The baselines lose too: a large part of the loss belongs to buying this universe with these exits, not to the trigger.
- The trigger raises the share of trades that reach +50 % / +100 % versus the baseline (see the column), and the average still does not turn positive: the initial stop ends more trades than the monsters pay for.

## Do the monsters give it back? Exit A (no trail), trades whose best point was ≥ +100 %

| Cohort | such trades | median final % | mean final % | finished below +50 % | finished below 0 |
|---|---|---|---|---|---|
| RUNAWAY LONG +8% | 100 | +56.5 | +73.4 | 48% | 23% |
| RUNAWAY LONG +12% | 91 | +49.9 | +66.9 | 51% | 24% |
| VOLUME LEADS PRICE LONG TOP50 | 62 | +57.2 | +73.2 | 48% | 23% |
| VOLUME LEADS PRICE LONG NEXT50 | 98 | +56.3 | +79.0 | 43% | 15% |

## Events not traded (exit A)

| Cohort | events | no data / < 30 d history | pair already in a position | data ends before the hold |
|---|---|---|---|---|
| RUNAWAY LONG +8% | 2188 | 0 | 914 | 61 |
| RUNAWAY LONG +12% | 1557 | 0 | 579 | 38 |
| VOLUME LEADS PRICE LONG TOP50 | 1307 | 0 | 273 | 28 |
| VOLUME LEADS PRICE LONG NEXT50 | 2579 | 0 | 463 | 67 |
| BASELINE TOP50 (every 6 h) | 48450 | 1841 | 42119 | 838 |
| BASELINE NEXT50 (every 6 h) | 48450 | 1468 | 41187 | 929 |

## Verdict and limits

- No sleeve. No evidence of an edge in any of the 16 cells; most ranges span zero, so this is "no edge found", not "proven loser".
- The no-trigger baselines are negative too: Jan–Sep 2026 was a poor period for holding these alts for days — one 8-month span.
- Survivorship (delisted pumps missing) flatters longs, so the true result is worse, not better.
- Stops and trails fill at their level inside a 5m bar (optimistic in fast bars, e.g. the TUT exit); funding is ignored.
- Not tested: delisted pairs, sub-minute entries, other periods.

## Runaway +12 %, exit B — the 10 best and 10 worst trades

| date | pair | result % | best point % |
|---|---|---|---|
| 2026-08-07 | TUTUSDT | +416.8 | +589.3 |
| 2026-09-11 | LSKUSDT | +265.4 | +387.5 |
| 2026-08-03 | SKYAIUSDT | +218.3 | +324.6 |
| 2026-03-29 | STOUSDT | +181.3 | +275.3 |
| 2026-08-31 | USELESSUSDT | +144.3 | +226.0 |
| 2026-09-18 | AKEUSDT | +142.1 | +223.1 |
| 2026-02-24 | POWERUSDT | +132.9 | +226.6 |
| 2026-04-15 | ORDIUSDT | +128.3 | +204.7 |
| 2026-05-19 | EDENUSDT | +120.0 | +193.7 |
| 2026-07-22 | BANKUSDT | +117.2 | +210.3 |
| 2026-06-02 | SKYAIUSDT | -20.2 | +23.7 |
| 2026-05-30 | UBUSDT | -20.2 | +3.1 |
| 2026-05-30 | SKYAIUSDT | -20.2 | +3.8 |
| 2026-05-31 | HIVEUSDT | -20.2 | +0.8 |
| 2026-05-31 | AIAUSDT | -20.2 | +6.5 |
| 2026-05-31 | UBUSDT | -20.2 | +16.1 |
| 2026-02-24 | POWERUSDT | -20.2 | +25.1 |
| 2026-06-01 | VICUSDT | -20.2 | +4.0 |
| 2026-06-01 | SIRENUSDT | -20.2 | +0.6 |
| 2026-06-01 | HUSDT | -20.2 | +4.8 |
