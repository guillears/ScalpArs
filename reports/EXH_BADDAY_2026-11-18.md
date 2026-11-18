# EXHAUSTION bad days — theory, indicators, 2D screen, regime (2026-11-18)

Pre-registration: `reports/EXH_BADDAY_2026-11-18_PREREG.md` (main + 3 addenda, all frozen before the outcomes they cover).
Data: `reports/EXH_BADDAY_2026-11-18_tests.csv` (1D + top 2D), `_cells_top2000_tdc.csv`, `_days.csv`, `_replay.csv`. Chart: `reports/EXH_BADDAY_2026-11-18.png`.
Research only. Nothing in the bot was changed.

## Verdict first

**NO switch.** None of the 17 theory tests, none of 117,841 2D cells, no BTC regime and no bot regime label passes the pre-registered bar.
Every 1D theory "bad bucket" is still profitable (+0.30 to +0.70 %/trade). The best 2D cell for the terrible days is at the luck level
(family p 0.08) and, applied as a switch, it lowers the compound rate (+0.356 → +0.292 %/day) and halves the holdout rate.
The stop-cluster days are real, but at entry they look like ordinary days. Nothing we can read at the entry tells them apart.

One **OBSERVE** line is proposed (not armed): see "What it means".

## 1. Theory (written before any number)

**Why the short works.** A Frenzy coin pumps. Then the 15m RSI rolls over below 70 while volume is still high. High volume with fading
momentum means sellers are absorbing the late buyers. A falling spot share means the last leg was carried by leveraged perp longs.
Those longs are fragile: when price stops rising, they close, and price falls. We short that.

**Why it fails, and why fails come in clusters.** The short loses when new buying keeps arriving after our entry:
- (a) **Market-wide inflow / rotation** (alt season). Fresh money lifts every pumped coin again. One cause hits many trades → a cluster.
- (b) **Crowded shorts on the coin.** Negative funding and rising OI near the high mean many traders are already short. A small bounce
  forces them to buy back (a squeeze).
- (c) **The pattern itself stops working that day.** If other coins' RSI roll-overs keep making new highs, exhaustion is not exhaustion
  today. That is the most direct "today is bad" signal we can measure, and it is measured on all Frenzy coins, not on our own fills.
- (d) (operator addendum) **Buyers still in control:** RSI rolls over while +DI still dominates and is rising = a pause, not exhaustion.

**Indicators derived (all lag-safe, frozen in the prereg):**

| Indicator | Theory | Definition (short) | Pre-registered bad bucket |
|---|---|---|---|
| SIGFAIL_6H / _24H | (c) | Share of other Frenzy coins' 15m RSI 70 → ≤ 70 crosses, resolved in the last 6 / 24 h, that made a new high ≥ +3 % within 2 h (15,500 crosses, 2023-07 → 2026-11) | ≥ 0.50 · > median · top tercile |
| ALTVOL_EXP | (a) | Alt-perp quote volume, last 4 closed hours ÷ same hours on the 7 prior days (matched pairs, ≈ 180 → 530 perps) | > 1 · > median · top tercile |
| SPOT_LED | (a) | 22-alt basket spot share of volume, last 4 h minus the prior 7 days (rising = real buyers everywhere) | > 0 · > median · top tercile |
| COIN_CROWD2 / 3 | (b) | Funding ≤ 0 AND OI 1 h rising (CROWD3: and entry within the median distance of the 24 h high) | flag |
| COIN_DI15 | (d) | Coin 15m +DI − −DI at the signal; its change over 1 h | spread > 0 · rising · both |
| Short-liquidation proxy | (b) | dropped — already tested in SQUEEZER (`liq_short`) | — |

## 2. Method

- Trades: 3,998 simulated EXHAUSTION_SHORT trades, exit `B|tp10|S2|24` (TP 10 %, SL min(1.5×ATR, 6 %), 24 h), net %.
  DESIGN = 2023-07 → 2026-06 (3,328 trades, WR 39.8 %, avg win +8.39, avg loss −4.51 → **breakeven WR 35.0 %**).
  HOLDOUT = 2026-07 → 2026-11-16 (670 trades), read once.
- 1D: blocked vs kept, z, two family nulls (2,000 trade shuffles; 2,000 circular time shifts ≥ 7 days, keeps day clustering), day units
  (days with most entries in the bad bucket: day net and day stop share), per half-year, expectancy bar on the blocked cohort, holdout.
- 2D (addendum 1): 97 variables (theory 6 · 16 market · 75 coin / BTC / DI / label), natural cuts only → 494 buckets → **117,841 cells**
  with DESIGN N ≥ 30. Two scores: Δ = kept avg − all avg (trade level) and **TDC = share of the terrible-day stops removed − share of
  all winners removed** (window level; 27 terrible days = bottom decile of day net among days with ≥ 5 trades, ≤ −21.5 % Σ). Also z.
  300 day-block shuffles of the outcomes → family max null for each score.
- Regime (addenda 2 + 3): BTC 1d EMA50/200, BTC 1d close vs EMA200, an alt index regime, and the **bot's own label**
  (services/regime.py on BTC 5m, rebuilt at every entry; parity checked on real orders).
- DCR = one size 0.183 × equity per trade, P&L on the exit day, every calendar day, compounded.

## 3. Results — 1D theory indicators (DESIGN unless stated)

| Indicator | cut | blocked N · days | WR | avg blk | avg kept | z | family p (shuffle / shift) | bad-day vs other-day net Σ % | bad-day vs other stop share | blocked avg per half 23H2 · 24H1 · 24H2 · 25H1 · 25H2 · 26H1 | halves < 0 | P(avg<0) | HOLD N · avg blk / kept | BTC bull: blk / kept | BTC bear: blk / kept | survivor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SIGFAIL_6H | ge05 | 1406 · 441 | 40 % | +0.62 | +0.62 | -0.01 | 1.00 / 1.00 | +1.3 vs +2.8 | 0.61 vs 0.55 | +0.64 · +0.14 · +0.56 · -0.10 · +0.89 · +0.67 | 1 | 0.00 | 356 · +1.17 / +0.36 | +0.58 / +0.69 | +0.67 / +0.36 | no |
| SIGFAIL_6H | med | 1063 · 383 | 40 % | +0.54 | +0.65 | -0.45 | 0.98 / 0.97 | +1.4 vs +2.6 | 0.61 vs 0.56 | +0.80 · +0.13 · -0.10 · +0.12 · +0.84 · +0.58 | 1 | 0.01 | 281 · +1.07 / +0.59 | +0.51 / +0.71 | +0.59 / +0.49 | no |
| SIGFAIL_6H | T3 | 559 · 259 | 40 % | +0.51 | +0.64 | -0.41 | 0.98 / 0.98 | +3.8 vs +2.3 | 0.59 vs 0.56 | +0.31 · -0.67 · -0.52 · +0.48 · +0.74 · +0.67 | 2 | 0.04 | 166 · +1.33 / +0.61 | +0.65 / +0.66 | +0.36 / +0.60 | no |
| SIGFAIL_24H | ge05 | 1863 · 499 | 40 % | +0.58 | +0.67 | -0.41 | 0.98 / 0.98 | +2.4 vs +2.4 | 0.58 vs 0.56 | +0.31 · +0.29 · +0.43 · +0.31 · +0.82 · +0.53 | 0 | 0.00 | 546 · +0.97 / -0.01 | +0.61 / +0.69 | +0.52 / +0.57 | no |
| SIGFAIL_24H | med | 1497 · 422 | 40 % | +0.48 | +0.73 | -1.10 | 0.79 / 0.79 | +2.0 vs +2.6 | 0.58 vs 0.56 | +0.48 · +0.25 · +0.73 · -0.01 · +0.67 · +0.38 | 1 | 0.01 | 495 · +1.08 / -0.03 | +0.61 / +0.68 | +0.30 / +0.94 | no |
| SIGFAIL_24H | T3 | 939 · 297 | 39 % | +0.30 | +0.74 | -1.72 | 0.40 / 0.40 | +0.3 vs +2.9 | 0.59 vs 0.56 | -0.27 · +0.13 · +0.25 · +0.54 · +0.41 · +0.19 | 1 | 0.10 | 336 · +1.06 / +0.52 | +0.48 / +0.70 | +0.14 / +0.88 | no |
| ALTVOL_EXP | gt1 | 1498 · 499 | 40 % | +0.70 | +0.55 | +0.66 | 1.00 / 1.00 | +3.2 vs +1.9 | 0.56 vs 0.57 | +1.65 · +0.18 · +1.14 · -0.51 · +1.11 · +0.67 | 1 | 0.00 | 314 · +0.72 / +0.85 | +0.73 / +0.59 | +0.62 / +0.48 | no |
| ALTVOL_EXP | med | 1664 · 549 | 39 % | +0.60 | +0.64 | -0.20 | 0.99 / 0.99 | +2.6 vs +2.2 | 0.56 vs 0.57 | +1.44 · +0.26 · +0.99 · -0.47 · +0.89 · +0.53 | 1 | 0.00 | 371 · +0.80 / +0.78 | +0.65 / +0.66 | +0.46 / +0.60 | no |
| ALTVOL_EXP | T3 | 1109 · 396 | 40 % | +0.66 | +0.60 | +0.26 | 1.00 / 1.00 | +3.3 vs +2.1 | 0.56 vs 0.57 | +1.57 · +0.02 · +1.26 · -1.24 · +1.02 · +0.98 | 1 | 0.00 | 200 · +0.48 / +0.92 | +0.61 / +0.68 | +0.81 / +0.42 | no |
| SPOT_LED | gt0 | 1759 · 601 | 39 % | +0.51 | +0.74 | -1.03 | 0.82 / 0.81 | +2.3 vs +2.5 | 0.57 vs 0.56 | +0.97 · +0.79 · +0.58 · -0.74 · +0.62 · +0.49 | 1 | 0.00 | 364 · +0.76 / +0.82 | +0.52 / +0.82 | +0.48 / +0.59 | no |
| SPOT_LED | med | 1663 · 577 | 39 % | +0.51 | +0.73 | -0.99 | 0.85 / 0.83 | +2.1 vs +2.6 | 0.57 vs 0.56 | +0.79 · +0.73 · +0.70 · -0.67 · +0.69 · +0.42 | 1 | 0.00 | 349 · +0.73 / +0.86 | +0.54 / +0.78 | +0.42 / +0.63 | no |
| SPOT_LED | T3 | 1109 · 448 | 40 % | +0.57 | +0.64 | -0.28 | 0.99 / 0.99 | +2.9 vs +2.2 | 0.55 vs 0.57 | +1.39 · +0.80 · +1.10 · -1.06 · +0.74 · +0.27 | 1 | 0.00 | 238 · +0.72 / +0.83 | +0.70 / +0.63 | +0.21 / +0.67 | no |
| COIN_CROWD 2 | flag | 1288 · 526 | 40 % | +0.57 | +0.65 | -0.32 | 0.99 / 0.99 | +2.3 vs +2.4 | 0.56 vs 0.56 | +1.03 · +1.71 · -0.58 · +0.10 · +0.87 · +0.61 | 1 | 0.00 | 287 · +1.16 / +0.51 | +0.58 / +0.69 | +0.57 / +0.50 | no |
| COIN_CROWD 3 | flag | 679 · 385 | 38 % | +0.52 | +0.64 | -0.42 | 0.98 / 0.97 | +1.6 vs +2.5 | 0.57 vs 0.56 | +1.32 · +2.54 · -0.23 · +0.16 · +1.36 · -0.02 | 2 | 0.01 | 126 · +0.92 / +0.76 | +0.77 / +0.63 | +0.15 / +0.68 | no |
| COIN_DI15 (+DI>−DI) | gt0 | 3305 · 860 | 40 % | +0.62 | +0.22 | +0.29 | 1.00 / 1.00 | +2.4 vs +0.8 | 0.57 vs 0.50 | +0.85 · +0.67 · +0.76 · +0.08 · +0.75 · +0.58 | 0 | 0.00 | 665 · +0.80 / -0.36 | +0.65 / +1.95 | +0.57 / -4.71 | no |
| COIN_DI15 spread rising 1h | gt0 | 1439 · 643 | 40 % | +0.70 | +0.56 | +0.61 | 1.00 / 1.00 | +3.1 vs +2.1 | 0.55 vs 0.57 | +1.28 · +0.60 · +1.01 · -0.28 · +0.62 · +1.02 | 1 | 0.00 | 275 · +1.11 / +0.57 | +0.59 / +0.71 | +0.96 / +0.25 | no |
| COIN_DI15 both | flag | 1439 · 643 | 40 % | +0.70 | +0.56 | +0.61 | 1.00 / 1.00 | +3.1 vs +2.1 | 0.55 vs 0.57 | +1.28 · +0.60 · +1.01 · -0.28 · +0.62 · +1.02 | 1 | 0.00 | 275 · +1.11 / +0.57 | +0.59 / +0.71 | +0.96 / +0.25 | no |

**Read in plain words.**
- **(c) "exhaustion failing elsewhere" (SIGFAIL).** The direction is right in DESIGN: when ≥ 60 % of other coins' roll-overs failed in the
  last 24 h, our trades made +0.30 instead of +0.74. But the blocked trades are still profitable, z −1.7 is what the best of 17 random cuts
  gives 40 % of the time, it is negative in only 1 of 6 halves, and in the HOLDOUT the "bad" bucket did BETTER (+1.06 vs +0.52).
  The day-level picture is the same: cluster days sit at the 48th percentile of SIGFAIL-24h (chart, left panel: ρ = −0.10).
- **(a) inflow (ALTVOL, SPOT_LED).** High alt volume is, if anything, GOOD for the short (+0.70 vs +0.55). Rising spot share is slightly
  worse (+0.51 vs +0.74), but the blocked trades are positive in 5 of 6 halves and in the holdout.
- **(b) crowded shorts (CROWD2 / 3).** Blocked +0.57 / +0.52 vs kept +0.65 / +0.64. No effect.
- **(d) +DI still in control.** At the signal the coin's 15m +DI is above −DI in 99 % of trades (3,305 of 3,328) — a roll-over right after a
  pump. A rising DI spread is BETTER for the short (+0.70 vs +0.56; holdout +1.11 vs +0.57). The "pause, not exhaustion" theory is refuted.
- 0 of 17 tests survive. 0 of 17 even reach P(avg < 0) ≥ 0.95 on the blocked cohort; the highest is 0.10.

## 4. Results — exhaustive 2D screen (117,841 cells)

Family null (300 day-block shuffles): 95th percentile of the family max Δ = +2.44 pp, of max TDC = 0.270, 5th percentile of min z = −4.85.
**No cell beats the null on any score.** Best observed: Δ +0.52 (p 0.95), TDC 0.257 (p 0.08), z −4.62 (p 0.10).
About 2.5 % of cells have a negative blocked average in DESIGN (2,976); of those with ≥ 30 holdout trades, 25 % stay negative in the
holdout vs a 12 % base rate — a weak "something", consistent with a few small real effects too small to find one cell reliably.

**Top 20 by TDC** (blocked = the cell; HOLDOUT read once for the top 10)

| # | cell (blocked) | N · days | WR | avg blk | avg kept | Δ pp | TDC | fam p (Δ / TDC) | P(avg<0) | halves < 0 (of 6) | bar | bad-day stop share vs other | HOLD N · avg blk / kept | bull / bear avg blk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | sf24T3(>0.6) ∧ VSW15==0 | 833 · 284 | 38 % | +0.24 | +0.75 | +0.13 | 0.257 | 1.00 / 0.08 | 0.16 | 1 | no | 0.59 vs 0.56 | 309 · +0.95 / +0.65 | +0.46 (404) / +0.03 (429) |
| 2 | sf24T3(>0.6) ∧ ret_4h>0 | 929 · 294 | 38 % | +0.26 | +0.76 | +0.14 | 0.246 | 1.00 / 0.10 | 0.14 | 1 | no | 0.60 vs 0.56 | 333 · +1.01 / +0.58 | +0.41 (457) / +0.12 (472) |
| 3 | sf24T3(>0.6) ∧ lab_BULL_EXHAUSTED==0 | 910 · 295 | 39 % | +0.28 | +0.74 | +0.13 | 0.244 | 1.00 / 0.11 | 0.12 | 1 | no | 0.59 vs 0.56 | 318 · +0.81 / +0.77 | +0.46 (444) / +0.11 (466) |
| 4 | sf24T3(>0.6) ∧ fund_med>0 | 937 · 297 | 38 % | +0.29 | +0.75 | +0.13 | 0.242 | 1.00 / 0.11 | 0.13 | 1 | no | 0.59 vs 0.56 | 336 · +1.06 / +0.52 | +0.46 (463) / +0.12 (474) |
| 5 | sf24T3(>0.6) ∧ ret_15m<=0 | 939 · 297 | 39 % | +0.30 | +0.74 | +0.12 | 0.241 | 1.00 / 0.11 | 0.11 | 1 | no | 0.59 vs 0.56 | 336 · +1.06 / +0.52 | +0.48 (464) / +0.14 (475) |
| 6 | sf24T3(>0.6) ∧ has_spot==1 | 939 · 297 | 39 % | +0.30 | +0.74 | +0.12 | 0.241 | 1.00 / 0.11 | 0.10 | 1 | no | 0.59 vs 0.56 | 336 · +1.06 / +0.52 | +0.48 (464) / +0.14 (475) |
| 7 | sf24T3(>0.6) ∧ ssc<=0 | 939 · 297 | 39 % | +0.30 | +0.74 | +0.12 | 0.241 | 1.00 / 0.11 | 0.11 | 1 | no | 0.59 vs 0.56 | 336 · +1.06 / +0.52 | +0.48 (464) / +0.14 (475) |
| 8 | sf24T3(>0.6) ∧ bk_imb2>0 | 915 · 283 | 39 % | +0.33 | +0.73 | +0.11 | 0.241 | 1.00 / 0.11 | 0.10 | 0 | no | 0.59 vs 0.56 | 334 · +1.06 / +0.52 | +0.53 (441) / +0.15 (474) |
| 9 | sf24T3(>0.6) ∧ bk_imb5>0 | 915 · 283 | 39 % | +0.33 | +0.73 | +0.11 | 0.241 | 1.00 / 0.11 | 0.10 | 0 | no | 0.59 vs 0.56 | 334 · +1.06 / +0.52 | +0.53 (441) / +0.15 (474) |
| 10 | sf24T3(>0.6) ∧ lab_HEALTHY_BULL==0 | 808 · 283 | 38 % | +0.24 | +0.74 | +0.12 | 0.240 | 1.00 / 0.11 | 0.20 | 2 | no | 0.61 vs 0.56 | 297 · +1.10 / +0.54 | +0.48 (402) / -0.00 (406) |
| 11 | sf24T3(>0.6) ∧ c15_spread>0 | 932 · 294 | 39 % | +0.31 | +0.74 | +0.12 | 0.238 | 1.00 / 0.12 | 0.11 | 1 | no | 0.59 vs 0.56 | — | +0.45 (460) / +0.17 (472) |
| 12 | sf24T3(>0.6) ∧ oi_chg_4h>0 | 928 · 294 | 38 % | +0.27 | +0.75 | +0.13 | 0.237 | 1.00 / 0.12 | 0.13 | 1 | no | 0.60 vs 0.56 | — | +0.42 (458) / +0.13 (470) |
| 13 | sf24T3(>0.6) ∧ taker_chg<=0 | 710 · 263 | 38 % | +0.20 | +0.73 | +0.11 | 0.236 | 1.00 / 0.12 | 0.24 | 1 | no | 0.59 vs 0.56 | — | +0.35 (343) / +0.06 (367) |
| 14 | sf24T3(>0.6) ∧ vs_vwap24>0 | 858 · 288 | 38 % | +0.23 | +0.75 | +0.14 | 0.235 | 1.00 / 0.12 | 0.18 | 1 | no | 0.60 vs 0.56 | — | +0.34 (427) / +0.11 (431) |
| 15 | sf24T3(>0.6) ∧ RSI5==0 | 885 · 293 | 38 % | +0.28 | +0.74 | +0.12 | 0.234 | 1.00 / 0.13 | 0.14 | 2 | no | 0.59 vs 0.56 | — | +0.56 (438) / +0.01 (447) |
| 16 | sf24T3(>0.6) ∧ ret_24h>0 | 928 · 294 | 39 % | +0.31 | +0.74 | +0.12 | 0.234 | 1.00 / 0.13 | 0.10 | 1 | no | 0.60 vs 0.56 | — | +0.49 (460) / +0.14 (468) |
| 17 | sf24T3(>0.6) ∧ RSI1H==0 | 925 · 297 | 38 % | +0.30 | +0.74 | +0.12 | 0.231 | 1.00 / 0.14 | 0.12 | 1 | no | 0.59 vs 0.56 | — | +0.51 (456) / +0.09 (469) |
| 18 | sf24T3(>0.6) ∧ oi_chg_24h>0 | 921 · 293 | 39 % | +0.32 | +0.73 | +0.11 | 0.230 | 1.00 / 0.15 | 0.09 | 1 | no | 0.60 vs 0.56 | — | +0.45 (453) / +0.20 (468) |
| 19 | sf24T3(>0.6) ∧ lab_BEAR_EXHAUSTED==0 | 904 · 295 | 38 % | +0.30 | +0.74 | +0.12 | 0.228 | 1.00 / 0.16 | 0.11 | 1 | no | 0.59 vs 0.56 | — | +0.40 (449) / +0.20 (455) |
| 20 | dom24>0 ∧ r2_btc_c200==-1 | 742 · 228 | 37 % | +0.14 | +0.75 | +0.14 | 0.228 | 1.00 / 0.16 | 0.29 | 3 | no | 0.59 vs 0.56 | — | -0.22 (137) / +0.23 (605) |

**Top 10 by z (blocked worse than kept, DESIGN)**

| # | cell (blocked) | N · days | WR | avg blk | avg kept | z | fam p (z) | P(avg<0) | halves < 0 | bar | HOLD N · avg blk / kept | bull / bear avg blk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | oi_chg_1h<=med(8.64) ∧ mcapT1(<=7.83e+07) | 448 · 261 | 31 % | -0.72 | +0.83 | -4.62 | 0.10 | 0.98 | 2 | no | 260 · +0.51 / +0.97 | -0.31 / -0.93 |
| 2 | mcapT1(<=7.83e+07) ∧ pre1h_short<=med(0.84) | 463 · 289 | 32 % | -0.63 | +0.82 | -4.40 | 0.24 | 0.98 | 3 | no | 234 · +0.69 / +0.84 | +0.25 / -1.21 |
| 3 | ret_1hT1(<=-0.176) ∧ mcapT1(<=7.83e+07) | 318 · 226 | 32 % | -0.84 | +0.77 | -4.15 | 0.44 | 0.99 | 2 | no | 171 · +0.65 / +0.84 | +0.13 / -1.34 |
| 4 | ssc>med(-0.0366) ∧ oi_volT3(>0.128) | 512 · 347 | 30 % | -0.49 | +0.82 | -4.15 | 0.45 | 0.98 | 4 | PASS | 119 · -0.16 / +1.00 | -0.54 / -0.37 |
| 5 | fund_neg_shareT3(>0.24) ∧ oi_usdT2(6.81e+06..1.44e+07] | 362 · 197 | 32 % | -0.74 | +0.78 | -4.15 | 0.45 | 0.98 | 4 | PASS | 13 · +1.70 / +0.77 | -0.26 / -1.17 |
| 6 | mcapT1(<=7.83e+07) ∧ pre1h_shortT1(<=-0.928) | 321 · 229 | 32 % | -0.83 | +0.77 | -4.14 | 0.45 | 0.99 | 3 | no | 162 · +0.67 / +0.83 | -0.01 / -1.28 |
| 7 | ret_1h<=0 ∧ mcapT1(<=7.83e+07) | 338 · 236 | 32 % | -0.78 | +0.78 | -4.12 | 0.47 | 0.99 | 2 | no | 178 · +0.72 / +0.82 | +0.32 / -1.35 |
| 8 | ret_1h<=med(1.49) ∧ mcapT1(<=7.83e+07) | 451 · 285 | 33 % | -0.55 | +0.80 | -4.07 | 0.54 | 0.96 | 3 | no | 231 · +0.60 / +0.89 | +0.21 / -1.03 |
| 9 | mcapT1(<=7.83e+07) ∧ pre1h_short<=0 | 396 · 266 | 33 % | -0.64 | +0.79 | -4.04 | 0.58 | 0.97 | 2 | no | 205 · +0.63 / +0.86 | +0.16 / -1.12 |
| 10 | oi_chg_1h<=med(8.64) ∧ ls_glob_chg4h<=0 | 1260 · 617 | 35 % | +0.03 | +0.98 | -4.01 | 0.64 | 0.43 | 3 | no | 259 · +0.18 / +1.18 | +0.17 / -0.32 |

**Top by Δ** (first 9 shown; all 20 with their holdout reads in `_tests.csv`)

| # | cell (blocked) | N · days | WR | avg blk | avg kept | Δ pp | TDC | fam p (Δ / TDC) | P(avg<0) | halves < 0 (of 6) | bar | bad-day stop share vs other | HOLD N · avg blk / kept | bull / bear avg blk |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | fund_med>0 ∧ ret_4h>0 | 3275 · 858 | 40 % | +0.61 | +1.14 | +0.52 | 0.013 | 0.95 / 1.00 | 0.00 | 0 | no | 0.57 vs 0.55 | 664 · +0.76 / +3.73 | +0.64 (2262) / +0.54 (1013) |
| 2 | RSI5==0 ∧ STOCH15==0 | 2287 · 774 | 39 % | +0.41 | +1.08 | +0.46 | 0.043 | 0.97 / 1.00 | 0.00 | 1 | no | 0.57 vs 0.54 | 420 · +0.91 / +0.59 | +0.44 (1574) / +0.33 (713) |
| 3 | ssc<=0 ∧ ls_glob_chg4h<=0 | 2578 · 822 | 39 % | +0.50 | +1.04 | +0.42 | -0.010 | 0.98 / 1.00 | 0.00 | 0 | no | 0.57 vs 0.54 | 431 · +0.62 / +1.10 | +0.51 (1942) / +0.47 (636) |
| 4 | ls_glob_chg4h<=0 ∧ has_spot==1 | 2578 · 822 | 39 % | +0.50 | +1.04 | +0.42 | -0.010 | 0.98 / 1.00 | 0.00 | 0 | no | 0.57 vs 0.54 | 431 · +0.62 / +1.10 | +0.51 (1942) / +0.47 (636) |
| 5 | ls_glob_chg4h<=0 ∧ ret_15m<=0 | 2578 · 822 | 39 % | +0.50 | +1.04 | +0.42 | -0.010 | 0.98 / 1.00 | 0.00 | 0 | no | 0.57 vs 0.54 | 431 · +0.62 / +1.10 | +0.51 (1942) / +0.47 (636) |
| 6 | fund_med>0 ∧ ls_glob_chg4h<=0 | 2561 · 820 | 39 % | +0.49 | +1.03 | +0.41 | -0.005 | 0.98 / 1.00 | 0.00 | 0 | no | 0.57 vs 0.54 | 431 · +0.62 / +1.10 | +0.51 (1929) / +0.46 (632) |
| 7 | ssc<=0 ∧ STOCH15==0 | 2417 · 789 | 39 % | +0.46 | +1.03 | +0.41 | 0.019 | 0.98 / 1.00 | 0.00 | 1 | no | 0.57 vs 0.54 | 441 · +0.91 / +0.57 | +0.47 (1666) / +0.44 (751) |
| 8 | ret_15m<=0 ∧ STOCH15==0 | 2417 · 789 | 39 % | +0.46 | +1.03 | +0.41 | 0.019 | 0.98 / 1.00 | 0.00 | 1 | no | 0.57 vs 0.54 | 441 · +0.91 / +0.57 | +0.47 (1666) / +0.44 (751) |
| 9 | has_spot==1 ∧ STOCH15==0 | 2417 · 789 | 39 % | +0.46 | +1.03 | +0.41 | 0.019 | 0.98 / 1.00 | 0.00 | 1 | no | 0.57 vs 0.54 | 441 · +0.91 / +0.57 | +0.47 (1666) / +0.44 (751) |

**Read in plain words.**
- **By Δ** the top cells block 2,200–3,300 of 3,328 trades and keep a small, lucky remainder. That is what a max-Δ search finds by
  construction; the null says so (p ≥ 0.95). Not a bad-day filter. (Rows 11–20 in the CSV.)
- **By TDC (the terrible-day target)** every top cell is "SIGFAIL-24h top tercile ∧ something almost always true". The cell removes many
  terrible-day stops but also 448 winners; p 0.08 – 0.18. Its blocked trades are still positive (+0.20 to +0.33) and in the HOLDOUT
  they did better than the kept ones (+0.95 vs +0.65 for #1).
- **By z** the strongest cells are about **small market cap** (bottom tercile, ≤ $78 M, approximate: today's supply × entry price) with
  a weak last hour. They lose in DESIGN (−0.6 to −0.8 %/trade, P(avg<0) ≈ 0.98) but the family null puts them at the luck level
  (p 0.10 – 0.58) and they are POSITIVE in the holdout (+0.51 to +0.72).
- Two cells pass the expectancy bar in DESIGN only (#4 spot-share-not-falling ∧ high OI/volume; #5 many negative-funding perps ∧
  mid OI). Both fail the family null (p 0.45). #4 is still negative in the holdout (−0.16 vs +1.00, N 119 · 68 days, P(avg<0) 0.63).

**Best cell for the terrible days: block when sf24T3(>0.6) ∧ VSW15==0 (family p 0.08 — does NOT pass)**

| Half | before N · WR · avg | after N · WR · avg | DCR before → after (%/day) | max DD before → after | worst day before → after (% equity) |
|---|---|---|---|---|---|
| 2023-H2 | 283 · 40 % · +0.83 | 263 · 41 % · +0.89 | +0.222 → +0.224 | -10.1 → -9.0 % | -4.57 → -4.57 |
| 2024-H1 | 449 · 38 % · +0.66 | 423 · 38 % · +0.70 | +0.284 → +0.284 | -16.3 → -15.8 % | -5.67 → -5.67 |
| 2024-H2 | 449 · 40 % · +0.75 | 408 · 40 % · +0.81 | +0.307 → +0.301 | -11.4 → -13.0 % | -8.91 → -8.91 |
| 2025-H1 | 434 · 37 % · +0.17 | 367 · 35 % · +0.04 | +0.058 → -0.002 | -17.0 → -22.7 % | -5.80 → -5.08 |
| 2025-H2 | 887 · 42 % · +0.76 | 541 · 44 % · +1.07 | +0.626 → +0.553 | -18.6 → -9.2 % | -9.96 → -5.13 |
| 2026-H1 | 826 · 40 % · +0.54 | 493 · 42 % · +0.84 | +0.408 → +0.394 | -22.6 → -16.0 % | -9.87 → -6.37 |
| 2026-H2 | 670 · 42 % · +0.79 | 361 · 41 % · +0.65 | +0.655 → +0.291 | -21.6 → -13.7 % | -6.90 → -5.78 |
| DESIGN | 3328 · 40 % · +0.62 | 2495 · 40 % · +0.75 | +0.318 → +0.293 | -22.6 → -22.7 % | -9.96 → -8.91 |
| HOLD | 670 · 42 % · +0.79 | 361 · 41 % · +0.65 | +0.655 → +0.291 | -21.6 → -13.7 % | -6.90 → -5.78 |
| ALL | 3998 · 40 % · +0.65 | 2856 · 41 % · +0.73 | +0.356 → +0.292 | -22.6 → -22.7 % | -9.96 → -8.91 |

**Top-15 stop-cluster days and each indicator (day mean over the day's entries; percentile among the 333 days with ≥ 5 trades)**

| Day | trades | stops | day net Σ % | SIGFAIL 6 h | SIGFAIL 24 h | ALTVOL exp | SPOT_LED Δ | CROWD2 share | CROWD3 share | BTC 1d 50/200 | best cell: stops removed / winners blocked / net blocked |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-24 | 13 | 13 | -65.4 | 0.63 (63) | 0.64 (77) | 0.62 (4) | +1.18 pp (79) | 0.85 | 0.46 | bear | 10 / 0 / -50.4 |
| 2024-11-16 | 14 | 13 | -52.5 | 0.50 (35) | 0.48 (29) | 0.77 (14) | +1.20 pp (80) | 0.29 | 0.14 | bull | 0 / 0 / +0.0 |
| 2024-12-03 | 15 | 12 | -28.8 | 0.59 (56) | 0.45 (20) | 1.91 (97) | +1.12 pp (77) | 0.07 | 0.00 | bull | 2 / 0 / -9.8 |
| 2024-11-24 | 26 | 12 | +65.6 | 0.55 (46) | 0.55 (48) | 1.29 (83) | +0.17 pp (48) | 0.00 | 0.00 | bull | 0 / 1 / +4.6 |
| 2026-09-25 | 14 | 11 | -36.3 | 0.39 (16) | 0.50 (33) | 1.02 (52) | +0.55 pp (61) | 0.43 | 0.21 | bull | 1 / 0 / -6.1 |
| 2024-11-25 | 17 | 11 | +7.3 | 0.48 (29) | 0.45 (20) | 1.20 (74) | +0.30 pp (53) | 0.06 | 0.00 | bull | 0 / 0 / +0.0 |
| 2024-11-27 | 11 | 10 | -34.8 | 0.61 (59) | 0.69 (86) | 0.78 (17) | +0.15 pp (47) | 0.09 | 0.00 | bull | 9 / 1 / -28.7 |
| 2024-11-23 | 12 | 10 | -32.8 | 0.30 (8) | 0.36 (6) | 1.29 (83) | +1.98 pp (92) | 0.25 | 0.08 | bull | 0 / 0 / +0.0 |
| 2024-02-26 | 13 | 10 | -32.7 | 0.48 (28) | 0.46 (22) | 1.12 (66) | +0.32 pp (54) | 0.00 | 0.00 | bull | 0 / 0 / +0.0 |
| 2025-11-17 | 13 | 10 | -28.9 | 0.62 (61) | 0.70 (87) | 1.02 (53) | -0.19 pp (34) | 0.77 | 0.69 | bull | 10 / 3 / -28.9 |
| 2026-09-20 | 15 | 10 | -10.0 | 0.43 (21) | 0.52 (38) | 1.10 (65) | +1.48 pp (87) | 0.07 | 0.07 | bull | 0 / 1 / +9.9 |
| 2024-03-02 | 14 | 10 | -1.8 | 0.55 (46) | 0.55 (50) | 1.37 (89) | +0.58 pp (62) | 0.00 | 0.00 | bull | 0 / 0 / +0.0 |
| 2024-12-04 | 19 | 10 | +24.8 | 0.43 (21) | 0.55 (50) | 1.41 (90) | +1.27 pp (81) | 0.00 | 0.00 | bull | 2 / 3 / +15.1 |
| 2026-03-31 | 10 | 9 | -38.2 | 0.62 (60) | 0.59 (61) | 1.10 (63) | -0.48 pp (25) | 0.60 | 0.50 | bear | 3 / 0 / -15.4 |
| 2026-05-05 | 11 | 9 | -34.5 | 0.60 (57) | 0.63 (74) | 1.13 (68) | +1.07 pp (77) | 0.09 | 0.00 | bear | 8 / 0 / -36.2 |

Median percentile of the 15 cluster days: SIGFAIL 6 h 46 · SIGFAIL 24 h 48 · ALTVOL 66 · SPOT_LED 62. Best cell on these 15 days removes 45 of 160 stops and blocks 9 winners (net blocked -145.7 %). Over all 3998 trades it blocks 1142 trades: 687 stops, 448 winners, net +491 %.

**Read.** On the 15 worst days the best cell removes 45 of 160 stops (28 %). It removes most of 2026-01-24, 2024-11-27, 2025-11-17 and
2026-05-05, but nothing on 2024-11-16, 2024-11-23 or 2024-02-26. Over the whole sample it blocks 1,142 trades: 687 stops, 448 winners,
net **+491 %** of P&L given away. The DCR goes down in 5 of 7 halves and halves in the holdout (+0.655 → +0.291 %/day).
Drawdown improves in 2025-H2 / 2026-H1 / 2026-H2, which is the visible appeal, but it is paid for with more profit than it saves.

**No indicator marks the cluster days.** Their median percentile among the 333 busy days: SIGFAIL 6 h 46th, SIGFAIL 24 h 48th,
ALTVOL 66th, SPOT_LED 62nd. 2024-11-16 (13 of 14 stopped) shows normal or "good" values on every indicator.

## 5. Regime

**Baseline (all trades, one size 0.183 × equity, booked on exit day)**

| Period | N | WR | avg net % | DCR %/day | max DD | worst day (% equity) |
|---|---|---|---|---|---|---|
| 2023-H2 | 283 | 40 % | +0.83 | +0.222 | -10.1 % | -4.57 % |
| 2024-H1 | 449 | 38 % | +0.66 | +0.284 | -16.3 % | -5.67 % |
| 2024-H2 | 449 | 40 % | +0.75 | +0.307 | -11.4 % | -8.91 % |
| 2025-H1 | 434 | 37 % | +0.17 | +0.058 | -17.0 % | -5.80 % |
| 2025-H2 | 887 | 42 % | +0.76 | +0.626 | -18.6 % | -9.96 % |
| 2026-H1 | 826 | 40 % | +0.54 | +0.408 | -22.6 % | -9.87 % |
| 2026-H2 | 670 | 42 % | +0.79 | +0.655 | -21.6 % | -6.90 % |
| DESIGN | 3328 | 40 % | +0.62 | +0.318 | -22.6 % | -9.96 % |
| HOLD | 670 | 42 % | +0.79 | +0.655 | -21.6 % | -6.90 % |
| ALL | 3998 | 40 % | +0.65 | +0.356 | -22.6 % | -9.96 % |

**Daily regime (BTC 1d)**


| Regime | block | N | days | WR | avg | DCR (regime days only) | max DD | worst day | day stop share |
|---|---|---|---|---|---|---|---|---|---|
| BTC 1d EMA50 vs EMA200 above (bull) | DESIGN | 2303 | 644 | 40 % | +0.65 | +0.295 | -17.0 % | -8.91 % | 0.56 |
| BTC 1d EMA50 vs EMA200 above (bull) | HOLD | 313 | 63 | 38 % | +0.18 | +0.120 | -18.3 % | -6.90 % | 0.57 |
| BTC 1d EMA50 vs EMA200 below (bear) | DESIGN | 1025 | 218 | 40 % | +0.54 | +0.091 | -22.6 % | -9.96 % | 0.57 |
| BTC 1d EMA50 vs EMA200 below (bear) | HOLD | 357 | 73 | 45 % | +1.32 | +1.111 | -8.1 % | -5.77 % | 0.55 |
| BTC 1d close vs EMA200 above (bull) | DESIGN | 2021 | 556 | 39 % | +0.67 | +0.219 | -16.3 % | -8.91 % | 0.57 |
| BTC 1d close vs EMA200 above (bull) | HOLD | 409 | 76 | 40 % | +0.49 | +0.437 | -21.6 % | -6.90 % | 0.55 |
| BTC 1d close vs EMA200 below (bear) | DESIGN | 1307 | 306 | 40 % | +0.54 | +0.113 | -22.6 % | -9.96 % | 0.56 |
| BTC 1d close vs EMA200 below (bear) | HOLD | 261 | 60 | 46 % | +1.26 | +0.412 | -8.1 % | -5.77 % | 0.56 |

Per half-year × BTC 1d EMA50/200 (avg net % · N):

| Half | bull avg · N | bear avg · N |
|---|---|---|
| 2023-H2 | +0.80 · 279 | +2.49 · 4 |
| 2024-H1 | +0.66 · 449 | n/a · 0 |
| 2024-H2 | +0.75 · 449 | n/a · 0 |
| 2025-H1 | +0.17 · 434 | n/a · 0 |
| 2025-H2 | +0.83 · 692 | +0.49 · 195 |
| 2026-H1 | n/a · 0 | +0.54 · 826 |
| 2026-H2 | +0.18 · 313 | +1.32 · 357 |

Bear-regime as a switch (block EXHAUSTION in BTC bear, DESIGN): blocked N 1025 · days 218 · WR 40 % · avg +0.54 · P(avg<0) 0.01 — bull blocked: N 2303 · avg +0.65 · P(avg<0) 0.00.
Alt regime (equal-weight alt index EMA50 vs EMA200): below EMA200 on 3288 of 3288 scored trades — no split possible (an equal-weight alt index trends down for the whole sample).

**Read (daily regime).**
- EXHAUSTION is positive in BOTH BTC daily regimes. DESIGN: bull +0.65 (2,303 trades), bear +0.54 (1,025). HOLDOUT: bull +0.18, bear +1.32.
- The CONFIRMED_TOP_SHORT hint (Frenzy shorts better in BTC bear) holds in the HOLDOUT only. In DESIGN, bear is a bit weaker, so the
  sign flips between eras → not a regime rule. Regime is almost the calendar here (2024 all bull, 2026-H1 all bear), so a regime split
  cannot be told apart from a half-year effect.
- Neither regime as a switch passes the bar (blocked P(avg<0) 0.00 / 0.01: both clearly profitable).
- The alt regime (equal-weight alt index, EMA50 vs EMA200) was below on every scored day — an equal-weight index of listed alts trends
  down for the whole sample, so it cannot split anything. Listed as a blind spot.
- Within regime, the 1D tests do not flip in a useful way: SIGFAIL-24h top tercile is +0.48 / +0.70 (bull) and +0.14 / +0.88 (bear) —
  worse in bear in DESIGN — but in the HOLDOUT the bear bucket made +1.91 %/trade (190 trades). Flip → noise.
- One holdout-only pattern, NOT supported by DESIGN: in the 2026-H2 BTC-bull weeks, high alt volume / rising spot share buckets lost
  (ALTVOL top tercile −0.86, N 93; SPOT_LED top tercile −0.99, N 106), while in DESIGN-bull the same buckets were positive
  (+0.61 / +0.70). Reading it as a rule would be fitting on the holdout. Watch only.

### Bot regime label (services/regime.py rebuilt)

**Parity of the rebuilt label** on 1324 real master orders (2026-06-17 → 11-17) with a stamped entry_btc_regime: 88.1 % exact match with 99 closed 5m bars + the forming bar at its open (used), 85.1 % with 100 closed bars. The rest are mostly boundary cases (ADX near 18 / 28, slope near ±0.02 %): the engine reads a partly formed bar at an arbitrary second.
Coin DI parity: entry_frenzy_di_spread (300 closed 5m bars) rebuilt on 211 orders — median |Δ| 0.000, sign match 100 %. entry_pos_di / entry_neg_di (100-bar window incl. the forming bar, so not exact): +DI−−DI sign match 90 % on 1322 orders.

| Label | block | N | days | WR | avg net % | Σ day net per active day | day stop share | P(avg<0) | DESIGN halves (avg · N) |
|---|---|---|---|---|---|---|---|---|---|
| CHOPPY_WEAK | DESIGN | 940 | 505 | 39 % | +0.56 | +1.0 | 0.57 | 0.00 | 23-H2 +1.01 (74) · 24-H1 +0.57 (145) · 24-H2 +0.38 (143) · 25-H1 +0.90 (113) · 25-H2 +1.04 (259) · 26-H1 -0.27 (206) |
| CHOPPY_WEAK | HOLD | 180 | 93 | 41 % | +0.72 | +1.4 | 0.58 | 0.06 |  |
| CHOPPY_FLAT | DESIGN | 564 | 352 | 42 % | +0.96 | +1.5 | 0.55 | 0.00 | 23-H2 +0.57 (62) · 24-H1 +2.17 (58) · 24-H2 +2.01 (54) · 25-H1 -0.21 (60) · 25-H2 +0.60 (155) · 26-H1 +1.08 (175) |
| CHOPPY_FLAT | HOLD | 159 | 82 | 43 % | +0.91 | +1.8 | 0.53 | 0.07 |  |
| HEALTHY_BULL | DESIGN | 459 | 328 | 41 % | +0.73 | +1.0 | 0.59 | 0.01 | 23-H2 +1.78 (24) · 24-H1 +0.56 (56) · 24-H2 +1.72 (61) · 25-H1 +0.08 (74) · 25-H2 +0.40 (119) · 26-H1 +0.81 (125) |
| HEALTHY_BULL | HOLD | 62 | 50 | 40 % | +0.59 | +0.7 | 0.60 | 0.29 |  |
| STRONG_BULL | DESIGN | 344 | 252 | 38 % | +0.42 | +0.6 | 0.58 | 0.11 | 23-H2 +0.15 (28) · 24-H1 +0.29 (40) · 24-H2 +0.37 (49) · 25-H1 -1.02 (53) · 25-H2 +0.50 (93) · 26-H1 +1.45 (81) |
| STRONG_BULL | HOLD | 86 | 57 | 42 % | +0.78 | +1.2 | 0.55 | 0.14 |  |
| BULL_EXHAUSTED | DESIGN | 102 | 90 | 40 % | +0.75 | +0.9 | 0.59 | 0.12 | 23-H2 -0.39 (12) · 24-H1 +0.52 (13) · 24-H2 +0.69 (17) · 25-H1 +1.12 (15) · 25-H2 +0.83 (24) · 26-H1 +1.26 (21) |
| BULL_EXHAUSTED | HOLD | 38 | 32 | 50 % | +2.05 | +2.4 | 0.50 | 0.04 |  |
| HEALTHY_BEAR | DESIGN | 468 | 321 | 37 % | +0.28 | +0.4 | 0.60 | 0.17 | 23-H2 +1.04 (37) · 24-H1 +0.43 (80) · 24-H2 +0.18 (59) · 25-H1 +0.02 (69) · 25-H2 +0.43 (126) · 26-H1 -0.06 (97) |
| HEALTHY_BEAR | HOLD | 74 | 51 | 39 % | +0.18 | +0.3 | 0.56 | 0.39 |  |
| STRONG_BEAR | DESIGN | 331 | 237 | 42 % | +0.90 | +1.3 | 0.57 | 0.01 | 23-H2 +1.22 (36) · 24-H1 +0.63 (48) · 24-H2 +0.28 (44) · 25-H1 +0.11 (30) · 25-H2 +1.45 (86) · 26-H1 +0.97 (87) |
| STRONG_BEAR | HOLD | 56 | 38 | 45 % | +0.95 | +1.4 | 0.55 | 0.16 |  |
| BEAR_EXHAUSTED | DESIGN | 120 | 99 | 37 % | +0.04 | +0.1 | 0.62 | 0.46 | 23-H2 -0.08 (10) · 24-H1 -2.73 (9) · 24-H2 +0.72 (22) · 25-H1 +0.53 (20) · 25-H2 +0.66 (25) · 26-H1 -0.36 (34) |
| BEAR_EXHAUSTED | HOLD | 14 | 12 | 43 % | +0.86 | +1.0 | 0.62 | 0.34 |  |

Label as a switch (block EXHAUSTION in label X) is judged on the same bar:
- CHOPPY_WEAK: no (WR 39 % vs breakeven 35 %, avg +0.56)
- CHOPPY_FLAT: no (WR 42 % vs breakeven 35 %, avg +0.96)
- HEALTHY_BULL: no (WR 41 % vs breakeven 35 %, avg +0.73)
- STRONG_BULL: no (WR 38 % vs breakeven 35 %, avg +0.42)
- BULL_EXHAUSTED: no (WR 40 % vs breakeven 35 %, avg +0.75)
- HEALTHY_BEAR: no (WR 37 % vs breakeven 35 %, avg +0.28)
- STRONG_BEAR: no (WR 42 % vs breakeven 35 %, avg +0.90)
- BEAR_EXHAUSTED: no (WR 37 % vs breakeven 35 %, avg +0.04)

**Top-15 cluster days — labels at their entries (count per label):**

| Day | stops / trades | day net | labels at entry |
|---|---|---|---|
| 2026-01-24 | 13 / 13 | -65.4 | CHOPPY_FLAT 5, CHOPPY_WEAK 5, HEALTHY_BULL 1, HEALTHY_BEAR 1, BEAR_EXHAUSTED 1 |
| 2024-11-16 | 13 / 14 | -52.5 | CHOPPY_WEAK 6, STRONG_BEAR 4, HEALTHY_BEAR 3, STRONG_BULL 1 |
| 2024-12-03 | 12 / 15 | -28.8 | CHOPPY_WEAK 7, STRONG_BULL 4, HEALTHY_BEAR 3, HEALTHY_BULL 1 |
| 2024-11-24 | 12 / 26 | +65.6 | CHOPPY_WEAK 10, CHOPPY_FLAT 6, STRONG_BULL 3, HEALTHY_BEAR 3, STRONG_BEAR 2, HEALTHY_BULL 1, BEAR_EXHAUSTED 1 |
| 2026-09-25 | 11 / 14 | -36.3 | HEALTHY_BEAR 4, HEALTHY_BULL 3, CHOPPY_WEAK 3, BULL_EXHAUSTED 1, STRONG_BULL 1, CHOPPY_FLAT 1, STRONG_BEAR 1 |
| 2024-11-25 | 11 / 17 | +7.3 | CHOPPY_WEAK 6, HEALTHY_BEAR 4, STRONG_BEAR 3, HEALTHY_BULL 2, CHOPPY_FLAT 1, BEAR_EXHAUSTED 1 |
| 2024-11-27 | 10 / 11 | -34.8 | CHOPPY_WEAK 5, HEALTHY_BULL 3, CHOPPY_FLAT 1, BULL_EXHAUSTED 1, STRONG_BULL 1 |
| 2024-11-23 | 10 / 12 | -32.8 | CHOPPY_WEAK 4, BEAR_EXHAUSTED 4, STRONG_BEAR 2, HEALTHY_BULL 1, CHOPPY_FLAT 1 |
| 2024-02-26 | 10 / 13 | -32.7 | STRONG_BULL 4, CHOPPY_WEAK 2, HEALTHY_BEAR 2, BEAR_EXHAUSTED 2, BULL_EXHAUSTED 2, CHOPPY_FLAT 1 |
| 2025-11-17 | 10 / 13 | -28.9 | HEALTHY_BEAR 4, CHOPPY_WEAK 3, STRONG_BULL 3, HEALTHY_BULL 2, BEAR_EXHAUSTED 1 |
| 2026-09-20 | 10 / 15 | -10.0 | CHOPPY_WEAK 8, STRONG_BEAR 2, STRONG_BULL 2, CHOPPY_FLAT 1, HEALTHY_BULL 1, BULL_EXHAUSTED 1 |
| 2024-03-02 | 10 / 14 | -1.8 | CHOPPY_WEAK 10, CHOPPY_FLAT 2, HEALTHY_BULL 1, HEALTHY_BEAR 1 |
| 2024-12-04 | 10 / 19 | +24.8 | CHOPPY_WEAK 5, STRONG_BULL 4, CHOPPY_FLAT 3, HEALTHY_BULL 3, STRONG_BEAR 2, HEALTHY_BEAR 1, BULL_EXHAUSTED 1 |
| 2026-03-31 | 9 / 10 | -38.2 | CHOPPY_FLAT 2, BEAR_EXHAUSTED 2, STRONG_BEAR 2, CHOPPY_WEAK 1, STRONG_BULL 1, BULL_EXHAUSTED 1, HEALTHY_BULL 1 |
| 2026-05-05 | 9 / 11 | -34.5 | STRONG_BULL 5, CHOPPY_FLAT 3, CHOPPY_WEAK 2, HEALTHY_BEAR 1 |

Label mix, cluster-day entries vs all entries: CHOPPY_WEAK 35 % vs 28 % · CHOPPY_FLAT 12 % vs 18 % · HEALTHY_BULL 9 % vs 13 % · STRONG_BULL 13 % vs 11 % · BULL_EXHAUSTED 3 % vs 4 % · HEALTHY_BEAR 12 % vs 14 % · STRONG_BEAR 8 % vs 10 % · BEAR_EXHAUSTED 6 % vs 3 %

**1D theory tests within each label (DESIGN, blocked avg / kept avg · blocked N; cells with blocked N < 30 shown as —):**

| Test | CHOPPY_WEAK | CHOPPY_FLAT | HEALTHY_BULL | STRONG_BULL | BULL_EXHAUSTED | HEALTHY_BEAR | STRONG_BEAR | BEAR_EXHAUSTED |
|---|---|---|---|---|---|---|---|---|
| SIGFAIL_6H ge05 | +0.52 / +0.59 · 397 | +1.31 / +0.67 · 248 | +0.80 / +0.68 · 189 | +0.81 / +0.15 · 142 | +1.10 / +0.57 · 36 | -0.26 / +0.72 · 209 | +0.29 / +1.34 · 138 | +0.96 / -0.55 · 47 |
| SIGFAIL_6H med | +0.28 / +0.69 · 302 | +0.98 / +0.94 · 195 | +0.90 / +0.66 · 136 | +0.92 / +0.17 · 113 | — | +0.03 / +0.40 · 150 | -0.01 / +1.32 · 104 | +1.35 / -0.50 · 35 |
| SIGFAIL_6H T3 | +0.17 / +0.64 · 155 | +0.63 / +1.03 · 102 | +1.56 / +0.59 · 67 | +0.91 / +0.33 · 56 | — | -0.31 / +0.40 · 75 | +0.34 / +1.03 · 61 | — |
| SIGFAIL_24H ge05 | +0.52 / +0.61 · 526 | +1.06 / +0.80 · 339 | +0.84 / +0.59 · 258 | +0.59 / +0.21 · 191 | -0.17 / +1.72 · 52 | -0.04 / +0.65 · 250 | +0.43 / +1.51 · 185 | +0.82 / -0.79 · 62 |
| SIGFAIL_24H med | +0.35 / +0.74 · 435 | +0.75 / +1.14 · 267 | +1.01 / +0.51 · 201 | +0.60 / +0.29 · 146 | +0.32 / +1.07 · 43 | +0.07 / +0.47 · 215 | +0.13 / +1.47 · 140 | +0.61 / -0.37 · 50 |
| SIGFAIL_24H T3 | +0.10 / +0.73 · 257 | +0.10 / +1.33 · 173 | +0.69 / +0.74 · 131 | +0.78 / +0.28 · 99 | — | +0.03 / +0.38 · 128 | +0.32 / +1.11 · 87 | +0.43 / -0.12 · 35 |
| ALTVOL_EXP gt1 | +0.48 / +0.61 · 364 | +1.55 / +0.59 · 215 | +0.88 / +0.61 · 203 | +0.09 / +0.83 · 192 | +1.92 / -0.66 · 56 | +0.34 / +0.23 · 215 | +1.01 / +0.75 · 195 | -0.53 / +0.58 · 58 |
| ALTVOL_EXP med | +0.41 / +0.68 · 415 | +1.17 / +0.79 · 242 | +0.80 / +0.66 · 225 | +0.04 / +0.98 · 204 | +1.32 / -0.15 · 63 | +0.46 / +0.10 · 238 | +0.86 / +0.97 · 211 | -0.32 / +0.48 · 66 |
| ALTVOL_EXP T3 | +0.46 / +0.60 · 250 | +0.99 / +0.94 · 149 | +1.31 / +0.45 · 151 | +0.02 / +0.74 · 152 | +2.01 / -0.28 · 46 | -0.06 / +0.45 · 153 | +1.25 / +0.56 · 162 | -0.34 / +0.28 · 46 |
| SPOT_LED gt0 | +0.17 / +1.02 · 514 | +0.88 / +1.03 · 280 | +1.28 / +0.19 · 228 | +0.58 / +0.26 · 177 | +1.18 / +0.01 · 65 | +0.20 / +0.39 · 265 | +0.66 / +1.15 · 166 | -1.21 / +1.48 · 64 |
| SPOT_LED med | +0.24 / +0.91 · 490 | +0.83 / +1.07 · 266 | +1.33 / +0.19 · 218 | +0.67 / +0.18 · 166 | +1.29 / -0.15 · 64 | +0.03 / +0.58 · 251 | +0.62 / +1.13 · 147 | -1.29 / +1.42 · 61 |
| SPOT_LED T3 | +0.24 / +0.73 · 328 | +0.85 / +1.01 · 178 | +1.50 / +0.40 · 139 | +1.09 / +0.03 · 126 | +1.44 / +0.44 · 32 | +0.11 / +0.39 · 177 | +0.73 / +0.97 · 91 | -1.90 / +0.94 · 38 |
| COIN_CROWD flag | +0.74 / +0.45 · 354 | +0.63 / +1.19 · 237 | +0.78 / +0.70 · 191 | +0.81 / +0.17 · 133 | +0.59 / +0.85 · 38 | -0.20 / +0.56 · 169 | +0.25 / +1.28 · 121 | +1.08 / -0.58 · 45 |
| COIN_CROWD flag | +0.76 / +0.51 · 191 | +0.20 / +1.17 · 127 | +1.04 / +0.65 · 98 | +0.39 / +0.43 · 70 | — | -0.34 / +0.42 · 82 | +0.89 / +0.91 · 64 | — |
| COIN_DI15 (+DI>−DI) gt0 | — | — | — | — | — | — | — | — |
| COIN_DI15 spread rising 1h gt0 | +0.76 / +0.41 · 404 | +0.88 / +1.02 · 253 | +0.67 / +0.78 · 206 | -0.07 / +0.83 · 158 | +1.52 / -0.01 · 51 | +0.29 / +0.28 · 192 | +1.33 / +0.62 · 132 | +0.91 / -0.44 · 43 |
| COIN_DI15 both flag | +0.76 / +0.41 · 404 | +0.88 / +1.02 · 253 | +0.67 / +0.78 · 206 | -0.07 / +0.83 · 158 | +1.52 / -0.01 · 51 | +0.29 / +0.28 · 192 | +1.33 / +0.62 · 132 | +0.91 / -0.44 · 43 |

**Top 2D cells within each label (DESIGN, blocked avg · N):**

| Cell | CHOPPY_WEAK | CHOPPY_FLAT | HEALTHY_BULL | STRONG_BULL | BULL_EXHAUSTED | HEALTHY_BEAR | STRONG_BEAR | BEAR_EXHAUSTED |
|---|---|---|---|---|---|---|---|---|
| sf24T3(>0.6) ∧ VSW15==0 | -0.05 · 227 | -0.06 · 149 | +0.93 · 120 | +1.00 · 91 | +0.88 · 27 | -0.31 · 113 | +0.40 · 78 | -0.22 · 28 |
| sf24T3(>0.6) ∧ ret_4h>0 | +0.00 · 253 | +0.14 · 172 | +0.69 · 131 | +0.78 · 99 | +1.00 · 29 | +0.00 · 126 | +0.18 · 85 | +0.15 · 34 |
| sf24T3(>0.6) ∧ lab_BULL_EXHAUSTED==0 | +0.10 · 257 | +0.10 · 173 | +0.69 · 131 | +0.78 · 99 | — | +0.03 · 128 | +0.32 · 87 | +0.43 · 35 |
| oi_chg_1h<=med(8.64) ∧ mcapT1(<=7.83e+07) | -0.99 · 120 | -0.33 · 81 | -0.82 · 68 | -1.98 · 47 | — | -0.21 · 53 | -0.21 · 48 | — |
| mcapT1(<=7.83e+07) ∧ pre1h_short<=med(0.84) | -0.57 · 133 | -0.22 · 79 | -1.40 · 62 | -1.84 · 47 | — | -0.46 · 59 | +0.09 · 56 | — |
| ret_1hT1(<=-0.176) ∧ mcapT1(<=7.83e+07) | -1.13 · 84 | -0.85 · 63 | -1.32 · 45 | -2.00 · 26 | — | -0.82 · 40 | +0.48 · 36 | — |
| fund_med>0 ∧ ret_4h>0 | +0.55 · 922 | +0.99 · 558 | +0.67 · 456 | +0.48 · 335 | +0.66 · 101 | +0.29 · 462 | +0.85 · 325 | -0.03 · 116 |
| RSI5==0 ∧ STOCH15==0 | +0.21 · 653 | +0.88 · 379 | +0.79 · 316 | +0.41 · 235 | +0.07 · 59 | +0.01 · 323 | +0.79 · 234 | -0.82 · 88 |


**Read (bot label).** The label rebuild matches the stamped label on 88 % of 1,324 real orders (the misses are near the ADX 18 / 28
and slope ±0.02 % borders; the engine reads a half-formed bar). EXHAUSTION is positive in every label. The weakest are BEAR_EXHAUSTED
(+0.04, 120 trades) and HEALTHY_BEAR (+0.28, 468): neither is close to the bar (P(avg<0) 0.46 / 0.17, WR 37 % vs 35 % breakeven), and
both are positive in the holdout. Cluster days carry a little more CHOPPY_WEAK (35 % vs 28 %) — CHOPPY_WEAK as a whole earns +0.56.
Within-label re-reads of the theory tests flip sign from label to label at N 35–500 (136 cells, no pattern holds across neighbours,
e.g. SIGFAIL hurts in HEALTHY_BEAR / STRONG_BEAR but helps in STRONG_BULL / BEAR_EXHAUSTED): that is the noise signature, not a rule.
The small-cap 2D cells are negative in almost every label in DESIGN, but positive in the holdout (section 4).

## 6. Replay 11-16 → 11-18 (HUMA, BLESS, ZKC, ZKP, PIPPIN, BREV)

Every 15m RSI14 70 → ≤ 70 cross on the six coins from 11-16 00:00 to 11-18 14:00 UTC (live 5m klines). These are raw crosses, not all
EXHAUSTION signals (the engine also needs the Frenzy window, vd ≥ 0.6366 and a falling spot share; "vd" is shown only where the study
cache has the bar). "Short P&L to now" = entry open vs the 11-18 14:00 close, no stop. SIGFAIL is n/a after the cross cache ends
(11-17 15:30).

| Coin | 15m cross (entry UTC) | RSI | in Frenzy cache (vd) | known fill | max rise 2 h | short P&L to 11-18 14:00 | SIGFAIL 6 h · 24 h (n) | ALTVOL | SPOT_LED pp | bot label | 15m DI spread · Δ1h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ZKC | 11-16 00:15 | 66 | n/a |  | +0.2 % | -17.5 % | n/a · 0.57 (14) | 0.66 | +0.23 | CHOPPY_WEAK | +24 · +5 |
| BLESS | 11-16 09:45 | 61 | 1.00 |  | +25.7 % | -8.1 % | 0.40 · 0.50 (14) | 0.82 | +0.56 | STRONG_BEAR | +49 · +27 |
| HUMA | 11-16 11:30 | 66 | n/a |  | +1.5 % | -8.2 % | n/a · 0.50 (12) | 0.91 | -1.90 | STRONG_BULL | +30 · +5 |
| ZKP | 11-16 15:00 | 67 | n/a |  | +0.3 % | -12.7 % | 0.75 · 0.62 (13) | 1.07 | -0.49 | STRONG_BULL | +22 · +19 |
| HUMA | 11-16 17:30 | 66 | 0.72 | EXH sim trade: SL −6.28 % | +5.3 % | +13.5 % | 0.80 · 0.58 (12) | 0.88 | -0.75 | HEALTHY_BULL | +25 · -28 |
| BLESS | 11-16 17:45 | 61 | 0.65 |  | +12.3 % | +36.0 % | 0.67 · 0.55 (11) | 0.88 | -0.75 | HEALTHY_BULL | +6 · -6 |
| HUMA | 11-16 20:00 | 68 | 0.51 |  | +17.2 % | +19.0 % | 1.00 · 0.64 (14) | 0.81 | -1.12 | CHOPPY_WEAK | +24 · +13 |
| HUMA | 11-16 22:45 | 64 | 0.57 |  | +3.9 % | +25.6 % | n/a · 0.69 (13) | 0.75 | -0.80 | HEALTHY_BULL | +17 · -8 |
| ZKC | 11-17 14:15 | 64 | n/a |  | +1.0 % | -17.7 % | n/a · 0.82 (11) | 0.94 | +0.49 | CHOPPY_FLAT | +30 · +12 |
| ZKP | 11-17 14:15 | 68 | n/a |  | +2.7 % | -7.9 % | n/a · 0.82 (11) | 0.94 | +0.49 | CHOPPY_FLAT | +34 · +9 |
| ZKP | 11-17 15:30 | 61 | n/a |  | +0.2 % | -7.7 % | 0.67 · 0.75 (12) | 0.86 | +0.29 | HEALTHY_BULL | +22 · -7 |
| BREV | 11-17 17:45 | 70 | n/a |  | +5.6 % | +10.1 % | n/a · n/a (0) | 0.92 | -1.08 | HEALTHY_BEAR | +19 · -5 |
| BREV | 11-17 19:00 | 68 | n/a |  | +15.9 % | +10.5 % | n/a · n/a (0) | 1.04 | -2.01 | CHOPPY_WEAK | +19 · +1 |
| BREV | 11-17 23:00 | 64 | n/a |  | +5.5 % | +18.6 % | n/a · n/a (0) | 0.76 | +0.43 | STRONG_BULL | +23 · -4 |
| ZKP | 11-18 01:45 | 65 | n/a |  | +3.1 % | -5.5 % | n/a · n/a (0) | 0.87 | -0.70 | HEALTHY_BEAR | +24 · +8 |
| BREV | 11-18 02:00 | 48 | n/a |  | +9.4 % | +18.0 % | n/a · n/a (0) | 0.90 | -1.07 | BEAR_EXHAUSTED | +4 · -10 |
| BLESS | 11-18 04:45 | 53 | n/a |  | +2.7 % | -7.1 % | n/a · n/a (0) | 0.64 | -1.18 | CHOPPY_WEAK | +7 · -18 |
| ZKP | 11-18 05:45 | 65 | n/a |  | +14.2 % | +0.1 % | n/a · n/a (0) | 0.61 | -0.99 | HEALTHY_BULL | +27 · +22 |
| ZKP | 11-18 06:15 | 68 | n/a |  | +13.1 % | +5.3 % | n/a · n/a (0) | 0.63 | -1.15 | HEALTHY_BULL | +38 · +27 |
| ZKC | 11-18 06:30 | 66 | n/a |  | +11.6 % | -15.1 % | n/a · n/a (0) | 0.63 | -1.15 | CHOPPY_FLAT | +28 · +31 |
| ZKP | 11-18 07:30 | 66 | n/a |  | +9.5 % | +8.4 % | n/a · n/a (0) | 0.68 | -0.63 | CHOPPY_WEAK | +20 · -10 |
| ZKC | 11-18 07:30 | 64 | n/a |  | +24.5 % | -14.1 % | n/a · n/a (0) | 0.68 | -0.63 | CHOPPY_WEAK | +21 · -7 |
| BLESS | 11-18 07:45 | 57 | n/a |  | +0.6 % | +2.4 % | n/a · n/a (0) | 0.68 | -0.63 | HEALTHY_BEAR | +16 · +6 |
| ZKP | 11-18 08:30 | 70 | n/a | EXH live paper fill: TP +4.94 % | +4.0 % | +11.4 % | n/a · n/a (0) | 0.84 | -0.24 | CHOPPY_WEAK | +22 · +2 |
| ZKP | 11-18 09:15 | 59 | n/a |  | +0.4 % | +9.4 % | n/a · n/a (0) | 0.92 | -0.49 | CHOPPY_WEAK | +13 · -10 |
| ZKC | 11-18 10:45 | 66 | n/a |  | +4.3 % | +2.4 % | n/a · n/a (0) | 0.84 | -0.06 | HEALTHY_BEAR | +24 · -15 |
| PIPPIN | 11-18 12:15 | 65 | n/a |  | n/a | +3.3 % | n/a · n/a (0) | 0.86 | +0.03 | CHOPPY_FLAT | +17 · -11 |

**Read.** The two days were quiet for the market (ALTVOL 0.61 – 1.07, mostly below its median 0.96; spot share mostly falling).
SIGFAIL-24h was high on 11-17 (0.75 – 0.82, top tercile) — and the ZKC / ZKP crosses at 11-17 14:15 – 15:30 then fell 8 – 18 %:
the "bad day" switch would have blocked winners. The crosses that failed (HUMA 17:30 / 20:00, BLESS 17:45, BREV 17:45 – 02:00) had
middling values on every indicator; HUMA 17:30 (the stopped sim trade) shows SIGFAIL-24h 0.58, ALTVOL 0.88, spot share falling,
+DI spread already falling over the last hour. Nothing at those entries said "don't". ZKP 08:30 (the live TP +4.94 %) looked the same.

## 7. What it means

- **Why many stops in one day.** EXHAUSTION stops 59 % of the time (wins +8.4 %, losses −4.5 %). If trades were independent, 9 of the
  66 days with ≥ 10 trades would have ≥ 75 % stops; we see 19. So about half the cluster days are plain chance on a busy day, and the
  other half come from correlation: Frenzy coins pump together and keep pumping together. The
  theory explains the mechanism (new buying arrives after the roll-over), but none of the measurable causes is visible AT ENTRY:
  not the market inflow, not the pattern failing on other coins, not crowded shorts, not +DI, not BTC's regime.
- **The failure is decided after the entry.** On cluster days the extra buying shows up in the next hours, after our signal. That is
  why entry-time indicators cannot see it. This is the same conclusion as the SQUEEZER addendum, now with the theory indicators,
  117,841 2D cells and the bot's own regime labels added.
- **Switching off costs more than it saves.** Every candidate that removes cluster-day stops also removes winners. The sleeve's edge is
  wide (+0.62 %/trade DESIGN, +0.79 HOLDOUT) and every bad-looking bucket is still profitable. Stop clusters are the price of the
  sleeve. Size, not a switch, is the lever (per-stop cost × the realistic same-day stop count; see SIZING_EXH_FLATBASE).
- **The weak "something".** Negative 2D cells stay negative in the holdout 25 % of the time vs 12 % for any cell. So small real
  effects may exist, too small for one rule to be found and trusted. The closest candidate is OBSERVE below.

## 8. Verdict

**NO** — no switch, no filter. Pre-registered bar not met by any indicator, cell, regime or label.

**OBSERVE (not armed, frozen now, never re-fit):** `EXH_OBS_SSC_OIVOL` = spot_share_chg > −0.0366 (spot share NOT falling much)
∧ OI / 24 h volume > 0.128 (top tercile). Theory fit: the move is less leverage-driven than our signal assumes AND positioning is
heavy relative to trading. DESIGN 512 · 347 days · WR 30 % · −0.49 %/trade (passes the window bar, 4 of 6 halves negative) but family
p 0.45 (luck level). HOLDOUT 119 · 68 days · −0.16 vs +1.00 kept (P(avg<0) 0.63). Read at each EXHAUSTION batch review on real
paper/live fills. Gate to consider arming: ≥ 30 NEW fills over ≥ 10 days, blocked WR < 35 %, avg < 0 with day-bootstrap P ≥ 0.95,
no day ≥ 50 % of the loss. Revert gate if armed later: first 20 blocked counterfactual fills Σ ≥ 0 → off.

## 9. Next step

1. Nothing to deploy. Keep EXHAUSTION as configured.
2. If the operator wants the line tracked, add `EXH_OBS_SSC_OIVOL` to the EXHAUSTION scout (stamps ssc and OI / vol at entry;
   both are already computed in the study rows; OI needs the metrics feed live).
3. The only route left for "bad days" is intra-trade, not entry: an early-exit rule when the market keeps lifting AFTER entry
   (e.g. ≥ 50 % of open EXHAUSTION shorts underwater after 1 h). That is an exit study (ENTRY × EXIT rule, live-stopped cohort only,
   two-sided), not a filter. Not run here.

## 10. Assumptions and blind spots

- Trades are the simulated grid cell `B|tp10|S2|24` (slippage modelled as in EXH_EXIT_GRID). Not the live exit (TP min(3×ATR, 5 %)).
  The live exit has a different win/loss shape; the "no filter" conclusion is about entries and should carry, but was not re-run.
- SIGFAIL universe = Frenzy-window crosses with vd ≥ 0.40 and vol24 ≥ $20 M (the study cache). Crosses at lower vd are not in it.
  Coverage: 6 h value exists for 66 % of trades (else "thin" → kept), 24 h for 92 %.
- ALTVOL universe grows from ≈ 180 to ≈ 530 perps; matched pairs per window handle it, but new listings change the mix.
- SPOT_LED uses a fixed 22-alt large-cap basket, not the Frenzy coins. Spot timestamps in µs from 2025 were converted.
- mcap = today's circulating supply × entry price — NOT point-in-time. The small-cap cells may be biased by that.
- OI / long-short / book columns come from data.binance.vision metrics (create_time ≤ entry − 5 min); book depth has partial coverage.
- Bot label parity 88 % (forming-bar timing cannot be replayed). The 12 % mismatches sit at thresholds; reads by label are slightly blurred.
- Alt regime (equal-weight alt index) had no variance → untested. BTC regime ≈ calendar (no bear in 2024).
- 2D null shuffles day blocks of outcomes; it keeps within-day clustering but not multi-day runs (alt seasons last weeks). That makes
  the null slightly too easy to beat, so the "no cell passes" verdict is conservative in the right direction.
- Several 2D variables are near-constant at the signal (STOCH15, RSI5, ret_15m, c15_spread > 0, has_spot ≡ ssc defined). Cells built
  on them are "block almost everything" cells; they are scored but meaningless.
- Replay rows after the study caches end use live klines (fapi, generic UA, < 150 weight/min); SIGFAIL after 2026-11-17 13:30 is n/a.
- 2D and 1D tests were re-run once after the DI / label addendum (family enlarged, same cuts); no threshold was changed after a read.

![EXH bad days — SIGFAIL vs day P&L · best cell equity · regime](EXH_BADDAY_2026-11-18.png)
