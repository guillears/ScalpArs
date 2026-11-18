# UNIFIED SHORT — is there a high-frequency SHORT on pump coins? (2026-11-18)

Research only. Nothing in the bot changed and nothing was committed.
**Pre-registered:** `UNIFIED_SHORT_2026-11-18_PREREG.md`, frozen 15:45 UTC before any result. Addendum A ("drops down the list") was
frozen at 15:52 and addendum B ("point of no return below the average price") at 16:52. Both were frozen before any result on them.
An interim file went out at 16:05 (`_INTERIM.md`).

## Question
The operator: "With ~42 pumps and dumps per day, it's not acceptable that we only have 1–3 trades a day. There is something for SHORTs
you are not finding — maybe a better COMBINATION of the filters we already have."

Can we find a SHORT on pump / Frenzy coins with **≥ 5 trades a day** and a positive, consistent edge after real costs including funding?

## Plain answer
1. **No. Nothing passes the bar. No rule and no model reaches 5 trades a day with an edge after costs.**
   - We searched 45,571 rules (every trigger, every feature cut, every pair of them, greedy triples).
   - Of the 38,228 rules that trade ≥ 5 times a day on DESIGN, **only 88 are positive at all, and none is significant** (best day-clustered t = 1.85).
   - The best of those DESIGN rules **lose money in the older holdout** (2023-07 → 2024-12).
2. **The "~42 pumps" are real, but most of them are not shortable for profit.** At any moment about 45 coins carry the Frenzy flag (since mid-2025).
   - Shorting a random Frenzy bar loses **−0.25 % per trade**, in every half-year.
   - Every trigger we have ever tested, used alone, loses **−0.06 to −0.29 %**.
   - The only exceptions are EXHAUSTION (+0.29), the second-peak RSI cross (+0.11) and the "out of the top-5 list" cell (+0.07).
3. **The models agree.** We ran two walk-forward models (an additive ridge model and boosted trees, trained only on the past).
   - They rank trades in the right order: a higher score means a better trade.
   - At ≥ 5 trades a day their best cell is +0.07 %/trade (GBM top 2 %, 7.4/day). It has two negative half-years and is negative at 0.20 % slippage.
   - GBM's top 1 % (2.8/day) is +0.22 with every half-year > 0, mostly NOT overlapping EXHAUSTION. But it fails Kelly, best-5 %-out and the drawdown limit, and it is no more frequent than EXHAUSTION.
4. **The operator's two theories were tested exactly as stated.**
   - **"Drops down the FRENZY list":** the rank cells lose −0.02 to −0.21 %/trade.
     - The variables do not separate winners from losers.
     - EXHAUSTION's own trades fire 90 % of the time while the coin is still at its BEST rank.
   - **"Point of no return below the average price":**
     - The deeper a coin falls below its VWAP, the less often it reclaims it (84 % within 24 h at 0 % below → 33 % at 10 % below).
     - The typical (median) coin keeps falling (+1.3 to +3.0 % at 24 h).
     - But the AVERAGE further move is only +0.2 to +0.6 % at 24 h and ≈ 0 at 72 h, because about half of the coins squeeze +10 % before falling another −10 %.
     - The 135 tradable V cells (depth × count gate × exit) are −0.25 to +0.05 %/trade.
5. **EXHAUSTION stays the highest-frequency short with an edge** (≈ 3.3 trades/day, +0.29 %/trade at this study's TP5 exit, +0.43 at its live exit). It was reproduced bar for bar (4,263 / 4,263).
6. **One real lead, but it is a regime, not a filter.** While BTC's daily EMA50 sat below its EMA200 (2025-11-18 → 2026-09-13, one ~300-day window), broad Frenzy shorts on crowded-long coins (global L/S ≥ 1.45) paid **+0.28 %/trade at ~27 trades a day** with a TP 10 % / fixed 8 % stop exit. In BTC-bull days the same rule paid −0.30.
   - That is **one window**, so it is one observation (the window-units rule).
   - The exit was picked after seeing results.
   - The regime switched off on 2026-09-13.
   - It goes in as an OBSERVE line, not a build.

## Theory (written before the test, in the prereg)
- Prior work said the exhaustion edge is a **state** edge (volume still high, spot share falling), not a trigger edge.
- If that is right, a combination search will mostly find state variables again. The frequency ceiling is then the number of coins in that state per day, not the number of triggers.
- Squeezes are the short's tail risk. A combination can only add frequency if another state (funding, OI, crowding, BTC regime) carries edge of its own.
- **Result: the theory held.** No second state was found that works in every era. The one candidate (BTC daily bear) is a market-wide regime, seen in one window.

## Pool and parity
- **Universe:** every bar-rule spike, 2023-05 → 2026-11-17. That is 9,959 spikes on 645 pairs, with no later shortlist selection.
  - Window = spike → +72 h. Volume over 24 h must be ≥ $20 M. BTC, ETH and the blacklist are excluded.
- **Pool:** 1,883,126 candidate rows.
  - Every complete 15m close in a window, plus every 5m RSI cross.
  - 30 trigger bits and 7 rank-drop (R) bits.
  - ≈ 100 entry features.
- **Exits:**
  - The main exit is TP 5 % maker with a stop at min(1.5 × ATR15, 6 %), gap-aware taker, 24 h ("T5A").
  - Also TP 3 / 5 / 10 % × stop A / fixed 8 %, and EXHAUSTION's live exit.
- **Costs:** taker 0.05 %, slippage 0.10 % per taker leg (stress 0.20 / 0.40 %), maker 0.02 %, real funding including hourly coins. One position per pair.

| parity check | result |
|---|---|
| EXHAUSTION bit vs the EXHAUSTION_EXPAND all-spikes build | **4,263 / 4,263 identical rows** |
| 36 real FLATBASE_SHORT fills | 35 sit on a pool row at the exact entry bar. 26 of those carry the FB4 / FBC4 trigger bit. The other 9 are the engine's re-entry / turnover route or a different spike anchor. |
| 1 real EXHAUSTION_SHORT fill (ZKP 11-18 08:30) | after the data end (11-17), so not checkable |

**Feature coverage:**
- vol_decay, funding, breadth, BTC, regime and rank: 100 % in every half-year.
- OI, top-trader L/S and taker: 32–45 % in 2023-H2 / 2024-H1, then 100 %.
- Premium: 0–21 % before 2024-H2.
- Spot share: 56–93 %, because no-spot coins are unscored.
- Market cap proxy: 71–90 %.

So rules built on OI or crowding are only partly testable in the older holdout. This is said again under each candidate.

## Method
1. **Rules (DESIGN 2025-01 → 2026-06 only):**
   - 309 conditions: 37 triggers + "any trigger" + quintile cuts and levels of every feature.
   - Exhaustive 1D, every 2D pair (45,571 rules), and greedy 3D from the top 50 2D.
   - Pre-registered candidate gate: ≥ 5/day and day-clustered t ≥ 2.
   - Family-wise null: the max t over every ≥ 5/day rule, on 50 circular time-shifts of the outcome vector. Its 95 % point is t = 1.74.
2. **Models:**
   - GAM (ridge on decile one-hot bins of every feature + trigger bits), refit monthly.
   - Boosted depth-2 trees (numpy, 150 trees), refit quarterly. This is a deviation, for runtime.
   - Both are expanding-window. Training rows must have exited before the predicted month starts. The first traded month is 2024-01.
   - The thresholds are training-window quantiles, so they are causal.
3. **Pass bar:**
   1. ≥ 5/day.
   2. Every half-year > 0.
   3. Best-5 %-out > 0.
   4. > 0 at 0.20 % slippage.
   5. Family-wise p < 0.05.
   6. Beats random shorts in the same state (same episode, same non-trigger conditions).
   7. Kelly at the 95 % Wilson floor with n = days > 0.
   8. Both holdouts > 0.
   9. The union with EXHAUSTION raises the daily compound at 0.18 × equity per trade without deepening the max drawdown by > 5 points.
4. **Holdouts:**
   - H-OLD = 2023-07 → 2024-12. It was read before by other studies, for other rules.
   - H-NEW = 2026-07 → 2026-11-17.
   - 2023-H1 is not buildable (5m data start 2023-05).

## Frontier
![frontier](UNIFIED_SHORT_2026-11-18_frontier.png)

- **Left:** all 45,571 DESIGN rules (grey) and the single triggers (blue). Above 5 trades/day (red line) the cloud sits below zero. Its top edge is ≈ +0.17 with t < 2.
- **Right:** the best rules on all blocks, and the two walk-forward models at 7 thresholds. More trades always means a worse average.

| trades/day | best thing we have at that frequency | avg %/trade | note |
|---|---|---|---|
| ≈ 3 | **EXHAUSTION (live rule)** | **+0.29** | every block > 0 except 25-H1 −0.08 |
| ≈ 3 | RSI15 cross ∧ spot share chg ≤ −0.039 | +0.23 | an EXHAUSTION variant, no frequency gain |
| ≈ 4 | VSW red candle ∧ top-trader position L/S falling over 4 h | +0.17 | DESIGN-picked, fw p 0.42 |
| ≈ 5 | P2R second-peak RSI cross | +0.11 | DESIGN ≈ 0 |
| 2.8 | GBM top 1 % | +0.22 | walk-forward, every half > 0, fails Kelly / best-5 % |
| 6.6 | GAM top 1 % | −0.01 | walk-forward |
| 7.4 | GBM top 2 % | +0.07 | walk-forward, 25-H1 −0.20 |
| 6–8 | VWAP depth cells (V) | −0.25 … +0.05 | addendum B |
| 6–8 | rank-drop cells (R) | −0.21 … −0.17 | addendum A |
| 11 | GAM top 2 % | −0.00 | walk-forward |
| 16 | GBM top 5 % | +0.02 | walk-forward |
| 20–40 | best ≥ 20/day rules / model top 5–10 % | −0.05 … −0.15 | |
| 190 | every Frenzy 15m bar | −0.25 | the baseline |

## Every trigger alone (one position per pair, T5A, all blocks)
The full table is in `_INTERIM.md` §3 and `_triggers.csv`. Summary:
- **Positive:**
  - EXHAUSTION +0.29 (4.0/day on DESIGN).
  - P2R +0.11 (one negative half).
  - R out-of-top-5 with ≥ 2 newer above +0.07 (2.5/day; +0.79 in 26-H2 carries it).
- **About zero:**
  - VSW −0.06.
  - TIME2 −0.07.
  - RSI15 cross without filters −0.08.
- **Clearly negative:**
  - FB4 −0.11, VWAP fail −0.10, T1 −0.18.
  - Divergence −0.25, double top −0.23, wait-and-see −0.17 / −0.25.
  - Time +4 … +24 h −0.13 … −0.28, VWAP stretch −0.29, any trigger −0.22.

## The 5 closest candidates (nothing passes; full per-period tables)
The candidate gate was empty (no ≥ 5/day rule with t ≥ 2). So the 9-item bar was run on the 25 best DESIGN rules plus the two best low-frequency triggers.
These are the five closest, with EXHAUSTION for reference. Avg is net %/trade at T5A and 0.10 % slippage; (N) is trades in that half-year.

| rule | /day | WR | avg | 23-H2 | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | H-OLD | DESIGN | H-NEW | best-5 % out | @0.20 | @0.40 | Kelly | random diff (p) | fw p | own %/day @0.18 | own max DD | union %/day | union max DD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **C1** RSI15 cross ∧ spot share chg ≤ −0.039 | 3.0 | 49.6 | +0.23 | +0.20 (214) | −0.03 (399) | +0.27 (464) | +0.27 (469) | +0.36 (784) | +0.10 (823) | +0.40 (523) | +0.14 | +0.24 | +0.40 | −0.01 | +0.08 | −0.22 | +0.001 | +0.55 (0.00) | 0.00 | +0.113 | −36 % | +0.145 | −48 % |
| **C2** VSW red candle ∧ top-trader position L/S 4 h chg ≤ −0.011 | 3.9 | 46.8 | +0.17 | +0.74 (206) | +0.31 (345) | +0.10 (510) | +0.01 (653) | +0.18 (1270) | −0.01 (1068) | +0.35 (759) | +0.29 | +0.08 | +0.35 | −0.08 | +0.01 | −0.30 | −0.010 | +0.25 (0.00) | 0.42 | +0.104 | −26 % | +0.233 | −42 % |
| **C3** funding ≤ −0.081 % ∧ top-trader position L/S ≥ 1.43 | 3.8 | 49.8 | +0.10 | n/a (0) | +3.39 (4) | +0.84 (28) | +0.08 (247) | +0.00 (1824) | +0.19 (1988) | +0.05 (642) | +1.16 | +0.10 | +0.05 | −0.15 | −0.05 | −0.36 | −0.050 | +0.29 (0.00) | 0.18 | +0.070 | −40 % | +0.201 | −31 % |
| **C4** VSW red candle ∧ RSI 1h ≥ 55 | 4.1 | 47.3 | +0.12 | +0.40 (436) | +0.19 (464) | +0.06 (516) | −0.17 (612) | +0.20 (1263) | +0.02 (1032) | +0.23 (753) | +0.21 | +0.06 | +0.23 | −0.13 | −0.03 | −0.34 | −0.019 | +0.13 (0.02) | 0.74 | +0.078 | −31 % | +0.204 | −45 % |
| **C5** global L/S ≥ 1.82 ∧ volume share ≥ 6 % (best DESIGN rule) | 4.9 | 47.0 | +0.04 | +0.19 (212) | −0.17 (776) | −0.37 (1067) | +0.06 (776) | +0.19 (1384) | +0.23 (1272) | +0.23 (593) | **−0.24** | +0.17 | +0.23 | −0.21 | −0.11 | −0.42 | −0.047 | +0.05 (0.19) | 0.04 | +0.016 | −69 % | +0.181 | −53 % |
| *EXHAUSTION (live rule)* | 3.3 | 53.0 | +0.29 | +0.38 (312) | +0.37 (475) | +0.21 (472) | −0.08 (446) | +0.35 (884) | +0.07 (850) | +0.71 (687) | +0.31 | +0.15 | +0.71 | +0.05 | +0.15 | −0.15 | +0.011 | +0.00 (0.47) | — | +0.164 | −39 % | +0.164 | −39 % |

| rule | 1 ≥ 5/day | 2 every half > 0 | 3 best-5 % out | 4 @0.20 | 5 fw p | 6 beats random | 7 Kelly | 8 holdouts | 9 union | passed | fails by |
|---|---|---|---|---|---|---|---|---|---|---|---|
| C1 | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 5/9 | 3.0/day (−2.0) · 24-H1 −0.03 · best-5 % −0.01 · union −0.02 %/day and DD −8.3 pts |
| C2 | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | 4/9 | 3.9/day (−1.1) · 26-H1 −0.01 · best-5 % −0.08 · fw p 0.42 · Kelly −0.010 |
| C3 | ✗ | ✓* | ✗ | ✗ | ✗ | ✓ | ✗ | ✓* | ✓ | 4/9 | 3.8/day · *untestable before 2025 (crowding data 33–45 % coverage; 0–28 trades)* · @0.20 −0.05 · Kelly −0.05 · gone after removing its 10 best days (−1.6 % summed) |
| C4 | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✗ | 2/9 | 25-H1 −0.17 · @0.20 −0.03 · fw p 0.74 |
| C5 | ✓ (DESIGN 6.3; all 4.9 ✗) | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | 1/9 | H-OLD −0.24 · 78 % of its profit from 2 pairs (FOLKS, BSB) · −3.3 % summed without its 10 best days |
| EXHAUSTION | ✗ | ✗ | ✓ | ✓ | — | ✗ | ✓ | ✓ | — | 4/9 | 3.3/day · 25-H1 −0.08 · random-in-state equal (the state is the edge) |

**Per-pair / per-day concentration.**
- C1 and C2 are spread out:
  - The top 2 pairs carry 11–14 % of the profit.
  - They stay positive without their 10 best days (C1 +585 % summed, C2 +477 %).
- C3 and C5 are not:
  - They are window-driven.
  - Without their 10 best days they turn negative.

**Reading the candidates:**
- **C1** is EXHAUSTION with a tighter spot cut and no volume-decay gate. It is not new frequency (3.0/day). Its trades overlap EXHAUSTION's.
- **C2** is the only genuinely new combination that looks clean across eras:
  - +0.29 in H-OLD, +0.35 in H-NEW, spread over many pairs.
  - The union with EXHAUSTION lifts the daily compound from +0.164 to +0.233 %/day at 0.18 × equity.
  - Max drawdown goes −39 → −42 %.
  - But it was the top of 38,000 rules on DESIGN (+0.08 there, fw p 0.42), and it fails best-5 %-out and Kelly. It is a lead, not a find.

## The models (walk-forward 2024-01 → 2026-11, never trained on the month they trade)
"Top X %" is the score threshold taken from the training window. The realised frequency is in the /day column. Avg is net %/trade at T5A, 0.10 % slippage. There is no 2023-H2 here: it is training data only.

| model | threshold | /day | WR | avg | 24-H1 | 24-H2 | 25-H1 | 25-H2 | 26-H1 | 26-H2 | best-5 % out | @0.20 | Kelly | vs random in same episode | trades shared with EXH | union %/day @0.18 (EXH alone +0.173) | union max DD (EXH −39 %) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GAM | top 10 % | 36.9 | 42.7 | −0.06 | +0.06 | −0.19 | −0.04 | −0.01 | −0.12 | +0.03 | −0.32 | −0.21 | −0.056 | +0.22 | 365 | −0.746 | −100 % |
| GAM | top 5 % | 22.8 | 42.3 | −0.05 | +0.10 | −0.19 | −0.06 | −0.02 | −0.09 | +0.05 | −0.31 | −0.21 | −0.055 | +0.21 | 217 | −0.253 | −99 % |
| GAM | top 2 % | 11.5 | 42.6 | −0.00 | +0.12 | −0.06 | −0.09 | +0.05 | −0.08 | +0.11 | −0.26 | −0.16 | −0.048 | +0.26 | 104 | +0.076 | −82 % |
| GAM | top 1 % | 6.6 | 42.6 | −0.01 | +0.31 | −0.13 | −0.33 | +0.13 | −0.11 | −0.00 | −0.27 | −0.17 | −0.053 | +0.26 | 51 | +0.105 | −72 % |
| GBM | top 10 % | 27.8 | 46.5 | +0.01 | +0.36 | −0.05 | −0.24 | +0.20 | −0.08 | +0.05 | −0.25 | −0.15 | −0.050 | +0.29 | 303 | −0.087 | −98 % |
| GBM | top 5 % | 16.1 | 46.9 | +0.02 | +0.42 | +0.01 | −0.33 | +0.16 | −0.07 | +0.15 | −0.23 | −0.13 | −0.053 | +0.28 | 185 | +0.079 | −95 % |
| **GBM** | **top 2 %** | **7.4** | 46.3 | **+0.07** | +0.48 | +0.05 | **−0.20** | +0.19 | **−0.01** | +0.32 | −0.18 | −0.08 | −0.071 | +0.33 | 89 | +0.184 | **−80 %** |
| **GBM** | **top 1 %** | **2.8** | 46.4 | **+0.22** | +0.93 | +0.08 | +0.01 | +0.18 | +0.15 | +0.72 | −0.03 | +0.07 | −0.075 | +0.50 | 37 | **+0.261** | −47 % |

Lower thresholds (top 20–50 %, 42–104/day) are −0.05 … −0.16. Full table: `_model.csv`.

- **Both models rank correctly.** Every step up the score raises the average, and they beat random entries in the same episode by +0.14 … +0.50. But random entries there average −0.27.
- **At ≥ 5 trades a day the best model cell is GBM top 2 %:** 7.4/day, +0.07. It fails 6 of the items:
  - every-half (25-H1 −0.20, 26-H1 −0.01);
  - best-5 %-out −0.18;
  - @0.20 −0.08;
  - Kelly −0.07;
  - the union's max drawdown −80 %.
- **GBM top 1 % is the cleanest out-of-sample cell in the study:** 2.8/day, +0.22, every half-year > 0, and only 37 trades shared with EXHAUSTION, so it is mostly additive.
  - But it fails ≥ 5/day, best-5 %-out (−0.03), Kelly (−0.075) and the drawdown limit (union −47 % vs −39 %, 3 points over).
  - It is not a frequency answer. It is a second EXHAUSTION-sized lead (see OBSERVE line 3).

## Addendum A — "drops down the FRENZY list" (operator theory)
Rank = the coin's 24 h volume rank among all flagged coins at that 15m close.

| cell | trades | /day | WR | avg | H-OLD | DESIGN | H-NEW | best-5 % | @0.20 | mcap ≤ $50 M | mcap > $50 M | mcap n/a |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R_D2 rank drop ≥ 2, ≥ 1 newer above | 7,591 | 6.1 | 36.9 | −0.21 | −0.08 | −0.28 | −0.17 | −0.48 | −0.37 | −0.12 | −0.17 | −0.43 |
| R_D3 drop ≥ 3 | 7,339 | 5.9 | 34.8 | −0.19 | −0.09 | −0.27 | −0.04 | −0.46 | −0.36 | −0.30 | −0.12 | −0.25 |
| R_D5 drop ≥ 5 | 6,362 | 5.2 | 33.2 | −0.17 | −0.09 | −0.23 | −0.06 | −0.44 | −0.34 | −0.33 | −0.12 | −0.08 |
| R_D3N2 drop ≥ 3, ≥ 2 newer above | 7,273 | 5.9 | 34.9 | −0.17 | −0.11 | −0.25 | +0.04 | −0.43 | −0.34 | −0.32 | −0.06 | −0.25 |
| R_K3 out of top-3 | 1,862 | 1.5 | 38.6 | −0.14 | +0.02 | −0.31 | +0.12 | −0.40 | −0.30 | +0.04 | −0.20 | −0.09 |
| R_K5 out of top-5 | 2,503 | 2.0 | 38.8 | −0.02 | −0.14 | −0.11 | +0.59 | −0.28 | −0.18 | +0.05 | +0.07 | −0.34 |
| R_K5N2 out of top-5, ≥ 2 newer above | 2,465 | 2.0 | 39.0 | +0.07 | −0.06 | −0.05 | **+0.79** | −0.19 | −0.10 | +0.13 | +0.18 | −0.31 |

- **The rank-drop cells lose in every block and every market-cap bucket.**
  - Their win rate is 33–37 %.
  - When a coin slides down the list, the price usually has *already* dropped. The 1.5 × ATR stop then sits inside the noise of a dead-cat bounce.
- **The out-of-top-K cells are ≈ 0 overall.** Their only positive block is H-NEW (+0.6 / +0.8, 2026-07 → now). That is the BTC-bear-to-bull transition, a regime read.
- **Inside EXHAUSTION the rank variables do not help.**
  - 3,719 of EXHAUSTION's 4,126 trades fire while the coin is still at its best rank (+0.34).
  - Only 40 fire after a drop ≥ 7 (+0.23).
  - "Rank ≥ 21" (+0.37, 704 trades) vs "rank ≤ 5" (+0.26, 1,042) is within noise.
- **Variables on all 15m state rows:** rank ≤ 5 −0.26 vs ≥ 21 −0.27; drop ≤ 0 −0.24 vs ≥ 7 −0.22; vshare_rel ≤ 0.35 −0.18 vs ≥ 1 −0.24. That is no separation, in either market-cap bucket.
- The rank variables were also in the 2D search and the models. They did not produce a candidate. C5's "volume share ≥ 6 %" is the opposite of the theory (it shorts coins at the TOP of the list), and it fails the holdout.
- Full table: `_rank.csv`.

## Addendum B — "point of no return below the average price" (operator theory)
**(A) No-return curve.** This is the first 15m close X % below the spike-anchored VWAP, 8,509 episodes. The short move is measured from that close; positive = the short wins.

| X below VWAP | episodes | reclaims VWAP within 6 h | 24 h | 72 h | median move +24 h | **MEAN move +24 h** | median +72 h | **MEAN +72 h** | worst 10 % at 24 h | +10 % squeeze before −10 % |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 % | 8,509 | 74 % | 84 % | 90 % | +2.98 | +0.46 | +4.51 | +0.32 | −15.5 | 49 % |
| 1 % | 8,436 | 65 % | 79 % | 87 % | +2.78 | +0.44 | +4.30 | +0.25 | −15.0 | 48 % |
| 2 % | 8,298 | 55 % | 73 % | 83 % | +2.49 | +0.41 | +4.06 | +0.31 | −14.5 | 49 % |
| 3 % | 8,046 | 45 % | 67 % | 79 % | +2.19 | +0.47 | +3.90 | +0.35 | −14.2 | 49 % |
| 4 % | 7,642 | 36 % | 60 % | 74 % | +2.11 | +0.61 | +3.82 | +0.50 | −13.5 | 47 % |
| 6 % | 6,513 | 24 % | 49 % | 65 % | +1.85 | +0.49 | +3.68 | +0.01 | −13.5 | 48 % |
| 8 % | 5,314 | 17 % | 39 % | 58 % | +1.55 | +0.43 | +3.50 | +0.09 | −13.7 | 49 % |
| 10 % | 4,154 | 13 % | 33 % | 52 % | +1.58 | +0.43 | +3.13 | −0.01 | −14.5 | 49 % |
| 15 % | 2,195 | 9 % | 25 % | 43 % | +1.26 | +0.16 | +2.74 | −0.23 | −18.6 | 50 % |

- **The operator is half right.**
  - Reclaiming does get rarer with depth (84 % → 25 % within 24 h).
  - The typical coin does keep falling: the median is +1.3 to +3.0 % at 24 h and +2.7 to +4.5 % at 72 h.
- **But there is no point of no return.**
  - The MEAN stays at +0.2 to +0.6 % (24 h) and ≈ 0 (72 h) at every depth.
  - About half of all coins squeeze +10 % before falling another 10 %, at every depth.
- The pre-stated test (reclaim ≤ 20 % ∧ mean ≥ +2 % at the same X in every half) fails at every X.
- **By half-year (mean 24 h move):**
  - 26-H1 / 26-H2 are +0.7 to +1.7 at every depth (BTC-bear era).
  - 24-H1 / 24-H2 turn to −0.9 … −4.2 at depth ≥ 6 %.
  - Again a regime, not a rule.
- **By market cap:** ≤ $50 M coins have a *higher* median but a *negative* mean (−0.2 … −0.7 at 24 h): fatter squeeze tails. > $50 M coins are +0.4 … +0.9.

**(B) Tradable V cells.**
- The cells are 15 (depth 2 / 4 / 6 / 8 / 10 % × count gate none / ≥ 8 of 12 / 12 of 12) × 9 exits (TP 3 / 5 / 10 × stop ATR / 8 % / back above the VWAP + 0.5 %).
- Results:
  - **Avg −0.25 … +0.05 %/trade.** The best is 4 % below, no gate, TP 10 / stop 8 %: +0.05, 6.2/day.
  - The best pass count is 3 of 9.
  - The VWAP-reclaim stop does not help (−0.06 … −0.25).
  - The family-wise null 95 % is t = 1.61, and no cell reaches it.
- The new VWAP variables (closes below VWAP in the last 12, hours since the first close below, depth) entered a 2D screen against every condition: 4,032 rules, **0 pass the candidate gate**, 6 are positive at ≥ 5/day (best +0.14, t 1.46).
- Full tables: `_vwap_A.csv`, `_vwap_V.csv`.
- **Consistent with** POST_FRENZY_PATHS "VWAP break", CONFIRMED_TOP T2 and this study's VWAP fail (−0.10).

## The one lead: BTC daily regime (post-hoc, OBSERVE only)
The secondary search at exit TP 10 % / fixed 8 % stop (T10F) was declared as a deviation and run after the main result. It found **497 rules** at ≥ 5/day with t ≥ 2 on DESIGN. Their top is dominated by `btc_d1_regime == −1` (BTC daily EMA50 < EMA200).

Decomposition by BTC daily regime (one position per pair, all blocks):

| set (exit) | BTC daily bear: avg · trades per bear day | BTC daily bull: avg · trades per bull day |
|---|---|---|
| every Frenzy 15m bar (T5A) | −0.23 · 224 | −0.27 · 119 |
| every Frenzy 15m bar (T10F) | −0.16 · 88 | −0.37 · 42 |
| global L/S ≥ 1.45 (T10F) | **+0.28 · 27** (25-H2 +0.51, 26-H1 +0.22, 26-H2 +0.31) | −0.30 · 18 (−0.26 … −0.58 every bull half, 26-H2 +0.19) |
| top-trader account L/S ≥ 1.49 (T10F) | **+0.28 · 26** | −0.31 · 17 |
| EXHAUSTION (T10F) | +0.97 · 4.3 | +0.48 · 2.7 |
| EXHAUSTION (T5A) | +0.30 · 4.6 | +0.29 · 2.9 |

- **Why it is not a build:**
  1. The bear regime existed for **one ~300-day window** (2025-11-18 → 2026-09-13) plus 9 days in 2023-09/10. In those 9 days the same rules lost (−0.7 to −1.0 on 12 trades).
  2. Under the window-units rule a market-wide gate is a sleeve on/off switch, and one window is one observation.
  3. The exit (T10F) and the regime split were chosen after seeing the T5A result.
  4. The best regime rules fail Kelly (−0.01 … −0.04), best-5 %-out is borderline, and the worst day is −74 … −146 % summed (fat clustering).
  5. The regime is **OFF now** (BTC daily turned bull on 2026-09-13).
- **What it does say:** the "~42 pumps" become a high-frequency short only when BTC itself is in a daily downtrend. In a BTC uptrend, shorting crowded-long pump coins loses.
- It also matches the CLAUDE.md lesson that entry rules behave opposite across the BTC daily regime.

## Event study
![events](UNIFIED_SHORT_2026-11-18_events.png)

Mean and median gross short path from the entry open, −2 h → +24 h:
- EXHAUSTION: +0.30 / +0.57 / +1.39 % mean at 1 / 4 / 24 h; median +4.6 at 24 h.
- C1: +1.68 at 24 h.
- C2: +1.23 at 24 h.
- C5: +3.35 at 24 h, but most of it is in a few windows.
- R_K5N2: +0.39 at 24 h.
- P2R: +0.98 at 24 h.
- Random state rows: −0.17 at 24 h.

The medians are positive for every set, including random state rows (+1.5 % at 24 h). The typical Frenzy coin drifts down, but the mean is eaten by the squeeze tail. That is the whole story of this study in one chart.

## Per-half-year chart
![halves](UNIFIED_SHORT_2026-11-18_halves.png)

## Replays (operator's coins)
- **Not replayable:** the 5m caches end 2026-11-16 16:00 for PIPPIN / ZKC / ZKP / TLM, and 11-17 16:50 for the others. PIPPIN's 11-18 sell-down, ZKC and ZKP are after the data end. BREV's 11-17 listing has no 24 h volume history yet (excluded by the vol24 gate).
- **HUMA 11-16 17:30:**
  - EXHAUSTION shorts and is stopped (−6.5 % at T5A).
  - V (4 % below VWAP) enters the same bar, also stopped.
  - V (8 % below) on 11-17 08:00 wins +4.3.
  - HUMA was rank 15 → 3, so it climbed the list after the spike. No R cell fired.
- **BLESS 11-15/16:**
  - R_D2 / D3 / D5 fired as it slid from rank 13 to 18. All three were stopped (−1.7 … −1.9).
  - BLESS then climbed to rank 1–2 on 11-17.
  - V at 4 % below on 11-14 19:30 won +4.8. V at 8 % below lost −4.6.
  - BLESS's 24 h reclaims at every depth (11-14 episode): the point of no return did not hold.
- **EVAA:**
  - V won on 11-05, 11-08 and 11-14 (TP5) and lost on 11-07.
  - On 11-14 EVAA closed 6 % below VWAP, then squeezed: the price rose +56 % within 24 h (+81 % from the 17 % mark), with a −19 % dip on the way. This is the squeeze tail in one coin.
- **ZIL 11-09:** V at 4 % below won +4.6.
- **Mix:** the replays are a mix, as the averages say.
- Files: `_replay.csv`, `UNIFIED_SHORT_data/replay_vwap.txt`.

## Verdict
**NO** — no new SHORT sleeve, no expansion of EXHAUSTION. No rule, model, rank cell or VWAP-depth cell passes more than 5 of 9 bar items, and none at ≥ 5 trades/day passes more than 3.

- **EXHAUSTION_SHORT stays as is.** It is still the highest-frequency short with an edge: ≈ 3.3/day; +0.29 at T5A; random-in-state equal, so the state is the edge.
- The honest frequency answer:
  - At today's costs, a pump-coin short with an edge exists only in a narrow state (EXHAUSTION's) on ≈ 3–4 coins a day.
  - The other ~40 flagged coins a day are, on average, money-losing shorts because of the squeeze tail.
  - Their median trade wins. The mean does not.

**OBSERVE lines (frozen now, zero cost, read at batch reviews; no arming before they meet the bar on new data). Three of them; lines 1 and 3 are the cleanest out-of-design candidates:**
1. **C2: VSW red 15m candle ∧ top-trader position L/S 4 h change ≤ −0.011** (frozen thresholds), exit T5A.
   - Revert / retire if the first 60 new fills average < 0 or if they are below +0.08 (the DESIGN value).
   - Its union with EXHAUSTION is the only add-on that raised the daily compound (+0.164 → +0.233) without breaking the drawdown limit.
2. **BTC-daily-bear crowded-long short:** global L/S ≥ 1.45 ∧ `btc_d1_regime == −1`, exit TP 10 % / fixed 8 % / 24 h.
   - Read only when BTC daily turns bear again. It is OFF now.
   - Its evidence is one window.
   - Gate: the first 2 new bear windows (≥ 30 days each) both > 0 at 0.20 % slippage.
3. **GBM top-1 % score (frozen model = the 2026-10 refit in `UNIFIED_SHORT_data/pred_gbm.pkl`; its training-quantile threshold).**
   - 2.8/day, walk-forward +0.22 with every half-year > 0, only 37 of its 2,904 trades shared with EXHAUSTION.
   - To observe it, the scout must score the model on live bars. That is a build task, and it is not done here.
   - Gate: first 60 new fills > 0 at 0.20 % slippage and Kelly at the Wilson floor > 0.

## Assumptions and blind spots
- **Slippage** is a flat 0.10 % per taker leg. Live Frenzy fills had a median of 0.06 %, with thin-coin outliers of 0.3–0.6 %. Every candidate is ≤ 0 at 0.40 %.
- **Spikes** are the bar rule (allspk). This is ≈ the engine flag, but not identical: 26 / 35 Flatbase fills matched the trigger bit.
- **Market cap** uses today's circulating supply (look-ahead / unlock bias). 10–29 % of rows are unscored.
- **Crowding data** (OI, top-trader L/S, taker) has 32–45 % coverage before 2024-H2. Rules built on it are half-tested in H-OLD.
- **Holdouts were read before** by earlier studies, for other rules. The candidate rules were fitted only on DESIGN.
- **The circular-shift null** keeps the pool's negative mean. It tests "is this subset better than the pool, beyond luck", which is lenient (95 % point t = 1.74). It still rejected everything at ≥ 5/day except C5 (p 0.04), which fails 8 other items.
- **Exits:** one fixed family. Exit tuning per rule was not done (by design). The T10F secondary run is post-hoc.
- **Not modelled:** order-book depth, liquidations (no feed), news / unlocks / listings, and Alpha listing (not point-in-time).
- **GBM** was refit quarterly instead of monthly (runtime). GAM was refit monthly.
- **2023-H1** is not buildable. 11-18 is not in the data.

## Next step
1. Keep EXHAUSTION_SHORT as is (paper). Nothing to build from this study.
2. Register the OBSERVE lines above in the scout with frozen definitions (line 3 needs a model-scoring hook), so they are priced on new fills and need no new study.
3. If the operator wants more frequency anyway, the lever is the regime, not more triggers:
   - When BTC daily turns bear again, re-read line 2 on its first window.
   - Until then, more shorts per day means more squeezes, not more profit.

## Files
- **Report:** `reports/UNIFIED_SHORT_2026-11-18.md`, `_PREREG.md` (+ addenda A, B), `_INTERIM.md`.
- **Tables:**
  - `_triggers.csv`, `_rules.csv` (T5A closest 27 + EXH + STATE), `_rules_T10F.csv`, `_model.csv`.
  - `_rank.csv`, `_vwap_A.csv`, `_vwap_V.csv`, `_parity.csv`, `_replay.csv`.
- **Charts:** `_frontier.png`, `_halves.png`, `_events.png`.
- **Scripts:** `scripts/unified_short_{build,global,rank,lib,screen,search,closest,model,model_eval,parity,interim,rank_eval,vwap,vwap_eval,vwap_2d,event,replay,charts}.py`.
- **Data:** `reports/UNIFIED_SHORT_data/`. Includes rows/, pool.pkl, global.pkl, rank.pkl, vwap/, search*.pkl, pred_gam.pkl, pred_gbm.pkl, closest.log, concentration.csv, vwap2d.csv.
