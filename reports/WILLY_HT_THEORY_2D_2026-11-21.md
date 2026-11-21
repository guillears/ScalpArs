# WILLY_HT — PRE_VOL_SURGE × a theory variable (2D filter study, 2026-11-21)

**Research only.** No config, engine, test or git change. **Every number is UNVERIFIED exploratory.** The assumptions are listed in §4.

- **Pre-registration:** `WILLY_HT_THEORY_2D_2026-11-21_PREREG.md`, frozen at 22:55 UTC before any 2D number was computed.
- **Scripts:** `scripts/willy_ht_theory_2d_1121_build.py` (features) · `_oi_api.py` (OI for the 11-20 / 11-21 batch fills) · `_eval.py` (cells, gates, screen, null) · `_mkt.py` + `_mkt_eval.py` (post-hoc breadth replication) · `_chart.py`.
- **Outputs:** `reports/WILLY_HT_THEORY_2D_2026-11-21_{prereg_cells, gates, screen_bt, screen_real, mkt}.csv`, `_null.json`, `_tables.md` (every table in full), `_event_study.png`, `_data/features.csv`.

## Question
Operator: *"What about a 2D filter of PRE_VOL_SURGE + something else? Focus on the theory of what the trade is about."*

## Short answer
- **No 2D rule. Nothing passes the filter bar, and no cell beats the shuffled null.** Nothing is armed. Nothing changes.
- **Of the 6 theory variables:**
  - 4 are refuted on HOLDOUT or untestable: spike age, taker-buy share, OI change, BTC 1 h.
  - 2 keep their predicted sign on HOLDOUT and on the real fills:
    - **V4, the burst closes in the lower half of its 15-min range** ("rejection on volume");
    - **V6, the burst is stretched above the spike VWAP.**
  - Neither beats luck. Each one's own permutation p is about 0.20. The best of the 6 has p 0.61. They rank 22nd and 16th of 177 screen cells, and their family-wise p is 1.00.
  - Both fail criterion ② of the bar (P(mean < 0) ≈ 0.78). Neither holds across half-years.
- **The real-24 screen's best cell does not replicate.** It was market breadth `entry_mkt_up4h_pct` ≤ 0.47 inside the burst: 6 fills, 6 losers, family-wise p 0.014. Re-tested on the backtest (post hoc), it **reverses** in all 3 HOLDOUT half-years.
- **The theory's main idea was wrong in sign.**
  - I expected "burst right after the spike = buying the climax" to be bad. The reverse held, on HOLDOUT and on the real fills:
    - **burst with a young spike (≤ 20 min):** HOLDOUT 110 · 45 % · +0.53; real 7 · 57 % · +1.91;
    - **burst with an older spike:** HOLDOUT 48 · 35 % · −0.50; real 6 · 17 % · −2.42.
  - The bad burst is a **late second volume wave, with price stretched above the VWAP**. That is the same finding as V6 (spike age vs stretch: Spearman 0.74).
  - It is post hoc, so it can only be watched.
- **Recommendation (pre-registered rule: passes HOLDOUT and agrees on the real fills → OBSERVE only):**
  - Register **V4 = HT_BURST_REJECTION** as an observe line, frozen cut 0.50. It is the more stable of the two: positive in both BTC daily regimes on HOLDOUT, and in 2 of 3 HOLDOUT halves.
  - **V6 = HT_BURST_STRETCHED** qualifies by the rule but is weaker. On HOLDOUT, only 26-H2 is positive. DESIGN points the other way. Its "agreement" on the real fills comes from 2 master fills, and the batch contrast is 0.00.
  - My recommendation is to register V4 only. V6 is listed so the choice is yours.
- **Risk, stated plainly:**
  - 24 real fills with 8 losers cannot carry a 2D rule. V4's real cell is 3 fills; V6's is 6.
  - The backtest's 267 HOLDOUT fills are the only real test, and there both cells are within luck.
  - Even the best cells only move the backtest sleeve from ≈ 0 to ≈ +0.1 %/fill. A 2D cell re-sorts a breakeven sleeve. It does not create an edge.

## 1 · What WILLY_HT trades (from the engine: `services/frenzy.py`, `trading_config.json`)

```
 price
   │            H = spike high
   │           ╱╲   ← first RED 5m bar (≤ 60 min after the spike)  → BUY at market
   │          ╱  ╲▁   must be ≥ 1.4 % below H (DL 594) and still ABOVE the spike VWAP (FAILED_SPIKE, DL 527)
   │         ╱      ╲___ … second leg?  TP +6 net (resting maker)          ← WINNERS: +7 … +12 within ~15 min – 2 h
   │  ______╱  spike: 30-min return ≥ +5 %, last-hour volume ≥ 20× normal, ≥ $2 M
   │                     ╲___ "worked, then gave back": +2 … +5, then −4 stop   ← REAL LOSERS (5 of 8)
   └──────────────────────────────────────────────── time (24 h cap)
   HT = the setups whose 24 h volume ÷ market cap ≥ 1.0 (very high turnover — the ones plain WILLY refuses)
```

- **The bet:** buy the first pause of a fresh, violent, high-turnover pump, betting on a second leg (attention and momentum continuation).
- **The other side:** spike buyers taking profit, shorts fading the pump, market makers unloading inventory.
- **Why it can continue:** new attention keeps bringing buyers for hours. While price holds above the spike VWAP, the average spike buyer is in profit and under no pressure to sell.
- **What kills it:** the spike was the climax. The buyers who would carry a second leg already bought inside the spike. Late buyers and short covering push +2 … +5 %, and then the coin drifts back.
  - Frenzy coins mean-revert. The median backtest HT fill is −6 % after 24 h (WILLY_HT_THEORY 11-19).

**Where pre_vol comes in (a mechanical point).**
- `pre_vol15_vs60` = the volume of the last 15 one-minute bars ÷ the volume of the 60 bars before them.
- The spike needs ≥ 20 × normal hourly volume.
- **Spike bar inside the last 15 minutes:** the 15-min window holds the spike, and the base window holds the quiet hour before it. pre_vol is large *by construction*.
- **Spike older than about 20 minutes:** the base window already holds the spike. pre_vol > 1.6 then means volume is *still accelerating* after the spike.
- So "burst" mixes two different situations. Spearman(pre_vol, spike age) = −0.51.

**Theory (pre-registered).**
- **A burst is GOOD when it is fresh demand:** buyers are the aggressors, new positions are opening (OI up), price holds the top of its range, BTC is rising, and price is not stretched far above the buyers' average cost.
- **A burst is BAD when it is a climax:** sellers are the aggressors, OI is falling (shorts covering), the range closes near its low (rejection), BTC is falling, or price is stretched far above the VWAP.

## 2 · Pre-registration (frozen 22:55 UTC; full text in `_PREREG.md`)

Block = burst (pre_vol > 1.60, frozen DL 596) ∧ Vi on its theory-BAD side.

| # | Variable (entry-time, rebuilt identically for real and backtest) | BAD side | Natural cut | Why bad |
|---|---|---|---|---|
| V1 | spike age (min) | young | ≤ 20 (spike bar inside the 15-min window) | the burst is the climax bar itself |
| V2 | taker-buy share of the 15-min window volume | low | < 0.50 | sellers are the aggressors (distribution) |
| V3 | OI value change over 30 min (aligned, ≤ T−5 min) | falling | ≤ 0 | short covering, not new longs |
| V4 | close location in the 15-min high–low range | low | < 0.50 | rejection on volume |
| V5 | BTC 60-min return (MARKET-WIDE → day units) | falling | ≤ 0 | no tide |
| V6 | % above the spike-anchored VWAP | stretched | > median 0.878 | supply from profit-taking |

- **Granularities:** ① natural cut · ② label-free median (all 324 fills) · ③ outer tercile.
- **SUPPORTED** = on HOLDOUT, the within-burst contrast has the predicted sign AND the filter's Δ$ > 0, AND the real 24 agree.

## 3 · Coverage, parity and blind spots

| Item | Real 24 | Backtest 300 | Parity / note |
|---|---|---|---|
| pre_vol (rebuilt from k1m) | 100 % | 100 % | max \|Δ\| vs the values used 0.00005 (= live stamp, DL 596) |
| V1 spike age | 100 % | 100 % | = WL stamp-derived age to 0.2 min |
| V2 taker-buy share | 100 % | 100 % | vision 1m `taker_buy_quote` — no live stamp exists |
| V3 OI change | 100 % | 100 % | vision metrics, calendar-aligned (see §4); 8 batch fills (11-19…11-21, not yet on vision) from the live API |
| V4 close location | 100 % | 100 % | no live stamp — would be computed by the scout from 1m klines |
| V5 BTC 60 min | 100 % | 100 % | BTCUSDT 1m |
| V6 vs VWAP | 100 % | 99.3 % | rebuild ≈ live `entry_frenzy_vs_vwap_pct`: Spearman 0.91, same side of 0.878 on 22 / 24 |
| Screen features (backtest) | — | 44 rebuilt features + dip + BTC 1d gap | price / return / range / volume / taker / RSI / ADX / ATR / EMA gaps / OI / top-trader / taker L/S / funding / BTC / hour / dip |
| Screen stamps (real) | 427 cells from 129 numeric `entry_*` / derived stamps + the rebuilt features | — | price levels, size, fee and booking columns excluded |
| Market breadth `mkt_up4h` (post hoc) | 21 / 24 stamped | 99 % rebuilt | rebuild vs stamp: max \|Δ\| 0.01 on 9 overlaps; universe = cached 5m files (survivorship) |

**Lag check.**
- Every pair window uses 1m bars opened before the fill minute.
- 5m indicators use 5m bars closed by the fill minute.
- OI and ratios use snapshots ≤ T − 5 min (the live read).
- Funding is the last settlement ≤ the fill.
- Breadth is the last closed hour.

**Blind spots (not tested).**
- Spot vs perp volume share: no spot klines were rebuilt.
- Order-book depth and imbalance, and liquidations: there is no history before Route B.
- Funding: the rebuild is the last *settled* rate, while the live stamp is the *predicted* rate. These are different definitions, so funding is read in the screen only.
- About 80 live-only `entry_*` stamps could be screened on the 24 real fills only. Nothing in the backtest can refute or confirm them.

## 4 · Method and assumptions (UNVERIFIED)

**Outcome.**
- Each fill's TP +6 / SL −4 / 24 h net %, re-walked by the DL 608 engine (`grid_tp1.pkl` G6, parity 0.000 with DL 608).
- Maker TP; stop = taker + 0.10 % slip; stop checked first inside a bar; real or backtest funding.
- One size, $3,050 per fill.
- DCR = (Π (1 + Σ fill % × 0.61 / 100))^(1/D) − 1, over every calendar day of the block.

**Blocks.**
- ① batch 12 · ② master 12 (`KEPT_LIST_WILLY_HT_2026-11-21.csv`).
- ③ backtest: the DL 608 pool, 300 fills. DESIGN < 2025-07 (33) and **HOLDOUT ≥ 2025-07 (267) = the judge**, plus half-years.
- There is no seed dimension, because the pool is one deterministic replay.

**OI alignment.**
- The cached metrics files carry no contract count, so the per-file offset cannot be measured.
- The calendar rule of `OI_TS_AUDIT_2026-11-21_aligned.py` is used instead: +5 min for 2024-05 → 2026-11-15, else 0. The audit found that shift on 95–100 % of files.
- A file with a different offset could be read 5 min off.

**Screen.**
- Cells: burst ∧ each feature at > median, ≤ median, ≤ T1 and ≥ T2. Medians and terciles are label-free.
- Backtest: 177 cells with ≥ 15 blocked HOLDOUT fills. Real: 427 cells with ≥ 3 blocked fills.
- Statistic: the filter's Δ avg per fill over all fills (= −Σ blocked / N).
- Null: labels shuffled (within master / batch for the real fills), 1,000×. Family-wise p = the share of shuffles whose best cell is ≥ the observed value.

## 5 · Per-pair results (natural cut; all three granularities are in `_prereg_cells.csv`)

Cells read N · WR · avg TP6 %. "Contrast" = the burst-good avg minus the burst-bad avg (> 0 = the theory's direction).

### Verdict table

| Pair | ① natural | ② median | ③ tercile | HOLDOUT contrast · Δ$ | DESIGN contrast · Δ$ | Real 24 contrast · Δ$ (master · batch) |
|---|---|---|---|---|---|---|
| V1 spike young | **REFUTED** | REFUTED | REFUTED | **−1.02 · −$1,767** | +8.59 · +$1,186 | −4.32 · −$407 (−7.98 · −1.36) |
| V2 taker-buy low | REFUTED | REFUTED | HOLDOUT ok / real disagrees | +0.18 · −$138 | +3.77 · +$500 | −2.43 · −$223 |
| V3 OI falling | **UNTESTABLE** (1 of 158 HOLDOUT bursts had OI ≤ 0; 0 real) | REFUTED (−$1,926) | REFUTED (−$597) | (median) −1.56 · −$1,926 | +5.35 · +$1,052 | +0.84 · +$67 |
| **V4 close in low half** | **SUPPORTED** | REFUTED (cut 0.60: −$132) | SUPPORTED | **+1.10 · +$720** | +3.11 · +$125 (N 1) | +5.21 · +$375 (+7.98 · +3.37) |
| V5 BTC 60 min ≤ 0 | REFUTED | REFUTED | REFUTED | −0.45 · −$1,029 | −1.02 · +$179 | +0.72 · +$87 |
| **V6 stretched > 0.878** | **SUPPORTED** | SUPPORTED | SUPPORTED | **+1.01 · +$816** | **−4.32 · −$299** | +4.32 · +$442 (+10.64 · **+0.01**) |

Reference: PRE_VOL_SURGE alone (1D) on HOLDOUT: −$1,037.

### V4 · the burst closed in the lower half of its 15-min range: three blocks, before → after (block burst ∧ cloc15 < 0.50)

| Block | burst ∧ GOOD | burst ∧ BAD (blocked) | calm ∧ good · calm ∧ bad | Before: N · WR · avg · $ · DCR %/day | After | Δ$ |
|---|---|---|---|---|---|---|
| ① batch | 6 · 33 % · −0.73 | 2 · 0 % · −4.10 (BLESS, ENSO) | 2 · +5.98 · 2 · +6.00 | 12 · 50 % · +0.95 · +$348 · 1.57 | 10 · 60 % · +1.96 · +$598 · 2.81 | +$250 |
| ② master | 4 · 75 % · +3.88 | 1 · 0 % · −4.10 (EVAA 11-14) | 6 · +6.01 · 1 · +6.00 | 12 · 83 % · +4.46 · +$1,632 · 1.99 | 11 · 91 % · +5.24 · +$1,757 · 2.16 | +$125 |
| ①+② real 24 | 10 · 50 % · +1.11 | **3 · 0 % · −4.10** | 8 · +6.01 · 3 · +6.00 | 24 · 67 % · +2.70 · +$1,980 · 1.66 | 21 · 76 % · +3.68 · +$2,354 · 1.99 | +$375 |
| ③ DESIGN | 17 · 29 % · −0.99 | 1 · 0 % · −4.10 | 5 · +1.96 · 10 · −1.90 | 33 · 30 % · −0.91 · −$918 · −0.15 | 32 · 31 % · −0.81 · −$793 · −0.13 | +$125 |
| ③ **HOLDOUT** | 120 · 45 % · +0.48 | **38 · 34 % · −0.62** | 60 · −0.39 · 49 · −0.13 | 267 · 40 % · +0.02 · +$140 · −0.02 | 229 · 41 % · +0.12 · +$861 · +0.01 | **+$720** |
| ③ halves 25-H2 · 26-H1 · 26-H2 | | 9 · +1.51 · 10 · −1.05 · 19 · −1.40 | | | | −$414 · +$321 · +$813 |

### V6 · the burst is stretched above the spike VWAP: three blocks (block burst ∧ vs_vwap > 0.878)

| Block | burst ∧ GOOD | burst ∧ BAD (blocked) | Before | After | Δ$ |
|---|---|---|---|---|---|
| ① batch | 4 · 25 % · −1.57 | 4 · 25 % · −1.58 | 12 · 50 % · +0.95 · +$348 · 1.57 | 8 · 62 % · +2.21 · +$540 · 2.62 | +$192 |
| ② master | 3 · 100 % · +6.54 | 2 · 0 % · −4.10 (ICNT, EVAA) | 12 · 83 % · +4.46 · +$1,632 · 1.99 | 10 · 100 % · +6.17 · +$1,882 · 2.31 | +$250 |
| ①+② real 24 | 7 · 57 % · +1.91 | 6 · 17 % · −2.42 | 24 · 67 % · +2.70 · +$1,980 · 1.66 | 18 · 83 % · +4.41 · +$2,422 · 2.06 | +$442 |
| ③ DESIGN | 13 · 15 % · −2.36 | **5 · 60 % · +1.96** (reversed) | 33 · −0.91 · −0.15 | 28 · −1.42 · −0.19 | −$299 |
| ③ **HOLDOUT** | 96 · 46 % · +0.57 | 61 · 36 % · −0.44 | 267 · 40 % · +0.02 · +$140 · −0.02 | 206 · 42 % · +0.15 · +$956 · +0.02 | **+$816** |
| ③ halves 25-H2 · 26-H1 · 26-H2 | | 16 · +0.32 · 17 · +1.27 · 28 · −1.91 | | | −$158 · **−$657** · +$1,631 |

The V4 and V6 cells overlap on HOLDOUT (18 shared fills). Their union blocks 80 fills at −0.42 avg, Δ +$1,029.

### Gate reads on the blocked cohort ("Pattern C → FILTER — EXPECTANCY bar")

| Cell · block | N · days · pairs | WR vs BE | avg · P(mean < 0), day bootstrap · 95 % CI | top pair / day share of loss | LOO fill · worst pair drop | ① WR < BE | ② P ≥ 0.95 | ③ ≥ 8 days, < 50 % | ④ N ≥ 15 | Gain → haircut 30–50 % |
|---|---|---|---|---|---|---|---|---|---|---|
| V4 · HOLDOUT | 38 · 35 · 32 | 34 % vs 40 % | −0.62 · **0.78** · [−2.16, +1.08] | EVAA 16 % (4 fills) / 8 % | −0.80…−0.53 · −0.21 | ✔ | **✗** | ✔ | ✔ | +$720 → +$360…+$504 |
| V4 · real 24 | 3 · 3 · 3 | 0 % vs 40 % | −4.10 · 1.00 | 33 % / 33 % | −4.10 | ✔ | ✔ | ✗ | ✗ | +$375 → +$187…+$262 |
| V6 · HOLDOUT | 61 · 50 · 49 | 36 % vs 40 % | −0.44 · **0.77** · [−1.59, +0.73] | EVAA 10 % / 5 % | −0.55…−0.38 · −0.18 | ✔ | **✗** | ✔ | ✔ | +$816 → +$408…+$571 |
| V6 · real 24 | 6 · 5 · 5 | 17 % vs 40 % | −2.42 · 1.00 · [−4.10, −0.32] | EVAA 40 % / 20 % | −4.10…−1.57 | ✔ | ✔ | ✗ | ✗ | +$442 → +$221…+$310 |

**Both cells fail the bar.** On HOLDOUT they fail ② (no confidence). On the real fills they fail ③ and ④ (3 and 6 fills).

### BTC daily trend (1d EMA50 vs EMA200): within-burst contrast (good − bad)

| Pair | Real up | Real down | HOLDOUT up | HOLDOUT down | DESIGN (all up) |
|---|---|---|---|---|---|
| V4 | +6.38 (1 bad fill) | +4.04 (2) | **+1.00** (13 bad · 12 d) | **+1.14** (25 · 23 d) | +3.11 (1) |
| V6 | +7.98 (2) | +0.85 (4) | +2.70 (18) | +0.33 (42) | −3.34 |
| V1 (young = bad) | −6.38 | −3.05 | −1.75 | −0.62 | +8.59 |
| V5 (BTC 1 h, market-wide) | −3.87 | +5.07 | −2.34 | +0.65 | −1.02 |

- **V4 keeps its sign in both BTC regimes on HOLDOUT.** It is the only pair that does.
- V5 flips sign with the regime, the "opposite across the BTC daily regime" pattern. Being market-wide, its evidence counts in days, and it is refuted.

## 6 · Screen vs the shuffled null

| Read | Observed | Null (shuffled labels) | Verdict |
|---|---|---|---|
| Backtest HOLDOUT, 177 cells: best Δ/fill | +0.17 (`hour_utc ≥ T2`, a time-of-day cell) | best-of-177 median +0.28, p95 +0.42 → family-wise p **0.93** | the best real cell is **weaker than luck's usual best** |
| Cells with Δ > 0 on BOTH DESIGN and HOLDOUT | 35 | median 57, p95 112 → P 0.76 | **fewer than chance** |
| 6 theory picks (natural cut), best-of-6 on HOLDOUT | +0.10 (V6) | P 0.61 | theory picks ≈ luck |
| Theory picks, own permutation p | V4 0.20 · V6 0.21 · V2 0.56 · V5 0.82 · V1 0.92 | — | none < 0.10 |
| Theory picks' rank among 177 | V6 16 · V4 22 · V3 60 · V2 80 · V5 131 · V1 166 | — | the two survivors sit in the top 12 %, not at the top |
| Real 24, 427 cells: best | `entry_mkt_up4h_pct ≤ 0.47` (breadth low): 6 blocked, all losers, Δ +1.02/fill | family-wise p **0.014** | beats its own null, **but see the replication below** |
| Real 24: cells better in BOTH master and batch | 58 | median 3 → P 0.013 | the WL 11-21 effect again: the 2 master burst losers (ICNT 11-08, EVAA 11-14) can be isolated by many collinear columns |

**Post-hoc replication of the real-24 winner on the backtest.** Breadth was rebuilt with the `services.mkt_up4h` rules (`_mkt.csv`). It is market-wide, so it is read in day units.

| Block | burst ∧ breadth ≤ 0.47 (blocked) | burst ∧ breadth > 0.47 | Contrast (theory "no tide" = bad) | Δ$ |
|---|---|---|---|---|
| Real 24 | 7 · 14 % · −2.44 (6 days) | 6 · 67 % · +2.66 | +5.09 | +$521 |
| DESIGN | 3 · 0 % · −3.31 (3 days) | 15 · 33 % · −0.73 | +2.58 | +$303 |
| **HOLDOUT** | **69 · 48 % · +0.78** (63 days) | 88 · 39 % · −0.18 | **−0.96** | **−$1,646** |
| HOLDOUT halves 25-H2 · 26-H1 · 26-H2 | +0.95 · +0.70 · +0.76 | +0.19 · −0.72 · −0.04 | −0.76 · −1.42 · −0.80 | all three reversed |

**Refuted.** On the backtest, low-breadth bursts are the *better* ones, in every HOLDOUT half and in both BTC daily regimes (+1.03 vs +0.25 when BTC is up, +0.61 vs −0.38 when it is down). The real cell is 6 fills on 5 days.

## 7 · Event study

![event study](WILLY_HT_THEORY_2D_2026-11-21_event_study.png)

- **What is plotted:** solid lines are the mean 1m close vs entry, and dashed lines are the median best-so-far high. These are hindsight paths, not cut at TP or stop.
- **Backtest HOLDOUT (right):**
  - The blocked cells (red) slide steadily and sit at about −4.4 % by 6 h.
  - The burst-good fills (blue) stay at +0.5 … +2.
  - The red cells' best-so-far still reaches +6.8 … +7.5. Many of them hit +6 before sliding, which is why the TP6 gap (−0.62 vs +0.48, −0.44 vs +0.57) is much smaller than the gap between the mean paths.
- **Real 24 (left):** the red cells are 3 and 6 fills. Their shape (+2 … +5 first, then through −4) is the same "worked, then gave back" path as all the real losers. That is far too few fills to separate.
- Even the best 2D cell picks fills whose *average* path is worse. Its *TP6* outcome is only modestly worse, and that difference is within luck.

## 8 · Meaning (what the theory taught us)

1. **A volume burst at entry is not a climax signal on its own.** Fresh-spike bursts, where the window simply contains the spike, are fine or even better.
2. **The burst that hurts is a late one.** The spike is already 20 min or more old, volume is still accelerating, and price is stretched above the spike VWAP. That reads as a second FOMO wave that does not hold.
   - The real fills and HOLDOUT agree on this direction.
   - It is post hoc for spike age (V1 was registered with the opposite sign), and V6, the registered stretch variable that carries it, is not stable across half-years.
3. **Rejection inside the burst (V4) is the cleanest theory signal.** When a burst's 15-min range closes in its lower half, sellers met the volume. It keeps its sign in both BTC regimes and in 2 of 3 HOLDOUT halves. It is still within luck (p 0.20), and the median cut (0.60) loses the effect, so the effect is fragile.
4. **Aggressor share, OI and BTC 1 h do not separate.**
   - Bursts almost always come with OI rising: 157 of 158 HOLDOUT bursts. So the "short-squeeze burst" in the theory does not show up here.
   - Bursts with *less* OI growth did better.
5. **The ceiling.** The best theory cell moves the 267 HOLDOUT fills from +0.02 to +0.12 %/fill. HT's backtest problem is the post-spike drift, and no entry-time 2D cell found here fixes it.

## 9 · Verdict

| Pair | Verdict | Action |
|---|---|---|
| V1 spike age (young = bad) | **REFUTED**: the opposite sign holds on HOLDOUT and on the real fills (post hoc, so not tradable as a rule) | none; the reversed reading is the same finding as V6 |
| V2 taker-buy share | **REFUTED** | none |
| V3 OI change | **UNTESTABLE** at the natural cut (no falling-OI bursts); **REFUTED** at median and tercile | none |
| V4 burst close location < 0.50 | **SUPPORTED in sign** (HOLDOUT + real + DESIGN N 1, both BTC regimes). **Fails** the luck test (p 0.20) and gate ② (P 0.78); not robust at the median cut | **OBSERVE line proposed** (below) |
| V5 BTC 60-min return | **REFUTED** (and flips with the BTC daily regime) | none |
| V6 stretch above the spike VWAP > 0.878 | **SUPPORTED in sign** on HOLDOUT + real. **Fails** the luck test (p 0.21) and gate ②; DESIGN reversed; HOLDOUT carried by 26-H2 alone; batch contrast 0.00 | qualifies by the pre-registered rule; **I recommend not registering it** (operator's call) |
| Post hoc: breadth `mkt_up4h` ≤ 0.47 (real-screen best) | **REFUTED** on the backtest (reversed in 3 / 3 HOLDOUT halves) | none |

**Proposed OBSERVE line (needs the operator's OK; not built). 👁 HT_BURST_REJECTION.**
- **Zone:** FRENZY_WILLY_HT fills with `entry_frenzy_pre_vol15_vs60` > 1.60 AND close location < 0.50.
  - Close location = (last 1m close − 15-min low) ÷ (15-min high − 15-min low), over the 15 one-minute bars opened before the fill minute.
  - It is computed by the scout from 1m klines, so no engine stamp is needed.
- **Frozen cut:** 0.50.
- **Counting:** forward fills from the registration commit + 10 min, at the DL 608 exit (exit era pinned; ⚠ STALE on an exit change).
- **🔒 Decided once,** at zone N ≥ 15 on ≥ 8 days, none pending:
  - zone avg ≥ rest avg, or zone avg ≥ 0 → **RETIRED**;
  - zone WR < the sleeve breakeven WR AND P(mean < 0) ≥ 0.95 (day-block bootstrap) AND no pair or day ≥ 50 % of the zone loss → **FILTER CANDIDATE** (an operator call, with the 30–50 % haircut and its own revert gate);
  - else → **KEEP OBSERVING**.
- **Expected pace:** 3 of 24 real fills (≈ 12 %) and 38 of 267 backtest fills (14 %) fall in the zone. At about 3–4 HT fills a day, that is ≈ 0.4 zone fills a day, so ≈ 6 weeks to N 15.
- **Show at each read:** the BTC daily sign split and EVAA's share.
- **If V6 is wanted anyway:** freeze the cut on the live stamp `entry_frenzy_vs_vwap_pct` > 0.88 (the rebuild and the stamp agree on 22 / 24 at that cut), with the same gate.

**What stays unchanged:** PRE_VOL_SURGE stays observe-only at 1.60 (DL 596, counted from DL 608), TP +6, and the armed dip and FAILED_SPIKE filters. Nothing is armed, and no config changes.
