# NEGFLANK — is there a second variable? (conditional 2D study, 2026-10-08)

Read-only research. No engine, config, scout, test or CLAUDE file was touched. `scripts/validate_against_master.py` ran first: **ALL CHECKS PASS**.

**Question.** Inside NEGFLANK (momentum LONG fills with BTC 1h EMA20 slope ≤ −0.05), the master pool splits sharply by "washed-out". The operator believes some second variable separates NEGFLANK winners from losers. Washed-out is the first candidate for it.

**Answer.**
- **No second variable survives.** We searched about 17,700 one- and two-variable splits inside NEGFLANK, using 89 variables (47 stamped, 42 rebuilt from klines).
  - The best split is no stronger than the best split found in shuffled data. Scan-corrected p is 0.22 to 0.94, depending on statistic and cohort.
  - The "discover on the first half of the year, test on the second half" check replicates at coin-flip rate (13 of 25).
- **Washed-out is not a NEGFLANK separator.**
  - In the replay it does nothing: washed 59 % · −0.12 %, not washed 60 % · −0.13 %.
  - On master it is a single market episode (Jun-22→Jul-1, plus one fill on Jun-18). That episode lifted every long, not only NEGFLANK longs.
- The closest thing to a real pattern is one family. Call it **"BTC daily uptrend + weak alt trend"**: NEGFLANK ∧ pair EMA20/50 gap ≤ 0.24 ∧ BTC 1d EMA20 slope > 0.28, or BTC above its 1d EMA20.
  - All three B18 losers sit in it.
  - It does not beat the scan null, and master has too few fills to confirm it.
  - At most, it is an **observe-only** pre-registration.

**Ship: nothing.**

Scripts (new): `scripts/study_negflank2d_features.py` (feature build), `scripts/study_negflank2d_fetch.py` (small cache extension), `scripts/study_negflank2d_screen.py` (scans + nulls + OOS), `scripts/study_negflank2d_followup.py` (washed / families / B18 / rules). Raw output: `reports/NEGFLANK_2D_STUDY_tables.md`, `reports/NEGFLANK_2D_sweep_yr5_all.csv`, `reports/NEGFLANK_2D_sweep_yr5_exwash.csv`, `reports/NEGFLANK_2D_oos_H1top25.csv`, `reports/NEGFLANK_2D_rebuild_validation.csv`, `reports/NEGFLANK_2D_features.pkl`.

---

## 1. Cohorts and data

Cohorts are taken unchanged from `study_ml_b18_common.py`.

| cohort | definition | NEGFLANK fills |
|---|---|---|
| **DISCOVERY: yr5 replay** | `yr5_fills_trimmed` ML fills, Jan-04 → Oct-02 | 753 = **251/seed** · 140 days |
| **CONFIRMATION: master** | kept, non-probe, CLOSED, ex-B1, `stack_pct` / `stack_pnl` | **37** · 22 days (13 washed + 24 not) |

**How the three yr5 seeds are used:** they are treated as **replicates**, not as independent fills. Every test clusters on the calendar day: all three seeds' fills on one day form one cluster. Each null shuffles whole days. N is quoted per seed.

**Operator's figure vs ours:** the operator quoted "27 · 59 % · −0.04 % · −$354" for the non-washed half. With today's loader it is **24 · 58 % · −0.061 % · −$354**. The dollars are the same; the N is 24, not 27.

**Rebuilt variables.** These were rebuilt from the local kline cache, using engine conventions (forming-bar EMA, slope = (ema − ema[−4]) / ema[−4]). The Oct-04→Oct-08 alt / ETH gap for the B17/B18 fills was filled by one small fetch: 9 × 5m + 583 × 1h requests, weight-guarded, no 418/429. Each rebuild was validated against stamped fills where both exist:

| rebuild | vs stamp | r (master / yr5) |
|---|---|---|
| BTC 1h EMA20 slope | entry_btc_1h_slope | 0.9995 / 0.9994 |
| BTC % below 30d high | entry_btc_off30d_high_pct | 0.997 / 0.9999 |
| BTC % below 24h high / above 24h low | off24h / off24lo | 0.998 / 0.998 · 0.999 / 0.998 |
| BTC previous-day return | entry_btc_1d_ret_pct | 1.000 / 0.9998 |
| BTC 72h return · eff72 | r72 · eff72 | 0.9998 / 0.9994 · 0.97 / 0.97 |
| pair 1h EMA20/200 gap | entry_pair_1h_ema20_200_gap_pct | 0.993 / 0.992 |

**Rebuilt with no stamp to validate against:**
- BTC 4h / 1d EMA gaps and slopes, BTC vs its 1d EMA20, today's 1d candle
- BTC 7d high / low distance
- BTC 1h slope change over 1, 2 and 3 h, and hours since the 1h slope turned negative
- BTC realised vol over 1h / 24h and their ratio
- ETH 1h slope and 24h return
- Alt index (median 24h return and up-share across 583 symbols) and the BTC-dominance proxy (BTC 24h − median alt 24h)
- Pair 1h slope, 4h gap, 1h / 24h return, distance from its 24h high / low, 24h volume vs 7d, and 24h return relative to BTC
- Hour (UTC) and weekend

Washed-out rebuilt as off30 ≤ −15 matches the Jun-18→Jul-2 date window on master except for 1 fill.

## 2. The operator's candidate — BTC distance from its 30-day high

### 2.1 Continuous dose-response inside NEGFLANK

Each cell: N · WR · avg % · $ · days · P(<0).

| BTC off 30d high | yr5 NEG (per seed) | yr5 non-NEG | master NEG | master non-NEG |
|---|---|---|---|---|
| ≤ −15 (washed) | 88 · 59% · **−0.122** · 44d | 80 · 55% · −0.094 | 13 · 100% · **+0.615** · 9d | 6 · 100% · **+0.600** |
| (−15, −10] | 30 · 66% · −0.060 | 22 · 64% · −0.040 | 0 | 1 |
| (−10, −6] | 42 · 57% · −0.122 | 41 · 69% · +0.006 | 4 · 75% · +0.298 | 9 · 78% · +0.140 |
| (−6, −3] | 57 · 66% · −0.079 | 103 · 61% · −0.048 | 15 · 47% · −0.368 | 30 · 77% · +0.188 |
| > −3 | 34 · 47% · −0.266 | 97 · 68% · +0.047 | 5 · 80% · +0.573 | 32 · 69% · +0.111 |

**Correlation of off30 with outcome inside NEGFLANK (Spearman):**
- yr5: +0.002 by fill and +0.02 by day-mean.
- yr5 ex-washed: −0.02.
- **The replay shows no dose-response at all.**

**yr5 washed-out NEGFLANK fills come from 4 separate episodes:**

| episode | avg % |
|---|---|
| Jan-30→Mar-3 | −0.17 |
| Jun-2→Jun-11 | −0.14 |
| Jun-16→Jun-18 | −0.01 |
| Jun-22→Jul-1 | **+0.11** |

The only positive episode is the one master also contains. Master's 13 washed fills are **one tape**: a single fill on Jun-18 plus Jun-22→Jul-1. Under the WINDOW-UNITS rule that is 1–2 observations.

### 2.2 Interaction: is washed-out special to NEGFLANK?

| cohort | Δ washed − not, in NEG | Δ in non-NEG | interaction [95 % day-CI] |
|---|---|---|---|
| yr5 | +0.003 | −0.090 | +0.09 [−0.14, +0.33] |
| master ex-B1 | +0.676 | +0.449 | +0.23 [−0.50, +0.93] |
| master ex-B1 ex-B18 | +0.585 | +0.449 | +0.14 [−0.59, +0.81] |

**Verdict on washed-out:**
- On master, the Jun-22→Jul-1 tape lifted every long, NEGFLANK or not. That is a sleeve-wide main effect of one episode, not a NEGFLANK-specific second variable.
- In the replay it separates nothing.
- In the exhaustive scan, every 1D off30 split has a scan p of 1.00.

### 2.3 If shipped anyway: "block NEGFLANK unless washed-out"

| cohort | before → after | Δ $ (haircut 30–50 %) |
|---|---|---|
| master ex-B1 | 115 · +$3,016 → 91 · +$3,370 | +$354 (+$177…+$248) |
| master ex-B1 **ex-B18** | 112 · +$3,463 → 91 · +$3,370 | **−$92** |
| B18 | 3 · −$447 → 0 | +$447 |
| yr5 (per seed) | 595 · −$9,081 → 431 · −$4,587 | +$4,494 (+$2,247…+$3,146) |

Expectancy bar on the blocked side (NEG ∧ not washed):

| cohort | N | WR vs BE | P(mean<0) | days | concentration | verdict |
|---|---|---|---|---|---|---|
| yr5 | 163 | 60 % < 66.9 % ✔ | 0.99 ✔ | 98 ✔ | 6 % / 6 % ✔ | PASS |
| master ex-B1 | 24 | 58 % < 61.3 % ✔ | **0.63 ✗** | 13 ✔ | 23 % / 22 % ✔ | **FAIL** |
| master ex-B18 | 21 | **67 % > 61.8 % ✗** | 0.46 ✗ | 12 | 28 % / 28 % | **FAIL** |

This is the same verdict as the B18 study: yr5 passes and master fails.

The washed-out exemption does not change the yr5 number, because washed NEGFLANK is just as bad there (−0.122). So the exemption has no replay support. It is justified by one master episode only.

## 3. Exhaustive conditional scan inside NEGFLANK

**Masks:** every variable × {sign, median, low/high tercile, low/high quintile}, plus every pair of variables × 4 median quadrants, plus 4 sign quadrants when both variables are signed.

**Eligibility:** zone and rest must each hold ≥ 30 fills and ≥ 10 days. Unscored fills count on neither side.

**Two statistics:**
- day-clustered z of Δ mean pct
- Welch z

Each is judged against its own **whole-day block-permutation null** (1,000×, maximum over the same full scan). A trade-shuffle null is shown for reference only.

| scan | masks | obs max \|z\| (cl / Welch) | null 95th (cl / Welch) | scan p (cl / Welch) | masks with \|z\| ≥ 3 vs null median (p) | survivors |
|---|---|---|---|---|---|---|
| yr5 NEG, all (251/seed, 140 d) | 17,748 | 3.66 / 5.24 | 5.55 / 6.41 | **0.94 / 0.49** | cl 23 vs 58 (0.87) · W 517 vs 340 (0.26) | **0** |
| yr5 NEG ex-washed (163/seed, 98 d) | 17,510 | 3.88 / 5.86 | 5.58 / 6.57 | **0.90 / 0.22** | cl 42 vs 72 (0.76) · W 774 vs 380 (0.12) | **0** |
| yr5 H1 only, then test on H2 / master | 17.7k | 4.21 (cl) | 5.70 | 0.77 | — | H1 top-25 → **H2 same sign 13/25** (1 with \|z\| ≥ 1.96) · master same sign 14/21 |

How to read this:
- **Survivor gate:** scan p < 0.05 (either statistic) ∧ same sign in both yr5 halves ∧ same sign with each month left out ∧ master points the same way with ≥ 5 zone fills. **Zero masks pass in any cohort.**
- **Number of strong splits:** the count of "strong-looking" splits is no higher than chance produces, so there is no hidden crowd of real effects either.
- **Halves/LOMO/master without the scan correction:** 3,008 masks pass these (2,280 ex-washed). This is exactly the over-fitting trap that the scan null exists to catch.

**What the tops look like:**
- The all-cohort top masks are mostly yr5-only stamps (`btc_rsi_closed`, `eth_5m_ret1`, `gap_5_20_signed`). These have 0–2 scored master fills, so they cannot be tested on master.
- The ex-washed top is one coherent family (§4).

## 4. Closest family: "BTC daily uptrend + weak alt trend" (not a survivor)

Ex-washed scan #1 by Welch: **NEG ∧ pair EMA20/50 gap ≤ 0.238 ∧ BTC 1d EMA20 slope > 0.281** (B1). Scan p is 0.22 (Welch) / 0.97 (cl).

Its twin B2 (BTC above its 1d EMA20 instead of slope) behaves the same. Its relatives are BTC 4h EMA20/50 gap > 0.27 and pair EMA50 slope low.

Each cell: N · WR · avg % · $.

| cohort | NEG ∧ B1 | NEG ∧ not B1 | Δ in NEG | Δ in **non-NEG** | interaction [95 % day-CI] |
|---|---|---|---|---|---|
| yr5 (per seed) | 46 · 36% · **−0.385** · −$4,029 | 205 · 65% · −0.065 | −0.320 | +0.084 | **−0.40 [−0.66, −0.16]** |
| master ex-B1 | 10 · 50% · −0.245 · −$574 | 27 · 81% · +0.333 | −0.578 | +0.289 | −0.87 [−1.49, −0.15] |
| master ex-B18 | **7 · 71% · −0.052** · −$127 | 27 · 81% · +0.333 | −0.385 | +0.289 | −0.67 [−1.32, −0.04] |
| master ex-washed ex-B18 | 7 · 71% · −0.052 | 14 · 64% · +0.071 | −0.123 | +0.340 | −0.46 [−1.27, +0.37] |

**What favours it:**
- **It is an interaction, not a main effect.** The same split does nothing in non-NEGFLANK longs.
- yr5 is consistent across seeds (−0.40 / −0.39 / −0.37 vs ≈ −0.07) and across halves (−0.27 / −0.42).
- yr5 months: Δ is negative in 6 of the 7 months that have zone fills (May is +0.09 on n = 2).
- Concentration is low: largest day 12 %, largest pair 8 %.
- It has **0 % overlap with washed-out**. Washed means the BTC daily trend is down; this zone requires it to be up. So this is not washed-out in disguise. If anything it is the mirror image of it.
- **B18:** ARB, UNI and LIT are all in the zone, and yr5 ends Oct-02, so this is an out-of-sample hit. Their values: BTC 1d slope +0.57, BTC vs 1d EMA20 +0.01…+0.05, pair EMA20/50 gap −0.21 / −0.33 / +0.02. Other context at the time: BTC 1h slope negative for about 24 h, dominance proxy +3.0…+3.5 %, only 5–6 % of alts up on 24h.

**What sinks it:**
- **Scan null.** A split this strong turns up by chance in 22 % of shuffled scans.
- **Dose-response is non-monotone.** On the yr5 NEG ex-washed tercile grid, pair gap low × BTC slope low is +0.145, but the *middle* pair-gap row is bad in every BTC column (−0.30 / −0.23 / −0.21). That is the confound signature the rules warn about.
- **One-variable legs are weak.** BTC 1d slope > 0 alone gives −0.18 vs −0.05 (H1 −0.05 / H2 −0.19), and every split has scan p 1.00.
- **Master before B18 is 7 fills, 71 % WR, −0.05 %.** That is the right sign but carries no weight. With B18 it becomes 10 fills on 6 days, with the largest day holding 55 % of the loss.

**Rule arithmetic, if "block NEGFLANK when B1" were shipped:**

| cohort | Δ $ |
|---|---|
| master ex-B1 | +$574 (haircut +$287…+$402) |
| master ex-B1 ex-B18 | **+$127** |
| B18 | +$447 |
| yr5 (per seed) | +$4,029 (−0.067 → −0.040 %/fill) |

Expectancy bar on the blocked side:

| cohort | WR vs BE | P(mean<0) | days | N | concentration | verdict |
|---|---|---|---|---|---|---|
| yr5 | 36 % < 66.9 % ✔ | 1.00 ✔ | 36 ✔ | 46 ✔ | 12 % / 8 % ✔ | PASS |
| master ex-B1 | 50 % ✔ | 0.90 ✗ | 6 ✗ | 10 ✗ | day 55 % ✗ | **FAIL** |
| master ex-B18 | 71 % ✗ | 0.68 ✗ | 5 ✗ | 7 ✗ | 52 % ✗ | **FAIL** |

B2 is nearly identical: master 11 fills · 45 % WR · −0.40 %, P 0.98, but only 7 days, so it still FAILS.

**The other families tested** (full tables in the raw file):

| family | result |
|---|---|
| BTC 1h slope negative for > 9 h | yr5 −0.19 vs −0.04. Master ex-B18 10 · 90 % · +0.23. **Master refutes it.** |
| 1h slope still falling over 3 h ∧ pair EMA50 slope > 0.17 (winner side) | yr5 +0.06 vs −0.16. Master +0.44 vs +0.05. Interaction CI crosses 0. |
| Dominance proxy > 0.61 ∧ ETH 1h slope > −0.25 (winner side) | yr5 +0.08 vs −0.18. Master +0.28 vs +0.12. Interaction ≈ 0 on master. |
| Global volume > 0.95 (the B18 study's post-hoc note) | yr5 ex-washed −0.003 / +0.055 vs −0.21, interaction CI [+0.02, +0.44]. Master 7 · 100 %. **The most consistent "harmless NEGFLANK" side**, but post-hoc and not a scan top. |

## 5. Verdict

| candidate | yr5 scan null | yr5 halves / LOMO | master direction | expectancy bar yr5 / master | decision |
|---|---|---|---|---|---|
| washed-out (off30 ≤ −15) | p 1.00 (no effect) | — | + (one episode) | n/a (no yr5 effect) | **nothing**: a one-episode main effect, not a NEGFLANK separator |
| B1/B2 BTC daily up ∧ weak pair trend | p 0.22 | ✔ / ✔ | ✔ (7 fills pre-B18) | PASS / FAIL | **observe-only** (optional pre-registration) |
| any other of 17.7k masks | p ≥ 0.22 | — | — | — | nothing |

**Recommendation: ship nothing.**
- NEGFLANK stays on its existing observe line (scout 3/15 fills · 1/8 days).
- The washed-out exemption the scout already applies stays as it is. This study gives no reason to tighten or widen it.
- **Optional:** register **NEG_DAILYUP_WEAKPAIR** as an observe-only scout tally with frozen thresholds: NEGFLANK ∧ entry_pair_ema20_ema50_gap_pct ≤ 0.24 ∧ BTC 1d EMA20 slope > 0.28.
  - It is judged on fills after registration only, using the locked expectancy bar (≥ 15 fills, ≥ 8 days).
  - Count B18 as its first OOS window (1 window, 3 fills, all lost).
  - The BTC 1d slope is not stamped live. It would need a stamp (one 1d-klines read per scan) or a review-time rebuild like this one.
  - If shipped one day, the pre-committed revert would be: the first 10 refused signals, re-priced, show WR ≥ 61 % or Σ > 0 → revert.

## 6. Blind spots (not tested, or tested weakly)

- **Master power is tiny.** NEGFLANK has 37 fills, the washed half is 1–2 episodes, and NEGFLANK ex-washed ex-B18 is 21 fills / 12 days. A true Δ of −0.3 % would usually not be detectable there. Master can refute but almost never confirm.
- **yr5 fidelity.** The replay reproduces live ML about 64 %. Its ML sleeve is net-negative (−0.067 %/fill) while master is positive (+0.18 %), so the populations differ. yr5 ends Oct-02, so it has no B18-era tape.
- **Seeds are replicates, not independent.** That is handled by day clustering and day-block nulls. Per-seed consistency is therefore weak evidence.
- **Low or absent master coverage.** Eight yr5-only stamps were scanned (`btc_rsi_closed`, `eth_5m_ret1`, `gap_5_8/5_20 signed`, `ema50_100 gap`, `above72`, `pair_age_days`), but master has 0–15 scored fills on them, so they are master-untestable. Note that `entry_gap` duplicates `gap_5_20_signed`.
- **Not available on yr5:** funding rate, market cap / CMC rank, order book, tick microstructure, news, sector.
- **Unvalidated rebuilds** (no stamp exists): 4h / 1d EMAs, ETH slope, alt index / dominance proxy, realised vol, slope-change and time-since-negative. The alt index is an equal-weight median over 583 cached symbols. The universe is whatever is cached, so it carries survivorship and listing drift. It is not volume-weighted.
- **Granularity.** 2D splits are median (and sign) quadrants only. There are no 2D tercile cells, no 3-variable cells, no continuous threshold optimisation, and no tree models; all of these were left out deliberately because of over-fitting. Hour of day was a linear median split, with no session buckets. Day of week was tested only as weekend vs weekday.
- **The day-clustered z has a heavy-tailed null** (its trade-shuffle null exceeds the observed). That is why the Welch statistic under the same day-block null is also reported. Neither finds anything.
- **Exit side and sequencing not modelled:** no path / stop counterfactuals, and no cross-sleeve slot effects (FRENZY / fades / bear-run positions open at the same time).
- **Cache extension.** B17/B18 pair, ETH and alt values for Oct-05→07 come from a fresh fetch (`reports/backtest_cache/negflank2d_ext/`). BTC values come from the existing cache.
