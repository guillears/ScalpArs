# SPIKE_FADE winners vs losers, and was KMNO 10-17 predictable? (2026-10-19)

Research only. No engine, config, `trading_config.json`, template, test or git change, and no Binance or bot-API calls (cache only).
Scripts: `scripts/study_fade_wl_screen.py` (exhaustive screen + null + OOS), `_prereg.py` (theory hypotheses + gate 41h), `_liq.py`
(liquidity follow-up), `_scored.py` (mcap/turnover scored-only). `scripts/validate_against_master.py` → **ALL CHECKS PASS**.

## Plain English

**Answer: no. The KMNO loss could not have been predicted from anything known at entry.** On every axis we can measure, KMNO looked
like a typical *winning* fade:
- very liquid: 24 h volume in the 97th–100th percentile, 7-day average volume $8.3M;
- large mcap ($328M);
- a small spike stretch (33rd–45th percentile);
- low ATR, calm BTC (BTC ATR in the 6th–27th percentile).

Its one unusual stamp was volume inflation (24 h volume 2.15× the 7-day average, 92nd percentile). In the backtest, inflated fades did
**better**, not worse. The loss came from a $107k third-party buy sweep after entry, plus 0.58 % of our own exit impact. The sweep is
post-entry information, and the impact is a sizing question that the size/mcap study already covers.

**What does separate winners from losers in general:**
1. **Nothing separates fade P&L at entry beyond chance.** The screen covered 80 stamps plus inflation, mcap, turnover, hour of day and
   regime dummies: 203 one-variable zones and 11,583 two-variable cells, against 300 day-preserving shuffles. Real backtest P&L hits sit
   *below* the null (1D 5 vs 8.2; 2D 2 vs 21.7). The master is null too (1D 7 vs 5.7, P 0.32; 2D 20 vs 8.7, P 0.13). mcap and turnover
   (scored-only) find nothing.
2. **There is one real structure, and it predicts *stopping*, not *losing money*.** In the backtest, a bigger spike stretch
   (distance from EMA13 / EMA5 / 5-8 gap) means more −1.5 % stops: about 27 % vs 15 %. It beats the shuffled null (count P 0.02,
   family-wise max P 0.02), and it confirms in both directions between 2025 and 2026 (7/7, null 1.0; 50/100, null 9.8). But the stretched
   fades also win bigger, so **P&L is unchanged** (2025: +0.12 vs +0.12). The master does not confirm it (1 of 107 vs a null of 9.9). It
   is not a filter. KMNO was on the *low*-stretch side.
3. **Pair liquidity is the only P&L family that passes out of sample, and it fails the filter bar.** Fades on pairs whose 7-day average
   daily volume is below $4.89M (the master median, frozen) do worse:
   - master: 37 · 81 % · +0.13 vs 37 · 97 % · +0.59, Δ P = 1.00;
   - backtest: 247 · 71 % · −0.04 vs +0.13, Δ P = 0.96.

   Master→BT confirmation count: 11 vs a null of 2.6, P 0.035. But in the backtest the zone itself sits at breakeven: WR 71 % against a
   breakeven WR of 70 %, and P(avg < 0) is only 0.73. It is flat in 2026-H1. The dose-response is **non-monotonic**: the thinnest bin
   (< $2M/day) wins in both blocks, which is the confound signature. KMNO was on the *liquid* side.
4. **Gate 41h (HEALTHY_BEAR ∧ BTC trend gap ≥ −0.13) is the only registered line KMNO falls into.** Today's-stack backtest now
   *contradicts* it: the zone does better, 121 · 77 % · +0.23 vs +0.02.

**Recommendation: nothing to ship. No filter, no new observe line.** The liquidity effect is real as a *difference*, but its practical
lever is ticket size vs liquidity, which the size/mcap study already handles (`min(0.6× ticket, 0.05 % of 24 h volume)`). One change at a
time. Keep gate 41h open with the updated tally below, and record that the today's-stack replay now points the other way.

---

## 0 · Theory first (written before the screen)

A fade shorts a one-bar RSI explosion. It wins when the spike was a **liquidity event**: one impatient buyer, a stop cascade or a
thin-book print, after which price reverts as the book refills. It loses when the spike is the **first print of new information** (news,
a listing, an accumulating buyer, a squeeze) and flow keeps coming. Candidate observables at entry:
- attention/inflation (volume vs its own baseline);
- absolute liquidity (thin pairs are easy to spike but also easy to push further);
- the higher-timeframe trend the fade leans against (pair 1h EMA20/200, DI spread);
- trend maturity (low, rising ADX = ignition; this was the original leg-6 logic);
- a breakout to a new range high (range position);
- the tape (weak BTC sells spikes, or makes an alt spike idiosyncratic);
- the registered gate 41h and the hot-tape line;
- mcap.

These became the 11 pre-registered directional tests in §4. **The one thing theory says is not observable at entry is the
information/flow that arrives afterwards. KMNO is exactly that case.**

## 1 · Data, outcome, coverage

| block | source | N | days | WR | avg % | stop % | BE-WR |
|---|---|---|---|---|---|---|---|
| ① LIVE | KMNO 2026-10-17 00:29 (scored, not in any statistic) | 1 | 1 | 0 % | −2.13 | stop | – |
| ② MASTER | `MASTER_POOL_stacked.csv` kept SPIKE_FADE, `stack_pct` (STACK 2026-10-19a), B1–B21 | 74 | 43 | 89.2 % | +0.356 | 13.5 % | 70.5 % |
| ③ BT | `ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv` SPIKE_FADE (today's stack, seed 1); pct = tick-tail-corrected from `FADE_SIZE_MCAP_2026-10-19_rows.csv` | 625 | 303 | 73.1 % | +0.060 | 23.0 % | 70.2 % |
| BT 2025 BULL | | 223 | 128 | 66.8 % | +0.120 | 25.1 % | 61.6 % |
| BT 2026-H1 BEAR | | 282 | – | 76.6 % | +0.010 | 22.0 % | 76.1 % |
| BT 2026-H2 BULL | | 120 | – | 76.7 % | +0.067 | 21.7 % | 73.3 % |

- **Outcomes** (all three screened): `pct` (continuous, today's exit); `loser` = pct < 0; `stop` = STOP_LOSS close or pct ≤ −1.49.
- **Coverage** (`_coverage.csv`, 136 candidate columns):
  - BT screens 80 columns at ≥ 90 % coverage (every core pair stamp, every BTC/breadth stamp incl. the 1h/4h/1d context:
    `btc_rsi_1h`, `btc_1d_ret`, `btc_4h_ema50_200_gap`, r72/eff72/above72/off24h/off30d, plus `pair_1h_ema20_200_gap`,
    `pair_1d_ndi`, infl, 7-day average volume, 24 h volume, hour, weekend, regime/macro dummies).
  - The master screens 66 columns. The BTC multi-timeframe stamps and `pair_1d_ndi`/`pair_1h_ema20_200` cover only 24–41 % of master
    fades (they were stamped later), so they are **not** screened on the master and are never treated as "rest".
  - mcap/turnover (BT 86 %) get their own scored-only screen.
- **Units:** z-scores are day-cluster-robust, with an iid floor so a zero-variance zone cannot produce an infinite z. The null is a
  **day-stratified permutation**: whole days' outcome vectors are swapped among days with the same fill count, so same-window fills stay
  one observation. Minimum zone size: N ≥ 15 and ≥ 8 days, on both sides.

## 2 · Exhaustive screen against the null (300 shuffles; every test in `_screen_<block>_<y>.csv`)

1D: sign / median / outer terciles. 2D: **all pairs** of screened columns, four median quadrants each.

| block | y | 1D \|z\|>2 real vs null (P) | 1D max \|z\| vs null p95 | 2D \|z\|>3 real vs null (P) | 2D max vs p95 | read |
|---|---|---|---|---|---|---|
| BT | pct | 5 vs 8.2 (0.71) | 2.27 vs 3.49 | 2 vs 21.7 (0.93) | 3.10 vs 4.30 | **nothing** |
| BT | stop | **29 vs 9.2 (0.02)** | **3.90 vs 3.67** | **127 vs 26 (0.03)** | 4.42 vs 4.41 | **real (stretch family)** |
| BT | loser | 24 vs 10.2 (0.05) | 3.31 vs 3.65 | 70 vs 36 (0.14) | 3.78 vs 4.45 | borderline |
| BT 2025 | pct / stop / loser | 7/6/5 vs 8.4/6.9/9.9 | < p95 | 0/1/23 vs 19/19/34 | < p95 | nothing |
| BT 2026 | pct | 12 vs 9.0 (0.26) | 2.39 vs 3.55 | 59 vs 27 (0.11) | 3.73 vs 4.56 | weak |
| BT 2026 | stop | 26 vs 10.7 (0.03) | 3.19 vs 3.72 | 74 vs 31 (0.10) | 3.91 vs 4.39 | stretch family |
| MASTER | pct | 7 vs 5.7 (0.32) | 2.87 vs 3.28 | 20 vs 8.7 (0.13) | 3.68 vs 4.02 | nothing |
| MASTER | stop | 11 vs 5.4 (0.12) | 2.69 vs 3.17 | 17 vs 4.7 (0.06) | 3.90 vs 4.10 | weak |
| MASTER | loser | 14 vs 5.7 (0.06) | 2.53 vs 3.15 | 2 vs 3.6 (0.36) | 3.13 vs 3.74 | weak |
| BT scored-only mcap/turnover | all | 0 vs 0.2 | 1.52 vs 2.28 | – | – | **nothing** |
| MASTER scored-only mcap/turnover | all | 0–1 vs 0.3 | 2.17 vs 2.36 | – | – | nothing |

**Not one P&L test, in any block, exceeds the family-wise null maximum.**

## 3 · Out-of-sample confirmation (frozen thresholds)

Confirmation rule: same sign, day-bootstrap P ≥ 0.90, and the zone on ≥ 5 days in the confirmation block. The confirmation count is
tested against a day-stratified shuffle of the confirmation block (200×).

| discover → confirm | pct | stop | loser |
|---|---|---|---|
| BT 2025 → BT 2026 | 0/7 (null 1.0, P 1.00) | **7/7 (null 1.0, P 0.00)** | 4/28 (null 4.1, P 0.55) |
| BT 2026 → BT 2025 | 0/71 (null 4.2, P 1.00) | **50/100 (null 9.8, P 0.005)** | 16/90 (null 5.4, P 0.055) |
| BT → MASTER | 2/3 (null 0.8, P 0.17) | 1/107 (null 9.9, P 0.86) | 3/61 (null 6.6, P 0.68) |
| MASTER → BT | **11/27 (null 2.6, P 0.035)** | 1/28 (null 2.4, P 0.57) | 4/16 (null 1.7, P 0.22) |

Two families survive.

- **The stop family is spike stretch** (`entry_ema_gap_5_8`, `entry_dist_from_ema13_pct`, `entry_ema5_stretch`,
  `entry_price_vs_ema5_pct`). It confirms both ways between eras but **not** on the master, and **not in P&L**.
- **The MASTER→BT P&L family is pair liquidity.** All 11 confirmations contain `7-day average daily volume < $4.89M`. The two BT→MASTER
  P&L confirmations are its mirror: 24 h volume ≥ $7.85M gives master 14 · 100 % · +0.78.

### 3a · Survivor 1: spike stretch predicts stops, not P&L

| `entry_dist_from_ema13_pct ≤ 1.525` (BT T1) | low stretch | rest |
|---|---|---|
| ① LIVE KMNO | 1.577: just above T1, **33rd–45th percentile** (low-stretch half) | |
| ② MASTER | 31 · WR 94 · stop 10 % · +0.305 | 43 · 86 · 16 % · +0.393 |
| ③ BT all | 208 · 80 · 15 % · +0.105 | 417 · 70 · 27 % · +0.038 |
| BT 2025 | 74 · 77 · 14 % · +0.119 | 149 · 62 · 31 % · +0.120 |
| BT 2026 | 134 · 82 · 16 % · +0.096 | 268 · 74 · 25 % · −0.008 |

A bigger stretch means a bigger, more violent move, so the fixed −1.5 % stop is hit more often. When it is not hit, the reversion is
bigger too, so P&L comes out even (2025) or slightly lower (2026, not confirmed). This is a stop-geometry fact, not an entry filter. Any
use of it would be a stretch-scaled stop, which is an exit counterfactual: it must be read on the live-stopped cohort only (CLAUDE.md,
Sep-23 rule) and is not proposed here.

### 3b · Survivor 2: pair liquidity (frozen zone: 7-day average daily volume < $4.89M)

| block | zone | rest | zone avg CI [day boot] | P(zone avg < 0) | Δ CI | P(Δ < 0) | worst pair / day |
|---|---|---|---|---|---|---|---|
| ① LIVE KMNO | **not in zone**: $8.26M (62nd–70th percentile) | | | | | | |
| ② MASTER | 37 · 28 d · WR 81.1 · stop 18.9 % · +0.126 | 37 · 97.3 · 8.1 % · +0.586 | [−0.10, +0.35] | 0.13 | [−0.80, −0.13] | 1.00 | BEL 16 % of zone loss |
| ③ BT all | 247 · 161 d · 71.3 · 23.9 % · −0.043 | 374 · 74.3 · 22.5 % · +0.130 | [−0.18, +0.10] | 0.73 | [−0.39, +0.02] | 0.96 | F 5 % · 2025-11-21 |
| BT 2025 | 63 · 58.7 · −0.176 | 157 · 69.4 · +0.234 | [−0.54, +0.19] | 0.83 | [−0.89, +0.06] | 0.96 | |
| BT 2026-H1 | 128 · 75.8 · +0.004 | 153 · 77.8 · +0.025 | [−0.16, +0.16] | 0.46 | [−0.25, +0.21] | 0.57 | |
| BT 2026-H2 | 56 · 75.0 · −0.003 | 64 · 78.1 · +0.127 | [−0.29, +0.26] | 0.52 | [−0.51, +0.24] | 0.76 | |

Master per batch (zone): B2 11 · 9W (BEL, SXT lost) · B3 10 · 8W (PORTAL, FORM) · B18 4 · 3W (KOMA) · B10 1 · 0W (ENSO) ·
B1 2 · 1W (AIN). B5, B7, B8, B9, B13, B16 and B17 had only winners. **7 of the 8 master losers sit in the zone**; the eighth, PLUME, is
at $5.6M, just above it. Five of the 8 losers fall in one 10-day window (Aug 6–15, B2/B3).

**Dose-response** (7-day average volume, $M: < 2 / 2–3.2 / 3.2–4.9 / 4.9–7.9 / 7.9–15.8 / > 15.8):
- master: +0.27 / −0.04 / +0.13 / +0.40 / +0.49 / +1.10;
- BT: +0.28 / −0.09 / −0.14 / +0.15 / +0.03 / +0.22.

**The thinnest bin wins in both blocks.** The curve is non-monotonic, and the effect lives in the $2–4.9M band.

**Expectancy bar (blocking the zone):**
- ① WR below breakeven: master 81.1 % vs 70.5 % ✗; BT 71.3 % vs 70.2 % ✗.
- ② avg < 0 at 95 %: master 0.13 ✗; BT 0.73 ✗.
- ③ ≥ 8 windows, no concentration ✓.
- ④ N ✓.

**Fails ① and ②. It is not a filter.** As a *difference* it is real (it confirms out of sample), and its mechanism (impact and
continuation on mid-thin pairs) is what the liquidity-capped ticket in the size/mcap study addresses.

## 4 · Pre-registered theory tests (`_prereg.csv`; zone avg vs rest avg, Δ with day-bootstrap P(Δ < 0))

| hypothesis (expected: zone worse) | KMNO | ② MASTER zone · Δ · P | ③ BT zone · Δ · P | BT 2025 / H1 / H2 Δ | verdict |
|---|---|---|---|---|---|
| H1 infl ≥ 2 (attention surge) | **in** | 7 · 71 % · +0.16 · −0.21 · 0.73 | 59 · 81 % · +0.19 · **+0.14** · 0.13 | +0.11 / +0.22 / +0.05 | **refuted in BT** (inflated = better) |
| H2 24 h volume low tercile | out | 40 · 85 % · −0.19 · 0.88 | 209 · 72 % · −0.00 · 0.49 | −0.14 / +0.00 / +0.22 | flat in BT |
| H3 turnover ≥ median | out | −0.12 · 0.75 | **+0.14** · 0.06 | +0.27 / +0.12 / +0.08 | refuted (opposite) |
| H4 pair 1h EMA20 > EMA200 | **in** | 9 vs 9 · −0.13 · 0.66 | +0.04 · 0.33 | +0.05 / +0.04 / +0.05 | refuted |
| H5 DI spread top tercile | out | +0.11 · 0.24 | −0.04 · 0.64 | +0.14 / −0.16 / −0.16 | inconsistent |
| H6 ADX < 20 and rising (ignition) | **in** | 19 · 95 % · +0.08 · 0.31 | +0.01 · 0.46 | +0.04 / −0.05 / +0.05 | refuted |
| H7 range position ≥ 95 | **in** | +0.13 · 0.33 | −0.04 · 0.63 | −0.12 / +0.04 / −0.08 | noise |
| H8 BTC 5m RSI < 40 (two-sided) | **in** | 32 · 91 % · **+0.29** · 0.04 | −0.08 · 0.82 | −0.20 / −0.08 / +0.13 | blocks disagree |
| H9 = gate 41h | **in** | 12 · 83 % · +0.11 · −0.29 · 0.90 | 121 · 77 % · +0.23 · **+0.21** · 0.04 | +0.44 / +0.07 / +0.07 | **BT contradicts** |
| H10 hot tape (BTC 1h RSI ≥ 70) | out | 2 fills | 8 fills | – | untestable (N) |
| H11 mcap < $50M | out | 33 · 82 % · −0.20 · 0.88 | −0.05 · 0.67 | **−0.50 (0.98)** / +0.14 / +0.02 | 2025 only |

KMNO falls in five of the theory zones: H1, H4, H6, H7, H9. **None of the five is worse than its complement in the backtest.** H8, the
only zone where the master reads clearly, says weak BTC is *better* for fades, and the backtest disagrees.

### Gate 41h tally (registered observe line; bar: zone N ≥ 20 ∧ WR ≤ 55 % ∧ Σ < 0, in window units)

- **Master + live under today's stack:** 12 master (12 days, WR 83 %, Σ +1.37 %) + KMNO −2.13 = **13 · 77 % · Σ −0.76 % · 13 days**.
  The bar is not met (N 13/20, WR 77 > 55).
- **Today's-stack replay contradicts the line:** zone 121 · 76.9 % · +0.231 vs rest 504 · 72.2 % · +0.019 (Δ +0.21, P(Δ < 0) 0.04).
  The zone is better in all three regimes. The Sep-18 registration evidence ("replay zone −0.49") came from an older replay/stack.
- Reported for the next batch review; the CURRENT_STATE line was not edited.

## 5 · KMNO on every axis (percentile within BT / MASTER; winners' vs losers' medians)

| stamp | KMNO | pctile BT / M | BT W / L median | read |
|---|---|---|---|---|
| 24 h volume | $17.8M | 97 / 100 | $5.4M / $4.5M | liquid side (good) |
| 7-day average volume | $8.26M | 62 / 70 | 6.80 / 6.75 log | liquid side |
| infl (24 h / 7-day average) | 2.15 | 92 / 92 | 0.83 / 0.78 | high; BT says high = better |
| mcap | $328M | 88 / 96 | | large (master ≥ $200M: 9/9 wins) |
| turnover V/mcap | 0.054 | 40 / 41 | | ordinary |
| ema5 stretch / dist EMA13 | 1.19 / 1.58 | 33 / 36 | 1.45 / 1.66 | **low-stop side** |
| pair ADX | 18.9 | 26 / 19 | 24.0 / 24.4 | ordinary in BT (master losers 30.8) |
| pair RSI | 83.2 | 86 / 92 | 75.9 / 77.8 | high |
| range position | 96.7 | 87 / 82 | 87 / 86 | no separation |
| BTC 5m RSI / BTC ATR % | 34.0 / 0.054 | 14 / 6 (BT) | 41 / 0.126 | weak, quiet BTC: no separation |
| bear breadth | 70.2 % | 60 / 69 | 62.8 / 63.8 | no separation |
| pair 1h EMA20/200 gap | +2.49 | 65 / 72 | 0.86 / 0.20 | BT losers are *lower* |

**Verdict: on this evidence the KMNO fill was not predictable at entry.** It was winner-shaped on the axes that carry any signal
(liquidity, stretch), and every theory zone it falls into is neutral or better in the backtest. The −0.58 % beyond the stop was our own
ticket against a thin book: sizing, as covered by the size/mcap study.

## 6 · Blind spots (what this screen could NOT test)

1. **Post-entry flow.** The $107k third-party sweep, order-flow imbalance, open interest, liquidations, news and listing announcements
   are not stamped and not in the replay. This is the KMNO mechanism, and it is invisible to every entry screen.
2. **Order-book depth / spread at entry.** `manual_ob_*` is manual-only, `entry_book_*` is 0 % covered, and `funding_rate` is not
   stamped. The size study's one depth snapshot is not per-fill.
3. **The master cannot test the BTC multi-timeframe context.** `btc_1d_ret`, `btc_4h_ema50_200_gap`, `btc_rsi_closed`, `eth_5m`,
   `off24h`, `off30d`, r72, `pair_1d_ndi` and `pair_1h_ema20_200` cover only 24–41 % of master fades. They were tested in BT only. A
   master rebuild from klines via `entry_feature_factory` was not done.
4. **Not stamped anywhere on fades:** `btc_1h_slope`, `btc_ema20_slope`, `global_volume_ratio`, `chop_burst`.
5. **Replay fidelity.**
   - The replay finds about half the live fades. The YR6_WL study found 35 of 73, and the 38 it misses average +0.49.
   - 2025 fades exit on 1m bars (non-top-50), so the BT stop rate (23 %) is about twice the master's (13.5 %). Stop-family results are
     replay-specific until real fills confirm them, and the master does not.
   - BT pct carries no live-impact model, apart from the tick-corrected stop tails.
6. **Power.** The master has 8 losers, so the minimum detectable effect is large; any absence of signal in the master is weak evidence.
7. **Single seed.** The BT seed 1 is the only YR6W seed carrying fades.
8. **Window units.** Market variables are clustered by UTC day, not by trigger window; fills from two windows on the same day count as
   one observation, which is conservative.

## 7 · Recommendation

- **Filter: none.** No P&L separator beats the shuffled null. The one OOS-confirmed P&L family (liquidity) fails expectancy-bar items
  ① and ② in both blocks and is non-monotonic.
- **Observe line: none new.** The liquidity mechanism is already being acted on through ticket sizing (size/mcap study). Shipping or
  observing a liquidity *filter* on top of that would break one-change-at-a-time attribution. The stretch→stop structure is recorded
  here as a fact for any future fade-stop study (live-stopped cohort, two-sided).
- **Gate 41h:** keep it open, and update the tally to 13 · 77 % · Σ −0.76 % · 13 days with KMNO. Flag at the next review that the
  today's-stack replay now points the opposite way (zone +0.23 vs +0.02, all three regimes). If the live zone reaches N ≥ 20 without
  WR ≤ 55 %, apply the DROP bar.
- **KMNO:** an unforecastable post-entry flow event. Its cost was magnified by ticket size relative to the book, so the lever is the size
  rule, not an entry rule.
