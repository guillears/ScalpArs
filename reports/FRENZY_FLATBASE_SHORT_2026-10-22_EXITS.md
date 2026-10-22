# FRENZY flat-base short: do quicker exits fix it?

Date: 2026-10-22. Research only. No code or config changed, and nothing was staged.

This follows up the main study (`FRENZY_FLATBASE_SHORT_2026-10-22.md`, REJECT).
- The rules are in `FRENZY_FLATBASE_SHORT_2026-10-22_EXITS_PREREG.md`, together with two revisions the operator and coordinator asked for:
  - a walk-forward test by half-year, in place of "pick on 2025";
  - a fair (non-anticipating) random-bar test, plus simple fixed-delay entries.
- The second revision was written down after the run that uses it. Its design had been specified before any result was seen.
- Scripts: `scripts/study_flatbase_short_exits.py`.
- Outputs:
  - `_EXITS_cells.csv`: all 504 cells, every period;
  - `_EXITS_trades.csv`: trades of the walk-forward picks;
  - `_EXITS_tables.md`: the full printed output.

## Verdict: NO. No shorter exit passes the ship bars out of sample, and this does not add a sleeve.

**Quicker exits do help a little**, mainly by making the losses smaller.
- The best pattern is consistent: short the flat coin on the first 4-hour close below the spike's average price (S4), take profit at **−8 %**, give it up to 48–72 h, with a 10–25 % stop.
- That makes about **+0.8 to +0.9 %/trade in 2026** after all costs and funding, against +1.16 for the main study's 7-day version.
- The worst trade shrinks to −16 % to −33 %, and the drawdown at the 5 %-risk size drops to roughly 40–45 % of the account. It was over 100 %.

**It still fails three things.**
1. **No single exit is chosen consistently.** In the walk-forward test each half-year picks a different exact cell. Three of the four picks share the same shape (S4, take profit −8 %, 72 h) and differ only in the stop. The stability bar was set at the exact cell, so it fails.
2. **Held-out results never clear zero.** None of the four picks has a held-out 95 % range above zero.
3. **The edge is no better than the fair random-entry benchmark.** In 2026 the best picks beat a random entry in the same flat episodes by only about +0.4–0.5 %/trade. The p-values are 0.05–0.21, and the bar is p < 0.05.

**Too short is worse.** Exits within 12 h make about 0 %/trade in every year. The money comes from the coin sliding back over 1–3 days, so "no need to hold" is not supported. What *is* supported is "no need to hold 7–14 days".

## Walk-forward (each half-year held out; the exit is chosen on the other three)

| held-out half | exit chosen on the other three | held-out result: N · WR · avg % [95 %] | fair random entry (same exit + costs + funding + one per pair) · p | best 5 % removed |
|---|---|---|---|---|
| ⚠ 2025-H1 | S4 · TP −8 % · 72 h · stop 15 | 357 · 66 % · **+1.00** [−0.21, +2.17] | +0.32 · 0.07 | +0.64 |
| ⚠ 2025-H2 | S4 · TP −8 % · 72 h · stop 10 | 783 · 57 % · **+0.19** [−0.45, +0.80] | +0.48 · 0.80 | −0.23 |
| 2026-H1 | D2 (first eligible bar 2 h after the spike) · TP −20 % · 48 h · stop 25 | 736 · 64 % · **+0.85** [−0.47, +2.11] | +0.33 · 0.17 | −0.16 |
| 2026-H2 | S4 · TP −8 % · 72 h · stop 25 | 381 · 72 % · **+0.46** [−0.77, +1.64] | −0.06 · 0.21 | +0.04 |

The pre-registered candidate rule was: chosen in at least 3 of 4 folds, and positive in both 2026 halves. **No cell meets it.** Every held-out mean is positive, which is encouraging. But every 95 % range includes zero, and none beats the fair random entry at p < 0.05.

## The four picks: before → after per block, against the main study's primary

The primary is S2 · 7 d · take profit at the pre-pump price · 25 % stop, from the main report. The 2026 numbers below are truly out of sample only for a pick chosen without that half. The walk-forward table above is the clean read.

| exit | ② master (real episodes) | ① live (real episodes; open = MTM) | 2026-H1 | 2026-H2 | 2026 total [95 %] vs fair random · p | max drawdown at $3,200 (2026) |
|---|---|---|---|---|---|---|
| main primary (S2, 7 d, base, stop 25) | 30 · −2.98 | 10 · +4.89 | +2.42 | −0.62 | — | −102 % (year) |
| **S4 · TP −8 · 72 h · stop 15** | 38 · +0.32 (window mean +0.80) | 13 · +1.64 | +1.08 | +0.66 | +0.93 [+0.28, +1.53] vs +0.39 · **0.050** | −$1,294 (−40 %), sum +$5,011 at $965 per short |
| S4 · TP −8 · 72 h · stop 10 | 38 · **−1.18** (window mean −1.05) | 13 · +2.83 | +0.77 | +0.76 | +0.77 [+0.23, +1.27] vs +0.24 · 0.04 | −$2,211 (−69 %) |
| D2 · TP −20 · 48 h · stop 25 | 37 · **−2.13** (window mean −0.18) | 13 · −0.56 | +0.85 | +1.48 | +1.07 [+0.10, +2.05] vs −0.00 · 0.02 | −$1,553 (−49 %) |
| S4 · TP −8 · 72 h · stop 25 | 37 · +2.10 (window mean +2.64) | 13 · −0.66 | +1.05 | +0.46 | +0.85 [+0.11, +1.56] vs +0.33 · 0.10 | −$1,442 (−45 %) |

Ship bars for the closest one (**S4 · TP −8 · 72 h · stop 15**):

| bar | result |
|---|---|
| N | ✅ |
| 2026 lower bound | ✅ |
| 2026-H1 and 2026-H2 both > 0 | ✅ |
| ①② windows: 17, mean +0.90, 9 positive | ✅ |
| haircut | ✅ |
| best 5 % removed | ✅ |
| concentration | ✅ |
| ② not negative | ✅ |
| **beats fair random, p 0.050** (needs < 0.05) | ❌ |
| **walk-forward stability: chosen once, not 3 of 4 times** | ❌ |

Its full year, 2025 included, draws down about 100 %, because 2025-H2 brings −21 % trades and one day carried 41 % of that half's profit. The "2026 total" for this cell is not fully out of sample either: it was chosen on folds that include 2026.

The tally across the family adds to the caution. 25 of the 504 cells beat their random benchmark in 2026 at p < 0.05, which is what chance alone predicts (5 %).

## Does "coin choice matters, timing is secondary" hold?
Mostly yes. The plain fixed-delay entries (D2 / D6 / D12 / D24) do about as well as the timed ones out of sample:
- D2 with TP −15 or −20 % and 24–72 h makes +0.7 to +1.1 in 2026;
- S4 with TP −8 % makes +0.8 to +0.9.

D2 was the walk-forward pick for 2026-H1. **But D2 is negative on ② master (−2.13)**, the block with priority. S4 holds up there (+0.3 to +2.1, except at the 10 % stop).

Neither beats the fair random-entry benchmark convincingly. "Short a flat coin some hours after its spike, take −8 to −15 %" is the whole effect, and it is a few tenths of a percent above random timing.

## Holding time (average 2026 result over all entries and stops)

| take profit | 12 h | 24 h | 48 h | 72 h |
|---|---|---|---|---|
| −5 % | +0.06 | +0.22 | +0.25 | +0.25 |
| −8 % | +0.10 | +0.39 | +0.49 | +0.49 |
| −10 % | +0.10 | +0.43 | +0.51 | +0.49 |
| −15 % | +0.07 | +0.42 | +0.43 | +0.36 |
| −20 % | +0.04 | +0.39 | +0.39 | +0.28 |
| trail (arm −8 %, trail 4 %) | +0.01 | +0.24 | +0.28 | +0.23 |

2025 has the same shape, but lower: 12 h is about 0 and 72 h is +0.25 to +0.38. The trailing stop is the weakest variant.

## ⚠ 2025, reported separately: why it is less reliable

| half-year | episodes | pairs | flat share | episodes on pairs now delisted | funding history missing | BTC move | BTC volatility |
|---|---|---|---|---|---|---|---|
| 2025-H1 | 1,139 | 324 | 33 % | **28 %** | 0.0 % | +13 % | 52 % |
| 2025-H2 | 2,368 | 504 | 34 % | 19 % | 0.0 % | −17 % | 35 % |
| 2026-H1 | 2,158 | 457 | 35 % | 8 % | 0.0 % | −32 % | 45 % |
| 2026-H2 | 1,029 | 356 | 39 % | 3 % | 0.0 % | +45 % | 36 % |

- **Delisted coins:** 28 % of 2025-H1 episodes are on coins that no longer trade. Their pre-delisting price and volume history is in the data, but these were thin, dying books where fills at the bar price are least realistic.
- **Reconstructed universe:** the engine's shortlist and blacklists are applied as configured today. The bot did not exist in this form in 2025, so the 2025 universe is a reconstruction.
- **Funding history** is complete in every period, so funding is not a reason to distrust 2025.
- **Regime:** 2025-H1 was a high-volatility rally and 2025-H2 a slow decline. Both differ from 2026.
- **Not measured:** missing-bar gaps in the 5m history.

Even judged on 2026 alone, the best picks do not beat the fair random entry at p < 0.05. So the failure is not caused by 2025.

## Correctness notes
- **Fair random benchmark.** The random benchmark here is non-anticipating: it draws a delay first, then enters at the first eligible bar using only data up to that bar. It uses the same exit, costs, funding and one-short-per-pair rule.
- **Correction to the main study's benchmark.** The main study's random benchmark looked into the future. It drew from bars that stay eligible over the whole 96 h, which depends on whether the coin's volume dies later. The main study did include costs and funding in that benchmark. It did not apply the one-short-per-pair rule.
  - The hostile review's fair version gives +1.27 %/trade [+0.32, +2.19] against the primary's +0.94.
  - A correction note was added to the main report. No other number there changed.
- **No entry or exit rule uses future eligibility.**
  - S2, S3, S4 and D-entries check the volume gate on the trailing 24 h at the entry bar.
  - The shortlist check uses the first-seen time at or before the entry bar.
  - The flat label uses only data from before the spike.
  - Exits use only bars after the entry.
- **Funding is charged on the entry notional,** not the live position value. Shorts pay most of their funding during squeezes, when the position is worth more than at entry, so the true cost there is higher than charged. When the price falls the charge is slightly too high. Overall this bias mildly **favours** the short, which makes the rejection conservative.
- **Engine parity, real fills and `validate_against_master.py`:** as in the main report. Validation passed, and no analysis tooling outside these study scripts changed.

## Blind spots
- Open interest, funding at entry and order-book depth are not available.
- Margin is modelled as isolated, while the operator runs cross margin.
- Prices are checked on 5m bars, so gaps inside a bar and borrow limits are not modelled.
- Block ① is 9 days long. Some trades there are still open and marked to market.
- In the random benchmark, the one-short-per-pair rule is applied in episode order, not strict time order (a rare overlap case).
