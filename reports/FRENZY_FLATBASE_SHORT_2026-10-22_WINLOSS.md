# FRENZY flat-base short: what separates winners from losers?

Date: 2026-10-22. Research only. No bot code or config was changed, and nothing was staged or committed.
- Rules were frozen before any result: `FRENZY_FLATBASE_SHORT_2026-10-22_WINLOSS_PREREG.md`. The operator's turnover and tighter-stop additions were written into it before the run.
- Script: `scripts/study_flatbase_short_winloss.py` (stages `fetch`, `build`, `screen`, `candidate`).
- Outputs: `_WINLOSS_trades.csv` (every trade with all features), `_WINLOSS_screen.csv`, `_WINLOSS_stops.csv`, `_WINLOSS_coverage.csv`, `_WINLOSS_wl_medians.csv`, `_WINLOSS_mcap_calibration.csv`, and the full printouts in `_WINLOSS_tables.md` and `_WINLOSS_candidate.md`.
- The trades reproduce the exit study exactly: 2,351 of 2,351 trades match, with a maximum difference of 0.0005 %.
- `validate_against_master.py`: **ALL CHECKS PASS**.

## Verdict

**Market conditions do not separate winners from losers.** One coin condition does: **turnover**.

**Market conditions.** I tested BTC trend, momentum and volatility, market breadth, alt-vs-BTC strength, the number of other pumps, and weekday, all counted in day units. None survives the random-label test; the best family-wise p is 0.16. Winners and losers start in the same market.

**Candidate filter: keep only shorts where 24 h volume > market cap (turnover > 1).**

On the closest cell (S4 · TP −8 % · 72 h · stop 15 %), in 2026:

| | WR | avg %/trade |
|---|---|---|
| without the filter | 68 % | +0.93 |
| with the filter | 73 % | +1.65 |
| with the filter, after the 40 % haircut | | about +1.34 |

- **Confirmed out of sample.** It was discovered on 2025 + 2026-H1 and confirmed on the held-out data, in these blocks:
  - 2026-H2: WR 68 → 71 %, avg +0.66 → +1.25.
  - ② master: WR 63 → 65 %, avg +0.32 → +1.04.
- **Not confirmed in ① live.** There it goes the other way (13 → 7 trades, WR 77 → 71 %), but on 13 trades that is noise.
- **First time this sleeve beats random timing.** With the filter on, the short now beats a fair random entry in 2026 (p = 0.01), even when the random entry gets the same turnover filter (p = 0.01).

**Why it is not a ship-ready filter:**
1. **It fails the locked expectancy bar.** What it blocks is about break-even rather than losing: +0.10 %/trade, WR 62 % against a 64 % breakeven, and only a 39 % chance that its true average is below zero. So it works as a *definition* of the sleeve (fewer, better trades), not as a way to remove losers.
2. **Discovery on 2026-H1 alone (variant B) does not pick it out family-wise** (p = 0.41), although the direction is the same.
3. **It cuts the trade count by more than half.** Total dollars at a fixed $190/short fall from +$1,974 to +$1,513 in 2026. The worst drawdown improves from −$354 to −$225.

**Tighter stops: no.** A 5 % or 8 % stop costs more winners than it saves in losses:
- the 2026 average falls from +0.93 (15 % stop) to +0.42 (8 %) and +0.09 (5 %);
- 39–47 % of the trades the tight stop closes later reach the −8 % take-profit inside the 72 h anyway.

## 1. Theory first: why a flat-base pump short would fail or succeed

| mechanism | predicts the short FAILS when… | measured by | result |
|---|---|---|---|
| Squeeze fuel (crowded shorts) | funding is negative at entry | funding at entry, 24 h sum, trend | **Opposite of the theory.** Winners had *more* negative funding (median −0.033 vs −0.006 %/8 h). The "squeeze" is not visible at entry. Negative funding does cost money over the hold, since shorts pay it. |
| BTC rallies during the hold | BTC goes up after entry | BTC return over the hold (look-ahead, not tradable) | **Real but unusable.** Losers' median BTC over the hold is +0.44 %, winners' −0.31 % (rank correlation −0.29). Nothing measured at entry predicts it. |
| BTC regime at entry | BTC is trending up or strong | 1/4/24 h and 7 d return, EMA20/50 gap, ADX, RSI, distance from the 24 h high/low, volatility | No separator. Best day-unit FW p = 0.16. |
| Breadth / alt season | alts broadly rising | share of perps up over 24 h / 7 d, alt − BTC | No separator. Low breadth looks mildly worse, FW p ≥ 0.16. |
| News-driven, sustained bid | volume keeps coming | volume decay (4 h vs peak), volume vs pre-pump | No separator on its own. |
| **Speculative churn (operator's turnover idea)** | low turnover = a real, "owned" move | **24 h volume ÷ market cap** | **The separator.** Turnover > 1 collapses more often: TP is hit on 67 % of trades vs 48 %, and only 12 % run out the clock vs 30 %. |
| Pump size / stretch | big pumps keep going | pump %, % above base, % below VWAP, retrace, time since peak | Bigger pumps are *better* shorts. This is mostly the same signal as turnover (see §3). |
| Size / liquidity / age | big coins hold gains | mcap, 24 h volume, listing age | Big-mcap coins are worse shorts, again largely the turnover effect. |
| Sector pumping together | many pumps at once | number of FRENZY spikes in the prior 24 h | No separator. |
| Hour / weekday | – | 4 h block, weekday | No separator. |

**Open interest** is not tested. Binance's public OI history covers only about 30 days, which would score about 50 trades (① and ②) and could never pass a gate.

## 2. Screens (variant A: discover on 2025 + 2026-H1, 1,867 trades; confirm on 2026-H2 + ② + ①, 436 trades)

| family | tests | best | family-wise p (day-preserving / plain shuffle) | out of sample |
|---|---|---|---|---|
| 1D coin | 271 | **turnover ≤ 1 blocked**: WR 62 vs 72 | **0.016 / 0.011** | blocked 61 % vs kept 70 %, avg −0.58 vs +1.21, p = 0.036. 2026-H2 alone 60 vs 71 %. All 4 neighbour cells agree. **CONFIRMED** |
| 1D coin | | volume vs pre-pump ≤ 4.7×, pump ≤ 19 %, peak turnover ≤ 1, turnover low tercile/quintile | 0.017–0.043 | same direction in confirmation (p 0.03–0.07). All are the same "churn" signal |
| 1D market (trade units) | 162 | low breadth (≤ 15 %) | 0.47 / 0.22 | none survives |
| 1D market (DAY units, 460 days) | 162 | low breadth, BTC ADX > 35 | 0.16 | none survives |
| 2D (8,102 tests) | | coin 1 h return mid-tercile × coin 24 h return low tercile (WR 51 vs 68) | **0.003** | **FAILS OOS**: blocked 62 % vs 68 %, but avg +1.54 vs +0.59, p = 0.32. In-sample luck |
| 2D, next 9 | | various turnover × flatness / volume pairs | 0.10–0.43 | none is a family-wise survivor |

**Variant B (discover on 2026-H1 only, 730 trades) finds no family-wise survivor** in any family.
- Its top 1D result is again low turnover: peak turnover in the low tercile, WR 60 vs 73 %, FW p = 0.086, confirmed out of sample with p = 0.021.
- Turnover ≤ 1 itself has FW p = 0.41 in variant B, with the same direction (64 vs 74 %).
- So the signal shows up with 2025 in the discovery set, and when it is left out the direction holds but the statistics weaken.

## 3. Is turnover just something else in disguise?

Inside every tercile of each competing variable, turnover > 1 is still the better short (main cell, all periods):

| within terciles of… | turnover > 1 WR / avg | turnover ≤ 1 WR / avg |
|---|---|---|
| market cap: small / mid / large | 72 % +1.68 · 72 % +1.54 · 67 % +0.02 | 64 % +0.75 · 63 % +0.48 · 60 % −0.45 |
| pump size: small / mid / large | 69 % +1.21 · 66 % +0.62 · 74 % +1.91 | 59 % −0.09 · 62 % +0.08 · 66 % +0.50 |
| volume vs pre-pump | 65 % +0.62 · 70 % +1.37 · 74 % +1.78 | 60 % −0.24 · 60 % −0.13 · 70 % +1.22 |
| FLAT14 | 70 % +1.11 · 75 % +2.14 · 69 % +1.15 | 60 % +0.15 · 64 % +0.37 · 60 % −0.33 |

It also holds within 24 h volume terciles and coin 24 h return terciles (see `_WINLOSS_candidate.md`). Turnover is not just a small-cap or big-pump proxy. It carries information of its own.

**Concentration:**
- Kept cohort: 726 trades on 273 pairs. The top pair is 8 % of profit (RARE) and the top day 6 %.
- With the best 5 % of trades removed, it still averages +1.07.
- Blocked cohort: no pair and no day carries more than 2 % of its loss.

## 4. Before → after, the candidate filter "turnover > 1"

Main cell S4 · TP −8 · 72 h · stop 15, at 1× and $190 per short.
- "After" blocks trades with no market cap (fail-closed). The bot stamps market cap on real fills, so ① and ② are 100 % scored.
- Drawdown is realised P&L by exit time, with no concurrency cap.

| block | before: N · WR · avg % [95 % day CI] · $ · worst · max DD | after: N · WR · avg % [CI] · $ · worst · max DD | blocked: N · WR · avg | WR lift |
|---|---|---|---|---|
| ① live (open = MTM) | 13 · 77 % · +1.64 [−4.6, +7.4] · +$41 · −16.1 · −$88 | 7 · 71 % · +0.79 · +$11 · −16.1 · −$59 | 6 · 83 % · +2.63 | −5.5 pp |
| ② master | 38 · 63 % · +0.32 [−2.9, +3.9] · +$23 · −16.5 · −$135 | 23 · 65 % · +1.04 · +$45 · −16.5 · −$88 | 15 · 60 % · −0.77 | +2.1 pp |
| ③ 2025 ⚠ | 1,137 · 64 % · +0.52 [−0.2, +1.2] · +$1,120 · −21.2 · −$602 | 235 · 69 % · +1.12 [−0.3, +2.4] · +$498 · −20.7 · −$217 | 652 · 61 % · +0.05 | +5.1 pp |
| ③ 2026-H1 | 730 · 68 % · +1.08 [+0.3, +1.8] · +$1,493 · −21.0 · −$354 | 295 · 74 % · +1.91 [+0.8, +3.1] · +$1,071 · −21.0 · −$141 | 362 · 64 % · +0.44 | +6.0 pp |
| ③ 2026-H2 (held out) | 385 · 68 % · +0.66 [−0.4, +1.7] · +$481 · −17.1 · −$314 | 187 · 71 % · +1.25 [−0.1, +2.4] · +$443 · −17.1 · −$225 | 154 · 60 % · −0.69 | +3.1 pp |
| ③ 2026 total | 1,115 · 68 % · +0.93 [+0.3, +1.5] · +$1,974 · −21.0 · −$354 | 482 · 73 % · +1.65 [+0.8, +2.5] · +$1,513 · −21.0 · −$225 | 516 · 63 % · +0.10 | +4.8 pp |

**Fair random entry in the same episodes** (same exit, costs, funding, one short per pair), with the same turnover > 1 rule applied at the random bar:
- 2026: random +0.36 vs filtered S4 +1.65, p = 0.01.
- 2025: random −0.06 vs +1.12, p = 0.08.
- 2026-H2: random −0.17 vs +1.25, p = 0.06.

Without the filter, the sleeve beat random timing only marginally (p ≈ 0.04–0.05).

**Expectancy bar (locked), applied to the blocked cohort — FAIL in every set:**
- Discovery: WR 62.2 % vs a 64.0 % breakeven. Avg is +0.22, P(avg < 0) = 0.26, over 401 windows.
- Confirmation: WR 60.3 %, avg −0.63, P(avg < 0) = 0.76.
- All periods: avg +0.10, P(avg < 0) = 0.39.
- Concentration and N pass.
- **Haircut:** the kept-vs-all lift is +0.68 %/trade, or +0.41 after the 40 % haircut.

**Unscored trades.** These are coins with no Info-panel market cap, mostly delisted. They are not "rest", and on their own they are good shorts: 2026 117 trades, 72 %, +1.62. A fail-closed rule would lose them.

**Other thresholds (all FAIL the expectancy bar):**
- *Block only the lowest turnover tercile (≤ 0.37).* WR 68 → 71 %, avg +0.93 → +1.44 in 2026. Blocked cohort avg −0.37, P(avg < 0) = 0.82.
- *Peak turnover > 1.* Nearly identical to turnover > 1.

### Under each stop (2026 total, before → after turnover > 1)

| stop | before: WR · avg | after: WR · avg · beats random-with-filter p |
|---|---|---|
| 5 % | 43 % · +0.09 | 47 % · +0.60 · 0.08 |
| 8 % | 56 % · +0.42 | 61 % · +1.27 · 0.03 |
| 10 % | 62 % · +0.77 | 67 % · +1.61 · 0.01 |
| **15 %** | **68 % · +0.93** | **73 % · +1.65 · 0.01** |
| 25 % | 72 % · +0.85 | 79 % · +1.82 · 0.00 (worst trade −31 %) |

The filter lifts every stop. ② master agrees at every stop from 5 % to 15 %. ① live (N = 7 vs 6) does not.

## 5. Tighter stops (S4 · TP −8 · 72 h), full per-block table

| stop | block | N | WR | avg % [95 % day CI] | $ at $190 | worst % | max DD | stopped, then later hit −8 % | fair random · p |
|---|---|---|---|---|---|---|---|---|---|
| 5 % | ① live | 13 | 54 % | +1.45 [−1.2, +4.3] | +$36 | −5.4 | −$30 | 2 of 6 | – |
| 5 % | ② master | 38 | 32 % | −1.24 [−3.3, +0.8] | −$89 | −5.6 | −$133 | 15 of 26 (58 %) | – |
| 5 % | ③ 2025 ⚠ | 1,144 | 44 % | +0.31 [−0.1, +0.7] | +$676 | −7.2 | −$402 | 219 of 614 (36 %) | +0.26 · 0.40 |
| 5 % | ③ 2026-H1 | 734 | 43 % | +0.06 [−0.5, +0.6] | +$84 | −8.2 | −$430 | 179 of 398 (45 %) | +0.10 · 0.54 |
| 5 % | ③ 2026-H2 | 385 | 43 % | +0.13 [−0.6, +0.9] | +$97 | −8.9 | −$277 | 108 of 215 (50 %) | −0.07 · 0.33 |
| 8 % | ① live | 13 | 69 % | +2.51 [−3.3, +7.4] | +$62 | −8.7 | −$48 | 0 of 4 | – |
| 8 % | ② master | 38 | 45 % | −1.05 [−4.0, +1.5] | −$76 | −9.3 | −$103 | 10 of 20 (50 %) | – |
| 8 % | ③ 2025 ⚠ | 1,143 | 55 % | +0.51 [+0.0, +1.0] | +$1,107 | −14.1 | −$552 | 125 of 442 (28 %) | +0.45 · 0.38 |
| 8 % | ③ 2026-H1 | 734 | 55 % | +0.27 [−0.3, +0.9] | +$370 | −12.2 | −$369 | 108 of 294 (37 %) | +0.26 · 0.50 |
| 8 % | ③ 2026-H2 | 385 | 57 % | +0.71 [−0.2, +1.6] | +$520 | −10.6 | −$278 | 63 of 148 (43 %) | +0.05 · 0.07 |
| 15 % (ref) | ① live | 13 | 77 % | +1.64 [−5.1, +7.4] | +$41 | −16.1 | −$88 | 0 of 3 | – |
| 15 % | ② master | 38 | 63 % | +0.32 [−3.0, +3.9] | +$23 | −16.5 | −$135 | 3 of 10 | – |
| 15 % | ③ 2025 ⚠ | 1,137 | 64 % | +0.52 [−0.1, +1.2] | +$1,120 | −21.2 | −$602 | 55 of 247 (22 %) | +0.57 · 0.57 |
| 15 % | ③ 2026-H1 | 730 | 68 % | +1.08 [+0.3, +1.8] | +$1,493 | −21.0 | −$354 | 32 of 141 (23 %) | +0.47 · 0.07 |
| 15 % | ③ 2026-H2 | 385 | 68 % | +0.66 [−0.4, +1.6] | +$481 | −17.1 | −$314 | 29 of 93 (31 %) | +0.05 · 0.14 |

**2026 totals:**

| stop | WR | avg %/trade | fair random | p |
|---|---|---|---|---|
| 5 % | 43 % | +0.09 | +0.04 | 0.43 |
| 8 % | 56 % | +0.42 | +0.18 | 0.20 |
| 10 % | 62 % | +0.77 | +0.19 | 0.04 |
| 15 % | 68 % | +0.93 | +0.32 | 0.04 |

The 10 %, 25 % and TP −10 % rows are in `_WINLOSS_stops.csv`.

**In plain English:** the tight stop does cap the worst trade, from −21 % down to −9 % or −14 %. But it closes many shorts that would have won: 36–50 % of 5 %-stop-outs went on to hit the −8 % target within 72 h. Win rate falls from 68 % to 43–56 %. The average falls to roughly zero at 5 % and to half at 8 %, and neither tight stop beats a random entry. **A tighter stop costs more winners than it saves in losses.** ② master is the one block where the 5 % and 8 % stops lose money outright (−1.2 % and −1.1 %/trade).

## 6. Conditions that only ADD trades (a separate question, not a recommendation)

Loosening flatness beyond the frozen 44.4 % does not help:
- **44–55 % band:** 2025 +0.23, 2026-H1 +0.02, 2026-H2 +2.17, ① −15.2 on 3 trades.
- **55–70 % band:** negative in every ③ period (−0.29 / −1.06 / −1.09).

Flatness stays the gate.

## 7. Market cap source and coverage (turnover)

- **Market cap at entry** = the Binance Info-panel circulating supply (the same CMC source as the bot's `mcap_service`) × the entry price. It is the established proxy from the Oct-8 turnover study.
  - Supply came from that study's cache, and the pairs missing from it were fetched today: 1 request per symbol, ≥ 1.5 s spacing, no 418/429, cooldown honoured. The supply is readable for 463 of 605 pairs.
  - The rest have no market cap on the endpoint, mostly because they are delisted. Supply is saved in `FRENZY_FLATBASE_SHORT_2026-10-22_data/winloss_supply.json`.
- **Calibration against the bot's real stamps:** 221 stamped fills on 90 cohort pairs (master pool + live export). Proxy ÷ stamped market cap has a median of **1.003** (IQR 0.996–1.018), and 92 % fall within ±10 %.
- **Coverage of cohort trades:**

| block | turnover scored |
|---|---|
| ③ 2025 | 78 % |
| ③ 2026-H1 | 90 % |
| ③ 2026-H2 | 89 % |
| ② master | 100 % |
| ① live | 100 % |

  Unscored trades are excluded from both sides of every test.
- Every other feature is scored on ≥ 92 % of trades (DIST90 in 2025 is the lowest). Most are at 100 %. Funding history is complete.

## 8. Blind spots (what this could NOT test)

- **Open interest**: not tested (only about 30 days of public history).
- **Historical circulating supply.** Today's supply is used for 2025 trades. Token unlocks since then make the 2025 market cap too high and the turnover too low, so some 2025 "turnover > 1" trades are labelled ≤ 1. The 2026 blocks are much less affected.
- **Delisted coins** have no market cap: 22 % of 2025 is unscored. That unscored group is profitable, so fail-closed would cost trades.
- **No sector labels.** "Sector pumping together" is proxied by the count of all FRENZY spikes.
- **No order-book depth, no spot-vs-futures volume split.** Turnover uses futures volume only, as the bot stamps it.
- **BTC over the hold is the strongest loser signature, but it is look-ahead.** Nothing at entry predicts it.
- **① live is 13 trades** (7 kept): it can neither confirm nor refute anything.
- **The fair random entry's one-short-per-pair rule** is applied in episode order (a rare overlap case).
- **Funding is charged on entry notional.** This slightly favours the short.
- **Margin** is modelled as isolated, while the operator runs cross.
- **Gaps inside 5m bars** are not modelled.
- **2D uses terciles only.** Quintile × quintile was not run, because the cells would be too small at this N.

## What I would do with this (no change made)

Turnover > 1 is the strongest, most robust thing found in this sleeve's data. It matches the operator's theory, holds inside every competing variable, confirms on held-out 2026-H2 and ② master, and is the first rule that makes the sleeve clearly beat random timing.

It is still not shippable as a filter: the locked expectancy bar fails, because what it blocks is break-even, not losing.

The honest route is to treat "flat base + turnover > 1" as a **re-defined candidate sleeve** and put it through the exit study's walk-forward and ship bars (pre-registered, threshold frozen at 1.0), then observe it on real FRENZY episodes. Its ① live read is currently 7 trades.
