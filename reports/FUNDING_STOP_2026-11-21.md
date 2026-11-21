# FUNDING_STOP — should the short sleeves' stop count accrued funding? (2026-11-21)

Research only. No engine / service / config / template change, nothing committed.
Scripts: `scripts/FUNDING_STOP_2026-11-21_{fetch,common,real,bt,report}.py` · data `reports/FUNDING_STOP_2026-11-21_data/` · full tables `reports/FUNDING_STOP_2026-11-21_tables.md` (generated, pasted under the summary below).

## 0. Pre-committed decision rule (written before any result was read)

Per sleeve: **recommend B** if on REAL fills the mean Δ (B − A) ≥ 0 **and** SAVED $ ≥ CUT $, **and** the backtest HOLDOUT day-block 95 % CI of the mean Δ does not lie entirely below 0. Otherwise **keep A** and report arm C (entry skip) as the alternative lever.

## 1. Arms and how they were walked

- **A (today):** stop when the PRICE move from the fill reaches −SL. This is what `services/flatbase|exhaustion|distribution *_exit_for` do. Note: the engine stop is price only. It does not count fees either. The Open Orders % is price − fees (entry + estimated exit) + funding so far.
- **B (net stop):** stop when price move % + funding accrued so far (signed: a short receives a positive rate) reaches −SL. The stop line in bar j is fill × (1 + (SL + F_j)/100). F_j is updated at each real settlement, on the bar that settlement opens. The fill is at max(line, bar open) as a taker market order.
- **Bp (paid-only):** same as B, but received funding never moves the line out (F_j → min(F_j, 0)). This is a sensitivity check and was not pre-registered.
- Walk details: the stop is checked first on the bar HIGH. The maker TP fills only when the low trades ≥ 0.05 % through the line, and it fills at the line. Time exits happen at the last close inside the hold. Each fill keeps its own TP / SL / hold: FB 5 / 10 / 72 h; EXH and DIST use their stamped levels (backtest: min(20 × ATR, 10) / min(1.5 × ATR, 6)) / 24 h.
- Costs: entry taker 0.05 % · maker TP 0.02 % · stop / time taker 0.05 % + slip. Slip is 0.25 % for real FB (the TP_GRID FB real-slip figure) and 0.10 % for real EXH / DIST. Backtest rows use their own slip and fee conventions. Funding is booked as Σ settlements in (fill, exit bar open], each one = rate × price ÷ fill.
- Real-fill walks start the minute AFTER the fill. FB fills opened before FB_CONFIRM use the rebuilt confirm entry, as in TP_GRID FB.
- One size is X = 0.61 × equity for all three sleeves (as in the TP_GRID 11-20 studies). "$" = Σ Δ % × 61, i.e. dollars on a $10,000 account. DCR-cal counts every calendar day, with 0 on days with no fill.
- Cohorts: **FB real** = master `stack_keep` + B28 (TP_GRID FB cohort, 31) + **KERNEL #25 added by hand** (−11.69 % booked) = 32. That is **28 on today's stack** after the DL 601 funding-above-base cut; 15 of them have MODEL_CONFIRM entries. **EXH / DIST real** = the 5 + 5 closed fills in `scalpars_orders_paper_2026-11-20_16-01-39.csv`; the master has none. No probes in any cohort, CLOSED only, deduplicated on (opened_at, pair, direction). **Backtest:** FB = the OOS_REAL_FLATBASE NEAR_BASE rebuild on today's stack (545, 5 m bars). EXH = TP_GRID rows_FRENZY (3,368, 1 m). DIST = the DL 586 kept cohort (1,547, 1 m). DESIGN = entries before 2025-07-01; HOLDOUT = entries from 2025-07-01 on.

## 2. Parity (arm A replay vs what was booked / recorded)

| cohort | check | result |
|---|---|---|
| FB real (32) | A in paper costs vs booked (stack_pct / pnl %) | mean \|gap\| **0.021** pts (median 0.001, max 0.188 MMT: the booked stop filled better) · exit kind matches the re-priced booking on every fill |
| EXH real (5) | same | mean \|gap\| **0.045** (max 0.179 ZKC, funding timing) · kind 5 / 5 |
| DIST real (5) | same | mean \|gap\| **0.015** (max 0.036) · kind 5 / 5 |
| FB real, live-cost arm A vs booked | (A carries 0.25 % stop slip + 0.05 / 0.02 fees) | mean −0.20 · median −0.05 |
| FB backtest (614) | parity walk vs X0_net | **exact on 614 / 614** (max \|gap\| 0.0000) |
| EXH backtest (3,368) | parity walk vs sim_FRENZY REF | gross mean \|gap\| **0.004**, funding 0.001 · kind 3,364 / 3,368 |
| DIST backtest (1,547) | parity walk vs dist.pkl net | mean \|gap\| **0.012** (median 0.000) · kind 1,545 / 1,547 |
| every cohort | fills where B's path = A's path (same bar, same non-stop reason) | **Δ = 0 exactly: 0 violations** in every block (column `exactness_violations`) |

## 3. Summary per sleeve (arm B unless stated; Δ = B − A per fill, % of notional)

| sleeve | cohort | N | A avg % | B avg % | mean Δ (95 % day-block CI) | P(Δ>0) | DCR A → B %/day | SAVED N · $ | CUT N · $ | rule |
|---|---|---|---|---|---|---|---|---|---|---|
| FLATBASE | REAL (today's stack) | 28 | +3.09 | +3.14 | **+0.06** (−0.00 … +0.19) | 0.65 | +1.69 → +1.73 | 3 · **+$97** (all KERNEL) | 0 · $0 | real leg PASS |
| FLATBASE | REAL without KERNEL | 27 | +3.65 | +3.65 | −0.00 | — | +1.95 → +1.95 | 2 · −$0.2 | 0 | (LOO) |
| FLATBASE | BT DESIGN | 94 | +1.30 | +1.35 | +0.05 (+0.01 … +0.10) | 1.00 | +0.18 → +0.19 | 16 · +$278 | 0 | |
| FLATBASE | BT HOLDOUT | 451 | +1.10 | +1.11 | +0.01 (**−0.06 … +0.05**) | 0.69 | +0.52 → +0.53 | 58 · +$1,059 | 1 · −$775 (ORCA) | HOLDOUT leg PASS |
| FLATBASE | BT ALL | 545 | +1.14 | +1.15 | +0.02 (−0.04 … +0.06) | 0.76 | +0.38 → +0.39 · max DD −41.3 → −37.5 % | 74 · +$1,338 | 1 · −$775 | **B PASSES** |
| EXHAUSTION | REAL | 5 | +5.08 | +5.08 | 0.00 (no fill changed) | — | +3.84 → +3.84 | 0 | 0 | real leg vacuous |
| EXHAUSTION | BT DESIGN | 1,349 | +0.57 | +0.57 | +0.00 (−0.04 … +0.04) | 0.59 | +0.48 → +0.48 | 311 · +$3,652 | 9 · −$4,307 | |
| EXHAUSTION | BT HOLDOUT | 2,019 | +0.74 | +0.69 | **−0.05 (−0.12 … +0.01)** | **0.05** | +1.37 → +1.26 | 393 · +$10,666 | 22 · −$16,885 | |
| EXHAUSTION | BT ALL | 3,368 | +0.67 | +0.65 | −0.03 (−0.07 … +0.01) | 0.08 | +0.84 → +0.80 | 704 · +$14,318 | 31 · −$21,192 | **keep A** (see §6) |
| DISTRIBUTION | REAL | 5 | −2.32 | −2.33 | **−0.01** (EVAA −0.07) | 0.00 | −1.87 → −1.88 | 1 · −$4 | 0 | **real leg FAIL** |
| DISTRIBUTION | BT DESIGN | 559 | +0.98 | +0.96 | −0.02 (−0.09 … +0.03) | 0.33 | +0.39 → +0.38 | 81 · +$661 | 2 · −$1,236 | |
| DISTRIBUTION | BT HOLDOUT | 988 | +0.87 | +0.84 | −0.03 (−0.09 … +0.02) | 0.14 | +0.85 → +0.81 | 119 · +$1,726 | 5 · −$3,465 | |
| DISTRIBUTION | BT ALL | 1,547 | +0.91 | +0.89 | −0.02 (−0.07 … +0.01) | 0.11 | +0.58 → +0.56 | 200 · +$2,388 | 7 · −$4,701 | **keep A** |

Paid-only **Bp** is the same as B or slightly better in every FLATBASE block: real +0.06 (KERNEL only), HOLDOUT +0.01 (−0.06 … +0.06), ALL +0.02, max DD −37.5 %. It removes B's 1 HOLDOUT "WIDENED" fill. It does not rescue EXH (HOLDOUT −0.05, P 0.05) or DIST (HOLDOUT −0.03, P 0.15). The real DIST fill that B worsened (EVAA) is unchanged under Bp.

## 4. What the numbers say (mechanism)

- **How funding behaves.** Most fills pay nothing: the median funding paid is 0.00 % in every sleeve and cohort. The tail is real, though. FB backtest p90 is 0.58 % and the worst is 5.5 %. EXH p90 is 0.69 % (worst 4.8 %), DIST 0.23 % (worst 4.7 %). Hourly funding at entry: FB 9 % of fills (real 11 %), EXH 2.5 %, DIST 1.4 %. **Stopped fills do NOT pay more funding than winners.** FB backtest: stops 0.18 % vs TPs 0.21 %. EXH: stops 0.12 % vs TPs 0.30 %. The entry funding rate predicts the funding paid (rank correlation +0.4 … +0.7: a more negative entry rate means more paid).
- **FLATBASE (10 % stop, 72 h hold, 5 % TP).** Winners usually reach +5 % long before price nears −10 %, so a funding-tightened line rarely catches a winner: 1 CUT in 545 fills (ORCA 2025-11-27, hourly funding −0.09 %/h, B stopped it at 19:10 on its way to a +2.2 % TP). Losers that stop anyway pay their funding on the way, and B saves roughly that amount: 74 SAVED fills, +$1,338. The net effect is small and positive: +0.02 %/fill, about +0.01 %/day DCR, and a lower worst drawdown.
- **EXHAUSTION / DISTRIBUTION (≤ 6 % ATR stops, 24 h hold, 10 % TP).** Many winners come within a hair of their stop before running to TP. Shaving even 0.1–1 % off the line converts those near-misses into full stops. EXH: 31 CUTs averaging −11 pts each outweigh 704 small SAVES averaging +0.33 each. DIST shows the same pattern (7 CUTs at −11 vs 200 SAVES at +0.20). A tighter stop on these sleeves is a stop-width change, and the EXH / DIST stop-width studies already priced that.
- **KERNEL #25, re-walked on 1 m + real hourly settlements.** Funding accrued was −1.595 % ($108 on $6,777, matching the booking). The price-plus-funding net first reached −10 % at **01:06 UTC**, **one minute before** the price stop (01:07). Around midnight UTC, the same net, with fees included as in Open Orders, read about −6.5 … −7.5 %. So **the "crossed −10 % around midnight" is NOT reproduced** from Binance 1 m data. A time-zone reading of 01:06 UTC is the likeliest explanation. B's saving on KERNEL (+1.60 pts) comes only from the final spike blowing through both lines within 2 minutes. B would not have exited hours earlier.

## 5. Per-pair concentration

- FB real: the entire +1.60 Σ Δ is KERNEL (100 %). This is the fill that motivated the question, so the real leg is in-sample by construction (DL 561 "never refit around the trade that broke it"). Without it the real Δ is exactly 0.
- FB backtest: 66 pairs move. The top gainer, LRC (+3.36), is 15 % of Σ+. One pair, ORCA (−12.71), is 95 % of Σ−.
- EXH backtest: 305 pairs move. The top loser, SXP, is 8 % of Σ−, so the loss is spread across pairs, not concentrated.
- DIST backtest: 171 pairs. The top loser, LEVER, is 20 % of Σ−.

## 6. Decision (pre-committed rule applied)

- **FLATBASE_SHORT → rule PASSES for B.** Real Δ +0.06 ≥ 0, SAVED $97 ≥ CUT $0, and the HOLDOUT CI (−0.06 … +0.05) is not entirely below 0. Bp passes too, and is equal or better in every block. **Recommendation:** build the net stop for FLATBASE in its **paid-only form (Bp)**. The line can only move IN, never past today's −10 % price stop, so it cannot make the existing tail worse. Choosing Bp over B was decided after seeing the results; the two differ by ≤ 0.01 %/fill, which is noise, and the choice is a design preference, not evidence. **Size of the prize: small and UNVERIFIED.** It is about +0.01–0.05 %/fill in the backtest, and the only real-fill evidence is the motivating fill. After a 30–50 % in-sample haircut it is roughly +0.01 %/fill. Its value is as a cost guard on hourly-funding squeezes, not as a profit lever. Proposed revert gate (alert-only, frozen): price every FLATBASE fill where Bp exits before the price stop would have, against the price-only stop. At the first 5 such fills, Σ Δ < 0 → REVERT candidate. Any single CUT (a fill the price stop would have carried to TP) → REVIEW.
- **EXHAUSTION_SHORT → keep A.** Taken literally, the rule "passes": no real fill changed (Δ = 0, SAVED = CUT = $0) and the HOLDOUT CI upper bound is +0.01. But the real leg carries zero information. The only informative evidence points against B in both eras: HOLDOUT −0.05/fill, P(B better) 0.05, and CUT $ > SAVED $ in DESIGN and HOLDOUT. I do not treat a vacuous pass as a pass. This is flagged so the operator can overrule.
- **DISTRIBUTION_SHORT → keep A.** The rule FAILS on real fills: Δ −0.01 (EVAA: received funding moved B's line out and it stopped later at a worse price). Bp's real leg is vacuous, and its backtest is negative in both eras.
- **Arm C (entry skip: hourly funding ∧ entry rate against the short), context only:**

| cohort | C-skip N · WR · avg A | rest N · WR · avg A |
|---|---|---|
| FB real | 3 · 67 % · −0.70 (KERNEL, ORDER, T) | 25 · 92 % · +3.54 |
| FB backtest | 47 · 77 % · **+0.65** (paid avg 0.90 %) | 498 · 76 % · +1.18 |
| EXH backtest | 83 · 52 % · **+1.41** | 3,285 · 42 % · +0.66 |
| DIST backtest | 19 · 63 % · **+4.24** | 1,528 · 42 % · +0.87 |

  C would skip positive-expectancy fills in every backtest cohort. On EXH and DIST it would skip the better half. For FB the C cohort is weaker but still positive (+0.65), far from the expectancy-filter bar. **Arm C is not a supported lever.** If anything it would be an observe line for FB only.

## 7. Limitations / blind spots

1. **Real-fill evidence is thin.** FB has 28 fills on today's stack, and only one changes materially: KERNEL, the motivating case, so it is in-sample. EXH and DIST have 5 fills each and none (EXH) or one (DIST) changes. The master holds no EXH / DIST fills.
2. 15 of the FB real fills have MODEL_CONFIRM entries (the confirm bar rebuilt on 1 m with +0.25 % drift), not real fill prices.
3. The FB backtest walks 5 m bars; the EXH / DIST backtests and all real fills walk 1 m bars. The B line moves only on the bar of a settlement, and intrabar ordering within a bar is unknown (stop first).
4. **Stop-counterfactual caveat (DL 110).** The live engine polls; the replica stops on the 1 m / 5 m HIGH. EXH / DIST CUTs are near-miss wicks, the class live polling sometimes rides through, so CUT may be overstated in the backtest. SAVED is also priced on replica stops. On real fills the rule holds: fills live did not stop show Δ = 0, and the live-stopped FB set splits into SAVED (KERNEL +1.60) and no deeper fill.
5. Funding uses settled public rates × the bar-open price as a mark proxy (the paper accrual uses rate × qty × mark). Missing pair-months: FB 4 fills and DIST 2 fills carry the cohort's imputed funding with B = A.
6. "Hourly" is measured at entry only. Coins that switch to hourly during the hold (KERNEL went hourly at 16:00 on 11-20, after the EXH fill) are not tabulated separately.
7. B was tested WITHOUT fees in the trigger, matching the engine's price-only stop. A trigger that matches the Open Orders display (fees ≈ 0.1 % included) would sit slightly tighter and add CUTs; it was not tested.
8. Stop / time slip is applied identically in both arms. A B exit right after a settlement fills at the bar open (gap), the same as A on a gap.
9. The EXH backtest is the FRENZY universe only (rows_FRENZY); the WIDE extension was not run. DIST uses EXH's exit.
10. One size of 0.61 × equity is used for all three sleeves, as in TP_GRID 11-20. The DCR is a one-size read, not an account forecast. Bootstrap clusters = entry day. There is no family-wise correction for the three arms; B was the pre-specified one.

## 8. Verdict for the operator (plain English)

- **FLATBASE:** making the stop count funding passes the test we set before looking. In practice it rarely matters. It only kicks in on the few trades that pay heavy funding and are already losing, and it trims those losses by about the funding they paid. On KERNEL it would have saved about $108. But it would have exited only one minute earlier, not at midnight: the price shot through both lines at once. Over two years of backtest it adds very little profit (about +0.01 % per day) and slightly lowers the worst drawdown. If you want it, build the version that can only tighten the stop, never loosen it, with a revert alarm after its first 5 uses.
- **EXHAUSTION and DISTRIBUTION:** don't change them. Their stops are already tight (≈ 6 %), and many of their best trades come within a hair of the stop before winning. Counting funding would knock out some of those winners, and that costs more than it saves: about −0.03 to −0.05 % per trade in the backtest, in both periods.
- **Skipping hourly-funding entries (option C):** not supported. Those trades are profitable on average in every sleeve.

---
## Full generated tables

## FLATBASE_SHORT

**FLATBASE_SHORT — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**

| cohort | N | hourly_at_entry_pct | paying_pct | paid_median | paid_p90 | paid_worst | received_best | TP N | TP paid avg | STOP N | STOP paid avg | TIME N | TIME paid avg | corr(entry rate, received) | spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 28 | +10.71 | +42.86 | +0.00 | +0.49 | +1.82 | +0.03 | 25 | +0.16 | 3 | +0.53 | 0 |  | +0.15 | +0.68 |
| real (stack_keep, before DL 601) | 32 | +9.38 | +37.50 | +0.00 | +0.38 | +1.82 | +0.03 | 27 | +0.15 | 5 | +0.31 | 0 |  | +0.18 | +0.73 |
| backtest | 545 | +8.99 | +40.73 | +0.00 | +0.58 | +5.53 | +0.12 | 402 | +0.21 | 118 | +0.18 | 25 | +0.12 | +0.26 | +0.63 |

**FLATBASE_SHORT — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**

| block | arm | N | WR | avg | sum | DCR | maxDD | d_avg | d_lo | d_hi | P_gain | loo_min | loo_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | A (today) | 28 | +89.29 | +3.09 | +86.45 | +1.69 | -10.38 |  |  |  |  |  |  |
| REAL | B net | 28 | +89.29 | +3.14 | +88.05 | +1.73 | -9.44 | +0.06 | -0.00 | +0.19 | +0.65 | -0.00 | +0.06 |
| REAL | Bp paid-only | 28 | +89.29 | +3.14 | +88.05 | +1.73 | -9.44 | +0.06 | +0.00 | +0.19 | +0.65 | +0.00 | +0.06 |
| REAL w/o KERNEL #25 | A (today) | 27 | +92.59 | +3.65 | +98.43 | +1.95 | -6.33 |  |  |  |  |  |  |
| REAL w/o KERNEL #25 | B net | 27 | +92.59 | +3.65 | +98.42 | +1.95 | -6.33 | -0.00 | -0.00 | +0.00 | +0.00 | -0.00 | -0.00 |
| REAL w/o KERNEL #25 | Bp paid-only | 27 | +92.59 | +3.65 | +98.43 | +1.95 | -6.33 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| REAL stack_keep (pre-DL 601) | A (today) | 32 | +84.38 | +2.36 | +75.60 | +1.45 | -16.05 |  |  |  |  |  |  |
| REAL stack_keep (pre-DL 601) | B net | 32 | +84.38 | +2.41 | +77.18 | +1.48 | -15.18 | +0.05 | -0.00 | +0.17 | +0.65 | -0.00 | +0.05 |
| REAL stack_keep (pre-DL 601) | Bp paid-only | 32 | +84.38 | +2.41 | +77.20 | +1.48 | -15.16 | +0.05 | +0.00 | +0.17 | +0.65 | +0.00 | +0.05 |
| BT DESIGN | A (today) | 94 | +76.60 | +1.30 | +122.15 | +0.18 | -22.00 |  |  |  |  |  |  |
| BT DESIGN | B net | 94 | +76.60 | +1.35 | +126.71 | +0.19 | -21.05 | +0.05 | +0.01 | +0.10 | +1.00 | +0.03 | +0.05 |
| BT DESIGN | Bp paid-only | 94 | +76.60 | +1.35 | +126.82 | +0.19 | -21.03 | +0.05 | +0.01 | +0.10 | +1.00 | +0.03 | +0.05 |
| BT HOLDOUT | A (today) | 451 | +75.61 | +1.10 | +497.61 | +0.52 | -41.26 |  |  |  |  |  |  |
| BT HOLDOUT | B net | 451 | +75.39 | +1.11 | +502.19 | +0.53 | -37.50 | +0.01 | -0.06 | +0.05 | +0.69 | +0.00 | +0.04 |
| BT HOLDOUT | Bp paid-only | 451 | +75.39 | +1.11 | +502.73 | +0.53 | -37.47 | +0.01 | -0.06 | +0.06 | +0.70 | +0.00 | +0.04 |
| BT ALL | A (today) | 545 | +75.78 | +1.14 | +619.77 | +0.38 | -41.26 |  |  |  |  |  |  |
| BT ALL | B net | 545 | +75.60 | +1.15 | +628.91 | +0.39 | -37.50 | +0.02 | -0.04 | +0.06 | +0.76 | +0.01 | +0.04 |
| BT ALL | Bp paid-only | 545 | +75.60 | +1.16 | +629.55 | +0.39 | -37.47 | +0.02 | -0.04 | +0.06 | +0.77 | +0.01 | +0.04 |

**FLATBASE_SHORT — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**

| block | arm | unchanged_N | unchanged_max_abs_d | exactness_violations | SAVED N | SAVED Σ% | SAVED $ | CUT N | CUT Σ% | CUT $ | WIDENED N | WIDENED Σ% | WIDENED $ | OTHER N | OTHER Σ% | OTHER $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | B | 25 | +0.00 | 0 | 3 | +1.60 | +97.35 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL | Bp | 27 | +0.00 | 0 | 1 | +1.60 | +97.58 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL w/o KERNEL #25 | B | 25 | +0.00 | 0 | 2 | -0.00 | -0.23 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL w/o KERNEL #25 | Bp | 27 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL stack_keep (pre-DL 601) | B | 28 | +0.00 | 0 | 4 | +1.58 | +96.13 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL stack_keep (pre-DL 601) | Bp | 31 | +0.00 | 0 | 1 | +1.60 | +97.58 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | B | 78 | +0.00 | 0 | 16 | +4.56 | +278.38 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | Bp | 85 | +0.00 | 0 | 9 | +4.67 | +284.92 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | B | 391 | +0.00 | 0 | 58 | +17.37 | +1059.39 | 1 | -12.71 | -775.29 | 1 | -0.08 | -4.76 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | Bp | 412 | +0.00 | 0 | 38 | +17.83 | +1087.53 | 1 | -12.71 | -775.29 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT ALL | B | 469 | +0.00 | 0 | 74 | +21.93 | +1337.78 | 1 | -12.71 | -775.29 | 1 | -0.08 | -4.76 | 0 | +0.00 | +0.00 |
| BT ALL | Bp | 497 | +0.00 | 0 | 47 | +22.50 | +1372.45 | 1 | -12.71 | -775.29 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |

**FLATBASE_SHORT real — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| KERNELUSDT 2026-11-20 | -11.975 | STOP | -10.375 | STOP | +1.600 | +1.595 | True | -0.192 |
| MMTUSDT 2026-11-19 | -10.378 | STOP | -10.380 | STOP | -0.003 | -0.003 | False | +0.003 |
| BANKUSDT 2026-10-27 | -10.379 | STOP | -10.380 | STOP | -0.001 | -0.001 | False | +0.005 |

**FLATBASE_SHORT backtest — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| ORCAUSDT 2025-11-27 | +2.240 | TP | -10.470 | STOP | -12.710 | +2.698 | True | -0.089 |
| LRCUSDT 2025-12-12 | -13.891 | STOP | -10.878 | STOP | +3.013 | +2.990 | True | -0.064 |
| STORJUSDT 2025-12-28 | -12.551 | STOP | -10.622 | STOP | +1.929 | +1.919 | True | -0.099 |
| ENJUSDT 2026-03-18 | -11.792 | STOP | -10.230 | STOP | +1.562 | +1.560 | True | -0.056 |
| LPTUSDT 2025-05-31 | -11.909 | STOP | -10.418 | STOP | +1.491 | +1.487 | True | -0.146 |
| MEMEUSDT 2026-01-20 | -12.074 | STOP | -10.872 | STOP | +1.202 | +1.193 | False | -0.224 |
| HYPERUSDT 2025-12-10 | -11.483 | STOP | -10.387 | STOP | +1.096 | +1.093 | False | -0.122 |
| AGLDUSDT 2024-11-04 | -11.968 | STOP | -10.895 | STOP | +1.073 | +1.065 | False | -0.341 |
| STEEMUSDT 2026-02-24 | -11.183 | STOP | -10.116 | STOP | +1.067 | +1.066 | True | -0.180 |
| LAYERUSDT 2026-06-22 | -11.161 | STOP | -10.215 | STOP | +0.946 | +0.944 | False | -0.258 |
| MAGICUSDT 2025-12-09 | -11.003 | STOP | -10.240 | STOP | +0.763 | +0.762 | False | -0.351 |
| XAIUSDT 2026-03-11 | -10.916 | STOP | -10.242 | STOP | +0.674 | +0.673 | False | -0.120 |

**FLATBASE_SHORT real — per-pair Δ (arm B − A, Σ % pts)**: 3 pairs move · Σ+ +1.60 · Σ− -0.00 · top gainer share of Σ+ 100 % · top loser share of Σ− 67 %

| pair | d_sum |
|---|---|
| MMTUSDT | -0.00 |
| BANKUSDT | -0.00 |
| KERNELUSDT | +1.60 |

**FLATBASE_SHORT backtest — per-pair Δ (arm B − A, Σ % pts)**: 66 pairs move · Σ+ +22.46 · Σ− -13.32 · top gainer share of Σ+ 15 % · top loser share of Σ− 95 %

| pair | d_sum |
|---|---|
| ORCAUSDT | -12.71 |
| MUBARAKUSDT | -0.10 |
| DEGENUSDT | -0.09 |
| B3USDT | -0.09 |
| FLOCKUSDT | -0.06 |
| LPTUSDT | +1.49 |
| MEMEUSDT | +1.71 |
| ENJUSDT | +1.73 |
| STORJUSDT | +1.93 |
| LRCUSDT | +3.36 |

**FLATBASE_SHORT real — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 3 | +66.67 | -0.70 | -2.11 | +0.53 | 1 |
| rest | 25 | +92.00 | +3.54 | +88.57 | +0.16 | 2 |

**FLATBASE_SHORT backtest — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 47 | +76.60 | +0.65 | +30.46 | +0.90 | 10 |
| rest | 498 | +75.70 | +1.18 | +589.30 | +0.14 | 108 |

## EXHAUSTION_SHORT

**EXHAUSTION_SHORT — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**

| cohort | N | hourly_at_entry_pct | paying_pct | paid_median | paid_p90 | paid_worst | received_best | TP N | TP paid avg | STOP N | STOP paid avg | TIME N | TIME paid avg | corr(entry rate, received) | spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | +20.00 | +20.00 | +0.00 | +1.87 | +3.11 | +0.00 | 4 | +0.78 | 1 | +0.00 | 0 |  | -0.19 | +0.00 |
| backtest | 3368 | +2.49 | +33.28 | +0.00 | +0.69 | +4.83 | +0.28 | 1127 | +0.30 | 1933 | +0.12 | 308 | +0.28 | +0.32 | +0.49 |

**EXHAUSTION_SHORT — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**

| block | arm | N | WR | avg | sum | DCR | maxDD | d_avg | d_lo | d_hi | P_gain | loo_min | loo_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | A (today) | 5 | +80.00 | +5.08 | +25.41 | +3.84 | +0.00 |  |  |  |  |  |  |
| REAL | B net | 5 | +80.00 | +5.08 | +25.41 | +3.84 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| REAL | Bp paid-only | 5 | +80.00 | +5.08 | +25.41 | +3.84 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| BT DESIGN | A (today) | 1349 | +39.81 | +0.57 | +768.17 | +0.48 | -52.98 |  |  |  |  |  |  |
| BT DESIGN | B net | 1349 | +39.36 | +0.57 | +773.82 | +0.48 | -51.07 | +0.00 | -0.04 | +0.04 | +0.59 | -0.00 | +0.01 |
| BT DESIGN | Bp paid-only | 1349 | +39.21 | +0.56 | +761.47 | +0.47 | -52.45 | -0.00 | -0.05 | +0.03 | +0.42 | -0.01 | +0.00 |
| BT HOLDOUT | A (today) | 2019 | +43.14 | +0.74 | +1501.80 | +1.37 | -67.64 |  |  |  |  |  |  |
| BT HOLDOUT | B net | 2019 | +42.05 | +0.69 | +1399.81 | +1.26 | -67.32 | -0.05 | -0.12 | +0.01 | +0.05 | -0.05 | -0.04 |
| BT HOLDOUT | Bp paid-only | 2019 | +42.05 | +0.69 | +1400.94 | +1.26 | -67.30 | -0.05 | -0.11 | +0.01 | +0.05 | -0.05 | -0.04 |
| BT ALL | A (today) | 3368 | +41.81 | +0.67 | +2269.97 | +0.84 | -67.64 |  |  |  |  |  |  |
| BT ALL | B net | 3368 | +40.97 | +0.65 | +2173.63 | +0.80 | -67.32 | -0.03 | -0.07 | +0.01 | +0.08 | -0.03 | -0.02 |
| BT ALL | Bp paid-only | 3368 | +40.91 | +0.64 | +2162.41 | +0.79 | -67.30 | -0.03 | -0.07 | +0.00 | +0.05 | -0.03 | -0.03 |

**EXHAUSTION_SHORT — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**

| block | arm | unchanged_N | unchanged_max_abs_d | exactness_violations | SAVED N | SAVED Σ% | SAVED $ | CUT N | CUT Σ% | CUT $ | WIDENED N | WIDENED Σ% | WIDENED $ | OTHER N | OTHER Σ% | OTHER $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | B | 5 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL | Bp | 5 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | B | 1017 | +0.00 | 0 | 311 | +59.87 | +3651.96 | 9 | -70.60 | -4306.62 | 12 | +16.39 | +999.49 | 0 | +0.00 | +0.00 |
| BT DESIGN | Bp | 1181 | +0.00 | 0 | 159 | +63.90 | +3898.15 | 9 | -70.60 | -4306.62 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | B | 1602 | +0.00 | 0 | 393 | +174.85 | 10,666 | 22 | -276.81 | -16,885 | 2 | -0.03 | -2.10 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | Bp | 1704 | +0.00 | 0 | 293 | +175.94 | 10,733 | 22 | -276.81 | -16,885 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT ALL | B | 2619 | +0.00 | 0 | 704 | +234.72 | 14,318 | 31 | -347.41 | -21,192 | 14 | +16.35 | +997.39 | 0 | +0.00 | +0.00 |
| BT ALL | Bp | 2885 | +0.00 | 0 | 452 | +239.85 | 14,631 | 31 | -347.41 | -21,192 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |

EXHAUSTION_SHORT real: no fill changes under arm B.

**EXHAUSTION_SHORT backtest — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| SHELLUSDT 2026-11-15 | +9.733 | TP | -6.309 | STOP | -16.042 | +0.099 | False | +0.005 |
| SENTUSDT 2026-06-09 | +8.827 | TP | -6.308 | STOP | -15.134 | +1.005 | False | -0.018 |
| COSUSDT 2026-05-13 | +8.676 | TP | -6.307 | STOP | -14.983 | +1.156 | False | -0.734 |
| ARPAUSDT 2026-01-19 | +8.653 | TP | -6.307 | STOP | -14.960 | +1.179 | False | -0.044 |
| NILUSDT 2026-07-27 | +9.337 | TP | -5.406 | STOP | -14.743 | +0.495 | False | -0.040 |
| ONEUSDT 2026-09-28 | +8.894 | TP | -5.785 | STOP | -14.679 | +0.938 | True | -0.203 |
| MINAUSDT 2025-11-04 | +9.027 | TP | -5.097 | STOP | -14.124 | +0.805 | False | -0.130 |
| SOPHUSDT 2026-10-30 | +7.713 | TP | -6.306 | STOP | -14.019 | +2.119 | False | -0.195 |
| KATUSDT 2026-03-26 | +7.428 | TP | -6.426 | STOP | -13.854 | +2.404 | False | +0.003 |
| RONINUSDT 2026-05-19 | +7.312 | TP | -6.308 | STOP | -13.619 | +2.520 | False | -1.396 |
| LAYERUSDT 2026-05-10 | +7.007 | TP | -6.305 | STOP | -13.313 | +2.825 | False | -0.013 |
| ZILUSDT 2026-07-28 | +8.098 | TP | -5.048 | STOP | -13.146 | +1.734 | False | -0.305 |

EXHAUSTION_SHORT real: Δ is zero for every pair.

**EXHAUSTION_SHORT backtest — per-pair Δ (arm B − A, Σ % pts)**: 305 pairs move · Σ+ +199.16 · Σ− -295.49 · top gainer share of Σ+ 4 % · top loser share of Σ− 8 %

| pair | d_sum |
|---|---|
| SXPUSDT | -24.40 |
| SPELLUSDT | -17.74 |
| SHELLUSDT | -16.04 |
| ONEUSDT | -14.69 |
| NILUSDT | -14.59 |
| HYPERUSDT | +4.65 |
| LSKUSDT | +4.65 |
| REDUSDT | +5.13 |
| GMTUSDT | +6.56 |
| TUSDT | +8.53 |

**EXHAUSTION_SHORT real — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 1 | +100.00 | +6.82 | +6.82 | +3.11 | 0 |
| rest | 4 | +75.00 | +4.65 | +18.59 | +0.00 | 1 |

**EXHAUSTION_SHORT backtest — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 83 | +51.81 | +1.41 | +117.43 | +0.99 | 40 |
| rest | 3285 | +41.55 | +0.66 | +2152.54 | +0.17 | 1893 |

## DISTRIBUTION_SHORT

**DISTRIBUTION_SHORT — funding paid (% of notional; + = paid by the short) under today's exit (arm A)**

| cohort | N | hourly_at_entry_pct | paying_pct | paid_median | paid_p90 | paid_worst | received_best | TP N | TP paid avg | STOP N | STOP paid avg | TIME N | TIME paid avg | corr(entry rate, received) | spearman |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| real | 5 | +0.00 | +0.00 | +0.00 | -0.00 | -0.00 | +0.07 | 1 | +0.00 | 4 | -0.02 | 0 |  | +0.45 | +0.71 |
| backtest | 1547 | +1.36 | +19.13 | +0.00 | +0.23 | +4.73 | +0.29 | 537 | +0.14 | 881 | +0.04 | 129 | +0.16 | +0.19 | +0.37 |

**DISTRIBUTION_SHORT — arm A vs B (net stop) vs Bp (paid-only) · one size 0.61 × equity · Δ = arm − A per fill, day-block bootstrap 95 % CI**

| block | arm | N | WR | avg | sum | DCR | maxDD | d_avg | d_lo | d_hi | P_gain | loo_min | loo_max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | A (today) | 5 | +20.00 | -2.32 | -11.58 | -1.87 | -9.33 |  |  |  |  |  |  |
| REAL | B net | 5 | +20.00 | -2.33 | -11.65 | -1.88 | -9.38 | -0.01 | -0.02 | +0.00 | +0.00 | -0.02 | +0.00 |
| REAL | Bp paid-only | 5 | +20.00 | -2.32 | -11.58 | -1.87 | -9.33 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| BT DESIGN | A (today) | 559 | +41.68 | +0.98 | +546.67 | +0.39 | -34.62 |  |  |  |  |  |  |
| BT DESIGN | B net | 559 | +41.32 | +0.96 | +536.84 | +0.38 | -34.76 | -0.02 | -0.09 | +0.03 | +0.33 | -0.02 | +0.01 |
| BT DESIGN | Bp paid-only | 559 | +41.32 | +0.96 | +538.34 | +0.38 | -34.46 | -0.01 | -0.08 | +0.03 | +0.36 | -0.02 | +0.01 |
| BT HOLDOUT | A (today) | 988 | +42.71 | +0.87 | +861.00 | +0.85 | -55.07 |  |  |  |  |  |  |
| BT HOLDOUT | B net | 988 | +42.21 | +0.84 | +832.41 | +0.81 | -55.09 | -0.03 | -0.09 | +0.02 | +0.14 | -0.03 | -0.01 |
| BT HOLDOUT | Bp paid-only | 988 | +42.21 | +0.84 | +834.07 | +0.81 | -54.98 | -0.03 | -0.08 | +0.02 | +0.15 | -0.03 | -0.01 |
| BT ALL | A (today) | 1547 | +42.34 | +0.91 | +1407.66 | +0.58 | -55.07 |  |  |  |  |  |  |
| BT ALL | B net | 1547 | +41.89 | +0.89 | +1369.25 | +0.56 | -55.09 | -0.02 | -0.07 | +0.01 | +0.11 | -0.03 | -0.01 |
| BT ALL | Bp paid-only | 1547 | +41.89 | +0.89 | +1372.41 | +0.56 | -54.98 | -0.02 | -0.07 | +0.01 | +0.13 | -0.02 | -0.01 |

**DISTRIBUTION_SHORT — what changed (SAVED = A stopped anyway, B out earlier · CUT = A reached TP / time, B stopped · WIDENED = received funding moved B's line out past A's stop bar) · $ = Σ Δ % × 61 (a $10,000 account at one size)**

| block | arm | unchanged_N | unchanged_max_abs_d | exactness_violations | SAVED N | SAVED Σ% | SAVED $ | CUT N | CUT Σ% | CUT $ | WIDENED N | WIDENED Σ% | WIDENED $ | OTHER N | OTHER Σ% | OTHER $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REAL | B | 4 | +0.00 | 0 | 1 | -0.07 | -4.29 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| REAL | Bp | 5 | +0.00 | 0 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT DESIGN | B | 471 | +0.00 | 0 | 81 | +10.84 | +661.22 | 2 | -20.26 | -1235.80 | 5 | -0.40 | -24.57 | 0 | +0.00 | +0.00 |
| BT DESIGN | Bp | 525 | +0.00 | 0 | 32 | +11.93 | +727.78 | 2 | -20.26 | -1235.80 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | B | 862 | +0.00 | 0 | 119 | +28.30 | +1726.40 | 5 | -56.80 | -3465.09 | 2 | -0.08 | -5.01 | 0 | +0.00 | +0.00 |
| BT HOLDOUT | Bp | 918 | +0.00 | 0 | 65 | +29.88 | +1822.62 | 5 | -56.80 | -3465.09 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |
| BT ALL | B | 1333 | +0.00 | 0 | 200 | +39.14 | +2387.62 | 7 | -77.06 | -4700.88 | 7 | -0.48 | -29.58 | 0 | +0.00 | +0.00 |
| BT ALL | Bp | 1443 | +0.00 | 0 | 97 | +41.81 | +2550.40 | 7 | -77.06 | -4700.88 | 0 | +0.00 | +0.00 | 0 | +0.00 | +0.00 |

**DISTRIBUTION_SHORT real — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| EVAAUSDT 2026-11-19 | -4.786 | STOP | -4.857 | STOP | -0.070 | -0.070 | False | +0.053 |

**DISTRIBUTION_SHORT backtest — the changed fills (largest |Δ| first, up to 12)**

| fill | A | A_why | B | B_why | d | paid_A | hourly | entry_rate_pct |
|---|---|---|---|---|---|---|---|---|
| LEVERUSDT 2025-06-30 | +9.358 | TP | -6.308 | STOP | -15.666 | +0.474 | False | -0.008 |
| STEEMUSDT 2026-09-13 | +8.590 | TP | -6.307 | STOP | -14.897 | +1.242 | False | +0.002 |
| MIRAUSDT 2026-02-26 | +7.417 | TP | -5.607 | STOP | -13.024 | +2.415 | False | -0.004 |
| CVXUSDT 2026-01-03 | +5.099 | TP | -5.986 | STOP | -11.086 | +4.733 | False | +0.005 |
| METUSDT 2026-04-22 | +4.424 | TIME | -6.309 | STOP | -10.733 | +0.427 | False | -0.000 |
| ZORAUSDT 2026-09-04 | +1.629 | TIME | -5.436 | STOP | -7.065 | +0.497 | False | -0.100 |
| ONTUSDT 2025-01-13 | +1.590 | TIME | -3.003 | STOP | -4.593 | +4.615 | False | +0.010 |
| RVNUSDT 2025-06-05 | -9.087 | STOP | -6.305 | STOP | +2.782 | +2.778 | False | -1.066 |
| ZILUSDT 2026-02-02 | -7.866 | STOP | -5.301 | STOP | +2.566 | +2.563 | False | -0.020 |
| ONEUSDT 2026-10-03 | -7.127 | STOP | -4.886 | STOP | +2.241 | +2.238 | False | -0.017 |
| BMTUSDT 2026-11-03 | -7.047 | STOP | -4.841 | STOP | +2.206 | +2.203 | False | -0.933 |
| STORJUSDT 2026-05-15 | -7.745 | STOP | -5.944 | STOP | +1.801 | +1.799 | False | -0.008 |

**DISTRIBUTION_SHORT real — per-pair Δ (arm B − A, Σ % pts)**: 1 pairs move · Σ+ +0.00 · Σ− -0.07 · top gainer share of Σ+ nan % · top loser share of Σ− 100 %

| pair | d_sum |
|---|---|
| EVAAUSDT | -0.07 |

**DISTRIBUTION_SHORT backtest — per-pair Δ (arm B − A, Σ % pts)**: 171 pairs move · Σ+ +41.32 · Σ− -79.73 · top gainer share of Σ+ 7 % · top loser share of Σ− 20 %

| pair | d_sum |
|---|---|
| LEVERUSDT | -15.67 |
| STEEMUSDT | -14.90 |
| MIRAUSDT | -13.02 |
| CVXUSDT | -11.09 |
| METUSDT | -10.73 |
| BLURUSDT | +1.80 |
| BMTUSDT | +2.21 |
| ONEUSDT | +2.24 |
| ZILUSDT | +2.53 |
| RVNUSDT | +2.78 |

**DISTRIBUTION_SHORT real — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 0 |  |  | +0.00 |  | 0 |
| rest | 5 | +20.00 | -2.32 | -11.58 | -0.01 | 4 |

**DISTRIBUTION_SHORT backtest — arm C context (entry skip, NOT tested as a filter): fills on hourly funding with the entry rate against the short**

| cohort | N | WR | avg_A | sum_A | paid_avg | stops |
|---|---|---|---|---|---|---|
| C-skip: hourly ∧ entry rate < 0 | 19 | +63.16 | +4.24 | +80.63 | +0.32 | 7 |
| rest | 1528 | +42.08 | +0.87 | +1327.03 | +0.08 | 874 |
