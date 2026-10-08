# SPIKE_FADE H1 deep review + bug hunt across the halves table (2026-10-08)

**Question (operator):** "Spike fade is strange in the 1st half, something does not add up — make a deep review." Then: treat it as a bug hunt and check every strategy in `scripts/study_yr5_halves_today.py` and the Oct-6 table it reproduces.

**Validation first:** `scripts/validate_against_master.py` → **ALL CHECKS PASS**.
**Inputs, read-only:** the yr5 replay frame (`study_yr5_halves_today.build_replay`, code 181131e, 3 seeds, warm-up trimmed), `reports/MASTER_POOL_stacked.csv` (STACK 2026-10-08b), `trading_config.json` vs `reports/backtest_cache/replay/frozen_config_yr5_181131e.json`, and the engine source.
**New files:** `scripts/study_fade_h1_fixed_halves.py` (corrected copy; it imports the original and does not edit it) → `reports/study_fade_h1_fixed_halves.csv`. No Binance fetch. No code or config change.

## Plain-English answer

- **The fade dollars are computed correctly.** The ticket, the volume units, the P&L % and the $-per-% all check out. Nothing in the fade arithmetic is wrong, and the H1 number is not a calculation error.
- **What does not add up is the comparison with live.** The live "90 % WR, +0.44 %" figure and the replay measure different things:
  - **The master re-prices 7 of the 73 kept fades.** Five were stopped at the old −0.70 stop and are re-priced at today's −1.5 stop; three of those become winners. Two more get the optimistic late-arm counterfactual. As actually traded, the live fades are **83.6 % WR, +0.34 %/trade**.
  - **The live fades all come from Jul-28 → Oct-8.** The bot was offline Aug-27 → Sep-11, and on those days the replay's fades lost −0.46 %/trade.
- **The replay engine prices a given fade signal almost exactly like live, but it picks a different set of signals.**
  - **Same signals:** on the 32 fades both found, replay +0.13 vs live +0.18, with the same exit reason 96 % of the time.
  - **Live-only fades:** the 37 kept live fades the replay never found won +0.46.
  - **Replay-only fades:** the replay fades live never took lost −0.24.
  - **Net gap:** over the window where both ran, the replay is about **0.4 %/trade more pessimistic** than live paper. That gap is 4× larger than H1's whole deficit (−0.09).
- **Inside the replay, the H1 loss is solid.** All 3 seeds lose, every month Feb–May loses, and no single pair or day explains it. The cause is more stop-outs: 25 % of fills stopped vs 15 % live.
- **H2's profit is the fragile half.** It comes from one stretch, the Jun-18 → Jul-2 washed-out window (+$3,141). Without it H2 is −$425 and the year is −$3,724.
- **Did spike fade really lose money in H1?** On this harness yes, about −0.09 %/trade (−$3.3k on a $3k book at 2× / 20×). But the harness under-reads fades by about 0.4 %/trade where it can be checked against live. So **an H1 loss is not proven, and neither is an H1 profit.** The replay cannot settle the sign of fades at this precision.
- **Bug hunt across all strategies:** two real issues were found. Neither changes any sign.
  - **FIX-A:** a sizing leak. It moves the halves TOTAL (without WILLY) from −$7,320 / +$12,111 to −$7,456 / +$12,399.
  - **FIX-B:** a missing WILLY global hold. It moves the TOTAL with WILLY from −$3,590 / +$18,255 to −$1,464 / +$15,594.

## Findings ranked by $ impact (per seed, flat $3k book)

| # | Finding | Type | $ effect |
|---|---|---|---|
| 1 | The replay samples different fade signals than live: 32/111 live fades reproduced. Replay-only fades lose, live-only fades win. | fidelity limit (open) | ≈ 0.4 %/trade × 156 H1 fades × $22k ≈ **$13k** if live's edge held in H1 (unverifiable) |
| 2 | The live "90 % / +0.44" is stack-repriced (CF_FADE_SL15 ×5, CF_FADE_LATE_ARM ×2) and comes from a live-only window. | comparison error, not code | raw live 83.6 % / +0.34 |
| 3 | WILLY global hold not modelled (TOTAL-with-WILLY row) | **bug (parity) — FIX-B** | TOTAL with WILLY, FIX-B alone ≈ H1 +$2,262 · H2 −$2,949 · FY −$687 (FIX-A + FIX-B together: +$2,125 / −$2,661 / −$535); fade alone FY +$370 |
| 4 | Fade stop fills at the first print past −1.5 (avg −1.57; H1 13.8 % of stops below −1.6) | modelling choice (stated) | H1 −$608 · H2 −$700 vs paper-at-line (SENS-S1) |
| 5 | Replay-book liquidity-cap ratio `fr` carried onto the $3k ruler | **bug (sizing) — FIX-A** | MOM_LONG −$121 / +$212 · FLIP −$16 / +$35 · FRENZY_LONG +$35 (H2) · MOM_SHORT +$7 |
| 6 | Oct-6 report says 14 % of MOM_LONG/FLIP fills shrank for "not enough free margin". All 315 shrunk fills sit exactly at the 0.1 % liquidity cap (notional = cap on 100 %). | mislabelled cause | same as #5 |

## 1 · $ vs % consistency (item 1): no error

- **Ticket rule.** The script prices each fade at ticket = min(desired, 0.5 % × 24 h volume, $500k), with desired = 3000 × 0.24375 × 2 × 20 = **$29,250**. This is the live engine's rule:
  - code: `trading_engine.py` 10829–10860 (LIQ2 pct for spike species);
  - config: `spike_lowvol_liq_cap_pct` 0.5, threshold $1T since Oct-1, `max_notional_hard_ceiling` 500k.
- **Volume units are USD.** Replay fade `entry_pair_volume_24h_usd` runs from $2.01M to $19.99M. That is exactly the FLOOR_2M and FADE_MAXVOL band. No volume is missing (0 NaN).
- **Ticket ratio range.** It runs 0.34–1.00 (mean 0.76). No scale above 1 and no default fallback.
- **P&L % checks.** P&L % = (entry − exit) / entry − fees, exact on all 1,183 rows (max |Δ| 4e-16). The fee is 0.09 % round trip.
- **Implied notional matches.** $3,299 / (0.091 % × 156) = **$23.2k** per trade, against an actual average ticket of $22.2k. Winners and losers carry the same ticket ($22.2k vs $22.3k), and the $-weighted avg (−0.095) ≈ the unweighted avg (−0.091). No sizing outlier drives the $.
- **Oct-6 → Oct-8 change.** The fade moved from −$1,593 to −$583. The stated reason, "bigger tickets", is correct: Oct-6 carried the replay's own fill ratio (median 0.45, measured on the replay's bigger compounding book) onto the $3k ruler; Oct-8 recomputes the ticket on $3k (median 0.79).

## 2 · Month table, SPIKE_FADE (per seed)

| Month | N | WR | avg % | Net $ | avg ticket |
|---|---|---|---|---|---|
| Jan | 28.7 | 80 % | +0.09 | +313 | $21.1k |
| Feb | 30.7 | 68 % | −0.14 | −1,192 | $22.1k |
| Mar | 38.7 | 70 % | −0.10 | −797 | $23.0k |
| Apr | 38.3 | 70 % | −0.16 | −718 | $22.1k |
| May (to 20th) | 19.7 | 75 % | −0.13 | −904 | $22.8k |
| May (20th on) | 18.0 | 69 % | −0.02 | −483 | $21.2k |
| Jun | 112.3 | 84 % | +0.10 | **+3,523** | $23.6k |
| Jul | 50.0 | 84 % | +0.16 | +1,325 | $20.5k |
| Aug | 26.3 | 76 % | −0.02 | −11 | $19.4k |
| Sep | 28.3 | 65 % | −0.21 | −1,198 | $21.9k |
| Oct (1–3) | 3.3 | 50 % | −0.66 | −441 | $15.5k |

**H1 concentration:**
- The loss is spread over 139 pairs and 96 days. The worst pair is CC (−$909); the worst 5 pairs together lose −$2,667 of the −$3,299.
- The worst 5 days lose −$2,311, which leaves −$988 for the other days.
- The 15 worst single fills (across seeds) carry 19 % of the gross loss.
- **This is not a few-pair or few-day artefact.**

**H2 concentration:**
- H2 is carried by the washed-out window Jun-18 → Jul-2: 62.7 fades per seed, +0.18 %, **+$3,141**.
- Without that window: H2 is **−$425** and the year is **−$3,724** (−0.045 %/trade).

## 3 · Winner/loser shape (item 3)

| | WR | avg win | avg loss | stop exits | breakeven WR |
|---|---|---|---|---|---|
| Replay H1 | 72.0 % | +0.42 | −1.41 | 24.8 % (avg −1.570) | 77 % |
| Replay H2 | 79.2 % | +0.45 | −1.49 | 19.5 % (avg −1.579) | 77 % |
| Replay Jul-28 → Oct-4 | 69.1 % | +0.51 | −1.60 | 29.2 % | — |
| Live master kept, raw | 83.6 % | +0.61 | −1.06 ᵃ | 15 % | 63 % |

ᵃ Live losses mix the old −0.70 stop with today's −1.5 stop; with today's stop alone they are about −1.26.

- **Wins:** on the same signal, wins are the same size (both won: live +0.55 vs replay +0.51). So the H1 problem is not the trail exit.
- **Stop rate:** the H1 problem is the stop rate, 24.8 % vs live 15 %. 81 % of H1 stops never reached +0.3 first. These are dead-on-arrival squeezes, not round-trips.
- **Overshoot beyond the line:** −$608 (H1) and −$700 (H2) per seed, measured against booking at −1.50 the way live paper does.
- **SENS-S1, stops booked at the line:**

  | | avg %/trade | Net $ |
  |---|---|---|
  | H1 | −0.074 | −$2,690 |
  | H2 | +0.060 | +$3,416 |
  | Year | +0.007 | +$726 |

  Real money is **worse** than the replay, not better: the scout STOP_SLIP live proxy is about −0.32 % per stop.

## 4 · Entry-rule parity (item 4): the replay already applies today's fade rules

- **Config:** a flattened diff of `trading_config.json` against the frozen 181131e config shows **0 fade / spike / liquidity / fresh / ceiling keys changed** and no blacklist change.
- **Code:** since 181131e, the only change on the fade path is the **FRENZY_WILLY global hold** (DECISION_LOG 251).
- **Checks on the replay fades:**

  | Rule | Result on replay fades |
  |---|---|
  | FLOOR_2M / FADE_MAXVOL | volume $2.01M–$19.99M ✓ |
  | FADE_BRSI50 | max BTC RSI 49.97 ✓ |
  | FADE_LAGGARD | 0 fills match its condition (the gate is active since the BUGHUNT ③ fix) ✓ |
  | Blacklist (incl. 龙虾USDT) | 0 fades on blacklisted pairs ✓ |
  | FADE_FRESHBREAK / BD13 | not recomputable from the stamps; the engine applied them |
  | Funding | not charged in the replay (empty column); fades hold for minutes, so the effect is negligible |

- **WILLY hold:** it would refuse **9.2 %** of fades (36 per seed). H1 −$3,299 → −$3,333 · H2 +$2,716 → +$3,121 · year −$583 → −$213.

## 5 · Seed dispersion (item 5)

| | s1 | s2 | s3 |
|---|---|---|---|
| H1 Net $ | −4,753 | −2,969 | −2,174 |
| H1 avg % | −0.137 | −0.074 | −0.061 |
| H1 stop share | 27.8 % | 25.3 % | 20.9 % |
| H2 Net $ | +6,680 | +1,307 | +161 |

- **The H1 sign is consistent across seeds; the H2 size is not.**
- **The $ are not driven by compounding (the book is flat) or by sizing outliers.**
- **Per-signal noise is large.** Seeds differ only in the phase of the synthetic scan clock. Yet the same fade (pair ± 5 min) has a different sign in 14–17 % of matches and a different stop/no-stop outcome in 45–50 of about 310. A fade's outcome flips on the entry second.

## 6 · Live vs replay, Jul-28 → Oct-4 (item 6)

**Match:**
- 111 live fades (66 kept by today's stack). 32 are reproduced by at least one seed, 29 of them kept; per seed 24–27.
- On the 78 matched rows: live +0.177 vs replay +0.132, sign agreement 74 %, correlation 0.42. Stop rate 15.4 % in both. Entry price differs by a median 0.00 % (sd 0.45).

| Cohort | N | WR | avg % |
|---|---|---|---|
| Live kept, reproduced in 3/3 seeds | 19 | 79 % | +0.17 raw (+0.29 stack) |
| Live kept, **never reproduced** | 37 | 84 % | **+0.46 raw** (+0.61 stack) |
| Live fills blocked by today's stack | 45 | — | −0.34 raw (FRESHBREAK 26, LAGGARD 10, MAXVOL 6, BD13 2, FLOOR_2M 1) |
| Replay, live up, **not taken live** | 18.3 / seed | 60 % | **−0.24** |
| Replay, live up, matched | 25.3 / seed | 84 % | +0.12 |
| Replay, **live offline** (Aug-10/11, Aug-28 → Sep-10) | 15.7 / seed | 53 % | −0.46 |
| **Replay total while live was up** | 43.7 / seed | 74 % | **−0.03** vs live kept **+0.37 raw** |

**Reading:**
- The engine replica is faithful per signal.
- Recall is low and **not neutral on this sample**: what live caught and the replay missed won, and what the replay added lost.
- The Sep-29 audit (yr3) saw neutral averages for the extras (+0.04 vs matched +0.23); yr5 at today's gates does not.
- N is small and live is dominated by B2 (Aug). Treat the 0.4 gap as a calibration warning, not as a correction factor.
- No entry stamp separates replay-only fades cleanly: BTC 4h EMA50/200 gap median 3.2 vs 0.2, BTC RSI 46 vs 40, quality score 1 vs 2. These are small-N hints only, not filter candidates.

## Bug hunt — every strategy in the halves table and the Oct-6 table

Bug classes checked:
- **Sizing / ticket scale:** FIX-A.
- **$ conversion:** P&L % = pnl / notional exact; flat-book formula verified for fades.
- **Volume units:** one USD column everywhere.
- **Stop-fill booking:** fades measured here; momentum long ≈ 0 and flips −0.05/fill (Sep-29 audit).
- **Filter parity:** fade unchanged; the WILLY hold is FIX-B.
- **Seed / warm-up handling:** N and $ divided by the seed count; `yr5_fills_trimmed` asserts no duplicates.
- **Half-split:** by `opened_at`, so each fill counts once.

| Strategy | FIX-A (replay-book cap ratio) | FIX-B (WILLY hold) | Other classes |
|---|---|---|---|
| MOM_LONG | **yes**: 75.7 fills/seed capped at the replay book, never on $3k | yes: 9.4 % refused | clean |
| MOM_SHORT | **yes** (4.3 fills, +$7) | yes: 11.3 % | clean |
| FLIP_SHORT | **yes** (23 fills) | yes: 12.4 % | clean |
| SPIKE_FADE | no (Oct-8 already recomputes the ticket; Oct-6 **was** affected, now fixed) | yes: 9.2 % | clean; stop overshoot = SENS-S1 |
| BULLRUN | no (0 capped) | yes: 20.5 % | clean |
| BEARRUN | no | 0 % | clean |
| FRENZY_LONG | **yes**, 0.3 fills (+$35 H2) | yes: 10.1 % | clean |
| FRENZY_WIDE | no | yes: 11.3 % | clean |
| FRENZY_LITE | no (no `fr` in its pricing; the cap never binds at $3k) | yes: 12.8 % | study cohort (unchanged caveats) |
| FRENZY_WILLY | no | n/a (it is the hold) | study cohort |
| SURGE_LONG / SURGE_SHORT | no (0 capped) | yes: 5.0 % / 12.9 % | clean |
| Heat re-admits | no | yes: 8.8 % | — |

**Master builder / screen_pool:** not affected. Neither uses the replay fill ratio; `fade_cap05_scale` re-prices live fills from their own desired notional and volume. No edit made.

### Corrected table (mean of 3 seeds; flat $3k book per strategy)

**Changed rows, old → new Net $:**

| Strategy | Change | H1 | H2 | Full year |
|---|---|---|---|---|
| MOM_LONG (replay part) | FIX-A | −5,001 → **−5,121** | −1,659 → **−1,447** | −6,660 → **−6,568** |
| MOM_SHORT | FIX-A | −1,328 → −1,327 | −199 → −194 | −1,527 → −1,520 |
| FLIP_SHORT | FIX-A | −909 → **−925** | −471 → **−435** | −1,380 → **−1,360** |
| FRENZY_LONG | FIX-A | +506 → +506 | +1,363 → **+1,398** | +1,870 → **+1,904** |
| SPIKE_FADE | none | −3,299 | +2,716 | −583 |
| ↳ SENS-S1, stops at the line (not a fix) | — | −2,690 | +3,416 | +726 |
| ↳ under the WILLY hold | FIX-B view | −3,333 | +3,121 | −213 |
| TOTAL replay sleeves only | FIX-A | −9,507 → **−9,643** | +7,393 → **+7,681** | −2,114 → **−1,962** |
| **TOTAL live without WILLY** | FIX-A | −7,320 → **−7,456** | +12,111 → **+12,399** | +4,791 → **+4,944** |
| **TOTAL live with WILLY (paper)** | FIX-A + FIX-B | −3,590 → **−1,464** | +18,255 → **+15,594** | +14,665 → **+14,130** |

**Unchanged rows:** BULLRUN, BEARRUN, FRENZY_WIDE, SURGE_LONG, SURGE_SHORT, LITE, WILLY and the heat re-admits.

**Notes on the totals:**
- The "TOTAL replay sleeves only" row here excludes the heat re-admits. The original table's equivalent row includes them (−$10,409 / +$8,522).
- Avg % and WR do not change under FIX-A, because it is sizing only.

**Oct-6 flat Net $ with FIX-A** (year): SPIKE_FADE −1,593 → −500 (already superseded by Oct-8) · MOM_LONG −7,740 → −7,501 · FRENZY_LONG +2,770 → +2,881 · FLIP −2,759 → −2,720 · MOM_SHORT −1,527 → −1,520 · the rest unchanged.
- The Oct-6 **compounding** columns carry the same ratio onto books that shrink to a few hundred dollars, where the $-absolute cap binds even less. Those columns understate size on shrinking books. This is not re-run.

## Open fidelity limits

1. **Fade recall.** The replay reproduces 29 % of live fades, and its extras lose while live's unreproduced fills win. This is the biggest unknown, about 0.4 %/trade on the only checkable window.
   - The cause is not identified. Candidates: discrete scan instants, the forming-1m fallback on minutes without ticks, live slot / pair-held state.
   - The next step is a per-signal trace: for each live-only fade, log the replay's nearest scan of that pair and why it did not fire.
2. **Stop fills.** The replay fills at the first print past the line (−0.07 per stop vs paper). Live paper fills at the line; real money slips about −0.32.
3. **Entry-second sensitivity.** About 15 % of identical signals change outcome between seeds. Fade results need many seeds or many windows before any sign call.
4. **Live sample.** 66–73 kept fades, Jul-28 → Oct-8, mostly B2. There are no live fades in H1, so H1 can only be judged by the replay.
5. **Not modelled:** WILLY's own sequencing against the other sleeves (beyond the hold), and the LITE slot limits (both already stated in the halves report).
