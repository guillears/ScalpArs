# FRENZY on REAL fills: exit math, runners, the 70× entry, the LONG/LITE volume boundary, the timeline, the LITE window (2026-10-19)

**Operator:** "About 1 in 3 entries is a runner (+5 % before −3 %) in every period. We need to identify winners vs losers, or find the perfect TP / SL / trailing math. Focus on MASTER + current batch, not the backtest." Add-ons: Part C (70× early entry), Part D (LITE and the volume boundary), Part E (entry timeline), Part F (LITE window / green bar, "why LONG and not LITE").

**Scope:** research only. No code, config or git change. No commit. The backtest is a footnote only: it never chose a rule here, it only re-reads rules frozen on real fills.

## The answer in plain English

1. **The runners are real, but you can't see them coming, and the "perfect exit" found on 29 fills does not survive an honest test.**
   - 33 of the 88 real fills (37.5 %) ran +5 before −3.
   - In every sleeve, the best in-sample exit loses to today's exit once it is chosen leave-one-day-out (LODO):
     - LONG: −0.14 vs +0.02 %/fill
     - LITE: −1.15 vs +0.27
     - WILLY: −0.92 vs +0.06
     - pooled: −0.54 vs +0.12
   - The shuffled-path null cannot tell the best-of-grid gain from luck (p = 0.29 to 0.90).
2. **Why "1 in 3 runners" doesn't turn into profit.** A runner pays only if you hold it to +5, and holding to +5 means taking every −3 on the way.
   - On real hit ordering: P(+5 before −3) = 37.5 %, and the breakeven for a +5 / −3 bet is exactly 37.5 %. EV ≈ 0.
   - Runners give back a median 2.7 points between +1.5 and +5, and 48 % give back ≥ 3 points. Any trail tight enough to protect the +1.5 kills the runner. Any trail loose enough to keep it hands the losers back to the stop.
3. **The one exit pattern that does stand out is narrower: FRENZY_LONG's −3 stop looks one point too tight on real fills.**
   - Of the 9 LONG fills stopped at −3, 5 came back to +1.5 without touching −4.
   - Changing only the stop to −4: +0.67 vs +0.02 %/fill. The LODO picks −4 in 13/13 folds: Δ +0.65 [+0.08, +1.36], P 0.99. Fills that weren't stopped are untouched (Δ = 0 exactly).
   - **But the 2-year backtest says no:** only 18 % of −3-stopped LONGs get saved at −4 (vs 56 % real), Δ −0.003, and the newest regime (2026-H2) is clearly negative (−0.25, P 0.00).
   - So this is **an observe line, not a ship** (§A5). At blocker size it is worth about $1 a day either way.
4. **Winner vs loser / runner vs rest (Part B).**
   - Screens covered every one of 131 variables, exhaustive 2D (19k–29k tests per sleeve), and an L1 model, all scored LODO against a day-block null.
   - **No LONG, LITE or pooled separator survives.** LONG LODO AUC is 0.35.
   - WILLY's runner model does beat its null (LODO AUC 0.78, p < 0.02), but it is pure BTC market state (BTC above its 72 h average). That is 3 days of runners followed by 4 days without. The backtest shows no runner lift at the same cut on 1,516 WILLY fills (38.7 vs 37.4 %).
   - It belongs to the existing WILLY_BTC1H observation, not a new rule.
5. **70× instead of 100× (Part C / D): no.**
   - The 70× bar is ~9 h earlier and ~12 % cheaper, but a +1.5 / −3 trade closes ~6 minutes after entry. The cheaper price is never held long enough to matter.
   - Held longer, 56 % of those early entries hit −3 before the 100× bar arrives.
   - Lowering the threshold also *moves* the entry rather than adding one. The 100× bar stops being "fresh ON", so today's good LONG fills disappear, and the earlier bar is often a refused green / ATR bar.
   - Real-fill before/after (today's sizing):
     - T85: −$132 (95 % CI −327 to +8)
     - T70: −$322 (95 % CI −584 to −96)
     - T50: −$103
   - The best LITE fills (70–100× band) would be handed to the $100 LONG.
   - A "scale-in" starter at 70× loses: LODO −0.77 %/fill.
6. **LITE (Part D / F): keep the rules.**
   - Widening LITE's 16.75 h window adds losing fills: 24 h adds 11 fills at −0.36 %, 48 h adds 21 at −0.62 %.
   - Late-episode entries at LITE size would have lost: −$80 on the 14 real late LONG bars, vs +$1.9 at LONG size.
   - LITE's < 20× volume band looks weak on real fills (12 · 50 % · −0.59 %), but it fails the expectancy bar, and in the backtest it is LITE's *best* band (+0.18 on 613 fills). No floor.
   - The one LITE rule worth watching is the **green-bar "judged once" rule.** 92 of 184 replica stretches were consumed by a green candle. Letting the next red bar of the same stretch trade added 48 fills at +0.23 %: +$295, CI −$761 to +$1,287.
   - A 1,992-fill study already refuted "retry later bars" (DECISION_LOG 243). → **observe line only** (§F2).

**Recommendation:**
- **Nothing to ship.** Keep today's exits, sizes, 100× threshold, LITE window and LITE rules.
- Register two observe-only lines, with frozen gates in §G:
  - **LONG_STOP4**: shadow the −4 stop on every FRENZY_LONG.
  - **LITE_GREEN_RETRY**: shadow the next red bar of each green-consumed LITE stretch.
- Fold the WILLY BTC-state runner read into WILLY_BTC1H.

## 0. Data, scope, validation

| block | source | LONG | LITE | WILLY | days |
|---|---|---|---|---|---|
| ② master | `MASTER_POOL_stacked.csv`, kept non-probe CLOSED (B16–B22) | 26 | 24 | 27 | |
| ① live B23 | `scalpars_orders_live_2026-10-19_14-54-01.csv` | 3 | 4 | 4 | |
| **real total** | dedup (opened_at, pair, direction) | **29** | **28** | **31** | 16 (13 / 12 / 9 per sleeve) |
| ③ backtest (footnote) | yr6 seed 1, < 2026-10-03 (`rid/cohort.pkl`) | 623 | 1,452 | 1,516 | |

**Coverage checks:**
- Every `scalpars_orders_*.csv` in ~/Downloads was scanned: 88 exports, 93 unique FRENZY fills.
- Nothing is added beyond master + B23. The 5 master rows with `stack_keep = False` stay out (bearish-day block).
- `scripts/validate_against_master.py`: **ALL CHECKS PASS.**

**Paths:**
- Each fill is walked from its own fill price on ticks (1-second bars).
- The tick archive covered all but 7 fill-days; those were fetched from data.binance.vision (no API weight).
- 10-19 uses public 1m klines (fapi, ≈ 25 requests, weight checked, no 418/429). The last three fills (AGT 10-19 ×2, AVA 10-19) are 1m-provisional, and two of them are censored at 2.0 / 4.5 h.

**Fees:**
- Entry is taker 0.045 %. A resting TP exits as maker (0.018 %).
- Every other exit is taker 0.045 % + 0.03 % slippage.

**Parity of today's exit (walker) vs booked:**
- The exit type is identical on 87 of 88 fills. The exception is BAT WILLY 10-17: booked late-TP +0.27, walker +1.16.
- Values are offset by the fee model: booked master `stack_pct` books TPs at the line, gross.
- DUSK LITE 10-15 and ICNT LONG 10-16 closed early live (−2.54 / −2.35 vs −3.12 walked).
- All comparisons below use the same walker and fees on both sides.

**Sizing for $:**
- LONG $100 notional (1× blocker)
- LITE $2,450 (8×)
- WILLY $2,300 (10×)

These are today's B23 notionals at the $1,225 account.

## Part A — exit math on real fills

### A1. Grid and honest out-of-sample

**Grid:** about 1,000 cells per sleeve (fixed, trailing, breakeven-then-trail, split, ATR-scaled; time caps 1/2/4/12 h).
- Full surface: `_A_surface.csv`.
- Top cells: `_A_grid_top.csv`.
- LODO detail: `_A_lodo.csv`.

| sleeve | today (walked) avg · Σ pts · worst day | in-sample best cell (avg) | **LODO, whole grid** (avg · Δ vs today [95 % day CI] · P) | LODO picks |
|---|---|---|---|---|
| LONG (29) | +0.023 · +0.7 · −3.1 | FIX +5 / −4 / 4 h (+1.06) | **−0.142 · −0.17 [−1.50, +1.45] · 0.42** | 5 different cells |
| LITE (28) | +0.272 · +7.6 · −6.2 | ATR 4× / 3× / 4 h (+0.67) | **−1.147 · −1.42 [−2.40, −0.65] · 0.00** | 7 different |
| WILLY (31) | +0.063 · +2.0 · −5.9 | FIX +6 / −2.5 / 1 h (+0.54) | **−0.916 · −0.98 [−2.07, +0.52] · 0.08** | 5 different |
| pooled (88) | +0.116 · +10.2 · −4.4 | FIX +6 / −4 / 1 h (+0.36) | **−0.540 · −0.66 [−1.07, −0.16] · 0.01** | 6 different |

**Per family (LODO inside one family, still chosen after seeing the grid):**
- LONG's BETR (BE at +1.5, trail 1 from +2, stop −4, 4 h) is chosen in 13/13 folds: +0.71, Δ +0.69 [−0.03, +1.56].
- Pooled BETR and pooled SPLIT: +0.23 (P 0.69).
- Every other family is ≤ today's exit.

As §A3 shows, LONG BETR's edge is the −4 stop, not the trail.

**Shuffled-path null** (1m bars permuted inside each fill's 12 h path, FIX + TRAIL cells, 100 draws): the best-of-grid gain over today is not above the null.

| sleeve | best gain | null mean | null p95 | p |
|---|---|---|---|---|
| LONG | +0.85 | 0.67 | 1.26 | 0.29 |
| LITE | +0.35 | 0.76 | 1.28 | 0.90 |
| WILLY | +0.19 | 0.38 | 0.86 | 0.74 |
| pooled | +0.03 | 0.24 | 0.52 | 0.87 |

**Smoothness:** the "best" fixed cells are isolated peaks. LONG +5/−4 is +1.06, but +5/−3 is −0.02 and +4/−3 is −0.40. LITE +5/−2.5 is +0.62, but +5/−2 is −0.10. WILLY +6/−2.5 is +0.54, but +8/−2 is −0.11. The real-fill surface is noise around today's exit, except along the stop axis for LONG.

### A2. Why 1 in 3 runners doesn't pay (`_A_ev_hit_order.csv`, `_A_runner_anatomy.csv`)

These are real hit orderings (first touch within 12 h, gross), pooled over 88 fills, with the stop at −3.

| target X | P(+X before −3) | breakeven P for +X / −3 | gross EV |
|---|---|---|---|
| +1 | 75.0 % | 75.0 % | 0.00 |
| **+1.5 (LONG today)** | **69.3 %** | 66.7 % | **+0.12** |
| +2 | 58.0 % | 60.0 % | −0.10 |
| +3 | 46.6 % | 50.0 % | −0.21 |
| **+5 ("runner")** | **37.5 %** | **37.5 %** | **0.00** |
| +8 | 20.5 % | 27.3 % | −0.68 |

**The EV is flat across targets.** The runner share falls exactly as fast as the target rises: P(runner) × 5 − P(stop) × 3 = 0.375 × 5 − 0.625 × 3 = 0.

The 1-in-3 runners are the same thing as the 2-in-3 stops of a +5 target. A fixed +1.5 is the only target that sits above breakeven on the real ordering, and only by +0.12 gross.

**Why trailing doesn't fix it:**
- Between +1.5 and +5, the 33 runners give back a median 2.7 points.
- 97 % give back ≥ 1, 73 % ≥ 2, and 48 % ≥ 3.
- They reach +1.5 in a median 17–19 min, and +5 in 60 min (WILLY 33 min).
- A trail of 1–2 points exits most runners near +1.5–3. A trail of 3 points lets non-runners fall from +1.5 back to the −3 stop.

### A3. The one pattern that holds within real fills: LONG's −3 stop (`_A_stopwidth.csv`)

**What the read is.** It is read on the live-stopped cohort (CLAUDE.md rule), on ticks, with the TP fixed at +1.5 / 12 h.

| LONG, stop | avg %/fill | Δ vs −3 | live-stopped 9 fills (9 days): saved · went deeper | Δ on fills NOT stopped |
|---|---|---|---|---|
| −2.5 | +0.178 | +0.155 | (tighter stop) | 0 |
| **−3 (today)** | +0.023 | – | – | – |
| **−4** | **+0.671** | **+0.648** | **5 saved · 4 deeper (−4.12)** | **0.000** (integrity ✓) |
| −5 | +0.533 | +0.510 | 5 · 4 | 0.000 |
| **LODO over the stop axis** | **+0.671** | **+0.648 [+0.08, +1.36], P 0.988** | picks −4 in 13/13 folds | |

**The saved five:** ORCA 10-06, MAGIC 10-11, BAT 10-13, ICNT 10-16 and AGT 10-18 00:30. All five dipped below −3, then went to +1.5 without touching −4. The top fill carries 24 % of Σ Δ. Without AGT 10-19 the result is +0.70.

**LITE is the opposite:** all 11 of its −3 stops go deeper. At −4, LITE averages −0.12 vs +0.27 (Δ −0.39). WILLY already runs −4.

**Backtest at the frozen −4 (footnote; `_A_backtest_check.csv`):**

| period | N | −3-stopped | saved at −4 | Δ (−4 − −3) [CI] | P |
|---|---|---|---|---|---|
| 2025 | 269 | 85 | 14 (16 %) | −0.027 | 0.34 |
| 2026-H1 | 230 | 75 | 20 (27 %) | +0.157 [−0.02, +0.35] | 0.96 |
| 2026-H2 (Jul → Oct 2) | 124 | 42 | **2 (5 %)** | **−0.249 [−0.36, −0.13]** | 0.00 |
| all | 623 | 202 | 36 (18 %) | −0.003 [−0.09, +0.09] | 0.47 |

**Verdict:** the real era saves 56 % of the stops vs the backtest's 18 %. A ≥ 5/9 count at an 18 % base rate has p ≈ 0.02, so the real era is unusual, not typical.
- The backtest refutes a permanent change. Its newest regime is clearly negative.
- The real fills can't be ignored either.
- → **observe line LONG_STOP4 (§G).**
- At blocker size the stakes are tiny: 29 fills × +0.65 % × $100 ≈ +$19 over 16 days. A worse stop costs +$1 per stop.
- A wider stop also holds the pair longer. That changes which LITE entries the LONG blocks, so it must be re-priced if it is ever promoted.

### A4. Other families in the backtest at the real-fill cells (footnote)

- LONG BETR (be1.5 / a2 / g1 / sl4 / 4 h): −0.019 overall, H2 −0.278 (P 0.01).
- SPLIT ½@1.5 + ½@5 BE sl4: −0.070.
- LITE FIX +5 / −2.5 / 4 h: −0.06. LITE ATR 4× / 3×: −0.28 (P 0.02).
- WILLY FIX +6 / −2.5 / 1 h: +0.12 [−0.03, +0.28] (P 0.94), with 2025 +0.17 · H1 +0.10 · H2 −0.06. It is not stable either, and real LODO on it is −1.08.

**Nothing from the real-fill grid holds up in the backtest across regimes.**

### A5. Reference rows (today's exit, $ at today's size, real fills)

| sleeve | N | WR | avg % | Σ $ | worst day $ |
|---|---|---|---|---|---|
| LONG ($100) | 29 | 69 % | +0.02 | +$0.7 | −$3.1 |
| LITE ($2,450) | 28 | 61 % | +0.27 | +$187 | −$153 |
| WILLY ($2,300) | 31 | 81 % | +0.06 | +$45 | −$135 |

## Part B — runner vs rest, winner vs loser (real fills, every variable)

**Variables:** 131 numeric features (coverage: `_B_coverage.csv`).
- every `entry_*` / `entry_frenzy_*` stamp
- `x_*` episode / concurrency features
- 19 `j_*` order-book journal features (BOOK row ≤ 2 min before the entry, same mode)
- 27 `h_*` / `b_*` multi-day kline features from the runner study
- sleeve dummies

Coverage ≥ 70 % for 125 / 123 / 101 / 117 features (LONG / LITE / WILLY / pooled).

Low coverage, read apart: `entry_book_*`, `entry_order_lag_s`, `entry_fill_vs_book_pct` (live execution stamps, also excluded as post-fill), `entry_frenzy_willy_*`, and WILLY's journal book (68 %). Unscored fills sit in neither zone nor rest.

**Outcomes:** run5 (+5 before −3, 12 h), run3, run8, and win at today's exit.

**Screen:**
- 1D at sign / median / terciles.
- Exhaustive 2D median quadrants: 19,012–29,040 tests.
- L1-logistic model (λ 0.05).
- **Every rule chosen leave-one-day-out, cuts included.**
- Null: day-block shuffle (200 draws in-sample; 60 draws of the whole LODO procedure).

| group (N · days) | outcome | in-sample max\|z\| 1D / 2D (null p) | 2D hits vs null mean | **LODO 1D** zone vs rest (p) | **LODO 2D** (p) | **LODO model AUC** (null p95 · p) |
|---|---|---|---|---|---|---|
| LONG (29 · 13) | run5 | 2.58 (0.84) / 3.89 (0.25) | 12 vs 5.0 (0.14) | 20 % vs 47 % (0.73) | 40 vs 38 % (0.42) | **0.35** (0.70 · 0.80) |
| LONG | win | 2.62 (0.77) / 4.10 (0.26) | 6 vs 9.3 | 70 vs 68 % (0.43) | 77 vs 50 % (0.18) | 0.39 (0.67 · 0.63) |
| LITE (28 · 12) | run5 | 3.39 (0.19) / 3.65 (0.41) | 11 vs 5.6 (0.14) | 44 vs 38 % (0.37) | 43 vs 39 % (0.37) | 0.52 (0.74 · 0.37) |
| LITE | win | 2.93 (0.57) / 3.80 (0.51) | 2 vs 5.2 | 50 vs 67 % (0.72) | 53 vs 78 % (0.80) | 0.59 (0.72 · 0.23) |
| WILLY (31 · 9) | run5 | 3.47 (0.09) / 4.76 (**0.015**) | 25 vs 7.3 (0.06) | 36 vs 29 % (0.47) | 43 vs 29 % (0.38) | **0.78 (0.70 · < 0.02)** |
| WILLY | win | 2.98 (0.58) / 3.75 (0.79) | 4 vs 14.3 | 90 vs 60 % (0.07) | 83 vs 86 % | 0.23 |
| pooled (88 · 16) | run5 | 2.87 (0.59) / 4.41 (0.12) | 9 vs 9.1 (0.35) | 32 vs 43 % (0.82) | 54 vs 30 % (0.18) | 0.49 (0.64 · 0.48) |
| pooled | win | 2.80 (0.63) / 4.45 (0.08) | 18 vs 9.7 (0.16) | 66 vs 79 % (0.80) | 72 vs 59 % (0.33) | 0.53 |

**FRENZY_LONG: no separator, at any granularity.**
- The top LONG cells are era / size proxies (`entry_fee`, `entry_desired_notional`) and AGT-heavy multi-day cells. AGT is 40–43 % of the zone in the top 1D / 2D LONG cells, which fails the per-pair rule.
- The LODO model scores *below* chance: AUC 0.35, after 0.39 in the runner study.
- The LONG LODO rule picked most often is "BTC below its 5m EMA13" (6 / 13 folds). It points the wrong way out of sample.

**WILLY runners: the only LODO survivor, and why it is not a rule.**
- The L1 model's weight is BTC above its 72 h average (+0.85), turnover (−0.67), ATR (+0.54) and BTC distance from its EMA13. The top 1D cell is `entry_btc_above72_pct > 50.8`: 9 fills, 78 % runners vs 14 %.
- **In day units it is 3 days.** The WILLY run5 rate by day:
  - 10-09 → 10-13: 25–100 %
  - 10-14 → 10-17: 0 of 13
- It is one market-regime switch inside a 9-day window: the WINDOW-UNITS rule makes it 2 observations.
- Priced as a zone-specific exit (`_B_willy_zone.csv`):
  - LODO on real fills picks FIX +6 / −2.5 / 1 h in the zone in every fold: +1.33 vs +0.06, Δ +1.27 [+0.29, +2.57], P 0.995.
  - But at the frozen cut the backtest (1,516 WILLY fills, 100 % covered) shows **no runner lift**: 38.7 vs 37.4 %.
  - The exit switch is worth +0.04 / +0.05 %/fill overall, and its sign varies by period (2025 +0.09 / H1 −0.02 / H2 −0.01 for +5/−4).
- **Not a rule.** It is the BTC-trend family the existing **WILLY_BTC1H** line already watches. Record it there.

**Pricing:** no LONG / LITE / pooled survivor, so there is nothing to price as a 2× / 3× zone. The WILLY zone is refuted by the backtest and has 3 days of evidence.

## Part C — the 70× early entry

### C1. Real fills: did the same episode have an earlier lower-threshold READY bar? (`_C_real_fill_pairs.csv`)

The engine walk at `frenzy_state_vol_mult` 50 / 70 / 85 found an earlier READY bar for 12 / 9 / 6 of the 57 LONG + LITE fills. At 70×, 4 were LONG fills and 5 were LITE fills.

| real fill | 70× bar | earlier by | cheaper by | 70× entry (today's exit) | real fill | MAE before the real fill |
|---|---|---|---|---|---|---|
| SAND LONG 10-04 05:05 | 03:45 | 1.3 h | 2.6 % | +1.44 TP | +1.44 | −1.4 |
| SAND LONG 10-04 14:05 | 12:20 | 1.8 h | 2.4 % | −3.12 | −3.12 | −3.2 |
| STRK LONG 10-11 13:30 | 10-10 21:40 | 15.8 h | 4.2 % | +1.44 | +1.44 | −3.3 (hit −3) |
| WAL LONG 10-14 14:25 | 06:00 | 8.4 h | **11.7 %** | +1.44 | +1.44 | −0.3 |
| STRK LITE 10-11 05:40 | 10-10 21:40 | 8.0 h | 3.4 % | +1.44 (as LONG) | **+2.94** | −3.3 |
| BAT LITE 10-10 05:40 | 00:25 | 5.3 h | −4.3 % (dearer) | **−3.12** | +2.94 | −7.6 |
| AVA LITE ×3 (10-18 / 19) | 10-14 08:25 / 10-18 14:45 | 21–92 h | 2.7–24 % | **−3.12 / −3.12 / +1.44** | +1.94 / −3.12 / +1.94 | −18 to −20 |

**On every real LONG pair the earlier entry earns exactly the same as the 100× fill.** The cheaper price is irrelevant because a +1.5 / −3 trade closes within minutes.

### C2. The mechanism, on every first 70× READY bar of the real era (replica; `_C_replica_first_bars.csv`, `_C_later_summary.csv`)

| 70× first bar is… | N · days | today's exit at the 70× bar | runners | notes |
|---|---|---|---|---|
| the SAME bar as 100× | 17 · 11 | −0.17 | 35 % | no change |
| EARLIER, 100× comes later | 9 · 7 | −0.08 vs −0.08 at the 100× bar | 44 % | 9.3 h earlier, 11.7 % cheaper (medians) |
| NEVER reaches 100× (a new trade) | 14 · 10 | **−0.84** | 36 % | the cost of entering at 70× |

**Why 11.7 % cheaper ≠ more P&L (LATER class):**
- The early trade exits a median **6.5 minutes** after entry, long before the 100× bar.
- Between the two bars, the early entry's median MAE is **−3.2 %**. **56 % touch −3** and 22 % touch −5 before the 100× bar arrives.
- To *hold* the cheaper price until the move, you need a stop of 5–8 % or more.
- Across the exit grid on the same paths (early vs 100×):

| exit | early vs 100× |
|---|---|
| +1.5 / −3 | −0.08 vs −0.08 |
| +1.5 / −4 | −0.42 vs −0.41 |
| +3 / −3 | +0.92 vs −0.43 |
| +5 / −5 | +1.58 vs +0.47 |
| +8 / −5 | +2.93 vs +2.13 |
| trail a3 / g3 / sl5 | +0.16 vs +1.92 |
| BETR | −0.52 vs −0.58 |

- Those are 9 trades on 7 days, and they flip sign by cell. At 85× the gap is only 0.7 h and the results are identical.

**Scale-in** (`_C_scalein.json`): a starter at the 70× bar with its own stop (3 / 5 / 8 / 12 %) and TP (1.5 / 3 / 5 %); the main leg at 100× is unchanged.
- Base: 23 first-70× bars that are not the 100× bar (LATER + NEVER) on 13 days.
- Best in-sample: stop 12 / TP 3, +0.46 %/fill.
- **LODO: −0.77 %/fill** (it picks 12 / 3 in 11 of 13 folds and still loses out of sample).
- Worst starter day: −6 to −24 pts.
- $ at a $100 starter ≈ −$18 over the era. The cheaper price cannot be harvested.

### C3. The threshold itself, priced (Part D below). Keep 100×.

## Part D — LITE and the volume boundary (before / after)

**Replica parity:**
- The event replica (`study_frenzy_realfills_engine.py`: engine-exact walk, LONG / LITE status, bearish block, WILLY global hold from the real WILLY fills, pair-held, slots, pair-day cap, dislocation) reproduces 27 of 29 real LONG fills (93 %) and 27 of 28 real LITE fills (96 %) on the same bar and sleeve, at T = 100.
- The 3 misses are explained: ORCA at a 99.9× vs 100.4× stamp; ERA was a live catch-up; W LITE was refused for slots in the replica.
- The replica's per-fill outcome equals the real walk on the matched fills (median |Δ| 0).

### D1. Before / after on the REAL fills (hybrid: real fills kept / moved / gone + new fills the lower threshold creates; `_D_hybrid_*.csv`)

| T | same | owner change | gone | new | Σ $ before → after | Δ $ [95 % day CI] | worst day before → after |
|---|---|---|---|---|---|---|---|
| 85× | 39 | 0 | 15 | 19 | +$187 → +$56 | **−$132 [−327, +8]** | −$149 → −$149 |
| 70× | 35 | 3 (LITE→LONG: MET 71×, STRK 75×, WAL 83×) | 16 | 24 | +$187 → −$135 | **−$322 [−584, −96]** | −$149 → −$182 |
| 50× | 24 | 9 | 21 | 34 | +$187 → +$84 | −$103 [−611, +443] | −$149 → −$106 |

**Per batch at T70 (Δ $):**

| batch | Δ $ |
|---|---|
| B16 | +1 |
| B17 | −8 |
| B18 | −45 |
| B19 | −74 |
| B20 | 0 |
| B21 | −72 |
| B22 | −79 |
| B23 | −46 |

Not one batch improves.

**Why so many fills are GONE:**
- At a lower T the state switches ON earlier, so today's 100× bar is no longer a fresh ON ("ON, entry bar passed"). That is the walk code on 13 of the 15 fills lost at 85×, including AGT 10-16 / 10-18 / 10-19 14:10.
- The entry then moves to the earlier crossing bar: those are the 19 / 24 / 34 "new" fills. Or there is no trade at all, when that earlier bar is refused (green candle / ATR high / bearish) and consumed.
- Lower T also makes episodes last longer (more state bars), so LITE's ≤ 16.75 h window closes on episodes it trades today. AVA 10-19 is an example.

**Simulated before / after** (replica, today's rules, every pair; `_D_sim_variants.csv`):

| T | LONG fills | LITE fills | WR | avg % | Σ $ | Δ $ [CI] | P | worst day |
|---|---|---|---|---|---|---|---|---|
| 100 | 39 | 61 | 63 % | +0.10 | +$216 | – | – | −$285 |
| 85 | 37 | 58 | 61 % | 0.00 | +$46 | −$170 [−441, +103] | 0.11 | −$284 |
| 70 | 41 | 53 | 60 % | −0.11 | −$144 | −$360 [−719, +29] | 0.03 | −$291 |
| 50 | 52 | 38 | 67 % | +0.16 | +$197 | −$19 [−1,033, +917] | 0.49 | −$106 |

**LODO choice of T** (day units): it picks 100 on 10 days and 50 on 8. The held-out result is −$567 vs +$216 for 100× (Δ −$783 [−1,427, −293]).

**Keep `frenzy_state_vol_mult` = 100.**

### D2. LITE's own entry volume multiple (`_D_lite_volbands.csv`, `_D_lite_floor.json`)

| band | real N · days · WR · avg % · Σ $ | replica T100 | backtest yr6 (2-year) |
|---|---|---|---|
| < 20× | 12 · 8 · 50 % · **−0.59** · −$174 (LITE C sends all of these to TP +2) | 26 · 62 % · −0.01 | **613 · 65 % · +0.18** (LITE's best band) |
| 20–50× | 6 · 5 · 67 % · +0.75 · +$110 | 18 · 50 % · −0.15 | 487 · 55 % · +0.01 |
| 50–70× | 6 · 5 · 50 % · −0.09 · −$13 | 11 · 55 % · +0.09 | 140 · 55 % · +0.05 |
| 70–100× | 4 · 4 · **100 % · +2.69** · +$263 | 8 · 75 % · +1.30 | **210 · 52 % · −0.16** (LITE's worst band) |

**The < 20× band against the expectancy bar:**
- WR 50 % vs LITE breakeven 55.8 %: passes.
- P(avg < 0) 0.74 (needs ≥ 0.95): fails.
- 8 days (OK), but N 12 < 15: fails.
- **Not a filter candidate.**

**A LITE volume floor, chosen LODO** from {0, 10, 15, 20, 30, 50}×:
- It picks 30× in 10 of 12 folds: +11.6 vs +7.6 pts (+$98 at LITE size on 28 fills).
- Day-shuffle null for the best floor's gain: p 0.087.
- The backtest is the exact opposite: < 20× is LITE's *best* band and 70–100× its worst.
- **No floor. No boundary change.** The real-fill band pattern reverses on the longer record. The 4 real 70–100× winners (MET, STRK, CFX, WAL) are 4 days.

## Part E — the entry timeline (`_E_timeline.csv` per episode, `_E_buckets.csv`)

### E1. Worked example — AGT, spike 10-15 15:15 UTC (hours since spike)

| h | event | vol× | outcome |
|---|---|---|---|
| 2.0 | LONG ON → refused ATR_HIGH (ATR 5.96 %) | 676 | – |
| 6.2 / 8.5 | LONG ON → ATR_HIGH | 220 / 208 | – |
| 10.7 | LONG ON → GREEN_BAR | 109 | – |
| 10.8 | **LITE's only judgement of the stretch → GREEN_BAR (stretch consumed)** | – | – |
| **12.4** | **FRENZY_LONG fill** (10-16 03:40) | 101 | +1.44 (MFE 8.1) |
| 16.75 | LITE window closes | | |
| 55.3 | (85× only: READY 10-17 22:30) | 87 | (ran +40 % before the stop; runner study) |
| **57.3** | **FRENZY_LONG fill** (10-18 00:30), +41 % above VWAP | 102 | −3.12 (MFE 0.0); **saved by a −4 stop → +1.44** |
| 71.1 / 73.2 | LONG ON → GREEN_BAR / ATR_HIGH | 112 / 148 | – |
| **84.5** | **FRENZY_LONG fill** (10-19 03:45) = also the first 70× READY bar of the episode | 136 | +1.44 live (MFE **44.3**) |
| **94.9** | **FRENZY_LONG fill** (10-19 14:10) | 102 | +1.44 (MFE 5.4) |

**Reading:**
- At 70× the episode's ON state starts at 2.0 h and never switches off for an hour. Its only READY bar is 84.5 h, so the 12.4 h / 57.3 h / 94.9 h LONG trades don't exist at 70×.
- LITE never had a chance: its single judgement at 10.8 h was a green candle, and by 16.75 h the window was shut.

### E2. Aggregate — when entries happen and how they do

| hours since spike | sleeve | real N | WR | avg % (today, walked) | runners (+5 before −3) | replica T100 N · avg |
|---|---|---|---|---|---|---|
| 0–2 h | WILLY | 31 | 81 % | +0.06 | 32 % | – |
| 2–6 h | LITE | 9 | 89 % | **+1.93** | 67 % | 23 · +0.57 |
| 2–6 h | LONG | 6 | 67 % | −0.08 | 17 % | 9 · −0.08 |
| 6–16.75 h | LITE | 19 | 47 % | **−0.51** | 32 % | 38 · −0.11 |
| 6–16.75 h | LONG | 9 | 67 % | −0.08 | 44 % | 10 · +0.07 |
| 16.75–96 h | LONG only | 14 | 71 % | +0.14 | 43 % | 20 · +0.07 |

**Where the 70× first bar sits (replica):**

| hours | SAME | LATER | NEVER |
|---|---|---|---|
| 2–6 h | 9 · −0.59 | 1 | 1 |
| 6–16.75 h | 5 · −0.39 | 7 · −0.52 | 6 · **−2.36** |
| 16.75–96 h | 3 | 1 | 7 · +0.14 |

The early-70× entries cluster in LITE's 6–16.75 h window, which is the window where entries do worst.

**LITE early vs late in its window (2–6 h vs 6–16.75 h):** this is the W/L screen's "frenzy hours ≤ 7" cell, already refuted in the backtest (1 of 3 regimes; FRENZY_WL_SCREEN §4). It is descriptive here, not a candidate.

## Part F — LITE window / green bar (why LONG and not LITE)

### F1. LITE's max-hours window (replica, T100; `_F_lite_variants.csv`)

| window | LITE fills | added (avg %) | lost | Σ $ | Δ $ [CI] | P | worst day |
|---|---|---|---|---|---|---|---|
| **16.75 h (today)** | 61 | – | – | +$216 | – | – | −$285 |
| 24 h | 72 | 11 (**−0.36**) | 0 | +$118 | −$97 [−508, +245] | 0.33 | −$237 |
| 36 h | 75 | 15 (−0.02) | 1 | +$137 | −$79 | 0.37 | −$237 |
| 48 h | 79 | 21 (**−0.62**) | 3 | −$20 | −$236 | 0.19 | −$237 |
| 72 / 96 h | 81 | 23 (−0.61) | 3 | −$49 | −$265 | 0.15 | −$237 |

- LODO picks 16.75 h in 17 of 18 folds.
- The added late LITE entries have a median age of 20–24 h and lose. The LONG interaction is small: 0 to 1 LONG fills lost to pair-held.
- **Keep 16.75 h.**

### F2. Green-bar refusal and one judgement per stretch

**Counts:**
- Replica, real era: 184 LITE stretches judged. **92 were consumed by a green candle**, and 63 traded.
- Journal (real bot): 52 `FRENZY_LITE_GREEN_BAR` rows on 22 pairs.

| variant (T100) | LITE fills | added · avg % | Σ $ | Δ $ [CI] | P |
|---|---|---|---|---|---|
| today (green bar consumes the stretch) | 61 | – | +$216 | – | – |
| a green bar does NOT consume; the next red bar of the stretch may trade | 107 | 48 · **+0.23** | +$510 | **+$295 [−761, +1,287]** | 0.72 |
| retry every bar until a fill | 128 | 70 · +0.23 | +$715 | +$499 [−595, +1,591] | 0.81 |

**Prior evidence:** the shipped rule came from the opposite finding on the backtest. Retrying later bars until a fill turned +0.163 into −0.005 %/trade on 1,992 fills (DECISION_LOG 243). Here we have 18 real-era days with a CI that spans zero widely. → **observe line LITE_GREEN_RETRY (§G)**, no change.

### F3. Late episodes (> 16.75 h, LONG-only territory): LONG at blocker size vs a LITE-style entry at LITE size, same bars (`_F_late_long_vs_lite.csv`)

| block | N | LONG +1.5 / −3 at $100 | LITE C (+2 / +3, −3) at $2,450 |
|---|---|---|---|
| real late LONG fills | 14 | +0.14 %/fill · **+$1.9** | −0.23 %/fill · **−$80** |
| replica T100 late LONG | 20 | +0.07 · +$1.4 | −0.19 · −$94 |
| backtest yr6 late LONG (footnote) | 260 | +0.017 | +2 / −3: +0.011 · +3 / −3: −0.075 |

AGT 10-19 03:45 (+44 % MFE) is the case that makes multi-day LITE look attractive. Its own bar would have booked +2.94 as a LITE. But AGT 10-18 00:30 on the same multi-day trend would have cost −$76, as would ENJ, SAND, RLC, UMA, CTSI and ERA in the same zone.

**Late-episode entries don't deserve LITE size.** The operator is right that only LONG can buy multi-day trends. That is the correct allocation, not a missed edge.

## G. Proposed observe lines (operator decides; nothing armed)

**🔒 LONG_STOP4 (observe-only).**
- **Cohort:** every FRENZY_LONG fill opened after this report (2026-10-19 16:00 UTC). The 9 motivating stops never count.
- **Shadow:** each fill is re-walked on ticks with the stop at −4 (TP +1.5, 12 h, maker TP, same fees) vs the live −3.
- **Read on the LIVE-STOPPED cohort** (fills the live −3 stopped), at **≥ 15 live-stopped fills on ≥ 8 UTC days.**
- **CANDIDATE** only if all of:
  - saved share (later reached +1.5 without −4) ≥ 50 %;
  - mean Δ (−4 minus −3) over **all** LONG fills in the cohort > 0, with day-bootstrap P ≥ 0.90;
  - Δ = 0 on every fill the live −3 did not stop (walker integrity);
  - no day carries > 50 % of Σ Δ;
  - then re-price the LITE entries the longer LONG hold would block, and apply the 30–50 % haircut.
- **RETIRE** if the saved share is ≤ 25 % (the backtest base rate is 18 %) or the mean Δ ≤ 0 at the read.
- Never re-fit the −4.

**🔒 LITE_GREEN_RETRY (observe-only, low prior).**
- **Cohort:** every FRENZY_LITE stretch, opened after this report, whose judged bar is refused `FRENZY_LITE_GREEN_BAR`.
- **Shadow entry:** the next red / flat bar of the same stretch that passes `frenzy_lite_status` (vol < 100×, 2 h ≤ age ≤ 16.75 h, ≥ 12 closes above VWAP, 24 h volume), at close + 8 s, priced at LITE C on ticks.
- **Read at ≥ 20 shadow fills on ≥ 10 UTC days.**
- **CANDIDATE** only if the shadow mean > 0 with day-bootstrap P ≥ 0.90, it is ≥ the same-period live LITE mean, and no day or pair is ≥ 50 % of Σ.
- **RETIRE** if the shadow mean ≤ 0.
- Never re-fit.

**WILLY BTC-state runners:** no new line. Record the real-fill observation (runners on BTC-above-72 h days, 3 days) under the existing WILLY_BTC1H line. Its frozen gate is unchanged.

## H. What cannot be concluded (blind spots)

- **N.** 29 / 28 / 31 real fills on 13 / 12 / 9 days. Every real-fill effect is below the 8-window gate except the LONG stop, which has 9 stops on 9 days. The backtest refutes or fails to confirm every real-fill finding.
- **Real outcomes are walked, not booked.**
  - The walker is identical to booked on exit type for 87 / 88 fills.
  - $ are at today's size on fills that traded at older sizes and exits.
  - The 3 last fills are 1m-provisional; 2 are censored at 2.0 / 4.5 h.
- **The replica (Parts C / D / F) does not model:**
  - the ≤ 25-pair FRENZY shortlist
  - WIDE / manual / momentum positions (they hold pairs and slots)
  - pauses and restarts inside a batch
  - the LONG catch-up beyond the WILLY hold
  - the bearish block, which uses rebuilt stamps; the BTC 1h slope for LITE C is rebuilt
- **Data:**
  - Five CJK-named pairs had no 10-19 5m data. No real fill uses them.
  - 32 candidate bars (PUMPBTC) are unpriced.
  - 2 of 50 fetched tick days were missing, so those used 1m.
- **Order-book journal features** exist only from 10-03 16:00, with 68 % coverage on WILLY. They are not in any backtest.
- **No sizing rule was priced** because no LONG / LITE / pooled separator survived LODO.
- **The backtest exits are today's exits, walked with the same walker.** They are not the engine's own `pct_new`. LITE C and the WILLY late TP are included per fill.

## Files

- **Report:** `reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19.md`.
- **Per-fill table:** `reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19.csv` (88 real fills: booked, walked, MFE / runners, key cells, Part C / D status).
- **Tables:** `_A_grid_top` · `_A_surface` · `_A_lodo` · `_A_lodo_picks` · `_A_smooth` · `_A_null` · `_A_ev_hit_order` · `_A_runner_anatomy` · `_A_stopwidth` · `_A_backtest_check` · `_A_fills` · `_B_coverage` · `_B_screen_lodo` · `_B_screen_top` · `_B_lodo_picks` · `_B_outcomes` · `_B_willy_zone` · `_C_real_fill_pairs` · `_C_replica_first_bars` · `_C_later_pairs_grid` · `_C_later_summary` · `_C_scalein.json` · `_D_parity` · `_D_sim_variants` · `_D_hybrid_rows` · `_D_hybrid_summary` · `_D_lite_volbands` · `_D_lite_floor.json` · `_E_timeline` · `_E_buckets` · `_F_lite_variants` · `_F_late_long_vs_lite` · `_CDEF_summary.json`.
- **Scripts** (`scripts/study_frenzy_realfills_*.py`):
  - `fetch` (data.binance.vision + budgeted fapi)
  - `common` (cohort, paths, walker, fees)
  - `paths`
  - `exitgrid`
  - `btgrid` / `btcheck` (footnote)
  - `ev`
  - `stopwidth`
  - `runner`
  - `willyzone`
  - `walk` (engine-exact 5m walk at 50 / 70 / 85 / 100×)
  - `engine` (candidates, pricing, event replica)
  - `parts` (C / D / E / F)
  - `partc2`
  - `csv`
