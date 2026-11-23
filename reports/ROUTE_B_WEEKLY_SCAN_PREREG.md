# 📡 Route B weekly research scan — PRE-REGISTRATION

**Frozen 2026-11-23 15:01 UTC.** No outcome of this scan has been read or printed. (The loader computes and stores outcome
columns, `out_*`, `fwd_*` and `g_*`. The H-trigger outcomes have never been printed; the sleeve outcomes printed so far are listed in §0.)

**Version history**
- v1 was written at 14:52 UTC. A caveman review and a deep review both ran on it before any commit, and this v2 applies their findings.
- No data was read between v1 and v2.

**Asks behind it**
- Operator, 2026-11-19: "a WEEKLY research scan once ~3 weeks of data exist".
- Operator, 2026-11-23: "write the PREREG file for the weekly scan".

**Rules for this file**
- Research only: no engine, config, recorder or AWS change.
- Any later change is an **ADDENDUM** at the bottom, timestamped and written before the data it affects is read. Nothing above the addenda is ever edited.
- The git commit that adds this file is its freeze proof.

---

## 0 · What has already been seen (disclosure, repeated in every scan report)

| Already seen | Where | Consequence |
|---|---|---|
| 18 closed fills: state at entry and during the trade | ROUTE_B_REPORT 11-21 / 11-23 §2 | REAL-fill rows are **descriptive only**, never a DESIGN input |
| Imbalance (`imb_05`) and taker-buy share vs the next 1 / 5 / 15 min, every hot coin, on **11-20, 11-21 and 11-22**. Imbalance had t up to +4.9 on 11-21. High buy share was followed by lower prices, the same sign in 24 of 24 hours | route_b_daily first view §③ | **Every variable built from `imb_05` or taker share is HOLDOUT-only**: H3, H5, S1-c, S4-b, and any Q1 variable of those families. Their DESIGN results are printed but never count |
| The daily first view **keeps running** (daily_report.py → first_view) and keeps reading those two variables | route-b-daily-health task | Same rule. It does not read any other Route B variable |
| Taken vs blocked net, WR, MFE and MAE per gate, on the loader's priced signals | ROUTE_B_SIGNALS 11-21 / 11-22 §5 (printed on every build) | These are gate-level outcomes, not Route B variables. **The Q1 cohort excludes core-rule blocks (§2)**, so no rule gate is re-tested through a proxy |
| Priced refusals: WILLY / HT FAILED_SPIKE (frozen KEEP); DIST NOT_VSW / VOL24 / POS_CHG / RSI1H; EXH VOL24 / VOL_DECAY / NO_SPOT | DL 628, DL 630 | Gate-level outcomes, same treatment. S5's sample starts at the DL 630 commit |
| Base rates of the EXH (62), DIST (77) and HT (7) SHADOW cohorts, and order-flow EXIT signals on them | DL 626 (PEAK_STUDY, PEAK_VS_REAL_EXIT) | Exits stay with **DL 626's own pre-registered re-read**. They are not re-tested here |
| The 11-18 → 11-22 batch re-priced; positioning at entry → fast stop (backtest) | DL 627, not supported | S2-c is a pre-declared re-test with a low prior |
| ENSO 11-21 (OI rising into a squeeze) and BLESS 11-21 (a short after a −21 % crash) | ROUTE_B_REPORT §2b | **S1-b and S2-a are HOLDOUT-only.** They were built from DESIGN-day cases |
| H1 / H2 / H5 / H6 on 3.4 years of klines and metrics, **including OI-falling legs** (dOI ≤ −3 % / −1 %): all **REFUTED** | NEWSLEEVE 11-21 | Low prior. The only new leg is the **book-depth** condition (§5) |
| DIST after a crash (refuted and reversed on 1,547 fills), DIST thin book, DL 602's 112 variables (no filter) | DL 612 area, DIST_AFTERCRASH / DIST_THIN, DL 602 | S2-a and S2-b have **low priors** (stated) |
| H1–H7 trigger counts per day | ROUTE_B_REPORT §4 | Counts only. H outcomes have never been printed |

---

## 1 · Questions

| # | Question | Unit | Status |
|---|---|---|---|
| **Q1** | Per sleeve: does any coin-level book / tape / liquidation / funding / OI / positioning state **at the signal** separate signals that WORK? | one row per sleeve-possibility | DESIGN screen + HOLDOUT |
| **Q2** | Pre-declared theory lines per sleeve (§4): does each hold in its predicted direction? | same | DESIGN (except the HOLDOUT-only lines) + HOLDOUT |
| **Q3** | New sleeves H1–H7 (§5): does a trigger beat same-coin random entries **after the recorded cost at our size**? | one row per trigger-possibility | DESIGN screen (except H3 / H5) + HOLDOUT |
| **Q4** | The MARKET row: a sleeve on/off context | **UTC days** (WINDOW-UNITS rule) | **HOLDOUT-only**, descriptive until §6 B1 |
| **Q5** | Cost and capacity on the real book (entry and stop), against the model slip and the paper drift | descriptive, weekly | no gate |

Out of scope:
- Exits (DL 626 owns them).
- Plain FRENZY_WILLY: the loader cannot price its V2b exit, so it is shown as UNTESTABLE with its N.
- Momentum and flips: their entries are off.

---

## 2 · Data and cohorts

**Source:** the DL 612 loader (`scripts/route_b_signals/build.py`).

**Q1 / Q2 cohort, per sleeve.** Possibilities (`poss_dup` False) that met the sleeve's **core rule** (`core_rule_met` True) and were then:
- taken;
- blocked by an armed **filter** line; or
- refused by **capacity** (pair held, slots, the depth guard, the open path).

Rows blocked by a core-rule gate are left out of Q1 / Q2:
- EXH_NO_SPOT, EXH_VOL24_LOW, EXH_VOL_DECAY;
- DIST_NOT_VSW, DIST_VOL24_LOW;
- FLATBASE NOT_FLAT / SKIPPED;
- every other gate of the sleeve's own rule.

They appear only in a labelled descriptive table.

Other rules:
- Full size only. Each sleeve is its own family and is never pooled.
- **Power, stated now:** at the recent rate the EXH and DIST cohorts grow by ~1–3 a day, FLATBASE by less. Most Q1 lines will read COLLECTING for weeks. That is the expected result, not a failure.

**Q3 cohort:** H openers (`triggers_H.*`, one per possibility), alts only.

**Build items before the 12-10 run.** Each is a code change with tests, reviewed and committed separately. None reads any outcome.
1. **DL 611 columns in the state block:**
   - full book: `fb_bid_usd_{01,02,05,1,2,5}` / `fb_ask_usd_…`, `fb_cost_{buy,sell}_{1000,3000,6000,10000,25000}`, `fb_spread_bps`, `fb_reach_*`, and a derived `fb_imb_05`;
   - open interest: `oi_chg_1m` / `oi_chg_5m`, and derived 15 / 60 / 240-min OI change;
   - positioning: `top_ls_acct`, `top_ls_pos`, `global_ls_acct`, `taker_bs_ratio`, and their 60-min changes;
   - a coverage flag on every one (unscored stays unscored, never "rest").
   - **OI / ratio lag rule (look-ahead guard):** features.py stamps OI and ratio rows at the **sweep start**, so a row is usable only if its minute ≤ the state minute − 1. A positioning value is usable only if its `ratio_ts` + 10 min ≤ T. Both rules are unit-tested.
2. **Full-book pricing:** the sleeve NET uses the full-book cost (`fb_cost_*_6000`, entry and exit) where the coin was in the full book. Otherwise it uses the 20-level cost and is flagged "lower-bound cost".
3. **The WORKED and REACH labels** (§3), plus path decomposition per sleeve (CLAUDE.md entry × exit ①), descriptive.
4. **The BTC daily-trend stamp** (1 d EMA50 vs EMA200 at T).
5. **H7, the conditioned variants and the random-minute controls** (§5, §6).
6. **The S0 market columns** that do not exist yet.
7. **Q2 variable definitions** (§4), built exactly as written.

**Exit changes:**
- The loader re-prices every row under the exit live at run time.
- A line whose HOLDOUT verdict is already frozen keeps it.
- A DESIGN survivor not yet judged is re-checked **once** on the re-priced DESIGN rows. If it fails, it is dropped.
- **No new DESIGN screen and no new candidates.** An exit change never gives a line a second chance.

**DESIGN / HOLDOUT, by signal time in UTC**
- **DESIGN** = rows with signal time in 2026-11-19 23:23 → 2026-12-07 00:00 that are **visible at the 12-10 run**. Late arrivals with a DESIGN signal time are excluded for good.
- **DESIGN halves:**
  - H-a: → 2026-11-28 12:00.
  - H-b: 2026-11-28 12:00 → 12-07 00:00.
  - For DL 611 columns and H7, whose data starts at 2026-11-21 22:02: H-a = 11-21 22:02 → 2026-11-29 11:00, and H-b = from 2026-11-29 11:00.
- **HOLDOUT** = signal time ≥ 2026-12-07 00:00, accumulating.
- **Optional-stopping guard:** a scan counts only rows with signal time ≤ the scan cutoff − the sleeve's max hold (EXH / DIST / HT 24 h, FB 72 h, triggers 4 h). This stops fast exits from being over-represented at the cut.

---

## 3 · Outcomes (labels)

**Sleeve signals.** The **pass is decided on WORKED**; REACH and NET are shown beside it.
- **Primary: WORKED.**
  - The 1m path is walked from the entry minute, exactly as the loader walks it.
  - WORKED = true if the price moves +1.0 % our way before the sleeve's stop distance, within the max hold.
  - A minute that hits both +1 % and the stop counts as **NOT** worked (AMBIG → 0, conservative).
  - Neither hit within the hold also counts as NOT worked.
  - Known weakness: the base rate is likely ≥ 75 % (most losers are give-backs), so power is low.
- **REACH:** the maximum favourable % before the stop.
- **NET %:** under today's exit, the full-book cost where it exists, fees and real settled funding.
- **NET agreement:** a pass on WORKED also needs the NET difference to point the **same way**. If it points the other way, the result is "contradicted".

**H triggers:**
- Entry at the open of minute m + 2.
- **Primary: the stored grid net `g_tp{a}_sl{b}_net`, used AS IS.** It already includes fees, the slip on the taker legs and funding. Nothing is subtracted again.
- One cell per trigger is chosen on DESIGN (§6 A1-Q3). H3 and H5 use a fixed **TP 2 / SL 2** cell.
- Rows whose coin had **no recorded book** (the default 0.10 % slip, mostly H6) are flagged. They are left out of every Q3 pass statistic and shown separately.

---

## 4 · Variables

### Q1 — the exhaustive screen (W/L = EVERY variable)
Q1 covers every coin-level `rb_*` and DL 611 column at the signal, plus every entry_* stamp the sleeve's journal row carries.
- **`mk_*` market columns are NOT in Q1.** They are market-wide, so they go to Q4 in day units.
- Each 1D variable is tested at 3 granularities:
  - **sign** (natural zero), positive vs non-positive;
  - **terciles**: top vs bottom tercile;
  - **quintiles**: top vs bottom quintile.
  - Cuts are DESIGN quantiles, frozen.
- **2D:** every pair of variables, each of the 4 quadrants (DESIGN medians) vs the other three.
- **Statistic:** the difference in WORKED rate. Its t-value uses a **window-clustered** standard error (window15).
- **First tables:** coverage and blind spots. A variable scored on < 50 % of the cohort gets its own read on the scored rows only.

### Q2 — theory lines

Direction is frozen. "−" means more of X gives fewer WORKED. Definitions are exact. "p90" and "p99" are the coin's own distribution over the **7 days before the state minute** (minimum 20 non-zero values, else unscored). A "prior median" is over the **24 h before** (minimum 60 recorded minutes).

| Line | Sleeve | Definition at the state minute | Dir | DESIGN? | Prior |
|---|---|---|---|---|---|
| S1-a | EXH | 5-min short liquidations ≥ the coin's p99 of non-zero 5-min sums | − | yes | new |
| S1-b | EXH | OI change 15 min > 0 AND mark return 15 min > 0 | − | **HOLDOUT-only** (ENSO) | new |
| S1-c | EXH | taker-buy share 15 min ≤ 0.45 | + | **HOLDOUT-only** (first view) | new |
| S2-a | DIST | coin mark return 15 min ≤ −10 % AND 15-min long liquidations ≥ the coin's p99 of non-zero 15-min sums | − | **HOLDOUT-only** (BLESS) | **low** (AFTERCRASH reversed) |
| S2-b | DIST | full-book ask $ within 0.5 % < the coin's prior median | − | yes | **low** (DIST_THIN) |
| S2-c | DIST | funding ≤ the lower tercile of DIST DESIGN rows AND `global_ls_acct` ≤ its lower tercile on the same rows | − | yes | **low** (DL 627) |
| S3-a | FLATBASE | OI change 240 min > 0 | − | yes | new |
| S3-b | FLATBASE | 240-min short liquidations ≥ the coin's p90 of non-zero 240-min sums | − | yes | new |
| S4-a | WILLY_HT | full-book bid $ within 0.5 % ≥ the coin's prior median | + | yes | new |
| S4-b | WILLY_HT | taker-buy share 5 min ≥ 0.55 | + | **HOLDOUT-only** (first view) | new |
| S5 | EXH_NO_SPOT (DL 630 cohort) | 15-min short liquidations ≥ the coin's p90 of non-zero 15-min sums, OR OI change 15 min > 0 | − | yes, from the DL 630 commit | new |

- S5 uses this scan's WORKED label on the DL 630 scout's cached entry prices (build item 3).
- S5 never touches DL 630's own frozen gate: it is a split of that cohort, not a second verdict on it.

### Q4 — the MARKET row (S0, HOLDOUT-only, day units)
Each variable is a **daily mean** of its per-window value:
- breadth: % of coins up over 15 / 60 min;
- market long and short liquidations over 15 min;
- BTC $ within 0.2 %;
- BTC taker-buy share over 15 min;
- mean funding and mean premium across coins.

The per-sleeve daily WORKED rate is compared across days, two-sided (no direction is declared), at sign granularity first, then by tercile.

---

## 5 · New-sleeve triggers

**H1–H6** are exactly as in DL 612's `scripts/route_b_signals/triggers.py`:
- alts only, with BTC and ETH out;
- one possibility per coin per 60 min (H6: per 24 h);
- H1 and H2 need the coin's book recorded;
- H4 needs ≥ 60 prior rows;
- H5 needs ≥ 60 prior recorded minutes.

| # | Trigger | Side |
|---|---|---|
| H1 | 5-min long liquidations ≥ the coin's prior p99 (≥ 20 non-zero) and ≥ $50k | LONG |
| H2 | the same, on short liquidations | SHORT |
| H3 | `imb_05` ≥ +0.5 for 3 recorded minutes in a row, bid + ask within 0.5 % < $40k (≤ −0.5 → SHORT) | follow — **HOLDOUT-only** |
| H4 | ask $ within 0.5 % < 0.35 × the prior median → LONG; the bid side → SHORT | toward the vacuum |
| H5 | 15-min traded $ ≥ 5 × the prior norm and taker-buy share ≥ 0.6 → SHORT (≤ 0.4 → LONG) | fade — **HOLDOUT-only** |
| H6 | funding ≤ −0.5 % per settlement → LONG · ≥ +0.1 % → SHORT (the current rate; 1 h and 8 h settlement coins mixed, stated) | against the crowd |

**H7 is new.** It uses the recorder's own OI sweep under the §2 lag rule (hot coins only), the minutes s − 5 … s − 1, and is three separate keys, each with its own episodes:
- **H7a:** OI change 5 min ≤ −3 % AND mark return 5 min ≤ −2 % → LONG.
- **H7b:** OI ≤ −3 % AND mark return ≥ +2 % → SHORT.
- **H7c:** OI ≥ +5 % AND mark return ≥ +2 % → SHORT.

**Conditioned variants** (H1c, H2c, H4c, H6c, H7ac / bc / cc): the trigger AND the full-book depth within 0.5 % on the side that absorbs our adverse move, which is ≥ the coin's prior median. That side is the **bid** for a LONG and the **ask** for a SHORT.
- The OI leg is **not** used: it was refuted on history.
- Each variant picks its own grid cell inside the same null as its parent.

---

## 6 · Statistics and pass rules (frozen)

**Nulls**
- **Q1 / Q2: rotation null.** Within a sleeve, sort the possibilities by time. Rotate the outcome series against the variable rows by a random offset of at least one UTC day of rows, wrapping round. This keeps the clustering inside each window and over time. 1,000 rotations, seed 20261210, and **each rotation re-runs the whole family**: every Q1 variable × granularity × 2D quadrant, per sleeve.
- **Q3: matched random-minute control.**
  - For each trigger possibility, draw **5** control minutes on the same coin, the same UTC day and the same side, with seed 20261210.
  - Eligible control minutes: the coin's book was recorded, and the minute is not within ±60 min of any trigger of the same family.
  - Each control is priced on the same grid, with the same cost source.
  - **Statistic:** mean trigger net − mean control net.
  - **Null:** shuffle the trigger / control tags within each coin-day, 1,000 times, re-picking the best grid cell each time across the **whole Q3 family**. The family is every DESIGN-eligible trigger key (H1, H2, H4, H6, H7a / b / c), plus their conditioned variants, × 16 cells.

**A · DESIGN screen (run once, at the 12-10 scan)**
- **A1-Q1:** a separator survives if all of these hold:
  - |t| > the 95th percentile of the rotation null's family-maximum |t|;
  - the same sign in DESIGN halves H-a and H-b;
  - N ≥ 10 on each side and ≥ 5 distinct days on each side;
  - the NET difference points the same way.
- **A1-Q3:** a trigger key or variant survives if all of these hold:
  - its best-cell Δ beats the 95th percentile of the family-maximum Δ;
  - the same sign in H-a and H-b;
  - ≥ 20 possibilities and ≥ 5 days.
  - Its cell is frozen.
- **A2 (Q2 lines that run on DESIGN):**
  - a one-sided window-clustered bootstrap with p < 0.10 in the predicted direction;
  - the same N / day floor as A1;
  - a line that comes out the wrong way is marked **contradicted** and dropped. Eleven lines → about one false pass is expected; HOLDOUT has to confirm.
- **A3:** the survivor list (cut, side, cell, DESIGN effect) is written to `reports/ROUTE_B_WEEKLY_SCAN_SURVIVORS.json` and **committed before any HOLDOUT row is read**.
  - HOLDOUT-only lines go on the list automatically, with their frozen definition.
  - Nothing else is ever read on HOLDOUT as a candidate.

**B · HOLDOUT verdict, read ONCE per line, at the first weekly scan where the line's HOLDOUT cohort reaches its N**
- **B1, minimum N:**
  - Q1 / Q2: ≥ 15 on each side, on ≥ 8 distinct days.
  - Q3: ≥ 30 possibilities on ≥ 10 days.
  - Q4: ≥ 10 days on each side.
- **B2, for Q1 / Q2 / Q4, all must hold:**
  - the same sign as DESIGN, and an effect ≥ 50 % of DESIGN. A HOLDOUT-only line needs only the predicted sign; Q4 is two-sided;
  - a window-clustered bootstrap 95 % CI that excludes 0 (Q4: a day bootstrap);
  - no pair and no day carries ≥ 50 % of the effect;
  - **leave-one-out:** dropping any single pair, and separately any single UTC day, keeps the sign and ≥ 50 % of the effect;
  - the NET difference points the same way.
- **B3, for Q3, all must hold:**
  - the frozen cell's mean net > 0, with a window-clustered 95 % CI above 0;
  - Δ vs the matched control with its CI above 0;
  - Kelly at the one-sided 95 % Wilson WR floor > 0;
  - B2's concentration and leave-one-out checks;
  - capacity is quoted.
  - H3 / H5: the predicted sign replaces "same as DESIGN".
- **B3b, conditioned variants:** must also beat **their own unconditioned trigger on the same coin set** (rows where the depth was scorable), with a Δ CI above 0.
- **B4, regime:** if HOLDOUT has seen only one BTC daily-trend sign, a pass is labelled **"one regime"** and is capped at OBSERVE.
- **B5, haircut:** any Δ quoted from DESIGN gets the 30–50 % in-sample haircut.
- **Accepted risk:** each sleeve is its own family, so there is no correction across sleeves. HOLDOUT is the guard.

**C · What a pass can become**
- A Q1 / Q2 pass → an **OBSERVE line**: a scout with its own frozen revert gate, priced on the real book. ARMING needs the locked expectancy bar or the W bar, the joint entry × exit check, the one-size DCR before → after, and the operator's OK at a batch review.
- A Q3 pass → a new-sleeve **proposal**: design + scout first, and live only after the operator's OK (DL 597 phase 3).
- A Q4 pass → a **sleeve on/off observe line** in day units.
- Never: a re-fit on the data that failed, a cut moved after DESIGN, or a pass from REAL fills alone.

**D · Every status is reported.** COLLECTING · SURVIVED DESIGN · PASSED · FAILED · CONTRADICTED · STALE (exit re-check) · UNTESTABLE.

---

## 7 · Schedule and report

- **Weekly, Wednesdays, from 2026-12-10.** The scheduled task is to be created with the operator's OK.
- **12-10:**
  1. Run the loader.
  2. Run the DESIGN screen, once.
  3. Commit SURVIVORS.json.
  4. Start the HOLDOUT status.
- **Later weeks:** HOLDOUT status, plus Q5 and the Q4 descriptives. DESIGN is never re-run; a code fix re-runs it with the frozen cuts and cells only.
- **One organised report per week:** `reports/ROUTE_B_WEEKLY_SCAN_<date>.md`.
  - Contents: §0; coverage and blind spots; a status table; then the sections in a fixed order (EXH · DIST · FLATBASE · WILLY_HT · no-spot · triggers · market); an event-study chart for any pass; the BTC-sign split; path decomposition; capacity.

## 8 · Known limits
- **Liquidations** are a 1-per-second-per-coin sample.
- **The 20-level book** understates depth on fine-tick coins (up to 11–13×). The full book covers only ≤ 30 priority coins, and only from 11-21 22:02.
- **Visibility:** signals exist only once an export holds them.
- **FLATBASE** will likely read COLLECTING for months.
- **Paper drift is not slippage:** Q5 uses the recorded book.
- **The WORKED base rate is high**, so power is low. REACH and NET are shown to give context.

---

## Addenda
*(any change goes here, timestamped, before the data it touches is read)*

### Addendum 1 — implementation pins and stated deviations for build items 1–7 (written 2026-11-23 17:42 UTC, before any scan outcome was read)

**Why.** The build items needed exact readings of a few phrases above. The reviews (caveman + deep, per item, then on this addendum)
asked for each choice to be frozen here before any DESIGN row is read.

**What was read while writing it.** No outcome column (`out_*`, `fwd_*`, `g_*`, labels) was printed or summarised. Only coverage
counts were read (rows scored, openers, controls, stubs).

**Freeze proof.** The git commit that adds this addendum.

**Scope.** These pins fix how the code reads §0–§8. Items marked **[DEVIATION]** narrow or widen a cohort for data-quality reasons;
each is stated here, with its reason, before any DESIGN outcome is read. No question, label, threshold or gate is changed.

**Item 1 — DL 611 state (OI / positioning / full book)**
- **OI**
  - The newest row with minute ≤ s − 1, at most 3 min before s.
  - `oi_chg_1m` / `oi_chg_5m` are the recorder's own % on that row.
  - Derived 15 / 60 / 240-min change = % of that row's `oi` vs the newest row at or before (its minute − w), at most 5 min older.
- **Positioning**
  - Rows are de-duplicated per `ratio_ts`, keeping the earliest minute it was seen.
  - Usable = `ratio_ts` + 10 min ≤ T **and** that first-seen minute ≤ s − 1 (the same sweep-start rule as OI).
  - Read the newest usable row, if T − `ratio_ts` ≤ 30 min.
  - The 60-min change = relative % vs the usable row whose `ratio_ts` is nearest `ratio_ts` − 60 min (± 10 min).
- **Cross-hour merge:** each family comes whole from one row (OI: the first row with an `oi`; positioning: the newest `ratio_ts`).
- **Full book:** the last fb row in [s − 2, s]. Band columns carry a `_lb` flag when the trusted reach is shorter than the band.
- **Unscored** = NaN on every key, plus `has_fb` / `has_oi` / `has_ratio`.
- **HOLDOUT-only per §0** (the imbalance and taker-share families):
  - `rb_imb_05`, `fb_imb_05`;
  - `rb_taker_buy_share_5`, `rb_taker_buy_share_15`;
  - `taker_bs_ratio`, `taker_bs_ratio_chg60_pct`;
  - every 2D pair containing one of them.
  - Their DESIGN results are printed, but never count.

**Item 2 — full-book pricing**
- **Entry-leg order:** the real fill price → the full book → the 20-level book → the journal model slip → the default slip.
- **Exit-leg order:** the full book at the exit minute → the 20-level book at the exit minute → the full book at the signal → the
  20-level book at the signal → the default slip.
- **[DEVIATION — narrows "where the coin was in the full book"]** A full-book leg is used only when that row's `fb_<side>_usd_5`
  (itself a per-second mean) is ≥ $6k on the side taken. Reason: the recorder's `fb_cost_*` is the mean over only the seconds where
  the size fitted within 5 % (thin seconds are dropped — a low bias).
- **`out_cost_flag` values**
  - `full_book`: every taker leg was full-book costed. A real fill's own price counts as costed, so a real fill with a full-book
    exit is `full_book`.
  - `no_book_leg`: a fill's own price plus a maker TP.
  - `lower-bound cost`: any other source. The label means "not the full book", whichever way the bias runs.
  - empty when unpriced.
- **Finality:** a taker exit at or past the recorded data end (or the full-book data end, for a full-book coin) is never final.
- **Triggers and controls keep the 20-level cost** (item 2 covers the **sleeve** NET only). `h_cost_src` takes three values:
  - `route_b_book`;
  - `no_book` — no recorded book; excluded per §3, together with that trigger's controls;
  - `book_unwalkable` — a book row existed but the $6k walk did not fit; default slip.
- **[DEVIATION — `book_unwalkable` is kept, with a robustness condition]**
  - Every Q3 pass statistic must ALSO hold with the `book_unwalkable` triggers and controls excluded; counts per key are reported.
  - A control is priced on its own minute's cost source; mismatches with its trigger are counted.

**Item 3 — labels**
- **Minutes** are counted in bars, like the outcome walk on the same path; "within the max hold" = the first `hold` bars.
- **REACH** includes the stop minute (the `path_labels` convention; REACH never decides a pass). `lbl_pass_min` is the bar count
  to the passage.
- **Decomposition:** the walk's real stop is the −Y passage for the group (so a FLATBASE funding-offset stop groups as NEVER).
  `lbl_worked` stays the exit-independent price label.
- **Finality:** the label has its own `out_lbl_done`; pricing finality is unchanged. If the label's bars never arrive, it is closed
  12 h after hold + 180 min: a still-unknown label becomes `NO_DATA`, kept in the coverage counts and excluded from the WORKED rate
  (`lbl_worked` NaN).
- **S5**
  - Uses the scout cache's `entry` and `sl`, key `pair|T` (T in whole seconds).
  - The walk starts at the first whole minute after T + 8 s and runs over the scout's hold.
  - It needs a gap-free path: PENDING until then, `NO_DATA` after 12 h.
  - Rows count from the DL 630 commit (b6f73117, 2026-11-23 14:33:21 UTC), and only where the scout cache holds an entry.

**Item 4 — BTC daily trend**
- Perpetual daily closes (the last 1,500 days), `ewm(span, adjust=False)`, read on the last day whose close ≤ T.
- An exact tie stamps 0, and B4 treats 0 as **no sign**.
- Stale (T ≥ 1 day after the last closed day's close) or unavailable → NaN.

**Item 5 — H7, variants, controls**
- **H7**
  - At trigger minute m, H7 uses the OI row stamped m − 1 (its `oi_chg_5m`) and mark(m − 1) ÷ mark(m − 6) − 1. Each mark is the
    last row ≤ that minute, at most 2 min old.
  - Thresholds are inclusive (≤ −3, ≥ +2, ≥ +5).
  - Hot coins with an OI sweep only. No book is needed **to fire**, but a `no_book` H7 opener is still out of the Q3 statistics (§3).
- **Variant depth**
  - The newest fb row ≤ m (≤ 2 min old) on the absorbing side, against the median of the 24 h of rows before **that row** (the row
    excluded, ≥ 60 rows).
  - Variant hit minutes = the parent's hit minutes where the condition holds; episodes are built on them.
  - B3b's "same coin set" = the parent openers with `cond_depth_scored` True.
- **Control-exclusion family:** the H number. Both sides of H3 / H4 / H5 / H6 go together, H7a / b / c are separate, and a variant
  goes with its parent.
- **[DEVIATION — OPEN, the operator decides before 12-10] Exclusion around every HIT minute, not only openers.**
  - The rule excludes ± 60 min (inclusive) around every minute the trigger condition held. §6's "any trigger of the same family"
    could also mean its openers.
  - Consequence at the 11-23 17:38 UTC run (counts only): **161 of 197 H6 openers have no eligible control.** A funding extreme
    lasts hours, so the whole coin-day is excluded. Every other family loses 0–12 %.
  - Those H6 openers leave the Δ statistic, and H6 reads UNTESTABLE for them.
  - The alternative is to exclude around openers only. This addendum may be amended once, before 12-10, to pick it.
- **Control minutes**
  - in the trigger **minute's** UTC day, at or after REC_START, with a 20-level book row stamped at that minute;
  - drawn without replacement within an opener and independently across openers (a minute can repeat across openers of the same
    coin-day and family);
  - no depth condition for variants;
  - a trigger with zero eligible minutes gets a stub row. It is excluded from Δ and does **not** count toward the A1-Q3 / B1
    possibility floors; stubs are counted in the report;
  - a draw is final only when every hourly file from D−1 23:00 to D+1 01:00 LOADED this run, or is permanently absent (still missing
    48 h after the hour ended — counted in `ctrl_day_missing_hours`). Hours before the recording start count as present. Q3 reads
    `ctrl_final` rows only;
  - numpy 2.4.1, pinned by a golden-draw test.

**Item 6 — S0 (Q4)**
- The per-window value = the market row at the window's last minute. The daily mean is over the windows that have a value.
- **[DEVIATION — strict completeness]** `complete` = every S0 column is scored in all 96 windows, and Q4 reads complete days only.
  One unscored window in one column drops the day (e.g. 11-20 lost to the 60-min breadth warm-up; 2 of 5 days complete at 11-23).
  Q4 still needs ≥ 10 complete days a side.
- **Outages:** a minute with no mark row from ANY coin is a recorder outage. Market liquidations there are NaN (never 0), and every
  rolling **liquidation** sum needs its full window. The BTC taker-buy share keeps its existing ≥ 7-of-15-minute rule.
- **The daily mean is descriptive:** it includes windows after a day's signals. A Q4 line acted on during a day would read the
  previous day's S0.
- **Funding mean** = the raw per-settlement rate over every coin with a mark row (BTC and ETH included; 1 h and 8 h coins mixed).
- **Premium** = the mean over coins of (mark ÷ index − 1), each from its own row.

**Item 7 — Q2**
- **The p90 / p99 sample** = the coin's w-min liquidation sums at each of its liquidation minutes in [s − 7 d, s), non-zero only,
  ≥ 20 values. The 7 days are truncated at the recording start early in DESIGN. numpy linear quantile.
- **Firing:** a line fires when the current w-min sum is > 0 and ≥ the quantile.
- **[DEVIATION] Outage windows:** a window (s − w, s] that overlaps a recorder outage is unscored. The 7-day sample keeps every
  window as recorded.
- **Prior median:** the fb rows in [row − 24 h, row) of the row the state read, ≥ 60 non-NaN values.
- **[DEVIATION — strict three-valued logic]** Every AND / OR line (S1b, S2a, S2c, S5) is scored only when every leg is scored.
  - S5 is therefore scored only when there is an OI sweep (hot coin) **and** a scorable 15-min liquidation p90 (≥ 20 non-zero sums
    in 7 d, window clear of outages).
  - This narrows the DL 630 cohort (11-23 17:38: 93 EXH_NO_SPOT rows, 48 with OI, 41 with S5 scored). The coverage is reported.
- **S2-c cuts:** the DIST Q2-cohort DESIGN rows (the optional-stopping guard applied) where **both** variables are scored. pandas
  linear quantile at 1/3, inclusive ≤. Funding = `rb_funding_pct` (the current rate at the state minute).
- The `_lb` flags are carried for the audit only and do not change any line.
- **Which statistic judges what:** the rotation null (§6) is the family-maximum null of the **Q1** screen. A **Q2** line is judged by
  A2's window-clustered bootstrap, as A2 states.

**Schedule (§7)**
- 2026-12-10 is a **Thursday**, not a Wednesday. The date governs: the first run is Thursday 2026-12-10, weekly on Thursdays after.

### Addendum 2 — presentation: verdict colours and summary-only tables (written 2026-11-23 17:55 UTC, before any scan outcome was read)

Display only: no question, cohort, label, threshold, null or gate changes. It carries the colours agreed on 2026-11-20 (DECISION_LOG
598: "the weekly scan will use 🟢 passed every locked gate · 🟡 below the bar (watch line) · 🔴 refuted · ⚪ too few") into §6 D's statuses,
which were frozen without them.

| Status (§6 D) | Colour | Meaning in the report |
|---|---|---|
| PASSED | 🟢 | passed every locked gate on HOLDOUT |
| SURVIVED DESIGN | 🟡 | a watch line, waiting on HOLDOUT |
| STALE | 🟡 | waiting on its one re-check after an exit change |
| FAILED · CONTRADICTED | 🔴 | refuted |
| COLLECTING · UNTESTABLE | ⚪ | too few rows / cannot be tested |

- Every status-table row carries its colour in the first column, before the status word. A colour never replaces the status word.
- Coverage rows carry the data-quality marks of 2026-11-20 (🟢 ≥ 90 % of the fills with the book recorded at entry · 🟡 50–90 % or the
  recorder degraded · 🔴 < 50 % or the recorder down · ⚪ none; the worst mark wins; "degraded" / "down" per day as DL 598 defines
  them). They describe the data, never a result.
- A PASSED line labelled "one regime" (§6 B4) keeps 🟢 with the label beside it, and stays capped at OBSERVE.
- Operator 2026-11-23: the weekly report shows **summaries only** — per sleeve / key / line aggregates (N, medians, effects, CIs). It
  never lists single trades or signals. A pass still gets its event-study chart (§7), as an aggregate path.

### Addendum 3 — H6 control exclusion: Addendum 1's open choice closed, the ORIGINAL rule stays (written 2026-11-23 18:18 UTC, before any scan outcome was read)

**Operator decision (2026-11-23):** "keep original rule". This closes the one open choice Addendum 1 left: the openers-only option is
withdrawn, and no further change to the exclusion is possible. No Q3 outcome had been read; only coverage counts were.

- **The rule stays as Addendum 1 wrote it:** H6, like every family, excludes ± 60 min (inclusive) around every hit minute.
- **Correction to Addendum 1's diagnosis.** Most H6 openers lacked a control because their coin had **no order book recorded**, not
  because of the exclusion. Funding-extreme coins are mostly not on the recorder's hot list. Counts at the 11-23 18:00 UTC data
  (3.78 days):
  - 190 H6 base openers (plus 10 "|c" variant rows);
  - 161 of the 190 had no recorded book at the trigger minute, so they are out of Q3 under §3 whatever the rule;
  - 29 had a book: 6 of them have no eligible control under this rule (median eligible minutes 531). That leaves **23 testable, about
    6 a day**.
- **Disclosure:** the openers-only alternative was built and run once (outcome columns computed, never printed), then reverted. Its
  control draws are recomputed at the next build (the carry is keyed by the drawn minute).
- **What this means for H6:** it reaches the §6 floors (A1-Q3 ≥ 20, B1 ≥ 30 on ≥ 10 days) slowly. More book coverage of
  funding-extreme coins would be a recorder change, outside this pre-registration.
- **Stale labels on disk:** one interim build wrote `h_cost_src = default`, a pre-split label. CODE_VERSION 7 re-prices every row, so
  `h_cost_src` takes only the three Addendum-1 values (`route_b_book` / `book_unwalkable` / `no_book`), and a test pins them.
- Addendum 1's **[DEVIATION — OPEN]** item is closed by this addendum. Nothing else changes.

### Addendum 4 — priority order for reporting and follow-up (written 2026-11-23 18:18 UTC, before any scan outcome was read)

**Operator (2026-11-23):** "prioritize the ones that have more trades per day". Order only: every key stays in the Q3 family and the
luck test, and no gate, null or floor changes. Dropping the low-volume keys would lower the bar for the rest, so none is dropped.

- **The order:** per family, the Q3-eligible base openers per day over DESIGN (a book at the trigger minute and at least one final
  control), counted once at the 12-10 run and frozen. This uses no outcome. Variants and H7a / b / c are listed under their parent
  family.
- **Raw possibilities per day on 11-23, for orientation only** (the order is fixed by the eligible counts at 12-10): H5 178 · H4 128 ·
  H3 66 · H6 50 (≈ 6 eligible) · H1 23 · H2 18 · H7 7.
- **The weekly report** lists the Q3 keys in that order, each with its separate-window count beside it (the statistical unit).
- **Follow-up work** for a key that passes runs in that order: the winners-vs-losers study under its frozen cell (the standing
  entry × exit rule; its own pre-registration), then capacity, then the sleeve design and scout. This is an addition to §6 C's steps,
  not a gate.
- **Fresh-data-only keys:** H3 and H5 can only pass on HOLDOUT, so their place in the order applies from their first HOLDOUT verdict.

### Addendum 5 — today's filter stack on every signal (written 2026-11-23 22:05 UTC, before any scan outcome was read)

**Operator (2026-11-23):** Route B must judge every signal under today's filters, not only today's exits. Two filters were armed after
the recording began, so the journal decisions before them never ran them: FLATBASE_FUNDING_ABOVE_BASE (DL 601, committed 11-20 17:27
UTC) and WILLY_HT_BURST_REJECTION (DL 619, committed 11-22 00:09 UTC).
- **FLATBASE:** no signal reached the funding check before it went live. The 4 earlier FB signals were 24 h-volume blocks, and the
  one FB fill came after. On today's data the FB re-check is forward safety only.
- **HT:** all 8 HT fills came before DL 619, and they are the rows re-checked. 9 earlier HT signals were already blocked by another
  filter and never reached the burst check.

- **New columns** (`scripts/route_b_signals/today_stack.py`, DL 635): `today_recheck` · `today_recheck_state` (zone / rest /
  unread) · `today_recheck_value` · `today_block` · `today_block_class` · `today_filters_pass`. Each filter is re-applied with the
  engine's own function to the rows that reached it before its commit + 60 min. After that the engine's own decision stands.
  - Input unreadable to the engine (a missing journal funding stamp; Binance returning no bars): the engine fails open, so the row
    passes (flagged).
  - Input not fetched by a build (Binance stopped, network): `nofetch`, so `today_filters_pass` is NA. Such rows are left out of every
    today's-stack read, counted, and re-tried at the next build.
  - Empty strings in these columns read back as NaN from the CSV; readers use `.fillna("")`.
- **How the scan reads them:**
  - **Q1 / Q2 cohorts: unchanged.** §2 already includes taken and filter-blocked rows. A row today's stack refuses is a filter-blocked
    row, so it stays in.
  - **Every "what today's bot trades" read** (taken vs blocked, before → after daily compound return, per-sleeve kept cohort, Q5
    capacity of kept fills) uses `today_filters_pass` / `today_block_class` in place of the journal `status` / `block_class`.
  - A blocked-filter breakdown names `today_block` where it is set.
- **Guard:** the test suite pins every ENTRY setting of the Route B sleeves, switches and cuts. That includes the WILLY watch blocks
  and the shared funding guard. Exit, size and sleeve on / off keys are excluded because exits are read live. A changed entry setting
  in `trading_config.json` fails the suite until a re-check decision is made.
- **Not re-checked, by design:** DL 629 (committed 2026-11-23 01:39 UTC) is a timing bug fix for WILLY's turnover read. It delayed a decision; it is
  not a filter, and the delayed setups were journaled again when re-judged.
- No question, cohort, label, threshold or gate above changes.
