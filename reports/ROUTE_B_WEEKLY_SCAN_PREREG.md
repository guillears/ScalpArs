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
*(none — any change goes here, timestamped, before the data it touches is read)*
