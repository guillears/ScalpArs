# Next batch review — operator plan (registered 2026-10-10)

Run all of this when the operator sends the next batch export (`scalpars_orders_paper_*.csv` + decisions).
Goal stated by the operator: make the bot highly profitable going forward (≈ 10 %/day ambition) — FRENZY and SPIKE_FADE are the
high-potential sleeves; MOMENTUM LONG must become profitable now.

**Every analysis shows before → after in three blocks: ① current batch · ② master batch, per batch · ③ backtest (yr5, today's stack,
one seed, eligible pairs, de-duplicated — see memory feedback_backtest_ref_today_stack.md). Columns: N · WR · net $ (and avg %/trade).
Then a clear recommendation (+ pre-committed revert gate if a change is proposed).**

## 0. Method (operator 2026-10-10: methodical, by sleeve; learnings first; NO recommendation built on a bug)
**0a. Learnings before any recommendation.** Per sleeve, read everything we already know before analysing: CLAUDE_CURRENT_STATE (every
gate / line of that sleeve), the DECISION_LOG entries for that sleeve (operator authorised reading the history for this review),
the sleeve's study reports in reports/, and the relevant memories (refuted lines, cohort floors, known pitfalls). Write a short
"what we already know / already tried / refuted" box per sleeve so nothing refuted is re-proposed.

**0a+. FRENZY theory first (operator 2026-10-11) — BEFORE any batch analysis.** For FRENZY_LONG, FRENZY_WIDE, FRENZY_LITE and
FRENZY_WILLY: what market behaviour each one exploits (who is buying / selling at the entry, why price should continue or revert,
what the spike / VWAP / normal-hour / red-candle / streak / stretch conditions are meant to capture), from services/frenzy.py,
the design reports (reports/FRENZY_SLEEVE_DESIGN.md and the LITE / WIDE / WILLY studies) and DECISION_LOG. Then: how the four relate
— is any one just a small derivative of the core FRENZY_LONG setup (same bars, same pairs, correlated outcomes), measured by signal
overlap and outcome correlation, not asserted. The theory defines which variables SHOULD separate winners from losers (hypotheses
first, then data): write each sleeve's predicted separators (variable + expected sign) and FREEZE them before the 0c screen runs;
data-mined separators the theory did not predict need out-of-sample confirmation before they count. Principle (operator): if a
sleeve's theory is robust it is NOT disabled — the work is finding its correct entries. "Robust" is judged by evidence (theory-
predicted separators confirmed out-of-sample), not asserted; a sleeve switch-off still follows the SLEEVE-KILL CHECKLIST
(sweep_separators.py output cited) — the principle orders the work (entries first), it does not remove the checklist.

**0b. Every Scout line, by sleeve, each with a recommendation.** For every Scout section: what it tests · progress vs its frozen bar ·
data health (parity, unscored / provisional fills, Unavailable sections) · verdict now · recommendation (keep observing / act / retire /
fix the line).

**0c. Bug gate — mandatory before ANY recommendation.**
- Run validate_against_master.py (also step 2 of the order below); read the Scout run log for crashes / "Unavailable".
- State each cohort's definition explicitly (gates on/off as in trading_config.json, bearish block, eligibility, slots, seeds, dedup on
  (opened_at, pair, direction) never id, manual / probe fills out, era floors).
- Replica parity: any walker / replay must reproduce the live fills (reason + %) before its counterfactual is trusted.
- Every headline number derived TWO independent ways (two scripts or script + spreadsheet-style recount); mismatch = stop and find out why.
- Statistics: day/window-clustered bootstrap, both halves, leave-one-month-out, shuffled-label null for any separator, top-pair / top-day
  concentration, 30–50 % in-sample haircut; screens list what they could NOT test.
- An independent reviewer agent re-derives every number behind a recommendation before it is shown (the dual-review idea applied to
  analysis, not only code).
- **Filters:** a block / filter proposal must pass the locked EXPECTANCY bar — WR below the sleeve's breakeven WR at 1×, avg < 0 with
  ≥ 95 % confidence (window-clustered bootstrap), ≥ 8 windows, no window / pair ≥ 50 % of the loss, N ≥ 15; below the bar it ships
  OBSERVE-ONLY (observe-first rule). Frozen Scout bars are never re-fit at decision time.
- **Pools:** flip / short / sleeve cross-batch stats use reports/SCREENED_BASELINE.csv (re-run scripts/screen_pool.py on the new
  batch first), never the raw pool; a rule that extends an existing gate adds back the fills that gate already screened out
  (cohort completeness).
- **Exit / stop counterfactuals:** read on the live-stopped cohort, split saved vs deeper, fills live did not stop must show Δ = 0,
  two-sided, per era.
- **One change at a time** per sleeve (clean attribution), each with a pre-committed revert gate.
- **Winners vs losers = 100 % of the available variables** (operator 2026-10-11): every entry_* / stamped column on the fills (and the
  decision journal where a fill lacks it), 1D at every granularity, exhaustive 2D, and 3D ONLY for triples pre-registered from the
  0a+ theory or a 2D near-miss (list and count them BEFORE running; never an exhaustive 3D sweep — ~117 entry columns ≈ 6,800 pairs /
  ≈ 260,000 triples per granularity). Every screen reports how many tests it ran and uses a max-statistic shuffled-label null over ALL
  tests run (or BH-FDR), then out-of-sample confirmation on the half not used to find it; survivors are a SCREEN, not a ship list
  (locked gates + theory + haircut still apply) (memory feedback_exhaustive_2d_screen.md). The UI table
  "Entry Conditions by Strategy" shows only a subset — never treat it as the full screen. Every screen lists the columns it could NOT
  score (coverage first).

Order of work: 0a learnings → 0a+ FRENZY theory (predicted separators frozen) → validate_against_master.py → run Scout on the new
export → batch watchlist check (every open gate in CLAUDE_CURRENT_STATE + every Scout tracker) → 0b Scout line by line → §1–§6
analyses (§3b only after §3) → §7 sizing last → 0c bug gate on every finding → recommendations.

## 1. FRENZY_WILLY
- Reference numbers below come from the yr5 A 1,067-fill cohort (pre-270 recipe): re-derive them on today's stack, one seed,
  eligible pairs, de-duplicated before quoting. Four pre-registered tests share the same WILLY fills (TIMECAP, TP125, SL4, combo) —
  read any single candidate with that in mind; ship at most one WILLY exit change per review.
- [ ] TP: +1 vs +1.25 (Scout WILLY_TP125; yr5 ref +0.135 vs +0.191).
- [ ] Time cap: is 120 min right or can it be shorter? Use the hold-time distribution of winners vs losers (Scout WILLY_TIMECAP; yr5
      CAP15/20 lose).
- [ ] Stop besides the time cap: needed? which value? (Scout WILLY_STOP −4 + context −2/−3/−5/−8; combo +1.25/−4 test; yr5: every stop
      costs vs none, −4 ≈ −0.05 %/fill as insurance; +1.25/−4 ≈ today's expectancy with the tail capped).

## 2. FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE
- Corrected yr5 refs (DL 270, today's stack): +3/−3 LONG 320 · 49 % · −0.039 %/fill · WIDE 204 · 49 % · −0.047 · LITE 546 · 54 % ·
  +0.267 (≈ +0.23 with slots / caps); no grid cell beats +3/−3 significantly in any sleeve. Live re-priced on +3/−3: LONG 15 · 20 % ·
  −1.81 · WIDE 8 · 38 % · −0.76 · LITE 12 · 58 % · +0.50. Never quote the retracted +0.084 / +0.088 / +0.257.
- [ ] TP grid as WILLY (+1 / +1.25 × −2 / −2.5 / −3, +1.25 / −4) → Scout FRENZY_WILLY_TP (7 cells, per sleeve, 12 h kept).
- [ ] Duration (12 h cap vs shorter) → NEW analysis, no Scout line yet: hold-time distribution of winners vs losers on live fills +
      the yr5 walk.
- [ ] Average trough of winners per sleeve → narrow the −3 stop, keep it, or widen it (live trough_pnl column + yr5 walk). Refs:
      yr5 reach-+3 winners LONG −2.82 / median −1.94 · WIDE −3.14 / −1.78 · LITE −2.65 / −1.75 (pre-270 cohort — re-derive); live
      winners LONG −0.77 · WIDE −0.80 · LITE −1.47.

## 3. FRENZY entry strategy (all FRENZY sleeves incl. WILLY)
- [ ] Winners vs losers per sleeve — the §0c 100 %-of-variables screen (1D / 2D / pre-registered 3D, null + OOS).
- [ ] Every FRENZY watchlist / Scout line: GVOL_SPLIT, HOLD_FRENZY_EXEMPT, FRENZY_REENTRY_AFTER_WIN, FRENZY_STRETCHED, LITE_OFF30H3,
      LITE_GVOL24_LOW, FRENZY_STRONG revert gate (197 — Scout-only, scripts/scout_revert_gates.py, not in CURRENT_STATE), ATR cap 3.0 (250), bearish block (250), GREEN_CLOCK, WILLY_TURNOVER_BLOCKED, WILLY_HOLD.

- [ ] **+DI / −DI** (operator): entry_frenzy_di_spread (+DI − −DI, last CLOSED 5m bar) and entry_frenzy_adx_delta (ADX now − 3 bars
      earlier) — NOT entry_pos_di / entry_neg_di / entry_adx_delta, which read the forming bar. FRENZY_STRONG is not a column: rebuild
      it as FRENZY_LONG ∧ adx_delta > 0 ∧ di_spread > 0. DI is already partly used (FRENZY_LONG strong fills at lev 0.5 vs 0.32, revert
      gate 197 in Scout): check the cell registry first, compare in Avg P&L % at 1× (the strong fills carry other leverage), winners vs
      losers per sleeve.
- [ ] **Entry timing — first vs second red candle** (operator). As coded: FRENZY_LONG and FRENZY_LITE do NOT wait — they judge ONE
      signal bar (FRENZY: the fresh ON bar; LITE: the first signal bar of each above-VWAP stretch) and refuse it if it is green (red =
      close ≤ open, flat counts as red); a refused setup is never retried (LITE marks the stretch judged). FRENZY_WIDE is the GREEN-candle
      variant (it takes only FRENZY's green-bar refusals, and with hold_green_streak 12 only when > 12 closes in a row were above VWAP;
      the ATR-high refusals are blocked). Only FRENZY_WILLY waits: it opens on the first STRICTLY red 5m bar (close < open, flat is not
      red; the trigger bar counts) within 60 min of the trigger (entry_frenzy_willy_wait_bars stores the wait). Test per sleeve: re-price
      every fill as if it had entered on the NEXT red (WILLY: second red) bar within a pre-registered max wait (60 min, WILLY's own), same
      exit. Report three cohorts: kept (entry moved) · skipped (no second red within the wait → removed) · added (setups refused today
      that would fire under the new rule, e.g. FRENZY / LITE green signal bars followed by a red bar; slots / pair-day cap freed by the
      skipped ones). The replica must reproduce today's live entry bar and price for the fills (parity) before the counterfactual is
      trusted; read in DAY units, two-sided, both halves. Prior evidence to beat: entering on a later ON bar is refuted (−0.49 %/trade,
      FRENZY_STAIRCASE_STUDY_2026-10-06).

## 3b. FRENZY losers → FRENZY SHORT opportunity (after §3 finds a loser signature)
- [ ] Operator 2026-10-11: if §3 isolates a robust LOSER signature on FRENZY_LONG / WIDE / LITE, test SHORTING that signature with a
      fixed TP +1 % or +1.2 % net and a −3 % stop (12 h cap). Mandatory guards: (i) the signature is DISCOVERED and frozen on one half
      (by date) only — the §3 screen for this purpose runs on that half alone; the other half and forward fills are never looked at
      before the freeze. (ii) The short cohort = EVERY fire that matches the signature (winners and losers of the long, plus setups not
      taken for slots / caps), not the long's losers. (iii) Null: the same short (same TP / SL / 12 h cap) on all FRENZY-flagged bars
      without the signature, and on shuffled signatures — the signature must beat the generic pump-fade short. (iv) Breakeven WR: TP +1 /
      SL −3 ≈ 75 %, TP +1.2 ≈ 71 %, plus fees and the FRENZY live ruler (8 s + ≥ 0.10 % slip); pick ONE TP before the out-of-sample read
      (two TPs = two tests). (v) Expectancy bar (window-clustered bootstrap, ≥ 8 windows, N ≥ 15) and the 30–50 % haircut; below it =
      observe-only. Before → after in the three blocks.

## 4. MOMENTUM LONG — make it profitable
- [ ] Entry strategy in detail vs CURRENT_STATE + Scout watchlist (ML_B1H_NEGFLANK, NEG_DAILYUP_WEAKPAIR, PAIR_1H_DOWNTREND,
      TREND_ALIGNED, heat lines (heat block × BTC regime, heat re-scope / revert gates), LOADX, LONG_CHOP_BURST, mega-cap, ML cooldown): winners vs losers in current batch, master (per batch) and
      backtest. Start from the MOMENTUM-LONG ANALYSIS PIN in CURRENT_STATE; per DL 204 the master-kept ML figure is in-sample —
      judge ML on as-traded / forward fills. Filters first, not sizing (memory feedback_filters_not_sizing.md). Propose concrete rules with before/after + revert gate.

## 5. Every other sleeve (SPIKE_FADE priority, flips, momentum short, bull-run, bear-run, surge)
- [ ] Full read of everything in CLAUDE_CURRENT_STATE + Scout; any finding → before/after in the three blocks + recommendation.
- [ ] SPIKE_FADE: high potential and strong in every batch — look for scale-up / more-fills levers as well as risk.

## 6. Added by Claude (portfolio-level levers toward the profit target)
- [ ] **Missed moves:** Scout's missed-move diagnosis (FILTER_NEAR gate sets with more good than bad pair-days, FILTER_FAR / SLEEVE
      clusters, shortlist misses like MOVR / NMR) → loosen-a-gate or new-sleeve candidates, engine-replay tested.
- [ ] **WILLY global hold cost:** WILLY_HOLD + HOLD_FRENZY_EXEMPT — what the hold blocks vs what WILLY earns.
- [ ] **Execution realism:** STOP_SLIP (Scout-only, scripts/opportunity_scout.py), entry latency / slippage per sleeve (live vs paper fills) — the gap to real money.
- [ ] **Concurrency and ruin:** a today's-stack portfolio run of all sleeves together (daily compound, max drawdown, worst day, open
      exposure) — the profit target only counts if the book survives it.
- [ ] **Concentration:** per-pair and per-day share of each sleeve's P&L; blacklist candidates (pair-level beats dimension filters).
- [ ] **Time / regime splits:** hour-of-day, weekday, BTC regime per sleeve, read in DAY units (window-units rule).

## 7. Sizing — LAST, after the filters (operator 2026-10-11)
Run only after §1–§5 have set the entries / exits, on the FILTERED cohorts, with the §6 concurrency / ruin run as the binding check.
Sizing is judged on the FILTERED cohort with the 30–50 % in-sample haircut applied to the filters' Δ (the filter and the size-up were
fitted on the same fills). A size-up is a multiplier: it must pass the Pattern W gate (N ≥ 30, WR ≥ 70 %, Avg P&L % ≥ +0.10 %, Total $
> 0) and be STAGED (e.g. 0.32 → 0.5 → 1.0, next step after +50 fills), with a pre-committed revert gate. Avg P&L % does not change with
size — compare $, worst day and max drawdown; the ruin line uses −3 % × full leverage × max concurrent FRENZY slots (2 per sleeve, 7 total)
at once.
- [ ] **FRENZY_LONG / WIDE / LITE at 1× invest × 1× leverage, like WILLY** (today: LONG lev 0.32 (strong 0.5), WIDE 0.2, LITE 0.32,
      invest 1.0; WILLY invest 1.0 × lev 1.0). Before → after $ in the three blocks, worst day / max drawdown with all sleeves together,
      and the ruin line (a −3 stop at full leverage vs the book). Recommend per sleeve, only where the filtered sleeve has a proven edge.
- [ ] **MOMENTUM LONG invest down to 1×** (operator: frequency used to need the bigger ticket; FRENZY now supplies volume). ML today:
      base invest 1.0 (lev 20) · UNMATCHED LONG cell inv 1.5 · BTC RSI×ADX long cells 60-65/28-30 inv 2.0 and 60-65/22-25 inv 2.0 × lev
      1.5 · long_unmatched_quiet 1.5 (no-op, take-the-max) · caps 2.5 invest / 2.0 lev (re-read trading_config.json at review time). Per
      cell: its promotion / revert history (cell registry) and its multiplier-cell verdict (★ WORKING / ⚠ DRAG / ✗ HARMFUL on ≥ 5 fresh
      fires) — a WORKING cell is not cut to 1× without evidence. $ and drawdown before → after, and the freed margin / slots for the other
      sleeves. Only after §4's filter work; a size cut is never the headline fix for ML's profitability (memory feedback_filters_not_sizing)
      — it is judged per cell on the freed margin / slots and the concurrency run.

