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

Order of work: 0a learnings → validate_against_master.py → run Scout on the new export → batch watchlist check (every open gate in
CLAUDE_CURRENT_STATE + every Scout tracker) → 0b Scout line by line → the analyses below → 0c bug gate on every finding → recommendations.

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
- [ ] Winners vs losers per sleeve on every stamped entry column (sweep_separators-style, exhaustive 2D + null + OOS rules).
- [ ] Every FRENZY watchlist / Scout line: GVOL_SPLIT, HOLD_FRENZY_EXEMPT, FRENZY_REENTRY_AFTER_WIN, FRENZY_STRETCHED, LITE_OFF30H3,
      LITE_GVOL24_LOW, FRENZY_STRONG revert gate (197 — Scout-only, scripts/scout_revert_gates.py, not in CURRENT_STATE), ATR cap 3.0 (250), bearish block (250), GREEN_CLOCK, WILLY_TURNOVER_BLOCKED, WILLY_HOLD.

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
