# Next batch review — operator plan (registered 2026-10-10)

Run all of this when the operator sends the next batch export (`scalpars_orders_paper_*.csv` + decisions).
Goal stated by the operator: make the bot highly profitable going forward (≈ 10 %/day ambition) — FRENZY and SPIKE_FADE are the
high-potential sleeves; MOMENTUM LONG must become profitable now.

**Every analysis shows before → after in three blocks: ① current batch · ② master batch, per batch · ③ backtest (yr5, today's stack,
one seed, eligible pairs, de-duplicated — see memory feedback_backtest_ref_today_stack.md). Columns: N · WR · net $ (and avg %/trade).
Then a clear recommendation (+ pre-committed revert gate if a change is proposed).**

**🔒 THREE BLOCKS IN EVERY TABLE, EVERY SLEEVE (operator 2026-10-12, corrected after a year-only filter table): every filter, watchlist
line, exit variant and sizing test is shown as ① current batch · ② master per batch · ③ backtest — for the trades it would BLOCK and
the trades it KEEPS, before → after (N · WR · avg % · $). Never a "Year:" column alone. When a block cannot be scored (e.g. a live
stamp missing), say how many fills are unscored — never drop the block. This applies to ALL sleeves (WILLY, FRENZY LONG / WIDE / LITE,
MOMENTUM LONG / SHORT, flips, SPIKE_FADE, bull-run, bear-run, surge).**

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

## B19 carry-forward — re-read on the next export (the same batch B19 running longer, more trades) — registered 2026-10-12
Context: B19 review (reports/B19_RERUN_*.md, verified) + DECISION_LOG 271 (paper = live stops; FRENZY −3 / WILLY −4 in both modes,
per-order exchange backstop). The next export is B19 extended: re-run every number below on ALL B19 fills (old + new), and show the
new fills (opened after the 271 deploy, 2026-10-12) as their own line — they are the first trades priced with the same stop live and paper.

**WILLY — recommended exit and leverage (Claude's B19 recommendation; check it still holds with more trades):**
- **Exit: TP +1 %, SL −4 %, 120-min cap** (live in paper and live since 271). +1.25 is still unproven (Scout WILLY_TP125 / combo
  +1.25/−4 frozen bars not met). FRENZY LONG / WIDE / LITE stay at +3 / −3 / 12 h.
  - −4 evidence: on the live-reachable yr5 cohort (532 fills) −4 beats real-money no-stop by +0.076 %/fill (P 0.995, both halves, every
    leave-one-month-out); B19 under −4 = 14 fills +$659 (only BAT stopped).
- **Why 10× beats 20× (operator decision pending — WILLY still runs 20×, `frenzy_willy_lev_mult` 1.0; 10× = 0.5):**
  - at 20× one −4 stop costs ≈ 19.5 % of the book → a few in a row erase the gains even with a positive average (compounding);
  - yr5 year book: 10× ≈ $9,035, the best result under every position-size cap ($10.3–11.8k) vs 20× ≈ $2,356 uncapped (≈ $9,100 only
    with a $10k cap);
  - at 10× the stop is safe from liquidation for any maintenance-margin rate up to 2.5 %; at 20× liquidation on small caps can sit
    inside −4 %, and the bot-stop vs exchange-backstop gap is only ≈ 0.23 % (< the 0.3 % rule — CURRENT_STATE go-live check ⑥);
  - growth-optimal (Kelly) leverage ≈ 6×.
- [ ] Re-check on the extended batch: WILLY fills under TP +1 / −4 / 120 (N · WR · avg % · $) at 20× vs 10× (and 18×); count stops
      (each −4 at 20× ≈ −19.5 % of the book); are the B19 conclusions (−4 > no stop, +1 ≥ +1.25, 10× > 20×) unchanged?
- [ ] New-fill parity: every post-271 WILLY / FRENZY stop closes at its own line (−4 / −3) in paper exactly as live would — any fill
      closing at −2.2 or at a backstop = bug.
- [ ] **WILLY's exit on ALL FRENZY trades** (operator 2026-10-12): FRENZY LONG / WIDE / LITE priced with WILLY's exit (+1 / −4, also
      +1 / −2, +1 / −3, +1.25 / −4; 12 h cap) vs today's +3 / −3, each WITH and WITHOUT the market-volume filter (GVOL ≥ 1.0 blocked),
      in all three blocks (① B19 extended · ② master per batch · ③ backtest). Script: scripts/study_b19_fz_gvol_willyexit.py (re-point
      EXPORT at the new file). B19 first read (reports/B19_FZ_GVOL_WILLYEXIT.txt): LONG +3/−3 → +1/−4: ① 11 · −1.37 → +0.09 · ② 9 · −1.67
      → −1.22 · ③ 320 · −0.04 → +0.01 (P 0.63); LONG GVOL<1 with +3/−3: ① 7 · −1.29 · ② 8 · −1.50 · ③ 170 · +0.21. WIDE +1/−4: ① +1.00 ·
      ② −0.66 · ③ −0.22 (P 0.17). LITE +1/−4: ① −1.28 · ② +1.01 · ③ −0.04 vs +0.27 (P 0.00 — LITE needs its +3).
- [ ] FRENZY carry-forward (B19 §2–§3 verdict "no change"): re-read GVOL ≥ 1.0 LONG (closest filter, P 0.88 on the year — frozen
      GVOL_SPLIT bar), LITE −4 stop near-miss (P 0.93), LONG 30-min cap near-miss (P 0.88), the bearish-day block's LITE cost (blocked
      LITE +0.19 on the year), and the LITE "EMA13/50 gap > 1.87 ∧ not first entry" borderline zone (if the operator registers it).

## 1. FRENZY_WILLY
- Reference numbers below come from the yr5 A 1,067-fill cohort (pre-270 recipe): re-derive them on today's stack, one seed,
  eligible pairs, de-duplicated before quoting. The WILLY exit FAMILY on the same fills: TIMECAP, TP125, SL4, combo +1.25/−4, the
  6 EMA-cross variants and the runner leg below — every candidate faces a max-statistic shuffled null over the whole family (state
  the count); ship at most one WILLY exit change per review.
- [ ] TP: +1 vs +1.25 (Scout WILLY_TP125; yr5 ref +0.135 vs +0.191).
- [ ] Time cap: is 120 min right or can it be shorter? Use the hold-time distribution of winners vs losers (Scout WILLY_TIMECAP; yr5
      CAP15/20 lose).
- [ ] Stop besides the time cap: needed? which value? (Scout WILLY_STOP −4 + context −2/−3/−5/−8; combo +1.25/−4 test; yr5: every stop
      costs vs none, −4 ≈ −0.05 %/fill as insurance; +1.25/−4 ≈ today's expectancy with the tail capped).

- [ ] **Price vs EMA13 / EMA20 / EMA50 at entry and at exit** (operator 2026-10-11): the % distance of price from each EMA at WILLY's
      entry and at its exit, winners vs losers (loser = pnl < 0 whatever the close reason, incl. FRENZY_TP_LATE) (5m CLOSED bars only —
      the engine's ta EMAIndicator = ewm(span, adjust=False, min_periods=n), seeded on ≥ 300 bars of history; the engine's own window can
      include the forming bar, so check parity against the stamped entry_dist_from_ema13_pct on every live WILLY fill before use). EMA20 /
      EMA50 distances are not stamped: entry_pair_ema20_ema50_gap_pct is EMA13 − EMA50 (misnamed) — never use it as an EMA20 read.
      exit_price_vs_ema5_pct looks corrupt on WILLY rows (+24 … +37 % on +1 % exits) — rebuild exit distances from klines and flag the
      stamp as a bug (§0c). Only the ENTRY distance is a separator (§0c screen, predicted in 0a+); the exit distance is outcome-
      conditioned and descriptive only (input for the exit test below).
- [ ] **EMA-cross exit for WILLY** (operator): exit on the first 5m close below EMA13 / EMA20 / EMA50 (three pre-registered variants,
      each with and without today's +1 TP kept as a cap), the 120-min cap kept as the backstop, vs today's +1 / no stop / 120 min — same
      entries, same path (ticks / 1m walker, the bot's accounting, live latency). Pre-registered: the cross counts only from the first 5m
      close after entry (an entry already below the EMA — LPT / CAP were below EMA13 — fires there; report those fills as their own line).
      The no-TP variants are a pure EMA cross with no stop at 20×: model liquidation (≈ −4.5 % adverse; 14 % of yr5 WILLY fills dip there)
      or the live backstop, never paper-infinite. Learnings first: the Scout FRENZY / WIDE shadows (scout_frenzy_exits.py EMA20 / EMA50)
      arm only after a +3 peak (−3 stop until then, then a +2 floor) — they never act on a +1-TP sleeve and have no EMA13, so they are not
      evidence for WILLY; the relevant result is reports/FRENZY_ATR_AND_PURE_EMA_EXIT_2026-10-05.md Q2 (a pure EMA20 / EMA50 cross with no
      stop and no floor was clearly worse than the lock, with liquidation tails). 3 EMAs × 2 = 6 variants → best-of-6 guard (day-bootstrap
      P ≥ 0.95) AND the family max-statistic null (§1 intro). Two-sided: winners cut before +1 AND cap losers saved; fills the EMA never
      touches must show Δ = 0 (else the walker is lying). Parity first: reuse the WILLY_TIMECAP walker (it reproduces the live exits).
- [ ] **What WILLY leaves after its +1** (Claude; a WILLY EXIT test — moved here from §3): over ALL WILLY fills (live + yr5 A 1,067,
      today's stack), the distribution of the move from the TP exit to 1 / 2 / 4 / 12 h later (max favourable AND max adverse), not chosen
      examples. Live so far: KAIA +12.5 at 60 min / +18.5 at 120 min after its TP (its "+62.7 %" in the missed-moves table includes the
      pre-entry run; MAGIC's +50 % had no WILLY fill); LPT / MINA / W were −7.8 / −5.2 / −4.0 at 120 min. A runner leg counts in the WILLY
      exit family: freeze its rule (size split, its stop or liquidation model at 20×, its cap) before the read, find on H1, confirm on H2;
      price the longer global hold too (the runner keeps every other sleeve blocked; HOLD_FRENZY_EXEMPT / WILLY_HOLD).

## 2. FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE
- [ ] **LITE × bearish-day block (operator 2026-10-12 — review in LITE):** the only Scout line is FRENZY_BEARISH_BLOCKED (the DL 250 revert
      gate, scripts/scout_frenzy_exits.py, store reports/SCOUT_FRENZY_BEARISH_BLOCKED.csv): every FRENZY / WIDE / LITE bearish-day refusal
      priced as if opened (+3 / −3 / 12 h, ticks once published), frozen at the first 15 counted blocked signals on ≥ 8 days, blocked mean > 0
      → block off — POOLED over the three sleeves, NO LITE-only read. B19 tally 3 / 15 counted · 1 / 8 days · mean +0.90 %: LITE MAGIC 10-09
      14:15 +2.91 (counted) · LITE BAT 10-09 19:40 −3.11 (same episode, not counted) · WIDE BAT 17:40 −3.11 · WIDE KAIA 21:30 +2.91. Year
      (backtest): blocked LITE 176 · +0.19 % (the block costs LITE ≈ +$1.4k / year; 722 · +0.25 without it vs 546 · +0.27 with it) while
      blocked LONG −0.15 / WIDE −0.04. Review: split the line per sleeve in all three blocks (① batch · ② master per batch · ③ backtest), read
      LITE's blocked side on its own, and bring the "remove the block for LITE only" question back to the operator (a LITE-only Scout sub-read
      with the same frozen bar — added to Scout 2026-10-12, DECISION_LOG 272: read its own tally).
- [ ] **Small / fast TP on LONG & WIDE — B19 "bug?" check (2026-10-12, operator):** no backtest bug — the backtest's own logic on the B19
      days reproduces live (LONG 10/11 matched, 9 same outcome under +3/−3 and +1/−2; WIDE 6/6), and the small TP wins there too (LONG +0.14 vs
      live +0.46). Month by month on the year, LONG +1/−2 beats +3/−3 only in losing months (Jan +0.25, Mar +0.37, Aug +0.27) and loses in
      winning months (Apr −0.58, Jun −1.25); year WR 62 % vs the 67 % break-even (B19 82 %; B19 reach-+3 27 % vs year 49 %). Adaptive switch
      (small TP after the last 3 / 5 / 10 closed trades averaged < 0, no look-ahead) never helps (LONG −0.09 / −0.07 / 0.00; WIDE and LITE
      worse). Re-read on the extended batch: monthly table + the same-days backtest twin, in all three blocks.
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

- [ ] **WILLY trigger B re-test — ONE pre-registered run, final either way** (Claude, 2026-10-11): trigger B = an episode's fresh ON bar
      that FRENZY_LONG / WIDE / LITE did not take (DECISION_LOG 251). OFF since 256 on the yr5 tick year: −0.244 %/trade no stop [−0.68,
      +0.16] · −0.325 [−0.59, −0.07] with the ≈ −2.2 % backstop; 251 also found B negative at every turnover R (−0.19). That test ALREADY
      used today's WILLY exit (+1 / no stop / 120 min) and the turnover rule. What changed since: the market-volume gate is OFF (263), so
      B's population (the refusals of today's stack) differs, and 256 did not model LITE — re-test only for that reason. Rule frozen in
      the script docstring BEFORE the read: B = fresh ON bars refused by today's FRENZY_LONG / WIDE / LITE (gate off, bearish block, ATR
      3.0, LITE stretches), turnover R < 1, first strictly red 5m bar within 60 min, today's exit, one seed, eligible pairs, de-duplicated;
      live cross-check = the journal's FRENZY_* refusals priced as B (not WILLY_HOLD rows — those are hold-blocked setups). Bar: WR below
      its breakeven WR and mean < 0 with ≥ 95 % confidence in both halves → closed for good, never re-run; the full locked bar (mean > 0
      with 95 % window-bootstrap confidence, ≥ 8 windows, no window / pair ≥ 50 %, N ≥ 15) → an observe line, never armed directly;
      anything between → closed (no third test).
- [ ] **One timeline per FRENZY episode** (Claude): per FRENZY episode — key = (pair, entry_frenzy_spike_at), as in services/frenzy.py
      (starts at the spike bar, ends after 24 h with no state bar; flag ≤ frenzy_max_hours); SPIKE_FADE and yr5 rows get the key by
      rebuilding with services.frenzy on klines, never by time proximity — when each sleeve fires (WILLY at the flag / first red, LONG at
      the ON bar, LITE per stretch, WIDE on green reclaims) and its outcome by phase (minutes / % since the spike, VWAP distance). Turns
      0a+ into numbers: which phase pays, and whether the four sleeves are one idea at different moments (signal overlap + outcome
      correlation). Include the refused setups from the decision journal (refusal code per bar), not only fills. The phase cuts
      (entry_frenzy_hours, entry_frenzy_vs_vwap_pct, minutes since spike) are separator tests: predict them in 0a+ and count them in the
      §0c screen (null + OOS) — not free extra cuts.

## 3b. FRENZY losers → FRENZY SHORT opportunity (after §3 finds a loser signature)
- [ ] **B19 first read (2026-10-12, operator "LONG converted to SHORT?") — every LONG / WIDE / LITE entry mirrored as a SHORT on the same
      tick path** (scripts/study_b19_fz_short_mirror.py → reports/B19_FZ_SHORT_MIRROR.txt; 16 pre-declared cells TP 1–3 × stop 2–4, 12 h,
      + TP1/SL3 2 h). LONG: ① B19 −1.37 long → best short +1.37 (S +3/−3) · ② master −1.67 → +0.45…+2.23 (every cell +) · ③ year −0.04 →
      every short cell −0.11…−0.35 (P(short better) ≤ 0.37, H2 −0.28…−0.53). WIDE: year −0.05 → −0.18…−0.41. LITE: year +0.27 → −0.27…−0.77
      (P 0.00 — LITE's long edge is real). Read: the second-wave entry has no direction edge after costs on the year; B19 + master (20
      LONG fills, ≈ 9 days) are the losing window the long side already shows. Re-read on the extended batch in all three blocks; a SHORT
      sleeve needs a loser signature or a regime switch that holds out of sample, not the window.
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
- [ ] **Observe lines registered 2026-10-12 (DECISION_LOG 273), arm decision deferred by the operator:** BTC24H_DOWN (block ML when BTC
      24-h ≤ 0) and ML_ADX_RISING_IN_REGIME (step 2: keep only rising closed-bar ADX on the fills step 1 keeps). Re-read both on the new
      fills in all three blocks (① batch · ② master per batch · ③ backtest) against their frozen bar in CURRENT_STATE; then bring the
      arm / observe question back to the operator.
- [ ] Entry strategy in detail vs CURRENT_STATE + Scout watchlist (ML_B1H_NEGFLANK, NEG_DAILYUP_WEAKPAIR, PAIR_1H_DOWNTREND,
      TREND_ALIGNED, heat lines (heat block × BTC regime, heat re-scope / revert gates), LOADX, LONG_CHOP_BURST, mega-cap, ML cooldown): winners vs losers in current batch, master (per batch) and
      backtest. Start from the MOMENTUM-LONG ANALYSIS PIN in CURRENT_STATE; per DL 204 the master-kept ML figure is in-sample —
      judge ML on as-traded / forward fills. Filters first, not sizing (memory feedback_filters_not_sizing.md). Propose concrete rules with before/after + revert gate.

## 5. Every other sleeve (SPIKE_FADE priority, flips, momentum short, bull-run, bear-run, surge)
- [ ] **🌊 Bull-run REARM door AGE CAP (57l, DECISION_LOG 104 / 274) — NEVER TESTED live** (operator 2026-10-12). Shipped 2026-09-21 16:01
      UTC (`bullrun_rearm_max_entry_age_min` 60, flap-merge 30); 0 REARM episodes since (the door arms only ≤ 48 h after a GREEN end).
      At every review: list REARM episodes since the ship (logs `[BULLRUN_MONITOR]` + `entry_br_door_age_min` on fills + `blocked_age`
      refusals); per episode price kept (age < 60 min) vs refused-by-age entries; apply the 57l gate (first 2 episodes with net-negative
      KEPT fills → age cap 0). The old 3-window REARM gate is superseded (its windows predate the cap) — do not use it to close the door.
      If there is no REARM / GREEN at all, say so and check the logs for why (memory: absence = logs question).
- [ ] Full read of everything in CLAUDE_CURRENT_STATE + Scout; any finding → before/after in the three blocks + recommendation.
- [ ] SPIKE_FADE: high potential and strong in every batch — look for scale-up / more-fills levers as well as risk.
- [ ] **SPIKE_FADE vs FRENZY on the same pump** (Claude): the engine never holds both on one pair — open_position refuses any second
      position (generic PAIR_HELD, the counter does not name the holder), a 5-min cooldown follows any close on the pair, LITE / WILLY
      refuse as *_PAIR_HELD, FRENZY / WIDE as *_PAIR_HELD_BY_LITE, and the WILLY global hold blocks fades. So read it as a pair-occupancy
      contest, not a hedge: per FRENZY episode (key above), the fades opened inside it and the FRENZY entries refused because a fade held
      the pair (or the reverse; holder from the fills' open intervals). Price the refused side at its own live exit and the Δ of a priority
      rule (fade-first vs FRENZY-first), DAY units, two-sided, before → after in the three blocks. Fades blocked by the WILLY hold are counted
      once, in §6 "WILLY global hold cost".

## 6. Added by Claude (portfolio-level levers toward the profit target)
- [ ] **Missed moves:** Scout's missed-move diagnosis (FILTER_NEAR gate sets with more good than bad pair-days, FILTER_FAR / SLEEVE
      clusters, shortlist misses like MOVR / NMR) → loosen-a-gate or new-sleeve candidates, engine-replay tested.
- [ ] **WILLY global hold cost:** WILLY_HOLD + HOLD_FRENZY_EXEMPT — what the hold blocks vs what WILLY earns.
- [ ] **Execution realism:** STOP_SLIP (Scout-only, scripts/opportunity_scout.py), entry latency / slippage per sleeve (live vs paper fills) — the gap to real money.
- [ ] **Concurrency and ruin:** a today's-stack portfolio run of all sleeves together (daily compound, max drawdown, worst day, open
      exposure) — the profit target only counts if the book survives it.
- [ ] **Concentration:** per-pair and per-day share of each sleeve's P&L; blacklist candidates (pair-level beats dimension filters).
- [ ] **Several FRENZY sleeves on one coin** (Claude): per episode (key = pair + entry_frenzy_spike_at), the count of FRENZY-family
      entries (e.g. CFX 10-10: WILLY +1.01 · LONG −3.00 · LITE +3.01 · LONG −3.01). Today the per-sleeve cap is frenzy_max_entries_per_pair_day
      = 3 and FRENZY_REENTRY_AFTER_WIN (DECISION_LOG 261) already observes same-pair re-entries — start from them. Candidate rule, defined
      AT ENTRY (no look-ahead): refuse the k-th (k ≥ 2) FRENZY-family entry of an episode; state the priority first (WILLY enters first by
      construction → variants WILLY counts / WILLY exempt). The BLOCKED cohort faces the EXPECTANCY bar per sleeve at 1×: WR below the
      sleeve's breakeven WR, mean < 0 with ≥ 95 % confidence by an EPISODE / day-clustered bootstrap, ≥ 8 distinct episodes, no episode /
      pair ≥ 50 % of the loss, N ≥ 15; then the 30–50 % haircut, observe-first if below the bar, a pre-committed revert gate. Cohort
      completeness: add back the entries the pair-day cap, the LITE once-per-stretch rule, pair-held and the WILLY hold already removed.
      Two-sided: the freed slots / pairs taken by other setups count. Before → after in the three blocks, with the concurrency run.
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
- [ ] **FRENZY leverage / investment OBSERVATION lines — read them all, three blocks each** (operator 2026-10-12: "for frenzies we have
      in observation too the leverage and investment size"). Today's config (re-read trading_config.json at review time): LONG invest
      1.0 × lev 0.32 (6×), strong (ADX rising ∧ +DI > −DI) lev 0.5 (10×) · WIDE 1.0 × 0.2 (4×) · LITE 1.0 × 0.32 (6×) · WILLY 1.0 × 1.0
      (20×; 10× = 0.5 recommended in B19, operator decision pending). Lines to read, each as ① B19 extended · ② master per batch · ③
      backtest, $ / worst day / max drawdown before → after (avg % does not move with size):
      · FRENZY LEVERAGE WATCH (CURRENT_STATE, DECISION_LOG 207) — its review condition and candidate step;
      · strong-leverage gate (DECISION_LOG 197 / 262; 7 / 20 at B19, both legs on the revert side: live strong −1.87 vs normal −1.67;
        year strong +0.20 vs −0.36) — frozen gate decides;
      · 1× invest × 1× leverage for LONG / WIDE / LITE (the item above) and the ALL-FRENZY scenario (B19 scenario TP1 / SL2.5 / 20× →
        book to $0: reports/B19_SCENARIO_ALLFRENZY_TP1_SL25_20X.md);
      · WILLY 20× vs 18× vs 10× (B19 carry-forward section);
      · each sizing line read on the exit / filter the review keeps (sizing LAST), with the −3 / −4 stop × full leverage × max concurrent
        FRENZY slots as the ruin line.
- [ ] **SIZING OBSERVATIONS (DECISION_LOG 276, operator 2026-10-12): WILLY 20× vs 10× (vs 6×) and LITE 6× vs 8× vs 10×** — price the
      live fills at each leverage in all three blocks ($, worst day, max DD, concurrent stops per day, fills past the 20× liquidation
      distance) and read the frozen bars in CURRENT_STATE (WILLY: ≥ 20 fills under −4; LITE: ≥ 30 fills on ≥ 10 days). Year-book refs:
      WILLY 20× / 10× / 6× $2,356 / $9,035 / $10,272; LITE 6× / 8× / 10× (WILLY 10×) $9,035 / $12,160 / $14,615 (DD −67 / −72 / −77 %); LITE-alone book $9,236 / $10,757 / $11,142 (DD −48 / −59 / −68 %) — 8× ≈ growth optimum; B19 +$100 / +$134 / +$167.
- [ ] **⚖️ SHIPPED 2026-10-12 (DECISION_LOG 277) — read its revert gate (first 20 LITE fills at 0.4 avg < 0, or a LITE day < −25 % of the book; judge on the recorded leverage / bracket / liquidity-cap columns — thin pairs can fill below 8×) — LITE to 8× leverage, 1× invest (operator 2026-10-12: "save it for next batch review, increase LITE
      to 8x and 1x inv")** = `frenzy_lite_lev_mult` 0.32 → 0.4 (20× base), `frenzy_lite_invest_mult` stays 1.0. Evidence so far: LITE-alone
      year book 6× $9,236 (DD −48 %) → 8× $10,757 (−59 %) → 10× $11,142 (−68 %) — 8× ≈ the growth optimum; B19 +$100 → +$134; master B18
      +$124 → ≈ +$165. Re-read on the extended batch in all three blocks (LITE avg % still > 0? worst day / concurrent LITE stops at 8×) and
      bring the change to the operator; Pattern W (WR ≥ 70 %) cannot pass for LITE → it ships as a declared operator override with a
      pre-committed revert gate (first 20 LITE fills at 8× average < 0 or a LITE day worse than −25 % of the book → back to 6×).
- [ ] **MOMENTUM LONG invest down to 1×** (operator: frequency used to need the bigger ticket; FRENZY now supplies volume). ML today:
      base invest 1.0 (lev 20) · UNMATCHED LONG cell inv 1.5 · BTC RSI×ADX long cells 60-65/28-30 inv 2.0 and 60-65/22-25 inv 2.0 × lev
      1.5 · long_unmatched_quiet 1.5 (no-op, take-the-max) · caps 2.5 invest / 2.0 lev (re-read trading_config.json at review time). Per
      cell: its promotion / revert history (cell registry) and its multiplier-cell verdict (★ WORKING / ⚠ DRAG / ✗ HARMFUL on ≥ 5 fresh
      fires) — a WORKING cell is not cut to 1× without evidence. $ and drawdown before → after, and the freed margin / slots for the other
      sleeves. Only after §4's filter work; a size cut is never the headline fix for ML's profitability (memory feedback_filters_not_sizing)
      — it is judged per cell on the freed margin / slots and the concurrency run.

