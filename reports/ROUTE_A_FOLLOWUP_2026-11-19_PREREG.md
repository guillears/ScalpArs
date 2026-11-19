# ROUTE A FOLLOW-UP — holdout check · VWAP-hold selector · small take-profits — PRE-REGISTRATION

**Frozen 2026-11-19 16:30 UTC**, before any new outcome was computed: no 1m kline had been fetched for the full trigger population,
no Q2 / Q3 number had been read. Feasibility checks only (local 1m cache coverage: 22,071 of 33,399 triggers covered; 18,019
pair-days to fetch). Research only: no engine / config / test / CLAUDE file changes, no state-changing git.
Anything added after the first result is read is labelled EXPLORATORY in the report.

Inputs (frozen, from Route A 11-19): `reports/ROUTE_A_TICKS_data/order.csv` (33,399 W15_R3_V10 triggers 2024-01 → 2026-11-17),
`ev/*.npz` (1,550 sampled events: ticks [sig − 75 min, sig + 10 min) + 1m bars from ticks to the end of the last downloaded day),
`features.pkl`, `eval.pkl` (Route A BOTH-model scores: DESIGN out-of-fold + HOLDOUT, 80th-pct threshold frozen in DESIGN).
DESIGN = 2024-01 → 2026-06 · HOLDOUT = 2026-07 → 2026-11-17 (unchanged from Route A).

Costs everywhere: entry taker 0.05 % + slip on the entry price; TP exits maker 0.02 %, no slip; stop / trail / time exits taker
0.05 % + slip on the exit price; real funding (settlements inside the hold; long pays a positive rate) — missing funding months
charged 0.01 % per 8 h (Binance default) and counted.

## Q1 — is the HOLDOUT result a model edge, a regime artefact or noise?
**(a) Full trigger population, priced on 1m klines.**
- Every one of the 33,399 triggers. 1m bars from local caches (`reports/backtest_cache/k1m`, `reports/backtest_cache_2025/k1m`)
  where they cover [sig − 20 min, sig + 25 h] without gaps, otherwise data.binance.vision daily 1m kline zips (CHECKSUM verified,
  0 API weight), cached under `reports/ROUTE_A_FOLLOWUP_data/k1m/`.
- Entry (primary): trigger close + 8 s ≈ the 1m bar [sig, sig + 60 s) open × (1 + d), d = median (first aggTrade ≥ sig + 8 s ÷ that
  1m open − 1) measured on the 1,550 tick events (a calibration constant, not an outcome); × (1 + slip 0.10 %), stress 0.30 %.
  Path evaluated from the bar [sig + 60 s, …) onward (the first minute's range is not used). Pessimistic variant: entry at the
  close of the first 1m bar (≈ +60 s). Cross-check on the sampled events: 1m-priced vs tick-priced C0 net (corr + mean bias).
- Exits Q3 (TP +3 % maker / SL −4 % / 4 h) · TR (arm +3 %, trail 3 % off the high, SL −5 %, 24 h) · T1 (1 h time), same `sim()`
  as Route A (same-bar stop first).
- One position per coin per exit (a trigger while that coin's previous trade of the same exit is still open is skipped).
- Primary population = all triggers; secondary = not-yet-flagged (Frenzy flag not closed at or before the trigger close).
- Tables: per half-year (24-H1 … 26-H2) and per month: N, trades/day, WR, avg net %, day-block bootstrap 95 % CI, @0.30 %.
- Question read: is 2026-H2 positive for all triggers (CI), what is its rank among the 6 half-years, and how many earlier months /
  halves were as good or better.

**(b) Selection lift in the sampled events (population-weighted, already-flagged excluded as in Route A).**
- For each checkpoint C0 / C1 / C2 / C5 × exit Q3 / TR / T1: lift = avg net (top-20 % score) − avg net (all) and
  top-20 % − bottom-80 %; day-block bootstrap (1,000) 95 % CI of the lift; in HOLDOUT, and the same in DESIGN-OOF and per
  DESIGN half for comparison.
- **Model edge** requires: HOLDOUT lift (top-20 vs bottom-80) > 0 with CI low > 0 on the primary cell (C2 TR, the Route A
  headline) AND DESIGN-OOF lift > 0 on the same cell. Otherwise the HOLDOUT positive is not attributable to selection.

**(c) Regime in WINDOW units.**
- Variables (lag-safe prior-day, `FRENZY_LONG_SEARCH_data/tape_daily.csv`): R1 BTC daily EMA50 > EMA200 · R5 breadth > 50 % of
  perps above daily EMA50 · RF Frenzy 7-day count ≥ median · B30 BTC 30-day return > 0.
- Window = maximal run of days with the same sign; runs shorter than 7 days are merged into the preceding window.
- 2026-H2's sign = its majority sign per variable. Per window (≥ 10 triggers): all-trigger avg net (TR, Q3, T1) on the full
  population. Read: number of earlier windows with the 2026-H2 sign, share positive, and windows-of-sign vs other-sign (Mann-Whitney
  on window means). **"Regime switch" may only be claimed with ≥ 5 independent earlier windows of the same sign that show the same
  direction (≥ 4 of 5, or a one-sided binomial p < 0.05).** With fewer windows: "cannot be called".
- Also: monthly all-trigger TR net vs monthly regime shares (Spearman over months, descriptive).

**Verdict rule:** model edge (b passes) · regime artefact (b fails AND a shows 2026-H2 all-trigger P&L clearly above DESIGN AND c has
≥ 5 same-sign windows agreeing) · otherwise noise / not attributable.

## Q2 — "price holding above the burst's average price" as a long selector
- Feature: `vwap_gap` (last price ÷ BURST VWAP − 1, BURST = [sig − 15 min, decision)), at C0 / C1 / C2 / C5, sampled events,
  already-flagged excluded, population-weighted.
- Selectors (thresholds frozen on DESIGN, weighted 80th percentile per checkpoint): **S_VW** = top quintile of vwap_gap ·
  **S_VA** = top quintile of z(vwap_gap) + z(tps_accel) (z from DESIGN, 1 / 99 % winsorised).
- Labels: PATH +X % before −X % for X ∈ {1, 2, 3, 5} after entry (entry = first trade ≥ decision + 8 s; ticks → 1m from ticks →
  local 5m; same-bar both = loss), RUNNER (Frenzy flag within 2 h). AUC of the raw feature and of the S_VA score per label.
- Priced net % under Q3 / TR / T1 (Route A's tick-priced outcomes), slip 0.10 % and 0.30 %, DESIGN then HOLDOUT.
- Dose-response: weighted net % and PATH rate by DESIGN quintile of the feature, per checkpoint (DESIGN and HOLDOUT, HOLDOUT on the
  DESIGN cut-points).
- Secondary (full population, C0 only): 1m proxy `vwap1m` = close at sig ÷ (Σ quote vol ÷ Σ vol over the 15 1m bars before sig) − 1,
  top quintile (DESIGN cut) on the Q1(a) priced population, per half-year.
- **PASS (selector):** DESIGN avg > 0 with day-CI low > 0 on ≥ 1 exit at a checkpoint, monotone dose-response (Spearman of quintile
  vs net > 0 in DESIGN), HOLDOUT avg > 0 on the same cell, > 0 at 0.30 % slip. Separately reported: does it raise the RUNNER rate
  (picks future Frenzies) while PATH AUC ≈ 0.5 (does not pick profitable longs)?

## Q3 — small take-profits on the tick windows
- Grid: TP ∈ {0.5, 1.0, 1.5, 2.0} % (maker resting at entry × (1 + TP) rounded UP to the tick; filled only when a trade prints
  ≥ 1 tick through it) × stop ∈ {0.5, 1.0, 2.0, 3.0} % (taker; fill = min(stop level, triggering trade price) × (1 − slip)) × max hold
  {5, 15, 60} min (time exit = taker at the first trade ≥ entry + hold × (1 − slip)) × checkpoint C0 / C1 / C2 / C5 = 192 cells.
- Entry = first trade ≥ decision + 8 s, × (1 + slip); slip 0.10 % (stress 0.20 / 0.30 %, applied to entry and taker exits).
  Net = (exit ÷ entry − 1) − 0.05 − exit fee (0.02 maker TP / 0.05 taker) − funding.
- Resolution: exact trade order from ticks until sig + 10 min; after that the 1m bars built from ticks (then local 5m if the day
  ends); a bar where both the TP and the stop trade = STOP first (primary) and TP first (optimistic bound, reported). Share of
  outcomes decided outside the tick window is reported per hold.
- Tick sizes: `reports/backtest_cache_2025/exchange_ticks.json`, else the smallest price step seen in the event's ticks.
- Cost arithmetic per cell: avg win, avg loss, breakeven WR = |avg loss| ÷ (avg win + |avg loss|), observed WR.
- Selections: (i) all sampled triggers (not-yet-flagged) · (ii) Route A top-20 % BOTH score (DESIGN-OOF + HOLDOUT; 24-H1 has no
  OOF score) · (iii) Q2 S_VW.
- **Random-entry family-wise null:** each selected event's entry time is replaced by a random minute on the same coin-day (from the
  1m bars available in its npz, ≥ 61 min before their end); 300 pre-drawn minutes per event; observed AND null priced at the SAME 1m
  convention (entry = close of the decision minute's 1m bar × (1 + slip), path from the next bar, same-bar both = stop) so the
  resolution cannot favour either; statistic = max DESIGN weighted mean net over the 192 cells; 300 draws; family-wise p and
  per-cell excess over the null mean.
- Trades/day on the full population = selected share × the population trigger rate (1 position at a time per coin; holds ≤ 60 min
  vs the 6 h debounce, so overlaps cannot bind).
- Daily compound: Kelly fraction (notional ÷ equity) at the one-sided 95 % Wilson floor of the cell's WR with its avg win / loss;
  DCR = (1 + f × avg net)^(trades/day) − 1; f ≤ 0 → no size, 0 %/day.
- **PASS (cell):** DESIGN avg > 0 with day-CI low > 0 · family-wise null p < 0.05 · HOLDOUT avg > 0 · > 0 at 0.20 % slip · ≥ 3
  trades/day on the full population · stop-first result (not the TP-first bound).

## Discipline
Window units for anything market-wide; LOO coin / day on any positive cell; best-5 %-out; 30–50 % in-sample haircut on any DESIGN
number; HOLDOUT read once per question; no threshold re-fit after reading. Prior evidence engaged (not repeated):
HOT_SCALP_BACKTEST_2026-10-01 (1 s, hit rate = no-edge rate), FRENZY_SCALP_1M_SEARCH / MODEL_WF_2026-10-03 (0 pass),
FRENZY_ON_SCALP_STUDY_2026-10-06 (0 / 1,205 cells).
