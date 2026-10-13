# Two-year winners vs losers — shared brief for every analysis agent (2026-10-13)

**Operator ask (2026-10-12/13):** "the analysis of winners vs losers deep as a true quant, to identify the macro or by-pair variable that
separates winners vs losers per sleeve, in bullish macro regime and bearish macro regime (2025 bullish, 1st half 2026 bearish, 2nd half
bullish again)". Project rules: CLAUDE.md (repo root) — binding. Plan: reports/NEXT_BATCH_REVIEW_PLAN.md "TWO-YEAR WINNERS vs LOSERS".

## Data
- **Fills (re-priced to TODAY's config, 443a7b5):** `reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv` — 21,933 trimmed CLOSED engine-replay
  fills, 2025-01-01 → 2026-10-11, 3 seeds (seed only changes scan cadence; FRENZY seeds ≈ one path → use seed 1 for FRENZY tests, pool
  seeds for the others and cluster by day). Columns: every order column incl. all `entry_*` stamps (pair + macro), `sleeve`, `seed`,
  `regime` ("2025 BULL" / "2026-H1 BEAR" / "2026-H2 BULL"), `pct` (as replayed), **`pct_new` (today's exit — use this)**, `usd_new`
  ($ at today's size on a fixed $2,900 book), `t` (open time). Built by scripts/yr6_reprice_today.py (FRENZY LONG / WIDE TP +1.5 by
  first touch on the replayed peak — exact for a lower TP; LONG 2× / WIDE 1× / LITE 10× own +3 / WILLY 10× +1.25/−4/90 / momentum long ×0.5).
- Per-sleeve × regime summary: reports/ENGINE_REPLAY_YR6_2Y_TODAY.txt (+ _regimes.csv, _monthly.csv).
- Real fills for out-of-sample checks: master pool reports/MASTER_POOL_stacked.csv (B1…B20, today's stack) and SCREENED_BASELINE.csv
  for momentum / flip sleeves.

## Data-fidelity caveats you MUST carry (bug gate)
- 2025 ticks exist only for top-50 ∪ prior-day top-50 pair-days; other pairs exit on 1m bars. Measured exit bias of 1m vs ticks on the
  same 2026 fills: ≤ −0.03 %/fill (reports/YR6_1M_BIAS.txt) — small. But 2025 FRENZY / WILLY non-top-50 fills average far worse than
  top-50 ones (WILLY −0.28 vs +0.03; LITE −0.20 vs −0.03; LONG −0.20 vs +0.17), while 2026 shows WILLY the OTHER way (non-top-50 +0.16
  vs −0.01) → the 2025 non-top-50 gap may be partly an ENTRY-side artefact. Every 2025 FRENZY/WILLY finding must be shown split by
  top-50 membership (reports/backtest_cache/universe_daily_top50.csv + reports/backtest_cache_2025/universe_daily_top50.csv; pair-day in
  top-50 or prior-day top-50) and must hold inside the top-50 subset to count.
- 2025 market caps = today's circulating supply × 2025 price (proxy); delisted coins have none (WILLY refused). Supply growth makes the
  WILLY turnover filter looser in 2025.
- The replay's ENTRIES are the 01419d4 config's; today's sizing / exits are re-priced, second-order slot effects are not modelled.

## Method (non-negotiable — CLAUDE.md quant discipline + memory feedback files)
1. Per sleeve × regime: N · WR · avg % · day-clustered CI. Re-state the sleeve's breakeven WR.
2. **Coverage first:** list every `entry_*` column with its non-null share per sleeve × regime; a variable scored on < 80 % of fills
   gets its own read, never a zone-vs-"rest" table where unscored rows sit in "rest".
3. **Exhaustive screen:** every numeric `entry_*` column at sign / median / outer-tercile granularity, PLUS every 2D pair of the top
   ~40 columns (quadrant splits), each with a shuffled-label null (≥ 200 shuffles, same statistic) — report the survivors' rank vs null.
   Macro variables (BTC / breadth / market-wide) are read in DAY units (same-day fills = one observation) — window-units rule.
4. **Discover → confirm:** discover in one regime, confirm out-of-sample in the other same-direction regime (2025 BULL ↔ 2026-H2 BULL),
   and report the bear half (2026-H1) separately. A separator must have the same sign in discovery and confirmation, with confirmation
   P ≥ 0.90 (day bootstrap). Then check against the real master fills where the sleeve has them.
5. Per-pair and per-day concentration (≥ 50 % of a zone's loss from 1–2 pairs or one day → it is a blacklist / day effect, not a filter).
6. Non-monotonic single-variable patterns = confound. Uniform degradation across all cohorts in a regime = regime cause (sleeve on/off
   switch by regime), not an entry filter.
7. Any proposed filter faces the locked EXPECTANCY bar (WR below the sleeve's breakeven at 1×, avg < 0 at 95 % window-bootstrap, ≥ 8
   windows, no window/pair ≥ 50 % of the loss, N ≥ 15) + 30–50 % in-sample haircut + a pre-committed revert gate. Observe-first otherwise.
8. **List what could NOT be tested** (columns missing in 2025, low coverage, fidelity limits).

## Rules
- Cache-only: no Binance calls. Never edit engine code, config, trading_config.json, templates, tests; never commit / push; never call the
  bot API; never touch reports/backtest_cache/replay/ or running processes.
- Scripts in scripts/ named `study_yr6wl_<area>_*.py`; report `reports/YR6_WL_<AREA>.md` with: plain-English summary first → per-regime
  sleeve table → coverage table → top separators per regime with null rank and OOS confirmation → regime-switch read → recommendation
  (or "no change") with a pre-committed gate → could-not-test → exact commands. Every headline number computed two independent ways.
