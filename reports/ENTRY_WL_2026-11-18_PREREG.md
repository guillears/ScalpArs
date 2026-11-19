# ENTRY_WL — can the ENTRY separate the shorts that RUN from the ones that GIVE BACK? PRE-REGISTRATION

**Frozen 2026-11-18 23:05 UTC, before any entry-variable result for this question was computed.** Research only, BACKTEST. No engine,
config or test change. No commit.

What I had already seen before writing this (disclosed so it is not mistaken for a blind prior):
- the three exit studies (EXH_EXIT2, DIST_EXIT, NOSPOT_EXIT): runners and give-backs look the same at +2–4 %; locks help random entries
  and hurt these entries;
- DIST_WL: on L1 the RSI1h "still strong" factor separated REACH, and RSI1h ≥ 74.07 IS the DIST sleeve. Inside DIST it is already
  applied, so RSI1h is re-tested here only as a within-cohort dose check;
- the EXH baseline file's own stamp columns and their coverage (no outcome split looked at).

## 1. Theory: why a short after exhaustion / distribution FOLLOWS THROUGH vs GIVES BACK
A short entered after a pump makes money only if supply keeps beating demand for hours (the +10 % TP needs a slow bleed). A give-back
is a short that got the first dip (+2–4 %) and was then squeezed back through the stop. The first dip is common to both (the signal
candle's own mean reversion). What decides the rest is whether there is (a) still fuel to fall, (b) someone forced to sell or forced
to buy, (c) a market that lets the coin fall, (d) a coin that can be pushed around.

| # | theory | variable(s) (entry-time) | predicted RUNNER side |
|---|---|---|---|
| T1 | **Fuel left.** A pump that is still near its high has the whole run to give back; one already far below its high has spent it. | `dd_peak_pct`, `run_pct`, `ret4h_pct`, `ret24h_pct`, `dist_vwap_pct`, `dist_base_pct` | near the high (small dd), big run, above VWAP |
| T2 | **Exhaustion depth.** More volume decay = buyers gone. | `vd`, `vd24`, `vmult15` | high `vd` |
| T3 | **Trapped longs = fuel.** Longs crowded into the top (high L/S, positive funding, OI built into the signal) must sell into the fall. Crowded shorts (low L/S, negative funding, falling OI) = squeeze fuel → give-back. | `top_pos_ls`, `top_acct_ls`, `glob_ls`, `fund_last`, `prem_last`, `oi_chg1h`, `oi_chg4h`, `oi_to_vol24`, `top_pos_chg4h` | high L/S, funding > 0, OI rising |
| T4 | **Market support.** A falling BTC / weak breadth lets alts bleed; a rising market lifts them back. Read at SIGN granularity, in WINDOW (day) units. | `btc_d1_regime`, `btc_ret1h/4h/1d`, `btc_slope1h_sign`, `btc_regime`, `breadth` | BTC daily bear, BTC falling, weak breadth |
| T5 | **Coin quality.** Thin, high-ATR coins are easy to squeeze. | `q5`, `vol24`, `mcap`, `atr15_pct`, `has_spot`, `list_age_d` | deeper / larger coin (two-sided on ATR) |
| T6 | **Time since the spike / second peak.** Late signals come after the move is spent; a second peak (re-test of the high) is a failed breakout. | `hrs_since_spike`, `hrs_since_peak`, prior signals in the episode, prior fill in the episode | early, or a fresh re-test of the high |
| T7 | **Taker flow.** Aggressive buying at the top that fails = trapped buyers (runner); aggressive selling already done = capitulation (give-back). | `taker_1h` | high taker buy ratio (two-sided) |
| T8 | **DIST strength factor (DIST only).** Inside DIST, a hotter coin keeps running down. | `rsi1h`, `rsi4h`, `gap1h`, `di_diff1h` | higher RSI1h |

Any other entry-time column is screened too (W/L = every variable), but only as a screen.

## 2. Cohorts, exits, splits (frozen)
- **EXHAUSTION_SHORT:** the 3,368 kept fills of `BASELINE_TRADES_EXHAUSTION_2026-11-18.csv` (DL 578 rule, DL 582 filter), today's exit on 1m
  from `EXH_EXIT2` (REF: TP min(20 × ATR15, 10 %) maker / SL min(1.5 × ATR15, 6 %) / 24 h, model slip ≥ 0.10 %, real funding).
  Features: the UNIFIED_SHORT pool row at the same (pair, entry_ms) + the baseline stamps + DIST_WL's lag-safe extras.
- **DISTRIBUTION_SHORT:** the DL 586 cohort (L1 ∧ RSI1h ≥ 74.07), REF = Exit B on 1m from `DIST_EXIT` (1,547 one-position fills),
  features from `DIST_WL` rows.
- **DESIGN** 2023-07 → 2025-06 · **HOLDOUT** 2025-07 → 2026-11-18, read ONCE after the survivor list is frozen. EXH also reports the clean
  2023-H1 block **E** only if entry features can be rebuilt for it (they cannot from the frozen pool, which starts 2023-07: reported as a
  blind spot if so).

## 3. Labels (each trade's own stop distance, 1m path, REF exit)
- **RUNNER** = REF hit the TP, or REACH ≥ +8 % before the stop.
- **GAVE_BACK** = REF stopped, REACH ≥ +2 % first (and < 8 %).
- **NEVER_WORKED** = REF stopped, REACH < +1 %.
- **OTHER** = the rest (time exits under +8 %, stops with REACH 1–2 %).
- **REACH** = max favourable move before the stop (continuous). **WORKED** = +1 % before the stop.
- **Primary contrast: RUNNER (1) vs GAVE_BACK (0).** Secondary: booked net %, WORKED, REACH (all fills).

## 4. Screen
- Every entry-time variable at 3 granularities: SIGN (> 0 vs ≤ 0, only where both sides have ≥ 30 rows; for binary / categorical:
  each level vs the rest), TERCILE (top vs bottom, edges on DESIGN), QUINTILE (top vs bottom).
- Every 2D pair: sign × sign and tercile × tercile cells, cell vs rest. Minimum 30 rows in the cell and in the rest (RUNNER + GAVE_BACK).
- Statistic: two-proportion z on the RUNNER share (primary); Welch t on net % / WORKED / REACH (secondary). AUC for every continuous variable.
- **Family-wise null:** 200 label shuffles (max |z| over all 1D + 2D tests) AND 200 time-rotations of the label vector (offset ≥ 30 days,
  keeps day clustering). The stricter 95 % bar of the two governs.
- **Multivariate check:** an L2 logistic on the T1–T8 theory variables (standardised, median-imputed), fit on DESIGN, AUC on HOLDOUT.
- Coverage table + blind spots first. Market-wide variables (BTC / breadth / Frenzy counts) also read in DAY units, with per-day
  concentration.

## 5. Survivors (DESIGN only)
A test survives if |z| > the family-wise bar. If none survives, the 5 strongest **theory** reads (T1–T8, best granularity) go forward as
theory-led reads, which can at most be OBSERVE. For each: DESIGN half-years (≥ 3 / 4 same sign), both BTC daily signs (same sign where
N ≥ 30 each), day-block (10-day) bootstrap CI of Δ avg net % (kept − blocked) excluding 0, top pair and top day < 50 % of Δ,
best-5 %-out same sign, holds at 0.20 % slip. Frozen list → HOLDOUT once.

## 6. Ship bar (per action, per sleeve)
- **HOLDOUT:** same direction; AUC (RUNNER vs GAVE_BACK) ≥ 0.55 on the separator; Δ avg net % CI excludes 0.
- **(a) Entry FILTER:** the blocked side at 1× meets the locked EXPECTANCY bar: WR below the sleeve's breakeven WR · avg < 0 with 95 %
  window-clustered (day) bootstrap confidence · ≥ 8 windows, no window / pair ≥ 50 % of the loss · N ≥ 15. Plus one-size DCR at 0.61 after
  ≥ before (after a 30–50 % haircut on Δ), joint book DCR ≥ before.
- **(b) SIZE tilt:** up-size only on the predicted-runner side, down-size only on the predicted-give-back side (never cross). Judged by
  Kelly at the Wilson floor per side, DCR at equal average exposure, max DD, worst day, joint add. The locked Pattern-W multiplier bar
  (WR ≥ 70 %) cannot apply to a 42 %-WR TP-runner sleeve; this is disclosed and the tilt can at most be OBSERVE unless it clears both
  HOLDOUT and the Kelly test.
- **(c) Conditional exit:** a lock / break-even / small-TP only on predicted give-backs, cells reused from the EXH_EXIT2 / DIST_EXIT
  sims (paired entries). Read on the live-stopped cohort (saved vs deeper), and the winners it cuts.
- Every action: N, trades/day, WR, avg %, one-size DCR at 0.61 before → after, max DD, worst day, Kelly at the Wilson floor, joint book
  add (`joint_short_sim.sim`), leave-one-out, 30–50 % haircut. A pass on DESIGN + HOLDOUT = SHIP-candidate; a miss on any locked bar =
  OBSERVE naming the bar; no HOLDOUT direction = nothing.
- Master cross-check: real paper fills of each sleeve since they started (expected: very few).
