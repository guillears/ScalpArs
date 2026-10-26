# ML REGIME W/L 2026-10-26 — theory + pre-registration (written BEFORE any regime × outcome table of this study)

Frozen when this file was written; its SHA-256 is printed in the report. The per-fill regime file was built and checked against the engine stamps
(eff72 / above72 / r72 / off30d / off24h corr ≥ 0.998) but **no regime variable had been crossed with an outcome yet**.

## 0. What the author had already seen (declared contamination)
- PREP_B25_ML_WL (same 158 KEEP fills): STRONG_BULL / BTC 5m ADX > 28 = winner side; BTC 15-min return ≤ 0, top-10 corr15 > 0.365 = loser side;
  breadth-low ∧ quiet pair = winner side; recent eras G5/G6 lose; no separator beats the null.
- ML_REGIME_2026-10-16 (backtest, 50 day-level variables): nothing beats chance; persistence (variance ratio / autocorrelation) points the wrong
  way between eras; weekly autocorrelation of results ≈ 0.
- Washed-out window Jun-18→Jul-2 (BTC 15–24 % below 30-day high): every sleeve won (one episode = one window).
- DL 273 BTC24H_DOWN (BTC 24 h return ≤ 0 → worse), DL 171 BTC near 24 h low refuted, DL 159 BTC 1 h slope < 0 refuted, DL 161 eff72 chop
  burst line, heat block (bull ≥ 85).
These are therefore NOT counted as confirmations if they re-appear; they are re-reads.

## 1. Theory (first principles)
A 5-minute momentum long buys an alt right after a short breakout (EMA fan, ADX rising, breadth not overheated), targets +0.4…+1.0 %,
stops at −0.7…−1.0 %. Its edge is *continuation* over the next 15–60 minutes. Breakeven WR ≈ 61.5 % — it needs most pops to continue.
Anything that makes alt pops get **sold into** (mean reversion) or makes the move **BTC-noise rather than alt-specific** turns it into a loser.

| # | mechanism | measurable variable(s) | predicted direction |
|---|---|---|---|
| T1 | **Trend vs chop.** In a directional BTC tape, alt breakouts ride the drift; in a range, a 5 m breakout is a range edge that is faded. | `btc_eff_24h`, `btc_eff_72h`, `btc_eff_6h`; 1 h / 4 h / 1 d EMA20 slope, close vs EMA20, EMA20 vs EMA50, ADX 1h/4h | higher efficiency / positive HTF slope → **better** |
| T2 | **Drawdown / bounce.** After a capitulation (BTC far below its 30 d high) sellers are exhausted and alt pops have high beta to the rebound; near the 30 d high longs are crowded and pops are sold. | `btc_offhi_30d`, `btc_offhi_7d`, `btc_offlo_7d`, `btc_r30d`, `btc_r7d` | deeper below 30 d high → **better**; within ~3 % of the high → **worse** |
| T3 | **Alt rotation.** When alts outperform BTC (risk-on rotation) alt-specific breakouts continue; when BTC dominance rises, alt pops are relief moves. | `alt_minus_btc_7d`, `alt_minus_btc_24h`, `ethbtc_r7d`, `ethbtc_r24h` | alts > BTC → **better** |
| T4 | **Breadth.** Broad participation means the pop has support; narrow tape means isolated pumps that revert. (Overheated breadth ≥ 85 is already gated.) | `br_above_ema50_1h`, `br_adv_24h`, `br_med_r24h`, `br_above_ema20_5m` | broader (not overheated) → **better** |
| T5 | **Correlation / dispersion.** In a high-correlation tape every alt "breakout" is BTC beta and reverses with BTC noise; high dispersion = idiosyncratic trends persist. | `br_corr24h_top20`, `br_beta_btc_24h`, `br_disp_24h`, `br_disp_1h` | high corr / high beta → **worse**; high dispersion → **better** |
| T6 | **Volatility regime.** Very low vol: targets not reached before noise; rising vol in a sell-off: stops run. Compression → expansion helps breakouts. | `btc_rv_24h`, `btc_rv_7d`, `btc_rv24_pctl90d`, `btc_rv24_rv7`, `btc_rv_1h` | extreme-high rv percentile → **worse**; very low rv → **worse** (non-monotone; buckets only) |
| T7 | **Liquidity / time.** Weekend and Asia-night books are thin; pops on thin books revert. Low aggregate volume = no follow-through. | `weekend`, `hour`, `dow`, `br_qv24_x7d`, `btc_qv24_x7d` | weekend / low volume → **worse** |
| T8 | **Funding / leverage build-up.** High positive funding = crowded longs; breakouts are sold and long-liquidation cascades hit stops. | `br_funding_med`, `br_funding_med3d`, `br_funding_pos_share`, stamp `entry_funding_rate` | high funding → **worse** (coverage ends 2026-10-02 — partial) |
| T9 | **Short-horizon persistence (direct).** Trending micro-structure → pushes continue. | `btc_vr15_3d/7d` > 1, `btc_ac5_3d/7d` > 0 | > 1 / > 0 → **better** (already failed in the Oct-16 backtest — re-read on master) |
| T10 | **Short-term BTC drift.** BTC rising into the entry is a tailwind. | `btc_r1h`, `btc_r4h`, `btc_r24h`, `btc_rsi_1h/4h/1d`, `btc_di_1h/4h` | up → **better** (BTC24H_DOWN is an existing line) |

**Uniform-degradation prediction:** if the loss is regime-driven, every sub-cohort (confidence, pattern cells, pair-size tiers, ATR terciles,
hours, ADX direction) degrades together in G5/G6 and the never-green share rises everywhere; if it is an entry problem, the loss concentrates in
specific cells.

## 2. Primary hypotheses (single tests; everything else is the exploratory family under the family-wise null)
| id | variable | zone predicted to LOSE | reasoning |
|---|---|---|---|
| PA | `btc_eff_24h` | ≤ median (chop) | T1 |
| PB | `btc_offhi_30d` | > −3 % (near the 30 d high) | T2 |
| PC | `br_corr24h_top20` | > median | T5 |
| PD | `alt_minus_btc_7d` | ≤ 0 (BTC outperforming) | T3 |
| PE | `br_funding_med` | > median (partial coverage) | T8 |

## 3. Method (frozen)
- Cohort: MOMENTUM LONG CLOSED, non-probe, non-MANUAL. Discovery = master KEEP (today's stack, `stack_pct`, % at 1×, N 158).
  Robustness = TRAD (as traded, N 210). B25 provisional (9) descriptive only. PROBE fills excluded. Backtest yr6 replay (MOM-long, 3 seeds)
  REFUTE-ONLY.
- Era groups G0 BASE · G1 B1 · G2 B2 · G3 B3 · G4 B4–B11 · G5 B12–B16 · G6 B17–B24 (as PREP_B25). "Good" = G0–G3, "bad" = G5–G6.
- Every regime variable: sign split first (variables with a natural zero), then median, then terciles (≤ q33 / > q67), then quintile shape.
- Statistics: zone vs rest Δ avg %; DAY units (days on each side, mean of day-means, share of negative days); per-day and per-pair share of
  the zone's net loss; circular-shift label null on the time-sorted outcome (1,000 shifts; day clustering kept); family-wise p by max-|Δ| over
  the whole regime family; ERA-STRATIFIED Δ (fill-weighted mean of within-era-group Δ) — a regime variable that only separates *between* eras
  is indistinguishable from "era"; leave-one-era-group-out sign with thresholds refit on the other groups; with / without the washed-out window;
  TRAD same sign; replay sign per regime.
- Gap decomposition: OLS y ~ late-era dummy (G5/G6 vs G0–G3) with and without the regime zone — how much of the era gap the variable absorbs.
- Within-regime entry screen (step 4) only for the top 1–3 regime splits: every dashboard column + every entry_* stamp + prep features, 1D,
  then exhaustive 2D (median × median quadrants) with a circular-shift null, inside each regime side; interaction = zone Δ in the bad regime
  minus zone Δ in the good regime.
- Verdicts: expectancy bar (WR < 61.5 %, window-clustered bootstrap P(avg<0) ≥ 0.95, ≥ 8 windows, N ≥ 15, no window / pair ≥ 50 % of the
  zone loss, 30–50 % haircut). Regime candidates = sleeve on/off switch: N counted in DAY windows.
