# Momentum LONG — winners vs losers BY MARKET REGIME (2026-10-26)

Research only (recorded as DECISION_LOG 382 on 2026-10-26 with the scout `scripts/scout_ml_regime_lines.py`; the report and scripts ship with that entry). No engine / config / template edit, no bot API call, **no Binance call by the study**
(every kline came from the caches: `backtest_cache[_2025]/k5m_full`, `PREP_B25_ML_WL_fetch`, `ML_EXIT_DEEP_2026-10-26_fetch`, `ml_regime/fetch`).
Pre-registration `reports/ML_REGIME_WL_2026-10-26_PREREG.md` (SHA-256 `2848c046…f83eac7`) was written after the regime file was built and
validated but **before any regime × outcome table**. Cohort = the PREP_B25 loader (158 KEEP master at 1× + 9 B25 provisional, PROBE excluded). `scripts/validate_against_master.py` → ALL CHECKS PASS.

---

## Plain-language summary

1. **The losing eras are a different market, and every slice of the sleeve got worse together.**
   - 31 of 31 sub-cohorts lose more in G5–G6 (B12–B24) than in G0–G3: confidence, regime label, W/C patterns, pair size, rank, ATR, ADX, RSI, hour, weekday, same-scan.
   - The never-green-loser share rises in all of them, from 8 % to 34 % of fills.
   - No pair carries more than 7 % of the bad-era loss.
   - Checklist ③ verdict: the cause is a regime / unmeasured variable, not a broken entry cell.
2. **What the bad tape looks like.** From 2026-09-21 on, four things changed at once:
   - alts started moving in lock-step with BTC: top-20 24 h correlation 0.15–0.34 → 0.42, alt beta to BTC 0.8–1.4 → 1.4–1.8;
   - dispersion fell (24 h cross-alt dispersion 9.6–14.9 → 6.0–8.4);
   - BTC moved from below its 1-day EMA200 (EMA50 −8…−12 % under EMA200) to above it (+3.6…+6.1 %);
   - BTC 7-day volatility was lower.

   These variables move together and changed once, so they are **one regime episode, not many observations**.
3. **The best regime separator is the one the theory predicted: high alt-to-BTC correlation (pre-registered primary PC).**
   - Fills taken when the top-20 24 h correlation was above 0.316 (vs at or below it): 79·62 %·−0.08 % vs 79·80 %·+0.33 %, Δ −0.40.
     Fill-level shift p 0.001. Era-stratified Δ −0.41.
   - Alt beta to BTC (> 1.372) shows the same thing with a cleaner, monotone dose-response: +0.42 / +0.38 / +0.03 / −0.08 / −0.12 by quintile.
     It points the same way in all three backtest eras (−0.05 / −0.08 / −0.12).
   - The correlation / dispersion ordering also ranks the three BACKTEST eras correctly:
     2025 corr 0.48 → −0.19 %/fill; 2026-H1 0.30 → −0.09; 2026-H2 0.28 → −0.02.
   - **But it is not a usable switch (yet):**
     - in DAY units it is weak: day ρ −0.21, rotation p 0.38, and within an era −0.03;
     - the "bad" side still breaks even rather than loses (WR 62 % ≥ 61.5 %, P(avg<0) 0.80);
     - it is basically one regime change;
     - **all 9 B25 fills were taken in high-correlation tape and 7 of them won** (+0.16 %/fill).
4. **Inside the bad regime, no entry variable separates winners from losers beyond chance.** This was checked three ways:
   - the 1D screen inside each of 3 regime splits;
   - the exhaustive 2D screen: 9.8 k–11.9 k quadrant tests per bad side, with p<0.01 counts *below* the shifted-label null;
   - the regime × entry interaction test.

   Three regime-conditional zones pass the expectancy bar in-sample, but a same-statistic null shows that **random slices of a losing regime pass the bar just as often**: 69 cohorts pass on the real data vs a median of 92 under the within-side null, P 0.90.
5. **A warning sign against "pure market regime":** today's engine replayed on the same calendar does **not** lose in the bad windows.
   - Replay: G5 +0.06 %/fill, G6-to-10-11 +0.23.
   - Master: −0.08 and +0.07 on 12 fills.
   - Master bad-era fills that the replay also took earn **+0.13** (27 fills; the replay prices them at +0.18).
   - The 13 bad-era fills the replay did not take lost **−0.38**; 54 % of them never went green.
   - The replay journal shows a gate failing on the closed bar at each of those moments.
   - So part of the bad-era loss looks like **live entries on borderline / forming-bar gate passes in a correlated tape**. That is a lead, not a result (13 fills).
6. **Recommendation:** no regime switch and no regime-conditional filter.
   - Three frozen observe lines start on fills opened ≥ 2026-10-27: OBS-R1 / R1b, OBS-R2 and OBS-C1, defined in §8.
   - The ML-OFF decision stays with the existing DL 289 kill gate.
   - The one new lever worth building is the "closed-bar gate re-check" scout (§9).
   - **✝ RETIRED 2026-10-26 (DL 382):** the 13-fill lead was tested twice and does not hold — the closed-bar re-check failed its parity (`reports/ML_CLOSEDBAR_REFUTED_PARITY_2026-10-26.json`) and the live-vs-replay tag failed its machinery parity because the replay's per-trade selection is path noise (`reports/ML_LIVE_VS_REPLAY_REFUTED_PARITY_2026-10-26.json`).

---

## 1 · Theory and pre-registered predictions (full text in `_PREREG.md`)

| # | mechanism (first principles) | predicted | result on master KEEP (median split unless stated) | verdict |
|---|---|---|---|---|
| T1 | Trend vs chop: a breakout in a range is faded | BTC efficiency / HTF slope ↑ → better | eff_24h Δ +0.10 p 0.44 · eff_72h −0.09 · eff_6h −0.01 · 4h EMA20 slope +0.15 p 0.39 · day ρ ≈ 0 | **REFUTED (PA)** — no chop effect |
| T2 | Drawdown / bounce: near the 30 d high = crowded, worse | near the high → worse | `btc_offhi_30d` > −3.3 % (near the high): **+0.22 vs +0.08** (opposite); deep (≤ −5.8 %) also +0.23 → non-monotone; washed-out only in BASE | **REFUTED (PB)** |
| T3 | Alt rotation: alts > BTC → better | alt−BTC 7 d > 0 → better | +0.13 vs +0.12 (Δ 0.01); ETHBTC 7 d up → *worse* (−0.15, p 0.44) | **REFUTED (PD)** |
| T4 | Breadth broad → better | above-EMA50-1h / adv 24h ↑ → better | +0.16 (p 0.32) / +0.17 (p 0.18), 6/7 era groups same sign, day-level ≈ 0 | right sign, **not significant** |
| T5 | Correlation / beta / dispersion: alt pops in a BTC-beta tape are noise | corr / beta ↑ → worse; dispersion ↑ → better | **corr Δ −0.40 (p 0.001)**, beta −0.40 (p 0.02), dispersion +0.22 (p 0.21); era-level order holds in master AND backtest | **CONFIRMED in direction (PC)** — fill-level and era-level; NOT at day level |
| T6 | Vol regime: extremes worse | extreme rv percentile → worse | rv24 percentile flat (−0.03); **7 d BTC vol high → better** (+0.30, p 0.02, 0 never-green losers in the top tercile, 53 fills); backtest flat | non-monotone threshold; backtest refutes |
| T7 | Liquidity / weekend worse | weekend / low volume → worse | weekend **+0.09** (opposite); top-50 volume vs 7 d high → −0.12 (p 0.36) | **REFUTED** |
| T8 | Funding build-up worse | high funding → worse | −0.15 (p 0.48, N 23); funding coverage ends 10-02 (0 of G6 / B25) | **UNTESTABLE (PE)** |
| T9 | Persistence (variance ratio / autocorrelation) | > 1 / > 0 → better | VR15 7 d +0.05, AC5 3 d +0.11, all p > 0.4 (as Oct-16) | **REFUTED** (again) |
| T10 | BTC drift into the entry | up → better | BTC 24 h > median +0.22 (p 0.11, 6/7 groups); 1 h / 4 h ≈ 0 | right sign for 24 h (= BTC24H_DOWN line), not significant |

## 2 · Regime variables and coverage

- **80 rebuilt per-fill variables** from closed 5m bars (`scripts/study_ml_regime_wl_build.py`):
  - BTC: returns 1h / 4h / 24h / 3d / 7d / 30d; distance from the 24h / 7d / 30d high and low; efficiency 6h / 24h / 72h (engine formula); above72; realised vol 1h / 24h / 7d + 90-day percentile + ratio; qvol ratio; variance ratio and autocorrelation over 3d / 7d; RSI / ADX / ADX change / DI spread / EMA20 slope / close-vs-EMA20 / EMA20-50 / EMA50-200 on 5m / 1h / 4h / 1d.
  - Top-50 (previous-day list): share above 5m EMA20 and 1h EMA50; advance share 1h / 24h; median return 1h / 24h / 7d; dispersion 1h / 24h; 24 h pairwise correlation (top-20); median beta.
  - Alt index − BTC 24h / 7d; ETHBTC 24h / 7d; volume ratio; funding (median / 3-day / positive share).
  - Hour / weekday / weekend.
- **Plus:** 25 entry_* BTC / macro stamps, 11 regime-label dummies, and 47 Oct-16 day-level features (`ml_regime/day_features.csv`, ≤ 10-16).
- **Validation vs engine stamps** (where stamped):

  | rebuilt | engine stamp | corr | median gap |
  |---|---|---|---|
  | eff72 | `entry_btc_eff72` | 0.999 | 0.000 |
  | above72 | `entry_btc_above72_pct` | 1.000 | — |
  | r72 | `entry_btc_r72_pct` | 1.000 | — |
  | off30d | `entry_btc_off30d_high_pct` | 0.999 | — |
  | off24h hi / lo | `entry_btc_off24h_pct` / `entry_btc_off24lo_pct` | 0.998 / 0.999 | — |
  | ADX 5m | `entry_btc_adx` | 0.985 | — |

  - Live RSI / trend-gap differ (corr 0.76 / 0.22) because the engine stamps the forming bar. Both versions were screened.

**Coverage** (`_coverage.csv`, KEEP-coverage column):

| variable family | G0 | G1 | G2 | G3 | G4 | G5 | G6 | B25 |
|---|---|---|---|---|---|---|---|---|
| 80 rebuilt BTC / breadth / alt / time vars (all but funding) | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 |
| rebuilt top-50 funding (3 vars) | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | **0/37** | **0/9** |
| stamps eff72 / above72 / r72 / off30d / off24h | 0 | 0 | 0 | 0 | 2–9/23 | 37/37 | 37/37 | 9/9 |
| stamps BTC 1d ret / EMA50-100 / ETH 5m ret | 0 | 0 | 0 | 0 | 0 | 7/37 | 37/37 | 9/9 |
| `entry_btc_rsi_closed` | 0 | 0 | 0 | 0 | 0 | 0 | 37/37 | 9/9 |
| Oct-16 day set (persistence 19 vars) | full | full | full | full | full | full | 29/37 | 0/9 |
| Oct-16 day set (LITE regime 28 vars) | full | full | full | full | full | full | 20/37 | 0/9 |
| impulse `imp_corr15` (fill-level) | full | full | full | full | full | full | 19/37 | 0/9 |

Recent-only stamps were not trusted for regime claims, because a recent-only stamp is perfectly confounded with "bad era". Their kline rebuilds are full-coverage and were used instead.

## 3 · Tape context per era group (checklist ④; medians at entry; `_tape.csv`)

| | G0 BASE | G1 B1 | G2 B2 | G3 B3 | G4 B4–B11 | G5 B12–B16 | G6 B17–B24 | B25 |
|---|---|---|---|---|---|---|---|---|
| span | 06-18→07-10 | 07-11→07-30 | 08-01→08-10 | 08-11→08-24 | 08-24→09-20 | 09-21→10-02 | 10-05→10-20 | 10-24→10-26 |
| KEEP | 32·94%·+0.49 | 15·80%·+0.26 | 10·80%·+0.19 | 22·77%·+0.15 | 14·79%·+0.06 | 28·57%·−0.08 | 37·49%·−0.09 | 9·78%·+0.16 |
| never-green losers % of fills | 3 | 13 | 10 | 9 | 14 | **32** | **35** | 11 |
| BTC over the span / max DD | −0.7 % / −11.3 | +1.1 / −6.0 | +1.6 / −2.4 | **+23.5** / −4.3 | +4.6 / −8.6 | +3.8 / −5.3 | −4.0 / −8.1 | −0.7 / −2.3 |
| BTC 30 d return | −17.6 | +3.5 | +2.0 | +16.8 | +21.8 | +8.0 | +7.2 | −4.8 |
| BTC off 30 d high | −18.7 | −3.9 | −3.8 | −3.5 | −2.5 | −3.1 | −5.5 | −8.1 |
| BTC 1d EMA50 vs EMA200 % | −12.5 | −12.0 | −11.1 | −8.7 | −0.3 | **+3.6** | **+6.1** | +6.0 |
| BTC efficiency 24 h | 0.062 | 0.045 | 0.052 | 0.053 | 0.050 | 0.060 | 0.039 | 0.061 |
| BTC 24 h rv percentile (90 d) | 70 | 42 | 29 | 53 | 78 | 74 | 56 | 43 |
| top-20 24 h correlation | 0.29 | 0.23 | 0.15 | 0.34 | 0.45 | **0.42** | **0.42** | **0.51** |
| median alt beta to BTC | 1.11 | 1.13 | 0.84 | 1.44 | 1.53 | **1.75** | 1.44 | 1.59 |
| alt 24 h dispersion | 9.6 | 9.7 | 14.9 | 10.5 | 8.8 | 8.4 | **6.0** | 7.5 |
| alt−BTC 7 d | +1.3 | −0.1 | +1.3 | +3.1 | +4.5 | +10.2 | −2.1 | −6.3 |
| top-50 above 1h EMA50 % | 41 | 35 | 53 | 53 | 64 | 60 | 40 | 13 |

**Reading.**
- The good eras ran under a BTC macro downtrend with a bounce (BASE was the washed-out capitulation) and low alt correlation, so alts moved on their own.
- The bad eras ran with BTC in a 1-day uptrend and alts as high-beta copies of BTC.
- B25 is the most correlated tape so far (0.51) and won 7 of 9. That is 3 days, one window.
  **Correction (DL 382, 2026-10-26):** the B25 column used the stale 10-11 list (the cache list ends 10-11; the build falls back to the latest list). On the correct lists (dated D − 1, ranked on D − 2 volume, full universe) B25 corr24 is 0.30–0.44 (median 0.41), beta 1.54–1.78, rv7 0.097–0.100 — high but not "the most correlated tape"; 8 of 9 fills are in the R1 zone (ADA 10-25 at 0.298 is out).

## 4 · Uniform-degradation test (checklist ③) — `_uniform.csv`

| dimension | cohort | good G0–G3 | bad G5–G6 | Δ | never-green losers good→bad (% of fills) |
|---|---|---|---|---|---|
| ALL | — | 79·85%·+0.31 | 65·52%·−0.09 | −0.40 | 8→34 |
| confidence | STRONG_BUY / VERY_STRONG | +0.28 / +0.34 | −0.04 / −0.13 | −0.32 / −0.47 | 10→28 / 5→39 |
| regime stamp | HEALTHY_BULL / STRONG_BULL | +0.24 / +0.46 | −0.15 / +0.02 | −0.39 / −0.44 | 10→32 / 4→38 |
| pattern | W-match / no-W | +0.35 / +0.31 | −0.15 / −0.06 | −0.50 / −0.37 | 11→50 / 7→28 |
| pair 24 h vol tercile | low / mid / high | +0.19 / +0.29 / +0.52 | −0.08 / −0.02 / −0.15 | −0.27 / −0.30 / −0.66 | 9→18 / 9→43 / 5→37 |
| pair rank | ≤ 25 / > 25 | +0.45 / +0.20 | −0.10 / −0.08 | −0.54 / −0.29 | 11→33 / 5→34 |
| ATR % tercile | low / mid / high | +0.28 / +0.48 / +0.24 | +0.07 / −0.15 / −0.16 | −0.21 / −0.62 / −0.41 | 10→37 / 0→31 / 10→35 |
| pair ADX direction | rising / flat-falling | +0.30 / +0.33 | −0.08 / −0.11 | −0.38 / −0.43 | 8→38 / 7→26 |
| pair ADX tercile | low / mid / high | +0.44 / +0.09 / +0.44 | +0.00 / +0.01 / −0.28 | −0.44 / −0.07 / −0.73 | |
| RSI tercile | low / mid / high | +0.50 / +0.24 / +0.23 | −0.13 / −0.17 / +0.17 | −0.63 / −0.42 / −0.06 | 5→44 / 10→32 / 8→15 |
| hour UTC | 00–07 / 08–15 / 16–23 | +0.29 / +0.29 / +0.35 | −0.15 / −0.06 / −0.03 | −0.44 / −0.35 / −0.39 | |
| weekday / weekend | | +0.33 / +0.25 | −0.11 / +0.08 | −0.44 / −0.17 | |
| same-scan | multi / single | +0.20 / +0.34 | +0.03 / −0.14 | −0.17 / −0.48 | 11→11 / 7→43 |

- **31 of 31 sub-cohorts** (n ≥ 5 on both sides) degrade.
- The bad-era loss is spread across pairs: the top pair (UNI) carries 6.6 % of the bad-era gross loss, and the top 8 pairs together 45 %.
- Mild heterogeneity, all within noise at these N: high-RSI and multi-fill scans degrade least; high pair ADX, low RSI and high pair-volume degrade most.
- **Verdict: UNIFORM.** The sleeve did not break in one cell. The whole sleeve lost its edge at once.

## 5 · Regime separators

**Screen** (`_regime_screen.csv`, fill level):
- 163 regime variables, sign → median → terciles: 524 zone tests.
- p<0.05: 23, against ≈ 26 expected by chance. Family-wise p<0.05: 0.

**DAY units** (`_regime_day.csv`): per-day mean outcome vs per-day mean variable, Spearman, every day rotation as the null.
- 150 variables. p<0.05: 9, against 7.5 expected.
- **At chance.**

**Never-green-loser outcome** (`_ng_screen.csv`): 522 tests. p<0.05: 35 (26 expected). Family-wise: 0.

**Era level** (`_era_level.csv`): 7 era-group medians vs group outcome, with the 3 backtest eras as refute-only.

| variable | era ρ, master (7 groups) | era ρ, backtest (3 eras) | note |
|---|---|---|---|
| corr24h_top20 | −0.71 | **−1.0** | both say "high corr = bad" |
| disp_24h | +0.71 | **+1.0** | both say "low dispersion = bad" |
| BTC 1d EMA50-200 | −1.00 | −0.5 | does not survive the backtest |
| BTC 1d EMA20-50 | −0.89 | +0.5 | does not survive the backtest |

Top separators (KEEP N = 158; ①–⑤ = fill Δ / era-stratified Δ and groups / LOEO / day ρ (within-era) / backtest Δ 2025 · H1 · H2):

| zone (bad side) | zone vs rest | ① Δ, p | ② era-strat | ③ LOEO | ④ day ρ | ⑤ backtest | gap absorbed |
|---|---|---|---|---|---|---|---|
| **corr24h_top20 > 0.3164** (PC) | 79·62%·−0.077 vs 79·80%·+0.326 | −0.40, 0.001 | −0.41 (4/5) | 4/5 | −0.21 (−0.03) | +0.05 · −0.07 · −0.13 | 47 % |
| beta_btc_24h > 1.372 | 79·62%·−0.075 vs 79·80%·+0.324 | −0.40, 0.020 | −0.31 (5/6) | 5/5 | −0.34 (−0.07) | −0.05 · −0.08 · −0.12 | 40 % |
| disp_24h ≤ 8.737 | 79·61%·+0.012 vs 79·81%·+0.237 | −0.22, 0.21 | −0.01 | 4/6 | +0.18 (+0.09) | −0.02 · 0.00 · +0.04 | 10 % |
| BTC rv 7 d ≤ 0.1122 | 79·59%·−0.028 vs 79·82%·+0.277 | −0.30, 0.022 | −0.25 (3/4) | 3/4 | +0.34, p 0.045 (+0.19) | −0.01 · 0.00 · −0.03 | 21 % |
| BTC 1d EMA20−50 > 2.49 % | 75·55%·−0.072 vs 83·86%·+0.302 | −0.37, 0.019 | only 1 group has both sides | 0/0 | −0.45, p 0.015 (−0.15) | +0.01 · +0.10 · −0.02 | 55 % |
| BTC 30 d return > 0 | 121·66%·+0.045 vs 37·86%·+0.384 | −0.34, 0.07 | +0.03 | 1/3 | −0.33, p 0.03 (−0.03) | −0.02 · +0.01 · −0.13 | 20 % |
| STRONG_BULL (seen, winner) | 53·77%·+0.279 vs 105·68%·+0.047 | +0.23, 0.06 | +0.33 (6/7) | 6/7 | +0.17 (+0.25) | +0.06 · +0.12 · +0.06 | — |

- Quintile dose-response, corr: +0.20 / +0.46 / +0.21 / **−0.20** / −0.05. It steps down above about 0.39 rather than falling smoothly.
- Quintile dose-response, beta: +0.42 / +0.38 / +0.03 / −0.08 / −0.12. **Monotone.**
- Within the bad eras almost every fill is high-correlation (G5 26/28, G6 28/37), so the within-era evidence comes mostly from G4 and G5.
- **Read.** The only regime family that is consistent across master and the backtest is **correlation / beta / dispersion (T5)**. It acts as a slow, era-level variable and does not work as a day-level switch.
- The BTC-macro-trend variables are the same single regime change described in a different way. They carry no within-era information and the backtest refutes them.

**Same-calendar replay** (`_replay_calendar.csv`, `_replay_match.csv`). The question: did today's engine, replayed on the same days, also lose?

| window | master KEEP | replay, 3 seeds, today's config |
|---|---|---|
| G0 | +0.485 | +0.089 |
| G1 | +0.256 | −0.118 |
| G2 | +0.191 | +0.001 |
| G3 | +0.147 | +0.124 |
| G4 | +0.060 | −0.114 |
| **G5** | **−0.083** | **+0.062** |
| G6 to 10-11 | +0.068 | +0.231 |

- The master and replay era profiles are uncorrelated (ρ −0.39 across the 7 groups).
- Master KEEP fills that a replay fill also took (same pair, ±10 min):

  | eras | matched fills | master | replay's price for the same fills | unmatched fills |
  |---|---|---|---|---|
  | good G0–G3 | 42 | +0.30 | +0.23 | 37 · +0.32 |
  | bad G5–G6 (to 10-11) | 27 | **+0.13** | +0.18 | **13 · 38 % · −0.38**, 54 % never-green |

- At every unmatched moment the replay journal logged closed-bar gate failures (`_unmatched_journal.csv`): BTC_ACCEL_CHASE_LONG, BTC_RSI_ADX_CROSS, PAIR_EMA_GAP_NOT_EXPANDING, RSI_MOMENTUM_LOADX.
- The same gate mix appears behind the good-era unmatched fills, and those won.
- This is the only place where bad-era losers concentrate. To 10-11 the bad-era net is −1.5 %-sum: the 13 unmatched fills sum −4.9 % and the 27 matched fills sum +3.4 %.
- N 13 → **lead only**.

## 6 · Within-regime entry winners vs losers (step 4)

Three regime splits:
- **A — HICORR:** corr24 > 0.3164. This is the pre-registered winner of step 3.
- **C — LATE ERA:** G5–G6. Observational: it describes the bad regime but cannot be known in advance.
- **D — LOWRV7:** BTC 7 d rv ≤ 0.1122, the best day-level variable.

Screened variables, inside each side:
- 155 screenable entry variables: every dashboard W/L column (RSI, ADX, ADXDir, RSIDir, Gap, Gap5-8, Gap8-13, Stretch, EMA20Slp, ADXΔ, RngPos, **ATR%**, PairTrend, PairVol$, Ext%, 1hSlope, BTCSlp, BTCRSI, BTCADX, BTCADXDir, BTCRSIDir, Breadth, BTCTrend, GlobalVol, BTCExt%, **Conf**, **FanR**, BTC ATR%, EMA5 dist);
- every entry_* stamp;
- every PREP feature (impulse, pair-strength, cooldown, burst / scan, hour / day).

| split · side | N · days | 1D tests · p<.05 · fw | 2D tests (full-cov vars) | 2D p<.01 observed vs shifted-label null median / p95 | 2D fw |
|---|---|---|---|---|---|
| A HICORR · bad | 79·62%·−0.077 · 33 d | 409 · 15 · 0 | 9,777 (71) | 112 vs 173 / 335 | 0 |
| A HICORR · good | 79·80%·+0.326 · 40 d | 353 · 19 · 0 | 22,309 (107) | 349 vs 362 / 891 | 0 |
| C LATE · bad | 65·52%·−0.087 · 22 d | 408 · 23 · 1 (`ps_btc_d50`, N 6 on 3 days — coverage artefact) | 11,879 (78) | 148 vs 234 / 596 | 0 |
| C LATE · good | 93·84%·+0.272 · 49 d | 352 · 8 · 0 | 22,435 (107) | 167 vs 308 / 731 | 0 |
| D LOWRV7 · bad | 79·59%·−0.028 · 35 d | 412 · 22 · 0 | 9,816 (71) | 201 vs 196 / 342 | 0 |
| D LOWRV7 · good | 79·82%·+0.277 · 36 d | 362 · 22 · 0 | 22,338 (107) | 257 vs 339 / 656 | 0 |

**Regime × entry interaction** (`_interaction.csv`): the entry zone's Δ in the bad regime minus its Δ in the good regime.

| split | tests | p<0.05 | family-wise |
|---|---|---|---|
| A HICORR | 356 | 16 | 0 |
| C LATE ERA | 344 | 9 | 0 |
| D LOWRV7 | 369 | 21 | 0 |

All three are at chance (≈ 18 expected per split).

**Same-statistic null for "a regime-conditional filter passes the bar"** (`study_ml_regime_wl_barnull.py`):
- 1,025 cohorts (bad side ∧ entry zone, N ≥ 15). **69 pass** the expectancy bar on the real outcome.
- Against the shifted-label null, 69 looks high: median 6, P 0.02. That null is wrong for this question, because it also destroys the losing level of the bad side.
- Against the right null — rotate outcomes *inside* the bad side, so the side keeps its losing mean — the median is **92**, the 95th percentile 115, and P(null ≥ 69) = **0.90**.
- **Slicing a losing regime at random passes the bar as often as the best entry variables do.** No regime-conditional entry filter exists in this data.

The one recurring signal is the **fill-level 15-minute top-10 correlation `imp_corr15`**, inside every bad side:

| side | high-corr15 zone | low-corr15 rest |
|---|---|---|
| HICORR | 30·43%·−0.35 | 31·77%·+0.17 |
| LATE ERA | 23·26%·−0.45 | 24·67%·+0.24 |
| LOWRV7 | 20·25%·−0.46 | 41·73%·+0.20 |

- Inside the good sides it is flat. The interaction is −0.75 (p 0.001, family-wise 0.85).
- It is the same mechanism as T5, at the entry minute rather than the day. It is already registered (PREP P6 `corr15 > 0.365`).
- It is not engine-stamped and its coverage stops at 10-15.

## 7 · Candidates vs the expectancy bar (step 5) — `_candidates.csv`, `_before_after.csv`

ML breakeven WR is 61.5 %. Windows are days. The cuts are frozen pooled KEEP cuts. Backtest = 2025 · H1 · H2, zone vs rest.

| candidate | blocked cohort (KEEP) | WR < BE | P(avg<0) | windows · scan windows | worst day / pair % | bar | KEEP before → after (Δ, haircut 30–50 %) | TRAD zone | B25 zone | backtest | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **R1 HICORR switch** (corr24 > 0.3164) | 79·62%·−0.077 | ✗ (62.0) | 0.80 ✗ | 33 · 68 | 16 / 8 | FAIL | +0.124 → +0.326 (+0.20 → +0.10…+0.14) | 100·60%·−0.13 | **9·78%·+0.16** | +0.05 · −0.07 · −0.13 | **KEEP OBSERVING** (OBS-R1) |
| R1b beta alias (> 1.372) | 79·62%·−0.075 | ✗ | 0.82 ✗ | 34 · 69 | 11 / 8 | FAIL | +0.124 → +0.324 | 100·61%·−0.11 | 9·78%·+0.16 | −0.05 · −0.08 · −0.12 | read with R1 |
| R2 LOWRV7 switch | 79·59%·−0.028 | ✓ | 0.60 ✗ | 35 · 71 | 16 / 9 | FAIL | +0.124 → +0.277 | 109·58%·−0.07 | 9·78%·+0.16 | flat · flat · flat | **KEEP OBSERVING** (OBS-R2, descriptive) |
| R3 BTC 1d macro uptrend switch | 75·55%·−0.072 | ✓ | 0.72 ✗ | 30 · 65 | 15 / 7 | FAIL | +0.124 → +0.302 | 94·51%·−0.15 | 0 | +0.01 · +0.10 · −0.02 | **DECIDED: no** — time-confounded, 1 episode, backtest says no |
| C1 HICORR ∧ corr15 > 0.365 | 34·44%·−0.324 | ✓ | 0.985 ✓ | 20 · 33 | 13 / 12 | in-sample PASS | +0.124 → +0.247 (+0.12 → +0.06…+0.09) | 51·51%·−0.29 | 0 (unscored) | n/a | **KEEP OBSERVING** (OBS-C1); ships only via the forward cohort, see below |
| C2 LOWRV7 ∧ pair EMA20 slope ≤ 0.1667 | 56·54%·−0.184 | ✓ | 0.95 ✓ | 29 · 50 | 19 / 11 | in-sample PASS | +0.124 → +0.294 | 79·52%·−0.20 | 8·75%·+0.07 | flat · flat · −0.12 | **DECIDED: rejected** — within-side null P 0.90; backtest flat in 2/3; B25 contradicts |
| C3 LOWRV7 ∧ Ext% ≤ 0.2718 | 30·43%·−0.264 | ✓ | 0.986 ✓ | 18 · 29 | 15 / 15 | in-sample PASS | +0.124 → +0.216 | 48·44%·−0.26 | 3·67%·+0.27 | **+0.07 · +0.04** · −0.10 | **DECIDED: rejected** — backtest opposite in 2/3 eras, within-side null |
| Primaries PA / PB / PD | — | — | — | — | — | — | — | — | — | — | **DECIDED: refuted** (§1) |
| PE funding | — | — | — | — | — | — | — | — | — | — | could not test (§10) |

R1 before → after per era group (KEEP):

| group | before | after (zone removed) | zone |
|---|---|---|---|
| G0 | +0.485 | +0.451 | 5 fills · +0.67 |
| G1 | +0.256 | +0.256 | 0 |
| G2 | +0.191 | +0.191 | 0 |
| G3 | +0.147 | +0.163 | 9 · +0.12 |
| G4 | +0.060 | +0.501 (3 fills) | 11 · −0.06 |
| G5 | −0.083 | +1.198 (2 fills) | 26 · −0.18 |
| G6 | −0.091 | +0.200 (9 fills) | 28 · −0.18 |
| **B25** | +0.156 | **nothing left** | 9 · **+0.16** |

- A correlation switch would have turned ML off for almost all of the bad eras **and for all of B25**, which is winning.
- As a sleeve switch its evidence counts in windows. That is ≈ 2 regime episodes (pre-09-21 low correlation / post-09-21 high), far below any gate.

C1 per era group (zone): G3 5·100%·+0.46 · G4 7·71%·−0.15 · **G5 14·29%·−0.61** · G6 8·12%·−0.47.
- G5 carries 41 % of the zone fills. Without G5 the zone is 20 fills at about −0.13, which is not significant.

## 8 · FROZEN observe lines (thresholds fixed 2026-10-26, never re-tuned)

**Cohort:** MOMENTUM LONG, CLOSED, non-probe, non-MANUAL, opened **≥ 2026-10-27 00:00 UTC**. % at 1×. DAY units: a day is one window.
The 9 B25 fills are SEEN and reported apart.

| id | definition | side | data | decision rule |
|---|---|---|---|---|
| **OBS-R1** ML_HICORR_REGIME | mean pairwise Pearson correlation of 5m log returns over the trailing 288 closed bars among the top-20 of the list DATED D − 1 (= every USDT perpetual ranked on its day D − 2 quote volume, `universe_daily_top50.csv` convention; ex BTC / stables) for a fill on UTC day D **> 0.3164**. Read with alias **R1b**: median alt beta to BTC over the same bars **> 1.372** | loser regime | BTC + top-20 5m klines (`study_ml_regime_wl_build.py`) | SWITCH-CANDIDATE needs **all** of: zone N ≥ 15 on ≥ 8 days ∧ WR < 61.5 % ∧ day-boot P(avg<0) ≥ 0.95 ∧ no day / pair ≥ 50 % of the zone loss ∧ ≥ 3 **separate regime episodes** (episodes split by ≥ 5 consecutive outside-zone noon readings, corr ≤ 0.3164, between two zone-fill days). RETIRE at N ≥ 15 if zone avg ≥ 0. Read once at N = 15 and once at N = 30. Current tape is in the zone (B25 corr 0.42–0.54 on the stale 10-11 list; 0.30–0.44 on the correct lists, 8 of 9 in the zone — DL 382), so this line is effectively "is ML net-positive in a correlated tape?" and is read jointly with the DL 289 kill gate. Evidence is not double-counted. |
| **OBS-R2** ML_LOWRV7 | BTC 7-day realised vol = std of 2016 closed 5m log returns × 100 **≤ 0.1122** | loser regime (descriptive) | BTC 5m | Same bar as R1 (episodes on rv7 > 0.1122 noon readings). RETIRE at N ≥ 15 if zone avg ≥ 0. |
| **OBS-C1** ML_HICORR_CORR15 | R1 zone ∧ `corr15 > 0.365` (Oct-15 impulse-grid definition, PREP P6) | loser, regime-conditional | impulse grid rebuild (`study_ml_impulse_fetch.py` + `_features.py`) | Expectancy bar on forward fills only. P6 and C1 share fills, and a fill counts once. If C1 passes and P6 outside HICORR is ≥ 0, the conditional form is the one to promote, at OPERATOR CALL, observe-first. |

## 9 · What this means for the operator, and the next step

- **The ML loss in B12–B24 does not come from a bad entry rule.**
  - It is the whole sleeve losing its edge in a tape where alts move as high-beta copies of BTC.
  - It may also partly come from *which* moments the live engine fired on: borderline, forming-bar gate passes.
- **Nothing measurable before the entry separates the losers inside that tape** beyond what random slicing does.
- **No switch can be justified on ≈ 2 regime episodes.** B25 has already traded a highly correlated tape (corr 0.30–0.44 on the correct lists) and won 7 of 9.
- **Highest-value next build: a "closed-bar gate re-check" scout.**
  - At each live ML fill, record whether every gate also passes on the last *closed* 5m bar. This is what the replay does.
  - That tests the matched / unmatched lead on forward fills, cheaply, with no rule change.
  - The design and gate need the operator's OK. It is not built here.
  - **✝ RETIRED 2026-10-26 (DL 382):** built, tested twice and retired — see `reports/ML_CLOSEDBAR_REFUTED_PARITY_2026-10-26.json` and `reports/ML_LIVE_VS_REPLAY_REFUTED_PARITY_2026-10-26.json`. The regime lines live on in `scripts/scout_ml_regime_lines.py`.
- This study does not touch the DL 289 kill gate, BTC24H_DOWN, the ADX lines or PAIR_1H_DOWNTREND. Those keep their own frozen reads.

## 10 · What we could NOT test (blind spots)

1. **Open interest, long / short ratio, liquidations, order-book depth, taker flow:** not cached. The public OI / ratio history only reaches 30 days back, so it would cover G6 / B25 alone and was not fetched.
2. **Funding:** BTC / ETH funding is not cached. The alt funding cache ends 2026-10-02, so G6 and B25 are unscored. T8 / PE is untestable on the losing era.
3. **Implied volatility (DVOL), BTC dominance index, stablecoin flows, cross-exchange premium, news / listing / macro-calendar (CPI / FOMC) events:** not available. Alt−BTC and ETHBTC proxies were used for dominance.
4. **Regime episodes ≈ 2.** The co-moving regime variables (correlation, beta, dispersion, BTC macro trend, 7 d vol) cannot be separated from each other or from "time" (competition, microstructure, live-engine changes) with this history. Era-level ρ uses 7 points, and the backtest era check uses 3.
5. **Recent-only stamps** (eff72, off24h, r72, off30d, rsi_closed, BTC 1d ret, ETH 5m ret) are confounded with the bad era. They were replaced by full-coverage kline rebuilds.
6. **`imp_corr15` and the Oct-16 day set stop at 10-15 / 10-16** (19/37 G6, 0/9 B25). The top-50 universe list is stale after 10-11, by up to 15 days.
7. **Backtest:**
   - it is today's config, and it cannot show trades today's rules block;
   - it ends 10-11;
   - corr15 / impulse / pair-strength features do not exist in it.

   It was used to refute only.
8. **Not exhaustive:** 3-way interactions, 2D cuts other than median × median, and thresholds other than sign / median / tercile.
9. **Power:** 158 KEEP fills over 71 days, with each regime side ≈ 33–40 days. A true +0.2 %/fill regime effect inside one era is below the detectable size.
10. **The matched / unmatched lead rests on 13 fills.** The cause of each unmatched fill was inferred from the replay journal; live forming-bar values were not reconstructed.

## 11 · Files

- **Scripts** (all new, read-only on inputs):
  - `scripts/study_ml_regime_wl_build.py` — per-fill regime features, master + replay;
  - `_analyse.py` — coverage, uniform, tape, regime screen, day level;
  - `_deep.py` — panel, never-green screen, era level, within-regime 1D / 2D, interaction;
  - `_barnull.py` — same-statistic null;
  - `_candidates.py` — bar and before → after;
  - `_replaycal.py` — same-calendar replay and the matched / unmatched split.
- **Outputs** (`reports/ML_REGIME_WL_2026-10-26*`):
  - `_PREREG.md`
  - `_fill_regime.csv`, `_replay_regime.csv` (**local only** — rebuild: `study_ml_regime_wl_build.py`)
  - `_coverage.csv`, `_uniform.csv`, `_tape.csv`
  - `_regime_screen.csv`, `_regime_day.csv`, `_regime_panel.csv`, `_era_level.csv`, `_ng_screen.csv`
  - `_within_1d.csv`, `_within_2d.csv.gz`, `_within_meta.csv`, `_interaction.csv`, `_barnull_cohorts.csv`
  - `_candidates.csv`, `_before_after.csv`
  - `_replay_calendar.csv`, `_replay_match.csv`, `_unmatched_journal.csv`
  - `_D.pkl` (analysis frame — **local only**, rebuild: `study_ml_regime_wl_analyse.py`)
- **Rerun:** `build` → `analyse` → `deep` → `barnull` / `candidates` / `replaycal`, each with `venv/bin/python -W ignore`, run from `scripts/` for the last four.
