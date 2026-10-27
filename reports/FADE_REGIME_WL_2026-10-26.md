# SPIKE_FADE — winners vs losers BY MARKET REGIME (2026-10-26)

Research study. Its frozen observe line OBS-F1 ships as an observe-only scout (`scripts/scout_fade_late_weak.py`, DECISION_LOG 384); no engine / config / template change, no bot API call.
Binance: public endpoints only, ≈ 600 weight in total over ~3 min (71 5m-kline pairs, 134 one-hour 1m windows, 99 funding histories),
used-weight header read on every call, no 418 / 429, cooldown file honoured. Everything else came from the existing caches.
Pre-registration `reports/FADE_REGIME_WL_2026-10-26_PREREG.md` (SHA-256 `6be24994e80e4f7f…`) was frozen before any
regime-variable × outcome table existed. `scripts/validate_against_master.py` → **ALL CHECKS PASS** (run first).

**Cohort.**
- **Master:** 76 SPIKE_FADE fills from `MASTER_POOL_stacked.csv`, kept by today's stack, non-probe, outcome `stack_pct`.
  - 45 UTC days, B1–B23.
  - `stack_pct` is a price %. The 2× invest multiplier never moves it, so every % below is already "at 1×".
- **B25 (provisional):** 8 fades from `scalpars_orders_paper_2026-10-26_15-11-28.csv`, CLOSED, deduplicated on (opened_at, pair, direction).
  - All 8 opened **before** the 10-26 13:50 UTC paper top-up, so they all sit on one side of the size boundary. Compare in %.
- **Stack-blocked master fades (50):** used only for the cohort-completeness check. They are as-traded, under old mechanisms.
- **Backtest (yr6, 3 seeds, 1,832 fades):** used only as a caveat.

**Breakeven WR** (today's stack, master kept fills) = **71.0 %**: avg win +0.562 / avg loss −1.374.
DL 374 used ≈ 72 %, so the two agree.

---

## Plain-language summary

1. **No market-regime variable explains fade wins vs losses beyond chance.**
   - The screen covered 117 market variables (82 rebuilt from closed klines + 35 engine stamps) in 360 zone tests.
   - Fill level: 12 tests at p < 0.05 against ≈ 18 expected by chance, and 0 survive the family-wise null.
   - Day level: 3 of 170 at p < 0.05 against ≈ 8 by chance.
   - The few "top" splits are calendar proxies. The best is BTC's 1-day EMA structure, which is simply "early eras were better".
   - In every case the backtest shows no effect, or the opposite one.
2. **Mirror of the ML study: pre-registered as "high alt correlation / beta helps fades". REFUTED.**
   - High beta (top-50 median beta > 1.03): 38·87 %·+0.19 vs 38·87 %·+0.43 (p 0.09).
   - High correlation: +0.23 vs +0.39.
   - So fades did *slightly worse* in the lock-step tape, the same sign as momentum longs, but not significant.
   - The backtest goes the other way on all 3 seeds (+0.03 … +0.11).
   - It is also basically one regime episode (09-21 →). **Not a switch.**
3. **The recent damage is near-uniform, which by checklist ③ points to an unmeasured tape variable.**
   - G6 + B25 (10-05 → 10-26): 23·74 %·−0.16 %/fill, against 61·89 %·+0.40 earlier.
   - 33 of 36 sub-cohorts degraded together, and the stop rate rose from 15 % to 26 %.
   - The only cohorts that held up are *vertical spikes on thin books*:
     - lowest 24 h-volume tercile: 7·100 %;
     - high pair-ADX tercile: 7·100 %.
4. **Inside the recent "late tape" (BTC 1d EMA50 above EMA200, since 09-13), weak spikes lose.**
   - Weak spike = +DI ≤ 48.5 and stretch above EMA13 ≤ 1.80 %.
   - Master + B25: 17·59 %·−0.44 vs 67·91 %·+0.42 for every other fade.
   - In the *early* tape the same weak spikes **won** (26·88 %·+0.33), so the effect is an interaction.
   - It passes the expectancy bar on master + B25: WR 59 % < 71 %, P(avg < 0) 0.978, 14 days, 15 pairs, top pair 14 %.
   - **Master alone does not pass** (12·67 %·−0.30, P 0.89).
   - The 2D excess over the null exists **only once B25 is added**. With B25: 366 cells at |t| ≥ 3 vs a null median of 87, P 0.000 by rotation / 0.007 by permutation. Master alone: 13 vs 30.
   - **The backtest refutes it on all 3 seeds.** In BTC-above-EMA200 tape (2025) weak spikes win: zone +0.05…+0.10 vs rest −0.02…+0.05.
   - → **KEEP OBSERVING** as a frozen line (OBS-F1, §8). Not a ship.
5. **The 41h HEALTHY_BEAR ∧ BTC-trend-gap ≥ −0.13 line is the only pre-registered regime prediction confirmed in the master.**
   - Master: 14·71 %·−0.12 vs 62·90 %·+0.40, p 0.0005, day Δ −0.52, held-out-era sign 4/4.
   - Forward since its 09-18 registration it kept going the same way: 9·67 %·−0.25 · −$196 on 9 days.
   - It still fails the expectancy bar on confidence: P(avg < 0) 0.79 on master + B25.
   - It does not meet its own frozen SHIP bar: N 16 / 20, WR 69 % vs ≤ 55 %.
   - The backtest contradicts it on all 3 seeds: zone +0.12…+0.22 *better*.
   - → **KEEP OBSERVING** (its own gate).
6. **DL 374 lines ①–⑥: KEEP OBSERVING.**
   - The forward cohort is still 0. The last fade (PARTI 13:19) opened before the commit + 10 min.
   - On the full data none passes the bar (table §7).
7. **Size.** The B25 loss is three full −1.5 % stops at 2× (−$754), not a regime this data can isolate.
   - At 2×, 8 master fills and 3 B25 fills each lost > 4 % of a $2,900 account. At 1× that would be 3 and 2.
   - No regime-conditioned size cut is supported:
     - every regime "bad side" is still positive in the master;
     - HEALTHY_BEAR is +0.13 in the master and is the BEST regime in the backtest.
   - The size question remains the stop-width × size question (plan §5), with ③ / ④ deciding forward.
8. **Surprises against theory (all n.s., for the record):**
   - Pair funding < 0 ("crowded shorts = squeeze fuel") was predicted to lose. Those fades **won**: 12·92 %·+0.74 vs +0.23.
   - Weekend fades did worse (+0.17 vs +0.37), not better.
   - Pre-spike runs (pair 4 h / 24 h up) did better, not worse.

**Bottom line: no DECIDED change and no new OPERATOR CALL. One new frozen observe line (OBS-F1); 41h and DL 374 ①–⑥ keep collecting.**

---

## 0 · Theory and pre-registered predictions → result

Mechanism (frozen in the PREREG). A fade shorts a +25-RSI wick on a thin alt. The edge is a 1–2 minute mean reversion.
Regime matters through three channels:
- (a) whether sellers are present after the spike;
- (b) beta drag, i.e. whether the market pulls the coin back down or up;
- (c) squeeze fuel.

Verdict rule:
- **CONFIRMED** = fill-level and day-level direction match, with p < 0.10 under the circular-shift null.
- **RIGHT SIGN** = both directions match, not significant.
- **REFUTED** = the opposite sign.
- **MIXED** = fill-level and day-level disagree.

Full table: `_prereg_score.csv`. Counts: CONFIRMED 2 · RIGHT SIGN 10 · MIXED 7 · REFUTED 27 · UNTESTABLE 1.

| # | prediction (better side) | master result (zone vs rest, %/fill) | BT Δ s1/s2/s3 | verdict |
|---|---|---|---|---|
| P1 | BTC 15 m / 1 h / 4 h down → better | 1 h up 14·86 %·+0.21 vs +0.33 · 4 h up +0.21 vs +0.35 · 30 m opposite | mixed | RIGHT SIGN (n.s., p ≥ 0.46) |
| P2 | BTC 24 h / 7 d / close-vs-EMA20 down → better | 24 h up 35·97 %·+0.41 vs +0.22; 4 h above EMA20 +0.42 vs +0.14 | +0.06…+0.23 | **REFUTED** (fades did better with BTC firm) |
| P3 | deep BTC drawdown → worse | off-30 d-high flat (+0.31 / +0.30); trend gap > 0: +0.27 vs +0.32 | ≈ 0 | REFUTED / flat |
| **P4** | **high alt corr / beta → better (mirror of ML)** | corr > 0.209: +0.23 vs +0.39 · beta > 1.03: +0.19 vs +0.43 (p 0.09) · the pair's own beta > 0.78: +0.16 vs +0.45 (p 0.08) | +0.02…+0.16 (opposite of master) | **REFUTED** (same sign as ML: lock-step tape slightly worse; BT disagrees) |
| P5 | high dispersion → worse | 1 h dispersion high +0.38 vs +0.24 | mixed | REFUTED |
| P6 | bearish breadth → better | share above 1h EMA50 low better (+0.38 vs +0.23); Bear % opposite | mixed | RIGHT SIGN (n.s.) |
| P7 | alts weak vs BTC → better | alt − BTC 7 d > 0: +0.36 vs +0.20 | −0.01…−0.19 | REFUTED |
| P8 | high BTC vol → worse | rv24 high +0.36 vs +0.25; BTC ATR high +0.40 vs +0.22 | ≈ 0 / − | REFUTED |
| P9 | weekend / quiet market → better | weekend 24·83 %·+0.17 vs +0.37 | +0.09…+0.16 | REFUTED |
| P10 | pair funding < 0 → worse | funding < 0: **12·92 %·+0.74** vs 64·86 %·+0.23 (p 0.08) | not scored | **REFUTED** (and 1 B25 fill, RAYSOL, went the predicted way) |
| P11 | news / real-move proxies → worse | pair 24 h volume vs 7 d > 0.82: 38·82 %·+0.15 vs 38·92 %·+0.47 (p 0.03, day Δ −0.32) **CONFIRMED** · pre-runs 4 h / 24 h up REFUTED | volume: +0.03…+0.15 (opposite) | split |
| P12 | spike geometry (bigger run better, late in bar worse) | all flat (|Δ| ≤ 0.13) | not scored | RIGHT SIGN / MIXED, n.s. |
| P13 | mcap ≥ $200M → worse | 1 / 76 master fills stamped | — | UNTESTABLE |
| **P14** | **HEALTHY_BEAR / HB ∧ trend gap ≥ −0.13 → worse** | HB 22·73 %·+0.13 vs 54·93 %·+0.38 (n.s.) · **41h zone 14·71 %·−0.12 vs +0.40, p 0.0005** | **+0.14…+0.23 (opposite)** | **CONFIRMED in master; the BT contradicts** |

---

## 1 · Regime variables and coverage (`_coverage.csv`, per variable × era group)

- **Rebuilt (closed bars only)** with `study_ml_regime_wl_build.build()`, unchanged:
  - the 80 ML regime variables: BTC returns / drawdowns / efficiency / volatility / persistence / indicators on 5m / 1h / 4h / 1d; top-50 breadth (D − 1 list); dispersion; 24 h correlation; median beta; alt − BTC; ETHBTC; volume ratio; funding; time;
  - plus BTC 15 m / 30 m returns.
- **Fade-specific, rebuilt:**
  - pair 5m: returns 1 h / 4 h / 24 h / 7 d, off 7 d high / low, rv24, 24 h-vs-7 d volume, 24 h corr / beta to BTC, previous bar, 1 h EMA50 gap;
  - spike geometry from closed 1 m bars: forming-bar run, distance below the 15-min high, 15-min run, seconds into the bar, 5-min volume burst;
  - pair funding at entry (last settlement ≤ t) and its 3-settlement mean.
- **Engine stamps:** every `entry_*`, the regime label, the 41h zone, and the dashboard W/L columns (FanR, ADXDir, RSIDir, DI spread, RSI jump, AltCorr / AltBeta = the rebuild, same OBS-R1 definition).
- **Validation against the live stamps on fade fills:**

  | rebuilt | stamp | r | N |
  |---|---|---|---|
  | eff72 | `entry_btc_eff72` | 1.000 | 39 |
  | above72 | `entry_btc_above72_pct` | 1.000 | 39 |
  | off30d | `entry_btc_off30d_high_pct` | 1.000 | 39 |
  | BTC ADX 5m | `entry_btc_adx` | 0.996 | 84 |
  | closed BTC RSI | `entry_btc_rsi_closed` | 0.998 | 25 |
  | BTC RSI 1h | `entry_btc_rsi_1h` | 0.978 | 84 |

| family | G1 B1 | G2 B2 | G3 B3 | G4 B4–B11 | G5 B12–B16 | G6 B17–B24 | B25 |
|---|---|---|---|---|---|---|---|
| 79 rebuilt BTC / breadth / alt / time | 6/6 | 17/17 | 17/17 | 9/9 | 12/12 | 15/15 | 8/8 |
| 12 pair-5m · 5 spike-1m · 3 pair-funding | full | full | full | full | full | full | full |
| rebuilt top-50 funding (cache ends 10-02) | 6/6 | 17/17 | 17/17 | 9/9 | 8/12 | **0/15** | **0/8** |
| ~65 core entry stamps (ADX, RSI, DI, gaps, stretch, vol, BTC RSI/ADX/trend gap, regime label …) | full | full | full | full | full | full | full |
| eff72 / above72 / r72 / off30d / off24h stamps | 0 | 0 | 0 | 4–5/9 | 12/12 | 15/15 | 8/8 |
| BTC 1d ret / 4h EMA50-200 / EMA50-100 / pair 1d −DI / pair 1h EMA20-200 / ETH 5m stamps | 0 | 0 | 0 | 0 | 5/12 | 15/15 | 8/8 |
| mcap / CMC rank / vol-mcap (DL 311) | 0 | 0 | 0 | 0 | 0 | 1/15 | 6/8 |
| BTCSlp / GlobalVol / 1hSlope stamps | 0 | 0 | 0 | 0 | 0 | 0 | 0 (blind) |

Recent-only stamps are perfectly confounded with the "late" era. Regime claims used their full-coverage kline rebuilds instead.

---

## 2 · Tape context per era (checklist ④; medians at entry; `_tape.csv`)

| | G1 B1 | G2 B2 | G3 B3 | G4 B4–B11 | G5 B12–B16 | G6 B17–B24 | B25 |
|---|---|---|---|---|---|---|---|
| span | 07-28→07-31 | 08-01→08-09 | 08-12→08-21 | 08-27→09-20 | 09-23→10-03 | 10-05→10-20 | 10-23→10-26 |
| kept fades | 6·83 %·+0.61 | 17·88 %·+0.55 | 17·82 %·+0.12 | 9·89 %·+0.39 | 12·100 %·+0.47 | 15·80 %·−0.05 | 8·62 %·−0.35 |
| stops | 1 | 3 | 3 | 1 | 1 | 3 | 3 |
| BTC over span / max DD % | −1.4 / −4.3 | +3.1 / −2.2 | +23.2 / −3.5 | +3.0 / −8.6 | −1.7 / −5.3 | −4.0 / −8.1 | −0.8 / −3.4 |
| BTC 1d EMA50 vs EMA200 % | −11.4 | −11.1 | −10.3 | +0.5 | +3.6 | +5.9 | +6.0 |
| top-20 alt corr 24 h | 0.21 | 0.13 | 0.11 | 0.26 | 0.44 | 0.38 | 0.31 |
| median alt beta | 0.97 | 0.84 | 0.80 | 1.54 | 1.95 | 1.51 | 1.48 |
| spiked pair's own beta to BTC | 0.32 | 0.61 | 0.41 | 0.95 | 1.38 | 1.30 | 1.69 |
| dispersion 24 h | 10.4 | 13.9 | 11.9 | 17.9 | 7.1 | 7.7 | 11.2 |
| alt − BTC 7 d | +0.6 | +1.4 | +3.1 | +3.0 | +8.5 | −1.8 | −1.8 |
| BTC rv24 percentile (90 d) | 49 | 15 | 19 | 74 | 65 | 52 | 36 |
| HEALTHY_BEAR share | 17 % | 47 % | 24 % | 11 % | 17 % | 40 % | 25 % |

**Read.**
- The two losing groups (G6, B25) share three things:
  - BTC above its 1d EMA200;
  - alts under-performing BTC on 7 d;
  - spiked pairs with a high own-beta to BTC (1.3–1.7, vs 0.3–0.6 in G1–G3).
- But G5 had the same macro (BTC above EMA200, the highest correlation and beta) and went **12·100 %**.
- No single tape variable orders the eras.

---

## 3 · Uniform-degradation test (checklist ③; `_uniform.csv`)

Early G1–G5 is 61·89 %·+0.40. Late G6 + B25 is 23·74 %·−0.16. The stop rate goes 15 % → 26 %.
- **33 of 36 sub-cohorts degraded**: regime label, pair volume, pair ADX, ADX Δ, RSI, RSI jump, pre-spike RSI, ATR, pair trend, BTC RSI, hour, weekend, liquidity-capped.
- Exceptions:
  - lowest 24 h-volume tercile: 21·81 %·+0.21 → 7·100 %·+0.32;
  - high pair-ADX tercile: +0.28 → 7·100 %·+0.34;
  - mid BTC-RSI (N = 2).
- The worst degradation:
  - low pair-ADX tercile: 21·95 %·+0.41 → 7·29 %·−0.97;
  - mid RSI tercile: → 5·40 %;
  - low BTC-RSI tercile: → 9·56 %.
- **No pair concentration:** the 6 late losers are 6 different pairs on 6 different days: KOMA, KMNO, SOLV, 1000LUNC, RAYSOL, PARTI.

**Verdict.**
- Near-uniform degradation, so the cause is a **tape / unmeasured variable, not a broken entry cell**.
- None of the 117 measured regime variables isolates it (§4).
- The degradation spares vertical spikes on thin, trending pairs. That is the opening for the within-regime read in §5.

---

## 4 · Regime separators (1D, sign → median → terciles; fill units + day units; `_regime_screen.csv`, `_regime_day.csv`)

**Screen totals.**
- 360 zone tests on 117 market variables, master 76 fills.
- p < 0.05: **12** (chance ≈ 18). Family-wise p < 0.05: **0**.
- Day level (45 days, rotation null): **3 / 170** at p < 0.05 (chance ≈ 8).
  - The three are BTC 1d EMA50-200 (ρ −0.44), 1 h dispersion (+0.34) and ETHBTC 7 d (+0.43).
  - All three have a within-era ρ near 0 (−0.08 / +0.16 / −0.01). They are era proxies.

Top splits (master; Δ = zone − rest; BT Δ per seed):

| split | zone | rest | Δ | days | held-out-era sign | B25 zone | BT Δ s1/s2/s3 | read |
|---|---|---|---|---|---|---|---|---|
| BTC 1d EMA20 vs EMA50 ≤ q33 | 29·90 %·+0.54 | 47·85 %·+0.17 | +0.37 | 15 | 1/1 | — | −0.03 / −0.03 / +0.04 | era proxy (G1–G3) |
| BTC off 30 d low > 9.2 % | 38·97 %·+0.43 | 38·76 %·+0.18 | +0.25 | 26 | 2/2 | — | −0.06 / +0.03 / −0.04 | BT flat; 1 zone loser |
| BTC 5m ADX rising (> q67) | 25·96 %·+0.65 | 51·82 %·+0.14 | +0.51 | 21 | 5/6 | 3·33 % | −0.06 / +0.06 / +0.06 | B25 opposite |
| top-50 advance 1 h ≤ 29 % | 26·92 %·+0.62 | 50·84 %·+0.15 | +0.47 | 18 | 5/6 | 4·50 % | −0.02 / −0.02 / −0.12 | B25 opposite; BT opposite |
| **41h HB ∧ trend gap ≥ −0.13** | **14·71 %·−0.12** | **62·90 %·+0.40** | **−0.52** | 14 | **4/4** | 2·50 % | **+0.20 / +0.17 / +0.23** | pre-registered; BT opposite |
| alt beta > 1.03 (P4) | 38·87 %·+0.19 | 38·87 %·+0.43 | −0.24 | 27 | 4/4 | 8·62 % | +0.11 / +0.07 / +0.03 | n.s.; BT opposite; ~1 episode |

### Regime splits taken forward (`_splits.csv`)

| split | side | master | days · episodes (gap > 3 d) | B25 | BT s1 / s2 / s3 | stack-blocked (as traded) |
|---|---|---|---|---|---|---|
| S1 41h zone | zone | 14·71 %·−0.12 | 14 · 7 | 2·50 %·−0.62 | 121·77 %·+0.22 / 127·76 %·+0.12 / 122·80 %·+0.17 | 10·30 %·−0.66 |
| | rest | 62·90 %·+0.40 | 40 · 5 | 6·67 %·−0.27 | +0.02 / −0.05 / −0.06 | 40·50 %·−0.16 |
| S2 alt beta > 1.030 | zone | 38·87 %·+0.19 | 27 · 9 | 8·62 %·−0.35 | +0.08 / +0.00 / −0.01 | 18·61 % |
| | rest | 38·87 %·+0.43 | 21 · 3 | 0 | −0.03 / −0.07 / −0.04 | 32·38 % |
| S3 BTC 1d EMA50 > EMA200 | zone | 35·89 %·+0.22 | 23 · 3 | 8·62 %·−0.35 | +0.14 / −0.04 / +0.02 | 6·33 % |
| | rest | 41·85 %·+0.38 | 22 · 2 | 0 | +0.02 / −0.01 / −0.03 | 44·48 % |

- 82 % of the S2 zone is also S3: the same regime change of 09-13 / 09-21. As switches, S2 and S3 are **~1 episode**, so the WINDOW-UNITS rule makes them unusable.
- **Cohort completeness.** The 10 stack-blocked fades inside the 41h zone lost as traded (10·30 %·−0.66). The existing gates (mostly FRESHBREAK / LAGGARD) already remove the worst 41h-zone fills, so 41h is not "missing" those.

---

## 5 · Within-regime entry W/L (`_within_1d.csv`, `_within_2d.csv`, `_within_null.csv`)

**Setup.**
- Variables: 64 entry variables. These are every pair-level `entry_*` stamp, the dashboard W/L columns (Conf is constant STRONG_BUY on all 84 fades, so it carries no information) and the fade-specific rebuilds.
- 1D: sign / median / tercile zones.
- Exhaustive 2D: median splits within the side, every variable pair × 4 quadrants, N ≥ 6; 5.0–5.8 k quadrants per side.
- Two nulls:
  - rotation (every circular shift of the time-sorted side);
  - permutation (300–1,000).
- "Cheap bar" = the number of quadrants with WR < 71 %, avg < 0 and N ≥ 8, compared real vs null.

| split · side · pool | N | 1D p < .05 (rot / perm / chance) | 2D \|t\| ≥ 3 real vs null median (P rot / perm) | 2D cheap-bar real vs null (P rot / perm) |
|---|---|---|---|---|
| S1 41h zone · master | 14 | 0 / 13 / 7 | — (N < 20) | — |
| S1 rest · master | 62 | 7 / 14 / 8 | 13 vs 2 (0.14 / 0.10) | 62 vs 15 (**0.017 / 0.030**) |
| S1 rest · master + B25 | 68 | 13 / 14 / 8 | 10 vs 5 (0.37 / 0.21) | 70 vs 49 (0.29 / 0.22) |
| S2 zone · master | 38 | 5 / 6 / 8 | 14 vs 39 (0.91 / 0.85) | 217 vs 195 (0.37 / 0.32) |
| S2 zone · master + B25 | 46 | 3 / 12 / 8 | 213 vs 147 (0.19 / 0.17) | 613 vs 511 (0.33 / 0.19) |
| S3 zone · master | 35 | 3 / 14 / 8 | 13 vs 30 (0.72 / 0.60) | 175 vs 128 (0.25 / 0.13) |
| **S3 zone · master + B25** | 43 | 14 / 18 / 8 | **366 vs 87 (0.000 / 0.007)** | **753 vs 391 (0.025 / 0.023)** |
| S2 / S3 rest (both pools) | 38–41 | ≤ 8 | 2–5 vs 2–3 (≥ 0.38) | below null |

**Read.**
- **The only within-regime structure beyond chance is in the late tape (S3), and only once the 8 provisional B25 fills are added.**
- The quadrants that drive it share one theme: a **weak spike**. These variables are near-collinear measures of the same thing:
  - low +DI;
  - low stretch above EMA5 / EMA13;
  - low DI spread;
  - also low ATR in places.
- Top cells:
  - +DI ≤ 48.3 ∧ Ext% ≤ 1.79: 16·56 %·−0.49 vs 27·100 %·+0.47 (t −4.2);
  - DI spread ≤ 33.9 ∧ Ext% ≤ 1.79: 14·50 %·−0.59.
- The S1-rest "excess" (P 0.017) is a mirror W-side: quadrants that *win* more than chance, with +DI low ∧ pair funding ≥ 0 as the weak side 28·79 %·+0.05. It has no loser cell below breakeven with N ≥ 15.
- Mechanism (theory-consistent with the PREP prereg's frozen +DI / Stretch / Ext% W_HIGHER rows):
  - In a quiet, lock-step tape a vertical, isolated spike is still a wick, and it still reverts.
  - A soft spike (small +DI jump, small overshoot) on a high-beta pair is more likely part of a correlated grind, so it does not snap back.
  - In the early tape (low correlation, BTC below EMA200) soft spikes reverted fine.

---

## 6 · Candidates — expectancy bar on the fade's own breakeven (71.0 %), window(day)-clustered bootstrap 4,000

**Bar:** WR < 71 % · P(avg < 0) ≥ 0.95 · ≥ 8 days · no day / pair ≥ 50 % of the zone loss · N ≥ 15. Then a 30–50 % haircut.

| candidate (block) | master blocked → kept | B25 blocked (prov.) | ALL blocked | bar master / ALL (P_neg) | BT s1 / s2 / s3 blocked | verdict |
|---|---|---|---|---|---|---|
| **OBS-F1** late tape ∧ +DI ≤ 48.5 ∧ Ext% ≤ 1.80 | 12·67 %·−0.30 → 64·91 %·+0.42 | 5·40 %·−0.76 | 17·59 %·−0.44 (14 d, 15 pairs, top pair 14 %, top day 28 %) | fail (p 0.89, n) / **PASS (0.978)** | 46·80 %·+0.10 / 46·78 %·+0.07 / 43·81 %·+0.05 (all better than rest) | **KEEP OBSERVING** (frozen §8) |
| OBS-F1b late tape ∧ +DI ≤ 43.6 (PREP-prereg cut, not fitted) | 11·73 %·−0.20 | 3·67 % | 14·71 %·−0.22 | fail / fail (0.83) | +0.16 / −0.18 / +0.06 | robustness only: weaker at the unfitted cut |
| (check) EARLY tape ∧ same weak spike | 26·88 %·+0.33 | — | — | — | — | interaction confirmed in live data |
| (check) weak spike, no regime | 38·82 %·+0.13 | 5·40 % | 43·77 %·+0.02 | fail | +0.09 / +0.07 / +0.13 | not a filter |
| C1 = 41h zone | 14·71 %·−0.12 → 62·90 %·+0.40 | 2·50 %·−0.62 | 16·69 %·−0.18 (16 d) | fail (wr, p, n) / fail (p 0.79) | +0.22 / +0.12 / +0.17 | KEEP OBSERVING (its own gate) |
| C2 HEALTHY_BEAR (all) | 22·73 %·+0.13 | 2·50 % | 24·71 %·+0.07 | fail / fail (0.44) | +0.15 / +0.10 / +0.14 | REJECT as a filter (positive cohort) |
| C3 alt beta > 1.03 (switch) | 38·87 %·+0.19 | 8·62 % | 46·83 %·+0.09 | fail / fail | +0.08 / 0.00 / −0.01 | REJECT (≈ 1 episode, positive) |
| C4 BTC 1d EMA50 > EMA200 (switch) | 35·89 %·+0.22 | 8·62 % | 43·84 %·+0.12 | fail / fail | +0.14 / −0.04 / +0.02 | REJECT (3 episodes, positive) |
| W2D S2 ∧ ATR ≤ 0.52 ∧ pair 4 h ≤ −0.37 | 9·44 %·−0.67 | 2·50 % | 11·45 % | fail (n) | +0.11 / 0.00 / +0.05 | screen artefact (fitted ×2; BT flat) |
| W2D S3 ∧ ATR ≤ 0.51 ∧ +DI ≤ 49.2 | 10·60 %·−0.43 | 4·50 % | 14·57 % | fail (n) | +0.03 / +0.19 / +0.02 | same family as OBS-F1 |
| funding < 0 → **W side** (multiplier?) | 12·92 %·+0.74 (win zone) | 1·0 % (RAYSOL) | 13·85 %·+0.57 | — | not scored | below the W gate (N ≥ 30); 1 B25 counter-example; info only |

### Per batch, before → after (`_obs_per_batch.csv`, `_per_batch.csv`; $ = stack ticket at 2×)

**OBS-F1** (the only line with an in-sample net gain):

| batch | before | blocked | after |
|---|---|---|---|
| B1–B6, B8, B9, B15, B16, B20 | unchanged (no zone fills) | — | — |
| B7 | 2·100 %·+0.39 · $198 | 1·100 %·+0.44 · $125 | 1·100 % · $73 |
| B10 | 2·50 %·−0.38 · −$43 | 1·0 %·−1.50 · −$175 | 1·100 % · $131 |
| B12 | 3·100 % · $406 | 1·100 % · $57 | 2·100 % · $349 |
| B13 | 4·100 % · $311 | 2·100 % · $112 | 2·100 % · $200 |
| B17 | 3·100 % · $184 | 1·100 % · $30 | 2·100 % · $154 |
| B18 | 4·75 %·−0.11 · −$139 | 2·50 %·−0.60 · −$233 | 2·100 % · $94 |
| B19 | 3·100 % · $147 | 1·100 % · $36 | 2·100 % · $111 |
| B21 | 1·100 % · $40 | 1·100 % · $40 | 0 |
| B23 | 2·0 %·−1.50 · −$249 | 2·0 % · −$249 | 0 |
| **master total** | 76·87 %·+0.31 · $4,237 | 12·67 %·−0.30 · −$257 | 64·91 %·+0.42 · **$4,494 (+$257; haircut +$130…+$180)** |
| **B25 (prov.)** | 8·62 %·−0.35 · −$538 | 5·40 %·−0.76 · −$674 | 3·100 %·+0.33 · **+$136 (+$674)** |

- It removes winners in 6 batches and losers in 4.
- **Leave-one-batch-out:** the zone stays negative with any one batch dropped. The best case, dropping B23, is −0.29.

**41h zone**, before → after:
- B2: +$1,797 → +$1,412 (it blocks 4 winners).
- B3: +$283 → +$775 (it blocks 3·33 % · −$491).
- B23: −$249 → 0. B25: −$538 → −$271.
- Master total: +$4,237 → +$4,271 (+$35).
- In % it is +0.31 → +0.40 /fill, because the zone has small winners and one big loss in B3.

---

## 7 · Watch-list items scored on the full data

| item | full-data read | forward cohort | verdict |
|---|---|---|---|
| DL 374 ① ADX Δ < 2.359 (block) | master 37·81 %·+0.13 vs 39·92 %·+0.48; ALL 41·80 %·+0.10; bar fail (WR 80 > 71, P 0.17) | 0 (PARTI 13:19 is before commit 13:19 + 10 min) | **KEEP OBSERVING** (0 / 15) |
| DL 374 ② pair ADX ≤ 19.2 | 18·83 %·+0.17 vs +0.35; ALL 20·80 %·+0.10; fail | 0 | **KEEP OBSERVING** (late-tape low-ADX 7·29 % is in-sample, noted) |
| DL 374 ⑤ HB ∧ ADX Δ ≤ 2.4341 | 14·71 %·−0.05; ALL 15·67 %·−0.14; fail (p 0.75) | 0 | **KEEP OBSERVING** |
| DL 374 ⑥ HB ∧ BTC ADX > 24.59 | 7·43 %·−0.47; fail (N, days, p 0.91) | 0 | **KEEP OBSERVING** |
| DL 374 ③ mcap ≥ $200M → 1.5× | stamped ≥ $200M since DL 311: 1000LUNC 10-25 05:12 −1.50 % and 1000LUNC 10-25 23:04 +0.39 % → 2 / 10, avg −0.55 | 2 / 10 | **KEEP OBSERVING** (avg < 0 so far; decided at 10) |
| DL 374 ④ HB → 1× (size) | HB master 22·73 %·+0.13 (+$978 at 2× → +$489 at 1×); B25 2·50 %·−0.62; BT: HB is the BEST regime (+0.10…+0.15) | 0 / 15 | **KEEP_2X pending** (KEEP OBSERVING) |
| 41h HB ∧ gap ≥ −0.13 | 16·69 %·−0.18 · −$302 · 16 days; forward since 09-18: 9·67 %·−0.25 · −$196 · 9 days | SHIP bar N ≥ 20 ∧ WR ≤ 55 ∧ Σ < 0 → N 16, WR 69 | **KEEP OBSERVING** (not met) |
| DL 165 tight revert ($10–20M band, any fill losing > $300 or Σ Δ < 0 at N ≥ 5) | fills ≥ $10M since 10-01: RESOLV +$32, ORDI +$65, KMNO −$178; none cap-limited, so Δ vs 0.1 % = 0. 1000LUNC (−$308) had $4.1M 24 h volume, outside the band | 0 differing fills | **NOT TRIGGERED** |

**PREP_B25_PREREG fade rows** (`_prereg_rows.csv`):
- Master direction at the frozen thresholds: 23 as predicted, 7 opposite.
  - These are master in-sample reads: the medians came from the master.
  - Opposite: PairTrend ≤ −0.40, PairVol$, pair rank, pair age, quality score, BTC 1d return, pre-spike RSI.
- **B25 refute-only read: not readable for every row**, because no row has ≥ 5 B25 fills on each side.
- Primary rows:
  - #1 PairTrend: flat (+0.30 vs +0.32), not confirmed.
  - #2 mcap: unreadable. Excluding 1000LUNC leaves none.
  - #3 ADX Δ: master as predicted. B25 ex-registration = PARTI only (strong side, lost).
  - #4 pair ADX: as predicted in master.
  - #5 BTC RSI rising: as predicted, tiny (+0.28 vs +0.33).

---

## 8 · Frozen observe line (new) — OBS-F1 "LATE-TAPE WEAK SPIKE" (no config; scout `scripts/scout_fade_late_weak.py`, DECISION_LOG 384)

- **Zone.** A SPIKE_FADE fill where all of the following hold:
  - BTCUSDT 1d EMA50 > EMA200;
    - computed on UTC daily closes, `ewm(span, adjust=False)` over ≥ 400 daily bars;
    - read on the last daily bar CLOSED before the fill;
  - `entry_pos_di` ≤ **48.5**;
  - `entry_dist_from_ema13_pct` ≤ **1.80**.

  🔒 Thresholds are frozen now and rounded outward from the fitted 48.33 / 1.786. They are never re-fit.
- **Cohort.** CLOSED SPIKE_FADE fills opened **≥ 2026-10-27 00:00 UTC**, paper + live pooled in % (book shown per fill), operator-closed excluded.
  - A missing stamp makes the fill unscored, never "rest".
- **Read once**, at the first prefix with **N ≥ 15 zone fills on ≥ 8 UTC days**, using the CLAUDE.md expectancy bar:
  - WR < the forward kept-fade breakeven (71 % if < 5 losses);
  - day-bootstrap P(avg < 0) ≥ 0.95;
  - no day / pair ≥ 50 % of the zone loss.
  - If all pass → **FILTER_CANDIDATE** (operator decides; 30–50 % haircut; revert if the first 10 blocked fades re-price avg > 0).
  - If zone avg ≥ +0.20 → **RETIRE**.
  - Otherwise **KEEP_OBSERVING**.
  - Also track the complement (late tape ∧ strong spike) as the control.
- **Expected date.**
  - Zone rate ≈ 0.39 fills / calendar day while the late tape lasts (17 over 09-13 → 10-26). That puts the read around **early December**.
  - If BTC's 1d EMA50 falls below EMA200, the zone empties. The line then pauses and is not judged.
- **Why not more:**
  - in-sample, found by an exhaustive screen;
  - it passes the bar only with the provisional motivating batch included;
  - the backtest refutes it on 3 / 3 seeds (2025 above-EMA200 tape: weak spikes won).
  - Per the OBSERVE-FIRST rule it is not an arm.
  - As a declared live-bleed override it would still need a revert gate. **My recommendation is not to override.**

---

## 9 · Size implications (stated apart; `_size.csv`)

- Every loss is a full −1.5 % stop. At today's ≈ $20k ticket (2×) each stop costs ≈ $270–$315 ≈ 9–11 % of a $2,900 account.
  - Fills losing > 4 % of $2,900 at 2×: master **8**, B25 **3**. At 1×: master **3**, B25 **2**.
  - No regime filter here removes that tail. The question stays stop width × size (plan §5), which the DL 374 ③ / ④ lines decide forward.
- Regime-conditioned size cuts are **not supported**. Every regime "bad side" is positive in the master:
  - HB +0.13;
  - beta-high +0.19;
  - late tape +0.22.
  - Halving them costs $490–$850 in-sample.
  - The backtest calls HB the best-earning regime.
- The 2× sleeve multiplier itself: the master at 1× would be +$2,233 (S1 rest) − $17 (S1 zone) ≈ +$2.2k, against +$4.2k at 2×. B25 at 1× would be −$269 instead of −$538.
  - That is the whole-sleeve de-mux question, refuted in DL 55 / 09-14. Nothing here re-opens it beyond the existing lines.

---

## 10 · Could NOT test (blind spots)

- **News.** There is no news feed. Only proxies were used: pair volume inflation, pre-runs, 1 m burst. The volume-inflation proxy was the one pre-registered "real move" prediction that confirmed in the master (p 0.03), but the backtest disagrees.
- **Open interest / liquidations.** OI history is public for 30 days only, so it would be era-confounded and was not fetched. Squeeze fuel was read through funding only.
- **Order-book depth / spread at entry.** `entry_book_*` is 0 / 84 on fades.
- **Market cap (P13).** Only 7 fills are stamped (since 10-19), so it is untestable. DL 374 ③ continues.
- **Top-50 median funding after 10-02.** 0 / 23 late fills; the cache ends.
- **Top-50 lists dated 10-12 … 10-22 do not exist in any cache.** Fills of 10-12 … 10-23 (10 master + 1 B25) use the 10-11 list. Only the breadth variables are affected, and the list changes slowly.
- **Stamps blind on fades:** BTCSlp, GlobalVol, 1hSlope (0 / 84).
- **Constant / degenerate:** Conf (always STRONG_BUY), RSIDir, ADXDir (spike-driven).
- **AltCorr / AltBeta engine stamps (DL 383)** are not yet on any fade. The rebuilds use the identical OBS-R1 definition.
- **Exit × regime.** Not studied (entry side only). DL 93 / 111 / 265 exit refutations stand.
- **Stack-blocked fades (50)** are as-traded under older mechanisms and are not re-priced at today's exit. They are used only for the cohort-completeness check.
- **Backtest limits:**
  - the WILLY-hold gap (≈ 8 % of fades missing);
  - the replay reproduces ~29 % of live fades;
  - 1 m pricing;
  - seeds share entries;
  - no 1 m spike geometry or pair funding in BT (s_* / f_* unscored).
- **Statistical power.** 76 master fills on 45 days. The late tape is ~1–3 episodes. A regime switch cannot be judged in window units with this data.
- **Hour-of-day** was screened only as a variable. It was refuted in DL 55 and not re-hunted.

---

## Files

- **Scripts:**
  - `scripts/study_fade_regime_wl_data.py` (cohort loader)
  - `scripts/study_fade_regime_wl_fetch.py` (the only network step)
  - `scripts/study_fade_regime_wl_build.py` (features; reuses `study_ml_regime_wl_build.build`)
  - `scripts/study_fade_regime_wl_common.py`
  - `scripts/study_fade_regime_wl_analyse.py` (steps 1–3 + prereg)
  - `scripts/study_fade_regime_wl_within.py` (steps 4–5, DL 374, 41h, DL 165, size, PREP prereg rows)
  - `scripts/study_fade_regime_wl_deep.py` (frozen lines)
- **Outputs:**
  - `reports/FADE_REGIME_WL_2026-10-26_{PREREG.md, fill_regime, coverage, uniform, tape, regime_screen, regime_day, entry_screen_all, prereg_score, splits, within_1d, within_2d, within_null, candidates, per_batch, size, prereg_rows, obs_lines, obs_per_batch, obs_rows}.csv`
  - `_analyse.txt`, `_within.txt`, `_deep.txt`
  - `_bt_regime.csv.gz` (0.7 MB, gzipped)
- **Local-only (do not commit):** `reports/FADE_REGIME_WL_2026-10-26_fetch/` (18 MB of klines and funding).
- **Rerun:** `fetch → build → analyse → within → deep`, each as `venv/bin/python -W ignore scripts/<name>.py`.
