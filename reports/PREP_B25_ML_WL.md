# PREP B25 — Momentum-LONG winners vs losers (2026-10-26)

Prepared for the B25 batch review (sleeve order: … → LITE / FRENZY_LONG → **momentum long** → …). Study only — nothing shipped, no engine / config / template edits, no git, no bot API. Binance: ≈140 public 1m + ≈370 public 5m kline calls over three runs (peak used weight 281/min of 1,200, cooldown file honoured) for the 32 fills the earlier feature files did not cover.

**Validation:** `scripts/validate_against_master.py` → ALL CHECKS PASS (run before the study; master STACK 2026-10-26a). Rebuilt impulse features reuse `study_ml_wl_1015_features.feats` unchanged. ⚠ Caught during the build: the Oct-15 impulse file's `Y_btc` / `Y_alt` are FORWARD 15-min returns (the study's targets) — they were excluded from the entry screen and appear only as post-entry description; including them had produced fake 'top separators' (Δ −0.5, p 0.001).

## 1 · TL;DR

- **Sleeve now (master, today's stack, full-size):** 158·71%·+0.124 over 71 days; ML breakeven WR = **61.5%** (avg win +0.510 / avg loss -0.814). Without the washed-out window: 138·67%·+0.055. As traded: 210·65%·+0.025. Recent eras are the problem: B12–B16 28·57%·-0.083, B17–B24 37·49%·-0.091. B25 provisional: 9·78%·+0.156 (3 days).
- **Every variable was tested** (225 candidates: all 31 dashboard columns, every entry_* stamp, the Oct-4 pair-strength, Oct-15 impulse and 10-15 market-impulse features, cooldown/burst, hour/day): 155 screenable, 70 not (listed in §11).
- **No separator beats chance.** 1D: 21 tests at p<0.05 vs ≈33 expected by chance; one family-wise hit (a recent-era-only stamp, refuted below). Exhaustive 2D (24,280 quadrant tests): 172 at p<0.01 vs a circular-shift label null (≤ 157 distinct shifts) of median 104 / 95th pct 426 — inside the null; 0 family-wise. Replay agreement of the 2D candidates (17% same sign in all three regimes) equals the random-test base rate (16%).
- **Survivors vs the bar:** no loser zone passes the full expectancy bar with out-of-sample support. Closest: **H2 pair ADXΔ ≤ 0.31 ∧ regime ≠ STRONG_BULL** (58·59%·−0.110; same sign in 6/6 held-out era groups, 2nd half, as-traded and all three replay regimes) — **fails only on confidence** (P(avg<0) 0.84 < 0.95) → pre-registered observe line (§12). It is the third mining of ADX direction (DL 273 ②) — fresh fills only.
- **Winner signature:** **STRONG_BULL regime / BTC ADX > 28** (53·77%·+0.279, Pattern-W bar met in-sample, replay positive in all three regimes) — but the newest era group is OPPOSITE (B17–B24 zone 15·40%·−0.20) and B25 is flat (5 fills 3W/2L ±0). Not a multiplier: observe line with a frozen bar; cell-registry history (NONEXP·SBULL / CALM3D at 1× since 10-06) must be read before any sizing talk.
- **Refuted this round:** EMA5−8 gap ≤ 0.082 (the only family-wise hit) is a recent-era artefact — on the 115 older fills the same cut gives +0.217 vs +0.217, and B25 already contradicts (7 fills · 71% · +0.07). BTC 15-min return ≤ 0 is regime-dependent (replay 2025 and 2026-H1 OPPOSITE sign).
- **Read:** where the loser zones separate at all, the effect sits in the losing recent eras (B12–B24); in BASE–B3 the same zones are flat or positive. That is what an unmeasured regime variable (sleeve-kill checklist ③) would produce just as well as a new entry edge — the forward fills decide. B25 so far contradicts every loser line (P2 zone 2·100 %, P3 7·71 %, P4 4·75 %, DL 273 BTC24H_DOWN 8·88 %·+0.21). **No ship and no sizing change recommended from this study.**

## 2 · Blocks and sleeve baseline

```
blocks (MOMENTUM LONG CLOSED; % at 1×; 95% DAY-bootstrap CI)
  KEEP master (today's stack)    N 158 ·  71 d · WR   71% · avg +0.124 [-0.017,+0.274] · Σ  +19.67% · BE-WR 61%
  TRAD master (as traded)        N 210 ·  82 d · WR   65% · avg +0.025 [-0.091,+0.147] · Σ   +5.20% · BE-WR 63%
  B25 provisional                N   9 ·   3 d · WR   78% · avg +0.156 [-0.306,+0.350] · Σ   +1.40% · BE-WR 62%
  PROBE (B1/B2, separate)        N 166 ·  23 d · WR   48% · avg -0.149 [-0.224,-0.053] · Σ  -24.69% · BE-WR 60%
  ML breakeven WR on KEEP = 61.5% (avg win +0.510 / avg loss -0.814)
  KEEP without washed-out Jun-18→Jul-2   N 138 ·  58 d · WR   67% · avg +0.055 [-0.093,+0.219] · Σ   +7.64% · BE-WR 62%
  KEEP washed-out window only            N  20 ·  13 d · WR  100% · avg +0.601 [+0.424,+0.786] · Σ  +12.02% · BE-WR nan%
  TRAD without washed-out                N 189 ·  69 d · WR   61% · avg -0.039 [-0.156,+0.086] · Σ   -7.39% · BE-WR 65%

  per era group (KEEP | TRAD):
   G0 BASE      KEEP N  32 ·  18 d · WR   94% · avg +0.485 [+0.293,+0.701] · Σ  +15.51% · BE-WR 56%
                TRAD N  35 ·  18 d · WR   89% · avg +0.411 [+0.193,+0.671] · Σ  +14.40% · BE-WR 58%
   G1 B1        KEEP N  15 ·   9 d · WR   80% · avg +0.256 [-0.116,+0.790] · Σ   +3.85% · BE-WR 63%
                TRAD N  25 ·  11 d · WR   72% · avg +0.140 [-0.142,+0.487] · Σ   +3.51% · BE-WR 61%
   G2 B2        KEEP N  10 ·   6 d · WR   80% · avg +0.191 [-0.409,+0.525] · Σ   +1.91% · BE-WR 65%
                TRAD N  25 ·  10 d · WR   60% · avg -0.086 [-0.355,+0.087] · Σ   -2.15% · BE-WR 68%
   G3 B3        KEEP N  22 ·   7 d · WR   77% · avg +0.147 [-0.097,+0.361] · Σ   +3.23% · BE-WR 65%
                TRAD N  28 ·   9 d · WR   79% · avg +0.117 [-0.108,+0.287] · Σ   +3.28% · BE-WR 68%
   G4 B4-B11    KEEP N  14 ·   9 d · WR   79% · avg +0.060 [-0.337,+0.394] · Σ   +0.83% · BE-WR 75%
                TRAD N  23 ·  12 d · WR   65% · avg -0.114 [-0.407,+0.191] · Σ   -2.63% · BE-WR 74%
   G5 B12-B16   KEEP N  28 ·   9 d · WR   57% · avg -0.083 [-0.407,+0.292] · Σ   -2.32% · BE-WR 63%
                TRAD N  37 ·  10 d · WR   49% · avg -0.210 [-0.465,+0.081] · Σ   -7.76% · BE-WR 65%
   G6 B17-B24   KEEP N  37 ·  13 d · WR   49% · avg -0.091 [-0.436,+0.351] · Σ   -3.35% · BE-WR 56%
                TRAD N  37 ·  13 d · WR   49% · avg -0.093 [-0.437,+0.348] · Σ   -3.45% · BE-WR 56%
   per-era KEEP: BASE 32·94%·+0.48 · B1 15·80%·+0.26 · B2 10·80%·+0.19 · B3 22·77%·+0.15 · B4 1·100%·+0.12 · B5 3·100%·+0.45 · B6 2·50%·-0.90 · B7 1·100%·+0.10 · B8 4·75%·+0.11 · B9 2·50%·+0.25 · B10 1·100%·+0.10 · B12 15·67%·+0.12 · B13 3·67%·+0.14 · B14 4·25%·-0.66 · B15 1·0%·-0.99 · B16 5·60%·-0.18 · B17 6·67%·+0.65 · B18 3·0%·-0.70 · B19 4·25%·-0.44 · B20 2·0%·-0.70 · B21 3·0%·-0.71 · B22 6·67%·+0.16 · B23 13·69%·-0.07
```

PROBE fills (B1/B2 probe era, separate line, never pooled): 166·48%·-0.149.

## 3 · Dashboard columns → master columns

Source: `main.py` `_ec_row` (the builder behind *Entry Conditions by Outcome* and *Entry Conditions by Strategy — Winners vs Losers*, `templates/index.html` L1510–1620; text exports ~L10110 / ~L11969).

| dashboard column | column tested |
|---|---|
| RSI | `entry_rsi` |
| ADX | `entry_adx` |
| ADXDir | `d_adx_dir (entry_adx − entry_adx_prev; Rising = > 0)` |
| RSIDir | `d_rsi_dir (entry_rsi − entry_rsi_prev; Rising = > 0)` |
| Gap | `entry_gap` |
| Gap5-8 | `entry_ema_gap_5_8` |
| Gap8-13 | `entry_ema_gap_8_13` |
| Stretch | `entry_ema5_stretch` |
| EMA20Slp | `d_abs_ema20_slope (|entry_ema20_slope|) + signed entry_ema20_slope` |
| ADXΔ | `entry_adx_delta` |
| RngPos | `entry_range_position` |
| ATR% | `entry_atr_pct` |
| PairTrend | `entry_pair_ema20_ema50_gap_pct (holds EMA13−EMA50)` |
| PairVol$ | `entry_pair_volume_24h_usd` |
| Ext% (LONG) | `entry_dist_from_ema13_pct` |
| 1hSlope | `entry_btc_1h_slope` |
| BTCSlp | `d_abs_btc_ema20_slope (|entry_btc_ema20_slope|) + signed entry_btc_ema20_slope` |
| BTCRSI | `entry_btc_rsi` |
| BTCADX | `entry_btc_adx` |
| BTCADXDir | `d_btc_adx_dir (entry_btc_adx − entry_btc_adx_prev)` |
| BTCRSIDir | `d_btc_rsi_dir (entry_btc_rsi − entry_btc_rsi_prev) + d_btc_rsi_30m_dir (vs entry_btc_rsi_prev6)` |
| Breadth (LONG) | `entry_bull_pct` |
| BTCTrend | `entry_btc_trend_gap_pct` |
| GlobalVol | `entry_global_volume_ratio` |
| BTCExt% (LONG) | `entry_btc_dist_from_ema13_pct` |
| Conf | `confidence (cat__confidence==*)` |
| Fan ratio | `d_fan_ratio` |
| BTC ATR% (perf payload) | `entry_btc_atr_pct` |
| EMA5 dist (perf payload) | `entry_price_vs_ema5_pct` |
| BTCTrend(exit) — EXIT var | `exit_btc_trend_gap_pct (descriptive only)` |
| Peak — POST-ENTRY | `peak_pnl (descriptive only)` |
| Trough — POST-ENTRY | `trough_pnl (descriptive only)` |
| Duration — POST-ENTRY | `dur_min (descriptive only)` |
| SLProf — POST-ENTRY | `peak_pnl of losers (descriptive only)` |

## 4 · Coverage (variable × era group: scored / fills) — unscored is never 'rest'

Full-size momentum longs. Screened = ≥30 scored KEEP fills and ≥2 values; ≥80 % KEEP coverage → exhaustive 2D; lower coverage → 1D on scored rows only (flagged). Full CSV: `reports/PREP_B25_ML_WL_coverage.csv`.

| variable | G0 | G1 | G2 | G3 | G4 | G5 | G6 | B25 | KEEP cov | status |
|---|---|---|---|---|---|---|---|---|---|---|
| `agg_vol15_x24h` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `br_ret15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `br_ret60` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `br_up15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `br_up30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `br_up30_chg1h` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `br_up60` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `btc_r15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `btc_r30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `btc_r5` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `btc_r60` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `btc_vol5_x24h` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `cat__confidence==STRONG_BUY` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `cat__confidence==VERY_STRONG` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `cat__entry_btc_regime==BEARISH` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==BEAR_EXHAUSTED` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==BULLISH` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==CHOPPY_FLAT` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==CHOPPY_WEAK` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `cat__entry_btc_regime==HEALTHY_BEAR` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==HEALTHY_BULL` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `cat__entry_btc_regime==NEUTRAL` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==STRONG_BEAR` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `cat__entry_btc_regime==STRONG_BULL` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `cat__entry_macro_trend==BULLISH` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `d_abs_btc_ema20_slope` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_abs_ema20_slope` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_adx_dir` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_btc_adx_dir` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_btc_rsi_1h_dir` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_btc_rsi_30m_dir` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_btc_rsi_dir` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_burst_prior_min` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_di_spread` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_dow` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_fan_ratio` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_fills_same_scan` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_hour` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `d_rsi_dir` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `disp15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_adx` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_adx_delta` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_adx_prev` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_atr_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_bear_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_1h_slope` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_adx` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_adx_prev` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_atr_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_dist_from_ema13_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_ema20_slope` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_rsi` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_rsi_1h` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_rsi_1h_prev` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_rsi_prev` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_rsi_prev6` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_btc_trend_gap_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_bull_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_dist_from_ema13_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_ema20_slope` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_ema50_slope` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_ema5_stretch` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_ema_gap_5_8` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_ema_gap_8_13` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_funding_rate` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_gap` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_gap_expand_marginal` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_global_volume_ratio` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_neg_di` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pair_ema20_ema50_gap_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pair_rank` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pair_volume_24h_usd` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pair_volume_ratio` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pattern_c1_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c2_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c3_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c4_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c5_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c6_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pattern_c7_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c8_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c9_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_c_any_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pattern_w1_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_w2_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pattern_w3_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_w4_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pattern_w5_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_w6_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | NOT testable |
| `entry_pattern_w_any_match` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_pos_di` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_price_vs_ema5_pct` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_quality_score` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_range_position` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_rsi` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `entry_rsi_prev` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `pair_vol15_x24h` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 100% | 2D+1D |
| `pair_r15` | 30/35 | 25/25 | 24/25 | 26/28 | 22/23 | 36/37 | 35/37 | 9/9 | 94% | 2D+1D |
| `pair_r60` | 30/35 | 25/25 | 24/25 | 26/28 | 22/23 | 36/37 | 35/37 | 9/9 | 94% | 2D+1D |
| `rs15` | 30/35 | 25/25 | 24/25 | 26/28 | 22/23 | 36/37 | 35/37 | 9/9 | 94% | 2D+1D |
| `rs60` | 30/35 | 25/25 | 24/25 | 26/28 | 22/23 | 36/37 | 35/37 | 9/9 | 94% | 2D+1D |
| `imp_CONJ` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L1_btc` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L2_eth` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L3_sol` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L4_xrp` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L5_zec` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L6_top10` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L7_brrise` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L8_taker` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_L9_vsurge` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_SCORE` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_SPIKE` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_br_trend` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_br_trend_d15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_corr15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_er_15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_er_30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_eth_r15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_sol_r15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_spike_15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_spike_30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_top10_tbr` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_top10_up` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_upsh_15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_upsh_30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_volconc_15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_volconc_30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_volslope_15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_volslope_30` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_vsurge` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_vwap1h` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_xrp_r15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `imp_zec_r15` | 35/35 | 25/25 | 25/25 | 28/28 | 23/23 | 37/37 | 19/37 | 0/9 | 89% | 2D+1D |
| `entry_pair_age_days` | 0/35 | 20/25 | 25/25 | 28/28 | 23/23 | 37/37 | 37/37 | 9/9 | 77% | 1D scored-only |
| `ps_BURST` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_CB` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_CHOP` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_L_rel24d` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_L_relE200` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_btc1h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_btc24h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_btc4h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_btc_d13` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_btc_d50` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_p15` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_p1h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_p24h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_p4h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_rel15` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_rel1h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_rel24h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_rel4h` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `ps_zoneD` | 33/35 | 16/25 | 10/25 | 16/28 | 10/23 | 28/37 | 0/37 | 0/9 | 68% | 1D scored-only |
| `cd_c120_flag` | 2/35 | 17/25 | 0/25 | 7/28 | 3/23 | 15/37 | 37/37 | 0/9 | 46% | 1D scored-only |
| `cd_c120_prior` | 2/35 | 17/25 | 0/25 | 7/28 | 3/23 | 15/37 | 37/37 | 0/9 | 46% | 1D scored-only |
| `cd_cd_flag` | 2/35 | 17/25 | 0/25 | 7/28 | 3/23 | 15/37 | 37/37 | 0/9 | 46% | 1D scored-only |
| `entry_btc_off24h_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 9/23 | 37/37 | 37/37 | 9/9 | 46% | 1D scored-only |
| `entry_btc_off24lo_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 9/23 | 37/37 | 37/37 | 9/9 | 46% | 1D scored-only |
| `entry_btc_off30d_high_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 4/23 | 37/37 | 37/37 | 9/9 | 43% | 1D scored-only |
| `entry_btc_r72_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 4/23 | 37/37 | 37/37 | 9/9 | 43% | 1D scored-only |
| `entry_long_heat_flags` | 0/35 | 0/25 | 0/25 | 0/28 | 4/23 | 37/37 | 37/37 | 9/9 | 43% | 1D scored-only |
| `entry_btc_above72_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 2/23 | 37/37 | 37/37 | 9/9 | 42% | 1D scored-only |
| `entry_btc_eff72` | 0/35 | 0/25 | 0/25 | 0/28 | 2/23 | 37/37 | 37/37 | 9/9 | 42% | 1D scored-only |
| `entry_cmc_rank` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 15/37 | 37/37 | 9/9 | 30% | 1D scored-only |
| `entry_mcap_usd` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 15/37 | 37/37 | 9/9 | 30% | 1D scored-only |
| `d_gap5_20_chg` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_btc_1d_ret_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_btc_ema50_100_gap_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_eth_5m_ret1_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_gap_5_20_prev_signed_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_gap_5_20_signed_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_gap_5_8_signed_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_pair_1h_ema20_200_gap_pct` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 7/37 | 37/37 | 9/9 | 27% | 1D scored-only |
| `entry_btc_rsi_closed` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 0/37 | 37/37 | 9/9 | 23% | 1D scored-only |
| `entry_vol_mcap_ratio` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 0/37 | 28/37 | 9/9 | 18% | NOT testable |
| `entry_chop_burst_prior_fill_s` | 0/35 | 0/25 | 0/25 | 0/28 | 0/23 | 0/37 | 9/37 | 3/9 | 6% | NOT testable |
| `cd_cd_min_since_stop` | 1/35 | 2/25 | 0/25 | 0/28 | 0/23 | 2/37 | 3/37 | 0/9 | 5% | NOT testable |

Never stamped on a momentum long (0 fills in every era; listed for completeness): `d_rsi_dir2`, `entry_bear_below24`, `entry_bear_bypass`, `entry_bear_eff24`, `entry_bear_off24lo`, `entry_bear_r24`, `entry_book_drift_pct`, `entry_book_spread_pct`, `entry_br_above`, `entry_br_bear_pct_top10`, `entry_br_bull_pct_top10`, `entry_br_door_age_min`, `entry_br_eff`, `entry_br_off24h`, `entry_br_r72`, `entry_br_top10_n`, `entry_btc_4h_ema50_200_gap_pct`, `entry_flatbase_flat14_pct`, `entry_flatbase_flat_cov_pct`, `entry_flatbase_off_peak_pct`, `entry_frenzy_above_share`, `entry_frenzy_above_streak`, `entry_frenzy_adx_delta`, `entry_frenzy_bar_ret_pct`, `entry_frenzy_catchup`, `entry_frenzy_catchup_bars`, `entry_frenzy_catchup_move_pct`, `entry_frenzy_di_spread`, `entry_frenzy_gvol`, `entry_frenzy_hours`, `entry_frenzy_retry_green_ret`, `entry_frenzy_retry_wait_min`, `entry_frenzy_run_pct`, `entry_frenzy_stop_atr`, `entry_frenzy_stop_pct`, `entry_frenzy_vol_mult`, `entry_frenzy_vol_trend`, `entry_frenzy_vs_vwap_pct`, `entry_frenzy_willy_nomiss`, `entry_frenzy_willy_spike_high`, `entry_frenzy_willy_trigger`, `entry_frenzy_willy_wait_bars`, `entry_pair_1d_ndi`, `entry_surge_btc_move_pct`, `entry_surge_gvol`, `entry_surge_pair_move_pct`, `entry_willy_ht_size_frac`.

## 5 · Method and null calibration

- Outcome: % at 1× (leverage-invariant). Discovery block KEEP = master full-size fills kept by today's stack, y = `stack_pct` (in-sample to the stack's own filters — DL 204 reading rule); robustness block TRAD = as traded; B25 descriptive; replay (`ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv`, MOM-long, 3 seeds, 2025 BULL / 2026-H1 BEAR / 2026-H2 BULL) REFUTE-ONLY.
- 1D zones per variable: sign (when both signs ≥5 %), binary ==1/==0, > / ≤ median, ≤ q33, > q67; quintile tables for every variable (`_quintiles.csv`). 2D: exhaustive all pairs of the full-coverage variables, median × median, all four quadrants (zone ≥10 fills).
- Null: circular shifts of the time-sorted outcome (keeps day clustering): 1,000 for 1D, 200 for 2D; per-test p and family-wise p (max-|Δ| over the whole family). Second null for 2D: the count of p<0.01 tests under 20 shifted outcomes.
- OOS: leave-one-era-group-out with thresholds re-fit on the training groups; chronological halves (fit on the first, test the second); TRAD same sign; replay sign per regime with day-bootstrap P.
- Bars: expectancy bar (filters) on KEEP — WR < breakeven, P(avg<0) ≥ 0.95 by DAY-clustered bootstrap, ≥ 8 days, N ≥ 15, no day / pair ≥ 50 % of the zone's net loss; Pattern-W bar (multipliers) N ≥ 30, WR ≥ 70 %, avg ≥ +0.10 %, $ > 0. Market-wide variables read in DAY units (and 2-min scan windows reported).

```
## S1 coverage: 225 candidate variables · screened 155 (≥30 scored KEEP fills, ≥2 values) · full-coverage (≥80% of KEEP) 114 · low-coverage scored-only 41 · NOT testable 70
## S2 1D: 668 zone tests on KEEP (N=158); family-wise 95% |Δ| threshold = 0.823; tests with p_null<0.05: 21 (chance ≈ 33); p_fw<0.05: 1
## S3 exhaustive 2D: 114 full-coverage variables → 6441 pairs → 24280 valid quadrant tests (median × median)
  2D tests p_null<0.01: 172 (chance ≈ 243) · p_null<0.05: 1006 (chance ≈ 1214) · family-wise 95% |Δ| = 0.952 · p_fw<0.05: 0
  circular-shift (label-rotation, day clustering kept, ≤ 157 distinct shifts) replicate: # tests with p<0.01 under 20 shifted outcomes: median 104, 95th pct 426 (observed 172)
  replay loaded: 4309 MOM-long fills · 10-15 impulse matched 100% · Oct-15 legs matched 100%
## S3b replay agreement (refute-only): random 2D tests (n=270) — all-3-regimes same sign as master 16%, ≥2 strong (P≥0.80) 12% · 2D candidates (n=144) — all-3 17%, ≥2 strong 15%
```

## 6 · Dashboard variables, 1D (KEEP)

Best zone per variable by p_null (sign / median / tercile); p_fw = family-wise over all 668 1D tests. None reaches p_fw < 0.05.

| variable | scored | median W / L | best zone | zone N·WR·avg | Δ vs rest | p | p_fw | macro |
|---|---|---|---|---|---|---|---|---|
| `entry_rsi` | 158 | +60.06 / +59.44 | <=q33(58.85) | 53·66%·+0.200 | +0.113 | 0.312 | 1.00 |  |
| `entry_adx` | 158 | +21.45 / +22.88 | <=q33(19.29) | 53·75%·+0.200 | +0.113 | 0.368 | 1.00 |  |
| `d_adx_dir` | 158 | +0.3257 / +0.2107 | >med(0.3093) | 79·76%·+0.199 | +0.150 | 0.208 | 1.00 |  |
| `d_rsi_dir` | 158 | +2.115 / +1.66 | >q67(3.437) | 53·72%·+0.025 | -0.150 | 0.268 | 1.00 |  |
| `entry_gap` | 158 | +0.3574 / +0.3329 | >q67(0.396) | 53·79%·+0.293 | +0.254 | 0.070 | 1.00 |  |
| `entry_ema_gap_5_8` | 158 | +0.08405 / +0.08095 | >med(0.08205) | 79·75%·+0.205 | +0.161 | 0.179 | 1.00 |  |
| `entry_ema_gap_8_13` | 158 | +0.1308 / +0.1181 | >med(0.1254) | 79·76%·+0.226 | +0.203 | 0.093 | 1.00 |  |
| `entry_ema5_stretch` | 158 | +0.1169 / +0.09695 | >q67(0.1474) | 53·74%·+0.046 | -0.118 | 0.345 | 1.00 |  |
| `d_abs_ema20_slope` | 158 | +0.1527 / +0.1469 | >med(0.1503) | 79·73%·+0.206 | +0.164 | 0.170 | 1.00 |  |
| `entry_ema20_slope` | 158 | +0.1527 / +0.1469 | >med(0.1503) | 79·73%·+0.206 | +0.164 | 0.170 | 1.00 |  |
| `entry_adx_delta` | 158 | +0.3257 / +0.2107 | >med(0.3094) | 79·76%·+0.199 | +0.150 | 0.208 | 1.00 |  |
| `entry_range_position` | 158 | +82.7 / +79.05 | >q67(85.43) | 53·79%·+0.262 | +0.207 | 0.172 | 1.00 |  |
| `entry_atr_pct` | 158 | +0.4577 / +0.4567 | >med(0.4577) | 79·71%·+0.115 | -0.019 | 0.827 | 1.00 |  |
| `entry_pair_ema20_ema50_gap_pct` | 158 | +0.3201 / +0.3859 | >q67(0.4622) | 53·77%·+0.233 | +0.164 | 0.200 | 1.00 |  |
| `entry_pair_volume_24h_usd` | 158 | +9.346e+07 / +1.048e+08 | >med(9.617e+07) | 79·65%·+0.059 | -0.130 | 0.353 | 1.00 |  |
| `entry_dist_from_ema13_pct` | 158 | +0.3471 / +0.2943 | >med(0.3316) | 79·75%·+0.178 | +0.107 | 0.375 | 1.00 |  |
| `entry_btc_1h_slope` | 158 | +0.0601 / +0.071 | <=q33(-0.06733) | 53·70%·+0.079 | -0.068 | 0.683 | 1.00 | DAY |
| `d_abs_btc_ema20_slope` | 158 | +0.0544 / +0.06055 | >med(0.0555) | 79·68%·+0.064 | -0.121 | 0.248 | 1.00 | DAY |
| `entry_btc_ema20_slope` | 158 | +0.0544 / +0.06055 | >med(0.0555) | 79·68%·+0.064 | -0.121 | 0.248 | 1.00 | DAY |
| `entry_btc_rsi` | 158 | +62.5 / +62.6 | <=q33(61) | 58·76%·+0.175 | +0.079 | 0.563 | 1.00 | DAY |
| `entry_btc_adx` | 158 | +25.57 / +26.54 | >q67(27.99) | 53·77%·+0.279 | +0.232 | 0.047 | 1.00 | DAY |
| `d_btc_adx_dir` | 158 | +0.4534 / +0.6501 | >q67(0.8338) | 53·64%·+0.021 | -0.156 | 0.324 | 1.00 | DAY |
| `d_btc_rsi_dir` | 158 | +0.75 / +0.2 | >q67(2) | 50·80%·+0.261 | +0.200 | 0.182 | 1.00 | DAY |
| `d_btc_rsi_30m_dir` | 158 | +4.3 / +2.65 | >q67(9.2) | 52·77%·+0.274 | +0.223 | 0.066 | 1.00 | DAY |
| `entry_bull_pct` | 158 | +70.8 / +73.6 | >q67(76.6) | 50·72%·+0.161 | +0.053 | 0.665 | 1.00 | DAY |
| `entry_btc_trend_gap_pct` | 158 | +0.0905 / +0.0958 | >0 | 117·68%·+0.070 | -0.211 | 0.153 | 1.00 | DAY |
| `entry_global_volume_ratio` | 158 | +0.8217 / +0.8229 | >med(0.8217) | 79·71%·+0.067 | -0.114 | 0.412 | 1.00 | DAY |
| `entry_btc_dist_from_ema13_pct` | 158 | +0.1505 / +0.1301 | >q67(0.1836) | 53·77%·+0.212 | +0.132 | 0.369 | 1.00 | DAY |
| `d_fan_ratio` | 158 | +0.7089 / +0.7263 | >q67(0.7755) | 53·68%·+0.018 | -0.161 | 0.223 | 1.00 |  |
| `entry_btc_atr_pct` | 158 | +0.1346 / +0.1343 | >q67(0.1644) | 53·79%·+0.267 | +0.214 | 0.190 | 1.00 | DAY |
| `entry_price_vs_ema5_pct` | 158 | +0.117 / +0.09705 | >q67(0.1476) | 53·74%·+0.046 | -0.118 | 0.345 | 1.00 |  |

Quintile shape (KEEP N·WR·avg per quintile) for the dashboard variables:

| variable | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| `entry_rsi` | ≤57.4: 32·62%·+0.20 | ≤59.4: 31·65%·+0.06 | ≤60.2: 32·69%·-0.07 | ≤62.7: 31·87%·+0.41 | >62.7: 32·72%·+0.04 |
| `entry_adx` | ≤17.8: 32·78%·+0.27 | ≤20.4: 31·74%·+0.13 | ≤22.9: 32·75%·+0.15 | ≤26.2: 31·55%·-0.09 | >26.8: 32·72%·+0.15 |
| `d_adx_dir` | ≤-0.301: 32·72%·+0.08 | ≤0.0977: 31·68%·+0.15 | ≤0.444: 32·72%·+0.08 | ≤0.752: 31·74%·+0.25 | >0.762: 32·69%·+0.08 |
| `d_rsi_dir` | ≤-1.76: 32·66%·+0.10 | ≤0.94: 31·68%·+0.17 | ≤2.95: 32·78%·+0.19 | ≤4.75: 31·68%·+0.18 | >4.87: 32·75%·-0.02 |
| `entry_gap` | ≤0.253: 32·66%·+0.15 | ≤0.314: 31·71%·+0.01 | ≤0.375: 32·69%·-0.02 | ≤0.477: 31·68%·+0.18 | >0.477: 32·81%·+0.29 |
| `entry_ema_gap_5_8` | ≤0.0647: 32·72%·+0.15 | ≤0.077: 31·74%·+0.09 | ≤0.0899: 32·56%·-0.12 | ≤0.112: 31·77%·+0.35 | >0.112: 32·75%·+0.15 |
| `entry_ema_gap_8_13` | ≤0.0927: 32·81%·+0.33 | ≤0.113: 31·52%·-0.22 | ≤0.137: 32·69%·-0.02 | ≤0.168: 31·74%·+0.29 | >0.17: 32·78%·+0.23 |
| `entry_ema5_stretch` | ≤0.0306: 32·62%·+0.08 | ≤0.0871: 31·74%·+0.16 | ≤0.136: 32·72%·+0.19 | ≤0.188: 31·74%·+0.17 | >0.195: 32·72%·+0.03 |
| `d_abs_ema20_slope` | ≤0.103: 32·66%·+0.10 | ≤0.132: 31·74%·+0.08 | ≤0.159: 32·62%·-0.08 | ≤0.189: 31·71%·+0.16 | >0.191: 32·81%·+0.37 |
| `entry_ema20_slope` | ≤0.103: 32·66%·+0.10 | ≤0.132: 31·74%·+0.08 | ≤0.159: 32·62%·-0.08 | ≤0.189: 31·71%·+0.16 | >0.191: 32·81%·+0.37 |
| `entry_adx_delta` | ≤-0.301: 32·72%·+0.08 | ≤0.0977: 31·68%·+0.15 | ≤0.444: 32·72%·+0.08 | ≤0.752: 31·74%·+0.25 | >0.761: 32·69%·+0.08 |
| `entry_range_position` | ≤71.6: 32·59%·+0.08 | ≤78.3: 31·71%·+0.14 | ≤84: 32·72%·+0.04 | ≤87.9: 31·74%·+0.08 | >88: 32·78%·+0.29 |
| `entry_atr_pct` | ≤0.348: 32·75%·+0.13 | ≤0.436: 31·74%·+0.12 | ≤0.486: 32·69%·+0.20 | ≤0.583: 31·65%·+0.02 | >0.588: 32·72%·+0.14 |
| `entry_pair_ema20_ema50_gap_pct` | ≤0.0228: 32·66%·+0.06 | ≤0.269: 31·81%·+0.27 | ≤0.398: 32·69%·+0.07 | ≤0.663: 31·61%·+0.03 | >0.675: 32·78%·+0.20 |
| `entry_pair_volume_24h_usd` | ≤5.82e+07: 32·81%·+0.22 | ≤7.73e+07: 31·74%·+0.08 | ≤1.1e+08: 32·69%·+0.20 | ≤1.72e+08: 31·68%·+0.09 | >1.72e+08: 32·62%·+0.04 |
| `entry_dist_from_ema13_pct` | ≤0.23: 32·62%·+0.08 | ≤0.29: 31·65%·+0.02 | ≤0.375: 32·75%·+0.20 | ≤0.45: 31·71%·+0.11 | >0.451: 32·81%·+0.21 |
| `entry_btc_1h_slope` | ≤-0.134: 32·75%·+0.16 | ≤0.0358: 31·71%·+0.06 | ≤0.0822: 32·69%·+0.11 | ≤0.137: 31·71%·+0.24 | >0.137: 32·69%·+0.06 |
| `d_abs_btc_ema20_slope` | ≤0.0319: 32·81%·+0.24 | ≤0.0482: 31·61%·+0.07 | ≤0.0619: 32·78%·+0.22 | ≤0.0811: 31·65%·-0.10 | >0.0811: 32·69%·+0.19 |
| `entry_btc_ema20_slope` | ≤0.0319: 32·81%·+0.24 | ≤0.0482: 31·61%·+0.07 | ≤0.0619: 32·78%·+0.22 | ≤0.0811: 31·65%·-0.10 | >0.0811: 32·69%·+0.19 |
| `entry_btc_rsi` | ≤60.2: 32·78%·+0.21 | ≤61.5: 31·71%·+0.12 | ≤63.8: 32·62%·+0.05 | ≤66.1: 31·65%·+0.01 | >66.2: 32·78%·+0.23 |
| `entry_btc_adx` | ≤21.1: 32·69%·+0.16 | ≤22.8: 31·74%·+0.03 | ≤27.4: 32·59%·-0.00 | ≤31.6: 31·81%·+0.26 | >31.7: 32·72%·+0.18 |
| `d_btc_adx_dir` | ≤-0.525: 32·88%·+0.33 | ≤0.227: 31·61%·-0.04 | ≤0.666: 32·78%·+0.21 | ≤1.09: 31·65%·+0.01 | >1.1: 32·62%·+0.10 |
| `d_btc_rsi_dir` | ≤-2.5: 32·62%·+0.06 | ≤0: 31·74%·+0.13 | ≤1.3: 32·59%·-0.13 | ≤3: 31·74%·+0.24 | >3.1: 32·84%·+0.33 |
| `d_btc_rsi_30m_dir` | ≤-4.3: 32·72%·+0.12 | ≤1.9: 31·58%·-0.11 | ≤7.2: 32·78%·+0.14 | ≤14.4: 31·71%·+0.31 | >14.4: 32·75%·+0.16 |
| `entry_bull_pct` | ≤56.8: 32·78%·+0.20 | ≤66.7: 31·71%·+0.07 | ≤74.5: 32·62%·+0.01 | ≤79.1: 31·77%·+0.28 | >79.2: 32·66%·+0.07 |
| `entry_btc_trend_gap_pct` | ≤-0.0359: 32·84%·+0.43 | ≤0.0654: 31·71%·-0.04 | ≤0.141: 32·50%·-0.08 | ≤0.228: 31·81%·+0.18 | >0.231: 32·69%·+0.13 |
| `entry_global_volume_ratio` | ≤0.567: 32·69%·+0.19 | ≤0.756: 31·68%·+0.14 | ≤0.903: 32·75%·+0.15 | ≤1.09: 31·71%·+0.11 | >1.1: 32·72%·+0.03 |
| `entry_btc_dist_from_ema13_pct` | ≤0.0748: 32·69%·+0.03 | ≤0.124: 31·74%·+0.17 | ≤0.164: 32·59%·+0.01 | ≤0.217: 31·74%·+0.17 | >0.222: 32·78%·+0.25 |
| `d_fan_ratio` | ≤0.575: 32·75%·+0.28 | ≤0.671: 31·68%·+0.01 | ≤0.746: 32·81%·+0.34 | ≤0.803: 31·52%·-0.14 | >0.807: 32·78%·+0.13 |
| `entry_btc_atr_pct` | ≤0.111: 32·72%·+0.27 | ≤0.128: 31·68%·+0.08 | ≤0.155: 32·66%·-0.07 | ≤0.211: 31·68%·+0.06 | >0.215: 32·81%·+0.28 |
| `entry_price_vs_ema5_pct` | ≤0.0209: 32·62%·+0.06 | ≤0.0872: 31·74%·+0.17 | ≤0.136: 32·72%·+0.19 | ≤0.188: 31·74%·+0.17 | >0.196: 32·72%·+0.03 |

No dashboard variable reaches family-wise significance; the BTC ADX / regime pair is the only one whose sign is confirmed in all three replay regimes (H1).

## 7 · Top separators — before → after per era (KEEP, today's stack)

Thresholds are the KEEP-pool cuts and are FROZEN as printed. Full candidate list with every test's OOS / replay / B25 column: `reports/PREP_B25_ML_WL_candidates.csv`; all 1D tests `_screen1d.csv`; all 2D tests `_screen2d.csv`.

**H1 STRONG_BULL regime (winner side)** — KEEP scored 158/158 · zone 53·77%·+0.279 (95% day-CI [+0.09, +0.47], P(avg<0) 0.00) vs rest 105·68%·+0.047 · Δ +0.232
- DAY units: zone on 31 days · mean of day-means +0.317 (rest +0.089) · days with zone mean < 0: 26% · 2-min scan windows 49
- concentration: worst day 2026-10-12 23% of the zone's net loss · worst pair WLDUSDT 19% · gain side: best day 13% / best pair 19% of the net gain
- without the washed-out window: zone 47·74%·+0.211 vs rest 91·63%·-0.025 · washed-out only: zone 6·100%·+0.811
- TRAD (as traded, all full-size): zone 82·70%·+0.119 vs rest 128·62%·-0.035 · PROBES (separate): 35·46%·-0.224
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 105·68%·+0.047 · Δ avg -0.078 → with the 30–50 % in-sample haircut -0.039 … -0.055

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 25·92%·+0.389 | 7·100%·+0.828 | 25·92%·+0.389 |
| G1 B1 | 15·80%·+0.256 | 10·70%·+0.174 | 5·100%·+0.421 | 10·70%·+0.174 |
| G2 B2 | 10·80%·+0.191 | 2·50%·-0.362 | 8·88%·+0.329 | 2·50%·-0.362 |
| G3 B3 | 22·77%·+0.147 | 16·75%·+0.115 | 6·83%·+0.230 | 16·75%·+0.115 |
| G4 B4-B11 | 14·79%·+0.060 | 11·73%·-0.134 | 3·100%·+0.769 | 11·73%·-0.134 |
| G5 B12-B16 | 28·57%·-0.083 | 19·42%·-0.313 | 9·89%·+0.402 | 19·42%·-0.313 |
| G6 B17-B24 | 37·49%·-0.091 | 22·55%·-0.013 | 15·40%·-0.205 | 22·55%·-0.013 |

- B25 provisional: scored 9/9 · zone 5·60%·+0.000 (ENAUSDT -0.89, LTCUSDT +0.71, STRKUSDT +0.09, UNIUSDT +0.40, ADAUSDT -0.31) vs rest 4·100%·+0.350

**H1b BTC ADX > 27.99 (alias of H1)** — KEEP scored 158/158 · zone 53·77%·+0.279 (95% day-CI [+0.09, +0.47], P(avg<0) 0.00) vs rest 105·68%·+0.047 · Δ +0.232
- DAY units: zone on 31 days · mean of day-means +0.317 (rest +0.089) · days with zone mean < 0: 26% · 2-min scan windows 49
- concentration: worst day 2026-10-12 23% of the zone's net loss · worst pair WLDUSDT 19% · gain side: best day 13% / best pair 19% of the net gain
- without the washed-out window: zone 47·74%·+0.211 vs rest 91·63%·-0.025 · washed-out only: zone 6·100%·+0.811
- TRAD (as traded, all full-size): zone 82·70%·+0.119 vs rest 128·62%·-0.035 · PROBES (separate): 54·50%·-0.141
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 105·68%·+0.047 · Δ avg -0.078 → with the 30–50 % in-sample haircut -0.039 … -0.055

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 25·92%·+0.389 | 7·100%·+0.828 | 25·92%·+0.389 |
| G1 B1 | 15·80%·+0.256 | 10·70%·+0.174 | 5·100%·+0.421 | 10·70%·+0.174 |
| G2 B2 | 10·80%·+0.191 | 2·50%·-0.362 | 8·88%·+0.329 | 2·50%·-0.362 |
| G3 B3 | 22·77%·+0.147 | 16·75%·+0.115 | 6·83%·+0.230 | 16·75%·+0.115 |
| G4 B4-B11 | 14·79%·+0.060 | 11·73%·-0.134 | 3·100%·+0.769 | 11·73%·-0.134 |
| G5 B12-B16 | 28·57%·-0.083 | 19·42%·-0.313 | 9·89%·+0.402 | 19·42%·-0.313 |
| G6 B17-B24 | 37·49%·-0.091 | 22·55%·-0.013 | 15·40%·-0.205 | 22·55%·-0.013 |

- B25 provisional: scored 9/9 · zone 5·60%·+0.000 (ENAUSDT -0.89, LTCUSDT +0.71, STRKUSDT +0.09, UNIUSDT +0.40, ADAUSDT -0.31) vs rest 4·100%·+0.350

**H2 pair ADXΔ ≤ 0.309 ∧ regime ≠ STRONG_BULL (loser)** — KEEP scored 158/158 · zone 58·59%·-0.110 (95% day-CI [-0.31, +0.13], P(avg<0) 0.84) vs rest 100·78%·+0.261 · Δ -0.371
- DAY units: zone on 39 days · mean of day-means -0.018 (rest +0.273) · days with zone mean < 0: 41% · 2-min scan windows 55
- concentration: worst day 2026-10-16 13% of the zone's net loss · worst pair UNIUSDT 13% · gain side: best day 20% / best pair 20% of the net gain
- without the washed-out window: zone 53·55%·-0.160 vs rest 85·74%·+0.190 · washed-out only: zone 5·100%·+0.419
- TRAD (as traded, all full-size): zone 74·54%·-0.186 vs rest 136·71%·+0.139 · PROBES (separate): 90·51%·-0.146
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 100·78%·+0.261 · Δ avg +0.136 → with the 30–50 % in-sample haircut +0.068 … +0.095

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 21·100%·+0.599 | 11·82%·+0.266 | 21·100%·+0.599 |
| G1 B1 | 15·80%·+0.256 | 6·100%·+0.449 | 9·67%·+0.128 | 6·100%·+0.449 |
| G2 B2 | 10·80%·+0.191 | 10·80%·+0.191 | 0 | 10·80%·+0.191 |
| G3 B3 | 22·77%·+0.147 | 14·79%·+0.238 | 8·75%·-0.013 | 14·79%·+0.238 |
| G4 B4-B11 | 14·79%·+0.060 | 8·75%·+0.149 | 6·83%·-0.060 | 8·75%·+0.149 |
| G5 B12-B16 | 28·57%·-0.083 | 17·76%·+0.269 | 11·27%·-0.627 | 17·76%·+0.269 |
| G6 B17-B24 | 37·49%·-0.091 | 24·54%·-0.010 | 13·38%·-0.240 | 24·54%·-0.010 |

- B25 provisional: scored 9/9 · zone 2·100%·+0.458 (BRUSDT +0.09, LIGHTUSDT +0.82) vs rest 7·71%·+0.069

**H3 EMA5−8 gap ≤ 0.082 (loser; dashboard Gap5-8 — the signed stamp holds the same values)** — KEEP scored 158/158 · zone 79·67%·+0.044 (95% day-CI [-0.13, +0.22], P(avg<0) 0.30) vs rest 79·75%·+0.205 · Δ -0.161
- DAY units: zone on 48 days · mean of day-means +0.062 (rest +0.231) · days with zone mean < 0: 31% · 2-min scan windows 74
- concentration: worst day 2026-10-16 18% of the zone's net loss · worst pair BCHUSDT 13% · gain side: best day 9% / best pair 14% of the net gain
- without the washed-out window: zone 70·63%·-0.011 vs rest 68·71%·+0.124 · washed-out only: zone 9·100%·+0.470
- TRAD (as traded, all full-size): zone 106·59%·-0.054 vs rest 104·71%·+0.105 · PROBES (separate): 98·48%·-0.112
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 79·75%·+0.205 · Δ avg +0.081 → with the 30–50 % in-sample haircut +0.040 … +0.056

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 17·94%·+0.547 | 15·93%·+0.414 | 17·94%·+0.547 |
| G1 B1 | 15·80%·+0.256 | 10·70%·+0.237 | 5·100%·+0.295 | 10·70%·+0.237 |
| G2 B2 | 10·80%·+0.191 | 7·71%·+0.109 | 3·100%·+0.382 | 7·71%·+0.109 |
| G3 B3 | 22·77%·+0.147 | 11·64%·-0.036 | 11·91%·+0.329 | 11·64%·-0.036 |
| G4 B4-B11 | 14·79%·+0.060 | 5·100%·+0.110 | 9·67%·+0.032 | 5·100%·+0.110 |
| G5 B12-B16 | 28·57%·-0.083 | 9·67%·+0.020 | 19·53%·-0.131 | 9·67%·+0.020 |
| G6 B17-B24 | 37·49%·-0.091 | 20·65%·+0.171 | 17·29%·-0.399 | 20·65%·+0.171 |

- B25 provisional: scored 9/9 · zone 7·71%·+0.069 (ENAUSDT -0.89, LTCUSDT +0.71, STRKUSDT +0.09, UNIUSDT +0.40, ADAUSDT -0.31, UNIUSDT +0.37, LINKUSDT +0.11) vs rest 2·100%·+0.458

**H4 BTC 15-min return ≤ 0 (loser)** — KEEP scored 158/158 · zone 40·57%·-0.100 (95% day-CI [-0.31, +0.11], P(avg<0) 0.83) vs rest 118·75%·+0.200 · Δ -0.300
- DAY units: zone on 27 days · mean of day-means -0.115 (rest +0.211) · days with zone mean < 0: 52% · 2-min scan windows 37
- concentration: worst day 2026-09-29 20% of the zone's net loss · worst pair 0GUSDT 11% · gain side: best day 24% / best pair 16% of the net gain
- without the washed-out window: zone 38·55%·-0.134 vs rest 100·71%·+0.127 · washed-out only: zone 2·100%·+0.556
- TRAD (as traded, all full-size): zone 58·53%·-0.182 vs rest 152·70%·+0.104 · PROBES (separate): 0
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 118·75%·+0.200 · Δ avg +0.076 → with the 30–50 % in-sample haircut +0.038 … +0.053

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 28·96%·+0.537 | 4·75%·+0.116 | 28·96%·+0.537 |
| G1 B1 | 15·80%·+0.256 | 10·90%·+0.426 | 5·60%·-0.083 | 10·90%·+0.426 |
| G2 B2 | 10·80%·+0.191 | 4·50%·-0.286 | 6·100%·+0.509 | 4·50%·-0.286 |
| G3 B3 | 22·77%·+0.147 | 17·82%·+0.229 | 5·60%·-0.132 | 17·82%·+0.229 |
| G4 B4-B11 | 14·79%·+0.060 | 12·75%·+0.012 | 2·100%·+0.347 | 12·75%·+0.012 |
| G5 B12-B16 | 28·57%·-0.083 | 22·64%·+0.018 | 6·33%·-0.451 | 22·64%·+0.018 |
| G6 B17-B24 | 37·49%·-0.091 | 25·56%·+0.043 | 12·33%·-0.368 | 25·56%·+0.043 |

- B25 provisional: scored 9/9 · zone 4·75%·+0.323 (UNIUSDT +0.40, ADAUSDT -0.31, LIGHTUSDT +0.82, UNIUSDT +0.37) vs rest 5·80%·+0.022

**H5 top-10 corr15 > 0.365 (loser)** — KEEP scored 140/158 · zone 47·53%·-0.157 (95% day-CI [-0.40, +0.09], P(avg<0) 0.90) vs rest 93·81%·+0.302 · Δ -0.459
- DAY units: zone on 31 days · mean of day-means -0.112 (rest +0.316) · days with zone mean < 0: 52% · 2-min scan windows 46
- concentration: worst day 2026-09-29 13% of the zone's net loss · worst pair BCHUSDT 12% · gain side: best day 15% / best pair 23% of the net gain
- without the washed-out window: zone 42·48%·-0.200 vs rest 78·77%·+0.219 · washed-out only: zone 5·100%·+0.207
- TRAD (as traded, all full-size): zone 67·54%·-0.201 vs rest 125·71%·+0.158 · PROBES (separate): 0
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 111·78%·+0.244 · Δ avg +0.119 → with the 30–50 % in-sample haircut +0.060 … +0.083

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 24·96%·+0.555 | 8·88%·+0.274 | 24·96%·+0.555 |
| G1 B1 | 15·80%·+0.256 | 14·79%·+0.203 | 1·100%·+1.000 | 14·79%·+0.203 |
| G2 B2 | 10·80%·+0.191 | 9·78%·+0.181 | 1·100%·+0.279 | 9·78%·+0.181 |
| G3 B3 | 22·77%·+0.147 | 17·71%·+0.055 | 5·100%·+0.458 | 17·71%·+0.055 |
| G4 B4-B11 | 14·79%·+0.060 | 7·86%·+0.270 | 7·71%·-0.151 | 7·86%·+0.270 |
| G5 B12-B16 | 28·57%·-0.083 | 14·86%·+0.441 | 14·29%·-0.607 | 14·86%·+0.441 |
| G6 B17-B24 | 37·49%·-0.091 | 26·62%·+0.009 | 11·18%·-0.326 | 8·50%·+0.162 |

- B25 provisional: scored 0/9 · zone 0 (—) vs rest 0

**H6 breadth ret15 ≤ med ∧ pair vol15 ≤ med (winner)** — KEEP scored 158/158 · zone 31·74%·+0.448 (95% day-CI [+0.06, +0.84], P(avg<0) 0.01) vs rest 127·70%·+0.045 · Δ +0.403
- DAY units: zone on 24 days · mean of day-means +0.440 (rest +0.053) · days with zone mean < 0: 33% · 2-min scan windows 30
- concentration: worst day 2026-08-01 15% of the zone's net loss · worst pair ALGOUSDT 17% · gain side: best day 26% / best pair 13% of the net gain
- without the washed-out window: zone 26·69%·+0.362 vs rest 112·66%·-0.016 · washed-out only: zone 5·100%·+0.896
- TRAD (as traded, all full-size): zone 44·73%·+0.289 vs rest 166·63%·-0.045 · PROBES (separate): 0
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 127·70%·+0.045 · Δ avg -0.079 → with the 30–50 % in-sample haircut -0.040 … -0.055

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 26·92%·+0.395 | 6·100%·+0.873 | 26·92%·+0.395 |
| G1 B1 | 15·80%·+0.256 | 12·75%·+0.079 | 3·100%·+0.967 | 12·75%·+0.079 |
| G2 B2 | 10·80%·+0.191 | 9·89%·+0.307 | 1·0%·-0.847 | 9·89%·+0.307 |
| G3 B3 | 22·77%·+0.147 | 15·73%·+0.126 | 7·86%·+0.191 | 15·73%·+0.126 |
| G4 B4-B11 | 14·79%·+0.060 | 13·77%·-0.012 | 1·100%·+0.996 | 13·77%·-0.012 |
| G5 B12-B16 | 28·57%·-0.083 | 23·57%·-0.178 | 5·60%·+0.356 | 23·57%·-0.178 |
| G6 B17-B24 | 37·49%·-0.091 | 29·48%·-0.201 | 8·50%·+0.310 | 29·48%·-0.201 |

- B25 provisional: scored 9/9 · zone 4·75%·+0.176 (ENAUSDT -0.89, UNIUSDT +0.40, LIGHTUSDT +0.82, UNIUSDT +0.37) vs rest 5·80%·+0.139

**H7 imp spike_30 ≤ med ∧ corr15 ≤ med (winner)** — KEEP scored 140/158 · zone 30·87%·+0.466 (95% day-CI [+0.22, +0.75], P(avg<0) 0.00) vs rest 110·67%·+0.061 · Δ +0.404
- DAY units: zone on 25 days · mean of day-means +0.464 (rest +0.079) · days with zone mean < 0: 12% · 2-min scan windows 28
- concentration: worst day 2026-07-28 37% of the zone's net loss · worst pair WLDUSDT 31% · gain side: best day 16% / best pair 23% of the net gain
- without the washed-out window: zone 23·83%·+0.420 vs rest 97·63%·-0.010 · washed-out only: zone 7·100%·+0.616
- TRAD (as traded, all full-size): zone 41·80%·+0.348 vs rest 151·61%·-0.052 · PROBES (separate): 0
- sleeve before → after removing the zone (KEEP): 158·71%·+0.124 → 128·67%·+0.045 · Δ avg -0.080 → with the 30–50 % in-sample haircut -0.040 … -0.056

| era group | before (all) | after (zone removed) | zone | rest (scored) |
|---|---|---|---|---|
| G0 BASE | 32·94%·+0.485 | 23·91%·+0.445 | 9·100%·+0.586 | 23·91%·+0.445 |
| G1 B1 | 15·80%·+0.256 | 9·78%·+0.088 | 6·83%·+0.508 | 9·78%·+0.088 |
| G2 B2 | 10·80%·+0.191 | 8·75%·+0.117 | 2·100%·+0.487 | 8·75%·+0.117 |
| G3 B3 | 22·77%·+0.147 | 17·76%·+0.166 | 5·80%·+0.080 | 17·76%·+0.166 |
| G4 B4-B11 | 14·79%·+0.060 | 11·82%·+0.021 | 3·67%·+0.203 | 11·82%·+0.021 |
| G5 B12-B16 | 28·57%·-0.083 | 26·54%·-0.146 | 2·100%·+0.736 | 26·54%·-0.146 |
| G6 B17-B24 | 37·49%·-0.091 | 34·47%·-0.163 | 3·67%·+0.731 | 16·25%·-0.280 |

- B25 provisional: scored 0/9 · zone 0 (—) vs rest 0

## 8 · Survivors vs the bars

Candidates = 1D p<0.05 ∧ |Δ|≥0.10, 2D p<0.01 ∧ |Δ|≥0.20 (top 150). OOS✓ = LOEO same sign in ≥75 % of ≥4 held-out era groups ∧ 2nd-half same sign ∧ TRAD same sign. Replay columns are refute-only.

- 1D candidates 21 (OOS✓ 13); 2D candidates 150 (OOS✓ 101). Expectancy-bar passes 5 (with OOS✓ 2); Pattern-W passes 82 (with OOS✓ 65; with OOS✓ ∧ replay same sign in all 3 regimes 13).
- Every family-wise p ≥ 0.045 (1D) / ≥ 0.5 (2D): **no candidate survives the multiple-testing correction.** The bar passes below are therefore in-sample patterns, not evidence.

### 8a · Loser zones that meet the expectancy bar (in-sample)

| zone | N | days | WR | avg | P(avg<0) | LOEO | 2nd half | TRAD Δ | replay (refute-only) | B25 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| entry_gap_5_8_signed_pct <=med(0.0819) | 22 | 11 | 32% | -0.404 | 0.99 | 1/1 | +nan | -0.52 | 2025 BU +0.026(n1171,P>0 0.73) · 2026-H1 +0.042(n483,P>0 0.74) · 2026-H2 -0.106(n238,P>0 0.19) | B25 zone 7/9 scored: ENAUSDT-0.89, LTCUSDT+0.71, STRKUSDT+0.09, UNIUSDT+0.40, ADAUSDT-0.31, UNIUSDT+0.37, LINKUSDT+0.11 | REFUTED (older eras Δ 0.00; B25 opposite) |
| d_gap5_20_chg <=q33(-0.0154) | 15 | 9 | 20% | -0.452 | 1.00 | 0/0 | +nan | -0.47 | 2025 BU -0.007(n554,P>0 0.43) · 2026-H1 -0.011(n195,P>0 0.44) · 2026-H2 +0.070(n157,P>0 0.78) | B25 zone 1/9 scored: ADAUSDT-0.31 | no OOS / replay null → not a candidate |
| btc_vol5_x24h>med ∧ imp_L3_sol<=med | 18 | 14 | 50% | -0.291 | 0.97 | 3/4 | -0.44 | -0.43 | 2025 BU +0.039(n337,P>0 0.71) · 2026-H1 -0.080(n144,P>0 0.14) · 2026-H2 -0.059(n82,P>0 0.29) | B25 zone 0/0 scored: none | no OOS / replay null → not a candidate |
| btc_vol5_x24h>med ∧ imp_L4_xrp<=med | 18 | 14 | 50% | -0.311 | 0.99 | 4/5 | -0.55 | -0.51 | 2025 BU -0.011(n352,P>0 0.42) · 2026-H1 +0.005(n151,P>0 0.50) · 2026-H2 -0.028(n79,P>0 0.40) | B25 zone 0/0 scored: none | no OOS / replay null → not a candidate |
| d_rsi_dir>med ∧ imp_L4_xrp<=med | 20 | 17 | 55% | -0.253 | 0.96 | 4/6 | -0.43 | -0.38 | 2025 BU +0.002(n349,P>0 0.51) · 2026-H1 +0.121(n181,P>0 0.93) · 2026-H2 -0.079(n72,P>0 0.22) | B25 zone 0/0 scored: none | no OOS / replay null → not a candidate |

### 8b · Loser zones with OOS✓ and replay same sign in all three regimes (fail the bar only on confidence)

| zone | N | days | WR | avg | P(avg<0) | fails | LOEO | 2nd half | replay |
|---|---|---|---|---|---|---|---|---|---|
| d_adx_dir<=med ∧ cat__entry_btc_regime==STRONG_BULL<=med | 58 | 39 | 59% | -0.110 | 0.84 | P<95% | 6/6 | -0.43 | 2025 BU -0.070(n1060,P>0 0.04) · 2026-H1 -0.068(n474,P>0 0.15) · 2026-H2 -0.230(n183,P>0 0.01) |
| d_adx_dir<=med ∧ cat__entry_btc_regime==HEALTHY_BULL>med | 55 | 38 | 58% | -0.086 | 0.78 | P<95% | 5/6 | -0.37 | 2025 BU -0.069(n1051,P>0 0.04) · 2026-H1 -0.060(n471,P>0 0.17) · 2026-H2 -0.217(n179,P>0 0.01) |
| entry_adx_delta<=med ∧ cat__entry_btc_regime==STRONG_BULL<=med | 58 | 39 | 59% | -0.110 | 0.84 | P<95% | 6/6 | -0.43 | 2025 BU -0.068(n1062,P>0 0.04) · 2026-H1 -0.068(n474,P>0 0.15) · 2026-H2 -0.230(n183,P>0 0.01) |
| entry_adx_delta<=med ∧ cat__entry_btc_regime==HEALTHY_BULL>med | 55 | 38 | 58% | -0.086 | 0.78 | P<95% | 5/6 | -0.37 | 2025 BU -0.067(n1053,P>0 0.05) · 2026-H1 -0.060(n471,P>0 0.17) · 2026-H2 -0.217(n179,P>0 0.01) |
| cat__entry_btc_regime==STRONG_BULL<=med ∧ imp_corr15>med | 53 | 30 | 62% | -0.060 | 0.69 | WR≥BE,P<95% | 4/5 | -0.49 | 2025 BU -0.065(n1638,P>0 0.06) · 2026-H1 -0.093(n453,P>0 0.07) · 2026-H2 -0.155(n157,P>0 0.03) |
| entry_gap<=med ∧ imp_corr15>med | 35 | 24 | 51% | -0.160 | 0.90 | P<95% | 4/4 | -0.39 | 2025 BU -0.019(n925,P>0 0.31) · 2026-H1 -0.016(n253,P>0 0.42) · 2026-H2 -0.242(n116,P>0 0.00) |
| cat__entry_btc_regime==STRONG_BULL ==0 | 105 | 55 | 68% | +0.047 | 0.30 | WR≥BE,P<95% | 5/6 | -0.25 | 2025 BU -0.064(n1959,P>0 0.09) · 2026-H1 -0.120(n899,P>0 0.07) · 2026-H2 -0.056(n308,P>0 0.28) |

The ADXΔ ≤ med ∧ (≠ STRONG_BULL / = HEALTHY_BULL) rows are one zone (two labels of the same fills); the corr15 rows need the Oct-15 impulse grid (not scored after 10-15).

## 9 · Winner signatures (Pattern-W bar — multiplier candidates)

Pattern-W passes with OOS✓ and replay same sign in all three regimes (base rate of 'all three agree' for random 2D tests = 16%, so ~1 in 6 of these is expected by luck):

| zone | N | days | WR | avg | $ (stack) | LOEO | 2nd half | ex-washed | replay | B25 |
|---|---|---|---|---|---|---|---|---|---|---|
| entry_pattern_w4_match<=med ∧ cat__entry_btc_regime==STRONG_BULL>med | 35 | 28 | 91% | +0.450 | +108 | 6/7 | +0.56 | 29·+0.38 | 2025 BU +0.027(n452,P>0 0.69) · 2026-H1 +0.119(n204,P>0 0.89) · 2026-H2 +0.044(n155,P>0 0.67) | B25 zone 4/9 scored: ENAUSDT-0.89, LTCUSDT+0.71, STRKUSDT+0.09, UNIUSDT+0.40 |
| btc_r5>med ∧ imp_L1_btc>med | 50 | 33 | 84% | +0.367 | +147 | 4/5 | +0.32 | 39·+0.30 | 2025 BU +0.012(n815,P>0 0.64) · 2026-H1 +0.009(n390,P>0 0.55) · 2026-H2 +0.141(n175,P>0 0.93) | B25 zone 0/0 scored: none |
| br_ret15<=med ∧ pair_vol15_x24h<=med | 31 | 24 | 74% | +0.448 | +122 | 5/5 | +0.53 | 26·+0.36 | 2025 BU +0.058(n727,P>0 0.89) · 2026-H1 +0.036(n369,P>0 0.71) · 2026-H2 +0.261(n140,P>0 0.97) | B25 zone 4/9 scored: ENAUSDT-0.89, UNIUSDT+0.40, LIGHTUSDT+0.82, UNIUSDT+0.37 |
| imp_volslope_15>med ∧ imp_corr15<=med | 31 | 26 | 81% | +0.516 | +141 | 5/5 | +0.43 | 24·+0.45 | 2025 BU +0.050(n277,P>0 0.76) · 2026-H1 +0.020(n315,P>0 0.60) · 2026-H2 +0.263(n165,P>0 1.00) | B25 zone 0/0 scored: none |
| imp_spike_30<=med ∧ imp_corr15<=med | 30 | 25 | 87% | +0.466 | +118 | 4/5 | +0.43 | 23·+0.42 | 2025 BU +0.108(n316,P>0 0.93) · 2026-H1 +0.095(n338,P>0 0.92) · 2026-H2 +0.083(n130,P>0 0.73) | B25 zone 0/0 scored: none |
| entry_ema_gap_5_8>med ∧ d_fan_ratio<=med | 35 | 27 | 83% | +0.485 | +139 | 5/6 | +0.80 | 30·+0.43 | 2025 BU +0.015(n544,P>0 0.66) · 2026-H1 +0.052(n236,P>0 0.77) · 2026-H2 +0.178(n128,P>0 0.90) | B25 zone 2/9 scored: BRUSDT+0.09, LIGHTUSDT+0.82 |
| pair_vol15_x24h<=med ∧ imp_corr15<=med | 37 | 27 | 84% | +0.435 | +140 | 5/6 | +1.07 | 32·+0.40 | 2025 BU +0.069(n256,P>0 0.82) · 2026-H1 +0.014(n331,P>0 0.58) · 2026-H2 +0.288(n124,P>0 0.99) | B25 zone 0/0 scored: none |
| btc_vol5_x24h<=med ∧ imp_corr15<=med | 34 | 23 | 85% | +0.441 | +114 | 5/5 | +0.69 | 30·+0.43 | 2025 BU +0.030(n206,P>0 0.66) · 2026-H1 +0.029(n316,P>0 0.68) · 2026-H2 +0.163(n130,P>0 0.92) | B25 zone 0/0 scored: none |
| entry_btc_ema20_slope<=med ∧ imp_spike_30<=med | 35 | 31 | 80% | +0.406 | +126 | 5/6 | +0.09 | 29·+0.34 | 2025 BU +0.090(n581,P>0 0.96) · 2026-H1 +0.027(n221,P>0 0.63) · 2026-H2 +0.108(n142,P>0 0.84) | B25 zone 0/0 scored: none |
| entry_btc_ema20_slope<=med ∧ d_btc_rsi_30m_dir>med | 35 | 27 | 83% | +0.412 | +114 | 5/6 | +0.14 | 29·+0.37 | 2025 BU +0.057(n466,P>0 0.83) · 2026-H1 +0.021(n188,P>0 0.59) · 2026-H2 +0.264(n124,P>0 0.99) | B25 zone 0/9 scored: none |
| entry_btc_adx_prev >q67(28.3) | 53 | 32 | 77% | +0.287 | +99 | 5/6 | +0.32 | 47·+0.22 | 2025 BU +0.083(n618,P>0 0.95) · 2026-H1 +0.140(n250,P>0 0.95) · 2026-H2 +0.003(n198,P>0 0.53) | B25 zone 4/9 scored: ENAUSDT-0.89, LTCUSDT+0.71, STRKUSDT+0.09, UNIUSDT+0.40 |
| entry_btc_adx >q67(27.99) | 53 | 31 | 77% | +0.279 | +101 | 5/6 | +0.25 | 47·+0.21 | 2025 BU +0.065(n660,P>0 0.92) · 2026-H1 +0.115(n273,P>0 0.92) · 2026-H2 +0.056(n213,P>0 0.71) | B25 zone 5/9 scored: ENAUSDT-0.89, LTCUSDT+0.71, STRKUSDT+0.09, UNIUSDT+0.40, ADAUSDT-0.31 |
| cat__entry_btc_regime==STRONG_BULL ==1 | 53 | 31 | 77% | +0.279 | +101 | 5/6 | +0.25 | 47·+0.21 | 2025 BU +0.064(n658,P>0 0.91) · 2026-H1 +0.120(n272,P>0 0.93) · 2026-H2 +0.056(n213,P>0 0.71) | B25 zone 5/9 scored: ENAUSDT-0.89, LTCUSDT+0.71, STRKUSDT+0.09, UNIUSDT+0.40, ADAUSDT-0.31 |

Read: the recurring theme is a **quiet, idiosyncratic tape** (top-10 coins NOT moving together, no BTC spike, low pair 15-min volume burst, flat 15-min breadth) and **BTC in a strong trend (STRONG_BULL / ADX > 28)**. None clears the family-wise null and each is in-sample; the newest era group (B17–B24) is the weakest block for most of them. Multipliers stay off the table (operator DL 289: momentum longs at 1×; memory: filters, not sizing). Registered as observe lines in §11.

## 10 · Post-entry (descriptive only — not entry separators)

| variable | scored | median winners | median losers |
|---|---|---|---|
| `peak_pnl` | 158 | +0.762 | +0.030 |
| `trough_pnl` | 158 | -0.285 | -0.754 |
| `dur_min` | 158 | +22.200 | +15.327 |
| `peak_min` | 138 | +17.437 | +3.406 |
| `trough_min` | 152 | +3.117 | +15.086 |
| `post_Y_btc` | 140 | +0.051 | -0.077 |
| `post_Y_alt` | 140 | +0.055 | -0.146 |
| `btc_d5` | 158 | -0.004 | -0.013 |
| `btc_d15` | 158 | +0.051 | -0.061 |
| `br_d5` | 158 | +0.024 | -0.032 |
| `br_d15` | 158 | +0.060 | -0.146 |
| `pnl_m1` | 148 | +0.000 | -0.058 |
| `pnl_m3` | 148 | +0.053 | -0.142 |
| `pnl_m5` | 148 | +0.056 | -0.166 |
| `pnl_m15` | 148 | +0.190 | -0.432 |

`post_Y_btc` / `post_Y_alt` = BTC / top-50 return over the 15 min AFTER entry: winners see BTC rise, losers see it fall — the same finding as the 10-15 study (the trade's fate is decided by the next 15 min of BTC, which no entry variable predicts). Duration and peak timing: losers are stopped within ~the first half hour; winners peak late.

## 11 · What we could NOT test (blind spots)

1. **Never stamped on momentum longs** (0 scored in every era): Bull-Run door stamps `entry_br_*`, Bear-Run `entry_bear_*`, FRENZY / FLATBASE / SURGE / WILLY stamps, fade-only `entry_btc_4h_ema50_200_gap_pct` and `entry_pair_1d_ndi`, order-book `entry_book_*` (new, 0 ML fills). `entry_rsi_prev2` (the RSI-momentum filter's own leg) is not stamped — RSIDir uses `rsi_prev`.
2. **Low coverage, 1D scored-only, no 2D, no LOEO (all recent-era):** BTC 24h / 30-day / 72h stamps (`entry_btc_off24h_pct`, `off24lo`, `off30d_high`, `r72`, `eff72`, `above72`, heat flags; 66–72 KEEP fills since mid-Sep), signed gaps and EMA5−20 change (43), `entry_pair_1h_ema20_200_gap_pct`, `entry_btc_1d_ret_pct`, `entry_btc_ema50_100_gap_pct`, `entry_eth_5m_ret1_pct` (43), `entry_cmc_rank` / `entry_mcap_usd` (47), `entry_btc_rsi_closed` (37), `entry_vol_mcap_ratio` (28), `entry_pair_age_days` (121), `entry_chop_burst_prior_fill_s` (9), cooldown minutes since a stop (8). Any 'effect' here is confounded with the recent-era loss.
3. **Prior-study features that stop before the newest fills:** Oct-15 impulse legs / score / corr15 / spike / volume shape (`imp_*`, fills ≤ 10-15; 0 of the B22–B25 fills) and Oct-4 pair-strength (`ps_*`, ≤ 10-04). Rebuilding them needs the impulse-grid fetch (`study_ml_impulse_fetch.py` + `_features.py`, the ML_MARKET_TURN scout step) — not run here (shared Binance budget).
4. **Not available anywhere:** open interest, funding history beyond the entry stamp, long/short ratio, liquidations, per-pair taker-flow / tick data at entry, order-book depth for bot fills, news / listing events, cross-exchange premium.
5. **Universe list lag:** the top-50 breadth for fills after 2026-10-11 uses the 10-11 list (cache end; up to 15 days stale).
6. **Not tested exhaustively:** 3-way interactions; non-monotonic mid-bands beyond the quintile tables; thresholds other than sign / median / terciles; regime-conditional exits (exit side is a separate study).
7. **Replay coverage:** only engine-stamped columns exist in the replay — `confidence`, burst / scan, pair-strength and cooldown features have no replay check (n/a in the tables). The replay refutes, never proves.
8. **Statistical power:** KEEP N = 158 over 71 days; the most recent era group (where the losses are) has 37 fills over 13 days. A real +0.2 % separator in a 30-fill zone is below the detectable size with this N (family-wise 95 % |Δ| ≈ 0.82 in 1D).

## 12 · FROZEN pre-registration for B25 (thresholds fixed 2026-10-26, never re-tuned)

Cohort for every line: MOMENTUM LONG, CLOSED, non-probe, non-MANUAL fills opened **≥ 2026-10-23 18:00 UTC** (B25 onward; the master fills that produced the hypotheses never count). ⚠ The 9 B25 fills shown in 'B25 so far' were SEEN by this study → tagged SEEN and reported apart; a line confirms only on fills opened after 2026-10-26 (the prep date). % at 1×. DAY units for market-wide lines. ML breakeven WR frozen at **61.5%** (KEEP, 2026-10-26). Filter verdict (expectancy bar): N ≥ 15 on ≥ 8 days ∧ WR < breakeven ∧ day-bootstrap P(avg<0) ≥ 0.95 ∧ no day / pair ≥ 50 % of the zone's net loss → FILTER CANDIDATE (then 30–50 % haircut + tight revert gate at promotion); RETIRE at N ≥ 15 if zone avg ≥ rest avg. Winner verdict (Pattern-W): N ≥ 30 ∧ WR ≥ 70 % ∧ avg ≥ +0.10 ∧ $ > 0 ∧ zone − rest > 0 at day-bootstrap P ≥ 0.90 → SIZING CANDIDATE (operator call, cell-registry read first); RETIRE at N ≥ 15 if zone avg ≤ rest avg. Verdict frozen at the first crossing; one re-read at N = 30.

| # | line | zone definition (frozen) | side | data needed | B25 so far |
|---|---|---|---|---|---|
| P1 | ML_SBULL_W | `entry_btc_regime == STRONG_BULL` | winner | stamp | scored 9/9 · zone 5·60%·+0.000 · rest 4·100%·+0.350 |
| P1b | ML_BTCADX_W (alias, read together) | `entry_btc_adx > 27.99` | winner | stamp | scored 9/9 · zone 5·60%·+0.000 · rest 4·100%·+0.350 |
| P2 | ML_ADXFLAT_NOTSBULL | `entry_adx_delta ≤ 0.309` ∧ `entry_btc_regime ≠ STRONG_BULL` | loser | stamp | scored 9/9 · zone 2·100%·+0.458 · rest 7·71%·+0.069 |
| P3 | ML_GAP58_LOW (refute-tracking) | `entry_ema_gap_5_8 ≤ 0.082` | loser | stamp | scored 9/9 · zone 7·71%·+0.069 · rest 2·100%·+0.458 |
| P4 | ML_BTC15_DOWN | BTC 1m close at entry ≤ close 15 min earlier (`btc_r15 ≤ 0`, kline rebuild) | loser | BTC 1m klines | scored 9/9 · zone 4·75%·+0.323 · rest 5·80%·+0.022 |
| P5 | ML_QUIET_TAPE_W | top-50 15-min mean return ≤ +0.1706 % ∧ pair 15-min quote vol / 24h mean ≤ 0.712 | winner | 5m klines (top-50 + pair) | scored 9/9 · zone 4·75%·+0.176 · rest 5·80%·+0.139 |
| P6 | ML_CORR15_HIGH | top-10 15-min return correlation `corr15 > 0.365` (Oct-15 grid definition) | loser | impulse grid rebuild | scored 0/9 · zone 0 · rest 0 |
| P7 | ML_LOWSPIKE_LOWCORR_W | BTC `spike_30 ≤ 0.1869` ∧ `corr15 ≤ 0.3200` | winner | impulse grid rebuild | scored 0/9 · zone 0 · rest 0 |

P3 is registered only to close it: it is already refuted on the 115 older master fills (Δ 0.000) — retire at N ≥ 15 unless B25+ show WR < breakeven at P ≥ 0.95. P2 is the third read of ADX direction (DL 273 ② uses ADXΔ ≤ 0 on a closed-bar rebuild); it does not replace that line — both are read, and evidence counts once per fill.

### 12b · Existing frozen ML lines — B25 tally from stamps (definitions unchanged; full reads by their scouts)

| line | B25 zone (N·WR·avg) | fills |
|---|---|---|
| DL 125 BTC-drop + pair-rollover | 0 | — |
| DL 160 burst ≤ 2 min after another bot fill | 2·100%·+0.412 | LTCUSDT +0.71, LINKUSDT +0.11 |
| DL 161 BTC-chop eff72 ≤ 0.007 (live stamp) | 0 | — |
| DL 258 PAIR_1H_DOWNTREND (pair 1h EMA20 ≤ EMA200) | 8·75%·+0.072 | ENAUSDT -0.89, LTCUSDT +0.71, STRKUSDT +0.09, UNIUSDT +0.40, ADAUSDT -0.31, BRUSDT +0.09, UNIUSDT +0.37, LINKUSDT +0.11 |
| DL 273 ① BTC24H_DOWN (BTC 5m close vs 288 bars earlier ≤ 0) | 8·88%·+0.213 | ENAUSDT -0.89, LTCUSDT +0.71, STRKUSDT +0.09, UNIUSDT +0.40, BRUSDT +0.09, LIGHTUSDT +0.82, UNIUSDT +0.37, LINKUSDT +0.11 |
| DL 273 ② ADX Δ ≤ 0 (live forming-bar stamp — the line uses the closed-bar rebuild) | 6·83%·+0.204 | ENAUSDT -0.89, LTCUSDT +0.71, STRKUSDT +0.09, UNIUSDT +0.40, BRUSDT +0.09, LIGHTUSDT +0.82 |
| DL 126 LOW-ADX RSI-momentum (should be 0 — live gate) | 0 | — |
| DL 110 mega-cap rank ≤ 10 (should be 0 — live gate) | 0 | — |

Read by their own scouts at the review (not recomputed here): ML_MARKET_TURN (DL 299, `scout_ml_turn.py`, needs the impulse grid extended), TREND_ALIGNED (DL 258, `scout_ml_trend_lines.py`), NEG_DAILYUP_WEAKPAIR (DL 253), ML_STOP_COOLDOWN (DL 248), C∪D (DL 127), REBOUND watch, the LONG_CHOP_BURST / mega-cap / heat revert tallies.

## 13 · Files

- Scripts: `scripts/study_ml_wl_b25prep_data.py` (loader) · `_fetch.py` (Binance top-up) · `_features.py` (impulse rebuild) · `_analyse.py` (screen) · `_report.py` (this file)
- Outputs: `reports/PREP_B25_ML_WL.md` · `_engine.txt` · `_coverage.csv` · `_screen1d.csv` · `_screen2d.csv` · `_candidates.csv` · `_quintiles.csv` · `_dashboard_vars.csv` · `_post_entry.csv` · `_replay_baserate.csv` · `_features_new.csv` · kline top-up `reports/PREP_B25_ML_WL_fetch/`