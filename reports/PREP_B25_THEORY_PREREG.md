# PREP B25 — first-principles theory + FROZEN winner/loser predictions (new sleeves)

**Frozen:** 2026-10-26, before any B25 read. **Blindness:** no `~/Downloads/scalpars_orders_paper_2026-10-2*` export was opened. Inputs were CLAUDE.md, CLAUDE_CURRENT_STATE.md (grep only), NEXT_BATCH_REVIEW_PLAN.md, the prior theory reports (LITE_THEORY, WILLY_THEORY, B19_REVIEW_FRENZY_THEORY, FRENZY_SLEEVE_DESIGN, FRENZY_WL_SCREEN theory, the BEARRUN_* preregs, FLATBASE WINLOSS, WILLY_TURNOVER_REPRICE, FADE_WL_SCREEN), DECISION_LOG (grep by sleeve), the engine (`services/frenzy.py`, `services/flatbase.py`, `services/surge.py`, `services/trading_engine.py`) and `reports/MASTER_POOL_stacked.csv` (eras B1–B24 only; B25 is not in it).
**Machine-readable:** `reports/PREP_B25_PREREG.csv` has 1,218 rows (1,154 + 64 added at review 10-26: Conf + FanR per sleeve as no-prediction rows, and the 6 post-entry / exit dashboard columns as DESCRIPTIVE-only rows): 8 sleeves × every dashboard W/L column, every `entry_*` column on `Order` (models.py, 141 columns) and the path variables. 233 non-NONE (230 directional) rows carry a directional prediction. The rest are explicit `NONE` rows, meaning no mechanism, screen-only. Columns: `sleeve, variable, column_name, direction, threshold, rationale, evidence_source, top_rank, master_scored, refute_if`.

## 0. Rules for reading this at the B25 review

- **Direction:** `W_HIGHER` / `W_LOWER` means winners carry the higher / lower value, and the other side of the threshold is the predicted loser zone. `W_BAND` = winners inside the band. `W_TRUE` / `W_FALSE` = the flag value winners carry. `NONE` = no prediction.
- **Thresholds are frozen** and are never re-fit on B25. Where an earlier frozen line exists (DL 374, DL 288, DL 301, DL 287, DL 275, B19 §3 L/T lists, WL theory list) **its direction and cut are adopted unchanged**, even when the master disagrees. The disagreement is flagged in the CSV as `MASTER_IS CONFLICTS`.
- **Evidence labels (what is in-sample):**
  - `THEORY` = first principles only.
  - `PRIOR_FROZEN` = an earlier registered line.
  - `PRIOR` = an earlier study result.
  - `BACKTEST` = yr5 / yr6 / tick re-walks. Out-of-master, but often the data the rule was designed on.
  - `MASTER_IS` = the master pool ≤ B24. The cut is the master median, so it is in-sample. Thresholds labelled this way are the weakest.
  - **`CONTAINS_B25`** = DL 374 fade watch lines ① ② ③ ④ ⑤ ⑥. Their registration pool was 76 master + **7 B25 fades** (DL 374). Those 7 B25 fills must be excluded when the lines are read on B25, or B25 grades its own lines.
- **Read protocol (per the 10-26 method):**
  - On B25, a prediction can only be **refuted** (refute-only), or kept on the watchlist.
  - **Refuted** = the predicted zone's average P&L % (1×, de-multiplied, on today's stack) has the opposite sign to the rest's, with N ≥ 5 per side. That is a refute-only flag.
  - A formal call needs N ≥ 15 per side on ≥ 8 days and a day-bootstrap P ≥ 0.90.
  - **Market-wide variables** (BTC*, Breadth, GlobalVol, 1hSlope, regime) are counted in **DAY / window units**, not fills (window-units rule).
  - **Confirmation** only makes a line a watchlist item. Filter promotion still needs the expectancy bar: WR < the sleeve breakeven, avg < 0 at 95 % window-bootstrap, ≥ 8 windows, N ≥ 15, haircut.
- **Multiple-testing warning:** 233 non-NONE rows (230 directional) will produce about 10–12 "confirmations" by luck at p 0.05. Only the ranked top-5 per sleeve (40 rows) are the primary predictions. Score them first. The rest are secondary.

## 1. Column semantics traps (verified in code)

| Dashboard column | Order column(s) | Trap |
|---|---|---|
| Gap / Gap5-8 / Gap8-13 / Stretch | `entry_gap`, `entry_ema_gap_5_8`, `entry_ema_gap_8_13`, `entry_ema5_stretch` | **Unsigned** (`abs()`) on spike-scanner fills (trading_engine.py ~12088). Direction cannot be read from them. |
| ADXDir / RSIDir / BTCADXDir / BTCRSIDir | `entry_adx` vs `entry_adx_prev`, `entry_rsi` vs `entry_rsi_prev`, BTC pairs | On fades `entry_rsi_prev` = `rsi_prev2` (the pre-spike base). RSIDir/ADXDir are degenerate on fades (always up), on WILLY/LITE red bars (mostly down) and on FRENZY (forming bar). |
| ADX, +DI/−DI, ADXΔ on FRENZY-family fills | `entry_adx`, `entry_pos_di`, `entry_neg_di`, `entry_adx_delta` | Read the **forming** bar. Use `entry_frenzy_adx_delta` / `entry_frenzy_di_spread` (closed bars) for LITE/LONG. They are not stamped on WILLY. |
| PairTrend | `entry_pair_ema20_ema50_gap_pct` | Holds **EMA13−EMA50** (misnamed). Signed. |
| BTCSlp | `entry_btc_ema20_slope` | The dashboard shows `abs()`; the prediction uses the raw signed value. |
| Breadth | `entry_bull_pct` (LONG) / `entry_bear_pct` (SHORT) | Direction-specific. |
| Ext% / BTCExt% | `entry_dist_from_ema13_pct` / `entry_btc_dist_from_ema13_pct` | The dashboard **flips the sign for shorts**. The CSV uses the raw column. |
| 1hSlope | `entry_btc_1h_slope` | Raw signed. |

## 2. Coverage — scored fills in the master (≤ B24) per sleeve

FLATBASE fills stamp **no pair or BTC 5m technicals** (RSI/ADX/gaps/BTC RSI/1hSlope = 0/1). Every dashboard technical column is **BLIND** for FLATBASE and must be rebuilt from klines. WILLY_HT has **0 fills in the master** (deployed 10-21; B24 had none), so every HT prediction is THEORY + BACKTEST. SURGE (3), Bear-Run (2) and FLATBASE (1) are effectively blind in the master: their predictions lean on backtests and theory.

| Variable | column | SPIKE_FADE (N=126) | WILLY (N=38) | LITE (N=36) | FLONG (N=32) | FLATBASE (N=1) | SURGE (N=3) | BEARRUN (N=2) | WILLY_HT (N=0) |
|---|---|---|---|---|---|---|---|---|---|
| RSI | `entry_rsi` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| ADX / ADXDir | `entry_adx` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| RSIDir | `entry_rsi_prev` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| Gap | `entry_gap` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| Gap5-8 | `entry_ema_gap_5_8` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| Gap8-13 | `entry_ema_gap_8_13` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| Stretch | `entry_ema5_stretch` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| EMA20Slp | `entry_ema20_slope` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| BTCSlp | `entry_btc_ema20_slope` | 3 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| BTCRSI / BTCRSIDir | `entry_btc_rsi` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| BTCADX / BTCADXDir | `entry_btc_adx` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| ADXΔ | `entry_adx_delta` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| RngPos | `entry_range_position` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| Breadth | `entry_bull_pct` | 126 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| ATR% | `entry_atr_pct` | 126 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| BTCTrend | `entry_btc_trend_gap_pct` | 126 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| PairTrend | `entry_pair_ema20_ema50_gap_pct` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| GlobalVol | `entry_global_volume_ratio` | 3 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| PairVol$ | `entry_pair_volume_24h_usd` | 126 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| Ext% | `entry_dist_from_ema13_pct` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| BTCExt% | `entry_btc_dist_from_ema13_pct` | 126 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| 1hSlope | `entry_btc_1h_slope` | 3 | 38 | 36 | 32 | 0 | 3 | 2 | 0 |
| — mcap | `entry_mcap_usd` | 1 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| — turnover R | `entry_vol_mcap_ratio` | 1 | 38 | 32 | 23 | 1 | 3 | 0 | 0 |
| — BTC 1d ret | `entry_btc_1d_ret_pct` | 20 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| — BTC off24h/lo, eff72, r72, off30d | `entry_btc_off24lo_pct` | 36 | 38 | 36 | 32 | 1 | 3 | 2 | 0 |
| — frenzy core (hours, vwap, vol_mult, run, bar_ret) | `entry_frenzy_hours` | 0 | 38 | 36 | 32 | 1 | 0 | 0 | 0 |
| — frenzy above_share | `entry_frenzy_above_share` | 0 | 38 | 36 | 28 | 0 | 0 | 0 | 0 |
| — frenzy di_spread / adx_delta / vol_trend | `entry_frenzy_di_spread` | 0 | 0 | 36 | 32 | 0 | 0 | 0 | 0 |
| — frenzy gvol | `entry_frenzy_gvol` | 0 | 16 | 21 | 14 | 0 | 0 | 0 | 0 |
| — frenzy stop_atr | `entry_frenzy_stop_atr` | 0 | 24 | 36 | 32 | 0 | 0 | 0 | 0 |
| — willy trigger / wait_bars | `entry_frenzy_willy_trigger` | 0 | 38 | 0 | 0 | 0 | 0 | 0 | 0 |
| — willy spike high H | `entry_frenzy_willy_spike_high` | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| — willy nomiss | `entry_frenzy_willy_nomiss` | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| — fade laggard (1d −DI, BTC 4h gap) | `entry_pair_1d_ndi` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| — pair rank / age | `entry_pair_rank` | 126 | 0 | 0 | 0 | 0 | 3 | 2 | 0 |
| — flatbase flat14 / cov / off_peak | `entry_flatbase_flat14_pct` | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| — funding | `entry_funding_rate` | 3 | 0 | 0 | 0 | 1 | 0 | 2 | 0 |
| — surge btc/pair move, gvol, trigger | `entry_surge_btc_move_pct` | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| — bear r24/below24/eff24/off24lo/bypass | `entry_bear_r24` | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |
| — book spread/drift, order lag | `entry_book_spread_pct` | 0 | 14 | 13 | 6 | 1 | 3 | 0 | 0 |

**Blind everywhere (0 scored on all 7 sleeves):** `entry_btc_regime_started_at`, `entry_chop_burst_prior_fill_s`, `entry_br_*` (the 9 Bull-Run door stamps). `entry_willy_ht_size_frac` is not in the master at all.

**Partial coverage to remember:**
- **SPIKE_FADE:**
  - `entry_global_volume_ratio`, `entry_btc_1h_slope` and `entry_btc_ema20_slope`: 3/126.
  - `entry_mcap_usd`: 1/126 (backfilled only since 10-19 05:08 UTC).
  - The laggard stamps, `entry_btc_1d_ret_pct` and `entry_pair_1h_ema20_200_gap_pct`: 20/126.
  - BTC off24 / eff72 / r72: about 33/126.
- **Frenzy family:**
  - `entry_frenzy_gvol`: 14–21 of 32–38.
  - `entry_frenzy_stop_atr` on WILLY: 24/38.
  - V2b fields (`entry_frenzy_willy_spike_high`, `frenzy_willy_tp_target_pct`): 1/38. The master WILLY was re-priced at the live V2b exit in STACK 2026-10-26a (DECISION_LOG 377): H from the stamp on 1, rebuilt from 5m / ticks on 36, and 1 target-invariant (ICNT), so V2b pricing is complete even though only 1/38 carries the stamp.
- **Path variables not stamped on any Order** (derive from klines / journal at review):
  - fade: spike-candle return, seconds from trigger to fill;
  - LITE: no-new-high, drop below the 30-min high, k-th entry in the episode;
  - WILLY: last-minute taker share;
  - Bear-Run: coin-low age, time since monitor ON;
  - SURGE: pick order.

## 3. Direction grid — the 25 dashboard columns (↑ = winners higher, ↓ = winners lower, · = no prediction; threshold after the arrow)

Notes on the grid:
- LITE / LONG ADXΔ uses `entry_frenzy_adx_delta`.
- SPIKE_FADE BTCADX uses `entry_btc_adx_prev` (the DL 374 ⑥ field).
- SURGE PairTrend appears in the CSV as "Pair EMA13−50 gap" ↑ 1.0.
- FLATBASE technicals are theory-only and BLIND.

| Column | SPIKE_FADE | FRENZY_WILLY | FRENZY_WILLY_HT | FRENZY_LITE | FRENZY_LONG | FLATBASE_SHORT | SURGE_LONG | BEARRUN_SHORT |
|---|---|---|---|---|---|---|---|---|
| RSI | ↑ 74.4 | ↓ 74.4 | ↓ 74.4 | band 50–65 | ↓ 59.0 | ↓ 45 | ↑ 60 | ↑ 35 |
| ADX | ↑ 19.2 | · | · | ↓ 29.45 | ↓ 30.2 | ↑ 25 | ↑ 25 | ↑ 25 |
| ADXDir | · | · | · | · | · | · | T rising | T rising |
| RSIDir | · | · | · | · | · | · | T rising | F falling |
| Gap | ↓ 0.284 | ↑ 2.87 | ↑ 2.87 | ↑ 0.87 | ↓ 1.52 | · | ↑ 0.5 | · |
| Gap5-8 | ↓ 0.176 | ↑ 0.764 | ↑ 0.764 | ↑ 0.130 | · | · | ↑ 0.15 | · |
| Gap8-13 | ↓ 0.093 | ↑ 1.076 | ↑ 1.076 | ↑ 0.30 | ↓ 0.564 | · | ↑ 0.15 | · |
| Stretch | ↑ 1.23 | · | · | · | · | · | ↓ 1.5 | ↓ 0.8 |
| EMA20Slp | ↓ 0.112 | ↑ 1.357 | ↑ 1.357 | ↑ 0.327 | ↓ 0.52 | · | ↑ 0.2 | ↓ 0 |
| BTCSlp | ↓ 0 | ↑ 0 | ↑ 0 | ↑ 0 | ↑ 0 | · | ↑ 0 | ↓ 0 |
| BTCRSI | ↓ 40 | ↑ 50 | ↑ 50 | ↑ 51.8 | ↑ 50 | ↓ 50 | · | ↑ 30 |
| BTCADX | ↓ 24.5874 | · | · | · | · | · | ↑ 20 | ↑ 25 |
| BTCADXDir | · | · | · | · | · | · | T rising | T rising |
| BTCRSIDir | F | · | · | · | · | · | · | F falling |
| ADXΔ | ↑ 2.359 | ↑ 1.63 | ↑ 1.63 | ↓ -1.08 | ↑ 0.77 | · | ↑ 0 | ↑ 0 |
| RngPos | · | ↓ 79.6 | ↓ 79.6 | ↑ 67.85 | ↓ 64.75 | · | · | ↓ 30 |
| Breadth | ↑ 59.1 | ↑ 34.4 | ↑ 34.4 | ↑ 55.35 | · | ↑ 40 | ↑ 50 | ↑ 50 |
| ATR% | · | ↑ 1.62 | ↑ 1.62 | ↑ 1.75 | ↓ 2.5 | · | ↑ 2.0 | · |
| BTCTrend | ↓ -0.13 | ↑ 0 | ↑ 0 | ↑ 0 | ↑ 0 | ↓ 0 | ↑ 0 | ↓ 0 |
| PairTrend | ↑ -0.40 | · | · | ↑ 2.23 | ↓ 2.97 | ↓ 0 | — | ↓ 0 |
| GlobalVol | ↑ 1.0 | ↓ 1.0 | ↓ 1.0 | ↓ 1.0 | ↓ 1.0 | · | ↑ 1.0 | ↑ 1.0 |
| PairVol$ | ↓ 3760000 | ↑ 31000000 | ↑ 31000000 | ↑ 76500000 | ↑ 100000000 | · | · | · |
| Ext% | ↑ 1.53 | ↑ 2.53 | ↑ 2.53 | ↑ 0.33 | ↓ 0.53 | · | ↓ 2.0 | ↑ -1.0 |
| BTCExt% | ↓ -0.054 | ↑ 0 | ↑ 0 | · | · | · | · | ↑ -0.3 |
| 1hSlope | ↓ 0 | ↑ 0 | ↑ 0 | ↑ 0 | ↑ 0 | ↓ 0 | ↑ 0 | ↓ 0 |

---

## 4. SPIKE_FADE (short; stake-order #1)

**Mechanism.** In a non-bull tape, a thin alt (≤ $20M 24 h) jumps +25 RSI points on a ≥ 5× volume candle. The jump is mostly one aggressive taker walking a thin book, or a short-squeeze wick, and nobody owns the new price. The edge is a fast mean reversion: a median hold of about 1.5 min and +0.25…+0.57 trail exits.

**What breaks it:**
1. **A real move rather than a wick.** This shows as a breakout from a dead, flat base (low ADX, low pre-spike RSI) or a run already under way before the spike bar (wide slow-EMA gaps).
2. **A short squeeze or laggard catch-up.** The pair is in a deep downtrend with crowded shorts (EMA13−50 ≪ 0, high −DI, daily −DI with BTC 4h up).
3. **BTC beta tailwind.** BTC RSI or price is turning up at entry.
4. **Execution.** Large caps / deep books flip the "who arrives first" coin. Slippage on a ~0.5 % edge. At 2× size each −1.5 stop costs about 10 % of the paper account.

**Breakeven note.** DL 374 uses about 72 % WR, DL 334 about 79 %. **Freeze 72 %** for B25 reads. That is today's exit (avg win +0.546 / loss −1.394), computed on the DL 374 pool. State both if they disagree.

**Ranked predictions (primary):**

| # | Prediction | Frozen rule | Evidence (in-sample?) | Refuted if (B25, refute-only) |
|---|---|---|---|---|
| 1 | **Squeeze class loses** | PairTrend `entry_pair_ema20_ema50_gap_pct` ≤ −0.40 → loser zone (W_HIGHER) | THEORY + FRESHBREAK / LAGGARD mechanisms. Cut = the FRESHBREAK gate value. Master: high half +0.22 vs low half −0.05 (consistent, MASTER_IS). | Fades with pgap ≤ −0.40 average ≥ the rest with N ≥ 5 per side |
| 2 | **Large caps lose** | `entry_mcap_usd` ≥ $200M → loser zone | PRIOR_FROZEN DL 374 ③ / DL 311. **CONTAINS_B25** (1000LUNC). Coverage only since 10-19. | Exclude 1000LUNC. If the remaining ≥ $200M fades average > 0 → refuted. Gate stays first-10 avg < 0. |
| 3 | **Grind loses, vertical spike wins** | `entry_adx_delta` < 2.359 → loser zone | PRIOR_FROZEN DL 374 ① (p 0.10; CONTAINS_B25). Theory: a one-bar ADX jump = isolated vertical spike. | Excluding DL 374's 7 B25 fills, weak-side avg ≥ strong-side avg |
| 4 | **Flat-base breakout loses** | `entry_adx` ≤ 19.2 → loser zone | PRIOR_FROZEN DL 374 ② (p 0.35; CONTAINS_B25). Same mechanism as FRESHBREAK. | Same rule as #3 |
| 5 | **BTC beta tailwind loses** | BTC RSI rising at entry (`entry_btc_rsi > entry_btc_rsi_prev`) → loser zone | THEORY (loser class 1). MASTER_IS consistent (+0.04 rising vs +0.11 falling). Market-wide → count DAYS. | Rising-BTC-RSI days average ≥ falling days (≥ 5 days per side) |

**Secondary predictions (all in CSV):**

| Direction | Variables |
|---|---|
| Winners higher | RSI (74.4), Stretch / Ext% (spike overshoot), Bear% breadth, GlobalVol, pair rank (thin), +DI, EMA50 slope |
| Winners lower | slow-EMA gaps (Gap / 5-8 / 8-13 / EMA20Slp: a pre-built run = trend), BTC RSI (40), BTCExt%, BTC 1h slope, BTCTrend (DL 66 HB line ≥ −0.13), PairVol$ (3.76M), spike-bar volume multiple, −DI, laggard stamps, slippage |
| Context lines | HEALTHY_BEAR (DL 374 ④ / ⑤, CONTAINS_B25; the backtest contradicts ④) |

**No prediction:**
- ATR% (ceiling refuted).
- RngPos, RSIDir and ADXDir (degenerate).
- BTCADXDir.
- RSI-jump size (predicts stop rate, not P&L).

**Known conflicts:**
- The BTC 4h EMA50/200 gap (LAGGARD leg) went the wrong way in the master (20 fills).
- FADE_WL_SCREEN's "7-day volume < $4.89M" line has an unclear sign versus the PairVol$ ↓ prediction. Read both.

---

## 5. FRENZY_WILLY (long, first red 5m bar after a fresh FRENZY flag; 10×; V2b exit)

**Mechanism (two layers):**
1. **Bracket layer.** A +1.25 / −4 / 90-min bracket on a very volatile, freshly pumped coin wins about 76 % on a zero-drift path. Breakeven is about 77.4 %, so WILLY lives on a small positive drift.
2. **Drift layer.** The first red bar is profit-taking inside fresh attention. Continuation buyers lift price about 1 %. On engine-matched fills, WILLY beats a random bar of the same window by +0.19 (P 0.99).
- **V2b** adds a runner leg: the target runs to the spike high H × 0.99, capped at 8 %.

**What breaks it.** The −4 tail. Losers are **coin-specific crashes** (coin −2.8 % in 15 min while BTC is flat; 45 % of stops come within 10 min). They come from:
- crowded leverage (turnover R),
- a falling BTC 1h with no bid under the coin,
- large caps with no attention premium,
- weak 20–25× spikes,
- entry slippage on a ~0.2 % drift edge,
- slow (low-ATR) coins that cannot reach +1.25 in 90 min.

**Ranked predictions:**

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **BTC 1h falling loses** | `entry_btc_1h_slope` ≤ 0 → loser zone | PRIOR_FROZEN DL 275. yr5 +0.185 vs −0.098; real 23 · +0.71 vs 13 · −0.52. **Reverses in 2026-H1.** Day units. | Slope ≤ 0 days average ≥ slope > 0 days (≥ 5 days per side) |
| 2 | **Distant spike high pays under V2b** | `frenzy_willy_tp_target_pct` ≥ 3 → higher avg P&L % (and lower WR) than < 3 | PRIOR DL 356 / 358 (backtest 2026 +0.10 [−0.03, +0.24]); THEORY. Master: 1/38 scored. | Target ≥ 3 fills average below target < 3 fills (N ≥ 5 each) |
| 3 | **Large caps lose** | `entry_mcap_usd` ≥ $100M → loser zone | PRIOR WILLY_THEORY review, engine-matched (< $100M +0.12 vs −0.08). MASTER_IS **conflicts** (large caps did better on 38). | ≥ $100M fills average ≥ the rest |
| 4 | **Below the spike VWAP loses** | `entry_frenzy_vs_vwap_pct` ≤ 0 → loser zone | PRIOR W5_BELOWVWAP near-miss (1,110 · −0.100, P 0.94). MASTER_IS consistent. | Below-VWAP fills average ≥ the rest |
| 5 | **Slow coins lose** | `entry_atr_pct` < 1.62 → loser zone (mirror: `entry_frenzy_stop_atr` > 2.34) | THEORY (the TP is far in ATR units). MASTER_IS consistent (+0.98 vs +0.45; stop_atr −0.61 vs +1.93). | Low-ATR half averages ≥ the high half |

**Secondary predictions:**

| Direction | Variables |
|---|---|
| Winners higher | fast-EMA gaps and EMA20Slp (continuation buyers), ADXΔ (1.63), above_share (50), spike vol_mult (≤ 25× loser), PairVol$, BTC RSI / BTCTrend / BTCExt / BTC 5m slope (tape) |
| Winners lower | RngPos (deeper dip), turnover R inside < 1 (weak; W7 0.5–1 refuted on engine fills), episode age > 1 h (weak), wait_bars ≥ 4 (weak), slippage, GlobalVol / frenzy gvol (< 1 better) |
| Losers | R_NOMISS route (weak) |
| Context line | DL 305 MKTDIP zone (observe) |

**No prediction:** red-bar depth (W1 refuted), run size (W3_BIG refuted), BTC direction pairs, ADX level.

---

## 6. FRENZY_WILLY_HT (same trigger; routed only when turnover R ≥ 1; ¼ size; WILLY's exit)

**Mechanism.** HT is not a different setup. It is WILLY's setup on the **one sub-population the evidence says is negative**: high turnover, meaning futures volume exceeds the market cap.

The leverage-overhang theory:
- When 24 h volume exceeds the coin's cap, the bid is crowded late leveraged longs.
- The first red bar is not a dip that gets bought. It is the first liquidation.
- The −4 stop is hit more often (backtest stop rate 24.5 % vs 17.4 %), and WR (72.5 %) falls below the bracket's ~77 % breakeven.

Backtest cohort (R ≥ 1, ticks, live exit, **in-sample to the R ≥ 1 cut**): 477 · 72.5 % · **−0.237** [−0.42, −0.06], worse in 8 of 10 months. The live shadow was mixed: blocked +0.13 on 15, and the v2 deep read Δ −0.06 on 21. **0 HT fills in the master.**

**Ranked predictions:**

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **HT under-performs WILLY on the same days** | HT avg P&L % ≤ same-period WILLY − 0.20 (= DL 341 CONFIRMED) | PRIOR_FROZEN DL 341 gate + BACKTEST. 1× / % basis; the size difference is ignored. | HT avg ≥ WILLY avg on the same-days line (the DL 341 REMOVE_FILTER side) |
| 2 | **Bigger caps lose inside HT** | `entry_mcap_usd` ≥ $50M → loser zone | BACKTEST blocked cohort: < $50M −0.19 vs ≥ $50M ≈ −0.43 (derived from the §4 grid), ≥ $100M ≈ −0.81 | ≥ $50M HT fills average ≥ the < $50M fills |
| 3 | **BTC 1h falling loses (inherited)** | `entry_btc_1h_slope` ≤ 0 → loser zone | PRIOR DL 275 (WILLY), assumed to transfer | As WILLY #1 |
| 4 | **HT stops more** | HT −4 stop rate ≥ 1.4 × WILLY's | BACKTEST 24.5 % vs 17.4 % | HT stop rate ≤ WILLY's |
| 5 | **Non-monotonic in R** | Inside HT, R ≥ 2 beats R 1–2 (W_HIGHER at 2.0) | BACKTEST grid (derived): R 1–1.5 ≈ −0.20, **1.5–2 ≈ −0.51**, 2–3 ≈ −0.21, ≥ 3 −0.06. **Flagged as a confound** (non-monotonic single variable) — never a filter. | R 1–2 average ≥ the R ≥ 2 average |

**The operator's three HT questions — frozen theory answers (to be checked against data, not replaced by it):**
1. **Should HT exist?** Theory says the R ≥ 1 cohort is negative-expectancy under WILLY's exit. HT is justified only as a **measuring probe at ¼ size** until its DL 341 gate reads (30 fills on ≥ 10 days). The prediction is CONFIRMED (HT ≤ WILLY − 0.20), which means "the turnover block was right; HT should not be upsized". Any B25 answer is below the gate and labelled operator-discretion.
2. **Should HT have its own exit?**
   - Prediction: yes in direction, but no exit rescues it.
   - Because hazard rises with time in a crowded-leverage coin, HT should prefer the **earliest late TP** (LATE30 ≥ LATE45 / LATE60; NOLATE worst).
   - HT should gain **less** from V2b's high target than WILLY (V2b − +1.25-shadow Δ: HT < WILLY), because crowded longs sell into the first pop, so runners to H are rarer.
   - A **tighter stop is predicted to be worse.** −3 / −2.5 were worse than −4 for WILLY (DL 303). HT's flushes are deeper and come as 1-minute wicks.
   - Read on DL 373's per-sleeve shadow split, on the "same days only" line.
3. **Can filtering HT winners vs losers give it WILLY's exit?**
   - Predicted **no**. The kept HT fills after the best plausible filter (mcap < $50M ∧ BTC 1h slope > 0) are predicted to still average below same-period WILLY (Δ < 0). The turnover penalty lives in the same −4 tail that the entry stamps cannot see (losers are post-entry coin crashes; market-only AUC 0.45–0.51).
   - HT N in B25 will be far below the expectancy bar (N ≥ 15, ≥ 8 windows), so any HT filter is **observe-only** by rule.

---

## 7. FRENZY_LITE (long; held ≥ 12 closes above the spike VWAP, vol < 100×, 2–16.75 h, red bar; 8×; TP +3 / +2 by LITE C, −3)

**Mechanism.** Absorption. The spike cohort is in profit, and price holds an hour above its cost basis (the VWAP) on cooling volume. That builds a base for a second leg while attention lasts.

**What breaks it:**
- **Attention decay:** late in the episode.
- **Weak hold:** a low share of closes above VWAP, or no new high since the spike bar.
- **Bearish tape:** BTC 1h falling (the bearish-day block already covers the worst days).
- **Mature trend mis-read as a base:** high ADX.
- **A tired episode:** the 2nd / 3rd entry.

LITE stops do not recover (11/11 go deeper). Its P&L follows the regime (2025 negative, 2026 positive), and 0 of 53 earlier theory cells passed.

**Ranked predictions:**

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **Late = loser** | `entry_frenzy_hours` ≥ 12 → loser zone | PRIOR A4 (12–16.75 h negative both years: −0.44 / −0.21) + DL 353. MASTER_IS consistent (> 7.96 h −0.70 vs +1.37). | ≥ 12 h fills average ≥ the < 12 h fills |
| 2 | **Mature trend = loser** | `entry_adx` > 29.45 → loser zone | PRIOR_FROZEN WL L2 (backtest 3/3 regimes, P 0.986; in-sample cut). Caveat: LONG blocks most of these, so few will fill. | ADX > 29.45 fills average ≥ the rest |
| 3 | **BTC 1h falling = loser** | `entry_btc_1h_slope` ≤ 0 → loser zone | PRIOR_FROZEN DL 288 LITE_BTC1H_FALLING. **H1 reverses; MASTER_IS conflicts.** Day units. | Falling days average ≥ rising days |
| 4 | **Weak hold = loser** | `entry_frenzy_above_share` ≤ 58.97 → loser zone | PRIOR WL T1 (frozen cut) + B19 T11. MASTER_IS consistent (+1.12 vs −0.45). | Weak-hold fills average ≥ the rest |
| 5 | **First entry of the episode wins** | k-th FRENZY-family entry ≥ 2 (same `entry_frenzy_spike_at`) → loser zone | PRIOR B19 T12 + data-mined (+0.46 vs +0.07; reverses in H1) | Re-entries average ≥ first entries |

**Secondary predictions:**

| Direction | Variables |
|---|---|
| Winners higher | fast-EMA gaps / EMA20Slp / PairTrend (2.23) / RngPos / Ext% (strong hold), vs_vwap (2.23; conflicts with B19 T2), vol_mult ≥ 20 (LITE C leg; conflicts with B19 T1), DI spread, PairVol$ (thin = loser), ATR% (needs range), BTC RSI ≥ 51.8 (MASTER_IS conflicts) |
| Winners lower | frenzy ADXΔ (falling = real consolidation), vol_trend ≤ 0.914 (cooling volume; B19 T6), gvol (< 1) |
| In band | RSI 50–65 (non-monotonic, so not a filter) |
| Losers | RETRY fills, no-new-high (A9, BLIND), ≥ 3 % below the 30-min high (DL 259, BLIND) |

**No prediction:** bar_ret (A2 unstable), run size, above_streak.

---

## 8. FRENZY_LONG (long; fresh ON bar ≥ 100× vol, ≥ 12 closes ≥ VWAP, ATR ≤ 3, red bar; blocker size 1×; +1.5 / −4)

**Mechanism.** A loud second wave bought on a red pullback.

**Empirical record:**
- About −0.15 %/fill over two years in every regime.
- Its kept value is as **LITE's blocker**: in the backtest, LITE fills blocked by an open LONG were 29 % WR, −1.46.
- It loses when the second wave is distribution: a mature high-ADX trend, a down day for BTC, or price far above support.

**Frozen directions.** Adopted from **B19 §3 L1–L16 (2026-10-11)** wherever they exist. Several conflict with the master and are marked in the CSV.

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **Mature trend = loser** | `entry_adx` > 30.2 → loser zone | THEORY + LITE L2 analogue. MASTER_IS (losers' median ADX 35.6 vs winners' 24.8; −1.03 vs +0.70). | High-ADX fills average ≥ the rest |
| 2 | **BTC down day = loser** | `entry_btc_1d_ret_pct` ≤ 0 → loser zone | PRIOR B19 L15. MASTER_IS consistent (+0.86 vs −0.87). Day units. | Down days average ≥ up days |
| 3 | **Far above VWAP = loser** | `entry_frenzy_vs_vwap_pct` > 3.49 → loser zone | PRIOR B19 L3. MASTER_IS consistent. | — |
| 4 | **Deeper red pullback wins** | `entry_frenzy_bar_ret_pct` > −0.53 → loser zone | PRIOR B19 L1. MASTER_IS consistent. | — |
| 5 | **First entry wins** | k ≥ 2 → loser zone | PRIOR B19 data-mined (holds both halves) | — |

The refutation rule is the same for all five: the predicted loser zone averages ≥ the rest.

**Blocker read (not a W/L line):** LITE setups refused as PAIR_HELD by a LONG should price below LITE's average.

---

## 9. FLATBASE_SHORT (short at the first 4-h-boundary bar below the spike VWAP ≥ 4 h after a pump on a 14-day-flat coin; turnover > 1; 5×; TP −8 / SL +15 / 72 h)

**Mechanism.** A pump on a coin that sat dead and flat is speculative churn with no owner. Turnover above 1 means the volume is mostly the same coins changing hands. Once price is below the spike VWAP, the pump cohort is underwater, and the move round-trips toward the base within 72 h.

**What breaks it:**
- re-pumps / squeezes (GLM / SAHARA / ONT type);
- BTC rallying during the hold (look-ahead, not separable at entry);
- low turnover (a real owner; already routed to SHADOW);
- fat-tail dependence.

Market-state separators were already screened (best family-wise p 0.16), so every market prediction here is weak. **Master: 1 fill.** Live: 6 fills, 3 at −15.

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **More churn wins (dose above the routing cut)** | `entry_vol_mcap_ratio` < 2.0 → loser zone (among traded fills) | PRIOR WINLOSS (R > 1: 482 · 73 % · +1.65 vs all +0.93). The dose above 2 is THEORY. Live went the other way (7 · 71 %). | R < 2 fills average ≥ R ≥ 2 fills |
| 2 | **Flatter base wins** | `entry_flatbase_flat14_pct` > 0.25 → loser zone | PRIOR flat dose-response (label-shuffle p ≈ 0.01); cut is THEORY | — |
| 3 | **Deeper under VWAP wins** | `entry_frenzy_vs_vwap_pct` > −3 → loser zone | THEORY | — |
| 4 | **Still near the peak wins** | `entry_flatbase_off_peak_pct` > 20 → loser zone | THEORY | — |
| 5 | **Negative funding wins** | `entry_funding_rate` ≥ 0 → loser zone | PRIOR WINLOSS (inverse to the cost theory; in-sample backtest) | — |

The refutation rule is the same for all five: the zone averages ≥ the rest. With N tiny, report counts only.

**Blind on FLATBASE fills:** all pair and BTC 5m technicals (RSI / ADX / gaps / BTC RSI / 1hSlope / GlobalVol / Ext). Theory rows are kept in the CSV (RSI ≤ 45 wins, ADX ≥ 25 wins, PairTrend < 0 wins, BTC RSI / 1hSlope lower wins), and each needs a kline rebuild.

---

## 10. SURGE_LONG (BTC 30-min ≥ +0.3 % to a new 24 h high on ≥ 5× volume ∧ market vol ≥ 1 → top-20 high-ATR leaders with EMA13 > EMA50; full size 20×; Bull-Run trail without the lock)

**Mechanism.** A BTC breakout to a new 24 h high on market-wide volume pulls the high-beta leaders, the alts already outrunning BTC, into multi-percent runs. A wide 1×ATR trail with no lock lets the few runners pay. The edge is concentrated in a few triggers (top 3 ≈ 91 %).

**What breaks it:**
- chop or bear regimes;
- BTC moving on isolated volume;
- laggard or low-ATR picks;
- late-cycle BTC (already up for days);
- the tight initial stop on over-stretched picks.

**Master: 3 fills**, so everything here is backtest + theory.

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **Early-leg BTC wins** | `entry_btc_r72_pct` > 2.7 → loser zone | PRIOR_FROZEN DL 200 probe test (+0.51 on 22 triggers) | Count TRIGGERS, not fills |
| 2 | **Bigger BTC impulse wins** | `entry_surge_btc_move_pct` < 0.5 → loser zone (option-B-only triggers) | PRIOR DL 202 (A +0.113 vs B +0.008 per trigger) | Option-B-only triggers average ≥ option-A triggers |
| 3 | **Leader strength wins** | `entry_surge_pair_move_pct` < 2.0 → loser zone | PRIOR MISSED_OPPS (non-leaders −0.30); cut is THEORY | — |
| 4 | **Higher-frame uptrend wins** | `entry_btc_1h_slope` ≤ 0 → loser zone | THEORY | — |
| 5 | **First pick of a trigger wins** | 2nd–4th pick → loser zone | PRIOR MISSED_OPPS (2nd / 3rd movers −0.72) | — |

**Secondary predictions:**
- Winners higher: EMA13−50 gap above 1.0, ATR ≥ 2.0, surge gvol ≥ 1.5, momentum technicals (RSI ≥ 60, ADX ≥ 25 rising, gaps).
- Winners lower: Stretch / Ext% (over-stretched picks stop out first).
- Losers: bracket-capped fills.

---

## 11. BEARRUN_SHORT (momentum short signal while the BTC bear monitor is ON — r24 ≤ −4 %, ≥ 50 % bars below EMA20, eff ≥ 0.15 — BTC ≤ 2 % off its 24 h low, bear breadth ≥ 40; 5×; momentum-short exit incl. EMA13 cross)

**Mechanism.** A confirmed, efficient 24 h BTC decline is a regime in which most short signals pay ("the money was the regime", DL 60).

**What breaks it:**
- **Asymmetry of bear legs.** They are short flushes followed by bounces. The 24 h trigger enters with 80–94 % of the drop already done.
- A BTC bounce ≥ 0.5 % off the post-ON low.
- Shorting stale coin lows.
- Too few windows (11 in 22 months).

Every earlier separator screen at ON was null.

**Master: 2 fills.** Read in WINDOW units only.

| # | Prediction | Frozen rule | Evidence | Refuted if |
|---|---|---|---|---|
| 1 | **Shorting into the low wins; shorting a BTC bounce loses** | `entry_bear_off24lo` > 1.0 % → loser zone | PRIOR FALSESTART R1 (BTC bounce ≥ 0.5 % carries timing information; null percentile 100) + the ≤ 2 gate | Windows with entries > 1 % off the low average ≥ the others |
| 2 | **Stale coin low loses** | coin low ≥ 30 min old → loser zone | PRIOR G5b (−0.075 vs +0.120, P 0.865; fails 2026). **BLIND**, rebuild from klines. | — |
| 3 | **Efficient decline wins** | `entry_bear_eff24` < 0.25 → loser zone | THEORY (DL 60 regime thesis) | — |
| 4 | **C1 pattern loses** | `entry_pattern_c1_match` = True → loser zone | PRIOR checklist (23 fills · −0.365; observe) | — |
| 5 | **Late in the window loses** | > 4 h after monitor ON → loser zone | PRIOR TRIGGER_TIMING / EXHAUSTION (median 3.8 h to first fill). **BLIND** (derive from the window start). | — |

**No prediction:** r24 depth. Exhaustion and strength cancel; the 10-07 winners were more exhausted than the 10-22 losers.

---

## 12. What this prereg deliberately does NOT do

- No threshold was fit on B25.
- Where a prior frozen line and the master disagree, the prior was kept and the conflict printed. The data that failed a line never re-fits it.
- No sizing predictions, apart from the DL 374 ③ / ④ size lines the fade sleeve already carries (filters before sizing).
- Exit predictions appear only where the operator asked: WILLY_HT, via DL 373's shadows.
- `NONE` rows are not "no effect" claims. They mean no mechanism, so any B25 separation there is a screen finding (needs a null + OOS), not a confirmation.

**Sources used** (in-sample status per row in the CSV):
- DECISION_LOG 55, 60, 66, 111, 112, 128, 187, 200, 202, 203, 263, 275, 287, 288, 300–305, 311, 313, 315, 317, 334, 341, 348, 350, 353, 356, 358–360, 365, 367–370, 373, 374, 377
- reports/LITE_THEORY_2026-10-15.md, LITE_THEORY_PREREG.txt, WILLY_THEORY_2026-10-16.md, WILLY_THEORY_PREREG.txt, willy_theory/tables.md, B19_REVIEW_FRENZY_THEORY_AND_LEARNINGS.md, FRENZY_SLEEVE_DESIGN.md, FRENZY_WL_SCREEN_2026-10-19*, WILLY_TURNOVER_REPRICE_2026-10-19.md, WILLY_TURNOVER_V2_DEEP_2026-10-21.md, FRENZY_FLATBASE_SHORT_2026-10-22.md (+ WINLOSS, LOCK), SURGE_MISSED_OPPS_2026-10-19.md, SURGE_TRAIL_CAP_2026-10-15.md, BEARRUN_* (checklist, early trigger, exhaustion guard, false-start exit, trigger timing), FADE_WL_SCREEN_2026-10-19.md, FADE_STOP_EXECUTION_2026-10-21.md
- reports/MASTER_POOL_stacked.csv (≤ B24; medians used as MASTER_IS cuts)
