# Sleeve baselines — what each active sleeve SHOULD look like (2026-11-18)

Purpose (operator 2026-11-18): "save the studies, for when we check the paper trades winners vs losers against their backtests — of each active sleeve".
At every batch review, compare each sleeve's paper / live fills with the baseline below: trades/day, WR, avg win / avg loss, avg net, exit mix,
and the winners-vs-losers separators the study found. A sleeve whose forward fills sit far outside its baseline is investigated BEFORE any
re-size or go-live decision. All numbers: net % per trade (fees, slippage 0.10 %/side, real funding), backtest unless marked "real".

| Sleeve | Status (11-18) | Trades/day | WR | Avg win / avg loss | Avg net %/trade | Exit mix | Source report | Trade-level baseline file |
|---|---|---|---|---|---|---|---|---|
| **EXHAUSTION_SHORT** (Frenzy universe; DL 578 rule, DL 579 exit TP min(20×ATR,10 %) / SL min(1.5×ATR,6 %) / 24 h, DL 582 filter) | paper, armed | **2.7** | **42 %** | +8.7 / −4.8 | **+0.85** (clean 2023-H1 rule +0.10 before the new exit — the realistic low end) | TP 34 % · SL 57 % · time 9 % | EXH_EXIT_GRID_2026-11-18 · EXH_BADDAY_2026-11-18 · EXHAUSTION_EXPAND_2026-11-18 | BASELINE_TRADES_EXHAUSTION_2026-11-18.csv (`dl582_block` = the filter's blocked rows; entry features incl. ssc, oi_vol, vd, rsi15, atrp, L/S) |
| **EXHAUSTION WIDE universe** (DL 585, L4: spike 15× ∧ $5M floor, + DL 582 filter) | building (paper) | ≈ +1.0 (pre-filter +1.18) | — | — | **+0.55** added trades kept by the filter (blocked +0.04); 3 of 7 half-years negative | — | LEADS_DEEP_2026-11-18 (L4) · EXPAND_EDGES_2026-11-18 | BASELINE_TRADES_EXH_WIDE_L4_2026-11-18.csv.gz (`x4` / filter columns; EX_r / EX_k / EX_x = exit result) |
| **DISTRIBUTION_SHORT** (DL 586, L1, 0.5 × 5×: red 15m volume candle ∧ top-trader position L/S 4 h change ≤ −0.011 ∧ 🔒 1h RSI ≥ 74.07 (armed, declared override; ≈ 1.3/day, kept HOLDOUT +0.79); exit = EXHAUSTION's) | building (paper) | **3.9** | 35.6 % (EXH exit) · 47 % (TP 5 %) | +8.05 / −3.98 | **+0.30** at the EXHAUSTION exit (+0.17 at TP 5 %); 26-H1 ≈ 0; opposite case (positions rising) −0.36 | — | LEADS_DEEP_2026-11-18 (L1) · DIST_WL_2026-11-18 · UNIFIED_SHORT_2026-11-18 | DIST_WL_2026-11-18_trades.csv.gz (4,649 fills, every feature + path label) |
| ~~CRASH_DIP_LONG~~ (dropped 11-18 — too few trades, one window = 77 % of profit: BTC 24 h ≤ −2.63 % ∧ 1 h ≤ −0.86 % ∧ BTC daily EMA50 > EMA200; red 15m bar on pump coins; TP 5 % / SL 6 % / 24 h) | building (paper) | crash days only (≈ 6–8 episodes / year; ≈ 0.3–0.6/day averaged) | 67 % of episodes positive | — | **+0.39** (+0.24 at 0.20 % slip) | — | REGIME_LONGS_2026-11-18 · LONG_LEADS_DEEP_2026-11-18 | REGIME_LONGS_2026-11-18_crashdip_R1_episodes.csv (each episode) · LONG_LEADS_DEEP_2026-11-18_episodes.csv |
| **FLATBASE_SHORT** (DL 348 → 577, NEAR_BASE) | paper, armed | **0.63** | **74 %** | +4.7 / −9.4 | **+1.04** (independent rebuild +0.91 · NEAR_BASE kept 619 · 74 % · +0.97) | TP / 10 % stop / 72 h | HIGH_ATR_2026-11-18 · VERIFY_FLATBASE_SHORT_2026-11-17 · DL 577 | BASELINE_TRADES_FLATBASE_2026-11-18.csv |
| **FRENZY_WILLY** (V2b exit; DL 527 filters) | paper, 0.5 × 5× (DL 581) | 0.8 | 60 % | — | **−0.04** full history (corrected) · real master 24 · 71 % · +2.75 (filters fitted on those fills) | — | WILLY_BACKTEST_WHY_2026-11-18 · WILLY_EXIT_V2_2026-11-18 · HT_TP6_WL_2026-11-18 | BASELINE_TRADES_WILLY_HT_2026-11-18.csv (sleeve = FRENZY_WILLY; ~60 entry features) |
| **FRENZY_WILLY_HT** (+2 / −4) | paper, 0.5 × 5× | 0.6 | 64 % | — | **−0.19** full history · real master 15 · 80 % · +1.78 | — | same | same file (sleeve = FRENZY_WILLY_HT) |
| **FRENZY_LONG** | paper, 0.2 × 0.05 | — | — | — | negative on 2024 (fails) | — | VERIFY_FRENZY_LONG_2026-11-17 | VERIFY_FRENZY_LONG_2026-11-17_verify.csv |

## Winners vs losers: what the studies found (check the forward fills against these)
- **EXHAUSTION:** the edge is the STATE (volume still ≥ 64 % of peak, spot share falling), not the RSI moment (random entries in the same state earn the same). Losers: spot share falling only a little ∧ open interest high vs volume (DL 582 filter). Stop-cluster days are NOT visible at entry (EXH_BADDAY: 117,841 pairs, regimes, DI — nothing); about half of them are chance. After a TP the price fell a median 7.1 % further (why TP 10 %).
- **EXHAUSTION WIDE:** the DL 582 filter separates the added trades (kept +0.55 vs blocked +0.04); positive in both BTC daily-bear windows.
- **DISTRIBUTION:** (DIST_WL) the coin must still be STRONG at the red candle — RSI1h ≥ 74.07 kept +0.79 vs blocked −0.00 on HOLDOUT, 7/7 half-years (ARMED 11-18 as a declared override, distribution_rsi1h_min; stamps entry_dist_rsi1h / di_diff1h / gap15). Losses are mainly exit (43 % GAVE_BACK after ≈ +3.2 %). OI, whale-vs-crowd, candle shape, spot share, funding: no effect.
- **FLATBASE:** the flat 14-day base is the edge; NEAR_BASE (≤ 2.09 % above the base → skip) separates kept +0.97 vs blocked −0.30.
- **WILLY / HT:** no separator survives (4 h trend, crowding, spike-high distance all fail across periods); losses are the stop being hit first (entry problem). Watch lines DL 580.

## How to use at a batch review
1. For each sleeve, compute forward N · trades/day · WR · avg win / loss · avg · exit mix (paper and live separately; paper avg % from DL 584 includes entry walk).
2. Compare with the row above (and the period breakdown in the source report); a gap > 2 standard errors → investigate (rule drift, data, costs) before any decision.
3. Run the sleeve's winners-vs-losers sweep on the forward fills and check the separators listed above still point the same way.
4. Quote the capacity stamps (CLAUDE.md capacity rule) alongside.
