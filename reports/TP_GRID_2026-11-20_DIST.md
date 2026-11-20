# TP grid — DISTRIBUTION_SHORT (2026-11-20)

## Question (operator)
DISTRIBUTION_SHORT exits at TP = min(20 × ATR15, 10 %) (≈ 10 % on 99.6 % of fills), resting maker; stop = min(1.5 × ATR15, 6 %); 24 h.
"Maybe 10 % is too ambitious — would 6, 7 or 8 % raise WR and improve results? Tell me the optimal TP."

## Verdict — keep TP 10 % (today). Confidence: high that 4–7 % is worse; moderate that 8–9 % is not better.
- **Yes, a lower TP raises WR — and lowers the result.** WR climbs from 42.5 % (TP 10) to 46 % (8), 51.5 % (6), 60 % (4) — but the
  breakeven WR climbs faster (35.6 → 39.3 → 45.4 → 55.0 %), so the edge over breakeven SHRINKS (+6.8 → +6.7 → +6.1 → +5.2 pts at 10 → 8 → 6 → 4) while each
  win gets smaller. Avg net / fill: +0.93 (10) · +0.85 (9) · +0.81 (8) · +0.67 (7) · +0.64 (6) · +0.60 (5) · +0.44 % (4).
- **Why:** every TP X < 10 has two groups. *Gainers* = Exit-B stop / time fills that reach X first (they flip from ≈ −6 % to +X). *Payers* =
  the 10 % winners now cut to X (all ~540 of them, each losing 10 − X). Across ALL 1,547 fills the payers outweigh the gainers at every X:
  TP 8: +868 vs −1,052 · TP 6: +1,663 vs −2,116 · TP 4: +2,419 vs −3,178 (Σ net % points). Most fills that reach 6–8 % keep going to 10 %.
- **Every block agrees on the sign for 4–7 %** (DESIGN, HOLDOUT, 2026; TP 4–6 negative in all 7 half-years, TP 7 in 6 of 7); TP 8 is a coin-flip in HOLDOUT (−0.005)
  and 2026 (+0.005), negative in DESIGN (−0.32, CI excludes 0). The best-of-grid luck test finds no alternative that beats today
  (reality-check p 0.76–0.98 in every block) — the "TP 8 ties in 2026" read is what one expects from the best of 6 tries by chance.
  Only 26-H1 shows TP 7–8 ahead (+0.08 / +0.16, CIs span 0) — one half-year, 297 fills: watch-line at most.
- **What a lower TP DOES buy: a smoother curve.** HOLDOUT max drawdown (one size 0.61 × equity, every fill) −54.9 % at TP 10 vs −46 % at 8,
  −27 % at 5; worst day −21.8 % vs −18.4 % vs −14.1 %. That is a SIZE question (clustered stops), not a TP question: the same smoothing at
  higher return is had by cutting the ticket, not by cutting the winners (Size-by-edge rule, DL 547).
- **Real fills (N = 5, illustrative only):** today −2.21 %/fill avg; only TP 4 helps (EVAA reached 4.55 % before its stop); TP 5–9 all
  worse than 10 (only CYBER's 10 % winner changes, cut to X). Nothing in the current batch argues for 6–8 %.
- Optimal TP on the evidence: **10 % (unchanged)**. Runner-up 9 % (−0.08 %/fill, CI touches 0) — no reason to move. Ship nothing.

## Method
- **Master**: `reports/MASTER_POOL_stacked.csv` has 0 DISTRIBUTION fills → nothing to re-price.
- **Current batch** (`exports_archive/scalpars_orders_paper_2026-11-20_16-01-39.csv`, CLOSED, no probes): 5 DIST fills, each re-walked on its own
  1m path (cached in `reports/DIST_FB_LOSERS_2026-11-20_data/k1m`, data to 11-20 16:35 UTC — every fill resolves: a lower TP only changes a fill
  before its booked exit) from the REAL fill price, today's stop (`entry_dist_sl_pct`), TP = min(20 × ATR15, X). Stop first on the bar HIGH (gap-aware);
  maker TP only when the LOW trades through (1 tick, and ≥ 0.05 % — identical here). Costs: entry taker 0.05 %, TP maker 0.02 %, stop taker 0.05 % +
  slip (model 0.10 % or the fill's REAL stop slippage vs its stop line), real funding (Binance history: EVAA received +0.069 % at 16:00 UTC).
  `scripts/TP_GRID_2026-11-20_DIST_real.py` → `reports/TP_GRID_2026-11-20_DIST_data/real_*.csv`.
- **Backtest**: the DL 586 kept cohort (`reports/ENTRY_WL_2026-11-18_data/dist.pkl`, 1,547 fills, DESIGN < 2025-07 ≤ HOLDOUT) re-walked with the
  DIST_EXIT engine (`scripts/dist_exit_sim.py run_rows`: same fill rules, slip model, real funding, cost formula as Exit B) under the 7 TPs × 2
  trade-through rules. `scripts/TP_GRID_2026-11-20_DIST_backtest.py` → `bt_rows.csv.gz`; read-out `scripts/TP_GRID_2026-11-20_DIST_report.py`
  (`bt_1tick.md`, `bt_005.md`, `bt_grid_*.csv`).
- **Parity**: TP 10 re-walk = dist.pkl Exit B net on 1,546 / 1,547 fills (mean |d| 0.001 %; one MIRAUSDT fill books time +7.55 vs TP +9.35 — the same
  1-fill mismatch as the stop-width study). Real fills: TP 10 with real slip reproduces each booked net within 0.09 % (fee 0.045 vs 0.05 modelled).
- DCR = (Π_d (1 + r_d))^(1/D) − 1 over every calendar day D of the block, r_d = Σ net % of the fills closed that day × 0.61 / 100 (one size,
  DL 503); maxDD and worst day on the same curve. Paired Δ vs TP 10 per fill, 10-day-block bootstrap 95 % CI (4,000 draws). Luck test: centred
  block bootstrap of the max mean Δ over the 6 alternatives (White reality check). Breakeven WR = |avg loss| / (avg win + |avg loss|).
- Simplifications (stated): the entry set is held at Exit B's one-position-per-pair cohort (a faster TP frees the pair sooner and could admit a later
  signal — not modelled); 1m bars, stop first when stop and TP share a bar (conservative for a lower TP only on that bar). Concurrent fills are
  compounded at full one size each — drawdowns are a sizing read, not an account forecast. No in-sample haircut needed: nothing is shipped.
- Cross-check: the reach-based approximation (`scripts/DIST_LOSERS2_2026-11-20_tpcell.py`) is confirmed (TP 4 −0.72 / −0.36 vs −0.71 / −0.31,
  TP 5 −0.55 / −0.20 vs −0.55 / −0.20, TP 6 −0.54 / −0.15 vs −0.52 / −0.09, TP 8 −0.32 / −0.005 vs −0.28 / +0.01 DESIGN / HOLDOUT) — the true re-walk
  is slightly WORSE for lower TPs (stop-first bars, funding).

## Step 1 — real fills (current batch, N = 5 — ILLUSTRATIVE ONLY, 3 calendar days 11-18 → 11-20)
Parity (TP 10 = today):

| pair | booked | re-walk (model slip 0.10) | re-walk (real slip) | exit |
|---|---|---|---|---|
| SENT | −4.41 | −4.30 | −4.41 | stop |
| NAORIS | −5.97 | −6.20 | −5.98 | stop |
| EVAA | −4.38 | −4.85 | −4.47 | stop |
| CYBER | +9.95 | +9.93 | +9.93 | TP |
| KERNEL | −6.11 | −6.20 | −6.12 | stop |

Grid (real stop slippage; the 0.05 % trade-through rule gives identical results; model slip shifts every row by ≈ −0.1):

| TP | N | WR % | avg net % | Σ % | DCR %/day | maxDD % | worst day % | what changes |
|---|---|---|---|---|---|---|---|---|
| 4 | 5 | 40 | −1.73 | −8.65 | −1.77 | −5.2 | −3.9 | EVAA reaches 4.55 % before its stop → +3.93; CYBER cut to +3.93 |
| 5 | 5 | 20 | −3.21 | −16.05 | −3.35 | −9.7 | −9.1 | only CYBER cut to +4.93 |
| 6 | 5 | 20 | −3.01 | −15.05 | −3.16 | −9.2 | −9.1 | CYBER +5.93 |
| 7 | 5 | 20 | −2.81 | −14.05 | −2.96 | −9.1 | −9.1 | CYBER +6.93 |
| 8 | 5 | 20 | −2.61 | −13.05 | −2.76 | −9.1 | −9.1 | CYBER +7.93 |
| 9 | 5 | 20 | −2.41 | −12.05 | −2.57 | −9.1 | −9.1 | CYBER +8.93 |
| 10 (today) | 5 | 20 | −2.21 | −11.05 | −2.37 | −9.1 | −9.1 | — |

SENT (best 3.65 %), NAORIS (0.16 %) and KERNEL (2.39 %) never reach 4 % before the stop: no TP in the grid saves them — those are entry losses,
not exit losses. N = 5, one fill decides TP 4: no conclusion.

## Step 2 — backtest (1,547 kept fills, DIST_EXIT engine re-walk, engine 1-tick trade-through)
Key blocks are below; every half-year follows. The ≥ 0.05 % trade-through rule (stricter for every TP) gives the same ranking
(ALL avg: 4 +0.42 · 5 +0.57 · 6 +0.63 · 7 +0.65 · 8 +0.78 · 9 +0.84 · 10 +0.91; reality-check p 0.86–0.97) — `bt_005.md`.


### DESIGN (1tick)  N 559 · 732 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 559 | 55.5 | +0.309 | +172 | 51.5 | 309/249/1 | +0.117 | -30.6 | -13.6 | -0.718 [-1.09, -0.39] | 0.00 |
| 5 | 559 | 51.5 | +0.480 | +268 | 46.1 | 283/269/7 | +0.190 | -32.8 | -16.3 | -0.546 [-0.86, -0.24] | 0.00 |
| 6 | 559 | 47.2 | +0.483 | +270 | 42.2 | 249/292/18 | +0.182 | -35.2 | -16.3 | -0.543 [-0.81, -0.27] | 0.00 |
| 7 | 559 | 44.7 | +0.574 | +321 | 39.2 | 225/306/28 | +0.217 | -36.1 | -16.3 | -0.452 [-0.67, -0.25] | 0.00 |
| 8 | 559 | 43.3 | +0.709 | +396 | 36.9 | 203/314/42 | +0.274 | -35.7 | -19.2 | -0.317 [-0.49, -0.16] | 0.00 |
| 9 | 559 | 42.9 | +0.906 | +506 | 35.2 | 184/316/59 | +0.360 | -33.0 | -19.2 | -0.121 [-0.25, +0.02] | 0.04 |
| 10 (today) | 559 | 42.0 | +1.026 | +574 | 33.7 | 168/319/72 | +0.411 | -34.5 | -19.2 | — | — |

### HOLDOUT (1tick)  N 988 · 505 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 988 | 63.0 | +0.515 | +509 | 57.1 | 621/364/3 | +0.551 | -28.0 | -16.0 | -0.363 [-0.68, -0.04] | 0.01 |
| 5 | 988 | 58.5 | +0.674 | +666 | 51.7 | 574/408/6 | +0.721 | -26.9 | -14.1 | -0.204 [-0.52, +0.12] | 0.10 |
| 6 | 988 | 53.9 | +0.726 | +717 | 47.3 | 528/452/8 | +0.764 | -31.4 | -12.3 | -0.152 [-0.43, +0.14] | 0.15 |
| 7 | 988 | 49.9 | +0.730 | +721 | 43.7 | 482/492/14 | +0.748 | -40.6 | -13.9 | -0.147 [-0.38, +0.09] | 0.11 |
| 8 | 988 | 47.6 | +0.872 | +862 | 40.7 | 449/514/25 | +0.890 | -46.0 | -18.4 | -0.005 [-0.17, +0.16] | 0.49 |
| 9 | 988 | 44.5 | +0.815 | +806 | 38.5 | 404/543/41 | +0.799 | -43.9 | -18.4 | -0.062 [-0.18, +0.08] | 0.18 |
| 10 (today) | 988 | 42.7 | +0.877 | +867 | 36.6 | 372/561/55 | +0.853 | -54.9 | -21.8 | — | — |

### ALL (1tick)  N 1547 · 1236 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 1547 | 60.2 | +0.440 | +681 | 55.0 | 930/613/4 | +0.294 | -30.6 | -16.0 | -0.491 [-0.74, -0.25] | 0.00 |
| 5 | 1547 | 56.0 | +0.604 | +934 | 49.6 | 857/677/13 | +0.407 | -32.8 | -16.3 | -0.327 [-0.57, -0.09] | 0.00 |
| 6 | 1547 | 51.5 | +0.638 | +987 | 45.4 | 777/744/26 | +0.420 | -35.2 | -16.3 | -0.293 [-0.51, -0.08] | 0.01 |
| 7 | 1547 | 48.0 | +0.674 | +1042 | 42.1 | 707/798/42 | +0.434 | -40.6 | -16.3 | -0.258 [-0.43, -0.08] | 0.00 |
| 8 | 1547 | 46.0 | +0.813 | +1258 | 39.3 | 652/828/67 | +0.525 | -46.0 | -19.2 | -0.118 [-0.25, +0.01] | 0.04 |
| 9 | 1547 | 44.0 | +0.848 | +1312 | 37.4 | 588/859/100 | +0.540 | -43.9 | -19.2 | -0.083 [-0.18, +0.02] | 0.05 |
| 10 (today) | 1547 | 42.5 | +0.931 | +1441 | 35.6 | 540/880/127 | +0.592 | -54.9 | -21.8 | — | — |

### 2026 (1tick)  N 531 · 321 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 531 | 63.7 | +0.536 | +285 | 57.7 | 338/192/1 | +0.485 | -28.0 | -8.7 | -0.483 [-0.93, -0.05] | 0.01 |
| 5 | 531 | 58.9 | +0.670 | +356 | 52.2 | 312/217/2 | +0.606 | -26.9 | -12.0 | -0.349 [-0.75, +0.08] | 0.05 |
| 6 | 531 | 55.2 | +0.802 | +426 | 47.9 | 291/237/3 | +0.720 | -31.4 | -11.5 | -0.217 [-0.59, +0.19] | 0.14 |
| 7 | 531 | 51.2 | +0.836 | +444 | 44.2 | 268/258/5 | +0.735 | -31.7 | -11.5 | -0.183 [-0.51, +0.14] | 0.12 |
| 8 | 531 | 49.2 | +1.024 | +543 | 41.2 | 251/268/12 | +0.904 | -30.4 | -18.4 | +0.005 [-0.24, +0.24] | 0.51 |
| 9 | 531 | 45.8 | +0.935 | +497 | 38.9 | 227/285/19 | +0.805 | -34.7 | -18.4 | -0.084 [-0.24, +0.09] | 0.16 |
| 10 (today) | 531 | 43.9 | +1.019 | +541 | 36.9 | 212/295/24 | +0.871 | -33.6 | -21.8 | — | — |

### 23-H2 (1tick)  N 96 · 184 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 96 | 63.5 | +1.183 | +114 | 47.2 | 61/35/0 | +0.361 | -11.7 | -7.4 | -1.531 [-2.32, -0.82] | 0.00 |
| 5 | 96 | 61.5 | +1.616 | +155 | 41.8 | 58/37/1 | +0.493 | -11.2 | -7.4 | -1.098 [-1.85, -0.44] | 0.00 |
| 6 | 96 | 59.4 | +1.934 | +186 | 37.9 | 53/39/4 | +0.587 | -11.7 | -7.4 | -0.780 [-1.45, -0.19] | 0.01 |
| 7 | 96 | 57.3 | +2.122 | +204 | 35.7 | 48/41/7 | +0.641 | -10.2 | -7.4 | -0.592 [-1.17, -0.13] | 0.01 |
| 8 | 96 | 56.2 | +2.345 | +225 | 33.9 | 43/42/11 | +0.707 | -16.2 | -7.4 | -0.369 [-0.78, +0.01] | 0.03 |
| 9 | 96 | 56.2 | +2.582 | +248 | 32.6 | 38/42/16 | +0.775 | -15.2 | -7.4 | -0.132 [-0.42, +0.20] | 0.19 |
| 10 (today) | 96 | 55.2 | +2.714 | +261 | 31.4 | 34/43/19 | +0.817 | -14.2 | -7.4 | — | — |

### 24-H1 (1tick)  N 143 · 180 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 143 | 54.5 | +0.199 | +28 | 52.0 | 77/65/1 | +0.068 | -29.7 | -13.6 | -0.423 [-1.38, +0.25] | 0.12 |
| 5 | 143 | 51.0 | +0.407 | +58 | 46.4 | 71/70/2 | +0.163 | -30.6 | -13.6 | -0.215 [-1.03, +0.39] | 0.25 |
| 6 | 143 | 46.2 | +0.351 | +50 | 42.5 | 60/76/7 | +0.123 | -35.2 | -13.6 | -0.272 [-0.91, +0.26] | 0.16 |
| 7 | 143 | 42.0 | +0.187 | +27 | 40.1 | 50/82/11 | +0.040 | -36.1 | -13.6 | -0.436 [-0.88, -0.11] | 0.00 |
| 8 | 143 | 41.3 | +0.339 | +48 | 38.1 | 43/83/17 | +0.113 | -35.7 | -13.6 | -0.284 [-0.60, -0.05] | 0.01 |
| 9 | 143 | 41.3 | +0.492 | +70 | 36.8 | 37/83/23 | +0.181 | -31.3 | -13.6 | -0.131 [-0.33, +0.06] | 0.09 |
| 10 (today) | 143 | 40.6 | +0.623 | +89 | 35.2 | 35/84/24 | +0.232 | -34.5 | -14.1 | — | — |

### 24-H2 (1tick)  N 144 · 179 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 144 | 50.0 | -0.006 | -1 | 50.1 | 72/72/0 | -0.034 | -23.6 | -10.6 | -0.638 [-0.96, -0.37] | 0.00 |
| 5 | 144 | 44.4 | -0.022 | -3 | 44.7 | 62/79/3 | -0.048 | -32.8 | -16.3 | -0.654 [-0.98, -0.24] | 0.00 |
| 6 | 144 | 41.0 | -0.005 | -1 | 41.0 | 55/84/5 | -0.040 | -31.3 | -16.3 | -0.637 [-0.96, -0.25] | 0.00 |
| 7 | 144 | 38.9 | +0.121 | +17 | 37.7 | 50/87/7 | +0.010 | -24.5 | -16.3 | -0.511 [-0.77, -0.27] | 0.00 |
| 8 | 144 | 38.2 | +0.345 | +50 | 35.1 | 48/88/8 | +0.113 | -21.4 | -16.3 | -0.287 [-0.55, -0.04] | 0.01 |
| 9 | 144 | 37.5 | +0.521 | +75 | 33.0 | 44/89/11 | +0.197 | -21.2 | -16.3 | -0.112 [-0.29, +0.15] | 0.18 |
| 10 (today) | 144 | 36.1 | +0.632 | +91 | 30.9 | 40/89/15 | +0.243 | -21.2 | -16.3 | — | — |

### 25-H1 (1tick)  N 176 · 179 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 176 | 56.2 | +0.178 | +31 | 54.1 | 99/77/0 | +0.072 | -27.0 | -9.9 | -0.578 [-1.31, +0.10] | 0.05 |
| 5 | 176 | 52.3 | +0.331 | +58 | 48.7 | 92/83/1 | +0.153 | -22.8 | -10.7 | -0.425 [-1.06, +0.19] | 0.10 |
| 6 | 176 | 46.6 | +0.200 | +35 | 44.6 | 81/93/2 | +0.060 | -34.5 | -15.3 | -0.556 [-1.11, +0.00] | 0.03 |
| 7 | 176 | 44.9 | +0.414 | +73 | 41.2 | 77/96/3 | +0.180 | -31.6 | -15.3 | -0.342 [-0.79, +0.12] | 0.08 |
| 8 | 176 | 42.0 | +0.415 | +73 | 38.6 | 69/101/6 | +0.168 | -29.9 | -19.2 | -0.341 [-0.68, +0.05] | 0.04 |
| 9 | 176 | 41.5 | +0.642 | +113 | 36.4 | 65/102/9 | +0.299 | -33.0 | -19.2 | -0.114 [-0.36, +0.21] | 0.22 |
| 10 (today) | 176 | 40.9 | +0.756 | +133 | 35.2 | 59/103/14 | +0.366 | -30.9 | -19.2 | — | — |

### 25-H2 (1tick)  N 457 · 185 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 457 | 62.1 | +0.490 | +224 | 56.5 | 283/172/2 | +0.663 | -26.4 | -16.0 | -0.223 [-0.71, +0.23] | 0.17 |
| 5 | 457 | 58.0 | +0.678 | +310 | 51.0 | 262/191/4 | +0.918 | -24.9 | -14.1 | -0.035 [-0.50, +0.44] | 0.43 |
| 6 | 457 | 52.5 | +0.638 | +292 | 46.6 | 237/215/5 | +0.838 | -29.8 | -12.3 | -0.075 [-0.48, +0.36] | 0.36 |
| 7 | 457 | 48.4 | +0.607 | +277 | 43.2 | 214/234/9 | +0.765 | -40.6 | -13.9 | -0.106 [-0.47, +0.26] | 0.28 |
| 8 | 457 | 45.7 | +0.696 | +318 | 40.2 | 198/246/13 | +0.862 | -46.0 | -17.3 | -0.017 [-0.28, +0.23] | 0.45 |
| 9 | 457 | 43.1 | +0.676 | +309 | 38.1 | 177/258/22 | +0.786 | -43.9 | -17.7 | -0.037 [-0.23, +0.19] | 0.36 |
| 10 (today) | 457 | 41.4 | +0.713 | +326 | 36.3 | 160/266/31 | +0.817 | -54.9 | -21.5 | — | — |

### 26-H1 (1tick)  N 297 · 180 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 297 | 60.6 | +0.261 | +77 | 57.7 | 180/117/0 | +0.205 | -28.0 | -8.7 | -0.279 [-0.82, +0.25] | 0.17 |
| 5 | 297 | 56.6 | +0.438 | +130 | 52.2 | 168/129/0 | +0.373 | -23.4 | -12.0 | -0.102 [-0.63, +0.48] | 0.37 |
| 6 | 297 | 52.5 | +0.502 | +149 | 47.9 | 155/141/1 | +0.418 | -31.4 | -11.5 | -0.038 [-0.57, +0.59] | 0.44 |
| 7 | 297 | 49.5 | +0.622 | +185 | 44.3 | 144/150/3 | +0.525 | -30.0 | -10.9 | +0.082 [-0.30, +0.50] | 0.65 |
| 8 | 297 | 46.5 | +0.704 | +209 | 41.0 | 134/158/5 | +0.588 | -28.0 | -18.4 | +0.164 [-0.11, +0.43] | 0.89 |
| 9 | 297 | 42.8 | +0.520 | +154 | 38.9 | 118/168/11 | +0.400 | -34.7 | -18.4 | -0.020 [-0.23, +0.25] | 0.42 |
| 10 (today) | 297 | 40.7 | +0.540 | +160 | 37.0 | 108/174/15 | +0.403 | -33.6 | -21.8 | — | — |

### 26-H2 (1tick)  N 234 · 140 calendar days
| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 234 | 67.5 | +0.886 | +207 | 57.7 | 158/75/1 | +0.849 | -12.7 | -7.2 | -0.741 [-1.42, -0.07] | 0.02 |
| 5 | 234 | 62.0 | +0.964 | +226 | 52.3 | 144/88/2 | +0.911 | -13.6 | -11.5 | -0.663 [-1.28, -0.06] | 0.01 |
| 6 | 234 | 58.5 | +1.182 | +276 | 47.8 | 136/96/2 | +1.114 | -25.9 | -11.5 | -0.446 [-0.90, -0.01] | 0.02 |
| 7 | 234 | 53.4 | +1.108 | +259 | 44.2 | 124/108/2 | +1.012 | -29.5 | -11.5 | -0.519 [-0.97, -0.03] | 0.02 |
| 8 | 234 | 52.6 | +1.430 | +335 | 41.5 | 117/110/7 | +1.318 | -28.2 | -14.4 | -0.197 [-0.59, +0.23] | 0.17 |
| 9 | 234 | 49.6 | +1.462 | +342 | 39.0 | 109/117/8 | +1.333 | -26.8 | -14.4 | -0.165 [-0.38, +0.07] | 0.08 |
| 10 (today) | 234 | 47.9 | +1.627 | +381 | 36.8 | 104/121/9 | +1.482 | -25.5 | -14.4 | — | — |

### Decomposition vs today (1tick) — Σ Δ net % by group
| block | TP | gainers N | gainers ΣΔ | payers N | payers ΣΔ | other ΣΔ | net ΣΔ | Δavg w/o best 5 % | top pair (ΣΔ) | Δavg w/o top pair |
|---|---|---|---|---|---|---|---|---|---|---|
| DESIGN | 4 | 141 | +582 | 168 | -983 | +0 | -401 | -1.270 | YGGUSDT (+20) | -0.753 |
| DESIGN | 5 | 115 | +517 | 168 | -822 | +0 | -305 | -1.127 | YGGUSDT (+22) | -0.585 |
| DESIGN | 6 | 81 | +352 | 168 | -656 | +0 | -303 | -1.128 | FLMUSDT (+20) | -0.579 |
| DESIGN | 7 | 57 | +239 | 168 | -492 | +0 | -253 | -0.897 | RVNUSDT (+16) | -0.481 |
| DESIGN | 8 | 35 | +149 | 167 | -327 | +1 | -177 | -0.613 | RVNUSDT (+17) | -0.348 |
| DESIGN | 9 | 16 | +96 | 167 | -164 | +0 | -67 | -0.309 | RVNUSDT (+18) | -0.153 |
| HOLDOUT | 4 | 248 | +1837 | 373 | -2195 | +0 | -358 | -0.927 | FOLKSUSDT (+29) | -0.392 |
| HOLDOUT | 5 | 201 | +1623 | 373 | -1825 | +0 | -201 | -0.812 | FOLKSUSDT (+32) | -0.236 |
| HOLDOUT | 6 | 155 | +1311 | 373 | -1460 | +0 | -150 | -0.809 | NAORISUSDT (+31) | -0.183 |
| HOLDOUT | 7 | 109 | +949 | 373 | -1094 | +0 | -146 | -0.852 | EPTUSDT (+26) | -0.174 |
| HOLDOUT | 8 | 76 | +719 | 370 | -725 | +1 | -5 | -0.717 | EPTUSDT (+28) | -0.034 |
| HOLDOUT | 9 | 31 | +298 | 368 | -363 | +4 | -61 | -0.387 | DIAUSDT (+15) | -0.077 |
| ALL | 4 | 389 | +2419 | 541 | -3178 | +0 | -759 | -1.060 | FOLKSUSDT (+29) | -0.510 |
| ALL | 5 | 316 | +2140 | 541 | -2647 | +0 | -507 | -0.940 | FOLKSUSDT (+32) | -0.348 |
| ALL | 6 | 236 | +1663 | 541 | -2116 | +0 | -453 | -0.954 | NAORISUSDT (+31) | -0.313 |
| ALL | 7 | 166 | +1188 | 541 | -1586 | +0 | -399 | -0.944 | EPTUSDT (+26) | -0.275 |
| ALL | 8 | 111 | +868 | 537 | -1052 | +2 | -183 | -0.696 | EPTUSDT (+28) | -0.136 |
| ALL | 9 | 47 | +394 | 535 | -527 | +4 | -129 | -0.359 | RVNUSDT (+18) | -0.095 |

### Best-of-grid luck test (1tick) — max mean Δ over the 6 alternatives vs centred 10-day-block bootstrap
- DESIGN: best TP 9 mean Δ -0.121 %/fill, reality-check p = 0.984
- HOLDOUT: best TP 8 mean Δ -0.005 %/fill, reality-check p = 0.762
- ALL: best TP 9 mean Δ -0.083 %/fill, reality-check p = 0.982
- 2026: best TP 8 mean Δ +0.005 %/fill, reality-check p = 0.763

Reading the decomposition: "Δavg w/o best 5 %" drops the 5 % of fills where the lower TP helped most — every cell stays negative; no single pair
carries the gainers (top pair ≤ +32 of ΣΔ; removing it leaves every Δ negative). "Other" ≈ 0 confirms the grid only moves gainers and payers.

## WR vs result — the trade-off in one table (ALL, 1,547 fills)
| TP | WR % | breakeven WR % | WR − BE (pts) | avg win % | avg loss % | avg net % |
|---|---|---|---|---|---|---|
| 4 | 60.2 | 55.0 | +5.2 | +3.80 | -4.65 | +0.440 |
| 5 | 56.0 | 49.6 | +6.4 | +4.76 | -4.68 | +0.604 |
| 6 | 51.5 | 45.4 | +6.1 | +5.68 | -4.72 | +0.638 |
| 7 | 48.0 | 42.1 | +5.9 | +6.56 | -4.77 | +0.674 |
| 8 | 46.0 | 39.3 | +6.7 | +7.39 | -4.79 | +0.813 |
| 9 | 44.0 | 37.4 | +6.6 | +8.08 | -4.83 | +0.848 |
| 10 | 42.5 | 35.6 | +6.8 | +8.76 | -4.85 | +0.931 |

Higher WR is not better results when the average win shrinks faster than the WR rises: the stop (≈ −4 to −6 %) is unchanged, so each
lower TP needs a much higher WR just to break even, and the fills that reach 6–8 % mostly go on to reach 10 %.

## Revert / re-open gate
Nothing shipped. Re-open only if a forward cohort (paper/live DIST fills on today's entry stack) of ≥ 30 fills shows TP 8 ahead of TP 10 on the
re-walk with a 10-day-block CI excluding 0 — the 26-H1 tilt (+0.16, CI −0.11 / +0.43) is the only cell that would need it.

## Equal-drawdown read (added 2026-11-20, scripts/TP_GRID_2026-11-20_DIST_risk.py)

Method (operator 2026-11-20 "what is safer"): each cell is sized so ITS backtest max drawdown is −30 %, then its daily compound return at that size is compared with today's exit; 10-day-block bootstrap (500 resamples, size re-solved in each), paired by day. HOLDOUT = entries ≥ 2025-07-01. Caveats: the drawdown comes from one history (bootstrapped) and assumes every fill at full size with NO shared-book slot cap — the cluster days that set it are partly out of reach for the live book; P(better) is not a p-value and the cells were not pre-registered (best-of-few).

Bootstrap run on TP 5 / 6 / 8 only (chosen after the same-size read; TP 9 not bootstrapped).

| block | TP | size for −30 % DD | daily compound at that size | worst day | P(better than 10 %) | gain (90 % range) |
|---|---|---|---|---|---|---|
| ALL | 4 % | 0.60 × | +0.29 %/day | -15.6 % | — | — |
| ALL | 5 % | 0.55 × | +0.37 %/day | -14.8 % | 0.52 | +0.01 (-0.18 … +0.16) |
| ALL | 6 % | 0.51 × | +0.36 %/day | -13.6 % | 0.35 | -0.04 (-0.19 … +0.11) |
| ALL | 7 % | 0.43 × | +0.32 %/day | -11.5 % | — | — |
| ALL | 8 % | 0.37 × | +0.34 %/day | -11.6 % | 0.50 | +0.00 (-0.12 … +0.11) |
| ALL | 9 % | 0.40 × | +0.37 %/day | -12.4 % | — | — |
| ALL | 10 % (today) | 0.29 × | +0.31 %/day | -10.5 % | — | — |
| HOLDOUT | 4 % | 0.66 × | +0.59 %/day | -17.2 % | — | — |
| HOLDOUT | 5 % | 0.69 × | +0.80 %/day | -16.0 % | 0.81 | +0.19 (-0.16 … +0.61) |
| HOLDOUT | 6 % | 0.58 × | +0.73 %/day | -11.7 % | 0.75 | +0.12 (-0.23 … +0.46) |
| HOLDOUT | 7 % | 0.43 × | +0.55 %/day | -9.8 % | — | — |
| HOLDOUT | 8 % | 0.37 × | +0.57 %/day | -11.1 % | 0.79 | +0.10 (-0.12 … +0.33) |
| HOLDOUT | 9 % | 0.40 × | +0.56 %/day | -11.9 % | — | — |
| HOLDOUT | 10 % (today) | 0.29 × | +0.46 %/day | -10.5 % | — | — |

Read: over 2023–26 a coin flip (P 0.35–0.52); HOLDOUT leans to a lower TP (P 0.75–0.81), every range includes 0. Keep 10 %; the risk lever is size.
