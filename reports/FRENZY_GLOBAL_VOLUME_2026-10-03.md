# 🔥 FRENZY / FRENZY_WIDE × market-wide volume (operator hypothesis, 2026-10-03)

Script: scripts/frenzy_global_volume_test.py (frozen cuts before reading). Global volume = the engine's ratio (Σ 5m base volume of the day's
top-50 ÷ Σ their 48-bar mean), rebuilt on CLOSED bars for Jan–Sep 2026 (reports/backtest_cache/k5m_full). Check vs the 565 live master stamps:
Spearman 0.66, same side of 0.7 on 82 % (live reads the forming bar → noisier). Trades: all 1,455 FRENZY first candles (100×), LAG 1, live exit, real costs.

| cohort | trades | avg %/trade | Jan–Apr / May–Sep | random beats it |
|---|---|---|---|---|
| ALL, global vol < 1.0 | 850 · 245 days | **+0.225** | +0.275 / +0.183 | 3 % |
| ALL, global vol ≥ 1.0 | 605 · 227 days | **−0.185** | −0.266 / −0.120 | 97 % |
| FRENZY (ATR ≤ 2.5 ∧ red), < 1.0 | 193 | +0.369 | +0.661 / +0.055 | 4 % |
| FRENZY, ≥ 1.0 | 134 | −0.411 | −0.356 / −0.479 | 96 % |
| WIDE (the rest), < 1.0 | 657 | +0.182 | +0.138 / +0.215 | 12 % |
| WIDE, ≥ 1.0 | 471 | −0.121 | −0.232 / −0.042 | 89 % |

Dose-response (all): <0.5 +0.51 · 0.5–0.7 +0.09 · 0.7–0.85 +0.25 · 0.85–1.0 +0.21 | 1.0–1.25 −0.19 · 1.25–1.6 −0.12 · ≥1.6 −0.24 (step at 1.0).
Low−high gap in every leave-one-month-out (+0.32 … +0.53), in every UTC 8-h block and weekday/weekend; high-volume losses spread (top pair 2 %,
top day 2 %); low-volume gains: top-3 pairs 44 % of the sum, +0.134 without them. Months where low beat high: 7 of 9.
Mechanism read: a pair's frenzy in a QUIET market is its own story and keeps going; in a market-wide volume surge it is the tide and reverts.

Expectancy bar for BLOCK at ≥ 1.0 (all): ① WR 36 % < breakeven ≈ 38–39 % ✓ · ② mean < 0 at 95 % (day bootstrap [−0.49, +0.15]) ✗ ·
③ 227 days, no concentration ✓ · ④ N ✓ → fails on confidence only → pre-registered OBSERVE candidate, threshold frozen at 1.0.
NOT tested: the live stamp's forming-bar definition at the FRENZY moment (stale: last scan's value); slots across pairs; delisted pairs.
