# FLATBASE confirmation-depth filter (margin 1 %, frozen DL 429) · exit +5 / −10 / 72 h · one size 0.38 × account

### ① MASTER all batches (today's stack, re-simulated) — 38 episodes · 35 decided (3 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 23 | 91% | +3.63 | +83.6 | — | — | — | +1.15 |
| M1 | 22 | 95% | +4.26 | +93.6 | +10.1 | +0.0 | 0.64 | +1.29 |
| SK1 | 20 | 95% | +4.19 | +83.8 | +0.2 | -9.9 | 0.51 | +1.16 |

### ① B24 — 2 episodes · 2 decided (0 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 1 | 100% | +4.94 | +4.9 | — | — | — | +0.31 |
| M1 | 1 | 100% | +4.94 | +4.9 | +0.0 | +0.0 | 0.00 | +0.31 |
| SK1 | 1 | 100% | +4.94 | +4.9 | +0.0 | +0.0 | 0.00 | +0.31 |

### ① B25 — 33 episodes · 32 decided (1 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 21 | 90% | +3.51 | +73.7 | — | — | — | +1.30 |
| M1 | 20 | 95% | +4.19 | +83.8 | +10.1 | +0.0 | 0.64 | +1.49 |
| SK1 | 18 | 94% | +4.10 | +73.9 | +0.2 | -9.9 | 0.51 | +1.31 |

### ① B26 — current batch — 3 episodes · 1 decided (2 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 1 | 100% | +4.94 | +4.9 | — | — | — | +0.47 |
| M1 | 1 | 100% | +4.94 | +4.9 | +0.0 | +0.0 | — | +0.47 |
| SK1 | 1 | 100% | +4.94 | +4.9 | +0.0 | +0.0 | — | +0.47 |

**Episodes the filter changes (master):**
- LAUSDT 2026-10-24T10:10 (B25) depth 0.26 % · LIVE TP +4.94 → M1 TP +4.94 · SK1 SKIPPED_SHALLOW +0.00
- BANKUSDT 2026-10-26T12:45 (B25) depth 0.73 % · LIVE STOP -10.09 → M1 SKIPPED +0.00 · SK1 SKIPPED_SHALLOW +0.00
- EDGEUSDT 2026-11-09T15:30 (B25) depth 0.89 % · LIVE TP +4.94 → M1 TP +4.94 · SK1 SKIPPED_SHALLOW +0.00
- BARDUSDT 2026-11-12T16:35 (B26) depth 0.98 % · LIVE STOP -10.09 → M1 PENDING +nan · SK1 SKIPPED_SHALLOW +0.00
- DODOXUSDT 2026-11-12T15:50 (B26) depth 0.90 % · LIVE STOP -10.09 → M1 PENDING +nan · SK1 SKIPPED_SHALLOW +0.00

### ② BACKTEST all (study FLAT episodes 2024-11 → 2026-10-22, turnover unknown) — 2455 episodes · 2449 decided (6 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 1989 | 70% | +0.81 | +1616.9 | — | — | — | +0.77 |
| M1 | 1855 | 69% | +0.67 | +1237.1 | -379.8 | -364.8 | 0.00 | +0.56 |
| SK1 | 1472 | 70% | +0.72 | +1054.2 | -562.7 | -572.8 | 0.00 | +0.50 |

### ② backtest 2025H1 — 397 episodes · 397 decided (0 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 312 | 71% | +1.15 | +359.3 | — | — | — | +0.56 |
| M1 | 287 | 73% | +1.29 | +370.1 | +10.8 | +25.8 | — | +0.58 |
| SK1 | 241 | 74% | +1.55 | +373.0 | +13.6 | +3.6 | — | +0.60 |

### ② backtest 2025H2 — 807 episodes · 807 decided (0 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 654 | 68% | +0.56 | +365.0 | — | — | — | +0.63 |
| M1 | 621 | 66% | +0.27 | +170.6 | -194.4 | -179.4 | — | +0.23 |
| SK1 | 489 | 66% | +0.25 | +121.0 | -244.0 | -254.0 | — | +0.16 |

### ② backtest 2026H1 — 753 episodes · 753 decided (0 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 641 | 70% | +0.59 | +378.9 | — | — | — | +0.66 |
| M1 | 591 | 69% | +0.49 | +287.5 | -91.4 | -76.4 | — | +0.47 |
| SK1 | 451 | 69% | +0.37 | +166.0 | -212.9 | -223.0 | — | +0.25 |

### ② backtest 2026H2 — 498 episodes · 492 decided (6 still running, out)
| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |
|---|---|---|---|---|---|---|---|---|
| LIVE | 382 | 75% | +1.34 | +513.7 | — | — | — | +1.56 |
| M1 | 356 | 73% | +1.15 | +408.9 | -104.8 | -89.8 | — | +1.23 |
| SK1 | 291 | 75% | +1.35 | +394.2 | -119.5 | -129.6 | — | +1.21 |

② backtest — the SHALLOW cohort under today's rule (depth < 1 %): N 517 · WR 71% · avg +1.09 · deep: N 1474
  depth 0–0.5 %: N 255 · WR 67% · avg +0.62
  depth 0.5–1 %: N 264 · WR 75% · avg +1.57
  depth 1–2 %: N 392 · WR 70% · avg +0.81
  depth 2–4 %: N 558 · WR 72% · avg +1.01
  depth 4–100 %: N 522 · WR 68% · avg +0.33
