# EXH spike age: do EXHAUSTION_SHORT entries < 1 h after the Frenzy spike do worse at today's exit? (2026-11-26)

**Question (operator):** INITUSDT 11-26 10:00 entered 0.25 h after the spike and was stopped −5.38 % in 10 min. All other recent EXH fills entered 1.3–23 h after the spike. Do early entries lose at today's exit (TP min(20×ATR, 7 %) maker / SL min(1.5×ATR, 6 %) / 24 h, DL 609) with today's filters?

**Prior study:** reports/ENTRY_WL_2026-11-18.md tested `hrs_since_spike` at the OLD exit (TP 10) on a runner-share label and found nothing. What is new here: booked NET % at today's exit, the < 1 h bucket, a dose-response by hour bucket, and the real fills.

## Method
- Inputs are local-only research data (reports/TP_GRID_2026-11-20_EXH_data, reports/ENTRY_WL_2026-11-18_data are gitignored) — the script is not re-runnable from the public repo alone.
- **Backtest:** TP_GRID 2026-11-20 EXH rows (`rows_FRENZY.pkl`) and cell F7 from `sim_FRENZY.npz`. The net recipe is SHORT_AFTER_LOSS `exh()`: slip floor 0.10, taker 0.05, maker 0.02, funding included. Period: 2023-07 → 2026-11-16. Hours since spike come from `ENTRY_WL_2026-11-18_data/exh.pkl`, joined on `xi` and verified on pair + entry_ms. **The join rate is 100 %**, and 3,358 of 3,368 rows have an hour value.
- **Filters:** spot market required (every row has has_spot = 1; the 10 no-spot / no-hours rows are dropped). EXH_SSC_OIVOL was already removed upstream (0 rows match the block). Every row is a one-position-per-coin fill. The block is applied *before* one-position, but it freed no later signals.
- **Not applied:** EXH_BEAR80 and EXH_AFTER_DROP are observe-only, so they are correctly left out. The shared book of 4 slots is not modelled.
- **Real fills:** all 17 closed EXHAUSTION_SHORT fills since the sleeve started on 11-18, from the current batch export. MASTER_POOL_stacked has no EXH rows. The export has no is_probe column, and none of these fills are probes. P&L is `pnl_percentage` (net % of notional). The TP7 era starts at the DL 609 deploy, 2026-11-21 20:59 UTC.
- **Kept-fill breakeven WR:** **43.3 %**.

## 1 · Dose-response, backtest (N · WR · avg net %)
| bucket | ALL | DESIGN (< 2025-07) | HOLDOUT | stop % | TP % |
|---|---|---|---|---|---|
| 0–0.5 h | 216 · 38.4 % · **−0.23** | 97 · 35.1 % · −0.46 | 119 · 41.2 % · −0.04 | **61** | **32** |
| 0.5–1 h | 364 · 48.1 % · **+0.90** | 198 · 38.9 % · +0.07 | 166 · 59.0 % · +1.89 | 50 | 45 |
| 1–2 h | 474 · 46.6 % · +0.46 | 196 · 44.9 % · +0.51 | 278 · 47.8 % · +0.43 | 52 | 44 |
| 2–4 h | 479 · 49.1 % · +0.46 | 185 · 44.3 % · +0.11 | 294 · 52.0 % · +0.69 | 51 | 47 |
| 4–8 h | 412 · 48.5 % · +0.46 | 132 · 47.7 % · +0.57 | 280 · 48.9 % · +0.41 | 51 | 47 |
| 8–24 h | 804 · 51.2 % · +0.84 | 312 · 51.6 % · +1.17 | 492 · 51.0 % · +0.63 | 49 | 50 |
| 24–72 h | 609 · 50.4 % · +0.76 | 226 · 45.1 % · +0.36 | 383 · 53.5 % · +1.01 | 50 | 49 |

**By half-year, < 1 h vs ≥ 1 h avg net % (N):**

| half | < 1 h | ≥ 1 h |
|---|---|---|
| 23-H2 | +1.29 (57) | +0.42 |
| 24-H1 | −0.53 (80) | +0.46 |
| 24-H2 | −0.09 (89) | +1.11 |
| 25-H1 | −0.78 (69) | +0.38 |
| 25-H2 | +0.99 (148) | +0.58 |
| 26-H1 | +0.92 (82) | +0.40 |
| 26-H2 | +1.58 (55) | +1.07 |

The 0–0.5 h bucket was negative in 5 of 7 halves (23-H2 +1.08, 24-H1 −0.86, 24-H2 −1.13, 25-H1 −1.12, 25-H2 −0.29, 26-H1 −0.63, 26-H2 +1.69).

## 2 · < 1 h vs ≥ 1 h (10-day block bootstrap of the difference)
| scope | < 1 h | ≥ 1 h | Δ 95 % CI | days |
|---|---|---|---|---|
| ALL | 580 · 44.5 % · +0.48 | 2,778 · 49.5 % · +0.64 | [−0.74, +0.35] | 424 |
| DESIGN | 295 · 37.6 % · −0.11 | 1,051 · 47.2 % · +0.61 | [−1.50, +0.02] | 216 |
| HOLDOUT | 285 · 51.6 % · **+1.09** | 1,727 · 50.9 % · +0.65 | [−0.30, +1.19] | 208 |
| ALL, post-hoc < 0.5 h | 216 · 38.4 % · −0.23 | 3,142 · 49.3 % · +0.67 | **[−1.72, −0.20]** | 189 |

**Concentration:** the zone loss is spread out. No day carries more than 2–5 % of the zone loss, no pair more than 2–5 %, and no episode more than 3 %. Nearly every zone fill is its own spike episode (575 episodes for 580 fills).

## 3 · Locked Pattern-C bar
| zone | WR vs BE 43.3 | avg | day-bootstrap 95 % upper bound | windows | max share | N | verdict |
|---|---|---|---|---|---|---|---|
| < 1 h ALL | 44.5 ✗ | +0.48 ✗ | +0.84 ✗ | 424 ✓ | 2 % ✓ | 580 ✓ | **FAIL** |
| < 1 h HOLDOUT | 51.6 ✗ | +1.09 ✗ | +1.59 ✗ | 208 ✓ | 3 % ✓ | 285 ✓ | **FAIL** |
| < 0.5 h ALL (post-hoc) | 38.4 ✓ | −0.23 ✓ | **+0.27 ✗** | 189 ✓ | 3 % ✓ | 216 ✓ | fails on confidence |
| < 0.5 h HOLDOUT (post-hoc) | 41.2 ✓ | −0.04 | +0.80 ✗ | 107 ✓ | 5 % ✓ | 119 ✓ | fails |

## 3b · Breakeven-bot read
One size (0.61 × equity per fill), every calendar day, no slot cap. The no-slot-cap model inflates drawdown, so read these numbers relative to each other only.

| variant | DCR ALL | DCR DESIGN | DCR HOLDOUT | max DD | worst day |
|---|---|---|---|---|---|
| today | 0.821 % | 0.385 % | 1.456 % | −62.7 % | −26.6 % |
| block < 1 h | 0.715 % ↓ | 0.434 % | 1.123 % ↓ | −62.4 % | −25.2 % |
| block < 0.5 h (post-hoc) | 0.856 % | 0.429 % | 1.478 % | −54.4 % | −26.3 % |

## 4 · Real fills (pnl_percentage)
| era | < 1 h | ≥ 1 h | all |
|---|---|---|---|
| TP7 (≥ 11-21 21:00 UTC) | 1 · 0 % · −5.38 | 6 · 33 % · −1.68 | 7 · 29 % · −2.21 (4 days) |
| TP10 (before) | 0 | 10 · 60 % · +1.16 | 10 · 60 % · +1.16 |

INIT is the only real fill under 1 h. The TP7-era losses come from fills at 4–11 h (BIGTIME, MANTRA ×2), not from early entries. With N = 1, the real data can say nothing about the zone.

## 5 · Theory and monotonicity
**Why early entries could fail.** Within the first half hour, the RSI 70 → ≤ 70 cross can be a pause between the first and second leg of a pump. Spot and perp buyers are still arriving, volume has not decayed on a slower clock, and late shorts are the fuel for a squeeze. That predicts more stops and fewer TPs, which is what the 0–0.5 h bucket shows: 61 % stopped and 32 % TP, against about 50 / 47 in every other bucket.

**Why the result is weak.** It is *not monotonic*. The 0.5–1 h bucket is one of the best (+0.90, holdout +1.89), and the next three buckets are flat. That is the "mid-range hole / single edge bucket" shape the rules call noise. The 0.5 h cut was also picked after seeing the buckets, and it came from a case-driven ask (INIT), so it is in-sample by construction.

## Verdict
**< 1 h: nothing.** It fails every bar, holdout < 1 h fills beat the rest, and blocking them lowers DCR (0.82 → 0.72 %/day).

**< 0.5 h: at most an observe line.** It is post-hoc, frozen at 0.5 h, with the mechanism (stop 61 % / TP 32 %) on record. It fails the 95 % confidence bar, the holdout is flat (−0.04), and the block changes DCR only marginally (+0.03 pp/day). Bot unchanged.
