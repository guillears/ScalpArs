# "Dying frenzy" rebuild test — 2026-11-09 (pre-registered, DECISION_LOG 512)

Research only, no bot change. The pre-registration was written before any result was read and is followed exactly. Exploratory reads come
afterwards, are labelled, and play no part in the verdict.

## What was measured
- **E (energy left)** = 5m ATR(14) at the signal bar ÷ 5m ATR(14) at the spike bar. The ATR is the engine's own formula: `services/surge.wilder_atr_pct`
  over the 300 closed 5m bars ending at that bar, the same call FRENZY makes. Signal bar = the last closed 5m bar when the fill opened. Spike bar
  = the anchor of the episode, found by running the engine's own `frenzy_walk` (1,499 closed 5m bars plus `normal_hour_usd` on 1h bars, as live does).
- **K** = the fill's number inside its episode = 1 + every earlier bot fill on that pair (any sleeve, manual entries out) opened after the spike bar closed.
  The fill is inside a live episode, so every earlier fill since the spike belongs to the same episode, and the episode end is not needed.
  History = every master-pool row (kept, blocked and probe rows: the bot really traded them) + the whole 11-09 export.
- **Frozen cuts**: E below the pooled master + B25 median of the sleeve = "low" (FRENZY_LONG 1.285 · LITE 0.907 · RETRY 0.909). K ≥ 2 vs K = 1.
- **Data**: master = `MASTER_POOL_stacked.csv` stack_keep, probes out, `stack_pct`, FRENZY_LONG re-priced at −4 (`master_exits.reprice_long_stop4`).
  B25 = the export, CLOSED, opened ≥ 10-23 18:30, LITE C +2 closes re-priced at +3 (`reprice_b25_exits`). Operator-closed fills out (master: LITE ICNT 10-20,
  RETRY AUSDT 10-20; B25: none). These match the best-case and FRENZY_LONG review counts (LONG 29 · 86 % · +0.74 / 17 · 47 % · −1.41).
- **Klines**: Binance public fapi, 67 pairs, weight header watched, no 418 / 429.
- **Daily compound (DCR)**: one size (`one_size.py`, X = LONG 0.05 · LITE 1.94 · RETRY 0.98), master = mean of per-era DCRs (the `best_dcr` method).
- **P** = day-block bootstrap (5,000 draws), P(the predicted-bad side's avg < the other side's). **LOO** = the range of the difference (bad − good, avg %)
  when each fill is left out in turn.

## Coverage
| Sleeve | Master scored | B25 scored | Walk vs the `entry_frenzy_hours` stamp |
|---|---|---|---|
| FRENZY_LONG | 29 / 29 | 17 / 17 | 45 / 46 agree; 1 differs by one bar |
| FRENZY_LITE | 30 / 30 | 31 / 31 | 60 / 61 agree; 1 differs by one bar |
| FRENZY_LITE_RETRY | 2 / 2 | 37 / 37 | 35 / 39 agree: 2 differ by one bar; BIRB 11-06 differs (the walk re-anchored on a newer spike, and per the rule the walk wins); ARIA 11-05 did not rebuild, so the stamp was used |

## Confirmatory results (frozen cuts; each cell = N · WR · avg % · Σ % · days)
| Sleeve | Measure | Block | Predicted-bad side | Other side | Diff (avg %) | LOO | P | DCR %/day before → after blocking the bad side |
|---|---|---|---|---|---|---|---|---|
| FRENZY_LONG | E low | Master | 14 · 86 % · +0.71 · +10.0 · 9 d | 15 · 87 % · +0.77 · +11.5 · 9 d | −0.05 | −0.39 … +0.31 | 0.52 | +0.061 → +0.035 |
| FRENZY_LONG | E low | B25 | 9 · 33 % · −2.17 · −19.6 · 6 d | 8 · 62 % · −0.56 · −4.5 · 4 d | −1.61 | −2.11 … −1.32 | 0.87 | −0.065 → −0.012 |
| FRENZY_LONG | K ≥ 2 | Master | 19 · 84 % · +0.63 · +12.0 · 9 d | 10 · 90 % · +0.95 · +9.5 · 9 d | −0.32 | −0.87 … −0.06 | 0.69 | +0.061 → +0.022 |
| FRENZY_LONG | K ≥ 2 | B25 | 12 · 42 % · −1.71 · −20.5 · 7 d | 5 · 60 % · −0.70 · −3.5 · 5 d | −1.02 | −1.85 … −0.46 | 0.77 | −0.065 → −0.009 |
| FRENZY_LITE | E low | Master | 12 · 50 % · +0.00 · +0.0 · 8 d | 18 · 56 % · +0.35 · +6.3 · 11 d | −0.34 | −0.62 … −0.07 | 0.58 | −1.023 → −1.003 |
| FRENZY_LITE | E low | B25 | 18 · 56 % · +0.26 · +4.7 · 9 d | 13 · 69 % · +1.15 · +15.0 · 7 d | −0.89 | −1.24 … −0.70 | 0.73 | +1.910 → +1.481 |
| FRENZY_LITE | K ≥ 2 | Master | 16 · 44 % · −0.35 · −5.6 · 8 d | 14 · 64 % · +0.85 · +11.9 · 10 d | −1.20 | −1.53 … −1.02 | 0.86 | −1.023 → +0.558 |
| FRENZY_LITE | K ≥ 2 | B25 | 26 · 69 % · +1.11 · +28.8 · 11 d | 5 · 20 % · −1.81 · −9.0 · 4 d | **+2.92** | +2.61 … +4.12 | 0.00 | +1.910 → −1.000 |
| RETRY | E low | Master | 1 · 100 % · +3.00 | 1 · 100 % · +3.00 | 0.00 | – | – | +1.147 → +0.580 |
| RETRY | E low | B25 | 18 · 50 % · −0.00 · −0.0 · 11 d | 19 · 53 % · +0.15 · +2.9 · 10 d | −0.16 | −0.33 … +0.02 | 0.57 | +0.099 → +0.118 |
| RETRY | K ≥ 2 | Master | 0 | 2 · 100 % · +3.00 | – | – | – | +1.147 → +1.147 |
| RETRY | K ≥ 2 | B25 | 23 · 48 % · −0.13 · −3.0 · 11 d | 14 · 57 % · +0.42 · +5.9 · 10 d | −0.56 | −0.82 … −0.36 | 0.78 | +0.099 → +0.304 |

## Verdicts (rule: the predicted sign on avg % holds on master AND B25, with ≥ 5 fills on each side)
| Sleeve | E (low energy is worse) | K (later fills are worse) |
|---|---|---|
| **FRENZY_LONG** (primary) | **CONFIRMED by the rule, but weak.** Master diff −0.05 (LOO crosses 0, P 0.52); B25 −1.61, P 0.87 | **CONFIRMED by the rule.** Master −0.32 and B25 −1.02, LOO never crosses 0, P 0.69 / 0.77 |
| FRENZY_LITE | **CONFIRMED by the rule.** −0.34 / −0.89, LOO never crosses 0, P 0.58 / 0.73 | **REFUTED.** On B25, K ≥ 2 is far better (+2.92, P 0.00) |
| FRENZY_LITE_RETRY | **TOO FEW** (master N = 2) | **TOO FEW** (master N = 2) |

How strong the confirmations are:
- **No confirmation comes close to the filter bar**: every P is < 0.95, no confirmed cut has ≥ 15 bad-side fills in BOTH blocks, and B25 is the batch the theory was drawn from (FRENZY_LONG review §3), so its leg is not independent — the independent reads are master (E −0.05 ≈ 0; K −0.32) and the replay (K opposite in both years).
- **Blocking does not improve the daily compound in both blocks.**
  - FRENZY_LONG E and K: blocking costs master (+0.061 → +0.035 / +0.022) and only lifts B25.
  - LITE E: blocking costs B25 (+1.910 → +1.481), because its low-E side is still profitable.
- Per-pair check (FRENZY_LONG B25 low-E losers): BMT carries −8.0 % of the −24.1 % loss (33 %), so no single pair dominates.

## Engine-replay cross-check (YR6W 2Y TODAY, seed 1, in-window, no duplicates; never a veto)
E was rebuilt from the cached 5m klines and the replay's `entry_frenzy_spike_at`. K = replay fills of every sleeve on the pair since the spike. RETRY is not in the replay.

| Sleeve | E low vs high (live cut) | K ≥ 2 vs K = 1 |
|---|---|---|
| FRENZY_LONG all (623) | 272 · 46 % · −0.22 vs 351 · 48 % · −0.14 (P 0.63) | 361 · 49 % · −0.06 vs 262 · 45 % · −0.33 (**opposite**, P 0.13) |
| FRENZY_LONG 2025 / 2026 | +0.20 vs −0.10 (opposite) / −0.51 vs −0.18 (predicted) | +0.16 vs −0.21 / −0.26 vs −0.41 (both opposite) |
| FRENZY_LITE all (1,455) | 725 · 48 % · +0.02 vs 730 · 49 % · −0.03 (flat, P 0.38) | 728 · 49 % · −0.03 vs 727 · 48 % · +0.01 (flat, P 0.60) |

The backtest does not support K for FRENZY_LONG; it says the opposite in both years. For E it splits by regime: the predicted way in 2026, the opposite way in 2025.

## EXPLORATORY (not part of the verdict; looked at after the confirmatory read)
- **E is not monotonic on FRENZY_LONG master.** By tercile (low / mid / high): 8 · 100 % · **+1.50** / 11 · 82 % · +0.50 / 10 · 80 % · +0.40.
  The lowest-energy third is the best one there. On B25 the terciles line up as predicted: −2.43 / −1.25 / −0.33.
- **E is partly episode age.** The rank correlation of E with hours since the spike is −0.40 to −0.54 for FRENZY_LONG / RETRY (−0.23 / −0.30 for LITE). Hours were already tested as H3 LATE in the review.
- **The same split on absolute ATR** (price-scaled), FRENZY_LONG: master 14 · +1.11 vs 15 · +0.40 (opposite); B25 9 · −2.79 vs 8 · +0.13 (predicted). The sign depends on the ATR form, which is a fragility flag.
- **Volume now ÷ volume at the spike** (last-hour quote volume, median split):
  - FRENZY_LONG: flat in both blocks (+0.77 vs +0.71; −1.25 vs −1.56).
  - LITE: opposite between the blocks (master low = −0.72, B25 low = +1.17).
- **K is not monotonic.**
  - FRENZY_LONG master: K1 +0.95 · K2 +0.40 · K3+ +0.89.
  - LITE B25: K1 −1.81 · K2 +1.94 · K3+ −0.76. Most of LITE's K = 2 fills follow a WILLY fill on the same spike.
  - RETRY B25: K3+ 7 · +1.28 is the best bucket.

## Read
- The "dying frenzy" story explains B25's FRENZY_LONG losers well: low E and K ≥ 2 hold almost all of the −24 %.
- Master does not repeat it in size. The master E effect is −0.05 and goes the opposite way on terciles.
- The 2-year replay is flat or opposite.
- This matches a batch-specific pattern (one bad stretch of tired episodes around 10-29 → 11-06, mostly 11-02 → 11-06) more than a durable separator.
- Nothing here is a ship candidate. A possible observe line for FRENZY_LONG (E < 1.285 ∧ K ≥ 2, cuts frozen) — NOT registered; the replay contradicts it.

Note: the FRENZY_LONG daily-compound baseline here (−0.065 / +0.061) differs from the review's (−0.067 / +0.062) only by the B25 day span (export time vs last close).

## Are live stamps worth building?
**Not now.** Both measures rebuild exactly from data we already have:
- E: the spike time is already stamped (`entry_frenzy_spike_at`); the rest is public klines plus the engine's own ATR and walk, which matched the stamp on 97 % of fills.
- K: the order history.

A scout script could therefore track it with no engine change. That also avoids the D13 stamp class work for an unproven signal. Build a stamp only if the observe line clears the gates at a later review.

Scripts and intermediate files: `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/dying/`
(`load.py`, `fetch.py`, `measure.py`, `stats.py`, `replay.py`, `explore.py`; `D.pkl` holds the per-fill E / K).
Caveat: master K can be undercounted when an earlier fill opened during a gap between archived batches.
