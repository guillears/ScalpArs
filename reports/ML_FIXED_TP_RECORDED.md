# Momentum longs — fixed take-profits from the RECORDED trade data (observe-only)

Full-size fills only: **master 158** (kept on today's stack; 166 probe fills excluded) · **B25 34** (`scalpars_orders_paper_2026-11-05_19-52-51.csv`, every momentum long since 2026-10-23 18:00; 0 probe fills excluded). Before = the fill's booked % at 1× (master: stack_pct). After = the same fill with a plain fixed TP, read from its own recorded peak and the 45 minutes the bot recorded after its exit (rule in the script header; stop -0.70 %). No replica, no klines.

## MASTER (all archived batches)

| Exit | N | Avg % before → after | WR before → after | Σ % before → after | Better / worse | TP hit in trade · after exit | New stops | Open at 45 min | Unpriced | p (luck, this TP) | p after 4-TP search | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed TP +0.5 % | 156 | +0.12 → **+0.02** | 71 % → 62 % | +19.2 → +2.4 | 51 / 54 | 79 · 12 | 9 | 6 | 2 | 1.00 | 1.00 | NOT ABOVE LUCK |
| Fixed TP +1 % | 156 | +0.12 → **+0.13** | 71 % → 54 % | +19.2 → +20.0 | 60 / 42 | 41 · 25 | 15 | 25 | 2 | 0.45 | 0.76 | NOT ABOVE LUCK |
| Fixed TP +2 % | 156 | +0.12 → **+0.14** | 71 % → 46 % | +19.2 → +21.2 | 50 / 54 | 6 · 16 | 19 | 64 | 2 | 0.42 | 0.70 | NOT ABOVE LUCK |
| Fixed TP +3 % | 156 | +0.12 → **+0.15** | 71 % → 46 % | +19.2 → +23.6 | 51 / 53 | 2 · 6 | 19 | 78 | 2 | 0.31 | 0.55 | NOT ABOVE LUCK |

- TP +0.5 %: by fill Δ -16.74 pts · without the biggest (LITUSDT 10-05 -2.04, 12% of Δ): -14.70 pts · by DAY Δ -16.74 pts · without the biggest (2026-10-05 -4.38, 26% of Δ): -12.36 pts
- TP +1 %: by fill Δ +0.78 pts · without the biggest (FETUSDT 09-25 +0.94, 120% of Δ): -0.16 pts ⚠ one fill carries ≥ 50 % of the result · by DAY Δ +0.78 pts · without the biggest (2026-10-16 +2.75, 355% of Δ): -1.98 pts ⚠ one fill carries ≥ 50 % of the result
- TP +2 %: by fill Δ +2.05 pts · without the biggest (FETUSDT 09-25 +1.94, 94% of Δ): +0.12 pts ⚠ one fill carries ≥ 50 % of the result · by DAY Δ +2.05 pts · without the biggest (2026-09-25 +3.18, 155% of Δ): -1.13 pts ⚠ one fill carries ≥ 50 % of the result
- TP +3 %: by fill Δ +4.44 pts · without the biggest (CRVUSDT 08-21 +2.91, 66% of Δ): +1.52 pts ⚠ one fill carries ≥ 50 % of the result · by DAY Δ +4.44 pts · without the biggest (2026-09-25 +3.91, 88% of Δ): +0.53 pts ⚠ one fill carries ≥ 50 % of the result

## B25 (current batch, every fill)

| Exit | N | Avg % before → after | WR before → after | Σ % before → after | Better / worse | TP hit in trade · after exit | New stops | Open at 45 min | Unpriced | p (luck, this TP) | p after 4-TP search | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed TP +0.5 % | 33 | +0.17 → **+0.02** | 67 % → 61 % | +5.6 → +0.7 | 9 / 8 | 16 · 1 | 0 | 2 | 1 | 0.66 | 0.97 | NOT ABOVE LUCK |
| Fixed TP +1 % | 33 | +0.17 → **+0.21** | 67 % → 58 % | +5.6 → +6.8 | 13 / 5 | 5 · 9 | 0 | 4 | 1 | 0.39 | 0.70 | NOT ABOVE LUCK |
| Fixed TP +2 % | 33 | +0.17 → **+0.32** | 67 % → 58 % | +5.6 → +10.7 | 14 / 4 | 2 · 1 | 0 | 15 | 1 | 0.12 | 0.45 | NOT ABOVE LUCK |
| Fixed TP +3 % | 33 | +0.17 → **+0.42** | 67 % → 58 % | +5.6 → +13.7 | 14 / 4 | 1 · 2 | 0 | 15 | 1 | 0.02 | 0.06 | UNCLEAR |

- TP +0.5 %: by fill Δ -4.89 pts · without the biggest (ARBUSDT 10-30 -4.16, 85% of Δ): -0.73 pts ⚠ one fill carries ≥ 50 % of the result · by DAY Δ -4.89 pts · without the biggest (2026-10-30 -5.42, 111% of Δ): +0.54 pts ⚠ one fill carries ≥ 50 % of the result
- TP +1 %: by fill Δ +1.27 pts · without the biggest (INJUSDT 10-29 +0.90, 71% of Δ): +0.36 pts ⚠ one fill carries ≥ 50 % of the result · by DAY Δ +1.27 pts · without the biggest (2026-11-04 +2.40, 189% of Δ): -1.13 pts ⚠ one fill carries ≥ 50 % of the result
- TP +2 %: by fill Δ +5.14 pts · without the biggest (LIGHTUSDT 10-26 +1.18, 23% of Δ): +3.97 pts · by DAY Δ +5.14 pts · without the biggest (2026-11-04 +2.88, 56% of Δ): +2.26 pts ⚠ one fill carries ≥ 50 % of the result
- TP +3 %: by fill Δ +8.14 pts · without the biggest (LIGHTUSDT 10-26 +2.18, 27% of Δ): +5.97 pts · by DAY Δ +8.14 pts · without the biggest (2026-11-04 +2.88, 35% of Δ): +5.26 pts

## ALL

| Exit | N | Avg % before → after | WR before → after | Σ % before → after | Better / worse | TP hit in trade · after exit | New stops | Open at 45 min | Unpriced | p (luck, this TP) | p after 4-TP search | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fixed TP +0.5 % | 189 | +0.13 → **+0.02** | 70 % → 62 % | +24.8 → +3.1 | 60 / 62 | 95 · 13 | 9 | 8 | 3 | 1.00 | 1.00 | NOT ABOVE LUCK |
| Fixed TP +1 % | 189 | +0.13 → **+0.14** | 70 % → 54 % | +24.8 → +26.8 | 73 / 47 | 46 · 34 | 15 | 29 | 3 | 0.40 | 0.71 | NOT ABOVE LUCK |
| Fixed TP +2 % | 189 | +0.13 → **+0.17** | 70 % → 48 % | +24.8 → +32.0 | 64 / 58 | 8 · 17 | 19 | 79 | 3 | 0.23 | 0.46 | NOT ABOVE LUCK |
| Fixed TP +3 % | 189 | +0.13 → **+0.20** | 70 % → 48 % | +24.8 → +37.3 | 65 / 57 | 3 · 8 | 19 | 93 | 3 | 0.11 | 0.22 | NOT ABOVE LUCK |

- TP +0.5 %: by fill Δ -21.63 pts · without the biggest (ARBUSDT 10-30 -4.16, 19% of Δ): -17.47 pts · by DAY Δ -21.63 pts · without the biggest (2026-10-30 -5.42, 25% of Δ): -16.20 pts
- TP +1 %: by fill Δ +2.04 pts · without the biggest (FETUSDT 09-25 +0.94, 46% of Δ): +1.11 pts · by DAY Δ +2.04 pts · without the biggest (2026-10-16 +2.75, 135% of Δ): -0.71 pts ⚠ one fill carries ≥ 50 % of the result
- TP +2 %: by fill Δ +7.19 pts · without the biggest (FETUSDT 09-25 +1.94, 27% of Δ): +5.26 pts · by DAY Δ +7.19 pts · without the biggest (2026-09-25 +3.18, 44% of Δ): +4.02 pts
- TP +3 %: by fill Δ +12.58 pts · without the biggest (CRVUSDT 08-21 +2.91, 23% of Δ): +9.67 pts · by DAY Δ +12.58 pts · without the biggest (2026-09-25 +3.91, 31% of Δ): +8.68 pts

## Where the change comes from — Σ Δ % by what happened (count)

| Block | TP | TP in trade (cuts a bigger win) | TP after the real exit | New stop after the exit | Still open at 45 min (marked) |
|---|---|---|---|---|---|
| MASTER | +0.5 % | -12.0 (79) | +4.0 (12) | -7.2 (9) | -1.5 (6) |
| MASTER | +1 % | +0.2 (41) | +18.8 (25) | -12.8 (15) | -4.7 (25) |
| MASTER | +2 % | +0.3 (6) | +21.3 (16) | -19.0 (19) | -0.7 (64) |
| MASTER | +3 % | +1.0 (2) | +11.9 (6) | -19.0 (19) | +10.4 (78) |
| B25 | +0.5 % | -4.2 (16) | +0.4 (1) | +0.0 (0) | -0.7 (2) |
| B25 | +1 % | -3.5 (5) | +5.7 (9) | +0.0 (0) | -1.0 (4) |
| B25 | +2 % | -2.3 (2) | +1.2 (1) | +0.0 (0) | +6.2 (15) |
| B25 | +3 % | -1.7 (1) | +3.6 (2) | +0.0 (0) | +6.2 (15) |

A gain in the last column is not a take-profit: it is the trade held 45 min longer than today's exit, marked where it stood.

## Entry filter ML_MKT_HOT (watch line, DL 416) × fixed TP — master and B25, before → after

Kept = fills NOT in the hot zone (≤ 50 % of 108 coins above their 4 h-earlier close at entry) — the fills the candidate filter would let through. Blocked = the hot-zone fills it would refuse. Unscored fills (no market data) are in neither.

### MASTER (all archived batches) · 0 unscored

| Cohort · exit | N | Avg % | WR | Σ % | p (luck vs today's exit, kept fills) | Verdict |
|---|---|---|---|---|---|---|
| All fills · today's exit | 158 | +0.12 | 71 % | +19.7 | — |  |
| Blocked by MKT_HOT (hot) · today's exit | 73 | +0.00 | 66 % | +0.0 | — |  |
| Kept by MKT_HOT · today's exit | 85 | +0.23 | 75 % | +19.6 | — |  |
| Kept by MKT_HOT · fixed TP +0.5 % | 85 | +0.04 (today +0.23 on these) | 65 % | +3.8 | 1.00 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +1 % | 85 | +0.21 (today +0.23 on these) | 60 % | +17.8 | 0.64 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +2 % | 85 | +0.17 (today +0.23 on these) | 49 % | +14.8 | 0.76 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +3 % | 85 | +0.18 (today +0.23 on these) | 49 % | +15.5 | 0.74 | NOT ABOVE LUCK |

### B25 (current batch, every fill) · 1 unscored

| Cohort · exit | N | Avg % | WR | Σ % | p (luck vs today's exit, kept fills) | Verdict |
|---|---|---|---|---|---|---|
| All fills · today's exit | 34 | +0.17 | 68 % | +5.7 | — |  |
| Blocked by MKT_HOT (hot) · today's exit | 13 | -0.24 | 46 % | -3.1 | — |  |
| Kept by MKT_HOT · today's exit | 20 | +0.43 | 80 % | +8.7 | — |  |
| Kept by MKT_HOT · fixed TP +0.5 % | 20 | +0.17 (today +0.43 on these) | 75 % | +3.4 | 0.75 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +1 % | 20 | +0.44 (today +0.43 on these) | 70 % | +8.8 | 0.51 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +2 % | 20 | +0.66 (today +0.43 on these) | 70 % | +13.1 | 0.16 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +3 % | 20 | +0.81 (today +0.43 on these) | 70 % | +16.1 | 0.00 | REAL |

### ALL · 1 unscored

| Cohort · exit | N | Avg % | WR | Σ % | p (luck vs today's exit, kept fills) | Verdict |
|---|---|---|---|---|---|---|
| All fills · today's exit | 192 | +0.13 | 70 % | +25.3 | — |  |
| Blocked by MKT_HOT (hot) · today's exit | 86 | -0.04 | 63 % | -3.1 | — |  |
| Kept by MKT_HOT · today's exit | 105 | +0.27 | 76 % | +28.3 | — |  |
| Kept by MKT_HOT · fixed TP +0.5 % | 105 | +0.07 (today +0.27 on these) | 67 % | +7.2 | 1.00 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +1 % | 105 | +0.25 (today +0.27 on these) | 62 % | +26.6 | 0.59 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +2 % | 105 | +0.27 (today +0.27 on these) | 53 % | +27.9 | 0.51 | NOT ABOVE LUCK |
| Kept by MKT_HOT · fixed TP +3 % | 105 | +0.30 (today +0.27 on these) | 53 % | +31.6 | 0.32 | NOT ABOVE LUCK |

B25 daily compound with the filter (blocked fills never opened) — momentum-long sleeve, real account path:

| | All · today | Kept · today | Kept · TP +0.5 | Kept · TP +1 | Kept · TP +2 | Kept · TP +3 |
|---|---|---|---|---|---|---|
| DCR-cal %/day | +0.159 | +0.207 | +0.073 | +0.195 | +0.305 | +0.375 |

## Per batch — Σ % at 1× before → after

| Batch | N | Today | TP +0.5 | TP +1 | TP +2 | TP +3 |
|---|---|---|---|---|---|---|
| BASE | 31 | +15.5 | +9.3 | +13.6 | +12.9 | +12.8 |
| B1 | 15 | +3.8 | +0.7 | +3.1 | +5.3 | +5.6 |
| B2 | 9 | +1.5 | -0.7 | +1.3 | +1.0 | +1.0 |
| B3 | 22 | +3.2 | +3.7 | +8.6 | +3.9 | +4.9 |
| B4 | 1 | +0.1 | -0.7 | -0.7 | -0.7 | -0.7 |
| B5 | 3 | +1.4 | +1.0 | +2.0 | +1.6 | +1.6 |
| B6 | 2 | -1.8 | -1.8 | -1.8 | -1.8 | -1.8 |
| B7 | 1 | +0.1 | -0.7 | -0.7 | -0.7 | -0.7 |
| B8 | 4 | +0.5 | +0.8 | +1.5 | +2.5 | +1.9 |
| B9 | 2 | +0.5 | -0.2 | +0.3 | +1.3 | +1.6 |
| B10 | 1 | +0.1 | +0.5 | +1.0 | +1.1 | +1.1 |
| B12 | 15 | +1.8 | +0.9 | +4.7 | +8.1 | +6.9 |
| B13 | 3 | +0.4 | +0.3 | +0.4 | +1.4 | +1.1 |
| B14 | 4 | -2.6 | -2.4 | -3.6 | -3.6 | -3.6 |
| B15 | 1 | -1.0 | -1.0 | -1.0 | -1.0 | -1.0 |
| B16 | 5 | -0.9 | -0.3 | -3.9 | -3.9 | -3.9 |
| B17 | 6 | +3.9 | -0.5 | +1.0 | +2.0 | +4.0 |
| B18 | 3 | -2.1 | -2.1 | -2.1 | -2.1 | -2.1 |
| B19 | 4 | -1.8 | -1.6 | -2.1 | -2.1 | -2.1 |
| B20 | 2 | -1.4 | -1.4 | -1.4 | -1.4 | -1.4 |
| B21 | 3 | -2.1 | -2.1 | -2.1 | -2.1 | -2.1 |
| B22 | 6 | +1.0 | +0.5 | +2.5 | +0.1 | +1.1 |
| B23 | 13 | -0.9 | +0.1 | -0.6 | -0.7 | -0.7 |
| B25 | 33 | +5.6 | +0.7 | +6.8 | +10.7 | +13.7 |

## B25 daily compound (momentum-long sleeve, real account path)

| | Today | TP +0.5 | TP +1 | TP +2 | TP +3 |
|---|---|---|---|---|---|
| DCR-cal %/day | +0.159 | +0.040 | +0.168 | +0.270 | +0.340 |

Unpriced fills (no post-exit record) stay as booked in the DCR. 
Master DCR is not shown: master momentum longs are a few fills a day spread over 128 days (the per-batch Σ above carries it).

**Blind spots:** ① the 45-min post-exit window — a trade still open at 45 min is marked there, not followed to its real end · ② post_exit_peak/trough are the extremes, not first-touch times: when both the TP and the stop were reached after the exit, the stop is booked (order unknown) · ③ in-trade peak_pnl ≥ TP books the TP — unless the fill fell to today's −0.70 stop BEFORE its peak (older, wider stops), then it is compared as booked · ④ master exits ran under older exit rules: their post-exit window starts at THAT exit, and 'same exit' losses keep their old-era stop (some below −0.70) while new stops use −0.70 · ⑤ the 180-min no-expansion exit is kept as booked · ⑥ a gain in 'still open at 45 min' is holding longer, not a TP — read the by-DAY leave-one-out before trusting it.
