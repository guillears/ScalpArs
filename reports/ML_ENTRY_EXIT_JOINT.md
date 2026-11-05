# Momentum longs — entry × exit joint read (observe-only)

Cohort: master kept on today's stack + every B25 momentum-long fill of `scalpars_orders_paper_2026-11-05_19-52-51.csv` · 192 fills (158 master · 34 B25) · priced on 192. Δ = exit − today's exit, both through the validated replica, % at 1× per fill (fees in). **72 entry × exit combinations were looked at** — the luck p after the exit search counts the exits of a row only; across rows, expect ~1 in 20 to show p ≤ 0.05 by chance.

Entry-line coverage (unscored fills sit in no 'kept by' row of that line — unscored ≠ kept): ML_MKT_HOT scored 191 (in zone 86) · ML_W4 scored 192 (in zone 20) · STOP_COOLDOWN scored 191 (in zone 9) · PAIR_1H_DOWNTREND scored 77 (in zone 58).

Exits: `W_LAD_off` = OBS-1 ladder off · `W_LOCK0.2` = OBS-2 lock +0.20 · `W_TW30_k1.5` = OBS-3 trail ×1.5 after 30 min · `W_K1.25` = OBS-3 trail ×1.25 · `W_LAD_loose` = OBS-3 looser ladder · `FTP0.5` = fixed TP +0.5 % · `FTP1` = fixed TP +1 % · `FTP2` = fixed TP +2 % · `FTP3` = fixed TP +3 % · `FTP1_R` = TP +1 % + runner · `FTP2_R` = TP +2 % + runner · `FTP3_R` = TP +3 % + runner. Fixed TP = a take-profit at +X % net with today's stop with the 180-min clock and recovery hold as today (runner and ladder off), booked AT +X; `_R` keeps today's runner trail with the ladder off and the TP as a cap.


## ALL fills — 192 fills on 83 days · today's exit +0.14 % avg

| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |
|---|---|---|---|---|---|---|
| fixed TP +2 % | +0.17 | +6.0 (-0.1 · +6.1) | 34 % / 35 % | 0.351 | 0.718 | NOT ABOVE LUCK |
| OBS-3 trail ×1.5 after 30 min | +0.17 | +5.5 (+0.8 · +4.7) | 4 % / 10 % | 0.146 | 0.757 | NOT ABOVE LUCK |
| OBS-1 ladder off | +0.16 | +3.3 (+3.3 · +0.0) | 2 % / 7 % | 0.292 | 0.903 | NOT ABOVE LUCK |
| OBS-3 trail ×1.25 | +0.15 | +1.8 (+0.3 · +1.5) | 7 % / 30 % | 0.351 | 0.956 | NOT ABOVE LUCK |
| fixed TP +1 % | +0.15 | +1.4 (+1.7 · -0.3) | 38 % / 27 % | 0.466 | 0.971 | NOT ABOVE LUCK |
| TP +3 % + runner | +0.15 | +1.2 (+2.5 · -1.3) | 3 % / 8 % | 0.378 | 0.974 | NOT ABOVE LUCK |
| OBS-3 looser ladder | +0.14 | +0.5 (+0.5 · +0.0) | 2 % / 7 % | 0.429 | 0.988 | NOT ABOVE LUCK |
| TP +2 % + runner | +0.14 | -0.6 (+2.4 · -2.9) | 5 % / 7 % | 0.522 | 0.998 | NOT ABOVE LUCK |
| OBS-2 lock +0.20 | +0.14 | -0.8 (-1.2 · +0.5) | 23 % / 4 % | 0.651 | 0.998 | NOT ABOVE LUCK |
| TP +1 % + runner | +0.12 | -3.8 (+1.5 · -5.3) | 14 % / 9 % | 0.685 | 1.000 | NOT ABOVE LUCK |
| fixed TP +3 % | +0.08 | -11.0 (-17.8 · +6.8) | 30 % / 39 % | 0.771 | 1.000 | NOT ABOVE LUCK |
| fixed TP +0.5 % | +0.03 | -21.6 (-15.4 · -6.2) | 36 % / 32 % | 0.994 | 1.000 | NOT ABOVE LUCK |

## kept by ML_MKT_HOT — 105 fills on 57 days · today's exit +0.30 % avg

| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |
|---|---|---|---|---|---|---|
| fixed TP +2 % | +0.34 | +4.0 (-2.4 · +6.4) | 39 % / 37 % | 0.366 | 0.744 | NOT ABOVE LUCK |
| OBS-3 trail ×1.5 after 30 min | +0.33 | +3.5 (-1.2 · +4.7) | 7 % / 14 % | 0.171 | 0.784 | NOT ABOVE LUCK |
| OBS-3 trail ×1.25 | +0.31 | +0.9 (-1.0 · +1.8) | 10 % / 32 % | 0.367 | 0.993 | NOT ABOVE LUCK |
| OBS-3 looser ladder | +0.30 | +0.4 (+0.4 · +0.0) | 2 % / 8 % | 0.474 | 0.998 | NOT ABOVE LUCK |
| OBS-1 ladder off | +0.30 | +0.4 (+0.4 · +0.0) | 2 % / 8 % | 0.475 | 0.998 | NOT ABOVE LUCK |
| TP +3 % + runner | +0.30 | +0.1 (+1.4 · -1.3) | 4 % / 9 % | 0.490 | 1.000 | NOT ABOVE LUCK |
| OBS-2 lock +0.20 | +0.29 | -1.1 (-1.3 · +0.2) | 24 % / 5 % | 0.734 | 1.000 | NOT ABOVE LUCK |
| TP +2 % + runner | +0.28 | -1.7 (+1.2 · -2.9) | 8 % / 9 % | 0.594 | 1.000 | NOT ABOVE LUCK |
| fixed TP +3 % | +0.25 | -5.6 (-12.7 · +7.1) | 36 % / 40 % | 0.675 | 1.000 | NOT ABOVE LUCK |
| fixed TP +1 % | +0.25 | -5.6 (-4.9 · -0.7) | 41 % / 32 % | 0.733 | 1.000 | NOT ABOVE LUCK |
| TP +1 % + runner | +0.23 | -7.2 (-1.9 · -5.3) | 16 % / 13 % | 0.877 | 1.000 | NOT ABOVE LUCK |
| fixed TP +0.5 % | +0.10 | -21.4 (-14.9 · -6.5) | 34 % / 43 % | 0.998 | 1.000 | NOT ABOVE LUCK |

## kept by ML_W4 — 172 fills on 80 days · today's exit +0.17 % avg

| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |
|---|---|---|---|---|---|---|
| fixed TP +2 % | +0.21 | +7.1 (+1.0 · +6.1) | 36 % / 36 % | 0.321 | 0.601 | NOT ABOVE LUCK |
| OBS-3 trail ×1.5 after 30 min | +0.21 | +6.3 (+1.6 · +4.7) | 5 % / 9 % | 0.073 | 0.662 | NOT ABOVE LUCK |
| OBS-1 ladder off | +0.19 | +3.3 (+3.3 · +0.0) | 2 % / 8 % | 0.278 | 0.890 | NOT ABOVE LUCK |
| fixed TP +1 % | +0.19 | +2.7 (+3.0 · -0.3) | 40 % / 28 % | 0.407 | 0.913 | NOT ABOVE LUCK |
| OBS-3 trail ×1.25 | +0.18 | +2.6 (+1.1 · +1.5) | 8 % / 29 % | 0.266 | 0.915 | NOT ABOVE LUCK |
| TP +3 % + runner | +0.18 | +1.2 (+2.5 · -1.3) | 3 % / 9 % | 0.371 | 0.970 | NOT ABOVE LUCK |
| OBS-3 looser ladder | +0.17 | +0.5 (+0.5 · +0.0) | 2 % / 8 % | 0.414 | 0.985 | NOT ABOVE LUCK |
| TP +2 % + runner | +0.17 | -0.6 (+2.4 · -2.9) | 6 % / 8 % | 0.541 | 0.996 | NOT ABOVE LUCK |
| OBS-2 lock +0.20 | +0.16 | -0.8 (-1.3 · +0.5) | 26 % / 5 % | 0.672 | 0.997 | NOT ABOVE LUCK |
| TP +1 % + runner | +0.14 | -4.9 (+0.4 · -5.3) | 13 % / 9 % | 0.749 | 1.000 | NOT ABOVE LUCK |
| fixed TP +3 % | +0.09 | -13.0 (-19.8 · +6.8) | 31 % / 40 % | 0.813 | 1.000 | NOT ABOVE LUCK |
| fixed TP +0.5 % | +0.06 | -19.6 (-13.5 · -6.2) | 38 % / 34 % | 0.989 | 1.000 | NOT ABOVE LUCK |

## kept by STOP_COOLDOWN — 182 fills on 82 days · today's exit +0.15 % avg

| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |
|---|---|---|---|---|---|---|
| fixed TP +2 % | +0.19 | +8.2 (+1.3 · +7.0) | 35 % / 34 % | 0.296 | 0.555 | NOT ABOVE LUCK |
| OBS-3 trail ×1.5 after 30 min | +0.18 | +5.9 (+1.2 · +4.7) | 4 % / 9 % | 0.126 | 0.723 | NOT ABOVE LUCK |
| OBS-1 ladder off | +0.17 | +3.3 (+3.3 · +0.0) | 2 % / 8 % | 0.292 | 0.898 | NOT ABOVE LUCK |
| OBS-3 trail ×1.25 | +0.16 | +1.9 (+0.4 · +1.5) | 7 % / 30 % | 0.340 | 0.948 | NOT ABOVE LUCK |
| TP +3 % + runner | +0.15 | +1.2 (+2.5 · -1.3) | 3 % / 8 % | 0.378 | 0.975 | NOT ABOVE LUCK |
| OBS-3 looser ladder | +0.15 | +0.5 (+0.5 · +0.0) | 2 % / 7 % | 0.429 | 0.989 | NOT ABOVE LUCK |
| TP +2 % + runner | +0.15 | -0.6 (+2.4 · -2.9) | 5 % / 8 % | 0.522 | 0.998 | NOT ABOVE LUCK |
| OBS-2 lock +0.20 | +0.14 | -1.0 (-1.2 · +0.3) | 24 % / 4 % | 0.679 | 0.998 | NOT ABOVE LUCK |
| fixed TP +1 % | +0.14 | -1.1 (+0.2 · -1.3) | 37 % / 28 % | 0.553 | 0.998 | NOT ABOVE LUCK |
| TP +1 % + runner | +0.12 | -4.9 (+0.4 · -5.3) | 14 % / 9 % | 0.740 | 1.000 | NOT ABOVE LUCK |
| fixed TP +3 % | +0.12 | -5.6 (-13.3 · +7.7) | 31 % / 38 % | 0.657 | 1.000 | NOT ABOVE LUCK |
| fixed TP +0.5 % | +0.02 | -23.2 (-16.1 · -7.0) | 35 % / 34 % | 0.996 | 1.000 | NOT ABOVE LUCK |

## kept by PAIR_1H_DOWNTREND — 19 fills on 11 days · today's exit -0.19 % avg

| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |
|---|---|---|---|---|---|---|
| fixed TP +1 % | -0.14 | +0.9 (+1.5 · -0.6) | 26 % / 21 % | 0.428 | 0.683 | NOT ABOVE LUCK |
| TP +2 % + runner | -0.15 | +0.7 (+0.4 · +0.4) | 11 % / 5 % | 0.244 | 0.877 | NOT ABOVE LUCK |
| fixed TP +2 % | -0.17 | +0.4 (+0.1 · +0.4) | 26 % / 21 % | 0.375 | 1.000 | NOT ABOVE LUCK |
| OBS-2 lock +0.20 | -0.18 | +0.2 (+0.2 · +0.0) | 16 % / 0 % | 0.121 | 1.000 | NOT ABOVE LUCK |
| OBS-3 trail ×1.25 | -0.18 | +0.2 (-0.3 · +0.5) | 5 % / 11 % | 0.497 | 1.000 | NOT ABOVE LUCK |
| OBS-3 trail ×1.5 after 30 min | -0.19 | +0.0 (+0.0 · +0.0) | 0 % / 0 % | 1.000 | 1.000 | NOT ABOVE LUCK |
| OBS-1 ladder off | -0.19 | -0.1 (-0.1 · +0.0) | 0 % / 11 % | 1.000 | 1.000 | NOT ABOVE LUCK |
| OBS-3 looser ladder | -0.19 | -0.1 (-0.1 · +0.0) | 0 % / 11 % | 1.000 | 1.000 | NOT ABOVE LUCK |
| TP +3 % + runner | -0.19 | -0.1 (-0.1 · +0.0) | 0 % / 11 % | 1.000 | 1.000 | NOT ABOVE LUCK |
| TP +1 % + runner | -0.21 | -0.5 (+0.1 · -0.6) | 11 % / 11 % | 0.619 | 1.000 | NOT ABOVE LUCK |
| fixed TP +3 % | -0.23 | -0.7 (-2.1 · +1.4) | 21 % / 26 % | 0.557 | 1.000 | NOT ABOVE LUCK |
| fixed TP +0.5 % | -0.26 | -1.4 (-0.3 · -1.1) | 26 % / 21 % | 0.685 | 1.000 | NOT ABOVE LUCK |

## kept by EVERY line — 10 fills on 5 days · today's exit +0.21 % avg

| Exit | Avg % | Σ Δ (master · B25) | Better / worse | p alone | p after exit search | Verdict |
|---|---|---|---|---|---|---|
| fixed TP +1 % | +0.39 | +1.8 (+2.4 · -0.6) | 50 % / 30 % | — | — | TOO FEW |
| fixed TP +2 % | +0.34 | +1.3 (+1.0 · +0.4) | 50 % / 30 % | — | — | TOO FEW |
| TP +2 % + runner | +0.28 | +0.7 (+0.4 · +0.4) | 20 % / 10 % | — | — | TOO FEW |
| OBS-3 trail ×1.25 | +0.24 | +0.3 (-0.2 · +0.5) | 10 % / 10 % | — | — | TOO FEW |
| fixed TP +3 % | +0.23 | +0.2 (-1.2 · +1.4) | 40 % / 40 % | — | — | TOO FEW |
| OBS-2 lock +0.20 | +0.23 | +0.2 (+0.2 · +0.0) | 20 % / 0 % | — | — | TOO FEW |
| OBS-3 trail ×1.5 after 30 min | +0.21 | +0.0 (+0.0 · +0.0) | 0 % / 0 % | — | — | TOO FEW |
| OBS-1 ladder off | +0.20 | -0.1 (-0.1 · +0.0) | 0 % / 20 % | — | — | TOO FEW |
| OBS-3 looser ladder | +0.20 | -0.1 (-0.1 · +0.0) | 0 % / 20 % | — | — | TOO FEW |
| TP +3 % + runner | +0.20 | -0.1 (-0.1 · +0.0) | 0 % / 20 % | — | — | TOO FEW |
| TP +1 % + runner | +0.16 | -0.5 (+0.1 · -0.6) | 20 % / 20 % | — | — | TOO FEW |
| fixed TP +0.5 % | +0.04 | -1.7 (-0.6 · -1.1) | 40 % / 40 % | — | — | TOO FEW |

## Path quality — ALL fills (read before trusting a gain)

Fills newly stopped = stopped under the exit but not under today's exit (a 1m-bar stop may be a wick live polling rode through — CLAUDE.md stop rule). Σ Δ split by path: real ticks vs 1m bars (1m resolves a bar's high / low order by its colour, so it can over- or under-state either side).

| Exit | Newly stopped (tick · 1m) | Σ Δ on them | Σ Δ tick paths (n) | Σ Δ 1m paths (n) |
|---|---|---|---|---|
| OBS-1 ladder off | 0 (0 · 0) | +0.0 | +3.8 (138) | -0.6 (54) |
| OBS-2 lock +0.20 | 0 (0 · 0) | +0.0 | -1.7 (138) | +1.0 (54) |
| OBS-3 trail ×1.5 after 30 min | 0 (0 · 0) | +0.0 | +0.5 (138) | +5.0 (54) |
| OBS-3 trail ×1.25 | 0 (0 · 0) | +0.0 | +0.3 (138) | +1.5 (54) |
| OBS-3 looser ladder | 0 (0 · 0) | +0.0 | +1.1 (138) | -0.6 (54) |
| fixed TP +0.5 % | 13 (11 · 2) | -10.6 | -17.8 (138) | -3.8 (54) |
| fixed TP +1 % | 31 (25 · 6) | -28.6 | -1.5 (138) | +2.8 (53) |
| fixed TP +2 % | 51 (42 · 9) | -55.5 | -1.3 (138) | +7.3 (53) |
| fixed TP +3 % | 60 (50 · 10) | -73.2 | -19.6 (138) | +8.6 (53) |
| TP +1 % + runner | 0 (0 · 0) | +0.0 | +1.2 (138) | -5.0 (54) |
| TP +2 % + runner | 0 (0 · 0) | +0.0 | +2.9 (138) | -3.5 (54) |
| TP +3 % + runner | 0 (0 · 0) | +0.0 | +3.1 (138) | -1.9 (54) |

## Blind spots

- BTC-reactive exits (OBS-B1 / B2) not priced here (BTC tick path) — read in their own scout.
- ML_MKT_HOT / STOP_COOLDOWN are scored only where the best-case zones.csv scored the fill (latest best-case run); PAIR_1H_DOWNTREND only where the fill carries its 1h EMA20/200 stamp (stamped late in the master) — see the coverage line; 'kept by EVERY line' needs all four scored, so it is the smallest row.
- Fills without a price path or with a replica error: 0; per exit, a path that ran out before the exit (END_OF_DATA) is left out, never priced at its last tick.
- Same-bar fixed TP vs stop on 1m bars follows the bar colour (green: low first, red: high first) — a heuristic.
- Exit lines registered on ATR ≥ 0.50 fills from 2026-10-27 (DL 380) keep THEIR frozen cohort and bar; this table is wider (every kept fill, master + B25) and decides nothing on its own.
