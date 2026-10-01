# 🌀 BTC-chop rules — backtest support (yr3 engine replay, momentum longs, current gates)

1112 unique trades (5 seeds collapsed), 238 trading days, Jan-04 → Sep-24 2026. Chop (eff72 ≤ 0.007): 127 trades on 53 days = 28 episodes. Cells: trades · days · WR · avg %.

## A. Chop vs the rest

| Period | chop | chop ∧ 2nd+ burst | chop, other fills | not chop | Δ chop − rest |
|---|---|---|---|---|---|
| Jan–Apr | 54 · 21d · 48% · -0.204 | 5 · 3d · 20% · -0.602 | 49 · 21d · 51% · -0.163 | 468 · 105d · 56% · -0.074 | -0.130 |
| May–Sep | 73 · 32d · 41% · -0.234 | 11 · 7d · 27% · -0.305 | 62 · 32d · 44% · -0.222 | 517 · 126d · 58% · -0.077 | -0.158 |
| ALL | 127 · 53d · 44% · -0.221 | 16 · 10d · 25% · -0.398 | 111 · 53d · 47% · -0.196 | 985 · 231d · 57% · -0.075 | -0.146 |

| Month | chop | not chop | Δ |
|---|---|---|---|
| 2026-01 | 5 · 4d · 80% · +0.030 | 128 · 26d · 53% · -0.130 | +0.160 |
| 2026-02 | 34 · 8d · 44% · -0.229 | 126 · 25d · 52% · -0.124 | -0.105 |
| 2026-03 | 6 · 4d · 50% · -0.112 | 131 · 30d · 60% · -0.005 | -0.108 |
| 2026-04 | 9 · 5d · 44% · -0.299 | 83 · 24d · 60% · -0.019 | -0.280 |
| 2026-05 | 17 · 9d · 59% · -0.041 | 120 · 25d · 59% · -0.112 | +0.071 |
| 2026-06 | 15 · 7d · 53% · +0.075 | 132 · 30d · 56% · -0.100 | +0.175 |
| 2026-07 | 16 · 7d · 19% · -0.503 | 118 · 25d · 53% · -0.081 | -0.423 |
| 2026-08 | 13 · 4d · 31% · -0.524 | 71 · 25d · 70% · +0.051 | -0.576 |
| 2026-09 | 12 · 5d · 42% · -0.222 | 76 · 21d · 57% · -0.094 | -0.128 |

## B. Is the chop gap luck? (like-for-like: the same fill-level Δ in the observed and in the null)

- Jan–Apr: Δ -0.130 over 10 chop episodes · circular-shift p = **0.120** · day-block bootstrap 95 % CI [-0.309, +0.096], P(Δ ≥ 0) = **0.107**
- May–Sep: Δ -0.158 over 18 chop episodes · circular-shift p = **0.026** · day-block bootstrap 95 % CI [-0.363, +0.074], P(Δ ≥ 0) = **0.094**
- ALL: Δ -0.146 over 28 chop episodes · circular-shift p = **0.011** · day-block bootstrap 95 % CI [-0.283, +0.017], P(Δ ≥ 0) = **0.036**
- ALL without 2026-08: Δ -0.102 · circular-shift p = 0.073 · P(Δ ≥ 0) = 0.108
- ALL without 2026-07 and 2026-08: Δ -0.049 · circular-shift p = 0.258 · P(Δ ≥ 0) = 0.280
- By how many of the 5 replays took the trade (single-replay fills are the fragile ones): 1 seed: chop 46 · 31d · 37% · -0.332 vs rest -0.153 · 2 seeds: chop 33 · 20d · 33% · -0.362 vs rest -0.085 · 3 seeds: chop 20 · 15d · 60% · -0.006 vs rest +0.030 · 4+ seeds: chop 28 · 23d · 57% · -0.028 vs rest +0.000

## C. What-if PER SEED — one replay = one bot run (the union of five replays would triple the gain)

| Seed | chop trades (winners) | R1 gain 1× | R1 gain ×mult | 2nd+ burst in chop (winners) | R2 gain 1× | R2 gain ×mult |
|---|---|---|---|---|---|---|
| 1 | 59 (29) | +11.6 | +16.4 | 8 (2) | +4.1 | +4.6 |
| 2 | 63 (32) | +12.2 | +19.6 | 8 (3) | +3.1 | +4.0 |
| 3 | 58 (29) | +9.9 | +20.0 | 11 (5) | +3.0 | +4.7 |
| 4 | 56 (24) | +10.1 | +14.1 | 8 (3) | +3.3 | +3.8 |
| 5 | 60 (33) | +4.1 | +3.8 | 9 (5) | +1.5 | +2.4 |
| **mean** | | **+9.6** | **+14.8** | | **+3.0** | **+3.9** |

In Σ-of-% points over ~9 months, before the 30–50 % in-sample haircut (0.007 was cut on the live master) → about +4.8 to +6.7 at 1× for R1.

## D. Robustness

Per seed (same year replayed — NOT independent): Δ chop − rest, and the chop ∧ 2nd+ burst average

- seed 1: chop 59 · -0.197 vs rest -0.030 (Δ -0.167) · chop ∧ 2nd+ burst 8 · -0.510
- seed 2: chop 63 · -0.194 vs rest -0.052 (Δ -0.142) · chop ∧ 2nd+ burst 8 · -0.386
- seed 3: chop 58 · -0.171 vs rest -0.044 (Δ -0.127) · chop ∧ 2nd+ burst 11 · -0.274
- seed 4: chop 56 · -0.180 vs rest -0.002 (Δ -0.179) · chop ∧ 2nd+ burst 8 · -0.413
- seed 5: chop 60 · -0.068 vs rest -0.040 (Δ -0.028) · chop ∧ 2nd+ burst 9 · -0.165

Non-nested eff72 buckets (trades · days · WR · avg %):

| bucket | Jan–Apr | May–Sep |
|---|---|---|
| 0.000–0.003 | 24 · 14d · 62% · +0.038 | 31 · 16d · 42% · -0.249 |
| 0.003–0.007 | 30 · 13d · 37% · -0.397 | 42 · 23d · 40% · -0.224 |
| 0.007–0.01 | 19 · 14d · 37% · -0.323 | 14 · 8d · 79% · +0.219 |
| 0.010–0.015 | 33 · 17d · 58% · -0.041 | 39 · 22d · 69% · -0.018 |
| 0.015–0.026 | 101 · 44d · 58% · +0.013 | 116 · 47d · 50% · -0.181 |
| 0.026–… | 315 · 83d · 56% · -0.090 | 348 · 96d · 59% · -0.061 |

Pre-gate (raw replay, 1358 trades): chop 153 · 58d · 46% · -0.212 vs rest 1205 · 243d · 55% · -0.106 · chop ∧ 2nd+ burst 21 · 12d · 29% · -0.330

BTC 1h slope inside and outside chop (trades · days · WR · avg %):

| | Jan–Apr | May–Sep |
|---|---|---|
| chop ∧ slope ≥ 0 | 34 · 16d · 47% · -0.241 | 49 · 23d · 35% · -0.275 |
| not chop ∧ slope ≥ 0 | 240 · 71d · 57% · -0.062 | 303 · 82d · 61% · -0.017 |
| chop ∧ slope < 0 | 20 · 8d · 50% · -0.141 | 24 · 11d · 54% · -0.151 |
| not chop ∧ slope < 0 | 228 · 63d · 55% · -0.086 | 214 · 69d · 55% · -0.162 |

Never-green trades (peak P&L never above 0): chop 11.0 % vs rest 12.6 % — no distinct loser mechanism.

## Verdict — OBSERVE, not arm (dual review 2026-10-01)

- Significance is borderline: full year p 0.011 (circular shift) / 0.036 (day-block bootstrap, 95 % CI [-0.28, +0.02] still spans 0); Jan–Apr not significant (0.12 / 0.11); May–Sep 0.026 / 0.094; without Jul–Aug 0.26.
- No dose-response: the choppiest bucket (≤ 0.003) is fine in Jan–Apr; the signal sits in 0.003–0.007.
- The live master's chop losers are the two days that prompted the idea (Sep-29, Oct-01); its five earlier chop trades netted +$564 (3 winners +$751; the Jul-10 ADA+LIT burst lost −$187).
- Inside chop the gap appears when BTC's 1h slope is UP, not down (table in D).
- The frozen CURRENT_STATE bar (fresh fills, N ≥ 15 on ≥ 8 days over ≥ 4 episodes) stays as registered.

## Limits

- Fill-level what-if: a blocked trade frees a slot the engine might have used — the true effect needs a replay with the rule coded.
- The replay finds live winners more easily than live losers (replay selection bias) and the whole backtest sleeve is slightly negative.
- One 9-month span; chop days cluster into few episodes, which is what limits the statistical power.
