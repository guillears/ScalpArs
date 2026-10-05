## 0. Headline (yr5 admitted ML fills, replica +60 s, 1×, n_seeds/3-weighted)

| cohort | N · days · WR · avg % [day-block 95 % CI] |
|---|---|
| all | 1122 · 235 d · 63% · -0.054 [-0.107, -0.002] |
| H1 EMA20 rising | 630 · 166 d · 65% · -0.001 [-0.067, +0.075] |
| H1 EMA20 falling | 492 · 144 d · 60% · -0.125 [-0.205, -0.039] |
| gap falling − rising | -0.124 [-0.229, -0.014] |

## 1. Overlap: H1_EMA20 tag × engine stamp `entry_btc_1h_slope` (3-bar EMA20 % change incl. FORMING 1h bar)

Live gate geometry: LONG_BTC1H_DEADBAND blocks (−0.05, +0.025); 1hPullback_L 2× cell = [−0.20, −0.10) ∧ BTC 5m ADX [18, 25).

| engine 1h-slope zone | rising: N · days · WR · avg [CI] | falling: N · days · WR · avg [CI] |
|---|---|---|
| ≤−0.20 | 0 · – | 230 · 78 d · 58% · -0.126 [-0.244, -0.001] |
| (−0.20,−0.10] | 0 · – | 156 · 70 d · 63% · -0.169 [-0.281, -0.060] |
| (−0.10,−0.05] | 11 · 6 d · 82% · +0.019 [-0.334, +0.218] | 89 · 44 d · 58% · -0.070 [-0.271, +0.123] |
| (−0.05,+0.025) dead-band | 0 · – | 0 · – |
| [+0.025,+0.05) | 80 · 37 d · 65% · -0.075 [-0.268, +0.100] | 12 · 7 d · 75% · +0.080 [-0.490, +0.428] |
| [+0.05,+0.10) | 136 · 69 d · 69% · +0.034 [-0.092, +0.152] | 5 · 5 d · 75% · -0.109 [-0.509, +0.097] |
| [+0.10,+0.20) | 191 · 79 d · 61% · -0.044 [-0.168, +0.076] | 0 · – |
| ≥+0.20 | 212 · 72 d · 65% · +0.042 [-0.096, +0.171] | 0 · – |

Sign agreement between tag (falling) and engine stamp (<0): 97.5 % of fills. Falling-tag fills with engine stamp ≥ +0.025 (gate sees RISING): 17; rising-tag fills with stamp ≤ −0.05: 11.
Falling-tag fills 'near' the dead-band edges (stamp in (−0.10, −0.05] or [+0.025, +0.05)): 101 of 492.

| 1hPullback_L 2× cell (UNMATCHED, stamp [−0.20,−0.10), BTC ADX [18,25)) | N · days · WR · avg [CI] |
|---|---|
| cell, all | 94 · 51 d · 68% · -0.112 [-0.245, +0.019] |
| cell ∧ falling | 94 · 51 d · 68% · -0.112 [-0.244, +0.019] |
| cell ∧ rising | 0 · – |

| sub-sleeve | rising | falling |
|---|---|---|
| UNMATCHED | 471 · 147 d · 66% · +0.005 [-0.076, +0.089] | 486 · 144 d · 60% · -0.125 [-0.208, -0.042] |
| NONEXP_CALM3D | 150 · 61 d · 62% · -0.019 [-0.135, +0.092] | 0 · – |
| ADX_SURGE_OPEN | 9 · 4 d · 62% · -0.019 [-0.108, +0.475] | 6 · 4 d · 62% · -0.068 [-0.697, +1.198] |

## 2. Dose-response

### deciles of tag's continuous form: 1-bar EMA20 change, % of price (closed bars)

| decile | range | N · days · WR · avg [CI] |
|---|---|---|
| D1 | -0.442 … -0.121 | 113 · 42 d · 58% · -0.138 [-0.337, +0.047] |
| D2 | -0.117 … -0.072 | 113 · 55 d · 66% · -0.035 [-0.201, +0.143] |
| D3 | -0.072 … -0.045 | 111 · 51 d · 58% · -0.238 [-0.393, -0.082] |
| D4 | -0.045 … -0.014 | 112 · 61 d · 64% · -0.032 [-0.204, +0.127] |
| D5 | -0.014 … +0.014 | 114 · 57 d · 61% · -0.127 [-0.259, -0.003] |
| D6 | +0.015 … +0.029 | 110 · 51 d · 61% · -0.081 [-0.231, +0.083] |
| D7 | +0.029 … +0.049 | 112 · 55 d · 74% · +0.131 [-0.008, +0.263] |
| D8 | +0.049 … +0.069 | 113 · 52 d · 60% · -0.093 [-0.250, +0.070] |
| D9 | +0.069 … +0.107 | 115 · 49 d · 58% · -0.002 [-0.150, +0.173] |
| D10 | +0.107 … +0.368 | 109 · 43 d · 68% · +0.042 [-0.166, +0.251] |

### deciles of engine stamp entry_btc_1h_slope

| decile | range | N · days · WR · avg [CI] |
|---|---|---|
| D1 | -1.325 … -0.324 | 113 · 41 d · 58% · -0.119 [-0.313, +0.061] |
| D2 | -0.322 … -0.204 | 112 · 48 d · 58% · -0.138 [-0.303, +0.051] |
| D3 | -0.204 … -0.128 | 112 · 51 d · 61% · -0.191 [-0.338, -0.049] |
| D4 | -0.127 … -0.073 | 114 · 53 d · 60% · -0.111 [-0.239, +0.023] |
| D5 | -0.073 … +0.045 | 110 · 51 d · 71% · +0.007 [-0.148, +0.149] |
| D6 | +0.045 … +0.079 | 112 · 54 d · 72% · +0.011 [-0.133, +0.146] |
| D7 | +0.080 … +0.128 | 112 · 58 d · 61% · -0.019 [-0.156, +0.119] |
| D8 | +0.129 … +0.194 | 112 · 54 d · 60% · -0.059 [-0.220, +0.110] |
| D9 | +0.194 … +0.317 | 112 · 49 d · 64% · +0.094 [-0.077, +0.290] |
| D10 | +0.318 … +1.214 | 113 · 40 d · 66% · -0.029 [-0.200, +0.140] |

### falling-tag cohort only, by engine-stamp zone (is the loss in a sub-range the current gates nearly cut?)

| engine zone | falling N · days · WR · avg [CI] | share of falling-cohort net % |
|---|---|---|
| ≤−0.20 | 230 · 78 d · 58% · -0.126 [-0.242, -0.000] | 47 % |
| (−0.20,−0.10] | 156 · 70 d · 63% · -0.169 [-0.281, -0.060] | 43 % |
| (−0.10,−0.05] | 89 · 44 d · 58% · -0.070 [-0.278, +0.124] | 10 % |
| [+0.025,+0.05) | 12 · 7 d · 75% · +0.080 [-0.462, +0.438] | -1 % |
| [+0.05,+0.10) | 5 · 5 d · 75% · -0.109 [-0.509, +0.097] | 1 % |

## 3. Day units (market-wide variable → a day is one observation)

| state | days | mean of per-day means [95 % CI] | share of days negative |
|---|---|---|---|
| falling | 144 | -0.086 [-0.173, +0.001] | 52 % |
| rising | 166 | -0.002 [-0.077, +0.071] | 43 % |
| paired (days with both states) falling − rising | 75 | +0.001 [-0.151, +0.146] | falling worse on 51 % |

### halves

| half | rising | falling | gap [CI] |
|---|---|---|---|
| Jan–Apr | 280 · 77 d · 66% · -0.003 [-0.108, +0.118] | 237 · 67 d · 56% · -0.160 [-0.291, -0.030] | -0.157 [-0.307, +0.014] |
| May–Oct | 350 · 89 d · 65% · +0.000 [-0.080, +0.082] | 255 · 77 d · 64% · -0.093 [-0.193, +0.013] | -0.093 [-0.221, +0.044] |

### leave-one-month-out

| month left out | falling avg | rising avg | gap |
|---|---|---|---|
| 01 | -0.113 | +0.001 | -0.113 |
| 02 | -0.110 | +0.002 | -0.112 |
| 03 | -0.125 | +0.002 | -0.127 |
| 04 | -0.130 | -0.008 | -0.122 |
| 05 | -0.132 | +0.004 | -0.135 |
| 06 | -0.147 | +0.015 | -0.162 |
| 07 | -0.120 | -0.000 | -0.120 |
| 08 | -0.127 | -0.014 | -0.113 |
| 09 | -0.117 | -0.012 | -0.105 |
| 10 | -0.125 | -0.000 | -0.125 |

### per month (context)

| month | rising | falling |
|---|---|---|
| 01 | 69 · 18 d · 65% · -0.016 | 58 · 17 d · 53% · -0.210 |
| 02 | 73 · 19 d · 63% · -0.024 | 82 · 20 d · 53% · -0.193 |
| 03 | 75 · 23 d · 63% · -0.023 | 63 · 16 d · 62% · -0.120 |
| 04 | 63 · 17 d · 72% · +0.067 | 34 · 14 d · 63% · -0.033 |
| 05 | 88 · 20 d · 66% · -0.030 | 45 · 14 d · 76% · -0.050 |
| 06 | 56 · 13 d · 49% · -0.174 | 84 · 21 d · 70% · -0.011 |
| 07 | 73 · 21 d · 56% · -0.006 | 46 · 15 d · 55% · -0.161 |
| 08 | 42 · 16 d · 83% · +0.191 | 44 · 15 d · 59% · -0.102 |
| 09 | 83 · 18 d · 73% · +0.067 | 35 · 11 d · 55% · -0.230 |
| 10 | 8 · 1 d · 60% · -0.103 | 1 · 1 d · 100% · +0.279 |

### per seed (each seed's own fills; replica price; also the as-replayed pnl)

| seed | rising (replica) | falling (replica) | gap [CI] | as-replayed rising / falling avg |
|---|---|---|---|---|
| 1 | 345 · 147 d · 68% · +0.026 [-0.050, +0.104] | 271 · 124 d · 59% · -0.130 [-0.224, -0.031] | -0.156 [-0.280, -0.027] | -0.012 / -0.145 |
| 2 | 336 · 145 d · 64% · +0.009 [-0.074, +0.093] | 250 · 116 d · 61% · -0.105 [-0.200, -0.006] | -0.114 [-0.235, +0.000] | -0.019 / -0.105 |
| 3 | 343 · 144 d · 63% · -0.038 [-0.109, +0.033] | 239 · 116 d · 62% · -0.139 [-0.236, -0.043] | -0.101 [-0.223, +0.028] | -0.038 / -0.127 |

As-replayed pricing (de-dup, engine's own exits): rising 630 · 166 d · 61% · -0.023 [-0.086, +0.035] · falling 492 · 144 d · 57% · -0.126 [-0.206, -0.044] · gap -0.104

### shuffled-day null (1000 rounds; each day's tag replaced by another same-month day's tag at the same time of day)

observed gap -0.124; null mean -0.005, sd 0.056, 2.5–97.5 % [-0.121, +0.100]; one-sided p(null ≤ obs) = 0.021; two-sided p(|null| ≥ |obs|) = 0.027

## 4. Sleeve-level 'skip ML while H1 EMA20 falling' — expectancy bar + Δ (yr5)

- Sleeve breakeven WR (all admitted, replica): avg win +0.397 / avg loss −0.821 → **67.4 %**
- Blocked cohort: 492 · 144 d · 60% · -0.125 [-0.199, -0.039]; WR 60.3 % vs breakeven 67.4 %
- Window(day)-clustered bootstrap P(avg<0) = 0.999; distinct days 144; N 492
- Largest single-day share of cohort net loss: 9 % (2026-01-29); largest single-pair share: 14 % (UNIUSDT)
- Winners removed: 296 of 492 (of all winners 703); gross winner % removed +52.7 pts, gross loser % removed -84.3 pts
- Δ (1×, %-points summed over fills): +31.6 pts over 273 calendar days = +0.116 pts/day; after 30–50 % haircut +0.058 … +0.081 pts/day. Fills removed 492 of 1122 (44 %).
- Avg of KEPT (rising) fills -0.001 vs all -0.054 (the sleeve stays ~flat on yr5 even after the skip).

Alternative scopes (same bar, for orientation only — NOT pre-registered):

| scope | N · days · WR · avg [CI] | P(avg<0) | Δ pts/day (raw) |
|---|---|---|---|
| falling ∧ stamp ≤ −0.05 (engine also sees falling) | 475 · 139 d · 60% · -0.129 [-0.210, -0.043] | 0.998 | +0.116 |
| falling ∧ stamp ≥ +0.025 (engine sees rising) | 17 · 12 d · 75% · +0.017 [-0.366, +0.322] | 0.496 | -0.000 |
| falling ∧ NOT 1hPullback 2× cell | 398 · 129 d · 58% · -0.128 [-0.216, -0.035] | 0.994 | +0.094 |
| stamp ≤ −0.05 (engine-only definition, any tag) | 486 · 140 d · 60% · -0.126 [-0.200, -0.045] | 1.000 | +0.116 |

## 5. Real fills (refute-only): master pool, MOMENTUM LONG, CLOSED, kept, non-probe, non-MANUAL, as traded (pnl_percentage)

Span 2026-06-18 → 2026-10-02.

| cohort | rising: N · days · WR · avg [CI] | falling: N · days · WR · avg [CI] | gap |
|---|---|---|---|
| all | 77 · 38 d · 78% · +0.211 [+0.033, +0.400] | 44 · 27 d · 77% · +0.154 [-0.081, +0.375] | -0.057 |
| without washed-out Jun-18→Jul-2 | 70 · 33 d · 76% · +0.174 [-0.020, +0.387] | 31 · 18 d · 68% · -0.040 [-0.357, +0.207] | -0.214 |
| washed-out window only | 7 · 5 d · 100% · +0.576 [+0.357, +1.055] | 13 · 9 d · 100% · +0.615 [+0.350, +0.931] | +0.039 |

| engine zone (master) | rising | falling |
|---|---|---|
| ≤−0.20 | 0 · – | 14 · 9 d · 86% · +0.226 |
| (−0.20,−0.10] | 0 · – | 18 · 15 d · 67% · -0.045 |
| (−0.10,−0.05] | 0 · – | 10 · 8 d · 80% · +0.343 |
| [+0.025,+0.05) | 8 · 6 d · 62% · -0.012 | 2 · 2 d · 100% · +0.487 |
| [+0.05,+0.10) | 25 · 18 d · 80% · +0.249 | 0 · – |
| [+0.10,+0.20) | 31 · 16 d · 77% · +0.168 | 0 · – |
| ≥+0.20 | 13 · 8 d · 85% · +0.377 | 0 · – |

| era (master) | rising | falling |
|---|---|---|
| BASE | 17 · 9 d · 88% · +0.328 | 15 · 10 d · 100% · +0.662 |
| B3 | 11 · 6 d · 82% · +0.241 | 11 · 4 d · 73% · +0.051 |
| B1 | 4 · 3 d · 100% · +0.861 | 11 · 6 d · 73% · +0.037 |
| B12 | 14 · 5 d · 71% · +0.187 | 1 · 1 d · 0% · -0.826 |
| B2 | 8 · 5 d · 88% · +0.297 | 2 · 2 d · 50% · -0.232 |
| B16 | 5 · 1 d · 60% · -0.184 | 0 · – |
| B8 | 3 · 2 d · 67% · +0.118 | 1 · 1 d · 100% · +0.098 |
| B14 | 4 · 1 d · 25% · -0.656 | 0 · – |
| B5 | 2 · 1 d · 100% · +0.655 | 1 · 1 d · 100% · +0.047 |
| B13 | 3 · 1 d · 67% · +0.145 | 0 · – |
| B6 | 1 · 1 d · 100% · +0.193 | 1 · 1 d · 0% · -1.990 |
| B9 | 2 · 1 d · 50% · +0.254 | 0 · – |
| B4 | 1 · 1 d · 100% · +0.117 | 0 · – |
| B7 | 1 · 1 d · 100% · +0.098 | 0 · – |
| B10 | 1 · 1 d · 100% · +0.096 | 0 · – |
| B15 | 0 · – | 1 · 1 d · 0% · -0.993 |

Master skip-rule read: breakeven WR 63.9 %; blocked cohort 44 · 27 d · 77% · +0.154 [-0.071, +0.382]; winners removed 34; Δ -6.76 pts over 58 trading days with ML fills (-0.117 pts/day); without washed-out: Δ +1.23 pts.

