# 🔬 ARKUSDT — the last 48 hours, every variable (09-29 22:00 → 10-01 22:00 UTC)

## What made the pair special

- Price 0.2509 → 0.2382 (-5 %), low 0.2072, high 0.4262 (range 106 %).
- Daily futures volume: 30-day normal $24M → last three days $8M (0×) · $308M (13×) · $94M (4×)
- 5-minute ATR: median 1.42 % in these 48 h (first 24 h 2.24 %, last 24 h 1.17 %); the bot's usual pair is ~0.46 %.
- Trend state by minute: EMA stack fully up 33 % of the time, fully down 37 %, mixed the rest.

## Baseline — entering at EVERY minute (no rule)

| Side | Half | minutes | A +0.59/−1.11: win % · net % | B +1.09/−1.11: win % · net % | C +2.09/−1.11: win % · net % | D +3.09/−1.51: win % · net % |
|---|---|---|---|---|---|---|
| LONG | first 24h | 1439 | 66% · -0.066 | 54% · -0.020 | 45% · +0.087 | 45% · +0.203 |
| LONG | last 24h | 1410 | 61% · -0.170 | 48% · -0.141 | 36% · -0.201 | 36% · -0.291 |
| SHORT | first 24h | 1439 | 58% · -0.200 | 46% · -0.204 | 37% · -0.196 | 36% · -0.192 |
| SHORT | last 24h | 1410 | 66% · -0.090 | 52% · -0.090 | 42% · -0.044 | 47% · +0.035 |

## Which variable separates winners from losers? (exit A; terciles cut on the first 24 h; LOW · MID · HIGH win %)

Signed variables are in the direction of the trade (HIGH = strongly with the trade).

| Variable | LONG first 24 h | LONG last 24 h | SHORT first 24 h | SHORT last 24 h | same best end on all four? |
|---|---|---|---|---|---|
| gap5_20 | 67 · 62 · 68 | 65 · 61 · 52 | 58 · 56 · 59 | 73 · 67 · 63 | no |
| gap5_8 | 73 · 61 · 64 | 62 · 64 · 54 | 62 · 59 · 52 | 75 · 65 · 65 | no |
| gap8_13 | 72 · 60 · 66 | 66 · 59 · 53 | 60 · 61 · 53 | 73 · 69 · 63 | no |
| px_vs_ema20 | 68 · 62 · 67 | 65 · 62 · 52 | 59 · 57 · 57 | 77 · 66 · 62 | YES (LOW) |
| px_vs_ema50 | 65 · 63 · 70 | 66 · 57 · 52 | 56 · 53 · 63 | 76 · 70 · 61 | no |
| ema20_slope | 68 · 63 · 67 | 67 · 58 · 52 | 60 · 54 · 59 | 74 · 68 · 62 | YES (LOW) |
| rsi5m | 67 · 65 · 65 | 67 · 58 · 54 | 61 · 54 · 58 | 76 · 70 · 59 | YES (LOW) |
| di_diff | 67 · 67 · 64 | 65 · 59 · 56 | 60 · 55 · 58 | 70 · 68 · 63 | YES (LOW) |
| range_pos24h | 64 · 76 · 59 | 61 · – · – | 63 · 46 · 65 | – · – · 66 | no |
| ret_1h | 70 · 63 · 64 | 66 · 62 · 51 | 60 · 58 · 55 | 76 · 67 · 62 | YES (LOW) |
| ret_4h | 64 · 65 · 68 | 67 · 57 · 51 | 58 · 54 · 61 | 82 · 64 · 64 | no |
| ema_stack | 73 · 67 · 61 | 66 · 58 · 58 | – · 63 · 51 | – · 70 · 63 | no |
| gap1m_5_13 | 68 · 62 · 68 | 58 · 65 · 57 | 57 · 57 · 59 | 77 · 63 · 64 | no |
| rsi1m | 69 · 63 · 65 | 59 · 64 · 60 | 59 · 58 · 56 | 70 · 67 · 63 | no |
| ret_1m | 63 · 67 · 68 | 61 · 63 · 59 | 58 · 54 · 62 | 70 · 64 · 65 | no |
| ret_3m | 64 · 66 · 67 | 58 · 66 · 59 | 59 · 53 · 62 | 75 · 62 · 66 | no |
| ret_5m | 65 · 66 · 66 | 60 · 64 · 57 | 58 · 54 · 61 | 76 · 64 · 63 | no |
| ret_15m | 67 · 60 · 71 | 60 · 63 · 59 | 56 · 57 · 60 | 77 · 62 · 66 | no |
| buy_share_1m | 69 · 61 · 68 | 65 · 59 · 60 | 54 · 62 · 57 | 70 · 68 · 61 | no |
| buy_share_5m | 66 · 67 · 65 | 66 · 60 · 58 | 58 · 59 · 56 | 73 · 67 · 60 | no |
| stretch_1m_ema13 | 67 · 62 · 68 | 61 · 63 · 57 | 56 · 57 · 61 | 75 · 65 · 63 | no |
| pos_15m | 69 · 62 · 67 | 62 · 62 · 60 | 56 · 61 · 56 | 73 · 67 · 61 | no |
| adx5m | 72 · 64 · 62 | 69 · 59 · 45 | 49 · 62 · 63 | 59 · 68 · 88 | no |
| atr5m | 64 · 73 · 61 | 55 · 62 · – | 52 · 54 · 67 | 69 · 66 · – | no |
| vol5m_ratio | 70 · 64 · 63 | 58 · 65 · 60 | 53 · 58 · 62 | 67 · 65 · 66 | no |
| vol1h_vs_24h | 70 · 63 · 65 | 63 · 33 · – | 52 · 62 · 59 | 65 · 85 · – | no |
| vol1m_ratio | 64 · 68 · 65 | 62 · 60 · 62 | 57 · 57 · 59 | 67 · 67 · 65 | no |
| bar_range_1m | 65 · 69 · 63 | 65 · 57 · 59 | 52 · 57 · 64 | 62 · 70 · 74 | no |

## Rules read off MOVR, applied unchanged — one position at a time (trades · won · total % of position, after costs)

| Rule | Exit | First 24 h | Last 24 h | 48 h total |
|---|---|---|---|---|
| LONG after a ≥ 5 % drop from the 4-hour high | +3.09 / −1.51 | 91 · 33% · -8.5% | 47 · 28% · -29.4% | 138 · -37.9% |
| LONG after a ≥ 5 % drop from the 15-min high | +1.09 / −1.11 | 62 · 48% · -9.6% | 5 · 60% · +0.5% | 67 · -9.1% |
| LONG always in | +3.09 / −1.51 | 149 · 36% · -4.0% | 105 · 29% · -45.4% | 254 · -49.5% |
| LONG always in | +0.59 / −1.11 | 358 · 65% · -36.7% | 267 · 63% · -37.5% | 625 · -74.2% |
| SHORT after a ≥ 5 % bounce from the 4-hour low | +3.09 / −1.51 | 114 · 26% · -48.8% | 56 · 43% · +10.8% | 170 · -38.0% |
| SHORT always in | +3.09 / −1.51 | 144 · 31% · -31.7% | 89 · 48% · +9.4% | 233 · -22.3% |
| SHORT always in | +0.59 / −1.11 | 353 · 60% · -67.2% | 279 · 68% · -14.4% | 632 · -81.5% |
