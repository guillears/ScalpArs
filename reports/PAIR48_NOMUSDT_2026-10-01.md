# 🔬 NOMUSDT — the last 48 hours, every variable (09-29 22:00 → 10-01 22:00 UTC)

## What made the pair special

- Price 0.00212 → 0.002726 (+29 %), low 0.002081, high 0.003266 (range 57 %).
- Daily futures volume: 30-day normal $14M → last three days $5M (0×) · $32M (2×) · $95M (7×)
- 5-minute ATR: median 1.29 % in these 48 h (first 24 h 0.82 %, last 24 h 1.55 %); the bot's usual pair is ~0.46 %.
- Trend state by minute: EMA stack fully up 40 % of the time, fully down 33 %, mixed the rest.

## Baseline — entering at EVERY minute (no rule)

| Side | Half | minutes | A +0.59/−1.11: win % · net % | B +1.09/−1.11: win % · net % | C +2.09/−1.11: win % · net % | D +3.09/−1.51: win % · net % |
|---|---|---|---|---|---|---|
| LONG | first 24h | 1439 | 66% · -0.067 | 55% · -0.019 | 49% · +0.059 | 49% · +0.120 |
| LONG | last 24h | 1410 | 66% · -0.093 | 53% · -0.045 | 43% · +0.057 | 43% · +0.089 |
| SHORT | first 24h | 1439 | 55% · -0.206 | 44% · -0.205 | 39% · -0.205 | 41% · -0.225 |
| SHORT | last 24h | 1410 | 62% · -0.170 | 48% · -0.167 | 37% · -0.162 | 38% · -0.190 |

## Which variable separates winners from losers? (exit A; terciles cut on the first 24 h; LOW · MID · HIGH win %)

Signed variables are in the direction of the trade (HIGH = strongly with the trade).

| Variable | LONG first 24 h | LONG last 24 h | SHORT first 24 h | SHORT last 24 h | same best end on all four? |
|---|---|---|---|---|---|
| gap5_20 | 71 · 61 · 66 | 73 · 67 · 59 | 61 · 54 · 50 | 71 · 62 · 52 | YES (LOW) |
| gap5_8 | 72 · 61 · 64 | 69 · 71 · 60 | 59 · 58 · 48 | 71 · 54 · 56 | no |
| gap8_13 | 70 · 60 · 66 | 72 · 66 · 59 | 61 · 54 · 50 | 71 · 62 · 52 | YES (LOW) |
| px_vs_ema20 | 70 · 65 · 62 | 71 · 62 · 63 | 65 · 51 · 49 | 68 · 68 · 51 | YES (LOW) |
| px_vs_ema50 | 71 · 58 · 67 | 71 · 64 · 62 | 61 · 56 · 48 | 68 · 66 · 54 | YES (LOW) |
| ema20_slope | 74 · 58 · 64 | 72 · 64 · 61 | 63 · 56 · 46 | 70 · 60 · 55 | YES (LOW) |
| rsi5m | 72 · 66 · 59 | 70 · 66 · 62 | 66 · 52 · 48 | 68 · 63 · 53 | YES (LOW) |
| di_diff | 75 · 60 · 62 | 70 · 67 · 61 | 65 · 57 · 43 | 67 · 64 · 53 | YES (LOW) |
| range_pos24h | 75 · 65 · 57 | 66 · 83 · 62 | 69 · 52 · 45 | 70 · 31 · 61 | no |
| ret_1h | 72 · 57 · 68 | 70 · 70 · 60 | 58 · 59 · 48 | 69 · 58 · 57 | no |
| ret_4h | 72 · 61 · 63 | 73 · 59 · 62 | 61 · 59 · 46 | 67 · 69 · 52 | no |
| ema_stack | – · 69 · 59 | – · 71 · 59 | – · 59 · 47 | – · 66 · 52 | no |
| gap1m_5_13 | 61 · 73 · 62 | 66 · 65 · 65 | 59 · 51 · 55 | 67 · 56 · 59 | no |
| rsi1m | 64 · 70 · 63 | 65 · 67 · 65 | 57 · 55 · 54 | 69 · 54 · 60 | no |
| ret_1m | 63 · 69 · 65 | 66 · 67 · 65 | 59 · 49 · 57 | 63 · 59 · 62 | no |
| ret_3m | 63 · 68 · 65 | 69 · 64 · 64 | 57 · 52 · 56 | 64 · 58 · 62 | no |
| ret_5m | 63 · 67 · 66 | 66 · 60 · 68 | 56 · 54 · 55 | 63 · 58 · 63 | no |
| ret_15m | 63 · 70 · 64 | 66 · 68 · 64 | 59 · 52 · 55 | 67 · 52 · 61 | no |
| buy_share_1m | 65 · 66 · 66 | 70 · 65 · 64 | 55 · 58 · 53 | 60 · 65 · 56 | no |
| buy_share_5m | 66 · 69 · 61 | 66 · 66 · 65 | 58 · 57 · 50 | 63 · 62 · 60 | no |
| stretch_1m_ema13 | 61 · 70 · 66 | 67 · 61 · 67 | 57 · 52 · 56 | 67 · 57 · 59 | no |
| pos_15m | 59 · 72 · 67 | 72 · 59 · 67 | 54 · 55 · 57 | 67 · 62 · 56 | no |
| adx5m | 60 · 67 · 70 | 65 · 73 · 62 | 55 · 55 · 56 | 62 · 55 · 66 | no |
| atr5m | 65 · 67 · 64 | – · 69 · 64 | 43 · 58 · 64 | – · 58 · 64 | no |
| vol5m_ratio | 72 · 64 · 60 | 69 · 64 · 64 | 53 · 51 · 61 | 60 · 59 · 66 | no |
| vol1h_vs_24h | 58 · 72 · 66 | 68 · 67 · 55 | 55 · 47 · 63 | 58 · 63 · 74 | no |
| vol1m_ratio | 66 · 66 · 64 | 63 · 62 · 73 | 52 · 55 · 58 | 60 · 64 · 61 | no |
| bar_range_1m | 66 · 69 · 62 | 72 · 68 · 64 | 49 · 53 · 63 | 55 · 56 · 66 | no |

## Rules read off MOVR, applied unchanged — one position at a time (trades · won · total % of position, after costs)

| Rule | Exit | First 24 h | Last 24 h | 48 h total |
|---|---|---|---|---|
| LONG after a ≥ 5 % drop from the 4-hour high | +3.09 / −1.51 | 17 · 35% · -5.0% | 56 · 34% · -8.7% | 73 · -13.7% |
| LONG after a ≥ 5 % drop from the 15-min high | +1.09 / −1.11 | 4 · 75% · +1.7% | 15 · 40% · -5.1% | 19 · -3.4% |
| LONG always in | +3.09 / −1.51 | 65 · 48% · +6.2% | 101 · 37% · -7.4% | 166 · -1.3% |
| LONG always in | +0.59 / −1.11 | 172 · 62% · -22.5% | 293 · 65% · -31.0% | 465 · -53.6% |
| SHORT after a ≥ 5 % bounce from the 4-hour low | +3.09 / −1.51 | 35 · 34% · -10.9% | 66 · 39% · -1.7% | 101 · -12.6% |
| SHORT always in | +3.09 / −1.51 | 67 · 36% · -13.7% | 94 · 38% · -8.8% | 161 · -22.4% |
| SHORT always in | +0.59 / −1.11 | 186 · 61% · -28.7% | 288 · 64% · -36.0% | 474 · -64.7% |
