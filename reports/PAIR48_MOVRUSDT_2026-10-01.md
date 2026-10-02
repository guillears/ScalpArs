# 🔬 MOVRUSDT — the last 48 hours, every variable (09-29 22:00 → 10-01 22:00 UTC)

## What made the pair special

- Price 1.211 → 2.841 (+135 %), low 1.163, high 3.34 (range 187 %).
- Daily futures volume: 30-day normal $7M → last three days $30M (4×) · $784M (112×) · $1,496M (213×)
- 5-minute ATR: median 3.06 % in these 48 h (first 24 h 2.48 %, last 24 h 3.39 %); the bot's usual pair is ~0.46 %.
- Trend state by minute: EMA stack fully up 58 % of the time, fully down 22 %, mixed the rest.

## Baseline — entering at EVERY minute (no rule)

| Side | Half | minutes | A +0.59/−1.11: win % · net % | B +1.09/−1.11: win % · net % | C +2.09/−1.11: win % · net % | D +3.09/−1.51: win % · net % |
|---|---|---|---|---|---|---|
| LONG | first 24h | 1439 | 70% · -0.030 | 57% · +0.044 | 46% · +0.256 | 48% · +0.463 |
| LONG | last 24h | 1410 | 66% · -0.100 | 52% · -0.084 | 36% · -0.069 | 38% · +0.063 |
| SHORT | first 24h | 1439 | 60% · -0.192 | 43% · -0.266 | 26% · -0.384 | 22% · -0.664 |
| SHORT | last 24h | 1410 | 63% · -0.143 | 49% · -0.132 | 31% · -0.224 | 31% · -0.229 |

## Which variable separates winners from losers? (exit A; terciles cut on the first 24 h; LOW · MID · HIGH win %)

Signed variables are in the direction of the trade (HIGH = strongly with the trade).

| Variable | LONG first 24 h | LONG last 24 h | SHORT first 24 h | SHORT last 24 h | same best end on all four? |
|---|---|---|---|---|---|
| gap5_20 | 75 · 65 · 71 | 66 · 65 · 67 | 64 · 62 · 55 | 62 · 63 · 64 | no |
| gap5_8 | 73 · 66 · 71 | 67 · 64 · 66 | 66 · 59 · 56 | 61 · 68 · 63 | no |
| gap8_13 | 76 · 63 · 71 | 65 · 66 · 67 | 64 · 63 · 55 | 61 · 64 · 65 | no |
| px_vs_ema20 | 75 · 66 · 69 | 66 · 69 · 64 | 65 · 61 · 56 | 63 · 62 · 65 | no |
| px_vs_ema50 | 78 · 63 · 70 | 67 · 65 · 66 | 64 · 64 · 53 | 64 · 61 · 64 | no |
| ema20_slope | 74 · 64 · 71 | 65 · 66 · 67 | 64 · 62 · 56 | 59 · 67 · 65 | no |
| rsi5m | 74 · 67 · 69 | 66 · 67 · 63 | 65 · 59 · 57 | 61 · 63 · 65 | no |
| di_diff | 71 · 73 · 66 | 67 · 62 · 68 | 68 · 55 · 59 | 59 · 66 · 64 | no |
| range_pos24h | 72 · 69 · 69 | 73 · 67 · 62 | 64 · 60 · 57 | 65 · 63 · 58 | YES (LOW) |
| ret_1h | 74 · 65 · 71 | 65 · 68 · 65 | 65 · 60 · 56 | 62 · 64 · 64 | no |
| ret_4h | 74 · 69 · 67 | 68 · 66 · 64 | 66 · 58 · 58 | 66 · 65 · 59 | YES (LOW) |
| ema_stack | 74 · 74 · 68 | 66 · 66 · 66 | – · 62 · 58 | – · 64 · 63 | no |
| gap1m_5_13 | 72 · 72 · 65 | 67 · 66 · 65 | 70 · 55 · 56 | 62 · 66 · 63 | no |
| rsi1m | 72 · 70 · 67 | 65 · 67 · 65 | 69 · 57 · 56 | 63 · 62 · 64 | no |
| ret_1m | 70 · 67 · 73 | 65 · 68 · 66 | 61 · 62 · 59 | 64 · 62 · 64 | no |
| ret_3m | 69 · 70 · 71 | 68 · 63 · 66 | 63 · 59 · 59 | 63 · 67 · 61 | no |
| ret_5m | 71 · 69 · 70 | 68 · 63 · 66 | 63 · 62 · 57 | 63 · 68 · 60 | no |
| ret_15m | 70 · 75 · 65 | 66 · 68 · 64 | 70 · 57 · 55 | 63 · 62 · 64 | no |
| buy_share_1m | 67 · 70 · 72 | 64 · 66 · 68 | 59 · 60 · 62 | 65 · 62 · 64 | no |
| buy_share_5m | 71 · 68 · 71 | 66 · 67 · 64 | 62 · 60 · 60 | 63 · 64 · 63 | no |
| stretch_1m_ema13 | 72 · 71 · 67 | 67 · 65 · 65 | 68 · 58 · 55 | 64 · 65 · 62 | no |
| pos_15m | 71 · 73 · 66 | 68 · 66 · 63 | 69 · 56 · 57 | 64 · 67 · 60 | no |
| adx5m | 71 · 67 · 71 | 61 · 68 · 68 | 57 · 64 · 60 | 67 · 62 · 61 | no |
| atr5m | 67 · 73 · 69 | – · 58 · 67 | 59 · 60 · 63 | – · 71 · 62 | no |
| vol5m_ratio | 70 · 74 · 67 | 64 · 68 · 66 | 57 · 58 · 66 | 68 · 62 · 59 | no |
| vol1h_vs_24h | 73 · 68 · 69 | 65 · 69 · – | 55 · 64 · 62 | 65 · 59 · – | no |
| vol1m_ratio | 70 · 69 · 71 | 68 · 65 · 65 | 61 · 59 · 61 | 64 · 61 · 66 | no |
| bar_range_1m | 73 · 66 · 71 | 68 · 65 · 66 | 55 · 62 · 64 | 64 · 65 · 62 | no |
