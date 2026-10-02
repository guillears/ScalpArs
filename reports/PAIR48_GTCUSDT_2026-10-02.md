# 🔬 GTCUSDT — the last 48 hours, every variable (09-30 00:00 → 10-02 00:00 UTC)

## What made the pair special

- Price 0.08646 → 0.1442 (+67 %), low 0.08592, high 0.1532 (range 78 %).
- Daily futures volume: 30-day normal $1M → last three days $50M (67×) · $61M (82×) · $6M (8×)
- 5-minute ATR: median 0.99 % in these 48 h (first 24 h 0.85 %, last 24 h 1.16 %); the bot's usual pair is ~0.46 %.
- Trend state by minute: EMA stack fully up 50 % of the time, fully down 30 %, mixed the rest.

## Baseline — entering at EVERY minute (no rule)

| Side | Half | minutes | A +0.59/−1.11: win % · net % | B +1.09/−1.11: win % · net % | C +2.09/−1.11: win % · net % | D +3.09/−1.51: win % · net % |
|---|---|---|---|---|---|---|
| LONG | first 24h | 1439 | 62% · -0.109 | 52% · -0.122 | 46% · -0.162 | 46% · -0.214 |
| LONG | last 24h | 1410 | 71% · +0.015 | 63% · +0.163 | 55% · +0.268 | 57% · +0.325 |
| SHORT | first 24h | 1439 | 57% · -0.120 | 48% · -0.096 | 43% · -0.090 | 45% · -0.069 |
| SHORT | last 24h | 1410 | 49% · -0.361 | 37% · -0.389 | 29% · -0.425 | 28% · -0.631 |

## Which variable separates winners from losers? (exit A; terciles cut on the first 24 h; LOW · MID · HIGH win %)

Signed variables are in the direction of the trade (HIGH = strongly with the trade).

| Variable | LONG first 24 h | LONG last 24 h | SHORT first 24 h | SHORT last 24 h | same best end on all four? |
|---|---|---|---|---|---|
| gap5_20 | 56 · 66 · 65 | 75 · 75 · 70 | 54 · 42 · 76 | 50 · 44 · 48 | no |
| gap5_8 | 61 · 66 · 60 | 73 · 83 · 67 | 55 · 47 · 70 | 53 · 35 · 51 | no |
| gap8_13 | 55 · 70 · 63 | 79 · 74 · 70 | 55 · 39 · 77 | 50 · 45 · 40 | no |
| px_vs_ema20 | 57 · 67 · 63 | 71 · 80 · 69 | 54 · 45 · 73 | 51 · 39 · 51 | no |
| px_vs_ema50 | 55 · 71 · 61 | – · 83 · 69 | 58 · 36 · 77 | 52 · 33 · – | no |
| ema20_slope | 55 · 70 · 63 | 77 · 76 · 69 | 54 · 41 · 76 | 50 · 44 · 47 | no |
| rsi5m | 63 · 60 · 64 | 59 · 79 · 69 | 52 · 55 · 65 | 51 · 39 · 61 | no |
| di_diff | 59 · 70 · 58 | 79 · 77 · 68 | 57 · 44 · 70 | 53 · 41 · 37 | no |
| range_pos24h | 68 · 57 · 63 | 73 · 75 · 68 | 54 · 61 · 56 | 54 · 42 · 45 | no |
| ret_1h | 62 · 65 · 61 | 77 · 77 · 68 | 54 · 52 · 66 | 53 · 37 · 46 | no |
| ret_4h | 54 · 74 · 60 | – · 73 · 70 | 61 · 32 · 78 | 50 · 45 · – | no |
| ema_stack | – · 62 · 63 | – · 72 · 71 | 54 · 58 · 59 | 51 · 40 · 49 | no |
| gap1m_5_13 | 58 · 68 · 61 | 74 · 68 · 71 | 59 · 49 · 64 | 51 · 48 · 43 | no |
| rsi1m | 63 · 62 · 62 | 73 · 74 · 69 | 53 · 62 · 57 | 52 · 44 · 45 | no |
| ret_1m | 60 · 66 · 61 | 70 · 75 · 71 | 61 · 49 · 61 | 49 · 41 · 53 | no |
| ret_3m | 57 · 67 · 63 | 72 · 73 · 70 | 60 · 46 · 66 | 51 · 45 · 48 | no |
| ret_5m | 59 · 67 · 61 | 73 · 72 · 70 | 61 · 46 · 64 | 51 · 45 · 48 | no |
| ret_15m | 61 · 68 · 58 | 71 · 74 · 70 | 61 · 47 · 63 | 51 · 45 · 46 | no |
| buy_share_1m | 64 · 60 · 64 | 73 · 69 · 72 | 56 · 63 · 52 | 44 · 52 · 48 | no |
| buy_share_5m | 63 · 58 · 66 | 69 · 71 · 75 | 54 · 65 · 53 | 46 · 50 · 49 | no |
| stretch_1m_ema13 | 57 · 70 · 60 | 73 · 72 · 70 | 60 · 47 · 65 | 51 · 47 · 45 | no |
| pos_15m | 60 · 66 · 62 | 73 · 72 · 70 | 59 · 57 · 56 | 53 · 45 · 45 | no |
| adx5m | 70 · 56 · 62 | 70 · 75 · 70 | 36 · 67 · 69 | 48 · 46 · 54 | no |
| atr5m | 73 · 63 · 51 | 63 · 72 · 75 | 29 · 66 · 76 | 60 · 43 · 48 | no |
| vol5m_ratio | 62 · 64 · 61 | 83 · 70 · 65 | 58 · 57 · 56 | 33 · 51 · 56 | no |
| vol1h_vs_24h | 59 · 68 · 60 | 74 · 63 · 72 | 59 · 48 · 65 | 44 · 55 · 49 | no |
| vol1m_ratio | 63 · 62 · 63 | 77 · 70 · 69 | 57 · 59 · 56 | 41 · 49 · 54 | no |
| bar_range_1m | 68 · 64 · 55 | 79 · 73 · 68 | 39 · 62 · 71 | 42 · 44 · 54 | no |

## Rules read off MOVR, applied unchanged — one position at a time (trades · won · total % of position, after costs)

| Rule | Exit | First 24 h | Last 24 h | 48 h total |
|---|---|---|---|---|
| LONG after a ≥ 5 % drop from the 4-hour high | +3.09 / −1.51 | 42 · 33% · -18.7% | 17 · 47% · +8.4% | 59 · -10.4% |
| LONG after a ≥ 5 % drop from the 15-min high | +1.09 / −1.11 | 19 · 42% · -5.6% | 11 · 45% · -2.4% | 30 · -8.0% |
| LONG always in | +3.09 / −1.51 | 85 · 42% · -13.6% | 91 · 45% · +6.4% | 176 · -7.3% |
| LONG always in | +0.59 / −1.11 | 178 · 57% · -36.1% | 231 · 66% · -21.8% | 409 · -57.9% |
| SHORT after a ≥ 5 % bounce from the 4-hour low | +3.09 / −1.51 | 43 · 37% · +0.3% | 67 · 25% · -34.5% | 110 · -34.2% |
| SHORT always in | +3.09 / −1.51 | 83 · 43% · +2.9% | 92 · 26% · -47.1% | 175 · -44.2% |
| SHORT always in | +0.59 / −1.11 | 195 · 65% · -13.1% | 227 · 57% · -53.8% | 422 · -66.9% |
