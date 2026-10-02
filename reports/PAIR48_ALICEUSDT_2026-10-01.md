# 🔬 ALICEUSDT — the last 48 hours, every variable (09-29 22:00 → 10-01 22:00 UTC)

## What made the pair special

- Price 0.1676 → 0.219 (+31 %), low 0.1602, high 0.2467 (range 54 %).
- Daily futures volume: 30-day normal $3M → last three days $24M (9×) · $8M (3×) · $95M (34×)
- 5-minute ATR: median 0.54 % in these 48 h (first 24 h 0.55 %, last 24 h 0.54 %); the bot's usual pair is ~0.46 %.
- Trend state by minute: EMA stack fully up 44 % of the time, fully down 30 %, mixed the rest.

## Baseline — entering at EVERY minute (no rule)

| Side | Half | minutes | A +0.59/−1.11: win % · net % | B +1.09/−1.11: win % · net % | C +2.09/−1.11: win % · net % | D +3.09/−1.51: win % · net % |
|---|---|---|---|---|---|---|
| LONG | first 24h | 1439 | 57% · -0.103 | 48% · -0.100 | 47% · -0.072 | 47% · -0.106 |
| LONG | last 24h | 1410 | 61% · -0.052 | 54% · -0.009 | 48% · -0.034 | 48% · -0.002 |
| SHORT | first 24h | 1439 | 57% · -0.114 | 49% · -0.125 | 49% · -0.120 | 50% · -0.123 |
| SHORT | last 24h | 1410 | 52% · -0.194 | 43% · -0.213 | 38% · -0.281 | 38% · -0.325 |

## Which variable separates winners from losers? (exit A; terciles cut on the first 24 h; LOW · MID · HIGH win %)

Signed variables are in the direction of the trade (HIGH = strongly with the trade).

| Variable | LONG first 24 h | LONG last 24 h | SHORT first 24 h | SHORT last 24 h | same best end on all four? |
|---|---|---|---|---|---|
| gap5_20 | 59 · 54 · 57 | 75 · 61 · 60 | 59 · 56 · 56 | 58 · 47 · 36 | YES (LOW) |
| gap5_8 | 51 · 51 · 67 | 74 · 59 · 54 | 52 · 54 · 66 | 67 · 48 · 41 | no |
| gap8_13 | 56 · 57 · 56 | 65 · 60 · 61 | 58 · 55 · 58 | 57 · 49 · 49 | no |
| px_vs_ema20 | 54 · 53 · 62 | 79 · 60 · 56 | 57 · 55 · 59 | 62 · 48 · 33 | no |
| px_vs_ema50 | 61 · 57 · 52 | – · 68 · 57 | 65 · 55 · 51 | 60 · 36 · – | no |
| ema20_slope | 56 · 54 · 60 | 72 · 63 · 57 | 57 · 55 · 59 | 60 · 45 · 40 | no |
| rsi5m | 53 · 57 · 59 | 78 · 64 · 55 | 59 · 55 · 57 | 60 · 47 · 26 | no |
| di_diff | 57 · 60 · 52 | 66 · 69 · 54 | 67 · 48 · 56 | 63 · 42 · 32 | no |
| range_pos24h | 64 · 50 · 55 | – · 83 · 57 | 68 · 61 · 41 | 53 · 46 · – | no |
| ret_1h | 50 · 62 · 57 | 67 · 57 · 63 | 58 · 49 · 63 | 57 · 51 · 46 | no |
| ret_4h | 56 · 69 · 45 | 80 · 54 · 61 | 74 · 43 · 56 | 57 · 41 · 29 | no |
| ema_stack | – · 56 · 57 | – · 70 · 52 | – · 56 · 58 | – · 58 · 28 | no |
| gap1m_5_13 | 52 · 56 · 62 | 71 · 56 · 56 | 55 · 56 · 60 | 62 · 47 · 44 | no |
| rsi1m | 49 · 56 · 65 | 73 · 61 · 52 | 53 · 56 · 62 | 65 · 50 · 38 | no |
| ret_1m | 56 · 53 · 62 | 63 · 56 · 67 | 52 · 63 · 57 | 57 · 48 · 54 | no |
| ret_3m | 56 · 54 · 60 | 70 · 53 · 62 | 56 · 58 · 56 | 56 · 52 · 47 | no |
| ret_5m | 54 · 54 · 61 | 67 · 56 · 60 | 53 · 62 · 56 | 58 · 48 · 50 | no |
| ret_15m | 55 · 56 · 59 | 69 · 58 · 56 | 57 · 56 · 58 | 63 · 48 · 45 | no |
| buy_share_1m | 52 · 58 · 60 | 57 · 63 · 60 | 52 · 58 · 61 | 47 · 56 · 50 | no |
| buy_share_5m | 53 · 55 · 62 | 60 · 65 · 54 | 55 · 58 · 58 | 53 · 53 · 48 | no |
| stretch_1m_ema13 | 52 · 57 · 61 | 71 · 53 · 58 | 54 · 57 · 60 | 61 · 49 · 46 | no |
| pos_15m | 51 · 58 · 60 | 67 · 60 · 57 | 55 · 59 · 57 | 59 · 53 · 43 | no |
| adx5m | 53 · 45 · 71 | 57 · 64 · 63 | 63 · 67 · 41 | 51 · 50 · 56 | no |
| atr5m | 55 · 53 · 62 | 48 · 68 · 70 | 51 · 62 · 58 | 52 · 43 · 57 | no |
| vol5m_ratio | 60 · 54 · 56 | 62 · 62 · 58 | 47 · 65 · 59 | 48 · 48 · 60 | no |
| vol1h_vs_24h | 57 · 59 · 54 | 61 · 43 · 66 | 53 · 60 · 58 | 40 · 56 · 53 | no |
| vol1m_ratio | 53 · 59 · 58 | 60 · 62 · 61 | 59 · 57 · 56 | 47 · 53 · 55 | no |
| bar_range_1m | 51 · 57 · 62 | 54 · 60 · 67 | 58 · 58 · 55 | 46 · 51 · 58 | no |

## Rules read off MOVR, applied unchanged — one position at a time (trades · won · total % of position, after costs)

| Rule | Exit | First 24 h | Last 24 h | 48 h total |
|---|---|---|---|---|
| LONG after a ≥ 5 % drop from the 4-hour high | +3.09 / −1.51 | 3 · 67% · +1.3% | 33 · 24% · -19.4% | 36 · -18.0% |
| LONG after a ≥ 5 % drop from the 15-min high | +1.09 / −1.11 | 0 | 21 · 43% · -5.8% | 21 · -5.8% |
| LONG always in | +3.09 / −1.51 | 48 · 48% · -5.5% | 79 · 42% · +4.1% | 127 · -1.4% |
| LONG always in | +0.59 / −1.11 | 76 · 55% · -12.4% | 177 · 64% · -11.7% | 253 · -24.1% |
| SHORT after a ≥ 5 % bounce from the 4-hour low | +3.09 / −1.51 | 1 · 100% · +1.1% | 47 · 26% · -23.2% | 48 · -22.1% |
| SHORT always in | +3.09 / −1.51 | 49 · 51% · -3.6% | 78 · 33% · -30.1% | 127 · -33.7% |
| SHORT always in | +0.59 / −1.11 | 75 · 55% · -7.5% | 173 · 57% · -32.5% | 248 · -40.0% |
