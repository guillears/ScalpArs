# 🔥 Don't chase — the order-flow test across the year

58,548 long entries (30 s apart) in 1,527 episodes · 230 pairs · 239 days, with per-second taker-buy volume. Target-first rate (episode-weighted): Jan–Apr 54.3 % · May–Sep 51.7 % · needs 72 % to pay.

| Class (frozen from the MOVR session) | entries | episodes | target first Jan–Apr | May–Sep | net % / trade +0.59 / −1.11, by day [95 %] | +1.09 / −1.11 |
|---|---|---|---|---|---|---|
| CHASE (price up ∧ flow ≥ +10) | 19,622 | 1461 | 56.9% | 55.1% | -0.261 [-0.294, -0.228] | -0.337 [-0.376, -0.298] |
| PAUSE (price flat/down ∧ flow < 0) | 17,193 | 1397 | 55.2% | 52.7% | -0.297 [-0.328, -0.266] | -0.376 [-0.413, -0.340] |
| MILD (flow in [−10, 0)) | 5,356 | 1159 | 55.6% | 56.5% | -0.268 [-0.311, -0.225] | -0.309 [-0.361, -0.258] |
| REST | 21,733 | 1439 | 54.6% | 53.9% | -0.300 [-0.330, -0.270] | -0.366 [-0.403, -0.329] |
| ALL | 58,548 | 1527 | 54.3% | 51.7% | -0.321 [-0.345, -0.296] | -0.400 [-0.431, -0.369] |

**H1 (direction): PAUSE − CHASE target-first = -1.7 points Jan–Apr · -2.4 May–Sep → does NOT hold in both halves.**
H2 / H3 (tradable): read the PAUSE and MILD rows — positive only if the bracket is entirely above 0.

## Flow features by quintile (edges from Jan–Apr)

| Feature | target-first % by quintile, Jan–Apr (low → high) | May–Sep |
|---|---|---|
| flow_5s | 59 · 56 · 56 · 56 · 59 | 57 · 54 · 56 · 56 · 55 |
| flow_15s | 55 · 56 · 55 · 58 · 60 | 54 · 54 · 57 · 56 · 56 |
| flow_60s | 58 · 55 · 57 · 56 · 58 | 56 · 55 · 55 · 56 · 56 |
| flow_300s | 53 · 56 · 57 · 57 · 56 | 54 · 56 · 57 · 57 · 53 |
| vol_speed | 53 · 57 · 56 · 59 · 58 | 55 · 54 · 56 · 56 · 56 |
| trades_per_s | 53 · 57 · 58 · 59 · 58 | 54 · 54 · 55 · 56 · 57 |
| usd_15s | 54 · 54 · 56 · 59 · 60 | 53 · 55 · 56 · 56 · 60 |
| r15 | 56 · 56 · 56 · 55 · 58 | 57 · 53 · 55 · 55 · 58 |
