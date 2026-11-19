| block | policy | trades | WR | avg / trade (day-block 95 % CI) | Σ points |
|---|---|---|---|---|---|
| H2024 | wait for confirm (today) | 50 | 68% | +0.05 (-1.88 … +2.39) | +2 |
| H2024 | enter at signal | 79 | 66% | -0.07 (-1.67 … +1.72) | -5 |
| DESIGN | wait for confirm (today) | 570 | 74% | +0.90 (+0.30 … +1.53) | +514 |
| DESIGN | enter at signal | 875 | 70% | +0.32 (-0.13 … +0.72) | +283 |
| FWD | wait for confirm (today) | 4 | 100% | +4.94 (+nan … +nan) | +20 |
| FWD | enter at signal | 4 | 75% | +1.13 (+nan … +nan) | +5 |
| ALL | wait for confirm (today) | 624 | 74% | +0.86 (+0.27 … +1.46) | +536 |
| ALL | enter at signal | 958 | 70% | +0.29 (-0.13 … +0.71) | +282 |

| half (spike) | confirm: N · avg · Σ | signal: N · avg · Σ |
|---|---|---|
| 24-H1 | 2 · +2.57 · +5 | 3 · -2.35 · -7 |
| 24-H2 | 58 · +0.21 · +12 | 90 · +0.01 · +1 |
| 25-H1 | 59 · +1.54 · +91 | 95 · -0.24 · -23 |
| 25-H2 | 157 · -0.17 · -26 | 241 · +0.32 · +76 |
| 26-H1 | 202 · +0.69 · +140 | 277 · +0.82 · +226 |
| 26-H2 | 146 · +2.15 · +314 | 252 · +0.03 · +9 |

### Where the difference comes from (signal-entry trades, all blocks, by what the confirm path did with them)

| what confirm did | N | signal-entry avg | signal-entry Σ | confirm Σ on the same episodes |
|---|---|---|---|---|
| confirmed and traded | 624 | +2.19 | +1367 | +536 |
| confirmed but NEAR_BASE at confirm | 107 | +2.13 | +227 | +0 |
| never confirmed (SKIP_CONFIRM) | 147 | -8.15 | -1199 | +0 |
| higher high (SKIP_HH) | 80 | -1.42 | -114 | +0 |
| confirm-only (NEAR_BASE at the signal, not at confirm) | 0 | — | 0 | +0 |
