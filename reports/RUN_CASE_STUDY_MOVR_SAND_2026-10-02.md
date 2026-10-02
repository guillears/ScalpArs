# 📒 Case study — MOVR and SAND traded both ways with a tight stop and a trailing exit (in-sample)

## MOVRUSDT since 2026-09-29 20:00 UTC — price 1.1589 → high 3.34 → now 2.1708

### Stop 2 % · trail starts at +2 %, gives back 1.5 %

| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |
|---|---|---|---|---|---|---|
| 09-30 05:25 | LONG | staircase ON | 1.4383 | +0.64% | trail | 9 min |
| 09-30 11:50 | SHORT | close below EMA50 | 1.6718 | +0.68% | trail | 5 min |
| 09-30 16:15 | LONG | staircase ON | 1.7571 | -2.11% | stop | 5 min |
| 09-30 17:15 | SHORT | close below EMA50 | 1.6638 | +1.03% | trail | 4 min |
| 09-30 18:50 | LONG | staircase ON | 1.7545 | +2.46% | trail | 45 min |
| 10-01 01:30 | SHORT | close below EMA50 | 2.1647 | -2.11% | stop | 1 min |
| 10-01 04:40 | SHORT | close below EMA50 | 2.1867 | -2.11% | stop | 5 min |
| 10-01 08:50 | SHORT | close below EMA50 | 2.7158 | +5.60% | trail | 15 min |
| 10-01 12:10 | SHORT | close below EMA50 | 2.7659 | -2.11% | stop | 6 min |
| 10-01 15:05 | SHORT | close below EMA200 | 2.5221 | -2.11% | stop | 9 min |
| 10-01 17:40 | SHORT | close below EMA50 | 2.8994 | -2.11% | stop | 2 min |
| 10-01 19:25 | SHORT | close below EMA50 | 2.9272 | -2.11% | stop | 1 min |
| 10-01 21:50 | SHORT | close below EMA200 | 2.7792 | -2.11% | stop | 5 min |
| 10-02 01:05 | SHORT | close below EMA50 | 2.9248 | +2.25% | trail | 15 min |
| 10-02 03:20 | SHORT | close below EMA200 | 2.8072 | +0.46% | trail | 4 min |

**LONG: 3 trades · 2 won · total +1.0 % on price = +20 % of margin at 20× · stops 1**

**SHORT: 12 trades · 5 won · total -4.7 % on price = -95 % of margin at 20× · stops 7**

**Both sides: 15 trades · -3.8 % on price = -75 % of margin at 20×**

### Stop 3 % · trail starts at +3 %, gives back 2 %

| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |
|---|---|---|---|---|---|---|
| 09-30 05:25 | LONG | staircase ON | 1.4383 | -3.11% | stop | 15 min |
| 09-30 11:50 | SHORT | close below EMA50 | 1.6718 | +6.38% | trail | 18 min |
| 09-30 16:15 | LONG | staircase ON | 1.7571 | +2.73% | trail | 35 min |
| 09-30 17:15 | SHORT | close below EMA50 | 1.6638 | +1.13% | trail | 27 min |
| 09-30 18:50 | LONG | staircase ON | 1.7545 | +1.94% | trail | 46 min |
| 10-01 01:30 | SHORT | close below EMA50 | 2.1647 | -3.11% | stop | 4 min |
| 10-01 04:40 | SHORT | close below EMA50 | 2.1867 | +2.97% | trail | 7 min |
| 10-01 08:50 | SHORT | close below EMA50 | 2.7158 | +5.13% | trail | 15 min |
| 10-01 12:10 | SHORT | close below EMA50 | 2.7659 | -3.11% | stop | 6 min |
| 10-01 15:05 | SHORT | close below EMA200 | 2.5221 | -3.11% | stop | 20 min |
| 10-01 17:40 | SHORT | close below EMA50 | 2.8994 | -3.11% | stop | 23 min |
| 10-01 19:25 | SHORT | close below EMA50 | 2.9272 | -3.11% | stop | 8 min |
| 10-01 21:50 | SHORT | close below EMA200 | 2.7792 | -3.11% | stop | 12 min |
| 10-02 01:05 | SHORT | close below EMA50 | 2.9248 | +1.77% | trail | 15 min |
| 10-02 03:20 | SHORT | close below EMA200 | 2.8072 | +7.81% | trail | 12 min |

**LONG: 3 trades · 2 won · total +1.6 % on price = +31 % of margin at 20× · stops 1**

**SHORT: 12 trades · 6 won · total +6.5 % on price = +131 % of margin at 20× · stops 6**

**Both sides: 15 trades · +8.1 % on price = +162 % of margin at 20×**

### Stop 3 % · trail starts at +5 %, gives back 3 %

| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |
|---|---|---|---|---|---|---|
| 09-30 05:25 | LONG | staircase ON | 1.4383 | -3.11% | stop | 15 min |
| 09-30 11:50 | SHORT | close below EMA50 | 1.6718 | +5.47% | trail | 18 min |
| 09-30 16:15 | LONG | staircase ON | 1.7571 | -3.11% | stop | 56 min |
| 09-30 17:15 | SHORT | close below EMA50 | 1.6638 | -3.11% | stop | 38 min |
| 09-30 18:50 | LONG | staircase ON | 1.7545 | +11.77% | trail | 120 min |
| 10-01 01:30 | SHORT | close below EMA50 | 2.1647 | -3.11% | stop | 4 min |
| 10-01 04:40 | SHORT | close below EMA50 | 2.1867 | +3.31% | trail | 8 min |
| 10-01 08:50 | SHORT | close below EMA50 | 2.7158 | +4.70% | trail | 18 min |
| 10-01 12:10 | SHORT | close below EMA50 | 2.7659 | -3.11% | stop | 6 min |
| 10-01 15:05 | SHORT | close below EMA200 | 2.5221 | -3.11% | stop | 20 min |
| 10-01 17:40 | SHORT | close below EMA50 | 2.8994 | -3.11% | stop | 23 min |
| 10-01 19:25 | SHORT | close below EMA50 | 2.9272 | -3.11% | stop | 8 min |
| 10-01 21:50 | SHORT | close below EMA200 | 2.7792 | -3.11% | stop | 12 min |
| 10-02 01:05 | SHORT | close below EMA50 | 2.9248 | -3.11% | stop | 48 min |
| 10-02 03:20 | SHORT | close below EMA200 | 2.8072 | +6.91% | trail | 12 min |

**LONG: 3 trades · 1 won · total +5.6 % on price = +111 % of margin at 20× · stops 2**

**SHORT: 12 trades · 4 won · total -4.5 % on price = -90 % of margin at 20× · stops 8**

**Both sides: 15 trades · +1.1 % on price = +21 % of margin at 20×**

## SANDUSDT since 2026-10-02 05:00 UTC — price 0.04617 → high 0.07179 → now 0.06376

### Stop 2 % · trail starts at +2 %, gives back 1.5 %

| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |
|---|---|---|---|---|---|---|
| 10-02 09:05 | LONG | staircase ON | 0.06017 | +2.73% | trail | 11 min |
| 10-02 14:35 | SHORT | close below EMA50 | 0.06429 | +0.96% | trail | 16 min |

**LONG: 1 trades · 1 won · total +2.7 % on price = +55 % of margin at 20× · stops 0**

**SHORT: 1 trades · 1 won · total +1.0 % on price = +19 % of margin at 20× · stops 0**

**Both sides: 2 trades · +3.7 % on price = +74 % of margin at 20×**

### Stop 3 % · trail starts at +3 %, gives back 2 %

| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |
|---|---|---|---|---|---|---|
| 10-02 09:05 | LONG | staircase ON | 0.06017 | +2.21% | trail | 13 min |
| 10-02 14:35 | SHORT | close below EMA50 | 0.06429 | +0.71% | open | 23 min |

**LONG: 1 trades · 1 won · total +2.2 % on price = +44 % of margin at 20× · stops 0**

**SHORT: 1 trades · 1 won · total +0.7 % on price = +14 % of margin at 20× · stops 0**

**Both sides: 2 trades · +2.9 % on price = +58 % of margin at 20×**

### Stop 3 % · trail starts at +5 %, gives back 3 %

| Time (UTC) | Side | Signal | Entry | Result | Exit | Held |
|---|---|---|---|---|---|---|
| 10-02 09:05 | LONG | staircase ON | 0.06017 | +3.55% | trail | 23 min |
| 10-02 14:35 | SHORT | close below EMA50 | 0.06429 | +0.71% | open | 23 min |

**LONG: 1 trades · 1 won · total +3.5 % on price = +71 % of margin at 20× · stops 0**

**SHORT: 1 trades · 1 won · total +0.7 % on price = +14 % of margin at 20× · stops 0**

**Both sides: 2 trades · +4.3 % on price = +85 % of margin at 20×**

