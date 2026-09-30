# BTC spike event study — do alts follow a BTC push? (2026-09-30)

Operator idea: a 'BTC-spike' sleeve — when BTC pushes up hard, buy the alts that will follow (e.g. MOVR, Sep-30: pair RSI 31, 3.6 %
under its EMA20, BTC RSI 78 → +0.72 % manual win). Event = BTC 30-min return crossing ≥ +TH % (≥ 4 h apart), Jan–Sep 2026, top-60
alts by volume, entry next 5m open, stack-lite exit over 2 h (SL −0.70, arm +0.40, exit max(0.10, 0.5×peak), fee 0.09).
Units = BTC events. Control = the same alts at the same clock time 24 h / 48 h earlier. scripts/btc_spike_event_study.py

## BTC +1.0 % in 30 min
```
BTC spike events (30-min ≥ +1.0%): 166
ALL alts at a BTC spike            fills= 9173 events=166 | per-event avg: 15m -0.106 30m -0.021 60m -0.037 stack -0.073 | events positive (stack)  42% | fill WR  51%
LAGGARD                            fills= 1067 events=166 | per-event avg: 15m -0.057 30m -0.142 60m -0.179 stack -0.092 | events positive (stack)  42% | fill WR  47%
FOLLOWER                           fills= 2834 events=165 | per-event avg: 15m -0.086 30m +0.019 60m +0.002 stack -0.029 | events positive (stack)  48% | fill WR  54%
LEADER                             fills= 5272 events=166 | per-event avg: 15m -0.121 30m -0.041 60m -0.052 stack -0.091 | events positive (stack)  43% | fill WR  50%
  LAGGARD month 06                 fills=  168 events= 31 | per-event avg: 15m -0.063 30m -0.122 60m -0.454 stack -0.086 | events positive (stack)  48% | fill WR  46%
  LAGGARD month 07                 fills=   80 events= 11 | per-event avg: 15m -0.219 30m -0.244 60m -0.314 stack -0.131 | events positive (stack)  27% | fill WR  40%
  LAGGARD month 08                 fills=  116 events= 13 | per-event avg: 15m -0.232 30m +0.030 60m -0.095 stack -0.155 | events positive (stack)  38% | fill WR  42%
  LAGGARD month 09                 fills=   59 events=  7 | per-event avg: 15m +0.130 30m +0.224 60m +0.269 stack -0.023 | events positive (stack)  43% | fill WR  53%
CONTROL: same alts, 24/48 h before fills=18337 events=330 | per-event avg: 15m -0.062 30m -0.081 60m -0.126 stack -0.084 | events positive (stack)  40% | fill WR  51%
```
## BTC +1.5 % in 30 min
```
BTC spike events (30-min ≥ +1.5%): 66
ALL alts at a BTC spike            fills= 3643 events= 66 | per-event avg: 15m -0.104 30m -0.057 60m -0.113 stack -0.021 | events positive (stack)  55% | fill WR  55%
LAGGARD                            fills=  307 events= 66 | per-event avg: 15m -0.076 30m -0.149 60m -0.264 stack -0.026 | events positive (stack)  52% | fill WR  50%
FOLLOWER                           fills= 1140 events= 65 | per-event avg: 15m -0.035 30m +0.022 60m -0.027 stack +0.004 | events positive (stack)  62% | fill WR  60%
LEADER                             fills= 2196 events= 66 | per-event avg: 15m -0.112 30m -0.049 60m -0.099 stack -0.017 | events positive (stack)  52% | fill WR  53%
  LAGGARD month 06                 fills=   61 events= 14 | per-event avg: 15m -0.162 30m -0.413 60m -0.297 stack -0.049 | events positive (stack)  50% | fill WR  44%
  LAGGARD month 07                 fills=   13 events=  3 | per-event avg: 15m -0.029 30m +0.169 60m +0.220 stack +0.030 | events positive (stack)  33% | fill WR  62%
  LAGGARD month 08                 fills=   25 events=  3 | per-event avg: 15m -0.665 30m -0.641 60m -0.946 stack +0.024 | events positive (stack)  67% | fill WR  60%
  LAGGARD month 09                 fills=   43 events=  4 | per-event avg: 15m +0.332 30m +0.639 60m +1.205 stack -0.000 | events positive (stack)  50% | fill WR  53%
CONTROL: same alts, 24/48 h before fills= 7282 events=132 | per-event avg: 15m -0.125 30m -0.127 60m -0.135 stack -0.155 | events positive (stack)  36% | fill WR  47%
```

Reading: after a BTC spike alts are no better than a random time at +1.0 % (−0.073 vs control −0.084 per event) and only
modestly better at +1.5 % (−0.021 vs −0.155; 55 % vs 36 % events positive) — never positive on average. LAGGARDS (the MOVR type)
do NOT catch up: 30/60-min returns are the worst of the three groups. FOLLOWERS at +1.5 % are the best cell (+0.004, 62 % events
positive, 65 events) — breakeven, not an edge. No sleeve from this. The Sep-30 spike itself is not in the cache yet.
