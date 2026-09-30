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

## v2 — operator's setup (pre-registered, scripts/btc_spike_follow_study.py)
Primary: BTC +1.0 %/30 min AND 24 h-high breakout AND volume ≥ 3× median · top-10 tradeable alts by 24 h volume (Bull-Run universe)
· live LONG momentum exit (stop −0.70; runner arms at +0.40, floor max(peak − 1×ATR, +0.10); 4 h max) · units = events · 95 % CI by
bootstrap over events · control = same universe, same clock time 1–3 days earlier.
```
PRIMARY events (BTC +1.0 %/30 min, 24 h-high breakout, volume ≥ 3× median): 65  (2026-01-02 → 2026-09-21)

== PRIMARY: top-10 tradeable alts, live LONG momentum exit ==
entry at detection                                         events= 65 fills= 650 | per event -0.058 [95% -0.159,+0.043] positive  48% | fill WR  49% | Jan-Apr -0.094 (n39) May-Sep -0.003 (n26)
entry +15 min (late)                                       events= 65 fills= 650 | per event +0.015 [95% -0.089,+0.125] positive  37% | fill WR  56% | Jan-Apr -0.034 (n39) May-Sep +0.090 (n26)
entry +30 min (late)                                       events= 65 fills= 650 | per event -0.090 [95% -0.173,-0.004] positive  38% | fill WR  52% | Jan-Apr -0.058 (n39) May-Sep -0.138 (n26)
CONTROL: same universe, same clock time 1-3 days earlier   events= 65 fills=1940 | per event -0.060 [95% -0.123,+0.010] positive  37% | fill WR  50% | Jan-Apr -0.066 (n39) May-Sep -0.050 (n26)

== ROBUSTNESS (not the decision) ==
BTC +1.5 %, breakout + volume · top-10                     events= 34 fills= 340 | per event -0.078 [95% -0.200,+0.044] positive  41% | fill WR  48% | Jan-Apr -0.107 (n20) May-Sep -0.037 (n14)
BTC +1.0 %, volume only · top-10                           events=145 fills=1450 | per event -0.086 [95% -0.158,-0.012] positive  40% | fill WR  48% | Jan-Apr -0.118 (n82) May-Sep -0.044 (n63)
BTC +1.0 %, breakout only · top-10                         events= 66 fills= 660 | per event -0.053 [95% -0.150,+0.043] positive  48% | fill WR  49% | Jan-Apr -0.085 (n40) May-Sep -0.003 (n26)
BTC +1.0 %, no condition · top-10                          events=166 fills=1660 | per event -0.077 [95% -0.145,-0.004] positive  40% | fill WR  49% | Jan-Apr -0.098 (n95) May-Sep -0.048 (n71)
PRIMARY events · top-20                                    events= 65 fills=1300 | per event -0.048 [95% -0.149,+0.052] positive  42% | fill WR  48% | Jan-Apr -0.066 (n39) May-Sep -0.022 (n26)
PRIMARY events · top-60                                    events= 65 fills=3900 | per event -0.051 [95% -0.136,+0.037] positive  42% | fill WR  50% | Jan-Apr -0.087 (n39) May-Sep +0.003 (n26)

PRIMARY per event (top-10, entry at detection):
```

## This morning (2026-09-30), same rule on live Binance prices (scripts/btc_spike_today_check.py)
```
BTC events today (primary rule): ['12:35', '12:45', '12:50']
BTC bars with +1% in 30 min today: ['12:35', '12:40', '12:45', '12:50', '12:55']
top-10 tradeable by 24h volume now: ['SOLUSDT', 'ZECUSDT', 'QNTUSDT', 'SOXLUSDT', 'XRPUSDT', 'CLUSDT', 'SNDKUSDT', 'SPCXUSDT', 'NEARUSDT', 'HYPEUSDT']
using event 2026-09-30 12:35:00
delay_min             0      15     30
pair     in_top10                     
CLUSDT   True      0.307  0.939  0.593
HYPEUSDT True      0.155 -0.700 -0.570
MOVRUSDT False    -0.700  0.100  0.100
NEARUSDT True     -0.700 -0.700 -0.700
QNTUSDT  True     -0.700 -0.700 -0.700
SNDKUSDT True     -0.210 -0.393 -0.074
SOLUSDT  True      0.563  0.100 -0.400
SOXLUSDT True      0.100  0.100 -0.700
SPCXUSDT True     -0.578 -0.618 -0.651
XRPUSDT  True     -0.155 -0.700 -0.162
ZECUSDT  True      0.100  0.592  0.100
top-10, entry +0 min: avg -0.112  WR 50%  (n10)
top-10, entry +15 min: avg -0.208  WR 40%  (n10)
top-10, entry +30 min: avg -0.326  WR 20%  (n10)
```

Reading: over 65 past spikes the rule is indistinguishable from the control (−0.058 vs −0.060 per event; CI −0.16…+0.04);
entering 15 min late is the best variant (+0.015, CI −0.09…+0.13, only 37 % of events positive). This morning the mechanical
rule lost: −0.112 per fill at detection (WR 50 %), −0.208 at +15 min, −0.326 at +30 min (4 h exits completed; an earlier
run with the exits still open read +0.004). The operator's winners (NEAR runner +0.46, MOVR manual close
+0.72, and the two QNT runners the lock fix now banks at +0.10) came from PAIR SELECTION and manual timing inside one window —
not from 'buy the top alts when BTC spikes'. Next step = observe-first: manual trades now carry every entry stamp, so the
selection can be measured across spike windows (bar: ≥ 8 distinct windows before any rule).

## v3 — 1-minute bars, simulator validated on the operator's trades (scripts/btc_spike_follow_1m.py)
The 5m simulator mis-scored 3 of the operator's 7 spike trades (adverse-first inside a 5m bar that spans +0.9 % and −0.7 %). v3
walks 1m bars along the candle path (falling candle O→H→L→C, rising O→L→H→C) and reproduces the trades within exit-rule differences.
Result: the BTC spike alone still does not beat the control at any entry delay (0–45 min).

## ⚡ SURGE sleeve design grid (scripts/surge_sleeve_design.py · reports/SURGE_SLEEVE_GRID_2026-09-30.csv)
72 pre-declared cells (4 selections × 3 entry delays × 6 exits incl. the live Bull-Run exit at 2×/1× ATR trail), top-20 universe,
train Jan–Apr / test May–Sep, control = same cell 1 day earlier. Ship bar: positive in both halves, beats control in both, 95 % CI > 0,
≥ 20 events. PASSED: none.
```
          sel  entry         exit  events  fills  per_event            ci  train   test  ctrl_train  ctrl_test  pos_events  fill_wr PASS
HI_ATR_LEADER      0 BULLRUN_1ATR      55    117      0.320 [-0.15,+0.89]  0.413  0.199      -0.120      0.120          49       56     
HI_ATR_LEADER     10 BULLRUN_1ATR      55    117      0.278 [-0.21,+0.81]  0.138  0.458      -0.118      0.140          42       48     
HI_ATR_LEADER     10       KEEP60      55    117      0.263 [-0.04,+0.63]  0.234  0.300       0.172      0.281          45       52     
HI_ATR_LEADER      0 BULLRUN_2ATR      55    117      0.247 [-0.22,+0.82]  0.321  0.152      -0.138      0.081          51       56     
HI_ATR_LEADER      0          BOT      55    117      0.224 [-0.15,+0.71]  0.545 -0.192      -0.047     -0.055          44       56     
HI_ATR_LEADER     10 BULLRUN_2ATR      55    117      0.224 [-0.25,+0.75]  0.083  0.406      -0.101      0.108          42       48     
HI_ATR_LEADER     10        QUICK      55    117      0.221 [-0.09,+0.68]  0.353  0.050       0.171      0.078          47       47     
```
Best family = HI_ATR_LEADER (pair 5m ATR ≥ 1.5 % AND outrunning BTC over the spike's 30 min): top of the table for EVERY exit;
best cell entry at the spike + Bull-Run exit (1×ATR trail): +0.32 %/event, train +0.41 / test +0.20, control −0.12 / +0.12 — but CI
−0.15…+0.89 (117 fills / 55 events), WR 56 %, avg win +1.43 vs avg loss −1.20, max fill +11 %, and the top 3 events carry 113 % of
the total (the rest net negative; AGT/BTW/SIREN pumps ≈ 85 % of P&L). Fat-tailed: most spikes lose small, a few pumps pay.
Verdict: observe-first candidate (phantom tracking), not a sleeve.
