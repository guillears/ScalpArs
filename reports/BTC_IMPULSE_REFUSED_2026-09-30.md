# BTC impulse: longs the BTC gates refused — 2026-09-30

Question (operator, after the Sep-30 12:30–13:00 UTC BTC +1.9 % push, no bot longs): when BTC impulses up, do alts follow — are the
BTC_RSI_ADX_CROSS / BTC_ADX_GATE_HIGH refusals leaving money on the table?

Instrument: replay decision journal (yr3, Jun–Sep, seed 1), BLOCK events of the two gates on LONGs, one episode per pair-hour,
priced with the refused-signals 'stack-lite' exit (scripts/gate_refused_signals_read.py). Calibration on the same months: sim avg
−0.020 vs live +0.013, sign agreement 84 % → RELATIVE read only. Market-wide gate → window units (2 h windows).

```
ALL refused by the two BTC gates (LONG)                        N=23462 windows(2h)=1241 days=119 WR= 54% avg=-0.081 per-window avg=-0.109 positive windows= 35% stops= 45%
BTC impulse up: BTC RSI >= 70 and BTC EMA20 slope > 0          N=1435 windows(2h)=152 days= 64 WR= 57% avg=-0.030 per-window avg=-0.119 positive windows= 41% stops= 42%
  ... and BTC ADX rising                                       N=1415 windows(2h)=151 days= 64 WR= 58% avg=-0.028 per-window avg=-0.113 positive windows= 42% stops= 42%
  ... BTC RSI >= 75                                            N= 677 windows(2h)= 93 days= 51 WR= 58% avg=-0.038 per-window avg=-0.113 positive windows= 42% stops= 42%
BTC RSI < 70 (other refusals)                                  N=22027 windows(2h)=1232 days=119 WR= 54% avg=-0.085 per-window avg=-0.098 positive windows= 35% stops= 45%
  impulse, month 05                                            N=  25 windows(2h)=  2 days=  1 WR= 76% avg=+0.068 per-window avg=+0.154 positive windows=100% stops= 20%
  impulse, month 06                                            N= 197 windows(2h)= 30 days= 14 WR= 55% avg=-0.134 per-window avg=-0.211 positive windows= 40% stops= 45%
  impulse, month 07                                            N= 345 windows(2h)= 37 days= 16 WR= 52% avg=-0.144 per-window avg=-0.245 positive windows= 30% stops= 46%
  impulse, month 08                                            N= 484 windows(2h)= 52 days= 21 WR= 58% avg=+0.031 per-window avg=-0.004 positive windows= 40% stops= 41%
  impulse, month 09                                            N= 384 windows(2h)= 31 days= 12 WR= 61% avg=+0.043 per-window avg=-0.090 positive windows= 55% stops= 39%
```

Longs the bot TOOK in the same sim: N 270 · WR 56 % · avg −0.034. Impulse-window refusals: −0.030 per signal, −0.119 per window,
41 % positive windows (152 windows / 64 days) — no edge; Aug–Sep per-signal slightly positive (+0.03 / +0.04) but per-window flat to
negative. Reading: by the time BTC RSI ≥ 70 the move is mostly done; a 'follow BTC' entry would have to fire at the START of the
impulse, which is a different rule (untested).
