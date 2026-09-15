# Bear-run SHORT sleeve — year-to-date study (2026-09-15)

Operator request: "run a longer period analysis since the beginning of the year to check the variables we will put, using the bull-run sleeve as reference for inverse filters." Scripts: scratchpad `bear/` (monitor.py, lib.py, study.py, study2-4.py). Data: Binance USDT-M 5m klines 2025-12-25 → 2026-09-15 for BTC and 15 liquid alts (SOL XRP DOGE ADA AVAX LINK SUI BNB LTC DOT NEAR HYPE XMR BCH AAVE).

## Method
- Regime monitor mirrored from the bull-run sleeve (r72 ≤ −5% / bars-below-EMA20 ≥ 56% / trend-efficiency ≥ 0.10, stay 4/53/0.10, squeeze latch r6h ≥ +3% or price > 1h EMA50), then a 24h-window variant swept over r24 {2,3,4} × share {0.50,0.56,0.62} × eff {0.10,0.15,0.20}.
- Entries: momentum-short proxy on 5m (bear EMA stack, EMA5−EMA20 gap ≤ −0.08%, RSI 25-50, pair ADX ≥ 20 rising, EMA20 slope ≤ −0.06%/3 bars, ATR ≥ 0.25%) and a rally-reject mirror of the dip-reclaim; random-entry control (6-12 seeds) in the same windows; monitor-OFF control.
- Exits: LIVE momentum-short stack (stop −0.70, ATR-widened to −1.2 floor, runner trail arm +0.4 / give-back 0.5×ATR capped 0.35×peak / lock +0.10, hard-TP ladder, 180-min sweep) and the bull-run BR set (same stop, BE arm +1.0 lock +0.2, trail 2×ATR). 5m bars, worst intrabar ordering (stop and trail floor both tested against the bar high, floor also against the close). Taker fees both sides. 4 slots, 2h pair spacing, 5-min loss cooldown. $12,000 notional per fill.
- Calibration: random longs all year = +0.005%/trade vs −0.09% expected → the replica flatters every fill by ≈ +0.10%. Sep-15 14-21h random shorts: 5m replica 81% / +0.28 vs 1m replica 84% / +0.30. All conclusions are read as rule-vs-random inside the same windows and ON-vs-OFF, never as absolute money.

## Findings
1. The 72h bull-run mirror does not fit bears: 101h ON, 12 windows ≥1h, covers only 5 of the 14 worst BTC days and 0 hours of Sep-15. Bear moves resolve inside a day; the 72h efficiency window arrives late.
2. A 24h-window monitor (r24 ≤ −4%, bars-below-EMA20 over 24h ≥ 50%, 24h efficiency ≥ 0.15, stay 3/47/0.15, squeeze latch) gives 96h ON, 14 windows ≥1h, covers 9 of the 14 worst days incl. Sep-15 (18:40-21:30). Neighbouring thresholds (r24 ≤ 3, share 0.56) behave the same way: 15-28 windows, signal 2-3× random.
3. Momentum-short proxy inside those windows (gates: BTC within 2% of its 24h low, pair below own ~1h EMA50): N 269 · 75% WR · +0.28%/fill · +$8,194 in windows (+$9,046 incl. sub-1h ON stretches) vs random-entry +$3,999; signal beats random in 9 of 14 windows; 9 of 14 windows positive (random also 9); biggest window 27% of the total; ex-biggest +$5,975 vs random +$3,133; Jan-Apr +$5,014 vs +$2,915, May-Sep +$3,180 vs +$1,084; optimism-adjusted (−0.10%/fill) +$4,966 vs random +$771. Exits: 200 trail / 67 stop / 2 sweep, median 10 min, 2.8 fills per ON hour.
4. Regime discrimination: same rule when the monitor is OFF = 4,269 fills · 66% · +0.071%/fill (≈ the replica's optimism, i.e. nothing) vs +0.19-0.28%/fill ON.
5. Exits: the bull-run BR set is WORSE for shorts in window units: 170 fills · 58% · +$2,602 · 4 of 14 windows positive · biggest window 66% · optimism-adjusted +$562. LIVE stack keeps 10 of 14 positive. Same ordering on the 72h monitor (BR 5/11 positive vs LIVE 9/11).
6. Gate ablation (candidate monitor, LIVE exits): off-24h-low ≤ 2%: keeps stops down (75 → 67), Σ flat, keep; ≤ 1% slightly better in-sample. Pair below own EMA50: neutral (keep as hygiene or drop). PVR ≤ 1.2: HURTS (+$9,046 → +$5,758, 10 → 8 positive windows) — do not port the bull-run PVR ceiling. BTC ADX ≥ 18 floor: neutral (N −13, Σ same) — keep for consistency with the long side. Breadth ≥ 0.5/0.6: neutral. Squeeze latch variants (2/3/5%, with/without EMA50): neutral. Rally-reject entry: fewer fills, no edge over random.

## Per-window table (candidate monitor, LIVE exits, base gates) — signal Σ$ | random-mean Σ$
-- per window, MOM entries, LIVE exits, gates {'off24lo': 2.0, 'pair_e50': True}   (signal Σ$ | random-mean Σ$)
  2026-01-20T19:35   4.7h BTC -1.12%  N   9 WR  67% Σ$    +79 | rnd   -114  stops 3
  2026-01-29T15:10  23.3h BTC -3.63%  N  35 WR  80% Σ$  +1764 | rnd   +665  stops 7
  2026-01-31T18:35  12.9h BTC -0.26%  N   4 WR 100% Σ$  +2163 | rnd   +637  stops 0
  2026-02-05T18:35   6.2h BTC -4.21%  N  25 WR  80% Σ$  +1040 | rnd   +836  stops 5
  2026-02-23T01:15   5.6h BTC +0.28%  N  10 WR  80% Σ$   +245 | rnd    -94  stops 2
  2026-02-28T06:30   3.7h BTC -1.56%  N  13 WR  77% Σ$   +140 | rnd   +302  stops 3
  2026-03-07T07:00   1.2h BTC +0.26%  N   1 WR   0% Σ$    -84 | rnd    -98  stops 1
  2026-03-18T14:55  22.3h BTC -3.20%  N  28 WR  61% Σ$   -161 | rnd   +741  stops 11
  2026-03-27T10:55  12.2h BTC -0.93%  N   8 WR  50% Σ$   -173 | rnd    +40  stops 4
  2026-06-02T02:05  26.9h BTC -5.82%  N  71 WR  80% Σ$  +2218 | rnd   +866  stops 14
  2026-06-18T15:55   3.0h BTC +0.21%  N   8 WR  62% Σ$   -171 | rnd    -84  stops 3
  2026-06-23T12:30   3.5h BTC +0.12%  N   7 WR  43% Σ$   -281 | rnd     -7  stops 4
  2026-06-24T16:50   2.4h BTC -0.67%  N  11 WR  91% Σ$   +508 | rnd    +70  stops 1
  2026-09-15T18:40   2.9h BTC -0.89%  N  17 WR  88% Σ$   +907 | rnd   +239  stops 1
  TOTAL N 269 WR 75% avg +0.280% Σ$ +8194 | random Σ$ +3999 · signal>random in 9/14 windows · positive windows 9 (random 9) · max window 27% · ex-max: signal +5975 vs random +3133
  split: Jan-Apr windows 9: signal +5014 vs random +2915 (pos 6) · May-Sep windows 5: signal +3180 vs random +1084 (pos 3)

## Caveats on record
- 5m-bar replica ≈ +0.10%/fill optimistic (random-long calibration); paper fills vs real books unmeasured.
- Fixed 15-pair universe ≠ the live scanner's top-N by volume (today's top-20 held XAU/XAG/SNDK/SOXL); the proxy signal ≠ the engine's full ladder (quality score, promo router, weak-cap not replicated).
- Threshold chosen from a 70-config sweep → selection bias; neighbours agree, but the first live windows are the out-of-sample test.
- 5 of 14 windows negative; window-level WR ≈ 65-70%, not the fill-level 75%.
- Slots are shared with momentum shorts; 2.8 fills/hour ON fits 4 slots at a 10-min median hold.

## Recommendation
Build `BEARRUN_SHORT` as a regime-conditional bypass of the four BTC short gates (RSI×ADX cross cells, ADX 24-30 kill band, 1h-RSI oversold floor, 5m slope gate) using the 24h monitor above, the LIVE momentum-short exits (not BR), BTC ADX ≥ 18 kept, off-24h-low ≤ 2% gate, DB-backed 2h pair spacing, coin-only universe with BTC/ETH blacklisted, 1× size. Ship observe-only first (PASS phantoms with stamped entries); arm bar = first 3 live windows ≥ 2 positive and Σ > 0; kill bar = first 10 fills WR ≤ 45% or Σ < 0, or 2 consecutive net-negative windows.
