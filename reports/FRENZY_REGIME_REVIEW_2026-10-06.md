# FRENZY regime review: did the ORCA 18:05 loss come from a bearish market? (2026-10-06)

**Status:** research only, unreviewed (no caveman or deep review has run). No code, config or test was changed, and nothing was committed. Scripts and intermediate files are in the session scratchpad under `frr/` (`grid.py`, `parity.py`, `core.py`, `run.py`, `run2.py`, `extra.py`, `uni.py`, `on_tab.py`). The pre-registration (`frr/PREREG.txt`, copied in §9) was frozen before any outcome was split by regime.

## Bottom line (plain words)

1. **No regime variable separates FRENZY_LONG winners from losers, under either reading of "bearish".**
   - I tested 35 splits on the 204 live-like FRENZY_LONG fills, January 10 to September 27. Each was read at sign / label level first, then by terciles.
   - **0 of 35 pass.** A random re-labelling of days passes 2 on average (median 1).
   - **The bot's label reading:** fills labelled bullish (HEALTHY / STRONG / EXHAUSTED BULL) averaged **+0.47 %** (49 fills, 40 days). The rest averaged +0.32 %. If anything, the bullish label was slightly *better*.
   - **The "weak day" reading:** this is the pre-registered BEARDAY composite, BTC 1-day return < 0 ∧ BTC trend gap < 0.
     - On a bearish day FRENZY_LONG averaged **−0.06 %** (59 fills, 43 days, WR 47 %). Other days averaged **+0.53 %** (145 fills, 102 days).
     - The gap is **−0.59 %, CI [−1.64, +0.46]**. It points the same way in both halves of the year and in every leave-one-month-out run.
     - It is **not** beyond chance. The day-shuffle p is 0.10 one-sided and 0.34 two-sided.
     - So the operator's direction is visible in the data, but it is not established.
     - Read as the rolling 24 h BTC return instead, the sign reverses: BTC down over 24 h averaged +0.65 vs +0.00 when up. Two readings of "weak day" disagree.
2. **FRENZY does not move with the regime the way momentum longs do.** On 18 coarse splits, the sign of the FRENZY effect matches momentum longs on 8, which is a coin flip. FRENZY's bad months (Feb, Jul, Aug) and good months (Jan, Jun, Sep) cover both BTC crashes and BTC rallies.
3. **ORCA 18:05 was an ordinary FRENZY loss.** Its regime readings sit in states that averaged −0.06 % to +0.47 % per fill. **About 47 % of all FRENZY_LONG fills hit the −3 % stop in every regime state.** The bearish-day flag was on, and that state is about breakeven, not a loser cohort.
4. **The live run is cold, but not because of the regime.**
   - Live FRENZY_LONG has won 1 of 7 fills since Oct 3. If the year's 53 % win rate still held, that would happen about 4 % of the time.
   - Only ORCA 18:05 was on a bearish day. 5 of the other 6 lost on non-bearish days.
   - Four days of fills are about four market windows, so this is worth watching, not acting on.
5. **ON-scalp (operator addition): regime does tilt "enter on ON", but the tail stays ruinous in every regime.**
   - When BTC is rising at the ON bar, the no-stop TP +3 / 2 h exit turns positive on the strong subset S3. Examples: BTC 5m slope > 0 gives +0.45 %, CI [+0.03, +0.89]. BTC 1h slope > 0 gives +0.51 %. Breadth bull > bear gives +0.41 %.
   - This cluster is no stronger than luck. 3 states pass the four return legs, against a null median of 5 (p = 0.85). The separation count is 22 against a null mean of 17.5 (p = 0.27).
   - Above all, **P(−10 % within 12 h) is 49–59 % in every state of every split. The lowest is 48 %.** The pre-registered tail bar was ≤ 20 %, and **0 states meet it**.
   - There is no regime where "enter on ON, no stop" becomes safe.
6. **Recommendation: no change to FRENZY.** No regime filter, no regime sizing, and no probe.
   - Log BEARDAY as one free **observe-only** line, with the threshold frozen below. It is already stamped on every fill.
   - Promotion only through the locked expectancy bar on live fills (§8).

## 0. Cohort and parity (checked first)

| check | result |
|---|---|
| Signal bars = engine bars | The cohort is `FRENZY_ENGINE_COHORT_2026-10-05.csv`, built with the real `services.frenzy.frenzy_walk`. The revalidation report shows 8/8 live FRENZY/WIDE fills Oct 3–5 on the same bar and code. |
| Live-like universe + sequencing | I joined the overnight cohort (`live_elig`, `gvol_live_U2 < 1`, priced tick/1m). I then sequenced LONG together with the HOLD_GREEN WIDE using the reviewed sequencer (`rhg/base.py`: 2+2 slots, pair flat, ≤ 3 per pair-day). **FRENZY_LONG 204 fills · 132 days · WR 53.4 % · +0.359 %/fill (LOCK2, 1×). HOLD_GREEN WIDE 124 · 93 days · +0.43 %.** Both match the HOLD_GREEN review exactly. Breakeven WR for FRENZY_LONG = 47.9 % (avg win 3.39, avg loss 3.12). Stop rate 47 %. |
| Regime readings, causal | BTC 5m and 1h indicators come from `services.indicators.calculate_indicators` on the last 100 **closed** bars ≤ the signal-bar close. The label comes from `services.regime.classify_btc_regime`. The monitor-style readings (off-24h-high, above-24h-low, eff72, off-30d-high) and the 4h EMA50/200 gap use the engine formulas on closed bars. Klines are from public Binance REST. Breadth bull/bear % = the yr5 replay's SCAN line ≤ t (engine breadth). Market volume = `gvr_year.pkl`. |
| Rebuilt vs yr5 replay SCAN (78,625 five-minute points) | 5m slope corr 0.998 (sign agree 98 %), RSI 0.982, ADX 0.9998, eff72 0.996, off24h 0.996, off30d 0.9998. **Regime label agrees 96 %.** |
| Rebuilt vs live stamps (all 37 B16/B17/B18 fills) | label agree 76 % (10/12 on FRENZY fills); 5m slope corr 0.98 (sign 93 %); RSI 0.98; trend gap 0.995 (sign 100 %); 1h slope 0.92 (sign 93 %); 1h RSI 0.89; **1d return exact**; off24h 0.99; above-24h-low 0.99; off30d 0.98; 4h gap exact (7 stamped); eff72 0.99; ATR 0.998. The live scan reads the forming 5m bar, so labels near a threshold can differ (ORCA 09:40 CHOPPY_FLAT both; ENJ CHOPPY_WEAK live vs CHOPPY_FLAT rebuilt). **ORCA 18:05: rebuilt HEALTHY_BULL = live HEALTHY_BULL; 1d −0.885 = stamp; trend gap −0.229 vs −0.227.** |
| ON-scalp outcomes | Rebuilt from the study's per-signal file (`tX*`, `tY*`, `at60/at120`, `LOCK2`). They reproduce the study's headline. S0 TP1/60m/no-stop −0.09 (study −0.08), S3 TP3/2h/no-stop +0.14 (+0.15; TP overshoot), S0 TP1/SL3 −0.19 (−0.19), S0 LOCK2 −0.19 (−0.19), P(+1 ≤ 60 m) 81 %, P(−10 ≤ 12 h) 54 %. |

**Earlier FRENZY regime work, and which cohort it used:**

| study | cohort | FRENZY_LONG regime result | valid? |
|---|---|---|---|
| `FRENZY_LONG_SEPARATORS_2026-10-02` | research rule (pre-engine), BTC r1h/r4h/r24h/vs-EMA | none consistent across halves | **pre-parity** |
| `FRENZY_REGIME_2026-10-05` (34 splits, 196 FRENZY trades) | P2 "moments" cohort (`frenzy_scalp_moments_v2.pkl`), whose ON bar lagged the engine on ~60 % of signals | 0 / 34 pass, null 1.5 | **pre-parity**: listed as redo #10 in `FRENZY_REDO_PLAN`, never re-run until now |
| `FRENZY_LOSER_SEPARATOR_LEVERAGE_2026-10-05` | same lagged cohort | BTC 24h return T1 ≤ −0.48: −0.41 vs +0.28 (65 fills): the "weak day" lead | **pre-parity**. Re-tested here as rolling R24H (§2): reversed on the engine cohort |
| `FRENZY_WIDE_SLEEVE_CHECKLIST_2026-10-05`, `WIDE_FULL_QUANT_REVIEW_2026-10-06` | engine cohort | **WIDE only**; no macro switch beyond the null | engine, but not FRENZY_LONG |
| `FILTER_REGIME_MATRIX_2026-10-05` | momentum-long refusals | not FRENZY | — |

**So FRENZY_LONG had never been regime-tested on the engine cohort. This report is the first.**

## 1. FRENZY_LONG (204 fills): every split

"State · N · days · WR · mean" is the mean LOCK2 % per fill at 1×. The gap CI is a day-block bootstrap with one UTC day as the block, 2,000 reps. Half = Jan–Apr / May–Sep. LOMO = gap when one month is left out. PASS needs ALL of: N ≥ 15 and ≥ 8 days per state, gap CI excluding 0, the same sign in both halves and in every LOMO, and the worst day < 50 % of the losing state's loss.

| split | state A · N · days · WR · mean | state B · N · days · WR · mean | gap A−B [day CI] | gap Jan–Apr / May–Sep | LOMO gap range | worst day share of losing-state loss | verdict |
|---|---|---|---|---|---|---|---|
| REG bull-family vs rest | bull label · 49 · 40 d · 57 % · **+0.47** | other · 155 · 110 d · 52 % · **+0.32** | +0.15 [-0.89, +1.21] | +0.07 / +0.26 | -0.16 … +0.37 | 4 % | no |
| BULL% > BEAR% | bull>bear · 101 · 78 d · 57 % · **+0.42** | bull≤bear · 103 · 82 d · 50 % · **+0.30** | +0.12 [-0.86, +1.13] | -0.23 / +0.55 | -0.03 … +0.30 | 5 % | no |
| SLOPE5 sign | >0 · 103 · 80 d · 53 % · **+0.17** | ≤0 · 101 · 74 d · 53 % · **+0.55** | -0.38 [-1.34, +0.48] | -0.30 / -0.41 | -0.65 … -0.18 | 3 % | no |
| SLOPE1H sign | >0 · 95 · 78 d · 54 % · **+0.28** | ≤0 · 109 · 75 d · 53 % · **+0.42** | -0.14 [-1.12, +0.90] | -0.19 / -0.05 | -0.25 … +0.03 | 5 % | no |
| TGAP sign | >0 · 100 · 77 d · 57 % · **+0.50** | ≤0 · 104 · 79 d · 50 % · **+0.22** | +0.28 [-0.63, +1.17] | -0.19 / +0.78 | +0.16 … +0.41 | 5 % | no |
| R1D sign | >0 · 93 · 61 d · 54 % · **+0.41** | ≤0 · 111 · 71 d · 53 % · **+0.31** | +0.10 [-0.79, +1.03] | +0.65 / -0.52 | -0.35 … +0.54 | 6 % | no |
| GAP4H sign | >0 · 96 · 62 d · 54 % · **+0.41** | ≤0 · 108 · 72 d · 53 % · **+0.31** | +0.10 [-0.82, +1.08] | +0.80 / -0.47 | -0.26 … +0.35 | 6 % | no |
| ETH_R1D sign | >0 · 96 · 64 d · 51 % · **+0.27** | ≤0 · 108 · 68 d · 56 % · **+0.44** | -0.16 [-1.06, +0.79] | +0.40 / -0.72 | -0.51 … +0.24 | 6 % | no |
| RSI5 >50 | >50 · 106 · 82 d · 53 % · **+0.17** | ≤50 · 98 · 73 d · 54 % · **+0.56** | -0.39 [-1.39, +0.54] | -0.26 / -0.46 | -0.73 … -0.11 | 5 % | no |
| RSI1H >50 | >50 · 93 · 76 d · 53 % · **+0.23** | ≤50 · 111 · 75 d · 54 % · **+0.47** | -0.23 [-1.23, +0.75] | +0.11 / -0.52 | -0.42 … -0.05 | 5 % | no |
| EFF72 chop ≤0.007 | chop · 20 · 20 d · 45 % · **+0.64** | trend · 184 · 121 d · 54 % · **+0.33** | +0.31 [-1.84, +2.45] | +1.96 / -2.71 | -1.56 … +1.06 | 3 % | no |
| OFF30D washed ≤−15 | washed · 27 · 22 d · 56 % · **+0.30** | not · 177 · 110 d · 53 % · **+0.37** | -0.07 [-1.32, +1.32] | -0.45 / +0.39 | -0.23 … +0.56 | 11 % | no |
| OFF24H median | >-1.23 · 97 · 77 d · 53 % · **+0.20** | ≤-1.23 · 107 · 73 d · 54 % · **+0.50** | -0.30 [-1.38, +0.73] | -0.01 / -0.50 | -0.45 … -0.15 | 5 % | no |
| ABOVE24LO median | >1.27 · 92 · 70 d · 59 % · **+0.73** | ≤1.27 · 112 · 72 d · 49 % · **+0.05** | +0.68 [-0.31, +1.63] | +0.64 / +0.66 | +0.37 … +0.93 | 5 % | no |
| OFF30D median | >-6.07 · 102 · 68 d · 53 % · **+0.36** | ≤-6.07 · 102 · 69 d · 54 % · **+0.36** | -0.01 [-0.92, +0.92] | +1.10 / -1.00 | -0.71 … +0.30 | 6 % | no |
| ATR5 median | >0.145 · 94 · 70 d · 53 % · **+0.39** | ≤0.145 · 110 · 79 d · 54 % · **+0.33** | +0.05 [-0.89, +0.98] | -0.15 / +0.09 | -0.25 … +0.26 | 6 % | no |
| GVOL median | >0.845 · 52 · 48 d · 58 % · **+0.86** | ≤0.845 · 152 · 109 d · 52 % · **+0.19** | +0.67 [-0.46, +1.91] | +0.15 / +1.24 | +0.31 … +0.98 | 4 % | no |
| BEARDAY (R1D<0 ∧ TGAP<0) | bearish day · 59 · 43 d · 47 % · **-0.06** | not · 145 · 102 d · 56 % · **+0.53** | -0.59 [-1.64, +0.46] | -0.78 / -0.39 | -0.86 … -0.39 | 8 % | no |
| BULL T3 vs T1 | T3>52.1 · 76 · 58 d · 57 % · **+0.47** | T1≤24.4 · 59 · 51 d · 54 % · **+0.73** | -0.25 [-1.58, +1.00] | -0.34 / -0.10 | -0.58 … -0.03 | 8 % | no |
| BEAR T3 vs T1 | T3>58.3 · 58 · 50 d · 55 % · **+0.79** | T1≤29.5 · 72 · 58 d · 54 % · **+0.27** | +0.52 [-0.70, +1.77] | +0.20 / +0.87 | +0.16 … +1.00 | 7 % | no |
| SLOPE5 T3 vs T1 | T3>0.0208 · 59 · 47 d · 58 % · **+0.54** | T1≤-0.0207 · 57 · 48 d · 53 % · **+0.53** | +0.01 [-1.31, +1.25] | -0.08 / +0.13 | -0.41 … +0.34 | 8 % | no |
| SLOPE1H T3 vs T1 | T3>0.079 · 65 · 54 d · 48 % · **-0.04** | T1≤-0.0833 · 67 · 52 d · 46 % · **-0.15** | +0.11 [-1.16, +1.36] | -0.11 / +0.38 | -0.09 … +0.43 | 6 % | no |
| TGAP T3 vs T1 | T3>0.0744 · 69 · 54 d · 59 % · **+0.66** | T1≤-0.0766 · 60 · 50 d · 55 % · **+0.73** | -0.07 [-1.39, +1.23] | -0.51 / +0.43 | -0.26 … +0.14 | 6 % | no |
| RSI5 T3 vs T1 | T3>55.4 · 61 · 49 d · 61 % · **+0.69** | T1≤44.7 · 61 · 51 d · 48 % · **+0.04** | +0.65 [-0.56, +1.80] | +0.82 / +0.45 | +0.30 … +1.05 | 7 % | no |
| RSI1H T3 vs T1 | T3>55.7 · 59 · 48 d · 51 % · **+0.08** | T1≤44.5 · 68 · 53 d · 46 % · **-0.02** | +0.10 [-1.24, +1.45] | +0.24 / +0.04 | -0.18 … +0.27 | 6 % | no |
| R1D T3 vs T1 | T3>0.621 · 60 · 41 d · 55 % · **+0.24** | T1≤-0.704 · 77 · 47 d · 53 % · **+0.29** | -0.05 [-1.04, +1.00] | +0.44 / -0.65 | -0.55 … +0.26 | 11 % | no |
| OFF24H T3 vs T1 | T3>-0.79 · 66 · 55 d · 50 % · **+0.11** | T1≤-1.89 · 64 · 49 d · 52 % · **+0.21** | -0.10 [-1.38, +1.21] | +0.47 / -0.21 | -0.45 … +0.10 | 7 % | no |
| ABOVE24LO T3 vs T1 | T3>1.85 · 51 · 40 d · 57 % · **+0.31** | T1≤0.871 · 79 · 59 d · 49 % · **+0.08** | +0.23 [-0.91, +1.33] | +0.61 / -0.41 | +0.10 … +0.37 | 7 % | no |
| OFF30D T3 vs T1 | T3>-4.12 · 68 · 47 d · 51 % · **+0.42** | T1≤-9.26 · 51 · 37 d · 53 % · **+0.12** | +0.29 [-0.95, +1.54] | +1.19 / -0.67 | -0.07 … +0.54 | 11 % | no |
| GAP4H T3 vs T1 | T3>1.07 · 67 · 43 d · 51 % · **+0.26** | T1≤-2.19 · 60 · 39 d · 53 % · **+0.12** | +0.13 [-1.00, +1.28] | +1.00 / -0.49 | -0.22 … +0.42 | 12 % | no |
| EFF72 T3 vs T1 | T3>0.0438 · 66 · 49 d · 53 % · **+0.22** | T1≤0.0226 · 69 · 53 d · 41 % · **-0.27** | +0.49 [-0.69, +1.78] | -0.55 / +1.74 | +0.01 … +0.87 | 6 % | no |
| ATR5 T3 vs T1 | T3>0.186 · 55 · 44 d · 47 % · **-0.06** | T1≤0.115 · 78 · 58 d · 58 % · **+0.65** | -0.71 [-1.78, +0.47] | -0.72 / -1.43 | -1.19 … -0.52 | 8 % | no |
| ETH_R1D T3 vs T1 | T3>0.755 · 59 · 40 d · 61 % · **+0.68** | T1≤-0.878 · 71 · 46 d · 51 % · **+0.17** | +0.51 [-0.55, +1.65] | +0.95 / -0.15 | -0.34 … +0.78 | 8 % | no |
| GVOL T3 vs T1 | T3>1.03 · 15 · 14 d · 47 % · **-0.64** | T1≤0.701 · 99 · 81 d · 51 % · **+0.11** | -0.75 [-2.27, +0.74] | -1.34 / -0.10 | -1.05 … -0.44 | 14 % | no |

**The bot's own label, in full** (descriptive; ORCA 18:05 was HEALTHY_BULL):

| label | N | days | WR | mean %/fill | stop rate |
|---|---|---|---|---|---|
| CHOPPY_FLAT | 59 | 53 | 54 % | +0.35 | 46 % |
| CHOPPY_WEAK | 49 | 44 | 45 % | −0.30 | 55 % |
| STRONG_BEAR | 26 | 24 | 62 % | +1.23 | 38 % |
| STRONG_BULL | 23 | 21 | 65 % | +0.73 | 35 % |
| HEALTHY_BULL | 21 | 20 | 48 % | −0.07 | 52 % |
| HEALTHY_BEAR | 20 | 19 | 50 % | +0.44 | 50 % |
| BULL_EXHAUSTED / BEAR_EXHAUSTED | 5 / 1 | 5 / 1 | — | +1.56 / +3.55 | — |
| **groups:** bull 49 · bear 47 · chop 108 | | 40 · 40 · 92 | 57 · 57 · 50 % | **+0.47 · +0.94 · +0.05** | 43 · 43 · 50 % |

- Bear-labelled fills did best, so the label's *direction* does not hurt FRENZY.
- Reading the table after the fact, "trending vs chop" stands out: trending +0.70 (96 fills) vs chop +0.05 (108 fills). Gap +0.65, CI [−0.32, +1.65], both halves positive, LOMO +0.44…+0.98, day-shuffle p 0.20.
- This split was found by looking, so it is a watch item only. It is not a candidate.

### Family-wise null (shuffle the regime readings by day)

Each round replaces every day's regime readings with another day of the **same month** at the same time of day. The outcomes and fills stay fixed. Every cell is then re-read with the same statistic and legs. 500 rounds.

| family | observed | null mean · median · 95th pct | p |
|---|---|---|---|
| FRENZY_LONG splits passing all legs (35 cells) | **0** | 2.1 · 1 · 7 | 1.00 |
| FRENZY_LONG splits passing all legs **except** the CI (direction-consistent) | 11 | 14.0 · 14 · 21 | 0.80 |
| largest \|z\| among the direction-consistent FRENZY_LONG splits | 1.43 | 2.26 median | 0.96 |
| BEARDAY gap z (pre-registered single cell) | −1.09 | +0.33 mean | **0.10 one-sided** · 0.34 two-sided |
| Full family: FRENZY_LONG + ON-scalp S0/S3 × 7 outcomes (510 cells) passing all legs | 22 (all ON-scalp) | 17.5 · 16 · ~38 | 0.27 |

### HOLD_GREEN WIDE (124 fills, secondary)

- **0 of 35 pass.** The largest gaps:
  - BTC 1h RSI > 50: +0.96 vs +0.01, CI [−0.21, +2.08]
  - 4h EMA50 > EMA200: +0.06 vs +0.83, CI [−2.01, +0.32]
  - BTC off its 24 h high, above the median: +0.86 vs −0.02
- None clears its CI. On a bearish day HOLD_GREEN did **better**: +0.69 (30 fills) vs +0.35.
- Full table: `frr/show.py C2`.

## 2. The "bearish day" reading (BEARDAY = BTC 1d < 0 ∧ BTC trend gap < 0) against the locked bar

| | N | days | WR | mean %/fill | day-clustered 95 % CI | worst day / worst pair share of loss |
|---|---|---|---|---|---|---|
| FRENZY_LONG on a bearish day (the cohort a filter would block) | 59 | 43 | 47.5 % | **−0.06** | [−0.97, +0.80]; P(mean ≥ 0) = 0.43 | 8 % / 8 % (BANANA −6.2, C −6.2) |
| FRENZY_LONG other days | 145 | 102 | 56 % | +0.53 | — | — |
| bearish day, by month | Jan +1.72 (5) · Feb −3.11 (1) · Mar −0.22 (18) · Apr −0.45 (6) · May −0.67 (7) · Jun +1.79 (3) · Jul −2.13 (7) · Aug +1.10 (6) · Sep +0.87 (6) | | | | | |

Expectancy bar (locked, Sep-25):

| leg | value | result |
|---|---|---|
| ① WR below breakeven | 47.5 % vs 47.9 % | pass, barely |
| ② mean < 0 at 95 %, window-clustered | P(mean ≥ 0) = 0.43 | **FAIL** |
| ③ ≥ 8 windows, no window ≥ 50 % of the loss | 43 days, worst day 8 % | pass |
| ④ N ≥ 15 | 59 | pass |

**Verdict: not a filter.** Even if it were real, the blocked cohort is about breakeven. Blocking it would add about +0.06 × 59 ≈ +3.5 % of trade-sum a year, or about +2 % after the 30–50 % haircut. The other side (+0.53, WR 56 %) is far from the multiplier bar (WR ≥ 70 %).

Neither leg works alone, which is an interaction warning:
- BTC 1d return: T3 +0.24 vs T1 +0.29.
- Trend gap: T3 +0.66 vs T1 +0.73.

The rolling 24 h BTC return (R24H) was added **after** the pre-registration, because it is the pre-parity "weak day" lead. On the engine cohort it points the **opposite** way to that lead:
- BTC **up** over 24 h: +0.00 (91 fills, 74 days).
- BTC **down** over 24 h: **+0.65** (113 fills, 72 days).
- Gap (up − down) −0.64, CI [−1.69, +0.39], both halves negative (−0.61 / −0.64).
- The terciles do not agree: T3 +0.14 vs T1 +0.19.

So "BTC weak on the day" is *worse* for FRENZY when read as the daily bar together with the trend gap (BEARDAY), and *better* when read as the rolling 24 h return. Two readings of the same idea give opposite signs, and the terciles are non-monotone. That is the signature of a confound, not of a regime effect. The pre-parity lead (BTC 24 h T1 −0.41 vs +0.28) does not survive on the engine cohort.

## 3. Washed-out window (Jun-18 → Jul-2) with and without

| | N | days | WR | mean | bearish day vs not | bull label vs other | BTC 1d > 0 vs ≤ 0 |
|---|---|---|---|---|---|---|---|
| all | 204 | 132 | 53.4 % | +0.36 | −0.06 vs +0.53 | +0.47 vs +0.32 | +0.41 vs +0.31 |
| without the window | 196 | 126 | 52.6 % | +0.32 | −0.21 vs +0.54 (gap −0.75 [−1.86, +0.39], halves −0.78 / −0.71) | +0.28 vs +0.33 | +0.45 vs +0.21 |
| window only | 8 | 6 | 75 % | +1.28 | — | — | — |

- Removing the window sharpens BEARDAY a little. It still does not clear its CI.
- Removing the window removes the bull-label edge entirely.
- ATR T3 vs T1 without the window: −0.19 vs +0.70, CI [−2.03, +0.31]. That is an ATR-of-BTC lead (high BTC 5m ATR is worse for FRENZY), and it also fails its CI.

## 4. Per month and tape context (is it the regime, or the sleeve?)

| month | N | days | WR | mean | stop rate | bearish-day share | bull-label share | BTC off 30d high (mean) | BTC month return |
|---|---|---|---|---|---|---|---|---|---|
| Jan | 25 | 16 | 64 % | **+1.87** | 36 % | 20 % | 20 % | −7.2 % | −10.2 % |
| Feb | 13 | 10 | 46 % | −0.38 | 54 % | 8 % | 31 % | −26.4 % | −15.0 % |
| Mar | 45 | 24 | 51 % | +0.27 | 49 % | 40 % | 16 % | −8.0 % | +2.0 % |
| Apr | 22 | 12 | 55 % | +0.01 | 45 % | 27 % | 32 % | −5.5 % | +11.8 % |
| May | 30 | 21 | 53 % | +0.42 | 47 % | 23 % | 20 % | −5.6 % | −3.5 % |
| Jun | 11 | 9 | 64 % | +0.54 | 36 % | 27 % | 36 % | −19.6 % | −20.4 % |
| Jul | 19 | 16 | 37 % | **−0.97** | 63 % | 37 % | 37 % | −4.8 % | +7.3 % |
| Aug | 21 | 14 | 48 % | −0.15 | 52 % | 29 % | 14 % | −4.1 % | +25.0 % |
| Sep | 18 | 10 | 67 % | **+1.23** | 33 % | 33 % | 33 % | −3.9 % | +6.4 % |

- The best months are Jan (BTC −10 %), Sep (+6 %) and Jun (−20 %).
- The worst months are Jul (+7 %), Feb (−15 %) and Aug (+25 %).
- **Good and bad months both include BTC crashes and BTC rallies.** The bearish-day share is about the same in good and bad months (20–37 %).
- What changes between months is the **stop rate** (33–36 % vs 52–63 %), not the market state. This repeats the pre-parity study's finding on the engine's own bars.
- Concentration: **the top 5 pairs carry 90 % of the year's net** (BERA, PHA, AXS, DEGO, G). FRENZY_LONG's year is a handful of runners.

### Uniform-degradation check (FRENZY vs momentum longs, same regime readings)

The momentum longs are the kept MOMENTUM LONG fills of `MASTER_POOL_stacked.csv`: 287 fills, 70 days, Jun 18 → Oct 2, price % per fill. The FRENZY gap is shown full-year and on the overlapping Jun 18 → Sep window.

| split | MOM-long gap (A−B) [day CI] · N A/B | FRENZY full-year gap | FRENZY Jun-18→Sep gap | same direction? |
|---|---|---|---|---|
| REG bull-family vs rest | +0.09 [-0.14, +0.25] · 202/85 | +0.15 | +0.77 | yes |
| BULL% > BEAR% | +0.25 [+0.03, +0.43] · 231/56 | +0.12 | +0.95 | yes |
| SLOPE5 sign | +0.23 [+0.07, +0.40] · 237/50 | -0.38 | -0.32 | no |
| SLOPE1H sign | +0.13 [-0.06, +0.29] · 144/143 | -0.14 | -0.27 | no |
| TGAP sign | +0.04 [-0.17, +0.26] · 182/105 | +0.28 | +0.64 | yes |
| R1D sign | +0.18 [-0.02, +0.36] · 135/152 | +0.10 | -0.57 | yes |
| GAP4H sign | -0.14 [-0.42, +0.06] · 170/117 | +0.10 | -0.38 | no |
| ETH_R1D sign | +0.09 [-0.17, +0.28] · 189/98 | -0.16 | -1.06 | no |
| RSI5 >50 | +0.21 [-0.12, +0.44] · 245/42 | -0.39 | -0.39 | no |
| RSI1H >50 | +0.13 [-0.06, +0.33] · 155/132 | -0.23 | -0.16 | no |
| EFF72 chop ≤0.007 | -0.13 [-0.46, +0.25] · 25/262 | +0.31 | -2.00 | no |
| OFF30D washed ≤−15 | +0.66 [+0.43, +0.88] · 19/268 | -0.07 | +1.27 | no |
| OFF24H median | +0.10 [-0.09, +0.31] · 173/114 | -0.30 | -0.38 | no |
| ABOVE24LO median | +0.17 [+0.01, +0.34] · 127/160 | +0.68 | +0.12 | yes |
| OFF30D median | -0.39 [-0.63, -0.16] · 246/41 | -0.01 | -0.91 | yes |
| ATR5 median | -0.01 [-0.16, +0.18] · 122/165 | +0.05 | -0.22 | no |
| GVOL median | +0.00 [-0.17, +0.19] · 158/119 | +0.67 | +0.74 | yes |
| BEARDAY (R1D<0 ∧ TGAP<0) | -0.19 [-0.42, +0.04] · 60/227 | -0.59 | +0.14 | yes |

- **The signs agree on 8 of 18 splits, which is chance.**
- The clearest disagreement is BTC's *immediate* direction. Momentum longs want BTC rising now: 5m slope > 0 gives +0.23, CI [+0.07, +0.40]. FRENZY_LONG leans the other way: −0.38 full-year, CI spans 0.
- FRENZY's results do not rise and fall with the market state the way the momentum engine's do. Under checklist item ③, there is no common regime factor to hunt here. FRENZY's variance looks sleeve-specific: the stop rate and a few runner pairs.

## 5. ON-scalp ("enter immediately on ON"): does regime make it work? (operator addition)

**Pre-registered (§9) before reading:**
- Outcomes:
  - hit rates P(+1 % net ≤ 60 min), P(+3 % ≤ 2 h), P(−10 % ≤ 12 h)
  - exits TP1/60m/no-stop, TP3/2h/no-stop (the frozen observe line), TP1/60m/SL3, LOCK2
- Subsets: S0 = all 1,443 ON bars; S3 = the live strong flag (ADX rising ∧ +DI > −DI), 893 bars.
- Same splits, all counted in one family with FRENZY_LONG.
- "Regime makes it work" requires a state with all of:
  - mean > 0 with day-CI low > 0
  - both halves > 0
  - LOMO min > 0
  - **P(−10 % ≤ 12 h) ≤ 20 % with CI upper ≤ 25 %**
  - survival of the family null.

**Result: 0 states pass. The tail leg fails everywhere.** The lowest P(−10 % ≤ 12 h) in any state of any split is **48 %** (S0) / **49 %** (S3). Main splits:

| subset | split | state | N | days | P(+1 % ≤60 m) | P(+3 % ≤2 h) | P(−10 % ≤12 h) | TP1/60m/no-stop | TP3/2h/no-stop [day CI] | TP1/60m/SL3 | LOCK2 | TP3/2h worst · P(≤−5 %) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S0 | REG bull-family vs rest | bull label | 414 | 188 | 83 % | 66 % | 54 % | +0.13 | +0.08 [-0.46, +0.59] | -0.18 | -0.09 | -31.4 · 15 % |
| S0 | REG bull-family vs rest | other | 1029 | 247 | 80 % | 64 % | 54 % | -0.18 | -0.16 [-0.50, +0.16] | -0.20 | -0.23 | -38.7 · 16 % |
| S0 | BULL% > BEAR% | bull>bear | 709 | 238 | 83 % | 66 % | 52 % | +0.08 | +0.15 [-0.22, +0.50] | -0.13 | -0.15 | -31.4 · 14 % |
| S0 | BULL% > BEAR% | bull≤bear | 734 | 233 | 80 % | 63 % | 56 % | -0.26 | -0.33 [-0.77, +0.09] | -0.25 | -0.24 | -38.7 · 18 % |
| S0 | SLOPE5 sign | >0 | 740 | 232 | 83 % | 67 % | 53 % | +0.10 | +0.18 [-0.18, +0.52] | -0.13 | -0.11 | -31.4 · 14 % |
| S0 | SLOPE5 sign | ≤0 | 703 | 232 | 79 % | 62 % | 55 % | -0.29 | -0.39 [-0.82, +0.03] | -0.26 | -0.28 | -38.7 · 18 % |
| S0 | SLOPE1H sign | >0 | 739 | 205 | 83 % | 67 % | 49 % | +0.01 | +0.18 [-0.28, +0.59] | -0.18 | -0.18 | -31.4 · 14 % |
| S0 | SLOPE1H sign | ≤0 | 704 | 185 | 79 % | 62 % | 59 % | -0.20 | -0.38 [-0.78, -0.01] | -0.21 | -0.20 | -38.7 · 18 % |
| S0 | TGAP sign | >0 | 731 | 226 | 81 % | 65 % | 52 % | -0.03 | +0.04 [-0.34, +0.43] | -0.19 | -0.21 | -31.4 · 14 % |
| S0 | TGAP sign | ≤0 | 712 | 225 | 81 % | 63 % | 56 % | -0.15 | -0.24 [-0.62, +0.12] | -0.19 | -0.18 | -38.7 · 18 % |
| S0 | R1D sign | >0 | 725 | 125 | 82 % | 64 % | 53 % | -0.09 | -0.14 [-0.61, +0.30] | -0.18 | -0.24 | -38.7 · 17 % |
| S0 | R1D sign | ≤0 | 718 | 130 | 80 % | 64 % | 56 % | -0.09 | -0.05 [-0.39, +0.26] | -0.20 | -0.14 | -37.3 · 15 % |
| S0 | GAP4H sign | >0 | 706 | 122 | 81 % | 64 % | 56 % | -0.15 | -0.22 [-0.64, +0.16] | -0.20 | -0.30 | -31.4 · 18 % |
| S0 | GAP4H sign | ≤0 | 737 | 138 | 82 % | 65 % | 52 % | -0.03 | +0.03 [-0.37, +0.39] | -0.18 | -0.09 | -38.7 · 14 % |
| S0 | RSI5 >50 | >50 | 747 | 235 | 82 % | 65 % | 53 % | +0.06 | +0.08 [-0.29, +0.42] | -0.17 | -0.19 | -31.4 · 14 % |
| S0 | RSI5 >50 | ≤50 | 696 | 231 | 80 % | 63 % | 55 % | -0.25 | -0.29 [-0.73, +0.12] | -0.22 | -0.20 | -38.7 · 18 % |
| S0 | ABOVE24LO median | >1.27 | 692 | 182 | 83 % | 67 % | 52 % | +0.05 | +0.24 [-0.19, +0.67] | -0.16 | -0.12 | -31.4 · 12 % |
| S0 | ABOVE24LO median | ≤1.27 | 751 | 185 | 79 % | 62 % | 56 % | -0.22 | -0.41 [-0.81, -0.03] | -0.22 | -0.26 | -38.7 · 19 % |
| S0 | OFF30D washed ≤−15 | washed | 243 | 58 | 83 % | 67 % | 53 % | +0.01 | +0.29 [-0.32, +0.85] | -0.19 | -0.09 | -37.3 · 9 % |
| S0 | OFF30D washed ≤−15 | not | 1200 | 201 | 81 % | 64 % | 54 % | -0.11 | -0.17 [-0.50, +0.14] | -0.19 | -0.21 | -38.7 · 17 % |
| S0 | EFF72 chop ≤0.007 | chop | 151 | 78 | 82 % | 62 % | 50 % | -0.17 | -0.77 [-1.97, +0.30] | -0.08 | -0.17 | -37.3 · 18 % |
| S0 | EFF72 chop ≤0.007 | trend | 1292 | 253 | 81 % | 65 % | 55 % | -0.08 | -0.02 [-0.30, +0.25] | -0.20 | -0.19 | -38.7 · 16 % |
| S0 | BEARDAY (R1D<0 ∧ TGAP<0) | bearish day | 358 | 113 | 76 % | 62 % | 58 % | -0.33 | -0.36 [-0.86, +0.13] | -0.36 | -0.26 | -37.3 · 18 % |
| S0 | BEARDAY (R1D<0 ∧ TGAP<0) | not | 1085 | 237 | 83 % | 65 % | 53 % | -0.01 | -0.01 [-0.34, +0.31] | -0.14 | -0.17 | -38.7 · 15 % |
| S3 | REG bull-family vs rest | bull label | 254 | 148 | 85 % | 70 % | 51 % | +0.21 | +0.40 [-0.27, +0.99] | -0.19 | +0.08 | -31.4 · 13 % |
| S3 | REG bull-family vs rest | other | 639 | 231 | 82 % | 67 % | 54 % | -0.09 | +0.03 [-0.39, +0.44] | -0.17 | -0.12 | -38.7 · 16 % |
| S3 | BULL% > BEAR% | bull>bear | 450 | 213 | 84 % | 70 % | 51 % | +0.13 | +0.41 [-0.01, +0.83] | -0.10 | +0.01 | -31.4 · 12 % |
| S3 | BULL% > BEAR% | bull≤bear | 443 | 199 | 82 % | 66 % | 55 % | -0.15 | -0.14 [-0.67, +0.34] | -0.25 | -0.14 | -38.7 · 18 % |
| S3 | SLOPE5 sign | >0 | 474 | 212 | 85 % | 71 % | 51 % | +0.15 | +0.45 [+0.03, +0.89] | -0.11 | +0.06 | -31.4 · 13 % |
| S3 | SLOPE5 sign | ≤0 | 419 | 193 | 80 % | 64 % | 55 % | -0.19 | -0.22 [-0.77, +0.32] | -0.25 | -0.19 | -38.7 · 18 % |
| S3 | SLOPE1H sign | >0 | 453 | 180 | 86 % | 73 % | 49 % | +0.18 | +0.51 [+0.04, +0.97] | -0.16 | -0.06 | -31.4 · 12 % |
| S3 | SLOPE1H sign | ≤0 | 440 | 163 | 79 % | 63 % | 57 % | -0.20 | -0.25 [-0.78, +0.25] | -0.19 | -0.06 | -38.7 · 18 % |
| S3 | TGAP sign | >0 | 452 | 202 | 85 % | 71 % | 53 % | +0.11 | +0.43 [-0.05, +0.90] | -0.15 | -0.03 | -31.4 · 12 % |
| S3 | TGAP sign | ≤0 | 441 | 197 | 81 % | 64 % | 54 % | -0.12 | -0.16 [-0.67, +0.33] | -0.19 | -0.10 | -38.7 · 18 % |
| S3 | R1D sign | >0 | 452 | 122 | 85 % | 69 % | 52 % | -0.00 | +0.09 [-0.43, +0.63] | -0.15 | -0.10 | -38.7 · 15 % |
| S3 | R1D sign | ≤0 | 441 | 123 | 81 % | 67 % | 54 % | -0.00 | +0.18 [-0.23, +0.59] | -0.20 | -0.03 | -27.0 · 15 % |
| S3 | GAP4H sign | >0 | 450 | 118 | 82 % | 69 % | 53 % | -0.11 | +0.15 [-0.34, +0.62] | -0.21 | -0.09 | -31.4 · 17 % |
| S3 | GAP4H sign | ≤0 | 443 | 130 | 84 % | 67 % | 53 % | +0.10 | +0.13 [-0.36, +0.58] | -0.14 | -0.03 | -38.7 · 14 % |
| S3 | RSI5 >50 | >50 | 475 | 210 | 84 % | 70 % | 51 % | +0.14 | +0.39 [-0.05, +0.82] | -0.16 | +0.01 | -31.4 · 13 % |
| S3 | RSI5 >50 | ≤50 | 418 | 193 | 81 % | 66 % | 55 % | -0.17 | -0.15 [-0.73, +0.38] | -0.18 | -0.15 | -38.7 · 18 % |
| S3 | ABOVE24LO median | >1.27 | 435 | 164 | 86 % | 72 % | 51 % | +0.16 | +0.54 [+0.07, +0.98] | -0.19 | -0.03 | -31.4 · 12 % |
| S3 | ABOVE24LO median | ≤1.27 | 458 | 158 | 80 % | 64 % | 55 % | -0.16 | -0.25 [-0.74, +0.23] | -0.16 | -0.10 | -38.7 · 18 % |
| S3 | OFF30D washed ≤−15 | washed | 145 | 53 | 86 % | 72 % | 54 % | +0.23 | +0.66 [+0.05, +1.31] | -0.16 | +0.08 | -27.0 · 8 % |
| S3 | OFF30D washed ≤−15 | not | 748 | 195 | 82 % | 67 % | 53 % | -0.05 | +0.03 [-0.34, +0.41] | -0.18 | -0.09 | -38.7 · 17 % |
| S3 | EFF72 chop ≤0.007 | chop | 93 | 56 | 88 % | 67 % | 49 % | +0.33 | +0.02 [-1.10, +1.07] | -0.01 | +0.12 | -31.4 · 14 % |
| S3 | EFF72 chop ≤0.007 | trend | 800 | 235 | 82 % | 68 % | 54 % | -0.04 | +0.15 [-0.22, +0.47] | -0.19 | -0.08 | -38.7 · 15 % |
| S3 | BEARDAY (R1D<0 ∧ TGAP<0) | bearish day | 219 | 100 | 75 % | 62 % | 54 % | -0.28 | -0.22 [-0.83, +0.39] | -0.40 | -0.27 | -21.2 · 18 % |
| S3 | BEARDAY (R1D<0 ∧ TGAP<0) | not | 674 | 223 | 85 % | 70 % | 53 % | +0.08 | +0.25 [-0.14, +0.64] | -0.10 | +0.01 | -38.7 · 14 % |

How to read it:
- **Regime does tilt ON-scalp.** When BTC is firm at the ON bar, the pop comes a little more often: P(+3 % ≤ 2 h) is 71–73 % vs 63–64 %. The no-stop exits turn positive, and the four signals agree:
  - BTC 5m slope > 0
  - BTC 1h slope > 0
  - BTC above its 24 h low by more than the median
  - breadth bull > bear
- On S3, TP3/2h/no-stop reads +0.45 to +0.54 with the day-CI just above 0 in four states (5m slope, 1h slope, above-24h-low, 5m RSI T3). The washed-out state reads +0.66 [+0.05, +1.31] on 53 days, but that is mostly one episode.
- **It is not beyond chance.** States passing the four return legs (excluding the tail): observed 3, null mean 6.2, p = 0.85. Separation cells: 22 vs null mean 17.5, p = 0.27.
- The fixed-stop exits (TP1/SL3) and the live lock stay negative in **every** state (best TP1/SL3 −0.01, chop, N = 93).
- **The tail does not move.** About half of all ON entries are 10 % under water within 12 h in every regime. Within the 2 h hold, 8–19 % of fills still end at −5 % or worse, and the worst fill is −21 % to −39 % in every state.
- With "no stop", liquidation is the stop. The ON-scalp study already found P(book DD ≥ 50 %) of 80–100 % for every S0/S3 design. Nothing here reduces the tail enough to change that.
- **Answer to the operator's belief:** a firm BTC makes the pop a few points more likely, but it does not make "enter on ON" safe. No regime brings the 12 h −10 % probability below about half.

## 6. ORCA 18:05 and the other live FRENZY fills since Oct 3 (anecdotes, not evidence)

**ORCA 18:05 (FRENZY_LONG strong 10×, −3.01 %).** Year-cohort mean %/fill of the state it sat in:

| reading | ORCA 18:05 | its state | year mean in that state (N) |
|---|---|---|---|
| bot label | HEALTHY_BULL | bull family | +0.47 (49); HEALTHY_BULL alone −0.07 (21) |
| breadth | bull 56.5 / bear 28.3 | bull > bear | +0.42 (101) |
| BTC 5m slope / RSI | +0.020 / 54.0 (live +0.022 / 55) | > 0 / > 50 | +0.17 (103) / +0.17 (106) |
| BTC 1h slope | −0.089 | ≤ 0 | +0.42 (109) |
| BTC trend gap | −0.229 | ≤ 0 | +0.22 (104) |
| BTC 1d return | −0.885 | ≤ 0 | +0.31 (111) |
| **BEARDAY** | **yes** | bearish day | **−0.06 (59)** |
| BTC off 24 h high / above 24 h low | −1.09 / +0.76 | both ≤ median | +0.50 (107) / +0.05 (112) |
| BTC 4h EMA50/200 gap | +4.27 | > 0 (T3) | +0.41 (96) |
| BTC ATR 5m % | 0.137 | ≤ median | +0.33 (110) |
| market volume (live stamp) | 0.60 | T1 | +0.11 (99) |

ORCA sat in the weakest common state, BEARDAY, and also in "above the 24 h low ≤ median" (+0.05). Both are about breakeven, and neither is a losing cohort. **About 47 % of FRENZY_LONG fills stop out at −3 % in every state.** ORCA was one of them.

**Live FRENZY fills since Oct 3** (rebuilt readings; P&L as booked):

| fill | sleeve | lev | P&L % | live label | BEARDAY | BTC 1d | BTC 1h slope | above 24h low |
|---|---|---|---|---|---|---|---|---|
| ENJ 10-03 03:36 | LONG | 10× | −3.02 | CHOPPY_WEAK | no | −0.41 | −0.13 | +0.89 |
| AIN 10-03 23:25 | WIDE | 4× | −3.01 | CHOPPY_FLAT | **yes** | −0.41 | −0.03 | +0.42 |
| SAND 10-04 05:05 | LONG | 6× | −3.01 | CHOPPY_WEAK | no | +0.27 | +0.01 | +0.38 |
| AIN 10-04 11:15 | LONG | 6× | **+3.47** | CHOPPY_FLAT | no | +0.27 | +0.10 | +0.88 |
| SAND 10-04 14:05 | LONG | 6× | −3.01 | CHOPPY_WEAK | no | +0.27 | +0.08 | +0.86 |
| MOVR 10-05 09:15 | WIDE | 4× | **+3.00** | HEALTHY_BEAR | no | +2.09 | +0.14 | +1.05 |
| RLC 10-05 12:00 | WIDE | 4× | **+3.01** | CHOPPY_WEAK | no | +2.09 | +0.06 | +1.32 |
| AIN 10-05 16:15 | WIDE | 4× | −3.02 | STRONG_BEAR | no | +2.09 | −0.07 | +0.11 |
| FLUID 10-06 02:30 | WIDE | 4× | −3.04 | HEALTHY_BEAR | **yes** | −0.89 | −0.01 | +0.81 |
| ORCA 10-06 09:40 | LONG | 6× | −3.00 | CHOPPY_FLAT | no | −0.89 | −0.02 | +1.13 |
| UMA 10-06 10:10 | LONG | 6× | −3.00 | STRONG_BULL | no | −0.89 | +0.06 | +1.26 |
| ORCA 10-06 18:05 | LONG | 10× | −3.01 | HEALTHY_BULL | **yes** | −0.89 | −0.09 | +0.76 |

- Bearish-day live fills went 0 for 3. The other fills went 3 for 9.
- FRENZY_LONG alone is 1 for 7. At the year's 53 % win rate, a run that bad happens about **4 %** of the time (8 % without ENJ). WIDE + LONG together are 3 for 12, which is also about 4 %.
- This is about four market days. Four days is about four window units, so it carries little weight in either direction.
- It sits on the 40-fill review gate already in place (DECISION_LOG 176: mean > 0 and ≥ 12 winners). It is not a regime signal: 6 of the 9 non-bearish-day fills also lost.

## 7. What was tested, and the blind spots

**Tested (all on the engine cohort, causal, closed bars):**
- bot regime label (bull family; 3 groups; 8 labels)
- breadth bull % / bear % / bull > bear
- BTC 5m EMA20 slope
- BTC 1h EMA20 slope
- BTC trend gap (EMA13−50)
- BTC RSI 5m and 1h
- BTC 1-day return (daily bar) and rolling 24 h return (post-hoc)
- BTC off its 24 h high and above its 24 h low
- BTC off its 30-day high (median, T3/T1, washed-out ≤ −15)
- BTC 4h EMA50/200 gap
- eff72 (chop flag and terciles)
- BTC 5m ATR %
- ETH 1-day return
- market volume
- BEARDAY composite

Each was read at sign / label level, then by T3 vs T1, with:
- halves
- LOMO
- day concentration
- the same-month day-shuffle family null.

Also run: the washed-out window with and without, per-month tape, the uniform-degradation comparison with momentum longs, the ON-scalp family (S0/S3 × 7 outcomes), and the HOLD_GREEN secondary.

**Not tested or weak (blind spots):**
1. **Breadth is the yr5 replay's scan breadth, not live breadth.** The replay scans the replay universe. Live stamps exist only since the stamping started, and the 12 live FRENZY fills cannot test anything. Breadth also ends Oct 3 in the replay, so Oct 4–6 fills use the live stamps.
2. **Market volume** is truncated to < 1 on FRENZY_LONG by its own gate, so only the admitted band is read. The year grid (`gvr_year`) is not the live `gvol_live_U2`.
3. **Live 5m readings use the forming bar.** Labels near the thresholds differ in about 1 in 4 live fills (§0). A rule armed live would see those values, not the closed-bar ones.
4. **2D interactions** were not scanned exhaustively, beyond BEARDAY (the one pre-registered composite) and the post-hoc chop-vs-trend label read. The FRENZY_LONG N (204 fills, 132 days) cannot carry a 2D regime scan with a null at useful power. A "no regime separator" claim here covers **1D and the pre-registered composite only**.
5. **Funding, open interest, BTC dominance and alt-season measures** were not read. The pre-parity study's TOP50 breadth / BTC_DOM / froth variables were not re-run on the engine cohort.
6. **Momentum-long comparison** uses the master pool (Jun 18 → Oct 2, 70 days). It covers only 4 of FRENZY's 9 months.
7. **Pairs absent from `k5m_full`** (delisted) are missing from the cohort. The same caveat applies in every engine-cohort report.

## 8. Proposal (observe-only; nothing armed)

- **No FRENZY filter, no regime sizing, no probe.** No split passes, the family is at chance, and FRENZY does not share the momentum engine's regime response.
- **One observe-only line, frozen now:** `BEARDAY = entry_btc_1d_ret_pct < 0 AND entry_btc_trend_gap_pct < 0` on FRENZY_LONG fills. Both stamps already exist on every live fill, so this costs nothing at batch reviews.
  - **Promotion (to a filter):** the locked expectancy bar on live BEARDAY fills only:
    - N ≥ 15 on ≥ 8 distinct days
    - WR < the sleeve breakeven (47.9 % today; recompute at review)
    - mean < 0 at 95 % by a day-clustered bootstrap
    - no day or pair ≥ 50 % of the loss
    - 30–50 % haircut on any projected Δ.
  - **Retire the line** (pre-committed): if the first 15 live BEARDAY fills average ≥ 0 %.
  - The threshold is frozen. Do not re-fit to "1d < −0.5" or similar after this data.
- **ON-scalp:** no regime rescues it. The ON-scalp study's single free scout observe line stays as frozen there. Do not add a regime-conditioned version.
- **Post-hoc watch items** (do not act on these; read them at the next FRENZY year review):
  - chop label vs trending: +0.05 vs +0.70, p 0.20
  - BTC 5m ATR T3 worse: −0.71 gap

## 9. Pre-registration (frozen before any regime-split outcome was computed)

```
FRENZY REGIME REVIEW — PRE-REGISTRATION (frozen 2026-10-06 ~19:00 UTC, before any regime-conditioned outcome was computed)

COHORTS
 C1 FRENZY_LONG: engine cohort (FRENZY_ENGINE_COHORT_2026-10-05) ∩ overnight live_elig, gvol_live_U2<1, priced outcome (tick/1m),
    sequenced as live today with HOLD_GREEN WIDE (2+2 slots, pair flat, ≤3 per pair-day) — outcome LOCK2 % at 1×.
 C2 WIDE HOLD_GREEN fills of the same sequenced book (secondary).
 C3 ON control: every fresh ON bar (FRENZY_ON_SCALP_SIGNALS, 1,443) — S0 all, S3 = adx_delta>0 ∧ di_spread>0.
REGIME VARIABLES (closed bars ≤ signal-bar close; BTC 5m indicators = services.indicators.calculate_indicators on last 100 closed bars)
 REG label (services.regime.classify_btc_regime) · BULL% / BEAR% (yr5 replay SCAN line ≤ t) · SLOPE5 · SLOPE1H · TGAP (EMA13−50 5m)
 · RSI5 · RSI1H · R1D (last closed daily bar) · OFF24H · ABOVE24LO · OFF30D · GAP4H (EMA50/200 4h) · EFF72 · ATR5% · ETH_R1D · GVOL
 (gvr_year grid) · BEARDAY = R1D<0 ∧ TGAP<0.
SPLITS (coarse first, then terciles; tercile cuts fixed on the year's 5m grid Jan 10–Sep 27)
 coarse: sign (>0 vs ≤0) for SLOPE5 SLOPE1H TGAP R1D GAP4H ETH_R1D; RSI5/RSI1H >50; BULL>BEAR; REG bull-family vs rest
 (and bull / bear / choppy 3 groups, descriptive); EFF72 ≤0.007 chop flag; OFF24H/ABOVE24LO/OFF30D/ATR5/GVOL median split;
 OFF30D ≤ −15 (washed-out) flag; BEARDAY yes/no.  fine: T1 vs T3 for every continuous variable.
STATISTIC  mean %/fill per state; gap = A − B; day-block bootstrap (UTC day = block, 2000 reps) 95 % CI.
SEPARATES (all): each state N ≥ 15 and ≥ 8 days; gap CI excludes 0; gap same sign in Jan–Apr and May–Sep; gap same sign in every
 leave-one-month-out; no single day ≥ 50 % of the losing state's loss.
FAMILY NULL  every cell of the family (C1 + C3 cells counted together) re-read with each day's regime readings replaced by another
 day of the SAME month at the same time of day (within-month day permutation of a 5m regime grid), 300 rounds; same statistic both sides;
 family p = share of rounds with ≥ observed pass count; also max-|z| of gap.
ON-SCALP (operator addition) outcomes per ON bar: H1 = P(+1 % net within 60 min), H3 = P(+3 % within 2 h), T10 = P(−10 % within 12 h);
 exits TP1/T60/noSL, TP3/T120/noSL, TP1/T60/SL3, LOCK2. Subsets S0, S3.
 "Regime makes enter-on-ON work" requires ONE state of ONE split with, for some exit: mean > 0 with day-CI low > 0, both halves > 0,
 LOMO min > 0, AND T10 ≤ 20 % with day-CI upper ≤ 25 % in that state; plus surviving the family null (adjusted p ≤ 0.05).
VERDICT if a split separates for C1: blocked side vs the locked expectancy bar (WR < sleeve breakeven, avg < 0 at 95 % window-clustered,
 ≥ 8 windows, no window ≥ 50 % of loss, N ≥ 15) + 30–50 % haircut; otherwise OBSERVE-ONLY line with the frozen threshold.
WASHED-OUT window Jun-18→Jul-2 shown with and without for C1.
```
