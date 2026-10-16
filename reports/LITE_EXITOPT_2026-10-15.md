# LITE: is there a better exit combination? Exhaustive walk-forward search (2026-10-15)

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing was committed and the bot API was not called. Cache-only: no Binance call was made.
- The full search space, the selection rule, the gates and the operator's market-state addition were pre-registered in `reports/LITE_EXITOPT_PREREG.txt` (frozen 18:57:15 UTC) before any combination was priced.
- This study has not been reviewed, so it supports no arm and no ship.
- Today's LITE exit: TP +3 net as a resting maker limit, stop −3, 12 h cap, 10×.

## Plain-English summary

**Short answer: none beats today out of sample.** 4,347 exit combinations were searched together: 3,024 fixed and 1,323 scaled to volatility. On top of those came entry-dependent versions and a market-state screen.
- Every way of picking "the best combination" on one period and then testing it on a period it had not seen did no better than today's +3 / −3 / 12 h. Most did worse.
- In-sample, the best combinations look 0.1 to 0.4 %/trade better than today. That gap is overfitting. Out of sample it shrinks to between −0.27 and +0.06 %/trade, and never with confidence.

**(a) The best combination out of sample, and how much better than today.**
- The pre-registered selection rule picks the combination with the best average on the training period. Out of sample, its results against today were:

| Out-of-sample test | Combination picked on the training data | Δ %/trade vs today (OOS) | Chance it is really better |
|---|---|---|---|
| Train 2025 → test 2026 | TP +2 / stop −3.5 / lock (+1.5 → 0) / trail 2 | **−0.096** | 15 % |
| Train 2026-H1 → test 2026-H2 | TP +4 / stop −4 | **−0.036** | 41 % |
| Train 2026 → test 2025 | TP +3.5 / stop −4 | **−0.033** | 33 % |
| Month by month (train on every earlier month) | changes monthly | +0.027 | 65 % |
| Same folds, tick-priced trades only | as above | −0.240 / −0.036 / −0.212 | 0–41 % |

- **The overfitting gap.**
  - Picked on 2025, the best combination looked **+0.165 %/trade** better than today on 2025 itself and came in at **−0.096** on 2026.
  - Picked on 2026, it looked **+0.237** and came in at **−0.033** on 2025.
  - So the in-sample advantage loses 0.11 to 0.33 %/trade out of sample. The entry-conditional rules lose up to 0.47.
- **The two years want opposite exits.**
  - 2026 rewards letting winners run: TP +3.5 / stop −4 is +0.237 in 2026.
  - 2025 rewards taking +2 early: TP +2 / stop −3.5 is +0.165 in 2025.
  - Each one loses in the other year.
- **Not one of the 4,347 combinations beats today on both 2026 and the tick-priced 2025 trades: 0 of 4,347.**
  - Only one beats today in all three test periods: TP +3.5 / stop −3 (2026 +0.061, P 0.91; 2025 +0.005).
  - It loses on 2025 tick-priced trades (−0.110, P 0.10), and no training period would have picked it.
  - One lucky combination out of 4,347 is what chance alone produces.
- **All 11 selection procedures fail the gates.** That covers fixed, volatility-scaled, joint, entry-ATR regimes (two selectors each) and the three secondary entry variables. None passes even G1 (beat today in every fold with P ≥ 0.90).
- The best single fold result anywhere is +0.063 (P 0.66).

**(b) What it would do to EPIC-type trades and to winners.**
- **How common EPIC-type trades are.** About 1 LITE trade in 10 is an EPIC type: it reaches +2 and ends at −3. That is 64 of 624 trades in 2026 and 93 of 828 in 2025.
  - **For every EPIC, about 2 trades reach +2, fall at least a full point below their peak (at a 1-minute close), and still finish at +3.** That is 137 vs 64 in 2026 and 158 vs 93 in 2025.
  - At the moment a trade has dipped a full point from its high (average peak +2.42, now +1.11), cashing out and holding are worth the same on average. Holding is worth +0.91. Cashing out is worth +1.11 − 0.20 slippage = +0.91.
  - That is why no cut-the-loser rule pays: at the decision point it is a coin flip.
- **Every rule that saves EPIC pays for it with winners.** Example: the all-data in-sample favourite, TP +2 / stop −4 / trail 2 after +2.
  - EPIC: books +2.03 instead of −3.2 (Δ +5.2).
  - On 2026 it saves 111 losers (+565 points). It also cuts 336 winners from +3 to +2 (−339 points) and makes 174 stops deeper at −4 (−174 points).
  - DEEP (a live +3 winner) becomes +2 (−0.98).
  - On the 20 real master trades it is **−0.69 %/trade**.
  - Out of sample it is negative in every fold.
- **What helps EPIC, by family:**
  - A lower target (+2 to +2.5) books +2.0 to +2.5.
  - A lock or trail after +2 books about +0.8 after slippage.
  - Wider-TP combinations, the ones 2026 likes, do nothing for EPIC.
  - DUSK cannot be scored, because today's −3 never prints on its fetched 1m bars (live booked −2.54). Tighter stops would have booked −1.7 to −2.7 there.

**(c) Do volatility-scaled exits help? No. They are the worst family.**
- TP / stop set as multiples of the entry 5m ATR, with vol-scaled locks and trails: picked on train, they lose **−0.17 / −0.27 / −0.16 %/trade** out of sample (P 0.05–0.07 that they are better), −0.095 month by month.
- On tick-priced trades they lose −0.27 to −0.42.
- No neighbour of the chosen combination is positive (0/7).
- Splitting trades into three ATR regimes and giving each its own best combination is also negative out of sample: −0.17 / −0.02 / −0.04.

**Recommendation (operator decides): keep LITE at +3 resting / −3 / 12 h.**
- This search covered the joint space the earlier single-family studies could not.
- It tells the same story more strongly. An "EPIC can never happen again" exit exists, but it costs more on winners than it saves on EPICs, and the choice of which one to use is driven by which year you look at.
- An EPIC-type loss is a structural feature of a +3 / −3 sleeve that reaches +2 on two-thirds of its trades. The cost of preventing it is larger than the loss.

## Market-state 2D: EPIC vs dippers (operator addition)

The question: "there must be a 2D of what was going on in the MARKET too, before deciding to cut the winners." The idea was to protect only when the market says the move is failing.

**Answer: the market at the decision moment does not tell EPICs from dippers. Zero separators survive, so the conditional-protection rule (Step B) was not run, as pre-registered.**

- **Decision moments.**
  - (i) The trade first reaches +2.0 or +2.5: 837 / 689 BT trades.
  - (ii) The first 1-minute close after that at least 1.0 below the peak: 461 / 253 trades. This is the exact EPIC-vs-dipper moment.
- **Variables (12).**
  - Market: BTC 5 / 15 min and since entry, top-50 breadth and its 15 min change, top-10 share up over 15 min, aggregate taker-buy share, and market volume vs its 24 h mean.
  - Coin: return vs BTC since the peak, 1m volume vs pre-entry, taker-buy share since the peak, and RSI(5m) change since the peak.
  - Coverage was 87–100 %.
- **Tests: 2,256.** Every variable at sign / median / outer-tercile, plus every 2D pair (66) as quadrants, × 4 decision sets. Market variables were read in day units.
- **Null: 300 day-block shuffles,** with the same statistic.
  - The strongest real separator scores |z| 3.40.
  - Shuffled data produces a best |z| of 3.45 at the median and 4.70 at the 95th percentile.
  - The best "pattern" is weaker than what random shuffles produce half the time. Family-wise p ≤ 0.05: **0 tests**.
  - Raw |z| ≥ 1.96: 74, against about 107 expected by chance.
- **Even the most "failing-looking" cells are far from paying for protection.**
  - At +2, the worst cells have a 22–27 % stop rate against a 19 % baseline. Example: the coin outran BTC most over the last 15 min (top tercile): 22 % vs 13 %, family-wise p 0.92.
  - A lock at +1 pays only if P(stop) / P(TP) > 0.5, which means a stop rate of about 33 % or more. No cell gets there.
- **At the dip moment the baseline is 34 % stop / 64 % TP,** which is already the coin flip described above. No market or coin variable moves it reliably: no 2D quadrant holds its sign in 2025, 2026 and tick-only data together at family-wise significance.
- **EPIC's own state** is in `reports/lite_exitopt/live.csv` (moments). There is no separator to classify it with.

## Gates (pre-registered; LS ruler = resting-maker TP, −0.20 on stop / lock / trail / cap)

| procedure | final (selected on all BT) | G1 folds a / b / c (OOS Δ, P) | G2 monthly | G3 tick-only | G4 master | G5 neighbours | pass |
|---|---|---|---|---|---|---|---|
| J fixed, SEL-A | TP2 S4 trail 2 | −0.096 (0.15) / −0.036 (0.41) / −0.033 (0.33) | +0.027 (0.65) | −0.24 / −0.04 / −0.21 | −0.69 | 3/5 | no |
| J fixed, SEL-B (smoothed) | TP2 S4 lock(2,1) trail 1 | −0.097 / **+0.063** (0.66) / −0.033 | +0.018 (0.60) | −0.28 / +0.06 / −0.21 | −0.69 | 7/7 | no |
| V vol-scaled, SEL-A | kTP 1.5 kSL 3 | −0.170 (0.05) / −0.274 (0.07) / −0.158 (0.05) | −0.095 (0.10) | −0.27 / −0.27 / −0.42 | −0.15 | 0/7 | no |
| V vol-scaled, SEL-B | kTP 1 kSL 3 | −0.166 / −0.162 / −0.145 | −0.092 (0.11) | −0.28 / −0.16 / −0.41 | −0.25 | 0/6 | no |
| J ∪ V, SEL-A | = J SEL-A | −0.096 / −0.036 / −0.033 | −0.011 (0.43) | as J | −0.69 | 3/5 | no |
| J ∪ V, SEL-B | = J SEL-B | −0.097 / +0.063 / −0.033 | +0.005 (0.51) | as J | −0.69 | 7/7 | no |
| E: ATR terciles, SEL-A | 3 regime picks | −0.173 (0.04) / −0.023 (0.43) / −0.044 (0.29) | −0.018 (0.39) | −0.15 / −0.02 / −0.18 | −0.31 | n/a | no |
| E: ATR terciles, SEL-B | 3 regime picks | −0.195 (0.03) / +0.039 (0.57) / +0.040 (0.73) | −0.027 (0.35) | −0.18 / +0.04 / −0.15 | −0.03 | n/a | no |
| E: vs-VWAP stretch (2nd) | 2 regime picks | −0.104 / −0.125 / −0.078 | – | −0.15 / −0.13 / −0.26 | −0.39 | n/a | no |
| E: hours since spike (2nd) | 2 regime picks | −0.119 / −0.037 / +0.058 | – | −0.24 / −0.04 / −0.03 | −0.37 | n/a | no |
| E: signal-bar return (2nd) | 2 regime picks | −0.098 / −0.024 / +0.062 | – | −0.22 / −0.02 / −0.18 | −0.47 | n/a | no |

- Passing procedures: **0 of 11** (about 0.4 expected by chance).
- LSe (−0.43 stops) and RM0 (zero slip) rulers, the in-sample-best-on-test, the deflation for every fold, per-month picks, two-sided SAVED / KILLED / DEEPER / GAINED and the live table are all in `reports/LITE_EXITOPT_tables.md`.

**Simple named rules for reference** (one fixed combination each, Δ %/trade vs today, LS; P in brackets):

| rule | 2025 | 2025 tick-only | 2026-H1 | 2026-H2 | master (20) |
|---|---|---|---|---|---|
| trail 1.0 after +2 | +0.061 (0.84) | −0.186 (0.02) | −0.082 (0.15) | −0.005 (0.49) | −0.59 |
| lock +1 after +2 | +0.062 (0.86) | −0.177 (0.03) | −0.123 (0.06) | +0.029 (0.61) | −0.35 |
| TP +2.5 | +0.064 (0.92) | −0.050 (0.21) | −0.025 (0.34) | −0.039 (0.28) | +0.01 |
| TP +2 | +0.134 (0.98) | −0.096 (0.16) | −0.022 (0.40) | +0.024 (0.57) | −0.29 |
| TP +3.5 | +0.005 (0.55) | −0.110 (0.10) | +0.065 (0.87) | +0.054 (0.79) | +0.28 |
| stop −2 | −0.039 (0.28) | +0.026 (0.61) | −0.133 (0.06) | −0.210 (0.05) | −0.33 |
| 4 h cap | −0.049 (0.07) | −0.034 (0.25) | −0.079 (0.03) | +0.004 (0.55) | 0.00 |
| 2 h cap | −0.068 (0.09) | −0.104 (0.06) | −0.032 (0.32) | −0.019 (0.40) | −0.24 |

The pattern is the same throughout. Protection "works" only on the 2025 trades priced on 1-minute bars. It reverses on tick prints and in 2026.

## Parity and verification (every headline two ways)

- **Two independent walkers agree.** The vectorized all-combinations walker was checked against an independent event-by-event loop: 4,416 random fill × combination checks inside the pricing run, plus 99 combinations (every selected / final / in-sample-best / top-10, plus today) on 400 random BT fills + the 20 master fills, under both rulers: 83,160 checks (see `reports/lite_exitopt/verify.txt`). There were **0 mismatches**.
- **Every no-protection combination (S0)** was asserted equal to `FX.evaluate` on every fill.
- **LITE_PROTECT's own cells** were asserted equal to its `eval_v` on every fill, under S0 and RM: P1 locks A2 / A2.5, P2 trails, and P4 lower TPs.
- **Today's exit** was asserted equal to LITE_PROTECT's `base_pct` / `rm_base_pct` on all 1,472 fills.
- **Exit-counterfactual check.** Every today-like combination whose protection never armed while today's exit was open gives Δ = 0. This was asserted on every fill, under both rulers.
- **Today vs the replay's recorded pct, using LITE_PROTECT's definition (same exit kind and |Δ| < 0.10):**
  - 2025 tick-priced 100 % · 2025 1m-priced 99.1 % · 2026-H1 90.2 % · 2026-H2 90.6 % (the known 1 Hz monitor gap) · master 20 / 20.
  - On the stricter |Δ| ≤ 0.05 test, the 2025 1m-priced fills match only 63 %. The exit kind still agrees 99.6 %; the gap is the TP booked at the line vs the replay's print on 1m bars.

## Could not test / limits

- **ZK:** in no export, so it was not tested.
- **DUSK:** today's exit is unresolved on its fetched 1m bars, so Δ vs today is not scorable.
- **EPIC / DUSK** are 1m replicas, with entry = 5m close + 5 s, and their bars end 16:25 UTC. EPIC / DUSK ATR was computed from their fetched 5m bars (Wilder ATR14; EPIC 3.69 %, DUSK 1.80 %), not stamped.
- **DEEP:** ticks then 1m. Its path ends on 10-14 (a 12 h cap is unresolved, but every TP-type combination resolves).
- **Market-state data limits.** Coin taker-buy share has 5m resolution and only covers coins in the top-50 5m taker set (87–90 % coverage). Market variables cover BT only; master's 20 fills were not screened, since N is too small.
- **Slippage model.** Lock / trail / stop slippage is a flat −0.20 (−0.43 sensitivity). Latency beyond that was not modelled. Partial exits / scale-outs were not in this grid; LITE_PROTECT P3 tested them, and they were negative.
- **Step B** (market-conditional protection) was not run because Step A found no separator (pre-registered).
- **Size of the search and the haircut.** 4,347 combinations × 8 + 3 procedures means many chances of a lucky pass; none occurred. The 30–50 % haircut is moot because no OOS result is positive with confidence.

## Exact commands

```
venv/bin/python scripts/study_lite_exitopt_prereg.py          # pre-registration (refuses overwrite)
venv/bin/python scripts/study_lite_exitopt_price.py --jobs 9 --loop-every 3   # all 4,347 combos × 1,472 fills + live, asserts (~1 min)
venv/bin/python scripts/study_lite_exitopt_wf.py              # walk-forward folds, monthly, finals, gates, landscape
venv/bin/python scripts/study_lite_exitopt_market.py --shuffles 300   # market-state 2D screen (Step A)
venv/bin/python scripts/study_lite_exitopt_verify.py --jobs 9 # way-2 headline re-price + rec_pct parity
venv/bin/python scripts/study_lite_exitopt_report.py          # → reports/LITE_EXITOPT_tables.md
```

Outputs: `reports/lite_exitopt/` (price_BT.npz, price_LIVE.npz, moments.csv, live.csv, wf_folds.csv, wf_monthly.csv, wf_final.csv, landscape.csv, market_features.csv, market_tests.csv, market_summary.json, verify.txt).
