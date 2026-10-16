# Why a WILLY trade keeps going up, or doesn't: continuation after +1.25, recovery after 30 min, market vs coin (2026-10-16)

Research only. No engine, config, `trading_config.json`, template or test file was changed. Nothing was committed and the bot API was not
touched. `backtest_cache` and `scripts/engine_replay*.py` were only read. **This study has not been reviewed, so it cannot support arming
or shipping anything.**
- **Pre-registered** in `reports/WILLY_CONT_PREREG.txt`, frozen at **16:08:57 UTC** with input hashes, before any outcome table. The
  code was hashed at 16:18:30 UTC, before the first run.
  - The coordinator's three corrections from the adversarial WILLY_THEORY review are built in from the start:
    - engine replay fills only;
    - the late TP is a market close at the print when the price is already ≥ +0.25 at minute 30, in every arm;
    - the theory study's "selection adds nothing" is not cited.
  - The follow-up reads (Addendum A) were declared and hashed **after** the main tables. They are post-hoc.
- Every number is at 1× (net % of notional). The verdict uses tick-priced rows. 1m-tail rows are refute-only.
- Binance public data: 1,548 kline requests (weight 1 each) for the coins' 1m taker-buy flow around every fill. Peak used weight was
  161/min, against the pause threshold of 700. There were no 418 or 429 responses, and the cooldown file was honoured.

## Plain-English answer

**1. Holding past +1.25 is worth exactly nothing on average.**
- When a WILLY trade touches +1.25 (64 % of fills, median 4 minutes after entry), it goes on to +2 before minute 30 in **77.7 %** of
  cases.
- Holding for +2 only beats banking +1.25 if that rate is above **77.7 %**. The trades that don't make +2 average −1.36 %.
- So the continuation rate sits **on the breakeven line**. "Always hold for +2" changes the result by **+0.001 %/fill**:
  - 2025: +0.07;
  - 2026-H1: +0.04;
  - 2026-H2: **−0.16**.
- The TP +2 gain the theory study found does not survive on engine fills with the late TP kept.

**2. Continuation is driven by the coin, not the market. The one market signal that matters runs backwards.**
- **The market variables do not work.** None of the market variables (BTC, ETH or SOL moves, top-50 breadth, market taker flow,
  market volume, hour, day regime, trend persistence, dispersion, recent pump follow-through) predicts continuation out of sample.
- **The market-only models do no better than chance.** Held-out AUC is 0.45–0.51 in both directions for all three questions.
- **Several "risk-on" predictions point the wrong way.** If BTC has risen since entry, the trade is *less* likely to continue (2025
  and 2026-H1).
- **Coin-only models carry a little information.** At entry (Q3) their held-out AUC is 0.555 for 2025→2026 and 0.570 for 2026→2025,
  both beating the shuffled null (p 0.04 / 0.02), but they fail H1→H2 (0.42).
- **What helps is a less stretched entry.** The best coin variables, the same sign in both eras, are:
  - a younger FRENZY episode;
  - a lower DI+;
  - a lower range position.

**3. One combination survived the full search: a "market dip on a strong coin".**
- **The rule:** WILLY's red entry candle came on a **market-wide red 5-minute bar** (≤ 30 % of the top-50 up) while the coin's short
  trend is strong (EMA8–13 gap > 1.39 %).
- **The result:** in that case, once the trade touches +1.25, it reached +2 in **45 of 46** cases:
  - 46 days, 42 pairs, 18 months;
  - every era: 100 % / 96 % / 100 %;
  - against 76 % for all other touches.
- **The likely mechanism:** the coin's dip was the market's dip, not the pump running out of buyers, so its own demand carries on.
- **Statistics:**
  - It is the top of 28,646 tests, but it beats the shuffled null over the whole search: **p_fw 0.010**.
  - Leave-one-month-out keeps it positive in 22 of 22 months.
  - Cuts re-derived in one era hold in the others.
- **The weaknesses:**
  - It is a **corner effect**. The 3×3 grid is not monotone outside the corner.
  - The walk-forward procedure confirmed 0 zones.
  - It is worth only **+0.04 %/signal**, about **+0.02–0.03 after the 30–50 % haircut**.
  - The 1m ruler agrees in 2025 (+0.03), but its 2026 rows are too few to read (4 held trades).
  - Master: 3 fills were in the zone and all 3 reached +2. Live: none of the 8 fills was in the zone.

**4. After minute 30, recovery depends on where the price is, nothing else.**
- 22.5 % of fills are still open at minute 30. They average **−0.95 %** under today's exit.
- 38 % of them recover to +1.25, and 24 % stop later. Whether a trade recovers is set by **how far below entry it sits at minute 30**:
  - ≤ −2 %: 20 % recover;
  - −1 to −2 %: 34 %;
  - around 0: 48 %;
  - above +0.25: 72 %.
- No market or coin variable adds to that in a way that pays.
- Today's late TP is already the best of the three choices:
  - keep +1.25 instead: −0.05 %/open trade;
  - close at minute 30: −0.09.
  - A model-score "keep if likely to recover" rule flips sign between eras: −0.06 / +0.10.

**What to do (the operator decides):** change nothing on the WILLY exit.
- The market-dip × strong-coin cell is at most an **observe-only scout line**: "at +1.25 the TP would move to +2", frozen cut. It is
  small, post-search and unreviewed. It must not be armed before a review plus fresh fills (see the gate below).

## The three blocks for the best rule ("at +1.25 hold for +2 if market-dip & strong coin, else bank"; FAILED-to-be-actionable label: observe only)

| block | N | today avg % | rule avg % | Δ %/fill | Δ Σ %-pts | Δ $ | held (in zone) |
|---|---|---|---|---|---|---|---|
| Backtest seed 1 tick 2025 | 228 | −0.053 | −0.017 | +0.036 | +8.2 | – | 11 |
| Backtest seed 1 tick 2026-H1 | 393 | +0.091 | +0.137 | +0.046 | +18.1 | – | 25 |
| Backtest seed 1 tick 2026-H2 | 202 | +0.015 | +0.052 | +0.037 | +7.5 | – | 10 |
| Backtest seeds 2 / 3 tick (all eras) | 822 / 820 | +0.040 / +0.047 | +0.081 / +0.089 | +0.041 / +0.041 | – | – | 46 / 46 (the same fills) |
| Backtest 1m ruler (refute-only), 2025 | 668 | −0.466 | −0.438 | +0.028 (P 0.92) | – | – | 76 |
| Backtest 1m ruler, 2026 | 25 | – | – | −0.25 on 4 held | – | – | 4 |
| MASTER B19 | 14 | +0.718 | +0.718 | 0 | 0 | $0 | 0 |
| MASTER B20 | 3 | −0.576 | −0.076 | +0.500 | +1.50 | **+$95.9** | 2 (CAP, RSR → +2) |
| MASTER B21 | 2 | +1.250 | +1.625 | +0.375 | +0.75 | **+$52.4** | 1 (BTW → +2) |
| LIVE 10-15/16 | 8 | −0.238 | −0.238 | 0 | 0 | $0 | 0 |

Master / live per trade, today's exit vs the rule, in % and $:
- The changed trades:
  - B20 CAPUSDT 10-12 12:55: +1.25 → +2.00 (+$42.9);
  - B20 RSRUSDT 10-12 05:15: +1.25 → +2.00 (+$53.0);
  - B21 BTWUSDT 10-14 03:55: +1.25 → +2.00 (+$52.4).
- The other 24 trades are unchanged.
- Today's totals reproduce the theory study exactly: master +10.82 pts / +$1,203, live −1.90 / −$42. The rule gives master +13.07 /
  +$1,351 and live unchanged.
- Full per-trade sheet: `reports/willy_cont/ms_rows.csv`.

For comparison, "always hold for +2" on master/live:
- B19: −4.41 pts / **−$416**. The LPT and W touches went on to stop.
- B20: +$96.
- B21: +$106.
- Live: −4.20 pts / −$105. BAT stopped and RLC fell to the late TP.

The unconditional hold is what the earlier TP +2 idea amounts to, and master and live both refute it.

## Q1 — at the +1.25 touch (seed 1 tick)

| era | touches | share of fills | median min to touch | reach +2 by min 30 | reach +2 by 90 | reach +3 by 90 | avg if not +2 | breakeven rate | hold2 − bank | P(>0) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 135 | 59 % | 5.5 | 78.5 % | 88.9 % | 73.3 % | −0.94 | 74.5 % | +0.119 | 0.82 |
| 2026-H1 | 269 | 68 % | 3.5 | 79.9 % | 84.4 % | 67.3 % | −1.45 | 78.2 % | +0.058 | 0.72 |
| 2026-H2 | 125 | 62 % | 4.8 | 72.0 % | 80.0 % | 63.2 % | −1.57 | 79.0 % | −0.250 | 0.10 |
| ALL | 529 | 64 % | 4.2 | **77.7 %** | 84.5 % | 67.9 % | −1.36 | **77.7 %** | +0.001 | 0.52 |

**The theory test.** Each pre-registered prediction was read as the sign of (high − low) on hold2 − bank, per era.
- **The coin's own flow is mixed.** Taker share at the touch matches the prediction in 2 of 3 eras. Volume vs pre-entry matches in 1
  of 3. Speed to the touch matches in 1 of 3.
- **The market predictions fail or reverse:**
  - BTC since entry: 0 of 3;
  - BTC over the last 15 min: 0 of 3;
  - breadth over the last 15 min: 1 of 3;
  - day regime r1, br_ma7, ft12, persistence (ac15) and market volume over 60 min: 0 of 3 each.
- **No variable matched the predicted sign in all three eras.**
- In 1D the best era-consistent S is 1.02, against a null median of 1.46. **1D finds nothing.**

## Q2 — open at minute 30 (seed 1 tick, 185 fills on 150 days)

| era | open at 30 | share | ≥ +0.25 at 30 (market close now) | recover to +1.25 | stop later | today | keep +1.25 | close at 30 | keep − today (P) | close − today (P) |
|---|---|---|---|---|---|---|---|---|---|---|
| 2025 | 69 | 30 % | 16 % | 38 % | 26 % | −1.16 | −1.06 | −1.17 | +0.10 (0.75) | −0.02 (0.47) |
| 2026-H1 | 68 | 17 % | 12 % | 34 % | 25 % | −0.95 | −1.14 | −0.93 | −0.20 (0.11) | +0.02 (0.53) |
| 2026-H2 | 48 | 24 % | 13 % | 46 % | 21 % | −0.65 | −0.69 | −0.98 | −0.04 (0.43) | −0.33 (0.07) |
| ALL | 185 | 22.5 % | 13.5 % | 38 % | 24 % | −0.95 | −0.99 | −1.03 | −0.05 (0.31) | −0.09 (0.21) |

By the price at minute 30 (all eras):

| at minute 30 | N | recover | today | keep | close |
|---|---|---|---|---|---|
| ≤ −2 % | 40 | 20 % | −2.24 | −2.33 | −2.71 |
| −2 to −1 | 47 | 34 % | −1.37 | −1.24 | −1.45 |
| −1 to −0.25 | 44 | 34 % | −0.63 | −1.09 | −0.61 |
| ±0.25 | 29 | 48 % | −0.19 | −0.00 | +0.02 |
| > +0.25 | 25 | 72 % | +0.47 | +0.62 | +0.47 |

**The Q2 model reads.**
- **The COIN logistic does predict recovery out of sample.** AUC is 0.667 for 2025→2026 and 0.635 for 2026→2025, against a null 95th
  percentile of about 0.60. Its largest same-sign input is the price at minute 30.
- **Turned into a rule, it does not pay:**
  - The single-variable rule "keep +1.25 if the price at minute 30 is above the discovery median" gives **−0.18 (P 0.04, wrong way)**
    on 2026 and +0.15 on 2025.
  - The model-score rule gives −0.06 / +0.10.
- **The market-only models never beat the null** (AUC 0.43–0.53), apart from one H1→H2 tree.

## Q3 — at entry: market or coin? (823 tick fills on 390 days)

| features | 2025→2026 AUC (p) | 2026→2025 AUC (p) | H1→H2 AUC (p) |
|---|---|---|---|
| MARKET logit / tree | 0.456 (0.96) / 0.467 (0.90) | 0.453 (0.92) / 0.465 (0.82) | 0.508 (0.43) / 0.563 (0.10) |
| COIN logit / tree | **0.555 (0.04)** / 0.537 (0.14) | **0.570 (0.02)** / **0.586 (0.02)** | 0.418 (0.97) / 0.485 (0.60) |
| ALL logit / tree | 0.514 (0.31) / 0.532 (0.14) | 0.526 (0.24) / 0.561 (0.04) | 0.430 (0.94) / 0.605 (0.02) |

Label: +2 reached before −4 within 90 min (base rate about 60 %).
- **Top coin inputs, same sign in both training eras:**
  - FRENZY hours: a younger episode continues more;
  - DI+: lower is better;
  - range position: lower is better;
  - above-VWAP share: higher is better.
- **The best single variable:** `entry_frenzy_hours ≤ 0.17 h`. Fills on that side continue 63–65 % of the time vs 54 %, with AUC 0.55–0.58.
- **No economic separator.** In 1D and 2D on hold2 − bank, the best S is 2.50 against a null median of 3.25: **0 survivors**.

## Searches (every family; S = the sign-agreeing minimum |t| across the three eras; null = 300 day-block shuffles over the full search)

| family | tests | max abs S | null q50 / q95 | survivors p_fw < 0.10 | walk-forward confirmed (null mean) |
|---|---|---|---|---|---|
| Q1 1D, hold2 − bank | 568 | 1.02 | 1.46 / 2.30 | 0 | 0 (1.6) |
| Q1 1D + 2D MARKET × COIN, hold2 − bank | 28,646 | **4.63** | 3.70 / 4.46 | **1** (the market-dip × strong-coin cell, p 0.010) | 0 (2.6) |
| Q1 1D + 2D, label "reached +2 by min 30" | 28,646 | 3.70 | 2.50 / 4.26 | 1 (same cell, p 0.08) | 0 (1.5) |
| Q2 1D + 2D, keep − today | 13,302 | 2.20 | 2.26 / 2.80 | 0 | 1 (2.1) |
| Q2 1D + 2D, recovered | 13,302 | 1.95 | 1.97 / 2.56 | 0 | 4 (0.96, p 0.07) |
| Q3 1D + 2D, hold2 − bank | 23,951 | 2.50 | 3.25 / 3.89 | 0 | 2 (3.5) |
| Q3 1D + 2D, reached +2 | 23,951 | 2.21 | 2.10 / 2.60 | 0 | 2 (0.96) |

Day units (day-mean outcome, median split, prior-day regime variables):
- Q1: 3 of 25 variables keep the same sign in all eras, which is chance level (alt_rs7, spk72, flags96).
- Q3: 4 of 25.
- Q2: 12 of 25 point the same way. Higher BTC 7-day return and a higher realised-vol ratio lean toward "keep pays", but there are only
  150 days, and the fill-level tests do not support it.

### The survivor, robustness (Addendum A, post-hoc)

- **Zone:** `e_br_up5 ≤ 0.2979 & entry_ema_gap_8_13 > 1.393`, applied to Q1 touches.
  - 46 touches: 45 reached +2 by minute 30, against 76 % for all other touches. The 95 % lower bound is 0.90, above the 0.78 bar.
  - 38 of 46 reached +3 by minute 90. HOLD3 − bank averages +0.83 in the zone.
- **The coin side by itself:** of all 62 fills in the zone, 46 touch +1.25 and 12 stop (19 %, the cohort's normal rate). The cell
  tells you nothing about whether a trade will stop. It only tells you what happens **after** the touch.
- **Era-frozen cuts** (terciles derived in one era, applied to another):

  | discovery → test | +2 rate in zone | rest | Δ per signal |
  |---|---|---|---|
  | 2025 → 2026-H1 | 93 % | 79 % | +0.027 |
  | 2025 → 2026-H2 | 100 % | 71 % | +0.022 |
  | 2026 → 2025 | 100 % | 76 % | +0.043 |
  | 2026-H1 → 2025 | 100 % | 76 % | +0.046 |
  | 2026-H1 → 2026-H2 | 100 % | 70 % | +0.033 |

- **Dose-response (share reaching +2):**

  | breadth (top-50 up) | gap tercile 1 (lowest) | gap tercile 2 | gap tercile 3 (highest) |
  |---|---|---|---|
  | lowest tercile | 0.78 | 0.73 | **0.98** |
  | middle tercile | 0.70 | 0.75 | 0.77 |
  | highest tercile | 0.73 | 0.71 | 0.86 |

  - The gap alone is monotone: > 1.39 gives 0.86 vs 0.74.
  - Breadth alone is weak: 0.82 vs 0.76.
  - The corner is the effect. CLAUDE.md flags a non-monotone pattern as a possible confound, which is why this is observe-only.
- **Concentration:** top pair STRK 2 of 46, top day 1 of 46. Leave-one-month-out keeps it positive in 22 of 22 months.

**Proposed scout gate, if the operator wants a line (observe only, not armed).**
- Tag WILLY fills on the frozen cut with breadth taken from the last closed 5m bar at entry.
- Record whether the touch reaches +2 by minute 30. That costs nothing, because the path is already walked.
- **CANDIDATE** at ≥ 15 fresh in-zone touches on ≥ 8 days with a +2 rate ≥ 85 % AND the rest of the cohort below 80 %.
- **RETIRE** if the rate is < 78 % at N ≥ 10.
- Even a CANDIDATE needs a reviewed study before anything changes the order.

## Coverage — every variable family tested

| question | family | variables | tested | min scored % |
|---|---|---|---|---|
| Q1/Q2 | decision-moment coin flow (taker share over the last 5 closed 1m bars, taker share since entry, volume vs the 5 min before entry, coin 5-min return) | 5 | 5 | 91–100 |
| Q1/Q2 | decision-moment market (BTC/ETH/SOL since entry, last 5, last 15 min; market taker share; top-50 breadth over 5 / 15 min; top-50 aggregate taker share over 5 / 15 min) | 13 | 13 | 100 |
| Q1/Q2 | path state (minutes to the touch, worst before the touch; price, best and worst at minute 30) | 2 / 3 | all | 100 |
| all | entry-minute market (BTC/ETH/SOL over 5 / 15 min, market taker share, breadth, top-50 taker share) | 11 | 11 | 100 |
| all | entry-minute coin flow (taker share and 5-min return before entry) | 2 | 2 | 100 |
| all | intraday market (hour, BTC since 00:00 UTC, BTC 60-min realised vol vs 7 days, BTC+ETH+SOL 60-min quote volume vs 7 days) | 4 | 4 | 100 |
| all | day regime, prior closed days (BTC r1/r3/r7/r30, dma20/50/200, rv7, rvtr, off30, breadth br_up3/7, br_ma7/20, alt_rs7, voltr, disp7, spk24/72, flags96, ft2/ft12/ftr12, persistence ac15_3d / ac5_3d) | 26 | 25 | fund3 not tested (71 % coverage) |
| all | entry stamps MARKET (BTC slope/RSI/ADX/ATR/trend gap/off24/off30d/r72/eff72, bull/bear %, gvol, ETH 5m) | 26 | 26 | 100 |
| all | entry stamps COIN (spike volume multiple, run, vs VWAP, hours, wait bars, bar return, above share, turnover, mcap, cmc rank, 24h volume, RSI/ADX/DI/EMA gaps, range position, pattern flags …) | 52 | 50 | 2 pattern flags are constant |

**Could NOT test:**
- **Coin age.** `entry_pair_age_days` is not stamped on replay fills.
- **Order book and open interest.**
- **Trade-by-trade taker flow.** Only 1m klines were used.
- **Live ICNT 10-16 day regime.** The day-features file ends 10-15.
- **Master/live breadth** comes from the WILLY_THEORY context fetch: the top-50 list is the last available day, at 5m only.
- **Tick coverage:** the verdict rows are the 823 tick-priced seed-1 fills (54 %), with 2025 thin (228). Tick coverage is not random.
- **Seeds are not independent.** Seeds 2 and 3 replicate on largely shared fills.
- **Portfolio effects are not emulated.** A longer hold holds the WILLY slot longer, which the signal-level arms ignore.

## Parity / validation
- **Walker parity:** 1,549 of 1,549 (100 %) identical to `WILLY_LOWVWAP_SHORTRUN_rows.csv`, both for the plain +1.25/−4/90 walk and for
  `evaluate_late`.
- **Today's exit with the corrected late booking** (market at the print when ≥ +0.25 at minute 30: 25 fills; resting +0.25: 87):
  - It reproduces master +10.82 pts / +$1,203 and live −1.90 / −$42, identical to the theory study.
  - The raw `evaluate_late` print-booking average (+0.078) differs from the booked +0.033 mainly because a resting maker TP books the
    line, not the print.
- **`scripts/validate_against_master.py`: ALL CHECKS PASS.**

## Files / commands
```
venv/bin/python scripts/study_willy_cont_fetch.py          # reports/willy_cont/k1m (1,548 windows, weight 1 each)
venv/bin/python scripts/study_willy_cont_build.py          # paths, arms, features → reports/willy_cont/rows.pkl (+ parity.txt)
venv/bin/python -W ignore scripts/study_willy_cont_analyse.py --nperm 300 --nnull 200   # tables.md, tests_1d/2d.csv, families.csv,
                                                           # walkforward.csv, models.csv, coverage.csv, ms_rows.csv
venv/bin/python -W ignore scripts/study_willy_cont_followup.py                          # Addendum A → followup.md
venv/bin/python scripts/validate_against_master.py
```
Full tables: `reports/willy_cont/tables.md` and `reports/willy_cont/followup.md`.
