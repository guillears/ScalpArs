# FRENZY flag: trade math study (2026-10-08)

## Plain-English summary

**Question:** if a bot buys a pair right when FRENZY flags it (or a few minutes later), with a fixed take-profit, an optional stop and a time limit, is there a combination that makes money after real costs?

**Answer: no.** I tested 5,376 combinations on 2,320 real flags from Jan to Oct 2026, using 1-minute prices. None passed:

1. **Picking the best rules on one half of the year and checking them on the other half fails.** I picked the 5 best rules on Jan–May. On Jun–Oct, all 5 lost money, averaging −0.15 % per trade. In the other direction (pick on Jun–Oct, check on Jan–May) the average was −0.18 %.
2. **The flag timing adds nothing.** I re-ran the whole grid with the flag moved to a random minute in the 30 minutes after it, 200 times. That random-time version found "best rules" as good as or better than the real flag time in 52–97 % of the runs.
3. **The best rule overall** was: buy at the flag, but only if price is already above the flag close 8 s later; take profit at +1.5 %; no stop; close after 120 min. It averaged +0.08 %/trade, which is about +0.04–0.055 % after the required 30–50 % haircut. Its 95 % day-block range is −0.12 % to +0.26 %, so it is indistinguishable from zero. Its average loss is −4.9 % and its worst is −25 %. On a $3,000 account at 20× it would have been liquidated within its first 10 trades.
4. **The "quick small win" style the operator trades by hand loses money as a fixed rule.** Rule: buy at the flag, +0.3 % take-profit, no stop, 30-minute limit. It wins 90 % of trades but averages **−0.10 %/trade**, 95 % range [−0.16, −0.04]. Each win nets +0.24 %, while the average loss is −3.0 %, so one loss wipes out about 13 wins. At 4–20× on a $3,000 account it ends in liquidation.
5. **What the average path after a flag looks like.** The mean barely moves, staying within ±0.1 % of the flag close for 3 hours. The median drifts down slowly, to −0.5 % at 60 min and −1.1 % at 180 min. The spread widens fast, to about ±2 % by 30 min. There is no dependable "dip, then bounce". The lowest point in the first 30 min is at a median depth of −2.3 % (mean −3.4 %). It comes in minutes 0–2 for 23 % of flags, minutes 3–9 for 21 % and minutes 10–29 for 56 %. Waiting before buying barely changes how far price goes for you or against you (see the structure table).

**Correction to the previous study (flag_x, 10-07):** the earlier "+0.3 % TP ≈ +0.008 %/trade, break-even" figure was too high by about 0.1 %/trade. That study allowed the take-profit to fill on prices from the first 8 seconds of the first bar, which came *before* the buy (a look-ahead error). Measured from ticks after the fill, the same rule averages −0.098 %/trade, which is clearly negative.

**Verdict:** there is no mechanical rule for trading the flag. I am not proposing a scout line. Any edge in the operator's hand trades comes from his own judgement or information the bot does not record, such as the order book or how price reacts to round numbers. A fixed delay, dip or continuation condition, take-profit, stop and time limit do not reproduce it.

---

## Data

- **Cohort:** cohort-a, the engine-reachable first-flag events from `flag_x/paths5.pkl`: 2,320 events on 417 pairs, 2026-01-10 → 2026-10-08. Long only, one entry per flag.
- **1m klines,** flag close → +181 min. 2,265 events came from local caches (`backtest_cache/k1m*`, `manualall/m1fetch`) and 55 were fetched.
  - **Binance requests: 55** (weight 2 each), one thread, 0.6 s apart. **Peak X-MBX-USED-WEIGHT-1M = 86.** No 418/429, no pause needed, 0 delisted or empty.
- **Validation:** 1m data rebuilt into 5m bars vs the 5m paths: 83,496 blocks, 0.0 % with a high or low gap > 0.05 %. 1 event has fewer than 181 bars; it is dropped from cells it cannot fill.
- **d=0 entry:** `entry_raw`, the first tick at or after flag close + 8 s (1,886 events from ticks, 434 from the 5m open). The first 1m bar's high and low are rebuilt from **post-fill ticks only** (fix described above).
- **Costs:** entry taker 0.045 % + entry slip 0.035 %. TP is a maker limit order (0.018 %, no slip). SL and time exits are taker 0.045 % + 0.05 % slip. A stop fills at the lower of the stop price and the bar open (gap risk).
- **Same-bar TP and SL:** counted as SL (primary). This affects 2.34 % of stop-cell trades. The mean gap between TP-first and SL-first is 0.042 %/trade (max 0.165 %, only in deep-negative tight-stop cells). No conclusion changes.

## Pre-registration (written 2026-10-08 02:38 UTC, before any P&L; file `scratchpad/flag_math/PREREG.txt`)

- **Grid:** 8 × 4 × 7 × 6 × 4 = **5,376 cells**.
  - Delay d ∈ {0 (flag + 8 s), 1, 2, 3, 5, 10, 15, 30} min.
  - Condition: any / dip (price at d below the flag close) / dip05 (≥ 0.5 % below) / cont (above).
  - TP ∈ {0.2, 0.3, 0.5, 0.75, 1, 1.5, 2} %.
  - SL ∈ {none, 1, 1.5, 2, 3, 5} %.
  - T ∈ {15, 30, 60, 120} min.
- **Selection:** rank on Jan–May (N ≥ 200 per cell), freeze the top 5, test on Jun–Oct, then the reverse. Day-block bootstrap, per month, leave-one-month-out (LOMO), shuffled-time null, 30–50 % haircut.
- **Addendum (02:38 UTC, before P&L):** the data window is 181 min, so the null draws the pseudo-flag minute uniformly from 1–30 min after the flag.
- **Post-result data fix (disclosed):** the first-bar look-ahead at d=0 was found because d=0 / dip05 showed a 100 % hit rate at +0.3 %. That bar was rebuilt from post-fill ticks. This is a data-correctness fix, not a parameter fit; the grid and selection rules are unchanged.

## Results

### Grid overview (full sample)

| Measure | Value |
|---|---|
| Cells with avg > 0 after costs | 27 of 5,376 |
| Median cell | −0.18 %/trade |
| Best cell | +0.079 %/trade |
| Cells positive in both halves (N ≥ 200 each) | 10 of 5,208 |
| Same count under the shuffled-time null | mean 15, p95 44 |

Every stop setting does worse on average than no stop: avg cell −0.14 % with no stop, −0.17 % with a 3 % stop, −0.22 % with a 1 % stop.

### Split-sample selection (the key test)

| Direction | Top-5 in-sample avg | Top-5 out-of-sample avg | Out-of-sample positive |
|---|---|---|---|
| Rank Jan–May → test Jun–Oct | +0.13 … +0.20 % | **−0.08 … −0.27 % (mean −0.149)** | 0 / 5 |
| Rank Jun–Oct → test Jan–May | +0.08 … +0.12 % | **−0.63 … +0.07 % (mean −0.182)** | 3 / 5 |

The Jan–May winners were all "wait 15 min, buy only if price is above the flag close (cont), TP 1–2 %". They lost on Jun–Oct, so less than 0 % of the in-sample edge survived out of sample.

### Shuffled-time null (200 draws: same events, pseudo-flag at a random minute 1–30)

| Statistic | Real flag | Null mean | Null p95 | P(null ≥ real) |
|---|---|---|---|---|
| Best Jan–May cell | +0.200 | +0.225 | +0.412 | 0.52 |
| Its Jun–Oct result | −0.084 | −0.134 | +0.189 | 0.41 |
| Best Jun–Oct cell | +0.121 | +0.195 | +0.387 | 0.82 |
| Its Jan–May result | −0.381 | −0.094 | +0.135 | 0.97 |
| Best full-sample cell | +0.079 | +0.136 | +0.279 | 0.77 |
| Top-5 A→B out of sample | −0.149 | −0.128 | +0.150 | 0.56 |

The real flag is not better than a random minute after it.

### Cells examined in detail (full sample, SL-first)

| Cell | N | WR | Avg % | Day 95 % CI | Avg win | Avg loss | Worst | Max consecutive losses |
|---|---|---|---|---|---|---|---|---|
| d0 cont, TP 1.5, no SL, 120 m (only top-5 member positive both ways) | 1,034 | 78.6 % | +0.079 | [−0.122, +0.261] | +1.42 | −4.86 | −25.0 | 4 |
| d0 cont, TP 2, SL 3, 120 m | 1,034 | 61.7 % | +0.050 | [−0.088, +0.182] | +1.89 | −2.92 | −3.14 | 5 |
| d15 cont, TP 2, no SL, 30 m (Jan–May best) | 1,076 | 62.9 % | +0.069 | [−0.100, +0.237] | +1.76 | −2.79 | −22.3 | 6 |
| **d0 any, TP 0.3, no SL, 30 m (operator style)** | 2,320 | 89.8 % | **−0.098** | **[−0.160, −0.043]** | +0.24 | −3.05 | −24.6 | 3 |
| d0 any, TP 0.5, no SL, 30 m | 2,320 | 84.7 % | −0.095 | [−0.165, −0.030] | +0.44 | −3.04 | −24.6 | 3 |

**Per month, for the best cell (d0 cont TP 1.5 120 m):**

| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct (N=19) |
|---|---|---|---|---|---|---|---|---|---|
| −0.27 | +0.08 | +0.23 | +0.16 | +0.08 | +0.03 | +0.02 | +0.14 | +0.08 | +0.62 |

- Leave-one-month-out ranges +0.06 to +0.12.
- No pair concentration: the top 2 pairs carry 5.6 % of the loss.
- After the 30–50 % haircut it is ≈ +0.04–0.055 %. The CI still spans zero, and the cell failed the A→B selection test, so it does not pass.

### $ per trade and book simulation

Assumptions: $630 margin per trade, $3,000 cross account, at most 2 positions at once. The liquidation check is crude and worst-case: the new trade's MAE and the open trades' MAE are assumed to hit together.

| Cell | Leverage | $/trade avg | Worst trade $ | Final equity | Max drawdown | Liquidated |
|---|---|---|---|---|---|---|
| d0 cont TP 1.5 120 m | 4× | +$1.99 | −$630 | $4,849 | 50 % | no |
| d0 cont TP 1.5 120 m | 6× | +$2.99 | −$945 | $5,774 | 75 % | no |
| d0 cont TP 1.5 120 m | 20× | +$9.97 | −$3,149 | — | — | **yes, at trade 10 (Jan)** |
| d0 cont TP 2 SL 3 120 m | 6× | +$1.89 | −$119 | $5,279 | 54 % | no |
| d0 cont TP 2 SL 3 120 m | 20× | +$6.29 | −$395 | $385 | 93 % | **yes** |
| operator style TP 0.3 30 m | 4× / 6× / 20× | −$2.5 / −$3.7 / −$12.4 | up to −$3,097 | — | — | **yes, at all three** |

Even the positive-in-sample cells carry 50–75 % drawdowns at 4–6×. At 20×, a single flag that drops 5–25 % ends the account. The −25 % tail is real: the 5th percentile of MAE is −9.4 %.

## Structure: the average path after a flag

**Return vs the flag close (all 2,320 events):**

| Minute | Mean | Median | % above flag close | p25 / p75 |
|---|---|---|---|---|
| 1 | +0.01 | −0.02 | 47 % | −0.44 / +0.43 |
| 3 | +0.03 | 0.00 | 48 % | −0.74 / +0.78 |
| 5 | +0.02 | 0.00 | 49 % | −0.93 / +0.89 |
| 10 | −0.00 | −0.10 | 47 % | −1.39 / +1.18 |
| 15 | +0.00 | −0.14 | 47 % | −1.64 / +1.33 |
| 30 | +0.04 | −0.22 | 47 % | −2.29 / +1.92 |
| 60 | −0.11 | −0.53 | 44 % | −3.14 / +2.08 |
| 90 | +0.12 | −0.62 | 44 % | −3.69 / +2.56 |
| 120 | +0.04 | −0.82 | 42 % | −4.06 / +2.77 |
| 180 | −0.08 | −1.11 | 42 % | −4.87 / +3.07 |

The mean stays flat, the median sinks slowly, and the spread widens fast. Before costs this is a coin flip with fat tails; after the ~0.13–0.16 % round-trip cost it is negative.

**Low in the first 30 min:** median −2.3 % (mean −3.4 %). It falls in minutes 0–2 for 23 % of flags, 3–9 for 21 % and 10–29 for 56 % (median minute 12). In the first 60 min the median low is −3.0 % at median minute 24. Any "best dip" sits at a different minute every time, so there is no fixed delay that catches it.

**MFE / MAE by entry delay (median, % of entry):**

| d | MFE 30 m | MAE 30 m | MAE 30 m (mean) | MFE 120 m | MAE 120 m | Hold to 30 m (mean) | Hold to 120 m (mean) |
|---|---|---|---|---|---|---|---|
| 0 | +2.22 | −2.32 | −3.31 | +3.73 | −3.99 | +0.07 | +0.07 |
| 1 | +2.17 | −2.31 | −3.32 | +3.69 | −4.06 | +0.01 | +0.03 |
| 3 | +2.09 | −2.25 | −3.27 | +3.59 | −3.98 | +0.04 | +0.03 |
| 5 | +2.07 | −2.25 | −3.23 | +3.61 | −3.97 | +0.05 | +0.02 |
| 10 | +2.03 | −2.15 | −3.09 | +3.45 | −3.82 | +0.08 | −0.03 |
| 15 | +1.96 | −2.08 | −2.99 | +3.43 | −3.72 | +0.03 | −0.02 |
| 30 | +1.76 | −1.93 | −2.83 | +3.27 | −3.60 | −0.14 | −0.10 |

The "Hold to …" columns are gross, before costs. At every delay the move for you (MFE) and against you (MAE) are about the same size, so on average the entry minute does not matter.

## What could not be tested

- **Order book / depth / spread** at the flag. Not recorded; no historical L2 data.
- **The operator's discretion:** choosing which flags to take, reading the tape, sizing up into strength, and exiting early on "feel". Only fixed rules were tested.
- **Shorts and scaling in or out:** outside this brief (long only, one entry per flag).
- **Pre-flag features** (volume, breadth, BTC state) as filters: deliberately not in the grid to keep the search size honest. Earlier FRENZY pre-ON studies refuted these (see memory: FRENZY pre-ON lines refuted).
- **Leftover first-minute error at d=0** for the 434 events without ticks: entry is the 5m open at the flag close, so the first bar is fully post-fill. These events are fine.
- **Real liquidation and maintenance-margin tiers per symbol:** the book simulation uses a crude 1 % maintenance rule.

## Files (scratch: `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/flag_math/`)

- `PREREG.txt`: the pre-registration.
- `fetch.py`, `cov.py`, `build.py`, `bar0.py`: data preparation.
- `outcomes.py`: the full outcome array.
- `grid.py`: the grid and split-sample selection.
- `null.py`: the shuffled-time null.
- `detail.py`: cell detail, CI, months, book simulation.
- `structure.py`: the average-path tables.
- Outputs: `grid.pkl`, `null.pkl`, `detail.pkl`, `outcomes.npz`, `m1/`.
