# FRENZY seconds-delay entry study (2026-10-08)

**Status:** research only. No code or config changed, nothing committed. Local tick caches only; no network calls were made.

## Plain-English summary

**Question:** does buying a few **seconds** (or a few minutes) after FRENZY fires make money? I tested two moments:
- **(A)** the first FRENZY flag, and
- **(B)** the moment the FRENZY setup turns ON.

Delays ran from 2 s to 5 min. Each delay was paired with a fixed take-profit and a time limit, with or without a stop. Every entry and exit was priced from real trade prints, after costs.

**Answer: no.** No combination passes the bar for either moment.

1. **The quick-scalp style (+0.2 to +0.5 %, no stop, 5 to 30 min) loses money at every delay.** That is 108 combinations per moment. All 108 are negative for ON, and 107 of 108 for the flag; the one exception is +0.004 %. The typical result is **−0.05 to −0.15 % per trade** with a 75–95 % win rate. A loss averages −1.5 to −4.5 %, which wipes out 10 to 40 wins. The same holds whether you wait 2 s, 10 s, 30 s or 5 min.
2. **The best-looking cells are "+2 to +3 % target, no stop, 1–2 h".** None of them survives being chosen on one half of the year and checked on the other.
   - Flag: the best Jan–May pick reads −0.02 % on Jun–Oct.
   - ON: the Jan–May picks all lose on Jun–Oct (−0.09 to −0.17 %), and the Jun–Oct picks all lose on Jan–May (−0.04 to −0.17 %).
   - No cell in either cohort has a day-block 95 % range above zero.
   - Their losses average −4.5 to −6 %, and the worst single trades are −30 to −59 %. At 4–20× that is liquidation.
3. **There is no seconds-level pattern to exploit.**
   - The first 60 s are a coin flip: 49 % of events are up at +60 s and 49–50 % are down, in both cohorts.
   - The mean move in the first 30 s is −0.03 to +0.01 %.
   - A "pop then dump" does happen. In about 48 % of events price jumps +0.5 % or more inside the first minute, and 38 % of those are below the signal price 5 min later. But the events that did *not* pop end lower more often (61 %). So the pop predicts mild continuation, not a dump, and the next 9 minutes after an up first minute average about −0.01 %. Nothing in the first minute tells a pop-and-hold from a pop-and-dump.
4. **Flag timing vs a random moment.**
   - Flag (A): picking an entry second at random within the first 10 min after the flag gives results as good as the real flag second (P ≈ 0.5–0.9). The flag second adds nothing.
   - ON (B): the real ON second does beat random seconds on the *in-sample* best (P = 0.01–0.04). Seven cells are positive in both halves, against 0.4 under the null. So the minutes right after ON are slightly better than later minutes. That reflects how a pump fades over 10 minutes, and it is too small to pay for costs and the tail: every such cell has a day-CI spanning zero and fails the half-year swap.

**Verdict:** there is nothing to propose, not even a free observe-only scout line. Whatever edge the operator's hand entries have, it is not in "X seconds after FRENZY + fixed TP/T". This matches the minute-level flag study (`FRENZY_FLAG_TRADE_MATH_2026-10-08.md`) and the ON-scalp study (`FRENZY_ON_SCALP_STUDY_2026-10-06.md`).

---

## Data & coverage

| | Cohort A: first flag | Cohort B: fresh ON |
|---|---|---|
| Source | `flag_math/ev2.pkl` (cohort-a engine-reachable first-flag events) | `FRENZY_ENGINE_COHORT_2026-10-05.csv` (real `services.frenzy.frenzy_walk` fresh-ON bars) ∩ `FRENZY_WIDE_OVERNIGHT_COHORT_2026-10-06.csv` live_elig, the same 1,443-signal universe as the ON-scalp study |
| Reference time / price | flag 5m close / last print before it | ON bar close / last print before it |
| Events | 2,320 | 1,443 |
| **With ticks (used)** | **1,884** (81 %), 398 pairs, 270 days | **1,430** (99 %), 267 pairs, 255 days |
| Excluded, no tick file | 436. These are mostly the same events that had no ticks in the minute study. **Oct: 37 of 64 missing** | 13 |
| Period | Jan 10 → Oct 8 2026 (Jan–May 1,038 / Jun–Oct 846) | Jan 10 → Sep 27 2026 (Jan–May 820 / Jun–Oct 610). The cohort build stops at Sep 27 |

**How the ticks were used:**
- Ticks come from `reports/backtest_cache/ticks_q` (first) and `ticks` (fallback). These are full-UTC-day aggTrade files.
- Every event needs ticks from close − 120 s to close + 8,300 s.
- Ticks are compressed to 1-second bins: first print, high/low excluding the first print of the entry bin, and last print.
- Entry = the first print at or after close + d. Because delays are whole seconds, this is exact. The median entry lag is 0 s at every delay.
- TP and SL can only be hit by prints *after* the entry print, so there is no look-ahead.
- A brute-force tick check of first-hit times matched the binned engine exactly on the events tested.

**Costs:**
- Entry: taker 0.045 % + slip 0.035 %.
- TP: maker limit, 0.018 %. A TP win nets TP − 0.098 %.
- Stop or time exit: taker 0.045 % + slip 0.05 %, so 0.175 % round trip.
- A stop fills at the stop line, or at the bin's open if the bin opens through the line. Same-second TP and SL counts as SL.

## Pre-registration

Written **2026-10-08 02:49:44 UTC, before any P&L** (`scratchpad/flag_secs/PREREG.txt`).

- **Grid:** d ∈ {0 (= +2 s), 5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300} s × TP ∈ {0.2, 0.3, 0.5, 0.75, 1, 2, 3} % × T ∈ {5, 15, 30, 60, 120} min × SL ∈ {none (primary), 2, 3, 5 %} = **1,680 cells per cohort** (420 with no stop).
- **Selection:** rank on Jan–May, freeze the top 5, read them on Jun–Oct, then the reverse. This is done for two families: all cells, and no-stop only.
- **Uncertainty:** day-block bootstrap 95 % CI, per-month results, leave-one-month-out (LOMO), and a 30–50 % haircut.
- **Shuffled-time null:** 200 draws. Same events, but the pseudo-event is placed at close + U{0..600} s.
- **Ship bar:** positive out of sample in both directions AND full-sample CI lower bound > 0.

## Results

### Grid overview (full sample)

| | A: flag | B: ON |
|---|---|---|
| Cells with avg > 0 | 146 / 1,680 (no-stop 62 / 420) | 23 / 1,680 (all no-stop) |
| Cells with day-CI lower bound > 0 | **0** | **0** |
| Median cell | −0.092 % | −0.119 % |
| Mean cell by stop: none / 2 / 3 / 5 % | −0.066 / −0.095 / −0.081 / −0.077 | −0.087 / −0.137 / −0.125 / −0.125 |
| Best cell | d300s TP3 T120 no-SL **+0.187** | d0s TP2 T60 no-SL **+0.067** |

Stops never help. For ON, every stop cell is negative.

### Split-sample selection (the key test)

The "all cells" and "no-stop only" families select the same cells, because the top 5 are all no-stop.

| Cohort | Direction | Top-1 pick (in-sample → OOS) | Top-5 OOS mean | OOS positive |
|---|---|---|---|---|
| A | Jan–May → Jun–Oct | d10s TP3 T120: +0.262 → **−0.024** | −0.012 | 2 / 5 |
| A | Jun–Oct → Jan–May | d30s TP3 T120: +0.118 → **+0.184** | +0.138 | 5 / 5 |
| B | Jan–May → Jun–Oct | d45s TP2 T120: +0.116 → **−0.087** | **−0.142** | 0 / 5 |
| B | Jun–Oct → Jan–May | d30s TP3 T120: +0.178 → **−0.162** | **−0.131** | 0 / 5 |

- **Cohort A** fails one direction.
- **Cohort B** fails both directions badly.

A's Jun→Jan direction looks good, but the shuffled-time null produces just as good a result from random seconds (next table). It is the long-hold +3 % pump drift, not the timing.

### Shuffled-time null (200 draws, pseudo-event at close + U{0..600} s)

| Statistic | A real | A null mean / p95 | A P(null ≥ real) | B real | B null mean / p95 | B P(null ≥ real) |
|---|---|---|---|---|---|---|
| Best full-sample cell | +0.187 | +0.210 / +0.266 | 0.74 | +0.067 | −0.005 / +0.055 | **0.04** |
| Best Jan–May cell | +0.262 | +0.274 / +0.346 | 0.65 | +0.116 | +0.014 / +0.081 | **0.01** |
| → its Jun–Oct | −0.024 | +0.077 / +0.234 | 0.88 | −0.087 | −0.114 / +0.017 | 0.38 |
| Best Jun–Oct cell | +0.118 | +0.198 / +0.293 | 0.91 | +0.178 | +0.167 / +0.267 | 0.40 |
| → its Jan–May | +0.184 | +0.173 / +0.283 | 0.46 | −0.162 | −0.259 / −0.097 | 0.17 |
| Top-5 A→B OOS | −0.012 | +0.053 / +0.166 | 0.84 | −0.142 | −0.114 / −0.028 | 0.71 |
| Cells positive in both halves | 51 | 47 / 105 | 0.36 | 7 | 0.4 / 2 | **0.02** |

- **A:** the flag second is no better than a random second in the next 10 minutes.
- **B:** the ON second is better than later seconds in-sample, so there is real decay over 10 minutes. But it does not carry out of sample, and no cell clears costs with confidence.

### Best cells, full sample (no-stop)

| Cohort | Cell | N | WR | Avg % | Day 95 % CI | Jan–May / Jun–Oct | Avg win / loss | Worst | Range by month | Haircut 30–50 % |
|---|---|---|---|---|---|---|---|---|---|---|
| A | d300s TP3 T120 | 1,884 | 66.1 % | +0.187 | [−0.02, +0.40] | +0.251 / +0.108 | +2.77 / −4.85 | −30.1 | −0.83 (Oct, N 27) … +0.68 | +0.09–0.13 |
| A | d20s TP3 T120 | 1,884 | 66.6 % | +0.160 | [−0.08, +0.37] | +0.234 / +0.068 | +2.78 / −5.06 | −57.7 | | |
| A | d5s TP2 T120 | 1,884 | 75.8 % | +0.125 | [−0.06, +0.30] | +0.169 / +0.071 | +1.87 / −5.35 | −34.0 | | |
| B | d0s TP2 T60 | 1,430 | 71.8 % | +0.067 | [−0.12, +0.25] | +0.073 / +0.058 | +1.86 / −4.51 | −30.1 | −0.37 (Jul) … +0.30 | +0.03–0.05 |
| B | d0s TP2 T120 | 1,430 | 77.6 % | +0.064 | [−0.15, +0.27] | +0.044 / +0.091 | +1.89 / −6.25 | −38.3 | | |

- These are the best of 1,680 cells, so they carry selection bias on top of the haircut.
- Pair concentration is low: the top 2 pairs carry about 5 % of the loss. The tail comes from many pairs, not a few bad ones.
- One fat loser cancels about 2–3 wins at TP3 and 2.5–3.5 wins at TP2.

### Quick-TP / no-stop row (operator style), avg net % per trade

| TP / T | A d0 (+2 s) | A d10 | A range over 12 delays | B d0 (+2 s) | B d10 | B range over 12 delays |
|---|---|---|---|---|---|---|
| 0.2 / 5 m | −0.137 | −0.118 | −0.148 … −0.111 | −0.121 | −0.099 | −0.149 … −0.096 |
| 0.2 / 15 m | −0.137 | −0.104 | −0.155 … −0.104 | −0.138 | −0.096 | −0.139 … −0.075 |
| 0.2 / 30 m | −0.128 | −0.091 | −0.164 … −0.091 | −0.148 | −0.101 | −0.162 … −0.067 |
| 0.3 / 5 m | −0.124 | −0.100 | −0.147 … −0.099 | −0.096 | −0.099 | −0.140 … −0.083 |
| 0.3 / 15 m | −0.122 | −0.067 | −0.144 … −0.067 | −0.107 | −0.096 | −0.132 … −0.066 |
| **0.3 / 30 m** | **−0.109** [−0.18, −0.04], WR 92 % | −0.039 | −0.143 … −0.039 | **−0.107** [−0.17, −0.04], WR 93 % | −0.107 | −0.150 … −0.059 |
| 0.5 / 5 m | −0.094 | −0.066 | −0.147 … −0.066 | −0.075 | −0.096 | −0.128 … −0.075 |
| 0.5 / 15 m | −0.076 | −0.027 | −0.128 … −0.027 | −0.069 | −0.083 | −0.126 … −0.047 |
| 0.5 / 30 m | −0.053 | **+0.004** | −0.119 … +0.004 | −0.051 | −0.091 | −0.145 … −0.028 |

- The win rate is 75–95 %, but the average loss is −1.5 % at 5 m, −2.5 to −3 % at 15 m and −3.5 to −4.5 % at 30 m. Worst trades are −8 to −37 %.
- The pattern across delays is noise, not a gradient. 10 s is a little less bad for A, but the same row for B is not.
- **Note:** 4–7 % of the TP 0.3 hits land inside the entry second itself, and 25–30 % within 5 s. A real limit order placed after a market fill would miss some of these, so live results would be slightly *worse* than the table.

## The seconds path (descriptive)

Return vs the reference price, in %, with each second forward-filled from the last print:

| s | A mean | A median | A % up | A p25 / p75 | B mean | B median | B % up | B p25 / p75 |
|---|---|---|---|---|---|---|---|---|
| 1 | −0.03 | −0.00 | 42 | −0.14 / +0.09 | −0.01 | 0.00 | 45 | −0.14 / +0.11 |
| 5 | −0.04 | 0.00 | 47 | −0.21 / +0.17 | −0.01 | 0.00 | 48 | −0.21 / +0.19 |
| 10 | −0.03 | −0.01 | 46 | −0.26 / +0.22 | +0.01 | 0.00 | 48 | −0.26 / +0.25 |
| 30 | −0.02 | 0.00 | 48 | −0.37 / +0.34 | +0.01 | 0.00 | 50 | −0.40 / +0.36 |
| 60 | +0.01 | 0.00 | 49 | −0.48 / +0.48 | +0.05 | 0.00 | 49 | −0.49 / +0.49 |
| 120 | +0.01 | 0.00 | 48 | −0.67 / +0.69 | +0.07 | −0.01 | 49 | −0.65 / +0.68 |
| 300 | +0.06 | −0.02 | 48 | −1.01 / +0.98 | +0.02 | 0.00 | 49 | −0.97 / +0.97 |
| 600 | +0.09 | −0.08 | 48 | −1.45 / +1.31 | +0.07 | −0.04 | 48 | −1.39 / +1.35 |

**First minute after the signal:**

| | A | B |
|---|---|---|
| Up / down / flat at +60 s | 48.7 / 49.6 / 1.8 % | 48.7 / 48.6 / 2.7 % |
| Highest point in the first 60 s, median (mean) | +0.47 (+0.81) % | +0.48 (+0.78) % |
| Lowest point in the first 60 s, median (mean) | −0.51 (−0.82) % | −0.50 (−0.76) % |
| Pop ≥ +0.5 % inside 60 s | 47.5 % | 48.7 % |
| → of those, below ref at 300 s | 38 % | 38 % |
| → of the rest, below ref at 300 s | 61 % | 61 % |
| Move 60 → 600 s after an up first minute, mean (median) | −0.01 (−0.11) % | −0.02 (−0.11) % |
| Move 60 → 600 s after a down first minute, mean (median) | +0.18 (0.00) % | +0.06 (−0.02) % |

**Reading:**
- The first minute is a symmetric ±0.5 % wiggle around the signal price, with no drift in either direction.
- There is no reliable "immediate pop then dump". The sign of the first minute only says where price sits at 5 min, because most of the 5-minute move has already happened by then. It does not forecast the next 9 minutes in a tradable way: the medians are −0.11 and 0.00 %.
- Waiting does not buy a better price on average. The mean entry vs ref ranges from −0.03 to +0.08 % across all delays, and the median is 0.00 % at every delay.

## Verdict against the pre-registered bar

| Cohort | Positive OOS both ways? | Any cell with CI lower bound > 0? | Pass |
|---|---|---|---|
| A: first flag | No. Jan–May → Jun–Oct top-1 is −0.024 | No (0 / 1,680) | **No** |
| B: fresh ON | No. Both directions negative, top-5 −0.14 / −0.13 | No (0 / 1,680) | **No** |

**No scout line is proposed.** Seconds-level timing after a FRENZY flag or ON bar, with a fixed TP, time cap and optional stop, has no edge after costs. The no-stop cells that look positive in-sample carry −30 % to −59 % single-trade tails, which fails the ruin leg at any live leverage.

## Caveats

- **d = 0 is close + 2 s, which is faster than live** (about 7.5–8 s to fill). d = 5 s and d = 10 s bracket the live latency, and both read the same.
- **Cohort A is missing 19 % of events (no ticks).** That includes 58 % of October (37 of 64), so October results for A are thin (N = 27).
- **Cohort B ends Sep 27** (the cohort build stops there).
- **Prices are float32 tick prices.** The 1-second bins are exact for entries. Stop fills use the line or the bin open, not the exact crossing print; this only matters for the secondary stop cells, which are all negative anyway.
- **One position per event, no book or overlap simulation,** because nothing passed the per-trade bar.

## Files

All in the scratchpad at `flag_secs/`:
- `PREREG.txt`: the pre-registration.
- `build.py`: the 1-second tick arrays and first-hit engine (binary-lifting sparse tables).
- `analyze.py`: the grid, selection, null and path tables.
- `supp.py`: CI over all positive cells, the quick-TP matrix, same-second share and months.
- `out_A.txt`, `out_B.txt`: full outputs, including every quick-TP cell for all 12 delays.
- `arr_*.npz`, `ev_*.pkl`, `res_*.pkl`: the underlying arrays and results.
