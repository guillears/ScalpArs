# EXH early-spike deep dive: is an RSI turn in the first 30 min a pause before a second leg? (2026-11-26)

**Question (operator):** EXHAUSTION_SHORT entries 0–0.5 h after the Frenzy spike lost in the backtest (216 · WR 38 % · −0.23 %/fill, 61 % stopped). The hypothesis is that an early RSI turn is only a pause, and the pump's second leg then squeezes the new shorts. Study this deeper, and show it on our real fills with today's filters.

**What was already done (prior-study check):**
- reports/EXH_SPIKE_AGE_2026-11-26.md: hour buckets, the < 1 h and < 0.5 h Pattern-C bar, and DCR.
- reports/ENTRY_WL_2026-11-18.md: `hrs_since_spike` / `hrs_since_peak` at the old TP 10, read as a runner share. It found nothing.

**What is new here:**
- A: the path mechanism.
- B: minute buckets.
- C: a W/L screen inside the 0–0.5 h bucket.
- D: real fills re-priced at today's exit.
- F: an exhaustive 2D screen of hours × every variable.
- E: re-states the bar.

Script: `scripts/EXH_EARLY_SPIKE_DEEP_2026-11-26.py`. CSVs: `reports/EXH_EARLY_SPIKE_DEEP_2026-11-26*.csv`. Chart: `.png`.

## Method
- **Backtest.** TP_GRID 2026-11-20 EXH cell F7, which is today's exit (DL 609): TP min(20×ATR, 7 %) maker, SL min(1.5×ATR, 6 %), 24 h. Net includes the taker 0.05 / maker 0.02 fees, slip ≥ 0.10 and funding.
  - Today's filters are applied: spot required, EXH_SSC_OIVOL removed upstream, and one position per coin.
  - That leaves **3,358 kept fills, 2023-07 → 2026-11**. The breakeven WR is **43.3 %**.
  - DESIGN is before 2025-07-01; HOLDOUT is from that date on.
- **Path data.** The mechanism uses the 1m path for 3,357 fills and 5m for 1, read from the k1m caches.
- **Validating the walker.** Every booked stop reached its stop price on the path, and every TP traded through (100 % / 100 %).
- **Run peak.** This is the max 5m high from the spike bar to the signal bar, the same definition as `unified_short_build`. It is known for 100 % of fills.
- **Real fills.** All 17 closed EXH fills (11-18 → 11-26) come from exports_archive plus the newest export, deduped on (opened_at, pair, direction). No probe flag exists.
  - Filters: **ZKP has no spot-symbol stamp** (it was the first fill), so it is treated as no-spot and blocked. EXH_SSC_OIVOL blocks 0. That leaves **16 kept fills on 7 days**.
  - Each kept fill is re-priced at today's exit on its real 1m path from its real fill price, with real funding.
  - **Parity:** walking each fill at its own stamped TP/SL reproduces **17/17 booked exit kinds**. The median exit-price error is 0.07 %. TP7 era: booked avg −2.21 vs model −2.31; TP10 era: +1.16 vs +1.30.

## A · Mechanism: do early stops come from a second leg making a new high? (backtest)
| bucket | N | stop % | stops that made a **new high** before the stop | stop sits below the run peak | median peak above entry | median min to stop | stops ≤ 60 min | new high within 24 h (any exit) | stopped, then traded to TP level ≤ 24 h | avg net |
|---|---|---|---|---|---|---|---|---|---|---|
| 0–0.5 h | 216 | **61** | **22 %** | 84 % | 6.2 % | 66 | 49 % | 49 % | **32 %** | −0.23 |
| 0.5–1 h | 364 | 50 | 15 % | 91 % | 7.4 % | 85 | 41 % | **37 %** | 36 % | +0.90 |
| 1–4 h | 953 | 52 | 24 % | 85 % | 7.0 % | 70 | 46 % | 48 % | 46 % | +0.46 |
| 4–24 h | 1,216 | 50 | 44 % | 70 % | 5.8 % | 71 | 45 % | 55 % | 53 % | +0.71 |
| 24–72 h | 609 | 50 | 29 % | 78 % | 7.0 % | 55 | 52 % | 52 % | 46 % | +0.76 |

The pattern is the same in DESIGN and HOLDOUT. In the 0–0.5 h bucket, 27 % (DESIGN) and 17 % (HOLDOUT) of stops made a new high before the stop.

**Reading.**
- **The theory's specific mechanism is not what kills early entries.** Only about 1 in 5 early stops was a second leg above the run peak. 84 % of early fills have their stop *below* the peak, so a stop is a ~1.5×ATR bounce (median 4.2 %), not a new high.
- Within 24 h, early fills make a new high no more often than ≥ 1 h fills (49 % vs 48–55 %). The chart's right panel shows the same thing: the 0–0.5 h curve lies on top of the ≥ 1 h curve.
- What *is* different about early fills is that more of them get stopped (61 vs ~50 %). After the stop they also less often go on to dump to the TP level (32 % vs 36–53 %). In other words, the early bounce holds rather than wicks, but it does not make a new high.
- The 0.5–1 h bucket is the outlier on the good side: fewest new highs (37 %) and the best result.

## B · Finer dose-response (N · WR · avg net % · stop %)
| min after spike | ALL | DESIGN | HOLDOUT |
|---|---|---|---|
| 0–10 | 12 · 33 % · −0.83 · 67 | 4 · 0 % · −4.70 | 8 · 50 % · +1.11 |
| 10–20 | 84 · 35 % · **−0.67** · 65 | 39 · 36 % · −0.07 | 45 · 33 % · **−1.19** |
| 20–30 | 120 · 42 % · +0.14 · 58 | 54 · 37 % · −0.43 | 66 · 45 % · +0.60 |
| 30–45 | 179 · 50 % · +1.22 · 47 | 100 · 35 % · −0.27 | 79 · 70 % · +3.11 |
| 45–60 | 185 · 46 % · +0.59 · 53 | 98 · 43 % · +0.42 | 87 · 49 % · +0.79 |
| ≥ 60 | 2,778 · 50 % · +0.64 · 50 | 1,051 · 47 % · +0.61 | 1,727 · 51 % · +0.65 |

**Half-years at 10–20 min:** 23-H2 +1.53 · 24-H1 −1.19 · 24-H2 +0.42 · 25-H1 −2.25 · 25-H2 −1.15 · 26-H1 −0.80 · 26-H2 −2.70. That is 5/7 negative, N 4–28 per half.

**Is the weakness monotonic toward 0?**
- In ALL, roughly yes up to 30 min (−0.83 → −0.67 → +0.14). After that, 30–45 min is the *best* bucket.
- DESIGN is weak all the way to 45 min.
- HOLDOUT is weak only at 10–20 min.
- The stable piece is **10–20 min (both blocks negative, WR 33–36 %)**. The 0–10 min bucket has N = 12, which is noise.
- The edge is not a clean ramp, and the cut point moves between blocks.

## C · Inside 0–0.5 h: what separates winners from losers at entry? (N 216)
**Coverage.** The screen draws on every exh.pkl entry-time column (122 candidates): price/run, volume decay, RSI, OI, L/S, taker, funding, premium, BTC/market and the g_* trigger flags. **93 of them produced testable groups** with N ≥ 15 on each side. The rest are rare trigger flags or constant inside the bucket, so they are a blind spot here. Each variable was tested at sign, median, terciles and =1 for flags, which gives 472 tests. The null is 300 within-month shuffles, family-wise max |t|, with a 95 % threshold of 3.91.

**Result: no survivor.** The best test is `glob_chg4h ≤ −0.081`, a 4 h fall in the global long/short account ratio: 108 · 28 % · −1.28 vs 107 · 49 % · +0.77. Its t is 2.9, p_raw is 0.007 and **p_fw is 0.68**. It is direction-consistent in DESIGN (Δ −1.5) and HOLDOUT (Δ −2.8). Next come `btc_ret1h` (a mid-tercile hole, which is the confound shape) and `top_acct_chg4h_rel ≤ −7.9 %` (p_fw 0.97).

**The theory's own predictions fail at entry.** In each case below, "bad" is the theory's prediction and Δ is DESIGN / HOLDOUT.

| prediction ("bad" =) | result |
|---|---|
| near the peak (`dd_peak_pct` top tercile) | −0.54 vs rest −0.07; Δ −1.12 / +0.30, flips |
| peak very recent (`hrs_since_peak` ≤ 0.25 h) | −0.58 vs +0.65; Δ −2.65 / 0.00, holdout flat |
| volume still high (`vmult15` top) | opposite sign: high vmult is the better half |
| taker buys still dominant (`taker_1h` top tercile) | +0.08 vs rest, nothing |
| OI still rising (`oi_chg15m` / `oi_chg30m` top) | nothing |

## D · Real fills, today's filters, re-priced at today's exit
| hours since spike | N | days | WR | re-priced avg | booked avg | stop % | new high before exit | new high ≤ 24 h |
|---|---|---|---|---|---|---|---|---|
| 0–0.5 h | **1** | 1 | 0 % | −5.56 | −5.38 | 100 | 0/1 | 1/1 |
| 0.5–1 h | 0 | – | – | – | – | – | – | – |
| 1–4 h | 8 | 5 | 63 % | +1.63 | +0.40 | 38 | 1/8 | 2/8 |
| 4–24 h | 7 | 5 | 57 % | +1.08 | −0.95 | 43 | 1/7 | 4/7 |
| all kept | 16 | 7 | 56 % | **+0.94** | −0.55 | 44 | 2/16 | 7/16 |

- **The single real early fill (INIT, 0.25 h) did not make a new high before its stop.** Its run peak was 12.7 % above entry, and the stop was hit at +5.4 % after 11 min. It went above the peak only later, after the stop.
- Only 2 of the 7 real stops were new highs before the stop, both at ≥ 3.8 h.
- One-size DCR (0.61 × equity, 9 calendar days): today **0.82 %/day**, max DD −17.0 %, worst day −14.1 %. Blocking < 0.5 h gives 1.20 %/day, −14.1 %, −14.1 %. That is **N = 1 removed fill = one window**, which is evidence of nothing.
- Booked TP10-era results improve under today's TP7, mostly from ENSO and RARE turning TP-before-stop.
- The live `entry_frenzy_hours` and the backtest `hrs_since_spike` (5m-quantised) were not cross-checked bar-for-bar.

## E · Locked Pattern-C bar + breakeven-bot read (frozen 0–0.5 h zone)
| zone | N | WR vs BE 43.3 | avg | 10-day-block 95 % upper bound | windows (days) | max day / pair share of loss | verdict |
|---|---|---|---|---|---|---|---|
| 0–0.5 h ALL | 216 | 38.4 ✓ | −0.23 | **+0.27 ✗** | 189 ✓ | 2.6 / 3.4 % ✓ | **FAIL (confidence)** |
| 0–0.5 h DESIGN | 97 | 35.1 ✓ | −0.46 | +0.30 ✗ | 82 | 3.9 / 3.8 % | FAIL |
| 0–0.5 h HOLDOUT | 119 | 41.2 ✓ | −0.04 | +0.78 ✗ | 107 | 4.5 / 4.7 % | FAIL |
| < 20 min (post-hoc) | 96 | 34.4 ✓ | −0.69 | +0.20 ✗ | 88 | 2.7 / 4.2 % | FAIL |

**Breakeven-bot read.** This is one size with no slot cap, so drawdowns are inflated; read them relative to each other.

| | DCR ALL | DESIGN | HOLDOUT | max DD | worst day |
|---|---|---|---|---|---|
| today | 0.821 % | 0.385 % | 1.456 % | −62.7 % | −26.6 % |
| block < 0.5 h | 0.856 % | 0.429 % | 1.478 % | −54.4 % | −26.3 % |

- The zone vs rest Δ is −0.90 %/fill. After the 30–50 % in-sample haircut it is **−0.45 to −0.63**, on about 5 % of fills.
- The DCR gain is +0.02 to +0.04 pp/day and becomes ≈ 0 after the haircut.
- The worst day is unchanged.

## F · Exhaustive 2D screen: hours × every entry variable (backtest, today's exit)
- **Hours axis:** < 0.5 / ≥ 0.5, < 1 / ≥ 1, and DESIGN terciles.
- **Second axis:** all **121** variables at sign / median / terciles / flag. Each cell is compared with the rest of the scored fills (unscored is not counted as rest).
- **Size:** 3,569 cells with N ≥ 15.
- **Selection:** DESIGN selects with thresholds frozen on DESIGN; HOLDOUT confirms. The null is 300 within-month shuffles on DESIGN, and the **family-wise 95 % |t| is 5.16**. t is also shown day-clustered.
- **Coverage and blind spots:**
  - prem_last / prem_chg1h / prem_vs24h are only 60 % scored in DESIGN (100 % in HOLDOUT).
  - mcap is 77 / 90 %, st_wr24h_closed 86 %, st_prior_fill_won 27 / 35 %.
  - btc_d1_regime and vshare_rel cannot be split (a single value in DESIGN).

**Top 10 DESIGN cells** (N · WR · avg vs rest):

| # | cell | DESIGN | t (day) | p_fw | HOLDOUT | Δ HOLDOUT | halves same sign |
|---|---|---|---|---|---|---|---|
| 1 | h<0.5 & prem_chg1h mid-tercile | 18 · 11 % · −2.86 (+0.69) | −4.8 | 0.13 | 33 · 48 % · +0.82 (+0.71) | +0.11 ✗ | 4/6 |
| 2 | h<0.5 & oi_chg15m mid | 24 · 12 % · −2.62 | −4.0 | 0.31 | 27 · 56 % · +1.66 | +0.96 ✗ | 4/7 |
| 3 | h<0.5 & bar_ret_pct mid | 37 · 22 % · −1.98 | −4.2 | 0.34 | 28 · 54 % · +1.60 | +0.90 ✗ | 5/7 |
| 4 | h<0.5 & prem_last mid | 26 · 23 % · −2.10 | −3.8 | 0.42 | 48 · 38 % · −0.22 | −0.95 (t −1.3) | 5/6 |
| 5 | h≥1 & prem_vs24h mid | 198 · 58 % · +1.92 | +4.0 | 0.45 | 376 · 45 % · +0.16 | −0.68 ✗ | 3/6 |
| 6 | h≥0.5 & prem_vs24h mid | 247 · 55 % · +1.72 | +4.2 | 0.47 | 432 · 46 % · +0.33 | −0.48 ✗ | 3/6 |
| 7 | h<1 & gap1h ≤ 4.26 | 169 · 31 % · −0.82 | −3.8 | 0.49 | 160 · 53 % · +1.30 | +0.64 ✗ | 4/7 |
| 8 | h≥1 & top_pos_chg4h low | 302 · 55 % · +1.51 | +3.7 | 0.51 | 793 · 51 % · +0.73 | +0.02 | 5/7 |
| 9 | h≥1 & top_pos_chg1h low | 272 · 55 % · +1.56 | +3.6 | 0.62 | 671 · 53 % · +0.92 | +0.32 | 5/7 |
| 10 | h<0.5 & breadth ≤ 0.615 | 36 · 19 % · −2.03 | −3.6 | 0.68 | 81 · 42 % · +0.00 | −0.74 (t −1.1) | 6/7 |

**No survivor.**
- No DESIGN cell beats the family-wise null; the best p_fw is 0.13.
- The four strongest are all "early × middle tercile". That is the non-monotonic confound shape, and all four flip or fade in HOLDOUT.
- Row 10 (early & low breadth) keeps its sign in HOLDOUT and in 6/7 halves, but HOLDOUT is only 0.00 vs +0.74 at t −1.1. **At most it is a pre-registered observe candidate** with a frozen definition: EXH hrs < 0.5 ∧ breadth ≤ 0.615. It is not a filter, and breadth is market-wide, so it counts in window units.

**Heat table: minute bucket × the best-ranked variable (prem_chg1h, DESIGN terciles), N · WR · avg:**

| min | T1 ≤ −0.031 | T2 | T3 > 0.122 | unscored |
|---|---|---|---|---|
| 0–10 | 8 · 50 % · +1.40 | 2 · 0 % · −4.82 | – | 2 · 0 % · −5.76 |
| 10–20 | 38 · 26 % · −1.76 | 22 · 41 % · +0.17 | 1 · 0 % · −6.30 | 23 · 43 % · +0.57 |
| 20–30 | 52 · 40 % · −0.03 | 27 · 33 % · −0.69 | 16 · 62 % · +1.77 | 25 · 40 % · +0.35 |
| 30–45 | 72 · 58 % · +1.96 | 48 · 46 % · +0.73 | 22 · 59 % · +2.05 | 37 · 35 % · −0.09 |
| 45–60 | 69 · 51 % · +0.81 | 52 · 52 % · +1.33 | 33 · 33 % · −0.80 | 31 · 39 % · +0.35 |
| ≥ 60 | 814 · 51 % · +0.71 | 638 · 47 % · +0.57 | 903 · 53 % · +0.80 | 423 · 43 % · +0.25 |

There is no coherent gradient across the row, so it is noise.

## Chart
`reports/EXH_EARLY_SPIKE_DEEP_2026-11-26.png`.
- **Left:** mean price path vs entry (5m closes, −2 h … +12 h), split into 0–0.5 h vs ≥ 0.5 h and winners vs losers. Early winners and losers trace the same paths as late ones. The only difference is the mix: 38 % vs 49 % winners.
- **Right:** cumulative share of fills trading above the pre-entry run peak. The 0–0.5 h curve overlaps the ≥ 1 h curve, and 0.5–1 h is lowest.

## Verdict
- **The second-leg theory is not supported as the mechanism.**
  - Early entries do get stopped more (61 vs 50 %) and recover less often after the stop.
  - But only 22 % of those stops are new highs above the spike peak.
  - Early fills make new highs no more often than late fills.
  - None of the theory's entry-time fingerprints separates early losers (near the peak, fresh peak, volume or taker buying still strong, OI rising).
- **The weakness is real-looking but small and unstable.**
  - It holds at 10–20 min in both blocks, but the cut moves between blocks (DESIGN to 45 min, HOLDOUT 10–20 min).
  - It fails the locked Pattern-C bar on confidence (upper bound +0.27).
  - The DCR gain is about 0 after the haircut.
- **Real fills cannot speak.** There is 1 kept fill under 0.5 h, and it was a bounce stop, not a new high.
- **Recommendation: no bot change.**
  - Keep the existing frozen **observe line** "EXH hrs_since_spike < 0.5 h" (from EXH_SPIKE_AGE). Tally it at each batch review in window units, with N, WR, re-priced avg and whether the stop was a new high.
  - Optionally register the 2D observe candidate "hrs < 0.5 ∧ breadth ≤ 0.615" with its frozen definition. Neither is a filter candidate.

## Caveats and blind spots
- The < 0.5 h cut and the 10–20 min read are post-hoc / in-sample, and the INIT case motivated the ask.
- The backtest has no slot cap, so max DD is inflated.
- Backtest entries are at the 5m open after the 15m signal; live fills come ~8 s later with drift.
- The live `entry_frenzy_hours` definition was not reconciled bar-for-bar with the backtest spike bar.
- The real sample is 16 fills / 7 days.
- **Blind spots (not on any row):**
  - order-book depth and liquidations at entry;
  - second-level aggTrades / taker bursts after entry;
  - spot-side intrabar flow;
  - the live-only entry_frenzy_* stamps (not in the backtest);
  - news.
- Premium vars are 60 % scored in DESIGN, and btc_d1_regime / vshare_rel are unsplittable in DESIGN.
