# LITE green retry — "on a green candle, wait for the next red one instead of giving up" (2026-10-19)

Research only. No code, config or git change. Scripts: `scripts/study_lite_green_retry_deep.py` (real era, variants frozen in its docstring),
`scripts/study_lite_green_retry_deep_bt.py` (two-year backtest), `scripts/study_lite_green_retry_deep_report.py` (tables).
Full tables: `reports/LITE_GREEN_RETRY_DEEP_2026-10-19_tables.md`. Per-fill CSVs: `_era_fills.csv`, `_bt_fills.csv` (V0 plus every variant's retry fills),
`_era_variants.csv`, `_bt_variants.csv`. `validate_against_master.py`: **ALL CHECKS PASS**.

## The answer in plain English

- **AGT 10-16 is real.** The green 01:55 bar (it closed at 02:00) used up LITE's one judgement of the stretch. The next red bar opened at 02:15, and an entry at its close would have booked **+2.94 (TP +3, ticks)**. The replica finds the same trade.
- **On the 18 days of the real era the rule looks mildly positive, but it is not significant.**
  - The primary rule (SXr: next red bar of the same stretch, LITE filters re-checked) adds **49 fills on 17 days**: 59 % WR, **+0.16 %/fill**.
  - That is **+$218** at today's size, with a day CI of [−$934, +$1,251] and P = 0.67.
  - The best window, 15 min, shows +$380 (P 0.78). It was picked from 10 cells, so it carries the in-sample haircut.
- **The two-year backtest refutes it.** That is 2025-01 → 2026-10-11, 3,505 green-refused stretches, and the same engine functions, exit and size.
  - The retry adds **1,927 fills at −0.109 %/fill** (56 % WR, while LITE C breaks even at 58 %).
  - Day-bootstrap P(mean > 0) = 0.04. That is **−$5,661** over two years.
  - It loses in **every period**: 2025 −0.13, 2026-H1 −0.08, 2026-H2 −0.10.
  - Only 9 of 22 months are positive, and leave-one-month-out stays between −$6.5k and −$4.0k.
  - Every window from 5 to 60 min and "until the stretch breaks" is negative. Skipping the filter re-check is worse still (P(Δ>0) ≤ 0.01).
- **Theory does not rescue it.**
  - On the year, entering ON the green bar earns −0.04, and waiting for the next red on the same stretches earns −0.10.
  - The cheaper entry (median −0.23 % below the green close, 5 min later) buys nothing. The stretches LITE refuses for a green bar are not worse because of that one candle.
  - The real era shows the opposite (+0.26 vs −0.11 on 46 stretches), but that sits inside the year's noise.
- **Verdict: neither a ship bar nor a candidate.**
  - It fails every promotion gate on the year: N is huge, the mean is negative, and it is negative in all three periods.
  - The observe-only scout being built is the right place for it, but with a **low prior**. Keep it as a frozen watch line with a retire rule. Do **not** change LITE.

## 1. What DECISION_LOG 243 actually refuted vs this rule

| test | what was retried | which bar | exit / pricing | result |
|---|---|---|---|---|
| **DL 243 / FRENZY_LITE_ABOVE_CLOSES 6 vs 12 §6 "DEFER"** (2026-10-07, 2026-01 → 10 year) | **Any** refusal: green bar, gvol ≥ 1, dislocation, pair-day cap | The first later bar of the stretch that passes **all** filters (median 3 bars later) | Old FRENZY lock (−3 until +3, then max(+2, peak−2)), 12 h, +12 s, ticks, no slots | 1,992 fills · −0.005; the 1,272 *extra* entries −0.093 |
| **B19 review item 3, LITE "added"** (2026-10-11, same year) | **Green-bar refusals only** | Next red / flat bar ≤ 60 min, any stretch, eligible ∧ not bearish, LITE status not re-checked | +3 / −3 / 12 h, ticks, no slots | 719 · 47 % · **−0.20** |
| Part F (this morning, real era) | Green only | Next red bar of the same stretch passing LITE status | LITE C, $2,450, replica | 48 · +0.23 · +$295 [−761, +1,287] |

- **DL 243 never tested the narrow rule.** It mixed green refusals with gvol, dislocation and cap refusals, under an exit that no longer exists.
- **The narrow green-only, ≤ 60 min version was already tested and refuted** on the 2026 year (B19 item 3, −0.20 %/fill). That test used +3 / −3, had no slots, and did not require the stretch to survive.
- **This study is the first to test the operator's exact rule** (same stretch, LITE status re-checked, today's LITE C exit and size, slots / pair-held / WILLY hold) on two years.
- **Today's state:** `frenzy_gvol_max = 0`, so the gvol < 1 gate is OFF. It is not applied in any variant here.

## 2. Frozen variants (docstring of `study_lite_green_retry_deep.py`, written before any variant result was read)

A green judged bar still judges the stretch on the normal path. The variant adds ONE pending retry, which ends at its first qualifying bar (filled or refused) or when the window closes.

| cell | rule |
|---|---|
| V0 | Today: the green bar consumes the stretch. |
| **SXr (primary)** | First red / flat bar (close ≤ open) of the **same stretch**, **until the stretch breaks**. LITE status is re-checked (verified, not in state, ≥ 12 closes, vol < 100×, 2–16.75 h, 24 h vol ≥ $20M). Open filters are re-checked (bearish day, pair held, 2 slots, pair-day cap 3, 1 % dislocation). The WILLY hold refuses without judging (≤ 6 bars). |
| S5r / S15r / S30r / S60r | Same as SXr, within N min of the green close. |
| S5n … SXn | Same windows, **no** re-check: only pair held, slots and the WILLY hold apply. |
| W60 | WILLY-style: the first strictly red bar ≤ 60 min, the stretch need not survive, open filters re-checked. |
| RETRY_ANY | Reference: retry every refused bar until a fill (DL 243 class). |

**Pricing** (identical in both blocks):
- Entry: first print ≥ close + 8 s, × 1.001.
- Exit, LITE C: TP +3 if vol ≥ 20× ∧ BTC 1h EMA20 slope > 0, else +2. Stop −3. 12 h.
- Fees: maker TP at the line; taker stop plus 0.03 slip.
- Path: ticks first, then a 1m stop-first fallback (flagged).
- Size: $2,450 notional (8× at the $1,225 account).

**Family:** 10 cells besides V0 and the reference. Any best cell carries the 30–50 % haircut.

## 3. Block ① — real era (replica, 2026-10-02 → 10-19 14:54; master + live period; 18 days)

**Parity:**
- The replica's V0 has the same LITE fill bar as **27 of 28** real LITE fills (Part F's 96 %).
- 0 retry bars coincide with a real fill, as expected: the engine had consumed those stretches.
- 94 green judged bars.

| variant | LITE N | Σ $ (LITE + LONG) | Δ $ vs V0 [day CI] | P(Δ>0) | added · days | added WR (breakeven) | added avg %/fill [day CI] | lost today LITE ($) | LONG lost ($) |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 61 · 59 % · +0.143 | +216 | – | – | – | (be 56 %) | – | – | – |
| S5r | 85 | +310 | +95 [−653, +808] | 0.61 | 26 · 14 d | 54 % (55) | −0.09 [−1.22, +1.06] | 2 (−153) | 1 (+1) |
| **S15r** | 102 | +596 | **+380 [−663, +1,344]** | 0.78 | 43 · 17 d | 63 % (57) | +0.34 [−0.65, +1.26] | 2 (−29) | 1 (+1) |
| S30r | 106 | +587 | +371 [−673, +1,327] | 0.76 | 47 · 17 d | 62 % (56) | +0.30 [−0.61, +1.17] | 2 (−29) | 1 (+1) |
| S60r | 107 | +510 | +295 [−880, +1,336] | 0.71 | 48 · 17 d | 60 % (56) | +0.23 [−0.69, +1.16] | 2 (−29) | 1 (+1) |
| **SXr (primary)** | 108 | +434 | **+218 [−934, +1,251]** | 0.67 | **49 · 17 d** | 59 % (56) | **+0.16 [−0.75, +1.09]** | 2 (−29) | 1 (+1) |
| S5n | 105 | −178 | −394 [−1,221, +440] | 0.18 | 46 | 48 % (57) | −0.48 | 2 (−153) | 1 |
| S15n / S30n / S60n / SXn | 135–142 | −118 … −32 | −334 … −248 | 0.26–0.31 | 76–83 | 54–55 % | −0.19 … −0.14 | 2 | 1 |
| W60 | 110 | +405 | +189 [−1,015, +1,275] | 0.65 | 51 · 17 d | 59 % (56) | +0.13 | 2 (−29) | 1 |
| RETRY_ANY (ref) | 128 | +715 | +499 [−608, +1,589] | 0.82 | 70 | 60 % | +0.23 | 3 (−105) | 1 |

- **LODO choice of cell (days):** it picks S15r on 16 of 18 days. The held-out Δ is +$227 [−862, +1,214], P 0.66.
- **Concentration (SXr added):**
  - Top pair: MAGIC, 7 % of |Σ|. Top day: 10-11, 18 %.
  - With the top 5 removed: −0.16 %/fill. The positive mean rests on a handful of +3 TPs.
- **Wait time (era SXr):** ≤ 5 min +0.03 (25) · 5–15 min +0.75 (18) · 15–60 min −0.70 (5) · > 60 min −3.12 (1).
- **SXr retry outcomes:** 49 opened, **33 refused by the bearish-day block** (they would not trade today either), 7 expired, 3 slots, 1 pair held, 1 cap.
  - ⚠ The scout must apply the bearish block. Otherwise its shadow cohort contains fills the engine would refuse.

### ① Per-fill list — primary SXr retry fills
Bar times are bar OPEN UTC; entry is at the red bar's close + 8 s. All paths are ticks except SIREN 10-19 (1m).

| pair | green bar | red bar | wait | move green→entry | age h | vol × | TP | net % | exit | $ |
|---|---|---|---|---|---|---|---|---|---|---|
| MEGA | 10-02 01:20 | 01:25 | 5 | −0.70 | 10.0 | 75 | +3 | +2.94 | TP | +72 |
| BTW | 10-02 04:40 | 04:45 | 5 | −0.19 | 16.1 | 2 | +2 | −3.12 | STOP | −76 |
| NIGHT | 10-02 10:00 | 10:05 | 5 | −1.16 | 2.1 | 34 | +3 | +2.94 | TP | +72 |
| MANA | 10-02 14:15 | 14:20 | 5 | +0.02 | 7.1 | 24 | +3 | −3.12 | STOP | −76 |
| ONE | 10-03 17:20 | 17:35 | 15 | +0.04 | 8.4 | 9 | +2 | +1.94 | TP | +47 |
| MOVR | 10-03 17:40 | 18:05 | 25 | +5.57 | 5.2 | 65 | +3 | +2.94 | TP | +72 |
| AIN | 10-04 04:35 | 04:40 | 5 | −0.04 | 14.3 | 91 | +3 | +2.94 | TP | +72 |
| STRK | 10-04 07:20 | 07:35 | 15 | +0.84 | 14.0 | 16 | +2 | +1.94 | TP | +47 |
| AIO | 10-04 16:15 | 16:20 | 5 | −0.63 | 4.2 | 28 | +3 | +2.94 | TP | +72 |
| MUBARAK | 10-05 05:50 | 05:55 | 5 | −0.10 | 15.9 | 26 | +2 | −3.12 | STOP | −76 |
| NIL | 10-05 05:45 | 06:05 | 20 | +0.29 | 4.0 | 29 | +2 | −3.12 | STOP | −76 |
| 1000FLOKI | 10-05 09:35 | 09:40 | 5 | −0.17 | 12.0 | 4 | +2 | −3.12 | STOP | −76 |
| NIL | 10-05 16:15 | 16:20 | 5 | −0.19 | 14.2 | 19 | +2 | +1.94 | TP | +47 |
| EDU | 10-05 19:55 | 20:05 | 10 | −0.54 | 7.1 | 44 | +2 | +1.94 | TP | +47 |
| EDU | 10-06 00:50 | 01:00 | 10 | +0.05 | 12.0 | 37 | +3 | +2.94 | TP | +72 |
| API3 | 10-06 05:15 | 05:25 | 10 | +0.68 | 13.0 | 30 | +2 | +1.94 | TP | +47 |
| BR | 10-06 13:55 | 14:00 | 5 | −2.31 | 5.8 | 21 | +3 | −3.12 | STOP | −76 |
| HEMI | 10-07 16:35 | 16:40 | 5 | −0.73 | 9.2 | 5 | +2 | +1.94 | TP | +47 |
| GRIFFAIN | 10-08 14:35 | 14:55 | 20 | +1.09 | 4.2 | 36 | +2 | −3.12 | STOP | −76 |
| PYTH | 10-08 19:25 | 19:40 | 15 | +1.32 | 5.4 | 7 | +2 | +1.94 | TP | +47 |
| GRIFFAIN | 10-08 21:40 | 21:45 | 5 | −0.32 | 11.1 | 3 | +2 | +1.94 | TP | +47 |
| PYTH | 10-09 02:15 | 02:25 | 10 | +0.18 | 12.2 | 4 | +2 | +1.94 | TP | +47 |
| STRK | 10-09 02:25 | 02:30 | 5 | −0.74 | 15.0 | 20 | +2 | +1.94 | TP | +47 |
| MAGIC | 10-09 10:25 | 10:30 | 5 | −1.36 | 3.8 | 59 | +3 | −3.12 | STOP | −76 |
| JCT | 10-09 11:55 | 12:00 | 5 | −0.95 | 16.4 | 99 | +3 | +2.94 | TP | +72 |
| PIXEL | 10-09 23:50 | 10-10 00:05 | 15 | +1.06 | 5.9 | 29 | +3 | +2.94 | TP | +72 |
| CAP | 10-10 13:45 | 13:55 | 10 | −0.33 | 9.8 | 25 | +3 | −3.12 | STOP | −76 |
| GRIFFAIN | 10-10 17:15 | 17:25 | 10 | +0.22 | 4.9 | 3 | +2 | −3.12 | STOP | −76 |
| CHIP | 10-10 20:25 | 20:30 | 5 | −0.54 | 2.1 | 45 | +3 | +2.94 | TP | +72 |
| GRIFFAIN | 10-11 01:10 | 01:15 | 5 | −0.39 | 12.8 | 2 | +2 | −3.12 | STOP | −76 |
| MANTA | 10-11 17:35 | 18:30 | 55 | +0.24 | 10.2 | 91 | +3 | −3.12 | STOP | −76 |
| W | 10-11 19:50 | 19:55 | 5 | −0.72 | 2.1 | 86 | +3 | −3.12 | STOP | −76 |
| MAGIC | 10-11 20:30 | 20:35 | 5 | −0.24 | 12.1 | 90 | +3 | −3.12 | STOP | −76 |
| SKL | 10-13 02:00 | 02:05 | 5 | −1.47 | 9.7 | 53 | +2 | +1.94 | TP | +47 |
| Q | 10-13 05:30 | 05:35 | 5 | −0.64 | 3.0 | 23 | +3 | +2.94 | TP | +72 |
| BAT | 10-13 06:35 | 08:10 | 95 | +0.11 | 5.1 | 94 | +3 | −3.12 | STOP | −76 |
| AVA | 10-14 04:20 | 04:25 | 5 | −0.40 | 14.0 | 25 | +3 | +2.94 | TP | +72 |
| DEEP | 10-14 17:35 | 17:45 | 10 | −0.37 | 10.7 | 15 | +2 | +1.94 | TP | +47 |
| AVA | 10-14 19:25 | 19:35 | 10 | −0.94 | 2.7 | 59 | +2 | +1.94 | TP | +47 |
| MAGIC | 10-15 15:20 | 15:30 | 10 | −0.37 | 12.7 | 27 | +3 | −3.12 | STOP | −76 |
| ANIME | 10-15 21:10 | 21:25 | 15 | +1.22 | 4.8 | 71 | +3 | −3.12 | STOP | −76 |
| ANIME | 10-15 23:05 | 23:10 | 5 | −0.18 | 6.5 | 22 | +3 | −3.12 | STOP | −76 |
| **AGT** | **10-16 01:55** | **02:15** | 20 | +0.38 | 11.1 | 73 | +3 | **+2.94** | TP | **+72** |
| TRUTH | 10-16 21:10 | 21:20 | 10 | +0.85 | 5.4 | 11 | +2 | −3.12 | STOP | −76 |
| FOLKS | 10-17 05:10 | 05:15 | 5 | −0.80 | 11.2 | 72 | +2 | +1.94 | TP | +47 |
| Q | 10-17 06:25 | 06:35 | 10 | −0.20 | 7.3 | 40 | +2 | +1.94 | TP | +47 |
| AVA | 10-17 20:20 | 20:35 | 15 | +0.79 | 6.0 | 22 | +3 | +2.94 | TP | +72 |
| ERA | 10-18 01:10 | 01:20 | 10 | −0.12 | 16.6 | 56 | +3 | +2.94 | TP | +72 |
| SIREN | 10-18 23:55 | 10-19 00:00 | 5 | −0.84 | 7.9 | 17 | +2 | −3.12 | STOP | −76 (1m) |

**Total: 49 fills · 29 W / 20 L · +$218 net of cannibalisation** (the 2 lost today-LITE fills −$29 and the 1 LONG fill +$1.4).

## 4. Block ② — per batch (Δ $ vs V0, LITE + LONG)

| variant | B16 | B17 | B18 | B19 | B20 | B21 | B22 | B23 |
|---|---|---|---|---|---|---|---|---|
| **SXr** | +187 | +129 | −105 | −131 | 0 | +162 | −110 | +86 |
| S15r | +115 | +205 | −29 | −54 | 0 | +239 | −182 | +86 |
| SXn | +34 | +100 | −403 | −92 | 0 | +133 | −44 | −53 |
| W60 | +187 | +53 | −105 | −160 | 0 | +239 | −110 | +86 |

- SXr is positive in 4 of 7 trading batches and negative in 3. It is not a consistent per-batch effect.
- SXr retry fills per batch: B16 6 · B17 10 · B18 3 · B19 14 · B20 0 · B21 5 · B22 5 · B23 6.

## 5. Block ③ — two-year backtest (2025-01-01 → 2026-10-11)

**Universe:**
- An engine-function enumeration of every LITE stretch judgement over 613 pairs and 1.86 M walked bars. It uses the real `frenzy_walk` / `normal_hour_usd` / `frenzy_lite_status` and the bars within 20 h of every spike lead.
- Concurrency:
  - LITE slots and the pair-day cap are simulated.
  - Pair held and the WILLY global hold come from the **yr6 replay seed-1** fills (every non-LITE position holds its pair).
  - The bearish block and the LITE C slope are rebuilt from BTC 5m.
- **Parity:** the yr6 replay's seed-1 LITE fills are 1,454. My V0 has 1,339 of them as fills (92 %) and 1,409 as judged bars (97 %), with 90 % sign agreement.
  - My V0 has 2,183 fills, because the ≤ 25-pair shortlist and the global position cap are not modelled. The reachable-episode sensitivity below handles that.
- **Warm-up duplicates:** the TODAY file has only in-window rows, and duplicates are dropped on (opened_at, pair, direction).
- **Paths:** ticks first. 38 % of retry fills (41 % of V0) use the 1m stop-first fallback, mostly in 2025. 2.7 % of bars are unpriced (no data) and listed as UNPRICED.

| variant | LITE N | Σ $ | Δ $ vs V0 [day CI] | P(Δ>0) | added · days | added WR (be) | added avg %/fill [day CI] | P(added > 0) | lost today LITE |
|---|---|---|---|---|---|---|---|---|---|
| V0 | 2,183 · 56 % · −0.093 | −4,982 | – | – | – | (be 58 %) | – | – | – |
| S5r | 3,141 | −8,313 | −3,332 [−7,574, +839] | 0.06 | 1,061 · 473 d | 55 % (58) | −0.130 [−0.29, +0.03] | 0.06 | 103 |
| S15r | 3,753 | −9,004 | −4,023 [−9,758, +1,429] | 0.08 | 1,711 · 548 d | 56 % (58) | −0.089 [−0.22, +0.04] | 0.09 | 141 |
| S30r | 3,915 | −10,531 | −5,550 [−11,635, +392] | 0.03 | 1,896 | 56 % | −0.108 | 0.04 | 164 |
| S60r | 3,931 | −10,318 | −5,336 [−11,270, +290] | 0.03 | 1,913 | 56 % | −0.101 | 0.06 | 165 |
| **SXr (primary)** | 3,944 | −10,643 | **−5,661 [−11,605, +305]** | **0.03** | **1,927 · 561 d** | 56 % (58) | **−0.109 [−0.23, +0.02]** | **0.04** | 166 |
| S5n … SXn | 3,554–4,716 | −11.6k … −15.5k | −6.6k … −10.5k (CI < 0) | 0.00–0.01 | 1,483–2,722 | 55–56 % | −0.19 … −0.12 | ≤ 0.01 | 112–189 |
| W60 | 4,023 | −10,739 | −5,757 [−12,152, +262] | 0.03 | 2,016 | 56 % | −0.111 | 0.03 | 176 |
| RETRY_ANY (ref) | 4,106 | −10,985 | −6,003 [−12,406, +421] | 0.03 | 2,103 | 56 % | −0.115 | 0.03 | 180 |

- **Concentration:** no single pair or day carries more than 1 % of the added Σ, so the loss is broad.
- **Top 5 removed:** −0.117 %/fill.
- **Sensitivity — replay-reachable episodes only** (2,656 episodes with a seed-1 FRENZY-family fill; V0 1,441 fills · −0.104):

  | variant | added | added avg % [CI] | Δ $ [CI] | P |
  |---|---|---|---|---|
  | SXr | 837 | −0.054 [−0.24, +0.12] | −1,378 | 0.25 |
  | S15r | 738 | −0.036 | −632 | 0.38 |
  | W60 | 871 | −0.032 | −1,019 | 0.32 |
  | SXn | 1,129 | +0.007 | +269 | 0.55 |

  Smaller losses, still ≤ 0, still below breakeven WR.

### ③ Per period (SXr / S15r)

| period | V0 LITE N · avg | SXr added · avg [CI] | SXr Δ $ (P) | S15r added · avg | S15r Δ $ (P) |
|---|---|---|---|---|---|
| 2025 | 1,167 · −0.258 | 1,010 · −0.133 [−0.30, +0.04] | −3,022 (0.09) | 896 · −0.140 | −2,922 (0.07) |
| 2026-H1 | 629 · +0.060 | 598 · −0.075 [−0.31, +0.15] | −1,876 (0.14) | 529 · −0.050 | −1,125 (0.24) |
| 2026-H2 | 387 · +0.155 | 319 · −0.097 [−0.38, +0.20] | −763 (0.26) | 286 · −0.002 | +25 (0.51) |

- The added retry fills are **below today's LITE fills in every period** and negative in every period.
- In 2026-H2, the period nearest today, they run −0.10 against LITE's +0.155.

### ③ Per month and leave-one-month-out (Δ $ vs V0)

| variant | months Δ > 0 | Σ Δ | LOMO range | worst | best |
|---|---|---|---|---|---|
| SXr | 9 / 22 | −5,661 | −6,523 … −4,026 | 2026-05 −1,635 | 2025-04 +862 |
| S15r | 10 / 22 | −4,023 | −5,006 … −2,784 | 2026-05 −1,238 | 2025-04 +983 |
| W60 | 8 / 22 | −5,757 | −6,986 … −4,132 | 2026-05 −1,626 | 2025-04 +1,229 |

Per-month detail is in the tables file.

- **Wait-time dose (bt SXr):** ≤ 5 min −0.11 (1,014) · 5–15 min −0.05 (670) · 15–60 min −0.22 (218) · > 60 min −1.30 (15).
  - The era's "5–15 min" bump (+0.75 on 18 fills) does not survive: the year shows −0.05.
- **2026-only (Jan–Oct):** 888 retry fills · −0.085. This is consistent with the B19 item-3 refutation (−0.20 under +3 / −3, without status re-check).

## 6. Mechanism and theory

| cohort | N | WR | avg % | age h | vol × | vs VWAP % | TP +3 share | BTC 1h slope > 0 | MFE % | stopped | wait min | move green→entry | green bar ret |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| era: today's LITE fills | 61 | 59 % | +0.143 | 8.0 | 31 | +1.81 | 46 % | 79 % | +3.52 | 41 % | – | – | – |
| era: SXr retry fills | 49 | 59 % | +0.159 | 9.2 | 28 | +2.52 | 51 % | 65 % | +3.94 | 41 % | 5 | −0.19 | +0.54 |
| bt: today's LITE fills | 2,183 | 56 % | −0.093 | 6.9 | 22 | +1.87 | 34 % | 67 % | +2.98 | 43 % | – | – | – |
| bt: SXr retry fills | 1,917 | 56 % | −0.110 | 7.1 | 23 | +2.94 | 37 % | 68 % | +2.81 | 44 % | 5 | −0.23 | +0.56 |

**Are retry fills different?** Barely.
- Median wait is 5 min (88 % of fills within 15 min).
- The entry is about 0.2 % below the green close.
- The one consistent difference: they sit **further above the spike VWAP** (+2.9 vs +1.9 % on the year). They buy a stretch that has already extended.
- Same age, same volume multiple, same BTC state, same stop rate.

**Do they cannibalise?** Very little.
- Era: 2 of today's LITE fills lost (−$29) and 1 LONG fill (+$1.4 at blocker size).
- Year: 166 of today's fills displaced, mostly by slots. Their Σ was +$523, so displacement makes the year result slightly worse.

**Theory — why a green signal bar could be worse, and whether a later red bar inherits it.** The hypothesis was that a green bar is a chase (buying the top of a mini-pump). A later red bar would then be the pullback that resets the price.

| cohort (LITE C, non-bearish, no slots) | N | WR | avg % |
|---|---|---|---|
| bt: enter ON the green bar (all green-judged stretches) | 2,446 | 57 % | −0.037 |
| bt: same stretches with a retry — green-bar entry | 1,846 | 57 % | −0.042 |
| bt: same stretches — next-red entry | 1,846 | 56 % | **−0.100** |
| bt: green bar ≥ median size (+0.54 %) / < median | 1,223 / 1,223 | 58 / 56 % | +0.053 / −0.127 |
| era: enter ON the green bar | 58 | 50 % | −0.332 |
| era: same stretches — green-bar entry / next-red entry | 46 | 54 / 61 % | −0.110 / +0.263 |
| era: green bar ≥ median / < median | 29 / 29 | 38 / 62 % | −0.96 / +0.30 |

- **The year evidence does not support the chase story for LITE.**
  - The green-refused stretches are about as bad whether entered on the green bar or the next red bar.
  - Waiting for the cheaper price makes it slightly worse (−0.06 %/fill, with 331 outcomes flipping). This is the same "waiting buys cheaper but loses the runners" mechanism as MINHOURS_AND_WAITRED and B19 item 3.
- **The green refusal works as a stretch filter, not a candle filter.** The stretches whose first signal bar closes green are, as a group, worse than the red-first ones (bt retry −0.11 vs V0 −0.09). The later red candle inherits that.
- The era's big-green-bar effect (−0.96) reverses on the year (+0.05). It is noise at N = 29.

## 7. Expectancy vs the gates (CLAUDE.md)

| bar | real era (SXr) | two-year (SXr) | verdict |
|---|---|---|---|
| Expectancy > 0 for the added cohort | +0.16, P 0.65 | **−0.109, P 0.04** | ✗ |
| WR vs LITE C breakeven | 59 % vs 56 % | 56 % vs 58 % | ✗ on the year |
| ≥ today's LITE mean | +0.16 vs +0.14 | −0.11 vs −0.09 (every period below) | ✗ |
| Consistency (periods / months) | 4 of 7 batches | 0 of 3 periods, 9 of 22 months | ✗ |
| After the 30–50 % haircut | +0.08 … +0.11, CI spans 0 | n/a (negative) | ✗ |
| Best cell S15r (family of 10, LODO-picked) | +0.34 (in-sample) | −0.089 (P 0.09); H2 −0.002 | ✗ |

- **Ship bar:** not met. The year refutes it with N ≈ 1,900 on ≈ 560 days.
- **Observe-first bar:** not met either. A cohort that is negative in all three periods of two years is not an observe-first "candidate". It is a refuted idea with an attractive anecdote.
- Keeping the observe-only scout running is cheap and honest, because it shadows today's live distribution, which the backtest only approximates. It must run as a **low-prior watch line with a frozen retire rule**, not a promotion path.

## 8. Recommendation (plain English)

**Recommendation:**
- **Do not change LITE.** One judgement per stretch stays. A green signal bar still ends the stretch.
- AGT 10-16 was a real +2.94 the rule would have caught. Over two years, though, the same rule adds about 1,900 trades that lose about 0.11 % each:
  - −$5.7k at today's size;
  - worse in 2025, still negative in 2026;
  - below today's LITE in every period.
- The 18-day real-era gain (+$218) is within noise and sits on a few +3 TPs.
- **Let the sibling's scout (`scripts/scout_lite_green_retry.py`) run as an observe-only line,** with the gate below frozen now:
  - **Cohort:** every live stretch refused `FRENZY_LITE_GREEN_BAR`, counted from the scout's commit.
  - **Shadow entry (= SXr):** the next red / flat bar (close ≤ open) of the SAME stretch that passes `frenzy_lite_status`, entered at close + 8 s.
    - It must also pass the **open filters the engine would apply**: bearish-day block, pair-day cap, 1 % dislocation. Slots / pair held are recorded as a flag, not refused. *(Superseded at commit, DL 315 review: the scout REFUSES pair held / LITE slots / WILLY hold too — a shadow fill must be a trade the live engine could have taken.)*
    - In the real era 33 of 82 retries were bearish-day refusals. Leaving them in would contaminate the cohort.
  - **Pricing:** LITE C on ticks (1m stop-first fallback flagged), maker TP at the line.
  - **Read** at ≥ 20 shadow fills on ≥ 10 UTC days.
  - **CANDIDATE (operator decides)** only if ALL hold:
    - shadow mean > 0 with day-bootstrap P ≥ 0.90;
    - shadow mean ≥ the same-period live LITE mean;
    - shadow WR ≥ LITE C breakeven WR;
    - no day / pair ≥ 50 % of Σ.
  - Even then, any ship needs the 30–50 % haircut and must be re-checked against this two-year refutation. A forward pass alone cannot overturn N ≈ 1,900.
  - **RETIRE** if the shadow mean ≤ 0 at the read. Never re-fit the window (no 15-min variant afterwards: S15r was the in-sample best cell here).

## 9. Blind spots

- **Backtest universe:**
  - No ≤ 25-pair shortlist and no global max-open-positions cap, so V0 has 2,183 fills vs the replay's 1,454.
  - The reachable-episode sensitivity shrinks the loss (SXr −0.054, S15r −0.036) but does not flip it.
  - Pair held and the WILLY hold use replay seed-1 positions. The replay's own LITE positions are replaced by mine.
- **Fidelity:**
  - 2025 paths are about 40 % 1m stop-first, which is conservative on wick-heavy coins.
  - 2.7 % of priced bars have no data (UNPRICED refusals, symmetric across variants).
- **The real era is a replica, not live:**
  - WIDE, manual and momentum positions are not modelled there.
  - The bearish block and LITE C slope are rebuilt stamps.
- **LONG at blocker size:** cannibalisation of LONG is visible only in the era (1 fill). The year uses fixed replay LONG fills.
- **gvol is not applied** (the gate is OFF today). If it is re-armed, this study needs re-running.
- **Not re-tested here:** a different exit for retry fills. Exits were held at today's LITE C on purpose (one change at a time).
