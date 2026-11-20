# DIST_LOSERS2: is there an entry filter for DISTRIBUTION_SHORT losers? (2026-11-20)

Research only. No bot code or config changed. Follows `reports/DIST_FB_LOSERS_2026-11-20.md` (13 mapped variables) and `reports/DIST_WL_2026-11-18.md` (115 variables on the pre-RSI cohort).

## 1. Question
After the 4 stops in 5 batch fills: does anything known at entry separate DIST winners from losers **inside today's kept cohort** (RSI1h ≥ 74.07), strongly enough to block the losers?

## 2. Method
- **Trades.** 1,547 backtest fills (DL 586 kept cohort, one position per pair, Exit B, slip 0.10 %, real funding). DESIGN 2023-07 → 2025-06 (559 trades) · HOLDOUT 2025-07 → 2026-11 (988).
  - Real fills: 5 in the batch, 0 in master, so the backtest carries this read (stated caveat).
- **Variables.** 112 entry-time variables. They include the sleeve-state variables (stops / wins in the last 24 h, prior fill in the episode) that the earlier studies did not have.
- **Bins.** Sign, terciles and quintiles, frozen on DESIGN.
- **Tests.** 710 one-variable cells, plus 42,519 two-variable cells (tercile × tercile, every pair).
- **Outcomes.** Booked net %, WORKED (did it reach the +1 % path label), and REACH.
- **Null.** 200 label shuffles; the 95 % line is the family-wise max |t|.
- **BTC daily split.** Separate 1D screens on HOLDOUT bull and bear, because entry rules can flip across regimes.
- **Leads.** The best cells were judged against the locked filter EXPECTANCY bar, using a 4 h-window bootstrap.
- Scripts: `scripts/DIST_LOSERS2_2026-11-20_screen.py`, `scripts/DIST_LOSERS2_2026-11-20_leads.py`. Data: `reports/DIST_LOSERS2_2026-11-20_data/`.

## 3. Screen vs the shuffled null (DESIGN)
Cells need n ≥ 30 on both sides. WORKED is tested with a two-proportion z, so all-0 or all-1 cells cannot blow up the t. The sign split counts as one test per variable.

| outcome | family | tests | best \|t\| | null 95 % | pass | HOLDOUT of the passers |
|---|---|---|---|---|---|---|
| booked net % | 1D | 710 | 3.29 | 4.14 | **0** | — |
| booked net % | 2D | 42,519 | 5.27 | 6.32 | **0** | — |
| WORKED | 1D | 710 | 3.66 | 3.80 | 0 | — |
| WORKED | 2D | 42,519 | 4.13 | 5.03 | 0 | — |
| REACH | 1D | 710 | 4.55 | 3.92 | 4 | t −1.9 · +0.3 · −1.5 · −2.2 |
| REACH | 2D | 42,519 | 6.19 | 5.43 | 1 | n < 30 on HOLDOUT (not testable) |

None of the REACH passes is a loser marker:
- **Top-trader 1 h position change < 0** (and its top quintile, the mirror) goes **less** far our way. When top traders did not cut in the last hour, the trade reaches further. That makes it a reach split, not a loss split: booked net does not pass.
- **Low 1 h ATR**, and **small 15 m range ∧ low 15 m RSI** (2D): a quieter coin moves less. This is mechanical.
- **OI up 9–25 % in 30 min (middle tercile):** this is priced in §4.

## 4. Leads vs the locked filter bar (the blocked side must lose)
| lead (would block) | block | N | windows | WR | sleeve breakeven WR | avg % | rest avg % | 95 % upper of avg |
|---|---|---|---|---|---|---|---|---|
| thin coin (24 h vol < $42M) | DESIGN | 186 | 173 | 35 % | 34 % | 0.00 | +1.54 | +0.68 |
| | HOLDOUT | 492 | 440 | 41 % | 37 % | +0.69 | +1.07 | +1.21 |
| | 2026 | 293 | 263 | 41 % | 37 % | +0.68 | +1.45 | +1.37 |
| OI 30 m change mid-tercile (+9 … +25 %) | ALL | 474 | 443 | 36 % | 36 % | +0.10 | +1.30 | +0.59 |
| small 15 m range (< 5 %) | ALL | 309 | 290 | 39 % | 36 % | +0.98 | +0.92 | +1.51 |

- **Thin coins** earn less but stay **positive** (5 of 7 half-years > 0). Blocking them would cut profitable trades. This is the "high WR but net-losing / weaker but positive" case, so it is a **sizing question, not a filter**. It stays the existing observe line.
- **OI mid-tercile** is a non-monotonic hole (both flanks win), which is a confound by our rule. It is also ≈ 0, not negative.

## 5. BTC daily regime split (HOLDOUT)
| regime | N | tests | best \|t\| | null 95 % | pass |
|---|---|---|---|---|---|
| bull | 456 | 253 | 3.04 | 3.86 | 0 |
| bear (the batch's regime, DIST_FB_LOSERS §5.4) | 531 | 255 | 3.39 | 3.85 | 0 |

The regime screens use terciles of each regime subset. The closest bear-regime hint is the coin still hot on the 15 m RSI when the red candle prints. The table below re-reads it on **frozen DESIGN terciles** (RSI15 ≥ 78.9; `leads.py` → `data/rsi15_regime.csv`). It does not hold up:

| | RSI15 < 70.2 | 70.2 – 78.9 | ≥ 78.9 |
|---|---|---|---|
| DESIGN (bull) | +0.33 | +1.75 | +1.00 |
| HOLDOUT bull | +1.10 | +0.17 | +0.13 |
| HOLDOUT bear | +2.20 | +1.37 | **−0.22** |

- It is negative in one block only and fails the null.
- The upper bound of its average is clearly above 0.
- One regime block counts as ONE observation.
- The live fills do not stamp a 15 m RSI.

**Not a filter, and too weak even for an observe line.**

## 6. What it means
- The losers are short squeezes: a fresh leg up after the red candle.
- Across about 43,000 tests of the 112 entry variables we have, none predicts which signals get squeezed better than chance.
- This is the third screen to say so (DIST_WL 11-18, DIST_FB_LOSERS 11-20, this one). All three use the same backtest cohort, so they are not independent evidence.
- The sleeve's edge is a **payoff shape**, not a high win rate:
  - about 42 % of trades win, averaging +8.8 % (TP fills +9.7 %);
  - about 58 % stops at about −4 to −6 %.
- Runs of 4 stops in 5 are normal for that shape. Five independent fills at a 57.5 % stop rate give 4 or more stops about 30 % of the time. DIST_FB_LOSERS §5.2 puts it at 17–30 %, depending on the pool.

## 7. Verdict
**No DIST entry filter.** Nothing passes the null, and no lead's blocked side is negative.

The levers that remain:
1. **Size.** 0.61 × equity is about 3–6× the Kelly range of 0.10–0.20 found in DIST_WL. This is already on the pre-live list.
2. **Squeeze data.** Order-book / tape data from the Route B recorder, at the weekly scan (≈ 2026-12-10). It is the first data source that could see a squeeze coming.

## 8. Blind spots
- Backtest-only: there are 5 real fills and no master block.
- Book depth, liquidation flow and the tape at entry are not in the backtest. This is the Route B gap.
- HOLDOUT has now been read for this cohort, so any later cut on it is in-sample.
- Missing values sit in "rest" for 8 variables: premium (37 % missing on DESIGN), `st_wr24h_closed` (34 %), mcap (31 %), spot share (11 %). The null has the same structure, so this cannot create a pass, but those cells are weaker reads.
- One HOLDOUT fill has no BTC daily regime value. The RSI15 table counts it as bear (218); the bear screen drops it (531).
- The nulls shuffle single trades. BTC and regime variables are constant within a market window, so their null is too lenient. This is anti-conservative and cannot turn a null verdict into a pass.

## 9. Follow-up: can an entry variable pick out the trades that need a LOWER take profit?
- **Method.** For each fill, a lower TP of X % (resting maker, filled when price trades through it) is priced from the fill's own 1 m reach before the stop.
  - If the reach is at least X + 0.05, the fill exits at X. Otherwise the Exit B result is unchanged.
  - Parity check: the 541 Exit B take-profit fills reproduce their booked net. The median difference is 0.000; the remainder is funding.
  - Hit-rule check: reach is measured before the stop on every fill. No stop or time exit reaches TP + 0.05, and every TP fill reaches TP − 0.01. A touch fill rule instead of trade-through moves the deltas by at most +0.07 (HOLDOUT 5 %: −0.20 → −0.13; 8 %: +0.01 → +0.06).
- **Cohorts.** 737 entry cohorts (the §3 bins with both sign halves) with frozen DESIGN bins, and X ∈ {3, 4, 5, 6, 8} %.
- **Score.** The mean gain per fill against Exit B.
- **Null.** 200 permutations of the per-fill gains, scored on the best cohort's mean.
- Script: `scripts/DIST_LOSERS2_2026-11-20_tpcell.py`. Data: `data/tp_cells.csv`.

| TP | whole cohort Δ DESIGN | whole cohort Δ HOLDOUT | null 95 % (best cohort, DESIGN) | cohorts passing | HOLDOUT Δ of the passer |
|---|---|---|---|---|---|
| 3 % | −0.85 | −0.36 | +1.09 | 1 (second-peak flag, n 56) | **−0.53** |
| 4 % | −0.71 | −0.31 | +1.04 | 1 (same) | **−0.27** |
| 5 % | −0.55 | −0.20 | +1.01 | 1 (same) | +0.02 |
| 6 % | −0.52 | −0.09 | +0.77 | 1 (same) | −0.08 |
| 8 % | −0.28 | +0.01 | +0.53 | 0 | — |

- Only one cohort passes on DESIGN: the coin printed a second peak (`g_P2R`).
- That cohort loses or stays flat on HOLDOUT at every TP.
- Thin coins (24 h vol < $33M, DESIGN quintile 1) gain on DESIGN but lose on HOLDOUT at 3–6 %. At 8 % they are flat (+0.01). This is consistent with DIST_THIN_TP (11-19).
- **Bias note.** A lower TP fills only when price trades 0.05 % through it. That is stricter than Exit B's own fills: only 91.9 % of Exit B TP fills reach tp + 0.05. The rule leans slightly against a lower TP, but not enough to close gaps of −0.2 … −0.85.

**No entry cohort earns a lower TP.** Keep Exit B (10 % TP) on every DIST fill.

## 10. Route B recorder: what it shows so far (first look, N = 2)
- **Coverage.** The recording starts 2026-11-19 at 23 UTC. Two DIST fills fall inside it: CYBER (win, +9.9 %) and KERNEL (loss, −6.1 %). One EXHAUSTION fill does too (KERNEL 14:15, win).
- Script: `scripts/DIST_LOSERS2_2026-11-20_routeb.py`. It reads 5-min buckets from entry −60 min to exit +30 min.

| at entry (±5 min) | CYBER (W) | KERNEL (L) |
|---|---|---|
| book imbalance within 0.5 % | −0.05 | −0.07 |
| net taker flow, 5 min | +$55k | −$84k |
| $ within 0.5 % (bid / ask) | $40k / $45k | $19k / $21k |
| short liquidations in the 20 min before | ≈ $55k | ≈ $62k |
| funding rate recorded at entry (mid-interval reading) | −1.53 % | −1.68 % (→ −2.00 % cap) |

- **Nothing at entry tells them apart.** Both coins were deeply crowded short. Imbalance and pre-entry short liquidations were similar. Taker flow at entry was small relative to volume (+$55k vs −$84k against ≈ $2.4–2.8M traded per 5 min). KERNEL's book was about half as deep.
- **KERNEL's squeeze came 35 min after entry**, in one leg: about +4 % → +10 % in 10 min. Volume doubled and short liquidations burst ($13k → $38k per 5 min). The stop hit inside that leg. There was no warning time between the first liquidation burst and the stop.
- **Funding.** Shorts pay when the rate is negative. Binance's settled rates (public funding history, settlements at 00 / 04 / 08 / 12 / 16 UTC, so a 4 h interval):

  | coin | 08:00 | 12:00 | 16:00 |
  |---|---|---|---|
  | CYBER | −0.12 % | −0.91 % | −0.89 % |
  | KERNEL | −0.03 % | −0.92 % | −2.00 % |

  - The recorder's readings at entry (−1.5 / −1.7 %) are mid-interval predictions, not settlements.
  - None of the three fills crossed 12:00 or 16:00, so the paper funding of $0 is correct.
  - KERNEL switched to **1 h funding** at about 16:01 UTC on 11-20. Any later KERNEL short pays hourly.
  - The backtest finds no funding effect on DIST (AUC 0.46–0.50). Here the winner was just as crowded.
- **Verdict:** N = 2, so this is descriptive only. The squeeze-build question (liquidation bursts, ask-side thinning, flow) is a weekly-scan item (≈ 2026-12-10) with the locked gates. A post-entry "short-liquidation burst" exit is a NEW mechanism (it is not one of the refuted lock / BE / trail exits), but only the recording can test it.
