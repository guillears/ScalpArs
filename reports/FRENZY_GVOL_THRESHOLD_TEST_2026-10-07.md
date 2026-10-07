# FRENZY market-volume gate: is 1.0 the right line? (2026-10-07)

Research only. No code, config or test was touched, and nothing was committed. This is an **unreviewed backtest**, so under the no-arm-before-review rule it supports no ship.

## Plain-English summary

**Recommendation: keep the line at 1.0. Don't raise it and don't lower it.**

- **What was asked.** Today the gate blocked four setups at market volume 2.92×, 1.95×, 1.34× and 1.11×, and two of them would have won. Would a higher line keep the protection and block fewer winners?
- **The answer on the year's data (588 FRENZY + WIDE signals, Jan–Sep):**
  - **1.0 gives the best result of all ten lines tested.** It compounds to +349 % for the year.
  - **1.1 is a tie** at +344 %.
  - Every other line is clearly worse: 1.2 gives +203 %, 1.3 +239 %, 1.5 +272 %, 2.0 +156 %, no gate +73 %. A lower line is worse too: 0.9 gives +104 %.
- **Above 1.0 the result gets worse step by step.** Average per trade by market-volume band:

  | market volume | avg per trade |
  |---|---|
  | 1.0–1.2 | −0.13 % |
  | 1.2–1.5 | +0.02 % |
  | 1.5–2.0 | −0.40 % |
  | ≥ 2.0 | −0.76 % |

  The worst setups are the very-high-volume ones. Even so, the band just above 1.0 is not good enough to let back in, so raising the line gives back profit.
- **Checked out of sample. No line beats 1.0 when tested on data it was not chosen on:**
  - Choosing the best line on Jan–May picks 1.0 itself.
  - Choosing on Jun–Sep picks 2.0, and 2.0 then loses 208 points of book on Jan–May.
  - In the leave-one-month-out test, no line beats 1.0 in more than 4 of 9 months. 1.1 is the only one that ties sometimes.
- **The honest weak spot.** In Jun–Sep the gate barely helps at any line: 1.0 gives +12 %, 2.0 gives +32 %, no gate gives +22 %.
  - This is the same summer fade already known from the 10-06 revalidation.
  - It is a reason to keep watching the gate, but not a reason to move the line: the Jun–Sep pick failed on the other half.
- **One thing worth watching: WIDE alone.**
  - For WIDE, 1.1 looks better than 1.0 (book +113 % vs +61 %). Its [1.0, 1.1) band won +1.50 % per trade over 21 signals.
  - That band is small and comes from slicing the data by sleeve. It does not survive the luck test (p 0.46), and the out-of-sample check is mixed (+27 pp one way, −10 pp the other).
  - So it is a watch item, not a change. The frozen observe band in §6 covers [1.0, 1.2), so a SAND-type 1.11 fill would count.
- **Today's four blocks are one day, so one observation.** A 1.1 line would still have blocked SAND (1.11). A 1.2 line would have let in only SAND (+4.7, provisional). A 1.5 line adds NMR at 1.34 (−3.1). A 2.0 line adds ORCA at 1.95 (−3.1). Netted, today does not argue for any line.
- **Proposed instead (free, observe-only):** split the existing GVOL_BLOCKED scout tracker by frozen bands, so live data can tell us later whether the band just above 1.0 is really losing. The live revert gate (11/20, −1.33 %) and the GVOL_BLOCKED tracker keep running unchanged.

## 1. Pre-registration

Written to `scratchpad/gvol_threshold/PREREG.txt` at **2026-10-07 21:14:43 UTC**, before any threshold outcome was computed:

- **Lines tested:** {0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, off}. The rule blocks a signal when V1 ≥ line. The primary comparison is each line against the live 1.0. FRENZY and WIDE are also reported separately.
- **Cohort:** the 10-06 engine-bar cohort (`gvg/step2.E`). It has 588 eligible signals: 352 FRENZY_LONG READY and 236 WIDE hold-green.
  - V1 is the engine market-volume ratio (`gvolformula/yr_versions.pkl`). Parity with the engine U2 is exact (max diff 0.0), and ORCA 10-06 18:05 = 0.7802.
  - The cohort covers 222 days, 465 4-h windows and 585 distinct signal bars, from 2026-01 to 2026-09.
- **Pricing (same as the revalidation):**
  - the live lock (arm +3, floor +2, trail 2 pts, stop −3) on real ticks;
  - entry at the first print ≥ signal close + 12 s, the slow end of the live 8–12 s;
  - 0.09 % fees plus 0.10 % exit slip;
  - 12 h cap.
- **Book:** compounded with live sizing (FRENZY 0.32/0.5 lev, WIDE 0.2), 2 slots per sleeve, one position per pair and a cap of 3 per pair per day. This uses `step2.seq/path` unchanged.
- **Units:** this is a market-wide variable, so confidence intervals bootstrap whole days and the expectancy bar uses 4-h-window clusters.
- **Selection checks:** choose the best line on one half and confirm it on the other, plus a within-month shuffled-gvol null with 2,000 draws.
- **Halves:** Jan–May against Jun–Sep. The data ends in September, so "Jun–Oct" in the brief becomes Jun–Sep.

## 2. Dose-response (by market-volume bucket)

| market volume | N | days | WR | avg %/trade | day CI | Σ % |
|---|---|---|---|---|---|---|
| < 0.6 | 111 | 85 | 53 % | +0.41 | [−0.29, +1.08] | +45 |
| 0.6–0.8 | 103 | 78 | 54 % | +0.15 | [−0.47, +0.77] | +16 |
| 0.8–1.0 | 116 | 92 | 59 % | **+0.65** | [+0.02, +1.29] | +75 |
| 1.0–1.2 | 86 | 75 | 48 % | −0.13 | [−0.79, +0.53] | −11 |
| 1.2–1.5 | 60 | 53 | 48 % | +0.02 | [−0.88, +0.94] | +1 |
| 1.5–2.0 | 66 | 53 | 45 % | −0.40 | [−1.16, +0.39] | −26 |
| ≥ 2.0 | 46 | 43 | 41 % | **−0.76** | [−1.58, +0.04] | −35 |

| bucket | FRENZY avg (N) | WIDE avg (N) |
|---|---|---|
| < 0.6 | +0.13 (68) | +0.85 (43) |
| 0.6–0.8 | +0.21 (70) | +0.02 (33) |
| 0.8–1.0 | +0.83 (67) | +0.40 (49) |
| 1.0–1.2 | **−0.57** (49) | **+0.45** (37) |
| 1.2–1.5 | +0.27 (30) | −0.23 (30) |
| 1.5–2.0 | −0.47 (41) | −0.29 (25) |
| ≥ 2.0 | −0.31 (27) | **−1.40** (19) |

**Shape.**
- **Below and above 1.0.** Positive below 1.0, negative above, and worst at ≥ 1.5. Win rate falls steadily from 59 % to 41 % from the 0.8–1.0 bucket upward.
- **Not monotonic.** 0.6–0.8 is weak, 0.8–1.0 is the best bucket, and 1.2–1.5 is flat.
- **Weak at day level.** Day-level Spearman correlation is only −0.07 (permutation p 0.28). Per signal it is −0.10.
- **Reading.** This is a step around 1.0 plus a bad tail at ≥ 1.5. It is not a clean dose-response. Each bucket's CI is wide (≈ ±0.7).
- **The sharp step right at the live line is a caution flag.** 1.0 came from earlier work on overlapping data (DECISION_LOG 194), so part of "1.0 is best" can be in-sample. The out-of-sample checks in §4 are the defence against that.
- **The two sleeves disagree on 1.0–1.2.** FRENZY is −0.57 there and WIDE is +0.45.

## 3. Per line, versus the live 1.0 (both sleeves)

| line | kept N | kept WR | kept avg | blocked N | blocked avg | book % | max DD | Δ book vs 1.0 | Jan–May book | Jun–Sep book | LOMO months beating 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.8 | 214 | 54 % | +0.29 | 374 | +0.01 | +58 % | −42 % | −291 | +104 % | −23 % | 0/9 |
| 0.9 | 273 | 55 % | +0.30 | 315 | −0.05 | +104 % | −41 % | −245 | +146 % | −17 % | 0/9 |
| **1.0** | 330 | 56 % | **+0.41** | 258 | −0.28 | **+349 %** | −54 % | — | +302 % | +12 % | — |
| 1.1 | 375 | 55 % | +0.39 | 213 | −0.39 | +344 % | −54 % | −5 | +299 % | +11 % | 4/9 |
| 1.2 | 416 | 54 % | +0.30 | 172 | −0.35 | +203 % | −64 % | −147 | +260 % | −16 % | 0/9 |
| 1.3 | 445 | 54 % | +0.29 | 143 | −0.44 | +239 % | −64 % | −110 | +242 % | −1 % | 0/9 |
| 1.5 | 476 | 53 % | +0.27 | 112 | −0.55 | +272 % | −64 % | −77 | +257 % | +4 % | 0/9 |
| 1.75 | 525 | 53 % | +0.23 | 63 | −0.86 | +213 % | −61 % | −136 | +140 % | +30 % | 0/9 |
| 2.0 | 542 | 52 % | +0.19 | 46 | −0.76 | +156 % | −64 % | −193 | +95 % | +32 % | 0/9 |
| off | 588 | 52 % | +0.11 | 0 | — | +73 % | −63 % | −276 | +42 % | +22 % | 0/9 |

Per sleeve (book %, lines 1.0 / 1.1 / 1.3 / 1.5 / 2.0 / off):
- **FRENZY:** 225 / 143 / 132 / 156 / 91 / 61. 1.0 is best by a wide margin.
- **WIDE:** 61 / **113** / 71 / 70 / 56 / 24. 1.1 is best, and it is better in both halves (Jan–May 72 vs 45, Jun–Sep 24 vs 11).

### Marginal bands: the signals that change side when the line moves

The expectancy bar calls a band LOSING (so it stays blocked on evidence) when all of these hold:
- WR is below breakeven (≈ 49.8 %);
- P(mean < 0) ≥ 0.95 by window bootstrap;
- it spans ≥ 8 windows;
- no window or pair carries ≥ 50 % of the loss;
- N ≥ 15.

| band | group | N | days | WR | avg | day CI | P(mean<0) | losing at 95 %? |
|---|---|---|---|---|---|---|---|---|
| [0.9, 1.0) | both | 57 | 52 | 61 % | **+0.97** | [+0.10, +1.91] | 0.03 | no (a winner) |
| [1.0, 1.1) | both | 45 | 41 | 51 % | +0.24 | [−0.76, +1.27] | 0.33 | no |
| [1.0, 1.1) | FRENZY | 24 | 22 | 42 % | **−0.86** | [−1.96, +0.34] | 0.94 | no (just misses) |
| [1.0, 1.1) | WIDE | 21 | 21 | 62 % | **+1.50** | [−0.09, +3.10] | 0.04 | no (looks like a winner) |
| [1.0, 1.2) | both | 86 | 75 | 48 % | −0.13 | [−0.77, +0.54] | 0.64 | no |
| [1.0, 1.3) | both | 115 | 88 | 48 % | −0.08 | [−0.62, +0.49] | 0.61 | no |
| [1.0, 1.5) | both | 146 | 106 | 48 % | −0.07 | [−0.61, +0.45] | 0.59 | no |
| [1.0, 2.0) | both | 212 | 138 | 47 % | −0.17 | [−0.60, +0.25] | 0.77 | no |
| [1.0, ∞) (whole blocked side) | both | 258 | 152 | 46 % | −0.28 | [−0.67, +0.11] | 0.90 | no (known; DECISION_LOG 194 override) |

What the bands show:
- **No band above 1.0 passes the "losing" bar.** The bands up to 1.5 are about break-even (−0.07 to −0.13 per trade). That is "not clearly losing", but also not worth re-admitting.
- **Why raising the line still costs book.** The kept side is +0.41 per trade. Re-admitting a ≈ 0 band dilutes it, and with compounding and fixed slots that costs book: −77 to −147 pp for 1.2–1.5.
- **[1.0, 1.1) stays positive by month only partly.** It is positive in 4 of 9 months, and leave-one-month-out ranges from +0.00 to +0.50.
- **The FRENZY half of that band is negative in 6 of 8 months.** The WIDE half is positive in 6 of 7 months: 01 +0.98 (7), 03 +1.89 (3), 05 +1.31 (2), 06 +5.35 (2), 08 +0.77 (4), 09 −0.61 (2). Its top 3 winners carry 33 % of the gross wins.

### By period (both sleeves, band avg (N))

| period | [0.8, 0.9) | [0.9, 1.0) | [1.0, 1.1) | [1.1, 1.2) | [1.2, 1.3) | [1.3, 1.5) | [1.5, 2.0) | ≥ 2.0 |
|---|---|---|---|---|---|---|---|---|
| Jan–May | +0.21 (41) | +0.86 (36) | +0.15 (31) | +0.08 (24) | −0.29 (20) | −0.10 (21) | **−1.06** (46) | −0.76 (37) |
| Jun–Sep | +0.63 (18) | +1.16 (21) | +0.44 (14) | −1.41 (17) | +0.90 (9) | +0.10 (10) | **+1.11** (20) | −0.74 (9) |

- **The summer flip sits in the 1.5–2.0 band:** −1.06 in Jan–May against +1.11 in Jun–Sep. That is why Jun–Sep prefers a 2.0 line.
- **≥ 2.0 is bad in both halves.**

## 4. Selection, null and haircut

| procedure | result |
|---|---|
| Choose on Jan–May → test on Jun–Sep | chooses **1.0** → Δ 0 |
| Choose on Jun–Sep → test on Jan–May | chooses **2.0** (+20 pp in-sample) → **−208 pp** on Jan–May, kept avg −0.35 |
| Full-sample best line | **1.0** (gain over 1.0 = 0) |
| Within-month shuffled-gvol null, 2,000× | random gvol gives a "best line" that beats its own 1.0 by a median log-gain of +0.58 (95th +1.38). The observed gain is 0, so p = 1.00: **nothing to haircut** |
| FRENZY alone | best = 1.0; the Jun–Sep pick (1.75) loses 132 pp on Jan–May |
| WIDE alone | best = 1.1 (+52 pp in-sample; +26 to +36 pp after the 30–50 % haircut). OOS mixed: Jan–May picks 1.2 → −10 pp on Jun–Sep; Jun–Sep picks 1.1 → +27 pp on Jan–May. Null p = 0.46 |

**Verdict under the pre-registered rule:**
- **Keep 1.0.** No line beats it out of sample in both directions, and none passes the null.
- **The band just above 1.0 is "not clearly losing", but also not clearly winning** (≈ −0.1 per trade up to 1.5). So the evidence does not support moving the line up.
- **WIDE at 1.1 is the only hint.** It fails the null and one OOS direction, and it is a per-sleeve slice of 21 signals.

## 5. Today (2026-10-07) under each line, as an anecdote (1 day = 1 observation)

| block | gvol | outcome | re-admitted at line |
|---|---|---|---|
| SAND WIDE | 1.11 | +4.7 (provisional) | ≥ 1.2 (1.1 still blocks: 1.11 ≥ 1.1) |
| NMR | 1.34 | −3.1 | ≥ 1.5 |
| ORCA | 1.95 | −3.1 | ≥ 2.0 |
| NMR | 2.92 | +1.9 | off only |

| line | re-admitted today | net |
|---|---|---|
| 1.2 or 1.3 | SAND | +4.7 |
| 1.5 | SAND and NMR | +1.6 |
| 2.0 | SAND, NMR and ORCA | −1.5 |

The year table already says 1.2–1.3 cost about 110–150 pp.

## 6. Proposed observe line (free, scout-only, nothing armed)

Split the existing `GVOL_BLOCKED` scout tracker by **frozen** bands: [1.0, 1.1), [1.1, 1.2), [1.2, 1.5), [1.5, 2.0), ≥ 2.0. Report them per sleeve (FRENZY / WIDE / LITE as stamped), on the same ruler the tracker already uses.

Pre-registered hypothesis, frozen now: **"WIDE hold-green in [1.0, 1.2) is not a losing cohort."**
- **Decision read** at ≥ 20 fresh WIDE band fills spanning ≥ 10 days.
- **To propose moving WIDE alone to 1.2,** the band needs all three:
  - mean > 0 with day-block P(mean > 0) ≥ 0.90;
  - no day carrying ≥ 50 % of the gain;
  - the band mean ≥ the WIDE taken side's mean minus 0.3 in the same period.
- **Otherwise WIDE stays at 1.0.**

The thresholds are not to be re-fitted on the data that fails them. FRENZY has no such hypothesis, because its [1.0, 1.1) band is negative.

The live revert gate (11/20, −1.33 %) and the GVOL_BLOCKED same-ruler tracker continue as they are.

## Files

- `scratchpad/gvol_threshold/PREREG.txt` (timestamped pre-registration)
- `scratchpad/gvol_threshold/load.py` and `run.py` (analysis)
- `scratchpad/gvol_threshold/run_out.md` (full tables, including the per-sleeve threshold tables, all marginal bands per sleeve, per-month bands and LOMO)
- Reused: `scratchpad/gvg/step2.py` (cohort, LOCK2 pricing, book) and `scratchpad/gvolformula/yr_versions.pkl` (V1)
