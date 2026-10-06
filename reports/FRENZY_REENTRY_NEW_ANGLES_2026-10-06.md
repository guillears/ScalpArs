# 🔁 FRENZY re-entry: four new angles + delayed entry (catch-up limits), 2026-10-06

Research only. No bot file, config, test, template or `scripts/` file was touched, and nothing was committed. **Unreviewed**: no caveman or deep review has run on these numbers yet, so no arm recommendation is made from them (feedback_no_arm_before_review).
Scripts and raw output are in the scratchpad `…/scratchpad/rna/`:
- `PREREG.txt`: every variant and threshold, frozen at 14:24 UTC before any outcome was computed. Angle E was added at 14:24, also before outcomes.
- `price.py`: the book-independent pricing.
- `analyze.py`: books, statistics and nulls.
- `e_tab.py` and `c_wide.py`: the E and C detail.
- `live/`: NMR and RLC on public klines.

## 0. Engine parity (read first)

| check | result |
|---|---|
| Signal bars | `FRENZY_ENGINE_COHORT_2026-10-05.csv` = the real `services.frenzy` walk. Tradeable universe (`live_elig`, `gvol_live_U2 < 1`) under today's rules (all FRENZY; WIDE only on hold-green, i.e. a green bar with streak > 12) gives **330 signals, all priced**. |
| Base lock re-price | My LOCK2 vs the cohort's LOCK2: max \|Δ\| = 1.8e-15 on 330 / 330 |
| Base book | **328 fills (FRENZY 204, WIDE 124), $3k → $13,474, max DD −54 %, +0.39 %/fill [CI +0.01, +0.76]**. The FRENZY-only book reproduces the WIDE review exactly: 205 fills, $9,741, DD −43 %. |
| "Setup ON" for re-entries | In-state bars of the real `frenzy_walk` at live min-hours 2 (`mh/bars_mh.pkl`), same `spike_ts` as the anchor, not a fresh bar, gvr < 1, vol24 ≥ $20M |
| Re-entry rule | **New logic, no engine path.** Parity holds by specification only. |
| "Volume ≥ 100×" variant | **Degenerate.** Every in-state bar already has rolling-hour volume ≥ 100× normal (min 100.0×), because the walk only runs there. I replaced it with V300 (vol_mult ≥ 300, the audit's frozen (c) threshold). |
| Live cap semantics | `trading_engine.py:7553`: the cap is 3 entries per pair per UTC day **per strategy tag** (FRENZY_LONG and FRENZY_WIDE counted separately), open or closed. It was modelled exactly. |

## 1. Answer in plain language

1. **Why is there a limit, and what is it?**
   - There are two limits. `frenzy_max_entries_per_pair_day = 3` (DECISION_LOG 177) is the visible one, and **it never binds** (§5). On the tradeable year, 255 of the 291 pair-sleeve-days had a single signal, 33 had two and 3 had three. Setting the cap to 3, 5 or unlimited gives the identical book.
   - **The real limit is "one entry per fresh bar".** A new entry needs the setup to go OFF for at least 1 h and turn back ON. On NMR and RLC that never happened, so the cap was irrelevant.
   - Lowering the cap would hurt: cap 2 gives $10.1k and cap 1 gives $9.8k, against $13.5k at cap 3.
2. **A. Breakout re-entry (buy when price makes a new high above the winner's pre-exit peak) is refuted, and clearly so.**
   - All 24 variants are negative, from −0.14 to −0.90 %/fill. 10 of 24 have the whole day-CI below 0.
   - Every variant loses money in both halves, and every one shrinks the book (−7 % to −66 % vs base).
   - The breakout bar is **worse than a random ON bar** of the same window in all 24 variants (raw timing p 0.50–0.98).
   - **Why it fails:** about 70 % of winners do print a new high within 24 h, and a third of those re-entries see +15 % within 12 h. But half of them are stopped at −3 first. A breakout above a fresh spike top is the classic exhaustion print.
3. **B. Pullback-reclaim is refuted.**
   - All 4 variants are negative (−0.29 to −0.66). The best, EMA20 K2, is −0.29 [−0.78, +0.18] and shrinks the book 48 %.
   - Its timing beats a random ON bar (raw p 0.04), but not after selection (p 0.64). It is "less bad than random", not good.
4. **C. The dislocation pocket is not a pocket under the live exit.**
   - Under today's rules in the tradeable universe there are only **8 refusals in 9 months**. Entering anyway (cap 2 / 3 / 5 %) gives +0.01 / −0.14 / −0.14 %/fill and a book 2 % lower.
   - Across all 147 dislocation refusals in the cohort (any rule or universe): +0.14 %/fill [−0.32, +0.61], with a median 12 h peak of +13.7 %.
   - The "+33 % median in 48 h" in the staircase study measures how far the episode ran. It is not what the lock captures. The lock banks these like any other fill (≈ 0).
5. **D. The pair-day cap does not bind** (see point 1). It also refuses only 4–8 re-entries in the A / B books.
6. **E. Delayed entry (the catch-up limits): lateness itself costs money, even when the price has not moved.**
   - **Same signals, entered late vs on time:** Δ is −0.46 (k = 1), −0.77 (k = 3), −0.90 (k = 6) and −1.9 (k = 12), all with the day-CI below 0.
   - **Within "price moved ≤ 1 %"** the cost remains: −0.40 [−0.69, −0.10] at k = 1, −0.86 at k = 3, −0.61 at k = 6.
   - **Absolute value of a catch-up fill (vs doing nothing) at ≤ 1 %:**
     - k = 1: **+0.46** [−0.10, +1.03], positive in both halves.
     - k = 3: −0.13.
     - k = 6: −0.00.
     - Looser caps are worse: at 1–3 % and k = 1 the mean is −0.49.
   - **The data supports a tighter catch-up, not a looser one:** k ≤ 1–2 bars (≤ 10 min) at ≤ 1 %. The planned k ≤ 6 at ≤ 1 % is roughly break-even beyond the first bar. It is harmless, not helpful.
   - **The in-state check is right.** Signals whose setup had already turned OFF were on-time losers (−0.55 to −0.89).
7. **Is re-entry a hindsight trap? Yes.** This is the 7th to 10th refutation of the re-entry family, with 43 pre-registered variants here and **0 passing**. The best selection-adjusted p in the whole family is 0.65.
   - NMR and RLC look like re-entry cases only after the fact. On the year, the same entries (the breakout above the previous top, the reclaim after the pullback) are stopped half the time before any run.
   - What the year does reward is **being on time to the fresh bar**. That supports the catch-up, kept short.

## 2. A / B: re-entry after a lock-exit winner (added fills actually taken in the sequenced book)

Base book: 328 fills, $13,474, max DD −54 %.
- Columns are as pre-registered.
- **max-t p** is the single-step max-t over all 43 variants (A 24, B 4, C 3, E 12), with a joint day-block bootstrap. It is the same statistic on both sides.
- **The timing null** replaces each taken re-entry with a random eligible ON bar from its own window. Both sides use the 5m-bar lock pricer, with 500 draws, maximised over the 28 A + B variants.

### A
| variant | N added | days | WR | mean %/fill | day 95 % CI | Jan–Apr / May–Sep | LOMO min … max | drop top 5 / 10 | book end (vs base) | max DD | base fills displaced | max-t p (43) | timing-null p (A+B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_m0.5_W24_V100_K1 | 97 | 80 | 41 % | **-0.87** | [-1.42, -0.32] | -0.56 / -1.17 | -0.98 … -0.75 | -1.14 / -1.37 | $5,349 (-60 %) | -70 % | 0 | 1.00 | 1.00 |
| A_m0.5_W24_V100_K2 | 126 | 84 | 44 % | **-0.65** | [-1.15, -0.15] | -0.48 / -0.82 | -0.75 … -0.59 | -0.91 / -1.09 | $5,155 (-62 %) | -71 % | 1 | 1.00 | 1.00 |
| A_m0.5_W24_V300_K1 | 56 | 49 | 50 % | **-0.36** | [-1.11, +0.42] | -0.31 / -0.42 | -0.51 … -0.20 | -0.83 / -1.18 | $10,714 (-20 %) | -55 % | 0 | 1.00 | 1.00 |
| A_m0.5_W24_V300_K2 | 73 | 52 | 49 % | **-0.37** | [-1.05, +0.30] | -0.43 / -0.31 | -0.61 … -0.16 | -0.74 / -1.04 | $10,259 (-24 %) | -54 % | 1 | 1.00 | 1.00 |
| A_m0.5_W6_V100_K1 | 82 | 68 | 40 % | **-0.90** | [-1.51, -0.29] | -0.49 / -1.31 | -1.05 … -0.73 | -1.23 / -1.51 | $5,942 (-56 %) | -67 % | 0 | 1.00 | 1.00 |
| A_m0.5_W6_V100_K2 | 106 | 73 | 43 % | **-0.69** | [-1.23, -0.16] | -0.46 / -0.93 | -0.81 … -0.55 | -1.00 / -1.21 | $5,757 (-57 %) | -66 % | 0 | 1.00 | 1.00 |
| A_m0.5_W6_V300_K1 | 45 | 39 | 44 % | **-0.59** | [-1.46, +0.35] | -0.49 / -0.71 | -0.71 … -0.42 | -1.21 / -1.71 | $10,366 (-23 %) | -55 % | 1 | 1.00 | 1.00 |
| A_m0.5_W6_V300_K2 | 55 | 41 | 44 % | **-0.64** | [-1.43, +0.13] | -0.76 / -0.47 | -0.75 … -0.45 | -1.15 / -1.58 | $9,309 (-31 %) | -56 % | 1 | 1.00 | 1.00 |
| A_m0_W24_V100_K1 | 100 | 82 | 44 % | **-0.72** | [-1.30, -0.14] | -0.33 / -1.10 | -0.86 … -0.59 | -0.96 / -1.18 | $6,066 (-55 %) | -70 % | 0 | 1.00 | 1.00 |
| A_m0_W24_V100_K2 | 134 | 86 | 44 % | **-0.69** | [-1.21, -0.21] | -0.50 / -0.87 | -0.76 … -0.60 | -0.91 / -1.08 | $4,549 (-66 %) | -73 % | 1 | 1.00 | 1.00 |
| A_m0_W24_V300_K1 | 56 | 49 | 50 % | **-0.36** | [-1.13, +0.43] | -0.31 / -0.42 | -0.51 … -0.20 | -0.83 / -1.18 | $10,713 (-20 %) | -55 % | 0 | 1.00 | 1.00 |
| A_m0_W24_V300_K2 | 74 | 52 | 47 % | **-0.54** | [-1.17, +0.08] | -0.59 / -0.49 | -0.71 … -0.36 | -0.90 / -1.17 | $9,099 (-32 %) | -60 % | 1 | 1.00 | 1.00 |
| A_m0_W6_V100_K1 | 87 | 71 | 44 % | **-0.72** | [-1.35, -0.09] | -0.23 / -1.17 | -0.88 … -0.53 | -0.99 / -1.26 | $6,633 (-51 %) | -68 % | 0 | 1.00 | 1.00 |
| A_m0_W6_V100_K2 | 115 | 76 | 44 % | **-0.64** | [-1.22, -0.12] | -0.45 / -0.84 | -0.74 … -0.51 | -0.90 / -1.10 | $5,629 (-58 %) | -68 % | 0 | 1.00 | 1.00 |
| A_m0_W6_V300_K1 | 45 | 39 | 44 % | **-0.59** | [-1.50, +0.33] | -0.49 / -0.71 | -0.71 … -0.42 | -1.21 / -1.71 | $10,365 (-23 %) | -55 % | 1 | 1.00 | 1.00 |
| A_m0_W6_V300_K2 | 57 | 41 | 42 % | **-0.74** | [-1.48, +0.01] | -0.83 / -0.63 | -0.80 … -0.58 | -1.24 / -1.65 | $8,737 (-35 %) | -60 % | 1 | 1.00 | 1.00 |
| A_m1_W24_V100_K1 | 93 | 77 | 41 % | **-0.79** | [-1.38, -0.17] | -0.68 / -0.91 | -0.93 … -0.65 | -1.15 / -1.41 | $6,126 (-55 %) | -70 % | 0 | 1.00 | 1.00 |
| A_m1_W24_V100_K2 | 121 | 81 | 47 % | **-0.43** | [-1.00, +0.13] | -0.57 / -0.28 | -0.58 … -0.29 | -0.73 / -0.93 | $7,557 (-44 %) | -64 % | 0 | 1.00 | 1.00 |
| A_m1_W24_V300_K1 | 53 | 46 | 53 % | **-0.14** | [-0.93, +0.67] | -0.08 / -0.21 | -0.27 … +0.07 | -0.67 / -1.04 | $12,473 (-7 %) | -56 % | 0 | 1.00 | 1.00 |
| A_m1_W24_V300_K2 | 71 | 49 | 52 % | **-0.18** | [-0.90, +0.52] | -0.22 / -0.13 | -0.39 … +0.07 | -0.56 / -0.88 | $12,217 (-9 %) | -55 % | 1 | 1.00 | 1.00 |
| A_m1_W6_V100_K1 | 77 | 64 | 40 % | **-0.79** | [-1.41, -0.15] | -0.57 / -1.00 | -0.94 … -0.60 | -1.22 / -1.54 | $7,010 (-48 %) | -67 % | 0 | 1.00 | 1.00 |
| A_m1_W6_V100_K2 | 99 | 69 | 46 % | **-0.42** | [-1.04, +0.18] | -0.52 / -0.32 | -0.63 … -0.31 | -0.80 / -1.05 | $8,341 (-38 %) | -60 % | 0 | 1.00 | 1.00 |
| A_m1_W6_V300_K1 | 41 | 35 | 46 % | **-0.39** | [-1.36, +0.62] | -0.23 / -0.59 | -0.55 … -0.15 | -1.11 / -1.69 | $11,856 (-12 %) | -56 % | 1 | 1.00 | 1.00 |
| A_m1_W6_V300_K2 | 52 | 37 | 46 % | **-0.43** | [-1.24, +0.38] | -0.49 / -0.34 | -0.52 … -0.19 | -0.99 / -1.46 | $10,891 (-19 %) | -57 % | 1 | 1.00 | 1.00 |

### B
| variant | N added | days | WR | mean %/fill | day 95 % CI | Jan–Apr / May–Sep | LOMO min … max | drop top 5 / 10 | book end (vs base) | max DD | base fills displaced | max-t p (43) | timing-null p (A+B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_E20_K1 | 130 | 100 | 46 % | **-0.48** | [-1.02, +0.07] | -0.49 / -0.48 | -0.66 … -0.30 | -0.74 / -0.93 | $6,721 (-50 %) | -63 % | 0 | 1.00 | 0.74 |
| B_E20_K2 | 193 | 105 | 49 % | **-0.29** | [-0.78, +0.18] | -0.16 / -0.42 | -0.37 … -0.18 | -0.48 / -0.61 | $6,990 (-48 %) | -61 % | 4 | 1.00 | 0.64 |
| B_XP_K1 | 109 | 86 | 44 % | **-0.66** | [-1.22, -0.08] | -0.56 / -0.77 | -0.81 … -0.50 | -0.92 / -1.13 | $5,943 (-56 %) | -70 % | 0 | 1.00 | 1.00 |
| B_XP_K2 | 153 | 90 | 44 % | **-0.64** | [-1.12, -0.18] | -0.43 / -0.87 | -0.78 … -0.58 | -0.82 / -0.98 | $4,256 (-68 %) | -75 % | 3 | 1.00 | 1.00 |


**Timing (A + B, 5m pricer on both sides): breakout bar vs random ON bar in the same window**
- **A:** the observed mean is below the null in **24 / 24 variants**. For example, A m0 W24 V100 K1 is −0.56 vs −0.32, and A m1 W6 V100 K1 is −0.66 vs +0.01. Raw p ranges from 0.50 to 0.98, and the adjusted p is 1.00 for all.
- **B:**
  - E20 K1: −0.18 vs −0.62 (raw p 0.05, adjusted 0.74).
  - E20 K2: −0.21 vs −0.58 (raw p 0.04, adjusted 0.64).
  - XP: equal to the null or worse.

**Before sequencing, the triggers look less bad, and here is why.** Of the 184 winning anchors:
- **74 % get an A (m0 W24) trigger.** These trades average −0.17 unsequenced, with a median 12 h peak of +7.9, 32 % reaching ≥ +15 and 49 % stopped at −3.
- **The 36 triggers the book drops are the winners (+0.7 to +1.9).** They collide with a *new fresh base entry* on the same pair (18), a second anchor in the same episode (10), or the pair-day cap (5). Those runs are already traded by the base sleeve.
- **What is genuinely incremental is −0.72 %/fill.**

**Per month (mean %/fill, N):**
- A m0 W24 V100 K1: Jan −0.45 (13) · Feb −0.80 (7) · Mar −0.31 (19) · Apr +0.13 (10) · May −0.97 (13) · Jun −1.86 (8) · Jul −1.51 (11) · Aug +0.94 (8) · Sep −1.75 (11). Positive in 2 of 9 months.
- A m1 W24 V300 K1 (the least bad): Jan −1.53 · Feb +1.47 (2) · Mar +0.09 · Apr +0.65 · May −0.82 · Jun +0.64 · Jul −0.70 · Aug +0.89 · Sep −0.61. Positive in 5 of 9, but its CI is [−0.93, +0.67] and both halves are below 0.
- B E20 K2: positive in 3 of 9 months (Jan / Apr / May, each about +0.1–0.2).

**OOS (pick on one half, read on the other):**
- **A:** the best Jan–Apr variant (A m1 W24 V300 K1) is −0.08, and it reads **−0.21** on May–Sep. The best May–Sep variant is −0.13 and reads −0.22.
- **B:** the best is E20 K2 in both directions: −0.16 then −0.42.

## 3. C: dislocation pocket (refused by the 1 % guard, entered anyway at the 12 s print, live lock)

### C
| variant | N added | days | WR | mean %/fill | day 95 % CI | Jan–Apr / May–Sep | LOMO min … max | drop top 5 / 10 | book end (vs base) | max DD | base fills displaced | max-t p (43) | timing-null p (A+B) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C_cap2 | 6 | 6 | 50 % | **+0.01** | [-2.26, +2.51] | -1.48 / +3.00 | -0.78 … +0.64 | -3.13 / – | $13,259 (-2 %) | -51 % | 0 | 1.00 | – |
| C_cap3 | 8 | 8 | 50 % | **-0.14** | [-2.30, +1.95] | -1.48 / +1.19 | -1.17 … +0.28 | -3.12 / – | $13,166 (-2 %) | -50 % | 0 | 1.00 | – |
| C_cap5 | 8 | 8 | 50 % | **-0.14** | [-2.30, +2.01] | -1.48 / +1.19 | -1.17 … +0.28 | -3.12 / – | $13,166 (-2 %) | -50 % | 0 | 1.00 | – |


**All 147 dislocation refusals in the engine cohort (descriptive, outside today's tradeable rules):**

| subset | N | days | WR | mean %/fill (lock) | day CI | median 12 h peak | 12 h peak ≥ +15 |
|---|---|---|---|---|---|---|---|
| all refused | 147 | 102 | 56 % | +0.14 | [−0.32, +0.61] | +13.7 % | 46 % |
| today's rules | 29 | 27 | 55 % | +0.02 | [−0.98, +0.99] | +12.8 % | 34 % |
| today's rules ∧ published gate | 14 | 13 | 50 % | −0.34 | [−1.72, +1.14] | +14.4 % | 43 % |
| price ran up (dl > 0) | 85 | 65 | 55 % | +0.12 | [−0.54, +0.81] | +13.7 % | 47 % |
| price fell (dl < 0) | 62 | 52 | 56 % | +0.17 | [−0.59, +0.91] | +12.8 % | 44 % |
| \|dl\| ≤ 2 / ≤ 3 / ≤ 5 % | 111 / 133 / 143 | | | +0.01 / +0.07 / +0.20 | all span 0 | | |

**Reading the table:**
- 135 of the 147 refusals are WIDE signals that today's hold-green rule refuses anyway.
- **The big runs are real**: 46 % reach a 12 h peak of +15 %. But under the lock, a dislocated entry banks what any fill banks.
- **Slippage-adjusted** (fees 0.09 + slip 0.10, entry at the actual 12 s print): no cap level is positive with a CI above 0.

## 4. E: delayed entry of a valid signal (catch-up after pause or restart)

**Per k:**
- Setup still ON at bar fresh+k is required.
- The entry is the first print ≥ that bar's close + 12 s, with no guard.
- Δ is paired against the on-time entry of the SAME signals, with a day-block CI.

| k (bars / min) | signals still ON | late mean | day CI | on-time mean, same signals | **Δ late − on time** | Δ CI | \|move\| > 1 / 2 / 3 / 5 % | signals already OFF: their on-time mean |
|---|---|---|---|---|---|---|---|---|
| 1 / 5 | 297 / 330 | +0.06 | [−0.37, +0.49] | +0.52 | **−0.46** | [−0.75, −0.17] | 42 / 18 / 7 / 1 % | −0.55 (33) |
| 3 / 15 | 270 | −0.09 | [−0.50, +0.33] | +0.68 | **−0.77** | [−1.19, −0.33] | 65 / 42 / 22 / 8 % | −0.77 (60) |
| 6 / 30 | 236 | −0.00 | [−0.40, +0.40] | +0.90 | **−0.90** | [−1.39, −0.41] | 75 / 52 / 38 / 17 % | −0.80 (94) |
| 12 / 60 | 178 | −0.39 | [−0.90, +0.12] | +1.53 | **−1.91** | [−2.67, −1.16] | 83 / 70 / 53 / 33 % | −0.89 (152) |
| 24 / 120 | 131 | +0.43 | [−0.12, +1.05] | +2.26 | **−1.82** | [−2.56, −1.11] | 85 / 77 / 66 / 47 % | −0.80 (199) |
| 48 / 240 | 96 | −0.28 | [−0.91, +0.39] | +2.01 | **−2.29** | [−3.15, −1.38] | 93 / 89 / 81 / 67 % | −0.24 (234) |

**k × price moved (absolute move from the fresh-bar close to the late entry):**

| k | bucket | N | days | WR | late mean [CI] | on-time, same | Δ [CI] | Jan–Apr / May–Sep |
|---|---|---|---|---|---|---|---|---|
| 1 | **≤ 1 %** | 173 | 116 | 55 % | **+0.46** [−0.10, +1.03] | +0.85 | −0.40 [−0.69, −0.10] | +0.47 / +0.43 |
| 1 | 1–3 % | 102 | 82 | 44 % | −0.49 [−1.07, +0.13] | −0.18 | −0.31 [−0.90, +0.26] | −0.96 / −0.10 |
| 1 | > 3 % | 22 | 20 | 45 % | −0.52 | +1.13 | −1.65 | +0.07 / −1.02 |
| 3 | **≤ 1 %** | 95 | 74 | 52 % | **−0.13** [−0.77, +0.54] | +0.73 | −0.86 [−1.41, −0.36] | +0.38 / −0.87 |
| 3 | 1–3 % | 115 | 86 | 50 % | −0.07 | +0.40 | −0.46 | −0.20 / +0.09 |
| 3 | > 3 % | 60 | 58 | 52 % | −0.07 | +1.13 | −1.20 | −0.69 / +0.32 |
| 6 | **≤ 1 %** | 59 | 45 | 54 % | **−0.00** [−0.67, +0.64] | +0.61 | −0.61 [−1.28, +0.02] | +0.22 / −0.38 |
| 6 | 1–3 % | 88 | 72 | 57 % | +0.37 | +0.53 | −0.16 | +0.28 / +0.47 |
| 6 | > 3 % | 89 | 67 | 49 % | −0.37 | +1.45 | −1.82 | −0.68 / −0.09 |
| 12 | ≤ 1 % | 31 | 25 | 61 % | +0.43 [−0.60, +1.32] | +1.21 | −0.78 | +0.42 / +0.45 |
| 24 | ≤ 1 % | 20 | 19 | 60 % | +1.23 [−0.42, +3.00] | +1.69 | −0.46 | +0.21 / +2.47 |

**Signed detail at k = 1:**
- **Price below the fresh close.** −1…0 %: +0.57 (76 fills), and the Δ vs on time is +0.16. Below −1 %: −0.85.
- **Price above the fresh close.** 0…+1 %: +0.37 (Δ −0.84). Above +1 %: −0.23 (Δ −1.89).

**Sleeve split, ≤ 1 % (late mean / Δ):**

| sleeve | k = 1 | k = 3 | k = 6 |
|---|---|---|---|
| FRENZY | +0.62 / −0.28 (112 fills) | −0.14 / −0.48 | −0.32 / −0.41 |
| WIDE | +0.15 / −0.61 | −0.11 / −1.60 | +0.62 / −1.02 (20 fills) |

**Selection and books.** The E variants are part of the 43-variant max-t family:
- E k1 ≤ 1 %: p 0.69.
- E k24 no cap: p 0.65.
- E k12 / k24 ≤ 1 %: 0.95 / 0.79.

None passes. The books below delay *every* signal, so they compare against the full on-time book only as a scale reference:

| book (every signal delayed) | end $ |
|---|---|
| k1 ≤ 1 % | $9,348 |
| k1, no cap | $3,232 |
| k3 ≤ 1 % | $2,512 |
| k6 ≤ 1 % | $2,395 |
| on time | $13,474 |

**OOS:** the best Jan–Apr E variant is k1 ≤ 1 % (+0.45), and it reads **+0.43** on May–Sep. That is the only variant in the study that holds out of sample. The best May–Sep variant (k24 no cap, +1.00) reads +0.04 on Jan–Apr.

**What this means for the catch-up feature:**
- **Lateness hurts on its own.** Even with the price within 1 %, every bar of delay gives up edge: −0.40 at 5 min, −0.86 at 15 min, −0.61 at 30 min, all vs the on-time fill of the same signal.
- **Still, vs missing the trade entirely (0), a catch-up is:**
  - **positive only at k = 1** (+0.46 at ≤ 1 %, both halves, OOS-stable, CI touching 0);
  - **about zero at k = 3–6**, at ≤ 1 % and also at 1–3 %.
- **Larger k:** not supported. The k12 and k24 "+" cells are 20–31 fills with CIs spanning 0 and p ≥ 0.79. They are a survivor effect: signals still ON 1–2 h later were the strong ones, with on-time means of +1.2 to +2.3.
- **Looser price cap:** not supported. At k = 1, moves of 1–3 % give −0.49, and > 3 % gives −0.52.
- **Tighter:** the evidence favours **k ≤ 1–2 bars**. k = 2 was not tested, on purpose: it was not pre-registered. The 1 % cap is right. Keep the in-state requirement.
- **The current spec (k ≤ 6, ≤ 1 %) is acceptable but not evidence-positive beyond the first bar.**
- **Pre-registered observe line** for catch-up fills:
  - Stamp k and move on each fill.
  - Review at 20 catch-up fills.
  - If the mean of the k ≥ 3 fills is ≤ 0, cut the window to k ≤ 2.

## 5. D: pair-day cap sensitivity (base book, live one-entry-per-fresh-bar rule)

| cap per pair · sleeve · UTC day | fills | book end | max DD |
|---|---|---|---|
| 1 | 291 | $9,843 | −51 % |
| 2 | 325 | $10,094 | −57 % |
| **3 (live)** | **328** | **$13,474** | **−54 %** |
| 5 | 328 | $13,474 | −54 % |
| unlimited | 328 | $13,474 | −54 % |

- **Signals per pair · sleeve · day:** 1 on 255 days, 2 on 33, 3 on 3. The two fills dropped vs the 330 signals are lost to slots or pair-flat, not to the cap.
- **Inside the re-entry books:** cap 3 refuses 4 re-entries in A m0 W24 V100 K2 (138 uncapped → 134) and 8 in B XP K2 (161 → 153).

**Verdict: the cap is not what stopped NMR or RLC.** Raising it changes nothing. Lowering it costs money.

## 6. Out-of-cohort anecdotes (public 1m / 5m klines; real `frenzy_walk` on them; gvol gate assumed passed; data end 10-06 14:29 UTC)

**Engine state:**
- **NMR:** fresh FRENZY_READY at 13:40. The setup stayed ON through 14:25, the end of the data.
- **RLC:** fresh WIDE (green) at 10-05 12:00. The setup stayed ON continuously to 10-06 14:25.

**NMR 10-06, base fill at 13:40 (15.927):** lock +1.90 at 14:02, peak +3.1.

| rule | what it would have done |
|---|---|
| **A, every variant** | Entered 14:05 at 16.886. That is the breakout bar, +6 % above the fresh close and +3 % above the pre-exit peak. **Marked −1.20 at 14:30, still open.** |
| **B, both lines** | The same 14:05 entry, so −1.20 (marked) |
| **E k = 1** (13:45, move +0.77 %) | +2.96 |
| **E k = 2** (13:50, +0.92 %) | +2.80 |
| **E k = 3** (13:55, +0.12 %): the operator's resume case | **+3.64** (exit 14:05, peak +5.7) |
| **E k = 4** (14:00, +1.27 %) | +2.44. Refused at a 1 % cap. |
| **E k = 5 / 6** (+6.0 / +5.7 %) | −1.20 / −0.91 (marked). Refused at a 1 % cap. |

- The planned catch-up (k ≤ 6, ≤ 1 %) would have taken **13:45** if resumed then. It would have taken **13:55** on the actual 13:53 resume and made **+3.6 %**.
- The +7.7 % run the operator mentions came after this data window, or within it on prints after 14:05. The re-entries above are marked, not exited.

**RLC 10-05, base WIDE at 12:00 (0.4649):** lock +5.27 at 12:10, peak +7.4.

| rule | what it would have done |
|---|---|
| A (all m / W / V), K = 1 | 12:15 at 0.5164 → +1.90 |
| A, K = 2 | A second entry at 12:20 (m 0 / 0.5) or 13:00 (m 1) → −3.10. **Net −1.20** |
| B (both lines) | No trigger: price never pulled back to the exit price within 6 h |
| E k = 1 / 2 / 3 / 6 / 12 / 24 / 48 | +1.90 / +5.71 / +1.90 / +1.90 / −3.10 / +3.74 / −3.10, with moves of +3.5 % to +18 %. Every one is refused at a 1 % cap. |

**None of the angles catches RLC's +110 %.** Under the lock, every RLC re-entry is a 2-point trade, because the lock, not the entry, decides what a ride pays. That matches the split-runner and WIDE reviews.

## 7. Verdict per angle

| angle | verdict | why |
|---|---|---|
| A breakout re-entry (24 variants) | **REFUTED** | All 24 negative. 10 have the CI below 0. Both halves negative in every variant. Worse than random timing 24 / 24. Book −7 to −66 %. Best OOS −0.21. |
| B pullback-reclaim (4) | **REFUTED** | All 4 negative. The best is −0.29 [−0.78, +0.18]. Book −48 % or worse. The timing edge over random fails selection (0.64). |
| C dislocation pocket (3) | **REFUTED as a pocket** | 8 tradeable cases in 9 months, mean ≈ 0. The whole 147-refusal class is +0.14 [−0.32, +0.61]. The big runs exist, but the lock cannot bank them. |
| D pair-day cap | **No change.** Keep 3 | It never binds at 3. Cap ≤ 2 costs $3.4k–3.6k on the book. |
| E delayed entry (12) | **Catch-up: keep it short.** k ≤ 1–2 bars, ≤ 1 %, in-state required. Observe line on k ≥ 3. | Lateness costs −0.4 to −0.9 per fill vs on time, even within 1 %. Only k = 1 at ≤ 1 % is positive vs nothing (+0.46, OOS +0.43; p 0.69, not significant). k 3–6 ≈ 0. Larger k and looser caps are not supported. |

**Nothing passes the pre-registered ship gate.** The best selection-adjusted p across the 43 variants is 0.65, so no haircut applies.

**Re-entry family status: closed again.** These are new angles, not refuted re-runs. Breakout-above-peak and pullback-reclaim had not been tested before (see the coverage list below).

**The operator's frustration is pointing at the exit, not the entry count.** NMR and RLC both had the trade, and the lock sold it at +2 to +5. Every re-entry rule re-buys into the same lock, so it can only add more 2-point trades, at worse prices.

## 8. What earlier studies already covered (cited, not redone)

| report | re-entry definitions covered |
|---|---|
| `FRENZY_REDO_REENTRY_2026-10-05.md` | **#2:** re-entry while ON under the lock (FRESH vs WHILE_ON 15 / 60 min × cap 3 / uncapped × FRENZY / WIDE / WIDE-no-gate) on red engine bars. **#3:** re-entry #1 vs #2 (caps 1–2). Conditional (a) previous won · (b) signal close above the last exit · (c) vol ≥ 300× · (d) a ∧ b. **#7:** ride-signature S1–S4 × FIX / FC × 15 / 30 / 60 × ONE / TWO. **#8:** the same + runner exits R1–R4. **#12:** after a WINNING first trade (lock ≥ +2) × 8 exits × 15 / 60 min × 1 / 2. Total ≈ 540 tests, 0 pass. |
| `FRENZY_RLC_REENTRY_SIGNATURE_2026-10-06.md` | The operator's "everything improving" panel (C7 / L7, single legs, C5 / C6) × lock / EMA20 / EMA50 (± floor) × K, on FRENZY and WIDE candidate bars. Plus an exhaustive 2D screen with a shuffled null and walk-forward. 0 / 220 pass. |
| `FRENZY_STAIRCASE_STUDY_2026-10-06.md` | (a1) second chance: the first 50× fresh bar after the first 100× bar. (a2) volume-faded staircase entry. Both refuted. Post-exit descriptive: 37 % make +15 % within 24 h, 71 % are below the exit 24 h later. |
| `FRENZY_SPLIT_RUNNER_EMA200_2026-10-06.md` | Not re-entry. The split-position runner (EMA200 / EMA100 / 1h EMA50 × stop × cap). |
| `WIDE_FULL_QUANT_REVIEW_2026-10-06.md` | Not re-entry. The WIDE sleeve checklist. Ride anatomy: 84 % of +50 % rides first trade through −3. Post-entry hold-vs-cut. |
| `FRENZY_MINHOURS_AND_WAITRED_2026-10-06.md` | Not re-entry. The min-hours clock and "wait for a red candle" (a delayed entry *conditioned on a red bar*). Refuted. It found the same mechanism E finds: waiting filters out the runners. |
| **New here** | A, breakout above the pre-exit peak (never tested). B, pullback-then-reclaim of EMA20 or the exit price (overlaps #3(b) only loosely: (b) used red bars and any close above the exit). C, entering dislocation refusals. D, cap sensitivity. E, unconditional delay of k bars with a price-move cross. |

## 9. Blind spots (what this study could NOT test)

1. **The cohort ends 2026-09-27.** NMR and RLC are anecdotes on public 1m klines, not part of any statistic. NMR's data ends 14:29, so its re-entries are marks, not exits.
2. **Re-entry gates differ from the base gates.** Re-entries use `gvr_year < 1` at the trigger bar. The base uses the live-like `gvol_live_U2`, which is only stamped on signal bars. The live gvol gate on re-entry bars may differ.
3. **Re-entry sizing is fixed:** FRENZY re-entries at 0.32 (no strong-flag computation), WIDE at 0.2. Strong sizing would amplify the losses of A and B, not reverse them.
4. **A re-entry that finds its pair or slot busy is dropped, not deferred.** A deferred version would re-buy later and higher, which in A is worse.
5. **The timing null uses a 5m-bar pricer for both observed and null.** That is the same statistic, but coarser than ticks. The tick / 1m figures are the P&L of record.
6. **Small pools:**
   - C is only 8 fills tradeable (29 under today's rules in any universe), so its verdict rests on the 147-refusal descriptive class.
   - E at k = 12–48 with ≤ 1 % has only 7–31 fills.
7. **E prices the late entry on the next print with no guard and no orderbook.** In a fast move, a real catch-up could fill worse. k = 2, 4 and 5 were not tested (not pre-registered). The NMR anecdote shows k = 2 and k = 4 behave like their neighbours.
8. **Funding and the 12 h cap** apply as in the cohort. Runner exits were not combined with re-entries; that was tested in #8 and refuted.
9. **No winner / loser separator screen was run on the re-entry triggers.** These are rule tests, not a separator search, so **no "nothing separates good re-entries" claim is made beyond the earlier 2D screens** (RLC signature study).
10. **Unreviewed.** The caveman and deep reviews are pending, and no ship or arm recommendation is made before them.
