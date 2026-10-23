# WILLY take profit at the spike high? (2026-10-23)

**Question (operator):** FRENZY_WILLY takes profit at a fixed +1.25 % net. Should it aim at the spike high (or just below it) instead?
**Motivating case:** ARKUSDT 10-23 13:25 UTC. Entry 0.2125, TP 0.2154 (booked +1.29 %, +$57.45). The spike high was 0.2222, and price
went on to 0.2236 within about 20 minutes.

Research only. No code, config or git change. Pre-registration with three operator addenda:
`reports/WILLY_HIGHTARGET_2026-10-23_PREREG.md`. The timing of each addendum is stated there honestly. Addendum 1 (BTC split) was
written after the un-split results had been read, but before any BTC-split result. Addendum 2 came before any BTC-split result.
Addendum 3 (+0.5) came after the BTC-split results had been read, but before any +0.5 result.

## The answer in plain English

**Keep the fixed +1.25 / −4.** Nothing passes the frozen ship bar:
- none of the 8 high-anchored variants
- none of the 15 BTC-conditional target rules (8 × C(v) plus C1–C5 / C4a–c)
- not the BTC-falling TP × stop grid / C6 (§6b)

1. **A higher target does help on the trades that run, and it costs about the same on the trades that don't.**
   - Across 4,129 backtest fills, aiming at 1 % below the spike high (H1 k = 1) gains +1,530 points on the 920 fills that reach it.
   - It gives back −1,497 points on the 633 fills that would have banked +1.25 and then fell back (to the 30-minute late TP of
     +0.25, or to the −4 stop).
   - Net Δ is +0.01 %/fill. This is the same "runners pay for themselves exactly" pattern as the 10-19 exit study.
2. **It depends on the period, and the walk-forward does not hold.**
   - 2026-H1 likes every high target: about +0.12 to +0.15 %/fill.
   - 2025-H2 dislikes all of them: about −0.03 to −0.12.
   - 2026-H2 (the newest) is about zero: +0.01 to +0.04.
   - The best variant, chosen leave-one-half-out, wins 3 of 4 held-out halves, but the 2026 confidence interval still crosses zero:
     H1 k = 1 is +0.10 [−0.03, +0.24]; H1 k = 2 is +0.09 [−0.02, +0.21].
   - After the 30–50 % in-sample haircut, the 2026 gain is about +0.05 to +0.07 %/fill. That is not enough to clear the bar.
3. **Real fills lean positive, but they are 36 fills on 11 days.**
   - ② master H1 k = 1: +0.24 %/fill, CI [−0.13, +0.72]. Almost all of it comes from the 2–5 % distance bucket and B20's 3 fills.
   - ① live (B22–B23, 17 fills): +0.07 [−0.16, +0.19].
4. **"Bad when BTC is falling, great when BTC is rising" is not supported.** It holds in 3 of 4 backtest halves. It is reversed in
   2026-H1 (falling side +0.26 vs not falling +0.04 at H1 k = 1), and reversed again on the real fills: ② falling +0.24 vs not
   falling +0.23; ① falling +0.35 vs not falling −0.12. ARK itself was a BTC-falling fill.
   - The conditional rules all fail:
     - best C(v) in 2026: +0.05 [−0.03, +0.14], and LOHO is positive in only 2 / 4 halves
     - C1–C5 with the high-target variant fixed by walk-forward: at most 2 / 4 halves
5. **Lower targets on BTC-falling days (+1.0 / +0.75 / +0.5) don't help either.** In the backtest they are below +1.25 in 2025 and
   2026-H1 on the falling side.
   - Only 2026-H2 falling improves: +0.75 gives +0.14 [−0.19, +0.53], +0.5 gives +0.06.
   - All targets are net, as live. A lower target needs a much higher win rate against the −4 stop. The breakeven WR is
     76 / 80 / 84 / 89 % for +1.25 / +1.0 / +0.75 / +0.5. The falling side reaches it only in 2026-H1 (89 % at +0.5, exactly
     breakeven).
   - **Tighter stops on BTC-falling fills (Addendum 4)** are the most interesting side result. At +1.25, a −2.5 stop is +0.12
     (2025), −0.05 (2026-H1) and +0.10 (2026-H2), and positive on both real blocks (4 days each).
   - But the walk-forward cell (T1.25 / S2.5) wins only 2 of 4 held-out halves. C6 fails the bar: 2026 Δ −0.003
     [−0.07, +0.06]. → observe only (§6b).
6. **ARK 10-23 was the textbook winner for this idea.**
   - It was 4.6 % below the high, and the move came within 15–20 minutes, before the 30-minute late TP.
   - Every high variant books +2.4 to +3.4 % instead of +1.25.
   - In the backtest, the same "> 5 % below the high" bucket gains in 2026 (+0.84 H1, +0.32 H2) but loses in 2025 (−0.08).

**What to do:** nothing ships. Register one observe-only shadow line (Scout, below) that prices every new live WILLY fill at the
walk-forward-chosen variant, so the real-fill evidence builds up on fills that did not motivate it.

## 0. Data, calibration

| block | source | N · days | notes |
|---|---|---|---|
| ① live | live WILLY fills B22–B23 (exports = master, dedup (opened_at[:19], pair, direction), CLOSED, 0 operator-closed) + **ARK 10-23** (not in any export yet; public 1m path to 13:42 UTC, provisional) | 17 + 1 · 5 + 1 | |
| ② master | `MASTER_POOL_stacked.csv` FRENZY_WILLY, stack_keep (all 36 kept), B19–B23; WILLY_HT has 0 fills | 36 · 11 | exports ↔ master: 36 = 36, no difference |
| ③ backtest | engine replay `yr6a` (2025) + `yr6` (2026 → 10-11), seeds 1 / 2 / 3, trimmed to each chunk window (`yr5_fills_trimmed.load`), entry A within 60 min of the spike; replay's own fill price and time; 1m paths from the WILLY late-join 1m cache | 4,129 (of 4,156; 27 without data) · 520 | 2025: 2,435 · 2026-H1: 1,103 · 2026-H2: 591 |

**Spike high H:** the highest print from the spike bar's open to the close of the entry red bar. That is exactly `frenzy_walk`'s
`peak` at the entry pass, so it is known at entry. It comes from ticks on 35 of 36 real fills (ICNT 10-16 used public 1m; ARK used
public 1m) and from 1m in ③.

**Walker:** `scout_willy_timecap.evaluate_late(book="live")`.
- ①②: hybrid tick → 1m path.
- **Fee convention (as live, no double-count):**
  - The TP and the stop trigger on **net** P&L. Net = gross − 0.045 entry − 0.045 × exit/entry.
  - A TP win books **exactly its net target**: +1.25 → +1.25 on all 2,679 ③ TP wins and every ①② TP win. The stop books
    −4.00 or worse (gaps).
  - The live resting limit (`exec_shadow.resting_tp_price`) is solved with a maker exit (0.018). So the walker's trigger price sits
    ≈ 0.027 % above the live limit: slightly conservative, with the same booking.
  - **Maker-line sensitivity** (trigger at the live resting price, book the target) changes no conclusion:
    - ③ today 2025 / 26-H1 / 26-H2 = −0.298 / +0.176 / +0.052 (main: −0.308 / +0.174 / +0.020)
    - H1 k 1 Δ = −0.038 / +0.153 / +0.012
    - 2026 CIs: H1 k 1 [−0.025, +0.238], H1 k 2 [−0.019, +0.207], H3 [−0.012, +0.119]
  - This study does not use the gross-line-minus-fees walker of `study_frenzy_realfills_common.walk` / `scout_long_stop4.walk`.

**Parity:**
- ①②: today's exit re-walked = booked `stack_pct` on **36 / 36 fills within 0.05** (37 / 37 incl. ARK: walked +1.25 vs booked +1.29, the real fill printed slightly above the line) (max 0.016, ICNT 10-20's booked +1.27 resting
  fill). Mean +0.263 vs +0.263.
- ③: walked today vs the replay's as-run result:

| period | walked today | replay as-run |
|---|---|---|
| 2025 | −0.31 | −0.22 |
| 2026-H1 | +0.17 | +0.14 |
| 2026-H2 | +0.02 | +0.07 |

  The replay's exit era differs (no late TP), so this is a level check, not parity. **All reads are paired Δs on the same path.**
  The memory-note walker pessimism (≈ −0.49 %/trade on the entry model) does not apply to Δ.
- **Integrity:** on fills whose variant target equals +1.25, Δ = 0.0000 exactly, in every block and every variant.

## 1. Before → after per block (avg % net per fill; Δ = variant − today, day-clustered 95 % CI)

**Variant definitions:**
- **H1(k):** TP at H × (1 − k), never below +1.25, capped at +8.
- **H2(k):** the same without the +1.25 floor (floored at +0.25).
- **H3:** half at +1.25, half at H1(1 %).
- **H4:** H1(1 %) with a 180-min cap when the target is above +1.25.
- **D:** H1(1 %) with no late TP on the far fills. Diagnostic only, not ship-eligible.

All variants keep the −4 stop, the 90-min cap and the +0.25 late TP from minute 30.

| exit | ① live 17 | ② master 36 | ③ 2025 (2,435) | ③ 2026-H1 (1,103) | ③ 2026-H2 (591) | ③ 2026 Δ [95 % CI] |
|---|---|---|---|---|---|---|
| **today +1.25** | 76 % · −0.103 | 83 % · +0.263 | 72 % · −0.308 | 80 % · +0.174 | 79 % · +0.020 | – |
| H1 k 0.5 | +0.138 (Δ +0.24) | +0.240 (−0.02) | −0.316 (−0.01) | +0.307 (+0.13) | −0.050 (−0.07) | +0.06 [−0.08, +0.21] |
| **H1 k 1** | −0.029 (+0.07) | **+0.499 (+0.24 [−0.13, +0.72])** | −0.366 (−0.06) | +0.325 (+0.15) | +0.032 (+0.01) | **+0.10 [−0.03, +0.24]** |
| **H1 k 2** | −0.136 (−0.03) | +0.265 (+0.00) | −0.305 (+0.00) | +0.296 (+0.12) | +0.058 (+0.04) | **+0.09 [−0.02, +0.21]** |
| H2 k 0.5 | +0.079 (+0.18) | +0.147 (−0.12) | −0.321 (−0.01) | +0.242 (+0.07) | −0.129 (−0.15) | −0.01 [−0.16, +0.16] |
| H2 k 1 | −0.234 (−0.13) | +0.423 (+0.16) | −0.395 (−0.09) | +0.206 (+0.03) | +0.017 (−0.00) | +0.02 [−0.12, +0.17] |
| H2 k 2 | −0.111 (−0.01) | +0.266 (+0.00) | −0.372 (−0.06) | +0.124 (−0.05) | +0.021 (+0.00) | −0.03 [−0.18, +0.12] |
| H3 split | −0.066 (+0.04) | +0.381 (+0.12 [−0.06, +0.36]) | −0.337 (−0.03) | +0.250 (+0.08) | +0.026 (+0.01) | +0.05 [−0.01, +0.12] |
| H4 (180 min) | −0.029 (+0.07) | +0.499 (+0.24) | −0.373 (−0.07) | +0.330 (+0.16) | +0.020 (−0.00) | +0.10 [−0.03, +0.24] |
| D no-late (diag.) | +0.113 (+0.22) | +0.590 (+0.33) | −0.399 (−0.09) | +0.326 (+0.15) | +0.139 (+0.12) | +0.14 [−0.02, +0.30] |
| F +1.0 | −0.181 (−0.08) | +0.122 (−0.14) | −0.358 (−0.05) | | | |
| F +0.75 | −0.093 (+0.01) | +0.073 (−0.19) | −0.398 (−0.09) | | | |

Win rate drops with any higher target. For ③ all: 72 → 63–66 % (H1). The worst single fill is unchanged (the stop).

**Per seed:** the three seeds agree to within 0.01–0.02 on every Δ. For example, H1 k 1 is −0.06 / +0.15 / +0.01 for
2025 / 26-H1 / 26-H2 in every seed. So the seeds are one market realisation, not three independent ones.

**② per batch (Δ, H1 k 1):**

| batch | N | Δ |
|---|---|---|
| B19 | 14 | +0.06 |
| B20 | 3 | **+2.16** |
| B21 | 2 | −0.05 |
| B22 | 8 | +0.20 |
| B23 | 9 | −0.04 |

Without B20's 3 fills, master Δ is about +0.06.

**H4 (the 180-minute cap) adds nothing.** H4 equals H1 k 1 in ①② and is within 0.01 in ③. The 30-minute late TP decides almost
every far-target fill before 90 minutes.

## 2. Affected cohort — who gains, who loses

"Affected" = the variant's target is not +1.25. Unaffected fills: Δ = 0 exactly.

| block · variant | affected N (days) | gained (reached the new target): N · Σ Δ | lost (today took +1.25, variant didn't): N · Σ Δ | net Σ Δ |
|---|---|---|---|---|
| ② H1 k 1 | 20 (11) | 12 · +14.0 | 5 · −5.4 | +8.5 |
| ② H1 k 0.5 | 23 (11) | 11 · +11.8 | 8 · −12.6 | −0.8 |
| ③ all · H1 k 1 | 2,381 (412) | 920 · +1,529 | 633 · −1,497 | +32 |
| ③ all · H1 k 2 | 1,588 (322) | 634 · +1,053 | 388 · −888 | +165 |
| ③ 2026 · H1 k 1 | 873 (175) | 393 · +616 | 231 · −442 | +174 |
| ③ all · H2 k 1 (no floor) | 4,129 (520) | 2,357 · +1,317 | 633 · −1,497 | −180 |

**The "lost" fills are the real cost.** Of every 100 far-target fills, roughly 39 reach the high target and 27 had +1.25 in hand
and gave it back. Dropping the +1.25 floor (H2) is worse: on fills already near the high, it sells below +1.25.

## 3. Distance to the high (entry vs H)

| bucket | ② master: N · today · Δ H1 k 1 | ③ 2025 · Δ H1 k 1 | ③ 2026-H1 · Δ | ③ 2026-H2 · Δ |
|---|---|---|---|---|
| ≤ 2 % | 15 · +0.29 · 0.00 | 720 · 0.00 | 383 · 0.00 | 277 · 0.00 |
| 2–5 % | 16 · −0.07 · **+0.57** | 1,118 · −0.08 | 503 · −0.03 | 242 · −0.07 |
| > 5 % | 5 · +1.25 · −0.12 | 597 · −0.08 | 217 · **+0.84** | 72 · **+0.32** |

- The variants only differ from today when H is far, and "far" behaves inconsistently.
- Real fills gain in 2–5 %, where the backtest is negative in every period.
- The backtest gains in > 5 % in 2026, but not in 2025.
- This is a screen, not a rule. A "> 5 % only" rule was not pre-registered and is not proposed. It goes to the Scout line as a
  sub-read.

## 4. Walk-forward (③, seeds pooled)

**Leave-one-half-out** (choose among the 8 on the other three halves):

| held-out | chosen | held-out Δ [95 % CI] |
|---|---|---|
| 2025-H1 | H1 k 2 | +0.050 [−0.12, +0.21] |
| 2025-H2 | H1 k 1 | −0.115 [−0.27, +0.04] |
| 2026-H1 | H1 k 2 | +0.122 [−0.03, +0.27] |
| 2026-H2 | H1 k 2 | +0.038 [−0.10, +0.17] |

That is 3 / 4 halves positive.

**Anchored forward** (2025-H2 / 2026-H1 / 2026-H2): H2 k 0.5 → −0.112 · H1 k 2 → +0.122 · H1 k 2 → +0.038.

**Δ per half:**

| variant | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 |
|---|---|---|---|---|
| H1 k 1 | +0.04 | −0.12 | +0.15 | +0.01 |
| H1 k 2 | +0.05 | −0.03 | +0.12 | +0.04 |
| H3 | +0.02 | −0.06 | +0.08 | +0.01 |

## 5. Frozen ship bar

| variant | halves Δ > 0 (of 4) | 2026-H1 | 2026-H2 | ② master Δ | ③ 2026 CI lower | verdict |
|---|---|---|---|---|---|---|
| H1 k 0.5 | 2 | +0.132 | −0.070 | −0.023 | −0.084 | no |
| **H1 k 1** | 3 ✓ | +0.151 ✓ | +0.012 ✓ | +0.236 ✓ | **−0.026 ✗** | no |
| **H1 k 2** | 3 ✓ | +0.122 ✓ | +0.038 ✓ | +0.002 ✓ | **−0.019 ✗** | no |
| H2 k 0.5 / 1 / 2 | 2 / 2 / 2 | | | | ≤ −0.12 | no |
| H3 | 3 ✓ | +0.075 ✓ | +0.006 ✓ | +0.118 ✓ | **−0.013 ✗** | no |
| H4 | 2 | +0.155 | −0.001 ✗ | +0.236 | −0.030 | no |

H1 k 1, H1 k 2 and H3 fail only on the confidence condition. Per the CLAUDE.md rule, a cohort that fails only on confidence becomes
a pre-registered observe candidate with a frozen threshold, never re-fit. That is the Scout line below.

## 6. BTC direction (Addenda 1–3)

**BTC side definition:** the exact WILLY_BTC1H scout feature: BTC 1 h EMA20 slope of the last closed hour vs the one before;
≤ 0 = falling.
- The scout **computes it from BTC klines; it does not use the stamp.** Here it is computed the same way on every block.
- The stamped `entry_btc_1h_slope` agrees in sign on 35 / 36 real fills. 0 fills were unscored.

**Δ per side, H1 k 1** (avg Δ [95 % day CI]):

| block | BTC not falling: N · days · Δ | BTC falling: N · days · Δ |
|---|---|---|
| ① live + ARK | 6 · 3 · −0.12 [−0.28, +0.05] ⚠ small | 12 · 4 · **+0.35** [+0.09, +1.00] ⚠ small |
| ② master | 23 · 9 · +0.23 [−0.24, +1.05] | 13 · 4 · +0.24 [+0.09, +0.44] ⚠ small |
| ③ 2025 | 1,437 · 201 · +0.00 | 998 · 173 · −0.15 [−0.36, +0.06] |
| ③ 2026-H1 | 542 · 94 · +0.04 | 561 · 101 · **+0.26** [−0.01, +0.53] |
| ③ 2026-H2 | 364 · 58 · **+0.18** [−0.01, +0.37] | 227 · 49 · **−0.25** [−0.58, +0.05] |

- The operator's direction (high target better when BTC is not falling) holds in ③ 2025-H1, 2025-H2 and 2026-H2.
- It **reverses** in ③ 2026-H1 and on both real blocks.
- Inside each side, the LOHO walk-forward is positive in only 2 / 4 halves (not falling: +0.08 / +0.01 / −0.03 / −0.01).

**Conditional rules** (BTC not falling → high-anchored V* fixed by walk-forward; falling → the stated fixed TP):

| rule | held-out halves Δ > 0 | ③ 2026 Δ [95 % CI] | ② master Δ | ① live Δ (18) | verdict |
|---|---|---|---|---|---|
| C1 (falling +1.25) | 2 | −0.013 [−0.11, +0.08] | −0.171 | +0.024 | no |
| C2 (falling +1.0) | 1 | −0.035 [−0.14, +0.06] | −0.180 | +0.006 | no |
| C3 (falling +0.75) | 2 | +0.004 [−0.12, +0.12] | −0.110 | +0.145 | no |
| C5 (falling +0.5) | 1 | −0.062 [−0.18, +0.06] | −0.180 | +0.007 | no |
| C4a (+1.25 / falling +1.0) | 1 | −0.022 [−0.06, +0.02] | −0.009 | −0.018 | no |
| C4b (+1.25 / falling +0.75) | 2 | +0.016 [−0.06, +0.10] | +0.061 | +0.121 | no |
| C4c (+1.25 / falling +0.5) | 1 | −0.050 [−0.13, +0.04] | −0.009 | −0.018 | no |
| best C(v) with a fixed v (C_H1 k 1) | 3 | +0.052 [−0.03, +0.14] | +0.149 | −0.041 | no (CI) |

- The walk-forward V* flips between H1 k 2 and H1 k 0.5 across folds, which is unstable.
- V* chosen on all ③ halves = H1 k 0.5. That is the variant that loses on ② master (−0.27 on the not-falling side).

**Falling-side target ladder** (avg % net; ③):

| target | 2025 | 2026-H1 | 2026-H2 | ② master (13) | ① live (12) |
|---|---|---|---|---|---|
| +1.25 (today) | −0.526 | **+0.181** | −0.325 | −0.518 | −0.225 |
| +1.0 | −0.553 | +0.120 | −0.340 | −0.542 | −0.252 |
| +0.75 | −0.583 | +0.172 | **−0.183** | **−0.350** | **−0.043** |
| +0.5 | −0.639 | +0.005 | −0.261 | −0.542 | −0.252 |

**Breakeven WR vs observed WR** (all targets net, stop −4; BTC falling; observed WR in ③ 2025 / 2026-H1 / 2026-H2 · ② · ①):

| target | breakeven WR vs −4 | observed WR |
|---|---|---|
| +1.25 | 76 % | 68 / 80 / 71 · 69 · 75 |
| +1.0 | 80 % | 70 / 82 / 74 · 69 · 75 |
| +0.75 | 84 % | 73 / **88** / 80 · 77 · 83 |
| +0.5 | 89 % | 76 / **89** / 83 · 77 · 83 |

The +0.25 late-TP "wins" count as wins but pay less than the target, so the real breakeven is a little higher than shown.
Lowering the target raises the WR by less than it raises the breakeven, except in 2026-H1.

**WILLY_BTC1H lock (DECISION_LOG 275 / 281, frozen, still collecting):**
- From the DL 281 deploy (2026-10-12 04:11 UTC) there are 23 WILLY fills incl. ARK (22 exported).
- **BTC falling: 14 fills on 5 days** (13 on 4 without the provisional ARK), today avg −0.39. Not falling: 9 fills, +0.66.
- The gate needs N ≥ 15 on ≥ 8 days. **This study does not override it.** It only informs it.
- In this study, the falling side's problem is not fixed by changing the target in either direction. That is consistent with
  WILLY_BTC1H being an on / off question, not an exit question. The scout's own tally rules.

## 6b. Addendum 4 — BTC-falling TP × stop grid (16 cells), walk-forward, C6

Each cell shows Δ vs +1.25 / −4 (avg % net); WR / breakeven WR is in the per-fill tables (`_grid_*.csv`). Full grid, no cherry-pick:

| TP \ stop | ③ 2025 (998 · 173 d): −2 / −2.5 / −3 / −4 | ③ 2026-H1 (561 · 101 d) | ③ 2026-H2 (227 · 49 d) | ② master (13 · 4 d) ⚠ | ① live+ARK (12 · 4 d) ⚠ |
|---|---|---|---|---|---|
| +0.5 | −0.06 / −0.01 / −0.02 / −0.11 | −0.31 / −0.23 / −0.23 / −0.18 | +0.21 / +0.18 / +0.13 / +0.06 | +0.44 / +0.32 / +0.21 / −0.03 | +0.31 / +0.23 / +0.14 / −0.03 |
| +0.75 | +0.02 / +0.06 / +0.03 / −0.06 | −0.17 / −0.06 / −0.06 / −0.01 | +0.27 / +0.23 / +0.17 / +0.14 | +0.63 / +0.52 / +0.40 / +0.17 | +0.52 / +0.43 / +0.35 / +0.18 |
| +1.0 | +0.07 / +0.10 / +0.07 / −0.03 | −0.22 / −0.10 / −0.10 / −0.06 | +0.15 / +0.11 / +0.06 / −0.02 | +0.59 / +0.44 / +0.29 / −0.03 | +0.47 / +0.35 / +0.22 / −0.03 |
| +1.25 | +0.10 / **+0.12** / +0.08 / 0 | −0.19 / **−0.05** / −0.04 / 0 | +0.16 / **+0.10** / +0.09 / 0 | +0.36 / **+0.46** / +0.31 / 0 | +0.22 / **+0.38** / +0.25 / 0 |

Breakeven WR per cell = |stop| / (TP + |stop|):

| TP \ stop | −2 | −2.5 | −3 | −4 |
|---|---|---|---|---|
| +0.5 | 80 % | 83 % | 86 % | 89 % |
| +0.75 | 73 % | 77 % | 80 % | 84 % |
| +1.0 | 67 % | 71 % | 75 % | 80 % |
| +1.25 | 62 % | 67 % | 71 % | 76 % |

**Walk-forward (LOHO, falling fills):**

| held-out | chosen cell | held-out Δ |
|---|---|---|
| 2025-H1 | T0.75 / S2.5 | −0.054 |
| 2025-H2 | T1.25 / S2.5 | +0.085 |
| 2026-H1 | T1.25 / S2.5 | −0.050 |
| 2026-H2 | T1.25 / S2.5 | +0.103 [−0.19, +0.39] |

- That is **2 / 4** halves positive. The cell chosen on all halves is **+1.25 / −2.5** (it is today's TP with a tighter stop).
- **Smoothness:** a gentle plateau, not an isolated peak (+0.064 vs neighbours +0.042 / +0.015 / +0.044). But the plateau has the
  opposite sign in 2026-H1.

**Family-wise null:** the whole LOHO procedure was re-run on 300 random day-relabels.
- Real mean held-out Δ is +0.021, vs a null mean of −0.046 (p95 +0.000): **p = 0.02**.
- So the falling-side choice does better than random day sets, but the out-of-sample gain is only +0.02 %/fill.
- Selecting from 16 cells loses money out of sample on random days. That is the multiple-testing cost, and it is why the bar
  needs walk-forward.

**Stop-width CF at TP +1.25, falling, two-sided** (Δ = 0.0000 on every untouched fill in every block ✓):

| block | stop | live-stopped (loss shrunk): N · Σ Δ | cut winners: N · Σ Δ | mean Δ [95 % day CI] |
|---|---|---|---|---|
| ③ 2025 | −2.5 | 296 · +416 | 83 · −284 | +0.119 [−0.01, +0.25] |
| ③ 2026-H1 | −2.5 | 99 · +149 | 51 · −174 | −0.050 [−0.21, +0.11] |
| ③ 2026-H2 | −2.5 | 57 · +84 | 18 · −62 | +0.103 [−0.19, +0.39] |
| ② master | −2.5 | 4 · +6.0 | 0 | +0.463 [+0.19, +0.75] ⚠ 4 days |
| ① live+ARK | −2.5 | 3 · +4.5 | 0 | +0.375 [+0.00, +0.65] ⚠ 4 days |

At −2 the cut winners grow fastest (2026-H1: 102 cut, Δ −0.19).

**C6** (not falling → +1.25 / −4, since no high-anchored variant passed on that side; falling → the walk-forward cell):

| check | result |
|---|---|
| held-out Δ per half | −0.020 / +0.037 / −0.026 / +0.040 (2 / 4) |
| ③ 2026 Δ | −0.003 [−0.071, +0.062] |
| ② master | +0.167 (B20 +0.50 · B22 +0.19 · B23 +0.34) |
| ① live | +0.250 |
| **verdict** | **no** |

**Reading:**
- The real fills (13 / 12 falling fills on 4 days) love the tighter stop because their 3–4 stops never came back.
- The 2-year backtest says it depends on the period, and it loses in 2026-H1.
- This sits squarely in the WILLY_BTC1H question (BTC falling = the sleeve's weak side). It is information for that frozen gate,
  not a reason to bypass it.
- If anything, it suggests a pre-registered observe line `WILLY_FALL_S25`: shadow −2.5 on BTC-falling fills, read on the
  live-stopped cohort per the CLAUDE.md stop rule. It is listed in the Scout section.

## 7. ARKUSDT 2026-10-23 13:25 (① live, provisional public 1m)

**The setup:**
- Spike from 12:40, high 0.2222 in the 12:55 bar. A red bar at 13:00 was not taken; entry came on the 13:20–13:25 red bar at 0.2125.
- dist-to-high +4.56 %.
- BTC 1 h slope −0.002, so the **falling** side.

**The path:** a −1.3 % dip in the 13:25 bar, then TP 0.2154 in the 13:35 bar, then 0.2236 in the 13:40 bar (≈ 15–17 min after entry).

| exit | target | ARK result |
|---|---|---|
| today | +1.25 | +1.25 walked (booked +1.29) |
| H1 / H2 k 2 | +2.38 | +2.38 |
| H1 / H2 k 1, H4, D | +3.43 | +3.43 |
| H3 | split | +2.34 |
| H1 k 0.5 | +3.95 (0.2211) | unresolved on the 1m data to 13:41 (the 0.2236 print is in the 13:42 minute) — very likely a TP given the operator's report |
| C3 / C4b / C5 (falling → lower TP) | +0.75 / +0.5 | would have **lost** 0.5–0.75 on ARK |

ARK is exactly the fill the idea is built for: far from the high, with a fast move before the 30-minute late TP. The 2026 backtest
gains in that bucket. The 2025 backtest and the rest of the cohort show why it doesn't generalise: the fills that fall back cost as
much as the ARKs earn.

## 8. Recommendation

- **Keep `frenzy_willy_tp_pct` = +1.25. No ship, no revert gate needed.**
- Haircut reference: the best 2026 un-split Δ is +0.10 (H1 k 1), or **+0.05 to +0.07 after the 30–50 % haircut**, with a CI that
  includes 0.
- Don't add a BTC-conditional target, and don't lower the falling-side target. Leave the BTC question to the frozen WILLY_BTC1H gate.
- The 30-minute late TP is what caps the upside of any high target (diagnostic D: +0.33 on master, +0.14 on ③ 2026, CI including 0).
  If the operator wants to explore this further, the open question is the late-TP rule on far-from-high fills. That would be a
  new pre-registration, not a conclusion of this study.
- Don't tighten the falling-side stop yet. +1.25 / −2.5 is the walk-forward pick but wins only 2 / 4 held-out halves, and C6 fails
  (2026 Δ −0.003). Shadow it (Scout).

## 9. Scout — frozen observe-only shadow line `WILLY_HIGHTARGET`

- **Cohort:** every CLOSED FRENZY_WILLY fill (non-operator-closed; dedup (opened_at[:19], pair, direction)) opened **≥ 2026-10-23
  15:00 UTC**. ARK 10-23 and every fill in this study are excluded (motivating).
- **Shadow:** each fill is re-walked on the hybrid tick path, with H = the max print from (`entry_frenzy_spike_at` − 5 min) to the
  entry red bar's close.
  - **Primary:** H1 k = 2 % (the LOHO pick in 3 / 4 folds): target = clamp(net(H × 0.98), +1.25, +8).
  - **Secondary:** H1 k = 1 %.
  - Both keep the −4 stop, the 90-min cap and the +0.25 late TP from minute 30. Live booking: a resting TP books the line.
  - Δ = shadow − today's walked exit.
- **Per-fill stamps:** H, dist-to-high, target, shadow why / %, BTC side (WILLY_BTC1H feature), the today-TP-then-shadow-not flag.
- **Integrity check:** Δ = 0 exactly on fills whose target = +1.25. Walker parity vs booked ≥ 90 %.
- **Frozen read, once,** at **≥ 30 affected fills (target > +1.25) on ≥ 10 UTC days:**
  - **CANDIDATE** (operator decides; then still observe → arm) iff all of:
    - mean Δ over ALL cohort fills > 0, with day-bootstrap P(Δ > 0) ≥ 0.90
    - gained Σ Δ > |lost Σ Δ| on the affected fills
    - no single day or pair ≥ 50 % of Σ Δ
    - Δ ≥ 0 on BOTH BTC sides OR the operator explicitly accepts a side split (then reported in day units)
  - **RETIRE** if mean Δ ≤ 0 at the read.
  - Otherwise keep observing to 60 affected fills, then a final read.
- **Sub-reads (info only, never the verdict):** dist-to-high > 5 % vs 2–5 %; BTC side; late-TP-capped share of far fills.
- **Never re-fit k.**

**🔒 `WILLY_FALL_S25` (observe-only, same cohort start).**
- **Shadow:** every BTC-falling WILLY fill (WILLY_BTC1H feature) re-walked at stop −2.5, TP +1.25, same cap and late TP.
- **Read** on the live −4-stopped cohort plus the cut winners (two-sided), Δ = 0 on untouched fills, at **≥ 15 falling fills on
  ≥ 8 UTC days** (the WILLY_BTC1H unit).
- **CANDIDATE** iff mean Δ over all falling fills > 0 with day-bootstrap P ≥ 0.90, saved Σ > |cut-winner Σ|, and no day ≥ 50 % of
  Σ Δ.
- **RETIRE** if mean Δ ≤ 0.
- It never overrides the WILLY_BTC1H gate. If that gate arms (switch WILLY off on falling BTC), this line becomes moot.

## Blind spots / what could not be tested

- **Real-fill N:** 36 + ARK on 12 days. Every per-side real number is ⚠ small (falling 13 fills on 4 days in master). Real blocks can
  refute, not confirm.
- **ARK is 1m-provisional.** Its path ends at 13:42 UTC, and it is not in any orders export (the latest export is 10-21 02:14). Its
  entry second (13:25:08) and its spike bar (12:40–12:55; H is the same for any of them) are inferred.
- **Backtest:**
  - Entries are the yr6 / yr6a replay's (config 01419d4, Oct-12), not the newest entry guards (turnover ≤ 1.0, R_NOMISS, cold-bar
    window).
  - Paths are 1m, not ticks (a bar touching both lines = stop).
  - 27 fills had no data.
  - The three seeds are near-copies of one realisation.
- **Second-order effects not modelled:** a longer-held far-target WILLY occupies WILLY's single slot longer and can block the next
  WILLY. Δ is per fill on the same entries.
- **Fees:** targets are net and book exactly the target. The walker's TP trigger price is ≈ 0.027 % above the live maker
  limit. The maker-line sensitivity changes no conclusion (§0).
- **Pre-registration timing:** Addenda 1, 3 and 4 were written after some results were known (stated in the PREREG). Nothing
  passed, so this cannot have produced a false ship.
- **Multiple testing:** 8 + 8 + 7 + 16 cells / rules were read. The falling-grid family-wise null (p = 0.02) covers only the grid.

## Files

- Pre-registration: `reports/WILLY_HIGHTARGET_2026-10-23_PREREG.md`
- Per-fill:
  - `reports/WILLY_HIGHTARGET_2026-10-23_real.csv` / `_real_btc.csv` (①② + ARK)
  - `_bt.csv` / `_bt_btc.csv` (③)
- Ship-bar tables: `_shipbar.csv` · `_shipbar_btc.csv` · `_shipbar_rules.csv`
- Falling grid: `_grid_bt.csv` · `_grid_real.csv`
- Scripts: `scripts/study_willy_hightarget_common.py` · `_real.py` · `_bt.py` · `_report.py` · `_btc.py` · `_grid.py`
- Maker-line sensitivity script (scratch): `/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad/maker_sens.py`
- Binance: 3 public fapi kline requests (ICNT 10-15 1m, ARK 10-23 1m, BTC 5m from 10-20). Used weight ≤ 6, no 418 / 429, cooldown
  checked first.
