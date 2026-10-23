# 🐻✂ Bear-Run false-start EXIT study: can a post-ON exit save V2? (2026-10-23)

> Pre-registration, frozen before any result: `reports/BEARRUN_FALSESTART_EXIT_2026-10-23_PREREG.md`.
> Scripts: `scripts/study_bearrun_fsexit.py` (main), `scripts/study_bearrun_fsexit_checks.py` (seed 2, live and master, sizing),
> `scripts/study_bearrun_fsexit_last48.py` and `_fetch.py` (V2 state after the replay's data end; public klines, ~30 requests).
> Data files: `reports/BEARRUN_FALSESTART_EXIT_2026-10-23_{cands,wf,wf_per_trigger,classes,cf,null,kills,seed2,live,sizing,last48,fills_*}.csv`.
> Research only: no engine or config change.

## Verdict (plain English)

**NO-SHIP.** No exit rule makes V2 (or T0) beat today's live sleeve out of sample. The pre-registered bar fails on walk-forward
and on the haircut confidence bound.

What I found:

1. **One rule family really does separate false starts from clean slides after ON: R1, "BTC bounces ≥ 0.5 % off its post-ON low".**
   - On V2-C1 it removes 98–140 of the −158 false-start loss and gives up only 18–52 of the +93 clean-slide gain.
   - It turns V2-C1 from **Σ −143 (exit d)** into **Σ +19 … +56**.
   - The timing carries real information. On the shuffled-timing null (same kill delays dealt to random windows), R1 sits at the
     **100th / 99.5th percentile**, while random cutting averages Σ −37.
   - It holds on seed 2: on the 12 finished blocks, every R1 0.5 variant beats exit d on both seeds.
2. **Almost all of R1's value comes from what it STOPS, not what it closes.** The mechanism is "stop opening new shorts once BTC
   has bounced 0.5 %". On V2-C1 R1(0.5 %, 1 h), the blocked later entries account for **+160** of the gain, while closing the open
   shorts early adds only **+1.8** (two-sided: 31 saved, 10 cost; untouched fills Δ = 0 exactly). So it works as an entry gate,
   not an exit.
3. **It is still not an edge over today.** The best candidate is V2-C1 + R1(0.5 %, 1 h) + 1 % trail:
   - Σ +55.9 vs T0 −7.0, so ΔΣ +63.
   - Window-bootstrap CI is −52 … +183, P(Δ > 0) = 0.86. After the 50 % haircut the lower bound is −83.
   - It loses in 2026-H2 (−3.9).
   - The pre-registered walk-forward over all 102 candidates picks 4 different rules in 4 folds, and the held-out ΔΣ vs T0 is
     **negative in all 4** (−28.7, −53.0, −17.1, −22.9).
   - The per-trigger walk-forward, choosing only among V2-C1 exits, picks R1 0.5 % every time but is negative in **both 2026 halves**
     (−17.1, −3.6).
4. **R2 (no follow-through), R3 (1 h close back above ON) and R4 (sleeve P&L stop) all fail:**
   - R2 kills clean slides as often as false starts. On 10-22 it would have closed at 03:00, one hour into the best slide of the
     month.
   - R4 makes things worse (V2-C1 R4 0.5: Σ −154, CI entirely below 0). It cuts at the bottom of the noise.
   - The trail alone (T) helps every exit by about +60 to +90, but it is not enough on its own (V2-C1 T: Σ −53).
5. **On today's T0 trigger the rules hardly matter.** There are 10 windows and 24 fills, so one window carries 70–90 % of any
   difference.
   - T0 + R1(0.5 %, 1 h) is Σ +4.6 vs −7.0, ΔΣ +11.6 (CI −3 … +39), but 89 % of it sits in one window. That fails the
     concentration bar.

**Sizing (operator ask):** keep the 10-23 cut (0.2 × 5×, ≈ $500/short). Do not revert to 1.0 × 5×. Details are in the
Recommendation section.

## ③ Replay: before → after (seed 1, E0 entry arm, 1×, Σ % per short summed)

T0 = today's trigger with today's exit a: **44 fills · 52 % · −0.160 · Σ −7.03** (−7.33 / +0.60 / 0.00 / −0.30 by half), 10 windows.
My exit-d re-pricing reproduces the price script: V2-C1 698 fills, Σ −142.6 (script: 697, −144.8); V2-C2 904, −73.3 (903, −75.5).

| trigger · exit | N | WR | avg % | Σ % | window mean | windows | kills | 25-H1 | 25-H2 | 26-H1 | 26-H2 | ΔΣ vs d | ΔΣ vs T0 [95 % CI] | lo after haircut | P>0 | max window share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V2-C1 · d | 698 | 34 | −0.204 | −142.6 | −0.47 | 185 | 0 | −37.3 | −74.0 | −35.0 | +3.7 | 0 | −135.5 [−315, +63] | −247 | 0.09 | 0.13 |
| V2-C1 · T (trail) | 809 | 46 | −0.066 | −53.2 | −0.40 | 185 | 0 | +3.1 | −16.0 | −14.9 | −25.4 | +89.4 | −46.2 [−178, +98] | −155 | 0.26 | 0.08 |
| V2-C1 · R1 0.5 %/1 h | 449 | 38 | +0.043 | +19.2 | −0.22 | 136 | 345 | +21.5 | −26.5 | 0.0 | +24.2 | +161.7 | +26.2 [−132, +211] | −145 | 0.60 | 0.16 |
| **V2-C1 · R1 0.5 %/1 h + T** | 532 | 49 | +0.105 | **+55.9** | −0.13 | 136 | 345 | +36.7 | +7.8 | +15.4 | **−3.9** | +198.5 | **+63.0 [−52, +183]** | **−83** | 0.86 | 0.09 |
| V2-C1 · R1 0.5 %/4 h | 276 | 37 | +0.140 | +38.8 | −0.10 | 100 | 542 | +10.0 | −3.0 | −17.1 | +48.8 | +181.3 | +45.8 [−43, +151] | −66 | 0.82 | 0.18 |
| V2-C1 · R1 0.5 %/4 h + T | 309 | 44 | +0.143 | +44.2 | −0.06 | 100 | 542 | +9.7 | +21.1 | −8.4 | +21.9 | +186.8 | +51.3 [−20, +131] | −46 | 0.91 | 0.11 |
| V2-C1 · R1 0.75 %/2 h + T | 656 | 49 | +0.045 | +29.5 | −0.31 | 158 | 330 | +20.8 | +16.7 | +8.4 | −16.4 | +172.1 | +36.6 [−84, +166] | −102 | 0.71 | 0.08 |
| V2-C1 · R1 1.0 %/2 h | 661 | 34 | −0.194 | −128.0 | −0.46 | 183 | 214 | −31.0 | −67.4 | −35.6 | +5.8 | +14.5 | −121.0 | −233 | 0.10 | 0.14 |
| V2-C1 · R2 1 %/1 h | 208 | 38 | −0.121 | −25.2 | −0.27 | 81 | 424 | −16.1 | +9.2 | −24.8 | +6.5 | +117.4 | −18.2 [−66, +31] | −57 | 0.23 | 0.18 |
| V2-C1 · R3 | 448 | 35 | −0.038 | −17.0 | −0.32 | 128 | 240 | +13.4 | −37.4 | −16.6 | +23.6 | +125.5 | −10.0 [−160, +166] | −155 | 0.43 | 0.17 |
| V2-C1 · R4 0.5 % | 493 | 21 | −0.312 | −154.1 | −0.38 | 185 | 163 | −81.6 | −27.9 | −32.2 | −12.4 | −11.5 | −147.0 [−238, −53] | −164 | 0.00 | 0.13 |
| V2-C2 · d | 904 | 36 | −0.081 | −73.3 | −0.42 | 210 | 0 | −30.6 | −45.3 | +12.1 | −9.5 | 0 | −66.2 [−280, +163] | −247 | 0.27 | 0.10 |
| V2-C2 · T | 1046 | 48 | −0.015 | −15.3 | −0.39 | 210 | 0 | −24.7 | +7.4 | +26.9 | −24.9 | +57.9 | −8.3 | −165 | 0.46 | 0.06 |
| V2-C2 · R1 0.5 %/1 h + T | 748 | 49 | +0.065 | +48.3 | −0.23 | 172 | 421 | +33.6 | −6.5 | +44.5 | −23.2 | +121.6 | +55.4 [−83, +205] | −111 | 0.77 | 0.07 |
| V2-C2 · R1 0.75 %/2 h + T | 869 | 49 | +0.069 | +59.6 | −0.28 | 191 | 400 | +21.5 | +23.8 | +21.8 | **−7.7** | +132.8 | +66.6 [−80, +218] | −113 | 0.80 | 0.06 |
| V2-C2 · R3 | 607 | 36 | +0.015 | +9.4 | −0.28 | 156 | 354 | +56.4 | −52.4 | +17.8 | −12.4 | +82.6 | +16.4 | −175 | 0.55 | 0.13 |
| T0 · d | 24 | 46 | +0.035 | +0.8 | −0.19 | 10 | 0 | +5.9 | −1.2 | −2.7 | −1.2 | 0 | +7.9 [−9, +36] | −13 | 0.64 | 0.93 |
| T0 · R1 0.5 %/1 h | 18 | 44 | +0.256 | +4.6 | +0.14 | 7 | 84 | +5.9 | 0 | −0.6 | −0.7 | +3.8 | +11.6 [−3, +39] | −9 | 0.73 | **0.89** |
| T0 · R1 1.0 %/2 h + T | 28 | 43 | +0.210 | +5.9 | +0.17 | 9 | 33 | +4.3 | +3.4 | −0.6 | −1.2 | +5.1 | +12.9 [−2, +40] | −9 | 0.89 | 0.71 |

All 102 rows are in `_cands.csv`. Note **2026-H2 (the live half): every positive V2 candidate is negative there except the R1
4 h variants** (+48.8, +21.9), and those are negative in 2026-H1.

## Window classes: what each rule removes and what it gives up (Σ %)

| trigger · exit | clean slides: d → rule (given up) | false starts: d → rule (removed) | other true starts: d → rule |
|---|---|---|---|
| V2-C1 · T | +92.8 → +64.8 (−28) | −157.6 → −108.5 (+49) | −77.7 → −9.5 |
| V2-C1 · R1 0.5/1 h | +92.8 → +74.9 (**−18**) | −157.6 → −59.4 (**+98**) | −77.7 → +3.6 |
| V2-C1 · R1 0.5/1 h + T | +92.8 → +51.8 (−41) | −157.6 → −31.7 (+126) | −77.7 → +35.8 |
| V2-C1 · R1 0.5/4 h | +92.8 → +42.7 (−50) | −157.6 → −27.5 (+130) | −77.7 → +23.6 |
| V2-C1 · R2 1 %/1 h | +92.8 → −11.5 (**−104**) | −157.6 → −15.1 (+143) | −77.7 → +1.3 |
| V2-C1 · R3 | +92.8 → +75.6 (−17) | −157.6 → −68.4 (+89) | −77.7 → −24.2 |
| V2-C1 · R4 0.5 | +92.8 → +6.5 (−86) | −157.6 → −64.5 (+93) | −77.7 → −96.0 |
| V2-C2 · R1 0.5/1 h | +118.3 → +99.1 (−19) | −100.2 → −37.4 (+63) | −91.3 → −67.3 |
| V2-C2 · R1 0.75/2 h + T | +118.3 → +61.4 (−57) | −100.2 → −33.8 (+66) | −91.3 → +32.0 |

**R1 0.5 % / 1 h has the best trade-off:** it removes about 5 % of false-start loss for every 1 % of clean-slide gain it gives up.
R2 and R4 cut both kinds of window about equally, so they carry no information.

## Two-sided exit counterfactual vs exit d (same fills; untouched fills Δ = 0)

| trigger · exit | fills in both | exit changed | saved / cost | Σ Δ on changed | blocked fills (Σ removed) | slot-freed new fills (Σ) | untouched max \|Δ\| |
|---|---|---|---|---|---|---|---|
| V2-C1 · R1 0.5/1 h | 449 | 41 | 31 / 10 | **+1.8** | 249 (**+159.9**) | 0 | 0.0 |
| V2-C1 · R1 0.5/1 h + T | 448 | 250 | 167 / 83 | +56.9 | 250 (+154.3) | 84 (−12.7) | 0.0 |
| V2-C1 · R1 0.5/4 h | 276 | 162 | 105 / 57 | +8.2 | 422 (+173.1) | 0 | 0.0 |
| V2-C1 · R3 | 448 | 33 | 26 / 7 | +13.0 | 250 (+112.5) | 0 | 0.0 |
| V2-C1 · R4 0.5 | 493 | 254 | 166 / 86 | −44.4 | 205 (+32.9) | 0 | 0.0 |
| V2-C2 · R1 0.5/1 h | 643 | 41 | 22 / 17 | −16.6 | 261 (+84.3) | 0 | 0.0 |
| V2-C2 · R1 0.75/2 h + T | 750 | 416 | 271 / 143 | +103.3 | 154 (+37.5) | 119 (−8.0) | 0.0 |

Reading: **the R1 "exit" is really an entry gate.** Closing the open shorts at the bounce is roughly neutral (V2-C1 +1.8, V2-C2
−16.6). The money comes from not opening new shorts after BTC has bounced 0.5 % since ON. The trail (T) adds a real exit gain
(+57 on changed fills) by locking in +2 % runners. It costs about a third of the clean-slide tail.

## Walk-forward (leave-one-half-out, the deciding test)

**Over all 102 candidates (3 triggers × 34 exits), picking the best training ΔΣ vs T0:**

| held-out half | pick (trained on the other 3) | train ΔΣ | **held-out ΔΣ vs T0** |
|---|---|---|---|
| 2025-H1 | V2-C2 · R1 0.75 %/4 h + T | +93.6 | **−28.7** |
| 2025-H2 | V2-C2 · R3 | +69.4 | **−53.0** |
| 2026-H1 | V2-C1 · R1 0.5 %/4 h | +62.8 | **−17.1** |
| 2026-H2 | V2-C2 · R1 0.5 %/1 h + T | +78.3 | **−22.9** |

0 of 4 folds beat T0. Four different picks in four folds means the pick is unstable.

**Per trigger (choosing only that trigger's 34 exits):**

| trigger | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | folds > T0 |
|---|---|---|---|---|---|
| V2-C1 (picks: R1 0.5/4 h+T, R1 0.5/1 h+T, R1 0.5/4 h, R1 0.5/1 h+T) | +17.0 | +7.2 | **−17.1** | **−3.6** | 2/4, 0/2 in 2026 |
| V2-C2 (R1 0.75/4 h+T, R3, R1 0.75/2 h+T, R1 0.5/1 h+T) | −28.7 | −53.0 | +21.8 | −22.9 | 1/4 |
| T0 (R1 1.0/2 h+T, R1 0.5/1 h, R3+T, R1 1.0/2 h+T) | +11.6 | −0.6 | −2.7 | −0.9 | 1/4 |

V2-C1 is the only trigger with a stable rule family (R1 0.5 % in all 4 folds), but it fails in both 2026 halves.

## Shuffled-timing null (200 permutations of kill delays across the trigger's windows)

| trigger · exit | observed Σ | null mean | null 95th pct | rank |
|---|---|---|---|---|
| V2-C1 · R1 0.5 %/4 h (modal-tie pick) | +38.8 | −36.6 | +17.6 | **100 %** ✓ |
| V2-C1 · R1 0.5 %/1 h + T (best in-sample C1) | +55.9 | −31.8 | +28.7 | **99.5 %** ✓ |
| V2-C2 · R1 0.75 %/2 h + T (best in-sample C2) | +59.6 | −11.0 | +60.3 | 94 % ✗ |

The R1 timing on V2-C1 is information, not just "cutting earlier somewhere". That information is real but too small to clear the
T0 bar out of sample.

## Seed-2 consistency (12 finished seed-2 blocks; seed 1 restricted to the same blocks)

| trigger · exit | Σ seed 1 | Σ seed 2 |
|---|---|---|
| T0 · a (today) | +0.4 | −0.4 |
| V2-C1 · d | −22.5 | −17.2 |
| V2-C1 · R1 0.5/1 h | +24.5 | +15.3 |
| V2-C1 · R1 0.5/1 h + T | +15.0 | +11.0 |
| V2-C1 · R1 0.5/4 h | −3.9 | +29.5 |
| V2-C1 · R1 0.75/2 h + T | +19.6 | +18.2 |
| V2-C2 · d | +11.5 | +25.9 |
| V2-C2 · R1 0.75/2 h + T | +25.5 | +2.8 |

The direction replicates on V2-C1: R1 beats d by +26 to +47 on both seeds. Seed 2 is not complete, so it cannot rescue the
walk-forward.

## ② master and ① live, priced under the rules (rules measured from today's T0 ON, which is the live trigger)

| fill | live result | R1 0.5 % (1 h or 4 h) | R3 | R2 1 %/1 h | R1 ≥ 0.75 % |
|---|---|---|---|---|---|
| 10-07 UNI 14:49 (master + live) | +0.68 % | **blocked** (kill 13:57, BTC bounced 0.5 % 7 min after ON) | blocked (14:00) | closed 14:50: −0.09 % | no kill: as live |
| 10-07 XLM 14:49 (master + live) | +0.40 % | **blocked** | blocked | closed 14:50: −0.13 % | as live |
| 10-22 SOL 17:16 @103.45 | −$20.67 | closed at kill 17:20 @103.74: **−0.38 %** (≈ −$8.6 at $2,250 notional) | no kill | no kill | as live |
| 10-22 APT 18:04 | −$19.07 | **blocked** (kill 17:20) | no kill | no kill | as live |
| 10-22 HBAR 18:10 | −$29.01 | **blocked** | no kill | no kill | as live |
| **total, 5 live fills** | **−$61.29** | **≈ −$8.6** (gives up +$7.46 on 10-07, saves about $60 on 10-22) | −$68.75 | ≈ −$73.7 | −$61.29 |

Master (2 fills): +1.08 % → 0 under R1 0.5 % and R3, → −0.22 % under R2. Five fills on two days cannot decide anything. They do
match the replay mechanism: once BTC bounced after ON, the late entries were the losers.

## 10-22 and the last 48 h (10-21 → 10-23 17:58 UTC)

- **The V2-C1 window ran from 10-22 02:00 to 10-23 05:00.** My rebuild from fresh public klines matches `v2_states.pkl` on 99.7 %
  of the 1,940 overlapping 5 m bars. The replay's 17:35 OFF is only its data end.
- **BTC moved** from 83,357 at ON to a 80,889 low at 16:53 (−2.96 %), and was −2.15 % at the true OFF.
- **No R1, R3 or R2 0.5 % rule fires all window long,** so the clean slide is left to run. Only R2 1 % fires (03:00 / 04:00),
  which would have killed the best slide of the month in its first hour.
- **No new V2-C1 window has opened since 10-23 05:00.**
- **The replay's own V2 fills on 10-22 came late:** XMR 13:52, PENGU 15:32, STRK 16:43, QNT 16:45. Σ under d is −0.03 (to the
  17:35 data end). Under d + T it is −1.18, because XMR is trailed at +0.99 instead of +2.14.
- **The 10-22 problem is the late entries, not the exit.** The engine's momentum-short signals did not fire while the slide was
  young (see BEARRUN_V2_2026-10-22 §2).
- **On the T0 window (ON 16:35),** R1 0.5 % kills at 17:20, after the 16:53 low and the bounce. That blocks APT and HBAR.

## Ship bar (pre-registered) — scorecard for the best candidate (V2-C1 · R1 0.5 %/1 h + T)

| bar | result |
|---|---|
| held-out ΔΣ vs T0 > 0 in ≥ 3 of 4 folds and both 2026 folds | ✗ (all-candidate WF: 0/4; per-trigger WF: 2/4, 0/2 in 2026) |
| window-bootstrap CI lower bound after 50 % haircut > 0 | ✗ (−83) |
| ≥ 8 windows, no window ≥ 50 % of the gain | ✓ (136 windows, max share 9 %) |
| shuffled-timing null above 95th pct | ✓ (99.5 %) |

**→ NO-SHIP.**

## Recommendation

1. **Do not build V2 as a live trigger, with or without these exits.** No candidate beats T0 out of sample, and 2026-H2 (the live
   half) is negative for the best one.
2. **Keep R1 (BTC +0.5 % bounce off the post-ON low ⇒ stop new sleeve shorts until the next ON) as a pre-registered observe
   candidate.**
   - It is the first post-ON variable to pass a timing null, and it replicates on seed 2.
   - It also explains the live 10-22 losses (APT and HBAR came after the bounce).
   - Frozen as is: B = 0.5 %, H = 1 h, measured from the trigger's ON, BTC 1 m close vs its lowest close since ON. Never re-fit
     on the data that failed it.
3. **Sizing: KEEP the 10-23 cut (`bearrun` invest 0.2 × 5×, ≈ $500 notional per short). Do not revert to 1.0 × 5×.**
   On today's ≈ $1,870 account (window $ = Σ % of the window's shorts × notional; seed-1 replay, 1.8 years):

   | sleeve | size | windows | window mean | $ per window | worst window | worst as % of account | 5th-pct window | $ per year |
   |---|---|---|---|---|---|---|---|---|
   | **T0 · exit a (live today)** | 0.2 × 5× | 10 | −0.70 % | **−$3.5** | −$31 | 1.7 % | −$21 | −$19 |
   | T0 · exit a | 1.0 × 5× | 10 | −0.70 % | −$15.8 | **−$140** | **7.5 %** | −$95 | −$87 |
   | V2-C1 · d | 0.2 × 5× | 185 | −0.77 % | −$3.9 | −$55 | 2.9 % | −$42 | −$394 |
   | V2-C1 · R1 0.5/1 h + T (best, NOT passed) | 0.2 × 5× | 136 | +0.41 % | +$2.1 | −$46 | 2.5 % | −$33 | +$154 (in-sample) |
   | V2-C1 · R1 0.5/1 h + T | 1.0 × 5× | 136 | +0.41 % | +$9.3 | −$207 | 11.1 % | −$147 | +$695 (in-sample; ≈ +$350 after haircut) |

   - **Why keep it small:** today's sleeve has negative window expectancy in the replay (−0.70 %/window, Σ −7) and live (5 fills,
     −$61).
   - **At 1.0 × 5×** one bad window costs 7.5 % of the account for no proven edge.
   - **At 0.2 × 5×** the same window is −$31 (1.7 %), and the yearly bleed is about −$19. That is a cheap price for keeping live
     fills flowing into the gates.
   - **Going lower (0.1 or the 0.05 probe)** is defensible but saves only about $10/yr. Keep 0.2 so live fills stay comparable to
     FLATBASE.
   - **Frozen step-up gate:** go back above 0.2 only when a Bear-Run trigger + exit passes this study's full ship bar on the
     complete seed 1–3 replay, AND its live/forward shadow has ≥ 20 windows over ≥ 6 weeks with Σ > 0 and window-bootstrap
     P(Σ > 0) ≥ 0.90. Then step 0.2 → 0.5 → 1.0, each step after +10 windows with Σ > 0, and revert to 0.2 if the first 10
     windows at a new size have Σ ≤ 0.
4. **Nothing changes in the config.** No V2 config is proposed (it did not pass). The R1 shadow below needs code, so it is a
   proposal for the operator to approve, not a ship.

## Scout: what a live shadow should record (R1 entry-gate shadow; observe-only, no orders)

- **Per T0 window and per V2-C1 window** (V2-C1 computed as in `study_bearrun_fsexit_last48.py`): ON time, BTC p0, running
  low, time of the first +0.5 % bounce (R1 kill), and whether it came within 1 h.
- **Per Bear-Run fill (live and shadow):** the stamp `entry_bear_r1_bounced` (was the R1 bounce already true at entry?), plus
  minutes since ON. A real Order column (D12) is needed so the CSV carries it.
- **Read at each batch, window units:**
  - Σ % of fills entered after the bounce vs before it;
  - the false-start vs clean-slide class of each window (F1 definition) once it resolves at +24 h;
  - Σ given up on clean slides vs Σ removed on false starts.
- **Frozen gate (to promote R1 to an armed entry gate on T0):**
  1. ≥ 15 post-bounce fills across ≥ 8 windows;
  2. their 1× avg < 0 with window-clustered bootstrap P ≥ 0.95;
  3. no window ≥ 50 % of the loss;
  4. the pre-bounce fills of the same windows are not worse.
- **Kill the shadow** if after 20 windows the post-bounce fills' Σ ≥ 0.

## What I could not test

- **Seeds 2–3 are incomplete** (12 of 93 seed-2 blocks finished; seed 3 not started), so the walk-forward is seed 1 only.
- **10-22 replay fills are priced to the replay's 17:35 data end,** not the true 10-23 05:00 OFF. Pair klines beyond 10-22 23:09
  were not fetched; the effect is ≤ 4 fills.
- **Live 10-22 fills have no export row.** % and $ for them come from context ($ figures, entry times and prices in
  BEARRUN_EXHAUSTION_GUARD); the SOL rule exit is priced at the assumed ≈ $2,250 notional.
- **Fees are 0.10 % round trip with no extra slippage** (same as exit d). The trail exits at the exact trail level, so a live
  trail would slip about 0.05–0.10 % more on 1 m polling.
- **The E1 entry arm (sub-filters lifted) was not re-run** under the rules. E0 is primary; E1 was worse under every exit in the
  replay.
- **R4 is a fixed point of price → select → kill** (converged in 2 passes). The real engine would apply it sequentially; the
  difference should be small, and R4 failed anyway.
