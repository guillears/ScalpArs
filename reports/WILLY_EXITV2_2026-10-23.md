# WILLY exit V2 — spike-high target + arm/floor, and a −2.5 stop when BTC is falling (2026-10-23)

Research only. No code, config or git change. Pre-registration (written before any V2 result): `reports/WILLY_EXITV2_2026-10-23_PREREG.md`.
Extends `reports/WILLY_HIGHTARGET_2026-10-23.md` (same fill sets, spike high H, BTC side labels, walker).
Scripts: `scripts/study_willy_exitv2.py` (build) · `scripts/study_willy_exitv2_report.py` (tables) · `scripts/study_willy_exitv2_sens.py`
(intrabar sensitivity). Data: `reports/WILLY_EXITV2_2026-10-23_{bt,real,bt_sens}.csv`.

## Verdict in plain English

| component | ③ backtest 2026 Δ/fill (pooled H1 + H2) | pre-registered bar (≥ 0) | stricter study bar |
|---|---|---|---|
| **A** = TP at spike high × 0.99 + arm at +1.25 + floor F | **+0.022** [−0.056, +0.103] at walk-forward **F = 0.5** | **SHIP** | fail (CI crosses 0; 3/4 halves) |
| **B** = −2.5 stop when BTC is falling | **−0.003** [−0.071, +0.062] | **NO SHIP** (point estimate below 0) | fail |
| AB (both, F 0.5) — information | +0.013 [−0.086, +0.116] | – | fail |

- **Ship A at F = 0.5, not F = 1.0.** The anchored walk-forward picks F 0.5 for both 2026 halves (and LOHO picks 0.5 for 3 of 4
  halves). F 1.0 — the floor the operator had in mind — is the worst floor in 2026 (−0.030/fill): it sells too many runners that dip
  just under +1.0 and then reach the target.
- **A's gain is small and fragile.** +0.022 %/fill is ≈ +0.011 to +0.015 after the 30–50 % in-sample haircut. On 1m bars the order
  inside the arming minute is unknown; the result swings from −0.17 (assume the high always came first) to +0.038 (red bar = high
  first, the usual OHLC heuristic). The pre-registered convention and the OHLC heuristic both clear the bar; only the extreme
  "high first on every bar" bound does not (see §4). The real fills, which are on ticks and need no assumption, split: ② master
  +0.13/fill, ① live −0.16/fill (7 armed live fills).
- **B does not ship by the pre-registered bar** — it is a coin flip in 2026: it saves +232 points on 156 BTC-falling fills that
  today stop at −4, and costs −237 points on 84 falling fills that today end as TP / late TP / cap but dip to −2.5 first. 2026-H1
  −0.026, 2026-H2 +0.040, 2025 +0.049. It is the same cell as the earlier study's C6 (T 1.25 / S 2.5 on falling), and reproduces it
  exactly (−0.003). On the real fills it looks great (② +0.17, ① +0.25) only because none of the 13 falling real fills dipped to
  −2.5 and then recovered — 4 saved, 0 cut, on 4 days. That is not evidence the backtest's 84 cut winners won't happen live.
- **Master per batch, before → after (AB):** ALL +0.263 → **+0.611** at F 1.0 (operator expected ≈ +0.64 — reproduced within
  0.03; the gap is the 0.03 floor slippage plus tick floors that print below +1.0, e.g. RSR +0.59, ALT +0.75) and → **+0.560 at the
  chosen F 0.5**. Of that, A alone (F 0.5) gives +0.393 and B alone +0.430 — B carries most of the master gain, but B fails in the
  2026 backtest.

**What this means for the build:** A clears the operator-approved bar (point estimate ≥ 0) at F 0.5 and should ship with a
pre-committed tight revert gate and its scout line (proposal in §6). B does not ship; keep it as the existing observe line.

## 1. Data and calibration

- ③ backtest: 4,129 fills (yr6a 2025 + yr6 2026, seeds 1–3, trimmed; 1m paths) — **1,393 unique (pair, time)**: the three seeds
  share most fills, so seeds are reported pooled and per seed (per-seed Δ are within 0.001 of each other — they are not independent
  evidence). 2025: 2,435 · 2026-H1: 1,103 · 2026-H2: 591.
- ② master: 36 fills B19–B23 (tick paths, ICNT 10-16 public 1m). ① live: B22–B23 live fills + ARK 10-23 (public 1m, provisional), 18.
- **Parity:** the V2 walker's TODAY column equals the study's TODAY on every fill (max |Δ| < 1e-9); ② today mean +0.263 = stack_pct
  +0.263. No unresolved fills.
- **Asserts (all pass):** unarmed fills Δ ≡ 0 (A vs today, AB vs B); fills with T = +1.25 Δ ≡ 0; B Δ ≡ 0 outside the BTC-falling
  fills that hit −2.5.


## 2. F table — ③ Δ vs today per half (A, stop −4)

| F | 2025-H1 | 2025-H2 | 2026-H1 | 2026-H2 | 2026 pooled [95 % CI] | all |
|---|---|---|---|---|---|---|
| 1 | -0.032 | +0.006 | -0.032 | -0.029 | -0.030 [-0.08, +0.03] | -0.017 |
| 0.75 | -0.019 | +0.006 | -0.035 | +0.022 | -0.015 [-0.08, +0.06] | -0.008 |
| 0.5 | -0.020 | +0.030 | +0.017 | +0.030 | +0.022 [-0.06, +0.10] | +0.015 |
| 0.25 | -0.012 | +0.001 | +0.015 | +0.022 | +0.018 [-0.07, +0.10] | +0.005 |

| held-out half | anchored WF pick (trained on earlier halves) | LOHO pick (other 3 halves) |
|---|---|---|
| 2025-H1 | – | 0.5 |
| 2025-H2 | 0.25 | 0.25 |
| 2026-H1 | 0.5 | 0.5 |
| 2026-H2 | 0.5 | 0.5 |

**Deployment F (anchored WF trained on 2025-H1..2026-H1): 0.5** · best in-sample on all 4 halves (NOT used): 0.5

## 3. Ship bar (pre-registered: ③ 2026 pooled Δ point estimate ≥ 0)

| component | 2026-H1 Δ | 2026-H2 Δ | 2026 pooled Δ [95 % CI] | per seed (1 / 2 / 3) | verdict | study bar (CI lo > 0 ∧ ≥ 3/4 halves) |
|---|---|---|---|---|---|---|
| A (WF F 0.5 / 0.5) | +0.017 | +0.030 | +0.022 [-0.056, +0.103] | +0.022 / +0.022 / +0.021 | SHIP | fail (3/4 LOHO halves, CI lo -0.056) |
| AB (WF F 0.5 / 0.5) | -0.018 | +0.071 | +0.013 [-0.086, +0.116] | +0.013 / +0.013 / +0.012 | SHIP | fail (3/4 LOHO halves, CI lo -0.086) |
| B (−2.5 falling) | -0.026 | +0.040 | -0.003 [-0.071, +0.062] | -0.003 / -0.003 / -0.003 | NO | fail (3/4 halves, CI lo -0.071) |

B per half: 2025-H1 +0.068 · 2025-H2 +0.037 · 2026-H1 -0.026 · 2026-H2 +0.040

## 4. Intrabar sensitivity (③ only — 1m bars; real fills are on ticks and need no assumption)

The arming 1m bar both rises through +1.25 and often has a low under F (it usually opened lower). Pre-registered convention: the
arming bar never fires the floor (the next bar can, at min(F, its open) − 0.03). Two alternatives on the same fills (A, stop −4):
- **pessimistic bound** — every arming bar whose low ≤ F went high first, then down: floor at F − 0.03 on that bar;
- **OHLC heuristic** — only a red arming bar (close < open) went high first; a green one went low first (pre-reg rule).

```
F 1: arming bars re-booked 895 · 2026 Δ pre-reg -0.030 → pessimistic -0.094 [-0.108, -0.078] · 2025 -0.052
      OHLC heuristic (red arming bar = high first): re-booked 357 · 2026 Δ +0.009 [-0.038, +0.066] · per half +0.024 / +0.067 / +0.019 / -0.009
F 0.75: arming bars re-booked 754 · 2026 Δ pre-reg -0.015 → pessimistic -0.158 [-0.189, -0.123] · 2025 -0.127
      OHLC heuristic (red arming bar = high first): re-booked 282 · 2026 Δ +0.014 [-0.047, +0.086] · per half +0.026 / +0.050 / +0.001 / +0.038
F 0.5: arming bars re-booked 591 · 2026 Δ pre-reg +0.022 → pessimistic -0.167 [-0.220, -0.109] · 2025 -0.151
      OHLC heuristic (red arming bar = high first): re-booked 222 · 2026 Δ +0.038 [-0.038, +0.119] · per half +0.011 / +0.062 / +0.037 / +0.040
F 0.25: arming bars re-booked 444 · 2026 Δ pre-reg +0.018 → pessimistic -0.156 [-0.224, -0.081] · 2025 -0.163
      OHLC heuristic (red arming bar = high first): re-booked 171 · 2026 Δ +0.031 [-0.052, +0.115] · per half +0.011 / +0.028 / +0.032 / +0.028
```

Reading: the pessimistic bound is not realistic (a green bar that closes near its high almost always made its low first), but it
shows A's backtest Δ lives inside the intrabar uncertainty. The OHLC heuristic is *better* for A than the pre-registered convention
(+0.038 at F 0.5, positive in all four halves), because the pre-registered rule delays a red-bar floor to the next bar's open,
which is often well below F. Both clear the bar; only tick-level forward data settles it.

## 5. ③ backtest per period (all variants)

**③ 2025 (284 days)** — today: N · WR · avg · Σ · worst = 2435 · 72 % · -0.308 · -749.3 · -40.27

| variant | N · WR · avg · Σ · worst | Δ/fill [95 % CI] |
|---|---|---|
| B | 2435 · 68 % · -0.259 · -630.8 · -40.27 | +0.049 [-0.01, +0.10] |
| A_F1 | 2435 · 68 % · -0.316 · -769.4 · -40.27 | -0.008 [-0.06, +0.05] |
| A_F0.75 | 2435 · 68 % · -0.311 · -757.2 · -40.27 | -0.003 [-0.06, +0.06] |
| A_F0.5 | 2435 · 68 % · -0.297 · -722.5 · -40.27 | +0.011 [-0.06, +0.09] |
| A_F0.25 | 2435 · 68 % · -0.312 · -759.9 · -40.27 | -0.004 [-0.08, +0.08] |
| AB_F1 | 2435 · 65 % · -0.266 · -647.4 · -40.27 | +0.042 [-0.04, +0.12] |
| AB_F0.75 | 2435 · 65 % · -0.259 · -631.3 · -40.27 | +0.048 [-0.03, +0.13] |
| AB_F0.5 | 2435 · 65 % · -0.245 · -596.3 · -40.27 | +0.063 [-0.03, +0.15] |
| AB_F0.25 | 2435 · 65 % · -0.258 · -628.2 · -40.27 | +0.050 [-0.04, +0.15] |

**③ 2026-H1 (153 days)** — today: N · WR · avg · Σ · worst = 1103 · 80 % · +0.174 · +192.2 · -4.00

| variant | N · WR · avg · Σ · worst | Δ/fill [95 % CI] |
|---|---|---|
| B | 1103 · 75 % · +0.149 · +163.9 · -4.00 | -0.026 [-0.11, +0.06] |
| A_F1 | 1103 · 77 % · +0.143 · +157.4 · -4.00 | -0.032 [-0.10, +0.05] |
| A_F0.75 | 1103 · 77 % · +0.139 · +153.6 · -4.00 | -0.035 [-0.11, +0.05] |
| A_F0.5 | 1103 · 77 % · +0.191 · +210.9 · -4.00 | +0.017 [-0.08, +0.13] |
| A_F0.25 | 1103 · 77 % · +0.190 · +209.1 · -4.00 | +0.015 [-0.09, +0.13] |
| AB_F1 | 1103 · 73 % · +0.116 · +128.3 · -4.00 | -0.058 [-0.17, +0.06] |
| AB_F0.75 | 1103 · 73 % · +0.115 · +126.7 · -4.00 | -0.059 [-0.18, +0.06] |
| AB_F0.5 | 1103 · 73 % · +0.156 · +172.0 · -4.00 | -0.018 [-0.15, +0.11] |
| AB_F0.25 | 1103 · 73 % · +0.156 · +172.4 · -4.00 | -0.018 [-0.15, +0.12] |

**③ 2026-H2 (83 days)** — today: N · WR · avg · Σ · worst = 591 · 79 % · +0.020 · +11.9 · -4.00

| variant | N · WR · avg · Σ · worst | Δ/fill [95 % CI] |
|---|---|---|
| B | 591 · 76 % · +0.060 · +35.3 · -4.00 | +0.040 [-0.08, +0.15] |
| A_F1 | 591 · 78 % · -0.008 · -5.0 · -4.00 | -0.029 [-0.09, +0.05] |
| A_F0.75 | 591 · 78 % · +0.042 · +24.9 · -4.00 | +0.022 [-0.08, +0.15] |
| A_F0.5 | 591 · 78 % · +0.050 · +29.8 · -4.00 | +0.030 [-0.08, +0.16] |
| A_F0.25 | 591 · 78 % · +0.042 · +25.1 · -4.00 | +0.022 [-0.10, +0.16] |
| AB_F1 | 591 · 75 % · +0.037 · +21.9 · -4.00 | +0.017 [-0.12, +0.15] |
| AB_F0.75 | 591 · 75 % · +0.083 · +49.0 · -4.00 | +0.063 [-0.09, +0.22] |
| AB_F0.5 | 591 · 75 % · +0.091 · +53.8 · -4.00 | +0.071 [-0.09, +0.23] |
| AB_F0.25 | 591 · 75 % · +0.084 · +49.8 · -4.00 | +0.064 [-0.10, +0.23] |

**③ 2026 pooled** — today: N · WR · avg · Σ · worst = 1694 · 80 % · +0.120 · +204.1 · -4.00

| variant | N · WR · avg · Σ · worst | Δ/fill [95 % CI] |
|---|---|---|
| B | 1694 · 75 % · +0.118 · +199.2 · -4.00 | -0.003 [-0.07, +0.06] |
| A_F1 | 1694 · 78 % · +0.090 · +152.4 · -4.00 | -0.030 [-0.08, +0.03] |
| A_F0.75 | 1694 · 78 % · +0.105 · +178.6 · -4.00 | -0.015 [-0.08, +0.06] |
| A_F0.5 | 1694 · 78 % · +0.142 · +240.7 · -4.00 | +0.022 [-0.06, +0.10] |
| A_F0.25 | 1694 · 78 % · +0.138 · +234.2 · -4.00 | +0.018 [-0.07, +0.10] |
| AB_F1 | 1694 · 74 % · +0.089 · +150.2 · -4.00 | -0.032 [-0.12, +0.06] |
| AB_F0.75 | 1694 · 74 % · +0.104 · +175.7 · -4.00 | -0.017 [-0.11, +0.08] |
| AB_F0.5 | 1694 · 74 % · +0.133 · +225.8 · -4.00 | +0.013 [-0.09, +0.12] |
| AB_F0.25 | 1694 · 74 % · +0.131 · +222.2 · -4.00 | +0.011 [-0.09, +0.12] |

## 5b. ③ per BTC side (Δ/fill [CI]) at deployment F 0.5

| period | side | N | today avg | A | B | AB |
|---|---|---|---|---|---|---|
| 2025 | not falling | 1437 | -0.156 | +0.045 [-0.05, +0.14] | +0.000 [+0.00, +0.00] | +0.045 [-0.05, +0.14] |
| 2025 | falling | 998 | -0.526 | -0.038 [-0.14, +0.06] | +0.119 [-0.01, +0.25] | +0.089 [-0.06, +0.24] |
| 2026-H1 | not falling | 542 | +0.167 | -0.022 [-0.13, +0.09] | +0.000 [+0.00, +0.00] | -0.022 [-0.13, +0.09] |
| 2026-H1 | falling | 561 | +0.181 | +0.054 [-0.11, +0.23] | -0.050 [-0.21, +0.11] | -0.015 [-0.23, +0.22] |
| 2026-H2 | not falling | 364 | +0.235 | +0.123 [-0.04, +0.31] | +0.000 [+0.00, +0.00] | +0.123 [-0.04, +0.31] |
| 2026-H2 | falling | 227 | -0.325 | -0.118 [-0.22, -0.01] | +0.103 [-0.19, +0.39] | -0.012 [-0.32, +0.29] |

## 5c. Affected-cohort accounting

**A: armed fills with T > 1.25 (the only fills A can change)** — per F: N armed · outcome mix · Σ Δ · mean Δ on armed

| block | F | armed N | TP at T | floor | cap | Σ Δ | mean Δ armed |
|---|---|---|---|---|---|---|---|
| ③ 2025 | 1 | 929 | 265 | 664 | 0 | -20.0 | -0.022 |
| ③ 2025 | 0.75 | 929 | 313 | 616 | 0 | -7.8 | -0.008 |
| ③ 2025 | 0.5 | 929 | 355 | 571 | 3 | +26.8 | +0.029 |
| ③ 2025 | 0.25 | 929 | 382 | 544 | 3 | -10.5 | -0.011 |
| ③ 2026 | 1 | 624 | 167 | 457 | 0 | -51.7 | -0.083 |
| ③ 2026 | 0.75 | 624 | 215 | 409 | 0 | -25.5 | -0.041 |
| ③ 2026 | 0.5 | 624 | 263 | 361 | 0 | +36.6 | +0.059 |
| ③ 2026 | 0.25 | 624 | 284 | 331 | 9 | +30.1 | +0.048 |
| ② master | 1 | 17 | 7 | 10 | 0 | +6.5 | +0.383 |
| ② master | 0.75 | 17 | 7 | 10 | 0 | +4.5 | +0.264 |
| ② master | 0.5 | 17 | 9 | 8 | 0 | +4.7 | +0.275 |
| ② master | 0.25 | 17 | 11 | 6 | 0 | +11.9 | +0.703 |
| ① live+ARK | 1 | 7 | 2 | 5 | 0 | -1.6 | -0.224 |
| ① live+ARK | 0.75 | 7 | 2 | 5 | 0 | -2.5 | -0.350 |
| ① live+ARK | 0.5 | 7 | 3 | 4 | 0 | -2.9 | -0.408 |
| ① live+ARK | 0.25 | 7 | 3 | 4 | 0 | -3.9 | -0.551 |

**B: BTC-falling fills stopped at −2.5** — saved (today stopped at −4) vs cut (today not stopped)

| block | falling N | saved N · Σ Δ | cut N · Σ Δ (today's outcome mix) | net Σ Δ | Δ/fill all fills [CI] |
|---|---|---|---|---|---|
| ③ 2025 | 998 | 296 · +415.5 | 104 · -296.9 ({'TP': 56, 'LATE_TP': 27, 'CAP': 21}) | +118.5 | +0.049 [-0.01, +0.10] |
| ③ 2026-H1 | 561 | 99 · +148.5 | 63 · -176.8 ({'TP': 33, 'LATE_TP': 18, 'CAP': 12}) | -28.3 | -0.026 [-0.11, +0.06] |
| ③ 2026-H2 | 227 | 57 · +83.8 | 21 · -60.4 ({'TP': 12, 'LATE_TP': 6, 'CAP': 3}) | +23.4 | +0.040 [-0.08, +0.15] |
| ③ 2026 | 788 | 156 · +232.3 | 84 · -237.2 ({'TP': 45, 'LATE_TP': 24, 'CAP': 15}) | -4.9 | -0.003 [-0.07, +0.06] |
| ② master | 13 | 4 · +6.0 | 0 · +0.0 ({}) | +6.0 | +0.167 [+0.00, +0.36] |
| ① live+ARK | 12 | 3 · +4.5 | 0 · +0.0 ({}) | +4.5 | +0.250 [+0.00, +0.52] |

## 5d. ② master (36, B19–B23) and ① live


**② master (36)** — today: N · WR · avg · Σ · worst = 36 · 83 % · +0.263 · +9.5 · -4.06

| variant | N · WR · avg · Σ · worst | Δ/fill [95 % CI] |
|---|---|---|
| B | 36 · 83 % · +0.430 · +15.5 · -4.06 | +0.167 [+0.00, +0.36] |
| A_F1 | 36 · 83 % · +0.444 · +16.0 · -4.06 | +0.181 [-0.05, +0.43] |
| A_F0.75 | 36 · 83 % · +0.387 · +13.9 · -4.06 | +0.125 [-0.10, +0.37] |
| A_F0.5 | 36 · 83 % · +0.393 · +14.1 · -4.06 | +0.130 [-0.14, +0.41] |
| A_F0.25 | 36 · 83 % · +0.595 · +21.4 · -4.06 | +0.332 [-0.16, +0.87] |
| AB_F1 | 36 · 83 % · +0.611 · +22.0 · -4.06 | +0.348 [+0.11, +0.57] |
| AB_F0.75 | 36 · 83 % · +0.554 · +20.0 · -4.06 | +0.292 [+0.06, +0.51] |
| AB_F0.5 | 36 · 83 % · +0.560 · +20.2 · -4.06 | +0.297 [+0.03, +0.54] |
| AB_F0.25 | 36 · 83 % · +0.762 · +27.4 · -4.06 | +0.499 [+0.02, +1.05] |

**① live incl. ARK 10-23 provisional (18)** — today: N · WR · avg · Σ · worst = 18 · 78 % · -0.028 · -0.5 · -4.06

| variant | N · WR · avg · Σ · worst | Δ/fill [95 % CI] |
|---|---|---|
| B | 18 · 78 % · +0.222 · +4.0 · -4.06 | +0.250 [+0.00, +0.52] |
| A_F1 | 18 · 78 % · -0.115 · -2.1 · -4.06 | -0.087 [-0.15, -0.04] |
| A_F0.75 | 18 · 78 % · -0.164 · -3.0 · -4.06 | -0.136 [-0.21, -0.06] |
| A_F0.5 | 18 · 78 % · -0.187 · -3.4 · -4.06 | -0.159 [-0.31, -0.08] |
| A_F0.25 | 18 · 78 % · -0.242 · -4.4 · -4.06 | -0.214 [-0.41, -0.10] |
| AB_F1 | 18 · 78 % · +0.135 · +2.4 · -4.06 | +0.163 [-0.13, +0.44] |
| AB_F0.75 | 18 · 78 % · +0.086 · +1.5 · -4.06 | +0.114 [-0.15, +0.39] |
| AB_F0.5 | 18 · 78 % · +0.064 · +1.1 · -4.06 | +0.092 [-0.24, +0.33] |
| AB_F0.25 | 18 · 78 % · +0.008 · +0.1 · -4.06 | +0.036 [-0.32, +0.28] |

**② master per batch — before (today) → after (AB at F 0.5; also AB at F 1.0, A, B)**

| batch | N | falling | today | AB_F0.5 | AB_F1 | A_F0.5 | B |
|---|---|---|---|---|---|---|---|
| B19 | 14 | 0 | +0.732 | +1.187 | +1.207 | +1.187 | +0.732 |
| B20 | 3 | 2 | -0.510 | +0.117 | +0.158 | -0.386 | -0.007 |
| B21 | 2 | 0 | +1.250 | +1.256 | +1.505 | +1.256 | +1.250 |
| B22 | 8 | 5 | -0.195 | -0.094 | -0.079 | -0.280 | -0.008 |
| B23 | 9 | 6 | -0.022 | +0.158 | +0.250 | -0.177 | +0.313 |
| ALL | 36 | 13 | +0.263 | +0.560 | +0.611 | +0.393 | +0.430 |

per-batch mean of batch means (today → AB_F1): +0.251 → +0.608

**② fill-level (AB at F 1.0 / deployment F)**

| opened | pair | era | btc | T | today | AB_F1 | why | AB_F0.5 | why |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-09T08:45:09 | KAIAUSDT | B19 | not falling | 1.97 | +1.25 (TP) | +0.96 | FLOOR | +0.45 | FLOOR |
| 2026-10-09T09:05:08 | WUSDT | B19 | not falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-09T14:25:07 | API3USDT | B19 | not falling | 1.25 | +0.25 (LATE_TP) | +0.25 | LATE_TP | +0.25 | LATE_TP |
| 2026-10-09T16:35:08 | ZKUSDT | B19 | not falling | 1.25 | +0.25 (LATE_TP) | +0.25 | LATE_TP | +0.25 | LATE_TP |
| 2026-10-10T04:30:07 | CAPUSDT | B19 | not falling | 3.11 | +1.25 (TP) | +0.97 | FLOOR | +0.46 | FLOOR |
| 2026-10-10T05:20:08 | MINAUSDT | B19 | not falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-10T06:45:07 | LPTUSDT | B19 | not falling | 7.97 | +1.25 (TP) | +0.95 | FLOOR | +0.44 | FLOOR |
| 2026-10-10T11:35:12 | ORCAUSDT | B19 | not falling | 2.18 | +1.25 (TP) | +0.94 | FLOOR | +2.18 | TP_T |
| 2026-10-10T13:35:12 | CFXUSDT | B19 | not falling | 6.31 | +1.25 (TP) | +6.31 | TP_T | +6.31 | TP_T |
| 2026-10-10T18:40:09 | CHIPUSDT | B19 | not falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-11T11:25:09 | SENTUSDT | B19 | not falling | 2.67 | +1.25 (TP) | +2.67 | TP_T | +2.67 | TP_T |
| 2026-10-11T15:25:07 | SUSDT | B19 | not falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-11T16:05:07 | BATUSDT | B19 | not falling | 2.02 | -4.00 (STOP) | -4.00 | STOP | -4.00 | STOP |
| 2026-10-11T18:10:08 | WUSDT | B19 | not falling | 2.60 | +1.25 (TP) | +2.60 | TP_T | +2.60 | TP_T |
| 2026-10-12T05:15:11 | RSRUSDT | B20 | not falling | 6.57 | +1.25 (TP) | +0.59 | FLOOR | +0.46 | FLOOR |
| 2026-10-12T10:40:09 | OGNUSDT | B20 | falling | 1.25 | -4.03 (STOP) | -2.52 | STOP | -2.52 | STOP |
| 2026-10-12T12:55:08 | CAPUSDT | B20 | falling | 2.41 | +1.25 (TP) | +2.41 | TP_T | +2.41 | TP_T |
| 2026-10-13T14:35:12 | ZKUSDT | B21 | not falling | 2.04 | +1.25 (TP) | +2.04 | TP_T | +2.04 | TP_T |
| 2026-10-14T03:55:07 | BTWUSDT | B21 | not falling | 5.06 | +1.25 (TP) | +0.97 | FLOOR | +0.47 | FLOOR |
| 2026-10-15T01:45:12 | DUSKUSDT | B22 | falling | 1.25 | +0.25 (LATE_TP) | +0.25 | LATE_TP | +0.25 | LATE_TP |
| 2026-10-15T04:40:13 | ZKUSDT | B22 | falling | 2.73 | +1.25 (TP) | +0.97 | FLOOR | +0.47 | FLOOR |
| 2026-10-15T09:30:12 | BLURUSDT | B22 | falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-15T10:20:11 | STRKUSDT | B22 | falling | 1.25 | -4.00 (STOP) | -2.51 | STOP | -2.51 | STOP |
| 2026-10-15T13:30:11 | RLCUSDT | B22 | falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-15T15:30:10 | PYTHUSDT | B22 | not falling | 1.35 | +1.25 (TP) | +0.96 | FLOOR | +1.35 | TP_T |
| 2026-10-15T16:00:10 | BATUSDT | B22 | not falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-16T00:05:09 | ICNTUSDT | B22 | not falling | 1.54 | -4.06 (STOP) | -4.06 | STOP | -4.06 | STOP |
| 2026-10-17T06:30:09 | ZILUSDT | B23 | falling | 1.72 | +1.25 (TP) | +0.96 | FLOOR | +0.43 | FLOOR |
| 2026-10-17T06:40:11 | DUSKUSDT | B23 | falling | 1.74 | -4.02 (STOP) | -2.50 | STOP | -2.50 | STOP |
| 2026-10-17T16:55:09 | BATUSDT | B23 | falling | 1.25 | +0.32 (LATE_TP) | +0.32 | LATE_TP | +0.32 | LATE_TP |
| 2026-10-17T19:40:10 | ALTUSDT | B23 | falling | 1.25 | -4.00 (STOP) | -2.50 | STOP | -2.50 | STOP |
| 2026-10-19T18:45:09 | PHAUSDT | B23 | falling | 1.29 | +1.25 (TP) | +1.29 | TP_T | +1.29 | TP_T |
| 2026-10-19T20:45:12 | BATUSDT | B23 | falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-20T04:10:09 | ALTUSDT | B23 | not falling | 6.47 | +1.25 (TP) | +0.75 | FLOOR | +0.46 | FLOOR |
| 2026-10-20T14:20:10 | BATUSDT | B23 | not falling | 1.42 | +1.25 (TP) | +1.42 | TP_T | +1.42 | TP_T |
| 2026-10-20T18:10:10 | ICNTUSDT | B23 | not falling | 1.25 | +1.25 (TP) | +1.25 | TP | +1.25 | TP |
| 2026-10-23T13:25:08 | ARKUSDT | B24* | falling | 3.43 | +1.25 (TP) | +0.82 | FLOOR | +0.47 | FLOOR |

## 6. Proposal if A ships (for the operator — not built here)

- Config: target = max(+1.25, min(+8, net % at H × 0.99)); arm at +1.25; floor **+0.5** net (market, taker); no late TP once armed;
  stop −4 unchanged (B not shipped).
- **Pre-committed revert gate (tighter than standard, because A passed only the override bar):** shadow-price today's exit on every
  armed live WILLY fill with T > +1.25; at **15 armed fills on ≥ 6 days**, revert if Σ Δ(V2 − today) < 0; re-read at 30.
- **Scout lines:** armed N · TP-at-T / floor / cap mix · Σ Δ vs today shadow; floor slippage vs F (live booking minus F, the 0.03
  assumption); B shadow line on BTC-falling fills (saved vs cut) continues as observe.

## 7. Caveats / what could not be tested

- **Intrabar order on 1m** (§4) is the largest uncertainty for A; no tick data for the 2025/2026 backtest fills.
- **Floor fills:** booked at min(F, open) − 0.03 on bars, print − 0.03 on ticks. Real slippage on a market exit in a fast WILLY
  name could exceed 0.03; each extra 0.05 of slip costs about 361 floors × 0.05 / 1,694 ≈ −0.011/fill in 2026 at F 0.5 — half
  of A's edge.
- **Seeds are not independent** (1,393 unique fills of 4,129); the day-clustered CI is the honest width.
- **In-sample haircut:** F was chosen walk-forward, but T (H × 0.99) came from the earlier study that looked at the same data;
  haircut 30–50 % → A ≈ +0.011 to +0.015/fill.
- **② master fits nicely, ① live does not:** A on the 7 armed live fills is −0.16/fill (floors on ZK, ZIL, ARK, ALT…). Small N.
- **ARK 10-23** is on public 1m to 13:42 UTC (provisional); under V2 it floors at +0.47 (F 0.5) / +0.82 (F 1.0) instead of +1.25 —
  on 1m the arming minute is ambiguous; it actually ran to the high within ~20 min, so a tick path may differ.
- Per BTC side at F 0.5, A is −0.118 [−0.22, −0.01] on 2026-H2 falling fills and +0.123 on not-falling — post-hoc, observe only
  (the earlier study already refuted BTC-conditional target rules).
