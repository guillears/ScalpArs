# WILLY exit V2 — pre-registration (2026-10-23, written before any V2 result)

Research only. No code, config or git change. Extends `reports/WILLY_HIGHTARGET_2026-10-23.md`; reuses its fill sets, spike high H,
BTC-side labels and walker (`scout_willy_timecap.evaluate_late(book="live")`). Script: `scripts/study_willy_exitv2.py`.

## Rule under test (operator-approved "V2"), on every WILLY fill
- **TP target (net)** T = max(+1.25, min(+8, net % at price H × 0.99)) — identical to the study's `H1_k1_tgt`. Resting maker limit:
  a TP books exactly T.
- **Stop before arming:** −4 net if BTC not falling, −2.5 net if BTC falling (BTC 1 h EMA20 slope ≤ 0 at entry; the study's `btc`
  label; `unscored` → −4).
- **Arm** when net P&L first touches +1.25 (before the 30-min late minute). From then: floor F net; first event with net ≤ F →
  market exit booking min(F, bar open) − 0.03 on a 1m bar (bar opening through F books its open), the print − 0.03 on a tick. The stop
  no longer applies once armed. Armed trades keep T (no late TP) until the 90-min cap (CAP books the last close ≤ cap).
- **Unarmed:** after 30 min → late TP +0.25 (today's booking); 90-min cap. Unarmed outcomes equal today's walker with the same stop
  (asserted fill by fill).
- **Intrabar:** the arming event never fires the floor (conservative per operator spec) but can fill T if its high reaches T; on any
  later shared bar floor before TP; stop before arm/TP on a shared bar (as today's walker).
- If T = +1.25 the arm touch IS the TP (+1.25) — identical to today.

## Variants
- TODAY = +1.25 / −4 / late +0.25 @ 30 / 90.
- **A(F)** = T + floor F, stop −4 everywhere. F ∈ {1.0, 0.75, 0.5, 0.25}.
- **B** = TP +1.25 (today), stop −2.5 on BTC-falling fills, −4 otherwise.
- **AB(F)** = both.

## Blocks
- ① live (B22–B23 live fills + ARK 10-23 provisional, 18) — informational.
- ② master: 36 fills B19–B23 (`reports/WILLY_HIGHTARGET_2026-10-23_real_btc.csv`, ARK excluded), per batch (era).
- ③ backtest: the study's bt set (`_bt_btc.csv`, 4,129 fills, yr6a 2025 + yr6 2026, seeds 1–3, trimmed), pooled and per seed; per
  period 2025 / 2026-H1 / 2026-H2 and per BTC side.

## Measures
avg % net per fill, WR, Σ, worst; Δ vs TODAY with day-clustered bootstrap 95 % CI (`study_willy_hightarget_common.day_boot`).
Affected-cohort accounting: A's Δ only on armed fills (unarmed Δ asserted exactly 0); B's Δ only on BTC-falling fills that hit −2.5,
split saved (today stopped at −4: loss shrunk) vs cut (today not stopped: now stopped); every other fill asserted Δ = 0.

## Choice of F (never best-in-sample)
- **Anchored walk-forward** over the 4 half-years: the F for 2025-H2 is picked on 2025-H1, for 2026-H1 on 2025-H1+H2, for 2026-H2 on
  2025-H1..2026-H1 (pick = highest mean Δ of A(F) vs TODAY on the training halves). The **deployment F** = the pick trained on
  2025-H1..2026-H1 (the latest out-of-sample-tested pick). Full F table reported for every block. LOHO picks reported as information.
- AB uses the same F as A (no separate fit); AB's full F table also reported.

## Ship bar (operator-approved, declared override of the study bar)
- **A ships iff** its ③ backtest 2026 Δ (pooled 2026-H1 + 2026-H2 fills, each half at its anchored walk-forward F) has point estimate ≥ 0.
- **B ships iff** its ③ backtest 2026 Δ (pooled 2026-H1 + H2) point estimate ≥ 0.
- Informational: the stricter study bar (2026 CI lower bound > 0 AND held-out Δ > 0 in ≥ 3 / 4 halves — LOHO for A, per half for B).
- Also reported: ② master per batch before → after for AB at the chosen F (operator expectation ≈ +0.26 → +0.64 at F 1.0).
