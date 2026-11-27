# HT blockers — why no HT trades since 11-21, and the 2-year audit of each blocker (2026-11-27, research only)

**Question (operator):** why are we not getting HT trades; which blockers refuse them; do the blockers hold up in the backtest?
**Prior (re-used):** DL 527 / 628 FAILED_SPIKE gate frozen KEEP · HT_FAILED_SPIKE_PARITY 11-19 (engine = rebuild) · DL 594 NEAR_HIGH · DL 619 BURST ·
11-27 log check (25 setups since 11-21, all refused). **New:** every blocker judged on the SAME 1,142 unfiltered rebuild HT signals
(2025-01 → 2026-11-17) at today's exit (TP +6 net / SL −4 / 24 h, funding, 0.10 slip), alone, sole, and as a stack.
Script: scripts/HT_BLOCKERS_AUDIT_2026-11-27.py · data reports/HT_BLOCKERS_AUDIT_2026-11-27.csv · _summary.txt.
Parity: 299 of the 300 already-priced kept rows re-price identically (TUTUSDT 2025-10-10, a +6 TP, is missing from the 1,142 set; with it
"today" Σ −5 → ≈ +1). NEAR_HIGH uses the true spike high on 415 rows and a 6 h-max proxy on the 727 FAILED_SPIKE-blocked rest — the proxy catches
only 63 of 117 true NEAR_HIGH blocks (54 %), so "NEAR_HIGH alone", "FS sole" and "today minus FAILED_SPIKE" carry a noisy NEAR_HIGH mask
(FS sole ≈ −0.28 → ≈ −0.26 corrected); the deep NEAR_HIGH read below uses the true H on 372 / 373 rows.

## Live — refused since the last HT fill (11-21 02:30 UTC), priced on HT's exit (scripts/HT_BLOCKERS_LIVE_2026-11-27.py → _LIVE csv / summary)
| Blocker | Refused (priced) | TP / STOP | WR | avg | Σ |
|---|---|---|---|---|---|
| FAILED_SPIKE | 14 (13) | 3 / 10 | 23 % | −1.70 | −22.0 |
| NEAR_HIGH | 4 | 2 / 2 | 50 % | +1.00 | +4.0 |
| BURST_REJECTION | 1 | 1 / 0 | 100 % | +6.00 | +6.0 |
| Funding guard | 1 | 1 / 0 | 100 % | +6.00 | +6.0 |
| All | 20 (19) | 7 / 12 | 37 % | −0.32 | −6.0 (6 pts saved) |
Since arming (11-12): FAILED_SPIKE 33 refused / 32 priced on 12 days · 19 % · −2.28 · Σ −72.8 (gate frozen KEEP 11-22). Kept HT fills 11-12 → 11-21: 13 · 54 % · +0.58.

## Backtest — each blocker alone (blocked | kept)
| Blocker | Blocked | Kept | DESIGN b / k | HOLDOUT b / k | P(blocked < 0) | Sole blocker |
|---|---|---|---|---|---|---|
| FAILED_SPIKE | 726 · 37 % · −0.30 | 416 · 37 % · −0.37 | +0.73 / −0.39 | −0.42 / −0.37 | 0.94 | 464 · −0.28 |
| NEAR_HIGH | 137 · 34 % · −0.70 | 1,005 · −0.27 | +0.38 / +0.25 | −0.91 / −0.33 | 0.95 | 113 · −0.93 |
| BURST | 287 · 35 % · −0.50 | 855 · −0.27 | +1.00 / +0.04 | −0.67 / −0.31 | 0.96 | 39 · −0.71 |
Half-years: the blocked sides mostly won or broke even in 2025 (2025-H2: FS −0.01, NH +0.06, BURST −0.48) and lost from 2026-H1 on (2026-H2: FS −0.71 · NH −1.07 · BURST −0.91 vs kept +0.47).

## Backtest — stack before → after (one per coin, one size 0.61)
| Stack | N · WR · avg | DCR %/day | max DD | worst day |
|---|---|---|---|---|
| no blockers | 1,136 · 37 % · −0.34 | −0.437 | −96.6 % | −15.0 % |
| FAILED_SPIKE + NEAR_HIGH | 299 · 39 % · −0.11 | −0.051 | −53.0 % | −7.5 % |
| **today (all three)** | **260 · 40 % · −0.02** | **−0.023** | **−44.2 %** | **−7.5 %** |
| today minus FAILED_SPIKE | 722 · −0.19 | −0.179 | −83.5 % | −8.8 % |
| today minus NEAR_HIGH | 372 · −0.31 | −0.132 | −69.6 % | −7.5 % |
| today minus BURST | 299 · −0.11 | −0.051 | −53.0 % | −7.5 % |
2026 only: no blockers 775 · −0.60 → today 170 · −0.14.

## Meaning
- HT is not trading because the setups since 11-21 were failing spikes (14 of 20); the bot is working as designed, not broken.
- Each blocker improves the stack in the backtest (removing any one of them lowers avg and deepens the drawdown); all three point the same way
  in 2025-H2 → 2026 and the opposite way in 2025-H1 (one era flip, shared by all three).
- FAILED_SPIKE alone does not separate in the backtest (blocked −0.30 ≈ kept −0.37); its value is inside the stack (+0.17 %/fill) and on
  live refusals (−2.28 on 32 priced). NEAR_HIGH is the strongest backtest separator (sole −0.93).
- Caveat: the backtest's HT on today's stack is breakeven (−0.02 %/fill, 2026 −0.14) while real kept HT fills are positive
  (master +4.46 on 12 at TP 6, batch +0.58 on 13). Master over backtest; HT stays on (operator 11-18).

## Verdict
No change. Keep all three blockers. Gates already running: FAILED_SPIKE frozen KEEP; NEAR_HIGH 4 / 15 and BURST 1 / 10 still collecting.

---
## Deep dive (same day) — is each filter correct? (scripts/HT_BLOCKERS_DEEP_2026-11-27.py, _summary.txt)
Criteria (set AFTER the audit's sole-blocker column was seen — not pre-registered): each blocker judged on what the other two let through
(one position per coin on both sides); passes if blocked WR < HT breakeven 40.3 % · worse than kept in BOTH blocks · P(blocked < 0) ≥ 0.95
(day bootstrap) · gap beats a within-month shuffle at α 0.05 · no pair / day ≥ 50 % of the loss · N ≥ 15 on ≥ 8 days.

| Blocker | Blocked | Kept | WR < 40 % | Both blocks | P(<0) ≥ 0.95 | Shuffle p < 0.05 | Concentration | Passes |
|---|---|---|---|---|---|---|---|---|
| NEAR_HIGH | 113 · 32 % · −0.93 | 260 · 40 % · −0.02 | ✅ | ❌ DESIGN +0.52 vs −0.81 (19 fills) | ✅ 0.98 | ❌ 0.079 | ✅ | 4 / 6 |
| FAILED_SPIKE | 462 · 37 % · −0.29 | 260 · −0.02 | ✅ | ❌ DESIGN +0.51 vs −0.81 | ❌ 0.89 | ❌ 0.20 | ✅ | 2 / 6 |
| BURST | 39 · 33 % · −0.71 | 260 · −0.02 | ✅ | ❌ (DESIGN 1 fill) | ❌ 0.82 | ❌ 0.15 | ✅ (top pair 15 %) | 2 / 6 |
2026 only: NEAR_HIGH blocked −1.52 · FAILED_SPIKE −0.57 · BURST −1.28 vs kept −0.14. HOLDOUT (≥ 2025-07): −1.22 / −0.37 / −0.62 vs kept +0.09.

Dose-response ([lo, hi) buckets): NEAR_HIGH dip < 0.5 15 · −2.47 · 0.5–1.0 −0.48 · 1.0–1.4 −0.92 · 1.4–2.0 +0.74 · 2–4 111 · −0.69 · > 4 +0.33 (not monotone;
the gap leans on the 1.4–2.0 bucket) · FAILED_SPIKE vs_vwap ≤ −3 −0.63 · −3…−1 −0.91 · −1…0 264 · +0.05 · 0…1 −0.25 · 1…3 +0.50 · > 3 −0.10 (loss in
the deep failures; shallow ≈ kept) · BURST cloc15 < 0.25 13 · 15 % · −2.54 · 0.25–0.50 26 · +0.20 · 0.50–0.75 +0.05 · ≥ 0.75 +1.34.

In-sample caveat: NEAR_HIGH's blocked cohort is essentially DL 594's discovery set (117 trades, the 1.4 cut found on this backtest) and BURST's
is DL 618's 38-fill cohort — this is a re-read at TP 6, not new evidence. Live: the 3 shallow FAILED_SPIKE refusals all stopped (3 fills — noise);
FAILED_SPIKE since arming 32 priced on 12 days · 19 % · −2.28 is consistent with the rule (no window-clustered test at this N).

**Verdict:** all three blockers point the right way (blocked below breakeven and worse than kept since 2025-07 / in 2026; removing any one worsens
the stack), none passes the full bar on the backtest. They stay as armed declared overrides and their frozen gates decide: FAILED_SPIKE KEEP
(11-22) · NEAR_HIGH 4 / 15 · BURST 1 / 10. No change. Watch notes (no re-fit): BURST's loss sits at cloc < 0.25; FAILED_SPIKE's in the deep failures.
