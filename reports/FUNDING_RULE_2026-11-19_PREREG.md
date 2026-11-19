# FUNDING_RULE pre-registration — written 2026-11-19T12:52Z, before any result is read

Question: does a rule on the funding rate at entry (shorts pay when it is negative) improve EXHAUSTION_SHORT or DISTRIBUTION_SHORT?
Trigger: ZKC 11-19 paid $195 funding (31 % of gross) at −0.78 %/h.

- Entries (frozen): EXH = BASELINE_TRADES_EXHAUSTION_2026-11-18.csv kept fills (dl582_block False), net = EXH_EXIT2 1m net_REF (today's exit, fees, slip, real funding; the baseline `r` is the older 5m result).
  DIST = DIST_WL trades with keep_strength True (RSI1h ≥ 74.07) joined to DIST_EXIT REF_B net (Exit B, real funding).
- Variable: last settled funding rate at entry (% per interval): EXH `f_last`, DIST `fund_last`. Missing = rule does not fire.
- Rules (10 per sleeve, frozen): SKIP if rate ≤ T, HALF size if rate ≤ T, T ∈ {−0.05, −0.10, −0.20, −0.50, −1.00}.
- Metric: Δ total net % at one size (sum over fills; HALF counts 0.5 × net) vs today, per block DESIGN 2023-07→2025-06 / HOLDOUT 2025-07→.
- Ship bar (all): ① DESIGN Δ > 0 with day-block (10-day) bootstrap 95 % CI excluding 0 · ② HOLDOUT Δ > 0 (read once) · ③ blocked cohort
  WR below the sleeve breakeven WR and avg < 0 · ④ blocked cohort ≥ 8 distinct days and N ≥ 15, no pair ≥ 50 % of its loss ·
  ⑤ beats the family-wise shuffled null (rate shuffled within half-year, max Δ over the 10 rules, 1,000 draws, 95th pct) on DESIGN.
- Descriptive only: funding paid by rate bucket; funding share of gross on winners.
- Blind spot (stated now): a funding-conditional hold cap / pre-settlement exit needs 1m paths + settlement times; not tested here.

**Amended after a first run:** EXH net switched from the baseline `r` (older 5m result) to EXH_EXIT2 1m `net_REF` (today's exit), so both sleeves use their exit-study net. The first-run verdict was the same (0 / 20 pass; every rule negative on HOLDOUT).
