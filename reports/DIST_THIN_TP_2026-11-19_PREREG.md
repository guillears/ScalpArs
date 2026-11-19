# DIST_THIN_TP pre-registration — written 2026-11-19T13:16Z, before any result is read

Question (operator): should DISTRIBUTION_SHORT use a lower take profit on thin coins (24 h volume ≤ $42M, the frozen ENTRY_WL threshold)?
Mechanism (DIST_THIN_DEEPDIVE): thin coins go +1 % first as often as liquid ones but reach less far and give back more.

- Entries (frozen): the 1,534 RSI-kept one-position DIST fills (DIST_WL keep_strength ∧ DIST_EXIT REF_B row). Thin = vol24 ≤ 42e6.
- Engine: scripts/dist_exit_sim.py sim_trade (1m, stop checked first on the high, maker TP ≥ 1 tick through, gap-aware stop, real funding,
  model slip ≥ 0.10 %, fees taker 0.05 / maker 0.02) — the DIST_EXIT engine, unchanged.
- Cells (5, frozen): on thin fills only, TP = min(X, today's TP) with X ∈ {4, 5, 6, 7, 8} %; stop = today's min(1.5 × ATR15, 6 %); 24 h.
  Liquid fills stay on Exit B. Control (descriptive): the same cells applied to liquid fills.
- Metric: paired Δ net % per thin fill vs Exit B; Σ Δ; per block DESIGN (< 2025-07) / HOLDOUT (≥ 2025-07, read once).
- Ship bar (all): ① DESIGN mean Δ > 0, day-block (10-day) bootstrap 95 % CI excluding 0 · ② HOLDOUT mean Δ > 0 · ③ beats the
  family-wise null (day-block sign flips of Δ, max DESIGN mean Δ over the 5 cells, 2,000 draws, 95th pct) · ④ positive in ≥ 5 of 7
  half-years · ⑤ no pair > 25 % of Σ Δ · ⑥ thin-specific: Δ on thin > Δ on liquid for the same cell (else it is a generic exit effect,
  already refuted by DIST_EXIT). Then the 30–50 % haircut.
- Decomposition (CLAUDE.md exit rule): split Δ into REF-stopped fills (saved / deeper) vs REF take-profit / time fills (cut); fills
  whose path never reaches X must show Δ = 0.
