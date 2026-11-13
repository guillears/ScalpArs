# FLATBASE_SHORT — earlier-entry study · PRE-REGISTRATION (written 2026-11-12 BEFORE any result)

Operator 2026-11-12 (PRL short, 16.6 % below the peak, 11 h after it): "seems like we are always entering LATE" → "yes run the earlier entry study".
Research only — nothing ships from this file; any candidate goes to the operator and, under the OBSERVE-FIRST rule, ships observe-only.

## Universe
- **① Live (master)**: every episode in the engine's own Flatbase ledger (decisions exports, e=FLATBASE, Downloads + exports_archive) that passed FLAT
  and is not a SHADOW (turnover ≤ 1): final status OPENED / SKIPPED / EXPIRED / MISSED / REFUSED / TRACKING. Spike = fb_spike_at, first-seen = fb_first_seen_at.
  SHADOW episodes (turnover ≤ 1) are a side line only.
- **② Backtest**: the 2026-10-22 study's FLAT episodes (episodes.csv label = flat, 2024-11 → 2026-10-22) on its own 5m grid; turnover unknown (caveat).

## Entry rules (frozen family — LIVE is the reference, 6 challengers → Bonferroni ×6)
Common gates for every rule: signal bar ≥ first-seen · ≤ 96 h after the spike bar · 24 h volume (Σ v × typical price, 288 bars) ≥ $20M ·
close < the spike-anchored VWAP (same formula as services/flatbase.s4_scan). One entry per episode, at the NEXT 5m open after the decisive close.
| Rule | Signal grid / min wait | Confirm |
|---|---|---|
| LIVE | 4-h closes, ≥ 4 h after the spike | first 4-h close below the signal close within 24 h, else no trade |
| S4 | 4-h closes, ≥ 4 h | none (pre-DL 428) |
| H1_4 | 1-h closes, ≥ 4 h | none |
| H1C_4 | 1-h closes, ≥ 4 h | first 1-h close below the signal close within 24 h |
| H1C_2 | 1-h closes, ≥ 2 h | same 1-h confirm |
| RED5 | any 5m close ≥ 30 min after the spike, after the peak bar, candle body ≤ −5 % | none |
| OFF10C | 1-h closes, ≥ 2 h, close ≤ running peak × 0.90 | same 1-h confirm |

## Exit (today's live exit, all rules): stop +10 % gross (checked first on a bar touching both; a gap books the open) → net −10.09 · TP −5 % → net +4.94 · else market at 72 h (net − 0.09). No funding in ① (paper); ② shown with and without funding.

## Read
Per rule: N trades · WR · avg % · Σ pts per episode (no trade = 0) · Δ vs LIVE per episode · day-clustered bootstrap (4,000, seed 7) of the per-episode Δ ·
LOO (largest single-episode Δ removed) · daily compound at ONE size (notional 0.38 × account ≈ today's $10k sizing) · entry lateness (hours after the spike / peak, % below the peak).
Backtest per half-year. Undecided (still running) trades → PENDING, the episode leaves every rule's comparison.

## Candidate bar (decided once, here)
A rule is an **observe CANDIDATE** only if ALL hold: ② backtest Δ/episode > 0 with the day-clustered one-sided lower bound > 0 at 1 − 0.05/6 · Δ > 0 in ≥ 3 of the 4
half-years · ① live Δ Σ ≥ 0 (master must not contradict). Else REJECTED. A candidate then needs the operator, the 30–50 % haircut on ①, and an observe line before any arm.

## ADDENDUM (2026-11-12, operator mid-run: "the 4 h should be rolling? … should it be 4 h from the peak?") — added after the live block printed, BEFORE the backtest ran
Four more challengers (family now 10 → Bonferroni ×10, bar otherwise unchanged):
| ROLL4 | any 5m close ≥ 4 h after the spike (no UTC 4-h grid) | none |
| ROLL4C | same | rolling: the 5m close exactly 4 / 8 / … / 24 h after the signal, first one below the signal close |
| PEAK4 | any 5m close whose running peak is ≥ 4 h old | none |
| PEAK4C | same | rolling confirm as ROLL4C |
The live block for these four is in-sample to the operator's question (PRL) → backtest carries the decision.

## CORRECTION (2026-11-12, operator: "first focus on all real trades, then if we find something we focus on backtest — you are doing the opposite")
The candidate bar above put the backtest first — WRONG order (memory feedback-master-first-then-backtest, 11-11). Path: ① real trades (live ledger, today's exit) decide;
② the backtest runs ONLY for a rule that beats LIVE on ①. Read on ①: no rule beats LIVE (LIVE 21 · 90 % · +3.51 · Σ +73.7 · +0.97 %/day; best challenger
H1C_2 Σ +59.3) → the backtest was NOT run. Result: reports/FLATBASE_EARLY_ENTRY_2026-11-12.md.
