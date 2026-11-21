# DIST_PEAKSTOP — pre-registration (frozen 2026-11-21, before any cell of this grid is simulated)

## Why
Operator: "For DIST, 'stop at pump top +1 %' calls my attention — what is it? does it make theoretical sense? deep dive, complete rule and
recommendation." The first study (reports/DIST_THEORY_EXIT_2026-11-21.md) tested the pump-top stop capped at 15 % (C_PK0 / C_PK1 / C_PKT):
HOLDOUT Δ −0.07 / −0.14 / −0.05 %/fill. On the 6 real fills C_PK1 = +$141 vs today's −$560 (rescues SENT / BLESS, kills CYBER). Operator
constraint: a −25 % loss on one trade is not acceptable → every cell here carries a HARD CAP ≤ 15 %. This study is a follow-up on the same
1,547-fill cohort, so its DESIGN half has already been seen for C_PK* / W_SL* (stated; the HOLDOUT gate set is the protection).

## Definitions (frozen)
- **E** = entry price (backtest: the 1m open of the entry minute; real: the real fill price).
- **Today's stop** S0 = E × (1 + min(1.5 × ATR15 %, 6 %)/100). TP = E × (1 − min(20 × ATR15 %, 10 %)/100), resting maker. Time exit 24 h. Unchanged in every cell.
- **Pump top PK** = the highest price traded from the FRENZY spike (episode on-bar: backtest `ep_on_ms`, 5m bars; real: `entry_frenzy_spike_at` − 5 min,
  1m bars) up to the last bar that closed BEFORE the entry — it includes the distribution candle itself. Known at entry; no look-ahead.
- **Capped top** TOP(CAP) = min(PK × 1.01, E × (1 + CAP/100)).
- Missing PK → today's stop (Δ = 0), counted.

## Cells (frozen)
| family | cells | stop level | note |
|---|---|---|---|
| REF | REF | S0 | today |
| P replace | P8 · P10 · P12 · P15 | TOP(CAP) if PK ≥ E × 1.01, else S0 | can be TIGHTER than today (pump top close to entry) |
| Q wider-of | Q8 · Q10 · Q12 · Q15 | max(S0, TOP(CAP)) | never tighter than today → fills today did not stop must show Δ = 0 |
| W fixed control | W8 · W10 · W12 · W15 | E × (1 + CAP/100) | "is it just a wider stop?" |
| S shuffled-top control | S{8,10,12,15}_{1..5} | max(S0, min(E × (1 + d_j/100), E × (1 + CAP/100))), d_j = (PK/E − 1) × 100 + 1.01-buffer of a random OTHER fill j in the same block (seeded permutations 1…5) | same width distribution as Q, anchor removed: tests whether the LOCATION of the top matters |
| DQ descriptive | DQ10 · DQ15 | max(S0, min(DH × 1.01, E × (1 + CAP/100))), DH = distribution-candle high | theory contrast (selling candle's high vs pump top); not pickable |

## Method (frozen)
- Cohort: the 1,547 DIST_EXIT REF_B backtest fills (`reports/ENTRY_WL_2026-11-18_data/dist.pkl`), engine = DIST_THEORY_EXIT sim (`sim_one`, family W
  with a per-fill stop %): 1m path (5m fallback), stop first on the HIGH (fill max(stop, open)), maker TP 1 tick through, both in a bar → stop, 24 h.
  Nets = DIST_THEORY_EXIT eval `nets` (entry slip + 0.05 taker; exit 0.05 taker + slip model floor 0.10, or 0.02 maker; real funding).
- **Parity:** my REF must equal sim_BT.npz REF (gross within 0.01 %, same kind) on ≥ 99.9 % of fills, and P15 must equal C_PK1 the same way. Else stop.
- DESIGN = entries < 2025-07-01, HOLDOUT ≥ 2025-07-01, read ONCE. Δ = cell net − REF net per fill.
- **Pick:** the DESIGN-best mean Δ among the 8 pump-top cells (P ∪ Q). W and S are controls; DQ descriptive only.
- **HOLDOUT gates (ALL required):** ① mean Δ > 0 with 95 % day-block (10-day) bootstrap CI > 0 · ② Δ > 0 in ≥ 2/3 HOLDOUT half-years ·
  ③ Δ > 0 at a flat 0.20 % slip per taker leg · ④ no single pair or day ≥ 50 % of the summed Δ · ⑤ on the LIVE-STOPPED cohort (fills REF stopped):
  saved $ > deeper $ (+ any $ lost on non-stopped fills; Q / W / S must show non-stopped Δ = 0, asserted) · ⑥ one-size DCR (0.61 × equity per fill,
  every calendar day) after ≥ before · ⑦ HOLDOUT Δ > the W control at the same CAP AND > the mean of the 5 shuffled-top controls at the same CAP
  (the theory claim: the top's location carries information) · ⑧ one-size max drawdown after not deeper than before by more than 5 points.
- Risk read beside every cell: avg win, avg loss, worst trade, worst day, max DD, TP / STOP / TIME counts, average and median duration (h).
- Multiple testing: 8 pickable cells, one pick; the shuffled control is the null for the anchor.
- **Real fills** (6 closed DISTRIBUTION_SHORT; newest export scalpars_orders_paper_2026-11-21_19-22-18.csv, cross-checked against every
  export, dedup on (opened_at[:19], pair, direction)): re-walked with the same engine on cached 1m from the real fill price. N = 6 = about 3
  market windows: may refute (caveat), never pass a cell. There is no master batch for DIST (sleeve live since DL 586, 11-18; to be confirmed on
  reports/MASTER_POOL_stacked.csv).
- **Decision rule:** all 8 gates → propose ship to the operator with a priced alert-only revert gate · ⑦ passes and Δ ≥ 0 but ① fails → OBSERVE line
  (frozen definition, read at ≥ 30 real fills over ≥ 10 days) · otherwise NO (keep today's stop).
