# DIST_THEORY_EXIT — pre-registration (frozen 2026-11-21, before any cell is simulated)

## Why
Operator: "build all studies, and bring a solid exit strategy per sleeve". DISTRIBUTION_SHORT real fills so far 1 W / 5 L (NAORIS,
SENT, EVAA, KERNEL, BLESS stopped; CYBER TP). Template = EXH_THEORY_EXIT_2026-11-21 (same families, adapted to what DIST enters on).

## Theory of the trade
DIST shorts a Frenzy coin (≤ 72 h after its spike) on a **closed red 15m candle with quote volume ≥ 5 × its 20-candle mean**, while the
top traders cut their long positions (top-trader position L/S 4 h change ≤ −0.011) and the 1h RSI is still ≥ 74. Reading: a large holder is
**distributing into the pump** — the red high-volume candle is supply hitting the bid, the smart-money ratio says who is selling.
Predictions (each turned into a cell family):
1. **The distribution candle is the supply level.** Its high (DH) is where the sellers sold; price back above DH means the supply was
   absorbed and the thesis is dead. A stop above DH (or above the pump's top, PK) is the structural invalidation; today's
   min(1.5 × ATR15, 6 %) is blind to it (pre-sim entry look, no outcomes: DH sits median 6.1 % above the signal close, P10 2.6 %, P90 16 %;
   PK median 7.0 %; today's stop median 4.7 %). → families C (intrabar) and R (closed 15m bar).
2. **Absorption ends distribution.** A closed GREEN 15m candle carrying volume comparable to the distribution candle = buyers absorbing
   the supply. → family G.
3. **The follow-through is bounded by the pump.** The move unwinds the pump toward its landmarks (episode VWAP, 38 / 50 / 62 % retraces,
   pre-pump base), or by a measured move of the distribution candle's own range. → families A / N (pump anchors, replace / nearer) and
   M (measured move).
4. **Distribution is a one-leg event.** Once the drop stops making new lows the sell program is done. → family B (stall).
5. **Crowded short pays funding.** → family F (funding budget).
6. **Control:** a plain fixed wider stop (8 / 10 / 12 / 15 %, today's TP) — so a structure-stop effect is not just "a wider stop".

## Already tested (cited, not redone)
DIST_EXIT_2026-11-18 (54 cells: fixed TP × SL, BE +2/3/4, locks, trails, time 4/12/24/48 h — 0 pass; REF_B = Exit B) ·
DIST_FB_LOSERS_2026-11-20 §6.1 (ATR stop widths 1–3 × ATR, caps 4–12 % — keep today's) · DIST_FUNDING_EXIT_2026-11-20 (0 / 16) ·
FUNDING_RULE_2026-11-19 · FUNDING_STOP_2026-11-21 (funding-counting stop: cuts > saves on DIST) · TP_GRID_2026-11-20_DIST / DL 603–605
(TP 4–10; DIST_TP_SHADOW TP 5 / 8 watch line) · DIST_LOSERS2_2026-11-20 (per-cohort TP 3–8 %) · DIST_THIN_TP_2026-11-19 · 11-18
short-exit refutations. None repeated.

## Cohort and engine
- 1,547 DIST backtest fills = DIST_EXIT REF_B one-position cohort (`reports/ENTRY_WL_2026-11-18_data/dist.pkl`, `zi` → L1 rows of
  `DIST_EXIT_data/sim_L1.npz`). Entry set held fixed at REF's (a rule that exits earlier could free the coin for a later signal — not modelled,
  stated).
- Engine = DIST_EXIT (`scripts/dist_exit_sim.py`): 1m path, entry = 1m open; stop checked first on the HIGH (fill max(stop, open)), maker TP
  when the LOW trades ≥ 1 tick through; both in one bar → stop; time exit = close of the last bar < 24 h; censored paths excluded (k = 9);
  rule exits (B / R / G / F) = taker at the open of the minute after the trigger. Costs as DIST_EXIT net_mat: entry slip + 0.05 taker;
  exit taker 0.05 + slip (model, floor 0.10) or maker 0.02; real funding to the exit.
- **Parity gate:** my REF must reproduce DIST_EXIT REF_B (exit kind and gross r within 0.01 %) on ≥ 99.9 % of fills, else stop.
- Landmarks (5m bars, all known at entry): distribution candle = the three 5m bars [T − 15 m, T): DO = its open, DH = its high, DL = its
  low, DQV = its quote volume; episode on-bar = `ep_on_ms`; base = 5m close 30 min before the on-bar; PK = max high from the on-bar to the bar
  before entry; VWAP = Σ typical × volume ÷ Σ volume over the same bars. Fills with missing landmarks keep today's exit (Δ = 0), counted.
- 15m bars after entry are aligned to T (T is a 15m boundary); volume from the 5m bars.

## Cells (41 + REF) — frozen
| family | cells | rule |
|---|---|---|
| A target replace | A_VWAP · A_R382 · A_R50 · A_R618 · A_BASE | maker TP at the anchor (VWAP / PK − R·(PK − base) / base); anchor not ≥ 1 % below entry → today's TP; TP capped 25 % |
| N target nearer | N_VWAP · N_R382 · N_R50 · N_R618 · N_BASE | maker TP at the nearer of the anchor (if ≥ 1 % below entry) and today's TP |
| M measured move | M_1 · M_15 · M_2 | maker TP at entry × (1 − m × (DH − DL) ÷ entry), m = 1 / 1.5 / 2; < 1 % → today's; capped 25 % |
| B stall | B_X{3,5,7}_N{60,120,240} | after the best low reaches +X %, no new low for N min and the 1m close in profit → market exit next minute |
| C structure stop | C_DH0 · C_DH1 · C_DHT · C_PK0 · C_PK1 · C_PKT | stop at DH (or PK) × 1.00 / × 1.01, capped 15 % from entry, level < entry × 1.01 → today's stop · *T = tighter of today's stop and level × 1.005 |
| R closed-bar structure | R_DO · R_DH | R_DO = today's stop AND exit when a closed 15m bar closes above DO · R_DH = intrabar stop replaced by: a closed 15m bar closes above DH → exit; catastrophe stop entry × 1.15 (DH < entry × 1.01 → today's exit) |
| G absorption | G_ABS50 · G_ABS100 · G_ABS100L | closed GREEN 15m bar with quote volume ≥ 0.5 / 1.0 × DQV → exit (L: only if that bar closes above entry) |
| F funding budget | F_P1 · F_P2 · F_P1N | cumulative funding PAID ≥ 1 % / 2 % of notional at a settlement → exit · P1N = ≥ 1 % and price P&L < paid |
| W control | W_SL8 · W_SL10 · W_SL12 · W_SL15 | fixed stop 8 / 10 / 12 / 15 %, today's TP |

## Reads and the decision rule (pre-committed)
- Δ = cell net − REF net per fill (% at one size). DESIGN = entries < 2025-07-01 (559), HOLDOUT ≥ 2025-07-01 (988), read once.
- Per family the DESIGN-best mean Δ is the pick. A pick **passes** only if ALL hold on HOLDOUT: ① mean Δ > 0 with the 95 % day-block
  (10-day) bootstrap CI above 0 · ② Δ > 0 in ≥ 2 / 3 of the HOLDOUT half-years · ③ still > 0 at a flat 0.20 % slip per taker leg ·
  ④ no single pair or day ≥ 50 % of the summed Δ · ⑤ saved (REF losers improved) $ > cut (REF winners reduced) $ · ⑥ one-size DCR
  (0.61 × equity per fill, every calendar day) after ≥ before · ⑦ C and R picks only: HOLDOUT Δ also > the W control cell whose stop is
  nearest the pick's median stop distance on its changed fills (else it is "just a wider stop").
- Multiple-testing note (pre-stated): 9 picks each at a one-sided 2.5 % CI → ≈ 20 % chance that at least one passes ① by luck; a lone pass
  with all other families flat is an OBSERVE line, not an arm.
- Risk at one size beside every pick: avg loss, worst day, max drawdown (sum of every fill, no slot cap — overstates clustering).
- Real fills (every exports_archive orders export, DISTRIBUTION_SHORT, CLOSED, no probes, dedup (opened_at[:19], pair, direction) keeping
  the newest export) re-priced with the same engine on 1m (vision / fapi) from the real fill price (path from the minute after the fill),
  landmarks from 1m (episode on = entry_frenzy_spike_at − 5 min), real funding. N ≈ 6 → they can refute (caveat), never pass a cell.
- No ship from this study; a passing pick becomes a proposal to the operator with a priced revert gate. If nothing passes: keep Exit B.
