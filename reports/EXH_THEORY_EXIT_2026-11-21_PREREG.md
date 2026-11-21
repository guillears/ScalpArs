# EXH_THEORY_EXIT — pre-registration (frozen 2026-11-21, before any cell is simulated)

## Why
RARE #26 (EXH short, 11-20 20:00) peaked +7.87 % and was stopped −7.42 % (price −5.90 % on a 5.63 % stop, slip 0.20 %, funding −1.43 %).
ENSO #29 peaked +9.62 % (target 10 %) and 15 h later sits at +0.16 % on price with −3.72 % funding paid. Operator: "study theory of what we
are trading and track possible exits".

## Theory of the trade
EXHAUSTION_SHORT shorts a Frenzy coin ≤ 72 h after its spike, when 15m RSI falls back through 70 with volume decaying and the spot share
falling: the buyers who made the pump are done, so price should revert toward where the pump started. The trade is a **mean reversion of
the pump**. Three things follow:
1. **Its natural target is a level of the pump, not a multiple of ATR.** Reversion stalls at the pump's own landmarks (the episode VWAP =
   the average buyer's cost, the 38 / 50 / 62 % retraces, the pre-pump base). Today's TP is min(20 × ATR15, 10 %) — blind to them.
2. **Reversion is fast or not at all.** Once the drop stops making new lows the exhaustion leg is over; the rest of the 24 h hold is
   exposure to the next pump.
3. **The thesis is wrong only above the pump's top.** A new high means the buyers are back. Today's stop (min(1.5 × ATR, 6 %)) can sit
   below or above that level by accident.
4. **Shorts pay when the crowd is short.** Deep negative funding is the cost of being in a crowded short; once it has eaten a set budget
   the trade's expectancy is spent.

## Already tested (cited, not redone)
EXH_EXIT_GRID (trails armed +2…+5, TP 4–10, SL 4–6, 12 / 48 h) · EXH_EXIT2 (break-even, locks, late trails, no-TP trails, small TP × tight
SL, 4 / 8 / 16 h holds, no-progress) · TP_GRID_2026-11-20 · STATE_EXIT_2026-11-21 (RSI / volume / OI / taker-buy / EMA / climax state
exits) · FUNDING_RULE_2026-11-19 (skip / half at entry) · DIST_FUNDING_EXIT_2026-11-20 (close before a deep settlement, EXH run 0 / 16) ·
FUNDING_STOP_2026-11-21 (stop counting funding: cuts > saves on EXH). None is repeated here.

## Cohort and engine
- 3,368 EXH backtest fills (`reports/EXH_EXIT2_data/rows_EXH.pkl`, today's entry rule + DL 582), 1m paths from the EXH_EXIT2 caches;
  rows without a full 1m path are dropped and counted.
- Engine rules identical to EXH_EXIT2: stop-first within a bar (stop on the high, filled at max(stop, bar open)); maker TP fills when the
  low trades 1 tick through it; costs = entry slip + 0.05 taker; exit taker 0.05 + slip (stop / time / rule exits) or maker 0.02 (TP);
  real funding at every settlement. **Parity gate: my REF must reproduce EXH_EXIT2 `REF` (exit kind and gross r) on ≥ 99.9 % of fills,
  else stop.**
- Pump anchors from the matching Frenzy episode (latest `on_ms` ≤ entry ≤ spike + 72 h, same pair; 5m bars): base = close 30 min before
  the on-bar; peak = max high from the on-bar to the bar before entry; VWAP = Σ typical × volume ÷ Σ volume over the same bars. All known
  at entry. Fills without an episode keep today's exit in every cell (Δ = 0) and are counted.

## Cells (23 + REF) — frozen
| family | cell | rule |
|---|---|---|
| A target (replace) | A_VWAP · A_R382 · A_R50 · A_R618 · A_BASE | maker TP at the anchor (VWAP / peak − R × (peak − base) / base) instead of today's TP; anchor not ≥ 1 % below entry → today's TP; capped 25 % |
| A target (nearer) | N_VWAP · N_R382 · N_R50 · N_R618 · N_BASE | maker TP at whichever is nearer to entry: the anchor (if ≥ 1 % below entry) or today's TP |
| B stall | B_X{3,5,7}_N{60,120,240} | after the best low reaches +X %, if no new low for N minutes and the 1m close is in profit → market exit next minute |
| C structure stop | C_PK0 · C_PK1 · C_TIGHT | stop at peak × 1.00 / × 1.01 (capped 15 % from entry; peak below entry +1 % → today's stop) · C_TIGHT = the tighter of today's stop and peak × 1.005 |
| F funding budget | F_P1 · F_P2 · F_P1N | at a settlement, cumulative funding PAID since entry ≥ 1 % / ≥ 2 % of notional → market exit at the next minute · F_P1N = ≥ 1 % paid and price P&L < funding paid (net under water) |

## Reads and the decision rule (pre-committed)
- Δ = rule net − REF net per fill (% at one size), summed and averaged. DESIGN = entries < 2025-07-01, HOLDOUT ≥ 2025-07-01 (read once).
- Per family the DESIGN-best cell is the family's pick. A pick **passes** only if ALL hold on HOLDOUT: ① mean Δ > 0 with the 95 %
  day-block bootstrap CI (10-day blocks) above 0 · ② Δ > 0 in ≥ 2 / 3 of the half-years · ③ still > 0 at a flat 0.20 % slip per taker leg ·
  ④ no single pair or day carries ≥ 50 % of the summed Δ · ⑤ saved (losers improved) $ > cut (winners reduced) $ · ⑥ the one-size DCR
  (0.61 × equity per fill) after ≥ before.
- Real fills (18:44 export, EXH closed rows; RARE / ENSO / ORCA open rows as illustration) are re-priced with the same engine on fapi /
  vision 1m and reported per fill. EXH has no master before 11-18, so they can refute (a pass that loses on every real fill gets a caveat)
  but cannot pass a cell on their own.
- No ship from this study without the operator; a passing pick becomes a proposal with a priced revert gate.

## Addendum (frozen 2026-11-21 ~19:20 UTC, after DESIGN / HOLDOUT, BEFORE the E block is read)
Post-hoc CONTROL showed the C_PK0 gain is a stop-WIDTH effect, not structure: fixed stops 7 / 8 / 10 / 12 % all beat today's
min(1.5 × ATR, 6 %) on DESIGN (+0.16…+0.32, CI spans 0) and HOLDOUT (+0.23…+0.41, CI > 0), a plateau, not a spike. EXH_EXIT_GRID only
tested stops 4–6 % (6 = the grid edge). Frozen pick for confirmation: **K_SL7 (fixed 7 % price stop, TP / 24 h unchanged)** — the best
DCR in both blocks and the smallest tail of the plateau. Confirmation test = the untouched 2023-H1 E block (rows_E.pkl, never used for any
pick): pass = Δ > 0 on E AND the one-size DCR after ≥ before; descriptive for 8 / 10 / C_PK0. Risk (worst day, max drawdown at 0.61 ×
equity) is reported beside every number — a wider stop ships only with a size read (Kelly at the Wilson floor, clustering).
