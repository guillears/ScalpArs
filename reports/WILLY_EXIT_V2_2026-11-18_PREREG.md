# WILLY / WILLY_HT EXIT V2 — PRE-REGISTRATION

**Frozen 2026-11-18 12:38 UTC, before any new exit result.** Research only: no edit to services/, templates/, tests/, trading_config.json or CLAUDE*.md; no commit.

Operator question: "Can it be that WILLY HT and WILLY do not work in the backtest due to the exit? … maybe trying different exits?"

## Known before freezing (from WILLY_DEEP_2026-11-18, not re-fit)
- Today's exit, full history: WILLY 545 trades −0.083 %/trade; HT 417 trades −0.268 %/trade. HOLDOUT-D: WILLY −0.18, HT −0.21.
- Path buckets: STOPPED_RECOVERED WILLY 96 / −393 pts, HT 81 / −331 pts (the largest loss bucket in both).
- Fixed TP {2,3,5,8} × SL {2,3,4,6} with a 90-min cap (+240 min for TP ≥ 5), ATR caps and no-follow-through exits: none beat today's consistently.
- **HOLDOUT-D (2023-H2) was already read once by WILLY_DEEP.** It is no longer fully clean. It is read here once more, as a declared second read.

## Trades (re-used, unchanged)
- The WILLY_DEEP base trades (`scratchpad/deep/feat_base.pkl`, 962 rows: 545 WILLY + 417 HT, today's entry rules, real-time flags) and HOLDOUT-D (`D_build.pkl`; each sleeve's rule on every D entry, routing not rebuildable — declared).
- Entries are kept fixed. A slot / one-position-per-pair effect of a longer hold is measured (how many base entries would have been refused because the same pair is still open) and reported; the base headline re-uses the fixed entry list (the portfolio is not re-walked), declared.
- **Parity check first:** the new simulator with today's parameters must reproduce `net` of the base trades (median |Δ| < 0.01 pts, ≥ 95 % within 0.05). If not, stop and fix before any cell is read.

## Blocks
- DESIGN = entries 2025-01-13 → 2026-06-30 (choice is made here only).
- SEEN = 2026-07-01 → 2026-11-17 (read after the choice).
- HOLDOUT-D = 2023-07 → 2023-12 (second read, declared).
- Per sleeve, WILLY and HT separately. Never pooled.

## Realism
- 1m bars for the whole hold (daily 1m zips from data.binance.vision, extended to 48 h + 1 h). 5m fallback only where 1m is missing; count reported.
- Maker TP fills only when the high trades through the limit by ≥ 1 tick; maker fee 0.02 %.
- Stops / trails / time exits: taker 0.05 % + slippage 0.10 % (also 0.05 and 0.20), gap-aware (a bar opening through the line fills at the open).
- Inside a 1m bar the stop is checked before the TP (pessimistic). Trail / break-even lines move only on the highs of CLOSED bars (no intrabar look-ahead).
- Real funding between entry and exit (hourly coins included, from the settlement files).
- Stop distance convention as today: a "−X %" stop books net −X % before slippage (fees inside).

## "Today's TP" (TODAY-TP)
Today's live target logic: target = max(1.25, min(8, spike-high −1 % net)), HT floor 2 %; late TP to +0.25 % after 30 min; WILLY runner (2×ATR trail, floor +0.10) when the coin 1h gap < 0. Held to the cell's hold.

## Families (all cells, both sleeves)
1. **Wide stop + long hold:** SL {4, 6, 8, 10, 12 %} × hold {90 m, 4 h, 12 h, 24 h, 48 h} × TP {TODAY-TP, fixed 3 %, fixed 5 %} (fixed TPs: no late TP, no runner). 75 cells. SL4 / 90 m / TODAY-TP = today (the reference).
   - **Headline (operator request, named in advance): HT · SL −10 % · TODAY-TP (+2 % floor) · hold 4 h / 12 h / 24 h**, shown first, with the full saved / deeper / timed-out split and the deepest-dip distribution of the 81 STOPPED_RECOVERED HT trades (below −6 / −8 / −10 / −12 %). This is a reporting priority, not a selection: it is still judged by the same bar.
2. **Structural stop** at a level × (1 − 0.5 %): spike-bar low · anchored spike VWAP (live ÷ (1 + vs_vwap %)) · pre-pump base (close 6 bars before the spike bar, the run_pct base). No trade if the stop is > 12 % away or at / above the entry. TODAY-TP, hold {4 h, 24 h}. 6 cells. A skipped entry counts 0.
3. **Break-even lock:** after the high reaches +a % gross (a ∈ {0.75, 1, 1.5}) the stop moves to the net-zero price (entry + fees). TODAY-TP, −4 % stop, 90 m. 3 cells.
4. **Trailing:** after +a % gross (a ∈ {1, 2, 3}) a trail b % below the highest closed-bar high (b ∈ {1, 2, 3}); no TP; −4 % stop until armed; hold {90 m, 24 h}. 18 cells.
5. **Time grace:** no stop for the first N min (N ∈ {15, 30, 60}), then the −4 % stop (a bar already below fills at its open); a −15 % hard stop always on. TODAY-TP, 90 m. 3 cells.

Live-engine support is reported per family (what the engine already has vs needs code).

## Reading (per cell)
- N · trades/day · WR · avg win / avg loss · breakeven WR · avg net % · day-clustered 95 % CI (day-block bootstrap, 2000) · each half-year · best-5 % removed · @0.20 % slip · Kelly at the one-sided 95 % Wilson WR floor with N = distinct trading days · max loss per trade · DCR at one size (X = 1, i.e. fill % ÷ 100 summed per calendar day over every day of the block; the sleeve's true X scales it linearly).
- **Live-stopped split (every cell):** trades stopped under today's exit → saved (now ≥ 0 or better) / deeper (worse) / other, Δ pts; trades NOT stopped today → Δ pts (must be ≈ 0 for a TP-unchanged, hold-unchanged stop change, else the replica is lying). Two-sided, per period, day-block bootstrap CI of the total Δ.
- **Plateau:** a cell counts only if ≥ 2 of its grid neighbours (one step in SL or hold) also have DESIGN Δ > 0.
- **Family-wise null:** day-block sign-flip of every cell's per-trade Δ vs today (one sign per day, applied to all cells together), 2000 draws; statistic = max over cells of DESIGN mean Δ; p = share ≥ real max.
- **Haircut:** the chosen cell's DESIGN Δ is reported with a 30–50 % in-sample haircut.

## Choice (DESIGN only, per sleeve)
The cell with the highest DESIGN mean net per base entry (skips = 0). Also the best of each family is reported, not selected.

## Bar for "CHANGE EXIT" (all required)
1. DESIGN Δ vs today > 0 and family-wise null p < 0.05.
2. SEEN Δ > 0.
3. HOLDOUT-D Δ > 0 (second read, declared).
4. Plateau (≥ 2 neighbours Δ > 0 on DESIGN).
5. Δ > 0 at 0.20 % slippage (DESIGN + SEEN).
6. Master re-price (kept fills, per batch B25 / B26 / B27 + the 11-18 paper export): Δ ≥ 0 in total; no flip on the observe-line-kept trades (joint check: HT 4 h slope ≤ 0; HT top-trader position L/S ≤ 1).
- If 1–6 pass but the sleeve with the new exit is still negative (DESIGN + SEEN avg ≤ 0, or Kelly at the floor ≤ 0) → **OBSERVE** ("a better exit, still no edge").
- If 1–6 pass and the sleeve is positive with Kelly at the floor > 0 → **CHANGE EXIT** (exact numbers).
- Otherwise → **KEEP** today's exit.

## ADDENDUM 1 — frozen 2026-11-18 12:39 UTC, before any cell result (operator request)
- Family 1 fixed-TP part widened to the full grid: **TP {2, 3, 4, 5, 6 %} (maker, through by ≥ 1 tick; no late TP, no runner) × SL {4, 6, 8, 10, 12 %} × hold {90 m, 4 h, 12 h, 24 h, 48 h}** = 125 cells, plus TODAY-TP × the same SL × hold = 25 cells. (This replaces "fixed 3 %, fixed 5 %".)
- Shown per sleeve as a heat-map table (avg net %/trade, WR, breakeven WR) for holds 4 h / 12 h / 24 h, the per-period table for the top cells, and the plateau check of the best cell.
- All these cells enter the family-wise null and the DESIGN choice. The HT SL −10 / TODAY-TP (+2 % floor) headline row stays first.
