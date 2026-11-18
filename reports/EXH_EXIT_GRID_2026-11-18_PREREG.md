# EXH EXIT GRID — pre-registration (frozen 2026-11-18 12:40 UTC, before any exit-grid result)

Research only. No bot / config / test change. Seen so far: only the REF parity check (shipped exit, numbers already published in
EXHAUSTION_EXPAND_2026-11-18.md): realisable REF N 4,015 · 3.25/day · +0.458 %/trade · 23-H2 +0.59 · halves +0.59/+0.47/+0.46/+0.21/+0.55/+0.13/+0.84
(matches the published (2) row). No other exit has been computed.

## Question
EXHAUSTION_SHORT exits at TP = min(3×ATR15, 5 %) resting maker, SL = min(1.5×ATR15, 5 %), 24 h. Would a different TP, a different stop,
or a trailing exit earn more? (No split / partial exit — operator excluded it.)

## Entries (frozen)
Exactly the shipped rule, realisable version (2): all.pkl rows (scripts/exh_expand_build.py) with the `sl` realisability mask
(entry ≥ first shortlist ∧ ≤ flag end), Frenzy frame, 15m bar close, vd ≥ 0.6366, spot_share_chg < −0.0066, RSI14 15m cross 70 → ≤ 70,
entry = next 5m open. One position per pair, re-applied per cell (a longer exit can block a later signal on the same pair).

## Price path
1m futures klines from data.binance.vision (daily zips, entry day → +2 days). Entry price = the 5m open (= 1m open at the entry minute).
If a pair-day of 1m is missing, that trade falls back to the 5m path (stop-first when a bar touches both) and is counted/flagged.
Within a 1m bar: the stop (initial or trailing, as it stood at the bar's open) is checked first on the bar high; then the maker TP on the
bar low (must trade THROUGH by ≥ 1 tick); a bar touching both = stop. The trailing best price updates from the bar low AFTER the checks
and the new trail level applies from the next bar. Stops fill at max(stop, bar open) (gap-aware).

## Cells
Initial stop variants used in A, B and C (operator addendum, coordinator message ~12:35 UTC, included before results):
S1 = min(1.5×ATR, 5 %) (shipped) · S2 = min(1.5×ATR, 6 %) · S3 = fixed 5 % · S4 = fixed 6 %. Family B additionally keeps fixed 4 %.
- **A. ATR TP:** TP = k×ATR, k ∈ {1, 1.5, 2, 2.5, 3}, cap ∈ {none, 5 %, 8 %} × stop {S1..S4} → 60 cells. Shipped = A k3 cap5 S1.
- **B. Fixed TP:** TP ∈ {4, 5, 6, 7, 8, 10} % × stop {fixed 4, fixed 5, fixed 6, S1, S2} → 30 cells.
- **C. Trailing:** arm when the short is +a % (best low ≤ entry × (1 − a/100)), a ∈ {2, 3, 4, 5}; trail b behind the best low,
  b ∈ {1, 1.5, 2, 3} % or {0.5, 1}×ATR %; initial stop {S1..S4}; maker TP cap {none (C), 10 %, 15 % (C2)} → 4×6×4×3 = 288 cells.
  The trail level = best × (1 + b/100); the active stop = min(initial, trail). Trailing / stop / time exits are TAKER market.
- **D. Hold:** 12 h / 24 h / 48 h for the top-3 cells (by the DESIGN selection metric) of each family A, B, C → ≤ 18 extra cells.
  All A/B/C cells use 24 h.
Total pre-registered cells: 378 + ≤ 18 = ≤ 396 (all reported in _cells.csv).

## Costs (binding)
Entry taker 0.05 % + slip · TP maker 0.02 % no slip · stop / trail / time exit taker 0.05 % + slip. Slip = the live model
max(0, −0.042 + 0.276·√(notional ÷ Q5)) at $5k (PRIMARY: ≈ the shipped 0.5 × equity notional on $10k) and at $1k; flat 0.05 / 0.10 / 0.20
per taker leg as stress. Real funding from entry to each cell's own exit (8 h and hourly coins, missing months = −0.05 %/8 h penalty).

## Blocks
E = 2023-01 → 2023-06 (NEW, never read, built with the same walk / build pipeline if the data builds; read ONCE at the end) ·
D 2023-H2 · C 2024-01→05 · A 2024-06→12 · DESIGN 2025-01 → 2026-06 · B 2026-07 → 11-17. Every block except E was read before for the
shipped exit (robustness reads on seen data for the ENTRY; the exit cells themselves are new on every block).

## Selection (DESIGN only)
Metric = one-size daily compound return on DESIGN (X = 0.5 notional ÷ equity; r_d = Σ net % × 0.5 ÷ 100 over fills CLOSED that day; every
calendar day). Plateau-smoothed pick: the cell with the best mean of (itself + its immediate grid neighbours) in its own family;
the overall pick = the best plateau-smoothed cell across families. Its neighbours are shown raw.

## Change gate (all must hold to say CHANGE THE EXIT)
1. DESIGN Δ one-size DCR vs shipped > 0 and the family-wise max-T sign-flip null (day blocks, all cells, same flips across cells,
   2,000 draws) p_fw < 0.05.
2. Δ avg net %/trade > 0 in ≥ 6 of 7 half-years (23-H2 … 26-H2) and > 0 in holdout E (if built).
3. Δ > 0 at flat 0.20 % slip and with the best 5 % of trades removed (both on the full pool).
4. Plateau: ≥ 75 % of the pick's immediate neighbours have DESIGN Δ > 0.
5. The 30–50 % in-sample haircut on the DESIGN Δ leaves Δ > 0 (half of the Δ reported as the planning number).
Positive but failing 1 or 2 → OBSERVE (paper shadow line, frozen). Otherwise KEEP.

## Stop dimension (operator addendum)
Own table: same TP, stop S1 / S2 / S3 / S4. For S3 (5 %) vs S4 (6 %): split fills stopped at 5 % into SAVED (survive at 6 %) vs DEEPER
(stopped at 6 % anyway, booking ~1 % more loss); fills not stopped at 5 % must show Δ = 0 (replica check). Two-sided, per period,
day-block bootstrap 95 % CI on the Δ. Caveat: "live-stopped" is the 1m replica's stop (paper fills too few to read live).

## Outputs per cell
N · trades/day · WR · avg win / loss · breakeven WR · avg net % · day-clustered 95 % CI · each half-year · best-5 %-removed · @0.20 flat ·
@$1k model · Kelly at the one-sided 95 % Wilson floor (n = distinct entry days) · one-size DCR · max drawdown · Δ vs shipped per period.
Left-on-the-table: for shipped TP exits, the further favourable move from the TP fill to the best low within 24 h of entry, share ≥ +10 %
total move (entry → best low). ZKPUSDT 2026-11-18 08:30 UTC replayed under the shipped and top exits.

## Addendum — selection frozen 2026-11-18 12:52 UTC (after the seen-block grid, BEFORE holdout E is simulated or read)
- Holdout E built: 2023-01 → 2023-06, 567 episodes on 150 pairs (same walk, T0 2022-12-01), 248 REF candidates on 100 pairs, 1m path for all.
- D family: top-3 of A, B and of C ∪ C2 combined (9 cells × 12 h / 48 h = 18 cells, within the ≤ 18 registered).
- Pre-registered overall pick (best plateau-smoothed DESIGN one-size DCR): **A|k2.5|cnone|S4|24** = TP 2.5 × ATR (no cap), SL fixed 6 %, 24 h.
  Family picks: A A|k2.5|cnone|S4|24 · B B|tp10|S2|24 · C C|a5|b3|S2|tpnone|24 · C2 C2|a5|b1A|S2|tp10|24.
- Gate status on seen data: 1 FAIL (p_fw 0.37) · 2 6/7 halves (E pending) · 3 @0.20 pass, best-5 %-out FAIL · 4 pass · 5 pass.
  → by the pre-registered rule the verdict cannot be CHANGE; E is read once to decide OBSERVE vs KEEP.
