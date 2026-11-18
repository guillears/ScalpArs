# EXPAND EDGES — pre-registration (frozen 2026-11-18 16:35 UTC, before any result of this study)

Operator: "We need MORE TRADES PER DAY AT THE SAME QUALITY." Research only. No bot change, no commit.

## Theory
The two proven edges (FLATBASE_SHORT, EXHAUSTION_SHORT) live in their CONDITIONS (flat base + pump without follow-through;
exhaustion while volume is still high and spot leaves). Their UNIVERSE gates (spike size / volume multiple, the verified +
shortlist selection, the $20 M 24 h floor, the 72 / 96 h windows, the flatness cut) were set for other reasons. Relaxing ONE universe
gate at a time, with every edge condition kept exactly, may add trades at similar quality.

## Engines (reuse, parity first)
- **Flatbase:** `scripts/high_atr_flatbase.py` worker (the bot's own `services.frenzy` / `services.flatbase` code, today's
  trading_config.json, the HIGH_ATR 5m store 2023-05 → 2026-11-17, delisted included) + `scripts/high_atr_fb_eval.py` pipeline
  (turnover route proxy → NEAR_BASE 2.09 % → 4 shared slots; TP 5 / SL 10 / 72 h; flat 0.10 % slip + funding). Parity target:
  778 · 73.5 % · +0.91 %/trade. Relaxations patch the threshold namespace only.
- **Exhaustion:** `scripts/exh_expand_build.py` (unchanged code, patched window / volume floor / one extra exit) on bar-rule spike
  lists (`exh_expand_realis.spikes_one` logic, no verification / shortlist — the EXHAUSTION_EXPAND §0 bias rule). Rule: 15m RSI14
  70 → ≤ 70, vd ≥ 0.6366, spot_share_chg < −0.0066, vol24 ≥ floor, one position per pair (re-entry after any exit = today's engine).
  Exit = shipped B|tp10|S2|24: TP min(20 × ATR15, 10 %) resting maker (≥ 1 tick through), SL min(1.5 × ATR15, 6 %), 24 h, 5m bars,
  stop-first. Costs: taker 0.05 %, maker 0.02 %, live slip model at $5k notional on every taker leg, real funding (missing month →
  the builder's conservative penalty). Parity target: EXH_EXIT_GRID B|tp10|S2|24 3,998 trades (realisable list); the bar-rule
  list is expected ≈ +3 % trades (EXHAUSTION_EXPAND §0 (3)). DL 582 (EXH_SSC_OIVOL) is applied only if OI can be attached to ≥ 90 % of
  trades; otherwise the read is pre-582 and says so.

## Relaxations (one gate at a time; base = today's rule on the same engine)
Flatbase: F1 spike ret 5 → 4 % · F2 → 3 % · F3 spike vol 20 → 15× · F4 → 10× · F5 drop verified + shortlist (bar-rule flag, any
universe coin readable) · F6 24 h volume floor $20 M → $10 M (shortlist floor and S4 floor) · F7 → $5 M · F8 S4 window 96 → 144 h ·
F9 flatness 0.444 → 0.55 (NEAR_BASE kept).
Exhaustion: X1 spike ret 5 → 4 % · X2 → 3 % · X3 vol 20 → 15× · X4 → 10× · X5 vol24 floor $20 M → $10 M · X6 → $5 M · X7 window
72 → 120 h · X8 → 168 h · X9 repeated signals per episode after a TP — NOTE (found while reading the engine before results):
`_exhaustion_track` keeps the episode TRACKING after a trade, so re-entry after an exit is ALREADY today's rule; X9 is reported as
"already in" with the share and quality of 2nd+ entries (descriptive). "Drop verified / shortlist" for Exhaustion = the bar-rule
base itself vs the realisable list (already measured in EXHAUSTION_EXPAND §0: 3.34 vs 3.25/day) — reported, not re-tested.
Best pair per sleeve: the two passing relaxations with the highest union DCR; if < 2 pass, the two with the highest added-trade avg
are combined DESCRIPTIVELY (labelled "not a pass").

## Reads (ADDED trades = in the relaxed run, not in the base run; key Flatbase (sym, spike_ts), Exhaustion (pair, entry_ms))
N, trades/day, WR, breakeven WR, avg net %, per half-year (23-H2 · 24-H1 · 24-H2 · 25-H1 · 25-H2 · 26-H1 · 26-H2), best-5 %-out,
@ flat 0.20 % slip, Kelly at the one-sided 95 % Wilson floor with n = distinct entry days (daily clustering), per-pair / per-day
share; base trades DISPLACED by the relaxation (slots / pair busy). Union: DCR at 0.18 × equity per trade (r_d = Σ net % × 0.18 / 100
over trades closed that day, every calendar day), max drawdown, worst day — base vs union.

## PASS (added trades alone, all of)
1. N ≥ 30; 2. every half-year with N ≥ 5 > 0 and ≥ 5 such halves; 3. best-5 %-out > 0; 4. > 0 at flat 0.20 % slip;
5. Kelly at the Wilson floor > 0; 6. DESIGN (2024-07 → 2026-06) > 0 AND holdouts > 0: 2026-H2 and 2023-H2 (both were read in
earlier studies of the base rules — stated, not clean); 2023-H1 only where buildable (Exhaustion E data covers today's episodes only
→ relaxations not buildable; Flatbase store starts 2023-05 → not buildable); 7. family-wise null: day-block sign-flip of the added
trades' daily mean, max-T over ALL relaxations of both sleeves, 2,000 permutations, p_fw < 0.05; 8. union DCR > base DCR and union
max DD not deeper than base by > 5 points.
Verdict: all 8 → BUILD candidate (then the 30–50 % in-sample haircut + a frozen revert gate) · fails only 7 → OBSERVE · else NO.
Thin books (F6/F7/X5/X6): also report the added trades at the live slip model with $5k notional; no order-book history exists, so the
depth guard is NOT rebuildable — the Q5-based slip model is the proxy (assumption).
