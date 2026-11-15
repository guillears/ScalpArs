# Momentum Long exit review — 2026-11-15 (operator: "the momentum longs new exit, i think is costing us money")

Data: B26 (paper, 11-11 → 11-14, `reports/BASELINE26_batch1111-1114_orders.csv`) + live B27 so far (CloudWatch logs, 11-15 13:23–13:57 UTC) + master (MASTER_POOL_stacked, today's stack, kept full-size) for context. Price paths: Binance 1m klines.

## The "new exit" = ML SECOND CHANCE (DL 532, armed 11-11)
Zone: momentum long with ADX falling at entry (`entry_adx_delta` ≤ 0). When the runner trail / hard-TP ladder would close at L, the trade is held; exits SC_FLOOR (L − 0.30) · SC_TRAIL (post-trigger high − 1 × ATR) · SC_CLOCK (180 min).

| Cohort | ML fills | in SC zone / exits changed by SC | booked |
|---|---|---|---|
| B26 (paper) | 1 (ICP) | 0 (ADX rising, Δ +0.39) | RUNNER_TRAIL +0.25 % · +$93 |
| Live 11-15 | 3 (INJ, AVAX, ARB) | 0 (no SC_* exit, no hold) | INJ RUNNER_TRAIL +$9.77 · AVAX STOP −$246 · ARB STOP −$188 |
| scout_ml_second_chance.py | — | **0 of 15 changed exits** | gate OBSERVING |

**SC has never changed a single exit since it shipped → it has cost (or made) $0.** The losses are elsewhere.

## Where today's −$424 (3 live ML) came from
| Trade | Path label | Peak | Booked | After the exit |
|---|---|---|---|---|
| AVAX $33.5k @ 15× | NEVER_WORKED | +0.00 % | −0.73 % → −$246 (fees $30) | fell to −1.20 % (stop saved ≈ $160) |
| ARB $24.4k (risk-capped) | NEVER_WORKED | +0.01 % | −0.77 % → −$188 | fell to −1.18 % |
| INJ $33.5k | GAVE_BACK | +0.63 % | +0.09 % → +$9.77 (raw +$39.90, fees $30.13) | ran to +1.83 % (13:49) then −1.71 % (14:00) |

- The larger realised side is the ENTRY: 2 of 3 never went green (−$434). Both entered within 60 s of each other on the same market move and stopped 2 min apart → one market window, not two independent losses.
- INJ's exit is the RUNNER TRAIL (arm +0.40, floor +0.10) — the long-standing exit, not SC. Holding it would have caught +1.83 % only to see it fall to −1.71 % 11 min later; the trail locked a small win before a −1.7 % reversal.
- Fees: at 15× one ticket ≈ $33.5k notional → ≈ $30 per round trip (0.09 %) — 75 % of INJ's gross gain. 2 of 3 entries fell back to a market order after the 20-s maker wait (taker fee + 20 s drift).

## Master context (124 kept ML fills · 77 % · +0.24 %/fill)
- Exits: RUNNER_TRAIL 79 (avg booked +0.47 % from an avg peak +0.89 %) · STOP_LOSS L1 23 · ladder / trailing 11.
- 18 fills never reached +0.10 % → avg −0.81 % (the stop cohort = the entry side); 95 reached ≥ +0.40 % → avg +0.56 %.

## Recommendation
1. **No exit change.** SC has zero changed exits — nothing to judge, and its pre-committed gate (15 changed exits) stays. The runner trail is the master's profit engine (79 exits, +0.47 % avg). Today's ML loss is an entry / clustering outcome, not an exit one.
2. **The real cost to watch is clustering at 15×:** three ML longs opened inside 60 s (≈ 3 × 3.6 × the account in notional); the DL 533 risk cap held them to ≈ 10 % worst-case, and that is what the account took (−$424 ≈ −4.5 %). This belongs to the size-by-edge question (DL 547), not the exit.
3. Next read at the batch review with more live ML fills: per-fill path labels (NEVER_WORKED vs GAVE_BACK), maker-vs-taker entry cost, and the SC gate if it ever fires.

# Part 2 — Momentum Long winners vs losers on today's filters (operator 11-15 evening)
Master 124 kept · 77 % · +0.24 % (breakeven WR ≈ 59 %) · current batch B27 live 3 · B26 1 (in master) · backtest yr6 2-year replay (Oct-13 stack, older exits).
- Sweep (scripts/sweep_separators.py MOMENTUM:LONG, 45 dims × 3 granularities + all pairs + 3D): best 1D beats a 200-shuffle null (0.53 vs median 0.31, p 0.005): EMA gap (entry_gap), pair EMA20 slope, BTC RSI 6 bars earlier.
- Single filters (block weak trend gap ≤ 0.27 · flat EMA20 ≤ 0.134 · BTC RSI6 > 58.6): blocked cohorts ≈ break-even (+0.04 / +0.05 / −0.00), master Σ +30.1 → +28.3 / +27.0 / +30.1 → NOT filters (P(mean<0) 0.34 / 0.31 / 0.49).
- 2D inside weak trend (all variables, both eras, 300-shuffle null p 0.05): best cell WEAK × entry_bear_pct > 20.65 → 21 · 57 % · −0.18 (18 days, 16 pairs) vs rest 103 · 81 % · +0.33; master Σ +30.08 → +33.92; day-bootstrap P(mean<0) 0.88 (fails the 0.95 bar only). Live B27: ARB (gap 0.27, bear 30.4) in the cell, lost −0.77 %; AVAX / INJ outside. Runner-up WEAK × global volume ≤ 0.709 (12 days, P 0.91).
- Backtest: momentum long −0.13 %/fill overall on the Oct-13 stack; no zone separates in any seed (WEAK×BEAR −0.06 / −0.06 / −0.11 vs rest −0.14) → refutation stated as a caveat (master over backtest).
- Theory: ML buys continuation of a 5 m uptrend; a small EMA gap = a young / weak trend that needs a market tailwind; > 20 % of the scanned coins bearish = no rising tide → the weak trend fizzles (NEVER_WORKED). Coherent.
- Verdict under the locked bar: meets N ≥ 15, ≥ 8 windows, WR below breakeven, no pair ≥ 50 %; fails 95 % confidence → pre-registered OBSERVE candidate with frozen cuts (gap ≤ 0.27 ∧ bear_pct > 20.65), never re-fit.
