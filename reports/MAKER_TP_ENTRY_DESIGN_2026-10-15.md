# Resting take-profit (maker exit) + maker entry for the FRENZY family — design (2026-10-15)

Status: DESIGN ONLY. No code, config or study has run. Nothing ships before the Phase-1 evidence gate below passes both reviews.

## 1. Why (live facts, B-live 2026-10-14/15)

| Trade | Exit | Bot fired at | Filled at | Gap | Decision → fill |
|---|---|---|---|---|---|
| ZK WILLY | TP | +1.25 % | +1.12 % | −0.13 | 0.9 s |
| BLUR WILLY | TP | +1.32 % (jumped past +1.25) | +1.09 % | −0.22 | 1.0 s |
| DEEP LITE | TP | +3.02 % | +2.80 % | −0.22 | ~1 s |
| STRK WILLY | stop | −4.00 % | −3.79 % | +0.21 | 0.8 s |
| DUSK WILLY | stop | −4.00 % | −3.43 % | +0.57 | 2.6 s |

- The software exit fires exactly at the setting; the market order fills 0.6–2.6 s later on thin, just-spiked coins.
- Take profits fill WORSE (the price touched the level and was already falling back); stops fill BETTER (bounce).
- Entry fills vs the decision price: STRK +0.22, BLUR −0.03, ZK +0.73, DUSK +0.28, DEEP +0.13 (avg ≈ +0.27 % paid).
- Fees today: taker 0.045 % per side, maker 0.018 % (BNB discount) → a maker exit saves 0.027 % on top of the slippage.
- N = 3 take profits, 2 stops, 5 entries: direction is clear, size is not. The study below measures it on thousands of fills.

## 2. Scope — which sleeves

| Sleeve | Exit today | Resting TP? | Why |
|---|---|---|---|
| **FRENZY WILLY** | fixed +1.25 / −4 / 90 min | **YES — first** | most fills, short holds, fixed TP; every TP pays the slip |
| **FRENZY LITE** | fixed +3 / −3 | **YES — second** | same mechanism; fewer TP hits; DEEP showed the same −0.22 |
| FRENZY LONG (blocker) | fixed +1.5 | technically yes, not worth it | blocker size (0.05× lev, 0.2× inv) → cents per trade |
| Spike fades (fixed TP) | fixed TP | candidate after WILLY | tiny TPs → slip is a large share of the edge; check they are live first |
| Momentum / flip / SURGE | trailing / signal exits | **NO** | no fixed price to rest an order at (maker_exit_enabled exists for them, off) |

The code is built ONCE, generic: "any open order with a fixed TP price may carry a resting TP", switched per sleeve by config
(`resting_tp_willy_enabled`, `resting_tp_lite_enabled`, … default OFF), so each sleeve is armed only on its own evidence.

## 3. Mechanism (live)

1. **Place at open.** After the fill and the backstop: a reduce-only LIMIT sell (LONG) at the TP price, time-in-force GTX
   (post-only → always maker; Binance rejects it instead of filling as taker). TP price from the ACTUAL fill and the NET target with a
   maker exit fee (the net +1.25 is reached at a slightly lower price than with a taker exit → it fills sooner). Stored on the row:
   `protective_tp_order_id` (column already exists, unused), new `tp_limit_price`, `tp_limit_filled_qty`.
2. **Software TP stays as the safety net, changed to "through + grace":** if the price trades ≥ N ticks THROUGH the limit and the order
   is still not (fully) filled after G seconds (start N = 2, G = 3 s) → cancel the limit, market-close the rest (today's path). A
   touch that does not trade through does nothing — that is exactly the case a limit should not chase.
3. **Fill detection:** query the order only when the live price is at/through the limit (bounded API weight), plus the 60 s reconciler.
   A filled limit books the close as `FRENZY_TP` at the limit price, exit type MAKER, maker fee.
4. **Partial fills (the hard part):** price prints at the limit, fills part of the size, then reverses. The rest stays protected by the
   close-all backstop + the software stop + the time cap. Every other exit cancels the limit FIRST, re-reads the filled qty, then
   market-closes ONLY the remainder (reduce-only — can never flip the position). Booking = blended exit price and fees (maker part +
   taker part), new close reason `FRENZY_TP_PARTIAL+<other>`. Fully unit-tested before anything is armed.
5. **Every close path cancels the resting TP first** (stop, time cap, manual close, untracked flatten, booking-failure flatten), like the
   backstop pre-close cancel. A cancel that fails because it just filled → treated as a fill, never as an error.
6. **Orphans / restarts / config changes:** the reconciler also clears regular (non-algo) open orders on symbols with no OPEN row; on
   startup the heal sweep re-places missing resting TPs; a TP setting change re-prices them through the DL 292 re-price sweep
   (cancel → place, crossed-price guard).
7. **Paper = live:** paper books the TP at the limit price, maker fee, ONLY when the tracked price trades ≥ 1 tick THROUGH it (a touch
   does not fill — conservative), full size (partial fills are not modelled: stated as a known optimism, measured live).

## 4. Risks

- **Missed fills (adverse selection):** the trades that touch +1.25 and reverse without trading through are exactly the ones a market
  order rescues today. With a limit they continue — some to the stop. This can cancel the whole gain; only ticks can measure it.
- Queue position on thin books (a print AT the price may not reach us) → paper rule requires trade-THROUGH.
- More moving parts on every close path → each path is unit-tested; the build ships with every sleeve switch OFF.

## 5. Maker ENTRY for FRENZY (question 2)

- Momentum / flip entries already use maker (`maker_entry_enabled` true: 20 s limit at best bid − 1 tick, taker fallback).
- FRENZY / SURGE / spike entries deliberately bypass it: spike fires were 18/18 taker fallbacks (the price runs away from a passive
  bid) and the 2026-10-07 latency study found entry delay 1–13 s is flat in expectation.
- The real risk is **selection**, not drift: a resting buy fills only when the price comes BACK to it → you get the trades that dip
  first (more of them losers) and miss the ones that run straight up (the winners). A study must price exactly that.
- Candidate variants (same study): limit at the decision price / at best bid, timeout 5 / 10 / 20 s, unfilled → skip vs taker.
- Today's live entry cost (+0.27 % avg, ZK +0.73 %) is also checked against `frenzy_max_entry_dislocation_pct` (the guard that should
  refuse a far-away book).

## 6. Phase plan and gates

**Phase 1 — tick study (cache-only, no Binance calls, ~1 day).** Cohorts: WILLY and LITE fills in yr6 2025 + 2026 replays (today's
config), every master batch, and the live fills. For every fill, on ticks (1m only where ticks are missing, flagged):
- A (today): market exit at touch + latency L ∈ {0.8, 1.5, 2.6 s} (live-measured), taker fee.
- B (resting TP): fills iff a print trades ≥ 1 tick through the limit before the stop / time cap; else the trade runs to its own exit;
  maker fee. Variant with the through + grace safety net (N = 2 ticks, G = 3 s).
- Entry: maker variants of §5 vs today's taker at the measured ~8 s.
- Report per sleeve: Δ %/fill, fill rate, missed-fill cost, per-day bootstrap CI — **three blocks** (live batch · master per batch ·
  backtest per year). Pre-registered gate: **Δ > 0 with P ≥ 0.90 in BOTH backtest years, master Δ not negative, and the missed-fill
  cost reported separately.** Fails → not built.
**Phase 2 — build** behind per-sleeve switches (default OFF) + an observe-only shadow: on every live TP the bot logs what a resting
limit would have done (filled / missed, price) — real queue data at zero risk. Caveman + deep review, full tests.
**Phase 3 — arm WILLY** after Phase 1 passes AND ≥ 15 live WILLY TPs in the shadow agree with the study (no arm before review).
Revert gate: after 20 armed TPs, live Δ vs the shadow's market-exit counterfactual < 0 → OFF. LITE follows on its own evidence.

## 7. Profit lock after a gain (operator 2026-10-15: "DUSK reached +1.15 % then went all the way to −4")

Never tested for WILLY (past WILLY exit work: TP size, stop size, time caps, EMA exits — B19_REVIEW_WILLY §1). Rule family: once the
peak reaches **A**, the stop moves up to **L** (exit at market when the price falls back to L).

- **Trade-off (two-sided — CLAUDE.md stop-width rule):** it SAVES a trade like DUSK (−4 → about L) and KILLS a winner that dips back to L
  after A and would still have reached +1.25 (+1.25 → about L). At L = 0 one saved loser (+4) pays for ~3 killed winners (−1.25 each),
  so the test is the ratio on real paths, not intuition.
- **Pitfall:** A = +1.0 sits only 0.25 below the +1.25 TP — on these coins that is ~1 second of noise; and the lock exits at MARKET on a
  falling price (live stops filled 0.2–0.6 % away from the trigger), so L = 0 really books ≈ −0.2 … −0.5. The study prices that.
- **Grid (pre-registered, multiplicity flagged):** A ∈ {+0.6, +0.8, +1.0} × L ∈ {−1.0, 0.0, +0.3}; LITE analogue A ∈ {+1.5, +2.0} ×
  L ∈ {0.0, +0.5}.
- **Method:** tick paths (the WILLY walker of scripts/scout_frenzy_willy_exit.py), exit at L + the measured live stop slip; report the
  SAVED (would-be stops / time-cap losses rescued) and KILLED (would-be TPs cut) counts and $ separately, Δ %/fill, day-bootstrap,
  three blocks (live · master per batch · yr6 2025 + 2026). Gate: Δ > 0 with P ≥ 0.90 in both years, master not negative,
  leave-one-month-out stable → then observe-only Scout line first (no arm before review).
- Live so far: DUSK peaked +1.15 → −4 (saved by any A ≤ +1.0); STRK peaked +0.77 → −4 (saved only by A ≤ +0.6); ZK / BLUR went
  straight to TP (no dip below +0.3 after A? to be read on ticks).
