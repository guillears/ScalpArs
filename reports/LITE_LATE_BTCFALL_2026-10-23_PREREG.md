# LITE late × BTC falling — PRE-REGISTRATION (written 2026-10-23 before any outcome of this study was computed)

Already known before writing (prior studies, read first, not re-fitted here):
- LITE_BTC1H_FALLING (DL 288, frozen observe line): slope ≤ 0 — yr6 at +3/−3: 2025 −0.674 · 2026-H1 **+0.321 (reverses)** · 2026-H2 −0.495.
- YR6_LITE_LATE_ADX (2026-10-13): hours ≥ 12 at +3/−3: ALL 297 · −0.063 vs +0.017 (p 0.36) → late alone refuted at +3.
- FRENZY_WL_SCREEN 2026-10-19: "frenzy hours ≤ 7" refuted (1/3 regimes). FRENZY_REALFILLS E2: real 6–16.75 h LITE 19 · −0.51 (descriptive).
- LITE C (DL 301) already sends every BTC-slope ≤ 0 fill to TP +2 — the falling cohort is re-priced at +2 here.

## Theory (mechanism) — written before results
LITE buys a FRENZY coin that has held ≥ 12 closes above its spike VWAP, betting on a second leg within 12 h.
1. **Late:** speculative inflow decays with time since the spike (attention half-life). At 10–16 h, holding above VWAP is more
   often low-volume drift than fresh demand; the early buyers are sellers into strength → a +2/+3 leg is less likely.
2. **BTC falling:** small caps carry >1 beta to BTC on the downside; a falling BTC trend lowers risk appetite and the −3 stop is
   reached more easily.
3. **Interaction (the operator's claim):** early in the episode the coin's own flow dominates its beta; late, the flow has faded,
   so BTC direction dominates → BTC falling should hurt MORE when late. Prediction: zone (late ∧ falling) < both single axes, and the
   late-minus-early gap is larger on falling than on rising BTC.
**Refuted if:** the combo is not worse than rest (or not worse than each axis alone); the sign flips across BTC measures (only one
definition works = measurement noise); it reverses in a period (2025 / 2026-H1 / 2026-H2); it fails 2025→2026 OOS or the day-shift
null; or it is concentrated in few days/pairs (market-wide variable → DAY units).

## Cohort / pricing
- ③ BT: yr6 2Y_TODAY engine replay FRENZY_LITE (real engine code; bearish-day block on), seeds 1/2/3, priced by the LITE timecap
  tick/1m walker at TODAY's exit (LITE C TP +3 iff vol ≥ 20 ∧ slope > 0 else +2 · −3 · 12 h · resting maker TP · 0.10 slip on market
  exits). Seed 1 = analysis, seeds 2/3 = recomputation. FRENZY_LITE_RETRY = the deep-dive S15r replica (shipped rule), shown apart.
- ② master: MASTER_POOL_stacked kept non-probe LITE + LITE_RETRY, `stack_pct` (today's stack). ① live: post-go-live export fills.
- Window = UTC day. $ ruler: LITE notional $1,900 (8×, AVA −2.85 % = −$54), RETRY $950 (lev 0.2).

## Variables (sign first, cut 0, never fitted)
- LATE: entry_frenzy_hours ≥ H, H ∈ {6, 9, 12}; **primary H = 12** (the earlier pre-declared cut; AVA 14.9 h).
- BTC falling: S1 `entry_btc_1h_slope` ≤ 0 (**primary**, engine stamp) · S2 BTC 15m return < 0 · S3 BTC 1h return < 0 · S4 BTC 4h
  return < 0 · S5 BTC price < 1h EMA20 (forming hour) · S6 BTC 1h −DI > +DI (14, closed hours) · S7 `entry_btc_ema20_slope` (5m) < 0.
- Coin (axis-alone only): C1 last close < 5m EMA50 · C2 off-peak (high since spike → entry) ≥ 8 %.
- Buckets after sign (S1 and hours only): terciles of the BT seed-1 feature distribution (outcome-blind).
- **Family:** 3 hour-alone + 7 BTC-alone + 2 coin-alone + 21 combos (H × S) = 33 tests. Primary = H12 ∧ S1.

## Decision rules (frozen)
- Zone vs rest: Δ = zone avg − rest avg (%/fill), day-clustered bootstrap (4,000, seed 7).
- FILTER candidate only if the zone passes the CLAUDE.md expectancy bar on BT seed 1 all periods (WR < LITE breakeven from today's
  kept fills · P(avg < 0) ≥ 0.95 day-clustered · ≥ 8 days · no day/pair ≥ 50 % of the loss · N ≥ 15) AND avg < 0 in each of 2025 /
  2026-H1 / 2026-H2 AND 2025→2026 OOS (2026 zone avg < 0 with P ≥ 0.90) AND family-wise day-shift null p < 0.05 (BTC series shifted by
  whole days, circular, ≥ 7 d; statistic = min Δ over the family's BTC-dependent tests) AND seeds 2/3 agree in sign.
- ROUTE (smaller size) candidate: zone avg < rest avg with P ≥ 0.90 in all three periods but not proven negative → observe line.
- Otherwise: nothing (or an observe line if real fills point the other way and N is tiny).
- Any survivor: leave-one-month-out + 30–50 % haircut on the counterfactual Δ (winners blocked priced in).
