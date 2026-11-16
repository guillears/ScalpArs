# Momentum LONG exit — today's trail vs fixed take-profit (2026-11-16)
Operator: "run the momentum long exit study, trail vs fixed TP". Mirror of reports/FAN_FLIP_EXIT_STUDY_2026-11-16.md, with live costs (CLAUDE.md § Live-mode lessons).

## Today's live momentum-long exit (code + trading_config.json)
- **Stop:** −0.70 (−1.00 while the 5m signal is still on), widened to −1.5 × entry ATR %, widening capped at −1.20. Recovery hold armed (BTC RSI 60–66 premise).
- **Runner trail:** arms at peak +0.40; line = peak − 1.0 × entry ATR %, never below +0.10 (BE lock, ratchet on).
- **Hard-TP ladder:** 1.25 → floor 1.00 · 1.5 → 1.20 · 2.0 → 1.60 · 3.0 → 2.40 · 4.0 → 3.20.
- **ML second chance (DL 532):** when ADX was falling at entry, a runner / ladder exit is held (floor L − 0.30, trail 1 × ATR, 180 min clock).
- **Time:** NO_EXPANSION 180 min, max hold 1,200 min. EMA13 long exit, fast exit and BE levels are OFF. All exits are market (taker) orders.

## Data and method
- Cohort: master momentum longs, kept, full-size, after every filter live today (ML_WEAK_RESCUE + ML_WEAK_MARKET + ML_FADING_QUIET): 93 fills (BASE 20 · Jul–Oct B1–B21 42 · live B22–B23 10 · B25 20 · B26 1) + live B27 8 = **101 fills**.
- Walker: the validated ML exit replica (scripts/ml_exit_optimize_yr5.py, today's stack = RP.BASE + the DL 532 second chance via scout_ml_matched_refusals), ticks where archived, 1m bars elsewhere (26 top-up requests). Agreement with booked: mean |Δ| 0.19 pp, 78 % within 0.25, bias +0.08 (the replica is a little kinder than live).
- Variants (13 tested — multiple-testing caveat): pure fixed resting TP +0.3 / 0.4 / 0.5 / 0.6 / 0.8 / 1.0 (no trail, no ladder; today's stop, recovery hold and clocks), and hybrids (TP at X on top of today's trail / ladder / second chance).
- **Live costs:** a resting TP books exactly net +X with the maker fee; every market exit (trail, ladder, stop, time) pays the taker fee + live momentum-long exit slippage (26 live ML market exits: mean +0.013 %, median +0.009 %, max +0.105 %) — base case 0.02 %, stress 0.10 %.
- Two-sided check: the 17 fills stopped under today book the identical stop under every TP variant (Δ = 0.00 ✓).
- One-size DCR at today's 10× ticket: X = 2.85 (live ML notional at 15× $32.8k → 10× ≈ $21.9k on ≈ $7.65k).

## Path decomposition (booked, entry label +1 % before the stop)
| Group | N | avg % | Σ % |
|---|---|---|---|
| CAPTURED | 52 | +0.59 | +30.6 |
| LEFT_ON_TABLE (avg 1.75 pts left) | 29 | +0.56 | +16.2 |
| NEVER_WORKED | 17 | −0.75 | **−12.8** |
| STOPPED_RECOVERED | 3 | −0.96 | −2.9 |
| GAVE_BACK (worked, still lost) | 0 | — | — |
→ The loss side is the ENTRY. The trail never turned a trade that worked into a loss; winners run past +1 % often.

## Results (base cost 0.02 %) — WR · avg % · Δ vs today · day-boot 95 % CI · DCR %/day
| Exit | ALL 101 | Aug → live 71 | B25 + B26 + live 29 |
|---|---|---|---|
| **Today (trail)** | **76 % · +0.37 · DCR +0.65** | **72 % · +0.31 · +0.52** | **72 % · +0.43 · +1.39** |
| TP +0.3 | 81 % · +0.10 · −0.27 (−0.50…−0.10) | 77 % · +0.06 · −0.25 | 72 % · +0.00 · −0.42 |
| TP +0.4 | 79 % · +0.15 · −0.22 (−0.44…−0.05) | 75 % · +0.10 · −0.21 | 72 % · +0.08 · −0.35 |
| TP +0.5 | 74 % · +0.17 · −0.20 (−0.42…−0.02) · DCR +0.31 | 70 % · +0.12 · −0.19 | 69 % · +0.11 · −0.32 |
| TP +0.6 | 70 % · +0.19 · −0.18 (−0.40…−0.00) | 66 % · +0.13 · −0.18 | 66 % · +0.12 · −0.31 |
| TP +0.8 | 65 % · +0.24 · −0.12 (−0.33…+0.05) | 63 % · +0.22 · −0.09 | 59 % · +0.15 · −0.28 |
| TP +1.0 | 60 % · +0.27 · −0.09 (−0.33…+0.08) | 58 % · +0.22 · −0.09 | 52 % · +0.13 · −0.30 |
| Hybrid TP +0.5 | 79 % · +0.17 · −0.19 | 75 % · +0.14 · −0.17 | 72 % · +0.13 · −0.29 |
| Hybrid TP +0.8 | 78 % · +0.25 · −0.12 | 75 % · +0.21 · −0.10 | 72 % · +0.18 · −0.25 |
| Hybrid TP +1.0 | 77 % · +0.29 · −0.08 | 73 % · +0.23 · −0.08 | 72 % · +0.16 · −0.26 |
How they end (ALL): today trail 80 / stop 21 · TP +0.5 TP 74 / stop 27 · TP +1.0 TP 58 / stop 40 / time 3.

**Stress cost 0.10 % on every market exit:** today 58 % · +0.29 (DCR +0.50) vs TP +0.5 74 % · +0.15 (Δ −0.14) · TP +0.8 65 % · +0.22 (Δ −0.07) · TP +1.0 60 % · +0.24 (Δ −0.05) — the trail still wins on average; the TP's win rate is higher only because the trail's small wins turn ≤ 0 after costs.

**Per era (avg %, base cost): today → TP +0.5 · TP +0.8**
| Era | N | Today | TP +0.5 | TP +0.8 |
|---|---|---|---|---|
| BASE Jun–Jul | 20 | +0.58 (90 %) | +0.38 | +0.47 |
| Jul–Oct B1–B21 | 42 | +0.33 (76 %) | +0.16 | +0.22 |
| Live B22–B23 | 10 | −0.08 (60 %) | −0.02 | **+0.16** |
| B25 | 20 | +0.84 (90 %) | +0.31 | +0.41 |
| B26 | 1 | +0.04 | +0.50 | −0.80 |
| Live B27 | 8 | −0.56 (25 %) | −0.45 | −0.38 |
Live B27: TP +0.5 changes only INJ (+0.06 → +0.50) and XLM (+0.15 → +0.50); the 6 losers never reached +0.5 (5 peaked ≤ +0.09, AAVE +0.24).

## Backtest cross-check (yr6, 2 years, upper bound for the TP: TP books when the recorded peak reached it)
Seed 1 today 1,458 · 57 % · −0.131 → TP +0.5 −0.118 · TP +1.0 −0.126 · seed 2 −0.147 → −0.155 / −0.144 · seed 3 −0.138 → −0.130 / −0.133. Flat — even the TP's optimistic bound does not separate.

## Verdict
- **Keep today's trail for momentum longs.** Every fixed TP and every hybrid is WORSE on average in every window except the two live losing batches (B22–B23, B27): ALL Δ −0.09 … −0.27 per fill, CI excludes 0 for TP ≤ +0.6, LOO agrees; B25 (−0.43 … −0.56) shows why — the trail lets real runners go (29 LEFT_ON_TABLE fills still averaged +0.56, CAPTURED +0.59).
- **Unlike FAN flips, slippage does not flip it:** momentum longs trade liquid coins (live ML exit slippage mean 0.013 %); even at a 0.10 % stress the trail keeps a +0.05 … +0.14 edge.
- **TP raises the win rate only at small sizes** (+0.3 / +0.4: 79–81 % vs 76 %), by capping winners at +0.3–0.4 while the trail averages +0.6 — a lower expectancy. Not a ship.
- The live B22–B23 / B27 losses are ENTRY losses (NEVER_WORKED, peaks ≤ +0.24) — today's filters (DL 559) target exactly those; no exit change helps them.
- No observe line proposed (no variant beats today in any non-live window).
