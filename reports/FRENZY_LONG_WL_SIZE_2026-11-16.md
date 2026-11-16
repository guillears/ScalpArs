# FRENZY_LONG — master check, winners vs losers, full-size question (2026-11-16)
Operator: "Frenzy Long is correct? 25 and 76%? make a deep dive of winners vs losers, to think if we can change it to full size".

## 1. The master number — confirmed, with two caveats
- MASTER_POOL_stacked FRENZY_LONG: 52 rows · 25 kept · **25 · 76 % · +0.44 %/fill** (19 × +1.50, 6 × −3.0) · 16 days · no pair > 2 fills. Not probe-flagged (is_probe False on all 52); blocked: LONG_WEAK_FRENZY 24 (DL 525) · FRENZY_SLEEVE / BEARISH_DAY 3.
- Live config: `frenzy_long_invest_mult` 0.2 × `frenzy_long_lev_mult` 0.05 (= 1×) → blocker size (≈ $360 notional, a stop ≈ −$11; B27 SKR −$14). TP +1.5 resting maker, stop −3 (DL 524 reverted −4 → −3 after 1 / 10 recoveries; live −4 cohort was 19 · 47 % · −1.40).
- Caveat A — **re-pricing:** 7 of the 19 "wins" were BOOKED as −3 stops under the old +3 TP (UMA, ERA ×2, CTSI, GTC, RSR, …) and 3 booked +3 wins (CFX, RLC, RSR) — all re-priced to today's +1.5 because their peak touched +1.5 first. Sound logic, but as-booked the cohort is 13 / 25.
- Caveat B — **in-sample:** LONG_WEAK_FRENZY (vol multiple ≤ 119×) was fitted on these same fills (blocked 24); the kept 25 are the survivors of that fit.

## 2. Winners vs losers (6 losers — too thin for any separator)
- Path label (sweep_separators FRENZY_LONG --label path, 26 real fills incl. live SKR): WORKED 20 · NEVER 6. Blind spots (< 80 % scored): funding, book, alt beta, mkt_up4h, pair age, 4h gap, signed gaps.
- Best 1D (both eras): pair volume ratio top tercile worse (−0.50 / −0.36), BTC off-24h-low above median worse, global volume above median worse, BTC ADX above median better. Losers' medians: pair-vol 0.85 vs 0.61 · global vol 1.03 vs 0.83 · BTC ADX 26.5 vs 31.4. Best 2D: BTC ADX ≤ med × BTC off-24h-low > med (−0.40 / −0.57). 0 consistent 3D cells. With 6 losers on 6 days every cell is noise → **no filter**; pair-vol top tercile = observe idea only.
- Clustering: other Frenzy / Flatbase fills within ±60 min of a FRENZY_LONG: mean 0.2, max 1 (20 % of fills) — low. Worst master day −3.02 pts (BMT 11-03).

## 3. Size by edge (DL 547)
| | WR | 95 % floor | breakeven (+1.5 / −3) | Kelly | Kelly at floor |
|---|---|---|---|---|---|
| master 25 | 76.0 % | 59.9 % | 66.7 % | +0.28 | **−0.20** |
| master + live 26 | 73.1 % | 57.1 % | 66.7 % | +0.19 | **−0.29** |
| backtest yr6 (today's exit approx., LONG_WEAK_FRENZY on) | 64–65 % per seed (≈ 275 fills each) | — | 66.7 % | — | below breakeven, −0.08 %/fill every seed |

| Size | notional | one stop | % of a $7.65k account | one-size DCR (its 40 live days) |
|---|---|---|---|---|
| today 0.2 × 1× | ≈ $360 | ≈ −$11 | 0.14 % | ≈ +0.01 %/day |
| LITE-like 0.85 × 8× | ≈ $12.2k | ≈ −$370 | ≈ 4.8 % | ≈ +0.4 %/day (master, in-sample) |

## 4. Recommendation — do NOT go to full size yet
- Observed 76 % sits just 9 points above the 67 % breakeven of a +1.5 / −3 exit; the 95 % floor is BELOW breakeven (Kelly at floor < 0); the 2-year backtest is below breakeven in all 3 seeds; the live −4 history was 47 %; the master figure is in-sample and leans on re-priced stops.
- Step plan instead (pre-committed): keep 0.2 × 1× until **15 live fills on today's stack (+1.5 / −3, LONG_WEAK_FRENZY on)** → if WR ≥ 75 % and avg > 0, step to **0.5 × 5×** (≈ $4.5k notional, a stop ≈ 1.8 % of the account); another 15 at WR ≥ 75 % → LITE size. Any 15-fill read below 67 % → back to blocker size. No filter now; watch pair-volume-ratio top tercile.
