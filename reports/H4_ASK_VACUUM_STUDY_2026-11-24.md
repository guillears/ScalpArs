# H4 ask vacuum — theory · winners vs losers · TP / stop / hold (2026-11-24, INTERIM, research only)

**Question (operator):** "run winners vs losers here" · "focus on theory of what the trade is about" · "different TP and SL levels? what's the optimal".
**Status:** registered before computing (PREREG Addendum 8). 4.7 days (11-19 23:23 → 11-24 16:00), one BTC regime, 261 triggers on 105 coins
(the interim cohort: final, book-priced), 1,071 matched random-minute controls. Anything found here is HOLDOUT-only (judged on data ≥ 12-07).
Code: scripts/H4_ASK_VACUUM_STUDY_2026-11-24.py · data: _grid.csv · _wl.csv · _wl_theory.csv.

## 1 · What the trade is (theory)
H4 buys a coin when the sell orders within 0.5 % of the price drop below 35 % of that coin's usual level (an "ask vacuum").
Why it could work: with few sellers resting above the price, little buying is needed to lift it — the path of least resistance is up. What
decides whether the vacuum is a real opening or noise:

| # | theory | variable | predicted winning side |
|---|---|---|---|
| T1 | one-sided: only the asks left, the bids are normal | ask ÷ bid $ within 0.5 % | LOW (HOLDOUT-only: = imbalance) |
| T2 | the asks were EATEN by real buying, not just pulled | traded $ 15 min ÷ normal | HIGH |
| T2b | buyers are the aggressors | taker-buy share 15 min | HIGH (HOLDOUT-only) |
| T3 | generic thinning: the bids are thin too (a volatility regime, no direction) | bid $ within 0.5 % | HIGH |
| T3b | makers stepping back on both sides | spread | LOW |
| T4 | sellers exhausted: the coin fell into the vacuum | return 60 min | LOW |
| T4b | late: it already ran up | return 15 min | LOW |
| T5 / T5b | market tide | BTC 60 min · breadth 60 min | HIGH |
| T6 / T6b | crowded shorts = squeeze fuel | funding · premium | LOW |
| T7 | longs flushed | long liquidations 15 min | HIGH |

## 2 · Result — theory lines (median split, three outcomes: WORKED = +1 % before −2 % in 4 h · net at TP 5 / stop 5 / 4 h · move after 4 h)
Luck bar (rotation null, family max |t| over all 63 variables): 3.53 / 4.08 / 4.48. **Best theory line |t| 2.48 — none passes.**
| theory | good − bad: worked (pp of rate) | net | 4 h move | read |
|---|---|---|---|---|
| T2b buyers aggressors | +0.08 | **+0.84** | **+1.66** | right way on all three, t ≤ 2.5 — HOLDOUT-only variable |
| T7 longs flushed | +0.04 | +0.83 | −0.06 | mixed |
| T5 BTC rising | +0.10 | −0.05 | +0.74 | mixed |
| T1 one-sided | +0.01 | −0.29 | +0.48 | mixed |
| T2 asks eaten (volume) | +0.08 | −0.08 | −0.88 | mixed |
| T3 / T4 / T5b | ≈ 0 | ≈ 0 … −0.64 | ≈ 0 | nothing |
| T4b not chased (low 15-min run-up) | **−0.13** (t 2.17) | −0.26 | +0.04 | **wrong way** on WORKED: coins already rising did better |
| **T3b tight spread** | −0.05 | **−0.90** | **−1.28** | **wrong way**: WIDE spreads did better |
| **T6 / T6b crowded shorts** | −0.01 / −0.14 | **−0.65 / −0.70** | −0.14 / −0.16 | **wrong way** |

## 3 · Result — every recorded variable (blind)
Only one variable crosses the luck bar, and only on the raw 4 h move: **BTC's own order-book depth** (thin BTC book → H4 +1.40 vs −1.47,
t −4.79 vs bar 4.48; on net 3.06 < 4.08). It is a MARKET-wide variable (window-units rule: one day = one observation): its day means line up across the 5 days (BTC depth
$568k / $623k / $777k / $791k / $984k → 4 h move +1.96 / +1.21 / −2.06 / −1.00 / −2.57) and without 11-21 the 4 h-move split is still +0.13 vs
−1.62. That is **5 day-observations** — monotone, but far below Q4's ≥ 10 days per side, and on the net outcome it does not pass → no ship.

## 4 · Result — TP / stop / hold (net % per trade, live costs, funding)
| hold (N) | best cell | net | WR | vs random | Kelly @ floor |
|---|---|---|---|---|---|
| 1 h (261) | every cell | −0.11 … −0.40 | | | < 0 |
| 4 h (261) | TP 5 / stop 5 | +0.17 | 49 % | +1.04 | −0.03 |
| 12 h (244) | TP 5 / stop 3 | +0.07 | | | < 0 |
| 24 h (225 — the last day's paths are not complete) | **TP 5 / stop 3** | **+0.33** | 47 % | +0.98 | −0.02 |
Shape (all holds): stops of 1–2 % lose everywhere (noise stop-outs on thin books); TP 3–5 with stop 3–5 is the best area; TP 8–12 is worse
(the move rarely goes that far). **No cell makes money with confidence** (every Kelly at the pessimistic WR is below 0); the best cell's halves
are +0.10 / +0.64 and it is picked in-sample on 4.7 days (30–50 % haircut → ≈ +0.2).

## 5 · Meaning
The vacuum is real in one sense — H4 beats a random entry on the same coin (by ≈ +1 pp in the TP 3–5 / stop 3–5 area; over all 120
cells the gap ranges 1 h +0.15 … +0.45 · 4 h +0.14 … +1.04 · 12 h −0.11 … +1.05 · 24 h −0.28 … +1.00) — but that is mostly because random longs
on these coins lost during the window (−0.7); the trade itself is break-even. None of the "why would it work" theories separates winners from
losers beyond luck; two go the wrong way (crowded shorts and tight spreads should have helped and did not). The one blind hit is a market-wide variable with 5 day-observations.
The only theory pointing the right way on every outcome (buyers are the aggressors, T2b) is a HOLDOUT-only variable.

## 6 · Verdict
No sleeve, no filter, no change. **Frozen lines = PREREG Addendum 9 only** (operator 11-24, judged on data ≥ 12-07):
H4-T — ask $ within 0.5 % ≥ $12k, TP 5 / stop 3 / 24 h · H4-B — H4-T split by BTC's 4 h return at the trigger (day units). The taker-share and
BTC-depth items an earlier draft listed here are retired (Addendum 9). H4 ask vacuum itself still gets its pre-registered read at 12-10.

## 7 · Exhaustive 2D screen + what differed by day (added after the operator's question; scripts/H4_ASK_VACUUM_2D_2026-11-24.py)
**2D (TP 5 / stop 3 / 24 h, 225 trades, every pair of 63 variables × 4 quadrants, cell N ≥ 15, rotation null 1,000×):** best cell
|t| 4.96 (OI 60 min rising × top traders long, 15 trades on **2 days**) vs a family bar of 6.97. The bar stays 6.97 with
zero-spread cells removed on both sides; it is high because small 15-trade cells with little spread and a coarse rotation null (≈ 111
distinct offsets on ~4 days of rows) produce large |t| by chance. It does not decide anything: every top-10 cell sits on 2–4 days (the window-units floor is ≥ 8) and most use the
OI / positioning columns scored on only 38 % of trades. **No 2D separator.**

**By day — the trade follows the market day:**
| day | BTC day | H4 net (TP 5 / stop 3 / 24 h) | WR | random buys, same coins | H4 − random |
|---|---|---|---|---|---|
| 11-20 | **−4.91 %** | **−2.30** | 15 % | −2.10 | −0.20 |
| 11-21 | +2.17 % | +1.18 | 56 % | −0.21 | +1.39 |
| 11-22 | +1.74 % | +0.88 | 54 % | −0.61 | +1.49 |
| 11-23 | 0.00 % | +0.46 | 49 % | −0.29 | +0.75 |
Other day medians, 11-20 vs 11-21: the coin's own 60-min run-up into the trigger +0.19 vs +1.37 % · BTC book depth $984k vs $568k ·
breadth 5 min 37 vs 51 % · premium −0.047 vs −0.032. H4 ranks exactly with BTC's daily move (caveats: with 4 days a perfect rank has p = 1/24;
each 24 h hold spills into the next day, so the trigger day is not the trade's whole market window; the random same-coin column does NOT
follow BTC's order — −0.21 > −0.29 > −0.61). It is a long on thin-book alts, so the market drives much of its result; its own edge shows only as +0.75 … +1.49 over random on the three non-crash days (−0.20 on the crash day).
Four days = four market observations (one down) — a BTC-direction switch cannot be tested until many more days exist (PREREG Q4: ≥ 10
days per side).
