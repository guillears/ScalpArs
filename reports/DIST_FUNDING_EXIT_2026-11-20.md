# DIST_FUNDING_EXIT: close a DISTRIBUTION short before a deeply negative funding settlement? (2026-11-20)

**Verdict: NO rule. 0 of 16 rules pass for DIST, and 0 of 16 for EXHAUSTION. Holding through the settlement beats closing before it.** Every
stay-out (B1) rule loses on DESIGN and on HOLDOUT. The re-enter variants (B2) lose less, because they mostly pay twice the costs to avoid
the payment and then get back in. The one B2 cell that comes out positive in both blocks (X 2 % · M 5 · B2: +1.4 / +1.5 summed %) covers 7
fills, its CI spans 0, and it sits far below the null (+23.0). That is noise. Keep Exit B unchanged. Backtest only, so these numbers are
unverified on live fills (see the live check below).

## Question
KERNELUSDT DIST short, 2026-11-20 12:30 UTC, with a predicted funding rate of −2.00 % per 4 h (Binance's cap). On $6.4k notional one payment
is about $129. Shorts pay when the rate is negative. Is it better to close a few minutes before such a settlement than to hold through it?

## Already known (cited, not redone)
`reports/FUNDING_RULE_2026-11-19.md` tested SKIP / HALF at entry by funding rate: 0 / 20 rules passed. For DIST, the ≤ −1.0 % entry bucket was
the best one (33 fills · 52 % · +2.10 % net after funding). The backtests already charge real funding at every settlement, hourly coins
included. That study left the pre-settlement close untested, and this study fills that gap.

## Method (pre-registered: `DIST_FUNDING_EXIT_2026-11-20_PREREG.md`, frozen before any rule was simulated)
- **Cohort:** 1,534 DIST fills (RSI1h ≥ 74.07 kept). Hold = DIST_EXIT `REF_B` (TP min(20×ATR, 10 %) resting maker · SL min(1.5×ATR, 6 %) ·
  24 h). Paths, funding and costs come exactly from `scripts/dist_exit_sim.py` (1m klines with 5m fallback, `TB.funding_for` settlement
  times and rates, net = gross − entry slip − exit slip if taker − 0.05 − (0.05 taker | 0.02 maker) + funding).
- **Rules (16):** at each settlement while the fill is open, if the settled rate is ≤ −X % per interval (X ∈ 0.25 / 0.5 / 1 / 2), close at
  market M min before (M ∈ 1 / 5). The close pays taker 0.05 % plus the modelled slip.
  - **B1** stays out after the close.
  - **B2** re-enters 1 min after the settlement with the original TP / SL / 24 h clock. It pays another entry slip and 0.05 %, and the rule
    applies again at later settlements.
- **Reads:** DESIGN 2023-07 → 2025-06 and HOLDOUT 2025-07 → 2026-11 (HOLDOUT read once). Δ = summed net % at one size (rule − hold).
  - Day-block bootstrap CI (10-day blocks).
  - Family-wise null: 300 shuffles of the settlement rates within each half-year. Each shuffled rate drives the trigger, while the real
    funding is still paid. The null threshold is the 95th percentile of the max DESIGN Δ over all 16 variants.
  - Concentration, leave-one-out, saved vs cut, half-year table.
- **Look-ahead (stated):** the rules use the SETTLED rate as if it were known at s − M. Binance's rate is the average premium over the
  interval (clamped), so at s − 1 min ≥ 99.6 % of an 8 h average and ≥ 98 % of a 1 h average is already fixed, and at the cap it is exact.
  This look-ahead can only flatter the rules, and they fail anyway. A no-look-ahead sensitivity triggers on the previous settled rate
  instead.
- **Scripts:** `scripts/dist_funding_exit_study.py` (sim; `--sleeve=EXH` for EXH) and `scripts/dist_funding_exit_report.py` (tables +
  chart). Data: `reports/DIST_FUNDING_EXIT_2026-11-20_data{,_EXH}/`.

## Replica parity
- **DIST:** 1,534 / 1,534 paths (1,533 on 1m). My hold re-sim matches DIST_EXIT REF_B net per fill (|d| ≤ 0.001 %) on 1,533 / 1,534.
  - The one miss is MIRAUSDT 2026-03-01 (TP fill): the funding file now holds more settlements than when DIST_EXIT ran (−2.28 % vs
    −0.48 %).
  - All Δ are taken against my own re-sim, so they are internally exact.
- **EXH:** 3,368 / 3,368 match EXH_EXIT2 `net_REF` (max |d| 0.0001). EXH uses the same engine and the same Exit-B formulas, so the
  machinery applied at no extra cost.
- Fills a rule never closes have Δ = 0 exactly (rule − own re-sim).

## Results — DIST (Δ = summed net % at one size; touched = fills the rule actually closed)

| rule | DESIGN touched N · days · coin-days | DESIGN Δ (95 % CI) | HOLDOUT touched N · days · coin-days | HOLDOUT Δ (95 % CI) | half-years Δ>0 | pass |
|---|---|---|---|---|---|---|
| X0.25 M1 B1 | 47 · 41 · 47 | -77.4 (-154.2 … -7.2) | 84 · 78 · 83 | -18.2 (-137.5 … +97.5) | 2/7 | ✗ |
| X0.25 M1 B2 | 47 · 41 · 47 | -16.4 (-28.9 … -5.4) | 84 · 78 · 83 | -35.5 (-61.5 … -13.2) | 0/7 | ✗ |
| X0.25 M5 B1 | 48 · 42 · 48 | -65.9 (-137.7 … +1.7) | 85 · 79 · 84 | -49.4 (-174.4 … +65.4) | 0/7 | ✗ |
| X0.25 M5 B2 | 48 · 42 · 48 | -0.9 (-15.2 … +13.0) | 85 · 79 · 84 | -74.9 (-111.7 … -38.5) | 2/7 | ✗ |
| X0.5 M1 B1 | 22 · 20 · 22 | -56.1 (-123.0 … -4.7) | 43 · 41 · 43 | -40.5 (-133.1 … +51.3) | 1/7 | ✗ |
| X0.5 M1 B2 | 22 · 20 · 22 | -2.2 (-9.8 … +4.7) | 43 · 41 · 43 | -17.7 (-39.0 … +1.6) | 4/7 | ✗ |
| X0.5 M5 B1 | 23 · 21 · 23 | -52.4 (-113.8 … -3.7) | 44 · 42 · 44 | -55.5 (-154.9 … +38.8) | 0/7 | ✗ |
| X0.5 M5 B2 | 23 · 21 · 23 | +5.6 (-2.4 … +13.9) | 44 · 42 · 44 | -36.9 (-69.0 … -9.5) | 4/7 | ✗ |
| X1 M1 B1 | 10 · 10 · 10 | -15.6 (-44.0 … +15.6) | 17 · 17 · 17 | -52.5 (-107.1 … +0.8) | 0/6 | ✗ |
| X1 M1 B2 | 10 · 10 · 10 | -2.3 (-8.1 … +2.3) | 17 · 17 · 17 | -9.2 (-20.3 … +0.3) | 1/6 | ✗ |
| X1 M5 B1 | 10 · 10 · 10 | -14.0 (-42.0 … +14.8) | 18 · 18 · 18 | -55.4 (-116.3 … -2.8) | 0/6 | ✗ |
| X1 M5 B2 | 10 · 10 · 10 | -1.1 (-8.1 … +5.1) | 18 · 18 · 18 | -12.3 (-29.7 … +2.1) | 3/6 | ✗ |
| X2 M1 B1 | 2 · 2 · 2 | -8.7 (-24.0 … +0.0) | 4 · 4 · 4 | -26.5 (-57.1 … -1.3) | 0/5 | ✗ |
| X2 M1 B2 | 2 · 2 · 2 | -0.9 (-3.5 … +0.8) | 4 · 4 · 4 | -1.6 (-5.1 … +0.3) | 1/5 | ✗ |
| X2 M5 B1 | 2 · 2 · 2 | -6.4 (-19.2 … +0.0) | 5 · 5 · 5 | -23.4 (-47.7 … -2.7) | 0/5 | ✗ |
| X2 M5 B2 | 2 · 2 · 2 | +1.4 (+0.0 … +3.7) | 5 · 5 · 5 | +1.5 (-5.0 … +9.3) | 4/5 | ✗ |

- **Null 95th (max DESIGN Δ over 16):** +23.0. No DESIGN Δ reaches it.
- **Trading possibilities:** the rules are rare. At X 1 % only 27–28 DIST fills in 3.4 years (27 days, 27 coin-days) ever meet a settlement
  that deep while open. At X 2 % only 6–7 fills do.
- **Haircut:** the two positive DESIGN cells (X0.5 M5 B2 +5.6 and X2 M5 B2 +1.4) become +2.8…3.9 and +0.7…1.0 after the 30–50 %
  haircut. Both fail on HOLDOUT or on the null anyway.

### Saved vs cut · funding avoided vs price give-up · concentration · LOO (all periods)

| rule | saved N · Σ% | cut N · Σ% | avg Δ / closed fill | funding avoided Σ% | price + extra costs Σ% | top pair (share) | LOO best fill | LOO top pair |
|---|---|---|---|---|---|---|---|---|
| X0.25 M1 B1 | 51 · +337.7 | 80 · -433.3 | -0.73 | +126.2 | -221.7 | TREEUSDT (4 %) | -83.3 | -79.4 |
| X0.25 M1 B2 | 46 · +26.4 | 85 · -78.2 | -0.40 | +118.3 | -170.1 | STORJUSDT (8 %) | -46.0 | -46.0 |
| X0.5 M1 B1 | 21 · +147.7 | 44 · -244.4 | -1.49 | +92.5 | -189.2 | STEEMUSDT (4 %) | -86.5 | -86.5 |
| X0.5 M1 B2 | 29 · +18.2 | 36 · -38.1 | -0.31 | +83.9 | -103.8 | STORJUSDT (16 %) | -14.1 | -14.1 |
| X1 M1 B1 | 6 · +42.9 | 21 · -111.0 | -2.52 | +57.9 | -126.0 | STEEMUSDT (9 %) | -57.9 | -57.9 |
| X1 M1 B2 | 12 · +4.6 | 15 · -16.0 | -0.42 | +48.8 | -60.3 | RIVERUSDT (18 %) | -8.6 | -8.6 |
| X2 M1 B1 | 0 · +0.0 | 6 · -35.2 | -5.86 | +17.5 | -52.7 | MIRAUSDT (29 %) | -24.9 | -24.9 |
| X2 M5 B2 | 5 · +5.4 | 2 · -2.6 | +0.41 | +12.0 | -9.1 | FIOUSDT (41 %) | +0.6 | +0.6 |

(The M5 rows are in `DIST_FUNDING_EXIT_2026-11-20_tables.md` and read the same way.)

- **Funding saved, price lost:** the funding avoided is real (+17 to +126 summed %). Every rule gives back more in price move plus extra
  costs.
- **Losses are broad-based:** no pair carries ≥ 50 % of Δ, and leave-one-out never turns a rule positive except the X2 M5 B2 noise cell
  (+0.6).
- **Mechanism (descriptive):** the fills that are still open into a deeply negative settlement are DIST's best fills, not its worst.

| first settled rate ≤ −X while open | N | hold WR | hold avg net % | hold avg funding % |
|---|---|---|---|---|
| 0.25 | 131 | 63 % | +2.77 | −0.97 |
| 0.5 | 65 | 68 % | +3.23 | −1.44 |
| 1 | 27 | 78 % | +3.92 | −2.14 |
| 2 | 6 | 100 % | +6.30 | −2.92 |

Deeply negative funding means a crowded long squeeze is paying shorts to stay away, which is exactly DIST's setup. Price tends to fall right
after the settlement. From the close point to +30 min, the held short (funding paid) averages:

| settled rate | DIST, from close point to +30 min | EXH, from close point to +30 min |
|---|---|---|
| ≤ −0.5 % | +0.89 % (69 % positive, N 65) | +0.31 % |
| ≤ −1 % | +2.16 % (78 % positive, N 27) | +0.26 % |

B2 re-enters 1 min after the settlement. It misses part of that first move and pays a second round trip, which is why it is mildly
negative.

### Half-years (summed Δ %, closed fills only)

| rule | 23H2 | 24H1 | 24H2 | 25H1 | 25H2 | 26H1 | 26H2 |
|---|---|---|---|---|---|---|---|
| X0.5 M1 B1 | -2.7 | -5.3 | -9.7 | -38.5 | -25.7 | -17.8 | +2.9 |
| X0.5 M1 B2 | +0.2 | +0.9 | +0.9 | -4.2 | +2.0 | -8.2 | -11.5 |
| X1 M1 B1 | -2.7 |  | -9.6 | -3.4 | -20.4 | -21.5 | -10.5 |
| X1 M1 B2 | +0.2 |  | -0.7 | -1.8 | -0.1 | -4.3 | -4.8 |
| X2 M1 B1 |  |  | -6.6 | -2.1 | -6.6 | -10.2 | -9.6 |
| X2 M5 B2 |  |  | +0.5 | +0.9 | +1.4 | +2.6 | -2.6 |

The other rules are in the tables file.

### No-look-ahead sensitivity (trigger on the previous settled rate)
- Same picture. DESIGN is negative for every B1 / B2 except X1 B1 (+1.4 / +0.7, 5 fills).
- Two HOLDOUT cells come out positive: X0.5 B1 (+30.1 / +20.9) and X0.25 M1 B1 (+4.5). Both are negative on DESIGN, a sign flip, so they
  are noise.

## EXHAUSTION_SHORT (same machinery, 3,368 fills, hold = EXH_EXIT2 REF)
- **Result:** 0 / 16 pass. Every HOLDOUT Δ is negative. At X ≥ 0.5 the B1 losses run −14 to −267 summed % on HOLDOUT, and several CIs lie
  fully below 0 (e.g. X0.5 M5 B1 −267.4, CI −506 … −40; X1 M1 B1 −172.9, CI −328 … −31).
- **DESIGN:** small positives for B1 at X 0.25 / 0.5 (+0.2 … +20.4) against a null 95th of +74.2.
- **Same mechanism as DIST:** at X 0.5 the funding avoided is +486 summed %, but the price given back plus extra costs is −703.
- Full tables are in `DIST_FUNDING_EXIT_2026-11-20_tables.md`.

## Chart
`reports/DIST_FUNDING_EXIT_2026-11-20.png`. Left and middle panels: DIST fills open into a settlement ≤ −0.5 % (N 65) or ≤ −1 % (N 27). Each
line is the held short's P&L from the close point (s − 1 min), funding included, frozen at the Exit-B exit, as mean, median and 90 % band.
The flat red line is the close (0, before its ~0.15–0.3 % exit cost). The held line jumps above 0 within minutes of the settlement and stays
there. Right panel: Δ per closed fill for X 0.5 · M1 · B1, DIST and EXH.

## Live / paper cross-check
- **Exports:** the newest export (`~/Downloads/scalpars_orders_paper_2026-11-19_16-44-26.csv`) and every archived export together hold
  only **2 DIST fills**: NAORIS and SENT, 11-19. Both were stopped within 15 min, crossed no settlement and paid $0 funding. There are no
  real fills to cross-check against, and the backtest result stays unverified live.
- **KERNELUSDT 11-20** (public fapi, 4 weight):
  - It runs on 4 h funding. The 12:00 UTC settlement was −0.916 %; the predicted rate for 16:00 is −2.00 % (the cap).
  - Price rose from about 0.0510 (12:30 open) to about 0.0566 (13:15), roughly +11 % against the short. That is past the 6 % maximum
    Exit-B stop, so Exit B most likely stopped this fill before the 16:00 settlement. I could not verify this because the fill is not in
    any export.
  - The $129 payment was probably never due. Either way, one fill is one observation.

## Meaning
1. Funding is a real cost on these fills: −1.4 % to −2.9 % per fill that reaches a deep settlement.
2. Being open into such a settlement selects DIST's best trades. They are already in profit or the squeeze is spent, and price tends to drop
   right after the settlement.
3. Closing before the settlement therefore cuts more winners than it saves losers. Re-entering after it costs a second round trip and
   misses the first post-settlement move.

The 11-19 entry study and this exit study agree: deeply negative funding is a cost DIST and EXH are paid enough to carry.

## Verdict
**No rule. Hold through funding settlements (Exit B unchanged).** No observe line is proposed either. The only positive cell (X2 M5 B2) has
7 fills over 3.4 years, so a forward read would take years to mean anything.

## Blind spots / not tested
- **Predicted vs settled rate:** not measured, because there is no premium-index history in the cache. The bound is argued above. Since the
  look-ahead favours the rules, the "no" stands.
- **Live costs:**
  - A pre-settlement close on a squeezed coin may slip more than the q5 model (live INIT 0.37 %). This would make the rules worse.
  - Live funding is charged on mark price, not entry notional. The difference is small.
- **One position per pair:** a B1 close could free the pair for another DIST fill that one-position-per-pair blocked. This is not modelled,
  and the fill set is fixed.
- **Partial close:** not tested (e.g. close half before the settlement). Neither pre-registered nor implied by these results.
- **Other sleeves:** Flatbase and other shorts were not tested.
