# NO-SPOT FRENZY SHORT SLEEVE — pre-registration (frozen 2026-11-18 21:40 UTC)

Frozen BEFORE any result of this study (no rule, base or exit P&L has been computed). Inputs already known: the NOSPOT_MCAP result
(EXH trigger on no-spot coins NS0 = −0.09 %/trade; OI-1h ≤ 0 proxy NS-P = +0.08, both fail), the SECOND_PEAK result (P2 · K cell
−0.21 over all Frenzy coins), and the data coverage counts (2,731 no-spot episodes, 222 pairs; metrics 9,996 / 10,802 pair-days cached,
premium 1,510 / 1,554 pair-months cached). Research only. Nothing in the bot changes. No commit.

## Question (operator)
Can a SHORT sleeve be built on Frenzy coins with NO Binance spot market (perp-only: PIPPIN, BLESS, JELLYJELLY, PHAROS …)? They often dump
after the Frenzy peak, but the squeezes before the dump hit the stops. EXHAUSTION_SHORT cannot trade them (its edge = spot share falling).

## Theory of the trade (written before data)
On a perp-only coin there is no spot float to sell into. The pump is built from futures leverage: new longs open (OI up), funding turns
positive, the perp trades at a premium to the index (other exchanges' spot). The dump comes when that leverage UNWINDS: longs close or are
liquidated (OI falls), the premium collapses, aggressive buyers fade (taker buy/sell < 1), the smart accounts (top traders) cut while the
crowd stays long. The squeeze risk is the mirror: when SHORTS are crowded (global L/S < 1, funding negative), every dip is bought back by
short covering, and a short also PAYS negative funding (hourly on squeezed coins).

Implications (each a rule below):
1. Shorting while OI still RISES is shorting into new leverage → bad, unless the price has stalled (trapped late longs, R2).
2. OI falling while the price is still near the high = longs leaving before the price shows it (R1, R10). OI falling fast with price falling
   fast = a liquidation cascade that tends to continue (R8).
3. Funding: extreme positive funding that turns down = longs giving up (R3). Negative funding at entry = crowded shorts = squeeze risk.
4. Top traders cutting their long POSITION while the crowd's account ratio rises = distribution (R4; the DISTRIBUTION_SHORT idea, DL 586).
5. Aggressive buyers fading near the high = demand exhausted (R5).
6. The perp premium collapsing from its episode peak = the leverage bid is gone (R6).
7. Price structure after the peak: a lower high plus the loss of the episode VWAP = the late longs are under water (R7).
8. Squeeze guard: short only when the crowd is LONG (global account L/S high) and funding ≥ 0, after the peak (R9).

## Universe
- Episodes: `reports/EXHAUSTION_EXPAND_data/allspk_meta.csv` (the all-spike list NOSPOT_MCAP used), window = [on_ms, spike_ms + 72 h).
- No-spot: `nospot_mcap_common.spot_status()` per (pair, month of the entry) = 0 (no spot 15m zip on data.binance.vision that month).
  Unknown (−1) rows dropped. Pairs whose name starts with "1000" are excluded (their base HAS a spot market; the engine's exact-symbol
  check would call them no-spot — a builder note, not a sleeve member). BTC / ETH and non-ASCII pairs excluded.
- Eligible bars: every CLOSED 15m bar (3 × 5m present) inside the window, with vol24 ≥ $20 M at the signal bar and ATR15 defined.
  Entry = the next 5m bar's open (T). The NOSPOT `fr` vol-decay ≥ 0.40 gate is NOT applied (post-peak entries often have decayed volume);
  vol_decay is a reported variable.
- Lags: every 5m / 15m bar input closes ≤ T. Metrics (OI, L/S, taker): rows with create_time (period START) ≤ T − 5 min only. Premium:
  5m premium-index bars closed ≤ T. Funding: settlements ≤ T.

## Features (at T)
dd_hi = % below the running max high since on_ms · peak_age_h · ret1h / ret4h (close-to-close) · anchored VWAP since on_ms (typical price ×
volume) and close vs it · lh2 = max high of the last 2 h ÷ running high · ATR15 % · vol_decay (episode, as NOSPOT) · vol24 · Q5 · RSI15 /
RSI1h · oi_chg1h / oi_chg4h / oi_vs_max (OI ÷ max OI since on_ms) · top-trader position ratio (sum_toptrader_long_short_ratio) and its 1 h
change · top-trader account ratio (count_toptrader_long_short_ratio) · global account ratio (count_long_short_ratio) and its 1 h change ·
taker buy/sell ratio mean of the last 12 rows and of the 12 before · last funding rate, its interval, its 8 h-equivalent, previous
settlement · premium 1 h mean, episode peak of the premium 1 h mean · hour UTC · BTC R1 (daily EMA50 vs EMA200 sign) and R4 (services/
regime.py label on BTC 5m: bull +1 / bear −1 / chop 0) from `reports/LEADS_DEEP_2026-11-18_data/regimes.pkl`.

## Candidate rules (10 primary, parameters frozen; plateau grid in brackets, stated in advance)
| rule | theory | primary condition (all at T) | plateau grid |
|---|---|---|---|
| R1 OI_UNWIND | longs leave near the high | oi_chg1h ≤ −3 % ∧ dd_hi ≤ 15 % ∧ ret1h ≤ 0 | oi {−1, −2, −3, −5, −8} × dd cap {10, 15, 25} |
| R2 OI_TRAP | late longs trapped in a stall | oi_chg1h ≥ +5 % ∧ abs(ret1h) ≤ 1.5 % ∧ dd_hi ≤ 10 % | oi {+3, +5, +8, +12} × stall {1, 1.5, 3} |
| R3 FUND_TURN | extreme + funding turns down | prev funding (8 h-eq.) ≥ +0.05 % ∧ last < prev ∧ dd_hi ≤ 20 % | prev {0.01, 0.03, 0.05, 0.10} × dd cap {10, 20, 30} |
| R4 TOP_DISTRIB | top traders cut, crowd stays long | top POSITION ratio 1 h change ≤ −5 % ∧ global account ratio 1 h change ≥ 0 ∧ dd_hi ≤ 20 % | chg {−2, −5, −10} × dd cap {10, 20, 30} |
| R5 TAKER_FADE | aggressive buyers fade | taker ratio mean last 1 h ≤ 0.90 ∧ mean of the hour before ≥ 1.0 ∧ dd_hi ≤ 15 % | thr {0.80, 0.85, 0.90, 0.95} × dd cap {10, 15, 25} |
| R6 PREM_COLLAPSE | perp premium gone | episode peak premium (1 h mean) ≥ 0.25 % ∧ premium 1 h mean ≤ 0.25 × that peak ∧ dd_hi ≤ 20 % | peak {0.15, 0.25, 0.50} × frac {0.10, 0.25, 0.50} |
| R7 LH_VWAP | lower high + VWAP lost | peak_age ≥ 2 h ∧ lh2 ≤ 0.95 ∧ close < anchored VWAP ∧ 8 % ≤ dd_hi ≤ 30 % | lh {0.97, 0.95, 0.90} × age {1, 2, 4} h |
| R8 CASCADE | liquidation cascade continues | oi_chg1h ≤ −8 % ∧ ret1h ≤ −5 % | oi {−5, −8, −12} × ret {−3, −5, −8} |
| R9 CROWD_LONG | squeeze guard after the peak | dd_hi ≥ 8 % ∧ peak_age ≥ 1 h ∧ global account ratio ≥ 1.5 ∧ last funding ≥ 0 ∧ oi_chg1h ≤ 0 | L/S {1.2, 1.5, 2.0, 2.5} × dd min {5, 8, 15} |
| R10 OI_FROM_MAX | OI unwound, price has not | oi_vs_max ≤ 0.85 ∧ dd_hi ≤ 20 % | frac {0.95, 0.90, 0.85, 0.80} × dd cap {10, 20, 30} |

Reference cohorts: **B0** = every eligible bar (random timing; the "no-spot base") · **NS0** = the EXHAUSTION trigger on these rows
(15m RSI14 cross 70 → ≤ 70 ∧ vol_decay ≥ 0.6366). A rule bar with a missing input is not a signal (fail-closed); coverage is reported.

## Exits (6, frozen; all 24 h unless stated; TP = resting maker, fills only when a 5m low trades ≥ 1 tick THROUGH it; stops gap-aware,
fill at max(bar open, stop); TP and stop in one bar → stop)
- **XE** EXHAUSTION shipped: TP min(20 × ATR15 %, 10 %) / SL min(1.5 × ATR15 %, 6 %)
- **XA** old EA: TP min(3 × ATR15 %, 5 %) / SL min(1.5 × ATR15 %, 5 %)
- **F55** TP 5 / SL 5 · **F106** TP 10 / SL 6 · **F1012** TP 10 / SL 12 (wide stop vs squeezes) · **F1510** TP 15 / SL 10, 48 h

## Costs and positions
Taker 0.05 % per taker leg, maker 0.02 % on TP. Slippage per taker leg = max(0.10 %, slip model at $6.1 k notional: −0.042 + 0.276 ·
√(6100 ÷ Q5)); stress at flat 0.20 %. Real funding at every settlement (hourly when Binance switched the coin); a missing pair-month is
charged 0.05 %/8 h against. One position per pair per cell (re-entry after the exit allowed); first-signal-per-episode shown as robustness.
One size for DCR: notional 0.61 × equity (0.5 × 5× sizing ≈ 0.61 × equity per the brief).

## Splits
DESIGN 2023-07-01 → 2025-07-01 · HOLDOUT 2025-07-01 → data end (2026-11-18), read ONCE. 2022 not available in these caches (context: none).
Half-years 23-H2 … 26-H2.

## Selection on DESIGN (survivor = all of)
N ≥ 30 · avg net > 0 · avg > B0's avg (same exit) · ≥ 2/3 of DESIGN half-years with N ≥ 10 positive · ≥ 2/3 of the rule's 12–20 plateau
cells positive (same exit). The HOLDOUT is computed for survivors (and, descriptive only, for every primary cell so a null still shows
tables — no ship can come from a non-survivor).

## Ship bar (all; read on HOLDOUT for a survivor, with the full-span read shown beside)
1. HOLDOUT avg > 0 and > B0's HOLDOUT avg · 2. ≥ 2/3 of all half-years positive (N ≥ 10) · 3. avg > 0 at 0.20 % slip · 4. best-5 %-out
avg > 0 · 5. Kelly at the one-sided 95 % Wilson WR floor (n = distinct trade days) > 0 · 6. family-wise shuffled null p < 0.05: each
trade's entry is replaced by a uniform random eligible bar of the SAME episode (same exit, same one-position rule not re-applied), 500
draws; statistic = cell avg − its null mean; max over the 60 primary cells (10 rules × 6 exits), on the full span · 7. no pair and no day
≥ 50 % of the net profit · 8. ≥ 8 distinct windows (trade days) · 9. real funding included (by construction).
Then a 30–50 % in-sample haircut on any projected Δ. Verdict: **SHIP-candidate** = all 9 · **OBSERVE** = HOLDOUT > 0 and ≥ 6 of 9 ·
else **NOTHING**.

## Reads after selection (descriptive, no re-fit)
Path decomposition (5m, path_labels definitions: WORKED = +1 % before −Y, Y = the exit's stop distance; NEVER_WORKED / STOPPED_RECOVERED
(+1 % within 120 min after a stop) / GAVE_BACK / DRIFT_LOSS / LEFT_ON_TABLE / CAPTURED) for B0, NS0 and every rule (XE exit). W/L on the
best 1–2 cells: coverage table, every feature at 3 granularities (sign/median, terciles, quintiles), every 2D pair at terciles, shuffled
label null (500), blind-spot list. BTC R1 and R4 at sign granularity per half-year. Before → after at one size (N · /day · WR · avg · DCR
%/day · max DD · worst day · Kelly), and the joint add to EXHAUSTION + DISTRIBUTION where their trade lists exist. Operator coins:
PIPPIN, BLESS, JELLYJELLY, PHAROS (11-18), and HUMA / ZKC / ZKP only if no-spot (they have spot → stated, not replayed).

## ADDENDUM A (frozen 2026-11-18 21:45 UTC, after the row build, BEFORE any cohort / rule / exit result was read)
Operator deep-dive request (via coordinator). Disclosure: one sanity print of PIPPIN's rows (column ranges incl. its mean gross XE r)
was looked at to check the build; no cohort, rule, base or exit table had been computed.
- **W/L on NS0 (the EXHAUSTION trigger on these rows) AND on every primary rule** (its best exit on DESIGN), on booked net %, the path
  ENTRY label (WORKED) and REACH. Variables: oi_chg15m / 1h / 4h, oi level $ and oi ÷ vol24, oi_vs_max, funding last / prev / 8 h-eq / sign /
  hourly flag (f_int_h = 1), topP, topP 1 h / 4 h change, topA, topA 4 h change, glA, glA 1 h / 4 h change, taker1h, taker_prev1h, prem1h,
  prem_pk, rsi15, rsi1h, di15 (+DI − DI, 15m), dd_hi, lh2, vwap_gap, peak_age_h, hrs_since_spike, vd, volmult, atrp, vol24, q5, mcap,
  turnover, ret15m / 1h / 4h / 24h, maxup1h_24h (recent squeeze), rise_on_to_hi, hour, BTC R1 / R4 (sign) and btc1h / 4h / 24h.
  Granularities: sign (or median for positive-only), terciles, quintiles (cut points from DESIGN rows); every 2D pair at terciles.
  Family-wise null: 500 shuffles of the P&L across the cohort's trades, max over variables × granularities (1D) and over cells with
  N ≥ 30 (2D). Coverage table and blind spots first.
- **Squeeze class:** a loser whose path hit the stop before +1 % (WORKED = 0, stopped). Compared with the winners on every variable.
- **Theory filters (frozen, block when):** Q1 glA < 1.0 (shorts crowded) · Q2 f_last < 0 (negative funding) · Q3 oi_chg1h > 0 (new leverage
  still coming in) · Q4 maxup1h_24h ≥ 20 % (recent squeeze on the coin) · Q5 f_int_h = 1 (hourly funding) · Q6 mcap < $50 M (unscored kept) ·
  Q7 = Q1 ∨ Q2. Unscored input → kept (reported).
  Applied to NS0 and each primary rule (each at its exit). DESIGN survivor = kept avg > 0 ∧ kept > unfiltered ∧ blocked avg ≤ 0 ∧ kept N ≥ 30
  ∧ ≥ 2/3 DESIGN halves positive. Survivors → HOLDOUT once + the 9-item ship bar (same family-wise null idea: the filter family = 7 filters ×
  11 cohorts, label-shuffle of kept/blocked within the cohort) + both BTC R1 signs + day-block bootstrap CI.
- Screen survivors (data-mined) can only reach OBSERVE.
