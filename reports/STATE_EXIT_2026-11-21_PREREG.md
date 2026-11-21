# STATE_EXIT 2026-11-21 — pre-registration (written 2026-11-21 ~00:30 UTC, BEFORE any outcome was computed)

**Question (operator).** Trades go +2…+8 % our way, then give it all back to the stop / time exit while waiting for a fixed TP. Is there a
pair / market STATE variable, observable during the trade, that says "sell now" and beats today's exit when added on top of it?
Sleeves: FLATBASE_SHORT (FB), FRENZY_WILLY_HT (HT), EXHAUSTION_SHORT (EXH). Research only — no engine / config change.

## Cohorts
- **REAL** — every archived export (exports_archive/scalpars_orders_*.csv, paper + live), CLOSED, no probes, operator-closed out, dedup
  (opened_at[:19], pair, direction), back to each sleeve's first fill. **MASTER** = opened before 2026-11-17 17:00 UTC; **BATCH** = after
  (current batch). Entry = the real fill price; the walk starts the minute after the fill. Every fill is RE-PRICED under TODAY's exit.
- **BACKTEST** — FB: bt_FB.csv minus the DL 601 entries (today's stack, 545) · HT: HT_WILLY_2026-11-20 cohort (300, DL 594 dip stack) ·
  EXH: 30 % stratified sample (half-year × exit kind, seed 7) of bt_EXH.csv (3,368 → 1,011) — the full set was not fetched (time / disk).
  DESIGN < 2025-07-01 ≤ HOLDOUT.
- **CASES** (illustration only, never counted): ENSO and EVAA HT (operator), RARE EXH.

## Today's exit (replayed on 1m, parity vs booked reported)
FB: TP −5 % price, resting maker (fills when the low trades 0.05 % through) · stop +10 % price counting the funding PAID (DL 607) · 72 h.
HT: TP +5 net (maker) · stop −4 net · 24 h (DL 587: late TP off, high target off). EXH: the row's TP / SL (min(20 × ATR15, 10) /
min(1.5 × ATR15, 6)) price · price stop · 24 h. Costs: entry taker 0.05 (+ entry slip on EXH backtest rows), maker TP 0.02, stop / time /
state exit taker 0.05 + slip (FB real 0.25, FB backtest the row's slip, HT 0.10, EXH the row's ≥ 0.10). Funding at real settlements.

## Decision points and label
Every minute close while the trade is open. Gates: **G1** gross P&L now ≥ +1 % · **G2** now ≥ +2 % · **G3** in-trade peak so far ≥ +2 %
(any P&L now). Label = (taker exit at the next minute's open, with slip + fee + funding so far) − (today's exit result on the same path).
One FILL = one unit; minutes are never counted as N; market-wide variables read in day-block clusters.
Per-fill labels: GAVE_BACK = peak ≥ +2 % then net < 0 · REACHED_TP · TIME_WIN / TIME_LOSS · NEVER (+2 never reached, loss).

## Layer 1 — thesis variables (pre-registered, frozen thresholds, own null)
Entry thesis → exit = "the thesis is exhausted or invalidated", read with the SAME variables as the entry.
- **Common (operator's ENSO signs, mirrored for shorts; 5m CLOSED candles):** RSI_ROLL — 5m RSI14 in-trade max ≥ 75 and now ≥ 10 pts below
  it (short: min ≤ 25, now ≥ 10 above) · LH2 — ≥ 2 consecutive lower highs (short: higher lows), close off the in-trade extreme · EMA_X — 5m
  EMA5 crosses EMA8 against us and EMA5 beyond EMA13 against us · CLIMAX — an against-us 5m candle on quote volume ≥ 2 × its prior-20 mean,
  its extreme within 1 % of the in-trade extreme · FUND_RECV / FUND_PAY — the position receives / pays ≥ 0.05 % per settlement.
- **FB** (thesis: a pump on a 14-day flat coin fades back to the base; entry = 4-h close below the spike VWAP, confirmed): FB_VWAP — last
  closed 1h close back above the spike VWAP · FB_4H — last closed 4h close back above the entry · FB_BASE — price within 1 % of the pre-pump
  base (thesis complete) · FB_VOL — rolling 1h quote volume ≥ 50 % of the episode's peak hourly volume (a second pump).
- **HT** (thesis: buy the first red 5m dip ≥ 1.4 % under the spike high inside a live Frenzy burst → continuation to the high): HT_DI — 5m
  +DI − −DI < 0 · HT_ADX — 5m ADX below 3 bars ago · HT_VOLFADE — 5m volume trend (12 ÷ 12 bars) < 0.5 · HT_NEWHIGH — price ≥ the spike
  high (thesis complete) · HT_VWAP — price below the spike VWAP · HT_EMAGAP — 5m EMA5 ≤ EMA20.
- **EXH** (thesis: an overextended pump runs out of buyers — 15m RSI falls through 70, volume still ≥ 64 % of peak, spot share falling):
  EXH_RSI_OS — 15m RSI ≤ 30 · EXH_RSI_UP — 15m RSI up 2 closed bars in a row while < 50 · EXH_VOLBACK — 1h volume ≥ 80 % of the
  24 h / episode peak with the last hour up · EXH_OI_UP — open interest +5 % in the last hour · EXH_TB_UP — taker-buy share 15 min ≥ 55 %.
  (Spot share during the trade is NOT available — blind spot.)
Null: circular shift of each fill's condition sequence within its own eligible minutes, 300 shuffles, family-wise max of the per-fill mean Δ.

## Layer 2 — broad screen
Every computed state feature (coin 1m / 5m / 15m / 1h / 4h momentum, RSI and its turn, EMA distances, ADX / DI, ATR, volume ratios / decay,
taker-buy share, range expansion, in-trade P&L / peak / drawdown / minutes / minutes since peak, spike VWAP / base / spike-high distance,
BTC 1m / 5m / 1h, funding at t and accrued, OI / top-trader / taker L/S from the 5m metrics) at sign (or median) / terciles / quintiles with
thresholds frozen on the BACKTEST DESIGN decision points (never fitted on the deciding fills), + every 2D sign / median quadrant; gate G1;
statistic t of the per-fill Δ; null = the same circular shift, family-wise max t.

## Verdict rule (pre-committed; master decides — coordinator revision)
A rule is a **CANDIDATE** only if on the **MASTER** real fills: Δ ≥ 0 AND SAVED $ ≥ CHOPPED $ AND day-block bootstrap P(Δ > 0) ≥ 0.80 AND
Δ > the best FIXED-TP alternative's Δ (FB TP 3 / 4 / 6 / 7 / 8 % · HT TP +3 / +4 / +4.5 / +6 net · EXH TP 5 … 9 %). The backtest
DESIGN / HOLDOUT is reported as the cross-check (agree / disagree; a refutation is a stated caveat, not a veto). A broad-screen rule must
also beat its family-wise null on master. A sleeve with < 8 master fills (EXH) cannot produce a candidate — its backtest read is exploratory.
Any candidate goes **OBSERVE-only** first. Otherwise: "no state exit found", with the coverage table.
