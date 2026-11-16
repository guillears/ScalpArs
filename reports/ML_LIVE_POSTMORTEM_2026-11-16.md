# Momentum LONG live post-mortem — 2026-11-16 (operator: "what the fuck happened to momentum long, winners vs losers well done")
Data: live B27 (go-live 11-14 23:45 → 11-16 19:58, export scalpars_orders_live_2026-11-16_19-58-21) · live B22/B23 (Oct, small size) · paper B25+B26 (as traded, probes out) · master STACK 2026-11-16a (kept, full-size) · CloudWatch logs for every B27 fill · Binance aggTrades at signal / fill.

## Headline
| Cohort | N | WR | avg % | $ |
|---|---|---|---|---|
| **Live B27** | 10 | **20 %** | **−0.57** | **−$1,624** |
| Live B22 + B23 (Oct, ~$300 tickets) | 19 | 68 % | ≈ 0.00 | ≈ $0 |
| Paper B25 + B26 as traded (pre-go-live) | 54 | 63 % | +0.03 | — |
| Master kept (fitted filters, in-sample) | 93 | 85 % | +0.38 | — |
| OOS audit raw master (no fitted filters) | 267 | 65 % | +0.02 [−0.08, +0.14] | — |

## 1 · Theory
Momentum long buys continuation of a 5m uptrend (EMA stack, RSI 55–70, ADX rising) and lets winners run on a trail (arm +0.40). Failure modes: (a) the move is already over when we fill (execution), (b) no continuation — the push fades at once (entry / regime), (c) gave back (exit), (d) size turns normal losses into big $.

## 2 · Path decomposition (live B27)
8 losers, all stops at ≈ −0.70 to −0.80 %: **6 never went green (peak ≤ +0.09 %)**, LINK +0.13, AAVE +0.24 (never reached the +0.40 trail arm). Time to stop 10–57 min. Never-worked share: live 60 % · paper B25/B26 26 % · master 19 %. → (b) entry / no-continuation, not exit.

## 3 · Execution parity — NOT the cause
| | Live B27 |
|---|---|
| signal → fill delay | 40–88 s (mean 60 s); maker limit failed 8 / 10 → taker after 20 s |
| price drift signal → fill | **+0.03 % mean** (range −0.06 … +0.12) ≈ **$130 total** |
| entry slippage stamp | +0.04 % (paper B25/B26 +0.03 %) |
| fees (≈ 0.08 % round trip on $294 k notional) | ≈ $235 |
Execution + fees ≈ $365 of −$1,624 (≈ 22 %). The trades failed even from the signal price.

## 4 · Market regime at the live fills vs master
| | Live B27 median | Master winners | Master losers |
|---|---|---|---|
| BTC ADX | **19.5** (10th pct of winners) | 24.8 | 22.8 |
| BTC RSI (5m) | 65.5 | 62.5 | 62.6 |
| bull % | 79 | 72 | 70 |
| BTC 1h slope | −0.03 | −0.08 | −0.10 |
BTC was range-bound (low ADX) with 5m RSI hot. But master kept fills at BTC ADX ≤ 21: 19 · 79 % · +0.32; BTC RSI ≥ 65 ∧ ADX ≤ 22: 15 · 87 % · +0.35 → **not a separator** (no filter; never refit on these 10).

## 5 · Winners vs losers — every variable
Coverage: 62 entry_* variables scored on ≥ 70 % master / ≥ 60 % live; blind spots: btc_off24h/off30d/r72/eff72, regime label, alt beta/corr, book stamps (low coverage). Live fills sit OUTSIDE the master winners' 10–90 % band mainly on SIZE (desired notional 97th pct) and BTC ADX (10th pct); every other variable is inside the winners' range. sweep_separators.py ML (SCREENED_BASELINE pool only, 41 fills — the script's ML mode): path / reach / pnl — 1D consistent 18 / 24 of ~91; **2D consistent 165 of ~3,280 ≈ 5 % = the chance rate**; 3D 0 of 390. → no separator distinguishes these losers.

## 6 · Concentration
3 bursts carry 86 % of the loss: 11-15 13:23 (AVAX, INJ, ARB in 46 s) −$424 · 11-16 07:54–08:00 (XRP, AVAX, DOGE) −$651 · 11-16 17:22 (WLD, LINK) −$314. 10 fills = **5 market windows**; 2 / 10 wins is ~1 % if fills were independent, but ~10–20 % in window units.

## 7 · Conclusion
- **What happened:** a sleeve with no out-of-sample edge (raw master +0.02, paper as traded +0.03, unseen month −0.03) hit a bad run concentrated in 3 correlated bursts, at a size (≈ 3.5 × the account per ticket at 15×) that turned ordinary −0.7 % stops into −$200 to −$380 each.
- **$ attribution (approx.):** no-edge + bad-luck entries ≈ −$1,260 (the trades would lose from the signal price) · execution drift ≈ −$130 · fees ≈ −$235. Size is the multiplier on all of it.
- **Not the cause:** execution (0.03 %), exits (nothing to give back), a missing filter (no variable separates; 2D survivors = chance).
- **Fixable:** size (already 15× → 8×) and correlation exposure (3 longs in one minute = one bet ×3). **Not fixable by filters:** the edge itself.
- **Recommendation:** run momentum long at minimum size (1×) as a forward test, or switch it off after the sleeve-kill checklist (status: ① ② run — sweep, no separator; ③ uniform degradation paper → live — yes; ④ tape: BTC daily uptrend, low BTC ADX — done). Restore only on ≥ 30 forward fills / ≥ 10 days with a day-bootstrap CI above 0.
Confidence: high that execution and exits are not the cause; high that no entry variable separates; medium on the $ split (10 fills, 5 windows).
