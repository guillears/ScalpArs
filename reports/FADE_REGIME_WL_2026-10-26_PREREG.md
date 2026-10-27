# SPIKE_FADE winners vs losers BY MARKET REGIME — pre-registration (frozen 2026-10-26)

**Frozen BEFORE any regime-variable × outcome table was computed.** At freeze time the author had seen only: the era-level kept-fade
tallies (G1 6·83%·+0.61 · G2 17·88%·+0.55 · G3 17·82%·+0.12 · G4 9·89%·+0.39 · G5 12·100%·+0.47 · G6 15·80%·−0.05 · B25 8·62%·−0.35),
the CURRENT_STATE fade lines (41c/e/h/i/k–p) and the PREP_B25 prereg / backtest blocks. No regime feature had been built.

Cohort: MASTER_POOL_stacked SPIKE_FADE stack-kept non-probe (76, outcome `stack_pct` = price %, 1× — the 2× invest multiplier never moves %),
B25 provisional (8 CLOSED fades of scalpars_orders_paper_2026-10-26_15-11-28.csv, all opened BEFORE the 13:50 UTC paper top-up), yr6
replay fades (3 seeds, refute-only caveat). Stack-blocked master fades (50) only for cohort-completeness checks.
Sleeve breakeven WR (frozen formula) = |avg loss| / (avg win + |avg loss|) on the kept master fades (computed at run time; DL 374 ≈ 72 %).

## Mechanism (first principles)
A SPIKE_FADE shorts a thin alt (< $20M 24 h volume) right after a +25-point 5m RSI jump on ≥ 5× volume, in a non-bull tape
(BTC 5m RSI ≤ 50, BTC ≤ 5m EMA13). The edge is a ~1–2 min mean reversion of a wick that one aggressive taker (or a short-squeeze
cascade) printed into an empty book. It loses when the move is REAL (somebody keeps buying) or when the TAPE lifts the coin.
Regime only matters through three channels: (a) **who is on the other side** after the spike (sellers present vs absent),
(b) **beta drag** (does the market pull the coin back down or up over the next minutes), (c) **squeeze fuel** (crowded shorts, liquidations).

## Predictions (direction frozen; tested at SIGN granularity first where a natural zero exists, else median, then terciles)

| # | channel | variable(s) | predicted direction (fade P&L) | rationale |
|---|---|---|---|---|
| P1 | BTC short-term drift | `btc_r1h`, `btc_r4h`, BTC 15 m / 30 m return | **down → better** (W_LOWER) | sellers in the tape absorb the spike; a rising BTC lifts the alt (beta tailwind = loser class 1) |
| P2 | BTC 24 h / 7 d trend | `btc_r24h`, `btc_r7d`, `btc_c_e20_1h/4h` | **down → better** (weak) | same, higher frame |
| P3 | BTC drawdown / oversold | `btc_offhi_30d`, `btc_offhi_7d`, `entry_btc_trend_gap_pct` | **deep drawdown (lowest tercile) → worse** | the flip-short analogue (DL 165 trend-gap floor): in an oversold tape alt spikes are relief bounces that squeeze shorts |
| P4 | **alt correlation / beta (mirror of ML)** | `br_corr24h_top20`, `br_beta_btc_24h`, pair's own 24 h corr / beta to BTC | **high → BETTER** (W_HIGHER) | in a lock-step tape an idiosyncratic spike is an anomaly against the factor and is dragged back to the index; in a decoupled (low-corr, high-dispersion) tape single-coin stories persist. Competing mechanism (corr × BTC rising = drag UP) is tested as the interaction P4b: high corr ∧ BTC 1 h up → worse |
| P5 | dispersion | `br_disp_1h`, `br_disp_24h` | **high → worse** | rotation tape: idiosyncratic runs persist |
| P6 | breadth | `br_above_ema50_1h`, `br_above_ema20_5m`, `br_adv_1h`, `br_adv_24h`, `entry_bear_pct` | **bearish breadth → better** | sellers present across the market |
| P7 | alt vs BTC | `alt_minus_btc_24h/7d`, `ethbtc_r24h/7d` | **alts weak → better** | no alt bid to support the spiked coin |
| P8 | volatility regime | `btc_rv_24h`, `btc_rv24_pctl90d`, `btc_rv_7d`, `btc_atr` stamp | **high vol → worse** | liquidation tapes: spikes are short-squeeze cascades with follow-through; in calm tape a wick is an anomaly |
| P9 | liquidity / time | `weekend`, `hour`, `br_qv24_x7d`, `btc_qv24_x7d` | **weekend / quiet market → better** | one taker walking an empty book prints a wick nobody defends |
| P10 | funding / squeeze fuel | pair funding at entry (last settlement ≤ t), pair 3-settlement mean; top-50 median funding | **pair funding < 0 → worse** | negative funding = crowded shorts = squeeze fuel; the spike is the squeeze starting |
| P11 | news / real move proxies | pair volume inflation 24 h / 7 d, pair return 1 h / 4 h / 24 h BEFORE the spike bar, pair off 7 d high | **inflated volume / pre-run / near 7 d high → worse** | a coin already moving on news (volume ↑ for a day, price climbing) is a real move; a spike from a quiet base after a sell-off is a wick |
| P12 | spike geometry (closed 1 m bars before the fill) | spike-bar run (entry vs forming 5 m open), 1 m-high distance (high of the last 15 closed 1 m bars vs entry), seconds into the 5 m bar | **bigger run → better; entry already ≥ 0.5 % below the high → worse; late in the bar → worse** | more overshoot to give back; a fill after the snap-back started buys the bottom |
| P13 | mcap (DL 311 / 374 ③) | `entry_mcap_usd` ≥ $200M | **worse** | depth + owners; coverage only since 10-19 → watch only |
| P14 | regime label (DL 374 ④ / 41h) | `entry_btc_regime` == HEALTHY_BEAR; HB ∧ trend gap ≥ −0.13 | **worse** (master); backtest contradicts | orderly bear: dips get bought, spikes partly continue |

**No prediction (screen-only):** BTC efficiency (6/24/72 h), above72, variance ratio / autocorrelation, BTC ADX on 5m/1h/4h/1d (DL 374 ⑥
keeps its own frozen read), ETH 5 m return, BTC 1d EMA50/200 — no first-principles sign for a 90-second wick.

## Evidence rules (frozen)
- Regime variables are market-wide → **evidence in DAY units** (fills of one UTC day = one observation); per-day and per-pair loss
  concentration reported; **day-rotation null** (circular shift of the day sequence) and a fill-level circular-shift null for the full screen.
- Uniform-degradation test on era groups G1–G6 (+B25 provisional): if every sub-cohort degrades together the cause is the tape.
- **Filter bar = CLAUDE.md expectancy bar** on the blocked cohort at 1×: WR < breakeven ∧ P(avg < 0) ≥ 0.95 by window(day)-clustered
  bootstrap ∧ ≥ 8 windows ∧ no window / pair ≥ 50 % of the cohort loss ∧ N ≥ 15; then a 30–50 % in-sample haircut.
- Per-pair rule: if ≥ 60 % of a zone's loss is 1–2 pairs → pair blacklist, not a dimension filter.
- Any surviving zone becomes a **frozen observe line** (threshold fixed now, cohort = fills opened ≥ 2026-10-27), never an armed filter
  from this in-sample study (OBSERVE-FIRST rule). The backtest only refutes.
- A prediction is CONFIRMED only if the master day-level direction matches AND the fill-level Δ has p < 0.10 under the null; RIGHT SIGN
  if direction matches without significance; REFUTED if opposite sign; UNTESTABLE if coverage < 30 kept fills.
