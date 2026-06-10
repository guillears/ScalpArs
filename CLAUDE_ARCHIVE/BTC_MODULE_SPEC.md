# BTC Module — Parked Spec & Analysis (2026-06-10)

> Status: **PARKED** (operator decision — focus stays on the alt scalper forward test).
> Revisit triggers: balance > ~$25–50k (diversifier sleeve) · maker-exit capability ships ·
> or operator decision. Scripts: `scripts/btc_backtest_fetch.py`, `scripts/btc_backtest.py`,
> `scripts/btc_backtest_directional.py`. Kline caches (`reports/btc_klines_{5m,1h}_cache.csv`)
> are regenerable via the fetch scripts.

## Why BTC is no-trade today (confirmed twice)
Live trades (N=26, Apr 28–Jun 2, old stack): LONG 3W/15L (17% WR), SHORT 4W/4L, −$296 total.
Root cause is STRUCTURAL, not filterable: BTC 5m ATR median 0.13% → median MFE +0.156% vs
0.063–0.090% round-trip fees = **fees eat 40–58% of the median best-case move** (alts: 24–34%).
Lower-TP/tighter-SL grids raise WR but never flip $ positive (confirmed by Jun-3 archive analysis
AND the Phase-1a backtest). EMA13 cross helps BTC (+1.10pp on N=8) — keep ON if ever traded.

## Phase 1a — 5m scalp backtest (51,841 candles, Dec 14 → Jun 10, BTC −30%)
- LONGS with fixed TPs: negative everywhere (31–47% WR; BTC longs grind, never spike to targets).
- Trailing + deep 1h-trend gates rescue both sides into thin positive territory:
  - SHORT champion: 1h EMA20-slope<−0.15 + 5m EMA5<8<13 + RSI 32–48 + gap≤−0.03 + ADX≥18,
    trail arm 0.30/pullback 0.25, SL 0.40 → N=733 (4.1/day), 54% WR, **+17.3%/6mo @MT fees**
    (+37% @maker-both), exp +0.024%/trade, hold ~40min. IS +6.6 / OOS +10.6. Pays in crash
    months (Jun +13.2); grinds otherwise. Slope-gate sweep is a plateau (−0.10/−0.15/−0.20 alike).
  - LONG champion: slope>+0.20 mirror, trail 0.50/0.40, SL 0.60 → N=285 (1.6/day), 58% WR,
    **+5.8%/6mo @MT**, hold ~100min. IS +4.6 / OOS +1.2 (weaker, chop-sensitive side).
  - COMBINED @MT: N=1,018 (5.7/day), 55% WR, exp +0.023%/trade, +23.1%/6mo, maxDD −13.1% (1×).
    Daily compound @10×/20% alloc ≈ +0.22%/day, maxDD −23%.
- **Verdict: real but thin; fee tier (maker vs taker exit, Δ0.027%) is larger than the per-trade
  edge — fill quality IS the strategy. At small account size (~$4/day) not worth the build.**

## Phase 1b — 4h DIRECTIONAL backtest (3 years 2023-06 → 2026-06, $25.9k→$62.3k)
**ALL 24 cells positive** (D1 EMA20/50 cross · D2 Donchian-20 breakout · D3 EMA50-slope regime ×
both sides × 4 trail grids) — robustness across systems/sides/params; the edge is BTC's 4h trend
persistence, not a tuned parameter. Fees ~5–10% of edge (problem solved by timeframe).

**Champion portfolio (PARKED SPEC): Donchian-20 on 4h bars, two sleeves:**
- LONG: enter close > 20-bar high; trail 3.0% (hard SL 3.0% pre-arm) → N=114, exp +0.89%/tr
- SHORT: enter close < 20-bar low; trail 2.0% (SL 2.0%) → N=127, exp +0.61%/tr, maxDD −11.4%
- Combined: **N=241 (6.6 trades/MONTH), holds 1–7 days, sum +180%/3yr**
- Per-year: **2023 +41% · 2024 +90% · 2025 +54% · 2026 −4.9%** (short sleeve hedges the bear)
- vs Buy&Hold +108% with −52% maxDD
- Compounded: 1× = ×5.15/3yr (+73%/yr, maxDD −29%) · **2× = ×19.9 (+171%/yr, maxDD −51%)** ·
  3× = ×59.7 but maxDD −66% (too hot). Recommended sizing if built: ≤2× leverage, own sleeve.

**Unmodeled / caveats:** perp funding (±0.01%/8h — may shave ~10–20% off the long sleeve in bull
years) · stop-fill slippage at 4h scale (small) · 2026 bear only partially complete · single asset.

## If/when built (Phase 2 design notes)
- Independent module: one check per closed 4h bar (Donchian + trail state machine). NO interaction
  with the 5m scalper (own capital sleeve, own position slot, not counted in max_open_positions).
- Maker entries where possible; taker stops acceptable (fees are 5–10% of edge here).
- Forward gate: N≥20 trades (~3 months) matching backtest WR/expectancy bands before sizing up.
- This is a WEALTH/diversifier sleeve (+0.15–0.27%/day compounding, pays most in crash months),
  NOT a growth rocket — nothing on BTC produces multi-%/day.
