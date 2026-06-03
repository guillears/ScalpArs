# SCALPARS — Current Live State

> Read at every session start (together with CLAUDE.md). **Edited in place** on every ship/revert (never appended — the archive grows, this stays small). Snapshot as of **2026-06-02**.

## Mode & capital
Paper. Balance **$2,500 + $500 BNB ≈ $3,000**. 5 max positions, equal-split, reserve 0. Leverage **20×** (both VERY_STRONG + STRONG_BUY).

## Strategy in brief
EMA5/8/13 + RSI + ADX scalper on 5m, BTC-macro-gated. Entry passes a BTC/pair filter stack → sized by liquidity caps × multiplier/pattern-cell rules → exited by a tiered ladder.

### Exit stack (current)
- **FAST_EXIT OFF** (both L1 +0.20%/2min and L2 disabled).
- **Trailing**: arms at peak ≥ **0.45%** (tp_min), exits at peak − **0.25%** (pullback).
- **Runner Stretch-Trail** (LONG, high-ATR) **ON**: entry_atr≥1.0 & peak≥0.70% → hands tight trailing off to a 0.5×-peak *signed*-stretch trail (lets IDU-class runners run).
- **EMA13 strict cross exit.**
- **SL −0.70% base**, ATR-widened (×1.5, floor −1.20%).
- **Pattern fixed exits**: C1 SHORT fixed_sl −0.70%.
- **BE is OFF** (be_levels_enabled false). RSI-handoff OFF. Regime-change-exit OFF.

### Liquidity-aware sizing (LIVE, unproven forward — see Go-Live Watch)
① per-pair cap: notional ≤ min(0.10% × pair 24h vol, $500k). ② gross cap: Σ open notional ≤ balance × **30×**. ③ redeploy-leftover **ON**, hard count cap 10. min_investment_size $100 (skip if throttled below).

## Active entry filters
- **Global Volume Filter ON** (re-enabled 2026-06-02): Min L 0.70 · Min S 0.50 · Max S 1.10 + capitulation override (BTC RSI<30 & slope<0 & GV≤2) · LONG rescue $50M/ceiling 0.60 · **SHORT rescue $0 (load-bearing — do NOT add)**.
- **ADX Δ × BTC ADX ON**: SHORT block ΔADX≥2.0 & BTC ADX≥24 · LONG block ΔADX 1.0-2.0 & BTC ADX 18-30.
- **Pair ADX Direction = `rising` BOTH sides** (shipped 2026-06-02): blocks any entry where pair ADX ≤ prev (falling-ADX = momentum exhaustion). Re-activation of the May 28 `rising` setting; counter = PAIR_ADX_DIR.
- **BTC 1h Slope MIN floor (shipped 2026-06-03):** `btc_1h_slope_min_short = -0.60` (block SHORT when BTC 1h slope < -0.60 = shorting into a steep 1h crash/exhaustion). `btc_1h_slope_min_long = 0.0` (DISABLED — plumbed but off). 0 = disabled, negative = active. Counter = BTC_1H_SLOPE_MIN_GATE.
- BTC RSI×ADX LONG `70-100:40,60-65:22-25,60-65:27-30,55-60:20-25,50-55:99-100` · SHORT `30-35:30,35-40:20-26,45-50:25,0-30:25-30`.
- BTC ADX gate L/S [18,40] · BTC Gap×ADX LONG `0.10-0.20:0-22` · RngPos×ADXΔ LONG `90-95:0.0-0.3` / SHORT `5-10:1.0-2.0` · SHORT EMA20-slope-min 0.06 · Entry Quality Score filter ON.
- Full filter list = `trading_config.json`. No pair blacklist currently active.

## Active multipliers (target "both"; inv cap 2× · lev cap 2×)
- **BTC_60-65_22-25 LONG = 2.0×inv × 1.5×lev = 3.0× eff** (lev-stacked; cross-batch 73.7% WR / N=38). · BTC_60-65_28-30 LONG 2.0×.
- BTC_35-40_33-36 SHORT 2.0× · BTC_25-30_28-30 SHORT 2.0×.
- PAIR_35-40_30-35 SHORT = **1.0× (demoted 2026-06-02)**.

## Active Pattern Cell Ship rules
- **C1 SHORT** = 2.0×inv × 1.5×lev (3× eff) + fixed_sl −0.70%.
- **W2 SHORT** = 2.0× × 1.5× (3× eff) · **W6 SHORT** = 2.0× · **UNMATCHED LONG** = 2.0× (no fixed exits — deliberate, Jun 1).
- C4/C8 L+S, W1 L+S, W2 LONG, W4 L+S, W6 LONG, UNMATCHED SHORT = all baseline 1.0× (observation).

## 🚨 LOCKED GO-LIVE WATCH — liquidity sizing (gross 30× = biggest systemic knob)
Unproven forward; read before each live checkpoint. All signals in UI (Gross gauge, Filter Blocks, Liquidity Sizing table).
- **Revert 30×→25× if a correlated event causes ≥25% equity DD in <15min.** (Clean SLs ≈ 30 × 0.70% ≈ 21%; ≥25% means SLs gapped through — the tail the cap exists to bound.)
- **GROSS_CAP_SKIP > ~20% of entry attempts with Gross pinned red** → 30× is the binding limiter; decide raise vs accept. <5% = fine.
- **① slippage verdict (live, N≥10):** capped fills should show ≤ uncapped entry slippage. Clean read = same-pair (aggregate skews thin-pair, so raw capped>uncapped is EXPECTED — not a verdict).
- **REDEPLOY_OPEN must stay 0 at ~$3k balance** (① dormant here). If it fires → bug (equal-split divisor / margin gate).
- ① per-pair cap is dormant at this balance + paper-invisible (paper fills ~0 slippage); ② gross cap is the one that can bind now via multiplier stacks. ① verdict is **live-only**.

## Active watchlist & locked revert gates (pending — apply at next ≥30-trade checkpoint)
- **Global Vol filter (Jun 2):** revert a SHORT side if would-be-blocked SHORTs ≥55% WR on N≥10 fresh · drop LONG floor→0 if it clips ≥3 runner winners (peak≥+3%)/batch with no offsetting saves. Watch VOL_GATE / VOL_GATE_MAX_SHORT counters fire.
- **BTC 1h Slope MIN floor `-0.60` SHORT (Jun 3, below-gate ship):** evidence = BTC 1h slope < −0.60 SHORT = **0W/4L** (SEI, XRP, BTC, JTO) across 2 sessions; not covered by existing filters (these had rising pair-ADX + BTC ADX outside the 24-30 kill-zone). N=4 / 2-3 correlated events = below the N≥30 bar → tighter gate. **Revert `btc_1h_slope_min_short`→0 if would-be-blocked (slope<−0.60) SHORTs show ≥50% WR on N≥6 fresh, OR if BTC_1H_SLOPE_MIN_GATE blocks a would-be-winner SHORT on 3+ separate sessions.** Re-confirm at next ≥30-trade checkpoint. LONG floor stays DISABLED (its loser zone is the FLAT band -0.10..0, structural N=60/6-dates — a separate band-filter, not yet built).
- **Pair ADX Direction = `rising` (Jun 2, below-gate re-activation):** evidence = falling-ADX entries 1W/9L cross-batch+today (LONG 1/8, SHORT 0/2), all samples 06-02 + prior May 28 A/B (`rising` beat `both`). **Revert a side to `both` if would-be-blocked (falling-ADX) entries on that side show ≥50% WR on N≥6 fresh.** Also: if PAIR_ADX_DIR blocks >15% of that side's attempts (vs ~1% historical base rate) → regime shift, re-examine. Watch PAIR_ADX_DIR counter.
- **ADX Δ filter (Jun 2):** SHORT would-be-blocked ≥50% WR on N≥6 → blank SHORT rule / re-disable · LONG ≥55% WR on N≥10 → blank LONG rule. WATCHLIST: filter is **blind to FALLING ADX** (signed match, only catches rising/climax). If a batch bleeds on sharply-falling-ADX entries → add a signed negative-range rule (parser needs signed lo-hi fix first).
- **PAIR_35-40_30-35 SHORT demote:** re-promote 1.5×→2.0× only if N≥5 fresh AND ≥70% WR. Any single fresh Δ$≤−$60 → keep 1.0× indefinitely.
- **UNMATCHED LONG 2× (Jun 1):** revert to 1.0× if next batch <65% WR OR Total$ negative; ✗ HARMFUL in Multiplier Cell table → revert immediately.
- **Runner Stretch-Trail (Jun 1):** revert if Runner Trail table net-gain-vs-tight ≤0 on N≥5; if %Max<40% try k=0.3.
- **BTC_60-65_22-25 LONG 3× eff:** keep if N≥5 & WR≥70% & Total$+ · WR 55-70% → drop lev to 2.0× inv-only · WR≤55% → drop both. Any single leveraged Δ$≤−$60 → drop lev immediately.

## Metrics tracked
**Avg P&L %** (leverage-invariant — the comparison metric) · Daily Compound Return · per-cell verdict (Multiplier / Pattern-Cell Ship tables) · Filter Blocks counters · Gross gauge (% + $ tooltip) · entry/exit slippage · Pattern C/W trackers (observation-only).

## Known risks
- **Gross 30× tail**: a −3.3% correlated gap ≈ −100% equity (per-position SLs are primary protection; gross cap is flash-crash insurance).
- **Multiplier amplification on fat-tail losers**: C6/C1/W2 cohorts are 62-74% WR *winners* net-dragged by a few 3×-amplified losers → fix is surgical sizing on the specific losers, NOT blocking the cohort ("high-WR-but-net-losing" trap).
- **LONG book structurally weak** (negative in most cross-batch dimensions); LONG vol floor occasionally clips high-ATR runners.

## Do not change for now
- **BE stays OFF** (proven not to help; runner-trail + trailing is the current exit thesis (FAST_EXIT OFF too)).
- **SHORT volume rescue stays $0** (mirroring LONG $50M would un-block dead-tape SHORTs like SEI).
- **Don't add more entry filters reflexively** — 15+ already, diminishing returns. Exit-side + sizing is the active lever.

## Next ~1-week forward-test focus
1. **Liquidity-sizing live validation** — the Go-Live Watch signals (Gross gauge, GROSS_CAP_SKIP, ① slippage, REDEPLOY_OPEN=0).
2. **Global Vol filter** firing correctly + not clipping runners (VOL_GATE counters).
3. **ADX Δ filter** + the recent BTC RSI/ADX cross-filter openings holding.
4. Combined **Avg P&L %** vs the recent −$1,941/207-trade baseline; SHORT-side drag (dead-tape, C1/W2 amplified losers).
5. **Reset partition:** after the user's reset, analyze only `opened_at ≥ reset timestamp` (don't pool pre-reset trades).
