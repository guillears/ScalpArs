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
- **Pair ADX Direction: LONG=`both`, SHORT=`rising`** (LONG reverted to `both` 2026-06-03). SHORT blocks falling-ADX (7-pool falling-SHORT 45% WR/−0.24 vs rising 62%). LONG `rising` was REVERTED — on the proper 7-pool proxy it was backwards (cut falling-LONG breakeven +0.002 while keeping rising-LONG losers −0.174); the original LONG ship rested on the full pool's broken `adx_delta` field (6 negative/558 vs 59/222 in the 7-pool). Counter = PAIR_ADX_DIR.
- **BTC 1h Slope MIN floor (shipped 2026-06-03):** `btc_1h_slope_min_short = -0.60` (block SHORT when BTC 1h slope < -0.60 = shorting into a steep 1h crash/exhaustion). `btc_1h_slope_min_long = 0.0` (DISABLED — plumbed but off). 0 = disabled, negative = active. Counter = BTC_1H_SLOPE_MIN_GATE.
- **BTC-Accel Chase Filter (shipped 2026-06-03, STATEFUL, LONG only) — DISABLED 2026-06-04 (A/B test):** `evo_chase_filter_long_enabled=FALSE` (was true), `evo_chase_window_min=30`. Operator-directed OFF to test the hypothesis that the LONG bleed is driven by **C-pattern levels (C4/C6/C7), not by BTC-chasing**. When enabled, blocks a LONG when live BTC EMA20 slope > the slope at the most recent LONG opened within 30min. Engine tracks `_last_long_open_ts/_btc_ema20_slope`. SHORT side `evo_chase_filter_short_enabled=false`. Counter = BTC_ACCEL_CHASE_LONG (will read 0 while off).
- BTC RSI×ADX LONG `70-100:40,60-65:22-25,60-65:27-30,55-60:20-25,50-55:99-100` · SHORT `30-35:30,35-40:20-26,45-50:25,0-30:25-30`.
- BTC ADX gate L/S [18,40] · BTC Gap×ADX LONG `0.10-0.20:0-22` · RngPos×ADXΔ LONG `90-95:0.0-0.3` / SHORT `5-10:1.0-2.0` · SHORT EMA20-slope-min 0.06 · Entry Quality Score filter ON.
- Full filter list = `trading_config.json`. **Pair blacklist = 10 pairs (trimmed from 23 on 2026-06-03; BTCUSDT re-whitelisted same day).**
  - **BLACKLISTED (10):** `BNBUSDT, ENAUSDT, FILUSDT, MUSDT, RAVEUSDT, TRUMPUSDT, VVVUSDT, XAGUSDT, XAUUSDT, ZECUSDT`. Of these: 4 evidenced losers (BNB/FIL/TRUMP/VVV, N≥15) + XAG/XAU (commodities) + ENA/MU/RAVE/ZEC (no-data, new-listing/illiquid).
  - **🔓 RELEASED / WHITELISTED from blacklist (12, on 2026-06-03):** `ADAUSDT, ASTERUSDT, BCHUSDT, DOGEUSDT, ERAUSDT, HYPEUSDT, ICPUSDT, LABUSDT, LINKUSDT, PUMPUSDT, SKYAIUSDT, WLFIUSDT`. Released for thin pre-filter evidence (N<15). **REVERT GATE: re-blacklist any released pair at ≤35% WR on N≥10 fresh (current-stack) trades.**
  - **👁 NO-TRADE / TRACK-ONLY (`no_trade_pairs`, new field, 2026-06-03):** `BTCUSDT`. Stays in the Top-Pair-by-Volume universe (subscribed, scanned, **displayed**) but entries are BLOCKED (counter PAIR_NO_TRADE). This is the *third* pair state, distinct from blacklist (removed from universe) and tradeable. Used for BTC: visible for reference, never opens a position (matches the edge<fee evidence while keeping it on the dashboard). BTC is NOT in pair_blacklist and NOT tradeable.

## Active multipliers (target "both"; inv cap 2× · lev cap 2×)
- **BTC_60-65_22-25 LONG = 2.0×inv × 1.5×lev = 3.0× eff** (lev-stacked; cross-batch 73.7% WR / N=38). · BTC_60-65_28-30 LONG 2.0×.
- BTC_35-40_33-36 SHORT 2.0× · BTC_25-30_28-30 SHORT 2.0×.
- PAIR_35-40_30-35 SHORT = **1.0× (demoted 2026-06-02)**.

## Active Pattern Cell Ship rules
- **C1 SHORT** = 2.0×inv × 1.5×lev (3× eff) + fixed_sl −0.70%.
- **W2 SHORT** = 2.0× × 1.5× (3× eff) · **W6 SHORT** = 2.0× · **UNMATCHED LONG** = 2.0× (no fixed exits — deliberate, Jun 1).
- C4/C8 L+S, W1 L+S, W2 LONG, W4 L+S, W6 LONG, UNMATCHED SHORT = all baseline 1.0× (observation).
- **Extension multiplier (LONG) Ext0.4-0.6_L family = 1.0× (DEMOTED 2026-06-04 from 2×)** — tags kept for tracking, sizing killed. Cross-batch FULL turned ✗ HARMFUL (base N=5/−$235, +QuietVol N=5/−$252); +SlowADX N=3 noise. Long side has no gross edge → "caps for losers."

## 🚨 LOCKED GO-LIVE WATCH — liquidity sizing (gross 30× = biggest systemic knob)
Unproven forward; read before each live checkpoint. All signals in UI (Gross gauge, Filter Blocks, Liquidity Sizing table).
- **Revert 30×→25× if a correlated event causes ≥25% equity DD in <15min.** (Clean SLs ≈ 30 × 0.70% ≈ 21%; ≥25% means SLs gapped through — the tail the cap exists to bound.)
- **GROSS_CAP_SKIP > ~20% of entry attempts with Gross pinned red** → 30× is the binding limiter; decide raise vs accept. <5% = fine.
- **① slippage verdict (live, N≥10):** capped fills should show ≤ uncapped entry slippage. Clean read = same-pair (aggregate skews thin-pair, so raw capped>uncapped is EXPECTED — not a verdict).
- **REDEPLOY_OPEN must stay 0 at ~$3k balance** (① dormant here). If it fires → bug (equal-split divisor / margin gate).
- ① per-pair cap is dormant at this balance + paper-invisible (paper fills ~0 slippage); ② gross cap is the one that can bind now via multiplier stacks. ① verdict is **live-only**.

## Active watchlist & locked revert gates (pending — apply at next ≥30-trade checkpoint)
- **Global Vol filter (Jun 2):** revert a SHORT side if would-be-blocked SHORTs ≥55% WR on N≥10 fresh · drop LONG floor→0 if it clips ≥3 runner winners (peak≥+3%)/batch with no offsetting saves. Watch VOL_GATE / VOL_GATE_MAX_SHORT counters fire.
- **BTC-Accel Chase Filter LONG — DISABLED 2026-06-04 (A/B test; was shipped Jun 3 below-gate).** Original evidence: "BTC EMA20 slope improving vs last LONG (30min)" = 7-batch proxy block cohort 30.8% WR, N=26, Σ−3.1% (net-losing), caught the 06-03 4-loss cluster 0/4. Now OFF to test whether the LONG bleed is **C-levels (C4/C6/C7), not chasing**. **RE-ENABLE GATE: if LONGs that the filter WOULD have blocked (BTC-accel-since-last-LONG-in-30min) show ≤35% WR on N≥10 fresh while OFF → the chase signal is real, turn back on.** Conversely, if those would-be-blocked LONGs are ≥50% WR → chase was a false signal, keep off and pursue the C-level filters (C7-no-W / C1+C6 / C6-LONG gates). Re-evaluate at next ≥30-trade checkpoint. (window=30 setting retained for if re-enabled.)
- **BTC 1h Slope MIN floor `-0.60` SHORT (Jun 3, below-gate ship):** evidence = BTC 1h slope < −0.60 SHORT = **0W/4L** (SEI, XRP, BTC, JTO) across 2 sessions; not covered by existing filters (these had rising pair-ADX + BTC ADX outside the 24-30 kill-zone). N=4 / 2-3 correlated events = below the N≥30 bar → tighter gate. **Revert `btc_1h_slope_min_short`→0 if would-be-blocked (slope<−0.60) SHORTs show ≥50% WR on N≥6 fresh, OR if BTC_1H_SLOPE_MIN_GATE blocks a would-be-winner SHORT on 3+ separate sessions.** Re-confirm at next ≥30-trade checkpoint. LONG floor stays DISABLED (its loser zone is the FLAT band -0.10..0, structural N=60/6-dates — a separate band-filter, not yet built).
- **Pair ADX Direction = `rising` (Jun 2, below-gate re-activation):** evidence = falling-ADX entries 1W/9L cross-batch+today (LONG 1/8, SHORT 0/2), all samples 06-02 + prior May 28 A/B (`rising` beat `both`). **Revert a side to `both` if would-be-blocked (falling-ADX) entries on that side show ≥50% WR on N≥6 fresh.** Also: if PAIR_ADX_DIR blocks >15% of that side's attempts (vs ~1% historical base rate) → regime shift, re-examine. Watch PAIR_ADX_DIR counter.
- **ADX Δ filter (Jun 2):** SHORT would-be-blocked ≥50% WR on N≥6 → blank SHORT rule / re-disable · LONG ≥55% WR on N≥10 → blank LONG rule. WATCHLIST: filter is **blind to FALLING ADX** (signed match, only catches rising/climax). If a batch bleeds on sharply-falling-ADX entries → add a signed negative-range rule (parser needs signed lo-hi fix first).
- **PAIR_35-40_30-35 SHORT demote:** re-promote 1.5×→2.0× only if N≥5 fresh AND ≥70% WR. Any single fresh Δ$≤−$60 → keep 1.0× indefinitely.
- **UNMATCHED LONG 2× (Jun 1):** revert to 1.0× if next batch <65% WR OR Total$ negative; ✗ HARMFUL in Multiplier Cell table → revert immediately.
- **Extension multiplier LONG demote (Jun 4):** RESTORE 2× only if `Ext0.4-0.6_L` reaches **N≥15 fresh AND Total$>0** (current stack). Watch the Extension Multiplier Performance table — now fires at 1.0× but stays tagged, so the cohort's true 1× edge is observable before any re-promote.
- **C1+C6 SHORT toxic-combo (Jun 4, WATCHLIST — observation only, NOT shipped):** by C-signature (UI Pattern-C Combination Tracker convention, W ignored) the 7-pool C6-SHORT bleed is NOT C6 itself (C6-sig N=24, 67% WR, ≈flat −$135) — it's the **C1+C6 co-occurrence: N=3, 0W/3L, −0.69%, −$667, NP 67%** (capitulation that's also macro-over-extended → bounces; both are multiplied cells so the loss is amplified). N=3 ≪ gate. **GATE: cap effective multiplier to 1.0× on C1+C6 SHORT if it holds ≤30% WR on N≥6 fresh.** Until then track in the Pattern-C Combination Tracker; do nothing.
- **C7 LONG-alone / no-W (Jun 4, WATCHLIST — observation only, NOT shipped):** C7 = Pair Countertrend Bounce; a LONG on a countertrend bounce with no W (trend) confirmation = buying a falling knife. Recent (7-pool+batch, BE-off), **C7 LONG with no W: N=3, 0W/3L, ≈−$265** (1000LUNC, ONDO, HOME — HOME this batch). Cross-ref: UI C-signature "C7" LONG row (W-ignored) = N=5, 20% WR, −$245. Definition matters — the *block target* is **C7 present AND no W match** (don't block C7 trades that carry trend confirmation). N=3 ≪ gate. **GATE: ship a LONG entry-block on "C7-match AND w_any=False" only if it holds ≤30% WR on N≥8 fresh.** Until then observe; do nothing.
- **C6 LONG (Jun 4, OBSERVATION-tracking — NOT a loser, do NOT block):** recent (7-pool+batch, BE-off) C6 LONG = N=7, 4W/3L, **57% WR**, Avg ≈−0.002% (flat), Tot −$46.6 but **de-multiplied +$18.9** (positive). The as-sized loss is entirely 2× amplification on RENDER (−172) + FET (−117); RENDER's 2× = the extension mult already demoted Jun 4, so the main amplifier is handled. C6 LONG always fires **no-W** (macro-over-extended, no trend confirmation) yet still de-mux positive → no action. **WATCH-FOR-DETERIORATION GATE: only consider a cap/block if C6 LONG turns net-negative DE-MULTIPLIED with ≤35% WR on N≥10 fresh.** Track the Pattern-C Combination Tracker "C6 LONG" row.
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
