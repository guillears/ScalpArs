# 🔥 FRENZY sleeve — design for approval (2026-10-02, NOT built)

Operator brief: follow a pair once it is in a volume frenzy; trade it with 2× investment × 1× leverage (20×). Evidence and history: RUN_SLEEVE_PLAN.md.

## 1 · The FRENZY flag (per pair)
- ON: a 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M, none in the prior 24 h. Anchored VWAP starts at that bar.
- OFF: 24 h without the staircase state, or 96 h after the spike, whichever first. (NOT "closed below EMA50/EMA200": MOVR did that 9× in 3 days and kept running.)
- Scan: once per 5m close; shortlist from the 24 h ticker (up ≥ 15 %, ≥ $20M) + pairs already flagged; same walk as scripts/scout_staircase.py, moved to a
  pure module services/frenzy.py (shared by engine, scout, tests). Flag state persisted in a ledger table (restart-proof, like the SURGE ledger).

## 2 · FRENZY_LONG (the only traded leg)
- Entry: the staircase state turns ON after ≥ 1 h off — ≥ 2 h after the spike ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume
  ≥ 100× normal ∧ 24 h volume ≥ $100M. Market entry on the next tick. No pair-EMA / BTC / momentum filter (not part of the tested rule).
- Exit (own function, intercepts before the momentum stack in both paths, like SURGE): stop −3 % · trail arms at +5 %, closes 1.5 % below the best price ·
  12 h cap. Reasons FRENZY_STOP / FRENZY_TRAIL / FRENZY_TIME.
- Limits: one position per pair · at most 2 FRENZY positions open · no attempt cap and no pause after stops (year data: both remove winners).
- Year test of exactly this cell: 1,209 trades · 41 % won · +5.2 / −3.1 · +0.23 / +0.34 per trade · by day +0.30 [+0.02, +0.58] · longest losing run 11.
  The only passing cell of 80; not hostile-reviewed; stops of 1–2 % are zero or negative.

## 3 · FRENZY_SHORT — observe only
Every EMA50 / EMA200 break of a flagged pair is logged (time, price, distance from peak, breadth) with NO trade. Five year tests found no short rule.

## 4 · Sizing — 2× investment × 1× leverage (operator)
- frenzy_long_invest_mult 2.0 · frenzy_long_lev_mult 1.0 (absolute-assign, never re-multiplied; liquidity cap, gross cap and leverage brackets still apply).
- What one stop costs: 3.11 % × 20× × 2 = 124 % of a normal slot's margin. With equal split over 4 slots that is ≈ 31 % of the account per stop
  (≈ 16 % at 1×; a normal momentum stop is ≈ 4–6 %). The year's longest losing run was 11.
- Recommendation: 0.25× (≈ 4 % of the account per stop) until the kill / keep bar reads. The multiplier is a UI field either way.
- Live only: the exchange backstop sits at 2.5 % — inside the 3 % stop. Live needs the backstop moved for this sleeve or the stop set to −2.2 %.

## 5 · Kill / keep bar (pre-committed)
- NO automatic off (operator 2026-10-02: paper mode, no auto-kill rule). The sleeve is switched only by the operator's toggle.
- Judged at 40 closed trades: keep if per trade > 0 after costs AND ≥ 12 winners; else off. (A 10-trade bar kills a 40 %-win sleeve by luck too often.)
- Cohort: full-size fills after the deploy only. Below every promotion gate → an operator-directed armed override, recorded as such.

## 6 · Surfaces (D11 / D12)
- Config (config.py + trading_config.json + UI input + load + save): frenzy_long_enabled, frenzy_short_observe, frenzy_spike_ret_pct 5, frenzy_spike_vol_mult 20,
  frenzy_state_vol_mult 100, frenzy_min_volume_usd 100e6, frenzy_min_hours 2, frenzy_max_hours 96, frenzy_stop_pct 3, frenzy_trail_arm_pct 5,
  frenzy_trail_giveback_pct 1.5, frenzy_max_hold_minutes 720, frenzy_max_slots 2, frenzy_long_invest_mult, frenzy_long_lev_mult. (No kill-verdict field.)
- Order columns: entry_frenzy_spike_at, entry_frenzy_hours, entry_frenzy_vwap, entry_frenzy_vs_vwap_pct, entry_frenzy_vol_mult, entry_frenzy_run_pct.
- Tables: "🔥 FRENZY" (flagged pairs now · fires · observed short breaks) in the UI and in BOTH text exports; post-exit regret whitelist; filter-block
  counters FRENZY_SLOTS / FRENZY_LIQ; scout staircase section reads the same module.
- Tests: pure-rule tests (flag on / off, state, exit walk, sizing absolute-assign) added to tests/.

## 7 · Not covered
Delisted pairs · order-book slippage on thin pairs · exchange position limits (MOVR ≈ $5k at 25×, so 2× size often will not fill in live) · no hostile review of the long cell yet.

## 8 · DESIGN REVIEW 2026-10-02 (scripts/frenzy_long_review.py → FRENZY_LONG_REVIEW_2026-10-02.md) — the long cell does NOT hold on the strict ruler
- Stop 3 / trail 5-1.5: +0.29 as tested → +0.19 with 0.10 % slippage → +0.05 gap-aware (+0.01 / +0.07); 95 % by day [−0.18, +0.28]; best 5 % removed −0.43;
  5 of 9 months negative (Aug carries it, +115 of +67 total); longest losing run 11; top 3 pairs = 179 % of the total.
- Stop 2 / 2.2 (needed for the live 2.5 % backstop): −0.07 / −0.01. Trail 3/1.5: −0.17.
- NOT "few trades": median 4 per day, max 14, every day of the year; median hold 19 min. Max-2-open barely binds (1,206 of 1,209).
- Account path (25 % slot, 20×, compounding): 2× and 1× wiped out; 0.5× ×0.01; 0.25× ends ×0.49 with a −97 % drawdown.
- ATR at entry (not in the design; after-the-fact cut): 5m ATR 1–2 % +0.23/+0.24 (348 trades), ATR ≥ 2 % −0.01 / −0.07 (851 trades) → the 3 % stop is
  < 1.5 ATR on 70 % of entries (inside the noise). Hypothesis to freeze: enter only when the stop is ≥ 1.5× the 5m ATR, or scale the stop with ATR.
- Flag-off rule is safe: the entry itself needs volume ≥ 100× normal and an hour of closes above the VWAP, so no long can fire once the frenzy fades.
- VERDICT: do not arm as designed. Candidates before a build: ATR-gated entry (frozen, re-tested on the strict ruler) · the ride-it exit (close below VWAP).

## 9 · FOLLOW-UP TESTS 2026-10-02 (scripts/frenzy_long_followup.py → FRENZY_LONG_FOLLOWUP_2026-10-02.md) — nothing passes; the ATR gate is the best version
- SEEN = ≥ $100M pairs (1,220 entries) · UNSEEN = $20–100M pairs never tested (989). Strict ruler.
- ATR GATE (enter only when 5m ATR ≤ 2 %; stop 3, trail 5/1.5): SEEN 357 trades · 44 % won · +0.25 (+0.39 / +0.10) · by day [−0.14, +0.65] · losing run 6 ·
  account at 2 %/stop ×1.57 (−46 %). UNSEEN 235 · 39 % · +0.10 (+0.27 / −0.09) · [−0.42, +0.61] · ×1.06 (−37 %). Same direction on both, range spans zero.
  The rest (ATR > 2 %): −0.04 SEEN / −0.14 UNSEEN → the gate removes the losing 70 %.
- ATR-SCALED stops (1.5× / 2× ATR): about zero on both sets — widening the stop does not rescue high-ATR entries.
- RIDE-IT exit (sell on a close below the VWAP): SEEN −0.9 to −1.0 per trade at every stop; UNSEEN +0.9 to +1.2 but Jan–Apr +3 / May–Sep −0.6, carried by one
  +1,409 % trade; losing runs 25–38; account at 2 %/stop −79 % to −100 %. NOT usable. (Does not reproduce the earlier +0.87 / +0.37 swing result — that test
  used different entries and plain fills; unreconciled, treat the earlier figure as not confirmed.)
- DESIGN CHANGE: entry adds "5m ATR(14) ≤ 2 %" (new field frenzy_max_atr_pct 2.0, new column entry_frenzy_stop_atr). ≈ 590 entries/year on $20M+ pairs ≈ 2 per day.
- Still an operator-directed ARMED override below every gate. Sizing: 2 % of the account per stop = investment multiplier ≈ 0.125× at 20×.

## 10 · Dashboard marking + final operator decisions (2026-10-02)
- Top Pairs table: flagged pairs PINNED to the top even outside the Top-N · badge in the Pair cell (🔥 flagged · 🔥★ long setup ON · 🔥⏳ still flagged ≥ 32 h)
  with the detail on hover · Block Reason cell shows the FRENZY status for flagged pairs (momentum gate moves to the hover) · ATR cell green ≤ limit / red above.
  No new column, no filter chip.
- Operator: build ARMED at 2× investment × 1× leverage, no automatic-off rule, shorts observe-only. One stop at 2× ≈ 32 % of the account (on record).
