# WILLY late joiners — PRE-REGISTRATION (frozen 2026-10-22, before any outcome of this study was computed)

Operator case (2026-10-22): BTRUSDT spike bar closed 17:30 UTC (+37 %, dead coin). Its 24 h volume was < $20M (`frenzy_min_volume_usd`)
at the spike, so it joined the FRENZY shortlist only at the 18:05 pass → WILLY refused: "7 bars old > the 3-bar late window — no entry A".
Hypothesis: dead-coin pumps reach the shortlist only AFTER their spike, so the 3-bar late window systematically misses them.

## Theory (written before any number)
For: a pump on a coin with no prior volume is the purest "frenzy" flow; WILLY buys the first red 5m bar after the flag, i.e. the first
pull-back of a fresh pump — the same mechanism as an on-time A. Against: (1) by the time the pair joins (the 24 h volume needs the pump
itself to pass $20M) the pump is 20–60 min old: the first red bar after joining is more likely the START of the mean reversion than a dip
inside a continuing run; (2) dead / low-cap coins carry R = 24 h vol / mcap > 1 → the full WILLY is refused anyway (turnover filter,
DL 314 evidence R > 1 negative) and only the ¼-size WILLY_HT would take them; (3) thin books → worse fills / gap-through stops.

## Engine rules mirrored (read from the code, 2026-10-22)
- Shortlist (`services/trading_engine.frenzy_shortlist`): max(|24 h change|, 24 h low→high range) ≥ `frenzy_shortlist_change_pct` 15 ∧
  24 h quote volume ≥ `frenzy_min_volume_usd` $20M ∧ not blacklisted ∧ ASCII; ≤ 25 by move (+ follow list ≤ 40). The walk mirrors the
  move / volume / blacklist / ASCII legs (rolling 288 closed 5m bars); it does NOT mirror the 25-cap, the new-listing / Alpha /
  non-COIN pre-filters, or the follow list (a pair already followed is "warm" → it would arm A on time). Listed as blind spots.
- Episode / flag = `scripts/study_frenzy_swing_short_walk.py` (96.5 % bar parity with `services.frenzy.frenzy_walk`), data of the flat-base
  study (`reports/FRENZY_FLATBASE_SHORT_2026-10-22_data/episodes_compact.pkl`: verified ∧ ever on the shortlist while flagged; incl. delisted).
- L = lateness = (first shortlisted-while-flagged bar − spike bar) in 5m bars = exactly the engine's `frenzy_willy_late_bars` at the first
  pass that reads the pair cold. Engine today (R0): A arms iff L ≤ `frenzy_willy_a_cold_max_bars` 3; armed on the join bar j; opens on the
  first closed RED 5m bar k (close < open; bar j counts) with close(k) ≤ close(j) + 60 min (`frenzy_willy_red_max_wait_minutes`).
- Turnover at the red bar: R = 24 h quote vol (288 closed 5m bars to the red close) / (circulating supply × red close);
  R < 1.0 → full WILLY · R ≥ 1.0 → WILLY_HT (¼ size, DL 341) · supply unknown → UNREAD (live refuses both; reported, not counted).
  Supply = the bot's own source `reports/cache_vol_mcap/bapi_supply.json` ∪ `…FLATBASE…_data/winloss_supply.json` (today's supply —
  an approximation for older dates; stated).
- Exit = trading_config.json today: TP +1.25 net resting maker (books the line) · stop −4 net (gap-aware: a bar opening through fills at
  its open) · 90-min cap · late TP +0.25 from minute 30 · `scout_willy_timecap.evaluate_late(book="live")` (the master / scout walker).
- Fill = red close + 8 s → the 1m bar opening at the red close, its open × (1 + 0.10 %) (live ruler: 8 s + ≥ 0.10 slip, latency study).
  Path = 1m klines (data.binance.vision daily zips, 0 API weight; fapi only for days not yet archived), the entry minute skipped.
  Fees in net_pct (0.045 % per side, the scout's accounting).

## Rules compared (pre-registered family of 3 candidates vs R0)
- **R0 (today)**: L ≤ 3 → arm at the join bar, wait 60 min from arming. L > 3 → no trade.
- **R1-30**: also L ∈ 4…6 (joined ≤ 30 min after the spike close) → arm at the join bar, wait 60 min from arming.
- **R1-60**: also L ∈ 4…12 (≤ 60 min) → same.
- **R2**: also L ∈ 4…12 → arm at the join bar, but the red bar must close ≤ spike close + 60 min (wait measured from the spike).
Late-joiner cohort = the fills each rule ADDS (L > 3). On-time cohort = R0 fills (L ≤ 3), same walker, same period.

## Blocks
③ backtest = spikes 2025-01-01 → 2026-10-08 15:23 UTC (WILLY deploy) · ② master = 2026-10-08 15:23 → 2026-10-14 18:36 (paper era) ·
① live = 2026-10-14 18:36 → now. ①② = (a) the engine's OWN refusals from CloudWatch (`/aws/elasticbeanstalk/SCALPARS-env-2/var/log/
web.stdout.log`, "late window — no entry A", 14-day retention) priced the same way, (b) the walk's episodes in the window (parity check:
walk late joiners vs CloudWatch refusals), (c) on-time comparator = real WILLY fills (master `stack_pct`; live export `pnl_percentage`)
plus the same fills re-walked by this walker (parity). Eras for the bar: 2025 · 2026-H1 · 2026-H2 (≥ 2026-07-01, incl. ②① walk episodes).

## Statistics (per rule, per cohort)
N · days · pairs · WR · avg net % per position (equal weight; HT is a position like any other) · median · p5 · worst · stop rate ·
minutes spike → entry · routing (full / HT / UNREAD) and per-route avg · $-weighted avg (HT × 0.25) · day-clustered bootstrap 95 % CI
(5,000 resamples of UTC entry days) · per-era · per-pair concentration (top-1 / top-2 share of |Σ|, avg without the top 2 pairs) ·
best 5 % removed · trades added per week (unconstrained, and with WILLY's 1 slot + HT's 1 slot occupied by on-time fills → refused).
**Fair null (descriptive)**: lag-matched on-time comparator — the on-time episodes re-entered as if armed at spike + L bars, L drawn
from the late-joiner L distribution — separates "entered later" from "a different kind of coin".

## Pre-registered bar to RECOMMEND a rule change (all must hold for the rule's late-joiner cohort, routed trades, UNREAD excluded)
1. avg ≥ on-time avg (same walker, same period) − 0.20 %/trade;
2. avg > 0 in 2026-H1 AND in 2026-H2;
3. day-bootstrap 95 % lower bound > 0;
4. N ≥ 30 on ≥ 10 distinct days;
5. not driven by 1–2 pairs: avg > 0 with the top-2 pairs (by Σ) removed, and top-2 share of the positive Σ < 50 %.
Then a 30–50 % in-sample haircut on the projected Δ and a tight pre-committed revert gate. If several rules pass: R1-30 unless the
L 7–12 increment alone also meets 1–2. Otherwise **R0 stays**; a rule meeting 1 and 4 but failing 2/3/5 → observe-only line at most.
Never re-fit X, the wait, or the cohort after seeing outcomes.
