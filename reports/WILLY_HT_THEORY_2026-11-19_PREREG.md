# WILLY_HT THEORY — pre-registration (frozen 2026-11-19 ≈ 14:20 UTC by file time, before any new result is computed)

Research only. No engine / config / test change (report-only commit).

## Question
The operator: "the market is bullish and more coins are in frenzy — WILLY_HT is my bet against your analysis; find WHY it does not
work in the backtest and what fine-tuning fixes it." This study audits our own null with the gaps the earlier work left open.

## What is already known (not re-tested, see the inventory in the report)
Exit grid TP 2–6 × SL 4–12 × hold 90 m–48 h, structural / BE / trail / grace exits (WILLY_EXIT_V2, 180 cells) · 46-variable 1D+2D
W/L screens + null (HT_TP6_WL, WL_DEEP, STACK3D, WILLY_DEEP) · regime axes on the old exit (WILLY_BACKTEST_WHY H6, FRENZY_LONG_REGIME)
· entry timing FIRST_GREEN / RED2 / random bar on the OLD +1.25 / −4 / 90 m exit and on all WILLY (WILLY_THEORY 10-16).

## Gaps this study fills (each frozen here)
1. **Entry mechanics × TODAY's big-target exit (never tested jointly).** Entries, all inside the same HT episode (decisive red bar
   = the bar today's engine routes to HT: first red bar with a readable turnover R ≥ 1.0, DISLOC ≤ 1 %, nomiss 24 h volume floor):
   - **E0 RED** (today): market at the next 5m open after the decisive red bar.
   - **E1 CONFIRM**: market at the next open after the first 5m bar (within 12 bars) that CLOSES above the red bar's high.
     Theory: demand came back (late buyers / short covering visible) → skips the dips that keep fading.
   - **E2 DIP3**: resting maker bid at red close × 0.97, valid 120 min. Theory: liquidity to panic sellers inside a live frenzy.
   - **E3 VWAP**: resting maker bid at the spike-anchored VWAP (red close ÷ (1 + vs_vwap / 100)), valid 120 min. Theory: the
     average cost of the frenzy cohort is support. (Skipped when the VWAP is above the red close.)
   - **E4 BREAKOUT**: buy-stop at the spike high × 1.001, valid 240 min, fill at max(level, bar open) + slip. Theory:
     re-acceleration / squeeze of the shorts that faded the spike — enter only when the second leg proves itself.
2. **Exit grid (NET %, as live):** TP ∈ {3, 5, 8} × SL ∈ {3, 4, 6} × hold ∈ {180 min, 1440 min} = 18 exits. TODAY = TP5 / SL4 / 1440.
   Family = 5 entries × 18 exits = **90 combos**.
3. **FAILED_SPIKE cut** (signal bar vs spike VWAP): today ≤ 0 → block. Read the blocked cohort on E0 × TODAY and the cuts
   ≤ −1 / ≤ +1 / ≤ +2 (4 cuts, descriptive; a cut change must pass the same bar as a combo).
4. **Turnover cut that defines HT:** R ≥ {0.7, 1.0 (today), 1.5, 2.5} on E0 × TODAY (descriptive; same bar to propose).
5. **Regime at SIGN granularity, in WINDOW (day) units**, on E0 × TODAY and on the best combo: BTC 1d EMA50 vs EMA200 (sign),
   BTC prior-day return (sign), BTC 4h EMA20 slope (sign), alt breadth (DESIGN median), Frenzy breadth = number of ARMED Frenzy
   episodes market-wide in the prior 24 h (DESIGN median). Per half-year: in / out N · days · WR · avg. A regime switch PASSES
   only if the in-regime avg is > 0 in ≥ 3 of 4 half-years (each with ≥ 8 distinct days) AND the day-clustered 95 % lower bound of
   the in-regime avg over the whole 2025-01 → 2026-11 is > 0 AND in − out > 0 in ≥ 3 of 4 halves. Today's regime values are
   printed so the operator can see which side "now" is on.

## Periods
DESIGN = 2025-01-13 → 2025-12-31 · HOLDOUT = 2026-01-01 → 2026-11-17 (new for every entry variant; earlier studies read these
months for other rules — stated) · D = 2023-H2 (every WILLY setup routed as HT, market caps unknown — cross-check only, already
read ≥ 3× for other rules) · MASTER = the 49 real HT fills (10-23 → 11-15) + the 3 newer export fills (11-18 / 11-19).

## Costs
Taker 0.05 %, maker 0.02 %, 0.10 % slippage on every market fill (entries E0 / E1 / E4, stops, time exits), limit entries
(E2 / E3) fill at the bid with no slippage when a bar's low trades THROUGH the bid by ≥ 1 tick (≈ 0.01 %), TP is a resting maker
limit that fills when traded through. Real funding over the hold. Stop checked before TP inside a bar. 1m bars when every day is
cached, else 5m (share reported).

## Portfolio
One open trade per pair (pair-held → that episode is skipped), HT slots 10, plain WILLY off (DL 587).

## Pass bar for any proposal (all must hold)
① DESIGN avg gain vs TODAY (E0 × TP5/SL4/1440) beats the family-wise random-entry null: p < 0.05 for the max-over-90 statistic
(200 replicates; each replicate = 5 independent random-entry draws per episode, uniform over the 5m bars in [decisive bar,
+4 h], each priced on the 18 exits). ② HOLDOUT avg > 0 and > TODAY's HOLDOUT avg. ③ MASTER re-priced: avg > 0 and ≥ TODAY's
master avg. ④ avg > 0 in ≥ 3 of 4 half-years (or a regime explanation that passes §5). ⑤ LOO (drop the best pair, the best day)
stays > 0. ⑥ 30–50 % haircut on the DESIGN gain still > 0. ⑦ Kelly at the one-sided 95 % Wilson WR floor (n = distinct days) > 0.
⑧ DCR at one size (X = 0.61 × equity) before → after, every calendar day, full history and per half.

## Labels (path first)
WORKED = +1 % gross before −4 % net within 24 h · REACH = best gross % before the stop (24 h) · reach +5 before −4 · path groups
NEVER_WORKED / GAVE_BACK / CAPTURED / other — read before booked P&L for every entry.

## Outputs
reports/WILLY_HT_THEORY_2026-11-19.md (+ _combos.csv, _regime.csv, _master.csv, _trades.csv, _event_study.png);
scripts/willy_ht_theory_build.py, scripts/willy_ht_theory_eval.py.
