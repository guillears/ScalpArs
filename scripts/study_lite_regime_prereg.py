#!/usr/bin/env python3
"""LITE MACRO-REGIME study 2026-10-15 — pre-registration (refuses overwrite; --grids appends the operator's BTC-grid section once).
Usage: venv/bin/python scripts/study_lite_regime_prereg.py           # base pre-registration
       venv/bin/python scripts/study_lite_regime_prereg.py --grids   # append the BTC REGIME GRID section (operator addition), once
"""
import os, sys, hashlib, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "LITE_REGIME_PREREG.txt")
FROZEN = ["reports/LITE_PROTECT_rows.csv", "reports/TICK_STUDY_2026-10-15_cohort.csv", "reports/lite_wlsweep/matrix.csv",
          "reports/backtest_cache/universe_daily_top50.csv", "reports/backtest_cache_2025/universe_daily_top50.csv",
          "reports/backtest_cache/k1d/BTCUSDT.csv", "reports/backtest_cache_2025/k1d/BTCUSDT.csv"]
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_19-55-15.csv")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


BASE = """LITE MACRO-REGIME STUDY — PRE-REGISTRATION (frozen {ts} UTC, before any regime-variable x LITE-outcome table was computed)
Operator 2026-10-15: "there must be a macro condition of LITE being successful or not, if TP and SL is not a variable."
Already seen before freezing (declared): era means of rm_base_pct (2025 -0.165, 2026-H1 +0.297, 2026-H2 +0.201, master by batch), the
coordinator's brief (2025 high-ATR -0.63 vs 2026 +0.37 %/fill; LITE_BTC1H_FALLING split), the 4 live LITE fills' P&L (DEEP +2.80, DUSK -2.54,
EPIC -3.43, ZK -2.93; Oct-14/15 = 2 days). No regime variable has been joined to outcomes.

0. SCOPE  research only; no engine/config/trading_config/templates/tests edit; no commit; no bot API; cache-only (no Binance call planned:
   BTC daily from 2024-06-01 and 5m for ~650 pairs 2024-06 -> 2026-10 are cached). Funding cache starts 2025-12-30 -> 2026 only.

1. COHORT  frozen TICK_STUDY cohort via reports/LITE_PROTECT_rows.csv (BT 2025 828 [era '2025', incl. 2026-01-01..03] / 2026-H1 369 /
   2026-H2 255) + MASTER 20 (B18-B21) + LIVE closed FRENZY_LITE in {live} (DEEP, DUSK, EPIC, ZK). WILLY fills (TICK_STUDY cohort, rec_pct)
   are used ONLY as a regime input (rolling WILLY result), never as outcome.
2. OUTCOME  y = rm_base_pct (today's exit +3 resting TP / -3 / 12 h). Live = booked pnl_percentage. Day d = UTC day of entry.
   Per-day outcome over horizon H in {{1d, 3d, 7d}}: pooled avg %/fill of fills entered in [d, d+H) (also N, WR); only days with >= 1 fill
   in the window are scored. WEEK unit: ISO week (Mon 00:00 UTC), feature at week start, outcome = the week's pooled avg.
3. VARIABLES (all at day start 00:00 UTC from data strictly before it; daily bars = CLOSED days <= d-1):
   BTC (10): r1, r3, r7, r30 (close-to-close %), dma20, dma50, dma200 (close / SMA - 1, %; sign = above/below), rv7 (std of 7 daily log
     returns), rvtr = rv7 / rv30, off30 (close / max 30d high - 1, %).
   ALT (7): universe U(d) = top-50 USDT perps by quote volume of day d-1 (own rank from cached k1d; BTC + stablecoins excluded).
     br_up3 / br_up7 = share of U with positive 3d / 7d return; br_ma7 / br_ma20 = share of U closing above own SMA7 / SMA20;
     alt_rs7 = equal-weight U 7d return - BTC 7d return; voltr = sum quote volume (all cached pairs) 7d avg / 30d avg;
     disp7 = mean over the 7 prior days of the cross-sectional std of U daily returns.
   FUNDING (1, 2026 only): fund3 = mean funding rate of U over the 3 prior days (cache gap 2025 -> could not test cross-year; H1 -> H2 only).
   FRENZY ACTIVITY (6): market-wide FRENZY spikes rebuilt from cached 5m klines with the engine's spike rule (services/frenzy.py:
     30-min return >= 5 % AND last-hour quote volume >= 20x the pair's normal hour [median of <= 720 hourly volumes ending >= 24 h earlier,
     >= 240 needed] AND >= $2M). Episode approximation: a lead bar starts a NEW spike unless a lead bar or a >= 100x-volume bar occurred on the
     pair in the prior 24 h (the engine's 24 h-no-state-bar rule, VWAP-hold leg ignored -> documented approximation; parity = share of BT
     LITE fills whose entry_frenzy_spike_at matches a rebuilt spike within 10 min).
     spk24 / spk72 = spikes in the prior 24 h / 72 h; flags96 = distinct pairs with a spike in the prior 96 h (= FRENZY flags);
     ft2 / ft12 = share of the last 20 spikes whose follow-through window ENDED before d (spike close -> +2 h / +12 h close > spike close);
     ftr12 = mean +12 h return of those 20 spikes.
   OWN ROLLING (8): lite_n10 / lite_n20 = avg y of the last 10 / 20 BT LITE fills CLOSED before d (close = te + base_xm min);
     lite_d3 / lite_d7 = avg y of BT LITE fills closed in the prior 3 / 7 days; willy_n10 / willy_n20 / willy_d3 / willy_d7 likewise on
     BT WILLY rec_pct (close = te + 90 min cap, conservative). Within-BT only (master / live have too few fills to roll).
   TOTAL 32 variables (31 cross-year + fund3 H1->H2 only).
4. CUTS (frozen from the TRAIN period only, never refit on test): NAT = sign for r1/r3/r7/r30/dma*/alt_rs7/lite_*/willy_*/ftr12, train
   median for the rest; T3 = train terciles (top vs bottom). Spearman rho on day-level (H) series.
   TESTS per variable x unit {{1d,3d,7d,week}} x stat {{rho, NAT diff, T3 diff}} = 32 x 4 x 3 = 384 (x 2 directions for discover/confirm).
   NAT/T3 diff = pooled fill avg (good side) - (other side), fill-weighted, per period.
5. NULL  7-day block shuffle of the FEATURE series relative to outcomes within year (blocks of 7 consecutive days permuted), R = 500,
   same statistic; per-test two-sided p and family-wise p (max |z| across the 128 1d/3d/7d/week tests of a stat family per year).
   Bootstrap P = day-block (7-day blocks) bootstrap of the NAT diff, P(same sign).
6. DISCOVER -> CONFIRM  cut + direction chosen on 2025, confirmed on 2026 iff same sign and bootstrap P >= 0.90; and 2026 -> 2025;
   and 2026-H1 -> 2026-H2. A variable SURVIVES iff it confirms BOTH ways (2025->2026 and 2026->2025) at the 1d or 3d unit with NAT or T3
   and its 2025 and 2026 block-null p <= 0.10 each. H1 -> H2 reported as a third check.
7. ATR  per-fill reads by ATR tercile of entry_atr_pct, cuts = 2025 BT terciles frozen, applied to all blocks. Per variable: NAT diff
   inside each tercile per year (does the regime matter more for high-ATR coins?).
8. POLICY (survivors only, + the equity-curve switches always):
   P1 "trade LITE only when regime = good" (good side = train side with higher avg); P2 "skip HIGH-ATR LITE fills when regime = bad".
   Walk-forward: (a) train 2025 -> test 2026, train 2026 -> test 2025, train H1 -> test H2; (b) expanding monthly WF over 2025-01..2026-10
   (first 3 months train-only; each month's cut + side from all data before that month). Per-signal accounting: skipped fill = 0.
   Report delta %/fill (kept vs all) and delta total % (= -sum skipped y), two-sided (sum of skipped winners lost vs skipped losers avoided),
   per-week contribution (no single week >= 50 % of the positive benefit; >= 8 distinct weeks with positive contribution),
   leave-one-month-out (delta total sign per dropped month), 7-day-block bootstrap P(delta total > 0).
   EQUITY-CURVE switches (priced same way, shadow fills during the pause, rolling stats on the full real+shadow sequence by close time):
   E1/E2 pause when the last 10 / 20 closed LITE fills average < 0, resume when the last K = 5 (E1a/E2a) or 10 (E1b/E2b) closed fills
   average >= 0 (4 variants); E3/E4 skip day d when lite_d3 / lite_d7 < 0 (2 variants).
   (Slot / portfolio effects of skipped fills are NOT re-simulated — skips are counted as 0 in a fixed BT sequence.)
9. GATE (all must hold for a candidate): delta total > 0 with bootstrap P >= 0.90 in BOTH 2025 and 2026 (out-of-sample direction),
   LOMO-stable (delta total > 0 when any single month is dropped, in both years), >= 8 distinct weeks carrying positive contribution and
   no week >= 50 % of it, then a 30-50 % in-sample haircut on the projected delta. Master + live = refute-only (a sign reversal on master is
   reported; it cannot support). Pass -> observe-only Scout line with frozen cut; nothing is armed from this study (no arm before review).
10. MULTIPLICITY  384 variable tests + policy variants (2 x survivors + 6 equity-curve) + the grid section (if appended). Every count is
   reported with the null expectation.

FROZEN INPUTS (sha256[:16]):
{hashes}
"""

GRIDS = """
=== ADDENDUM {ts} UTC — BTC REGIME GRID (operator: "2D or 3D based on BTC"), appended BEFORE any grid cell was priced ===
Cuts frozen from the TRAIN period only (sign or train median); cell read requires >= 8 distinct ISO weeks in the period; cells with fewer
are listed as "thin". Per cell: LITE N, distinct days, distinct weeks, WR, avg %/fill, and the same for HIGH-ATR (top 2025 tercile) fills.
 G1 BTC r7 sign x BTC dma50 sign x rv7 (train median high/low)                         = 8 cells
 G2 BTC r30 sign x off30 band {{near: > train median; far: <= median and > -15 %; deep: <= -15 % (incl. the 15-24 % washed-out band)}}
    x BTC r3 sign                                                                       = 12 cells
 G3 BTC r1 x r7 x r30 signs (8 cells) + collapsed {{aligned up / mixed / aligned down}} (3)  = 11 cells
 G4 BTC dma200 sign x br_ma20 (train median)                                            = 4 cells
 G5 BTC r7 sign x ft12 (train median)                                                   = 4 cells
 G6 BTC rv7 (train median) x alt_rs7 sign                                               = 4 cells
 TOTAL 6 grids, 43 cells.
 READ  discover 2025 -> confirm 2026 and 2026 -> 2025 (cell avg sign agreement weighted by fills; per-grid stat = fill-weighted
   between-cell variance of cell means (eta^2-like) vs a 7-day block-shuffle null of the cell labels, R = 500; grid-wise p).
   Cell-level confirm: train cell avg sign == test cell avg sign with test bootstrap P >= 0.90 (7-day blocks).
 POLICY  per grid: G-trade = trade LITE only in cells whose TRAIN avg > 0 (cells thin in train count as tradeable = default on);
   G-skipATR = skip HIGH-ATR LITE fills in cells whose TRAIN high-ATR avg < 0. Walk-forward 2025->2026, 2026->2025, H1->H2, and expanding
   monthly WF; same two-sided accounting, LOMO, week concentration, bootstrap and GATE as section 9. 6 grids x 2 policies = 12 variants.
 MULTIPLICITY total = 384 tests + 43 cells (+ 6 grid-wise) + 12 grid policies + 6 equity-curve + 2 x survivors policies.
"""


DEV1 = """
=== DEVIATION 1 {ts} UTC — appended BEFORE any regime x outcome table (feature QA only, outcomes not yet joined) ===
 QA of day_features: the day-(d-1) top-50-by-volume universe is pump-biased (coins enter the top-50 BECAUSE they just ran; alt_rs7
 averaged +13 %, disp7 11 %). Fixed definitions: U(d) = top-50 by the 30-day AVERAGE quote volume ending d-1 (BTC + stables excluded);
 alt_rs7 = MEDIAN U 7d return - BTC 7d return; disp7 = mean of the 7 prior days' cross-sectional IQR/1.349 of U daily returns.
 Spike-rule parity (rebuilt vs engine entry_frenzy_spike_at, same pair, <= 10 min): 2025 87.0 % / 2026-H1 79.7 % / 2026-H2 76.9 %.
"""


def main():
    if "--dev1" in sys.argv:
        txt = open(OUT).read()
        if "DEVIATION 1" in txt:
            sys.exit("deviation 1 already present - refusing")
        with open(OUT, "a") as f:
            f.write(DEV1.format(ts=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
        print("appended deviation 1")
        return
    if "--grids" in sys.argv:
        txt = open(OUT).read()
        if "BTC REGIME GRID" in txt:
            sys.exit("grid addendum already present - refusing")
        with open(OUT, "a") as f:
            f.write(GRIDS.format(ts=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
        print("appended grid section")
        return
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists - refusing to overwrite")
    hs = "\n".join(f"  {p}: {sha(os.path.join(ROOT, p))}" for p in FROZEN) + f"\n  {LIVE}: {sha(LIVE)}"
    with open(OUT, "w") as f:
        f.write(BASE.format(ts=datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), live=LIVE, hashes=hs))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
