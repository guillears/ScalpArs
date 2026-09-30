#!/usr/bin/env python3
"""🧬 Parity gate for scripts/scout_features.py (DECISION_LOG 152): rebuild the entry_* stamps for REAL bot fills from public
candles (as the scout does for missed moves) and compare with what the bot stamped live. Fills of the last ~40 h (the 5m fetch
depth): reports/MASTER_POOL_stacked.csv + every ~/Downloads orders export. The live bot reads the FORMING bar and the scout the
last CLOSED bar, so each fill is compared at both alignments (closed bar before the fill, and the one after) and the closer one is
kept — as scripts/validate_against_master.py F2 accepts shift 0 or +1.
Usage: venv/bin/python scripts/scout_features_parity.py"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opportunity_scout as S  # noqa: E402
import scout_features as SF  # noqa: E402

BAR = SF.BAR
COLS = ["entry_rsi", "entry_adx", "entry_gap", "entry_ema_gap_5_8", "entry_atr_pct", "entry_range_position", "pre_pair_volume_ratio_closed",
        "entry_dist_from_ema13_pct", "entry_pair_ema20_ema50_gap_pct", "entry_btc_rsi", "entry_btc_adx", "entry_btc_ema20_slope",
        "entry_btc_trend_gap_pct", "entry_btc_rsi_1h", "entry_btc_1h_slope", "entry_btc_off24h_pct", "entry_btc_r72_pct",
        "entry_btc_above72_pct", "entry_btc_off30d_high_pct", "entry_bull_pct", "entry_bear_pct", "entry_pair_1h_ema20_200_gap_pct",
        "entry_pair_1d_ndi", "entry_btc_4h_ema50_200_gap_pct", "entry_btc_ema50_100_gap_pct"]
LIVE = {"pre_pair_volume_ratio_closed": "entry_pair_volume_ratio"}      # scout name → the live column it is read against
ALIGN = ("entry_rsi", "entry_btc_rsi", "entry_adx", "entry_btc_adx")    # alignment chosen on bounded-scale columns (abs error)


def main():
    now = int(time.time() * 1000); last_closed = (now // BAR - 1) * BAR
    frames = [pd.read_csv(os.path.join(SF.ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)]
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    o = pd.concat(frames, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    o["ms"] = (pd.to_datetime(o.opened_at, utc=True, format="ISO8601") - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)
    o = o[(o.ms >= now - 40 * 3600_000) & o.entry_rsi.notna()].copy()
    if not len(o):
        print("no fills in the last 40 h"); return
    cfg = S.load_cfg(); scan, rank, cutoff, limit = S.bot_universe(cfg)
    in_now = [p for p in scan if rank.get(p, 999) <= limit]
    btc = S.k5("BTC/USDT:USDT", last_closed, 1500)
    alts = {p: S.k5(p[:-4] + "/USDT:USDT", last_closed) for p in set(in_now) | set(o.pair)}
    market = (btc, S._frame(S._retry(S.EX.fetch_ohlcv, "BTC/USDT:USDT", "1h", limit=1000)),
              S._frame(S._retry(S.EX.fetch_ohlcv, "BTC/USDT:USDT", "4h", limit=1000)),
              S._frame(S._retry(S.EX.fetch_ohlcv, "BTC/USDT:USDT", "1d", limit=10)),
              S.k5("ETH/USDT:USDT", last_closed, 400), {p: alts[p] for p in in_now if alts.get(p) is not None})
    ext, cache, rows = {}, {}, []
    for r in o.itertuples():
        t_prev = (int(r.ms) // BAR - 1) * BAR                          # the last bar CLOSED before the fill
        best = None
        for t in (t_prev, t_prev + BAR):
            ev = dict(type="FILL", pair=r.pair, side="UP" if r.direction == "LONG" else "DOWN", bar_ts=t, start_ts=t)
            if alts.get(r.pair) is None or t not in alts[r.pair].index:
                continue
            if r.pair not in ext:
                ext[r.pair] = SF.fetch_pair_extras(S.EX, S._retry, r.pair, t)
            st = SF.event_features(ev, alts, cache, market, lambda p: ext[p])
            e = [abs(float(st[c]) - float(getattr(r, c))) for c in ALIGN if st.get(c) is not None and pd.notna(getattr(r, c, np.nan))]
            err = float(np.mean(e)) if e else float("inf")
            if best is None or err < best[0]:
                best = (err, st, t - t_prev)
        if best:
            rows.append(dict(pair=r.pair, opened=r.opened_at, shift=best[2] // BAR,
                             **{c: (best[1].get(c), getattr(r, LIVE.get(c, c), None)) for c in COLS}))
    if not rows:
        print("no fill had candles in the fetched window"); return
    print(f"{len(rows)} fills compared (scout rebuild vs live stamp)\n")
    print(f"{'column':34s} {'n':>3s} {'median |Δ|':>11s} {'p90 |Δ|':>9s} {'corr':>6s}")
    for c in COLS:
        a = np.array([(x[c][0], x[c][1]) for x in rows if x[c][0] is not None and pd.notna(x[c][1])], dtype=float)
        if len(a) < 2:
            print(f"{c:34s} {len(a):3d}   (too few)"); continue
        dd = np.abs(a[:, 0] - a[:, 1])
        cc = np.corrcoef(a[:, 0], a[:, 1])[0, 1] if np.std(a[:, 0]) > 0 and np.std(a[:, 1]) > 0 else float("nan")
        print(f"{c:34s} {len(a):3d} {np.median(dd):11.4f} {np.percentile(dd, 90):9.4f} {cc:6.3f}")


if __name__ == "__main__":
    main()
