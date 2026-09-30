#!/usr/bin/env python3
"""BTC-spike follow study v3 — 1-MINUTE bars (operator, 2026-09-30).

Why v3: the 5m simulator assumed the adverse extreme first inside every 5m bar. On fast, high-ATR alts a single 5m bar spans
both +0.9 % and −0.7 %, so a real +0.72 % MOVR trade scored as a −0.70 stop — 3 of the operator's 7 spike trades were mis-scored.
Here every fill is walked on 1m bars (adverse-first per MINUTE), and the simulator is first VALIDATED on the operator's real
manual trades.

Same pre-registered design as v2 (scripts/btc_spike_follow_study.py): event = BTC +1.0 %/30 min ∧ 24 h-high breakout ∧ volume
≥ 3× median, ≥ 4 h apart · universe = top-10 tradeable alts by 24 h volume · exit = the live LONG momentum exit (stop −0.70,
runner arms at +0.40, floor max(peak − 1×ATR, +0.10), 4 h max) · units = events, bootstrap CI · control = the same universe at
the same clock time 1 day earlier. Entry offsets 0 / 10 / 20 / 30 / 45 min after the event bar closes (the operator's entries
this morning were 24–50 min after BTC's 12:35 UTC detection).
Usage: venv/bin/python scripts/btc_spike_follow_1m.py [--validate path/to/orders.csv]"""
import argparse, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_spike_follow_study as V2
import ccxt

ROOT = V2.ROOT
K1 = os.path.join(ROOT, "reports", "backtest_cache", "k1m")
ONDEMAND = os.path.join(ROOT, "reports", "backtest_cache", "k1m_spike")
os.makedirs(ONDEMAND, exist_ok=True)
EX = ccxt.binanceusdm({"enableRateLimit": True})
MIN = 60_000
_full = {}


def bars_1m(pair, t_from, t_to):
    """1m bars [t_from, t_to): the year cache when it covers the window, else one cached exchange fetch per window."""
    if pair not in _full:
        f = os.path.join(K1, pair + ".csv"); _full[pair] = None
        if os.path.exists(f):
            d = pd.read_csv(f, usecols=["open_time", "o", "h", "l", "c"]).drop_duplicates("open_time").set_index("open_time").sort_index()
            _full[pair] = d
    d = _full[pair]
    if d is not None and len(d) and d.index[0] <= t_from and d.index[-1] >= t_to - MIN:
        w = d.loc[t_from:t_to - MIN]
        if len(w) >= (t_to - t_from) // MIN * 0.95:
            return w
    fn = os.path.join(ONDEMAND, f"{pair}_{t_from}_{t_to}.csv")
    if os.path.exists(fn):
        return pd.read_csv(fn).set_index("open_time")
    rows, cur = [], t_from
    while cur < t_to:
        for attempt in range(3):
            try:
                got = EX.fetch_ohlcv(pair[:-4] + "/USDT:USDT", "1m", since=cur, limit=1500); break
            except Exception:
                time.sleep(1 + attempt); got = []
        if not got:
            break
        rows += [r for r in got if r[0] < t_to]
        cur = got[-1][0] + MIN
        if len(got) < 1500:
            break
    w = pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c", "v"]).drop_duplicates("open_time").set_index("open_time")[["o", "h", "l", "c"]]
    w.to_csv(fn)
    return w


def walk(w, entry, atr, hold_min=240, entry_px=None, exit_fn=None):
    """A LONG exit on 1m bars. Inside each minute the price path follows the candle: a falling candle (close < open) goes
    open → high → low → close, a rising one open → low → high → close (the standard OHLC path; plain adverse-first mis-scored
    the operator's QNT 13:02 trade, which peaked +0.40 and THEN fell inside one minute). exit_fn(peak) → stop level; default =
    the live momentum exit: −0.70 until the +0.40 arm, then max(peak − 1×ATR, +0.10). Returns (net %, peak %)."""
    w = w.loc[entry:entry + hold_min * MIN - MIN]
    if len(w) < 5:
        return None
    e = float(entry_px or w.o.iloc[0]); peak = 0.0
    if exit_fn is None:
        exit_fn = lambda pk: -0.70 if pk < 0.40 else max(pk - atr, 0.10)
    for o, h, l, c in zip(w.o.values, w.h.values, w.l.values, w.c.values):
        path = (o, h, l, c) if c < o else (o, l, h, c)
        for px in path:
            v = (px / e - 1) * 100 - V2.FEE
            stop = exit_fn(peak)
            if v <= stop:
                return stop, peak
            peak = max(peak, v)
    return (w.c.iloc[-1] / e - 1) * 100 - V2.FEE, peak

def atr_at(pair, t):
    d = V2.D.get(pair)
    if d is None or len(d) == 0:
        return 1.0
    i = d.index.searchsorted(t, side="right") - 1
    return float(d.atrp.iloc[max(i, 0)])


def validate(path):
    m = pd.read_csv(path); m = m[m.entry_strategy == "MANUAL"].sort_values("opened_at")
    print("VALIDATION on the operator's real manual trades (1m walk from the real fill price):")
    for _, r in m.iterrows():
        t = int(pd.Timestamp(r.opened_at, tz="UTC").timestamp() * 1000)
        t0 = (t // MIN) * MIN
        w = bars_1m(r.pair, t0, t0 + 250 * MIN)
        atr = float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) else atr_at(r.pair, t)
        # the fill minute's bar also holds prices from BEFORE the click: start at the next full minute
        res = walk(w, t0 + MIN, atr, entry_px=r.entry_price)
        print(f"  {pd.Timestamp(r.opened_at):%H:%M} {r.pair:<10} actual {r.pnl_percentage:+.3f} (peak {r.peak_pnl:+.2f}, {r.close_reason:<16}) "
              f"sim {res[0]:+.3f} (peak {res[1]:+.2f})" if res else f"  {r.pair} no data")


def run(events, offsets, control=False):
    rows = []
    for t0 in events:
        tc = t0 - 288 * V2.BAR if control else t0
        start = tc + V2.BAR                                   # the event bar has closed
        for p in V2.universe(tc, 10):
            atr = atr_at(p, tc)
            w = bars_1m(p, start, start + (max(offsets) + 245) * MIN)
            if w is None or len(w) < 30:
                continue
            for off in offsets:
                res = walk(w, start + off * MIN, atr)
                if res:
                    rows.append((t0, p, off, res[0], res[1]))
    return pd.DataFrame(rows, columns=["ev", "pair", "off", "r", "peak"])


def line(name, R):
    e = R.groupby("ev").r.mean(); lo, hi = V2.ci(e.values)
    arm = (R.peak >= 0.40).mean() * 100
    print(f"{name:<44} events={len(e):>3} fills={len(R):>4} | per event {e.mean():+.3f} [95% {lo:+.3f},{hi:+.3f}] positive {(e > 0).mean()*100:3.0f}% "
          f"| fill WR {(R.r > 0).mean()*100:3.0f}% | reached +0.40 {arm:3.0f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--validate", default=None); A = ap.parse_args()
    if A.validate:
        validate(A.validate); print()
    P = V2.events(1.0, True, True)
    offs = (0, 10, 20, 30, 45)
    R = run(P, offs); C = run(P, offs, control=True)
    print(f"PRIMARY events: {len(P)} · top-10 tradeable alts · live LONG momentum exit · 1m bars")
    for off in offs:
        line(f"BTC spike, entry +{off} min", R[R.off == off])
        line(f"   control (1 day earlier), entry +{off} min", C[C.off == off])
    late = R[R.off.isin((20, 30, 45))]; latec = C[C.off.isin((20, 30, 45))]
    line("BTC spike, entries +20..45 min (operator timing)", late)
    line("   control, same offsets", latec)
    R.to_csv(os.path.join(ROOT, "reports", "backtest_cache", "btc_spike_1m_fills.csv"), index=False)
