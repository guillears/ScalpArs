#!/usr/bin/env python3
"""🌋 MOVR-likes of the last two months (operator, 2026-10-01: "search for others like MOVR in the last 2 months … find the new
sleeve shape"). EXPLORATION, not a test: every split below is read after the fact on ~2 months.

  universe  every futures pair in the 5m cache, extended to now with public futures klines (futures-only pairs INCLUDED —
            exits are walked on futures 1m bars, stop first inside a bar = conservative for a +3.09 / −1.51 exit)
  frenzy    5m bar, CLOSED-bar inputs: 24 h volume ≥ 30× the pair's normal day (median 24 h volume over the 30 days ending
            2 days earlier) ∧ 24 h return ≥ +30 % ∧ 5m ATR(14) ≥ 2 % ∧ 24 h volume ≥ $20M. Bars < 6 h apart = one episode.
            (30× instead of the frozen 100× so the volume DOSE can be read: 30–50 / 50–100 / 100–200 / 200+.)
  entry     at a minute close inside a frenzy bar, LONG when price ≤ 5 % below the highest high of the prior 4 h
  exit      +3.09 % / −1.51 % gross, 30-min limit, one position per pair, 1 min pause; cost 0.09 % fees + 0.02 % slippage
Usage: venv/bin/python scripts/frenzy_search_2m.py → reports/backtest_cache/frenzy2m_trades.csv + frenzy2m_episodes.csv"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import movr_48h_study as M  # noqa: E402  (cached kline fetch)

ROOT = M.ROOT; K5 = os.path.join(ROOT, "reports", "backtest_cache", "k5m_full"); EXT = os.path.join(ROOT, "reports", "backtest_cache", "k5m_ext")
FAPI = "https://fapi.binance.com/fapi/v1/klines"; BAR = 300_000; COST = 0.11; HOLD = 30; TP, SL = 3.09, 1.51
START = int(pd.Timestamp("2026-08-01", tz="UTC").timestamp() * 1000); NOW = int(time.time() // 300 * 300 * 1000)
TR = os.path.join(ROOT, "reports", "backtest_cache", "frenzy2m_trades.csv"); EP = os.path.join(ROOT, "reports", "backtest_cache", "frenzy2m_episodes.csv")


def frame(pair):
    d = pd.read_csv(os.path.join(K5, pair + ".csv"))
    f = os.path.join(EXT, pair + ".csv")
    if not os.path.exists(f):
        r = M.get(f"{FAPI}?symbol={pair}&interval=5m&startTime={int(d.open_time.max()) + BAR}&limit=1500") or []
        os.makedirs(EXT, exist_ok=True)
        pd.DataFrame([(int(x[0]), x[1], x[2], x[3], x[4], x[5], x[7]) for x in r if int(x[6]) < time.time() * 1000],
                     columns=d.columns).to_csv(f, index=False); time.sleep(0.25)
    try:
        d = pd.concat([d, pd.read_csv(f)])
    except Exception:
        pass
    d = d.astype(float).drop_duplicates("open_time").set_index("open_time").sort_index(); d.index = d.index.astype("int64")
    if len(d) < 288 * 40:
        return None, []
    q24 = d.qvol.rolling(288).sum(); norm = q24.shift(288 * 2).rolling(288 * 30).median()
    pc = d.c.shift(1); tr = np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs()))
    # everything shifted one bar: known DURING the bar it is attached to
    d["volx"] = (q24 / norm).shift(1); d["ret24"] = ((d.c / d.c.shift(288) - 1) * 100).shift(1); d["q24"] = q24.shift(1)
    d["atr"] = (tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100).shift(1); d["hi4"] = d.h.rolling(48).max().shift(1)
    d["ret7d"] = ((d.c / d.c.shift(288 * 7) - 1) * 100).shift(1); d["lo30"] = d.l.rolling(288 * 30).min().shift(1)
    d["frenzy"] = (d.volx >= 30) & (d.ret24 >= 30) & (d.atr >= 2) & (d.q24 >= 20e6)
    t = d.index.values[d.frenzy.values & (d.index.values >= START)]
    eps = [] if not len(t) else [(int(e[0]), int(e[-1])) for e in np.split(t, np.where(np.diff(t) > 6 * 3600_000)[0] + 1)]
    return d, eps


def episode(pair, d5, t0, t1):
    m = M.klines(FAPI, pair, "1m", t0 - 4 * 3600_000, min(t1 + BAR + HOLD * 60_000, NOW), 1500)
    if len(m) < 300:
        return [], None
    ts = m.index.values.astype("int64"); h, l, c, q, tb = m.h.values, m.l.values, m.c.values, m.q.values, m.tb.values
    st = d5.reindex(ts // BAR * BAR); fz = st.frenzy.fillna(False).values.astype(bool); hi4 = st.hi4.values
    i0 = int(np.searchsorted(ts, t0)); p0 = c[max(i0 - 1, 0)]; out = []; free = 0
    for i in range(max(i0, 240), len(c) - 1):
        if i < free or not fz[i]:
            continue
        top = max(hi4[i], h[i - 240:i + 1].max()); depth = (1 - c[i] / top) * 100
        if depth < 5:
            continue
        e = c[i]; H, L = h[i + 1:i + 1 + HOLD], l[i + 1:i + 1 + HOLD]
        hs = np.nonzero(L <= e * (1 - SL / 100))[0]; ht = np.nonzero(H >= e * (1 + TP / 100))[0]
        a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
        r, k = ((c[min(i + HOLD, len(c) - 1)] / e - 1) * 100, HOLD) if a == b == 10**9 else ((-SL, a + 1) if a <= b else (TP, b + 1))
        s = st.iloc[i]; lo15 = l[i - 14:i + 1].min(); hi15 = h[i - 14:i + 1].max()
        out.append(dict(pair=pair, eid=f"{pair}:{t0}", t=int(ts[i]), r=round(r, 4), mins=k, volx=s.volx, ret24=s.ret24, atr=s.atr, q24=s.q24,
                        ret7d=s.ret7d, off30lo=(e / s.lo30 - 1) * 100, depth=depth, hrs=(ts[i] - t0) / 3600e3, vs_onset=(e / p0 - 1) * 100,
                        ret5=(e / c[i - 5] - 1) * 100, ret15=(e / c[i - 15] - 1) * 100, pos15=(e - lo15) / max(hi15 - lo15, 1e-12),
                        tb5=tb[i - 4:i + 1].sum() / max(q[i - 4:i + 1].sum(), 1e-9), vol1x=q[i] / max(q[i - 20:i].mean(), 1e-9)))
        free = i + k + 1
    w = d5.loc[t0:t1]; a24 = d5.loc[t1:t1 + 24 * 3600_000]
    ep = dict(pair=pair, eid=f"{pair}:{t0}", start=pd.Timestamp(t0, unit="ms"), hours=(t1 - t0) / 3600e3 + 1 / 12, frenzy_bars=int(w.frenzy.sum()),
              volx_on=w.volx.iloc[0], volx_max=w.volx.max(), ret24_on=w.ret24.iloc[0], ret24_max=w.ret24.max(), atr_max=w.atr.max(), q24_max=w.q24.max(),
              ret7d_on=w.ret7d.iloc[0], peak_vs_onset=(w.h.max() / p0 - 1) * 100, end_vs_onset=(w.c.iloc[-1] / p0 - 1) * 100,
              after24=(a24.c.iloc[-1] / w.c.iloc[-1] - 1) * 100 if len(a24) > 12 else np.nan, n=len(out), net=sum(x["r"] - COST for x in out))
    return out, ep


if __name__ == "__main__":
    rows, eps_ = [], []; t_ = time.time(); files = sorted(glob.glob(os.path.join(K5, "*.csv")))
    for n_, f in enumerate(files):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT"):
            continue
        try:
            d5, eps = frame(pair)
        except SystemExit:
            print("skip (fetch)", pair, flush=True); continue
        for t0, t1 in eps:
            try:
                r, ep = episode(pair, d5, t0, t1)
            except SystemExit:
                print("skip episode (fetch)", pair, t0, flush=True); continue
            if ep:
                rows += r; eps_.append(ep)
        if eps:
            print(f"[{n_ + 1}/{len(files)}] {pair}: {len(eps)} episodes · trades {len(rows)} · {time.time() - t_:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(TR, index=False); pd.DataFrame(eps_).to_csv(EP, index=False); print("done", len(eps_), "episodes", len(rows), "trades")
