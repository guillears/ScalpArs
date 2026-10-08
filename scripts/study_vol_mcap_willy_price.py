#!/usr/bin/env python3
"""📊 Oct-8 research (vol24h / mcap study) — price FRENZY_WILLY research cohorts on REAL TICKS under WILLY's live exit.

Cohorts (input CSV with columns pair, t0 [signal 5m close, ms], plus anything else passed through):
  A = first-flag events (flag_math/ev2.pkl cohort-a, engine-reachable) · B = fresh-ON bars of FRENZY_ENGINE_COHORT not taken by
  today's FRENZY_LONG / WIDE (built by study_vol_mcap_analyze.py).
Entry: first aggTrade print ≥ t0 + 8 s (live ≈ 8 s), FRENZY dislocation guard (|entry print / last print before t0 − 1| > 1 %
→ refused), fill = print × (1 + 0.035 % slip). Exit (services.frenzy.frenzy_willy_exit_for): net P&L = gross − 0.045 entry taker
− 0.045 exit taker; TP when a later print reaches net ≥ +1.0 → books +1.0 · stop when a later print reaches net ≤ −3.0 → books
that print's net − 0.05 slip (gap fill) · 60 min after entry → last print's net − 0.05 slip. Ticks from reports/backtest_cache/
ticks_q (first) else ticks. Local caches only — no network.
Usage: venv/bin/python scripts/study_vol_mcap_willy_price.py <in.csv> <out.csv>"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "reports", "backtest_cache")
DELAY, SLIP_IN, TAKER, SLIP_OUT, TP, SL, HOLD, DISLOC = 8_000, 0.035, 0.045, 0.05, 1.0, -3.0, 3_600_000, 1.0


def day(pair, d):
    for b in ("ticks_q", "ticks"):
        f = f"{C}/{b}/{pair}/{d}.npz"
        if os.path.exists(f):
            z = np.load(f); t = z["t"]; o = np.argsort(t, kind="stable"); return t[o], z["p"][o].astype(np.float64)
    return None


def ticks(pair, a, b, cache):
    ts, ps = [], []
    for d in pd.date_range(pd.Timestamp(a, unit="ms").normalize(), pd.Timestamp(b, unit="ms").normalize()):
        k = (pair, f"{d:%Y-%m-%d}")
        if k not in cache:
            cache[k] = day(*k)
        if cache[k] is None:
            return None, None
        ts.append(cache[k][0]); ps.append(cache[k][1])
    t = np.concatenate(ts); p = np.concatenate(ps); i, j = np.searchsorted(t, a), np.searchsorted(t, b)
    return t[i:j], p[i:j]


def one(pair, t0, cache):
    t, p = ticks(pair, t0 - 120_000, t0 + DELAY + HOLD + 60_000, cache)
    if t is None:
        return dict(st="nofile")
    pre = t < t0
    if not pre.any():
        return dict(st="noprint_pre")
    ref = p[pre][-1]; k = np.searchsorted(t, t0 + DELAY)
    if k >= len(t):
        return dict(st="noprint_post")
    raw = p[k]; te = int(t[k])
    if abs(raw / ref - 1) * 100 > DISLOC:
        return dict(st="disloc", te=te)
    e = raw * (1 + SLIP_IN / 100)
    end = np.searchsorted(t, te + HOLD, side="right")
    tt, pp = t[k + 1:end], p[k + 1:end]
    net = (pp / e - 1) * 100 - 2 * TAKER
    itp = np.flatnonzero(net >= TP); isl = np.flatnonzero(net <= SL)
    a = itp[0] if len(itp) else 10**12; b = isl[0] if len(isl) else 10**12
    if a == 10**12 and b == 10**12:
        last = net[-1] if len(net) else -2 * TAKER
        return dict(st="ok", te=te, xt=te + HOLD, why="time", pct=float(last - SLIP_OUT), pk=float(net.max()) if len(net) else np.nan)
    if a < b:
        return dict(st="ok", te=te, xt=int(tt[a]), why="tp", pct=TP, pk=float(net[:a + 1].max()))
    return dict(st="ok", te=te, xt=int(tt[b]), why="stop", pct=float(net[b] - SLIP_OUT), pk=float(net[:b + 1].max()))


def work(items):
    cache, out = {}, []
    for i, pair, t0 in items:
        try:
            r = one(pair, int(t0), cache)
        except Exception as ex:  # noqa
            r = dict(st=f"err:{ex}")
        out.append((i, r))
        if len(cache) > 8:
            cache.clear()
    return out


if __name__ == "__main__":
    E = pd.read_csv(sys.argv[1]).reset_index(drop=True)
    pairs = sorted(E.pair.unique()); g = {p: k % 8 for k, p in enumerate(pairs)}
    items = list(zip(E.index, E.pair, E.t0))
    chunks = [[it for it in items if g[it[1]] == k] for k in range(8)]
    with Pool(8) as P:
        res = dict(sum(P.map(work, chunks), []))
    R = pd.DataFrame([res[i] for i in E.index], index=E.index)
    out = pd.concat([E, R], axis=1); out.to_csv(sys.argv[2], index=False)
    print(out.st.value_counts().to_dict()); ok = out[out.st == "ok"]
    print(f"priced {len(ok)} · WR {(ok.pct > 0).mean() * 100:.1f}% · avg {ok.pct.mean():+.3f} · {ok.why.value_counts().to_dict()}")
