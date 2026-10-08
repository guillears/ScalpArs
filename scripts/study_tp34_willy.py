#!/usr/bin/env python3
"""🎲 Oct-8 (operator: every halves-table strategy priced on ticks) — FRENZY_WILLY under its CURRENT design on aggTrades, bot accounting.

Same triggers / turnover rule / red-bar entry / dislocation guard / one-at-a-time sequencer as scripts/study_yr5_halves_willy.py (imported).
Differences = the bot's accounting instead of that script's conventions:
  net at every print = (p/E − 1)·100 − 0.045 − 0.045·p/E (E = first print ≥ red close + 8 s × 1.00035)
  TP: closes AT the first print with net ≥ +1.0 (that script books exactly +1.0)
  cap: 120 min after entry → the last print's net (that script subtracts a further 0.05 slip)
  variants: NOSTOP (live paper design) · BACKSTOP −2.2 (the exchange backstop as a stop on net; first print ≤ −2.2; a same-ms TP/stop tie → stop)
  CONV = that script's conventions re-run here (parity check vs its CSV).
Each variant is sequenced on its own (exit times differ → slot occupancy differs).
Usage: venv/bin/python scripts/study_tp34_willy.py"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_yr5_halves_willy as WY                              # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = os.path.join(SCR, "tp34")
VARS = ("CONV", "NOSTOP", "BACKSTOP")
BSTOP = 2.2


def price(t, p, k, te, e):
    end = np.searchsorted(t, te + WY.HOLD, side="right")
    tt, pp = t[k + 1:end], p[k + 1:end]
    r = pp / e
    net = (r - 1) * 100 - WY.TAKER - WY.TAKER * r
    netc = (pp / e - 1) * 100 - 2 * WY.TAKER                    # the old script's flat-fee convention
    res = {}
    if not len(net):
        for v in VARS:
            res[v] = dict(xt=te + WY.HOLD, why="time_cap", pct=-2 * WY.TAKER, worst=-2 * WY.TAKER)
        return res
    itp = np.flatnonzero(netc >= WY.TP)
    if len(itp):
        a = itp[0]; res["CONV"] = dict(xt=int(tt[a]), why="tp", pct=WY.TP, worst=float(netc[:a + 1].min()))
    else:
        res["CONV"] = dict(xt=te + WY.HOLD, why="time_cap", pct=float(netc[-1] - WY.SLIP_OUT), worst=float(netc.min()))
    itp = np.flatnonzero(net >= WY.TP)
    i_t = int(itp[0]) if len(itp) else None
    if i_t is not None:
        res["NOSTOP"] = dict(xt=int(tt[i_t]), why="tp", pct=float(net[i_t]), worst=float(net[:i_t + 1].min()))
    else:
        res["NOSTOP"] = dict(xt=te + WY.HOLD, why="time_cap", pct=float(net[-1]), worst=float(net.min()))
    ist = np.flatnonzero(net <= -BSTOP)
    i_s = int(ist[0]) if len(ist) else None
    if i_s is not None and (i_t is None or i_s < i_t or tt[i_s] == tt[i_t]):
        res["BACKSTOP"] = dict(xt=int(tt[i_s]), why="stop", pct=float(net[i_s]), worst=float(net[:i_s + 1].min()))
    else:
        res["BACKSTOP"] = dict(res["NOSTOP"])
    return res


def walk(pair, t0, cache):
    rb = WY.red_bars(pair, t0)
    if rb is None:
        return [dict(st="no_klines")]
    if not rb:
        return [dict(st="red_expired")]
    t, p = WY.ticks(pair, int(t0) - 120_000, rb[-1] + WY.DELAY + WY.HOLD + 60_000, cache)
    if t is None:
        return [dict(st="no_ticks")]
    out = []
    for c in rb:
        pre = t < c
        if not pre.any():
            out.append(dict(st="noprint_pre", red_close=c)); continue
        ref = p[pre][-1]; k = np.searchsorted(t, c + WY.DELAY)
        if k >= len(t):
            out.append(dict(st="noprint_post", red_close=c)); continue
        raw = p[k]; te = int(t[k])
        if abs(raw / ref - 1) * 100 > WY.DISLOC:
            out.append(dict(st="disloc", red_close=c, te=te)); continue
        e = raw * (1 + WY.SLIP_IN / 100)
        out.append(dict(st="ok", red_close=c, te=te, E=e, v=price(t, p, k, te, e)))
    return out


def work(items):
    cache, out = {}, []
    for i, pair, t0 in items:
        try:
            r = walk(pair, int(t0), cache)
        except Exception as ex:  # noqa
            r = [dict(st=f"err:{ex}")]
        out.append((i, r))
        if len(cache) > 8:
            cache.clear()
    return out


if __name__ == "__main__":
    D = pd.read_csv(WY.SRC)
    E = D[D.cohort.isin(["willyA", "willyB"])][["cohort", "pair", "t0", "day", "R"]].reset_index(drop=True)
    E["trigger"] = E.cohort.str[-1]
    E["turnover_ok"] = E.R < WY.RMAX
    G = E[E.turnover_ok].copy()
    pairs = sorted(G.pair.unique()); g = {p: k % 8 for k, p in enumerate(pairs)}
    items = list(zip(G.index, G.pair, G.t0))
    with Pool(8) as P:
        cands = dict(sum(P.map(work, [[it for it in items if g[it[1]] == k] for k in range(8)]), []))
    allr = []
    for v in VARS:
        cv = {i: [({**{k: x for k, x in c.items() if k != "v"}, **c["v"][v]} if c.get("st") == "ok" else c) for c in cs] for i, cs in cands.items()}
        S = WY.sequence(G, cv).assign(variant=v)
        allr.append(S)
    R = pd.concat(allr, ignore_index=True)
    R.to_pickle(os.path.join(OUT, "willy_ticks.pkl"))
    for v, x in R[R.st == "ok"].groupby("variant"):
        print(v, len(x), f"WR {(x.pct > 0).mean() * 100:.1f}% avg {x.pct.mean():+.3f} worst {x.pct.min():+.2f} dip≤−4.5 {(x.worst <= -4.5).mean() * 100:.1f}%",
              x.groupby("trigger").pct.agg(["count", "mean"]).round(3).to_dict())
