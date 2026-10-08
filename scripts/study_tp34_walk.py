#!/usr/bin/env python3
"""🎯 Oct-8 study (+3 vs +4 tick check) — walk every cohort fill on aggTrades ticks with every exit (study_tp34_common).

Rulers:
  live   entry = first print ≥ signal close + 8 s, E = print × 1.0010 (the live ruler: 8 s + 0.10 % slip), bot fees, crossing-print fills
  noslip same timing, E = the print (sensitivity)
  old    (cohort b only, parity with frenzy_exit_ticks_botexact.py): entry = close of the first 1-min bar after the signal (FU.path cc[0]),
         prints from signal + 60 s, flat 0.09 fees
  asfilled (cohort d only): the live fill's own opened_at / entry_price, era-correct exit, for the walker-vs-live match rate
Outputs scratch tp34/walk_<cohort>.pkl.  Usage: venv/bin/python scripts/study_tp34_walk.py"""
import glob, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_tp34_common as C                                   # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
OUT = os.path.join(SCR, "tp34")


def flat(r, pre):
    d = {}
    for x in C.EXITS:
        v = r[x]; d[f"{pre}{x}"] = v[0]; d[f"{pre}{x}_how"] = v[1]; d[f"{pre}{x}_xms"] = v[2]
    d[f"{pre}pk_before_stop"] = r["peak_before_stop"]; d[f"{pre}pk12h"] = r["peak_12h"]; d[f"{pre}entry_ms"] = r["entry_ms"]
    d[f"{pre}E"] = r["E"]; d[f"{pre}complete"] = r["cap_complete"]
    return d


def job(items):
    out = []
    for key, pair, sig in items:
        d = dict(key=key)
        for pre, slip in (("", 0.10), ("ns_", 0.0)):
            r = C.walk_entry(pair, sig, 8_000, slip)
            if r is None:
                d["st"] = "no_ticks"; break
            d.update(flat(r, pre)); d["st"] = "ok"
        out.append(d)
    return out


_FU = None


def old_job(items):
    """parity with scripts/frenzy_exit_ticks_botexact.py (entry = 1m close after the signal, 0.09 flat fees)."""
    global _FU
    if _FU is None:
        _a, sys.argv = sys.argv, ["x"]
        import frenzy_scalp_followup as FU
        sys.argv = _a; _FU = (FU, FU.episode_index())
    FU, idx = _FU
    out = []
    for key, pair, sig in items:
        pth = FU.path(idx, pair, int(sig))
        if pth is None:
            out.append(dict(key=key, old_st="no_path")); continue
        tt, hh, ll, cc = pth; e = float(cc[0]); te = int(tt[0]) + 60_000
        x = C.ticks(pair, te, te + C.HOLD)
        if x is None or len(x[0]) < 10:
            out.append(dict(key=key, old_st="no_ticks")); continue
        t, p = x
        net = (p / e - 1) * 100 - 0.09
        r = C.walk_all(t, net, ["fix3", "fix4", "fix6", "lock", "trail5"])
        out.append(dict(key=key, old_st="ok", old_t1m=int(tt[0]), **{f"old_{k}": r[k][0] for k in ("fix3", "fix4", "fix6", "lock", "trail5")}))
    return out


def run(cohort, fn=job):
    D = pd.read_pickle(os.path.join(OUT, f"cohort_{cohort}.pkl"))
    U = D[["pair", "sig"]].drop_duplicates().reset_index(drop=True)
    U["key"] = U.pair + "|" + U.sig.astype(str)
    pairs = sorted(U.pair.unique()); g = {p: k % 8 for k, p in enumerate(pairs)}
    chunks = [[(r.key, r.pair, int(r.sig)) for r in U.itertuples() if g[r.pair] == k] for k in range(8)]
    with Pool(8) as P:
        res = sum(P.map(fn, chunks), [])
    return pd.DataFrame(res)


# ── cohort d: live fills, as filled ──
DEPLOY = {  # commit time + 10 min (UTC) of the FRENZY exit eras (trading_config.json history)
    "tp4": pd.Timestamp("2026-10-04 13:48:49") + pd.Timedelta(minutes=10),
    "tp3": pd.Timestamp("2026-10-04 19:23:15") + pd.Timedelta(minutes=10),
    "lock": pd.Timestamp("2026-10-05 15:49:16") + pd.Timedelta(minutes=10),
    "tp3b": pd.Timestamp("2026-10-08 13:37:25") + pd.Timedelta(minutes=10),
}


def era(o_ms):
    t = pd.Timestamp(o_ms, unit="ms")
    if t < DEPLOY["tp4"]:
        return "trail5"
    if t < DEPLOY["tp3"]:
        return "fix4"
    if t < DEPLOY["lock"]:
        return "fix3"
    if t < DEPLOY["tp3b"]:
        return "lock"
    return "fix3"


def ticks_d(pair, a, b):
    """cache days, plus scratch API windows for today's (unpublished) day; None when a needed span is missing."""
    ts, ps = [], []
    for ds in C.days_for(a, b):
        x = C.tick_day(pair, ds)
        if x is None:
            fs = glob.glob(os.path.join(OUT, "api_ticks", f"{pair}_*.npz"))
            if not fs:
                return None
            z = np.load(fs[0]); x = (z["t"], z["p"])
        ts.append(x[0]); ps.append(x[1])
    t = np.concatenate(ts); p = np.concatenate(ps); o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    m = (t >= a) & (t <= b)
    return t[m], p[m]


def run_d():
    D = pd.read_pickle(os.path.join(OUT, "cohort_d.pkl"))
    rows = []
    for r in D.itertuples():
        d = dict(pair=r.pair, sig=r.sig, live_open=r.live_open, era=era(r.live_open))
        x = ticks_d(r.pair, int(r.live_open), int(r.live_open) + C.HOLD)
        if x is None:   # today's archive not published: the as-filled validation only needs the path up to the live close + 15 min
            x = ticks_d(r.pair, int(r.live_open), int(r.live_close) + 15 * C.MIN)
        if x is None or not len(x[0]):
            d["st"] = "no_ticks"; rows.append(d); continue
        t, p = x
        complete = t[-1] >= r.live_open + C.HOLD - 2 * C.MIN
        net = C.net_path(p, r.live_E)
        w = C.walk_all(t, net)
        for k in C.EXITS:
            d[f"af_{k}"], d[f"af_{k}_how"], d[f"af_{k}_xms"] = w[k]
            if w[k][1] == "CAP" and not complete:
                d[f"af_{k}"] = np.nan; d[f"af_{k}_how"] = "OPEN(no ticks yet)"
        d["af_complete"] = complete; d["st"] = "ok"
        # live ruler from the signal close (only where the window is fully on ticks)
        y = ticks_d(r.pair, int(r.sig) + 8_000, int(r.sig) + 8_000 + C.HOLD)
        if y is not None and len(y[0]):
            E = float(y[1][0]) * 1.001; ty = y[0][y[0] <= y[0][0] + C.HOLD]; py = y[1][: len(ty)]
            w2 = C.walk_all(ty, C.net_path(py, E))
            for k in C.EXITS:
                d[k] = w2[k][0] if not (w2[k][1] == "CAP" and not complete) else np.nan
                d[f"{k}_how"] = w2[k][1]
            d["ruler_entry_lag_s"] = (int(y[0][0]) - int(r.sig)) / 1000; d["ruler_E_vs_live"] = (E / r.live_E - 1) * 100
        rows.append(d)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for c in ("a", "b", "c"):
        W = run(c)
        if c == "b":
            W = W.merge(run(c, old_job), on="key", how="left")
        W.to_pickle(os.path.join(OUT, f"walk_{c}.pkl")); print(c, len(W), W.st.value_counts().to_dict(), flush=True)
    Wd = run_d(); Wd.to_pickle(os.path.join(OUT, "walk_d.pkl")); print("d", len(Wd), Wd.st.value_counts().to_dict())
