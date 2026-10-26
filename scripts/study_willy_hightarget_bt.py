#!/usr/bin/env python3
"""WILLY high target — block ③ backtest: engine-replay WILLY fills (yr6a 2025 + yr6 2026, seeds 1-3, trimmed) re-walked on 1m paths
(WILLY late-join 1m cache) under today's exit and each variant. Entry = the replay's own fill price / time.
Output: reports/WILLY_HIGHTARGET_2026-10-23_bt.csv
Usage: venv/bin/python scripts/study_willy_hightarget_bt.py [--procs 8]
"""
import argparse, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_hightarget_common as C  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402

K1 = os.path.join(C.R, "reports", "WILLY_LATE_JOIN_2026-10-22_data", "k1m")
OUT = os.path.join(C.R, "reports", "WILLY_HIGHTARGET_2026-10-23_bt.csv")
MIN = C.MIN
LAG_MAX_MIN = 61.0                 # today's entry A: red close ≤ 60 min after the spike close (+ fill latency)


def k1(pair, days):
    fr = []
    for d in days:
        f = os.path.join(K1, pair, f"{d}.npz")
        if os.path.exists(f):
            with np.load(f) as z:
                fr.append(z["a"])
    if not fr:
        return None
    a = np.concatenate(fr)
    a = a[np.argsort(a[:, 0], kind="stable")]
    _, i = np.unique(a[:, 0], return_index=True)
    return a[i]


def path(a, te, E, horizon):
    lim = te + (horizon + 1) * MIN
    first = -(-int(te) // MIN) * MIN
    b = a[(a[:, 0] >= first) & (a[:, 0] + MIN <= lim)]
    if len(b):
        o = b[:, 0].astype("int64")
        if o[0] != first:
            b = b[:0]
        else:
            g = np.flatnonzero(np.diff(o) != MIN)
            if len(g):
                b = b[:g[0] + 1]
    o = b[:, 0].astype("int64")
    net = lambda x: WT.net_pct(x, E)  # noqa: E731
    return dict(t=o + MIN, hi=net(b[:, 2]), lo=net(b[:, 3]), cl=net(b[:, 4]), op=net(b[:, 1]), bar=np.ones(len(b), bool),
                end_ms=int(o[-1] + MIN) if len(b) else int(te), n_tick=0, n_bar=int(len(b)), te=int(te))


def work(rows):
    out = []
    for r in rows:
        te, sp = r["te"], r["sp"]
        red = (te // C.BAR) * C.BAR
        days = sorted({pd.Timestamp(x, unit="ms").strftime("%Y-%m-%d") for x in (sp - C.BAR, te, te + (C.HORIZON + 1) * MIN)})
        a = k1(r["pair"], days)
        d = dict(r)
        if a is None:
            d["st"] = "NODATA"; out.append(d); continue
        s = a[(a[:, 0] >= sp - C.BAR) & (a[:, 0] < red)]
        if len(s) != (red - (sp - C.BAR)) // MIN or len(s) == 0:
            d["st"] = "NOHIGH"; out.append(d); continue
        H = float(s[:, 2].max()); E = r["E"]
        P = path(a, te, E, C.HORIZON)
        d.update(st="ok", H=H, dist=(H / E - 1) * 100, n_bar=P["n_bar"])
        for v, (p, why, tg) in C.walk_all(P, H, E).items():
            d[f"{v}_pct"] = p; d[f"{v}_why"] = why; d[f"{v}_tgt"] = tg
        out.append(d)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    import yr5_fills_trimmed as YT
    fr = []
    for p in ("yr6a", "yr6"):
        F = YT.load(prefix=p); w = F[F.sleeve == "FRENZY_WILLY"].copy(); w["prefix"] = p; fr.append(w)
    W = pd.concat(fr, ignore_index=True)
    W["spt"] = pd.to_datetime(W.entry_frenzy_spike_at)
    W["lag"] = (W.t - W.spt).dt.total_seconds() / 60
    n0 = len(W)
    W = W[(W.lag >= 0) & (W.lag <= LAG_MAX_MIN) & (W.entry_frenzy_willy_trigger.astype(str) == "A")]
    print(f"replay WILLY fills {n0} → entry-A within 60 min of the spike {len(W)}")
    rows = [dict(prefix=p, seed=int(s), pair=pa, te=int(pd.Timestamp(t).value // 1_000_000), sp=int(pd.Timestamp(spt).value // 1_000_000),
                 E=float(e), replay_pct=float(pc), replay_why=str(cr), run_pct=rp, wait_bars=wb, lag=lg)
            for p, s, pa, t, spt, e, pc, cr, rp, wb, lg in zip(W.prefix, W.seed, W.pair, W.t, W.spt, W.entry_price, W.pnl_percentage,
                                                                W.close_reason, W.entry_frenzy_run_pct, W.entry_frenzy_willy_wait_bars, W.lag)]
    pairs = sorted({r["pair"] for r in rows}); g = {p: i % (A.procs * 4) for i, p in enumerate(pairs)}
    chunks = [[r for r in rows if g[r["pair"]] == n] for n in range(A.procs * 4)]
    with mp.Pool(A.procs) as P:
        res = sum(P.map(work, chunks), [])
    D = pd.DataFrame(res).sort_values(["seed", "te"])
    D["day"] = pd.to_datetime(D.te, unit="ms").dt.strftime("%Y-%m-%d")
    t = pd.to_datetime(D.te, unit="ms")
    D["period"] = np.where(t < "2026-01-01", "2025", np.where(t < "2026-07-01", "2026-H1", "2026-H2"))
    D["half"] = np.where(t < "2025-07-01", "2025-H1", np.where(t < "2026-01-01", "2025-H2", np.where(t < "2026-07-01", "2026-H1", "2026-H2")))
    D.to_csv(OUT, index=False)
    print(D.st.value_counts().to_dict())
    print(D[D.st == "ok"].TODAY_why.value_counts().to_dict())
