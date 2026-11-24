#!/usr/bin/env python3
"""TP_RUNNER_DIVE_2026-11-24 — deep dive on the HT (floor) and EXH (no floor) runner recommendations (research only).
Adds what the earlier passes skipped: FUNDING paid / received while the runner is held (settlements between the TP touch and the
runner exit, mark-scaled; a long pays a positive rate), a finer k × floor grid, concentration (top trades / leave-one-coin-out),
floor-exit slippage 0.30 %, 5m vs 15m ATR, a 50 / 50 split, the anatomy (share closing at the floor / stop vs running), hold time.
Runner Δ = runner net − TP net (pp of entry). TP net = the maker TP; runner exit = taker 0.05 + slip."""
import os, sys, importlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DP = importlib.import_module("TP_RUNNER_DEEP_2026-11-24"); T = DP.T; C = T.C
OUT = T.OUT; MIN, HR = 60_000, 3_600_000


def trail_ix(p, tp, d, w, floor_px):
    """→ (exit_px, bar index, why) — why = 'floor' when the floor was the binding stop, 'trail', or 'time'."""
    if d < 0:
        best = np.minimum.accumulate(np.concatenate([[tp], p["l"]]))[:-1]
        tr = best * (1 + w / 100); st = np.minimum(tr, floor_px) if floor_px is not None else tr
        hit = np.nonzero(p["h"] >= st)[0]
    else:
        best = np.maximum.accumulate(np.concatenate([[tp], p["h"]]))[:-1]
        tr = best * (1 - w / 100); st = np.maximum(tr, floor_px) if floor_px is not None else tr
        hit = np.nonzero(p["l"] <= st)[0]
    if not len(hit):
        return p["c"][-1], len(p["c"]) - 1, "time"
    i = hit[0]
    px = max(st[i], p["o"][i]) if d < 0 else min(st[i], p["o"][i])
    why = "floor" if (floor_px is not None and st[i] == floor_px) else "trail"
    return px, i, why


def fund_between(pair, t0, t1, E, px_ref, d):
    ft, fr = C.funding(pair)
    if not len(ft):
        return 0.0
    m = (ft > t0) & (ft <= t1)
    return float((-d * fr[m] * 100.0 * px_ref / E).sum())


def cohort(sleeve):
    B = pd.read_csv(os.path.join(OUT, "bt_tp_hits.csv")); B = B[B.sleeve == sleeve].copy(); B["blk"] = B.period
    return B


def eval_rows(B, d, specs, slip=0.10, atr_col="atr15", tp_net=None, stop_fn=None):
    """specs: name → (k, floor_net or None). Returns per-hit Δ incl. funding + anatomy columns."""
    pairs = sorted(set(B.pair))
    for p in pairs:
        months = sorted({pd.Timestamp(t, unit="ms").strftime("%Y-%m") for t in B[B.pair == p].t_touch} |
                        {(pd.Timestamp(t, unit="ms") + pd.Timedelta(days=1)).strftime("%Y-%m") for t in B[B.pair == p].t_touch})
        try:
            C.fetch_funding(p, months)
        except SystemExit:
            pass
    rows = []
    for r in B.itertuples():
        p = DP.path(r.pair, int(r.t_touch))
        atr = getattr(r, atr_col)
        if p is None or not np.isfinite(atr):
            continue
        E, tp = float(r.E), float(r.tp_px)
        tpn = tp_net if tp_net is not None else d * (tp / E - 1) * 100 - 0.07
        o = dict(pair=r.pair, t=int(r.t_touch), blk=r.blk)
        for nm, (k, fnet) in specs.items():
            fl = None if fnet is None else (E * (100 + fnet + 0.05) / (100 - 0.05) if d > 0 else E * (100 - fnet - 0.05) / (100 + 0.05))
            if fl is None and stop_fn is not None:
                fl = stop_fn(E, atr)
            px, i, why = trail_ix(p, tp, d, k * atr, fl)
            px = px * (1 - d * slip / 100)
            net = d * (px / E - 1) * 100 - 0.05 - 0.05
            fund = fund_between(r.pair, int(r.t_touch), int(r.t_touch) + (i + 1) * MIN, E, px, d)
            o[nm] = net + fund - tpn
            o[nm + "_fund"] = fund; o[nm + "_why"] = why; o[nm + "_min"] = i + 1
        rows.append(o)
    O = pd.DataFrame(rows)
    ts = pd.to_datetime(O.t, unit="ms"); O["half"] = ts.dt.year.astype(str) + "-H" + ((ts.dt.month > 6) + 1).astype(str)
    return O


def ci(O, c, rng, n=2000):
    x = O.assign(day=pd.to_datetime(O.t, unit="ms").dt.floor("D")).groupby("day")[c].sum()
    bs = [x.sample(len(x), replace=True, random_state=int(rng.integers(1e9))).sum() / len(O) for _ in range(n)]
    return np.quantile(bs, .025), np.quantile(bs, .975)


def report(title, O, names, rng):
    print(f"\n==== {title} — runner Δ vs TP (pp per TP hit, funding INCLUDED) · N per half {O.half.value_counts().sort_index().to_dict()}")
    t = O.groupby("half")[names].mean().round(2).T
    t["older"] = O[O.blk == "DESIGN"][names].mean().round(2); t["newer"] = O[O.blk == "HOLDOUT"][names].mean().round(2)
    t["all"] = O[names].mean().round(2); t["+halves"] = (O.groupby("half")[names].mean() > 0).sum()
    t["median"] = O[names].median().round(2); t["better"] = (O[names] > 0).mean().round(2)
    t["CI_all"] = [("%+.2f…%+.2f" % ci(O, c, rng)) for c in names]
    t["fund"] = [O[c + "_fund"].mean().round(2) for c in names]
    print(t.to_string())


def anatomy(O, c):
    g = O
    why = g[c + "_why"]
    out = {}
    for w in ("floor", "trail", "time"):
        m = why == w
        out[w] = f"{100*m.mean():.0f}% · avg Δ {g.loc[m, c].mean():+.2f} · median hold {g.loc[m, c + '_min'].median():.0f} min"
    s = g[c].sort_values(ascending=False)
    out["top3_share_of_total"] = round(s.iloc[:3].sum() / s.sum(), 2) if s.sum() > 0 else None
    out["total_wo_top3"] = round(s.iloc[3:].sum(), 1); out["total"] = round(s.sum(), 1)
    lo = min(((g[g.pair != p][c].sum(), p) for p in g.pair.unique()))
    out["leave_one_coin_out_min_total"] = round(lo[0], 1)
    out["funding_avg"] = round(g[c + "_fund"].mean(), 2); out["funding_worst"] = round(g[c + "_fund"].min(), 2)
    return out


def main():
    rng = np.random.default_rng(21)
    # ───────── HT (long): floors near the +6 TP ─────────
    H = cohort("FRENZY_WILLY_HT")
    specs = {f"{k}x|+{f:g}": (k, f) for k in (3, 4, 5, 6) for f in (4.0, 4.5, 5.0, 5.5)}
    specs["4x|stop−4"] = (4, -4.0)
    O = eval_rows(H, 1, specs, tp_net=6.0)
    report("HT backtest grid (slip 0.10)", O, list(specs), rng)
    print("\nHT anatomy 4x|+5:", anatomy(O, "4x|+5"))
    print("HT anatomy 4x|+4:", anatomy(O, "4x|+4"))
    O3 = eval_rows(H, 1, {"4x|+5": (4, 5.0), "4x|+4": (4, 4.0)}, slip=0.30, tp_net=6.0)
    print("\nHT floor slip 0.30:", O3[["4x|+5", "4x|+4"]].mean().round(2).to_dict(), "by half", O3.groupby("half")[["4x|+5", "4x|+4"]].mean().round(2).to_dict())
    H5 = H.copy(); H5["atr5"] = [DP.atr_tf(r.pair, int(r.t_touch) // MIN * MIN, 5) for r in H5.itertuples()]
    O5 = eval_rows(H5, 1, {"5m4x|+5": (4, 5.0), "5m6x|+5": (6, 5.0), "5m8x|+5": (8, 5.0)}, atr_col="atr5", tp_net=6.0)
    print("HT with 5m ATR:", O5.groupby("half")[["5m4x|+5", "5m6x|+5", "5m8x|+5"]].mean().round(2).to_dict(), "all", O5[["5m4x|+5", "5m6x|+5", "5m8x|+5"]].mean().round(2).to_dict())
    print("HT 50/50 split 4x|+5: Δ per hit", round(0.5 * O["4x|+5"].mean(), 2), "· median", round(0.5 * O["4x|+5"].median(), 2))
    O.to_csv(os.path.join(OUT, "dive_HT.csv"), index=False)
    # ───────── EXH (short): no floor, stop kept ─────────
    X = cohort("EXHAUSTION_SHORT")
    stop = lambda E, atr: E * (1 + min(1.5 * atr, 6) / 100)
    xs = {f"{k}x|stop": (k, None) for k in (3, 3.5, 4, 4.5, 5, 6)}
    OX = eval_rows(X, -1, xs, tp_net=None, stop_fn=stop)
    report("EXH backtest grid (no floor, stop kept, slip 0.10)", OX, list(xs), rng)
    print("\nEXH anatomy 4x|stop:", anatomy(OX, "4x|stop"))
    OX3 = eval_rows(X, -1, {"4x|stop": (4, None)}, slip=0.30, stop_fn=stop)
    print("EXH exit slip 0.30:", round(OX3["4x|stop"].mean(), 2), OX3.groupby("half")["4x|stop"].mean().round(2).to_dict())
    print("EXH 50/50 split 4x: Δ per hit", round(0.5 * OX["4x|stop"].mean(), 2))
    OX.to_csv(os.path.join(OUT, "dive_EXH.csv"), index=False)


if __name__ == "__main__":
    main()
