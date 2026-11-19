#!/usr/bin/env python3
"""ENTRY_WL (PREREG reports/ENTRY_WL_2026-11-18_PREREG.md) — BUILD the labelled trade rows for EXHAUSTION_SHORT and DISTRIBUTION_SHORT.
Research only, BACKTEST. READ-ONLY on the bot.

EXH : EXH_EXIT2 rows / 1m sim (REF = today's exit) + the UNIFIED_SHORT pool row at the same (pair, entry_ms) + the baseline stamps
      + DIST_WL's lag-safe extras (dist_wl_build.one_pair on the k5m series) + the 5m anchored path (2 h before … 24 h after).
DIST: DIST_EXIT 1m sim (REF_B = Exit B) on the DL 586 cohort (one position per pair under REF) + every DIST_WL feature + path.
Both: derived entry-time state columns (prior signals / fills in the episode, open same-sleeve positions, fills in the last 24 h,
      stops in the last 24 h — all from fills whose exit is ≤ the entry, no lookahead).
Labels (REF exit, 1m, own stop): RUNNER = TP or reach ≥ 8 · GAVE_BACK = stop ∧ 2 ≤ reach < 8 · NEVER_WORKED = stop ∧ reach < 1 · OTHER.
Output: reports/ENTRY_WL_2026-11-18_data/{exh,dist}.pkl + {exh,dist}_path.npy
Usage: venv/bin/python -I scripts/entry_wl_build.py [--procs 6]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_wl_build as WB      # noqa: E402  (patches k5m / funding sources, one_pair)
import exh_exit2_eval as XE     # noqa: E402
import dist_exit_eval as DE     # noqa: E402
R = os.path.abspath(os.path.join(HERE, "..")); RP = os.path.join(R, "reports")
OUT = os.path.join(RP, "ENTRY_WL_2026-11-18_data"); os.makedirs(OUT, exist_ok=True)
DAY, H1 = 86_400_000, 3_600_000
HO_MS = int(pd.Timestamp("2025-07-01").value // 10**6)
EXTRA = ["close_pos15", "lowwick15", "range15_pct", "last5_share", "oi_chg15m", "oi_chg30m", "top_pos_chg1h", "top_acct_chg4h",
         "top_pos_chg4h_rel", "top_acct_chg4h_rel", "pos_minus_acct_rel", "glob_chg4h"]


def labels(k, reach, net):
    lab = np.full(len(k), "OTHER", dtype=object)
    stop = k == 1
    lab[(k == 0) | (reach >= 8)] = "RUNNER"
    lab[stop & (reach >= 2) & (reach < 8)] = "GAVE_BACK"
    lab[stop & (reach < 1)] = "NEVER_WORKED"
    return lab


def state_cols(D, sleeve_x):
    """entry-time portfolio / episode state from the sleeve's own fills (exit ≤ entry only)."""
    T = D.entry_ms.values.astype(np.int64); X = sleeve_x.astype(np.int64); net = D.net.values; k = D.k.values
    o = np.argsort(T); n = len(D)
    open_n = np.zeros(n); f24 = np.zeros(n); st24 = np.zeros(n); w24 = np.full(n, np.nan)
    xs = np.sort(X); xo = np.argsort(X)
    for i in range(n):
        t = T[i]
        open_n[i] = ((T < t) & (X > t)).sum()
        f24[i] = ((T < t) & (T >= t - 24 * H1)).sum()
        done = (X <= t) & (X >= t - 24 * H1)
        st24[i] = (done & (k == 1)).sum()
        if done.any():
            w24[i] = (net[done] > 0).mean()
    D["st_open_same"] = open_n; D["st_fills24h"] = f24; D["st_stops24h"] = st24; D["st_wr24h_closed"] = w24
    # prior fill of the same pair in the same episode, closed before entry
    pf = np.full(n, np.nan); pfr = np.full(n, np.nan); npf = np.zeros(n)
    g = D.groupby("ep").indices
    for ep, ii in g.items():
        ii = ii[np.argsort(T[ii])]
        for a, i in enumerate(ii):
            prev = [j for j in ii[:a] if X[j] <= T[i]]
            npf[i] = len(prev)
            if prev:
                j = prev[-1]; pf[i] = float(net[j] > 0); pfr[i] = D.reach.values[j]
    D["st_prior_fills_ep"] = npf; D["st_prior_fill_won"] = pf; D["st_prior_fill_reach"] = pfr
    return D


def extras(S, procs, xcol, rcol):
    S = S.reset_index(drop=True)
    tmp = S[["entry_ms", "atr15_pct"]].copy(); tmp["LIVE_x"] = S[xcol].values; tmp["LIVE_r"] = S[rcol].values
    jobs = [(p, tmp.loc[S.pair == p]) for p in S.pair.unique()]
    P = np.full((len(S), WB.PRE + WB.HOLD + 1), np.nan, np.float32)
    cols = {c: np.full(len(S), np.nan) for c in EXTRA}
    with mp.get_context("fork").Pool(procs) as pool:
        for k, (p, ix, out, path) in enumerate(pool.imap_unordered(WB.one_pair, jobs, chunksize=2)):
            for c in EXTRA:
                cols[c][ix] = out[c]
            P[ix] = path
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), k, len(jobs), flush=True)
    return cols, P


def build_exh(procs):
    X, Z = XE.load("EXH")
    NET = XE.nets(X, Z); N20 = XE.nets(X, Z, flat=0.20)
    j = XE.REF
    D = pd.DataFrame(dict(pair=X.pair.astype(str).values, entry_ms=X.entry_ms.values.astype(np.int64), xi=X.xi.values,
                          net=NET[:, j], net20=N20[:, j], k=Z["k"][:, j], x=Z["xm"][:, j], reach=Z["reach"][:, j],
                          worked=Z["worked"][:, j], best=Z["best"][:, j], slip=X.slip.values, q5_row=X.q5.values))
    D["row"] = np.arange(len(D))
    B = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    st = B[["pair", "entry_ms", "oi_vol", "ls_top_acc", "ls_top_pos", "ls_glob", "taker_1h", "f_last", "ssc", "vd", "atrp"]].rename(
        columns={c: "s_" + c for c in ["oi_vol", "ls_top_acc", "ls_top_pos", "ls_glob", "taker_1h", "f_last", "ssc", "vd", "atrp"]})
    D = D.merge(st, on=["pair", "entry_ms"], how="left")
    # exh_pool_rows.pkl = the EXHAUSTION study pool rows (a scratch artifact of the EXH_EXIT2 / EXH study pool build, not kept in git;
    # set SCR to its folder). The committed outputs in reports/ENTRY_WL_2026-11-18* are the reproducible record.
    Pp = pd.read_pickle(os.path.join(os.environ.get("SCR", "/tmp"), "exh_pool_rows.pkl"))
    Pp = Pp.drop(columns=[c for c in Pp.columns if c[:3] in ("T3A", "T5A", "T10", "T3F", "T5F", "LIV") or c in ("block", "day")])
    Pp["pair"] = Pp.pair.astype(str); Pp["ep"] = Pp.ep.astype(str)
    D = D.merge(Pp, on=["pair", "entry_ms"], how="left", suffixes=("", "_pool"))
    D["in_pool"] = D.ep.notna()
    # prior EXH-flag candidates in the same episode (from the pool, entry-time)
    X0 = pd.read_pickle(os.path.join(R, "reports", "UNIFIED_SHORT_data", "pool.pkl"), compression=None)[["pair", "entry_ms", "ep", "EXH", "g_VSW15"]]
    X0 = X0[X0.EXH | X0.g_VSW15]; X0["ep"] = X0.ep.astype(str)
    gE = X0[X0.EXH].groupby("ep").entry_ms.apply(lambda s: np.sort(s.values)).to_dict()
    gV = X0[X0.g_VSW15].groupby("ep").entry_ms.apply(lambda s: np.sort(s.values)).to_dict()
    del X0
    D["n_prior_exh_sig"] = [np.searchsorted(gE.get(e, np.array([])), t) if isinstance(e, str) else np.nan for e, t in zip(D.ep, D.entry_ms)]
    D["n_prior_vsw_red"] = [np.searchsorted(gV.get(e, np.array([])), t) if isinstance(e, str) else np.nan for e, t in zip(D.ep, D.entry_ms)]
    D["ep"] = D.ep.fillna(D.pair + "|na")
    D["atr15_pct"] = D.atr15_pct.fillna(D.s_atrp)
    D = state_cols(D, D.x.values)
    cols, P = extras(D, procs, "x", "net")
    for c, v in cols.items():
        D[c] = v
    D["label"] = labels(D.k.values, D.reach.values, D.net.values)
    D["block"] = np.where(D.entry_ms.values >= HO_MS, "HOLDOUT", "DESIGN")
    D["half"] = X.half.values if "half" in X else ""
    D["day"] = D.entry_ms.values // DAY
    D["sl_pct"] = np.minimum(1.5 * D.atr15_pct.values, 6.0)
    return D, P


def build_dist(procs):
    S, Z, V, slip, prim, pair, T = DE.build()
    j = 0                                                          # REF_B
    m = DE.one_pos(pair, T, Z["xm"][:, j], prim & np.isfinite(V[:, j]))
    N20 = DE.net_mat(Z, slip, flat=0.20)
    D = S[m].copy().reset_index(drop=True)
    ii = np.nonzero(m)[0]
    D["zi"] = ii
    D["net"] = V[ii, j]; D["net20"] = N20[ii, j]; D["k"] = Z["k"][ii, j]; D["x"] = Z["xm"][ii, j]
    D["reach_1m"] = Z["reach"][ii, j]; D["worked_1m"] = Z["worked"][ii, j]; D["best"] = Z["best"][ii, j]
    D = D.rename(columns={"reach": "reach_5m", "label": "label_5m"})
    D["reach"] = D.reach_1m; D["worked"] = D.worked_1m; D["slip"] = slip[ii]
    D["pair"] = D.pair.astype(str); D["ep"] = D.ep.astype(str)
    D = state_cols(D, D.x.values)
    D["label"] = labels(D.k.values, D.reach.values, D.net.values)
    D["block"] = np.where(D.entry_ms.values >= HO_MS, "HOLDOUT", "DESIGN")
    D["day"] = D.entry_ms.values // DAY
    D["sl_pct"] = np.minimum(1.5 * D.atr15_pct.values, 6.0)
    Pall = np.load(os.path.join(RP, "DIST_WL_2026-11-18_data", "paths.npy"))
    P = Pall[D.orig_idx.values] if False else None
    # DIST_WL paths.npy is aligned to rows.pkl order (S); take the same rows
    P = Pall[ii]
    return D, P


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--only", default="exh,dist")
    A = ap.parse_args()
    if "dist" in A.only:
        D, P = build_dist(A.procs)
        print("DIST", D.shape, D.label.value_counts().to_dict(), "avg", D.net.mean().round(3), flush=True)
        D.to_pickle(os.path.join(OUT, "dist.pkl")); np.save(os.path.join(OUT, "dist_path.npy"), P)
    if "exh" in A.only:
        D, P = build_exh(A.procs)
        print("EXH", D.shape, "in_pool", int(D.in_pool.sum()), D.label.value_counts().to_dict(), "avg", D.net.mean().round(3), flush=True)
        D.to_pickle(os.path.join(OUT, "exh.pkl")); np.save(os.path.join(OUT, "exh_path.npy"), P)
