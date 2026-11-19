#!/usr/bin/env python3
"""ENTRY_WL — extra reads after the HOLDOUT read (descriptive): theory-model HOLDOUT AUC CI + terciles by half / BTC daily sign;
prior-fill-won (low coverage, its own read); EXH clean 2023-H1 block E on q5; REACH in stop units; per-half tables of the frozen
theory-led reads. Usage: venv/bin/python -I scripts/entry_wl_extra.py"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import entry_wl_lib as W  # noqa: E402
RNG = np.random.default_rng(5)
out = {}


def auc_ci(sc, y, days, nb=1000):
    ok = np.isfinite(y); sc, y, days = sc[ok], y[ok], days[ok]
    ud = np.unique(days); pos = {d: np.nonzero(days == d)[0] for d in ud}
    a = []
    nbk = int(np.ceil(len(ud) / 10))
    for _ in range(nb):
        st = RNG.integers(0, max(len(ud) - 10, 1), nbk)
        ii = np.concatenate([pos[ud[s + k]] for s in st for k in range(10) if s + k < len(ud)])
        a.append(W.auc(sc[ii], y[ii]))
    return np.nanpercentile(a, [2.5, 97.5])


for s in ("exh", "dist"):
    D = W.load(s); H = pd.read_pickle(os.path.join(W.D_OUT, f"{s}_holdout.pkl")); F = json.load(open(os.path.join(W.D_OUT, f"{s}_frozen.json")))
    sc = H["sc"]; ho = (D.block == "HOLDOUT").values; y = D.y_rg.values
    lo, hi = auc_ci(sc[ho], y[ho], D.day.values[ho])
    print(f"\n== {s}: theory model HOLDOUT AUC {H['a_ho']:.3f} (day-block 95 % CI {lo:.3f} … {hi:.3f})")
    q1, q2 = F["model"]["q_edges_all"]
    t = np.where(sc <= q1, "1 low (pred. give-back)", np.where(sc <= q2, "2 mid", "3 high (pred. runner)"))
    D["mt"] = t
    rows = []
    for h in W.HALVES:
        for tt in sorted(set(t)):
            m = (D.half.values == h) & (t == tt)
            if m.sum():
                rows.append(dict(half=h, tercile=tt, N=int(m.sum()), run_share=np.nanmean(y[m]), avg=D.net.values[m].mean()))
    T = pd.DataFrame(rows).pivot(index="half", columns="tercile", values=["run_share", "avg"])
    print(T.round(2).to_string())
    for sg in (-1, 1):
        m = ho & (np.sign(D.btc_d1_regime.values) == sg)
        g = D[m].groupby("mt").agg(N=("net", "size"), avg=("net", "mean"), rs=("y_rg", "mean"))
        print("HOLDOUT BTC daily", "bear" if sg < 0 else "bull", g.round(2).to_dict())
    # low-vs-high tercile Δ with day-block CI on HOLDOUT
    m = ho & (t != "2 mid")
    c = W.day_boot_diff(D.net.values[m], (t[m] == "3 high (pred. runner)"), D.day.values[m])
    print(f"HOLDOUT Δ avg net (high − low tercile) {D.net.values[ho & (t == '3 high (pred. runner)')].mean() - D.net.values[ho & (t == '1 low (pred. give-back)')].mean():+.2f}  CI {c[0]:+.2f} … {c[1]:+.2f}")
    kb = W.kel(D.net.values[ho & (t == "1 low (pred. give-back)")]); kh = W.kel(D.net.values[ho & (t == "3 high (pred. runner)")])
    print(f"HOLDOUT Kelly/trade at Wilson floor: low {kb[0]:.3f} · high {kh[0]:.3f}")
    out[f"{s}_model"] = dict(auc=H["a_ho"], lo=lo, hi=hi, half=T)
    # prior fill in the same episode won / lost (low coverage)
    print("prior fill in the episode (closed before entry):")
    for blk in ("DESIGN", "HOLDOUT"):
        for v_, nm in ((1.0, "prior WON"), (0.0, "prior LOST"), (np.nan, "no prior fill")):
            m = (D.block.values == blk) & ((D.st_prior_fill_won.values == v_) if np.isfinite(v_) else ~np.isfinite(D.st_prior_fill_won.values))
            print(f"  {blk} {nm:14s} N {m.sum():4d} run_share {np.nanmean(y[m]):.3f} avg {D.net.values[m].mean():+.2f} WR {(D.net.values[m] > 0).mean() * 100:.0f}")
    # REACH in stop units (is the REACH / ATR link mechanical?)
    rr = D.reach.values / D.sl_pct.values
    dm = ~ho
    for v in ("atr15_pct", "atr1h_pct", "run_pct", "dist_base_pct", "q5", "vol24"):
        x = pd.to_numeric(D[v], errors="coerce").values
        ok = dm & np.isfinite(x)
        print(f"  DESIGN corr({v}, reach %) {np.corrcoef(x[ok], D.reach.values[ok])[0, 1]:+.3f} · corr({v}, reach ÷ stop) {np.corrcoef(x[ok], rr[ok])[0, 1]:+.3f} · corr({v}, RUNNER|R+GB) {np.corrcoef(x[ok & np.isfinite(y)], y[ok & np.isfinite(y)])[0, 1]:+.3f}")

# per-half tables of the theory-led best read per sleeve
for s, var, edge, side in (("exh", "q5", 1.157e6, "le"), ("dist", "vol24", 4.198e7, "le")):
    D = W.load(s); x = pd.to_numeric(D[var], errors="coerce").values; bad = x <= edge
    rows = []
    for h in W.HALVES:
        m = D.half.values == h
        rows.append(dict(half=h, N_bad=int((m & bad).sum()), N_kept=int((m & ~bad).sum()), rs_bad=np.nanmean(D.y_rg.values[m & bad]),
                         rs_kept=np.nanmean(D.y_rg.values[m & ~bad]), avg_bad=D.net.values[m & bad].mean(), avg_kept=D.net.values[m & ~bad].mean(),
                         gb_bad=(D.label.values[m & bad] == "GAVE_BACK").mean() * 100, gb_kept=(D.label.values[m & ~bad] == "GAVE_BACK").mean() * 100,
                         share_bad=(m & bad).sum() / m.sum() * 100))
    T = pd.DataFrame(rows); T["delta"] = T.avg_kept - T.avg_bad
    print(f"\n{s} {var} ≤ {edge:.4g} per half:"); print(T.round(2).to_string(index=False))
    T.to_csv(os.path.join(W.D_OUT, f"{s}_{var}_halves.csv"), index=False)

# EXH clean block E (2023-H1): q5 split with the DESIGN edge
import exh_exit2_eval as XE
X, Z = XE.load("E"); NET = XE.nets(X, Z); ev = pd.read_pickle(os.path.join(XE.D, "eval.pkl")); keep = np.asarray(ev["keepE"])
v = NET[keep, 0]; q = X.q5.values[keep]; k = Z["k"][keep, 0]; reach = Z["reach"][keep, 0]
lab = np.where((k == 0) | (reach >= 8), "R", np.where((k == 1) & (reach >= 2), "G", "o"))
for nm, m in (("q5 ≤ 1.157e6 (thin)", q <= 1.157e6), ("rest", q > 1.157e6)):
    rs = (lab[m] == "R").sum() / max(((lab[m] == "R") | (lab[m] == "G")).sum(), 1)
    print(f"E 2023-H1 {nm}: N {m.sum()} avg {v[m].mean():+.2f} run_share {rs:.3f}")
