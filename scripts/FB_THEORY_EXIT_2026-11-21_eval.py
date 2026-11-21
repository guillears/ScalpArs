#!/usr/bin/env python3
"""FB_THEORY_EXIT_2026-11-21 — EVAL (PREREG reports/FB_THEORY_EXIT_2026-11-21_PREREG.md). Research only.
Reads bt_cells.csv / bt_anchors.csv / real_fills.csv / real_cells.csv → every cell on every block, the DESIGN family picks, the master
gates, the HOLDOUT gates, LOO, one-size DCR / risk. Output: reports/FB_THEORY_EXIT_2026-11-21_tables.md + _data/eval.pkl
Usage: venv/bin/python -I scripts/FB_THEORY_EXIT_2026-11-21_eval.py"""
import importlib.util, os, pickle, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_data")
SPLIT = pd.Timestamp("2025-07-01").value // 10**6
USD = 30.5           # $ per 1 % at one size: 0.61 × $5,000 = $3,050 per fill
FAMS = ("A", "C", "B", "T", "F")
MATCH = {"C_PK0": "K_MPK", "C_PK1": "K_MPK", "C_TIGHT": "K_S8", "C_VWAP": "K_MVWAP", "C_CFHI": "K_MCFHI"}


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


K = _imp("FBT", "FB_THEORY_EXIT_2026-11-21_common.py")


def load():
    A = pd.read_csv(os.path.join(D, "bt_anchors.csv"))
    B = pd.read_csv(os.path.join(D, "bt_cells.csv"))
    B = B.merge(A[["rid", "pair", "cf_entry_t"]], on="rid")
    B["te"] = B.cf_entry_t.astype("int64")
    B["block"] = np.where(B.te < SPLIT, "DESIGN", "HOLDOUT")
    B["fid"] = B.rid
    RF = pd.read_csv(os.path.join(D, "real_fills.csv"))
    RF["today"] = ~(RF.entry_funding_rate > 0.00005) & (RF.dist_base > 2.09)
    RC = pd.read_csv(os.path.join(D, "real_cells.csv")).merge(RF[["fid", "pair", "te", "period", "today", "k"]], on="fid")
    RC["mode"] = "row"
    return B, RF, RC


def wide(Z, mode="row"):
    z = Z[Z["mode"] == mode]
    N = z.pivot(index="fid", columns="cell", values="net")
    W = z.pivot(index="fid", columns="cell", values="why")
    XT = z.pivot(index="fid", columns="cell", values="xt")
    meta = z.drop_duplicates("fid").set_index("fid")[["pair", "te"]]
    return N, W, XT, meta


def read(N, XT, meta, cell, cluster="day", d0=None, d1=None):
    ref = N["REF"]; x = N[cell]; ok = ref.notna() & x.notna()
    ref, x = ref[ok], x[ok]; dl = x - ref; m = meta.loc[ok.index[ok]]
    day = pd.to_datetime(m.te, unit="ms").dt.floor("D")
    keys = (m.te // 86_400_000 // (10 if cluster == "10d" else 1)).values
    mean, lo, hi, p = K.block_boot(dl.values, keys)
    saved = dl[(ref < 0) & (dl > 0)].sum() * USD
    gained = dl[(ref >= 0) & (dl > 0)].sum() * USD
    cut = -dl[dl < 0].sum() * USD
    lmin, lmax = K.loo(dl.values)
    pair_share = day_share = np.nan
    if dl.sum() > 0:
        pair_share = dl.groupby(m.pair.values).sum().max() / dl.sum()
        day_share = dl.groupby(day.values).sum().max() / dl.sum()
    d0 = d0 or pd.to_datetime(m.te.min(), unit="ms"); d1 = d1 or pd.to_datetime(max(XT.loc[ok.index[ok], "REF"].max(), XT.loc[ok.index[ok], cell].max()), unit="ms")
    b_dcr, b_dd, b_wd = K.dcr(ref.values, XT.loc[ok.index[ok], "REF"].values, d0, d1)
    a_dcr, a_dd, a_wd = K.dcr(x.values, XT.loc[ok.index[ok], cell].values, d0, d1)
    return dict(cell=cell, N=int(ok.sum()), WR=(x > 0).mean() * 100, avg=x.mean(), ref_avg=ref.mean(), d=mean, lo=lo, hi=hi, P=p, sumd=dl.sum(),
                usd=dl.sum() * USD, saved=saved, gained=gained, cut=cut, loo_min=lmin, loo_max=lmax, pair_sh=pair_share, day_sh=day_share,
                dcr_b=b_dcr, dcr_a=a_dcr, dd_b=b_dd, dd_a=a_dd, wd_b=b_wd, wd_a=a_wd,
                avgloss_b=ref[ref < 0].mean(), avgloss_a=x[x < 0].mean(), n_changed=int((dl.abs() > 1e-9).sum()))


def table(rows, cols):
    return K.md(pd.DataFrame(rows)[cols])


def main():
    B, RF, RC = load()
    CL = [c["name"] for c in K.cells()]
    fam = {c["name"]: c["fam"] for c in K.cells()}
    out, md = {}, []
    blocks = {}
    for blk in ("DESIGN", "HOLDOUT"):
        for mode in ("row", "s20"):
            blocks[(blk, mode)] = wide(B[B.block == blk], mode)
    # real blocks
    RCm = RC.copy()
    real = {"MASTER as taken": RCm[RCm.period == "MASTER"], "MASTER today's filters": RCm[(RCm.period == "MASTER") & RCm.today],
            "BATCH (≥ 11-17 17:00)": RCm[RCm.period == "BATCH"], "REAL all, today's filters": RCm[RCm.today]}
    rw = {k: wide(v) for k, v in real.items()}
    bounds = {("DESIGN", "row"): ("2024-06-24", "2025-07-03"), ("HOLDOUT", "row"): ("2025-07-01", "2026-11-20"),
              ("DESIGN", "s20"): ("2024-06-24", "2025-07-03"), ("HOLDOUT", "s20"): ("2025-07-01", "2026-11-20")}
    allr = {}
    for (blk, mode), (N, W, XT, meta) in blocks.items():
        allr[(blk, mode)] = {c: read(N, XT, meta, c, "10d", *map(pd.Timestamp, bounds[(blk, mode)])) for c in CL if c != "REF"}
    for k, (N, W, XT, meta) in rw.items():
        allr[(k, "row")] = {c: read(N, XT, meta, c, "day") for c in CL if c != "REF"}
    # REF summary per block
    md.append("## REF (today's exit) per block\n")
    rr = []
    for key in [("DESIGN", "row"), ("HOLDOUT", "row")] + [(k, "row") for k in rw]:
        N, W, XT, meta = blocks[key] if key in blocks else rw[key[0]]
        ref = N["REF"].dropna(); wh = W["REF"].loc[ref.index]
        r0 = allr[key]["X_TP6"]
        rr.append(dict(block=key[0], N=len(ref), WR=(ref > 0).mean() * 100, avg=ref.mean(), sum=ref.sum(), usd=ref.sum() * USD,
                       TP=int((wh == "TP").sum()), STOP=int((wh == "STOP").sum()), TIME=int((wh.isin(["TIME", "OPEN"])).sum()),
                       DCR=r0["dcr_b"], maxDD=r0["dd_b"], worst_day=r0["wd_b"], avg_loss=ref[ref < 0].mean()))
    md.append(K.md(pd.DataFrame(rr)))
    # picks on DESIGN
    des = allr[("DESIGN", "row")]
    picks = {}
    for f in FAMS:
        cs = [c for c in CL if fam[c] == f]
        picks[f] = max(cs, key=lambda c: des[c]["d"])
    out["picks"] = picks
    md.append("\n## DESIGN family picks (frozen before MASTER / HOLDOUT were read)\n")
    md.append(", ".join(f"{f}: **{c}** (DESIGN Δ {des[c]['d']:+.3f})" for f, c in picks.items()))
    # every cell, every block
    cols = ["cell", "N", "WR", "avg", "d", "lo", "hi", "P", "usd", "saved", "gained", "cut", "dcr_b", "dcr_a"]
    for key in [("MASTER today's filters", "row"), ("MASTER as taken", "row"), ("BATCH (≥ 11-17 17:00)", "row"), ("REAL all, today's filters", "row"),
                ("DESIGN", "row"), ("HOLDOUT", "row"), ("HOLDOUT", "s20")]:
        md.append(f"\n## Every cell — {key[0]}{' · flat 0.20 % slip' if key[1] == 's20' else ''} (Δ = cell − REF per fill; $ at $3,050 / fill)\n")
        rows = [dict(allr[key][c], cell=f"{'**' if c in picks.values() else ''}{c}{'**' if c in picks.values() else ''}") for c in CL if c != "REF"]
        md.append(table(rows, cols))
    # gates
    md.append("\n## Gates per pick\n")
    G = []
    M = allr[("MASTER today's filters", "row")]; H = allr[("HOLDOUT", "row")]; H20 = allr[("HOLDOUT", "s20")]
    NH, WH, XTH, metaH = blocks[("HOLDOUT", "row")]
    for f, c in picks.items():
        m = M[c]
        g = dict(fam=f, cell=c, M_d=m["d"], M_P=m["P"], M_saved=m["saved"], M_cut=m["cut"], M_tp6=M["X_TP6"]["d"],
                 M1=m["d"] >= 0, M2=m["saved"] >= m["cut"], M3=(m["P"] >= 0.80) if m["P"] == m["P"] else False, M4=m["d"] > M["X_TP6"]["d"])
        if f == "C":
            g["M5"] = m["d"] > M[MATCH[c]]["d"]; g["M_ctrl"] = M[MATCH[c]]["d"]
        h = H[c]
        dl = (NH[c] - NH["REF"]).dropna(); hy = pd.to_datetime(metaH.loc[dl.index, "te"], unit="ms")
        half = hy.dt.year.astype(str) + np.where(hy.dt.month <= 6, "-H1", "-H2")
        hs = dl.groupby(half.values).mean()
        g.update(H_d=h["d"], H_lo=h["lo"], H_hi=h["hi"], H_halves=" / ".join(f"{k} {v:+.2f}" for k, v in hs.items()),
                 H1=h["lo"] > 0, H2=(hs > 0).sum() >= 2, H3=H20[c]["d"] > 0, H4=(h["pair_sh"] < 0.5 and h["day_sh"] < 0.5) if h["sumd"] > 0 else False,
                 H5=h["saved"] >= h["cut"], H6=h["dcr_a"] >= h["dcr_b"], H_s20=H20[c]["d"], M_loo_min=m["loo_min"])
        mp_ = all(g[k] for k in ("M1", "M2", "M3", "M4")) and (g.get("M5", True))
        hp = all(g[k] for k in ("H1", "H2", "H3", "H4", "H5", "H6"))
        g["verdict"] = ("SHIP CANDIDATE" if hp else "OBSERVE (HOLDOUT caveat)") if mp_ else "REJECT (master)"
        if mp_ and m["loo_min"] < 0:
            g["verdict"] += " · FRAGILE"
        G.append(g)
    Gd = pd.DataFrame(G)
    md.append(K.md(Gd))
    # structure vs its distance-matched fixed stop (is it more than "a different stop width"?)
    md.append("\n## Structure stop vs its distance-matched fixed stop (paired Δ = C − K per fill; CI = day / 10-day blocks)\n")
    sv = []
    for blk, (N_, XT_, meta_, cl_) in {"MASTER today's filters": (*[rw["MASTER today's filters"][i] for i in (0, 2, 3)], "day"),
                                        "MASTER as taken": (*[rw["MASTER as taken"][i] for i in (0, 2, 3)], "day"),
                                        "BATCH": (*[rw["BATCH (≥ 11-17 17:00)"][i] for i in (0, 2, 3)], "day"),
                                        "DESIGN": (*[blocks[("DESIGN", "row")][i] for i in (0, 2, 3)], "10d"),
                                        "HOLDOUT": (*[blocks[("HOLDOUT", "row")][i] for i in (0, 2, 3)], "10d")}.items():
        for c, kc in (("C_PK0", "K_MPK"), ("C_PK1", "K_MPK"), ("C_VWAP", "K_MVWAP"), ("C_CFHI", "K_MCFHI")):
            dl = (N_[c] - N_[kc]).dropna()
            keys = (meta_.loc[dl.index, "te"] // 86_400_000 // (10 if cl_ == "10d" else 1)).values
            mean, lo, hi, p = K.block_boot(dl.values, keys)
            sv.append(dict(block=blk, cell=c, control=kc, N=len(dl), d=mean, lo=lo, hi=hi, P=p))
    md.append(K.md(pd.DataFrame(sv)))
    # half-years + the backtest window that overlaps the master (≥ 2026-10-01)
    md.append("\n## Backtest Δ by half-year and on the master-overlap window (entries ≥ 2026-10-01) — mean Δ %/fill (N)\n")
    NB = pd.concat([blocks[("DESIGN", "row")][0], blocks[("HOLDOUT", "row")][0]])
    MB = pd.concat([blocks[("DESIGN", "row")][3], blocks[("HOLDOUT", "row")][3]])
    ts = pd.to_datetime(MB.loc[NB.index, "te"], unit="ms")
    hlab = ts.dt.year.astype(str) + np.where(ts.dt.month <= 6, "-H1", "-H2")
    hy = []
    for c in ("X_TP6", "A_FARB", "A_R786", "C_PK0", "C_PK1", "C_VWAP", "C_CFHI", "K_MPK", "K_MVWAP", "K_S12", "K_S15", "F_P1", "T_H24_UW", "B_X4_N960"):
        dl = NB[c] - NB["REF"]
        r = {"cell": c}
        for hname, gi in dl.groupby(hlab.values):
            r[hname] = f"{gi.mean():+.2f} ({len(gi)})"
        w = dl[(ts >= "2026-10-01").values]
        r["≥ 2026-10-01"] = f"{w.mean():+.2f} ({len(w)})"
        hy.append(r)
    md.append(K.md(pd.DataFrame(hy)))
    out.update(allr=allr, gates=Gd, real_fills=RF)
    pickle.dump(out, open(os.path.join(D, "eval.pkl"), "wb"))
    # per-fill real table for picks + controls
    N, W, XT, meta = rw["MASTER as taken"]
    Nb, Wb, XTb, metab = rw["BATCH (≥ 11-17 17:00)"]
    show = ["REF", "X_TP6"] + list(picks.values()) + ["K_S15", "K_MPK", "C_PK0"]
    show = list(dict.fromkeys(show))
    PF = pd.concat([N[show], Nb[show]]).join(RF.set_index("fid")[["k", "pair", "today", "booked", "dist_base", "peak_d", "vwap_d", "cfhi_d"]])
    md.append("\n## Real fills — per fill (net % at today's costs; today = passes today's filters)\n")
    md.append(K.md(PF.reset_index(drop=True)[["k", "pair", "today", "booked"] + show + ["dist_base", "peak_d", "vwap_d", "cfhi_d"]]))
    open(os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_tables.md"), "w").write("# FB_THEORY_EXIT_2026-11-21 — generated tables\n\n" + "\n".join(md) + "\n")
    print("\n".join(md[:6]))
    print(K.md(Gd))


if __name__ == "__main__":
    main()
