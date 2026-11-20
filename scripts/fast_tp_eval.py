#!/usr/bin/env python3
"""FAST_TP (PREREG reports/FAST_TP_2026-11-20_PREREG.md) — EVAL. Research only, BACKTEST.
Per sleeve: parity · Δ = 0 assertion on never-triggered fills · DESIGN grid + pick · HOLDOUT (picked cell, then the full grid) ·
day-block bootstrap CI · family-wise day sign-flip max-t null · decomposition (cut winners / saved / other, Σ net %) · random-entry
control · pair concentration + leave-one-pair-out · half-years · one-size (0.61) DCR before → after · chart.
Writes reports/FAST_TP_2026-11-20_grid.csv, _summary.json, _chart.png.  Usage: venv/bin/python -I scripts/fast_tp_eval.py"""
import json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_tp_sim as S   # noqa: E402
RP = S.RP; D = S.D; OUT = os.path.join(RP, "FAST_TP_2026-11-20")
DAY, MIN = 86_400_000, 60_000
SIZE = 0.61
CELLS = S.CELLS; NC = len(CELLS)
SPLIT = {"EXH": "2025-07-01", "DIST": "2025-07-01", "FLATBASE": "2025-07-01", "WILLY": "2026-01-01", "HT": "2026-01-01"}
START = {"EXH": "2023-07-01", "DIST": "2023-07-01", "FLATBASE": "2023-07-01", "WILLY": "2025-01-01", "HT": "2025-01-01"}
END = "2026-11-20"
MINTRIG = {"EXH": 30, "DIST": 30, "FLATBASE": 30, "WILLY": 15, "HT": 15}
NAME = {"EXH": "EXHAUSTION_SHORT", "DIST": "DISTRIBUTION_SHORT", "FLATBASE": "FLATBASE_SHORT", "WILLY": "FRENZY_WILLY", "HT": "FRENZY_WILLY_HT"}
RNG = np.random.default_rng(20261120)
B_BOOT, B_NULL = 4000, 2000


def ms(d):
    return int(pd.Timestamp(d).value // 10**6)


def load(sl, nm):
    Z = pd.read_pickle(os.path.join(D, f"sim_{sl}_{nm}.pkl"))
    Z = Z[(Z.ts >= ms(START[sl])) & (Z.ts < ms(END))].reset_index(drop=True)
    Z["day"] = Z.ts // DAY
    Z["block"] = np.where(Z.ts >= ms(SPLIT[sl]), "HOLDOUT", "DESIGN")
    Z["half"] = S.half_of(Z.ts.values)
    Z["ref_xms"] = Z.ts + (Z.ref_idx + 1) * MIN
    return Z


def mats(Z):
    V = np.stack(Z.nets.values).astype(float); Fi = np.stack(Z.fire.values).astype(int)
    Dl = V - Z.ref_net.values[:, None]
    assert np.abs(Dl[Fi < 0]).max(initial=0) < 1e-4, "Δ ≠ 0 on a never-triggered fill"
    return V, Fi, Dl


def day_t(dl, day):
    s = pd.Series(dl).groupby(day).sum()
    if len(s) < 3 or s.std() == 0:
        return 0.0
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s))))


def boot_ci(dl, day, B=B_BOOT):
    """day-block bootstrap (10-day blocks) of mean Δ per fill."""
    blk = (day - day.min()) // 10
    g = pd.DataFrame(dict(b=blk, d=dl)).groupby("b").d.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    if len(s) < 3:
        return np.nan, np.nan
    i = RNG.integers(0, len(s), (B, len(s)))
    m = s[i].sum(1) / c[i].sum(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def null_maxt(Dl, day, elig, B=B_NULL):
    """family-wise day sign-flip: same flip for every cell, max-t over the eligible cells."""
    if not elig.any():
        return np.nan
    Sd = pd.DataFrame(Dl[:, elig]).groupby(day).sum().values   # days × cells
    nd = len(Sd)
    out = np.empty(B)
    for b in range(B):
        f = RNG.choice([-1.0, 1.0], nd)[:, None]
        x = Sd * f
        mu = x.mean(0); sd = x.std(0, ddof=1)
        t = np.where(sd > 0, mu / (sd / np.sqrt(nd)), 0.0)
        out[b] = t.max()
    return float(np.percentile(out, 95))


def dcr(net, xms, d0, d1):
    r = pd.Series(net * SIZE / 100).groupby(xms // DAY).sum().reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values)
    return (eq[-1] ** (1 / len(r)) - 1) * 100 if eq[-1] > 0 else np.nan


def stats(v):
    v = np.asarray(v, float)
    return dict(N=len(v), WR=float((v > 0).mean() * 100) if len(v) else np.nan, avg=float(v.mean()) if len(v) else np.nan,
                usd=float(v.sum() * 6100 / 100))


def run(sl):
    Z = load(sl, "MAIN"); V, Fi, Dl = mats(Z)
    res = dict(sleeve=NAME[sl], n=len(Z), days=int(Z.day.nunique()))
    # parity
    par = {}
    if "parity" in Z:
        par["path_parity_rate"] = float(Z.parity.mean())
    if "file_net" in Z:
        col = "par_net" if "par_net" in Z else "ref_net"
        par["file_avg"] = float(Z.file_net.mean()); par["resim_avg_at_file_exit"] = float(Z[col].mean())
        par["max_abs_diff"] = float((Z.file_net - Z[col]).abs().max()); par["corr"] = float(np.corrcoef(Z.file_net, Z[col])[0, 1])
    par["ref_avg"] = {b: float(Z.ref_net[Z.block == b].mean()) for b in ("DESIGN", "HOLDOUT")}
    par["ref_avg"]["ALL"] = float(Z.ref_net.mean())
    par["ref_kind"] = Z.ref_kind.value_counts().to_dict()
    res["parity"] = par
    # grid
    rows = []
    for b in ("DESIGN", "HOLDOUT", "ALL"):
        m = (Z.block == b).values if b != "ALL" else np.ones(len(Z), bool)
        for j, c in enumerate(CELLS):
            fired = m & (Fi[:, j] >= 0)
            kinds = Z.ref_kind[fired]
            rows.append(dict(sleeve=NAME[sl], block=b, cell=c, N=int(m.sum()), trig=int(fired.sum()), d_mean=float(Dl[m, j].mean()),
                             d_sum=float(Dl[m, j].sum()), t=day_t(Dl[m, j], Z.day.values[m]), avg=float(V[m, j].mean()),
                             ref_avg=float(Z.ref_net[m].mean()), cut_tp=int((kinds == "TP").sum()), saved_sl=int((kinds == "SL").sum()),
                             other=int((~kinds.isin(["TP", "SL"])).sum()),
                             d_cut=float(Dl[fired & (Z.ref_kind == "TP").values, j].sum()),
                             d_saved=float(Dl[fired & (Z.ref_kind == "SL").values, j].sum()),
                             d_other=float(Dl[fired & (~Z.ref_kind.isin(["TP", "SL"])).values, j].sum()),
                             halves_pos=int(sum(Dl[(Z.half == h).values & m, j].sum() > 0 for h in Z.half[m].unique())),
                             halves=int(Z.half[m].nunique())))
    G = pd.DataFrame(rows)
    des = G[G.block == "DESIGN"].set_index("cell")
    elig = (des.trig >= MINTRIG[sl]).reindex(CELLS).values
    cand = des[elig]
    pick = cand.d_mean.idxmax() if len(cand) and cand.d_mean.max() > 0 else None
    best_des = cand.d_mean.idxmax() if len(cand) else None
    mD = (Z.block == "DESIGN").values
    nullp = null_maxt(Dl[mD], Z.day.values[mD], elig)
    res["design_best_cell"] = best_des; res["pick"] = pick; res["null95_maxt"] = nullp
    res["design_best_t"] = float(des.loc[best_des, "t"]) if best_des else np.nan
    allg = G[G.block == "ALL"].set_index("cell")
    res["full_best_cell_hindsight"] = allg.d_mean.idxmax()
    # chosen cell for the detail reads: the pick, else the DESIGN best (shown as rejected), else F_X2_N5
    cc = pick or best_des or "F_X2_N5"; j = CELLS.index(cc); res["detail_cell"] = cc
    det = {}
    for b in ("DESIGN", "HOLDOUT", "ALL"):
        m = (Z.block == b).values if b != "ALL" else np.ones(len(Z), bool)
        lo, hi = boot_ci(Dl[m, j], Z.day.values[m])
        d0, d1 = ms(START[sl] if b != "HOLDOUT" else SPLIT[sl]) // DAY, ms(SPLIT[sl] if b == "DESIGN" else END) // DAY
        xms_c = np.where(Fi[:, j] >= 0, Z.ts + (Fi[:, j] + 1) * MIN, Z.ref_xms)
        det[b] = dict(before=stats(Z.ref_net[m]), after=stats(V[m, j]), d_mean=float(Dl[m, j].mean()), ci=(lo, hi),
                      t=day_t(Dl[m, j], Z.day.values[m]), trig=int((Fi[m, j] >= 0).sum()),
                      dcr_before=dcr(Z.ref_net.values[m], Z.ref_xms.values[m], d0, d1), dcr_after=dcr(V[m, j], xms_c[m], d0, d1),
                      days=int(Z.day[m].nunique()), pairdays=int(Z[m].groupby(["pair", "day"]).ngroups))
    res["detail"] = det
    # halves
    res["halves"] = {h: dict(N=int((Z.half == h).sum()), trig=int(((Z.half == h).values & (Fi[:, j] >= 0)).sum()),
                             d_mean=float(Dl[(Z.half == h).values, j].mean()), ref=float(Z.ref_net[Z.half == h].mean()))
                     for h in sorted(Z.half.unique())}
    # concentration / LOO
    bp = pd.Series(Dl[:, j]).groupby(Z.pair.values).sum().sort_values()
    pos = bp[bp > 0].sum()
    res["top_pair"] = (bp.index[-1], float(bp.iloc[-1]), float(bp.iloc[-1] / pos) if pos > 0 else np.nan)
    res["worst_pair"] = (bp.index[0], float(bp.iloc[0]))
    loo = [float(Dl[(Z.pair != p).values, j].mean()) for p in bp.index]
    res["loo_min"] = float(min(loo)); res["loo_max"] = float(max(loo))
    # decomposition for the detail cell (ALL)
    f = Fi[:, j] >= 0
    dec = []
    for k in ("TP", "SL", "TIME", "DYN"):
        mm = f & (Z.ref_kind == k).values
        if mm.sum():
            dec.append(dict(ref_kind=k, n=int(mm.sum()), ref_sum=float(Z.ref_net[mm].sum()), cell_sum=float(V[mm, j].sum()),
                            d_sum=float(Dl[mm, j].sum()), usd=float(Dl[mm, j].sum() * 61)))
    res["decomp"] = dec
    # speed check: share of the REF full-TP winners whose first 5 / 15 min reached +X
    tpw = Z.ref_kind == "TP"
    res["tp_winners_fast"] = {f"mfe5>={x}": float((Z.mfe5[tpw] >= x).mean()) for x in (1, 2, 3, 4)}
    res["sl_losers_fast"] = {f"mfe5>={x}": float((Z.mfe5[Z.ref_kind == "SL"] >= x).mean()) for x in (1, 2, 3, 4)}
    # random control
    try:
        Rz = load(sl, "RND"); Vr, Fr, Dr = mats(Rz)
        ctl = {}
        for b in ("DESIGN", "HOLDOUT", "ALL"):
            m = (Rz.block == b).values if b != "ALL" else np.ones(len(Rz), bool)
            ctl[b] = dict(N=int(m.sum()), ref=float(Rz.ref_net[m].mean()), trig=int((Fr[m, j] >= 0).sum()), d_mean=float(Dr[m, j].mean()),
                          d_per_trig=float(Dr[m & (Fr[:, j] >= 0), j].mean()) if (m & (Fr[:, j] >= 0)).any() else np.nan)
        ctl["grid_all_dmean"] = {c: float(Dr[:, i].mean()) for i, c in enumerate(CELLS)}
        res["control"] = ctl
        G2 = pd.DataFrame(dict(cell=CELLS, rnd_d_mean=[float(Dr[:, i].mean()) for i in range(NC)],
                               rnd_trig=[int((Fr[:, i] >= 0).sum()) for i in range(NC)]))
        G = G.merge(G2.assign(block="ALL"), on=["cell", "block"], how="left")
    except FileNotFoundError:
        res["control"] = None
    # pass bar
    H = det["HOLDOUT"]
    halves_pos = sum(v["d_mean"] > 0 for v in res["halves"].values())
    res["bar"] = dict(holdout_pos_ci=bool(pick and H["d_mean"] > 0 and H["ci"][0] > 0), design_pos=bool(pick is not None),
                      beats_null=bool(pick and res["design_best_t"] > nullp),
                      halves=f"{halves_pos}/{len(res['halves'])}", halves_ok=bool(halves_pos >= 2 / 3 * len(res["halves"])),
                      top_pair_ok=bool(res["top_pair"][2] < 0.5 if np.isfinite(res["top_pair"][2]) else False), loo_ok=bool(res["loo_min"] > 0))
    res["chart"] = chart_data(Z, Fi, j)
    return res, G


def chart_data(Z, Fi, j):
    f = Fi[:, j] >= 0
    P = np.stack(Z.cpath.values).astype(float)
    out = {}
    for nm, m in (("trig_fullTP", f & (Z.ref_kind == "TP").values), ("trig_notTP", f & (Z.ref_kind != "TP").values),
                  ("notrig_fullTP", (~f) & (Z.ref_kind == "TP").values), ("all", np.ones(len(Z), bool))):
        out[nm] = (int(m.sum()), np.nanmean(P[m], 0).tolist() if m.sum() else None)
    return out


def plot(allres):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 5, figsize=(26, 5.2))
    col = {"trig_fullTP": "#1b7f3b", "trig_notTP": "#c0392b", "notrig_fullTP": "#2e86c1", "all": "#7f7f7f"}
    lab = {"trig_fullTP": "fast +X, REF later hit full TP (cut winners)", "trig_notTP": "fast +X, REF did not hit full TP (saved / other)",
           "notrig_fullTP": "no fast +X, REF hit full TP", "all": "all fills"}
    for a, (sl, r) in zip(ax, allres.items()):
        cc = r["detail_cell"]; X = float(cc.split("_")[1][1:]) if cc.startswith("F_") else None
        for k, (n, y) in r["chart"].items():
            if y is None:
                continue
            a.plot(np.arange(1, len(y) + 1), y, color=col[k], lw=2 if k != "all" else 1.2, ls="-" if k != "all" else "--", label=f"{lab[k]} (n={n})")
        if X:
            a.axhline(X, color="k", lw=0.6, ls=":")
            a.axvline(int(cc.split("_N")[1]), color="k", lw=0.6, ls=":")
        a.axhline(0, color="k", lw=0.5)
        a.set_xscale("log"); a.set_xlim(1, 360)
        a.set_title(f"{NAME[sl]} — cell {cc}", fontsize=10); a.set_xlabel("minutes after entry (log)"); a.set_ylabel("favourable move at 1m close, %")
        a.legend(fontsize=7, loc="lower left")
    fig.suptitle("FAST_TP 2026-11-20 (BACKTEST): average path of fills that hit +X within N min, split by whether the current exit later reached its full TP", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT + "_chart.png", dpi=110)


def main():
    allres = {}; Gs = []
    for sl in ("EXH", "DIST", "FLATBASE", "WILLY", "HT"):
        r, G = run(sl); allres[sl] = r; Gs.append(G)
        print(sl, "pick", r["pick"], "best", r["design_best_cell"], "t", round(r["design_best_t"], 2), "null", round(r["null95_maxt"], 2),
              {b: (round(v["d_mean"], 3), tuple(round(x, 3) for x in v["ci"]), v["trig"]) for b, v in r["detail"].items()}, flush=True)
    pd.concat(Gs).to_csv(OUT + "_grid.csv", index=False)
    for r in allres.values():
        r.pop("chart_cache", None)
    plot(allres)
    for r in allres.values():
        r.pop("chart")
    json.dump(allres, open(OUT + "_summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
