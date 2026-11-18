#!/usr/bin/env python3
"""DIST_EXIT (PREREG reports/DIST_EXIT_2026-11-18_PREREG.md) — EVAL.
  --phase design  : DESIGN grid (PRIMARY DIST cohort) + selection → reports/DIST_EXIT_data/survivors.json (frozen). No HOLDOUT number printed.
  --phase holdout : HOLDOUT ship bar for the frozen survivors + REF, then every descriptive read (null, control, saved / deeper, periods,
                    concentration, slip, Kelly, DCR, joint add, EXH cross-check, chart, trade rows).
Usage: venv/bin/python -I scripts/dist_exit_eval.py --phase design|holdout"""
import argparse, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_exit_sim as DS        # noqa: E402
import dist_wl_survivors as SV    # noqa: E402
import nospot_sleeve_lib as NL    # noqa: E402
R = DS.R; D = DS.D; RP = os.path.join(R, "reports")
DAY = 86_400_000; H1 = 3_600_000
TS = lambda s: pd.Timestamp(s).value // 10**6
DES = (TS("2023-07-01"), TS("2025-07-01")); HOL = (TS("2025-07-01"), TS("2026-11-18"))
HALVES = [("23-H2", "2023-07-01", "2024-01-01"), ("24-H1", "2024-01-01", "2024-07-01"), ("24-H2", "2024-07-01", "2025-01-01"),
          ("25-H1", "2025-01-01", "2025-07-01"), ("25-H2", "2025-07-01", "2026-01-01"), ("26-H1", "2026-01-01", "2026-07-01"),
          ("26-H2", "2026-07-01", "2026-11-18")]
REP = dict(kind="1D", var="rsi1h", lo=74.07103, hi=np.inf, mode="keep_only", var2="", lo2=np.nan, hi2=np.nan)
SIZE = 0.61
CL = DS.cells(); NAMES = [c["name"] for c in CL]; NC = len(CL); FAM = np.array([c["fam"] for c in CL])
SEL = np.array([c["fam"] != "REF" for c in CL])
RNG = np.random.default_rng(20261118)


# ---------------------------------------------------------------- loading / costs
def load_set(nm):
    z = np.load(os.path.join(D, f"sim_{nm}.npz"))
    return {k: z[k] for k in z.files}


def net_mat(Z, slip, flat=None, dyn2=False):
    """n × C net % (NaN = not simulated / censored)."""
    k = Z["k"]; tk = k != 0
    s = (np.full(len(slip), flat) if flat is not None else slip)[:, None] * np.ones((1, NC))
    sx = np.where(dyn2 & (k == 3), 2 * s, s)
    v = Z["r"].astype(float) - s - np.where(tk, sx, 0.0) - 0.05 - np.where(tk, 0.05, 0.02) + Z["f"].astype(float)
    bad = (~Z["ok"][:, None]) | (k == 9) | (k < 0)
    return np.where(bad, np.nan, v)


def one_pos(pair, T, X, valid):
    """greedy one position per pair, time order (= unified_short_lib.one_pos), on rows with valid."""
    ii = np.nonzero(valid)[0]
    o = np.lexsort((T[ii], pair[ii])); ii = ii[o]
    keep = []; lp = None; busy = -1
    for i in ii:
        if pair[i] != lp:
            lp = pair[i]; busy = -1
        if T[i] < busy:
            continue
        keep.append(i); busy = X[i]
    m = np.zeros(len(pair), bool); m[keep] = True
    return m


def wilson_floor(k, n, z=1.645):
    if n <= 0:
        return 0.0
    p = k / n; den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0))) / den


def kelly(v, day=None):
    w = v > 0
    if w.sum() == 0 or (~w).sum() == 0:
        return np.nan, np.nan
    aw, al = v[w].mean(), -v[~w].mean(); b = aw / al
    n = len(np.unique(day)) if day is not None else len(v)
    p = wilson_floor(w.mean() * n, n)
    f = p - (1 - p) / b
    return f, f / (al / 100)


def dcr(v, xms, span, size=SIZE):
    lo, hi = span[0] // DAY, span[1] // DAY - 1
    s = pd.Series(v / 100 * size).groupby(xms // DAY).sum()
    full = s.reindex(np.arange(lo, max(hi, s.index.max() if len(s) else hi) + 1), fill_value=0.0)
    eq = np.cumprod(1 + full.values)
    if eq[-1] <= 0 or (eq <= 0).any():
        return -100.0, -100.0, full.min() * 100
    return (eq[-1] ** (1 / len(full)) - 1) * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100, full.min() * 100


def block_boot_mean(d, x, span, nb=2000, block=10, rng=None):
    """day-block bootstrap of mean(x) with x grouped by day d; calendar-day blocks over span. → (lo, hi, draws)."""
    rng = rng if rng is not None else np.random.default_rng(7)
    lo = span[0] // DAY; nd = span[1] // DAY - lo
    di = (d - lo).astype(int); ok = (di >= 0) & (di < nd)
    s = np.bincount(di[ok], weights=x[ok], minlength=nd); c = np.bincount(di[ok], minlength=nd).astype(float)
    nblk = int(np.ceil(nd / block)); st = rng.integers(0, nd - block + 1, (nb, nblk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :nd]
    m = s[idx].sum(1) / np.maximum(c[idx].sum(1), 1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), m


# ---------------------------------------------------------------- the frame
def build():
    S = pd.read_pickle(os.path.join(RP, "DIST_WL_2026-11-18_data", "rows.pkl"))
    Z = load_set("L1")
    F = DS.frames()["L1"]
    slip = F.slip.values
    V = net_mat(Z, slip)
    prim = SV.keep_mask(S, REP)
    pair = S.pair.astype(str).values; T = S.entry_ms.values.astype(np.int64)
    return S, Z, V, slip, prim, pair, T


def cohort_onepos(V, Z, pair, T, base):
    """per cell one-position mask (n × C)."""
    M = np.zeros(V.shape, bool)
    for j in range(NC):
        M[:, j] = one_pos(pair, T, Z["xm"][:, j], base & np.isfinite(V[:, j]))
    return M


def cell_stats(v, xms, day, kind, span):
    n = len(v)
    if n == 0:
        return dict(N=0)
    w = v > 0
    nd = (span[1] - span[0]) / DAY
    d_, dd, wd = dcr(v, xms, span)
    s = np.sort(v)[::-1]
    return dict(N=n, tpd=n / nd, WR=w.mean() * 100, avg=v.mean(), avg_win=v[w].mean() if w.any() else np.nan,
                avg_loss=v[~w].mean() if (~w).any() else np.nan, b5=s[int(np.ceil(0.05 * n)):].mean() if n > 20 else np.nan,
                tp=(kind == 0).mean() * 100, stop=(kind == 1).mean() * 100, dyn=(kind == 3).mean() * 100, time=(kind == 2).mean() * 100,
                dcr=d_, maxdd=dd, worst=wd, kel_t=kelly(v)[0], kel_d=kelly(v, day)[0], kel_d_not=kelly(v, day)[1])


def neighbours(j):
    c = CL[j]; f = c["fam"]
    out = []
    if f == "A":
        TPs = [1, 1.5, 2, 3, 5, 10]; SLs = ["2", "3", "4", "ATR", "6"]
        nm = c["name"]; tp = float(nm.split("_")[1][2:]); sl = nm.split("_")[2][2:]
        ti, si = TPs.index(tp), SLs.index(sl)
        for a, b in ((ti - 1, si), (ti + 1, si), (ti, si - 1), (ti, si + 1)):
            if 0 <= a < len(TPs) and 0 <= b < len(SLs):
                out.append(f"A_TP{TPs[a]:g}_SL{SLs[b]}")
    elif f == "B":
        a = int(c["arm"]); out = [f"B_BE{x}" for x in (a - 1, a + 1) if x in (2, 3, 4)]
    elif f == "C":
        L = ["C_L31", "C_L42", "C_L53"]; i = L.index(c["name"]); out = [L[x] for x in (i - 1, i + 1) if 0 <= x < 3]
    elif f == "D":
        As = [2, 3, 4]; Gs = [1, 1.5, 2, 3]; ai, gi = As.index(int(c["arm"])), Gs.index(c["par"])
        for a, b in ((ai - 1, gi), (ai + 1, gi), (ai, gi - 1), (ai, gi + 1)):
            if 0 <= a < 3 and 0 <= b < 4:
                out.append(f"D_T{As[a]}_G{Gs[b]:g}")
    elif f == "D2":
        out = ["D_T3_A2" if c["name"] == "D_T3_A1" else "D_T3_A1"]
    elif f == "T":
        Hs = [4, 12, 24, 48]; i = Hs.index(c["hold"]); out = [f"T_{Hs[x]}h" for x in (i - 1, i + 1) if 0 <= x < 4]
    return out


def half_of(T):
    h = np.full(len(T), "", object)
    for nm, a, b in HALVES:
        h[(T >= TS(a)) & (T < TS(b))] = nm
    return h


# ---------------------------------------------------------------- phase DESIGN
def phase_design():
    S, Z, V, slip, prim, pair, T = build()
    print("1m share (all L1 rows):", round(Z["src1m"][Z["ok"]].mean() * 100, 1), "% ; ok", int(Z["ok"].sum()), "of", len(S))
    des = (T >= DES[0]) & (T < DES[1])
    M = cohort_onepos(V, Z, pair, T, prim & des)
    ref_set = M[:, 0]
    hv = half_of(T); day = T // DAY
    rows = []
    for j in range(NC):
        m = M[:, j]; v = V[m, j]
        st = cell_stats(v, Z["xm"][m, j], day[m], Z["k"][m, j], DES)
        pm = ref_set & np.isfinite(V[:, j]); dl = V[pm, j] - V[pm, 0]
        st["dpair"] = dl.mean()
        hp = [(dl[hv[pm] == h].mean() > 0) for h, _, _ in HALVES[:4] if (hv[pm] == h).sum() >= 10]
        st["halves_pos"] = f"{sum(hp)}/{len(hp)}"; st["halves_ok"] = len(hp) > 0 and sum(hp) >= 2 / 3 * len(hp)
        st["name"] = NAMES[j]; st["fam"] = FAM[j]
        rows.append(st)
    G = pd.DataFrame(rows).set_index("name")
    ref_avg = G.loc["REF_B", "avg"]
    G["beats_ref_avg"] = G.avg > ref_avg
    pl = []
    for j in range(NC):
        nb = neighbours(j)
        if not nb:
            pl.append((np.nan, False)); continue
        pos = [G.loc[x, "dpair"] > 0 for x in nb]
        pl.append((f"{sum(pos)}/{len(pos)}", sum(pos) >= 2 / 3 * len(pos)))
    G["plateau"] = [p[0] for p in pl]; G["plateau_ok"] = [p[1] for p in pl]
    G["survivor"] = SEL & (G.N >= 30) & G.beats_ref_avg & (G.dpair > 0) & G.halves_ok & G.plateau_ok
    sv = G[G.survivor].sort_values("dpair", ascending=False)
    picks = []; fams = set()
    for nm, r in sv.iterrows():
        f = "D" if r.fam in ("D", "D2") else r.fam
        if f in fams:
            continue
        picks.append(nm); fams.add(f)
        if len(picks) == 3:
            break
    G.to_csv(os.path.join(D, "design_grid.csv"))
    json.dump(dict(picks=picks, survivors=list(sv.index), frozen_utc=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
              open(os.path.join(D, "survivors.json"), "w"), indent=1)
    # L1 + RND (DESIGN only, descriptive)
    ML1 = cohort_onepos(V, Z, pair, T, des)
    l1 = []
    for j in range(NC):
        pm = ML1[:, 0] & np.isfinite(V[:, j]); l1.append((V[ML1[:, j], j].mean(), (V[pm, j] - V[pm, 0]).mean()))
    G["L1_avg"] = [a for a, _ in l1]; G["L1_dpair"] = [b for _, b in l1]
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 100)
    cols = ["fam", "N", "tpd", "WR", "avg", "avg_win", "avg_loss", "dpair", "halves_pos", "plateau", "tp", "stop", "dyn", "time",
            "dcr", "maxdd", "worst", "kel_d", "L1_avg", "L1_dpair", "survivor"]
    print(G[cols].round(3).to_string())
    print("PICKS", picks)
    G.to_csv(os.path.join(D, "design_grid.csv"))


# ---------------------------------------------------------------- phase HOLDOUT (+ all descriptive reads)
def phase_holdout():
    SJ = json.load(open(os.path.join(D, "survivors.json"))); picks = SJ["picks"]
    S, Z, V, slip, prim, pair, T = build()
    V20 = net_mat(Z, slip, flat=0.20); Vdy = net_mat(Z, slip, dyn2=True)
    hv = half_of(T); day = T // DAY
    J = [0] + [NAMES.index(p) for p in picks]
    res = dict(picks=picks)
    full = (T >= DES[0]) & (T < HOL[1]); hol = (T >= HOL[0]) & (T < HOL[1]); des = (T >= DES[0]) & (T < DES[1])
    MF = cohort_onepos(V, Z, pair, T, prim & full)       # one-position over the full span (used for all period reads)
    ref = MF[:, 0]
    out = {}
    for j in J:
        nm = NAMES[j]; o = {}
        for blk, msk, span in (("DESIGN", des, DES), ("HOLDOUT", hol, HOL), ("FULL", full, (DES[0], HOL[1]))):
            m = MF[:, j] & msk
            st = cell_stats(V[m, j], Z["xm"][m, j], day[m], Z["k"][m, j], span)
            st["avg20"] = V20[m, j].mean(); st["avgdyn2"] = Vdy[m, j].mean()
            st["kel_t_not"] = kelly(V[m, j])[1]
            pm = ref & msk & np.isfinite(V[:, j]); dl = V[pm, j] - V[pm, 0]; dl20 = V20[pm, j] - V20[pm, 0]
            st["dpair"] = dl.mean(); st["dpair20"] = dl20.mean(); st["Npair"] = int(pm.sum())
            lo, hi, _ = block_boot_mean(day[pm], dl, span)
            st["d_lo"], st["d_hi"] = lo, hi
            if j:
                pp = pd.Series(dl).groupby(pair[pm]).sum(); dd_ = pd.Series(dl).groupby(day[pm]).sum(); tot = dl.sum()
                st["top_pair_share"] = pp.max() / tot * 100 if tot > 0 else np.nan; st["top_pair"] = pp.idxmax()
                st["top_day_share"] = dd_.max() / tot * 100 if tot > 0 else np.nan
                loo = [(dl[pair[pm] != p].mean()) for p in pp.index]; st["loo_min"] = min(loo); st["loo_pair"] = pp.index[int(np.argmin(loo))]
            o[blk] = st
        hp = []
        for h, _, _ in HALVES:
            pm = ref & full & (hv == h) & np.isfinite(V[:, j])
            hp.append(dict(half=h, N=int((MF[:, j] & (hv == h)).sum()), avg=V[MF[:, j] & (hv == h), j].mean(),
                           ref_avg=V[ref & (hv == h), 0].mean(), dpair=(V[pm, j] - V[pm, 0]).mean() if pm.sum() else np.nan, Np=int(pm.sum())))
        o["halves"] = pd.DataFrame(hp)
        out[nm] = o
    res["cells"] = out

    # ----- family-wise null (White reality check, full span, all 54 + REF excluded)
    pm_all = ref & full
    dd = day[pm_all]; lo = DES[0] // DAY; nd = HOL[1] // DAY - lo; di = (dd - lo).astype(int)
    Dm = V[pm_all][:, :] - V[pm_all, 0][:, None]
    sums = np.zeros((nd, NC)); cnts = np.zeros((nd, NC))
    for j in range(1, NC):
        okj = np.isfinite(Dm[:, j])
        sums[:, j] = np.bincount(di[okj], weights=Dm[okj, j], minlength=nd); cnts[:, j] = np.bincount(di[okj], minlength=nd)
    mobs = sums.sum(0) / np.maximum(cnts.sum(0), 1)
    nb = 2000; block = 10; nblk = int(np.ceil(nd / block)); st_ = RNG.integers(0, nd - block + 1, (nb, nblk))
    idx = (st_[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :nd]
    mb = np.stack([sums[i].sum(0) / np.maximum(cnts[i].sum(0), 1) for i in idx])
    sd = mb.std(0); sd[sd == 0] = np.inf
    tobs = mobs / sd; tstar = ((mb - mobs) / sd)[:, 1:].max(1)
    res["null"] = dict(t_obs={NAMES[j]: float(tobs[j]) for j in range(1, NC)}, crit95=float(np.percentile(tstar, 95)),
                       p={NAMES[j]: float((tstar >= tobs[j]).mean()) for j in range(1, NC)}, best=NAMES[1 + int(np.argmax(tobs[1:]))])
    # ----- HOLDOUT, every cell (descriptive only, after the decision)
    allh = []
    MH = cohort_onepos(V, Z, pair, T, prim & hol)
    for j in range(NC):
        m = MH[:, j]; pm = MH[:, 0] & np.isfinite(V[:, j])
        st = cell_stats(V[m, j], Z["xm"][m, j], day[m], Z["k"][m, j], HOL); st["dpair"] = (V[pm, j] - V[pm, 0]).mean(); st["name"] = NAMES[j]
        st["t_full"] = tobs[j] if j else np.nan
        allh.append(st)
    res["hol_all"] = pd.DataFrame(allh).set_index("name")
    # ----- L1 secondary + RND control (Δ per cell, full span; DESIGN / HOLDOUT)
    ML1 = cohort_onepos(V, Z, pair, T, full)
    ZR = load_set("RND"); FR = DS.frames()["RND"]; VR = net_mat(ZR, FR.slip.values); TR = FR.entry_ms.values
    ctl = []
    for j in range(NC):
        r_ = dict(name=NAMES[j])
        for blk, a, b in (("DES", *DES), ("HOL", *HOL)):
            pm = ref & (T >= a) & (T < b) & np.isfinite(V[:, j]); r_[f"DIST_{blk}"] = (V[pm, j] - V[pm, 0]).mean()
            pm = ML1[:, 0] & (T >= a) & (T < b) & np.isfinite(V[:, j]); r_[f"L1_{blk}"] = (V[pm, j] - V[pm, 0]).mean()
            pr = (TR >= a) & (TR < b) & np.isfinite(VR[:, j]) & np.isfinite(VR[:, 0])
            r_[f"RND_{blk}"] = (VR[pr, j] - VR[pr, 0]).mean(); r_[f"RNDavg_{blk}"] = VR[pr, j].mean(); r_[f"RNDref_{blk}"] = VR[pr, 0].mean()
            r_[f"DISTavg_{blk}"] = V[ref & (T >= a) & (T < b) & np.isfinite(V[:, j]), j].mean()
        ctl.append(r_)
    res["control"] = pd.DataFrame(ctl).set_index("name")
    res["rnd_n"] = int((np.isfinite(VR[:, 0])).sum()); res["rnd_1m"] = float(ZR["src1m"][ZR["ok"]].mean())
    # ----- saved / deeper on REF's one-position entries (full + HOLDOUT)
    sd_rows = []
    for j in J[1:]:
        for blk, msk in (("FULL", full), ("HOLDOUT", hol)):
            pm = ref & msk & np.isfinite(V[:, j])
            d = V[pm, j] - V[pm, 0]; kr = Z["k"][pm, 0]
            for lab, g in (("REF stopped", kr == 1), ("REF TP", kr == 0), ("REF time", kr == 2)):
                dg = d[g]
                sd_rows.append(dict(cell=NAMES[j], blk=blk, ref_outcome=lab, N=int(g.sum()), saved=int((dg > 0.05).sum()),
                                    deeper=int((dg < -0.05).sum()), same=int((np.abs(dg) <= 0.05).sum()),
                                    sum_saved=dg[dg > 0.05].sum(), sum_deeper=dg[dg < -0.05].sum(), net=dg.sum(), avg=dg.mean() if len(dg) else np.nan))
            if CL[j]["fam"] in ("B", "C", "D", "D2"):
                na = Z["armhit"][pm, j] == 0
                res.setdefault("delta0", {})[f"{NAMES[j]}|{blk}"] = dict(n_never_armed=int(na.sum()), max_abs_delta=float(np.abs(d[na]).max()) if na.any() else 0.0)
    res["saved"] = pd.DataFrame(sd_rows)
    # assert Δ = 0 on never-armed paths for every mechanism cell
    for j in range(NC):
        if CL[j]["fam"] in ("B", "C", "D", "D2"):
            pm = ref & full & np.isfinite(V[:, j]) & (Z["armhit"][:, j] == 0)
            assert np.abs(V[pm, j] - V[pm, 0]).max() < 1e-4, NAMES[j]
    # ----- path decomposition before (REF) → after (picks), full + HOLDOUT
    wk = Z["worked"][:, 0].astype(bool)
    pdr = []
    for j in J:
        for blk, msk in (("FULL", full), ("HOLDOUT", hol)):
            m = MF[:, j] & msk
            v = V[m, j]; k = Z["k"][m, j]; best = Z["best"][m, j] - 0.09; rec = Z["rec"][m, j].astype(bool); w = wk[m]
            g = np.where(v > 0, np.where(best - v >= np.maximum(1.0, v), "LEFT_ON_TABLE", "CAPTURED"),
                         np.where(w, "GAVE_BACK", np.where(np.isin(k, (1, 3)), np.where(rec, "STOPPED_RECOVERED", "NEVER_WORKED"), "DRIFT_LOSS")))
            for gn in ("NEVER_WORKED", "STOPPED_RECOVERED", "GAVE_BACK", "DRIFT_LOSS", "LEFT_ON_TABLE", "CAPTURED"):
                mm = g == gn
                pdr.append(dict(cell=NAMES[j], blk=blk, group=gn, N=int(mm.sum()), share=mm.mean() * 100, avg=v[mm].mean() if mm.any() else np.nan,
                                eqpts=v[mm].sum() * SIZE, reach_med=np.median(Z["reach"][m, 0][mm]) if mm.any() else np.nan))
    res["pathdec"] = pd.DataFrame(pdr)
    # ----- regimes (sign granularity), on each cell's own full-span one-position set
    reg = []
    for j in J:
        for var in ("btc_d1_regime", "btc_regime"):
            x = S[var].values.astype(float)
            for blk, msk in (("DESIGN", des), ("HOLDOUT", hol)):
                for sg, lab in ((1, "+"), (-1, "−")):
                    m = MF[:, j] & msk & (np.sign(x) == sg)
                    pm = ref & msk & (np.sign(x) == sg) & np.isfinite(V[:, j])
                    reg.append(dict(cell=NAMES[j], var=var, blk=blk, sign=lab, N=int(m.sum()), avg=V[m, j].mean() if m.any() else np.nan,
                                    dpair=(V[pm, j] - V[pm, 0]).mean() if pm.any() else np.nan))
    res["regime"] = pd.DataFrame(reg)
    # ----- squeeze clustering: stops per day (REF vs picks), full
    clu = []
    for j in J:
        m = MF[:, j] & full; k = Z["k"][m, j]; xd = Z["xm"][m, j] // DAY
        st_d = pd.Series(np.isin(k, (1, 3)).astype(int)).groupby(xd).sum()
        pnl_d = pd.Series(V[m, j] * SIZE).groupby(xd).sum()
        worst = pnl_d.nsmallest(10)
        clu.append(dict(cell=NAMES[j], stops_mean_active=st_d[st_d > 0].mean(), stops_p99=np.percentile(st_d, 99), stops_max=st_d.max(),
                        worst10_avg_pct=worst.mean(), worst10_stops=st_d.reindex(worst.index).mean(),
                        worst_day_date=str(pd.to_datetime(worst.index[0] * DAY, unit="ms").date()), worst_day_pct=worst.iloc[0]))
    res["cluster"] = pd.DataFrame(clu)
    # ----- joint add
    res["joint"] = joint(S, Z, prim, J)
    # ----- EXH cross-check
    res["exh"] = exh_check(J)
    # ----- trade rows
    rows = []
    for j in J:
        m = MF[:, j]
        rows.append(pd.DataFrame(dict(cell=NAMES[j], pair=pair[m], entry_ms=T[m], exit_ms=Z["xm"][m, j], kind=Z["k"][m, j], gross=Z["r"][m, j],
                                      funding=Z["f"][m, j], slip=slip[m], net=V[m, j], net_flat20=V20[m, j], sl_pct=Z["slp"][m, j], tp_pct=Z["tpp"][m, j],
                                      worked=Z["worked"][m, 0], reach=Z["reach"][m, 0], src1m=Z["src1m"][m], half=hv[m],
                                      block=np.where(T[m] < HOL[0], "DESIGN", "HOLDOUT"), rsi1h=S.rsi1h.values[m],
                                      btc_d1_regime=S.btc_d1_regime.values[m], btc_regime=S.btc_regime.values[m])))
    TR_ = pd.concat(rows, ignore_index=True)
    TR_.to_csv(os.path.join(RP, "DIST_EXIT_2026-11-18_trades.csv.gz"), index=False, compression="gzip")
    res["MF"] = MF; res["J"] = J
    pd.to_pickle(res, os.path.join(D, "holdout_res.pkl"))
    chart(S, Z, V, MF, J, prim, T)
    report_print(res)


def joint(S, Z, prim, J):
    import joint_short_sim as JS
    Tall, _ = JS.load()
    base = Tall[Tall.sleeve.isin(["EXH", "FB", "L4"])]
    out = []; DAYS = JS.DAYS
    hm = DAYS >= HOL[0] // DAY
    def runit(lab, Tc):
        Tc = Tc[(Tc.entry_ms >= JS.D0ms) & (Tc.entry_ms < JS.D1ms)].copy(); Tc["prio"] = Tc.sleeve.map(JS.PRIO)
        Tc = Tc.sort_values(["entry_ms", "prio"]).reset_index(drop=True)
        s = JS.sim(Tc); r = JS.row(lab, Tc, s); sh = JS.stats(s["r_day"][hm])
        r["dcr_hol"] = sh["dcr"]; r["maxdd_hol"] = sh["maxdd"]; r["worst_hol"] = sh["worst_day"]
        r["dist_taken"] = int((s["taken"] & (Tc.sleeve.values == "DIST")).sum()); out.append(r)
    runit("EXH + FB + L4 (no DIST)", base)
    runit("all four · DIST Exit B (5m, as JOINT_SIM)", Tall[Tall.sleeve.isin(["EXH", "FB", "L4", "DIST"])])
    k = Z["k"]
    for j in J:
        m = prim & np.isfinite(Z["r"][:, j]) & Z["ok"] & (k[:, j] != 9)
        tk = k[m, j] != 0
        net = Z["r"][m, j] - 0.10 * (1 + tk) - 0.05 - np.where(tk, 0.05, 0.02) + Z["f"][m, j]
        dist = pd.DataFrame(dict(sleeve="DIST", pair=S.pair.astype(str).values[m], entry_ms=S.entry_ms.values[m], exit_ms=Z["xm"][m, j],
                                 net=net, nt=1 + tk, stop=np.isin(k[m, j], (1, 3)), depth=JS.depth_of(S.q5.values[m])))
        runit(f"all four · DIST {NAMES[j]} (1m)", pd.concat([base, dist], ignore_index=True))
    return pd.DataFrame(out)


def exh_check(J):
    Z = load_set("EXH"); F = DS.frames()["EXH"]
    E = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    V = net_mat(Z, F.slip.values)
    T = F.entry_ms.values; pair = F.pair.values
    out = []
    keep = ~E.dl582_block.astype(bool).values
    for j in J:
        for blk, a, b in (("DESIGN", *DES), ("HOLDOUT", *HOL)):
            base = keep & (T >= a) & (T < b)
            m = one_pos(pair, T, Z["xm"][:, j], base & np.isfinite(V[:, j]))
            mr = one_pos(pair, T, Z["xm"][:, 0], base & np.isfinite(V[:, 0]))
            pm = mr & np.isfinite(V[:, j])
            out.append(dict(cell=NAMES[j], blk=blk, N=int(m.sum()), avg=V[m, j].mean(), WR=(V[m, j] > 0).mean() * 100,
                            dpair=(V[pm, j] - V[pm, 0]).mean(), dcr=dcr(V[m, j], Z["xm"][m, j], (a, b))[0]))
    rec = E.n0.values  # as recorded (5m) for parity
    out.append(dict(cell="recorded n0 (5m, all 3,998, no slip)", blk="ALL", N=len(E), avg=rec.mean()))
    out.append(dict(cell="REF 1m (all 3,998, model slip)", blk="ALL", N=int(np.isfinite(V[:, 0]).sum()), avg=np.nanmean(V[:, 0]),
                    WR=float(np.mean(Z["src1m"][Z["ok"]]) * 100)))
    return pd.DataFrame(out)


def chart(S, Z, V, MF, J, prim, T):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = np.load(os.path.join(D, "paths_L1.npz")); pidx = P["idx"]; PP = P["p"]
    pos = {int(i): k for k, i in enumerate(pidx)}
    ref = MF[:, 0]
    tgrid = np.arange(-24, DS.NS.MAXH * 12 + 1) * 5 / 60
    fig = plt.figure(figsize=(13, 9), dpi=140)
    ax1 = fig.add_subplot(2, 2, 1); ax2 = fig.add_subplot(2, 2, 2); ax3 = fig.add_subplot(2, 1, 2)
    k0 = Z["k"][:, 0]
    groups = (("REF stopped (GAVE_BACK: worked first)", ref & (k0 == 1) & (Z["worked"][:, 0] == 1), "#eb6834"),
              ("REF stopped (never worked)", ref & (k0 == 1) & (Z["worked"][:, 0] == 0), "#6250d6"),
              ("REF TP 10 %", ref & (k0 == 0), "#1baf7a"), ("REF time exit", ref & (k0 == 2), "#52514e"))
    for lab, m, c in groups:
        ii = [pos[i] for i in np.nonzero(m)[0] if i in pos]
        if not ii:
            continue
        A = PP[ii]
        med = np.nanmedian(A, 0); q1 = np.nanpercentile(A, 25, 0); q3 = np.nanpercentile(A, 75, 0)
        ax1.plot(tgrid, med, color=c, lw=1.6, label=f"{lab} (N {len(ii)})"); ax1.fill_between(tgrid, q1, q3, color=c, alpha=0.08)
    ax1.axhline(0, color="#999", lw=0.6); ax1.axvline(0, color="#999", lw=0.6)
    atr_stop = np.median(Z["slp"][ref, 0])
    ax1.axhline(10, color="#1baf7a", ls=":", lw=1); ax1.text(30, 10.2, "Exit B TP 10 %", fontsize=7, color="#1baf7a")
    ax1.axhline(-atr_stop, color="#eb6834", ls=":", lw=1); ax1.text(30, -atr_stop - 0.7, f"Exit B stop (median {atr_stop:.1f} %)", fontsize=7, color="#eb6834")
    for j in J[1:]:
        c = CL[j]
        if c["fam"] == "A":
            ax1.axhline(c["tp"], color="#2a78d6", ls="--", lw=0.8); ax1.text(36, c["tp"] + 0.2, f"{NAMES[j]} TP", fontsize=7, color="#2a78d6")
        elif np.isfinite(c["arm"]):
            ax1.axhline(c["arm"], color="#2a78d6", ls="--", lw=0.8); ax1.text(36, c["arm"] + 0.2, f"{NAMES[j]} arm", fontsize=7, color="#2a78d6")
    ax1.set_xlim(-2, 48); ax1.set_ylim(-9, 16)
    ax1.set_title("DIST kept cohort: anchored price path (short-favourable %, median + IQR)\nby Exit B outcome — price path, not trade path", fontsize=9, loc="left")
    ax1.set_xlabel("hours from entry"); ax1.legend(fontsize=7, frameon=False, loc="upper left")
    # winners vs losers for the first pick
    jp = J[1] if len(J) > 1 else 0
    for lab, m, c in (("pick winners", MF[:, jp] & (V[:, jp] > 0), "#1baf7a"), ("pick losers", MF[:, jp] & (V[:, jp] <= 0), "#eb6834")):
        ii = [pos[i] for i in np.nonzero(m)[0] if i in pos]
        if ii:
            ax2.plot(tgrid, np.nanmedian(PP[ii], 0), color=c, lw=1.6, label=f"{lab} (N {len(ii)})")
            ax2.fill_between(tgrid, np.nanpercentile(PP[ii], 25, 0), np.nanpercentile(PP[ii], 75, 0), color=c, alpha=0.08)
    ax2.axhline(0, color="#999", lw=0.6); ax2.axvline(0, color="#999", lw=0.6); ax2.set_xlim(-2, 48); ax2.set_ylim(-9, 16)
    ax2.set_title(f"{NAMES[jp]}: winners vs losers (price path)", fontsize=9, loc="left"); ax2.legend(fontsize=7, frameon=False)
    ax2.set_xlabel("hours from entry")
    # equity curves at 0.61, full span, one size
    cols = ["#52514e", "#2a78d6", "#1baf7a", "#6250d6"]
    for j, c in zip(J, cols):
        m = MF[:, j]
        s = pd.Series(V[m, j] / 100 * SIZE).groupby(Z["xm"][m, j] // DAY).sum()
        full = s.reindex(np.arange(DES[0] // DAY, HOL[1] // DAY), fill_value=0.0)
        eq = np.cumprod(1 + full.values)
        ax3.plot(pd.to_datetime(full.index * DAY, unit="ms"), eq, color=c, lw=1.5 if j == 0 else 1.2,
                 label=("Exit B (REF)" if j == 0 else NAMES[j]))
    ax3.axvline(pd.Timestamp("2025-07-01"), color="#999", ls="--", lw=0.8); ax3.text(pd.Timestamp("2025-07-05"), 1.05, "HOLDOUT →", fontsize=8)
    ax3.set_yscale("log"); ax3.set_ylabel("equity multiple (log), 0.61 × equity per fill")
    ax3.set_title("DIST alone, one size 0.61 × equity — BACKTEST 2023-07 → 2026-11 (no haircut)", fontsize=9, loc="left")
    ax3.legend(fontsize=8, frameon=False); ax3.grid(True, color="#e4e3df", lw=0.5)
    for a in (ax1, ax2, ax3):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(RP, "DIST_EXIT_2026-11-18.png")); plt.close(fig)


def report_print(res):
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 400); pd.set_option("display.max_columns", 60)
    for nm, o in res["cells"].items():
        print("\n=====", nm)
        print(pd.DataFrame({k: o[k] for k in ("DESIGN", "HOLDOUT", "FULL")}).round(3).to_string())
        print(o["halves"].round(3).to_string())
    print("\nNULL crit95", round(res["null"]["crit95"], 3), "best", res["null"]["best"])
    for p in res["picks"]:
        print(p, "t", round(res["null"]["t_obs"][p], 3), "p", res["null"]["p"][p])
    print("\nHOLDOUT all cells (descriptive)\n", res["hol_all"][["N", "tpd", "WR", "avg", "dpair", "dcr", "maxdd", "worst", "kel_d", "t_full"]].round(3).to_string())
    print("\nCONTROL\n", res["control"].round(3).to_string(), "\nRND n", res["rnd_n"], "1m", res["rnd_1m"])
    print("\nSAVED\n", res["saved"].round(2).to_string())
    print("\nDELTA0", res.get("delta0"))
    print("\nPATHDEC\n", res["pathdec"].round(2).to_string())
    print("\nREGIME\n", res["regime"].round(3).to_string())
    print("\nCLUSTER\n", res["cluster"].round(2).to_string())
    print("\nJOINT\n", res["joint"].round(3).to_string())
    print("\nEXH\n", res["exh"].round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--phase", required=True); A = ap.parse_args()
    phase_design() if A.phase == "design" else phase_holdout()
