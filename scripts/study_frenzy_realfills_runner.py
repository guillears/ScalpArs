#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — PART B: runner vs rest (and winner vs loser) on the REAL fills, every variable, LODO-validated.

Frozen before any outcome was read here:
  OUTCOMES (each fill's own tick path, gross % vs the fill): run5 = MFE ≥ +5 before the first −3 within 12 h (also run3 / run8);
           mfe = that MFE (continuous, capped at 30); win = today's live exit (walked, net) > 0.
  FEATURES every numeric entry_* stamp (post-fill execution stamps entry_slippage_pct / entry_order_lag_s / entry_fill_vs_book_pct /
           entry_book_* flagged EXEC and kept apart), every x_* derived and j_* order-book journal feature of the W/L-screen pool, the h_* / b_*
           kline-rebuilt multi-day features of the runner-ID cohort, sleeve dummies (pooled). Coverage table first; unscored fills are in
           neither zone nor rest.
  SCREEN   1D: sign / median / low tercile / high tercile (zone vs rest) on variables with ≥ 50 % coverage in the group and ≥ 4 distinct values;
           2D: every pair of variables with ≥ 70 % coverage × 4 median quadrants. Statistic: two-proportion z on the binary outcome
           (runner-rich direction = the sign of z). Hits: |z| ≥ 3 (1D) / ≥ 3.5 (2D).
  MODEL    L1-logistic (ISTA, standardised, median-imputed, λ ∈ {0.02, 0.05}) on ≥ 70 %-covered features; depth-2 stump tree = the best 2D cell.
  LODO     for every UTC day: cuts AND the rule chosen on the other days (top |z| 1D cell, top |z| 2D cell, model), applied to the held-out day.
           Concatenated held-out predictions → OOS zone-vs-rest runner rate, z, and AUC (model).
  NULL     day-block shuffle (days permuted as blocks; the outcome vector is re-laid on the fills in time order), 200 draws: in-sample max |z|,
           hit count, and the WHOLE LODO procedure's OOS z.
Groups: FRENZY_LONG, FRENZY_LITE, FRENZY_WILLY, POOLED (88 fills). Usage: venv/bin/python scripts/study_frenzy_realfills_runner.py
"""
import os, sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
EXEC = ("entry_slippage_pct", "entry_order_lag_s", "entry_fill_vs_book_pct", "entry_book_drift_pct", "entry_book_spread_pct")
NNULL = int(os.environ.get("NNULL", "200"))


def build():
    D = pd.read_pickle(f"{RC.RF}/real.pkl"); PP = pd.read_pickle(f"{RC.RF}/real_paths.pkl")
    pool = pd.read_pickle(f"{RC.SCR}/pool.pkl")
    pool["key"] = pool.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + pool.pair + "|" + pool.direction
    coh = pd.read_pickle(f"{RC.SCR}/rid/cohort.pkl"); coh = coh[coh.blk != "BT"].copy()
    coh["key"] = coh.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + coh.pair + "|" + coh.direction
    hb = [c for c in coh.columns if c.startswith(("h_", "b_"))]
    X = D[["key", "sleeve", "pair", "day", "te"]].merge(pool[["key"] + [c for c in pool.columns if c.startswith(("entry_", "x_", "j_"))]],
                                                         on="key", how="left").merge(coh[["key"] + hb], on="key", how="left")
    o = []
    for _, r in D.iterrows():
        P = PP[r.key]
        m3, s3, _ = RC.mfe_before(P, 720, 3.0)
        o.append(dict(key=r.key, mfe=min(m3, 30.0), run3=m3 >= 3, run5=m3 >= 5, run8=m3 >= 8, win=RC.today(P, r)[0] > 0,
                      today=RC.today(P, r)[0]))
    X = X.merge(pd.DataFrame(o), on="key")
    assert len(X) == len(D)
    feats = []
    for c in X.columns:
        if not c.startswith(("entry_", "x_", "j_", "h_", "b_")):
            continue
        v = pd.to_numeric(X[c], errors="coerce")
        if v.notna().sum() >= 10 and v.nunique() >= 3:
            X[c] = v; feats.append(c)
    for s in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"):
        X["sl_" + s[7:].lower()] = (X.sleeve == s).astype(float)
    return X, feats


def zprop(a, b, na, nb):
    p = (a + b) / np.maximum(na + nb, 1)
    se = np.sqrt(np.maximum(p * (1 - p) * (1 / np.maximum(na, 1) + 1 / np.maximum(nb, 1)), 1e-12))
    return (a / np.maximum(na, 1) - b / np.maximum(nb, 1)) / se


def cells_1d(Xf, tr, feats, cov_min=0.5):
    """→ list of (name, fn(Xf)->bool zone (nan-aware via scored mask)) with cuts from the training rows."""
    out = []
    for f in feats:
        v = Xf[f].values; vt = v[tr]; ok = np.isfinite(vt)
        if ok.mean() < cov_min or len(np.unique(vt[ok])) < 4:
            continue
        q33, q50, q67 = np.nanpercentile(vt, [33.3, 50, 66.7])
        if np.nanmin(vt) < 0 < np.nanmax(vt):
            out.append((f"{f} > 0", f, 0.0, ">"))
        out.append((f"{f} > med {q50:.4g}", f, q50, ">"))
        out.append((f"{f} <= T1 {q33:.4g}", f, q33, "<="))
        out.append((f"{f} > T2 {q67:.4g}", f, q67, ">"))
    return out


def zone_mat(Xf, cl):
    Z = np.zeros((len(Xf), len(cl)), bool); S = np.zeros((len(Xf), len(cl)), bool)
    for j, (_, f, cut, op) in enumerate(cl):
        v = Xf[f].values; ok = np.isfinite(v)
        S[:, j] = ok; Z[:, j] = ok & ((v > cut) if op == ">" else (v <= cut))
    return Z, S


def cells_2d(Xf, tr, feats, cov_min=0.7):
    fs = [f for f in feats if np.isfinite(Xf[f].values[tr]).mean() >= cov_min and len(np.unique(Xf[f].values[tr][np.isfinite(Xf[f].values[tr])])) >= 4]
    med = {f: np.nanmedian(Xf[f].values[tr]) for f in fs}
    A = {f: Xf[f].values > med[f] for f in fs}; OK = {f: np.isfinite(Xf[f].values) for f in fs}
    names, Zs, Ss = [], [], []
    for a, b in itertools.combinations(fs, 2):
        s = OK[a] & OK[b]
        for qa, qb in ((1, 1), (1, 0), (0, 1), (0, 0)):
            z = s & (A[a] == bool(qa)) & (A[b] == bool(qb))
            names.append(f"{a} {'>' if qa else '<='} {med[a]:.4g} & {b} {'>' if qb else '<='} {med[b]:.4g}")
            Zs.append(z); Ss.append(s)
    if not Zs:
        return [], np.zeros((len(Xf), 0), bool), np.zeros((len(Xf), 0), bool)
    return names, np.array(Zs).T, np.array(Ss).T


def screen(Z, S, y, rows):
    Zr, Sr, yr = Z[rows], S[rows], y[rows].astype(float)
    nz = Zr.sum(0); ns = Sr.sum(0); az = (Zr * yr[:, None]).sum(0); as_ = (Sr * yr[:, None]).sum(0)
    nr = ns - nz; ar = as_ - az
    z = zprop(az, ar, nz, nr)
    z[(nz < 4) | (nr < 4)] = 0.0
    return z, nz


def l1logit(Xs, y, lam, it=400):
    n, p = Xs.shape; w = np.zeros(p); b = np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)); L = 0.25 * (np.linalg.norm(Xs, 2) ** 2) / n + 1e-6
    for _ in range(it):
        pr = 1 / (1 + np.exp(-(Xs @ w + b)))
        g = Xs.T @ (pr - y) / n; gb = (pr - y).mean()
        w = w - g / L; w = np.sign(w) * np.maximum(np.abs(w) - lam / L, 0); b -= gb / L
    return w, b


def auc(s, y):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def lodo(X, feats, y, days, do2d=True):
    """→ dict of OOS predictions (1D zone, 2D zone, model score) + the chosen rules."""
    n = len(X); p1 = np.full(n, np.nan); p2 = np.full(n, np.nan); pm = np.full(n, np.nan); ch1, ch2 = {}, {}
    mf = [f for f in feats if np.isfinite(X[f].values).mean() >= 0.7]
    for d in np.unique(days):
        tr = days != d; te = ~tr
        cl = cells_1d(X, tr, feats)
        Z, S = zone_mat(X, cl); z, _ = screen(Z, S, y, tr)
        j = int(np.argmax(np.abs(z))); sgn = np.sign(z[j]) or 1.0
        # predicted runner-rich side of the chosen cell: zone if z > 0, rest if z < 0 (unscored → NaN)
        p1[te] = np.where(S[te, j], (Z[te, j] if sgn > 0 else ~Z[te, j]).astype(float), np.nan); ch1[d] = (cl[j][0], round(float(z[j]), 2))
        if do2d:
            nm, Z2, S2 = cells_2d(X, tr, feats)
            if len(nm):
                z2, _ = screen(Z2, S2, y, tr)
                j2 = int(np.argmax(np.abs(z2))); sg2 = np.sign(z2[j2]) or 1.0
                p2[te] = np.where(S2[te, j2], (Z2[te, j2] if sg2 > 0 else ~Z2[te, j2]).astype(float), np.nan)
                ch2[d] = (nm[j2], round(float(z2[j2]), 2))
        if mf:
            M = X[mf].values.astype(float); medv = np.nanmedian(M[tr], 0); M = np.where(np.isfinite(M), M, medv)
            mu, sd = M[tr].mean(0), M[tr].std(0) + 1e-9; Ms = (M - mu) / sd
            w, b = l1logit(Ms[tr], y[tr].astype(float), 0.05)
            pm[te] = Ms[te] @ w + b
    return p1, p2, pm, ch1, ch2


def oos_stat(p, y):
    ok = np.isfinite(p)
    a, b = y[ok & (p == 1)].sum(), y[ok & (p == 0)].sum(); na, nb = (ok & (p == 1)).sum(), (ok & (p == 0)).sum()
    return float(zprop(np.array([a]), np.array([b]), np.array([na]), np.array([nb]))[0]), int(na), int(nb), \
        (a / na if na else np.nan), (b / nb if nb else np.nan)


def block_shuffle(y, days, rng):
    ud = np.unique(days); perm = rng.permutation(ud)
    blocks = [y[days == d] for d in perm]
    return np.concatenate(blocks)


if __name__ == "__main__":
    X, feats = build()
    X = X.sort_values("te").reset_index(drop=True)
    X.to_pickle(f"{RC.RF}/runner_X.pkl")
    # coverage
    cov = []
    for f in feats:
        row = dict(feature=f, exec_stamp=f in EXEC)
        for g in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"):
            row[g] = float(X.loc[X.sleeve == g, f].notna().mean())
        row["POOLED"] = float(X[f].notna().mean())
        cov.append(row)
    pd.DataFrame(cov).to_csv(f"{OUT}_B_coverage.csv", index=False)
    rng = np.random.default_rng(5)
    res, top, nullrows, picks = [], [], [], []
    feats_ok = [f for f in feats if f not in EXEC]
    for grp in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY", "POOLED"):
        G = X if grp == "POOLED" else X[X.sleeve == grp].reset_index(drop=True)
        fg = feats_ok + (["sl_long", "sl_lite", "sl_willy"] if grp == "POOLED" else [])
        days = G.day.values
        for oc in ("run5", "run3", "run8", "win"):
            y = G[oc].values.astype(bool)
            allr = np.ones(len(G), bool)
            cl = cells_1d(G, allr, fg); Z, S = zone_mat(G, cl); z1, n1 = screen(Z, S, y, allr)
            nm2, Z2, S2 = cells_2d(G, allr, fg); z2, n2 = screen(Z2, S2, y, allr)
            if oc == "run5":
                for j in np.argsort(-np.abs(z1))[:15]:
                    zz = Z[:, j] & S[:, j]
                    pairs = G.pair[zz].value_counts()
                    top.append(dict(group=grp, outcome=oc, layer="1D", cell=cl[j][0], z=z1[j], n_zone=int(n1[j]), zone_rate=y[zz].mean(),
                                    rest_rate=y[S[:, j] & ~Z[:, j]].mean(), zone_days=G.day[zz].nunique(),
                                    top_pair=pairs.index[0] if len(pairs) else "", top_pair_share=float(pairs.iloc[0] / max(1, zz.sum())) if len(pairs) else np.nan))
                for j in np.argsort(-np.abs(z2))[:15]:
                    zz = Z2[:, j]
                    pairs = G.pair[zz].value_counts()
                    top.append(dict(group=grp, outcome=oc, layer="2D", cell=nm2[j], z=z2[j], n_zone=int(n2[j]), zone_rate=y[zz].mean(),
                                    rest_rate=y[S2[:, j] & ~zz].mean(), zone_days=G.day[zz].nunique(),
                                    top_pair=pairs.index[0] if len(pairs) else "", top_pair_share=float(pairs.iloc[0] / max(1, zz.sum())) if len(pairs) else np.nan))
            p1, p2, pm, ch1, ch2 = lodo(G, fg, y, days, do2d=True)
            s1 = oos_stat(p1, y); s2 = oos_stat(p2, y); am = auc(pm[np.isfinite(pm)], y[np.isfinite(pm)])
            if oc == "run5":
                for d in ch1:
                    picks.append(dict(group=grp, held_out=d, rule_1d=ch1[d][0], z_1d_train=ch1[d][1], rule_2d=ch2.get(d, ("", 0))[0],
                                      z_2d_train=ch2.get(d, ("", 0))[1]))
            # null (in-sample max |z|, hits, and the full LODO OOS z) — run5 and win only (cost)
            nul = dict(max1=[], max2=[], hit1=[], hit2=[], o1=[], o2=[], am=[])
            if oc in ("run5", "win"):
                for k in range(NNULL):
                    ys = block_shuffle(y, days, rng)
                    a1, _ = screen(Z, S, ys, allr); a2, _ = screen(Z2, S2, ys, allr)
                    nul["max1"].append(np.abs(a1).max()); nul["max2"].append(np.abs(a2).max() if len(a2) else 0)
                    nul["hit1"].append((np.abs(a1) >= 3).sum()); nul["hit2"].append((np.abs(a2) >= 3.5).sum())
                    if k < int(os.environ.get("NNULL_LODO", "60")):
                        q1, q2, qm, _, _ = lodo(G, fg, ys, days, do2d=True)
                        nul["o1"].append(oos_stat(q1, ys)[0]); nul["o2"].append(oos_stat(q2, ys)[0])
                        nul["am"].append(auc(qm[np.isfinite(qm)], ys[np.isfinite(qm)]))
            base = y.mean()
            r = dict(group=grp, outcome=oc, N=len(G), days=len(np.unique(days)), base_rate=base,
                     tests_1d=len(cl), max_abs_z_1d=float(np.abs(z1).max()), hits_1d=int((np.abs(z1) >= 3).sum()),
                     tests_2d=len(nm2), max_abs_z_2d=float(np.abs(z2).max()) if len(z2) else np.nan, hits_2d=int((np.abs(z2) >= 3.5).sum()),
                     lodo1_z=s1[0], lodo1_n_zone=s1[1], lodo1_zone_rate=s1[3], lodo1_rest_rate=s1[4],
                     lodo2_z=s2[0], lodo2_n_zone=s2[1], lodo2_zone_rate=s2[3], lodo2_rest_rate=s2[4], lodo_model_auc=am)
            if nul["max1"]:
                r.update(null_max1_p95=np.percentile(nul["max1"], 95), p_max1=float(np.mean(np.array(nul["max1"]) >= r["max_abs_z_1d"])),
                         null_hits1_mean=np.mean(nul["hit1"]), p_hits1=float(np.mean(np.array(nul["hit1"]) >= r["hits_1d"])),
                         null_max2_p95=np.percentile(nul["max2"], 95), p_max2=float(np.mean(np.array(nul["max2"]) >= r["max_abs_z_2d"])),
                         null_hits2_mean=np.mean(nul["hit2"]), p_hits2=float(np.mean(np.array(nul["hit2"]) >= r["hits_2d"])),
                         p_lodo1=float(np.mean(np.array(nul["o1"]) >= s1[0])), p_lodo2=float(np.mean(np.array(nul["o2"]) >= s2[0])),
                         p_lodo_auc=float(np.nanmean(np.array(nul["am"]) >= am)), null_lodo_auc_p95=float(np.nanpercentile(nul["am"], 95)))
            res.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(res).to_csv(f"{OUT}_B_screen_lodo.csv", index=False)
    pd.DataFrame(top).to_csv(f"{OUT}_B_screen_top.csv", index=False)
    pd.DataFrame(picks).to_csv(f"{OUT}_B_lodo_picks.csv", index=False)
    X[["key", "sleeve", "pair", "day", "mfe", "run3", "run5", "run8", "win", "today"]].to_csv(f"{OUT}_B_outcomes.csv", index=False)
    print("done")
