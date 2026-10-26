#!/usr/bin/env python3
"""ML impulse study — STEP 1: does the market-impulse score / conjunction / spike-vs-trend features predict BTC's and the alt
market's forward 15 min? Walk-forward strictly forward, 200x day-circular-shift null. Pre-registration: reports/ML_IMPULSE_PREREG.txt.

    venv/bin/python -W ignore scripts/study_ml_impulse_step1.py [--shifts 200] > reports/ML_IMPULSE_step1.txt
Writes reports/ML_IMPULSE_step1.csv (one row per item x target) and reports/ML_IMPULSE_dose.csv (score dose-response).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_impulse_features import OUT5, CONT, LEGS, REP  # noqa: E402
CONT = list(CONT)  # noqa: E402

ITEMS = CONT + LEGS + ["SCORE", "CONJ"]
BIN = set(LEGS + ["CONJ"])
TARGETS = ["Y_btc", "Y_alt"]
BUCK = [(-1, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 9)]


def rank(a):
    return pd.Series(a).rank(method="average").values


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def auc_r(r, pos):
    n1 = pos.sum(); n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def bucket(s):
    out = np.full(len(s), -1)
    for i, (a, b) in enumerate(BUCK):
        out[(s >= a) & (s <= b)] = i
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shifts", type=int, default=200); a = ap.parse_args()
    G = pd.read_csv(OUT5, index_col=0)
    print(f"grid rows {len(G)} · coverage (share finite):")
    print(G[ITEMS + TARGETS].notna().mean().round(3).to_frame("cov").T.to_string())
    G = G.dropna(subset=ITEMS + TARGETS)
    t = pd.to_datetime(G.index, unit="ms")
    month = t.to_period("M").astype(str).values; year = t.year.values
    tests = sorted(m for m in set(month) if m >= "2025-04")
    mi = {m: np.flatnonzero(month == m) for m in tests}
    tri = {m: np.flatnonzero(month < m) for m in tests}
    n = len(G); nd = n // 288
    rng = np.random.default_rng(20261015)
    shifts = [0] + list(rng.integers(7, nd - 7, size=a.shifts) * 288)
    print(f"\ncommon-mask rows {n} · test months {tests[0]}..{tests[-1]} · {a.shifts} null shifts\n")
    X = {f: G[f].values.astype(float) for f in ITEMS}
    rtr = {(f, m): rank(X[f][tri[m]]) for f in ITEMS for m in tests}
    rte = {(f, m): rank(X[f][mi[m]]) for f in ITEMS for m in tests}
    yr_te = {m: int(m[:4]) for m in tests}
    QQ = {(f, m): np.quantile(X[f][tri[m]], [0.2, 0.8]) for f in CONT for m in tests}
    B1 = {(f, m): X[f][tri[m]] == 1 for f in BIN for m in tests}
    rows, dose = [], []
    for tg in TARGETS:
        y0 = G[tg].values.astype(float)
        null_auc = {f: [] for f in ITEMS}; null_spr = {f: [] for f in ITEMS}
        obs = {}
        for si, sh in enumerate(shifts):
            y = np.roll(y0, sh) if sh else y0
            acc = {f: dict(r=[], pos=[], spr_num={2025: 0.0, 2026: 0.0}, spr_n={2025: 0, 2026: 0}, signs={2025: [], 2026: []},
                           aucs=[]) for f in ITEMS}
            for m in tests:
                tr, te = tri[m], mi[m]
                ry = rank(y[tr]); yte = y[te]; up = yte > 0
                for f in ITEMS:
                    xtr, xte = X[f][tr], X[f][te]
                    if f in BIN:
                        ytr_ = y[tr]; b1 = B1[f, m]
                        d = ytr_[b1].mean() - ytr_[~b1].mean()
                        sg = 1.0 if d >= 0 else -1.0
                        hi, lo = xte == 1, xte == 0
                    elif f == "SCORE":
                        sg = 1.0 if corr(rtr[f, m], ry) >= 0 else -1.0
                        hi, lo = xte >= 7, xte <= 2
                    else:
                        sg = 1.0 if corr(rtr[f, m], ry) >= 0 else -1.0
                        q = QQ[f, m]
                        hi, lo = xte >= q[1], xte <= q[0]
                    if sg < 0:
                        hi, lo = lo, hi
                    sg_ = sg
                    rs = rte[f, m] if sg_ > 0 else (len(te) + 1 - rte[f, m])
                    au = auc_r(rs, up)
                    A = acc[f]; A["aucs"].append((len(te), au))
                    yy = yr_te[m]
                    if hi.any() and lo.any():
                        # pooled oriented spread per year: accumulate month spread weighted by month size
                        A["spr_num"][yy] += (yte[hi].mean() - yte[lo].mean()) * len(te); A["spr_n"][yy] += len(te)
                    A["signs"][yy].append(sg_)
                    if si == 0 and f == "SCORE":
                        b = bucket(xte)
                        for k in range(len(BUCK)):
                            mk = b == k
                            dose.append(dict(target=tg, month=m, year=yy, bucket=k, n=int(mk.sum()),
                                             sum_y=float(yte[mk].sum()), n_up=int((yte[mk] > 0).sum()),
                                             n_up10=int((yte[mk] >= 0.10).sum()), n_dn10=int((yte[mk] <= -0.10).sum())))
            for f in ITEMS:
                A = acc[f]
                w = np.array([k for k, _ in A["aucs"]]); v = np.array([x for _, x in A["aucs"]])
                pooled_auc = float(np.nansum(w * v) / w[np.isfinite(v)].sum())
                spr = {yy: (A["spr_num"][yy] / A["spr_n"][yy] * 100 if A["spr_n"][yy] else np.nan) for yy in (2025, 2026)}
                pooled_spr = (A["spr_num"][2025] + A["spr_num"][2026]) / max(A["spr_n"][2025] + A["spr_n"][2026], 1) * 100
                if si == 0:
                    obs[f] = dict(auc=pooled_auc, spr25=spr[2025], spr26=spr[2026], spr=pooled_spr,
                                  sign25=float(np.mean(A["signs"][2025])), sign26=float(np.mean(A["signs"][2026])))
                else:
                    null_auc[f].append(pooled_auc); null_spr[f].append(abs(pooled_spr))
            if si % 50 == 0:
                print(f"  {tg} shift {si}/{len(shifts) - 1}", file=sys.stderr, flush=True)
        D = pd.DataFrame(dose); D = D[D.target == tg]
        for f in ITEMS:
            o = obs[f]
            na, ns = np.array(null_auc[f]), np.array(null_spr[f])
            same_or = np.sign(o["sign25"]) == np.sign(o["sign26"]) and abs(o["sign25"]) > 0
            ga = (o["auc"] > np.quantile(na, 0.99)) if f not in BIN else (abs(o["spr"]) > np.quantile(ns, 0.99))
            gb = same_or and o["spr25"] >= 2.0 and o["spr26"] >= 2.0
            r = dict(target=tg, item=f, kind=("binary" if f in BIN else ("score" if f == "SCORE" else "cont")),
                     orient25=o["sign25"], orient26=o["sign26"], auc=o["auc"], null_auc99=np.quantile(na, 0.99),
                     spr_bps=o["spr"], null_absspr99=np.quantile(ns, 0.99), spr25_bps=o["spr25"], spr26_bps=o["spr26"],
                     g_a=bool(ga), g_b=bool(gb))
            if f == "SCORE":
                gc = True
                for yy in (2025, 2026):
                    d = D[D.year == yy].groupby("bucket")[["n", "sum_y"]].sum()
                    mu = d.sum_y / d.n
                    rho = corr(rank(mu.index.values.astype(float)), rank(mu.values))
                    gap = (mu.iloc[-1] - mu.iloc[0]) * 100
                    orient = np.sign(o["sign25"])
                    r[f"dose_rho_{yy}"] = rho; r[f"dose_gap_{yy}_bps"] = gap
                    gc &= (orient * rho >= 0.8) and (orient * gap >= 2.0)
                r["g_c"] = bool(gc)
            r["STEP1_PASS"] = bool(ga and gb and r.get("g_c", True))
            rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(REP, "ML_IMPULSE_step1.csv"), index=False)
    pd.DataFrame(dose).to_csv(os.path.join(REP, "ML_IMPULSE_dose.csv"), index=False)
    pd.set_option("display.width", 250)
    for tg in TARGETS:
        print(f"\n## STEP 1 — target {tg} (OOS, walk-forward; spread = oriented top − bottom, bps)")
        print(T[T.target == tg].drop(columns="target").to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    D = pd.DataFrame(dose)
    print("\n## SCORE dose-response (OOS test months; mean Y in %, P(up), P(>=+0.10), P(<=-0.10))")
    for tg in TARGETS:
        for yy in (2025, 2026):
            d = D[(D.target == tg) & (D.year == yy)].groupby("bucket")[["n", "sum_y", "n_up", "n_up10", "n_dn10"]].sum()
            d["mean_bps"] = d.sum_y / d.n * 100
            for c in ("n_up", "n_up10", "n_dn10"):
                d["P_" + c[2:]] = d[c] / d.n
            d.index = ["<=2", "3", "4", "5", "6", ">=7"][:len(d)]
            print(f"\n{tg} {yy}\n" + d[["n", "mean_bps", "P_up", "P_up10", "P_dn10"]].to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nSTEP1 passers:", T[T.STEP1_PASS][["target", "item"]].values.tolist() or "none")


if __name__ == "__main__":
    main()
