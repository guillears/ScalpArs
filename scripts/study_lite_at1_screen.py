#!/usr/bin/env python3
"""LITE AT-+1 study — step 3 (prereg + addenda A1 / B): tiers, unconditional at-touch policies, SIDE-FIRST (C zone inside / outside)
exhaustive screens at the +1 touch (T1 vs T2+T3; every variable x every at-touch variable) and at entry (T0 vs rest → SKIP; T1 vs rest →
TP +1), each with its own full-search day-block null, walk-forward 2025 ↔ 2026, classifiers (numpy L2 logistic + depth-3 tree;
sklearn is not installed), coverage table, master / live before-after. Primary verdict set = TICK-priced fills; ALL = refute-only line.
Input reports/lite_at1/rows.csv + reports/lite_wlsweep/matrix.csv (+ the live export for live entry stamps).
Usage: venv/bin/python -W ignore scripts/study_lite_at1_screen.py [--R 300]
"""
import os, sys, json, time, argparse, itertools
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "lite_at1")
LIVE_CSV = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_21-18-32.csv")
POL = ["BANK", "L0", "L0.3", "L-1", "D0.5", "D0.8", "D1"]
UNCOND = ["P1"] + POL
TOUCH = ["min_to_touch", "mae_pre", "cl_T", "pb05", "s08", "r3", "r5", "nbars_1m", "tb_k", "tb5", "d_tb5", "tb5m", "vr_k", "vr5",
         "ngreen_frac", "dvwap", "newhigh", "uwick_k", "dhi_epi", "btc_r_since", "eth_r_since", "sol_r_since", "btc_r5", "btc_r15",
         "eth_r15", "sol_r15", "btc_slope1h_t", "br_up5_t", "agg_tbr5_t", "mvol5_vs_entry5", "mvol5_vs_pre60", "pvol1m_vs60",
         "pvol5m_vs60"] + [f"{p}{x}" for p in ("t1_", "t5_") for x in ("rsi", "rsi_dir", "adx", "adx_dir", "gap58", "gap813", "gap520",
                                                                      "gap58_w", "gap813_w", "gap520_w", "dema13")]
NAMED_VOL = ["entry_pair_volume_24h_usd", "entry_pair_volume_ratio", "entry_vol_mcap_ratio", "entry_liquidity_cap_notional",
             "entry_global_volume_ratio", "entry_frenzy_gvol", "pvol1m_vs60", "pvol5m_vs60", "vr_k", "vr5", "mvol5_vs_entry5",
             "mvol5_vs_pre60", "sb_vmult", "v3mult", "vsurge"]
Q1 = (0.2, 0.33, 0.5, 0.67, 0.8)
Q2 = (0.33, 0.67)
MINZ = 15
YEARS = ("2025", "2026")
EXCL = {"fid", "block", "era", "era2", "year", "pair", "te", "E", "day", "month", "tick", "src", "peak_all", "rm_base_why", "rm_base_pct",
        "tick_only", "yblk", "cls", "y", "win", "entry_price", "entry_strategy", "entry_order_type", "entry_frenzy_spike_at",
        "entry_btc_regime_started_at", "entry_surge_trigger_at", "entry_frenzy_vwap", "entry_fee"}


# ───────────────────────── stats helpers ─────────────────────────
def dayblock_perm(te, day, R, rng):
    order = np.argsort(te, kind="stable")
    d = np.asarray(day)[order]
    starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    blocks = np.split(np.arange(len(d)), starts[1:])
    out = np.empty((len(d), R), np.int64)
    for r in range(R):
        perm = rng.permutation(len(blocks))
        seq = np.concatenate([blocks[k] for k in perm])
        out[order, r] = order[seq]
    return out


def day_boot(v, day, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(dict(v=v, d=day)).groupby("d").v.agg(["sum", "count"])
    s, c = df["sum"].values, df["count"].values
    ix = rng.integers(0, len(s), (n, len(s)))
    m = s[ix].sum(1) / c[ix].sum(1)
    return np.percentile(m, [2.5, 97.5]), float((m <= 0).mean())


def tstat(S1, S2, N):
    m = S1 / N
    var = np.maximum((S2 - S1 * S1 / N) / (N - 1), 1e-12)
    return m / np.sqrt(var / N)


# ───────────────────────── variables ─────────────────────────
def load():
    R = pd.read_csv(os.path.join(OUT, "rows.csv"), low_memory=False)
    R = R.rename(columns={"br_up5": "br_up5_t", "agg_tbr5": "agg_tbr5_t"})
    M = pd.read_csv(os.path.join(REP, "lite_wlsweep", "matrix.csv"), low_memory=False)
    Vt = pd.read_csv(os.path.join(REP, "lite_wlsweep", "variables.csv"))
    L = pd.read_csv(LIVE_CSV, low_memory=False)
    L = L[(L.entry_strategy == "FRENZY_LITE") & (L.status == "CLOSED")].copy()
    L["fid"] = [f"LIVE|{p}|{int(pd.Timestamp(o, tz='UTC').value // 1_000_000)}" for p, o in zip(L.pair, L.opened_at)]
    base = [c for c in R.columns]
    mcols = [c for c in M.columns if c not in base or c == "fid"]
    X = pd.concat([M[mcols], L[[c for c in L.columns if c in mcols]]], ignore_index=True)
    R = R.merge(X, on="fid", how="left")
    # universe: variables.csv + every other entry_* in matrix
    uni = list(Vt["var"]) + [c for c in M.columns if c.startswith("entry_") and c not in set(Vt["var"]) and c not in EXCL]
    uni = [c for c in dict.fromkeys(uni) if c not in EXCL and c not in TOUCH]
    ent, src = [], {}
    for c in uni:
        if c not in R:
            src[c] = "absent"; continue
        x = pd.to_numeric(R[c], errors="coerce")
        if x.notna().mean() >= 0.5 * R[c].notna().mean() and x.notna().any():
            R[c] = x; ent.append(c); src[c] = "num"
        else:      # string category → one-hot per level (≥ 5 %)
            vc = R[c].astype(str).value_counts(normalize=True)
            for lv in vc.index[(vc >= 0.05) & (vc.index != "nan")]:
                nm = f"{c}=={lv}"; R[nm] = (R[c].astype(str) == lv).astype(float); R.loc[R[c].isna(), nm] = np.nan
                ent.append(nm); src[nm] = "onehot"
            src[c] = "onehot-parent"
    # derived directions
    der = {"d_pair_rsi_dir": ("entry_rsi", "entry_rsi_prev"), "d_btc_rsi5m_dir": ("entry_btc_rsi", "entry_btc_rsi_prev"),
           "d_btc_rsi30m_dir": ("entry_btc_rsi", "entry_btc_rsi_prev6"), "d_btc_rsi1h_dir": ("entry_btc_rsi_1h", "entry_btc_rsi_1h_prev"),
           "d_adx_dir": ("entry_adx", "entry_adx_prev"), "d_btc_adx_dir": ("entry_btc_adx", "entry_btc_adx_prev"),
           "d_gap520_dir": ("entry_gap_5_20_signed_pct", "entry_gap_5_20_prev_signed_pct"), "d_di_spread": ("entry_pos_di", "entry_neg_di")}
    for k, (a, b) in der.items():
        if a in R and b in R:
            R[k] = pd.to_numeric(R[a], errors="coerce") - pd.to_numeric(R[b], errors="coerce"); ent.append(k); src[k] = "derived"
    # drop constants
    ent = [c for c in ent if R[c].nunique(dropna=True) > 1]
    for c in ("touch", "acted", "tick_only"):
        R[c] = R[c].fillna(False).astype(bool)
    R["year"] = R.year.astype(str)
    R["side"] = np.where(R.czone == 1, "IN", "OUT")
    R["d_SKIP"] = -R.C
    return R, ent, src


# ───────────────────────── chunked screen ─────────────────────────
def halfspaces(A, cutsrc, vars_, qs):
    """→ H (n_half x nA bool), meta list (var, q, side, cut), index {(var, q, side): row}."""
    H, meta, idx = [], [], {}
    for v in vars_:
        x = A[v].values.astype(float); s = cutsrc[v].values.astype(float); s = s[np.isfinite(s)]
        for q in qs:
            c = float(np.quantile(s, q)) if len(s) else np.nan
            with np.errstate(invalid="ignore"):
                for sd, m in (("le", x <= c), ("gt", x > c)):
                    idx[(v, q, sd)] = len(H); H.append(m); meta.append((v, q, sd, c))
    return np.array(H, bool).reshape(len(H), len(A)), meta, idx


def specs(vars1, pairs, idx):
    h1, h2 = [], []
    for v in vars1:
        for q in Q1:
            for sd in ("le", "gt"):
                h1.append(idx[(v, q, sd)]); h2.append(-1)
    for a, b in pairs:
        for qa in Q2:
            for qb in Q2:
                for sa in ("le", "gt"):
                    for sb in ("le", "gt"):
                        h1.append(idx[(a, qa, sa)]); h2.append(idx[(b, qb, sb)])
    return np.array(h1), np.array(h2)


def run_screen(A, Ny, D, vars1, pairs, R, rng, chunk=8000, cutsrc=None, years=YEARS):
    """A = the side's scored rows (acted fills for touch; all fills for entry), D (nA x P). → dict(Sc, t, nz per year, nullmax, h1, h2,
    meta). Δ outside a zone = 0; N per year = the side's LITE fills that year (Ny)."""
    allv = sorted(set(vars1) | {v for p in pairs for v in p})
    H, meta, idx = halfspaces(A, A if cutsrc is None else cutsrc, allv, sorted(set(Q1) | set(Q2)))
    h1, h2 = specs(vars1, pairs, idx)
    P = D.shape[1]
    yidx = {y: np.flatnonzero(A.year.values == y) for y in years}
    Dall = {}
    for y in years:
        ix = yidx[y]
        cols = [D[ix]]
        if R:
            pm = dayblock_perm(A.te.values[ix], A.day.values[ix], R, rng)
            cols += [D[ix][pm[:, r]] for r in range(R)]
        Dy = np.stack(cols, 0)                         # (R+1, nAy, P)
        Dall[y] = np.concatenate([Dy, Dy * Dy], 2).transpose(1, 0, 2).reshape(len(ix), -1).astype(np.float32)  # nAy x (R+1)*2P
    nS = len(h1)
    Sc = np.full((nS, P), -np.inf, np.float32); T = {y: np.zeros((nS, P), np.float32) for y in years}
    NZ = {y: np.zeros(nS, np.int32) for y in years}
    nullmax = np.full(R, -np.inf)
    for c0 in range(0, nS, chunk):
        a1, a2 = h1[c0:c0 + chunk], h2[c0:c0 + chunk]
        Mk = H[a1].copy()
        two = a2 >= 0
        Mk[two] &= H[a2[two]]
        tmin = None
        for y in years:
            ix = yidx[y]
            Mf = Mk[:, ix].astype(np.float32)
            nz = Mf.sum(1)
            Z = (Mf @ Dall[y]).reshape(len(Mf), R + 1, 2, P).astype(np.float64)
            t = tstat(Z[:, :, 0, :], Z[:, :, 1, :], Ny[y])
            t[nz < MINZ] = -np.inf
            T[y][c0:c0 + len(Mf)] = t[:, 0, :]; NZ[y][c0:c0 + len(Mf)] = nz
            tmin = t if tmin is None else np.minimum(tmin, t)
        Sc[c0:c0 + len(Mk)] = tmin[:, 0, :]
        if R:
            nullmax = np.maximum(nullmax, tmin[:, 1:, :].max(axis=(0, 2)))
    return dict(Sc=Sc, T=T, NZ=NZ, nullmax=nullmax, h1=h1, h2=h2, meta=meta, yidx=yidx)


def describe(res, i, j, pol):
    m1 = res["meta"][res["h1"][i]]
    s = f"{m1[0]} {'≤' if m1[2] == 'le' else '>'} {m1[3]:.4g}"
    sp = dict(v1=m1[0], s1=m1[2], c1=m1[3], v2="", s2="", c2=np.nan)
    if res["h2"][i] >= 0:
        m2 = res["meta"][res["h2"][i]]
        s += f" & {m2[0]} {'≤' if m2[2] == 'le' else '>'} {m2[3]:.4g}"
        sp.update(v2=m2[0], s2=m2[2], c2=m2[3])
    sp.update(zone=s, policy=pol[j])
    return sp


def zone_of(A, sp):
    def h(v, s, c):
        x = A[v].values.astype(float) if v in A else np.full(len(A), np.nan)
        with np.errstate(invalid="ignore"):
            return x <= c if s == "le" else x > c
    m = h(sp["v1"], sp["s1"], sp["c1"])
    if sp["v2"]:
        m &= h(sp["v2"], sp["s2"], sp["c2"])
    return m


def top_rows(res, pol, k, Ny, extra=None):
    Sc = res["Sc"]
    flat = np.argsort(Sc, axis=None)[::-1][:k]
    rows = []
    for fi in flat:
        i, j = np.unravel_index(fi, Sc.shape)
        if not np.isfinite(Sc[i, j]):
            break
        sp = describe(res, i, j, pol)
        sp.update(S=float(Sc[i, j]), fw_p=float((res["nullmax"] >= Sc[i, j]).mean()) if len(res["nullmax"]) else np.nan)
        for y in YEARS:
            sp[f"t_{y}"] = float(res["T"][y][i, j]); sp[f"nz_{y}"] = int(res["NZ"][y][i])
        rows.append(sp)
    return rows


# ───────────────────────── classifier (numpy) ─────────────────────────
def logit_fit(X, y, lam=0.1, it=50):
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]; w = np.zeros(p + 1)
    for _ in range(it):
        pr = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (pr - y) / n + lam * np.r_[0, w[1:]]
        Hs = (Xb * (pr * (1 - pr))[:, None]).T @ Xb / n + np.diag(np.r_[1e-6, np.full(p, lam)])
        st = np.linalg.solve(Hs, g); w -= st
        if np.abs(st).max() < 1e-7:
            break
    return w


def logit_pred(w, X):
    return 1 / (1 + np.exp(-np.clip(np.c_[np.ones(len(X)), X] @ w, -30, 30)))


def tree_fit(X, y, depth=3, minleaf=20, nq=12):
    def node(ix, d):
        p = y[ix].mean() if len(ix) else 0.5
        if d == 0 or len(ix) < 2 * minleaf:
            return ("leaf", p)
        best = None; g0 = len(ix) * p * (1 - p)
        for j in range(X.shape[1]):
            x = X[ix, j]
            for c in np.unique(np.quantile(x, np.linspace(0.08, 0.92, nq))):
                l = x <= c; nl = l.sum(); nr = len(ix) - nl
                if nl < minleaf or nr < minleaf:
                    continue
                pl = y[ix][l].mean(); pr = y[ix][~l].mean()
                g = nl * pl * (1 - pl) + nr * pr * (1 - pr)
                if best is None or g < best[0]:
                    best = (g, j, c, l)
        if best is None or best[0] >= g0 - 1e-12:
            return ("leaf", p)
        _, j, c, l = best
        return ("split", j, c, node(ix[l], d - 1), node(ix[~l], d - 1))
    return node(np.arange(len(y)), depth)


def tree_pred(t, X):
    out = np.empty(len(X))
    for i in range(len(X)):
        n = t
        while n[0] == "split":
            n = n[3] if X[i, n[1]] <= n[2] else n[4]
        out[i] = n[1]
    return out


def tree_str(t, names, ind=""):
    if t[0] == "leaf":
        return f"{ind}p={t[1]:.2f}\n"
    return f"{ind}{names[t[1]]} <= {t[2]:.3g}\n" + tree_str(t[3], names, ind + "  ") + f"{ind}else\n" + tree_str(t[4], names, ind + "  ")


def auc(s, y):
    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def prep(Atr, Ate, feats):
    Xtr = Atr[feats].values.astype(float); Xte = Ate[feats].values.astype(float)
    with np.errstate(all="ignore"):
        med = np.nanmedian(Xtr, 0)
    med = np.where(np.isfinite(med), med, 0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, med); Xte = np.where(np.isfinite(Xte), Xte, med)
    lo, hi = np.percentile(Xtr, 1, 0), np.percentile(Xtr, 99, 0)
    Xtr = np.clip(Xtr, lo, hi); Xte = np.clip(Xte, lo, hi)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def classifier(S, rowsel, label, dcol, feats, setname, qname, Rn, rng, trees=None):
    out = []
    for tr, te_ in (("2025", "2026"), ("2026", "2025")):
        Atr = S[(S.year == tr) & rowsel(S)]; Ate = S[(S.year == te_) & rowsel(S)]
        if len(Atr) < 40 or len(Ate) < 40:
            continue
        Ntr = int((S.year == tr).sum()); Nte = int((S.year == te_).sum())
        ytr = label(Atr).astype(float); yte = label(Ate).astype(float)
        Xtr, Xte = prep(Atr, Ate, feats)
        for mdl in ("logit", "tree"):
            if mdl == "logit":
                w = logit_fit(Xtr, ytr); ptr, pte = logit_pred(w, Xtr), logit_pred(w, Xte)
            else:
                t = tree_fit(Xtr, ytr); ptr, pte = tree_pred(t, Xtr), tree_pred(t, Xte)
                if trees is not None:
                    trees.append(f"[{setname} {qname} train {tr}]\n" + tree_str(t, feats))
            a = auc(pte, yte)
            nn = Rn if mdl == "logit" else min(Rn, 60)
            null = []
            for r in range(nn):
                ys = rng.permutation(ytr)
                pn = logit_pred(logit_fit(Xtr, ys, it=15), Xte) if mdl == "logit" else tree_pred(tree_fit(Xtr, ys), Xte)
                null.append(auc(pn, yte))
            null = np.array(null, float)
            dtr = Atr[dcol].fillna(0).values; dte = Ate[dcol].fillna(0).values
            best = (0.0, np.inf)
            for q in np.arange(0.5, 0.96, 0.05):
                th = np.quantile(ptr, q)
                v = dtr[ptr > th].sum()
                if v > best[0]:
                    best = (v, th)
            zone = pte > best[1]
            out.append(dict(set=setname, question=qname, train=tr, test=te_, model=mdl, n_train=len(Atr), n_test=len(Ate),
                            base_rate_test=float(yte.mean()), auc_train=auc(ptr, ytr), auc_test=a, null_mean=float(np.nanmean(null)),
                            null_q95=float(np.nanquantile(null, 0.95)), null_p=float(np.nanmean(null >= a)), n_null=nn,
                            thr=best[1], train_dpf=best[0] / Ntr, test_zone_n=int(zone.sum()), test_dpf=dte[zone].sum() / Nte,
                            test_label_rate_zone=float(yte[zone].mean()) if zone.any() else np.nan))
    return out


# ───────────────────────── main ─────────────────────────
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--R", type=int, default=300); a = ap.parse_args()
    t0 = time.time()
    R0, ENT, SRC = load()
    BT = R0[(R0.block == "BT") & R0.C.notna()].copy()
    ML = R0[R0.block.isin(["MASTER", "LIVE"])].copy()
    summ = dict(R=a.R, n_entry_vars=len(ENT), n_touch_vars=len(TOUCH), excluded_unres=int(((R0.block == "BT") & R0.C.isna()).sum()))
    SETS = (("TICK", BT[BT.tick_only]), ("ALL", BT))
    # ── 1. tiers ──
    rows = []
    def tierrow(s, **kw):
        d = dict(kw, N=len(s))
        for k in range(4):
            d[f"T{k}"] = int((s.tier == k).sum())
        for k in range(4):
            d[f"T{k}_pct"] = d[f"T{k}"] / max(len(s), 1) * 100
        d["C_avg"] = s.C.mean()
        return d
    for setn, S in SETS + (("1M-only", BT[~BT.tick_only]),):
        for y in YEARS:
            for sd in ("IN", "OUT", "both"):
                s = S[(S.year == y) & ((S.side == sd) | (sd == "both"))]
                rows.append(tierrow(s, set=setn, year=y, side=sd))
    for blk in ("MASTER", "LIVE"):
        for sd in ("IN", "OUT", "both"):
            s = ML[(ML.block == blk) & ((ML.side == sd) | (sd == "both"))]
            rows.append(tierrow(s, set=blk, year="-", side=sd))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tiers.csv"), index=False)
    # touch outcome detail
    rows = []
    for setn, S in SETS:
        for y in YEARS:
            for sd in ("IN", "OUT"):
                s = S[(S.year == y) & (S.side == sd)]
                tc = s[s.touch]; ac = s[s.acted]
                rows.append(dict(set=setn, year=y, side=sd, N=len(s), touch=len(tc), touch_exit_by_T=len(tc) - len(ac), acted=len(ac),
                                 acted_T1=int((ac.tier == 1).sum()), acted_T2=int((ac.tier == 2).sum()), acted_T3=int((ac.tier == 3).sum()),
                                 acted_TP=int((ac.C_why == "TP").sum()), acted_STOP=int((ac.C_why == "STOP").sum()),
                                 acted_CAP=int((ac.C_why == "CAP").sum()), acted_C_avg=ac.C.mean(),
                                 T1_C_avg=ac.C[ac.tier == 1].mean(), T1_cost_pf=ac.C[ac.tier == 1].sum() / len(s)))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "touch_outcomes.csv"), index=False)
    # ── 2. unconditional at-touch policies (per side + both) ──
    rows = []
    for setn, S in SETS + (("1M-only", BT[~BT.tick_only]),):
        for y in YEARS:
            for sd in ("IN", "OUT", "both"):
                s = S[(S.year == y) & ((S.side == sd) | (sd == "both"))]
                if len(s) < 10:
                    continue
                for p in UNCOND:
                    v = s["d_" + p].fillna(0).values
                    ci, pneg = day_boot(v, s.day.values)
                    rows.append(dict(set=setn, year=y, side=sd, policy=p, N=len(s), d_pf=v.mean(), ci_lo=ci[0], ci_hi=ci[1], p_le0=pneg,
                                     helped=int((v > 1e-9).sum()), hurt=int((v < -1e-9).sum()), gain=v[v > 0].sum(), cost=v[v < 0].sum()))
    UNC = pd.DataFrame(rows); UNC.to_csv(os.path.join(OUT, "uncond.csv"), index=False)
    print("tiers / uncond done", f"{time.time() - t0:.0f}s", flush=True)
    # ── 3. screens ──
    rng = np.random.default_rng(2026)
    TOUCHV = [v for v in TOUCH if v in R0]
    allv_touch = ENT + TOUCHV
    scr = {}; tops = []; covrows = {}; wfrows = []; named = []
    for Q in ("TOUCH", "ENTRY"):
        for setn, S in SETS:
            for sd in ("IN", "OUT"):
                Ss = S[S.side == sd]
                Ny = {y: int((Ss.year == y).sum()) for y in YEARS}
                if Q == "TOUCH":
                    A = Ss[Ss.acted].sort_values("te").reset_index(drop=True)
                    pol = POL
                    cov = {v: float(A[v].notna().mean()) if len(A) else 0 for v in allv_touch}
                    v1 = [v for v in allv_touch if cov[v] >= 0.5 and A[v].nunique() > 1]
                    tv = [v for v in TOUCHV if v in v1]
                    ev = [v for v in v1 if v not in tv]
                    pairs = [(e, t) for e in ev for t in tv] + list(itertools.combinations(tv, 2))
                else:
                    A = Ss.sort_values("te").reset_index(drop=True)
                    pol = ["SKIP", "P1"]
                    cov = {v: float(A[v].notna().mean()) for v in ENT}
                    v1 = [v for v in ENT if cov[v] >= 0.5 and A[v].nunique() > 1]
                    pairs = list(itertools.combinations(v1, 2))
                if min(Ny.values()) < 20 or len(A) < 40:
                    continue
                D = A[["d_" + p for p in pol]].fillna(0).values
                key = f"{Q}_{setn}_{sd}"
                ts = time.time()
                res = run_screen(A, Ny, D, v1, pairs, a.R, rng)
                obs = float(res["Sc"].max())
                scr[key] = dict(n_tests=int(res["Sc"].size), n_eligible=int(np.isfinite(res["Sc"]).sum()), vars_1d=len(v1),
                                pairs=len(pairs), obs_max=obs, fw_p=float((res["nullmax"] >= obs).mean()),
                                null_q50=float(np.median(res["nullmax"])), null_q95=float(np.quantile(res["nullmax"], 0.95)),
                                N=Ny, scored={y: int((A.year == y).sum()) for y in YEARS}, secs=time.time() - ts)
                tp = top_rows(res, pol, 25, Ny)
                for r in tp:
                    r.update(key=key, question=Q, set=setn, side=sd)
                    z = zone_of(A, r)
                    for y in YEARS:
                        s = A[(A.year == y) & z]
                        r[f"dpf_{y}"] = s["d_" + r["policy"]].fillna(0).sum() / Ny[y]
                        for k in range(4):
                            r[f"T{k}_{y}"] = int((s.tier == k).sum())
                        r[f"npairs_{y}"] = s.pair.nunique()
                        r[f"top_pair_share_{y}"] = float(s.groupby("pair")["d_" + r["policy"]].sum().clip(lower=0).max() /
                                                         max(s["d_" + r["policy"]].clip(lower=0).sum(), 1e-9)) if len(s) else np.nan
                    # master / live (refute-only)
                    Mrows = ML[(ML.side == sd) & (ML.acted if Q == "TOUCH" else True)]
                    zm = zone_of(Mrows, r)
                    for blk in ("MASTER", "LIVE"):
                        mm = Mrows[(Mrows.block == blk) & zm]
                        r[f"n_{blk}"] = len(mm); r[f"sum_d_{blk}"] = mm["d_" + r["policy"]].fillna(0).sum()
                tops += tp
                # coverage: best per variable (1D or any 2D containing it)
                Sc = res["Sc"].max(1)
                vv1 = np.array([res["meta"][h][0] for h in res["h1"]]); vv2 = np.array([res["meta"][h][0] if h >= 0 else "" for h in res["h2"]])
                dfb = pd.DataFrame(dict(v1=vv1, v2=vv2, S=Sc, d1=res["h2"] < 0))
                b1 = dfb[dfb.d1].groupby("v1").S.max()
                b2 = pd.concat([dfb[~dfb.d1].groupby("v1").S.max(), dfb[~dfb.d1].groupby("v2").S.max()]).groupby(level=0).max()
                for v, c in cov.items():
                    cr = covrows.setdefault(v, dict(var=v, src=SRC.get(v, "touch" if v in TOUCH else "?")))
                    cr[f"cov_{key}"] = round(c, 3)
                    cr[f"best1D_{key}"] = float(b1.get(v, np.nan)) if v in v1 else np.nan
                    cr[f"best2D_{key}"] = float(b2.get(v, np.nan)) if v in v1 else np.nan
                    cr[f"fwp1D_{key}"] = float((res["nullmax"] >= b1.get(v, np.inf)).mean()) if v in v1 else np.nan
                for v in NAMED_VOL:
                    if v in v1:
                        i = np.flatnonzero(dfb.d1.values & (vv1 == v))
                        j = i[np.argmax(Sc[i])]
                        pj = int(np.argmax(res["Sc"][j]))
                        sp = describe(res, j, pj, pol)
                        named.append(dict(key=key, var=v, best_1D=sp["zone"], policy=sp["policy"], S=float(Sc[j]),
                                          t25=float(res["T"]["2025"][j, pj]), t26=float(res["T"]["2026"][j, pj]),
                                          fw_p=float((res["nullmax"] >= Sc[j]).mean()), best_2D_S=float(b2.get(v, np.nan))))
                    else:
                        named.append(dict(key=key, var=v, best_1D="not scored (coverage < 50 % or constant)" if v in cov or v in R0 else "absent"))
                # walk-forward (discover one year, frozen cuts, confirm the other)
                for dy, cy in (("2025", "2026"), ("2026", "2025")):
                    Ad = A[A.year == dy].reset_index(drop=True)
                    rd = run_screen(Ad, {dy: Ny[dy]}, Ad[["d_" + p for p in pol]].fillna(0).values, v1, pairs, 0, rng, years=(dy,))
                    td = rd["T"][dy].copy(); td[rd["NZ"][dy] < MINZ] = -np.inf
                    order = np.argsort(td, axis=None)[::-1][:10]
                    Ac = A[A.year == cy]
                    for rk, fi in enumerate(order):
                        i, j = np.unravel_index(fi, td.shape)
                        sp = describe(rd, i, j, pol)
                        zc = zone_of(Ac, sp)
                        dv = np.zeros(Ny[cy]); vv = Ac["d_" + pol[j]].fillna(0).values[zc]; dv[:len(vv)] = vv
                        wfrows.append(dict(key=key, discover=dy, confirm=cy, rank=rk + 1, policy=pol[j], zone=sp["zone"],
                                           t_disc=float(td[i, j]), nz_disc=int(rd["NZ"][dy][i]),
                                           dpf_disc=float(Ad["d_" + pol[j]].fillna(0).values[zone_of(Ad, sp)].sum() / Ny[dy]),
                                           nz_conf=int(zc.sum()), dpf_conf=dv.sum() / Ny[cy], t_conf=float(tstat(dv.sum(), (dv ** 2).sum(), Ny[cy]))))
                print(key, json.dumps({k: v for k, v in scr[key].items() if k in ("n_tests", "obs_max", "fw_p", "null_q95", "secs")}),
                      flush=True)
    pd.DataFrame(tops).to_csv(os.path.join(OUT, "screen_top.csv"), index=False)
    pd.DataFrame(wfrows).to_csv(os.path.join(OUT, "walkforward.csv"), index=False)
    pd.DataFrame(named).to_csv(os.path.join(OUT, "named_volume.csv"), index=False)
    COV = pd.DataFrame(list(covrows.values()))
    COV.to_csv(os.path.join(OUT, "coverage.csv"), index=False)
    # ── 4. classifiers ──
    crng = np.random.default_rng(11)
    CL = []; trees = []
    for setn, S in SETS:
        TV = [v for v in TOUCHV if S[S.acted][v].notna().mean() >= 0.5]
        EV = [v for v in ENT if S[v].notna().mean() >= 0.5 and S[v].nunique() > 1]
        for sd in ("IN", "OUT", "both"):
            Ss = S[(S.side == sd) | (sd == "both")]
            tag = f"{setn}_{sd}"
            CL += classifier(Ss, lambda d: d.acted, lambda d: (d.tier == 1).values, "d_BANK", EV + TV, tag, "TOUCH_T1_bank", a.R, crng, trees)
            CL += classifier(Ss, lambda d: d.acted, lambda d: (d.tier == 1).values, "d_BANK", TV, tag, "TOUCH_T1_bank_touchonly", a.R, crng)
            CL += classifier(Ss, lambda d: d.acted, lambda d: (d.C <= 0).values, "d_BANK", EV + TV, tag, "TOUCH_Cle0_bank", a.R, crng)
            CL += classifier(Ss, lambda d: np.ones(len(d), bool), lambda d: (d.tier == 0).values, "d_SKIP", EV, tag, "ENTRY_T0_skip", a.R, crng, trees)
            CL += classifier(Ss, lambda d: np.ones(len(d), bool), lambda d: (d.tier == 1).values, "d_P1", EV, tag, "ENTRY_T1_tp1", a.R, crng, trees)
            print("classifier", tag, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(CL).to_csv(os.path.join(OUT, "classifier.csv"), index=False)
    open(os.path.join(OUT, "trees.txt"), "w").write("\n".join(trees))
    # ── 5. master / live per trade ──
    cols = ["block", "pair", "opened", "side", "tpC", "tier", "peakC", "live_booked", "live_why", "C", "C_why", "touch", "acted", "T_utc",
            "min_to_touch", "cl_T"] + UNCOND
    ML["opened"] = pd.to_datetime(ML.te, unit="ms").dt.strftime("%m-%d %H:%M")
    ML[[c for c in cols if c in ML]].to_csv(os.path.join(OUT, "master_live.csv"), index=False)
    summ.update(screen=scr, runtime_s=time.time() - t0)
    json.dump(summ, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=str)
    print("done", f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
