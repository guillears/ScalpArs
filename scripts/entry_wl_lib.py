#!/usr/bin/env python3
"""ENTRY_WL (PREREG reports/ENTRY_WL_2026-11-18_PREREG.md) — shared screen machinery. Research only, BACKTEST."""
import os
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RP = os.path.join(R, "reports"); D_OUT = os.path.join(RP, "ENTRY_WL_2026-11-18_data")
DAY = 86_400_000
SIZE = 0.61
MIN_N = 30
EXCL = {"pair", "entry_ms", "xi", "row", "net", "net20", "k", "x", "reach", "worked", "best", "slip", "q5_row", "ep_on_ms", "spike_ms",
        "label", "block", "half", "day", "ep", "in_pool", "is15", "EXH", "g_ANY", "orig_idx", "zi", "in_T5A", "net_T5A", "net20_T5A",
        "in_LIVE", "net_LIVE", "net20_LIVE", "p_fav", "p_adv", "reach_5m", "best_close", "rec", "label_5m", "group", "entry_label",
        "usd_per_10k", "keep_strength", "block_name", "entry_utc", "reach_1m", "worked_1m", "y_rg", "sl_pct", "s_atrp", "st_prior_fill_reach"}
EXCL_PREFIX = ("T3A_", "T5A_", "T10A_", "T3F_", "T5F_", "T10F_", "LIVE_")
MKT_PREFIX = ("btc_", "breadth", "frz_", "n_flagged", "st_open_same", "st_fills24h", "st_stops24h", "st_wr24h_closed")
THEORY = {
    "T1 fuel left": ["dd_peak_pct", "run_pct", "ret1h_pct", "ret4h_pct", "ret24h_pct", "dist_vwap_pct", "dist_base_pct"],
    "T2 exhaustion depth": ["vd", "vd24", "vmult15", "s_vd"],
    "T3 crowding": ["top_pos_ls", "top_acct_ls", "glob_ls", "fund_last", "prem_last", "oi_chg1h", "oi_chg4h", "oi_to_vol24",
                    "top_pos_chg4h", "s_ls_top_pos", "s_ls_top_acc", "s_ls_glob", "s_oi_vol", "s_f_last", "oi_chg15m", "oi_chg30m"],
    "T4 market support": ["btc_d1_regime", "btc_ret1h", "btc_ret4h", "btc_ret1d", "btc_slope1h_sign", "btc_regime", "breadth"],
    "T5 coin quality": ["q5", "vol24", "mcap", "atr15_pct", "has_spot", "list_age_d"],
    "T6 time / second peak": ["hrs_since_spike", "hrs_since_peak", "n_prior_exh_sig", "n_prior_vsw_red", "st_prior_fills_ep"],
    "T7 taker flow": ["taker_1h", "s_taker_1h"],
    "T8 DIST strength": ["rsi1h", "rsi4h", "gap1h", "di_diff1h"],
}
THEORY_VARS = [v for vs in THEORY.values() for v in vs]
HALF_MAP = {"2023-H2": "23-H2", "2024-H1": "24-H1", "2024-H2": "24-H2", "2025-H1": "25-H1", "2025-H2": "25-H2", "2026-H1": "26-H1",
            "2026-H2": "26-H2"}
HALVES = ["23-H2", "24-H1", "24-H2", "25-H1", "25-H2", "26-H1", "26-H2"]


def load(sleeve):
    D = pd.read_pickle(os.path.join(D_OUT, f"{sleeve}.pkl")).copy()
    T = pd.to_datetime(D.entry_ms, unit="ms")
    D["half"] = (T.dt.year % 100).astype(str) + np.where(T.dt.month <= 6, "-H1", "-H2")
    D["y_rg"] = np.where(D.label == "RUNNER", 1.0, np.where(D.label == "GAVE_BACK", 0.0, np.nan))
    return D


def theory_of(v):
    for k, vs in THEORY.items():
        if v in vs:
            return k
    return ""


def variables(D):
    out = []
    for c in D.columns:
        if c in EXCL or c.startswith(EXCL_PREFIX):
            continue
        s = D[c]
        if s.dtype == object:
            continue
        v = pd.to_numeric(s, errors="coerce").astype(float)
        if v.notna().mean() < 0.05 or v.nunique() < 2:
            continue
        out.append(c)
    return out


def is_mkt(v):
    return v.startswith(MKT_PREFIX)


def levels(D, vars_, dmask):
    """level masks. gran 'sign' (or 'cat' for ≤ 5 values), 'terc' (3), 'quint' (Q1, Q5). Edges from DESIGN rows."""
    L = []          # dict(var, gran, lev, mask, valid)
    for c in vars_:
        v = pd.to_numeric(D[c], errors="coerce").astype(float).values
        ok = np.isfinite(v); vd = v[ok & dmask]
        if len(vd) < 2 * MIN_N:
            continue
        u = np.unique(vd)
        if len(u) <= 5:
            for x in u:
                m = ok & (v == x)
                if (m & dmask).sum() >= MIN_N and ((ok & ~m) & dmask).sum() >= MIN_N:
                    L.append(dict(var=c, gran="cat", lev=f"={x:g}", mask=m, valid=ok))
            continue
        if (vd > 0).sum() >= MIN_N and (vd <= 0).sum() >= MIN_N:
            L.append(dict(var=c, gran="sign", lev="≤0", mask=ok & (v <= 0), valid=ok))
            L.append(dict(var=c, gran="sign", lev=">0", mask=ok & (v > 0), valid=ok))
        q1, q2 = np.quantile(vd, [1 / 3, 2 / 3])
        if q1 < q2:
            L.append(dict(var=c, gran="terc", lev=f"T1≤{q1:.4g}", mask=ok & (v <= q1), valid=ok, lo=-np.inf, hi=q1))
            L.append(dict(var=c, gran="terc", lev=f"T2", mask=ok & (v > q1) & (v <= q2), valid=ok, lo=q1, hi=q2))
            L.append(dict(var=c, gran="terc", lev=f"T3>{q2:.4g}", mask=ok & (v > q2), valid=ok, lo=q2, hi=np.inf))
        p1, p4 = np.quantile(vd, [0.2, 0.8])
        if p1 < p4:
            L.append(dict(var=c, gran="quint", lev=f"Q1≤{p1:.4g}", mask=ok & (v <= p1), valid=ok, lo=-np.inf, hi=p1))
            L.append(dict(var=c, gran="quint", lev=f"Q5>{p4:.4g}", mask=ok & (v > p4), valid=ok, lo=p4, hi=np.inf))
    return L


def contrasts_1d(L):
    """(i, j) pairs of level indices: sign ≤0 vs >0 · cat level vs rest (j = -1) · terc T3 vs T1 · quint Q5 vs Q1."""
    C = []
    by = {}
    for i, l in enumerate(L):
        by.setdefault((l["var"], l["gran"]), []).append(i)
    for (v, g), ii in by.items():
        if g == "cat":
            C += [(i, -1) for i in ii]
        elif g == "sign":
            C.append((ii[1], ii[0]))
        elif g == "terc":
            C.append((ii[2], ii[0]))
        elif g == "quint":
            C.append((ii[1], ii[0]))
    return C


def welch(n1, s1, q1, n2, s2, q2, binary=False):
    with np.errstate(divide="ignore", invalid="ignore"):
        if binary:      # pooled two-proportion z (a Welch t explodes on an all-0 / all-1 cell)
            m1, m2 = s1 / n1, s2 / n2; p = (s1 + s2) / (n1 + n2)
            t = (m1 - m2) / np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
            return np.where((n1 >= MIN_N) & (n2 >= MIN_N), t, np.nan), m1, m2
        m1, m2 = s1 / n1, s2 / n2
        v1 = np.maximum(q1 / n1 - m1 ** 2, 0) * n1 / np.maximum(n1 - 1, 1)
        v2 = np.maximum(q2 / n2 - m2 ** 2, 0) * n2 / np.maximum(n2 - 1, 1)
        t = (m1 - m2) / np.sqrt(v1 / n1 + v2 / n2)
    return np.where((n1 >= MIN_N) & (n2 >= MIN_N), t, np.nan), m1, m2


class Screen:
    """1D + 2D Welch t on rows `rows` for outcome y (rows with finite y)."""

    def __init__(self, D, L, rows):
        self.L = L
        self.M = np.stack([l["mask"][rows] for l in L], 1).astype(np.float64)
        self.vars = sorted(set(l["var"] for l in L))
        vi = {v: i for i, v in enumerate(self.vars)}
        self.lv = np.array([vi[l["var"]] for l in L])
        self.V = np.stack([L[[i for i, l in enumerate(L) if l["var"] == v][0]]["valid"][rows] for v in self.vars], 1).astype(np.float64)
        self.gran = np.array([l["gran"] for l in L])
        self.C1 = contrasts_1d(L)
        # 2D level pairs: same family (sign/cat × sign/cat ; terc × terc), different variables
        fam = np.where(np.isin(self.gran, ["sign", "cat"]), 0, np.where(self.gran == "terc", 1, 2))
        I, J = np.triu_indices(len(L), 1)
        keep = (fam[I] == fam[J]) & (fam[I] < 2) & (self.lv[I] != self.lv[J])
        self.I, self.J = I[keep], J[keep]

    def stats(self, y):
        M, V = self.M, self.V
        y = np.asarray(y, float); y2 = y * y
        bn = bool(np.isin(y, (0.0, 1.0)).all())
        n_l = M.sum(0); s_l = y @ M; q_l = y2 @ M
        n_v = V.sum(0); s_v = y @ V; q_v = y2 @ V
        out1 = []
        for i, j in self.C1:
            if j == -1:
                vv = self.lv[i]
                t, m1, m2 = welch(n_l[i], s_l[i], q_l[i], n_v[vv] - n_l[i], s_v[vv] - s_l[i], q_v[vv] - q_l[i], bn)
                out1.append((t, m1, m2, n_l[i], n_v[vv] - n_l[i]))
            else:
                t, m1, m2 = welch(n_l[i], s_l[i], q_l[i], n_l[j], s_l[j], q_l[j], bn)
                out1.append((t, m1, m2, n_l[i], n_l[j]))
        o1 = np.array(out1, float)
        I, J = self.I, self.J
        C0 = M.T @ M; C1 = M.T @ (y[:, None] * M); C2 = M.T @ (y2[:, None] * M)
        W0 = V.T @ V; W1 = V.T @ (y[:, None] * V); W2 = V.T @ (y2[:, None] * V)
        a, b = self.lv[I], self.lv[J]
        n1, s1, q1 = C0[I, J], C1[I, J], C2[I, J]
        n2, s2, q2 = W0[a, b] - n1, W1[a, b] - s1, W2[a, b] - q1
        t2, m1, m2 = welch(n1, s1, q1, n2, s2, q2, bn)
        return o1, np.stack([t2, m1, m2, n1, n2], 1)

    def maxabs(self, y):
        o1, o2 = self.stats(y)
        return np.nanmax(np.abs(np.r_[o1[:, 0], o2[:, 0]]))


def auc(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n1 = (y == 1).sum(); n0 = (y == 0).sum()
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(x).rank().values
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def day_boot_diff(v, side, days, nb=2000, block=10, rng=None):
    """10-day block bootstrap of mean(v | side) − mean(v | ~side)."""
    rng = rng or np.random.default_rng(7)
    df = pd.DataFrame(dict(d=days, a=np.where(side, v, 0.0), na=side.astype(float), b=np.where(~side, v, 0.0), nb_=(~side).astype(float)))
    g = df.groupby("d").sum()
    dd = np.arange(days.min(), days.max() + 1); g = g.reindex(dd, fill_value=0.0)
    n = len(dd); nbk = int(np.ceil(n / block)); st = rng.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    A = g.a.values[idx].sum(1) / np.maximum(g.na.values[idx].sum(1), 1)
    B = g.b.values[idx].sum(1) / np.maximum(g.nb_.values[idx].sum(1), 1)
    d = A - B
    return np.percentile(d, 2.5), np.percentile(d, 97.5), np.percentile(d, 5), np.percentile(d, 95)


def day_boot_mean(v, days, nb=2000, block=1, rng=None):
    """window-clustered (day) bootstrap of the mean → (2.5, 5, 95, 97.5) percentiles."""
    rng = rng or np.random.default_rng(11)
    g = pd.DataFrame(dict(d=days, s=v, n=1.0)).groupby("d").sum()
    k = len(g); idx = rng.integers(0, k, (nb, k))
    m = g.s.values[idx].sum(1) / g.n.values[idx].sum(1)
    return np.percentile(m, [2.5, 5, 95, 97.5])


def wilson_floor(k, n, z=1.645):
    if n <= 0:
        return 0.0
    p = k / n; den = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0))) / den


def kel(v):
    v = np.asarray(v, float)
    w = v[v > 0]; lo = v[v <= 0]
    if not len(w) or not len(lo):
        return np.nan, np.nan, np.nan
    aw, al = w.mean(), -lo.mean(); pf = wilson_floor(len(w), len(v))
    return pf - (1 - pf) / (aw / al), pf * 100, al / (aw + al) * 100


def onesize(v, xms, d0, d1, size=SIZE):
    """DCR-cal at one size: r_d = Σ fill % × size ÷ 100 over fills closed that day; every calendar day d0..d1-1."""
    size = np.broadcast_to(np.asarray(size, float), np.shape(v))
    r = pd.Series(np.asarray(v) * size / 100).groupby(np.asarray(xms) // DAY).sum().reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values)
    if eq[-1] <= 0 or (1 + r.values).min() <= 0:
        return dict(dcr=np.nan, maxdd=100.0, worst_day=r.min() * 100), r
    return dict(dcr=(eq[-1] ** (1 / len(r)) - 1) * 100, maxdd=(1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max() * 100,
                worst_day=r.min() * 100), r


def logistic_fit(X, y, lam=1.0, iters=50):
    n, p = X.shape
    Xb = np.c_[np.ones(n), X]; w = np.zeros(p + 1)
    for _ in range(iters):
        z = Xb @ w; pr = 1 / (1 + np.exp(-z))
        g = Xb.T @ (pr - y) + lam * np.r_[0, w[1:]]
        H = (Xb * (pr * (1 - pr))[:, None]).T @ Xb + lam * np.diag(np.r_[0, np.ones(p)])
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def logistic_score(X, w):
    return np.c_[np.ones(len(X)), X] @ w
