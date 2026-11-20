"""DIST_LOSERS2 (2026-11-20): full entry-variable screen on the DL 586 KEPT DIST cohort.

Cohort: reports/ENTRY_WL_2026-11-18_data/dist.pkl (1,547 fills, RSI1h >= 74.07 kept, one position per pair,
Exit B, model slip 0.10 %, real funding). Outcomes: net (booked %), worked, reach.
Bins frozen on DESIGN (2023-07 -> 2025-06): sign (when the variable crosses 0), terciles, quintiles;
2D = tercile x tercile for every variable pair. Welch t of cell vs rest (two-proportion z for WORKED; cells need n >= 30 on both sides; sign = one cell per variable). Family-wise null = 200 label
shuffles, max |t| over the whole family. DESIGN finds, HOLDOUT (2025-07 -> 2026-11) confirms (read once).
"""
import os, sys, itertools, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "reports", "DIST_LOSERS2_2026-11-20_data")
d = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)

OUTCOME_PAT = ("T3A_", "T5A_", "T10A_", "T3F_", "T5F_", "T10F_", "LIVE_")
EXCL = {"entry_ms", "ep_on_ms", "spike_ms", "orig_idx", "zi", "net", "net20", "k", "x", "reach_1m", "worked_1m",
        "best", "reach", "worked", "slip", "label", "reach_5m", "label_5m", "best_close", "rec", "p_fav", "p_adv",
        "in_T5A", "net_T5A", "net20_T5A", "in_LIVE", "net_LIVE", "net20_LIVE", "day", "EXH", "g_ANY"}
vars_ = []
for c in d.columns:
    if c in EXCL or c.startswith(OUTCOME_PAT):
        continue
    s = d[c]
    if s.dtype == object:
        continue
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if s.notna().mean() < 0.6 or s.nunique() < 2:
        continue
    d[c] = s
    vars_.append(c)

des = (d["block"] == "DESIGN").values
hol = ~des
y = d["net"].astype(float).values
yw = d["worked"].astype(float).values
yr = d["reach"].astype(float).values

# frozen bins on DESIGN
cells = {}   # name -> bool array (full cohort)
tercile_ix = {}
for v in vars_:
    s = d[v].values
    sd = s[des & ~np.isnan(s)]
    if len(np.unique(sd)) < 2:
        continue
    if np.nanmin(sd) < 0 < np.nanmax(sd):
        cells[f"{v}<0"] = s < 0
    if len(np.unique(sd)) <= 2:
        for u in np.unique(sd):
            cells[f"{v}=={u:g}"] = s == u
        tercile_ix[v] = [s == u for u in np.unique(sd)]
        continue
    for q, tag in ((3, "T"), (5, "Q")):
        e = np.unique(np.quantile(sd, np.linspace(0, 1, q + 1)[1:-1]))
        b = np.digitize(s, e)
        lst = []
        for k in range(len(e) + 1):
            lo = "-inf" if k == 0 else f"{e[k-1]:.4g}"
            hi = "inf" if k == len(e) else f"{e[k]:.4g}"
            m = (b == k) & ~np.isnan(s)
            cells[f"{v}:{tag}{k+1}[{lo},{hi})"] = m
            lst.append(m)
        if tag == "T":
            tercile_ix[v] = lst
names1 = list(cells)
M1 = np.stack([cells[n] for n in names1], 1)

names2, cols2 = [], []
for a, b in itertools.combinations(sorted(tercile_ix), 2):
    for i, ma in enumerate(tercile_ix[a]):
        for j, mb in enumerate(tercile_ix[b]):
            names2.append(f"{a}#{i+1} & {b}#{j+1}")
            cols2.append(ma & mb)
M2 = np.stack(cols2, 1)
print(f"vars {len(vars_)}  1D cells {len(names1)}  2D cells {len(names2)}", flush=True)


def tstats(M, Y, mask, minn=30, binary=False):
    """Welch t of cell vs rest for every column of M and every column of Y (restricted to mask)."""
    Mm = M[mask].astype(np.float32)
    Ym = Y[mask].astype(np.float32)
    n = Mm.sum(0)[:, None]
    N = Mm.shape[0]
    s1 = Mm.T @ Ym
    s2 = Mm.T @ (Ym ** 2)
    S1 = Ym.sum(0)[None, :]
    S2 = (Ym ** 2).sum(0)[None, :]
    nr = N - n
    m_in = s1 / np.maximum(n, 1)
    m_out = (S1 - s1) / np.maximum(nr, 1)
    v_in = (s2 - n * m_in ** 2) / np.maximum(n - 1, 1)
    v_out = (S2 - s2 - nr * m_out ** 2) / np.maximum(nr - 1, 1)
    if binary:   # two-proportion z (pooled): no zero-variance blow-up on all-0 / all-1 cells
        p = S1 / N
        t = (m_in - m_out) / np.sqrt(np.maximum(p * (1 - p) * (1 / np.maximum(n, 1) + 1 / np.maximum(nr, 1)), 1e-12))
    else:
        t = (m_in - m_out) / np.sqrt(np.maximum(v_in / np.maximum(n, 1) + v_out / np.maximum(nr, 1), 1e-12))
    t[(n[:, 0] < minn) | (nr[:, 0] < minn)] = 0
    return t, n[:, 0], m_in


rng = np.random.default_rng(586)
NSH = 200
res = {}
for oname, yy in (("net", y), ("worked", yw), ("reach", yr)):
    for fam, M, names in (("1D", M1, names1), ("2D", M2, names2)):
        ok = ~np.isnan(yy)
        bn = oname == "worked"
        t, n, m = tstats(M, yy[:, None], des & ok, binary=bn)
        # null: shuffle outcome within DESIGN
        idx = np.where(des & ok)[0]
        mx = []
        for b0 in range(0, NSH, 25):
            Ys = np.empty((len(d), 25), np.float32)
            Ys[:] = np.nan
            for c in range(25):
                Ys[idx, c] = yy[rng.permutation(idx)]
            ts, _, _ = tstats(M, np.nan_to_num(Ys), des & ok, binary=bn)
            mx.extend(np.abs(ts).max(0).tolist())
        null95 = float(np.quantile(mx, 0.95))
        th, nh, mh = tstats(M, yy[:, None], hol & ok, binary=bn)
        df = pd.DataFrame({"cell": names, "n_des": n, "t_des": t[:, 0], "mean_des": m[:, 0],
                           "n_hol": nh, "t_hol": th[:, 0], "mean_hol": mh[:, 0]})
        df["null95"] = null95
        df = df.reindex(df.t_des.abs().sort_values(ascending=False).index)
        df.to_csv(os.path.join(OUT, f"screen_{oname}_{fam}.csv"), index=False)
        res[f"{oname}_{fam}"] = dict(null95=null95, max_t=float(np.abs(t).max()), n_tests=len(names),
                                     n_pass=int((np.abs(t[:, 0]) > null95).sum()))
        print(oname, fam, res[f"{oname}_{fam}"], flush=True)
        print(df.head(12).to_string(index=False), flush=True)
json.dump(dict(res=res, vars=vars_), open(os.path.join(OUT, "screen_summary.json"), "w"), indent=1)
