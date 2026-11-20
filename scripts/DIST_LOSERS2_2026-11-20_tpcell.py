"""DIST_LOSERS2 (2026-11-20): is there an entry-time cohort where a LOWER take profit beats Exit B?

Per fill, a lower TP X (resting maker) is approximated from the fill's own 1m REACH (best favourable move
before the stop or 24 h): reach >= X + 0.05 (one tick through) -> exit at X; else the Exit B outcome is
unchanged. net_X = X - entry costs (taker 0.05 + slip 0.10) - maker 0.02 (no funding; parity IQR shows it is negligible).
The +0.05 trade-through rule is stricter than Exit B's own fill (91.9 % of Exit B TP fills reach tp + 0.05) -> a slight bias AGAINST a lower TP.
Parity check: fills that took Exit B's TP must reproduce their booked net with X = their own TP.
For every 1D cell (frozen DESIGN bins, 737 cells) and X in {3, 4, 5, 6, 8}: mean paired delta vs Exit B in
the cell. Family-wise null: 200 permutations of the per-fill delta vector, max cell mean (DESIGN).
"""
import os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "reports", "DIST_LOSERS2_2026-11-20_data")
d = pd.read_pickle(os.path.join(ROOT, "reports", "ENTRY_WL_2026-11-18_data", "dist.pkl")).reset_index(drop=True)
des = (d.block == "DESIGN").values
reach = d.reach_1m.fillna(d.reach).astype(float).values
net = d.net.astype(float).values
# parity: k 0 = TP, 1 = stop, 2 = time
print("exit kinds k:", d.k.value_counts().to_dict())
tp_pct = np.minimum(20 * d.atr15_pct.values, 10.0)
COST = 0.05 + 0.10 + 0.02
isTP = (d.k == 0).values  # k: 0 = TP, 1 = stop, 2 = time
if isTP is not None:
    fund = net - (tp_pct - COST)
    print("parity TP fills: N", isTP.sum(), "median net - (tp - cost)", np.median(fund[isTP]).round(3),
          "IQR", np.quantile(fund[isTP], [.25, .75]).round(3))
XS = [3, 4, 5, 6, 8]
D = {}
for X in XS:
    hit = reach >= X + 0.05
    nx = np.where(hit & (tp_pct > X), X - COST, net)
    D[X] = nx - net
    print(f"TP {X}%: whole cohort mean delta DESIGN {D[X][des].mean():+.3f} HOLDOUT {D[X][~des].mean():+.3f}")

# cells from the screen script (same frozen bins)
vars_ = json.load(open(os.path.join(OUT, "screen_summary.json")))["vars"]
cells = {}
for v in vars_:
    s = pd.to_numeric(d[v], errors="coerce").values; sd = s[des & ~np.isnan(s)]
    if len(np.unique(sd)) < 2: continue
    if np.nanmin(sd) < 0 < np.nanmax(sd): cells[f"{v}<0"] = s < 0; cells[f"{v}>=0"] = s >= 0
    if len(np.unique(sd)) <= 2:
        for u in np.unique(sd): cells[f"{v}=={u:g}"] = s == u
        continue
    for q, tag in ((3, "T"), (5, "Q")):
        e = np.unique(np.quantile(sd, np.linspace(0, 1, q + 1)[1:-1])); b = np.digitize(s, e)
        for k in range(len(e) + 1):
            cells[f"{v}:{tag}{k+1}[{'-inf' if k==0 else f'{e[k-1]:.4g}'},{'inf' if k==len(e) else f'{e[k]:.4g}'})"] = (b == k) & ~np.isnan(s)
names = list(cells); M = np.stack([cells[n] for n in names], 1).astype(np.float32)
rng = np.random.default_rng(7)
rows = []
for X in XS:
    dd = D[X]
    def cellmeans(vec, mask):
        Mm = M[mask]; n = Mm.sum(0); s = Mm.T @ vec[mask]
        m = s / np.maximum(n, 1); m[n < 30] = np.nan; return m, n
    m_d, n_d = cellmeans(dd, des); m_h, n_h = cellmeans(dd, ~des)
    idx = np.where(des)[0]; null = []
    for _ in range(200):
        p = dd.copy(); p[idx] = dd[rng.permutation(idx)]
        null.append(np.nanmax(cellmeans(p, des)[0]))
    n95 = float(np.quantile(null, .95))
    o = np.argsort(-np.nan_to_num(m_d, nan=-9))[:8]
    print(f"\nTP {X}%: null95 max cell mean DESIGN = {n95:+.3f}; cells passing = {(np.nan_to_num(m_d, nan=-9) > n95).sum()}")
    for i in o:
        print(f"  {names[i]:42s} n={int(n_d[i]):4d} DES {m_d[i]:+.3f}  HOL n={int(n_h[i]):4d} {m_h[i]:+.3f}")
        rows.append(dict(tp=X, cell=names[i], n_des=int(n_d[i]), d_des=m_d[i], n_hol=int(n_h[i]), d_hol=m_h[i], null95=n95))
    rows.append(dict(tp=X, cell="__null__", null95=n95, n_pass=int((np.nan_to_num(m_d, nan=-9) > n95).sum())))
pd.DataFrame(rows).to_csv(os.path.join(OUT, "tp_cells.csv"), index=False)
