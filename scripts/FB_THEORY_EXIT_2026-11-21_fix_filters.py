#!/usr/bin/env python3
"""FB_THEORY_EXIT — CORRECTION (operator 11-21): today's FLATBASE master = MASTER_POOL_stacked stack_keep (FB_CONFIRM + HIGHER_HIGH,
28) → DL 577 NEAR_BASE skips (LA, BANK, NIL, SIGN, STO, PRL, KAT, EDGE = DL 577's '28 → 20, 7 W / 1 L') → DL 601 funding > +0.005 %
(BTR 10-23 20:00, DODOX 11-13, TAC) = 17. Batch = current batch minus DL 601 (PHAROS). Every cell re-read on both.
→ reports/FB_THEORY_EXIT_2026-11-21_cells_fixed.csv"""
import os, importlib.util
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_data")
F = pd.read_csv(os.path.join(D, "real_fills.csv")); C = pd.read_csv(os.path.join(D, "real_cells.csv"))
M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False, usecols=["opened_at", "pair", "entry_strategy", "stack_keep"])
M = M[M.entry_strategy == "FLATBASE_SHORT"]; M["k"] = M.opened_at.str[:19]
pool = set(zip(M[M.stack_keep == True].k, M[M.stack_keep == True].pair))  # noqa: E712
NB = {("LAUSDT", "2026-10-24T16:00:19"), ("BANKUSDT", "2026-10-26T20:00:10"), ("NILUSDT", "2026-10-27T12:00:09"), ("SIGNUSDT", "2026-11-08T08:00:08"),
      ("STOUSDT", "2026-11-14T12:00:13"), ("PRLUSDT", "2026-11-12T16:00:08"), ("KATUSDT", "2026-10-24T08:00:13"), ("EDGEUSDT", "2026-11-10T08:00:16")}
F["keep_m"] = [(k, p) in pool and (p, k) not in NB and not (fr > 0.00005) for k, p, fr in zip(F.k, F.pair, F.entry_funding_rate.fillna(0))]
F["keep_b"] = (F.k >= "2026-11-17T17:00") & ~(F.entry_funding_rate.fillna(0) > 0.00005)
assert F.keep_m.sum() == 17, F.keep_m.sum()
W = C.pivot_table(index="fid", columns="cell", values="net"); XT = C.pivot_table(index="fid", columns="cell", values="xt")
NOT, X = 3050.0, 0.61
def dcr(net, xt, t0, t1):
    d = (np.asarray(xt, "int64")) // 86_400_000; days = np.arange(int(t0) // 86_400_000, int(t1) // 86_400_000 + 1)
    r = pd.Series(np.asarray(net) * X / 100).groupby(d).sum().reindex(days, fill_value=0.0).values
    return (np.prod(1 + r) ** (1 / len(days)) - 1) * 100
def boot(dl, day, B=3000):
    u = np.unique(day)
    if len(u) < 2: return np.nan
    s = pd.DataFrame({"d": day, "x": dl}).groupby("d").x.agg(["sum", "count"]).loc[u]
    ix = np.random.default_rng(7).integers(0, len(u), (B, len(u)))
    return float(((s["sum"].values[ix].sum(1) / s["count"].values[ix].sum(1)) > 0).mean())
rows = []
for coh, mask in (("MASTER_today", F.keep_m), ("BATCH_today", F.keep_b)):
    g = F[mask]; fids = g.fid.values; ref = W.loc[fids, "REF"].values
    t0 = g.te.min(); t1 = XT.loc[fids].max().max()
    for cell in W.columns:
        net = W.loc[fids, cell].values
        if np.isnan(net).any(): continue
        dl = net - ref; win = ref > 0
        rows.append(dict(cohort=coh, cell=cell, N=len(g), fired=int((np.abs(dl) > 1e-9).sum()), WR=(net > 0).mean() * 100, avg=net.mean(), d=dl.mean(),
                         P=boot(dl, g.te.values // 86_400_000), saved=dl[dl > 0].sum() * NOT / 100, cut=-dl[dl < 0].sum() * NOT / 100,
                         usd=net.sum() * NOT / 100, dcr_b=dcr(ref, XT.loc[fids, "REF"].values, t0, t1), dcr_a=dcr(net, XT.loc[fids, cell].values, t0, t1),
                         avg_loss=net[net < 0].mean() if (net < 0).any() else 0.0))
T = pd.DataFrame(rows); T.to_csv(os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_cells_fixed.csv"), index=False)
pd.set_option("display.width", 250)
for c in ("MASTER_today", "BATCH_today"):
    t = T[T.cohort == c].sort_values("d", ascending=False)
    print("\n==", c, "N", int(t.N.iloc[0])); print(t.drop(columns="cohort").round(2).head(14).to_string(index=False))
print(F[F.keep_m][["pair", "k", "booked", "reason"]].to_string())
