#!/usr/bin/env python3
"""FB_TP_GRID_2026-11-21 — eval: per block (MASTER 17 / BATCH 3 / backtest DESIGN < 2025-07 / HOLDOUT) × TP: N · WR · avg · Δ vs TP 5 ·
day-block P(Δ>0) + 95 % CI · $ one size ($3,050) · DCR (0.61 × equity / fill) · avg / median duration h · TP/STOP/TIME · saved / cut ·
min leave-one-out Δ · Δ at 0.20 % slip (backtest) · half-years with Δ > 0. Research only."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "reports", "FB_TP_GRID_2026-11-21_data")
SPLIT = int(pd.Timestamp("2025-07-01").value // 10**6)
def boot(dl, day, B=4000):
    u = np.unique(day)
    if len(u) < 2: return np.nan, np.nan, np.nan
    s = pd.DataFrame({"d": day, "x": dl}).groupby("d").x.agg(["sum", "count"]).loc[u]
    ix = np.random.default_rng(7).integers(0, len(u), (B, len(u))); m = s["sum"].values[ix].sum(1) / s["count"].values[ix].sum(1)
    return (m > 0).mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
def dcr(net, xt, t0, t1):
    d = np.asarray(xt, "int64") // 86_400_000; days = np.arange(int(t0) // 86_400_000, int(t1) // 86_400_000 + 1)
    r = pd.Series(np.asarray(net) * 0.61 / 100).groupby(d).sum().reindex(days, fill_value=0.0).values
    eq = np.cumprod(1 + r)
    return (eq[-1] ** (1 / len(days)) - 1) * 100, ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
R = pd.read_csv(os.path.join(D, "real.csv")); B = pd.read_csv(os.path.join(D, "bt.csv"))
B = B[B["mode"] == "row"].copy(); B2 = pd.read_csv(os.path.join(D, "bt.csv")); B2 = B2[B2["mode"] == "s20"]
B["block"] = np.where(B.te < SPLIT, "DESIGN", "HOLDOUT"); B["key"] = B.rid
R["key"] = R.pair + R.k
blocks = {"MASTER": R[R.block == "MASTER"], "BATCH": R[R.block == "BATCH"], "DESIGN": B[B.block == "DESIGN"], "HOLDOUT": B[B.block == "HOLDOUT"]}
rows = []
for bn, G in blocks.items():
    W = G.pivot_table(index="key", columns="cell", values="net"); XT = G.pivot_table(index="key", columns="cell", values="xt")
    WHY = G.pivot_table(index="key", columns="cell", values="why", aggfunc="first"); TE = G.groupby("key").te.first()
    PA = G.groupby("key").pair.first()
    if bn in ("DESIGN", "HOLDOUT"):
        W2 = B2[B2.rid.isin(W.index)].pivot_table(index="rid", columns="cell", values="net")
    day = (TE // 86_400_000).values; half = pd.to_datetime(TE.values, unit="ms").strftime("%Y") + np.where(pd.to_datetime(TE.values, unit="ms").month <= 6, "-H1", "-H2")
    t0, t1 = TE.min(), XT.max().max()
    for cell in ["REF", "TP3", "TP4", "TP4.5", "TP5.5", "TP6", "TP7", "TP8", "TP10"]:
        net = W[cell].values; ref = W["REF"].values; dl = net - ref
        P, lo, hi = boot(dl, day) if cell != "REF" else (np.nan, np.nan, np.nan)
        dur = (XT[cell].values - TE.values) / 3.6e6
        dc, dd = dcr(net, XT[cell].values, t0, t1)
        win = ref > 0
        bp = pd.Series(dl, index=PA.values).groupby(level=0).sum()
        loo = min(((dl.sum() - bp[p]) / max(len(dl) - (PA.values == p).sum(), 1)) for p in bp.index) if cell != "REF" else np.nan
        hp = pd.Series(dl).groupby(half).sum()
        r = dict(block=bn, exit="TP 5 (today)" if cell == "REF" else cell.replace("TP", "TP "), N=len(net), WR=(net > 0).mean() * 100, avg=net.mean(),
                 d=dl.mean(), P=P, ci=f"[{lo:+.2f},{hi:+.2f}]" if cell != "REF" else "", usd=net.sum() * 30.5, DCR=dc, maxDD=dd,
                 dur_avg_h=dur.mean(), dur_med_h=np.median(dur), tp=int((WHY[cell] == "TP").sum()), stop=int((WHY[cell] == "STOP").sum()),
                 time=int(WHY[cell].isin(["TIME", "OPEN"]).sum()), saved=dl[dl > 0].sum() * 30.5, cut=-dl[dl < 0].sum() * 30.5, loo_min=loo,
                 halves=f"{int((hp > 0).sum())}/{len(hp)}" if cell != "REF" else "")
        if bn in ("DESIGN", "HOLDOUT") and cell != "REF":
            r["d_s20"] = (W2[cell] - W2["REF"]).mean()
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv(os.path.join(ROOT, "reports", "FB_TP_GRID_2026-11-21_cells.csv"), index=False)
pd.set_option("display.width", 260)
for bn in blocks:
    print(f"\n== {bn}"); print(T[T.block == bn].drop(columns="block").round(2).to_string(index=False))
# per fill, real
P = R.pivot_table(index=["block", "pair", "k"], columns="cell", values="net").round(2)
H = R.assign(h=(R.xt - R.te) / 3.6e6).pivot_table(index=["block", "pair", "k"], columns="cell", values="h").round(1)
print("\nper fill net (real):"); print(P[["REF", "TP3", "TP4", "TP4.5", "TP5.5", "TP6", "TP7", "TP8"]].to_string())
print("\nper fill duration h (real):"); print(H[["REF", "TP4", "TP6", "TP8"]].to_string())
