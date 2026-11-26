#!/usr/bin/env python3
"""HT_PREVOL_AUDIT_2026-11-26 — why does PRE_VOL_SURGE (pre_vol15_vs60 > 1.60) separate on REAL HT fills but not on the backtest?
(operator 2026-11-26: "PRE_VOL_SURGE seems a great filter in all our trades, audit deeply the backtest"; research only)
Prior: DL 596 (master 22 · −1.21) · DL 618 (pre_vol parity backtest vs live stamp 0.00005; PRE_VOL alone on HOLDOUT −$1,037) ·
HT_FILTER_BT_2026-11-26 (261 fills on today's stack: zone +0.29 vs rest −0.31).
Audit legs: ① real 60 fills (all real HT, STATE_EXIT fills_out REAL) + the kept 24 (today's filters), TP 6 re-walk · ② replica: real fills the
backtest also took (same pair, |Δt| ≤ 30 min) — same pre_vol? same outcome? · ③ backtest in the real months (Oct–Nov 2026) and per month ·
④ spike age split (fresh spike inside the 15-min window vs an older spike still accelerating) · ⑤ luck on the real side (within-month label
shuffle) + coin concentration. Out: reports/HT_PREVOL_AUDIT_2026-11-26.txt / .csv"""
import os, sys, importlib
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
B = importlib.import_module("HT_FILTER_BT_2026-11-26")
from services.watch_filters import ht_burst_state
MIN = 60_000; CUT = 1.60


def grp(D, lab):
    z, r = D[D.pv], D[~D.pv]
    f = lambda g: f"{len(g):3d} · {(g.net > 0).mean():4.0%} · {g.net.mean():+.2f}" if len(g) else "  0"
    return f"{lab:46s} zone {f(z)} | rest {f(r)} | gap {(r.net.mean() - z.net.mean()) if len(z) and len(r) else float('nan'):+.2f}"


def luck(D, n=5000, seed=7):
    k = D.pv.values; obs = D.net[~k].mean() - D.net[k].mean(); mo = D.month.values; net = D.net.values; rng = np.random.default_rng(seed); c = 0
    for _ in range(n):
        kk = k.copy()
        for m in np.unique(mo):
            ii = np.nonzero(mo == m)[0]; kk[ii] = rng.permutation(kk[ii])
        c += (net[~kk].mean() - net[kk].mean()) >= obs
    return c / n


def main():
    f = pd.read_csv(os.path.join(R, "reports", "STATE_EXIT_2026-11-21_data", "fills_out.csv"), low_memory=False)
    f = f[(f.sleeve == "HT") & f.kind.isin(["BT", "REAL"])][["kind", "period", "pair", "te", "spike_ms", "tp6"]].rename(columns={"tp6": "net"}).reset_index(drop=True)
    pv, cl = zip(*[B.prevol(p, int(t)) for p, t in zip(f.pair, f.te)])
    f["pre_vol"], f["cloc15"] = pv, cl
    f = f[f.pre_vol.notna()].copy()
    f["pv"] = f.pre_vol > CUT; f["burst"] = [ht_burst_state(a, b) for a, b in zip(f.pre_vol, f.cloc15)]
    f["age_min"] = (f.te - f.spike_ms) / MIN
    ts = pd.to_datetime(f.te, unit="ms"); f["month"] = ts.dt.strftime("%Y-%m"); f["t"] = ts
    K = pd.read_csv(os.path.join(R, "reports", "KEPT_LIST_WILLY_HT_2026-11-21.csv")); kk = set(zip(K.pair, K.k.str[:16]))
    f["kept"] = [(p, str(t)[:16].replace(" ", "T")) in kk for p, t in zip(f.pair, f.t)]
    f.to_csv(os.path.join(R, "reports", "HT_PREVOL_AUDIT_2026-11-26.csv"), index=False)
    RL, BT = f[f.kind == "REAL"], f[f.kind == "BT"]
    L = ["# HT PRE_VOL_SURGE audit — zone (pre_vol > 1.60) vs rest · N · WR · avg TP-6 net % (today's exit)", "",
         "## ① real fills"]
    for lab, D in (("real ALL 60 (every HT fill, any filter era)", RL), ("real ALL minus burst zone (today's DL 619)", RL[RL.burst != "zone"]),
                   ("real KEPT (today's filters)", RL[RL.kept]), ("real KEPT minus burst zone", RL[RL.kept & (RL.burst != "zone")]),
                   ("real master only", RL[RL.period == "MASTER"]), ("real batch only", RL[RL.period == "BATCH"])):
        L.append(grp(D, lab))
    L.append(f"luck (within-month shuffle, real ALL 60): p {luck(RL):.3f} · real minus burst: p {luck(RL[RL.burst != 'zone']):.3f}")
    zc = RL[RL.pv].pair.value_counts(); L.append(f"real zone coins: {zc.to_dict()}")
    L += ["", "## ② replica — real fills the backtest also took (same pair, |Δt| ≤ 30 min)"]
    m = []
    for r in RL.itertuples():
        c = BT[(BT.pair == r.pair) & ((BT.te - r.te).abs() <= 30 * MIN)]
        if len(c):
            b = c.iloc[(c.te - r.te).abs().argmin()]
            m.append(dict(pair=r.pair, t=r.t, dt_min=(b.te - r.te) / MIN, pv_real=r.pre_vol, pv_bt=b.pre_vol, net_real=r.net, net_bt=b.net))
    M = pd.DataFrame(m)
    L.append(f"real fills in the backtest window (≥ 2025-02): {len(RL)} · matched in the backtest: {len(M)}")
    if len(M):
        L.append(f"   same pre_vol side: {(np.sign(M.pv_real - CUT) == np.sign(M.pv_bt - CUT)).mean():.0%} · same outcome sign: {(np.sign(M.net_real) == np.sign(M.net_bt)).mean():.0%} · median |Δt| {M.dt_min.abs().median():.0f} min")
    um = RL[~RL.set_index(["pair", "te"]).index.isin([(a, b) for a, b in zip(M.pair, (M.t.astype("int64") // 10**6))])] if len(M) else RL
    L.append(grp(RL[RL.pair.isin(M.pair) & RL.t.isin(M.t)] if len(M) else RL.iloc[:0], "real fills the backtest also took"))
    L += ["", "## ③ backtest by period"]
    for lab, D in (("backtest ALL 300", BT), ("backtest minus burst zone", BT[BT.burst != "zone"]),
                   ("backtest Oct–Nov 2026 (the real months)", BT[BT.t >= "2026-10-01"]),
                   ("backtest Oct–Nov 2026 minus burst", BT[(BT.t >= "2026-10-01") & (BT.burst != "zone")]),
                   ("backtest Jun–Sep 2026", BT[(BT.t >= "2026-06-01") & (BT.t < "2026-10-01")]),
                   ("backtest 2025", BT[BT.t < "2026-01-01"])):
        L.append(grp(D, lab))
    L.append("   per month (zone N/avg | rest N/avg): " + " · ".join(
        f"{mo} {len(g[g.pv])}/{g[g.pv].net.mean():+.1f} | {len(g[~g.pv])}/{g[~g.pv].net.mean():+.1f}" for mo, g in BT[BT.t >= "2026-01-01"].groupby("month")))
    L += ["", "## ④ spike age (fresh = spike inside the 15-min window, ≤ 20 min)"]
    for lab, D in (("real", RL), ("backtest", BT)):
        for a, g in (("fresh ≤ 20 min", D[D.age_min <= 20]), ("older > 20 min", D[D.age_min > 20])):
            L.append(grp(g, f"{lab} · {a}"))
    L.append(f"   share of fills with a fresh spike: real {(RL.age_min <= 20).mean():.0%} · backtest {(BT.age_min <= 20).mean():.0%} · "
             f"median pre_vol real {RL.pre_vol.median():.2f} · backtest {BT.pre_vol.median():.2f}")
    out = "\n".join(L); print(out); open(os.path.join(R, "reports", "HT_PREVOL_AUDIT_2026-11-26.txt"), "w").write(out + "\n")
    if len(M): print(M.round(2).to_string())


if __name__ == "__main__":
    main()
