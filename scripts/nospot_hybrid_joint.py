#!/usr/bin/env python3
"""NOSPOT_HYBRID — JOINT ADD. The joint short book of scripts/joint_short_sim.py (EXH + FB + L4 + DIST, one account, 0.609 × equity, coin
lock, book 10, margin; flat 0.10 % slip per taker leg like the other sleeves there) with the hybrid's trades added at the lowest priority
(DIST wins a shared coin). Variants: HYB (XE / TP4SL10), HYB without the DIST-shared trades, V1 (1 h change). Full window 2023-07 → 2026-11
and the HOLDOUT days. Usage: venv/bin/python -I scripts/nospot_hybrid_joint.py [extra cohort cells via core.pkl]"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import joint_short_sim as J   # noqa: E402
import nospot_exit_eval as E  # noqa: E402
import nospot_sleeve_lib as L  # noqa: E402
OD = os.path.join(L.R, "reports", "NOSPOT_HYBRID_2026-11-18_data")
EX = {"XE": E.NAMES.index("REF_XE"), "TP4SL10": E.NAMES.index("A_TP4_SL10")}


def hyb_rows(Y, Z, idx, e, name, w=None):
    j = EX[e]; k = Z["k"][idx, j]; tk = k != 0
    net = Z["r"][idx, j].astype(float) + Z["f"][idx, j].astype(float) - 0.10 - 0.05 - np.where(tk, 0.10 + 0.05, 0.02)
    if w is not None:
        net = net * w
    return pd.DataFrame(dict(sleeve="HYB", pair=Y.pair.values[idx], entry_ms=Y.entry_ms.values[idx].astype("int64"),
                             exit_ms=Z["xm"][idx, j].astype("int64"), net=net, nt=1 + tk, stop=(k == 1), depth=J.depth_of(Y.q5.values[idx]),
                             name=name))


def run(T0, extra, label):
    J.PRIO["HYB"] = 5
    out = []
    hd = np.arange(len(J.DAYS))[J.DAYS >= L.TS("2025-07-01") // L.DAY]
    for nm, add in [("all four (EXH+FB+L4+DIST)", None)] + extra:
        T = T0 if add is None else pd.concat([T0, add], ignore_index=True)
        T = T[(T.entry_ms >= J.D0ms) & (T.entry_ms < J.D1ms)].copy(); T["prio"] = T.sleeve.map(J.PRIO)
        T = T.sort_values(["entry_ms", "prio"]).reset_index(drop=True)
        for cap in (False, True):
            s = J.sim(T, cap=cap); r = J.row(nm, T, s)
            hs = J.stats(s["r_day"][hd])
            m = s["taken"] & (T.sleeve.values == "HYB")
            out.append(dict(label=label, config=nm, depth_cap=cap, tpd=r["tpd"], avg=r["avg"], dcr=r["dcr"], dcr_1size=r["dcr_1size"],
                            maxdd=r["maxdd"], worst_day=r["worst_day"], p30=r["p30"], p50=r["p50"], hold_dcr=hs["dcr"], hold_maxdd=hs["maxdd"],
                            hyb_taken=int(m.sum()), hyb_offered=int((T.sleeve.values == "HYB").sum()),
                            hyb_avg_taken=T.net.values[m].mean() if m.any() else np.nan))
    return pd.DataFrame(out)


def main():
    T0, _ = J.load(); T0 = T0[T0.sleeve != "WILLY"].reset_index(drop=True)
    Y = pd.read_pickle(os.path.join(OD, "Y.pkl"))
    _, _, Z = E.load()
    C = pickle.load(open(os.path.join(OD, "core.pkl"), "rb")); SEL, sh = C["SEL"], C["shared"]
    extra = []
    for e in EX:
        idx = SEL[("HYB", e)]
        extra.append((f"+ HYB {e}", hyb_rows(Y, Z, idx, e, "HYB")))
        extra.append((f"+ HYB {e} non-DIST-shared only", hyb_rows(Y, Z, idx[~sh[e]], e, "HYB")))
    extra.append(("+ V1_pos1h XE (secondary)", hyb_rows(Y, Z, SEL[("V1_pos1h", "XE")], "XE", "V1")))
    extra.append(("+ NS0 XE (reference)", hyb_rows(Y, Z, SEL[("NS0", "XE")], "XE", "NS0")))
    for nm, fn in [("filters", "joint_extra.pkl")]:
        f = os.path.join(OD, fn)
        if os.path.exists(f):
            for lab, (idx, e, w) in pickle.load(open(f, "rb")).items():
                extra.append((lab, hyb_rows(Y, Z, idx, e, "HYB", w)))
    JT = run(T0, extra, "joint")
    JT.to_csv(os.path.join(OD, "joint.csv"), index=False)
    pd.set_option("display.width", 250)
    print(JT.round(3).to_string())


if __name__ == "__main__":
    main()
