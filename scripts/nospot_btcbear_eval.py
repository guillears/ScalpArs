#!/usr/bin/env python3
"""NOSPOT_BTCBEAR stage 1 (PREREG §main): definition × cohort × exit {REF, A_TP4_SL10} × switch state {ON bear, OFF bull, all},
per block; RND control; rotation null (HOLDOUT, 1,000, family-wise); Addendum E1 ranking. Writes reports/NOSPOT_BTCBEAR_2026-11-18_main.csv,
_control.csv, _null.csv, _e1.csv and NOSPOT_BTCBEAR_data/stage1.pkl.  Usage: venv/bin/python -I scripts/nospot_btcbear_eval.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nospot_btcbear_lib as B   # noqa: E402
L = B.L; S = B.S
OUT = os.path.join(B.R, "reports", "NOSPOT_BTCBEAR_2026-11-18")
CL = S.cells(); NAMES = [c["name"] for c in CL]
MAIN_EX = ["REF_XE", "A_TP4_SL10"]
CONTAM = {(d, c, e) for d in ("a", "d") for c in ("R4", "NS0", "R1") for e in ("REF_XE", "A_TP4_SL10")}


def main():
    RS = B.regime_series(); Y, Z = B.load(); NET = B.nets(Y, Z); net, net20, _ = NET
    T = Y.entry_ms.values.astype(np.int64); blk = Y.blk.values; pair = Y.pair.values; half = Y.half.values
    REG = {d: B.at(RS, d, T) for d in B.DEFS}
    rows = []; SETS = {}
    d0a, d1a = B.SPAN0 // B.DAY, B.SPAN1 // B.DAY
    DUTY = {d: B.duty(RS, d, d0a, d1a) for d in B.DEFS}
    for c in B.COHORTS:
        base = np.nonzero(Y[c].values)[0]
        for e in MAIN_EX:
            j = NAMES.index(e)
            for d in B.DEFS:
                for state, sv in (("ON", -1), ("OFF", 1), ("ALL", None)):
                    ii = base if sv is None else base[REG[d][base] == sv]
                    ii = B.one_pos(Y, Z, ii, j)
                    SETS[(d, c, e, state)] = ii
                    for bk in ("DESIGN", "HOLDOUT", "ALL"):
                        q = ii if bk == "ALL" else ii[blk[ii] == bk]
                        Dn, dd0, dd1 = L.cal_days(bk)
                        du = DUTY[d].loc[dd0:dd1 - 1]
                        on_days = du.sum()
                        rec = dict(defn=d, cohort=c, exit=e, state=state, blk=bk, contaminated=(d, c, e) in CONTAM,
                                   duty=du.mean() * 100, on_days_eq=on_days,
                                   tpd_on=len(q) / on_days if (state == "ON" and on_days > 0) else np.nan, tpd_avg=len(q) / Dn)
                        rec.update(B.cell_stats(net[q, j], net20[q, j], Y.day.values[q], pair[q], half[q]))
                        rows.append(rec)
        print("cohort", c, flush=True)
    MT = pd.DataFrame(rows); MT.to_csv(OUT + "_main.csv", index=False)
    # RND control (no one-position), same switch, same exits
    ct = []
    for e in MAIN_EX:
        j = NAMES.index(e)
        for d in B.DEFS:
            for state, sv in (("ON", -1), ("OFF", 1), ("ALL", None)):
                m = np.ones(len(Y), bool) if sv is None else REG[d] == sv
                for bk in ("DESIGN", "HOLDOUT", "ALL"):
                    mm = m if bk == "ALL" else m & (blk == bk)
                    v = net[mm, j]
                    ct.append(dict(defn=d, exit=e, state=state, blk=bk, N=int(mm.sum()), avg=v.mean(), WR=(v > 0).mean() * 100,
                                   f20=net20[mm, j].mean()))
    CT = pd.DataFrame(ct); CT.to_csv(OUT + "_control.csv", index=False)
    # rotation null on HOLDOUT (approximate sets = ungated one-position cohort trades, HOLDOUT)
    lo, hi = B.HOLD0, B.SPAN1
    cells = []
    for c in B.COHORTS:
        for e in MAIN_EX:
            j = NAMES.index(e); ii = SETS[("a", c, e, "ALL")]; ii = ii[blk[ii] == "HOLDOUT"]
            cells.append((c, e, ii, net[ii, j], T[ii]))
    def stat(shift):
        out = {}
        for d in B.DEFS:
            for c, e, ii, v, tt in cells:
                r = B.at(RS, d, tt, shift, lo, hi) == -1
                out[(d, c, e)] = v[r].mean() - v.mean() if r.sum() >= 10 else np.nan
        return out
    obs = stat(0)
    rng = np.random.default_rng(11)
    span = hi - lo; NB = 1000
    maxes = np.zeros(NB); per = {k: np.zeros(NB) for k in obs}
    for b in range(NB):
        sh = int(rng.uniform(14 * B.DAY, span - 14 * B.DAY))
        st = stat(sh)
        vals = np.array([x for x in st.values() if np.isfinite(x)])
        maxes[b] = vals.max()
        for k, x in st.items():
            per[k][b] = x
        if b % 100 == 0:
            print("null", b, flush=True)
    nu = []
    for k, x in obs.items():
        nu.append(dict(defn=k[0], cohort=k[1], exit=k[2], stat=x, p_cell=np.mean(per[k] >= x) if np.isfinite(x) else np.nan,
                       p_fw=np.mean(maxes >= x) if np.isfinite(x) else np.nan, contaminated=k in CONTAM))
    NU = pd.DataFrame(nu); NU.to_csv(OUT + "_null.csv", index=False)
    np.save(os.path.join(B.DD, "null_maxes_main.npy"), maxes)
    # E1 ranking: DESIGN window t of ON trades under REF, ≥ 10 DESIGN windows
    e1 = MT[(MT.state == "ON") & (MT.exit == "REF_XE") & (MT.blk == "DESIGN")][["defn", "cohort", "N", "windows", "avg", "t_win"]].copy()
    e1["eligible"] = e1.windows >= 10
    e1 = e1.sort_values("t_win", ascending=False); e1.to_csv(OUT + "_e1.csv", index=False)
    picks = []; used = set()
    for _, r in e1[e1.eligible].iterrows():
        if r.defn in used or not np.isfinite(r.t_win):
            continue
        picks.append((r.defn, r.cohort)); used.add(r.defn)
        if len(picks) == 2:
            break
    print("E1 picks", picks)
    pd.to_pickle(dict(picks=picks, SETS=SETS), os.path.join(B.DD, "stage1.pkl"))


if __name__ == "__main__":
    main()
