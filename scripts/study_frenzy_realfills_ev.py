#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — Part A EV arithmetic: real hit ordering. For each sleeve and stop S ∈ {3, 4, 5}: P(+X before −S
within 12 h) for X ∈ {1, 1.5, 2, 3, 5, 8}, P(−S first), P(neither), and the implied gross EV of a fixed +X / −S (X·P_hit − S·P_stop + the
12 h close of the rest), next to the walked net value. Also: runners (+5 before −3) — what share hit +1.5 first, and how many of them
dipped −3 AFTER +1.5 and before +5 (the trail killers).
Usage: venv/bin/python scripts/study_frenzy_realfills_ev.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
OUT = "reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19"
if __name__ == "__main__":
    D = pd.read_pickle(f"{RC.RF}/real.pkl"); PP = pd.read_pickle(f"{RC.RF}/real_paths.pkl")
    rows, rr = [], []
    for g in ("FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY", "POOLED"):
        X = D if g == "POOLED" else D[D.sleeve == g]
        for S in (3.0, 4.0, 5.0):
            for Xp in (1, 1.5, 2, 3, 5, 8):
                hit = stop = nei = 0; cl = []
                for r in X.itertuples():
                    P = PP[r.key]; n = int(np.searchsorted(P["t"], P["te"] + 720 * RC.MIN, side="right"))
                    lo, hi = P["lo"][:n], P["hi"][:n]
                    il = np.flatnonzero(lo <= -S); ih = np.flatnonzero(hi >= Xp)
                    a = il[0] if len(il) else n; b = ih[0] if len(ih) else n
                    if a < n and a <= b:
                        stop += 1
                    elif b < n:
                        hit += 1
                    else:
                        nei += 1; cl.append(P["cl"][n - 1])
                N = len(X)
                ev = (Xp * hit - S * stop + sum(cl)) / N
                rows.append(dict(group=g, stop=S, target=Xp, N=N, P_target_first=hit / N, P_stop_first=stop / N, P_neither=nei / N,
                                 gross_EV=ev, breakeven_P=S / (S + Xp)))
        # runner anatomy at −3
        for r in X.itertuples():
            P = PP[r.key]; n = int(np.searchsorted(P["t"], P["te"] + 720 * RC.MIN, side="right"))
            lo, hi, t = P["lo"][:n], P["hi"][:n], P["t"][:n]
            i3 = np.flatnonzero(lo <= -3); k = i3[0] if len(i3) else n
            mfe = hi[:k].max() if k else 0
            i15 = np.flatnonzero(hi >= 1.5); i5 = np.flatnonzero(hi >= 5)
            dd_after = np.nan
            if len(i15) and len(i5) and i5[0] < k:
                seg_hi = np.maximum.accumulate(hi[i15[0]:i5[0] + 1]); seg_lo = lo[i15[0]:i5[0] + 1]
                dd_after = float((seg_hi - seg_lo).max())
            rr.append(dict(group=g, key=r.key, pair=r.pair, sleeve=r.sleeve, mfe_before_m3=mfe, runner5=mfe >= 5,
                           min_to_15=(t[i15[0]] - P["te"]) / RC.MIN if len(i15) else np.nan,
                           min_to_5=(t[i5[0]] - P["te"]) / RC.MIN if len(i5) and i5[0] < k else np.nan,
                           max_giveback_between_15_and_5=dd_after))
    pd.DataFrame(rows).to_csv(f"{OUT}_A_ev_hit_order.csv", index=False)
    R = pd.DataFrame(rr); R[R.group != "POOLED"].to_csv(f"{OUT}_A_runner_anatomy.csv", index=False)
    pd.set_option("display.width", 200)
    E = pd.DataFrame(rows); print(E[E.group == "FRENZY_LONG"].round(3).to_string())
    z = R[(R.group == "POOLED") & R.runner5]
    print("runners", len(z), "giveback between +1.5 and +5: median", z.max_giveback_between_15_and_5.median(),
          "share >= 1", (z.max_giveback_between_15_and_5 >= 1).mean(), ">=2", (z.max_giveback_between_15_and_5 >= 2).mean(),
          ">=3", (z.max_giveback_between_15_and_5 >= 3).mean())
    print(R[R.group != "POOLED"].groupby("sleeve")[["min_to_15", "min_to_5"]].median())
