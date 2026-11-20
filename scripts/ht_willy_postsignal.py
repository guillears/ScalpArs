#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 — CORRECTED random-entry control (found during the review of the pre-registered controls). Research only.
WILLY / HT enter on the FIRST red 5m bar after the spike: every minute between the spike close and the real entry lies in a stretch that hindsight
knows was green (rising). Random draws placed there (C_LOC −30…−1, the early part of C_EP) ride that known rise → biased in favour of random.
Clean control = draws AFTER the real signal only (same episode, same exit): C_LOC_POST (+1…+30 min) and C_EP_POST (signal < t < window end).
Also the pre-signal share and its average, to show the size of the bias. Same 50-draw sets as ht_willy_sim (no new draws).
Output reports/HT_WILLY_2026-11-20_random_post.csv
Usage: venv/bin/python -I scripts/ht_willy_postsignal.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ht_willy_sim as W  # noqa: E402
import ht_willy_eval as EV  # noqa: E402


def block_rows(Z, sl, label, ctrls):
    rows = []
    for cn in ctrls:
        if cn + "_net" not in Z:
            continue
        for part in ("PRE", "POST"):
            per = []
            for _, r in Z.iterrows():
                f = r[cn + "_f"]; v = r[cn + "_net"]
                if not len(v):
                    continue
                m = (f[:, 0] < r.ts) if part == "PRE" else (f[:, 0] > r.ts)
                if m.sum() >= 3:
                    per.append(dict(row=r.row, real=r.ref_net, rnd=float(np.mean(v[m])), n=int(m.sum()), blk=r.blk, half=r.half, day=r.day,
                                    rk=float((v[m] < r.ref_net).mean())))
            P = pd.DataFrame(per)
            if not len(P):
                continue
            blocks = [("ALL", P)] + (list(P.groupby("blk")) + list(P.groupby("half")) if label == "BACKTEST" else [])
            for b, pp in blocks:
                if len(pp) < 3:
                    continue
                d = pp.real - pp.rnd; ci = EV.boot_ci(d.values, pp.day.values)
                rows.append(dict(set=label, sleeve=sl, control=f"{cn}_{part}", block=b, fills=len(pp), draws=int(pp.n.sum()), days=pp.day.nunique(),
                                 real=round(pp.real.mean(), 3), random=round(pp.rnd.mean(), 3), d=round(d.mean(), 3), lo=round(ci[0], 3), hi=round(ci[1], 3),
                                 rank=round(pp.rk.mean(), 3)))
    return rows


def main():
    C = pd.read_pickle(os.path.join(W.D, "cohorts.pkl"))
    REAL = pd.read_pickle(os.path.join(W.D, "real.pkl"))
    rows = []
    rows += block_rows(C["WILLY"], "WILLY", "BACKTEST", ("C_LOC", "C_EP"))
    rows += block_rows(C["HT"], "HT", "BACKTEST", ("C_LOC", "C_EP"))
    rows += block_rows(C["HT_nodip"], "HT_nodip", "BACKTEST", ("C_LOC", "C_EP"))
    Rw = REAL[REAL.sleeve == "WILLY"].copy(); Rh = REAL[(REAL.sleeve == "HT") & (pd.to_datetime(REAL.opened_at) >= pd.Timestamp("2026-11-18T22:56"))].copy()
    for Z in (Rw, Rh):
        Z["blk"] = "REAL"; Z["half"] = "REAL"; Z["day"] = EV.day(Z.ts)
    rows += block_rows(Rw, "WILLY", "MASTER B19-B27", ("C_LOC", "C_EP"))
    rows += block_rows(Rh, "HT", "CURRENT (post-587)", ("C_LOC", "C_EP"))
    T = pd.DataFrame(rows); T.to_csv(EV.OUT + "_random_post.csv", index=False)
    with pd.option_context("display.width", 220):
        print(T.to_string(index=False))


if __name__ == "__main__":
    main()
