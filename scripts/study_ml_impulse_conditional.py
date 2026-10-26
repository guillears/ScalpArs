#!/usr/bin/env python3
"""ML impulse study — EXPLORATORY (not gated, flagged): the operator's literal question, conditional on BTC ALREADY RISING.
Among 5m moments with BTC r15 >= +0.10 % (and, second way, >= +0.20 %): does a "real market-wide move" continue more often than
a "BTC spike"? Groups: SPIKE vs TREND (r15 > 0, er_15 >= q_er, spike_15 < q80) · SCORE bucket · CONJ. Y_btc / Y_alt 15-min forward,
P(BTC keeps rising: Y_btc > 0 / >= +0.10), per year, day-bootstrap CI of the TREND − SPIKE and CONJ − not gaps.
Also the same split on momentum-long fills (replica base %).

    venv/bin/python -W ignore scripts/study_ml_impulse_conditional.py > reports/ML_IMPULSE_conditional.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from study_ml_impulse_features import OUT5, REP, load_frozen  # noqa: E402


def boot_gap(y, a, b, day, n=1000, seed=3):
    y = np.asarray(y, float); day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    sa = np.bincount(inv, weights=y * a, minlength=len(u)); ca = np.bincount(inv, weights=a.astype(float), minlength=len(u))
    sb = np.bincount(inv, weights=y * b, minlength=len(u)); cb = np.bincount(inv, weights=b.astype(float), minlength=len(u))
    idx = np.random.default_rng(seed).integers(0, len(u), size=(n, len(u)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = sa[idx].sum(1) / ca[idx].sum(1) - sb[idx].sum(1) / cb[idx].sum(1)
    bs = bs[np.isfinite(bs)]
    return y[a].mean() - y[b].mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    frz = load_frozen()
    G = pd.read_csv(OUT5, index_col=0)
    G["year"] = pd.to_datetime(G.index, unit="ms").year
    G["day"] = pd.to_datetime(G.index, unit="ms").floor("D")
    G["TREND"] = ((G.btc_r15 > 0) & (G.er_15 >= frz["q_er"]) & (G.spike_15 < frz["q_spike80"])).astype(float)
    G["sb"] = pd.cut(G.SCORE, [-1, 2, 4, 6, 9], labels=["0-2", "3-4", "5-6", "7-9"])
    for thr in (0.10, 0.20):
        S = G[G.btc_r15 >= thr]
        print(f"\n## Moments with BTC already up >= +{thr:.2f} % over 15 min (N {len(S)} of {len(G)})")
        rows = []
        for yr, s in S.groupby("year"):
            for lab, m in (("SPIKE", s.SPIKE == 1), ("TREND", s.TREND == 1), ("CONJ", s.CONJ == 1), ("not CONJ", s.CONJ == 0),
                           *[(f"score {k}", s.sb == k) for k in ["0-2", "3-4", "5-6", "7-9"]]):
                g = s[m]
                if len(g) < 30:
                    continue
                rows.append(dict(year=yr, group=lab, N=len(g), days=g.day.nunique(), btc_fwd_bps=g.Y_btc.mean() * 100,
                                 P_btc_up=(g.Y_btc > 0).mean(), P_btc_up10=(g.Y_btc >= 0.10).mean(), P_btc_dn10=(g.Y_btc <= -0.10).mean(),
                                 alt_fwd_bps=g.Y_alt.mean() * 100, P_alt_up=(g.Y_alt > 0).mean()))
            for tg in ("Y_btc", "Y_alt"):
                d, lo, hi = boot_gap(s[tg].values * 100, (s.TREND == 1).values, (s.SPIKE == 1).values, s.day.values)
                d2, lo2, hi2 = boot_gap(s[tg].values * 100, (s.CONJ == 1).values, (s.CONJ == 0).values, s.day.values)
                d3, lo3, hi3 = boot_gap(s[tg].values * 100, (s.SCORE >= 7).values, (s.SCORE <= 4).values, s.day.values)
                print(f"   {yr} {tg}: TREND − SPIKE {d:+.2f} bps [{lo:+.2f}, {hi:+.2f}] · CONJ − not {d2:+.2f} [{lo2:+.2f}, {hi2:+.2f}]"
                      f" · score>=7 − <=4 {d3:+.2f} [{lo3:+.2f}, {hi3:+.2f}]")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    F = pd.read_csv(os.path.join(REP, "ML_IMPULSE_fill_features.csv"))
    F["TREND"] = ((F.btc_r15 > 0) & (F.er_15 >= frz["q_er"]) & (F.spike_15 < frz["q_spike80"])).astype(float)
    F["day"] = pd.to_datetime(F.o_ms, unit="ms").dt.floor("D")
    F["grp"] = F.block.replace({"2026-H1 BEAR": "2026", "2026-H2 BULL": "2026", "2025 BULL": "2025"})
    print("\n## Momentum-long fills entered while BTC was already up >= +0.10 % over 15 min (replica base %, 1x)")
    rows = []
    for b, s in F[F.btc_r15 >= 0.10].groupby("grp"):
        for lab, m in (("SPIKE", s.SPIKE == 1), ("TREND", s.TREND == 1), ("CONJ", s.CONJ == 1), ("not CONJ", s.CONJ == 0),
                       ("score>=7", s.SCORE >= 7), ("score<=4", s.SCORE <= 4), ("all", s.SCORE.notna())):
            g = s[m]
            if len(g):
                rows.append(dict(block=b, group=lab, N=len(g), days=g.day.nunique(), WR=(g.base > 0).mean(), avg=g.base.mean(),
                                 engine=g.today_engine.mean()))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
