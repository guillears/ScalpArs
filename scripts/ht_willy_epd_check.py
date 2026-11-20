#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 — LOOK-AHEAD CHECK of the C_EPD control (found after the rule test). Research only.
C_EPD used the fill's H (spike high = the highest high from the spike bar up to the REAL red signal bar). A random minute placed BEFORE the bar
that set H 'knows' price will later trade at H. Clean version: the dip rule on the high KNOWN at the random minute (max high of the spike bar
[spike − 5 min, spike) and every bar up to the previous minute). Re-reads C_EP draws (same draws) restricted by the clean rule, and the share of
C_EPD-style draws that sat before H's bar.
Output reports/HT_WILLY_2026-11-20_epd_check.csv
Usage: venv/bin/python -I scripts/ht_willy_epd_check.py"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ht_willy_sim as W  # noqa: E402
import ht_willy_eval as EV  # noqa: E402
import fast_tp_data as FD  # noqa: E402
MIN, H1 = 60_000, 3_600_000


def main():
    C = pd.read_pickle(os.path.join(W.D, "cohorts.pkl"))
    rows = []
    for sl in ("HT", "WILLY"):
        Z = C[sl]
        for _, r in Z.iterrows():
            a = FD.get1m(r.pair, int(r.spike) - 6 * MIN, int(max(r.spike + 6 * H1, r.ts + H1)) + MIN)
            if a is None:
                continue
            t, o, h, l, c = a
            f = r.C_EP_f; v = r.C_EP_net
            isp = int(np.searchsorted(t, int(r.spike) - 5 * MIN))
            runhi = np.maximum.accumulate(np.where(np.arange(len(t)) >= isp, h, -np.inf))
            iH = isp + int(np.argmax(h[isp:int(np.searchsorted(t, r.ts))])) if int(np.searchsorted(t, r.ts)) > isp else isp
            tH = int(t[iH])
            for k in range(len(v)):
                tj = int(f[k, 0]); j = int(np.searchsorted(t, tj))
                if j < 1 or j >= len(t):
                    continue
                Ej = c[j - 1]; rh = runhi[j - 1]
                rows.append(dict(sleeve=sl, row=r.row, blk=r.blk, half=r.half, day=r.day, real=r.ref_net, net=float(v[k]),
                                 dipH=(r.H - Ej) / r.H * 100, dip_clean=(rh - Ej) / rh * 100, beforeH=tj <= tH, since=(tj - r.spike) / MIN))
    M = pd.DataFrame(rows)
    out = []
    for sl, g in M.groupby("sleeve"):
        for nm, m in (("C_EP (all draws)", np.ones(len(g), bool)), ("C_EPD as run (dip vs fill H ≥ 1.4)", (g.dipH >= 1.4).values),
                      ("C_EPD clean (dip vs high known then ≥ 1.4)", (g.dip_clean >= 1.4).values),
                      ("clean dip ≥ 4", (g.dip_clean >= 4).values), ("clean dip 1.4–4", ((g.dip_clean >= 1.4) & (g.dip_clean < 4)).values)):
            s = g[m]
            per = s.groupby("row").agg(net=("net", "mean"), real=("real", "first"), blk=("blk", "first"), day=("day", "first"), half=("half", "first"))
            for b, pp in [("ALL", per)] + list(per.groupby("blk")):
                d = pp.real - pp.net
                ci = EV.boot_ci(d.values, pp.day.values)
                out.append(dict(sleeve=sl, control=nm, block=b, fills=len(pp), draws=int(m.sum()) if b == "ALL" else np.nan,
                                beforeH_share=round(s.beforeH.mean(), 3) if b == "ALL" else np.nan,
                                real=round(pp.real.mean(), 3), random=round(pp.net.mean(), 3), d=round(d.mean(), 3), lo=round(ci[0], 3), hi=round(ci[1], 3)))
        # draws before vs after H's bar, among the H-dip-qualified draws
        q = g[g.dipH >= 1.4]
        out.append(dict(sleeve=sl, control="H-dip draws BEFORE H's bar (look-ahead)", block="ALL", draws=int(q.beforeH.sum()), random=round(q[q.beforeH].net.mean(), 3)))
        out.append(dict(sleeve=sl, control="H-dip draws AFTER H's bar", block="ALL", draws=int((~q.beforeH).sum()), random=round(q[~q.beforeH].net.mean(), 3)))
    T = pd.DataFrame(out)
    T.to_csv(EV.OUT + "_epd_check.csv", index=False)
    with pd.option_context("display.width", 220):
        print(T.to_string(index=False))
    # clean timing map (DESIGN only) on the known-high dip
    print("\nclean-dip map, DESIGN draws (mean net, draws)")
    g = M[(M.blk == "DESIGN")]
    g = g.assign(b=pd.cut(g.dip_clean, [-1e9, 0, 1.4, 4, 8, 15, 1e9]))
    print(g.groupby(["sleeve", "b"], observed=True).net.agg(["size", "mean"]).round(3).to_string())
    M.to_pickle(os.path.join(W.D, "epd_draws.pkl"))


if __name__ == "__main__":
    main()
