#!/usr/bin/env python3
"""DIST_FUNDING_EXIT tables + chart (reads the results of scripts/dist_funding_exit_study.py for DIST and EXH).
Usage: venv/bin/python -I scripts/dist_funding_exit_report.py"""
import os, sys, pickle
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import dist_funding_exit_study as ST   # noqa: E402
MIN, H1 = 60_000, 3_600_000
rng = np.random.default_rng(11)
HALVES = ["23H2", "24H1", "24H2", "25H1", "25H2", "26H1", "26H2"]


def boot(v, day, n=2000):
    blk = day // 10; ub = np.unique(blk); idx = {b: np.where(blk == b)[0] for b in ub}
    sums = np.array([v[idx[b]].sum() for b in ub]); s = []
    for _ in range(n):
        s.append(sums[rng.integers(0, len(ub), len(ub))].sum())
    return np.percentile(s, [2.5, 97.5])


def sleeve_tables(tag, d, L):
    C = d["C"]; res = d["res"]; q95 = np.percentile(d["nullmax"], 95)
    C["day"] = C.entry_ms // 86_400_000; C["h"] = C.half.str[2:]
    D = (C.blk == "DESIGN").values; Hm = ~D
    rows = []
    for (src, X, M, B), (v, nc, fp) in res.items():
        if src != "real":
            continue
        t = nc > 0
        dD, dH = v[D].sum(), v[Hm].sum()
        ciD = boot(v[D], C.day.values[D]); ciH = boot(v[Hm], C.day.values[Hm])
        bypair = pd.Series(v).groupby(C.pair.values).sum()
        tot = v.sum()
        top_share = (bypair.max() / bypair[bypair > 0].sum() * 100) if tot > 0 and (bypair > 0).any() else (bypair.min() / bypair[bypair < 0].sum() * 100 if (bypair < 0).any() else 0)
        top_pair = bypair.idxmax() if tot > 0 else bypair.idxmin()
        big = np.argmax(v) if tot > 0 else np.argmin(v)
        loo1 = tot - v[big]; loo_pair = tot - bypair[top_pair]
        hs = pd.Series(v[t]).groupby(C.h.values[t]).sum()
        hpos = f"{(hs > 0).sum()}/{len(hs)}"
        ok = dH > 0 and ciH[0] > 0 and dD > 0 and dD > q95 and top_share < 50 and (hs > 0).sum() >= 2 / 3 * len(hs)
        sv = v[t & (v > 0)]; ct = v[t & (v < 0)]
        dfund = (fp - C.ref_f.values)[t]
        rows.append(dict(rule=f"X{X:g} M{M} {B}", X=X, M=M, B=B,
                         nD=int((t & D).sum()), daysD=C.day[t & D].nunique(), cdD=len(set(zip(C.pair[t & D], C.day[t & D]))),
                         nH=int((t & Hm).sum()), daysH=C.day[t & Hm].nunique(), cdH=len(set(zip(C.pair[t & Hm], C.day[t & Hm]))),
                         dD=dD, ciD=ciD, dH=dH, ciH=ciH, null=q95, top=top_pair, top_share=top_share, loo1=loo1, loo_pair=loo_pair,
                         hpos=hpos, hs=hs, ok=ok, n_saved=len(sv), s_saved=sv.sum(), n_cut=len(ct), s_cut=ct.sum(),
                         fund_avoided=dfund.sum(), price_cost=v[t].sum() - dfund.sum(), avg_t=v[t].mean() if t.any() else 0))
    R_ = pd.DataFrame(rows)
    nref = C.ref.notna().sum()
    win = C.ref[C.ref > 0].mean(); los = -C.ref[C.ref <= 0].mean()
    L += [f"\n## {tag}", "",
          f"- Fills {len(C)} (DESIGN {D.sum()} · HOLDOUT {Hm.sum()}); REF avg net {C.ref.mean():+.2f} %; WR {(C.ref > 0).mean():.0%}; avg funding in REF {C.ref_f.mean():+.3f} % / fill.",
          f"- Family-wise null (300 shuffles, max DESIGN Δ over 16 variants) 95th = **{q95:+.1f}** summed %.", "",
          "### Per rule (Δ = rule − hold, summed net % at one size; touched = fills the rule actually closed)", "",
          "| rule | DESIGN touched N · days · coin-days | DESIGN Δ (95 % CI) | HOLDOUT touched N · days · coin-days | HOLDOUT Δ (95 % CI) | half-years Δ>0 | pass |",
          "|---|---|---|---|---|---|---|"]
    for r in R_.itertuples():
        L.append(f"| {r.rule} | {r.nD} · {r.daysD} · {r.cdD} | {r.dD:+.1f} ({r.ciD[0]:+.1f} … {r.ciD[1]:+.1f}) | {r.nH} · {r.daysH} · {r.cdH} | "
                 f"{r.dH:+.1f} ({r.ciH[0]:+.1f} … {r.ciH[1]:+.1f}) | {r.hpos} | {'✓' if r.ok else '✗'} |")
    L += ["", "### Saved vs cut · funding avoided vs price give-up · concentration · leave-one-out (all periods)", "",
          "| rule | saved N · Σ% | cut N · Σ% | avg Δ / touched fill | funding avoided Σ% | price + extra costs Σ% | top pair (share) | LOO best fill | LOO top pair |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in R_.itertuples():
        L.append(f"| {r.rule} | {r.n_saved} · {r.s_saved:+.1f} | {r.n_cut} · {r.s_cut:+.1f} | {r.avg_t:+.2f} | {r.fund_avoided:+.1f} | {r.price_cost:+.1f} | "
                 f"{r.top} ({r.top_share:.0f} %) | {r.loo1:+.1f} | {r.loo_pair:+.1f} |")
    L += ["", "### Δ by half-year (summed %, touched fills only; blank = none touched)", "", "| rule | " + " | ".join(HALVES) + " |", "|---" * (len(HALVES) + 1) + "|"]
    for r in R_.itertuples():
        L.append(f"| {r.rule} | " + " | ".join(f"{r.hs[h]:+.1f}" if h in r.hs.index else "" for h in HALVES) + " |")
    # prev-rate sensitivity
    L += ["", "### Sensitivity — trigger on the PREVIOUS settled rate (no look-ahead)", "", "| rule | DESIGN touched · Δ | HOLDOUT touched · Δ |", "|---|---|---|"]
    for (src, X, M, B), (v, nc, fp) in res.items():
        if src == "prev":
            t = nc > 0
            L.append(f"| X{X:g} M{M} {B} | {(t & D).sum()} · {v[D].sum():+.1f} | {(t & Hm).sum()} · {v[Hm].sum():+.1f} |")
    # descriptive: what the held fills did around a settlement ≤ −X
    L += ["", "### Hold outcome of the fills a rule touches (REF net, descriptive)", "", "| X (first settled rate ≤ −X while open) | N | REF WR | REF avg net % | REF avg funding % |", "|---|---|---|---|---|"]
    for X in ST.XS:
        v, nc, fp = res[("real", X, 1, "B1")]; t = nc > 0
        L.append(f"| {X:g} | {t.sum()} | {(C.ref[t] > 0).mean():.0%} | {C.ref[t].mean():+.2f} | {C.ref_f[t].mean():+.2f} |")
    return R_


def event_paths(d, PP, X, M=1):
    """for each fill closed by X M B1: held mark-to-market from the decision bar (s − M) incl. funding at s and later, frozen at the REF exit."""
    C = d["C"]; v, nc, fp = d["res"][("real", X, M, "B1")]
    ks = np.arange(-60, 481, 5); out = []
    for i in np.nonzero(nc > 0)[0]:
        row = C.iloc[i]; P = PP[(row.pair, int(row.entry_ms))]; st = d["SET"][i]
        t, o, c = P[0], P[1], P[4]
        T = int(row.entry_ms)
        s = next(s for s, r in zip(st[0], st[1]) if r <= -X and s - M * MIN >= T + MIN)
        refl = ST.run_rule(row, P, None, np.zeros(0, "int64"), 1, "B1") if False else None
        # REF exit from the no-rule leg
        _, E0, S0, TPp, end_t = ST.setup(row, P)
        g, k, xm, xi = ST.leg(P, 0, E0, S0, TPp, end_t)
        Xp = E0 * (1 - g / 100)
        dec = s - M * MIN; ci = int(np.searchsorted(t, dec, "right")) - 1; Pd = o[ci]
        ft, rates = st[0], st[1]
        y = []
        for kk in ks:
            tm = s + kk * MIN
            if tm < dec:
                j = int(np.searchsorted(t, tm, "right")) - 1; pr = c[max(j, 0)] if j >= 0 else np.nan
                y.append(-(pr / Pd - 1) * 100 if tm >= T else np.nan); continue
            pr = Xp if tm >= xm else c[max(int(np.searchsorted(t, tm, "right")) - 1, 0)]
            fpaid = rates[(ft > dec) & (ft <= min(tm, xm))].sum()
            y.append(-(pr / Pd - 1) * 100 + fpaid)
        out.append(y)
    return ks, np.array(out, float)


def chart(dd, pps):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (tag, X) in zip(ax[:2], (("DIST", 0.5), ("DIST", 1.0))):
        ks, Y = event_paths(dd[tag], pps[tag], X)
        m = np.nanmean(Y, 0); md = np.nanmedian(Y, 0)
        bs = np.array([np.nanmean(Y[rng.integers(0, len(Y), len(Y))], 0) for _ in range(500)])
        lo, hi = np.nanpercentile(bs, [5, 95], 0)
        a.fill_between(ks / 60, lo, hi, color="#4C78A8", alpha=.2)
        a.plot(ks / 60, m, color="#4C78A8", lw=2, label="hold: mean short P&L from the close point (incl. funding, frozen at Exit-B exit)")
        a.plot(ks / 60, md, color="#4C78A8", lw=1, ls="--", label="hold: median")
        a.axhline(0, color="#E45756", lw=1.5, label="close 1 min before (0, before the ≈0.15–0.3 % exit cost)")
        a.axvline(0, color="k", lw=.6, ls=":"); a.set_xlabel("hours from the settlement"); a.set_ylabel("% (short side)")
        a.set_title(f"{tag}: settled rate ≤ −{X:g} %, N={len(Y)} fills"); a.legend(fontsize=7, loc="lower left")
    # Δ distribution per touched fill (X 0.5 M1 B1) DIST vs EXH
    for tag, col in (("DIST", "#4C78A8"), ("EXH", "#F58518")):
        if tag not in dd:
            continue
        v, nc, _ = dd[tag]["res"][("real", 0.5, 1, "B1")]
        ax[2].hist(np.clip(v[nc > 0], -12, 12), bins=48, alpha=.55, color=col, label=f"{tag} (N={int((nc > 0).sum())}, Σ {v.sum():+.0f} %)")
    ax[2].axvline(0, color="k", lw=.6); ax[2].set_title("Δ per closed fill, X 0.5 · M1 · B1 (close − hold, %)"); ax[2].legend(fontsize=8)
    ax[2].set_xlabel("Δ net % (clipped ±12)")
    plt.tight_layout(); plt.savefig(f"{R}/reports/DIST_FUNDING_EXIT_2026-11-20.png", dpi=110)


def main():
    L = []; dd = {}; pps = {}
    for tag, sub in (("DIST", ""), ("EXH", "_EXH")):
        p = f"{R}/reports/DIST_FUNDING_EXIT_2026-11-20_data{sub}"
        dd[tag] = pickle.load(open(f"{p}/results.pkl", "rb"))
        pps[tag] = pickle.load(open(f"{p}/paths.pkl", "rb"))[0]
        sleeve_tables(tag, dd[tag], L)
    open(f"{R}/reports/DIST_FUNDING_EXIT_2026-11-20_tables.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    chart(dd, pps)


if __name__ == "__main__":
    main()
