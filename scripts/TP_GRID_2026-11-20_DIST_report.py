#!/usr/bin/env python3
"""TP_GRID_2026-11-20_DIST — STEP 2 read-out of reports/TP_GRID_2026-11-20_DIST_data/bt_rows.csv.gz (built by ..._backtest.py).
Per block (DESIGN < 2025-07 <= HOLDOUT · ALL · 2026 · each half-year) and TP X: N · WR · avg net % · sum · breakeven WR · daily compound return
(ONE size 0.61 x equity per fill, r_d = sum of the fills' net % closed that day x 0.61 / 100, every calendar day of the block, 0 on no-fill
days) · max drawdown of that compounded curve · worst day. Paired delta vs X = 10 (today) with a 10-day-block bootstrap CI;
best-of-grid luck test (centered block bootstrap of the max over the 6 alternatives, White reality-check style); leave-best-5%-out;
top-pair share; decomposition gainers (Exit B stop / time fills that reach X first) vs payers (Exit B 10 % winners cut to X).
Output: printed markdown + reports/TP_GRID_2026-11-20_DIST_data/bt_grid.csv
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_DIST_report.py [--rule 1tick]
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_DIST_data")
XS = [4, 5, 6, 7, 8, 9, 10]
SIZE = 0.61
D2026 = (pd.Timestamp("2026-01-01") - pd.Timestamp(0)).days


def dcr_curve(net, xday, d0, d1):
    days = np.arange(d0, d1 + 1)
    r = pd.Series(net * SIZE / 100).groupby(xday).sum().reindex(days, fill_value=0.0).values
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return (eq[-1] ** (1 / len(days)) - 1) * 100, (eq / peak - 1).min() * 100, r.min() * 100, len(days)


def boot(d, day, B=4000, block=10, seed=11):
    b = np.asarray(day) // block
    ub, inv = np.unique(b, return_inverse=True)
    sums = np.bincount(inv, weights=d, minlength=len(ub))
    cnt = np.bincount(inv, minlength=len(ub))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ub), (B, len(ub)))
    m = sums[ix].sum(1) / cnt[ix].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


def luck(Dm, day, B=4000, block=10, seed=13):
    """reality check: is the best alternative's mean delta larger than the best of 6 centred bootstrap draws?"""
    b = np.asarray(day) // block
    ub, inv = np.unique(b, return_inverse=True)
    S = np.stack([np.bincount(inv, weights=Dm[:, j], minlength=len(ub)) for j in range(Dm.shape[1])], 1)
    cnt = np.bincount(inv, minlength=len(ub))
    obs = Dm.mean(0)
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(ub), (B, len(ub)))
    mb = S[ix].sum(1) / cnt[ix].sum(1)[:, None] - obs[None, :]
    return obs.max(), int(obs.argmax()), (mb.max(1) >= obs.max()).mean()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--rule", default="1tick")
    A = ap.parse_args()
    D = pd.read_csv(os.path.join(OUT, "bt_rows.csv.gz"))
    ru = A.rule
    blocks = [("DESIGN", D.block == "DESIGN"), ("HOLDOUT", D.block == "HOLDOUT"), ("ALL", D.block.notna()), ("2026", D.day >= D2026)]
    blocks += [(h, D.half == h) for h in sorted(D.half.unique())]
    rows = []
    for bn, m in blocks:
        g = D[m]
        d0 = int(g.day.min()); d1 = int(max(g[[f"xday_TP{x}_{ru}" for x in XS]].max().max(), g.day.max()))
        ref = g[f"TP10_{ru}"].values
        for x in XS:
            v = g[f"TP{x}_{ru}"].values
            dcr, mdd, worst, nd = dcr_curve(v, g[f"xday_TP{x}_{ru}"].values, d0, d1)
            W = v[v > 0]; L = v[v <= 0]
            be = abs(L.mean()) / (W.mean() + abs(L.mean())) * 100 if len(W) and len(L) else np.nan
            dl = v - ref
            lo, hi, pg = boot(dl, g.day.values) if x != 10 else (0, 0, np.nan)
            kb = g.k_exitB.values; kx = g[f"k_TP{x}_{ru}"].values
            gain = (kb != 0) & (kx == 0); pay = (kb == 0) & (kx == 0) & (dl < -1e-9)
            srt = np.sort(dl)[::-1]; cut = int(np.ceil(0.05 * len(dl)))
            ps = pd.Series(dl).groupby(g.pair.values).sum()
            rows.append(dict(block=bn, X=x, N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum(), avg_win=W.mean(), avg_loss=L.mean(),
                             be_WR=be, n_tp=int((kx == 0).sum()), n_stop=int((kx == 1).sum()), n_time=int((kx == 2).sum()),
                             dcr=dcr, maxDD=mdd, worst_day=worst, days=nd, d_avg=dl.mean(), d_sum=dl.sum(), ci_lo=lo, ci_hi=hi, p_gain=pg,
                             gain_n=int(gain.sum()), gain_sum=dl[gain].sum(), pay_n=int(pay.sum()), pay_sum=dl[pay].sum(),
                             other_sum=dl[~gain & ~pay].sum(), d_avg_lo5=srt[cut:].mean(), top_pair=ps.idxmax() if x != 10 else "",
                             top_pair_d=ps.max() if x != 10 else 0.0, d_wo_top_pair=(ps.sum() - ps.max()) / len(v) if x != 10 else 0.0))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, f"bt_grid_{ru.replace('%', 'pct')}.csv"), index=False)
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    for bn, _ in blocks:
        t = T[T.block == bn]
        print(f"\n### {bn} ({ru})  N {t.N.iloc[0]} · {t.days.iloc[0]} calendar days")
        print("| TP | N | WR % | avg net % | Σ % | BE WR % | TP/stop/time | DCR %/day | maxDD % | worst day % | Δavg vs 10 [95 % CI] | P(Δ>0) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            ci = "—" if r.X == 10 else f"{r.d_avg:+.3f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]"
            pg = "—" if r.X == 10 else f"{r.p_gain:.2f}"
            print(f"| {r.X:g}{' (today)' if r.X == 10 else ''} | {r.N} | {r.WR:.1f} | {r.avg:+.3f} | {r['sum']:+.0f} | {r.be_WR:.1f} | "
                  f"{r.n_tp}/{r.n_stop}/{r.n_time} | {r.dcr:+.3f} | {r.maxDD:.1f} | {r.worst_day:.1f} | {ci} | {pg} |")
    print(f"\n### Decomposition vs today ({ru}) — Σ Δ net % by group")
    print("| block | TP | gainers N | gainers ΣΔ | payers N | payers ΣΔ | other ΣΔ | net ΣΔ | Δavg w/o best 5 % | top pair (ΣΔ) | Δavg w/o top pair |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in T[(T.X != 10) & T.block.isin(["DESIGN", "HOLDOUT", "ALL"])].iterrows():
        print(f"| {r.block} | {r.X:g} | {r.gain_n} | {r.gain_sum:+.0f} | {r.pay_n} | {r.pay_sum:+.0f} | {r.other_sum:+.0f} | {r.d_sum:+.0f} | "
              f"{r.d_avg_lo5:+.3f} | {r.top_pair} ({r.top_pair_d:+.0f}) | {r.d_wo_top_pair:+.3f} |")
    print(f"\n### Best-of-grid luck test ({ru}) — max mean Δ over the 6 alternatives vs centred 10-day-block bootstrap")
    for bn, m in blocks[:4]:
        g = D[m]
        Dm = np.stack([g[f"TP{x}_{ru}"].values - g[f"TP10_{ru}"].values for x in XS[:-1]], 1)
        best, j, p = luck(Dm, g.day.values)
        print(f"- {bn}: best TP {XS[j]} mean Δ {best:+.3f} %/fill, reality-check p = {p:.3f}")


if __name__ == "__main__":
    main()
