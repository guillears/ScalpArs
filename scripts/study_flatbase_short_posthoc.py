#!/usr/bin/env python3
"""FRENZY flat-base short (2026-10-22) — POST-HOC diagnostic (NOT pre-registered; refute-only, can never support a ship).

Why: every pre-registered entry lost to random bars in the same flat episodes (random-bar mean ≈ +2 %/trade vs +0.3…+1.2). This asks WHEN
inside the episode a short does best: fixed delays after the spike bar (2 / 6 / 12 / 24 / 48 h), same universe gates, same exits, costs +
funding, flat vs non-flat, year block, plus the 2025 / 26-H1 / 26-H2 split, best-5 % removal and the day-bootstrap CI.
Writes reports/FRENZY_FLATBASE_SHORT_2026-10-22_posthoc.csv and prints a table.
Usage: venv/bin/python scripts/study_flatbase_short_posthoc.py
"""
import os, sys
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402
import study_flatbase_short_sim as SS  # noqa: E402

DELAYS = [2, 6, 12, 24, 48, 72, 96]
JS = {"7d|tp_base|stop25|1x": C.EXITS.index((7, "base", 25.0, 1)), "14d|tp_none|stop0|1x": C.EXITS.index((14, "none", 0.0, 1)),
      "7d|tp_none|stop25|1x": C.EXITS.index((7, "none", 25.0, 1))}


def worker(args):
    pair, eps = args
    full = C.load_full(pair)
    if full is None:
        return []
    t, o, h, l, c, v, qv, real = full
    n = len(t)
    F = C.funding_table(pair)
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    rows, idx = [], []
    for e in eps.itertuples():
        on = int((e.on_t - t[0]) // C.BAR)
        fs = int((e.first_seen_t - t[0]) // C.BAR) if e.first_seen_t is not None and not pd.isna(e.first_seen_t) else 10**12
        for d in DELAYS:
            i = on + d * 12
            if 0 <= i < n - 1 and i >= fs and vol24[i] >= 20e6:
                rows.append(dict(pair=pair, on_t=e.on_t, delay=d, entry_ms=int(t[i]) + C.BAR, E=c[i] * (1 - C.SLIP), base=e.base,
                                 run=(h[on:i + 1].max() / e.base - 1) * 100, off_peak=(c[i] / h[on:i + 1].max() - 1) * 100))
                idx.append(i)
    if not rows:
        return []
    R = pd.DataFrame(rows)
    H, Lo, O, Cc, T, av = SS.paths((t, o, h, l, c), np.array(idx), n)
    G, XT, RS = SS.simulate(R.E.values, R.base.values, H, Lo, O, Cc, T, av, False)
    for nm, j in JS.items():
        fu, _ = C.funding_sum(F, R.entry_ms.values, XT[:, j])
        R[nm] = G[:, j] - C.FEES + fu
        R[nm + "_nofund"] = G[:, j] - C.FEES
    return R.to_dict("records")


if __name__ == "__main__":
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    E = E[(E.spike >= C.B_YEAR0) & (E.spike < C.B_YEAR1)]
    thr = E[E.spike < C.FREEZE_CUT].FLAT14.quantile(1 / 3)
    lab = dict(zip(zip(E.pair, E.on_t), np.where(E.FLAT14.isna(), "unscored", np.where(E.FLAT14 <= thr, "flat", "nonflat"))))
    out = []
    with mp.Pool(7) as pool:
        for r in pool.imap_unordered(worker, [(p, g) for p, g in E.groupby("pair")]):
            out += r
    X = pd.DataFrame(out)
    X["label"] = [lab[k] for k in zip(X.pair, X.on_t)]
    X["dt"] = pd.to_datetime(X.entry_ms, unit="ms"); X["day"] = X.dt.dt.strftime("%Y-%m-%d")
    X["period"] = np.where(X.dt < pd.Timestamp("2026-01-01"), "2025", np.where(X.dt < pd.Timestamp("2026-07-01"), "2026H1", "2026H2"))
    X.to_csv(C.OUTP + "_posthoc.csv", index=False)
    rng = np.random.default_rng(5)

    def ci(g, col):
        d = g.groupby("day")[col].agg(["sum", "count"]); s, cn = d["sum"].values, d["count"].values
        i = rng.integers(0, len(d), (2000, len(d))); m = s[i].sum(1) / cn[i].sum(1)
        return np.percentile(m, 2.5), np.percentile(m, 97.5)
    print("| exit | delay | cohort | N | WR | avg | 95 % day-boot | pre-funding | 2025 / 26-H1 / 26-H2 | best 5 % removed |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for nm in JS:
        for d in DELAYS:
            for coh in ("flat", "nonflat"):
                g = X[(X.delay == d) & (X.label == coh)]
                lo, hi = ci(g, nm); k = int(np.ceil(len(g) * 0.05))
                per = " / ".join(f"{g[g.period == p][nm].mean():+.2f}" for p in ("2025", "2026H1", "2026H2"))
                print(f"| {nm} | {d} h | {coh} | {len(g)} | {(g[nm] > 0).mean() * 100:.0f}% | {g[nm].mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] | "
                      f"{g[nm + '_nofund'].mean():+.2f} | {per} | {np.sort(g[nm].values)[:-k].mean():+.2f} |")
