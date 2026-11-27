#!/usr/bin/env python3
"""FLATBASE_THEORY_2026-11-27 — is the FLATBASE trigger shorting what the theory says? (operator: "it was supposed to short a coin that pumped,
that was previously FLAT … focus on theory, see if the trigger is correct, or the difference in winners vs losers, in master, current +
backtest"). Research only.

THEORY. A coin that sat FLAT (no trend, low noise) and then pumped on a one-off event has no structural buyer: once the pump fails (first 4-h
close below the spike VWAP, confirmed), price drifts back toward the old base → short it. Today's FLAT = max / min close − 1 ≤ 44.4 % over the
14 days ending 30 min before the spike (flatbase.flat14_from_bars). A 44 % range is wide: it admits coins that TRENDED (slid / climbed) or
CHOPPED (swung every day) inside the band — not "flat" in the theory's sense.
PRIOR: flat14 tested as a W/L variable (OOS_REAL 11-17; inconsistent across halves; real losers flatter than winners) · DL 649 falling base
(no edge) · DL 649 entry timing (keep). NEW: alternative "flat" measures, pre-registered with the predicted direction BEFORE any result:
  F1 range (today's flat14)                         lower → better
  F2 base daily volatility (std of daily log returns) lower → better
  F3 |base trend| = |close(end) / close(start) − 1|  lower → better
  F4 quiet-day share = days with high / low < 6 %    higher → better
  F5 trend R² of log price over the 14 days          lower → better
  F6 pump vs base noise = run_pct ÷ F2               higher → better (a true anomaly)
Cohorts (today's exit, today's filters): BACKTEST 545 (reports/FLATBASE_NOPROGRESS_STOP_2026-11-27_data/base.csv BT, rebuild features) ·
REAL = KEPT_LIST master 17 + batch kept 5 + the open MOVE / LAYER (reports/FLATBASE_FUNDING_RERUN_2026-11-27_data/real.pkl), each mapped to
its rebuild episode (pair, confirm entry ± 10 min) for the spike bar. Base = the 5m closes of the 14 days ending at the base bar (spike − 30 min).
Read: terciles ALL / DESIGN (< 2025-07) / HOLDOUT + half-year signs; real winners vs losers medians + the backtest-median split.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, R)
import oos_real_flatbase_recon as RC   # noqa: E402  (sets the walk's 5m sources incl. the 11-27 top-up)
W = RC.W
BAR, DAY = 300_000, 86_400_000
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)
OUT = os.path.join(R, "reports", "FLATBASE_THEORY_2026-11-27")
METRICS = [("F1_range", -1, "14-day range % (today's flat14)"), ("F2_vol", -1, "base daily volatility %"),
           ("F3_trend_abs", -1, "|base trend| %"), ("F4_quiet", +1, "quiet-day share (daily range < 6 %)"),
           ("F5_r2", -1, "trend R² of the base"), ("F6_pump_vs_noise", +1, "pump size ÷ base volatility")]
_CACHE = {}


def base_metrics(pair, on_t, run_pct):
    if pair not in _CACHE:
        _CACHE[pair] = W.load(pair)
    x = _CACHE[pair]
    if x is None:
        return None
    t, o, h, l, c = x[0], x[1], x[2], x[3], x[4]
    on = int(np.searchsorted(t, int(on_t)))
    if on >= len(t) or int(t[on]) != int(on_t):
        return None
    b = on - 6
    if b - 4031 < 0:
        return None
    sl = slice(b - 4031, b + 1)
    cc, hh, ll, tt = c[sl], h[sl], l[sl], t[sl]
    d = pd.DataFrame(dict(t=tt, h=hh, l=ll, c=cc))
    d["day"] = (d.t - d.t.iloc[0]) // DAY
    g = d.groupby("day").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"))
    if len(g) < 12:
        return None
    lr = np.diff(np.log(g.c.values))
    y = np.log(cc); xx = np.arange(len(y))
    r2 = np.corrcoef(xx, y)[0, 1] ** 2 if np.std(y) > 0 else 0.0
    vol = float(np.std(lr) * 100)
    return dict(F1_range=(cc.max() / cc.min() - 1) * 100, F2_vol=vol, F3_trend_abs=abs(cc[-1] / cc[0] - 1) * 100,
                F3_trend=(cc[-1] / cc[0] - 1) * 100, F4_quiet=float(((g.h / g.l - 1) < 0.06).mean()), F5_r2=float(r2),
                F6_pump_vs_noise=(float(run_pct) / vol) if (vol > 0 and run_pct == run_pct) else np.nan)


def main():
    os.makedirs(OUT + "_data", exist_ok=True)
    b = pd.read_csv(os.path.join(R, "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27_data", "base.csv"))
    b = b[b.src == "BT"]
    w = pd.read_csv(os.path.join(R, "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    B = b.merge(w[["key", "on_t", "run_pct"]], left_on="ep", right_on="key", how="left")
    rows = []
    for r in B.itertuples():
        m = base_metrics(r.pair, r.on_t, r.run_pct)
        if m:
            rows.append(dict(set="BT", pair=r.pair, te=int(r.te), net=float(r.net), **m))
    # real: map each fill to its rebuild episode
    E = pd.concat([pd.read_pickle(os.path.join(R, "reports", "OOS_REAL_FLATBASE_data", "recon.pkl")),
                   pd.read_pickle(os.path.join(R, "reports", "FLATBASE_REBUILD_2026-11-27_data", "recon_1127.pkl"))])
    E = E[E.cf_entry_t.notna()]
    real = pd.read_pickle(os.path.join(R, "reports", "FLATBASE_FUNDING_RERUN_2026-11-27_data", "real.pkl"))
    miss = []
    for r in real.itertuples():
        e = E[(E.pair == r.pair) & ((E.cf_entry_t - r.te).abs() <= 10 * 60_000)]
        if not len(e):
            e = E[(E.pair == r.pair) & ((E.cf_entry_t - r.te).abs() <= 6 * 3_600_000)]
        if not len(e):
            miss.append(r.pair); continue
        e = e.iloc[0]
        m = base_metrics(r.pair, e.on_t, e.run_pct)
        if m:
            rows.append(dict(set=("OPEN" if r.open else r.src), pair=r.pair, te=int(r.te), net=float(r.base_net), **m))
        else:
            miss.append(r.pair)
    D = pd.DataFrame(rows)
    D.to_csv(OUT + "_data/metrics.csv", index=False)
    bt = D[D.set == "BT"].copy(); real = D[D.set != "BT"].copy()
    print(f"BACKTEST {len(bt)} · REAL {len(real)} (master {int((real.set == 'MASTER').sum())} · batch {int((real.set == 'BATCH').sum())} · "
          f"open {int((real.set == 'OPEN').sum())}) · unmapped {miss}")
    bt["half"] = pd.to_datetime(bt.te, unit="ms").dt.year.astype(str) + "-H" + np.where(pd.to_datetime(bt.te, unit="ms").dt.month <= 6, "1", "2")
    out = []
    for col, sgn, lab in METRICS:
        v = bt[col]
        lo, hi = v.quantile([1 / 3, 2 / 3])
        good = (v <= lo) if sgn < 0 else (v > hi); bad = (v > hi) if sgn < 0 else (v <= lo); mid = ~good & ~bad
        row = dict(metric=lab, theory="low better" if sgn < 0 else "high better", cut_lo=round(lo, 3), cut_hi=round(hi, 3),
                   good=f"{int(good.sum())} · {(bt.net[good] > 0).mean()*100:.0f}% · {bt.net[good].mean():+.2f}",
                   mid=f"{int(mid.sum())} · {bt.net[mid].mean():+.2f}",
                   bad=f"{int(bad.sum())} · {(bt.net[bad] > 0).mean()*100:.0f}% · {bt.net[bad].mean():+.2f}")
        for blk, m in (("DES", bt.te < HOL), ("HOL", bt.te >= HOL)):
            row[f"gap_{blk}"] = round(bt.net[m & good].mean() - bt.net[m & bad].mean(), 2)
        hv = [bt.net[(bt.half == h) & good].mean() - bt.net[(bt.half == h) & bad].mean() for h in sorted(bt.half.unique())]
        row["halves_agree"] = f"{sum(x > 0 for x in hv if x == x)}/{sum(1 for x in hv if x == x)}"
        rw, rl = real[real.net > 0], real[real.net <= 0]
        med = v.median()
        rg = (real[col] <= med) if sgn < 0 else (real[col] > med)
        row["real_W_med"] = round(rw[col].median(), 3); row["real_L_med"] = round(rl[col].median(), 3)
        row["real_theory_side"] = f"{int(rg.sum())} · {(real.net[rg] > 0).mean()*100:.0f}% · {real.net[rg].mean():+.2f}"
        row["real_other_side"] = f"{int((~rg).sum())} · {(real.net[~rg] > 0).mean()*100:.0f}% · {real.net[~rg].mean():+.2f}"
        out.append(row)
    T = pd.DataFrame(out)
    T.to_csv(OUT + "_data/table.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(T.to_string(index=False))
    print("\nREAL fills (base metrics):")
    print(real[["set", "pair", "net", "F1_range", "F2_vol", "F3_trend", "F4_quiet", "F5_r2", "F6_pump_vs_noise"]].round(2).sort_values("net").to_string(index=False))


if __name__ == "__main__":
    main()
