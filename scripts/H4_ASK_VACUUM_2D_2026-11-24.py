#!/usr/bin/env python3
"""H4 ask vacuum — exhaustive 2D winners-vs-losers screen + day-by-day market context (2026-11-24, INTERIM, research only, PREREG Addendum 8).

Operator: "you did a proper 2d analysis on winners vs losers? what was different on 11-20 vs 11-21?"
Outcome: net % at TP 5 / stop 3 / 24 h (the interim best cell) and win (net > 0). Every pair of recorded entry variables (median splits,
coverage ≥ 30 %), all 4 quadrants: the quadrant's mean vs the rest of the rows scored on both (Welch t), cell N ≥ 15 and rest N ≥ 15
(cell days are reported for the top cells, not filtered). Non-finite t (zero-variance cells) is dropped on BOTH the observed and null side. Luck test: rotation null (outcome series rotated by ≥ one day of rows, 1,000×), family max |t| over every
quadrant of every pair. Day table: per UTC day medians of the market-wide stamps + the trigger's and the random minutes' outcome.
Run: venv/bin/python -I scripts/H4_ASK_VACUUM_2D_2026-11-24.py → reports/H4_ASK_VACUUM_2D_2026-11-24_{screen,days}.csv
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b_signals"))
import interim_q3 as I  # noqa: E402

_spec = importlib.util.spec_from_file_location("S", os.path.join(ROOT, "scripts", "H4_ASK_VACUUM_STUDY_2026-11-24.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
OUT = os.path.join(ROOT, "reports", "H4_ASK_VACUUM_2D_2026-11-24")
CELL = (5, 3, 1440)
MIN_CELL = 15
N_ROT = 1000
SEED = 20261124


def nets(df, now_ms):
    S.HOLDS, S.TPS, S.SLS = (CELL[2],), (CELL[0],), (CELL[1],)
    return np.array([S.walk_row(r, now_ms).get(CELL, np.nan) for r in df.itertuples()])


def quad_t(H, Cv, y):
    """all-pairs quadrant t. H = {0,1} hi masks (n × V, 0 where unscored), Cv = scored masks (n × V). → dict q → (t, n_cell, n_rest,
    mean_cell, mean_rest) arrays V × V for the 4 quadrants (hi/hi, hi/lo, lo/hi, lo/lo)."""
    L = (1 - H) * Cv
    y0 = np.nan_to_num(y)
    ok = (~np.isnan(y)).astype(float)
    Hs, Ls, Cs = H * ok[:, None], L * ok[:, None], Cv * ok[:, None]
    both_n = Cs.T @ Cv
    both_s = (Cs * y0[:, None]).T @ Cv
    both_q = (Cs * (y0 ** 2)[:, None]).T @ Cv
    out = {}
    for name, A, B in (("hh", Hs, H), ("hl", Hs, L), ("lh", Ls, H), ("ll", Ls, L)):
        n = A.T @ B
        s = (A * y0[:, None]).T @ B
        q = (A * (y0 ** 2)[:, None]).T @ B
        rn, rs, rq = both_n - n, both_s - s, both_q - q
        with np.errstate(divide="ignore", invalid="ignore"):
            m1, m2 = s / n, rs / rn
            v1 = (q - n * m1 ** 2) / (n - 1)
            v2 = (rq - rn * m2 ** 2) / (rn - 1)
            t = (m1 - m2) / np.sqrt(v1 / n + v2 / rn)
        out[name] = (t, n, rn, m1, m2)
    return out


def cell_days(H, Cv, day, a, b, name):
    ha = H[:, a] if name[0] == "h" else (1 - H[:, a]) * Cv[:, a]
    hb = H[:, b] if name[1] == "h" else (1 - H[:, b]) * Cv[:, b]
    return len(set(day[(ha * hb) > 0]))


def main():
    T, C, meta = I.load()
    t, c = I.cohort(T, C, pd.Timestamp(meta["data_end"]))
    t, c = t[t.trigger == "H4_ask_vacuum"].copy(), c[c.trigger == "H4_ask_vacuum"].copy()
    now_ms = int(pd.Timestamp.now("UTC").value // 1_000_000)
    t["net"] = nets(t, now_ms)
    c["net"] = nets(c, now_ms)
    t = t.dropna(subset=["net"]).sort_values("ms").reset_index(drop=True)
    t["win"] = (t.net > 0).astype(float)
    t["ask_bid_ratio_05"] = t.rb_ask_usd_05 / t.rb_bid_usd_05
    t["depth_05_total"] = t.rb_ask_usd_05 + t.rb_bid_usd_05
    num = [x for x in t.columns if x.startswith(("rb_", "mk_", "oi_", "top_", "global_", "taker_", "fb_"))
           and pd.api.types.is_numeric_dtype(t[x]) and t[x].notna().mean() >= 0.3 and t[x].nunique() > 3
           and x not in ("rb_mid", "rb_mark")] + ["ask_bid_ratio_05", "depth_05_total"]
    X = t[num].to_numpy(float)
    Cv = (~np.isnan(X)).astype(float)
    med = np.nanmedian(X, 0)
    H = np.where(np.isnan(X), 0.0, (X > med).astype(float))
    day = t.day.to_numpy()
    iu = np.triu_indices(len(num), 1)
    day_rows = int(np.ceil(len(t) / t.day.nunique()))
    rng = np.random.default_rng(SEED)
    rows = []
    for ycol in ("net", "win"):
        y = t[ycol].to_numpy(float)
        res = quad_t(H, Cv, y)
        mx = np.empty(N_ROT)
        for i in range(N_ROT):
            yr = np.roll(y, rng.integers(day_rows, len(y) - day_rows))
            rr = quad_t(H, Cv, yr)
            best = 0.0
            for name, (tt, n, rn, *_rest) in rr.items():
                ok = (n[iu] >= MIN_CELL) & (rn[iu] >= MIN_CELL) & np.isfinite(tt[iu])
                v = np.abs(tt[iu][ok])
                if v.size and np.isfinite(v).any():
                    best = max(best, np.nanmax(v))
            mx[i] = best
        bar = float(np.percentile(mx, 95))
        for name, (tt, n, rn, m1, m2) in res.items():
            for a, b in zip(*iu):
                if n[a, b] >= MIN_CELL and rn[a, b] >= MIN_CELL and np.isfinite(tt[a, b]):
                    rows.append(dict(outcome=ycol, var_a=num[a], var_b=num[b], quadrant=name, n_cell=int(n[a, b]),
                                     n_rest=int(rn[a, b]), mean_cell=m1[a, b], mean_rest=m2[a, b], t=tt[a, b], null95=bar,
                                     cut_a=med[a], cut_b=med[b]))
    R = pd.DataFrame(rows)
    R["abs_t"] = R.t.abs()
    R = R.sort_values(["outcome", "abs_t"], ascending=[True, False])
    top = R.groupby("outcome").head(40).copy()
    top["cell_days"] = [cell_days(H, Cv, day, num.index(r.var_a), num.index(r.var_b), r.quadrant) for r in top.itertuples()]
    top.to_csv(OUT + "_screen.csv", index=False)

    # ── day table: market-wide context per UTC day (window units: one row per day) ──
    t["d"] = pd.to_datetime(t.minute, unit="ms", utc=True).dt.strftime("%m-%d")
    c["d"] = pd.to_datetime(c.minute, unit="ms", utc=True).dt.strftime("%m-%d")
    mk = [x for x in num if x.startswith("mk_")] + ["rb_ret_60m_pct", "rb_ret_15m_pct", "rb_funding_pct", "rb_premium_pct",
                                                    "rb_spread_bps", "rb_ask_usd_05", "rb_bid_usd_05", "rb_traded_vs_norm_15",
                                                    "rb_taker_buy_share_15"]
    D = t.groupby("d").agg(trades=("net", "size"), coins=("pair", "nunique"), net=("net", "mean"), WR=("win", "mean"))
    D["random_same_coin"] = c.dropna(subset=["net"]).groupby("d").net.mean()
    D = D.join(t.groupby("d")[mk].median())
    D.T.to_csv(OUT + "_days.csv")
    print("bar", R.groupby("outcome").null95.first().round(2).to_dict())


if __name__ == "__main__":
    main()
