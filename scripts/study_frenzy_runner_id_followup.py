#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 — follow-up reads of the OOS survivors (window units, sign granularity, pricing as a SIZING rule).

Frozen before reading: the survivors' market-wide content is read at SIGN granularity on the STAMPED columns the engine already has
(entry_btc_1d_ret_pct, entry_btc_r72_pct) and on the rebuilt b_ret24, in DAY units (window-units rule); pair-level survivor (WILLY
h_4h_gap20_50 > 0) as found. Sizing: zone at 1× / 2× / 3× on today's exit (x0) and the runner exits (r1, r2 in the zone only), $ at the
sleeve notional (LITE $2,000 · WILLY $2,250 · LONG priced at $2,000 as if sized up from the $100 blocker), worst UTC day and max
drawdown of the sleeve's daily $ path, per period; Pattern-W + BE-compat; per-day / per-pair share of the zone's gain.
Usage: venv/bin/python scripts/study_frenzy_runner_id_followup.py <scratch>
"""
import os, sys
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"; os.chdir(ROOT)
S = sys.argv[1]
OUT = "reports/FRENZY_RUNNER_ID_2026-10-19"
RNG = np.random.default_rng(7)
NOT = {"FRENZY_LONG": 2000.0, "FRENZY_LITE": 2000.0, "FRENZY_WILLY": 2250.0}
D = pd.read_pickle(f"{S}/cohort.pkl")
D = D[~((D.blk == "BT") & (D.ts >= "2026-10-03"))].copy()
D["day"] = D.ts.dt.strftime("%Y-%m-%d")
D["per"] = np.where(D.blk != "BT", "REAL", np.where(D.ts < "2026-01-01", "2025", np.where(D.ts < "2026-07-01", "26H1", "26H2")))
D["run5"] = (D.mfe_720 >= 5).astype(float)
for c in ("entry_btc_1d_ret_pct", "entry_btc_r72_pct", "b_ret24", "h_4h_gap20_50", "entry_frenzy_vol_mult", "entry_frenzy_vs_vwap_pct"):
    D[c] = pd.to_numeric(D[c], errors="coerce")

CELLS = [
    ("FRENZY_LITE", "BTC 24 h return > 0 (stamp entry_btc_1d_ret_pct)", lambda X: X.entry_btc_1d_ret_pct > 0, ["entry_btc_1d_ret_pct"]),
    ("FRENZY_LITE", "BTC 24 h return > 0 (rebuilt b_ret24)", lambda X: X.b_ret24 > 0, ["b_ret24"]),
    ("FRENZY_LITE", "BTC 72 h return > 0 (stamp entry_btc_r72_pct)", lambda X: X.entry_btc_r72_pct > 0, ["entry_btc_r72_pct"]),
    ("FRENZY_LITE", "BTC 24 h > +1.217 (screen cut, hi tercile)", lambda X: X.b_ret24 > 1.217, ["b_ret24"]),
    ("FRENZY_WILLY", "pair 4h EMA20 > EMA50 (h_4h_gap20_50 > 0)", lambda X: X.h_4h_gap20_50 > 0, ["h_4h_gap20_50"]),
    ("FRENZY_LONG", "volume multiple > 122.6 (2025 hi tercile)", lambda X: X.entry_frenzy_vol_mult > 122.6, ["entry_frenzy_vol_mult"]),
    ("FRENZY_LONG", "BTC 24 h return > 0 (stamp)", lambda X: X.entry_btc_1d_ret_pct > 0, ["entry_btc_1d_ret_pct"]),
]


def boot_mean(x, day, n=3000):
    df = pd.DataFrame(dict(x=x, d=day)).dropna()
    if len(df) < 3:
        return np.nan, np.nan, np.nan
    g = df.groupby("d").x.agg(["sum", "count"]); s, c = g["sum"].values, g["count"].values
    idx = RNG.integers(0, len(g), (n, len(g)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float((m > 0).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_diff(z, x, day, n=3000):
    df = pd.DataFrame(dict(z=z, x=x, d=day)).dropna()
    g = df.groupby(["d", "z"]).x.agg(["sum", "count"]).unstack(fill_value=0)
    s1, c1, s0, c0 = (g[(a, b)].values if (a, b) in g else np.zeros(len(g)) for a, b in (("sum", True), ("count", True), ("sum", False), ("count", False)))
    idx = RNG.integers(0, len(g), (n, len(g)))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = s1[idx].sum(1) / c1[idx].sum(1) - s0[idx].sum(1) / c0[idx].sum(1)
    d = d[np.isfinite(d)]
    return float((d > 0).mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def path(x, col_zone, mult, notional):
    usd = np.where(x.zone, x[col_zone] * mult, x.x0) / 100 * notional
    d = pd.Series(usd, index=x.index).groupby(x.day).sum().sort_index()
    eq = d.cumsum()
    return float(np.nansum(usd)), float(d.min()), float((eq - eq.cummax()).min())


rows, prices = [], []
for sv, name, fn, cols in CELLS:
    X = D[D.sleeve == sv].copy()
    sc = X[cols].notna().all(axis=1)
    X = X[sc].copy(); X["zone"] = fn(X).values
    for per in ("2025", "26H1", "26H2", "2026", "BT_ALL", "REAL"):
        x = X[(X.per == per) | ((per == "2026") & X.per.isin(["26H1", "26H2"])) | ((per == "BT_ALL") & (X.blk == "BT"))]
        if len(x) < 5:
            continue
        zz, rr = x[x.zone], x[~x.zone]
        e = dict(sleeve=sv, cell=name, period=per, scored=len(x), n_zone=len(zz), zone_days=zz.day.nunique(), n_rest=len(rr),
                 run5_zone=zz.run5.mean() * 100, run5_rest=rr.run5.mean() * 100,
                 x0_zone=zz.x0.mean(), x0_rest=rr.x0.mean(), r1_zone=zz.r1.mean(), r2_zone=zz.r2.mean(),
                 wr_x0_zone=(zz.x0 > 0).mean() * 100, wr_r1_zone=(zz.r1 > 0).mean() * 100)
        if len(zz) >= 5 and len(rr) >= 5:
            P, lo, hi = boot_diff(x.zone.values, x.x0.values, x.day.values)
            e.update(P_x0_zone_gt_rest=P, d_x0_lo=lo, d_x0_hi=hi)
            P, lo, hi = boot_diff(x.zone.values, x.run5.values, x.day.values)
            e.update(P_run5_zone_gt_rest=P)
        for col in ("x0", "r1"):
            P, lo, hi = boot_mean(zz[col].values, zz.day.values)
            e[f"{col}_zone_P_pos"] = P; e[f"{col}_zone_ci"] = f"[{lo:+.2f}, {hi:+.2f}]" if np.isfinite(lo) else ""
        g = zz.groupby("day").x0.sum()
        e["top_day_share_of_zone_gain"] = float(g.max() / g[g > 0].sum()) if (g > 0).any() else np.nan
        gp = zz.groupby("pair").x0.sum()
        e["top_pair_share_of_zone_gain"] = float(gp.max() / gp[gp > 0].sum()) if (gp > 0).any() else np.nan
        e["agt_in_zone"] = int((zz.pair == "AGTUSDT").sum())
        rows.append(e)
        if per in ("2025", "2026", "BT_ALL"):
            for col in ("x0", "r1", "r2"):
                for k in (1, 2, 3):
                    u, wd, dd = path(x, col, k, NOT[sv])
                    u0, wd0, dd0 = path(x.assign(zone=False), "x0", 1, NOT[sv])
                    los = zz[zz[col] <= 0]
                    prices.append(dict(sleeve=sv, cell=name, period=per, zone_exit=col, mult=k, n_zone=len(zz),
                                       zone_WR=(zz[col] > 0).mean() * 100, zone_avg=zz[col].mean(), zone_sum=zz[col].sum(),
                                       patternW=bool(len(zz) >= 30 and (zz[col] > 0).mean() >= .7 and zz[col].mean() >= .1 and zz[col].sum() > 0),
                                       be_compat=(los.mfe_720 >= 0.2).mean() * 100 if len(los) else np.nan,
                                       usd=u, usd_today=u0, d_usd=u - u0, worst_day=wd, worst_day_today=wd0, maxdd=dd, maxdd_today=dd0))
F = pd.DataFrame(rows).round(3); F.to_csv(f"{OUT}_followup.csv", index=False)
Pz = pd.DataFrame(prices).round(2); Pz.to_csv(f"{OUT}_followup_pricing.csv", index=False)
pd.set_option("display.width", 320)
print(F.to_string())
print(Pz[(Pz.period != "BT_ALL")].to_string())
