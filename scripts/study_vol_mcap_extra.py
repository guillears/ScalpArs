#!/usr/bin/env python3
"""📊 Oct-8 research (vol24h / mcap) — robustness add-ons on reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv (no network):
  (a) supply growth: CoinGecko implied circulating supply (market_cap / price) at each fill vs CG's latest point — how wrong the
      constant-supply reconstruction is by month (definition-free: both ends from the same source)
  (b) is R just market cap or just volume? the same threshold scan + shuffled null on log mcap and log vol alone
  (c) a stricter null for the primary R: labels shuffled WITHIN each day (keeps day effects) — 1,000×, same scan
  (d) primary R with today's supply corrected by the CG supply ratio where CG covers the pair (sensitivity)
Usage: venv/bin/python scripts/study_vol_mcap_extra.py"""
import json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("S", "/tmp"); os.environ.setdefault("MAPF", "reports/cache_vol_mcap/mapping.json")
import study_vol_mcap_analyze as A  # noqa: E402

F = pd.read_csv("reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv", low_memory=False)
RNG = np.random.default_rng(7)


def supply_ratio(pair, t):
    c = A.cg_series(pair)
    if c is None:
        return np.nan
    j = json.load(open(f"reports/cache_vol_mcap/chart/{A.MAP[pair]['id']}.json"))
    a = pd.DataFrame(j["market_caps"], columns=["t", "mc"]).merge(pd.DataFrame(j["prices"], columns=["t", "p"]), on="t")
    a = a[(a.mc > 0) & (a.p > 0)]; s = (a.mc / a.p).values; tt = a.t.values
    i = np.searchsorted(tt, t, side="right") - 1
    return s[-1] / s[i] if i >= 0 else np.nan


def within_day_null(T, col, minn, n=1000):
    x, y = T[col].values, T.pct.values; cuts = A.cuts_for(x); d0, c, blk = A.scan(x, y, cuts, minn)
    groups = [np.flatnonzero(T.day.values == d) for d in np.unique(T.day.values)]; cnt = 0
    for _ in range(n):
        yy = y.copy()
        for g in groups:
            if len(g) > 1:
                yy[g] = y[RNG.permutation(g)]
        cnt += abs(A.scan(x, yy, cuts, minn)[0]) >= abs(d0)
    return d0, c, blk, (cnt + 1) / (n + 1)


print("## Robustness add-ons\n")
for coh, minn in (("frenzy_today", 25), ("frenzy_today_bear", 25), ("frenzy_gated", 25), ("willyA", 40), ("willyB", 40)):
    T = F[F.cohort == coh].copy(); T["t0"] = T.t0.astype("int64")
    T["lR"] = np.log10(T.R); T["lmcap"] = np.log10(T.mcap_b); T["lvol"] = np.log10(T.vol_fut)
    U = T[T.lR.notna()].copy()
    print(f"### {coh} (scored {len(U)})\n")
    d0, c, blk, p = within_day_null(U, "lR", minn)
    print(f"- primary R, within-day shuffled null: best cut block {blk} R {'≤' if blk == 'low' else '>'} {10 ** c:.3g}, Δ {d0:+.3f}, **p = {p:.3f}**")
    for col, lab in (("lmcap", "market cap alone"), ("lvol", "24h futures volume alone")):
        V = U[U[col].notna()]; x, y = V[col].values, V.pct.values; cuts = A.cuts_for(x); d, cc, b = A.scan(x, y, cuts, minn)
        pn = A.null_p(x, y, cuts, minn, d)
        qs = pd.qcut(V[col].rank(method="first"), 5, labels=False); qm = V.groupby(qs).pct.mean().round(3).tolist()
        print(f"- {lab}: quintile avgs {qm} · best cut block {b} {col} {'≤' if b == 'low' else '>'} {10 ** cc:.3g} Δ {d:+.3f}, shuffled p = {pn:.3f}")
    # (a) supply growth & (d) corrected R
    U["sg"] = [supply_ratio(p_, t_) for p_, t_ in zip(U.pair, U.t0)]
    s = U[U.sg.notna()].copy(); s["m"] = s.day.str[:7]
    print(f"- CG implied supply today / at fill (covered {len(s)}): median {s.sg.median():.3f}, IQR {s.sg.quantile(.25):.3f}–{s.sg.quantile(.75):.3f}, "
          f"share > 1.25: {(s.sg > 1.25).mean() * 100:.0f}% · by month median: " + ", ".join(f"{m} {g.sg.median():.2f}" for m, g in s.groupby("m")))
    if len(s) >= 2 * minn:
        s["lRc"] = s.lR + np.log10(s.sg)       # mcap at fill = today's supply / growth × price → R up by the growth factor
        print(f"- corrected vs uncorrected log R Spearman {A.sp(s.lR, s.lRc):+.3f}; on the covered subset: uncorrected Spearman(R, P&L) "
              f"{A.sp(s.lR, s.pct):+.3f}, corrected {A.sp(s.lRc, s.pct):+.3f}")
    print()
