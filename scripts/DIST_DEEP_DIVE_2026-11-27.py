#!/usr/bin/env python3
"""DIST_DEEP_DIVE_2026-11-27 — DISTRIBUTION_SHORT after 10 live fills at −1.01 %/fill (operator: "run the DIST deep dive now"; DIST is a
disable candidate in the operator's view). Research only, no bot change. Prior studies (not re-run): DIST_WL 11-18 · DL 602 / 603 / 627 /
639 / 645 / 649. NEW here:
  A  how unusual 10 fills · 30 % · −1.01 is against DIST's own 2-year kept cohort (rolling 10-trade windows, in time order)
  B  per-half-year / per-month stability of the kept cohort (does DIST have losing regimes, and do they persist?)
  C  CLAUDE.md sleeve-kill checklist ② — market variables at SIGN granularity first, then buckets, in DAY units: BTC daily regime,
     BTC 1 d return, BTC 1 h return, breadth, BTC 30-day drawdown (NEW for DIST), BTC above its 24 h low (NEW for DIST)
  D  are the live fills typical of the backtest? each live fill's entry stamps as a percentile of the backtest kept distribution
  E  path decomposition: live vs backtest — losers that never worked vs gave back
Backtest = reports/DIST_WL_2026-11-18_trades.csv.gz kept by today's RSI1h ≥ 74.07 filter (1,536 fills), net_LIVE = Exit B at today's
costs; one size 0.61 × equity for DCR / max DD / worst day.
Usage: venv/bin/python -I scripts/DIST_DEEP_DIVE_2026-11-27.py <orders.csv>
"""
import os
import sys

import numpy as np
import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
import study_frenzy_swing_short_walk as W  # noqa: E402
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-27_15-23-36.csv")
OUT = os.path.join(R, "reports", "DIST_DEEP_DIVE_2026-11-27")
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)
X = 0.61


def path(net, day):
    r = pd.Series(np.asarray(net, float) * X / 100).groupby(np.asarray(day)).sum()
    r = r.reindex(range(int(min(day)), int(max(day)) + 1), fill_value=0.0).values
    eq = np.cumprod(1 + r)
    return (eq[-1] ** (1 / len(r)) - 1) * 100, (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() * 100, r.min() * 100


def btc_dd():
    rows, s = [], int(pd.Timestamp("2023-05-01").value // 10**6)
    while True:
        js = requests.get("https://fapi.binance.com/fapi/v1/klines", params=dict(symbol="BTCUSDT", interval="1d", startTime=s, limit=1000),
                          headers={"User-Agent": "research-client/1.0"}, timeout=30).json()
        rows += js
        if len(js) < 1000:
            break
        s = int(js[-1][0]) + 86_400_000
    d = pd.DataFrame([(pd.to_datetime(int(r[0]), unit="ms"), float(r[2]), float(r[4])) for r in rows], columns=["d", "h", "c"]).set_index("d")
    return ((d.c / d.h.rolling(30).max() - 1) * 100).shift(1)


def off24lo(entry_ms):
    t, o, h, l_, c = W.load("BTCUSDT")[:5]
    v = (c / pd.Series(l_).rolling(288).min().values - 1) * 100
    k = np.searchsorted(t, np.asarray(entry_ms) // 300_000 * 300_000) - 1
    out = v[np.clip(k, 0, len(v) - 1)].astype(float)
    out[(np.asarray(entry_ms) - t[np.clip(k, 0, len(t) - 1)]) > 3_600_000] = np.nan
    return out


def main():
    os.makedirs(OUT + "_data", exist_ok=True)
    t = pd.read_csv(os.path.join(R, "reports", "DIST_WL_2026-11-18_trades.csv.gz"), low_memory=False)
    K = t[t.rsi1h >= 74.07].sort_values("entry_ms").reset_index(drop=True)
    K["net"] = K.net_LIVE; K["win"] = K.net > 0
    K["blk"] = np.where(K.entry_ms >= HOL, "HOLDOUT", "DESIGN")
    out = []
    P = lambda s="": (print(s), out.append(s))  # noqa: E731
    w, l = K.net[K.win], K.net[~K.win]
    be = -l.mean() / (w.mean() - l.mean()) * 100
    dcr, dd, wd = path(K.net, K.day)
    P(f"# DIST deep dive — backtest kept {len(K)} fills · WR {K.win.mean()*100:.1f} % · avg {K.net.mean():+.2f} %/fill · win {w.mean():+.2f} / "
      f"loss {l.mean():+.2f} · breakeven WR {be:.1f} % · DCR {dcr:+.2f} %/day · max DD {dd:.1f} % · worst day {wd:.1f} %")
    # A — how unusual
    r10w, r10m = K.win.rolling(10).sum(), K.net.rolling(10).mean()
    P(f"\n## A · 10-fill windows (in time order, {r10w.notna().sum()}): ≤ 3 wins {(r10w <= 3).mean()*100:.1f} % · mean ≤ −1.01 "
      f"{(r10m <= -1.01).mean()*100:.1f} % · both {((r10w <= 3) & (r10m <= -1.01)).mean()*100:.1f} %")
    nxt = [K.net.iloc[i + 1:i + 31].mean() for i in np.flatnonzero(((r10w <= 3) & (r10m <= -1.01)).values) if i + 31 <= len(K)]
    P(f"   after such a window, the NEXT 30 fills averaged {np.mean(nxt):+.2f} %/fill (median {np.median(nxt):+.2f}; {len(nxt)} windows, "
      f"overlapping) vs the sleeve's {K.net.mean():+.2f} — does a bad stretch predict more bad?")
    # B — stability
    K["half"] = pd.to_datetime(K.entry_ms, unit="ms").dt.year.astype(str) + "-H" + np.where(pd.to_datetime(K.entry_ms, unit="ms").dt.month <= 6, "1", "2")
    K["month"] = pd.to_datetime(K.entry_ms, unit="ms").dt.to_period("M").astype(str)
    hb = K.groupby("half").agg(N=("net", "size"), WR=("win", "mean"), avg=("net", "mean"))
    hb["WR"] *= 100
    hb["DCR"] = [path(g.net, g.day)[0] for _, g in K.groupby("half")]
    P("\n## B · per half-year\n" + hb.round(2).to_string())
    mb = K.groupby("month").agg(N=("net", "size"), avg=("net", "mean"))
    P(f"   months: {len(mb)} · negative {int((mb.avg < 0).sum())} · worst {mb.avg.min():+.2f} ({mb.avg.idxmin()}) · "
      f"negative months followed by a negative month {int(((mb.avg < 0) & (mb.avg.shift(-1) < 0)).sum())} of {int((mb.avg < 0).sum())}")
    # C — market variables (day units)
    dd30 = btc_dd()
    K["btc_dd30"] = pd.to_datetime(K.entry_ms, unit="ms").dt.floor("D").map(dd30)
    K["btc_off24lo"] = off24lo(K.entry_ms.values)
    rows = []
    def split(name, m_good, label_good, label_bad):
        m_good = pd.Series(m_good, index=K.index)
        ok = m_good.notna()
        g, b = K[ok & (m_good == True)], K[ok & (m_good == False)]   # noqa: E712
        rows.append(dict(var=name, side_a=label_good, N_a=len(g), days_a=g.day.nunique(), avg_a=g.net.mean(), WR_a=g.win.mean() * 100,
                         side_b=label_bad, N_b=len(b), days_b=b.day.nunique(), avg_b=b.net.mean(), WR_b=b.win.mean() * 100,
                         DES_a=g.net[g.blk == "DESIGN"].mean(), DES_b=b.net[b.blk == "DESIGN"].mean(),
                         HOL_a=g.net[g.blk == "HOLDOUT"].mean(), HOL_b=b.net[b.blk == "HOLDOUT"].mean(),
                         halves_b_neg=f"{int((b.groupby('half').net.mean() < 0).sum())}/{b.half.nunique()}"))
    split("BTC daily regime (EMA50 > 200)", K.btc_d1_regime.map({1.0: True, -1.0: False}), "bull", "bear")
    split("BTC 1 d return", np.where(K.btc_ret1d.isna(), np.nan, K.btc_ret1d > 0), "up", "down")
    split("BTC 1 h return", np.where(K.btc_ret1h.isna(), np.nan, K.btc_ret1h > 0), "up", "down")
    split("breadth (share of coins up)", np.where(K.breadth.isna(), np.nan, K.breadth > K.breadth.median()), "above median", "below")
    split("BTC below its 30 d high", np.where(K.btc_dd30.isna(), np.nan, K.btc_dd30 > -20), "> −20 %", "≤ −20 % (TODAY: −23…−30)")
    split("BTC above its 24 h low", np.where(np.isnan(K.btc_off24lo), np.nan, K.btc_off24lo < 2), "< 2 %", "≥ 2 %")
    C = pd.DataFrame(rows)
    P("\n## C · market variables, sign granularity (a = first side · b = second side; DES / HOL = block means)\n" + C.round(2).to_string(index=False))
    K["ddb"] = pd.cut(K.btc_dd30, [-100, -20, -15, -10, -5, 0.01])
    P("\n   BTC 30 d drawdown buckets\n" + K.groupby("ddb", observed=True).agg(N=("net", "size"), days=("day", "nunique"), WR=("win", "mean"),
                                                                              avg=("net", "mean")).round(2).to_string())
    # D — live fills vs the backtest profile
    o = pd.read_csv(EXPORT, low_memory=False)
    o = o[(o.entry_strategy == "DISTRIBUTION_SHORT") & (o.status == "CLOSED")]
    m = dict(entry_dist_rsi1h="rsi1h", entry_dist_di_diff1h="di_diff1h", entry_dist_gap15="gap15", entry_dist_bar_ret_pct="bar_ret_pct",
             entry_dist_vsw_mult="vmult15", entry_frenzy_hours="hrs_since_spike", entry_pair_volume_24h_usd="vol24", entry_dist_atr_pct="atr15_pct")
    rows = []
    for lc, bc in m.items():
        if bc not in K or lc not in o:
            continue
        v = pd.to_numeric(o[lc], errors="coerce")
        pct = [(K[bc] <= x).mean() * 100 if x == x else np.nan for x in v]
        rows.append(dict(stamp=lc, backtest_median=K[bc].median(), live_median=v.median(), live_pctile_median=np.nanmedian(pct),
                         live_outside_p5_p95=int(sum((p < 5) or (p > 95) for p in pct if p == p))))
    D = pd.DataFrame(rows)
    P("\n## D · live fills on the backtest's scale (percentile of each live value inside the 1,536 kept fills)\n" + D.round(2).to_string(index=False))
    # E — path decomposition
    o["peak"] = pd.to_numeric(o.peak_pnl, errors="coerce")
    lz = o[o.pnl_percentage <= 0]
    P(f"\n## E · live losers {len(lz)}: never worked (peak < +1 %) {int((lz.peak < 1).sum())} · gave back from +1…+5 % {int(lz.peak.between(1, 5).sum())} · "
      f"from > +5 % {int((lz.peak > 5).sum())}; live winners {int((o.pnl_percentage > 0).sum())} (all TP)")
    if "reach" in K:
        lb = K[~K.win]
        P(f"   backtest losers {len(lb)}: reach < 1 % {(lb.reach < 1).mean()*100:.0f} % · 1–5 % {lb.reach.between(1, 5).mean()*100:.0f} % · "
          f"> 5 % {(lb.reach > 5).mean()*100:.0f} %")
    C.to_csv(OUT + "_data/market_splits.csv", index=False); D.to_csv(OUT + "_data/live_vs_bt.csv", index=False); hb.to_csv(OUT + "_data/halves.csv")
    open(OUT + "_data/console.txt", "w").write("\n".join(out))


if __name__ == "__main__" and os.environ.get("EP") not in ("1", "2"):
    main()


def episodes():
    """F — the BTC ≤ −20 % below-30 d-high zone in WINDOW units (CLAUDE.md window-units rule): contiguous zone-day runs (a gap ≤ 3 days
    joins them) → per episode N · avg for DIST (kept) and EXH (TP 7 kept), and how many episodes are negative."""
    import importlib.util
    dd = btc_dd()
    z = dd[dd <= -20].index
    eps, cur = [], []
    for d in z:
        if cur and (d - cur[-1]).days > 3:
            eps.append((cur[0], cur[-1])); cur = []
        cur.append(d)
    if cur:
        eps.append((cur[0], cur[-1]))
    t = pd.read_csv(os.path.join(R, "reports", "DIST_WL_2026-11-18_trades.csv.gz"), low_memory=False)
    K = t[t.rsi1h >= 74.07].assign(net=lambda d: d.net_LIVE)
    sp = importlib.util.spec_from_file_location("sa", os.path.join(HERE, "EXH_SPIKE_AGE_2026-11-26.py"))
    SA = importlib.util.module_from_spec(sp); sp.loader.exec_module(SA)
    E = SA.kept(SA.load_bt()[0])
    rows = []
    for a, b in eps:
        lo, hi = int(a.value // 10**6), int((b + pd.Timedelta(days=1)).value // 10**6)
        kd, ke = K[(K.entry_ms >= lo) & (K.entry_ms < hi)], E[(E.entry_ms >= lo) & (E.entry_ms < hi)]
        rows.append(dict(episode=f"{a:%Y-%m-%d} → {b:%Y-%m-%d}", days=(b - a).days + 1, BTC_min_dd=round(float(dd[a:b].min()), 1),
                         DIST_N=len(kd), DIST_avg=round(kd.net.mean(), 2) if len(kd) else np.nan,
                         EXH_N=len(ke), EXH_avg=round(ke.net.mean(), 2) if len(ke) else np.nan))
    T = pd.DataFrame(rows)
    T.to_csv(OUT + "_data/btc_dd20_episodes.csv", index=False)
    print("\n## F · BTC ≤ −20 % below its 30 d high — episodes (window units)\n" + T.to_string(index=False))
    for s in ("DIST", "EXH"):
        v = T[T[f"{s}_N"] > 0]
        print(f"   {s}: {len(v)} episodes with fills · negative {int((v[f'{s}_avg'] < 0).sum())} · mean of episode means {v[f'{s}_avg'].mean():+.2f} · "
              f"biggest episode share of fills {v[f'{s}_N'].max() / v[f'{s}_N'].sum() * 100:.0f} %")


if __name__ == "__main__" and os.environ.get("EP") == "1":
    episodes()


def zone_bar():
    """G — DIST in the BTC ≤ −20 % zone on the locked Pattern-C expectancy bar (window units) + the sleeve before → after (one size)."""
    dd = btc_dd()
    t = pd.read_csv(os.path.join(R, "reports", "DIST_WL_2026-11-18_trades.csv.gz"), low_memory=False)
    K = t[t.rsi1h >= 74.07].copy(); K["net"] = K.net_LIVE
    K["dd"] = pd.to_datetime(K.entry_ms, unit="ms").dt.floor("D").map(dd)
    K["half"] = pd.to_datetime(K.entry_ms, unit="ms").dt.year.astype(str) + "-H" + np.where(pd.to_datetime(K.entry_ms, unit="ms").dt.month <= 6, "1", "2")
    z = (K.dd <= -20).values
    Z = K[z]
    # episodes = runs of zone days (gap ≤ 3 d)
    zd = sorted(set(pd.to_datetime(Z.entry_ms, unit="ms").dt.floor("D")))
    ep, k = {}, 0
    for i, d in enumerate(zd):
        if i and (d - zd[i - 1]).days > 3:
            k += 1
        ep[d] = k
    Z = Z.assign(ep=pd.to_datetime(Z.entry_ms, unit="ms").dt.floor("D").map(ep))
    rng = np.random.default_rng(650); eps = Z.ep.unique()
    g = {e: Z.net[Z.ep == e].values for e in eps}
    bs = [np.concatenate([g[e] for e in rng.choice(eps, len(eps))]).mean() for _ in range(5000)]
    w, l = K.net[K.net > 0], K.net[K.net <= 0]
    be = -l.mean() / (w.mean() - l.mean()) * 100
    loss = Z.groupby("ep").net.sum()
    print(f"\n## G · Pattern-C bar on the zone (window = drawdown episode): N {len(Z)} · WR {(Z.net > 0).mean()*100:.1f} % vs breakeven "
          f"{be:.1f} % · avg {Z.net.mean():+.2f} · P(mean < 0) by episode bootstrap {np.mean(np.array(bs) < 0):.2f} · episodes {len(eps)} · "
          f"biggest episode share of the zone loss {(-loss[loss < 0]).max() / -loss[loss < 0].sum() * 100:.0f} %")
    rows = []
    for sc, m in [("ALL", np.ones(len(K), bool))] + [(h, (K.half == h).values) for h in sorted(K.half.unique())]:
        a = K[m]; b = K[m & ~z]
        if not len(a):
            continue
        p0, p1 = path(a.net, a.day), path(b.net, b.day if len(b) else a.day)
        rows.append(dict(scope=sc, N_before=len(a), avg_before=a.net.mean(), DCR_before=p0[0], DD_before=p0[1], worst_before=p0[2],
                         N_after=len(b), avg_after=b.net.mean(), DCR_after=p1[0], DD_after=p1[1], worst_after=p1[2], blocked=int(m.sum() - len(b))))
    T = pd.DataFrame(rows)
    T.to_csv(OUT + "_data/zone_before_after.csv", index=False)
    print("   DIST before → after blocking the zone (backtest, one size 0.61)\n" + T.round(2).to_string(index=False))


if __name__ == "__main__" and os.environ.get("EP") == "2":
    zone_bar()
