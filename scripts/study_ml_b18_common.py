#!/usr/bin/env python3
"""Shared loaders for the B18 momentum-long losers study (2026-10-08). Read-only.
master: MASTER_POOL_stacked kept, non-probe, CLOSED, MOMENTUM LONG; pct = stack_pct (today's rules), usd = stack_pnl (today's sizing).
yr5   : engine replay ML fills trimmed to each chunk window (scripts/yr5_fills_trimmed.py); usd = fixed $3k book at live sizing."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)
import yr5_fills_trimmed as YT

WASH = ("2026-06-18", "2026-07-03")   # Jun-18 → Jul-2 inclusive
B18_KEYS = {("ARBUSDT", "2026-10-07T16:31:56"), ("UNIUSDT", "2026-10-07T16:36:04"), ("LITUSDT", "2026-10-07T17:13:13")}


def _btc_daily():
    b = pd.read_csv("reports/backtest_cache/k5m_full/BTCUSDT.csv", usecols=["open_time", "c"]).drop_duplicates("open_time")
    b["t"] = pd.to_datetime(b.open_time, unit="ms")
    g = b.set_index("t").c.resample("1D")
    dc = g.last().where(g.count() >= 288)          # complete UTC days only (the cache's last day is partial)
    return (dc / dc.shift(1) - 1) * 100, b.t.max()


def _btc_off30():
    """BTC % below its 30-day high (5m highs, 8640 bars) at each 5m close — rebuild for fills without the stamp."""
    b = pd.read_csv("reports/backtest_cache/k5m_full/BTCUSDT.csv", usecols=["open_time", "h", "c"]).drop_duplicates("open_time")
    b["t"] = pd.to_datetime(b.open_time, unit="ms") + pd.Timedelta(minutes=5)   # value known at bar close
    b = b.set_index("t").sort_index()
    return (b.c / b.h.rolling(8640, min_periods=2000).max() - 1) * 100


def _is_ml(d):
    return (d.direction == "LONG") & (d.entry_strategy.fillna("MOMENTUM").astype(str).isin(["MOMENTUM", "", "nan"]))


def history_master():
    """every non-probe as-traded MOMENTUM LONG fill (kept or not) — for stop / cluster sequencing."""
    P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    h = P[_is_ml(P) & (P.status == "CLOSED") & ~P.is_probe.astype(str).str.lower().isin(["true", "1"])].copy()
    h["o"] = pd.to_datetime(h.opened_at.astype(str).str[:19]); h["c"] = pd.to_datetime(h.closed_at.astype(str).str[:19], errors="coerce")
    return h


def seq_flags(fills, hist, group_col=None):
    """cooldown (opened < 30 min after an ML STOP_LOSS* close; sequential: a cohort fill never starts a cooldown) and
    cluster2_120 (>= 2 accepted ML fills opened in the prior 120 min). hist = all accepted fills (same group)."""
    cd = pd.Series(False, index=fills.index); cl = pd.Series(False, index=fills.index)
    groups = [(None, fills, hist)] if group_col is None else [(g, fills[fills[group_col] == g], hist[hist[group_col] == g]) for g in fills[group_col].unique()]
    for _, f, h in groups:
        h = h.sort_values("o")
        # sequential cooldown over the history order
        in_cd = {}
        last_stop_close = None
        for i, r in h.iterrows():
            flag = last_stop_close is not None and (r.o - last_stop_close).total_seconds() < 1800 and r.o >= last_stop_close
            in_cd[i] = flag
            if (not flag) and str(r.close_reason).startswith("STOP_LOSS") and pd.notna(r.c):
                last_stop_close = r.c if last_stop_close is None else max(last_stop_close, r.c)
        os_ = h.o.values
        for i, r in f.iterrows():
            if i in in_cd:
                cd[i] = in_cd[i]
            n = ((os_ < np.datetime64(r.o)) & (os_ >= np.datetime64(r.o - pd.Timedelta(minutes=120)))).sum()
            cl[i] = n >= 2
    return cd, cl


def master(include_b1=False):
    P = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    m = P[_is_ml(P) & (P.status == "CLOSED") & P.stack_keep.astype(str).str.lower().isin(["true", "1"])
          & ~P.is_probe.astype(str).str.lower().isin(["true", "1"])].copy()
    if not include_b1:
        m = m[m.era != "B1"]
    m["o"] = pd.to_datetime(m.opened_at.astype(str).str[:19])
    m["c"] = pd.to_datetime(m.closed_at.astype(str).str[:19], errors="coerce")
    m["pct"] = pd.to_numeric(m.stack_pct, errors="coerce").fillna(pd.to_numeric(m.pnl_percentage, errors="coerce"))
    m["usd"] = pd.to_numeric(m.stack_pnl, errors="coerce")
    m["day"] = m.o.dt.strftime("%Y-%m-%d")
    m["wash"] = (m.o >= WASH[0]) & (m.o < WASH[1])
    m["b18"] = [(p, str(o)[:19]) in B18_KEYS for p, o in zip(m.pair, m.opened_at)]
    dret, kmax = _btc_daily()
    reb = (m.o.dt.floor("D") - pd.Timedelta(days=1)).map(dret)
    st = pd.to_numeric(m.get("entry_btc_1d_ret_pct"), errors="coerce")
    m["btc1d_src"] = np.where(st.notna(), "stamp", np.where(reb.notna(), "rebuild", "none"))
    m["btc1d"] = st.fillna(reb)
    m["btc1d_rebuild"] = reb
    o30 = _btc_off30()
    r30 = pd.Series(o30.reindex(m.o.dt.floor("5min"), method="ffill").values, index=m.index)
    st30 = pd.to_numeric(m.get("entry_btc_off30d_high_pct"), errors="coerce")
    m["off30"] = st30.fillna(r30); m["off30_rebuild"] = r30
    m["washed30"] = m.off30 <= -15
    h = history_master()
    cd, cl = seq_flags(m.assign(), h.assign(), group_col=None)
    # history is per era (a reset separates batches; fills of other eras never overlap in time anyway)
    m["cooldown"] = cd.reindex(m.index).fillna(False); m["cluster2"] = cl.reindex(m.index).fillna(False)
    m["src"] = "master"; m["unit"] = m.day
    return m.reset_index(drop=True)


def yr5():
    F = YT.load(sleeves=["MOM-long"])
    F["o"] = F.t; F["c"] = pd.to_datetime(F.closed_at.astype(str).str[:23].str.replace("T", " "), format="mixed", errors="coerce")
    F["usd"] = YT.fixed_book_usd(F)
    F["day"] = F.o.dt.strftime("%Y-%m-%d")
    F["wash"] = (F.o >= WASH[0]) & (F.o < WASH[1])
    F["btc1d"] = pd.to_numeric(F.entry_btc_1d_ret_pct, errors="coerce"); F["btc1d_src"] = "stamp"
    F["off30"] = pd.to_numeric(F.entry_btc_off30d_high_pct, errors="coerce"); F["washed30"] = F.off30 <= -15
    cd, cl = seq_flags(F, F, group_col="seed")
    F["cooldown"] = cd; F["cluster2"] = cl
    F["b18"] = False; F["src"] = "yr5"; F["unit"] = F.day
    F["era"] = F.o.dt.strftime("%Y-%m")
    return F.reset_index(drop=True)


def boot_day(d, n=4000, seed=7):
    """day-clustered bootstrap of the per-trade mean: returns (lo, hi, P(mean<0))."""
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    g = d.groupby("unit").pct.agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    k = len(s)
    idx = rng.integers(0, k, size=(n, k))
    means = s[idx].sum(1) / c[idx].sum(1)
    return (np.percentile(means, 2.5), np.percentile(means, 97.5), (means < 0).mean())


def stats(d, nseeds=1):
    if len(d) == 0:
        return dict(N=0)
    lo, hi, p = boot_day(d)
    w = d[d.pct > 0].pct.mean(); l = d[d.pct <= 0].pct.mean()
    loss = d[d.pct < 0]
    tl = -loss.pct.sum()
    mx_day = (-loss.groupby("unit").pct.sum()).max() / tl if tl > 0 else 0
    mx_pair = (-loss.groupby("pair").pct.sum()).max() / tl if tl > 0 else 0
    return dict(N=len(d) / nseeds, WR=(d.pct > 0).mean() * 100, avg=d.pct.mean(), usd=d.usd.sum() / nseeds,
                days=d.unit.nunique(), lo=lo, hi=hi, p_neg=p, maxday=mx_day * 100, maxpair=mx_pair * 100,
                avgwin=w, avgloss=l)


def be_wr(d):
    w = d[d.pct > 0].pct.mean(); l = -d[d.pct <= 0].pct.mean()
    return l / (w + l) * 100
