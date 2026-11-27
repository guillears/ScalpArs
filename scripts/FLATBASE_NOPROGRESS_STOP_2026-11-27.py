#!/usr/bin/env python3
"""FLATBASE_NOPROGRESS_STOP_2026-11-27 — "72 h of holding is luck" (operator 11-27, the open MOVE / LAYER shorts never went our way).
Research only, no bot / config change.

PRIOR (not re-run): DL 622 plain time caps (FB 24 h −0.166 · 48 h −0.059 · 60 h +0.006 %/fill → keep 72 h) · STATE_EXIT (DL 608–610: FB_4H /
FB_VWAP / FB_BASE / FB_VOL state exits, null). NEW here: a CONDITIONAL "never went our way" stop — at checkpoint T h after the entry, if the
short's best favourable move so far (MFE, from 5m lows, net of nothing) is below X %, close at market (bar close at T, taker 0.05 % + stop
slip ≥ 0.10 %); else today's exit runs unchanged (TP −5 maker / stop +10 / 72 h). Funding to the new exit (rebuild table).
PRE-REGISTERED GRID (frozen before any result was read): T ∈ {2, 4, 8, 12, 24, 36, 48} h × X ∈ {1, 2, 3} % = 21 cells + the descriptive
"what happened next" table (the fills a cell would close: their result under today's exit). Every cell counts in the pick-the-best null.
Ship bar (DL 601 / 603 style): Δ %/fill > 0 in ALL and HOLDOUT, one-sided 95 % day-bootstrap P(Δ > 0), beats the pick-the-best null, master
real fills not worse, DCR one size up, max DD not worse.
DATA: the FLATBASE_ENTRY_TIMING_2026-11-26 sim (545 backtest episodes on today's stack + 17 master kept + batch kept) — LIVE variant only,
same 5m arrays and entry price; + the batch fills after it (SLX / BIGTIME closed, MOVE / LAYER open, from the orders export / dashboard).
Usage: venv/bin/python -I scripts/FLATBASE_NOPROGRESS_STOP_2026-11-27.py <orders.csv>
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-27_14-36-23.csv")
sys.argv = [sys.argv[0], EXPORT]
sp = importlib.util.spec_from_file_location("fbet", os.path.join(HERE, "FLATBASE_ENTRY_TIMING_2026-11-26.py"))
ET = importlib.util.module_from_spec(sp); sp.loader.exec_module(ET)
C, W = ET.C, ET.W
BAR, HR = ET.BAR, ET.HR
OUT = os.path.join(os.path.dirname(HERE), "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27")
TS, XS = [2, 4, 8, 12, 24, 36, 48], [1.0, 2.0, 3.0]
CELLS = [(T, X) for T in TS for X in XS]
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)


def walk(t, o, h, l_, c, j0, E, sx, T=None, X=None):
    """SHORT at E from bar j0 (market entry). today's exit, plus the conditional close at bar j0 + 12T − 1 when MFE < X. → (gross, fee_x, why, jx)."""
    n = len(t)
    S, P = E * (1 + ET.SL / 100), E * (1 - ET.TP / 100)
    sxs = max(sx, ET.STOP_SLIP_MIN)
    last = min(n - 1, j0 + ET.HOLD_B - 1)
    jc = j0 + int(T * 12) - 1 if T else 10**12
    for j in range(j0, last + 1):
        if h[j] >= S:
            X_ = max(S, o[j]) * (1 + sxs / 100)
            return (1 - X_ / E) * 100, ET.TAKER, "STOP", j
        if l_[j] <= P:
            X_ = min(P, o[j])
            return (1 - X_ / E) * 100, ET.MAKER, "TP", j
        if j == jc:
            mfe = (1 - l_[j0:j + 1].min() / E) * 100
            if mfe < X:
                X_ = c[j] * (1 + sxs / 100)
                return (1 - X_ / E) * 100, ET.TAKER, "NOPROG", j
    X_ = c[last] * (1 + sx / 100)
    return (1 - X_ / E) * 100, ET.TAKER, ("TIME" if last == j0 + ET.HOLD_B - 1 else "OPEN"), last


def episodes():
    S = pd.read_pickle(os.path.join(os.path.dirname(HERE), "reports", "FLATBASE_ENTRY_TIMING_2026-11-26_data", "sim.pkl"))
    L = S[(S["var"] == "LIVE") & S.traded].copy()
    wl = pd.read_csv(os.path.join(os.path.dirname(HERE), "reports", "OOS_REAL_FLATBASE_data", "wl_features.csv"), low_memory=False)
    sx_map = dict(zip(wl.key, wl.slip_x))
    out = []
    for r in L.itertuples():
        if r.src == "BT":
            arr = W.load(r.pair)
            t, o, h, l_, c = arr[:5]
            sx = float(sx_map.get(r.ep, 0.10))
        else:
            t, o, h, l_, c, _ = ET.real_arrays(r.pair, r.ep_t - 210 * HR, r.ep_t + 125 * HR)
            sx = 0.10
        j0 = int(np.searchsorted(t, r.te))
        out.append(dict(src=r.src, pair=r.pair, ep=r.ep, te=int(r.te), E=float(r.E), sx=sx, arrs=(t, o, h, l_, c), j0=j0, booked=r.booked))
    # batch fills after the sim (today's stack, closed) — entered at the confirm close (real fill price)
    x = pd.read_csv(EXPORT, low_memory=False)
    x = x[(x.entry_strategy == "FLATBASE_SHORT") & (x.status == "CLOSED") & (pd.to_datetime(x.opened_at) >= "2026-11-21")]
    for r in x.itertuples():
        te = int(pd.Timestamp(r.opened_at).value // 10**6) // BAR * BAR
        t, o, h, l_, c, _ = ET.real_arrays(r.pair, te - 210 * HR, te + 125 * HR)
        out.append(dict(src="BATCH", pair=r.pair, ep=f"{r.pair}|{r.opened_at[:19]}", te=te, E=float(r.entry_price), sx=0.10,
                        arrs=(t, o, h, l_, c), j0=int(np.searchsorted(t, te)), booked=float(r.pnl_percentage)))
    return out


def price(eps, F, T=None, X=None):
    rows = []
    for e in eps:
        t, o, h, l_, c = e["arrs"]
        g, fx, why, jx = walk(t, o, h, l_, c, e["j0"], e["E"], e["sx"], T, X)
        tx = int(t[jx]) + BAR
        fu, miss = C.fund(F, e["pair"], e["te"] + 10_000, tx)
        rows.append(dict(src=e["src"], ep=e["ep"], pair=e["pair"], te=e["te"], tx=tx, why=why, net=g - ET.TAKER - fx + fu, hold_h=(tx - e["te"]) / HR))
    return pd.DataFrame(rows)


def path_stats(D, x=ET.X_SIZE):
    d = pd.to_datetime(D.tx, unit="ms").dt.normalize()
    r = (D.net * x / 100).groupby(d.values).sum()
    r = r.reindex(pd.date_range(d.min(), d.max()), fill_value=0.0).values
    eq = np.cumprod(1 + r)
    return (eq[-1] ** (1 / len(r)) - 1) * 100, (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() * 100, r.min() * 100


def day_boot_p(delta, day, B=4000, seed=27):
    rng = np.random.default_rng(seed)
    g = pd.Series(delta).groupby(np.asarray(day)).sum()
    n = pd.Series(delta).groupby(np.asarray(day)).size()
    k = len(g)
    idx = rng.integers(0, k, (B, k))
    m = g.values[idx].sum(1) / n.values[idx].sum(1)
    return float((m > 0).mean())


def main():
    os.makedirs(OUT + "_data", exist_ok=True)
    F = C.funding_table()
    eps = episodes()
    base = price(eps, F)
    base.to_csv(OUT + "_data/base.csv", index=False)
    print("base parity (BT net vs entry-timing sim LIVE):", base.groupby("src").net.agg(["size", "mean"]).round(3).to_dict())
    # descriptive: MFE at each checkpoint and the final result under today's exit
    desc = []
    for (T, X) in CELLS:
        D = price(eps, F, T, X)
        D["day"] = D.te // 86_400_000
        hit = (D.why == "NOPROG").values
        b = base.net.values
        for scope, m in (("BT ALL", D.src.eq("BT")), ("BT DESIGN", D.src.eq("BT") & (D.te < HOL)), ("BT HOLDOUT", D.src.eq("BT") & (D.te >= HOL)),
                         ("MASTER", D.src.eq("MASTER")), ("BATCH", D.src.eq("BATCH"))):
            m = m.values
            if not m.any():
                continue
            dl = D.net.values[m] - b[m]
            dcr0, dd0, wd0 = path_stats(base[m]) if scope.startswith("BT") else (np.nan,) * 3
            dcr1, dd1, wd1 = path_stats(D[m]) if scope.startswith("BT") else (np.nan,) * 3
            desc.append(dict(T=T, X=X, scope=scope, N=int(m.sum()), cut=int(hit[m].sum()),
                             cut_today_avg=float(b[m][hit[m]].mean()) if hit[m].any() else np.nan,
                             cut_today_wr=float((b[m][hit[m]] > 0).mean() * 100) if hit[m].any() else np.nan,
                             cut_new_avg=float(D.net.values[m][hit[m]].mean()) if hit[m].any() else np.nan,
                             avg_before=float(b[m].mean()), avg_after=float(D.net.values[m].mean()), delta=float(dl.mean()),
                             P_gt0=day_boot_p(dl, D.day.values[m]) if scope.startswith("BT") else np.nan,
                             DCR_b=dcr0, DCR_a=dcr1, DD_b=dd0, DD_a=dd1, WD_b=wd0, WD_a=wd1))
        print(f"  T {T} X {X} done", flush=True)
    R = pd.DataFrame(desc)
    R.to_csv(OUT + "_cells.csv", index=False)
    # pick-the-best null on BT ALL: resample days, best-cell mean Δ distribution vs observed best
    print(R[R.scope.isin(["BT ALL", "BT HOLDOUT", "MASTER", "BATCH"])].round(2).to_string(index=False))


if __name__ == "__main__" and os.environ.get("CHART") != "1":
    main()


def chart():
    """Event-study: backtest FB fills grouped by their best move in the first 12 h (MFE, from 5m lows) → mean P&L path under TODAY's exit
    (hourly, the realised result once closed) — does a short that 'never went our way' early end up a loser?"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = pd.read_pickle(os.path.join(os.path.dirname(HERE), "reports", "FLATBASE_ENTRY_TIMING_2026-11-26_data", "sim.pkl"))
    L = S[(S["var"] == "LIVE") & S.traded & (S.src == "BT")].copy()
    eps = {e["ep"]: e for e in episodes() if e["src"] == "BT"}
    g = []
    for r in L.itertuples():
        e = eps[r.ep]; t, o, h, l_, c = e["arrs"]; j0 = e["j0"]
        jx = int(np.searchsorted(t, r.tx - BAR))
        j12 = min(j0 + 12 * 12 - 1, jx)
        mfe = (1 - l_[j0:j12 + 1].min() / e["E"]) * 100
        g.append("closed < 12 h" if r.tx - r.te <= 12 * HR else ("MFE < 1 %" if mfe < 1 else ("MFE 1–3 %" if mfe < 3 else "MFE ≥ 3 %")))
    L["grp"] = g
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col in (("closed < 12 h", "#999999"), ("MFE ≥ 3 %", "#2a9d8f"), ("MFE 1–3 %", "#e9c46a"), ("MFE < 1 %", "#e76f51")):
        P = np.vstack(L[L.grp == k].path_exit.values)
        ax.plot(range(73), np.nanmean(P, 0), color=col, lw=2, label=f"{k}: N {len(P)} · final {np.nanmean(P[:, -1]):+.2f} % · WR {(L[L.grp == k].net > 0).mean() * 100:.0f} %")
    ax.axvline(12, color="k", ls=":", lw=1); ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("hours after entry"); ax.set_ylabel("mean P&L % (today's exit; realised once closed)")
    ax.set_title("FLATBASE backtest (545, today's stack): fills still open at 12 h, by their best move so far")
    ax.legend(fontsize=8, loc="lower right"); fig.tight_layout(); fig.savefig(OUT + "_chart.png", dpi=110)
    print(L.groupby("grp").net.agg(["size", "mean", lambda x: (x > 0).mean()]).round(2))


if __name__ == "__main__" and os.environ.get("CHART") == "1":
    chart()
