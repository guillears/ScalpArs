#!/usr/bin/env python3
"""EXH_SPIKE_AGE (2026-11-26, research only) — do EXHAUSTION_SHORT entries < 1 h after the Frenzy spike do worse at TODAY's exit?
(operator 2026-11-26: INITUSDT 10:00 entered 0.25 h after the spike and stopped −5.38 % in 10 min; other recent fills entered 1.3–23 h after.)

INPUTS: the TP_GRID / ENTRY_WL data folders are gitignored (local research data) — the script is not reproducible from the public repo alone.
REAL FILLS: read from the raw export (argv[1], default the 11-26 export); checked 11-26: 17 CLOSED EXH rows, 0 duplicate
(opened_at, pair, direction), 16 FRENZY-universe + 1 unstamped — the scout's XS.load_fills view counts the same fills.
PRIOR: reports/ENTRY_WL_2026-11-18.md — hrs_since_spike / hrs_since_peak on EXH at the OLD exit (TP 10), runner-share label: nothing.
NEW: booked NET % at today's exit (TP min(20×ATR, 7 %) maker / SL min(1.5×ATR, 6 %) / 24 h = TP_GRID 2026-11-20 cell F7, DL 609),
the < 1 h bucket, a dose-response by hour bucket, and the real fills.

Backtest: rows_FRENZY.pkl + sim_FRENZY.npz (cell F7, net = r + funding − entry slip − 0.05 − (taker exit slip + 0.05 | maker 0.02),
slip floor 0.10 — the SHORT_AFTER_LOSS_2026-11-26 exh() recipe). Rows are already today's EXH rule set: spot market required
(has_spot = 1 on every row) and EXH_SSC_OIVOL removed upstream (0 rows match the block on s_ssc / s_oi_vol). Hours since spike from
reports/ENTRY_WL_2026-11-18_data/exh.pkl (same 3,368 rows, joined on xi + verified pair / entry_ms). One position per coin is applied
AFTER the block (a blocked early fill frees the coin for a later signal — stack reachability).
Real: current batch export (= every EXH fill since the 11-18 sleeve start; MASTER_POOL_stacked has no EXH rows), pnl_percentage
(% of notional, net of fees + funding), split by exit era at the DL 609 deploy (2026-11-21 20:59 UTC).
Out: reports/EXH_SPIKE_AGE_2026-11-26.csv (+ _daily.csv) — summaries only."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dist_exit_eval as EV  # noqa: E402

R = os.path.dirname(HERE); RP = os.path.join(R, "reports")
DAY, H1 = 86_400_000, 3_600_000
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)
TP7_DEPLOY = pd.Timestamp("2026-11-21 20:59:49")   # b3e5a19d, UTC
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-26_13-02-05.csv")
SIZE = 0.61   # one size: 0.5 × 5× ≈ 0.61 × equity notional per fill
BUCKETS = [(0, .5), (.5, 1), (1, 2), (2, 4), (4, 8), (8, 24), (24, 72.01)]
BL = lambda a, b: f"{a:g}-{b:g}h" if b < 72 else "24-72h"
RNG = np.random.default_rng(1126)
OUT = os.path.join(RP, "EXH_SPIKE_AGE_2026-11-26")


def load_bt():
    GD = os.path.join(RP, "TP_GRID_2026-11-20_EXH_data")
    X = pd.read_pickle(os.path.join(GD, "rows_FRENZY.pkl")).reset_index(drop=True)
    Z = dict(np.load(os.path.join(GD, "sim_FRENZY.npz"), allow_pickle=True)); names = [str(x) for x in Z.pop("names")]
    assert (Z["xi"] == X.xi.values).all()
    j = names.index("F7"); s = np.maximum(X.slip.values.astype(float), 0.10); tk = Z["k"][:, j] != 0
    net = Z["r"][:, j].astype(float) + Z["f"][:, j].astype(float) - s - 0.05 - np.where(tk, s + 0.05, 0.02)
    E = pd.read_pickle(os.path.join(RP, "ENTRY_WL_2026-11-18_data", "exh.pkl")).reset_index(drop=True)
    E = E.set_index("xi").reindex(X.xi.values)
    ok_join = (E.pair.values == X.pair.values) & (E.entry_ms.values == X.entry_ms.values)
    D = pd.DataFrame(dict(pair=X.pair.astype(str).values, entry_ms=X.entry_ms.values.astype(np.int64), xm=Z["xm"][:, j].astype(np.int64),
                          net=net, k=Z["k"][:, j].astype(int), half=X.half.values, hrs=np.where(ok_join, E.hrs_since_spike.values, np.nan),
                          spike_ms=np.where(ok_join, E.spike_ms.values, np.nan), exh_flag=E.EXH.values,
                          ok=Z["ok"] & (Z["k"][:, j] >= 0) & (Z["k"][:, j] != 9)))
    info = dict(rows=len(X), join_rate=float(ok_join.mean()), hrs_known=int(np.isfinite(D.hrs).sum()),
                ssc_oivol_hits=int(((E.s_ssc > -0.0366) & (E.s_oi_vol > 0.128)).sum()), no_spot=int((E.has_spot != 1).sum()))
    D = D[D.ok & np.isfinite(D.net) & np.isfinite(D.hrs)].reset_index(drop=True)
    D["day"] = D.entry_ms // DAY; D["blk"] = np.where(D.entry_ms >= HOL, "HOLDOUT", "DESIGN")
    D["bucket"] = pd.cut(D.hrs, [b[0] for b in BUCKETS] + [BUCKETS[-1][1]], right=False, labels=[BL(*b) for b in BUCKETS]).astype(str)
    D["ep"] = D.pair + ":" + D.spike_ms.astype(np.int64).astype(str)
    return D, info


def kept(D, block_lt=None):
    v = np.ones(len(D), bool) if block_lt is None else (D.hrs.values >= block_lt)
    return D[EV.one_pos(D.pair.values, D.entry_ms.values, D.xm.values, v)].reset_index(drop=True)


def day_boot(v, days, nb=4000, block=10):
    if not len(v):                       # (review) an empty zone has no bootstrap
        return float("nan"), float("nan"), float("nan")
    s_ = pd.Series(v).groupby(np.asarray(days)).agg(["sum", "count"])
    dd = np.arange(s_.index.min(), s_.index.max() + 1); s_ = s_.reindex(dd, fill_value=0)
    n = len(dd); nbk = int(np.ceil(n / block)); st = RNG.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    m = s_["sum"].values[idx].sum(1) / np.maximum(s_["count"].values[idx].sum(1), 1)
    return float(np.percentile(m, 95)), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def diff_boot(a, da, b, db, nb=4000, block=10):
    """zone (a) − rest (b) mean, 10-day moving blocks over the joint calendar span."""
    lo0 = min(da.min(), db.min()); n = max(da.max(), db.max()) - lo0 + 1
    sa = np.bincount(da - lo0, a, n); ca = np.bincount(da - lo0, None, n); sb = np.bincount(db - lo0, b, n); cb = np.bincount(db - lo0, None, n)
    nbk = int(np.ceil(n / block)); st = RNG.integers(0, max(n - block, 1), (nb, nbk))
    idx = (st[:, :, None] + np.arange(block)[None, None, :]).reshape(nb, -1)[:, :n]
    d = sa[idx].sum(1) / np.maximum(ca[idx].sum(1), 1) - sb[idx].sum(1) / np.maximum(cb[idx].sum(1), 1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def stats(g):
    v = g.net.values
    if len(v) == 0:
        return dict(N=0)
    return dict(N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum(), days=g.day.nunique(), eps=g.ep.nunique() if "ep" in g else np.nan)


def path(K, d0, d1):
    """one-size daily compound over every calendar day [d0, d1]: r_d = Σ net % × SIZE / 100 of fills CLOSED that day."""
    cd = K.xm.values // DAY
    r = pd.Series(K.net.values * SIZE / 100).groupby(cd).sum().reindex(np.arange(d0, d1 + 1), fill_value=0.0).values
    eq = np.cumprod(1 + r); dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    return dict(DCR=(eq[-1] ** (1 / len(r)) - 1) * 100, maxDD=dd * 100, worst_day=r.min() * 100, days=len(r), fills=len(K)), r


def main():
    D, info = load_bt()
    rows = []
    K0 = kept(D)                                   # today's kept fills (no spike-age block)
    w, l = K0.net[K0.net > 0], K0.net[K0.net <= 0]
    be = -l.mean() / (w.mean() - l.mean()) * 100
    # 1 · buckets by block / half
    for scope, m in [("ALL", np.ones(len(K0), bool)), ("DESIGN", K0.blk == "DESIGN"), ("HOLDOUT", K0.blk == "HOLDOUT")] + \
            [(h, K0.half == h) for h in sorted(K0.half.unique())]:
        B = K0[np.asarray(m)]
        for bk in [BL(*b) for b in BUCKETS]:
            rows.append(dict(table="bucket", scope=scope, cohort=bk, **stats(B[B.bucket == bk])))
        for nm, mm in (("<1h", B.hrs < 1), (">=1h", B.hrs >= 1), ("<0.5h (post-hoc)", B.hrs < .5), (">=0.5h", B.hrs >= .5), ("all", B.hrs >= 0)):
            r = dict(table="split", scope=scope, cohort=nm, **stats(B[mm]))
            if nm.startswith("<") and mm.sum() >= 5 and (~mm).sum() >= 5:
                r["diff_lo"], r["diff_hi"] = diff_boot(B.net[mm].values, B.day[mm].values, B.net[~mm].values, B.day[~mm].values)
            rows.append(r)
    # 1b · mechanism by bucket: stop / TP / time-out share, median minutes held, avg of stopped and of TP fills
    for bk in [BL(*b) for b in BUCKETS]:
        g = K0[K0.bucket == bk]; mins = (g.xm - g.entry_ms) / 60_000
        rows.append(dict(table="mech", scope="ALL", cohort=bk, N=len(g), stop_pct=(g.k == 1).mean() * 100, tp_pct=(g.k == 0).mean() * 100,
                         timeout_pct=(g.k == 2).mean() * 100, med_min_held=mins.median(), med_min_to_stop=mins[g.k == 1].median()))
    # 2 · Pattern-C bar on the < 1 h zone (ALL / DESIGN / HOLDOUT)
    for scope, zc, thr in [(sc, z, t) for z, t in (("<1h", 1.0), ("<0.5h (post-hoc)", 0.5)) for sc in ("ALL", "DESIGN", "HOLDOUT")]:
        B = K0 if scope == "ALL" else K0[K0.blk == scope]
        Zn = B[B.hrs < thr]; v = Zn.net.values
        ub, lo, hi = day_boot(v, Zn.day.values)
        neg = np.minimum(v, 0)
        dl = pd.Series(neg).groupby(Zn.day.values).sum(); pl = pd.Series(neg).groupby(Zn.pair.values).sum()
        el = pd.Series(neg).groupby(Zn.ep.values).sum()
        rows.append(dict(table="patternC", scope=scope, cohort=zc, **stats(Zn), BE=be, ci_lo=lo, ci_hi=hi, ub95=ub,
                         top_day_loss_share=dl.min() / neg.sum() * 100 if neg.sum() < 0 else 0,
                         top_pair_loss_share=pl.min() / neg.sum() * 100 if neg.sum() < 0 else 0,
                         top_ep_loss_share=el.min() / neg.sum() * 100 if neg.sum() < 0 else 0,
                         halves_neg=f"{int((Zn.groupby('half').net.mean() < 0).sum())}/{Zn.half.nunique()}"))
    # 3 · breakeven-bot read: one-size DCR / max DD / worst day, with vs without the < 1 h block (block applied BEFORE one_pos)
    d0, d1 = int(D.day.min()), int(max(D.xm.max() // DAY, D.day.max()))
    daily = {}
    for nm, thr in (("today", None), ("block <1h", 1.0), ("block <0.5h", 0.5)):
        K = kept(D, thr)
        for scope, a, b in (("ALL", d0, d1), ("DESIGN", d0, HOL // DAY - 1), ("HOLDOUT", HOL // DAY, d1)):
            Kc = K[(K.xm // DAY >= a) & (K.xm // DAY <= b)]
            p, r = path(Kc, a, b)
            rows.append(dict(table="path", scope=scope, cohort=nm, N=len(Kc), WR=(Kc.net > 0).mean() * 100, avg=Kc.net.mean(), sum=Kc.net.sum(), **p))
            if scope == "ALL":
                daily[nm] = r
    pd.DataFrame(daily, index=np.arange(d0, d1 + 1)).to_csv(OUT + "_daily.csv", index_label="day")
    # 4 · real fills
    Bx = pd.read_csv(EXPORT, low_memory=False)
    Rf = Bx[(Bx.entry_strategy == "EXHAUSTION_SHORT") & (Bx.status == "CLOSED")].copy()
    Rf["net"] = Rf.pnl_percentage.astype(float); Rf["hrs"] = Rf.entry_frenzy_hours.astype(float)
    Rf["day"] = pd.to_datetime(Rf.opened_at).dt.strftime("%Y-%m-%d")
    Rf["era"] = np.where(pd.to_datetime(Rf.opened_at) >= TP7_DEPLOY, "TP7", "TP10")
    for era in ("ALL", "TP7", "TP10"):
        B = Rf if era == "ALL" else Rf[Rf.era == era]
        for nm, mm in (("<1h", B.hrs < 1), ("1-2h", (B.hrs >= 1) & (B.hrs < 2)), ("2-4h", (B.hrs >= 2) & (B.hrs < 4)),
                       ("4-8h", (B.hrs >= 4) & (B.hrs < 8)), ("8-24h", (B.hrs >= 8) & (B.hrs < 24)), (">=1h", B.hrs >= 1), ("all", B.hrs >= 0)):
            rows.append(dict(table="real", scope=era, cohort=nm, **stats(B[mm])))
    T = pd.DataFrame(rows)
    T.to_csv(OUT + ".csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(info); print("kept fills", len(K0), "breakeven WR", round(be, 1), "| EXH-flag subset of <1h:", int(K0[K0.hrs < 1].exh_flag.eq(True).sum()))
    for t in ("bucket", "split", "mech", "patternC", "path", "real"):
        print("\n==", t); print(T[T.table == t].dropna(axis=1, how="all").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
