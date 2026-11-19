#!/usr/bin/env python3
"""ROUTE A FOLLOW-UP (PREREG reports/ROUTE_A_FOLLOWUP_2026-11-19_PREREG.md) — Q1(a) full population per half-year / month,
Q1(c) regime in window units, Q2 secondary (1m VWAP proxy on the full population), 1m-vs-tick cross-check.
Reads reports/ROUTE_A_FOLLOWUP_data/pop1m.pkl (route_a_followup_price1m.py). Writes
reports/ROUTE_A_FOLLOWUP_2026-11-19_{q1a_half,q1a_month,q1a_xcheck,q1c_windows,q1c_summary,q1c_monthly,q2_vwap1m}.csv."""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import route_a_followup_common as C   # noqa: E402
D = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_data"); SRC = os.path.join(R, "reports", "ROUTE_A_TICKS_data")
OUT = os.path.join(R, "reports", "ROUTE_A_FOLLOWUP_2026-11-19")
DAY = 86_400_000
EXN = ("Q3", "TR", "T1")
NB = 1000


def netc(X, ex, var, slip):
    g = X[f"{ex}_{var}_g"].values / 100; k = X[f"{ex}_{var}_k"].values; f = X[f"{ex}_{var}_f"].values; s = slip / 100
    out_px = np.where(k == 2, 1 + g, (1 + g) * (1 - s)); fee = np.where(k == 2, 0.02, 0.05)
    n = ((out_px / (1 + s)) - 1) * 100 - 0.05 - fee - f
    return np.where(X[f"{ex}_{var}_complete"].fillna(False).astype(bool).values, n, np.nan)


def one_pos(X, ex, var):
    """keep a trigger only if the same coin has no open trade of this exit (entry ≥ previous kept exit time)"""
    keep = np.zeros(len(X), bool)
    xcol = X[f"{ex}_{var}_x"].values; ok = np.isfinite(X[f"{ex}_{var}_g"].values)
    for p, idx in X.groupby("pair").indices.items():
        idx = idx[np.argsort(X.sig_ms.values[idx])]
        busy = -1
        for i in idx:
            if not ok[i]:
                continue
            if X.sig_ms.values[i] >= busy:
                keep[i] = True; busy = xcol[i]
    return keep


def stats(x, w, day, span):
    ok = np.isfinite(x)
    if ok.sum() < 3:
        return dict(N=int(ok.sum()))
    lo, hi = C.day_boot(x, w, day, NB)
    return dict(N=int(ok.sum()), days=len(np.unique(day[ok])), trades_day=ok.sum() / span, WR=C.wmean(x > 0, w) * 100,
                avg=C.wmean(x, w), med=float(np.nanmedian(x)), ci_lo=lo, ci_hi=hi)


def windows(sign, min_len=7):
    """maximal same-sign runs; a run shorter than min_len days takes the sign of the preceding window (the first run: of the
    next); adjacent same-sign runs are then merged. Returns (window id per day, window sign per day)."""
    v = sign.values.astype(int).copy()
    for _ in range(5000):
        st = np.r_[0, np.where(v[1:] != v[:-1])[0] + 1]; ln = np.diff(np.r_[st, len(v)])
        short = np.where(ln < min_len)[0]
        if not len(short):
            break
        i = short[0]
        if i > 0:
            v[st[i]:st[i] + ln[i]] = v[st[i] - 1]
        elif len(st) > 1:
            v[:ln[0]] = v[st[1]]
        else:
            break
    wid = np.r_[0, np.cumsum(v[1:] != v[:-1])]
    return pd.Series(wid, index=sign.index), pd.Series(v, index=sign.index)


def perm_p(a, b, n=5000, rng=C.RNG):
    """one-sided permutation p that mean(a) > mean(b)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    obs = a.mean() - b.mean(); z = np.r_[a, b]; c = 0
    for _ in range(n):
        p = rng.permutation(z); c += (p[:len(a)].mean() - p[len(a):].mean()) >= obs
    return c / n


def binom_p(k, n):
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n if n else np.nan


def main():
    P = pickle.load(open(os.path.join(D, "pop1m.pkl"), "rb"))
    X = P["X"]; print("status", X.status.value_counts().to_dict(), "drift", P["d"])
    X = X[X.status == "ok"].reset_index(drop=True)
    X["day"] = X.sig_ms // DAY; X["half"] = C.half_of(X.sig_ms.values); X["w"] = 1.0
    span_all = (X.sig_ms.max() - X.sig_ms.min()) / DAY
    hspan = X.groupby("half").sig_ms.agg(lambda v: max(1.0, (v.max() - v.min()) / DAY))
    mspan = X.groupby("month").sig_ms.agg(lambda v: max(1.0, (v.max() - v.min()) / DAY))
    for ex in EXN:
        for var in ("p", "x"):
            X[f"keep_{ex}_{var}"] = one_pos(X, ex, var)
            for sl in (0.10, 0.30):
                X[f"n_{ex}_{var}_{int(sl*100)}"] = netc(X, ex, var, sl)
    # ── Q1(a) per half / month
    H, M = [], []
    for pop, pm in (("all", np.ones(len(X), bool)), ("not_flagged", X.flagged.values == 0)):
        for ex in EXN:
            for var in ("p", "x"):
                k = pm & X[f"keep_{ex}_{var}"].values
                for slip in (10, 30):
                    x = X[f"n_{ex}_{var}_{slip}"].values
                    for hf, idx in X[k].groupby("half").groups.items():
                        H.append(dict(pop=pop, exit=ex, entry=var, slip=slip / 100, half=hf,
                                      **stats(x[idx], X.w.values[idx], X.day.values[idx], hspan[hf])))
                    if var == "p":
                        for mo, idx in X[k].groupby("month").groups.items():
                            M.append(dict(pop=pop, exit=ex, slip=slip / 100, month=mo,
                                          **stats(x[idx], X.w.values[idx], X.day.values[idx], mspan[mo])))
                    st = stats(x[k], X.w.values[k], X.day.values[k], span_all)
                    H.append(dict(pop=pop, exit=ex, entry=var, slip=slip / 100, half="ALL", **st))
    Hd = pd.DataFrame(H); Md = pd.DataFrame(M)
    Hd.to_csv(OUT + "_q1a_half.csv", index=False); Md.to_csv(OUT + "_q1a_month.csv", index=False)
    # ── cross-check vs the tick-priced C0 (sampled events, Route A features, all incl. flagged)
    F = pickle.load(open(os.path.join(SRC, "features.pkl"), "rb"))
    F0 = F[(F.kind == "event") & (F.chk == 0)][["pair", "sig_ms", "TR_g", "Q3_g", "T1_g", "entry_px"]]
    J = F0.merge(X, on=["pair", "sig_ms"])
    xc = []
    for ex in EXN:
        for var in ("p", "x"):
            a = J[f"{ex}_g"].values; b = J[f"{ex}_{var}_g"].values; ok = np.isfinite(a) & np.isfinite(b)
            xc.append(dict(exit=ex, entry=var, N=int(ok.sum()), corr=np.corrcoef(a[ok], b[ok])[0, 1], tick_mean=a[ok].mean(),
                           k1m_mean=b[ok].mean(), bias=(b[ok] - a[ok]).mean()))
    XC = pd.DataFrame(xc); XC.to_csv(OUT + "_q1a_xcheck.csv", index=False)
    # ── Q1(c) regime windows
    T = pd.read_csv(os.path.join(R, "reports", "FRENZY_LONG_SEARCH_data", "tape_daily.csv"), index_col=0)
    T = T.loc[(T.index >= X.day.min()) & (T.index <= X.day.max())]
    T["B30"] = (T.btc_r30_prev > 0).astype(int)
    regs = {"R1": T.R1.fillna(0).astype(int), "R5": T.R5.fillna(0).astype(int), "RF": T.RF.fillna(0).astype(int), "B30": T.B30}
    h2 = T.index >= (pd.Timestamp("2026-07-01").value // 10**6 // DAY)
    Wrows, Srows = [], []
    for ex in EXN:
        k = X[f"keep_{ex}_p"].values
        x = X[f"n_{ex}_p_10"].values
        for nm, sg in regs.items():
            wid, ws = windows(sg)
            X["_w"] = X.day.map(wid); X["_s"] = X.day.map(ws)
            h2sign = int(sg[h2].mean() >= 0.5)
            g = pd.DataFrame(dict(w=X["_w"].values[k], s=X["_s"].values[k], x=x[k], day=X.day.values[k])).dropna()
            agg = g.groupby("w").agg(sign=("s", "first"), N=("x", "size"), avg=("x", "mean"), d0=("day", "min"), d1=("day", "max"))
            agg["start"] = pd.to_datetime(agg.d0 * DAY, unit="ms").dt.strftime("%Y-%m-%d")
            agg["end"] = pd.to_datetime(agg.d1 * DAY, unit="ms").dt.strftime("%Y-%m-%d")
            agg["h2"] = agg.d1 >= (pd.Timestamp("2026-07-01").value // 10**6 // DAY)
            for wi, r in agg.iterrows():
                Wrows.append(dict(exit=ex, var=nm, window=int(wi), **r.to_dict()))
            e = agg[(~agg.h2) & (agg.N >= 10)]
            same, oth = e[e.sign == h2sign], e[e.sign != h2sign]
            h2v = agg[agg.h2]
            h2avg = np.nanmean(x[k & (X.day.values >= (pd.Timestamp("2026-07-01").value // 10**6 // DAY))])
            npos = int((same.avg > 0).sum())
            Srows.append(dict(exit=ex, var=nm, h2_sign=h2sign, h2_days_share=float(sg[h2].mean()), h2_windows=len(h2v), h2_avg=h2avg,
                              earlier_same_sign_windows=len(same), same_sign_positive=npos, same_sign_mean=same.avg.mean(),
                              other_sign_windows=len(oth), other_sign_positive=int((oth.avg > 0).sum()), other_sign_mean=oth.avg.mean(),
                              binom_p=binom_p(npos, len(same)) if len(same) else np.nan,
                              perm_p_same_gt_other=perm_p(same.avg.values, oth.avg.values)))
    Wd = pd.DataFrame(Wrows); Sd = pd.DataFrame(Srows)
    Wd.to_csv(OUT + "_q1c_windows.csv", index=False); Sd.to_csv(OUT + "_q1c_summary.csv", index=False)
    # monthly regime shares vs monthly all-trigger net (descriptive)
    T["month"] = pd.to_datetime(T.index * DAY, unit="ms").strftime("%Y-%m")
    mreg = T.groupby("month").agg(R1=("R1", "mean"), R5=("R5", "mean"), RF=("RF", "mean"), B30=("B30", "mean"),
                                  breadth=("breadth_prev", "mean"), spikes7=("spikes7_prev", "mean"), btc_r30=("btc_r30_prev", "mean"))
    mm = Md[(Md["pop"] == "all") & (Md.slip == 0.10)].pivot_table(index="month", columns="exit", values="avg")
    mreg = mreg.join(mm, how="inner")
    mreg.to_csv(OUT + "_q1c_monthly.csv")
    rk = mreg.rank()
    print("monthly Spearman vs TR:", {c: round(rk[c].corr(rk["TR"]), 2) for c in ["R1", "R5", "RF", "B30", "breadth", "spikes7", "btc_r30"]})
    # ── Q2 secondary: 1m VWAP proxy top quintile (DESIGN cut) on the full population, C0
    des = X.block == "DESIGN"
    thr = np.nanquantile(X.loc[des, "vwap1m"], 0.80)
    V = []
    for ex in EXN:
        k = X[f"keep_{ex}_p"].values
        for slip in (10, 30):
            x = X[f"n_{ex}_p_{slip}"].values
            for sel, m in (("top20_vwap1m", (X.vwap1m.values >= thr)), ("rest", X.vwap1m.values < thr), ("all", np.ones(len(X), bool))):
                for hf in list(hspan.index) + ["DESIGN", "HOLDOUT"]:
                    mm_ = k & m & ((X.half.values == hf) if hf.endswith(("H1", "H2")) else (X.block.values == hf))
                    sp = hspan[hf] if hf in hspan.index else (span_all * (30 / 35) if hf == "DESIGN" else span_all * (5 / 35))
                    V.append(dict(exit=ex, slip=slip / 100, select=sel, period=hf, **stats(x[mm_], X.w.values[mm_], X.day.values[mm_], sp)))
    Vd = pd.DataFrame(V); Vd["thr_vwap1m"] = thr; Vd.to_csv(OUT + "_q2_vwap1m.csv", index=False)
    # dose by vwap1m quintile (DESIGN cuts)
    cuts = np.nanquantile(X.loc[des, "vwap1m"], [0.2, 0.4, 0.6, 0.8]); X["vq"] = np.searchsorted(cuts, X.vwap1m.values, side="right") + 1
    Dq = []
    for blk in ("DESIGN", "HOLDOUT"):
        for q_ in range(1, 6):
            o = dict(block=blk, quintile=q_)
            for ex in EXN:
                m = X[f"keep_{ex}_p"].values & (X.vq.values == q_) & (X.block.values == blk)
                o[ex] = np.nanmean(X[f"n_{ex}_p_10"].values[m]); o["N"] = int(m.sum())
            o["runner_pct"] = X.runner.values[(X.vq.values == q_) & (X.block.values == blk)].mean() * 100
            Dq.append(o)
    pd.DataFrame(Dq).to_csv(OUT + "_q2_vwap1m_dose.csv", index=False)
    pickle.dump(dict(X=X, H=Hd, M=Md, XC=XC, W=Wd, S=Sd, mreg=mreg, V=Vd, Dq=pd.DataFrame(Dq)), open(os.path.join(D, "q1.pkl"), "wb"))
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(XC.round(3).to_string())
    print(Hd[(Hd.slip == 0.10) & (Hd.entry == "p")].pivot_table(index="half", columns=["pop", "exit"], values="avg").round(2).to_string())
    print(Sd.round(3).to_string())


if __name__ == "__main__":
    main()
