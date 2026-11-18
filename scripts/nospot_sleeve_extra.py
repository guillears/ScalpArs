#!/usr/bin/env python3
"""NOSPOT_SLEEVE — report tables: path decomposition (XE + each cohort's best exit), BTC regime (R1 / R4 sign) per half-year, one-size
before → after and the joint add to EXHAUSTION + DISTRIBUTION, the trade-level CSV, and the event chart (anchored paths: candidate
winners vs losers, NS0, operator coins). Writes reports/NOSPOT_SLEEVE_2026-11-18_{paths,regime,joint}.csv, _trades.csv(.gz), _events.png"""
import os, pickle, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import nospot_sleeve_lib as L   # noqa: E402
import nospot_sleeve_build as NB   # noqa: E402
OUT = os.path.join(L.R, "reports", "NOSPOT_SLEEVE_2026-11-18")
CAND = ("R1_OI_UNWIND", "F1012", "Q6")          # OBSERVE candidate (kept = mcap ≥ $50 M or unscored)


def cand_idx(X, sel):
    idx = sel[(CAND[0], CAND[1])]
    small = np.nan_to_num(X.mcap.values[idx], nan=np.inf) < 50e6
    return idx[~small], idx


def daily(X, idx, e, col=None):
    v = X[f"{e}_net"].values[idx] if col is None else col
    return pd.Series(v * L.SIZE / 100).groupby(X.day.values[idx] if col is None else idx).sum()


def onesize(r, d0, d1):
    r = r.reindex(np.arange(d0, d1), fill_value=0.0)
    eq = np.cumprod(1 + r.values); D = d1 - d0
    return dict(dcr=(eq[-1] ** (1 / D) - 1) * 100 if eq[-1] > 0 else np.nan,
                maxdd=(1 - eq / np.maximum.accumulate(np.r_[1.0, eq])[1:]).max() * 100, worst_day=r.min() * 100)


def main():
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    S = pickle.load(open(os.path.join(L.ND, "sel.pkl"), "rb")); sel = S["sel"]
    cohorts = pickle.load(open(os.path.join(L.ND, "wl_cohorts.pkl"), "rb"))
    # 1. path decomposition
    rows = []
    for c in ["B0", "NS0"] + [k for k in L.rule_masks(X.iloc[:5])]:
        for e in sorted({"XE", *[v[1] for k, v in cohorts.items() if v[0] == c]}):
            idx = sel[(c, e)]; vv = X[f"{e}_net"].values[idx]
            g = L.path_group(X, idx, e)
            tot_loss = vv[vv <= 0].sum()
            for grp in ["NEVER_WORKED", "STOPPED_RECOVERED", "GAVE_BACK", "DRIFT_LOSS", "LEFT_ON_TABLE", "CAPTURED"]:
                m = g == grp
                rows.append(dict(cohort=c, exit=e, group=grp, N=int(m.sum()), share=m.mean() * 100, sum_pct=vv[m].sum(),
                                 loss_share=vv[m].sum() / tot_loss * 100 if grp in ("NEVER_WORKED", "STOPPED_RECOVERED", "GAVE_BACK", "DRIFT_LOSS") else np.nan,
                                 avg=vv[m].mean() if m.any() else np.nan))
    P = pd.DataFrame(rows); P.to_csv(OUT + "_paths.csv", index=False)
    # 2. regime per half-year (R1 daily EMA50 vs 200 sign; R4 5m label sign)
    rows = []
    kept, allr1 = cand_idx(X, sel)
    REG = {"NS0|XA": sel[("NS0", "XA")], "NS0|XE": sel[("NS0", "XE")], "R1|F1012": allr1, "R1|F1012|Q6-kept": kept,
           "R4|F1510": sel[("R4_TOP_DISTRIB", "F1510")]}
    exits = {"NS0|XA": "XA", "NS0|XE": "XE", "R1|F1012": "F1012", "R1|F1012|Q6-kept": "F1012", "R4|F1510": "F1510"}
    for nm, idx in REG.items():
        e = exits[nm]; vv = X[f"{e}_net"].values[idx]
        for rg in ("R1", "R4"):
            for s in (-1, 0, 1):
                if rg == "R1" and s == 0:
                    continue
                m = X[rg].values[idx] == s
                rec = dict(cohort=nm, regime=rg, sign=s, N=int(m.sum()), avg=vv[m].mean() if m.any() else np.nan)
                for h, _, _ in L.HALVES:
                    mh = m & (X.half.values[idx] == h)
                    rec[h] = f"{vv[mh].mean():+.2f} ({mh.sum()})" if mh.sum() else "–"
                rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT + "_regime.csv", index=False)
    # 3. one size before → after, joint with EXHAUSTION + DISTRIBUTION (ALL span 2023-07 → 2026-11-18 and HOLDOUT)
    E = pd.read_csv(os.path.join(L.R, "reports", "BASELINE_TRADES_EXHAUSTION_2026-11-18.csv"))
    tk = E.exit_kind.astype(str).str.upper().str.startswith("TP")
    E["net"] = E.r - 0.10 - np.where(tk, 0, 0.10) - 0.05 - np.where(tk, 0.02, 0.05) + E.fund
    E["day"] = E.entry_ms // L.DAY
    Dd = pd.read_csv(os.path.join(L.R, "reports", "DIST_WL_2026-11-18_trades.csv.gz"), usecols=["pair", "entry_ms", "rsi1h", "in_LIVE", "net_LIVE", "LIVE_x"])
    Dd = Dd[(Dd.in_LIVE == True) & (Dd.rsi1h >= 74.07)].copy(); Dd["day"] = Dd.entry_ms // L.DAY  # noqa: E712
    rows = []
    for blk in ("ALL", "HOLDOUT"):
        D, d0, d1 = L.cal_days(blk)
        t0 = d0 * L.DAY
        e_ = E[E.entry_ms >= t0]; dd_ = Dd[Dd.entry_ms >= t0]
        series = {"EXHAUSTION": pd.Series(e_.net.values * L.SIZE / 100).groupby(e_.day.values).sum(),
                  "DISTRIBUTION": pd.Series(dd_.net_LIVE.values * L.SIZE / 100).groupby(dd_.day.values).sum()}
        for nm, (idx, e) in {"NS0|XA": (sel[("NS0", "XA")], "XA"), "R1|F1012": (allr1, "F1012"), "R1|F1012|Q6-kept": (kept, "F1012"),
                             "R4|F1510": (sel[("R4_TOP_DISTRIB", "F1510")], "F1510")}.items():
            ii = idx[X.entry_ms.values[idx] >= t0]
            series[nm] = daily(X, ii, e)
        ns = {"EXHAUSTION": len(e_), "DISTRIBUTION": len(dd_)}
        for k_, s in series.items():
            vv = (E.net.values if k_ == "EXHAUSTION" else None)
            r = onesize(s, d0, d1)
            rows.append(dict(blk=blk, book=k_, **r))
        base = series["EXHAUSTION"].add(series["DISTRIBUTION"], fill_value=0)
        rows.append(dict(blk=blk, book="EXH + DIST", **onesize(base, d0, d1)))
        for k_ in ("NS0|XA", "R1|F1012", "R1|F1012|Q6-kept", "R4|F1510"):
            rows.append(dict(blk=blk, book=f"EXH + DIST + {k_}", **onesize(base.add(series[k_], fill_value=0), d0, d1)))
        # overlap with DISTRIBUTION (same pair, entries within 24 h)
        for k_, (idx, e) in {"R1|F1012|Q6-kept": (kept, "F1012")}.items():
            ii = idx[X.entry_ms.values[idx] >= t0]
            dk = set(zip(dd_.pair, dd_.entry_ms // (24 * 3_600_000)))
            ov = sum((p, t // (24 * 3_600_000)) in dk for p, t in zip(X.pair.values[ii], X.entry_ms.values[ii]))
            rows.append(dict(blk=blk, book=f"overlap {k_} vs DISTRIBUTION (same pair, same UTC day)", dcr=ov, maxdd=len(ii), worst_day=np.nan))
    J = pd.DataFrame(rows); J.to_csv(OUT + "_joint.csv", index=False)
    print(J.round(3).to_string())
    # 4. trades CSV: every primary cell's selected trades (rule × exit) + NS0 + B0 is huge → NS0, rules (all exits), with features
    keep_cols = ["pair", "entry_ms", "on_ms", "spike_ms", "half", "blk", "px", "atrp", "vol24", "q5", "vd", "dd_hi", "peak_age_h", "vwap_gap",
                 "lh2", "ret15m", "ret1h", "ret4h", "ret24h", "rsi15", "rsi1h", "di15", "hrs_since_spike", "maxup1h_24h", "volmult", "mcap",
                 "turnover", "oi_usd", "oi_chg15m", "oi_chg1h", "oi_chg4h", "oi_vs_max", "oi_vol", "topP", "topP_chg1h", "topP_chg4h", "topA",
                 "topA_chg4h", "glA", "glA_chg1h", "glA_chg4h", "taker1h", "taker_prev1h", "prem1h", "prem_pk", "f_last", "f_prev", "f_int_h",
                 "f_last8", "btc1h", "btc4h", "btc24h", "R1", "R4", "slip"]
    T = []
    for (c, e), idx in sel.items():
        if c == "B0":
            continue
        d = X.loc[idx, keep_cols].copy()
        d.insert(0, "exit", e); d.insert(0, "cohort", c)
        for k in ("r", "k", "f", "x", "net", "net20", "worked", "reach", "rec", "best", "adv", "sld", "tpd"):
            d[k] = X[f"{e}_{k}"].values[idx]
        d["path_group"] = L.path_group(X, idx, e)
        T.append(d)
    T = pd.concat(T, ignore_index=True)
    T["entry_utc"] = pd.to_datetime(T.entry_ms, unit="ms").dt.strftime("%Y-%m-%d %H:%M")
    T.to_csv(OUT + "_trades.csv.gz", index=False, compression="gzip")
    print("trades rows", len(T))
    # 5. event chart
    chart(X, sel, kept)


def path_of(p, t_ms, hours=24, cache={}):
    if p not in cache:
        cache[p] = NB.TB.load(p)
    x = cache[p]
    if x is None:
        return None
    t, o, h, l, c = x[0], x[1], x[2], x[3], x[4]
    j = np.searchsorted(t, t_ms)
    if j >= len(t) or t[j] != t_ms:
        return None
    n = hours * 12 + 1
    seg = c[j - 1:j - 1 + n] if j >= 1 else None
    if seg is None or len(seg) < n:
        return None
    return -(seg / o[j] - 1) * 100       # gross short P&L vs the entry open, at entry−5m … +24 h


def chart(X, sel, kept):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    hrs = np.arange(0, 24 * 12 + 1) / 12
    e = "F1012"
    vv = X[f"{e}_net"].values[kept]
    for lab, m, col in (("winners", vv > 0, "#2a7a3b"), ("losers", vv <= 0, "#b5452f")):
        ii = kept[m]; ii = rng.choice(ii, min(600, len(ii)), replace=False)
        P = [path_of(X.pair.values[i], int(X.entry_ms.values[i])) for i in ii]
        P = np.array([p for p in P if p is not None])
        ax[0].plot(hrs, np.nanmean(P, 0), color=col, lw=2, label=f"{lab} mean (n={len(P)})")
        ax[0].fill_between(hrs, np.nanpercentile(P, 25, 0), np.nanpercentile(P, 75, 0), color=col, alpha=0.12)
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_title("R1 OI_UNWIND · F1012 · mcap ≥ $50M (OBSERVE)\nshort P&L from entry (gross %)")
    ax[0].set_xlabel("hours after entry"); ax[0].legend(fontsize=8)
    for c, e2, col in (("NS0", "XA", "#555"), ("R1_OI_UNWIND", "F1012", "#1f5fa8"), ("R4_TOP_DISTRIB", "F1510", "#8a4fb0")):
        ii = sel[(c, e2)]; ii = rng.choice(ii, min(800, len(ii)), replace=False)
        P = np.array([p for p in (path_of(X.pair.values[i], int(X.entry_ms.values[i])) for i in ii) if p is not None])
        ax[1].plot(hrs, np.nanmean(P, 0), color=col, lw=2, label=f"{c} (n={len(P)})")
        ax[1].plot(hrs, np.nanmedian(P, 0), color=col, lw=1, ls="--")
    rnd = rng.choice(len(X), 1500, replace=False)
    P = np.array([p for p in (path_of(X.pair.values[i], int(X.entry_ms.values[i])) for i in rnd) if p is not None])
    ax[1].plot(hrs, np.nanmean(P, 0), color="#c9a227", lw=2, label=f"B0 random bar (n={len(P)})")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_title("tradable entries: mean (solid) / median (dashed) short path")
    ax[1].set_xlabel("hours after entry"); ax[1].legend(fontsize=8)
    # operator coins: price vs the Frenzy-window peak (hindsight anchor) — live data
    LD = os.path.join(L.R, "reports", "NOSPOT_SLEEVE_data", "live", "k5m")
    for p, col in (("PIPPINUSDT", "#1f5fa8"), ("BLESSUSDT", "#b5452f"), ("JELLYJELLYUSDT", "#2a7a3b"), ("PHAROSUSDT", "#8a4fb0")):
        d = pd.read_csv(os.path.join(LD, p + ".csv"))
        d = d[d.open_time >= pd.Timestamp("2026-11-12").value // 10**6]
        k = int(np.argmax(d.h.values)); t0 = d.open_time.values[k]
        w = d[(d.open_time >= t0 - 24 * 3_600_000) & (d.open_time <= t0 + 36 * 3_600_000)]
        ax[2].plot((w.open_time.values - t0) / 3_600_000, (w.c.values / d.h.values[k] - 1) * 100, color=col, lw=1.3,
                   label=f"{p[:-4]} peak {pd.Timestamp(int(t0), unit='ms'):%m-%d %H:%M}")
    ax[2].axvline(0, color="k", lw=0.6); ax[2].set_title("operator coins: close vs the peak (hindsight anchor, %)")
    ax[2].set_xlabel("hours from the peak"); ax[2].legend(fontsize=8)
    fig.suptitle("NOSPOT_SLEEVE 2026-11-18 — BACKTEST, gross paths (costs not included)", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT + "_events.png", dpi=110)


if __name__ == "__main__":
    main()
