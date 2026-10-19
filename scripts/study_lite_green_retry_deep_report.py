#!/usr/bin/env python3
"""LITE GREEN-RETRY deep dive 2026-10-19 — tables for blocks ① real era · ② per batch · ③ two-year backtest + mechanism.
Reads the outputs of study_lite_green_retry_deep.py (era) and study_lite_green_retry_deep_bt.py (bt). Tabulation only (variants frozen there).
Writes reports/LITE_GREEN_RETRY_DEEP_2026-10-19_tables.md + _era_fills.csv / _bt_fills.csv / _variants.csv.
Usage: venv/bin/python scripts/study_lite_green_retry_deep_report.py"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_lite_green_retry_deep as GR  # noqa: E402
OUTD = GR.OUTD; OUT = GR.OUT
MS5, MIN = 300_000, 60_000
rng = np.random.default_rng(11)
NB = 4000
VN = [v["name"] for v in GR.VARIANTS]


def dayof(ms):
    return pd.to_datetime(ms, unit="ms").strftime("%Y-%m-%d")


def dboot_sum(d):
    d = np.asarray(d, float)
    if not len(d):
        return 0.0, np.nan, np.nan, np.nan
    bs = d[rng.integers(0, len(d), (NB, len(d)))].sum(1)
    return float(d.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float((bs > 0).mean())


def dboot_mean(F, col="pct"):
    if not len(F):
        return np.nan, np.nan, np.nan, np.nan
    g = F.groupby(F.te.map(dayof))[col].agg(["sum", "size"])
    s, c = g["sum"].values, g["size"].values
    idx = rng.integers(0, len(s), (NB, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())


def be_wr(x):
    w, l = x[x > 0], x[x <= 0]
    if not len(w) or not len(l):
        return np.nan
    return abs(l.mean()) / (w.mean() + abs(l.mean()))


def lite(X):
    return X[X.sleeve == "FRENZY_LITE"] if "sleeve" in X.columns else X


def key(F):
    return set(zip(F.pair, F.t_close))


def compare(A, V0, scope_days=None, with_long=False):
    """A, V0 = fills of one variant / today (LITE only unless with_long). → dict of the variant row."""
    la, l0 = lite(A), lite(V0)
    ka, k0 = key(la), key(l0)
    added = la[[k not in k0 for k in zip(la.pair, la.t_close)]]
    lost = l0[[k not in ka for k in zip(l0.pair, l0.t_close)]]
    tot_a, tot_0 = (A, V0) if with_long else (la, l0)
    da = tot_a.groupby(tot_a.te.map(dayof)).usd.sum(); d0 = tot_0.groupby(tot_0.te.map(dayof)).usd.sum()
    days = sorted(set(da.index) | set(d0.index) | set(scope_days or []))
    dd = da.reindex(days, fill_value=0) - d0.reindex(days, fill_value=0)
    s, lo, hi, p = dboot_sum(dd.values)
    m, mlo, mhi, mp = dboot_mean(added)
    o = dict(N_lite=len(la), WR_lite=(la.pct > 0).mean() if len(la) else np.nan, avg_lite=la.pct.mean() if len(la) else np.nan,
             usd=tot_a.usd.sum(), d_usd=s, d_lo=lo, d_hi=hi, P_gain=p,
             added=len(added), added_days=added.te.map(dayof).nunique() if len(added) else 0, added_WR=(added.pct > 0).mean() if len(added) else np.nan,
             added_avg=m, added_lo=mlo, added_hi=mhi, added_P=mp, added_be_wr=be_wr(added.pct.values) if len(added) else np.nan,
             added_usd=added.usd.sum(), lost=len(lost), lost_avg=lost.pct.mean() if len(lost) else np.nan, lost_usd=lost.usd.sum(),
             worst_day=float(da.min()) if len(da) else np.nan)
    if with_long:
        L_a, L_0 = A[A.sleeve == "FRENZY_LONG"], V0[V0.sleeve == "FRENZY_LONG"]
        kl = key(L_a); ll = L_0[[k not in kl for k in zip(L_0.pair, L_0.t_close)]]
        o.update(long_lost=len(ll), long_lost_usd=ll.usd.sum(), long_added=len(L_a) - (len(L_0) - len(ll)))
    return o, added, lost, dd


def concentration(added):
    if not len(added):
        return {}
    s = added.usd.sum()
    bp = added.groupby("pair").usd.sum(); bd = added.groupby(added.te.map(dayof)).usd.sum()
    top5 = added.sort_values("pct", ascending=False).iloc[5:]
    return dict(top_pair=bp.abs().idxmax(), top_pair_share=float(bp.abs().max() / max(1e-9, bp.abs().sum())),
                top_day=bd.abs().idxmax(), top_day_share=float(bd.abs().max() / max(1e-9, bd.abs().sum())),
                pair_n=bp.size, top5_removed_avg=float(top5.pct.mean()) if len(top5) else np.nan,
                top5_removed_usd=float(top5.usd.sum()))


def fmt(x, d=2, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"


def table(rows, cols, head):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def vrow(name, o, c=None):
    r = dict(v=name, N=o["N_lite"], WR=f"{o['WR_lite'] * 100:.0f} %" if np.isfinite(o["WR_lite"]) else "–", avg=fmt(o["avg_lite"], 3),
             usd=fmt(o["usd"], 0), d=f"{fmt(o['d_usd'], 0)} [{fmt(o['d_lo'], 0)}, {fmt(o['d_hi'], 0)}]", P=f"{o['P_gain']:.2f}",
             add=f"{o['added']} / {o['added_days']} d", addwr=f"{o['added_WR'] * 100:.0f} % (be {o['added_be_wr'] * 100:.0f} %)" if o["added"] else "–",
             addavg=f"{fmt(o['added_avg'], 3)} [{fmt(o['added_lo'], 2)}, {fmt(o['added_hi'], 2)}] P {o['added_P']:.2f}" if o["added"] else "–",
             lost=f"{o['lost']} ({fmt(o['lost_usd'], 0)})", wd=fmt(o["worst_day"], 0))
    if "long_lost" in o:
        r["ll"] = f"{o['long_lost']} ({fmt(o['long_lost_usd'], 1)})"
    if c:
        r.update(tp=f"{c['top_pair']} {c['top_pair_share'] * 100:.0f} %", td=f"{c['top_day']} {c['top_day_share'] * 100:.0f} %",
                 t5=fmt(c["top5_removed_avg"], 3))
    return r


def mech(F0, added, label):
    """retry fills vs today's normal LITE fills."""
    def d(F, nm):
        if not len(F):
            return dict(cohort=nm, N=0)
        return dict(cohort=nm, N=len(F), WR=f"{(F.pct > 0).mean() * 100:.0f} %", avg=fmt(F.pct.mean(), 3),
                    hours=f"{F.hours.median():.1f}", vol=f"{F.vol_mult.median():.0f}×", vwap=fmt(F.vs_vwap.median(), 2),
                    tp3=f"{(F.tp == 3).mean() * 100:.0f} %", slope=f"{(F.btc_h1slope > 0).mean() * 100:.0f} %",
                    mfe=fmt(F.mfe.median(), 2), stop=f"{(F.why == 'STOP').mean() * 100:.0f} %",
                    wait=(f"{F.wait_min.median():.0f}" if "wait_min" in F and F.wait_min.notna().any() else "–"),
                    move=(fmt(F.move_from_green.median(), 2) if "move_from_green" in F and F.move_from_green.notna().any() else "–"),
                    gret=(fmt(F.green_ret.median(), 2) if "green_ret" in F and F.green_ret.notna().any() else "–"),
                    bret=fmt(F.bar_ret.median(), 2))
    l0 = lite(F0)
    return [d(l0, f"{label}: today's LITE fills"), d(added, f"{label}: added retry fills")]


MCOLS = ["cohort", "N", "WR", "avg", "hours", "vol", "vwap", "tp3", "slope", "mfe", "stop", "wait", "move", "gret", "bret"]
MHEAD = ["cohort", "N", "WR", "avg %", "age h (med)", "vol mult (med)", "vs VWAP % (med)", "TP +3 share", "BTC 1h slope > 0", "MFE % (med)",
         "stopped", "wait min (med)", "move green→entry % (med)", "green bar ret % (med)", "entry bar ret % (med)"]
VCOLS = ["v", "N", "WR", "avg", "usd", "d", "P", "add", "addwr", "addavg", "lost", "wd"]
VHEAD = ["variant", "LITE N", "WR", "avg %", "Σ $", "Δ $ vs V0 [day CI]", "P(Δ>0)", "added / days", "added WR (breakeven)", "added avg % [day CI] P(>0)",
         "lost today fills ($)", "worst day $"]


def era_block(lines):
    F = pd.read_pickle(f"{OUTD}/era_fills.pkl")
    V0 = F[F.variant == "V0"]
    allday = sorted(set(F.te.map(dayof)))
    rows, extra, dd_all = [], [], {}
    for v in VN[1:]:
        A = F[F.variant == v]
        o, added, lost, dd = compare(A, V0, allday, with_long=True)
        c = concentration(added)
        r = vrow(v, o, c); rows.append(r); dd_all[v] = dd
        o["variant"] = v; o.update(c); extra.append(o)
    r0 = V0[V0.sleeve == "FRENZY_LITE"]
    rows.insert(0, dict(v="V0 (today)", N=len(r0), WR=f"{(r0.pct > 0).mean() * 100:.0f} %", avg=fmt(r0.pct.mean(), 3), usd=fmt(V0.usd.sum(), 0),
                        d="–", P="–", add="–", addwr=f"be {be_wr(r0.pct.values) * 100:.0f} %", addavg="–", lost="–", ll="–",
                        wd=fmt(V0.groupby(V0.te.map(dayof)).usd.sum().min(), 0)))
    lines.append("## ① Real era (replica 2026-10-02 → 10-19 14:54, live + master period, T100) — every variant\n")
    lines.append("Σ $ = LITE + FRENZY_LONG (LONG at $100 blocker notional, so LONG cannibalisation is visible); Δ $ paired day-block bootstrap "
                 f"over {len(allday)} days.\n")
    lines.append(table(rows, VCOLS + ["ll", "tp", "td", "t5"], VHEAD + ["LONG fills lost ($)", "top pair share of added", "top day share",
                                                                      "added avg, top-5 removed"]))
    lines.append("")
    # LODO choice of variant (days)
    daily = pd.DataFrame({v: F[F.variant == v].groupby(F[F.variant == v].te.map(dayof)).usd.sum() for v in VN[:-1]}).reindex(allday).fillna(0)
    oos = []; picks = []
    for d in daily.index:
        j = daily.drop(index=d).sum().idxmax(); picks.append(j); oos.append(daily.loc[d, j])
    o = pd.Series(oos, index=daily.index)
    s, lo, hi, p = dboot_sum((o - daily["V0"]).values)
    lines.append(f"**LODO choice of variant (day units, 11 cells excl. RETRY_ANY):** held-out Σ {fmt(o.sum(), 0)} vs V0 {fmt(daily['V0'].sum(), 0)} → "
                 f"Δ {fmt(s, 0)} [{fmt(lo, 0)}, {fmt(hi, 0)}] P {p:.2f}; picks {pd.Series(picks).value_counts().to_dict()}\n")
    # per batch
    brow = []
    for v in ["SXr", "S15r", "S30r", "S60r", "SXn", "W60", "RETRY_ANY"]:
        A = F[F.variant == v]
        r = dict(v=v)
        for e in [x[0] for x in GR.EN.ERAS]:
            a, b = A[A.era == e].usd.sum(), V0[V0.era == e].usd.sum()
            r[e] = fmt(a - b, 0)
        brow.append(r)
    eras = [x[0] for x in GR.EN.ERAS]
    lines.append("## ② Per batch — Δ $ vs V0 (LITE + LONG)\n")
    lines.append(table(brow, ["v"] + eras, ["variant"] + eras))
    nb = {e: (V0[(V0.era == e) & (V0.sleeve == "FRENZY_LITE")].shape[0]) for e in eras}
    ad = {e: int(((F.variant == "SXr") & (F.how == "RETRY") & (F.era == e)).sum()) for e in eras}
    lines.append("\nV0 LITE fills per batch: " + " · ".join(f"{e} {nb[e]}" for e in eras) + "  \nSXr retry fills per batch: " +
                 " · ".join(f"{e} {ad[e]}" for e in eras) + "\n")
    # per-fill list (primary) + AGT
    A = F[(F.variant == "SXr") & (F.how == "RETRY")].copy()
    A["green"] = pd.to_datetime(A.green_t - MS5, unit="ms").dt.strftime("%m-%d %H:%M")
    A["entry_bar"] = pd.to_datetime(A.t_close - MS5, unit="ms").dt.strftime("%m-%d %H:%M")
    rows = [dict(pair=r.pair, era=r.era, g=r.green, e=r.entry_bar, w=f"{r.wait_min:.0f}", mv=fmt(r.move_from_green, 2), h=f"{r.hours:.1f}",
                 vm=f"{r.vol_mult:.0f}", tp=f"+{r.tp:.0f}", pct=fmt(r.pct, 2), why=r.why, usd=fmt(r.usd, 0),
                 src=("1m" if r.n_bar > 0 else "ticks")) for r in A.sort_values("t_close").itertuples()]
    lines.append("### ① per-fill list — primary SXr retry fills (green bar = bar OPEN time UTC; entry at that bar's close + 8 s)\n")
    lines.append(table(rows, ["pair", "era", "g", "e", "w", "mv", "h", "vm", "tp", "pct", "why", "usd", "src"],
                       ["pair", "batch", "green bar", "red bar", "wait min", "move %", "age h", "vol ×", "TP", "net %", "exit", "$", "path"]))
    lines.append("")
    # refusals at the retry bar
    Lg = pd.read_pickle(f"{OUTD}/era_log.pkl")
    rc = Lg[(Lg.variant == "SXr") & Lg.code.str.startswith("RETRY")].code.value_counts().to_dict()
    lines.append(f"SXr retry outcomes: {rc}\n")
    # mechanism
    added = F[(F.variant == "SXr") & (F.how == "RETRY")]
    mrows = mech(V0, added, "era") + mech(V0, F[(F.variant == "W60") & (F.how == "RETRY")], "era W60")[1:]
    pd.DataFrame(extra).to_csv(f"{OUT}_era_variants.csv", index=False)
    F.to_csv(f"{OUT}_era_fills.csv", index=False)
    return mrows


def bt_block(lines):
    D = os.path.join(OUTD, "bt")
    if not os.path.exists(f"{D}/bt_fills.pkl"):
        lines.append("## ③ backtest — NOT RUN\n"); return []
    F = pd.read_pickle(f"{D}/bt_fills.pkl")
    V0 = F[F.variant == "V0"]
    allday = sorted(set(F.te.map(dayof)))
    per = lambda ms: np.where(ms < pd.Timestamp("2026-01-01").value // 1e6, "2025", np.where(ms < pd.Timestamp("2026-07-01").value // 1e6, "2026-H1", "2026-H2"))
    F["period"] = per(F.te.values); F["month"] = pd.to_datetime(F.te, unit="ms").dt.strftime("%Y-%m")
    rows, extra = [], []
    for v in VN[1:]:
        A = F[F.variant == v]
        o, added, lost, dd = compare(A, V0, allday)
        c = concentration(added)
        rows.append(vrow(v, o, c)); o["variant"] = v; o.update(c); extra.append(o)
    r0 = V0
    rows.insert(0, dict(v="V0 (today)", N=len(r0), WR=f"{(r0.pct > 0).mean() * 100:.0f} %", avg=fmt(r0.pct.mean(), 3), usd=fmt(r0.usd.sum(), 0),
                        d="–", P="–", add="–", addwr=f"be {be_wr(r0.pct.values) * 100:.0f} %", addavg="–", lost="–",
                        wd=fmt(r0.groupby(r0.te.map(dayof)).usd.sum().min(), 0)))
    lines.append("## ③ Two-year backtest (engine-function LITE enumeration 2025-01-01 → 2026-10-11, LITE C, $2,450, slots / pair-held / WILLY hold)\n")
    lines.append(table(rows, VCOLS + ["tp", "td", "t5"], VHEAD + ["top pair share of added", "top day share", "added avg, top-5 removed"]))
    lines.append("")
    # reachable-universe sensitivity: only episodes (pair, spike) in which the yr6 replay (seed 1) took ANY FRENZY-family fill
    Fr = pd.read_csv("reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv", low_memory=False, usecols=["pair", "seed", "entry_strategy", "entry_frenzy_spike_at"])
    Fr = Fr[(Fr.seed == 1) & Fr.entry_strategy.str.startswith("FRENZY")]
    reach = set(zip(Fr.pair, pd.to_datetime(Fr.entry_frenzy_spike_at).dt.as_unit("ms").astype("int64")))
    FR = F[[k in reach for k in zip(F.pair, F.spike_ts)]]
    V0r = FR[FR.variant == "V0"]; rrow = []
    for v in ["SXr", "S15r", "S30r", "S60r", "SXn", "W60"]:
        o, added, lost, dd = compare(FR[FR.variant == v], V0r, sorted(set(FR.te.map(dayof))))
        rrow.append(vrow(v, o))
    rrow.insert(0, dict(v="V0 (today)", N=len(V0r), WR=f"{(V0r.pct > 0).mean() * 100:.0f} %", avg=fmt(V0r.pct.mean(), 3), usd=fmt(V0r.usd.sum(), 0)))
    lines.append(f"### ③ sensitivity — replay-reachable episodes only ({len(reach)} episodes with a yr6 seed-1 FRENZY-family fill)\n")
    lines.append(table(rrow, VCOLS, VHEAD))
    lines.append("")
    # per period for key variants
    prow = []
    for v in ["SXr", "S15r", "S30r", "S60r", "SXn", "W60", "RETRY_ANY"]:
        for pdn in ["2025", "2026-H1", "2026-H2"]:
            A = F[(F.variant == v) & (F.period == pdn)]; B = V0[V0.period == pdn] if "period" in V0 else F[(F.variant == "V0") & (F.period == pdn)]
            B = F[(F.variant == "V0") & (F.period == pdn)]
            days = [d for d in allday if per(np.array([pd.Timestamp(d).value // 1e6]))[0] == pdn]
            o, added, lost, dd = compare(A, B, days)
            prow.append(dict(v=v, p=pdn, n0=f"{len(B)} · {fmt(B.pct.mean(), 3)}", add=f"{o['added']} · {o['added_WR'] * 100:.0f} %" if o["added"] else "0",
                             aa=f"{fmt(o['added_avg'], 3)} [{fmt(o['added_lo'], 2)}, {fmt(o['added_hi'], 2)}]", lost=f"{o['lost']}",
                             d=f"{fmt(o['d_usd'], 0)} [{fmt(o['d_lo'], 0)}, {fmt(o['d_hi'], 0)}]", P=f"{o['P_gain']:.2f}"))
    lines.append("### ③ per period\n")
    lines.append(table(prow, ["v", "p", "n0", "add", "aa", "lost", "d", "P"],
                       ["variant", "period", "V0 LITE N · avg %", "added N · WR", "added avg % [day CI]", "lost", "Δ $ [day CI]", "P(Δ>0)"]))
    lines.append("")
    # per month + LOMO for primary and the best era cells
    mrow = []
    for v in ["SXr", "S15r", "W60"]:
        A = F[F.variant == v]
        ms = sorted(F.month.unique())
        dm = {m: A[A.month == m].usd.sum() - V0[F.loc[V0.index, "month"] == m].usd.sum() for m in ms}
        am = {m: A[(A.month == m) & (A.how == "RETRY")] for m in ms}
        tot = sum(dm.values())
        lomo = [tot - dm[m] for m in ms]
        mrow.append(dict(v=v, pos=f"{sum(1 for m in ms if dm[m] > 0)} / {len(ms)}", tot=fmt(tot, 0),
                         lomo=f"{fmt(min(lomo), 0)} … {fmt(max(lomo), 0)}", worst=f"{min(dm, key=dm.get)} {fmt(min(dm.values()), 0)}",
                         best=f"{max(dm, key=dm.get)} {fmt(max(dm.values()), 0)}",
                         months=" · ".join(f"{m[2:]} {dm[m]:+.0f}" for m in ms)))
    lines.append("### ③ per month Δ $ vs V0 and leave-one-month-out\n")
    lines.append(table(mrow, ["v", "pos", "tot", "lomo", "worst", "best", "months"],
                       ["variant", "months Δ>0", "Σ Δ $", "LOMO Σ Δ range", "worst month", "best month", "per month Δ $"]))
    lines.append("")
    Lg = pd.read_pickle(f"{D}/bt_log.pkl")
    lines.append("V0 judged bars: " + str(Lg[Lg.variant == "V0"].code.value_counts().to_dict()) + "  \nSXr retry outcomes: " +
                 str(Lg[(Lg.variant == "SXr") & Lg.code.str.startswith("RETRY")].code.value_counts().to_dict()) + "\n")
    added = F[(F.variant == "SXr") & (F.how == "RETRY")]
    mrows = mech(V0, added, "bt") + mech(V0, F[(F.variant == "W60") & (F.how == "RETRY")], "bt W60")[1:]
    pd.DataFrame(extra).to_csv(f"{OUT}_bt_variants.csv", index=False)
    F[(F.variant == "V0") | (F.how == "RETRY")].to_csv(f"{OUT}_bt_fills.csv", index=False)   # V0 + every variant's retry fills
    return mrows


def theory(G, PA, BT, R, label):
    """G = V0 green judged bars (pair, t_close, vol_mult); PA = prices incl. the green bars; R = SXr retry fills.
    → green-bar entry (what dropping the colour rule would buy) vs the retry entry on the same stretches, non-bearish only."""
    idx = BT.index.values
    rows = []
    for g in G.itertuples():
        pr = PA.get(f"{g.pair}|{int(g.t_close)}")
        j = int(np.searchsorted(idx, g.t_close, side="right")) - 1
        b = BT.iloc[j] if j >= 0 else None
        bear = bool(b is not None and np.isfinite(b.d1) and np.isfinite(b.gap) and b.d1 < 0 and b.gap < 0)
        if pr is None or "err" in pr:
            continue
        tp = 3 if (np.isfinite(g.vol_mult) and g.vol_mult >= 20 and b is not None and np.isfinite(b.h1slope) and b.h1slope > 0) else 2
        rows.append(dict(pair=g.pair, t_close=int(g.t_close), bear=bear, disloc=pr.get("disloc", False), g_pct=pr[f"lite{tp}_pct"],
                         g_bar=g.bar_ret))
    T = pd.DataFrame(rows)
    if not len(T):
        return []
    R = R.assign(gk=list(zip(R.pair, R.green_t)))
    rm = dict(zip(R.gk, R.pct))
    T["r_pct"] = [rm.get((p, t), np.nan) for p, t in zip(T.pair, T.t_close)]
    nb = T[~T.bear & ~T.disloc]
    pr_ = nb[nb.r_pct.notna()]
    out = [dict(c=f"{label}: enter ON the green bar (all non-bearish green judged bars)", N=len(nb), WR=f"{(nb.g_pct > 0).mean() * 100:.0f} %",
                avg=fmt(nb.g_pct.mean(), 3)),
           dict(c=f"{label}: same stretches with a retry fill — green-bar entry", N=len(pr_), WR=f"{(pr_.g_pct > 0).mean() * 100:.0f} %",
                avg=fmt(pr_.g_pct.mean(), 3)),
           dict(c=f"{label}: same stretches — retry (next red) entry", N=len(pr_), WR=f"{(pr_.r_pct > 0).mean() * 100:.0f} %",
                avg=fmt(pr_.r_pct.mean(), 3)),
           dict(c=f"{label}: same stretches — outcome changed (green → retry)", N=int((np.sign(pr_.g_pct) != np.sign(pr_.r_pct)).sum()),
                WR="", avg=f"Δ {fmt((pr_.r_pct - pr_.g_pct).mean(), 3)}")]
    big = nb[nb.g_bar >= nb.g_bar.median()]; small = nb[nb.g_bar < nb.g_bar.median()]
    out.append(dict(c=f"{label}: green-bar entry, green bar ≥ median ret ({nb.g_bar.median():+.2f} %)", N=len(big),
                    WR=f"{(big.g_pct > 0).mean() * 100:.0f} %", avg=fmt(big.g_pct.mean(), 3)))
    out.append(dict(c=f"{label}: green-bar entry, green bar < median ret", N=len(small), WR=f"{(small.g_pct > 0).mean() * 100:.0f} %",
                    avg=fmt(small.g_pct.mean(), 3)))
    return out


def main():
    lines = ["# LITE green-retry deep dive — tables (generated by scripts/study_lite_green_retry_deep_report.py)\n"]
    m1 = era_block(lines)
    m3 = bt_block(lines)
    th = []
    Lg = pd.read_pickle(f"{OUTD}/era_log.pkl"); F = pd.read_pickle(f"{OUTD}/era_fills.pkl")
    G = Lg[(Lg.variant == "V0") & (Lg.code == "FRENZY_LITE_GREEN_BAR")]
    th += theory(G, pd.read_pickle(f"{OUTD}/green_priced.pkl"), GR.EN.btc_frame(), F[(F.variant == "SXr") & (F.how == "RETRY")], "era")
    Dd = os.path.join(OUTD, "bt")
    if os.path.exists(f"{Dd}/bt_fills.pkl"):
        import study_lite_green_retry_deep_bt as B
        Lb = pd.read_pickle(f"{Dd}/bt_log.pkl"); Fb = pd.read_pickle(f"{Dd}/bt_fills.pkl")
        th += theory(Lb[(Lb.variant == "V0") & (Lb.code == "FRENZY_LITE_GREEN_BAR")], pd.read_pickle(f"{Dd}/priced.pkl"), B.btc_frame(),
                     Fb[(Fb.variant == "SXr") & (Fb.how == "RETRY")], "bt")
    lines.append("## Theory — entering ON the green bar vs waiting for the next red (LITE C, non-bearish, no slots)\n")
    lines.append(table(th, ["c", "N", "WR", "avg"], ["cohort", "N", "WR", "avg %"]))
    lines.append("")
    lines.append("## Mechanism — retry fills vs today's normal LITE fills\n")
    lines.append(table(m1 + m3, MCOLS, MHEAD))
    open(f"{OUT}_tables.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
