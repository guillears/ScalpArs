#!/usr/bin/env python3
"""B25 prep — block ③ BACKTEST for the WILLY V2b exit vs its scout shadows (DL 356 / 357 / 358 / 364; scripts/scout_willy_exit_v2.py).

Every fill is re-walked on its own path with the SCOUT's walker (scout_willy_exit_v2.walk_v2 / shadow_walk — the live V2b exit at
stop −4 · late +0.25 @30 · cap 90 · resting maker TP):
  OLD   = pinned +1.25 exit (shadow a)            V2b = target max(+1.25, min(+8, net at H × 0.99))   (live)
  FLOOR10 · STEPS · TRAIL10 · TRAIL15 · HALF (floor exits pay 0.05) · NOLATE · TGT15 / TGT20 (k 1.5 / 2.0) · LATE45 / LATE60
  Target accounting = the walker's net (gross − 0.045 − 0.045 × exit/entry; scripts/study_willy_hightarget_common.tgt_raw) — the
  engine uses the fill's own entry fee + maker exit fee; the difference is ≤ ≈ 0.03 pts on the target.
Arm A (WILLY, engine replay): reports/WILLY_HIGHTARGET_2026-10-23_bt.csv rows (yr6a 2025 + yr6 2026, seeds 1-3, warm-up trimmed, entry A
  ≤ 60 min after the spike) — the replay config ran the turnover filter (R < 1) → these are WILLY fills only; 1m paths
  (reports/WILLY_LATE_JOIN_2026-10-22_data/k1m), H = max 1m high over [spike bar open, red close) — the study's own conventions.
Arm B (WILLY vs WILLY_HT split, study cohort): reports/WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv (today's WILLY entry A replica,
  2026-01-10 → 10-07, one realisation, NOT an engine replay: no slots / hold / pair-held / HT single slot) — WILLY = R_red < 1,
  WILLY_HT = R_red ≥ 1, R unread → no entry (counted). Tick paths first (study_b19_fzexit_common.load_path), H from ticks else 1m.
Outputs: reports/PREP_B25_BT_WILLY_rows.csv · _summary.csv · _monthly.csv
"""
import argparse, os, sys
import multiprocessing as mp
import numpy as np
import pandas as pd

H_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H_)
import prep_b25_bt_common as S              # noqa: E402
import scout_willy_exit_v2 as W             # noqa: E402
import study_willy_hightarget_common as C   # noqa: E402
import study_willy_hightarget_bt as HB      # noqa: E402
import study_b19_fzexit_common as FX        # noqa: E402

R = os.path.dirname(H_)
OUT = os.path.join(R, "reports", "PREP_B25_BT_WILLY")
MIN, BAR = 60_000, 300_000
NAMES = ["OLD", "V2b"] + list(W.SHADOWS)


def tgt(H, E, k):
    return float(min(max(C.tgt_raw(H, E, k), W.OLD_TP), W.TGT_MAX))


def walk(P, H, E):
    o = {}
    a = W.shadow_today(P); o["OLD"] = (a["pct"], a["why"])
    T = tgt(H, E, W.LIVE_K)
    v = W.walk_v2(P, T); o["V2b"] = (v["pct"] if v["why"] != "UNRES" else np.nan, v["why"])
    for n in W.SHADOWS:
        if n in W.TARGET_KS:
            w = W.walk_v2(P, tgt(H, E, W.TARGET_KS[n])); o[n] = (w["pct"] if w["why"] != "UNRES" else np.nan, w["why"])
        else:
            o[n] = (W.shadow_walk(P, T, n), "")
    return T, o


def work_a(rows):
    out = []
    for r in rows:
        te, sp, E = int(r["te"]), int(r["sp"]), float(r["E"])
        red = (te // BAR) * BAR
        days = sorted({pd.Timestamp(x, unit="ms").strftime("%Y-%m-%d") for x in (sp - BAR, te, te + (C.HORIZON + 1) * MIN)})
        a = HB.k1(r["pair"], days)
        d = dict(arm="A_replay", sleeve="FRENZY_WILLY", prefix=r["prefix"], seed=int(r["seed"]), pair=r["pair"], te=te, E=E)
        if a is None:
            d["st"] = "NODATA"; out.append(d); continue
        s = a[(a[:, 0] >= sp - BAR) & (a[:, 0] < red)]
        if len(s) != (red - (sp - BAR)) // MIN or not len(s):
            d["st"] = "NOHIGH"; out.append(d); continue
        H = float(s[:, 2].max()); P = HB.path(a, te, E, C.HORIZON)
        T, o = walk(P, H, E)
        d.update(st="ok", H=H, T=T, dist=(H / E - 1) * 100, n_tick=0)
        for n, (p, why) in o.items():
            d[n] = p; d[n + "_why"] = why
        out.append(d)
    return out


def work_b(rows):
    out = []
    tc = {}
    for r in rows:
        te, E, t0, tr = int(r["te"]), float(r["E"]), int(r["t0"]), int(r["t_red"])
        d = dict(arm="B_study", sleeve=r["sleeve"], prefix="willyA", seed=1, pair=r["pair"], te=te, E=E, R_red=r["R_red"])
        H = C.spike_high_ticks(r["pair"], t0, tr, tc)
        src = "ticks"
        if H is None:
            k = FX.WT.load_1m(r["pair"])
            if len(k):
                H = C.spike_high_1m(k, t0, tr); src = "1m"
        if H is None:
            d["st"] = "NOHIGH"; out.append(d); continue
        P = FX.load_path(r["pair"], te, E, horizon=C.HORIZON, tcache=tc)
        T, o = walk(P, H, E)
        d.update(st="ok", H=H, H_src=src, T=T, dist=(H / E - 1) * 100, n_tick=P["n_tick"])
        for n, (p, why) in o.items():
            d[n] = p; d[n + "_why"] = why
        out.append(d)
        if len(tc) > 40:
            tc.clear()
    return out


def chunks(rows, n):
    pairs = sorted({r["pair"] for r in rows}); g = {p: i % n for i, p in enumerate(pairs)}
    return [[r for r in rows if g[r["pair"]] == k] for k in range(n)]


def build(procs):
    h = pd.read_csv(os.path.join(R, "reports", "WILLY_HIGHTARGET_2026-10-23_bt.csv"))
    h = h[h.st == "ok"]
    ra = h[["prefix", "seed", "pair", "te", "sp", "E"]].to_dict("records")
    b = pd.read_csv(os.path.join(R, "reports", "WILLY_TURNOVER_REPRICE_2026-10-19_willyA.csv"))
    b = b[b.st == "ok"].copy()
    b["sleeve"] = np.where(b.R_red.isna(), "UNREAD", np.where(b.R_red >= 1.0, "FRENZY_WILLY_HT", "FRENZY_WILLY"))
    unread = int((b.sleeve == "UNREAD").sum())
    b = b[b.sleeve != "UNREAD"]
    rb = b[["pair", "te", "E", "t0", "t_red", "R_red", "sleeve"]].to_dict("records")
    with mp.Pool(procs) as P:
        A = sum(P.map(work_a, chunks(ra, procs * 4)), [])
        B = sum(P.map(work_b, chunks(rb, procs * 4)), [])
    D = pd.DataFrame(A + B)
    D.attrs["unread"] = unread
    t = pd.to_datetime(D.te, unit="ms")
    D["day"] = t.dt.strftime("%Y-%m-%d"); D["month"] = t.dt.strftime("%Y-%m")
    D["period"] = np.where(t < "2026-01-01", "2025", np.where(t < "2026-07-01", "2026-H1", "2026-H2"))
    D["runner"] = D["T"] > W.OLD_TP + 1e-9
    D.to_csv(OUT + "_rows.csv", index=False)
    print("arm B R-unread (no entry, excluded):", unread)
    return D


def analyse(D):
    D = D[D.st == "ok"].copy()
    rows, mrows = [], []
    groups = []
    for (pre, sd), g in D[D.arm == "A_replay"].groupby(["prefix", "seed"]):
        pass
    A = D[D.arm == "A_replay"]
    for sd in (1, 2, 3):
        groups.append((f"A_replay s{sd}", "FRENZY_WILLY", A[A.seed == sd]))
    B = D[D.arm == "B_study"]
    for sl in ("FRENZY_WILLY", "FRENZY_WILLY_HT"):
        groups.append(("B_study", sl, B[B.sleeve == sl]))
    for arm, sl, G in groups:
        for per in ("ALL", "2025", "2026-H1", "2026-H2", "2026"):
            P0 = G if per == "ALL" else (G[G.period.isin(["2026-H1", "2026-H2"])] if per == "2026" else G[G.period == per])
            for pop in ("all", "runners"):
                P = P0 if pop == "all" else P0[P0.runner]
                if len(P) < 1:
                    continue
                for n in NAMES:
                    Q = P[P[n].notna() & P.V2b.notna()]
                    if not len(Q):
                        continue
                    r = S.summ(Q.assign(_x=Q[n]), col="_x")
                    dd = (Q[n] - Q.V2b).values
                    lo, hi, pn = S.day_boot(dd, Q.day.values)
                    pday = pd.Series(dd, index=Q.day.values).groupby(level=0).sum()
                    pfill = pd.Series(dd)
                    sp = {}
                    for cl in ("STOP", "TP", "LATE_TP", "CAP"):
                        m = (Q.V2b_why == cl).values
                        x = dd[m]
                        sp[f"v2b_{cl}_n"] = int(m.sum()); sp[f"v2b_{cl}_up"] = round(float(x[x > 1e-9].sum()), 1); sp[f"v2b_{cl}_dn"] = round(float(x[x < -1e-9].sum()), 1)
                    rows.append(dict(arm=arm, sleeve=sl, period=per, pop=pop, variant=n, **r, dAvg=round(float(dd.mean()), 4),
                                     dCI_lo=round(lo, 4), dCI_hi=round(hi, 4), dP_pos=round(1 - pn, 3) if np.isfinite(pn) else np.nan,
                                     top_day_share_posD=round(float(pday.max() / pday[pday > 0].sum()), 2) if (pday > 0).any() else np.nan,
                                     top_fill_share_posD=round(float(pfill.max() / pfill[pfill > 0].sum()), 2) if (pfill > 0).any() else np.nan,
                                     runners=int(Q.runner.sum()), **sp))
        for (m, ), g in G.groupby(["month"]):
            for n in NAMES:
                q = g[g[n].notna() & g.V2b.notna()]
                if len(q):
                    mrows.append(dict(arm=arm, sleeve=sl, month=m, variant=n, N=len(q), runners=int(q.runner.sum()), WR=round(100 * (q[n] > 0).mean(), 1),
                                      avg=round(q[n].mean(), 4), dAvg_vs_V2b=round((q[n] - q.V2b).mean(), 4)))
    Sm = pd.DataFrame(rows); Mm = pd.DataFrame(mrows)
    Sm.to_csv(OUT + "_summary.csv", index=False); Mm.to_csv(OUT + "_monthly.csv", index=False)
    cons = []
    for (arm, sl, n), g in Mm[Mm.variant != "V2b"].groupby(["arm", "sleeve", "variant"]):
        g = g[g.N >= 10]
        cons.append(dict(arm=arm, sleeve=sl, variant=n, months=len(g), better=int((g.dAvg_vs_V2b > 0).sum()), worse=int((g.dAvg_vs_V2b < 0).sum())))
    pd.DataFrame(cons).to_csv(OUT + "_monthcons.csv", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    k = Sm[(Sm.period.isin(["ALL", "2026"]))]
    print(k[["arm", "sleeve", "period", "pop", "variant", "N", "runners", "WR", "avg", "ci_lo", "ci_hi", "dAvg", "dCI_lo", "dCI_hi", "dP_pos",
             "top_day_share_posD", "top_fill_share_posD"]].to_string())
    print(pd.DataFrame(cons).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--reuse", action="store_true"); A_ = ap.parse_args()
    D = pd.read_csv(OUT + "_rows.csv") if A_.reuse and os.path.exists(OUT + "_rows.csv") else build(A_.procs)
    print(D.groupby(["arm", "sleeve", "st"]).size())
    analyse(D)
