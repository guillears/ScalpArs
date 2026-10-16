#!/usr/bin/env python3
"""LITE THEORY — step 4: tables + pre-registered gates → reports/lite_theory/tables.md, gates.csv (the hand-written summary sits on top
of these in reports/LITE_THEORY_2026-10-15.md)."""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_lite_theory_common as T  # noqa: E402
import study_lite_theory_signals as S  # noqa: E402

LS = -0.20
O = T.OUT
LIVE_CSV = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_19-55-15.csv")
MASTER_FROM = int(pd.Timestamp("2026-10-06", tz="UTC").value // 1_000_000)
L = []


def w(s=""):
    L.append(s)


def f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def load():
    B = S.load_bars()
    Y1 = pd.read_pickle(os.path.join(O, "y1m.pkl"))
    Cc = pd.read_pickle(os.path.join(O, "cells.pkl"))
    RH = pd.read_pickle(os.path.join(O, "yrh.pkl"))
    base = pd.read_pickle(os.path.join(O, "base_judged.pkl"))
    A = pd.read_pickle(os.path.join(O, "a67.pkl"))
    AM = pd.read_pickle(os.path.join(O, "a67_master.pkl"))
    M = pd.read_pickle(os.path.join(O, "master_rows.pkl"))
    return B, Y1, Cc, RH, base, A, AM, M


def feats(B, base):
    """per-bar features used by the filter cells."""
    F = B[["pair", "T", "spike_ts", "hours", "streak", "vol_mult", "vs_vwap", "run_pct", "spike_volx", "spike_high", "hi_since", "c",
           "day", "month", "year", "era", "ep"]].copy()
    F["dist_hi"] = (F.c / F.spike_high - 1) * 100
    F["newhi"] = F.hi_since > F.spike_high
    F = F.merge(base[["pair", "T", "ord"]], on=["pair", "T"], how="left")
    F["prior"] = F["ord"] - 1
    return F


def bucket(v, edges, labels):
    return pd.cut(v, edges, labels=labels, right=False)


FILTERS = {   # name → (feature, predicate of the DROPPED signals)
    "A2_ORD1": ("prior", lambda x: x >= 1),          # keep only the 1st judged stretch → drop prior ≥ 1
    "A2_ORD2P": ("prior", lambda x: x < 1),
    "A5_DROPLT20": ("vol_mult", lambda x: x < 20),
    "A6_CAP2": ("vs_vwap", lambda x: x > 2), "A6_CAP3": ("vs_vwap", lambda x: x > 3), "A6_CAP5": ("vs_vwap", lambda x: x > 5),
    "A6_FLOOR1": ("vs_vwap", lambda x: x < 1),
    "A8_run_lt10": ("run_pct", lambda x: x < 10), "A8_run_10_20": ("run_pct", lambda x: (x >= 10) & (x < 20)), "A8_run_ge20": ("run_pct", lambda x: x >= 20),
    "A8_svx_lt50": ("spike_volx", lambda x: x < 50), "A8_svx_50_100": ("spike_volx", lambda x: (x >= 50) & (x < 100)),
    "A8_svx_ge100": ("spike_volx", lambda x: x >= 100),
    "A8_dhi_ge-3": ("dist_hi", lambda x: x >= -3), "A8_dhi_-3_-8": ("dist_hi", lambda x: (x < -3) & (x >= -8)), "A8_dhi_lt-8": ("dist_hi", lambda x: x < -8),
    "A8_prior_0": ("prior", lambda x: x == 0), "A8_prior_1": ("prior", lambda x: x == 1), "A8_prior_ge2": ("prior", lambda x: x >= 2),
    "A9_NEWHI": ("newhi", lambda x: ~x.astype(bool)), "A9_NONEWHI": ("newhi", lambda x: x.astype(bool)),
}
EMU = ["A2_GREEN", "A2_ANY", "A2_WAITRED", "A3_N6", "A3_N24", "A3_N36", "A4_H2_8", "A4_H2_12", "A4_H4_1675", "A4_H8_1675", "A4_H2_24",
       "A5_V20", "A5_V50"]
A6L = [f"A6_L{k:g}_W{w}" for k in (0.5, 1.0, 1.5) for w in (30, 60, 120)]
A7 = [f"A7_C5_b{b:g}" for b in (0, 0.5, 1)] + [f"A7_C1_b{b:g}" for b in (0, 0.5, 1)] + ["A7_R5"] + [f"A7_VS_b{b:g}" for b in (0, 0.5, 1)]
NEIGH = {"A3_N24": ["A3_N36"], "A3_N36": ["A3_N24"], "A4_H2_8": ["A4_H2_12"], "A4_H2_12": ["A4_H2_8"], "A4_H4_1675": ["A4_H8_1675"],
         "A4_H8_1675": ["A4_H4_1675"], "A5_V20": ["A5_V50"], "A5_V50": ["A5_V20"], "A6_CAP2": ["A6_CAP3"], "A6_CAP3": ["A6_CAP2", "A6_CAP5"],
         "A6_CAP5": ["A6_CAP3"], "A8_run_lt10": ["A8_run_10_20"], "A8_run_10_20": ["A8_run_lt10", "A8_run_ge20"], "A8_run_ge20": ["A8_run_10_20"],
         "A8_svx_lt50": ["A8_svx_50_100"], "A8_svx_50_100": ["A8_svx_lt50", "A8_svx_ge100"], "A8_svx_ge100": ["A8_svx_50_100"],
         "A8_dhi_ge-3": ["A8_dhi_-3_-8"], "A8_dhi_-3_-8": ["A8_dhi_ge-3", "A8_dhi_lt-8"], "A8_dhi_lt-8": ["A8_dhi_-3_-8"],
         "A8_prior_0": ["A8_prior_1"], "A8_prior_1": ["A8_prior_0", "A8_prior_ge2"], "A8_prior_ge2": ["A8_prior_1"]}
for _k in (0.5, 1.0, 1.5):
    for _w in (30, 60, 120):
        nb = []
        for kk in (0.5, 1.0, 1.5):
            for ww in (30, 60, 120):
                if (kk, ww) != (_k, _w) and ((kk == _k and abs((30, 60, 120).index(ww) - (30, 60, 120).index(_w)) == 1) or
                                             (ww == _w and abs((0.5, 1.0, 1.5).index(kk) - (0.5, 1.0, 1.5).index(_k)) == 1)):
                    nb.append(f"A6_L{kk:g}_W{ww}")
        NEIGH[f"A6_L{_k:g}_W{_w}"] = nb
for fam in ("C5", "C1", "VS"):
    for b, nb in ((0, [0.5]), (0.5, [0, 1]), (1, [0.5])):
        NEIGH[f"A7_{fam}_b{b:g}"] = [f"A7_{fam}_b{x:g}" for x in nb] + ([f"A7_{'C1' if fam == 'C5' else 'C5'}_b{b:g}"] if fam != "VS" else [])
NEIGH["A7_R5"] = ["A7_C5_b0"]


def stats_rows(R):
    """R: rows (day, month, year, pair, v, t) — v = variant value, t = today value (0 where absent). → dict of gate stats."""
    R = R.copy(); R["d"] = R.v.fillna(0) - R.t.fillna(0)
    out = {}
    for y in ("2025", "2026"):
        r = R[R.year == y]
        dd = r.groupby("day").d.sum()
        p, lo, hi = T.day_boot_diff(dd.values, dd.index.values)
        n_t = int(r.t.notna().sum())
        out[y] = dict(sum=float(r.d.sum()), per_today=float(r.d.sum() / max(n_t, 1)), P=p, n_days=len(dd),
                      nv=int(r.v.notna().sum()), nt=n_t, mv=float(r.v.mean()) if r.v.notna().any() else np.nan,
                      mt=float(r.t.mean()) if r.t.notna().any() else np.nan)
    mm = R.groupby("month").d.sum()
    tot = mm.sum()
    lo_ = [tot - x for x in mm.values]
    out["lomo"] = bool(len(lo_) and all(np.sign(x) == np.sign(tot) for x in lo_))
    out["day_share"], out["top_day"] = T.conc(R.groupby("day").d.sum().values, R.groupby("day").d.sum().index.values)
    pp = R.groupby("pair").d.sum()
    out["pair_share"], out["top_pair"] = T.conc(pp.values, pp.index.values)
    out["sum"] = float(R.d.sum())
    return out


def main():
    B, Y1, Cc, RH, base, A, AM, M = load()
    F = feats(B, base)
    sig = Cc.merge(RH[["pair", "T", "yh", "whyh", "srch"]], on=["pair", "T"], how="left").merge(Y1[["pair", "T", "y1", "why1"]], on=["pair", "T"], how="left")
    sig = sig.merge(F, on=["pair", "T"], how="left")
    tod = sig[sig.cell == "TODAY"].copy()
    nres = tod.yh.notna().sum()
    # live
    LV = pd.read_csv(LIVE_CSV, low_memory=False)
    LV = LV[LV.entry_strategy == "FRENZY_LITE"].copy()
    M["y"] = M.rm_base_pct + np.where(M.rm_base_why == "TP", 0.0, LS)
    M = M.rename(columns={"entry_frenzy_vs_vwap_pct": "vs_vwap_m", "entry_frenzy_vol_mult": "vol_mult_m", "entry_frenzy_run_pct": "run_pct_m"})
    M["dist_hi"] = (M.c / M.spike_high - 1) * 100
    M["newhi"] = M.hi_since > M.spike_high
    M = M.merge(base[["pair", "T", "ord"]], on=["pair", "T"], how="left")
    M["prior"] = M["ord"] - 1
    M["vs_vwap"] = M.vs_vwap_m; M["vol_mult"] = M.vol_mult_m; M["run_pct"] = M.run_pct_m

    w("# LITE THEORY — tables (generated by scripts/study_lite_theory_report.py)")
    w()
    w(f"Today's LITE signals (rebuilt, non-bearish): {len(tod)} · RH-resolved {nres} · R1-resolved {tod.y1.notna().sum()} · "
      f"RH source {tod.srch.value_counts().to_dict()}")
    for y in ("2025", "2026"):
        t = tod[tod.year == y]
        w(f"- {y}: N {len(t)} · RH avg {f(t.yh.mean())} · WR {f((t.yh > 0).mean() * 100, 1, False)}% · Σ {f(t.yh.sum(), 1)} · "
          f"R1 avg {f(t.y1.mean())}")
    w(f"- MASTER 20 fills (LS): avg {f(M.y.mean())} · WR {f((M.y > 0).mean() * 100, 1, False)}% · Σ {f(M.y.sum(), 2)}")
    w(f"- LIVE 4: {', '.join(f'{r.pair} {r.pnl_percentage:+.2f}' for r in LV.itertuples())}")
    w()

    # ═════════ A0 ═════════
    w("## A0 — does the LITE selection add anything over entering anywhere in the same episodes? (R1 1m ruler, LS slip)")
    w()
    U = Y1.merge(B[["pair", "T", "ep", "hours", "streak", "bar_red", "year", "era", "day", "above_share"]], on=["pair", "T"])
    Uw = U[(U.hours <= 16.75) & U.y1.notna()]
    sets = {"(i) every bar": Uw, "(ii) every close ≥ VWAP": Uw[Uw.streak >= 1], "(iii) every RED bar ≥ VWAP": Uw[(Uw.streak >= 1) & Uw.bar_red]}
    tt = tod[tod.y1.notna()].copy()
    w("Paired per LITE signal: Y(LITE) − mean Y of the same episode's bars in the set (window 2–16.75 h, not in state, vol < 100, non-bearish).")
    w()
    w("| year | baseline set | N signals | LITE avg | baseline avg | Δ/signal | P(Δ>0) day-boot | 95% CI |")
    w("|---|---|---|---|---|---|---|---|")
    a0rows = []
    first_sig = tt.groupby("ep")["T"].min().rename("T0")
    variants = []
    for nm, Sx in sets.items():
        variants.append((nm + " [as pre-registered: whole window — LOOK-AHEAD biased]", Sx))
        Sy = Sx.merge(first_sig, left_on="ep", right_index=True)
        variants.append((nm + " [corrected: bars at/after the episode's first LITE signal]", Sy[Sy["T"] >= Sy.T0]))
    for nm, Sx in variants:
        mb = Sx.groupby("ep").y1.mean().rename("bm")
        x = tt.merge(mb, left_on="ep", right_index=True, how="inner")
        x["d"] = x.y1 - x.bm
        for y in ("2025", "2026", "all"):
            xx = x if y == "all" else x[x.year == y]
            p, lo, hi = T.day_boot_diff(xx.d.values, xx.day.values)
            w(f"| {y} | {nm} | {len(xx)} | {f(xx.y1.mean())} | {f(xx.bm.mean())} | {f(xx.d.mean())} | {f(p, 2, False)} | [{f(lo)}, {f(hi)}] |")
            a0rows.append(dict(set=nm, year=y, n=len(xx), lite=xx.y1.mean(), base=xx.bm.mean(), d=xx.d.mean(), P=p, lo=lo, hi=hi))
    w()
    w("Unpaired, ALL flagged episodes (not only those holding a LITE signal), same window — mean Y per entry, TP-first share:")
    w()
    w("| year | set | N bars | episodes | avg Y | TP-first % |")
    w("|---|---|---|---|---|---|")
    for y in ("2025", "2026"):
        for nm, Sx in sets.items():
            s = Sx[Sx.year == y]
            w(f"| {y} | {nm} | {len(s):,} | {s.ep.nunique():,} | {f(s.y1.mean())} | {f((s.why1 == 'TP').mean() * 100, 1, False)} |")
        s = tt[tt.year == y]
        w(f"| {y} | **today's LITE signals** | {len(s):,} | {s.ep.nunique():,} | {f(s.y1.mean())} | {f((s.why1 == 'TP').mean() * 100, 1, False)} |")
    w()
    pd.DataFrame(a0rows).to_csv(os.path.join(O, "a0.csv"), index=False)

    # ═════════ A1 ═════════
    w("## A1 — does 'held above VWAP' predict continuation? (all bars, window 2–16.75 h, R1)")
    w()
    Uw = Uw.copy()
    Uw["sb"] = pd.cut(Uw.streak, [-1, 0, 5, 11, 23, 35, 10_000], labels=["0", "1–5", "6–11", "12–23", "24–35", "36+"])
    Uw["ab"] = pd.cut(Uw.above_share, [-1, 50, 70, 85, 101], labels=["<50%", "50–70%", "70–85%", "≥85%"], right=False)
    for col, lab in (("sb", "above_streak"), ("ab", "above_share (time above VWAP since the spike)")):
        w(f"**by {lab}**")
        w()
        w("| year | bucket | N bars | episodes | avg Y (+3/−3/12h) | day-boot 95% CI | TP-first % | fwd 1h % | fwd 4h % | fwd 12h % |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for y in ("2025", "2026"):
            for bk, s in Uw[Uw.year == y].groupby(col, observed=True):
                p, lo, hi = T.day_boot_diff(s.y1.values, s.day.values, n=1500)
                w(f"| {y} | {bk} | {len(s):,} | {s.ep.nunique():,} | {f(s.y1.mean())} | [{f(lo)}, {f(hi)}] | {f((s.why1 == 'TP').mean() * 100, 1, False)} | "
                  f"{f(s.f60.mean(), 2)} | {f(s.f240.mean(), 2)} | {f(s.f720.mean(), 2)} |")
        w()

    # ═════════ bucket tables on today's signals ═════════
    w("## Bucket tables on today's signals (RH, LS) — three blocks: backtest per year · master (stamps) · live")
    w()
    BK = [("A4 hours since spike", "hours", [2, 4, 8, 12, 16.76, 99], ["2–4", "4–8", "8–12", "12–16.75", ">16.75"]),
          ("A5 vol_mult", "vol_mult", [0, 20, 50, 100], ["<20", "20–50", "50–100"]),
          ("A6 entry stretch above VWAP %", "vs_vwap", [-99, 1, 2, 3, 5, 999], ["0–1", "1–2", "2–3", "3–5", "5+"]),
          ("A8 spike run_pct %", "run_pct", [-99, 10, 20, 9999], ["<10", "10–20", "≥20"]),
          ("A8 spike volume multiple", "spike_volx", [0, 50, 100, 1e9], ["<50", "50–100", "≥100"]),
          ("A8 close vs spike-bar high %", "dist_hi", [-999, -8, -3, 999], ["<−8", "−8…−3", "≥−3"]),
          ("A8 prior judged stretches", "prior", [-1, 0.5, 1.5, 99], ["0", "1", "≥2"]),
          ("A9 new high above the spike-bar high since the spike", "newhi", None, None)]
    LV["vs_vwap"] = LV.entry_frenzy_vs_vwap_pct; LV["vol_mult"] = LV.entry_frenzy_vol_mult; LV["run_pct"] = LV.entry_frenzy_run_pct
    LV["hours"] = LV.entry_frenzy_hours
    for title, col, ed, lb in BK:
        w(f"**{title}**")
        w()
        w("| bucket | 2025 N · WR · avg | 2026 N · WR · avg | MASTER N · avg | LIVE (pnl %) |")
        w("|---|---|---|---|---|")
        tb = tod.copy(); mb = M.copy(); lv = LV.copy()
        if ed is None:
            tb["bk"] = tb[col].map({True: "yes", False: "no"}); mb["bk"] = mb[col].map({True: "yes", False: "no"})
            lv["bk"] = None; labs = ["yes", "no"]
        else:
            tb["bk"] = pd.cut(tb[col], ed, labels=lb, right=False); mb["bk"] = pd.cut(mb[col], ed, labels=lb, right=False) if col in mb else None
            lv["bk"] = pd.cut(lv[col], ed, labels=lb, right=False) if col in lv else None
            labs = lb
        for b in labs:
            cells = []
            for y in ("2025", "2026"):
                s = tb[(tb.year == y) & (tb.bk == b) & tb.yh.notna()]
                cells.append(f"{len(s)} · {f((s.yh > 0).mean() * 100, 0, False)}% · {f(s.yh.mean())}" if len(s) else "0")
            s = mb[mb.bk == b] if mb["bk"] is not None else mb.iloc[:0]
            lvv = lv[lv.bk == b] if lv["bk"] is not None else lv.iloc[:0]
            w(f"| {b} | {cells[0]} | {cells[1]} | {len(s)} · {f(s.y.mean()) if len(s) else '—'} | "
              f"{', '.join(f'{r.pair[:-4]} {r.pnl_percentage:+.2f}' for r in lvv.itertuples()) or '—'} |")
        if mb["bk"] is not None:
            w(f"| (master unscored) | | | {int(mb.bk.isna().sum())} | |")
        w()

    # ═════════ cells + gates ═════════
    G = []

    def gate_row(name, fam, R, master_d, master_note, r1=None):
        st = stats_rows(R)
        g1 = st["2025"]["P"] >= 0.90 and st["2026"]["P"] >= 0.90 and st["2025"]["sum"] > 0 and st["2026"]["sum"] > 0
        g2 = (master_d is None) or (master_d >= 0)
        g4 = (np.isfinite(st["day_share"]) and st["day_share"] < 0.5 and np.isfinite(st["pair_share"]) and st["pair_share"] < 0.5)
        G.append(dict(cell=name, fam=fam, d25=st["2025"]["sum"], pt25=st["2025"]["per_today"], P25=st["2025"]["P"], nv25=st["2025"]["nv"],
                      nt25=st["2025"]["nt"], mv25=st["2025"]["mv"], mt25=st["2025"]["mt"],
                      d26=st["2026"]["sum"], pt26=st["2026"]["per_today"], P26=st["2026"]["P"], nv26=st["2026"]["nv"], nt26=st["2026"]["nt"],
                      mv26=st["2026"]["mv"], mt26=st["2026"]["mt"], lomo=st["lomo"], day_share=st["day_share"], top_day=st["top_day"],
                      pair_share=st["pair_share"], top_pair=st["top_pair"], master_d=master_d, master_note=master_note,
                      g1=g1, g2=g2, g3=st["lomo"], g4=g4, r1P25=(r1 or {}).get("2025", {}).get("P"), r1P26=(r1 or {}).get("2026", {}).get("P"),
                      r1d25=(r1 or {}).get("2025", {}).get("sum"), r1d26=(r1 or {}).get("2026", {}).get("sum")))

    keyc = ["day", "month", "year", "pair"]
    k3 = ["day", "month", "year"]
    todk = tod[["pair", "T", "yh", "y1"] + k3]
    # master window (rebuilt) for replacement cells
    for nm in EMU:
        v = sig[sig.cell == nm][["pair", "T", "yh", "y1"] + k3]
        R = v.merge(todk, on=["pair", "T"] + k3, how="outer", suffixes=("_v", "_t"))
        Rh = R.rename(columns={"yh_v": "v", "yh_t": "t"})
        R1 = R.rename(columns={"y1_v": "v", "y1_t": "t"})
        mw = Rh[(Rh.year == "2026") & (pd.to_datetime(Rh.day) >= pd.Timestamp("2026-10-06"))]
        md = float(mw.v.fillna(0).sum() - mw.t.fillna(0).sum()) if len(mw) else None
        gate_row(nm, nm[:2], Rh[keyc + ["v", "t"]], md, f"REBUILT master window ({len(mw)} signal rows)", r1=stats_rows(R1[keyc + ["v", "t"]]))
    for nm, (col, drop) in FILTERS.items():
        t = tod[["pair", "T", "yh", "y1", col] + k3].copy()
        dmask = drop(t[col]).fillna(False).astype(bool) if t[col].dtype != bool else drop(t[col])
        t["vh"] = np.where(dmask, np.nan, t.yh); t["v1"] = np.where(dmask, np.nan, t.y1)
        mm = M[M[col].notna()] if col in M else M.iloc[:0]
        md = float(-mm.y[drop(mm[col]).astype(bool)].sum()) if len(mm) else None
        gate_row(nm, nm[:2], t.rename(columns={"vh": "v", "yh": "t"})[keyc + ["v", "t"]], md, f"master stamps scored {len(mm)}/20",
                 r1=stats_rows(t.rename(columns={"v1": "v", "y1": "t"})[keyc + ["v", "t"]]))
    Ar = A[A.ruler == "RH"].merge(tod[["pair", "T"] + k3], on=["pair", "T"])
    A1r = A[A.ruler == "R1"].merge(tod[["pair", "T"] + k3], on=["pair", "T"])
    AMr = AM[AM.ruler == "RH"]
    # A7 C5/C1 on master need the rebuilt VWAP path; rows without a rebuilt spike are n/a
    m_ok = M.set_index("fid").spike_ts.notna()
    for nm in A6L + A7:
        R = Ar[keyc + [nm, "TODAY"]].rename(columns={nm: "v", "TODAY": "t"})
        R = R[R.t.notna()]
        R.loc[R.v.isna(), "v"] = np.nan
        R1x = A1r[keyc + [nm, "TODAY"]].rename(columns={nm: "v", "TODAY": "t"}); R1x = R1x[R1x.t.notna()]
        am = AMr.copy()
        if nm.startswith("A7_C"):
            am = am[am.fid.map(m_ok).fillna(False).astype(bool)]
        md = float((am[nm] - am.TODAY).sum()) if len(am) else None
        gate_row(nm, nm[:2], R, md, f"master priced {len(am)}/20", r1=stats_rows(R1x))
    GD = pd.DataFrame(G)
    gd = GD.set_index("cell")
    pos = lambda c: (c in gd.index) and gd.loc[c, "d25"] > 0 and gd.loc[c, "d26"] > 0
    GD["g5"] = [all(pos(n) for n in NEIGH.get(c, [])) if NEIGH.get(c) else None for c in GD.cell]
    GD["cand"] = GD.g1 & GD.g2 & GD.g3 & GD.g4 & GD.g5.fillna(True).astype(bool)
    GD.to_csv(os.path.join(O, "gates.csv"), index=False)
    w(f"## Gated cells ({len(GD)} pre-registered) — RH ruler, LS slip; Δ = Σ variant − Σ today (%-points at 1×)")
    w()
    w("Columns: ΣΔ and Δ per today-signal per year · day-bootstrap P(Δ>0) · N signals variant / today · avg per signal variant / today · "
      "R1 (1m) P as sensitivity · master Δ · LOMO · largest day / pair share · neighbours · CANDIDATE")
    w()
    w("| cell | 2025 ΣΔ (per sig) · P | 2026 ΣΔ (per sig) · P | N v/t 25 | N v/t 26 | avg v/t 25 | avg v/t 26 | R1 P 25/26 | master Δ | LOMO | day / pair share | nbrs | CAND |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in GD.itertuples():
        w(f"| {r.cell} | {f(r.d25, 1)} ({f(r.pt25)}) · {f(r.P25, 2, False)} | {f(r.d26, 1)} ({f(r.pt26)}) · {f(r.P26, 2, False)} | "
          f"{r.nv25}/{r.nt25} | {r.nv26}/{r.nt26} | {f(r.mv25)}/{f(r.mt25)} | {f(r.mv26)}/{f(r.mt26)} | {f(r.r1P25, 2, False)}/{f(r.r1P26, 2, False)} | "
          f"{f(r.master_d, 2) if r.master_d is not None else 'n/a'} ({r.master_note}) | {'✓' if r.lomo else '✗'} | "
          f"{f(r.day_share, 2, False)} / {f(r.pair_share, 2, False)} | {'—' if r.g5 is None else ('✓' if r.g5 else '✗')} | {'**YES**' if r.cand else 'no'} |")
    w()
    n1 = int(((GD.P25 >= 0.9) & (GD.d25 > 0)).sum()); n2 = int(((GD.P26 >= 0.9) & (GD.d26 > 0)).sum())
    nb = int(GD.g1.sum())
    w(f"Cells with P ≥ 0.90 (Δ > 0): 2025 {n1}/{len(GD)} · 2026 {n2}/{len(GD)} · BOTH {nb} · full candidates {int(GD.cand.sum())}.")
    w()
    # expectancy read of filter cells' dropped cohorts
    w("**Filter cells — the dropped cohort at 1× (CLAUDE.md expectancy bar read)**")
    w()
    ok = tod[tod.yh.notna()]
    wins, loss = ok.yh[ok.yh > 0], ok.yh[ok.yh <= 0]
    be = abs(loss.mean()) / (wins.mean() + abs(loss.mean()))
    w(f"Sleeve breakeven WR (today's signals, RH LS): {be * 100:.1f}%")
    w()
    w("| cell | dropped N | WR | avg | day-boot P(avg<0) | days |")
    w("|---|---|---|---|---|---|")
    for nm, (col, drop) in FILTERS.items():
        t = ok[ok[col].notna()]
        dm = drop(t[col]).astype(bool)
        d = t[dm]
        p, lo, hi = T.day_boot_diff(-d.yh.values, d.day.values)
        w(f"| {nm} | {len(d)} | {f((d.yh > 0).mean() * 100, 1, False)}% | {f(d.yh.mean())} | {f(p, 2, False)} | {d.day.nunique()} |")
    w()
    open(os.path.join(O, "tables.md"), "w").write("\n".join(L))
    print("\n".join(L[:12]))
    print(GD[["cell", "d25", "P25", "d26", "P26", "master_d", "lomo", "g5", "cand"]].to_string())


if __name__ == "__main__":
    main()
