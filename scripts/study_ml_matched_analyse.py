#!/usr/bin/env python3
"""ML MATCHED-LONG SHADOW study (2026-10-26) — step 3: per-cell tables, verdicts, report.

Unit = one shadow EPISODE (a refused matched long, one per pair at a time, after today's post-block gates), priced by the replica at
RP.BASE (today's ML exit), taker entry fee, % at 1×. LIVE = PJUL + JSEP + CWOCT (decisive); RPL = engine replay yr6/yr6a (refute-only,
3 seeds pooled; N quoted per seed, every CI clustered by UTC DAY across seeds).
Allow bar (Pattern-W style, adapted; pre-stated in the brief): N ≥ 30 on ≥ 10 days · WR ≥ 61.5 % · avg ≥ +0.10 %/fill · day-block
bootstrap LB > 0 at the Bonferroni-corrected level over every cell tested in that dataset · avg ≥ the SAME-DAYS unmatched control
(same replica, same fee) · no single day / pair ≥ 50 % of the cell's net · then a 30–50 % in-sample haircut.
Regime cells are market-wide → evidence in DAY units (days column is the unit; N alone never qualifies a regime cell).

    venv/bin/python -W ignore scripts/study_ml_matched_analyse.py
Outputs: reports/ML_MATCHED_SHADOW_2026-10-26_cells.csv, _regime.csv (per-episode regime features), _summary.txt
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_ml_matched_common as M  # noqa: E402

PFX = M.PFX
R_BOOT = 20000
LIVE = ("PJUL", "JSEP", "CWOCT")


def load():
    S = pd.read_csv(PFX + "_shadows.csv.gz", low_memory=False)
    C = pd.read_csv(PFX + "_controls.csv.gz", low_memory=False)
    S = S[S.episode.astype(bool)].copy()
    for D in (S, C):
        D["day"] = pd.to_datetime(D.o_ms, unit="ms").dt.strftime("%Y-%m-%d")
    S["y"] = S.pct
    C["y_rep"] = C.pct_taker
    # label used by the cells: PJUL = the label the live engine recorded (entry_cohort; today's re-label loses W2 / C2 wherever the
    # July stamps lack entry_btc_trend_gap_pct) · CWOCT = rebuilt label only where its C/W family equals the logged family (parity
    # rows), else the logged family alone (specific-label cells skip it) · JSEP / RPL = rebuilt label.
    S["label_used"] = S.label.fillna("").astype(str)
    pj = S.cohort == "PJUL"
    S.loc[pj, "label_used"] = S.loc[pj, "label_live"].astype(str)
    cw = S.cohort == "CWOCT"
    par = cw & (S.family.replace({"NONE": ""}) == S.family_log.fillna(""))
    S.loc[cw & ~par, "label_used"] = S.loc[cw & ~par, "family_log"].map({"C": "C?", "W": "W?", "C+W": "C?+W?"})
    S["label_parity"] = np.where(cw, par, True)
    return S, C


def regime_feats(S, C):
    f = PFX + "_regime.csv"
    T = np.unique(np.r_[S.o_ms.values, C.o_ms.values].astype(np.int64))
    if os.path.exists(f):
        G = pd.read_csv(f)
        if set(T) <= set(G.o_ms):
            return G
    import study_ml_regime_wl_build as RB
    G = RB.build(T)
    G.insert(0, "o_ms", T)
    keep = ["o_ms", "btc_r1h", "btc_r24h", "btc_offhi_30d", "btc_e20slp_1h", "btc_e20slp_5m", "btc_eff_24h", "br_above_ema50_1h",
            "br_adv_24h", "br_med_r24h", "br_corr24h_top20", "alt_minus_btc_7d", "btc_rv24_pctl90d", "hour", "weekend"]
    G = G[[c for c in keep if c in G.columns]]
    G.to_csv(f, index=False)
    return G


def add_regime(D, G, label_col):
    D = D.merge(G, on="o_ms", how="left")
    from services.regime import classify_btc_regime
    if label_col == "shadow":
        D["regime_lbl"] = [r if isinstance(r, str) and r not in ("nan", "") else classify_btc_regime(_f(a), _f(b), _f(c))
                           for r, a, b, c in zip(D.get("regime", pd.Series([None] * len(D))), D.btc_adx, D.btc_rsi, D.btc_slope)]
    else:
        D["regime_lbl"] = D.entry_btc_regime.astype(str)
    D["z_1h"] = np.where(D.btc_e20slp_1h > 0, "1h>0", np.where(D.btc_e20slp_1h <= 0, "1h<=0", None))
    D["z_24h"] = np.where(D.btc_r24h > 0, "24h>0", np.where(D.btc_r24h <= 0, "24h<=0", None))
    D["z_30d"] = pd.cut(D.btc_offhi_30d, [-1e9, -10, -3, 1e9], labels=["off30<=-10", "off30(-10,-3]", "off30>-3"]).astype(str)
    D["z_br"] = np.where(D.br_above_ema50_1h > 50, "breadth>50", np.where(D.br_above_ema50_1h <= 50, "breadth<=50", None))
    d = pd.to_datetime(D.o_ms, unit="ms")
    D["era"] = np.where(d.dt.year == 2025, "2025", np.where(d.dt.month <= 6, "2026-H1", "2026-H2"))
    return D


def _f(x):
    try:
        v = float(x)
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def boot_means(y, day, R=R_BOOT, seed=20261026):
    y = np.asarray(y, float); day = np.asarray(day)
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, y); n = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (R, len(u)))
    return s[idx].sum(1) / n[idx].sum(1)


def stats(g, ctl, nseeds, m_tests):
    y = g.y.values
    n = len(y)
    out = dict(N=n, N_per_seed=round(n / nseeds, 1), days=g.day.nunique(), pairs=g.pair.nunique(),
               WR=round((y > 0).mean() * 100, 1), avg=round(y.mean(), 3), net=round(y.sum(), 2))
    if g.day.nunique() >= 2:
        bm = boot_means(y, g.day.values)
        out["LB95"] = round(float(np.quantile(bm, 0.05)), 3)
        out["LB_bonf"] = round(float(np.quantile(bm, 0.05 / max(m_tests, 1))), 3)
        out["P_avg_le0"] = round(float((bm <= 0).mean()), 4)
    tot = y.sum()
    ds = g.groupby("day").y.sum(); ps = g.groupby("pair").y.sum()
    out["top_day_share"] = round(float(ds.max() / tot), 2) if tot > 0 else np.nan
    out["top_pair_share"] = round(float(ps.max() / tot), 2) if tot > 0 else np.nan
    # same-days control (same seeds for the replay)
    c = ctl[ctl.day.isin(set(g.day))]
    if "seed" in g and g.seed.nunique() > 1:
        c = c[c.seed.isin(set(g.seed))]
    out["ctl_N"] = len(c)
    out["ctl_avg"] = round(float(c.y_rep.mean()), 3) if len(c) else np.nan
    out["ctl_WR"] = round(float((c.y_rep > 0).mean() * 100), 1) if len(c) else np.nan
    out["delta_vs_ctl"] = round(out["avg"] - out["ctl_avg"], 3) if len(c) else np.nan
    return out


def verdict(r, dataset):
    """the allow bar; dataset RPL can only refute (master over backtest; cross-period = refute-only)."""
    fails = []
    if r["N_per_seed"] < 30: fails.append("N<30")
    if r["days"] < 10: fails.append("days<10")
    if r["WR"] < M.BE_WR: fails.append("WR<61.5")
    if r["avg"] < 0.10: fails.append("avg<+0.10")
    if not (r.get("LB_bonf", np.nan) > 0): fails.append("LB_bonf<=0")
    if not (r["delta_vs_ctl"] >= 0): fails.append("<ctl")
    if (r["top_day_share"] >= 0.5) or (r["top_pair_share"] >= 0.5): fails.append("conc>=50%")
    r["fails"] = ";".join(fails)
    r["pass_bar"] = not fails
    return r


CELLS_PAT = None


def pattern_cells(D):
    lab = D.label_used.fillna("").astype(str)
    toks = lab.str.split("+")
    has = lambda L: toks.map(lambda t: L in t)
    hasC = toks.map(lambda t: any(x.startswith("C") for x in t if x))
    hasW = toks.map(lambda t: any(x.startswith("W") for x in t if x))
    cells = {"ALL matched": lab != "", "C-only (no W)": hasC & ~hasW, "W-only (C-free)": hasW & ~hasC, "C+W": hasC & hasW}
    for L in M.C_L + M.W_L:
        cells[f"has {L}"] = has(L)
    for L in M.W_L:
        cells[f"{L} C-free"] = has(L) & ~hasC
    for L in M.W_L + M.C_L:
        cells[f"{L} alone"] = lab == L
    for combo, n in lab.value_counts().items():
        if "+" in combo and n >= 30:
            cells[f"= {combo}"] = lab == combo
    return cells


def regime_cells(D):
    out = {}
    for z in ("regime_lbl", "z_1h", "z_24h", "z_30d", "z_br"):
        for v in sorted(D[z].dropna().astype(str).unique()):
            if v in ("None", "nan", "UNKNOWN"):
                continue
            out[f"{z}={v}"] = D[z].astype(str) == v
    return out


def run(D, ctl, dataset, nseeds):
    pc = pattern_cells(D)
    rc = regime_cells(D)
    fam = {"ALL matched": pc["ALL matched"], "W-only (C-free)": pc["W-only (C-free)"], "C-only (no W)": pc["C-only (no W)"],
           "C+W": pc["C+W"], "W2 C-free": pc["W2 C-free"], "W4 C-free": pc["W4 C-free"], "W6 C-free": pc["W6 C-free"]}
    cells = [(k, v, "pattern") for k, v in pc.items()]
    for fk, fv in fam.items():
        for rk, rv in rc.items():
            cells.append((f"{fk} × {rk}", fv & rv, "pattern×regime"))
    for e in sorted(D.era.unique()):
        for fk in ("ALL matched", "W-only (C-free)", "C-only (no W)", "C+W"):
            cells.append((f"{fk} × era={e}", fam[fk] & (D.era == e), "era"))
    cells = [(k, v, t) for k, v, t in cells if v.sum() >= 5]
    m = len(cells)
    rows = []
    for k, v, t in cells:
        g = D[v.values]
        r = dict(dataset=dataset, cell=k, kind=t, **stats(g, ctl, nseeds, m))
        rows.append(verdict(r, dataset))
    T = pd.DataFrame(rows)
    T["m_tests"] = m
    return T


def control_line(C, label):
    y = C.y_rep.dropna()
    return dict(cell=label, N=len(y), days=C.loc[y.index].day.nunique(), WR=round((y > 0).mean() * 100, 1), avg=round(y.mean(), 3))


def main():
    S, C = load()
    G = regime_feats(S, C)
    S = add_regime(S, G, "shadow")
    C = add_regime(C, G, "control")
    S.to_csv(PFX + "_episodes.csv", index=False)
    live = S[S.cohort.isin(LIVE)].copy()
    rpl = S[S.cohort == "RPL"].copy()
    CL = C[(C.cohort == "CTL_LIVE") & C.y_rep.notna()].copy()
    CR = C[(C.cohort == "CTL_RPL") & C.y_rep.notna()].copy()
    T = pd.concat([run(live, CL, "LIVE", 1), run(rpl, CR, "RPL", 3)]
                  + [run(live[live.cohort == co], CL, co, 1) for co in LIVE], ignore_index=True)
    T.to_csv(PFX + "_cells.csv", index=False)
    print(T.groupby("dataset").pass_bar.sum())
    print(T[T.pass_bar])


if __name__ == "__main__":
    main()
