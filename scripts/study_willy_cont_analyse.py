#!/usr/bin/env python3
"""WILLY CONTINUATION — step 3: the pre-registered tests (reports/WILLY_CONT_PREREG.txt, asserted) on reports/willy_cont/rows.pkl.
T1 continuation shares + economic bar + arms · T2 1D sign-first / halves / terciles · T3 exhaustive 2D MARKET × COIN with a day-block
shuffled null over the full search + walk-forward · T4 shallow models (numpy logistic + depth-3 tree) held-out AUC vs shuffled-label null
· T5 dynamic exit only if a separator survives (else the best rule is priced and labelled FAILED).
Usage: venv/bin/python scripts/study_willy_cont_analyse.py [--nperm 300] [--nnull 200]
Outputs reports/willy_cont/{tables.md, tests_1d.csv, tests_2d.csv, null.csv, models.csv, coverage.csv, ms_rows.csv}
"""
import os, sys, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports"); OUT = os.path.join(REP, "willy_cont")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
assert os.path.exists(os.path.join(REP, "WILLY_CONT_PREREG.txt")), "pre-registration missing"
import study_yr6wl_frenzy_common as C  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--nperm", type=int, default=300); ap.add_argument("--nnull", type=int, default=200)
A = ap.parse_args()
ERAS = ["2025", "2026-H1", "2026-H2"]
MINZ, MIND, TOPWF = 10, 5, 20
EXCL = {"entry_frenzy_vwap", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_slippage_pct", "entry_frenzy_catchup_bars",
        "entry_frenzy_catchup_move_pct", "entry_frenzy_catchup", "entry_btc_regime_started_at", "entry_frenzy_spike_at", "entry_surge_trigger_at",
        "entry_bracket_cap_notional", "entry_bracket_max_leverage", "entry_gap_expand_marginal"}
MD = []


def P(*s):
    line = " ".join(str(x) for x in s); print(line, flush=True); MD.append(line)


def tbl(df, fl=3):
    return C.md_table(df, fl) if len(df) else "(empty)"


R = pd.read_pickle(os.path.join(OUT, "rows.pkl"))
R["tick"] = R.tick90.astype(bool)
for c in list(R.columns):
    if c.startswith("entry_"):
        R[c] = C.num(R, c)
R["e_vratio"] = np.nan                                             # meaningless at entry (= 1 by construction)

# ───────────────────────── cohorts / outcomes ─────────────────────────
R["q1"] = (R["t_1.25"] <= 30) & R.why_today.eq("TP")
R["y1"] = R.y_hold2 - R.y_today
R["y1_3"] = R.y_hold3 - R.y_today
R["C2_30"] = R.why_hold2.eq("TP2").astype(float)
R["C2_90"] = R["t_2"].notna().astype(float); R["C3_90"] = R["t_3"].notna().astype(float)
R["q2"] = ~(R["t_1.25"] <= 30) & ~(R["t_stop"] <= 30) & R.y_today.notna() & (R.x_today > 30)
R["y2"] = R.y_keep - R.y_today; R["y2c"] = R.y_close30 - R.y_today
R["R125"] = R.r125_after30
R["q3"] = R.y_today.notna() & R.y_hold2.notna()
R["y3"] = R.y1.where(R.q1, 0.0)
R["late_mkt30"] = (R.cl30 >= 0.25).astype(float)
BT = R[R.block == "BT"].copy(); BT["seed"] = BT.seed.astype(int)
S1 = BT[BT.seed == 1]
MS = R[R.block != "BT"].copy()

# ───────────────────────── variable registry ─────────────────────────
ent = [c for c in C.entry_cols(S1) if c not in EXCL]
DAYV = [c for c in R.columns if c.startswith("d_")] + ["ac15_3d", "ac5_3d"]
INTRA = ["i_btc_rv60r", "i_mkt_qv60r", "i_btc_day", "hour"]
E_MKT = ["e_btc_r5", "e_btc_r15", "e_eth_r5", "e_eth_r15", "e_sol_r5", "e_sol_r15", "e_mkt_tbs5", "e_br_up5", "e_br_up15", "e_agg_tbr5", "e_agg_tbr15"]
E_COIN = ["e_tbs5", "e_cret5"]
MKT_ENTRY = [c for c in ent if C.is_macro(c)] + E_MKT + INTRA + DAYV
COIN_ENTRY = [c for c in ent if not C.is_macro(c)] + E_COIN


def dec(pre):
    mk = [f"{pre}_{a}_{b}" for a in ("btc", "eth", "sol") for b in ("since", "r5", "r15")] + [f"{pre}_mkt_tbs5", f"{pre}_br_up5", f"{pre}_br_up15",
                                                                                           f"{pre}_agg_tbr5", f"{pre}_agg_tbr15"]
    co = [f"{pre}_tbs5", f"{pre}_tbs_since", f"{pre}_vratio", f"{pre}_cret5"]
    return mk, co


A_MK, A_CO = dec("a"); M_MK, M_CO = dec("m")
QS = {
    "Q1": dict(mask="q1", y="y1", lab="C2_30", mkt=A_MK + MKT_ENTRY, coin=A_CO + ["t_1.25", "mae_pre"] + COIN_ENTRY,
               pred={"a_tbs5": +1, "a_vratio": +1, "t_1.25": -1, "mae_pre": +1, "a_btc_since": +1, "a_btc_r5": +1, "a_btc_r15": +1,
                     "a_eth_since": +1, "a_sol_since": +1, "a_mkt_tbs5": +1, "a_br_up5": +1, "a_br_up15": +1, "a_agg_tbr5": +1,
                     "a_agg_tbr15": +1, "d_r1": +1, "d_r3": +1, "d_r7": +1, "d_dma20": +1, "d_br_up3": +1, "d_br_up7": +1, "d_br_ma7": +1,
                     "ac15_3d": +1, "ac5_3d": +1, "d_disp7": +1, "d_voltr": +1, "d_ft2": +1, "d_ft12": +1, "d_ftr12": +1, "i_btc_day": +1,
                     "i_mkt_qv60r": +1, "entry_frenzy_vol_mult": +1, "entry_vol_mcap_ratio": -1, "entry_mcap_usd": -1,
                     "entry_pair_age_days": -1, "entry_frenzy_run_pct": -1, "entry_frenzy_vs_vwap_pct": +1, "entry_frenzy_willy_wait_bars": -1}),
    "Q2": dict(mask="q2", y="y2", lab="R125", mkt=M_MK + MKT_ENTRY, coin=M_CO + ["cl30", "mfe30", "mae30"] + COIN_ENTRY,
               pred={"cl30": +1, "mae30": +1, "mfe30": +1, "m_tbs5": +1, "m_vratio": +1, "m_btc_since": +1, "m_br_up5": +1, "m_br_up15": +1,
                     "m_mkt_tbs5": +1, "m_agg_tbr15": +1}),
    "Q3": dict(mask="q3", y="y3", lab="C2_90", mkt=MKT_ENTRY, coin=COIN_ENTRY,
               pred={"e_btc_r15": +1, "e_mkt_tbs5": +1, "e_br_up15": +1, "e_tbs5": +1, "d_r3": +1, "d_br_up3": +1, "ac15_3d": +1, "d_ft2": +1,
                     "d_voltr": +1, "entry_frenzy_vol_mult": +1, "entry_vol_mcap_ratio": -1, "entry_mcap_usd": -1, "entry_pair_age_days": -1}),
}


def num(F, c):
    return pd.to_numeric(F[c], errors="coerce").astype(float) if c in F.columns else pd.Series(np.nan, index=F.index)


def cuts(x):
    x = pd.Series(x).dropna()
    if x.nunique() < 2:
        return []
    if x.nunique() == 2:
        v = float(x.max()); return [("on", "==", v), ("off", "!=", v)]
    q1, md, q2 = x.quantile([1 / 3, 0.5, 2 / 3])
    out = []
    if (x > 0).mean() > 0.05 and (x < 0).mean() > 0.05:
        out += [("pos", ">", 0.0), ("neg", "<=", 0.0)]
    out += [("hi", ">", float(md)), ("lo", "<=", float(md)), ("lo3", "<=", float(q1)), ("hi3", ">", float(q2))]
    return out


def m_of(x, op, v):
    x = np.asarray(x, float); ok = ~np.isnan(x)
    r = {">": x > v, "<=": x <= v, "==": np.isclose(x, v), "!=": ~np.isclose(x, v)}[op]
    return r & ok


def eligible(m, era, day, sc):
    for e in ERAS:
        ie = (era == e) & sc
        z = m & ie
        if z.sum() < MINZ or (ie & ~m).sum() < MINZ or len(set(day[z])) < MIND:
            return False
    return True


def era_t(Z, SC, Y, era):
    """Welch t of zone vs scored rest per era. Z, SC n×k (SC = scored mask), Y n×p → {era: k×p}."""
    out = {}
    Y = np.asarray(Y, np.float32)
    for e in ERAS:
        ie = (era == e).astype(np.float32)[:, None]
        Zi = Z * ie; Ri = (SC - Z) * ie
        n1 = Zi.sum(0)[:, None]; n0 = Ri.sum(0)[:, None]
        s1 = Zi.T @ Y; s0 = Ri.T @ Y; q1 = Zi.T @ (Y ** 2); q0 = Ri.T @ (Y ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1 = s1 / n1; m0 = s0 / n0
            v1 = (q1 - n1 * m1 ** 2) / np.maximum(n1 - 1, 1); v0 = (q0 - n0 * m0 ** 2) / np.maximum(n0 - 1, 1)
            out[e] = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    return out


def S_of(T):
    a = np.stack([np.nan_to_num(T[e]) for e in ERAS])
    sg = np.sign(a); same = (sg == sg[0]).all(0) & (sg[0] != 0)
    return np.where(same, sg[0] * np.abs(a).min(0), 0.0)


def build_tests(F, vars1, pairs, era, day):
    tests, seen = [], set()

    def add(lbl, m, sc):
        k = np.packbits(m).tobytes()
        if k in seen or not eligible(m, era, day, sc):
            return
        seen.add(k); tests.append((lbl, m, sc))
    X = {c: num(F, c).values for c in set(vars1) | {a for p in pairs for a in p}}
    CU = {c: cuts(X[c]) for c in X}
    for c in vars1:
        sc = ~np.isnan(X[c])
        for g, op, v in CU[c]:
            add((c, g, op, v, None, None, None, None), m_of(X[c], op, v), sc)
    for a, b in pairs:
        sc = ~np.isnan(X[a]) & ~np.isnan(X[b])
        ca = dict((g, (op, v)) for g, op, v in CU[a]); cb = dict((g, (op, v)) for g, op, v in CU[b])
        for kind in (("hi", "lo", "on", "off"), ("lo3", "hi3", "on", "off")):
            for ga in kind:
                if ga not in ca: continue
                for gb in kind:
                    if gb not in cb: continue
                    add((a, ga, *ca[ga], b, gb, *cb[gb]), m_of(X[a], *ca[ga]) & m_of(X[b], *cb[gb]), sc)
    return tests


def lab(l):
    a, g, oa, va, b, gb, ob, vb = l
    s = f"{a} {oa} {va:.4g}"
    return s + (f" & {b} {ob} {vb:.4g}" if b else "")


def family(name, tests, yv, era, day, seed, nperm):
    Z = np.column_stack([m for _, m, _ in tests]).astype(np.float32); SC = np.column_stack([s for _, _, s in tests]).astype(np.float32)
    T0 = era_t(Z, SC, yv[:, None], era); S0 = S_of(T0)[:, 0]
    Yp = np.empty((len(yv), nperm))
    for e in ERAS:
        ie = np.flatnonzero(era == e)
        Yp[ie] = C.block_draws(yv[ie], day[ie], nperm, seed + 97 * ERAS.index(e))
    mx, wf = [], []
    for s in range(0, nperm, 25):
        Tp = era_t(Z, SC, Yp[:, s:s + 25], era); Sp = S_of(Tp)
        mx.append(np.abs(Sp).max(0)); wf.append(wf_count(Tp))
    mx = np.concatenate(mx); wf = np.concatenate(wf)
    padj = np.array([(mx >= abs(s)).mean() if s != 0 else 1.0 for s in S0])
    WF = wf_list(T0)
    rows = [dict(family=name, zone=lab(l), var=l[0], var2=l[4], S=s, p_fw=p, t_2025=T0["2025"][i, 0], t_H1=T0["2026-H1"][i, 0],
                 t_H2=T0["2026-H2"][i, 0], n_zone=int(m.sum())) for i, ((l, m, _), s, p) in enumerate(zip(tests, S0, padj))]
    summ = dict(family=name, tests=len(tests), max_abs_S=float(np.abs(S0).max()), null_q50=float(np.median(mx)), null_q95=float(np.percentile(mx, 95)),
                survivors_p10=int((padj < 0.10).sum()), wf_confirmed=int(WF.confirmed.sum()), wf_null_mean=float(wf.mean()),
                wf_p=float((wf >= WF.confirmed.sum()).mean()))
    WF["zone"] = [lab(tests[i][0]) for i in WF.idx]; WF["family"] = name
    return pd.DataFrame(rows), summ, WF


def wf_count(T):
    tot = 0
    for d in ERAS:
        td = np.nan_to_num(T[d]); k = td.shape[0]
        top = np.argpartition(-np.abs(td), min(TOPWF, k - 1), axis=0)[:TOPWF]
        sg = np.sign(np.take_along_axis(td, top, 0)); ok = np.ones_like(sg, bool)
        for o in ERAS:
            if o != d:
                ok &= sg * np.nan_to_num(np.take_along_axis(T[o], top, 0)) >= 1.28
        tot = tot + ok.sum(0)
    return tot


def wf_list(T):
    rows = []
    for d in ERAS:
        td = np.nan_to_num(T[d][:, 0]); top = np.argsort(-np.abs(td))[:TOPWF]
        for i in top:
            sg = np.sign(td[i])
            rows.append(dict(disc=d, idx=i, t_disc=td[i], **{f"t_{o}": T[o][i, 0] for o in ERAS},
                             confirmed=all(sg * np.nan_to_num(T[o][i, 0]) >= 1.28 for o in ERAS if o != d)))
    return pd.DataFrame(rows)


# ───────────────────────── models ─────────────────────────
def auc(s, y):
    y = np.asarray(y, bool); s = np.asarray(s, float)
    if y.sum() == 0 or (~y).sum() == 0:
        return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())


def prep(Xtr, Xte):
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xte = np.where(np.isnan(Xte), med, Xte)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def logit_fit(X, y, lam=1.0, it=50):
    X1 = np.column_stack([np.ones(len(X)), X]); w = np.zeros(X1.shape[1])
    L = lam * np.eye(len(w)); L[0, 0] = 0
    for _ in range(it):
        p = 1 / (1 + np.exp(-np.clip(X1 @ w, -30, 30)))
        g = X1.T @ (p - y) + L @ w
        H = X1.T @ (X1 * (p * (1 - p))[:, None]) + L
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def logit_pred(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


def tree_fit(X, y, depth=3, minleaf=30):
    def node(idx, d):
        p = y[idx].mean()
        if d == 0 or len(idx) < 2 * minleaf:
            return p
        best = None; g0 = len(idx) * p * (1 - p)
        for j in range(X.shape[1]):
            xj = X[idx, j]
            for q in np.unique(np.quantile(xj, np.linspace(0.1, 0.9, 9))):
                L = xj <= q; nl = L.sum(); nr = len(idx) - nl
                if nl < minleaf or nr < minleaf:
                    continue
                pl = y[idx][L].mean(); pr = y[idx][~L].mean()
                g = nl * pl * (1 - pl) + nr * pr * (1 - pr)
                if g < g0 - 1e-9 and (best is None or g < best[0]):
                    best = (g, j, q, L)
        if best is None:
            return p
        _, j, q, L = best
        return (j, q, node(idx[L], d - 1), node(idx[~L], d - 1))
    return node(np.arange(len(y)), depth)


def tree_pred(t, X):
    out = np.empty(len(X))
    for i in range(len(X)):
        n = t
        while isinstance(n, tuple):
            n = n[2] if X[i, n[0]] <= n[1] else n[3]
        out[i] = n
    return out


def model_runs(F, feats, lab, tag, nnull):
    rows = []
    splits = [("2025→2026", F.era.eq("2025"), F.era.ne("2025")), ("2026→2025", F.era.ne("2025"), F.era.eq("2025")),
              ("H1→H2", F.era.eq("2026-H1"), F.era.eq("2026-H2"))]
    y = F[lab].values.astype(float)
    for fs_name, cols in feats.items():
        cols = [c for c in cols if c in F.columns and num(F, c).notna().mean() >= 0.5 and num(F, c).nunique() > 1]
        X = np.column_stack([num(F, c).values for c in cols])
        for sp, tr, te in splits:
            tr, te = tr.values & ~np.isnan(y), te.values & ~np.isnan(y)
            if tr.sum() < 60 or te.sum() < 30:
                continue
            Xtr, Xte = prep(X[tr], X[te]); ytr = y[tr]; yte = y[te]
            for mname in ("logit", "tree"):
                def fitpred(yy):
                    if mname == "logit":
                        return logit_pred(logit_fit(Xtr, yy), Xte)
                    return tree_pred(tree_fit(Xtr, yy), Xte)
                a0 = auc(fitpred(ytr), yte)
                n = nnull if mname == "logit" else max(nnull // 4, 30)
                Yn = C.block_draws(ytr, F.day.values[tr], n, 5)
                an = np.array([auc(fitpred(Yn[:, k]), yte) for k in range(n)])
                rows.append(dict(q=tag, features=fs_name, n_feat=len(cols), split=sp, model=mname, n_train=int(tr.sum()), n_test=int(te.sum()),
                                 base_rate_test=yte.mean(), AUC=a0, null_q50=np.nanmedian(an), null_q95=np.nanpercentile(an, 95),
                                 p=float((an >= a0).mean())))
    return pd.DataFrame(rows)


# ───────────────────────── helpers for reads ─────────────────────────
def boot(y, day):
    lo, hi, pp = C.day_boot(np.asarray(y, float), np.asarray(day))
    return lo, hi, pp


def era_line(F, col, extra=None):
    out = []
    for e in ERAS + ["ALL"]:
        G = F if e == "ALL" else F[F.era == e]
        v = G[col].astype(float).dropna()
        if not len(v):
            continue
        lo, hi, pp = boot(v, G.loc[v.index, "day"])
        d = dict(era=e, N=len(v), days=G.loc[v.index, "day"].nunique(), mean=v.mean(), ci=f"{lo:+.3f}/{hi:+.3f}", P_gt0=pp)
        if extra:
            d.update(extra(G))
        out.append(d)
    return pd.DataFrame(out)


# ═════════════════════════ T1 ═════════════════════════
VT = S1[S1.tick]
P("# WILLY CONTINUATION — tables (scripts/study_willy_cont_analyse.py)\n")
P(open(os.path.join(OUT, "parity.txt")).read())
P(f"Engine replay WILLY fills: seed 1 {len(S1)} (tick-verdict {len(VT)}), seeds 2/3 {len(BT[BT.seed == 2])}/{len(BT[BT.seed == 3])}; "
  f"MASTER {len(MS[MS.block == 'MASTER'])} · LIVE {len(MS[MS.block == 'LIVE'])}")
P(f"Tick-verdict per era (seed 1): {VT.groupby('era').size().to_dict()} · unresolved today {S1.y_today.isna().sum()}")
P(f"TODAY booking: why {VT.why_today.value_counts().to_dict()} · evaluate_late (line booking) vs corrected market booking, avg "
  f"{VT.evl_pct.mean():+.3f} raw vs today {VT.y_today.mean():+.3f}")

P("\n## T1 — the arms, seed 1 tick verdict (avg %/fill at 1x; day-bootstrap 95 % CI; P(mean>0))\n")
rows = []
for arm in ("today", "keep", "hold2", "hold3", "close30"):
    for e in ERAS + ["ALL"]:
        G = VT if e == "ALL" else VT[VT.era == e]
        v = G[f"y_{arm}"].dropna(); lo, hi, pp = boot(v, G.loc[v.index, "day"])
        dv = (G[f"y_{arm}"] - G.y_today).dropna(); dl, dh, dp = boot(dv, G.loc[dv.index, "day"])
        rows.append(dict(arm=arm.upper(), era=e, N=len(v), WR=100 * (v > 0).mean(), avg=v.mean(), ci=f"{lo:+.3f}/{hi:+.3f}",
                         d_vs_today=dv.mean(), P_d_gt0=dp))
P(tbl(pd.DataFrame(rows)))
P("\nSame arms — seeds 2 / 3 (tick) and the 1m ruler (seed 1, non-tick rows; refute-only), Δ vs TODAY:\n")
rows = []
for nm, G in (("seed2 tick", BT[(BT.seed == 2) & BT.tick]), ("seed3 tick", BT[(BT.seed == 3) & BT.tick]), ("seed1 1m-ruler", S1[~S1.tick])):
    for e in ERAS:
        g = G[G.era == e]
        rows.append(dict(set=nm, era=e, N=len(g), today=g.y_today.mean(), d_keep=(g.y_keep - g.y_today).mean(), d_hold2=(g.y_hold2 - g.y_today).mean(),
                         d_hold3=(g.y_hold3 - g.y_today).mean(), d_close30=(g.y_close30 - g.y_today).mean()))
P(tbl(pd.DataFrame(rows)))

P("\n## T1 — Q1: at the +1.25 touch (≤ 30 min), how often does it continue? (seed 1 tick)\n")
Q1 = VT[VT.q1]
rows = []
for e in ERAS + ["ALL"]:
    G = Q1 if e == "ALL" else Q1[Q1.era == e]
    non = G[G.C2_30 == 0]
    L = non.y_hold2.mean()
    pstar = (1.25 - L) / (2.0 - L)
    lo, hi, pp = boot(G.y1, G.day)
    rows.append(dict(era=e, N_touch=len(G), share_touch=len(G) / max(1, len(VT if e == "ALL" else VT[VT.era == e])),
                     med_min_to_touch=G["t_1.25"].median(), C2_30=G.C2_30.mean(), C2_90=G.C2_90.mean(), C3_90=G.C3_90.mean(),
                     L_nonreach=L, p_star=pstar, hold2_minus_bank=G.y1.mean(), ci=f"{lo:+.3f}/{hi:+.3f}", P_gt0=pp,
                     hold2_why=str(G.why_hold2.value_counts().to_dict())))
P(tbl(pd.DataFrame(rows)))

P("\n## T1 — Q2: open at minute 30 (seed 1 tick)\n")
Q2 = VT[VT.q2]
rows = []
for e in ERAS + ["ALL"]:
    G = Q2 if e == "ALL" else Q2[Q2.era == e]
    if not len(G):
        continue
    rows.append(dict(era=e, N_open30=len(G), share=len(G) / max(1, len(VT if e == "ALL" else VT[VT.era == e])), above_025_at30=G.late_mkt30.mean(),
                     R125=G.R125.mean(), stop_after30=G.stop_after30.mean(), today=G.y_today.mean(), keep=G.y_keep.mean(),
                     close30=G.y_close30.mean(), keep_minus_today=G.y2.mean(), P_keep_gt=boot(G.y2, G.day)[2],
                     close30_minus_today=G.y2c.mean(), P_close_gt=boot(G.y2c, G.day)[2]))
P(tbl(pd.DataFrame(rows)))
P("\nQ2 split by the state at minute 30 (TODAY: ≥ +0.25 → market now; < +0.25 → resting +0.25):\n")
rows = []
for nm, g in (("at30 ≥ +0.25 (market now)", Q2[Q2.late_mkt30 == 1]), ("at30 < +0.25 (resting +0.25)", Q2[Q2.late_mkt30 == 0])):
    for e in ERAS + ["ALL"]:
        G = g if e == "ALL" else g[g.era == e]
        if len(G):
            rows.append(dict(state=nm, era=e, N=len(G), R125=G.R125.mean(), today=G.y_today.mean(), keep=G.y_keep.mean(), close30=G.y_close30.mean()))
P(tbl(pd.DataFrame(rows)))

# ═════════════════════════ T2 / T3 per question ═════════════════════════
COV, ALL1D, ALL2D, SUMS, WFS, MODELS = [], [], [], [], [], []
for q, spec in QS.items():
    F = VT[VT[spec["mask"]].astype(bool)].reset_index(drop=True)
    era = F.era.values; day = F.day.values
    mk = [c for c in dict.fromkeys(spec["mkt"])]; co = [c for c in dict.fromkeys(spec["coin"])]
    okv = {}
    for c in mk + co:
        x = num(F, c); cov = x.notna().mean(); nun = x.nunique()
        ok = cov >= 0.8 and nun >= 2
        COV.append(dict(q=q, var=c, block="MARKET" if c in mk else "COIN", scored=round(100 * cov, 1), nunique=nun, tested=ok,
                        why_not="" if ok else ("constant" if nun < 2 else "coverage < 80 %")))
        okv[c] = ok
    mk = [c for c in mk if okv[c]]; co = [c for c in co if okv[c]]
    P(f"\n## {q} — cohort {len(F)} tick fills on {F.day.nunique()} days; {len(mk)} MARKET + {len(co)} COIN variables tested "
      f"(y = {spec['y']}, label = {spec['lab']})\n")
    # T2 1D on y and on the label
    for out_name in (spec["y"], spec["lab"]):
        yv = F[out_name].values.astype(float)
        t1 = build_tests(F, mk + co, [], era, day)
        r1, s1, w1 = family(f"{q} 1D on {out_name}", t1, yv, era, day, 11, A.nperm)
        r1["block"] = np.where(r1["var"].isin(mk), "MARKET", "COIN"); r1["q"] = q; r1["outcome"] = out_name
        ALL1D.append(r1); SUMS.append(s1); WFS.append(w1)
        P(f"T2 1D {out_name}: {s1}")
    # predicted-sign table (sign-first, then median halves) on y
    yv = F[spec["y"]].values.astype(float); lv = F[spec["lab"]].values.astype(float)
    rows = []
    for c, sgn in spec["pred"].items():
        if c not in F.columns or not okv.get(c, False):
            rows.append(dict(var=c, pred="+" if sgn > 0 else "−", note="not scored")); continue
        x = num(F, c).values
        md = np.nanmedian(x)
        cut = 0.0 if ((x > 0).mean() > 0.05 and (x < 0).mean() > 0.05) else md
        hi = x > cut; sc = ~np.isnan(x)
        rec = dict(var=c, pred="+" if sgn > 0 else "−", cut=cut)
        for e in ERAS:
            ie = (era == e) & sc
            a, b = yv[ie & hi], yv[ie & ~hi]
            rec[f"Δy_{e}"] = a.mean() - b.mean() if len(a) > 2 and len(b) > 2 else np.nan
            rec[f"Δlab_{e}"] = lv[ie & hi].mean() - lv[ie & ~hi].mean() if len(a) > 2 and len(b) > 2 else np.nan
        sg = [np.sign(rec[f"Δy_{e}"]) for e in ERAS]
        rec["as_predicted_all_eras"] = all(s == sgn for s in sg)
        rec["n_eras_as_pred"] = int(sum(s == sgn for s in sg))
        rows.append(rec)
    P(f"\n{q} predicted-sign check (high side = above the cut; Δ = high − low; y = {spec['y']}, lab = {spec['lab']}):\n")
    P(tbl(pd.DataFrame(rows)))
    # T3 exhaustive 2D MARKET × COIN on y (and the label)
    pairs = [(a, b) for a in mk for b in co]
    for out_name in (spec["y"], spec["lab"]):
        yv = F[out_name].values.astype(float)
        t2 = build_tests(F, mk + co, pairs, era, day)
        r2, s2, w2 = family(f"{q} 1D+2D MKTxCOIN on {out_name}", t2, yv, era, day, 22, A.nperm)
        r2["q"] = q; r2["outcome"] = out_name
        ALL2D.append(r2); SUMS.append(s2); WFS.append(w2)
        P(f"T3 2D {out_name}: {s2}")
        top = r2.reindex(r2.S.abs().sort_values(ascending=False).index).head(8)
        P(tbl(top[["zone", "S", "p_fw", "t_2025", "t_H1", "t_H2", "n_zone"]]))
    # day units for day-level regime variables
    DD = F.groupby("day").agg(y=(spec["y"], "mean"), lab=(spec["lab"], "mean"), era=("era", "first"),
                              **{c: (c, "first") for c in DAYV if okv.get(c, False)}).reset_index()
    rows = []
    for c in [c for c in DAYV if okv.get(c, False)]:
        x = DD[c].values.astype(float); md = np.nanmedian(x)
        rec = dict(var=c)
        for e in ERAS:
            g = DD[DD.era == e]; xx = g[c].values
            rec[f"Δy_{e}"] = g.y[xx > md].mean() - g.y[xx <= md].mean()
        rec["same_sign"] = len({np.sign(rec[f'Δy_{e}']) for e in ERAS}) == 1
        rows.append(rec)
    P(f"\n{q} day-level regime variables in DAY units (day-mean y, median split, {len(DD)} days):\n")
    P(tbl(pd.DataFrame(rows)))
    # T4 models
    feats = {"MARKET": mk, "COIN": co, "ALL": mk + co}
    Mo = model_runs(F, feats, spec["lab"], q, A.nnull)
    MODELS.append(Mo)
    P(f"\n{q} T4 shallow models, held-out AUC on {spec['lab']} vs day-block shuffled-label null:\n")
    P(tbl(Mo))

pd.DataFrame(COV).to_csv(os.path.join(OUT, "coverage.csv"), index=False)
T1D = pd.concat(ALL1D, ignore_index=True); T1D.to_csv(os.path.join(OUT, "tests_1d.csv"), index=False)
T2D = pd.concat(ALL2D, ignore_index=True); T2D.to_csv(os.path.join(OUT, "tests_2d.csv"), index=False)
SU = pd.DataFrame(SUMS); SU.to_csv(os.path.join(OUT, "families.csv"), index=False)
pd.concat(WFS).to_csv(os.path.join(OUT, "walkforward.csv"), index=False)
MOD = pd.concat(MODELS, ignore_index=True); MOD.to_csv(os.path.join(OUT, "models.csv"), index=False)
P("\n## Family summary (every family: max |S| vs the null, survivors p_fw < 0.10, walk-forward confirmations vs null)\n")
P(tbl(SU))

# ═════════════════════════ survivors / T5 ═════════════════════════
surv = T2D[(T2D.p_fw < 0.10) & T2D.outcome.isin(["y1", "y2", "y3"])]
mpred = MOD.groupby(["q", "features", "model"]).apply(lambda g: bool((g[g.split.isin(["2025→2026", "2026→2025"])].p < 0.05).sum() == 2)).reset_index(name="oos_both")
P("\nModels predicting out of sample in BOTH 2025→2026 and 2026→2025 (p < 0.05):\n")
P(tbl(mpred[mpred.oos_both]) if mpred.oos_both.any() else "none")
P(f"\nT3 survivors (p_fw < 0.10) on the economic outcome: {len(surv)}")


def zone_mask(F, zr):
    x = num(F, zr["var"]).values
    op, v = zr["op"], zr["cut"]
    m = m_of(x, op, v)
    if isinstance(zr.get("var2"), str):
        m &= m_of(num(F, zr["var2"]).values, zr["op2"], zr["cut2"])
    return m


# the best rule (survivor if any, else the top |S| Q1 2D test on y1) → priced on every block, labelled
best_q1 = T2D[(T2D.q == "Q1") & (T2D.outcome == "y1")].copy()
best_q1 = best_q1.reindex(best_q1.S.abs().sort_values(ascending=False).index)
cand = (surv[surv.q == "Q1"] if len(surv[surv.q == "Q1"]) else best_q1).iloc[0]
status = "SURVIVOR" if len(surv[surv.q == "Q1"]) else "FAILED (no survivor — top |S| test, priced for the record only)"
P(f"\n## T5 — Q1 dynamic exit read: \"at +1.25 hold for +2 if [{cand.zone}] else bank\" — {status}\n")
# parse the zone back (label format 'var op cut [& var2 op2 cut2]')
parts = [p.strip().split(" ") for p in cand.zone.split("&")]
zr = dict(var=parts[0][0], op=parts[0][1], cut=float(parts[0][2]))
if len(parts) > 1:
    zr.update(var2=parts[1][0], op2=parts[1][1], cut2=float(parts[1][2]))
hold_if = cand.S > 0                                  # S > 0: the zone gains from holding → hold inside the zone


def rule_y(F):
    m = zone_mask(F, zr)
    if not hold_if:
        m = ~m & (num(F, zr["var"]).notna().values)
    return np.where(F.q1.values & m, F.y_hold2.values, F.y_today.values)


rows = []
for nm, G in (("seed1 tick", VT), ("seed2 tick", BT[(BT.seed == 2) & BT.tick]), ("seed3 tick", BT[(BT.seed == 3) & BT.tick]), ("seed1 1m-ruler", S1[~S1.tick])):
    for e in ERAS + ["ALL"]:
        g = G if e == "ALL" else G[G.era == e]
        g = g[g.y_today.notna() & g.y_hold2.notna()]
        d = rule_y(g) - g.y_today.values
        lo, hi, pp = boot(d, g.day)
        rows.append(dict(set=nm, era=e, N=len(g), today=g.y_today.mean(), rule=np.mean(rule_y(g)), delta=d.mean(), P_gt0=pp,
                         held=int((g.q1.values & (rule_y(g) != g.y_today.values)).sum())))
P(tbl(pd.DataFrame(rows)))

# master / live per trade
MSV = MS[MS.y_today.notna()].copy()
MSV["y_rule"] = rule_y(MSV) if len(MSV) else []
MSV["d_rule"] = MSV.y_rule - MSV.y_today
MSV["d_hold2"] = MSV.y_hold2 - MSV.y_today
cols = ["block", "batch", "pair", "opened_at", "pnl_percentage", "why_today", "y_today", "why_hold2", "y_hold2", "y_rule", "d_rule", "d_hold2", "usd_pt",
        "t_1.25", "t_2", "cl30", zr["var"]] + ([zr["var2"]] if "var2" in zr else [])
MSV[[c for c in cols if c in MSV.columns]].to_csv(os.path.join(OUT, "ms_rows.csv"), index=False)
P("\nMASTER / LIVE per trade (today's exit re-priced; % at 1x; $ = Δ% × the trade's $/pt):\n")
P(tbl(MSV[[c for c in cols if c in MSV.columns]]))
rows = []
for (b, bt), g in MSV.groupby(["block", "batch"], dropna=False):
    rows.append(dict(block=b, batch=bt, N=len(g), today_pts=g.y_today.sum(), rule_pts=g.y_rule.sum(), d_rule_pts=g.d_rule.sum(),
                     d_rule_usd=(g.d_rule * g.usd_pt).sum(), hold2_pts=g.y_hold2.sum(), d_hold2_usd=(g.d_hold2 * g.usd_pt).sum(),
                     today_usd=(g.y_today * g.usd_pt).sum()))
P(tbl(pd.DataFrame(rows)))
open(os.path.join(OUT, "tables.md"), "w").write("\n".join(MD) + "\n")
print("wrote", os.path.join(OUT, "tables.md"))
