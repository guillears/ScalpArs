#!/usr/bin/env python3
"""WILLY exhaustive 2D winners-vs-losers screen under TODAY's exit, tick-priced verdict cohort (2026-10-16). Research only, cache-only.

Pre-registration: reports/WILLY_2D_PREREG.txt (frozen before any outcome table; this script asserts it exists).
  venv/bin/python -I scripts/study_willy_2d_screen.py [--nperm 300]
Outputs: reports/WILLY_2D_tests.csv (every eligible test, observed) · WILLY_2D_null.csv (per-shuffle max|S|, counts, WF confirmations)
         · WILLY_2D_top.csv (top tests + full verdict reads) · WILLY_2D_coverage.csv · WILLY_2D_blocks.csv · WILLY_2D_tables.md
"""
import argparse, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_yr6wl_frenzy_common as C  # noqa: E402

assert os.path.exists(os.path.join(REP, "WILLY_2D_PREREG.txt")), "pre-registration missing"
ap = argparse.ArgumentParser(); ap.add_argument("--nperm", type=int, default=300); A = ap.parse_args()
ERAS = ["2025", "2026-H1", "2026-H2"]
EXCL = {"entry_frenzy_vwap", "entry_desired_notional", "entry_liquidity_cap_notional", "entry_slippage_pct", "entry_frenzy_catchup_bars",
        "entry_frenzy_catchup_move_pct", "entry_frenzy_catchup", "entry_btc_regime_started_at", "entry_frenzy_spike_at", "entry_surge_trigger_at"}
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-16_13-10-21 (1).csv")
MINZ, MIND, MINDAY = 10, 5, 8
TOPWF = 20

# ───────────────────────── cohort ─────────────────────────
R = pd.read_csv(os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_rows.csv"), low_memory=False)
F = pd.read_csv(C.FILLS, low_memory=False)
F = F[(F.sleeve == "FRENZY_WILLY") & (F.status == "CLOSED") & (~F.is_probe.astype(bool))].copy()
F["ts"] = pd.to_datetime(F.opened_at, format="mixed")
F["ep"] = F.pair + "|" + F.entry_frenzy_spike_at.astype(str)
F = F.sort_values(["seed", "ts"])
F["k_ep"] = F.groupby(["seed", "ep"]).cumcount() + 1.0
F["fz_age_min"] = (F.ts - pd.to_datetime(F.entry_frenzy_spike_at, errors="coerce", format="mixed")).dt.total_seconds() / 60
F["opened_at"] = F.opened_at.astype(str)
BT = R[R.block == "BT"].copy(); BT["seed"] = BT.seed.astype(int); BT["opened_at"] = BT.opened_at.astype(str)
J = BT.merge(F.drop(columns=[c for c in F.columns if c in BT.columns and c not in ("seed", "pair", "opened_at")]),
             on=["seed", "pair", "opened_at"], how="left", indicator=True)
assert (J._merge == "both").all(), f"join incomplete: {(J._merge != 'both').sum()}"
J = J.drop(columns="_merge")
J["ts"] = pd.to_datetime(J.opened_at, format="mixed")
J["hour"] = J.ts.dt.hour.astype(float)
J["top50f"] = J.top50.astype(str).str.lower().map({"true": 1.0, "false": 0.0})
J["lvol24"] = np.log10(pd.to_numeric(J.entry_pair_volume_24h_usd, errors="coerce").clip(lower=1))
PANEL = pd.read_csv(os.path.join(REP, "YR6_WL_FRENZY_regime_daily_panel.csv")).set_index("date")
PCOLS = [c for c in PANEL.columns if c != "alt_n"]
PANEL = PANEL[PCOLS].add_prefix("pnl_")
PCOLS = list(PANEL.columns)
J = J.join(PANEL, on="day")
ALL = J  # all seeds, all pricing
CO = J[(J.seed == 1) & (J.tick_priced.astype(str).str.lower() == "true")].reset_index(drop=True)
print("cohort", len(CO), CO.groupby("era").size().to_dict())
y = CO.y.values.astype(float); era = CO.era.values; day = CO.day.values

cand = [c for c in C.entry_cols(CO) if c not in EXCL] + ["hour", "top50f", "lvol24", "fz_age_min", "k_ep"]
VARS, cov_rows = [], []
for c in cand + PCOLS:
    x = C.num(CO, c) if c.startswith("entry_") else pd.to_numeric(CO[c], errors="coerce")
    CO[c] = x
    cov = x.notna().mean(); nun = x.nunique()
    kind = "MACRO" if (C.is_macro(c) or c.startswith("pnl_")) else "PAIR"
    ok = cov >= 0.8 and nun >= 2
    cov_rows.append(dict(var=c, kind=kind, scored_cohort=round(100 * cov, 1), nunique=nun, tested=ok,
                         why_not="" if ok else ("constant" if nun < 2 else "coverage < 80 %")))
    if ok:
        VARS.append((c, kind))
PV = [c for c, k in VARS if k == "PAIR"]; MV = [c for c, k in VARS if k == "MACRO"]
print("PAIR vars", len(PV), "MACRO vars", len(MV))


def cuts(x):
    x = pd.Series(x).dropna()
    if x.nunique() == 2:
        v = float(x.max()); return [("on", "==", v), ("off", "!=", v)]
    q1, md, q2 = x.quantile([1 / 3, 0.5, 2 / 3])
    out = [("hi", ">", float(md)), ("lo", "<=", float(md)), ("lo3", "<=", float(q1)), ("hi3", ">", float(q2))]
    if (x > 0).mean() > 0.05 and (x < 0).mean() > 0.05 and abs(md) > 1e-12:
        out += [("pos", ">", 0.0), ("neg", "<=", 0.0)]
    return out


def m_of(x, op, v):
    x = np.asarray(x, float); ok = ~np.isnan(x)
    if op == ">": r = x > v
    elif op == "<=": r = x <= v
    elif op == "==": r = np.isclose(x, v)
    else: r = ~np.isclose(x, v)
    return r & ok


def two_d_cuts(x):
    """median halves + outer terciles for one variable (binary flags: on/off only)."""
    cs = dict((g, (op, v)) for g, op, v in cuts(x))
    if "on" in cs:
        return {"half": [cs["on"], cs["off"]], "ter": []}
    return {"half": [cs["hi"], cs["lo"]], "ter": [cs["lo3"], cs["hi3"]]}


def eligible(m, era_arr, day_arr, minz=MINZ, mind=MIND):
    for e in ERAS:
        ie = era_arr == e
        z = m & ie
        if z.sum() < minz or (ie & ~m).sum() < minz or len(set(day_arr[z])) < mind:
            return False
    return True


def build(frame, cols_a, cols_b=None, same=True, era_arr=None, day_arr=None, minz=MINZ, mind=MIND):
    """1D (cols_b None) or 2D tests. Returns list of (label, mask)."""
    tests, seen = [], set()
    C2 = {c: two_d_cuts(frame[c]) for c in set(cols_a) | set(cols_b or [])}

    def add(lbl, m):
        k = np.packbits(m).tobytes()
        if k in seen or not eligible(m, era_arr, day_arr, minz, mind):
            return
        seen.add(k); tests.append((lbl, m))
    if cols_b is None:
        for c in cols_a:
            for g, op, v in cuts(frame[c]):
                add((c, g, op, v, None, None, None, None), m_of(frame[c], op, v))
        return tests
    for i, a in enumerate(cols_a):
        bs = cols_a[i + 1:] if same else cols_b
        xa = frame[a].values.astype(float)
        for b in bs:
            xb = frame[b].values.astype(float)
            for kind in ("half", "ter"):
                for (oa, va) in C2[a][kind] or C2[a]["half"]:
                    for (ob, vb) in C2[b][kind] or C2[b]["half"]:
                        add((a, kind, oa, va, b, kind, ob, vb), m_of(xa, oa, va) & m_of(xb, ob, vb))
    return tests


def era_t(Z, Y, era_arr):
    """Z n×k, Y n×p → dict era → k×p Welch t."""
    out = {}
    for e in ERAS:
        ie = era_arr == e
        out[e] = C.tstats(Z[ie], Y[ie])
    return out


def S_of(T):
    a = np.stack([T[e] for e in ERAS])           # 3×k×p
    sg = np.sign(a); same = (sg == sg[0]).all(0) & (sg[0] != 0)
    return np.where(same, sg[0] * np.nanmin(np.abs(a), 0), 0.0)


def wf_count(T):
    """walk-forward: per discovery era, top TOPWF by |t_disc|, confirmed if same sign t >= 1.28 in both other eras. → (p,) count."""
    tot = 0
    for d in ERAS:
        td = np.nan_to_num(T[d]); k = td.shape[0]
        top = np.argpartition(-np.abs(td), min(TOPWF, k - 1), axis=0)[:TOPWF]   # TOPWF×p
        sg = np.sign(np.take_along_axis(td, top, 0))
        ok = np.ones_like(sg, bool)
        for o in ERAS:
            if o == d: continue
            to = np.nan_to_num(np.take_along_axis(T[o], top, 0))
            ok &= sg * to >= 1.28
        tot = tot + ok.sum(0)
    return tot


def wf_list(T, labels):
    rows = []
    for d in ERAS:
        td = np.nan_to_num(T[d][:, 0]); top = np.argsort(-np.abs(td))[:TOPWF]
        for i in top:
            sg = np.sign(td[i]); conf = all(sg * np.nan_to_num(T[o][i, 0]) >= 1.28 for o in ERAS if o != d)
            rows.append(dict(disc=d, idx=i, t_disc=td[i], **{f"t_{o}": T[o][i, 0] for o in ERAS}, confirmed=conf))
    return pd.DataFrame(rows)


def run_family(name, tests, yv, era_arr, blocks, null_kind, seed):
    Z = np.column_stack([m for _, m in tests]).astype(float)
    T0 = era_t(Z, yv[:, None], era_arr); S0 = S_of(T0)[:, 0]
    rng = np.random.default_rng(seed)
    if null_kind == "dayblock":
        Yp = np.empty((len(yv), A.nperm))
        for e in ERAS:
            ie = np.flatnonzero(era_arr == e)
            Yp[ie] = C.block_draws(yv[ie], blocks[ie], A.nperm, seed + 97 * ERAS.index(e))
    else:
        Yp = np.empty((len(yv), A.nperm))
        for e in ERAS:
            ie = np.flatnonzero(era_arr == e)
            for p in range(A.nperm):
                Yp[ie, p] = yv[ie][rng.permutation(len(ie))]
    mx, c10, c15, wf = [], [], [], []
    emx = {e: [] for e in ERAS}; cor = {(a, b): [] for a, b in (("2025", "2026-H1"), ("2025", "2026-H2"), ("2026-H1", "2026-H2"))}
    for s in range(0, A.nperm, 25):
        Tp = era_t(Z, Yp[:, s:s + 25], era_arr); Sp = S_of(Tp)
        for e in ERAS:
            emx[e].append(np.nanmax(np.abs(Tp[e]), 0))
        for (a, b) in cor:
            for j in range(Tp[a].shape[1]):
                ta, tb = np.nan_to_num(Tp[a][:, j]), np.nan_to_num(Tp[b][:, j]); cor[(a, b)].append(np.corrcoef(ta, tb)[0, 1])
        mx.append(np.nanmax(np.abs(Sp), 0)); c10.append((np.abs(Sp) >= 1.0).sum(0)); c15.append((np.abs(Sp) >= 1.5).sum(0)); wf.append(wf_count(Tp))
    mx, c10, c15, wf = map(np.concatenate, (mx, c10, c15, wf))
    padj = np.array([(mx >= abs(s)).mean() if s != 0 else 1.0 for s in S0])
    WF = wf_list(T0, tests)
    obs_wf = int(WF.confirmed.sum())
    rec = []
    for (lbl, m), s, pa, *ts in zip(tests, S0, padj, *[T0[e][:, 0] for e in ERAS]):
        a, g, oa, va, b, gb, ob, vb = lbl
        rec.append(dict(family=name, var=a, gran=g, op=oa, cut=va, var2=b, op2=ob, cut2=vb, S=s, p_fw=pa,
                        t_2025=ts[0], t_H1=ts[1], t_H2=ts[2], n_zone=int(m.sum())))
    summ = dict(family=name, tests=len(tests), max_abs_S=float(np.nanmax(np.abs(S0))), null_q50=float(np.median(mx)), null_q95=float(np.percentile(mx, 95)),
                n_S_ge1=int((np.abs(S0) >= 1).sum()), null_med_S_ge1=float(np.median(c10)), p_count_ge1=float((c10 >= (np.abs(S0) >= 1).sum()).mean()),
                n_S_ge15=int((np.abs(S0) >= 1.5).sum()), null_med_S_ge15=float(np.median(c15)), p_count_ge15=float((c15 >= (np.abs(S0) >= 1.5).sum()).mean()),
                survivors_p10=int((padj < 0.10).sum()), survivors_p05=int((padj < 0.05).sum()),
                wf_confirmed=obs_wf, wf_null_mean=float(wf.mean()), wf_null_q95=float(np.percentile(wf, 95)), wf_p=float((wf >= obs_wf).mean()))
    for e in ERAS:
        summ[f"maxT_{e}"] = float(np.nanmax(np.abs(T0[e]))); summ[f"null_maxT_{e}_q50"] = float(np.median(np.concatenate(emx[e])))
    for (a, b), v in cor.items():
        summ[f"corr_t_{a}_vs_{b}"] = float(np.corrcoef(np.nan_to_num(T0[a][:, 0]), np.nan_to_num(T0[b][:, 0]))[0, 1])
        summ[f"null_corr_{a}_vs_{b}_q05_q95"] = f"{np.percentile(v, 5):+.2f}/{np.percentile(v, 95):+.2f}"
    WF["family"] = name
    WF["zone"] = [lab(tests[i][0]) for i in WF.idx]
    null = pd.DataFrame(dict(family=name, perm=np.arange(A.nperm), max_abs_S=mx, n_S_ge1=c10, n_S_ge15=c15, wf_confirmed=wf))
    return pd.DataFrame(rec), summ, WF, null


def lab(l):
    a, g, oa, va, b, gb, ob, vb = l
    s = f"{a} {oa} {va:.4g}"
    return s + (f" & {b} {ob} {vb:.4g}" if b else "")


# ───────────────────────── F1 / F2 (fill units) ─────────────────────────
t1 = build(CO, PV, era_arr=era, day_arr=day)
t1 += build(CO, PV, cols_b=PV, same=True, era_arr=era, day_arr=day)
# drop masks duplicated across 1D/2D
_s, T1 = set(), []
for t in t1:
    k = np.packbits(t[1]).tobytes()
    if k not in _s: _s.add(k); T1.append(t)
t2 = build(CO, PV, cols_b=MV, same=False, era_arr=era, day_arr=day)
print("F1 tests", len(T1), "F2 tests", len(t2), flush=True)
r1, s1, w1, n1 = run_family("F1 PAIR (1D+PAIRxPAIR)", T1, y, era, day, "dayblock", 11)
print(s1, flush=True)
r2, s2, w2, n2 = run_family("F2 PAIRxMACRO", t2, y, era, day, "dayblock", 22)
print(s2, flush=True)

# ───────────────────────── F3 (DAY units) ─────────────────────────
DD = CO.groupby("day").agg(**{c: (c, "mean") for c in MV}, y=("y", "mean"), era=("era", "first"), n=("y", "size")).reset_index()
t3 = build(DD, MV, era_arr=DD.era.values, day_arr=DD.day.values, minz=MINDAY, mind=MINDAY)
t3 += build(DD, MV, cols_b=MV, same=True, era_arr=DD.era.values, day_arr=DD.day.values, minz=MINDAY, mind=MINDAY)
print("F3 tests", len(t3), flush=True)
r3, s3, w3, n3 = run_family("F3 MACRO (day units)", t3, DD.y.values.astype(float), DD.era.values, DD.day.values, "perm", 33)
print(s3, flush=True)

TESTS = pd.concat([r1, r2, r3], ignore_index=True)
TESTS["zone"] = [lab((r["var"], r["gran"], r["op"], r["cut"], r["var2"] if isinstance(r["var2"], str) else None, None, r["op2"], r["cut2"])) for _, r in TESTS.iterrows()]
TESTS.to_csv(os.path.join(REP, "WILLY_2D_tests.csv"), index=False)
SUMM = pd.DataFrame([s1, s2, s3]); WFALL = pd.concat([w1, w2, w3], ignore_index=True)
pd.concat([n1, n2, n3]).to_csv(os.path.join(REP, "WILLY_2D_null.csv"), index=False)
WFALL.to_csv(os.path.join(REP, "WILLY_2D_walkforward.csv"), index=False)

# ───────────────────────── verdict reads ─────────────────────────
BE = C.breakeven_wr(y)
print("cohort breakeven WR", BE)


def zmask(frame, r):
    m = m_of(frame[r["var"]], r["op"], r["cut"])
    sc = frame[r["var"]].notna().values
    if isinstance(r["var2"], str):
        m = m & m_of(frame[r["var2"]], r["op2"], r["cut2"]); sc = sc & frame[r["var2"]].notna().values
    return m, sc


def read_zone(S, r):
    """fill-level read: zone stats, refusal Δ per fill, bar, concentration (S = frame with y, day, pair, month)."""
    for c in (r["var"], r["var2"]):
        if isinstance(c, str) and c not in S.columns:
            return {}
        if isinstance(c, str):
            S[c] = C.num(S, c) if c.startswith("entry_") else pd.to_numeric(S[c], errors="coerce")
    m, sc = zmask(S, r)
    Sz = S[m]; yy = S.y.values.astype(float); yz = yy[m]
    out = dict(N=len(S), scored=int(sc.sum()), n_zone=int(m.sum()), days_zone=Sz.day.nunique())
    if m.sum() == 0:
        return out
    lo, hi, ppos = C.day_boot(yz, Sz.day.values)
    out.update(wr_zone=100 * (yz > 0).mean(), avg_zone=yz.mean(), avg_rest=yy[sc & ~m].mean() if (sc & ~m).any() else np.nan,
               P_zone_neg=1 - ppos, refuse_delta=-yz.sum() / len(S), saved_losers=int((yz <= 0).sum()), lost_winners=int((yz > 0).sum()),
               P_refuse_pos=C.boot_diff_p(yy, m, S.day.values, -1) if (~m).sum() >= 3 and m.sum() >= 3 else np.nan)
    loss = Sz[Sz.y < 0]
    gl = -loss.y.sum()
    out["top_day_loss_share"] = (-loss.groupby("day").y.sum()).max() / gl if gl > 0 else np.nan
    out["top_pair_loss_share"] = (-loss.groupby("pair").y.sum()).max() / gl if gl > 0 else np.nan
    return out


def lomo(S, r):
    m, _ = zmask(S, r); yy = S.y.values.astype(float); sg = []
    for mo in sorted(S.month.unique()):
        k = S.month.values != mo
        sg.append(np.sign(-(yy[k & m]).sum()))
    return f"{int(sum(s > 0 for s in sg))}/{len(sg)} months refuse-Δ>0"


top = []
for fam, g in TESTS.groupby("family"):
    g = g[g.S != 0]
    sel = pd.concat([g[g.p_fw < 0.10], g.reindex(g.S.abs().sort_values(ascending=False).index).head(5)]).drop_duplicates(["zone"])
    top.append(sel)
TOP = pd.concat(top, ignore_index=True)
OUT = []
seed_frames = {s: ALL[(ALL.seed == s) & (ALL.tick_priced.astype(str).str.lower() == "true")].copy() for s in (1, 2, 3)}
onem = ALL[(ALL.seed == 1) & (ALL.tick_priced.astype(str).str.lower() != "true") & (ALL.era == "2025")].copy()
for _, r in TOP.iterrows():
    rec = dict(family=r.family, zone=r.zone, S=r.S, p_fw=r.p_fw, t_2025=r.t_2025, t_H1=r.t_H1, t_H2=r.t_H2)
    if r.family.startswith("F3"):
        # day-unit zone → implementable fill gate = each fill's OWN stamp at the frozen day cut (panel vars are per day anyway)
        pass
    allr = read_zone(CO.copy(), r)
    rec.update({f"all_{k}": v for k, v in allr.items()})
    for e in ERAS:
        er = read_zone(CO[CO.era == e].copy(), r)
        rec.update({f"{e}_{k}": er.get(k) for k in ("n_zone", "days_zone", "wr_zone", "avg_zone", "avg_rest", "refuse_delta", "P_zone_neg", "P_refuse_pos")})
    for s in (2, 3):
        sr = read_zone(seed_frames[s].copy(), r); rec[f"seed{s}_refuse_delta"] = sr.get("refuse_delta"); rec[f"seed{s}_avg_zone"] = sr.get("avg_zone")
    o = read_zone(onem.copy(), r); rec["1m2025_n_zone"] = o.get("n_zone"); rec["1m2025_avg_zone"] = o.get("avg_zone"); rec["1m2025_refuse_delta"] = o.get("refuse_delta")
    rec["lomo"] = lomo(CO.copy(), r)
    loser = (allr.get("avg_zone", 0) or 0) < 0
    bar = [allr.get("wr_zone", 100) < 100 * BE, (allr.get("P_zone_neg") or 0) >= 0.95,
           allr.get("days_zone", 0) >= 8 and (allr.get("top_day_loss_share") or 1) < 0.5 and (allr.get("top_pair_loss_share") or 1) < 0.5,
           allr.get("n_zone", 0) >= 15]
    rec["bar_items"] = "".join("✔" if b else "✗" for b in bar) if loser else "winner-zone"
    rec["bar_pass"] = loser and all(bar)
    OUT.append(rec)
TOPR = pd.DataFrame(OUT)
TOPR.to_csv(os.path.join(REP, "WILLY_2D_top.csv"), index=False)

# ───────────────────────── master / live blocks ─────────────────────────
MS = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
MS = MS[MS.entry_strategy == "FRENZY_WILLY"].copy()
RM = R[R.block == "MASTER"][["pair", "opened_at", "era", "y", "why", "src"]].copy()
MS["k"] = MS.pair + "|" + MS.opened_at.astype(str).str[:19]; RM["k"] = RM.pair + "|" + RM.opened_at.astype(str).str[:19]
MS = MS.merge(RM[["k", "y", "why", "era"]], on="k", how="inner", suffixes=("", "_r"))
assert len(MS) == 19, len(MS)
MS["day"] = pd.to_datetime(MS.opened_at.astype(str).str[:19]).dt.strftime("%Y-%m-%d")
MS["ts"] = pd.to_datetime(MS.opened_at.astype(str).str[:19])
LV = pd.read_csv(LIVE, low_memory=False); LV = LV[LV.entry_strategy == "FRENZY_WILLY"].copy()
LV["y"] = LV.pnl_percentage.astype(float); LV["day"] = pd.to_datetime(LV.opened_at).dt.strftime("%Y-%m-%d"); LV["ts"] = pd.to_datetime(LV.opened_at)
for X in (MS, LV):
    X["hour"] = X.ts.dt.hour.astype(float)
    X["lvol24"] = np.log10(pd.to_numeric(X.entry_pair_volume_24h_usd, errors="coerce").clip(lower=1))
    X["fz_age_min"] = (X.ts - pd.to_datetime(X.entry_frenzy_spike_at, errors="coerce", format="mixed").dt.tz_localize(None)).dt.total_seconds() / 60
    X["ep"] = X.pair + "|" + X.entry_frenzy_spike_at.astype(str)
    X.sort_values("ts", inplace=True); X["k_ep"] = X.groupby("ep").cumcount() + 1.0
    X["top50f"] = np.nan
    for c in PCOLS:
        X[c] = X.day.map(PANEL[c]) if c in PANEL else np.nan
cov2 = []
for c, k in VARS:
    cov2.append(dict(var=c, master_scored=round(100 * pd.to_numeric(C.num(MS, c) if c.startswith("entry_") else MS[c], errors="coerce").notna().mean(), 1),
                     live_scored=round(100 * pd.to_numeric(C.num(LV, c) if c.startswith("entry_") else LV[c], errors="coerce").notna().mean(), 1)))
COV = pd.DataFrame(cov_rows).merge(pd.DataFrame(cov2), on="var", how="left")
COV.to_csv(os.path.join(REP, "WILLY_2D_coverage.csv"), index=False)

blk = []
show = TOPR.sort_values(["p_fw"]).groupby("family").head(3)
for _, r in show.iterrows():
    rr = TESTS[TESTS.zone == r.zone].iloc[0]
    for name, X, usd in (("MASTER", MS, "pnl"), ("LIVE", LV, "pnl")):
        X = X.copy()
        for c in (rr["var"], rr["var2"]):
            if isinstance(c, str):
                X[c] = C.num(X, c) if c.startswith("entry_") else pd.to_numeric(X[c], errors="coerce")
        m, sc = zmask(X, rr)
        for _, x in X.iterrows():
            i = X.index.get_loc(x.name)
            blk.append(dict(family=r.family, zone=r.zone, block=name, batch=(x.get("batch") if isinstance(x.get("batch"), str) else x.get("era_r", x.get("era"))) if name == "MASTER" else "LIVE 10-15/16",
                            pair=x.pair, opened=str(x.opened_at)[:16], scored=bool(sc[i]), in_zone=bool(m[i]) if sc[i] else None,
                            pct_before=float(x.y), usd_before=float(x[usd]), pct_after=(0.0 if (sc[i] and m[i]) else float(x.y)),
                            usd_after=(0.0 if (sc[i] and m[i]) else float(x[usd]))))
BLK = pd.DataFrame(blk); BLK.to_csv(os.path.join(REP, "WILLY_2D_blocks.csv"), index=False)

# ───────────────────────── tables ─────────────────────────
md = ["## Screen size vs null\n", C.md_table(SUMM, 3), f"\nCohort {len(CO)} tick fills · days {CO.day.nunique()} · breakeven WR {100 * BE:.1f} % · avg {y.mean():+.3f}\n",
      "\n## Top tests + verdict reads\n", C.md_table(TOPR, 3), "\n## Walk-forward confirmed\n", C.md_table(WFALL[WFALL.confirmed], 2) if WFALL.confirmed.any() else "(none)"]
open(os.path.join(REP, "WILLY_2D_tables.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
