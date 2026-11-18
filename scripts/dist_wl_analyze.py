#!/usr/bin/env python3
"""DIST_WL (PREREG reports/DIST_WL_2026-11-18_PREREG.md) — winners vs losers of L1 / DISTRIBUTION_SHORT under the EXHAUSTION live exit.
Steps: path decomposition → coverage → theory predictions (DESIGN) → exhaustive 1D + 2D screen on 3 outcomes with a family-wise
shuffled null → survivor steps (DESIGN) → HOLDOUT once → before / after + Kelly. Input reports/DIST_WL_2026-11-18_data/rows.pkl
(scripts/dist_wl_build.py). Outputs → reports/DIST_WL_2026-11-18_data/*.csv / *.pkl + console log.
Usage: venv/bin/python -I scripts/dist_wl_analyze.py [--shuffles 200]"""
import argparse, itertools, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import unified_short_lib as L  # noqa: E402
OUT = os.path.join(L.R, "reports", "DIST_WL_2026-11-18_data")
DAY = 86_400_000
HOLD_START = L.TS("2025-07-01")
SIZE = 0.61
MIN_N = 50
DES_HALVES = ["23-H2", "24-H1", "24-H2", "25-H1"]
THEORY = {"P1": ["dd_peak_pct", "dist_vwap_pct", "dist_base_pct"],
          "P2": ["close_pos15", "range15_pct", "bar_ret_pct", "rsi15"],
          "P3": ["oi_chg15m", "oi_chg1h"],
          "P4": ["top_acct_chg4h", "pos_minus_acct_rel", "top_pos_ls"],
          "P5": ["top_pos_chg4h", "top_pos_chg4h_rel", "top_pos_chg1h"],
          "P6": ["last5_share", "vmult15", "n_prior_vsw_red"],
          "P7": ["spot_share1h", "ssc"],
          "P8": ["fund_last", "fund_hourly", "prem_last"],
          "P9": ["hrs_since_spike", "hrs_since_peak"],
          "H1": ["ret1h_pct"], "H2": ["btc_d1_regime"]}
# theory: the side expected to be BETTER for the short ("hi" = high values better, "lo" = low values better)
PRED = {"dd_peak_pct": "hi", "dist_vwap_pct": "hi", "dist_base_pct": "hi", "close_pos15": "lo", "range15_pct": "lo", "bar_ret_pct": "hi",
        "rsi15": "hi", "oi_chg15m": "lo", "oi_chg1h": "lo", "top_acct_chg4h": "hi", "pos_minus_acct_rel": "lo", "top_pos_ls": "hi",
        "top_pos_chg4h": "lo", "top_pos_chg4h_rel": "lo", "top_pos_chg1h": "lo", "last5_share": "hi", "vmult15": "lo",
        "n_prior_vsw_red": "lo", "spot_share1h": "lo", "ssc": "lo", "fund_last": "hi", "fund_hourly": "lo", "prem_last": "hi",
        "hrs_since_spike": "lo", "hrs_since_peak": "lo", "ret1h_pct": "hi", "btc_d1_regime": "hi"}
EXCL_PREFIX = ("T3A_", "T5A_", "T10A_", "T3F_", "T5F_", "T10F_", "LIVE_", "net", "in_")
EXCL = {"pair", "entry_ms", "ep_on_ms", "spike_ms", "orig_idx", "day", "block", "half", "ep", "is15", "g_VSW15", "p_fav", "p_adv",
        "reach", "best_close", "rec", "label", "lowwick15", "spot_tried"}


# ───────────────────────── helpers
def one_pos(S, mask):
    """unified_short_lib.one_pos on the row frame (LIVE exit): greedy per pair in time order, drops censored (k == 3)."""
    ii = np.nonzero(mask & (S.LIVE_k.values != 3))[0]
    o = np.lexsort((S.entry_ms.values[ii], S.pair.values[ii])); ii = ii[o]
    keep = []; lp = None; busy = -1
    P = S.pair.values; T = S.entry_ms.values; XX = S.LIVE_x.values
    for i in ii:
        if P[i] != lp:
            lp = P[i]; busy = -1
        if T[i] < busy:
            continue
        keep.append(i); busy = XX[i]
    m = np.zeros(len(S), bool); m[keep] = True
    return m


def wilson_floor(k, n, z=1.645):
    return L.wilson_floor(k, n, z)


def kelly(v, day=None):
    w = v > 0
    if w.sum() == 0 or (~w).sum() == 0:
        return np.nan, np.nan, np.nan
    aw, al = v[w].mean(), -v[~w].mean(); b = aw / al
    n = len(np.unique(day)) if day is not None else len(v)
    p = wilson_floor(w.mean() * n, n)
    f = p - (1 - p) / b
    return f, p, f / (al / 100)                      # bet fraction · WR floor · notional × equity it implies


def dcr(v_pct, xday, size=SIZE, span=None):
    s = pd.Series(v_pct / 100 * size).groupby(xday).sum()
    lo, hi = span if span else (s.index.min(), s.index.max())
    full = s.reindex(np.arange(lo, hi + 1), fill_value=0.0)
    eq = np.cumprod(1 + full.values)
    return (eq[-1] ** (1 / len(full)) - 1) * 100, (eq / np.maximum.accumulate(eq) - 1).min() * 100, full.min() * 100


def welch(M, Sc, y, ok):
    """cell (M) vs rest (Sc & ~M) Welch t, rows with ok only. M, Sc: n × K float32 0/1; y: n or n × B."""
    y = np.where(ok[:, None] if y.ndim == 2 else ok, y, 0.0)
    okf = ok.astype(np.float32)
    Mo = M * okf[:, None]; So = Sc * okf[:, None]
    n1 = Mo.sum(0); n0 = So.sum(0) - n1
    Y = y if y.ndim == 2 else y[:, None]
    s1 = Mo.T @ Y; q1 = Mo.T @ (Y * Y); sA = So.T @ Y; qA = So.T @ (Y * Y)
    s0 = sA - s1; q0 = qA - q1
    n1_, n0_ = n1[:, None], n0[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        m1 = s1 / n1_; m0 = s0 / n0_
        v1 = (q1 - n1_ * m1 ** 2) / (n1_ - 1); v0 = (q0 - n0_ * m0 ** 2) / (n0_ - 1)
        t = (m1 - m0) / np.sqrt(v1 / n1_ + v0 / n0_)
    t[(n1_ < MIN_N).repeat(t.shape[1], 1) | (n0_ < MIN_N).repeat(t.shape[1], 1)] = np.nan
    return (t if y.ndim == 2 else t[:, 0]), n1, m1[:, 0] if y.ndim == 1 else None, m0[:, 0] if y.ndim == 1 else None


def bins_for(x, des):
    """list of (granularity, label, lo, hi) bins on DESIGN edges; categorical when ≤ 6 distinct values."""
    xd = x[des & np.isfinite(x)]
    out = []
    if len(xd) < 2 * MIN_N:
        return out
    u = np.unique(xd)
    if len(u) <= 6:
        for a in u:
            out.append(("cat", f"={a:g}", a, a))
        return out
    if (xd < 0).sum() >= MIN_N and (xd >= 0).sum() >= MIN_N:
        out += [("sign", "<0", -np.inf, 0.0), ("sign", "≥0", 0.0, np.inf)]
    for g, q in (("terc", 3), ("quint", 5)):
        e = np.unique(np.quantile(xd, np.linspace(0, 1, q + 1)[1:-1]))
        ed = np.r_[-np.inf, e, np.inf]
        for k in range(len(ed) - 1):
            out.append((g, f"{g[0].upper()}{k + 1}/{q}", ed[k], ed[k + 1]))
    return out


def in_bin(x, lo, hi):
    with np.errstate(invalid="ignore"):
        if lo == hi:
            return x == lo
        return (x >= lo) & (x < hi) if np.isfinite(hi) else (x >= lo)


def block_boot(v, day, keep, B=2000, seed=7):
    """day-block bootstrap of Δavg = avg(kept) − avg(all) on the fixed trade set."""
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(day, return_inverse=True)
    sa = np.bincount(inv, v); na = np.bincount(inv); sk = np.bincount(inv, v * keep); nk = np.bincount(inv, keep.astype(float))
    W = rng.multinomial(len(ud), np.full(len(ud), 1 / len(ud)), size=B).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = (W @ sk) / (W @ nk) - (W @ sa) / (W @ na)
    return np.nanpercentile(d, [2.5, 97.5])


def block_boot_mean(v, day, B=2000, seed=11):
    rng = np.random.default_rng(seed)
    ud, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, v); n = np.bincount(inv)
    W = rng.multinomial(len(ud), np.full(len(ud), 1 / len(ud)), size=B).astype(float)
    return np.nanpercentile((W @ s) / (W @ n), [2.5, 97.5, 95.0])


def b5(v):
    s = np.sort(v)[::-1]; return s[int(np.ceil(0.05 * len(s))):].mean() if len(s) > 20 else np.nan


# ───────────────────────── main
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shuffles", type=int, default=200); A = ap.parse_args()
    S = pd.read_pickle(os.path.join(OUT, "rows.pkl"))
    chk = one_pos(S, np.ones(len(S), bool))
    assert (chk == S.in_LIVE.values).all(), "one_pos re-implementation drifted"
    base = S.in_LIVE.values
    B_ = S[base].reset_index(drop=True)
    v = B_.net_LIVE.values.astype(float); day = B_.day.values
    des = B_.entry_ms.values < HOLD_START
    hol = ~des
    R = {}                                            # everything the report writer needs
    R["repro"] = dict(T5A_N=int(S.in_T5A.sum()), T5A_avg=float(S.net_T5A[S.in_T5A].mean()), T5A_WR=float((S.net_T5A[S.in_T5A] > 0).mean() * 100),
                      LIVE_N=int(base.sum()), LIVE_avg=float(v.mean()), LIVE_WR=float((v > 0).mean() * 100))
    print("repro", R["repro"], flush=True)
    # ── path decomposition (all trades; DESIGN / HOLDOUT split too)
    lab = B_.label.values; best_net = B_.best_close.values - 0.30
    grp = np.where(v > 0, np.where(best_net - v >= np.maximum(1.0, v), "LEFT_ON_TABLE", "CAPTURED"),
                   np.where(lab == 1, "GAVE_BACK", np.where(lab == 0, np.where(B_.rec.values > 0, "STOPPED_RECOVERED", "NEVER_WORKED"),
                                                            np.where(lab == 3, "AMBIG", "DRIFT_LOSS"))))
    B_["group"] = grp
    B_["entry_label"] = pd.Series(lab).map({1: "WORKED", 0: "NEVER", 2: "FLAT", 3: "AMBIG"}).values
    B_["usd_per_10k"] = v * SIZE * 100            # $ per $10k equity at 0.61 × equity notional
    rows = []
    for nm, m in (("ALL", np.ones(len(v), bool)), ("DESIGN", des), ("HOLDOUT", hol)):
        for g in ("NEVER_WORKED", "STOPPED_RECOVERED", "GAVE_BACK", "DRIFT_LOSS", "AMBIG", "LEFT_ON_TABLE", "CAPTURED"):
            mm = m & (grp == g)
            rows.append(dict(period=nm, group=g, N=int(mm.sum()), share=mm.sum() / m.sum() * 100, avg=v[mm].mean() if mm.any() else np.nan,
                             usd_10k=B_.usd_per_10k.values[mm].sum(), reach_med=np.median(B_.reach.values[mm]) if mm.any() else np.nan,
                             exit_tp=(B_.LIVE_k.values[mm] == 0).mean() * 100 if mm.any() else np.nan,
                             exit_sl=(B_.LIVE_k.values[mm] == 1).mean() * 100 if mm.any() else np.nan))
    PD = pd.DataFrame(rows); PD.to_csv(os.path.join(OUT, "path_decomp.csv"), index=False)
    print(PD.round(2).to_string(), flush=True)
    R["exit_mix"] = {str(k): int(c) for k, c in pd.Series(B_.LIVE_k.values).value_counts().items()}
    R["tp_dist_med"] = float(np.median(np.minimum(B_.atr15_pct * 20, 10))); R["sl_dist_med"] = float(np.median(np.minimum(B_.atr15_pct * 1.5, 6)))
    R["tp_at_cap"] = float((B_.atr15_pct * 20 >= 10).mean() * 100)
    # ── variables + coverage
    feats = [c for c in B_.columns if c not in EXCL and not c.startswith(EXCL_PREFIX) and c not in ("group", "entry_label", "usd_per_10k")]
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(B_[c]) or pd.api.types.is_bool_dtype(B_[c])]
    Xf = {c: B_[c].values.astype(float) for c in feats}
    feats = [c for c in feats if np.nanstd(Xf[c][des]) > 0 and np.isfinite(Xf[c][des]).sum() >= 2 * MIN_N]
    cov = pd.DataFrame([dict(var=c, cov_all=np.isfinite(Xf[c]).mean() * 100, cov_design=np.isfinite(Xf[c][des]).mean() * 100,
                             cov_holdout=np.isfinite(Xf[c][hol]).mean() * 100, n_unique=len(np.unique(Xf[c][np.isfinite(Xf[c])])),
                             theory=next((k for k, vs in THEORY.items() if c in vs), "")) for c in feats])
    cov.to_csv(os.path.join(OUT, "coverage.csv"), index=False)
    print("features", len(feats), flush=True)
    # ── outcomes (DESIGN rows)
    d_idx = np.nonzero(des)[0]; nd = len(d_idx)
    y_net = v[d_idx]
    y_wk = np.where(lab[d_idx] == 1, 1.0, np.where(lab[d_idx] == 0, 0.0, np.nan))     # FLAT / AMBIG excluded
    y_rc = np.minimum(B_.reach.values[d_idx], 10.0)
    OUTC = {"net": (y_net, np.ones(nd, bool)), "worked": (np.nan_to_num(y_wk), np.isfinite(y_wk)), "reach": (y_rc, np.ones(nd, bool))}
    # ── 1D tests
    tests = []; Mcols = []; Scols = []
    binsd = {}
    for c in feats:
        x = Xf[c][d_idx]; bs = bins_for(Xf[c], des); binsd[c] = bs
        sc = np.isfinite(x)
        for g, lb, lo, hi in bs:
            m = in_bin(x, lo, hi) & sc
            tests.append(dict(kind="1D", var=c, var2="", gran=g, bin=lb, lo=lo, hi=hi, lo2=np.nan, hi2=np.nan, bin2=""))
            Mcols.append(m); Scols.append(sc)
    n1d = len(tests)
    # ── 2D tests (sign × sign, tercile × tercile; categorical used as is)
    def coarse(c):
        bs = binsd[c]
        s = [b for b in bs if b[0] == "sign"]; t = [b for b in bs if b[0] == "terc"]; k = [b for b in bs if b[0] == "cat"]
        return [("sign", s), ("terc", t)] if (s or t) else [("cat", k)]
    for c1, c2 in itertools.combinations(feats, 2):
        x1 = Xf[c1][d_idx]; x2 = Xf[c2][d_idx]; sc = np.isfinite(x1) & np.isfinite(x2)
        if sc.sum() < 2 * MIN_N:
            continue
        for g1, b1 in coarse(c1):
            for g2, b2 in coarse(c2):
                if not b1 or not b2 or (g1 == "sign") != (g2 == "sign") and "cat" not in (g1, g2):
                    continue
                for (_, l1, lo1, hi1), (_, l2, lo2, hi2) in itertools.product(b1, b2):
                    m = in_bin(x1, lo1, hi1) & in_bin(x2, lo2, hi2) & sc
                    if m.sum() < MIN_N or (sc & ~m).sum() < MIN_N:
                        continue
                    tests.append(dict(kind="2D", var=c1, var2=c2, gran=f"{g1}×{g2}", bin=l1, bin2=l2, lo=lo1, hi=hi1, lo2=lo2, hi2=hi2))
                    Mcols.append(m); Scols.append(sc)
    T = pd.DataFrame(tests); K = len(T)
    print("tests", n1d, "1D +", K - n1d, "2D =", K, flush=True)
    rng = np.random.default_rng(20261118)
    CH = 4000
    for oname, (y, ok) in OUTC.items():
        tt = np.full(K, np.nan); n1 = np.zeros(K); m1 = np.full(K, np.nan); m0 = np.full(K, np.nan)
        oki = np.nonzero(ok)[0]                    # shuffle the outcome among scored rows (scored set held fixed)
        Yp = np.repeat(y[:, None], A.shuffles, 1)
        for b in range(A.shuffles):
            Yp[oki, b] = y[rng.permutation(oki)]
        nullmax = np.zeros(A.shuffles); nullmax_th = np.zeros(A.shuffles); null_gt3 = np.zeros(A.shuffles)
        th_vars = set(sum(THEORY.values(), []))
        is_th = (T.kind.values == "1D") & T["var"].isin(th_vars).values
        for a0 in range(0, K, CH):
            M = np.stack(Mcols[a0:a0 + CH], 1).astype(np.float32); Sc = np.stack(Scols[a0:a0 + CH], 1).astype(np.float32)
            t_, n_, mm1, mm0 = welch(M, Sc, y.astype(np.float32), ok)
            tt[a0:a0 + CH] = t_; n1[a0:a0 + CH] = n_; m1[a0:a0 + CH] = mm1; m0[a0:a0 + CH] = mm0
            # null: permute (y, ok) jointly across DESIGN rows
            tn = np.zeros((M.shape[1], A.shuffles), np.float32)
            for b0 in range(0, A.shuffles, 50):
                yy = Yp[:, b0:b0 + 50].astype(np.float32)
                tn[:, b0:b0 + 50] = welch(M, Sc, yy, ok)[0]
            at = np.abs(np.nan_to_num(tn))
            nullmax = np.maximum(nullmax, at.max(0)); null_gt3 += (at > 3).sum(0)
            th = is_th[a0:a0 + CH]
            if th.any():
                nullmax_th = np.maximum(nullmax_th, at[th].max(0))
            print(time.strftime("%H:%M:%S"), oname, a0, K, flush=True)
        T[f"t_{oname}"] = tt; T[f"m1_{oname}"] = m1; T[f"m0_{oname}"] = m0
        if oname == "net":
            T["n_cell"] = n1
        R[f"null_{oname}"] = dict(q95=float(np.quantile(nullmax, 0.95)), q50=float(np.median(nullmax)),
                                  q95_theory=float(np.quantile(nullmax_th, 0.95)), exp_gt3=float(null_gt3.mean()),
                                  obs_gt3=int((np.abs(np.nan_to_num(tt)) > 3).sum()),
                                  obs_gt_q95=int((np.abs(np.nan_to_num(tt)) > np.quantile(nullmax, 0.95)).sum()),
                                  obs_th_gt_q95th=int((np.abs(np.nan_to_num(tt[is_th])) > np.quantile(nullmax_th, 0.95)).sum()))
        print(oname, R[f"null_{oname}"], flush=True)
    T.to_pickle(os.path.join(OUT, "screen.pkl"))
    json.dump(R, open(os.path.join(OUT, "stage1.json"), "w"), indent=1, default=float)
    B_.to_pickle(os.path.join(OUT, "base.pkl"))
    pd.to_pickle(dict(binsd=binsd, feats=feats), os.path.join(OUT, "bins.pkl"))
    print("stage1 done", flush=True)
