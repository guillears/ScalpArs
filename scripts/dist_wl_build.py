#!/usr/bin/env python3
"""DIST_WL (PREREG reports/DIST_WL_2026-11-18_PREREG.md) — BUILD the L1 / DISTRIBUTION_SHORT trade rows with every entry-time feature,
the theory columns, the path labels and the anchored price path.  READ-ONLY on the bot.

Trade set = the frozen L1 (g_VSW15 ∧ top_pos_chg4h ≤ −0.01097) from reports/UNIFIED_SHORT_data/pool.pkl, exits / costs / one-position
exactly as scripts/leads_deep_l1.py (unified_short_lib.net / one_pos). Every L1 candidate row is kept (flag in_LIVE / in_T5A = the
one-position sets) so a cut can re-run one_pos.
New lag-safe columns (signal 15m bar = 5m bars j−2..j, closes at T = entry_ms; metrics create_time ≤ T − 5 min):
  close_pos15, lowwick15, range15_pct, last5_share, n_prior_vsw_red, oi_chg15m, oi_chg30m, top_pos_chg1h, top_acct_chg4h,
  top_pos_chg4h_rel, top_acct_chg4h_rel, pos_minus_acct_rel, glob_chg4h.
Path (5m bars from the entry bar J, short-favourable %, gross): WORKED = fav ≥ 1 % before adv ≥ the trade's own stop distance
Y = min(1.5 × ATR15, 6 %); NEVER = adv first; AMBIG = one bar touches both; FLAT = neither in 24 h. REACH = best fav before the
adv ≥ Y passage (or 24 h). Groups as scripts/path_labels.py (X = 1 %, recovered window 120 min, best-after-close window 45 min,
round-trip cost 0.30 %).  Output reports/DIST_WL_2026-11-18_data/rows.pkl + paths.npy
Usage: venv/bin/python -I scripts/dist_wl_build.py [--procs 5]"""
import argparse, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
import unified_short_lib as L  # noqa: E402
import exh_expand_build as B   # noqa: E402,F401 (patches the k5m / funding sources)
import ta_wl_features as FW    # noqa: E402
TB = B.TB
R = L.R
OUT = os.path.join(R, "reports", "DIST_WL_2026-11-18_data"); os.makedirs(OUT, exist_ok=True)
M5, H1, H4 = 300_000, 3_600_000, 14_400_000
THR = -0.01097
HOLD = 288; PRE = 24           # anchored path: 2 h before the entry … 24 h after
X_WORK = 1.0; REC_BARS = 24; BEST_POST = 9; COST_RT = 0.30


def one_pair(a):
    p, rows = a                                   # rows: DataFrame (entry_ms, atr15_pct, LIVE_x, LIVE_r, ...)
    x = TB.load(p)
    n = len(rows); out = {k: np.full(n, np.nan) for k in (
        "close_pos15", "lowwick15", "range15_pct", "last5_share", "oi_chg15m", "oi_chg30m", "top_pos_chg1h", "top_acct_chg4h",
        "top_pos_chg4h_rel", "top_acct_chg4h_rel", "pos_minus_acct_rel", "glob_chg4h", "p_fav", "p_adv", "reach", "best_close",
        "rec", "label")}
    path = np.full((n, PRE + HOLD + 1), np.nan, np.float32)
    if x is None:
        return p, rows.index.values, out, path
    t, o, h, l, c, v, qv = x
    T = rows.entry_ms.values.astype("int64")
    J = np.searchsorted(t, T); ok = (J < len(t)) & (t[np.minimum(J, len(t) - 1)] == T) & (J >= 3)
    for i in np.nonzero(ok)[0]:
        j = J[i] - 1                                    # last 5m bar of the signal 15m candle
        O15, H15, L15, C15 = o[j - 2], h[j - 2:j + 1].max(), l[j - 2:j + 1].min(), c[j]
        rg = H15 - L15
        if rg > 0:
            out["close_pos15"][i] = (C15 - L15) / rg; out["lowwick15"][i] = (min(O15, C15) - L15) / rg
        out["range15_pct"][i] = rg / O15 * 100
        vv = qv[j - 2:j + 1].sum(); out["last5_share"][i] = qv[j] / vv if vv > 0 else np.nan
        Jn = J[i]; E = o[Jn]; Y = min(1.5 * rows.atr15_pct.values[i], 6.0)
        a0 = max(Jn - PRE, 0); a1 = min(Jn + HOLD + 1, len(c))
        path[i, (a0 - (Jn - PRE)):(a1 - (Jn - PRE))] = -(c[a0:a1] / E - 1) * 100
        e1 = min(Jn + HOLD, len(c))
        fav = (E - l[Jn:e1]) / E * 100; adv = (h[Jn:e1] - E) / E * 100
        hw = np.nonzero(fav >= X_WORK)[0]; hs = np.nonzero(adv >= Y)[0]
        fw = hw[0] if len(hw) else 10**9; fs = hs[0] if len(hs) else 10**9
        lab = 3 if (fw == fs and fw < 10**9) else (1 if fw < fs else (0 if fs < fw else 2))   # 1 WORKED · 0 NEVER · 2 FLAT · 3 AMBIG
        out["label"][i] = lab
        cut = fs if fs < 10**9 else len(fav)
        out["reach"][i] = fav[:max(cut, 1)].max() if cut > 0 else max(fav[0], 0.0)
        out["p_fav"][i] = fav.max(); out["p_adv"][i] = adv.max()
        xb = int((rows.LIVE_x.values[i] - T[i]) // M5)  # exit bar count from the entry
        xe = min(Jn + xb, len(c))
        bc = min(xe + BEST_POST, len(c)); out["best_close"][i] = ((E - l[Jn:bc]) / E * 100).max() if bc > Jn else np.nan
        re = min(xe + REC_BARS, len(c)); out["rec"][i] = float(((E - l[xe:re]) / E * 100 >= X_WORK).any()) if re > xe else 0.0
    me = FW.metrics_frame(p)
    if me is not None and len(me) > 60:
        ct = me.ct.values; oiv = me.sum_open_interest_value.values.astype(float)
        tpl = me.sum_toptrader_long_short_ratio.values.astype(float); tal = me.count_toptrader_long_short_ratio.values.astype(float)
        gl = me.count_long_short_ratio.values.astype(float)
        km = np.searchsorted(ct, T - M5, side="right") - 1
        fresh = (km >= 0) & (T - ct[np.maximum(km, 0)] <= 30 * 60_000)
        kc = ct[np.maximum(km, 0)]

        def lag(w):
            k = np.searchsorted(ct, kc - w, side="right") - 1
            return k, fresh & (k >= 0) & (kc - ct[np.maximum(k, 0)] <= w + 10 * 60_000)
        with np.errstate(divide="ignore", invalid="ignore"):
            for w, nm in ((15 * 60_000, "oi_chg15m"), (30 * 60_000, "oi_chg30m")):
                k, okk = lag(w); out[nm][okk] = (oiv[km[okk]] / oiv[k[okk]] - 1) * 100
            k, okk = lag(H1); out["top_pos_chg1h"][okk] = tpl[km[okk]] - tpl[k[okk]]
            k, okk = lag(H4)
            out["top_acct_chg4h"][okk] = tal[km[okk]] - tal[k[okk]]
            out["top_pos_chg4h_rel"][okk] = (tpl[km[okk]] / tpl[k[okk]] - 1) * 100
            out["top_acct_chg4h_rel"][okk] = (tal[km[okk]] / tal[k[okk]] - 1) * 100
            out["pos_minus_acct_rel"][okk] = out["top_pos_chg4h_rel"][okk] - out["top_acct_chg4h_rel"][okk]
            out["glob_chg4h"][okk] = gl[km[okk]] - gl[k[okk]]
    return p, rows.index.values, out, path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=5); A = ap.parse_args()
    X = L.load()
    vsw = X.g_VSW15.values
    frozen = vsw & (X.top_pos_chg4h.values <= THR)
    S = X[frozen].copy(); S["orig_idx"] = np.nonzero(frozen)[0]
    for e in ("T5A", "LIVE"):
        ii = L.one_pos(X, np.nonzero(frozen)[0], e); m = np.zeros(len(X), bool); m[ii] = True; S[f"in_{e}"] = m[frozen]
        v = L.net(X, ii, e); print(f"repro {e}: N {len(ii)} avg {v.mean():+.3f} WR {(v > 0).mean() * 100:.1f}", flush=True)
        S[f"net_{e}"] = L.net(X, S.orig_idx.values, e); S[f"net20_{e}"] = L.net(X, S.orig_idx.values, e, 0.20)
    # earlier red VSW candles in the same episode (all VSW rows of the pool, not just L1)
    V = X.loc[vsw, ["ep", "entry_ms"]].copy(); V["ep"] = V.ep.astype(str); V = V.sort_values(["ep", "entry_ms"])
    V["n_prior_vsw_red"] = V.groupby("ep").cumcount()
    nmap = dict(zip(V.ep + "|" + V.entry_ms.astype(str), V.n_prior_vsw_red))
    S["n_prior_vsw_red"] = (S.ep.astype(str) + "|" + S.entry_ms.astype(str)).map(nmap).astype(float)
    S = S.reset_index(drop=True); S["pair"] = S.pair.astype(str); S["ep"] = S.ep.astype(str)
    del X
    jobs = [(p, S.loc[S.pair == p, ["entry_ms", "atr15_pct", "LIVE_x", "LIVE_r"]]) for p in S.pair.unique()]
    cols = None; P = np.full((len(S), PRE + HOLD + 1), np.nan, np.float32)
    with mp.get_context("fork").Pool(A.procs) as pool:
        for k, (p, ix, out, path) in enumerate(pool.imap_unordered(one_pair, jobs, chunksize=2)):
            for nm, arr in out.items():
                if nm not in S.columns:
                    S[nm] = np.nan
                S.loc[ix, nm] = arr
            P[ix] = path
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), k, len(jobs), flush=True)
    S.to_pickle(os.path.join(OUT, "rows.pkl")); np.save(os.path.join(OUT, "paths.npy"), P)
    print("done", S.shape, flush=True)
