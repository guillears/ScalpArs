#!/usr/bin/env python3
"""OI_TS_AUDIT 2026-11-21 — re-run the two LIVE decisions that read vision metrics with ALIGNED timestamps (research only, no network).

  DIST  DL 586 DISTRIBUTION_SHORT trigger: VSW15 ∧ top-trader POSITION L/S 4 h change ≤ −0.01097 (+ the armed RSI1h ≥ 74.07 keep cut)
  OIV   DL 582 EXH_SSC_OIVOL filter: block EXH when ssc > −0.0366 ∧ OI_usd ÷ vol24 > 0.128

Pool, exit (LIVE = the EXHAUSTION live exit the decisions were made on), one-position rule and costs = scripts/unified_short_lib.py, as in
scripts/leads_deep_l1.py / dist_wl_build.py / leads_deep_l4.py. Three metric reads per row:
  ORIG   create_time ≤ T − 5 min (what the studies used; = snapshot ≤ T in 2024-05 … 2026-11-15)
  LIVE   aligned snapshot ≤ T − 5 min (= what the live engine reads: API timestamp = snapshot time, ≤ T − 5 min)
  NAIVE  create_time ≤ T (the look-ahead reader, for scale only)
Output reports/OI_TS_AUDIT_2026-11-21_data/*.csv
Usage: venv/bin/python -I "scripts/OI_TS_AUDIT_2026-11-21_rerun.py" [--procs 5]"""
import argparse, importlib.util, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import unified_short_lib as L  # noqa: E402
import exh_expand_build as B   # noqa: E402,F401 (patches the k5m sources, as dist_wl_build)
import ta_wl_features as FW    # noqa: E402
_s = importlib.util.spec_from_file_location("oi_al", os.path.join(HERE, "OI_TS_AUDIT_2026-11-21_aligned.py"))
A = importlib.util.module_from_spec(_s); _s.loader.exec_module(A)
TB = B.TB
OUT = os.path.join(L.R, "reports", "OI_TS_AUDIT_2026-11-21_data"); os.makedirs(OUT, exist_ok=True)
M5, H4, AGE, TOL = 300_000, 14_400_000, 30 * 60_000, 15 * 60_000
THR, RSI_MIN, SSC_MIN, OIV_MIN = -0.01097, 74.07, -0.0366, 0.128
SIZE = 0.61                                 # 0.5 × 5× ≈ 0.61 × equity notional (DL 581 / 586) — ONE size for every fill
DESIGN_END = L.TS("2025-07-01")             # DIST_WL split


def one_pair(a):
    p, T, v24 = a
    n = len(T); out = {k: np.full(n, np.nan) for k in ("chg_ORIG", "chg_LIVE", "chg_NAIVE", "oiv_ORIG", "oiv_LIVE", "oiv_NAIVE")}
    stats = dict(pair=p, rows=0, meas=0, off5=0, off0=0, other=0)
    me = FW.metrics_frame(p)
    if me is None or len(me) < 60:
        return p, out, stats
    me = me.dropna(subset=["sum_open_interest", "sum_open_interest_value"]).sort_values("ct").reset_index(drop=True)
    x = TB.load(p)
    f = A.frame(me, x[0] if x is not None else None, x[4] if x is not None else None)
    stats.update(rows=len(f), meas=int(f.meas.sum()), off5=int((f.meas & (f.off == M5)).sum()), off0=int((f.meas & (f.off == 0)).sum()),
                 other=int((f.meas & ~f.off.isin([0, M5])).sum()))
    tp_c = me.sum_toptrader_long_short_ratio.values.astype(float); oi_c = me.sum_open_interest_value.values.astype(float)
    tp_a = f.sum_toptrader_long_short_ratio.values.astype(float); oi_a = f.sum_open_interest_value.values.astype(float)
    ct = me.ct.values.astype("int64"); sn = f.snap.values.astype("int64")
    for nm, ts, lag, tp, oi in (("ORIG", ct, M5, tp_c, oi_c), ("LIVE", sn, M5, tp_a, oi_a), ("NAIVE", ct, 0, tp_c, oi_c)):
        km = A.pick(ts, T, lag=lag, max_age=AGE + (0 if lag else M5))
        k4 = A.pick_back(ts, km, H4, TOL)
        ok = (km >= 0) & (k4 >= 0)
        out["chg_" + nm][ok] = tp[km[ok]] - tp[k4[ok]]
        ok1 = km >= 0
        out["oiv_" + nm][ok1] = oi[km[ok1]] / v24[ok1]
    return p, out, stats


def card(X, idx, label):
    v = L.net(X, idx, "LIVE"); n = len(v)
    if n == 0:
        return dict(label=label, N=0)
    d = L.dcr(X, idx, "LIVE", size=SIZE)
    T = X.entry_ms.values[idx]; ho = T >= DESIGN_END
    return dict(label=label, N=n, WR=(v > 0).mean() * 100, avg=v.mean(), t_cl=L.clustered_t(v, X.day.values[idx]),
                N_design=int((~ho).sum()), avg_design=v[~ho].mean() if (~ho).any() else np.nan,
                N_holdout=int(ho.sum()), avg_holdout=v[ho].mean() if ho.any() else np.nan,
                kelly=L.kelly_daily(v, X.day.values[idx]), dcr=d["dcr"], maxdd=d["maxdd"], worst_day=d["worst_day"],
                halves_pos=int(sum(1 for h, _, _ in L.HALVES if (X.half.values[idx] == h).any() and v[X.half.values[idx] == h].mean() > 0)),
                halves_n=int(sum(1 for h, _, _ in L.HALVES if (X.half.values[idx] == h).any())))


def boot_lo(v, day, q=0.05, B_=2000, seed=7):
    """one-sided lower bound of the mean by a day-block bootstrap."""
    rng = np.random.default_rng(seed); d = pd.Series(v).groupby(day).agg(["sum", "count"])
    s, c = d["sum"].values, d["count"].values; k = len(s)
    if k < 3:
        return np.nan
    ii = rng.integers(0, k, (B_, k))
    return np.quantile(s[ii].sum(1) / c[ii].sum(1), q)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=5); a = ap.parse_args()
    X = L.load()
    vsw = X.g_VSW15.values; exh = X.EXH.values
    rows = np.nonzero(vsw | exh)[0]
    fo = os.path.join(OUT, "feat.pkl")
    if not os.path.exists(fo):
        F = {k: np.full(len(X), np.nan) for k in ("chg_ORIG", "chg_LIVE", "chg_NAIVE", "oiv_ORIG", "oiv_LIVE", "oiv_NAIVE")}
        P = X.pair.values[rows].astype(str)
        jobs = [(p, X.entry_ms.values[rows[P == p]].astype("int64"), X.vol24.values[rows[P == p]].astype(float)) for p in np.unique(P)]
        pos = {p: rows[P == p] for p in np.unique(P)}
        ST = []
        with mp.get_context("fork").Pool(a.procs) as pool:
            for k, (p, out, st) in enumerate(pool.imap_unordered(one_pair, jobs, chunksize=4)):
                for c in F:
                    F[c][pos[p]] = out[c]
                ST.append(st)
                if k % 100 == 0:
                    print(time.strftime("%H:%M:%S"), k, len(jobs), flush=True)
        pd.to_pickle(F, fo); pd.DataFrame(ST).to_csv(os.path.join(OUT, "align_stats_by_pair.csv"), index=False)
    F = pd.read_pickle(fo)
    st = pd.read_csv(os.path.join(OUT, "align_stats_by_pair.csv"))
    print("alignment rows", st.rows.sum(), "measured", st.meas.sum(), "+5", st.off5.sum(), "+0", st.off0.sum(), "other", st.other.sum())

    # ── parity: ORIG must reproduce the pool's columns ─────────────────────────────────────────────────────────────────────────────
    par = []
    for c_pool, c_new in (("top_pos_chg4h", "chg_ORIG"), ("oi_to_vol24", "oiv_ORIG")):
        m = (vsw | exh) & np.isfinite(X[c_pool].values) & np.isfinite(F[c_new])
        par.append(dict(col=c_pool, n=int(m.sum()), match=float(np.mean(np.abs(F[c_new][m] - X[c_pool].values[m]) <= 1e-6 + 1e-4 * np.abs(X[c_pool].values[m]))),
                        pool_finite=int(np.isfinite(X[c_pool].values[vsw | exh]).sum()), new_finite=int(np.isfinite(F[c_new][vsw | exh]).sum())))
    print(pd.DataFrame(par).to_string())
    pd.DataFrame(par).to_csv(os.path.join(OUT, "parity.csv"), index=False)

    # ── DIST (DL 586) ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
    rsi = X.rsi1h.values
    C = []; SETS = {}
    for v in ("POOL", "ORIG", "LIVE", "NAIVE"):
        chg = X.top_pos_chg4h.values if v == "POOL" else F["chg_" + v]
        l1 = vsw & (chg <= THR)
        i_l1 = L.one_pos(X, np.nonzero(l1)[0], "LIVE")
        i_k = L.one_pos(X, np.nonzero(l1 & (rsi >= RSI_MIN))[0], "LIVE")
        i_b = L.one_pos(X, np.nonzero(l1 & ~(rsi >= RSI_MIN))[0], "LIVE")
        SETS[v] = (i_l1, i_k, i_b)
        C += [card(X, i_l1, f"DIST L1 (no RSI cut) · {v}"), card(X, i_k, f"DIST armed (L1 ∧ RSI1h ≥ 74.07) · {v}"),
              card(X, i_b, f"DIST RSI-blocked side · {v}")]
        # RSI1h HOLDOUT Δ (kept − blocked), the DIST_WL representative test
        ho_k = i_k[X.entry_ms.values[i_k] >= DESIGN_END]; ho_b = i_b[X.entry_ms.values[i_b] >= DESIGN_END]
        C[-2]["rsi_holdout_delta"] = L.net(X, ho_k, "LIVE").mean() - L.net(X, ho_b, "LIVE").mean()
    D = pd.DataFrame(C); D.to_csv(os.path.join(OUT, "dist_cards.csv"), index=False)
    pd.set_option("display.width", 260)
    print(D.round(3).to_string())
    # which armed fills change (ORIG → LIVE)
    ko, kl = set(SETS["ORIG"][1].tolist()), set(SETS["LIVE"][1].tolist())
    lost = np.array(sorted(ko - kl), dtype=int); new = np.array(sorted(kl - ko), dtype=int); both = np.array(sorted(ko & kl), dtype=int)
    ch = pd.DataFrame([dict(group=g, N=len(ii), avg=L.net(X, ii, "LIVE").mean() if len(ii) else np.nan,
                            WR=(L.net(X, ii, "LIVE") > 0).mean() * 100 if len(ii) else np.nan) for g, ii in (("both", both), ("ORIG only (lost)", lost), ("LIVE only (new)", new))])
    ch.to_csv(os.path.join(OUT, "dist_changed_fills.csv"), index=False); print(ch.round(3).to_string())
    # per-month alignment era split of the armed cohort
    era = np.where(X.entry_ms.values < A.SHIFT_FROM, "pre-2024-05", np.where(X.entry_ms.values < A.SHIFT_TO, "2024-05..2026-11-15", "2026-11-16+"))
    E = []
    for v in ("ORIG", "LIVE"):
        ii = SETS[v][1]
        for e in ("pre-2024-05", "2024-05..2026-11-15"):
            jj = ii[era[ii] == e]; w = L.net(X, jj, "LIVE")
            E.append(dict(read=v, era=e, N=len(jj), WR=(w > 0).mean() * 100 if len(jj) else np.nan, avg=w.mean() if len(jj) else np.nan))
    pd.DataFrame(E).to_csv(os.path.join(OUT, "dist_by_era.csv"), index=False); print(pd.DataFrame(E).round(3).to_string())
    m = vsw & np.isfinite(F["chg_ORIG"]) & np.isfinite(F["chg_LIVE"])
    dd = F["chg_LIVE"][m] - F["chg_ORIG"][m]
    sh = dict(n=int(m.sum()), mean_abs_diff=float(np.mean(np.abs(dd))), median_abs_diff=float(np.median(np.abs(dd))),
              sign_flip_at_thr=int(((F["chg_ORIG"][m] <= THR) != (F["chg_LIVE"][m] <= THR)).sum()),
              orig_fire=int((F["chg_ORIG"][m] <= THR).sum()), live_fire=int((F["chg_LIVE"][m] <= THR).sum()))
    print("chg4h shift", sh); pd.DataFrame([sh]).to_csv(os.path.join(OUT, "dist_feature_shift.csv"), index=False)

    # ── EXH_SSC_OIVOL (DL 582) ─────────────────────────────────────────────────────────────────────────────────────────────────────
    base = L.one_pos(X, np.nonzero(exh)[0], "LIVE")
    ssc = X.ssc.values
    O = []
    for v in ("POOL", "ORIG", "LIVE", "NAIVE"):
        oiv = X.oi_to_vol24.values if v == "POOL" else F["oiv_" + v]
        blk = (ssc > SSC_MIN) & (oiv > OIV_MIN)          # NaN OI → not blocked (fail-open)
        bset = base[blk[base]]
        kept = L.one_pos(X, np.nonzero(exh & ~blk)[0], "LIVE")
        cb = card(X, bset, f"EXH blocked (on base) · {v}"); w = L.net(X, bset, "LIVE")
        cb["lo95_mean"] = boot_lo(w, X.day.values[bset]) if len(w) else np.nan
        cb["hi95_mean"] = -boot_lo(-w, X.day.values[bset]) if len(w) else np.nan
        cb["n_days"] = len(np.unique(X.day.values[bset]))
        O += [card(X, base, f"EXH base (no filter) · {v}"), cb, card(X, kept, f"EXH filtered sleeve · {v}")]
    Oo = pd.DataFrame(O); Oo.to_csv(os.path.join(OUT, "oivol_cards.csv"), index=False); print(Oo.round(3).to_string())
    m = exh & np.isfinite(F["oiv_ORIG"]) & np.isfinite(F["oiv_LIVE"])
    fl = ((ssc > SSC_MIN) & (F["oiv_ORIG"] > OIV_MIN)) != ((ssc > SSC_MIN) & (F["oiv_LIVE"] > OIV_MIN))
    sh2 = dict(n=int(m.sum()), median_rel_diff=float(np.median(np.abs(F["oiv_LIVE"][m] / F["oiv_ORIG"][m] - 1))),
               block_flips=int((fl & m).sum()))
    print("oi_vol shift", sh2); pd.DataFrame([sh2]).to_csv(os.path.join(OUT, "oivol_feature_shift.csv"), index=False)
