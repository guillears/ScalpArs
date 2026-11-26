#!/usr/bin/env python3
"""EXH_EARLY_SPIKE_DEEP (2026-11-26, research only — READ-ONLY on the bot).
Operator ask: "study deeper" the theory that an EXHAUSTION_SHORT RSI turn in the first 30 min after the Frenzy spike is a pause
before the pump's second leg (new shorts fuel a squeeze), and show it on OUR real fills with today's filters.

PRIOR: reports/EXH_SPIKE_AGE_2026-11-26.md (buckets, < 1 h / < 0.5 h Pattern-C bar, DCR) — its loaders are re-used, not repeated.
NEW here:
  A) mechanism — from each fill's 1m path (5m fallback): new high above the pre-entry run peak (max 5m high from the spike bar to the
     signal bar = unified_short_build `peak`) before the exit / within 24 h, time to stop, MAE, second-leg extension, stop-then-TP-level.
  B) finer dose-response 0-10 / 10-20 / 20-30 / 30-45 / 45-60 min (+ ≥ 60), by half-year and DESIGN (< 2025-07-01) / HOLDOUT.
  C) inside the 0-0.5 h bucket: every entry variable at sign / median / tercile vs the rest of the bucket, within-month shuffle null
     (family-wise max |t|), theory-predicted directions reported separately.
  D) every real EXH fill (exports_archive + the newest export, dedup (opened_at, pair, direction), CLOSED, probes out), today's filters
     (no-spot, EXH_SSC_OIVOL), re-priced at today's exit on the real 1m path from the real fill price; parity at each fill's own TP/SL.
  E) Pattern-C bar + breakeven-bot read for the frozen 0-0.5 h zone.
  F) exhaustive 2D screen: hours axis (< 0.5 / ≥ 0.5, < 1 / ≥ 1, DESIGN terciles) × EVERY entry variable (sign / median / terciles),
     DESIGN selects (thresholds frozen on DESIGN), HOLDOUT confirms, within-month shuffle null on DESIGN (family-wise max |t|),
     day-clustered t for the reported cells, half-year consistency, survivors → concentration / DCR / Pattern-C / real-fill read.
Backtest = TP_GRID 2026-11-20 EXH cell F7 (today's exit, DL 609) via EXH_SPIKE_AGE.load_bt / kept (one position per coin).
Binance public fapi only for real fills not cached (weight header read, pause > 1,000 / min, stop on 418 / 429).
Usage: venv/bin/python -I scripts/EXH_EARLY_SPIKE_DEEP_2026-11-26.py [newest_export.csv]
Out: reports/EXH_EARLY_SPIKE_DEEP_2026-11-26.csv (+ _F2D.csv, _C.csv, .png), cache in reports/EXH_EARLY_SPIKE_DEEP_2026-11-26_data/."""
import glob, importlib.util, json, os, sys, time, warnings
warnings.filterwarnings("ignore")
import multiprocessing as mp
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R = os.path.dirname(HERE); RP = os.path.join(R, "reports")


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m


SA = _imp("exh_spike_age", "EXH_SPIKE_AGE_2026-11-26.py")
import exh_exit2_sim as XS      # noqa: E402  (1m loader; patches the 5m sources)
import entry_wl_lib as WL       # noqa: E402
TB = XS.TB
XS.K1_CSV.append(os.path.join(RP, "TP_GRID_2026-11-20_EXH_data", "k1m"))

OUT = os.path.join(RP, "EXH_EARLY_SPIKE_DEEP_2026-11-26"); DD = OUT + "_data"; os.makedirs(DD, exist_ok=True)
TPG_REAL = os.path.join(RP, "TP_GRID_2026-11-20_EXH_data", "real")
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-26_13-02-05.csv")
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
HOL = SA.HOL; SIZE = SA.SIZE
TAKER, MAKER, SLIP = 0.05, 0.02, 0.10
NSHUF = 300
RNG = np.random.default_rng(112601)
MINB = [(0, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 10**6)]
ML = lambda a, b: f"{a}-{b}m" if b < 10**6 else ">=60m"
LIVE_MAP = {"hrs_since_spike": "entry_frenzy_hours", "ssc": "entry_exh_spot_share_chg", "s_ssc": "entry_exh_spot_share_chg",
            "s_oi_vol": "entry_exh_oi_vol", "oi_to_vol24": "entry_exh_oi_vol", "vd": "entry_exh_vol_decay", "s_vd": "entry_exh_vol_decay",
            "atr15_pct": "entry_exh_atr_pct", "s_atrp": "entry_exh_atr_pct", "rsi15": "entry_exh_rsi", "ret1h_pct": "entry_exh_ret_1h_pct",
            "run_pct": "entry_frenzy_run_pct", "fund_last": "entry_funding_rate", "s_f_last": "entry_funding_rate"}


# ───────────────────────── backtest frame ─────────────────────────
def frame():
    D0, info = SA.load_bt()
    E = pd.read_pickle(os.path.join(RP, "ENTRY_WL_2026-11-18_data", "exh.pkl")).reset_index(drop=True)
    E = E.copy(); E["eidx"] = np.arange(len(E))
    drop = [c for c in E.columns if c in ("net", "k", "half", "day", "ep")]
    D = D0.merge(E.drop(columns=drop).rename(columns={"spike_ms": "spike_ms_e"}), on=["pair", "entry_ms"], how="left", validate="1:1")
    assert D.eidx.notna().all() and len(D) == len(D0)
    for c in [c for c in D.columns if c.startswith("g_") and c != "g_ANY"]:
        D[c] = D[c].astype(float)
    K = SA.kept(D).copy()
    K["mins"] = (K.hrs * 60).round().astype(int)
    K["mb"] = pd.cut(K.mins, [b[0] for b in MINB] + [MINB[-1][1]], right=False, labels=[ML(*b) for b in MINB]).astype(str)
    K["month"] = pd.to_datetime(K.entry_ms, unit="ms").dt.strftime("%Y-%m")
    K["hh"] = K.half.map(WL.HALF_MAP).fillna(K.half)
    K["tp"] = np.minimum(20 * K.s_atrp, 7.0); K["sl"] = np.minimum(1.5 * K.s_atrp, 6.0)
    K["win"] = K.net > 0
    return D, K, info


# ───────────────────────── A · mechanism from the path ─────────────────────────
def mech_pair(a):
    p, rows = a
    out = []
    five = TB.load(p)
    days = sorted({int(T) // DAY + k for T in rows.entry_ms.values for k in range(3)})
    one, okd = XS.load1m(p, days)
    for r in rows.itertuples():
        T = int(r.entry_ms); res = dict(key=r.key, src="none")
        pk = np.nan
        if five is not None:
            t5 = five[0]; i0 = np.searchsorted(t5, int(r.ep_on_ms)); i1 = np.searchsorted(t5, T)
            if i1 > i0:
                pk = float(five[2][i0:i1].max())
        use1 = one is not None and {T // DAY + k for k in range(3)} <= okd
        if use1:
            t, o, h, l, c = one; a0 = int(np.searchsorted(t, T)); use1 = a0 < len(t) and t[a0] == T; nb = 1
        if not use1:
            if five is None:
                out.append(res); continue
            t, o, h, l, c = five[:5]; a0 = int(np.searchsorted(t, T)); nb = 5
            if a0 >= len(t) or t[a0] != T:
                out.append(res); continue
        b24 = int(np.searchsorted(t, T + 24 * H1)); col = int(np.searchsorted(t, int(r.xm) - nb * MIN))
        col = min(max(col, a0), b24 - 1)
        E0 = o[a0]; hx = h[a0:col + 1]; h24 = h[a0:b24]; l24 = l[a0:b24]
        nh = np.nonzero(h24 > pk)[0] if np.isfinite(pk) else np.array([], int)
        tpP = E0 * (1 - r.tp / 100)
        res.update(src="1m" if nb == 1 else "5m", E0=E0, peak=pk, peak_dist=(pk / E0 - 1) * 100,
                   mae_exit=(hx.max() / E0 - 1) * 100, mfe_exit=(1 - l[a0:col + 1].min() / E0) * 100,
                   mae24=(h24.max() / E0 - 1) * 100, mfe24=(1 - l24.min() / E0) * 100,
                   nh_exit=bool(np.isfinite(pk) and hx.max() > pk), nh24=bool(len(nh)),
                   t_nh=(t[a0 + nh[0]] - T) / MIN if len(nh) else np.nan,
                   ext24=(h24.max() / pk - 1) * 100 if np.isfinite(pk) else np.nan,
                   min_held=(int(r.xm) - T) / MIN,
                   tp_after_stop=bool(r.k == 1 and (l[col + 1:b24] <= tpP).any()),
                   stop_ok=bool(hx.max() >= E0 * (1 + r.sl / 100) * (1 - 1e-9)) if r.k == 1 else np.nan,
                   tp_ok=bool(l[a0:col + 1].min() < tpP) if r.k == 0 else np.nan)
        out.append(res)
    return out


def mechanism(K):
    f = os.path.join(DD, "mech.pkl")
    if os.path.exists(f):
        M = pd.read_pickle(f)
        if set(M.key) >= set(K.key):
            return M
    jobs = [(p, g) for p, g in K[["key", "pair", "entry_ms", "xm", "k", "ep_on_ms", "tp", "sl"]].groupby("pair")]
    res = []
    with mp.get_context("fork").Pool(7) as pool:
        for i, o in enumerate(pool.imap_unordered(mech_pair, jobs, chunksize=2)):
            res += o
    M = pd.DataFrame(res); M.to_pickle(f); return M


# ───────────────────────── shared stats ─────────────────────────
def st(g):
    v = g.net.values
    if not len(v):
        return dict(N=0)
    return dict(N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum(), stop_pct=(g.k == 1).mean() * 100,
                tp_pct=(g.k == 0).mean() * 100, days=g.day.nunique())


def welch(sa, qa, na, sb, qb, nb):
    ma, mb = sa / na, sb / nb
    va = np.maximum(qa / na - ma ** 2, 1e-12) * na / np.maximum(na - 1, 1); vb = np.maximum(qb / nb - mb ** 2, 1e-12) * nb / np.maximum(nb - 1, 1)
    return (ma - mb) / np.sqrt(va / na + vb / nb), ma - mb


def day_cluster_t(y, inC, inP, day):
    """cell vs rest (scored) difference, day-clustered SE (sandwich on a 2-group regression)."""
    m = inP; y = y[m]; c = inC[m].astype(float); d = day[m]
    if c.sum() < 2 or (1 - c).sum() < 2:
        return np.nan
    ma, mb = y[c == 1].mean(), y[c == 0].mean(); e = y - np.where(c == 1, ma, mb)
    ga = pd.Series(e * c).groupby(d).sum().values / c.sum(); gb = pd.Series(e * (1 - c)).groupby(d).sum().values / (1 - c).sum()
    se = np.sqrt(((ga - gb) ** 2).sum())
    return (ma - mb) / se if se > 0 else np.nan


def shuffles(y, month, n):
    Y = np.empty((len(y), n)); idx = pd.Series(np.arange(len(y))).groupby(np.asarray(month)).indices
    for s in range(n):
        yy = y.copy()
        for ii in idx.values():
            yy[ii] = y[RNG.permutation(ii)]
        Y[:, s] = yy
    return Y


def var_groups(v, ref):
    """groups of a variable with thresholds frozen on `ref` (the selecting sample): sign / median / terciles. Returns {name: mask}."""
    g = {}; ok = np.isfinite(v); r = ref[np.isfinite(ref)]
    if len(r) < 30:
        return g, ok
    if set(np.unique(r)) <= {0.0, 1.0}:                      # binary trigger / state flag
        g["=1"] = ok & (v == 1); g["=0"] = ok & (v == 0)
        return g, ok
    if (r < 0).mean() > .05 and (r > 0).mean() > .05:
        g["<0"] = ok & (v < 0); g[">=0"] = ok & (v >= 0)
    md = np.median(r)
    lo, hi = ok & (v <= md), ok & (v > md)
    if 0.1 < (r <= md).mean() < 0.9:
        g[f"<={md:.4g}"] = lo; g[f">{md:.4g}"] = hi
    q1, q2 = np.quantile(r, [1 / 3, 2 / 3])
    if q1 < q2:
        g[f"T1<={q1:.4g}"] = ok & (v <= q1); g[f"T2"] = ok & (v > q1) & (v <= q2); g[f"T3>{q2:.4g}"] = ok & (v > q2)
    return g, ok


# ───────────────────────── C · inside 0-0.5 h ─────────────────────────
THEORY_C = {"dd_peak_pct": "+", "hrs_since_peak": "-", "vmult15": "+", "taker_1h": "+", "s_taker_1h": "+", "oi_chg15m": "+",
            "oi_chg30m": "+", "vd": "+", "s_vd": "+", "ret1h_pct": "+", "bar_ret_pct": "+", "close_pos15": "+", "last5_share": "+"}
# '+' = theory says HIGH values (near the peak / volume or buying still strong) are the BAD ones


def screen_C(Z, vars_):
    y = Z.net.values; tests = []
    for v in vars_:
        x = pd.to_numeric(Z[v], errors="coerce").values.astype(float)
        g, ok = var_groups(x, x)
        for nm, m in g.items():
            if m.sum() >= 15 and (ok & ~m).sum() >= 15:
                tests.append((v, nm, m, ok))
    C = np.array([t[2] for t in tests], float); P = np.array([t[3] for t in tests], float)
    def tstat(Y):
        sa, qa, na = C @ Y, C @ Y ** 2, C.sum(1)[:, None]
        sp, qp, npp = P @ Y, P @ Y ** 2, P.sum(1)[:, None]
        return welch(sa, qa, na, sp - sa, qp - qa, npp - na)
    t0, d0 = tstat(y[:, None]); t0, d0 = t0[:, 0], d0[:, 0]
    Ys = shuffles(y, Z.month.values, NSHUF); ts, _ = tstat(Ys)
    mx = np.nanmax(np.abs(ts), 0)
    rows = []
    for i, (v, nm, m, ok) in enumerate(tests):
        zc = Z[m]; zr = Z[ok & ~m]
        p_raw = (np.abs(ts[i]) >= abs(t0[i])).mean()
        p_fw = (mx >= abs(t0[i])).mean()
        dsg = {}
        for blk in ("DESIGN", "HOLDOUT"):
            a, b = zc[zc.blk == blk].net, zr[zr.blk == blk].net
            dsg[blk] = a.mean() - b.mean() if len(a) >= 5 and len(b) >= 5 else np.nan
        rows.append(dict(var=v, group=nm, theory=WL.theory_of(v), N=int(m.sum()), WR=(zc.net > 0).mean() * 100, avg=zc.net.mean(),
                         stop_pct=(zc.k == 1).mean() * 100, rest_N=len(zr), rest_avg=zr.net.mean(), delta=d0[i], t=t0[i],
                         t_day=day_cluster_t(y, m, ok, Z.day.values), p_raw=p_raw, p_fw=p_fw,
                         d_design=dsg["DESIGN"], d_holdout=dsg["HOLDOUT"], pred=THEORY_C.get(v, "")))
    return pd.DataFrame(rows).sort_values("p_raw"), len(tests), float(np.percentile(mx, 95))


# ───────────────────────── F · exhaustive 2D (hours × every variable) ─────────────────────────
def hours_groups(h, ref):
    q1, q2 = np.quantile(ref, [1 / 3, 2 / 3])
    return {"h<0.5": h < .5, "h>=0.5": h >= .5, "h<1": h < 1, "h>=1": h >= 1, f"hT1<={q1:.2f}": h <= q1,
            f"hT2": (h > q1) & (h <= q2), f"hT3>{q2:.2f}": h > q2}


def screen_F(K, vars_):
    Dz = K[K.blk == "DESIGN"].reset_index(drop=True); Hz = K[K.blk == "HOLDOUT"].reset_index(drop=True)
    hD, hH = hours_groups(Dz.hrs.values, Dz.hrs.values), hours_groups(Hz.hrs.values, Dz.hrs.values)
    tests = []; cov = []
    for v in vars_:
        if v in ("hrs_since_spike",):
            continue
        xd = pd.to_numeric(Dz[v], errors="coerce").values.astype(float); xh = pd.to_numeric(Hz[v], errors="coerce").values.astype(float)
        gD, okD = var_groups(xd, xd); gH, okH = var_groups(xh, xd)
        cov.append(dict(var=v, theory=WL.theory_of(v), cov_design=okD.mean() * 100, cov_holdout=okH.mean() * 100, n_groups=len(gD)))
        for gn in gD:
            for hn in hD:
                mD = gD[gn] & hD[hn]
                if mD.sum() >= 15 and (okD & ~mD).sum() >= 15:
                    tests.append((v, gn, hn, mD, okD, gH[gn] & hH[hn], okH))
    y = Dz.net.values
    C = np.array([t[3] for t in tests], np.float32); P = np.array([t[4] for t in tests], np.float32)

    def tstat(Y):
        Y = Y.astype(np.float32)
        sa, qa, na = C @ Y, C @ Y ** 2, C.sum(1)[:, None]
        sp, qp, npp = P @ Y, P @ Y ** 2, P.sum(1)[:, None]
        return welch(sa, qa, na, sp - sa, qp - qa, npp - na)
    t0, d0 = tstat(y[:, None]); t0, d0 = t0[:, 0], d0[:, 0]
    mx = []
    Ys = shuffles(y, Dz.month.values, NSHUF)
    for s0 in range(0, NSHUF, 50):
        ts, _ = tstat(Ys[:, s0:s0 + 50]); mx.append(np.nanmax(np.abs(ts), 0))
    mx = np.concatenate(mx)
    rows = []
    for i, (v, gn, hn, mD, okD, mH, okH) in enumerate(tests):
        zc = Dz[mD]; zr = Dz[okD & ~mD]; hc = Hz[mH]; hr = Hz[okH & ~mH]
        halves = K[np.r_[mD, mH]] if False else None
        rows.append(dict(var=v, vgroup=gn, hgroup=hn, theory=WL.theory_of(v), D_N=int(mD.sum()), D_WR=(zc.net > 0).mean() * 100,
                         D_avg=zc.net.mean(), D_stop=(zc.k == 1).mean() * 100, D_rest=zr.net.mean(), D_delta=d0[i], D_t=t0[i],
                         D_pfw=(mx >= abs(t0[i])).mean(),
                         H_N=int(mH.sum()), H_WR=(hc.net > 0).mean() * 100 if len(hc) else np.nan, H_avg=hc.net.mean() if len(hc) else np.nan,
                         H_stop=(hc.k == 1).mean() * 100 if len(hc) else np.nan, H_rest=hr.net.mean() if len(hr) else np.nan,
                         H_delta=(hc.net.mean() - hr.net.mean()) if len(hc) >= 2 and len(hr) >= 2 else np.nan, i=i))
    T = pd.DataFrame(rows)
    T["absD_t"] = T.D_t.abs(); T = T.sort_values("absD_t", ascending=False).reset_index(drop=True)
    return T, tests, Dz, Hz, pd.DataFrame(cov), float(np.percentile(mx, 95))


def F_followup(T, tests, Dz, Hz, K):
    """top-10 DESIGN cells: HOLDOUT t (day-clustered), half-year consistency."""
    out = []
    for r in T.head(10).itertuples():
        v, gn, hn, mD, okD, mH, okH = tests[r.i]
        tD = day_cluster_t(Dz.net.values, mD, okD, Dz.day.values); tH = day_cluster_t(Hz.net.values, mH, okH, Hz.day.values)
        A = pd.concat([Dz[mD].assign(z=1), Dz[okD & ~mD].assign(z=0), Hz[mH].assign(z=1), Hz[okH & ~mH].assign(z=0)])
        hv = A.groupby(["hh", "z"]).net.mean().unstack()
        same = int((np.sign(hv[1] - hv[0]) == np.sign(r.D_delta)).sum()) if 1 in hv and 0 in hv else 0
        out.append(dict(rank=len(out) + 1, cell=f"{hn} & {v} {gn}", D=f"{r.D_N}·{r.D_WR:.0f}%·{r.D_avg:+.2f} (rest {r.D_rest:+.2f})",
                        D_t_day=tD, D_pfw=r.D_pfw, H=f"{r.H_N}·{r.H_WR:.0f}%·{r.H_avg:+.2f} (rest {r.H_rest:+.2f})",
                        H_delta=r.H_delta, H_t_day=tH, halves_same_sign=f"{same}/{hv.dropna().shape[0]}",
                        survivor=bool(r.D_pfw < 0.05 and np.sign(r.H_delta) == np.sign(r.D_delta) and abs(tH) >= 1.96)))
    return pd.DataFrame(out)


# ───────────────────────── E · Pattern-C bar + DCR ─────────────────────────
def patternC(K, zone_mask, be, label):
    Zn = K[zone_mask]; v = Zn.net.values
    if len(v) < 2:
        return dict(zone=label, N=len(v))
    ub, lo, hi = SA.day_boot(v, Zn.day.values)
    neg = np.minimum(v, 0); tot = neg.sum()
    sh = lambda g: (pd.Series(neg).groupby(g).sum().min() / tot * 100) if tot < 0 else 0.0
    ok = dict(wr=(v > 0).mean() * 100 < be, avg=ub < 0, win=Zn.day.nunique() >= 8,
              conc=max(sh(Zn.day.values), sh(Zn.pair.values)) < 50, n=len(v) >= 15)
    return dict(zone=label, N=len(v), WR=(v > 0).mean() * 100, BE=be, avg=v.mean(), ub95=ub, ci_lo=lo, ci_hi=hi,
                windows=Zn.day.nunique(), top_day_share=sh(Zn.day.values), top_pair_share=sh(Zn.pair.values),
                verdict="PASS" if all(ok.values()) else "FAIL (" + ",".join(k for k, b in ok.items() if not b) + ")")


def dcr_rows(D, K, block_fn, label):
    d0, d1 = int(D.day.min()), int(max(D.xm.max() // DAY, D.day.max()))
    out = []
    Dm = D
    valid = np.ones(len(Dm), bool) if block_fn is None else ~block_fn(Dm)
    Kb = Dm[XS_onepos(Dm, valid)].reset_index(drop=True)
    for scope, a, b in (("ALL", d0, d1), ("DESIGN", d0, HOL // DAY - 1), ("HOLDOUT", HOL // DAY, d1)):
        Kc = Kb[(Kb.xm // DAY >= a) & (Kb.xm // DAY <= b)]
        p, _ = SA.path(Kc, a, b)
        out.append(dict(variant=label, scope=scope, N=len(Kc), avg=Kc.net.mean(), **p))
    return out


def XS_onepos(Dm, valid):
    return SA.EV.one_pos(Dm.pair.values, Dm.entry_ms.values, Dm.xm.values, valid)


# ───────────────────────── D · real fills ─────────────────────────
def fapi(path, params):
    import requests
    r = requests.get("https://fapi.binance.com" + path, params=params, headers={"User-Agent": "research-client/1.0"}, timeout=20)
    if r.status_code in (418, 429):
        sys.exit(f"STOP: HTTP {r.status_code} from Binance ({path})")
    r.raise_for_status()
    w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0")); print(f"  {path} {params.get('symbol')} weight {w}", flush=True)
    if w > 1000:
        time.sleep(61 - time.time() % 60)
    return r.json()


def kl(sym, t0, t1, iv):
    """[t0, t1) klines; cache in DD (partial windows are re-fetched when the window was still open at fetch time)."""
    stepms = {"1m": MIN, "5m": 5 * MIN}[iv]
    f = os.path.join(DD, f"{sym}_{t0}_{t1}_{iv}.csv")
    g = os.path.join(TPG_REAL, f"{sym}_{t0}_1m.csv")
    if iv == "1m" and os.path.exists(g) and not os.path.exists(f):
        x = pd.read_csv(g)
        if len(x) and x.t.max() >= t1 - 2 * MIN:
            return x
    if os.path.exists(f):
        return pd.read_csv(f)
    rows = []; s = t0
    while s < t1:
        js = fapi("/fapi/v1/klines", dict(symbol=sym, interval=iv, startTime=s, endTime=t1 - 1, limit=1000))
        if not js:
            break
        rows += [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in js]
        s = int(js[-1][0]) + stepms
        if len(js) < 1000:
            break
    x = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).drop_duplicates("t")
    if time.time() * 1000 > t1 + 5 * MIN:
        x.to_csv(f, index=False)
    return x


def fund(sym, T):
    f = os.path.join(DD, f"{sym}_{T}_funding.json"); g = os.path.join(TPG_REAL, f"{sym}_{T}_funding.json")
    for h in (f, g):
        if os.path.exists(h):
            return json.load(open(h))
    js = fapi("/fapi/v1/fundingRate", dict(symbol=sym, startTime=T, endTime=T + 25 * H1, limit=1000))
    out = [(int(j["fundingTime"]), float(j["fundingRate"])) for j in js]
    if time.time() * 1000 > T + 25 * H1:
        json.dump(out, open(f, "w"))
    return out


def walk(b, T, E0, tp, sl, tick, fd):
    S = E0 * (1 + sl / 100); P = E0 * (1 - tp / 100)
    kind, X, xm = "TIME", float(b.c.iloc[-1]), int(b.t.iloc[-1]) + MIN
    for t, o, h, l in zip(b.t.values, b.o.values, b.h.values, b.l.values):
        if h >= S:
            kind, X, xm = "SL", max(S, o), int(t) + MIN; break
        if l <= P - tick:
            kind, X, xm = "TP", P, int(t) + MIN; break
    g = -(X / E0 - 1) * 100; f = sum(r for ft, r in fd if T < ft <= xm) * 100
    return kind, X, xm, g, f, g + f - TAKER - (MAKER if kind == "TP" else TAKER + SLIP)


def real_fills():
    fs = sorted(glob.glob(os.path.join(R, "exports_archive", "*.csv"))) + [EXPORT]
    parts = []
    for f in fs:
        try:
            x = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "entry_strategy" not in x:
            continue
        e = x[(x.entry_strategy == "EXHAUSTION_SHORT") & (x.status == "CLOSED")]
        if len(e):
            parts.append(e.assign(_src=os.path.basename(f)))
    A = pd.concat(parts)
    if "is_probe" in A:
        A = A[~A.is_probe.fillna(False).astype(bool)]
    A = A.drop_duplicates(["opened_at", "pair", "direction"], keep="last").reset_index(drop=True)
    n_all = len(A)
    ssc = pd.to_numeric(A.entry_exh_spot_share_chg, errors="coerce"); oiv = pd.to_numeric(A.entry_exh_oi_vol, errors="coerce")
    A["blk_nospot"] = A.entry_exh_spot_symbol.isna() | (A.entry_exh_spot_symbol.astype(str).str.len() < 3)
    A["blk_ssc"] = (ssc > -0.0366) & (oiv > 0.128)
    rows, par = [], []
    for r in A.itertuples():
        T = int(pd.Timestamp(r.opened_at).value // 10**6); T0 = T // MIN * MIN; sym = r.pair; E0 = float(r.entry_price)
        x = kl(sym, T0, T0 + 24 * H1 + MIN, "1m"); fd = fund(sym, T)
        b = x[(x.t >= T0) & (x.t < T + 24 * H1)]
        u = np.unique(np.r_[b.o, b.h, b.l, b.c]); dd_ = np.diff(u); tick = float(dd_[dd_ > 1e-15].min()) if (dd_ > 1e-15).any() else 0.0
        notional = r.quantity * E0; booked = r.pnl / notional * 100
        bk = walk(b, T, E0, r.entry_exh_tp_pct, r.entry_exh_sl_pct, tick, fd)
        bkind = {"FRENZY_TP": "TP", "STOP_LOSS": "SL", "MAX_HOLD_TIME": "TIME"}.get(r.close_reason, r.close_reason)
        par.append(dict(pair=sym, era="TP7" if pd.Timestamp(r.opened_at) >= SA.TP7_DEPLOY else "TP10", booked_kind=bkind, model_kind=bk[0],
                        exit_px_err=(bk[1] / r.exit_price - 1) * 100, booked_net=booked, model_net=bk[5]))
        tp, sl = min(20 * r.entry_exh_atr_pct, 7.0), min(1.5 * r.entry_exh_atr_pct, 6.0)
        kind, X, xm, g, f, net = walk(b, T, E0, tp, sl, tick, fd)
        sp = pd.Timestamp(r.entry_frenzy_spike_at).value // 10**6 if isinstance(r.entry_frenzy_spike_at, str) else T - r.entry_frenzy_hours * H1
        s5 = int(sp) // (5 * MIN) * (5 * MIN) - 5 * MIN
        x5 = kl(sym, s5, T0 // (5 * MIN) * (5 * MIN), "5m")
        pk = float(x5.h.max()) if len(x5) else np.nan
        bx = b[b.t < xm]
        rows.append(dict(pair=sym, T=T, day=T // DAY, hrs=float(r.entry_frenzy_hours), blk_nospot=r.blk_nospot, blk_ssc=r.blk_ssc,
                         booked_net=booked, booked_kind=bkind, kind=kind, net=net, fund=f, xm=xm, k={"TP": 0, "SL": 1, "TIME": 2}[kind],
                         peak_dist=(pk / E0 - 1) * 100, nh_exit=bool(len(bx) and bx.h.max() > pk), nh24=bool(b.h.max() > pk),
                         mae_exit=(bx.h.max() / E0 - 1) * 100 if len(bx) else np.nan, min_held=(xm - T) / MIN,
                         window_complete=bool(b.t.max() >= T + 24 * H1 - 2 * MIN) or kind != "TIME",
                         **{"v_" + k: pd.to_numeric(getattr(r, c), errors="coerce") for k, c in LIVE_MAP.items() if hasattr(r, c)}))
    return pd.DataFrame(rows), pd.DataFrame(par), n_all


# ───────────────────────── chart ─────────────────────────
def chart(K, M):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    P = np.load(os.path.join(RP, "ENTRY_WL_2026-11-18_data", "exh_path.npy"))
    x = (np.arange(P.shape[1]) - 24) * 5 / 60
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = {"0-0.5 h · winners": ("#eb6834", "-"), "0-0.5 h · losers": ("#eb6834", "--"),
            "≥ 0.5 h · winners": ("#2a78d6", "-"), "≥ 0.5 h · losers": ("#2a78d6", "--")}
    early = K.hrs < .5
    for nm, m in (("0-0.5 h · winners", early & K.win), ("0-0.5 h · losers", early & ~K.win),
                  ("≥ 0.5 h · winners", ~early & K.win), ("≥ 0.5 h · losers", ~early & ~K.win)):
        pp = -P[K.eidx[m].astype(int).values]          # price vs entry, % (pump = up)
        c, ls = cols[nm]; ax[0].plot(x, np.nanmean(pp, 0), color=c, ls=ls, lw=2, label=f"{nm} (N={int(m.sum())})")
    ax[0].axvline(0, color="#52514e", lw=1); ax[0].axhline(0, color="#52514e", lw=.6)
    ax[0].set_xlim(-2, 12); ax[0].set_xlabel("hours from entry"); ax[0].set_ylabel("price vs entry, % (5m close, mean)")
    ax[0].set_title("Backtest EXH at today's exit: mean price path"); ax[0].legend(frameon=False, fontsize=8)
    J = K.merge(M, on="key")
    grid = np.arange(0, 24 * 60 + 1, 5)
    for nm, m, c in (("0-0.5 h", J.hrs < .5, "#eb6834"), ("0.5-1 h", (J.hrs >= .5) & (J.hrs < 1), "#1baf7a"), ("≥ 1 h", J.hrs >= 1, "#2a78d6")):
        tn = J.t_nh[m].values
        ax[1].plot(grid / 60, [(tn <= g).mean() * 100 for g in grid], color=c, lw=2, label=f"{nm} (N={int(m.sum())})")
    ax[1].set_xlabel("hours from entry"); ax[1].set_ylabel("% of fills above the pre-entry run peak (cumulative)")
    ax[1].set_title("Second leg: share making a NEW HIGH after entry"); ax[1].legend(frameon=False, fontsize=8); ax[1].set_xlim(0, 24)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT + ".png", dpi=130)


# ───────────────────────── main ─────────────────────────
def main():
    pd.set_option("display.width", 260); pd.set_option("display.max_rows", 500); pd.set_option("display.max_columns", 40)
    D, K, info = frame()
    K["key"] = K.pair + ":" + K.entry_ms.astype(str)
    w, l = K.net[K.net > 0], K.net[K.net <= 0]; be = -l.mean() / (w.mean() - l.mean()) * 100
    print(info, "kept", len(K), "BE", round(be, 1))
    rows = []
    # A
    M = mechanism(K); J = K.merge(M, on="key")
    print("A src", J.src.value_counts().to_dict(), "stop parity", J.stop_ok.mean(), "tp parity", J.tp_ok.mean(), "peak known", J.peak.notna().mean())
    J["hb"] = np.where(J.hrs < .5, "0-0.5h", np.where(J.hrs < 1, "0.5-1h", np.where(J.hrs < 4, "1-4h", np.where(J.hrs < 24, "4-24h", "24-72h"))))
    for scope, Jm in (("ALL", J), ("DESIGN", J[J.blk == "DESIGN"]), ("HOLDOUT", J[J.blk == "HOLDOUT"])):
        for hb in ("0-0.5h", "0.5-1h", "1-4h", "4-24h", "24-72h"):
            g = Jm[Jm.hb == hb]; s = g[g.k == 1]
            rows.append(dict(table="A", scope=scope, cohort=hb, N=len(g), stop_pct=(g.k == 1).mean() * 100, nh_exit=g.nh_exit.mean() * 100,
                             nh24=g.nh24.mean() * 100, stops_nh=s.nh_exit.mean() * 100, stop_lt_peak=(g.sl < g.peak_dist).mean() * 100,
                             med_peak_dist=g.peak_dist.median(), med_min_to_stop=s.min_held.median(),
                             stops_in_60m=(s.min_held <= 60).mean() * 100, mae_stops=s.mae_exit.median(),
                             mae_all_med=g.mae24.median(), ext24_nh_med=g.ext24[g.nh24].median(), t_nh_med=g.t_nh.median(),
                             stop_then_tp=s.tp_after_stop.mean() * 100, avg=g.net.mean()))
    # B
    for scope, Km in [("ALL", K), ("DESIGN", K[K.blk == "DESIGN"]), ("HOLDOUT", K[K.blk == "HOLDOUT"])] + [(h, K[K.hh == h]) for h in WL.HALVES]:
        for mb in [ML(*b) for b in MINB]:
            rows.append(dict(table="B", scope=scope, cohort=mb, **st(Km[Km.mb == mb])))
    # C
    vars_ = [v for v in WL.variables(K) if v not in ("hrs", "mins", "net", "k", "xm", "eidx", "tp", "sl", "win", "spike_ms_e", "entry_ms",
                                                     "spike_ms", "atrp", "q5_row", "slip")]
    Z = K[K.hrs < .5].reset_index(drop=True)
    Cres, nC, mx95C = screen_C(Z, vars_)
    Cres.to_csv(OUT + "_C.csv", index=False)
    # E
    PC = [patternC(K, K.hrs < .5, be, "0-0.5h ALL"), patternC(K, (K.hrs < .5) & (K.blk == "DESIGN"), be, "0-0.5h DESIGN"),
          patternC(K, (K.hrs < .5) & (K.blk == "HOLDOUT"), be, "0-0.5h HOLDOUT")]
    for b0 in (10 / 60, 20 / 60):
        PC.append(patternC(K, K.hrs < b0, be, f"<{round(b0 * 60)}m ALL (post-hoc)"))
    DC = dcr_rows(D, K, None, "today") + dcr_rows(D, K, lambda x: x.hrs < .5, "block <0.5h")
    # F
    T, tests, Dz, Hz, COV, mx95F = screen_F(K, vars_)
    T.drop(columns=["i"]).to_csv(OUT + "_F2D.csv", index=False)
    TOP = F_followup(T, tests, Dz, Hz, K)
    COV.to_csv(OUT + "_F_coverage.csv", index=False)
    # heat: minute bucket × terciles (DESIGN thresholds) of the best-ranked variable
    best = T.iloc[0]["var"]
    xd = pd.to_numeric(Dz[best], errors="coerce"); q1, q2 = np.nanquantile(xd, [1 / 3, 2 / 3])
    xv = pd.to_numeric(K[best], errors="coerce")
    K["_bt"] = np.where(xv.isna(), "unscored", np.where(xv <= q1, f"T1<={q1:.3g}", np.where(xv <= q2, "T2", f"T3>{q2:.3g}")))
    HEAT = K.groupby(["mb", "_bt"]).apply(lambda g: f"{len(g)}·{(g.net > 0).mean() * 100:.0f}%·{g.net.mean():+.2f}").unstack()
    HEAT = HEAT.reindex([ML(*b) for b in MINB])
    # survivors follow-up
    SURV = []
    for r in TOP[TOP.survivor].itertuples():
        i = int(T.iloc[r.rank - 1].i); v, gn, hn, *_ = tests[i]
        xdv = pd.to_numeric(Dz[v], errors="coerce").values.astype(float)
        def zone(df, v=v, gn=gn, hn=hn, xdv=xdv):
            g, _ = var_groups(pd.to_numeric(df[v], errors="coerce").values.astype(float), xdv)
            return g[gn] & hours_groups(df.hrs.values, Dz.hrs.values)[hn]
        SURV.append(dict(cell=r.cell, pc=patternC(K, zone(K), be, r.cell), dcr=dcr_rows(D, K, lambda x: zone(x), "block " + r.cell)))
    # D
    RF, PAR, n_all = real_fills()
    RF["kept"] = ~(RF.blk_nospot | RF.blk_ssc)
    KR = RF[RF.kept]
    for era, P_ in (("ALL", PAR), ("TP7", PAR[PAR.era == "TP7"]), ("TP10", PAR[PAR.era == "TP10"])):
        rows.append(dict(table="D_parity", scope=era, N=len(P_), kind_match=(P_.booked_kind == P_.model_kind).mean() * 100,
                         med_abs_px_err=P_.exit_px_err.abs().median(), booked_avg=P_.booked_net.mean(), model_avg=P_.model_net.mean()))
    for nm, m in (("0-0.5h", KR.hrs < .5), ("0.5-1h", (KR.hrs >= .5) & (KR.hrs < 1)), ("1-4h", (KR.hrs >= 1) & (KR.hrs < 4)),
                  ("4-24h", (KR.hrs >= 4) & (KR.hrs < 24)), ("24-72h", KR.hrs >= 24), ("all", KR.hrs >= 0)):
        g = KR[m]
        rows.append(dict(table="D", scope="kept", cohort=nm, N=len(g), days=g.day.nunique(), WR=(g.net > 0).mean() * 100 if len(g) else np.nan,
                         avg=g.net.mean(), booked_avg=g.booked_net.mean(), stop_pct=(g.k == 1).mean() * 100 if len(g) else np.nan,
                         nh_exit=g.nh_exit.mean() * 100 if len(g) else np.nan, nh24=g.nh24.mean() * 100 if len(g) else np.nan,
                         stops_nh=g[g.k == 1].nh_exit.mean() * 100 if (g.k == 1).any() else np.nan, med_peak_dist=g.peak_dist.median()))
    d0 = int(RF.day.min()); d1 = int(pd.Timestamp(os.path.basename(EXPORT)[22:32]).value // 10**6 // DAY) if "paper_" in EXPORT else int(RF.day.max())
    for nm, Kc in (("today (kept)", KR), ("block <0.5h", KR[KR.hrs >= .5])):
        r_d = pd.Series(Kc.net.values * SIZE / 100).groupby(Kc.xm.values // DAY).sum().reindex(range(d0, d1 + 1), fill_value=0).values
        eq = np.cumprod(1 + r_d)
        rows.append(dict(table="D_dcr", cohort=nm, N=len(Kc), days=len(r_d), DCR=(eq[-1] ** (1 / len(r_d)) - 1) * 100,
                         maxDD=(eq / np.maximum.accumulate(np.r_[1, eq])[1:] - 1).min() * 100, worst_day=r_d.min() * 100))
    S = pd.DataFrame(rows); S.to_csv(OUT + ".csv", index=False)
    # ---- print
    for t in ("A", "B"):
        print(f"\n== {t}"); print(S[S.table == t].dropna(axis=1, how="all").round(2).to_string(index=False))
    print(f"\n== C  tests {nC}  null 95% family max |t| {mx95C:.2f}")
    print(Cres.head(15).round(3).to_string(index=False))
    print("\n-- C theory-predicted variables (top/bottom tercile)")
    print(Cres[Cres.pred != ""].sort_values(["var", "group"]).round(3).to_string(index=False))
    print("\n== E Pattern-C"); print(pd.DataFrame(PC).round(2).to_string(index=False))
    print(pd.DataFrame(DC).round(3).to_string(index=False))
    print(f"\n== F tests {len(tests)}  vars {COV.shape[0]}  null 95% family max |t| {mx95F:.2f}")
    print("coverage <90% design:", COV[COV.cov_design < 90][["var", "cov_design", "cov_holdout"]].round(0).to_string(index=False))
    print("vars with no groups:", COV[COV.n_groups == 0]["var"].tolist())
    print(T.head(10).drop(columns=["i", "absD_t"]).round(3).to_string(index=False))
    print(TOP.round(3).to_string(index=False))
    print("\nHEAT best var", best); print(HEAT.to_string())
    for s_ in SURV:
        print("\nSURVIVOR", s_["cell"]); print(s_["pc"]); print(pd.DataFrame(s_["dcr"]).round(3).to_string(index=False))
    print(f"\n== D real: {n_all} closed EXH fills, kept {len(KR)} (nospot {int(RF.blk_nospot.sum())}, ssc_oivol {int(RF.blk_ssc.sum())})")
    print(S[S.table.str.startswith("D")].dropna(axis=1, how="all").round(2).to_string(index=False))
    for v in [c for c in RF.columns if c.startswith("v_")]:
        pass
    RF.drop(columns=[c for c in RF.columns if c.startswith("v_")]).to_pickle(os.path.join(DD, "real.pkl"))
    RF.to_pickle(os.path.join(DD, "real_full.pkl")); TOP.to_csv(OUT + "_F_top10.csv", index=False); HEAT.to_csv(OUT + "_F_heat.csv")
    pd.DataFrame(PC).to_csv(OUT + "_E.csv", index=False); pd.DataFrame(DC).to_csv(OUT + "_E_dcr.csv", index=False)
    json.dump([dict(cell=s_["cell"], pc={k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in s_["pc"].items()},
                    dcr=s_["dcr"]) for s_ in SURV], open(OUT + "_F_survivors.json", "w"), default=float, indent=1)
    chart(K, M)
    print("done")


if __name__ == "__main__":
    main()
