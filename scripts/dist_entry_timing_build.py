#!/usr/bin/env python3
"""DIST_ENTRY_TIMING (PREREG reports/DIST_ENTRY_TIMING_2026-11-20_PREREG.md) — BUILD the early-trigger population. READ-ONLY on the bot.
  --stage legs    : every pool 15m row with full-candle vmult15 ≥ 5 (pool = reports/UNIFIED_SHORT_data/pool.pkl) → the exact pos_k / rsi_k
                    legs at T − k min (k = 0, 1, 2) from the builder's own 5m series + metrics (k = 0 must reproduce the pool) → legs.pkl
  --stage fetch   : 1m futures klines WITH quote volume (data.binance.vision daily zips, 0 API weight) for every signal day of a row that can
                    trigger at some k (pos_k ∧ rsi_k pass), + the next 2 days when no 1m file exists in the DIST_EXIT dirs → data/k1m/<P>/<day>.npz
  --stage partial : partial-candle legs at T − 1 / T − 2 from the 1m bars (red_k, vm_k, vol24_k, ATR_k, entry price) + the T0 check → cand.pkl
Usage: venv/bin/python -I scripts/dist_entry_timing_build.py --stage legs|fetch|partial [--procs 5]"""
import argparse, io, os, sys, time, zipfile
import concurrent.futures as cf
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import unified_short_lib as L   # noqa: E402
import unified_short_build as UB  # noqa: E402  (patches the k5m / metrics sources exactly like the builder)
import ta_wl_features as FW     # noqa: E402
import lead1_data as LD         # noqa: E402
TB = UB.TB
R = L.R
OUT = os.path.join(R, "reports", "DIST_ENTRY_TIMING_data"); os.makedirs(os.path.join(OUT, "k1m"), exist_ok=True)
K1OLD = [os.path.join(R, "reports", "EXH_EXIT_GRID_data", "k1m"), os.path.join(R, "reports", "NOSPOT_EXIT_data", "k1m"),
         os.path.join(R, "reports", "DIST_EXIT_data", "k1m")]
VIS = "https://data.binance.vision/data/futures/um/daily/klines"
M1, M5, M15, H1, H4, DAY = 60_000, 300_000, 900_000, 3_600_000, 14_400_000, 86_400_000
THR = -0.01097; RSI_MIN = 74.07103; VMULT = 5.0; MINVOL = 20e6
KS = (0, 1, 2)


# ------------------------------------------------------------------ stage legs
def legs_pair(a):
    p, T = a                                    # T = signal closes (ms) of the candidate rows
    n = len(T); out = {f"{nm}{k}": np.full(n, np.nan) for nm in ("pos", "rsi") for k in KS}
    out["vavg"] = np.full(n, np.nan); out["o15"] = np.full(n, np.nan)
    x = TB.load(p)
    if x is None:
        return p, T, out
    t, o, h, l, c, v, qv = x
    st15, en15, O15, H15, L15, C15, V15 = UB.agg(t, o, h, l, c, qv, M15)
    g15 = t[st15]
    vavg = TB.sh(pd.Series(V15).rolling(20).mean().values)
    k15 = np.searchsorted(g15, T - M15)          # the signal candle (opens at T − 15 min)
    okk = (k15 < len(g15)) & (g15[np.minimum(k15, len(g15) - 1)] == T - M15)
    out["vavg"][okk] = vavg[k15[okk]]; out["o15"][okk] = O15[k15[okk]]
    st1, en1, O1, H1a, L1_, C1, V1 = UB.agg(t, o, h, l, c, qv, H1)
    full1 = (en1 - st1) == 11
    R1 = TB.rsi(C1, 14)
    hend = t[st1] + H1                           # close time of each hour bar
    for k in KS:
        Tk = T - k * M1
        kk = np.searchsorted(hend, Tk, side="right") - 1      # last hour closed ≤ Tk
        ok = (kk >= 0) & full1[np.maximum(kk, 0)]
        out[f"rsi{k}"][ok] = R1[kk[ok]]
    me = FW.metrics_frame(p)
    if me is not None and len(me) > 60:
        ct = me.ct.values; tpl = me.sum_toptrader_long_short_ratio.values.astype(float)
        for k in KS:
            Tk = T - k * M1
            km = np.searchsorted(ct, Tk - M5, side="right") - 1
            fresh = (km >= 0) & (Tk - ct[np.maximum(km, 0)] <= 30 * 60_000)
            kc = ct[np.maximum(km, 0)]
            k4 = np.searchsorted(ct, kc - H4, side="right") - 1
            ok4 = fresh & (k4 >= 0) & (kc - ct[np.maximum(k4, 0)] <= H4 + 15 * 60_000)
            out[f"pos{k}"][ok4] = tpl[km[ok4]] - tpl[k4[ok4]]
    return p, T, out


def stage_legs(procs):
    X = L.load()
    m = X.is15.values & (X.vmult15.values >= VMULT)
    C = X.loc[m, ["pair", "entry_ms", "ep_on_ms", "spike_ms", "vmult15", "bar_ret_pct", "atr15_pct", "vol24", "q5", "top_pos_chg4h",
                  "rsi1h", "g_VSW15", "half", "day"]].copy()
    C["pair"] = C.pair.astype(str); C["pool_idx"] = np.nonzero(m)[0]; C = C.reset_index(drop=True)
    print("candidates", len(C), "pairs", C.pair.nunique(), flush=True)
    jobs = [(p, C.loc[C.pair == p, "entry_ms"].values.astype(np.int64)) for p in C.pair.unique()]
    res = {}
    with mp.get_context("fork").Pool(procs) as pool:
        for i, (p, T, out) in enumerate(pool.imap_unordered(legs_pair, jobs, chunksize=4)):
            res[p] = out
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), flush=True)
    for nm in list(next(iter(res.values())).keys()):
        C[nm] = np.nan
    for p, out in res.items():
        ix = C.index[C.pair == p]
        for nm, arr in out.items():
            C.loc[ix, nm] = arr
    C.to_pickle(os.path.join(OUT, "legs.pkl"))
    d0 = np.abs(C.pos0 - C.top_pos_chg4h); r0 = np.abs(C.rsi0 - C.rsi1h)
    print("parity pos0 vs pool: both finite", int((np.isfinite(C.pos0) & np.isfinite(C.top_pos_chg4h)).sum()),
          "max|d|", np.nanmax(d0), "nan-mismatch", int((np.isfinite(C.pos0) != np.isfinite(C.top_pos_chg4h)).sum()))
    print("parity rsi0 vs pool: max|d|", np.nanmax(r0), "nan-mismatch", int((np.isfinite(C.rsi0) != np.isfinite(C.rsi1h)).sum()))


# ------------------------------------------------------------------ stage fetch
def has_old(p, ds):
    return (os.path.exists(os.path.join(K1OLD[0], p, ds + ".csv")) or os.path.exists(os.path.join(K1OLD[1], p, ds + ".npz"))
            or os.path.exists(os.path.join(K1OLD[2], p, ds + ".npz")))


def fetch_one(a):
    p, d, needqv = a
    ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
    if not needqv and has_old(p, ds):
        return 0
    fo = os.path.join(OUT, "k1m", p, ds + ".npz"); fn = os.path.join(OUT, "k1m", p, ds + ".none")
    if os.path.exists(fo) or os.path.exists(fn):
        return 0
    os.makedirs(os.path.dirname(fo), exist_ok=True)
    z = LD.get(f"{VIS}/{p}/1m/{p}-1m-{ds}.zip")
    if not z:
        open(fn, "w").close(); return 0
    zz = zipfile.ZipFile(io.BytesIO(z)); lines = zz.open(zz.namelist()[0]).read().decode().strip().split("\n")
    if lines and not lines[0][0].isdigit():
        lines = lines[1:]
    a_ = np.array([ln.split(",")[:8] for ln in lines], dtype=float)
    t = a_[:, 0].astype(np.int64); t = np.where(t > 10**14, t // 1000, t)
    np.savez_compressed(fo, t=t, o=a_[:, 1], h=a_[:, 2], l=a_[:, 3], c=a_[:, 4], qv=a_[:, 7])
    return 1


def harmonise(C):
    """the metrics archive has grown since the pool was built: k = 0 uses the POOL's value (REF_B parity); k = 1, 2 are NaN wherever the pool's
    T0 value is NaN (same data availability as the study — no early trade the study's data could not have judged)."""
    C = C.copy(); nan0 = ~np.isfinite(C.top_pos_chg4h.values)
    C["pos0"] = C.top_pos_chg4h.values
    for k in (1, 2):
        C.loc[nan0, f"pos{k}"] = np.nan
    return C


def possible(C):
    ok = np.zeros(len(C), bool)
    for k in KS:
        ok |= (C[f"pos{k}"].values <= THR) & (C[f"rsi{k}"].values >= RSI_MIN)
    return ok


def stage_fetch():
    C = harmonise(pd.read_pickle(os.path.join(OUT, "legs.pkl")))
    P = C[possible(C)]
    last = pd.Timestamp("2026-11-18").value // 10**6 // DAY
    jobs = {}
    for p, T in zip(P.pair.values, P.entry_ms.values.astype(np.int64)):
        d0 = int(T - M1) // DAY
        jobs[(p, d0)] = True
        for kk in (1, 2):
            if d0 + kk < last:
                jobs.setdefault((p, d0 + kk), False)
    jl = [(p, d, q) for (p, d), q in sorted(jobs.items())]
    print("possible rows", len(P), "pair-days", len(jl), "with qv", sum(q for *_, q in jl), flush=True)
    n = 0; t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        for k, r in enumerate(ex.map(fetch_one, jl)):
            n += r
            if k % 500 == 0:
                print(time.strftime("%H:%M:%S"), k, len(jl), "fetched", n, f"{time.time() - t0:.0f}s", flush=True)
    print("done fetched", n, flush=True)


# ------------------------------------------------------------------ stage partial
def load_qv(p, d):
    ds = pd.to_datetime(d * DAY, unit="ms").strftime("%Y-%m-%d")
    f = os.path.join(OUT, "k1m", p, ds + ".npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {k: z[k] for k in z.files}


def partial_pair(a):
    p, rows = a
    n = len(rows)
    cols = ["red0", "vm0", "red1", "vm1", "red2", "vm2", "E0", "E1", "E2", "atr1", "atr2", "vol24_1", "vol24_2", "ok1m", "qv_ratio_5m"]
    out = {c: np.full(n, np.nan) for c in cols}
    x = TB.load(p)
    if x is None:
        return p, rows.index.values, out
    t, o, h, l, c, v, qv = x
    st15, en15, O15, H15, L15, C15, V15 = UB.agg(t, o, h, l, c, qv, M15)
    g15 = t[st15]
    cache = {}
    for i, (T, vavg, o15) in enumerate(zip(rows.entry_ms.values.astype(np.int64), rows.vavg.values, rows.o15.values)):
        d = int(T - M1) // DAY
        if d not in cache:
            cache[d] = load_qv(p, d)
        z = cache[d]
        if z is None:
            continue
        tt = z["t"]; a0 = int(np.searchsorted(tt, T - M15)); a1 = int(np.searchsorted(tt, T))
        if a1 - a0 != 15 or tt[a0] != T - M15:
            continue
        out["ok1m"][i] = 1
        k15 = int(np.searchsorted(g15, T - M15))
        if k15 >= len(g15) or g15[k15] != T - M15 or k15 < 110:
            continue
        q15 = z["qv"][a0:a1]; O = z["o"][a0]
        out["qv_ratio_5m"][i] = q15.sum() / V15[k15] if V15[k15] > 0 else np.nan
        # E_k = open of the 1m bar at T − k (index a1 − k); E0 = open of the bar at T (next day file if T is midnight)
        if a1 < len(tt) and tt[a1] == T:
            out["E0"][i] = z["o"][a1]
        else:
            z2 = load_qv(p, int(T) // DAY)
            if z2 is not None and len(z2["t"]) and z2["t"][0] == T:
                out["E0"][i] = z2["o"][0]
        out["red0"][i] = float(z["c"][a1 - 1] < O); out["vm0"][i] = q15.sum() / vavg if vavg > 0 else np.nan
        # the prior closed candles (for ATR / vol24 with the partial candle as the last bar)
        Hh, Ll, Cc, Vv = H15[:k15].copy(), L15[:k15].copy(), C15[:k15].copy(), V15[:k15].copy()
        for k in (1, 2):
            m = 15 - k
            qk = q15[:m].sum(); ck = z["c"][a0 + m - 1]
            hk = z["h"][a0:a0 + m].max(); lk = z["l"][a0:a0 + m].min()
            out[f"red{k}"][i] = float(ck < O)
            out[f"vm{k}"][i] = qk / vavg if vavg > 0 else np.nan
            out[f"E{k}"][i] = z["o"][a0 + m]
            A = TB.atr(np.r_[Hh[-200:], hk], np.r_[Ll[-200:], lk], np.r_[Cc[-200:], ck], 14)[-1]
            out[f"atr{k}"][i] = A / z["o"][a0 + m] * 100
            out[f"vol24_{k}"][i] = Vv[-95:].sum() + qk
    return p, rows.index.values, out


def stage_partial(procs):
    C = harmonise(pd.read_pickle(os.path.join(OUT, "legs.pkl")))
    C = C[possible(C)].copy()
    jobs = [(p, C[C.pair == p]) for p in C.pair.unique()]
    res = []
    with mp.get_context("fork").Pool(procs) as pool:
        for i, (p, ix, out) in enumerate(pool.imap_unordered(partial_pair, jobs, chunksize=4)):
            res.append((ix, out))
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(jobs), flush=True)
    for ix, out in res:
        for nm, arr in out.items():
            if nm not in C.columns:
                C[nm] = np.nan
            C.loc[ix, nm] = arr
    for k in KS:
        C[f"pr{k}"] = (C[f"pos{k}"] <= THR) & (C[f"rsi{k}"] >= RSI_MIN)
    C["trig0_pool"] = C.g_VSW15.values & (C.top_pos_chg4h.values <= THR) & (C.rsi1h.values >= RSI_MIN)
    C["trig0"] = (C.red0 == 1) & (C.vm0 >= VMULT) & C.pr0
    for k in (1, 2):
        C[f"trig{k}"] = (C[f"red{k}"] == 1) & (C[f"vm{k}"] >= VMULT) & C[f"pr{k}"] & (C[f"vol24_{k}"] >= MINVOL)
    C.to_pickle(os.path.join(OUT, "cand.pkl"))
    ok = C.ok1m == 1
    print("rows", len(C), "1m ok", int(ok.sum()), "qv 1m/5m ratio median", np.nanmedian(C.qv_ratio_5m), "p1/p99",
          np.nanpercentile(C.qv_ratio_5m, [1, 99]))
    print("T0 parity (1m vs pool): pool", int(C.trig0_pool.sum()), "1m", int(C.trig0.sum()), "both", int((C.trig0_pool & C.trig0).sum()),
          "pool&1m-missing", int((C.trig0_pool & ~ok).sum()))
    for k in (1, 2):
        print(f"trig{k}", int(C[f"trig{k}"].sum()), "of which trig0_pool", int((C[f"trig{k}"] & C.trig0_pool).sum()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", required=True); ap.add_argument("--procs", type=int, default=5)
    A = ap.parse_args()
    {"legs": lambda: stage_legs(A.procs), "fetch": stage_fetch, "partial": lambda: stage_partial(A.procs)}[A.stage]()
