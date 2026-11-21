#!/usr/bin/env python3
"""HT_THEORY_EXIT 2026-11-21 — SIM (PREREG reports/HT_THEORY_EXIT_2026-11-21_PREREG.md). Research only; no bot code / config touched.
Per fill (REAL HT / REAL WILLY from HT_THEORY_EXIT_2026-11-21_data/real_cohort.csv; BACKTEST HT_WILLY_2026-11-20_cohort_{HT,WILLY}.csv):
REF (today's exit) + every pre-registered cell + fixed-TP comparisons, at slip 0.10 and 0.20 (stress).
1m data, read-only sources first: STATE_EXIT_2026-11-21_data/k1m (npz with volume) → fast_tp_data.get1m; else own vision / fapi fetch into
HT_THEORY_EXIT_2026-11-21_data/k1m (days ≥ 2026-11-19 always fetched fresh for real fills).
→ HT_THEORY_EXIT_2026-11-21_data/sim.pkl (one row per fill: anchors, REF, per-cell net / kind / exit ms, event-study path)
Usage: venv/bin/python -I scripts/HT_THEORY_EXIT_2026-11-21_sim.py [--procs 8] [--n 0]"""
import argparse, importlib.util, math, os, sys, time, json
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
_sp = importlib.util.spec_from_file_location("SEC", os.path.join(HERE, "STATE_EXIT_2026-11-21_common.py"))
SEC = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(SEC)
import fast_tp_sim as S    # noqa: E402  willy_ref, price_for_net, net_taker_long, tick_for, fund_short, W_CUR
import fast_tp_data as FD  # noqa: E402
D = os.path.join(ROOT, "reports", "HT_THEORY_EXIT_2026-11-21_data")
OWN_K1 = os.path.join(D, "k1m"); OWN_F = os.path.join(D, "fund")
for _d in (OWN_K1, OWN_F):
    os.makedirs(_d, exist_ok=True)
SE_K1 = os.path.join(ROOT, "reports", "STATE_EXIT_2026-11-21_data", "k1m")
MIN, HR, DAY = 60_000, 3_600_000, 86_400_000
TAKER, MAKER = 0.05, 0.02
CUR_BATCH = "2026-11-17T17:00"
FRESH = "2026-11-19"
SPLIT = pd.Timestamp("2025-07-01").value // 10**6
HT_CELLS = ["A_H0", "A_Hm05", "A_Hp05", "A_H75", "N_H0", "N_Hm05"] + [f"B_X{x}_N{n}" for x in (2, 3) for n in (30, 60, 120)] + \
           ["C_DIPLOW", "C_VWAP5", "C_BASE", "S3", "S5", "S6", "D_HLOCK0", "D_HLOCK1", "E_H0_5", "E_Hm05_5", "E_H0_LOCK"]
HT_FIXED = [f"TP{x:g}" for x in (2, 2.5, 3, 3.5, 4, 4.5, 6)]
W_CELLS = ["WA_H0", "WA_Hp05", "WE_HALF"] + [f"WB_X{x}_N{n}" for x in (1, 2) for n in (15, 30)]
W_FIXED = ["WTP1.25", "WTP2", "WTP3"]


# ───────────────────────────── data ─────────────────────────────
def _npz(f):
    if not os.path.exists(f):
        return None
    z = np.load(f)
    if not len(z["t"]):
        return None
    return {c: z[c].astype(float) if c != "t" else z[c].astype("int64") for c in ("t", "o", "h", "l", "c", "qv")}


def own_fetch(p, ds):
    old = SEC.K1
    SEC.K1 = OWN_K1
    try:
        SEC.fetch_kday(p, ds)
    finally:
        SEC.K1 = old


def day(p, ds, fresh):
    if fresh and ds >= FRESH:
        f = os.path.join(OWN_K1, p, ds + ".npz")
        if not os.path.exists(f):
            own_fetch(p, ds)
        return _npz(f), True
    for base in (SE_K1, OWN_K1):
        a = _npz(os.path.join(base, p, ds + ".npz"))
        if a is not None:
            return a, True
    a = FD.day_arr(p, ds)
    if a is not None and len(a):
        return {"t": a[:, 0].astype("int64"), "o": a[:, 1], "h": a[:, 2], "l": a[:, 3], "c": a[:, 4], "qv": None}, False
    f = os.path.join(OWN_K1, p, ds + ".npz")
    if not os.path.exists(f):
        own_fetch(p, ds)
    return _npz(f), True


def bars(p, t0, t1, fresh=False):
    parts, vol_ok = [], True
    for ds in SEC.days_of(t0, t1):
        a, v = day(p, ds, fresh)
        if a is None:
            continue
        if not v or a["qv"] is None:
            vol_ok = False
            a = dict(a, qv=np.full(len(a["t"]), np.nan))
        parts.append(a)
    if not parts:
        return None
    out = {c: np.concatenate([x[c] for x in parts]) for c in ("t", "o", "h", "l", "c", "qv")}
    m = (out["t"] >= t0) & (out["t"] < t1)
    out = {c: v[m] for c, v in out.items()}
    _, ix = np.unique(out["t"], return_index=True)
    out = {c: v[ix] for c, v in out.items()}
    out["vol_ok"] = vol_ok
    return out


_FUN = {}


def real_funding(p):
    """settlements (t, rate) for real fills: own fapi fetch from 2026-09-01 (paged), cached in OWN_F."""
    if p in _FUN:
        return _FUN[p]
    f = os.path.join(OWN_F, p + ".csv")
    if not os.path.exists(f):
        rows, t = [], int(pd.Timestamp("2026-09-01").value // 10**6)
        while True:
            x = SEC._fapi(f"/fapi/v1/fundingRate?symbol={p}&startTime={t}&limit=1000") or []
            rows += [(int(r["fundingTime"]), float(r["fundingRate"])) for r in x]
            if len(x) < 1000:
                break
            t = int(x[-1]["fundingTime"]) + 1
        pd.DataFrame(rows, columns=["t", "rate"]).to_csv(f, index=False)
    x = pd.read_csv(f).drop_duplicates("t").sort_values("t")
    _FUN[p] = (x.t.values.astype("int64"), x.rate.values.astype(float))
    return _FUN[p]


def fund_long(fl, t_exit):
    """funding % to a LONG over (te, t_exit]: long pays a positive rate."""
    if fl["kind"] == "BT":
        return -float(S.fund_short(fl["pair"], int(fl["te"]), int(t_exit)))
    ft, fr = real_funding(fl["pair"])
    m = (ft > fl["te"]) & (ft <= t_exit)
    return float(-fr[m].sum() * 100)


# ───────────────────────────── HT engine ─────────────────────────────
def lp_net(E, net, fee, tick):
    return math.ceil(S.price_for_net(E, net, fee) / tick - 1e-9) * tick


def stop_px(E, sl):
    return E * (100 - sl + TAKER) / (100 - TAKER)


def run_ht(g, i0, jend, E, tick, tp_px, sl_px, slip, entry_cost, lock=None, k_over=None, has_end=True):
    """first event from bar i0: overlay market exit at the OPEN of k_over (prio 0) · lock stop (arm bar → line from the next bar, prio 1)
    · stop (prio 2) · maker TP ≥ 1 tick through (prio 3) · time at the open of jend. → (gross-net before funding, exit bar, kind)"""
    o, h, l = g["o"], g["h"], g["l"]
    n = jend - i0
    if n <= 0:
        return None
    cands = []
    hs = np.flatnonzero(l[i0:jend] <= sl_px)
    if len(hs):
        cands.append((hs[0], 2, "STOP"))
    if tp_px is not None:
        ht = np.flatnonzero(h[i0:jend] >= tp_px + tick - 1e-12)
        if len(ht):
            cands.append((ht[0], 3, "TP"))
    if lock is not None:
        arm_px, lock_px = lock
        ka = np.flatnonzero(h[i0:jend] >= arm_px)
        if len(ka):
            kl = np.flatnonzero(l[i0 + ka[0] + 1:jend] <= lock_px)
            if len(kl):
                cands.append((ka[0] + 1 + kl[0], 1, "LOCK"))
    if k_over is not None and i0 <= k_over < jend:
        cands.append((k_over - i0, 0, "RULE"))
    if cands:
        k, _, why = min(cands)
        j = i0 + k
        if why == "TP":
            X = tp_px; fee = MAKER
        else:
            lvl = {"STOP": sl_px, "LOCK": lock[1] if lock else None, "RULE": None}[why]
            X = (o[j] if lvl is None else min(lvl, o[j])) * (1 - slip / 100); fee = TAKER
        return (X / E - 1) * 100 - entry_cost - fee * X / E, j, why
    if has_end and jend < len(o):
        X = o[jend] * (1 - slip / 100)
        return (X / E - 1) * 100 - entry_cost - TAKER * X / E, jend, "TIME"
    j = jend - 1
    X = g["c"][j] * (1 - slip / 100)
    return (X / E - 1) * 100 - entry_cost - TAKER * X / E, j, "OPEN"


def stall_k(g, i0, jend, E, X, N, slip, entry_cost):
    h, c = g["h"][i0:jend], g["c"][i0:jend]
    if not len(h):
        return None
    run = np.maximum.accumulate(h)
    newhi = np.r_[True, h[1:] > run[:-1]]
    last = np.maximum.accumulate(np.where(newhi, np.arange(len(h)), 0))
    pk = (run / E - 1) * 100
    xn = c * (1 - slip / 100)
    now = (xn / E - 1) * 100 - entry_cost - TAKER * xn / E
    ok = (pk >= X) & (np.arange(len(h)) - last >= N) & (now > 0)
    f = np.flatnonzero(ok)
    return i0 + f[0] + 1 if len(f) else None


def vwap_k(g, i0, jend, isp):
    """first in-trade closed 5m bar (clock) whose close < VWAP × 0.995 → exit at the next 1m open."""
    if g["qv"] is None or not np.isfinite(g["qv"][isp:jend]).all():
        return None, np.nan
    t, c = g["t"], g["c"]
    tp = (g["h"] + g["l"] + c) / 3
    w = np.where(np.arange(len(t)) >= isp, g["qv"], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        vw = np.cumsum(tp * w) / np.cumsum(w)
    ks = np.arange(i0, jend)
    is5 = ((t[ks] + MIN) % (5 * MIN)) == 0
    hit = ks[is5 & (c[ks] < vw[ks] * 0.995)]
    return (int(hit[0]) + 1 if len(hit) else None), float(vw[i0 - 1] if i0 > 0 else np.nan)


def sim_ht(fl, slip):
    p, te, E = fl["pair"], int(fl["te"]), float(fl["E"])
    sp = int(fl["spike_ms"]) if fl["spike_ms"] == fl["spike_ms"] else te - HR
    a = (min(te, sp) - 60 * MIN) // MIN * MIN
    g = bars(p, a, te + 25 * HR, fresh=(fl["kind"] == "REAL"))
    if g is None:
        return None, "no data"
    t = g["t"]
    T0 = (te // MIN) * MIN + (MIN if fl["start"] == "next" else 0)
    i0 = int(np.searchsorted(t, T0))
    if i0 >= len(t) or t[i0] - T0 > 5 * MIN:
        return None, "no entry bar"
    t_end = T0 + 24 * HR
    jend = int(np.searchsorted(t, t_end, "left"))
    has_end = jend < len(t) and t[jend] - t_end < 30 * MIN
    tick = S.tick_for(p, E)
    ec = TAKER
    H = float(fl["H"])
    # anchors
    isp = int(np.searchsorted(t, sp))
    iclosed = int(np.searchsorted(t, (te // MIN) * MIN)) if fl["start"] == "next" else i0     # bars < the fill minute are closed
    seg = slice(isp, max(iclosed, isp + 1))
    iH = isp + int(np.argmax(g["h"][seg])) if iclosed > isp else isp
    L = float(min(g["l"][iH:iclosed].min() if iclosed > iH else E, E))
    ib = int(np.searchsorted(t, sp - 30 * MIN))
    base = float(g["c"][ib]) if 0 <= ib < len(t) and t[ib] <= sp - 30 * MIN + MIN else np.nan
    out = dict(H=H, L=L, base=base, tick=tick, vol_ok=bool(g["vol_ok"]), i0_t=int(t[i0]), cover=float(jend - i0) / 1440.0)
    REFTP, REFSL = lp_net(E, 5.0, MAKER, tick), stop_px(E, 4.0)

    def fin(r, name):
        if r is None:
            out[name] = np.nan; out[name + "_k"] = "NA"; out[name + "_x"] = 0; return
        net0, j, kind = r
        xm = int(t[j]) + MIN
        out[name] = net0 + fund_long(fl, xm); out[name + "_k"] = kind; out[name + "_x"] = xm

    R = lambda tp=REFTP, sl=REFSL, **kw: run_ht(g, i0, jend, E, tick, tp, sl, slip, ec, has_end=has_end, **kw)
    fin(R(), "REF")
    for x in (2, 2.5, 3, 3.5, 4, 4.5, 6):
        fin(R(tp=lp_net(E, x, MAKER, tick)), f"TP{x:g}")

    def anchor(px):
        L_ = math.ceil(px / tick - 1e-9) * tick
        nt = (L_ / E - 1) * 100 - TAKER - MAKER * L_ / E
        if nt < 1.0:
            return lp_net(E, 1.0, MAKER, tick)
        if nt > 10.0:
            return lp_net(E, 10.0, MAKER, tick)
        return L_
    A = {"A_H0": anchor(H), "A_Hm05": anchor(H * 0.995), "A_Hp05": anchor(H * 1.005), "A_H75": anchor(E + 0.75 * (H - E))}
    out["tpH_net"] = (A["A_H0"] / E - 1) * 100 - TAKER - MAKER * A["A_H0"] / E
    for k_, v in A.items():
        fin(R(tp=v), k_)
    fin(R(tp=min(REFTP, A["A_H0"])), "N_H0")
    fin(R(tp=min(REFTP, A["A_Hm05"])), "N_Hm05")
    for x in (2, 3):
        for n in (30, 60, 120):
            fin(R(k_over=stall_k(g, i0, jend, E, x, n, slip, ec)), f"B_X{x}_N{n}")
    sd = min(max(L * 0.997, E * 0.92), E * 0.985)
    out["dipstop_pct"] = (1 - sd / E) * 100
    fin(R(sl=sd), "C_DIPLOW")
    kv, vw0 = vwap_k(g, i0, jend, isp)
    out["vwap_entry"] = vw0
    if g["qv"] is None or not g["vol_ok"]:
        fin(R(), "C_VWAP5"); out["C_VWAP5_k"] = "NOVOL"
    else:
        fin(R(sl=stop_px(E, 8.0), k_over=kv), "C_VWAP5")
    if base == base:
        fin(R(sl=min(max(base * 0.995, E * 0.90), E * 0.985)), "C_BASE")
    else:
        fin(R(), "C_BASE"); out["C_BASE_k"] = "NOBASE"
    for s_ in (3, 5, 6):
        fin(R(sl=stop_px(E, s_)), f"S{s_}")
    fin(R(lock=(H, E * 1.0025)), "D_HLOCK0")
    fin(R(lock=(H, E * 1.0122)), "D_HLOCK1")
    for nm, a_, b_ in (("E_H0_5", "A_H0", "REF"), ("E_Hm05_5", "A_Hm05", "REF"), ("E_H0_LOCK", "A_H0", "D_HLOCK0")):
        out[nm] = 0.5 * (out[a_] + out[b_]); out[nm + "_k"] = out[a_ + "_k"] + "/" + out[b_ + "_k"]
        out[nm + "_x"] = max(out[a_ + "_x"], out[b_ + "_x"])
    # paper-parity mode (no slip / funding) for booked real rows
    pr = run_ht(g, i0, jend, E, tick, REFTP, REFSL, 0.0, 0.045, has_end=has_end)
    out["par_net"] = pr[0] if pr else np.nan
    # path facts + event-study path (close % from E, −60 … +1440 min, 5-min stride)
    jr = int(np.searchsorted(t, out["REF_x"] - MIN)) if out["REF_k"] != "NA" else jend
    hi = g["h"][i0:jr + 1]
    out["peak_ref"] = float((hi.max() / E - 1) * 100) if len(hi) else np.nan
    out["peak_ref_Hu"] = float((hi.max() - E) / (H - E)) if len(hi) and H > E else np.nan
    out["touchH"] = bool(len(hi) and hi.max() >= H)
    rel = np.arange(-12, 289) * 5
    ix = i0 + rel
    ok = (ix >= 0) & (ix < len(t))
    pp = np.full(len(rel), np.nan); pp[ok] = (g["c"][ix[ok]] / E - 1) * 100
    out["path"] = pp.astype(np.float32)
    hu = np.full(len(rel), np.nan)
    if H > E:
        hu[ok] = (g["c"][ix[ok]] - E) / (H - E)
    out["path_Hu"] = hu.astype(np.float32)
    return out, "ok"


# ───────────────────────────── WILLY engine ─────────────────────────────
def sim_willy(fl, slip):
    p, te, E = fl["pair"], int(fl["te"]), float(fl["E"])
    sp = int(fl["spike_ms"]) if fl["spike_ms"] == fl["spike_ms"] else te - HR
    a = (min(te, sp) - 60 * MIN) // MIN * MIN
    g = bars(p, a, te + 4 * HR, fresh=(fl["kind"] == "REAL"))
    if g is None:
        return None, "no data"
    t = g["t"]
    T0 = (te // MIN) * MIN + (MIN if fl["start"] == "next" else 0)
    i0 = int(np.searchsorted(t, T0))
    if i0 >= len(t) or t[i0] != T0:
        return None, "no entry bar"
    j1 = min(len(t), int(np.searchsorted(t, T0 + 92 * MIN, "left")) + 1)
    if j1 - i0 < 90:
        return None, "short path"
    tick = S.tick_for(p, E)
    live = E / (1 + slip / 100.0)            # willy_ref re-applies the entry slip
    H = float(fl["H"]); ra = fl.get("run_atr")
    ra = float(ra) if ra is not None and ra == ra else None
    out = dict(H=H, tick=tick, i0_t=int(t[i0]))
    sl_ = (t[i0:j1], g["o"][i0:j1], g["h"][i0:j1], g["l"][i0:j1], g["c"][i0:j1])

    def wr(P):
        y = S.willy_ref(*sl_, int(t[i0]), live, H, ra, P, tick, slip=slip)
        if y is None:
            return None
        net0, jj, why, tu, E_ = y
        return net0, i0 + jj, why, tu

    def fin(y, name):
        if y is None:
            out[name] = np.nan; out[name + "_k"] = "NA"; out[name + "_x"] = 0; return
        xm = int(t[y[1]]) + MIN
        out[name] = y[0] + fund_long(fl, xm); out[name + "_k"] = y[2]; out[name + "_x"] = xm

    W = S.W_CUR
    ref = wr(W); fin(ref, "REF")
    if ref is None:
        return None, "ref none"
    out["tu"] = ref[3]
    kind = "TP" if ref[2] == "FRENZY_TP" else ("SL" if ref[2] == "STOP_LOSS" else ("TIME" if ref[2] in ("MAX_HOLD_TIME", "DATA_END") else "DYN"))
    out["REF_kk"] = kind
    fin(wr(dict(W, HK=0.0)), "WA_H0")
    fin(wr(dict(W, HK=-0.5)), "WA_Hp05")
    h10 = wr(dict(W, HK=-1.0))
    fin(h10, "WA_Hp10")
    out["WE_HALF"] = 0.5 * (out["REF"] + out["WA_Hp10"]); out["WE_HALF_k"] = out["REF_k"] + "/" + out["WA_Hp10_k"]
    out["WE_HALF_x"] = max(out["REF_x"], out["WA_Hp10_x"])
    for tp in (1.25, 2, 3):
        fin(wr(dict(W, HIGH_ON=False, TP=float(tp))), f"WTP{tp:g}")
    jref = ref[1]
    for x in (1, 2):
        for n in (15, 30):
            k = stall_k(g, i0, min(j1, i0 + 90), E, x, n, slip, TAKER)
            nm = f"WB_X{x}_N{n}"
            if k is not None and k <= jref and k < len(t):
                X = g["o"][k] * (1 - slip / 100)
                xm = int(t[k]) + MIN
                out[nm] = (X / E - 1) * 100 - TAKER - TAKER * X / E + fund_long(fl, xm); out[nm + "_k"] = "RULE"; out[nm + "_x"] = xm
            else:
                out[nm] = out["REF"]; out[nm + "_k"] = out["REF_k"]; out[nm + "_x"] = out["REF_x"]
    hi = g["h"][i0:jref + 1]
    out["peak_ref"] = float((hi.max() / E - 1) * 100)
    return out, "ok"


# ───────────────────────────── cohorts ─────────────────────────────
def ms(s):
    return int(pd.Timestamp(str(s).replace("T", " ")[:19]).value // 10**6)


def fills():
    rows = []
    R = pd.read_csv(os.path.join(D, "real_cohort.csv"), low_memory=False)
    stk = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False,
                      usecols=lambda c: c in ("opened_at", "pair", "direction", "entry_strategy", "stack_keep"))
    stk = stk[stk.entry_strategy == "FRENZY_WILLY"]
    stk["k"] = stk.opened_at.astype(str).str.replace(" ", "T").str[:19]
    keepset = set(zip(stk.k[stk.stack_keep == True], stk.pair[stk.stack_keep == True]))  # noqa: E712
    inpool = set(zip(stk.k, stk.pair))
    for r in R.itertuples():
        sl = "HT" if r.entry_strategy == "FRENZY_WILLY_HT" else "WILLY"
        H = float(r.entry_frenzy_willy_spike_high)
        dip = r.entry_frenzy_willy_dip_pct if r.entry_frenzy_willy_dip_pct == r.entry_frenzy_willy_dip_pct else (H - r.entry_price) / H * 100
        rows.append(dict(sleeve=sl, kind="REAL", period="BATCH" if r.k >= CUR_BATCH else "MASTER", pair=r.pair, te=ms(r.opened_at),
                         k=r.k, E=float(r.entry_price), start="next", H=H, dip=dip,
                         spike_ms=ms(r.entry_frenzy_spike_at) if isinstance(r.entry_frenzy_spike_at, str) else np.nan,
                         booked=r.net_pct, booked_reason=r.close_reason, tp_target=r.frenzy_willy_tp_target_pct, mode=r.mode,
                         run_atr=(r.entry_atr_pct if r.frenzy_willy_runner_k == r.frenzy_willy_runner_k else np.nan),
                         today_keep=(dip >= 1.4) if sl == "HT" else ((r.k, r.pair) in keepset),
                         in_stack=True if sl == "HT" else ((r.k, r.pair) in inpool)))
    sys.path.insert(0, HERE)
    import ht_willy_sim as HWS  # noqa: E402
    for sl, fn in (("HT", "HT_WILLY_2026-11-20_cohort_HT.csv"), ("WILLY", "HT_WILLY_2026-11-20_cohort_WILLY.csv")):
        C = pd.read_csv(os.path.join(ROOT, "reports", fn))
        ra = HWS.load_fills(sl)["run_atr"] if sl == "WILLY" else None
        for r in C.itertuples():
            rows.append(dict(sleeve=sl, kind="BT", period="DESIGN" if r.ts < SPLIT else "HOLDOUT", pair=r.pair, te=int(r.ts),
                             k=str(pd.Timestamp(int(r.ts), unit="ms"))[:19], E=float(r.E), start="bar", H=float(r.H), dip=float(r.dip),
                             spike_ms=int(r.spike), booked=np.nan, booked_reason=r.ref_kind, tp_target=np.nan, mode="bt",
                             run_atr=(float(ra.loc[r.row]) if ra is not None else np.nan), today_keep=True, in_stack=True, half=r.half))
    Z = pd.DataFrame(rows)
    Z["fid"] = np.arange(len(Z))
    return Z


def work(fl):
    try:
        f = sim_ht if fl["sleeve"] == "HT" else sim_willy
        o1, why = f(fl, 0.10)
        if o1 is None:
            return fl["fid"], None, why
        o2, _ = f(fl, 0.20)
        cells = (["REF"] + HT_CELLS + HT_FIXED) if fl["sleeve"] == "HT" else (["REF"] + W_CELLS + W_FIXED)
        for c in cells:
            o1[c + "_s20"] = o2[c] if o2 is not None else np.nan
        return fl["fid"], o1, "ok"
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        return fl["fid"], None, f"error {type(e).__name__}: {e} {traceback.format_exc()[-300:]}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--n", type=int, default=0)
    A = ap.parse_args()
    Z = fills()
    if A.n:
        Z = Z.groupby(["sleeve", "kind"], group_keys=False).head(A.n)
    print(Z.groupby(["sleeve", "kind", "period"]).size(), flush=True)
    # real-fill funding first (serial, weight-light)
    for p in sorted(Z[Z.kind == "REAL"].pair.unique()):
        real_funding(p)
    recs = Z.to_dict("records")
    outs, fails = [], []
    with mp.Pool(A.procs) as pool:
        for i, (fid, r, why) in enumerate(pool.imap_unordered(work, recs, chunksize=2)):
            if r is None:
                fails.append((fid, why))
            else:
                r["fid"] = fid; outs.append(r)
            if i % 100 == 0:
                print(time.strftime("%H:%M:%S"), i, len(recs), flush=True)
    O = Z.merge(pd.DataFrame(outs), on="fid", how="left", suffixes=("", "_sim"))
    O.to_pickle(os.path.join(D, "sim.pkl"))
    print("fails", len(fails))
    for f in fails[:30]:
        print("  ", Z.set_index("fid").loc[f[0], ["sleeve", "kind", "pair", "k"]].tolist(), f[1][:200])


if __name__ == "__main__":
    main()
