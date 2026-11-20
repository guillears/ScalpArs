#!/usr/bin/env python3
"""FAST_TP (PREREG reports/FAST_TP_2026-11-20_PREREG.md) — FRAMES + SIM. Research only, BACKTEST.
Per sleeve (EXH · DIST · FLATBASE · WILLY · HT) and set (MAIN = the sleeve's frozen fills, RND = random-entry control):
  REF = the sleeve's CURRENT exit (EXH / DIST: the 1m-engine REF from EXH_EXIT2 / DIST_EXIT, aligned to the 1m path and checked;
        FLATBASE / WILLY / HT: re-simulated here on 1m), then the 33 pre-registered cells overlaid on the first 30 min:
  F_X{x}_N{n}: resting maker limit at ±X % live for bars starting in the first N minutes (fills ≥ 1 tick through);
  V_R{r}: best favourable excursion ÷ minutes ≥ R at a 1m close in the first 30 min → taker exit at that close.
  A cell changes a fill only when its trigger bar is strictly before REF's exit bar (same bar → REF; except a REF fixed TP farther
  than X in the same bar → the nearer limit fills first). Never-triggered fills keep REF's net exactly (Δ = 0, asserted in eval).
Steps:  need  → list + fetch the 1m pair-days (data.binance.vision, 0 weight)
        sim   → reports/FAST_TP_2026-11-20_data/sim_<SLEEVE>_<SET>.pkl
Usage: venv/bin/python -I scripts/fast_tp_sim.py need|sim [--sleeves EXH,DIST,FLATBASE,WILLY,HT] [--procs 7]"""
import argparse, json, math, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import fast_tp_data as FD   # noqa: E402
RP = os.path.join(R, "reports"); D = FD.D
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
TAKER, MAKER = 0.05, 0.02
XS_ = (1, 1.5, 2, 3, 4); NS_ = (1, 2, 3, 5, 10, 15); RS_ = (0.5, 1, 2)
CELLS = [f"F_X{x:g}_N{n}" for x in XS_ for n in NS_] + [f"V_R{r:g}" for r in RS_]
NWIN = 30
CHART_MIN = 360
TICKS = json.load(open(os.path.join(RP, "backtest_cache", "exchange_ticks.json")))
NOTIONAL = 6_100.0
# WILLY / HT exits: OLD = the baseline file's exit (parity); CUR = trading_config.json 2026-11-20
CFG = json.load(open(os.path.join(R, "trading_config.json")))["thresholds"]
W_CUR = dict(TP=float(CFG["frenzy_willy_tp_pct"]), STOP=-abs(float(CFG["frenzy_willy_stop_pct"])), HOLD=int(CFG["frenzy_willy_max_hold_minutes"]),
             LATE_ON=True, LATE_MIN=float(CFG["frenzy_willy_late_tp_minutes"]), LATE_TP=float(CFG["frenzy_willy_late_tp_pct"]),
             HIGH_ON=bool(CFG["frenzy_willy_high_target_enabled"]), HK=float(CFG["frenzy_willy_high_k_pct"]),
             HMAX=float(CFG["frenzy_willy_high_tp_max_pct"]), HT_TP=0.0, RUN=True, RUN_K=float(CFG["frenzy_willy_runner_atr_k"]),
             RUN_FLOOR=float(CFG["frenzy_willy_runner_floor_pct"]))
HT_CUR = dict(W_CUR, TP=float(CFG["frenzy_willy_ht_tp_pct"]), HOLD=int(float(CFG["frenzy_willy_ht_max_hold_minutes"])),
              LATE_ON=bool(CFG["frenzy_willy_ht_late_tp_enabled"]), HIGH_ON=bool(CFG["frenzy_willy_ht_high_target_enabled"]), HT_TP=0.0, RUN=False)
HT_OLD = dict(W_CUR, HT_TP=2.0, RUN=False)    # baseline file: WILLY logic, target ≥ 2, 90 min, late TP on, high target on


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def slip_of(q5):
    q5 = np.asarray(q5, float)
    return np.maximum(0.10, -0.042 + 0.276 * np.sqrt(NOTIONAL / np.maximum(q5, 1.0)))


def half_of(ms):
    t = pd.to_datetime(np.asarray(ms, "int64"), unit="ms")
    return np.array([f"{y % 100}-H{1 if m <= 6 else 2}" for y, m in zip(t.year, t.month)])


# ───────────────────────────── frames ─────────────────────────────
def frames(sl):
    """MAIN and RND rows for sleeve sl. Common columns: pair, T, dirn, hold_min, plus sleeve-specific REF inputs."""
    rng = np.random.default_rng(20261120)
    if sl == "EXH":
        X = pd.read_csv(os.path.join(RP, "EXH_EXIT2_2026-11-18_trades.csv.gz"))
        M = pd.DataFrame(dict(pair=X.pair, ts=X.entry_ms.astype("int64"), atrp=X.atrp, slip=X.slip, ref_net=X.net_REF, ref_k=X.k_REF,
                              ref_x=X.x_REF.astype("int64")))
        Rr = pd.read_pickle(os.path.join(RP, "EXH_EXIT2_data", "rows_RND.pkl")).reset_index(drop=True)
        Z = np.load(os.path.join(RP, "EXH_EXIT2_data", "sim_RND.npz"))
        assert (Z["xi"] == Rr.xi.values).all()
        r, k, xm, f, ok = Z["r"][:, 0].astype(float), Z["k"][:, 0], Z["xm"][:, 0], Z["f"][:, 0].astype(float), Z["ok"] & Z["src1m"]
        s = Rr.slip.values
        net = r + f - s - 0.05 - np.where(k != 0, s + 0.05, 0.02)
        RN = pd.DataFrame(dict(pair=Rr.pair, ts=Rr.entry_ms.astype("int64"), atrp=Rr.atrp, slip=s, ref_net=net, ref_k=k, ref_x=xm.astype("int64")))[ok]
        for F_ in (M, RN):
            F_["dirn"] = -1; F_["hold_min"] = 24 * 60; F_["tp_pct"] = np.minimum(20 * F_.atrp, 10.0); F_["sl_pct"] = np.minimum(1.5 * F_.atrp, 6.0)
        return M.reset_index(drop=True), RN.reset_index(drop=True)
    if sl == "DIST":
        X = pd.read_csv(os.path.join(RP, "DIST_EXIT_2026-11-18_trades.csv.gz")); X = X[X.cell == "REF_B"]
        M = pd.DataFrame(dict(pair=X.pair, ts=X.entry_ms.astype("int64"), slip=X.slip, ref_net=X.net, ref_k=X.kind.astype(int),
                              ref_x=X.exit_ms.astype("int64"), tp_pct=X.tp_pct, sl_pct=X.sl_pct, src1m=X.src1m))
        M = M[M.src1m.astype(bool)].drop(columns="src1m")
        Rr = pd.read_pickle(os.path.join(RP, "DIST_EXIT_data", "rnd.pkl")).reset_index(drop=True)
        Z = np.load(os.path.join(RP, "DIST_EXIT_data", "sim_RND.npz"))
        assert len(Z["r"]) == len(Rr) and str(Z["names"][0]) == "REF_B"
        r, k, xm, f = Z["r"][:, 0].astype(float), Z["k"][:, 0], Z["xm"][:, 0], Z["f"][:, 0].astype(float)
        ok = Z["ok"] & Z["src1m"] & (k >= 0) & (k != 9)
        s = slip_of(Rr.q5.values)
        net = r + f - s - 0.05 - np.where(k != 0, s + 0.05, 0.02)
        RN = pd.DataFrame(dict(pair=Rr.pair, ts=Rr.entry_ms.astype("int64"), slip=s, ref_net=net, ref_k=k, ref_x=xm.astype("int64"),
                               tp_pct=np.minimum(20 * Rr.atr15_pct.values, 10.0), sl_pct=np.minimum(1.5 * Rr.atr15_pct.values, 6.0)))[ok]
        for F_ in (M, RN):
            F_["dirn"] = -1; F_["hold_min"] = 24 * 60
        return M.reset_index(drop=True), RN.reset_index(drop=True)
    if sl == "FLATBASE":
        X = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_FLATBASE_2026-11-18.csv"))
        M = pd.DataFrame(dict(pair=X.pair, ts=X.entry_ms.astype("int64"), file_net=X.n0, file_why=X.why, file_x=X.exit_ms.astype("int64")))
        rows = []
        for i, rw in M.iterrows():
            ts = rng.choice(np.r_[np.arange(-144, 0), np.arange(1, 145)], 3, replace=False) * 300_000 + rw["ts"]
            for t_ in ts:
                rows.append(dict(pair=rw.pair, ts=int(t_), src=i))
        RN = pd.DataFrame(rows)
        for F_ in (M, RN):
            F_["dirn"] = -1; F_["hold_min"] = 72 * 60; F_["slip"] = 0.10; F_["tp_pct"] = 5.0; F_["sl_pct"] = 10.0
        return M, RN
    if sl in ("WILLY", "HT"):
        X = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_WILLY_HT_2026-11-18.csv"))
        X = X.loc[:, ~X.columns.duplicated()]
        X = X[X.sleeve == ("FRENZY_WILLY" if sl == "WILLY" else "FRENZY_WILLY_HT")].reset_index(drop=True)
        M = pd.DataFrame(dict(pair=X.pair, ts=X.entry_ms.astype("int64"), live=X.live, H=X.peak, run_atr=np.where(X.runner, X.atr, np.nan),
                              file_net=X.net, file_why=X.reason, file_x=X.exit_ms.astype("int64"), file_E=X.E, file_fund=X.funding))
        rows = []
        for i, rw in M.iterrows():
            ts = rng.choice(np.r_[np.arange(-144, 0), np.arange(1, 145)], 3, replace=False) * 300_000 + rw["ts"]
            for t_ in ts:
                rows.append(dict(pair=rw.pair, ts=int(t_), H=rw.H, run_atr=rw.run_atr, src=i))
        RN = pd.DataFrame(rows)
        hold = (W_CUR if sl == "WILLY" else HT_CUR)["HOLD"]
        for F_ in (M, RN):
            F_["dirn"] = 1; F_["hold_min"] = hold; F_["slip"] = 0.10
        return M, RN
    raise ValueError(sl)


def windows(sl, F_):
    """(pair, t0, t1) 1m windows each row needs."""
    if sl in ("EXH", "DIST"):
        return [(p, t, t + CHART_MIN * MIN + 5 * MIN) for p, t in zip(F_.pair, F_.ts)]
    if sl == "FLATBASE":
        return [(p, t - 5 * MIN, t + 72 * H1 + 5 * MIN) for p, t in zip(F_.pair, F_.ts)]
    hold = int(F_.hold_min.iloc[0])
    return [(p, t - 5 * MIN, t + max(hold, NWIN) * MIN + 10 * MIN) for p, t in zip(F_.pair, F_.ts)]


# ───────────────────────────── REF sims (FLATBASE / WILLY / HT) ─────────────────────────────
def tick_for(p, px):
    t = TICKS.get(p)
    return float(t) if t else px * 1e-4


def price_for_net(E, net, fee):
    return E * (1 + (net + TAKER) / 100.0) / (1 - fee / 100.0)


def net_taker_long(E, P, slip):
    x = P * (1 - slip / 100.0)
    return (x / E - 1) * 100 - TAKER - TAKER * x / E


def willy_ref(t, o, h, l, c, T, live, H, run_atr, P, tick, slip=0.10):
    """wsim.simulate_exit logic with explicit params. Bars start at T. → (net before funding, exit bar idx, reason, target, E)."""
    E = live * (1 + slip / 100.0)
    tu = P["TP"]
    if P["HIGH_ON"] and H and H > 0:
        x = H * (1 - P["HK"] / 100.0); t_raw = (x / E - 1) * 100 - TAKER - MAKER * x / E
        tu = max(P["TP"], min(P["HMAX"], t_raw))
    if P["HT_TP"] > 0 and tu < P["HT_TP"]:
        tu = P["HT_TP"]
    L_tp = math.ceil(price_for_net(E, tu, MAKER) / tick - 1e-9) * tick
    L_late = math.ceil(price_for_net(E, P["LATE_TP"], MAKER) / tick - 1e-9) * tick
    P_stop = E * (100 + P["STOP"] + TAKER) / (100 - TAKER)
    t_end = T + P["HOLD"] * MIN
    runner = (run_atr is not None) and np.isfinite(run_atr) and P["RUN"]
    arm_at = tu - (TAKER - MAKER); armed = False; peak = -1e9
    for j in range(len(t)):
        bt = t[j]; oo, hh, ll, cc = o[j], h[j], l[j], c[j]
        if bt >= t_end:
            return net_taker_long(E, oo, slip), j, "MAX_HOLD_TIME", tu, E
        age = (bt - T) / MIN
        late = P["LATE_ON"] and age >= P["LATE_MIN"]
        if not runner:
            if ll <= P_stop:
                return net_taker_long(E, min(P_stop, oo), slip), j, "STOP_LOSS", tu, E
            lim = L_late if late else L_tp
            if late and age < P["LATE_MIN"] + 1 and oo >= L_late:
                return (oo / E - 1) * 100 - TAKER - TAKER * oo / E, j, "WILLY_LATE_TP", tu, E
            if hh >= lim + tick - 1e-12:
                return (lim / E - 1) * 100 - TAKER - MAKER * lim / E, j, ("WILLY_LATE_TP" if (late and lim == L_late) else "FRENZY_TP"), tu, E
        else:
            if not armed:
                if ll <= P_stop:
                    return net_taker_long(E, min(P_stop, oo), slip), j, "STOP_LOSS", tu, E
                hi_net = (hh / E - 1) * 100 - TAKER - TAKER * hh / E
                if late and hi_net >= P["LATE_TP"] and hi_net < arm_at:
                    Pl = E * (100 + P["LATE_TP"] + TAKER) / (100 - TAKER)
                    return net_taker_long(E, max(Pl, oo), slip), j, "WILLY_LATE_TP", tu, E
                if hi_net >= arm_at:
                    armed = True; peak = hi_net
                    line = max(max(peak, tu) - P["RUN_K"] * run_atr, P["RUN_FLOOR"], P["STOP"])
                    c_net = (cc / E - 1) * 100 - TAKER - TAKER * cc / E
                    if c_net <= line:
                        return net_taker_long(E, E * (100 + line + TAKER) / (100 - TAKER), slip), j, "WILLY_RUNNER_TRAIL", tu, E
                continue
            line = max(max(peak, tu) - P["RUN_K"] * run_atr, P["RUN_FLOOR"], P["STOP"])
            Pline = E * (100 + line + TAKER) / (100 - TAKER)
            if ll <= Pline:
                return net_taker_long(E, min(Pline, oo), slip), j, "WILLY_RUNNER_TRAIL", tu, E
            peak = max(peak, (hh / E - 1) * 100 - TAKER - TAKER * hh / E)
    if not len(t):
        return None
    return net_taker_long(E, c[-1], slip), len(t) - 1, "DATA_END", tu, E


def flatbase_ref(t, o, h, l, c, T, entry_px, tick, slip=0.10, tp=5.0, stop=10.0, hold_h=72):
    """high_atr_flatbase.exit_sim on 1m. → (net before funding, exit idx, why, F)."""
    F = entry_px * (1 - slip / 100.0)
    tp_px = F * (1 - tp / 100.0); st_px = F * (1 + stop / 100.0)
    nb = min(len(t), int(np.searchsorted(t, T + hold_h * H1, "left")))
    if nb < 1:
        return None
    for j in range(nb):
        if h[j] >= st_px:
            x = max(st_px, o[j]) * (1 + slip / 100.0); g = (1 - x / F) * 100
            return g - TAKER - TAKER * x / F, j, "STOP_LOSS", F
        if l[j] <= tp_px - tick:
            g = tp / 1.0
            return g - TAKER - MAKER * (tp_px / F), j, "TP", F
    x = c[nb - 1] * (1 + slip / 100.0); g = (1 - x / F) * 100
    return g - TAKER - TAKER * x / F, nb - 1, ("MAX_HOLD" if nb >= int(hold_h * 60) - 1 else "OPEN"), F


# ───────────────────────────── funding ─────────────────────────────
_FUND = {}


def fund_short(p, T, xm):
    """funding % received by a SHORT over (T, xm] — the engine's TB.funding_for convention; missing months → engine's FUND_MISS proxy."""
    import exh_exit2_sim as XS
    TB = XS.TB
    if p not in _FUND:
        ft, fc, months = TB.funding_for(p); _FUND[p] = (np.asarray(ft, "int64"), np.asarray(fc, float), months)
    ft, fc, months = _FUND[p]
    xm = np.atleast_1d(np.asarray(xm, "int64"))
    if not len(ft):
        return float((-(xm - T) / (8 * H1) * TB.FUND_MISS)[0])
    fs = (fc[np.searchsorted(ft, xm, "right")] - fc[np.searchsorted(ft, T, "right")]) * 100
    me = pd.Timestamp(int(T), unit="ms").strftime("%Y-%m")
    miss = np.array([(me not in months) or (pd.Timestamp(int(x_), unit="ms").strftime("%Y-%m") not in months) for x_ in np.atleast_1d(xm)])
    return float(np.where(miss, -(xm - T) / (8 * H1) * TB.FUND_MISS, fs)[0])


# ───────────────────────────── per-row sim ─────────────────────────────
def sim_row(sl, rw):
    """→ dict or None. Bars start at T (1m)."""
    T = int(rw["ts"]); p = rw.pair
    if sl in ("EXH", "DIST"):
        a = FD.get1m(p, T, T + CHART_MIN * MIN + MIN)
        if a is None:
            return None
        t, o, h, l, c = a
        if t[0] != T or len(t) < NWIN + 1:
            return None
        E = o[0]
        tick = tick_for(p, E)
        ridx = int((rw.ref_x - T) // MIN) - 1
        k = int(rw.ref_k)
        # alignment check: REF exit bar consistent with the 1m path
        stp = E * (1 + rw.sl_pct / 100.0); tpp = E * (1 - rw.tp_pct / 100.0)
        n_ = min(ridx, len(h))
        pre_ok = (not (h[:max(n_, 0)] >= stp).any()) and (not (l[:max(n_, 0)] <= tpp - tick).any())
        if 0 <= ridx < len(h):
            at_ok = (h[ridx] >= stp) if k == 1 else ((l[ridx] <= tpp - tick) if k == 0 else True)
        else:
            at_ok = ridx >= len(h)   # exit after the stored window (≥ 6 h): only the pre-window part is checkable
        ref = dict(ref_net=float(rw.ref_net), ref_idx=ridx, ref_kind={0: "TP", 1: "SL", 2: "TIME", 3: "DYN"}.get(k, "OTHER"),
                   ref_tp=float(rw.tp_pct), parity=bool(pre_ok and at_ok), E=E, tick=tick, slip=float(rw.slip))
        ent_cost = float(rw.slip) + TAKER
    elif sl == "FLATBASE":
        a = FD.get1m(p, T - MIN, T + 72 * H1 + MIN)
        if a is None:
            return None
        t, o, h, l, c = a
        if t[0] != T - MIN or t[1] != T:
            return None
        entry_px = c[0]; t, o, h, l, c = t[1:], o[1:], h[1:], l[1:], c[1:]
        if len(t) < 72 * 60 - 5 and (T + 72 * H1) < int(pd.Timestamp("2026-11-18").value // 10**6):
            return None
        tick = tick_for(p, entry_px)
        x = flatbase_ref(t, o, h, l, c, T, entry_px, tick)
        if x is None:
            return None
        net0, ridx, why, F = x
        fr = float(fund_short(p, T, t[ridx] + MIN))
        ref = dict(ref_net=net0 + fr, ref_idx=ridx, ref_kind={"TP": "TP", "STOP_LOSS": "SL"}.get(why, "TIME"), ref_tp=5.0,
                   ref_why=why, ref_x=int(t[ridx] + MIN), E=F, tick=tick, slip=0.10, ref_fund=fr)
        if "file_why" in rw:
            ref["parity"] = bool(why.replace("OPEN", "MAX_HOLD") == rw.file_why and abs((t[ridx] + MIN) - rw.file_x) <= 5 * MIN)
            ref["file_net"] = float(rw.file_net)
        E = F; ent_cost = TAKER   # slip already inside F
    else:   # WILLY / HT (long)
        hold = int(rw.hold_min)
        a = FD.get1m(p, T - MIN, T + max(hold, NWIN) * MIN + 6 * MIN)
        if a is None:
            return None
        t, o, h, l, c = a
        if t[0] != T - MIN or t[1] != T:
            return None
        live = float(rw.live) if ("live" in rw and np.isfinite(rw.live)) else float(c[0])
        t, o, h, l, c = t[1:], o[1:], h[1:], l[1:], c[1:]
        if len(t) < min(hold, NWIN) + 1:
            return None
        tick = tick_for(p, live)
        P = W_CUR if sl == "WILLY" else HT_CUR
        ra = rw.run_atr if np.isfinite(rw.run_atr) else None
        x = willy_ref(t, o, h, l, c, T, live, rw.H, ra, P, tick)
        if x is None:
            return None
        net0, ridx, why, tu, E = x
        fr = -float(fund_short(p, T, t[ridx] + MIN))   # long pays what a short receives
        kind = "TP" if why == "FRENZY_TP" else ("SL" if why == "STOP_LOSS" else ("TIME" if why in ("MAX_HOLD_TIME", "DATA_END") else "DYN"))
        ref = dict(ref_net=net0 + fr, ref_idx=ridx, ref_kind=kind, ref_why=why, ref_tp=tu, E=E, tick=tick, slip=0.10, ref_fund=fr,
                   ref_x=int(t[ridx]))
        if "file_why" in rw:
            ref["file_net"] = float(rw.file_net)
            if sl == "HT":   # parity at the file's OLD HT exit
                y = willy_ref(t, o, h, l, c, T, live, rw.H, ra, HT_OLD, tick)
                ref["par_net"] = y[0] + (-float(fund_short(p, T, t[y[1]] + MIN))); ref["par_why"] = y[2]; ref["par_x"] = int(t[y[1]])
            else:
                ref["par_net"] = ref["ref_net"]; ref["par_why"] = why; ref["par_x"] = int(t[ridx])
            ref["parity"] = bool(ref["par_why"] == rw.file_why and abs(ref["par_x"] - rw.file_x) <= MIN)
        ent_cost = TAKER
    d = int(rw.dirn)
    n = min(NWIN, len(t))
    oo, hh, ll, cc = o[:n], h[:n], l[:n], c[:n]
    if d < 0:
        fav_thru = (1 - (ll + ref["tick"]) / E) * 100       # ≥ X ⇔ low ≤ level − 1 tick
        mfe = (1 - np.minimum.accumulate(ll) / E) * 100
        cfav = (1 - cc / E) * 100
    else:
        fav_thru = None
        mfe = (np.maximum.accumulate(hh) / E - 1) * 100
        cfav = (cc / E - 1) * 100
    out = dict(ref)
    out.update(pair=p, ts=T, dirn=d, mfe30=float(mfe[-1]), mfe5=float(mfe[min(4, n - 1)]), mfe15=float(mfe[min(14, n - 1)]))
    # path for the chart: close-based favourable % each minute, CHART_MIN minutes (NaN after the data)
    m_ = min(CHART_MIN, len(t))
    cp = np.full(CHART_MIN, np.nan, np.float32)
    cp[:m_] = ((1 - c[:m_] / E) * 100) if d < 0 else ((c[:m_] / E - 1) * 100)
    out["cpath"] = cp
    nets = np.full(len(CELLS), out["ref_net"], float); fire = np.full(len(CELLS), -1, np.int16)
    ridx = out["ref_idx"]
    for ci, nm in enumerate(CELLS):
        if nm.startswith("F_"):
            X = float(nm.split("_")[1][1:]); N = int(nm.split("_")[2][1:])
            if d < 0:
                hit = fav_thru[:min(N, n)] >= X - 1e-12
                lvl_net = X - ent_cost - MAKER * (1 - X / 100.0) if sl == "FLATBASE" else X - ent_cost - MAKER
            else:
                lvl = math.ceil(price_for_net(E, X, MAKER) / ref["tick"] - 1e-9) * ref["tick"]
                hit = hh[:min(N, n)] >= lvl + ref["tick"] - 1e-12
                lvl_net = (lvl / E - 1) * 100 - TAKER - MAKER * lvl / E
            if not hit.any():
                continue
            j = int(np.argmax(hit))
            if j < ridx or (j == ridx and out["ref_kind"] == "TP" and out["ref_tp"] > X + 1e-9):
                fr = float(fund_short(p, T, T + (j + 1) * MIN)) * (1 if d < 0 else -1)
                nets[ci] = lvl_net + fr; fire[ci] = j
        else:
            Rr = float(nm.split("_")[1][1:])
            ok = mfe / np.arange(1, n + 1) >= Rr
            if not ok.any():
                continue
            j = int(np.argmax(ok))
            if j < ridx:
                fr = float(fund_short(p, T, T + (j + 1) * MIN)) * (1 if d < 0 else -1)
                if d < 0:
                    if sl == "FLATBASE":
                        x = cc[j] * (1 + 0.10 / 100); g = (1 - x / E) * 100; nv = g - TAKER - TAKER * x / E
                    else:
                        nv = cfav[j] - ent_cost - ref["slip"] - TAKER
                else:
                    nv = net_taker_long(E, cc[j], 0.10)
                nets[ci] = nv + fr; fire[ci] = j
    out["nets"] = nets.astype(np.float32); out["fire"] = fire
    return out


def work(args):
    sl, g = args
    res = []
    for i, rw in g.iterrows():
        try:
            x = sim_row(sl, rw)
        except Exception as e:  # noqa: BLE001
            print("ERR", sl, rw.pair, rw["ts"], repr(e), flush=True); x = None
        if x is not None:
            x["row"] = i
            for k_ in ("src",):
                if k_ in rw:
                    x[k_] = rw[k_]
            res.append(x)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--sleeves", default="EXH,DIST,FLATBASE,WILLY,HT")
    ap.add_argument("--procs", type=int, default=7); ap.add_argument("--sets", default="MAIN,RND")
    A = ap.parse_args()
    os.makedirs(D, exist_ok=True)
    import exh_exit2_sim  # noqa: F401  (funding sources, loaded once before the fork)
    for sl in A.sleeves.split(","):
        M, RN = frames(sl)
        sets = {"MAIN": M, "RND": RN}
        if A.step == "need":
            need = set()
            for nm in A.sets.split(","):
                for p, t0, t1 in windows(sl, sets[nm]):
                    for ds in FD.days_of(t0, t1):
                        need.add((p, ds))
            miss = [x for x in need if not FD.have(*x)]
            log(sl, "pair-days", len(need), "missing", len(miss))
            if miss:
                FD.fetch(miss)
        elif A.step == "sim":
            for nm in A.sets.split(","):
                F_ = sets[nm]
                jobs = [(sl, g) for _, g in F_.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
                out = []
                with mp.get_context("fork").Pool(A.procs) as pool:
                    for r_ in pool.imap_unordered(work, jobs, chunksize=1):
                        out.extend(r_)
                Z = pd.DataFrame(out).sort_values("row").reset_index(drop=True)
                Z.to_pickle(os.path.join(D, f"sim_{sl}_{nm}.pkl"))
                log(sl, nm, "rows", len(F_), "simulated", len(Z), "parity" if "parity" in Z else "", Z["parity"].mean() if "parity" in Z else "")


if __name__ == "__main__":
    main()
