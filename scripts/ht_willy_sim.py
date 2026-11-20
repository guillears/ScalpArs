#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 (PREREG reports/HT_WILLY_2026-11-20_PREREG.md) — SIM. Research only, BACKTEST.
Per baseline fill (BASELINE_TRADES_WILLY_HT_2026-11-18.csv) on 1m bars:
  HT  : REF (TP 5 / SL 4 / 24 h) + the 40-cell grid T{3,4,5,6,8} x S{3,4,5,6} x H{12,24} + OLD2 (the old +2 exit), on the REAL entry;
  both: K = 50 random entries per control (C_LOC ±30 min · C_EP [spike, max(spike + 6 h, entry + 1 h)) · C_EPD (HT: C_EP with the dip ≥ 1.4 rule)
        · C_OLD12 ±12 h 5m grid, diagnostic) on the sleeve's REF exit; per draw the timing-map features
        (minutes since spike, % below the running episode high, % below H, return since the spike-bar open).
  paths: close path % from the entry, −60 … +360 min, for the real entry and the first 5 C_EP draws (event-study chart).
Output reports/HT_WILLY_2026-11-20_data/sim_<SLEEVE>.pkl (one row per fill; arrays per row).
Usage: venv/bin/python -I scripts/ht_willy_sim.py need|sim [--sleeves WILLY,HT] [--procs 8]"""
import argparse, math, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_tp_sim as S   # noqa: E402  (willy_ref, fund_short, tick_for, W_CUR / HT_CUR / HT_OLD, price_for_net, net_taker_long)
import fast_tp_data as FD  # noqa: E402
RP = S.RP
D = os.path.join(RP, "HT_WILLY_2026-11-20_data"); os.makedirs(D, exist_ok=True)
MIN, H1 = 60_000, 3_600_000
TAKER, MAKER, SLIP = S.TAKER, S.MAKER, 0.10
K = 50
TPS = (3, 4, 5, 6, 8); SLS = (3, 4, 5, 6); HOLDS = (12, 24)
CELLS = [f"T{t}_S{s}_H{hh}" for t in TPS for s in SLS for hh in HOLDS] + ["OLD2"]
REF_CELL = "T5_S4_H24"
DIP_MIN = 1.4
PATH_PRE, PATH_POST = 60, 360
CONTROLS = ("C_LOC", "C_EP", "C_EPD", "C_OLD12")
MARK_OPEN = False   # True only in ht_willy_real.py (current batch: data ends now)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_fills(sl):
    X = pd.read_csv(os.path.join(RP, "BASELINE_TRADES_WILLY_HT_2026-11-18.csv"))
    X = X.loc[:, ~X.columns.duplicated()]
    X = X[X.sleeve == ("FRENZY_WILLY" if sl == "WILLY" else "FRENZY_WILLY_HT")].reset_index(drop=True)
    X["dip"] = (X.peak - X.live) / X.peak * 100
    X["run_atr"] = np.where(X.runner, X.atr, np.nan)
    return X


def span(sl, rw):
    hold = 24 * 60 if sl == "HT" else 90
    T, sp = int(rw.entry_ms), int(rw.spike_ms)
    a = min(T - 12 * H1, sp) - 2 * MIN
    b = max(T + 12 * H1, sp + 6 * H1, T + H1) + (hold + 130) * MIN
    return a, b


# ───────────────────────────── exits ─────────────────────────────
def fixed_exit(t, o, h, l, c, i0, T, E, tick, tp, sl, hold_min, slip=SLIP):
    """HT-style fixed exit from bar i0 (bar start T): resting maker TP at net tp (≥ 1 tick through), stop at net −sl (taker + slip,
    stop checked first in a bar), time cap at the open of the first bar ≥ T + hold. → (net before funding, exit idx, kind) or None."""
    t_end = T + hold_min * MIN
    j1 = int(np.searchsorted(t, t_end, "left"))
    L_tp = math.ceil(S.price_for_net(E, tp, MAKER) / tick - 1e-9) * tick
    P_stop = E * (100 - sl + TAKER) / (100 - TAKER)
    hh, ll = h[i0:j1], l[i0:j1]
    s_hit = np.flatnonzero(ll <= P_stop); t_hit = np.flatnonzero(hh >= L_tp + tick - 1e-12)
    js = s_hit[0] if len(s_hit) else 10**9; jt = t_hit[0] if len(t_hit) else 10**9
    if js <= jt and js < 10**9:
        j = i0 + js
        return S.net_taker_long(E, min(P_stop, o[j]), slip), j, "SL"
    if jt < 10**9:
        j = i0 + jt
        return (L_tp / E - 1) * 100 - TAKER - MAKER * L_tp / E, j, "TP"
    if j1 < len(t) and t[j1] - t_end < 30 * MIN:
        return S.net_taker_long(E, o[j1], slip), j1, "TIME"
    if MARK_OPEN and len(hh):
        return S.net_taker_long(E, c[i0 + len(hh) - 1], slip), i0 + len(hh) - 1, "OPEN"   # real-fill script only: marked at the last close
    return None   # data ends before the cap → unpriced


def ref_exit(sl, t, o, h, l, c, i0, T, live, H, ra, tick, cell=None):
    """the sleeve's REF exit (or an HT grid cell) at entry bar i0, decision price live → (net incl. funding, exit idx, kind, E)."""
    p = ref_exit.pair
    if sl == "HT" and (cell is None or cell != "OLD2"):
        cell = cell or REF_CELL
        tp, s_, hh = [float(x[1:]) for x in cell.split("_")]
        E = live * (1 + SLIP / 100.0)
        x = fixed_exit(t, o, h, l, c, i0, T, E, tick, tp, s_, int(hh * 60))
        if x is None:
            return None
        net0, j, kind = x
    else:
        P = S.W_CUR if sl == "WILLY" else S.HT_OLD
        hold = P["HOLD"]
        j1 = min(len(t), int(np.searchsorted(t, T + (hold + 2) * MIN, "left")) + 1)
        if j1 - i0 < hold:
            return None
        y = S.willy_ref(t[i0:j1], o[i0:j1], h[i0:j1], l[i0:j1], c[i0:j1], T, live, H, ra, P, tick)
        if y is None:
            return None
        net0, jj, why, tu, E = y
        j = i0 + jj
        kind = "TP" if why == "FRENZY_TP" else ("SL" if why == "STOP_LOSS" else ("TIME" if why in ("MAX_HOLD_TIME", "DATA_END") else "DYN"))
    fr = -float(S.fund_short(p, T, int(t[j]) + MIN))
    return net0 + fr, j, kind, E


# ───────────────────────────── per fill ─────────────────────────────
def one(sl, rw, seed):
    a0, b0 = span(sl, rw)
    A = FD.get1m(rw.pair, a0, b0)
    if A is None:
        return None
    t, o, h, l, c = A
    T, sp = int(rw.entry_ms), int(rw.spike_ms)
    ref_exit.pair = rw.pair
    i0 = int(np.searchsorted(t, T))
    if i0 >= len(t) or t[i0] != T or i0 < 1 or t[i0 - 1] != T - MIN:
        return None
    live = float(rw.live) if np.isfinite(rw.live) else float(c[i0 - 1])
    tick = S.tick_for(rw.pair, live)
    H = float(rw.peak); ra = rw.run_atr if np.isfinite(rw.run_atr) else None
    out = dict(pair=rw.pair, ts=T, spike=sp, row=int(rw.name), dip=float(rw.dip), live=live, H=H)
    r = ref_exit(sl, t, o, h, l, c, i0, T, live, H, ra, tick)
    if r is None:
        return None
    out.update(ref_net=r[0], ref_idx=r[1] - i0, ref_kind=r[2], ref_x=int(t[r[1]]) + MIN, E=r[3])
    if sl == "HT":
        nets = np.full(len(CELLS), np.nan); kinds = []; xs = np.zeros(len(CELLS), "int64")
        for ci, cl in enumerate(CELLS):
            y = ref_exit(sl, t, o, h, l, c, i0, T, live, H, ra, tick, cell=cl)
            if y is not None:
                nets[ci] = y[0]; xs[ci] = int(t[y[1]]) + MIN; kinds.append(y[2])
            else:
                kinds.append("NA")
        out.update(g_net=nets.astype(np.float32), g_kind=np.array(kinds), g_x=xs)
    # path facts for the decomposition: bars from the entry to 27 h
    jE = int(np.searchsorted(t, T + 27 * H1))
    out["pb_t"] = t[i0:jE].astype(np.int64); out["pb_h"] = h[i0:jE].astype(np.float32); out["pb_l"] = l[i0:jE].astype(np.float32)
    # event-study path (close % from E), −PATH_PRE … +PATH_POST min
    def cpath(i_, E_):
        v = np.full(PATH_PRE + PATH_POST, np.nan, np.float32)
        lo, hi = max(0, i_ - PATH_PRE), min(len(t), i_ + PATH_POST)
        v[lo - (i_ - PATH_PRE): hi - (i_ - PATH_PRE)] = (c[lo:hi] / E_ - 1) * 100
        return v
    out["cpath"] = cpath(i0, r[3])
    # random entries
    rng = np.random.default_rng(seed)
    isp = int(np.searchsorted(t, sp))
    runhi = np.maximum.accumulate(np.where(np.arange(len(t)) >= isp, h, -np.inf))
    ep_end = max(sp + 6 * H1, T + H1)
    cand = {}
    loc = np.r_[np.arange(-30, 0), np.arange(1, 31)]
    cand["C_LOC"] = lambda: T + int(rng.choice(loc)) * MIN
    jep0, jep1 = int(np.searchsorted(t, sp)) + 1, int(np.searchsorted(t, ep_end))
    ep_idx = np.arange(max(jep0, 1), jep1)
    cand["C_EP"] = lambda: int(t[rng.choice(ep_idx)]) if len(ep_idx) else None
    ok_d = ep_idx[c[ep_idx - 1] <= H * (1 - DIP_MIN / 100.0)] if len(ep_idx) else ep_idx
    cand["C_EPD"] = lambda: int(t[rng.choice(ok_d)]) if len(ok_d) else None
    g12 = np.r_[np.arange(-144, 0), np.arange(1, 145)]
    cand["C_OLD12"] = lambda: T + int(rng.choice(g12)) * 300_000
    for cn in CONTROLS:
        if cn == "C_EPD" and sl != "HT":
            continue
        nets, feats, paths, pre = [], [], [], []
        tries = 0
        while len(nets) < K and tries < 4 * K:
            tries += 1
            tj = cand[cn]()
            if tj is None:
                break
            if cn == "C_LOC" and tj < sp:
                continue
            j = int(np.searchsorted(t, tj))
            if j < 1 or j >= len(t) or t[j] != tj or t[j - 1] != tj - MIN:
                continue
            Ej = float(c[j - 1])
            y = ref_exit(sl, t, o, h, l, c, j, tj, Ej, H, ra, tick)
            if y is None:
                continue
            rh = runhi[j - 1] if j - 1 >= isp else np.nan
            feats.append((tj, (tj - sp) / MIN, (rh - Ej) / rh * 100 if np.isfinite(rh) else np.nan, (H - Ej) / H * 100,
                          (Ej / o[isp] - 1) * 100 if isp < len(o) else np.nan, int(t[y[1]]) + MIN))
            nets.append(y[0])
            if cn == "C_EP" and len(paths) < 5:
                paths.append(cpath(j, y[3]))
        out[cn + "_net"] = np.array(nets, np.float32)
        out[cn + "_f"] = np.array(feats, float).reshape(-1, 6)
        if cn == "C_EP":
            out["rnd_paths"] = np.array(paths, np.float32).reshape(-1, PATH_PRE + PATH_POST)
    # the real entry's own features (same definitions)
    rh = runhi[i0 - 1] if i0 - 1 >= isp else np.nan
    out["f_since"] = (T - sp) / MIN; out["f_below_run"] = (rh - live) / rh * 100 if np.isfinite(rh) else np.nan
    out["f_below_H"] = (H - live) / H * 100; out["f_ret_sp"] = (live / o[isp] - 1) * 100 if isp < len(o) else np.nan
    return out


def work(args):
    sl, g = args
    res = []
    for i, rw in g.iterrows():
        try:
            x = one(sl, rw, 20261120 + int(i))
        except Exception as e:  # noqa: BLE001
            print("ERR", sl, rw.pair, rw.entry_ms, repr(e), flush=True); x = None
        if x is not None:
            res.append(x)
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("step"); ap.add_argument("--sleeves", default="HT,WILLY")
    ap.add_argument("--procs", type=int, default=8)
    A = ap.parse_args()
    import exh_exit2_sim  # noqa: F401  (funding sources, loaded once before the fork)
    for sl in A.sleeves.split(","):
        X = load_fills(sl)
        if A.step == "need":
            need = set()
            for _, rw in X.iterrows():
                a, b = span(sl, rw)
                for ds in FD.days_of(a, b):
                    need.add((rw.pair, ds))
            miss = [x for x in need if not FD.have(*x) and pd.Timestamp(x[1]) <= pd.Timestamp("2026-11-19")]
            log(sl, "pair-days", len(need), "missing", len(miss))
            if miss:
                FD.fetch(miss)
        else:
            jobs = [(sl, g) for _, g in X.groupby("pair")]; jobs.sort(key=lambda j: -len(j[1]))
            out = []
            with mp.get_context("fork").Pool(A.procs) as pool:
                for r_ in pool.imap_unordered(work, jobs, chunksize=1):
                    out.extend(r_)
            Z = pd.DataFrame(out).sort_values("row").reset_index(drop=True)
            Z.to_pickle(os.path.join(D, f"sim_{sl}.pkl"))
            log(sl, "rows", len(X), "simulated", len(Z))


if __name__ == "__main__":
    main()
