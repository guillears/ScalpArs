#!/usr/bin/env python3
"""STATE_EXIT_2026-11-21 — per fill: today's exit replay, fixed-TP alternatives, decision points + state features + labels (research only).

Decision point = the CLOSE of 1m bar k (time t_k + 60 s) while the trade is open under today's exit (k + 1 ≤ the exit bar). Exit-now value =
taker exit at the OPEN of bar k + 1 with slip (the sleeve's stop slip, ≥ 0.10 %) + taker fee, funding accrued so far; hold value = today's
exit result on the same path (maker TP, taker stop / time + slip, funding over the hold). label = exit-now − hold.
Rows kept where gross P&L at the close ≥ +1 % (G1) or the in-trade peak so far ≥ +2 % (G3); flags g1 / g2 (≥ +2 now) / g3.
Features use ONLY bars that closed at or before t_k + 60 s: 1m rolling windows ending at k; clock-aligned 5m / 15m / 1h / 4h bars that are
COMPLETE by then; metrics rows with create_time ≤ t_k (the bar OPEN — one minute conservative); funding = the last settlement ≤ t_k.
Lag check: `--lagcheck` recomputes features on the path TRUNCATED after bar k for random (fill, k) and asserts equality with the full run.
→ reports/STATE_EXIT_2026-11-21_data/{fills_out.csv, dec_<SLEEVE>.pkl, paths.pkl}
Usage: venv/bin/python -I scripts/STATE_EXIT_2026-11-21_build.py [--lagcheck] [--procs 8]"""
import argparse, os, sys, time
import importlib.util
import multiprocessing as mp
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
_sp = importlib.util.spec_from_file_location("SEC", os.path.join(HERE, "STATE_EXIT_2026-11-21_common.py"))
C = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(C)
MIN, HR = C.MIN, C.HR
TPGRID = {"FB": (3.0, 4.0, 6.0, 7.0, 8.0), "HT": (3.0, 4.0, 4.5, 6.0), "EXH": (5.0, 6.0, 7.0, 8.0, 9.0)}


# ───────────────────────────── indicators ─────────────────────────────
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.maximum(d, 0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.maximum(-d, 0)).ewm(alpha=1 / n, adjust=False).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, 100 - 100 / (1 + up / dn))
    r[:n] = np.nan
    return r


def adx_di(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    up = h - np.r_[h[0], h[:-1]]; dn = np.r_[l[0], l[:-1]] - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = lambda x: np.array(pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values, dtype=float)
    atr = a(tr)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * a(pdm) / atr; mdi = 100 * a(mdm) / atr
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = a(np.nan_to_num(dx))
    out = [pdi - mdi, adx, atr]
    for z in out:
        z[: 2 * n] = np.nan
    return out


def bars(g, P):
    """complete clock-aligned P-minute bars from the 1m grid → dict + 'avail' = minute index at which each bar is known (its last minute)."""
    t = g["t"]; bid = t // (P * MIN)
    u, first, cnt = np.unique(bid, return_index=True, return_counts=True)
    ok = cnt == P
    u, first = u[ok], first[ok]
    last = first + P - 1
    o = g["o"][first]; c = g["c"][last]
    h = np.array([g["h"][a:a + P].max() for a in first]); l = np.array([g["l"][a:a + P].min() for a in first])
    qv = np.array([g["qv"][a:a + P].sum() for a in first]); tb = np.array([g["tbq"][a:a + P].sum() for a in first])
    return dict(o=o, h=h, l=l, c=c, qv=qv, tb=tb, avail=last)


def to_min(n, avail, vals):
    """map bar series to minutes: minute k gets the latest bar with avail ≤ k (NaN before the first)."""
    j = np.searchsorted(avail, np.arange(n), "right") - 1
    out = np.where(j >= 0, vals[np.maximum(j, 0)], np.nan) if len(avail) else np.full(n, np.nan)
    return out


def roll_sum(x, w):
    cs = np.r_[0.0, np.cumsum(x)]
    out = np.full(len(x), np.nan)
    out[w - 1:] = cs[w:] - cs[:-w]
    return out


def lag(x, n):
    y = np.full(len(x), np.nan); y[n:] = x[:-n]; return y


def feats(g, b, fl, dirn, te_i, i0):
    """feature matrix for every minute of grid g (dict of arrays, length n). b = BTC grid (aligned), fl = fill dict."""
    n = len(g["t"]); c = g["c"]; qv = g["qv"]; tbq = g["tbq"]
    F = {}
    for w in (5, 15, 60):
        F[f"r{w}"] = (c / lag(c, w) - 1) * 100
    F["acc15"] = F["r15"] - lag(F["r15"], 15)
    m5, m60 = roll_sum(qv, 5) / 5, roll_sum(qv, 60) / 60
    with np.errstate(divide="ignore", invalid="ignore"):
        F["vr5_60"] = m5 / m60
        F["vr15_240"] = (roll_sum(qv, 15) / 15) / (roll_sum(qv, 240) / 240)
        for w in (5, 15, 60):
            F[f"tb{w}"] = roll_sum(tbq, w) / roll_sum(qv, w)
        rng = (g["h"] - g["l"]) / c * 100
        F["tr15_60"] = (roll_sum(rng, 15) / 15) / (roll_sum(rng, 60) / 60)
    pre = qv[max(0, te_i - 1440):te_i]
    F["vr60_pre24"] = (roll_sum(qv, 60)) / (pre.sum() / max(len(pre), 1) * 60 + 1e-12)
    F["tb15_chg"] = F["tb15"] - (F["tb15"][te_i] if te_i < n else np.nan)
    # 5m
    B5 = bars(g, 5)
    if len(B5["c"]) > 30:
        r5 = rsi(B5["c"]); di, adx, atr = adx_di(B5["h"], B5["l"], B5["c"])
        e5, e8, e13, e20, e50 = (ema(B5["c"], k) for k in (5, 8, 13, 20, 50))
        F["rsi_5m"] = to_min(n, B5["avail"], r5)
        F["rsi_5m_turn"] = to_min(n, B5["avail"], r5 - lag(r5, 3))
        F["di_5m"] = to_min(n, B5["avail"], di)
        F["adx_5m"] = to_min(n, B5["avail"], adx)
        F["adxd_5m"] = to_min(n, B5["avail"], adx - lag(adx, 3))
        F["atr_5m"] = to_min(n, B5["avail"], atr / B5["c"] * 100)
        for k_, e in (("e5", e5), ("e13", e13), ("e20", e20), ("e50", e50)):
            F[f"d{k_}_5m"] = to_min(n, B5["avail"], (B5["c"] / e - 1) * 100)
        F["g5_8_5m"] = to_min(n, B5["avail"], (e5 / e8 - 1) * 100)
        F["g5_20_5m"] = to_min(n, B5["avail"], (e5 / e20 - 1) * 100)
        cross = dirn * (e5 - e8) < 0
        F["x_ema58"] = to_min(n, B5["avail"], (cross & ~lag(cross.astype(float), 1).astype(bool) & (dirn * (e5 - e13) < 0)).astype(float))
        q12 = roll_sum(B5["qv"], 12)
        with np.errstate(divide="ignore", invalid="ignore"):
            F["voltr_5m"] = to_min(n, B5["avail"], q12 / lag(q12, 12))
            v20 = lag(roll_sum(B5["qv"], 20) / 20, 1)
            against = (B5["c"] < B5["o"]) if dirn > 0 else (B5["c"] > B5["o"])
            F["climax_bar"] = to_min(n, B5["avail"], (against & (B5["qv"] >= 2 * v20)).astype(float))
            F["climax_ext"] = to_min(n, B5["avail"], B5["h"] if dirn > 0 else B5["l"])
        # consecutive lower highs (long) / higher lows (short) on closed 5m bars
        ex = B5["h"] if dirn > 0 else -B5["l"]
        lower = np.r_[False, ex[1:] < ex[:-1]]
        run = np.zeros(len(lower));
        for i in range(1, len(lower)):
            run[i] = run[i - 1] + 1 if lower[i] else 0
        F["lh_run_5m"] = to_min(n, B5["avail"], run)
    # 15m
    B15 = bars(g, 15)
    if len(B15["c"]) > 30:
        r15 = rsi(B15["c"])
        F["rsi_15m"] = to_min(n, B15["avail"], r15)
        F["rsi_15m_d1"] = to_min(n, B15["avail"], r15 - lag(r15, 1))
        F["rsi_15m_d2"] = to_min(n, B15["avail"], lag(r15, 1) - lag(r15, 2))
        F["de20_15m"] = to_min(n, B15["avail"], (B15["c"] / ema(B15["c"], 20) - 1) * 100)
        F["de50_15m"] = to_min(n, B15["avail"], (B15["c"] / ema(B15["c"], 50) - 1) * 100)
    # 1h
    B60 = bars(g, 60)
    if len(B60["c"]) > 5:
        F["c1h"] = to_min(n, B60["avail"], B60["c"])
        F["de20_1h"] = to_min(n, B60["avail"], (B60["c"] / ema(B60["c"], 20) - 1) * 100)
        F["rsi_1h"] = to_min(n, B60["avail"], rsi(B60["c"]))
    B240 = bars(g, 240)
    F["c4h"] = to_min(n, B240["avail"], B240["c"]) if len(B240["c"]) else np.full(n, np.nan)
    # vol decay (rolling 1h quote volume ÷ its max since the spike, else since te − 24 h)
    r1 = roll_sum(qv, 60)
    s0 = fl["spike_i"] if fl["spike_i"] is not None else max(0, te_i - 1440)
    run = np.full(n, np.nan)
    if s0 < n:
        rr = np.nan_to_num(r1[s0:]); run[s0:] = np.maximum.accumulate(rr)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["vd_now"] = r1 / run
    # spike VWAP (typical price weighted by quote volume, from the spike bar)
    if fl["spike_i"] is not None:
        tp = (g["h"] + g["l"] + c) / 3
        w = np.where(np.arange(n) >= fl["spike_i"], qv, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            vw = np.cumsum(tp * w) / np.cumsum(w)
        F["dvwap"] = (c / vw - 1) * 100
        F["dvwap_1h"] = (F.get("c1h", c) / vw - 1) * 100
        if fl.get("base") == fl.get("base"):
            F["dbase"] = (c / fl["base"] - 1) * 100
    if fl.get("H") == fl.get("H") and fl.get("H"):
        F["dH"] = (c / fl["H"] - 1) * 100
    # BTC
    if b is not None:
        bc = b["c"]
        for w in (5, 15, 60):
            F[f"btc_r{w}"] = (bc / lag(bc, w) - 1) * 100
        BB = bars(b, 5)
        if len(BB["c"]) > 30:
            F["btc_rsi_5m"] = to_min(n, BB["avail"], rsi(BB["c"]))
        BH = bars(b, 60)
        if len(BH["c"]) > 20:
            F["btc_de20_1h"] = to_min(n, BH["avail"], (BH["c"] / ema(BH["c"], 20) - 1) * 100)
    return F


def met_feats(t, M, te, n):
    out = {}
    if M is None or not len(M["t"]):
        return out
    j = np.searchsorted(M["t"], t, "right") - 1            # create_time ≤ bar open t_k
    ok = j >= 0
    oi = np.where(ok, M["oi"][np.maximum(j, 0)], np.nan)
    j60 = np.searchsorted(M["t"], t - HR, "right") - 1
    oi60 = np.where(j60 >= 0, M["oi"][np.maximum(j60, 0)], np.nan)
    je = np.searchsorted(M["t"], te, "right") - 1
    oie = M["oi"][je] if je >= 0 else np.nan
    tpe = M["top_pos"][je] if je >= 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["oi_chg_1h"] = (oi / oi60 - 1) * 100
        out["oi_chg_in"] = (oi / oie - 1) * 100
        out["top_pos_chg_in"] = np.where(ok, M["top_pos"][np.maximum(j, 0)], np.nan) - tpe
        out["taker_ls"] = np.where(ok, M["taker_ls"][np.maximum(j, 0)], np.nan)
    return out


# ───────────────────────────── per fill ─────────────────────────────
def one(fl, lagcheck=False, rng=None):
    p = fl["pair"]; te = int(fl["te"]); hold = int(fl["hold_h"]) * HR; dirn = int(fl["dirn"])
    a = te - 26 * HR
    sp = fl["spike_ms"]
    if sp == sp:
        a = min(a, int(sp) - HR)
    a = a // MIN * MIN
    bnd = te + hold + 10 * MIN
    raw = C.k1m(p, a, bnd)
    if raw is None or not len(raw["t"]):
        return None, "no data"
    g = C.grid(raw, a, min(bnd, int(raw["t"][-1]) + MIN))
    T = g["t"]
    i0 = int(np.searchsorted(T, (te // MIN) * MIN + (MIN if fl["start"] == "next" else 0)))
    end_t = te + hold
    end_i = int(np.searchsorted(T, end_t, "left"))
    if i0 >= len(T) - 1:
        return None, "path ends before entry"
    win = g["present"][i0:end_i]
    cover = float(win.mean()) if len(win) else 0.0
    is_open = end_i > len(T) - 1 and fl["kind"] in ("CASE", "REAL")
    end_i = min(end_i, len(T))
    E = float(fl["E"])
    fs_t, fs_r = C.funding(p)
    Fc = C.fund_cum(T, g["o"], E, dirn, fs_t, fs_r, te)
    spike_i = None
    base = np.nan
    if sp == sp:
        so = int(sp) if fl["sleeve"] != "FB" or fl["kind"] == "BT" else int(sp) - 300_000
        spike_i = int(np.searchsorted(T, so))
        bi = int(np.searchsorted(T, so - 26 * MIN))
        if 0 <= bi < len(T):
            base = g["c"][bi]
    if spike_i is not None and spike_i >= len(T):
        spike_i = None
    H = fl.get("H")
    if (H != H or not H) and fl["sleeve"] == "HT":                # cases: spike high = max high of the 6 h before the fill
        j6 = int(np.searchsorted(T, te - 6 * HR)); H = float(g["h"][j6:i0].max()) if i0 > j6 else np.nan
    W = C.walk(g, i0, end_i, E, dirn, fl["tp_px"], fl["sl_px"], Fc, fl["slip"], fl["entry_cost"], fl["stop_mode"], fl["sl_pct"])
    if W is None:
        return None, "empty walk"
    # parity mode (paper costs, price stop, booked funding off) for real rows
    Wp = C.walk(g, i0, end_i, E, dirn, fl["tp_px"], fl["sl_px"], np.zeros(len(T)), 0.0, 0.045, "price", fl["sl_pct"])
    alts = {}
    for x in TPGRID[fl["sleeve"]]:
        if fl["sleeve"] == "HT":
            tpp = E * (100 + x + C.TAKER) / (100 - C.MAKER)
        elif fl["sleeve"] == "EXH":
            tpp = E * (1 - min(x, (1 - fl["tp_px"] / E) * 100) / 100)
        else:
            tpp = E * (1 - x / 100)
        w = C.walk(g, i0, end_i, E, dirn, tpp, fl["sl_px"], Fc, fl["slip"], fl["entry_cost"], fl["stop_mode"], fl["sl_pct"])
        alts[f"tp{x:g}"] = w["net"]
    if fl["sleeve"] == "HT":                       # WILLY's late TP (+0.25 net from minute 30) — HT runs it OFF today (DL 587)
        i30 = i0 + 30
        if W["j"] < i30 or i30 >= end_i:
            alts["late0.25"] = W["net"]
        else:
            lp = E * (100 + 0.25 + C.TAKER) / (100 - C.MAKER)
            w = C.walk(g, i30, end_i, E, dirn, lp, fl["sl_px"], Fc, fl["slip"], fl["entry_cost"], fl["stop_mode"], fl["sl_pct"])
            alts["late0.25"] = w["net"]
    j = W["j"]
    hi, lo, c = g["h"], g["l"], g["c"]
    fav = (hi if dirn > 0 else lo)
    pk_series = np.maximum.accumulate(dirn * (fav[i0:j + 1] / E - 1) * 100)
    peak = float(pk_series[-1]); kpk = int(np.argmax(pk_series)) + i0
    out = dict(fid=fl["fid"], base_net=W["net"], base_why=W["why"], base_xt=int(T[j]), base_gross=W["gross"], base_fund=W["fund"],
               par_net=Wp["net"], par_why=Wp["why"], peak=peak, peak_min=(kpk - i0), hold_min=(j - i0), cover=cover, is_open=is_open,
               base_px=base, H_used=H, **alts)
    # decision rows
    ks = np.arange(i0, j)
    ks = ks[ks + 1 <= j]
    if not len(ks):
        return dict(out=out, dec=None, path=None), "ok"
    pnl = dirn * (c[ks] / E - 1) * 100
    pk = pk_series[ks - i0]
    keep = (pnl >= 1.0) | (pk >= 2.0)
    Xn = g["o"][ks + 1] * (1 - dirn * fl["slip"] / 100)
    now = dirn * (Xn / E - 1) * 100 - fl["entry_cost"] - C.TAKER * Xn / E + Fc[ks]
    b = None
    rb = C.k1m("BTCUSDT", a, T[-1] + MIN)
    if rb is not None and len(rb["t"]):
        b = C.grid(rb, a, T[-1] + MIN)
        if len(b["t"]) != len(T):
            b = None
    fl2 = dict(spike_i=spike_i, base=base, H=H)
    Fe = feats(g, b, fl2, dirn, i0, i0)
    M = C.metrics(p, a, T[-1] + MIN)
    Fm = met_feats(T, M, te, len(T))
    Fe.update(Fm)
    # in-trade running quantities
    rs = Fe.get("rsi_5m", np.full(len(T), np.nan))
    seg = rs.copy(); seg[:i0] = np.nan
    rmax = pd.Series(seg).cummax().values; rmin = pd.Series(seg).cummin().values
    ext_px = np.full(len(T), np.nan)
    ext_px[i0:j + 1] = (np.maximum.accumulate(hi[i0:j + 1]) if dirn > 0 else np.minimum.accumulate(lo[i0:j + 1]))
    last_rate = np.full(len(T), np.nan)
    if len(fs_t):
        jj = np.searchsorted(fs_t, T, "right") - 1
        last_rate = np.where(jj >= 0, fs_r[np.maximum(jj, 0)], np.nan) * 100
    Fe["fund_recv"] = -dirn * last_rate              # % per settlement the POSITION receives (+) / pays (−)
    Fe["fund_cum"] = Fc
    if "c4h" in Fe:
        Fe["d4h_entry"] = dirn * (Fe["c4h"] / E - 1) * 100      # last closed 4h close vs entry, + = in our favour
    d = pd.DataFrame({k_: np.asarray(v, float)[ks] for k_, v in Fe.items() if k_ not in ("c1h", "c4h", "climax_ext")})
    d["rsi_max_in"] = rmax[ks]; d["rsi_min_in"] = rmin[ks]
    if "climax_ext" in Fe:
        d["climax_near"] = (dirn * (Fe["climax_ext"][ks] / ext_px[ks] - 1) * 100 >= -1.0).astype(float)
    d["below_peak"] = (dirn * (c[ks] / ext_px[ks] - 1) * 100 < 0).astype(float)
    d.insert(0, "fid", fl["fid"]); d.insert(1, "k", ks - i0)
    d["pnl"] = pnl; d["peak"] = pk; d["dd"] = pk - pnl; d["min_in"] = ks - i0
    d["min_since_pk"] = ks - (np.maximum.accumulate(np.where(np.r_[True, np.diff(pk_series) > 0], np.arange(len(pk_series)), 0)) + i0)[ks - i0]
    d["now"] = now; d["label"] = now - W["net"]
    d = d[keep].reset_index(drop=True)
    for c_ in d.columns:
        if d[c_].dtype == np.float64 and c_ not in ("label", "now", "pnl", "peak"):
            d[c_] = d[c_].astype("float32")
    # path around the peak for the event study (5-min stride, −6 h … +24 h; gross %), plus a few state series
    path = None
    if peak >= 2.0:
        rel = np.arange(-72, 289) * 5
        ix = kpk + rel
        okx = (ix >= i0) & (ix < len(T))
        pp = np.full(len(rel), np.nan); pp[okx] = dirn * (c[ix[okx]] / E - 1) * 100
        sv = {}
        for nm in ("rsi_5m", "voltr_5m", "di_5m", "tb15", "g5_8_5m", "oi_chg_1h", "dvwap", "vd_now", "rsi_15m"):
            if nm in Fe:
                s = np.full(len(rel), np.nan); s[okx] = np.asarray(Fe[nm], float)[ix[okx]]; sv[nm] = s
        path = dict(fid=fl["fid"], rel=rel, pnl=pp, after_exit=(ix > j), **sv)
    res = dict(out=out, dec=d, path=path)
    if lagcheck:
        res["lag"] = lag_check(g, b, fl2, dirn, i0, j, Fe, M, T, te, rng)
    return res, "ok"


def lag_check(g, b, fl2, dirn, i0, j, Fe, M, T, te, rng):
    bad = []
    for k in rng.integers(i0, max(j, i0 + 1), 3):
        k = int(k)
        gt = {kk: v[:k + 1] for kk, v in g.items()}
        bt = {kk: v[:k + 1] for kk, v in b.items()} if b is not None else None
        Ft = feats(gt, bt, dict(fl2, spike_i=(fl2["spike_i"] if fl2["spike_i"] is not None and fl2["spike_i"] <= k else None)), dirn, i0, i0)
        Ft.update(met_feats(T[:k + 1], M, te, k + 1))
        for nm, v in Ft.items():
            if nm in ("c1h", "c4h", "climax_ext"):
                continue
            a_, b_ = np.asarray(Fe[nm], float)[k], np.asarray(v, float)[k]
            if not ((np.isnan(a_) and np.isnan(b_)) or abs(a_ - b_) <= 1e-6 * max(1, abs(a_))):
                bad.append((nm, k, a_, b_))
    return bad


def work(args):
    fl, lc = args
    rng = np.random.default_rng(int(fl["fid"]))
    try:
        r, why = one(fl, lc, rng)
    except Exception as e:
        return fl["fid"], None, f"error {type(e).__name__}: {e}"
    return fl["fid"], r, why


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lagcheck", action="store_true"); ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--only", default=""); ap.add_argument("--n", type=int, default=0)
    A = ap.parse_args()
    Z = pd.read_csv(os.path.join(C.D, "fills.csv"))
    if A.only:
        Z = Z[Z.sleeve.isin(A.only.split(","))]
    if A.n:
        Z = Z.groupby(["sleeve", "kind"], group_keys=False).head(A.n)
    recs = Z.to_dict("records")
    outs, decs, paths, lagbad, fails = [], {}, [], [], []
    with mp.Pool(A.procs) as pool:
        for i, (fid, r, why) in enumerate(pool.imap_unordered(work, [(x, A.lagcheck) for x in recs], chunksize=4)):
            if r is None:
                fails.append((fid, why)); continue
            outs.append(r["out"])
            if r["dec"] is not None and len(r["dec"]):
                decs.setdefault(Z.set_index("fid").loc[fid, "sleeve"], []).append(r["dec"])
            if r["path"] is not None:
                paths.append(r["path"])
            if A.lagcheck:
                lagbad += [(fid,) + x for x in r.get("lag", [])]
            if i % 200 == 0:
                print(time.strftime("%H:%M:%S"), i, len(recs), flush=True)
    O = Z.merge(pd.DataFrame(outs), on="fid", how="left")
    suffix = "_lag" if A.lagcheck else ""
    if A.lagcheck:
        print("lag check: mismatches", len(lagbad), "over", len(outs), "fills × 3 minutes")
        for x in lagbad[:30]:
            print("  ", x)
        if not A.n:
            pass
        return
    O.to_csv(os.path.join(C.D, "fills_out.csv"), index=False)
    for s, L in decs.items():
        pd.concat(L, ignore_index=True).to_pickle(os.path.join(C.D, f"dec_{s}.pkl"))
    pd.to_pickle(paths, os.path.join(C.D, "paths.pkl"))
    print("fails", len(fails)); print(pd.Series([w for _, w in fails]).value_counts().head(10))


if __name__ == "__main__":
    main()
