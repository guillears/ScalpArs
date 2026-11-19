#!/usr/bin/env python3
"""FRENZY LONG SEARCH — FLAG family (PREREG addendum A, reports/FRENZY_LONG_SEARCH_2026-11-19_PREREG.md): buy AT FIRST DETECTION.
Detection = the first 5m close at which the engine both has the verified spike and has the pair shortlisted:
det = max(spike close, first_seen_ms). Entry = the open of the next 5m bar (≈ close + 8 s live latency; 0.10 % slip covers the drift).
Detection lag: base = the lowest close in the 6 h up to the spike bar; moved % = detection close ÷ base − 1; lag = det − base bar close.
Exits (frozen, long-shaped): Q3 TP 3 maker / SL 4 / 4 h · Q5 TP 5 / SL 5 / 24 h · TR arm +3 %, trail 3 % off the high, SL 5, 24 h ·
T1 1 h time (no stop) · T4 4 h time (no stop) · H72 72 h hold (no stop). HB = hindsight bound: the same Q3 / Q5 / TR entered at the
open after the base bar (not reachable). Output reports/FRENZY_LONG_SEARCH_data/flag_rows.pkl."""
import os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_build as B   # noqa: E402  (sets the k5m / funding sources)
TB = B.TB
D = B.D; BAR, H1, DAY = B.BAR, B.H1, B.DAY
EXITS = {"Q3": (3.0, 4.0, 48, None), "Q5": (5.0, 5.0, 288, None), "TR": (None, 5.0, 288, (3.0, 3.0)),
         "T1": (None, None, 12, None), "T4": (None, None, 48, None), "H72": (None, None, 864, None)}


def sim(o, h, l, c, i0, E0, tp, sl, nb, trail):
    """long exit on bars i0.. ; returns (gross %, kind, exit bar index); kind 0 time 1 stop 2 tp 3 trail"""
    n = len(c); end = min(n - 1, i0 + nb - 1)
    peak = E0; armed = False
    for i in range(i0, end + 1):
        if sl is not None and l[i] <= E0 * (1 - sl / 100):
            return (min(E0 * (1 - sl / 100), o[i]) / E0 - 1) * 100, 1, i
        if trail is not None and armed and l[i] <= peak * (1 - trail[1] / 100):
            return (min(peak * (1 - trail[1] / 100), o[i]) / E0 - 1) * 100, 3, i
        if tp is not None and h[i] >= E0 * (1 + tp / 100) * 1.0005:
            return tp, 2, i
        peak = max(peak, h[i])
        if trail is not None and peak >= E0 * (1 + trail[0] / 100):
            armed = True
    return (max(c[end], E0 * 0.05) / E0 - 1) * 100, 0, end


def work(args):
    pair, eps = args
    x = TB.load(pair)
    if x is None:
        return []
    t, o, h, l, c, v, qv = x
    n = len(t); real = qv > 0
    q24 = pd.Series(qv).rolling(288, min_periods=1).sum().values
    ft, fc, fmonths = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    out = []
    for e in eps:
        sp = int(e["spike_ms"]); i_sp = int(np.searchsorted(t, sp - BAR))
        if i_sp >= n or t[i_sp] != sp - BAR or i_sp < 72 or e["first_seen_ms"] < 0:
            continue
        det = max(sp, int(e["first_seen_ms"]))
        i_det = int(np.searchsorted(t, det - BAR))           # bar that closes at det
        if i_det >= n - 2 or q24[i_det] < 20e6 or not real[i_det + 1]:
            continue
        w0 = i_sp - 72
        ib = w0 + int(np.argmin(c[w0:i_sp + 1])); base = c[ib]
        i_ent = i_det + 1; E0 = o[i_ent]
        r = dict(pair=pair, spike_ms=sp, det_ms=det, entry_ms=int(t[i_ent]), late_min=(det - sp) / 60000,
                 moved=(c[i_det] / base - 1) * 100, lag_min=(det - (int(t[ib]) + BAR)) / 60000,
                 gain1h=(c[i_det] / c[max(0, i_sp - 12)] - 1) * 100, spike_bar=(c[i_sp] / o[i_sp] - 1) * 100,
                 vol24=q24[i_det], entry_px=E0)
        if len(ft):
            jf = int(np.searchsorted(ft, det, side="right")) - 1
            r["fund_last"] = rates[jf] if jf >= 0 else np.nan
        for k, (tp, sl, nb, tr) in EXITS.items():
            g, kind, xi = sim(o, h, l, c, i_ent, E0, tp, sl, nb, tr)
            x_ms = int(t[xi]) + BAR
            fcost = 0.0
            if len(ft):
                a_ = int(np.searchsorted(ft, int(t[i_ent]), side="right")); b_ = int(np.searchsorted(ft, x_ms, side="right"))
                fcost = float(rates[a_:b_].sum())
            r[k + "_g"], r[k + "_k"], r[k + "_x"], r[k + "_f"] = g, kind, x_ms, fcost
            r[k + "_complete"] = bool(kind != 0 or xi - i_ent + 1 >= nb)
        # hindsight bound: enter at the open after the base bar
        if ib + 1 < n:
            Eb = o[ib + 1]
            for k in ("Q3", "Q5", "TR"):
                tp, sl, nb, tr = EXITS[k]
                g, kind, xi = sim(o, h, l, c, ib + 1, Eb, tp, sl, nb, tr)
                r["HB_" + k + "_g"], r["HB_" + k + "_k"] = g, kind
        # 1 h path vs entry for the event study (−2 h … +24 h, 5m steps)
        idx = i_ent + np.arange(-24, 289)
        okk = (idx >= 0) & (idx < n)
        p = np.full(len(idx), np.nan, np.float32); p[okk] = (c[idx[okk]] / E0 - 1) * 100
        r["path"] = p
        r["reach24"] = (h[i_ent:min(n, i_ent + 288)].max() / E0 - 1) * 100
        r["mae24"] = (l[i_ent:min(n, i_ent + 288)].min() / E0 - 1) * 100
        out.append(r)
    return out


def main():
    E = pickle.load(open(os.path.join(B.UD, "eps_all.pkl"), "rb"))
    by = {}
    for e in E:
        by.setdefault(e["pair"], []).append({k: e[k] for k in ("spike_ms", "first_seen_ms", "off_ms")})
    rows = []
    with mp.Pool(9) as pool:
        for k, res in enumerate(pool.imap_unordered(work, list(by.items()))):
            rows += res
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), k, len(rows), flush=True)
    df = pd.DataFrame(rows)
    pickle.dump(df, open(os.path.join(D, "flag_rows.pkl"), "wb"))
    print("rows", len(df))


if __name__ == "__main__":
    main()
