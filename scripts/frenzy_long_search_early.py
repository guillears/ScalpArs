#!/usr/bin/env python3
"""FRENZY LONG SEARCH — EARLIER-DETECTOR grid (PREREG addendum B). 5m bars, full local universe (delisted incl.), 2022-11 → 2026-11.
27 triggers (window 1/3/6 bars × return ≥ 2/3/5 % × window quote volume ≥ 5/10/20× normal) × 5 long exits; vectorised exits;
random-timing pool per pair-month for the family-wise null. Output reports/FRENZY_LONG_SEARCH_data/early_rows.pkl, early_pool.pkl."""
import glob, os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_flag as F   # noqa: E402
TB = F.TB; D = F.D; H1 = F.H1; BAR = F.BAR
WINS = (1, 3, 6); RETS = (2.0, 3.0, 5.0); VOLS = (5.0, 10.0, 20.0)
EXN = ("Q3", "Q5", "TR", "T1", "T4")
SPK = None
RNG = np.random.default_rng(7)


def outcomes(o, h, l, c, i0):
    """vectorised FLAG exits from entry bar i0 (entry = o[i0]); returns dict name → (gross %, kind, exit idx)"""
    n = len(c); E0 = o[i0]; res = {}
    end = min(n, i0 + 288); hh = h[i0:end]; ll = l[i0:end]; oo = o[i0:end]; cc = c[i0:end]
    for k, (tp, sl, nb, tr) in (("Q3", (3.0, 4.0, 48, None)), ("Q5", (5.0, 5.0, 288, None))):
        m = min(nb, len(cc))
        s_hit = ll[:m] <= E0 * (1 - sl / 100); t_hit = hh[:m] >= E0 * (1 + tp / 100) * 1.0005
        si = int(np.argmax(s_hit)) if s_hit.any() else 10**9; ti = int(np.argmax(t_hit)) if t_hit.any() else 10**9
        if si <= ti and si < 10**9:
            res[k] = ((min(E0 * (1 - sl / 100), oo[si]) / E0 - 1) * 100, 1, i0 + si)
        elif ti < 10**9:
            res[k] = (tp, 2, i0 + ti)
        else:
            res[k] = ((cc[m - 1] / E0 - 1) * 100, 0, i0 + m - 1)
    # trail: arm +3 off entry (peak of PRIOR bars), trail 3 % off the prior peak, SL 5, 24 h
    m = len(cc)
    pk_prev = np.maximum.accumulate(np.r_[E0, hh[:-1]])
    armed = pk_prev >= E0 * 1.03
    s_hit = ll <= E0 * 0.95; tr_hit = armed & (ll <= pk_prev * 0.97)
    si = int(np.argmax(s_hit)) if s_hit.any() else 10**9; ri = int(np.argmax(tr_hit)) if tr_hit.any() else 10**9
    if si <= ri and si < 10**9:
        res["TR"] = ((min(E0 * 0.95, oo[si]) / E0 - 1) * 100, 1, i0 + si)
    elif ri < 10**9:
        res["TR"] = ((min(pk_prev[ri] * 0.97, oo[ri]) / E0 - 1) * 100, 3, i0 + ri)
    else:
        res["TR"] = ((cc[m - 1] / E0 - 1) * 100, 0, i0 + m - 1)
    for k, nb in (("T1", 12), ("T4", 48)):
        mm = min(nb, m)
        res[k] = ((cc[mm - 1] / E0 - 1) * 100, 0, i0 + mm - 1)
    # false positive: never +3 % before −5 %
    up = hh >= E0 * 1.03; dn = ll <= E0 * 0.95
    ui = int(np.argmax(up)) if up.any() else 10**9; di = int(np.argmax(dn)) if dn.any() else 10**9
    res["fp"] = ui >= di
    res["peak24"] = (hh.max() / E0 - 1) * 100
    return res


def work(pair):
    x = TB.load(pair)
    if x is None:
        return [], []
    t, o, h, l, c, v, qv = x
    n = len(t); real = qv > 0
    if n < 3000:
        return [], []
    q24 = pd.Series(qv).rolling(288, min_periods=144).sum().values
    hq = pd.Series(qv).rolling(12, min_periods=12).sum()
    # normal hour: median of hourly sums (hour-aligned) over 240 h ending 24 h before
    hour_ix = np.where(t % H1 == H1 - BAR)[0]
    hs = pd.Series(hq.values[hour_ix], index=hour_ix)
    nrm_h = hs.rolling(240, min_periods=120).median().shift(24)
    nrm = pd.Series(np.nan, index=np.arange(n)); nrm[nrm_h.index] = nrm_h.values; nrm = nrm.ffill().values
    ft, fc, _ = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    spikes = SPK.get(pair, np.zeros(0, "int64"))
    elig = real & (q24 >= 20e6) & np.isfinite(nrm) & (nrm > 0)
    cache = {}

    def get(i):
        if i not in cache:
            r = outcomes(o, h, l, c, i + 1)
            fcs = {}
            for k in EXN:
                x_ms = int(t[r[k][2]]) + BAR
                fcost = 0.0
                if len(ft):
                    a_ = int(np.searchsorted(ft, int(t[i + 1]), side="right")); b_ = int(np.searchsorted(ft, x_ms, side="right"))
                    fcost = float(rates[a_:b_].sum())
                fcs[k] = (r[k][0], r[k][1], x_ms, fcost)
            cache[i] = (fcs, r["fp"], r["peak24"])
        return cache[i]
    rows = []
    for w in WINS:
        ret = (c / np.r_[np.full(w, np.nan), c[:-w]] - 1) * 100
        wq = pd.Series(qv).rolling(w, min_periods=w).sum().values
        volx = wq / (nrm * w / 12)
        for rt in RETS:
            for vx in VOLS:
                sig = np.where(elig & (ret >= rt) & (volx >= vx))[0]
                last = -10**18
                for i in sig:
                    if t[i] - last < 6 * H1 or i + 2 >= n or not real[i + 1]:
                        continue
                    last = t[i]
                    fcs, fp, pk = get(i)
                    sc = int(t[i]) + BAR
                    nxt = spikes[(spikes >= sc) & (spikes <= sc + 2 * H1)]
                    r = dict(pair=pair, cell=f"W{w*5}_R{int(rt)}_V{int(vx)}", sig_ms=sc, entry_ms=int(t[i + 1]), fp=fp, peak24=pk,
                             spike_after=int(nxt[0]) if len(nxt) else -1)
                    if len(nxt):   # price at today's flag entry (next open after the spike close) for the move-left read
                        j = int(np.searchsorted(t, nxt[0]))
                        r["px_ratio_flag"] = o[j] / o[i + 1] if j < n else np.nan
                    for k in EXN:
                        g, kind, x_ms, fcost = fcs[k]
                        r[k + "_g"], r[k + "_k"], r[k + "_x"], r[k + "_f"] = g, kind, x_ms, fcost
                        r[k + "_complete"] = True
                    rows.append(r)
    # random-timing pool: up to 300 eligible bars per pair-month
    pool = []
    el = np.where(elig[:-300])[0]
    if len(el):
        mon = pd.to_datetime(t[el], unit="ms").strftime("%Y-%m").values
        for mo in np.unique(mon):
            ii = el[mon == mo]
            take = RNG.choice(ii, size=min(300, len(ii)), replace=False)
            for i in take:
                fcs, fp, pk = get(int(i))
                pr = dict(pair=pair, month=mo)
                for k in EXN:
                    g, kind, x_ms, fcost = fcs[k]
                    pr[k + "_g"], pr[k + "_k"], pr[k + "_f"] = g, kind, fcost
                pool.append(pr)
    return rows, pool


def init(s):
    global SPK
    SPK = s


def main():
    m = pd.read_csv(os.path.join(R, "reports", "UNIFIED_LONG_data", "eps_meta.csv"))
    spk = {p: np.sort(g.spike_ms.values.astype("int64")) for p, g in m.groupby("pair")}
    names = set()
    for d in TB.K5M_SRC:
        names |= {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*.csv"))}
    pairs = sorted(p for p in names if p.isascii() and p.endswith("USDT") and p not in ("BTCUSDT", "ETHUSDT"))
    rows, pool = [], []
    with mp.Pool(9, initializer=init, initargs=(spk,)) as P:
        for k, (r, p) in enumerate(P.imap_unordered(work, pairs)):
            rows += r; pool += p
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), k, len(rows), len(pool), flush=True)
    pickle.dump(pd.DataFrame(rows), open(os.path.join(D, "early_rows.pkl"), "wb"))
    pickle.dump(pd.DataFrame(pool), open(os.path.join(D, "early_pool.pkl"), "wb"))
    print("rows", len(rows), "pool", len(pool))


if __name__ == "__main__":
    main()
