#!/usr/bin/env python3
"""FRENZY LONG SEARCH — FAST 1m detector (PREREG addendum A). Pairs with local 1m klines (reports/backtest_cache/k1m, 2026-01-22 →
2026-10-02). D3 / D5: 5-min return ≥ +3 / +5 % ∧ last-5-min quote volume ≥ 10× the median 5-min quote volume of the prior 24 h ∧
24 h quote volume ≥ $20 M; 6 h debounce. Entry = open of the 2nd 1m bar after the signal minute (≥ 60 s latency) + 0.10 % slip.
Exits = FLAG exits on 1m bars. became_frenzy = a validated-walk spike on the pair within ±2 h (hindsight tag, descriptive only).
Output reports/FRENZY_LONG_SEARCH_data/fast_rows.pkl."""
import glob, os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_flag as F   # noqa: E402
TB = F.TB
D = F.D
K1 = os.path.join(R, "reports", "backtest_cache", "k1m")
M1, H1 = 60_000, 3_600_000
EX1 = {k: (tp, sl, nb * 5, tr) for k, (tp, sl, nb, tr) in F.EXITS.items()}
SPK = None


def work(pair):
    f = os.path.join(K1, pair + ".csv")
    try:
        d = pd.read_csv(f, usecols=["open_time", "o", "h", "l", "c", "qvol"]).drop_duplicates("open_time").sort_values("open_time")
    except Exception:
        return []
    if len(d) < 3000:
        return []
    d = d[d.open_time % M1 == 0]
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, M1, dtype="int64")
    d = d.set_index("open_time").reindex(grid)
    realb = d.c.notna().values
    c = d.c.ffill().values.astype(float)
    o = np.where(realb, d.o.values, c).astype(float); h = np.where(realb, d.h.values, c).astype(float); l = np.where(realb, d.l.values, c).astype(float)
    qv = np.nan_to_num(d.qvol.values.astype(float)); t = grid
    n = len(t)
    creal = np.r_[0, np.cumsum(realb)]
    r5 = c / np.r_[np.full(5, np.nan), c[:-5]] - 1
    q5 = pd.Series(qv).rolling(5, min_periods=5).sum().values
    med = pd.Series(q5[::5]).rolling(288, min_periods=144).median().shift(1)
    med_full = np.repeat(med.values, 5)[:n]
    q24 = pd.Series(qv).rolling(1440, min_periods=720).sum().values
    ft, fc, _ = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    spikes = SPK.get(pair, np.zeros(0, "int64"))
    out = []
    for thr, name in ((0.03, "D3"), (0.05, "D5")):
        sig = np.where((r5 >= thr) & (q5 >= 10 * med_full) & (q24 >= 20e6))[0]
        last = -10**18
        for i in sig:
            if t[i] - last < 6 * H1:
                continue
            last = t[i]
            ie = i + 2
            if ie >= n - 1 or not (realb[i] and realb[ie]):
                continue
            E0 = o[ie]
            rr = dict(pair=pair, det=name, sig_ms=int(t[i]) + M1, entry_ms=int(t[ie]), r5=r5[i] * 100, volx=q5[i] / med_full[i])
            near = spikes[(spikes >= t[i] - 2 * H1) & (spikes <= t[i] + 2 * H1)]
            rr["became_frenzy"] = bool(len(near)); rr["spike_ms"] = int(near[0]) if len(near) else -1
            for k, (tp, sl, nb, tr) in EX1.items():
                g, kind, xi = F.sim(o, h, l, c, ie, E0, tp, sl, nb, tr)
                x_ms = int(t[xi]) + M1
                fcost = 0.0
                if len(ft):
                    a_ = int(np.searchsorted(ft, int(t[ie]), side="right")); b_ = int(np.searchsorted(ft, x_ms, side="right"))
                    fcost = float(rates[a_:b_].sum())
                rr[k + "_g"], rr[k + "_k"], rr[k + "_x"], rr[k + "_f"] = g, kind, x_ms, fcost
                cov = (creal[xi + 1] - creal[ie]) / max(1, xi - ie + 1)
                rr[k + "_complete"] = bool((kind != 0 or xi - ie + 1 >= nb) and cov >= 0.95)
            out.append(rr)
    return out


def init(spk):
    global SPK
    SPK = spk


def main():
    m = pd.read_csv(os.path.join(R, "reports", "UNIFIED_LONG_data", "eps_meta.csv"))
    spk = {p: np.sort(g.spike_ms.values.astype("int64")) for p, g in m.groupby("pair")}
    pairs = sorted(os.path.basename(x)[:-4] for x in glob.glob(os.path.join(K1, "*.csv")))
    pairs = [p for p in pairs if p.isascii() and p not in ("BTCUSDT", "ETHUSDT")]
    rows = []
    with mp.Pool(9, initializer=init, initargs=(spk,)) as pool:
        for k, res in enumerate(pool.imap_unordered(work, pairs)):
            rows += res
            if k % 100 == 0:
                print(time.strftime("%H:%M:%S"), k, len(rows), flush=True)
    df = pd.DataFrame(rows)
    pickle.dump(df, open(os.path.join(D, "fast_rows.pkl"), "wb"))
    print("rows", len(df))


if __name__ == "__main__":
    main()
