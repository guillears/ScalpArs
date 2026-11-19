#!/usr/bin/env python3
"""FRENZY LONG SEARCH — FAST detector re-run on 5m bars over the FULL local universe (delisted coins included, 2023 → 2026-11) to test
survivorship / period generality of the 1m FAST result (PREREG addendum A, check added after the 1m read; descriptive robustness check).
D3_5m / D5_5m: 5m bar return ≥ +3 / +5 % ∧ bar quote volume ≥ 10× the median 5m quote volume of the prior 24 h ∧ 24 h vol ≥ $20 M;
6 h debounce; entry = next 5m open (≥ 5 min latency, worse than the 1m detector) + 0.10 % slip; FLAG exits.
Output reports/FRENZY_LONG_SEARCH_data/fast5_rows.pkl."""
import glob, os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_flag as F   # noqa: E402
TB = F.TB; D = F.D; H1 = F.H1
SPK = None


def work(pair):
    x = TB.load(pair)
    if x is None:
        return []
    t, o, h, l, c, v, qv = x
    n = len(t); realb = qv > 0
    if n < 600:
        return []
    r1 = c / np.r_[np.nan, c[:-1]] - 1
    med = pd.Series(np.where(realb, qv, np.nan)).rolling(288, min_periods=144).median().shift(1).values
    q24 = pd.Series(qv).rolling(288, min_periods=144).sum().values
    creal = np.r_[0, np.cumsum(realb)]
    ft, fc, fm = TB.funding_for(pair)
    rates = np.diff(fc) * 100 if len(ft) else np.zeros(0)
    spikes = SPK.get(pair, np.zeros(0, "int64"))
    out = []
    for thr, name in ((0.03, "D3_5m"), (0.05, "D5_5m")):
        sig = np.where((r1 >= thr) & (qv >= 10 * med) & (q24 >= 20e6) & realb)[0]
        last = -10**18
        for i in sig:
            if t[i] - last < 6 * H1:
                continue
            last = t[i]; ie = i + 1
            if ie >= n - 1 or not realb[ie]:
                continue
            E0 = o[ie]
            near = spikes[(spikes >= t[i] - 2 * H1) & (spikes <= t[i] + 2 * H1)]
            rr = dict(pair=pair, det=name, entry_ms=int(t[ie]), r1=r1[i] * 100, became_frenzy=bool(len(near)),
                      first_ms=int(t[np.argmax(realb)]))
            for k, (tp, sl, nb, tr) in F.EXITS.items():
                g, kind, xi = F.sim(o, h, l, c, ie, E0, tp, sl, nb, tr)
                x_ms = int(t[xi]) + F.BAR
                fcost = 0.0
                if len(ft):
                    a_ = int(np.searchsorted(ft, int(t[ie]), side="right")); b_ = int(np.searchsorted(ft, x_ms, side="right"))
                    fcost = float(rates[a_:b_].sum())
                cov = (creal[xi + 1] - creal[ie]) / max(1, xi - ie + 1)
                rr[k + "_g"], rr[k + "_k"], rr[k + "_x"], rr[k + "_f"] = g, kind, x_ms, fcost
                rr[k + "_complete"] = bool((kind != 0 or xi - ie + 1 >= nb) and cov >= 0.95)
            rr["delisted_after"] = bool(t[-1] < pd.Timestamp("2026-11-10").value // 10**6 and int(t[ie]) + 72 * H1 > t[-1] - 30 * 86400000)
            out.append(rr)
    return out


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
    print("pairs", len(pairs), flush=True)
    rows = []
    with mp.Pool(9, initializer=init, initargs=(spk,)) as pool:
        for k, res in enumerate(pool.imap_unordered(work, pairs)):
            rows += res
            if k % 200 == 0:
                print(time.strftime("%H:%M:%S"), k, len(rows), flush=True)
    pickle.dump(pd.DataFrame(rows), open(os.path.join(D, "fast5_rows.pkl"), "wb"))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
