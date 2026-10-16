#!/usr/bin/env python3
"""ML REGIME study 2026-10-16 — day-level regime features (no lookahead), per reports/ML_REGIME_PREREG.txt section 3.
P (trend persistence) + V (volatility) families from 5m closes (caches + LITE gap fetch + reports/ml_regime/fetch top-up),
O (own rolling ML result, BT only) and L (reused reports/lite_regime/day_features.csv). Every value for day d uses only UTC days <= d-1.
Output reports/ml_regime/day_features.csv (+ pair_day_stats.pkl cache).
    venv/bin/python -W ignore scripts/study_ml_regime_features.py [--rebuild]
"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_lite_regime_features import load_daily, STABLE   # noqa: E402

OUT = os.path.join(ROOT, "reports", "ml_regime")
SRC = [os.path.join(ROOT, "reports/backtest_cache_2025/k5m_full"), os.path.join(ROOT, "reports/backtest_cache/k5m_full"),
       os.path.join(ROOT, "reports/lite_regime/fetch/k5m"), os.path.join(OUT, "fetch/k5m")]
DAY, BAR = 86_400_000, 300_000
D0, D1 = pd.Timestamp("2024-12-20"), pd.Timestamp("2026-10-16")


def load5(pair):
    parts = []
    for s in SRC:
        f = os.path.join(s, pair + ".csv")
        if os.path.exists(f):
            try:
                x = pd.read_csv(f, usecols=[0, 4]); x.columns = ["t", "c"]; parts.append(x)
            except Exception:
                pass
    if pair == "BTCUSDT":
        f = os.path.join(ROOT, "reports/backtest_cache/scout_dailyup/BTCUSDT_5m.csv")
        if os.path.exists(f):
            x = pd.read_csv(f); x.columns = ["t", "c"]; parts.append(x)
    if not parts:
        return None
    k = pd.concat(parts).dropna().drop_duplicates("t", keep="last").sort_values("t")
    k = k[k.t >= int(D0.value // 10**6) - 8 * DAY]
    return k.reset_index(drop=True)


def _ac(x):
    if len(x) < 20 or np.std(x) == 0:
        return np.nan
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


def _vr(r, q):
    if len(r) < 10 * q or np.var(r) == 0:
        return np.nan
    s = np.convolve(r, np.ones(q), "valid")
    return float(np.var(s) / (q * np.var(r)))


def day_stats(pair, btc=False):
    """per UTC day: ac5, vr3, (vr12, ac15, ft15btc), ft (alt pop follow-through %), rv5, er. Returns DataFrame indexed by day_ms."""
    k = load5(pair)
    if k is None or len(k) < 2000:
        return None
    t = k.t.values.astype(np.int64); c = np.log(k.c.values.astype(float))
    ok = np.r_[False, np.diff(t) == BAR]
    r = np.r_[np.nan, np.diff(c)]
    r[~ok] = np.nan
    day = (t // DAY) * DAY
    rows = []
    for d, idx in pd.Series(np.arange(len(t))).groupby(day):
        ii = idx.values
        x = r[ii]
        if np.isfinite(x).sum() < 250:
            continue
        x = np.nan_to_num(x)                                    # gaps -> 0 return (rare)
        o = {"day_ms": int(d), "ac5": _ac(x), "vr3": _vr(x, 3), "rv5": float(np.std(x) * 100),
             "er": float(abs(x.sum()) / np.abs(x).sum()) if np.abs(x).sum() > 0 else np.nan}
        q90 = np.quantile(x, 0.9)
        nx = [x[i + 1:i + 4].sum() for i in range(len(x) - 3) if x[i] >= q90]
        o["ft"] = float(np.mean(nx) * 100) if len(nx) >= 5 else np.nan
        if btc:
            o["vr12"] = _vr(x, 12)
            m = len(x) // 3 * 3
            r15 = x[:m].reshape(-1, 3).sum(1)
            o["ac15"] = _ac(r15)
            sd = np.std(r15)
            nx15 = [r15[i + 1] for i in range(len(r15) - 1) if r15[i] > sd]
            o["ft15"] = float(np.mean(nx15) * 1e4) if len(nx15) >= 3 else np.nan
        rows.append(o)
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("day_ms")


def _alt(p):
    try:
        s = day_stats(p)
        return p, s
    except Exception as e:
        print("ERR", p, e, flush=True)
        return p, None


def universe_by_day(days_ms):
    D = load_daily()
    QV = D.pivot(index="t", columns="pair", values="qv").sort_index()
    drop = [p for p in ("BTCUSDT",) + STABLE if p in QV.columns]
    U = {}
    last = None
    for d in days_ms:
        q = QV.loc[:d - DAY]
        if len(q) >= 30:
            qv = q.iloc[-30:].mean().dropna().drop(labels=drop, errors="ignore")
            last = list(qv.sort_values(ascending=False).index[:50])
        U[d] = last
    return U


def own_rolling(days_ms):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from study_yr6wl_momentum_common import load_replay
    R = load_replay(["MOM-long"])
    R["cl"] = pd.to_datetime(R.closed_at, format="mixed").values.astype("datetime64[ms]").astype(np.int64)
    R = R.sort_values("cl")
    cl, y = R.cl.values, R.y.values
    out = []
    for d in days_ms:
        j = np.searchsorted(cl, d, side="left")
        o = {"day_ms": d}
        for k in (3, 7):
            i0 = np.searchsorted(cl, d - k * DAY, side="left")
            o[f"ml_d{k}"] = y[i0:j].mean() if j > i0 else np.nan
        o["ml_n30"] = y[max(0, j - 30):j].mean() if j >= 30 else np.nan
        out.append(o)
    return pd.DataFrame(out).set_index("day_ms")


def roll_prior(S, days_ms, k):
    """mean of daily values on days d-k .. d-1 (needs >= ceil(k/2) present)."""
    out = []
    idx = S.index.values
    for d in days_ms:
        w = S[(idx >= d - k * DAY) & (idx < d)]
        out.append(w.mean() if w.notna().sum() >= max(1, (k + 1) // 2) else np.nan)
    return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    days = pd.date_range("2025-01-01", D1, freq="D")
    dms = days.values.astype("datetime64[ms]").astype(np.int64)
    U = universe_by_day(dms)
    cache = os.path.join(OUT, "pair_day_stats.pkl")
    if os.path.exists(cache) and "--rebuild" not in sys.argv:
        ST = pd.read_pickle(cache)
    else:
        pairs = sorted({p for u in U.values() if u for p in u})
        print("alt pairs", len(pairs), flush=True)
        ST = {}
        with Pool(8) as pool:
            for p, s in pool.imap_unordered(_alt, pairs, chunksize=2):
                if s is not None:
                    ST[p] = s
        ST["__BTC__"] = day_stats("BTCUSDT", btc=True)
        pd.to_pickle(ST, cache)
    B = ST["__BTC__"]
    # alt cross-sectional medians per calendar day over U(d) membership of the day being described: use U(d) of the TARGET day
    # (membership known at d) applied to stats of days d-k..d-1.
    F = pd.DataFrame({"day_ms": dms}).set_index("day_ms")
    for col in ("ac5", "vr3", "rv5", "er", "vr12", "ac15", "ft15"):
        S = B[col]
        for k in ((1, 3, 7) if col in ("ac5", "vr3", "rv5", "er") else (3, 7)):
            if col in ("rv5",) and k == 3:
                continue
            if col == "er" and k == 3:
                continue
            if col in ("ac15", "ft15") and k == 3:
                continue
            F[f"btc_{col}_{k}"] = roll_prior(S, dms, k)
    F = F.rename(columns={"btc_ft15_7": "btc_ft_7"})
    F["btc_rvchg"] = F.btc_rv5_1 / F.btc_rv5_7
    alt = {}
    for col in ("vr3", "ac5", "ft", "rv5"):
        M = pd.DataFrame({p: s[col] for p, s in ST.items() if p != "__BTC__"})
        vals = {1: [], 3: [], 7: []}
        for d in dms:
            u = [p for p in (U[d] or []) if p in M.columns]
            for k in (1, 3, 7):
                w = M.loc[(M.index >= d - k * DAY) & (M.index < d), u]
                med = w.median(axis=1) if len(w) else pd.Series(dtype=float)
                vals[k].append(med.mean() if med.notna().sum() >= max(1, (k + 1) // 2) else np.nan)
        for k in (1, 3, 7):
            alt[f"alt_{col}_{k}"] = vals[k]
    for nm, ks in (("vr3", (1, 3, 7)), ("ac5", (3, 7)), ("ft", (3, 7)), ("rv5", (7,))):
        for k in ks:
            F[f"alt_{nm}_{k}"] = alt[f"alt_{nm}_{k}"]
    F = F.join(own_rolling(dms))
    L = pd.read_csv(os.path.join(ROOT, "reports/lite_regime/day_features.csv")).set_index("day_ms")
    keep = ["r1", "r3", "r7", "r30", "dma20", "dma50", "dma200", "rv7", "rvtr", "off30", "br_up3", "br_up7", "br_ma7", "br_ma20",
            "alt_rs7", "voltr", "disp7", "fund3", "spk24", "spk72", "flags96", "ft2", "ft12", "ftr12"]
    F = F.join(L[keep])
    F["day"] = pd.to_datetime(F.index, unit="ms").strftime("%Y-%m-%d")
    F.reset_index().to_csv(os.path.join(OUT, "day_features.csv"), index=False)
    print(F.describe().T[["count", "mean", "50%", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
