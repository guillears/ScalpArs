#!/usr/bin/env python3
"""NOSPOT_BTCBEAR (PREREG reports/NOSPOT_BTCBEAR_2026-11-18_PREREG.md) — shared: BTC regime series (a–g, lag-safe, valid-from steps),
row / trade loading from the NOSPOT_EXIT 1m sim, nets, one position, window stats (day-clustered bootstrap, LOWO, concentration).
Research only."""
import glob, json, os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_sleeve_lib as L     # noqa: E402
import nospot_exit_sim as S       # noqa: E402
import nospot_exit_eval as EV     # noqa: E402
DD = os.path.join(R, "reports", "NOSPOT_BTCBEAR_data"); os.makedirs(DD, exist_ok=True)
M5, H1, DAY = 300_000, 3_600_000, 86_400_000
DEFS = ["a", "b", "c", "d", "e", "f", "g"]
DEFNAME = dict(a="5m regime label (services/regime.py)", b="1h EMA20 slope < 0", c="4h return < 0", d="daily EMA50 < EMA200",
               e="1h close < EMA20 & EMA50", f="24h return < 0", g="breadth < 50 % above EMA50")
COHORTS = ["NS0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
RULEKEY = dict(R1="R1_OI_UNWIND", R2="R2_OI_TRAP", R3="R3_FUND_TURN", R4="R4_TOP_DISTRIB", R5="R5_TAKER_FADE", R6="R6_PREM_COLLAPSE",
               R7="R7_LH_VWAP", R8="R8_CASCADE", R9="R9_CROWD_LONG", R10="R10_OI_FROM_MAX")
ema = lambda x, n: pd.Series(x).ewm(span=n, adjust=False).mean().values
TS = L.TS
HOLD0 = TS("2025-07-01"); SPAN0 = TS("2023-07-01"); SPAN1 = TS("2026-11-18")


def btc5m_full():
    f = os.path.join(DD, "btc5m.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    import leads_deep_regimes as LR
    B = LR.btc5m()[["open_time", "o", "h", "l", "c"]]
    tail = json.load(open(os.path.join(DD, "btc_tail.json")))
    T = pd.DataFrame(dict(open_time=[int(r[0]) for r in tail], o=[float(r[1]) for r in tail], h=[float(r[2]) for r in tail],
                          l=[float(r[3]) for r in tail], c=[float(r[4]) for r in tail]))
    now_closed = T.open_time + M5 <= int(pd.Timestamp("2026-11-18 23:00").value // 10**6)
    T = T[now_closed & (T.open_time > B.open_time.max())]
    d = pd.concat([B, T]).drop_duplicates("open_time", keep="first").sort_values("open_time")
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, M5)
    d = d.set_index("open_time").reindex(grid).ffill().reset_index().rename(columns={"index": "open_time"})
    d.to_pickle(f)
    return d


def regime_series():
    """{def: (valid_from_ms, value ∈ {-1 bear(ON), +1 bull, 0 neither})}"""
    f = os.path.join(DD, "regimes.pkl")
    if os.path.exists(f):
        return pd.read_pickle(f)
    import frenzy_ta_build as TB
    from services.regime import classify_btc_regime
    B = btc5m_full(); t = B.open_time.values.astype("int64"); h, l, c = B.h.values, B.l.values, B.c.values
    out = {}
    adx, _, _ = TB.adx_di(h, l, c); r14 = TB.rsi(c, 14); e20 = TB.ema(c, 20)
    slope = (e20 - np.r_[np.full(3, np.nan), e20[:-3]]) / np.r_[np.full(3, np.nan), e20[:-3]] * 100
    lab = np.array([classify_btc_regime(a if np.isfinite(a) else None, r if np.isfinite(r) else None, s if np.isfinite(s) else None)
                    for a, r, s in zip(adx, r14, slope)])
    m = {"HEALTHY_BULL": 1, "STRONG_BULL": 1, "BULL_EXHAUSTED": 1, "HEALTHY_BEAR": -1, "STRONG_BEAR": -1, "BEAR_EXHAUSTED": -1}
    out["a"] = (t + M5, np.array([m.get(x, 0) for x in lab], float))
    out["a_label"] = (t + M5, lab)
    # 1 h bars (full hours only)
    g = t // H1; st = np.r_[0, np.nonzero(np.diff(g))[0] + 1]; en = np.r_[st[1:], len(t)] - 1
    full = (en - st) == 11; hc = c[en][full]; hv = (g[st] * H1 + H1)[full]
    e20h, e50h = ema(hc, 20), ema(hc, 50)
    sl = e20h - np.r_[np.full(3, np.nan), e20h[:-3]]
    out["b"] = (hv, np.where(np.isfinite(sl), np.where(sl < 0, -1.0, 1.0), np.nan))
    out["e"] = (hv, np.where((hc < e20h) & (hc < e50h), -1.0, 1.0))
    for nm, k in (("c", 48), ("f", 288)):
        rr = np.r_[np.full(k, np.nan), c[k:] / c[:-k] - 1]
        out[nm] = (t + M5, np.where(np.isfinite(rr), np.where(rr < 0, -1.0, 1.0), np.nan))
    dd = t // DAY; st = np.r_[0, np.nonzero(np.diff(dd))[0] + 1]; en = np.r_[st[1:], len(t)] - 1
    fulld = (en - st) == 287; dc = c[en][fulld]; dv = (dd[st] * DAY + DAY)[fulld]
    out["d"] = (dv, np.sign(ema(dc, 50) - ema(dc, 200)))
    reg = pd.read_pickle(os.path.join(R, "reports", "LEADS_DEEP_2026-11-18_data", "regimes.pkl"))
    out["g"] = reg["R5"]
    out["R1_ref"] = reg["R1"]; out["R4_ref"] = reg["R4"]
    pd.to_pickle(out, f)
    return out


def at(RS, d, T, shift=0, lo=None, hi=None):
    """regime value valid at entry T (optionally rotated by `shift` ms inside [lo, hi))."""
    T = np.asarray(T, "int64")
    if shift:
        T = lo + ((T - lo - shift) % (hi - lo))
    tv, vv = RS[d]; k = np.searchsorted(tv, T, side="right") - 1
    return np.where(k >= 0, vv[np.maximum(k, 0)], np.nan)


def duty(RS, d, d0, d1):
    """per UTC day: share of 5m bars ON (−1)."""
    grid = np.arange(d0 * DAY, d1 * DAY, M5)
    v = at(RS, d, grid) == -1
    return pd.Series(v).groupby(grid // DAY).mean()


def load():
    f = os.path.join(DD, "Y.pkl")
    if os.path.exists(f):
        Y, Z = pd.read_pickle(f)
        return Y, Z
    X = pd.read_pickle(os.path.join(L.ND, "X.pkl"))
    parts = [dict(np.load(p)) for p in sorted(glob.glob(os.path.join(S.D, "sim", "*.npz")))]
    Z = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    ok = Z["ok"]; Z = {k: v[ok] for k, v in Z.items()}
    xi = Z["xi"]
    M = L.rule_masks(X)
    num = [c for c in X.columns if not any(c.startswith(e + "_") for e in L.EXITS) and X[c].dtype.kind in "fiub"
           and c not in ("entry_ms", "on_ms", "spike_ms", "day", "epi", "R1", "R4")]
    Y = X.iloc[xi][["pair", "entry_ms", "epi", "half", "blk", "slip", "q5"] + [c for c in num if c not in ("slip", "q5")]].reset_index(drop=True)
    Y["NS0"] = L.ns0_mask(X)[xi]
    for k, v in RULEKEY.items():
        Y[k] = M[v][0][xi]
    Y["day"] = Y.entry_ms.values // DAY
    pd.to_pickle((Y, Z), f)
    return Y, Z


def nets(Y, Z):
    return EV.nets(Y, Z)


def one_pos(Y, Z, idx, j):
    idx = np.asarray(idx)
    if not len(idx):
        return idx
    return idx[EV.one_pos(Y.pair.values[idx], Y.entry_ms.values[idx].astype(np.int64), Z["xm"][idx, j])]


RNG = np.random.default_rng(20261118)


def clus_ci(v, days, nb=2000):
    s = pd.DataFrame(dict(v=v, d=days)).groupby("d").v.agg(["sum", "count"])
    n = len(s)
    if n < 2:
        return np.nan, np.nan
    ix = RNG.integers(0, n, (nb, n))
    a = s["sum"].values[ix].sum(1) / s["count"].values[ix].sum(1)
    return np.percentile(a, 2.5), np.percentile(a, 97.5)


def win_t(v, days):
    m = pd.Series(v).groupby(days).mean()
    if len(m) < 3 or m.std() == 0:
        return np.nan
    return m.mean() / (m.std(ddof=1) / np.sqrt(len(m)))


def cell_stats(v, v20, days, pairs, halves, kinds=None):
    rec = dict(N=len(v))
    if len(v) < 2:
        return rec
    tot = v.sum(); dsum = pd.Series(v).groupby(days).sum(); psum = pd.Series(v).groupby(pairs).sum()
    ci = clus_ci(v, days)
    kt, pf, be = EV.kel(v)
    top = dsum.sort_values(ascending=False).index[:10]
    lowo = min(v[days != d].mean() for d in top) if len(dsum) > 1 else np.nan
    tp_pair = psum.idxmax()
    rec.update(windows=len(dsum), WR=(v > 0).mean() * 100, avg=v.mean(), med_win=pd.Series(v).groupby(days).mean().median(),
               ci_lo=ci[0], ci_hi=ci[1], t_win=win_t(v, days), f20=np.mean(v20),
               day_share=dsum.max() / tot * 100 if tot > 0 else np.nan, pair_share=psum.max() / tot * 100 if tot > 0 else np.nan,
               top_pair=tp_pair, lowo_min=lowo, drop_top_pair=v[pairs != tp_pair].mean() if (pairs != tp_pair).any() else np.nan,
               kelly=kt, WRfloor=pf, BE=be, pos_days=(pd.Series(v).groupby(days).sum() > 0).mean() * 100)
    hv = []
    for h, _, _ in L.HALVES:
        m = halves == h
        rec[f"h_{h}"] = v[m].mean() if m.sum() >= 5 else np.nan; rec[f"h_{h}_N"] = int(m.sum())
        if m.sum() >= 5:
            hv.append(v[m].mean() > 0)
    rec["halves_pos"] = np.mean(hv) if hv else np.nan; rec["halves_n"] = len(hv)
    return rec
