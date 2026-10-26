#!/usr/bin/env python3
"""ML impulse study — feature + target grid (definitions: reports/ML_IMPULSE_PREREG.txt). No network.

    venv/bin/python -W ignore scripts/study_ml_impulse_features.py            -> reports/ML_IMPULSE_grid1m.csv.gz (1m features at
                                                                               every minute boundary) + reports/ML_IMPULSE_grid5m.csv.gz
                                                                               (all features + Y_btc / Y_alt at every 5m boundary)
    venv/bin/python -W ignore scripts/study_ml_impulse_features.py --assert   no-lookahead check (300 random t, bars after t deleted)

Time convention: a feature "at t" uses bars whose CLOSE time <= t (1m bar open_time <= t - 1 min; 5m bar open_time <= t - 5 min).
fill_features(o_ms) (Step 2) = 1m features at floor(o_ms, 1 min) + 5m features at floor(o_ms, 5 min).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REP = os.path.join(ROOT, "reports")
F1A = os.path.join(REP, "ml_btcfwd_fetch/k1m_taker")
F1B = os.path.join(REP, "ml_impulse_fetch/k1m_taker")
F5A = os.path.join(REP, "ml_btcfwd_fetch/k5m_taker")
F5B = os.path.join(REP, "ml_impulse_fetch/k5m_taker")
K1C = [os.path.join(REP, "backtest_cache_2025/k1m"), os.path.join(REP, "backtest_cache/k1m"),
       os.path.join(REP, "ml_impulse_fetch/k1m_live")]
OUT1 = os.path.join(REP, "ML_IMPULSE_grid1m.csv.gz")
OUT5 = os.path.join(REP, "ML_IMPULSE_grid5m.csv.gz")
FROZEN = os.path.join(REP, "ML_IMPULSE_frozen_params.json")
START = int(pd.Timestamp("2025-01-01").value // 10**6)
FRZ_END = int(pd.Timestamp("2025-04-01").value // 10**6)
MIN1, MIN5, DAY = 60_000, 300_000, 86_400_000
EXCL = {"BTCUSDT", "USDCUSDT", "PAXGUSDT", "XAUTUSDT", "FDUSDUSDT"}
ALTS = ("ETHUSDT", "SOLUSDT", "XRPUSDT", "ZECUSDT")

I1 = [f"{k}_{w}" for w in (15, 30) for k in ("er", "upsh", "spike", "volconc", "volslope")] + ["vwap1h"]
I3 = ["corr15"]
CONT = I1 + I3                                                    # 12 continuous
LEGS = ["L1_btc", "L2_eth", "L3_sol", "L4_xrp", "L5_zec", "L6_top10", "L7_brrise", "L8_taker", "L9_vsurge"]
EXTRA5 = ["top10_up", "br_trend", "br_trend_d15", "top10_tbr", "vsurge"]


# ───────────────────────── loaders ─────────────────────────
def minute_frame(sym):
    """continuous 1m grid (index = open_time) of the fetched taker klines; missing minutes NaN."""
    fp = os.path.join(F1A if os.path.exists(os.path.join(F1A, sym + ".csv")) else F1B, sym + ".csv")
    x = pd.read_csv(fp).drop_duplicates("open_time").set_index("open_time").sort_index()
    grid = np.arange(x.index.min(), x.index.max() + MIN1, MIN1, dtype=np.int64)
    return x.reindex(grid)


def load_universe():
    u = pd.concat([pd.read_csv(os.path.join(REP, b, "universe_daily_top50.csv")) for b in
                   ("backtest_cache_2025", "backtest_cache")]).drop_duplicates(["date_ms", "pair"])
    u = u[~u.pair.isin(EXCL)].sort_values(["date_ms", "rank"])
    days = {int(d): list(g.pair) for d, g in u.groupby("date_ms")}
    return days, np.array(sorted(days), dtype=np.int64)


def members_for_day(d0, days, keys):
    """previous UTC day's list (strictly before d0); after the last list, the last list (stale)."""
    i = np.searchsorted(keys, d0 - 1, side="right") - 1
    return days[keys[i]] if i >= 0 else []


_P5 = {}


def pair5(p):
    """regular 5m grid for a member: index open_time -> c, qvol, tbq, and derived lagged / rolling columns."""
    if p in _P5:
        return _P5[p]
    xs = [pd.read_csv(os.path.join(d, p + ".csv"), usecols=["open_time", "c", "qvol", "tbq"])
          for d in (F5A, F5B) if os.path.exists(os.path.join(d, p + ".csv"))]
    if not xs:
        _P5[p] = None
        return None
    x = pd.concat(xs).drop_duplicates("open_time").set_index("open_time").sort_index().astype(float)
    grid = np.arange(x.index.min(), x.index.max() + MIN5, MIN5, dtype=np.int64)
    x = x.reindex(grid)
    x["r3"] = x.qvol.rolling(3, min_periods=3).sum(); x["r288"] = x.qvol.rolling(288, min_periods=288).sum()
    _P5[p] = x
    return x


_P1 = {}
NEED1 = {}          # pair -> set of UTC day starts on which it is a TOP-10 member (build() fills it to bound memory)


def pair1_close(p):
    if p in _P1:
        return _P1[p]
    xs = [pd.read_csv(os.path.join(d, p + ".csv"), usecols=["open_time", "c"]) for d in K1C if os.path.exists(os.path.join(d, p + ".csv"))]
    if not xs:
        _P1[p] = None
        return None
    x = pd.concat(xs).drop_duplicates("open_time").sort_values("open_time")
    if NEED1.get(p):
        nd = np.array(sorted(NEED1[p]), dtype=np.int64)
        ot = x.open_time.values
        keep = np.isin((ot // DAY) * DAY, nd) | np.isin(((ot + 20 * MIN1) // DAY) * DAY, nd)
        x = x[keep]
    x = x.set_index("open_time").c.astype(float)
    _P1[p] = x
    return x


# ───────────────────────── 1m block (BTC I1 + alt legs) ─────────────────────────
def one_min_block(B, alts):
    """features at t = open_time + 1 min of bar i (bars closed <= t)."""
    c = B.c.astype(float); v = B.vol.astype(float); h = B.h.astype(float); l = B.l.astype(float)
    d = c.diff()
    up = d.clip(lower=0)
    f = pd.DataFrame(index=B.index + MIN1)
    for W in (15, 30):
        path = d.abs().rolling(W, min_periods=W).sum()
        f[f"er_{W}"] = ((c - c.shift(W)) / path).values
        f[f"upsh_{W}"] = (d > 0).astype(float).where(d.notna()).rolling(W, min_periods=W).mean().values
        su = up.rolling(W, min_periods=W).sum()
        f[f"spike_{W}"] = (up.rolling(W, min_periods=W).max() / su.where(su > 0)).values
        f[f"volconc_{W}"] = (v.rolling(W, min_periods=W).max() / v.rolling(W, min_periods=W).sum()).values
        k = W // 3
        s = v.rolling(k, min_periods=k).sum()
        f[f"volslope_{W}"] = np.log((s / s.shift(W - k)).where(lambda z: z > 0)).values
    typ = (h + l + c) / 3
    vw = (typ * v).rolling(60, min_periods=60).sum() / v.rolling(60, min_periods=60).sum()
    f["vwap1h"] = ((c / vw - 1) * 100).values
    f["btc_r15"] = (c.pct_change(15, fill_method=None) * 100).values
    for sym, X in alts.items():
        xc = X.c.astype(float).reindex(B.index)
        f[sym[:3].lower() + "_r15"] = (xc.pct_change(15, fill_method=None) * 100).values
    cv = c.values
    f["Y_btc"] = (np.r_[cv[15:], [np.nan] * 15] / cv - 1) * 100
    f.index.name = "t"
    return f


# ───────────────────────── 5m block (top-10 / top-50 legs, corr15, Y_alt) ─────────────────────────
def _at(x, col, T):
    """value of member grid x[col] at bar open_time T (NaN when absent)."""
    idx = x.index.values
    k = np.searchsorted(idx, T)
    kk = np.minimum(k, len(idx) - 1)
    ok = (k < len(idx)) & (idx[kk] == T)
    out = np.full(len(T), np.nan)
    out[ok] = x[col].values[kk[ok]]
    return out


def corr15_at(T, top10, cutoff=None):
    """mean pairwise corr of the 15 1m log-returns ending at each t in T (bars open t-15m..t-1m; returns need c at t-16m)."""
    R = []
    for p in top10:
        s = pair1_close(p)
        if s is None:
            R.append(np.full((len(T), 15), np.nan)); continue
        if cutoff is not None:
            s = s[s.index < cutoff]
        offs = np.arange(-16, 0) * MIN1                     # opens t-16m .. t-1m -> 16 closes -> 15 returns
        q = (T[:, None] + offs[None, :]).ravel()
        idx = s.index.values
        k = np.searchsorted(idx, q); kk = np.minimum(k, len(idx) - 1)
        ok = (k < len(idx)) & (idx[kk] == q)
        cc = np.where(ok, s.values[kk], np.nan).reshape(len(T), 16)
        R.append(np.diff(np.log(cc), axis=1))
    R = np.stack(R, axis=2)                                  # (T, 15, m)
    valid = np.isfinite(R).sum(1) >= 14                      # member has >= 14 returns
    Rm = np.where(np.isfinite(R), R, np.nan)
    mu = np.nanmean(Rm, axis=1, keepdims=True)
    Z = np.nan_to_num(Rm - mu)
    sd = np.sqrt((Z * Z).sum(1))                             # (T, m)
    good = valid & (sd > 0)
    Zn = Z / np.where(sd > 0, sd, 1)[:, None, :]
    C = np.einsum("tim,tin->tmn", Zn, Zn)
    out = np.full(len(T), np.nan)
    for i in range(len(T)):
        g = np.flatnonzero(good[i])
        if len(g) >= 6:
            sub = C[i][np.ix_(g, g)]
            out[i] = (sub.sum() - np.trace(sub)) / (len(g) * (len(g) - 1))
    return out


def five_min_feats(T, mem, cutoff=None):
    """5m features for boundaries T (int64 ms) given the member list (top-50 order by volume). cutoff -> drop bars closing
    after it (no-lookahead assert); targets are computed only when cutoff is None."""
    top10 = mem[:10]
    n = len(T)
    up10 = np.zeros(n); n10 = np.zeros(n); tb10 = np.zeros(n); q10 = np.zeros(n); n10q = np.zeros(n)
    brn = np.zeros(n); brc = np.zeros(n); brn_l = np.zeros(n); brc_l = np.zeros(n)
    q3 = np.zeros(n); q288 = np.zeros(n); nsurge = np.zeros(n)
    fw = np.zeros(n); nfw = np.zeros(n)
    for j, p in enumerate(mem):
        x = pair5(p)
        if x is None:
            continue
        if cutoff is not None:
            x = x[x.index + MIN5 <= cutoff].copy()
            if not len(x):
                continue
            x["r3"] = x.qvol.rolling(3, min_periods=3).sum(); x["r288"] = x.qvol.rolling(288, min_periods=288).sum()
        c0 = _at(x, "c", T - MIN5)                       # close of bar ending t
        c15 = _at(x, "c", T - 4 * MIN5)                  # close of bar ending t-15
        c60 = _at(x, "c", T - 13 * MIN5)                 # ending t-60
        c15b = c15
        c75 = _at(x, "c", T - 16 * MIN5)                 # ending t-75 (breadth lag: t-15 vs t-75)
        if j < 10:
            ok = np.isfinite(c0) & np.isfinite(c15)
            up10 += np.where(ok, c0 > c15, 0); n10 += ok
            qq = np.zeros(n); tt = np.zeros(n); okq = np.ones(n, bool)
            for lag in (1, 2, 3):
                a = _at(x, "qvol", T - lag * MIN5); b = _at(x, "tbq", T - lag * MIN5)
                okq &= np.isfinite(a) & np.isfinite(b)
                qq += np.nan_to_num(a); tt += np.nan_to_num(b)
            q10 += np.where(okq, qq, 0); tb10 += np.where(okq, tt, 0); n10q += okq
        ok = np.isfinite(c0) & np.isfinite(c60)
        brc += np.where(ok, c0 > c60, 0); brn += ok
        ok = np.isfinite(c15b) & np.isfinite(c75)
        brc_l += np.where(ok, c15b > c75, 0); brn_l += ok
        # volume surge: last 3 bars vs the last 288 bars / 96 (member needs all 288 bars)
        a3 = _at(x, "r3", T - MIN5); a288 = _at(x, "r288", T - MIN5)
        ok = np.isfinite(a3) & np.isfinite(a288)
        q3 += np.where(ok, a3, 0); q288 += np.where(ok, a288, 0); nsurge += ok
        if cutoff is None:
            cf = _at(x, "c", T + 2 * MIN5)               # close of bar ending t+15
            ok = np.isfinite(c0) & np.isfinite(cf)
            fw += np.where(ok, (cf / np.where(ok, c0, 1) - 1) * 100, 0); nfw += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        o = pd.DataFrame(index=T)
        o["top10_up"] = np.where(n10 >= 7, up10 / n10, np.nan)
        bt = np.where(brn >= 25, brc / brn, np.nan); bl = np.where(brn_l >= 25, brc_l / brn_l, np.nan)
        o["br_trend"] = bt
        o["br_trend_d15"] = bt - bl
        o["top10_tbr"] = np.where(n10q >= 7, tb10 / q10, np.nan)
        o["vsurge"] = np.where(nsurge >= 25, q3 / (q288 / 96), np.nan)
        o["n_top50"] = brn
        if cutoff is None:
            o["Y_alt"] = np.where(nfw >= 25, fw / nfw, np.nan)
    o["corr15"] = corr15_at(T, top10, cutoff)
    return o


# ───────────────────────── score ─────────────────────────
def add_legs(g, frz):
    def leg(cond, *cols):
        ok = g[list(cols)].notna().all(axis=1)
        return np.where(ok, cond.astype(float), np.nan)
    g["L1_btc"] = leg((g.btc_r15 > 0) & (g.er_15 >= frz["q_er"]), "btc_r15", "er_15")
    for a, L in (("eth", "L2_eth"), ("sol", "L3_sol"), ("xrp", "L4_xrp"), ("zec", "L5_zec")):
        g[L] = leg(g[a + "_r15"] > 0, a + "_r15")
    g["L6_top10"] = leg(g.top10_up >= 0.70, "top10_up")
    g["L7_brrise"] = leg(g.br_trend_d15 > 0, "br_trend_d15")
    g["L8_taker"] = leg(g.top10_tbr > 0.5, "top10_tbr")
    g["L9_vsurge"] = leg(g.vsurge >= 1.5, "vsurge")
    g["SCORE"] = g[LEGS].sum(axis=1, min_count=9)
    ok = g[["btc_r15", "L6_top10", "L7_brrise"]].notna().all(axis=1)
    g["CONJ"] = np.where(ok, ((g.btc_r15 > 0) & (g.L6_top10 == 1) & (g.L7_brrise == 1)).astype(float), np.nan)
    ok = g[["btc_r15", "spike_15", "er_15"]].notna().all(axis=1)
    g["SPIKE"] = np.where(ok, ((g.btc_r15 > 0) & (g.spike_15 >= frz["q_spike80"]) & (g.er_15 < frz["q_er"])).astype(float), np.nan)
    return g


def load_frozen():
    return json.load(open(FROZEN))


def build():
    B = minute_frame("BTCUSDT")
    alts = {s: minute_frame(s) for s in ALTS}
    f1 = one_min_block(B, alts)
    f1 = f1[f1.index >= START - DAY]
    # frozen params (features only, strictly before the first test month)
    fr = f1[(f1.index >= START) & (f1.index < FRZ_END) & (f1.index % MIN5 == 0)]
    frz = dict(q_er=float(fr.er_15[fr.btc_r15 > 0].median()), q_spike80=float(fr.spike_15.quantile(0.8)),
               window="2025-01-01..2025-03-31 5m boundaries", written=str(pd.Timestamp.utcnow()))
    if os.path.exists(FROZEN):
        old = load_frozen()
        assert abs(old["q_er"] - frz["q_er"]) < 1e-9 and abs(old["q_spike80"] - frz["q_spike80"]) < 1e-9, "frozen params changed"
        frz = old
    else:
        json.dump(frz, open(FROZEN, "w"), indent=1)
    print("frozen", frz)
    f1 = f1[f1.index >= START]
    f1.to_csv(OUT1, compression="gzip")
    # 5m boundaries
    g = f1[f1.index % MIN5 == 0].copy()
    T = g.index.values.astype(np.int64)
    days, keys = load_universe()
    for d0 in np.unique((T // DAY) * DAY):
        for p in members_for_day(d0, days, keys)[:10]:
            NEED1.setdefault(p, set()).add(int(d0))
    parts = []
    for d0 in np.unique((T // DAY) * DAY):
        sel = T[(T >= d0) & (T < d0 + DAY)]
        mem = members_for_day(d0, days, keys)
        o = five_min_feats(sel, mem)
        o["stale_list"] = float(d0 > keys[-1] + DAY)
        parts.append(o)
        if d0 % (30 * DAY) == 0:
            print("  day", pd.Timestamp(d0, unit="ms").date(), flush=True)
    o = pd.concat(parts)
    g = g.join(o)
    g = add_legs(g, frz)
    g.to_csv(OUT5, compression="gzip")
    print("wrote", OUT5, g.shape, "and", OUT1, f1.shape)
    print(g[CONT + LEGS + ["SCORE", "CONJ", "SPIKE", "Y_btc", "Y_alt"]].notna().mean().round(3).to_string())
    return g


# ───────────────────────── fill-time features ─────────────────────────
_C = {}


def fill_features(o_ms, lag5=0):
    """o_ms: array of entry times -> DataFrame of 1m features at floor(t,1m) + 5m features at floor(t,5m) - lag5 bars + legs/score.
    lag5 = 1 is the scan-lag sensitivity (the engine's per-scan stamps see the previous 5m bar for ~59 % of master fills)."""
    if "g1" not in _C:
        _C["g1"] = pd.read_csv(OUT1, index_col=0)
        _C["g5"] = pd.read_csv(OUT5, index_col=0)
    o_ms = np.asarray(o_ms, dtype=np.int64)
    t1 = (o_ms // MIN1) * MIN1; t5 = (o_ms // MIN5) * MIN5 - lag5 * MIN5
    A = _C["g1"].reindex(t1); A.index = range(len(o_ms))
    five = EXTRA5 + ["corr15", "Y_alt", "stale_list", "n_top50"]
    Bf = _C["g5"][five].reindex(t5); Bf.index = range(len(o_ms))
    F = pd.concat([A, Bf], axis=1)
    F = add_legs(F, load_frozen())
    F["t1"] = t1; F["t5"] = t5
    return F


# ───────────────────────── no-lookahead assert ─────────────────────────
def assert_no_lookahead(n=300):
    B = minute_frame("BTCUSDT"); alts = {s: minute_frame(s) for s in ALTS}
    G1 = pd.read_csv(OUT1, index_col=0); G5 = pd.read_csv(OUT5, index_col=0)
    days, keys = load_universe()
    rng = np.random.default_rng(11)
    ts = rng.choice(G5.index[(G5.index < G5.index.max() - DAY)].values, n, replace=False)
    for t in ts:
        d0 = int((t // DAY) * DAY)
        for p in members_for_day(d0, days, keys)[:10]:
            NEED1.setdefault(p, set()).add(d0)
    c1 = [c for c in I1 + ["btc_r15", "eth_r15", "sol_r15", "xrp_r15", "zec_r15"]]
    c5 = EXTRA5 + ["corr15"]
    bad1 = bad5 = 0
    for t in ts:
        cut = lambda X: X[(X.index < t) & (X.index >= t - DAY)]          # 1m bars with open < t  (close <= t)
        f = one_min_block(cut(B), {s: cut(X) for s, X in alts.items()})
        a = G1.loc[t, c1].astype(float).values; b = f.loc[t, c1].astype(float).values
        bad1 += not np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9)
        mem = members_for_day((t // DAY) * DAY, days, keys)
        o = five_min_feats(np.array([t], dtype=np.int64), mem, cutoff=t)
        a = G5.loc[t, c5].astype(float).values; b = o.iloc[0][c5].astype(float).values
        bad5 += not np.allclose(a, b, equal_nan=True, rtol=1e-6, atol=1e-9)
    print(f"no-lookahead assert: 1m block {n - bad1}/{n} identical · 5m block {n - bad5}/{n} identical")
    assert bad1 == 0 and bad5 == 0


if __name__ == "__main__":
    if "--assert" in sys.argv:
        assert_no_lookahead()
    else:
        build()
