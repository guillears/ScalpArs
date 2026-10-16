#!/usr/bin/env python3
"""LITE AT-+1 study — step 2: price C + every at-touch policy on each fill's hybrid path (FX.load_path, ticks → 1m tail; menu parity
asserted) and compute the at-touch features (path shape from the event path, taker flow / structure from the fetched 1m klines, market
from BTC/ETH/SOL 1m + the breadth grid). Pre-registration: reports/LITE_AT1_PREREG.txt (+ addendum A1). Research only, cache only.
Output: reports/lite_at1/rows.csv (one row per fill: C, policies, Δ, touch facts, features) + reports/lite_at1/price.log.
Usage: venv/bin/python -W ignore scripts/study_lite_at1_price.py [--jobs 9]
"""
import os, sys, time, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_lite_exitopt_common as O  # noqa: E402  (imports protect → late → tick-lock loader patches)
FX, WT, C = O.FX, O.WT, O.C
MIN = WT.MIN
TAKER, MAKER = O.TAKER, O.MAKER
OUT = os.path.join(REP, "lite_at1")
K1 = os.path.join(OUT, "k1m")
SLIP = -0.20
SL, CAP = 3.0, 720
TOUCH = 1.0
LIVE_CSV = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_21-18-32.csv")
POL = ["BANK", "L0", "L0.3", "L-1", "D0.5", "D0.8", "D1"]
POLDEF = {"L0": ("L", 0.0), "L0.3": ("L", 0.3), "L-1": ("L", -1.0), "D0.5": ("D", 0.5), "D0.8": ("D", 0.8), "D1": ("D", 1.0)}


# ───────────────────────── walker ─────────────────────────
def levels(E, tick, tp):
    _, th, bk = O.rm_levels(E, tick, np.array([tp]))
    return float(th[0]), float(bk[0])


def first_exit(P, j0, n, line, th):
    """first event j ≥ j0 (< n) with lo ≤ line[j] (stop, first) or hi ≥ th. → ('S'|'T'|None, j)."""
    if j0 >= n:
        return None, n
    lo, hi = P["lo"][j0:n], P["hi"][j0:n]
    ln = line if np.ndim(line) == 0 else line[j0:n]
    s = np.flatnonzero(lo <= ln); t = np.flatnonzero(hi >= th - 1e-12)
    a = s[0] + j0 if len(s) else n; b = t[0] + j0 if len(t) else n
    if a < n and a <= b:
        return "S", int(a)
    if b < n:
        return "T", int(b)
    return None, n


def book_stop(P, j, line):
    if P["bar"][j]:
        op = P["op"][j]
        return (min(line, op) if np.isfinite(op) else line) + SLIP
    return float(P["lo"][j]) + SLIP


def resolve(P, kind, j, n, line_j, bk):
    """→ (pct ruler incl. slip, why) for an exit found by first_exit; cap / UNRES when kind is None."""
    if kind == "S":
        return book_stop(P, j, line_j), ("STOP" if line_j <= -SL + 1e-12 else "PROT")
    if kind == "T":
        return bk, "TP"
    if n == 0 or P["end_ms"] < P["te"] + CAP * MIN:
        return np.nan, "UNRES"
    return float(P["cl"][n - 1]) + SLIP, "CAP"


def price_fill(P, E, tick, tpC):
    n = int(np.searchsorted(P["t"], P["te"] + CAP * MIN, side="right"))
    out = {}
    th, bk = levels(E, tick, tpC)
    kC, jC = first_exit(P, 0, n, -SL, th)
    out["C"], out["C_why"] = resolve(P, kC, jC, n, -SL, bk)
    out["C_xm"] = (P["t"][jC] - P["te"]) / MIN if jC < n else np.nan
    if jC < n:
        upto = jC if (kC == "S" and P["bar"][jC]) else jC + 1
    else:
        upto = n
    out["peakC"] = float(P["hi"][:upto].max()) if upto > 0 else -np.inf
    # tier levels: +1 = the touch (taker net, as the touch event); +2 / +3 = the resting-maker fill thresholds of a +2 / +3 limit
    # (a maker +2 fills while the taker-net high is ~1.98 — tiering on taker net mislabelled every C +2 fill as T1; fixed 2026-10-15
    #  before any screen output was read, declared in the prereg addendum C)
    th2, _ = levels(E, tick, 2.0); th3, _ = levels(E, tick, 3.0)
    pk = out["peakC"]
    out["tier"] = 0 if pk < TOUCH else (1 if pk < th2 - 1e-12 else (2 if pk < th3 - 1e-12 else 3))
    th1, bk1 = levels(E, tick, 1.0)
    k1_, j1_ = first_exit(P, 0, n, -SL, th1)
    out["P1"], out["P1_why"] = resolve(P, k1_, j1_, n, -SL, bk1)
    hi = P["hi"][:n]
    it = np.flatnonzero(hi >= TOUCH)
    out["touch"] = False; out["acted"] = False
    for p in POL:
        out[p] = out["C"]; out[p + "_why"] = out["C_why"]
    if not len(it) or (jC < n and jC < it[0]):
        return out, None
    k = int(it[0])
    out["touch"] = True
    T = -(-int(P["t"][k]) // MIN) * MIN
    jT = int(np.searchsorted(P["t"][:n], T, side="right")) - 1
    out["T"] = T; out["k"] = k; out["jT"] = jT
    if jC <= jT or jT >= n - 1:
        return out, (k, jT, T, n)     # C exits at / before the decision (or no event after T) → policies = C
    out["acted"] = True
    out["BANK"], out["BANK_why"] = float(P["cl"][jT]) + SLIP, "BANK"
    for p, (kind, v) in POLDEF.items():
        if kind == "L":
            line = np.full(n, -SL); line[jT + 1:] = max(-SL, v)
        else:
            rm = np.maximum.accumulate(P["hi"][:n])
            prev = np.concatenate([[-np.inf], rm[:-1]])
            line = np.full(n, -SL); line[jT + 1:] = np.maximum(-SL, prev[jT + 1:] - v)
        kk, jj = first_exit(P, jT + 1, n, line, th)
        out[p], out[p + "_why"] = resolve(P, kk, jj, n, line[jj] if jj < n else -SL, bk)
    return out, (k, jT, T, n)


def parity_loop(P, E, tick, tpC, k):
    """P3 / P4 with the decision forced at the touch EVENT (no wait to the bar close) must equal O.walk_loop (A = +1)."""
    n = int(np.searchsorted(P["t"], P["te"] + CAP * MIN, side="right"))
    th, bk = levels(E, tick, tpC)
    res = {}
    for p, (kind, v) in POLDEF.items():
        if kind == "L":
            line = np.full(n, -SL); line[k + 1:] = max(-SL, v)
            ref = O.walk_loop(P, E, tick, tpC, SL, 1.0, v, np.inf, CAP, "RM")
        else:
            rm = np.maximum.accumulate(P["hi"][:n]); prev = np.concatenate([[-np.inf], rm[:-1]])
            line = np.full(n, -SL); line[k + 1:] = np.maximum(-SL, prev[k + 1:] - v)
            ref = O.walk_loop(P, E, tick, tpC, SL, 1.0, -np.inf, v, CAP, "RM")
        # exits at or before k under C
        kC, jC = first_exit(P, 0, k + 1, -SL, th)
        if kC is not None:
            continue
        kk, jj = first_exit(P, k + 1, n, line, th)
        if kk == "S":
            mine = (book_stop(P, jj, line[jj]) - SLIP, 2 if line[jj] > -SL + 1e-12 else 1)
        elif kk == "T":
            mine = (bk, 0)
        else:
            mine = (np.nan, 9) if (n == 0 or P["end_ms"] < P["te"] + CAP * MIN) else (float(P["cl"][n - 1]), 3)
        ok = mine[1] == ref[1] and (mine[1] == 9 or abs(mine[0] - ref[0]) < 1e-9)
        res[p] = ok
    return res


# ───────────────────────── features ─────────────────────────
_K1C = {}


def k1(pair):
    if pair not in _K1C:
        f = os.path.join(K1, f"{pair}.csv")
        _K1C[pair] = pd.read_csv(f) if os.path.exists(f) else None
    return _K1C[pair]


def path_feats(P, k, jT, T, n):
    te = P["te"]; t = P["t"]; hi = P["hi"]; lo = P["lo"]
    d = {}
    d["min_to_touch"] = (T - te) / MIN
    d["mae_pre"] = float(lo[:jT + 1].min())
    d["cl_T"] = float(P["cl"][jT])
    d["path_tick_at_touch"] = int(not P["bar"][k])
    i5 = np.flatnonzero(hi[:k + 1] >= 0.5)
    if len(i5):
        a = i5[0]
        rm = np.maximum.accumulate(hi[a:k + 1])
        d["pb05"] = float(np.max(rm - lo[a:k + 1]))
    else:
        d["pb05"] = np.nan
    if P["bar"][k]:
        d["s08"] = float((hi[:k] >= 0.8).sum() * 60)
    else:
        tt = t[:k + 1]; above = hi[:k] >= 0.8
        dt = np.diff(tt) / 1000.0
        d["s08"] = float(dt[above].sum())
    return d


def _rsi(c, n=14):
    if len(c) < n + 2:
        return np.nan, np.nan
    dlt = np.diff(c); up = np.clip(dlt, 0, None); dn = np.clip(-dlt, 0, None)
    au, ad = up[:n].mean(), dn[:n].mean(); rs = []
    for u, d_ in zip(up[n:], dn[n:]):
        au = (au * (n - 1) + u) / n; ad = (ad * (n - 1) + d_) / n
        rs.append(100 - 100 / (1 + au / ad) if ad > 0 else 100.0)
    if len(rs) < 2:
        return np.nan, np.nan
    return rs[-1], rs[-1] - rs[-2]


def _adx(h, l, c, n=14):
    if len(c) < 2 * n + 2:
        return np.nan, np.nan
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    um = h[1:] - h[:-1]; dm = l[:-1] - l[1:]
    pdm = np.where((um > dm) & (um > 0), um, 0.0); ndm = np.where((dm > um) & (dm > 0), dm, 0.0)
    a, p_, m_ = tr[:n].sum(), pdm[:n].sum(), ndm[:n].sum(); dx = []
    for i in range(n, len(tr)):
        a = a - a / n + tr[i]; p_ = p_ - p_ / n + pdm[i]; m_ = m_ - m_ / n + ndm[i]
        pdi, ndi = 100 * p_ / a if a else 0, 100 * m_ / a if a else 0
        dx.append(100 * abs(pdi - ndi) / (pdi + ndi) if pdi + ndi else 0)
    if len(dx) < n + 1:
        return np.nan, np.nan
    adx = [np.mean(dx[:n])]
    for x in dx[n:]:
        adx.append((adx[-1] * (n - 1) + x) / n)
    return adx[-1], adx[-1] - adx[-2]


def _ema(c, n):
    return pd.Series(c).ewm(span=n, adjust=False).mean().values


def ind_feats(h, l, c, pre):
    """indicators on closed bars (arrays end at the decision bar). pre = prefix for the names."""
    d = {}
    d[pre + "rsi"], d[pre + "rsi_dir"] = _rsi(c)
    d[pre + "adx"], d[pre + "adx_dir"] = _adx(h, l, c)
    if len(c) >= 25:
        e5, e8, e13, e20 = _ema(c, 5), _ema(c, 8), _ema(c, 13), _ema(c, 20)
        g58 = (e5 - e8) / c * 100; g813 = (e8 - e13) / c * 100; g520 = (e5 - e20) / c * 100
        d[pre + "gap58"], d[pre + "gap813"], d[pre + "gap520"] = g58[-1], g813[-1], g520[-1]
        d[pre + "gap58_w"], d[pre + "gap813_w"], d[pre + "gap520_w"] = g58[-1] - g58[-4], g813[-1] - g813[-4], g520[-1] - g520[-4]
        d[pre + "dema13"] = (c[-1] / e13[-1] - 1) * 100
    return d


def kline_feats(pair, te, T, E, vwap, spike_at, k1df):
    d = {}
    if k1df is None or not len(k1df):
        return d
    o = k1df.open_time.values.astype("int64"); ce = o + MIN
    ent = (ce <= te)
    e5 = np.flatnonzero(ent)[-5:]
    first = -(-te // MIN) * MIN
    post = np.flatnonzero((o >= first) & (ce <= T))
    if not len(post) or ce[post[-1]] != T or len(e5) < 5:
        return d
    v = k1df.v.values.astype(float); tb = k1df.tbv.values.astype(float)
    op = k1df.o.values.astype(float); h = k1df.h.values.astype(float); c = k1df.c.values.astype(float)
    kb = post[-1]
    l5 = np.flatnonzero(ce <= T)[-5:]
    sh = lambda ix: tb[ix].sum() / v[ix].sum() if v[ix].sum() > 0 else np.nan
    d["tb_k"] = tb[kb] / v[kb] if v[kb] > 0 else np.nan
    d["tb5"] = sh(l5)
    d["d_tb5"] = d["tb5"] - sh(e5)
    # last closed 5m bucket
    b5 = (T // (5 * MIN)) * 5 * MIN
    ix5 = np.flatnonzero((o >= b5 - 5 * MIN) & (o < b5))
    d["tb5m"] = sh(ix5) if len(ix5) == 5 else np.nan
    ve = v[e5].mean()
    d["vr_k"] = v[kb] / ve if ve > 0 else np.nan
    d["vr5"] = v[l5].sum() / v[e5].sum() if v[e5].sum() > 0 else np.nan
    d["ngreen_frac"] = float((c[post] > op[post]).mean())
    cT = c[kb]
    d["dvwap"] = (cT / vwap - 1) * 100 if vwap and np.isfinite(vwap) else np.nan
    epi = np.flatnonzero((o >= spike_at) & (ce <= te))
    ehi = h[epi].max() if len(epi) else np.nan
    phi = h[post].max()
    d["newhigh"] = float(phi > ehi) if np.isfinite(ehi) else np.nan
    d["uwick_k"] = (h[kb] - c[kb]) / c[kb] * 100
    d["dhi_epi"] = (cT / np.nanmax([ehi, phi]) - 1) * 100
    i3 = np.flatnonzero(ce <= T)
    if len(i3) >= 6:
        d["r3"] = (c[i3[-1]] / c[i3[-4]] - 1) * 100
        d["r5"] = (c[i3[-1]] / c[i3[-6]] - 1) * 100
    d["nbars_1m"] = len(post)
    hist = np.flatnonzero(ce <= T)
    d.update(ind_feats(h[hist], k1df.l.values.astype(float)[hist], c[hist], "t1_"))
    # 5m aggregated (complete buckets closed by T)
    kk = k1df.iloc[hist]
    b = (kk.open_time.values // (5 * MIN))
    g = kk.assign(b=b).groupby("b").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), n=("c", "size"))
    g = g[g.n == 5]
    d.update(ind_feats(g.h.values.astype(float), g.l.values.astype(float), g.c.values.astype(float), "t5_"))
    p60 = np.flatnonzero(ent)[-60:]
    if len(p60) >= 30 and v[p60].mean() > 0:
        d["pvol1m_vs60"] = v[kb] / v[p60].mean()
        d["pvol5m_vs60"] = v[l5].sum() / (v[p60].mean() * 5)
    return d


class Market:
    def __init__(self):
        self.k = {}
        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            a = pd.read_csv(os.path.join(REP, "ml_btcfwd_fetch", "k1m_taker", f"{s}.csv"), usecols=["open_time", "c", "qvol"])
            fr = [a]
            f = os.path.join(K1, f"{s}.csv")
            if os.path.exists(f):
                fr.append(pd.read_csv(f, usecols=["open_time", "c", "qv"]).rename(columns={"qv": "qvol"}))
            if s == "BTCUSDT":
                fr.append(pd.read_csv(os.path.join(REP, "lite_fade_klines", "btc1m", "BTCUSDT.csv"), usecols=["open_time", "c", "qv"]
                                      ).rename(columns={"qv": "qvol"}))
            d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time")
            self.k[s] = (d.open_time.values.astype("int64") + MIN, d.c.values.astype(float), np.cumsum(d.qvol.values.astype(float)))
        ce, c, _ = self.k["BTCUSDT"]
        h = pd.Series(c, index=pd.to_datetime(ce, unit="ms"))
        # 1h closes: the 1m close at each hour boundary (close time = hour)
        hh = h[(ce % (60 * MIN)) == 0]
        ema = hh.ewm(span=20, adjust=False).mean()
        self.h_t = hh.index.values.astype("datetime64[ms]").astype("int64"); self.h_e = ema.values
        g = pd.read_csv(os.path.join(REP, "ML_BTCFWD_grid5m.csv.gz"), usecols=["t", "br_up5", "agg_tbr5"])
        self.g_t = g.t.values.astype("int64"); self.g = g[["br_up5", "agg_tbr5"]].values

    def qv(self, t0, t1):
        """BTC+ETH+SOL quote volume of the 1m bars closing in (t0, t1]."""
        tot = 0.0
        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            ce, _, cq = self.k[s]
            i0 = np.searchsorted(ce, t0, side="right") - 1; i1 = np.searchsorted(ce, t1, side="right") - 1
            if i0 < 0 or i1 < 0 or t1 - ce[i1] > 3 * MIN:
                return np.nan
            tot += cq[i1] - cq[i0]
        return tot

    def px(self, s, t):
        ce, c, _ = self.k[s]
        i = np.searchsorted(ce, t, side="right") - 1
        return c[i] if i >= 0 and t - ce[i] < 3 * MIN else np.nan

    def feats(self, te, T):
        d = {}
        for s, nm in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
            pT, pe = self.px(s, T), self.px(s, te)
            d[f"{nm}_r_since"] = (pT / pe - 1) * 100
            d[f"{nm}_r15"] = (pT / self.px(s, T - 15 * MIN) - 1) * 100
            if nm == "btc":
                d["btc_r5"] = (pT / self.px(s, T - 5 * MIN) - 1) * 100
        i = np.searchsorted(self.h_t, T, side="right") - 1
        d["btc_slope1h_t"] = (self.h_e[i] / self.h_e[i - 3] - 1) * 100 if i >= 3 else np.nan
        i0 = np.searchsorted(self.h_t, te, side="right") - 1
        d["btc_slope1h_te"] = (self.h_e[i0] / self.h_e[i0 - 3] - 1) * 100 if i0 >= 3 else np.nan
        te0 = (te // MIN) * MIN
        d["mvol5_vs_entry5"] = self.qv(T - 5 * MIN, T) / self.qv(te0 - 5 * MIN, te0)
        d["mvol5_vs_pre60"] = self.qv(T - 5 * MIN, T) / (self.qv(te0 - 60 * MIN, te0) / 12)
        j = np.searchsorted(self.g_t, T - 5 * MIN, side="right") - 1
        d["br_up5"], d["agg_tbr5"] = (self.g[j] if j >= 0 and T - self.g_t[j] < 30 * MIN else (np.nan, np.nan))
        return d


# ───────────────────────── cohort ─────────────────────────
def cohort():
    M = pd.read_csv(os.path.join(REP, "lite_wlsweep", "matrix.csv"), low_memory=False)
    U = pd.read_csv(os.path.join(REP, "lite_wlsweep", "menu.csv"))
    M = M.merge(U[["fid", "M2_3", "M3_3", "M1_3", "M2_3_why", "M3_3_why"]], on="fid", how="left")
    M["czone"] = ((M.entry_frenzy_vol_mult >= 20) & (M.entry_btc_1h_slope > 0)).astype(int)
    M["C_menu"] = np.where(M.czone == 1, M.M3_3, M.M2_3)
    M["tpC"] = np.where(M.czone == 1, 3.0, 2.0)
    M["tick_only"] = M.src.isin(["ticks", "ticks+1m"])
    return M


def live_cohort():
    L = pd.read_csv(LIVE_CSV, low_memory=False)
    L = L[(L.entry_strategy == "FRENZY_LITE") & (L.status == "CLOSED")].copy()
    rows = []
    for r in L.itertuples():
        te = int(pd.Timestamp(r.opened_at, tz="UTC").value // 1_000_000)
        cz = int(r.entry_frenzy_vol_mult >= 20 and r.entry_btc_1h_slope > 0)
        rows.append(dict(fid=f"LIVE|{r.pair}|{te}", block="LIVE", year="LIVE", pair=r.pair, te=te, E=float(r.entry_price),
                         tick=float(C.ex_tick(r.pair) or np.nan), src="1m(live fetch)", czone=cz, tpC=3.0 if cz else 2.0,
                         entry_frenzy_vol_mult=r.entry_frenzy_vol_mult, entry_btc_1h_slope=r.entry_btc_1h_slope,
                         entry_frenzy_hours=r.entry_frenzy_hours, entry_atr_pct=r.entry_atr_pct, entry_frenzy_vwap=r.entry_frenzy_vwap,
                         entry_frenzy_spike_at=r.entry_frenzy_spike_at, live_booked=float(r.pnl_percentage), live_why=r.close_reason,
                         day=pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d")))
    return pd.DataFrame(rows)


def live_path_k1(pair, te, E):
    k = k1(pair)
    first = -(-te // MIN) * MIN
    k = k[k.open_time >= first]
    o = k.open_time.values.astype("int64")
    return dict(t=o + MIN, hi=WT.net_pct(k.h.values, E), lo=WT.net_pct(k.l.values, E), cl=WT.net_pct(k.c.values, E),
                op=WT.net_pct(k.o.values, E), bar=np.ones(len(k), bool), end_ms=int(o[-1] + MIN) if len(o) else te, n_tick=0,
                n_bar=len(k), tick_end=te, lim=te + (CAP + 1) * MIN, te=te)


MK = None


def job(rows):
    global MK
    WT._NET_BLOCKED = True
    if MK is None:
        MK = Market()
    tc, out = {}, []
    for r in rows:
        te, E, tick = int(r["te"]), float(r["E"]), float(r["tick"])
        rec = dict(fid=r["fid"])
        if not np.isfinite(tick):
            rec["err"] = "no tick"; out.append(rec); continue
        if r["block"] == "LIVE" and r["pair"] != "DEEPUSDT":
            P = live_path_k1(r["pair"], te, E); rec["path_src"] = "1m(fetched)"
        else:
            P = FX.load_path(r["pair"], te, E, CAP, tc); rec["path_src"] = FX.path_src(P)
        res, tk = price_fill(P, E, tick, r["tpC"])
        rec.update(res)
        # menu parity (C and P1 = M1_3)
        if r["block"] != "LIVE":
            cm = r["C_menu"]
            rec["C_par"] = bool((np.isnan(cm) and np.isnan(res["C"])) or abs(cm - res["C"]) < 1e-9)
            rec["P1_par"] = bool((np.isnan(r["M1_3"]) and np.isnan(res["P1"])) or abs(r["M1_3"] - res["P1"]) < 1e-9)
        if tk is not None:
            k, jT, T, n = tk
            rec.update(path_feats(P, k, jT, T, n))
            pr = parity_loop(P, E, tick, r["tpC"], k)
            rec["loop_par"] = all(pr.values()) if pr else True
            sp = int(pd.Timestamp(r["entry_frenzy_spike_at"], tz="UTC").value // 1_000_000)
            rec.update(kline_feats(r["pair"], te, T, E, float(r["entry_frenzy_vwap"]), sp, k1(r["pair"])))
            rec.update(MK.feats(te, T))
            if r["block"] == "LIVE":
                rec["T_utc"] = pd.Timestamp(T, unit="ms").strftime("%m-%d %H:%M")
        rec["btc_slope1h_te"] = MK.feats(te, te + MIN)["btc_slope1h_te"]
        out.append(rec)
        if len(tc) > 300:
            tc.clear()
        _K1C.clear() if len(_K1C) > 50 else None
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=9); a = ap.parse_args()
    M = cohort()
    M = M[M.block.isin(["BT", "MASTER"])]
    Lv = live_cohort()
    keep = ["fid", "block", "year", "pair", "te", "E", "tick", "src", "czone", "tpC", "C_menu", "M1_3", "entry_frenzy_vol_mult",
            "entry_btc_1h_slope", "entry_frenzy_hours", "entry_atr_pct", "entry_frenzy_vwap", "entry_frenzy_spike_at", "day"]
    items = M[keep].to_dict("records") + Lv.to_dict("records")
    for r in items:
        r.setdefault("C_menu", np.nan); r.setdefault("M1_3", np.nan)
    pairs = sorted({r["pair"] for r in items}); g = {p: i % (a.jobs * 6) for i, p in enumerate(pairs)}
    chunks = [c for c in ([sorted([r for r in items if g[r["pair"]] == i], key=lambda r: (r["pair"], r["te"])) for i in range(a.jobs * 6)]) if c]
    from multiprocessing import Pool
    t0 = time.time()
    with Pool(a.jobs) as Pp:
        res = sum(Pp.map(job, chunks, chunksize=1), [])
    R = pd.DataFrame(res)
    base = pd.concat([M[keep], Lv[[c for c in keep if c in Lv] + ["live_booked", "live_why"]]], ignore_index=True)
    R = base.merge(R, on="fid", how="left")
    R["tick_only"] = R.src.isin(["ticks", "ticks+1m"])
    for p in ["P1"] + POL:
        R["d_" + p] = R[p] - R["C"]
    R.to_csv(os.path.join(OUT, "rows.csv"), index=False)
    bt = R[R.block != "LIVE"]
    msg = [f"priced {len(R)} fills in {time.time() - t0:.0f}s",
           f"C vs menu parity: {int(bt.C_par.sum())}/{len(bt)} (mismatch fids: {bt.fid[~bt.C_par.astype(bool)].tolist()[:10]})",
           f"P1 vs menu M1_3 parity: {int(bt.P1_par.sum())}/{len(bt)}",
           f"P3/P4 vs O.walk_loop (decision at the touch event): {int(R.loop_par.fillna(True).astype(bool).sum())}/{len(R)} ok",
           f"touch fills: {int(R.touch.fillna(False).sum())}, acted (C still open at T): {int(R.acted.fillna(False).sum())}",
           f"btc_slope1h at entry (recomputed) vs stamp: corr {R[['btc_slope1h_te', 'entry_btc_1h_slope']].corr().iloc[0, 1]:.3f}"]
    open(os.path.join(OUT, "price.log"), "w").write("\n".join(msg) + "\n")
    print("\n".join(msg))


if __name__ == "__main__":
    main()
