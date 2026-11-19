#!/usr/bin/env python3
"""FRENZY TA CATALOG (PREREG reports/FRENZY_TA_CATALOG_2026-11-17_PREREG.md, frozen 20:22 UTC) — BUILD step.
Per pair (multiprocessing): continuous 5m series → 15m / 1h resamples → 14 indicator families × LONG/SHORT signals on CLOSED bars of
5m and 15m inside the Frenzy windows [spike close, +72 h] → for EVERY eligible window bar (vol24 ≥ $20 M) the outcome of the three frozen
exits (E1/E2/E3) for both sides, entry at the next bar's open, TP maker strictly-through-by-1-tick, gap-aware stops, real funding →
one-position-per-pair trade selection for all 336 cells.
Output: reports/FRENZY_TA_CATALOG_data/pairs/<PAIR>.npz (eligible bars + outcomes + signal bits + cell trade indices).
READ-ONLY on the bot. Usage: venv/bin/python scripts/frenzy_ta_build.py [--pairs A,B] [--procs 8]"""
import argparse, glob, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "FRENZY_TA_CATALOG_data"); os.makedirs(os.path.join(D, "pairs"), exist_ok=True)
K5M_SRC = [os.path.join(R, "reports", "LEAD1_VWAP_SHORT_data", "k5m"), os.path.join(R, "reports", "backtest_cache_2025", "k5m_full"),
           os.path.join(R, "reports", "backtest_cache", "k5m_full"), os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20_data", "k5m"),
           os.path.join(R, "reports", "POST_FRENZY_PATHS_2026-11-17_data", "k5m")]
FUND_SRC = [os.path.join(R, "reports", "backtest_cache", "funding"), os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20_data", "funding"),
            os.path.join(R, "reports", "POST_FRENZY_PATHS_2026-11-17_data", "funding"), os.path.join(R, "reports", "POST_FRENZY_SHORT_V2_data", "funding"),
            os.path.join(R, "reports", "LEAD1_VWAP_SHORT_data", "funding"), os.path.join(R, "reports", "NEW_SLEEVES_2026-11-17_data", "funding"),
            os.path.join(D, "funding")]
BAR, HR = 300_000, 3_600_000
WIN = 72 * HR
FAMS = ["RSI14X", "RSI2", "MACD", "EMAX", "VWAPX", "BBRE", "BBSQ", "DON", "STOCH", "ST", "VSW", "VSA", "SWING", "ADX"]
TFS = ["5m", "15m"]
SIDES = [1, -1]                       # +1 LONG, −1 SHORT
EXITS = {"E1": (1.5, 1.0, 48), "E2": (3.0, 1.5, 144), "E3": (None, 2.0, 288)}
VOL_GATE = 20e6
FUND_MISS = 0.05                      # % per 8 h against the position when a pair-month is missing


def cells():
    out = []
    for f in FAMS:
        for tf in TFS:
            for s in SIDES:
                for ex in EXITS:
                    for tr in (0, 1):
                        out.append((f, tf, s, ex, tr))
    return out


CELLS = cells()


def load(pair):
    parts = []
    for d in K5M_SRC:
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f):
            try:
                parts.append(pd.read_csv(f, usecols=["open_time", "o", "h", "l", "c", "vol", "qvol"]))
            except Exception:
                pass
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time")
    d = d[d.open_time % BAR == 0]
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, BAR, dtype="int64")
    d = d.set_index("open_time").reindex(grid)
    c = d.c.ffill().values.astype(float)
    o = np.where(np.isnan(d.o.values), c, d.o.values).astype(float)
    h = np.where(np.isnan(d.h.values), c, d.h.values).astype(float)
    l = np.where(np.isnan(d.l.values), c, d.l.values).astype(float)
    v = np.nan_to_num(d.vol.values.astype(float)); qv = np.nan_to_num(d.qvol.values.astype(float))
    return grid, o, h, l, c, v, qv


# ───────────── indicators (pandas Series in, numpy out) ─────────────
def wilder(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy(copy=True)


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().to_numpy(copy=True)


def rsi(c, n):
    d = np.diff(c, prepend=c[0])
    up, dn = wilder(np.maximum(d, 0), n), wilder(np.maximum(-d, 0), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100 - 100 / (1 + up / dn)
    r[dn == 0] = 100.0
    r[:n] = np.nan
    return r


def atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    a = wilder(tr, n); a[:n] = np.nan
    return a


def sh(x, k=1):
    return np.r_[np.full(k, np.nan), x[:-k]]


def cross_up(a, b):
    return (sh(a) < sh(b)) & (a >= b)


def cross_dn(a, b):
    return (sh(a) > sh(b)) & (a <= b)


def supertrend_dir(h, l, c, n=10, m=3.0):
    a = atr(h, l, c, n); hl2 = (h + l) / 2
    ub, lb = hl2 + m * a, hl2 - m * a
    N = len(c); d = np.zeros(N, dtype=np.int8); fu = np.full(N, np.nan); fl = np.full(N, np.nan)
    for i in range(N):
        if np.isnan(a[i]):
            continue
        if i == 0 or np.isnan(fu[i - 1]):
            fu[i], fl[i], d[i] = ub[i], lb[i], 1
            continue
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if d[i - 1] == 1:
            d[i] = -1 if c[i] < fl[i] else 1
        else:
            d[i] = 1 if c[i] > fu[i] else -1
    return d


def adx_di(h, l, c, n=14):
    up = h - sh(h); dn = sh(l) - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = 0
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    atr_ = wilder(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * wilder(pdm, n) / atr_; mdi = 100 * wilder(mdm, n) / atr_
        dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    adx = wilder(np.nan_to_num(dx), n)
    adx[: 2 * n] = np.nan
    return adx, pdi, mdi


def swings(h, l, c):
    """Fractal pivots (2 bars each side), known 2 bars after the pivot. Returns LONG / SHORT break signals."""
    N = len(c)
    ph = np.zeros(N, bool); pl = np.zeros(N, bool)
    hm = pd.Series(h).rolling(5, center=True).max().values; lm = pd.Series(l).rolling(5, center=True).min().values
    ph[2:-2] = (h[2:-2] == hm[2:-2]) & (h[2:-2] > h[1:-3]); pl[2:-2] = (l[2:-2] == lm[2:-2]) & (l[2:-2] < l[1:-3])
    def last_two(flag, val):
        s = np.full(N, np.nan); idx = np.nonzero(flag)[0]
        conf = idx + 2; ok = conf < N
        s[conf[ok]] = val[idx[ok]]
        ser = pd.Series(s)
        last = ser.ffill().values
        prevv = np.full(N, np.nan)
        vals = s[conf[ok]]
        if len(vals) > 1:
            pv = np.full(N, np.nan); pv[conf[ok][1:]] = vals[:-1]
            prevv = pd.Series(pv).ffill().values
        return last, prevv
    lph, pph = last_two(ph, h); lpl, ppl = last_two(pl, l)
    long_ = (lpl > ppl) & (c > lph) & (sh(c) <= sh(lph))
    short_ = (lph < pph) & (c < lpl) & (sh(c) >= sh(lpl))
    return long_, short_


def family_signals(o, h, l, c, v, vwap):
    """Dict fam → (long bool, short bool) on this timeframe's closed bars."""
    S = {}
    r14 = rsi(c, 14)
    S["RSI14X"] = ((sh(r14) < 30) & (r14 >= 30), (sh(r14) > 70) & (r14 <= 70))
    r2 = rsi(c, 2)
    S["RSI2"] = ((r2 < 5) & (sh(r2) >= 5), (r2 > 95) & (sh(r2) <= 95))
    m = ema(c, 12) - ema(c, 26); sg = ema(m, 9); m[:35] = np.nan
    S["MACD"] = (cross_up(m, sg), cross_dn(m, sg))
    e9, e21 = ema(c, 9), ema(c, 21); e9[:21] = np.nan
    S["EMAX"] = (cross_up(e9, e21), cross_dn(e9, e21))
    S["VWAPX"] = ((sh(c) < sh(vwap)) & (c >= vwap), (sh(c) > sh(vwap)) & (c <= vwap))
    mid = pd.Series(c).rolling(20).mean().values; sd = pd.Series(c).rolling(20).std(ddof=0).values
    up, lo = mid + 2 * sd, mid - 2 * sd
    S["BBRE"] = ((sh(c) < sh(lo)) & (c >= lo), (sh(c) > sh(up)) & (c <= up))
    with np.errstate(divide="ignore", invalid="ignore"):
        bbw = (up - lo) / mid
    q10 = pd.Series(bbw).rolling(100).quantile(0.1).values
    sq = sh(bbw) <= sh(q10)
    S["BBSQ"] = (sq & (c > up), sq & (c < lo))
    hh = sh(pd.Series(h).rolling(20).max().values); ll = sh(pd.Series(l).rolling(20).min().values)
    S["DON"] = ((c > hh) & ~(sh(c) > sh(hh)), (c < ll) & ~(sh(c) < sh(ll)))
    h14 = pd.Series(h).rolling(14).max().values; l14 = pd.Series(l).rolling(14).min().values
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 * (c - l14) / (h14 - l14)
    K = pd.Series(raw).rolling(3).mean().values; Dd = pd.Series(K).rolling(3).mean().values
    S["STOCH"] = (cross_up(K, Dd) & (sh(K) < 20), cross_dn(K, Dd) & (sh(K) > 80))
    st = supertrend_dir(h, l, c)
    S["ST"] = ((st == 1) & (sh(st.astype(float)) == -1), (st == -1) & (sh(st.astype(float)) == 1))
    vavg = sh(pd.Series(v).rolling(20).mean().values)
    spk = (v >= 5 * vavg) & (vavg > 0)
    S["VSW"] = (spk & (c > o), spk & (c < o))
    S["VSA"] = (spk & (c < o), spk & (c > o))
    S["SWING"] = swings(h, l, c)
    adx, pdi, mdi = adx_di(h, l, c)
    S["ADX"] = ((adx > 25) & cross_up(pdi, mdi), (adx > 25) & cross_up(mdi, pdi))
    return {k: (np.nan_to_num(a).astype(bool) if a.dtype != bool else a, np.nan_to_num(b).astype(bool) if b.dtype != bool else b) for k, (a, b) in S.items()}


# ───────────── exits (vectorised over entries) ─────────────
def outcomes(o, h, l, c, J, A, side, ex, tick):
    """Entries at 5m bar J (raw open), ATR A. Returns r_raw %, kind (0 TP maker / 1 stop / 2 time), exit bar index."""
    kt, ks, H = EXITS[ex]
    N = len(c); n = len(J)
    r = np.zeros(n); kind = np.zeros(n, np.int8); xi = np.zeros(n, np.int64)
    for a0 in range(0, n, 3000):
        jj = J[a0:a0 + 3000]; aa = A[a0:a0 + 3000]
        idx = jj[:, None] + np.arange(H)[None, :]
        valid = idx < N
        idc = np.minimum(idx, N - 1)
        Oo, Hh, Ll, Cc = o[idc], h[idc], l[idc], c[idc]
        O = o[jj]
        last = np.minimum(jj + H - 1, N - 1) - jj               # time-exit column (truncated at data end)
        if ex != "E3":
            if side == 1:
                SL, TP = O - ks * aa, O + kt * aa
                hs = (Ll <= SL[:, None]) & valid; ht = (Hh >= TP[:, None] + tick) & valid
            else:
                SL, TP = O + ks * aa, O - kt * aa
                hs = (Hh >= SL[:, None]) & valid; ht = (Ll <= TP[:, None] - tick) & valid
            lvl = np.repeat(SL[:, None], H, 1)
        else:
            if side == 1:
                best = np.maximum.accumulate(Hh, axis=1)
                prev = np.concatenate([O[:, None], np.maximum(best[:, :-1], O[:, None])], 1)
                lvl = prev - ks * aa[:, None]
                hs = (Ll <= lvl) & valid
            else:
                best = np.minimum.accumulate(Ll, axis=1)
                prev = np.concatenate([O[:, None], np.minimum(best[:, :-1], O[:, None])], 1)
                lvl = prev + ks * aa[:, None]
                hs = (Hh >= lvl) & valid
            ht = np.zeros_like(hs)
            TP = O
        BIG = H + 1
        fs = np.where(hs.any(1), hs.argmax(1), BIG); ft = np.where(ht.any(1), ht.argmax(1), BIG)
        rows = np.arange(len(jj))
        stop = (fs <= ft) & (fs < BIG); tp = (ft < fs) & (ft < BIG)
        X = Cc[rows, last]; k_ = np.full(len(jj), 2, np.int8); col = last.copy()
        fsc = np.minimum(fs, H - 1)
        sl_lvl = lvl[rows, fsc]; op = Oo[rows, fsc]
        sfill = np.minimum(op, sl_lvl) if side == 1 else np.maximum(op, sl_lvl)
        X = np.where(stop, sfill, X); k_[stop] = 1; col = np.where(stop, fs, col)
        if ex != "E3":
            X = np.where(tp, TP, X); k_[tp] = 0; col = np.where(tp, ft, col)
        r[a0:a0 + len(jj)] = side * (X / O - 1) * 100
        kind[a0:a0 + len(jj)] = k_; xi[a0:a0 + len(jj)] = jj + col
    return r, kind, xi


def funding_for(pair):
    xs = []
    for d in FUND_SRC:
        for f in glob.glob(os.path.join(d, pair + ".*")):
            if not (f.endswith(".csv") or f.endswith(".part")):
                continue
            b = os.path.basename(f)
            if b.split(".")[0] != pair:
                continue
            try:
                x = pd.read_csv(f)
            except Exception:
                continue
            if len(x) and {"t", "rate"} <= set(x.columns):
                xs.append(x[["t", "rate"]])
            elif f.endswith(".part"):
                xs.append(pd.DataFrame({"t": [], "rate": []}))
    # months covered = months with a settlement, plus months whose .part file exists (an empty part = pair not listed that month)
    months = set()
    for d in FUND_SRC:
        for f in glob.glob(os.path.join(d, pair + ".*.part")):
            months.add(os.path.basename(f).split(".")[1])
    if not xs:
        return np.zeros(0, "int64"), np.zeros(1), months
    x = pd.concat(xs).dropna().drop_duplicates("t").sort_values("t")
    t = x.t.values.astype("int64") // 1000 * 1000
    months |= set(pd.to_datetime(t, unit="ms").strftime("%Y-%m"))
    return t, np.r_[0.0, np.cumsum(x.rate.values.astype(float))], months


def work(pair, eps):
    x = load(pair)
    if x is None:
        return pair, None
    t, o, h, l, c, v, qv = x
    N = len(t)
    tick = np.diff(np.unique(np.r_[o, h, l, c]))
    tick = float(tick[tick > 1e-15].min()) if len(tick) else 0.0
    q = v * (h + l + c) / 3.0
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    # 1h EMA50 slope, as of each 5m bar's close (last CLOSED 1h bar)
    hr = t // HR * HR
    g = pd.DataFrame({"hr": hr, "c": c}).groupby("hr").c.last()
    e50 = ema(g.values, 50); slope = e50 - sh(e50); slope[:50] = np.nan
    hk = np.searchsorted(g.index.values, t + BAR - HR, side="right") - 1      # last hour with open + 1 h ≤ bar close
    slope5 = np.where(hk >= 0, slope[np.maximum(hk, 0)], np.nan)
    # episode windows: per 5m bar → episode index (latest spike with spike_ms ≤ bar close ≤ spike_ms + 72 h), VWAP of that episode
    ep_of = np.full(N, -1, np.int32); vw = np.full(N, np.nan); vwp = np.full(N, np.nan); phase = np.zeros(N, np.int8)
    eps = eps.sort_values("on_ms").reset_index(drop=True)
    chk = []
    for k, e in eps.iterrows():
        on = int(np.searchsorted(t, e.on_ms))
        if on >= N or t[on] != e.on_ms:
            continue
        a = on; b = min(N, int(np.searchsorted(t, e.spike_ms + WIN - BAR, side="right")))
        tp_ = (h[a:b] + l[a:b] + c[a:b]) / 3.0
        cq = np.cumsum(q[a:b]); vv = np.where(cq > 0, np.cumsum(tp_ * q[a:b]) / np.where(cq > 0, cq, 1), c[a:b])
        ep_of[a:b] = k; vw[a:b] = vv; vwp[a + 1:b] = vv[:-1]; vwp[a] = np.nan
        ph_ = (t[a:b] < e.off_ms).astype(np.int8); phase[a:b] = ph_
        j = int(np.searchsorted(t, e.chk_ms))
        if a <= j < b and np.isfinite(e.chk_vwap):
            chk.append((vw[j], e.chk_vwap))
    out = dict(pair=pair, tick=tick, n_eps=len(eps), vwchk=np.array(chk) if chk else np.zeros((0, 2)))
    for tf in TFS:
        if tf == "5m":
            T, Oo, Hh, Ll, Cc, Vv = t, o, h, l, c, v
            last5 = np.arange(N)                     # 5m index of the signal bar's last 5m bar
            VW = vw; VWP = vwp
        else:
            g15 = t // (3 * BAR)
            st = np.r_[0, np.nonzero(np.diff(g15))[0] + 1]; en = np.r_[st[1:], N] - 1
            T = t[st]; Oo = o[st]; Cc = c[en]
            Hh = np.maximum.reduceat(h, st); Ll = np.minimum.reduceat(l, st); Vv = np.add.reduceat(v, st)
            last5 = en
            VW = vw[en]
            # previous 15m close vs the SAME episode's VWAP at that close
            VWP = np.full(len(st), np.nan)
            pe = np.r_[-1, en[:-1]]
            same = (pe >= 0) & (ep_of[np.maximum(pe, 0)] == ep_of[en]) & (ep_of[en] >= 0)
            VWP[same] = vw[pe[same]]
        sig = family_signals(Oo, Hh, Ll, Cc, Vv, np.where(np.isnan(VW), np.nan, VW))
        # VWAP cross needs the previous bar's VWAP of the same anchor (family_signals used shift of VW; redo exactly)
        cp = sh(Cc)
        sig["VWAPX"] = (np.nan_to_num((cp < VWP) & (Cc >= VW)).astype(bool), np.nan_to_num((cp > VWP) & (Cc <= VW)).astype(bool))
        A = atr(Hh, Ll, Cc, 14)
        el = np.nonzero((ep_of[last5] >= 0) & (vol24[last5] >= VOL_GATE) & np.isfinite(A) & (A > 0) & (last5 + 1 < N))[0]
        J = last5[el] + 1                            # entry 5m bar = next bar's open
        res = dict(sig_ms=(t[last5[el]] + BAR).astype("int64"), entry_ms=t[J].astype("int64"), ep=ep_of[last5[el]], phase=phase[last5[el]],
                   atr_pct=(A[el] / o[J] * 100).astype(np.float32), slope=slope5[last5[el]].astype(np.float32), vol24=vol24[last5[el]].astype(np.float32))
        bits = np.zeros(len(el), np.int64)
        for fi, f in enumerate(FAMS):
            lo_, sh_ = sig[f]
            bits |= lo_[el].astype(np.int64) << (2 * fi); bits |= sh_[el].astype(np.int64) << (2 * fi + 1)
        res["bits"] = bits
        ft, fc, months = funding_for(pair)
        for s in SIDES:
            for ex in EXITS:
                r, kd, xi = outcomes(o, h, l, c, J, A[el], s, ex, tick)
                xms = t[np.minimum(xi, N - 1)] + BAR
                if len(ft):
                    fs_ = fc[np.searchsorted(ft, xms, "right")] - fc[np.searchsorted(ft, t[J], "right")]
                else:
                    fs_ = np.zeros(len(J))
                fund = -s * fs_ * 100
                mo_e = pd.to_datetime(t[J], unit="ms").strftime("%Y-%m").values; mo_x = pd.to_datetime(xms, unit="ms").strftime("%Y-%m").values
                miss = np.array([(a_ not in months) or (b_ not in months) for a_, b_ in zip(mo_e, mo_x)], bool)
                fund = np.where(miss, -(xms - t[J]) / (8 * HR) * FUND_MISS, fund)
                key = f"{tf}_{'L' if s == 1 else 'S'}_{ex}"
                res[key + "_r"] = r.astype(np.float32); res[key + "_k"] = kd; res[key + "_x"] = xms.astype("int64")
                res[key + "_f"] = fund.astype(np.float32); res[key + "_m"] = miss
        out[tf] = res
    # one-position-per-pair trade selection for every cell (indices into the TF's eligible arrays)
    sel = {}
    for ci, (f, tf, s, ex, tr) in enumerate(CELLS):
        res = out[tf]; fi = FAMS.index(f)
        m = ((res["bits"] >> (2 * fi + (0 if s == 1 else 1))) & 1).astype(bool)
        if tr:
            m &= (res["slope"] > 0) if s == 1 else (res["slope"] < 0)
        idx = np.nonzero(m)[0]
        if not len(idx):
            continue
        key = f"{tf}_{'L' if s == 1 else 'S'}_{ex}"
        em, xm = res["entry_ms"][idx], res[key + "_x"][idx]
        keep = []; busy = -1
        for a_, e_, x_ in zip(idx, em, xm):
            if e_ < busy:
                continue
            keep.append(a_); busy = x_
        sel[ci] = np.array(keep, np.int64)
    out["sel"] = sel
    return pair, out


def _one(args):
    pair, eps = args
    try:
        p, out = work(pair, eps)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("ERR", pair, e, flush=True); return pair, 0
    if out is None:
        return pair, 0
    flat = {"tick": out["tick"], "vwchk": out["vwchk"]}
    for tf in TFS:
        for k, v in out[tf].items():
            flat[f"{tf}__{k}"] = v
    for ci, v in out["sel"].items():
        flat[f"sel__{ci}"] = v
    np.savez_compressed(os.path.join(D, "pairs", pair + ".npz"), **flat)
    return pair, len(out["5m"]["entry_ms"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=""); ap.add_argument("--procs", type=int, default=8)
    A = ap.parse_args()
    meta = pd.read_csv(os.path.join(D, "episodes_meta.csv"))
    pairs = sorted(meta.pair.unique())
    if A.pairs:
        pairs = A.pairs.split(",")
    jobs = [(p, meta[meta.pair == p]) for p in pairs]
    t0 = time.time(); tot = 0
    with mp.get_context("fork").Pool(A.procs) as pool:
        for k, (p, n) in enumerate(pool.imap_unordered(_one, jobs, chunksize=1)):
            tot += n
            if k % 25 == 0:
                print(time.strftime("%H:%M:%S"), k, len(pairs), p, n, tot, f"{time.time() - t0:.0f}s", flush=True)
    print("done", tot)
