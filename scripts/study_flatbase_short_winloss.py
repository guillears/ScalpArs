#!/usr/bin/env python3
"""FRENZY flat-base short — WINNERS vs LOSERS market conditions (2026-10-22). Pre-registration:
reports/FRENZY_FLATBASE_SHORT_2026-10-22_WINLOSS_PREREG.md. READ-ONLY on the bot.

Stages
  fetch   Info-panel circulating supply for cohort pairs missing from reports/cache_vol_mcap/bapi_supply.json (the bot's own mcap source,
          services/mcap_service: lookup_candidates + parse_detail(expect=)); ≥ 1.5 s spacing, stop on 418/429, honours binance_cooldown.
  build   per pair: S4 trades for TP −8/−10 · 72 h · stops 5/8/10/15/25 (same model as study_flatbase_short_exits.py), the fair random-entry
          null for the same exits, and every entry-bar feature; BTC + breadth panel; writes the trade/feature table.
  screen  1D / 2D screens with day-preserving + trade-shuffle nulls, OOS variants A/B, stop table, add-trades question, report tables.
Usage: venv/bin/python scripts/study_flatbase_short_winloss.py fetch|build|screen [--procs 7]
"""
import argparse, glob, io, json, math, os, sys, time, zlib
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402

OUT = C.OUTP + "_WINLOSS"
FEAT_PKL = os.path.join(C.D, "winloss_features.pkl")
SUPPLY = os.path.join(C.D, "winloss_supply.json")
HM = 72 * 12
NB = 200
# (tp, stop) at 72 h
EXITS = [(8, 5), (8, 8), (8, 10), (8, 15), (8, 25), (10, 15)]
EXN = [f"tp{tp}|72h|stop{s}" for tp, s in EXITS]
MAIN = "tp8|72h|stop15"
NEIGH = ["tp8|72h|stop15", "tp8|72h|stop10", "tp8|72h|stop25", "tp10|72h|stop15"]
NX = len(EXITS)
THR = 44.39
USD = 190.0


# ───────────────────────────── fetch ─────────────────────────────
def stage_fetch():
    import requests
    from services.mcap_service import URL, lookup_candidates, parse_detail
    import binance_cooldown as BC
    base = json.load(open(os.path.join(C.R, "reports", "cache_vol_mcap", "bapi_supply.json")))
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    S = pd.read_pickle(os.path.join(C.D, "signals.pkl"))
    pairs = sorted(set(S[S.entry == "S4"].pair) & set(E[E.spike >= C.B_YEAR0].pair))
    BD = os.path.join(C.R, "reports", "cache_vol_mcap", "bapi")
    last = [0.0]; out = {}

    def fetch(sym):
        f = os.path.join(BD, f"{sym}.json")
        if os.path.exists(f):
            return json.load(open(f))
        if BC.active():
            raise SystemExit("shared Binance cooldown active — stopping")
        w = 1.5 - (time.time() - last[0])
        if w > 0:
            time.sleep(w)
        last[0] = time.time()
        r = requests.get(URL, params={"symbol": sym}, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code in (418, 429):
            raise SystemExit(f"bapi {r.status_code} on {sym} — stopping")
        j = r.json() if r.status_code == 200 else {"_status": r.status_code}
        json.dump(j, open(f, "w")); return j
    for p in pairs:
        if p in base and base[p].get("cs"):
            out[p] = dict(base[p], src="oct08"); continue
        res = None
        for sym in lookup_candidates(p):
            j = fetch(sym); mc, rk = parse_detail(j, expect=sym)
            if mc is not None:
                d = j.get("data") or {}
                try:
                    cs = float(d.get("cs"))
                except (TypeError, ValueError):
                    cs = None
                res = dict(symbol=sym, mc=mc, rk=rk, cs=cs if cs and cs > 0 else None, mult=None, src="fetch1022"); break
        out[p] = res or dict(why="no mc")
    json.dump(out, open(SUPPLY, "w"), indent=1)
    print(f"supply readable {sum(1 for v in out.values() if v.get('cs'))}/{len(pairs)}")


# ───────────────────────────── helpers ─────────────────────────────
def rsi(x, n=14):
    d = np.diff(x, prepend=np.nan)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean().values
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 - 100 / (1 + up / dn)


def adx(h, l, c, n=14):
    pc = np.r_[np.nan, c[:-1]]; ph = np.r_[np.nan, h[:-1]]; pl = np.r_[np.nan, l[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - pc), np.abs(l - pc)], 1)
    up = h - ph; dn = pl - l
    pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = lambda x: pd.Series(x).ewm(alpha=1 / n, adjust=False).mean().values
    atr = a(tr)
    with np.errstate(invalid="ignore", divide="ignore"):
        pdi = 100 * a(pdm) / atr; ndi = 100 * a(ndm) / atr
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    return a(np.nan_to_num(dx))


def hourly(t, o, h, l, c):
    """complete 1 h bars from the 5m grid, indexed by hour END (ms)."""
    df = pd.DataFrame({"t": t, "o": o, "h": h, "l": l, "c": c}); df["hr"] = df.t // C.HR * C.HR
    g = df.groupby("hr").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("t", "size"))
    g = g[g.n == 12]
    return g.index.values.astype("int64") + C.HR, g.o.values, g.h.values, g.l.values, g.c.values


def first_idx(mask):
    return np.where(mask.any(1), mask.argmax(1), HM + 10)


def paths(arr, idx, n):
    t, o, h, l, c = arr
    m = len(idx)
    H = np.full((m, HM), np.nan); Lo = H.copy(); O = H.copy(); Cc = H.copy(); T = np.zeros((m, HM), dtype="int64"); av = np.zeros(m, dtype="int64")
    for r, i in enumerate(idx):
        a, b = i + 1, min(n, i + 1 + HM); k = b - a; av[r] = max(k, 0)
        if k > 0:
            H[r, :k] = h[a:b]; Lo[r, :k] = l[a:b]; O[r, :k] = o[a:b]; Cc[r, :k] = c[a:b]; T[r, :k] = t[a:b]
            if k < HM:
                T[r, k:] = t[b - 1] + C.BAR * np.arange(1, HM - k + 1)
    return H, Lo, O, Cc, T, av


def simulate(E, H, Lo, O, Cc, T, av):
    """same model as study_flatbase_short_exits.simulate (gap-aware stop at max(level, open) + slip, stop-first, TP at min(level, open) + slip,
    time exit at the 72 h close), restricted to EXITS; also returns 'stopped then later touched the TP inside 72 h'."""
    n = len(E); r = np.arange(n)
    hh = np.where(np.isnan(H), -np.inf, H); ll = np.where(np.isnan(Lo), np.inf, Lo)
    cmh = np.maximum.accumulate(hh, 1); cml = np.minimum.accumulate(ll, 1)
    G = np.zeros((n, NX)); XT = np.zeros((n, NX), dtype="int64"); RS = np.zeros((n, NX), dtype="int8"); LATE = np.zeros((n, NX), bool)
    for j, (tp, s) in enumerate(EXITS):
        hb = 72 * 12
        kmax = np.minimum(hb, av)
        ks = first_idx(cmh >= (E * (1 + s / 100))[:, None]); kt = first_idx(cml <= (E * (1 - tp / 100))[:, None])
        lvl = E * (1 + s / 100)
        ev = np.minimum(ks, kt); has = ev < kmax
        kx = np.where(has, ev, np.maximum(kmax - 1, 0)); kc = np.clip(kx, 0, HM - 1)
        g = (E - Cc[r, kc] * (1 + C.SLIP)) / E * 100
        rs = np.where(kmax < hb, 5, 0).astype("int8")
        st = has & (ks == ev)
        Sx = np.maximum(lvl, O[r, np.clip(ks, 0, HM - 1)]) * (1 + C.SLIP)
        g = np.where(st, np.maximum((E - Sx) / E * 100, -100.0), g); rs = np.where(st, 1, rs)
        tt = has & (kt == ev) & ~st
        Tx = np.minimum(E * (1 - tp / 100), O[r, np.clip(kt, 0, HM - 1)]) * (1 + C.SLIP)
        g = np.where(tt, (E - Tx) / E * 100, g); rs = np.where(tt, 3, rs)
        # stopped, then the TP level is touched later inside the same 72 h window (kt ≥ ks: same-bar counts as "stop-first")
        LATE[:, j] = st & (kt < kmax) & (kt >= ks)
        G[:, j] = g; RS[:, j] = rs; XT[:, j] = T[r, kc] + C.BAR
    return G, XT, RS, LATE


def one_per_pair(entry_ms, XT):
    busy = np.full(XT.shape[1], -1, dtype="int64"); TK = np.zeros(XT.shape, bool)
    for k in range(len(entry_ms)):
        tk = entry_ms[k] >= busy; TK[k] = tk; busy = np.where(tk, XT[k], busy)
    return TK


# ───────────────────────────── build: per pair ─────────────────────────────
def worker(args):
    pair, eps, sig, sup = args
    full = C.load_full(pair)
    if full is None:
        return None
    t, o, h, l, c, v, qv, real = full
    n = len(t)
    F = C.funding_table(pair)
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    q4 = pd.Series(qv).rolling(48, min_periods=48).sum().values
    q = v * (h + l + c) / 3.0; tpx = (h + l + c) / 3.0
    pos = lambda ts: int((ts - t[0]) // C.BAR)
    he, ho, hh_, hl, hc = hourly(t, o, h, l, c)
    hrsi = rsi(hc)
    mult = None; cs = None
    if sup and sup.get("cs"):
        cs = sup["cs"]; unit = sup["mc"] / cs
        mult = sup.get("mult") or (10 ** round(math.log10(c[-1] / unit)) if c[-1] > 0 else None)
    rows = []
    sig = sig.sort_values("entry_ms")
    epk = {e.on_t: e for e in eps.itertuples()}
    for s in sig.itertuples():
        e = epk.get(s.ep_on_t)
        if e is None:
            continue
        i = pos(s.i_t); on = pos(e.on_t)
        if i <= on or i >= n:
            continue
        seg = slice(on, i + 1)
        pk = on + int(np.argmax(h[seg])); hp = h[pk]
        vw = (tpx[seg] * q[seg]).sum() / max(q[seg].sum(), 1e-12)
        em = int(s.entry_ms)
        r_ = dict(pair=pair, ep_on_t=int(e.on_t), i_t=int(s.i_t), entry_ms=em, close=float(c[i]), E=float(s.E), base=float(e.base),
                  FLAT14=e.FLAT14, FLAT30=e.FLAT30, RV14=e.RV14, DEADVOL=e.DEADVOL, DIST90=e.DIST90, spike_ms=int(e.spike_ts))
        r_["pump_pct"] = (hp / e.base - 1) * 100
        r_["peak_age_h"] = (i - pk) / 12; r_["ep_age_h"] = (i - on) / 12
        r_["below_vwap_pct"] = (c[i] / vw - 1) * 100
        r_["retrace_frac"] = (hp - c[i]) / (hp - e.base) if hp > e.base else np.nan
        r_["above_base_pct"] = (c[i] / e.base - 1) * 100
        mq = np.nanmax(q4[seg]) if np.isfinite(q4[seg]).any() else np.nan
        r_["vol_decay"] = q4[i] / mq if mq and mq > 0 else np.nan
        r_["vol24_musd"] = vol24[i] / 1e6
        pre = vol24[on - 1] if on >= 1 else np.nan
        r_["vol_vs_prepump"] = vol24[i] / pre if pre and pre > 0 else np.nan
        for k, b in (("coin_r1h", 12), ("coin_r4h", 48), ("coin_r24h", 288)):
            r_[k] = (c[i] / c[i - b] - 1) * 100 if i >= b else np.nan
        kh = np.searchsorted(he, em, side="right") - 1
        r_["coin_rsi1h"] = hrsi[kh] if kh >= 15 else np.nan
        # funding known at entry
        if F is not None and len(F[0]):
            ts, rt = F
            a = np.searchsorted(ts, em, side="right")
            if a >= 2 and ts[a - 1] >= em - 9 * C.HR:
                iv = max(ts[a - 1] - ts[a - 2], C.HR)
                r_["fund8"] = rt[a - 1] * (8 * C.HR / iv) * 100
                w24 = (ts > em - 24 * C.HR) & (ts <= em)
                r_["fund24"] = rt[w24].sum() * 100
                wpr = (ts > em - 72 * C.HR) & (ts <= em - 24 * C.HR)
                if wpr.sum() >= 2:
                    ivs = np.diff(ts[wpr]); ivm = np.median(ivs) if len(ivs) else 8 * C.HR
                    r_["fund_trend"] = r_["fund8"] - rt[wpr].mean() * (8 * C.HR / ivm) * 100
        r_["age_days"] = (t[i] - t[0]) / C.DAY
        r_["age_capped"] = bool(t[0] <= C.T0_FEAT + C.DAY)
        if cs and mult:
            mc_i = cs * c[i] / mult
            r_["mcap_musd"] = mc_i / 1e6
            r_["turnover"] = vol24[i] / mc_i
            mcs = cs * c[seg] / mult
            r_["turnover_peak"] = np.nanmax(vol24[seg] / mcs)
            r_["turnover_pre"] = pre / (cs * c[on - 1] / mult) if on >= 1 and pre == pre else np.nan
        r_["hour"] = pd.Timestamp(em, unit="ms").hour
        r_["weekday"] = pd.Timestamp(em, unit="ms").weekday()
        rows.append((i, r_))
    if not rows:
        return None
    idx = np.array([a for a, _ in rows]); meta = pd.DataFrame([b for _, b in rows])
    H, Lo, O, Cc, T, av = paths((t, o, h, l, c), idx, n)
    E0 = meta.E.values
    G, XT, RS, LATE = simulate(E0, H, Lo, O, Cc, T, av)
    fu, ms = C.funding_sum(F, np.repeat(meta.entry_ms.values[:, None], NX, 1).ravel(), XT.ravel())
    NET = G - C.FEES + fu.reshape(G.shape)
    flat = (meta.FLAT14 <= THR).values
    TK = np.zeros(G.shape, bool)
    for grp in (flat, (~flat) & (meta.FLAT14 <= 55).values, (meta.FLAT14 > 55).values & (meta.FLAT14 <= 70).values):
        rr = np.nonzero(grp)[0]
        if len(rr):
            TK[rr] = one_per_pair(meta.entry_ms.values[rr], XT[rr])
    for j, nm in enumerate(EXN):
        meta[f"net|{nm}"] = NET[:, j]; meta[f"nof|{nm}"] = G[:, j] - C.FEES; meta[f"rs|{nm}"] = RS[:, j]
        meta[f"xt|{nm}"] = XT[:, j]; meta[f"tk|{nm}"] = TK[:, j]; meta[f"late|{nm}"] = LATE[:, j]
    # fair non-anticipating random-entry null (flat episodes only), same exits / costs / funding / one short per pair
    rng = np.random.default_rng(zlib.crc32(pair.encode()) + 11)
    busy = np.full((NB, NX), -1, dtype="int64"); NN = []
    for e in eps[eps.FLAT14 <= THR].sort_values("on_t").itertuples():
        on = pos(e.on_t)
        fs = pos(e.first_seen_t) if e.first_seen_t is not None and not pd.isna(e.first_seen_t) else 10**12
        hi = min(n - 2, on + C.SEARCH)
        cand = np.arange(on + 24, hi + 1)
        if len(cand):
            cand = cand[(cand >= fs) & (np.nan_to_num(vol24[cand]) >= 20e6)]
        if not len(cand):
            continue
        dd = rng.integers(on + 24, hi + 1, NB); pp = np.searchsorted(cand, dd); okd = pp < len(cand)
        if not okd.any():
            continue
        ri = cand[np.minimum(pp, len(cand) - 1)]
        Hn, Ln, On, Cn, Tn, avn = paths((t, o, h, l, c), ri, n)
        Gn, XTn, _, _ = simulate(c[ri] * (1 - C.SLIP), Hn, Ln, On, Cn, Tn, avn)
        emn = t[ri] + C.BAR
        fun, _ = C.funding_sum(F, np.repeat(emn[:, None], NX, 1).ravel(), XTn.ravel())
        net = Gn - C.FEES + fun.reshape(Gn.shape)
        tk = (emn[:, None] >= busy) & okd[:, None]
        busy = np.where(tk, XTn, busy)
        tov = (vol24[ri] / (cs * c[ri] / mult)).astype(np.float32) if (cs and mult) else np.full(NB, np.nan, np.float32)
        NN.append(dict(pair=pair, on_t=int(e.on_t), spike=int(e.spike_ts), em=emn, net=np.where(tk, net, np.nan).astype(np.float32), tov=tov))
    # hourly closes for the breadth panel are built separately
    return pair, meta, NN


def hourly_close(pair):
    parts = []
    for i, d in enumerate(C.W.SRC):
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f):
            try:
                x = pd.read_csv(f, usecols=["open_time", "c", "qvol"])
            except Exception:
                continue
            if i == 0:
                x = x[x.open_time < pd.Timestamp("2026-01-10").value // 10**6]
            parts.append(x)
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("open_time", keep="last")
    d = d[d.open_time >= pd.Timestamp("2024-12-01").value // 10**6]
    d = d[(d.open_time + C.BAR) % C.HR == 0]
    d = d[d.qvol > 0]
    if len(d) < 200:
        return None
    return pair, pd.Series(d.c.values, index=(d.open_time.values + C.BAR).astype("int64"))


def stage_build(procs):
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    E = E[E.spike >= C.B_YEAR0].copy()
    S0 = pd.read_pickle(os.path.join(C.D, "signals.pkl")); S0 = S0[S0.entry == "S4"]
    fk = set(zip(E.pair, E.on_t)); S0 = S0[[k in fk for k in zip(S0.pair, S0.ep_on_t)]]
    sup = json.load(open(SUPPLY)) if os.path.exists(SUPPLY) else {}
    jobs = [(p, g, S0[S0.pair == p], sup.get(p)) for p, g in E.groupby("pair") if (S0.pair == p).any() or (g.FLAT14 <= THR).any()]
    metas, nulls = [], []
    with mp.Pool(procs) as pool:
        for k, res in enumerate(pool.imap_unordered(worker, jobs)):
            if res is None:
                continue
            _, m, nn = res; metas.append(m); nulls += nn
    M = pd.concat(metas, ignore_index=True)
    print("trades rows", len(M), "null episodes", len(nulls))
    # breadth panel
    pairs = sorted({os.path.basename(f)[:-4] for d in C.W.SRC for f in glob.glob(os.path.join(d, "*USDT.csv"))})
    with mp.Pool(procs) as pool:
        hc = [x for x in pool.imap_unordered(hourly_close, pairs, chunksize=8) if x is not None]
    P = pd.DataFrame({p: s for p, s in hc if p not in ("BTCUSDT", "BTCDOMUSDT")})
    P = P.sort_index()
    idx = np.arange(P.index.min(), P.index.max() + 1, C.HR, dtype="int64"); P = P.reindex(idx)
    r24 = P / P.shift(24) - 1; r168 = P / P.shift(168) - 1
    nb24 = r24.notna().sum(1); br24 = (r24 > 0).sum(1) / nb24.replace(0, np.nan); br7 = (r168 > 0).sum(1) / r168.notna().sum(1).replace(0, np.nan)
    med24 = r24.median(1)
    # BTC
    bt = C.load_full("BTCUSDT"); t, o, h, l, c, v, qv, real = bt
    he, ho, hh, hl, hcl = hourly(t, o, h, l, c)
    e20 = pd.Series(hcl).ewm(span=20, adjust=False).mean().values; e50 = pd.Series(hcl).ewm(span=50, adjust=False).mean().values
    brsi = rsi(hcl); badx = adx(hh, hl, hcl)
    lr = np.r_[np.nan, np.diff(np.log(c))]
    rv24 = pd.Series(lr).rolling(288).std().values * np.sqrt(288 * 365) * 100
    hi24 = pd.Series(h).rolling(288).max().values; lo24 = pd.Series(l).rolling(288).min().values
    Eall = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    sp_all = np.sort(Eall.spike_ts.values.astype("int64")); sp_flat = np.sort(Eall[Eall.FLAT14 <= THR].spike_ts.values.astype("int64"))
    em = M.entry_ms.values.astype("int64")
    ib = ((em - C.BAR - t[0]) // C.BAR).astype(int)          # BTC bar closing at the entry
    ok = (ib >= 2016) & (ib < len(t))
    ibc = np.clip(ib, 0, len(t) - 1)
    for k, b in (("btc_r1h", 12), ("btc_r4h", 48), ("btc_r24h", 288), ("btc_r7d", 2016)):
        M[k] = np.where(ok, (c[ibc] / c[ibc - b] - 1) * 100, np.nan)
    kh = np.searchsorted(he, em, side="right") - 1
    M["btc_ema_gap"] = (e20[kh] / e50[kh] - 1) * 100; M["btc_rsi1h"] = brsi[kh]; M["btc_adx1h"] = badx[kh]
    M["btc_dist_hi24"] = (c[ibc] / hi24[ibc] - 1) * 100; M["btc_dist_lo24"] = (c[ibc] / lo24[ibc] - 1) * 100
    M["btc_rv24"] = rv24[ibc]
    kp = np.searchsorted(P.index.values, em, side="right") - 1
    M["breadth24"] = br24.values[kp] * 100; M["breadth7d"] = br7.values[kp] * 100
    M["alt_minus_btc24"] = med24.values[kp] * 100 - M.btc_r24h
    M["pumps24"] = np.searchsorted(sp_all, em, side="right") - np.searchsorted(sp_all, em - C.DAY, side="right")
    M["flatpumps24"] = np.searchsorted(sp_flat, em, side="right") - np.searchsorted(sp_flat, em - C.DAY, side="right")
    # descriptive look-ahead: BTC over the main cell's hold
    xt = M[f"xt|{MAIN}"].values.astype("int64"); ix = np.clip(((xt - C.BAR - t[0]) // C.BAR).astype(int), 0, len(t) - 1)
    M["btc_hold_ret"] = (c[ix] / c[ibc] - 1) * 100
    M["fund_paid_hold"] = M[f"net|{MAIN}"] - M[f"nof|{MAIN}"]
    pd.to_pickle(dict(M=M, NULL=nulls), FEAT_PKL)
    print("saved", FEAT_PKL)


# ───────────────────────────── screen ─────────────────────────────
COIN = ["pump_pct", "peak_age_h", "ep_age_h", "below_vwap_pct", "retrace_frac", "above_base_pct", "vol_decay", "vol24_musd", "vol_vs_prepump",
        "coin_r1h", "coin_r4h", "coin_r24h", "coin_rsi1h", "fund8", "fund24", "fund_trend", "age_days", "mcap_musd", "turnover",
        "turnover_peak", "turnover_pre", "hour", "FLAT14", "FLAT30", "RV14", "DEADVOL", "DIST90"]
MKT = ["btc_r1h", "btc_r4h", "btc_r24h", "btc_r7d", "btc_ema_gap", "btc_adx1h", "btc_rsi1h", "btc_dist_hi24", "btc_dist_lo24", "btc_rv24",
       "breadth24", "breadth7d", "alt_minus_btc24", "pumps24", "flatpumps24", "weekday"]
SIGNED = {"coin_r1h", "coin_r4h", "coin_r24h", "fund8", "fund24", "fund_trend", "btc_r1h", "btc_r4h", "btc_r24h", "btc_r7d", "btc_ema_gap",
          "alt_minus_btc24"}
CAT = {"hour", "weekday"}
NPERM = 1000


def periods(M):
    dt = pd.to_datetime(M.entry_ms, unit="ms")
    M["dt"] = dt; M["day"] = dt.dt.strftime("%Y-%m-%d")
    M["per"] = np.select([dt >= C.B_LIVE0, dt >= C.B_MAST0, dt >= pd.Timestamp("2026-07-01"), dt >= pd.Timestamp("2026-01-01")],
                         ["1_live", "2_master", "2026H2", "2026H1"], "2025")
    LMr = pd.read_csv(C.OUTP + "_live_master.csv", usecols=["pair", "spike", "real"])
    rmap = dict(zip(zip(LMr.pair, LMr.spike.astype(str).str[:16]), LMr.real))
    spk = pd.to_datetime(M.spike_ms - C.BAR, unit="ms")   # episodes_compact.spike = spike_ts − 5 min? resolved below
    M["real_ep"] = False
    return M


def bucketize(x, gran, edges=None, feat=""):
    """→ (codes float with nan for unscored, edges)."""
    x = np.asarray(x, float)
    if feat in CAT:
        if gran == "cat":
            return np.where(np.isnan(x), np.nan, x), None
        if gran == "t3":
            if feat == "hour":
                return np.where(np.isnan(x), np.nan, x // 8), None
            return np.where(np.isnan(x), np.nan, np.select([x <= 2, x <= 4], [0, 1], 2)), None
        if gran == "sign":
            if feat == "weekday":
                return np.where(np.isnan(x), np.nan, (x >= 5).astype(float)), None
            return np.where(np.isnan(x), np.nan, (x >= 12).astype(float)), None
        return None, None
    if gran == "sign":
        if feat == "turnover" or feat.startswith("turnover"):
            ed = np.array([1.0])
        elif feat in SIGNED:
            ed = np.array([0.0])
        else:
            ed = edges if edges is not None else np.array([np.nanmedian(x)])
    else:
        k = 3 if gran == "t3" else 5
        ed = edges if edges is not None else np.nanquantile(x, np.arange(1, k) / k)
    cd = np.searchsorted(ed, x, side="right").astype(float)
    cd[np.isnan(x)] = np.nan
    return cd, ed


def day_perm_index(days, rng, P):
    """index permutations that move whole days' outcome vectors among days with the same trade count."""
    d = pd.Series(np.arange(len(days))).groupby(days).apply(list)
    bys = {}
    for k, v in d.items():
        bys.setdefault(len(v), []).append(v)
    out = np.empty((P, len(days)), dtype=np.int64)
    for p in range(P):
        perm = np.empty(len(days), dtype=np.int64)
        for sz, lst in bys.items():
            o = rng.permutation(len(lst))
            for a, b in zip(lst, o):
                perm[a] = lst[b]
        out[p] = perm
    return out


def zstat(Bsum, Bn, Ssum, Sn):
    """one-sided z for WR(rest) − WR(bucket); arrays broadcast."""
    Rn = Sn - Bn; Rsum = Ssum - Bsum
    p = Ssum / np.maximum(Sn, 1)
    se = np.sqrt(np.maximum(p * (1 - p), 1e-9) * (1 / np.maximum(Bn, 1) + 1 / np.maximum(Rn, 1)))
    return (Rsum / np.maximum(Rn, 1) - Bsum / np.maximum(Bn, 1)) / se


def build_tests(D, feats, dims):
    """D = discovery frame → list of tests (name, feat(s), gran, bucket, edges) + boolean masks (bucket, scored)."""
    tests, Bm, Sm = [], [], []
    if dims == 1:
        for f in feats:
            for g in (["cat", "sign", "t3"] if f in CAT else ["sign", "t3", "q5"]):
                cd, ed = bucketize(D[f].values, g, None, f)
                if cd is None:
                    continue
                sc = ~np.isnan(cd)
                for b in np.unique(cd[sc]):
                    tests.append(dict(f1=f, g1=g, b1=b, e1=ed, f2=None)); Bm.append(sc & (cd == b)); Sm.append(sc)
    else:
        cds = {}
        for f in feats:
            cd, ed = bucketize(D[f].values, "t3", None, f); cds[f] = (cd, ed)
        for a in range(len(feats)):
            for b in range(a + 1, len(feats)):
                fa, fb = feats[a], feats[b]
                ca, ea = cds[fa]; cb, eb = cds[fb]
                sc = ~np.isnan(ca) & ~np.isnan(cb)
                for x in range(3):
                    for y in range(3):
                        tests.append(dict(f1=fa, g1="t3", b1=float(x), e1=ea, f2=fb, g2="t3", b2=float(y), e2=eb))
                        Bm.append(sc & (ca == x) & (cb == y)); Sm.append(sc)
    return tests, np.array(Bm), np.array(Sm)


def apply_test(T, X):
    """bucket + scored masks of a frozen test on any frame."""
    ca, _ = bucketize(X[T["f1"]].values, T["g1"], T["e1"], T["f1"])
    m = ca == T["b1"]; sc = ~np.isnan(ca)
    if T.get("f2"):
        cb, _ = bucketize(X[T["f2"]].values, T["g2"], T["e2"], T["f2"])
        m = m & (cb == T["b2"]); sc = sc & ~np.isnan(cb)
    return m & sc, sc


def tname(T):
    def one(f, g, b, e):
        if f in CAT and g == "cat":
            return f"{f}={int(b)}"
        if e is None:
            return f"{f}[{g}={int(b)}]"
        lo = "-inf" if b == 0 else f"{e[int(b) - 1]:.3g}"; hi = "+inf" if b >= len(e) else f"{e[int(b)]:.3g}"
        return f"{f} in ({lo}, {hi}] [{g}]"
    s = one(T["f1"], T["g1"], T["b1"], T["e1"])
    if T.get("f2"):
        s += " & " + one(T["f2"], T["g2"], T["b2"], T["e2"])
    return s


def screen(D, feats, dims, rng, min_n=15, min_days=8):
    tests, B, S = build_tests(D, feats, dims)
    w = (D[f"net|{MAIN}"].values > 0).astype(np.float32)
    days = D.day.values
    Bn = B.sum(1); Sn = S.sum(1)
    bd = np.array([len(set(days[m])) for m in B])
    ok = (Bn >= min_n) & (Sn - Bn >= min_n) & (bd >= min_days)
    tests = [t for t, o in zip(tests, ok) if o]; B = B[ok].astype(np.float32); S = S[ok].astype(np.float32); Bn = Bn[ok]; Sn = Sn[ok]
    z = zstat(B @ w, Bn, S @ w, Sn)
    out = {"z": z, "tests": tests, "Bn": Bn}
    for nm, perm in (("day", day_perm_index(days, rng, NPERM)), ("trade", np.array([rng.permutation(len(w)) for _ in range(NPERM)]))):
        W = w[perm].T.astype(np.float32)                    # trades × perms
        mx = np.empty(NPERM)
        for a in range(0, NPERM, 100):
            Wc = W[:, a:a + 100]
            zz = zstat(B @ Wc, Bn[:, None], S @ Wc, Sn[:, None])
            mx[a:a + 100] = zz.max(0)
        out[f"fw_{nm}"] = np.array([(mx >= zi).mean() for zi in z])
        out[f"null_max_{nm}"] = mx
    return out


def boot_day(day, x, B=4000, rng=np.random.default_rng(5)):
    d = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"])
    if len(d) < 3:
        return np.nan, np.nan
    s, cn = d["sum"].values, d["count"].values; i = rng.integers(0, len(d), (B, len(d)))
    m = s[i].sum(1) / cn[i].sum(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def boot_diff_p(day, x, blk, B=4000, rng=np.random.default_rng(6)):
    """one-sided day-bootstrap P(mean(x | ~blk) − mean(x | blk) ≤ 0)."""
    d = pd.DataFrame({"d": day, "x": x, "b": blk.astype(int)})
    g = d.groupby(["d", "b"]).x.agg(["sum", "count"]).unstack(fill_value=0)
    if len(g) < 3:
        return np.nan
    s1, c1 = g["sum"][1].values if 1 in g["sum"] else np.zeros(len(g)), g["count"][1].values if 1 in g["count"] else np.zeros(len(g))
    s0, c0 = g["sum"][0].values if 0 in g["sum"] else np.zeros(len(g)), g["count"][0].values if 0 in g["count"] else np.zeros(len(g))
    i = rng.integers(0, len(g), (B, len(g)))
    with np.errstate(invalid="ignore", divide="ignore"):
        diff = s0[i].sum(1) / c0[i].sum(1) - s1[i].sum(1) / c1[i].sum(1)
    return np.nanmean(diff <= 0)


def maxdd(X, col):
    if not len(X):
        return 0.0
    o = np.argsort(X[f"xt|{col}"].values)
    eq = np.cumsum(X[f"net|{col}"].values[o] / 100 * USD)
    return float((eq - np.maximum.accumulate(np.r_[0, eq])[1:]).min()) if len(eq) else 0.0


def blk_stats(X, col=MAIN):
    x = X[f"net|{col}"].values
    if not len(x):
        return dict(N=0)
    lo, hi = boot_day(X.day.values, x)
    return dict(N=len(x), days=X.day.nunique(), WR=(x > 0).mean() * 100, avg=x.mean(), lo=lo, hi=hi, usd=x.sum() / 100 * USD,
                worst=x.min(), dd=maxdd(X, col))


def fmt(s):
    if not s.get("N"):
        return "0 | | | | | |"
    return (f"{s['N']} | {s['WR']:.0f} % | {s['avg']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}] | ${s['usd']:+,.0f} | {s['worst']:+.1f} | "
            f"${s['dd']:+,.0f}")


def stage_screen():
    buf = io.StringIO()

    def P(*a):
        print(*a); print(*a, file=buf)
    Z = pd.read_pickle(FEAT_PKL); M = Z["M"]; NUL = Z["NULL"]
    M = periods(M)
    # real-episode flag (①②): episodes_compact.spike ↔ live_master.spike strings (same convention as study_flatbase_short_exits.py)
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    spk = dict(zip(zip(E.pair, E.on_t), E.spike.astype(str).str[:16]))
    LMr = pd.read_csv(C.OUTP + "_live_master.csv", usecols=["pair", "spike", "real"])
    rmap = dict(zip(zip(LMr.pair, LMr.spike.astype(str).str[:16]), LMr.real))
    M["real_ep"] = [bool(rmap.get((p, spk.get((p, k), "")), False)) for p, k in zip(M.pair, M.ep_on_t)]
    flat = M.FLAT14 <= THR
    BLK = {"③ 2025 ⚠": M.per == "2025", "③ 2026-H1": M.per == "2026H1", "③ 2026-H2": M.per == "2026H2",
           "② master (real eps)": (M.per == "2_master") & M.real_ep, "① live (real eps, open = MTM)": (M.per == "1_live") & M.real_ep}
    # ── reproduction check vs the EXITS study
    ref = pd.read_csv(C.OUTP + "_EXITS_trades.csv")
    for col in ("tp8|72h|stop15", "tp8|72h|stop10", "tp8|72h|stop25"):
        r = ref[ref.entry.eq("S4") & ref[f"net|S4|{col}"].notna()][["pair", "dt", f"net|S4|{col}"]].copy()
        r["dt"] = pd.to_datetime(r.dt)
        mine = M[flat & M[f"tk|{col}"]][["pair", "dt", f"net|{col}"]]
        j = r.merge(mine, on=["pair", "dt"], how="outer", indicator=True)
        both = j[j._merge == "both"]
        P(f"REPRO {col}: ref {len(r)} · mine {len(mine)} · matched {len(both)} · max |Δ| {np.abs(both[f'net|S4|{col}'] - both[f'net|{col}']).max():.4f}")
    C0 = M[flat & M[f"tk|{MAIN}"]].copy()
    C0["W"] = C0[f"net|{MAIN}"] > 0
    # ── coverage
    P("\n## Coverage per feature (share of cohort trades scored) by block")
    cov = pd.DataFrame({b: C0[m[C0.index]][COIN + MKT].notna().mean() * 100 for b, m in BLK.items()}).round(0)
    P(cov.to_string())
    cov.to_csv(OUT + "_coverage.csv")
    # ── W vs L descriptive (medians), all blocks pooled + per period
    P("\n## Winners vs losers: median feature value (all periods; W = net > 0)")
    desc = []
    for f in COIN + MKT + ["btc_hold_ret", "fund_paid_hold"]:
        x = C0[f]
        desc.append(dict(feature=f, W_med=x[C0.W].median(), L_med=x[~C0.W].median(),
                         **{f"{p}_W-L": x[C0.W & (C0.per == p)].median() - x[~C0.W & (C0.per == p)].median() for p in ("2025", "2026H1", "2026H2")},
                         spearman_pnl=x.rank().corr(C0[f"net|{MAIN}"].rank())))
    DS = pd.DataFrame(desc); P(DS.round(3).to_string(index=False)); DS.to_csv(OUT + "_wl_medians.csv", index=False)
    rng = np.random.default_rng(42)
    allres = []
    for var, disc_pers in (("A", ["2025", "2026H1"]), ("B", ["2026H1"])):
        D = C0[C0.per.isin(disc_pers)].reset_index(drop=True)
        conf = C0[C0.per.isin(["2026H2"]) | (C0.per.isin(["2_master", "1_live"]) & C0.real_ep)].reset_index(drop=True)
        P(f"\n# Variant {var}: discover on {disc_pers} (N {len(D)}, WR {(D.W.mean() * 100):.1f} %) → confirm on 2026-H2 + ② + ① (N {len(conf)})")
        for fam, feats, dims in (("1D coin", COIN, 1), ("1D market", MKT, 1), ("2D all", COIN + MKT, 2)):
            R = screen(D, feats, dims, rng)
            z = R["z"]; order = np.argsort(-z)
            P(f"\n## {var} · {fam}: {len(z)} tests · null max-z 95th pct: day-preserving {np.percentile(R['null_max_day'], 95):.2f} · "
              f"trade-shuffle {np.percentile(R['null_max_trade'], 95):.2f} · best observed z {z.max():.2f}")
            P("| rank | blocked bucket | disc N blk | disc WR blk → rest | disc avg blk / rest | z | FW p (day) | FW p (trade) | neighbours agree | "
              "conf N blk | conf WR blk → rest | conf avg blk / rest | conf p | 26-H2 WR blk → rest |")
            P("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for rk, k in enumerate(order[:10]):
                T = R["tests"][k]
                bm, sc = apply_test(T, D)
                nb = [((D.loc[bm, f"net|{c}"] > 0).mean() < (D.loc[sc & ~bm, f"net|{c}"] > 0).mean()) for c in NEIGH]
                cm, cs_ = apply_test(T, conf)
                h2 = (conf.per == "2026H2").values
                cw = lambda m: (conf.loc[m, "W"].mean() * 100) if m.sum() else np.nan
                ca = lambda m: conf.loc[m, f"net|{MAIN}"].mean() if m.sum() else np.nan
                cp = boot_diff_p(conf.day.values[cs_], conf.W.values[cs_].astype(float), cm[cs_]) if cm.sum() >= 3 else np.nan
                row = dict(variant=var, family=fam, rank=rk + 1, test=tname(T), disc_N=int(bm.sum()), disc_WR_blk=D.W[bm].mean() * 100,
                           disc_WR_rest=D.W[sc & ~bm].mean() * 100, disc_avg_blk=D.loc[bm, f"net|{MAIN}"].mean(),
                           disc_avg_rest=D.loc[sc & ~bm, f"net|{MAIN}"].mean(), z=z[k], fw_day=R["fw_day"][k], fw_trade=R["fw_trade"][k],
                           neigh=sum(nb), conf_N=int(cm.sum()), conf_WR_blk=cw(cm), conf_WR_rest=cw(cs_ & ~cm), conf_avg_blk=ca(cm),
                           conf_avg_rest=ca(cs_ & ~cm), conf_p=cp, h2_WR_blk=cw(cm & h2), h2_WR_rest=cw(cs_ & ~cm & h2),
                           conf_cov=cs_.mean() * 100)
                allres.append(row)
                P(f"| {rk + 1} | {row['test']} | {row['disc_N']} | {row['disc_WR_blk']:.0f} → {row['disc_WR_rest']:.0f} | {row['disc_avg_blk']:+.2f} / "
                  f"{row['disc_avg_rest']:+.2f} | {row['z']:.2f} | {row['fw_day']:.3f} | {row['fw_trade']:.3f} | {row['neigh']}/4 | {row['conf_N']} | "
                  f"{row['conf_WR_blk']:.0f} → {row['conf_WR_rest']:.0f} | {row['conf_avg_blk']:+.2f} / {row['conf_avg_rest']:+.2f} | {row['conf_p']:.3f} | "
                  f"{row['h2_WR_blk']:.0f} → {row['h2_WR_rest']:.0f} |")
            P(f"survivors (FW p day < 0.05): {(R['fw_day'] < 0.05).sum()} · (trade-shuffle FW p < 0.05): {(R['fw_trade'] < 0.05).sum()}")
            # DAY-unit screen for market variables
        if True:
            Dd = D.groupby("day").agg(**{f: (f, "mean") for f in MKT}, Wd=("W", "mean"), pnl=(f"net|{MAIN}", "mean"), n=("W", "size")).reset_index()
            tests, B, S = build_tests(Dd, MKT, 1)
            w = Dd.Wd.values.astype(np.float32)
            Bn = B.sum(1); Sn = S.sum(1); okk = (Bn >= 8) & (Sn - Bn >= 8)
            tests = [t for t, o in zip(tests, okk) if o]; B = B[okk].astype(np.float32); S = S[okk].astype(np.float32); Bn = Bn[okk]; Sn = Sn[okk]

            def zd(Wm):
                bm = (B @ Wm) / Bn[:, None] if Wm.ndim == 2 else (B @ Wm) / Bn
                rm = ((S @ Wm) - (B @ Wm)) / ((Sn - Bn)[:, None] if Wm.ndim == 2 else (Sn - Bn))
                sd = np.std(Wm, axis=0) if Wm.ndim == 2 else np.std(Wm)
                return (rm - bm) / (sd * np.sqrt(1 / Bn + 1 / (Sn - Bn)) if Wm.ndim == 1 else sd * np.sqrt(1 / Bn[:, None] + 1 / (Sn - Bn)[:, None]))
            z = zd(w)
            Wp = np.stack([w[rng.permutation(len(w))] for _ in range(NPERM)], 1)
            mx = zd(Wp).max(0)
            fw = np.array([(mx >= zi).mean() for zi in z])
            o = np.argsort(-z)[:8]
            P(f"\n## {var} · market variables in DAY units ({len(Dd)} days, {len(z)} tests, day-shuffle null max-z 95th {np.percentile(mx, 95):.2f})")
            P("| blocked day-bucket | days | day-WR blk → rest | z | FW p |")
            P("|---|---|---|---|---|")
            for k in o:
                bm = B[k] > 0; sc = S[k] > 0
                P(f"| {tname(tests[k])} | {int(bm.sum())} | {Dd.Wd[bm].mean() * 100:.0f} → {Dd.Wd[sc & ~bm].mean() * 100:.0f} | {z[k]:.2f} | {fw[k]:.3f} |")
                allres.append(dict(variant=var, family="market DAY units", test=tname(tests[k]), disc_N=int(bm.sum()), z=z[k], fw_day=fw[k]))
    RES = pd.DataFrame(allres); RES.to_csv(OUT + "_screen.csv", index=False)
    # ── tighter stops
    P("\n# Stop table: S4 · TP −8 · 72 h (flat cohort), per block")
    rows = []
    NULL = {}
    for nn in NUL:
        NULL[(nn["pair"], nn["on_t"])] = nn
    for nm in EXN:
        X = M[flat & M[f"tk|{nm}"]]
        P(f"\n## {nm}")
        P("| block | N | WR | avg % [95 % day CI] | $ at $190 | worst % | max DD $ | stop-outs | of those, later hit −8 % TP | fair random entry avg · p |")
        P("|---|---|---|---|---|---|---|---|---|---|")
        j = EXN.index(nm)
        for b, m in list(BLK.items()) + [("③ 2026 (H1+H2)", M.per.isin(["2026H1", "2026H2"]))]:
            Xb = X[m[X.index]]
            s = blk_stats(Xb, nm)
            st = (Xb[f"rs|{nm}"] == 1); late = Xb[f"late|{nm}"]
            # null over the same episodes
            keys = set(zip(Xb.pair, Xb.ep_on_t))
            if keys and not b.startswith(("①", "②")):
                arr = np.stack([NULL[k]["net"][:, j] for k in keys if k in NULL])
                nmn = np.nanmean(arr, 0); pnull = (nmn >= s.get("avg", np.nan)).mean(); nul = f"{np.nanmean(nmn):+.2f} · {pnull:.2f}"
            else:
                nul = "–"
            P(f"| {b} | {fmt(s)} | {st.sum()} | {late.sum()} ({late.sum() / max(st.sum(), 1) * 100:.0f} %) | {nul} |")
            rows.append(dict(exit=nm, block=b, **s, stops=int(st.sum()), stopped_later_tp=int(late.sum()), null=nul))
    pd.DataFrame(rows).to_csv(OUT + "_stops.csv", index=False)
    # ── add-trades question (looser flatness), main cell
    P("\n# ADD-trades question (NOT a recommendation): S4 · TP −8 · 72 h · stop 15 on looser flatness bands")
    P("| FLAT14 band | block | N | WR | avg % [CI] | $ | worst | DD |")
    P("|---|---|---|---|---|---|---|---|")
    for lab, mm in (("≤ 44.4 (current)", flat), ("44.4–55", (M.FLAT14 > THR) & (M.FLAT14 <= 55)), ("55–70", (M.FLAT14 > 55) & (M.FLAT14 <= 70))):
        X = M[mm & M[f"tk|{MAIN}"]]
        for b, m in BLK.items():
            P(f"| {lab} | {b} | {fmt(blk_stats(X[m[X.index]]))} |")
    # ── trades / features csv
    C0.drop(columns=[c for c in C0.columns if c.startswith(("xt|", "nof|"))]).to_csv(OUT + "_trades.csv", index=False)
    open(OUT + "_tables.md", "w").write(buf.getvalue())
    print("done")


def load_cohort():
    Z = pd.read_pickle(FEAT_PKL); M = periods(Z["M"])
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    spk = dict(zip(zip(E.pair, E.on_t), E.spike.astype(str).str[:16]))
    LMr = pd.read_csv(C.OUTP + "_live_master.csv", usecols=["pair", "spike", "real"])
    rmap = dict(zip(zip(LMr.pair, LMr.spike.astype(str).str[:16]), LMr.real))
    M["real_ep"] = [bool(rmap.get((p, spk.get((p, k), "")), False)) for p, k in zip(M.pair, M.ep_on_t)]
    return M, Z["NULL"]


def stage_candidate():
    """before → after tables, expectancy bar, concentration, confounds and proxy calibration for the screen's candidates."""
    buf = io.StringIO()

    def P(*a):
        print(*a); print(*a, file=buf)
    M, NUL = load_cohort()
    NULL = {(nn["pair"], nn["on_t"]): nn for nn in NUL}
    flat = M.FLAT14 <= THR
    BLK = [("① live (real eps, open = MTM)", (M.per == "1_live") & M.real_ep), ("② master (real eps)", (M.per == "2_master") & M.real_ep),
           ("③ 2025 ⚠", M.per == "2025"), ("③ 2026-H1", M.per == "2026H1"), ("③ 2026-H2", M.per == "2026H2"),
           ("③ 2026 (H1+H2)", M.per.isin(["2026H1", "2026H2"]))]
    CANDS = {"KEEP turnover > 1 (block vol24 ≤ mcap)": lambda X: X.turnover > 1,
             "KEEP turnover > 0.366 (block the low tercile)": lambda X: X.turnover > 0.366,
             "KEEP peak turnover > 1": lambda X: X.turnover_peak > 1}
    # ── proxy calibration vs the bot's real stamps
    MP = pd.read_csv(os.path.join(C.R, "reports", "MASTER_POOL_stacked.csv"), usecols=["pair", "opened_at", "entry_price", "entry_mcap_usd",
                                                                                         "entry_vol_mcap_ratio"], low_memory=False)
    LV = pd.read_csv(os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-21_02-14-46.csv"),
                     usecols=["pair", "opened_at", "entry_price", "entry_mcap_usd", "entry_vol_mcap_ratio"], low_memory=False)
    ST = pd.concat([MP, LV]).dropna(subset=["entry_mcap_usd"])
    ST = ST[ST.entry_mcap_usd > 0].copy()
    ST["ms"] = pd.to_datetime(ST.opened_at, format="mixed", utc=True).astype("int64") // 10**6
    sup = json.load(open(SUPPLY))
    cal = []
    for r in ST.drop_duplicates(["pair", "opened_at"]).itertuples():
        s = sup.get(r.pair)
        if not s or not s.get("cs"):
            continue
        X = M[(M.pair == r.pair)]
        if not len(X):
            continue
        unit = s["mc"] / s["cs"]
        mult = s.get("mult") or 10 ** round(math.log10(r.entry_price / unit))
        cal.append(dict(pair=r.pair, opened=r.opened_at, stamped=r.entry_mcap_usd, proxy=s["cs"] * r.entry_price / mult))
    CA = pd.DataFrame(cal)
    if len(CA):
        rr = CA.proxy / CA.stamped
        P(f"PROXY CALIBRATION (cohort pairs with a real stamped fill, Sep-28 →): {len(CA)} fills · {CA.pair.nunique()} pairs · proxy ÷ stamped "
          f"median {rr.median():.3f} · IQR {rr.quantile(.25):.3f}–{rr.quantile(.75):.3f} · within ±10 % {((rr - 1).abs() <= .1).mean() * 100:.0f} % · "
          f"same side of turnover = 1 when vol is fixed: {(np.sign(np.log(rr)) == 0).mean() * 100 if False else ((rr - 1).abs() <= .2).mean() * 100:.0f} % within ±20 %")
        CA.to_csv(OUT + "_mcap_calibration.csv", index=False)
    for col in [MAIN]:
        X0 = M[flat & M[f"tk|{col}"]].copy()
        P(f"\n# Cohort {col}: scored for turnover {X0.turnover.notna().mean() * 100:.0f} %")
        P("| block | unscored trades: N · WR · avg % |")
        P("|---|---|")
        for b, m in BLK:
            U = X0[m[X0.index] & X0.turnover.isna()]
            P(f"| {b} | {len(U)} · {(U[f'net|{col}'] > 0).mean() * 100 if len(U) else float('nan'):.0f} % · {U[f'net|{col}'].mean() if len(U) else float('nan'):+.2f} |")
    for cn, fn in CANDS.items():
        P(f"\n# CANDIDATE: {cn}")
        for col in EXN:
            X0 = M[flat & M[f"tk|{col}"]].copy()
            sc = X0.turnover.notna()
            keep = fn(X0) & sc
            P(f"\n## exit {col} — before = all flat S4 trades (unscored included) · after = scored AND kept (unscored blocked, fail-closed)")
            P("| block | before: N · WR · avg % [CI] · $ · worst · DD | after: N · WR · avg % [CI] · $ · worst · DD | blocked (scored): N · WR · avg % | "
              "WR lift | after vs fair random (kept episodes) · p |")
            P("|---|---|---|---|---|---|")
            for b, m in BLK:
                mb = m[X0.index]
                s0 = blk_stats(X0[mb], col); s1 = blk_stats(X0[mb & keep], col); bl = X0[mb & sc & ~keep]
                keys = set(zip(X0[mb & keep].pair, X0[mb & keep].ep_on_t))
                nul = "–"
                if keys and not b.startswith(("①", "②")):
                    j = EXN.index(col)
                    arr = np.stack([NULL[k]["net"][:, j] for k in keys if k in NULL])
                    nmn = np.nanmean(arr, 0); nul = f"{np.nanmean(nmn):+.2f} · {(nmn >= s1['avg']).mean():.2f}"
                if keys and not b.startswith(("①", "②")) and cn.startswith("KEEP turnover > 1"):
                    k0 = [k for k in set(zip(X0[mb].pair, X0[mb].ep_on_t)) if k in NULL]
                    arr = np.stack([NULL[k]["net"][:, j] for k in k0]); tv = np.stack([NULL[k]["tov"] for k in k0])
                    arr = np.where(tv > 1, arr, np.nan)
                    nmf = np.nanmean(arr, 0); nul += f" · same filter on the random bar: {np.nanmean(nmf):+.2f} · p {(nmf >= s1['avg']).mean():.2f}"
                P(f"| {b} | {fmt(s0).replace(' | ', ' · ')} | {fmt(s1).replace(' | ', ' · ')} | {len(bl)} · {(bl[f'net|{col}'] > 0).mean() * 100 if len(bl) else float('nan'):.0f} % · "
                  f"{bl[f'net|{col}'].mean() if len(bl) else float('nan'):+.2f} | {s1.get('WR', np.nan) - s0.get('WR', np.nan):+.1f} pp | {nul} |")
        # expectancy bar on the BLOCKED cohort (main cell), discovery A and all periods
        for lab, pm in (("discovery A (2025 + 26-H1)", M.per.isin(["2025", "2026H1"])),
                        ("confirmation (26-H2 + ② + ①)", (M.per == "2026H2") | (M.per.isin(["2_master", "1_live"]) & M.real_ep)),
                        ("all periods", M.per.isin(["2025", "2026H1", "2026H2"]) | (M.per.isin(["2_master", "1_live"]) & M.real_ep))):
            X0 = M[flat & M[f"tk|{MAIN}"] & pm]
            sc = X0.turnover.notna(); kp = X0[sc & fn(X0)]; bl = X0[sc & ~fn(X0)]
            x = kp[f"net|{MAIN}"]; aw = x[x > 0].mean(); al = -x[x <= 0].mean(); be = al / (aw + al) * 100
            y = bl[f"net|{MAIN}"].values
            # window (day)-clustered bootstrap P(avg < 0)
            d = pd.DataFrame({"d": bl.day.values, "x": y}).groupby("d").x.agg(["sum", "count"])
            rng = np.random.default_rng(3); ii = rng.integers(0, len(d), (4000, len(d)))
            bm = d["sum"].values[ii].sum(1) / d["count"].values[ii].sum(1)
            neg = -np.minimum(y, 0); lossp = pd.Series(neg).groupby(bl.pair.values).sum(); lossd = pd.Series(neg).groupby(bl.day.values).sum()
            dlt = x.mean() - X0[f"net|{MAIN}"].mean()
            P(f"\nEXPECTANCY BAR ({lab}) on the BLOCKED cohort: N {len(y)} · windows {bl.day.nunique()} · WR {(y > 0).mean() * 100:.1f} % vs breakeven "
              f"{be:.1f} % (kept avg win {aw:+.2f} / avg loss {-al:+.2f}) · avg {y.mean():+.2f} · P(avg < 0) {(bm < 0).mean():.3f} · top pair share of loss "
              f"{lossp.max() / neg.sum() * 100:.0f} % ({lossp.idxmax()}) · top window share {lossd.max() / neg.sum() * 100:.0f} % → "
              f"{'PASS' if ((y > 0).mean() * 100 < be and (bm < 0).mean() >= .95 and bl.day.nunique() >= 8 and lossp.max() / neg.sum() < .5 and lossd.max() / neg.sum() < .5 and len(y) >= 15) else 'FAIL'}"
              f" · kept-vs-all Δ avg {dlt:+.2f} → after 40 % haircut {dlt * 0.6:+.2f}")
        # concentration of the KEPT cohort's profit
        X0 = M[flat & M[f"tk|{MAIN}"] & M.per.isin(["2025", "2026H1", "2026H2"])]
        kp = X0[X0.turnover.notna() & fn(X0)]
        pp = kp.groupby("pair")[f"net|{MAIN}"].sum(); dd = kp.groupby("day")[f"net|{MAIN}"].sum()
        P(f"kept cohort (③): {len(kp)} trades · {kp.pair.nunique()} pairs · top pair share of profit {pp.max() / kp[f'net|{MAIN}'].sum() * 100:.0f} % ({pp.idxmax()}) · "
          f"top day share {dd.max() / kp[f'net|{MAIN}'].sum() * 100:.0f} % · best 5 % removed avg {np.sort(kp[f'net|{MAIN}'].values)[:-int(np.ceil(len(kp) * .05))].mean():+.2f}")
        # by period sign
        P("per half-year (kept WR / avg vs blocked WR / avg): " + " · ".join(
            f"{f}: {(X0[(X0.per == f) & X0.turnover.notna() & fn(X0)][f'net|{MAIN}'] > 0).mean() * 100:.0f} % {X0[(X0.per == f) & X0.turnover.notna() & fn(X0)][f'net|{MAIN}'].mean():+.2f} vs "
            f"{(X0[(X0.per == f) & X0.turnover.notna() & ~fn(X0)][f'net|{MAIN}'] > 0).mean() * 100:.0f} % {X0[(X0.per == f) & X0.turnover.notna() & ~fn(X0)][f'net|{MAIN}'].mean():+.2f}"
            for f in ("2025", "2026H1", "2026H2")))
    # ── confounds: does turnover separate within mcap / pump-size / volume-surge terciles?
    X0 = M[flat & M[f"tk|{MAIN}"] & M.turnover.notna()].copy()
    X0["hi_to"] = X0.turnover > 1
    P("\n# Confound check (all periods ③ + ② + ①, main cell): WR / avg for turnover > 1 vs ≤ 1 inside terciles of a competing variable")
    P("| variable | tercile | N >1 · WR · avg | N ≤1 · WR · avg |")
    P("|---|---|---|---|")
    for v in ("mcap_musd", "pump_pct", "vol_vs_prepump", "vol24_musd", "coin_r24h", "FLAT14"):
        q = pd.qcut(X0[v], 3, labels=False, duplicates="drop")
        for k in sorted(q.dropna().unique()):
            a = X0[(q == k) & X0.hi_to]; b_ = X0[(q == k) & ~X0.hi_to]
            P(f"| {v} | {int(k)} ({X0[v][q == k].min():.3g}–{X0[v][q == k].max():.3g}) | {len(a)} · {(a[f'net|{MAIN}'] > 0).mean() * 100:.0f} % · "
              f"{a[f'net|{MAIN}'].mean():+.2f} | {len(b_)} · {(b_[f'net|{MAIN}'] > 0).mean() * 100:.0f} % · {b_[f'net|{MAIN}'].mean():+.2f} |")
    # mechanism: hold path by turnover side
    P("\n# Mechanism (descriptive, look-ahead): exit reasons and hold-period BTC / funding by turnover side (main cell, all periods)")
    for side, m in (("> 1", X0.hi_to), ("≤ 1", ~X0.hi_to)):
        Y = X0[m]
        rs = Y[f"rs|{MAIN}"].map({0: "time", 1: "stop", 3: "tp", 5: "settle"}).value_counts(normalize=True) * 100
        P(f"turnover {side}: N {len(Y)} · TP {rs.get('tp', 0):.0f} % · stop {rs.get('stop', 0):.0f} % · time {rs.get('time', 0):.0f} % · "
          f"median BTC over hold {Y.btc_hold_ret.median():+.2f} % · funding paid {Y.fund_paid_hold.mean():+.3f} % · median pump {Y.pump_pct.median():.1f} % · "
          f"median mcap ${Y.mcap_musd.median():.0f}M · median 24 h vol ${Y.vol24_musd.median():.0f}M")
    open(OUT + "_candidate.md", "w").write(buf.getvalue())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("stage"); ap.add_argument("--procs", type=int, default=7); A = ap.parse_args()
    if A.stage == "fetch":
        stage_fetch()
    elif A.stage == "build":
        stage_build(A.procs)
    elif A.stage == "screen":
        stage_screen()
    elif A.stage == "candidate":
        stage_candidate()
