#!/usr/bin/env python3
"""LITE THEORY — step 3: pricing (pre-registration frozen first: reports/LITE_THEORY_PREREG.txt).

  R1  every universe bar (non-bearish, 2–24 h, not in state, vol < 100) on 1m bars → reports/lite_theory/y1m.pkl
  RH  every signal of every emulator cell on the hybrid tick path (FX.load_path) → yrh.pkl ; cell membership → cells.pkl
  A6 / A7 on today's signals (RH + R1) and on the master fills → a67.pkl / a67_master.pkl
Walker parity FIRST: X.eval_v (the LITE_PROTECT walker, RM ruler) on every replay fill with its own E / tick must equal the frozen
rm_base_pct; this file's own walkP must equal X.eval_v on every priced RH signal (asserted).
"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_lite_theory_common as T  # noqa: E402
import study_lite_theory_signals as S  # noqa: E402
import study_lite_protect_common as X  # noqa: E402  (imports C → 2025 loader patches)

C, FX, WT, W = X.C, X.FX, X.WT, X.W
WT._NET_BLOCKED = True
MIN = T.MIN
TAKER, MAKER = C.TAKER, C.MAKER
SLIP_T1 = 0.036            # % taker entry slip over the print
LS = -0.20                 # non-TP exit slip (stop / cap / VWAP exit)
CAP = 720
JOBS = int(os.environ.get("JOBS", "8"))

if not os.path.exists(T.PREREG):
    sys.exit("REFUSED: pre-registration missing")


# ───────────────────────── 1m data ─────────────────────────
_A = {}


def k1(pair):
    if pair not in _A:
        k = WT.load_1m(pair)
        _A[pair] = (k.open_time.values.astype("int64"), k.o.values.astype(float), k.h.values.astype(float), k.l.values.astype(float),
                    k.c.values.astype(float))
        if len(_A) > 6:
            _A.pop(next(iter(_A)))
    return _A[pair]


def entry(pair, te):
    """→ (s, E_raw) : the first 1m bar opening at ceil(te) and its open (prev close when the cache has no open), or (None, nan)."""
    ot, o, h, l, c = k1(pair)
    first = -(-int(te) // MIN) * MIN
    s = int(np.searchsorted(ot, first))
    if s >= len(ot) or ot[s] != first:
        return None, np.nan
    e = o[s]
    if not np.isfinite(e):
        e = c[s - 1] if s > 0 and ot[s - 1] == first - MIN else np.nan
    return s, e


def path1m(pair, s, te, E, horizon=CAP):
    """P dict (the FX / live_path format) from 1m bars starting at index s, contiguous, up to te + (horizon + 1) min."""
    ot, o, h, l, c = k1(pair)
    lim = te + (horizon + 1) * MIN
    e = int(np.searchsorted(ot, lim - MIN, side="right"))
    sl = slice(s, max(s, e))
    oo = ot[sl]
    if len(oo):
        g = np.flatnonzero(np.diff(oo) != MIN)
        if len(g):
            sl = slice(s, s + g[0] + 1); oo = ot[sl]
    return dict(t=oo + MIN, hi=WT.net_pct(h[sl], E), lo=WT.net_pct(l[sl], E), cl=WT.net_pct(c[sl], E), op=WT.net_pct(o[sl], E),
                bar=np.ones(len(oo), bool), end_ms=int(oo[-1] + MIN) if len(oo) else int(te), n_tick=0, n_bar=len(oo), tick_end=int(te),
                lim=int(lim), te=int(te))


def tick_of(pair, s, E):
    ot, o, h, l, c = k1(pair)
    seg = np.concatenate([h[s:s + 720], l[s:s + 720], c[s:s + 720]])
    seg = seg[np.isfinite(seg)]
    tk, _ = C.infer_tick(pair, seg) if len(seg) else (C.ex_tick(pair), "x")
    return float(tk) if tk is not None and np.isfinite(tk) else float(E) * 1e-4


# ───────────────────────── generic walker on a P path ─────────────────────────
def walkP(P, E, tick, sl=3.0, tp=3.0, t_x=None, x_net=None, cap=CAP):
    """→ (why, raw_pct, ls_pct, xm). TP = resting maker (W._rm_levels) · stop −sl (tick: the print; bar: min(line, open); tie → stop) ·
    optional market exit decided at t_x booked x_net (net %, 'VX') if neither fired at an event time ≤ t_x · cap at the last close."""
    te = P["te"]
    n = int(np.searchsorted(P["t"], te + cap * MIN, side="right"))
    if n == 0:
        return "UNRES", np.nan, np.nan, np.nan
    _, th, bk = W._rm_levels(E, tick, tp)
    hi, lo, t = P["hi"][:n], P["lo"][:n], P["t"][:n]
    ist = np.flatnonzero(lo <= -sl); itp = np.flatnonzero(hi >= th - 1e-12)
    b = int(ist[0]) if len(ist) else n
    a = int(itp[0]) if len(itp) else n
    k = min(a, b)
    if t_x is not None and t_x <= te + cap * MIN and (k >= n or t[k] > t_x):
        if P["end_ms"] < t_x and k >= n:
            return "UNRES", np.nan, np.nan, np.nan
        return "VX", float(x_net), float(x_net) + LS, (t_x - te) / MIN
    if b < n and b <= a:
        v = (min(-sl, float(P["op"][b])) if np.isfinite(P["op"][b]) else -sl) if P["bar"][b] else float(lo[b])
        return "STOP", v, v + LS, (t[b] - te) / MIN
    if a < n:
        return "TP", float(bk), float(bk), (t[a] - te) / MIN
    if P["end_ms"] < te + cap * MIN:
        return "UNRES", np.nan, np.nan, np.nan
    v = float(P["cl"][n - 1])
    return "CAP", v, v + LS, float(cap)


# ───────────────────────── 0. walker parity on the replay fills ─────────────────────────
def _par_job(rows):
    out = []
    tc = {}
    for r in rows:
        P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), CAP, tc)
        res = X.eval_v(P, None, "RM", float(r["E"]), float(r["tick"]))
        w = walkP(P, float(r["E"]), float(r["tick"]))
        out.append(dict(fid=r["fid"], ref=r["rm_base_pct"], ev=res["pct"], ev_why=res["why"], wp=w[1], wp_why=w[0]))
        if len(tc) > 200:
            tc.clear()
    return out


def walker_parity():
    K = pd.read_csv(T.COHORT, low_memory=False, usecols=["fid", "block", "pair", "te", "E", "tick", "rm_base_pct", "rm_base_why"])
    K = K[np.isfinite(K.tick) & K.rm_base_pct.notna()]
    items = K.to_dict("records")
    ch = [items[i::JOBS * 4] for i in range(JOBS * 4)]
    with Pool(JOBS) as Pp:
        R = pd.DataFrame(sum(Pp.map(_par_job, ch), []))
    d1 = (R.ev - R.ref).abs(); d2 = (R.wp - R.ev).abs()
    print(f"walker parity: {len(R)} cohort fills · X.eval_v vs frozen rm_base_pct max |Δ| {d1.max():.2e} · walkP vs eval_v max |Δ| {d2.max():.2e}")
    assert d1.max() < 1e-9 and d2.max() < 1e-9 and (R.ev_why == R.wp_why).all()
    return len(R)


# ───────────────────────── 1. R1 bulk ─────────────────────────
def _r1_job(args):
    pair, Ts = args
    out = []
    for Tb in Ts:
        te = int(Tb) + T.BAR_MS + T.T_LAG_MS
        s, e = entry(pair, te)
        if s is None or not np.isfinite(e):
            out.append(dict(pair=pair, T=Tb, why1="NOENTRY")); continue
        E = e * (1 + SLIP_T1 / 100)
        P = path1m(pair, s, te, E)
        tick = tick_of(pair, s, E)
        why, raw, ls, xm = walkP(P, E, tick)
        ot, o, h, l, c = k1(pair)
        f = {}
        for m in (60, 240, 720):
            j = s + m - 1
            f[f"f{m}"] = (c[j] / E - 1) * 100 if j < len(ot) and ot[j] == ot[s] + (m - 1) * MIN else np.nan
        out.append(dict(pair=pair, T=Tb, E1=E, tick1=tick, why1=why, y1=ls, xm1=xm, **f))
    return out


# ───────────────────────── 2. emulator cells ─────────────────────────
def build_cells(B):
    """→ DataFrame(cell, pair, T) of the ENTRIES of every emulator / filter cell (non-bearish), plus per-signal features of today."""
    U = B
    cells = []

    def add(name, df):
        df = df.merge(U[["pair", "T", "bear"]], on=["pair", "T"], how="left")
        df = df[df.code == "READY"]
        df = df[df.bear == False]  # noqa: E712
        cells.append(df[["pair", "T"]].assign(cell=name))

    base = S.signals(U, **S.TODAY)
    # judged-stretch ordinal within the episode (READY or COLOR = judged)
    base = base.merge(U[["pair", "T", "spike_ts"]], on=["pair", "T"], how="left").sort_values(["pair", "T"])
    base["ord"] = base.groupby(["pair", "spike_ts"]).cumcount() + 1
    add("TODAY", base)
    for nm, kw in (("A2_GREEN", dict(color="green")), ("A2_ANY", dict(color="any")), ("A3_N6", dict(need=6)), ("A3_N24", dict(need=24)),
                   ("A3_N36", dict(need=36)), ("A4_H2_8", dict(hmax=8.0)), ("A4_H2_12", dict(hmax=12.0)), ("A4_H4_1675", dict(hmin=4.0)),
                   ("A4_H8_1675", dict(hmin=8.0)), ("A4_H2_24", dict(hmax=24.0)), ("A5_V20", dict(vmax=20.0)), ("A5_V50", dict(vmax=50.0))):
        kk = dict(S.TODAY); kk.update(kw)
        add(nm, S.signals(U, **kk))
    # A2_WAITRED: today + (green first signal bar → next RED bar of the same stretch within 6 bars, signal conditions still met)
    g = base[base.code == "COLOR"]
    extra = []
    Ug = {p: G.sort_values("T").reset_index(drop=True) for p, G in U[U.pair.isin(g.pair.unique())].groupby("pair")}
    for r in g.itertuples():
        G = Ug[r.pair]
        i = int(np.searchsorted(G["T"].values, r.T))
        for j in range(i + 1, min(len(G), i + 7)):
            x = G.iloc[j]
            if x["T"] - r.T > 6 * T.BAR_MS:
                break
            sid = int(x["T"]) - (int(x.streak) - 1) * T.BAR_MS if x.streak >= 1 else None
            if sid != r.sid or x.in_state or not (x.vol_mult < 100) or x.hours > 16.75 or not (x.vol24 >= 20e6):
                break
            if x.bar_red:
                extra.append(dict(pair=r.pair, T=int(x["T"]), code="READY")); break
    add("A2_WAITRED", pd.concat([base[["pair", "T", "code"]], pd.DataFrame(extra, columns=["pair", "T", "code"])]))
    C_ = pd.concat(cells, ignore_index=True)
    return C_, base


# ───────────────────────── 3. RH pricing of signal bars ─────────────────────────
def _rh_job(rows):
    out = []
    tc = {}
    for r in rows:
        pair, Tb = r["pair"], int(r["T"])
        te = Tb + T.BAR_MS + T.T_LAG_MS
        s, e = entry(pair, te)
        if s is None or not np.isfinite(e):
            out.append(dict(pair=pair, T=Tb, whyh="NOENTRY")); continue
        E = e * (1 + SLIP_T1 / 100)
        P = FX.load_path(pair, te, E, CAP, tc)
        tick, _ = W.path_tick(P, pair, E)
        if not np.isfinite(tick):
            tick = tick_of(pair, s, E)
        why, raw, ls, xm = walkP(P, E, tick)
        ev = X.eval_v(P, None, "RM", E, tick)
        assert ev["why"] == why and (why == "UNRES" or abs(ev["pct"] - raw) < 1e-9), (pair, Tb, ev, why, raw)
        out.append(dict(pair=pair, T=Tb, Eh=E, tickh=tick, whyh=why, yh=ls, rawh=raw, xmh=xm, srch=FX.path_src(P)))
        if len(tc) > 150:
            tc.clear()
    return out


# ───────────────────────── 4. A6 limit / A7 VWAP exits ─────────────────────────
A6K, A6W = (0.5, 1.0, 1.5), (30, 60, 120)
A7B = (0.0, 0.5, 1.0)


def a67_one(pair, te, E, tick, vwap0, Tsig, vw5, P, ruler, tc, s_idx=None):
    """one signal → dict of cell results (LS %, NaN = unresolved). vw5 = DataFrame of this episode's 5m bars after the signal (T, c, vwap)."""
    out = {}
    base = walkP(P, E, tick)
    out["TODAY"] = base[2]
    ot, o, h, l, c = k1(pair)
    # ── A7 5m-close exits (complement) + replace
    for b in A7B:
        tx = None; xn = None
        if vw5 is not None and len(vw5):
            m = np.flatnonzero(vw5.c.values < vw5.vwap.values * (1 - b / 100))
            if len(m):
                tx = int(vw5["T"].values[m[0]]) + T.BAR_MS
                xn = float(WT.net_pct(vw5.c.values[m[0]], E))
        r = walkP(P, E, tick, t_x=tx, x_net=xn) if tx is not None else base
        out[f"A7_C5_b{b:g}"] = r[2]
        if b == 0.0:
            out["A7_R5"] = (walkP(P, E, tick, sl=10.0, t_x=tx, x_net=xn) if tx is not None else walkP(P, E, tick, sl=10.0))[2]
    # ── A7 1m-close exits vs the last closed 5m VWAP
    first = -(-int(te) // MIN) * MIN
    i0 = int(np.searchsorted(ot, first)); i1 = int(np.searchsorted(ot, te + CAP * MIN))
    if vw5 is not None and len(vw5) and i1 > i0:
        tclose = ot[i0:i1] + MIN
        cl1 = c[i0:i1]
        vT = vw5["T"].values + T.BAR_MS                # 5m close times
        vv = np.concatenate([[vwap0], vw5.vwap.values])
        vt = np.concatenate([[Tsig + T.BAR_MS], vT])
        idx = np.searchsorted(vt, tclose, side="right") - 1
        vref = vv[np.maximum(idx, 0)]
        lastok = vt[-1] + T.BAR_MS                    # beyond the episode's rebuilt bars → no VWAP read
        for b in A7B:
            m = np.flatnonzero((cl1 < vref * (1 - b / 100)) & (tclose <= lastok) & np.isfinite(cl1))
            if len(m):
                tx = int(tclose[m[0]]); xn = float(WT.net_pct(cl1[m[0]], E))
                out[f"A7_C1_b{b:g}"] = walkP(P, E, tick, t_x=tx, x_net=xn)[2]
            else:
                out[f"A7_C1_b{b:g}"] = base[2]
    else:
        for b in A7B:
            out[f"A7_C1_b{b:g}"] = base[2]
    # ── A7 stop at VWAP − b (replacing −3)
    for b in A7B:
        d = -float(WT.net_pct(vwap0 * (1 - b / 100), E))
        d = float(np.clip(d, 0.5, 10.0))
        out[f"A7_VS_b{b:g}"] = walkP(P, E, tick, sl=d)[2]
    # ── A6 limit entries
    e_raw = E / (1 + SLIP_T1 / 100)
    for k in A6K:
        Lp = np.floor(vwap0 * (1 + k / 100) / tick + 1e-9) * tick
        for w in A6W:
            nm = f"A6_L{k:g}_W{w}"
            if e_raw <= Lp:
                out[nm] = base[2]; continue
            if ruler == "RH":
                P0 = FX.load_path(pair, te, E, w + 1, tc)
            else:
                P0 = path1m(pair, s_idx, te, E, horizon=w + 1)
            lvl = float(WT.net_pct(Lp - 0.5 * tick, E))
            hit = np.flatnonzero((P0["lo"] <= lvl) & (P0["t"] <= te + w * MIN))
            if not len(hit):
                out[nm] = 0.0 if P0["end_ms"] >= te + w * MIN else np.nan
                continue
            tf = int(P0["t"][hit[0]])
            if ruler == "RH":
                P1 = FX.load_path(pair, tf, Lp, CAP, tc)
            else:
                s1 = int(np.searchsorted(ot, tf))
                P1 = path1m(pair, s1, tf, Lp) if s1 < len(ot) and ot[s1] == tf else None
            if P1 is None:
                out[nm] = np.nan; continue
            r = walkP(P1, Lp, tick)
            out[nm] = r[2] + (TAKER - MAKER) if np.isfinite(r[2]) else np.nan
    return out


def _a67_job(args):
    pair, rows, vwb = args
    out = []
    tc = {}
    for r in rows:
        Tb = int(r["T"]); te = int(r.get("te") or (Tb + T.BAR_MS + T.T_LAG_MS))
        vw5 = vwb[(vwb.spike_ts == r["spike_ts"]) & (vwb["T"] > Tb)] if vwb is not None and r.get("spike_ts") is not None else None
        for ruler in ("RH", "R1"):
            s, e = entry(pair, te)
            if r.get("E") is not None and np.isfinite(r.get("E", np.nan)):
                E = float(r["E"])                         # master: the fill's own price
            else:
                if s is None or not np.isfinite(e):
                    break
                E = e * (1 + SLIP_T1 / 100)
            if ruler == "RH":
                P = FX.load_path(pair, te, E, CAP, tc)
                tick, _ = W.path_tick(P, pair, E)
                if not np.isfinite(tick):
                    tick = tick_of(pair, s, E) if s is not None else E * 1e-4
            else:
                if s is None:
                    continue
                P = path1m(pair, s, te, E)
                tick = tick_of(pair, s, E)
            res = a67_one(pair, te, E, tick, float(r["vwap"]), Tb, vw5, P, ruler, tc, s_idx=s)
            out.append(dict(pair=pair, T=Tb, ruler=ruler, fid=r.get("fid"), **res))
        if len(tc) > 150:
            tc.clear()
    return out


def master_rows(B):
    import export_glob as EG
    K = pd.read_csv(T.COHORT, low_memory=False, usecols=["fid", "block", "era", "pair", "te", "E", "tick", "rm_base_pct", "rm_base_why"])
    M = K[K.block == "MASTER"].copy()
    fs = EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv"))
    cols = ["pair", "opened_at", "entry_frenzy_vwap", "entry_frenzy_vs_vwap_pct", "entry_frenzy_hours", "entry_frenzy_vol_mult",
            "entry_frenzy_run_pct", "entry_frenzy_above_streak", "entry_frenzy_spike_at", "entry_frenzy_bar_ret_pct"]
    fr = []
    for f in fs:
        d = pd.read_csv(f, low_memory=False)
        if "entry_strategy" in d and "entry_frenzy_vwap" in d:
            d = d[d.entry_strategy == "FRENZY_LITE"]
            if len(d):
                fr.append(d[[c for c in cols if c in d]])
    D = pd.concat(fr)
    D["te"] = pd.to_datetime(D.opened_at.astype(str).str[:19]).values.astype("datetime64[ms]").astype("int64")
    D = D.drop_duplicates(["pair", "te"])
    M = M.merge(D, on=["pair", "te"], how="left")
    M["T"] = (M.te // T.BAR_MS) * T.BAR_MS - T.BAR_MS
    M = M.merge(B[["pair", "T", "spike_ts", "vwap", "spike_volx", "spike_high", "hi_since", "c", "streak", "hours"]], on=["pair", "T"], how="left",
                suffixes=("", "_rb"))
    M["vwap"] = M.vwap.fillna(M.entry_frenzy_vwap)
    return M


def main():
    t0 = time.time()
    n = walker_parity()
    B = S.load_bars()
    U = B[(~B.bear) & (B.hours >= 2.0) & (B.hours <= 24.0) & (~B.in_state) & (B.vol_mult < 100)]
    # R1 bulk
    if not os.path.exists(os.path.join(T.OUT, "y1m.pkl")):
        jobs = [(p, G["T"].values.tolist()) for p, G in U.groupby("pair")]
        with Pool(JOBS) as Pp:
            R1 = pd.DataFrame(sum(Pp.map(_r1_job, jobs, chunksize=1), []))
        R1.to_pickle(os.path.join(T.OUT, "y1m.pkl"))
        print(f"R1 bulk: {len(R1):,} bars · {(R1.why1 == 'UNRES').mean() * 100:.1f}% UNRES · {(R1.why1 == 'NOENTRY').mean() * 100:.1f}% no 1m entry · "
              f"{time.time() - t0:.0f}s", flush=True)
    # cells
    Cc, base = build_cells(B)
    Cc.to_pickle(os.path.join(T.OUT, "cells.pkl")); base.to_pickle(os.path.join(T.OUT, "base_judged.pkl"))
    print(Cc.groupby("cell").size().to_string(), flush=True)
    uniq = Cc[["pair", "T"]].drop_duplicates()
    if not os.path.exists(os.path.join(T.OUT, "yrh.pkl")):
        items = uniq.sort_values(["pair", "T"]).to_dict("records")
        pairs = sorted(uniq.pair.unique()); gi = {p: k % (JOBS * 6) for k, p in enumerate(pairs)}
        ch = [[r for r in items if gi[r["pair"]] == k] for k in range(JOBS * 6)]
        with Pool(JOBS) as Pp:
            RH = pd.DataFrame(sum(Pp.map(_rh_job, [c for c in ch if c], chunksize=1), []))
        RH.to_pickle(os.path.join(T.OUT, "yrh.pkl"))
        print(f"RH: {len(RH):,} signal bars · UNRES {(RH.whyh == 'UNRES').sum()} · src {RH.srch.value_counts().to_dict()} · {time.time() - t0:.0f}s", flush=True)
    # A6 / A7 on today's signals + master
    if not os.path.exists(os.path.join(T.OUT, "a67.pkl")):
        tod = Cc[Cc.cell == "TODAY"].merge(B[["pair", "T", "spike_ts", "vwap"]], on=["pair", "T"])
        vw = B[["pair", "spike_ts", "T", "c", "vwap"]]
        jobs = [(p, G.to_dict("records"), vw[vw.pair == p]) for p, G in tod.groupby("pair")]
        with Pool(JOBS) as Pp:
            A = pd.DataFrame(sum(Pp.map(_a67_job, jobs, chunksize=1), []))
        A.to_pickle(os.path.join(T.OUT, "a67.pkl"))
        M = master_rows(B)
        M.to_pickle(os.path.join(T.OUT, "master_rows.pkl"))
        mj = [(p, G.assign(T=G["T"].astype("int64")).to_dict("records"), vw[vw.pair == p]) for p, G in M.groupby("pair")]
        with Pool(JOBS) as Pp:
            AM = pd.DataFrame(sum(Pp.map(_a67_job, mj, chunksize=1), []))
        AM.to_pickle(os.path.join(T.OUT, "a67_master.pkl"))
        print(f"A6/A7: {len(A):,} rows · master {len(AM)} · {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
