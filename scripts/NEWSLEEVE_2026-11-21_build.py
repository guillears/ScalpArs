#!/usr/bin/env python3
"""NEWSLEEVE 2026-11-21 — BUILD (PREREG reports/NEWSLEEVE_2026-11-21_PREREG.md, frozen before this script was run).
Per symbol (5m klines with taker-buy, reports/NEWSLEEVE_2026-11-21_data/k5m): triggers H5 (taker climax), H1/H2 raw candidates
(price + volume legs; the OI leg is joined later from 5m metrics), H6 (funding settlements). For every trigger: the 5m exit walk over
the PREREG exit grid with costs + real funding, one paired random-entry control (same coin, same UTC day, ≥ 1 h away, same direction),
and the shuffled null (100 random entries per possibility; H5 summed per cell in the worker, H1/H2/H6 stored per event).
Writes reports/NEWSLEEVE_2026-11-21_data/build_<IDEA>.pkl."""
import os, sys, pickle, time, zlib
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(R, "reports", "NEWSLEEVE_2026-11-21_data")
HA = os.path.join(R, "reports", "HIGH_ATR_2026-11-18_data")
B5 = 300_000
T0 = int(pd.Timestamp("2023-07-01").value // 1e6)
T1 = int(pd.Timestamp("2026-11-17").value // 1e6)
TD = int(pd.Timestamp("2025-07-01").value // 1e6)          # DESIGN < TD ≤ HOLDOUT
NREP = 100
PRE = 12                                                    # event-study bars before entry
EXIT_A = [(tp, sl, hb) for tp in (0.01, 0.02, 0.04) for sl in (0.02, 0.04) for hb in (12, 72)]     # H5 / H1 / H2
EXIT_F = [(tp, sl, hb) for tp in (0.03, 0.06) for sl in (0.03, 0.06) for hb in (96, 288)]          # H6
H5V = [(5.0, 0.60), (10.0, 0.65)]
TAKER, MAKER, THRU = 0.05, 0.02, 0.0005


def slip_base(v24):
    return np.where(v24 >= 200e6, 0.02, np.where(v24 >= 50e6, 0.05, 0.10))


def load_funding(sym):
    parts = []
    for d in ("funding", "funding_nov"):
        f = os.path.join(HA, d, sym + ".npy")
        if os.path.exists(f):
            a = np.load(f)
            if a.ndim == 2 and len(a):
                parts.append(a[:, :2])
    if not parts:
        return np.zeros((0, 2))
    a = np.concatenate(parts)
    a = a[np.argsort(a[:, 0])]
    _, ix = np.unique(a[:, 0], return_index=True)
    return a[ix]


def windows(arr, idx, W, pre=0):
    """rows arr[i-pre : i+W] for each i in idx (NaN outside)."""
    n = len(arr)
    j = idx[:, None] + np.arange(-pre, W)[None, :]
    ok = (j >= 0) & (j < n)
    out = np.full(j.shape, np.nan)
    out[ok] = arr[j[ok]]
    return out


def walk(Ow, Hw, Lw, Cw, Gw, d, v24, cells):
    """Ow..Cw: (n, W) prices from the entry bar on, already divided by the entry open. Gw: cumulative funding at each bar open
    minus at the entry bar open. d: +1 long / −1 short (scalar). Returns net % (n, ncells) and exit kind (n, ncells) 0 TP 1 SL 2 time."""
    if d > 0:
        fav, adv, op, cl = Hw - 1, Lw - 1, Ow - 1, Cw - 1
    else:
        fav, adv, op, cl = 1 - Lw, 1 - Hw, 1 - Ow, 1 - Cw
    cl = pd.DataFrame(cl).ffill(axis=1).values
    Gw = pd.DataFrame(Gw).ffill(axis=1).values
    n = len(Ow)
    base = slip_base(v24)
    out = np.full((n, len(cells)), np.nan, np.float32)
    kind = np.zeros((n, len(cells)), np.int8)
    rows = np.arange(n)
    for c, (tp, sl, hb) in enumerate(cells):
        a = np.nan_to_num(adv[:, :hb], nan=0.0) <= -sl
        f = np.nan_to_num(fav[:, :hb], nan=0.0) >= tp + THRU
        js = np.where(a.any(1), a.argmax(1), 10**6)
        jt = np.where(f.any(1), f.argmax(1), 10**6)
        is_sl = (js <= jt) & (js < hb)
        is_tp = (~is_sl) & (jt < hb)
        jx = np.where(is_sl, js, np.where(is_tp, jt, hb - 1))
        o_at = op[rows, np.minimum(jx, op.shape[1] - 1)]
        ret = np.where(is_sl, np.where(np.nan_to_num(o_at, nan=0.0) <= -sl, o_at, -sl),
                       np.where(is_tp, tp, cl[rows, hb - 1]))
        fee = TAKER + 2 * base + np.where(is_tp, MAKER, np.where(is_sl, TAKER + 2 * base, TAKER + base))
        fund = -d * np.nan_to_num(Gw[rows, jx], nan=0.0) * 100
        out[:, c] = ret * 100 - fee + fund
        kind[:, c] = np.where(is_tp, 0, np.where(is_sl, 1, 2))
    return out, kind


class Sym:
    def __init__(self, sym):
        z = np.load(os.path.join(D, "k5m", sym + ".npz"))
        t = z["t"]
        self.ok = len(t) > 3000
        if not self.ok:
            return
        t0 = t[0]
        n = int((t[-1] - t0) // B5) + 1
        ix = ((t - t0) // B5).astype(np.int64)
        self.t = t0 + np.arange(n, dtype=np.int64) * B5
        for k in ("o", "h", "l", "c"):
            a = np.full(n, np.nan); a[ix] = z[k]; setattr(self, k, a)
        qv = np.full(n, np.nan); qv[ix] = z["qv"]; tb = np.full(n, np.nan); tb[ix] = z["tb"]
        s = pd.Series(qv)
        self.qv15 = s.rolling(3, min_periods=3).sum().values
        tb15 = pd.Series(tb).rolling(3, min_periods=3).sum().values
        self.share = tb15 / np.where(self.qv15 > 0, self.qv15, np.nan)
        self.usual = pd.Series(self.qv15).shift(12).rolling(2016, min_periods=1800).median().values
        self.ratio = self.qv15 / np.where(self.usual > 0, self.usual, np.nan)
        cc = pd.Series(self.c)
        self.r15 = (cc / cc.shift(3) - 1).values
        self.v24 = s.rolling(288, min_periods=280).sum().values
        fu = load_funding(sym)
        self.fu = fu
        G = np.zeros(n)
        if len(fu):
            # settlement s counts for bar j when t_s <= t_j (bar open); cumulative
            k = np.searchsorted(self.t, fu[:, 0], side="left")
            add = np.zeros(n + 1)
            np.add.at(add, np.clip(k, 0, n), fu[:, 1])
            G = np.cumsum(add)[:n]
        self.G = G
        self.n = n
        self.day = self.t // 86_400_000
        self.univ = (self.v24 >= 10e6) & np.isfinite(self.usual) & np.isfinite(self.c)

    def paths(self, e, W, pre=0):
        P0 = self.o[e]
        o = windows(self.o, e, W, pre) / P0[:, None]
        h = windows(self.h, e, W, pre) / P0[:, None]
        l = windows(self.l, e, W, pre) / P0[:, None]
        c = windows(self.c, e, W, pre) / P0[:, None]
        g = windows(self.G, e, W, pre) - self.G[e][:, None]
        return o, h, l, c, g

    def random_bars(self, trig, reps, rng):
        """for each trigger bar i: `reps` random bars same UTC day, |j-i| >= 12, in universe, with an entry bar."""
        cand = np.where(self.univ[:-1] & np.isfinite(self.o[1:]))[0]
        cday = self.day[cand]
        out = np.full((len(trig), reps), -1, np.int64)
        for k, i in enumerate(trig):
            lo, hi = np.searchsorted(cday, self.day[i]), np.searchsorted(cday, self.day[i], side="right")
            pool = cand[lo:hi]
            pool = pool[np.abs(pool - i) >= 12]
            if len(pool):
                out[k] = pool[rng.integers(0, len(pool), reps)]
        return out


def dedup(idx, gap):
    keep, last = [], -10**12
    for i in idx:
        if i >= last + gap:
            keep.append(i); last = i
    return np.array(keep, np.int64)


def evaluate(S, trig, d, cells, W, rng, null_mode):
    """trig = trigger bar indices (entry = trig+1)."""
    e = trig + 1
    o, h, l, c, g = S.paths(e, W, 0)
    net, kind = walk(o, h, l, c, g, d, S.v24[trig], cells)
    _, _, _, cpre, _ = S.paths(e, W + 1, PRE)
    rb = S.random_bars(trig, 1 + NREP, rng)
    res = dict(trig=trig, net=net, kind=kind, path=(cpre.astype(np.float32)), v24=S.v24[trig])
    # control (rep 0)
    cb = rb[:, 0]; okc = cb >= 0
    cnet = np.full_like(net, np.nan); cpath = np.full_like(res["path"], np.nan)
    if okc.any():
        o2, h2, l2, c2, g2 = S.paths(cb[okc] + 1, W, 0)
        cnet[okc], _ = walk(o2, h2, l2, c2, g2, d, S.v24[cb[okc]], cells)
        _, _, _, cp2, _ = S.paths(cb[okc] + 1, W + 1, PRE)
        cpath[okc] = cp2
    res["cnet"], res["cpath"] = cnet, cpath
    # null reps 1..NREP
    nb = rb[:, 1:]
    flat = nb.reshape(-1)
    okn = flat >= 0
    nnet = np.full((len(flat), len(cells)), np.nan, np.float32)
    pos = np.where(okn)[0]
    step = max(1, 1_500_000 // W)
    for a0 in range(0, len(pos), step):
        pp = pos[a0:a0 + step]
        o3, h3, l3, c3, g3 = S.paths(flat[pp] + 1, W, 0)
        nnet[pp], _ = walk(o3, h3, l3, c3, g3, d, S.v24[flat[pp]], cells)
    nnet = nnet.reshape(len(trig), NREP, len(cells))
    if null_mode == "store":
        res["nnet"] = nnet.astype(np.float16)
    return res, nnet


def one(sym):
    rng = np.random.default_rng(zlib.crc32(sym.encode()))
    try:
        S = Sym(sym)
    except Exception as ex:
        return sym, {"err": str(ex)}
    if not S.ok:
        return sym, {}
    out = {}
    inrange = (S.t + B5 >= T0) & (S.t + B5 < T1)   # trigger bar close inside the study span
    base = S.univ & inrange & np.isfinite(np.r_[S.o[1:], np.nan])
    # ---- H5
    for vi, (M, Sh) in enumerate(H5V):
        for d, nm in ((-1, "H5B"), (1, "H5S")):
            cond = base & (S.ratio >= M) & ((S.share >= Sh) if d < 0 else (S.share <= 1 - Sh))
            raw = np.where(cond)[0]
            if not len(raw):
                continue
            keep = dedup(raw, 72)
            res, nnet = evaluate(S, keep, d, EXIT_A, 72, rng, "sum")
            des = (S.t[keep] + B5) < TD
            nn = nnet[des]
            res["null_sum"] = np.nansum(nn, axis=0)                         # (NREP, ncells) over DESIGN possibilities
            res["null_cnt"] = np.isfinite(nn).sum(axis=0)
            res["n_raw"] = len(raw)
            out[f"{nm}_v{vi}"] = res
    # ---- H1 / H2 raw candidates (price + volume legs at the loosest X)
    for d, nm in ((1, "H1"), (-1, "H2")):
        cond = base & (S.ratio >= 5) & ((S.r15 <= -0.05) if d > 0 else (S.r15 >= 0.05))
        raw = np.where(cond)[0]
        if len(raw):
            res, _ = evaluate(S, raw, d, EXIT_A, 72, rng, "store")
            res["r15"] = S.r15[raw]; res["ratio"] = S.ratio[raw]
            out[nm] = res
    # ---- H6 funding settlements
    if len(S.fu):
        ts = S.fu[:, 0].astype(np.int64); rate = S.fu[:, 1]
        e = np.searchsorted(S.t, ts + B5)            # entry bar opens 5 min after the settlement
        okk = (e < S.n) & (e >= 1)
        e, rate, ts = e[okk], rate[okk], ts[okk]
        okk = (S.t[e] == ts + B5)
        e, rate, ts = e[okk], rate[okk], ts[okk]
        trig = e - 1
        okk = S.univ[trig] & (ts >= T0) & (ts < T1) & np.isfinite(S.o[e])
        for d, nm, m in ((1, "H6L", rate <= -0.005), (-1, "H6S", rate >= 0.001)):
            sel = okk & m
            if sel.any():
                raw = trig[sel]
                keep = dedup(raw, 288)
                rr = dict(zip(raw, rate[sel]))
                res, _ = evaluate(S, keep, d, EXIT_F, 288, rng, "store")
                res["rate"] = np.array([rr[k] for k in keep]); res["n_raw"] = len(raw)
                out[nm] = res
    for k, res in out.items():
        res["t_trig"] = S.t[res["trig"]] + B5            # trigger bar close
        res["ratio_at"] = S.ratio[res["trig"]]; res["share_at"] = S.share[res["trig"]]; res["r15_at"] = S.r15[res["trig"]]
    return sym, out


def main():
    syms = sorted(f[:-4] for f in os.listdir(os.path.join(D, "k5m")) if f.endswith(".npz") and f != "BTCUSDT.npz")
    if len(sys.argv) > 1:
        syms = syms[: int(sys.argv[1])]
    print(time.strftime("%H:%M:%S"), "symbols", len(syms), flush=True)
    agg = {}
    with mp.Pool(8) as pool:
        for n, (sym, out) in enumerate(pool.imap_unordered(one, syms, chunksize=2)):
            if "err" in out:
                print("ERR", sym, out["err"], flush=True); continue
            for k, res in out.items():
                res["sym"] = np.array([sym] * len(res["trig"]))
                agg.setdefault(k, []).append(res)
            if n % 50 == 0:
                print(time.strftime("%H:%M:%S"), n, sym, flush=True)
    for k, lst in agg.items():
        merged = {}
        for key in lst[0]:
            vals = [r[key] for r in lst if key in r]
            if key in ("null_sum", "null_cnt"):
                merged[key] = np.sum(vals, axis=0)
            elif key == "n_raw":
                merged[key] = int(np.sum(vals))
            else:
                merged[key] = np.concatenate(vals)
        pickle.dump(merged, open(os.path.join(D, f"build_{k}.pkl"), "wb"), protocol=4)
        print(k, len(merged["trig"]), flush=True)


if __name__ == "__main__":
    main()
