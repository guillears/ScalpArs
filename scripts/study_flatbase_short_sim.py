#!/usr/bin/env python3
"""FRENZY flat-base short (2026-10-22) — SIGNALS + EXITS + RANDOM-BAR NULL (pre-registration: reports/FRENZY_FLATBASE_SHORT_2026-10-22_PREREG.md).

  --need   writes data/fund_needed.csv: every (pair, month) an episode can touch (spike − 1 d → spike + 96 h + 14 d), from episodes only
  default  per pair (multiprocessing): the four pre-registered entries S1–S4 on the full 5m grid, every one of the 54 exits
           (hold 3/7/14 d × TP none/base × stop none/25/50 × lev 1/2/3, gap-aware stops, liquidation = −100/L % of notional),
           0.20 % costs + REAL funding, the one-open-short-per-pair rule per cell, daily mark-to-market, and 200 random-bar null draws
           per episode (bar uniform in [spike + 2 h, spike + 96 h] passing the same universe gates; same exits; funding incl.)
           → data/signals.pkl (one row per signal: features + per-exit arrays) · data/null.npy (episodes × 200 × 54, float32) · data/null_eps.npy
Usage: venv/bin/python scripts/study_flatbase_short_sim.py [--need] [--procs 8]
"""
import argparse, os, pickle, sys, time, zlib
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402

NB = 200
HMAX = 14 * 288
EXITS = C.EXITS
NX = len(EXITS)
LIQ_X = {lv: 1.0 / lv - 0.01 for lv in C.LEVS}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def first_idx(mask):
    """first True per row, HMAX + 10 when none."""
    a = mask.argmax(1)
    return np.where(mask.any(1), a, HMAX + 10)


def simulate(E, base, H, Lo, O, Cc, T, avail, data_open):
    """rows = entries. H/Lo/O/Cc/T: n × HMAX paths starting at the bar AFTER the entry bar (NaN beyond the data).
    → gross % (before fees and funding), exit close time (ms), reason code (0 time, 1 stop, 2 liq, 3 tp, 4 open/MTM, 5 delist settle)."""
    n = len(E)
    hh = np.where(np.isnan(H), -np.inf, H); ll = np.where(np.isnan(Lo), np.inf, Lo)
    cmh = np.maximum.accumulate(hh, axis=1); cml = np.minimum.accumulate(ll, axis=1)
    k_stop = {s: first_idx(cmh >= (E * (1 + s / 100))[:, None]) for s in C.STOPS if s > 0}
    k_liq = {lv: first_idx(cmh >= (E * (1 + LIQ_X[lv]))[:, None]) for lv in C.LEVS}
    tp_ok = base < E * 0.99
    k_tp = np.where(tp_ok, first_idx(cml <= np.where(tp_ok, base, -np.inf)[:, None]), HMAX + 10)
    r = np.arange(n)
    G = np.zeros((n, NX)); XT = np.zeros((n, NX), dtype="int64"); RS = np.zeros((n, NX), dtype="int8")
    for j, (hd, tp, s, lv) in enumerate(EXITS):
        hb = hd * 288
        kmax = np.minimum(hb, avail)                      # bars available inside the hold
        ks = k_stop[s] if s > 0 else np.full(n, HMAX + 10)
        kl = k_liq[lv]
        kt = k_tp if tp == "base" else np.full(n, HMAX + 10)
        ev = np.minimum(np.minimum(ks, kl), kt)
        has = ev < kmax
        kx = np.where(has, ev, np.maximum(kmax - 1, 0))
        g = np.full(n, np.nan); rs = np.zeros(n, dtype="int8")
        # time / end of data
        xc = Cc[r, np.clip(kx, 0, HMAX - 1)] * (1 + C.SLIP)
        g[:] = (E - xc) / E * 100
        rs[:] = np.where(kmax < hb, np.where(data_open, 4, 5), 0)
        # stop (stop first on ties with TP; capped by the liquidation loss)
        st = has & (ks == ev)
        if st.any():
            Sx = np.maximum(E * (1 + s / 100), O[r, np.clip(ks, 0, HMAX - 1)]) * (1 + C.SLIP)
            gs = np.maximum((E - Sx) / E * 100, -100.0 / lv)
            g = np.where(st, gs, g); rs = np.where(st, 1, rs)
        lq = has & (kl == ev) & ~st
        g = np.where(lq, -100.0 / lv, g); rs = np.where(lq, 2, rs)
        tt = has & (kt == ev) & ~st & ~lq
        if tt.any():
            Tx = np.minimum(base, O[r, np.clip(kt, 0, HMAX - 1)]) * (1 + C.SLIP)
            g = np.where(tt, (E - Tx) / E * 100, g); rs = np.where(tt, 3, rs)
        G[:, j] = g; RS[:, j] = rs
        XT[:, j] = T[r, np.clip(kx, 0, HMAX - 1)] + C.BAR
    return G, XT, RS


def paths(arr, idx, n):
    t, o, h, l, c = arr
    m = len(idx)
    H = np.full((m, HMAX), np.nan); Lo = H.copy(); O = H.copy(); Cc = H.copy(); T = np.zeros((m, HMAX), dtype="int64")
    avail = np.zeros(m, dtype="int64")
    for r, i in enumerate(idx):
        a, b = i + 1, min(n, i + 1 + HMAX); k = b - a
        avail[r] = max(k, 0)
        if k > 0:
            H[r, :k] = h[a:b]; Lo[r, :k] = l[a:b]; O[r, :k] = o[a:b]; Cc[r, :k] = c[a:b]; T[r, :k] = t[a:b]
            if k < HMAX:
                T[r, k:] = t[b - 1] + C.BAR * np.arange(1, HMAX - k + 1)
    return H, Lo, O, Cc, T, avail


def worker(args):
    pair, eps, now_ms = args
    full = C.load_full(pair)
    if full is None:
        return [], [], []
    t, o, h, l, c, v, qv, real = full
    n = len(t)
    data_open = bool(t[-1] + C.BAR >= now_ms - 2 * C.HR)
    F = C.funding_table(pair)
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    q = v * (h + l + c) / 3.0
    tp_ = (h + l + c) / 3.0
    idx_of = lambda ts: int((ts - t[0]) // C.BAR)
    sigs, nulls, null_ids = [], [], []
    rng = np.random.default_rng(zlib.crc32(pair.encode()))
    for e in eps.itertuples():
        on = idx_of(e.on_t); off = idx_of(e.off_t)
        fs = idx_of(e.first_seen_t) if e.first_seen_t is not None and not pd.isna(e.first_seen_t) else 10**12
        if on < 6 or on >= n:
            continue
        gate = lambda i: bool(0 <= i < n - 1 and i >= fs and vol24[i] >= 20e6)
        hi = min(n - 2, on + C.SEARCH)
        found = {}
        if not e.new_ep_at_end and off < n - 1 and gate(off) and off <= on + C.SEARCH + 1:
            found["S1"] = off
        w = np.arange(on, hi + 1)
        if len(w):
            cm = np.maximum.accumulate(h[w]); jj = np.arange(len(w))
            pk = np.maximum.accumulate(np.where(h[w] >= cm, jj, 0))
            cond = (c[w] <= 0.85 * cm) & (jj - pk >= 72) & (w >= fs) & (np.nan_to_num(vol24[w]) >= 20e6)
            k = np.nonzero(cond)[0]
            if len(k):
                found["S2"] = int(w[k[0]])
            pv = np.cumsum(tp_[w] * q[w]); qq = np.cumsum(q[w]); vw = np.where(qq > 0, pv / np.where(qq > 0, qq, 1), c[w])
            end4 = ((t[w] + C.BAR) % (4 * C.HR) == 0) & (w >= on + 48)
            cond4 = end4 & (c[w] < vw) & (w >= fs) & (np.nan_to_num(vol24[w]) >= 20e6)
            k = np.nonzero(cond4)[0]
            if len(k):
                found["S4"] = int(w[k[0]])
        if gate(on + 288):
            found["S3"] = on + 288
        # peak for reporting
        for en, i in found.items():
            sigs.append(dict(ep_on_t=e.on_t, pair=pair, entry=en, i_t=int(t[i]), entry_ms=int(t[i]) + C.BAR, close=float(c[i]),
                             E=float(c[i]) * (1 - C.SLIP), base=float(e.base), peak=float(h[on:i + 1].max()),
                             vol24=float(vol24[i]), vwap_dummy=0.0, _i=i))
        # random-bar null
        cand = np.arange(on + 24, hi + 1)
        if len(cand):
            okc = (cand >= fs) & (np.nan_to_num(vol24[cand]) >= 20e6)
            cand = cand[okc]
        if len(cand):
            ri = rng.choice(cand, NB, replace=True)
            Hh, Ll, Oo, Cc, Tt, av = paths((t, o, h, l, c), ri, n)
            Ee = c[ri] * (1 - C.SLIP); Bb = np.full(NB, e.base)
            G, XT, _ = simulate(Ee, Bb, Hh, Ll, Oo, Cc, Tt, av, data_open)
            t0 = np.repeat((t[ri] + C.BAR)[:, None], NX, 1)
            fu, _ = C.funding_sum(F, t0.ravel(), XT.ravel())
            nulls.append((G - C.FEES + fu.reshape(G.shape)).astype(np.float32))
            null_ids.append(e.on_t)
    if not sigs:
        return [], nulls, [(pair, x) for x in null_ids]
    S = pd.DataFrame(sigs)
    idx = S._i.values
    Hh, Ll, Oo, Cc, Tt, av = paths((t, o, h, l, c), idx, n)
    G, XT, RS = simulate(S.E.values, S.base.values, Hh, Ll, Oo, Cc, Tt, av, data_open)
    t0 = np.repeat(S.entry_ms.values[:, None], NX, 1)
    fu, miss = C.funding_sum(F, t0.ravel(), XT.ravel())
    FU = fu.reshape(G.shape); MS = miss.reshape(G.shape)
    # daily mark-to-market of the raw short (no exit) for days 1..14
    MK = np.full((len(S), 14), np.nan)
    for d in range(1, 15):
        k = d * 288 - 1
        okk = av > k
        MK[okk, d - 1] = (S.E.values[okk] - Cc[okk, k]) / S.E.values[okk] * 100
    # one open short per pair per cell (signals in time order)
    TK = np.ones(G.shape, dtype=bool)
    order = np.argsort(S.entry_ms.values, kind="stable")
    for en in C.ENTRIES:
        rows = [r for r in order if S.entry.values[r] == en]
        for j in range(NX):
            busy = -1
            for r in rows:
                if S.entry_ms.values[r] < busy:
                    TK[r, j] = False
                else:
                    busy = XT[r, j]
    S["gross"] = list(G.astype(np.float32)); S["exit_ms"] = list(XT); S["reason"] = list(RS); S["fund"] = list(FU.astype(np.float32))
    S["fund_missing"] = list(MS); S["taken"] = list(TK); S["mtm"] = list(MK.astype(np.float32)); S["avail"] = av
    S["data_open"] = data_open; S["last_bar"] = int(t[-1])
    return S.drop(columns=["_i", "vwap_dummy"]).to_dict("records"), nulls, [(pair, x) for x in null_ids]


def need():
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    E = E[E.spike >= pd.Timestamp("2024-12-31")]
    mm = set()
    for p, a in zip(E.pair, E.spike_ts):
        a0 = pd.to_datetime(a - C.DAY, unit="ms"); b0 = min(pd.to_datetime(a + 4 * C.DAY + 15 * C.DAY, unit="ms"), pd.Timestamp("2026-10-22 23:00"))
        for m in pd.period_range(a0, b0, freq="M").strftime("%Y-%m"):
            mm.add((p, m))
    df = pd.DataFrame(sorted(mm), columns=["pair", "month"])
    df.to_csv(os.path.join(C.D, "fund_needed.csv"), index=False)
    log("fund needed", len(df), "pairs", df.pair.nunique(), "oct pairs", (df.month == "2026-10").sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--need", action="store_true"); ap.add_argument("--procs", type=int, default=8)
    A = ap.parse_args()
    if A.need:
        need(); sys.exit(0)
    E = pd.read_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    E = E[E.spike >= pd.Timestamp("2025-01-01")].copy()
    sol = pd.read_csv(os.path.join(C.K5, "SOLUSDT.csv"))
    now_ms = int(sol.open_time.max()) + C.BAR
    log("episodes", len(E), "now", pd.to_datetime(now_ms, unit="ms"))
    jobs = [(p, g, now_ms) for p, g in E.groupby("pair")]
    jobs.sort(key=lambda x: -len(x[1]))
    SIG, NUL, NID = [], [], []
    with mp.Pool(A.procs) as pool:
        for k, (s, nl, ni) in enumerate(pool.imap_unordered(worker, jobs)):
            SIG += s; NUL += nl; NID += ni
            if k % 50 == 0:
                log(k, len(jobs), len(SIG))
    S = pd.DataFrame(SIG)
    S.to_pickle(os.path.join(C.D, "signals.pkl"))
    np.save(os.path.join(C.D, "null.npy"), np.stack(NUL))
    pickle.dump(NID, open(os.path.join(C.D, "null_eps.pkl"), "wb"))
    pickle.dump(dict(now_ms=now_ms), open(os.path.join(C.D, "meta.pkl"), "wb"))
    log("signals", len(S), "null eps", len(NID))
