#!/usr/bin/env python3
"""EXHAUSTION EXPAND (PREREG reports/EXHAUSTION_EXPAND_2026-11-18_PREREG.md) — BUILD. Per pair, from the continuous 5m series:
Frenzy windows (episodes D ∪ C ∪ U1 from the validated walk) → every eligible 5m bar with vol_decay ≥ 0.40 (X1–X3, X5, REF + their
random pools) and every non-Frenzy 15m-close mover bar with vd24 ≥ 0.6366 (X4 + its random pool); signal bits (RSI14X 5m / 15m / 1h,
BBRE / VSW / STOCH 15m — catalog definitions), ATR15, Q5, spot_share_chg (REF only), and the 4 X6 exits (TP maker through by 1 tick,
gap-aware stops, 24 h). Funding is added in the eval step. Lag-safe: every input closes ≤ the signal bar's close; entry = next 5m open.
Output: reports/EXHAUSTION_EXPAND_data/rows/<PAIR>.pkl.  Usage: venv/bin/python scripts/exh_expand_build.py [--pairs A,B] [--procs 7]"""
import argparse, glob, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import frenzy_ta_build as TB   # noqa: E402

D = os.environ.get("EXH_D", os.path.join(R, "reports", "EXHAUSTION_EXPAND_data")); os.makedirs(os.path.join(D, "rows"), exist_ok=True)
SSD = os.path.join(R, "reports", "SPOT_SHARE_DEEP_data")
DD = os.path.join(R, "reports", "EXHAUSTION_EXPAND_data")
TB.K5M_SRC = [os.path.join(DD, "k5m"), os.path.join(SSD, "k5m")] + TB.K5M_SRC + [os.path.join(SSD, "k5m_jun")]
TB.FUND_SRC = TB.FUND_SRC + [os.path.join(SSD, "funding"), os.path.join(DD, "funding")]
SPOT_DIRS = [os.path.join(R, "reports", "FRENZY_EARLY_DETECTION_data", "spot"), os.path.join(R, "reports", "TA_WL_DEEP_2026-11-17_data", "spot"),
             os.path.join(SSD, "spot"), os.path.join(DD, "spot")]
BAR, M15, H1, DAY = 300_000, 900_000, 3_600_000, 86_400_000
WIN = 72 * H1
VD_MIN_F, VD_X4 = 0.40, 0.6366
EXITS = {"EA": ("atr", 3.0, 5.0, "atr", 1.5, 5.0), "EB": ("atr", 3.0, 3.0, "atr", 1.5, 5.0),
         "EC": ("pct", 3.0, None, "pct", 3.0, None), "ED": ("pct", 5.0, None, "pct", 5.0, None)}
HOLD = 288
SIG = ["RSI5", "RSI15", "BBRE15", "VSW15", "STOCH15", "RSI1H"]
EV_OFF = np.array([-24, -12, 12, 48, 144, 288])


def episodes():
    if os.environ.get("EXH_META"):
        return pd.read_csv(os.environ["EXH_META"]).assign(u="AS")
    fd = os.path.join(DD, "episodes_meta_D.csv")
    a = pd.read_csv(fd).assign(u="D") if os.path.exists(fd) else pd.DataFrame()
    b = pd.read_csv(os.path.join(SSD, "episodes_meta.csv")).assign(u="C")
    c = pd.read_csv(os.path.join(R, "reports", "FRENZY_TA_CATALOG_data", "episodes_meta.csv")).assign(u="U1")
    return pd.concat([a, b, c], ignore_index=True)


def spot_frame(p):
    xs = []; ok = set(); tried = set()
    for d in SPOT_DIRS:
        for f in glob.glob(os.path.join(d, p + ".*.csv")):
            m = os.path.basename(f).split(".")[1]; tried.add(m)
            try:
                with open(f) as fh:
                    if not fh.readline().startswith("open_time"):
                        continue
                xs.append(pd.read_csv(f)); ok.add(m)
            except Exception:
                pass
    if not xs:
        return None, ok, tried
    return pd.concat(xs).drop_duplicates("open_time").sort_values("open_time"), ok, tried


def agg(t, o, h, l, c, v, P):
    g = t // P
    st = np.r_[0, np.nonzero(np.diff(g))[0] + 1]; en = np.r_[st[1:], len(t)] - 1
    return st, en, o[st], np.maximum.reduceat(h, st), np.minimum.reduceat(l, st), c[en], np.add.reduceat(v, st)


def exits(o, h, l, c, J, atrp, tick):
    N = len(c); out = {}
    for a0 in range(0, len(J), 4000):
        jj = J[a0:a0 + 4000]; ap = atrp[a0:a0 + 4000]
        idx = jj[:, None] + np.arange(HOLD)[None, :]; valid = idx < N; idc = np.minimum(idx, N - 1)
        Oo, Hh, Ll, Cc = o[idc], h[idc], l[idc], c[idc]; O = o[jj]; rows = np.arange(len(jj))
        last = np.minimum(HOLD - 1, N - 1 - jj)
        for en, (tk, tm, tc, sk, sm, sc) in EXITS.items():
            tpd = ap * tm if tk == "atr" else np.full(len(jj), tm); tpd = np.minimum(tpd, tc) if tc else tpd
            sld = ap * sm if sk == "atr" else np.full(len(jj), sm); sld = np.minimum(sld, sc) if sc else sld
            TPp = O * (1 - tpd / 100); SLp = O * (1 + sld / 100)
            ht = (Ll <= TPp[:, None] - tick) & valid; hs = (Hh >= SLp[:, None]) & valid
            fs = np.where(hs.any(1), hs.argmax(1), 9999); ft = np.where(ht.any(1), ht.argmax(1), 9999)
            stop = (fs <= ft) & (fs < 9999); tp = (ft < fs) & (ft < 9999)
            X = np.where(stop, np.maximum(Oo[rows, np.minimum(fs, HOLD - 1)], SLp), np.where(tp, TPp, Cc[rows, last]))
            col = np.where(stop, fs, np.where(tp, ft, last))
            r = -(X / O - 1) * 100; k = np.where(stop, 1, np.where(tp, 0, 2)).astype(np.int8)
            for nm, v in ((f"{en}_r", r.astype(np.float32)), (f"{en}_k", k), (f"{en}_c", col.astype(np.int16))):
                out.setdefault(nm, []).append(v)
    return {k: np.concatenate(v) for k, v in out.items()}


def work(args):
    p, eps, skip = args
    fo = os.path.join(D, "rows", p + ".pkl")
    if os.path.exists(fo):
        return p, -1
    x = TB.load(p)
    if x is None:
        return p, 0
    t, o, h, l, c, v, qv = x
    N = len(t)
    if N < 600:
        return p, 0
    tick = np.diff(np.unique(np.r_[o, h, l, c])); tick = float(tick[tick > 1e-15].min()) if len(tick) else 0.0
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    roll1h = pd.Series(qv).rolling(12, min_periods=12).sum().values
    r1n = np.nan_to_num(roll1h, nan=0.0)
    # ── 15m frame (signals + ATR), mapped to the 5m index of each 15m bar's last 5m bar
    st, en, O15, H15, L15, C15, V15 = agg(t, o, h, l, c, v, M15)
    S15 = TB.family_signals(O15, H15, L15, C15, V15, np.full(len(C15), np.nan))
    A15 = TB.atr(H15, L15, C15, 14)
    is15 = np.zeros(N, bool); is15[en] = True
    # 15m bars must be complete (3 bars) for the close to be a real 15m close
    full15 = (en - st) == 2
    sig = {k: np.zeros(N, bool) for k in SIG}
    sig["RSI15"][en] = S15["RSI14X"][1] & full15; sig["BBRE15"][en] = S15["BBRE"][1] & full15
    sig["VSW15"][en] = S15["VSW"][1] & full15; sig["STOCH15"][en] = S15["STOCH"][1] & full15
    # ATR of the last CLOSED 15m bar at each 5m bar's close
    k15 = np.searchsorted(t[en], t, side="right") - 1          # last 15m bar whose last 5m bar index ≤ i
    atr_at = np.where(k15 >= 0, A15[np.maximum(k15, 0)], np.nan)
    c15_at = np.where(k15 >= 0, C15[np.maximum(k15, 0)], np.nan)
    # 5m RSI14X
    r5 = TB.rsi(c, 14); sig["RSI5"] = np.nan_to_num((TB.sh(r5) > 70) & (r5 <= 70)).astype(bool)
    # 1h RSI14X
    s1, e1, _, _, _, C1, _ = agg(t, o, h, l, c, v, H1)
    r1 = TB.rsi(C1, 14); x1 = np.nan_to_num((TB.sh(r1) > 70) & (r1 <= 70)).astype(bool) & ((e1 - s1) == 11)
    sig["RSI1H"][e1] = x1
    is1h = np.zeros(N, bool); is1h[e1] = True
    # ── Frenzy windows
    ep_of = np.full(N, -1, np.int32); ep_on = np.zeros(N, np.int64); ep_u = np.full(N, "", object)
    eps = eps.sort_values("on_ms").reset_index(drop=True)
    anyF = np.zeros(N, bool)
    eids = []
    for k, e in eps.iterrows():
        on = int(np.searchsorted(t, e.on_ms))
        if on >= N or t[on] != e.on_ms:
            continue
        b = min(N, int(np.searchsorted(t, e.spike_ms + WIN - BAR, side="right")))
        ep_of[on:b] = len(eids); ep_on[on:b] = on; anyF[on:b] = True
        eids.append((e.u, int(e.on_ms)))
    # vol_decay since the episode on-bar (as of bar i's close)
    vd = np.full(N, np.nan)
    if eids:
        idx = np.nonzero(ep_of >= 0)[0]
        grp = np.r_[0, np.nonzero(np.diff(ep_of[idx]) != 0)[0] + 1, len(idx)]
        for a_, b_ in zip(grp[:-1], grp[1:]):
            ii = idx[a_:b_]; on = ep_on[ii[0]]
            seg = r1n[on:ii[-1] + 1]; pk = np.fmax.accumulate(seg)
            vd[ii] = roll1h[ii] / pk[ii - on]
    # vd24 (X4): trailing 1h ÷ max trailing 1h over the previous 24 h incl. now
    pk24 = pd.Series(r1n).rolling(288, min_periods=1).max().values
    with np.errstate(divide="ignore", invalid="ignore"):
        vd24 = roll1h / pk24
    c288 = np.r_[np.full(288, np.nan), c[:-288]]
    hi24 = pd.Series(h).rolling(288, min_periods=288).max().values; lo24 = pd.Series(l).rolling(288, min_periods=288).min().values
    with np.errstate(divide="ignore", invalid="ignore"):
        mover = ((np.abs(c / c288 - 1) * 100 >= 15) | ((hi24 / lo24 - 1) * 100 >= 20)) & (vol24 >= 20e6)
    x4ok = mover & ~anyF & (p not in skip) & p.isascii() & (p not in ("BTCUSDT", "ETHUSDT"))
    # X4 segments: runs of mover bars merged across gaps ≤ 24 h
    seg_id = np.full(N, -1, np.int32)
    mi = np.nonzero(x4ok)[0]
    if len(mi):
        brk = np.r_[True, np.diff(mi) > 288]; seg_id[mi] = np.cumsum(brk) - 1
    base_ok = np.isfinite(atr_at) & (atr_at > 0) & (np.arange(N) + 1 < N) & (np.arange(N) >= 300)
    fr = base_ok & (ep_of >= 0) & (vol24 >= 20e6) & (np.nan_to_num(vd, nan=-1) >= VD_MIN_F)
    x4 = base_ok & is15 & x4ok & (np.nan_to_num(vd24, nan=-1) >= VD_X4)
    sel = np.nonzero(fr | x4)[0]
    if not len(sel):
        return p, 0
    J = sel + 1
    atrp = (atr_at[sel] / o[J] * 100)
    out = {"pair": np.full(len(sel), p), "entry_ms": t[J].astype("int64"), "fr": fr[sel], "x4": x4[sel], "is15": is15[sel], "is1h": is1h[sel],
           "u": np.where(fr[sel], ep_u_arr(eids, ep_of[sel]), "X4"), "ep": np.where(fr[sel], ep_of[sel], -1).astype(np.int32),
           "seg": np.where(x4[sel], seg_id[sel], -1).astype(np.int32), "vd": vd[sel].astype(np.float32), "vd24": vd24[sel].astype(np.float32),
           "atrp": atrp.astype(np.float32), "vol24": vol24[sel].astype(np.float32)}
    for k in SIG:
        out[k] = sig[k][sel]
    q5 = pd.Series(qv).rolling(12, min_periods=6).median().values
    out["q5"] = q5[sel].astype(np.float32)
    # spot_share_chg (TA_WL definition; REF only)
    T = t[J].astype("int64"); j = sel
    ssc = np.full(len(sel), np.nan)
    sp, sp_ok, tried = spot_frame(p)
    if sp is not None and len(sp):
        mo = pd.to_datetime(T, unit="ms").strftime("%Y-%m").values
        has = np.array([m in sp_ok for m in mo])
        st_ = sp.open_time.values.astype("int64"); sq = np.r_[0.0, np.cumsum(sp.qv.values.astype(float))]
        cq = np.r_[0.0, np.cumsum(qv)]
        a1 = np.searchsorted(st_, T - H1); b1 = np.searchsorted(st_, T - M15, side="right"); a24 = np.searchsorted(st_, T - 25 * H1)
        s1 = sq[b1] - sq[a1]; s24 = sq[a1] - sq[a24]
        jp = np.searchsorted(t, T - 25 * H1)
        p1 = roll1h[j]; p24 = cq[np.maximum(j - 11, 0)] - cq[np.maximum(jp, 0)]
        ok = has & ((b1 - a1) >= 3) & ((a1 - a24) >= 80)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh1 = s1 / (s1 + p1); sh24 = s24 / (s24 + p24)
        ssc[ok] = (sh1 - sh24)[ok]
    out["ssc"] = ssc.astype(np.float32)
    # event path (gross short P&L vs entry open at fixed offsets; NaN outside data)
    for off in EV_OFF:
        jj = J + off; okk = (jj >= 0) & (jj < N)
        vv = np.full(len(J), np.nan); vv[okk] = -(c[jj[okk]] / o[J[okk]] - 1) * 100
        out[f"ev{off}"] = vv.astype(np.float32)
    ex = exits(o, h, l, c, J, atrp, tick)
    for k, vv in ex.items():
        out[k] = vv
    for en_ in EXITS:
        out[f"{en_}_x"] = (T + (out[f"{en_}_c"].astype(np.int64) + 1) * BAR)
        del out[f"{en_}_c"]
    df = pd.DataFrame(out)
    df.to_pickle(fo)
    return p, len(df)


def ep_u_arr(eids, epi):
    us = np.array([u for u, _ in eids] + [""], dtype=object)
    return us[np.where(epi >= 0, epi, len(eids))]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=""); ap.add_argument("--procs", type=int, default=7)
    A = ap.parse_args()
    import study_frenzy_swing_short_walk as W
    skip = W.blacklist()
    E = episodes()
    us = open(os.path.join(R, "reports", "LEAD1_VWAP_SHORT_data", "symbols.txt")).read().split()
    have = set()
    for d in TB.K5M_SRC:
        have |= {os.path.basename(f)[:-4] for f in glob.glob(os.path.join(d, "*USDT.csv"))}
    pairs = sorted(set(us) | have) if not A.pairs else A.pairs.split(",")
    pairs = [q for q in pairs if q in have]
    jobs = [(q, E[E.pair == q], skip) for q in pairs]
    jobs.sort(key=lambda j: -len(j[1]))
    t0 = time.time(); tot = 0
    with mp.get_context("fork").Pool(A.procs) as pool:
        for k, (q, n) in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            tot += max(n, 0)
            if k % 50 == 0:
                print(time.strftime("%H:%M:%S"), k, len(jobs), q, n, tot, f"{time.time() - t0:.0f}s", flush=True)
    print("done", tot, flush=True)
