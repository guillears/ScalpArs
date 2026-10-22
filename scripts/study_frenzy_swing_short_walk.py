#!/usr/bin/env python3
"""FRENZY swing-short study (2026-10-20) — EPISODE step: the live engine's FRENZY episodes (services/frenzy.py frenzy_walk / frenzy_flagged)
re-implemented bar-by-bar over full history, every USDT perp, 2025-01-01 → 2026-10-20 (pre-registration:
reports/FRENZY_SWING_SHORT_2026-10-20_PREREG.md).

Data per pair (one continuous 5m grid; gaps filled flat with zero volume): reports/backtest_cache_2025/k5m_full (2024-06 → 2026-01-12, incl.
delisted) ∪ reports/backtest_cache/k5m_full (→ 2026-10-12) ∪ reports/FRENZY_SWING_SHORT_2026-10-20_data/k5m (→ 2026-10-20).
Engine parity: quote volume = volume × typical price · normal hour = median of the last ≤ 720 complete 1h bars that ended ≥ 24 h before the
bar (≥ 240 needed; services.frenzy.normal_hour_usd) · spike / VWAP / 12-close streak / state / 24 h end / verified / 96 h exactly as frenzy_walk.
Known approximation (validated in the analysis step against the real frenzy_walk on sampled bars): the engine evaluates on a 1499-bar
window with one normal hour per evaluation; here each bar uses its own as-of normal hour and the chain is global.

Output: reports/FRENZY_SWING_SHORT_2026-10-20_data/episodes.pkl — list of dicts, one per verified episode that was ever on the shortlist
while flagged, with the segment arrays [on − 300 … off + 600] (t, o, h, l, c, vwap, volx, vol24, seen, flagged).
Usage: venv/bin/python scripts/study_frenzy_swing_short_walk.py [--pairs A,B] [--procs 6]
"""
import argparse, glob, os, pickle, sys
import multiprocessing as mp
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R)
D = os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20_data")
SRC = [os.path.join(R, "reports", "backtest_cache_2025", "k5m_full"), os.path.join(R, "reports", "backtest_cache", "k5m_full"), os.path.join(D, "k5m")]
BAR, HR = 300_000, 3_600_000
T0 = pd.Timestamp("2024-11-15").value // 10**6        # history kept (norm needs ≥ 240 h + 24 h before 2025-01-01)
RET, VMULT, HMIN, SVOL, MINB, MAXH = 5.0, 20.0, 2e6, 100.0, 24, 96.0
CHG, VMIN = 15.0, 20e6
PRE, POST = 300, 600


def blacklist():
    import config
    tc, th = config.trading_config, config.trading_config.thresholds
    s = set()
    for src in (getattr(tc, "pair_blacklist", ""), getattr(tc, "no_trade_pairs", ""), getattr(th, "frenzy_pair_blacklist", "")):
        s |= {x.strip().upper() for x in str(src or "").split(",") if x.strip()}
    return s


def load(pair):
    parts = []
    for i, d in enumerate(SRC):
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f):
            x = pd.read_csv(f)
            if i == 0:
                x = x[x.open_time < pd.Timestamp("2026-01-10").value // 10**6]
            parts.append(x)
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time")
    d = d[d.open_time >= T0]
    if len(d) < 3000:
        return None
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, BAR, dtype="int64")
    d = d.set_index("open_time").reindex(grid)
    real = d.c.notna().values
    c = d.c.ffill().values; o = d.o.fillna(pd.Series(c, index=d.index)).values
    h = d.h.fillna(pd.Series(c, index=d.index)).values; l = d.l.fillna(pd.Series(c, index=d.index)).values
    v = d.vol.fillna(0).values; qv = d.qvol.fillna(0).values
    return grid, o.astype(float), h.astype(float), l.astype(float), c.astype(float), v.astype(float), qv.astype(float), real


def norm_per_bar(t, h, l, c, v, real):
    """services.frenzy.normal_hour_usd as of every bar (last_closed_ms = the bar's open)."""
    hr = t // HR * HR
    df = pd.DataFrame({"hr": hr, "h": h, "l": l, "c": c, "v": v, "real": real})
    g = df.groupby("hr").agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), n=("real", "sum"))
    g = g[g.n == 12]
    hq = g.v * (g.h + g.l + g.c) / 3.0
    med = hq.rolling(720, min_periods=240).median()
    ho = g.index.values.astype("int64")
    k = np.searchsorted(ho, t - 25 * HR, side="right") - 1       # last hour with open + 1 h ≤ t − 24 h
    out = np.full(len(t), np.nan)
    ok = k >= 0
    out[ok] = med.values[k[ok]]
    out[~(out > 0)] = np.nan
    return out


def walk(pair, skip):
    x = load(pair)
    if x is None:
        return []
    t, o, h, l, c, v, qv, real = x
    n = len(t)
    q = v * (h + l + c) / 3.0
    q1h = pd.Series(q).rolling(12).sum().values
    nh = norm_per_bar(t, h, l, c, v, real)
    volx = q1h / nh
    c6 = np.r_[np.full(6, np.nan), c[:-6]]
    with np.errstate(invalid="ignore", divide="ignore"):
        lead = np.nonzero((c6 > 0) & ((c / c6 - 1) * 100 >= RET) & (volx >= VMULT) & (q1h >= HMIN))[0]
        svb = np.nan_to_num(volx) >= SVOL
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    hi24 = pd.Series(h).rolling(288, min_periods=288).max().values; lo24 = pd.Series(l).rolling(288, min_periods=288).min().values
    c288 = np.r_[np.full(288, np.nan), c[:-288]]
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.maximum(np.abs(c / c288 - 1) * 100, (hi24 / lo24 - 1) * 100)
    short_ok = (move >= CHG) & (vol24 >= VMIN) & (pair.upper() not in skip) & pair.isascii()
    svc = np.r_[0, np.cumsum(svb)]
    eps = []; i_end = 0; prev_ver = False; prev_on = -10**9
    for on in lead:
        if on < i_end or on < 11:
            continue
        base_ver = bool(on >= PRE and svc[on] - svc[on - PRE] == 0)
        # engine: 'chained' only while the previous (verified) episode is still inside the 1499-bar window → per-bar below
        chain_until = (prev_on + 1498) if prev_ver else -1
        verified = bool(base_ver or chain_until > on)
        # VWAP / streak / state, extended in chunks until the episode ends
        end = None; L = 3000
        while True:
            j1 = min(n, on + L)
            tp = (h[on:j1] + l[on:j1] + c[on:j1]) / 3.0
            pv = np.cumsum(tp * q[on:j1]); qq = np.cumsum(q[on:j1])
            vw = np.where(qq > 0, pv / np.where(qq > 0, qq, 1), c[on:j1])
            ab = c[on:j1] >= vw
            idx = np.arange(j1 - on)
            lastb = np.maximum.accumulate(np.where(~ab, idx, -1))
            streak = idx - lastb
            bel = ~ab; lasta = np.maximum.accumulate(np.where(ab, idx, -1)); bstreak = idx - lasta
            state = (streak >= 12) & svb[on:j1] & (idx >= MINB)
            last_state = np.maximum.accumulate(np.where(state, idx, 0))
            prev_ls = np.r_[0, last_state[:-1]]
            gap = (idx - prev_ls) > 288
            gap[0] = False
            e = np.nonzero(gap)[0]
            if len(e):
                end = on + int(e[0]); break
            if j1 >= n:
                break
            L *= 2
        stop = end if end is not None else n
        k = stop - on
        vw, streak, bstreak, state = vw[:k], streak[:k], bstreak[:k], state[:k]
        jj = on + np.arange(k)
        flag = (base_ver | (jj <= chain_until)) & (np.arange(k) <= int(MAXH * 12))
        seen_c = np.maximum.accumulate(short_ok[on:stop] & flag)
        ever_seen = bool(seen_c.any())
        if end is not None:
            i_end = end; prev_ver = verified; prev_on = on
        if not (verified and ever_seen):
            if end is None:
                break
            continue
        fl_idx = np.nonzero(flag)[0]
        off = on + int(fl_idx[-1]) + 1                  # first unflagged bar (end, the 96 h cap, or the chain leaving the window)
        new_ep_at_end = bool(end is not None and off == end and np.any(lead == end))
        a, b = max(0, on - PRE), min(n, off + POST + 1)
        seg = slice(a, b)
        vwap_seg = np.full(b - a, np.nan); vwap_seg[on - a: on - a + min(k, b - on)] = vw[: b - on]
        bst = np.zeros(b - a, dtype=np.int32); bst[on - a: on - a + min(k, b - on)] = bstreak[: b - on]
        st = np.zeros(b - a, dtype=bool); st[on - a: on - a + min(k, b - on)] = state[: b - on]
        fl = np.zeros(b - a, dtype=bool); fl[on - a: off - a] = True
        se = np.zeros(b - a, dtype=bool); se[on - a: on - a + min(k, b - on)] = seen_c[: b - on]
        eps.append(dict(pair=pair, on_i=on - a, off_i=off - a, end_i=(end - a) if end is not None else None, alive=end is None,
                        spike_ts=int(t[on]) + BAR, base=float(c[on - 6]), new_ep_at_end=new_ep_at_end,
                        t=t[seg].copy(), o=o[seg].astype(np.float64), h=h[seg].astype(np.float64), l=l[seg].astype(np.float64),
                        c=c[seg].astype(np.float64), vwap=vwap_seg, volx=volx[seg].astype(np.float32), vol24=vol24[seg].astype(np.float32),
                        bstreak=bst, state=st, flag=fl, seen=se, real=real[seg].copy()))
        if end is None:
            break
    return eps


def _one(args):
    pair, skip = args
    try:
        return walk(pair, skip)
    except Exception as e:
        print("ERR", pair, e, flush=True)
        return []


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", default=""); ap.add_argument("--procs", type=int, default=6)
    A = ap.parse_args()
    skip = blacklist()
    pairs = sorted({os.path.basename(f)[:-4] for d in SRC for f in glob.glob(os.path.join(d, "*USDT.csv"))})
    if A.pairs:
        pairs = A.pairs.split(",")
    pairs = [p for p in pairs if p not in ("BTCUSDT", "ETHUSDT")]
    out = []
    with mp.Pool(A.procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_one, [(p, skip) for p in pairs], chunksize=4)):
            out += r
            if k % 100 == 0:
                print(k, len(pairs), len(out), flush=True)
    fn = os.path.join(D, "episodes.pkl" if not A.pairs else "episodes_sample.pkl")
    pickle.dump(out, open(fn, "wb"))
    print("episodes", len(out), "pairs", len({e['pair'] for e in out}), "->", fn)
