#!/usr/bin/env python3
"""📊 Oct-8 research (YR5 halves at today's stack) — FRENZY_WILLY re-priced under its CURRENT live design on real ticks.

Never backtested as shipped (DECISION_LOG 251/255). This is a STUDY-COHORT ESTIMATE, not an engine replay:
  triggers = the vol/mcap study's WILLY A (first-flag events, engine-reachable, flag_math/ev2.pkl) and WILLY B (fresh-ON bars not taken by
             today's FRENZY_LONG / WIDE, bearish-day block on — LITE not modelled) rows of reports/FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv
             (pair, t0 = trigger 5m close ms, R = vol24h / mcap estimated at the trigger).
  turnover = R < frenzy_willy_max_vol_mcap_ratio 1.0; R missing → refused (live is fail-closed: TURNOVER_UNREAD).
  entry    = the FIRST closed RED 5m bar (close < open, k5m_full klines) whose close is in [t0, t0 + 60 min]; fill = first aggTrade print
             ≥ that close + 8 s, × (1 + 0.035 % slip); dislocation guard |print / last print before the close − 1| > 1 % → that bar is
             skipped (the pending stays armed, next red bar tried).
  exit     = net = gross − 0.045 entry taker − 0.045 exit taker; first print with net ≥ +1.0 → books +1.0 (no TP slip, the study's
             convention) · NO stop · 120 min after entry → last print's net − 0.05 slip.
  sequencing = ONE WILLY at a time (frenzy_willy_max_slots 1 + the global hold also blocks WILLY): a red bar whose entry would land while
             another WILLY is open is skipped; the pending stays armed until its 60-min expiry. The global hold's effect on OTHER sleeves
             is NOT modelled.
Ticks: reports/backtest_cache/ticks_q then ticks (local only, no network).
Usage: venv/bin/python scripts/study_yr5_halves_willy.py <out.csv>"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "reports", "backtest_cache")
SRC = os.path.join(ROOT, "reports", "FRENZY_VOL_MCAP_STUDY_2026-10-08_fills.csv")
BAR, DELAY, SLIP_IN, TAKER, SLIP_OUT, TP, HOLD, WAIT, DISLOC, RMAX = 300_000, 8_000, 0.035, 0.045, 0.05, 1.0, 7_200_000, 3_600_000, 1.0, 1.0


def day(pair, d):
    for b in ("ticks_q", "ticks"):
        f = f"{C}/{b}/{pair}/{d}.npz"
        if os.path.exists(f):
            z = np.load(f); t = z["t"]; o = np.argsort(t, kind="stable"); return t[o], z["p"][o].astype(np.float64)
    return None


def ticks(pair, a, b, cache):
    ts, ps = [], []
    for d in pd.date_range(pd.Timestamp(a, unit="ms").normalize(), pd.Timestamp(b, unit="ms").normalize()):
        k = (pair, f"{d:%Y-%m-%d}")
        if k not in cache:
            cache[k] = day(*k)
        if cache[k] is None:
            return None, None
        ts.append(cache[k][0]); ps.append(cache[k][1])
    t = np.concatenate(ts); p = np.concatenate(ps); i, j = np.searchsorted(t, a), np.searchsorted(t, b)
    return t[i:j], p[i:j]


_K5 = {}


def k5(pair):
    if pair not in _K5:
        f = f"{C}/k5m_full/{pair}.csv"
        _K5[pair] = pd.read_csv(f).drop_duplicates("open_time").set_index("open_time") if os.path.exists(f) else None
        if len(_K5) > 6:
            _K5.pop(next(iter(_K5)))
    return _K5[pair]


def red_bars(pair, t0):
    """close times (ms) of closed red 5m bars with close in [t0, t0 + WAIT]; None if klines are missing."""
    K = k5(pair)
    if K is None:
        return None
    out = []
    for c in range(int(t0), int(t0) + WAIT + 1, BAR):
        o = c - BAR
        if o not in K.index:
            return None if not out else out
        r = K.loc[o]
        if float(r.c) < float(r.o):
            out.append(c)
    return out


def walk(pair, t0, cache):
    """every candidate entry (red bar) for one trigger, priced; the sequencer later picks the first free one."""
    rb = red_bars(pair, t0)
    if rb is None:
        return [dict(st="no_klines")]
    if not rb:
        return [dict(st="red_expired")]
    t, p = ticks(pair, int(t0) - 120_000, rb[-1] + DELAY + HOLD + 60_000, cache)
    if t is None:
        return [dict(st="no_ticks")]
    out = []
    for c in rb:
        pre = t < c
        if not pre.any():
            out.append(dict(st="noprint_pre", red_close=c)); continue
        ref = p[pre][-1]; k = np.searchsorted(t, c + DELAY)
        if k >= len(t):
            out.append(dict(st="noprint_post", red_close=c)); continue
        raw = p[k]; te = int(t[k])
        if abs(raw / ref - 1) * 100 > DISLOC:
            out.append(dict(st="disloc", red_close=c, te=te)); continue
        e = raw * (1 + SLIP_IN / 100)
        end = np.searchsorted(t, te + HOLD, side="right")
        tt, pp = t[k + 1:end], p[k + 1:end]
        net = (pp / e - 1) * 100 - 2 * TAKER
        itp = np.flatnonzero(net >= TP)
        mn = float(net.min()) if len(net) else np.nan
        if len(itp):
            a = itp[0]
            out.append(dict(st="ok", red_close=c, te=te, xt=int(tt[a]), why="tp", pct=TP, worst=float(net[:a + 1].min()))); continue
        last = net[-1] if len(net) else -2 * TAKER
        out.append(dict(st="ok", red_close=c, te=te, xt=te + HOLD, why="time_cap", pct=float(last - SLIP_OUT), worst=mn))
    return out


def work(items):
    cache, out = {}, []
    for i, pair, t0 in items:
        try:
            r = walk(pair, int(t0), cache)
        except Exception as ex:  # noqa
            r = [dict(st=f"err:{ex}")]
        out.append((i, r))
        if len(cache) > 8:
            cache.clear()
    return out


def sequence(E, cands):
    """one WILLY at a time, chronological by entry; a pending tries its red bars in order and takes the first that lands free."""
    rows = []
    E = E.assign(_first=[min([c.get("te", 9e18) for c in cands[i] if c.get("st") == "ok"] or [9e18]) for i in E.index]).sort_values("_first")
    busy_until = -1
    for i, r in E.iterrows():
        cs = [c for c in cands[i] if c.get("st") == "ok"]
        if not cs:
            st = cands[i][0].get("st") if cands[i] else "none"
            if any(c.get("st") == "disloc" for c in cands[i]):
                st = "disloc_all"
            rows.append({**r.to_dict(), "st": st}); continue
        took = None; skipped = 0
        for c in cs:
            if c["te"] > busy_until:
                took = c; break
            skipped += 1
        if took is None:
            rows.append({**r.to_dict(), "st": "slot_busy"}); continue
        busy_until = took["xt"]
        rows.append({**r.to_dict(), **took, "wait_bars": int((took["red_close"] - int(r.t0)) // BAR), "slot_skips": skipped})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    D = pd.read_csv(SRC)
    E = D[D.cohort.isin(["willyA", "willyB"])][["cohort", "pair", "t0", "day", "R", "pct", "why"]].rename(
        columns={"pct": "study_pct_old_exit", "why": "study_why_old_exit"}).reset_index(drop=True)
    E["trigger"] = E.cohort.str[-1]
    E["turnover_ok"] = E.R < RMAX               # NaN → False (fail-closed)
    G = E[E.turnover_ok].copy()
    pairs = sorted(G.pair.unique()); g = {p: k % 8 for k, p in enumerate(pairs)}
    items = list(zip(G.index, G.pair, G.t0))
    chunks = [[it for it in items if g[it[1]] == k] for k in range(8)]
    with Pool(8) as P:
        cands = dict(sum(P.map(work, chunks), []))
    S = sequence(G, cands)
    R = pd.concat([S, E[~E.turnover_ok].assign(st=np.where(E[~E.turnover_ok].R.isna(), "turnover_unread", "turnover_blocked"))],
                  ignore_index=True)
    R.to_csv(sys.argv[1], index=False)
    print(R.groupby("trigger").st.value_counts().to_string())
    ok = R[R.st == "ok"]
    for k, x in ok.groupby("trigger"):
        print(f"{k}: priced {len(x)} · WR {(x.pct > 0).mean() * 100:.1f}% · avg {x.pct.mean():+.3f} · worst {x.pct.min():+.2f} · "
              f"{x.why.value_counts().to_dict()} · wait bars median {x.wait_bars.median():.0f}")
    print(f"ALL: {len(ok)} · WR {(ok.pct > 0).mean() * 100:.1f}% · avg {ok.pct.mean():+.3f}")
