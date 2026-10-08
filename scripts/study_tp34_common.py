#!/usr/bin/env python3
"""🎯 Oct-8 study (operator: "+3 vs +4 tick check") — shared tick walker for the FRENZY family exit comparison.

Read-only over the aggTrades cache reports/backtest_cache/{ticks_q,ticks}/<PAIR>/<YYYY-MM-DD>.npz (t ms, p float32) — the same cache
scripts/frenzy_exit_ticks_botexact.py / frenzy_trail_v2.py / scout_frenzy_exits.py read. No network here.

Bot accounting (services.frenzy.frenzy_exit_for + the paper close): net P&L % at every print = (p/E − 1)·100 − 0.045 − 0.045·p/E
(taker 0.045 % on the entry notional and on the exit notional — reproduces live pnl_percentage to 1e-3), levels on that net P&L, the
position closes AT the print that crosses the line (a line set by prior prints never closes on the print that set it), 12 h cap at the
last print ≤ entry + 12 h. Ties: when the first stop print and the first TP print share the same millisecond → stop first (conservative).

Exits (all with the same 12 h cap):
  fixed +TP / −SL           first print net ≥ +TP or ≤ −SL
  lock +2@+3 trail 2        −3 until the prior-print peak ≥ +3, then line = max(+2, peak − 2)            (DECISION_LOG 205)
  old trail +5/1.5          −3 until the prior-print peak ≥ +5, then line = max(−3, peak − 1.5·(1 + peak/100))  (pre-196 live)
"""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "reports", "backtest_cache")
MIN, DAY, HOLD = 60_000, 86_400_000, 12 * 3600_000
FEE_SIDE = 0.045
EXITS = ["fix3", "fix4", "fix5", "fix6", "lock", "trail5"]
LABEL = {"fix3": "+3/−3 (live)", "fix4": "+4/−3", "fix5": "+5/−3", "fix6": "+6/−3", "lock": "lock +2@+3 trail 2", "trail5": "old trail +5/1.5"}
_DAYC = {}


def tick_day(pair, ds):
    k = (pair, ds)
    if k in _DAYC:
        return _DAYC[k]
    out = None
    for base in ("ticks_q", "ticks"):
        f = os.path.join(CACHE, base, pair, f"{ds}.npz")
        if os.path.exists(f):
            try:
                z = np.load(f); out = (z["t"].astype(np.int64), z["p"].astype(np.float64)); break
            except Exception:
                out = None
    if len(_DAYC) > 64:
        _DAYC.clear()
    _DAYC[k] = out
    return out


def has_day(pair, ds):
    return any(os.path.exists(os.path.join(CACHE, b, pair, f"{ds}.npz")) for b in ("ticks_q", "ticks"))


def days_for(t0, t1):
    return [pd.Timestamp(d, unit="ms").strftime("%Y-%m-%d") for d in range((t0 // DAY) * DAY, t1 + 1, DAY)]


def ticks(pair, t0, t1):
    """prints in [t0, t1] (sorted, stable) or None when any UTC day is missing."""
    ts, ps = [], []
    for ds in days_for(t0, t1):
        x = tick_day(pair, ds)
        if x is None:
            return None
        ts.append(x[0]); ps.append(x[1])
    t = np.concatenate(ts); p = np.concatenate(ps)
    if len(t) > 1 and np.any(np.diff(t) < 0):
        o = np.argsort(t, kind="stable"); t, p = t[o], p[o]
    m = (t >= t0) & (t <= t1)
    return t[m], p[m]


def net_path(p, E):
    r = p / E
    return (r - 1) * 100 - FEE_SIDE - FEE_SIDE * r


def _first(mask):
    i = np.flatnonzero(mask)
    return int(i[0]) if len(i) else None


def exit_fixed(t, net, tp, sl):
    i_s = _first(net <= -sl); i_t = _first(net >= tp)
    if i_s is None and i_t is None:
        return float(net[-1]), "CAP", int(t[-1])
    if i_t is None or (i_s is not None and (i_s < i_t or t[i_s] == t[i_t])):
        return float(net[i_s]), "SL", int(t[i_s])
    return float(net[i_t]), "TP", int(t[i_t])


def exit_line(t, net, line):
    i = _first(net <= line)
    if i is None:
        return float(net[-1]), "CAP", int(t[-1])
    return float(net[i]), ("SL" if line[i] < 0 else "TRAIL"), int(t[i])


def prior_peak(net):
    return np.concatenate([[-np.inf], np.maximum.accumulate(net)[:-1]])


def walk_all(t, net, which=EXITS):
    """t, net = prints from the entry print to entry + 12 h. → {exit: (pct, how, exit_ms)} + peak info."""
    out = {}
    pk = prior_peak(net)
    for x in which:
        if x.startswith("fix"):
            out[x] = exit_fixed(t, net, float(x[3:]), 3.0)
        elif x == "lock":
            out[x] = exit_line(t, net, np.where(pk >= 3.0, np.maximum(2.0, pk - 2.0), -3.0))
        elif x == "trail5":
            out[x] = exit_line(t, net, np.where(pk >= 5.0, np.maximum(-3.0, pk - 1.5 * (1 + pk / 100.0)), -3.0))
    i_s = _first(net <= -3.0)
    pre = net[: (i_s if i_s is not None else len(net))]
    out["peak_before_stop"] = float(pre.max()) if len(pre) else float("nan")      # max net before the first −3 print
    out["peak_12h"] = float(net.max())
    return out


def walk_entry(pair, sig_close_ms, lag_ms=8_000, slip_pct=0.10, which=EXITS, E_override=None, t_override=None):
    """live ruler: entry = first print ≥ signal close + lag, E = that print × (1 + slip) (the bot books the slipped fill price and puts
    its levels on it). E_override / t_override: a known fill (live master rows). → dict or None (missing ticks)."""
    t0 = int(t_override) if t_override is not None else int(sig_close_ms) + int(lag_ms)
    x = ticks(pair, t0, t0 + HOLD + 5 * MIN)
    if x is None or len(x[0]) < 5:
        return None
    t, p = x
    i0 = 0
    te = int(t[i0])
    if te - t0 > 10 * MIN:                      # no print for 10 min after the decision: unusable
        return None
    E = float(E_override) if E_override is not None else float(p[i0]) * (1 + slip_pct / 100.0)
    m = t <= te + HOLD
    t, p = t[m], p[m]
    net = net_path(p, E)
    r = walk_all(t, net, which)
    r.update(entry_ms=te, E=E, n_prints=len(t), cap_complete=bool(x[0][-1] >= te + HOLD - 2 * MIN))
    return r


def boot_day(v, d, B=4000, seed=7):
    """day-clustered bootstrap 95 % CI of a mean."""
    v = np.asarray(v, float); d = np.asarray(d)
    if len(v) < 3:
        return (np.nan, np.nan)
    g = pd.DataFrame({"v": v, "d": d}).groupby("d").v.agg(["sum", "count"])
    s, c, k = g["sum"].values, g["count"].values, len(g)
    i = np.random.default_rng(seed).integers(0, k, (B, k))
    return tuple(np.percentile(s[i].sum(1) / c[i].sum(1), [2.5, 97.5]))
