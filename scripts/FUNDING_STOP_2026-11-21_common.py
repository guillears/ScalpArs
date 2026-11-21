"""FUNDING_STOP_2026-11-21 — shared walker + statistics (research only; no bot code / config touched).

Question: should the stop of the short sleeves (FLATBASE / EXHAUSTION / DISTRIBUTION) count accrued FUNDING?
  arm A  = today: stop when the PRICE move from the fill ≤ −SL (services.flatbase / exhaustion / distribution *_exit_for: price only,
           no fees, no funding).
  arm B  = net stop: stop when price move % + accrued funding % of notional (signed: a short RECEIVES a positive rate) ≤ −SL. Funding
           accrues at each real settlement while open; the bar that contains / follows a settlement carries the new total (settlements sit
           on bar opens: hour marks for 1m and 5m bars). The stop line in bar j is fill × (1 + (SL + F_j) / 100); a bar whose OPEN is already
           past the moved line exits at the open (gap-aware, like arm A's max(line, open)).
  arm Bp = the paid-only variant: F_j replaced by min(F_j, 0) (funding received never widens the stop) — sensitivity.
Walk (both arms identical except the stop line): stop checked first on the bar HIGH; maker TP on the bar LOW, filled at the line only when
the low trades through it by `through` % (0.05 primary; parity mode = the cohort's own rule); time exit at the close of the last bar inside
the hold. Funding booked = Σ settlements in (fill, exit bar open] (the position is open at a settlement that sits on the exit bar's open),
each settlement = rate × price at that bar's open ÷ fill × 100 (mark-scaled; parity mode: rate × 100 like the cohort's own walker).
"""
import numpy as np
import pandas as pd

MIN = 60_000


def settle_on_bars(t, o, E, T_fill, fs_t, fs_r, nb, markscale=True):
    """→ per-bar cumulative funding % to the short (len(t)), funding counted at settlements s with T_fill < s, assigned to the first bar
    whose open ≥ floor(s to the bar)."""
    F = np.zeros(len(t))
    if fs_t is None or not len(fs_t) or not len(t):
        return F
    m = (fs_t > T_fill) & (fs_t <= t[-1] + nb * MIN)
    if not m.any():
        return F
    s = fs_t[m]; r = fs_r[m]
    sb = s - s % (nb * MIN)
    j = np.searchsorted(t, sb, "left")
    ok = j < len(t)
    j, r = j[ok], r[ok]
    px = o[j] / E if markscale else np.ones(len(j))
    add = np.zeros(len(t))
    np.add.at(add, j, r * px * 100.0)
    return np.cumsum(add)


def walk(t, o, h, l, c, i0, E, tp, sl, end_t, F, arm, slip_exit, through, entry_cost, taker, maker, tp_rule="through", tick=0.0):
    """one short from bar i0 (inclusive) to the last bar with t < end_t.
    F: per-bar cumulative funding % (settle_on_bars). arm: 'A' | 'B' | 'Bp'.
    tp_rule: 'through' (low ≤ T × (1 − through %), fill at T) · 'tick' (low ≤ T − tick, fill at T) · 'touch' (low ≤ T, fill min(T, open)).
    entry_cost: % charged at entry (taker fee [+ entry slip for backtest rows]); exit fees taker / maker are % of the EXIT notional.
    → dict(net, why, j, fund, gross)."""
    last = int(np.searchsorted(t, end_t, "left")) - 1
    if last < i0:
        return None
    sl_ = slice(i0, last + 1)
    hh, ll, oo = h[sl_], l[sl_], o[sl_]
    Fj = F[sl_]
    if arm == "A":
        adj = np.zeros(len(hh))
    elif arm == "B":
        adj = Fj
    else:
        adj = np.minimum(Fj, 0.0)
    line = E * (1 + (sl + adj) / 100.0)
    T = E * (1 - tp / 100.0)
    if tp_rule == "through":
        tl = ll <= T * (1 - through / 100.0)
    elif tp_rule == "tick":
        tl = ll <= T - tick
    else:
        tl = ll <= T
    hs = hh >= line
    ks = int(np.argmax(hs)) if hs.any() else len(hh)
    kt = int(np.argmax(tl)) if tl.any() else len(hh)
    if ks < len(hh) and ks <= kt:
        k = ks; X = max(line[k], oo[k]) * (1 + slip_exit / 100.0); why = "STOP"
        fee = taker * X / E
    elif kt < len(hh):
        k = kt; why = "TP"
        X = min(T, oo[k]) if tp_rule == "touch" else T
        fee = maker * X / E
    else:
        k = len(hh) - 1; why = "TIME" if t[last] + (t[1] - t[0] if len(t) > 1 else MIN) >= end_t - MIN else "OPEN"
        X = c[last] * (1 + slip_exit / 100.0)
        fee = taker * X / E
    gross = (1 - X / E) * 100.0
    fu = float(Fj[k])
    return dict(net=gross - entry_cost - fee + fu, why=why, j=i0 + k, fund=fu, gross=gross)


def classify(a, b):
    """change class of one fill: A vs B results (dicts)."""
    d = b["net"] - a["net"]
    if a["why"] == b["why"] and a["j"] == b["j"] and abs(d) < 1e-12:
        return "SAME"
    if a["why"] == "STOP" and b["why"] == "STOP" and b["j"] <= a["j"]:
        return "SAVED"          # A stopped anyway; B exits earlier (or same bar at a better line)
    if a["why"] in ("TP", "TIME", "OPEN") and b["why"] == "STOP":
        return "CUT"            # B stopped a fill A carried to TP / time
    if a["why"] == "STOP" and (b["why"] != "STOP" or b["j"] > a["j"]):
        return "WIDENED"        # received funding moved B's line out; A's stop bar passed
    return "OTHER"


def day_dcr(pct, close_day, d0, d1, x=0.61):
    s = pd.Series(np.asarray(pct, float) * x / 100).groupby(pd.to_datetime(pd.Series(list(close_day))).values).sum()
    days = pd.date_range(d0, d1, freq="D")
    r = s.reindex(days, fill_value=0.0).values
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min()
    return (eq[-1] ** (1 / len(r)) - 1) * 100, min(dd, 0) * 100


def cluster_boot(delta, cluster, B=4000, seed=7):
    d = np.asarray(delta, float)
    if len(d) < 2:
        return (d.mean() if len(d) else np.nan), np.nan, np.nan, np.nan
    u, inv = np.unique(np.asarray(cluster), return_inverse=True)
    sums = np.bincount(inv, d, len(u)); ns = np.bincount(inv, None, len(u))
    rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(u), (B, len(u)))
    m = sums[ix].sum(1) / ns[ix].sum(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), (m > 0).mean()


def loo(delta):
    d = np.asarray(delta, float)
    if len(d) < 2:
        return np.nan, np.nan
    m = (d.sum() - d) / (len(d) - 1)
    return m.min(), m.max()


def md(df, fl=2):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:+.{fl}f}" if abs(v) < 10000 else f"{v:,.0f}")
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def interval_h(fs_t, at_ms):
    """settlement interval (h) in force at `at_ms`: gap between the last two settlements ≤ at_ms (NaN if < 2)."""
    if fs_t is None or len(fs_t) < 2:
        return np.nan
    k = int(np.searchsorted(fs_t, at_ms, "right"))
    if k < 2:
        k = min(2, len(fs_t))
    g = (fs_t[k - 1] - fs_t[k - 2]) / 3_600_000
    return float(np.round(g))


def last_rate(fs_t, fs_r, at_ms):
    if fs_t is None or not len(fs_t):
        return np.nan
    k = int(np.searchsorted(fs_t, at_ms, "right"))
    return float(fs_r[k - 1]) if k else np.nan
