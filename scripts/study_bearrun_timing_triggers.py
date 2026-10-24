"""🐻 Bear-Run trigger-timing study (2026-10-22) — step 1 + 2: monitor readings, candidate trigger timelines, BTC drawdown
episodes and trigger lag. Pure BTC-tape mechanics (no trade outcome). Pre-registration:
reports/BEARRUN_TRIGGER_TIMING_2026-10-22_PREREG.md.

Readings replicate services/trading_engine.py _update_bullrun_monitor on closed 5m bars (r24 = close / close 287 bars ago,
below24 = % of the last 288 closes NOT above the 5m EMA20, eff24 = |net| / path over those 288 closes, r6 = close / close 71
bars ago, off24lo = close / min(low of the last 288 bars) − 1, e50h = 1 h EMA50 of closed 1 h bars) and the state machine
calls the REAL services.trading_engine._bearrun_next_state for T0 (parity) — variants use the same Schmitt structure with
their own leg.

Outputs (reports/backtest_cache/study_bearrun_timing/): readings.pkl, on_<VAR>.csv (ON stretches), windows_<VAR>.csv
(merged 180 min), episodes.csv, lag.csv; reports/BEARRUN_TRIGGER_TIMING_lag.csv.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "backtest_cache", "study_bearrun_timing")

VARIANTS = {
    # id: (leg, on, stay, eff_mode, off24lo_max)
    "T0":  ("r24", 4.0, 3.0, "eff24", 2.0),
    "T1a": ("dd24", 3.0, 2.0, "eff24", 2.0),
    "T1b": ("dd24", 3.5, 2.5, "eff24", 2.0),
    "T1c": ("dd24", 4.0, 3.0, "eff24", 2.0),
    "T1x": ("dd24", 3.5, 2.5, "effH", 2.0),
    "T2":  ("r12", 3.0, 2.0, "eff24", 2.0),
    "T3":  ("r6", 2.0, 1.0, "eff24", 2.0),
    "T4a": ("dd24", 3.5, 2.5, "eff24", 1.0),
    "T4b": ("dd24", 3.5, 2.5, "eff24", 0.5),
}
BELOW_ON, BELOW_OFF, EFF_ON, EFF_OFF, LATCH_R6 = 50.0, 47.0, 0.15, 0.15, 3.0
MERGE_MIN = 180


def readings():
    b = pd.read_csv(os.path.join(OUT, "btc_5m.csv"))
    R = compute_readings(b)
    R = R[R.t >= "2025-01-01"].reset_index(drop=True)
    R.to_pickle(os.path.join(OUT, "readings.pkl"))
    return R


def compute_readings(b):
    """the monitor readings on a 5m frame (open_time ms, o, h, l, c) — pure, no file I/O, every row kept (the warm-up rows carry NaN);
    readings() = this on the study's cached history + its 2025-01-01 cut + readings.pkl. Also used by scripts/scout_bearrun_r24.py
    (DECISION_LOG 369) on live public 5m klines."""
    b = b.copy()
    b["t"] = pd.to_datetime(b.open_time, unit="ms") + pd.Timedelta(minutes=5)     # reading available at the bar CLOSE
    c = b.c.values.astype(float); h = b.h.values.astype(float); lo = b.l.values.astype(float)
    n = len(c)
    ema = np.empty(n); e = c[0]; k = 2.0 / 21.0
    for i in range(n):
        e = c[i] * k + e * (1 - k); ema[i] = e
    above = (c > ema).astype(float)
    s = pd.Series
    below24 = 100.0 - 100.0 * s(above).rolling(288).mean().values
    r24 = (c / s(c).shift(287).values - 1) * 100
    r12 = (c / s(c).shift(143).values - 1) * 100
    r6 = (c / s(c).shift(71).values - 1) * 100
    path = s(np.abs(np.diff(c, prepend=c[0]))).rolling(287).sum().values
    eff24 = np.abs(c - s(c).shift(287).values) / path
    hi24 = s(h).rolling(288).max().values
    lo24 = s(lo).rolling(288).min().values
    dd24 = (c / hi24 - 1) * 100
    off24lo = (c / lo24 - 1) * 100
    # bars since the 24 h high bar + efficiency of the path from it (T1x)
    idx_hi = s(h).rolling(288).apply(lambda x: 287 - int(np.argmax(x)), raw=True).values
    absd = np.abs(np.diff(c, prepend=c[0])); cum = np.cumsum(absd)
    effH = np.zeros(n)
    for i in range(n):
        if np.isnan(idx_hi[i]):
            continue
        j = i - int(idx_hi[i])
        if i - j < 12:
            continue
        p = cum[i] - cum[j]
        effH[i] = (c[j] - c[i]) / p if p > 0 else 0.0
    # 1 h EMA50 on closed 1 h bars (engine: k1h[:-1], 150 bars, seeded at the first → converged on a long series)
    hb = b.set_index(pd.to_datetime(b.open_time, unit="ms")).c.resample("1h").last().dropna()
    he = hb.ewm(alpha=2.0 / 51.0, adjust=False).mean()
    he.index = he.index + pd.Timedelta(hours=1)           # usable once the hour closed
    R = pd.DataFrame({"t": b.t, "px": c, "hi": h, "lo": lo, "r24": r24, "r12": r12, "r6": r6, "below24": below24,
                      "eff24": eff24, "dd24": dd24, "off24lo": off24lo, "effH": effH, "hi24": hi24})
    e50 = he.reindex(R.t, method="ffill").values
    R["e50h"] = e50
    return R


def run_variant(R, vid, engine_kw=None):
    """→ bool ON state per reading. T0 = the REAL engine _bearrun_next_state; engine_kw (optional) = its threshold kwargs (r24_on,
    r24_off, below_on, ... — the live trading_config.json values, or a shadow such as DECISION_LOG 369's r24 2.5 / 1.5). None = the
    engine defaults (the study's T0, unchanged)."""
    from services.trading_engine import _bearrun_next_state
    leg, on_th, stay_th, effm, _ = VARIANTS[vid]
    on = False
    states = np.zeros(len(R), dtype=bool)
    L = R[leg].values; B = R.below24.values; E = R[effm].values if effm == "effH" else R.eff24.values
    r6 = R.r6.values; px = R.px.values; e50 = R.e50h.values
    for i in range(len(R)):
        if np.isnan(L[i]) or np.isnan(B[i]) or np.isnan(E[i]):
            on = False; continue
        if vid == "T0":
            on, _l = _bearrun_next_state(on, L[i], B[i], E[i], r6[i], px[i], None if np.isnan(e50[i]) else e50[i], **(engine_kw or {}))
        else:
            latched = (r6[i] >= LATCH_R6) or (not np.isnan(e50[i]) and px[i] > e50[i])
            if latched:
                on = False
            elif on:
                on = (L[i] <= -stay_th) and (B[i] >= BELOW_OFF) and (E[i] >= EFF_OFF)
            else:
                on = (L[i] <= -on_th) and (B[i] >= BELOW_ON) and (E[i] >= EFF_ON)
        states[i] = on
    return states


def stretches(R, st):
    t = R.t.values; out = []; s0 = None
    for i in range(len(st)):
        if st[i] and s0 is None:
            s0 = i
        if (not st[i]) and s0 is not None:
            out.append((t[s0], t[i])); s0 = None
    if s0 is not None:
        out.append((t[s0], t[-1] + np.timedelta64(5, "m")))
    S = pd.DataFrame(out, columns=["on", "off"])
    return S


def merge(S, gap_min=MERGE_MIN):
    rows = []
    for r in S.itertuples():
        if rows and (r.on - rows[-1][1]) <= pd.Timedelta(minutes=gap_min):
            rows[-1][1] = max(rows[-1][1], r.off); rows[-1][2] += 1
        else:
            rows.append([r.on, r.off, 1])
    return pd.DataFrame(rows, columns=["w_start", "w_end", "stretches"])


def episodes(R, th=3.0):
    t = R.t.values; px = R.px.values; hi = R.hi.values; lo = R.lo.values; dd = R.dd24.values
    hi24 = R.hi24.values
    out = []; i = 288; n = len(R); prev_low = -1
    while i < n:
        if not (dd[i] <= -th):
            i += 1; continue
        # high bar = the 24 h max high bar at the crossing; a NEW episode needs a NEW high (after the previous episode's low)
        w = hi[max(0, i - 287): i + 1]; jh = max(0, i - 287) + int(np.argmax(w)); H = hi[jh]
        if jh <= prev_low:
            i += 1; continue
        run_lo = lo[jh]; jl = jh; k = jh + 1; end = None
        lim = jh + 72 * 12
        while k < n:
            if lo[k] < run_lo:
                run_lo = lo[k]; jl = k
            if k >= i and (px[k] >= run_lo * 1.03 or px[k] > H or k >= lim):   # end only judged after the −3 % crossing
                end = k; break
            k += 1
        if end is None:
            end = n - 1
        out.append(dict(ep_high_t=t[jh], ep_high=H, cross_t=t[i], ep_low_t=t[jl], ep_low=run_lo, ep_end_t=t[end],
                        move_pct=(run_lo / H - 1) * 100, i_high=jh, i_low=jl, i_end=end))
        i = max(end, i) + 1; prev_low = jl
    E = pd.DataFrame(out)
    E["ep"] = np.arange(len(E))
    # half-move time
    half = []
    for r in E.itertuples():
        lvl = r.ep_high - 0.5 * (r.ep_high - r.ep_low)
        seg = lo[r.i_high: r.i_low + 1]
        jj = r.i_high + int(np.argmax(seg <= lvl)) if (seg <= lvl).any() else r.i_low
        half.append(t[jj])
    E["half_t"] = half
    return E


def lag(R, E, states):
    t = R.t.values; px = R.px.values; rows = []
    for r in E.itertuples():
        lo_i, hi_i = r.i_high, min(len(R) - 1, r.i_low + 288)
        for vid, st in states.items():
            seg = st[lo_i: hi_i + 1]
            # ON already at the high (carried from an earlier move) counts as ON at the high
            if seg.any():
                j = lo_i + int(np.argmax(seg))
                done = (r.ep_high - px[j]) / (r.ep_high - r.ep_low) * 100 if r.ep_high > r.ep_low else np.nan
                rows.append(dict(ep=r.ep, var=vid, fired=True, on_t=t[j],
                                 min_after_high=(t[j] - r.ep_high_t) / np.timedelta64(1, "m"),
                                 min_after_half=(t[j] - r.half_t) / np.timedelta64(1, "m"),
                                 min_after_low=(t[j] - r.ep_low_t) / np.timedelta64(1, "m"),
                                 pct_done=done, late=bool(t[j] >= r.ep_low_t), r24_at_on=R.r24.values[j], dd24_at_on=R.dd24.values[j],
                                 off24lo_at_on=R.off24lo.values[j]))
            else:
                rows.append(dict(ep=r.ep, var=vid, fired=False))
    return pd.DataFrame(rows)


def artefact(R, E):
    """at the bar where dd24 first ≤ −4 inside the episode: r24 there, gap r24 − dd24, T0 ON already?"""
    rows = []
    for r in E.itertuples():
        seg = R.dd24.values[r.i_high: r.i_end + 1]
        if not (seg <= -4).any():
            rows.append(dict(ep=r.ep, reached4=False)); continue
        j = r.i_high + int(np.argmax(seg <= -4))
        r24 = R.r24.values[j]
        prior = R.px.values[j - 287]          # the 24 h-ago comparator
        prior_hi = R.hi24.values[j]
        rows.append(dict(ep=r.ep, reached4=True, t4=R.t.values[j], r24_at_dd4=r24, gap=r24 - R.dd24.values[j],
                         comparator_below_hi_pct=(prior / prior_hi - 1) * 100))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    R = readings()
    states = {v: run_variant(R, v) for v in VARIANTS}
    allw = []
    for v, st in states.items():
        S = stretches(R, st); W = merge(S)
        S.to_csv(os.path.join(OUT, f"on_{v}.csv"), index=False); W.to_csv(os.path.join(OUT, f"windows_{v}.csv"), index=False)
        hrs = (S.off - S.on).dt.total_seconds().sum() / 3600
        print(f"{v:4s} stretches {len(S):4d} windows {len(W):3d} ON {hrs:7.1f} h")
    pd.DataFrame({v: st for v, st in states.items()}).assign(t=R.t).to_pickle(os.path.join(OUT, "states.pkl"))
    E = episodes(R)
    E.to_csv(os.path.join(OUT, "episodes.csv"), index=False)
    Lg = lag(R, E, states)
    Lg.to_csv(os.path.join(ROOT, "reports", "BEARRUN_TRIGGER_TIMING_lag.csv"), index=False)
    A = artefact(R, E)
    A.to_csv(os.path.join(OUT, "artefact.csv"), index=False)
    print("episodes", len(E), "median move", round(E.move_pct.median(), 2))
