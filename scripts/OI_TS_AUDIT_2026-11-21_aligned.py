#!/usr/bin/env python3
"""OI_TS_AUDIT 2026-11-21 — shared ALIGNED reader for Binance data.binance.vision `daily/metrics` files (research only; proposal for a
repo-wide replacement of the ad-hoc `create_time ≤ T − 5 min` readers).

FACTS (re-measured 2026-11-21, reports/OI_TS_AUDIT_2026-11-21.md):
  * The live fapi /futures/data endpoints (openInterestHist, top/global long-short ratios) stamp each row with the SNAPSHOT time:
    implied price sumOpenInterestValue ÷ sumOpenInterest = the 5m close ENDING at `timestamp` (SOL / DOGE, today).
  * Vision `create_time` = snapshot time through 2024-04 and again from 2026-11-16; from 2024-05 to 2026-11-15 it is the snapshot time
    MINUS 5 min (95–100 % of files). The ratio columns ride on the same row (the API rows match the vision rows at the same shift).
  * So `create_time ≤ T − 5 min` read the snapshot AT T in 2024-05 … 2026-11-15 (one period fresher than live, which reads the API row
    ≤ T − 5 min = the snapshot at T − 5) and the snapshot at T − 5 elsewhere. A naive `create_time ≤ T` read the snapshot at T + 5 min
    in 2024-05 … 2026-11-15 → 5 minutes of look-ahead.

THIS READER: each (pair, UTC day) file's true offset is measured from the implied price vs the 5m closes (offset ∈ {−5, 0, +5, +10} min,
best median |error|, accepted < 0.15 % on ≥ 20 rows); a file that cannot be measured falls back to the calendar rule (+5 min for
2024-05-01 ≤ day < 2026-11-16, else 0) and is flagged. `snap` = create_time + offset. Picks use `snap ≤ T − lag`; lag = LIVE_LAG_MS
(5 min) reproduces what the live engine reads (API timestamp ≤ T − 5 min).
"""
import numpy as np
import pandas as pd

M5 = 300_000
LIVE_LAG_MS = 5 * 60_000
OFFSETS = (-M5, 0, M5, 2 * M5)
TOL = 0.0015
SHIFT_FROM, SHIFT_TO = pd.Timestamp("2024-05-01").value // 10**6, pd.Timestamp("2026-11-16").value // 10**6


def calendar_offset(day_ms):
    return M5 if SHIFT_FROM <= day_ms < SHIFT_TO else 0


def align(ct, oi_contracts, oi_value, close_by_end):
    """ct int64 ms (create_time, sorted) · OI contracts / value arrays · close_by_end: pd.Series(close, index = bar END ms).
    → (snap int64 ms, offset per row ms, measured bool per row). Pure."""
    ct = np.asarray(ct, dtype="int64")
    off = np.zeros(len(ct), dtype="int64"); meas = np.zeros(len(ct), dtype=bool)
    if not len(ct):
        return ct.copy(), off, meas
    with np.errstate(divide="ignore", invalid="ignore"):
        px = np.asarray(oi_value, float) / np.where(np.asarray(oi_contracts, float) > 0, np.asarray(oi_contracts, float), np.nan)
    day = ct // 86_400_000
    for d in np.unique(day):
        m = day == d
        best, bo = np.inf, None
        if close_by_end is not None and len(close_by_end):
            for o in OFFSETS:
                ref = close_by_end.reindex(ct[m] + o).values
                ok = np.isfinite(ref) & np.isfinite(px[m])
                if ok.sum() >= 20:
                    e = np.median(np.abs(px[m][ok] / ref[ok] - 1))
                    if e < best:
                        best, bo = e, o
        if bo is not None and best < TOL:
            off[m] = bo; meas[m] = True
        else:
            off[m] = calendar_offset(int(d) * 86_400_000)
    return ct + off, off, meas  # caller re-sorts by snap (frame() does)


def pick(snap, T, lag=LIVE_LAG_MS, max_age=30 * 60_000):
    """index of the latest row with snap ≤ T − lag and T − snap ≤ max_age (−1 when none). snap sorted ascending. Vectorised over T."""
    snap = np.asarray(snap, dtype="int64"); T = np.asarray(T, dtype="int64")
    k = np.searchsorted(snap, T - lag, side="right") - 1
    ok = (k >= 0)
    ok &= (T - snap[np.maximum(k, 0)]) <= max_age
    return np.where(ok, k, -1)


def pick_back(snap, kc, win, tol):
    """for a current row index kc (≥ 0): the latest row with snap ≤ snap[kc] − win and snap[kc] − snap[k] ≤ win + tol (−1 when none)."""
    snap = np.asarray(snap, dtype="int64"); kc = np.asarray(kc)
    tc = snap[np.maximum(kc, 0)]
    k = np.searchsorted(snap, tc - win, side="right") - 1
    ok = (kc >= 0) & (k >= 0) & ((tc - snap[np.maximum(k, 0)]) <= win + tol)
    return np.where(ok, k, -1)


def frame(me, k5_t_open, k5_close):
    """metrics DataFrame with `ct` (ms) + OI columns, and the pair's 5m bars (open ms, close) → me sorted by `snap` with snap / off / meas."""
    if me is None or not len(me):
        return me
    cl = pd.Series(np.asarray(k5_close, float), index=np.asarray(k5_t_open, dtype="int64") + M5) if k5_t_open is not None else None
    me = me.sort_values("ct").reset_index(drop=True)
    snap, off, meas = align(me.ct.values, me.sum_open_interest.values, me.sum_open_interest_value.values, cl)
    me = me.assign(snap=snap, off=off, meas=meas)
    return me.drop_duplicates("snap", keep="last").sort_values("snap").reset_index(drop=True)
