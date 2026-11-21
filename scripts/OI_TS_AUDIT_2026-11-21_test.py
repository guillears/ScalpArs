#!/usr/bin/env python3
"""OI_TS_AUDIT 2026-11-21 — unit tests for the aligned reader (synthetic files, no network).
Run: venv/bin/python -I -m pytest -q "scripts/OI_TS_AUDIT_2026-11-21_test.py"   (or run the file directly)."""
import importlib.util, os
import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("oi_al", os.path.join(os.path.dirname(os.path.abspath(__file__)), "OI_TS_AUDIT_2026-11-21_aligned.py"))
A = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(A)
M5 = 300_000


def _synthetic(day="2025-10-10", shift_min=5, seed=0):
    """a random-walk price on 5m bars; a vision-style metrics file whose snapshot at time s is LABELLED s − shift_min."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(day).value // 10**6
    t_open = t0 - 86_400_000 + np.arange(3 * 288) * M5                      # bars covering the day before / of / after
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(t_open))))
    end = t_open + M5
    snaps = t0 + np.arange(288) * M5 + shift_min * 60_000                    # true snapshot times; labels (ct) cover the UTC day
    c_at = pd.Series(close, index=end).reindex(snaps).values
    contracts = 1e6 + np.arange(288) * 1000.0                                 # a strictly increasing OI → the value at a row identifies it
    me = pd.DataFrame(dict(ct=snaps - shift_min * 60_000, sum_open_interest=contracts, sum_open_interest_value=contracts * c_at))
    return me, t_open, close, snaps, contracts


def test_offset_measured_plus5():
    me, t, c, snaps, _ = _synthetic(shift_min=5)
    f = A.frame(me, t, c)
    assert f.meas.all() and (f.off == M5).all()
    assert np.array_equal(f.snap.values, snaps)


def test_offset_measured_zero():
    me, t, c, snaps, _ = _synthetic(day="2024-03-05", shift_min=0)
    f = A.frame(me, t, c)
    assert f.meas.all() and (f.off == 0).all()


def test_no_lookahead_and_live_parity():
    """for every decision time T the aligned pick never uses a snapshot after T − lag; a naive create_time ≤ T read does (the bug)."""
    me, t, c, snaps, contracts = _synthetic(shift_min=5)
    f = A.frame(me, t, c)
    T = snaps[20:-1] + 7_000                                                 # decision times a few seconds after a period edge
    k = A.pick(f.snap.values, T, lag=A.LIVE_LAG_MS)
    assert (k >= 0).all()
    assert (f.snap.values[k] <= T - A.LIVE_LAG_MS).all()                     # no look-ahead, 5-min publication lag like live
    # live = API timestamp (= snapshot) ≤ T − 5 min → identical row
    live = np.searchsorted(snaps, T - 5 * 60_000, side="right") - 1
    assert np.array_equal(f.sum_open_interest.values[k], contracts[live])
    # the naive reader on the raw labels reads a snapshot taken AFTER T
    kn = np.searchsorted(me.ct.values, T, side="right") - 1
    assert (snaps[kn] > T).all()
    # the old `create_time ≤ T − 5` reader reads the snapshot at ≤ T (one period fresher than live, no look-ahead)
    ko = np.searchsorted(me.ct.values, T - 5 * 60_000, side="right") - 1
    assert (snaps[ko] <= T).all() and (snaps[ko] > T - 5 * 60_000).all()


def test_calendar_fallback_when_unmeasurable():
    me, t, c, snaps, _ = _synthetic(shift_min=5)
    f = A.frame(me, None, None)                                              # no klines → calendar rule (+5 in 2024-05 … 2026-11-15)
    assert (~f.meas).all() and (f.off == M5).all()
    me2, *_ = _synthetic(day="2026-11-18", shift_min=0)
    f2 = A.frame(me2, None, None)
    assert (f2.off == 0).all()


def test_pick_back_window():
    snap = np.arange(0, 100) * M5
    kc = np.array([60, 47, 10])
    kb = A.pick_back(snap, kc, 48 * M5, 3 * M5)
    assert list(kb) == [12, -1, -1]                                          # 60 − 48 = 12 · 47 − 48 < 0 · 10 − 48 < 0
    assert (A.pick(snap, np.array([50 * M5]), lag=M5, max_age=6 * M5) == 49).all()
    assert (A.pick(snap, np.array([500 * M5]), lag=M5, max_age=6 * M5) == -1).all()   # stale → none


if __name__ == "__main__":
    for nm, fn in list(globals().items()):
        if nm.startswith("test_"):
            fn(); print("ok", nm)
