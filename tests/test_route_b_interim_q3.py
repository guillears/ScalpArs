"""Route B interim Q3 look (PREREG Addenda 6 / 7): luck-family membership by counts, the shuffle null, cohort, helpers."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "route_b_signals"))
import interim_q3 as I  # noqa: E402

DAY = I.DAY_MS
START = 1_795_000_000_000 // DAY * DAY + 3_600_000          # 01:00 UTC of a day


def _rows(key, n, days, val=0.0, pair="AUSDT", start=START, prefix=None):
    ms = [start + (i % days) * DAY + (i // days) * 60_000 for i in range(n)]
    d = pd.DataFrame({"trigger": key, "pair": pair, "ms": ms, "minute": ms,
                      "trigger_id": [f"{prefix or key}-{i}" for i in range(n)]})
    d["day"] = d.ms // DAY
    d["hour"] = d.ms // I.HOUR_MS
    d["cd"] = d.pair + "|" + d.day.astype(str)
    for c in I.CELLS:
        d[c] = val
    return d


def _ctrls(t, k=5, val=0.0):
    """k controls per trigger on the same coin-day (+2 h), linked by trigger_id."""
    c = t.loc[t.index.repeat(k)].copy()
    c["ms"] = c.ms + 2 * 3_600_000
    c["minute"] = c.ms
    c["hour"] = c.ms // I.HOUR_MS
    for col in I.CELLS:
        c[col] = val
    return c.reset_index(drop=True)


def test_floor_keys_counts_only_openers_with_controls():
    a = _rows("H4_ask_vacuum", 20, 5)
    b = _rows("H4_bid_vacuum", 19, 5)
    h1 = _rows("H1_liq_long_cascade", 40, 4)
    t = pd.concat([a, b, h1])
    c = pd.concat([_ctrls(a), _ctrls(b), _ctrls(h1)])
    keys = ["H4_ask_vacuum", "H4_bid_vacuum", "H1_liq_long_cascade"]
    assert I.floor_keys(t, c, keys) == ["H4_ask_vacuum"]
    # the 5th day's openers lose their controls (not final) → that key drops below 5 control days
    c2 = c[~((c.trigger == "H4_ask_vacuum") & (c.day == a.day.max()))]
    assert I.floor_keys(t, c2, keys) == []


def test_floor_keys_ignores_outcomes():
    a, b = _rows("H4_ask_vacuum", 25, 5, val=+9.0), _rows("H4_ask_vacuum", 25, 5, val=-9.0)
    assert I.floor_keys(a, _ctrls(a), ["H4_ask_vacuum"]) == I.floor_keys(b, _ctrls(b, val=5.0), ["H4_ask_vacuum"]) == ["H4_ask_vacuum"]


def test_perm_null_zero_when_no_difference():
    t = _rows("H4_ask_vacuum", 30, 5, val=1.0)
    P = I.perm_deltas(t, _ctrls(t, val=1.0), I.CELLS, np.random.default_rng(0), 50)
    assert np.allclose(P, 0.0)


def test_perm_keeps_shuffles_inside_coin_day():
    """a planted effect in one coin-day: every row there carries +10 (trigger and control alike), elsewhere 0. Shuffling inside coin-days
    cannot move the +10 rows to other days, so the null Δ stays equal to the observed Δ's coin-day mix (here 0 → all Δ identical)."""
    t = _rows("H4_ask_vacuum", 20, 5)
    c = _ctrls(t)
    d0 = t.day.min()
    for x in (t, c):
        for col in I.CELLS:
            x.loc[x.day == d0, col] = 10.0
    P = I.perm_deltas(t, c, I.CELLS, np.random.default_rng(3), 200)
    tm, cm = I.matched(t, c)
    want = tm[I.CELLS].mean().to_numpy() - cm[I.CELLS].mean().to_numpy()
    assert np.allclose(P, want[None, :])


def test_family_null_small_key_excluded_lowers_bar():
    rng = np.random.default_rng(1)
    big_t = _rows("H4_ask_vacuum", 40, 5)
    big_c = _ctrls(big_t)
    for d in (big_t, big_c):
        for col in I.CELLS:
            d[col] = rng.normal(0, 1, len(d))
    tiny_t = _rows("H7b_oi_flush_pump|c", 1, 1, pair="BUSDT")
    tiny_c = _ctrls(tiny_t)
    for d in (tiny_t, tiny_c):
        for col in I.CELLS:
            d[col] = rng.normal(0, 5, len(d))
    t, c = pd.concat([big_t, tiny_t]), pd.concat([big_c, tiny_c])
    keys = I.floor_keys(t, c, ["H4_ask_vacuum", "H7b_oi_flush_pump|c"])
    _, nm_floor = I.family_null(t, c, keys, np.random.default_rng(2), 300)
    _, nm_all = I.family_null(t, c, ["H4_ask_vacuum", "H7b_oi_flush_pump|c"], np.random.default_rng(2), 300)
    assert keys == ["H4_ask_vacuum"]
    assert np.percentile(nm_floor, 95) < np.percentile(nm_all, 95)


def test_cohort_guard_nobook_and_day_from_trigger_minute():
    m_late = START - 3_600_000 + DAY - 60_000                       # 23:59 of day 0 → signal_time 00:00 next day
    T = pd.DataFrame({"trigger": "H4_ask_vacuum", "pair": "AUSDT", "trigger_id": ["a", "b", "c", "d"],
                      "minute": [START, m_late, START, START + 3 * DAY],
                      "h_final": [True, True, True, True], "h_cost_src": ["route_b_book", "route_b_book", "no_book", "route_b_book"]})
    T["signal_time"] = pd.to_datetime(T.minute + 60_000, unit="ms", utc=True)
    for col in I.CELLS:
        T[col] = 0.0
    C = pd.DataFrame({"trigger_id": ["a", "b", "c", "a"], "trigger": "H4_ask_vacuum", "pair": "AUSDT", "ctrl_k": [0, 0, 0, -1],
                      "minute": [START + 600_000, START + 600_000, START, np.nan], "ctrl_final": [True, True, True, True],
                      "h_final": [True, True, True, False], "h_cost_src": ["route_b_book"] * 3 + [""]})
    C["signal_time"] = pd.to_datetime(C.minute + 60_000, unit="ms", utc=True)
    for col in I.CELLS:
        C[col] = 0.0
    data_end = pd.Timestamp(START + 3 * DAY + 2 * 3_600_000, unit="ms", tz="UTC")   # row d is inside the 4 h guard
    t, c = I.cohort(T, C, data_end)
    assert set(t.trigger_id) == {"a", "b"}                                        # c: no_book · d: guard
    assert set(c.trigger_id) == {"a", "b"} and (c.ctrl_k >= 0).all()            # c's control leaves with it · stub out
    assert t.set_index("trigger_id").day["b"] == START // DAY                    # the trigger MINUTE's day, not signal_time's


def test_signed_share_and_halves():
    t = _rows("H4_ask_vacuum", 4, 2)
    t["pair"] = ["A", "A", "B", "B"]
    t["g_tp5_sl5_net"] = [3.0, 1.0, -1.0, 1.0]
    assert abs(I.signed_share(t, "pair", "g_tp5_sl5_net") - 4 / 4) < 1e-9
    t["g_tp5_sl5_net"] = [-3.0, 1.0, -1.0, 1.0]
    assert np.isnan(I.signed_share(t, "pair", "g_tp5_sl5_net"))
    t2 = _rows("H4_ask_vacuum", 4, 2, val=1.0)
    c2 = _ctrls(t2, val=0.0)
    h1, h2 = I.halves_delta(t2, c2, "g_tp5_sl5_net")
    assert h1 == 1.0 and h2 == 1.0


def test_wilson_and_kelly():
    assert abs(I.wilson_lo(50, 100) - 0.418) < 0.01
    assert I.kelly_floor([1, 1, 1, -1]) < 0.75
    assert np.isnan(I.kelly_floor([1, 2, 3]))
    assert I.HOLDOUT_ONLY == ("H3", "H5")


def test_interim_never_reads_holdout():
    t = pd.DataFrame({"trigger_id": ["a", "b"], "signal_time": [pd.Timestamp("2026-12-06 23:59", tz="UTC"),
                                                                 pd.Timestamp("2026-12-07 00:00", tz="UTC")]})
    c = pd.DataFrame({"trigger_id": ["a", "b", "b"]})
    tt, cc = I.design_only(t, c)
    assert list(tt.trigger_id) == ["a"] and list(cc.trigger_id) == ["a"]
