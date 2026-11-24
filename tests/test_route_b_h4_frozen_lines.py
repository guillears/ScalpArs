"""PREREG Addendum 9 — the frozen H4 lines: cohort selection, BTC 4 h return, gate logic (pure parts)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "route_b_signals"))
import h4_frozen_lines as H  # noqa: E402

M = 60_000
H0 = H.HOLDOUT_START.value // 1_000_000


def test_frozen_constants():
    assert (H.TP, H.SL, H.HOLD, H.ASK_MIN_USD) == (5.0, 3.0, 1440, 12_000.0)
    assert str(H.HOLDOUT_START) == "2026-12-07 00:00:00+00:00"
    assert (H.T_MIN_N, H.T_MIN_DAYS, H.B_MIN_DAYS) == (30, 10, 10)


def test_holdout_rows_filters():
    T = pd.DataFrame({"trigger": ["H4_ask_vacuum"] * 5 + ["H4_bid_vacuum"], "pair": ["AUSDT"] * 6,
                      "minute": [H0 - M, H0, H0 + M, H0 + 2 * M, H0 + 3 * M, H0],
                      "h_cost_src": ["route_b_book", "route_b_book", "no_book", "book_unwalkable", "route_b_book", "route_b_book"],
                      "h_entry_ms": [1.0] * 6,
                      "rb_ask_usd_05": [20_000, 12_000, 20_000, 30_000, 11_999, 20_000]})
    got = H.holdout_rows(T)
    assert list(got.index) == [1, 3]          # before HOLDOUT · no_book · below $12k · other key → out
    T.loc[1, "pair"] = "BTCUSDT"
    assert list(H.holdout_rows(T).index) == [3]                         # alts only


def test_btc_ret4h_uses_closed_minutes_only():
    t0 = 1_800_000_000_000 // M * M
    btc = pd.DataFrame({"open_ms": [t0 - 241 * M, t0 - M, t0], "c": [100.0, 102.0, 999.0]}).set_index("open_ms").c
    assert abs(H.btc_ret4h(btc, t0) - 2.0) < 1e-9          # the trigger minute's own close (999) is never read
    assert np.isnan(H.btc_ret4h(btc.iloc[1:], t0))


def test_walk_net_needs_full_path_and_books_tp():
    t0 = 1_800_000_000_000 // M * M
    n = H.HOLD
    k = pd.DataFrame({"open_ms": t0 + np.arange(n) * M, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0})
    k.loc[10, "h"] = 106.0
    net = H.walk_net(k, t0, 0.0, 0.0, [])
    assert net > 4.5                                            # TP +5 booked, minus fees
    assert np.isnan(H.walk_net(k.iloc[:-1], t0, 0.0, 0.0, []))   # 24 h path incomplete → not final
    gap = k.drop(index=500).reset_index(drop=True)
    gap = pd.concat([gap, k.iloc[[-1]].assign(open_ms=k.open_ms.iloc[-1] + M)], ignore_index=True)
    assert np.isnan(H.walk_net(gap, t0, 0.0, 0.0, []))           # a missing minute → not contiguous → not final


def _rows(n_days, per_day, val, btc=1.0, trends=(1, -1)):
    r = []
    for d in range(n_days):
        for i in range(per_day):
            r.append(dict(net=val, day=d, hour=d * 24 + i, pair=f"C{i % 7}", trigger_id=f"{d}-{i}", btc4h=btc,
                          rb_ask_usd_05=15_000.0, btc_d_trend=trends[d % len(trends)]))
    return pd.DataFrame(r)


def test_gates_T_collecting_then_verdicts():
    rng = np.random.default_rng(0)
    assert H.gates_T(_rows(9, 5, 1.0), pd.DataFrame(columns=["trigger_id", "net"]), rng)["status"] == "COLLECTING"
    good = _rows(12, 4, 1.0)
    good["net"] = good.net + rng.normal(0, 0.2, len(good))
    ctrl = good[["trigger_id"]].assign(net=-0.5)
    assert H.gates_T(good, ctrl, rng)["status"] == "PASS"
    g = H.gates_T(good, ctrl, rng)
    assert g["capacity_median_ask_usd"] == 15_000.0
    bad = good.assign(net=good.net - 2.0)
    assert H.gates_T(bad, ctrl, rng)["status"] == "FAIL"
    one = good.assign(btc_d_trend=1)
    assert H.gates_T(one, ctrl, rng)["status"] == "PASS (one BTC regime → OBSERVE)"     # B4 cap
    allwin = good.assign(net=1.0)
    assert H.gates_T(allwin, ctrl, rng)["kelly_floor"] > 0                               # no losers → Wilson floor


def test_gates_B_day_units():
    rng = np.random.default_rng(1)
    up, dn = _rows(12, 3, 1.0, btc=+0.5), _rows(12, 3, -1.0, btc=-0.5)
    up["net"] = up.net + rng.normal(0, 0.3, len(up))
    assert H.gates_B(pd.concat([up, dn]), rng)["status"] == "PASS"
    assert H.gates_B(pd.concat([up, dn.iloc[:27]]), rng)["status"] == "COLLECTING"     # 9 DOWN days
    assert H.gates_B(pd.concat([up.assign(net=-1.0), dn]), rng)["status"] == "FAIL"
    z = pd.concat([up, dn.assign(btc4h=0.0)])
    assert H.gates_B(z, rng)["days_down"] == 12                                          # BTC 4 h = 0 → DOWN


def test_freeze_reads_once():
    st = {}
    r, st = H.freeze(st, "H4-T", {"status": "COLLECTING", "N": 5}, "c1")
    assert st == {} and r["status"] == "COLLECTING"
    r, st = H.freeze(st, "H4-T", {"status": "FAIL", "N": 31, "ci": (-1.0, 0.5)}, "c2")
    assert r["status"] == "FAIL" and st["H4-T"]["frozen_at"] == "c2"
    r, st = H.freeze(st, "H4-T", {"status": "PASS", "N": 60}, "c3")
    assert r["status"] == "FAIL" and r["frozen_at"] == "c2"                              # never re-judged
