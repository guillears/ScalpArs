"""👁 Oct-30: the FB_CONFIRM watch line (scripts/scout_flatbase_confirm_watch.py, DECISION_LOG 417) — frozen constants, the shadow walker, cohort, the bar, the deferrals, freeze once, damaged state, the hook and the registry.
Hermetic: temp exports and state, injected bars, no network, no git."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_flatbase_confirm_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")          # a 4-hour boundary
TE = int(T0.value // 10**6) + 10_000            # the signal fills just after the 4 h close
SINCE = int((T0 - pd.Timedelta(days=1)).value // 10**6)
NOW = int((T0 + pd.Timedelta(days=60)).value // 10**6)
E = 100.0


def _bars(path):
    """path = [(close price, high, low)] per 5m bar from T0; open = the previous close."""
    out, prev = [], E
    for i, (c, h, l) in enumerate(path):
        out.append([int(T0.value // 10**6) + i * W.M5, prev, h, l, c])
        prev = c
    return out


def _flat(n, c=100.5):
    return [(c, c + 0.1, c - 0.1)] * n


def test_frozen_constants():
    assert (W.DL, W.CONFIRM_H, W.MAX_H, W.STOP_GROSS, W.TP_GROSS, W.STOP_NET, W.TP_NET) == (417, 24, 72, 10.0, 5.0, -10.09, 4.94)
    assert (W.N_MIN, W.DAYS_MIN, W.D_MIN, W.P_MIN, W.SHARE_MAX, W.WAIT_MIN) == (15, 8, 0.50, 0.90, 0.50, 4320)
    assert "reports/SCOUT_FB_CONFIRM.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_skip_when_no_4h_close_below_within_24h():
    assert W.price_shadow(_bars(_flat(290)), TE, E, NOW) == ("SKIP", 0.0)
    assert W.price_shadow(_bars(_flat(100)), TE, E, NOW)[0] == "PENDING"            # the 24 h are not over
    assert W.price_shadow([], TE, E, NOW)[0] == "UNPRICED"


def test_only_a_4h_close_confirms_and_the_entry_is_the_next_open():
    p = _flat(47) + [(100.2, 100.3, 99.0)]                                           # a wick below inside the candle, closing above: no confirmation
    p += _flat(47, 100.1) + [(99.0, 100.1, 98.9)]                                    # bar 95 closes the second 4 h candle below E
    p += [(98.0, 99.2, 97.9)] + [(94.0, 98.0, 93.9)]                                 # entry at bar 96's open 99.0; TP line 94.05
    assert W.price_shadow(_bars(p), TE, E, NOW) == ("TP", W.TP_NET)


def test_stop_first_on_a_bar_touching_both_and_gap_books_the_open():
    p = _flat(47) + [(99.0, 100.0, 98.9)] + [(95.0, 110.0, 90.0)]                    # entry 99.0; the same bar touches both lines
    assert W.price_shadow(_bars(p), TE, E, NOW) == ("STOP", W.STOP_NET)
    b = _bars(_flat(47) + [(99.0, 100.0, 98.9)])                                     # entry = the next bar's open
    t = b[-1][0]
    b += [[t + W.M5, 99.0, 99.5, 98.5, 99.0], [t + 2 * W.M5, 112.0, 112.5, 111.5, 112.0]]   # the second bar GAPS through the 108.9 line
    why, v = W.price_shadow(b, TE, E, NOW)
    assert why == "STOP" and v == pytest.approx(-(112.0 / 99.0 - 1) * 100 - W.TIME_FEE)


def test_tp_needs_a_trade_through_the_limit_and_window_edge():
    p = _flat(47) + [(99.0, 100.0, 98.9)] + [(95.0, 99.0, 94.05)]                    # low exactly on the 94.05 line: no fill
    assert W.price_shadow(_bars(p), TE, E, NOW)[0] == "PENDING"
    p = _flat(47) + [(99.0, 100.0, 98.9)] + [(95.0, 99.0, 94.04)]
    assert W.price_shadow(_bars(p), TE, E, NOW)[0] == "TP"
    edge = _flat(287) + [(99.0, 100.5, 98.9)] + [(94.0, 99.0, 93.0)]                 # the close at +24 h still counts
    assert W.price_shadow(_bars(edge), TE, E, NOW)[0] == "TP"
    late = _flat(335) + [(99.0, 100.5, 98.9)] + [(94.0, 99.0, 93.0)]                 # the close at +28 h does not
    assert W.price_shadow(_bars(late), TE, E, NOW) == ("SKIP", 0.0)


def test_fetch_bars_stops_on_budget_or_error(monkeypatch):
    bud = W.LS.Budget()
    monkeypatch.setattr(W.LS, "fapi_klines", lambda *a, **k: [[1, 2, 3, 4, 5, 6]])
    assert W.fetch_bars("AUSDT", TE, NOW, bud) == [[1, 2, 3, 4, 5]]
    monkeypatch.setattr(W.LS, "fapi_klines", lambda *a, **k: None)
    assert W.fetch_bars("AUSDT", TE, NOW, bud) is None
    monkeypatch.setattr(W.LS, "fapi_klines", lambda *a, **k: [])
    bud.errors = 1
    assert W.fetch_bars("AUSDT", TE, NOW, bud) is None


def test_load_dedup_closed_beats_newer_open_and_newest_wins(tmp_path):
    a = _fills([(0, "AUSDT", -1.0)])
    b = a.copy(); b["pnl_percentage"] = 4.94
    o = a.copy(); o["status"] = "OPEN"
    p1, p2, p3 = _write(tmp_path, a, "2026-11-01_00-00-00"), _write(tmp_path, b, "2026-11-02_00-00-00"), _write(tmp_path, o, "2026-11-03_00-00-00")
    for k, f in enumerate((p1, p2, p3), 1):
        os.utime(f[0], (k, k))
    assert W.load(p1 + p2 + p3)[0].pct.tolist() == [4.94]


def test_time_exit_and_pending():
    p = _flat(47) + [(99.0, 100.0, 98.9)] + _flat(900, 98.5)
    why, v = W.price_shadow(_bars(p), TE, E, NOW)
    assert why == "TIME" and v == pytest.approx(-(98.5 / 99.0 - 1) * 100 - W.TIME_FEE)
    assert W.price_shadow(_bars(_flat(47) + [(99.0, 100.0, 98.9)] + _flat(10, 98.5)), TE, E, NOW)[0] == "PENDING"


def _fills(spec):
    """spec = [(day offset, pair, booked pct)]; rows at T0 + day offset (a 4-hour boundary)."""
    rows = []
    for i, (d, pair, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, seconds=10 + i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair, direction="SHORT", entry_strategy="FLATBASE_SHORT",
                         status="CLOSED", entry_price=E, pnl_percentage=pct, pnl=pct * 10, close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS"))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2027-01-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _rows(pct, shadow, days=8):
    n = len(pct)
    return pd.DataFrame(dict(o=[T0 + pd.Timedelta(days=i % days) for i in range(n)], day=[f"2026-11-{1 + i % days:02d}" for i in range(n)], pair=[f"P{i}USDT" for i in range(n)], pct=pct, shadow=shadow,
                             why=["SKIP" if s == 0 else "TP" for s in shadow]))


def test_verdict_outcomes():
    st, det = W.verdict(_rows([-10.09] * 6 + [4.94] * 10, [0.0] * 6 + [4.94] * 10))
    assert st == W.CANDIDATE and "skipped 6" in det, det
    assert W.verdict(_rows([4.94] * 16, [0.0] * 16))[0] == "RETIRED"                # confirmation skipping winners
    assert W.verdict(_rows([4.94] * 16, [4.94] * 16))[0] == "RETIRED"               # no gain
    assert W.verdict(_rows([1.0] * 16, [1.3] * 16))[0] == "KEEP OBSERVING"          # gain below +0.50
    assert W.verdict(_rows([-10.09] + [1.0] * 15, [20.0] + [1.0] * 15))[0] == "KEEP OBSERVING"   # one signal carries it
    assert W.verdict(_rows([-5.0] * 16, [-4.0] * 16))[0] == "KEEP OBSERVING"        # shadow still negative
    assert W.verdict(_rows([1.0] * 5, [2.0] * 5))[0] == "COLLECTING"


def _spec_candidate():
    return [(i % 8, f"S{i}USDT", -10.09) for i in range(6)] + [(i % 8, f"W{i}USDT", 4.94) for i in range(10)]


def _bars_fn(pair, te, now):
    if pair.startswith("S"):
        return [[(te // W.M5 * W.M5) + i * W.M5, E, E + 0.5, E, E + 0.2] for i in range(300)]    # never closes below → SKIP
    b = [[(te // W.M5 * W.M5) + i * W.M5, E, E + 0.5, E, E + 0.2] for i in range(48)]
    b[47][4] = 99.0
    b += [[b[-1][0] + W.M5, 99.0, 99.2, 90.0, 92.0]]                                            # confirmed, then TP
    return b


def test_freeze_once_deferrals_and_damaged_state(tmp_path):
    d = _fills(_spec_candidate())
    last = pd.to_datetime(d.opened_at).max()
    st = tmp_path / "w.json"
    early = _write(tmp_path, d, (last + pd.Timedelta(hours=1)).strftime("%Y-%m-%d_%H-%M-%S"))
    assert "newest export < the crossing signal + 4320 min" in "\n".join(W.run(NOW, early, str(st), SINCE, _bars_fn)) and not st.exists()
    late = _write(tmp_path, d, (last + pd.Timedelta(hours=73)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, late, str(st), SINCE, lambda *a: None))
    assert "kline fetch stopped" in out and not st.exists()
    out = "\n".join(W.run(NOW, late, str(st), SINCE, _bars_fn))
    assert "Frozen verdict" in out and "Before → after at the crossing: booked" in out and json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE
    W.run(NOW, _write(tmp_path, _fills([(i % 8, f"W{i}USDT", 4.94) for i in range(20)]), "2027-02-01_00-00-00"), str(st), SINCE, _bars_fn)
    assert json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE                     # never re-decided
    for bad in ('{"FLATBASE_SHORT": "x"}', '{"FLATBASE_SHORT": {"state": "x"}}'):
        st.write_text(bad)
        o = "\n".join(W.run(NOW, late, str(st), SINCE, _bars_fn))
        assert "frozen record damaged" in o and "Frozen verdict" not in o and st.read_text() == bad


def test_pending_signal_inside_the_prefix_holds_the_freeze(tmp_path):
    d = _fills(_spec_candidate())
    last = pd.to_datetime(d.opened_at).max()
    st = tmp_path / "w.json"
    p = _write(tmp_path, d, (last + pd.Timedelta(hours=73)).strftime("%Y-%m-%d_%H-%M-%S"))
    fn = lambda pair, te, now: [] if pair == "W0USDT" else _bars_fn(pair, te, now)
    out = "\n".join(W.run(NOW, p, str(st), SINCE, fn))
    assert "a signal inside the prefix is not priced yet" in out and not st.exists()


def test_cohort_floor_operator_closed_other_sleeves_and_empty(tmp_path):
    a = _fills([(0, "AUSDT", 4.94), (0, "MUSDT", 4.94), (-5, "OUSDT", 4.94), (0, "LUSDT", 4.94)])
    a.loc[1, "close_reason"] = "MANUAL"
    a.loc[3, "entry_strategy"] = "SPIKE_FADE"
    out = "\n".join(W.run(NOW, _write(tmp_path, a), None, SINCE, _bars_fn))
    assert "priced 1 ·" in out
    assert "priced 0 ·" in "\n".join(W.run(NOW, [], None, SINCE, _bars_fn))


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_flatbase_confirm_watch as _fbc")
    assert "_fbc.run(now_ms)" in src[j:j + 200] and "except Exception as _fbc_e" in src[j:j + 400]
    import observe_registry as OR
    era = OR.EXIT["scout_flatbase_confirm_watch.py"]["walks"]
    assert era["flatbase_tp_pct"] == 5.0 and era["flatbase_stop_pct"] == 10.0 and era["flatbase_max_hold_hours"] == 72.0
