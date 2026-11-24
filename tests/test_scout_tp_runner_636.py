"""🏃👁 DECISION_LOG 636 — TP_RUNNER observe lines (HT 4 × ATR + floor +5 · EXH 4 × ATR, stop kept): pure pricing + the frozen reads."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_tp_runner_636 as S        # noqa: E402
import observe_registry as OR          # noqa: E402

MIN = 60_000
T0 = int(pd.Timestamp("2026-12-01 12:00").value // 1_000_000)   # touch minute


def bars(prices, start=T0 + MIN):
    """1m bars [[t, o, h, l, c]] from (o, h, l, c) tuples, one per minute from start."""
    return [[start + i * MIN, *p] for i, p in enumerate(prices)]


def flat_history(px=100.0, rng=1.0, n=300):
    """n 1m bars before the touch, each h − l = rng, close = px (15m TR = rng → ATR% = rng / px × 100)."""
    return bars([(px, px + rng / 2, px - rng / 2, px)] * n, start=T0 - n * MIN)


# ─── ATR ───
def test_atr15_matches_a_constant_range():
    assert S.atr15_pct(flat_history(100.0, 1.0), T0) == pytest.approx(1.0)


def test_atr15_needs_enough_bars_and_ignores_the_touch_minute():
    assert S.atr15_pct(flat_history(n=200), T0) is None
    h = flat_history() + bars([(100, 200, 1, 100)], start=T0)        # the touch minute itself never enters the ATR
    assert S.atr15_pct(h, T0) == pytest.approx(1.0)


# ─── the runner walk ───
def test_long_floor_binds_first():
    # TP 106, w 4 % → trail 101.76 < floor 105 → the floor binds when the next bar dips to 104
    why, px, _ = S.runner_walk(bars([(106, 106.2, 104, 104.5)]), T0, 106.0, 1, 4.0, 105.0)
    assert (why, px) == ("guard", 105.0)


def test_long_trail_after_a_run_and_gap_books_the_open():
    p = [(106, 130, 106, 129), (125, 126, 120, 121)]                  # best 130 → trail 124.8; the next bar OPENS 125 then trades 120
    why, px, t = S.runner_walk(bars(p), T0, 106.0, 1, 4.0, 105.0)
    assert why == "trail" and px == pytest.approx(124.8) and t == T0 + 3 * MIN
    gap = [(106, 130, 106, 129), (110, 111, 109, 110)]                 # opens below the trail → the open
    assert S.runner_walk(bars(gap), T0, 106.0, 1, 4.0, 105.0)[:2] == ("trail", 110.0)


def test_stop_uses_the_best_up_to_the_previous_bar():
    # the same bar makes a new high AND dips: the stop is from the previous best (106 → trail 101.76, floor 105) — not hit at 105.5
    why, _, _ = S.runner_walk(bars([(106, 140, 105.5, 139)] + [(139, 139, 139, 139)] * 1439), T0, 106.0, 1, 4.0, 105.0)
    assert why == "time"


def test_time_exit_after_24h_and_pending_before():
    full = bars([(107, 107.5, 106.5, 107)] * 1440)
    why, px, t = S.runner_walk(full, T0, 106.0, 1, 4.0, 105.0)
    assert (why, px, t) == ("time", 107.0, T0 + 1441 * MIN)
    assert S.runner_walk(full[:100], T0, 106.0, 1, 4.0, 105.0)[0] == "PENDING"


def test_short_keeps_its_own_stop_and_trails_the_low():
    # short TP at 93, own stop 106: a rally to 107 → stop 106 (trail 96.72 above 93 binds first, actually)
    why, px, _ = S.runner_walk(bars([(93, 97, 92.5, 96)]), T0, 93.0, -1, 4.0, 106.0)
    assert (why, px) == ("trail", pytest.approx(96.72))
    why, px, _ = S.runner_walk(bars([(93, 93.5, 92, 93)]), T0, 93.0, -1, 20.0, 106.0)
    assert why == "PENDING"
    why, px, _ = S.runner_walk(bars([(93, 108, 92, 107)]), T0, 93.0, -1, 20.0, 106.0)
    assert (why, px) == ("guard", 106.0)


# ─── Δ ───
def test_delta_signs_fees_slip_and_funding():
    d = S.delta_pp(1, 100.0, 106.0, 110.0, 0.0, slip=0.10, maker=0.02, taker=0.05)
    assert d == pytest.approx((110 * 0.999 - 106) - 0.03)
    s = S.delta_pp(-1, 100.0, 93.0, 90.0, -0.5, slip=0.10, maker=0.02, taker=0.05)
    assert s == pytest.approx((93 - 90 * 1.001) - 0.03 - 0.5)


def test_ht_floor_price():
    assert S.ht_floor_px(100.0, 5.0, fee=0.05) == pytest.approx(100 * 105.05 / 99.95)


# ─── reads ───
def test_ht_read_is_the_sum_of_the_first_40():
    assert S.ht_read([1.0] * 39, ["trail"] * 39)[0] == "COLLECTING"
    assert S.ht_read([-1.3] * 36 + [12.0] * 4 + [-99] * 5, ["guard"] * 36 + ["trail"] * 4 + ["guard"] * 5)[0] == "ARM_CANDIDATE"
    assert S.ht_read([-1.3] * 40, ["guard"] * 40)[0] == "KEEP_TP"


def test_ht_info_warnings():
    w = S.ht_info([-2.6] * 10, ["guard"] * 10)
    assert any("floor slippage" in x for x in w) and any("far behind" in x for x in w)
    assert S.ht_info([-1.2] * 10, ["guard"] * 10) == []


def test_exh_read_needs_50_on_10_days_and_a_positive_lower_bound():
    assert S.exh_read([(d % 5, 1.0) for d in range(60)])[0] == "COLLECTING"        # 5 days only
    up = [(d % 12, 2.0 + (d % 3) * 0.1) for d in range(50)]
    assert S.exh_read(up)[0] == "ARM_CANDIDATE"
    noisy = [(d % 12, (5.0 if d % 2 else -5.0)) for d in range(50)]
    assert S.exh_read(noisy)[0] == "KEEP_TP"


def test_lower_bound_is_below_the_mean():
    rows = [(d % 11, float(x)) for d, x in enumerate(np.random.default_rng(1).normal(0.5, 3, 60))]
    assert S.lower_bound(rows) < np.mean([x for _, x in rows])


# ─── report on a fixture ───
def _frame(sleeve, rows):
    a = pd.DataFrame(rows).reindex(columns=list(S.COLS))
    a["entry_strategy"], a["status"], a["direction"] = sleeve, "CLOSED", "LONG" if sleeve == S.HT else "SHORT"
    a["o"] = pd.to_datetime(a.opened_at)
    return a.reindex(columns=list(S.COLS) + ["o"])


def test_report_blocks_freeze_and_pending(tmp_path):
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    era = {S.HT: int(pd.Timestamp("2026-11-21").value // 1_000_000), S.EXH: int(pd.Timestamp("2026-11-21").value // 1_000_000)}
    touch = pd.Timestamp(T0, unit="ms")
    ht = _frame(S.HT, [
        dict(opened_at="2026-11-15 10:00", pair="OLDUSDT", pnl_percentage=6.0, close_reason="FRENZY_TP", entry_price=100, exit_price=106.05,
             closed_at=str(touch), frenzy_willy_tp_target_pct=6.0),                       # older era → in neither block
        dict(opened_at="2026-11-25 10:00", pair="INSUSDT", pnl_percentage=6.0, close_reason="FRENZY_TP", entry_price=100, exit_price=106.05,
             closed_at=str(touch), frenzy_willy_tp_target_pct=6.0),                       # in-sample
        dict(opened_at="2026-11-25 11:00", pair="TP5USDT", pnl_percentage=5.0, close_reason="FRENZY_TP", entry_price=100, exit_price=105,
             closed_at=str(touch), frenzy_willy_tp_target_pct=5.0),                       # +5 target → out
        dict(opened_at="2026-11-25 12:00", pair="SLUSDT", pnl_percentage=-4.0, close_reason="STOP_LOSS", entry_price=100, exit_price=96,
             closed_at=str(touch), frenzy_willy_tp_target_pct=6.0)] +                    # not a TP → out
        [dict(opened_at=f"2026-12-01 0{i}:00", pair=f"F{i}USDT", pnl_percentage=6.0, close_reason="FRENZY_TP", entry_price=100,
              exit_price=106.05, closed_at=str(touch), frenzy_willy_tp_target_pct=6.0) for i in range(3)])
    exh = _frame(S.EXH, [])
    seen = []

    def bars_fn(pair, t, now):
        seen.append(pair)
        return flat_history(106.0, 1.06) + bars([(106, 106.1, 104, 104.5)])           # ATR 1 % → trail 4 %: the floor binds

    st = tmp_path / "s.json"
    L = "\n".join(S.report(frames={S.HT: ht, S.EXH: exh}, since_ms=since, era_ms=era, state_path=str(st), now_ms=T0 + 3 * 86_400_000,
                           bars_fn=bars_fn, funding_fn=lambda p, a, b: 0.0, newest=pd.Timestamp("2026-12-05")))
    assert "OLDUSDT" not in seen and "TP5USDT" not in seen and "SLUSDT" not in seen
    assert "IN-SAMPLE" in L and "FORWARD" in L and "COLLECTING" in L
    ins = [x for x in L.splitlines() if "IN-SAMPLE" in x and "N 1 " in x]
    fwd = [x for x in L.splitlines() if "FORWARD" in x and "N 3 " in x]
    assert ins and fwd and "'guard': 3" in fwd[0]
    assert not st.exists()                                                           # nothing decided → nothing frozen


def test_report_freezes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "HT_N", 2)
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    touch = str(pd.Timestamp(T0, unit="ms"))
    ht = _frame(S.HT, [dict(opened_at=f"2026-12-01 0{i}:00", pair=f"F{i}USDT", pnl_percentage=6.0, close_reason="FRENZY_TP", entry_price=100,
                            exit_price=106.05, closed_at=touch, frenzy_willy_tp_target_pct=6.0) for i in range(2)])
    kw = dict(frames={S.HT: ht, S.EXH: _frame(S.EXH, [])}, since_ms=since, era_ms={S.HT: since, S.EXH: since}, now_ms=T0 + 3 * 86_400_000,
              bars_fn=lambda p, t, n: flat_history(106.0, 1.06) + bars([(106, 106.1, 104, 104.5)]), funding_fn=lambda p, a, b: 0.0)
    st = tmp_path / "s.json"
    L = "\n".join(S.report(state_path=str(st), newest=pd.Timestamp("2026-11-30"), **kw))   # export too old → deferred
    assert "freezing deferred" in L and not st.exists()
    L = "\n".join(S.report(state_path=str(st), newest=pd.Timestamp("2026-12-05"), **kw))
    assert "FROZEN" in L and json.loads(st.read_text())["HT_RUNNER"]["verdict"] == "KEEP_TP"


def test_unread_funding_holds_then_counts_after_48h(tmp_path):
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    touch = str(pd.Timestamp(T0, unit="ms"))
    exh = _frame(S.EXH, [dict(opened_at="2026-12-01 01:00", pair="XUSDT", pnl_percentage=6.9, close_reason="FRENZY_TP", entry_price=100,
                              exit_price=93.0, closed_at=touch, entry_exh_tp_pct=7.0, entry_exh_atr_pct=1.0, entry_exh_sl_pct=6.0)])
    kw = dict(frames={S.HT: _frame(S.HT, []), S.EXH: exh}, since_ms=since, era_ms={S.HT: since, S.EXH: since}, state_path=str(tmp_path / "s.json"),
              bars_fn=lambda p, t, n: flat_history(93.0, 0.93) + bars([(93, 97, 92.5, 96)]), funding_fn=lambda p, a, b: None,
              newest=pd.Timestamp("2026-12-09"))
    assert "1 pending" in "\n".join(S.report(now_ms=T0 + 3600_000, **kw))
    L = "\n".join(S.report(now_ms=T0 + 50 * 3600_000, **kw))
    assert "N 1 " in L and "without funding" in L


def test_registry_and_hook():
    assert "scout_tp_runner_636.py" in OR.classified()
    w = OR.EXIT["scout_tp_runner_636.py"]["walks"]
    assert w["frenzy_willy_ht_tp_pct"] == 6.0 and w["exhaustion_tp_cap_pct"] == 7.0
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, "scripts", "opportunity_scout.py")).read()
    assert "import scout_tp_runner_636" in src


# ─── review fixes ───
def test_nat_closed_at_is_not_evaluable():
    r = pd.Series(dict(pair="X", entry_price=100, exit_price=106, closed_at=float("nan"), o=pd.Timestamp("2026-12-01")))
    from types import SimpleNamespace
    assert S.price_fill(SimpleNamespace(**r), 1, T0, None, bars_fn=lambda *a: pytest.fail("fetched")).startswith("NOT_EVALUABLE")


def test_ht_cohort_is_exactly_the_plus6_target():
    a = _frame(S.HT, [dict(opened_at="2026-12-01 01:00", pair=p, pnl_percentage=6, close_reason="FRENZY_TP", frenzy_willy_tp_target_pct=t)
                      for p, t in (("A", 6.0), ("B", 8.0), ("C", 7.69), ("D", 5.0))])
    assert list(S._cohort(S.HT, a).pair) == ["A"]


def test_paging_dedupes_overlapping_pages(monkeypatch):
    calls = []

    def fk(sym, iv, start, limit, bud, cache, now, fetch=True):
        calls.append(start)
        if len(calls) == 1:          # page 1 has a gap → 900 rows ending later than start + 900 min
            return [[start + i * MIN + (MIN if i > 10 else 0), 1, 1, 1, 1, 0] for i in range(limit)]
        return [[start + i * MIN, 1, 1, 1, 1, 0] for i in range(5)]
    monkeypatch.setattr(S.LS, "fapi_klines", fk)
    out = S._bars("X", T0, T0 + 10 * 86_400_000, S.LS.Budget())
    t = [q[0] for q in out]
    assert t == sorted(set(t)) and calls[1] == calls[0] + 900 * MIN + MIN


def _fwd_two(tmp_path, monkeypatch, funding):
    monkeypatch.setattr(S, "HT_N", 2)
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    touch = str(pd.Timestamp(T0, unit="ms"))
    ht = _frame(S.HT, [dict(opened_at=f"2026-12-01 0{i}:00", pair=f"F{i}USDT", pnl_percentage=6.0, close_reason="FRENZY_TP", entry_price=100,
                            exit_price=106.05, closed_at=touch, frenzy_willy_tp_target_pct=6.0) for i in range(2)])
    return dict(frames={S.HT: ht}, since_ms=since, era_ms={S.HT: since, S.EXH: since}, now_ms=T0 + 3 * 86_400_000,
                bars_fn=lambda p, t, n: flat_history(106.0, 1.06) + bars([(106, 106.1, 104, 104.5)]), funding_fn=funding,
                newest=pd.Timestamp("2026-12-09"))


def test_freeze_held_while_a_deciding_fill_lacks_funding(tmp_path, monkeypatch):
    st = tmp_path / "s.json"
    L = "\n".join(S.report(state_path=str(st), **_fwd_two(tmp_path, monkeypatch, lambda p, a, b: None)))
    assert "funding unread on 2" in L and not st.exists()


def test_bad_state_entry_warns_and_never_freezes(tmp_path, monkeypatch):
    st = tmp_path / "s.json"
    st.write_text(json.dumps({"HT_RUNNER": "oops"}))
    L = "\n".join(S.report(state_path=str(st), **_fwd_two(tmp_path, monkeypatch, lambda p, a, b: 0.0)))
    assert "HT_RUNNER entry" in L and json.loads(st.read_text())["HT_RUNNER"] == "oops"


def test_funding_window_ends_at_the_exit_bar_open():
    got = []
    r = pd.Series(dict(pair="X", entry_price=100.0, exit_price=106.05, closed_at=str(pd.Timestamp(T0, unit="ms")), o=pd.Timestamp("2026-12-01")))
    from types import SimpleNamespace
    S.price_fill(SimpleNamespace(**r), 1, T0 + 3 * 86_400_000, None,
                 bars_fn=lambda *a: flat_history(106.0, 1.06) + bars([(106, 106.1, 104, 104.5)]),
                 funding_fn=lambda p, a, b: got.append((a, b)) or 0.0)
    assert got == [(T0, T0 + MIN)]          # exit bar opens at touch minute + 1 min
