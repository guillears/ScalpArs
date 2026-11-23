"""Route B — today's filter stack on every signal (DL 635): filters armed after the recording began are re-applied to the signals from
before they went live, with the engine's own functions; a switch change in trading_config.json fails until a re-check is decided. No network."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b_signals"))
import today_stack as TS    # noqa: E402

M = 60_000
TH = {"flatbase_funding_above_base_block": True, "flatbase_max_entry_funding_rate": 0.00005,
      "frenzy_willy_ht_burst_rejection_enabled": True}
BEFORE_FB = pd.Timestamp("2026-11-20 12:00", tz="UTC")
BEFORE_HT = pd.Timestamp("2026-11-21 12:00", tz="UTC")
AFTER = pd.Timestamp("2026-11-23 12:00", tz="UTC")


def _row(**k):
    d = dict(sleeve="FLATBASE_SHORT", pair="AUSDT", status="OPENED", block_class="taken", signal_time=BEFORE_FB,
             journal_funding_rate=float("nan"), rb_funding_pct=float("nan"))
    d.update(k)
    return d


def _bars(decision_ms, burst=True, close_low=True, n=75):
    """75 one-minute rows before the decision minute: base volume 1 (60 bars) then 3 (15 bars) when burst; last close near the low."""
    m0 = decision_ms // M * M
    out = []
    for i in range(n):
        o = m0 - (n - i) * M
        v = 3.0 if (burst and i >= n - 15) else 1.0
        c = 1.0 if (close_low and i == n - 1) else 1.5
        out.append([o, 1.5, 2.0, 1.0, c, v])
    return out


def test_entry_snapshot_matches_live_config():
    with open(os.path.join(ROOT, "trading_config.json")) as f:
        th = json.load(f)["thresholds"]
    live = TS.entry_settings(th)
    assert live == TS.ENTRY_SNAPSHOT, (
        "an ENTRY setting of a Route B sleeve changed in trading_config.json: signals the engine judged with the old setting may need a "
        "today_stack.RECHECKS entry (or state why none is needed) — then move ENTRY_SNAPSHOT. Diff: "
        + str({k: (TS.ENTRY_SNAPSHOT.get(k), live.get(k)) for k in set(live) | set(TS.ENTRY_SNAPSHOT) if live.get(k) != TS.ENTRY_SNAPSHOT.get(k)}))


def test_every_recheck_switch_is_in_the_snapshot_and_on():
    for rc in TS.RECHECKS:
        assert TS.ENTRY_SNAPSHOT.get(rc.switch) is True
    assert "flatbase_max_entry_funding_rate" in TS.ENTRY_SNAPSHOT


def test_guard_covers_entry_filters_and_skips_exits_and_size():
    th = dict(TS.ENTRY_SNAPSHOT)
    for k, v in (("watch_block_willy_ht_failed_spike", False), ("frenzy_willy_ht_min_dip_pct", 0.0),
                 ("exhaustion_vol_decay_min", 0.7), ("funding_guard_rate_min_pct", 0.1)):
        assert TS.entry_settings(dict(th, **{k: v})) != TS.ENTRY_SNAPSHOT, k          # an entry change trips the guard
    for k in ("frenzy_willy_runner_enabled", "exhaustion_short_enabled", "flatbase_stop_counts_funding", "exhaustion_tp_cap_pct",
              "frenzy_willy_ht_tp_pct", "flatbase_invest_mult", "distribution_sl_atr_mult", "frenzy_willy_ht_max_hold_minutes",
              "frenzy_willy_late_tp_pct", "frenzy_willy_max_slots", "momentum_x_enabled"):
        assert TS.entry_settings(dict(th, **{k: 123})) == TS.ENTRY_SNAPSHOT, k       # exits / size / masters / other sleeves do not


def test_fb_funding_above_cut_refuses_before_arm_only():
    S = pd.DataFrame([_row(journal_funding_rate=0.0001), _row(journal_funding_rate=0.0001, signal_time=AFTER),
                      _row(journal_funding_rate=0.00005), _row(journal_funding_rate=-0.002)])
    R = TS.apply(S, TH)
    assert list(R.today_recheck_state) == ["zone", "", "rest", "rest"]      # after trust_from the engine's own decision stands
    assert list(R.today_block) == ["FLATBASE_FUNDING_ABOVE_BASE", "", "", ""]
    assert list(R.today_block_class) == ["filter", "taken", "taken", "taken"]
    assert list(R.today_filters_pass) == [False, True, True, True]


def test_fb_missing_journal_stamp_fails_open_like_the_engine():
    R = TS.apply(pd.DataFrame([_row(rb_funding_pct=0.01), _row()]), TH)       # the recorder's rate is audit text only, never a block
    assert list(R.today_recheck_state) == ["unread", "unread"]
    assert "recorder" in R.today_recheck_value.iloc[0]
    assert list(R.today_filters_pass) == [True, True]


def test_fb_only_rows_that_reached_the_order_decision():
    S = pd.DataFrame([_row(status="REFUSED:EXPIRED", block_class="capacity", journal_funding_rate=0.01),
                      _row(status="REFUSED:OTHER", block_class="capacity", journal_funding_rate=0.01, detail="refused by the open path"),
                      _row(status="BLOCKED:NOT_FLAT", block_class="core_rule", journal_funding_rate=0.01),
                      _row(status="REFUSED:PAIR_HELD", block_class="capacity"),          # pre-check pair held: no funding stamp
                      _row(status="REFUSED:OTHER", block_class="capacity", journal_funding_rate=0.01,
                           detail="one-per-episode check unreadable at the S4 bar (fail-closed)")])
    R = TS.apply(S, TH)
    assert list(R.today_recheck) == ["", "FLATBASE_FUNDING_ABOVE_BASE", "", "", ""]
    assert list(R.today_block_class) == ["capacity", "filter", "core_rule", "capacity", "capacity"]


def test_switch_off_means_no_recheck():
    R = TS.apply(pd.DataFrame([_row(journal_funding_rate=0.01)]), dict(TH, flatbase_funding_above_base_block=False))
    assert R.today_recheck.iloc[0] == "" and bool(R.today_filters_pass.iloc[0])


def test_ht_burst_zone_rest_unread():
    t = BEFORE_HT
    ms = int(t.value // 1_000_000)
    cases = {"zone": _bars(ms), "rest_upper": _bars(ms, close_low=False), "rest_quiet": _bars(ms, burst=False), "unread": _bars(ms, n=40)}
    S = pd.DataFrame([dict(sleeve="FRENZY_WILLY_HT", pair=k, status="OPENED", block_class="taken", signal_time=t) for k in cases])
    R = TS.apply(S, TH, bars_for=lambda sym, m: cases[sym])
    assert list(R.today_recheck_state) == ["zone", "rest", "rest", "unread"]
    assert list(R.today_filters_pass) == [False, True, True, True]
    assert list(R.today_block) == ["WILLY_HT_BURST_REJECTION", "", "", ""]


def test_ht_bars_at_or_after_the_decision_minute_are_ignored():
    t = BEFORE_HT
    ms = int(t.value // 1_000_000)
    rows = _bars(ms, burst=False) + [[ms // M * M + i * M, 1.5, 2.0, 1.0, 1.0, 50.0] for i in range(3)]   # a burst AT / after m0
    R = TS.apply(pd.DataFrame([dict(sleeve="FRENZY_WILLY_HT", pair="X", status="OPENED", block_class="taken", signal_time=t)]), TH,
                 bars_for=lambda s, m: rows)
    assert R.today_recheck_state.iloc[0] == "rest"


def test_ht_fetch_failure_is_unknown_not_a_pass_and_blocked_rows_untouched():
    S = pd.DataFrame([dict(sleeve="FRENZY_WILLY_HT", pair="X", status="OPENED", block_class="taken", signal_time=BEFORE_HT),
                      dict(sleeve="FRENZY_WILLY_HT", pair="Y", status="BLOCKED:WILLY_FAILED_SPIKE", block_class="filter", signal_time=BEFORE_HT),
                      dict(sleeve="FRENZY_WILLY_HT", pair="Z", status="OPENED", block_class="taken", signal_time=BEFORE_HT)])
    R = TS.apply(S, TH, bars_for=lambda s, m: None if s == "X" else [])
    assert list(R.today_recheck_state) == ["nofetch", "", "unread"]          # [] = Binance had no bars → the engine's unread (fail-open)
    assert R.today_filters_pass.isna().tolist() == [True, False, False]
    assert list(R.today_filters_pass.fillna(False)) == [False, False, True]


def test_ht_first_seconds_of_a_minute_are_flagged():
    t = BEFORE_HT + pd.Timedelta(seconds=1)
    R = TS.apply(pd.DataFrame([dict(sleeve="FRENZY_WILLY_HT", pair="X", status="OPENED", block_class="taken", signal_time=t)]), TH,
                 bars_for=lambda s, m: _bars(m))
    assert "first 3 s" in R.today_recheck_value.iloc[0]


def test_nat_signal_time_is_skipped():
    R = TS.apply(pd.DataFrame([_row(signal_time=pd.NaT, journal_funding_rate=0.01)]), TH)
    assert R.today_recheck.iloc[0] == ""


class _Lim:
    def __init__(self, js, stopped=False):
        self.js, self.stopped, self.calls = js, stopped, []

    def get(self, path, params, weight):
        self.calls.append(params)
        return self.js


def _fake_sources(monkeypatch, lim):
    import types
    mod = types.ModuleType("sources")
    mod.limiter = lambda: lim
    monkeypatch.setitem(sys.modules, "sources", mod)


def test_fetch_window_cache_and_failure(monkeypatch, tmp_path):
    m0 = int(BEFORE_HT.value // 1_000_000) // M * M
    js = [[m0 - (75 - i) * M, "1", "2", "0.5", "1.5", "7", 0, "9"] for i in range(75)] + [[m0, "1", "2", "0.5", "1.5", "99", 0, "9"]]
    lim = _Lim(js)
    _fake_sources(monkeypatch, lim)
    rows = TS.fetch_bars_1m("AUSDT", m0 + 5000, m0 + 10 * M, str(tmp_path))
    assert lim.calls[0]["startTime"] == m0 - 75 * M and lim.calls[0]["endTime"] == m0 - 1
    assert len(rows) == 75 and rows[-1][0] == m0 - M and rows[0][5] == 7.0        # base volume; the bar at m0 dropped
    assert (tmp_path / f"AUSDT_{m0}.pkl").exists()                                 # 75 bars, final → cached
    _fake_sources(monkeypatch, _Lim(None))
    assert TS.fetch_bars_1m("BUSDT", m0, m0 + 10 * M, str(tmp_path)) is None       # fetch failure → None (nofetch)
    _fake_sources(monkeypatch, _Lim(js[:40]))
    assert len(TS.fetch_bars_1m("CUSDT", m0, m0 + 10 * M, str(tmp_path))) == 40
    assert not (tmp_path / f"CUSDT_{m0}.pkl").exists()                             # short + recent → not cached (may backfill)
    _fake_sources(monkeypatch, _Lim(None, stopped=True))
    assert TS.fetch_bars_1m("AUSDT", m0, m0 + 10 * M, str(tmp_path)) is not None  # cache hit needs no Binance


def test_trust_from_boundary():
    rc = [r for r in TS.RECHECKS if r.name == "WILLY_HT_BURST_REJECTION"][0]
    edge = TS.trust_from(rc)
    ms = lambda t: int(t.value // 1_000_000)
    S = pd.DataFrame([dict(sleeve="FRENZY_WILLY_HT", pair="X", status="OPENED", block_class="taken", signal_time=edge - pd.Timedelta(seconds=1)),
                      dict(sleeve="FRENZY_WILLY_HT", pair="X", status="OPENED", block_class="taken", signal_time=edge)])
    R = TS.apply(S, TH, bars_for=lambda s, m: _bars(m))
    assert list(R.today_recheck_state) == ["zone", ""]
    assert ms(edge) > ms(rc.committed)


def test_unblocked_capacity_row_passes_today():
    R = TS.apply(pd.DataFrame([_row(status="REFUSED:PAIR_HELD", block_class="capacity", journal_funding_rate=-0.001)]), TH)
    assert R.today_recheck.iloc[0] == "FLATBASE_FUNDING_ABOVE_BASE"                  # open-path pair held (funding stamped) → reached
    assert bool(R.today_filters_pass.iloc[0]) and R.today_block_class.iloc[0] == "capacity"


def test_pre_check_pair_held_not_reached():
    R = TS.apply(pd.DataFrame([_row(status="REFUSED:PAIR_HELD", block_class="capacity")]), TH)   # no funding stamp → before the check
    assert R.today_recheck.iloc[0] == ""


def test_fb_unreadable_cut_fails_open():
    R = TS.apply(pd.DataFrame([_row(journal_funding_rate=0.01)]), dict(TH, flatbase_max_entry_funding_rate=None))
    assert R.today_recheck_state.iloc[0] == "unread" and bool(R.today_filters_pass.iloc[0])


def test_fetch_not_cached_while_the_window_is_still_forming(monkeypatch, tmp_path):
    m0 = int(BEFORE_HT.value // 1_000_000) // M * M
    js = [[m0 - (75 - i) * M, "1", "2", "0.5", "1.5", "7", 0, "9"] for i in range(75)]
    _fake_sources(monkeypatch, _Lim(js))
    assert len(TS.fetch_bars_1m("AUSDT", m0, m0 + M, str(tmp_path))) == 75
    assert not (tmp_path / f"AUSDT_{m0}.pkl").exists()
