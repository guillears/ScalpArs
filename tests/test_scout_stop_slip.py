"""🧯 Oct-7: tests for the scout STOP_SLIP tracker (scripts/scout_stop_slip.py). Covered: stop-reason parsing / classification, the
slip sign for LONG / SHORT, the net-line → price inversion, the engine's stop-line formulas, the two-sided line check, the TRIGGERING
crossing from synthetic ticks, write-once freezing, the tries cap, network errors never burning a try, the 418 / 429 / used-weight stop,
the page-cap truncation, archive header sniffing + temp cleanup, the load_stops dedup, and the scout hook surviving a crash. No network:
every URL call is monkeypatched."""
import io
import os
import socket
import sys
import time
import types
import urllib.error
import zipfile

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_stop_slip as SS  # noqa: E402

T0 = 1_790_000_000_000


@pytest.fixture(autouse=True)
def _no_real_store(tmp_path, monkeypatch):
    """never the real reports/SCOUT_STOP_SLIP.csv"""
    monkeypatch.setattr(SS, "CSV", str(tmp_path / "SCOUT_STOP_SLIP.csv"))
    monkeypatch.setattr(SS, "LOCK", str(tmp_path / ".scout.lock"))
    monkeypatch.setattr(SS.time, "sleep", lambda s: None)


# ─────────────── reasons ───────────────
@pytest.mark.parametrize("reason,kind", [
    ("STOP_LOSS L1", "STOP_LOSS"), ("STOP_LOSS", "STOP_LOSS"), ("STOP_LOSS_WIDE L2", "STOP_LOSS_WIDE"),
    ("FLIP_STOP_LOSS L1", "STOP_LOSS"), ("BR_STOP_LOSS", "STOP_LOSS"),
    ("FL_STOP_LOSS L1", "FL_STOP_LOSS"), ("FL_FLIP_STOP_LOSS L1", "FL_STOP_LOSS"), ("FLIP_FL_STOP_LOSS L1", "FL_STOP_LOSS"),
    ("FL_STOP_LOSS_WIDE L1", "FL_STOP_LOSS_WIDE"),
    ("FL_EMERGENCY_SL L1", "FL_EMERGENCY_SL"), ("FL_DEEP_STOP L3", "FL_DEEP_STOP"), ("FLIP_FL_DEEP_STOP L1", "FL_DEEP_STOP"),
    ("PATTERN_FIXED_SL L1", "PATTERN_FIXED_SL"), ("SPIKE_SL", "SPIKE_SL"),
    ("RH_HARD_STOP", "RH_HARD_STOP"), ("FL_RH_HARD_STOP", "RH_HARD_STOP"),
    ("RH_STOP_LOSS L1", "STOP_LOSS"), ("FL_RH_STOP_LOSS L1", "FL_STOP_LOSS"),          # a released hold's base reason
    ("RH_PREMISE_EXIT", None), ("RH_TIME_EXIT", None), ("RH_RUNNER_TRAIL", None), ("FL_RH_TRAILING_STOP L1", None),
    ("TRAILING_STOP L1", None), ("BR_TRAILING_STOP", None), ("RUNNER_TRAIL", None), ("BREAKEVEN_EXIT_L2", None), ("SPIKE_LOCK L1", None),
    ("FL_TRAILING_STOP L1", None), ("FRENZY_TP", None), ("MANUAL_SL", None), ("REGIME_CHANGE L1", None),
    ("FL_RECOVERED L1", None), ("HARD_TP_LADDER L2", None), ("", None), (None, None), (float("nan"), None),
])
def test_classify(reason, kind):
    assert SS.classify(reason) == kind


def test_parse_reason_flag():
    assert SS.parse_reason("FL_FLIP_STOP_LOSS L2") == ("STOP_LOSS", True)
    assert SS.parse_reason("FLIP_STOP_LOSS L2") == ("STOP_LOSS", False)
    assert SS.parse_reason("FL_RH_HARD_STOP") == ("RH_HARD_STOP", True)


# ─────────────── signs / lines ───────────────
def test_slip_sign_long_short():
    assert SS.signed_pct(99.0, 100.0, "LONG") == pytest.approx(-1.0)
    assert SS.signed_pct(101.0, 100.0, "LONG") == pytest.approx(+1.0)
    assert SS.signed_pct(101.0, 100.0, "SHORT") == pytest.approx(-1.0)
    assert SS.signed_pct(99.0, 100.0, "SHORT") == pytest.approx(+1.0)
    assert np.isnan(SS.signed_pct(None, 100.0, "LONG")) and np.isnan(SS.signed_pct(1.0, 0.0, "LONG"))


@pytest.mark.parametrize("d", ["LONG", "SHORT"])
def test_line_to_price_inverts_engine_pnl(d):
    """the engine's net P&L % (check_realtime_stop_loss) at the returned price equals the line."""
    E, q, fe, fx, line = 2.5, 400.0, 0.00045, 0.00045, -0.85
    P = SS.line_to_price(E, line, d, fe, fx)
    raw = (P - E) * q if d == "LONG" else (E - P) * q
    assert (raw - E * q * fe - P * q * fx) / (E * q) * 100 == pytest.approx(line, abs=1e-9)
    assert (P < E) if d == "LONG" else (P > E)


CFG_L = {"thresholds": {"sl_atr_multiplier": 1.5, "sl_atr_widen_floor_pct": -1.2, "frenzy_stop_pct": 3.0, "bullrun_base_sl_pct": -0.7,
                        "spike_fade_sl_pct": -1.5, "fl2_deep_stop": -0.8, "fl1_wide_sl_backstop": -0.95, "surge_atr_min_pct": 1.5},
         "confidence_levels": {"STRONG_BUY": {"stop_loss": -0.7, "signal_active_sl": -1.0},
                               "WIDEBASE": {"stop_loss": -1.5, "signal_active_sl": -1.5}}}


def _ln(**r):
    return SS.stop_line(r, CFG_L)


def test_stop_line_momentum_long_and_short():
    for d in ("LONG", "SHORT"):
        assert _ln(strategy="MOMENTUM", kind="STOP_LOSS", confidence="STRONG_BUY", entry_atr_pct=0.3, direction=d)[:2] == (-0.7, 0.01)
        assert _ln(strategy="MOMENTUM", kind="STOP_LOSS", confidence="STRONG_BUY", entry_atr_pct=0.6, direction=d)[0] == pytest.approx(-0.9)
        assert _ln(strategy="MOMENTUM", kind="STOP_LOSS", confidence="STRONG_BUY", entry_atr_pct=2.0, direction=d)[0] == pytest.approx(-1.2)
    assert _ln(strategy="FLIP:FAN_RATIO_GATE", kind="STOP_LOSS", confidence="STRONG_BUY", entry_atr_pct=0.6, direction="SHORT")[0] == pytest.approx(-0.9)
    assert _ln(strategy="MOMENTUM", kind="STOP_LOSS_WIDE", confidence="STRONG_BUY", entry_atr_pct=0.3, direction="LONG")[0] == pytest.approx(-1.0)


def test_stop_line_cap_applies_to_the_whole_line():
    """engine order: widen → cap the WHOLE effective_sl at the floor (a base wider than the cap is capped too)."""
    assert _ln(strategy="MOMENTUM", kind="STOP_LOSS", confidence="WIDEBASE", entry_atr_pct=0.2, direction="LONG")[0] == pytest.approx(-1.2)
    assert _ln(strategy="MOMENTUM", kind="STOP_LOSS_WIDE", confidence="WIDEBASE", entry_atr_pct=0.2, direction="SHORT")[0] == pytest.approx(-1.2)


def test_quiet_sl_applies_to_wide_too():
    cfg = {"thresholds": {**CFG_L["thresholds"], "momentum_long_sl_atr_threshold": 0.5, "momentum_long_sl_quiet_pct": -2.0},
           "confidence_levels": CFG_L["confidence_levels"]}
    for kind in ("STOP_LOSS", "STOP_LOSS_WIDE"):
        assert SS.stop_line(dict(strategy="MOMENTUM", kind=kind, confidence="STRONG_BUY", entry_atr_pct=0.3, direction="LONG"), cfg)[0] == -2.0
        assert SS.stop_line(dict(strategy="MOMENTUM", kind=kind, confidence="STRONG_BUY", entry_atr_pct=0.3, direction="SHORT"), cfg)[0] != -2.0


def test_stop_line_other_sleeves_and_epsilons():
    assert _ln(strategy="FRENZY_WIDE", kind="STOP_LOSS", entry_atr_pct=2.0)[:2] == (-3.0, 0.0)
    assert _ln(strategy="BULLRUN_LONG", kind="STOP_LOSS", entry_atr_pct=0.6)[:2] == (pytest.approx(-0.9), 0.0)
    assert _ln(strategy="SURGE_SHORT", kind="STOP_LOSS", entry_atr_pct=0.6)[:2] == (pytest.approx(-0.9), 0.0)
    assert _ln(strategy="SURGE_SHORT", kind="STOP_LOSS", entry_atr_pct=None)[0] == pytest.approx(-1.2)   # surge_atr_min_pct 1.5 × 1.5 → cap
    assert _ln(strategy="SPIKE_FADE", kind="STOP_LOSS", entry_atr_pct=0.6)[0] == -1.5
    assert _ln(strategy="MOMENTUM", kind="PATTERN_FIXED_SL", pattern_fixed_sl_pct=-0.55)[0] == -0.55
    assert _ln(strategy="MOMENTUM", kind="FL_DEEP_STOP")[:2] == (-0.8, 0.01)
    assert _ln(strategy="MOMENTUM", kind="FL_EMERGENCY_SL")[:2] == (-0.95, 0.01)
    assert _ln(strategy="MOMENTUM", kind="RH_HARD_STOP", rh_hard_stop_pct=-1.2)[:2] == (-1.2, 0.01)
    assert _ln(strategy="MOMENTUM", kind="STOP_LOSS", confidence="UNKNOWN_LEVEL", entry_atr_pct=0.3)[0] is None


def test_line_flag_two_sided():
    assert SS.line_flag(-0.70, -0.69) == "ok"                 # at the trigger
    assert SS.line_flag(-0.69 + SS.LINE_TOL_HIGH - 1e-6, -0.69) == "ok"
    assert SS.line_flag(-0.69 + SS.LINE_TOL_HIGH + 1e-3, -0.69) == "high"
    assert SS.line_flag(-0.69 - SS.LINE_TOL_LOW + 1e-6, -0.69) == "ok"
    assert SS.line_flag(-0.69 - SS.LINE_TOL_LOW - 1e-3, -0.69) == "low"
    assert SS.line_flag(float("nan"), -0.69) == "na" and SS.line_flag(-0.7, None) == "na"


# ─────────────── crossing / measure ───────────────
def _rec(d, trig, exit_px, opened=T0, closed=T0 + 60_000, stop=None):
    return dict(direction=d, exit=exit_px, opened_ms=opened, closed_ms=closed, trig_px=trig, stop_px=trig if stop is None else stop)


def test_triggering_crossing_ignores_an_early_wick():
    """an early wick through the line that recovers, then the real stop: the crossing is the start of the LAST run → small delay."""
    t = np.array([T0 - 5_000, T0 + 1_000, T0 + 10_000, T0 + 11_000, T0 + 20_000, T0 + 59_800, T0 + 59_850, T0 + 59_900, T0 + 61_000],
                 dtype=np.int64)
    p = np.array([98.0,        100.0,       98.9,         99.5,         99.6,         98.95,        98.8,         99.0,         99.9])
    rec = _rec("LONG", trig=99.0, exit_px=98.95)
    assert SS.measure(rec, t, p) == "ok"
    assert rec["cross_px"] == 98.95 and rec["delay_s"] == pytest.approx(0.2)
    assert rec["fill_vs_cross"] == pytest.approx(0.0) and rec["late"] is False
    assert rec["last_px"] == 99.0                             # last print at or before closed_at, not the +61 s one
    assert rec["slip_last"] == pytest.approx((98.95 / 99.0 - 1) * 100)
    assert rec["slip_live"] == pytest.approx((99.0 / 99.0 - 1) * 100)
    assert rec["move_delay"] == pytest.approx((99.0 / 98.95 - 1) * 100)


def test_late_flag_and_short():
    t = np.array([T0 + 1_000, T0 + 5_000, T0 + 7_000], dtype=np.int64)
    p = np.array([100.0, 101.2, 101.5])
    rec = _rec("SHORT", trig=101.0, exit_px=101.5, closed=T0 + 8_000)
    assert SS.measure(rec, t, p) == "ok"
    assert rec["delay_s"] == pytest.approx(3.0) and rec["cross_px"] == 101.2
    assert rec["move_delay"] == pytest.approx((1 - 101.5 / 101.2) * 100) and rec["move_delay"] < 0
    assert rec["late"] is True                                # filled 0.30 worse than the crossing print
    rec2 = _rec("SHORT", trig=101.0, exit_px=101.2, closed=T0 + 12_000)
    assert SS.measure(rec2, t, p) == "ok" and rec2["late"] is True     # delay 7 s > 5 s
    rec3 = _rec("SHORT", trig=105.0, exit_px=101.5, closed=T0 + 8_000)
    assert SS.measure(rec3, t, p) == "no_cross"
    rec4 = _rec("SHORT", trig=float("nan"), exit_px=101.5, closed=T0 + 8_000)
    assert SS.measure(rec4, t, p) == "no_line" and rec4["last_px"] == 101.5


def test_cross_grace_after_closed_at():
    t = np.array([T0 + 1_000, T0 + 61_500], dtype=np.int64)
    rec = _rec("LONG", trig=99.0, exit_px=98.9)
    assert SS.measure(rec, t, np.array([100.0, 98.9])) == "ok"
    assert rec["delay_s"] == pytest.approx(-1.5)


def test_rest_candidate_minutes_are_the_last_before_close(monkeypatch):
    kl = [(T0 + i * 60_000, 101.0, 98.0 if i in (0, 1, 5, 7, 8) else 100.0) for i in range(10)]
    monkeypatch.setattr(SS, "klines_1m", lambda *a, **k: kl)
    rec = dict(pair="AAAUSDT", direction="LONG", opened_ms=T0, closed_ms=T0 + 8 * 60_000 + 30_000, trig_px=99.0)
    w = SS._windows(rec, {})
    starts = [a for a, _ in w]
    assert T0 not in starts and T0 + 60_000 not in starts     # the early crossing minutes are not read
    assert T0 + 5 * 60_000 in starts                          # minutes 5, 7, 8 (+ the last 120 s) are
    assert w[-1][1] == rec["closed_ms"] + SS.CROSS_GRACE_MS


# ─────────────── store ───────────────
def _row(k, status, slip=-0.1, tries=0, pair="AAAUSDT", d="LONG"):
    return {c: np.nan for c in SS.COLS} | dict(k=k, pair=pair, direction=d, status=status, slip_stop=slip, tries=tries)


def test_write_once_freeze():
    old = pd.DataFrame([_row("2026-10-01T00:00:00", "ok", slip=-0.2), _row("2026-10-02T00:00:00", "pending", tries=1)])
    new = [_row("2026-10-01T00:00:00", "ok", slip=-9.9), _row("2026-10-02T00:00:00", "ok", slip=-0.3),
           _row("2026-10-03T00:00:00", "no_cross", slip=-0.4)]
    m = SS.merge_store(old, new).set_index("k")
    assert len(m) == 3
    assert m.loc["2026-10-01T00:00:00", "slip_stop"] == -0.2
    assert m.loc["2026-10-02T00:00:00", "slip_stop"] == -0.3 and m.loc["2026-10-02T00:00:00", "status"] == "ok"
    assert m.loc["2026-10-03T00:00:00", "status"] == "no_cross"
    assert len(SS.merge_store(old, [_row("2026-10-01T00:00:00", "ok", slip=-1.0, d="SHORT")])) == 3   # direction is in the key


def test_tries_cap_and_budget_waits():
    assert SS.after_failure(0) == ("pending", 1)
    assert SS.after_failure(1) == ("pending", 2)
    assert SS.after_failure(2) == ("failed", 3)
    assert SS.after_budget_wait(0, 0) == ("pending", 0, 1)
    assert SS.after_budget_wait(0, SS.BUDGET_WAITS_PER_TRY - 1) == ("pending", 1, 0)
    assert SS.after_budget_wait(2, SS.BUDGET_WAITS_PER_TRY - 1) == ("failed", 3, 0)
    assert "failed" in SS.FINAL and "pending" not in SS.FINAL


def _stops_frame(n=2):
    rows = []
    for i in range(n):
        rows.append(dict(k=f"2026-10-0{i + 1}T00:00:00", opened_at=f"2026-10-0{i + 1}T00:00:00", closed_at=f"2026-10-0{i + 1}T00:05:00",
                         pair="AAAUSDT", direction="LONG", entry_strategy="FRENZY_LONG", entry_price=1.0, exit_price=0.969,
                         pnl_percentage=-3.2, close_reason="STOP_LOSS", kind="STOP_LOSS", confidence="STRONG_BUY", entry_atr_pct=2.0,
                         entry_fee=0.00045 * 100, quantity=100.0))
    return pd.DataFrame(rows)


CFG = {"taker_fee": 0.00045, "thresholds": {"frenzy_stop_pct": 3.0}, "confidence_levels": {}}
NOW = SS._ms("2026-10-07T00:00:00")


def _budget():
    return {"deadline": time.monotonic() + 30, "dl": 0}


def test_data_failures_freeze_after_three(monkeypatch):
    def fail(rec, now_ms, budget):
        raise SS.DataFail("bad")
    monkeypatch.setattr(SS, "compute", fail)
    S, old = _stops_frame(1), pd.DataFrame(columns=SS.COLS)
    for want in (("pending", 1), ("pending", 2), ("failed", 3)):
        old, _, ch = SS.update(NOW, _budget(), cfg=CFG, S=S, old=old)
        assert ch == 1 and (old.status.iloc[0], int(old.tries.iloc[0])) == want
    calls = []
    monkeypatch.setattr(SS, "compute", lambda rec, now_ms, budget: calls.append(1) or "ok")
    old2, _, ch = SS.update(NOW, _budget(), cfg=CFG, S=S, old=old)
    assert not calls and ch == 0 and old2.status.iloc[0] == "failed"


@pytest.mark.parametrize("exc", [urllib.error.URLError("dns"), socket.timeout("t"), TimeoutError("t"), ConnectionResetError("r"),
                                 "HTTP500", "HTTP503"])
def test_network_errors_never_burn_a_try(monkeypatch, exc):
    seen = []

    def boom(url, timeout=20):
        seen.append(timeout)
        if exc == "HTTP500" or exc == "HTTP503":
            raise urllib.error.HTTPError(url, int(exc[4:]), "x", {}, io.BytesIO(b""))
        raise exc
    monkeypatch.setattr(SS.urllib.request, "urlopen", boom)
    S = _stops_frame(2)
    b = _budget()
    store, notes, ch = SS.update(SS._ms("2026-10-02T06:00:00"), b, cfg=CFG, S=S, old=pd.DataFrame(columns=SS.COLS))
    assert len(seen) == 1 and ch == 0 and len(store) == 0      # run-level stop: no row stored, no try counted
    assert notes["skipped_budget"] == 2 and "NetworkStop" in notes["stopped"]
    assert seen[0] <= SS.URL_TIMEOUT_S


def test_http_4xx_counts_a_try(monkeypatch):
    def boom(url, timeout=20):
        raise urllib.error.HTTPError(url, 400, "bad symbol", {}, io.BytesIO(b""))
    monkeypatch.setattr(SS.urllib.request, "urlopen", boom)
    store, notes, ch = SS.update(SS._ms("2026-10-02T06:00:00"), _budget(), cfg=CFG, S=_stops_frame(1), old=pd.DataFrame(columns=SS.COLS))
    assert ch == 1 and store.status.iloc[0] == "pending" and int(store.tries.iloc[0]) == 1 and "400" in store.err.iloc[0]


def test_budget_exceeded_is_a_budget_wait(monkeypatch):
    def slow(rec, now_ms, budget):
        raise SS.BudgetExceeded("run budget")
    monkeypatch.setattr(SS, "compute", slow)
    old = pd.DataFrame(columns=SS.COLS)
    for i in range(SS.BUDGET_WAITS_PER_TRY):
        old, _, _ = SS.update(NOW, _budget(), cfg=CFG, S=_stops_frame(1), old=old)
    assert int(old.tries.iloc[0]) == 1 and int(old.waits.iloc[0]) == 0 and old.status.iloc[0] == "pending"


def test_unpublished_archive_costs_nothing(monkeypatch):
    def wait(rec, now_ms, budget):
        raise SS.Wait("unpublished")
    monkeypatch.setattr(SS, "compute", wait)
    store, notes, ch = SS.update(NOW, _budget(), cfg=CFG, S=_stops_frame(1), old=pd.DataFrame(columns=SS.COLS))
    assert ch == 0 and len(store) == 0 and notes["wait"] == 1


class _Resp(io.BytesIO):
    def __init__(self, body, weight=None):
        super().__init__(body)
        self.headers = {"X-MBX-USED-WEIGHT-1m": str(weight)} if weight is not None else {}


@pytest.mark.parametrize("code", [418, 429])
def test_rate_limit_stops_the_run(monkeypatch, code):
    seen = []

    def boom(url, timeout=20):
        seen.append(url)
        raise urllib.error.HTTPError(url, code, "limit", {}, io.BytesIO(b""))
    monkeypatch.setattr(SS.urllib.request, "urlopen", boom)
    b = _budget()
    store, notes, ch = SS.update(SS._ms("2026-10-02T06:00:00"), b, cfg=CFG, S=_stops_frame(2), old=pd.DataFrame(columns=SS.COLS))
    assert len(seen) == 1 and b["stop_kind"] == "rate" and ch == 0
    assert notes["skipped_budget"] == 2 and str(code) in notes["stopped"]
    with pytest.raises(SS.RateLimited):
        SS._get_json("https://example.invalid", b)
    assert len(seen) == 1


def test_used_weight_stops_the_run(monkeypatch):
    seen = []
    monkeypatch.setattr(SS.urllib.request, "urlopen", lambda url, timeout=20: seen.append(url) or _Resp(b"[]", SS.WEIGHT_STOP))
    b = _budget()
    assert SS._get_json("https://example.invalid/1", b) == []          # this response is still used
    with pytest.raises(SS.RateLimited):
        SS._get_json("https://example.invalid/2", b)
    assert len(seen) == 1


def test_page_cap_truncation_is_a_data_failure(monkeypatch):
    page = [{"a": i, "p": "1.0", "T": T0 + i} for i in range(1000)]
    monkeypatch.setattr(SS, "_get_json", lambda url, budget, pace=0: page)
    t, p, trunc = SS.aggtrades_rest("AAAUSDT", T0, T0 + 3_000_000, _budget(), pages=3)
    assert trunc is True
    monkeypatch.setattr(SS, "_windows", lambda rec, budget: [(T0, T0 + 1_000)])
    with pytest.raises(SS.DataFail):
        SS.fetch_prints(dict(pair="AAAUSDT"), T0 + 60_000, _budget())


def test_budget_timeout_is_capped_by_time_left(monkeypatch):
    seen = []
    monkeypatch.setattr(SS.urllib.request, "urlopen", lambda url, timeout=20: seen.append(timeout) or _Resp(b"[]"))
    SS._get_json("https://example.invalid", {"deadline": time.monotonic() + 3})
    assert seen[0] <= 3
    with pytest.raises(SS.BudgetExceeded):
        SS._get_json("https://example.invalid", {"deadline": time.monotonic() - 1})


# ─────────────── archive ───────────────
def _zip(tmp_path, header):
    rows = "\n".join(f"{i},{100 + i / 10},1,{i},{i},{T0 + i * 1000},false" for i in range(10))
    body = ("agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n" if header else "") + rows + "\n"
    f = tmp_path / f"a{int(header)}.zip"
    with zipfile.ZipFile(f, "w") as z:
        z.writestr("X-aggTrades.csv", body)
    return str(f)


@pytest.mark.parametrize("header", [True, False])
def test_arch_slice_header_detection(tmp_path, header):
    t, p = SS._arch_slice(_zip(tmp_path, header), T0 + 2_000, T0 + 4_000)
    assert list(t) == [T0 + 2_000, T0 + 3_000, T0 + 4_000] and list(p) == pytest.approx([100.2, 100.3, 100.4])


def test_arch_slice_checks_the_deadline(tmp_path):
    with pytest.raises(SS.BudgetExceeded):
        SS._arch_slice(_zip(tmp_path, True), T0, T0 + 4_000, {"deadline": time.monotonic() - 1})


def test_archive_download_temp_removed_on_failure(monkeypatch, tmp_path):
    made = []
    real = SS.tempfile.mkstemp

    def mk(**kw):
        fd, p = real(dir=str(tmp_path), **kw)
        made.append(p)
        return fd, p

    class Broken(_Resp):
        def read(self, n=-1):
            raise ConnectionResetError("reset")
    monkeypatch.setattr(SS.tempfile, "mkstemp", mk)
    monkeypatch.setattr(SS.urllib.request, "urlopen", lambda url, timeout=20: Broken(b""))
    b = {"deadline": time.monotonic() + 30, "dl": 2}
    with pytest.raises(SS.NetworkStop):
        SS._download_archive("AAAUSDT", "2026-09-01", SS._ms("2026-09-10T00:00:00"), b)
    assert made and not os.path.exists(made[0]) and ("AAAUSDT", "2026-09-01") not in SS._ARCH


def test_archive_404_waits_then_missing(monkeypatch):
    monkeypatch.setattr(SS.urllib.request, "urlopen",
                        lambda url, timeout=20: (_ for _ in ()).throw(urllib.error.HTTPError(url, 404, "nf", {}, io.BytesIO(b""))))
    with pytest.raises(SS.Wait):
        SS._download_archive("AAAUSDT", "2026-09-01", SS._ms("2026-09-03T00:00:00"), {"deadline": time.monotonic() + 30, "dl": 2})
    with pytest.raises(SS.DataFail):
        SS._download_archive("AAAUSDT", "2026-09-01", SS._ms("2026-09-20T00:00:00"), {"deadline": time.monotonic() + 30, "dl": 2})
    with pytest.raises(SS.Wait) as w:
        SS._download_archive("AAAUSDT", "2026-09-01", SS._ms("2026-09-20T00:00:00"), {"deadline": time.monotonic() + 30, "dl": 0})
    assert w.value.kind == "budget"


# ─────────────── orders ───────────────
def test_load_stops_dedup(tmp_path):
    dl, rep = tmp_path / "dl", tmp_path / "rep"
    dl.mkdir(); rep.mkdir()
    base = dict(pair="AAAUSDT", direction="LONG", status="CLOSED", entry_strategy="MOMENTUM", entry_price=1.0, exit_price=0.99,
                pnl_percentage=-0.7, close_reason="STOP_LOSS L1", closed_at="2026-10-01 00:05:00")
    pd.DataFrame([dict(base, id=1, opened_at="2026-10-01 00:00:00", exit_price=0.5),              # master pool, lowest priority
                  dict(base, id=7, opened_at="2026-09-01 00:00:00")]).to_csv(rep / "MASTER_POOL_stacked.csv", index=False)
    pd.DataFrame([dict(base, id=99, opened_at="2026-10-01T00:00:00.123456", exit_price=0.98)]).to_csv(
        dl / "scalpars_orders_paper_2026-10-01_10-00-00.csv", index=False)
    pd.DataFrame([dict(base, id=5, opened_at="2026-10-01T00:00:00.123456", exit_price=0.97),       # newest export wins
                  dict(base, id=7, opened_at="2026-10-02T00:00:00", status="OPEN"),             # same id, other trade: OPEN → out
                  dict(base, id=8, opened_at="2026-10-03T00:00:00", entry_strategy="MANUAL"),   # MANUAL → out
                  dict(base, id=9, opened_at="2026-10-04T00:00:00", close_reason="TRAILING_STOP L1"),   # not a stop
                  dict(base, id=10, opened_at="2026-10-05T00:00:00", direction="SHORT")]).to_csv(
        dl / "scalpars_orders_paper_2026-10-05_10-00-00.csv", index=False)
    S = SS.load_stops(dl=str(dl), reports=str(rep)).set_index("k")
    assert sorted(S.index) == ["2026-09-01T00:00:00", "2026-10-01T00:00:00", "2026-10-05T00:00:00"]
    assert S.loc["2026-10-01T00:00:00", "exit_price"] == 0.97
    assert "id" not in S.columns                              # id is never read, let alone used as a key


# ─────────────── report / run / hook ───────────────
def test_report_lines_render():
    now = SS._ms("2026-10-07T00:00:00")
    rows = []
    for i, (d, sl, late, flag, path) in enumerate([("LONG", -0.05, False, "ok", "realtime"), ("SHORT", -0.30, True, "ok", "realtime"),
                                                   ("LONG", 0.01, False, "ok", "realtime"), ("LONG", -0.4, False, "low", "realtime"),
                                                   ("LONG", -0.2, False, "ok", "fl")]):
        r = _row(f"2026-10-0{i + 1}T00:00:00", "ok", slip=sl, d=d)
        r.update(closed_at=f"2026-10-0{i + 1}T00:05:00", sleeve="MOMENTUM", line_flag=flag, late=late, path=path, slip_last=sl - 0.1,
                 slip_live=sl + 0.05, delay_s=1.5, move_delay=-0.02, close_reason="STOP_LOSS L1", line_pct=-0.7)
        rows.append(r)
    df = pd.DataFrame(rows)
    txt = "\n".join(SS.lines(df, now, {"done": 5}))
    assert "| **ALL** | 5 · 2 · 5 |" in txt and "| LONG |" in txt and "| SHORT |" in txt and "Worst 5" in txt
    assert "- late (" in txt and "real gap-through: 1" in txt and "FL_* stops (polling + realtime paths): 1" in txt
    st = SS.group_stats(df)
    assert st["extra_paper"] == pytest.approx(-np.mean([-0.05, 0.01]) - SS.ASSUMED_SLIP)          # clean rows only
    assert st["extra_live"] == pytest.approx(-np.mean([0.0, -0.25, 0.06, -0.35, -0.15]) - SS.ASSUMED_SLIP)  # live proxy: every measured row whose line is not 'high'


def test_run_dry_run_never_writes(monkeypatch, tmp_path):
    p = tmp_path / "store.csv"
    monkeypatch.setattr(SS, "update", lambda now_ms, budget, old=None: (pd.DataFrame([_row("2026-10-01T00:00:00", "ok")]), {}, 1))
    SS.run(store_path=str(p), dry_run=True)
    assert not p.exists()
    SS.run(store_path=str(p))
    assert p.exists() and len(SS.load_store(str(p))) == 1
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_main_refuses_when_the_scout_holds_the_lock(monkeypatch, tmp_path):
    with open(SS.LOCK, "w") as fh:
        fh.write("12345")
    monkeypatch.setattr(SS, "run", lambda **k: pytest.fail("must not run"))
    assert SS.main(["--store", str(tmp_path / "s.csv")]) == 1
    called = []
    monkeypatch.setattr(SS, "run", lambda **k: called.append(k) or ["x"])
    assert SS.main(["--dry-run", "--store", str(tmp_path / "s.csv")]) == 0 and called[0]["dry_run"] is True


def test_scout_hook_survives_a_crash(monkeypatch):
    """the opportunity_scout.py STOP_SLIP block: a run() that raises leaves an 'Unavailable' section and the scout goes on."""
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read().splitlines()
    i = next(n for n, x in enumerate(src) if "import scout_stop_slip as _ss" in x)
    s = max(n for n in range(i) if src[n].strip().startswith("try:"))
    e = next(n for n in range(i, len(src)) if "Stop-exit slippage" in src[n])
    block = "\n".join(x[4:] for x in src[s:e + 1])
    fake = types.ModuleType("scout_stop_slip")
    fake.run = lambda now_ms: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "scout_stop_slip", fake)
    logs = []
    ns = {"os": os, "sys": sys, "_rg_sec": [], "now_ms": 0, "log": logs.append, "__file__": os.path.join(HERE, "scripts", "x.py")}
    exec(block, ns)
    assert ns["_rg_sec"][0] == "## 🧯 Stop-exit slippage" and "boom" in ns["_rg_sec"][2] and logs


def test_recross_after_close_does_not_make_a_negative_delay():
    """trigger print, a bounce, then a new run inside the clock grace: the crossing stays the run that started by closed_at."""
    t = np.array([T0 + 1_000, T0 + 59_900, T0 + 59_950, T0 + 61_000], dtype=np.int64)
    p = np.array([100.0, 98.9, 99.5, 98.8])
    rec = _rec("LONG", trig=99.0, exit_px=98.9)
    assert SS.measure(rec, t, p) == "ok" and rec["cross_px"] == 98.9 and rec["delay_s"] == pytest.approx(0.1)
