"""🩹 DECISION_LOG 593 — a fill opened before the first scan after a restart gets breadth / global volume from the FIRST scan after it
(bounded wait), never the 0 % / 1.0 defaults; timeout → {} (NULL, as before)."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import services.trading_engine as TE  # noqa: E402


def _eng():
    e = TE.TradingEngine.__new__(TE.TradingEngine)
    e._scan_pair_meta = {"EVAAUSDT": (25e6, 120, 400.0)}
    return e


def test_waits_for_first_scan_after_the_fill(monkeypatch):
    e = _eng()
    clock = {"t": 1000.0}
    monkeypatch.setattr(TE._leash_time, "time", lambda: clock["t"])
    monkeypatch.setattr(TE, "_market_stamps_at", 900.0)          # last scan BEFORE the fill (the pre-restart defaults)
    monkeypatch.setattr(TE, "_market_stamps_snap", (900.0, 0.0, 0.0, 1.0))
    monkeypatch.setattr(TE, "_market_bull_pct", 0.0)
    monkeypatch.setattr(TE, "_market_bear_pct", 0.0)
    monkeypatch.setattr(TE, "_global_volume_ratio", 1.0)

    async def fake_sleep(_s):                                      # the scan publishes on the 3rd poll
        clock["t"] += 5.0
        if clock["t"] >= 1015.0:
            TE._market_stamps_at = clock["t"]
            TE._market_stamps_snap = (clock["t"], 12.5, 80.0, 0.5804)
            TE._market_bull_pct = 0.0                              # a later partial reset of the globals must not leak in
    monkeypatch.setattr(TE.asyncio, "sleep", fake_sleep)
    out = asyncio.run(e._await_first_scan_stamps("EVAAUSDT", "DISTRIBUTION_SHORT", 1000.0, wait_s=300, poll_s=5))
    assert out["entry_global_volume_ratio"] == 0.5804 and out["entry_bull_pct"] == 12.5 and out["entry_bear_pct"] == 80.0
    assert out["entry_pair_volume_24h_usd"] == 25e6 and out["entry_pair_age_days"] == 400.0


def test_timeout_leaves_null(monkeypatch):
    e = _eng()
    clock = {"t": 1000.0}
    monkeypatch.setattr(TE._leash_time, "time", lambda: clock["t"])
    monkeypatch.setattr(TE, "_market_stamps_at", 900.0)

    async def fake_sleep(_s):
        clock["t"] += 5.0
    monkeypatch.setattr(TE.asyncio, "sleep", fake_sleep)
    assert asyncio.run(e._await_first_scan_stamps("EVAAUSDT", "DISTRIBUTION_SHORT", 1000.0, wait_s=30, poll_s=5)) == {}


def test_scan_already_published_after_fill_returns_at_once(monkeypatch):
    e = _eng()
    monkeypatch.setattr(TE._leash_time, "time", lambda: 1100.0)
    monkeypatch.setattr(TE, "_market_stamps_at", 1050.0)
    monkeypatch.setattr(TE, "_market_stamps_snap", (1050.0, 20.0, 70.0, 0.7))

    async def no_sleep(_s):
        raise AssertionError("must not wait")
    monkeypatch.setattr(TE.asyncio, "sleep", no_sleep)
    out = asyncio.run(e._await_first_scan_stamps("OTHERUSDT", "EXHAUSTION_SHORT", 1000.0))
    assert out == {"entry_bull_pct": 20.0, "entry_bear_pct": 70.0, "entry_global_volume_ratio": 0.7}   # pair not in the scan → no pair meta


def test_wiring():
    src = open(TE.__file__).read()
    assert "snap['_await_scan_since'] = now_s" in src and "_aw = st.pop('_await_scan_since', None)" in src
    assert "_market_stamps_at = _leash_time.time()" in src and "_market_stamps_snap = (_market_stamps_at" in src
    assert "if (now_s - (g.get('_market_stamps_at') or 0.0)) > 1800:" in src and "_qs2 = _calculate_quality_score" in src
