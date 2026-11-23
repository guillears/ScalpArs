"""🩺 DECISION_LOG 629 — WILLY's red-bar turnover read waits (bounded) for a cap that EXISTS but has not landed yet.

The bug (11-22 BULLA, 16 of 29 TURNOVER_UNREAD refusals 10-23 → 11-22): the arm fires a background one-pair cap fetch; when the red bar
closes on the same pass (or before the fetch lands) the cap is still missing → "turnover unreadable" → refused on
its designed entry bar (all 16 were judged 1–4 bars later — delayed, not lost). The fix: one bounded wait (4 s) on the SAME shared one-pair fetch, then re-read. A coin the
source has no cap for (ELSA, BIRB …) stays unreadable → refused, as before. The waits of one serial FRENZY pass share one budget.
No network (mcap_service stubbed; the mcap_real test fakes the endpoint call)."""
import asyncio
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_willy_nomiss import LATE, _cfg, _engine, _run  # noqa: E402


def _stub_mcap(monkeypatch, lands):
    """mcap_service with an empty cache whose one-pair fetch puts `lands` (or nothing) into the cache. → the list of fetch calls."""
    from services import mcap_service as M
    cache, calls = {}, []
    monkeypatch.setattr(M, "get", lambda pair: (cache.get(pair), None))

    async def fetch(pair, timeout_s=20.0):
        calls.append((pair, timeout_s))
        if lands is not None:
            cache[pair] = lands
        return cache.get(pair), None
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    return calls


def _eng_cap(TE):
    """the real _willy_turnover (vol / the cached cap) on a bare engine."""
    return object.__new__(TE.TradingEngine)


def test_cap_lands_during_the_wait_is_read(monkeypatch):
    import services.trading_engine as TE
    calls = _stub_mcap(monkeypatch, lands=2.5e7)
    R = asyncio.run(_eng_cap(TE)._willy_turnover_read("BULLAUSDT", 4.9e7))
    assert R == round(4.9e7 / 2.5e7, 6) and calls == [("BULLAUSDT", 4.0)]   # ONE bounded wait (4 s), then the re-read


def test_no_cap_at_the_source_stays_unreadable(monkeypatch):
    import services.trading_engine as TE
    calls = _stub_mcap(monkeypatch, lands=None)
    assert asyncio.run(_eng_cap(TE)._willy_turnover_read("ELSAUSDT", 9.5e7)) is None and len(calls) == 1


def test_readable_cap_never_waits(monkeypatch):
    import services.trading_engine as TE
    from services import mcap_service as M
    calls = _stub_mcap(monkeypatch, lands=None)
    monkeypatch.setattr(M, "get", lambda pair: (3e7, 100))
    assert asyncio.run(_eng_cap(TE)._willy_turnover_read("XVGUSDT", 1.5e7)) == 0.5 and calls == []


def test_missing_volume_does_not_wait_on_a_cached_cap(monkeypatch):
    import services.trading_engine as TE
    from services import mcap_service as M
    calls = _stub_mcap(monkeypatch, lands=None)
    monkeypatch.setattr(M, "get", lambda pair: (3e7, 100))
    assert asyncio.run(_eng_cap(TE)._willy_turnover_read("XVGUSDT", None)) is None and calls == []   # vol unreadable: nothing to wait for


def test_fetch_crash_is_unreadable_never_raises(monkeypatch):
    import services.trading_engine as TE
    from services import mcap_service as M
    _stub_mcap(monkeypatch, lands=None)

    async def boom(pair, timeout_s=20.0):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(M, "fetch_pair_now", boom)
    assert asyncio.run(_eng_cap(TE)._willy_turnover_read("BULLAUSDT", 4.9e7)) is None


def test_red_bar_routes_on_the_landed_cap(monkeypatch):
    """end to end through _frenzy_willy_eval: the red bar's cap lands during the wait → judged on R (here ≥ 1 → HT), never UNREAD."""
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_max_vol_mcap_ratio=1.0, frenzy_willy_ht_enabled=True, frenzy_willy_ht_max_slots=1)
    _stub_mcap(monkeypatch, lands=2.5e7)
    e = _engine(TE)
    e.__dict__.pop("_willy_turnover", None)               # the real read (the helper's default stub is removed)
    o, _ = _run(e, "g" * LATE + "r", vol=5e7)              # R = 2.0
    assert "FRENZY_WILLY_TURNOVER_UNREAD" not in e.blocks and e.blocks == ["FRENZY_WILLY_TURNOVER"] and [x["ht"] for x in o] == [True]


def test_red_bar_without_a_cap_is_refused_as_before(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_max_vol_mcap_ratio=1.0, frenzy_willy_ht_enabled=True, frenzy_willy_ht_max_slots=1)
    _stub_mcap(monkeypatch, lands=None)
    e = _engine(TE)
    e.__dict__.pop("_willy_turnover", None)
    o, _ = _run(e, "g" * LATE + "r", vol=5e7)
    assert o == [] and e.blocks == ["FRENZY_WILLY_TURNOVER_UNREAD"] and "BTRUSDT" in e._fz_willy_pending   # stays armed


def test_missing_volume_and_cap_never_waits(monkeypatch):
    import services.trading_engine as TE
    calls = _stub_mcap(monkeypatch, lands=2.5e7)
    assert asyncio.run(_eng_cap(TE)._willy_turnover_read("XVGUSDT", None, 1)) is None and calls == []   # the volume is the missing half


def test_waits_share_one_budget_per_pass(monkeypatch):
    """the FRENZY pass is serial: the red-bar cap waits of ONE pass share WILLY_CAP_WAIT_PASS_S; a new pass gets a fresh budget."""
    import services.trading_engine as TE
    from services import mcap_service as M
    monkeypatch.setattr(M, "get", lambda pair: (None, None))
    calls = []

    async def slow(pair, timeout_s=20.0):
        calls.append(round(timeout_s, 2))
        await asyncio.sleep(timeout_s)           # never lands: the whole wait is spent
        return None, None
    monkeypatch.setattr(M, "fetch_pair_now", slow)
    monkeypatch.setattr(TE.TradingEngine, "WILLY_CAP_WAIT_S", 0.2)
    monkeypatch.setattr(TE.TradingEngine, "WILLY_CAP_WAIT_PASS_S", 0.3)
    e = _eng_cap(TE)

    async def go():
        return [await e._willy_turnover_read(p, 5e7, 1000) for p in ("A1USDT", "A2USDT", "A3USDT")] + \
               [await e._willy_turnover_read("A4USDT", 5e7, 2000)]
    assert asyncio.run(go()) == [None] * 4
    assert len(calls) == 3 and calls[0] == 0.2 and calls[1] <= 0.11 and calls[2] == 0.2   # 0.2 + ≈0.1 left, 3rd skipped, new pass 0.2


@pytest.mark.mcap_real
def test_real_fetch_path_reuses_the_arm_task_and_reads_the_cap(monkeypatch):
    """the REAL mcap_service path: the arm's request_pair starts the one-pair fetch; the red-bar read joins that in-flight task (no second
    request) and reads the cap once it lands inside the wait. No network: the endpoint call (_fetch_pair) is faked."""
    import services.trading_engine as TE
    from services import mcap_service as M
    monkeypatch.setattr(M, "_enabled", lambda: True)
    monkeypatch.setattr(M, "save_persisted", lambda: asyncio.sleep(0))
    monkeypatch.setattr(M, "_cache", {}); monkeypatch.setattr(M, "_one_req", {}); monkeypatch.setattr(M, "_one_inflight", {})
    hits = []

    async def fake_fetch(client, pair):
        hits.append(pair)
        await asyncio.sleep(0.1)
        return 2.5e7, 500
    monkeypatch.setattr(M, "_fetch_pair", fake_fetch)
    e = _eng_cap(TE)

    async def go():
        assert M.request_pair("BULLAUSDT") is True              # the arm (fire-and-forget)
        return await e._willy_turnover_read("BULLAUSDT", 4.9e7, 1)   # the red bar, same pass
    assert asyncio.run(go()) == round(4.9e7 / 2.5e7, 6) and hits == ["BULLAUSDT"]


def test_wait_outcome_is_noted_for_the_unread_reason(monkeypatch):
    import services.trading_engine as TE
    _stub_mcap(monkeypatch, lands=None)
    e = _eng_cap(TE)
    asyncio.run(e._willy_turnover_read("ELSAUSDT", 9.5e7, 1))
    assert "no cap" in e._willy_cap_note["ELSAUSDT"]
    asyncio.run(e._willy_turnover_read("XVGUSDT", None, 1))
    assert e._willy_cap_note["XVGUSDT"] == "24 h volume unreadable"
