"""🏷 Oct-19 (DECISION_LOG 311) SPIKE_FADE entry market-cap stamp — OBSERVE-ONLY.

The open path stamps entry_mcap_usd / entry_cmc_rank / entry_vol_mcap_ratio from the in-memory cache (no network). A SPIKE_* fill whose
pair is not cached (KMNOUSDT 10-17 fade → NULL) gets ONE post-fill background backfill through the shared one-pair fetch
(services.mcap_service.fetch_pair_now): never awaited on the order path, never resizes / reprices the booked row, any failure → NULL.
"""
import asyncio
import datetime as dt
import os
import time

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENG = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()


def _fade_row(**kw):
    import models
    base = dict(pair="KMNOUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="SPIKE_FADE", confidence="STRONG_SELL",
                entry_price=0.0712, quantity=14044.0, investment=50.0, leverage=20, notional_value=1000.0,
                opened_at=dt.datetime(2026, 10, 17, 9, 30, 0), entry_pair_volume_24h_usd=166e6)
    base.update(kw)
    return models.Order(**base)


def _with_db(monkeypatch, rows):
    """an in-memory DB holding `rows`, wired as the engine's AsyncSessionLocal → (go(coro_fn), read(id))."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    import services.trading_engine as TE

    async def run(fn):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        try:
            async with eng.begin() as c:
                await c.run_sync(models.Base.metadata.create_all)
            SL = async_sessionmaker(eng, expire_on_commit=False)
            monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
            async with SL() as s:
                for r in rows:
                    s.add(r)
                await s.commit()
                ids = [r.id for r in rows]
            out = await fn(ids)
            async with SL() as s:
                got = [await s.get(models.Order, i) for i in ids]
            return out, got
        finally:
            await eng.dispose()
    return run


def _engine():
    import services.trading_engine as TE
    return object.__new__(TE.TradingEngine)


# ── engine: post-fill backfill ─────────────────────────────────────────────────────────────────────────────────────────────────────

def test_missing_cap_backfills_the_row_and_leaves_size_and_price(monkeypatch):
    from services import mcap_service as M
    asked = []

    async def fetch(pair, timeout_s=20.0):
        asked.append(pair)
        return 332e6, 287
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()

    async def fn(ids):
        assert e._spike_mcap_schedule(_fade_row(id=ids[0])) is True        # scheduled, not awaited
        await asyncio.gather(*list(e._spike_mcap_tasks))
        return None
    _, (row,) = asyncio.run(_with_db(monkeypatch, [_fade_row()])(fn))
    assert asked == ["KMNOUSDT"]
    assert row.entry_mcap_usd == 332e6 and row.entry_cmc_rank == 287
    assert abs(row.entry_vol_mcap_ratio - 166e6 / 332e6) < 1e-6
    assert (row.entry_price, row.quantity, row.investment, row.leverage, row.notional_value, row.status) == (0.0712, 14044.0, 50.0, 20, 1000.0, "OPEN")


def test_server_default_opened_at_row_is_stamped(monkeypatch):
    """deep review (311): production rows take opened_at from the server default (stored WITHOUT microseconds) and the engine passes the
    REFRESHED datetime — an equality guard never matched (rowcount 0, silent). The ±1 s window must stamp it."""
    from services import mcap_service as M
    import models
    import services.trading_engine as TE

    async def fetch(pair, timeout_s=20.0):
        return 332e6, 287
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()

    async def fn(ids):
        async with TE.AsyncSessionLocal() as s:                            # the row as open_position sees it after db.refresh
            o = await s.get(models.Order, ids[0])
            assert o.opened_at is not None
        assert e._spike_mcap_schedule(o) is True
        res = await asyncio.gather(*list(e._spike_mcap_tasks))
        return res
    res, (row,) = asyncio.run(_with_db(monkeypatch, [_fade_row(opened_at=None)])(fn))
    assert res == [True] and row.entry_mcap_usd == 332e6 and row.entry_cmc_rank == 287


def test_cached_cap_needs_no_backfill_and_no_request(monkeypatch):
    from services import mcap_service as M
    asked = []

    async def fetch(pair, timeout_s=20.0):
        asked.append(pair); return 1.0, 1
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()
    assert e._spike_mcap_schedule(_fade_row(id=7, entry_mcap_usd=332e6)) is False                     # stamped at the open from cache
    assert e._spike_mcap_schedule(_fade_row(id=8, entry_strategy="MOMENTUM")) is False                # other sleeves untouched
    assert e._spike_mcap_schedule(_fade_row(id=None)) is False
    assert asked == []


@pytest.mark.parametrize("strat", ["SPIKE_FADE", "SPIKE_CHASE", "SPIKE_BOUNCE"])
def test_all_spike_species_schedule(monkeypatch, strat):
    from services import mcap_service as M

    async def fetch(pair, timeout_s=20.0):
        return 5e8, 120
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()

    async def fn(ids):
        assert e._spike_mcap_schedule(_fade_row(id=ids[0], entry_strategy=strat)) is True
        await asyncio.gather(*list(e._spike_mcap_tasks))
    _, (row,) = asyncio.run(_with_db(monkeypatch, [_fade_row(entry_strategy=strat)])(fn))
    assert row.entry_mcap_usd == 5e8


@pytest.mark.parametrize("mode", ["none", "raise", "timeout"])
def test_provider_failure_leaves_null_and_never_raises(monkeypatch, mode):
    from services import mcap_service as M

    async def fetch(pair, timeout_s=20.0):
        if mode == "raise":
            raise RuntimeError("provider down")
        if mode == "timeout":
            raise asyncio.TimeoutError()
        return None, None
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()

    async def fn(ids):
        return await e._spike_mcap_backfill(ids[0], "KMNOUSDT", dt.datetime(2026, 10, 17, 9, 30, 0), 166e6, "SPIKE_FADE")
    ok, (row,) = asyncio.run(_with_db(monkeypatch, [_fade_row()])(fn))
    assert ok is False and row.entry_mcap_usd is None and row.entry_cmc_rank is None and row.entry_vol_mcap_ratio is None
    assert (row.entry_price, row.quantity, row.investment, row.status) == (0.0712, 14044.0, 50.0, "OPEN")


def test_backfill_never_overwrites_or_hits_a_reused_id(monkeypatch):
    from services import mcap_service as M

    async def fetch(pair, timeout_s=20.0):
        return 9e9, 9
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()

    async def fn(ids):
        a = await e._spike_mcap_backfill(ids[0], "KMNOUSDT", dt.datetime(2026, 10, 17, 9, 30, 0), 1e6, "SPIKE_FADE")   # already stamped
        b = await e._spike_mcap_backfill(ids[1], "KMNOUSDT", dt.datetime(2026, 10, 17, 9, 30, 0), 1e6, "SPIKE_FADE")   # id reused after a reset
        return a, b
    (a, b), (r0, r1) = asyncio.run(_with_db(monkeypatch, [_fade_row(entry_mcap_usd=332e6),
                                                         _fade_row(opened_at=dt.datetime(2026, 10, 18, 1, 0, 0))])(fn))
    assert (a, b) == (False, False) and r0.entry_mcap_usd == 332e6 and r1.entry_mcap_usd is None


def test_schedule_returns_immediately_while_the_fetch_hangs(monkeypatch):
    """no blocking: the open path's call returns at once even when the provider never answers."""
    from services import mcap_service as M
    gate = None

    async def fetch(pair, timeout_s=20.0):
        await gate.wait(); return None, None
    monkeypatch.setattr(M, "fetch_pair_now", fetch)
    e = _engine()

    async def go():
        nonlocal gate
        gate = asyncio.Event()
        t0 = time.monotonic()
        assert e._spike_mcap_schedule(_fade_row(id=1)) is True
        dt_s = time.monotonic() - t0
        pend = [t for t in e._spike_mcap_tasks if not t.done()]
        gate.set(); await asyncio.gather(*pend)
        return dt_s, len(pend)
    dt_s, n = asyncio.run(go())
    assert dt_s < 0.05 and n == 1


def test_replay_never_schedules(monkeypatch):
    monkeypatch.setenv("SCALPARS_REPLAY", "1")
    assert _engine()._spike_mcap_schedule(_fade_row(id=1)) is False


# ── mcap_service.fetch_pair_now: cache first, shared bounded one-pair fetch ────────────────────────────────────────────────────────

def _svc(monkeypatch, enabled=True):
    from services import mcap_service as M
    import config as C
    monkeypatch.setattr(M, "_cache", {}); monkeypatch.setattr(M, "_one_req", {}); monkeypatch.setattr(M, "_one_inflight", {})
    monkeypatch.setattr(C.trading_config, "mcap_fetch_enabled", enabled, raising=False)
    monkeypatch.delenv("SCALPARS_REPLAY", raising=False)

    async def _nosave():
        return False
    monkeypatch.setattr(M, "save_persisted", _nosave)
    return M


@pytest.mark.mcap_real
def test_fetch_now_uses_cache_without_a_request(monkeypatch):
    M = _svc(monkeypatch)
    calls = []

    async def fp(client, pair):
        calls.append(pair); return 1.0, 1
    monkeypatch.setattr(M, "_fetch_pair", fp)
    M._cache["KMNOUSDT"] = (332e6, 287, time.time())
    assert asyncio.run(M.fetch_pair_now("KMNOUSDT")) == (332e6, 287) and calls == []


@pytest.mark.mcap_real
def test_fetch_now_fills_the_shared_cache_and_dedupes(monkeypatch):
    M = _svc(monkeypatch)
    calls = []

    async def fp(client, pair):
        calls.append(pair); await asyncio.sleep(0.01); return 332e6, 287
    monkeypatch.setattr(M, "_fetch_pair", fp)

    async def go():
        a, b = await asyncio.gather(M.fetch_pair_now("KMNOUSDT"), M.fetch_pair_now("KMNOUSDT"))   # one in-flight fetch, both wait on it
        return a, b
    a, b = asyncio.run(go())
    assert a == b == (332e6, 287) and calls == ["KMNOUSDT"] and M.get("KMNOUSDT") == (332e6, 287)
    assert M.request_pair("KMNOUSDT") is False                                                    # fresh now → WILLY's path asks nothing


@pytest.mark.mcap_real
def test_fetch_now_timeout_and_failure_and_off_give_none(monkeypatch):
    M = _svc(monkeypatch)

    async def hang(client, pair):
        await asyncio.sleep(5); return 1.0, 1
    monkeypatch.setattr(M, "_fetch_pair", hang)

    async def go():
        t0 = time.monotonic()
        r = await M.fetch_pair_now("AAAUSDT", timeout_s=0.2)
        el = time.monotonic() - t0
        for t in list(M._one_tasks):
            t.cancel()
        return r, el
    r, el = asyncio.run(go())
    assert r == (None, None) and el < 1.0                                                          # bounded wait

    async def boom(client, pair):
        raise RuntimeError("down")
    monkeypatch.setattr(M, "_fetch_pair", boom)
    assert asyncio.run(M.fetch_pair_now("BBBUSDT")) == (None, None)
    assert asyncio.run(M.fetch_pair_now("BBBUSDT")) == (None, None)                                # deduped 10 min: no hammering

    M = _svc(monkeypatch, enabled=False)
    calls = []

    async def fp(client, pair):
        calls.append(pair); return 1.0, 1
    monkeypatch.setattr(M, "_fetch_pair", fp)
    assert asyncio.run(M.fetch_pair_now("CCCUSDT")) == (None, None) and calls == []               # fetching OFF → no request


# ── wiring (source grep) ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_wiring_post_fill_never_on_the_order_path():
    i = ENG.index("    async def open_position("); j = ENG.index("\n    @serialized_per_pair", i)
    body = ENG[i:j]
    s = body.index("            self._spike_mcap_schedule(order)")
    assert "await self._spike_mcap_schedule" not in ENG                                           # never awaited
    assert body.index("        await locked_commit(db)\n        _st_fill = _LIVE_FILL_CV.get()") < s                     # after the booking commit
    assert body.index("        await db.refresh(order)") < s
    assert "if spike_fade or spike_chase_probe or spike_bounce:" in body
    assert "fetch_pair_now" not in body and "request_pair" not in body                            # no network on the open path itself
    assert "_mcap_usd, _cmc_rank = _mcs.get(pair)" in body                                        # the cache stamp stays first
    # observe-only: no line touching the backfill feeds a gate / size / signal
    k = ENG.index("    def _spike_mcap_schedule("); k2 = ENG.index("    async def _rtp_arm(", k)
    for ln in ENG[k:k2].splitlines():
        if not ln.strip().startswith("#"):
            assert not any(x in ln for x in ("_record_filter_block", "investment", "leverage", "quantity", "entry_price")), ln


def test_d12_spike_fires_mcap_on_every_surface():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "\"mcap_usd\": getattr(o, 'entry_mcap_usd', None)" in main
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert ui.count("spikeMcapTxt(r.mcap_usd") == 3                                               # UI row + both text exports
    assert ui.count("${'RSImx'.padStart(6)} ${'Mcap'.padStart(8)}") == 2
    assert 'title="🏷 (DECISION_LOG 311) entry market cap' in ui and "function spikeMcapTxt(" in ui
    import models
    assert {"entry_mcap_usd", "entry_cmc_rank", "entry_vol_mcap_ratio"} <= {c.name for c in models.Order.__table__.columns}   # CSV rides the columns
