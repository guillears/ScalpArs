"""🛡 Go-live ① (DECISION_LOG 292): every order records the exchange backstop actually placed (backstop_pct / backstop_trigger_price); a
config change that moves an open order's configured backstop re-places it (live: place new → cancel old; paper: the record only).
All exchange calls are mocked — no network."""
import asyncio, os, sys
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import backstop_trigger_price, backstop_needs_replace, algo_status_resting, BACKSTOP_REPLACE_RETRY_S


def test_trigger_and_replace_rule():
    assert abs(backstop_trigger_price(100.0, "LONG", 2.5) - 97.5) < 1e-9 and abs(backstop_trigger_price(100.0, "SHORT", 2.5) - 102.5) < 1e-9
    assert backstop_trigger_price(0, "LONG", 2.5) is None and backstop_trigger_price(100, "LONG", 0) is None and backstop_trigger_price("x", "LONG", 1) is None
    assert backstop_needs_replace(None, None, 2.5, 97.5) is True                 # a row opened before the record existed → re-placed once
    assert backstop_needs_replace(2.5, 97.5, 2.5, 97.5) is False                 # unchanged
    assert backstop_needs_replace(2.5, 97.5, 2.51, 97.49) is False               # inside the 0.02-pt tolerance
    assert backstop_needs_replace(4.14, 95.86, 4.5, 95.5) is True                # WILLY 20× → 10×: the distance moved
    assert backstop_needs_replace(2.5, 97.5, 2.5, 98.0) is True                  # same %, entry repaired → the trigger moved
    assert backstop_needs_replace(2.5, 97.5, None, None) is False and backstop_needs_replace(2.5, 97.5, 0, 0) is False   # never cancel for an unplaceable want
    assert algo_status_resting({"algoStatus": "NEW"}) and not algo_status_resting({"algoStatus": "CANCELED"}) and not algo_status_resting(None)


def test_model_column_and_migration():
    import models
    cols = [c.name for c in models.Order.__table__.columns]
    assert "backstop_pct" in cols and "backstop_trigger_price" in cols           # model columns → the orders CSV carries them (D12)
    src = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "ALTER TABLE orders ADD COLUMN backstop_pct FLOAT" in src and "ALTER TABLE orders ADD COLUMN backstop_trigger_price FLOAT" in src


def test_open_paths_record_what_was_placed():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "backstop_pct=_bk_rec_pct, backstop_trigger_price=_bk_rec_trig," in eng                     # bot open
    assert "and (self.is_paper_mode or _bk_algo_id)):" in eng                                          # live: only a placed stop is recorded
    assert "order.backstop_pct = order_backstop_pct(th, \"MANUAL\", exit_mode, sl, leverage)" in eng   # manual open
    assert eng.count("backstop_pct, ") >= 2 and "order.backstop_pct, order.backstop_trigger_price = _bk_pct, _bk_tr" in eng   # failed-close re-place
    assert "_bko.backstop_pct, _bko.backstop_trigger_price = _bk_pct, _bk_tr" in eng                  # heal sweep
    assert "await self.backstop_reprice_sweep()" in eng


# ---------------------------------------------------------------- sweep harness (in-memory DB + mocked exchange)

def _setup(monkeypatch, paper, rows, place=None, query=None, cancel_ok=True, mark=0.0):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models, services.trading_engine as T
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    S = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False)
    monkeypatch.setattr(th, "broker_backstop_pct", 2.5, raising=False)
    monkeypatch.setattr(th, "frenzy_willy_stop_pct", 4.0, raising=False)
    monkeypatch.setattr(th, "frenzy_stop_pct", 3.0, raising=False)
    log = []
    async def _place(pair, direction, trig):
        log.append(("place", pair, round(trig, 6)))
        if paper:
            raise AssertionError("paper must never call the exchange")
        return place(pair) if callable(place) else place
    async def _cancel(pair, aid):
        log.append(("cancel", pair, aid)); return cancel_ok
    async def _query(aid):
        log.append(("query", aid)); return query
    monkeypatch.setattr(T.binance_service, "place_backstop_stop", _place)
    monkeypatch.setattr(T.binance_service, "cancel_backstop_stop", _cancel)
    monkeypatch.setattr(T.binance_service, "query_backstop_stop", _query)
    async def _mark(pair):
        if paper:
            raise AssertionError("paper must never call the exchange")
        return mark
    monkeypatch.setattr(T.binance_service, "get_current_price", _mark)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper
    blocks = []
    eng._record_filter_block = lambda name, d, had_room=True: blocks.append(name)

    async def seed():
        async with engine.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as s:
            for kw in rows:
                base = dict(direction="LONG", status="OPEN", current_price=100.0, investment=100.0, notional_value=1000.0, quantity=10.0,
                            confidence="STRONG_BUY", is_paper=paper, opened_at=datetime.utcnow(), closing_in_progress=False)
                base.update(kw)
                s.add(models.Order(**base))
            await s.commit()

    async def read():
        from sqlalchemy import select
        async with S() as s:
            return {o.pair: o for o in (await s.execute(select(models.Order))).scalars().all()}
    return T, eng, engine, seed, read, log, blocks


def test_paper_records_the_configured_backstop_without_exchange_calls(monkeypatch):
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_pct=4.14, backstop_trigger_price=95.86),
            dict(pair="MOMUSDT", entry_strategy="MOMENTUM", entry_price=50.0, leverage=20.0, backstop_pct=2.5, backstop_trigger_price=48.75),
            dict(pair="OLDUSDT", entry_strategy="FRENZY_LONG", entry_price=10.0, leverage=1.0)]   # legacy: nothing recorded
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, True, rows)
    async def go():
        await seed()
        out = await eng.backstop_reprice_sweep(now=1000.0)
        again = await eng.backstop_reprice_sweep(now=1001.0)            # throttled
        r = await read(); await engine.dispose(); return out, again, r
    out, again, r = asyncio.run(go())
    assert again == [] and log == []
    assert sorted(x[1] for x in out) == ["paper", "paper"]
    assert r["WILUSDT"].backstop_pct == 4.5 and abs(r["WILUSDT"].backstop_trigger_price - 95.5) < 1e-9     # 10×: own 4 + 0.5
    assert r["MOMUSDT"].backstop_pct == 2.5 and r["MOMUSDT"].backstop_trigger_price == 48.75                 # unchanged
    assert r["OLDUSDT"].backstop_pct == 4.5 and abs(r["OLDUSDT"].backstop_trigger_price - 9.55) < 1e-9     # 🛑 (317) FRENZY_LONG own stop 4 + 0.5


def test_live_replace_places_new_then_cancels_old(monkeypatch):
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86),
            dict(pair="MOMUSDT", entry_strategy="MOMENTUM", entry_price=50.0, leverage=20.0, backstop_algo_id="M1", backstop_pct=2.5, backstop_trigger_price=48.75),
            dict(pair="BAREUSDT", entry_strategy="MOMENTUM", entry_price=50.0, leverage=20.0),                     # no stop: the heal sweep's job
            dict(pair="CLOSEUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="C1", closing_in_progress=True)]
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place="NEW1")
    async def go():
        await seed(); out = await eng.backstop_reprice_sweep(now=0.0); r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert [x[1] for x in out] == ["replaced"]
    assert log == [("place", "WILUSDT", 95.5), ("cancel", "WILUSDT", "OLD1")]                  # place FIRST, then cancel the old one
    w = r["WILUSDT"]
    assert w.backstop_algo_id == "NEW1" and w.backstop_pct == 4.5 and abs(w.backstop_trigger_price - 95.5) < 1e-9
    assert r["MOMUSDT"].backstop_algo_id == "M1" and r["BAREUSDT"].backstop_algo_id is None and r["CLOSEUSDT"].backstop_algo_id == "C1"


def test_live_failed_place_keeps_a_resting_old_stop_and_retries_later(monkeypatch):
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86)]
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place=None, query={"algoStatus": "NEW"})
    async def go():
        await seed()
        out1 = await eng.backstop_reprice_sweep(now=0.0)
        out2 = await eng.backstop_reprice_sweep(now=31.0)                     # past the throttle, inside the per-order retry wait
        out3 = await eng.backstop_reprice_sweep(now=BACKSTOP_REPLACE_RETRY_S + 1)
        r = await read(); await engine.dispose(); return out1, out2, out3, r
    out1, out2, out3, r = asyncio.run(go())
    assert [x[1] for x in out1] == ["kept_old"] and out2 == [] and [x[1] for x in out3] == ["kept_old"]
    assert r["WILUSDT"].backstop_algo_id == "OLD1" and r["WILUSDT"].backstop_pct == 4.14              # the old stop is the record
    assert "cancel" not in [x[0] for x in log] and blocks.count("BACKSTOP_REPLACE_FAILED") == 2


def test_live_failed_place_with_the_old_stop_gone_hands_over_to_the_heal_sweep(monkeypatch):
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86)]
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place=None, query={"algoStatus": "CANCELED"})
    async def go():
        await seed(); out = await eng.backstop_reprice_sweep(now=0.0); r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert [x[1] for x in out] == ["lost"] and r["WILUSDT"].backstop_algo_id is None and "BACKSTOP_PLACE_FAILED" in blocks


def test_live_row_closed_during_the_replace_cancels_the_new_stop(monkeypatch):
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86)]
    holder = {}
    def _place(pair):   # the close path books the row while the exchange call is in flight
        holder["close"] = True
        return "NEW1"
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place=_place)
    orig = eng._backstop_replace_one
    async def _wrapped(*a, **k):
        res = await orig(*a, **k)
        from sqlalchemy import update
        import models
        async with T.AsyncSessionLocal() as s:
            await s.execute(update(models.Order).values(status="CLOSED")); await s.commit()
        return res
    eng._backstop_replace_one = _wrapped
    async def go():
        await seed(); out = await eng.backstop_reprice_sweep(now=0.0); r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert ("cancel", "WILUSDT", "NEW1") in log and r["WILUSDT"].backstop_algo_id == "OLD1"           # the unwanted new stop is pulled


def test_live_row_restopped_by_another_path_keeps_exactly_one_stop(monkeypatch):
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86)]
    for theirs_status, want_id, cancelled_new in (("NEW", "THEIRS", True), ("CANCELED", "NEW1", False)):
        T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place="NEW1", query={"algoStatus": theirs_status})
        orig = eng._backstop_replace_one
        async def _wrapped(*a, _orig=orig, _T=T, **k):
            res = await _orig(*a, **k)
            from sqlalchemy import update
            import models
            async with _T.AsyncSessionLocal() as s:   # e.g. the failed-close re-place wrote its own stop while ours was in flight
                await s.execute(update(models.Order).values(backstop_algo_id="THEIRS")); await s.commit()
            return res
        eng._backstop_replace_one = _wrapped
        async def go():
            await seed(); await eng.backstop_reprice_sweep(now=0.0); r = await read(); await engine.dispose(); return r
        r = asyncio.run(go())
        assert r["WILUSDT"].backstop_algo_id == want_id and ((("cancel", "WILUSDT", "NEW1") in log) == cancelled_new)


def test_live_crossed_trigger_keeps_the_old_stop_untouched(monkeypatch):
    # tightening while the price already sits past the new trigger: -2021 would come AFTER the -4130 clear → no stop at all. Skip instead.
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86)]
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place="NEW1", mark=95.4)   # LONG, new trigger 95.5 > mark
    async def go():
        await seed(); out = await eng.backstop_reprice_sweep(now=0.0); r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    assert [x[1] for x in out] == ["crossed"] and log == [] and r["WILUSDT"].backstop_algo_id == "OLD1" and r["WILUSDT"].backstop_pct == 4.14
    from services.golive_guards import trigger_already_crossed
    assert trigger_already_crossed("LONG", 95.5, 95.5) and not trigger_already_crossed("LONG", 95.5, 96)
    assert trigger_already_crossed("SHORT", 104.5, 104.6) and not trigger_already_crossed("SHORT", 104.5, 104)
    assert not trigger_already_crossed("LONG", 95.5, 0) and not trigger_already_crossed("LONG", 95.5, None)


def test_live_closing_row_keeps_our_stop_when_the_old_one_is_gone(monkeypatch):
    # the -4130 clear removed the old stop; the row went closing_in_progress meanwhile (a close that may fail and retry): never cancel ours
    rows = [dict(pair="WILUSDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id="OLD1", backstop_pct=4.14, backstop_trigger_price=95.86)]
    for old_status, want_id, cancelled_new in (("CANCELED", "NEW1", False), ("NEW", "OLD1", True)):
        T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place="NEW1", query={"algoStatus": old_status})
        orig = eng._backstop_replace_one
        async def _wrapped(*a, _orig=orig, _T=T, **k):
            res = await _orig(*a, **k)
            from sqlalchemy import update
            import models
            async with _T.AsyncSessionLocal() as s:
                await s.execute(update(models.Order).values(closing_in_progress=True)); await s.commit()
            return res
        eng._backstop_replace_one = _wrapped
        async def go():
            await seed(); await eng.backstop_reprice_sweep(now=0.0); r = await read(); await engine.dispose(); return r
        r = asyncio.run(go())
        assert r["WILUSDT"].backstop_algo_id == want_id and ((("cancel", "WILUSDT", "NEW1") in log) == cancelled_new)


def test_live_replaces_are_capped_per_pass(monkeypatch):
    from services.golive_guards import BACKSTOP_REPLACE_MAX_PER_PASS, BACKSTOP_REPRICE_INTERVAL_S
    rows = [dict(pair=f"W{i}USDT", entry_strategy="FRENZY_WILLY", entry_price=100.0, leverage=10.0, backstop_algo_id=f"O{i}",
                 backstop_pct=4.14, backstop_trigger_price=95.86) for i in range(5)]
    T, eng, engine, seed, read, log, blocks = _setup(monkeypatch, False, rows, place=lambda pair: "N" + pair)
    async def go():
        await seed()
        o1 = await eng.backstop_reprice_sweep(now=0.0)
        o2 = await eng.backstop_reprice_sweep(now=BACKSTOP_REPRICE_INTERVAL_S + 1)
        o3 = await eng.backstop_reprice_sweep(now=2 * BACKSTOP_REPRICE_INTERVAL_S + 2)
        r = await read(); await engine.dispose(); return o1, o2, o3, r
    o1, o2, o3, r = asyncio.run(go())
    assert BACKSTOP_REPLACE_MAX_PER_PASS == 2 and len(o1) == 2 and len(o2) == 2 and len(o3) == 1
    assert all(o.backstop_pct == 4.5 for o in r.values())


def test_heal_sweep_failed_placement_is_throttled():
    import inspect, services.trading_engine as T
    src = inspect.getsource(T)
    assert "_bk_heal_next" in src and "BACKSTOP_HEAL_RETRY_S" in src
