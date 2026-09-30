"""🔒 Sep-30 position recovery (live-only endpoint) imports an exchange position without an order row only if it is a real
orphan: never one the bot is placing right now (the bot holds the pair's open lock from its check to its insert), and never
one that closed after the position list was read."""
import asyncio, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _pos(sym="QNT/USDT:USDT"):
    return dict(symbol=sym, margin=100.0, leverage=10, notional=1000.0, contracts=3.7, entry_price=270.0, side="LONG", mark_price=270.5)


def _run(scenario, monkeypatch):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select, func
    import main, models
    E = main.trading_engine

    async def _init(db): return None
    monkeypatch.setattr(E, "initialize", _init)
    monkeypatch.setattr(E, "is_paper_mode", False)

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        out = await scenario(S, main, models, E)
        async with S() as db:
            n = (await db.execute(select(func.count(models.Order.id)).where(models.Order.status == "OPEN", models.Order.pair == "QNTUSDT"))).scalar()
        await eng.dispose()
        return out, n
    return asyncio.run(go())


def test_a_position_the_bot_is_placing_is_not_imported(monkeypatch):
    import main
    reads = []

    async def positions():
        reads.append(1); return [_pos()]
    monkeypatch.setattr(main.binance_service, "get_open_positions", positions)

    async def scenario(S, main, models, E):
        async def bot_open():                                  # the bot's live open: fill on the exchange, row 0.3 s later
            async with E._pair_open_guard("QNTUSDT"):
                await asyncio.sleep(0.3)
                async with S() as db:
                    db.add(models.Order(pair="QNTUSDT", direction="LONG", status="OPEN", entry_price=270.0, investment=100.0, leverage=10,
                                        notional_value=1000.0, quantity=3.7, confidence="STRONG_BUY", is_paper=False)); await db.commit()
        async def recover():
            await asyncio.sleep(0.05)
            async with S() as db:
                return await main.recover_positions(db=db)
        _, res = await asyncio.gather(bot_open(), recover())
        return res
    res, n = _run(scenario, monkeypatch)
    assert n == 1 and res["recovered"] == 0                   # waited for the bot's row, saw it, imported nothing


def test_a_position_closed_since_the_list_was_read_is_not_imported(monkeypatch):
    import main
    calls = []

    async def positions():
        calls.append(1); return [_pos()] if len(calls) == 1 else []   # first read: open; fresh read under the lock: closed
    monkeypatch.setattr(main.binance_service, "get_open_positions", positions)

    async def scenario(S, main, models, E):
        async with S() as db:
            return await main.recover_positions(db=db)
    res, n = _run(scenario, monkeypatch)
    assert n == 0 and res["recovered"] == 0 and res["skipped"][0]["reason"] == "no longer open on the exchange"


def test_a_real_orphan_is_imported_once(monkeypatch):
    import main

    async def positions(): return [_pos()]
    monkeypatch.setattr(main.binance_service, "get_open_positions", positions)

    async def scenario(S, main, models, E):
        async with S() as db:
            a = await main.recover_positions(db=db)
        async with S() as db:
            b = await main.recover_positions(db=db)             # a second run finds the row and imports nothing
        return a, b
    (a, b), n = _run(scenario, monkeypatch)
    assert n == 1 and a["recovered"] == 1 and b["recovered"] == 0


def test_a_failed_exchange_re_read_or_a_failing_pair_never_imports_and_never_stops_the_rest(monkeypatch):
    import main
    seq = []

    async def positions():
        seq.append(1)
        if len(seq) == 1:
            return [_pos("QNT/USDT:USDT"), _pos("SUI/USDT:USDT")]
        return None if len(seq) == 2 else [_pos("SUI/USDT:USDT")]     # QNT re-read fails; SUI re-read fine
    monkeypatch.setattr(main.binance_service, "get_open_positions", positions)

    async def scenario(S, main, models, E):
        async with S() as db:
            return await main.recover_positions(db=db)
    res, n = _run(scenario, monkeypatch)
    assert n == 0 and res["recovered"] == 1 and res["positions"][0]["pair"] == "SUIUSDT"
    assert res["skipped"][0] == {"pair": "QNTUSDT", "reason": "exchange read failed"}


def test_hedge_mode_imports_the_right_leg(monkeypatch):
    import main
    short = dict(_pos(), side="SHORT", contracts=9.9)

    async def positions(): return [_pos()] if not hasattr(positions, "n") else [short, _pos()]
    async def wrapped():
        r = await positions(); positions.n = 1; return r
    monkeypatch.setattr(main.binance_service, "get_open_positions", wrapped)

    async def scenario(S, main, models, E):
        async with S() as db:
            return await main.recover_positions(db=db)
    res, n = _run(scenario, monkeypatch)
    assert n == 1 and res["positions"][0]["direction"] == "LONG" and res["positions"][0]["quantity"] == 3.7
