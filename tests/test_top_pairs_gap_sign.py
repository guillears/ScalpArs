"""Top Pairs Gap 5-8 shown with its sign (Sep-30)."""

def test_top_pairs_gap_5_8_is_signed():
    """Sep-30 operator: Top Pairs Gap 5-8 was absolute (always looked positive) while Gap 5-20 was signed; the early-turn setup
    (EMA5 above EMA8, EMA5 still under EMA20) could not be read. Both are signed now; the gate colour uses the magnitude."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_src = open(os.path.join(root, "main.py"), encoding="utf-8").read()
    assert "gap_5_8 = round(((p.ema5 - p.ema8) / p.ema8) * 100, 4)" in main_src and "gap_5_8 = round(abs(" not in main_src
    ui = open(os.path.join(root, "templates", "index.html"), encoding="utf-8").read()
    assert ui.count("Math.abs(p.gap_5_8) >= gap58Min") == 2 and "p.gap_5_8 >= gap58Min" not in ui


def test_top_pairs_payload_carries_a_negative_gap_5_8():
    """Behavioural: the /api/pairs row of a pair with EMA5 under EMA8 (and EMA5 under EMA20) shows BOTH gaps negative."""
    import asyncio, datetime as dt, os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, root)
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with async_sessionmaker(eng, expire_on_commit=False)() as db:
            now = dt.datetime.utcnow()
            db.add(models.PairData(pair="NEARUSDT", price=4.91, volume_24h=5e8, ema5=4.900, ema8=4.910, ema13=4.925, ema20=4.940,
                                   rsi=43.0, adx=24.0, signal="NO_TRADE", confidence="NO_TRADE", updated_at=now))
            db.add(models.PairData(pair="QNTUSDT", price=268.3, volume_24h=4e8, ema5=268.0, ema8=267.7, ema13=267.4, ema20=267.0,
                                   rsi=55.0, adx=17.0, signal="LONG", confidence="STRONG_BUY", updated_at=now))
            await db.commit()
            out = await main.get_pairs(db=db, limit=50)
        await eng.dispose()
        return out
    rows = asyncio.run(run())
    rows = rows if isinstance(rows, list) else rows.get("pairs", rows)
    by = {r["pair"]: r for r in rows}
    assert by["NEARUSDT"]["gap_5_8"] == round((4.900 - 4.910) / 4.910 * 100, 4) < 0 and by["NEARUSDT"]["gap"] < 0
    assert by["QNTUSDT"]["gap_5_8"] == round((268.0 - 267.7) / 267.7 * 100, 4) > 0 and by["QNTUSDT"]["gap"] > 0


def test_top_pairs_stack_is_judged_on_raw_emas():
    """A sub-$1 pair whose EMAs tie at 2 decimals is still a stack: the server sends ema_stack from the raw values."""
    import asyncio, datetime as dt, os, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, root)
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with async_sessionmaker(eng, expire_on_commit=False)() as db:
            now = dt.datetime.utcnow()
            db.add(models.PairData(pair="PEPEUSDT", price=0.0101, volume_24h=5e8, ema5=0.01014, ema8=0.01012, ema13=0.01011, ema20=0.01009,
                                   rsi=55.0, adx=20.0, signal="LONG", confidence="STRONG_BUY", updated_at=now))
            db.add(models.PairData(pair="NEARUSDT", price=4.91, volume_24h=4e8, ema5=4.900, ema8=4.910, ema13=4.905, ema20=4.940,
                                   rsi=43.0, adx=24.0, signal="NO_TRADE", confidence="NO_TRADE", updated_at=now))
            db.add(models.PairData(pair="BONKUSDT", price=0.0101, volume_24h=3e8, ema5=0.01006, ema8=0.01008, ema13=0.01009, ema20=0.01011,
                                   rsi=35.0, adx=22.0, signal="SHORT", confidence="STRONG_BUY", updated_at=now))
            await db.commit()
            out = await main.get_pairs(db=db, limit=50)
        await eng.dispose()
        return out
    rows = asyncio.run(run()); rows = rows if isinstance(rows, list) else rows.get("pairs", rows)
    by = {r["pair"]: r for r in rows}
    assert by["PEPEUSDT"]["ema_stack"] == "BULL" and by["NEARUSDT"]["ema_stack"] is None and by["BONKUSDT"]["ema_stack"] == "BEAR"
    ui = open(os.path.join(root, "templates", "index.html"), encoding="utf-8").read()
    assert "p.ema_stack === 'BULL'" in ui and "p.ema_stack === 'BEAR'" in ui
