"""A FAILED position read is never "position gone" (DECISION_LOG 183). `get_position_for_symbol` used to swallow every error and
return None — the same value as "no position" — so the close path booked a still-open trade as closed and the reconciler's
per-symbol fallback (which runs only while the exchange is failing) closed every live row as EXTERNAL."""
import asyncio, os, sys
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def test_the_read_raises_on_error_and_returns_none_only_for_no_position():
    from services.binance_service import BinanceService
    svc = BinanceService.__new__(BinanceService)
    async def _lm(): return None
    svc.load_markets = _lm
    svc._detect_ban = staticmethod(lambda e: None)

    class Ex:
        def __init__(self, out): self.out = out
        async def fetch_positions(self, symbols):
            if isinstance(self.out, Exception):
                raise self.out
            return self.out
    svc.exchange = Ex([{"symbol": "FOO/USDT:USDT", "contracts": 0}])
    assert asyncio.run(svc.get_position_for_symbol("FOO/USDT:USDT")) is None                   # Binance answered: nothing open
    svc.exchange = Ex([{"symbol": "FOO/USDT:USDT", "contracts": 5, "side": "long", "entryPrice": 2.0, "markPrice": 2.1, "unrealizedPnl": 0.5, "leverage": 20}])
    pos = asyncio.run(svc.get_position_for_symbol("FOO/USDT:USDT"))
    assert pos["side"] == "LONG" and pos["contracts"] == 5.0
    svc.exchange = Ex(RuntimeError("network down"))
    try:
        asyncio.run(svc.get_position_for_symbol("FOO/USDT:USDT"))
        raise AssertionError("a failed read must raise, not look like 'no position'")
    except RuntimeError as e:
        assert "network down" in str(e)


def test_reconciler_fallback_keeps_rows_open_when_the_exchange_cannot_be_read():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from database import Base
    from models import Order
    import main

    async def go(per_symbol):
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as c:
            await c.run_sync(Base.metadata.create_all)
        S = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with S() as s:
            s.add(Order(pair="SUIUSDT", direction="LONG", status="OPEN", entry_price=1.0, current_price=1.0, investment=100.0, leverage=1.0,
                        notional_value=100.0, quantity=100.0, confidence="STRONG_BUY", entry_fee=0.04, is_paper=False, closing_in_progress=False,
                        opened_at=datetime.utcnow() - timedelta(minutes=10)))
            await s.commit()
        async with S() as s:
            with patch.object(main.binance_service, "get_position_for_symbol", new=per_symbol), \
                 patch("main._get_actual_fill_price", new=AsyncMock(return_value=1.0)):
                closed = await main._reconcile_per_symbol(s)
        async with S() as s:
            status = (await s.execute(select(Order.status))).scalar_one()
        await engine.dispose()
        return len(closed), status
    assert asyncio.run(go(AsyncMock(side_effect=RuntimeError("network down")))) == (0, "OPEN")   # unreadable → the row stays open
    assert asyncio.run(go(AsyncMock(return_value=None))) == (1, "CLOSED")                         # Binance answered "no position" → reconciled as before


def test_close_path_treats_a_failed_check_as_still_open():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("_check_pos = await binance_service.get_position_for_symbol(symbol)")
    block = eng[i - 200:i + 1500]
    assert "try:" in block and "if _check_pos is None:" in block and "except Exception as _check_err:" in block and "continuing retry" in block
    bs = open(os.path.join(ROOT, "services", "binance_service.py"), encoding="utf-8").read()
    j = bs.index("async def get_position_for_symbol"); body = bs[j:bs.index("async def ", j + 10)]
    assert body.rstrip().endswith("raise") and "return None\n        except Exception as e:" in body      # None only on the answered path
