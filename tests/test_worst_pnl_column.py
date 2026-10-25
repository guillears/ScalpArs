"""📉 Worst P&L % (operator 2026-10-24): the open / closed orders tables show each trade's trough (Order.trough_pnl) next to Peak P&L %.
Open: cache → DB, never above the current P&L (the peak's invariant mirrored) · closed: the stored trough, None → '—'. No network."""
import asyncio
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _order(models, pair, status, **kw):
    t0 = dt.datetime.utcnow() - dt.timedelta(hours=1)
    base = dict(pair=pair, direction="LONG", status=status, entry_price=100.0, exit_price=(101.0 if status == "CLOSED" else None),
                current_price=100.0, investment=500.0, leverage=10.0, notional_value=5000.0, quantity=50.0, confidence="STRONG_BUY",
                entry_strategy="MOMENTUM", is_paper=True, pnl=0.0, pnl_percentage=0.0, peak_pnl=1.2, trough_pnl=-0.7, entry_fee=2.25,
                total_fee=4.5, opened_at=t0, closed_at=(t0 + dt.timedelta(minutes=30) if status == "CLOSED" else None),
                close_reason=("TAKE_PROFIT" if status == "CLOSED" else None))
    base.update(kw)
    return models.Order(**base)


def _run(fn, rows, monkeypatch=None):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        async with S() as db:
            for r in rows(models):
                db.add(r)
            await db.commit()
            out = json.loads(json.dumps(await fn(main)(db=db), default=str))
        await eng.dispose()
        return out
    return asyncio.run(go())


def test_closed_orders_carry_the_trough():
    out = _run(lambda m: m.get_closed_orders, lambda M: [_order(M, "AUSDT", "CLOSED"), _order(M, "BUSDT", "CLOSED", trough_pnl=-3.456)])
    rows = out if isinstance(out, list) else out.get("orders", out)
    by = {r["pair"]: r for r in rows}
    assert by["AUSDT"]["trough_pnl"] == -0.7 and by["BUSDT"]["trough_pnl"] == -3.46
    # a close booked below the stored trough (reconcile / external close skipped the update) → the worst shows the close
    out = _run(lambda m: m.get_closed_orders, lambda M: [_order(M, "CUSDT", "CLOSED", trough_pnl=-0.2, pnl_percentage=-1.5)])
    rows = out if isinstance(out, list) else out.get("orders", out)
    assert [r for r in rows if r["pair"] == "CUSDT"][0]["trough_pnl"] == -1.5


def test_open_orders_trough_never_above_current(monkeypatch):
    import main
    monkeypatch.setattr(main, "_open_orders_cache", {}, raising=False)
    out = _run(lambda m: m.get_open_orders, lambda M: [_order(M, "CUSDT", "OPEN", trough_pnl=-0.3)])
    rows = out if isinstance(out, list) else out.get("orders", out)
    r = [x for x in rows if x["pair"] == "CUSDT"][0]
    assert r["trough_pnl"] <= r["pnl_percentage"] and r["trough_pnl"] <= -0.3 + 1e-9
    # the DB trough above the current P&L → clamped to the current P&L
    out = _run(lambda m: m.get_open_orders, lambda M: [_order(M, "DUSDT", "OPEN", trough_pnl=0.0)])
    rows = out if isinstance(out, list) else out.get("orders", out)
    r = [x for x in rows if x["pair"] == "DUSDT"][0]
    assert r["pnl_percentage"] < 0 and r["trough_pnl"] == r["pnl_percentage"]


def test_open_orders_prefer_the_fresher_cache(monkeypatch):
    import main

    class Cache(dict):
        def get(self, pair, default=None):                    # every pair's cache entry carries the id of the row being served
            return [dict(id=self.id_, trough_pnl=-2.5, peak_pnl=1.2)] if pair == "EUSDT" else (default or [])
    c = Cache()
    monkeypatch.setattr(main, "_open_orders_cache", c, raising=False)

    def rows(M):
        o = _order(M, "EUSDT", "OPEN", trough_pnl=-0.3, id=777)
        c.id_ = 777
        return [o]
    out = _run(lambda m: m.get_open_orders, rows)
    rs = out if isinstance(out, list) else out.get("orders", out)
    assert [x for x in rs if x["pair"] == "EUSDT"][0]["trough_pnl"] == -2.5


def test_both_tables_have_the_column_and_matching_colspans():
    h = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert h.count(">Worst P&L %</th>") == 2 and h.count("o.trough_pnl == null ? '—'") == 2
    assert 'colspan="19"' not in h and h.count('colspan="20"') == 5
    for tb in ("open-orders-table-body", "closed-orders-table-body"):
        i = h.index(f'<tbody id="{tb}"')
        head = h[h.rindex("<thead", 0, i):i]
        import re
        assert len(re.findall(r"<th[ >]", head)) == 20, (tb, len(re.findall(r"<th[ >]", head)))
