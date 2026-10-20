"""🤖 DECISION_LOG 323: the dashboard's All · Bot · Manual switch. Pins the classification, the touched count, and the money invariant:
the BOT view of a book with manual trades equals the book that never had them (same capital: the manual P&L is in the balance)."""
import asyncio
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.origin_view import norm_origin, order_origin, origin_keep, bot_touched, touched_summary  # noqa: E402


def test_classification():
    assert order_origin("MANUAL") == "MANUAL" and order_origin("manual") == "MANUAL"
    assert order_origin(None, "RECOVERED") == "MANUAL"                       # a position adopted from Binance = the operator's
    for es in ("FRENZY_WILLY", "MOMENTUM", "FLIP:FAN_RATIO_GATE", None, ""):
        assert order_origin(es, "STRONG_BUY") == "BOT"
    assert norm_origin("bot") == "BOT" and norm_origin("x") == "ALL" and norm_origin(None) == "ALL"
    assert origin_keep("MANUAL", None, "ALL") and not origin_keep("MANUAL", None, "BOT") and origin_keep("MANUAL", None, "MANUAL")


def test_touched():
    now = dt.datetime(2026, 10, 20)
    assert bot_touched("FRENZY_LONG", "X", now, "MANUAL_TP") and bot_touched("FRENZY_LONG", "X", None, "EXTERNAL_CLOSE")
    assert not bot_touched("FRENZY_LONG", "X", None, "FRENZY_TP") and not bot_touched("MANUAL", None, now, "EXTERNAL_CLOSE")

    class O:
        def __init__(self, **k): self.__dict__.update(k)
    rows = [O(entry_strategy="FRENZY_LONG", confidence="X", exit_override_at=now, close_reason="OVERRIDE_TP", pnl=12.5),
            O(entry_strategy="FRENZY_LITE", confidence="X", exit_override_at=None, close_reason="EXTERNAL_CLOSE", pnl=-2.0),
            O(entry_strategy="MANUAL", confidence="X", exit_override_at=now, close_reason="EXTERNAL_CLOSE", pnl=99.0),
            O(entry_strategy="FRENZY_WILLY", confidence="X", exit_override_at=None, close_reason="FRENZY_TP", pnl=5.0)]
    assert touched_summary(rows) == {"n": 2, "pnl": 10.5}


def _order(models, i, strat, direction, pnl_pct, **kw):
    t0 = dt.datetime(2026, 10, 18, 10, 0, 0) + dt.timedelta(hours=3 * i)
    base = dict(pair=f"P{i}USDT", direction=direction, status="CLOSED", entry_price=100.0, exit_price=100.0 * (1 + pnl_pct / 100),
                investment=500.0, leverage=20.0, notional_value=10000.0, quantity=100.0, confidence="STRONG_BUY", entry_strategy=strat,
                is_paper=True, pnl=pnl_pct * 100.0, pnl_percentage=pnl_pct, peak_pnl=max(pnl_pct, 0.1), trough_pnl=min(pnl_pct, -0.1),
                entry_fee=4.5, total_fee=9.0, opened_at=t0, closed_at=t0 + dt.timedelta(minutes=17),
                close_reason=("FRENZY_TP" if pnl_pct > 0 else "STOP_LOSS"), entry_macro_trend="BULLISH", cell_multiplier=1.0, cell_lev_multiplier=1.0)
    base.update(kw)
    return models.Order(**base)


BOT = [("LONG", 0.5), ("LONG", -0.7), ("SHORT", 0.4), ("LONG", 0.3), ("LONG", 0.6)]
MAN = [("LONG", 1.2), ("LONG", -3.0), ("LONG", 0.9)]


def _perf(with_manual, origin, paper_balance):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        main.trading_engine.paper_balance = paper_balance
        async with S() as db:
            for i, (d, p) in enumerate(BOT):
                db.add(_order(models, i, "FRENZY_LONG", d, p, **({"exit_override_at": dt.datetime(2026, 10, 18)} if i == 0 else {})))
            if with_manual:
                for j, (d, p) in enumerate(MAN):
                    db.add(_order(models, 10 + j, "MANUAL", d, p))
                db.add(_order(models, 20, None, "LONG", 0.25, confidence="RECOVERED"))
            await db.commit()
            out = json.loads(json.dumps(await main._compute_performance(db, origin=origin), default=str))
        await eng.dispose()
        return out
    return asyncio.run(run())


def test_bot_view_equals_the_book_without_manual_trades():
    man_pnl = sum(p * 100.0 for _, p in MAN) + 25.0
    clean = _perf(False, None, 10_000.0)
    bot = _perf(True, "BOT", 10_000.0 + man_pnl)          # the manual P&L sits in the real balance
    for k in ("total_trades", "win_rate", "total_pnl", "avg_win", "avg_loss", "expectancy", "return_multiple", "daily_compound_return"):
        assert bot[k] == clean[k], k
    assert bot["origin"] == "BOT" and bot["origin_touched"] == {"n": 1, "pnl": 50.0}
    allv = _perf(True, None, 10_000.0 + man_pnl)
    assert allv["total_trades"] == len(BOT) + len(MAN) + 1 and allv["origin"] == "ALL"
    man = _perf(True, "MANUAL", 10_000.0 + man_pnl)
    assert man["total_trades"] == len(MAN) + 1 and abs(man["total_pnl"] - man_pnl) < 1e-6 and man["origin_touched"]["n"] == 0


def test_dashboard_wiring():
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html")).read()
    assert html.count('id="origin-switch"') == 1 and html.count('id="origin-view-tag"') == 1
    assert "api/pnl-calendar?tz_offset_min=${tzOffsetMin}${_originParam('&')}" in html
    assert html.count("_originReportLine(") == 3                 # the helper + BOTH text exports (clipboard + split report)
    assert "params.push(`origin=${_ovPerf}`)" in html and "_params1.push(`origin=${_ov1}`)" in html and "_originParam('&');" in html


def _cal(with_manual, origin, realized_equity):
    """realized_equity = the account's real realized equity (live: Binance — it holds the manual P&L too)."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def _pc(db, fresh=False):
        return {"realized_equity": realized_equity, "total_equity": realized_equity}

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        _orig = main._portfolio_components
        main._portfolio_components = _pc
        async with S() as db:
            _d0 = dt.datetime.utcnow().replace(hour=1, minute=0, second=0, microsecond=0) - dt.timedelta(days=len(BOT))   # one active day each
            for i, (d, p) in enumerate(BOT):
                t = _d0 + dt.timedelta(days=i)
                db.add(_order(models, i, "FRENZY_LONG", d, p, opened_at=t, closed_at=t + dt.timedelta(minutes=17)))
            if with_manual:
                for j, (d, p) in enumerate(MAN):
                    t = _d0 + dt.timedelta(days=j, hours=5)
                    db.add(_order(models, 10 + j, "MANUAL", d, p, opened_at=t, closed_at=t + dt.timedelta(minutes=17)))
            await db.commit()
            try:
                out = json.loads(json.dumps(await main.get_pnl_calendar(tz_offset_min=0, origin=origin, db=db), default=str))
            finally:
                main._portfolio_components = _orig
        await eng.dispose()
        return out
    return asyncio.run(run())


def test_calendar_bot_view_equals_the_book_without_manual_trades():
    man_pnl = sum(p * 100.0 for _, p in MAN)
    clean = _cal(False, None, 3110.0)
    bot = _cal(True, "BOT", 3110.0 + man_pnl)
    assert bot["days"] == clean["days"] and bot["portfolio_now"] == clean["portfolio_now"]
    assert bot["portfolio_real"] == round(clean["portfolio_now"] + man_pnl, 2) and bot["origin_out_pnl"] == round(man_pnl, 2)
    assert bot["revisions_by_month"] == clean["revisions_by_month"] and bot["eom_now"] == clean["eom_now"]
    import datetime as _d
    if (_d.datetime.utcnow() - _d.timedelta(days=len(BOT))).month == _d.datetime.utcnow().month:   # ≥ 4 active days inside this month
        assert clean["revisions"] and clean["eom_now"] is not None                                # the EOM invariance is really exercised
    allv = _cal(True, None, 3110.0 + man_pnl)
    assert allv["portfolio_now"] == allv["portfolio_real"] == round(3110.0 + man_pnl, 2) and allv["origin"] == "ALL"
    assert sum(e["trades"] for e in allv["days"].values()) == len(BOT) + len(MAN)
    assert allv["eom_now"] != bot["eom_now"]
