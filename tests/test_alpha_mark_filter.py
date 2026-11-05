"""🅰️ DECISION_LOG 466: the α mark (open / closed trades from their own entry stamp, FRENZY pairs from Binance's current tag) and the
Performance "Alpha" filter (server param + UI + both text exports)."""
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_cell_filter_origin_switch as T  # noqa: E402

ROOT = T.ROOT


def _perf(alphas, **kw):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        main.trading_engine.paper_balance = 10_000.0
        async with S() as db:
            for i, a in enumerate(alphas):
                db.add(T._order(models, i, "FRENZY_LITE", "FRENZY_LITE", entry_pair_alpha=a))
            await db.commit()
            out = json.loads(json.dumps(await main._compute_performance(db, **kw), default=str))
        await eng.dispose()
        return out
    return asyncio.run(run())


def test_performance_alpha_filter():
    rows = [True, True, False, None, None]
    assert _perf(rows)["total_trades"] == 5
    assert _perf(rows, alpha="ALPHA")["total_trades"] == 2
    assert _perf(rows, alpha="NON_ALPHA")["total_trades"] == 3                    # stamped False + unstamped (pre-Nov-4)
    assert _perf(rows, alpha="")["total_trades"] == 5 and _perf(rows, alpha="junk")["total_trades"] == 5


def test_order_apis_and_frenzy_rows_carry_the_flag(monkeypatch):
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert src.count('"entry_pair_alpha": getattr(o, \'entry_pair_alpha\', None),') == 2                # open + closed orders
    assert '"alpha": _pair_is_alpha_now(f.get(\'pair\')),' in src
    import main
    monkeypatch.setattr(main.binance_service.public_exchange, "markets",
                        {"CAP/USDT:USDT": {"info": {"underlyingSubType": ["Crypto", "Alpha"]}},
                         "BTC/USDT:USDT": {"info": {"underlyingSubType": ["PoW"]}}}, raising=False)
    assert main._pair_is_alpha_now("CAPUSDT") is True and main._pair_is_alpha_now("BTCUSDT") is False
    assert main._pair_is_alpha_now("NOPEUSDT") is None and main._pair_is_alpha_now(None) is None


def test_ui_mark_and_filter_on_every_surface():
    h = T._html()
    assert "function alphaMark(flag, atEntry)" in h
    assert h.count("${o.pair}${alphaMark(o.entry_pair_alpha, true)}</td>") == 2                       # open + closed tables
    assert "${alphaMark(_fz.alpha, false)}" in h                                                       # FRENZY pairs rows
    assert '<select id="alpha-filter"' in h
    assert "params.push(`alpha=${encodeURIComponent(alphaFilter)}`)" in h                              # on screen
    assert "_params1.push(`alpha=${encodeURIComponent(_alpha1)}`)" in h and "_fbits.push(`alpha=${_alpha1}`)" in h   # Analytics export
    assert "&alpha=${encodeURIComponent(_alphaSplit)}" in h and "_fbits.push(`alpha=${_alphaSplit}`)" in h        # Split Report export


def test_frenzy_flagged_pairs_table_and_exports_carry_the_mark():
    h = T._html()
    assert "${frenzyMark(f)} ${_fzE(f.pair)}${alphaMark(f.alpha, false)}</td>" in h                    # Performance → FRENZY Flagged Pairs
    assert "(frenzyMark(f) + ' ' + f.pair + (f.alpha === true ? ' α' : ''))" in h                       # both text exports (shared lines)


def test_transactions_carry_the_order_stamp():
    """(467) both rows of an Alpha trade (open + close) show α — from the order's own stamp, joined by order id."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import datetime as dt
    import main, models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        _mode = main.trading_engine.is_paper_mode
        main.trading_engine.is_paper_mode = True
        async def _noinit(db): return None
        main.trading_engine.initialize, _orig = _noinit, main.trading_engine.initialize
        try:
            async with S() as db:
                o1 = T._order(models, 0, "FRENZY_LITE_RETRY", "FRENZY_LITE_RETRY", entry_pair_alpha=True)
                o2 = T._order(models, 1, "MOMENTUM", None)
                db.add_all([o1, o2]); await db.commit()
                for o in (o1, o2):
                    for act in ("OPEN_LONG", "CLOSE_LONG"):
                        db.add(models.Transaction(order_id=o.id, pair=o.pair, action=act, price=1.0, quantity=1.0, investment=10.0, leverage=4.0,
                                                  notional_value=40.0, fee=0.01, timestamp=dt.datetime(2026, 11, 4, 12), is_paper=True))
                await db.commit()
                return await main.get_transactions(db)
        finally:
            main.trading_engine.initialize = _orig
            main.trading_engine.is_paper_mode = _mode
            await eng.dispose()
    rows = asyncio.run(run())
    got = sorted((r["pair"], r["action"], r["entry_pair_alpha"]) for r in rows)
    assert got == [("P0USDT", "CLOSE_LONG", True), ("P0USDT", "OPEN_LONG", True), ("P1USDT", "CLOSE_LONG", None), ("P1USDT", "OPEN_LONG", None)]
    assert "${t.pair}${alphaMark(t.entry_pair_alpha, true)}</td>" in T._html()


def test_frenzy_header_counts_alpha_pairs():
    """(468) the FRENZY group header shows how many flagged pairs are Alpha: "FRENZY pairs (9 · 5 α)" — nothing when none."""
    h = T._html()
    assert "const _nAlpha = _fzs.filter(f => f && f.alpha === true).length;" in h
    assert "🔥 FRENZY pairs (${_fzIdx.length}${_nAlpha ? ` · <span" in h and "${_nAlpha} α</span>` : ''}) · ${_nOn} ON" in h
