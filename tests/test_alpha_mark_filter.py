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
