"""🛡 Go-live ④ (DECISION_LOG 292): the paper ↔ live switch is never a bare flag flip. PAPER → LIVE is refused while the bot runs, while
any paper position is OPEN, or when the read-only pre-flight fails (key · account · one-way mode · Multi-Assets off · every Binance
position tracked). LIVE → PAPER is refused while a live position is OPEN. All exchange reads mocked."""
import asyncio, os, sys
from datetime import datetime
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services.golive_guards import preflight_failures


def test_preflight_rules_pure():
    ok = dict(api_key_present=True, account_ok=True, dual_side=False, multi_assets=False, untracked=[], positions_ok=True)
    assert preflight_failures(**ok) == []
    assert any("API key" in m for m in preflight_failures(**dict(ok, api_key_present=False)))
    assert any("HEDGE" in m for m in preflight_failures(**dict(ok, dual_side=True)))
    assert any("could not be read" in m for m in preflight_failures(**dict(ok, dual_side=None)))
    assert any("Multi-Assets mode is ON" in m for m in preflight_failures(**dict(ok, multi_assets=True)))
    assert any("account could not be read" in m for m in preflight_failures(**dict(ok, account_ok=False)))
    assert any("positions could not be read" in m for m in preflight_failures(**dict(ok, positions_ok=False)))
    m = preflight_failures(**dict(ok, untracked=[{"pair": "FOOUSDT", "side": "LONG", "qty": 3, "reason": "NO_ROW"}]))
    assert len(m) == 1 and "FOOUSDT" in m[0]


def test_binance_mode_reads_are_read_only_and_parse():
    from services.binance_service import BinanceService
    svc = BinanceService.__new__(BinanceService)
    async def _nob(): return None
    svc._check_ban = _nob
    svc._detect_ban = staticmethod(lambda e: None)
    class Ex:
        def __init__(self, dual, multi): self.dual, self.multi = dual, multi
        async def fapiPrivateGetPositionSideDual(self, *a):
            if isinstance(self.dual, Exception): raise self.dual
            return {"dualSidePosition": self.dual}
        async def fapiPrivateGetMultiAssetsMargin(self, *a):
            return {"multiAssetsMargin": self.multi}
    svc.exchange = Ex(False, "false")
    assert asyncio.run(svc.get_position_mode_dual()) is False and asyncio.run(svc.get_multi_assets_mode()) is False
    svc.exchange = Ex(True, True)
    assert asyncio.run(svc.get_position_mode_dual()) is True and asyncio.run(svc.get_multi_assets_mode()) is True
    svc.exchange = Ex(RuntimeError("down"), None)
    assert asyncio.run(svc.get_position_mode_dual()) is None and asyncio.run(svc.get_multi_assets_mode()) is None
    class Ex2:
        def __init__(self, res): self.res = res
        async def fapiPrivateGetSymbolConfig(self, *a):
            if isinstance(self.res, Exception): raise self.res
            return self.res
    svc.exchange = Ex2([{"symbol": "AUSDT", "marginType": "CROSSED"}, {"symbol": "BUSDT", "marginType": "ISOLATED"}])
    assert asyncio.run(svc.get_isolated_margin_symbols()) == ["BUSDT"]
    svc.exchange = Ex2(RuntimeError("down"))
    assert asyncio.run(svc.get_isolated_margin_symbols()) is None


def _harness(monkeypatch, *, paper=True, running=False, rows=(), key="k", dual=False, multi=False, bal_ok=True, positions=(), isolated=()):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models, services.trading_engine as T
    engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
    S = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr(T.config.settings, "binance_api_key", key, raising=False)
    monkeypatch.setattr(T.config.settings, "binance_api_secret", key, raising=False)
    calls = []
    async def _bal(): calls.append("balance"); return {"ok": bal_ok}
    async def _dual(): calls.append("dual"); return dual
    async def _multi(): calls.append("multi"); return multi
    async def _pos(): calls.append("positions"); return None if positions is None else [dict(symbol=s, side=d, contracts=q, entry_price=10.0) for s, d, q in positions]
    async def _iso(): calls.append("isolated"); return None if isolated is None else list(isolated)
    async def _never(*a, **k): raise AssertionError("the switch must never place / cancel orders")
    for n, f in (("get_balance", _bal), ("get_position_mode_dual", _dual), ("get_multi_assets_mode", _multi), ("get_open_positions", _pos), ("get_isolated_margin_symbols", _iso),
                 ("create_market_order", _never), ("place_backstop_stop", _never), ("cancel_backstop_stop", _never), ("close_position", _never)):
        monkeypatch.setattr(T.binance_service, n, f)
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper; eng.is_running = running; eng.paper_balance = 1.0
    saved = []
    async def _save(db): saved.append(eng.is_paper_mode)
    eng.save_state = _save

    async def run(enabled):
        async with engine.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as s:
            for pair, d, is_paper in rows:
                s.add(models.Order(pair=pair, direction=d, status="OPEN", entry_price=1.0, current_price=1.0, investment=10.0, leverage=1.0,
                                   notional_value=10.0, quantity=10.0, confidence="STRONG_BUY", is_paper=is_paper, opened_at=datetime.utcnow()))
            await s.commit()
        try:
            async with S() as s:
                return await eng.set_paper_mode(enabled, s)
        finally:
            await engine.dispose()
    return T, eng, run, calls, saved


def _refused(T, run, enabled):
    with pytest.raises(T.ModeSwitchRefused) as ei:
        asyncio.run(run(enabled))
    return ei.value


def test_paper_to_live_refused_while_running_or_with_open_paper_rows(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch, running=True)
    e = _refused(T, run, False)
    assert e.status == 409 and "pause the bot" in str(e) and eng.is_paper_mode is True and calls == [] and saved == []
    T, eng, run, calls, saved = _harness(monkeypatch, rows=[("FOOUSDT", "LONG", True)])
    e = _refused(T, run, False)
    assert e.status == 409 and "paper position" in str(e) and eng.is_paper_mode is True and saved == []


@pytest.mark.parametrize("kw,needle", [
    (dict(key=""), "API key"),
    (dict(bal_ok=False), "account could not be read"),
    (dict(dual=True), "HEDGE"),
    (dict(multi=True), "Multi-Assets"),
    (dict(positions=None), "positions could not be read"),
    (dict(isolated=["FOOUSDT"]), "ISOLATED margin (FOOUSDT)"),
    (dict(isolated=None), "margin mode could not be read"),
    (dict(positions=[("FOO/USDT:USDT", "LONG", 3.0)]), "untracked Binance position FOOUSDT"),
])
def test_paper_to_live_refused_on_a_failed_preflight(monkeypatch, kw, needle):
    T, eng, run, calls, saved = _harness(monkeypatch, **kw)
    e = _refused(T, run, False)
    assert e.status == 412 and needle in str(e) and eng.is_paper_mode is True and saved == []


def test_paper_to_live_passes_a_clean_preflight(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch, rows=[("BARUSDT", "SHORT", False)], positions=[("BAR/USDT:USDT", "SHORT", 2.0)])
    out = asyncio.run(run(False))
    assert out == {"paper_mode": False, "changed": True} and eng.is_paper_mode is False and saved == [False]
    assert set(calls) == {"balance", "dual", "multi", "positions", "isolated"}


def test_live_to_paper_refused_with_live_rows_and_same_mode_is_a_noop(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch, paper=False, rows=[("BARUSDT", "SHORT", False)])
    e = _refused(T, run, True)
    assert e.status == 409 and "LIVE position" in str(e) and eng.is_paper_mode is False
    T, eng, run, calls, saved = _harness(monkeypatch, paper=False, rows=[("FOOUSDT", "LONG", True)])   # paper rows do not block going to paper
    assert asyncio.run(run(True)) == {"paper_mode": True, "changed": True} and eng.is_paper_mode is True and calls == []
    T, eng, run, calls, saved = _harness(monkeypatch, paper=True, running=True)
    assert asyncio.run(run(True)) == {"paper_mode": True, "changed": False} and saved == []


def test_endpoint_returns_the_refusal_as_4xx(monkeypatch):
    import main
    from fastapi import HTTPException
    from services.trading_engine import ModeSwitchRefused
    async def _init(db): return None
    async def _refuse(enabled, db): raise ModeSwitchRefused("LIVE pre-flight failed: account is in HEDGE", 412)
    monkeypatch.setattr(main.trading_engine, "initialize", _init)
    monkeypatch.setattr(main.trading_engine, "set_paper_mode", _refuse)
    with pytest.raises(HTTPException) as ei:
        asyncio.run(main.set_paper_mode(False, None))
    assert ei.value.status_code == 412 and "HEDGE" in ei.value.detail
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "if (!_pmRes.ok) {" in html and "e.target.checked = !isPaperMode;" in html            # the UI shows the reason, keeps the toggle


def test_live_to_paper_refused_while_an_untracked_position_is_known(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch, paper=False)
    eng._mark_untracked("FOOUSDT", "LONG", 3.0, 1.0, "NO_ROW", "RECONCILE")
    e = _refused(T, run, True)
    assert e.status == 409 and "FOOUSDT" in str(e) and eng.is_paper_mode is False


def test_one_switch_at_a_time_and_start_refused_while_switching(monkeypatch):
    # 2026-10-14 go-live: repeated toggle clicks ran two PAPER → LIVE pre-flights concurrently (DECISION_LOG 293)
    T, eng, run, calls, saved = _harness(monkeypatch)
    gate = {}
    async def _slow_bal():
        calls.append("balance"); gate["in"].set(); await gate["go"].wait(); return {"ok": True}
    monkeypatch.setattr(T.binance_service, "get_balance", _slow_bal)
    import models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    async def go():
        gate["in"], gate["go"] = asyncio.Event(), asyncio.Event()
        engine = create_async_engine("sqlite+aiosqlite://", poolclass=StaticPool)
        S = async_sessionmaker(engine, expire_on_commit=False)
        async with engine.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with S() as s1, S() as s2, S() as s3:
            first = asyncio.create_task(eng.set_paper_mode(False, s1))
            await gate["in"].wait()                                   # the first switch is inside its pre-flight
            assert eng.mode_switch_in_progress()
            with pytest.raises(T.ModeSwitchRefused) as e2:
                await eng.set_paper_mode(False, s2)                   # a second click → refused, not a second pre-flight
            with pytest.raises(T.ModeSwitchRefused) as e3:
                await eng.start(s3)                                   # start during the switch → refused
            gate["go"].set()
            r = await first
        await engine.dispose()
        return e2.value, e3.value, r
    e2, e3, r = asyncio.run(go())
    assert e2.status == 409 and "already in progress" in str(e2)
    assert e3.status == 409 and eng.is_running is False
    assert r == {"paper_mode": False, "changed": True} and calls.count("balance") == 1 and not eng.mode_switch_in_progress()


def test_bot_started_during_the_preflight_blocks_the_flip(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch)
    async def _bal_and_start():
        calls.append("balance"); eng.is_running = True; return {"ok": True}   # e.g. a start that slipped in before the lock existed
    monkeypatch.setattr(T.binance_service, "get_balance", _bal_and_start)
    e = _refused(T, run, False)
    assert e.status == 409 and "started during the pre-flight" in str(e) and eng.is_paper_mode is True and saved == []


def test_start_endpoint_and_ui_handle_the_refusal():
    import inspect, main
    src = inspect.getsource(main.start_bot)
    assert "ModeSwitchRefused" in src and "HTTPException" in src
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert "e.target.disabled = true" in html and "e.target.disabled = false" in html and "Request refused" in html


def test_a_failing_or_slow_preflight_releases_the_lock(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch)
    async def _boom(): raise RuntimeError("binance down")
    monkeypatch.setattr(T.binance_service, "get_balance", _boom)
    with pytest.raises(RuntimeError):
        asyncio.run(run(False))
    assert not eng.mode_switch_in_progress() and eng.is_paper_mode is True
    T, eng, run, calls, saved = _harness(monkeypatch)
    async def _hang(): await asyncio.sleep(60); return {"ok": True}   # e.g. an IP-ban sleep inside a read
    monkeypatch.setattr(T.binance_service, "get_balance", _hang)
    monkeypatch.setattr(T, "MODE_SWITCH_PREFLIGHT_TIMEOUT_S", 0.05)
    e = _refused(T, run, False)
    assert e.status == 412 and "timed out" in str(e) and not eng.mode_switch_in_progress() and eng.is_paper_mode is True


def test_live_to_paper_refused_while_running(monkeypatch):
    T, eng, run, calls, saved = _harness(monkeypatch, paper=False, running=True)
    e = _refused(T, run, True)
    assert e.status == 409 and "pause the bot before switching to PAPER" in str(e) and eng.is_paper_mode is False and saved == []
