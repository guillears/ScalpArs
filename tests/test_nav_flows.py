"""📈 Oct-30 (DECISION_LOG 422): outside money never reads as profit — pure math, the paper top-up mint, the one-time rebuild of the 10-26 case, and the live hold."""
import asyncio
from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import main
import models
from services import nav_flows as F

T0 = datetime(2026, 10, 23, 20, 0)
TOPUP = datetime(2026, 10, 26, 13, 50)


def test_pure_math():
    steps, final = F.rebuild_plan(3000.0, [(TOPUP, 7100.0)], [(datetime(2026, 10, 24), -1348.5), (datetime(2026, 10, 27), 1359.59)])
    t, after, nav, eq = steps[0]
    assert eq == pytest.approx(1651.5) and nav == pytest.approx(0.5505) and final == pytest.approx(3000 + 7100 / 0.5505) and after == final
    assert F.shares_at(datetime(2026, 10, 25), 3000.0, steps) == 3000.0 and F.shares_at(TOPUP, 3000.0, steps) == final
    p = F.prorata_mint({1: 75.0, 2: 25.0}, 200.0, 2.0)
    assert p[1] == (75.0, 150.0) and p[2] == (25.0, 50.0)
    with pytest.raises(ValueError):
        F.prorata_mint({1: 1.0}, -100.0, 1.0)
    assert F.unbooked_flow(1000.0, 1000.0, 5000.0) is None and F.unbooked_flow(1000.0, 0.0, 5000.0) == 1000.0 and F.unbooked_flow(10.0, 0.0, 5000.0) is None


async def _db():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as c:
        await c.run_sync(models.Base.metadata.create_all)
    return async_sessionmaker(eng, expire_on_commit=False)()


async def _case(db, monkeypatch, investors=1, ledger=1, realized=10111.09, opening=datetime(2026, 10, 23, 19, 0), flow=("2900.0", "10000.0", TOPUP)):
    db.add(models.BotState(runtime_initial_total_usd=3000.0, nav_epoch_start="2026-10-23", nav_epoch_started_at=T0, founding_allocated_usd=2012.02))
    for k in range(investors):
        db.add(models.Investor(name=f"I{k}", shares=2012.02, total_deposited=2012.02, total_withdrawn=0.0))
    await db.flush()
    for k in range(ledger):
        db.add(models.InvestorLedger(investor_id=1, type="FOUNDING", amount=2012.02, nav_at_time=1.0, shares_delta=2012.02, created_at=opening))
    db.add(models.ConfigChangeLog(field="paper_balance", old_value=flow[0], new_value=flow[1], changed_at=flow[2]))
    base = dict(pair="AUSDT", direction="LONG", status="CLOSED", is_paper=True, entry_price=1.0, quantity=1.0, investment=1.0, leverage=1, confidence="HIGH", notional_value=1.0)
    for o, c, p in ((datetime(2026, 10, 24), datetime(2026, 10, 24, 1), -1348.5), (datetime(2026, 10, 27), datetime(2026, 10, 27, 1), 1359.59)):
        db.add(models.Order(opened_at=o, closed_at=c, pnl=p, **base))
    for d, v, u in (("2026-10-24", 1651.5, datetime(2026, 10, 24, 12)), ("2026-10-26", 8751.5, datetime(2026, 10, 26, 23)), ("2026-10-30", 10049.0, datetime(2026, 10, 30, 23))):
        db.add(models.NavSnapshot(date=d, portfolio_value=v, total_shares=2012.02, nav_per_share=v / 2012.02, updated_at=u))
    await db.flush()

    async def comps(_d, fresh=True):
        return {"realized_equity": realized}
    monkeypatch.setattr(main, "_portfolio_components", comps)
    monkeypatch.setattr(main.config.trading_config, "paper_balance", 10000.0)
    monkeypatch.setattr(main.config.trading_config, "paper_bnb_initial_usd", 100.0)


def test_rebuild_restates_once_and_keeps_values(monkeypatch):
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", True)

    async def run():
        db = await _db()
        await _case(db, monkeypatch)
        assert await main._nav_rebuild_once(db) is True
        await db.commit()
        inv = (await db.execute(select(models.Investor))).scalars().one()
        s = {x.date: x for x in (await db.execute(select(models.NavSnapshot))).scalars().all()}
        assert inv.total_deposited == pytest.approx(10100.0) and inv.shares == pytest.approx(3000 + 7100 / 0.5505)
        assert s["2026-10-24"].nav_per_share == pytest.approx(0.5505, abs=1e-5) and s["2026-10-24"].portfolio_value == 1651.5
        assert s["2026-10-30"].nav_per_share == pytest.approx(10049 / inv.shares, abs=1e-5) and s["2026-10-30"].updated_at == datetime(2026, 10, 30, 23)
        led = (await db.execute(select(models.InvestorLedger))).scalars().all()
        assert sum(x.shares_delta for x in led) == pytest.approx(inv.shares, abs=0.01)
        shares = inv.shares
        assert await main._nav_rebuild_once(db) is False and inv.shares == shares     # once
        st = (await db.execute(select(models.BotState))).scalars().one()
        assert st.founding_allocated_usd == 3000.0
    asyncio.run(run())


@pytest.mark.parametrize("kw", [dict(investors=2), dict(ledger=2), dict(realized=10500.0), dict(opening=datetime(2026, 10, 28)),
                                dict(flow=("2900.0", "10000.0", datetime(2026, 11, 3))), dict(flow=("2900.0", "9000.0", TOPUP))])
def test_rebuild_refuses_data_that_does_not_fit(monkeypatch, kw):
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", True)

    async def run():
        db = await _db()
        await _case(db, monkeypatch, **kw)
        assert await main._nav_rebuild_once(db) is False
        inv = (await db.execute(select(models.Investor))).scalars().first()
        assert inv.shares == 2012.02
    asyncio.run(run())


def test_rebuild_never_runs_live(monkeypatch):
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", False)

    async def run():
        db = await _db()
        await _case(db, monkeypatch)
        assert await main._nav_rebuild_once(db) is False
    asyncio.run(run())


def test_topup_mint_keeps_nav_and_ownership():
    async def run():
        db = await _db()
        db.add(models.Investor(name="A", shares=75.0, total_deposited=75.0, total_withdrawn=0.0))
        db.add(models.Investor(name="B", shares=25.0, total_deposited=25.0, total_withdrawn=0.0))
        await db.flush()
        assert await main._paper_topup_mint(db, 300.0, 2.0, "t") is True
        a, b = (await db.execute(select(models.Investor).order_by(models.Investor.id))).scalars().all()
        assert (a.shares, b.shares) == (187.5, 62.5) and (a.total_deposited, b.total_deposited) == (300.0, 100.0)
        assert (200.0 + 300.0) / (a.shares + b.shares) == 2.0                         # value 200 at NAV 2 + 300: NAV unchanged
        assert await main._paper_topup_mint(db, -100.0, 2.0, "t") is True and a.shares + b.shares == 200.0   # a reduction burns
    asyncio.run(run())


def test_live_gap_flags_but_never_holds(monkeypatch):
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", False)

    async def run():
        db = await _db()
        db.add(models.BotState(nav_epoch_start="2026-10-23", nav_epoch_started_at=T0))
        db.add(models.Investor(name="A", shares=100.0, total_deposited=100.0, total_withdrawn=0.0))
        await db.flush()

        async def pv(_):
            return 1100.0

        async def flows(_d, _s, **k):
            return [(1, 1000.0)]
        monkeypatch.setattr(main, "_get_portfolio_value", pv)
        monkeypatch.setattr(main, "_external_flow_rows", flows)
        assert await main._nav_snapshot_once(db) is True and main._NAV_UNBOOKED["usd"] == 1000.0
        db.add(models.InvestorLedger(investor_id=1, type="DEPOSIT", amount=1000.0, nav_at_time=1.0, shares_delta=1000.0, created_at=datetime(2026, 10, 24)))
        await db.flush()
        assert await main._nav_snapshot_once(db) is True and main._NAV_UNBOOKED["usd"] is None
    asyncio.run(run())


def test_surfaces_and_save_order():
    h = open("templates/index.html", encoding="utf-8").read()
    i = h.index("function _buildNavHistoryLines")
    assert h.count('id="nav-unbooked"') == 1 and "getElementById('nav-unbooked')" in h and h.count("lines.push(..._buildNavHistoryLines(invData, hr, hr2));") == 2
    assert h.index("does not match the investor ledger`", i) < h.index("if (!hist.length) {", i)     # shown with no history too
    m = open("main.py", encoding="utf-8").read()
    u = m[m.index("async def update_config"):]
    assert u.index("[NAV_TOPUP_UNPRICED]") < u.index("if save_trading_config(new_config):") < u.index("_paper_topup_mint(_pdb") < u.index("_cfg_mod.trading_config = new_config") < u.index("db.add(ConfigChangeLog(")


def test_chart_refreshes_in_place():
    h = open("templates/index.html", encoding="utf-8").read()
    i = h.index("const hwm = data.nav_high_water;")
    j = h.index("window._navChart = new Chart(canvas", i)
    assert "_c.update('none');" in h[i:j] and "return;" in h[i:j]


def test_paper_preview_warns_about_double_counting():
    m = open("main.py", encoding="utf-8").read()
    i = m.index("async def investor_deposit")
    assert "do not do both" in m[i:i + 1800]
