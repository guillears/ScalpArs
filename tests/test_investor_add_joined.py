"""🗓 (DECISION_LOG 488) Add Investor with a JOINED date: today by default, or a chosen past day; never the future; never a date that pulls
an older holder's ledger rows (recycled id) into the new investor."""
import os
from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import main
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = date(2026, 11, 6)


def test_pure_rule():
    assert main.add_joined_at(None, TODAY, []) is None and main.add_joined_at("  ", TODAY, []) is None
    assert main.add_joined_at("2026-11-06", TODAY, []) is None                      # today = keep "now"
    assert main.add_joined_at("2026-10-23", TODAY, []) == datetime(2026, 10, 23, 3, 0, 0)   # 00:00 UTC-3 stored as 03:00 UTC
    with pytest.raises(ValueError, match="future"):
        main.add_joined_at("2026-11-07", TODAY, [])
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        main.add_joined_at("23/10/2026", TODAY, [])
    with pytest.raises(ValueError, match="older ledger entry"):
        main.add_joined_at("2026-10-01", TODAY, [datetime(2026, 10, 5, 12)])        # an old holder's row after the chosen day
    assert main.add_joined_at("2026-10-10", TODAY, [datetime(2026, 10, 5, 12)]) == datetime(2026, 10, 10, 3)


async def test_endpoint_sets_created_at_and_refuses_the_future(db, monkeypatch):
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)
    past = (datetime.now(timezone.utc) - timedelta(hours=3) - timedelta(days=10)).date()
    res = await main.add_investor(main.InvestorCreate(name="Ana", joined=past.isoformat()), db)
    inv = (await db.execute(select(models.Investor).where(models.Investor.id == res["id"]))).scalar_one()
    assert inv.created_at == datetime(past.year, past.month, past.day, 3, 0, 0) and res["joined"] == past.isoformat()
    res2 = await main.add_investor(main.InvestorCreate(name="Bea"), db)                 # blank → now (unchanged behaviour)
    inv2 = (await db.execute(select(models.Investor).where(models.Investor.id == res2["id"]))).scalar_one()
    assert abs((inv2.created_at - datetime.utcnow()).total_seconds()) < 120                  # "now" (naive UTC)
    future = (datetime.now(timezone.utc) - timedelta(hours=3) + timedelta(days=2)).date()
    with pytest.raises(main.HTTPException):
        await main.add_investor(main.InvestorCreate(name="Cid", joined=future.isoformat()), db)
    assert not (await db.execute(select(models.Investor).where(models.Investor.name == "Cid"))).scalars().all()   # nothing created


async def test_refused_after_the_id_exists_leaves_nothing(db, monkeypatch):
    """the recycled-id refusal happens AFTER the INSERT flush — the request rolls back: no investor, no ledger row."""
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)
    real = main.add_joined_at
    past = (datetime.now(timezone.utc) - timedelta(hours=3) - timedelta(days=5)).date().isoformat()
    calls = {"n": 0}

    def second_call_refuses(j, t, p):   # call 1 = the pre-check (passes) · call 2 = the post-flush scope check (refuses, as a recycled id would)
        calls["n"] += 1
        if calls["n"] == 2:
            raise ValueError("An older ledger entry under this investor id would join the new investor's history — choose a later date")
        return real(j, t, p)
    monkeypatch.setattr(main, "add_joined_at", second_call_refuses)
    with pytest.raises(main.HTTPException):
        await main.add_investor(main.InvestorCreate(name="Dan", joined=past), db)
    await db.rollback()                                                                 # get_db rolls back on the raised exception
    assert not (await db.execute(select(models.Investor).where(models.Investor.name == "Dan"))).scalars().all()
    assert not (await db.execute(select(models.InvestorLedger))).scalars().all()


def test_ui_wiring():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert 'id="investor-add-joined"' in html and "_jEl.value !== _jEl.dataset.def" in html
    assert "_jEl.value = _t3; _jEl.max = _t3; _jEl.dataset.def = _t3;" in html


def test_backdate_preview_is_a_transfer_between_investors():
    """(488, operator option 1) information only: Maria $1,000 priced at NAV 0.45 while today's NAV is 0.3883 — she shows a loss, the
    existing investors gain exactly that, the fund total is today's + the deposit."""
    b = main.backdate_preview(1000.0, 0.3883, 0.45, [("Guille", 15897.3653)])
    assert b["new_shares"] == pytest.approx(1000 / 0.45, rel=1e-6)
    assert b["fund_after"] == pytest.approx(15897.3653 * 0.3883 + 1000, abs=0.01)
    assert b["pnl"] < 0 and sum(o["delta"] for o in b["others"]) == pytest.approx(-b["pnl"], abs=0.02)
    assert b["value"] + sum(o["after"] for o in b["others"]) == pytest.approx(b["fund_after"], abs=0.02)
    up = main.backdate_preview(1000.0, 0.50, 0.40, [("A", 600.0), ("B", 400.0)])          # the fund rose since → she gains, they give it up
    assert up["pnl"] > 0 and all(o["delta"] < 0 for o in up["others"])
    assert up["others"][0]["delta"] == pytest.approx(1.5 * up["others"][1]["delta"], abs=0.02)   # pro rata by shares


async def test_preview_returns_the_backdate_block_and_writes_nothing(db, monkeypatch):
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)

    async def _nav(_db):
        return 0.40
    monkeypatch.setattr(main, "_get_nav_per_share", _nav)
    db.add(models.Investor(name="Guille", shares=1000.0, total_deposited=1000.0, total_withdrawn=0.0))
    today3 = (datetime.now(timezone.utc) - timedelta(hours=3)).date()
    for k, v in ((12, 0.50), (6, 0.46), (5, 0.45)):
        db.add(models.NavSnapshot(date=(today3 - timedelta(days=k)).isoformat(), portfolio_value=v * 1000, total_shares=1000.0, nav_per_share=v))
    await db.flush()
    joined = (today3 - timedelta(days=5)).isoformat()                                   # NAV at its start = the close of the day before (0.46)
    p = await main.add_investor(body=main.InvestorCreate(name="Maria", joined=joined, deposit_amount=1000.0, preview=True), db=db)   # keyword call = the real lock-free preview path
    assert not main._investor_lock.locked()
    assert p["preview"] and p["backdate"]["nav_then"] == pytest.approx(0.46) and p["backdate"]["pnl"] < 0
    assert not (await db.execute(select(models.Investor).where(models.Investor.name == "Maria"))).scalars().all()
    p2 = await main.add_investor(main.InvestorCreate(name="Maria", deposit_amount=1000.0, preview=True), db)   # no joined date → no block
    assert p2["backdate"] is None
