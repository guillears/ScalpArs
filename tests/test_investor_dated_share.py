"""🗓 (DECISION_LOG 490) Add Investor with NO new money: a dated share of an existing investor's capital — shares at the joined day's opening
NAV move from the giver, both ledger rows dated that day, nothing enters the fund, nobody else moves."""
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

import main
import models

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pure_core_moves_the_result_since_that_day():
    # $1,000 of Guille's capital from a day at NAV 0.45; today 0.3883 → Maria ≈ −$137, Guille's P&L improves by the same
    p = main.dated_share_transfer(1000.0, 0.45, 0.3883, 15897.3653, 15897.3653, 10100.0)
    assert p["shares"] == pytest.approx(1000 / 0.45, rel=1e-9) and p["basis"] == pytest.approx(1000.0)
    assert p["pnl"] == pytest.approx(1000 / 0.45 * 0.3883 - 1000, abs=0.01) and p["pnl"] < 0
    assert p["src_pnl_change"] == pytest.approx(-p["pnl"], abs=0.02)
    assert p["src_value_before"] - p["src_value_after"] == pytest.approx(p["value_now"], abs=0.02)   # value only changes hands
    with pytest.raises(ValueError, match="held"):
        main.dated_share_transfer(1000.0, 0.45, 0.3883, 15897.0, 100.0, 10100.0)                    # he held too few shares that day


async def _setup(db):
    today3 = (datetime.now(timezone.utc) - timedelta(hours=3)).date()
    start = datetime(today3.year, today3.month, today3.day, 3) - timedelta(days=20)
    g = models.Investor(name="Guille", shares=1000.0, total_deposited=1000.0, total_withdrawn=0.0, created_at=start)
    other = models.Investor(name="Ana", shares=500.0, total_deposited=500.0, total_withdrawn=0.0, created_at=start)
    db.add_all([g, other])
    await db.flush()
    db.add_all([models.InvestorLedger(investor_id=g.id, type="DEPOSIT", amount=1000.0, nav_at_time=1.0, shares_delta=1000.0, created_at=start),
                models.InvestorLedger(investor_id=other.id, type="DEPOSIT", amount=500.0, nav_at_time=1.0, shares_delta=500.0, created_at=start)])
    for k, v in ((12, 0.50), (6, 0.46), (5, 0.45)):
        db.add(models.NavSnapshot(date=(today3 - timedelta(days=k)).isoformat(), portfolio_value=v * 1500, total_shares=1500.0, nav_per_share=v))
    await db.flush()
    return today3, g, other


async def test_endpoint_preview_then_book(db, monkeypatch):
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)

    async def _nav(_db):
        return 0.40
    monkeypatch.setattr(main, "_get_nav_per_share", _nav)
    today3, g, other = await _setup(db)
    joined = (today3 - timedelta(days=5)).isoformat()                     # opening NAV = the close of the day before = 0.46
    body = dict(name="Maria", joined=joined, deposit_amount=230.0, from_investor_id=g.id)
    p = await main.add_investor(body=main.InvestorCreate(**body, preview=True), db=db)
    assert p["preview"] and p["nav_then"] == pytest.approx(0.46) and p["new"]["shares"] == pytest.approx(500.0)
    assert p["new"]["value_now"] == pytest.approx(200.0) and p["new"]["pnl"] == pytest.approx(-30.0)
    assert p["giver"]["pnl_change"] == pytest.approx(30.0, abs=0.01)
    assert not (await db.execute(select(models.Investor).where(models.Investor.name == "Maria"))).scalars().all()   # preview wrote nothing
    r = await main.add_investor(body=main.InvestorCreate(**body), db=db)
    m = (await db.execute(select(models.Investor).where(models.Investor.id == r["id"]))).scalar_one()
    g = (await db.execute(select(models.Investor).where(models.Investor.name == "Guille"))).scalar_one()     # re-read after the commit
    other = (await db.execute(select(models.Investor).where(models.Investor.name == "Ana"))).scalar_one()
    assert m.shares == pytest.approx(500.0) and g.shares == pytest.approx(500.0) and other.shares == pytest.approx(500.0)   # Ana untouched
    assert m.total_deposited == pytest.approx(230.0) and g.total_deposited == pytest.approx(770.0)
    jd = datetime.strptime(joined, "%Y-%m-%d")
    assert m.created_at == datetime(jd.year, jd.month, jd.day, 3)
    led = (await db.execute(select(models.InvestorLedger).where(models.InvestorLedger.type.in_(("TRANSFER_OUT", "TRANSFER_IN"))))).scalars().all()
    assert sorted((x.type, x.investor_id, round(x.shares_delta, 4)) for x in led) == [("TRANSFER_IN", m.id, 500.0), ("TRANSFER_OUT", g.id, -500.0)]
    assert all(x.created_at == datetime(jd.year, jd.month, jd.day, 3) and x.nav_at_time == pytest.approx(0.46) for x in led)   # DATED that day
    assert sum(i.shares for i in (await db.execute(select(models.Investor))).scalars().all()) == pytest.approx(1500.0)        # no share minted


async def test_refusals(db, monkeypatch):
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)

    async def _nav(_db):
        return 0.40
    monkeypatch.setattr(main, "_get_nav_per_share", _nav)
    today3, g, other = await _setup(db)
    old = (today3 - timedelta(days=30)).isoformat()                       # before the NAV history and before Guille joined
    with pytest.raises(main.HTTPException):
        await main.add_investor(body=main.InvestorCreate(name="Maria", joined=old, deposit_amount=100.0, from_investor_id=g.id, preview=True), db=db)
    with pytest.raises(main.HTTPException):                                # more than his capital that day
        await main.add_investor(body=main.InvestorCreate(name="Maria", joined=(today3 - timedelta(days=5)).isoformat(), deposit_amount=5000.0,
                                                         from_investor_id=g.id, preview=True), db=db)
    with pytest.raises(main.HTTPException):                                # never also a founding allocation
        await main.add_investor(body=main.InvestorCreate(name="Maria", deposit_amount=100.0, from_investor_id=g.id, founding=True, preview=True), db=db)


async def test_all_existing_investors_pro_rata(db, monkeypatch):
    """(490, operator: "All existing investors, pro rata by ownership") each giver gives by the shares held THAT day; every ownership %
    shrinks by the same factor; nothing is minted; each giver gets its own dated TRANSFER_OUT."""
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)

    async def _nav(_db):
        return 0.40
    monkeypatch.setattr(main, "_get_nav_per_share", _nav)
    today3, g, other = await _setup(db)                                    # Guille 1,000 sh · Ana 500 sh (2 : 1)
    joined = (today3 - timedelta(days=5)).isoformat()                      # opening NAV 0.46
    body = dict(name="Maria", joined=joined, deposit_amount=300.0, from_all_prorata=True)
    p = await main.add_investor(body=main.InvestorCreate(**body, preview=True), db=db)
    assert p["prorata"] and sorted((x["name"], x["amount"]) for x in p["givers"]) == [("Ana", 100.0), ("Guille", 200.0)]
    assert sum(x["pnl_change"] for x in p["givers"]) == pytest.approx(-p["new"]["pnl"], abs=0.02)
    r = await main.add_investor(body=main.InvestorCreate(**body), db=db)
    m = (await db.execute(select(models.Investor).where(models.Investor.id == r["id"]))).scalar_one()
    g = (await db.execute(select(models.Investor).where(models.Investor.name == "Guille"))).scalar_one()
    other = (await db.execute(select(models.Investor).where(models.Investor.name == "Ana"))).scalar_one()
    assert m.shares == pytest.approx(300 / 0.46, rel=1e-6)
    assert g.shares / other.shares == pytest.approx(2.0, rel=1e-9)        # the 2 : 1 ownership kept between the givers
    assert g.shares + other.shares + m.shares == pytest.approx(1500.0)
    outs = (await db.execute(select(models.InvestorLedger).where(models.InvestorLedger.type == "TRANSFER_OUT"))).scalars().all()
    assert sorted(x.investor_id for x in outs) == sorted([g.id, other.id])


async def test_review_fixes_490(db, monkeypatch):
    """(490 deep review) 🔴 a giver who withdrew AFTER the day can only give what he kept the whole time (his history never goes negative)
    · fund expenses after the day refuse · a ledger that does not reconcile refuses · a 'share' without a giver is a 400, never a deposit
    · one investor AND all at once is a 400."""
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)

    async def _nav(_db):
        return 0.40
    monkeypatch.setattr(main, "_get_nav_per_share", _nav)
    today3, g, other = await _setup(db)
    d5 = today3 - timedelta(days=5)
    at5 = datetime(d5.year, d5.month, d5.day, 3)
    # Guille withdraws 900 shares on D+2 and deposits them back on D+4 → the lowest running count after D is 100 shares
    db.add_all([models.InvestorLedger(investor_id=g.id, type="WITHDRAW", amount=414.0, nav_at_time=0.46, shares_delta=-900.0, created_at=at5 + timedelta(days=2)),
                models.InvestorLedger(investor_id=g.id, type="DEPOSIT", amount=414.0, nav_at_time=0.46, shares_delta=900.0, created_at=at5 + timedelta(days=4))])
    await db.flush()
    with pytest.raises(main.HTTPException):                                # 435 shares > the 100 he kept through D+2 … D+4
        await main.add_investor(body=main.InvestorCreate(name="Maria", joined=d5.isoformat(), deposit_amount=200.0, from_investor_id=g.id,
                                                         capital_source="share", preview=True), db=db)
    ok = await main.add_investor(body=main.InvestorCreate(name="Maria", joined=d5.isoformat(), deposit_amount=40.0, from_investor_id=g.id,
                                                          capital_source="share", preview=True), db=db)
    assert ok["preview"] and ok["new"]["shares"] == pytest.approx(40 / 0.46, rel=1e-6)          # 87 shares ≤ 100 kept
    # an expense after the day refuses (its slices are not re-split)
    db.add(models.InvestorLedger(investor_id=other.id, type="EXPENSE", amount=5.0, nav_at_time=0.45, shares_delta=0.0, created_at=at5 + timedelta(days=1)))
    await db.flush()
    with pytest.raises(main.HTTPException, match="expenses"):
        await main.add_investor(body=main.InvestorCreate(name="Maria", joined=d5.isoformat(), deposit_amount=40.0, capital_source="all",
                                                         from_all_prorata=True, preview=True), db=db)
    # 'share' without a giver → 400 (never booked as new money) · both sources at once → 400
    with pytest.raises(main.HTTPException, match="whose capital"):
        await main.add_investor(body=main.InvestorCreate(name="Maria", deposit_amount=40.0, capital_source="share", preview=True), db=db)
    with pytest.raises(main.HTTPException, match="not both"):
        await main.add_investor(body=main.InvestorCreate(name="Maria", deposit_amount=40.0, from_investor_id=g.id, from_all_prorata=True, preview=True), db=db)
    # a ledger that does not reconcile with the shares now refuses
    other.shares = 777.0
    await db.flush()
    with pytest.raises(main.HTTPException, match="does not match"):
        await main.add_investor(body=main.InvestorCreate(name="Maria", joined=d5.isoformat(), deposit_amount=10.0, from_investor_id=other.id,
                                                         capital_source="share", preview=True), db=db)


def test_ui_wiring():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for needle in ('id="investor-add-source"', 'id="investor-add-giver"', "function _addSourceChanged()", "async function _addSharePreview(",
                   "from_all_prorata: _srcV === 'all'", "Part of an existing investor's capital — no new money",
                   "Part of ALL existing investors' capital, pro rata by ownership — no new money"):
        assert needle in html, needle
