"""🧾 (DECISION_LOG 486) fund operating expense: shared pro rata, no share cancelled, booked per investor, statements keep it apart from the
trading result. Real endpoint on an in-memory DB with a monkeypatched portfolio value (no exchange calls)."""
import os
import sys
from datetime import date, datetime

import pytest
from sqlalchemy import select

import main
import models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from services import investor_statement as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def _add(db, name, shares):
    inv = models.Investor(name=name, shares=shares, total_deposited=shares, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()
    return inv


def _env(monkeypatch, equity, paper, free=1e9):
    async def _pv(_db):
        return equity

    async def _recalc(_db):
        return None
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", paper, raising=False)
    monkeypatch.setattr(main.trading_engine, "paper_balance", free, raising=False)
    monkeypatch.setattr(main.trading_engine, "_recalculate_paper_balance", _recalc, raising=False)


def test_split_sums_and_never_goes_negative():
    out = main.expense_split(100.0, [("a", 1.0), ("b", 1.0), ("c", 1.0)])
    assert sum(a for _, a in out) == pytest.approx(100.0) and all(a >= 0 for _, a in out)
    assert [a for _, a in out] == [33.33, 33.33, 33.34]
    assert main.expense_split(0.01, [("a", 1.0), ("b", 1.0)])[-1][1] >= 0


async def test_live_expense_books_slices_without_touching_shares(db, monkeypatch):
    a, b = await _add(db, "A", 750.0), await _add(db, "B", 250.0)
    _env(monkeypatch, equity=900.0, paper=False)                     # $100 already sent out of a $1,000 fund
    res = await main.fund_expense(main.FundExpenseIn(amount=100.0, category="servers", note="Nov hosting"), db)
    assert res["ok"] and not res["paper"]
    assert res["nav"] == pytest.approx(1.0) and res["nav_after"] == pytest.approx(0.9)   # NAV before = (equity + amount) / shares
    assert a.shares == 750.0 and b.shares == 250.0                     # no share cancelled
    rows = (await db.execute(select(models.InvestorLedger).where(models.InvestorLedger.type == "EXPENSE"))).scalars().all()
    assert sorted((r.investor_id, r.amount, r.shares_delta) for r in rows) == [(a.id, 75.0, 0.0), (b.id, 25.0, 0.0)]
    assert "servers" in rows[0].note and "Nov hosting" in rows[0].note
    fx = (await db.execute(select(models.FundExpense))).scalars().all()
    assert len(fx) == 1 and fx[0].is_paper is False and fx[0].amount == 100.0 and fx[0].category == "servers"
    assert a.total_withdrawn == 0.0 and a.total_deposited == 750.0     # never a withdrawal, never a deposit edit


async def test_paper_expense_preview_writes_nothing_and_free_balance_caps_it(db, monkeypatch):
    await _add(db, "A", 1000.0)
    _env(monkeypatch, equity=1000.0, paper=True, free=50.0)
    pv = await main.fund_expense(main.FundExpenseIn(amount=40.0, category="data", preview=True), db)
    assert pv["preview"] and pv["paper"] and pv["nav"] == pytest.approx(1.0) and pv["nav_after"] == pytest.approx(0.96)
    assert not (await db.execute(select(models.FundExpense))).scalars().all()
    with pytest.raises(main.HTTPException):
        await main.fund_expense(main.FundExpenseIn(amount=60.0, category="data"), db)      # more than the free paper balance
    with pytest.raises(main.HTTPException):
        await main.fund_expense(main.FundExpenseIn(amount=10.0, category="bonus"), db)     # unknown category
    res = await main.fund_expense(main.FundExpenseIn(amount=40.0, category="data"), db)
    fx = (await db.execute(select(models.FundExpense))).scalars().all()
    assert res["ok"] and len(fx) == 1 and fx[0].is_paper is True


def test_statement_keeps_the_expense_apart_from_the_trading_result():
    led = [dict(type="DEPOSIT", amount=1000.0, nav_at_time=1.0, shares_delta=1000.0, created_at=datetime(2026, 11, 2, 15)),
           dict(type="EXPENSE", amount=50.0, nav_at_time=1.1, shares_delta=0.0, created_at=datetime(2026, 11, 4, 15))]
    # NAV: +10 % trading by Nov 3, then the $50 expense (5 % of the $1,100 balance) on Nov 4, flat after
    snaps = {date(2026, 11, 1): 1.0, date(2026, 11, 2): 1.0, date(2026, 11, 3): 1.1, date(2026, 11, 4): 1.05, date(2026, 11, 5): 1.05}
    st = S.compute_statement("A", led, snaps, [], "2026-11", datetime(2026, 11, 5, 20), True)
    assert st.expenses == pytest.approx(50.0) and st.total_expenses == pytest.approx(50.0)
    assert st.result == pytest.approx(100.0)                           # trading result: +$100, the expense is NOT a trading loss
    assert st.end_balance == pytest.approx(1050.0)
    assert st.end_balance == pytest.approx(st.start_balance + st.deposits - st.withdrawals + st.result - st.expenses)
    assert st.ret_pct == pytest.approx(5.0)                            # the return is NET: 1,050 / 1,000
    assert st.withdrawals == 0.0 and st.deposits == pytest.approx(1000.0)
    assert st.total_profit == pytest.approx(50.0)                      # all-time profit is net
    d4 = [d for d in st.days if d.day == date(2026, 11, 4)][0]
    assert d4.expense == pytest.approx(50.0) and d4.pnl == pytest.approx(0.0) and d4.flow == 0.0
    assert "fund expenses" in S.summary_text(st) and "from trading" in S.summary_text(st)
    pdf = S.render_pdf(st)
    assert pdf[:5] == b"%PDF-"


def test_wiring_unbooked_check_paper_balance_and_ui():
    src = open(os.path.join(ROOT, "main.py")).read()
    assert 't in ("WITHDRAW", "CASHOUT", "EXPENSE")' in src                       # a booked live expense left the exchange
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "func.sum(FundExpense.amount)" in eng and "FundExpense.is_paper == True" in eng
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    for needle in ("openFundExpenseModal()", 'id="fund-expense-modal"', "/api/investors/expense", "t==='EXPENSE'",
                   "Fund expenses paid:", 'id="portfolio-expenses"'):
        assert needle in html, needle
    assert S.signed_flow(dict(type="EXPENSE", amount=10.0)) == 0.0 and S.expense_of(dict(type="EXPENSE", amount=10.0)) == 10.0


def test_tightest_statement_with_expenses_fits_one_page():
    """6 month rows + month in progress + paper → live + a withdrawal + fund expenses in two months (the extra column and summary line)."""
    from datetime import timedelta
    now = datetime(2026, 11, 6, 17)
    sn, d, v, i = {}, date(2025, 12, 31), 1.0, 0
    while d <= now.date():
        sn[d] = v
        v *= 1.003 if i % 4 else 0.995
        d += timedelta(days=1)
        i += 1
    nv = sn[date(2026, 11, 2)]
    led = [dict(type="DEPOSIT", amount=5000.0, nav_at_time=1.0, shares_delta=5000.0, created_at=datetime(2026, 1, 2, 15)),
           dict(type="EXPENSE", amount=120.0, nav_at_time=sn[date(2026, 9, 9)], shares_delta=0.0, created_at=datetime(2026, 9, 10, 15)),
           dict(type="WITHDRAW", amount=500.0, nav_at_time=nv, shares_delta=-500.0 / nv, created_at=datetime(2026, 11, 3, 15)),
           dict(type="EXPENSE", amount=80.0, nav_at_time=nv, shares_delta=0.0, created_at=datetime(2026, 11, 4, 15))]
    od, t, k = [], datetime(2026, 1, 2, 16), 0
    while t < now:
        od.append(dict(closed_at=t, pnl=4.0 if k % 3 else -6.0, pnl_percentage=0.6 if k % 3 else -0.9, is_paper=t < datetime(2026, 11, 3)))
        t += timedelta(hours=2)
        k += 1
    st = S.compute_statement("Maximiliano Alejandro de la Torre Fernández", led, sn, od, "2026-11", now, False)
    assert len(st.month_rows) == S.MAX_MONTH_ROWS and st.expenses == pytest.approx(80.0) and st.total_expenses == pytest.approx(200.0)
    assert any(r.get("expenses", 0) > 0 for r in st.month_rows)
    pdf = S.render_pdf(st)
    assert pdf.count(b"/Type /Page") - pdf.count(b"/Type /Pages") == 1


def test_review_fixes_486():
    assert main.expense_split(0.05, [(i, 1.0) for i in range(7)]) and sum(a for _, a in main.expense_split(0.05, [(i, 1.0) for i in range(7)])) == pytest.approx(0.05)
    src = open(os.path.join(ROOT, "main.py")).read()
    assert "_math.isfinite(amt)" in src and 'f"Fund expense #{fx.id}' in src and "await _nav_snapshot_once(db)" in src
    assert "a paper FUND EXPENSE lowers the paper equity without an order" in src                       # PnL calendar flow
    assert "paid fund expenses stay paid across a reset" in src
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count("FundExpense.is_paper == True") >= 2                                                # balance + BNB equity
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert "${_escHtml(e.note||'')}" in html and "futures wallet" in html


async def test_nan_amount_is_refused(db, monkeypatch):
    await _add(db, "A", 100.0)
    _env(monkeypatch, equity=100.0, paper=True)
    with pytest.raises(main.HTTPException):
        await main.fund_expense(main.FundExpenseIn(amount=float("nan"), category="data"), db)
