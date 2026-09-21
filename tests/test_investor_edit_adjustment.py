"""💰 Sep-21 — a deposit-total EDIT is a CORRECTION, never a cash flow.

OPERATOR-FOUND BUG. Creating an investor at $3,338.80 and then editing the figure to $3,000 —
ONE edit — booked a $338.80 WITHDRAW of money that never left the account. Because the UI computes
`pnl = value_usd - total_deposited + total_withdrawn`, the row then read +$677.60 (+22.59%) against
a true +$338.80 (+11.29%). The NAV/share of 1.1129 was correct the whole time; only the investor
row double-counted.

THE INVARIANT THIS FILE EXISTS TO PIN: `total_withdrawn` is not an output of an edit. Real money
movement goes through the deposit/withdraw endpoints. An edit moves `shares` and `total_deposited`
and nothing else.
"""
import pytest

from main import investor_edit_adjustment as adj


def test_the_operator_case_reduces_shares_and_deposited_only():
    """3338.80 -> 3000 at NAV 1.0: exactly the reported edit."""
    r = adj(3338.80, 3000.00, nav=1.0, shares=3338.80)
    assert r["delta"] == pytest.approx(-338.80)
    assert r["shares_delta"] == pytest.approx(-338.80)
    assert "withdrawn" not in r          # the whole bug in one assert


def test_the_operator_case_yields_the_correct_pnl():
    """End-to-end arithmetic: the row must read +$338.80 (+11.29%), not +$677.60 (+22.59%)."""
    deposited, shares, withdrawn = 3338.80, 3338.80, 0.0
    r = adj(deposited, 3000.00, nav=1.0, shares=shares)
    deposited += r["delta"]
    shares += r["shares_delta"]
    # withdrawn is deliberately NOT updated — that is the fix
    value = 3338.80                                        # portfolio grew to this
    pnl = value - deposited + withdrawn
    assert deposited == pytest.approx(3000.00)
    assert shares == pytest.approx(3000.00)
    assert withdrawn == 0.0
    assert pnl == pytest.approx(338.80)                    # was 677.60 before the fix
    assert pnl / deposited == pytest.approx(0.1129, abs=1e-4)   # matches NAV/share 1.1129


def test_increase_adds_shares_at_nav():
    r = adj(1000.0, 1500.0, nav=1.25, shares=800.0)
    assert r["delta"] == pytest.approx(500.0)
    assert r["shares_delta"] == pytest.approx(400.0)       # 500 / 1.25


def test_edit_is_reversible_round_trip_leaves_no_residue():
    """Edit up then back down must restore deposited AND shares exactly, with no phantom flow.

    Before the fix this was lossy: the down-leg wrote to total_withdrawn, so a round trip left a
    permanent phantom withdrawal behind.
    """
    deposited, shares, withdrawn = 3000.0, 3000.0, 0.0
    up = adj(deposited, 3500.0, nav=1.0, shares=shares)
    deposited += up["delta"]; shares += up["shares_delta"]
    down = adj(deposited, 3000.0, nav=1.0, shares=shares)
    deposited += down["delta"]; shares += down["shares_delta"]
    assert deposited == pytest.approx(3000.0)
    assert shares == pytest.approx(3000.0)
    assert withdrawn == 0.0


def test_noop_below_the_half_cent_threshold():
    assert adj(3000.0, 3000.0, nav=1.0, shares=3000.0) is None
    assert adj(3000.0, 3000.004, nav=1.0, shares=3000.0) is None
    assert adj(3000.0, 3000.01, nav=1.0, shares=3000.0) is not None


def test_reduction_beyond_the_share_balance_is_refused():
    """Guard preserved from the original handler — you cannot edit away shares you do not hold."""
    with pytest.raises(ValueError, match="needs"):
        adj(1000.0, 0.0, nav=1.0, shares=500.0)


def test_reduction_to_exactly_zero_shares_is_allowed():
    r = adj(1000.0, 0.0, nav=1.0, shares=1000.0)
    assert r["shares_delta"] == pytest.approx(-1000.0)


def test_non_positive_nav_is_refused():
    """A zero/negative NAV would divide by zero or invent shares from nothing."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="NAV"):
            adj(1000.0, 2000.0, nav=bad, shares=1000.0)


def test_shares_priced_at_the_prevailing_nav_not_at_par():
    """A correction made after the fund has grown must use the live NAV for the share math."""
    r = adj(3000.0, 2000.0, nav=1.1129, shares=3000.0)
    assert r["shares_delta"] == pytest.approx(-1000.0 / 1.1129)


# ───────────────────────────────────────────────────────────────────────────────
# HANDLER-LEVEL COVERAGE.
#
# Deep review, correctly: every test above exercises the PURE CORE, which by
# construction has no `inv` and therefore cannot touch `total_withdrawn` — so
# reintroducing `inv.total_withdrawn += (-_delta)` in the endpoint would leave
# them all green. `assert "withdrawn" not in r` pins the core's contract, not the
# bug. These run the REAL endpoint against an in-memory DB (same pattern as
# tests/test_nav_prorata.py) and are the ones that would actually fail.
# ───────────────────────────────────────────────────────────────────────────────
import main                                                    # noqa: E402
import models                                                  # noqa: E402
from sqlalchemy import select                                  # noqa: E402


def _mock_equity(monkeypatch, value):
    async def _pv(_db):
        return value
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)


async def test_handler_edit_does_not_invent_a_withdrawal(db, monkeypatch):
    """THE REGRESSION TEST. One edit 3338.80 -> 3000 must leave total_withdrawn at ZERO."""
    _mock_equity(monkeypatch, 3338.80)
    inv = models.Investor(name="Guille", shares=3338.80,
                          total_deposited=3338.80, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()

    await main.rename_investor(inv.id, main.InvestorRename(deposit_total=3000.0), db)

    assert inv.total_deposited == pytest.approx(3000.0)
    assert inv.total_withdrawn == 0.0            # ← fails if the bug is reintroduced
    assert inv.shares == pytest.approx(3000.0)

    value = 3338.80
    pnl = value - inv.total_deposited + inv.total_withdrawn
    assert pnl == pytest.approx(338.80)          # not 677.60


async def test_handler_logs_adjust_not_withdraw(db, monkeypatch):
    """The ledger row must not read as a cash flow, and must carry the direction."""
    _mock_equity(monkeypatch, 3338.80)
    inv = models.Investor(name="Guille", shares=3338.80,
                          total_deposited=3338.80, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()

    await main.rename_investor(inv.id, main.InvestorRename(deposit_total=3000.0), db)
    await db.flush()

    rows = (await db.execute(
        select(models.InvestorLedger).where(models.InvestorLedger.investor_id == inv.id)
    )).scalars().all()
    assert len(rows) == 1
    assert rows[0].type == "ADJUST"              # never WITHDRAW
    assert rows[0].shares_delta < 0              # direction survives the abs() on amount
    assert "3,338.80" in rows[0].note and "3,000.00" in rows[0].note


async def test_handler_noop_edit_writes_no_ledger_row(db, monkeypatch):
    _mock_equity(monkeypatch, 3000.0)
    inv = models.Investor(name="Guille", shares=3000.0,
                          total_deposited=3000.0, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()

    await main.rename_investor(inv.id, main.InvestorRename(deposit_total=3000.0), db)
    await db.flush()

    rows = (await db.execute(
        select(models.InvestorLedger).where(models.InvestorLedger.investor_id == inv.id)
    )).scalars().all()
    assert rows == []
    assert inv.shares == pytest.approx(3000.0)


async def test_handler_never_leaves_negative_shares(db, monkeypatch):
    """A full reduction must land at exactly 0 — a negative balance feeds the NAV denominator."""
    _mock_equity(monkeypatch, 1000.0)
    inv = models.Investor(name="Guille", shares=1000.0,
                          total_deposited=1000.0, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()

    await main.rename_investor(inv.id, main.InvestorRename(deposit_total=0.0), db)

    assert inv.shares >= 0.0
    assert inv.shares == pytest.approx(0.0, abs=1e-9)
    assert inv.total_withdrawn == 0.0
