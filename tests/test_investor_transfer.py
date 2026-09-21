"""💰 Sep-21 — transferring VALUE between investors. No money enters or leaves the fund.

OPERATOR: "in reality I will receive $1,000, then I decide if I add $1,000, or if I transfer part
of my $5,000." Both branches must leave the RECIPIENT identical, or the choice between them is
arbitrary — so the recipient's basis is the dollars they paid and their P&L starts at 0, while the
sender's basis drops by the same amount and keeps the gain they earned.

Why this exists at all: the edit path cannot express it. Reducing one investor and adding another
are two operations at two different NAVs, so the shares never reconcile and value silently moves
(the windfall documented on investor_edit_adjustment). A transfer moves shares directly.

INVARIANTS PINNED HERE — these are the ones that make the fund's books add up:
  · total SHARES conserved (nothing minted or burned)
  · total DEPOSITED conserved (so per-investor P&L still sums to fund P&L)
  · NAV and portfolio value untouched
"""
import pytest

import main
import models
from main import investor_transfer_split as split
from sqlalchemy import select


NAV = 1.1129


# ───────────────────────────────────────────────────────────── pure core

def test_dollars_convert_to_shares_at_nav():
    r = split(1000.0, NAV, from_shares=3000.0, from_deposited=3000.0)
    assert r["shares"] == pytest.approx(1000.0 / NAV)
    assert r["basis"] == pytest.approx(1000.0)      # basis moves 1:1 with the dollars


def test_recipient_matches_a_fresh_deposit_of_the_same_size():
    """The operator's framing: 'add $1,000' and 'transfer $1,000' must give the same result."""
    transferred = split(1000.0, NAV, 3000.0, 3000.0)
    deposited_shares = 1000.0 / NAV                  # what /api/investors/deposit would issue
    assert transferred["shares"] == pytest.approx(deposited_shares)
    assert transferred["basis"] == pytest.approx(1000.0)


def test_refuses_more_shares_than_the_sender_holds():
    with pytest.raises(ValueError, match="sender holds"):
        split(5000.0, NAV, from_shares=1000.0, from_deposited=9999.0)


def test_basis_clamp_applies_only_to_a_full_exit():
    """A profitable stake is worth more than its basis, so 'transfer everything' must still work.

    The clamp is symmetric — the recipient gets exactly what the sender gives — so total basis is
    unchanged. Refusing here (my first version) made a full handover impossible at any NAV > 1.
    """
    # partial (sender holds far more than they are moving) -> refused, not clamped
    with pytest.raises(ValueError, match="Partial transfer"):
        split(1000.0, NAV, from_shares=9999.0, from_deposited=500.0)
    # full exit -> clamped, and conserved
    r = split(1000.0 / NAV * NAV, NAV, from_shares=1000.0 / NAV, from_deposited=500.0)
    assert r["basis"] == pytest.approx(500.0)          # not 1000 — the sender only has 500
    assert r["shares"] == pytest.approx(1000.0 / NAV)


def test_rejects_nan_and_infinity():
    """🛑 THE exploit. `json.loads` accepts a bare NaN literal and every comparison below is False,
    so nan sails past `<= 0` AND the share guard; min(nan, x) is nan and max(0.0, nan) is 0.0, so
    one request zeroed the sender and NaN'd the recipient — poisoning NAV for the whole fund."""
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            split(bad, NAV, 3000.0, 3000.0)
    with pytest.raises(ValueError, match="finite"):
        split(1000.0, float("nan"), 3000.0, 3000.0)


def test_partial_transfer_above_the_basis_is_refused_not_clamped():
    """Clamping a PARTIAL transfer would hand the recipient a positive opening P&L and credit the
    sender with a gain from nowhere — breaking the contract. Only a FULL exit may clamp."""
    with pytest.raises(ValueError, match="Partial transfer"):
        split(3100.0, NAV, from_shares=3000.0, from_deposited=3000.0)   # $3,338.80 of value


def test_full_exit_above_the_basis_still_clamps():
    r = split(3000.0 * NAV, NAV, from_shares=3000.0, from_deposited=3000.0)
    assert r["shares"] == pytest.approx(3000.0)
    assert r["basis"] == pytest.approx(3000.0)


def test_rejects_non_positive_amount_and_bad_nav():
    with pytest.raises(ValueError, match="positive"):
        split(0.0, NAV, 1000.0, 1000.0)
    with pytest.raises(ValueError, match="NAV"):
        split(100.0, 0.0, 1000.0, 1000.0)


# ─────────────────────────────────────────────────────── endpoint end-to-end

async def _pair(db, monkeypatch, portfolio=3338.80):
    async def _pv(_db):
        return portfolio
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)
    a = models.Investor(name="Guille", shares=3000.0, total_deposited=3000.0, total_withdrawn=0.0)
    b = models.Investor(name="Roby", shares=0.0, total_deposited=0.0, total_withdrawn=0.0)
    db.add_all([a, b])
    await db.flush()
    return a, b


async def test_transfer_conserves_shares_deposited_and_nav(db, monkeypatch):
    """THE invariant set. Nothing is minted, nothing is burned, the fund is unchanged."""
    a, b = await _pair(db, monkeypatch)
    shares0 = a.shares + b.shares
    dep0 = a.total_deposited + b.total_deposited
    nav0 = 3338.80 / shares0

    await main.investor_transfer(main.InvestorTransfer(
        from_investor_id=a.id, to_investor_id=b.id, amount=1000.0), db)
    await db.flush()

    assert a.shares + b.shares == pytest.approx(shares0)
    assert a.total_deposited + b.total_deposited == pytest.approx(dep0)
    assert 3338.80 / (a.shares + b.shares) == pytest.approx(nav0)
    # pin the SYMMETRY directly — a one-sided credit shifts this even when the sums round clean
    assert (3000.0 - a.shares) == pytest.approx(b.shares - 0.0, abs=1e-12)
    assert (3000.0 - a.total_deposited) == pytest.approx(b.total_deposited, abs=1e-12)


async def test_recipient_starts_flat_sender_keeps_the_gain(db, monkeypatch):
    """Roby paid today's price: P&L 0. Guille earned the $338.80 and keeps all of it."""
    a, b = await _pair(db, monkeypatch)
    await main.investor_transfer(main.InvestorTransfer(
        from_investor_id=a.id, to_investor_id=b.id, amount=1000.0), db)
    await db.flush()

    nav = 3338.80 / (a.shares + b.shares)
    b_pnl = b.shares * nav - b.total_deposited
    a_pnl = a.shares * nav - a.total_deposited
    assert b.total_deposited == pytest.approx(1000.0)
    assert b.shares * nav == pytest.approx(1000.0)
    assert b_pnl == pytest.approx(0.0, abs=0.01)
    assert a_pnl == pytest.approx(338.80, abs=0.01)
    assert a_pnl + b_pnl == pytest.approx(338.80, abs=0.01)     # sums to FUND P&L


async def test_ledger_writes_a_paired_row_on_both_sides(db, monkeypatch):
    a, b = await _pair(db, monkeypatch)
    await main.investor_transfer(main.InvestorTransfer(
        from_investor_id=a.id, to_investor_id=b.id, amount=1000.0), db)
    await db.flush()

    rows = (await db.execute(select(models.InvestorLedger))).scalars().all()
    assert {r.type for r in rows} == {"TRANSFER_OUT", "TRANSFER_IN"}
    out = next(r for r in rows if r.type == "TRANSFER_OUT")
    inn = next(r for r in rows if r.type == "TRANSFER_IN")
    assert out.investor_id == a.id and inn.investor_id == b.id
    assert out.shares_delta < 0 < inn.shares_delta
    assert out.shares_delta == pytest.approx(-inn.shares_delta)
    assert out.note == inn.note                                  # one event, two rows


async def test_cannot_transfer_to_self(db, monkeypatch):
    a, _ = await _pair(db, monkeypatch)
    with pytest.raises(main.HTTPException) as e:
        await main.investor_transfer(main.InvestorTransfer(
            from_investor_id=a.id, to_investor_id=a.id, amount=100.0), db)
    assert e.value.status_code == 400


async def test_unknown_investor_is_404(db, monkeypatch):
    a, _ = await _pair(db, monkeypatch)
    with pytest.raises(main.HTTPException) as e:
        await main.investor_transfer(main.InvestorTransfer(
            from_investor_id=a.id, to_investor_id=99999, amount=100.0), db)
    assert e.value.status_code == 404


async def test_oversized_transfer_moves_nothing(db, monkeypatch):
    """A refused transfer must leave BOTH sides untouched — no partial application."""
    a, b = await _pair(db, monkeypatch)
    before = (a.shares, a.total_deposited, b.shares, b.total_deposited)
    with pytest.raises(main.HTTPException) as e:
        await main.investor_transfer(main.InvestorTransfer(
            from_investor_id=a.id, to_investor_id=b.id, amount=99_999.0), db)
    assert e.value.status_code == 400
    assert (a.shares, a.total_deposited, b.shares, b.total_deposited) == before


async def test_full_transfer_empties_the_sender_without_negative_shares(db, monkeypatch):
    a, b = await _pair(db, monkeypatch)
    dep0 = a.total_deposited + b.total_deposited
    everything = a.shares * (3338.80 / (a.shares + b.shares))      # $3,338.80 — more than the basis
    await main.investor_transfer(main.InvestorTransfer(
        from_investor_id=a.id, to_investor_id=b.id, amount=everything), db)
    await db.flush()
    assert a.shares >= 0.0
    assert a.shares == pytest.approx(0.0, abs=1e-9)
    assert b.shares == pytest.approx(3000.0)
    assert a.total_deposited == pytest.approx(0.0)                 # sender is fully out
    assert a.total_deposited + b.total_deposited == pytest.approx(dep0)   # still conserved
