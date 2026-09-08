"""NAV/share + pro-rata fund flow invariants, run against the REAL endpoints
with an in-memory DB and a monkeypatched portfolio value (no exchange calls).

Pinned properties:
- NAV reconstruction for already-executed flows is exact and ownership-invariant.
- Pro-rata splits sum to the requested amount, never mint or confiscate shares
  (the negative-remainder pathological case), and ledger every slice.
"""
import pytest
import main
import models


async def _add(db, name, shares):
    inv = models.Investor(name=name, shares=shares, total_deposited=shares,
                          total_withdrawn=0.0)
    db.add(inv)
    await db.flush()
    return inv


def _mock_equity(monkeypatch, value):
    async def _pv(_db):
        return value
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches",
                        lambda: None, raising=False)


async def test_withdraw_all_reconstruction_exact(db, monkeypatch):
    # the live Sep-3 repair case: equity 871.34 after an unregistered -3000
    inv = await _add(db, "Guille", 3547.6)
    _mock_equity(monkeypatch, 871.34)
    body = main.FundWithdraw(amount=3000.0, note=None, nav_override=None,
                             already_executed=True)
    res = await main.fund_withdraw(body, db)
    assert res["ok"] and res["reconstructed"]
    nav = res["nav"]
    assert abs(nav - (871.34 + 3000.0) / 3547.6) < 1e-6      # (equity+amt)/shares
    # post-registration NAV equals the reconstructed NAV (continuity)
    assert abs(871.34 / inv.shares - nav) < 1e-4


async def test_deposit_all_reconstruction_mirror(db, monkeypatch):
    inv = await _add(db, "Guille", 798.1341)
    _mock_equity(monkeypatch, 3871.35)                        # money already on exchange
    body = main.FundDeposit(amount=3000.0, note=None, nav_override=None,
                            already_executed=True)
    res = await main.fund_deposit(body, db)
    nav = res["nav"]
    assert abs(nav - (3871.35 - 3000.0) / 798.1341) < 1e-6   # (equity-amt)/shares
    assert abs(3871.35 / inv.shares - nav) < 1e-4            # continuity


async def test_deposit_all_refuses_impossible_reconstruction(db, monkeypatch):
    await _add(db, "A", 100.0)
    _mock_equity(monkeypatch, 500.0)
    body = main.FundDeposit(amount=600.0, note=None, nav_override=None,
                            already_executed=True)
    with pytest.raises(main.HTTPException) as e:
        await main.fund_deposit(body, db)
    assert e.value.status_code == 400                         # amount >= equity


async def test_prorata_split_sums_and_ownership(db, monkeypatch):
    a = await _add(db, "A", 600.0)
    b = await _add(db, "B", 300.0)
    c = await _add(db, "C", 100.0)
    _mock_equity(monkeypatch, 2000.0)                         # NAV = 2.0
    body = main.FundWithdraw(amount=500.0, note=None, nav_override=None,
                             already_executed=False)
    res = await main.fund_withdraw(body, db)
    amts = [s["amount"] for s in res["split"]]
    assert abs(sum(amts) - 500.0) < 0.01                      # split sums exactly
    # ownership percentages unchanged (pro-rata invariant)
    tot = a.shares + b.shares + c.shares
    assert abs(a.shares / tot - 0.6) < 1e-6
    assert abs(b.shares / tot - 0.3) < 1e-6
    assert all(x.shares >= 0 for x in (a, b, c))


async def test_prorata_negative_remainder_guard(db, monkeypatch):
    # Falsifiable construction (caveman-review R1): three 0.016-share holders round
    # their slices UP (0.016 -> 0.02), over-allocating 0.06 of a 0.049 withdrawal, so
    # the RAW last remainder is -0.01. Without the Sep-7 max(0, .) guard the dust
    # holder's shares INCREASE (minted from the others) and total_withdrawn goes
    # negative — both asserted below, so deleting the guard fails this test.
    a = await _add(db, "A", 0.016)
    b = await _add(db, "B", 0.016)
    c = await _add(db, "C", 0.016)
    d = await _add(db, "D", 0.001)
    _mock_equity(monkeypatch, 0.049)                          # NAV = 1.0
    body = main.FundWithdraw(amount=0.049, note=None, nav_override=None,
                             already_executed=False)
    res = await main.fund_withdraw(body, db)
    assert res["split"][-1]["amount"] == 0.0                  # clamped, not -0.01
    assert d.shares <= 0.001 + 1e-9                           # no share minting
    assert all(x.total_withdrawn >= 0 for x in (a, b, c, d))
    assert all(x.shares >= 0 for x in (a, b, c, d))


async def test_withdraw_all_rejects_over_fund_value(db, monkeypatch):
    await _add(db, "A", 100.0)
    _mock_equity(monkeypatch, 100.0)                          # NAV 1.0, fund worth $100
    body = main.FundWithdraw(amount=3000.0, note=None, nav_override=None,
                             already_executed=False)
    with pytest.raises(main.HTTPException) as e:
        await main.fund_withdraw(body, db)
    assert e.value.status_code == 400                         # the Sep-3 UX case


async def test_nav_override_beats_reconstruction(db, monkeypatch):
    await _add(db, "A", 1000.0)
    _mock_equity(monkeypatch, 500.0)
    body = main.FundWithdraw(amount=300.0, note=None, nav_override=1.0916,
                             already_executed=True)
    res = await main.fund_withdraw(body, db)
    assert abs(res["nav"] - 1.0916) < 1e-9                    # manual override wins
