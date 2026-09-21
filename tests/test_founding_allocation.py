"""💰 Sep-21 — a FOUNDING allocation divides capital the fund already holds.

OPERATOR-FOUND. Three people each took $1,000 of an existing $3,000 fund. Routed through the
normal deposit path they were priced at whatever NAV existed at that second:

    Guille  $1,000 @ NAV 1.0000 -> 1000.0000 shares -> 59.22%
    Roby    $1,000 @ NAV 3.3388 ->  299.5091 shares -> 17.74%
    Fede    $1,000 @ NAV 2.5693 ->  389.2147 shares -> 23.05%

Cause: dividing a fund does not grow it, so each founder re-priced it downward for the next. The
fix is not to credit the balance (that would double the capital — the seed IS their money); it is
to price a founding share at NAV 1.0 so order and interim P&L cannot matter.

THE GUARD pinned here: founding room comes from the CAPITAL SEED, never the portfolio value — P&L
must not create room to mint shares at par, or a profitable fund lets a latecomer buy in cheap.
"""
import pytest

import main
import models
from main import founding_allocation_shares as alloc
from sqlalchemy import select


SEED = 3000.0          # paper_balance 2900 + paper_bnb_initial_usd 100


# ─────────────────────────────────────────────────────────── the pure core

def test_founding_share_is_issued_at_par():
    assert alloc(1000.0, already_allocated=0.0, initial_capital=SEED) == 1000.0


def test_three_equal_founders_get_equal_shares_regardless_of_order():
    """The whole point: no drift between the first and the third."""
    allocated, shares = 0.0, {}
    for who in ("Guille", "Roby", "Fede"):
        shares[who] = alloc(1000.0, allocated, SEED)
        allocated += 1000.0
    assert shares["Guille"] == shares["Roby"] == shares["Fede"] == 1000.0
    total = sum(shares.values())
    for who in shares:
        assert shares[who] / total == pytest.approx(1 / 3)


def test_pnl_is_shared_pro_rata_after_the_round():
    """Operator: 'the P&L is shared'. $3,338.80 over 3,000 shares -> +11.29% each."""
    total_shares = sum(alloc(1000.0, i * 1000.0, SEED) for i in range(3))
    nav = 3338.80 / total_shares
    assert nav == pytest.approx(1.1129, abs=1e-4)
    for _ in range(3):
        value = 1000.0 * nav
        assert value == pytest.approx(1112.93, abs=0.01)
        assert (value - 1000.0) / 1000.0 == pytest.approx(0.1129, abs=1e-4)


def test_cannot_allocate_more_capital_than_the_fund_was_seeded_with():
    with pytest.raises(ValueError, match="exceeds the unallocated capital"):
        alloc(1000.0, already_allocated=3000.0, initial_capital=SEED)


def test_profit_does_not_create_founding_room():
    """THE guard. The fund is fully allocated and up $500 — a latecomer must NOT buy in at par."""
    with pytest.raises(ValueError, match="exceeds the unallocated capital"):
        alloc(400.0, already_allocated=3000.0, initial_capital=SEED)   # portfolio 3500, seed 3000


def test_the_last_founder_may_take_exactly_the_remainder():
    assert alloc(1000.0, already_allocated=2000.0, initial_capital=SEED) == 1000.0
    with pytest.raises(ValueError):
        alloc(1000.01, already_allocated=2000.0, initial_capital=SEED)


def test_non_positive_amount_refused():
    for bad in (0.0, -100.0):
        with pytest.raises(ValueError, match="positive"):
            alloc(bad, 0.0, SEED)


# ─────────────────────────────────────────────────── the endpoint end-to-end
#
# These seed a REAL BotState row and do NOT monkeypatch _paper_capital_seed, so the actual
# baseline read and the monotonic counter are exercised (deep review: both endpoint tests
# previously stubbed the seed, so a mistyped attribute would have degraded silently to 0).


async def _state(db, seed=SEED):
    st = models.BotState(runtime_initial_total_usd=seed, founding_allocated_usd=0.0)
    db.add(st)
    await db.flush()
    return st


def _paper(monkeypatch, portfolio):
    async def _pv(_db):
        return portfolio
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", True, raising=False)


async def test_seed_reads_the_immutable_baseline_not_hot_config(db):
    """The baseline must come from BotState, which the operator cannot hot-edit."""
    await _state(db, seed=4242.0)
    assert await main._paper_capital_seed(db) == pytest.approx(4242.0)


async def test_seed_falls_back_to_config_when_the_baseline_is_null(db):
    db.add(models.BotState(runtime_initial_total_usd=None, founding_allocated_usd=0.0))
    await db.flush()
    import config as _cfg
    expected = float(_cfg.trading_config.paper_balance) + float(_cfg.trading_config.paper_bnb_initial_usd)
    assert await main._paper_capital_seed(db) == pytest.approx(expected)


async def test_endpoint_three_founders_end_equal(db, monkeypatch):
    """Runs the REAL endpoint three times — the pure core alone cannot catch a wiring regression."""
    _paper(monkeypatch, 3338.80)
    st = await _state(db)

    for who in ("Guille", "Roby", "Fede"):
        await main.add_investor(main.InvestorCreate(name=who, deposit_amount=1000.0, founding=True), db)
    await db.flush()

    invs = (await db.execute(select(models.Investor))).scalars().all()
    assert len(invs) == 3
    assert all(i.shares == pytest.approx(1000.0) for i in invs)
    assert all(i.total_deposited == pytest.approx(1000.0) for i in invs)
    assert all(i.total_withdrawn == 0.0 for i in invs)
    assert st.founding_allocated_usd == pytest.approx(3000.0)

    total_shares = sum(i.shares for i in invs)
    assert total_shares == pytest.approx(3000.0)
    nav = 3338.80 / total_shares
    for i in invs:
        assert i.shares / total_shares == pytest.approx(1 / 3)      # 33.33% each, not 59/18/23
        assert i.shares * nav - i.total_deposited == pytest.approx(112.93, abs=0.01)


async def test_endpoint_rejects_founding_beyond_the_seed(db, monkeypatch):
    _paper(monkeypatch, 3338.80)
    await _state(db)
    for who in ("A", "B", "C"):
        await main.add_investor(main.InvestorCreate(name=who, deposit_amount=1000.0, founding=True), db)
    await db.flush()

    with pytest.raises(main.HTTPException) as e:
        await main.add_investor(main.InvestorCreate(name="D", deposit_amount=1000.0, founding=True), db)
    assert e.value.status_code == 400
    assert "unallocated capital" in str(e.value.detail)


async def test_deleting_an_investor_does_NOT_reopen_founding_room(db, monkeypatch):
    """THE EXPLOIT the deep review found. Room is monotonic — a cash-out cannot re-mint it.

    Before the fix: room was SUM(Investor.total_deposited), so deleting Fede (whose capital LEFT
    the fund) dropped the sum by $1,000 and a 4th founder could mint 1,000 shares for zero cash,
    crashing NAV 1.1129 -> 0.7420 and taking $371 from each remaining holder.
    """
    _paper(monkeypatch, 3338.80)
    st = await _state(db)
    ids = []
    for who in ("Guille", "Roby", "Fede"):
        r = await main.add_investor(main.InvestorCreate(name=who, deposit_amount=1000.0, founding=True), db)
        ids.append(r["id"] if isinstance(r, dict) and "id" in r else None)
    await db.flush()
    assert st.founding_allocated_usd == pytest.approx(3000.0)

    fede = (await db.execute(select(models.Investor).where(models.Investor.name == "Fede"))).scalar_one()
    await db.delete(fede)
    await db.flush()

    # the counter is untouched by the delete — that is the whole point
    assert st.founding_allocated_usd == pytest.approx(3000.0)
    with pytest.raises(main.HTTPException) as e:
        await main.add_investor(main.InvestorCreate(name="Exploit", deposit_amount=1000.0, founding=True), db)
    assert e.value.status_code == 400
    assert "unallocated capital" in str(e.value.detail)


async def test_editing_a_deposit_down_does_NOT_reopen_founding_room(db, monkeypatch):
    """Same hole via the edit path (deep review CRITICAL 2) — also closed by the counter."""
    _paper(monkeypatch, 3338.80)
    st = await _state(db)
    for who in ("A", "B", "C"):
        await main.add_investor(main.InvestorCreate(name=who, deposit_amount=1000.0, founding=True), db)
    await db.flush()

    a = (await db.execute(select(models.Investor).where(models.Investor.name == "A"))).scalar_one()
    await main.rename_investor(a.id, main.InvestorRename(deposit_total=1.0), db)
    await db.flush()

    # re-read: @_investor_serialized expire_all()s the session, so `st` is stale by design
    st = (await db.execute(select(models.BotState))).scalar_one()
    assert st.founding_allocated_usd == pytest.approx(3000.0)      # unchanged by the edit
    with pytest.raises(main.HTTPException):
        await main.add_investor(main.InvestorCreate(name="Exploit", deposit_amount=500.0, founding=True), db)


async def test_founding_without_an_amount_is_refused(db, monkeypatch):
    _paper(monkeypatch, 3000.0)
    await _state(db)
    with pytest.raises(main.HTTPException) as e:
        await main.add_investor(main.InvestorCreate(name="NoAmount", founding=True), db)
    assert e.value.status_code == 400
    assert "requires an amount" in str(e.value.detail)


async def test_endpoint_founding_is_paper_mode_only(db, monkeypatch):
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", False, raising=False)
    with pytest.raises(main.HTTPException) as e:
        await main.add_investor(main.InvestorCreate(name="Live", deposit_amount=1000.0, founding=True), db)
    assert e.value.status_code == 400
    assert "paper-mode only" in str(e.value.detail)


async def test_endpoint_without_the_flag_still_prices_at_live_nav(db, monkeypatch):
    """The normal deposit path must be untouched — new money still buys at the live NAV."""
    _paper(monkeypatch, 3000.0)
    await _state(db)
    db.add(models.Investor(name="Seed", shares=1000.0, total_deposited=1000.0, total_withdrawn=0.0))
    await db.flush()

    await main.add_investor(main.InvestorCreate(name="New", deposit_amount=1000.0), db)
    await db.flush()
    new = (await db.execute(select(models.Investor).where(models.Investor.name == "New"))).scalar_one()
    assert new.shares == pytest.approx(1000.0 / 3.0)      # NAV was 3000/1000 = 3.0, not 1.0
