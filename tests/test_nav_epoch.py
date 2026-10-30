"""📈 (DECISION_LOG 337) NAV history "fund start" epoch — a READ boundary, never a delete.

Pinned properties:
- total shares > 0 -> 0 (investor delete, full single withdrawal, full pro-rata withdraw-all) sets the epoch;
  shares staying > 0, or 0 -> 0, never moves it (idempotent).
- the manual POST /api/nav/epoch sets it any time and touches no snapshot / investor / ledger row.
- /api/investors returns only post-epoch history and a post-epoch HWM; pre-epoch rows stay in the DB.
- the epoch day's row counts only if written at/after the epoch moment (SQL clause == pure rule).
- the first investor after the epoch is minted at NAV 1.0 (never priced off the old HWM).
- /api/reset never sets or clears it (resets stay NAV-neutral).
- the startup backfill fires once for a share-less fund that still shows old history.
"""
import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import func, select

import main
import models


@pytest.fixture(autouse=True)
def _fresh_lock():
    main._investor_lock = asyncio.Lock()   # asyncio.Lock binds to the first loop that uses it (see test_investor_serialization)
    yield


def _mock_equity(monkeypatch, value):
    async def _pv(_db):
        return value

    async def _comps(_db, fresh=True):
        return {"free": value, "margin": 0.0, "unrealized": 0.0, "bnb": 0.0,
                "realized_equity": value, "total_equity": value}
    monkeypatch.setattr(main, "_get_portfolio_value", _pv)
    monkeypatch.setattr(main, "_portfolio_components", _comps)
    monkeypatch.setattr(main.binance_service, "invalidate_flow_caches", lambda: None, raising=False)


async def _state(db):
    st = models.BotState()
    db.add(st)
    await db.flush()
    return st


async def _inv(db, name, shares):
    inv = models.Investor(name=name, shares=shares, total_deposited=shares, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()
    return inv


async def _snap(db, date, nav, updated_at):
    s = models.NavSnapshot(date=date, portfolio_value=nav * 100, total_shares=100.0,
                           nav_per_share=nav, updated_at=updated_at)
    db.add(s)
    await db.flush()
    return s


async def _epoch(db):
    st = (await db.execute(select(models.BotState))).scalar_one()
    await db.refresh(st)
    return st.nav_epoch_start, st.nav_epoch_started_at


# ---------------------------------------------------------------- pure rule + SQL twin

def test_pure_rule():
    t0 = datetime(2026, 10, 21, 15, 0, 0)
    r = main.nav_snapshot_in_epoch
    assert r("2026-01-01", None, None, None)                                   # no epoch -> everything
    assert not r("2026-10-20", t0, "2026-10-21", t0)                          # before the epoch day
    assert r("2026-10-22", t0 - timedelta(days=9), "2026-10-21", t0)          # after the epoch day
    assert not r("2026-10-21", t0 - timedelta(seconds=1), "2026-10-21", t0)   # epoch day, written before
    assert r("2026-10-21", t0, "2026-10-21", t0)                              # epoch day, written at
    assert r("2026-10-21", t0 + timedelta(hours=1), "2026-10-21", t0)
    assert not r("2026-10-21", None, "2026-10-21", t0)


def test_utc3_day_key():
    assert main._nav_day_utc3(datetime(2026, 10, 21, 2, 59)) == "2026-10-20"   # 23:59 UTC-3
    assert main._nav_day_utc3(datetime(2026, 10, 21, 3, 0)) == "2026-10-21"


async def test_sql_clause_matches_pure_rule(db):
    t0 = datetime(2026, 10, 21, 15, 0, 0)
    rows = [("2026-10-19", t0 - timedelta(days=2)), ("2026-10-20", t0 - timedelta(days=1)),
            ("2026-10-21", t0 - timedelta(minutes=5)), ("2026-10-22", t0 + timedelta(days=1)),
            ("2026-10-23", t0 + timedelta(days=2))]
    for d, u in rows:
        await _snap(db, d, 1.1, u)
    for ep_day, ep_at in [("2026-10-21", t0), ("2026-10-21", t0 - timedelta(hours=1)), ("2026-10-22", t0)]:
        got = [s.date for s in (await db.execute(
            select(models.NavSnapshot).where(main._nav_epoch_clause(ep_day, ep_at)))).scalars().all()]
        want = [d for d, u in rows if main.nav_snapshot_in_epoch(d, u, ep_day, ep_at)]
        assert sorted(got) == sorted(want)
    assert main._nav_epoch_clause(None, None) is None


# ---------------------------------------------------------------- automatic trigger

async def test_delete_last_investor_sets_epoch(db, monkeypatch):
    await _state(db)
    inv = await _inv(db, "Guille", 1000.0)
    _mock_equity(monkeypatch, 1250.0)
    assert (await _epoch(db)) == (None, None)
    await main.remove_investor(inv.id, db)
    day, at = await _epoch(db)
    assert day == main._nav_day_utc3(at) and at is not None
    assert (datetime.utcnow() - at).total_seconds() < 60


async def test_full_single_withdrawal_sets_epoch(db, monkeypatch):
    await _state(db)
    inv = await _inv(db, "A", 100.0)
    _mock_equity(monkeypatch, 200.0)                      # NAV 2.0 -> full value $200
    await main.investor_withdraw(main.InvestorWithdraw(investor_id=inv.id, amount=200.0), db)
    assert inv.shares == 0.0
    assert (await _epoch(db))[0] is not None


async def test_full_prorata_withdraw_all_sets_epoch(db, monkeypatch):
    await _state(db)
    await _inv(db, "A", 600.0)
    await _inv(db, "B", 400.0)
    _mock_equity(monkeypatch, 1500.0)
    await main.fund_withdraw(main.FundWithdraw(amount=1500.0), db)
    assert (await main._get_total_shares(db)) <= main._NAV_SHARES_EPS
    assert (await _epoch(db))[0] is not None


async def test_partial_withdrawal_and_other_investor_delete_do_not_set_epoch(db, monkeypatch):
    await _state(db)
    a = await _inv(db, "A", 100.0)
    b = await _inv(db, "B", 100.0)
    b_id = b.id                                           # the decorator expire_all()s — no lazy loads after
    _mock_equity(monkeypatch, 200.0)
    await main.investor_withdraw(main.InvestorWithdraw(investor_id=a.id, amount=50.0), db)
    assert (await _epoch(db)) == (None, None)
    await main.remove_investor(a.id, db)                  # B still holds shares
    assert (await _epoch(db)) == (None, None)
    await main.investor_deposit(main.InvestorDeposit(investor_id=b_id, amount=10.0), db)
    assert (await _epoch(db)) == (None, None)


async def test_zero_to_zero_is_idempotent(db, monkeypatch):
    """Deleting a 0-share investor while the fund is already empty must not move the epoch."""
    st = await _state(db)
    _mock_equity(monkeypatch, 100.0)
    st.nav_epoch_start, st.nav_epoch_started_at = "2026-10-01", datetime(2026, 10, 1, 12, 0)
    await db.flush()
    ghost = await _inv(db, "Ghost", 0.0)
    await main.remove_investor(ghost.id, db)
    assert (await _epoch(db)) == ("2026-10-01", datetime(2026, 10, 1, 12, 0))


async def test_epoch_failure_never_blocks_the_money_op(db, monkeypatch):
    """No BotState row -> the epoch write fails, but the investor delete still commits (fail-open)."""
    inv = await _inv(db, "A", 10.0)
    _mock_equity(monkeypatch, 10.0)
    await main.remove_investor(inv.id, db)
    assert (await db.execute(select(func.count(models.Investor.id)))).scalar() == 0


# ---------------------------------------------------------------- manual endpoint

async def test_manual_endpoint_sets_epoch_and_touches_no_data(db, monkeypatch):
    await _state(db)
    inv = await _inv(db, "A", 100.0)
    _mock_equity(monkeypatch, 130.0)
    await _snap(db, "2026-09-01", 1.2, datetime(2026, 9, 1, 12))
    await _snap(db, "2026-09-02", 1.3, datetime(2026, 9, 2, 12))
    with pytest.raises(main.HTTPException) as e:                     # review I2: shares held -> 409 without force
        await main.start_nav_epoch(db=db)
    assert e.value.status_code == 409
    assert (await _epoch(db)) == (None, None)
    res = await main.start_nav_epoch(body=main.NavEpochRequest(force=True), db=db)
    assert res["ok"] and res["nav_epoch_start"]
    assert (await _epoch(db))[0] == res["nav_epoch_start"]
    assert (await db.execute(select(func.count(models.NavSnapshot.id)))).scalar() == 2
    await db.refresh(inv)
    assert inv.shares == 100.0                                       # shares untouched
    assert (await db.execute(select(func.count(models.InvestorLedger.id)))).scalar() == 0
    # pressing again moves the epoch forward (a fresh start each time), never backwards
    first = (await _epoch(db))[1]
    await main.start_nav_epoch(body=main.NavEpochRequest(force=True), db=db)
    assert (await _epoch(db))[1] >= first


def test_manual_endpoint_is_serialized_and_routed():
    from fastapi.routing import APIRoute
    r = next(r for r in main.app.routes if isinstance(r, APIRoute) and r.path == "/api/nav/epoch")
    assert "POST" in r.methods
    assert getattr(r.endpoint, "__investor_serialized__", False)


# ---------------------------------------------------------------- readers: chart + HWM

async def test_investors_endpoint_filters_history_and_hwm_but_keeps_rows(db, monkeypatch):
    st = await _state(db)
    _mock_equity(monkeypatch, 0.0)
    ep_at = datetime(2026, 10, 21, 15, 0)
    await _snap(db, "2026-10-19", 1.90, ep_at - timedelta(days=2))   # old fund peak
    await _snap(db, "2026-10-20", 1.80, ep_at - timedelta(days=1))
    await _snap(db, "2026-10-21", 1.70, ep_at - timedelta(minutes=30))   # epoch day, pre-epoch write
    st.nav_epoch_start, st.nav_epoch_started_at = "2026-10-21", ep_at
    await db.flush()

    out = await main.list_investors(db)
    assert out["nav_history"] == []
    assert out["nav_high_water"] == 1.0                     # shares 0 -> NAV 1.0; old 1.90 peak NOT used
    assert out["nav_epoch_start"] == "2026-10-21"
    assert out["nav_epoch_started_at"].startswith("2026-10-21T15:00")
    assert (await db.execute(select(func.count(models.NavSnapshot.id)))).scalar() == 3   # never deleted

    # the new fund's first hourly write overwrites the epoch-day row -> it now counts; next day too
    row = (await db.execute(select(models.NavSnapshot).where(models.NavSnapshot.date == "2026-10-21"))).scalar_one()
    row.nav_per_share, row.updated_at = 1.01, ep_at + timedelta(hours=1)
    await _snap(db, "2026-10-22", 1.05, ep_at + timedelta(days=1))
    out = await main.list_investors(db)
    assert [h["date"] for h in out["nav_history"]] == ["2026-10-21", "2026-10-22"]
    assert out["nav_high_water"] == 1.05


async def test_no_epoch_keeps_whole_history(db, monkeypatch):
    await _state(db)
    _mock_equity(monkeypatch, 0.0)
    await _snap(db, "2026-10-19", 1.9, datetime(2026, 10, 19, 12))
    await _snap(db, "2026-10-20", 1.8, datetime(2026, 10, 20, 12))
    out = await main.list_investors(db)
    assert len(out["nav_history"]) == 2 and out["nav_high_water"] == 1.9
    assert out["nav_epoch_start"] is None and out["nav_epoch_started_at"] is None


async def test_first_investor_after_epoch_is_minted_at_nav_one(db, monkeypatch):
    await _state(db)
    old = await _inv(db, "Old", 100.0)
    _mock_equity(monkeypatch, 190.0)                      # old fund NAV 1.90
    await _snap(db, "2026-10-19", 1.9, datetime(2026, 10, 19, 12))
    await main.remove_investor(old.id, db)
    assert (await _epoch(db))[0] is not None
    _mock_equity(monkeypatch, 0.0)
    res = await main.add_investor(main.InvestorCreate(name="New", deposit_amount=500.0), db)
    assert res["new_shares"] == 500.0                     # 500 shares for $500 -> NAV 1.00
    led = (await db.execute(select(models.InvestorLedger).where(
        models.InvestorLedger.investor_id == res["id"]))).scalar_one()
    assert led.nav_at_time == 1.0


# ---------------------------------------------------------------- reset + startup

@pytest.mark.parametrize("preset", [None, ("2026-10-01", datetime(2026, 10, 1, 12, 0))])
async def test_reset_never_touches_the_epoch(db, monkeypatch, preset):
    """Resets are NAV-neutral (Sep-7): a full /api/reset neither sets nor clears the epoch — with shares outstanding
    (no epoch stays no epoch) and with an empty fund (its epoch was already set by the share change or the startup
    backfill, and survives). Runs the REAL handler with the engine side effects stubbed."""
    st = await _state(db)
    if preset:
        st.nav_epoch_start, st.nav_epoch_started_at = preset
    else:
        await _inv(db, "A", 100.0)
    await _snap(db, "2026-09-01", 1.2, datetime(2026, 9, 1, 12))
    await db.commit()

    async def _noop(*a, **k):
        return None
    eng = main.trading_engine
    monkeypatch.setattr(main, "_willy_drain_hold_writes", _noop)
    monkeypatch.setattr(main, "_willy_reset_state", _noop)
    monkeypatch.setattr(main, "_paper_book_invalidate", lambda: None)
    monkeypatch.setattr(main, "set_ban_until", lambda v: None)
    monkeypatch.setattr(eng, "is_paper_mode", True, raising=False)
    monkeypatch.setattr(eng, "is_running", False, raising=False)
    monkeypatch.setattr(eng, "save_state", _noop)
    monkeypatch.setattr(eng, "rtp_clear_all", lambda: None, raising=False)
    import services.trading_engine as _te
    monkeypatch.setattr(_te, "reset_phantom_flip_state", lambda: None)
    for _a in ("paper_balance", "paper_bnb_balance_usd", "total_runtime_seconds", "started_at", "_bnb_emergency_threshold",
               "_bnb_projected_need", "_bnb_burn_rate", "_bnb_burn_rate_bot", "_last_bnb_check", "_bnb_usd_last",
               "_filter_block_counts"):
        monkeypatch.setattr(eng, _a, getattr(eng, _a, None), raising=False)   # the handler mutates these — restore after

    out = await main.reset_trading(direction="ALL", db=db)
    assert out["success"]
    assert (await _epoch(db)) == (preset or (None, None))
    assert (await db.execute(select(func.count(models.NavSnapshot.id)))).scalar() == 1   # history survives the reset


async def test_startup_backfill_once(db):
    await _state(db)
    await _snap(db, "2026-10-19", 1.9, datetime(2026, 10, 19, 12))
    assert await main._nav_epoch_startup_backfill(db) is True
    first = await _epoch(db)
    assert first[0] is not None
    assert await main._nav_epoch_startup_backfill(db) is False      # re-run (next deploy) = no-op
    assert (await _epoch(db)) == first


async def test_startup_backfill_skips_when_shares_or_no_history(db):
    await _state(db)
    assert await main._nav_epoch_startup_backfill(db) is False      # no history -> nothing to hide
    await _snap(db, "2026-10-19", 1.9, datetime(2026, 10, 19, 12))
    await _inv(db, "A", 10.0)
    assert await main._nav_epoch_startup_backfill(db) is False      # shares outstanding -> live fund, leave it
    assert (await _epoch(db)) == (None, None)


async def test_manual_endpoint_without_shares_needs_no_force(db, monkeypatch):
    await _state(db)
    _mock_equity(monkeypatch, 0.0)
    res = await main.start_nav_epoch(db=db)                      # empty fund: simple confirm, no force
    assert res["ok"] and (await _epoch(db))[0] == res["nav_epoch_start"]


def test_manual_endpoint_body_is_optional():
    """The UI posts {force}; an empty POST must still parse (body optional)."""
    from fastapi.routing import APIRoute
    r = next(r for r in main.app.routes if isinstance(r, APIRoute) and r.path == "/api/nav/epoch")
    assert [f.name for f in r.dependant.body_params] == ["body"]
    assert not r.dependant.body_params[0].required


# ---------------------------------------------------------------- review I1: NAV loop race + explicit updated_at

async def test_loop_stamps_explicit_updated_at_even_on_a_no_change_upsert(db, monkeypatch):
    await _state(db)
    await _inv(db, "A", 100.0)
    _mock_equity(monkeypatch, 150.0)
    t1 = datetime(2026, 10, 22, 15, 0, 0, 123456)
    assert await main._nav_snapshot_once(db, now_utc=t1) is True
    row = (await db.execute(select(models.NavSnapshot))).scalar_one()
    assert row.date == "2026-10-22" and row.updated_at == t1 and row.nav_per_share == 1.5
    t2 = t1 + timedelta(hours=1)
    await main._nav_snapshot_once(db, now_utc=t2)                  # identical values -> still re-stamped
    await db.refresh(row)
    assert row.updated_at == t2


async def test_loop_race_write_from_pre_epoch_reads_is_excluded(db, monkeypatch):
    """Shares read > 0, then the last investor is deleted (epoch set) while the loop awaits equity, then the loop
    writes: the row carries the OLD fund's NAV and must be stamped before the epoch -> excluded."""
    await _state(db)
    await _inv(db, "Old", 100.0)
    await db.commit()
    t_loop = datetime.utcnow()

    async def _pv_with_delete_landing(_db):
        for _i in (await _db.execute(select(models.Investor))).scalars().all():
            _i.shares = 0.0                                       # the delete's cash-out
        await main._set_nav_epoch(_db, reason="test: delete landed mid-loop")   # real clock, after the loop's stamp
        return 190.0                                              # old fund NAV 1.90
    monkeypatch.setattr(main, "_get_portfolio_value", _pv_with_delete_landing)
    assert await main._nav_snapshot_once(db) is True             # real clock: proves the stamp is taken BEFORE the reads
    row = (await db.execute(select(models.NavSnapshot))).scalar_one()
    assert row.updated_at >= t_loop
    ep_day, ep_at = await _epoch(db)
    assert row.updated_at < ep_at and row.nav_per_share == 1.9
    assert not main.nav_snapshot_in_epoch(row.date, row.updated_at, ep_day, ep_at)
    _mock_equity(monkeypatch, 0.0)
    out = await main.list_investors(db)
    assert out["nav_history"] == [] and out["nav_high_water"] == 1.0   # the stale 1.90 row neither plots nor sets the HWM


async def test_loop_skips_dust_shares(db, monkeypatch):
    await _state(db)
    await _inv(db, "Dust", 1e-13)
    _mock_equity(monkeypatch, 500.0)
    assert await main._nav_snapshot_once(db) is False             # 500 / 1e-13 would be a 5e15 NAV row
    assert (await db.execute(select(func.count(models.NavSnapshot.id)))).scalar() == 0


# ---------------------------------------------------------------- review I3: dust snapping

async def test_nav_per_share_treats_dust_as_no_shares(db, monkeypatch):
    await _inv(db, "Dust", 1e-13)
    _mock_equity(monkeypatch, 500.0)
    assert await main._get_nav_per_share(db) == 1.0


async def test_single_withdrawal_dust_snaps_to_exact_zero(db, monkeypatch):
    await _state(db)
    inv = await _inv(db, "A", 100.0 + 1e-13)
    inv_id = inv.id
    _mock_equity(monkeypatch, 100.0 + 1e-13)                       # NAV exactly 1.0
    async def _nav1(_db):
        return 1.0
    monkeypatch.setattr(main, "_get_nav_per_share", _nav1)
    res = await main.investor_withdraw(main.InvestorWithdraw(investor_id=inv_id, amount=100.0), db)
    inv = await db.get(models.Investor, inv_id)
    assert inv.shares == 0.0                                       # not 1e-13
    led = (await db.execute(select(models.InvestorLedger))).scalar_one()
    assert led.amount == 100.0
    assert abs(led.shares_delta + 100.0) < 1e-9                    # ledger burns what was held (rounded to 6 dp)
    assert (await _epoch(db))[0] is not None                       # dust no longer blocks the > 0 -> 0 edge


async def test_withdraw_all_dust_snaps_and_split_still_sums(db, monkeypatch):
    await _state(db)
    a = await _inv(db, "A", 50.0 + 1e-13)
    b = await _inv(db, "B", 50.0)
    _mock_equity(monkeypatch, 100.0)
    res = await main.fund_withdraw(main.FundWithdraw(amount=100.0, nav_override=1.0), db)
    assert abs(sum(x["amount"] for x in res["split"]) - 100.0) < 1e-9
    shares = (await db.execute(select(models.Investor.shares))).scalars().all()
    assert shares == [0.0, 0.0]
    assert (await _epoch(db))[0] is not None


async def test_partial_withdrawal_is_not_snapped(db, monkeypatch):
    inv = await _inv(db, "A", 100.0)
    inv_id = inv.id
    _mock_equity(monkeypatch, 100.0)
    await main.investor_withdraw(main.InvestorWithdraw(investor_id=inv_id, amount=99.99), db)
    inv = await db.get(models.Investor, inv_id)
    assert abs(inv.shares - 0.01) < 1e-9


# ---------------------------------------------------------------- review I4: first-deposit NAV start warning

def test_nav_start_warning_pure():
    w = main.nav_start_warning
    assert w(0.0, 1000.0) is None                                  # empty account, money still to arrive
    assert w(0.5, 1000.0) is None                                  # ≤ max($1, 0.5 %)
    assert w(1003.0, 1000.0) is None                               # the deposit IS the account (landed), within 0.5 %
    msg = w(5000.0, 1000.0)                                        # $4,000 of other money on the account
    assert msg and "$5,000.00" in msg and "6.0000" in msg and "5.0000" in msg and "$6,000.00" in msg
    msg2 = w(500.0, 1000.0)                                        # E < D: only the not-yet-landed reading is possible
    assert msg2 and "1.5000" in msg2 and "E/D" not in msg2
    assert w(100.0, 0.0) is None


async def test_add_investor_preview_warns_and_writes_nothing(db, monkeypatch):
    await _state(db)
    _mock_equity(monkeypatch, 5000.0)
    res = await main.add_investor(main.InvestorCreate(name="New", deposit_amount=1000.0, preview=True), db)
    assert res["preview"] and res["nav"] == 1.0 and res["new_shares"] == 1000.0
    assert res["nav_start_warning"] and "5,000.00" in res["nav_start_warning"]
    assert (await db.execute(select(func.count(models.Investor.id)))).scalar() == 0
    assert (await db.execute(select(func.count(models.InvestorLedger.id)))).scalar() == 0
    # minting math unchanged: the real call still mints at NAV 1.00
    real = await main.add_investor(main.InvestorCreate(name="New", deposit_amount=1000.0), db)
    assert real["new_shares"] == 1000.0


async def test_add_investor_preview_no_warning_when_deposit_is_the_account(db, monkeypatch):
    await _state(db)
    _mock_equity(monkeypatch, 1000.0)
    res = await main.add_investor(main.InvestorCreate(name="New", deposit_amount=1000.0, preview=True), db)
    assert res["nav_start_warning"] is None


async def test_deposit_preview_warns_only_for_a_shareless_fund(db, monkeypatch):
    await _state(db)
    z = await _inv(db, "Zero", 0.0)
    z_id = z.id
    _mock_equity(monkeypatch, 5000.0)
    res = await main.investor_deposit(main.InvestorDeposit(investor_id=z_id, amount=1000.0, preview=True), db)
    assert res["preview"] and res["nav_start_warning"]
    z = await db.get(models.Investor, z_id)
    assert z.shares == 0.0                                         # preview wrote nothing
    monkeypatch.setattr(main.trading_engine, "is_paper_mode", False)   # DL 422: paper previews carry their own top-up warning
    await _inv(db, "Holder", 100.0)                                # fund has shares -> live NAV pricing, no warning
    res = await main.investor_deposit(main.InvestorDeposit(investor_id=z_id, amount=1000.0, preview=True), db)
    assert res["nav_start_warning"] is None and res["nav"] == 50.0


# ---------------------------------------------------------------- M2: a failing mutator never sets the epoch

async def test_raising_share_mutator_does_not_set_epoch(db, monkeypatch):
    await _state(db)
    inv = await _inv(db, "A", 100.0)
    inv_id = inv.id
    await db.commit()

    @main._investor_serialized
    async def _zero_then_fail(db):
        (await db.get(models.Investor, inv_id)).shares = 0.0
        await db.flush()
        raise main.HTTPException(503, "exchange down")

    with pytest.raises(main.HTTPException):
        await _zero_then_fail(db=db)
    await db.rollback()                                            # what get_db does when the handler raises
    assert (await _epoch(db)) == (None, None)
    assert (await db.get(models.Investor, inv_id)).shares == 100.0
