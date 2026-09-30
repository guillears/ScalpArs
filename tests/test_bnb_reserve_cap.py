"""⛽ Sep-29 BNB fee-reserve fix (DECISION_LOG 135) — the reserve targets are capped by equity, one swap never exceeds the cap,
the immature-window branch still honours the real floor, the paper reserve never goes negative (NAV conserved), and a top-up
after the reserve ran dry really fills it (fees already paid in USDT are settled in their own row)."""
import asyncio, datetime as dt, json, os, sys
from types import SimpleNamespace as NS
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
import services.trading_engine as T

TC = NS(bnb_max_reserve_pct_of_equity=10.0, bnb_min_balance_usd=50.0, bnb_runway_hours=24)


@pytest.fixture
def cfg(monkeypatch):
    """The account of the Sep-29 incident, pinned — the tests never read the operator's live settings."""
    tc = T.config.trading_config
    for k, v in dict(paper_balance=2200.0, paper_bnb_initial_usd=100.0, bnb_min_balance_usd=50.0, bnb_swap_enabled=True,
                     bnb_max_reserve_pct_of_equity=10.0, bnb_runway_hours=24).items():
        monkeypatch.setattr(tc, k, v)
    monkeypatch.setattr(tc.investment, "min_investment_size", 100.0)

    async def _px(): return 600.0
    monkeypatch.setattr(T.binance_service, "get_bnb_price", _px)
    return tc


def test_cap_and_targets_on_the_sep29_case():
    """$126/hr on a $2,300 account extrapolated to $3,019 / $1,509; capped: target ≤ 10 % of equity, threshold ≤ 5 %."""
    need, emerg = T.bnb_reserve_targets(TC, 125.81, 125.81, 2300.0)
    assert need == 230.0 and emerg == 115.0
    assert T.bnb_reserve_cap_usd(TC, 2300.0) == 230.0
    # quiet book: the dollar floor still rules (unchanged behaviour)
    assert T.bnb_reserve_targets(TC, 0.5, 0.5, 2300.0) == (50.0, 25.0)
    assert T.bnb_reserve_targets(TC, 4.0, 4.0, 2300.0) == (96.0, 48.0)          # normal burn: untouched by the cap
    # the cap never goes below the floor, even on a tiny account
    assert T.bnb_reserve_cap_usd(TC, 200.0) == 50.0 and T.bnb_reserve_targets(TC, 100.0, 100.0, 200.0) == (50.0, 25.0)
    # pct 0 / unknown or junk equity = no cap (pre-existing behaviour), never a NaN cap
    assert T.bnb_reserve_targets(NS(bnb_max_reserve_pct_of_equity=0.0, bnb_min_balance_usd=50.0, bnb_runway_hours=24), 125.81, 125.81, 2300.0)[0] > 3000
    inf = float("inf")
    assert T.bnb_reserve_cap_usd(TC, None) == inf and T.bnb_reserve_cap_usd(NS(), 2300.0) == inf
    assert T.bnb_reserve_cap_usd(TC, float("nan")) == inf and T.bnb_reserve_cap_usd(TC, -5.0) == inf and T.bnb_reserve_cap_usd(TC, "x") == inf
    assert T.bnb_reserve_cap_usd(NS(bnb_max_reserve_pct_of_equity=float("nan"), bnb_min_balance_usd=50.0), 2300.0) == inf
    # the emergency threshold never sits above the top-up target (the swap it calls would buy nothing)
    need, emerg = T.bnb_reserve_targets(TC, 1.0, 20.0, 2300.0)
    assert (need, emerg) == (50.0, 50.0)
    # runway 0 is a legal setting: floor only
    assert T.bnb_reserve_targets(NS(bnb_max_reserve_pct_of_equity=10.0, bnb_min_balance_usd=50.0, bnb_runway_hours=0), 9.0, 1.0, 2300.0)[0] == 50.0


def test_paper_reserve_never_negative_and_nav_is_conserved():
    for initial, swaps, fees in [(100, 0, 40), (100, 0, 100), (100, 0, 172.18), (100, 230, 172.18), (0, 0, 9)]:
        bnb, usdt_charge = T.paper_bnb_split(initial, swaps, fees)
        assert bnb >= 0 and usdt_charge >= 0 and min(bnb, usdt_charge) == 0
        assert abs((bnb - usdt_charge) - (initial + swaps - fees)) < 1e-9       # reserve + what USDT paid == the raw ledger
    assert abs(T.paper_bnb_split(100, 0, 172.18)[1] - 72.18) < 1e-9
    assert str(T.paper_bnb_split(100, 0, 100)[1]) == "0.0"                      # not -0.0


def _engine(bnb=0.0, mature=True, threshold=25.0, need=50.0, stub_swap=True):
    e = T.TradingEngine.__new__(T.TradingEngine)
    e.is_paper_mode = True; e.paper_bnb_balance_usd = bnb; e.paper_balance = 0.0; e._bnb_data_mature = mature
    e._bnb_emergency_threshold = threshold; e._bnb_projected_need = need; e._bnb_burn_rate = 0.0; e._equity_usd_last = None
    e.is_running = False; e.started_at = None; e.total_runtime_seconds = 0
    e.swaps = []

    async def _save(db): return None
    e.save_state = _save
    if stub_swap:
        async def _swap(db, swap_type="scheduled"): e.swaps.append(swap_type)
        e._execute_bnb_swap = _swap
    return e


def test_immature_window_still_honours_the_real_floor(cfg):
    """The branch used to return unconditionally: a young batch could run the reserve to zero with no swap."""
    floor = 25.0                                                                                # max(10 % of 100, 50 × 0.5)
    e = _engine(bnb=floor + 40.0, mature=False, threshold=115.0, need=230.0)
    asyncio.run(e._deduct_fee_from_bnb(9.0, None)); assert e.swaps == []                       # above the real floor: suppressed, as before
    e = _engine(bnb=floor + 5.0, mature=False, threshold=115.0, need=230.0)
    asyncio.run(e._deduct_fee_from_bnb(9.0, None)); assert e.swaps == ["emergency"]             # fee takes it below the floor → swap
    e = _engine(bnb=3.0, mature=False, threshold=115.0, need=230.0)
    asyncio.run(e._deduct_fee_from_bnb(54.0, None)); assert e.swaps == ["emergency"] and e.paper_bnb_balance_usd == 0
    e = _engine(bnb=100.0, mature=True, threshold=115.0, need=230.0)
    asyncio.run(e._deduct_fee_from_bnb(9.0, None)); assert e.swaps == ["emergency"]             # mature data: threshold rules, as before
    # a real floor above the top-up target is clamped to it — no swap call that would buy nothing
    cfg.paper_bnb_initial_usd = 1000.0                                                          # floor 100 > target 50
    e = _engine(bnb=70.0, mature=False, threshold=115.0, need=50.0)
    asyncio.run(e._deduct_fee_from_bnb(9.0, None)); assert e.swaps == []
    cfg.bnb_swap_enabled = False
    e = _engine(bnb=1.0, mature=True, threshold=115.0, need=230.0)
    asyncio.run(e._deduct_fee_from_bnb(9.0, None)); assert e.swaps == []                       # auto-swap OFF: nothing fires


def _order(i, strat, fee, status="CLOSED", hours_ago=1.0, pnl=0.0):
    import models
    t1 = dt.datetime.utcnow() - dt.timedelta(hours=hours_ago)
    return models.Order(pair=f"P{i}USDT", direction="LONG", status=status, entry_price=100.0, exit_price=100.0, investment=400.0, leverage=20.0,
                        notional_value=8000.0, quantity=80.0, confidence="STRONG_BUY", entry_strategy=strat, is_paper=True, pnl=pnl,
                        pnl_percentage=0.0, entry_fee=fee / 2.0, total_fee=(fee if status == "CLOSED" else fee / 2.0),
                        opened_at=t1 - dt.timedelta(minutes=20), closed_at=(t1 if status == "CLOSED" else None))


def _run_db(scenario):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        async with async_sessionmaker(eng, expire_on_commit=False)() as db:
            out = await scenario(db)
        await eng.dispose()
        return out
    return asyncio.run(run())


async def _book(e, db):
    from sqlalchemy import select
    import models
    usdt = await e._recalculate_paper_balance(db); bnb = await e._recalculate_paper_bnb(db)
    rows = (await db.execute(select(models.BnbSwapLog).order_by(models.BnbSwapLog.id))).scalars().all()
    margin = sum(o.investment for o in (await db.execute(select(models.Order).where(models.Order.status == "OPEN"))).scalars().all())
    return usdt, bnb, rows, usdt + bnb + margin


def test_swap_after_the_reserve_ran_dry_fills_it_and_conserves_nav(cfg):
    """Sep-29 ledger: $100 reserve, $172.18 of fees → $72.18 was paid from USDT. The emergency swap must leave the reserve AT the
    target, take exactly the purchase from USDT, keep NAV, and not fire again on the next check."""
    async def scenario(db):
        db.add(_order(1, "MANUAL", 120.0, pnl=-120.0)); db.add(_order(2, "MOMENTUM", 52.18, pnl=-52.18)); await db.commit()
        e = _engine(need=50.0, threshold=25.0, stub_swap=False)
        usdt0, bnb0, rows0, nav0 = await _book(e, db)
        await e._execute_bnb_swap(db, swap_type="emergency")
        usdt1, bnb1, rows1, nav1 = await _book(e, db)
        await e._execute_bnb_swap(db, swap_type="emergency")                                    # second call: nothing left to do
        usdt2, bnb2, rows2, nav2 = await _book(e, db)
        return (usdt0, bnb0, rows0, nav0), (usdt1, bnb1, rows1, nav1), (usdt2, bnb2, rows2, nav2), e
    (u0, b0, r0, n0), (u1, b1, r1, n1), (u2, b2, r2, n2), e = _run_db(scenario)
    assert b0 == 0.0 and r0 == [] and abs(u0 - (2200.0 - 72.18)) < 1e-6                        # fees beyond the reserve came out of USDT
    assert abs(b1 - 50.0) < 1e-6 and abs(u1 - (u0 - 50.0)) < 1e-6 and abs(n1 - n0) < 1e-6      # reserve at target, USDT −purchase, NAV kept
    assert [r.swap_type for r in r1] == [T.PAPER_FEE_SETTLE, "emergency"]
    settle, buy = r1
    assert abs(settle.amount_usdt - 72.18) < 1e-6 and settle.pre_usdt == settle.post_usdt and settle.post_bnb_usd == 0.0
    assert abs(buy.amount_usdt - 50.0) < 1e-6 and abs(buy.post_bnb_usd - b1) < 1e-6 and abs(buy.post_usdt - u1) < 1e-6   # the log row tells the truth
    assert abs(e.paper_bnb_balance_usd - b1) < 1e-6                                            # in-memory == DB
    assert len(r2) == 2 and (u2, b2) == (u1, b1)                                               # idempotent


def test_swap_is_capped_and_never_drains_the_account(cfg):
    async def scenario(db):
        db.add(_order(1, "MOMENTUM", 40.0, pnl=-40.0)); await db.commit()
        e = _engine(need=3019.52, threshold=1509.76, stub_swap=False)                           # the uncapped Sep-29 target, fed straight in
        _, _, _, nav0 = await _book(e, db)
        await e._execute_bnb_swap(db, swap_type="emergency")
        u, b, rows, nav = await _book(e, db)
        return u, b, rows, nav0, nav
    u, b, rows, nav0, nav = _run_db(scenario)
    cap = 0.10 * (2200.0 + 100.0 - 40.0)
    assert len(rows) == 1 and abs(rows[0].amount_usdt - cap) < 1e-6 and abs(b - (60.0 + cap)) < 1e-6
    assert u > 1900.0 and abs(nav - nav0) < 1e-6

    async def nearly_empty(db):
        cfg.paper_balance = 120.0
        db.add(_order(1, "MOMENTUM", 90.0, pnl=0.0)); await db.commit()                         # reserve 10, USDT 120 + 90 fees added back − 0
        e = _engine(need=50.0, threshold=25.0, stub_swap=False)
        u0, b0, _, nav0 = await _book(e, db)
        await e._execute_bnb_swap(db, swap_type="emergency")
        u1, b1, rows, nav1 = await _book(e, db)
        cfg.paper_balance = 51.0; e._bnb_projected_need = 90.0                                  # wants $40 more, USDT is 101: $1 above the minimum → no swap
        await e._execute_bnb_swap(db, swap_type="emergency")
        _, _, rows2, _ = await _book(e, db)
        return u0, b0, u1, b1, rows, nav0, nav1, len(rows2)
    u0, b0, u1, b1, rows, nav0, nav1, n2 = _run_db(nearly_empty)
    assert abs(b0 - 10.0) < 1e-6 and abs(b1 - 50.0) < 1e-6 and abs(u1 - (u0 - 40.0)) < 1e-6 and u1 >= 100.0 and abs(nav1 - nav0) < 1e-6
    assert n2 == len(rows) == 1


def test_manual_buy_path_and_the_forecast_scope(cfg):
    """_paper_bnb_credit is the one ledger path (automatic swap + the manual BNB buy endpoint). The burn rate is the TRUE one
    (every fee paid, manual fills included — what the dashboard shows); the targets are capped; the bot-only rate is kept
    apart for the position-sizing fee leg."""
    async def scenario(db):
        e = _engine(need=0.0, threshold=0.0, stub_swap=False)
        await e._recompute_bnb_burn_rate(db)                                                    # no history at all
        empty = (e._bnb_projected_need, e._bnb_emergency_threshold)
        db.add(_order(1, "MANUAL", 226.5, pnl=-226.5)); db.add(_order(2, "MOMENTUM", 44.0, pnl=-44.0, hours_ago=2.0))
        db.add(_order(3, None, 44.0, pnl=-44.0)); db.add(_order(4, "MANUAL", 60.0, status="OPEN")); await db.commit()
        e.total_runtime_seconds = int(2.5 * 3600)
        fees_24h = await e._recompute_bnb_burn_rate(db)
        _, _, _, nav0 = await _book(e, db)
        pre_bnb, post_bnb, pre_usdt, post_usdt = await e._paper_bnb_credit(db, 80.0, "manual", 600.0)
        u, b, rows, nav = await _book(e, db)
        return empty, fees_24h, e, (pre_bnb, post_bnb, pre_usdt, post_usdt), (u, b, rows, nav0, nav)
    empty, fees_24h, e, (pre_bnb, post_bnb, pre_usdt, post_usdt), (u, b, rows, nav0, nav) = _run_db(scenario)
    assert empty == (50.0, 25.0)
    assert abs(fees_24h - 314.5) < 1e-6 and abs(e._bnb_burn_rate - 314.5 / 2.5) < 1e-6          # $125.80/hr: what the account really pays
    assert abs(e._bnb_burn_rate_bot - 88.0 / 2.5) < 1e-6                                        # $35.20/hr: the bot's own fills (sizing leg)
    cap = 0.10 * (2200.0 + 100.0 - 314.5)
    assert abs(e._bnb_projected_need - cap) < 1e-6 and abs(e._bnb_emergency_threshold - cap / 2) < 1e-6   # $3,019 / $1,509 uncapped
    assert pre_bnb == 0.0 and abs(post_bnb - 80.0) < 1e-6 and abs(post_usdt - (pre_usdt - 80.0)) < 1e-6
    assert [r.swap_type for r in rows] == [T.PAPER_FEE_SETTLE, "manual"] and abs(nav - nav0) < 1e-6 and abs(b - 80.0) < 1e-6


def test_scheduled_wake_refills_an_empty_reserve_on_a_manual_only_batch(cfg):
    """Sep-29b, the state right after the reserve-cap deploy: $2,900 seed all in 5 manual positions, $265.53 of fees paid
    (reserve $100 + $165.53 from USDT), no systematic fills, 6 h gate closed, data window under 2 h. The 15-min wake must
    still refill the reserve — within the min-investment guard — and the burn rate must show the fees really paid."""
    cfg.paper_balance = 2900.0

    async def scenario(db):
        for i, fee in enumerate([60.0, 50.0, 40.0]):
            db.add(_order(i, "MANUAL", fee, pnl=75.0))                                          # closed manual fills: fees 150, pnl +225
        for i, fee in enumerate([70.0, 60.0, 50.0, 30.0, 21.06]):
            o = _order(10 + i, "MANUAL", fee, status="OPEN"); o.investment = 580.0; db.add(o)   # 5 open × 580 = 2,900 margin, entry fees 115.53
        await db.commit()
        e = _engine(need=0.0, threshold=0.0, stub_swap=False)
        e._last_bnb_check = dt.datetime.utcnow(); e._filter_block_counts = {}                   # interval gate closed
        async def _nofee(): return None
        e._sync_fee_rates = _nofee
        u0, b0, r0, nav0 = await _book(e, db)
        await e.bnb_scheduled_check(db)
        u1, b1, r1, nav1 = await _book(e, db)
        await e.bnb_scheduled_check(db)                                                         # next wake: reserve above the floor → nothing
        u2, b2, r2, nav2 = await _book(e, db)
        return (u0, b0, r0, nav0), (u1, b1, r1, nav1), (u2, b2, len(r2)), e
    (u0, b0, r0, n0), (u1, b1, r1, n1), (u2, b2, n_rows2), e = _run_db(scenario)
    assert b0 == 0.0 and r0 == [] and abs(u0 - (2900.0 + 225.0 + 150.0 - 2900.0 - 165.53)) < 1e-6
    assert e._bnb_burn_rate > 100 and e._bnb_burn_rate_bot == 0 and e._bnb_data_mature is False   # true burn shown; nothing held back from bot sizing
    assert [r.swap_type for r in r1] == [T.PAPER_FEE_SETTLE, "emergency"] and abs(r1[0].amount_usdt - 165.53) < 1e-6
    # under 2 h of history the refill goes to the $50 dollar floor, never to the extrapolated (even capped) target
    assert abs(e._bnb_projected_need - 0.10 * (2900.0 + 100.0 + 225.0)) < 1e-6
    assert abs(b1 - 50.0) < 1e-6 and abs(u1 - (u0 - 50.0)) < 1e-6 and abs(n1 - n0) < 1e-6
    assert n_rows2 == 2 and (u2, b2) == (u1, b1)
    assert T.bnb_real_floor_usd(cfg, 50.0) == 25.0 and T.bnb_real_floor_usd(cfg, 20.0) == 20.0 and T.bnb_real_floor_usd(cfg, 0) == 0.0


def _wake_engine():
    e = _engine(need=0.0, threshold=0.0, stub_swap=False)
    e._last_bnb_check = dt.datetime.utcnow(); e._filter_block_counts = {}

    async def _nofee(): return None
    e._sync_fee_rates = _nofee
    return e


def test_wake_refill_refusals_write_nothing_and_a_healthy_reserve_is_left_alone(cfg):
    async def scenario(db, fees, seed_usdt, **kw):
        cfg.paper_balance = seed_usdt
        for k, v in kw.items(): setattr(cfg, k, v)
        db.add(_order(1, "MANUAL", fees, pnl=0.0)); await db.commit()
        e = _wake_engine(); e._last_bnb_check = None                                            # interval gate OPEN
        _, _, _, nav0 = await _book(e, db)
        for _ in range(4):
            await e.bnb_scheduled_check(db)
        u, b, rows, nav = await _book(e, db)
        return b, [r.swap_type for r in rows], abs(nav - nav0) < 1e-6
    # reserve 40: between the floor (25) and the target (50) → the wake leaves it alone (routine is suppressed: no bot history)
    assert _run_db(lambda db: scenario(db, 60.0, 2900.0)) == (40.0, [], True)
    # empty reserve but free USDT at the minimum kept for trading → refused on every wake, nothing written
    assert _run_db(lambda db: scenario(db, 130.0, 100.0 - 130.0 + 30.0)) == (0.0, [], True)      # USDT = seed + fees − deficit = 100
    # a $4 purchase is under the $5 minimum → refused, nothing written
    assert _run_db(lambda db: scenario(db, 130.0, 104.0 - 130.0 + 30.0)) == (0.0, [], True)      # USDT 104
    # auto-swap OFF → nothing fires
    assert _run_db(lambda db: scenario(db, 130.0, 2900.0, bnb_swap_enabled=False)) == (0.0, [], True)


def test_forced_check_and_concurrent_triggers(cfg):
    """A forced check refills like the wake does; a wake and a fee event landing together book ONE purchase."""
    cfg.paper_balance = 2900.0

    async def forced(db):
        db.add(_order(1, "MANUAL", 130.0, pnl=0.0)); await db.commit()
        e = _wake_engine()
        await e.bnb_scheduled_check(db, force=True)
        _, b, rows, _ = await _book(e, db)
        return b, [r.swap_type for r in rows]
    assert _run_db(forced) == (50.0, [T.PAPER_FEE_SETTLE, "emergency"])                         # young batch: dollar floor

    async def mature(db):                                                                       # 3 h of history: the capped target rules
        db.add(_order(1, "MANUAL", 130.0, pnl=0.0, hours_ago=3.0)); await db.commit()
        e = _wake_engine()
        await e.bnb_scheduled_check(db)
        _, b, rows, _ = await _book(e, db)
        return b, [r.swap_type for r in rows], e._bnb_data_mature, e._bnb_burn_rate, e._bnb_burn_rate_bot
    b, kinds, is_mature, burn, bot = _run_db(mature)
    assert is_mature is True and bot == 0 and abs(burn - 130.0 / 3.0) < 0.05                    # $43.33/hr shown; nothing for the sizing leg
    assert kinds == [T.PAPER_FEE_SETTLE, "emergency"] and abs(b - 300.0) < 1e-6                 # 10 % of $3,000

    async def slow_px():
        await asyncio.sleep(0.05); return 600.0
    T.binance_service.get_bnb_price = slow_px                                                   # (restored by the cfg fixture)

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    import models

    async def race():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        Session = async_sessionmaker(eng, expire_on_commit=False)
        async with Session() as db:
            db.add(_order(1, "MANUAL", 265.53, pnl=0.0)); await db.commit()
        e = _wake_engine(); e._bnb_emergency_threshold = 150.0; e._bnb_projected_need = 300.0; e._bnb_data_mature = False

        async def wake():
            async with Session() as db: await e.bnb_scheduled_check(db)

        async def fee_event():
            async with Session() as db: await e._deduct_fee_from_bnb(9.0, db)
        await asyncio.gather(wake(), fee_event(), wake())
        async with Session() as db:
            out = await _book(e, db)
        await eng.dispose()
        return out
    u, b, rows, nav = asyncio.run(race())
    assert [r.swap_type for r in rows] == [T.PAPER_FEE_SETTLE, "emergency"] and abs(b - 50.0) < 1e-6
    assert abs(rows[0].amount_usdt - 165.53) < 1e-6 and abs(u - (2900.0 + 265.53 - 165.53 - 50.0)) < 1e-6


def test_wiring_parity():
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
    assert "bnb_max_reserve_pct_of_equity" in cfg and 0 <= float(cfg["bnb_max_reserve_pct_of_equity"]) <= 50
    import config as C
    assert "bnb_max_reserve_pct_of_equity" in type(C.trading_config).model_fields
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("bnb_reserve_cap_usd(tc, await self._equity_usd(db))") == 2                 # paper + live swap size guard
    i = eng.index("    async def _recompute_bnb_burn_rate"); j = eng.index("    async def _sync_fee_rates", i)
    assert eng[i:j].count("_bot_open_filter()") == 1 and "bnb_reserve_targets(" in eng[i:j]      # true burn, capped; bot-only rate apart
    assert "_burn = float(getattr(self, '_bnb_burn_rate_bot', 0.0)" in eng and "_burn = float(getattr(self, '_bnb_burn_rate'," not in eng
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_burn = float(getattr(trading_engine, '_bnb_burn_rate_bot', 0.0)" in main            # the sizing leg's dashboard mirror
    assert '"bnb_burn_rate_bot"' in eng and '"burn_rate_bot_per_hour"' in main                   # both payloads carry the bot rate
    assert eng.count("self._equity_usd_last = float(") == 3                                       # stamped with every live balance read
    assert "bnb_max_reserve_pct_of_equity: Optional[float] = Field(default=None, ge=0, le=50)" in main
    assert "_paper_bnb_credit(db, amount, \"manual\", bnb_price)" in main                        # manual BNB buy uses the same ledger path
    ui = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert ui.count("config-bnb-max-reserve-pct") >= 3 and "BNB reserve ceiling (Sep 29)" in ui
    assert ui.count("fee_settle") >= 3                                                           # table badge + amount cell + report line
    assert "status.bnb_burn_rate_bot" in ui and ui.count("burn_rate_bot_per_hour") >= 2          # config report line; swap report line + tab tooltip
