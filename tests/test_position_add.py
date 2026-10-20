"""➕ Add to an open position (DECISION_LOG 324). Pins the pure math (average entry, invested, fees, stop / TP prices on the new average),
every refusal, the LIVE order of operations (positions read → TP withdraw → market order → row UPDATE → safety stop re-place → TP heal),
the flatten of an add that cannot be booked, the paper path (no exchange call at all), the ADD_ transaction and the preview endpoint.
No network: every exchange call is a monkeypatched stub."""
import asyncio, datetime as dt, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import pytest

from services.position_add import (add_fill_math, add_preview_math, add_exit_refusal, add_level_refusal, pnl_pct_net, price_at_pnl,
                                   rederive_peak_trough, bracket_max_add_margin, qty_matches, ADD_REFUSE_TRAILING)

TF = 0.00045
import itertools
_DB_SEQ = itertools.count()


# ── pure math ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_fill_math_weighted_average_and_money():
    m = add_fill_math(1000, 2.0, 100.0, 0.9, 20, 500, 1.94, TF)
    assert abs(m['quantity'] - 1500) < 1e-9
    assert abs(m['entry_price'] - (1000 * 2.0 + 500 * 1.94) / 1500) < 1e-12           # qty-weighted average
    assert abs(m['investment'] - (100.0 + 500 * 1.94 / 20)) < 1e-9                    # + fill notional / leverage
    assert abs(m['notional_value'] - m['quantity'] * m['entry_price']) < 1e-9         # new qty × new entry
    assert abs(m['entry_fee'] - (0.9 + 500 * 1.94 * TF)) < 1e-12 and abs(m['add_fee'] - 500 * 1.94 * TF) < 1e-12
    assert abs(m['add_margin'] - 48.5) < 1e-9
    for bad in ((0, 2.0, 100, 0, 20, 1, 1), (1000, 2.0, 100, 0, 20, 0, 1), (1000, 2.0, 100, 0, 20, 1, 0), (1000, 2.0, 100, 0, 0, 1, 1),
                (1000, float('nan'), 100, 0, 20, 1, 1)):
        with pytest.raises(ValueError):
            add_fill_math(*bad, TF)


def test_stop_and_target_prices_invert_the_exit_pnl():
    for d in ("LONG", "SHORT"):
        for lvl in (-1.5, -0.4, 0.5, 3.0):
            p = price_at_pnl(d, 2.0, 1500, 1.35, lvl, TF)
            assert abs(pnl_pct_net(d, 2.0, 1500, 1.35, p, TF) - lvl) < 1e-9
    assert price_at_pnl("LONG", 2.0, 1000, 0.9, -1.5, TF) < 2.0 < price_at_pnl("LONG", 2.0, 1000, 0.9, 3.0, TF)
    assert price_at_pnl("SHORT", 2.0, 1000, 0.9, -1.5, TF) > 2.0 > price_at_pnl("SHORT", 2.0, 1000, 0.9, 3.0, TF)
    assert pnl_pct_net("LONG", 0, 1, 0, 1, TF) is None and price_at_pnl("LONG", 2, 0, 0, -1, TF) is None


def test_preview_math_and_risk():
    pv = add_preview_math("LONG", 1000, 2.0, 100.0, 0.9, 20, 50.0, 1.9, TF, sl_pct=-1.5, tp_pct=3.0)
    assert abs(pv['add_qty'] - 50 * 20 / 1.9) < 1e-9 and abs(pv['add_notional'] - 1000.0) < 1e-9
    assert abs(pv['risk_usd_at_stop'] - 0.015 * pv['notional_value']) < 1e-9          # P&L % is on the entry notional, net of fees
    assert abs(pnl_pct_net("LONG", pv['entry_price'], pv['quantity'], pv['entry_fee'], pv['stop_price'], TF) + 1.5) < 1e-9
    assert abs(pnl_pct_net("LONG", pv['entry_price'], pv['quantity'], pv['entry_fee'], pv['tp_price'], TF) - 3.0) < 1e-9
    assert add_preview_math("LONG", 1000, 2.0, 100, 0.9, 20, 50, 1.9, TF, sl_pct=0.5)['risk_usd_at_stop'] < 0   # a profit-lock stop = no loss


def test_exit_scope_levels_and_helpers():
    now = dt.datetime(2026, 10, 20)
    assert add_exit_refusal("MANUAL", "FIXED", None) is None and add_exit_refusal("MANUAL", "FLOOR", None) is None
    assert add_exit_refusal("MANUAL", None, None) is None                                 # NULL mode = FIXED (the exits' default)
    for mode in ("MOMENTUM", "FRENZY", "BULLRUN", "BULLRUN_SL"):
        assert add_exit_refusal("MANUAL", mode, None) == ADD_REFUSE_TRAILING
    assert add_exit_refusal("FRENZY_LONG", None, None) == ADD_REFUSE_TRAILING and add_exit_refusal("MOMENTUM", None, now) is None
    assert "switch this trade to a fixed stop/target first" in ADD_REFUSE_TRAILING and "then add" in ADD_REFUSE_TRAILING
    assert "stop out at once" in add_level_refusal(-1.36, -1.5, 3.0) and "stop out at once" in add_level_refusal(-2.0, -1.5, 3.0)
    assert "stop out at once" in add_level_refusal(0.6, 0.5, 3.0)                       # a profit-lock stop the price sits on
    assert add_level_refusal(-1.34, -1.5, 3.0) is None and add_level_refusal(2.9, -1.5, 3.0) is None
    assert "take profit at once" in add_level_refusal(3.0, -1.5, 3.0) and add_level_refusal(9.0, -1.5, None) is None
    pk, tr = rederive_peak_trough("LONG", 2.0, 1000, 0.9, 2.1, 1.9, 0.0, TF)
    assert abs(pk - pnl_pct_net("LONG", 2.0, 1000, 0.9, 2.1, TF)) < 1e-12 and abs(tr - pnl_pct_net("LONG", 2.0, 1000, 0.9, 1.9, TF)) < 1e-12
    pk, tr = rederive_peak_trough("SHORT", 2.0, 1000, 0.9, 2.1, 1.9, 0.0, TF)          # SHORT: the peak is at the LOW
    assert pk > 0 > tr and abs(pk - pnl_pct_net("SHORT", 2.0, 1000, 0.9, 1.9, TF)) < 1e-12
    assert rederive_peak_trough("LONG", 2.0, 1000, 0.9, None, None, -0.2, TF) == (-0.2, -0.2)
    assert bracket_max_add_margin(10_000, 20, 1000, 2.0) == 400.0 and bracket_max_add_margin(1000, 20, 1000, 2.0) == 0.0
    assert bracket_max_add_margin(None, 20, 1000, 2.0) is None
    assert qty_matches(1004, 1000) and not qty_matches(1006, 1000) and qty_matches(1.0, 1.0001, step=0.001) and not qty_matches(None, 1)


# ── engine harness ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _harness(monkeypatch, tmp_path, *, paper=False, price=2.0, positions=(("LONG", 1000.0, 2.0),), positions_after=None, order_result="ok",
             filled=None, withdraw="withdrawn", replace="replaced", tiers=None, balance=10_000.0, close_ok=True, commit_fails=False,
             close_row_during_order=False, positions_seq=None, on_withdraw=None, on_replace=None, ack=None, commit_then_raise=None,
             bk_cancel_ok=True, **row):
    """A real engine method on a file SQLite DB with ONE open row and a faked exchange. Returns (run, log, T); log = every exchange /
    resting-TP / backstop step in order (plus 'commit' at the row UPDATE's commit)."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, services.trading_engine as T
    log = []
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper; eng.paper_balance = 0.0
    eng._record_filter_block = lambda name, d, had_room=True: log.append(("block", name))

    async def _noop(*a, **k): return None
    eng.update_orders_cache = _noop
    eng.save_state = _noop
    async def _ded(fee, db, defer_swap=False): log.append(("bnb_fee", round(fee, 6))); return False
    eng._deduct_fee_from_bnb = _ded
    async def _wd(oid, why):   # like the real one: a 'withdrawn' TP leaves the row with no TP id / price
        log.append(("tp_withdraw", why))
        if withdraw == "withdrawn":
            async with S() as s2:
                from sqlalchemy import update as _u
                await s2.execute(_u(models.Order).where(models.Order.id == oid).values(protective_tp_order_id=None, resting_tp_price=None)); await s2.commit()
        if on_withdraw is not None:
            await on_withdraw(S, oid, sum(1 for x in log if x[0] == "tp_withdraw"))
        return withdraw
    async def _heal(oid): log.append(("tp_heal", oid)); return "placed"
    eng._rtp_withdraw = _wd; eng._rtp_heal = _heal
    async def _repl(pair, direction, old_id, pct, trig):
        log.append(("backstop_replace", old_id, pct, round(trig, 6)))
        if on_replace is not None:
            await on_replace(S, log)
        return (replace, ("algo-new" if replace == "replaced" else (None if replace == "lost" else old_id)))
    eng._backstop_replace_one = _repl
    monkeypatch.setattr(T, "MANUAL_FILL_POLL_S", 0.0); monkeypatch.setattr(T, "ADD_RECHECK_POLL_S", 0.0)
    monkeypatch.setattr(T, "_open_orders_cache", {})
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "broker_backstop_pct", 2.5, raising=False)
    monkeypatch.setattr(T.config.trading_config, "taker_fee", TF, raising=False)
    monkeypatch.setattr(th, "resting_tp_manual_enabled", True, raising=False); monkeypatch.setattr(th, "resting_tp_override_enabled", True, raising=False)
    monkeypatch.setattr(T.config.trading_config.investment, "leverage_bracket_cap_enabled", tiers is not None, raising=False)
    B = T.binance_service
    state = {"pos": 0}

    def _boom(name):
        async def f(*a, **k):
            raise AssertionError(f"paper add must never call the exchange ({name})")
        return f
    if paper:
        for name in ("get_open_positions", "create_market_order", "close_position", "get_current_price", "get_balance", "place_backstop_stop",
                     "cancel_backstop_stop", "fetch_open_reduce_only_orders", "cancel_resting_tp"):
            monkeypatch.setattr(B, name, _boom(name))
    else:
        async def _positions():
            state["pos"] += 1; log.append(("positions", state["pos"]))
            if positions_seq is not None:
                src = positions_seq[min(state["pos"] - 1, len(positions_seq) - 1)]
            else:
                src = positions if (state["pos"] <= 2 or positions_after is None) else positions_after   # reads 1-2: the two pre-order checks
            if src is None:
                return None
            return [dict(symbol="FOO/USDT:USDT", side=t[0], contracts=t[1], entry_price=t[2], leverage=(t[3] if len(t) > 3 else 1),
                         notional=(t[4] if len(t) > 4 else 0.0), margin=(t[5] if len(t) > 5 else 0.0)) for t in src]
        async def _order(symbol, side, amount, leverage=1, is_close=False, status=None):
            log.append(("order", side, round(amount, 4), leverage, is_close))
            if close_row_during_order:
                async with S() as s2:
                    from sqlalchemy import update as _u
                    await s2.execute(_u(models.Order).values(status="CLOSED")); await s2.commit()
            if order_result == "notsent":
                return None
            if status is not None:
                status["sent"] = True
            if order_result is None:
                return None
            if ack is not None:
                return dict(ack, amount=amount)
            return {"id": "A-9", "price": 1.9, "amount": amount, "filled": (amount if filled is None else filled)}
        async def _close(symbol, side, amount):
            log.append(("flatten", side, round(amount, 4))); return {"id": "F-1"} if close_ok else None
        async def _bal(): log.append(("balance",)); return {"usdt_free": balance}
        async def _px(symbol): return price
        async def _cancel(pair, algo_id): log.append(("cancel_backstop", algo_id)); return bk_cancel_ok
        for name, fn in (("get_open_positions", _positions), ("create_market_order", _order), ("close_position", _close), ("get_balance", _bal),
                         ("get_current_price", _px), ("cancel_backstop_stop", _cancel)):
            monkeypatch.setattr(B, name, fn)
    if tiers is not None:
        async def _brk(): return {"FOOUSDT": tiers}
        monkeypatch.setattr(B, "get_leverage_brackets", _brk)

    class _Trk:
        last_price = price
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: _Trk())
    monkeypatch.setattr(T.websocket_tracker, "pair_silence_seconds", lambda p: 1.0)
    _real_commit = T.locked_commit
    async def _commit(db):
        if commit_fails:
            raise RuntimeError("database is locked")
        first = not any(x[0] == "commit" for x in log)
        if first:
            log.append(("commit",))
        r = await _real_commit(db)
        if first and commit_then_raise is not None:   # (round-4 #3) the DB commit SUCCEEDED, then the call reports an error / is cancelled
            raise commit_then_raise
        return r
    monkeypatch.setattr(T, "locked_commit", _commit)

    base = dict(pair="FOOUSDT", direction="LONG", status="OPEN", entry_price=2.0, quantity=1000.0, investment=100.0, leverage=20.0,
                notional_value=2000.0, confidence="STRONG_BUY", entry_fee=2000.0 * TF, is_paper=paper, entry_strategy="MANUAL",
                manual_exit_mode="FIXED", pattern_fixed_sl_pct=-1.5, pattern_fixed_tp_pct=3.0, backstop_algo_id="algo-old",
                protective_tp_order_id=(None if paper else "tp-1"), resting_tp_price=2.06, high_price_since_entry=2.05, low_price_since_entry=1.95,
                peak_pnl=2.4, trough_pnl=-2.6, opened_at=dt.datetime.utcnow() - dt.timedelta(minutes=30))
    base.update(row)
    eng_db = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'add_{next(_DB_SEQ)}.db'}")   # (R3) one file per harness call
    S = async_sessionmaker(eng_db, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)   # the late-add re-check (I1) reads the row in its own session

    async def run(fn):
        async with eng_db.begin() as c:
            await c.run_sync(models.Base.metadata.drop_all); await c.run_sync(models.Base.metadata.create_all)
        async with S() as db:
            o = models.Order(**base); db.add(o); await db.flush()
            db.add(models.Transaction(order_id=o.id, pair="FOOUSDT", action="OPEN_LONG", price=2.0, quantity=1000.0, investment=100.0, leverage=20.0,
                                      notional_value=2000.0, fee=0.9, is_paper=paper))
            await db.commit(); oid = o.id
        try:
            async with S() as db:
                try:
                    return await fn(eng, db, oid)
                except (ValueError, RuntimeError) as err:
                    return f"{type(err).__name__}: {err}"
        finally:
            # (R3) a late-add re-check (I1) must never outlive this harness's DB: cancel + await it before the dispose
            _tasks = [t for t in (eng.__dict__.get('_add_unresolved') or {}).values() if isinstance(t, asyncio.Future)]
            for t in _tasks:
                t.cancel()
            if _tasks:
                await asyncio.gather(*_tasks, return_exceptions=True)
            await eng_db.dispose()

    async def read(oid=1):
        from sqlalchemy import select
        async with S() as db:
            o = (await db.execute(select(models.Order).where(models.Order.id == oid))).scalar_one()
            tx = (await db.execute(select(models.Transaction).order_by(models.Transaction.id))).scalars().all()
            return o, tx
    return run, log, T, read, eng


def _add(amount):
    async def fn(eng, db, oid):
        return await eng.add_to_position(db, oid, amount)
    return fn


def _kinds(log):
    return [x[0] for x in log if x[0] not in ("block", "balance")]


def test_live_happy_path_order_of_operations(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path)
    o = asyncio.run(run(_add(50.0)))                                                  # $50 × 20 = $1,000 at 2.0 → 500 FOO, filled at 1.9
    assert not isinstance(o, str), o
    assert _kinds(log) == ["positions", "tp_withdraw", "positions", "order", "commit", "backstop_replace", "tp_heal"]   # read again under the book lock
    _ord = [x for x in log if x[0] == "order"][0]; _rep = [x for x in log if x[0] == "backstop_replace"][0]
    assert _ord == ("order", "buy", 500.0, 20, False) and type(_ord[3]) is int        # same side, NOT reduce-only, the row's leverage
    exp = add_fill_math(1000, 2.0, 100.0, 2000 * TF, 20, 500, 1.9, TF)
    assert abs(o.quantity - 1500) < 1e-9 and abs(o.entry_price - exp['entry_price']) < 1e-12 and abs(o.investment - exp['investment']) < 1e-9
    assert abs(o.entry_fee - exp['entry_fee']) < 1e-12 and abs(o.notional_value - exp['notional_value']) < 1e-9
    assert o.manual_add_count == 1 and abs(o.manual_add_investment - 47.5) < 1e-9 and o.manual_added_at is not None
    assert o.backstop_algo_id == "algo-new" and o.backstop_pct == 2.5
    assert abs(o.backstop_trigger_price - exp['entry_price'] * (1 - 0.025)) < 1e-9 and _rep[1] == "algo-old"   # re-placed on the NEW average
    assert o.pattern_fixed_sl_pct == -1.5 and o.pattern_fixed_tp_pct == 3.0 and o.manual_exit_mode == "FIXED" and o.status == "OPEN"
    pk_exp, tr_exp = rederive_peak_trough("LONG", exp['entry_price'], 1500, exp['entry_fee'], 2.05, 1.95,
                                          pnl_pct_net("LONG", exp['entry_price'], 1500, exp['entry_fee'], 1.9, TF), TF)
    assert abs(o.peak_pnl - round(pk_exp, 4)) < 1e-9 and abs(o.trough_pnl - round(tr_exp, 4)) < 1e-9   # re-based on the new entry


def test_live_add_writes_an_add_transaction_and_one_row(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path)
    async def go(eng, db, oid):
        await eng.add_to_position(db, oid, 50.0)
        from sqlalchemy import select, func
        import models
        n = (await db.execute(select(func.count(models.Order.id)))).scalar()
        tx = (await db.execute(select(models.Transaction).order_by(models.Transaction.id))).scalars().all()
        return n, [(t.action, t.quantity, t.price, round(t.investment, 6), t.binance_order_id) for t in tx]
    n, tx = asyncio.run(run(go))
    assert n == 1                                                                      # never a second row / position
    assert tx[0][0] == "OPEN_LONG" and tx[1] == ("ADD_LONG", 500.0, 1.9, 47.5, "A-9")
    assert sum(1 for t in tx if t[0].startswith("OPEN_")) == 1                         # the add is not an OPEN in any OPEN_/CLOSE_ read


def test_live_partial_fill_books_the_actual_quantity(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, filled=200.0)
    o = asyncio.run(run(_add(50.0)))
    assert abs(o.quantity - 1200) < 1e-9 and abs(o.manual_add_investment - 200 * 1.9 / 20) < 1e-9
    assert abs(o.entry_price - (2000 + 200 * 1.9) / 1200) < 1e-12


def test_live_refusals_send_nothing(monkeypatch, tmp_path):
    import services.trading_engine as T
    def refused(text, amount=50.0, **kw):
        run, log, T2, read, _ = _harness(monkeypatch, tmp_path, **kw)
        r = asyncio.run(run(_add(amount)))
        assert isinstance(r, str) and text in r, r
        assert "order" not in _kinds(log) and "flatten" not in _kinds(log)
        return log
    refused("switch this trade to a fixed stop/target first", manual_exit_mode="MOMENTUM")
    refused("switch this trade to a fixed stop/target first", manual_exit_mode="FRENZY")
    refused("switch this trade to a fixed stop/target first", manual_exit_mode="BULLRUN_SL")
    refused("switch this trade to a fixed stop/target first", entry_strategy="FRENZY_LONG", manual_exit_mode=None)
    refused("being closed right now", closing_in_progress=True)
    refused("not open", status="CLOSED")
    refused("other mode", is_paper=True)
    refused("amount to add must be > 0", amount=0)
    refused("under Binance's minimum order", amount=0.2)                               # $0.2 × 20 = $4 < $5
    # app quantity netted into the position / wrong side / read failure → refused at the positions read, nothing sent, TP untouched
    lg = refused("Binance holds LONG 1500 FOOUSDT but this trade is LONG 1000 — close the extra in the Binance app first", positions=(("LONG", 1500.0, 2.0),))
    assert _kinds(lg) == ["positions"]
    lg = refused("Binance holds SHORT 1000 FOOUSDT but this trade is LONG 1000", positions=(("SHORT", 1000.0, 2.0),))
    assert _kinds(lg) == ["positions"]
    lg = refused("the order was NOT sent", positions=None)
    assert _kinds(lg) == ["positions"]
    refused("shows no FOOUSDT position", positions=())
    # the TP: filled while withdrawing / a blocked cancel → refused, never added beside an unknown limit; the TP is re-healed
    lg = refused("the take profit already filled", withdraw="FILLED")
    assert _kinds(lg) == ["positions", "tp_withdraw", "tp_heal"]
    lg = refused("could not be cancelled (blocked)", withdraw="blocked")
    assert _kinds(lg) == ["positions", "tp_withdraw", "tp_heal"]
    # retry queue
    run, log, _, _, _ = _harness(monkeypatch, tmp_path)
    async def in_retry(eng, db, oid):
        T._exit_retry_queue[oid] = 1
        try:
            return await eng.add_to_position(db, oid, 50.0)
        finally:
            T._exit_retry_queue.pop(oid, None)
    r = asyncio.run(run(in_retry)); assert "being closed right now" in r and not log


def test_bracket_cap_balance_and_stop_refusals(monkeypatch, tmp_path):
    # tier cap $3,000 at ≥ 20× → $2,850 with the 5 % headroom; the trade holds $2,000 → at most $42.50 more margin
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, tiers=[(3000.0, 25), (50_000.0, 10)])
    r = asyncio.run(run(_add(50.0)))
    assert "at most $42.50 more can be added" in r and "order" not in _kinds(log)
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, tiers=[(3000.0, 25)])
    o = asyncio.run(run(_add(40.0))); assert not isinstance(o, str), o
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, tiers=[(3000.0, 10)])          # the pair now allows 10× only
    assert "nothing can be added at this leverage" in asyncio.run(run(_add(10.0)))
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, balance=30.0)
    assert "exceeds the available balance" in asyncio.run(run(_add(50.0)))
    # the trade sits at ≈ −1.42 % on the new average (the add itself lowers the average) → within 0.15 of the −1.5 stop: it would stop out at once
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, price=1.972, entry_price=2.0, pattern_fixed_sl_pct=-1.5,
                                 positions=(("LONG", 1000.0, 2.0),))
    r = asyncio.run(run(_add(5.0)))
    assert "stop out at once" in r and "order" not in _kinds(log)
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, price=2.08, pattern_fixed_tp_pct=1.0)   # already above the +1 % target
    assert "take profit at once" in asyncio.run(run(_add(50.0)))
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, entry_strategy="FRENZY_LONG", exit_override_at=dt.datetime.utcnow(),
                                 manual_exit_mode=None, pattern_fixed_sl_pct=None)
    assert "no fixed stop on record" in asyncio.run(run(_add(50.0)))


def test_override_bot_row_can_take_an_add(monkeypatch, tmp_path):
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, entry_strategy="FRENZY_LONG", manual_exit_mode=None, exit_override_at=dt.datetime.utcnow())
    o = asyncio.run(run(_add(50.0)))
    assert not isinstance(o, str), o
    assert o.entry_strategy == "FRENZY_LONG" and o.manual_add_count == 1 and o.exit_override_at is not None


def test_unknown_outcome_books_from_the_exchange_or_says_check_binance(monkeypatch, tmp_path):
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, order_result=None, positions_after=(("LONG", 1500.0, 1.9666666666666666),))
    o = asyncio.run(run(_add(50.0)))
    assert not isinstance(o, str), o
    assert abs(o.quantity - 1500) < 1e-9 and abs(o.entry_price - 1.9666666666666666) < 1e-9   # delta = 500 @ (1.9667·1500 − 2·1000)/500 = 1.9
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, order_result=None)             # nothing grew after 4 reads
    r = asyncio.run(run(_add(50.0)))
    assert r.startswith("RuntimeError") and "CHECK BINANCE NOW" in r and "flatten" not in _kinds(log)
    k = _kinds(log)
    assert k[:8] == ["positions", "tp_withdraw", "positions", "order"] + ["positions"] * 4 and "tp_heal" in k   # 4 quick reads; the unchanged row gets its TP back
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, order_result="notsent")
    r = asyncio.run(run(_add(50.0)))
    assert "was NOT sent" in r and "Nothing was added" in r


def test_closed_meanwhile_flattens_the_added_quantity(monkeypatch, tmp_path):
    run, log, _, read, _ = _harness(monkeypatch, tmp_path, close_row_during_order=True)
    r = asyncio.run(run(_add(50.0)))
    assert r.startswith("RuntimeError") and "closed again at market" in r
    assert ("flatten", "LONG", 500.0) in log and "backstop_replace" not in _kinds(log)
    assert _kinds(log).index("flatten") > _kinds(log).index("order")


def test_commit_failure_flattens_and_leaves_the_row(monkeypatch, tmp_path):
    run, log, _, read, _ = _harness(monkeypatch, tmp_path, commit_fails=True)
    r = asyncio.run(run(_add(50.0)))
    assert r.startswith("RuntimeError") and "closed again at market" in r and ("flatten", "LONG", 500.0) in log
    run, log, _, read, _ = _harness(monkeypatch, tmp_path, commit_fails=True, close_ok=False, positions_after=(("LONG", 1500.0, 1.9667),))
    r = asyncio.run(run(_add(50.0)))
    assert "could NOT be closed again" in r and "close 500 FOOUSDT on Binance now" in r
    # (review M2) the flatten failed but Binance already holds only the row's size → no false alarm
    run, log, _, read, _ = _harness(monkeypatch, tmp_path, commit_fails=True, close_ok=False, positions_after=(("LONG", 1000.0, 2.0),))
    r = asyncio.run(run(_add(50.0)))
    assert "already shows no extra quantity" in r and "could NOT be closed again" not in r


def test_backstop_failure_still_books_and_is_counted(monkeypatch, tmp_path):
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, replace="lost")
    o = asyncio.run(run(_add(50.0)))
    assert not isinstance(o, str) and o.backstop_algo_id is None and ("block", "BACKSTOP_PLACE_FAILED") in log
    run, log, _, _, _ = _harness(monkeypatch, tmp_path, replace="kept_old")
    o = asyncio.run(run(_add(50.0)))
    assert o.backstop_algo_id == "algo-old" and ("block", "BACKSTOP_PLACE_FAILED") in log


def test_paper_add_makes_no_exchange_call(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, paper=True)
    async def go(eng, db, oid):
        b0 = await eng._recalculate_paper_balance(db)
        o = await eng.add_to_position(db, oid, 50.0)
        b1 = await eng._recalculate_paper_balance(db)
        return o, b0, b1
    o, b0, b1 = asyncio.run(run(go))
    assert abs(o.quantity - 1500) < 1e-9 and abs(o.entry_price - 2.0) < 1e-12                  # filled at the live price
    assert abs((b0 - b1) - 50.0) < 1e-6                                                         # the added margin leaves free USDT once (fee from BNB)
    assert _kinds(log) == ["tp_withdraw", "commit", "bnb_fee", "tp_heal"]                       # no positions read, no order, no stop re-place
    assert o.backstop_pct == 2.5 and abs(o.backstop_trigger_price - 2.0 * 0.975) < 1e-9          # paper: the record of what live would place


def test_preview_endpoint(monkeypatch, tmp_path):
    import main
    run, log, T, read, eng = _harness(monkeypatch, tmp_path, tiers=[(3000.0, 25)])
    monkeypatch.setattr(main, "trading_engine", eng)
    async def go(eng, db, oid):
        a = await main.manual_add_preview(order_id=oid, add_investment=20.0, db=db)
        b = await main.manual_add_preview(order_id=oid, add_investment=80.0, db=db)
        c = await main.manual_add_preview(order_id=oid, add_investment=0.0, db=db)
        return a, b, c
    a, b, c = asyncio.run(run(go))
    pv = add_preview_math("LONG", 1000, 2.0, 100.0, 2000 * TF, 20, 20.0, 2.0, TF, -1.5, 3.0)
    assert a["refusal"] is None and a["price"] == 2.0 and abs(a["new_qty"] - 1200) < 1e-9
    assert abs(a["new_avg_entry"] - pv["entry_price"]) < 1e-12 and a["new_invested"] == round(pv["investment"], 2)
    assert a["new_notional"] == round(pv["notional_value"], 2) and abs(a["stop_price"] - pv["stop_price"]) < 1e-12
    assert abs(a["tp_price"] - pv["tp_price"]) < 1e-12 and a["risk_usd_at_stop"] == round(pv["risk_usd_at_stop"], 2)
    assert a["max_add_investment"] == 42.5 and a["max_add_bracket"] == 42.5
    assert "at most $42.50 more can be added" in b["refusal"] and b["new_qty"] is not None   # numbers shown even when refused
    assert c["refusal"] == "the amount to add must be > 0" and c["max_add_investment"] == 42.5 and c["new_qty"] is None
    assert [x for x in log if x[0] in ("order", "positions", "tp_withdraw")] == []           # a preview sends nothing, reads no position


def test_open_orders_api_wiring_and_ui():
    src = open(os.path.join(ROOT, "main.py")).read()
    assert '@app.post("/api/manual/add")' in src and '@app.get("/api/manual/add_preview")' in src
    assert src.count('"manual_add_count": getattr(o, \'manual_add_count\', None)') == 2 and '"add_block": _add_exit_refusal(' in src
    mdl = open(os.path.join(ROOT, "models.py")).read(); dbs = open(os.path.join(ROOT, "database.py")).read()
    for c in ("manual_add_count", "manual_add_investment", "manual_added_at"):
        assert f"    {c} = Column(" in mdl and f"('{c}', " in dbs
    import models
    assert {"manual_add_count", "manual_add_investment", "manual_added_at"} <= {c.name for c in models.Order.__table__.columns}   # rides the orders CSV
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    for needle in ('id="add-position-modal"', 'id="add-pos-investment"', 'id="add-pos-btn"', "function openAddModal(", "function closeAddModal(",
                   "/api/manual/add_preview?", "`${API_BASE}/api/manual/add`", "${addButtonHtml(o)}", "${addBadgeHtml(o)}",
                   "setTimeout(addPreview, 300)", "window._lastTotalPortfolio", "addModalBackdropClick(event)",
                   "const _addDetail = (d, fb)", "_addDetail(d, res.statusText)", "_addDetail(d, 'Could not add to the position')"):   # M3: a 422 list
        assert needle in html, needle
    assert html.count("addButtonHtml(") == 2 and html.count("addBadgeHtml(") == 2


def test_origin_view_counts_an_add_as_touched():
    from services.origin_view import bot_touched, touched_summary
    now = dt.datetime(2026, 10, 20)
    assert bot_touched("FRENZY_LONG", "X", None, "FRENZY_TP", now) and not bot_touched("FRENZY_LONG", "X", None, "FRENZY_TP")
    assert not bot_touched("MANUAL", "X", None, "MANUAL_TP", now)                            # a manual row is never "touched bot"

    class O:
        def __init__(self, **k): self.__dict__.update(k)
    rows = [O(entry_strategy="FRENZY_LONG", confidence="X", exit_override_at=None, close_reason="FRENZY_TP", manual_added_at=now, pnl=4.0),
            O(entry_strategy="FRENZY_LONG", confidence="X", exit_override_at=None, close_reason="FRENZY_TP", pnl=7.0)]
    assert touched_summary(rows) == {"n": 1, "pnl": 4.0}


# ── review fixes (DECISION_LOG 324 · Reviews) ─────────────────────────────────────────────────────────────────────────────────────────
def _stale_close_case(monkeypatch, paper, caplog):
    """C1: a close decided on the row BEFORE an add (the caller's object) runs after the add committed."""
    import logging
    from sqlalchemy import select, update
    import models
    from tests.test_resting_tp import _setup, TK
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, paper=paper, close_px=99.0, rows=[dict(
        backstop_algo_id="B1", entry_strategy="MANUAL", manual_exit_mode="FIXED", pattern_fixed_sl_pct=-1.5)])
    new_e = (100.0 * 10 + 98.0 * 5) / 15; new_fee = 0.45 + 5 * 98.0 * TK

    async def go():
        await seed()
        async with S() as a:
            stale = (await a.execute(select(models.Order).where(models.Order.id == 1))).scalar_one()
            stale.shadow_peak_stretch_at_close = 9.9                                   # (round-3 #5) a close-time stamp the caller set before the close
            async with S() as b:   # the add commits in its own session while the close waits
                await b.execute(update(models.Order).where(models.Order.id == 1).values(
                    quantity=15.0, entry_price=new_e, entry_fee=new_fee, investment=100.0 + 49.0, notional_value=15 * new_e, manual_add_count=1,
                    backstop_algo_id=("B2" if not paper else "B1")))
                await b.commit()
            r1 = await eng._close_position_locked(a, stale, 99.0, "MANUAL_SL")
            flag, q_obj = getattr(stale, '_close_aborted_size_changed', False), stale.quantity
        rows1 = await read(); log1 = list(log)
        async with S() as c:
            fresh = (await c.execute(select(models.Order).where(models.Order.id == 1))).scalar_one()
            r2 = await eng._close_position_locked(c, fresh, 99.0, "MANUAL_SL")
        rows2 = await read(); await engine.dispose()
        return r1, flag, q_obj, rows1, log1, r2, rows2, list(log)
    with caplog.at_level(logging.WARNING):
        r1, flag, q_obj, rows1, log1, r2, rows2, log2 = asyncio.run(go())
    assert r1 is None and flag and q_obj == 15.0                                     # aborted; the caller's object now holds the new size
    assert log1 == []                                                                  # no backstop cancel, no market order
    assert rows1[1].status == "OPEN" and not rows1[1].closing_in_progress             # the flag is cleared: nothing is stuck "closing"
    assert rows1[1].shadow_peak_stretch_at_close is None                              # the caller's close stamp did not land on the OPEN row
    assert "[MANUAL_ADD_CLOSE_ABORT]" in caplog.text
    o = rows2[1]
    assert r2 is not None and o.status == "CLOSED"
    assert abs(o.pnl - ((99.0 - new_e) * 15 - new_fee - 15 * 99.0 * TK)) < 1e-6      # blended: the new average, the whole size, both fees
    assert "[RESIDUAL_NOT_OURS]" not in caplog.text
    return log2


def test_c1_stale_close_after_an_add_aborts_then_closes_the_new_size_live(monkeypatch, caplog):
    log2 = _stale_close_case(monkeypatch, False, caplog)
    assert ("market_close", "WIL/USDT:USDT", "LONG", 15.0) in log2                   # the added coins are closed WITH the row (never "foreign")
    assert ("bk_cancel", "WILUSDT", "B2") in log2 and ("bk_cancel", "WILUSDT", "B1") not in log2   # the add's re-placed safety stop


def test_c1_stale_close_after_an_add_aborts_paper(monkeypatch, caplog):
    assert _stale_close_case(monkeypatch, True, caplog) == []


def test_c1_wiring_and_endpoint_message():
    import inspect, services.trading_engine as T
    src = inspect.getsource(T.TradingEngine._close_position_locked)
    i_mark, i_chk = src.index("_mark_close_in_progress("), src.index("await self._close_size_changed(", src.index("_mark_close_in_progress("))
    assert i_mark < i_chk < src.index("cancel_backstop_stop(order.pair, order.backstop_algo_id)") and i_chk < src.index("await self._rtp_settle(order")
    assert "[CLOSE_INTENT_FAIL]" in src
    main_src = open(os.path.join(ROOT, "main.py")).read()
    assert "_close_aborted_size_changed" in main_src and "press Close again" in main_src


def test_c2_cache_is_rebased_before_the_safety_stop_and_tp(monkeypatch, tmp_path):
    seen = {}
    async def _on_rep(S, log):
        import services.trading_engine as T
        seen['ci'] = dict(T._open_orders_cache["FOOUSDT"][0])
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_replace=_on_rep)
    async def go(eng, db, oid):
        T._open_orders_cache["FOOUSDT"] = [dict(id=oid, entry_price=2.0, quantity=1000.0, entry_fee=0.9, peak_pnl=2.4, trough_pnl=-2.6)]
        return await eng.add_to_position(db, oid, 50.0)
    o = asyncio.run(run(go))
    ci = seen['ci']
    assert ci['quantity'] == o.quantity == 1500.0 and abs(ci['entry_price'] - o.entry_price) < 1e-12 and abs(ci['entry_fee'] - o.entry_fee) < 1e-12
    assert ci['peak_pnl'] == o.peak_pnl and ci['trough_pnl'] == o.trough_pnl and ci['peak_pnl'] != 2.4   # re-derived on the new entry


def test_i1_unknown_outcome_late_fill_is_flattened_and_adds_blocked(monkeypatch, tmp_path):
    pre = (("LONG", 1000.0, 2.0),)
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, order_result=None, positions_seq=[pre, pre, pre, pre, pre, pre, (("LONG", 1500.0, 1.9667),)])
    async def go(eng, db, oid):
        try:
            await eng.add_to_position(db, oid, 50.0)
        except RuntimeError as e:
            msg = str(e)
        task = eng.__dict__['_add_unresolved'][oid]
        await task
        return msg, oid in eng.__dict__['_add_unresolved']
    msg, still = asyncio.run(run(go))
    assert "CHECK BINANCE NOW" in msg and "re-checks" in msg and not still
    assert ("flatten", "LONG", 500.0) in log                                          # the late fill of the add — exactly its delta
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, order_result=None, positions_seq=[pre, pre, pre, pre, pre, pre, (("LONG", 1800.0, 1.9),)])
    async def go2(eng, db, oid):
        try:
            await eng.add_to_position(db, oid, 50.0)
        except RuntimeError:
            pass
        await eng.__dict__['_add_unresolved'][oid]
    asyncio.run(run(go2))
    assert "flatten" not in _kinds(log)                                               # +800 is not the add's +500: alert only, never touched
    run, log, T, read, _ = _harness(monkeypatch, tmp_path)
    async def blocked(eng, db, oid):
        eng.__dict__['_add_unresolved'] = {oid: object()}
        return await eng.add_to_position(db, oid, 50.0)
    r = asyncio.run(run(blocked))
    assert "still being checked on Binance" in r and "order" not in _kinds(log)


def test_i2_tp_partial_fill_during_the_withdraw_refuses(monkeypatch, tmp_path):
    async def _partial(S, oid, n):
        import models
        from sqlalchemy import update
        async with S() as s2:
            await s2.execute(update(models.Order).where(models.Order.id == oid).values(resting_tp_filled_qty=300.0)); await s2.commit()
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_withdraw=_partial)
    r = asyncio.run(run(_add(50.0)))
    assert "filled in part (300 of 1000)" in r and "software take profit" in r
    assert _kinds(log) == ["positions", "tp_withdraw"]                                # no order, and the TP is NOT re-armed by the add


def test_i3_a_tp_healed_by_a_sweep_is_withdrawn_again_before_the_order(monkeypatch, tmp_path):
    async def _reappear(S, oid, n, times=(1,)):
        import models
        from sqlalchemy import update
        if n in times:
            async with S() as s2:
                await s2.execute(update(models.Order).where(models.Order.id == oid).values(protective_tp_order_id="tp-2")); await s2.commit()
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_withdraw=_reappear)
    o = asyncio.run(run(_add(50.0)))
    assert not isinstance(o, str), o
    k = _kinds(log)
    assert k[:5] == ["positions", "tp_withdraw", "tp_withdraw", "positions", "order"]
    async def _always(S, oid, n):
        await _reappear(S, oid, n, times=(1, 2))
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_withdraw=_always)
    r = asyncio.run(run(_add(50.0)))
    assert "could not be cleared" in r and "order" not in _kinds(log)


def test_i3_sweep_heal_is_skipped_during_an_add():
    import services.trading_engine as T
    eng = T.TradingEngine.__new__(T.TradingEngine); eng.is_paper_mode = False
    eng.__dict__['_add_hold'] = {7}
    class _NoDB:
        def __call__(self, *a, **k):
            raise AssertionError("the skipped heal must not open a session")
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    try:
        mp.setattr(T, "AsyncSessionLocal", _NoDB())
        assert asyncio.run(T.TradingEngine._rtp_heal(eng, 7, from_sweep=True)) == 'skipped'
    finally:
        mp.undo()
    import inspect
    assert "await self._rtp_heal(oid, from_sweep=True)" in inspect.getsource(T.TradingEngine.resting_tp_sweep)


def test_i4_backstop_record_refused_while_closing_cancels_the_new_stop(monkeypatch, tmp_path):
    async def _closing(S, log):
        import models
        from sqlalchemy import update
        async with S() as s2:
            await s2.execute(update(models.Order).values(closing_in_progress=True)); await s2.commit()
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_replace=_closing)
    o = asyncio.run(run(_add(50.0)))
    assert ("cancel_backstop", "algo-new") in log and o.backstop_algo_id == "algo-old"


def test_i5_leverage_mismatch_on_binance_refuses(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, positions=(("LONG", 1000.0, 2.0, 10),))
    r = asyncio.run(run(_add(50.0)))
    assert "the leverage on Binance is 10×, this trade is 20× — not added" in r and _kinds(log) == ["positions"]
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, positions=(("LONG", 1000.0, 2.0, 1, 2000.0, 100.0),))   # no reported lev → notional / margin = 20
    assert not isinstance(asyncio.run(run(_add(50.0))), str)
    from services.position_add import binance_position_leverage
    assert binance_position_leverage({"leverage": 1}) is None and binance_position_leverage({"leverage": 25}) == 25.0


def test_m1_ack_without_price_or_fill_uses_the_position_delta(monkeypatch, tmp_path):
    avg = (2.0 * 1000 + 1.95 * 400) / 1400
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, ack={"id": "A-9", "price": 0, "filled": 0},
                                    positions_seq=[(("LONG", 1000.0, 2.0),), (("LONG", 1000.0, 2.0),), (("LONG", 1400.0, avg),)])
    o = asyncio.run(run(_add(50.0)))
    assert abs(o.quantity - 1400) < 1e-9 and abs(o.entry_price - avg) < 1e-12 and abs(o.manual_add_investment - 400 * 1.95 / 20) < 1e-9


def test_m4_preview_reuses_a_fresh_balance_read(monkeypatch, tmp_path):
    run, log, T, read, eng = _harness(monkeypatch, tmp_path)
    async def go(eng, db, oid):
        await eng.manual_add_preview(db, oid, 10.0); await eng.manual_add_preview(db, oid, 20.0)
        n1 = sum(1 for x in log if x[0] == "balance")
        await eng.add_to_position(db, oid, 20.0)                                       # the add itself always reads fresh
        return n1, sum(1 for x in log if x[0] == "balance")
    n1, n2 = asyncio.run(run(go))
    assert n1 == 1 and n2 == 2


# ── round-2 review fixes ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_r1_cache_rebuild_never_uses_a_stale_identity_mapped_row(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path)
    async def go(eng, db, oid):
        from sqlalchemy import select, update
        import models
        stale = (await db.execute(select(models.Order).where(models.Order.id == oid))).scalar_one()   # the monitor pass's object
        assert stale.quantity == 1000.0
        from sqlalchemy.ext.asyncio import async_sessionmaker
        async with T.AsyncSessionLocal() as other:   # the add commits in its own session
            await other.execute(update(models.Order).where(models.Order.id == oid).values(quantity=1500.0, entry_price=1.9667)); await other.commit()
        await T.TradingEngine.update_orders_cache(eng, db)                             # the real rebuild, SAME session as the stale object
        return T._open_orders_cache["FOOUSDT"][0]
    ci = asyncio.run(run(go))
    assert ci['quantity'] == 1500.0 and ci['entry_price'] == 1.9667


def test_r2_late_add_recheck_never_flattens_when_a_new_row_exists(monkeypatch, tmp_path):
    pre = (("LONG", 1000.0, 2.0),)
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, order_result=None, positions_seq=[pre, pre, pre, pre, pre, pre, (("LONG", 1500.0, 1.9667),)])
    monkeypatch.setattr(T, "ADD_RECHECK_POLL_S", 0.2)                                  # the new row is booked before the first late look
    async def go(eng, db, oid):
        import models
        try:
            await eng.add_to_position(db, oid, 50.0)
        except RuntimeError:
            pass
        task = eng.__dict__['_add_unresolved'][oid]
        async with T.AsyncSessionLocal() as s2:   # a NEW position on the pair is opened (and booked) inside the window
            s2.add(models.Order(pair="FOOUSDT", direction="LONG", status="OPEN", entry_price=2.0, quantity=500.0, investment=50.0, leverage=20.0,
                                notional_value=1000.0, confidence="STRONG_BUY", entry_fee=0.45, is_paper=False, entry_strategy="MANUAL",
                                opened_at=dt.datetime.utcnow()))
            await s2.commit()
        await task
    asyncio.run(run(go))
    assert "flatten" not in _kinds(log) and _kinds(log).count("positions") >= 7       # the late look happened, and touched nothing
    import inspect
    src = inspect.getsource(T.TradingEngine._add_unknown_recheck)
    assert src.index("_guard.enter_async_context(self._pair_open_guard(pair, 30.0))") < src.index("_r = await self._add_pos_read(symbol)")


def test_r4_paper_add_books_under_the_close_lock(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, paper=True)
    async def go(eng, db, oid):
        from sqlalchemy import select
        import models
        monkeypatch.setattr(T, "_close_lock", asyncio.Lock())                         # a lock of THIS loop
        await T._close_lock.acquire()                                                  # a paper close is in flight
        task = asyncio.ensure_future(eng.add_to_position(db, oid, 50.0))
        for _ in range(50):
            await asyncio.sleep(0)
        async with T.AsyncSessionLocal() as s2:
            q_mid = (await s2.execute(select(models.Order.quantity).where(models.Order.id == oid))).scalar()
        done_mid = task.done()
        T._close_lock.release()
        o = await task
        return q_mid, done_mid, o.quantity
    q_mid, done_mid, q_end = asyncio.run(run(go))
    assert q_mid == 1000.0 and not done_mid and abs(q_end - 1500.0) < 1e-9             # nothing booked while the close held the lock
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, paper=True)
    monkeypatch.setattr(T, "ADD_PAPER_CLOSE_LOCK_WAIT_S", 0.05)
    async def held(eng, db, oid):
        monkeypatch.setattr(T, "_close_lock", asyncio.Lock())
        await T._close_lock.acquire()
        try:
            return await eng.add_to_position(db, oid, 50.0)
        finally:
            T._close_lock.release()
    r = asyncio.run(run(held))
    assert "a close is being booked right now" in r


def test_r5_failed_size_read_rolls_back_and_one_round_trip():
    import services.trading_engine as T, inspect
    calls = []
    class DB:
        async def execute(self, *a, **k):
            raise RuntimeError("boom")
        async def rollback(self):
            calls.append("rollback")
    class O:
        id, pair = 1, "FOOUSDT"
    eng = T.TradingEngine.__new__(T.TradingEngine)
    assert asyncio.run(eng._close_size_changed(DB(), O(), 1.0, 0, "X")) is False and calls == ["rollback"]
    src = inspect.getsource(T.TradingEngine._close_size_changed)
    assert "select(Order.quantity, Order.manual_add_count, Order.backstop_algo_id)" in src and "set_committed_value(order, 'backstop_algo_id', row[2])" in src


def test_r6_a_failure_after_booking_returns_success_with_a_warning(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, replace="lost")
    async def go(eng, db, oid):
        def _boom(*a, **k):
            raise RuntimeError("counter down")
        eng._record_filter_block = _boom                                               # raises inside the post-booking safety-stop step
        o = await eng.add_to_position(db, oid, 50.0)
        return o, oid in eng.__dict__.get('_add_hold', set()), oid in eng.__dict__.get('_rtp_heal_next', {}), oid in eng.__dict__.get('_bk_replace_next', {})
    o, hold, rn, bn = asyncio.run(run(go))
    assert not isinstance(o, str) and o.quantity == 1500.0 and "the add is booked" in o._add_warning
    assert not hold and not rn and not bn                                              # markers cleared in the finally
    src = open(os.path.join(ROOT, "main.py")).read()
    assert "\"warning\": getattr(order, '_add_warning', None)" in src


def test_r7_r8_r9_wiring_and_exit_edit_refused_during_an_add(monkeypatch, tmp_path):
    import inspect, services.trading_engine as T
    rp = inspect.getsource(T.TradingEngine.backstop_reprice_sweep)
    assert rp.index("if oid in (self.__dict__.get('_add_hold') or ()):") < rp.index("await self._backstop_replace_one(pair, direction, old_id, want, trig)")
    mon = inspect.getsource(T.TradingEngine)
    assert "if _bko.id in (self.__dict__.get('_add_hold') or ()):" in mon
    cl = inspect.getsource(T.TradingEngine._close_position_locked)
    i9 = cl.index("if not _close_marked and exit_result is None and await self._close_size_changed(")
    assert i9 < cl.index("result = await binance_service.close_position(")
    run, log, T, read, _ = _harness(monkeypatch, tmp_path)
    async def go(eng, db, oid):
        eng.__dict__['_add_hold'] = {oid}
        return await eng.set_exit_override(db, oid, sl_pct=-1.0)
    assert "an add is in progress on this trade" in asyncio.run(run(go))


# ── round-3 review fixes ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_r3_1_paper_add_never_deadlocks_with_a_close_taking_the_book_lock(monkeypatch, tmp_path):
    """a paper close holds _close_lock and then needs the book lock (BNB emergency top-up) while the add withdraws its TP (which takes
    _close_lock): the add must not be holding the book lock then."""
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, paper=True)
    async def go(eng, db, oid):
        monkeypatch.setattr(T, "_close_lock", asyncio.Lock())                         # a lock of THIS loop (another test may have bound the global)
        del eng.__dict__['_rtp_withdraw']                                             # the REAL withdraw: it takes _close_lock (no timeout)
        real_wd = eng._rtp_withdraw
        holding, in_withdraw = asyncio.Event(), asyncio.Event()
        async def wd(order_id, why):
            in_withdraw.set()                                                          # the add is now waiting on _close_lock
            return await real_wd(order_id, why)
        eng._rtp_withdraw = wd
        async def close_like():
            async with T._close_lock:
                holding.set()
                await in_withdraw.wait()
                rel = await eng._book_hold()                                           # what _execute_bnb_swap → _book_ctx does
                rel()
        c = asyncio.ensure_future(close_like())
        await holding.wait()
        o = await asyncio.wait_for(eng.add_to_position(db, oid, 50.0), 3.0)            # a deadlock would time out here
        await asyncio.wait_for(c, 3.0)
        return o
    o = asyncio.run(run(go))
    assert not isinstance(o, str), o
    assert abs(o.quantity - 1500.0) < 1e-9
    import inspect
    src = inspect.getsource(T.TradingEngine._add_to_position_locked)
    assert src.index("_book_release = await self._book_hold(10.0)") > src.index('await _withdraw_tp("add to position (re-check)")')


def test_r3_2_close_never_nulls_a_stop_id_an_add_re_placed(monkeypatch):
    """R9 race: the close-intent flag failed, an add commits between the size check and the pre-close backstop cancel (new size + new stop
    id Y): the close cancels its old id X and must NOT write NULL over Y; the R9 re-check then aborts the stale close."""
    from sqlalchemy import update
    import models
    from tests.test_resting_tp import _setup
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, close_px=99.0, rows=[dict(
        backstop_algo_id="X", entry_strategy="MANUAL", manual_exit_mode="FIXED", pattern_fixed_sl_pct=-1.5)])
    async def _no_mark(db, oid):
        return False
    eng._mark_close_in_progress = _no_mark
    async def _bk_cancel(pair, aid):
        log.append(("bk_cancel", pair, aid))
        async with S() as b:   # the add commits right now: new size + its re-placed stop
            await b.execute(update(models.Order).where(models.Order.id == 1).values(quantity=15.0, manual_add_count=1, backstop_algo_id="Y"))
            await b.commit()
        return True
    monkeypatch.setattr(T.binance_service, "cancel_backstop_stop", _bk_cancel)
    async def go():
        from sqlalchemy import select
        await seed()
        async with S() as a:
            o = (await a.execute(select(models.Order).where(models.Order.id == 1))).scalar_one()
            r = await eng._close_position_locked(a, o, 99.0, "MANUAL_SL")
        rows = await read(); await engine.dispose(); return r, rows
    r, rows = asyncio.run(go())
    assert r is None and rows[1].status == "OPEN" and rows[1].backstop_algo_id == "Y" and rows[1].quantity == 15.0
    assert not [x for x in log if x[0] == "market_close"]


def test_r3_3_recheck_skips_a_look_when_the_pair_lock_is_busy(monkeypatch, tmp_path):
    import contextlib as _cl
    pre = (("LONG", 1000.0, 2.0),)
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, order_result=None, positions_seq=[pre, pre, pre, pre, pre, pre, (("LONG", 1500.0, 1.9667),)])
    monkeypatch.setattr(T, "ADD_RECHECK_POLL_S", 0.1)                                  # the busy guard is installed before the first look
    async def go(eng, db, oid):
        try:
            await eng.add_to_position(db, oid, 50.0)
        except RuntimeError:
            pass
        task = eng.__dict__['_add_unresolved'][oid]
        real = eng._pair_open_guard
        n = {"c": 0}
        @_cl.asynccontextmanager
        async def busy_once(pair, wait_s=None):
            n["c"] += 1
            if n["c"] == 1:
                raise ValueError("FOOUSDT is being opened right now")
            async with real(pair, wait_s):
                yield
        eng._pair_open_guard = busy_once
        await task
        return n["c"]
    c = asyncio.run(run(go))
    assert c >= 2 and ("flatten", "LONG", 500.0) in log                               # the busy look was skipped, the next one flattened


def test_r3_4_cancel_after_booking_is_logged_reraised_and_cleans_up(monkeypatch, tmp_path):
    async def _cx(S, log):
        raise asyncio.CancelledError()
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_replace=_cx)
    async def go(eng, db, oid):
        from sqlalchemy import select
        import models
        try:
            await eng.add_to_position(db, oid, 50.0)
            raised = False
        except asyncio.CancelledError:
            raised = True
        async with T.AsyncSessionLocal() as s2:
            q = (await s2.execute(select(models.Order.quantity).where(models.Order.id == oid))).scalar()
        return raised, q, oid in eng.__dict__.get('_add_hold', set()), oid in eng.__dict__.get('_rtp_heal_next', {})
    raised, q, hold, rn = asyncio.run(run(go))
    assert raised and q == 1500.0 and not hold and not rn                             # booked, re-raised, markers cleared


# ── round-4 review fixes ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_r4_1_position_is_read_again_under_the_book_lock(monkeypatch, tmp_path):
    pre = (("LONG", 1000.0, 2.0),)
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, positions_seq=[pre, (("LONG", 1300.0, 2.0),)])   # app qty netted in after the first read
    r = asyncio.run(run(_add(50.0)))
    assert "Binance holds LONG 1300 FOOUSDT but this trade is LONG 1000" in r
    assert _kinds(log) == ["positions", "tp_withdraw", "positions", "tp_heal"]       # nothing sent; the unchanged row gets its TP back
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, positions_seq=[pre, (("LONG", 1000.0, 2.0, 10),)])   # leverage changed meanwhile
    assert "the leverage on Binance is 10×" in asyncio.run(run(_add(50.0))) and "order" not in _kinds(log)
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, positions_seq=[pre, None])                       # the second read fails
    assert "the order was NOT sent" in asyncio.run(run(_add(50.0))) and "order" not in _kinds(log)
    # the unknown-outcome delta uses the FRESH size (1004, within tolerance), never the first read's 1000
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, order_result=None,
                                    positions_seq=[pre, (("LONG", 1004.0, 2.0),), (("LONG", 1504.0, 1.97),)])
    o = asyncio.run(run(_add(50.0)))
    assert not isinstance(o, str), o
    assert abs(o.quantity - 1500.0) < 1e-9                                             # +500 (1504 − 1004), not +504


def test_r4_2_no_value_originals_are_never_written():
    import datetime as _dt
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    import models, services.trading_engine as T
    async def go():
        e = create_async_engine("sqlite+aiosqlite://")
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(e, expire_on_commit=False)
        async with S() as s:
            s.add(models.Order(pair="A", direction="LONG", status="OPEN", entry_price=1, quantity=1, investment=1, leverage=1, notional_value=1,
                               confidence="X", entry_fee=0, is_paper=True, opened_at=_dt.datetime(2026, 10, 20), shadow_peak_stretch_at_close=0.7))
            await s.commit()
            o = (await s.execute(select(models.Order))).scalar_one()
            o.shadow_peak_stretch_at_close = 1.5
            a = T.pending_column_originals(o)
            s.expire(o, ['quantity']); o.quantity = 3.0                                 # expired then set: the DB value is unknown here
            b = T.pending_column_originals(o)
        await e.dispose()
        return a, b
    a, b = asyncio.run(go())
    assert a == {'shadow_peak_stretch_at_close': 0.7} and 'quantity' not in b and b['shadow_peak_stretch_at_close'] == 0.7


def test_r4_3_error_after_a_successful_commit_is_never_flattened(monkeypatch, tmp_path):
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, commit_then_raise=RuntimeError("socket closed after COMMIT"))
    async def go(eng, db, oid):
        from sqlalchemy import select
        import models
        try:
            await eng.add_to_position(db, oid, 50.0)
            msg = "no error"
        except RuntimeError as e:
            msg = str(e)
        async with T.AsyncSessionLocal() as s2:
            q = (await s2.execute(select(models.Order.quantity).where(models.Order.id == oid))).scalar()
        return msg, q, oid in eng.__dict__.get('_add_hold', set())
    msg, q, hold = asyncio.run(run(go))
    assert "the add was recorded" in msg and "do NOT add again" in msg and q == 1500.0 and not hold
    assert "flatten" not in _kinds(log)
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, commit_then_raise=asyncio.CancelledError())
    async def go2(eng, db, oid):
        try:
            await eng.add_to_position(db, oid, 50.0)
        except asyncio.CancelledError:
            return "cancelled"
    assert asyncio.run(run(go2)) == "cancelled" and "flatten" not in _kinds(log)


def test_r4_4_unrecorded_stop_that_cannot_be_cancelled_is_critical(monkeypatch, tmp_path, caplog):
    import logging
    async def _closing(S, log):
        import models
        from sqlalchemy import update
        async with S() as s2:
            await s2.execute(update(models.Order).values(closing_in_progress=True)); await s2.commit()
    run, log, T, read, _ = _harness(monkeypatch, tmp_path, on_replace=_closing, bk_cancel_ok=False)
    with caplog.at_level(logging.CRITICAL):
        o = asyncio.run(run(_add(50.0)))
    assert ("block", "BACKSTOP_ORPHAN_UNCANCELED") in log and "[BACKSTOP_ORPHAN_UNCANCELED]" in caplog.text


def test_r4_5_close_cancels_a_newer_stop_id_when_it_continues(monkeypatch):
    """the re-price sweep moved the stop X → Y during the close's cancel (no size change): the close continues, cancels Y too, and closes."""
    from sqlalchemy import update
    import models
    from tests.test_resting_tp import _setup
    T, eng, engine, S, seed, read, log, blocks = _setup(monkeypatch, close_px=99.0, rows=[dict(
        backstop_algo_id="X", entry_strategy="MANUAL", manual_exit_mode="FIXED", pattern_fixed_sl_pct=-1.5)])
    async def _bk_cancel(pair, aid):
        log.append(("bk_cancel", pair, aid))
        if aid == "X":
            async with S() as b:
                await b.execute(update(models.Order).where(models.Order.id == 1).values(backstop_algo_id="Y")); await b.commit()
        return True
    monkeypatch.setattr(T.binance_service, "cancel_backstop_stop", _bk_cancel)
    async def go():
        from sqlalchemy import select
        await seed()
        async with S() as a:
            o = (await a.execute(select(models.Order).where(models.Order.id == 1))).scalar_one()
            r = await eng._close_position_locked(a, o, 99.0, "MANUAL_SL")
        rows = await read(); await engine.dispose(); return r, rows
    r, rows = asyncio.run(go())
    k = [x for x in log if x[0] in ("bk_cancel", "market_close")]
    assert k == [("bk_cancel", "WILUSDT", "X"), ("bk_cancel", "WILUSDT", "Y"), ("market_close", "WIL/USDT:USDT", "LONG", 10.0)]
    assert r is not None and rows[1].status == "CLOSED" and rows[1].backstop_algo_id is None
