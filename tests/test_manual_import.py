"""📥 Import an app-opened Binance position (DECISION_LOG 328). Pins every refusal (nothing written, nothing left on the exchange), the LIVE
order of operations (positions read → open-orders list → safety stop → positions RE-READ → commit → resting TP → cache), the row's fields,
the untracked state cleared, the candidates list (reconciler state + snapshot, no exchange read), the shared 🖐 exit checks, the pure
helpers and the UI wiring (banner, pop-up, ⇄ % / $ reuse, node --check). No network: every exchange call is a monkeypatched stub."""
import asyncio, datetime as dt, itertools, json, os, re, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import pytest

from services.position_import import (import_pick_position, import_whole_leverage, import_position_unchanged, import_level_refusal,
                                      IMPORT_SOURCE_BINANCE_APP)
from services.position_add import pnl_pct_net

TF = 0.00045
SYM = "FOO/USDT:USDT"
_DB_SEQ = itertools.count()
UI = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()


def _pos(side="LONG", contracts=1000.0, entry=2.0, lev=20, notional=None, margin=None, mark=2.0, upnl=0.0, symbol=SYM):
    n = contracts * entry if notional is None else notional
    return dict(symbol=symbol, side=side, contracts=contracts, entry_price=entry, mark_price=mark, unrealized_pnl=upnl, leverage=lev,
                notional=n, margin=(n / lev if margin is None and lev and lev > 1 else (margin or 0.0)))


# ── pure helpers ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_pick_position_refusals():
    p = import_pick_position([_pos(), _pos(symbol="BAR/USDT:USDT")], SYM, "FOOUSDT", "LONG", 5.0)
    assert p["contracts"] == 1000.0
    for positions, direction, txt in (([], "LONG", "shows no FOOUSDT position"), ([_pos(), _pos()], "LONG", "more than one"),
                                      ([_pos(side="SHORT")], "LONG", "holds a SHORT FOOUSDT position, not LONG"),
                                      ([_pos(contracts=0.0)], "LONG", "no size"), ([_pos(entry=0.0, notional=1.0)], "LONG", "no entry price"),
                                      ([_pos(contracts=2.0)], "LONG", "dust")):
        with pytest.raises(ValueError, match=re.escape(txt)):
            import_pick_position(positions, SYM, "FOOUSDT", direction, 5.0)


def test_whole_leverage_unchanged_and_levels():
    assert import_whole_leverage(20) == 20 and import_whole_leverage(19.99999) == 20 and import_whole_leverage(1.0) == 1
    assert import_whole_leverage(19.5) is None and import_whole_leverage(None) is None and import_whole_leverage(0.5) is None
    a = _pos()
    assert import_position_unchanged(a, _pos(contracts=1004.0))                                 # within 0.5 %
    assert not import_position_unchanged(a, _pos(contracts=900.0)) and not import_position_unchanged(a, _pos(side="SHORT"))
    assert not import_position_unchanged(a, _pos(entry=2.02)) and import_position_unchanged(a, _pos(entry=2.009))
    assert not import_position_unchanged(a, None) and import_position_unchanged(_pos(contracts=1.0), _pos(contracts=1.0001), step=0.001)
    assert import_level_refusal(-0.09, -1.5, 3.0) is None and "would close it at once" in import_level_refusal(-1.6, -1.5, 3.0)
    assert "already reached" in import_level_refusal(3.2, -1.5, 3.0) and import_level_refusal(9.0, None, None) is None
    assert "not available" in import_level_refusal(None, -1.5, None)


def test_shared_exit_checks_and_max_stop():
    import services.trading_engine as T
    th = T.config.trading_config.thresholds

    class TH:
        manual_floor_sl_pct = -3.0; broker_backstop_enabled = True; broker_backstop_pct = 2.5; frenzy_stop_pct = 3.0
        sl_atr_widen_floor_pct = -1.2; fl1_wide_sl_backstop = -1.2; bullrun_base_sl_pct = -0.7; sl_atr_multiplier = 0.0
    assert T.manual_exit_levels(TH, "FIXED", 1.5, 3.0, 20) == (-1.5, 3.0)
    assert T.manual_exit_levels(TH, "FLOOR", None, None, 20) == (-3.0, None)
    assert T.manual_exit_levels(TH, "FRENZY", 9, 9, 20) == (None, None)                       # its stop is read live; no TP stored
    with pytest.raises(ValueError, match="needs a stop loss"):
        T.manual_exit_levels(TH, "FIXED", None, None, 20)
    with pytest.raises(ValueError, match="wider than the widest stop allowed at 20×"):
        T.manual_exit_levels(TH, "FIXED", 3.5, None, 20)
    with pytest.raises(ValueError, match="too close to this trade's own stop"):                # 50×: safety stop ≤ 1.38 %, stop 1.2 + 0.3 gap
        T.manual_exit_levels(TH, "FIXED", 1.2, None, 50)
    assert T.manual_max_stop_pct(TH, 20) == 3.0 and abs(T.manual_max_stop_pct(TH, 50) - (1.5 * 0.92 - 0.3)) < 1e-9
    T.manual_exit_levels(TH, "FIXED", T.manual_max_stop_pct(TH, 50), None, 50)                # the shown limit is accepted …
    with pytest.raises(ValueError):
        T.manual_exit_levels(TH, "FIXED", T.manual_max_stop_pct(TH, 50) + 0.01, None, 50)     # … and nothing wider


# ── engine harness ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _harness(monkeypatch, tmp_path, *, paper=False, positions_seq=(None,), open_orders=None, bk_id="algo-1", bk_cancel_ok=True, price=2.0,
             commit_fails=False, rows=(), cap=8, untracked=True, open_orders_seq=None, place_delay=0.0):
    """The real import_binance_position on a file SQLite DB with a faked exchange. positions_seq[i] = the i-th positions read (a list,
    or None = a failed read); default = one LONG 1000 @ 2.0 at 20×. log = every exchange / engine step in order."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, services.trading_engine as T
    log = []
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = paper; eng.paper_balance = 0.0
    eng._record_filter_block = lambda name, d, had_room=True: log.append(("block", name))

    async def _cache(db): log.append(("cache",))
    async def _arm(db, order): log.append(("rtp_arm", order.id, order.pattern_fixed_tp_pct)); return None
    async def _stamps(pair, symbol, direction, fill_price): log.append(("stamps", fill_price)); return {"entry_atr_pct": 0.8, "entry_rsi": 55.0}
    eng.update_orders_cache = _cache; eng._rtp_arm = _arm; eng._manual_entry_stamps = _stamps
    if untracked:
        eng._untracked_positions = {"FOOUSDT": {"pair": "FOOUSDT", "side": "LONG", "qty": 1000.0, "entry_price": 2.0, "reason": "NO_ROW"}}
        eng._untracked_seen = {"FOOUSDT": 1}
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "broker_backstop_pct", 2.5, raising=False)
    monkeypatch.setattr(th, "manual_floor_sl_pct", -3.0, raising=False)
    monkeypatch.setattr(T.config.trading_config, "taker_fee", TF, raising=False)
    monkeypatch.setattr(T.config.trading_config.investment, "manual_max_open_positions", cap, raising=False)
    B = T.binance_service
    monkeypatch.setattr(B, "_markets_loaded", False, raising=False)
    state = {"pos": 0}
    seq = [([_pos()] if x is None else x) for x in positions_seq] if positions_seq != (None,) else [[_pos()]]

    def _boom(name):
        async def f(*a, **k):
            raise AssertionError(f"must never call the exchange here ({name})")
        return f
    async def _positions():
        state["pos"] += 1; log.append(("positions", state["pos"]))
        src = seq[min(state["pos"] - 1, len(seq) - 1)]
        return None if src == "FAIL" else [dict(p) for p in src]
    async def _oo(pair):   # 📥 (343) the symbol's open orders as list_symbol_open_orders returns them ([] = none, "FAIL" = unreadable)
        log.append(("open_orders", pair))
        n = sum(1 for x in log if x[0] == "open_orders")
        src = open_orders_seq[min(n - 1, len(open_orders_seq) - 1)] if open_orders_seq else open_orders
        return None if src == "FAIL" else [dict(o) for o in (src or [])]
    async def _place(pair, direction, trig):
        log.append(("backstop", pair, direction, round(trig, 8)))
        if place_delay:
            await asyncio.sleep(place_delay)
            log.append(("backstop_done", bk_id))
        return bk_id
    async def _cancel(pair, algo_id): log.append(("cancel_backstop", algo_id)); return bk_cancel_ok
    async def _cancel_reg(pair, oid): log.append(("cancel_regular", oid)); return {"result": "canceled", "filled": 0.0}
    for name, fn in (("get_open_positions", _positions), ("list_symbol_open_orders", _oo), ("place_backstop_stop", _place),
                     ("cancel_backstop_stop", _cancel), ("cancel_resting_tp", _cancel_reg)):
        monkeypatch.setattr(B, name, fn if not paper else _boom(name))
    for name in ("create_market_order", "close_position", "get_current_price", "cancel_all_backstops"):
        monkeypatch.setattr(B, name, _boom(name))

    class _Trk:
        last_price = price
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: _Trk())
    monkeypatch.setattr(T.websocket_tracker, "pair_silence_seconds", lambda p: 1.0)
    monkeypatch.setattr(T.websocket_tracker, "force_reset_tracking", lambda p, px=None: log.append(("ws_reset", p, px)))
    monkeypatch.setattr(T, "ws_subscribe_background", lambda p: log.append(("ws_sub", p)))
    _real_commit = T.locked_commit
    async def _commit(db):
        if commit_fails:
            raise RuntimeError("database is locked")
        log.append(("commit",))
        return await _real_commit(db)
    monkeypatch.setattr(T, "locked_commit", _commit)
    eng_db = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'imp_{next(_DB_SEQ)}.db'}")
    S = async_sessionmaker(eng_db, expire_on_commit=False)
    monkeypatch.setattr(T, "AsyncSessionLocal", S)

    async def run(fn):
        async with eng_db.begin() as c:
            await c.run_sync(models.Base.metadata.drop_all); await c.run_sync(models.Base.metadata.create_all)
        async with S() as db:
            for r in rows:
                db.add(models.Order(**dict(dict(direction="LONG", status="OPEN", entry_price=1.0, quantity=10.0, investment=1.0, leverage=10.0,
                                                notional_value=10.0, confidence="STRONG_BUY", is_paper=paper), **r)))
            await db.commit()
        try:
            async with S() as db:
                try:
                    return await fn(eng, db)
                except (ValueError, RuntimeError) as err:
                    return f"{type(err).__name__}: {err}"
        finally:
            await eng_db.dispose()

    async def read():
        from sqlalchemy import select
        async with S() as db:
            o = (await db.execute(select(models.Order).where(models.Order.pair == "FOOUSDT").order_by(models.Order.id.desc()))).scalars().first()
            tx = (await db.execute(select(models.Transaction))).scalars().all()
            return o, tx
    return run, log, T, read, eng, S, eng_db


def _imp(direction="LONG", exit_mode="FIXED", sl=1.5, tp=3.0, pair="FOOUSDT"):
    async def fn(eng, db):
        return await eng.import_binance_position(db, pair, direction, exit_mode, sl, tp, app_orders_seen=[])   # (343) what the pop-up showed
    return fn


def _kinds(log):
    return [x[0] for x in log if x[0] not in ("block", "stamps")]


def _nothing_written(run_read):
    o, tx = asyncio.run(run_read())
    return o is None and tx == []


def test_happy_path_order_of_operations_and_row(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path)
    o = asyncio.run(run(_imp()))
    assert not isinstance(o, str), o
    assert _kinds(log) == ["positions", "open_orders", "backstop", "positions", "commit", "rtp_arm", "cache", "ws_reset", "ws_sub"]
    o, tx = asyncio.run(read())
    assert o.status == "OPEN" and o.direction == "LONG" and o.entry_strategy == "MANUAL" and o.confidence == "STRONG_BUY"
    assert o.entry_price == 2.0 and o.quantity == 1000.0 and o.leverage == 20.0 and abs(o.notional_value - 2000.0) < 1e-9
    assert abs(o.investment - 100.0) < 1e-9 and abs(o.entry_fee - 2000.0 * TF) < 1e-12                 # notional / lev · taker estimate
    assert o.manual_exit_mode == "FIXED" and o.pattern_fixed_sl_pct == -1.5 and o.pattern_fixed_tp_pct == 3.0 and o.dynamic_tp_target == 3.0
    assert o.import_source == IMPORT_SOURCE_BINANCE_APP == "BINANCE_APP" and o.imported_at is not None and o.opened_at == o.imported_at
    assert abs((dt.datetime.utcnow() - o.imported_at).total_seconds()) < 60 and o.binance_order_id is None
    assert o.backstop_algo_id == "algo-1" and o.backstop_pct == 2.5 and abs(o.backstop_trigger_price - 2.0 * 0.975) < 1e-12
    assert [x for x in log if x[0] == "backstop"][0] == ("backstop", "FOOUSDT", "LONG", round(2.0 * 0.975, 8))
    assert o.entry_rsi == 55.0 and o.entry_atr_pct == 0.8                                                # the 🖐 entry stamps, at the import
    assert len(tx) == 1 and tx[0].action == "OPEN_LONG" and tx[0].quantity == 1000.0 and abs(tx[0].fee - 2000.0 * TF) < 1e-12
    assert "FOOUSDT" not in eng._untracked_positions and "FOOUSDT" not in eng._untracked_seen         # entries unblock at once
    assert ("block", "MANUAL_IMPORT") in log and ("rtp_arm", o.id, 3.0) in log


def test_short_floor_mode_and_import_is_a_manual_origin(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_pos(side="SHORT", contracts=500.0, entry=4.0, lev=10, mark=4.0)],))
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: type("K", (), {"last_price": 4.0})())
    o = asyncio.run(run(_imp(direction="SHORT", exit_mode="FLOOR", sl=None, tp=None)))
    assert not isinstance(o, str), o
    o, _ = asyncio.run(read())
    assert o.direction == "SHORT" and o.manual_exit_mode == "FLOOR" and o.pattern_fixed_sl_pct == -3.0 and o.pattern_fixed_tp_pct is None
    assert o.backstop_pct == 3.5 and abs(o.backstop_trigger_price - 4.0 * 1.035) < 1e-12 and o.investment == 200.0
    from services.origin_view import order_origin
    assert order_origin(o.entry_strategy, o.confidence) == "MANUAL" and order_origin(None, "RECOVERED") == "MANUAL"


@pytest.mark.parametrize("kw,call,expect", [
    (dict(positions_seq=("FAIL",)), _imp(), "could not read Binance — nothing imported"),
    (dict(positions_seq=([_pos(side="SHORT")],)), _imp(), "holds a SHORT FOOUSDT position, not LONG"),
    (dict(positions_seq=([],)), _imp(), "shows no FOOUSDT position"),
    (dict(positions_seq=([_pos(contracts=2.0)],)), _imp(), "dust, not a position"),
    (dict(positions_seq=([_pos(lev=1, notional=0.0, margin=0.0)],)), _imp(), "did not report this position's leverage"),
    (dict(positions_seq=([_pos(lev=1, notional=2000.0, margin=2000.0 / 19.5)],)), _imp(), "leverage reads 19.5× — not a whole number"),
    (dict(), _imp(sl=3.5), "wider than the widest stop allowed at 20×"),
    (dict(positions_seq=([_pos(lev=50)],)), _imp(sl=1.2), "too close to this trade's own stop"),
    (dict(), _imp(sl=None), "mode needs a stop loss"),
    (dict(price=1.97), _imp(), "a stop at -1.50% would close it at once"),
    (dict(price=2.07), _imp(), "a target at +3.00% is already reached"),
    (dict(), _imp(exit_mode="NOPE"), "exit mode must be"),
    (dict(), _imp(direction="UP"), "direction must be LONG or SHORT"),
])
def test_refusals_before_the_book_lock_touch_nothing(monkeypatch, tmp_path, kw, call, expect):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, **kw)
    r = asyncio.run(run(call))
    assert isinstance(r, str) and r.startswith("ValueError") and expect in r, r
    assert [x for x in log if x[0] in ("backstop", "open_orders", "commit", "cancel_backstop")] == []
    assert _nothing_written(read) and "FOOUSDT" in eng._untracked_positions


def test_paper_refused_without_any_exchange_call(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, paper=True)
    r = asyncio.run(run(_imp()))
    assert r == "ValueError: import works in live mode only" and _kinds(log) == []


def test_pair_already_held_any_direction(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, rows=[dict(pair="FOOUSDT", direction="SHORT")])
    r = asyncio.run(run(_imp()))
    assert r == "ValueError: the bot already holds FOOUSDT; use ➕ Add" and _kinds(log) == []


def test_manual_cap_counts_imported_rows(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, cap=1, rows=[dict(pair="BARUSDT", entry_strategy="MANUAL", import_source="BINANCE_APP")])
    r = asyncio.run(run(_imp()))
    assert "manual positions cap reached (1/1" in r and _kinds(log) == []
    run, log, *_ = _harness(monkeypatch, tmp_path, cap=0)
    assert "manual trades are off" in asyncio.run(run(_imp()))


ENTRY_LIMIT = dict(kind="regular", id="77", type="LIMIT", side="BUY", qty=500.0, price=1.9, trigger_price=0.0, reduce_only=False, close_position=False)


def test_other_binance_orders_refuse_before_the_stop(monkeypatch, tmp_path):
    """📥 (343 — replaces DL 328's "any own order refuses"): an order the bot cannot take over (here a resting ENTRY limit) refuses before
    anything is placed or cancelled; the app's own TP / SL no longer refuse (tests/test_import_nostop_block.py)."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=[ENTRY_LIMIT])
    r = asyncio.run(run(_imp()))
    assert "1 open order(s) on the symbol the bot cannot take over: LIMIT BUY @ 1.9" in r and "on the opening side" in r
    assert "cancel them in the Binance app first (nothing changed)" in r and "may have left a reduce-only limit" not in r
    assert _kinds(log) == ["positions", "open_orders"] and _nothing_written(read)
    run, log, *_ = _harness(monkeypatch, tmp_path, open_orders="FAIL")
    assert "could not read the open orders of FOOUSDT" in asyncio.run(run(_imp())) and _kinds(log) == ["positions", "open_orders"]


def test_backstop_placement_failure_refuses(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, bk_id=None)
    r = asyncio.run(run(_imp()))
    assert "safety stop for FOOUSDT could not be placed — nothing imported" in r and ("block", "BACKSTOP_PLACE_FAILED") in log
    assert _kinds(log) == ["positions", "open_orders", "backstop", "open_orders"] and _nothing_written(read)   # (review 1) re-count: no algo landed


@pytest.mark.parametrize("again,expect", [
    ([_pos(contracts=900.0)], "changed on Binance while it was being imported (now LONG 900 @ 2)"),
    ([_pos(entry=2.05)], "changed on Binance while it was being imported"),
    ([], "(now closed)"),
    ("FAIL", "could not re-read Binance right before booking"),
])
def test_position_changed_before_commit_cancels_the_stop(monkeypatch, tmp_path, again, expect):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_pos()], again))
    r = asyncio.run(run(_imp()))
    assert r.startswith("ValueError") and expect in r, r
    assert _kinds(log) == ["positions", "open_orders", "backstop", "positions", "cancel_backstop"]
    assert ("cancel_backstop", "algo-1") in log and _nothing_written(read) and "FOOUSDT" in eng._untracked_positions


def test_changed_and_stop_cancel_fails_tells_the_operator(monkeypatch, tmp_path):
    run, log, *_ = _harness(monkeypatch, tmp_path, positions_seq=([_pos()], [_pos(contracts=10.0)]), bk_cancel_ok=False)
    assert "safety stop could NOT be withdrawn — cancel it in the app" in asyncio.run(run(_imp()))


def test_commit_failure_withdraws_the_stop(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, commit_fails=True)
    r = asyncio.run(run(_imp()))
    assert r.startswith("RuntimeError") and "could not be recorded" in r and "nothing imported" in r
    assert ("cancel_backstop", "algo-1") in log and [x for x in log if x[0] in ("rtp_arm", "cache")] == [] and _nothing_written(read)


def test_import_waits_for_the_pair_lock(monkeypatch, tmp_path):
    """serialized_per_pair: a bot / manual open holding the pair's lock makes the import wait (and give up with a readable refusal)."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path)
    monkeypatch.setattr(T, "manual_open_wait_s", lambda tc: 0.05)
    async def go(eng, db):
        async with eng._pair_open_guard("FOOUSDT", None):
            return await asyncio.create_task(eng.import_binance_position(db, "FOOUSDT", "LONG", "FIXED", 1.5, 3.0, app_orders_seen=[]))
    r = asyncio.run(run(go))
    assert "is being opened right now" in r and _kinds(log) == []


def test_momentum_mode_runs_the_first_tick_check(monkeypatch, tmp_path):
    run, log, T, read, eng, S, _ = _harness(monkeypatch, tmp_path)
    import models
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "ema13_cross_exit_enabled", True, raising=False); monkeypatch.setattr(th, "ema13_cross_exit_long_enabled", True, raising=False)
    monkeypatch.setattr(th, "ema13_cross_requires_stack_flip", True, raising=False)
    async def go(eng, db):
        db.add(models.PairData(pair="FOOUSDT", price=2.0, volume_24h=1e6, ema5=2.05, ema8=2.06, ema13=2.1, updated_at=dt.datetime.utcnow()))   # LONG below EMA13, stack against
        await db.commit()
        return await eng.import_binance_position(db, "FOOUSDT", "LONG", "MOMENTUM", None, None, app_orders_seen=[])
    r = asyncio.run(run(go))
    assert "Momentum stack would close this LONG at the first tick" in r and [x for x in log if x[0] == "backstop"] == []


# ── candidates (reconciler state + snapshot) ─────────────────────────────────────────────────────────────────────────────────────
def test_candidates_join_the_snapshot_and_skip_side_mismatch(monkeypatch, tmp_path):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "manual_floor_sl_pct", -3.0, raising=False)
    eng._untracked_positions = {"FOOUSDT": {"pair": "FOOUSDT", "side": "LONG", "qty": 1000.0, "entry_price": 2.0, "reason": "NO_ROW", "since": "x"},
                                "BARUSDT": {"pair": "BARUSDT", "side": "SHORT", "qty": 5.0, "entry_price": 9.0, "reason": "SIDE_MISMATCH"},
                                "BAZUSDT": {"pair": "BAZUSDT", "side": "LONG", "qty": 10.0, "entry_price": 3.0, "reason": "NO_ROW"}}
    eng._untracked_snapshot = (dt.datetime.utcnow() - dt.timedelta(seconds=12), {
        "FOOUSDT": _pos(mark=2.1, upnl=100.0), "BAZUSDT": _pos(symbol="BAZ/USDT:USDT", contracts=10.0, entry=3.0, lev=1, notional=30.0, margin=30.0 / 7.5)})
    c = {x["pair"]: x for x in eng.import_candidates()}
    assert set(c) == {"FOOUSDT", "BAZUSDT"}
    f = c["FOOUSDT"]
    assert f["side"] == "LONG" and f["qty"] == 1000.0 and f["entry_price"] == 2.0 and f["leverage"] == 20 and f["notional"] == 2100.0   # (343) at mark
    assert f["notional_basis"] == "mark" and f["notional_entry"] == 2000.0
    assert f["investment"] == 100.0 and f["mark_price"] == 2.1 and f["unrealized_pnl"] == 100.0 and f["floor_stop_pct"] == -3.0
    assert f["max_stop_pct"] == 3.0 and f["importable"] is True and f["why_not"] is None and 11 <= f["snapshot_age_s"] <= 30
    assert c["BAZUSDT"]["importable"] is False and "not a whole number" in c["BAZUSDT"]["why_not"]
    eng.is_paper_mode = True
    assert eng.import_candidates() == []


def test_reconciler_stores_the_snapshot_and_never_flattens_a_pair_being_imported(monkeypatch, tmp_path):
    import services.trading_engine as T, models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    eng._record_filter_block = lambda *a, **k: None
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "untracked_position_action", "flatten", raising=False)
    closed = []
    async def _close(*a, **k): closed.append(a); return {"id": "x"}
    async def _ro(*a, **k): return []
    monkeypatch.setattr(T.binance_service, "close_position", _close)
    monkeypatch.setattr(T.binance_service, "fetch_open_reduce_only_orders", _ro)
    async def go():
        e = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'rec.db'}")
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(e, expire_on_commit=False)
        try:
            async with S() as db:
                async with eng._pair_open_guard("FOOUSDT", None):   # an import holds the pair
                    for _ in range(3):
                        await eng.check_untracked_positions(db, [_pos(mark=2.2)])
        finally:
            await e.dispose()
    asyncio.run(go())
    ts, snap = eng._untracked_snapshot
    assert snap["FOOUSDT"]["mark_price"] == 2.2 and (dt.datetime.utcnow() - ts).total_seconds() < 60
    assert closed == []   # held by an open / import → never flattened under it


# ── API + UI wiring ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_endpoints(monkeypatch, tmp_path):
    import main
    from fastapi import HTTPException
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path)
    monkeypatch.setattr(main, "trading_engine", eng)
    async def go(eng, db):
        c = await main.manual_import_candidates()
        ok = await main.manual_import(main.ManualImportRequest(pair="FOOUSDT", direction="LONG", exit_mode="FIXED", sl_pct=1.5, tp_pct=3.0, app_orders_seen=[]), db=db)
        try:
            await main.manual_import(main.ManualImportRequest(pair="FOOUSDT", direction="LONG"), db=db)
            bad = None
        except HTTPException as e:
            bad = (e.status_code, e.detail)
        return c, ok, bad
    c, ok, bad = asyncio.run(run(go))
    assert c["is_paper"] is False and [x["pair"] for x in c["candidates"]] == ["FOOUSDT"]
    assert ok["ok"] and ok["status"] == "imported" and ok["import_source"] == "BINANCE_APP" and ok["imported_at"] and ok["backstop_pct"] == 2.5
    assert ok["sl_pct"] == -1.5 and ok["tp_pct"] == 3.0 and ok["exit_mode"] == "FIXED"
    assert bad == (400, "the bot already holds FOOUSDT; use ➕ Add")
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '@app.get("/api/manual/import_candidates")' in src and '@app.post("/api/manual/import")' in src
    assert src.count('"imported_at": (o.imported_at.isoformat() if getattr(o, \'imported_at\', None) else None)') == 2   # open + closed payloads


def test_schema_migration_and_csv_columns():
    import models
    cols = [c.name for c in models.Order.__table__.columns]
    assert "imported_at" in cols and "import_source" in cols                                     # D12: the orders CSV dumps every column
    db_src = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "(('imported_at', 'DATETIME'), ('import_source', 'VARCHAR(20)'))" in db_src


def test_ui_banner_modal_and_badges():
    assert UI.count('id="import-candidates-banner"') == 1 and UI.index('id="import-candidates-banner"') < UI.index('id="tab-open"')
    a = UI.index('<div id="import-position-modal"'); m = UI[a:UI.index('<div class="glass rounded-xl overflow-hidden">', a)]
    for i in ("import-position-close", "import-pos-head", "import-pos-facts", "import-exit-mode", "import-sltp-unit-pct", "import-sltp-unit-usd",
              "import-sl", "import-tp", "import-sl-unit-lbl", "import-tp-unit-lbl", "import-sl-conv", "import-tp-conv", "import-risk-line",
              "import-what-line", "import-pos-btn"):
        assert m.count(f'id="{i}"') == 1 and UI.count(f'id="{i}"') == 1, i
    assert m.index('id="import-risk-line"') < m.index('id="import-pos-btn"') and 'onclick="importModalBackdropClick(event)"' in m
    # the same exit options and labels as the 🖐 pop-up, Floor the default
    sel = lambda html, i: re.search(rf'<select id="{i}"[^>]*>(.*?)</select>', html, re.S).group(1)
    opts = lambda h: re.findall(r'<option value="([A-Z_]+)"[^>]*>([^<]+)</option>', h)
    assert opts(sel(UI, "import-exit-mode")) == opts(sel(UI, "manual-exit-mode"))
    assert '<option value="FLOOR" selected>' in sel(UI, "import-exit-mode")
    js = UI[UI.index("// 📥 IMPORT FROM BINANCE (DECISION_LOG 328"):UI.index("// ➕ (DECISION_LOG 324) ADD TO POSITION — the ➕ button")]
    for f in ("slTpConvertTyped(", "slTpExactPct(", "slTpPctFromUsd(", "slTpUsdFromPct(", "slTpPaintToggle(", "slTpGetUnit('import')", "slTpSaveUnit('import'"):
        assert f in js, f                                                                         # ⇄ (326) helpers reused, not copied
    assert "/api/manual/import_candidates" in js and "`${API_BASE}/api/manual/import`" in js and "window._lastTotalPortfolio" in js
    assert "% of the account" in js and "e.key === 'Escape' && !e.defaultPrevented && !document.getElementById('near-confirm') && importModalIsOpen()" in js
    assert "📱 Binance holds" in js and "that the bot does not track" in js and "📥 Import" in js
    assert UI.count("${doorBadgeHtml(o)}${importBadgeHtml(o)}") == 2                              # Open + Closed orders badge
    assert "try { loadImportCandidates().catch(() => {}); } catch (e) {}" in UI[UI.index("async function loadOpenOrders()"):][:400]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_inline_scripts_pass_node_check():
    blocks = re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', UI, re.S)
    assert blocks
    for i, b in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(b); fn = f.name
        try:
            r = subprocess.run(["node", "--check", fn], capture_output=True, text=True)
            assert r.returncode == 0, f"script block {i}: {r.stderr[:400]}"
        finally:
            os.unlink(fn)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_import_risk_line_math_in_node():
    """The pop-up's risk line and % / $ field reading, run in node against a tiny DOM stub: FIXED 1.5 % on a 1000 × 2.0 position = −$30,
    3 % of a $1,000 account; a $ stop converts on quantity × entry."""
    a = UI.index("// ⇄ SLTP-UNITS BEGIN"); b = UI.index("// 🖐 manual entry: SL / TP typed in % or $")
    c = UI.index("// 📥 IMPORT FROM BINANCE (DECISION_LOG 328"); d = UI.index("function openImportModal(")
    code = UI[a:b] + UI[c:d]
    prog = """
const els = {};
function mk(id, v) { els[id] = { id, value: v || '', disabled: false, dataset: {}, validity: { badInput: false }, textContent: '',
  classList: { _s: new Set(), toggle(k, on) { on ? this._s.add(k) : this._s.delete(k); }, contains(k) { return this._s.has(k); }, add(k) { this._s.add(k); }, remove(k) { this._s.delete(k); } } }; return els[id]; }
global.document = { getElementById: id => els[id] || null, addEventListener() {} };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.window = { _cachedConfig: { thresholds: { manual_floor_sl_pct: -3 } }, _lastTotalPortfolio: 1000 };
global.showToast = () => {}; global._setHTMLIfChanged = () => {}; global.formatTime = x => x;
global.manualStopForLeverage = (lev, f) => ({ stop: Math.max(f, -(100 / lev - 0.5) * 0.8) });
['import-risk-line', 'import-what-line', 'import-sl', 'import-tp', 'import-sl-conv', 'import-tp-conv'].forEach(i => mk(i));
mk('import-exit-mode', 'FIXED');
""" + code + """
_importCands = [{ pair: 'FOOUSDT', side: 'LONG', qty: 1000, entry_price: 2.0, leverage: 20, floor_stop_pct: -3, max_stop_pct: 3, importable: true }];
_importModalPair = 'FOOUSDT';
els['import-sl'].value = '1.5'; importRiskLine();
const r1 = els['import-risk-line'].textContent, conv1 = els['import-sl-conv'].textContent;
els['import-sl'].value = '4'; importRiskLine(); const r2 = els['import-risk-line'].textContent;
_importSlTpUnit = 'usd'; els['import-sl'].value = '30'; const p3 = importFieldPct('import-sl');
console.log(JSON.stringify({ r1, conv1, r2, p3, n: importNotional() }));
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r["n"] == 2000 and r["p3"] == 1.5 and r["conv1"] == "= $30.00"
    assert "Position $2,000 · risk at the −1.5 % stop ≈ −$30.00 (3.0 % of the account)" == r["r1"], r["r1"]
    assert "the import will be refused" in r["r2"]


# ── review fixes (DECISION_LOG 328 "Reviews") ─────────────────────────────────────────────────────────────────────────────────────
def test_own_orders_message_flags_a_bot_leftover(monkeypatch, tmp_path):
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, open_orders=[ENTRY_LIMIT])
    eng._rtp_dirty_pairs = {"FOOUSDT"}   # the bot's last close on the pair could not verify its reduce-only limit
    r = asyncio.run(run(_imp()))
    assert "1 open order(s) on the symbol the bot cannot take over" in r and "the bot's last close on this pair may have left a reduce-only limit" in r


def test_cancel_during_placement_withdraws_the_placed_stop(monkeypatch, tmp_path):
    """(review 1) a task cancelled while place_backstop_stop is in flight waits for it (shielded) and withdraws the stop it placed."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, place_delay=0.05)
    async def go(eng, db):
        t = asyncio.ensure_future(eng.import_binance_position(db, "FOOUSDT", "LONG", "FIXED", 1.5, 3.0, app_orders_seen=[]))
        while not any(x[0] == "backstop" for x in log):
            await asyncio.sleep(0.001)
        t.cancel()
        try:
            await t
            return "not cancelled"
        except asyncio.CancelledError:
            return "cancelled"
    assert asyncio.run(run(go)) == "cancelled"
    k = _kinds(log)
    assert k[:3] == ["positions", "open_orders", "backstop"] and "backstop_done" in k and ("cancel_backstop", "algo-1") in log
    assert k.index("backstop_done") < k.index("cancel_backstop") and "commit" not in k and _nothing_written(read)


def test_failed_placement_that_may_have_landed_says_so(monkeypatch, tmp_path):
    """(review 1) the call failed (None) but Binance now shows an algo order → the refusal says a stop MAY rest there."""
    landed = dict(kind="algo", id="999", type="STOP_MARKET", side="SELL", qty=0.0, price=0.0, trigger_price=1.95, reduce_only=False, close_position=True)
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, bk_id=None, open_orders_seq=[[], [landed]])
    r = asyncio.run(run(_imp()))
    assert "could not be confirmed" in r and "A stop MAY rest on Binance" in r and _nothing_written(read)
    run, log, *_ = _harness(monkeypatch, tmp_path, bk_id=None, open_orders_seq=[[], "FAIL"])
    assert "A stop MAY rest on Binance" in asyncio.run(run(_imp()))


def test_commit_error_with_unreadable_row_keeps_the_stop(monkeypatch, tmp_path):
    """(review 2) the commit errored AND the is-it-booked re-read failed → the stop is KEPT, the operator is told to check."""
    run, log, T, read, eng, *_ = _harness(monkeypatch, tmp_path, commit_fails=True)
    class _NoDB:
        def __call__(self):
            raise RuntimeError("db gone")
    monkeypatch.setattr(T, "AsyncSessionLocal", _NoDB())
    r = asyncio.run(run(_imp()))
    assert r.startswith("RuntimeError") and "may or may not have been recorded" in r and "it was kept on purpose" in r
    assert [x for x in log if x[0] == "cancel_backstop"] == []


def test_flatten_rechecks_mark_and_row_before_each_close(monkeypatch, tmp_path):
    """(review 3) flatten mode: a pair whose mark was replaced (an import cleared / re-marked it) or that gained an OPEN row while the loop
    ran is never closed."""
    import services.trading_engine as T, models
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    eng._record_filter_block = lambda *a, **k: None
    monkeypatch.setattr(T.config.trading_config.thresholds, "untracked_position_action", "flatten", raising=False)
    closed = []
    e = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'flat.db'}")
    S = async_sessionmaker(e, expire_on_commit=False)
    async def _close(symbol, side, qty):
        closed.append(symbol)
        if symbol == "FOO/USDT:USDT":   # while FOO is being closed: BAR gets a row (an import), BAZ's mark is replaced
            async with S() as s2:
                s2.add(models.Order(pair="BARUSDT", direction="LONG", status="OPEN", entry_price=1.0, quantity=10.0, investment=1.0, leverage=10.0,
                                    notional_value=10.0, confidence="STRONG_BUY", is_paper=False))
                await s2.commit()
            eng._untracked_positions["BAZUSDT"] = dict(eng._untracked_positions["BAZUSDT"])
        return {"id": "x"}
    async def _none(*a, **k): return []
    async def _true(*a, **k): return True
    monkeypatch.setattr(T.binance_service, "close_position", _close)
    monkeypatch.setattr(T.binance_service, "fetch_open_reduce_only_orders", _none)
    monkeypatch.setattr(T.binance_service, "cancel_all_backstops", _true)
    pos = [_pos(), _pos(symbol="BAR/USDT:USDT"), _pos(symbol="BAZ/USDT:USDT")]
    eng._untracked_positions = {k: {"pair": k, "side": "LONG", "qty": 1000.0, "entry_price": 2.0, "reason": "NO_ROW"} for k in ("FOOUSDT", "BARUSDT", "BAZUSDT")}
    async def go():
        async with e.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        try:
            async with S() as db:
                await eng.check_untracked_positions(db, pos)
        finally:
            await e.dispose()
    asyncio.run(go())
    assert closed == ["FOO/USDT:USDT"]


def test_candidate_payload_flags_a_leverage_where_floor_is_refused(monkeypatch):
    import services.trading_engine as T
    TE = T.TradingEngine; eng = TE.__new__(TE); eng.is_paper_mode = False
    th = T.config.trading_config.thresholds
    monkeypatch.setattr(th, "broker_backstop_enabled", True, raising=False); monkeypatch.setattr(th, "manual_floor_sl_pct", -3.0, raising=False)
    eng._untracked_positions = {"FOOUSDT": {"pair": "FOOUSDT", "side": "LONG", "qty": 1000.0, "entry_price": 2.0, "reason": "NO_ROW"}}
    eng._untracked_snapshot = (dt.datetime.utcnow(), {"FOOUSDT": _pos(lev=100)})
    c = eng.import_candidates()[0]
    assert c["leverage"] == 100 and c["floor_exceeds_max"] is True and abs(c["max_stop_pct"] - (0.5 * 0.92 - 0.3)) < 1e-9
    _w = T.manual_bullrun_worst_stop_pct(th)
    assert c["bullrun_worst_stop_pct"] == (abs(_w) if _w is not None else 0.4) and c["frenzy_stop_pct"] > 0 and c["momentum_stop_pct"] > 0
    with pytest.raises(ValueError, match="too close to this trade's own stop"):    # the engine indeed refuses Floor there
        T.manual_exit_levels(th, "FLOOR", None, None, 100)
    eng._untracked_snapshot = (dt.datetime.utcnow(), {"FOOUSDT": _pos(lev=20)})
    assert eng.import_candidates()[0]["floor_exceeds_max"] is False


def test_ui_review_fixes_static():
    js = UI[UI.index("// 📥 IMPORT FROM BINANCE (DECISION_LOG 328"):UI.index("// ➕ (DECISION_LOG 324) ADD TO POSITION — the ➕ button")]
    assert "import needs a custom stop at ${_impEsc(c.leverage)}×" in js
    assert "sel.value = c.floor_exceeds_max ? 'FIXED' : 'FLOOR'" in js and "c.bullrun_worst_stop_pct" in js
    assert "!e.defaultPrevented && !document.getElementById('near-confirm') && importModalIsOpen()" in js   # Esc never closes under the confirm
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    m = eng[eng.index("    async def open_manual_position("):eng.index("    async def set_exit_override(")]
    assert "floor = manual_floor_for_leverage" not in m and "if exit_mode not in MANUAL_EXIT_MODES:" in m


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_risk_line_warns_for_floor_and_bullrun_too():
    a = UI.index("// ⇄ SLTP-UNITS BEGIN"); b = UI.index("// 🖐 manual entry: SL / TP typed in % or $")
    c = UI.index("// 📥 IMPORT FROM BINANCE (DECISION_LOG 328"); d = UI.index("function openImportModal(")
    prog = """
const els = {};
function mk(id, v) { els[id] = { id, value: v || '', disabled: false, dataset: {}, validity: { badInput: false }, textContent: '',
  classList: { _s: new Set(), toggle(k, on) { on ? this._s.add(k) : this._s.delete(k); }, contains(k) { return this._s.has(k); }, add(k) { this._s.add(k); }, remove(k) { this._s.delete(k); } } }; return els[id]; }
global.document = { getElementById: id => els[id] || null, addEventListener() {} };
global.localStorage = { getItem() { return null; }, setItem() {} };
global.window = { _cachedConfig: { thresholds: {} }, _lastTotalPortfolio: 1000 };
global.showToast = () => {}; global._setHTMLIfChanged = () => {}; global.formatTime = x => x;
['import-risk-line', 'import-what-line', 'import-sl', 'import-tp', 'import-sl-conv', 'import-tp-conv'].forEach(i => mk(i));
mk('import-exit-mode', 'FLOOR');
""" + UI[a:b] + UI[c:d] + """
_importCands = [{ pair: 'FOOUSDT', side: 'LONG', qty: 1000, entry_price: 2.0, leverage: 100, floor_stop_pct: -0.4, max_stop_pct: 0.16,
                  bullrun_worst_stop_pct: 0.4, frenzy_stop_pct: 3, momentum_stop_pct: 0.7, floor_exceeds_max: true, importable: true }];
_importModalPair = 'FOOUSDT';
importRiskLine(); const floorTxt = els['import-risk-line'].textContent;
els['import-exit-mode'].value = 'BULLRUN'; importRiskLine(); const brTxt = els['import-risk-line'].textContent;
els['import-exit-mode'].value = 'FIXED'; els['import-sl'].value = '0.15'; importRiskLine(); const okTxt = els['import-risk-line'].textContent;
console.log(JSON.stringify({ floorTxt, brTxt, okTxt }));
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert "the import will be refused" in r["floorTxt"] and "the import will be refused" in r["brTxt"]
    assert "−0.4 % Bull-Run stop" in r["brTxt"] and "refused" not in r["okTxt"]
