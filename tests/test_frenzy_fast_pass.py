"""🔥 FRENZY pass at the bar close (DECISION_LOG 191): its own task, serialised with the scan's inline call; journal context task-local."""
import asyncio
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import services.trading_engine as T  # noqa: E402


def _eng():
    TE = T.TradingEngine; return TE.__new__(TE)


def test_journal_context_is_task_local():
    eng = _eng()

    async def worker(name, out):
        eng._journal_pair = name; eng._journal_ctx = {"p": name}
        await asyncio.sleep(0.01)                     # the other task writes in between
        out[name] = (eng._journal_pair, eng._journal_ctx)

    async def main():
        out = {}
        await asyncio.gather(worker("SANDUSDT", out), worker("ENJUSDT", out))
        return out

    out = asyncio.run(main())
    assert out == {"SANDUSDT": ("SANDUSDT", {"p": "SANDUSDT"}), "ENJUSDT": ("ENJUSDT", {"p": "ENJUSDT"})}
    import contextvars
    fresh = contextvars.Context()                                  # order-independent: earlier tests may have set it in the main context
    assert fresh.run(lambda: (eng._journal_pair, eng._journal_ctx, eng._open_ctx_momentum)) == (None, None, True)


def test_one_pass_at_a_time_and_never_raises():
    eng = _eng(); state = {"active": 0, "max": 0, "runs": 0}

    async def fake_pass(db):
        state["active"] += 1; state["max"] = max(state["max"], state["active"]); state["runs"] += 1
        await asyncio.sleep(0.02)
        state["active"] -= 1

    eng._update_frenzy_pass = fake_pass

    async def main():
        await asyncio.gather(eng._update_frenzy(None), eng._update_frenzy(None), eng._update_frenzy(None))

    asyncio.run(main())
    assert state["runs"] == 3 and state["max"] == 1                # serialised (the real pass returns at once when the bar is done)

    async def boom(db):
        raise RuntimeError("exchange down")
    eng._update_frenzy_pass = boom
    asyncio.run(eng._update_frenzy(None))                          # logged, never raised


def test_bar_close_task_wiring():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "async def frenzy_loop():" in main and "_frenzy_task = asyncio.create_task(frenzy_loop())" in main
    assert "for task in (_monitor_task, _scan_task, _bnb_swap_task, _nav_task, _frenzy_task, _orderbook_task):" in main
    assert "(int(_now // 300) + 1) * 300 + 4 - _now" in main                   # ~4 s after every 5-minute close
    assert "if await trading_engine._update_frenzy(db):" in main and "await self._update_frenzy(db, wait=False)" in eng   # task + the scan's non-waiting fallback
    assert "@serialized_per_pair(lane=BOT_OPEN_LANE)" in eng and "_open_ctx_momentum = property(" in eng
    assert "async def _update_frenzy_pass(self, db):" in eng and "await self._update_frenzy_pass(db)" in eng
    assert "_journal_pair = property(" in eng and "_journal_ctx = property(" in eng


def test_scan_fallback_never_waits_for_the_task():
    eng = _eng(); ran = []

    async def slow_pass(db):
        ran.append("task"); await asyncio.sleep(0.05)

    eng._update_frenzy_pass = slow_pass

    async def main():
        t = asyncio.create_task(eng._update_frenzy(None))
        await asyncio.sleep(0.01)
        r = await asyncio.wait_for(eng._update_frenzy(None, wait=False), 0.02)   # returns at once, does not queue
        await t
        return r

    assert asyncio.run(main()) is False and ran == ["task"]


def test_bot_open_lane_serialises_bot_opens_across_pairs():
    """Both reviews: a FRENZY open (own task) and a scan open on ANOTHER pair must not both pass the slot check."""
    state = {"active": 0, "max": 0}

    class _E:
        _pair_open_guard = T.TradingEngine._pair_open_guard

        @T.serialized_per_pair(lane=T.BOT_OPEN_LANE)
        async def open_position(self, db, pair):
            state["active"] += 1; state["max"] = max(state["max"], state["active"])
            await asyncio.sleep(0.02)
            state["active"] -= 1
            return pair

        @T.serialized_per_pair()
        async def manual(self, db, pair):
            await asyncio.sleep(0.02); return pair

    e = _E()

    async def main():
        return await asyncio.gather(e.open_position(None, pair="AINUSDT"), e.open_position(None, pair="SANDUSDT"), e.manual(None, pair="MOVRUSDT"))

    assert asyncio.run(main()) == ["AINUSDT", "SANDUSDT", "MOVRUSDT"] and state["max"] == 1


def test_nested_bot_opens_never_hang_and_manual_runs_alongside():
    """Lane review: a lane holder re-entering open_position (same pair = the flip path, or another pair) must not deadlock; manual opens
    do not wait for the bot lane."""
    log = []

    class _E:
        _pair_open_guard = T.TradingEngine._pair_open_guard

        @T.serialized_per_pair(lane=T.BOT_OPEN_LANE)
        async def open_position(self, db, pair, depth=0):
            log.append(("bot+", pair)); await asyncio.sleep(0.03)
            if depth == 0:
                await self.open_position(None, pair=pair, depth=1)            # flip-style re-entry, same pair
                await self.open_position(None, pair="OTHERUSDT", depth=1)     # another pair from inside the lane
            log.append(("bot-", pair))

        @T.serialized_per_pair()
        async def manual(self, db, pair):
            log.append(("man+", pair)); await asyncio.sleep(0.01); log.append(("man-", pair))

    e = _E()

    async def main():
        await asyncio.wait_for(asyncio.gather(e.open_position(None, pair="AINUSDT"), e.manual(None, pair="SANDUSDT")), 2.0)

    asyncio.run(main())
    i_man_end = log.index(("man-", "SANDUSDT")); i_bot_end = max(i for i, x in enumerate(log) if x == ("bot-", "AINUSDT"))
    assert i_man_end < i_bot_end                                   # the manual open finished while the bot open still held the lane


def test_update_frenzy_reports_whether_a_pass_ran(monkeypatch):
    eng = _eng()
    monkeypatch.setattr(T, "_frenzy_status", {"pass_bar": None, "passes": 0})

    async def real_pass(db):
        T._frenzy_status["pass_bar"] = 123; T._frenzy_status["passes"] = int(T._frenzy_status.get("passes") or 0) + 1

    async def done_pass(db):
        return None                                               # the bar was already judged: nothing changes

    eng._update_frenzy_pass = real_pass
    assert asyncio.run(eng._update_frenzy(None)) is True
    eng._update_frenzy_pass = done_pass
    assert asyncio.run(eng._update_frenzy(None)) is False


def test_frenzy_lateness_rechecked_inside_the_open():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "frenzy_long=not (wide or lite), frenzy_wide=wide, frenzy_lite=lite, frenzy_bar_open_ms=(int(_leash_time.time() * 1000) if _cu else bar_open), frenzy_catchup=_cu," in eng   # ⏪ Oct-6: a catch-up re-checks only the lane wait
    assert "if _frenzy and frenzy_bar_open_ms is not None:" in eng and "_fz_late2 > FRENZY_ENTRY_MAX_LATE_S" in eng
    assert eng.index("_fz_late2 > FRENZY_ENTRY_MAX_LATE_S") < eng.index("binance_order_id = None")   # before the order path (deep review)
    assert "if not order and not _why:" in eng and "self._frenzy_open_refusal = None" in eng   # a late refusal is reported as such, counted once
    assert "await asyncio.wait_for(binance_service.get_ohlcv(f\"{pair[:-4]}/USDT:USDT\", '1h', 260), 8.0)" in eng
