"""🌡 DECISION_LOG 489 — Market-hot (top-100 up over 4 h) on the ROLLING top-100 (V2), stamped on every fill (orders.entry_mkt_up4h_pct / _n), replacing
the frozen 108 everywhere. Pins: the pure universe rule (exclusions, top-100 by volume, ties, statuses), the up-4h formula (NULL < 30,
stale > 2 h), the tracker (injected fakes, no network), the stamp contract + the Order build, the migration + the frozen day-list table,
the repair pass, the scout's stamp preference and the scout rebuild = the engine's pure function."""
import asyncio
import datetime as dt
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from services import mkt_up4h as MB  # noqa: E402

H = MB.HOUR
D0 = 1_793_923_200_000          # 2026-11-06 00:00 UTC


# ─────────────────────────────── universe rule ───────────────────────────────
def test_universe_exclusions_and_top100():
    vols = {f"C{i:03d}USDT": 1e6 * (300 - i) for i in range(150)}
    vols.update(BTCUSDT=9e12, ETHUSDT=8e12, USDCUSDT=7e12, FDUSDUSDT=6e12, PEGUSDT=5e12, ZEROUSDT=0, NANUSDT=float("nan"),
                BADUSDC=4e12, DEADUSDT=3e12)
    closes = {"PEGUSDT": [1.0] * 30, "C000USDT": [1.0] * 9 + [5.0]}        # PEG: 30 closes at $1 → pegged; C000: only 10 closes, 90 % → not
    m = MB.select_universe(vols, None, None, closes)
    assert len(m) == 100 and m[0] == "DEADUSDT" and m[1] == "C000USDT"
    assert not {"BTCUSDT", "ETHUSDT", "USDCUSDT", "FDUSDUSDT", "PEGUSDT", "ZEROUSDT", "NANUSDT", "BADUSDC"} & set(m)
    assert m == sorted(m, key=lambda s: -vols[s])                                        # rank order by volume
    st = {s: "TRADING" for s in vols}
    st["DEADUSDT"] = "SETTLING"
    del st["C001USDT"]                                                                   # missing status = not a live perpetual
    m2 = MB.select_universe(vols, st, None, closes)
    assert "DEADUSDT" not in m2 and "C001USDT" not in m2 and len(m2) == 100
    m3 = MB.select_universe(vols, None, {"C000USDT": "USDE"}, closes)                     # a named stable base (exchangeInfo baseAsset)
    assert m3[0] == "DEADUSDT" and "C000USDT" not in m3
    assert "PEGUSDT" in MB.select_universe(vols)                                         # no closes given → named list only


def test_universe_ties_and_garbage():
    vols = {"BBBUSDT": 5.0, "AAAUSDT": 5.0, "CCCUSDT": 7.0, "XUSDT": "x", 7: 1}
    assert MB.select_universe(vols, top_n=3) == ["CCCUSDT", "AAAUSDT", "BBBUSDT"]        # ties → symbol name
    assert MB.select_universe(None) == [] and MB.select_universe({}) == []


def test_peg_rule_matches_the_study():
    assert MB.is_pegged("USD1") and MB.is_pegged("U") and not MB.is_pegged("UNI")
    assert MB.is_pegged("ABC", [1.0] * 9 + [0.99] * 20) and not MB.is_pegged("ABC", [1.0] * 9)   # ≥ 10 closes needed
    assert not MB.is_pegged("ABC", [1.0] * 28 + [2.0] * 2)                                       # 93 % ≤ 95 %
    assert MB.is_pegged("ABC", [5.0] * 50 + [1.0] * 30)                                          # only the last 30 closes count


# ─────────────────────────────── up-4h formula ───────────────────────────────
def _closes(n_up, n_dn, bar=D0 + 10 * H, flat=0):
    c = {}
    for i in range(n_up):
        c[f"U{i}USDT"] = {bar - 4 * H: 1.0, bar: 1.1}
    for i in range(n_dn):
        c[f"D{i}USDT"] = {bar - 4 * H: 1.0, bar: 0.9}
    for i in range(flat):
        c[f"F{i}USDT"] = {bar - 4 * H: 1.0, bar: 1.0}
    return c


def test_up4h_share_strict_and_min_members():
    c = _closes(20, 10, flat=10)
    b, n, h = MB.up4h_from_closes(c, list(c), D0 + 10 * H + 59 * 60_000)
    assert (b, n, h) == (0.5, 40, D0 + 10 * H)                                           # flat is not up: 20 / 40
    assert MB.is_hot(b) is False and MB.is_hot(0.51) is True and MB.is_hot(None) is None
    c = _closes(20, 9)
    assert MB.up4h_from_closes(c, list(c), D0 + 10 * H) == (None, 29, D0 + 10 * H)   # < 30 members → NULL, n kept
    c = _closes(40, 0)
    assert MB.up4h_from_closes(c, list(c)[:35], D0 + 10 * H)[:2] == (1.0, 35)        # only members count


def test_up4h_stale_bar_and_forming_bar():
    c = _closes(30, 10)
    assert MB.up4h_from_closes(c, list(c), D0 + 12 * H)[0] == 0.75                    # exactly 2 h old → still read
    assert MB.up4h_from_closes(c, list(c), D0 + 12 * H + 1) == (None, 0, None)       # > 2 h → NULL
    assert MB.up4h_from_closes(c, list(c), D0 + 10 * H - 1) == (None, 0, None)       # the bar had not closed yet (no older bar)
    assert MB.up4h_from_closes(None, None, 0) == (None, 0, None)
    rows = [[D0 + i * H, 1, 1, 1, 2.0 + i, 1] for i in range(6)]
    assert sorted(MB.closes_from_rows(rows, D0 + 5 * H)) == [D0 + H * k for k in range(1, 6)]   # the forming bar is dropped


# ─────────────────────────────── tracker (injected fakes) ───────────────────────────────
def _fake_inputs(n=130):
    vols = {f"S{i:03d}USDT": 1e9 - i for i in range(n)}
    vols.update(BTCUSDT=9e12, USDCUSDT=8e12)
    return dict(vols=vols, statuses={s: "TRADING" for s in vols}, bases={"USDCUSDT": "USDC"}, prices={s: 5.0 for s in vols})


def _tracker(store, calls, now_ms, fail=(), up_from=40):
    async def kl(sym, interval, limit, end_ms=None):
        calls.append((sym, interval, end_ms))
        if sym in fail:
            return None
        if interval == "1d":
            day = MB.universe_day_ms(now_ms)
            return [[day - 2 * MB.DAY, 1, 1, 1, 5.0, 1, 0, 1.0], [day - MB.DAY, 1, 1, 1, 5.0, 1, 0, 1e6 + int(sym[1:4])]]
        last = MB.last_closed_bar_ms(end_ms + 1 if end_ms is not None else now_ms)
        i = int(sym[1:4])
        return [[last - (6 - k) * H, 1, 1, 1, (1.0 if k < 5 else (1.2 if i >= up_from else 0.8)), 1] for k in range(1, 6)] + \
               ([[last, 1, 1, 1, 9.0, 1]] if end_ms is None else [])

    async def uni():
        return _fake_inputs()

    async def load(day):
        return store.get(day)

    async def save(day, members, how):
        store.setdefault(day, list(members))
        return store[day]
    return MB.MktUp4hTracker(fetch_klines=kl, fetch_universe=uni, is_limited=lambda: False, load_universe=load, save_universe=save,
                                clock=lambda: now_ms)


def test_tracker_refresh_and_stamp_fresh_day():
    now = D0 + 5_000                                                                     # 00:00:05 — fresh day → 24 h ticker ranking
    store, calls = {}, []
    t = _tracker(store, calls, now)
    v = asyncio.run(t.refresh(now))
    assert v["bar_close_ms"] == D0 and v["n"] == 100 and abs(v["pct"] - 0.60) < 1e-9   # S040…S099 up of S000…S099
    assert store[D0][:2] == ["S000USDT", "S001USDT"] and "BTCUSDT" not in store[D0] and "USDCUSDT" not in store[D0]
    assert not [c for c in calls if c[1] == "1d"]                                       # no peg / exact reads (no near-$1 candidate, fresh day)
    assert t.stamp(D0 + 3_599_000) == (0.6, 100) and t.stamp(D0 + H) == (None, None) and t.stamp(D0 - 1) == (None, None)
    assert t.universe["how"] == "ticker-24h"
    calls.clear()
    assert asyncio.run(t.refresh(D0 + 600_000)) is v and not calls                       # idempotent per hour


def test_tracker_midday_cold_start_is_exact_and_stored_list_wins():
    now = D0 + 10 * H + 5_000
    store, calls = {}, []
    t = _tracker(store, calls, now)
    asyncio.run(t.refresh(now))
    assert t.universe["how"] == "1d-exact" and len([c for c in calls if c[1] == "1d"]) == 130   # every eligible candidate (≤ EXACT_CANDIDATES)
    assert store[D0][0] == "S129USDT"                                                    # ranked on the exact previous-day volume, not the 24 h ticker
    store2 = {D0: ["S005USDT"] * 1 + [f"S{i:03d}USDT" for i in range(6, 105)]}          # a restart: the stored list is THE list
    t2 = _tracker(store2, [], now)
    asyncio.run(t2.refresh(now))
    assert t2.universe["members"] == store2[D0] and t2.universe["how"] == "stored"


def test_tracker_failures_null_and_kept_list():
    now = D0 + 5_000
    calls = []
    t = _tracker({}, calls, now, fail={f"S{i:03d}USDT" for i in range(11)})             # 11 reads fail twice → no value for the hour
    assert asyncio.run(t.refresh(now)) is None and t.stamp(now) == (None, None)
    t = _tracker({}, [], now, fail={f"S{i:03d}USDT" for i in range(10)})                # 10 → counted as no data (90 members)
    v = asyncio.run(t.refresh(now))
    assert v["n"] == 90
    limited = _tracker({}, [], now)
    limited.is_limited = lambda: True
    assert asyncio.run(limited.refresh(now)) is None
    t = _tracker({}, [], now)
    asyncio.run(t.refresh(now))

    async def broken():
        raise RuntimeError("down")
    t.fetch_universe = broken
    nxt = D0 + MB.DAY + 30_000
    assert asyncio.run(t.ensure_universe(nxt)) == t.universe["members"] and t.universe["day_ms"] == D0   # yesterday's list kept


def test_tracker_repair_uses_stored_list_and_end_time():
    now = D0 + 20 * H
    store, calls = {D0: [f"S{i:03d}USDT" for i in range(100)]}, []
    t = _tracker(store, calls, now)
    res, why = asyncio.run(t.repair(D0 + 7 * H + 123))
    assert why is None and res["bar_close_ms"] == D0 + 7 * H and res["n"] == 100 and abs(res["pct"] - 0.6) < 1e-9
    assert {c[2] for c in calls} == {D0 + 7 * H - 1}
    assert asyncio.run(t.repair(D0 - MB.DAY + 5))[0] is None                            # no stored list for that day → never rebuilt


# ─────────────────────────────── engine wiring ───────────────────────────────
def test_stamp_contract_and_order_build_wiring():
    import models
    import services.trading_engine as TE
    E = TE.TradingEngine
    for c in ("entry_mkt_up4h_pct", "entry_mkt_up4h_n"):
        assert c in {x.name for x in models.Order.__table__.columns} and E.stamp_class(c) == "universal" and c in E.STAMP_SINCE
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = src.index("        order = Order(\n            binance_order_id=binance_order_id,")
    j = src.index("        db.add(order)", i)
    assert "entry_mkt_up4h_pct=_mb_stamp.get('entry_mkt_up4h_pct')" in src[i:j] and "_mb_stamp = mkt_up4h_stamp()" in src[i - 400:i]
    k = src.index("    async def _manual_entry_stamps(")
    assert "st.update(mkt_up4h_stamp())" in src[k:k + 12000]                         # MANUAL + app-trade imports


def test_stamp_reads_the_tracker_and_never_raises(monkeypatch):
    import services.trading_engine as TE
    now = int(dt.datetime(2026, 11, 6, 10, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
    monkeypatch.setattr(TE.mkt_up4h_tracker, "value", dict(bar_close_ms=MB.last_closed_bar_ms(now), pct=0.5612345678, n=97))
    assert TE.mkt_up4h_stamp(now) == {"entry_mkt_up4h_pct": 0.561235, "entry_mkt_up4h_n": 97}
    assert TE.mkt_up4h_stamp(now + H) == {"entry_mkt_up4h_pct": None, "entry_mkt_up4h_n": None}

    def boom(*a):
        raise RuntimeError("x")
    monkeypatch.setattr(TE.mkt_up4h_tracker, "stamp", boom)
    assert TE.mkt_up4h_stamp() == {"entry_mkt_up4h_pct": None, "entry_mkt_up4h_n": None}


def test_migration_adds_the_columns_and_the_day_table(tmp_path, monkeypatch):
    import models
    import database
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'old.db'}")

    async def go():
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
            for col in ("entry_mkt_up4h_pct", "entry_mkt_up4h_n"):
                await c.execute(text(f"ALTER TABLE orders DROP COLUMN {col}"))
            await c.execute(text("DROP TABLE mkt_up4h_universes"))
        monkeypatch.setattr(database, "engine", eng)
        await database.init_db()
        async with eng.begin() as c:
            ocols = {r[1]: r[2] for r in (await c.execute(text("PRAGMA table_info(orders)"))).all()}
            tabs = [r[0] for r in (await c.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))).all()]
        await eng.dispose()
        return ocols, tabs
    ocols, tabs = asyncio.run(go())
    assert ocols.get("entry_mkt_up4h_pct") == "FLOAT" and ocols.get("entry_mkt_up4h_n") == "INTEGER" and "mkt_up4h_universes" in tabs


def test_day_list_store_first_wins_and_repair_pass(monkeypatch):
    import models
    import main as M
    import services.trading_engine as TE
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        monkeypatch.setattr(TE, "AsyncSessionLocal", SL)
        monkeypatch.setattr(M, "AsyncSessionLocal", SL)
        a = await TE._mkt_universe_save(D0, ["AUSDT", "BUSDT"], "ticker-24h")
        b = await TE._mkt_universe_save(D0, ["CUSDT"], "1d-exact")
        now = dt.datetime(2026, 11, 6, 20, 0)
        base = dict(direction="LONG", status="CLOSED", is_paper=True, entry_price=1.0, quantity=1.0, investment=1.0, leverage=1,
                    notional_value=1.0, confidence="STRONG_BUY")
        async with SL() as s:
            s.add(models.Order(pair="XUSDT", entry_strategy="MOMENTUM", opened_at=dt.datetime(2026, 11, 6, 7, 10), **base))
            s.add(models.Order(pair="YUSDT", entry_strategy="MOMENTUM", opened_at=dt.datetime(2026, 11, 6, 7, 50), **base))
            s.add(models.Order(pair="ZUSDT", entry_strategy="MOMENTUM", opened_at=dt.datetime(2026, 11, 6, 9, 0), entry_mkt_up4h_n=12, **base))
            s.add(models.Order(pair="MUSDT", entry_strategy="MANUAL", opened_at=dt.datetime(2026, 11, 6, 8, 0), **base))
            await s.commit()

        class Tr:
            n = 0

            def _limited(self):
                return False

            async def repair(self, ms):
                Tr.n += 1
                return dict(pct=0.42, n=88, bar_close_ms=MB.last_closed_bar_ms(ms)), None
        monkeypatch.setattr(M, "_alt_repair_room", lambda tr: True)
        M._mb_repair_tries.clear()
        fixed = await M.mkt_up4h_repair_pass(Tr(), now)
        async with SL() as s:
            got = {o.pair: (o.entry_mkt_up4h_pct, o.entry_mkt_up4h_n) for o in (await s.execute(select(models.Order))).scalars()}
        await eng.dispose()
        return a, b, fixed, Tr.n, got
    a, b, fixed, n, got = asyncio.run(go())
    assert a == ["AUSDT", "BUSDT"] and b == a                                              # an earlier row for the day wins
    assert fixed == 2 and n == 1                                                         # one read set for the 07:00 hour, both fills filled
    assert got["XUSDT"] == (0.42, 88) and got["YUSDT"] == (0.42, 88)
    assert got["ZUSDT"] == (None, 12) and got["MUSDT"] == (None, None)                  # a real < 30 reading stays; MANUAL never repaired


def test_loop_wakes_hh0130_and_0030_at_the_day_turn():
    import main as M
    assert M._mkt_up4h_next_wake(3600 * 5 + 100) == 3600 - 100 + 90          # HH:01:30 — off the HH:00 5m-close scan burst (safety review)
    assert M._mkt_up4h_next_wake(86400 - 10) == 10 + 30
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_mkt_up4h_task = asyncio.create_task(mkt_up4h_loop())" in src and "_mkt_up4h_task.cancel()" in src


def test_wl_tables_and_both_exports_carry_up4h():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count(">Up4h</th>") == 1
    assert html.count("${'AltBeta'.padEnd(8)} ${'Up4h'}") == 4                       # outcome + strategy headers × BOTH exports
    assert html.count("${r.avg_mkt_up4h_pct != null ? r.avg_mkt_up4h_pct.toFixed(3) : '-'}") == 2
    assert html.count("${g(r.avg_mkt_up4h_pct,3)}") == 2
    assert html.count("safe(r.avg_mkt_up4h_pct).toFixed(3)") == 2                    # by-Outcome + by-Strategy UI rows
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert src.count('"avg_mkt_up4h_pct"') == 2


# ─────────────────────────────── scout side: same rule, stamp preferred ───────────────────────────────
def test_scout_series_equals_the_engine_function():
    import scout_ml_mkt_hot_watch as W
    rng = np.random.default_rng(3)
    idx = pd.date_range("2026-11-05 00:00", periods=40, freq="1h")
    syms = [f"S{i}USDT" for i in range(60)]
    cl = pd.DataFrame(rng.lognormal(0, 0.02, (40, 60)).cumprod(axis=0), index=idx, columns=syms)
    cl.iloc[5:9, :35] = np.nan                                                             # a hole: those bars have < 30 members
    U = {pd.Timestamp("2026-11-05"): syms[:50], pd.Timestamp("2026-11-06"): syms[10:]}
    br = W.breadth_series(cl, U)
    closes = {s: {int(t.value // 10**6): float(v) for t, v in cl[s].dropna().items()} for s in syms}
    for t in idx:
        b, n, h = MB.up4h_from_closes(closes, U[t.normalize()], int(t.value // 10**6))
        if h == int(t.value // 10**6):
            got = br.get(t)
            assert (b is None and pd.isna(got)) or abs(got - b) < 1e-12, (t, b, got)


def test_scout_prefers_the_stamp(tmp_path):
    import scout_ml_mkt_hot_watch as W
    t0 = pd.Timestamp("2026-11-08 10:00")
    rows = []
    for i in range(3):
        rows.append(dict(opened_at=(t0 + pd.Timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(t0 + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                         pair=f"P{i}USDT", direction="LONG", entry_strategy="MOMENTUM", status="CLOSED", pnl_percentage=0.5, pnl=5.0,
                         close_reason="RUNNER_TRAIL", cell_multiplier_source="UNMATCHED", entry_mkt_up4h_pct=[0.7, 0.3, np.nan][i]))
    p = tmp_path / "scalpars_orders_paper_2026-11-09_00-00-00.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    c, _ = W.load([str(p)])
    assert list(W.needs_rebuild(c)) == [False, False, True]
    seen = {}

    def fn(t_min, now):
        seen["t_min"] = t_min
        return pd.Series([0.9], index=[t0]), True
    s = W.score(c, fn(0, 0)[0])
    assert list(s.zone) == ["zone", "rest", "zone"]                                       # stamps win over the rebuilt 0.9; the NULL one is rebuilt
    out = "\n".join(W.run(int((t0 + pd.Timedelta(days=1)).value // 10**6), [str(p)], None, int((t0 - pd.Timedelta(hours=1)).value // 10**6), fn))
    assert seen["t_min"] == int((t0 + pd.Timedelta(minutes=2)).value // 10**6)           # only the unstamped fill drives the rebuild window


def test_scout_constants_are_the_engine_constants():
    import scout_ml_mkt_hot_watch as W
    assert W.HOT_CUT == MB.HOT_CUT and W.MIN_COINS == MB.MIN_MEMBERS and W.STAMP == "entry_mkt_up4h_pct"
    assert not hasattr(W, "UNIVERSE") and not hasattr(W, "MIN_UNIVERSE")                 # the frozen 108 is no longer a measure
    assert len(W.UNIVERSE_F108_RETIRED) == 108                                           # kept only for the 11-06 study's re-run


# ─────────────────────────────── 🔒 zero-risk contract (operator 11-06) ───────────────────────────────
def test_stamp_never_awaits_network_even_while_a_read_hangs():
    import time as _t
    now = D0 + 10 * H + 5_000
    hang = asyncio.Event()

    async def kl(*a, **k):
        await hang.wait()                                                                # a Binance read that never answers

    async def uni():
        await hang.wait()
    t = MB.MktUp4hTracker(fetch_klines=kl, fetch_universe=uni, is_limited=lambda: False, clock=lambda: now)
    t.value = dict(bar_close_ms=D0 + 10 * H, pct=0.7, n=99)

    async def go():
        task = asyncio.create_task(t.refresh(D0 + 11 * H + 5_000))                       # the next hour's refresh is stuck on the network
        await asyncio.sleep(0)
        t0 = _t.perf_counter()
        got = [t.stamp(now), t.stamp(D0 + 11 * H + 6_000)]
        dt_ms = (_t.perf_counter() - t0) * 1000
        task.cancel()
        return got, dt_ms
    got, dt_ms = asyncio.run(go())
    assert got == [(0.7, 99), (None, None)] and dt_ms < 50                               # in-memory: instant; the stuck hour → NULL at once
    assert not asyncio.iscoroutinefunction(MB.MktUp4hTracker.stamp)


def test_no_read_while_limited_banned_or_above_the_weight_ceiling():
    now = D0 + 5_000
    for kw in (dict(is_limited=lambda: True), dict(used_weight=lambda: MB.WEIGHT_CEILING + 1),
               dict(is_limited=lambda: (_ for _ in ()).throw(RuntimeError("x")))):         # an unreadable ban state counts as limited
        store, calls = {D0 - MB.DAY: ["S000USDT"] * 100}, []
        t = _tracker(store, calls, now)
        for k, v in kw.items():
            setattr(t, k, v)
        t.universe = dict(day_ms=D0, members=[f"S{i:03d}USDT" for i in range(100)], how="stored")
        assert asyncio.run(t.refresh(now)) is None and asyncio.run(t.repair(now - 3 * H))[0] is None
        assert calls == [], kw
    calls = []
    t = _tracker({}, calls, now)
    t.used_weight = lambda: MB.WEIGHT_CEILING                                             # at the ceiling (not above) → reads allowed
    assert asyncio.run(t.refresh(now)) is not None and calls

    seen = []                                                                             # the IP weight rises mid-read → the rest of the pass stops

    async def kl(sym, interval, limit, end_ms=None):
        seen.append(sym)
        return None
    t = MB.MktUp4hTracker(fetch_klines=kl, is_limited=lambda: False, used_weight=lambda: 900 if len(seen) >= 3 else 100, concurrency=1)
    t.universe = dict(day_ms=D0, members=[f"S{i:03d}USDT" for i in range(100)], how="stored")
    assert asyncio.run(t.refresh(now)) is None and len(seen) == 3


def test_engine_tracker_is_wired_to_the_shared_ban_weight_and_switch(monkeypatch):
    import services.trading_engine as TE
    import services.binance_service as BS
    tr = TE.mkt_up4h_tracker
    assert tr.is_limited == TE.binance_service.regime_limited and tr.used_weight == TE.binance_service.ip_used_weight_1m
    assert tr.concurrency <= 3 and tr.call_timeout_s <= 5.0 and MB.WEIGHT_CEILING == 500                      # (DL 491) 600 → 500
    assert tr.read_gap_s == MB.READ_GAP_S == 0.5 and tr.cold_gap_s == MB.COLD_GAP_S == 1.5                      # the LIVE tracker is paced
    monkeypatch.setattr(BS, "_ban_until", 1e12)                                           # an active ban (or the post-ban cooldown)
    assert TE.binance_service.regime_limited() is True and tr._limited() is True


def test_kill_switch_nulls_stamps_and_stops_every_read(monkeypatch):
    import config
    import services.trading_engine as TE
    now = int(dt.datetime(2026, 11, 6, 10, 30, tzinfo=dt.timezone.utc).timestamp() * 1000)
    monkeypatch.setattr(TE.mkt_up4h_tracker, "value", dict(bar_close_ms=MB.last_closed_bar_ms(now), pct=0.6, n=99))
    monkeypatch.setattr(config.trading_config, "mkt_up4h_enabled", True, raising=False)
    assert TE.mkt_up4h_stamp(now)["entry_mkt_up4h_pct"] == 0.6
    monkeypatch.setattr(config.trading_config, "mkt_up4h_enabled", False, raising=False)
    assert TE.mkt_up4h_stamp(now) == {"entry_mkt_up4h_pct": None, "entry_mkt_up4h_n": None}
    calls = []
    t = _tracker({}, calls, D0 + 5_000)
    t.enabled = lambda: False
    assert asyncio.run(t.refresh(D0 + 5_000)) is None and asyncio.run(t.repair(D0))[0] is None and calls == []
    import main as M
    assert M._mkt_up4h_on() is False


def test_kill_switch_is_d11_complete():
    import config
    import json as _json
    assert config.TradingConfig().mkt_up4h_enabled is True
    assert _json.load(open(os.path.join(ROOT, "trading_config.json")))["mkt_up4h_enabled"] is True
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "mkt_up4h_enabled: Optional[bool] = None" in src
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-mkt-up4h-enabled"') == 1 and html.count("getElementById('config-mkt-up4h-enabled')") == 2   # UI + load + save
    assert "Market-hot tracker (top-100 coins up over 4 h)" in html


def test_loop_boot_delay_and_daily_prune():
    import main as M
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    body = src[src.index("async def mkt_up4h_loop():"):src.index("async def bnb_swap_loop():")]
    assert M.MKT_UP4H_BOOT_S >= 60 and "await asyncio.sleep(MKT_UP4H_BOOT_S)" in body and "> 86400" in body
    assert "if not _mkt_up4h_on():" in body


def test_reads_are_paced_and_the_cold_pace_is_restored():
    """(11-06 tuning) read STARTS are spaced by read_gap_s across concurrent workers; the cold ranking's slower pace never leaks."""
    import time as _t
    starts = []

    async def kl(sym, interval, limit, end_ms=None):
        starts.append(_t.monotonic())
        return []
    t = MB.MktUp4hTracker(fetch_klines=kl, is_limited=lambda: False, concurrency=3, read_gap_s=0.05, cold_gap_s=0.1)

    async def run():
        await asyncio.gather(*[t._klines(f"S{i}USDT", "1h", 6) for i in range(6)])
    asyncio.run(run())
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert len(starts) == 6 and min(gaps) >= 0.045                                       # never two starts closer than the gap
    assert t._gap == t.read_gap_s
    unpaced = MB.MktUp4hTracker(fetch_klines=kl, is_limited=lambda: False)
    assert unpaced.read_gap_s == 0.0 and unpaced._gap == 0.0                               # the default (tests' fakes) is unpaced


def test_cold_pace_applied_in_build_and_restored_on_every_path():
    """(491 review) the mid-day cold ranking reads 1d volumes at cold_gap_s, the peg reads run at read_gap_s, and the gap is back to
    read_gap_s after a successful build AND after a failed one (too few volumes)."""
    day = MB.universe_day_ms(1_793_000_000_000)
    now = day + 12 * 3600_000                                                             # mid-day → the exact previous-day ranking
    syms = [f"C{i:03d}USDT" for i in range(160)]
    seen = []

    def make(ok):
        async def kl(sym, interval, limit, end_ms=None):
            seen.append((interval, t._gap))
            if not ok:
                return []
            if interval == "1d" and limit == 3:
                return [[day - MB.DAY, 1, 1, 1, 1, 1, 0, float(1000 - int(sym[1:4]))]]
            return [[day - MB.DAY * k, 1, 1, 1, 1.0] for k in range(1, 32)]
        return kl

    async def uni():
        return {"vols": {s: 1e6 - i for i, s in enumerate(syms)}, "statuses": {s: "TRADING" for s in syms},
                "bases": {s: s[:-4] for s in syms}, "prices": {s: (1.0 if s == "C150USDT" else 5.0) for s in syms}}
    t = MB.MktUp4hTracker(fetch_klines=make(True), fetch_universe=uni, is_limited=lambda: False, read_gap_s=0.001, cold_gap_s=0.002)
    members = asyncio.run(t._build(day, now))
    assert len(members[0] if isinstance(members, tuple) else members) == MB.TOP_N
    assert {g for i, g in seen if i == "1d" and g == 0.002} and all(g == 0.002 for i, g in seen[:MB.EXACT_CANDIDATES])
    assert t._gap == 0.001
    seen.clear()
    t2 = MB.MktUp4hTracker(fetch_klines=make(False), fetch_universe=uni, is_limited=lambda: False, read_gap_s=0.001, cold_gap_s=0.002)
    t = t2
    import pytest as _pt
    with _pt.raises(RuntimeError):
        asyncio.run(t2._build(day, now))
    assert t2._gap == 0.001                                                                # restored after the failure
