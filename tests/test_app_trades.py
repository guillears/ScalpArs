"""🧾 DECISION_LOG 329 — auto-record Binance-app trades. Fabricated fills only (no network: a fake exchange service). Pins the episode replay
(scale-in/out, flip through zero → two episodes, still open → nothing), the pure app trade's numbers (VWAPs, pnl = Σ realizedPnl − commission,
BNB commission at the fill minute + the estimate flag, funding attached NOT in pnl, pnl %), idempotency, the mixed episodes (fully the bot's →
nothing; the BTW case → one APP_ADD_SOLD_BY_BOT line = episode − booked; a row still OPEN → deferred), ambiguity → nothing + log, the era-start
bound, bot-stopped periods, ban / rate-limit stop, kill switch, the reconcile math + endpoint, the views (sleeve / strategy split, Bot view
excludes them), the export wiring, the config D11 wiring and the migration."""
import asyncio, datetime as dt, json, logging, os, re, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import pytest

from services import app_trades as A

UI = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
MS = lambda d: A.dt_to_ms(d)
T0 = dt.datetime(2026, 10, 1, 0, 0, 0)                       # the first live order (era start)
NOW = dt.datetime(2026, 10, 20, 12, 0, 0)


def raw(i, oid, t, side, price, qty, rp=0.0, comm=0.0, asset="USDT", maker=False, sym="FOOUSDT"):
    return {"id": i, "orderId": oid, "time": MS(t) if isinstance(t, dt.datetime) else int(t), "side": side, "price": str(price), "qty": str(qty),
            "realizedPnl": str(rp), "commission": str(comm), "commissionAsset": asset, "maker": maker, "symbol": sym}


def fills(*rows):
    return [A.parse_fill(r) for r in rows]


# ── pure: episodes ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_episode_scale_in_out_one_row_at_peak():
    t = dt.datetime(2026, 10, 5, 10)
    f = fills(raw(1, "a", t, "BUY", 2.0, 10), raw(2, "b", t + dt.timedelta(seconds=5), "BUY", 2.2, 10),
              raw(3, "c", t + dt.timedelta(seconds=9), "SELL", 2.3, 5, rp=1.0), raw(4, "d", t + dt.timedelta(seconds=12), "SELL", 2.4, 15, rp=4.0))
    eps, tail, lead = A.build_episodes("FOOUSDT", f)
    assert len(eps) == 1 and tail is None and lead == 0
    e = eps[0]
    assert e.direction == "LONG" and e.max_abs_net == 20 and e.key == "FOOUSDT:1" and e.n_fills == 4
    assert abs(A.vwap(e.opens()) - 2.1) < 1e-12 and abs(A.vwap(e.closes()) - (2.3 * 5 + 2.4 * 15) / 20) < 1e-12


def test_flip_through_zero_is_two_episodes_with_split_fill():
    t = dt.datetime(2026, 10, 5, 10)
    f = fills(raw(1, "a", t, "BUY", 2.0, 10, comm=0.2), raw(2, "b", t + dt.timedelta(minutes=1), "SELL", 2.1, 15, rp=1.0, comm=1.5),
              raw(3, "c", t + dt.timedelta(minutes=2), "BUY", 2.0, 5, rp=0.5, comm=0.1))
    eps, tail, _ = A.build_episodes("FOOUSDT", f)
    assert [e.direction for e in eps] == ["LONG", "SHORT"] and tail is None
    a, b = eps
    assert a.key == "FOOUSDT:1" and b.key == "FOOUSDT:2" and a.key != b.key
    assert a.closes()[0].qty == 10 and abs(a.closes()[0].frac - 10 / 15) < 1e-12 and a.closes()[0].rpnl == 1.0   # rp on the closing part
    assert b.opens()[0].qty == 5 and b.opens()[0].rpnl == 0.0 and b.max_abs_net == 5
    usd = lambda asset, t: (1.0, False)
    assert abs(A.comm_usdt(a.legs, usd)[0] - (0.2 + 1.0)) < 1e-12 and abs(A.comm_usdt(b.legs, usd)[0] - (0.5 + 0.1)) < 1e-12


def test_still_open_and_leading_partial_are_not_episodes():
    t = dt.datetime(2026, 10, 5, 10)
    eps, tail, lead = A.build_episodes("FOOUSDT", fills(raw(1, "a", t, "BUY", 2.0, 10)))
    assert eps == [] and tail is not None and tail.max_abs_net == 10
    # anchored replay: 10 already long at the start → the first close is a leading partial, not an episode
    eps, tail, lead = A.build_episodes("FOOUSDT", fills(raw(1, "a", t, "SELL", 2.0, 10, rp=3), raw(2, "b", t + dt.timedelta(minutes=1), "BUY", 2.0, 4),
                                                        raw(3, "c", t + dt.timedelta(minutes=2), "SELL", 2.1, 4, rp=0.4)), start_net=10)
    assert lead == 1 and len(eps) == 1 and eps[0].key == "FOOUSDT:2"


# ── fake exchange + DB harness ───────────────────────────────────────────────────────────────────────────────────────────────────────
class FakeSvc:
    """Production-shaped fake of the recorder's binance_service reads. hidden = trade ids missing from userTrades (lagged / lost fills) while
    the income still has them; extra_income = income rows added as-is (e.g. a TRANSFER sharing a tranId)."""
    def __init__(self, trades, positions=None, funding=(), lev=None, bnb=600.0, bnb_fail=False, bnb_now=650.0, trades_exc=None, hidden=(),
                 extra_income=(), income_time=None, exc_syms=(), uw=10, symcfg_exc=None):
        self.trades, self.positions = list(trades), dict(positions or {})
        self.funding, self.lev, self.bnb, self.bnb_fail, self.bnb_now, self.trades_exc = list(funding), lev or {}, bnb, bnb_fail, bnb_now, trades_exc
        self.hidden, self.extra_income = set(hidden), list(extra_income)
        self.income_time, self.exc_syms, self.uw, self.symcfg_exc = dict(income_time or {}), set(exc_syms), uw, symcfg_exc   # income_time = {trade id: income row time ms}
        self.calls = []

    def used_weight_1m(self):
        return self.uw

    def income(self):
        out, tid = [], 1000
        for r in self.trades:
            tid += 1
            r = dict(r, time=self.income_time.get(r["id"], r["time"]))
            if float(r["commission"]):
                out.append({"symbol": r["symbol"], "incomeType": "COMMISSION", "income": str(-float(r["commission"])), "asset": r["commissionAsset"],
                            "time": r["time"], "tranId": tid, "tradeId": str(r["id"])})
            if float(r["realizedPnl"]):
                tid += 1
                out.append({"symbol": r["symbol"], "incomeType": "REALIZED_PNL", "income": r["realizedPnl"], "asset": "USDT", "time": r["time"],
                            "tranId": tid, "tradeId": str(r["id"])})
        for s, t, x in self.funding:
            tid += 1
            out.append({"symbol": s, "incomeType": "FUNDING_FEE", "income": str(x), "asset": "USDT", "time": MS(t), "tranId": tid})
        return sorted(out + self.extra_income, key=lambda r: r["time"])

    async def fetch_income_page(self, lo, hi, limit=1000, symbol=None, income_type=None):
        self.calls.append(("income", lo, hi, symbol, income_type))
        return [r for r in self.income() if lo <= r["time"] <= hi and (symbol is None or r.get("symbol") == symbol)
                and (income_type is None or r.get("incomeType") == income_type)][:limit]

    async def fetch_user_trades_page(self, sym, lo=None, hi=None, limit=1000, from_id=None):
        self.calls.append(("trades", sym, lo, hi, from_id))
        if self.trades_exc and (not self.exc_syms or sym in self.exc_syms):
            raise self.trades_exc
        rows = sorted([r for r in self.trades if r["symbol"] == sym and r["id"] not in self.hidden], key=lambda r: (r["time"], r["id"]))
        if from_id is not None:
            assert lo is None and hi is None                    # Binance forbids fromId with a time window
            return [r for r in sorted(rows, key=lambda r: r["id"]) if r["id"] >= from_id][:limit]
        return [r for r in rows if lo <= r["time"] <= hi][:limit]

    async def get_net_positions(self):
        self.calls.append(("positions",))
        return dict(self.positions)

    async def get_symbol_leverages(self):
        self.calls.append(("symcfg",))
        if self.symcfg_exc:
            raise self.symcfg_exc
        return dict(self.lev)

    async def fetch_bnb_1m_closes(self, start, limit=1000):
        self.calls.append(("bnb", start))
        if self.bnb_fail:
            raise RuntimeError("klines down")
        return [(start + i * 60_000, self.bnb + (i % 7)) for i in range(limit)]

    async def fetch_bnb_price_raw(self):
        self.calls.append(("bnbnow",))
        if self.bnb_now is None:
            raise RuntimeError("ticker down")
        return self.bnb_now


def order(models, **kw):
    base = dict(pair="BARUSDT", direction="LONG", status="CLOSED", entry_price=1.0, exit_price=1.0, investment=50.0, leverage=20.0,
                notional_value=1000.0, quantity=1000.0, confidence="STRONG_BUY", entry_strategy="FRENZY_LONG", is_paper=False, pnl=1.0,
                pnl_percentage=0.1, peak_pnl=0.0, trough_pnl=0.0, entry_fee=0.0, opened_at=T0, closed_at=T0 + dt.timedelta(minutes=5))
    base.update(kw)
    return models.Order(**base)


def run(svc, setup=None, passes=1, enabled=True, paper=False, ban=False, now=NOW, after=None, commit_fn=None):
    """Run recorder pass(es) on a fresh in-memory DB (era start = a live bot order at T0). Returns (summaries, app rows, state, extra)."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import select
    import models

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        async with S() as db:
            db.add(models.BotState(is_running=False, is_paper_mode=False))
            db.add(order(models))
            if setup:
                await setup(db, models)
            await db.commit()

        async def commit(db):
            if commit_fn:
                await commit_fn(db, models)
            await db.commit()

        async def nosleep(*a):
            return None
        rec = A.AppTradeRecorder(svc=svc, session_factory=S, commit=commit, is_paper=lambda: paper, enabled=lambda: enabled,
                                 now_ms=lambda: MS(now), sleep=nosleep, ban_active=lambda: ban)
        sums = [await rec.run_pass() for _ in range(passes)]
        async with S() as db:
            rows = (await db.execute(select(models.Order).where(models.Order.import_source == A.IMPORT_SOURCE_APP_HISTORY)
                                     .order_by(models.Order.opened_at))).scalars().all()
            bs = (await db.execute(select(models.BotState))).scalar_one()
            state, cur = A.load_state(bs.app_trades_state_json), bs.app_trades_cursor_ms
            extra = (await after(db, models)) if after else None
        await eng.dispose()
        return sums, rows, state, cur, extra
    return asyncio.run(go())


def _app_long(sym="FOOUSDT", t=dt.datetime(2026, 10, 10, 9, 0, 7), asset="USDT", comm=(0.09, 0.0945), maker=(False, True)):
    return [raw(11, "app1", t, "BUY", 2.0, 100, comm=comm[0], asset=asset, maker=maker[0], sym=sym),
            raw(12, "app2", t + dt.timedelta(minutes=30), "SELL", 2.1, 100, rp=10.0, comm=comm[1], asset=asset, maker=maker[1], sym=sym)]


# ── pure app trade ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_pure_app_trade_numbers_funding_apart_and_idempotent():
    t = dt.datetime(2026, 10, 10, 9, 0, 7)
    svc = FakeSvc(_app_long(t=t), funding=[("FOOUSDT", t + dt.timedelta(minutes=10), -0.5), ("FOOUSDT", t + dt.timedelta(hours=5), -9.0)],
                  lev={"FOOUSDT": 10})
    sums, rows, state, cur, _ = run(svc, passes=2)
    assert len(rows) == 1 and sums[0]["recorded"] == 1 and sums[1]["recorded"] == 0               # re-run inserts nothing
    r = rows[0]
    assert (r.pair, r.direction, r.status, r.is_paper, r.entry_strategy, r.confidence) == ("FOOUSDT", "LONG", "CLOSED", False, "MANUAL", "STRONG_BUY")
    assert r.app_trade_key == "FOOUSDT:11" and r.app_fills == 2 and r.app_fee_estimated is False and r.close_reason == "APP_CLOSE"
    assert r.entry_price == 2.0 and r.exit_price == 2.1 and r.quantity == 100 and r.leverage == 10.0
    assert abs(r.notional_value - 200.0) < 1e-9 and abs(r.investment - 20.0) < 1e-9
    assert abs(r.entry_fee - 0.09) < 1e-12 and abs(r.exit_fee - 0.0945) < 1e-12 and abs(r.total_fee - 0.1845) < 1e-12
    assert abs(r.pnl - (10.0 - 0.1845)) < 1e-9 and abs(r.pnl_percentage - (10.0 - 0.1845) / 200.0 * 100) < 1e-9
    assert r.funding_fee_usd == -0.5                                                              # only the funding inside the trade, NOT in pnl
    assert r.opened_at == t and r.closed_at == t + dt.timedelta(minutes=30)
    assert (r.entry_order_type, r.exit_order_type) == ("TAKER", "MAKER")
    assert cur == MS(NOW) and state["era_ms"] == MS(T0)


def test_bnb_commission_at_fill_minute_and_estimate_flag():
    t = dt.datetime(2026, 10, 10, 9, 0, 7)
    svc = FakeSvc(_app_long(t=t, asset="BNB", comm=(0.0002, 0.0003)), bnb=600.0)
    _, rows, _, _, _ = run(svc)
    r = rows[0]
    m_open = MS(t) - MS(t) % 60_000; m_close = m_open + 30 * 60_000
    px = lambda m: 600.0 + ((m % (1000 * 60_000)) // 60_000) % 7                                  # the fake's kline closes (1000-minute blocks)
    assert abs(r.entry_fee - 0.0002 * px(m_open)) < 1e-9 and abs(r.exit_fee - 0.0003 * px(m_close)) < 1e-9
    assert r.app_fee_estimated is False and abs(r.pnl - (10.0 - r.total_fee)) < 1e-9
    # klines unreadable → the CURRENT BNB price, flagged
    _, rows, _, _, _ = run(FakeSvc(_app_long(t=t, asset="BNB", comm=(0.0002, 0.0003)), bnb_fail=True, bnb_now=650.0))
    r = rows[0]
    assert r.app_fee_estimated is True and abs(r.total_fee - 0.0005 * 650.0) < 1e-9 and "CURRENT BNB price" in r.notes
    assert r.leverage == 1.0 and "unreadable → 1" in r.notes                                     # no symbolConfig entry


def test_era_start_bound_and_bot_stopped_period():
    pre = dt.datetime(2026, 9, 28, 10)
    trades = [raw(1, "p1", pre, "BUY", 2.0, 10, sym="FOOUSDT"), raw(2, "p2", pre + dt.timedelta(hours=1), "SELL", 2.2, 10, rp=2.0, sym="FOOUSDT"),
              raw(3, "q1", T0 - dt.timedelta(hours=2), "BUY", 2.0, 10, sym="FOOUSDT"),                 # opened before the era …
              raw(4, "q2", T0 + dt.timedelta(hours=2), "SELL", 2.1, 10, rp=1.0, sym="FOOUSDT")]         # … closed after it
    trades += _app_long(t=dt.datetime(2026, 10, 12, 3, 0, 0))                                          # the bot is stopped (is_running False)
    svc = FakeSvc(trades)
    sums, rows, _, _, _ = run(svc)
    assert [r.app_trade_key for r in rows] == ["FOOUSDT:11"]
    assert all(c[2] >= MS(T0) for c in svc.calls if c[0] == "trades") and all(c[1] >= MS(T0) for c in svc.calls if c[0] == "income")


def test_open_position_anchor_and_busy_symbol():
    t = dt.datetime(2026, 10, 15, 9)
    trades = _app_long(t=dt.datetime(2026, 10, 14, 9)) + [raw(21, "o1", t, "SELL", 3.0, 7, sym="FOOUSDT")]   # still SHORT 7 now
    _, rows, state, _, _ = run(FakeSvc(trades, positions={"FOOUSDT": -7.0}))
    assert [r.app_trade_key for r in rows] == ["FOOUSDT:11"] and "FOOUSDT" in state["open_partials"]
    # a fill 20 s before the position read → the symbol waits for the next pass (nothing recorded, cursor still moves)
    trades = _app_long(t=dt.datetime(2026, 10, 14, 9)) + [raw(31, "z", NOW - dt.timedelta(seconds=20), "BUY", 2.0, 1, sym="FOOUSDT")]
    sums, rows, _, cur, _ = run(FakeSvc(trades, positions={"FOOUSDT": 1.0}))
    assert rows == [] and sums[0]["busy"] == ["FOOUSDT"] and cur == MS(NOW)


def test_backward_extension_finds_the_episode_start():
    """An incremental pass whose window starts mid-episode reads older windows until the replay starts flat."""
    t_open = dt.datetime(2026, 10, 3, 9)                    # 17 days before NOW — far outside cursor − 1 day
    trades = [raw(41, "x1", t_open, "BUY", 2.0, 50), raw(42, "x2", NOW - dt.timedelta(hours=3), "SELL", 2.4, 50, rp=20.0)]

    async def setup(db, models):
        from sqlalchemy import update
        await db.flush()
        await db.execute(update(models.BotState).values(app_trades_cursor_ms=MS(NOW - dt.timedelta(hours=4))))
    svc = FakeSvc(trades)
    _, rows, _, _, _ = run(svc, setup=setup)
    assert len(rows) == 1 and rows[0].app_trade_key == "FOOUSDT:41" and rows[0].opened_at == t_open
    assert min(c[2] for c in svc.calls if c[0] == "trades") < MS(t_open)


# ── mixed episodes ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
BTW = "BTWUSDT"
B_T0 = dt.datetime(2026, 10, 19, 14, 50, 0)
B_T1 = dt.datetime(2026, 10, 19, 15, 0, 0)
B_T2 = dt.datetime(2026, 10, 19, 15, 13, 0)


BTW_AVG = (2714 * 1.00 + 2779 * 1.01) / 5493                   # Binance's average entry of the netted position
BTW_BOOKED = -81.42 - (1.2214 + 1.18)                           # the bot row's real booking: (0.97 − 1.00) × 2714 − its fees


def _btw_fills(with_app=True):
    """Binance's own numbers: realizedPnl of each close = (price − average entry) × qty on the netted position (one-way mode)."""
    f = [raw(101, "B1", B_T0 + dt.timedelta(seconds=1), "BUY", 1.00, 2714, comm=1.2214, sym=BTW)]
    if with_app:
        f.append(raw(102, "A1", B_T1, "BUY", 1.01, 2779, comm=1.26, sym=BTW))
    f.append(raw(103, "B2", B_T2, "SELL", 0.97, 2714, rp=round((0.97 - (BTW_AVG if with_app else 1.0)) * 2714, 6), comm=1.18, sym=BTW))
    if with_app:
        f.append(raw(104, "S1", B_T2 + dt.timedelta(seconds=2), "SELL", 0.969, 2779, rp=round((0.969 - BTW_AVG) * 2779, 6), comm=1.21, sym=BTW))
    return f


def _btw_row(status="CLOSED", pnl=BTW_BOOKED, direction="LONG", notes=None, close_reason="STOP_LOSS", exit_price=0.97):
    """PRODUCTION shape (review I1): the CLOSE Transaction carries the ENTRY order id (the bot never stores its close order id)."""
    async def setup(db, models):
        o = order(models, pair=BTW, direction=direction, status=status, entry_price=1.0, exit_price=exit_price, quantity=2714.0, notional_value=2714.0,
                  investment=135.7, entry_strategy="MANUAL", pnl=(pnl if status == "CLOSED" else None), opened_at=B_T0,
                  closed_at=(B_T2 if status == "CLOSED" else None), binance_order_id="B1", close_reason=close_reason, notes=notes)
        db.add(o)
        await db.flush()
        for act in (f"OPEN_{direction}", f"CLOSE_{direction}"):
            db.add(models.Transaction(order_id=o.id, binance_order_id="B1", pair=BTW, action=act, price=1.0, quantity=2714.0, investment=135.7,
                                      leverage=20.0, notional_value=2714.0, is_paper=False))
    return setup


def test_fully_bot_episode_records_nothing():
    sums, rows, state, _, _ = run(FakeSvc(_btw_fills(with_app=False)), setup=_btw_row())
    assert rows == [] and sums[0]["bot"] == 1 and not state["ambiguous"]


def test_row_booked_before_its_own_fill_is_still_the_bots():
    """A row whose opened_at is minutes before its own entry fill (maker entry / slow booking) is matched by its order ids — still the bot's."""
    async def setup(db, models):
        o = order(models, pair=BTW, quantity=2714.0, entry_strategy="FRENZY_LONG", pnl=-81.42, opened_at=B_T0 - dt.timedelta(minutes=5),
                  closed_at=B_T2, binance_order_id="B1")
        db.add(o)
    sums, rows, state, _, _ = run(FakeSvc(_btw_fills(with_app=False)), setup=setup)
    assert rows == [] and sums[0]["bot"] == 1 and not state["ambiguous"]


def test_btw_app_add_sold_by_the_bots_residual_sweep():
    sums, rows, state, _, _ = run(FakeSvc(_btw_fills(), lev={BTW: 20}), setup=_btw_row())
    assert len(rows) == 1 and sums[0]["shares"] == 1
    r = rows[0]
    fl = _btw_fills()
    ep_total = sum(float(x["realizedPnl"]) for x in fl) - sum(float(x["commission"]) for x in fl)
    assert abs(r.pnl - (ep_total - BTW_BOOKED)) < 1e-9                                            # episode − the row's booked pnl
    assert abs(r.pnl - ((0.969 - 1.01) * 2779 - 1.26 - 1.21)) < 1e-6                              # = the app's real P&L when the booking is right
    assert r.close_reason == "APP_ADD_SOLD_BY_BOT" and r.quantity == 2779 and abs(r.entry_price - 1.01) < 1e-12
    # I1: the bot's close order id is never stored → the exit is the VWAP of ALL closing fills (blended, flagged on the row)
    assert abs(r.exit_price - (0.97 * 2714 + 0.969 * 2779) / 5493) < 1e-12 and "VWAP of ALL closing fills" in r.notes
    assert r.app_trade_key == f"{BTW}:101" and r.opened_at == B_T1 and r.closed_at == B_T2 + dt.timedelta(seconds=2)
    assert abs(r.entry_fee - 1.26) < 1e-12 and abs(r.exit_fee - (1.18 + 1.21) * 2779 / 5493) < 1e-9 and "APP SHARE" in r.notes
    assert abs(r.pnl_percentage - r.pnl / (1.01 * 2779) * 100) < 1e-9
    assert state["exact"][f"{BTW}:101"]["kind"] == "SHARE"


def test_share_refused_when_the_bot_rows_booking_is_off():
    """I2: a bot row booked $40 off Binance (or on an unverified / fallback exit) would be absorbed into the 📱 line → nothing recorded."""
    for kw, why in ((dict(pnl=BTW_BOOKED + 40.0), "booking differs from Binance"),
                    (dict(notes="x · RTP_FILL_UNVERIFIED"), "unverified exit"),
                    (dict(close_reason="EXTERNAL_CLOSE", exit_price=0.95), "fallback price")):
        sums, rows, state, _, _ = run(FakeSvc(_btw_fills()), setup=_btw_row(**kw))
        assert rows == [] and why in json.dumps(state["ambiguous"], ensure_ascii=False), kw
    sums, rows, _, _, _ = run(FakeSvc(_btw_fills()), setup=_btw_row(close_reason="EXTERNAL_CLOSE", exit_price=0.97))   # a real fill price → fine
    assert len(rows) == 1


def test_app_reducing_a_bot_position_is_ambiguous_not_bot(caplog):
    """I3: the bot holds 2714; the app sells 1000 of it 10 minutes before the bot's stop; the bot sells the rest → never silently 'the bot's'."""
    caplog.set_level(logging.WARNING, logger="services.app_trades")
    avg = 1.0
    f = [raw(101, "B1", B_T0 + dt.timedelta(seconds=1), "BUY", 1.00, 2714, comm=1.2214, sym=BTW),
         raw(102, "A9", B_T2 - dt.timedelta(minutes=10), "SELL", 0.99, 1000, rp=round((0.99 - avg) * 1000, 6), comm=0.45, sym=BTW),
         raw(103, "B2", B_T2, "SELL", 0.97, 1714, rp=round((0.97 - avg) * 1714, 6), comm=0.75, sym=BTW)]
    sums, rows, state, _, _ = run(FakeSvc(f), setup=_btw_row())
    assert rows == [] and sums[0]["bot"] == 0 and "reduced the bot's position" in json.dumps(state["ambiguous"], ensure_ascii=False)
    assert "[APP_TRADE_AMBIGUOUS]" in caplog.text
    # the bot's own resting TP filling part of it first (maker, at the row's TP limit) stays the bot's

    async def setup(db, models):
        await _btw_row()(db, models)
        from sqlalchemy import update
        await db.execute(update(models.Order).where(models.Order.pair == BTW).values(resting_tp_price=0.99))
    f[1]["maker"] = True
    sums, rows, state, _, _ = run(FakeSvc(f), setup=setup)
    assert rows == [] and sums[0]["bot"] == 1 and not state["ambiguous"]


def test_row_still_open_defers_then_records_once_closed():
    sums, rows, state, _, _ = run(FakeSvc(_btw_fills()), setup=_btw_row(status="OPEN"))
    assert rows == [] and sums[0]["deferred"] == 1 and f"{BTW}:101" in state["deferred"]
    assert not state["ambiguous"]


def test_ambiguous_records_nothing_and_logs(caplog):
    caplog.set_level(logging.WARNING, logger="services.app_trades")
    trades = [raw(201, "Z1", B_T0 + dt.timedelta(seconds=1), "SELL", 1.0, 100, sym=BTW), raw(202, "Z2", B_T2, "BUY", 0.99, 100, rp=1.0, sym=BTW)]
    sums, rows, state, _, _ = run(FakeSvc(trades), setup=_btw_row(direction="LONG"))   # a LONG bot row over a SHORT episode
    assert rows == [] and sums[0]["ambiguous_new"] == 1 and f"{BTW}:201" in state["ambiguous"]
    assert "[APP_TRADE_AMBIGUOUS]" in caplog.text and "opposite direction" in caplog.text and "NOTHING recorded" in caplog.text
    # app entry not identifiable (the row has no known order ids and the app add sits inside its window) → nothing
    async def setup(db, models):
        db.add(order(models, pair=BTW, quantity=2714.0, entry_strategy="MANUAL", pnl=-81.42, opened_at=B_T0, closed_at=B_T2,
                     binance_order_id=None))
    f = _btw_fills()
    f[1]["time"] = MS(B_T0 + dt.timedelta(seconds=30))                                            # the app add lands next to the row's open
    sums, rows, state, _, _ = run(FakeSvc(f), setup=setup)
    assert rows == [] and "cannot be told apart" in json.dumps(state["ambiguous"])


def test_imported_row_adopts_the_whole_position():
    """A 📥 imported row (DL 328) covers the fills before its import — the episode is the row's, nothing recorded."""
    t = dt.datetime(2026, 10, 18, 9)
    trades = [raw(301, "app", t, "BUY", 2.0, 500, sym=BTW), raw(302, "bot-close", t + dt.timedelta(hours=3), "SELL", 2.1, 500, rp=50.0, sym=BTW)]

    async def setup(db, models):
        db.add(order(models, pair=BTW, quantity=500.0, entry_strategy="MANUAL", import_source="BINANCE_APP", imported_at=t + dt.timedelta(hours=1),
                     opened_at=t + dt.timedelta(hours=1), closed_at=t + dt.timedelta(hours=3), pnl=49.0, binance_order_id=None))
    sums, rows, _, _, _ = run(FakeSvc(trades), setup=setup)
    assert rows == [] and sums[0]["bot"] == 1


# ── ban / kill switch / paper ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_ban_and_rate_limit_stop_the_pass_and_kill_switch():
    from ccxt.base.errors import RateLimitExceeded
    svc = FakeSvc(_app_long(), trades_exc=RateLimitExceeded("binanceusdm 429 Too many requests"))
    sums, rows, _, cur, _ = run(svc)
    assert sums[0]["status"] == "stopped" and rows == [] and cur is None                          # cursor not moved → resumes next cycle
    for kw, st in ((dict(ban=True), "ban"), (dict(enabled=False), "disabled"), (dict(paper=True), "paper")):
        svc = FakeSvc(_app_long())
        sums, rows, _, cur, _ = run(svc, **kw)
        assert sums[0]["status"] == st and rows == [] and svc.calls == [] and cur is None


def test_weight_budget_waits_when_full():
    waits, clock = [], [0.0]

    async def sl(s):
        waits.append(s); clock[0] += s

    async def go():
        b = A.WeightBudget(per_min=60, sleep=sl, clock=lambda: clock[0])
        for _ in range(3):
            await b.spend(30)
    asyncio.run(go())
    assert len(waits) == 1 and 59 < waits[0] < 61


# ── reconcile ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_reconcile_math_apples_to_apples():
    state = {"days": {"2026-10-01": [100.0, -10.0, -2.0, 0], "2026-10-02": [50.0, -5.0, -1.0, 3]},
             "open_partials": {"FOOUSDT": {"trading": 4.0, "funding": -0.5}}, "ambiguous": {"k": {"pnl": -7.0}}, "deferred": {}}
    r = A.reconcile_report(state, T0, db_pnl=120.0, db_funding=-2.0, db_rows=10, app_est_rows=1)
    assert r["binance"] == 132.0 and r["binance_parts"]["trading_net"] == 135.0 and r["db"] == 118.0
    assert r["gap"] == 14.0 and r["gap_trading"] == 15.0 and r["gap_funding"] == -1.0
    assert r["unexplained"] == round(14.0 - (4.0 - 0.5 - 7.0), 2) and r["binance_parts"]["commissions_converted_at_estimated_bnb_price"] == 3
    rows = [{"time": MS(T0) + 10, "incomeType": "COMMISSION", "income": "-0.001", "asset": "BNB"},
            {"time": MS(T0) + 20, "incomeType": "REALIZED_PNL", "income": "5", "asset": "USDT"},
            {"time": MS(T0) + 30, "incomeType": "TRANSFER", "income": "999", "asset": "USDT"}]
    d = A.income_day_sums(rows, lambda a, t: (600.0, False) if a == "BNB" else (1.0, False), MS(T0), MS(NOW))
    assert d == {"2026-10-01": [5.0, -0.6, 0.0, 0]}


def test_reconcile_after_a_pass_and_endpoint():
    import main
    t = dt.datetime(2026, 10, 10, 9, 0, 7)
    svc = FakeSvc(_app_long(t=t), funding=[("FOOUSDT", t + dt.timedelta(minutes=10), -0.5)])

    async def after(db, models):
        old = main.trading_engine.is_paper_mode
        main.trading_engine.is_paper_mode = False
        try:
            return await main.app_trades_reconcile(db=db)
        finally:
            main.trading_engine.is_paper_mode = old
    _, rows, state, _, rep = run(svc, after=after)
    # Binance: rp 10 − fees 0.1845 + funding −0.5; DB: the era bot row (+1.0) + the app row (10 − 0.1845, funding −0.5)
    assert abs(rep["binance"] - (10 - 0.1845 - 0.5)) < 0.006 and abs(rep["db"] - (1.0 + 10 - 0.1845 - 0.5)) < 0.006 and rep["gap"] == -1.0
    assert rep["since"] == T0.isoformat() and rep["cursor"] == NOW.isoformat() and rep["db_parts"]["closed_rows"] == 2


# ── views ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _view_rows(models):
    t = dt.datetime(2026, 10, 18, 10)
    mk = lambda i, es, pnl, **kw: order(models, pair=f"P{i}USDT", entry_strategy=es, pnl=pnl * 10, pnl_percentage=pnl, is_paper=True,
                                         opened_at=t + dt.timedelta(hours=i), closed_at=t + dt.timedelta(hours=i, minutes=9), entry_macro_trend="BULLISH", **kw)
    return [mk(0, "FRENZY_LONG", 0.5), mk(1, "FRENZY_LONG", -0.2), mk(2, "MANUAL", 1.0), mk(3, "MANUAL", -0.4, import_source=A.IMPORT_SOURCE_APP_HISTORY),
            mk(4, "MANUAL", 0.8, import_source=A.IMPORT_SOURCE_APP_HISTORY, app_trade_key="X:1")]


def test_sleeve_strategy_split_and_bot_view_excludes_app_rows():
    import main, models
    from services.origin_view import order_origin
    rows = _view_rows(models)
    sl = {r["sleeve"]: r for r in main._compute_sleeve_performance(rows)}
    assert sl["App"]["n"] == 2 and sl["Manual"]["n"] == 1 and sl["Frenzy-Long"]["n"] == 2 and sl["Total"]["n"] == 5
    assert [r["sleeve"] for r in main._compute_sleeve_performance(rows)][-3:] == ["Manual", "App", "Total"]
    st = {r["strategy"]: r for r in main._compute_strategy_performance(rows)}
    assert st["APP · L"]["n"] == 2 and st["MANUAL · L"]["n"] == 1
    assert all(order_origin(r.entry_strategy, r.confidence) == "MANUAL" for r in rows[2:])
    assert A.app_trades_summary(rows, 3) == {"n": 2, "wins": 1, "wr": 50.0, "pnl": 4.0, "avg_pct": 0.2, "ambiguous_skipped": 3}

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    async def perf(origin):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        main.trading_engine.paper_balance = 10_000.0
        async with S() as db:
            for r in _view_rows(models):
                db.add(r)
            await db.commit()
            out = json.loads(json.dumps(await main._compute_performance(db, origin=origin), default=str))
        await eng.dispose()
        return out
    bot, allv, man = asyncio.run(perf("BOT")), asyncio.run(perf(None)), asyncio.run(perf("MANUAL"))
    assert bot["total_trades"] == 2 and bot["app_trades"]["n"] == 0
    assert allv["total_trades"] == 5 and allv["app_trades"]["n"] == 2 and man["total_trades"] == 3 and man["app_trades"]["n"] == 2
    assert "App" not in {r["sleeve"] for r in bot["sleeve_performance"]}


# ── wiring: exports (D12), config (D11), migration, loop ─────────────────────────────────────────────────────────────────────────────
def test_exports_badge_and_config_wiring():
    assert UI.count("_appTradesReportLine(") == 3                                               # the helper + BOTH text exports
    a, b = UI.index("async function copyAnalytics()"), UI.index("async function copySplitReport()")
    assert "_appTradesReportLine([perf])" in UI[a:a + 6000] and "_appTradesReportLine([perfBull, perfBear])" in UI[b:b + 6000]
    assert "📱 App trades" in UI and "ambiguous episodes skipped" in UI
    assert UI.count('id="config-app-trade-record-enabled"') == 1
    assert "config.app_trade_record_enabled !== false" in UI and "app_trade_record_enabled: (document.getElementById('config-app-trade-record-enabled')" in UI
    assert "o.import_source === 'BINANCE_APP_HISTORY'" in UI and "'App': 'text-sky-300'" in UI
    import config, main
    assert config.TradingConfig().app_trade_record_enabled is True
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["app_trade_record_enabled"] is True
    assert "app_trade_record_enabled" in main.ConfigUpdate.model_fields
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_app_trade_task = asyncio.create_task(_app_trade_loop(lambda: should_stop))" in src and '@app.get("/api/app_trades/reconcile")' in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_inline_scripts_node_check_and_report_line():
    blocks = re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', UI, re.S)
    for i, b in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(b); fn = f.name
        try:
            r = subprocess.run(["node", "--check", fn], capture_output=True, text=True)
            assert r.returncode == 0, f"script block {i}: {r.stderr[:400]}"
        finally:
            os.unlink(fn)
    s = UI.index("function _appTradesReportLine("); e = UI.index("function setOriginView(", s)
    prog = UI[s:e] + ";console.log(JSON.stringify([_appTradesReportLine([{app_trades:{n:2,wins:1,pnl:4,avg_pct:0.2,ambiguous_skipped:3}}]),"
    prog += "_appTradesReportLine([{app_trades:{n:1,wins:1,pnl:5,avg_pct:1.0,ambiguous_skipped:1}},{app_trades:{n:1,wins:0,pnl:-2,avg_pct:-0.5,ambiguous_skipped:1}}])]))"
    one, two = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert one.startswith("📱 App trades: 2 · WR 50.0% · Σ +$4.00 · avg +0.200% · ambiguous episodes skipped: 3")
    assert "(BULLISH + BEARISH regimes): 2 · WR 50.0% · Σ +$3.00 · avg +0.250%" in two


def test_migration_adds_columns_and_unique_index(tmp_path, monkeypatch):
    import models, database
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'old.db'}")

    async def go():
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
            await c.execute(text("DROP INDEX ix_orders_app_trade_key"))
            for col in ("app_trade_key", "app_fee_estimated", "app_fills"):
                await c.execute(text(f"ALTER TABLE orders DROP COLUMN {col}"))
            for col in ("app_trades_cursor_ms", "app_trades_state_json"):
                await c.execute(text(f"ALTER TABLE bot_state DROP COLUMN {col}"))
        monkeypatch.setattr(database, "engine", eng)
        await database.init_db()
        async with eng.begin() as c:
            ocols = [r[1] for r in (await c.execute(text("PRAGMA table_info(orders)"))).all()]
            bcols = [r[1] for r in (await c.execute(text("PRAGMA table_info(bot_state)"))).all()]
            idx = {r[1]: r[2] for r in (await c.execute(text("PRAGMA index_list(orders)"))).all()}
        await eng.dispose()
        return ocols, bcols, idx
    ocols, bcols, idx = asyncio.run(go())
    assert {"app_trade_key", "app_fee_estimated", "app_fills"} <= set(ocols) and {"app_trades_cursor_ms", "app_trades_state_json"} <= set(bcols)
    assert idx.get("ix_orders_app_trade_key") == 1
    cols = [c.name for c in models.Order.__table__.columns]                                       # D12: the orders CSV dumps every column
    assert {"app_trade_key", "app_fee_estimated", "app_fills", "import_source"} <= set(cols)


def test_loop_survives_a_failing_pass(monkeypatch):
    calls = []

    class R:
        async def run_pass(self):
            calls.append(1)
            raise RuntimeError("boom")
    monkeypatch.setattr(A, "APP_FIRST_DELAY_S", 0)
    monkeypatch.setattr(A, "APP_INTERVAL_S", 0)
    stop = lambda: len(calls) >= 2
    asyncio.run(A.app_trade_loop(stop, recorder=R()))
    assert len(calls) == 2


# ── review round 1 (C1, I4, I5, I6, minor) ───────────────────────────────────────────────────────────────────────────────────────────
def test_income_dedup_keeps_rows_sharing_a_tranid_across_types():
    """C1: a TRANSFER and a COMMISSION with the SAME tranId are two rows — the commission must reach the reconcile sums."""
    t = dt.datetime(2026, 10, 10, 9, 0, 7)
    trades = _app_long(t=t)
    svc = FakeSvc(trades)
    clash = svc.income()[0]["tranId"]                                                             # the first COMMISSION row's tranId
    svc.extra_income = [{"symbol": "", "incomeType": "TRANSFER", "income": "500", "asset": "USDT", "time": MS(t) + 5, "tranId": clash}]
    _, rows, state, _, _ = run(svc)
    assert len(rows) == 1
    day = state["days"][t.strftime("%Y-%m-%d")]
    assert abs(day[1] - (-(0.09 + 0.0945))) < 1e-9 and abs(day[0] - 10.0) < 1e-9                 # both commissions kept, transfer not counted
    assert A.income_key({"incomeType": "TRANSFER", "tranId": 7}) != A.income_key({"incomeType": "COMMISSION", "tranId": 7})


def test_missing_fill_skips_the_symbol(caplog):
    """I4: a fill the income has but userTrades did not return (lag / loss) → the symbol is skipped (retried), nothing recorded."""
    caplog.set_level(logging.WARNING, logger="services.app_trades")
    t = dt.datetime(2026, 10, 10, 9)
    trades = [raw(1, "o1", t, "BUY", 2.0, 10, comm=0.01), raw(2, "o2", t + dt.timedelta(minutes=1), "BUY", 2.0, 10, comm=0.01),
              raw(3, "o3", t + dt.timedelta(minutes=2), "SELL", 2.1, 10, rp=1.0, comm=0.01),
              raw(4, "o4", t + dt.timedelta(minutes=3), "SELL", 2.1, 10, rp=1.0, comm=0.01)]
    # repeat-size: without fill 2 the replay would see buy 10 / sell 10 = a FALSE flat and a fake short episode after it
    sums, rows, state, _, _ = run(FakeSvc(trades, hidden={2}))
    assert rows == [] and "FOOUSDT" in state["retry"] and "[APP_TRADE_FILL_MISMATCH]" in caplog.text and "not in the fills read" in caplog.text
    # a missing CLOSING fill with no commission is caught by the realized P&L sum
    trades[3]["commission"] = "0"
    sums, rows, state, _, _ = run(FakeSvc(trades, hidden={4}))
    assert rows == [] and "FOOUSDT" in state["retry"] and "trade ids 4" in caplog.text
    bad = A.fill_income_bad_windows("FOOUSDT", [], [{"symbol": "FOOUSDT", "incomeType": "REALIZED_PNL", "income": "1.0", "time": 5}], 0, 10,
                                    lambda a, t: (1.0, False))
    assert bad and "without a tradeId" in bad[0][2]                                               # an unverifiable income row → that window is bad
    # all fills present → one episode, recorded once
    sums, rows, state, _, _ = run(FakeSvc(trades))
    assert len(rows) == 1 and rows[0].quantity == 20 and not state["retry"]


def test_user_trades_paging_is_lossless(monkeypatch):
    """I4: a full page continues with fromId (never skips same-millisecond fills)."""
    monkeypatch.setattr(A, "PAGE", 2)
    t = dt.datetime(2026, 10, 10, 9)
    trades = [raw(1, "o1", t, "BUY", 2.0, 5), raw(2, "o2", t, "BUY", 2.0, 5), raw(3, "o3", t, "BUY", 2.0, 5),       # same millisecond
              raw(4, "o4", t + dt.timedelta(minutes=5), "SELL", 2.2, 15, rp=3.0)]
    svc = FakeSvc(trades)
    _, rows, _, _, _ = run(svc)
    assert len(rows) == 1 and rows[0].quantity == 15 and rows[0].app_fills == 4
    assert any(c[0] == "trades" and c[4] is not None for c in svc.calls)


def test_recorder_reads_use_the_dedicated_client():
    """I5: every recorder read runs on binance_service's OWN recorder client — never the trading client's throttle queue."""
    from services.binance_service import BinanceService

    class Boom:
        def __getattr__(self, n):
            raise AssertionError(f"trading/public client used: {n}")

    class Rec:
        last_response_headers = {"X-MBX-USED-WEIGHT-1M": "123"}

        def __init__(self): self.calls = []

        async def fapiPrivateGetIncome(self, q): self.calls.append(("income", q)); return [{"tranId": 1}]
        async def fapiPrivateGetUserTrades(self, q): self.calls.append(("trades", q)); return []
        async def fapiPrivateGetSymbolConfig(self): self.calls.append(("symcfg",)); return [{"symbol": "FOOUSDT", "leverage": "10"}]
        async def fapiPrivateV3GetPositionRisk(self): self.calls.append(("pos",)); return [{"symbol": "FOOUSDT", "positionAmt": "-3"}]
        async def fapiPublicGetKlines(self, q): self.calls.append(("kl", q)); return [[60000, "1", "1", "1", "600.5", "1"]]
        async def fapiPublicGetTickerPrice(self, q): self.calls.append(("px", q)); return {"price": "601"}

    svc = BinanceService()
    real = (svc.exchange, svc.public_exchange, svc.spot_exchange)
    rec = Rec()
    svc.exchange = svc.public_exchange = svc.spot_exchange = Boom()
    svc.recorder_exchange = rec

    async def go():
        assert await svc.fetch_income_page(1, 2, 1000, "FOOUSDT", "FUNDING_FEE") == [{"tranId": 1}]
        await svc.fetch_user_trades_page("FOOUSDT", 1, 2)
        await svc.fetch_user_trades_page("FOOUSDT", None, None, 1000, 77)
        assert await svc.get_symbol_leverages() == {"FOOUSDT": 10}
        assert await svc.get_net_positions() == {"FOOUSDT": -3.0}
        assert await svc.fetch_bnb_1m_closes(60000, 1) == [(60000, 600.5)]
        assert await svc.fetch_bnb_price_raw() == 601.0
    asyncio.run(go())
    assert svc.used_weight_1m() == 123
    assert rec.calls[0] == ("income", {"startTime": 1, "endTime": 2, "limit": 1000, "symbol": "FOOUSDT", "incomeType": "FUNDING_FEE"})
    assert rec.calls[2] == ("trades", {"symbol": "FOOUSDT", "limit": 1000, "fromId": 77})                # fromId never with a time window
    assert svc._recorder_client() is rec and len(rec.calls) == 7

    async def close():
        for c in real:
            await c.close()
    asyncio.run(close())


def test_recorder_limit_error_reaches_the_shared_ban_state(monkeypatch):
    """I5: a 418 from the recorder's client still sets the bot-wide ban (BinanceService._detect_ban) and stops the pass."""
    from services import binance_service as bs
    from ccxt.base.errors import DDoSProtection
    monkeypatch.setattr(bs, "_ban_until", 0)
    monkeypatch.setattr(bs, "_ban_persist_callback", None)
    svc = FakeSvc(_app_long(), trades_exc=DDoSProtection('binanceusdm 418 {"code":-1003,"msg":"Way too many requests; IP banned until 9999999999999."}'))
    sums, rows, _, cur, _ = run(svc)
    assert sums[0]["status"] == "stopped" and rows == [] and cur is None and bs._ban_until == 9999999999.999


def test_exact_check_per_row_and_in_reconcile(caplog):
    """I6: each pure APP row vs its symbol's income over [opened_at, closed_at]; a mismatch is named by id in the endpoint and the log."""
    import main
    caplog.set_level(logging.INFO, logger="services.app_trades")
    t_open = dt.datetime(2026, 10, 3, 9)
    trades = [raw(41, "x1", t_open, "BUY", 2.0, 50), raw(42, "x2", NOW - dt.timedelta(hours=3), "SELL", 2.4, 50, rp=20.0)]
    trades += _app_long(sym="BAZUSDT", t=NOW - dt.timedelta(hours=2))                              # inside the pass window → exact

    async def setup(db, models):
        from sqlalchemy import update
        await db.flush()
        await db.execute(update(models.BotState).values(app_trades_cursor_ms=MS(NOW - dt.timedelta(hours=4))))

    async def after(db, models):
        main.trading_engine.is_paper_mode = False
        try:
            return await main.app_trades_reconcile(db=db)
        finally:
            main.trading_engine.is_paper_mode = True
    # (round-2 C) an income row of the old episode's OWN fill (tradeId 41) that the fill does not show → its exact check fails → NOT written
    svc = FakeSvc(trades, extra_income=[{"symbol": "FOOUSDT", "incomeType": "REALIZED_PNL", "income": "1.5", "asset": "USDT",
                                         "time": MS(t_open), "tranId": 99, "tradeId": "42"}])
    _, rows, state, _, rep = run(svc, setup=setup, after=after)
    assert [r.app_trade_key for r in rows] == ["BAZUSDT:11"]
    amb = state["ambiguous"]["FOOUSDT:41"]["reason"]
    assert "exact check failed" in amb and "Δ -1.5000" in amb
    ex = rep["app_rows_exact"]
    assert ex["n_exact"] == 1 and ex["n_mismatched"] == 0 and ex["rows"][0]["id"] == rows[0].id and ex["rows"][0]["ok"] is True
    assert "1 exact / 0 mismatched (none)" in caplog.text and rep["account_gap_noise_sources"] and "[APP_TRADE_AMBIGUOUS]" in caplog.text
    # the report names any stored mismatch by id (rows written before round 2 / by hand)
    es = A.exact_summary({"exact": {"K:1": {"kind": "APP", "ok": False, "row_pnl": 1.0, "income_pnl": 2.0}}}, {"K:1": 7})
    assert es["mismatched_ids"] == [7] and es["n_mismatched"] == 1


def test_app_rows_never_change_bot_behaviour_source_pins():
    """MINOR: 📱 rows never start the bot's same-pair re-entry cooldown nor the post-exit restart pick-up (🖐 rows unchanged)."""
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("cooldown_threshold = datetime.utcnow() - timedelta(minutes=cooldown_minutes)")
    assert "Order.import_source != 'BINANCE_APP_HISTORY'" in eng[i:i + 1200]
    j = eng.index("Order.post_exit_peak_pnl.is_(None),")
    assert "Order.import_source != 'BINANCE_APP_HISTORY'" in eng[j:j + 300]


# ── review round 2 (A–I) ─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_flip_through_zero_both_rows_exact():
    """A: long 100 → SELL 150 (closes 100, opens short 50) → BUY 50: two 📱 rows, each exact against its OWN tradeIds (split fill pro rata)."""
    t = dt.datetime(2026, 10, 10, 9)
    trades = [raw(1, "o1", t, "BUY", 2.0, 100, comm=0.09), raw(2, "o2", t + dt.timedelta(minutes=5), "SELL", 2.1, 150, rp=10.0, comm=0.14),
              raw(3, "o3", t + dt.timedelta(minutes=9), "BUY", 2.0, 50, rp=5.0, comm=0.045)]
    _, rows, state, _, _ = run(FakeSvc(trades))
    assert [(r.app_trade_key, r.direction, r.quantity) for r in rows] == [("FOOUSDT:1", "LONG", 100), ("FOOUSDT:2", "SHORT", 50)]
    assert abs(rows[0].pnl - (10.0 - 0.09 - 0.14 * 100 / 150)) < 1e-9 and abs(rows[1].pnl - (5.0 - 0.14 * 50 / 150 - 0.045)) < 1e-9
    assert state["exact"]["FOOUSDT:1"]["ok"] is True and state["exact"]["FOOUSDT:2"]["ok"] is True
    assert abs(state["exact"]["FOOUSDT:1"]["income_pnl"] - rows[0].pnl) < 1e-6


def test_bot_stop_filled_before_its_booking_is_the_bots():
    """B: the bot is down; its exchange stop sells everything at t+5 min, the row is booked at t+7 min → BOT (a flatten carries no app share)."""
    t = dt.datetime(2026, 10, 12, 9)
    f = [raw(1, "E1", t, "BUY", 2.0, 100, comm=0.09, sym=BTW), raw(2, "STOP", t + dt.timedelta(minutes=5), "SELL", 1.94, 100, rp=-6.0, comm=0.09, sym=BTW)]

    async def setup(db, models):
        db.add(order(models, pair=BTW, quantity=100.0, pnl=-6.18, opened_at=t, closed_at=t + dt.timedelta(minutes=7), binance_order_id="E1",
                     close_reason="BACKSTOP_STOP"))
    sums, rows, state, _, _ = run(FakeSvc(f), setup=setup)
    assert rows == [] and sums[0]["bot"] == 1 and not state["ambiguous"]


def test_external_close_booked_at_the_vwap_of_a_multi_fill_exit():
    """E: the bot's market exit filled in two pieces; the reconciler booked their VWAP → a real exit, the app share is recorded."""
    f = _btw_fills()
    b2 = f[2]
    f[2:3] = [dict(b2, id=103, qty="1000", realizedPnl=str(round((0.97 - BTW_AVG) * 1000, 6)), commission="0.43"),
              dict(b2, id=105, orderId="B2", price="0.9690", qty="1714", realizedPnl=str(round((0.969 - BTW_AVG) * 1714, 6)), commission="0.75")]
    vw = (0.97 * 1000 + 0.969 * 1714) / 2714
    booked = (vw - 1.0) * 2714 - 1.2214 - 1.18
    sums, rows, state, _, _ = run(FakeSvc(f), setup=_btw_row(close_reason="EXTERNAL_CLOSE", exit_price=round(vw, 7), pnl=booked))
    assert len(rows) == 1 and not state["ambiguous"]
    assert any(abs(p - vw) < 1e-12 for p in A.close_price_candidates(A.build_episodes(BTW, fills(*f))[0][0]))


def test_fill_read_padding_at_lo_and_one_bad_day_does_not_block_later_ones(caplog):
    """F: a fill 30 s before lo whose income row lands just after lo pairs up (fills read 2 min before lo); a bad day only holds back
    the episodes before it."""
    async def setup(db, models):
        from sqlalchemy import update
        await db.flush()
        await db.execute(update(models.BotState).values(app_trades_cursor_ms=MS(dt.datetime(2026, 10, 20, 1, 0))))   # lo = Oct-19 00:00
    lo = dt.datetime(2026, 10, 19)
    trades = [raw(1, "a", lo - dt.timedelta(seconds=30), "BUY", 2.0, 10, comm=0.01), raw(2, "b", dt.datetime(2026, 10, 20, 0, 55), "SELL", 2.1, 10, rp=1.0, comm=0.01)]
    _, rows, state, _, _ = run(FakeSvc(trades, income_time={1: MS(lo) + 5_000}), setup=setup)
    assert [r.app_trade_key for r in rows] == ["FOOUSDT:1"] and not state["retry"] and state["exact"]["FOOUSDT:1"]["ok"] is True
    # backfill: Oct-5 episode has a hidden fill (bad day) — the Oct-12 episode is still recorded, the symbol retries from Oct-5
    caplog.set_level(logging.WARNING, logger="services.app_trades")
    t5, t12 = dt.datetime(2026, 10, 5, 9), dt.datetime(2026, 10, 12, 9)
    trades = [raw(21, "x", t5, "BUY", 2.0, 10, comm=0.01), raw(22, "y", t5 + dt.timedelta(minutes=1), "BUY", 2.0, 10, comm=0.01),
              raw(23, "z", t5 + dt.timedelta(minutes=2), "SELL", 2.1, 20, rp=2.0, comm=0.02)] + _app_long(t=t12)
    _, rows, state, _, _ = run(FakeSvc(trades, hidden={22}))
    assert [r.app_trade_key for r in rows] == ["FOOUSDT:11"] and rows[0].opened_at == t12
    assert A.retry_point(state["retry"]["FOOUSDT"]) == MS(dt.datetime(2026, 10, 5)) and state["retry"]["FOOUSDT"]["first_seen_ms"] == MS(NOW) and "[APP_TRADE_FILL_MISMATCH]" in caplog.text


def test_retry_reads_from_its_own_point_global_read_stays():
    """G: a retried symbol re-reads ITS income from its retry point; the global income read stays at cursor − 1 day."""
    async def setup(db, models):
        from sqlalchemy import update
        await db.flush()
        await db.execute(update(models.BotState).values(app_trades_cursor_ms=MS(NOW - dt.timedelta(hours=1)),
                                                        app_trades_state_json=json.dumps({"era_ms": MS(T0), "retry": {"FOOUSDT": MS(dt.datetime(2026, 10, 10, 7))}})))
    svc = FakeSvc(_app_long(t=dt.datetime(2026, 10, 12, 9)))
    _, rows, state, _, _ = run(svc, setup=setup)
    inc = [c for c in svc.calls if c[0] == "income"]
    assert all(c[1] >= MS(dt.datetime(2026, 10, 19)) for c in inc if c[3] is None)                # global: cursor − 1 day, midnight
    assert any(c[3] == "FOOUSDT" and c[1] == MS(dt.datetime(2026, 10, 10)) for c in inc)          # the symbol's own read
    assert [r.app_trade_key for r in rows] == ["FOOUSDT:11"] and not state["retry"]


def test_stop_after_a_write_keeps_its_exact_entry_and_backfills_a_missing_one():
    """H: a 429 on a later symbol after a row was committed → the state (with that row's exact entry) is saved, cursor not moved; a row
    that exists without an exact entry gets one on the next read."""
    from ccxt.base.errors import RateLimitExceeded
    trades = _app_long(sym="AAAUSDT") + [raw(31, "z", dt.datetime(2026, 10, 11), "BUY", 1.0, 1, comm=0.001, sym="ZZZUSDT")]
    sums, rows, state, cur, _ = run(FakeSvc(trades, trades_exc=RateLimitExceeded("429"), exc_syms={"ZZZUSDT"}))
    assert sums[0]["status"] == "stopped" and cur is None and [r.app_trade_key for r in rows] == ["AAAUSDT:11"]
    assert state["exact"]["AAAUSDT:11"]["ok"] is True

    async def setup(db, models):
        db.add(order(models, pair="FOOUSDT", entry_strategy="MANUAL", import_source=A.IMPORT_SOURCE_APP_HISTORY, app_trade_key="FOOUSDT:11",
                     pnl=10.0 - 0.1845, opened_at=dt.datetime(2026, 10, 10, 9, 0, 7), closed_at=dt.datetime(2026, 10, 10, 9, 30, 7)))
    sums, rows, state, _, _ = run(FakeSvc(_app_long()), setup=setup)
    assert sums[0]["recorded"] == 0 and len(rows) == 1 and state["exact"]["FOOUSDT:11"]["ok"] is True


def test_header_pause_and_early_partial_bot_close():
    """I: X-MBX-USED-WEIGHT-1M above 1,800 → a 60 s pause; a bot-only episode whose early partial close is the bot's resting TP stays BOT
    (covered in the I3 test) and a bot flatten before its booking stays BOT (round-2 B test)."""
    slept = []

    async def sl(x):
        slept.append(x)
    svc = FakeSvc([], uw=1900)
    rec = A.AppTradeRecorder(svc=svc, session_factory=object(), commit=object(), is_paper=lambda: False, enabled=lambda: True,
                             now_ms=lambda: MS(NOW), sleep=sl, ban_active=lambda: False)
    assert asyncio.run(rec._call(1, svc.fetch_bnb_price_raw)) == 650.0 and slept == [60]
    svc.uw = 100
    asyncio.run(rec._call(1, svc.fetch_bnb_price_raw))
    assert slept == [60]


# ── review round 3 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def _with_state(state, cursor=None):
    async def setup(db, models):
        from sqlalchemy import update
        await db.flush()
        await db.execute(update(models.BotState).values(app_trades_cursor_ms=cursor, app_trades_state_json=json.dumps(state)))
    return setup


def test_stop_after_the_retry_pop_keeps_the_retry_point():
    """1: the symbol passes its guard (retry popped), then a 429 on the leverage read → the saved state still holds the retry point."""
    from ccxt.base.errors import RateLimitExceeded
    pt = MS(dt.datetime(2026, 10, 10))
    st = {"era_ms": MS(T0), "retry": {"FOOUSDT": {"point": pt, "first_seen_ms": MS(NOW) - 3_600_000}}}
    sums, rows, state, cur, _ = run(FakeSvc(_app_long(t=dt.datetime(2026, 10, 12, 9)), symcfg_exc=RateLimitExceeded("429")),
                                    setup=_with_state(st, MS(NOW - dt.timedelta(hours=1))))
    assert sums[0]["status"] == "stopped" and rows == [] and cur == MS(NOW - dt.timedelta(hours=1))
    assert state["retry"]["FOOUSDT"] == {"point": pt, "first_seen_ms": MS(NOW) - 3_600_000}


def test_exact_check_tolerates_an_income_row_just_after_its_fill():
    """2: the close's income rows land 30 s after the fill (and after the episode end) → still exact, the row is written; funding tight."""
    t = dt.datetime(2026, 10, 10, 9, 0, 7)
    close = t + dt.timedelta(minutes=30)
    svc = FakeSvc(_app_long(t=t), income_time={12: MS(close) + 30_000}, funding=[("FOOUSDT", close + dt.timedelta(seconds=40), -3.0)])
    _, rows, state, _, _ = run(svc)
    assert len(rows) == 1 and state["exact"]["FOOUSDT:11"]["ok"] is True and rows[0].funding_fee_usd == 0.0   # funding after the close: not the trade's


def test_retry_ages_on_first_seen_and_a_dropped_retry_is_ambiguous():
    """3: a backfill point weeks old but listed an hour ago is kept; one listed 15 days ago is dropped into `ambiguous`."""
    old_pt = MS(dt.datetime(2026, 10, 2))
    st = {"era_ms": MS(T0), "retry": {"BARXUSDT": {"point": old_pt, "first_seen_ms": MS(NOW) - 3_600_000}}}
    _, _, state, _, _ = run(FakeSvc([], exc_syms={"BARXUSDT"}, trades_exc=RuntimeError("down")), setup=_with_state(st, MS(NOW - dt.timedelta(hours=1))))
    assert A.retry_point(state["retry"]["BARXUSDT"]) == old_pt and not state["ambiguous"]
    st["retry"]["BARXUSDT"]["first_seen_ms"] = MS(NOW - dt.timedelta(days=15))
    _, _, state, _, _ = run(FakeSvc([]), setup=_with_state(st, MS(NOW - dt.timedelta(hours=1))))
    assert "BARXUSDT" not in state["retry"] and state["ambiguous"][f"BARXUSDT:retry:{old_pt}"]["reason"] == "dropped after 14 days of retries"


def test_multi_fill_flatten_before_booking_is_the_bots():
    """4: BUY 100; the stop order fills 60 + 40; the row is booked 120 s later → BOT (every fill of the flattening order is exempt)."""
    t = dt.datetime(2026, 10, 12, 9)
    f = [raw(1, "E1", t, "BUY", 2.0, 100, comm=0.09, sym=BTW),
         raw(2, "STOP", t + dt.timedelta(minutes=5), "SELL", 1.94, 60, rp=-3.6, comm=0.05, sym=BTW),
         raw(3, "STOP", t + dt.timedelta(minutes=5, milliseconds=3), "SELL", 1.939, 40, rp=-2.44, comm=0.04, sym=BTW)]

    async def setup(db, models):
        db.add(order(models, pair=BTW, quantity=100.0, pnl=-6.22, opened_at=t, closed_at=t + dt.timedelta(minutes=7), binance_order_id="E1",
                     close_reason="BACKSTOP_STOP"))
    sums, rows, state, _, _ = run(FakeSvc(f), setup=setup)
    assert rows == [] and sums[0]["bot"] == 1 and not state["ambiguous"]


def test_existing_key_reading_bot_now_gets_no_exact_entry():
    """5: a 📱 row exists for an episode that now reads as the bot's → no exact entry is invented for it."""
    async def setup(db, models):
        await _btw_row()(db, models)
        db.add(order(models, pair=BTW, entry_strategy="MANUAL", import_source=A.IMPORT_SOURCE_APP_HISTORY, app_trade_key=f"{BTW}:101",
                     pnl=1.0, opened_at=B_T0, closed_at=B_T2))
    _, rows, state, _, _ = run(FakeSvc(_btw_fills(with_app=False)), setup=setup)
    assert len(rows) == 1 and f"{BTW}:101" not in state["exact"]


def test_a_non_key_write_error_is_logged_and_kept_as_ambiguous(caplog):
    """6: an IntegrityError that is not the app_trade_key duplicate → WARNING + ambiguous with the error (never a silent skip)."""
    from sqlalchemy.exc import IntegrityError
    caplog.set_level(logging.WARNING, logger="services.app_trades")

    async def boom(db, models):
        if any(isinstance(o, models.Order) and o.import_source == A.IMPORT_SOURCE_APP_HISTORY for o in db.new):
            raise IntegrityError("INSERT", {}, Exception("NOT NULL constraint failed: orders.pair"))
    _, rows, state, _, _ = run(FakeSvc(_app_long()), commit_fn=boom)
    assert rows == [] and "row NOT written" in caplog.text and "write failed" in state["ambiguous"]["FOOUSDT:11"]["reason"]


def test_retry_age_survives_a_pop_then_a_new_failure():
    """final gate: a symbol whose retry entry is popped (guard passed) and that fails again later in the same pass keeps its original
    first_seen_ms — the 14-day drop can never be postponed forever."""
    from services.app_trades import retry_add
    st = {"retry": {}}
    retry_add(st, "IOSTUSDT", 1_000, 5_000, first_seen_ms=1_234)
    assert st["retry"]["IOSTUSDT"] == {"point": 1_000, "first_seen_ms": 1_234}
    retry_add(st, "IOSTUSDT", 900, 9_999)
    assert st["retry"]["IOSTUSDT"] == {"point": 900, "first_seen_ms": 1_234}
    st2 = {"retry": {}}
    retry_add(st2, "BTWUSDT", 50, 7_000)
    assert st2["retry"]["BTWUSDT"]["first_seen_ms"] == 7_000
    import inspect, services.app_trades as A
    src = inspect.getsource(A)
    assert src.count("retry_add(state, sym, sym_lo, tp, _retry_first)") == 1 and src.count("retry_add(state, sym, ep.start_ms, tp, _retry_first)") == 1
