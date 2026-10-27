"""🎨 DECISION_LOG 386: the Total Portfolio Value sparkline inputs on /api/pnl-calendar — a mid-run PAPER top-up (Settings edit of
paper_balance / paper_bnb_initial_usd, kept in ConfigChangeLog) is a flow: it lands in `flows` with its time and is stripped from the
day-return base (the day before the top-up is judged on the small account), an edit before the first paper order is starting capital
(ignored), hourly realized P&L buckets (`hours`, last 8 days) and `total_now` ride the payload. In-memory SQLite, no network."""
import asyncio
import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from test_origin_view import _order  # noqa: E402


def _cal(changes, realized_equity, total_equity=None):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def _pc(db, fresh=False):
        return {"realized_equity": realized_equity, "total_equity": total_equity if total_equity is not None else realized_equity}

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        _was_paper = main.trading_engine.is_paper_mode
        main.trading_engine.is_paper_mode = True
        _orig = main._portfolio_components
        main._portfolio_components = _pc
        d0 = dt.datetime.utcnow().replace(hour=1, minute=0, second=0, microsecond=0) - dt.timedelta(days=3)
        async with S() as db:
            for i, p in enumerate((-3.0, 0.5, 0.4)):                     # −$300 on day 0, then +$50 / +$40
                t = d0 + dt.timedelta(days=i)
                db.add(_order(models, i, "SPIKE_FADE", "SHORT", p, opened_at=t, closed_at=t + dt.timedelta(minutes=20)))
            for when, field, old, new in changes:
                db.add(models.ConfigChangeLog(field=field, old_value=old, new_value=new, changed_at=when(d0)))
            await db.commit()
            try:
                out = json.loads(json.dumps(await main.get_pnl_calendar(tz_offset_min=0, origin=None, db=db), default=str))
            finally:
                main._portfolio_components = _orig
                main.trading_engine.is_paper_mode = _was_paper
        await eng.dispose()
        return out, d0
    return asyncio.run(run())


def test_paper_topup_is_a_flow_not_a_return():
    topup = lambda d0: d0 + dt.timedelta(days=1, hours=12)               # day 1 12:00 UTC, after the day-0 loss
    seed = lambda d0: d0 - dt.timedelta(days=5)                          # before the first paper order → starting capital
    out, d0 = _cal([(seed, "paper_balance", "1000", "2900"), (topup, "paper_balance", "2900", "10000")], realized_equity=2900 - 300 + 50 + 40 + 7100)
    assert out["flows"] == [{"t": int(topup(d0).replace(tzinfo=dt.timezone.utc).timestamp() * 1000), "amount": 7100.0}]
    day0 = (d0).date().isoformat()
    # day 0 is judged on the 2,900 account (−300 / 2,900 = −10.34 %), not on the topped-up 10,000 (−3 %)
    assert out["days"][day0]["day_return_pct"] == round(-300 / 2900 * 100, 2)


def test_bnb_seed_edit_counts_and_garbage_is_skipped():
    later = lambda d0: d0 + dt.timedelta(days=2, hours=6)
    out, d0 = _cal([(later, "paper_bnb_initial_usd", "100", "150"), (later, "paper_balance", "abc", "10000")],
                   realized_equity=2900 - 300 + 50 + 40 + 50)
    assert [f["amount"] for f in out["flows"]] == [50.0]


def test_hours_and_total_now():
    out, d0 = _cal([], realized_equity=2690.0, total_equity=2650.0)
    assert out["total_now"] == 2650.0 and out["flows"] == []
    hrs = {h["t"]: h for h in out["hours"]}
    k = int((d0 + dt.timedelta(minutes=20)).replace(minute=0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert hrs[k]["trades"] == 1 and hrs[k]["pnl"] == -300.0
    assert sum(h["trades"] for h in out["hours"]) == 3


def _cal_fixed(trades, changes, realized_equity):
    """fixed dates (2026-09) so the EOM month walk is exercised deterministically. trades = [(closed_at, pnl_pct)] at $100 / pct."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def _pc(db, fresh=False):
        return {"realized_equity": realized_equity, "total_equity": realized_equity}

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        _was_paper = main.trading_engine.is_paper_mode
        main.trading_engine.is_paper_mode = True
        _orig = main._portfolio_components
        main._portfolio_components = _pc
        async with S() as db:
            for i, (t, p) in enumerate(trades):
                db.add(_order(models, i, "SPIKE_FADE", "SHORT", p, opened_at=t - dt.timedelta(minutes=10), closed_at=t))
            for when, field, old, new in changes:
                db.add(models.ConfigChangeLog(field=field, old_value=old, new_value=new, changed_at=when))
            await db.commit()
            try:
                out = json.loads(json.dumps(await main.get_pnl_calendar(tz_offset_min=0, origin=None, db=db), default=str))
            finally:
                main._portfolio_components = _orig
                main.trading_engine.is_paper_mode = _was_paper
        await eng.dispose()
        return out
    return asyncio.run(run())


SEP = [(dt.datetime(2026, 9, d, 15, 0), 0.5) for d in range(1, 9)]          # 8 active days, +$50 each


def test_topup_never_reads_as_month_growth():
    plain = _cal_fixed(SEP, [], realized_equity=2900 + 400)
    topped = _cal_fixed(SEP, [(dt.datetime(2026, 9, 5, 12, 0), "paper_balance", "2900", "10000")], realized_equity=2900 + 400 + 7100)
    rp, rt = plain["revisions_by_month"]["2026-09"], topped["revisions_by_month"]["2026-09"]
    assert len(rp) == len(rt) and len(rp) >= 4
    # the top-up (09-05 12:00, before that day's 15:00 close) only changes later days' base — the month rate stays small, never the
    # ≈ 4.7 %/day a capital jump read as before the fix
    assert all(abs(b["dcr"]) < 2.0 for b in rt)
    assert max(abs(a["dcr"] - b["dcr"]) for a, b in zip(rp, rt)) < 1.0
    assert rt[-1]["bal"] == round(2900 + 400 + 7100, 2)


def test_flow_day_return_is_exact_time_weighted():
    d = dt.datetime(2026, 9, 3)
    trades = [(dt.datetime(2026, 9, 1, 15), 0.5), (d.replace(hour=6), -3.0), (d.replace(hour=18), 1.0)]   # +50 · −300 · +100
    out = _cal_fixed(trades, [(d.replace(hour=12), "paper_balance", "2900", "9900")], realized_equity=2950 - 300 + 7000 + 100)
    start = 2950.0
    exact = (start - 300) / start * (1 + 100 / (start - 300 + 7000)) - 1
    assert out["days"]["2026-09-03"]["day_return_pct"] == round(exact * 100, 2)          # −9.13 %, not the diluted Dietz −3 %
