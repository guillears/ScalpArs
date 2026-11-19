"""🧹 Nov-19 decisions-export cleanup (operator review of a 75k-row 2-day paper export):
① the momentum ladder's BLOCK / FAILS / ADMIT journal lines are muted while BOTH momentum entry switches are off (counters untouched,
  every other path still journaled) · ② the dead FRENZY_WILLY hold's wh_* columns ride the export only when a WILLY_HOLD row exists ·
③ FRENZY followed-list transitions (FRENZY_LIST: FRENZY_FOLLOW / FRENZY_UNFOLLOW) — on transitions only, in export_rows ·
⑤ EXCHANGE_DAY on a paper day = PAPER (not CHECK); LIVE days unchanged."""
import asyncio
import contextvars
import json
import os
from datetime import datetime

import pandas as pd
import pytest

from services import decision_journal as dj
from services import trading_engine as te
from services import track_record as TR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── ① momentum journal mute ──────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def journal(monkeypatch):
    got = []
    monkeypatch.setattr(te._djournal, "note", lambda ev, **kw: got.append((ev, kw)))
    th = te.config.trading_config.thresholds

    def switches(long_on, short_on):
        monkeypatch.setattr(th, "momentum_long_entries_enabled", long_on, raising=False)
        monkeypatch.setattr(th, "momentum_short_entries_enabled", short_on, raising=False)
    return got, switches


def _engine():
    e = te.TradingEngine.__new__(te.TradingEngine)     # no __init__: only what _record_filter_block touches
    e._filter_block_counts = {}
    e._last_pair_block_reason = {}
    return e


def _in_ctx(fn):
    return contextvars.copy_context().run(fn)            # task-local scope never leaks between cases


def _block(e, scope, momentum_ctx=True):
    def go():
        e._journal_mom_scope = scope
        e._open_ctx_momentum = momentum_ctx
        e._journal_pair = "QNTUSDT"
        e._record_filter_block("PAIR_RSI_RANGE", "LONG")
    _in_ctx(go)


def test_block_muted_on_momentum_path_when_both_switches_off(journal):
    got, switches = journal
    switches(False, False)
    e = _engine()
    _block(e, scope=True)
    assert got == []                                                     # no journal line …
    assert sum(e._filter_block_counts.values()) == 1                     # … the dashboard counter still counts
    assert e._journal_suppressed == {"BLOCK": 1}


@pytest.mark.parametrize("long_on,short_on", [(True, False), (False, True), (True, True)])
def test_block_written_when_either_switch_on(journal, long_on, short_on):
    got, switches = journal
    switches(long_on, short_on)
    e = _engine()
    _block(e, scope=True)
    assert [ev for ev, _ in got] == ["BLOCK"]


def test_non_momentum_paths_never_muted(journal):
    got, switches = journal
    switches(False, False)
    e = _engine()
    _block(e, scope=False)                                   # FRENZY task / sleeve path: scope off
    _block(e, scope=True, momentum_ctx=False)                # open_position working for another sleeve inside the scan
    assert [ev for ev, _ in got] == ["BLOCK", "BLOCK"]
    assert not e.__dict__.get("_journal_suppressed")


def test_sleeve_hook_inside_the_scan_runs_outside_the_scope(journal):
    got, switches = journal
    switches(False, False)
    e = _engine()

    async def hook():                                        # e.g. a flip refused on its own gate, called from the momentum loop
        e._record_filter_block("FLIP_RSI_MIN", "SHORT")
        return "done"

    async def scan():
        e._journal_mom_scope = True
        e._open_ctx_momentum = True
        r = await e._outside_momentum_journal(hook())
        assert e._journal_mom_scope is True                  # restored for the rest of the momentum chain
        e._record_filter_block("BTC_SLOPE_GATE", "LONG")     # momentum chain → muted
        return r
    assert _in_ctx(lambda: asyncio.run(scan())) == "done"
    assert [kw["gate"] for _, kw in got] == ["FLIP_RSI_MIN"]


def test_fails_muted_only_for_the_momentum_source(journal):
    got, switches = journal
    switches(False, False)
    e = _engine()
    e._journal_fails(["EMA_STACK"], "LONG", "QNTUSDT", "MOMENTUM")
    e._journal_fails(["FLIP_RSI_MIN"], "SHORT", "QNTUSDT", "FLIP:FAN_RATIO_GATE")
    assert [kw["src"] for _, kw in got] == ["FLIP:FAN_RATIO_GATE"]
    switches(True, False)
    e._journal_fails(["RSI_MAX"], "LONG", "QNTUSDT", "MOMENTUM")
    assert [kw["src"] for _, kw in got] == ["FLIP:FAN_RATIO_GATE", "MOMENTUM"]


def test_scan_wiring_pins():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = src.index("    async def scan_and_trade("); j = src.index("    async def update_pair_data(", i)
    scan = src[i:j]
    assert scan.count("self._journal_momentum_muted('ADMIT', scoped=False)") == 3          # every ladder ADMIT site
    assert scan.count("_djournal.note('ADMIT'") == 3
    for hook in ("self._maybe_open_flip(", "self._bearrun_eligibility(", "self._maybe_open_bounce_long(", "self._maybe_open_bull_long("):
        n = scan.count(f"await self.{hook[5:]}") + scan.count(f"= await self.{hook[5:]}")
        assert n == 0, f"{hook} called inside the momentum scope without _outside_momentum_journal"
    assert scan.count("_outside_momentum_journal(self._maybe_open_flip(") == 2
    assert "self._journal_mom_scope = False   # 🔇 (Nov-19) the momentum loop is over" in scan
    assert "self._journal_mom_scope = False   # 🔇 (Nov-19) the spike chase / fade / bounce branch" in scan
    assert "_djournal.note('SCAN'" in scan                                                   # the heartbeat is kept


# ── ② wh_* columns only with a WILLY_HOLD row ───────────────────────────────────────────────────────────────────────────────────────
def test_wh_cols_only_when_a_willy_hold_row_exists():
    wh = ["wh_signal_at", "wh_reason"]
    assert dj.cols_if_event([{"e": "OPEN"}, {"e": "BOOK"}], "WILLY_HOLD", wh) == []
    assert dj.cols_if_event([{"e": "OPEN"}, {"e": "WILLY_HOLD"}], "WILLY_HOLD", wh) == wh
    assert dj.cols_if_event([], "WILLY_HOLD", wh) == [] and dj.cols_if_event(None, "WILLY_HOLD", wh) == []
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '_wh_cols = _dj.cols_if_event(rows, "WILLY_HOLD", _wh_cols)' in src
    assert "list(_dj.FRENZY_LIST_COLS)" in src


def test_scout_willy_hold_tolerates_a_file_without_wh_cols(tmp_path, monkeypatch):
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_willy_hold as S
    f = tmp_path / "scalpars_decisions_paper_x.csv"
    pd.DataFrame([dict(t="2026-11-19T00:00:00", e="OPEN", pair="AUSDT")]).to_csv(f, index=False)
    monkeypatch.setattr(S._EG, "glob_exports", lambda pat: [str(f)])
    assert S.export_rows().empty


# ── ③ FRENZY followed list: transitions only ────────────────────────────────────────────────────────────────────────────────────────
def _t(pair, chg, rng=0.0, vol=50e6):
    return {"pair": pair, "change_24h": chg, "range_24h": rng, "volume_24h": vol}


def test_frenzy_list_transitions_and_reasons():
    by = {p["pair"]: p for p in [_t("AUSDT", 20), _t("BUSDT", 18), _t("CUSDT", 3), _t("DUSDT", 30, vol=1e6), _t("EUSDT", 16)]}
    ev = te.frenzy_list_events(None, ["AUSDT", "BUSDT"], ["AUSDT", "BUSDT"], {}, by, set(), 15.0, 20e6, first=True)
    assert [(x["pair"], x["gate"]) for x in ev] == [("AUSDT", "FRENZY_FOLLOW"), ("BUSDT", "FRENZY_FOLLOW")]
    assert ev[0]["reason"].startswith("process start") and ev[0]["fl_rank"] == 1 and ev[0]["fl_chg24_pct"] == 20
    prev = {"AUSDT": "shortlist", "BUSDT": "shortlist"}
    assert te.frenzy_list_events(prev, ["AUSDT", "BUSDT"], ["AUSDT", "BUSDT"], {}, by, set(), 15.0, 20e6) == []   # no change → no row
    prev = {"AUSDT": "shortlist", "BUSDT": "shortlist", "CUSDT": "flagged", "DUSDT": "shortlist", "EUSDT": "shortlist",
            "ZUSDT": "shortlist", "XUSDT": "shortlist"}
    ev = te.frenzy_list_events(prev, ["AUSDT", "FUSDT"], ["AUSDT"], {"FUSDT": {}}, dict(by, XUSDT=_t("XUSDT", 25)), {"XUSDT"}, 15.0, 20e6)
    got = {x["pair"]: (x["gate"], x["reason"]) for x in ev}
    assert got["FUSDT"] == ("FRENZY_FOLLOW", "flagged episode")
    assert got["BUSDT"][0] == "FRENZY_UNFOLLOW" and got["BUSDT"][1].startswith("cap displaced")
    assert got["CUSDT"][1] == "episode ended · 24 h move below the shortlist threshold"
    assert got["DUSDT"][1] == "24 h volume below the floor"
    assert got["EUSDT"][1].startswith("cap displaced")
    assert got["ZUSDT"][1].startswith("no 24 h ticker")
    assert got["XUSDT"][1] == "blacklisted"
    assert "AUSDT" not in got


def test_frenzy_list_rides_export_rows(tmp_path):
    d = str(tmp_path)
    recs = [{"t": "2026-11-19T10:00:01.000", "e": "FRENZY_LIST", "pair": "AUSDT", "gate": "FRENZY_FOLLOW", "reason": "shortlist",
             "fl_chg24_pct": 20.5, "fl_range24_pct": 31.0, "fl_vol24_usd": 4.5e7, "fl_rank": 3},
            {"t": "2026-11-19T12:00:01.000", "e": "FRENZY_LIST", "pair": "AUSDT", "gate": "FRENZY_UNFOLLOW", "reason": "cap displaced"}]
    with open(os.path.join(d, "decisions-2026-11-19.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(r) for r in recs) + "\n")
    rows = dj.export_rows(days=1, now=datetime(2026, 11, 19, 13, 0), directory=d)
    fl = [r for r in rows if r["e"] == "FRENZY_LIST"]
    assert [(r["gate"], r["reason"]) for r in fl] == [("FRENZY_FOLLOW", "shortlist"), ("FRENZY_UNFOLLOW", "cap displaced")]
    assert fl[0]["fl_rank"] == 3 and fl[0]["fl_vol24_usd"] == 4.5e7 and fl[0]["t"].startswith("2026-11-19T10:00:01")
    assert all(set(r) <= set(dj.EXPORT_COLS) | set(dj.FRENZY_LIST_COLS) for r in fl)


def test_frenzy_pass_journals_transitions():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = src.index("_fz_names = list(names)")
    blk = src[i:i + 1200]
    assert "frenzy_list_events(_fl_prev or {}, _fz_names" in blk and "_djournal.note('FRENZY_LIST', **_ev)" in blk
    assert "self._frenzy_follow_prev = frenzy_list_origins(_fz_names, _fl_sl, _frenzy_flags)" in blk and "follow=_follow" in blk


# ── ⑤ EXCHANGE_DAY: PAPER on a paper day ────────────────────────────────────────────────────────────────────────────────────────────
S = dict(realized=-12.5, commission=-1.25, funding=0.0, transfer=0.0, other=0.0, n=6, n_est=0)
BOT0 = dict(gross=0.0, fees=0.0, funding=0.0, closed=0, opened=0)
SNAP, PREV = dict(at_ms=2, wallet=986.25, margin_balance=990.0), dict(at_ms=1, wallet=1000.0)


def test_paper_day_label():
    v = TR.exchange_day_verdict(S, BOT0, SNAP, PREV, -13.75, paper=True)
    assert v["status"] == "PAPER"
    assert v["note"] == "paper: live account not traded by the bot — Binance realized -12.50, fees 1.25"


def test_live_day_check_unchanged():
    v = TR.exchange_day_verdict(S, BOT0, SNAP, PREV, -13.75)
    assert v["status"] == "CHECK" and "realized P&L Binance -12.50 vs bot +0.00" in v["note"]


def test_paper_keeps_check_for_real_findings():
    assert TR.exchange_day_verdict(S, BOT0, SNAP, PREV, -5.0, paper=True)["status"] == "CHECK"                  # wallet gap
    assert TR.exchange_day_verdict(S, dict(BOT0, closed=1, gross=3.0), SNAP, PREV, -13.75, paper=True)["status"] == "CHECK"   # live row
    assert TR.exchange_day_verdict(dict(S, unknown=["XYZ"]), BOT0, SNAP, PREV, -13.75, paper=True)["status"] == "CHECK"
    z = dict(S, realized=0.0, commission=0.0, n=0)
    assert TR.exchange_day_verdict(z, BOT0, SNAP, dict(PREV, wallet=986.25), 0.0, paper=True)["status"] == "NO_ACTIVITY"


def test_paper_day_rule():
    assert TR.paper_day(dict(BOT0, paper=4)) is True                        # the DB says the bot ran paper that day
    assert TR.paper_day(BOT0) is False                                      # no rows at all → never PAPER (live-era day, rows gone)
    assert TR.paper_day(dict(BOT0, opened=1, paper=4)) is False             # any live row → live reconciliation


def test_recorder_paper_label_needs_db_evidence(monkeypatch):
    """review (Nov-19): the engine's CURRENT mode never labels a past day. A live-era day with Binance activity and no bot rows (the paper
    DB holds none) stays CHECK in the backfill AND after the recheck — even with the engine in paper; a day with paper rows → PAPER."""
    import datetime as dt
    from types import SimpleNamespace as NS
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from database import Base
    import models
    monkeypatch.setattr(te.trading_engine, "is_paper_mode", True)          # engine in PAPER now
    monkeypatch.setattr(TR, "_cfg", lambda: NS(backup_enabled=False, backup_s3_bucket="", exchange_record_enabled=True))
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    SL = async_sessionmaker(eng, expire_on_commit=False)
    d14, d15 = TR.day_bounds("2026-11-14")[0], TR.day_bounds("2026-11-15")[0]
    inc = lambda t, typ, x: {"time": t + 3_600_000, "incomeType": typ, "income": str(x), "asset": "USDT", "tranId": f"{typ}{t}", "symbol": "AUSDT"}
    rows = [inc(d14, "REALIZED_PNL", -40), inc(d14, "COMMISSION", -3), inc(d15, "REALIZED_PNL", 25), inc(d15, "COMMISSION", -2)]

    class App:
        async def _call(self, w, fn, *a):
            return await fn(*a)

        async def _income(self, lo, hi):
            return [r for r in rows if lo <= r["time"] <= hi]

        async def prepare_rates(self, x):
            pass

        def rate(self, a, t):
            return None

    async def commit(db):
        await db.commit()
    async def bal():   # never the real exchange
        return dict(ok=True, usdt_total=1000.0, usdt_margin_balance=1000.0, usdt_unrealized=0.0, bnb_total=0.0)
    now = dt.datetime(2026, 11, 16, 1, 0)
    rec = TR.TrackRecorder(session_factory=SL, commit=commit, now=lambda: now, app_recorder=App(), keys_configured=lambda: True,
                           balance_fn=bal)

    async def go():
        async with eng.begin() as c:
            await c.run_sync(Base.metadata.create_all)
        async with SL() as db:   # 11-15: the bot traded PAPER (no live row) · 11-14: no bot row at all (live-era, rows gone)
            db.add(models.Order(pair="P", direction="LONG", status="CLOSED", is_paper=True, entry_strategy="MOMENTUM", confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0, pnl=3.0,
                                opened_at=dt.datetime(2026, 11, 15, 2), closed_at=dt.datetime(2026, 11, 15, 3)))
            await db.commit()
        await rec._exchange_if_due(now)                                         # 7-day backfill (income only, no snapshot pair)
        async with SL() as db:
            first = {x.day: x.status for x in (await db.execute(select(models.ExchangeDay))).scalars().all()}
        await rec._recheck_recent(now)
        async with SL() as db:
            after = {x.day: (x.status, x.note) for x in (await db.execute(select(models.ExchangeDay))).scalars().all()}
        await eng.dispose()
        return first, after
    first, after = asyncio.run(go())
    assert first["2026-11-14"] == "CHECK" and after["2026-11-14"][0] == "CHECK"
    assert first["2026-11-15"] == "PAPER" and after["2026-11-15"][0] == "PAPER"
    assert after["2026-11-15"][1].startswith("paper: live account not traded by the bot")


def test_momentum_mute_off_while_bearrun_is_on(journal, monkeypatch):
    got, switches = journal
    switches(False, False)
    monkeypatch.setattr(te.config.trading_config.thresholds, "bearrun_sleeve_enabled", True, raising=False)
    e = _engine()
    _block(e, scope=True)
    assert [ev for ev, _ in got] == ["BLOCK"]                               # Bear-Run shorts come from the ladder → keep journaling
    monkeypatch.setattr(te.config.trading_config.thresholds, "bearrun_sleeve_enabled", False, raising=False)
    _block(e, scope=True)
    assert len(got) == 1


def test_frenzy_list_seed_nonascii_and_read_limit():
    by = {p["pair"]: p for p in [_t("AUSDT", 20), _t("SEEDUSDT", 2), _t("龙虾USDT", 40), _t("FLGUSDT", 3)]}
    sl = ["AUSDT"]
    names1 = ["AUSDT", "SEEDUSDT", "龙虾USDT", "FLGUSDT"]
    o = te.frenzy_list_origins(names1, sl, {"FLGUSDT": {}})
    assert o == {"AUSDT": "shortlist", "SEEDUSDT": "seed", "龙虾USDT": "seed", "FLGUSDT": "flagged"}
    ev = te.frenzy_list_events(None, names1, sl, {"FLGUSDT": {}}, by, set(), 15.0, 20e6, first=True, follow=["SEEDUSDT", "FLGUSDT"])
    assert {x["pair"]: x["reason"] for x in ev}["SEEDUSDT"] == "process start: restart seed (open episode / recent trade)"
    ev = te.frenzy_list_events(o, ["AUSDT"], sl, {}, by, set(), 15.0, 20e6, follow=["FLGUSDT"])   # pass 2
    got = {x["pair"]: x["reason"] for x in ev}
    assert got["SEEDUSDT"] == "restart seed (one pass)"
    assert got["龙虾USDT"] == "non-ASCII symbol (never shortlisted)"
    assert got["FLGUSDT"] == "read limit (followed coins beyond the read cap)"
