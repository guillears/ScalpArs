"""🔭 Sep-30 — "Download Decisions CSV": the decision journal as rows for the opportunity scout. BLOCK lines aggregate per 5 min ×
pair × direction × gate (n, no_room); OPEN / ADMIT / EXPIRED pass through; SCAN and bad lines are dropped; only the last `days`."""
import json
import os
from datetime import datetime

from services import decision_journal as dj


def _write(d, day, recs):
    with open(os.path.join(d, f"decisions-{day}.jsonl"), "w") as f:
        for r in recs:
            f.write((json.dumps(r) if isinstance(r, dict) else r) + "\n")


def test_blocks_aggregate_and_events_pass_through(tmp_path):
    d = str(tmp_path)
    _write(d, "2026-09-30", [
        {"t": "2026-09-30T14:01:10.000", "e": "BLOCK", "gate": "EMA_STACK", "dir": "LONG", "pair": "QNTUSDT", "room": True},
        {"t": "2026-09-30T14:03:59.000", "e": "BLOCK", "gate": "EMA_STACK", "dir": "LONG", "pair": "QNTUSDT", "room": False},
        {"t": "2026-09-30T14:06:00.000", "e": "BLOCK", "gate": "EMA_STACK", "dir": "LONG", "pair": "QNTUSDT", "room": True},
        {"t": "2026-09-30T14:02:00.000", "e": "BLOCK", "gate": "RSI_MAX", "dir": "SHORT", "pair": "ENAUSDT"},
        {"t": "2026-09-30T14:02:30.000", "e": "OPEN", "pair": "ENAUSDT", "dir": "SHORT", "strategy": "SURGE_SHORT", "price": 0.26},
        {"t": "2026-09-30T14:02:31.000", "e": "SCAN", "btc_rsi": 50},
        {"t": "2026-09-30T14:04:00.000", "e": "BLOCK", "gate": "SURGE_WINDOW_MISSED", "dir": "LONG"},   # no pair → MARKET
        {"t": "2026-09-30T14:04:30.000", "e": "SCAN"},
        {"t": "bad-time", "e": "BLOCK", "gate": "Y", "dir": "LONG", "pair": "A"},                  # malformed t: skipped alone
        "not json",
        {"t": "2026-09-30T14:07:00.000", "e": "OPEN", "pair": "AFTERBAD", "dir": "LONG", "strategy": "MOMENTUM"},
    ])
    _write(d, "2026-09-20", [{"t": "2026-09-20T01:00:00.000", "e": "OPEN", "pair": "OLDUSDT", "dir": "LONG"}])   # outside days
    rows = dj.export_rows(days=3, now=datetime(2026, 9, 30, 15, 0), directory=d)
    blocks = [r for r in rows if r["e"] == "BLOCK"]
    q = sorted((r["t"], r["n"], r["no_room"]) for r in blocks if r["pair"] == "QNTUSDT")
    assert q == [("2026-09-30T14:00:00", 2, 1), ("2026-09-30T14:05:00", 1, 0)]
    assert any(r["pair"] == "ENAUSDT" and r["gate"] == "RSI_MAX" and r["n"] == 1 for r in blocks)
    opens = [r for r in rows if r["e"] == "OPEN"]
    assert [o["pair"] for o in opens] == ["ENAUSDT", "AFTERBAD"]              # a bad line never skips the rest of the file
    assert any(r["pair"] == "MARKET" and r["gate"] == "SURGE_WINDOW_MISSED" for r in blocks)
    scans = [r for r in rows if r["e"] == "SCAN"]
    assert len(scans) == 1 and scans[0]["n"] == 2 and scans[0]["t"] == "2026-09-30T14:00:00"   # both scans share the 14:00 bucket
    assert not any(r.get("pair") == "OLDUSDT" for r in rows)
    assert set(dj.EXPORT_COLS) >= set(rows[0].keys())


def test_missing_directory_never_raises(tmp_path):
    assert dj.export_rows(days=3, directory=str(tmp_path / "nope")) == []


def test_fails_full_gate_sets_aggregate(tmp_path):
    """Sep-30: FAILS = the FULL gate set one candidate failed (overlap-aware attribution) — aggregated per 5 min × pair × dir ×
    set × source, never merged into BLOCK rows."""
    d = str(tmp_path)
    f = lambda t, gates, src="MOMENTUM", pair="QNTUSDT": {"t": t, "e": "FAILS", "pair": pair, "dir": "LONG", "gates": gates, "src": src, "n_gates": gates.count("+") + 1}
    _write(d, "2026-09-30", [
        f("2026-09-30T14:01:00.000", "EMA_STACK+RSI_MAX"),
        f("2026-09-30T14:02:00.000", "EMA_STACK+RSI_MAX"),
        f("2026-09-30T14:03:00.000", "EMA_STACK"),
        f("2026-09-30T14:03:30.000", "MACRO:BTC_ADX_GATE_LOW+EMA_STACK"),
        f("2026-09-30T14:04:00.000", "FLIP_RSI_MIN", src="FLIP:PAIR_RSI_OB"),
        {"t": "2026-09-30T14:04:10.000", "e": "BLOCK", "gate": "EMA_STACK", "dir": "LONG", "pair": "QNTUSDT"},
    ])
    rows = dj.export_rows(days=1, now=datetime(2026, 9, 30, 15, 0), directory=d)
    fr = sorted((r["gate"], r["n"], r["src"]) for r in rows if r["e"] == "FAILS")
    assert fr == [("EMA_STACK", 1, "MOMENTUM"), ("EMA_STACK+RSI_MAX", 2, "MOMENTUM"), ("FLIP_RSI_MIN", 1, "FLIP:PAIR_RSI_OB"),
                  ("MACRO:BTC_ADX_GATE_LOW+EMA_STACK", 1, "MOMENTUM")]
    assert [r["n"] for r in rows if r["e"] == "BLOCK"] == [1]
    assert all(set(r) <= set(dj.EXPORT_COLS) for r in rows)


def test_engine_journal_fails_dedupes_and_never_raises(monkeypatch):
    """One FAILS line per identical (src, dir, pair, set) per 5 min; macro gate prepended; bad input never raises."""
    import types
    from services import trading_engine as te
    got = []
    monkeypatch.setattr(te._djournal, "note", lambda ev, **kw: got.append((ev, kw)))
    self = types.SimpleNamespace()
    jf = te.TradingEngine._journal_fails
    jf(self, ["EMA_STACK", "RSI_MAX"], "LONG", "QNTUSDT", "MOMENTUM")
    jf(self, ["EMA_STACK", "RSI_MAX"], "LONG", "QNTUSDT", "MOMENTUM")          # same bucket → skipped
    jf(self, ["EMA_STACK"], "LONG", "QNTUSDT", "MOMENTUM", macro="BTC_ADX_GATE_LOW")
    jf(self, [], "LONG", "QNTUSDT", "MOMENTUM")                               # nothing failed → nothing written
    jf(self, object(), "LONG", "X", "MOMENTUM")                               # garbage → swallowed
    assert [k["gates"] for _, k in got] == ["EMA_STACK+RSI_MAX", "MACRO:BTC_ADX_GATE_LOW+EMA_STACK"]
    assert got[1][1]["n_gates"] == 2


import pytest  # noqa: E402


@pytest.mark.parametrize("event", dj.IMMEDIATE)
def test_fills_are_written_at_once_and_buffer_flushes(tmp_path, monkeypatch, event):
    """Sep-30: 4 SURGE fills vanished because a deploy restarted the bot before the next scan flushed the buffer. OPEN / EXPIRED
    are written immediately (forcing earlier lines out, in order); other lines wait for flush() (per scan, shutdown, atexit)."""
    monkeypatch.setattr(dj, "_dir", lambda: str(tmp_path))
    monkeypatch.setattr(dj, "_enabled", lambda: True)
    monkeypatch.setattr(dj, "_state", dict(dj._state))
    monkeypatch.setattr(dj, "_buffer", [])
    files = lambda: [p for p in os.listdir(tmp_path) if p.startswith("decisions-")]
    dj.note("BLOCK", gate="X", dir="LONG", pair="AUSDT")
    dj.note("ADMIT", gate="CROSS_OB_OPEN", dir="LONG", pair="AUSDT")
    assert files() == []                                            # refusals and waivers wait for the scan flush
    dj.note(event, pair="MOVRUSDT", dir="SHORT", strategy="SURGE_SHORT")
    lines = open(os.path.join(tmp_path, files()[0])).read().splitlines()
    assert [json.loads(x)["e"] for x in lines] == ["BLOCK", "ADMIT", event]   # the fill forced the whole buffer out, in order
    dj.note("BLOCK", gate="Y", dir="LONG", pair="BUSDT")
    dj.flush()                                                      # what shutdown / atexit call
    assert len(open(os.path.join(tmp_path, files()[0])).read().splitlines()) == 4 and dj._buffer == []


def test_shutdown_paths_flush_the_journal():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert "atexit.register(flush)" in open(dj.__file__).read()
    main_src = open(os.path.join(root, "main.py")).read()
    assert "await stop_background_tasks()\n    finally:" in main_src and "_djs.flush()" in main_src


def test_export_default_covers_the_whole_batch(monkeypatch):
    """🗂 DECISION_LOG 392: with no `days` the export reads the journal from the UTC day of this mode's first order to today (≤ 14) —
    a 5-day batch no longer loses its first two days of OPEN / BLOCK rows to the old fixed 3; an explicit `days` still wins."""
    import asyncio
    import datetime as dt
    import os
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main
    import models
    from services import decision_journal as dj
    seen = []
    monkeypatch.setattr(dj, "export_rows", lambda days, *a, **k: (seen.append(days), [])[1])

    async def _noinit(db):
        return None
    monkeypatch.setattr(main.trading_engine, "initialize", _noinit)

    async def run(first_days_ago, days):
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        async with S() as db:
            if first_days_ago is not None:
                t = dt.datetime.utcnow() - dt.timedelta(days=first_days_ago)
                db.add(models.Order(pair="AUSDT", direction="LONG", status="CLOSED", is_paper=main.trading_engine.is_paper_mode,
                                    confidence="STRONG_BUY", entry_price=1.0, quantity=1.0, investment=1.0, leverage=1, notional_value=1.0,
                                    opened_at=t, closed_at=t))
                await db.commit()
            await main.export_decisions_csv(days=days, db=db)
        await eng.dispose()
    _was = main.trading_engine.is_paper_mode
    try:
        main.trading_engine.is_paper_mode = True
        asyncio.run(run(4, None))
        asyncio.run(run(40, None))
        asyncio.run(run(None, None))
        asyncio.run(run(4, 2))
        main.trading_engine.is_paper_mode = False
        asyncio.run(run(4, None))                       # live has no batch → 3
    finally:
        main.trading_engine.is_paper_mode = _was
    assert seen == [5, dj.EXPORT_MAX_DAYS, 3, 2, 3] and dj.EXPORT_MAX_DAYS == 7
    ui = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()
    assert ui.count("/api/decisions/export.csv?t=") == 1                       # the button: the batch
    assert "async function _downloadDecisionsCsv(timeoutMs = 60e3, days = 3)" in ui   # auto + pre-reset: 3 days


def test_bearrun_refusals_ride_the_export_and_spike_blocks_name_their_pair(tmp_path):
    """📓 DECISION_LOG 394: a Bear-Run refusal (its own BEARRUN event — never a second BLOCK) is exported one per line with its rule and
    values; the spike scanner names the pair on its journal lines (its BLOCK rows were MARKET) and clears it after the cycle."""
    import os
    d = str(tmp_path)
    lines = [{"t": "2026-10-27T12:00:01.000", "e": "BEARRUN", "pair": "XUSDT", "dir": "SHORT", "gate": "BEARRUN_REFUSED_OFF24LO",
              "reason": "would bypass BTC_SLOPE_GATE; off24lo 1.2"},
             {"t": "2026-10-27T12:02:01.000", "e": "BEARRUN", "pair": "XUSDT", "dir": "SHORT", "gate": "BEARRUN_REFUSED_OFF24LO",
              "reason": "would bypass BTC_SLOPE_GATE; off24lo 1.3"},          # same 5-min bucket → one row, n 2, the first values
             {"t": "2026-10-27T12:00:01.000", "e": "BLOCK", "pair": "XUSDT", "dir": "SHORT", "gate": "BTC_SLOPE_GATE"}]
    with open(os.path.join(d, "decisions-2026-10-27.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = dj.export_rows(days=1, now=datetime(2026, 10, 27, 15, 0), directory=d)
    br = [r for r in rows if r["e"] == "BEARRUN"]
    assert len(br) == 1 and br[0]["gate"] == "BEARRUN_REFUSED_OFF24LO" and br[0]["n"] == 2 and br[0]["reason"].endswith("off24lo 1.2")
    assert sum(r.get("n", 0) for r in rows if r["e"] == "BLOCK") == 1                 # the refusal is not a second BLOCK
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services", "trading_engine.py"), encoding="utf-8").read()
    i = src.index("    async def _spike_scanner_cycle("); j = src.index("    async def open_position(", i)
    assert "self._journal_pair = p.get('pair'); self._journal_ctx = None" in src[i:j]
    assert src.count("self._journal_pair = None; self._journal_ctx = None   # 📓 (DL 394)") == 1
    assert "_djournal.note('BEARRUN', pair=pair, dir='SHORT', gate=f\"BEARRUN_REFUSED_{self._bear_refuse}\"" in src
