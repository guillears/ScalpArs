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
