"""🧹 DECISION_LOG 346: the Performance filter row — "Filter by Momentum cell" lists only momentum / flip sub-groups (sleeve-identity
tags hidden, decided by entry_strategy), the Sleeve select marks sleeves switched off in the live config, and a second All · Bot · Manual
switch shares one state with the calendar's (DL 323)."""
import asyncio
import datetime as dt
import json
import os
import re
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _html():
    return open(os.path.join(ROOT, "templates", "index.html")).read()


def _order(models, i, strat, cell, **kw):
    t0 = dt.datetime(2026, 10, 18, 10, 0, 0) + dt.timedelta(hours=i)
    base = dict(pair=f"P{i}USDT", direction="LONG", status="CLOSED", entry_price=100.0, exit_price=100.5, investment=500.0, leverage=20.0,
                notional_value=10000.0, quantity=100.0, confidence="STRONG_BUY", entry_strategy=strat, is_paper=True, pnl=50.0,
                pnl_percentage=0.5, peak_pnl=0.6, trough_pnl=-0.1, entry_fee=4.5, total_fee=9.0, opened_at=t0,
                closed_at=t0 + dt.timedelta(minutes=17), close_reason="TAKE_PROFIT", entry_macro_trend="BULLISH",
                cell_multiplier=1.0, cell_lev_multiplier=1.0, cell_multiplier_source=cell)
    base.update(kw)
    return models.Order(**base)


ROWS = [  # (entry_strategy, cell_multiplier_source)
    ("MOMENTUM", "NONEXP_CALM3D"), ("MOMENTUM", "UNMATCHED"), (None, "C1_SHORT"), ("FLIP:FAN_RATIO_GATE", "FLIPCELL"),
    ("FRENZY_WILLY", "FRENZY_WILLY"), ("FRENZY_LITE", "FRENZY_LITE"), ("FRENZY_LITE_RETRY", "FRENZY_LITE_RETRY"),
    ("FRENZY_WILLY_HT", "FRENZY_WILLY_HT"), ("SPIKE_FADE", "SPIKE_FADE"), ("SURGE_LONG", "SURGE_LONG"), ("MANUAL", "MANUALTAG"),
    ("MOMENTUM", "SPIKE_CHASE_PROBE"), ("MOMENTUM", "FGP_X"), ("MOMENTUM", None), ("SURGE_SHORT", "UNMATCHED"),
]


def _perf(rows, **kw):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        S = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        main.trading_engine.paper_balance = 10_000.0
        async with S() as db:
            for i, (es, cell) in enumerate(rows):
                db.add(_order(models, i, es, cell))
            await db.commit()
            out = json.loads(json.dumps(await main._compute_performance(db, **kw), default=str))
        await eng.dispose()
        return out
    return asyncio.run(run())


def test_cell_source_options_sqlite():
    out = _perf(ROWS)
    assert out["cell_source_options"] == ["C1_SHORT", "FLIPCELL", "NONEXP_CALM3D", "UNMATCHED"]
    # independent of the active filters (incl. the origin view)
    assert _perf(ROWS, origin="MANUAL")["cell_source_options"] == out["cell_source_options"]
    assert _perf(ROWS, flip_source="FRENZY_WILLY")["cell_source_options"] == out["cell_source_options"]


def test_cell_filter_slicing_unchanged():
    out = _perf(ROWS, cell_source="FRENZY_WILLY")   # a hidden sleeve tag still slices server-side (a stale selection keeps working)
    assert out["total_trades"] == 1
    assert _perf(ROWS, cell_source="UNMATCHED")["total_trades"] == 2


def test_cell_source_options_pure():
    import main
    O = lambda es, s: SimpleNamespace(entry_strategy=es, cell_multiplier_source=s)
    rows = [O("MOMENTUM", "W1"), O("", "W2"), O("flip:FAN_RATIO_GATE×2", "W3"), O("FRENZY_NEWTHING", "FRENZY_NEWTHING"),
            O("BULL_LONG", "BL"), O("MOMENTUM", "X_PROBE"), O("MOMENTUM", ""),
            O("BULLRUN_LONG", "BULLRUN_CELL"), O("BEARRUN_SHORT", "BEARRUN_CELL")]
    # Bull-Run / Bear-Run have no Sleeve-select option → their tags stay the only UI way to slice them
    assert main._cell_source_options(rows) == ["BEARRUN_CELL", "BULLRUN_CELL", "W1", "W2", "W3"]


def test_regime_sleeve_values_match_engine_stamps():
    src = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert '"BEARRUN_SHORT" if bearrun_short' in src and '"BULLRUN_LONG" if bullrun_long' in src


def test_sleeve_enabled_map():
    import main
    th = SimpleNamespace(flip_entry_enabled=True, spike_chase_probe_enabled=True, spike_chase_enabled=False, spike_fade_enabled=True,
                         spike_bounce_enabled=False, surge_long_enabled=True, surge_short_enabled=True, frenzy_long_enabled=True,
                         frenzy_wide_enabled=False, frenzy_lite_enabled=True, frenzy_lite_retry_enabled=True,
                         frenzy_willy_enabled=False, frenzy_willy_ht_enabled=True, flip_entry_sources="FAN_RATIO_GATE:1.0:0.5")
    assert main._sleeve_enabled(th, "FRENZY_WIDE") is False and main._sleeve_enabled(th, "FRENZY_LONG") is True
    assert main._sleeve_enabled(th, "SPIKE_CHASE") is False and main._sleeve_enabled(th, "SPIKE_FADE") is True
    assert main._sleeve_enabled(th, "FRENZY_WILLY_HT") is False          # HT needs WILLY on
    assert main._sleeve_enabled(th, "MOMENTUM") is None                  # unknown → active
    assert main._sleeve_enabled(SimpleNamespace(), "FRENZY_WIDE") is None  # switch missing from config → unknown
    assert main._sleeve_enabled(None, "FRENZY_WIDE") is None
    # FAN mirrors the engine's flip registry: listed in flip_entry_sources AND flip_entry_enabled
    assert main._sleeve_enabled(th, "FAN_RATIO_GATE") is True
    assert main._sleeve_enabled(SimpleNamespace(flip_entry_enabled=True, flip_entry_sources="PAIR_RSI_OB:1.0"), "FAN_RATIO_GATE") is False
    assert main._sleeve_enabled(SimpleNamespace(flip_entry_enabled=True, flip_entry_sources=""), "FAN_RATIO_GATE") is False
    assert main._sleeve_enabled(SimpleNamespace(flip_entry_enabled=False, flip_entry_sources="FAN_RATIO_GATE"), "FAN_RATIO_GATE") is False
    assert main._sleeve_enabled(SimpleNamespace(flip_entry_enabled=True), "FAN_RATIO_GATE") is None          # unreadable → active
    assert main._sleeve_enabled(SimpleNamespace(flip_entry_enabled=True, flip_entry_sources=None), "FAN_RATIO_GATE") is None
    rows = [SimpleNamespace(entry_strategy=e) for e in ("FLIP:FAN_RATIO_GATE×2", "FLIP:FAN_RATIO_GATE", "FAN_RATIO_GATE", "FRENZY_WIDE", "MOMENTUM")]
    st = main._sleeve_filter_status(th, rows)
    assert st["FAN_RATIO_GATE"] == {"enabled": True, "n": 3}          # same match as the server filter (prefix or bare)
    assert main._flip_source_match("flip:fan_ratio_gate×2", "FAN_RATIO_GATE") and not main._flip_source_match("FRENZY_LITE_RETRY", "FRENZY_LITE")
    assert st["FRENZY_WIDE"] == {"enabled": False, "n": 1} and st["SPIKE_BOUNCE"] == {"enabled": False, "n": 0}
    # every mapped value exists as an option of the Sleeve select, and every switch is a real config field
    html = _html()
    sel = re.search(r'<select id="flip-source-filter".*?</select>', html, re.S).group(0)
    import config
    for k, fields in main.SLEEVE_FILTER_SWITCHES.items():
        assert f'value="{k}"' in sel, k
        for f in fields:
            assert hasattr(config.trading_config.thresholds, f), f


def test_sleeve_status_in_payload():
    out = _perf(ROWS)
    st = out["sleeve_filter_status"]
    assert st["FRENZY_WILLY"]["n"] == 1 and st["SURGE_SHORT"]["n"] == 1 and st["FAN_RATIO_GATE"]["n"] == 1
    for k in ("FRENZY_WILLY", "SURGE_SHORT", "FAN_RATIO_GATE"):   # n == the trade count the server filter returns for that value
        assert _perf(ROWS, flip_source=k)["total_trades"] == st[k]["n"], k


def test_dropdown_helper_failure_never_blanks_performance(monkeypatch):
    import main
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(main, "_cell_source_options", boom)
    monkeypatch.setattr(main, "_sleeve_filter_status", boom)
    out = _perf(ROWS)
    assert out["cell_source_options"] == [] and out["sleeve_filter_status"] == {} and out["total_trades"] == len(ROWS)


def _node_sleeve_paint(status, selected):
    """Run _paintSleeveFilter from index.html against a tiny DOM stub in node."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        import pytest
        pytest.skip("node not installed")
    html = _html()
    fn = re.search(r"function _paintSleeveFilter\(status\) \{.*?\n        \}\n", html, re.S).group(0)
    js = """
const opts = %s.map(([v, t]) => ({ value: v, textContent: t, dataset: {}, disabled: false, title: '',
  _cls: new Set(), classList: { toggle(c, on) { on ? this._o._cls.add(c) : this._o._cls.delete(c); } } }));
opts.forEach(o => { o.classList._o = o; });
const sel = { value: %s, options: opts };
global.document = { getElementById: () => sel };
%s
_paintSleeveFilter(%s);
_paintSleeveFilter(%s);   // idempotent: never "(off) (off)"
console.log(JSON.stringify(opts.map(o => ({ v: o.value, t: o.textContent, d: o.disabled, off: o._cls.has('sleeve-off') }))));
""" % (json.dumps([["", "All Sleeves"], ["MOMENTUM", "Non-flip (momentum)"], ["FRENZY_WIDE", "🌐 FRENZY_WIDE"],
                   ["SPIKE_BOUNCE", "🏀 SPIKE_BOUNCE"], ["SPIKE_CHASE", "🚀 SPIKE_CHASE"], ["FRENZY_LONG", "🔥 FRENZY_LONG"]]),
       json.dumps(selected), fn, json.dumps(status), json.dumps(status))
    r = subprocess.run([node, "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return {o["v"]: o for o in json.loads(r.stdout)}


def test_sleeve_paint_rules():
    status = {"FRENZY_WIDE": {"enabled": False, "n": 3}, "SPIKE_BOUNCE": {"enabled": False, "n": 0},
              "SPIKE_CHASE": {"enabled": None, "n": 0}, "FRENZY_LONG": {"enabled": True, "n": 0}}
    o = _node_sleeve_paint(status, "")
    assert [k for k in o] == ["", "MOMENTUM", "FRENZY_WIDE", "SPIKE_BOUNCE", "SPIKE_CHASE", "FRENZY_LONG"]   # never reordered
    assert o["FRENZY_WIDE"] == {"v": "FRENZY_WIDE", "t": "🌐 FRENZY_WIDE (off)", "d": False, "off": True}   # off with trades: shaded, selectable
    assert o["SPIKE_BOUNCE"]["t"] == "🏀 SPIKE_BOUNCE (off)" and o["SPIKE_BOUNCE"]["d"] is True             # off, no trades: disabled
    assert o["SPIKE_CHASE"]["t"] == "🚀 SPIKE_CHASE" and not o["SPIKE_CHASE"]["d"]                          # unknown → active
    assert o["FRENZY_LONG"]["t"] == "🔥 FRENZY_LONG" and not o["FRENZY_LONG"]["d"]
    assert o["MOMENTUM"]["t"] == "Non-flip (momentum)" and not o["MOMENTUM"]["d"]                          # missing from status → active
    sel = _node_sleeve_paint(status, "SPIKE_BOUNCE")
    assert sel["SPIKE_BOUNCE"]["d"] is False and sel["SPIKE_BOUNCE"]["t"].endswith("(off)")                # the selected value is never disabled


def test_label_and_export_wording():
    html = _html()
    assert "Filter by Momentum cell:" in html and "Cell/Door" not in html
    assert html.count('id="cell-source-filter"') == 1 and html.count('id="cell-source-filter-cells"') == 1
    assert "`Momentum cell ${cellSourceFilter} only`" in html
    assert "_fbits.push(`momentum cell=${_cellSrc1}`)" in html and "_fbits.push(`momentum cell=${_cellSrcSplit}`)" in html
    assert "(not a momentum cell)" in html                                    # a vanished selection keeps its option
    # both exports print the active origin view (DL 323) — the ACTIVE FILTERS bit + the View line
    assert "if (_ov1 !== 'ALL') _fbits.push(`view=${ORIGIN_LABELS[_ov1]}`)" in html
    assert "if (_ov2 !== 'ALL') _fbits.push(`view=${ORIGIN_LABELS[_ov2]}`)" in html
    assert "lines.push(_originReportLine(_ov1, [perf]))" in html and "lines.push(_originReportLine(_ov2, [perfBull, perfBear]))" in html


def test_second_origin_switch_shares_state():
    html = _html()
    assert html.count('id="origin-switch"') == 1 and html.count('id="origin-switch-filters"') == 1
    grp = re.search(r'<div id="origin-switch-filters".*?</div>', html, re.S).group(0)
    for v in ("ALL", "BOT", "MANUAL"):
        assert f'data-origin="{v}" onclick="setOriginView(\'{v}\')"' in grp
    row = html[html.index('id="direction-filter-badge"'):html.index('id="copy-analytics-btn"')]
    assert 'id="origin-switch-filters"' in row and "Show:" in row           # after the Direction filter, before the buttons
    paint = re.search(r"function _paintOriginSwitch\(\) \{.*?\n        \}\n", html, re.S).group(0)
    assert "#origin-switch [data-origin], #origin-switch-filters [data-origin]" in paint and "aria-pressed" in paint
    setv = re.search(r"function setOriginView\(v\) \{.*?\n        \}\n", html, re.S).group(0)
    assert "_paintOriginSwitch();" in setv
    assert ".origin-seg button[aria-pressed=\"true\"]" in html


def test_empty_book_still_paints_the_dropdowns():
    """operator 10-22: after a reset (no closed orders) the empty-view early return must still carry the (off) map — it did not."""
    out = _perf([])
    assert out["total_trades"] == 0
    assert out["cell_source_options"] == []
    st = out["sleeve_filter_status"]
    assert st and all(v["n"] == 0 for v in st.values())
    assert set(st) >= {"FRENZY_WILLY", "SURGE_SHORT", "FAN_RATIO_GATE"}
    out2 = _perf(ROWS, flip_source="FRENZY_WILLY", direction="SHORT")   # a filtered-empty view: same keys, whole-book counts
    assert out2["total_trades"] == 0 and out2["sleeve_filter_status"]["FRENZY_WILLY"]["n"] == 1


def test_perf_exception_fallback_still_paints_off_marks(monkeypatch):
    """review: the all-zeros fallback (a perf exception) keeps the config-only (off) map instead of dropping it."""
    import main
    from unittest.mock import AsyncMock

    async def boom(*a, **k):
        raise RuntimeError("perf broke")
    monkeypatch.setattr(main, "_compute_performance", boom)
    monkeypatch.setattr(main.trading_engine, "initialize", AsyncMock(return_value=None))
    monkeypatch.setattr(main.trading_engine, "manual_block_state", AsyncMock(return_value=[]))
    out = asyncio.run(main.get_performance(db=None))
    assert out["_error"] == "perf broke" and out["cell_source_options"] == []
    st = out["sleeve_filter_status"]
    assert st and all(v["n"] == 0 for v in st.values()) and "FRENZY_WIDE" in st
