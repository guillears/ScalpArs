"""📈 DECISION_LOG 493: the Daily Compound Return chart — the calendar payload's new fields (per-sleeve day P&L, start-of-day base,
unrounded growth, batch start day, settings-change marks), the pure series math (every calendar day, 0 % on a no-close day,
today only once it has a close, batch + rolling-7 compound), the single sleeve rule, the UI block and BOTH text exports."""
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()


def _pure():
    a = HTML.index("const DCR_EARLY_DAYS")
    b = HTML.index("const _dcrPct")
    return HTML[a:b]


def _series(cal, sleeve, today):
    prog = _pure() + f"console.log(JSON.stringify({{s: dcrSeries({json.dumps(cal)}, {json.dumps(sleeve)}, {json.dumps(today)}), o: dcrSleeveOptions({json.dumps(cal)})}}));"
    return json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)


CAL = {
    "batch_start_day": "2026-10-23",
    "config_marks": [{"day": "2026-10-25", "fields": ["frenzy_willy_invest_mult"]}],
    "days": {
        "2026-10-24": {"pnl": 100.0, "trades": 2, "g": 1.02, "day_return_pct": 2.0, "base": 5000.0,
                       "sleeves": {"Mom-Long": 150.0, "Frenzy-Willy": -50.0}},
        "2026-10-26": {"pnl": -51.0, "trades": 1, "g": 0.99, "day_return_pct": -1.0, "base": 5100.0, "sleeves": {"Frenzy-Willy": -51.0}},
    },
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_every_calendar_day_counts_and_compounds():
    r = _series(CAL, "ALL", "2026-10-28")
    s = r["s"]
    assert [p["day"] for p in s] == ["2026-10-23", "2026-10-24", "2026-10-25", "2026-10-26", "2026-10-27"]   # today (28th) has no close → out
    assert [round(p["r"], 6) for p in s] == [0, 0.02, 0, -0.01, 0]
    assert s[-1]["batch"] == pytest.approx((1.02 * 0.99) ** (1 / 5) - 1)
    assert s[1]["batch"] == pytest.approx(1.02 ** 0.5 - 1)
    assert s[2]["marks"] == ["frenzy_willy_invest_mult"] and s[0]["marks"] is None
    assert s[-1]["roll7"] == pytest.approx(s[-1]["batch"])            # < 7 days: the window is the whole batch


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_rolling_window_is_the_last_seven_days_and_today_with_a_close_counts():
    days = {f"2026-11-{d:02d}": {"pnl": 1, "trades": 1, "g": 1.10 if d == 1 else 1.0, "base": 100} for d in range(1, 10)}
    s = _series({"batch_start_day": "2026-11-01", "days": days}, "ALL", "2026-11-09")["s"]
    assert len(s) == 9                                                 # today (9th) has a close → in
    assert s[6]["roll7"] == pytest.approx(1.10 ** (1 / 7) - 1)         # days 1–7 still hold the +10 %
    assert s[7]["roll7"] == pytest.approx(0)                           # days 2–8: the +10 % left the window
    assert s[-1]["batch"] == pytest.approx(1.10 ** (1 / 9) - 1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_sleeve_view_is_its_pnl_on_the_account_base():
    r = _series(CAL, "Frenzy-Willy", "2026-10-28")
    s = {p["day"]: p for p in r["s"]}
    assert s["2026-10-24"]["r"] == pytest.approx(-50 / 5000) and s["2026-10-26"]["r"] == pytest.approx(-51 / 5100)
    assert s["2026-10-25"]["r"] == 0
    assert r["o"] == ["Mom-Long", "Frenzy-Willy"]                     # by |P&L| desc (150 vs 101)
    ml = _series(CAL, "Mom-Long", "2026-10-28")["s"]
    assert ml[-1]["batch"] == pytest.approx((1 + 150 / 5000) ** (1 / 5) - 1)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_empty_and_missing_g_fall_back():
    assert _series({"days": {}}, "ALL", "2026-11-01")["s"] == []
    s = _series({"days": {"2026-11-01": {"pnl": 5, "trades": 1, "day_return_pct": 1.0}}}, "ALL", "2026-11-02")["s"]
    assert [p["day"] for p in s] == ["2026-11-01"] and s[0]["r"] == pytest.approx(0.01)   # no batch_start_day → the first close day


def test_sleeve_rule_is_shared():
    import main
    assert main._sleeve_label("FRENZY_WILLY", "LONG") == "Frenzy-Willy"
    assert main._sleeve_label("MOMENTUM", "LONG") == "Mom-Long" and main._sleeve_label(None, "SHORT") == "Mom-Short"
    assert main._sleeve_label("MOM_FLIP", "SHORT") == "Flip-Short"
    assert main._sleeve_label("MANUAL", "LONG", "BINANCE_APP_HISTORY") == "App" and main._sleeve_label("MANUAL", "LONG") == "Manual"
    import inspect
    assert "return _sleeve_label(o.entry_strategy, o.direction" in inspect.getsource(main._compute_sleeve_performance)


def test_calendar_payload_carries_the_chart_fields():
    from test_spark_paper_flows import _cal_fixed
    trades = [(dt.datetime(2026, 9, 1, 15), 0.5), (dt.datetime(2026, 9, 3, 15), -1.0)]          # +$50 · −$100 (Mom-Short)
    out = _cal_fixed(trades, [(dt.datetime(2026, 9, 2, 10), "frenzy_willy_invest_mult", "1.0", "0.5")], realized_equity=2950.0)
    d1, d3 = out["days"]["2026-09-01"], out["days"]["2026-09-03"]
    assert d1["sleeves"] == {"Mom-Short": 50.0} and d3["sleeves"] == {"Mom-Short": -100.0}
    assert d3["base"] == pytest.approx(3050.0) and d1["base"] == pytest.approx(3000.0)
    assert d3["g"] == pytest.approx(1 - 100 / 3050) and math.isclose(d1["g"], 1 + 50 / 3000)
    assert out["batch_start_day"] == "2026-09-01"
    assert out["config_marks"] == [{"day": "2026-09-02", "fields": ["frenzy_willy_invest_mult"]}]


def test_ui_block_and_both_exports():
    assert HTML.count('id="dcr-chart"') == 1 and 'id="dcr-chart-stats"' in HTML and 'dcr-chart-sleeve' not in HTML   # no sleeve filter (operator)
    assert HTML.count("lines.push(...dailyCompoundReportLines(hr2));") == 2          # clipboard + saved-file export (D12)
    assert "renderDcrChart();" in HTML[HTML.index("function loadPnlCalendar("):HTML.index("function pnlCalNav(")]
    # placed right after the equity curve
    assert HTML.index('id="equity-curve-chart"') < HTML.index('id="dcr-chart"') < HTML.index('id="pnl-distribution-chart"')


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_no_equity_means_no_series_and_future_start_is_empty():
    days = {"2026-11-01": {"pnl": 5, "trades": 1, "day_return_pct": None}}           # the equity read failed → no % anywhere
    assert _series({"batch_start_day": "2026-11-01", "days": days}, "ALL", "2026-11-03")["s"] == []
    assert _series({"batch_start_day": "2026-12-01", "days": {}}, "ALL", "2026-11-03")["s"] == []
    s = _series({"days": {"2026-11-01": {"pnl": -99, "trades": 1, "g": 0.0}}}, "ALL", "2026-11-01")["s"]   # a wiped-out day stays finite
    assert len(s) == 1 and s[0]["batch"] < -0.99


def test_marks_skip_paper_topups_and_origin_view_drops_the_other_side():
    from test_spark_paper_flows import _cal_fixed
    trades = [(dt.datetime(2026, 9, 1, 15), 0.5)]
    out = _cal_fixed(trades, [(dt.datetime(2026, 9, 2, 10), "paper_balance", "2900", "10000")], realized_equity=10050.0)
    assert out["config_marks"] == []
    import inspect, main
    src = inspect.getsource(main.get_pnl_calendar)
    assert src.index("if not origin_keep(_es, _conf, _origin):") < src.index('_sl = e.setdefault("sleeves", {})')
