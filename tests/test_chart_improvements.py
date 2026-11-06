"""📊 DECISION_LOG 494: the dashboard chart improvements — server data (adaptive time windows, real-time equity points, −5…+8
0.5 % histogram frame, breakeven WR, distinct days per hour / weekday, heatmap $) and the shared JS helpers + both exports."""
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()


def _o(closed, pnl, pct, direction="LONG", es="MOMENTUM"):
    return NS(closed_at=closed, opened_at=closed - dt.timedelta(minutes=30), pnl=pnl, pnl_percentage=pct, direction=direction,
              entry_strategy=es, entry_rsi=None, entry_gap=None, entry_adx=None, entry_ema_gap_5_8=None, entry_ema_gap_8_13=None,
              trough_pnl=min(pct, 0), peak_pnl=max(pct, 0), pair="XUSDT")


T0 = dt.datetime(2026, 10, 24, 15, 7)   # Sat 12:07 UTC-3


def test_time_windows_follow_the_span():
    import main
    one_day = [_o(T0 + dt.timedelta(hours=h), 1, 0.1) for h in (0, 5)]
    assert main._compute_time_buckets(one_day)[0]["bucket_min"] == 15
    three = [_o(T0, 1, 0.1), _o(T0 + dt.timedelta(hours=60), 1, 0.1)]
    assert main._compute_time_buckets(three)[0]["bucket_min"] == 60
    two_weeks = [_o(T0 + dt.timedelta(days=d), 1, 0.1) for d in range(14)]
    b = main._compute_time_buckets(two_weeks)
    assert b[0]["bucket_min"] == 240
    assert len(b) == 14 * 6 - 5 and sum(x["trades"] for x in b) == 14            # filled → a real linear-time axis (≤ MAX_BUCKETS)
    assert b[0]["time"].endswith("12:00")                                         # 4-h windows clock-aligned in UTC-3 (12:07 → 12:00)
    long_run = [_o(T0 + dt.timedelta(days=d), 1, 0.1) for d in (0, 40)]
    assert main._compute_time_buckets(long_run)[0]["bucket_min"] == 1440
    assert main._compute_time_buckets(two_weeks, bucket_minutes=15)[0]["bucket_min"] == 15   # explicit size still honoured


def test_equity_points_carry_real_time():
    import main
    pts = main._compute_equity_curve([_o(T0, 10, 0.1), _o(T0 + dt.timedelta(hours=3), -4, -0.2)])
    assert pts[0]["ts"] == int(T0.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert pts[1]["ts"] - pts[0]["ts"] == 3 * 3600 * 1000 and pts[1]["cumulative"] == 6


def test_histogram_frame_and_breakeven_wr():
    import main
    orders = [_o(T0, 1, p) for p in (-15.1, -4.0, -3.0, -0.3, 0.2, 1.3, 5.0, 7.9, 8.0)]
    b = main._compute_pnl_distribution(orders)
    assert b[0]["label_range"] == "< -5.00%" and b[0]["count"] == 1                 # only Flatbase's −15 overflows
    assert b[-1]["label_range"] == "≥ +8.00%" and b[-1]["count"] == 1
    assert len(b) == 26 + 2 and all(round(x["hi"] - x["lo"], 2) == 0.5 for x in b[1:-1])
    assert sum(x["count"] for x in b) == len(orders)
    st = main._compute_pnl_distribution_stats(orders)
    aw, al = st["avg_win"], st["avg_loss"]
    assert st["breakeven_wr"] == round(100 * abs(al) / (aw + abs(al)), 1)
    assert main._compute_pnl_distribution_stats([_o(T0, 1, 0.5)])["breakeven_wr"] is None   # no loss → no breakeven


def test_days_per_hour_and_weekday_and_heatmap_usd():
    import main
    sats = [T0, T0 + dt.timedelta(minutes=20), T0 + dt.timedelta(days=7)]          # 2 Saturdays, 3 trades, all at 12:xx local
    orders = [_o(t, 10 * (i + 1), 0.1 * (i + 1)) for i, t in enumerate(sats)]
    sat = main._compute_daily_performance(orders)[5]
    assert sat["label"] == "Sat" and sat["trades"] == 3 and sat["days"] == 2
    assert main._compute_daily_performance(orders)[0]["days"] == 0
    h12 = main._compute_hourly_performance(orders)[12]
    assert h12["trades"] == 3 and h12["days"] == 2
    cell = [c for c in main._compute_day_time_heatmap(orders) if c["day"] == 5 and c["block"] == 3][0]
    assert cell["pnl"] == 60.0 and cell["trades"] == 3
    assert [c for c in main._compute_day_time_heatmap(orders) if c["trades"] == 0][0]["pnl"] == 0.0


def _js(expr):
    a = HTML.index("function dtHeatmapAgg(")
    b = HTML.index("// Initialize", a)
    prog = HTML[a:b] + f"\nconsole.log(JSON.stringify({expr}));"
    return json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_round_axis_and_heatmap_helpers():
    assert _js("[niceHalfRange(1064), niceHalfRange(4911), niceHalfRange(7.3), niceHalfRange(0)]") == [1500, 6000, 8, 1]   # round end ticks (steps 500 / 2,000 / 2)
    agg = _js("dtHeatmapAgg([{trades:2,wins:1,avg_pnl_pct:1,pnl:30},{trades:3,wins:3,avg_pnl_pct:-1,pnl:-12.4},{trades:0}])")
    assert agg["trades"] == 5 and agg["pnl"] == pytest.approx(17.6)
    s = _js("dtHeatmapCellStr({trades:5,win_rate:80,avg_pnl_pct:-0.5,pnl:-12.4})")
    assert s.strip() == "5t 80%W -0.50% -$12" and len(s) == 24


def test_ui_and_both_exports_wired():
    assert HTML.count("lines.push(...chartSummaryReportLines(perf, hr2));") == 2
    assert 'id="mae-mfe-stats"' in HTML and 'id="performance-over-time-caption"' in HTML
    assert HTML.count("${'Days'.padStart(5)}") == 2                                # weekday export: distinct days column, both exports
    assert HTML.count("blockLabels.map(b => b.padStart(24))") == 2                 # heatmap export widened for the $ column
    assert "breakeven_wr" in HTML and "DOW_MIN_DAYS = 4" in HTML and "MM_CLIP = -6" in HTML


def test_equity_curve_stats_one_source_and_filtered_view_has_no_pct():
    import main
    curve = [{"cumulative": c} for c in (-50, 100, -400, -300)]                 # opening loss, peak +100, trough −400
    st = main._equity_curve_stats(curve, 10000)
    assert st["peak"] == 100 and st["max_dd"] == -500 and st["final"] == -300
    assert st["max_dd_pct"] == round(-500 / 10100 * 100, 2)
    assert main._equity_curve_stats(curve, None)["max_dd_pct"] is None          # filtered view → $ only
    assert main._equity_curve_stats([], 10000) is None
    import inspect
    src = inspect.getsource(main._compute_performance)
    assert "\"equity_curve_stats\": _equity_curve_stats(" in src and "_unfiltered_view = (not any((regime, window_hours" in src


def test_export_titles_name_the_window_size():
    assert "Performance Over Time (15min buckets)" not in HTML
    assert HTML.count("b0.bucket_min >= 1440 ? '1-day'") == 2
    assert HTML.count("(b.time||'').padEnd(11)") == 2


def test_real_performance_payload_runs_and_carries_the_new_fields():
    """CLAUDE.md: runtime-exercise /api/performance — it swallows exceptions into an all-zeros payload."""
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    from test_origin_view import _perf
    allv = _perf(False, None, 10_000.0)
    assert "_error" not in allv and allv["total_trades"] > 0
    es = allv["equity_curve_stats"]
    assert es and es["max_dd_pct"] is not None and es["trades"] == allv["total_trades"]
    assert allv["performance_over_time"][0]["bucket_min"] in (15, 60, 240, 1440)
    assert "breakeven_wr" in allv["pnl_distribution_stats"]
    assert all("days" in d for d in allv["daily_performance"]) and all("pnl" in c for c in allv["day_time_heatmap"])
    assert all("ts" in p for p in allv["equity_curve"])
    bot = _perf(True, "BOT", 10_000.0)
    assert "_error" not in bot and bot["equity_curve_stats"]["max_dd_pct"] is None      # a filtered view → $ only


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_pot_balance_walk_steps_at_the_topup():
    """B25 shape (DL 495): $2,900 start, -$968 on day 1, a +$7,100 top-up, then +$100 - the line must start at $2,900, not today's seed."""
    T = 1_000_000_000_000
    buckets = [{"t_end_ms": T + 1, "pnl": -968, "cumulative_pnl": -968}, {"t_end_ms": T + 3, "pnl": 100, "cumulative_pnl": -868}]
    flows = [{"t": T + 2, "amount": 7100}]
    now = 2900 - 968 + 7100 + 100
    cal = {"portfolio_now": now, "days": {"d1": {"pnl": -968}, "d2": {"pnl": 100}}, "flows": flows}
    S = _js(f"potStartBalance({json.dumps(cal)})")
    assert S == 2900
    w = _js(f"potBalanceWalk({json.dumps(buckets)}, {S}, {json.dumps(flows)})")
    assert w["values"] == [2900 - 968, now]
    assert w["capital"] == [2900, 10000]                      # the capital line steps at the top-up
    # (review) a performance payload NEWER than the calendar (a fresh +$50 close) moves only the last point, never the start
    b2 = buckets + [{"t_end_ms": T + 5, "pnl": 50, "cumulative_pnl": -818}]
    w2 = _js(f"potBalanceWalk({json.dumps(b2)}, {S}, {json.dumps(flows)})")
    assert w2["values"][:2] == w["values"] and w2["values"][2] == now + 50
    # a flow exactly at a window end belongs to the NEXT window (end exclusive); a flow before the first end is in window 1
    assert _js(f"potBalanceWalk([{{\"t_end_ms\": 10, \"cumulative_pnl\": 0}}], 100, [{{\"t\": 10, \"amount\": 5}}])")["capital"] == [100]
    assert _js(f"potBalanceWalk([{{\"t_end_ms\": 10, \"cumulative_pnl\": 0}}], 100, [{{\"t\": 3, \"amount\": 5}}])")["capital"] == [105]
    assert _js("potStartBalance({days: {}})") is None


def test_bar_colours_follow_avg_pnl_and_dcr_axes_align():
    assert "return h.avg_pnl_pct >= 0 ? `rgba(52, 211, 153, ${a})`" in HTML and "return h.win_rate >= 50 ?" not in HTML
    assert "return d.avg_pnl_pct >= 0 ? `rgba(52, 211, 153, ${a})`" in HTML and "return d.win_rate >= 50 ?" not in HTML
    assert "min: -_dcrL, max: _dcrL" in HTML and "min: -_dcrR, max: _dcrR" in HTML
    assert "min: eqData[0].ts, max: eqData[eqData.length - 1].ts" in HTML


def test_time_windows_carry_their_end_ms():
    import main
    b = main._compute_time_buckets([_o(T0, 1, 0.1), _o(T0 + dt.timedelta(days=5), 1, 0.1)])
    bm = b[0]["bucket_min"]
    assert all(x["t_end_ms"] - y["t_end_ms"] == bm * 60000 for y, x in zip(b, b[1:]))
    first_end = dt.datetime(2026, 10, 24, 16, 0, tzinfo=dt.timezone(dt.timedelta(hours=-3)))   # 4-h window 12:00–16:00 UTC-3
    assert b[0]["t_end_ms"] == int(first_end.timestamp() * 1000)


def test_pot_redraws_when_the_calendar_arrives_and_equity_axis_guard():
    i = HTML.index("function loadPnlCalendar(")
    assert "_rfGuard('perf', loadPerformance);" in HTML[i:HTML.index("function pnlCalNav(", i)]
    assert "window._potWantsCal = !!(data.unfiltered_view" in HTML
    assert "eqData[eqData.length - 1].ts > eqData[0].ts ? { min: eqData[0].ts" in HTML
