"""📆 DECISION_LOG 426: the Monthly P&L section — the daily calendar's own days grouped by month. Pins the money math against the daily header's conventions, the section's default state and both exports."""
import json
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()


def _pure():
    a = HTML.index("function pnlMonthlySummary(")
    b = HTML.index("function _pnlSignedPct(")
    return HTML[a:b]


def _run(days, today):
    prog = _pure() + f"const m = pnlMonthlySummary({json.dumps(days)}, {json.dumps(today)}); console.log(JSON.stringify({{m, t: pnlMonthlyTotal(m)}}));"
    return json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)


DAYS = {
    "2026-10-23": {"pnl": 100.0, "trades": 2, "wins": 2, "day_return_pct": 10.0},
    "2026-10-30": {"pnl": -50.0, "trades": 2, "wins": 1, "day_return_pct": -10.0},
    "2026-12-02": {"pnl": 30.0, "trades": 1, "wins": 1, "day_return_pct": None},
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_month_math_matches_the_daily_header():
    r = _run(DAYS, "2026-12-02")
    oct_, nov, dec = r["m"]
    assert [x["key"] for x in r["m"]] == ["2026-10", "2026-11", "2026-12"]
    assert oct_["pnl"] == 50.0 and oct_["trades"] == 4 and oct_["wr"] == 75.0
    assert oct_["compoundPct"] == pytest.approx(-1.0)               # 1.10 × 0.90
    assert oct_["elapsed"] == 9                                     # 23rd → 31st, zero-trade days count
    assert oct_["dcrPct"] == pytest.approx((0.99 ** (1 / 9) - 1) * 100)
    assert (oct_["greenDays"], oct_["redDays"], oct_["activeDays"]) == (1, 1, 2)
    assert oct_["best"]["day"] == "2026-10-23" and oct_["worst"]["day"] == "2026-10-30"
    assert oct_["partial"] is True                                  # started on the 23rd
    assert nov["trades"] == 0 and nov["elapsed"] == 30 and nov["partial"] is False and nov["compoundPct"] is None
    assert dec["compoundPct"] is None and dec["partial"] is True    # the running month; a null day % stays out of the compound
    assert r["t"]["pnl"] == 80.0 and r["t"]["compoundPct"] == pytest.approx(-1.0) and r["t"]["elapsed"] == 9 + 30 + 2
    assert (r["t"]["greenMonths"], r["t"]["redMonths"]) == (2, 0)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_today_counts_only_once_it_has_a_trade_and_empty_is_empty():
    days = {"2026-11-01": {"pnl": 5.0, "trades": 1, "wins": 1, "day_return_pct": 1.0}}
    assert _run(days, "2026-11-03")["m"][0]["elapsed"] == 2
    assert _run(days, "2026-11-01")["m"][0]["elapsed"] == 1
    assert _run({}, "2026-11-01")["m"] == []


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_a_full_month_that_ended_is_not_partial():
    days = {"2026-10-01": {"pnl": 5.0, "trades": 1, "wins": 1, "day_return_pct": 1.0}}
    assert _run(days, "2026-11-01")["m"][0]["partial"] is False
    assert _run(days, "2026-10-31")["m"][0]["partial"] is True


def test_minimized_by_default_redrawn_with_the_calendar_and_in_both_exports():
    assert re.search(r'id="pnl-monthly-card"[^>]*style="display: none;"', HTML)
    assert 'id="pnl-mon-chevron" class="text-xs text-gray-500">▶' in HTML
    assert "_renderEomPanel();\n            renderPnlMonthly();" in HTML
    assert HTML.count("lines.push(...monthlyPnlReportLines(hr2));") == 2
    assert HTML.index('id="pnl-calendar-card"') < HTML.index('id="pnl-monthly-card"') < HTML.index("<!-- Performance Metrics -->")


def _stub_prog(body):
    a, b = HTML.index("// ===== 📆 Monthly P&L"), HTML.index("// ── EOM Projection Revisions")
    fa, fb = HTML.index("function _pnlCalFmt$("), HTML.index("function renderPnlCalendar()")
    return ("const els = {}; const document = {getElementById: i => (els[i] = els[i] || {textContent: '', innerHTML: '', className: ''})};\n"
            "const ORIGIN_LABELS = {ALL: 'All trades', BOT: '🤖 Bot only', MANUAL: '🖐 Manual only'};\n"
            "let pnlCalData = null, pnlCalDataView = 'ALL', pnlCalYear = null, pnlCalMonth = null;\n"
            + HTML[fa:fb] + HTML[a:b] + body)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_renderer_and_export_run_on_real_shapes():
    prog = _stub_prog(f"""
const out = {{}};
out.none = monthlyPnlReportLines('--').join('|');
pnlCalData = {{detail: 'boom'}};
out.err = monthlyPnlReportLines('--').join('|');
pnlCalData = {{days: {json.dumps(DAYS)}}}; pnlCalDataView = 'BOT'; pnlMonYear = 2026;
renderPnlMonthly();
out.cells = els['pnl-monthly-grid'].innerHTML; out.total = els['pnl-mon-year-total'].textContent;
out.rep = monthlyPnlReportLines('--').join('|');
pnlMonYear = 2024; pnlCalData = {{days: {{'2023-12-31': {{pnl: 1, trades: 1, wins: 1, day_return_pct: 1}}}}}};
renderPnlMonthly();
out.empty = els['pnl-mon-year-total'].textContent;
console.log(JSON.stringify(out));
""")
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert "Not loaded yet" in r["none"] and "Not loaded yet" in r["err"]
    assert "pnlMonOpen(2026,10)" in r["cells"] and "· partial" in r["cells"] and "no trades" in r["cells"]
    assert "undefined" not in r["cells"] and "NaN" not in r["cells"]
    assert r["total"].startswith("+$80.00 in 2026")
    assert "View: 🤖 Bot only" in r["rep"] and "2026-11" in r["rep"] and "Total" in r["rep"] and "undefined" not in r["rep"]
    assert r["empty"] == "No trades in 2024"
