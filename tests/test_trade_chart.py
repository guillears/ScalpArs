"""📈 DECISION_LOG 536 — the Closed Orders trade chart: self-contained file, three tagged hooks (easy rollback), pure helpers via node."""
import json
import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "static", "trade_chart.js")


def test_three_tagged_hooks_only():
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert html.count("📈 (536) trade chart") == 3                       # rollback = these 3 lines + the file (or git revert)
    assert '<script src="/static/trade_chart.js?v=1" defer></script>' in html
    assert 'data-order-id="${o.id}"' in html
    assert "window._closedOrdersById = Object.fromEntries(data.map(o => [String(o.id), o]));" in html


def test_display_only_no_bot_api_public_binance_only():
    src = open(JS).read()
    assert "fapi.binance.com/fapi/v1/klines" in src
    assert "/api/" not in src.replace("fapi.binance.com/fapi/v1/klines", "")   # never calls the bot API
    assert "method:" not in src and "POST" not in src
    assert "credentials: 'omit'" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_pure_helpers():
    code = ("global.window={addEventListener(){}};global.document={addEventListener(){}};"
            f"require({json.dumps(JS)});const T=window.TradeChart;"
            "console.log(JSON.stringify([T.parseUtc('2026-11-12T08:20:10'),T.parseUtc('2026-11-12 08:20:10Z'),T.parseUtc(null),T.parseUtc('2026-11-12T08:20:10.422476'),T.barsFor(0,3600,'1m'),"
            "T.pickInterval(0,120),T.pickInterval(0,3600*10),T.pickInterval(0,3600*42),T.pickInterval(0,3600*200),"
            "T.windowFor(1794471610,1794471732,'1m'),"
            "T.priceAtPct({entry_price:100,direction:'LONG'},2),T.priceAtPct({entry_price:100,direction:'SHORT'},2),T.priceAtPct({entry_price:0},2)]))")
    out = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
    assert out[0] == out[1] == 1794471610 and out[2] is None and out[3] == 1794471610 and out[4] == 61
    assert out[5:9] == ["1m", "5m", "15m", "1h"]
    t0, t1 = out[9]
    assert t0 == 1794471610 // 60 * 60 - 30 * 60 and t1 >= 1794471732 + 45 * 60 and t0 % 60 == 0 and t1 % 60 == 0
    assert out[10] == 102 and out[11] == 98 and out[12] is None


def test_review_fixes_present():
    src = open(JS).read()
    assert "tok !== seq" in src and "cache.set(key, p)" in src and "MAX_BARS" in src and "e.detail > 1" in src
    assert "setTimeout(redraw, 200)" in src and "limit=${lim}" in src and "!e.defaultPrevented" in src
