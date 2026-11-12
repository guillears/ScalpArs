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
    assert '<script src="/static/trade_chart.js?v=2" defer></script>' in html
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
    code = ("global.window={addEventListener(){}};global.document={addEventListener(){},getElementById(){return null}};"
            f"require({json.dumps(JS)});const T=window.TradeChart;"
            "console.log(JSON.stringify([T.parseUtc('2026-11-12T08:20:10'),T.parseUtc('2026-11-12 08:20:10Z'),T.parseUtc(null),T.parseUtc('2026-11-12T08:20:10.422476'),T.barsFor(0,3600,'1m'),"
            "T.pickInterval(0,120),T.pickInterval(0,3600*10),T.pickInterval(0,3600*42),T.pickInterval(0,3600*200),"
            "T.windowFor(1794471610,1794471732,'1m'),"
            "T.priceAtPct({entry_price:100,direction:'LONG',entry_fee:0,notional_value:100},1.91),T.priceAtPct({entry_price:100,direction:'SHORT',entry_fee:0,notional_value:100},1.91),T.priceAtPct({entry_price:0},2),"
            "T.priceAtPct({entry_price:2.614,direction:'LONG',entry_fee:16.45,notional_value:36562.5},-0.5871),"
            "T.feePct({entry_fee:1.8,notional_value:10000}),T.feePct({entry_fee:null,notional_value:10000}),T.feePct({entry_fee:0,notional_value:10000}),"
            "T.priceAtPct({entry_price:100,direction:'SHORT',entry_fee:4.5,notional_value:10000},-0.5)]))")
    out = json.loads(subprocess.run(["node", "-e", code], capture_output=True, text=True, check=True).stdout)
    assert out[0] == out[1] == 1794471610 and out[2] is None and out[3] == 1794471610 and out[4] == 61
    assert out[5:9] == ["1m", "5m", "15m", "1h"]
    t0, t1 = out[9]
    assert t0 == 1794471610 // 60 * 60 - 30 * 60 and t1 >= 1794471732 + 45 * 60 and t0 % 60 == 0 and t1 % 60 == 0
    assert abs(out[10] - 102) < 1e-9 and abs(out[11] - 98) < 1e-9 and out[12] is None   # 1.91 net + 0.09 (no entry fee → taker both legs)
    assert abs(out[13] - 2.6011) < 0.0001          # ICP 11-11 (taker entry 0.045 %): worst −0.587 % net + 0.09 % fees → 2.6011 = the real 1m low 2.601
    assert abs(out[14] - 0.063) < 1e-9             # maker entry 0.018 % + taker exit 0.045 % (not 2 × the maker fee)
    assert abs(out[15] - 0.09) < 1e-9 and abs(out[16] - 0.09) < 1e-9   # missing / 0 entry fee → taker on both legs
    assert abs(out[17] - 100.41) < 1e-9            # SHORT: −0.5 % net + 0.09 % fees → price 0.41 % ABOVE the entry


def test_review_fixes_present():
    src = open(JS).read()
    assert "tok !== seq" in src and "cache.set(key, p)" in src and "MAX_BARS" in src and "e.detail > 1" in src
    assert "setTimeout(redraw, 200)" in src and "limit=${lim}" in src and "!e.defaultPrevented" in src
