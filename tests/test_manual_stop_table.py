"""🖐 The dashboard's read-only "manual stop by leverage" table must use the engine's constants and formula."""
import os
import re

import services.trading_engine as TE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HTML = open(os.path.join(ROOT, "templates", "index.html")).read()


def test_ui_constants_match_the_engine():
    m = re.search(r"const MANUAL_LIQ_SAFETY_UI = ([0-9.]+), MANUAL_MAINT_MARGIN_PCT_UI = ([0-9.]+);", HTML)
    assert m and float(m.group(1)) == TE.MANUAL_LIQ_SAFETY and float(m.group(2)) == TE.MANUAL_MAINT_MARGIN_PCT


def test_ui_formula_matches_the_engine_at_every_table_row():
    assert "const liq = Math.max(0, 100 / lev - MANUAL_MAINT_MARGIN_PCT_UI);" in HTML
    assert "stop: Math.max(cfg, -liq * MANUAL_LIQ_SAFETY_UI)" in HTML

    class TH:
        manual_floor_sl_pct = -3.0
    js = lambda lev, cfg=-3.0: max(cfg, -max(0.0, 100.0 / lev - TE.MANUAL_MAINT_MARGIN_PCT) * TE.MANUAL_LIQ_SAFETY)   # the JS, transcribed
    for lev in (5, 10, 20, 25, 30, 40, 50, 75, 100, 125):
        assert abs(js(lev) - TE.manual_floor_for_leverage(TH, lev)) < 1e-4
    TH2 = type("T", (), {"manual_floor_sl_pct": -2.0}); TH3 = type("T", (), {"manual_floor_sl_pct": 3.0})   # a tighter floor; a positive typed value
    for lev in (5, 20, 40, 100):
        assert abs(js(lev, -2.0) - TE.manual_floor_for_leverage(TH2, lev)) < 1e-4 and abs(js(lev, -3.0) - TE.manual_floor_for_leverage(TH3, lev)) < 1e-4
    assert "Math.floor(Math.abs(r.stop) * 100 + 1e-9) / 100" in HTML                                    # shown rounded toward the SAFE side (30× → 2.26, never 2.27)
    import math
    for lev in range(1, 126):                                                                         # the shown value is never wider than the engine's
        e = abs(TE.manual_floor_for_leverage(TH, lev)); shown = math.floor(e * 100 + 1e-9) / 100
        assert shown <= e + 1e-12 and e - shown <= 0.01
    assert TE.manual_floor_for_leverage(TH, 50) == -1.2 and TE.manual_floor_for_leverage(TH, 40) == -1.6 and TE.manual_floor_for_leverage(TH, 20) == -3.0


def test_table_is_wired():
    assert HTML.count('id="manual-stop-table"') == 1 and HTML.count("renderManualStopTable();") >= 2      # config load + the input listener
    assert "e.target.id === 'config-manual-floor-sl-pct'" in HTML
