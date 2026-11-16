"""⏬ 2026-11-16 (DECISION_LOG 560): momentum_short_lev_mult — every MOMENTUM SHORT open 20× → 15× (× the cell lev multiplier); longs,
flips, fades, SURGE, Bear-Run, Flatbase, manual never. D11 bundle: config default 1.0 + JSON 0.75 + engine wiring + UI input + load / save +
the export line."""
import json
import math
import os
from types import SimpleNamespace as NS

import pytest

import config
import services.trading_engine as te

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
ENG = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
F = te.momentum_short_lev_factor
TH = NS(momentum_short_lev_mult=0.75)


@pytest.mark.parametrize("es,d,want", [
    ("MOMENTUM", "SHORT", 0.75), ("MOMENTUM", "short", 0.75),
    ("MOMENTUM", "LONG", 1.0),
    ("FLIP:FAN_RATIO_GATE", "SHORT", 1.0), ("SPIKE_FADE", "SHORT", 1.0), ("BEARRUN_SHORT", "SHORT", 1.0), ("SURGE_SHORT", "SHORT", 1.0),
    ("FLATBASE_SHORT", "SHORT", 1.0), ("MANUAL", "SHORT", 1.0), (None, "SHORT", 1.0), ("", "SHORT", 1.0),
])
def test_factor_scope(es, d, want):
    assert F(TH, es, d) == want


@pytest.mark.parametrize("v,want", [(0.75, 0.75), (1.0, 1.0), (0.0, 1.0), (-1, 1.0), (None, 1.0), ("x", 1.0), (float("nan"), 1.0),
                                    (1.5, 1.0), (0.01, te.MOMENTUM_LONG_LEV_MULT_MIN)])
def test_factor_bounds(v, want):
    assert F(NS(momentum_short_lev_mult=v), "MOMENTUM", "SHORT") == want
    assert F(NS(), "MOMENTUM", "SHORT") == 1.0


def test_leverage_20_to_15():
    assert te.compose_leverage(20, 1.0, True, 0.75) == 15                     # base momentum short
    assert te.compose_leverage(20, 1.0, True, 1.0 * 0.75) == 15               # engine passes _ml_lev (1.0 for shorts) × _ms_lev
    assert te.compose_leverage(20, 1.0, True, 1.0) == 20                      # off = unchanged


def test_longs_unchanged_by_the_short_knob():
    th = NS(momentum_short_lev_mult=0.75, momentum_long_lev_mult=0.75)
    ml, ms = te.momentum_long_lev_factor(th, "MOMENTUM", "LONG"), F(th, "MOMENTUM", "LONG")
    assert ms == 1.0 and te.compose_leverage(20, 1.0, True, ml * ms) == 15


def test_d11_wiring():
    assert config.SignalThresholds().momentum_short_lev_mult == 1.0
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["momentum_short_lev_mult"] == 0.75
    assert "base_lev_multiplier=_ml_lev * _ms_lev" in ENG
    assert "_ms_lev = momentum_short_lev_factor(config.trading_config.thresholds, _wh_es, direction)" in ENG
    assert HTML.count("config-momentum-short-lev-mult") >= 4                # input + loader + save + sizing-table preview
    assert "momentum_short_lev_mult: (function()" in HTML
    assert "Momentum-short leverage (Nov 16, DECISION_LOG 560)" in HTML      # shared export line
