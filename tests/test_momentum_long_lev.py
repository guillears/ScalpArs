"""⚖️ 2026-10-12 (DECISION_LOG 283): momentum_long_lev_mult — every MOMENTUM LONG open 20× → 10× (× the cell lev multiplier, after its
2.0 hard cap and floors); shorts / FRENZY / SURGE / BULLRUN / BEARRUN / flips / fades / spikes / bull- and bounce-long / MANUAL never.
Pure helpers: services.trading_engine.momentum_long_lev_factor / compose_leverage. D11 bundle: config default 1.0 + JSON 0.5 + engine
wiring + UI input + load / save + the export line (shared _buildConfigLines → both text exports)."""
import json
import math
import os
import re
from types import SimpleNamespace as NS

import pytest

import services.trading_engine as te

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
ENG = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
F = te.momentum_long_lev_factor
TH = NS(momentum_long_lev_mult=0.5)


@pytest.mark.parametrize("es,d,want", [
    ("MOMENTUM", "LONG", 0.5), ("MOMENTUM", "long", 0.5),
    ("MOMENTUM", "SHORT", 1.0),
    ("FRENZY_LONG", "LONG", 1.0), ("FRENZY_WIDE", "LONG", 1.0), ("FRENZY_LITE", "LONG", 1.0), ("FRENZY_WILLY", "LONG", 1.0),
    ("SURGE_LONG", "LONG", 1.0), ("BULLRUN_LONG", "LONG", 1.0), ("BEARRUN_SHORT", "SHORT", 1.0),
    ("FLIP:FAN_RATIO_GATE", "LONG", 1.0), ("FLIP:PAIR_RSI_OB", "SHORT", 1.0),
    ("SPIKE_CHASE", "LONG", 1.0), ("SPIKE_BOUNCE", "LONG", 1.0), ("SPIKE_FADE", "SHORT", 1.0),
    ("BULL_LONG", "LONG", 1.0), ("BOUNCE_LONG", "LONG", 1.0), ("MANUAL", "LONG", 1.0), (None, "LONG", 1.0), ("", "LONG", 1.0),
])
def test_factor_scope(es, d, want):
    assert F(TH, es, d) == want


@pytest.mark.parametrize("v,want", [(0.5, 0.5), (1.0, 1.0), (0.0, 1.0), (-0.5, 1.0), (None, 1.0), ("x", 1.0), (float("nan"), 1.0),
                                    (1.7, 1.0), (0.01, te.MOMENTUM_LONG_LEV_MULT_MIN), (0.25, 0.25), (0.05, 0.05)])   # 0.05 = the live value since 289
def test_factor_bounds(v, want):
    """de-lever only: ≤ 0 / blank / unreadable = 1.0 (off), clamped to [0.05, 1.0]."""
    assert F(NS(momentum_long_lev_mult=v), "MOMENTUM", "LONG") == want
    assert F(NS(), "MOMENTUM", "LONG") == 1.0                                   # missing key = no change (config default 1.0)


@pytest.mark.parametrize("base,cell,apply,ml,want", [
    (20, 1.0, True, 0.5, 10),        # UNMATCHED 1× cell → 10×
    (20, 2.0, True, 0.5, 20),        # a 2.0-capped cell lev → 20 × 2 × 0.5
    (20, 1.5, True, 0.5, 15),
    (20, 0.05, True, 0.5, 1),        # a 1× probe stays 1× (round(0.5) = 0 → floor 1)
    (20, 0.5, True, 0.5, 5),         # the 0.5 momentum cell floor × 0.5
    (20, 2.0, False, 0.5, 10),       # target 'investment': the cell lev is inert, the momentum factor still applies
    (20, 2.0, True, 1.0, 40),        # pre-283 behaviour unchanged
    (20, 1.0, True, 1.0, 20),
    (20, 1.0, True, 0.05, 1),        # 289 live: UNMATCHED 1× cell → 1×
    (20, 2.0, True, 0.05, 2),        # 289 live: a 2.0-lev cell → 2×
    (20, 0.05, True, 0.05, 1),       # 289 live: a 1× probe stays 1× (floor 1)
])
def test_compose_leverage(base, cell, apply, ml, want):
    assert te.compose_leverage(base, cell, apply, ml) == want


def test_compose_leverage_untouched_when_nothing_multiplies():
    v = te.compose_leverage(20.0, 1.0, True, 1.0)
    assert v == 20.0 and isinstance(v, float)                                   # no rounding / int cast, exactly as before 283


def test_calculate_position_size_applies_the_factor():
    eng = te.trading_engine
    _, l0, _ = eng.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0, cell_lev_multiplier=1.0, multiplier_target="both")
    _, l1, _ = eng.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0, cell_lev_multiplier=1.0, multiplier_target="both",
                                           base_lev_multiplier=0.5)
    _, l2, _ = eng.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0, cell_lev_multiplier=2.0, multiplier_target="both",
                                           base_lev_multiplier=0.5)
    assert l0 == 20 and l1 == 10 and l2 == 20


def test_open_position_wiring():
    i = ENG.index("    async def open_position("); j = ENG.index("\n    async def ", i + 10)
    body = ENG[i:j]
    k = body.index("_ml_lev = momentum_long_lev_factor(config.trading_config.thresholds, _wh_es, direction)")
    c = body.index("investment, leverage, cell_capped = self.calculate_position_size(")
    assert body.index("_wh_es = (") < k < c and "multiplier_target=_mult_target, base_lev_multiplier=_ml_lev," in body[c:c + 400]
    # every sizing override (flip / bull / bounce / bull-run / bear-run / probes / spikes / doors / SURGE-FRENZY) sits BEFORE the factor
    for marker in ("if flip_source:\n            cell_mult, cell_lev_mult, cell_src = _flip_size_mult", "if _surge:\n            _th_sg",
                   "if _adx_surge_admit:", "if spike_chase_probe or spike_fade or spike_bounce or nonexp_calm3d:",
                   "if (cross_ob_open and", "if ((gap_probe or"):
        assert body.index(marker) < k, marker
    # the label the factor keys on IS the Order's entry_strategy (one expression)
    expr = body[body.index("_wh_es = ") + len("_wh_es = "):body.index("\n", body.index("_wh_es = "))]
    assert ("entry_strategy=" + expr.strip() + ",") in body


def test_config_d11():
    import config as C
    assert C.SignalThresholds.model_fields["momentum_long_lev_mult"].default == 1.0          # code default = no change
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["momentum_long_lev_mult"] == 0.25 and cfg["momentum_long_invest_mult"] == 0.2   # live: 5× at 0.2 margin = the 1× notional (DL 298; 289: 0.05 = 1×; 283: 0.5 = 10×)
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 283" in src and "momentum_long_lev_mult: float = 1.0" in src and "No revert gate (de-risking)" in src and "DECISION_LOG 289" in src


def test_ui_d11_and_exports():
    assert HTML.count('id="config-momentum-long-lev-mult"') == 1                            # the input (Sleeve Sizing, Momentum LONG base row)
    assert HTML.count("document.getElementById('config-momentum-long-lev-mult')") == 2      # load + save handlers
    assert "_e.value = config.thresholds.momentum_long_lev_mult !== undefined ? config.thresholds.momentum_long_lev_mult : 1;" in HTML
    assert "momentum_long_lev_mult: (function(){ const v = parseFloat((document.getElementById('config-momentum-long-lev-mult')" in HTML
    assert "_ssNum('config-momentum-long-lev-mult', 1)" in HTML                             # the sizing preview
    # the export text line lives in _buildConfigLines, which BOTH text exports call (clipboard + saved file)
    b = HTML.index("function _buildConfigLines(")
    e = HTML.index("\n        function ", b + 10)
    assert "⚖️ Momentum-long leverage (Oct 12, DECISION_LOG 283)" in HTML[b:e] and "th.momentum_long_lev_mult" in HTML[b:e]
    assert HTML.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2


def test_master_rule_matches_engine():
    """the master / screen today_size_rule prices a 20× momentum long at today's engine leverage (compose_leverage)."""
    from scripts import build_master_pool as B
    for cell in (1.0, 1.5, 2.0):
        traded = te.compose_leverage(20, cell, True, 1.0)
        today = te.compose_leverage(20, cell, True, B.MOMENTUM_LONG_LEV_FROZEN)
        f = B.momentum_long_lev_scale(cell, cell, traded)
        assert math.isclose(traded * f, today), cell
