"""⚖️ 2026-10-15 (DECISION_LOG 298): momentum_long_invest_mult — momentum LONG keeps the SAME notional and locks less margin (JSON 0.2 with
momentum_long_lev_mult 0.25 = 5× at 0.2 margin = the notional of 1× at the full margin). Factor scope / clamps, the open_position sizing
step on the REAL calculate_position_size (open_position calls momentum_long_resize right after it), and the D11 bundle."""
import json
import math
import os
from types import SimpleNamespace as NS

import pytest

import services.trading_engine as te

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
ENG = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
MAIN = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
F = te.momentum_long_inv_factor


@pytest.mark.parametrize("es,d,want", [
    ("MOMENTUM", "LONG", 0.2), ("MOMENTUM", "long", 0.2), ("MOMENTUM", "SHORT", 1.0),
    ("FRENZY_LONG", "LONG", 1.0), ("FRENZY_WILLY", "LONG", 1.0), ("FRENZY_LITE", "LONG", 1.0), ("SURGE_LONG", "LONG", 1.0),
    ("BULLRUN_LONG", "LONG", 1.0), ("BEARRUN_SHORT", "SHORT", 1.0), ("FLIP:FAN_RATIO_GATE", "LONG", 1.0), ("SPIKE_FADE", "SHORT", 1.0),
    ("SPIKE_CHASE", "LONG", 1.0), ("BULL_LONG", "LONG", 1.0), ("BOUNCE_LONG", "LONG", 1.0), ("MANUAL", "LONG", 1.0), (None, "LONG", 1.0),
])
def test_factor_scope(es, d, want):
    assert F(NS(momentum_long_invest_mult=0.2), es, d) == want


@pytest.mark.parametrize("v,want", [(0.2, 0.2), (1.0, 1.0), (1.5, 1.0), (0.01, 0.05), (0.0, 1.0), (-1, 1.0), (None, 1.0), ("x", 1.0),
                                    (float("nan"), 1.0), (float("inf"), 1.0)])
def test_factor_bounds(v, want):
    assert F(NS(momentum_long_invest_mult=v), "MOMENTUM", "LONG") == want
    assert F(NS(), "MOMENTUM", "LONG") == 1.0                                            # missing key = off


def test_margin_rule():
    M = te.momentum_long_margin
    assert math.isclose(M(1000.0, 5, 1), 200.0) and M(1000.0, 1, 1) == 1000.0 and M(1000.0, 1, 5) == 1000.0   # never above the full margin
    assert M(0, 5, 1) == 0 and M(1000.0, 0, 1) == 1000.0


def _sizes(monkeypatch, inv_mult, lev_mult, cell_lev=1.0, cell_inv=1.0, target="both", es="MOMENTUM", d="LONG", sched=None):
    th = te.config.trading_config.thresholds
    monkeypatch.setattr(th, "momentum_long_invest_mult", inv_mult, raising=False)
    monkeypatch.setattr(th, "momentum_long_lev_mult", lev_mult, raising=False)
    if sched is not None:
        monkeypatch.setattr(te.config.trading_config.investment, "leverage_balance_schedule", sched, raising=False)
    eng = te.trading_engine
    ml = te.momentum_long_lev_factor(th, es, d)
    inv, lev, _ = eng.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0, cell_multiplier=cell_inv,
                                              cell_lev_multiplier=cell_lev, multiplier_target=target, base_lev_multiplier=ml)
    inv2, scale, ref = eng.momentum_long_resize("XUSDT", inv, lev, 10_000.0, "STRONG_BUY", 10_000.0, cell_inv, cell_lev, target, es, d, ml)
    assert (ref is None) == (scale == 1.0 and inv2 == inv) or ref is not None
    if ref is not None:
        assert math.isclose(inv2 * lev, inv * ref)                                       # the stamp = the paired leverage at the full margin
    return inv2, lev, scale


@pytest.mark.parametrize("cell_lev", [1.0, 2.0, 1.5, 0.5, 0.05])
def test_open_position_sizing_keeps_the_notional(monkeypatch, cell_lev):
    """0.2 × 0.25 opens the SAME notional as today's 1.0 × 0.05 for every cell — incl. the 1×-floor probes and rounding cells (0.5:
    round(0.5)=0 → 1× today, round(2.5)=2 at 0.25 → the margin is set from the notional, never the plain × 0.2)."""
    i0, l0, s0 = _sizes(monkeypatch, 1.0, 0.05, cell_lev)
    i1, l1, s1 = _sizes(monkeypatch, 0.2, 0.25, cell_lev)
    assert s0 == 1.0 and math.isclose(i0 * l0, i1 * l1, rel_tol=1e-12) and i1 <= i0
    if cell_lev == 1.0:
        assert l0 == 1 and l1 == 5 and math.isclose(i1, i0 * 0.2) and math.isclose(s1, 0.2)   # the headline case: 5× at 0.2 margin


def test_off_and_other_sleeves_untouched(monkeypatch):
    for es, d in (("MOMENTUM", "SHORT"), ("FRENZY_LONG", "LONG"), ("SURGE_LONG", "LONG"), ("BULLRUN_LONG", "LONG"), ("FLIP:X", "LONG")):
        i, l, s = _sizes(monkeypatch, 0.2, 0.25, es=es, d=d)
        j, m, t = _sizes(monkeypatch, 1.0, 0.25, es=es, d=d)
        assert (i, l, s) == (j, m, t) and s == 1.0, es
    i, l, s = _sizes(monkeypatch, 1.0, 0.25)                                            # switch off → plain 5×, full margin (no resize)
    assert s == 1.0 and l == 5


def test_schedule_cap_keeps_the_full_margin(monkeypatch):
    # a balance → leverage schedule capping at 1× → the position cannot be levered up → the full margin stays (never a smaller notional)
    i0, l0, _ = _sizes(monkeypatch, 1.0, 0.05, sched="0:1")
    i1, l1, s1 = _sizes(monkeypatch, 0.2, 0.25, sched="0:1")
    assert l1 == 1 and s1 == 1.0 and math.isclose(i0 * l0, i1 * l1)


def test_open_position_wiring_and_min_guards():
    i = ENG.index("    async def open_position("); j = ENG.index("\n    async def ", i + 10)
    body = ENG[i:j]
    c = body.index("investment, leverage, cell_capped = self.calculate_position_size(")
    r = body.index("investment, _ml_min_scale, _ml_ref_lev = self.momentum_long_resize(")
    assert "momentum_long_ref_leverage=_ml_ref_lev," in body                                           # D12: stamped on the order
    assert c < r < body.index("_desired_notional = investment * leverage") < body.index("notional_value = investment * leverage")
    assert "_new_investment < _inv_cfg.min_investment_size * _ml_min_scale" in body                     # liquidity / gross throttle
    assert "_new_inv < config.trading_config.investment.min_investment_size * _ml_min_scale" in body     # exchange brackets


def test_config_json_d11():
    import config as C
    assert C.SignalThresholds.model_fields["momentum_long_invest_mult"].default == 1.0
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["momentum_long_invest_mult"] == 0.2 and cfg["momentum_long_lev_mult"] == 0.25
    assert math.isclose(cfg["momentum_long_invest_mult"] * cfg["momentum_long_lev_mult"], 0.05)       # same notional as DL 289's 1×
    assert "DECISION_LOG 298" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()


def test_ui_load_save_preview_export_and_put():
    assert HTML.count('id="config-momentum-long-invest-mult"') == 1
    assert HTML.count("document.getElementById('config-momentum-long-invest-mult')") == 2             # load + save
    assert "_ssNum('config-momentum-long-invest-mult', 1)" in HTML and "S[k].baseLev = ml; S[k].baseInv = mi;" in HTML
    assert HTML.count("sp.baseInv)") >= 1 and "S.mom_long.baseInv)" in HTML and "S.unm.baseInv)" in HTML
    row = HTML[HTML.index('<tr data-ss="mom_long"'):HTML.index("</tr>", HTML.index('<tr data-ss="mom_long"'))]
    assert "ss-ro-inv" not in row and "config-momentum-long-invest-mult" in row
    b = HTML.index("function _buildConfigLines("); e = HTML.index("\n        function ", b + 10)
    assert "Momentum-long margin (Oct 15, DECISION_LOG 298)" in HTML[b:e] and "th.momentum_long_invest_mult" in HTML[b:e]
    assert HTML.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2
    assert "momentum_long_invest_mult must be from 0.05 to 1" in MAIN


def test_ref_leverage_column_and_migration():
    import models
    assert "momentum_long_ref_leverage" in [c.name for c in models.Order.__table__.columns]      # rides the orders CSV
    assert "ADD COLUMN momentum_long_ref_leverage FLOAT" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()


def test_bracket_keeps_the_reference_notional_with_the_full_margin():
    # pair max 3×: the resized 0.2 margin at 3× would give 0.6 of the notional; the full margin at 3× carries the whole reference
    tiers = [(1_000_000.0, 3.0)]
    full, lev = 1000.0, 5
    inv = full * 0.2                                                                   # the resized margin (scale 0.2), reference = 1,000
    lim = te.leverage_bracket_limit(tiers, lev, inv)
    le = float(int(lim[0])); ni = min(inv, lim[1] / le)
    assert math.isclose(ni * le, 600.0)                                                # today's capped result: 0.6 × the notional
    m, l = te.momentum_long_bracket_keep(tiers, lev, inv, 0.2, ni, le)
    assert l == 3.0 and math.isclose(m * l, 1000.0) and m <= full                       # the reference notional kept, margin ≤ the full margin
    # a cap that binds the notional too (tier cap $800): never above it, never above the reference
    m2, l2 = te.momentum_long_bracket_keep([(800.0, 3.0)], lev, inv, 0.2, ni, le)
    assert math.isclose(m2 * l2, 800.0)
    # not resized (scale 1) / no table → unchanged
    assert te.momentum_long_bracket_keep(tiers, lev, inv, 1.0, ni, le) == (ni, le) and te.momentum_long_bracket_keep(None, lev, inv, 0.2, ni, le) == (ni, le)
    body = ENG[ENG.index("    async def open_position("):]
    assert "_ml_b = momentum_long_bracket_keep(_ml_tiers, leverage, investment, _ml_min_scale, _new_inv, _lev_eff)" in body


def test_logs_quiet_reference_and_scaled_min():
    i = ENG.index("    async def open_position("); j = ENG.index("\n    async def ", i + 10)
    body = ENG[i:j]
    assert "min ${_inv_cfg.min_investment_size * _ml_min_scale * _ht_min_scale:.2f}" in body and "min ${config.trading_config.investment.min_investment_size * _ml_min_scale * _ht_min_scale:.2f}" in body   # 🎲¼ (341) × HT's own scale (1 for every other open)
    assert "momentum-long margin ×{_ml_min_scale:.3g} (paired {_ml_ref_lev:g}x" in body and "momentum-long margin ×{_ml_min_scale:.3g}, paired" in body
    assert "base_lev_multiplier=(ml_lev or 1.0) * f, quiet=True)" in ENG and "if not quiet:" in ENG


def test_quiet_flag_changes_nothing(monkeypatch):
    monkeypatch.setattr(te.config.trading_config.investment, "leverage_balance_schedule", "0:3", raising=False)
    eng = te.trading_engine
    a = eng.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0, base_lev_multiplier=0.25)
    b = eng.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0, base_lev_multiplier=0.25, quiet=True)
    assert a == b and a[1] == 3


def test_paired_knob_note_in_the_preview():
    assert "paired Lev × Inv = ${paired}" in HTML and "differs from 0.05 (DECISION_LOG 289" in HTML
    assert HTML.count("Lev mult × Inv mult") >= 2 and "back to 10×: Lev 0.5 AND Inv 1.0" in HTML
    assert "PAIRED KNOBS" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()


def test_research_scripts_apply_the_resize():
    for f in ("study_yr6_wideoff_ab.py", "study_yr6_lwoff_ab.py", "yr6_reprice_today.py"):
        assert "ENG.momentum_long_resize(" in open(os.path.join(ROOT, "scripts", f), encoding="utf-8").read(), f
    _b21 = os.path.join(ROOT, "scripts", "study_b21_review.py")   # local-only (untracked) research script: checked when present
    if os.path.exists(_b21):
        assert 'cget("momentum_long_invest_mult")' in open(_b21, encoding="utf-8").read()
