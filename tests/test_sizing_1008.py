"""📏 Oct-8 sizing (DECISION_LOG 252): FAN flips 20× → 10× (flip_entry_sources FAN_RATIO_GATE:1.0:0.5) and BEARRUN_SHORT 1× → 5×
(bearrun_lev_mult 0.05 → 0.25, operator declared override).

Leverage trace (engine): a flip's cell leverage is RELATIVE to its registry lev — open_position does
    lev_mult = _flip_compose_lev(registry_lev, flip_cell_lev_mult)  = registry × (cell or 1)
then the hard cap (rsi_adx_multiplier_lev_hard_cap, 2.0), the 0.05 floor, and calculate_position_size:
    leverage = max(1, int(round(20 × lev_mult)))  → balance-schedule ceiling → (exchange bracket cap at order time).
The TG_SHALLOW / NEGDI15 cell blocks only raise the RELATIVE cell lev when explicitly configured > 1.0 (_flip_cell_lev_bump),
so a cell at 1.0 is "no change", never "reset to 20×". A cell explicitly at 2.0 doubles the SOURCE's leverage (FAN 10× → 20×) —
the same relative behaviour as before 252 (it then doubled 20× → 40×, clamped by the 2.0 hard cap)."""
import json
import os

import pytest

import config
import services.trading_engine as te

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = config.trading_config.thresholds


def _json_th():
    d = json.load(open(os.path.join(ROOT, "trading_config.json")))
    return d.get("thresholds", d)


def _lev(lev_mult):
    """leverage calculate_position_size gives a STRONG_BUY fill at this lev mult (sub-schedule-tier equity → 20× ceiling)."""
    inv, lev, _ = te.trading_engine.calculate_position_size(10_000.0, "STRONG_BUY", total_portfolio=10_000.0,
                                                             cell_multiplier=1.0, cell_lev_multiplier=lev_mult, multiplier_target="both")
    assert inv > 0
    return lev


def _flip_lev(source, cell_levs=()):
    """the engine's flip leverage path: _flip_filters / FAN qs-cell (1.0) → TG_SHALLOW / NEGDI15 bumps → registry × cell → cap / floor."""
    cell = 1.0                                       # _flip_filters / _fan_qs_cell_match return 1.0 at today's config
    for c in cell_levs:
        cell = te._flip_cell_lev_bump(cell, c)
    m = te._flip_compose_lev(te._flip_lev_mult(source), cell)
    m = max(0.05, min(m, float(getattr(TH, "rsi_adx_multiplier_lev_hard_cap", 2.0))))
    return _lev(m)


@pytest.fixture
def live_registry(monkeypatch):
    monkeypatch.setattr(TH, "flip_entry_enabled", True)
    monkeypatch.setattr(TH, "flip_entry_sources", _json_th()["flip_entry_sources"])
    monkeypatch.setattr(config.trading_config.investment, "leverage_balance_schedule", "0:20, 25000:15")


def test_json_values():
    th = _json_th()
    assert th["bearrun_lev_mult"] == 0.25
    d = config.SignalThresholds()                                                       # code defaults = the live values (D11)
    assert d.bearrun_lev_mult == 0.25


@pytest.mark.parametrize("spec", ["json", "default", "FAN_RATIO_GATE:1:0.5", "FAN_RATIO_GATE:1.0:0.50"])
def test_registry_value_parses_to_10x(monkeypatch, spec):
    """compared PARSED, not as a string — the UI re-emits "FAN_RATIO_GATE:1:0.5" on save (JS prints 1.0 as 1)."""
    s = {"json": _json_th()["flip_entry_sources"], "default": config.SignalThresholds().flip_entry_sources}.get(spec, spec)
    monkeypatch.setattr(TH, "flip_entry_enabled", True)
    monkeypatch.setattr(TH, "flip_entry_sources", s)
    assert te._flip_registry()["FAN_RATIO_GATE"] == (1.0, 0.5)


def test_registry_parse(live_registry):
    assert te._flip_registry() == {"FAN_RATIO_GATE": (1.0, 0.5)}
    assert te._flip_size_mult("FAN_RATIO_GATE") == 1.0 and te._flip_lev_mult("FAN_RATIO_GATE") == 0.5


def test_fan_flip_is_10x(live_registry):
    assert _flip_lev("FAN_RATIO_GATE") == 10


def test_fan_flip_stays_10x_when_cells_at_1_fire(live_registry):
    # TG_SHALLOW / NEGDI15 / the qs cell all at lev 1.0 today — firing them must not restore 20×
    assert float(_json_th()["flip_short_tg_shallow_lev_mult"]) == 1.0 and float(_json_th()["flip_short_negdi_lev_mult"]) == 1.0
    assert _flip_lev("FAN_RATIO_GATE", (1.0, 1.0)) == 10
    assert te._flip_cell_lev_bump(None, 1.0) is None and te._flip_cell_lev_bump(0.7, 1.0) == 0.7   # ≤ 1 = no change
    assert te._flip_cell_lev_bump(None, None) is None and te._flip_cell_lev_bump(None, "x") is None   # unreadable = no change
    assert te._flip_compose_lev(0.5, None) == 0.5 and te._flip_compose_lev(0.5, 0) == 0.5          # missing cell = ×1


def test_cell_at_2_doubles_the_source_lev(live_registry):
    # documented: relative to the source — FAN 10× → 20× (before 252: 20× → 40× requested, hard-capped at 2.0 = 40×)
    assert _flip_lev("FAN_RATIO_GATE", (2.0,)) == 20
    assert _flip_lev("FAN_RATIO_GATE", (1.0, 2.0)) == 20 and _flip_lev("FAN_RATIO_GATE", (2.0, 1.5)) == 20   # max of the bumps


def test_other_flip_sources_unchanged(monkeypatch):
    monkeypatch.setattr(TH, "flip_entry_enabled", True)
    monkeypatch.setattr(config.trading_config.investment, "leverage_balance_schedule", "0:20, 25000:15")
    monkeypatch.setattr(TH, "flip_entry_sources", "FAN_RATIO_GATE:1.0:0.5,PAIR_RSI_OB:1.0:0.05,LONG_UNMATCHED_ONLY:1.0")
    assert _flip_lev("PAIR_RSI_OB") == 1 and _flip_lev("LONG_UNMATCHED_ONLY") == 20 and _flip_lev("FAN_RATIO_GATE") == 10
    assert _flip_lev("LONG_UNMATCHED_ONLY", (1.0, 1.0)) == 20                           # a bare source at 1.0 is untouched by 1.0 cells
    monkeypatch.setattr(TH, "flip_entry_sources", "FAN_RATIO_GATE:1.0")                # the pre-252 registry still gives 20×
    assert _flip_lev("FAN_RATIO_GATE", (1.0,)) == 20


def test_balance_schedule_still_caps_after(monkeypatch):
    monkeypatch.setattr(config.trading_config.investment, "leverage_balance_schedule", "0:8")
    assert _lev(0.5) == 8                                                               # a ceiling below 10 still binds


def test_engine_uses_the_relative_helpers():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "max(_flip_cell_lev_mult or 1.0" not in src                                  # no inline replace/reset path left
    assert src.count("_flip_cell_lev_mult = _flip_cell_lev_bump(_flip_cell_lev_mult, ") == 2   # TG_SHALLOW + NEGDI15
    assert "cell_lev_mult = _flip_compose_lev(cell_lev_mult, flip_cell_lev_mult)" in src
    assert "cell_lev_mult = max(0.05, float(getattr(_th_bear, 'bearrun_lev_mult', 0.05) or 0.05))" in src


def test_bearrun_is_5x():
    lm = max(0.05, float(_json_th()["bearrun_lev_mult"] or 0.05))                       # the engine's BEARRUN branch
    assert lm == 0.25 and _lev(lm) == 5
    assert _lev(0.05) == 1                                                              # the rollback value is still the 1× probe


def test_ui_round_trips():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    # flip registry: load parses the 3rd field, sync re-emits SOURCE:size:lev when lev ≠ 1, the lev input allows 0.5
    assert "lev: parseFloat(bits[2]) || 1.0" in html and "`${s.key}:${szv}:${lvv}`" in html
    assert 'id="flip-lev-${s.key}" value="1.0" step="0.1" min="0.5"' in html
    # BEARRUN: load shows the JSON value (0.05 fallback, never 1.0), save keeps 0.25 and floors blank / 0 at 0.05
    assert "['config-be-lev-mult', 'bearrun_lev_mult', 0.05]" in html
    assert 'id="config-be-lev-mult" step="0.05" min="0.05" max="2" value="0.05"' in html          # static value never 1.0 (= 20×)
    assert "const x = v === '' ? 0.05 : safeFloat(v); return (x > 0) ? x : 0.05;" in html


def test_ui_flip_registry_never_saves_a_1x_fallback():
    """252 deep review: a blank / 0 / invalid size× or lev× box falls back to the server-loaded value; none → saveConfig refuses."""
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    i = html.index("function syncFlipRegistry()"); body = html[i:html.index("function loadFlipSources(", i)]
    assert "? lv : 1.0" not in body and "? sz : 1.0" not in body                      # the old 20× fallback is gone
    assert "(prev ? prev.lev : null)" in body and "(prev ? prev.size : null)" in body
    assert "if (bad.length) {" in body and "return; }   // keep the last valid string" in body
    assert "window._flipLoaded = map;" in html
    j = html.index("async function saveConfig() {")
    assert "if (window._flipRegistryError) { showToast(window._flipRegistryError, 'error'); return; }" in html[j:j + 400]


def test_ui_bearrun_labels_show_the_real_state():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "function bearLevLabel(lev)" in html and "ARM BAR (1x probe)" not in html
    assert html.count("Sizing now: ${bearLevLabel(") == 2                                # both text exports
    assert "' · sleeve ' + bearLevLabel(" in html and "sleeve ${_be.enabled ? bearLevLabel(" in html   # chip + tooltip


def test_master_builder_frozen_sizing():
    from scripts import build_master_pool as B
    th = _json_th()
    reg = dict((p.split(":")[0], p.split(":")) for p in th["flip_entry_sources"].split(","))
    assert B.FLIP_FAN_LEV_FROZEN == float(reg["FAN_RATIO_GATE"][2])
    assert B.BEARRUN_LEV_FROZEN == th["bearrun_lev_mult"]
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    assert "bearrun_lev_mult=0.25," in bld and 'STACK_VERSION = "2026-10-08b"' in bld
