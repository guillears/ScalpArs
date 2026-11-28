"""🔓 (DECISION_LOG 662) FRENZY long sleeves exempt from the funding guard (cost-only guard: one payment ≈ 0.1 % vs a ≈ +2 % trade) — config
default + JSON value + engine wiring + UI (D11) + export line."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _src(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def test_config_default_and_json():
    import config
    fld = [c for c in vars(config).values() if isinstance(c, type) and "funding_guard_exempt_frenzy_long" in getattr(c, "model_fields", {})]
    assert fld and fld[0].model_fields["funding_guard_exempt_frenzy_long"].default is False          # code default = the old behaviour
    tc = json.loads(_src("trading_config.json"))
    th = tc.get("thresholds", tc)
    assert th["funding_guard_exempt_frenzy_long"] is True and th["funding_guard_enabled"] is True


def test_engine_exempts_only_frenzy_longs():
    src = _src("services/trading_engine.py")
    assert ("_fg_exempt = bool(_frenzy and direction == \"LONG\" and getattr(_fg_th, 'funding_guard_exempt_frenzy_long', False) is True)"
            in src)
    assert "if bool(getattr(_fg_th, 'funding_guard_enabled', False)) and not _fg_exempt:" in src
    # _frenzy covers every FRENZY long sleeve (WILLY / WILLY_HT ride frenzy_willy) and nothing else
    assert '_frenzy = bool(frenzy_long or frenzy_wide or frenzy_lite or frenzy_willy) and direction == "LONG"' in src
    assert src.index("_fg_exempt = bool(") > src.index('_frenzy = bool(frenzy_long or frenzy_wide')


def test_ui_load_save_export():
    ui = _src("templates/index.html")
    assert ui.count("config-funding-guard-exempt-frenzy-long") == 3                                   # input + load + save
    assert "funding_guard_exempt_frenzy_long: !!document.getElementById('config-funding-guard-exempt-frenzy-long')?.checked" in ui
    assert "config.thresholds.funding_guard_exempt_frenzy_long === true" in ui
    assert "FRENZY longs exempt' : ''" in ui                                                         # the config export line


def _fund_run(monkeypatch, exempt, direction="LONG", **kw):
    import time
    import services.trading_engine as TE
    from tests import test_depth_guard_all_funding as D
    monkeypatch.setattr(TE.config.trading_config.thresholds, "funding_guard_exempt_frenzy_long", exempt, raising=False)
    soon = int(time.time() * 1000) + 90_000
    pay = 0.0018 if direction == "LONG" else -0.0018                                  # the payment is AGAINST the trade, +0.18 % in 1.5 min
    return D._run(monkeypatch, funding=(pay, soon), direction=direction, **kw)


def test_runtime_frenzy_long_exempt_others_guarded(monkeypatch):
    blocked_momentum = _fund_run(monkeypatch, True)                                  # a momentum long keeps the guard
    assert blocked_momentum[0] is None and "FUNDING_GUARD" in blocked_momentum[1]
    r_on = _fund_run(monkeypatch, True, frenzy_willy=True)                         # a WILLY long with the exemption ON → no FUNDING_GUARD
    assert "FUNDING_GUARD" not in r_on[1]
    r_off = _fund_run(monkeypatch, False, frenzy_willy=True)                       # the same with the switch OFF → refused by the guard
    assert r_off[0] is None and "FUNDING_GUARD" in r_off[1]
