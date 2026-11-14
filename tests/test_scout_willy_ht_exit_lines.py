"""🎯👁 DECISION_LOG 537 — WILLY HT take profit +2 % (shipped) revert gate + RUN15 watch line; engine / config wiring."""
import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import scout_willy_ht_exit_lines as W   # noqa: E402


def _days(n, per=2):
    return [f"2027-01-{1 + i // per:02d}" for i in range(n)]


def test_tp2_early_stop_and_final():
    D = pd.DataFrame({"TP2": [0.75] * 6 + [-5.25, -5.25] + [0.75] * 8, "RUN15": [np.nan] * 16})
    assert W.gate(D, _days(16))["TP2"][0] == "REVERT"                         # first 8: Σ −6.0 < 0 → early stop
    D2 = pd.DataFrame({"TP2": [0.75] * 7 + [-5.25] + [0.75] * 8, "RUN15": [np.nan] * 16})
    assert W.gate(D2.iloc[:8], _days(8))["TP2"][0] == "OBSERVING"            # first 8: Σ 0.0 → not < 0
    assert W.gate(D2, _days(16))["TP2"][0] == "KEEP"                          # at 15: Σ +5.25 > 0
    D3 = pd.DataFrame({"TP2": [0.75] * 7 + [-5.25] * 2 + [0.0] * 7, "RUN15": [np.nan] * 16})
    assert W.gate(D3, _days(16))["TP2"][0] == "REVERT"                        # early stop needs Σ < 0 at 8 (−4.5 at 9 is past it) → 15: Σ ≤ 0
    assert W.gate(D2, _days(16), held=True)["TP2"][0] == "OBSERVING"         # an earlier fill pending → no decision


def test_run15_pinned_verdicts_and_nan_never_counted():
    D = pd.DataFrame({"TP2": [0.75] * 16, "RUN15": [1.0] * 16})
    assert W.gate(D, _days(16))["RUN15"][0] == "OPERATOR CALL"
    D["RUN15"] = -0.5
    assert W.gate(D, _days(16))["RUN15"][0] == "NO CHANGE"
    D["RUN15"] = [np.nan] * 2 + [1.0] * 14                                   # NO_ATR fills: not counted for RUN15, TP2 unaffected
    g = W.gate(D, _days(16))
    assert g["RUN15"][0] == "OBSERVING" and "14 of 15" in g["RUN15"][1] and g["TP2"][0] == "KEEP"
    assert W.gate(pd.DataFrame({"TP2": [0.75] * 15, "RUN15": [1.0] * 15}), ["2027-01-01"] * 15)["RUN15"][0] == "OBSERVING"   # one day → waits


def test_frozen_state_overrides(tmp_path, monkeypatch):
    st = tmp_path / "s.json"
    json.dump({"TP2": {"verdict": "KEEP", "detail": "x", "at": "2027-01-01 00:00"}}, open(st, "w"))
    assert W._state(str(st))["TP2"]["verdict"] == "KEEP"
    W._save_state({"a": 1}, str(tmp_path / "t.json"))
    assert json.load(open(tmp_path / "t.json")) == {"a": 1}


def test_engine_config_ui_wiring():
    from services.frenzy import frenzy_willy_ht_tp
    assert frenzy_willy_ht_tp(types.SimpleNamespace(frenzy_willy_ht_tp_pct=2.0)) == 2.0
    for v in (0, None, -1, "x", float("nan"), 99):
        assert frenzy_willy_ht_tp(types.SimpleNamespace(frenzy_willy_ht_tp_pct=v)) is None
    import config
    assert config.SignalThresholds().frenzy_willy_ht_tp_pct == 0.0                               # code default OFF, the JSON arms 2.0
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_willy_ht_tp_pct"] == 2.0 and cfg["frenzy_willy_tp_pct"] == 1.25          # WILLY unchanged
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    i = eng.index("HT's OWN base take profit: T_used = max(HT TP")
    assert "_fz_es == FRENZY_WILLY_HT" in eng[i:i + 800] and "frenzy_willy_ht_target(_wv2_tp, _ht_tp)" in eng[i:i + 900]
    assert eng.index("frenzy_willy_tp_target_pct=_wv2_tp") > i                                 # stored on the row after the override
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert "['config-fz-willy-ht-tp', 'frenzy_willy_ht_tp_pct', 0]" in html and 'id="config-fz-willy-ht-tp"' in html


def test_hook_registry_ignore():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_willy_ht_exit_lines as _whx")
    assert "_whx.run(now_ms)" in src[j:j + 100] and "except Exception as _whx_e" in src[j:j + 200]
    import observe_registry as OR
    row = [r for r in OR.status() if r[0] == "scout_willy_ht_exit_lines.py"][0]
    assert row[4] == "OK", row[5]
    assert "reports/SCOUT_WILLY_HT_EXIT_LINES.json" in open(os.path.join(ROOT, ".gitignore")).read()


def test_loader_not_patched_by_import():
    import study_b19_fzexit_common as FX
    assert FX.WT.load_1m is W._TRUE_LOAD_1M


def test_not_evaluable_without_the_commit(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "Not evaluable yet" in "\n".join(W.run(fetch_ok=False, state_path=None))


def test_ht_target_rule_money_math():
    from services.frenzy import frenzy_willy_ht_target, frenzy_willy_ht_config_errors
    assert frenzy_willy_ht_target(1.25, 2.0) == 2.0 and frenzy_willy_ht_target(1.6, 2.0) == 2.0
    assert frenzy_willy_ht_target(3.88, 2.0) == 3.88 and frenzy_willy_ht_target(8.0, 2.0) == 8.0      # a spike target above is kept
    assert frenzy_willy_ht_target(1.25, None) == 1.25 and frenzy_willy_ht_target(None, 2.0) == 2.0
    th = lambda v: types.SimpleNamespace(frenzy_willy_ht_tp_pct=v, frenzy_willy_tp_pct=1.25, frenzy_willy_ht_enabled=True,   # noqa: E731
                                         frenzy_willy_ht_size_frac=1.0, frenzy_willy_ht_max_slots=1)
    assert not frenzy_willy_ht_config_errors(th(0)) and not frenzy_willy_ht_config_errors(th(2.0))
    for bad in (-1, 1.0, 1.25, 60, float("nan")):
        assert any("frenzy_willy_ht_tp_pct" in e for e in frenzy_willy_ht_config_errors(th(bad))), bad


def test_post_537_ht_fills_never_v2b_runners(monkeypatch):
    import scout_willy_exit_v2 as V
    monkeypatch.setattr(V, "_HT537", {"v": int(pd.Timestamp("2027-01-01").value // 1_000_000)})
    col = V.WEra.TP_COL
    assert V.is_runner({col: 2.0, "entry_strategy": "FRENZY_WILLY_HT", "opened_at": "2027-01-02T00:00:00"}) is False
    assert V.is_runner({col: 3.0, "entry_strategy": "FRENZY_WILLY_HT", "opened_at": "2026-12-30T00:00:00"}) is True     # pre-537 HT unchanged
    assert V.is_runner({col: 3.0, "entry_strategy": "FRENZY_WILLY", "opened_at": "2027-01-02T00:00:00"}) is True        # WILLY unchanged


# ── 🛑 DECISION_LOG 546 — HT wider-stop watch lines ──────────────────────────────────────────────────────────────────────────────────
def _F(stopped, n, saved=True):
    """n fills, the first `stopped` stopped live at −4; a wider stop saves them (+2) or not (−s)."""
    rows = []
    for i in range(n):
        ls = i < stopped
        rows.append(dict(live_stop=ls, NEW2=(-4.05 if ls else 2.0),
                         **{f"STOP{s}": ((2.0 if saved else -s - 0.05) if ls else (-s - 0.05 if i == n - 1 else 2.0)) for s in W.STOP_LINES}))
    return pd.DataFrame(rows)


def test_stop_lines_pinned_verdicts():
    g = W.gate_stop(_F(6, 20), _days(20, per=1))
    assert g["STOP6"][0] == "OPERATOR CALL" and g["STOP10"][0] == "OPERATOR CALL"
    assert g["STOP6"][2] == pytest.approx(6 * (2.0 + 4.05))      # Δ only on live-stopped fills (cohort = the first 6 fills)
    assert W.gate_stop(_F(6, 20, saved=False), _days(20, per=1))["STOP6"][0] == "NO CHANGE"
    assert W.gate_stop(_F(5, 20), _days(20, per=1))["STOP6"][0] == "OBSERVING"            # < 6 live-stopped
    assert W.gate_stop(_F(6, 20), ["2027-01-01"] * 3 + ["2027-01-02"] * 17)["STOP6"][0] == "OBSERVING"   # stops on < 4 days
    assert W.gate_stop(_F(6, 20), _days(20, per=1), held=True)["STOP6"][0] == "OBSERVING"
    # risk-equal: winners shrink to × 4/s — many winners + few small saves → NO CHANGE even with Σ Δ > 0
    F = _F(6, 200)
    F = pd.concat([F.iloc[6:], F.iloc[:6]], ignore_index=True)    # 194 winners first, the 6 stops last → the cohort = all 200
    g2 = W.gate_stop(F, _days(200, per=1))
    assert g2["STOP10"][3] < 0 and g2["STOP10"][0] == "NO CHANGE"
    # frozen prefix: fills after the 6th stop never join the decision (review 546)
    F3 = pd.concat([_F(6, 6), pd.DataFrame([dict(live_stop=True, NEW2=-4.05, **{f"STOP{s}": -s - 0.05 for s in W.STOP_LINES})] * 20)], ignore_index=True)
    assert W.gate_stop(F3, _days(26, per=1))["STOP6"][0] == "OPERATOR CALL"
    # a pending fill opened before the deciding fill holds the decision; one opened after does not
    F4 = _F(6, 6).assign(opened_at=[f"2027-01-0{i + 1}T00:00:00" for i in range(6)])
    assert W.gate_stop(F4, _days(6, per=1), pend=["2027-01-03T12:00:00"])["STOP6"][0] == "OBSERVING"
    assert W.gate_stop(F4, _days(6, per=1), pend=["2027-01-09T00:00:00"])["STOP6"][0] == "OPERATOR CALL"
    assert W.stop_cut(_F(5, 20), _days(20, per=1)) is None


def test_leg_stop_restores_constant():
    P = dict(te=0, t=np.array([60_000, 120_000]), hi=np.array([0.0, 0.0]), lo=np.array([-5.0, -5.0]), cl=np.array([-5.0, -5.0]),
             op=np.array([np.nan, np.nan]), bar=np.array([True, True]), end_ms=10 ** 9)
    old = W.S.STOP
    assert W._leg_stop(P, 2.0, 4)[1] == "STOP" and W._leg_stop(P, 2.0, 6)[1] != "STOP"
    assert W.S.STOP == old
    assert W.STOP_LINES == (6, 8, 10) and W.STOP_TODAY == 4.0 and (W.STOP_MIN_STOPPED, W.STOP_MIN_DAYS) == (6, 4) and W.DL_STOP == 546


def test_stop_section_not_evaluable_without_its_commit(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda n, **k: None if n == 546 else 0)
    L = "\n".join(W._stop_section(pd.DataFrame(columns=["key", "status", "opened_at"]), pd.DataFrame(columns=["k", "opened_at"]), {}, None))
    assert "Not evaluable yet" in L and "EVAA" in L
