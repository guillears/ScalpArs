"""🎯 Nov-10 (DECISION_LOG 519): Bull-Run REARM-door fixed take-profit (`bullrun_rearm_tp_pct`, operator declared override).

Invariants: the TP fires only on door='REARM' at pnl ≥ the value (net %); GREEN / None / SURGE (trail override) / no-lock rows never;
0 = off; below the TP the exit is unchanged; D11 parity (config.py default = JSON = UI input + load / save + exported config line);
the revert-gate scout (walk, episodes, gate) and its exit-era registry entry.
"""
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd

from services.trading_engine import _bullrun_exit_for

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_bullrun_rearm_tp as S  # noqa: E402


def _th(tp=1.0):
    return SimpleNamespace(bullrun_base_sl_pct=-0.7, sl_atr_multiplier=1.5, sl_atr_widen_floor_pct=-1.2, bullrun_be_arm_pct=1.0,
                           bullrun_be_lock_pct=0.2, bullrun_trail_atr_mult=2.0, bullrun_rearm_trail_atr_mult=1.0, bullrun_ladder="",
                           bullrun_rearm_tp_pct=tp)


def _x(monkeypatch, th, *a, **kw):
    monkeypatch.setattr("services.trading_engine.config", SimpleNamespace(trading_config=SimpleNamespace(thresholds=th)), raising=False)
    return _bullrun_exit_for(*a, **kw)


def test_rearm_tp_fires_at_the_value(monkeypatch):
    assert _x(monkeypatch, _th(), 1.0, 1.0, 0.5, "REARM") == (True, "REARM_TP", 1.0)
    assert _x(monkeypatch, _th(), 1.3, 1.3, 0.5, "rearm")[1] == "REARM_TP"
    close, why, _ = _x(monkeypatch, _th(), 0.99, 0.99, 0.5, "REARM")
    assert not close and why is None                                      # below the TP: today's exit, nothing fires


def test_never_on_green_surge_manual_or_off(monkeypatch):
    for door in ("GREEN", None, ""):
        assert _x(monkeypatch, _th(), 1.5, 1.5, 0.5, door)[1] != "REARM_TP"
    assert _x(monkeypatch, _th(), 1.5, 1.5, 0.5, "REARM", trail_mult_override=1.0)[1] != "REARM_TP"   # SURGE_LONG
    assert _x(monkeypatch, _th(), 1.5, 1.5, 0.5, "REARM", no_lock=True)[1] != "REARM_TP"
    assert _x(monkeypatch, _th(0.0), 1.5, 1.5, 0.5, "REARM")[1] != "REARM_TP"                          # 0 = off
    assert _x(monkeypatch, SimpleNamespace(**{k: v for k, v in vars(_th()).items() if k != "bullrun_rearm_tp_pct"}),
              1.5, 1.5, 0.5, "REARM")[1] != "REARM_TP"                                                     # absent = off


def test_stop_side_unchanged(monkeypatch):
    assert _x(monkeypatch, _th(), -0.8, 0.3, 0.5, "REARM") == (True, "STOP_LOSS", -0.75)


def test_d11_parity():
    import config
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["bullrun_rearm_tp_pct"] == config.SignalThresholds().bullrun_rearm_tp_pct == 1.0
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-br-rearm-tp-pct"') == 1
    assert "['config-br-rearm-tp-pct', 'bullrun_rearm_tp_pct', 1.0]" in html                          # load
    assert "bullrun_rearm_tp_pct: safeFloat(document.getElementById('config-br-rearm-tp-pct')" in html  # save
    assert "REARM TP ${" in html                                                                         # exported config line


def test_scout_walk_episodes_gate():
    hi, lo = [0.5, 1.2, 1.4, 0.6], [0.0, 0.4, 1.0, 0.2]                  # gross: reaches +1.11 net, then fades
    assert S.walk(hi, lo, 0.5, tp=1.0) == 1.0
    tr = S.walk(hi, lo, 0.5)
    assert abs(tr - 0.81) < 1e-9                                         # armed at 1.11 net (bar 2) → trail 1.31 − 0.5 = 0.81, hit on bar 3 low
    cens = []
    S.walk([0.1, 0.2], [0.0, 0.0], 0.5, censored=cens)
    assert cens == [True]                                                # still open at the horizon → censored
    o = [pd.Timestamp("2026-12-01T10:00"), pd.Timestamp("2026-12-01T10:40"), pd.Timestamp("2026-12-02T09:00")]
    assert S.episodes(o, [5.0, 45.0, 3.0]) == [0, 0, 1]                  # same door start → one episode; next day → a new one
    assert S.gate([120.0])[0] == "COLLECTING"
    assert S.gate([120.0, -1.0])[0].startswith("REVERT")
    assert S.gate([120.0, 0.0, -500.0])[0].startswith("KEEP")            # frozen: only the first 2 count
    assert S.gate([None, -500.0, -500.0])[0] == "COLLECTING"             # an incomplete episode 1 blocks the queue (review 519)


def test_tp_fires_when_already_armed(monkeypatch):
    assert _x(monkeypatch, _th(), 1.2, 1.5, 0.5, "REARM") == (True, "REARM_TP", 1.0)


def test_scout_never_caches_short_or_early_paths(monkeypatch):
    cache, calls = {}, []

    class R:
        def __init__(self, n): self.n = n
        def json(self): return [[0, 0, 1.0, 1.0] for _ in range(self.n)]
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: calls.append(1) or R(60))
    t = "2026-12-01T10:00:00"
    open_ms = int(pd.Timestamp(t, tz="UTC").timestamp() * 1000)
    assert S._klines("XUSDT", t, cache, open_ms + 60 * 60_000) is None and not calls        # before the horizon: no fetch
    assert S._klines("XUSDT", t, cache, open_ms + (S.HORIZON + 1) * 60_000) is None          # short answer: not cached
    assert cache == {}


def test_scout_registered_and_wired():
    import observe_registry as OR
    assert OR.EXIT["scout_bullrun_rearm_tp.py"]["walks"]["bullrun_rearm_tp_pct"] == 1.0
    assert {r[0]: r[4] for r in OR.status()}["scout_bullrun_rearm_tp.py"] == "OK"
    assert "import scout_bullrun_rearm_tp as _brtp" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()


def test_scout_runs_without_fills(tmp_path):
    lines = S.run(now_ms=0, paths=[], since_ms=1, fetch=False)
    assert any("No REARM fill" in x for x in lines)
