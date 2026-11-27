"""🌡⏱ DECISION_LOG 650 — EXH_BEAR80 + EXH_EARLY_SPIKE: EXHAUSTION_SHORT entry blocks (declared override) + their frozen shadow revert gates.

Pure rule (services/exhaustion.regime_block), the REAL engine judge path (tests/test_exhaustion_short._run), config / JSON / UI (D11), the
scout gate (pure), the opportunity-scout hook, the observe registry pin and the gitignore."""
import json
import os
import re
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services import exhaustion as X  # noqa: E402
from tests import test_exhaustion_short as TX  # noqa: E402
import scout_exh_regime_650 as S  # noqa: E402

H = 3_600_000


def _th(**kw):
    return TX._th(**dict(dict(exhaustion_bear80_block_enabled=True, exhaustion_early_spike_block_enabled=True), **kw))


# ── ① the pure rule ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_rule_cuts_are_strict_and_bear80_first():
    th = _th()
    assert X.regime_block(80.1, 2 * H, 0, "FRENZY", th)[0] == "EXH_BEAR80"
    assert X.regime_block(80.0, 2 * H, 0, "FRENZY", th)[0] is None                  # strict: exactly 80 % → trade
    assert X.regime_block(50, int(0.49 * H), 0, "FRENZY", th)[0] == "EXH_EARLY_SPIKE"
    assert X.regime_block(50, int(0.5 * H), 0, "FRENZY", th)[0] is None             # strict: exactly 0.5 h → trade
    assert X.regime_block(50, int(0.4951 * H), 0, "FRENZY", th)[0] is None          # hours rounded to 2 decimals like the stamp (0.50)
    assert X.regime_block(90, int(0.1 * H), 0, "FRENZY", th)[0] == "EXH_BEAR80"     # both zones → counted once, as BEAR80


def test_rule_wide_never_blocked_and_switches_and_unread():
    th = _th()
    assert X.regime_block(99, 0, 0, "WIDE", th) == (None, False, None)
    assert X.regime_block(99, 0, 0, None, th)[0] == "EXH_BEAR80"                    # unknown universe = FRENZY
    off = _th(exhaustion_bear80_block_enabled=False)
    assert X.regime_block(99, 2 * H, 0, "FRENZY", off)[0] is None
    assert X.regime_block(99, 0, 0, "FRENZY", off)[0] == "EXH_EARLY_SPIKE"
    off2 = _th(exhaustion_early_spike_block_enabled=False)
    assert X.regime_block(10, 0, 0, "FRENZY", off2)[0] is None
    code, unread, _ = X.regime_block(None, 2 * H, 0, "FRENZY", th)
    assert code is None and unread is True                                           # breadth unread → FAIL-OPEN, flagged
    assert X.regime_block(None, 0, 0, "FRENZY", th)[:2] == ("EXH_EARLY_SPIKE", True)
    assert X.regime_block("x", "y", 0, "FRENZY", th)[0] is None                       # never raises
    assert X.bear80_enabled(NS()) is False and X.early_spike_enabled(NS()) is False   # code default OFF
    assert X.bear80_enabled(_th(exhaustion_bear80_block_enabled="yes")) is False


def test_config_cuts_used_and_bad_values_fall_back():
    th = _th(exhaustion_bear_pct_max=60.0, exhaustion_min_hours_after_spike=2.0)
    assert X.regime_block(61, 3 * H, 0, "FRENZY", th)[0] == "EXH_BEAR80" and X.regime_block(59, int(1.5 * H), 0, "FRENZY", th)[0] == "EXH_EARLY_SPIKE"
    bad = _th(exhaustion_bear_pct_max=5.0, exhaustion_min_hours_after_spike=-1)
    p = X.exhaustion_params(bad)
    assert p["bear_max"] == 80.0 and p["min_h_spike"] == 0.5
    errs = X.exhaustion_config_errors(bad)
    assert any("exhaustion_bear_pct_max" in e for e in errs) and any("exhaustion_min_hours_after_spike" in e for e in errs)
    assert any("exhaustion_bear80_block_enabled" in e for e in X.exhaustion_config_errors(_th(exhaustion_bear80_block_enabled=1)))


# ── ② the REAL engine judge path ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _go(bear, th_kw=None, age_s=60.0):
    """bear = the bearish % of the scan's published breadth snapshot (None → no snapshot yet, as after a restart); age_s = its age."""
    import time
    import services.trading_engine as TE
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(TE, "_market_bear_pct", 0.0, raising=False)          # the global's default — never read by the block (review)
        mp.setattr(TE, "_market_stamps_snap", None if bear is None else (time.time() - age_s, 100 - bear, bear, 1.0), raising=False)

        def hook(e):
            async def sh(db):
                return 5000.0
            e._exh_shadow_notional = sh
        return TX._run(mp, k=1, th_kw=dict(dict(exhaustion_bear80_block_enabled=True, exhaustion_early_spike_block_enabled=True),
                                            **(th_kw or {})), oi_rows=[], engine_hook=hook)
    finally:
        mp.undo()


def test_engine_bear80_blocks_journals_and_never_orders():
    e, opens, sig, _, _ = _go(95.0)
    assert opens == [] and "EXH_BEAR80" in e.blocks
    s = sig[0]
    assert s.status == "BLOCKED" and s.block == "EXH_BEAR80" and "bearish" in s.reason
    assert s.decision_price and s.tp_pct and s.sl_pct and s.shadow_notional_usd == 5000.0      # what the revert scout prices with
    assert s.model_slip_pct == pytest.approx(X.model_slip_pct(5000.0, s.q5_usd))


def test_engine_early_spike_blocks():
    e, opens, sig, _, _ = _go(10.0, th_kw=dict(exhaustion_min_hours_after_spike=24.0))         # any fixture hour < 24 → the early zone
    assert opens == [] and sig[0].block == "EXH_EARLY_SPIKE" and "after the spike" in sig[0].reason


def test_engine_passes_outside_both_zones_and_when_off():
    e, opens, sig, _, _ = _go(10.0, th_kw=dict(exhaustion_min_hours_after_spike=0.0))
    assert len(opens) == 1 and sig[0].status == "OPENED" and not [b for b in e.blocks if b.startswith(("EXH_BEAR80", "EXH_EARLY"))]
    e, opens, sig, _, _ = _go(95.0, th_kw=dict(exhaustion_bear80_block_enabled=False, exhaustion_early_spike_block_enabled=False))
    assert len(opens) == 1 and sig[0].status == "OPENED"


def test_engine_unread_breadth_fails_open_and_counts():
    e, opens, sig, _, _ = _go(None, th_kw=dict(exhaustion_min_hours_after_spike=0.0))         # no snapshot yet (restart)
    assert len(opens) == 1 and "EXH_BEAR80_UNREAD" in e.blocks


def test_engine_stale_breadth_is_unread_not_a_block():
    e, opens, sig, _, _ = _go(95.0, th_kw=dict(exhaustion_min_hours_after_spike=0.0), age_s=3600)   # 95 % but 1 h old → unread
    assert len(opens) == 1 and "EXH_BEAR80_UNREAD" in e.blocks and "EXH_BEAR80" not in e.blocks
    assert opens[0]["entry_bear_pct"] is None                       # stale → NULL stamp (the DL 593 backfill fills it), never a fake 0 %


def test_engine_fresh_breadth_is_stamped():
    e, opens, sig, _, _ = _go(30.0, th_kw=dict(exhaustion_min_hours_after_spike=0.0))
    assert len(opens) == 1 and opens[0]["entry_bear_pct"] == 30.0


# ── ③ config / JSON / UI (D11) ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_json_ui_wiring():
    import config
    d = config.ThresholdsConfig() if hasattr(config, "ThresholdsConfig") else None
    j = json.load(open(os.path.join(ROOT, "trading_config.json")))
    t = j.get("thresholds", j)
    assert t["exhaustion_bear80_block_enabled"] is True and t["exhaustion_early_spike_block_enabled"] is True
    assert t["exhaustion_bear_pct_max"] == 80.0 and t["exhaustion_min_hours_after_spike"] == 0.5
    src = open(os.path.join(ROOT, "config.py")).read()
    assert "exhaustion_bear80_block_enabled: bool = False" in src and "exhaustion_early_spike_block_enabled: bool = False" in src
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    for i in ("config-exh-bear80-enabled", "config-exh-early-spike-enabled"):
        assert html.count(f"'{i}'") == 2 and f'id="{i}"' in html                # load + save + the input
    for i in ("config-exh-bear-pct-max", "config-exh-min-hours-spike"):
        assert f'id="{i}"' in html and f"['{i}'," in html                          # input + the shared load/save list
    assert "exhaustion_bear80_enabled" in open(os.path.join(ROOT, "main.py")).read()


# ── ④ the scout gate (pure) ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def _rows(vals, days):
    return [(f"2026-12-{1 + i % days:02d}", v, 1_800_000_000_000 + i * H) for i, v in enumerate(vals)]


def test_report_freezes_only_after_settling_and_reports_failed_save(tmp_path, monkeypatch):
    T0 = 1_800_000_000_000
    j = pd.DataFrame([dict(exh_block="EXH_BEAR80", exh_signal_at=str(pd.Timestamp(T0 + i * 86_400_000 // 2, unit="ms")), pair="XUSDT",
                           exh_tp_pct=7.0, exh_sl_pct=6.0, exh_decision_price=1.0, exh_model_slip_pct=0.0) for i in range(10)])
    monkeypatch.setattr(S.SO, "price_signal", lambda *a, **k: (-1.0, "SL"))
    monkeypatch.setattr(S.SO, "CACHE", str(tmp_path / "c.json"))
    last = T0 + 9 * 86_400_000 // 2
    st = tmp_path / "s.json"
    out = S.report(j, None, now_ms=last + 3_600_000, state_path=str(st), floor_ms=0)
    assert "settling" in out and not st.exists()                                   # window not settled → not frozen
    out = S.report(j, None, now_ms=last + S.SETTLE_MS + 1, state_path=str(st), floor_ms=0)
    assert json.loads(st.read_text())["EXH_BEAR80"]["verdict"] == "KEEP" and "FROZEN" in out
    bad = tmp_path / "nodir" / "s.json"
    out = S.report(j, None, now_ms=last + S.SETTLE_MS + 1, state_path=str(bad), floor_ms=0)
    assert "state not saved" in out


def test_gate_collects_then_freezes_on_the_first_qualifying_prefix():
    ref = lambda a, b: (0.5, 9, "kept")   # noqa: E731
    assert S.gate(_rows([1.0] * 9, 9), ref)[0] == "COLLECTING"
    assert S.gate(_rows([1.0] * 12, 4), ref)[0] == "COLLECTING"                     # < 5 days
    v, txt, n = S.gate(_rows([1.0] * 10, 5), ref)
    assert (v, n) == ("REVERT", 10) and "REF +0.500" in txt                         # blocked +1.0 ≥ kept +0.5 → REVERT
    v, _, n = S.gate(_rows([-1.0] * 10 + [9.0] * 5, 5), ref)
    assert (v, n) == ("KEEP", 10)                                                   # decided on the prefix, later rows ignored


def test_kept_ref_window_and_fallback():
    T0 = 1_800_000_000_000
    f = pd.DataFrame(dict(opened_at=[str(pd.Timestamp(T0 + i * H, unit="ms")) for i in range(6)], pnl_percentage=[1, 2, 3, 4, 5, 6],
                          entry_exh_universe=["FRENZY"] * 5 + ["WIDE"], close_reason=["FRENZY_TP"] * 6))
    assert S.kept_ref(f, T0, T0 + 10 * H)[:2] == (3.0, 5)                             # WIDE out (no slip stamp / notional → net = raw)
    f2 = f.assign(close_reason="STOP_LOSS", entry_exh_model_slip_pct=0.1)
    assert S.kept_ref(f2, T0, T0 + 10 * H)[0] == pytest.approx(2.8)                    # re-priced: − slip on entry and on the taker exit
    assert S.kept_ref(f.iloc[:3], T0, T0 + 10 * H)[0] == S.REF_FALLBACK
    assert S.kept_ref(None, T0, T0)[0] == S.REF_FALLBACK


def test_scout_not_evaluable_before_commit_and_hooks():
    j = pd.DataFrame([dict(exh_block="EXH_BEAR80", exh_signal_at="2026-12-01 00:00:00", pair="XUSDT")])
    assert "NOT EVALUABLE" in S.report(j, None, floor_ms=None, state_path=None)
    assert S.cohort(j, 0, "EXH_BEAR80") and not S.cohort(j, 0, "EXH_EARLY_SPIKE")
    j2 = j.assign(exh_universe="WIDE")
    assert S.cohort(j2, 0, "EXH_BEAR80") == []
    assert "scout_exh_regime_650" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    import observe_registry as OR
    assert "scout_exh_regime_650.py" in open(OR.__file__).read()
    assert re.search(r"^reports/SCOUT_EXH_REGIME_650\.json$", open(os.path.join(ROOT, ".gitignore")).read(), re.M)
