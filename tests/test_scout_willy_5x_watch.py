"""👁 Oct-31: WILLY at 5× (DECISION_LOG 425) — the setting everywhere, the gate's cohort, frozen bar, freeze once, hook and registry."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_5x_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
NOW, SINCE = int((T0 + pd.Timedelta(days=40)).value // 10**6), int(T0.value // 10**6)


def _fills(pcts, es="FRENZY_WILLY", start=0):
    rows = []
    for i, pct in enumerate(pcts):
        o = T0 + pd.Timedelta(days=start + i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"), pair=f"C{i}USDT", entry_strategy=es,
                         status="CLOSED", pnl_percentage=pct, close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS", leverage=5.0))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-20_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def test_the_setting_is_5x_everywhere():
    with open(os.path.join(ROOT, "trading_config.json")) as f:
        c = json.load(f)
    assert c.get("thresholds", c)["frenzy_willy_lev_mult"] == 0.4   # 📈 DL 533: live 8× (the 5× line is retired, DL 530); code default + UI default stay 0.25
    from config import SignalThresholds
    assert SignalThresholds.model_fields["frenzy_willy_lev_mult"].default == 0.25
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "['config-fz-willy-lev-mult', 'frenzy_willy_lev_mult', 0.25]" in html and "_ssNum('config-fz-willy-lev-mult', 0.25)" in html
    assert max(1, int(round(20 * 0.25))) == 5


def test_verdict_bar():
    def g(p):
        return pd.DataFrame(dict(pct=p))
    assert W.verdict(g([1.0] * 14))[0] == "COLLECTING"
    assert W.verdict(g([1.25] * 11 + [-4.0] * 4))[0] == W.REVIEW                # 73 % wins, avg < 0: the average decides first
    assert W.verdict(g([3.0] * 11 + [-4.0] * 4))[0] == W.RESTORE                # 73 % wins, avg > 0
    assert W.verdict(g([3.0] * 10 + [-4.0] * 5))[0] == W.STAY                   # 67 % wins, avg > 0
    assert W.verdict(g([4.0] * 10 + [-4.0] * 5))[0] == W.STAY
    ok = [3.0] * 69 + [-4.0] * 31
    assert W.verdict(g(ok))[0] == W.RESTORE                                     # exactly the breakeven passes


def test_cohort_floor_operator_closed_and_ht_apart(tmp_path):
    old = _fills([9.0], start=-5)
    op = _fills([9.0], start=1).assign(close_reason="MANUAL_CLOSE", pair="OPUSDT")
    ht = _fills([-4.0] * 20, es="FRENZY_WILLY_HT")
    live = _fills([1.25] * 3)
    out = "\n".join(W.run(NOW, _write(tmp_path, pd.concat([old, op, ht, live])), None, SINCE))
    assert "| FRENZY_WILLY | 3 | 100% | +1.250 %" in out and "⏳ COLLECTING (3/15 fills)" in out
    assert "FRENZY_WILLY_HT (information, never decides) | 20 |" in out


def test_freezes_once_and_defers_on_a_young_export(tmp_path):
    state = str(tmp_path / "s.json")
    f = _fills([3.0] * 11 + [-4.0] * 4)
    young = "\n".join(W.run(NOW, _write(tmp_path, f, stamp="2026-11-15_00-10-00"), state, SINCE))
    assert "freezing deferred" in young and not os.path.exists(state)
    out = "\n".join(W.run(NOW, _write(tmp_path, f), state, SINCE))
    assert f"**Frozen verdict" in out and W.RESTORE in out
    worse = _fills([3.0] * 11 + [-4.0] * 4 + [-4.0] * 20)
    again = "\n".join(W.run(NOW + 1, _write(tmp_path, worse, stamp="2026-12-21_00-00-00"), state, SINCE))
    assert W.RESTORE in again.split("Live (information")[0] and f"Live (information only, never re-decides): {W.REVIEW}" in again


def test_hook_registry_and_gitignore():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_5x_watch as" not in src and "scout_willy_5x_watch RETIRED 2026-11-11 (DECISION_LOG 530)" in src   # DL 530: unhooked
    reg = open(os.path.join(ROOT, "scripts", "observe_registry.py"), encoding="utf-8").read()
    assert '"scout_willy_5x_watch.py"' in reg and "scout_willy_5x_watch.py" in __import__("observe_registry").NONE   # DL 530: retired (size set by C5)
    assert "reports/SCOUT_WILLY_5X.json" in open(os.path.join(ROOT, ".gitignore")).read()
