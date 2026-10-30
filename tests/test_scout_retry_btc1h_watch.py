"""👁 Oct-30: the RETRY_BTC1H watch line (scripts/scout_retry_btc1h_watch.py, DECISION_LOG 415) — frozen constants, cohort, the bar's outcomes, the export-age deferral,
freeze once, damaged state, the hook and the registry. Hermetic: temp exports and state, no network, no git."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_retry_btc1h_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
Z, R = -0.1, 0.1


def _fills(spec):
    rows = []
    for i, (d, pair, s, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair, direction="LONG",
                         entry_strategy="FRENZY_LITE_RETRY", status="CLOSED", pnl_percentage=pct, pnl=pct * 10, close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS",
                         cell_multiplier_source="FRENZY_LITE_RETRY", entry_btc_1h_slope=s))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _zr(zone, rest=2.0, days=8):
    return [(i % days, f"Z{i}USDT", Z, p) for i, p in enumerate(zone)] + [(i % days, f"R{i}USDT", R, rest) for i in range(20)]


def _cohort(tmp_path, spec):
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    return c[c.zone == "zone"], c[c.zone == "rest"]


def test_frozen_constants():
    assert (W.DL, W.SLEEVE, W.SLOPE_CUT, W.N_MIN, W.DAYS_MIN, W.BE_WR, W.P_MIN, W.SHARE_MAX, W.WAIT_MIN) == (415, "FRENZY_LITE_RETRY", 0.0, 15, 8, 0.600, 0.95, 0.50, 780)
    assert "reports/SCOUT_RETRY_BTC1H.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_zone_boundary_unscored_and_cohort(tmp_path):
    d = _fills([(0, "AUSDT", 0.0, 2.0), (0, "BUSDT", 0.01, 2.0), (0, "CUSDT", None, 2.0), (0, "PUSDT", Z, 2.0), (0, "LUSDT", Z, 2.0), (0, "SUSDT", Z, 2.0)])
    d.loc[3, "cell_multiplier_source"] = "X_PROBE"
    d.loc[4, "entry_strategy"] = "FRENZY_LITE"
    d.loc[5, "direction"] = "SHORT"
    c, _ = W.load(_write(tmp_path, d))
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "zone", "BUSDT": "rest", "CUSDT": "unscored"}


def test_verdict_outcomes(tmp_path):
    z, r = _cohort(tmp_path, _zr([-3.0] * 12 + [2.0] * 3))
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE and "WR 20% (breakeven 60.0%)" in det, det
    assert W.verdict(*_cohort(tmp_path, _zr([2.5] * 15)))[0] == "RETIRED"
    assert W.verdict(*_cohort(tmp_path, _zr([0.2] * 15, rest=1.0)))[0] == "RETIRED"
    assert W.verdict(*_cohort(tmp_path, _zr([-0.5] * 15, rest=-1.0)))[0] == "RETIRED"
    st, det = W.verdict(*_cohort(tmp_path, _zr([-3.0] * 6 + [1.5] * 9)))
    assert st == "KEEP OBSERVING" and "WR 60%" in det, det                # mean −0.3 < 0, but WR 60 % is not below the 60 % breakeven
    assert W.verdict(*_cohort(tmp_path, _zr([-3.0] * 15, days=1)))[0] == "COLLECTING"


def test_freeze_once_wait_and_damaged_state(tmp_path):
    df = _fills(_zr([-3.0] * 12 + [2.0] * 3))
    last = pd.to_datetime(df.opened_at).max()
    st = tmp_path / "w.json"
    now, since = int((T0 + pd.Timedelta(days=30)).value // 10**6), int(T0.value // 10**6)
    out = "\n".join(W.run(now, _write(tmp_path, df, (last + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d_%H-%M-%S")), str(st), since))
    assert "newest export < the crossing fill + 780 min" in out and not st.exists()
    late = _write(tmp_path, df, (last + pd.Timedelta(hours=14)).strftime("%Y-%m-%d_%H-%M-%S"))
    assert "Frozen verdict" in "\n".join(W.run(now, late, str(st), since)) and json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE
    flip = _write(tmp_path, _fills(_zr([2.5] * 15)), "2026-12-05_00-00-00")
    W.run(now, flip, str(st), since)
    assert json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE            # never re-decided
    for bad in ('{"FRENZY_LITE_RETRY": "x"}', '{"FRENZY_LITE_RETRY": {"state": "x"}}'):
        st.write_text(bad)
        o = "\n".join(W.run(now, late, str(st), since))
        assert "frozen state corrupt" in o and "Frozen verdict" not in o and st.read_text() == bad


def test_floor_operator_closed_and_empty(tmp_path):
    since = int(T0.value // 10**6)
    a = _fills([(0, "AUSDT", Z, -3.0), (0, "MANUSDT", Z, -3.0), (-3, "OLDUSDT", Z, -3.0)])
    a.loc[1, "close_reason"] = "MANUAL"
    out = "\n".join(W.run(since, _write(tmp_path, a), None, since))
    assert out.count("| **zone (BTC 1 h slope ≤ 0)** | 1 |") == 1
    assert "| **zone (BTC 1 h slope ≤ 0)** | 0 |" in "\n".join(W.run(since, [], None, since))


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_retry_btc1h_watch as _rbt")
    assert "_rbt.run(now_ms)" in src[j:j + 200] and "except Exception as _rbt_e" in src[j:j + 400]
    import observe_registry as OR
    era = OR.BOOKED["scout_retry_btc1h_watch.py"]
    assert era["frenzy_stop_pct"] == 3.0 and era["frenzy_lite_c_enabled"] is True
