"""👁 Oct-30: the LONG_ADX35 watch line (scripts/scout_long_adx35_watch.py, DECISION_LOG 418) — frozen constants, cohort (FRENZY_LONG only,
forward, operator-closed out, unscored apart), the expectancy bar's outcomes, the export-age deferral, freeze once, the hook and the
registry. Hermetic: temp exports and state, no network, no git."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_long_adx35_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
NOW = int((T0 + pd.Timedelta(days=30)).value // 10**6)
SINCE = int(T0.value // 10**6)


def _fills(spec, es="FRENZY_LONG"):
    """spec = [(day offset, pair, adx, pct)] → CLOSED rows."""
    rows = []
    for i, (d, pair, adx, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
                         pair=pair, direction="LONG", entry_strategy=es, status="CLOSED", pnl_percentage=pct, pnl=pct,
                         close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS", entry_adx=adx))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _zone_rest(zone_pcts, rest_pct=1.5, days=8):
    return ([(i % days, f"Z{i}USDT", 45.0, p) for i, p in enumerate(zone_pcts)]
            + [(i % days, f"R{i}USDT", 20.0, rest_pct) for i in range(20)])


def _cohort(tmp_path, spec):
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    return c[c.zone == "zone"], c[c.zone == "rest"]


def test_frozen_constants():
    assert (W.DL, W.SLEEVE, W.ADX_CUT, W.N_MIN, W.DAYS_MIN, W.BE_WR, W.P_MIN, W.SHARE_MAX, W.CAP_DEFAULT) == (418, "FRENZY_LONG", 35.0, 15, 8, 0.727, 0.95, 0.50, 720)
    assert abs(4 / 5.5 - W.BE_WR) < 0.001
    assert W.cap_minutes(os.path.join(ROOT, "no_such_config.json")) == 720.0
    assert "reports/SCOUT_LONG_ADX35.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_cohort_boundary_unscored_and_other_sleeves(tmp_path):
    df = pd.concat([_fills([(0, "AUSDT", 35.0, 1.5), (0, "BUSDT", 35.01, -4.0), (0, "CUSDT", None, 1.5)]),
                    _fills([(0, "DUSDT", 70.0, -4.0)], es="FRENZY_WILLY"), _fills([(0, "EUSDT", 70.0, -3.0)], es="FRENZY_LITE")])
    c, _ = W.load(_write(tmp_path, df))
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "rest", "BUSDT": "zone", "CUSDT": "unscored"}    # 35 itself = rest
    out = "\n".join(W.run(NOW, _write(tmp_path, df), None, SINCE))
    assert "| **zone** | 1 |" in out and "| rest | 1 |" in out and "| UNSCORED (stamp missing) — never counted | 1 |" in out and "COLLECTING (zone N 1/15 · 1/8 days)" in out


def test_verdict_outcomes(tmp_path):
    z, r = _cohort(tmp_path, _zone_rest([-4.0] * 12 + [1.5] * 3))
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE and "WR 20% (breakeven 72.7%)" in det, det
    z, r = _cohort(tmp_path, _zone_rest([1.5] * 15))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone ≥ rest
    z, r = _cohort(tmp_path, _zone_rest([0.2] * 15))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone ≥ 0 never a filter
    z, r = _cohort(tmp_path, _zone_rest([-0.5] * 15, rest_pct=-1.0))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone < 0 but ≥ rest
    z, r = _cohort(tmp_path, _zone_rest([-4.0] * 4 + [1.0] * 11))
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "WR 73%" in det, det                 # mean −0.33 < 0, WR 73.3 % ≥ breakeven
    rows = [(i % 8, "SAMEUSDT", 45.0, -4.0) for i in range(10)] + [(i % 8, f"Z{i}USDT", 45.0, 1.5) for i in range(5)]
    rows += [(i % 8, f"R{i}USDT", 20.0, 1.5) for i in range(20)]
    z, r = _cohort(tmp_path, rows)
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "top pair 100%" in det, det          # one pair carries the loss
    z, r = _cohort(tmp_path, _zone_rest([-4.0] * 15, days=1))
    assert W.verdict(z, r)[0] == "COLLECTING"


def test_freeze_once_and_export_age_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cap_minutes", lambda *a, **k: 720.0)
    df = _fills(_zone_rest([-4.0] * 12 + [1.5] * 3))
    last = pd.to_datetime(df.opened_at).max()
    st = tmp_path / "w.json"
    early = _write(tmp_path, df, (last + pd.Timedelta(hours=11)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, early, str(st), SINCE))
    assert "newest export < the crossing fill + 720 min" in out and not st.exists()
    late = _write(tmp_path, df, (last + pd.Timedelta(hours=13)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, late, str(st), SINCE))
    assert "Frozen verdict" in out and json.load(open(st))["FRENZY_LONG"]["state"] == W.CANDIDATE
    flip = _fills(_zone_rest([1.5] * 15))
    out2 = "\n".join(W.run(NOW, _write(tmp_path, flip, "2026-12-05_00-00-00"), str(st), SINCE))
    assert W.CANDIDATE in out2.split("Live")[0] and json.load(open(st))["FRENZY_LONG"]["state"] == W.CANDIDATE   # never re-decided
    dry = "\n".join(W.run(NOW, late, None, SINCE))
    assert "Frozen verdict" not in dry and "Now:" in dry


def test_operator_closed_forward_floor_and_dedup(tmp_path):
    a = _fills([(0, "AUSDT", 60.0, -4.0), (0, "MANUSDT", 60.0, -4.0), (-3, "OLDUSDT", 60.0, -4.0)])
    a.loc[1, "close_reason"] = "MANUAL"
    old = a.copy(); old.loc[0, "pnl_percentage"] = -1.0
    p_old = _write(tmp_path, old, "2026-11-30_00-00-00"); p_new = _write(tmp_path, a, "2026-12-01_00-00-00")
    os.utime(p_old[0], (1, 1)); os.utime(p_new[0], (2, 2))
    out = "\n".join(W.run(NOW, p_old + p_new, None, SINCE))
    assert "MANUSDT" in out and "OLDUSDT" not in out and "| **zone** | 1 |" in out
    c, _ = W.load(p_old + p_new)
    assert c[c.pair == "AUSDT"].pct.tolist() == [-4.0]                       # newest export wins


def test_damaged_and_corrupt_state_never_refreeze(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cap_minutes", lambda *a, **k: 720.0)
    df = _fills(_zone_rest([-4.0] * 12 + [1.5] * 3))
    last = pd.to_datetime(df.opened_at).max()
    paths = _write(tmp_path, df, (last + pd.Timedelta(hours=13)).strftime("%Y-%m-%d_%H-%M-%S"))
    st = tmp_path / "w.json"
    bad = '{"FRENZY_LONG": {"state": "x"}}'
    st.write_text(bad)
    out = "\n".join(W.run(NOW, paths, str(st), SINCE))
    assert "frozen record damaged" in out and "Frozen verdict" not in out and st.read_text() == bad
    st.write_text("{not json")
    out = "\n".join(W.run(NOW, paths, str(st), SINCE))
    assert "frozen state corrupt" in out and "Frozen verdict" not in out and not st.exists()


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_long_adx35_watch as _lax")
    assert "_lax.run(now_ms)" in src[j:j + 200] and "except Exception as _lax_e" in src[j:j + 400]
    import observe_registry as OR
    era = OR.BOOKED["scout_long_adx35_watch.py"]
    assert era == {"frenzy_tp_pct": 1.5, "frenzy_long_stop_pct": 4.0, "frenzy_max_hold_minutes": 720, "frenzy_lock_arm_pct": 0.0}
    assert OR.diffs(era, OR.thresholds()) == []
