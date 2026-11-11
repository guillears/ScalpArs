"""👁 Nov-11: the ML_BULL90 watch line (scripts/scout_ml_bull90_watch.py, DECISION_LOG 523) — frozen constants, cohort, the bar's outcomes, the export-age deferral,
freeze once, damaged state, the hook and the registry. Hermetic: temp exports and state, no network, no git."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_ml_bull90_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
Z, R = 95.0, 50.0                    # zone = bull % ≥ 90


def _fills(spec):
    rows = []
    for i, (d, pair, s, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair, direction="LONG",
                         entry_strategy="MOMENTUM", status="CLOSED", pnl_percentage=pct, pnl=pct * 10, close_reason="RUNNER_TRAIL" if pct > 0 else "STOP_LOSS L1",
                         cell_multiplier_source="UNMATCHED", entry_bull_pct=s))
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
    assert (W.DL, W.SLEEVE, W.BULL_CUT, W.N_MIN, W.DAYS_MIN, W.BE_WR, W.P_MIN, W.SHARE_MAX, W.WAIT_MIN) == (523, "MOMENTUM", 90.0, 15, 8, 0.611, 0.95, 0.50, 1200)
    assert "reports/SCOUT_ML_BULL90.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_zone_boundary_unscored_and_cohort(tmp_path):
    d = _fills([(0, "AUSDT", 90.0, 2.0), (0, "BUSDT", 89.99, 2.0), (0, "CUSDT", None, 2.0), (0, "PUSDT", Z, 2.0), (0, "LUSDT", Z, 2.0), (0, "SUSDT", Z, 2.0)])
    d.loc[3, "cell_multiplier_source"] = "X_PROBE"
    d.loc[4, "entry_strategy"] = "FRENZY_LITE"
    d.loc[5, "direction"] = "SHORT"
    c, _ = W.load(_write(tmp_path, d))
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "zone", "BUSDT": "rest", "CUSDT": "unscored"}


def test_verdict_outcomes(tmp_path):
    z, r = _cohort(tmp_path, _zr([-3.0] * 16 + [2.0] * 4))
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE and "WR 20% (breakeven 61.1%)" in det, det
    assert W.verdict(*_cohort(tmp_path, _zr([2.5] * 20)))[0] == "RETIRED"
    assert W.verdict(*_cohort(tmp_path, _zr([0.2] * 20, rest=1.0)))[0] == "RETIRED"
    assert W.verdict(*_cohort(tmp_path, _zr([-0.5] * 20, rest=-1.0)))[0] == "RETIRED"
    st, det = W.verdict(*_cohort(tmp_path, _zr([-3.0] * 7 + [1.5] * 13)))
    assert st == "KEEP OBSERVING" and "WR 65%" in det, det                # mean −0.075 < 0, but WR 65 % is not below the 61.1 % breakeven
    assert W.verdict(*_cohort(tmp_path, _zr([-3.0] * 20, days=1)))[0] == "COLLECTING"


def test_freeze_once_wait_and_damaged_state(tmp_path):
    df = _fills(_zr([-3.0] * 16 + [2.0] * 4))
    last = pd.to_datetime(df.opened_at).max()
    st = tmp_path / "w.json"
    now, since = int((T0 + pd.Timedelta(days=30)).value // 10**6), int(T0.value // 10**6)
    out = "\n".join(W.run(now, _write(tmp_path, df, (last + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d_%H-%M-%S")), str(st), since))
    assert "newest export < the crossing fill + 1200 min" in out and not st.exists()
    late = _write(tmp_path, df, (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    assert "Frozen verdict" in "\n".join(W.run(now, late, str(st), since)) and json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE
    flip = _write(tmp_path, _fills(_zr([2.5] * 20)), "2026-12-05_00-00-00")
    W.run(now, flip, str(st), since)
    assert json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE            # never re-decided
    for bad in ('{"MOMENTUM": "x"}', '{"MOMENTUM": {"state": "x"}}'):
        st.write_text(bad)
        o = "\n".join(W.run(now, late, str(st), since))
        assert "frozen state corrupt" in o and "Frozen verdict" not in o and st.read_text() == bad


def test_floor_operator_closed_and_empty(tmp_path):
    since = int(T0.value // 10**6)
    a = _fills([(0, "AUSDT", Z, -3.0), (0, "MANUSDT", Z, -3.0), (-3, "OLDUSDT", Z, -3.0)])
    a.loc[1, "close_reason"] = "MANUAL"
    out = "\n".join(W.run(since, _write(tmp_path, a), None, since))
    assert out.count("| **zone (bull % ≥ 90)** | 1 |") == 1
    assert "| **zone (bull % ≥ 90)** | 0 |" in "\n".join(W.run(since, [], None, since))


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_ml_bull90_watch as _mb90")
    assert "_mb90.run(now_ms)" in src[j:j + 200] and "except Exception as _mb90_e" in src[j:j + 400]
    import observe_registry as OR
    era = OR.BOOKED["scout_ml_bull90_watch.py"]
    assert era == OR.EXIT["scout_ml_fixed_tp_watch.py"]["walks"] and era["runner_trail_enabled"] is True
    assert {r[0]: r[4] for r in OR.status()}["scout_ml_bull90_watch.py"] == "OK"     # live momentum-long exit = the pinned keys


def test_n15_boundary_and_crossing_prefix(tmp_path):
    z, r = _cohort(tmp_path, _zr([-3.0] * 14))
    assert W.verdict(z, r)[0] == "COLLECTING"                              # 14 zone fills on 8 days: still collecting
    z, r = _cohort(tmp_path, _zr([-3.0] * 15))
    assert W.verdict(z, r)[0] != "COLLECTING"
    c, _ = W.load(_write(tmp_path, _fills(_zr([-3.0] * 25)), "2026-12-02_00-00-00"))
    sc = c[c.zone != "unscored"]
    pre = W.crossing_prefix(sc)
    pz = pre[pre.zone == "zone"]
    assert pre.iloc[-1].zone == "zone" and len(pz) >= 15 and pz.day.nunique() == 8          # ends AT the zone fill where BOTH legs first hold
    before = pre.iloc[:-1][pre.iloc[:-1].zone == "zone"]
    assert len(before) < 15 or before.day.nunique() < 8                                      # one fill earlier, a leg still fails
    one_day = sc[sc.zone == "zone"].iloc[:15].assign(day="2026-11-01")
    assert W.crossing_prefix(one_day) is None and W.crossing_prefix(sc[sc.zone == "zone"].sort_values("day").iloc[:14]) is None
