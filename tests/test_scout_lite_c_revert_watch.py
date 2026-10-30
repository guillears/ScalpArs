"""👁 Oct-30: the LITE C revert (DECISION_LOG 420) — the tracker's shadow, cohort, frozen bar, freeze once, hook and registry, plus the switch, master and re-based lines. Hermetic."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_lite_c_revert_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
NOW, SINCE = int((T0 + pd.Timedelta(days=40)).value // 10**6), int(T0.value // 10**6)


def _fills(spec, es="FRENZY_LITE"):
    """spec = [(day, pair, zone, pct, peak)]"""
    rows = []
    for i, (d, pair, zone, pct, peak) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair, direction="LONG", entry_strategy=es,
                         status="CLOSED", pnl_percentage=pct, close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS", peak_pnl=peak, lite_c_zone=zone, frenzy_lite_tp_used=None))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-20_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def test_shadow_and_cohort(tmp_path):
    assert W.OLD_TP == 2.0 and W.shadow_pct(3.0, 3.1) == 2.0 and W.shadow_pct(-3.0, 2.2) == 2.0 and W.shadow_pct(-3.0, 1.9) == -3.0 and W.shadow_pct(1.0, 1.0) == 1.0 and W.shadow_pct(-3.0, 1.98) == 2.0 and W.shadow_pct(-3.0, 1.94) == -3.0
    d = _fills([(0, "AUSDT", False, 3.0, 3.1), (0, "BUSDT", False, -3.0, 2.5), (0, "CUSDT", False, -3.0, 0.5), (0, "DUSDT", True, 3.0, 3.1), (0, "EUSDT", False, 3.0, None), (0, "OLDUSDT", False, 2.0, 2.0)])
    d.loc[5, "frenzy_lite_tp_used"] = 2.0                               # an order from before the switch: out
    c, _ = W.load(_write(tmp_path, pd.concat([d, _fills([(0, "XUSDT", False, 3.0, 3.1)], es="FRENZY_WILLY")])))
    g = c.set_index("pair")
    assert list(g.index) == ["AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT"]
    assert g.delta["AUSDT"] == 1.0 and g.delta["BUSDT"] == -5.0 and g.delta["CUSDT"] == 0.0 and bool(g.zone["DUSDT"]) and not bool(g.scored["EUSDT"])


def test_zone_stamp_formats_and_missing(tmp_path):
    d = _fills([(0, "AUSDT", True, 3.0, 3.1), (0, "BUSDT", "true", 3.0, 3.1), (0, "CUSDT", 1, 3.0, 3.1), (0, "DUSDT", False, 3.0, 3.1), (0, "EUSDT", None, 3.0, 3.1)])
    g = W.load(_write(tmp_path, d))[0].set_index("pair")
    assert bool(g.zone["AUSDT"]) and bool(g.zone["BUSDT"]) and bool(g.zone["CUSDT"]) and not bool(g.zone["DUSDT"])
    assert bool(g.scored["DUSDT"]) and not bool(g.scored["EUSDT"])            # no stamp: unscored, never "outside"
    g = W.load(_write(tmp_path, d.drop(columns="lite_c_zone"), "2026-12-21_00-00-00"))[0]
    assert not g.scored.astype(bool).any()


def _o(tmp_path, spec):
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    return c


def test_verdict_outcomes(tmp_path):
    ok = [(i % 8, f"P{i}USDT", False, 3.0, 3.1) for i in range(20)]
    assert W.verdict(_o(tmp_path, ok))[0] == W.CONFIRMED
    assert W.verdict(_o(tmp_path, [(i % 8, f"P{i}USDT", False, -3.0, 2.5) for i in range(4)] + ok[4:]))[0] == W.REARM     # −20 + 16 ≤ 0
    assert W.verdict(_o(tmp_path, [(i % 8, "SAMEUSDT", False, 3.0, 3.1) for i in range(20)]))[0] == "KEEP OBSERVING"      # one pair carries the gain
    assert W.verdict(_o(tmp_path, ok[:19]))[0] == "COLLECTING"
    assert W.verdict(_o(tmp_path, [(0, f"P{i}USDT", False, 3.0, 3.1) for i in range(20)]))[0] == "COLLECTING"            # days


def test_sleeves_apart_freeze_once_and_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cap_minutes", lambda *a, **k: 720.0)
    df = _fills([(i % 8, f"P{i}USDT", False, 3.0, 3.1) for i in range(20)])
    last, st = pd.to_datetime(df.opened_at).max(), tmp_path / "w.json"
    early = _write(tmp_path, df, (last + pd.Timedelta(hours=11)).strftime("%Y-%m-%d_%H-%M-%S"))
    assert "newest export < the crossing fill + 720 min" in "\n".join(W.run(NOW, early, str(st), SINCE)) and not st.exists()
    late = _write(tmp_path, df, (last + pd.Timedelta(hours=13)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, late, str(st), SINCE))
    rec = json.load(open(st))
    assert "Frozen verdict" in out and rec["FRENZY_LITE"]["state"] == W.CONFIRMED and "FRENZY_LITE_RETRY" not in rec
    flip = _write(tmp_path, _fills([(i % 8, f"P{i}USDT", False, -3.0, 2.5) for i in range(20)]), "2026-12-25_00-00-00")
    assert json.load(open(st))["FRENZY_LITE"]["state"] == W.CONFIRMED and W.CONFIRMED in "\n".join(W.run(NOW, flip, str(st), SINCE)).split("Live")[0]
    st.write_text('{"FRENZY_LITE": {"state": "x"}}')
    out = "\n".join(W.run(NOW, late, str(st), SINCE))
    assert "frozen record damaged" in out and st.read_text() == '{"FRENZY_LITE": {"state": "x"}}'


def test_table_shows_before_and_after_with_compound(tmp_path):
    p = _write(tmp_path, _fills([(0, "AUSDT", False, 3.0, 3.1), (1, "BUSDT", False, -3.0, 2.5)]))
    out = "\n".join(W.run(int((T0 + pd.Timedelta(days=1)).value // 10**6), p, None, SINCE))
    assert "| booked (+3 %, today) | 2 | 50% | +0.000 % |" in out and "| if LITE C were still on (+2 %) | 2 | 100% | +2.000 % |" in out and "%/day" in out


def test_switch_master_lines_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["frenzy_lite_c_enabled"] is False and th["frenzy_lite_tp_pct"] == 3.0
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    j = src.index("import scout_lite_c_revert_watch as _lcr")
    assert "_lcr.run(now_ms)" in src[j:j + 200] and "except Exception as _lcr_e" in src[j:j + 400]
    assert "reports/SCOUT_LITE_C_REVERT.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")
    import observe_registry as OR
    st = {r[0]: r for r in OR.status(OR.thresholds())}
    assert st["scout_lite_c_revert_watch.py"][4] == "OK" and st["scout_retry_btc1h_watch.py"][4] == "OK" and st["scout_frenzy_v2b_shadow.py"][4] == "OK"
    assert st["scout_lite_c.py"][4] == "STALE" and st["scout_lite_green_retry.py"][4] == "STALE"     # resolved / measures the old rule: never read again
    import build_master_pool as B
    assert B.FRENZY_LITE_C_ENABLED_FROZEN is False
    assert B.frenzy_lite_c_tp_frozen(1.0, -1.0) == 3.0
    cf = pd.read_csv(os.path.join(ROOT, "reports", "LITE_C_REVERT_CF.csv"))
    assert len(cf) >= 11 and cf.cf_pct.between(-3.0, 3.0).all()
    import scout_frenzy_v2b_shadow as V
    assert V.today_exit(dict(entry_strategy="FRENZY_LITE", frenzy_lite_tp_used=2.0, entry_frenzy_vol_mult=1.0, entry_btc_1h_slope=-1.0)) == (3.0, 3.0)
