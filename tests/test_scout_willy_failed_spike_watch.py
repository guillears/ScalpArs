"""👁 Nov-9: the WILLY_FAILED_SPIKE watch line + the WILLY OFF trigger (scripts/scout_willy_failed_spike_watch.py, DECISION_LOG 509) — frozen constants,
zone boundary, WILLY only (HT never pooled), unscored apart, the expectancy bar, the off-trigger rule, the hook, the registry and the zone.
Hermetic: temp exports and state, no network, no git."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_failed_spike_watch as V  # noqa: E402
import watch_zones as WZ  # noqa: E402

T0 = pd.Timestamp("2026-11-10 00:00")


def _fills(spec, es="FRENZY_WILLY"):
    """spec = [(day offset, pair, vs_vwap, adx, pct)]."""
    rows = []
    for i, (d, pair, vw, adx, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
                         pair=pair, direction="LONG", entry_strategy=es, status="CLOSED", pnl_percentage=pct, pnl=pct * 10,
                         close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS", entry_frenzy_vs_vwap_pct=vw, entry_adx=adx, entry_ema_gap_8_13=2.0))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-30_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def test_frozen_constants_and_gitignore():
    assert (V.DL, V.VWAP_CUT, V.N_MIN, V.DAYS_MIN, V.BE_WR, V.P_MIN, V.SHARE_MAX, V.MIN_VWAP_KEPT) == (509, 0.0, 15, 8, 0.690, 0.95, 0.50, 5)
    assert V.SLEEVES == ("FRENZY_WILLY", "FRENZY_WILLY_HT") and V.HT_DL == 511 and V.ZONE_COL == "entry_frenzy_vs_vwap_pct"
    assert "reports/SCOUT_WILLY_FAILED_SPIKE.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_zone_boundary_unscored_and_willy_only(tmp_path):
    df = pd.concat([_fills([(0, "AUSDT", 0.0, 30, 1.0), (0, "BUSDT", 0.01, 30, 1.0), (0, "CUSDT", None, 30, 1.0)]),
                    _fills([(0, "DUSDT", -1.0, 30, -4.0)], es="FRENZY_WILLY_HT")])
    c, _ = V.load(_write(tmp_path, df))
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "zone", "BUSDT": "rest", "CUSDT": "unscored", "DUSDT": "zone"}   # 0 itself = zone; HT in (DL 511)
    assert dict(zip(c.pair, c.gap_zone)) == {"AUSDT": "rest", "BUSDT": "rest", "CUSDT": "rest", "DUSDT": "rest"}
    out = "\n".join(V.run(now_ms=int((T0 + pd.Timedelta(days=30)).value // 10**6), paths=_write(tmp_path, df), state_path=None,
                         since_ms=int(T0.value // 10**6)))
    assert "### FRENZY_WILLY_HT" in out and out.count("| **zone** | 1 |") == 2          # each sleeve its own table, never pooled
    assert WZ.zones(pd.DataFrame(dict(entry_frenzy_vs_vwap_pct=[-1.0])))["FRENZY_WILLY_HT"]["WILLY_FAILED_SPIKE"].tolist() == [True]


def test_verdicts(tmp_path):
    zone = pd.DataFrame(dict(pct=[-4.0] * 10 + [1.0] * 5, day=[f"d{i % 8}" for i in range(15)], pair=[f"P{i}" for i in range(15)]))
    rest = pd.DataFrame(dict(pct=[1.2] * 10, day=["d0"] * 10, pair=["R"] * 10))
    assert V.verdict(zone, rest)[0] == V.CANDIDATE
    assert V.verdict(zone.assign(pct=zone.pct.abs()), rest)[0] == "RETIRED"
    assert V.verdict(zone.iloc[:10], rest)[0] == "COLLECTING"


def test_off_trigger_rule():
    bad = pd.DataFrame(dict(pct=[-4.0, 1.0, 1.0]))          # 67 % WR, avg −0.67
    good = pd.DataFrame(dict(pct=[1.0] * 5))
    assert V.trigger(bad, bad.iloc[0:0])[0] is True          # FAILED_SPIKE-kept < 5 → GAP-kept alone decides
    assert V.trigger(bad, pd.concat([bad, bad]))[0] is True  # both bad
    assert V.trigger(bad, good)[0] is False                  # FAILED_SPIKE-kept good → WILLY stays
    assert V.trigger(good, pd.concat([bad, bad]))[0] is False


def test_hook_registry_and_zone():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "import scout_willy_failed_spike_watch as _wvw" in src and "_wvw.run(now_ms)" in src
    import observe_registry as R
    assert "scout_willy_failed_spike_watch.py" in R.BOOKED and R.diffs(R.BOOKED["scout_willy_failed_spike_watch.py"], R.thresholds()) == [] and "scout_willy_gap_watch.py" in R.NONE
    d = pd.DataFrame(dict(entry_frenzy_vs_vwap_pct=[-0.1, 0.0, 0.2, None]))
    assert WZ.zones(d)["FRENZY_WILLY"]["WILLY_FAILED_SPIKE"].tolist() == [True, True, False, False]   # NaN → never in the zone
    sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
    import confidence as CF
    assert CF.LINES["WILLY_FAILED_SPIKE"][:3] == (509, 15, 8) and "WILLY_FAILED_SPIKE" in CF.THEORY


def test_off_trigger_reads_gap_state_without_side_effects_and_freezes_once(tmp_path, monkeypatch):
    import json
    import scout_willy_gap_watch as WG
    assert (WG.ADX_CUT, WG.GAP_CUT) == (55.0, 0.90)                                  # the trigger's GAP-kept uses WILLY_GAP's own cuts
    spec = [(i % 8, f"P{i}USDT", 0.5, 30, -4.0 if i % 2 else 1.0) for i in range(12)]   # GAP-kept and FS-kept: 50 % · −1.5
    paths = _write(tmp_path, _fills(spec))
    since = int(T0.value // 10**6)
    now = int((T0 + pd.Timedelta(days=30)).value // 10**6)
    own, gap = str(tmp_path / "own.json"), str(tmp_path / "gap.json")
    out = "\n".join(V.off_trigger(now, paths, gap_state=gap, state_path=own, gap_since=since, vw_since=since))
    assert "waiting for WILLY_GAP" in out and "would be MET" in out and not os.path.exists(own)       # GAP not frozen: nothing frozen
    open(gap, "w").write("{broken")
    out = "\n".join(V.off_trigger(now, paths, gap_state=gap, state_path=own, gap_since=since, vw_since=since))
    assert "unreadable" in out and open(gap).read() == "{broken"                                        # never quarantines GAP's file
    json.dump({"FRENZY_WILLY": {"state": "KEEP OBSERVING"}}, open(gap, "w"))
    out = "\n".join(V.off_trigger(now, paths, gap_state=gap, state_path=own, gap_since=since, vw_since=since))
    assert "MET — recommend WILLY OFF" in out and json.load(open(own))["OFF_TRIGGER"]["result"].startswith("MET")
    good = _write(tmp_path, _fills([(i % 8, f"P{i}USDT", 0.5, 30, 1.0) for i in range(12)]), stamp="2026-12-31_00-00-00")
    out = "\n".join(V.off_trigger(now, good, gap_state=gap, state_path=own, gap_since=since, vw_since=since))
    assert "Frozen" in out and "MET — recommend" in out and "Live (information only" in out           # decided once, never re-decided
    monkeypatch.setattr(V._GK, "dl_deploy_ms", lambda *a, **k: None)                                # no git: a DL commit not found
    out = "\n".join(V.off_trigger(now, paths, gap_state=gap, state_path=None))
    assert "NOT EVALUABLE" in out
    open(own, "w").write("{broken")                                                                  # own state corrupt → not frozen, said so
    out = "\n".join(V.off_trigger(now, paths, gap_state=gap, state_path=own, gap_since=since, vw_since=since))
    assert "own state corrupt" in out


def test_ht_counts_from_its_own_commit_and_best_case_freeze(tmp_path, monkeypatch):
    """DL 511: an HT fill between the 509 and 511 floors is NOT counted; WILLY's is. The best case freezes HT at 511 too."""
    f509, f511 = T0, T0 + pd.Timedelta(days=2)
    monkeypatch.setattr(V._GK, "dl_deploy_ms", lambda dl, **k: int((f509 if dl == 509 else f511 if dl == 511 else T0).value // 10**6))
    df = pd.concat([_fills([(1, "AUSDT", -1.0, 30, -4.0)]), _fills([(1, "BUSDT", -1.0, 30, -4.0)], es="FRENZY_WILLY_HT"),
                    _fills([(3, "CUSDT", -1.0, 30, -4.0)], es="FRENZY_WILLY_HT")])
    out = "\n".join(V.run(now_ms=int((T0 + pd.Timedelta(days=30)).value // 10**6), paths=_write(tmp_path, df), state_path=None))
    w, h = out.split("### FRENZY_WILLY_HT")
    assert "| **zone** | 1 |" in w.split("### FRENZY_WILLY")[1] and "| **zone** | 1 |" in h      # HT: day 1 out, day 3 in
    sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
    import confidence as CF
    assert CF.freeze_spec("WILLY_FAILED_SPIKE", "FRENZY_WILLY_HT") == 511 and CF.freeze_spec("WILLY_FAILED_SPIKE", "FRENZY_WILLY") == 509


def test_off_trigger_v2_both_filters_cohort_freezes_once(tmp_path):
    """(DL 531) WILLY OFF v2: the first 15 FRENZY_WILLY fills since the DL 527 floor, WR < 69 % ∧ avg < 0 → MET; decided once; the retired
    DL 509 record shows as history in run()."""
    import json
    since = int(T0.value // 10**6)
    now = int((T0 + pd.Timedelta(days=30)).value // 10**6)
    pre = _fills([(-1, "OLDUSDT", 0.5, 30, -4.0)])                                     # before the floor: never counted
    bad = _fills([(i % 8, f"P{i}USDT", 0.5, 30, -4.0 if i % 2 else 1.0) for i in range(16)])
    own = str(tmp_path / "own.json")
    c, newest = V.load(_write(tmp_path, pd.concat([pre, bad])))
    out = "\n".join(V.off_trigger_v2(now, state_path=own, frame=(c.iloc[:10], newest), since_ms=since))
    assert "COLLECTING" in out and not os.path.exists(own)
    out = "\n".join(V.off_trigger_v2(now, state_path=own, frame=(c, newest), since_ms=since))
    assert "MET — recommend WILLY OFF" in out and json.load(open(own))["OFF_TRIGGER_531"]["detail"].startswith("15 · ")
    good = c.assign(pct=1.0)
    out = "\n".join(V.off_trigger_v2(now, state_path=own, frame=(good, newest), since_ms=since))
    assert "Frozen" in out and "MET" in out                                             # decided once
    assert "NOT MET" in "\n".join(V.off_trigger_v2(now, state_path=None, frame=(good, newest), since_ms=since))
    json.dump({"OFF_TRIGGER": dict(result="MET — old", detail="d", at="x")}, open(own, "w"))
    o2 = "\n".join(V.run(now_ms=now, paths=_write(tmp_path, bad), state_path=own, since_ms=since))
    assert "RETIRED (DL 531)" in o2 and "MET — old" in o2 and "WILLY OFF trigger v2" in o2
