"""👁 Oct-30: the FADE_NOSTRETCH watch line (scripts/scout_fade_nostretch_watch.py, DECISION_LOG 413) — frozen constants, cohort (forward, full size,
operator-closed out, unscored apart), the expectancy bar's outcomes, the export-age deferral, freeze once, the hook and the registry.
Hermetic: temp exports and state, no network, no git."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_fade_nostretch_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
Z, R = (40.0, 40.0, 1.0), (60.0, 60.0, 3.0)


def _fills(spec):
    """spec = [(day offset, pair, (btc rsi 1h, +di, stretch), pct)] → CLOSED fades."""
    rows = []
    for i, (d, pair, s, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair,
                         direction="SHORT", entry_strategy="SPIKE_FADE", status="CLOSED", pnl_percentage=pct, pnl=pct * 10,
                         close_reason="RUNNER_TRAIL" if pct > 0 else "STOP_LOSS L1", cell_multiplier_source="SPIKE_FADE",
                         entry_btc_rsi_1h=s[0], entry_pos_di=s[1], entry_dist_from_ema13_pct=s[2]))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _zone_rest(zone_pcts, rest_pct=0.5, days=8):
    return ([(i % days, f"Z{i}USDT", Z, p) for i, p in enumerate(zone_pcts)] + [(i % days, f"R{i}USDT", R, rest_pct) for i in range(20)])


def _cohort(tmp_path, spec):
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    return c[c.zone == "zone"], c[c.zone == "rest"]


def test_frozen_constants():
    assert (W.DL, W.BTC_RSI_1H_CUT, W.POS_DI_CUT, W.STRETCH_CUT, W.FILTER_DEF) == (413, 50.0, 45.0, 2.0, (47.15, 45.91, 1.65))
    assert (W.N_MIN, W.DAYS_MIN, W.BE_WR, W.P_MIN, W.SHARE_MAX, W.WAIT_MIN) == (15, 8, 0.721, 0.95, 0.50, 1200)
    assert "reports/SCOUT_FADE_NOSTRETCH.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_zone_needs_all_three_boundaries_and_unscored(tmp_path):
    spec = [(0, "AUSDT", (50.0, 45.0, 2.0), 0.3), (0, "BUSDT", (50.01, 40.0, 1.0), 0.3), (0, "CUSDT", (40.0, 45.01, 1.0), 0.3),
            (0, "DUSDT", (40.0, 40.0, 2.01), 0.3), (0, "EUSDT", (40.0, None, 1.0), 0.3)]
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "zone", "BUSDT": "rest", "CUSDT": "rest", "DUSDT": "rest", "EUSDT": "unscored"}


def test_filter_definition_tracked_apart_and_stored(tmp_path):
    spec = [(0, "AUSDT", (47.15, 45.91, 1.65), 0.3), (0, "BUSDT", (47.16, 40.0, 1.0), 0.3), (0, "CUSDT", (40.0, 45.92, 1.0), 0.3),
            (0, "DUSDT", (40.0, 40.0, 1.66), 0.3), (0, "EUSDT", (40.0, None, 1.0), 0.3)]
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    assert dict(zip(c.pair, c.fdef)) == {"AUSDT": True, "BUSDT": False, "CUSDT": False, "DUSDT": False, "EUSDT": False}
    assert dict(zip(c.pair, c.zone))["AUSDT"] == "rest"                      # +DI 45.91 > 45: in the definition, outside the deciding zone
    df = _fills(_zone_rest([-1.5] * 12 + [0.3] * 3))
    last = pd.to_datetime(df.opened_at).max()
    st = tmp_path / "w.json"
    out = "\n".join(W.run(int((T0 + pd.Timedelta(days=30)).value // 10**6),
                          _write(tmp_path, df, (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S")), str(st), int(T0.value // 10**6)))
    rec = json.load(open(st))[W.SLEEVE]
    assert "): N 15 ·" in rec["filter_definition"] and "Filter definition at the crossing" in out


def test_cohort_excludes_probes_other_sleeves_longs(tmp_path):
    d = _fills([(0, "AUSDT", Z, 0.3), (0, "PUSDT", Z, 0.3), (0, "OUSDT", Z, 0.3), (0, "LUSDT", Z, 0.3)])
    d.loc[1, "cell_multiplier_source"] = "SPIKE_FADE_PROBE"
    d.loc[2, "entry_strategy"] = "MOMENTUM"
    d.loc[3, "direction"] = "LONG"
    c, _ = W.load(_write(tmp_path, d))
    assert c.pair.tolist() == ["AUSDT"]


def test_verdict_outcomes(tmp_path):
    z, r = _cohort(tmp_path, _zone_rest([-1.5] * 12 + [0.3] * 3))
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE and "WR 20% (breakeven 72.1%)" in det, det
    z, r = _cohort(tmp_path, _zone_rest([0.8] * 15))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone ≥ rest
    z, r = _cohort(tmp_path, _zone_rest([0.2] * 15))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone ≥ 0 never a filter
    z, r = _cohort(tmp_path, _zone_rest([-0.5] * 15, rest_pct=-1.0))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone < 0 but ≥ rest
    z, r = _cohort(tmp_path, _zone_rest([-1.5] * 4 + [0.3] * 11))
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "WR 73%" in det, det                 # mean −0.16 < 0, but WR 73 % ≥ the 72.1 % breakeven
    rows = [(i % 8, "SAMEUSDT", Z, -1.5) for i in range(10)] + [(i % 8, f"Z{i}USDT", Z, 0.3) for i in range(5)]
    rows += [(i % 8, f"R{i}USDT", R, 0.5) for i in range(20)]
    z, r = _cohort(tmp_path, rows)
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "top pair 100%" in det, det          # one pair carries the loss
    z, r = _cohort(tmp_path, _zone_rest([-1.5] * 15, days=1))
    assert W.verdict(z, r)[0] == "COLLECTING"


def test_freeze_once_and_export_age_wait(tmp_path):
    df = _fills(_zone_rest([-1.5] * 12 + [0.3] * 3))
    last = pd.to_datetime(df.opened_at).max()
    st = tmp_path / "w.json"
    now, since = int((T0 + pd.Timedelta(days=30)).value // 10**6), int(T0.value // 10**6)
    early = _write(tmp_path, df, (last + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(now, early, str(st), since))
    assert "newest export < the crossing fill + 1200 min" in out and not st.exists()
    late = _write(tmp_path, df, (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(now, late, str(st), since))
    assert "Frozen verdict" in out and json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE
    flip = _fills(_zone_rest([0.8] * 15))
    out2 = "\n".join(W.run(now, _write(tmp_path, flip, "2026-12-05_00-00-00"), str(st), since))
    assert W.CANDIDATE in out2.split("Live")[0] and json.load(open(st))[W.SLEEVE]["state"] == W.CANDIDATE   # never re-decided
    dry = "\n".join(W.run(now, late, None, since))
    assert "Frozen verdict" not in dry and "Now:" in dry


def test_forward_floor_operator_closed_dedup(tmp_path):
    since = int(T0.value // 10**6)
    a = _fills([(0, "AUSDT", Z, -1.5), (0, "MANUSDT", Z, -1.5), (-3, "OLDUSDT", Z, -1.5)])
    a.loc[1, "close_reason"] = "MANUAL"
    old = a.copy()
    old.loc[0, "pnl_percentage"] = -1.0
    p_old = _write(tmp_path, old, "2026-11-30_00-00-00")
    p_new = _write(tmp_path, a, "2026-12-01_00-00-00")
    os.utime(p_old[0], (1, 1))
    os.utime(p_new[0], (2, 2))
    out = "\n".join(W.run(int((T0 + pd.Timedelta(days=30)).value // 10**6), p_old + p_new, None, since))
    assert "OLDUSDT" not in out and out.count("| **zone (round numbers)** | 1 |") == 1     # the pre-floor and the operator-closed fills not counted
    c, _ = W.load(p_old + p_new)
    assert c[c.pair == "AUSDT"].pct.tolist() == [-1.5]                       # newest export wins


def test_empty_exports_and_closed_beats_newer_open(tmp_path):
    out = "\n".join(W.run(int(T0.value // 10**6), [], None, int(T0.value // 10**6)))
    assert "| **zone (round numbers)** | 0 |" in out and "COLLECTING (zone N 0/15 · 0/8 days)" in out
    a = _fills([(0, "AUSDT", Z, -1.5)])
    op = a.copy()
    op.loc[0, "status"] = "OPEN"
    p_c = _write(tmp_path, a, "2026-11-30_00-00-00")
    p_o = _write(tmp_path, op, "2026-12-01_00-00-00")
    os.utime(p_c[0], (1, 1))
    os.utime(p_o[0], (2, 2))
    assert W.load(p_c + p_o)[0].pair.tolist() == ["AUSDT"]


def test_damaged_record_never_refreezes(tmp_path):
    df = _fills(_zone_rest([-1.5] * 12 + [0.3] * 3))
    last = pd.to_datetime(df.opened_at).max()
    paths = _write(tmp_path, df, (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    now, since = int((T0 + pd.Timedelta(days=30)).value // 10**6), int(T0.value // 10**6)
    for bad in ('{"SPIKE_FADE": "x"}', '{"SPIKE_FADE": {"state": "x"}}'):
        st = tmp_path / "w.json"
        st.write_text(bad)
        out = "\n".join(W.run(now, paths, str(st), since))
        assert "frozen state corrupt" in out and "Frozen verdict" not in out and st.read_text() == bad


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_fade_nostretch_watch as _fns")
    assert "_fns.run(now_ms)" in src[j:j + 200] and "except Exception as _fns_e" in src[j:j + 400]
    import observe_registry as OR
    era = OR.BOOKED["scout_fade_nostretch_watch.py"]
    assert era["spike_fade_sl_pct"] == -1.5 and era["runner_trail_short_arm_peak"] == 0.4
    assert OR.diffs(era, OR.thresholds()) == []                              # pinned to today's fade exit (a change → STALE)
