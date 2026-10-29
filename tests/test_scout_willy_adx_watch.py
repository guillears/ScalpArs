"""👁 Oct-29: the WILLY_ADX55 watch line (scripts/scout_willy_adx_watch.py, DECISION_LOG 412) — frozen constants, cohort (WILLY / HT
apart, forward, operator-closed out, unscored apart), the expectancy bar's outcomes, the export-age deferral, freeze once, the hook and
the registry. Hermetic: temp exports and state, no network, no git."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_adx_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")


def _fills(spec, es="FRENZY_WILLY"):
    """spec = [(day offset, pair, adx, pct)] → CLOSED WILLY-family rows."""
    rows = []
    for i, (d, pair, adx, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
                         pair=pair, direction="LONG", entry_strategy=es, status="CLOSED", pnl_percentage=pct, pnl=pct * 10,
                         close_reason="FRENZY_TP" if pct > 0 else "STOP_LOSS", entry_adx=adx))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2026-12-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _zone_rest(zone_pcts, rest_pct=1.25, days=8):
    return ([(i % days, f"Z{i}USDT", 60.0, p) for i, p in enumerate(zone_pcts)]
            + [(i % days, f"R{i}USDT", 30.0, rest_pct) for i in range(20)])


def _cohort(tmp_path, spec):
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    return c[c.zone == "zone"], c[c.zone == "rest"]


def test_frozen_constants():
    assert (W.DL, W.ADX_CUT, W.N_MIN, W.DAYS_MIN, W.BE_WR, W.P_MIN, W.SHARE_MAX, W.CAP_DEFAULT) == (412, 55.0, 15, 8, 0.690, 0.95, 0.50, 90)
    assert W.cap_minutes() == 90.0                                           # the live config's WILLY max hold
    assert W.SLEEVES == ("FRENZY_WILLY", "FRENZY_WILLY_HT") and W.ZONE_COL == "entry_adx"
    assert "reports/SCOUT_WILLY_ADX55.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_cohort_zone_boundary_and_unscored(tmp_path):
    df = pd.concat([_fills([(0, "AUSDT", 55.0, 1.0), (0, "BUSDT", 55.01, -4.0), (0, "CUSDT", None, 1.0)]),
                    _fills([(0, "DUSDT", 70.0, -4.0)], es="FRENZY_WILLY_HT"), _fills([(0, "EUSDT", 70.0, -3.0)], es="FRENZY_LITE")])
    c, _ = W.load(_write(tmp_path, df))
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "rest", "BUSDT": "zone", "CUSDT": "unscored", "DUSDT": "zone"}   # 55 itself = rest
    out = "\n".join(W.run(now_ms=int((T0 + pd.Timedelta(days=30)).value // 10**6), paths=_write(tmp_path, df), state_path=None,
                          since_ms=int(T0.value // 10**6)))
    assert out.count("| **zone (ADX > 55)** | 1 |") == 2 and "EUSDT" not in out and "COLLECTING (zone N 1/15 · 1/8 days)" in out


def test_verdict_outcomes(tmp_path):
    z, r = _cohort(tmp_path, _zone_rest([-4.0] * 12 + [1.25] * 3))
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE and "WR 20% (breakeven 69.0%)" in det, det
    z, r = _cohort(tmp_path, _zone_rest([1.5] * 15))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone ≥ rest
    z, r = _cohort(tmp_path, _zone_rest([0.2] * 15, rest_pct=1.25))
    assert W.verdict(z, r)[0] == "RETIRED"                                 # zone ≥ 0 never a filter
    z, r = _cohort(tmp_path, _zone_rest([-4.0] * 4 + [1.25] * 11))
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "WR 73%" in det, det                # mean −0.15 < 0, but WR 73 % ≥ the 69 % breakeven
    rows = [(i % 8, "SAMEUSDT", 60.0, -4.0) for i in range(10)] + [(i % 8, f"Z{i}USDT", 60.0, 1.25) for i in range(5)]
    rows += [(i % 8, f"R{i}USDT", 30.0, 1.25) for i in range(20)]
    z, r = _cohort(tmp_path, rows)
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "top pair 100%" in det, det          # one pair carries the loss
    z, r = _cohort(tmp_path, _zone_rest([-4.0] * 15, days=1))
    assert W.verdict(z, r)[0] == "COLLECTING"


def test_freeze_once_per_sleeve_and_export_age_wait(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "cap_minutes", lambda *a, **k: 90.0)             # independent of the live config (review L3)
    df = _fills(_zone_rest([-4.0] * 12 + [1.25] * 3))
    last = pd.to_datetime(df.opened_at).max()
    st = tmp_path / "w.json"
    now, since = int((T0 + pd.Timedelta(days=30)).value // 10**6), int(T0.value // 10**6)
    early = _write(tmp_path, df, (last + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(now, early, str(st), since))
    assert "newest export < the crossing fill + 90 min" in out and not st.exists()
    late = _write(tmp_path, df, (last + pd.Timedelta(hours=3)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(now, late, str(st), since))
    assert "Frozen verdict" in out and json.load(open(st))["FRENZY_WILLY"]["state"] == W.CANDIDATE
    assert "FRENZY_WILLY_HT" not in json.load(open(st))                    # HT decided on its own (nothing yet)
    flip = _fills(_zone_rest([1.5] * 15))
    out2 = "\n".join(W.run(now, _write(tmp_path, flip, "2026-12-05_00-00-00"), str(st), since))
    assert W.CANDIDATE in out2.split("Live")[0] and json.load(open(st))["FRENZY_WILLY"]["state"] == W.CANDIDATE   # never re-decided
    dry = "\n".join(W.run(now, late, None, since))
    assert "Frozen verdict" not in dry and "Now:" in dry


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_willy_adx_watch as _wax")
    assert "_wax.run(now_ms)" in src[j:j + 200] and "except Exception as _wax_e" in src[j:j + 400]
    import observe_registry as OR
    era = OR.BOOKED["scout_willy_adx_watch.py"]
    assert era["frenzy_willy_high_target_enabled"] is True and era["frenzy_willy_stop_pct"] == 4.0
    assert OR.diffs(era, OR.thresholds()) == []                              # pinned to today's V2b exit (a change → STALE)


def test_operator_closed_forward_floor_dedup_and_relative_retire(tmp_path):
    """review: operator-closed fills and fills before the floor are out; the same fill in two exports counts once (newest wins); a WILLY
    and an HT fill on the same pair and second stay apart; zone < 0 but ≥ rest retires."""
    since = int(T0.value // 10**6)
    a = _fills([(0, "AUSDT", 60.0, -4.0), (0, "MANUSDT", 60.0, -4.0), (-3, "OLDUSDT", 60.0, -4.0)])
    a.loc[1, "close_reason"] = "MANUAL"
    h = _fills([(0, "AUSDT", 60.0, 1.25)], es="FRENZY_WILLY_HT"); h["opened_at"] = a.loc[0, "opened_at"]
    old = a.copy(); old.loc[0, "pnl_percentage"] = -1.0
    p_old = _write(tmp_path, old, "2026-11-30_00-00-00"); p_new = _write(tmp_path, pd.concat([a, h]), "2026-12-01_00-00-00")
    os.utime(p_old[0], (1, 1)); os.utime(p_new[0], (2, 2))
    out = "\n".join(W.run(int((T0 + pd.Timedelta(days=30)).value // 10**6), p_old + p_new, None, since))
    assert "MANUSDT" in out and "OLDUSDT" not in out                         # listed as operator-closed; the pre-floor fill never counted
    assert out.count("| **zone (ADX > 55)** | 1 |") == 2                    # WILLY AUSDT and HT AUSDT apart, MANUAL out
    c, _ = W.load(p_old + p_new)
    assert c[(c.pair == "AUSDT") & (c.entry_strategy == "FRENZY_WILLY")].pct.tolist() == [-4.0]   # newest export wins
    z, r = _cohort(tmp_path, _zone_rest([-0.5] * 15, rest_pct=-1.0))
    assert W.verdict(z, r)[0] == "RETIRED"                                   # zone −0.5 ≥ rest −1.0
