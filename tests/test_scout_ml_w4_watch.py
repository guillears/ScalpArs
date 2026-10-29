"""👁 Oct-29: the ML_W4 watch line (scripts/scout_ml_w4_watch.py, DECISION_LOG 410) — frozen constants, cohort (full-size, forward,
operator-closed out, unscored apart), the expectancy bar's three outcomes, the freeze-once / OPEN-hold rule, and the scout hook."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_ml_w4_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")


def _fills(spec):
    """spec = [(day offset, pair, w4, pct, src)] → an orders export DataFrame (CLOSED momentum longs)."""
    rows = []
    for i, (d, pair, w4, pct, src) in enumerate(spec):
        o = T0 + pd.Timedelta(days=d, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                         pair=pair, direction="LONG", entry_strategy="MOMENTUM", status="CLOSED", pnl_percentage=pct, pnl=pct * 10,
                         close_reason="RUNNER_TRAIL", cell_multiplier_source=src, entry_pattern_w4_match=w4))
    return pd.DataFrame(rows)


def _write(tmp_path, df, name="scalpars_orders_paper_2026-11-30_00-00-00.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return [str(p)]


def test_frozen_constants():
    assert (W.DL, W.N_MIN, W.DAYS_MIN, W.BE_WR, W.P_MIN, W.SHARE_MAX, W.BOOT_N, W.BOOT_SEED) == (410, 15, 8, 0.620, 0.95, 0.50, 4000, 7)
    assert W.ZONE_COL == "entry_pattern_w4_match" and W.STATE.endswith(os.path.join("reports", "SCOUT_ML_W4_WATCH.json"))
    assert "reports/SCOUT_ML_W4_WATCH.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_cohort_full_size_forward_and_unscored(tmp_path):
    df = _fills([(0, "AUSDT", True, -1.0, "NONEXP_CALM3D"), (0, "BUSDT", False, 0.5, "UNMATCHED"),
                 (0, "CUSDT", True, -1.0, "GAPFLAT_PROBE"), (0, "DUSDT", None, 0.4, "UNMATCHED"), (-5, "EUSDT", True, -1.0, "NONEXP_CALM3D")])
    df.loc[len(df)] = dict(df.iloc[1], pair="FUSDT", entry_strategy="FRENZY_LITE")
    c, opens, n_probe = W.load(_write(tmp_path, df))
    assert n_probe == 1 and "CUSDT" not in set(c.pair) and "FUSDT" not in set(c.pair)
    assert dict(zip(c.pair, c.zone)) == {"AUSDT": "zone", "BUSDT": "rest", "DUSDT": "unscored", "EUSDT": "zone"}
    out = "\n".join(W.run(now_ms=int((T0 + pd.Timedelta(days=30)).value // 10**6), paths=_write(tmp_path, df),
                          state_path=None, since_ms=int(T0.value // 10**6)))
    assert "| **zone (W4-matched)** | 1 |" in out and "| rest (not W4) | 1 |" in out and "UNSCORED (no W4 stamp) — never counted | 1 |" in out
    assert "probe fires out (all exports: 1)" in out and "COLLECTING (zone N 1/15 · 1/8 days)" in out


def _zone_rest(zone_pcts, rest_pct=0.3, days=8):
    spec = [(i % days, f"Z{i}USDT", True, p, "NONEXP_CALM3D") for i, p in enumerate(zone_pcts)]
    spec += [(i % days, f"R{i}USDT", False, rest_pct, "UNMATCHED") for i in range(20)]
    return spec


def _cohort(tmp_path, spec):
    c, _, _ = W.load(_write(tmp_path, _fills(spec)))
    return c[c.zone == "zone"], c[c.zone == "rest"]


def test_verdict_three_outcomes(tmp_path):
    z, r = _cohort(tmp_path, _zone_rest([0.5] * 15))
    assert W.verdict(z, r)[0] == "RETIRED"                                  # zone avg ≥ rest avg
    z, r = _cohort(tmp_path, _zone_rest([-0.8] * 12 + [0.4] * 3))
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE and "WR 20% (breakeven 62.0%)" in det, det
    z, r = _cohort(tmp_path, _zone_rest([-0.5] * 3 + [0.05] * 12))           # avg < 0, losses spread over 3 days, but WR 80 % ≥ breakeven
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "WR 80%" in det, det
    z, r = _cohort(tmp_path, _zone_rest([0.1] * 15))                         # avg ≥ 0 but < rest → RETIRED (never a filter)
    assert W.verdict(z, r)[0] == "RETIRED"
    z, r = _cohort(tmp_path, _zone_rest([-3.0] + [-0.01] * 3 + [0.2] * 11))   # WR 73 % ≥ breakeven, one day carries the loss → keep
    assert W.verdict(z, r)[0] == "KEEP OBSERVING"
    z, r = _cohort(tmp_path, _zone_rest([-0.8] * 10 + [0.6] * 5))            # WR 33 %, mean −0.33, loss spread over pairs / days → CANDIDATE
    st, det = W.verdict(z, r)
    assert st == W.CANDIDATE, det
    rows = [(i % 8, "SAMEUSDT", True, -0.9, "NONEXP_CALM3D") for i in range(10)] + [(i % 8, f"Z{i}USDT", True, 0.5, "NONEXP_CALM3D") for i in range(5)]
    rows += [(i % 8, f"R{i}USDT", False, 0.3, "UNMATCHED") for i in range(20)]
    z, r = _cohort(tmp_path, rows)                                          # all the loss on one pair → KEEP OBSERVING
    st, det = W.verdict(z, r)
    assert st == "KEEP OBSERVING" and "top pair 100%" in det, det
    z, r = _cohort(tmp_path, _zone_rest([-0.8] * 15, days=1))                 # one day → still collecting (days < 8)
    assert W.verdict(z, r)[0] == "COLLECTING"


def test_freeze_once_and_open_hold(tmp_path):
    spec = _zone_rest([-0.8] * 12 + [0.4] * 3)
    paths = _write(tmp_path, _fills(spec))
    st = tmp_path / "w4.json"
    now = int((T0 + pd.Timedelta(days=30)).value // 10**6)
    since = int(T0.value // 10**6)
    held = "\n".join(W.run(now, paths, str(st), since, open_ts=[(T0, "OPEN XUSDT")]))
    assert "freezing deferred" in held and not st.exists()                 # an OPEN long opened ≤ the prefix end holds the freeze
    out = "\n".join(W.run(now, paths, str(st), since, open_ts=[]))
    assert "Frozen verdict" in out and W.CANDIDATE in out.split("Live")[0].split("FROZEN bar")[0] and json.load(open(st))["first"]["state"] == W.CANDIDATE
    flip = _fills(_zone_rest([0.9] * 15))                                   # later data never re-decides
    out2 = "\n".join(W.run(now, _write(tmp_path, flip, "scalpars_orders_paper_2026-11-30_01-00-00.csv"), str(st), since, open_ts=[]))
    assert W.CANDIDATE in out2.split("Live")[0] and json.load(open(st))["first"]["state"] == W.CANDIDATE


def test_not_evaluable_without_commit_and_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_ml_w4_watch as _mw4")
    assert "_mw4.run(now_ms)" in src[j:j + 200] and "except Exception as _mw4_e" in src[j:j + 400]
    import observe_registry as OR
    assert "scout_ml_w4_watch.py" in OR.BOOKED and OR.BOOKED["scout_ml_w4_watch.py"] is None


def test_dedup_open_hold_from_newest_export_and_stamp_variants(tmp_path):
    """review: CLOSED beats OPEN and the newer export wins; an OPEN row holds the freeze only while the NEWEST export still shows it;
    string / +00:00 / no-route-column variants load; no exports → empty, no crash."""
    import os as _os
    old, new = tmp_path / "scalpars_orders_paper_2026-11-01_00-00-00.csv", tmp_path / "scalpars_orders_paper_2026-11-02_00-00-00.csv"
    a = _fills([(0, "AUSDT", "True", -1.0, "NONEXP_CALM3D"), (0, "BUSDT", "1.0", 0.2, "NONEXP_CALM3D")])
    a.loc[0, "opened_at"] = a.loc[0, "opened_at"] + "+00:00"
    gone = dict(a.iloc[1], pair="GONEUSDT", status="OPEN")                     # open in the OLD export only (wiped by a reset)
    pd.concat([a.assign(status="OPEN"), pd.DataFrame([gone])]).to_csv(old, index=False)
    a2 = a.copy(); a2.loc[1, "pnl_percentage"] = 0.7
    live_open = dict(a.iloc[1], pair="LIVEUSDT", status="OPEN")
    pd.concat([a2, pd.DataFrame([live_open])]).drop(columns=["cell_multiplier_source"]).to_csv(new, index=False)
    _os.utime(old, (1, 1)); _os.utime(new, (2, 2))
    c, opens, _ = W.load([str(old), str(new)])
    assert dict(zip(c.pair, c.pct)) == {"AUSDT": -1.0, "BUSDT": 0.7} and set(c.zone) == {"zone"}
    assert [lab for _, lab in opens] == ["OPEN LIVEUSDT"]                     # GONEUSDT (old export only) never holds
    e, o, n = W.load([])
    assert len(e) == 0 and o == [] and n == 0


def test_freeze_waits_for_an_export_past_the_momentum_cap(tmp_path):
    """deep review: exports hold CLOSED rows only → the freeze waits until the newest export is ≥ crossing + the 1200-min cap."""
    spec = _zone_rest([-0.8] * 12 + [0.4] * 3)
    df = _fills(spec)
    last = pd.to_datetime(df.opened_at).max()
    early = (last + pd.Timedelta(hours=2)).strftime("%Y-%m-%d_%H-%M-%S")
    late = (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S")
    st = tmp_path / "w4.json"
    now, since = int((T0 + pd.Timedelta(days=30)).value // 10**6), int(T0.value // 10**6)
    out = "\n".join(W.run(now, _write(tmp_path, df, f"scalpars_orders_paper_{early}.csv"), str(st), since, open_ts=[]))
    assert "wait for a later export" in out and not st.exists()
    out = "\n".join(W.run(now, _write(tmp_path, df, f"scalpars_orders_paper_{late}.csv"), str(st), since, open_ts=[]))
    assert st.exists() and "Frozen verdict" in out
    assert W.cap_minutes() == 1200.0
    dry = "\n".join(W.run(now, _write(tmp_path, df, f"scalpars_orders_paper_{late}.csv"), None, since, open_ts=[]))
    assert "Frozen verdict" not in dry and "Now:" in dry                       # a dry run never claims a frozen verdict
