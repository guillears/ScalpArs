"""👁 DECISION_LOG 580 — the three WILLY / WILLY_HT watch lines (CROWDED_LONGS · NEAR_SPIKE_HIGH · REANCHORED): pure zones, lag-safe ratio
pick, the frozen bar per sleeve, the loader extras, registry + opportunity-scout hook."""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import scout_willy_ht_lines as W              # noqa: E402
import scout_willy_failed_spike_watch as WF   # noqa: E402
import observe_registry as OR                 # noqa: E402


def test_zone_crowded():
    assert W.zone_crowded(1.20, 0.9) == "zone" and W.zone_crowded(1.0, 1.07) == "zone"
    assert W.zone_crowded(1.19, 1.0624) == "rest"                 # cuts are strict ">"
    assert W.zone_crowded(None, None) == "unscored"
    assert W.zone_crowded(1.5, None) == "zone"                    # one side already crowded → zone
    assert W.zone_crowded(1.0, None) == "unscored"                # one side missing, the other calm → cannot call it rest
    assert W.zone_crowded("x", float("nan")) == "unscored"


def test_zone_near_high_and_off_peak():
    assert abs(W.off_peak(98.7, 100.0) - (-1.3)) < 1e-9
    assert W.zone_near_high(98.7, 100.0) == "zone" and W.zone_near_high(98.5, 100.0) == "rest"
    assert W.zone_near_high(98.6, 100.0) == "rest"                # exactly −1.4 → rest (strict >)
    for bad in ((None, 100), (100, None), (0, 100), (100, 0), ("x", 1)):
        assert W.zone_near_high(*bad) == "unscored"


def test_zone_reanchored():
    assert W.zone_reanchored("A", 1.5) == "zone" and W.zone_reanchored("a", 0.4) == "rest"
    assert W.zone_reanchored("A", 1.0) == "rest"                  # strict >
    assert W.zone_reanchored("B", 5.0) == "unscored" and W.zone_reanchored(None, 5.0) == "unscored"
    assert W.zone_reanchored("A", None) == "unscored"


def test_pick_ratio_is_lag_safe():
    e = 10_000_000
    rows = [dict(timestamp=e - 10 * 60_000, longShortRatio="1.1"), dict(timestamp=e - 60_000, longShortRatio="1.3"),
            dict(timestamp=e - 30_000, longShortRatio="9.9"), dict(timestamp=e, longShortRatio="9.9")]
    assert W.pick_ratio(rows, e) == 1.3                           # the last 5m period that ENDED ≤ the fill − 1 min
    assert W.pick_ratio(rows[2:], e) is None and W.pick_ratio([], e) is None
    assert W.pick_ratio([dict(timestamp="x", longShortRatio=1)], e) is None


def test_ratios_cache_and_old_fills(monkeypatch):
    calls = []
    monkeypatch.setattr(W, "_get", lambda path, q: calls.append(path) or [dict(timestamp=q["endTime"] - 6 * 60_000, longShortRatio="1.25")])
    cache, now = {}, 100 * 86_400_000
    assert W.ratios("AUSDT", now - 3_600_000, cache, now) == (1.25, 1.25) and len(calls) == 2
    assert W.ratios("AUSDT", now - 3_600_000, cache, now) == (1.25, 1.25) and len(calls) == 2   # cached
    assert W.ratios("BUSDT", now - 40 * 86_400_000, cache, now) == (None, None) and len(calls) == 2   # beyond the API window
    monkeypatch.setattr(W, "_get", lambda path, q: [])
    try:
        W.ratios("CUSDT", now - 3_600_000, cache, now)
        assert False
    except ValueError:
        pass
    assert "CUSDT|%d" % (now - 3_600_000) not in cache        # empty / partial → never cached, retried next run


def test_call_cap(monkeypatch):
    monkeypatch.setattr(W, "_calls", [W.MAX_CALLS])
    try:
        W._get("/x", {})
        assert False
    except W.NetStop:
        pass


def test_no_crowded_freeze_after_a_failed_fetch(monkeypatch, tmp_path):
    rows = []
    for i in range(20):
        rows.append(dict(opened_at=f"2026-11-{20 + i % 9:02d}T{i % 20:02d}:00:00", closed_at="", pair=f"P{i}USDT", direction="LONG",
                         entry_strategy="FRENZY_WILLY_HT", status="CLOSED", pnl_percentage=-1.0, pnl=-1.0, close_reason="STOP_LOSS",
                         entry_frenzy_vs_vwap_pct=1.0, entry_adx=30, entry_ema_gap_8_13=2.0, entry_price=90.0,
                         entry_frenzy_willy_spike_high=100.0, entry_frenzy_hours=0.3, entry_frenzy_willy_trigger="A"))
    p = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    monkeypatch.setattr(W, "_load_cache", lambda: {})
    monkeypatch.setattr(W, "_save_cache", lambda c: None)
    seq = iter([(1.5, 1.5)] * 16)
    def flaky(pair, e, cache, now):
        try:
            return next(seq)
        except StopIteration:
            raise W.urllib.error.URLError("down")
    monkeypatch.setattr(W, "ratios", flaky)
    st = tmp_path / "st.json"
    txt = "\n".join(W.run(paths=[str(p)], state_path=str(st), since_ms=int(pd.Timestamp("2026-11-19").value // 1_000_000),
                          now_ms=int(pd.Timestamp("2026-12-30").value // 1_000_000)))
    assert "no CROWDED_LONGS verdict is frozen this run" in txt
    import json
    saved = json.load(open(st)) if st.exists() else {}
    assert not any(k.endswith("|CROWDED_LONGS") for k in saved)


def _frame(n, zone_pct, rest_pct, days):
    rows = []
    for i in range(n):
        rows.append(dict(pct=zone_pct, day=f"d{i % days}", pair=f"P{i % 7}", z="zone"))
        rows.append(dict(pct=rest_pct, day=f"d{i % days}", pair=f"Q{i % 7}", z="rest"))
    return pd.DataFrame(rows)


def test_verdict_bar_per_sleeve():
    f = _frame(10, -1.0, 1.0, 9)
    assert W.verdict(f[f.z == "zone"], f[f.z == "rest"], 0.69)[0] == "COLLECTING"
    f = _frame(16, 1.0, -1.0, 9)
    assert W.verdict(f[f.z == "zone"], f[f.z == "rest"], 0.69)[0] == "RETIRED"
    z = pd.DataFrame([dict(pct=(-4.0 if i % 3 else 1.0), day=f"d{i % 12}", pair=f"P{i % 9}") for i in range(24)])
    r = pd.DataFrame([dict(pct=1.5, day=f"d{i % 10}", pair="R") for i in range(20)])
    assert W.verdict(z, r, 0.667)[0] == W.CANDIDATE
    assert W.BE_WR == {"FRENZY_WILLY": 0.690, "FRENZY_WILLY_HT": 0.444}   # 🔁 (587) HT re-based on TP +5 / SL −4 (was 0.667 at +2 / −4)


def test_not_evaluable_without_the_commit(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    txt = "\n".join(W.run(paths=[], state_path=None))
    assert "NOT EVALUABLE" in txt


def test_run_on_a_fixture(tmp_path):
    rows = []
    for i in range(3):
        rows.append(dict(opened_at=f"2026-11-20T0{i}:00:00", closed_at=f"2026-11-20T0{i}:30:00", pair=f"P{i}USDT", direction="LONG",
                         entry_strategy="FRENZY_WILLY_HT", status="CLOSED", pnl_percentage=2.0, pnl=1.0, close_reason="FRENZY_TP",
                         entry_frenzy_vs_vwap_pct=1.0, entry_adx=30, entry_ema_gap_8_13=2.0, entry_price=99.0,
                         entry_frenzy_willy_spike_high=100.0, entry_frenzy_hours=(1.5 if i == 0 else 0.3), entry_frenzy_willy_trigger="A"))
    p = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    c, _ = W.load([str(p)])
    assert {"entry_price", "entry_frenzy_willy_spike_high", "entry_frenzy_hours", "entry_frenzy_willy_trigger"} <= set(c.columns)
    plain, _ = WF.load([str(p)])
    assert "entry_frenzy_willy_trigger" not in plain.columns      # the default loader is unchanged
    txt = "\n".join(W.run(paths=[str(p)], state_path=None, since_ms=int(pd.Timestamp("2026-11-19").value // 1_000_000), fetch=False))
    assert "NEAR_SPIKE_HIGH" in txt and "REANCHORED" in txt and "COLLECTING" in txt
    sc, _ = W.score(c, now_ms=0, fetch=False)
    assert list(sc.z_NEAR_SPIKE_HIGH) == ["zone"] * 3 and sorted(sc.z_REANCHORED) == ["rest", "rest", "zone"]


def test_registry_and_hook():
    assert "scout_willy_ht_lines.py" in OR.classified()
    assert OR.BOOKED["scout_willy_ht_lines.py"]["frenzy_willy_ht_tp_pct"] == 5.0   # 🔁 re-based DL 587
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_ht_lines" in src and "import scout_exh_exit_override" in src
    gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read()
    for f in ("reports/SCOUT_WILLY_HT_LINES.json", "reports/.willy_ht_lines_cache.json", "reports/.exh_exit_override_cache.json"):
        assert f in gi
