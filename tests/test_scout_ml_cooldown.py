"""🧊 Oct-7: tests for the scout ML cooldown observations (scripts/scout_ml_cooldown.py). Covered: stop-close detection with ladder suffixes,
the strict 30.0-min boundary, merged stop windows, the ≥ 2-in-120-min cluster count, the $ de-multiply (the % is already 1×), bootstrap
determinism, the expectancy bar's collecting / propose / not-established states, the fresh floor, the export dedup + filters, the write-once
store, and the opportunity_scout hook surviving a crash. No network."""
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_ml_cooldown as MC  # noqa: E402

T = pd.Timestamp


@pytest.fixture(autouse=True)
def _no_real_store(tmp_path, monkeypatch):
    """never the real reports/SCOUT_ML_COOLDOWN.csv"""
    monkeypatch.setattr(MC, "CSV", str(tmp_path / "SCOUT_ML_COOLDOWN.csv"))


def test_stop_detection_with_suffixes():
    assert MC.is_stop("STOP_LOSS L1") and MC.is_stop("STOP_LOSS") and MC.is_stop("STOP_LOSS_WIDE L1") and MC.is_stop("STOP_LOSS L12")
    for r in ("TRAILING_STOP L2", "RUNNER_TRAIL", "FL_STOP_LOSS L1", "MANUAL_SL", "HARD_TP_LADDER L1", "RH_HARD_STOP", None, float("nan"), ""):
        assert not MC.is_stop(r), r
    assert MC.is_stop_alt("RH_HARD_STOP") and MC.is_stop_alt("STOP_LOSS L1") and not MC.is_stop_alt("TRAILING_STOP L2")


def test_thirty_minute_boundary_is_strict():
    s = T("2026-10-08 10:00:00")
    f, trig, m = MC.sequential([T("2026-10-08 09:50"), T("2026-10-08 10:29:59.999"), T("2026-10-08 10:30:00")], [s, None, None], [True, False, False])
    assert f == [False, True, False] and trig[1] == s and m[1] == pytest.approx(29.99998, abs=1e-4)
    f, _, _ = MC.sequential([T("2026-10-08 09:50"), T("2026-10-08 09:59")], [s, None], [True, False])
    assert f == [False, False]                                            # a stop that closes after the open never triggers


def test_sequential_cohort_fill_never_starts_a_cooldown():
    """the study's sequential semantics: B is in the cohort (after A's stop) and stops itself; C, 10 min after B's stop but 45 min after A's,
    is NOT flagged — a refused fill would not exist once armed."""
    f, _, _ = MC.sequential([T("2026-10-08 09:50"), T("2026-10-08 10:05"), T("2026-10-08 10:45")],
                            [T("2026-10-08 10:00"), T("2026-10-08 10:35"), None], [True, True, False])
    assert f == [False, True, False]
    f2, _, _ = MC.sequential([T("2026-10-08 09:50"), T("2026-10-08 10:35"), T("2026-10-08 10:45")],
                             [T("2026-10-08 10:00"), T("2026-10-08 10:40"), None], [True, True, False])
    assert f2 == [False, False, True]                                     # an accepted fill's stop does start one


def test_sixty_minute_chain_counting():
    o = [T("2026-10-08 10:00"), T("2026-10-08 10:59"), T("2026-10-08 11:59"), T("2026-10-08 13:00"), T("2026-10-08 13:01")]
    c = MC.chains(o)
    assert c == [o[0], o[0], o[0], o[3], o[3]]                           # gap exactly 60 continues, > 60 starts a chain
    coh = pd.DataFrame(dict(pct=[-0.1] * 20, window=[str(x) for x in (c * 4)], pair=[f"P{i}" for i in range(20)]))
    assert MC.decide(coh)[0] == "collecting" and "windows 2/8" in MC.decide(coh)[1]


def test_merged_30min_windows_display():
    w = MC.stop_windows([T("2026-10-08 10:00"), T("2026-10-08 10:20"), T("2026-10-08 11:00"), T("2026-10-08 10:50")])
    assert w == [(T("2026-10-08 10:00"), T("2026-10-08 10:50")), (T("2026-10-08 10:50"), T("2026-10-08 11:30"))]
    assert MC.window_of(T("2026-10-08 10:49"), w) == "2026-10-08T10:00:00" and MC.window_of(T("2026-10-08 12:00"), w) is None


def test_cluster_two_in_120_sequential_strict():
    o = [T("2026-10-08 10:00"), T("2026-10-08 11:00"), T("2026-10-08 11:30"), T("2026-10-08 11:40"), T("2026-10-08 12:00")]
    f, n = MC.cluster_cap(o)
    assert f == [False, False, True, True, False] and n == [0, 1, 2, 2, 1]   # 10:00 is exactly 120 min before 12:00 → out (strict)


def test_demultiply():
    assert MC.demux_usd(-150.0, 1.5) == pytest.approx(-100.0)
    assert MC.demux_usd(30.0, 1.5, 2.0) == pytest.approx(10.0)
    assert MC.demux_usd(10.0, None, None) == pytest.approx(10.0) and MC.demux_usd(10.0, 0, -1) == pytest.approx(10.0)
    assert MC.demux_usd(None, 1.5) is None


def test_bootstrap_deterministic_and_signed():
    w = [f"w{i}" for i in range(10) for _ in range(2)]
    x = np.array([-0.2, 0.1] * 10)
    assert MC.boot_p_neg(x, w) == MC.boot_p_neg(x, w)
    assert MC.boot_p_neg(x, w) > 0.95 and MC.boot_p_neg(-x, w) < 0.05
    assert MC.boot_p_neg([1.0, -1.0], ["a", "a"]) is None


def _coh(n=20, nw=10, pct=None):
    pct = np.array([-0.2, 0.1] * (n // 2)) if pct is None else np.asarray(pct)
    return pd.DataFrame(dict(pct=pct, window=[f"w{i * nw // n}" for i in range(n)], pair=[f"P{i}" for i in range(n)]))


def test_bar_states():
    assert MC.decide(_coh())[0] == "propose"
    assert MC.decide(_coh(n=14))[0] == "collecting"
    assert MC.decide(_coh(nw=7))[0] == "collecting"
    assert MC.decide(_coh(pct=[0.2, -0.1] * 10))[0] == "not_established"           # winning cohort
    c = _coh(); c.loc[0, "pct"] = -10.0
    assert MC.decide(c)[0] == "not_established"                                      # one window ≥ 50 % of the loss
    c = _coh(); c["pair"] = ["X"] * 10 + [f"P{i}" for i in range(10)]                 # pair X carries every loss in rows 0-9 (50 % of rows)
    c.loc[[i for i in range(10, 20)], "pct"] = 0.1; c.loc[[i for i in range(10)], "pct"] = [-0.2, 0.1] * 5
    c.loc[[1, 3, 5, 7, 9], "pct"] = 0.1
    assert MC.top_loss_share(c.pct.values, c.pair.values) == pytest.approx(1.0) and MC.decide(c)[0] == "not_established"   # pair leg
    n = pd.concat([_coh(n=14, nw=7), _coh(n=6, nw=3).assign(pct=np.nan)], ignore_index=True)
    assert MC.decide(n)[0] == "collecting" and "N 14/15" in MC.decide(n)[1]          # NaN pct dropped before every leg
    c = _coh(pct=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -1.0, -1.0, -1.0] * 2)           # WR 70 % ≥ 61.5 → never propose
    assert MC.decide(c)[0] == "not_established"


def _fills():
    return pd.DataFrame(dict(
        opened_at=["2026-10-07T23:30:00", "2026-10-07T23:50:00", "2026-10-08T00:10:00", "2026-10-08T00:30:00", "2026-10-08T03:00:00",
                   "2026-10-08T00:00:00+00:00"],
        closed_at=["2026-10-07T23:39:00.123", "2026-10-08T00:05:00", "2026-10-08T00:40:00", None, "2026-10-08T03:30:00", "2026-10-08T00:01:00"],
        pair=["A", "B", "C", "D", "E", "F"], status=["CLOSED", "CLOSED", "CLOSED", "OPEN", "CLOSED", "CLOSED"],
        close_reason=["STOP_LOSS L1", "STOP_LOSS_WIDE L1", "RUNNER_TRAIL", None, "RUNNER_TRAIL L1", "RH_HARD_STOP"],
        pnl=[-15.0, -21.0, 6.0, None, 3.0, -1.0], pnl_percentage=[-0.7, -0.7, 0.2, None, 0.1, -0.1], cell_multiplier=[1.5, 1.5, 1.5, 1.0, 1.0, 1.0]))


def test_tag_flags_and_fresh_floor():
    t = MC.tag(_fills()).set_index("pair")
    assert sorted(t.index) == ["A", "B", "C", "E", "F"]                   # OPEN fill not tallied
    assert not t.loc["A", "cd_flag"]
    assert t.loc["B", "cd_flag"] and not t.loc["B", "fresh"]               # 10 min after A's stop, before the floor → reference
    assert t.loc["F", "fresh"] and t.loc["F", "opened_at"] == "2026-10-08T00:00:00"   # tz-suffixed stamp parsed as UTC, floor inclusive
    assert t.loc["F", "cd_flag"]                                           # 21 min after A's stop (A accepted)
    assert not t.loc["C", "cd_flag"] and t.loc["C", "cd_any"]              # B's stop ignored (B is in the cohort); A's is > 30 min back
    assert t.loc["F", "cd_alt"] and not t.loc["C", "cd_alt"]               # F's RH_HARD_STOP ignored: F is itself in the alt cohort
    assert t.loc["B", "chain"] == t.loc["C", "chain"] == "2026-10-07T23:30:00" and t.loc["E", "chain"] == "2026-10-08T03:00:00"
    assert t.loc["C", "c120_prior"] == 2 and t.loc["C", "c120_flag"]        # A, B accepted; F flagged at 00:00 (A, B) → not accepted
    assert t.loc["C", "usd_1x"] == pytest.approx(4.0) and t.loc["C", "pnl_pct"] == pytest.approx(0.2) and t.loc["C", "pnl_usd"] == pytest.approx(6.0)


def test_load_orders_dedup_and_filters(tmp_path):
    base = dict(direction="LONG", status="CLOSED", close_reason="RUNNER_TRAIL", pnl=1.0, cell_multiplier=1.0, closed_at="2026-10-08T02:00:00")
    old = pd.DataFrame([dict(base, opened_at="2026-10-08T01:00:00.5", pair="A", entry_strategy="MOMENTUM", pnl_percentage=0.1, cell_multiplier_source="UNMATCHED")])
    new = pd.DataFrame([dict(base, opened_at="2026-10-08T01:00:00", pair="A", entry_strategy=None, pnl_percentage=0.3, cell_multiplier_source="UNMATCHED"),
                        dict(base, opened_at="2026-10-08T01:05:00", pair="B", entry_strategy="MANUAL", pnl_percentage=0.3, cell_multiplier_source=""),
                        dict(base, opened_at="2026-10-08T01:06:00", pair="C", entry_strategy="FRENZY_LONG", pnl_percentage=0.3, cell_multiplier_source=""),
                        dict(base, opened_at="2026-10-08T01:07:00", pair="D", entry_strategy="MOMENTUM", pnl_percentage=0.3, cell_multiplier_source="ADX_PROBE"),
                        dict(base, opened_at="2026-10-08T01:08:00", pair="E", entry_strategy="MOMENTUM", pnl_percentage=0.3, cell_multiplier_source="", direction="SHORT"),
                        dict(base, opened_at="2026-10-08T01:09:00", pair="F", entry_strategy="MOMENTUM", pnl_percentage=None, cell_multiplier_source="", status="SIGNAL_EXPIRED")])
    po, pn = tmp_path / "o.csv", tmp_path / "n.csv"
    old.to_csv(po, index=False); new.to_csv(pn, index=False)
    os.utime(po, (1, 1)); os.utime(pn, (2, 2))
    new = pd.concat([new, pd.DataFrame([dict(base, opened_at="2026-10-08T01:10:00", pair="G", entry_strategy="MOMENTUM", pnl_percentage=0.3,
                                              cell_multiplier_source="UNMATCHED", pattern_cell_source="W2_PROBE")])], ignore_index=True)
    new.to_csv(pn, index=False); os.utime(pn, (2, 2))
    o = MC.load_orders([str(po), str(pn)])
    assert list(o.pair) == ["A"] and o.pnl_percentage.iloc[0] == pytest.approx(0.3)   # newest export wins on (opened_at[:19], pair, direction)


def test_store_is_write_once(tmp_path):
    t = MC.tag(_fills())
    s1 = MC.merge_store(pd.DataFrame(), t, "2026-10-08T05:00:00")
    assert len(s1) == 5 and (s1.recorded_at == "2026-10-08T05:00:00").all()
    t2 = t.copy(); t2["pnl_pct"] = 9.9
    s2 = MC.merge_store(s1, t2, "2026-10-09T05:00:00")
    assert len(s2) == 5 and (s2.pnl_pct != 9.9).all() and (s2.recorded_at == "2026-10-08T05:00:00").all()
    extra = t.iloc[:1].assign(key="2026-10-08T09:00:00|Z|LONG", pair="Z")
    s3 = MC.merge_store(s2, extra, "2026-10-09T06:00:00")
    assert len(s3) == 6 and s3.recorded_at.iloc[-1] == "2026-10-09T06:00:00"


def test_run_end_to_end_writes_store_once(tmp_path):
    p = tmp_path / "x.csv"
    _fills().assign(direction="LONG", entry_strategy="MOMENTUM", cell_multiplier_source="UNMATCHED").to_csv(p, index=False)
    store = str(tmp_path / "store.csv")
    L = MC.run(0, paths=[str(p)], store=store)
    assert any("ML_STOP_COOLDOWN" in x for x in L) and any("CLUSTER2_120" in x and "NOT pre-registered" in x for x in L)
    assert any("collecting" in x for x in L) and any("WR ≥ 61 % or Σ > 0" in x for x in L)
    m1 = os.path.getmtime(store); d1 = pd.read_csv(store)
    MC.run(0, paths=[str(p)], store=store)
    assert os.path.getmtime(store) == m1 and len(pd.read_csv(store)) == len(d1) == 5


def test_unreadable_store_moves_to_bad(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("garbage,cols\n1,2\n")
    assert not len(MC.load_store(str(p))) and not p.exists() and list(tmp_path.glob("s.csv.*.bad"))


def test_scout_hook_survives_a_crash(monkeypatch):
    """the opportunity_scout.py ML cooldown block: a run() that raises leaves an 'Unavailable' section and the scout goes on."""
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read().splitlines()
    i = next(n for n, x in enumerate(src) if "import scout_ml_cooldown as _mc" in x)
    s = max(n for n in range(i) if src[n].strip().startswith("try:"))
    e = next(n for n in range(i, len(src)) if "## 🧊 ML cooldown" in src[n])
    block = "\n".join(x[4:] for x in src[s:e + 1])
    fake = types.ModuleType("scout_ml_cooldown")
    fake.run = lambda now_ms: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "scout_ml_cooldown", fake)
    logs = []
    ns = {"os": os, "sys": sys, "_rg_sec": [], "now_ms": 0, "log": logs.append, "__file__": os.path.join(HERE, "scripts", "x.py")}
    exec(block, ns)
    assert ns["_rg_sec"][0] == "## 🧊 ML cooldown" and "boom" in ns["_rg_sec"][2] and logs
