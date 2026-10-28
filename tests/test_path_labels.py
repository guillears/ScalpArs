"""🧭 DECISION_LOG 406 — the analysis methodology tools: path labels + decomposition (scripts/path_labels.py), leave-one-out and the
real-account DCR (scripts/review_math.py), the observation-line exit registry (scripts/observe_registry.py), the sweep's ENTRY-label
target and the checklist items. Pure / synthetic — no network."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import path_labels as PL  # noqa: E402
import review_math as RM  # noqa: E402
import observe_registry as OR  # noqa: E402

MIN = 60_000
T0 = 1_790_000_000_000 - (1_790_000_000_000 % MIN)


def _bars(path_pct, entry=100.0, t0=T0, short=False):
    """per-bar (high %, low %) moves from the entry → a 1m bar frame."""
    rows = []
    for i, (hi, lo) in enumerate(path_pct):
        rows.append(dict(open_time=t0 + i * MIN, o=entry, h=entry * (1 + hi / 100), l=entry * (1 + lo / 100), c=entry))
    return pd.DataFrame(rows)


# ── pure core ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_first_passage_orders_and_ambiguity():
    assert PL.first_passage([0.2, 1.1, 0.0], [-0.5, -0.2, -5.0], 1.0, 4.0) == ("WORKED", 1)
    assert PL.first_passage([0.2, 0.3, 2.0], [-1.0, -4.2, 0.0], 1.0, 4.0) == ("NEVER", 1)
    assert PL.first_passage([1.5], [-4.5], 1.0, 4.0) == ("AMBIG", 0)
    assert PL.first_passage([0.5, 0.9], [-1.0, -3.9], 1.0, 4.0) == ("FLAT", None)


def test_fav_adv_is_direction_aware():
    f, a = PL.fav_adv(None, [101.0], [98.0], 100.0, "LONG")
    assert f[0] == pytest.approx(1.0) and a[0] == pytest.approx(-2.0)
    f, a = PL.fav_adv(None, [101.0], [98.0], 100.0, "SHORT")
    assert f[0] == pytest.approx(2.0) and a[0] == pytest.approx(-1.0)


def test_groups():
    assert PL.group("NEVER", -4.0, 0.2, False) == "NEVER_WORKED"
    assert PL.group("NEVER", -4.0, 0.2, True) == "STOPPED_RECOVERED"
    assert PL.group("WORKED", -4.0, 3.3, False) == "GAVE_BACK"
    assert PL.group("FLAT", -1.0, 0.5, False) == "DRIFT_LOSS"
    assert PL.group("WORKED", 1.25, 21.9, False) == "LEFT_ON_TABLE"        # HUMA 10-26: +1.25 booked, +21.9 there
    assert PL.group("WORKED", 1.25, 1.9, False) == "CAPTURED"
    assert PL.group("WORKED", 3.0, 5.5, False) == "CAPTURED"
    assert PL.group("AMBIG", -4.0, None, None) == "AMBIG"
    assert PL.next_study(-900, -600) == "ENTRY first" and PL.next_study(-100, -600) == "EXIT first" and PL.next_study(0, 0) == "—"


def test_stop_table_config_first_losing_stops_only(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text('{"thresholds": {"frenzy_willy_stop_pct": 4.0, "flatbase_stop_pct": 15.0, "spike_fade_sl_pct": -1.5}}')
    m = tmp_path / "m.csv"
    pd.DataFrame(dict(entry_strategy=["SURGE_LONG"] * 5 + ["MOMENTUM"] * 4, opened_at=[f"2026-10-0{i + 1}T00:00:00" for i in range(9)],
                      pair="AUSDT", direction="LONG",
                      close_reason=["BR_TRAILING_STOP"] * 3 + ["BR_STOP_LOSS"] * 2 + ["STOP_LOSS"] * 3 + ["TRAILING_STOP L2"],
                      pnl_percentage=[0.8, 1.2, 0.9, -1.25, -1.3, -0.7, -0.75, -0.72, 0.55], status="CLOSED")).to_csv(m, index=False)
    PL._stop_table.cache_clear()
    t = PL._stop_table(master=str(m), cfg=str(cfg))
    assert t["FRENZY_WILLY_HT"][0] == PL._yround(4.0 - PL.FEE_RT) == 3.9 and t["FLATBASE_SHORT"][0] == PL._yround(15 - PL.FEE_RT)
    assert t["SPIKE_FADE"][0] == PL._yround(1.5 - PL.FEE_RT)
    assert t["MOMENTUM"][0] == PL._yround(0.72 - PL.FEE_RT)              # 3 losing STOP_LOSS, the winning TRAILING_STOP ignored
    assert "SURGE_LONG" not in t                                          # only 2 LOSING stops (trailing winners never count) → default
    assert PL.stop_y("SURGE_LONG", t) == (PL.DEFAULT_Y, "default")
    assert PL.stop_y("FRENZY_WILLY", t)[1].startswith("config")


def _facts(path, te_off=5_000, tc_min=3, booked=-4.0, y=4.0, short=False, **kw):
    te, tc = T0 + te_off, T0 + tc_min * MIN + te_off
    return PL.label_from_bars(_bars(path), te, tc, 100.0, "SHORT" if short else "LONG", booked, 1.0, y, now_ms=T0 + 10 ** 12, **kw)


def test_never_worked_recovered_pending_and_reach():
    path = [(0.1, -1.0), (0.3, -2.0), (0.0, -4.2)] + [(0.0, -4.0)] * (PL.HORIZON_MIN + 200)
    f = _facts(path, stopped=True, peak=0.2, trough=-4.0)
    assert f["label"] == "NEVER" and f["reach"] == pytest.approx(0.3) and PL.finish(f, -4.0)["group"] == "NEVER_WORKED"
    rec = path[:3] + [(0.0, -4.0)] * 30 + [(1.5, -4.0)] + [(0.0, -4.0)] * (PL.HORIZON_MIN + 200)
    f2 = _facts(rec, stopped=True, peak=0.2, trough=-4.0)
    assert PL.finish(f2, -4.0)["group"] == "STOPPED_RECOVERED"
    f3 = _facts(path[:5], stopped=True, peak=0.2, trough=-4.0)
    assert f3["label"] == "NEVER" and PL.finish(f3, -4.0)["group"] == "PENDING"     # decided label, after-close window missing
    hole = _bars([(0.1, -0.1)] * 60).drop(index=[2])
    assert PL.label_from_bars(hole, T0 + 5_000, T0 + 3 * MIN, 100.0, "LONG", 0.5, 1.0, 4.0, peak=0.5, trough=-0.2)["label"] == "PENDING"


def test_live_stop_counts_before_a_later_passage():                       # review H3
    path = [(0.2, -0.5)] * 5 + [(0.1, -0.6)] * 100 + [(1.5, 0.0)] * (PL.HORIZON_MIN + 200)   # the 1m low never reaches −Y …
    f = _facts(path, tc_min=4, booked=-0.7, y=4.0, stopped=True, peak=0.2, trough=-0.7)    # … but the fill's own stop closed it
    assert f["label"] == "NEVER" and PL.finish(f, -0.7)["group"] in ("NEVER_WORKED", "STOPPED_RECOVERED")
    g = _facts(path, tc_min=4, booked=-0.7, y=4.0, stopped=False, peak=0.2, trough=-0.7)    # not a stop close → WORKED later counts
    assert g["label"] == "WORKED"


def test_entry_bar_bounded_by_stamps_or_ambig():                          # review H2
    path = [(1.5, -0.2)] + [(0.2, -0.3)] * 30 + [(0.0, -1.6)] + [(0.0, 0.0)] * (PL.HORIZON_MIN + 200)
    f = _facts(path, tc_min=31, booked=-1.5, y=1.5, stopped=True, peak=0.10, trough=-1.5)    # pre-fill spike print in bar 0
    assert f["label"] == "NEVER"                                          # FORM 08-14 class: no longer a false WORKED / GAVE_BACK
    assert f["best"] is not None and f["best"] < 1.0                      # best not inflated by the pre-fill print
    u = _facts(path, tc_min=31, booked=-1.5, y=1.5, stopped=True)         # no stamps → the entry minute cannot decide
    assert u["label"] in ("AMBIG", "NEVER")
    v = _facts([(1.5, -0.2)] + [(0.2, -0.3)] * (PL.HORIZON_MIN + 300), tc_min=31, booked=0.3, y=1.5)
    assert v["label"] == "AMBIG"


def test_horizon_limits_the_passage():                                    # review: a long hold never counts a move after 24 h
    path = [(0.2, -0.3)] * (PL.HORIZON_MIN + 30) + [(1.5, -0.3)] * 500
    f = _facts(path, tc_min=PL.HORIZON_MIN + 200, booked=1.2, y=15.0, peak=1.5, trough=-0.3)
    assert f["label"] == "FLAT" and f["pass_min"] is None


def test_short_never_and_winner_left_on_table():
    path = [(0.5, -0.5), (0.2, -1.2)] + [(0.0, -6.0)] * 60 + [(0.0, 0.0)] * (PL.HORIZON_MIN + 10)   # price falls 6 % → short wins
    f = _facts(path, tc_min=10, booked=1.5, y=3.0, short=True, peak=1.5, trough=-0.5)
    fin = PL.finish(f, 1.5)
    assert f["label"] == "WORKED" and fin["group"] == "LEFT_ON_TABLE" and f["best"] == pytest.approx(6.0, abs=0.01)
    up = [(0.0, 0.0), (5.0, 0.0)] + [(0.0, 0.0)] * (PL.HORIZON_MIN + 200)          # price rises 5 % → short stopped
    g = _facts(up, tc_min=2, booked=-3.0, y=2.91, short=True, stopped=True, peak=0.0, trough=-3.0)
    assert g["label"] == "NEVER"


def test_finish_recomputes_group_from_the_current_booked():               # review M1: the cache holds path facts only
    facts = dict(label="WORKED", best=3.0, recovered=False, reach=3.0, pass_min=5.0)
    assert PL.finish(facts, 2.8)["group"] == "CAPTURED" and PL.finish(facts, -1.0)["group"] == "GAVE_BACK"
    assert PL.finish(facts, 0.5)["group"] == "LEFT_ON_TABLE" and PL.finish(facts, 0.5)["left"] == pytest.approx(3.0 - PL.FEE_RT - 0.5)


def _fill(pair="AUSDT", booked=-4.0, cr="STOP_LOSS", es="FRENZY_WILLY", peak=0.2, trough=-4.0):
    ts = pd.Timestamp(T0 + 5_000, unit="ms")
    return dict(opened_at=ts.isoformat(), closed_at=(ts + pd.Timedelta(minutes=3)).isoformat(), pair=pair, direction="LONG",
                entry_price=100.0, pnl_percentage=booked, pnl=booked * 75, notional_value=7500, entry_strategy=es, close_reason=cr,
                peak_pnl=peak, trough_pnl=trough)


def test_label_frame_cache_facts_only_and_never_raises(monkeypatch, tmp_path):
    import scout_willy_timecap as WT
    monkeypatch.setattr(PL, "CACHE", str(tmp_path / "c.json"))
    monkeypatch.setattr(PL, "stop_y", lambda s, table=None: (3.91, "config"))
    path = [(0.1, -1.0), (0.2, -2.0), (0.0, -4.2)] + [(0.0, -4.0)] * (PL.HORIZON_MIN + 200)
    monkeypatch.setattr(WT, "load_1m", lambda pair: _bars(path) if pair == "AUSDT" else pd.DataFrame(columns=["open_time", "h", "l"]))
    f = pd.DataFrame([_fill(), dict(_fill("BUSDT"), entry_price="bad")])
    notes = []
    lab = PL.label_frame(f, fetch=False, notes=notes)
    assert list(lab.group) == ["NEVER_WORKED", "PENDING"] and any("BUSDT" in n for n in notes)
    c = json.load(open(PL.CACHE))
    assert all(set(v) == {"label", "pass_min", "reach", "best", "recovered"} for k, v in c.items() if not k.startswith("_"))
    monkeypatch.setattr(WT, "load_1m", lambda pair: (_ for _ in ()).throw(AssertionError("cache must be used")))
    assert PL.label_frame(f.iloc[:1], fetch=False).group.iloc[0] == "NEVER_WORKED"
    assert list(PL.label_frame(f.iloc[:0]).columns)[-len(PL.PATH_COLS):] == list(PL.PATH_COLS)     # empty → the columns exist
    L = "\n".join(PL.report_lines(lab))
    assert "NEVER_WORKED" in L and "| FRENZY_WILLY | 2 |" in L and "reach median" in L


def test_fetch_budget_stops_and_never_fetches_the_future(monkeypatch, tmp_path):
    import scout_willy_timecap as WT
    monkeypatch.setattr(PL, "CACHE", str(tmp_path / "c.json"))
    monkeypatch.setattr(PL, "stop_y", lambda s, table=None: (3.91, "config"))
    monkeypatch.setattr(WT, "load_1m", lambda pair: pd.DataFrame(columns=["open_time", "h", "l"]))
    monkeypatch.setattr(WT.NF, "_cooldown_until", lambda: 0)
    monkeypatch.setattr(WT, "_append_1m", lambda pair, d: None)
    calls = []

    def boom(pair, start, limit, now):
        calls.append((pair, start, limit))
        raise TimeoutError("slow")
    monkeypatch.setattr(WT, "_fetch_1m", boom)
    f = pd.DataFrame([_fill(f"P{i}USDT") for i in range(5)])
    notes = []
    bud = {}
    PL.label_frame(f, fetch=True, notes=notes, budget=bud, now_ms=T0 + 10 ** 12)
    assert len(calls) == 1 and "network error" in bud["stopped"]               # the first error stops the run (review M3)
    assert calls[0][2] == PL.KLINE_LIMIT
    calls.clear()
    monkeypatch.setattr(WT, "_fetch_1m", lambda *a: calls.append(a) or (pd.DataFrame(), 0))
    bud2 = {}
    PL.label_frame(f.iloc[:1], fetch=True, budget=bud2, now_ms=T0 + 10 ** 12)
    assert "no X-MBX-USED-WEIGHT-1M header" in bud2["stopped"]
    calls.clear()
    monkeypatch.setattr(WT, "_fetch_1m", lambda *a: calls.append(a) or (pd.DataFrame(), 5))
    now = T0 + 30 * MIN                                                       # the fill's windows are mostly in the future
    PL.label_frame(pd.DataFrame([_fill("QUSDT")]), fetch=True, budget={}, now_ms=now)
    assert calls and all(c[1] < now for c in calls)


# ── review math ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_loo_flags_one_fill_results():
    r = RM.loo([3.89, 1.39, -4.37, 2.42], ["CFX", "CAP", "BTW", "ALT"])
    assert r["top_label"] == "CFX" and r["without_top"] == pytest.approx(3.33 - 3.89) and r["top_share"] > 1
    t = RM.loo_text([3.89, 1.39, -4.37, 2.42], ["CFX", "CAP", "BTW", "ALT"])
    assert "without the biggest (CFX +3.89" in t and "⚠ one fill" in t
    assert "⚠" not in RM.loo_text([1.85, 2.48, 2.06, 2.15], list("ABCD"))
    assert "no changed fill" in RM.loo_text([0, 0])
    assert RM.loo([-3.0, 1.0], ["A", "B"])["top_label"] == "A"            # a negative total → its biggest negative contributor


def test_dcr_account_real_path_with_flow():
    f = pd.DataFrame(dict(closed_at=["2026-10-24T10:00", "2026-10-26T10:00", "2026-10-26T15:00"], usd=[290.0, -100.0, 500.0],
                          s=[True, False, True]))
    got = RM.dcr_account(f, f.s, "usd", 2900, "2026-10-24", "2026-10-26", [("2026-10-26T13:50", 7100)])
    d1 = 290 / 2900                                                          # day 1: +10 %
    pre = 0 / 3190                                                           # day 3 before the flow: only the other sleeve's −100
    post = 500 / (3190 - 100 + 7100)
    exp = ((1 + d1) * 1.0 * (1 + pre) * (1 + post)) ** (1 / 3) - 1
    assert got == pytest.approx(exp * 100, rel=1e-9)
    assert RM.fill_usd(-4.0, 7770) == pytest.approx(-310.8)


# ── observation-line registry ───────────────────────────────────────────────────────────────────────────────────────────────────
def test_every_scout_file_is_classified():
    files = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "scripts", "scout_*.py"))}
    assert files - OR.classified() == set(), "classify the new scout file in scripts/observe_registry.py (EXIT / PROXY / BOOKED / NONE)"
    assert not (set(OR.EXIT) & set(OR.BOOKED)) and not (set(OR.EXIT) & OR.NONE)


def test_registry_flags_stale_lines():
    th = dict(frenzy_willy_high_target_enabled=True, frenzy_long_stop_pct=4.0, frenzy_lite_c_enabled=True, frenzy_willy_tp_pct=1.25,
              frenzy_willy_high_k_pct=1.0, frenzy_willy_high_tp_max_pct=8.0, frenzy_willy_stop_pct=4.0, frenzy_willy_late_tp_minutes=30.0,
              frenzy_willy_late_tp_pct=0.25)
    st = {r[0]: r for r in OR.status(th)}
    assert st["scout_willy_timecap.py"][4] == "STALE" and st["scout_willy_exit_v2.py"][4] == "OK"
    assert st["scout_frenzy.py"][5] == [("frenzy_long_stop_pct", 3.0, 4.0)]
    assert st["scout_bearrun_v2.py"][4] == "PROXY" and st["scout_willy_ht.py"][4] == "STALE" and st["scout_ml_turn.py"][4] == "OK"
    old = dict(th, frenzy_willy_high_target_enabled=False)
    assert {r[0]: r for r in OR.status(old)}["scout_willy_timecap.py"][4] == "OK"   # back on the old exit → its lines read again
    L = "\n".join(OR.run(th=th))
    assert "⚠ scout_willy_timecap.py" in L and "scout_willy_exit_v2.py" not in L


# ── sweep target + checklist + scout wiring ─────────────────────────────────────────────────────────────────────────────────────
def test_sweep_path_and_reach_targets(monkeypatch):
    import sweep_separators as SW
    SW._LABEL_MEMO.clear()
    df = pd.DataFrame(dict(opened_at=[f"2026-10-2{i}T00:00:00" for i in range(4)], pair="AUSDT", direction="LONG",
                           pnl_percentage=[1.0, -4.0, 2.0, -1.0], entry_x=[1, 2, 3, 4]))
    lab = df.assign(path_label=["WORKED", "NEVER", "FLAT", "WORKED"], reach=[3.0, 0.2, 0.5, None])

    def fake(d, label="path", **k):
        return lab, (lab.reach.where(~lab.path_label.isin(("AMBIG", "PENDING"))) if label == "reach"
                     else lab.path_label.map({"WORKED": 1.0, "NEVER": 0.0}))
    monkeypatch.setattr(PL, "entry_target", fake)
    out, note = SW.apply_label(df, "path")
    assert list(out.pnl_percentage) == [1.0, 0.0, 1.0] and "WORKED 2 · NEVER 1 · not counted 1" in note
    out2, note2 = SW.apply_label(df, "reach")
    assert list(out2.pnl_percentage) == [3.0, 0.2, 0.5] and "REACH" in note2
    same, n2 = SW.apply_label(df, "pnl")
    assert same is df and n2 == ""
    with pytest.raises(ValueError):
        SW.apply_label(df, "bogus")


def test_methodology_wired_everywhere():
    ck = open(os.path.join(ROOT, "scripts", "batch_review_checklist.py")).read()
    for k in ("LOSER / WINNER DECOMPOSITION", "--label path", "JOINT CHECK", "STALE LINES", "LEAVE-ONE-OUT", "REAL sizing"):
        assert k in ck, k
    opp = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "import observe_registry" in opp and "import scout_path_decomposition" in opp
    assert opp.index("import scout_path_decomposition") < opp.index("import scout_sweep_all")      # labels before the sweep
    assert "RM.loo_text" in open(os.path.join(ROOT, "scripts", "scout_willy_exit_v2.py")).read()
    cm = open(os.path.join(ROOT, "CLAUDE.md")).read()
    assert "ENTRY × EXIT" in cm and "DECISION_LOG (406)" in cm


def test_registry_missing_key_and_era_booked_lines():
    st = {r[0]: r for r in OR.status(dict(frenzy_long_stop_pct=4.0, frenzy_lite_c_enabled=True))}
    assert st["scout_frenzy_entry_lines.py"][4] == "STALE" and st["scout_frenzy_wl_lines.py"][4] == "STALE"
    assert ("frenzy_willy_high_target_enabled", False, "MISSING from config") in st["scout_willy_timecap.py"][5]
    assert OR.diffs({"k": 1.0}, {"k": "abc"}) == [("k", 1.0, "abc")]                 # a non-numeric value never crashes the section


def test_review_math_guards():
    with pytest.raises(ValueError):
        RM.loo([1.0, 2.0], ["A"])
    assert "1 unpriced" in RM.loo_text([1.0, None, 2.0], ["A", "B", "C"])
    f = pd.DataFrame(dict(closed_at=["2026-10-24T10:00:00+00:00", "2026-10-26T15:00"], usd=[290.0, 500.0], s=[True, True]))
    assert RM.dcr_account(f, f.s, "usd", 2900, "2026-10-24", "2026-10-26", [("2026-10-26T13:50", 7100)]) > 0
    with pytest.raises(ValueError):
        RM.dcr_account(f.assign(closed_at=[None, "2026-10-26T15:00"]), f.s, "usd", 2900, "2026-10-24", "2026-10-26")


def test_round2_stop_window_horizon_bar0_gap_and_winner_group():
    path = [(0.2, -0.3)] * 10
    f = _facts(path, tc_min=60, booked=-4.0, y=3.9, stopped=True, peak=0.2, trough=-4.0)
    assert f["label"] == "PENDING"                                       # bars end before the stop close → cannot place it yet
    long_ = [(0.2, -0.3)] * (PL.HORIZON_MIN + 300)
    g = _facts(long_, tc_min=PL.HORIZON_MIN + 200, booked=-3.0, y=14.9, stopped=True, peak=0.2, trough=-3.0)
    assert g["label"] == "FLAT"                                          # a stop after 24 h is not the entry's −Y passage
    spike = [(1.5, -0.2)] + [(0.3, -1.0)] * 3 + [(0.1, -2.0)] + [(1.6, 0.0)] * (PL.HORIZON_MIN + 200)
    h = _facts(spike, tc_min=PL.HORIZON_MIN + 100, booked=1.0, y=1.9, peak=1.6, trough=-2.0)
    assert h["label"] == "AMBIG"                                         # bar 0 says WORKED, bars 1+ say NEVER → unknown
    assert PL.finish(dict(h, best=1.7), 1.0)["group"] == "CAPTURED"      # a winner's group needs only the exit facts
    holed = _bars([(0.1, -0.1)] * 6 + [(1.5, -0.1)] * (PL.HORIZON_MIN + 300)).drop(index=[3, 4])
    k = PL.label_from_bars(holed, T0 + 5_000, T0 + 30 * MIN, 100.0, "LONG", 0.8, 1.0, 3.9, now_ms=T0 + 10 ** 12, peak=1.5, trough=-0.1)
    assert k["label"] == "WORKED"                                        # a 2-minute exchange hole never ends the path


def test_round2_stop_table_dedups_export_snapshots(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text('{"thresholds": {}}')
    one = pd.DataFrame(dict(entry_strategy="MOMENTUM", close_reason="STOP_LOSS", pnl_percentage=[-0.70, -0.71, -0.72],
                            opened_at=["2026-10-01T00:00:00", "2026-10-02T00:00:00", "2026-10-03T00:00:00"], pair="AUSDT", direction="LONG",
                            status="CLOSED"))
    m = tmp_path / "m.csv"
    one.to_csv(m, index=False)
    snaps = []
    for i in range(3):                                                   # the same 3 fills in three export snapshots + one extra stop
        p = tmp_path / f"e{i}.csv"
        pd.concat([one, one.iloc[:1].assign(opened_at=f"2026-10-1{i}T00:00:00", pnl_percentage=-1.5)]).to_csv(p, index=False)
        snaps.append(str(p))
    PL._stop_table.cache_clear()
    t = PL._stop_table(master=str(m), cfg=str(cfg), exports_key=tuple(snaps))
    assert t["MOMENTUM"][1] == "its 6 losing stops"                      # 3 + 3 distinct extra fills, never 12


def test_round2b_stamps_settle_the_entry_minute_and_family_y():
    # FLATBASE ORDER 10-24 class: stopped at −15, its own peak +1.36 net → it saw +1 before the stop → WORKED (was AMBIG)
    path = [(1.45, -0.5)] + [(0.3, -3.0)] * 10 + [(0.0, -15.2)] + [(0.0, 0.0)] * (PL.HORIZON_MIN + 200)
    f = _facts(path, tc_min=11, booked=-15.1, y=14.9, short=False, stopped=True, peak=1.36, trough=-15.1)
    assert f["label"] == "WORKED"
    # VANRY class: a 47 s winner, peak +4.83, trough −0.09 → WORKED
    g = _facts([(4.9, -0.1)] + [(0.5, -2.0)] * (PL.HORIZON_MIN + 200), tc_min=1, booked=4.7, y=1.4, peak=4.83, trough=-0.09)
    assert g["label"] == "WORKED"
    t = {"MOMENTUM": (0.6, "its 159 losing stops")}
    assert PL.stop_y("BEARRUN_SHORT", t) == (0.6, "its 159 losing stops (momentum stack)")
    holed = _bars([(0.1, -0.1)] * 3 + [(0.1, -0.1)] * 5 + [(1.5, -0.1)] * (PL.HORIZON_MIN + 300)).drop(index=[3, 4, 5])
    k = PL.label_from_bars(holed, T0 + 5_000, T0 + 30 * MIN, 100.0, "LONG", 0.8, 1.0, 3.9, now_ms=T0 + 10 ** 12, peak=1.5, trough=-0.1)
    assert k["label"] == "WORKED" and k["pass_min"] == pytest.approx(7.9, abs=0.1)     # real bar time after the hole (F4)
