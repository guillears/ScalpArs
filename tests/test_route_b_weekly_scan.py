"""Route B weekly scan runner (PREREG §6–§7 + Addendum 10): cohorts, the Q1 test family, the clustered t, the rotation null, the
A / B gates, Q2 / Q3 / Q4, read-once freezing, the phase guard and an end-to-end DESIGN → HOLDOUT run — synthetic data only."""
import json
import math
import re
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "route_b_signals"))
import weekly_scan as W  # noqa: E402
import interim_q3 as I  # noqa: E402

UTC = "UTC"


def _sig(n, start, sleeve="EXHAUSTION_SHORT", per_day=6, status="BLOCKED:EXH_SPOT_SHARE", cls="filter", seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start, tz=UTC)
    ts = [t0 + pd.Timedelta(days=i // per_day, minutes=37 * (i % per_day)) for i in range(n)]
    d = pd.DataFrame({"sleeve": sleeve, "pair": [f"C{i % 7}USDT" for i in range(n)], "direction": "SHORT", "signal_time": ts,
                      "status": status, "poss_dup": False, "core_rule_met": True, "block_class": cls})
    d["window15"] = d.signal_time.dt.floor("15min").dt.strftime("%Y-%m-%dT%H:%M")
    d["signal_id"] = d.sleeve + "|" + d.pair + "|" + d.signal_time.astype(str)
    d["rb_x"] = rng.normal(size=n)
    d["rb_noise"] = rng.normal(size=n)
    d["out_lbl_worked"] = (rng.random(n) < 0.5).astype(float)
    d["out_net_pct"] = np.where(d.out_lbl_worked == 1, 1.0, -1.0)
    d["btc_d_trend"] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    d["rb_has_book"] = True
    return d


# ── cohorts ──
def test_sleeve_cohort_keeps_only_section2_rows():
    d = _sig(8, "2026-11-20")
    d.loc[0, "status"] = "BLOCKED:EXH_NO_SPOT"            # §2 core gate (the loader files it as filter)
    d.loc[1, "poss_dup"] = True
    d.loc[2, "core_rule_met"] = False
    d.loc[3, "block_class"] = "data"
    d.loc[4, ["status", "block_class"]] = ["OPENED", "taken"]
    d.loc[5, ["status", "block_class"]] = ["BLOCKED:EXH_PAIR_HELD", "capacity"]
    d.loc[6, "status"] = "REFUSED:SKIPPED_CONFIRM"
    got = W.sleeve_cohort(d, "EXHAUSTION_SHORT")
    assert sorted(got.signal_id) == sorted(d.signal_id[[4, 5, 7]])
    assert {"y", "net", "day", "win"} <= set(got.columns)


def test_s5_cohort_from_dl630_only():
    d = _sig(4, "2026-11-23 14:00", status="BLOCKED:EXH_NO_SPOT")
    d["s5_lbl_worked"] = [1.0, 0.0, 1.0, math.nan]
    got = W.s5_cohort(d)
    assert (got.signal_time >= W.DL630_START).all() and len(got) == 3
    assert got.y.isna().sum() == 1


def test_guard_and_eras():
    d = W._prep(_sig(10, "2026-12-06", per_day=4), "out_lbl_worked")
    cut = pd.Timestamp("2026-12-08 12:00", tz=UTC)
    g = W.guard(d, cut, 24)
    assert (g.signal_time <= cut - pd.Timedelta(hours=24)).all() and len(g) < len(d)
    assert (W.design(d).signal_time < W.HOLDOUT_START).all() and (W.holdout(d).signal_time >= W.HOLDOUT_START).all()
    assert len(W.design(d)) + len(W.holdout(d)) == len(d)


# ── the Q1 family ──
def test_screen_vars_prefix_exclusions_and_constants():
    d = pd.DataFrame({"rb_a": [1.0, 2.0], "rb_a_lb": [1, 0], "rb_mid": [1.0, 2.0], "fb_c": [3.0, 3.0], "mk_m": [1.0, 2.0],
                      "j_s": ["x", "y"], "oi_chg_5m": [np.nan, np.nan], "journal_atr_pct": [1.0, 3.0], "q2_S1a": [1.0, 0.0]})
    v, skip = W.screen_vars(d)
    assert v == ["rb_a", "journal_atr_pct"]
    assert skip == {"fb_c": "constant", "j_s": "never scored / not numeric", "oi_chg_5m": "never scored / not numeric"}


def test_sides_and_make_tests_cuts():
    D = pd.DataFrame({"rb_a": np.arange(-4, 6, dtype=float), "rb_b": np.r_[np.arange(9, dtype=float), np.nan]})
    defs, A, B = W.make_tests(D, ["rb_a", "rb_b"])
    kinds = [(x["kind"], x.get("gran")) for x in defs]
    assert kinds.count(("1d", "sign")) == 2                  # rb_b's 0 is non-positive → it has a sign test too
    sgn = defs.index(dict(kind="1d", var="rb_a", gran="sign"))
    assert A[:, sgn].sum() == 5 and B[:, sgn].sum() == 5      # x > 0 vs x ≤ 0
    q = next(i for i, x in enumerate(defs) if x.get("gran") == "quintile" and x["var"] == "rb_a")
    assert A[:, q].sum() == 2 and B[:, q].sum() == 2          # top / bottom fifth, the middle in neither
    quads = [i for i, x in enumerate(defs) if x["kind"] == "2d"]
    assert len(quads) == 4
    tot = sum(A[:, i] for i in quads)
    assert (tot[:9] == 1).all() and tot[9] == 0               # the 4 quadrants partition the rows scored on both
    for i in quads:
        assert ((A[:, i] | B[:, i]) == ~np.isnan(D.rb_b.to_numpy())).all() and not (A[:, i] & B[:, i]).any()


def test_rb_b_has_no_sign_test_when_never_positive_split():
    D = pd.DataFrame({"rb_p": np.arange(1, 11, dtype=float)})   # all > 0 → no sign test
    defs, _, _ = W.make_tests(D, ["rb_p"])
    assert not any(x.get("gran") == "sign" for x in defs)


def test_floor_mask_counts_rows_and_days():
    days = np.array(["d%d" % (i % 5) for i in range(20)])
    a = np.zeros(20, bool)
    a[:10] = True
    b = ~a
    assert W.floor_mask(a[:, None], b[:, None], days)[0]
    assert not W.floor_mask(a[:, None], b[:, None], days, n_min=11)[0]
    days2 = days.copy()
    days2[:10] = "same"
    assert not W.floor_mask(a[:, None], b[:, None], days2)[0]            # A on one day only


def test_cluster_t_matches_manual_and_clusters_matter():
    rng = np.random.default_rng(1)
    n = 40
    y = (rng.random(n) < 0.5).astype(float)
    a = np.arange(n) < 20
    b = ~a
    eff, t = W.cluster_t(y, a[:, None], b[:, None], W.onehot(np.arange(n)))
    u = np.where(a, (y - y[a].mean()) / a.sum(), -(y - y[b].mean()) / b.sum())
    se = math.sqrt((u ** 2).sum() * n / (n - 1))
    assert eff[0] == pytest.approx(y[a].mean() - y[b].mean()) and t[0] == pytest.approx(eff[0] / se)
    # duplicating every row into the same cluster keeps the effect but the clustered SE grows (≈ √2 smaller t)
    y2, a2 = np.r_[y, y], np.r_[a, a]
    _, t2 = W.cluster_t(y2, a2[:, None], (~a2)[:, None], W.onehot(np.r_[np.arange(n), np.arange(n)]))
    assert abs(t2[0]) == pytest.approx(abs(t[0]), rel=0.05)


def test_cluster_t_zero_se_is_nan():
    y = np.r_[np.ones(10), np.zeros(10)]
    a = np.arange(20) < 10
    _, t = W.cluster_t(y, a[:, None], (~a)[:, None], W.onehot(np.arange(20)))
    assert np.isnan(t[0])


def test_rotation_null_offset_and_short_cohort():
    days = np.array(["a"] * 3 + ["b"] * 5 + ["c"] * 2)
    assert W.rotation_offset_min(days) == 5
    y = np.arange(10, dtype=float) % 2
    a = (np.arange(10) % 3 == 0)
    assert W.rotation_null(y, a[:, None], (~a)[:, None], W.onehot(np.arange(10)), days, np.random.default_rng(0), 5) is None
    days = np.array([f"d{i // 4}" for i in range(40)])
    y = (np.arange(40) % 3 == 0).astype(float)
    a = np.arange(40) % 2 == 0
    r1 = W.rotation_null(y, a[:, None], (~a)[:, None], W.onehot(np.arange(40)), days, np.random.default_rng(7), 20)
    r2 = W.rotation_null(y, a[:, None], (~a)[:, None], W.onehot(np.arange(40)), days, np.random.default_rng(7), 20)
    assert r1 is not None and len(r1) == 20 and np.array_equal(r1, r2)


def _planted(n=240, start="2026-11-20", effect=True, seed=3):
    d = _sig(n, start, per_day=14, seed=seed)
    rng = np.random.default_rng(seed + 1)
    if effect:
        p = np.where(d.rb_x > 0, 0.9, 0.2)
        d["out_lbl_worked"] = (rng.random(n) < p).astype(float)
        d["out_net_pct"] = np.where(d.out_lbl_worked == 1, 1.0, -1.2)
    return W._prep(d, "out_lbl_worked")


def test_design_q1_finds_a_planted_separator_and_its_definition_reapplies():
    d = _planted()
    out = W.design_q1(d, "EXHAUSTION_SHORT", np.random.default_rng(W.SEED), n_rot=60)
    assert out["bar"] is not None and out["n_family"] > 0
    lines = {s["line"] for s in out["survivors"]}
    assert "Q1|EXH|rb_x|sign" in lines
    s = next(s for s in out["survivors"] if s["line"] == "Q1|EXH|rb_x|sign")
    assert s["eff"] > 0 and s["halves_ok"] and s["net_ok"] and s["status"] == "SURVIVED DESIGN"
    assert not any(x["var"] == "rb_noise" and x["kind"] == "1d" for x in out["survivors"])
    json.dumps(W._jsonable(out))                                          # SURVIVORS.json-serialisable


def test_design_q1_too_few_rows():
    out = W.design_q1(_planted(n=12), "EXHAUSTION_SHORT", np.random.default_rng(0), n_rot=10)
    assert out["survivors"] == [] and "note" in out


def test_holdout_only_flag():
    assert W.is_holdout_only(dict(var="rb_x", var2="rb_imb_05")) and not W.is_holdout_only(dict(var="rb_x"))
    assert W.is_dl611(dict(var="rb_x", var2="oi_chg_15m")) and not W.is_dl611(dict(var="rb_x"))


# ── bootstrap / B gates ──
def test_boot_eff_and_loo():
    y = np.r_[np.ones(30), np.zeros(30)]
    a = np.r_[np.ones(30, bool), np.zeros(30, bool)]
    cl = np.arange(60) // 2
    lo, hi = W.ci95(W.boot_eff(y, a, ~a, cl, np.random.default_rng(0), 500))
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)
    pairs = np.array(["P"] * 30 + ["Q"] * 30)
    assert W.loo_min_ratio(y, a, ~a, pairs, 1.0) == 0.0          # dropping P empties side A → it carries the whole effect
    pairs = np.array([f"P{i % 6}" for i in range(60)])
    assert W.loo_min_ratio(y, a, ~a, pairs, 1.0) == pytest.approx(1.0)


def _hold(n=200, eff=True, seed=11):
    d = _planted(n=n, start="2026-12-07 00:30", effect=eff, seed=seed)
    return W.labelled(d)


def test_b_gate_collecting_pass_and_fail():
    H = _hold()
    a, b = (H.rb_x > 0).to_numpy(), (H.rb_x <= 0).to_numpy()
    rng = np.random.default_rng(0)
    small = H.iloc[:20]
    assert W.b_gate(small, a[:20], b[:20], 0.5, 1, rng)["status"] == "COLLECTING"
    r = W.b_gate(H, a, b, 0.6, 1, rng)
    assert r["status"] == "PASSED" and r["regime"] == ""
    assert W.b_gate(H, a, b, 0.6, -1, rng)["status"] == "CONTRADICTED"                 # the other way (§3 / §6 D)
    assert "< 50 % of DESIGN" in W.b_gate(H, a, b, 3.0, 1, rng)["why"]
    assert W.b_gate(H, a, b, 3.0, 1, rng, ratio_to_design=False)["status"] == "PASSED"  # HOLDOUT-only: sign only
    H1 = H.assign(btc_d_trend=1.0)
    assert W.b_gate(H1, a, b, 0.6, 1, rng)["regime"] == "one regime"                    # B4
    Hn = _hold(eff=False)
    an, bn = (Hn.rb_x > 0).to_numpy(), (Hn.rb_x <= 0).to_numpy()
    assert W.b_gate(Hn, an, bn, 0.6, 1, rng)["status"] in ("FAILED", "CONTRADICTED")


def test_b_gate_concentration_one_pair():
    H = _hold()
    H["pair"] = np.where(H.rb_x > 0, "ONLY", H.pair)            # every A row on one pair → it carries the whole effect
    a, b = (H.rb_x > 0).to_numpy(), (H.rb_x <= 0).to_numpy()
    r = W.b_gate(H, a, b, 0.6, 1, np.random.default_rng(0))
    assert r["status"] == "FAILED" and "one pair / day carries ≥ 50 %" in r["why"]


# ── Q2 ──
def _q2_frame(n=200, start="2026-11-20", line="S1a", good=True, seed=5):
    d = _sig(n, start, per_day=14, seed=seed)
    rng = np.random.default_rng(seed)
    v = (rng.random(n) < 0.5).astype(float)
    d[f"q2_{line}"] = v
    p = np.where(v == 1, 0.2, 0.8) if good else np.where(v == 1, 0.8, 0.2)    # S1a is "−": firing → fewer WORKED
    d["out_lbl_worked"] = (rng.random(n) < p).astype(float)
    d["out_net_pct"] = np.where(d.out_lbl_worked == 1, 1.0, -1.0)
    return W._prep(d, "out_lbl_worked")


def _cohorts(exh):
    empty = exh.iloc[:0]
    return {"EXHAUSTION_SHORT": exh, "DISTRIBUTION_SHORT": empty.assign(rb_funding_pct=[], global_ls_acct=[]),
            "FLATBASE_SHORT": empty, "FRENZY_WILLY_HT": empty}


def test_design_q2_survives_contradicts_and_auto_holdout_only():
    rows, cuts = W.design_q2(_cohorts(_q2_frame()), _q2_frame().iloc[:0], np.random.default_rng(0))
    r = {x["q2"]: x for x in rows}
    assert r["S1a"]["status"] == "SURVIVED DESIGN" and r["S1a"]["eff"] < 0
    for h in ("S1b", "S1c", "S2a", "S4b"):
        assert r[h]["status"] == "HOLDOUT-ONLY"
    assert r["S3a"]["status"] == "UNTESTABLE" and r["S5"]["status"] == "UNTESTABLE"
    rows, _ = W.design_q2(_cohorts(_q2_frame(good=False)), _q2_frame().iloc[:0], np.random.default_rng(0))
    assert {x["q2"]: x for x in rows}["S1a"]["status"] == "CONTRADICTED"


# ── Q3 ──
def _trig(key, n, start="2026-12-07 01:00", val=1.0, ctrl_val=0.0, pairs=6, days=12):
    t0 = pd.Timestamp(start, tz=UTC)
    st = [t0 + pd.Timedelta(days=i % days, minutes=7 * (i // days)) for i in range(n)]
    t = pd.DataFrame({"trigger": key, "pair": [f"P{i % pairs}USDT" for i in range(n)], "signal_time": st})
    t["minute"] = t.signal_time.map(lambda x: x.value // 10 ** 6 // 60_000 * 60_000 - 60_000)
    t["trigger_id"] = [f"{key}#{i}" for i in range(n)]
    t["h_final"] = True
    t["h_cost_src"] = "route_b_book"
    t["window15"] = t.signal_time.dt.floor("15min").astype(str)
    t["cond_depth_scored"] = True
    rng = np.random.default_rng(len(key))
    for c in I.CELLS:
        t[c] = val + rng.normal(0, 0.3, n)
    c = t.loc[t.index.repeat(5)].copy().reset_index(drop=True)
    c["ctrl_k"] = list(range(5)) * n
    c["ctrl_final"] = True
    c["signal_time"] = c.signal_time + pd.Timedelta(hours=2)
    for col in I.CELLS:
        c[col] = ctrl_val + rng.normal(0, 0.3, len(c))
    return t, c


def test_b3_gate_collecting_and_pass():
    t, c = _trig("H4_ask_vacuum", 60)
    th, ch = W.q3_holdout_cohort(t, c, pd.Timestamp("2026-12-30", tz=UTC))
    rng = np.random.default_rng(0)
    r = W.b3_gate(th, ch, "g_tp2_sl2_net", rng)
    assert r["status"] == "PASSED", r.get("why")
    assert W.b3_gate(th.iloc[:10], ch, "g_tp2_sl2_net", rng)["status"] == "COLLECTING"
    t2, c2 = _trig("H4_bid_vacuum", 60, val=0.0, ctrl_val=0.0)
    th2, ch2 = W.q3_holdout_cohort(t2, c2, pd.Timestamp("2026-12-30", tz=UTC))
    assert W.b3_gate(th2, ch2, "g_tp2_sl2_net", rng)["status"] == "FAILED"


def test_variant_must_beat_parent():
    tv, _ = _trig("H4_ask_vacuum|c", 40, val=1.0)
    tp, _ = _trig("H4_ask_vacuum", 60, val=1.0)
    for d in (tv, tp):
        d["hour"] = d.window15
    lo, eff = W.variant_vs_parent(tv, tp, "g_tp2_sl2_net", np.random.default_rng(0), 300)
    assert abs(eff) < 0.3 and not lo > 0                     # equal to its parent → B3b fails


# ── Q4 ──
def test_q4_cuts_and_gate():
    days = pd.date_range("2026-11-21", periods=40, freq="D").strftime("%Y-%m-%d")
    m = pd.DataFrame({"day": days, "complete": True})
    for v in W.S0:
        m[f"s0_{v}"] = np.linspace(-1, 1, 40)
    cuts = W.q4_cuts(m)
    assert cuts["funding_mean_pct"]["sign"] == 0.0 and cuts["breadth_up_15"]["sign"] == pytest.approx(
        float(m[m.day < "2026-12-07"].s0_breadth_up_15.median()))
    daily = pd.DataFrame({"day": days, "x": np.r_[np.ones(20), -np.ones(20)], "y": np.r_[np.full(20, 0.9), np.full(20, 0.2)],
                          "net": np.r_[np.ones(20), -np.ones(20)]})
    rng = np.random.default_rng(0)
    assert W.q4_gate(daily, dict(sign=0.0), "sign", rng)["status"] == "PASSED"
    assert W.q4_gate(daily.iloc[:15], dict(sign=0.0), "sign", rng)["status"] == "COLLECTING"
    assert W.q4_gate(daily, dict(sign=None), "sign", rng)["status"] == "UNTESTABLE"


# ── freeze / helpers / phase guard ──
def test_freeze_reads_once():
    st = {}
    r, st = W.freeze(st, "L", dict(status="COLLECTING"), "c1")
    assert st == {}
    r, st = W.freeze(st, "L", dict(status="FAILED", eff=np.float64(0.1), ci=(np.nan, 1.0)), "c2")
    assert st["verdicts"]["L"]["frozen_at"] == "c2" and st["verdicts"]["L"]["ci"] == [None, 1.0]
    r, st2 = W.freeze(st, "L", dict(status="PASSED"), "c3")
    assert r["status"] == "FAILED" and st2 == st
    _, st3 = W.freeze(st, "M", dict(status="STALE"), "c3")
    assert "M" not in st3["verdicts"]


def test_haircut_text():
    assert W.haircut_text(0.2, "pp") == "+10.0 … +14.0 pp"
    assert W.haircut_text(-1.0, "%") == "-0.7 … -0.5 %"
    assert W.haircut_text(None, "pp") == "–"


def test_before_first_run_reads_nothing(monkeypatch, capsys):
    monkeypatch.setattr(W.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-12-09 23:59", tz=UTC)))
    monkeypatch.setattr(W, "load_all", lambda: (_ for _ in ()).throw(AssertionError("read before 12-10")))
    assert W.main([]) == 0
    assert "nothing read" in capsys.readouterr().out


def test_committed_clean_false_for_untracked(tmp_path):
    p = os.path.join(W.ROOT, "reports", "_weekly_scan_untracked_probe.json")
    try:
        open(p, "w").write("{}")
        assert not W.committed_clean(p)
    finally:
        os.remove(p)


# ── end to end on synthetic datasets ──
def _world():
    exh = pd.concat([_sig(220, "2026-11-20", per_day=13, seed=1), _sig(220, "2026-12-07 00:20", per_day=14, seed=2)], ignore_index=True)
    rng = np.random.default_rng(9)
    p = np.where(exh.rb_x > 0, 0.9, 0.2)
    exh["out_lbl_worked"] = (rng.random(len(exh)) < p).astype(float)
    exh["out_net_pct"] = np.where(exh.out_lbl_worked == 1, 1.0, -1.2)
    exh["signal_id"] = exh.signal_id + "#" + exh.index.astype(str)
    for q in W.Q2_ALL:
        exh[f"q2_{q}"] = np.nan
    exh["today_filters_pass"] = True
    exh["out_lbl_group"] = np.where(exh.out_lbl_worked == 1, "CAPTURED", "NEVER_WORKED")
    exh["s5_lbl_worked"] = np.nan
    S = exh
    t, c = _trig("H4_ask_vacuum", 40, start="2026-11-20 01:00", days=16)
    t2, c2 = _trig("H4_ask_vacuum", 40, start="2026-12-07 01:00", days=12)
    t2["trigger_id"] = t2.trigger_id + "h"
    c2["trigger_id"] = c2.trigger_id + "h"
    T, C = pd.concat([t, t2], ignore_index=True), pd.concat([c, c2], ignore_index=True)
    days = pd.date_range("2026-11-21", periods=30, freq="D").strftime("%Y-%m-%d")
    md = pd.DataFrame({"day": days, "complete": True})
    for v in W.S0:
        md[f"s0_{v}"] = np.random.default_rng(1).normal(size=30)
    meta = {"data_end": "2026-12-31 00:00:00+00:00", "run_utc": "2026-12-31T00:05:00+00:00"}
    return S, T, C, meta, md


def test_end_to_end_design_then_holdout(monkeypatch, tmp_path):
    world = _world()
    monkeypatch.setattr(W, "load_all", lambda: world)
    monkeypatch.setattr(W, "SURVIVORS", str(tmp_path / "SURV.json"))
    monkeypatch.setattr(W, "STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(W, "ROOT", str(tmp_path))
    monkeypatch.setattr(W, "exit_hashes", lambda m: {s: "h0" for s in W.SLEEVES})
    monkeypatch.setattr(W, "data_problem", lambda m, now: None)
    monkeypatch.setattr(W, "load_meta", lambda: world[3])
    monkeypatch.setattr(W, "head_json", lambda p: None)
    monkeypatch.setattr(W, "git_head", lambda: "abc123")
    monkeypatch.setattr(W, "N_ROT", 40)
    monkeypatch.setattr(I, "N_PERM", 40)
    monkeypatch.setattr(W, "h4_status", lambda: {"H4-T": "COLLECTING", "H4-B": "COLLECTING"})
    drawn = []
    monkeypatch.setattr(W, "event_chart_for", lambda ln, *a, **k: drawn.append(ln["line"]))    # no network in tests
    os.makedirs(tmp_path / "reports")
    monkeypatch.setattr(W.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-12-31 10:00", tz=UTC)))
    monkeypatch.setattr(W, "in_head", lambda p: False)
    assert W.main([]) == 2                                                   # after 12-10 without the operator's OK → refused
    assert not os.path.exists(W.SURVIVORS)
    assert W.main(["--late-design"]) == 0                                    # DESIGN (late, operator OK)
    surv = json.load(open(W.SURVIVORS))
    assert any(s["line"] == "Q1|EXH|rb_x|sign" for s in surv["q1"]["EXH"]["survivors"])
    assert {k["key"]: k["status"] for k in surv["q3"]["keys"]}["H4_ask_vacuum"] in ("SURVIVED DESIGN", "FAILED")
    assert surv["q3"]["addendum4_order"][0]["family"] == "H4"
    rep = (tmp_path / "reports" / "ROUTE_B_WEEKLY_SCAN_2026-12-31_DESIGN.md").read_text()
    assert "HOLDOUT is NOT read" in rep and "LATE" in rep and "## EXH" in rep
    assert "not read until SURVIVORS.json is committed" in rep and "BTC daily-trend split" not in rep   # A3: no HOLDOUT outcome yet
    assert "Path decomposition" not in rep and "Q5 ·" not in rep and "Event studies" not in rep
    for row in [r for r in rep.splitlines() if r.startswith("| ") and "`Q" in r]:
        cells = re.split(r"(?<!\\)\|", row)
        assert len(cells) == 9 and cells[5].strip() == "–" and cells[6].strip() == "–", row    # 7 columns; no HOLDOUT effect / CI
    assert "| H4_ask_vacuum |" in rep and "net %" in rep
    monkeypatch.setattr(W, "committed_clean", lambda p: False)
    assert W.main([]) == 2                                                   # A3: uncommitted → refused
    monkeypatch.setattr(W, "committed_clean", lambda p: True)
    assert W.main([]) == 0                                                   # HOLDOUT
    st = json.load(open(W.STATE))
    assert st["verdicts"]["Q1|EXH|rb_x|sign"]["status"] == "PASSED"
    rep = (tmp_path / "reports" / "ROUTE_B_WEEKLY_SCAN_2026-12-31.md").read_text()
    assert "🟢 PASSED" in rep and "`Q1\\|EXH\\|rb_x\\|sign`" in rep
    assert "BTC daily-trend split" in rep and "`Q4\\|DIST` — 16 market lines" in rep
    assert "Q1|EXH|rb_x|sign" in drawn                                       # §7: a PASSED line gets its event study
    # read once: a changed HOLDOUT never re-judges a frozen line
    S, T, C, meta, md = world
    S2 = S.copy()
    S2.loc[S2.signal_time >= W.HOLDOUT_START, "out_lbl_worked"] = 0.0
    monkeypatch.setattr(W, "load_all", lambda: (S2, T, C, meta, md))
    W.main([])
    assert json.load(open(W.STATE))["verdicts"]["Q1|EXH|rb_x|sign"]["status"] == "PASSED"
    # an exit change on a not-yet-judged survivor triggers ONE re-check (frozen FAILED if it no longer holds)
    st = json.load(open(W.STATE))
    other = [s["line"] for s in surv["q1"]["EXH"]["survivors"] if s["line"] not in st["verdicts"]]
    monkeypatch.setattr(W, "exit_hashes", lambda m: {s: "h1" for s in W.SLEEVES})
    W.main([])
    st = json.load(open(W.STATE))
    for ln in other:
        assert st["recheck"][ln]["hash"] == "h1"


def test_design_refuses_when_survivors_in_head_but_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "SURVIVORS", str(tmp_path / "SURV.json"))
    monkeypatch.setattr(W, "in_head", lambda p: True)
    monkeypatch.setattr(W.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-12-31 10:00", tz=UTC)))
    monkeypatch.setattr(W, "load_all", lambda: (_ for _ in ()).throw(AssertionError("must not run DESIGN twice")))
    assert W.main([]) == 2


def test_fav_path_and_band():
    k = pd.DataFrame({"open_ms": [0, 60_000, 120_000, 180_000], "o": [100.0, 101, 102, 99], "c": [101.0, 102, 99, 98]})
    assert np.allclose(W.fav_path(k, 0, 1, 3), [1.0, 2.0, -1.0])
    assert np.allclose(W.fav_path(k, 0, -1, 3), [-1.0, -2.0, 1.0])
    assert W.fav_path(k, 0, 1, 5) is None and W.fav_path(k, 30_000, 1, 2) is None
    m, lo, hi = W.band([np.array([0.0, 1.0]), np.array([2.0, 3.0]), np.array([4.0, 5.0])])
    assert np.allclose(m, [2.0, 3.0]) and np.allclose(lo, [1.0, 2.0]) and np.allclose(hi, [3.0, 4.0])


# ── review round 1: added coverage ──
def test_exclusions_match_addendum10_patterns():
    for c in ("j_exh_has_spot", "rb_has_book", "has_fb", "fb_age_min", "rb_book_age_min", "rb_tape_minutes_15", "rb_ask_usd_05_lb",
              "rb_mid", "rb_mark", "journal_t"):
        assert W.excluded_var(c), c
    for c in ("rb_imb_05", "j_exh_vol24_usd", "journal_atr_pct", "oi_chg_15m", "fb_cost_sell_6000"):
        assert not W.excluded_var(c), c


def test_bool_is_na_safe():
    got = W._bool(pd.Series([True, False, pd.NA, None, np.nan, 1, 0, "True", "x"], dtype=object))
    assert got.tolist() == [True, False, False, False, False, True, False, True, False]


def test_line_rng_reproducible_per_line():
    a = W.line_rng("Q1|EXH|rb_x|sign").random(3)
    assert np.array_equal(a, W.line_rng("Q1|EXH|rb_x|sign").random(3))
    assert not np.array_equal(a, W.line_rng("Q1|EXH|rb_x|tercile").random(3))


def test_net_unpriced_is_never_a_contradiction():
    assert W.net_verdict(math.nan, 1) == "unpriced" and W.net_verdict(0.3, 1) == "ok" and W.net_verdict(-0.3, 1) == "contra"
    H = _hold()
    H["net"] = np.where(H.rb_x > 0, np.nan, H.net)
    a, b = (H.rb_x > 0).to_numpy(), (H.rb_x <= 0).to_numpy()
    r = W.b_gate(H, a, b, 0.6, 1, np.random.default_rng(0))
    assert r["status"] == "UNTESTABLE"
    st, _ = W.freeze({}, "L", r, "c")
    assert st["status"] == "UNTESTABLE"                                   # not frozen
    assert W.b_gate(H, a, b, math.nan, 1, np.random.default_rng(0))["status"] in ("UNTESTABLE",)


def test_b_gate_nan_design_effect_fails_ratio():
    H = _hold()
    a, b = (H.rb_x > 0).to_numpy(), (H.rb_x <= 0).to_numpy()
    r = W.b_gate(H, a, b, math.nan, 1, np.random.default_rng(0))
    assert r["status"] == "FAILED" and "< 50 % of DESIGN" in r["why"]


def test_q4_one_regime_label():
    days = pd.date_range("2026-12-07", periods=40, freq="D").strftime("%Y-%m-%d")
    daily = pd.DataFrame({"day": days, "x": np.r_[np.ones(20), -np.ones(20)], "y": np.r_[np.full(20, 0.9), np.full(20, 0.2)],
                          "net": np.r_[np.ones(20), -np.ones(20)]})
    assert W.q4_gate(daily, dict(sign=0.0), "sign", np.random.default_rng(0), regime_n=1)["regime"] == "one regime"


def test_design_q3_unit():
    tA, cA = _trig("H4_ask_vacuum", 40, start="2026-11-20 01:00", val=1.0, ctrl_val=0.0, days=16)
    tB, cB = _trig("H4_bid_vacuum", 40, start="2026-11-20 01:00", val=0.0, ctrl_val=0.0, days=16)
    tS, cS = _trig("H1_liq_long_cascade", 5, start="2026-11-20 01:00", days=5)          # below the floor
    tH, cH = _trig("H3_imb_bid_heavy_thin", 30, start="2026-11-20 01:00", days=10)      # HOLDOUT-only
    tX, cX = _trig("H4_ask_vacuum", 10, start="2026-12-08 01:00")                       # HOLDOUT rows: never in DESIGN
    tX["trigger_id"] += "x"
    cX["trigger_id"] += "x"
    T = pd.concat([tA, tB, tS, tH, tX], ignore_index=True)
    C = pd.concat([cA, cB, cS, cH, cX], ignore_index=True)
    rows, bar, order = W.design_q3(T, C, pd.Timestamp("2026-12-31", tz=UTC), np.random.default_rng(0), n_perm=50)
    r = {x["key"]: x for x in rows}
    assert r["H4_ask_vacuum"]["status"] == "SURVIVED DESIGN" and r["H4_ask_vacuum"]["cell"] in I.CELLS
    assert r["H4_ask_vacuum"]["n_eligible"] == 40 and all(not i.endswith("x") for i in r["H4_ask_vacuum"]["design_ids"])
    assert r["H4_bid_vacuum"]["status"] == "FAILED"
    assert r["H1_liq_long_cascade"]["status"] == "UNTESTABLE" and r["H3_imb_bid_heavy_thin"]["status"] == "HOLDOUT-ONLY"
    assert r["H3_imb_bid_heavy_thin"]["cell"] == I.FIXED
    assert [o["family"] for o in order][:1] == ["H4"]
    assert W.q3_split("H7a_oi_flush_dump") == W.HALF_SPLIT_611 and W.q3_split("H4_ask_vacuum|c") == W.HALF_SPLIT_611
    assert W.q3_split("H4_ask_vacuum") == W.HALF_SPLIT and W.order_fam("H7b_oi_flush_pump") == "H7"


def test_b3_unwalkable_robustness_branch():
    t, c = _trig("H4_ask_vacuum", 60)
    th, ch = W.q3_holdout_cohort(t, c, pd.Timestamp("2026-12-30", tz=UTC))
    empty_t, empty_c = th.iloc[:0], ch.iloc[:0]
    r = W.b3_gate(th, ch, "g_tp2_sl2_net", np.random.default_rng(0), tu=empty_t, cu=empty_c)
    assert r["status"] == "FAILED" and "fails without book_unwalkable rows" in r["why"]
    r = W.b3_gate(th, ch, "g_tp2_sl2_net", np.random.default_rng(0), tu=th, cu=ch)
    assert r["status"] == "PASSED"


def test_recheck_failed_and_stale_paths():
    d = _planted()
    out = W.design_q1(d, "EXHAUSTION_SHORT", np.random.default_rng(W.SEED), n_rot=40)
    sv = next(s for s in out["survivors"] if s["line"] == "Q1|EXH|rb_x|sign")
    sv = dict(sv, sleeve="EXHAUSTION_SHORT", _bar=out["bar"], _hash0="h0", design_ids=out["design_ids"])
    co = {"EXHAUSTION_SHORT": d}
    assert W.recheck_stale(sv["line"], sv, co, d.iloc[:0], None, {}, {"EXHAUSTION_SHORT": "h0"}) == (None, {})   # no change
    d2 = d.copy()
    d2["y"] = np.where(d2.rb_x > 0, 0.0, 1.0)                     # re-priced rows now point the other way
    st, state = W.recheck_stale(sv["line"], sv, {"EXHAUSTION_SHORT": d2}, d.iloc[:0], None, {}, {"EXHAUSTION_SHORT": "h1"})
    assert st == "FAILED" and state["recheck"][sv["line"]] == {"hash": "h1", "ok": False}
    res, state = W.stale_result(st, state, sv["line"], "c")
    assert res["status"] == "FAILED" and state["verdicts"][sv["line"]]["status"] == "FAILED"
    st, state2 = W.recheck_stale(sv["line"], sv, {"EXHAUSTION_SHORT": d.iloc[:0]}, d.iloc[:0], None, {}, {"EXHAUSTION_SHORT": "h1"})
    assert st == "STALE" and state2 == {}
    st, _ = W.recheck_stale(sv["line"], sv, co, d.iloc[:0], None, {}, {"EXHAUSTION_SHORT": "h1"})
    assert st == "SURVIVED DESIGN"


def test_data_problem():
    now = pd.Timestamp("2026-12-10 10:00", tz=UTC)
    ok = dict(run_utc="2026-12-10T09:00:00+00:00", data_end="2026-12-10 08:00:00+00:00")
    import weekly_scan as WW
    orig = WW.config_hashes
    try:
        WW.config_hashes = lambda: WW.exit_hashes(ok)
        assert W.data_problem(ok, now) is None
        assert "> 24 h" in W.data_problem(dict(ok, run_utc="2026-12-08T09:00:00+00:00"), now)
        assert "72 h" in W.data_problem(dict(ok, data_end="2026-12-09 23:00:00+00:00"), now)
        WW.config_hashes = lambda: {s: "other" for s in W.SLEEVES}
        assert "re-run build.py" in W.data_problem(ok, now)
    finally:
        WW.config_hashes = orig
    assert "no run_utc" in W.data_problem({}, now)


def test_state_problem_and_design_ran_before(monkeypatch, tmp_path):
    monkeypatch.setattr(W, "STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(W, "ROOT", str(tmp_path))
    monkeypatch.setattr(W, "head_json", lambda p: {"verdicts": {"L": {"status": "FAILED"}}})
    assert "missing" in W.state_problem()
    json.dump({"verdicts": {}}, open(W.STATE, "w"))
    assert "L" in W.state_problem()
    json.dump({"verdicts": {"L": {"status": "FAILED"}, "M": {"status": "PASSED"}}}, open(W.STATE, "w"))
    assert W.state_problem() is None
    os.makedirs(tmp_path / "reports")
    monkeypatch.setattr(W, "SURVIVORS", str(tmp_path / "reports" / "SURV.json"))
    assert not W.design_ran_before()
    (tmp_path / "reports" / "ROUTE_B_WEEKLY_SCAN_2026-12-10.md").write_text("# 📡 Route B weekly research scan — 2026-12-10 (DESIGN — run once)\n")
    assert W.design_ran_before()
    monkeypatch.setattr(W.pd.Timestamp, "now", classmethod(lambda cls, tz=None: pd.Timestamp("2026-12-17 10:00", tz=UTC)))
    monkeypatch.setattr(W, "in_head", lambda p: False)
    monkeypatch.setattr(W, "load_all", lambda: (_ for _ in ()).throw(AssertionError("must not run DESIGN twice")))
    assert W.main([]) == 2


def test_cohort_keeps_flatbase_filter_skips_and_flags_gate_inputs():
    d = _sig(3, "2026-11-20", sleeve="FLATBASE_SHORT", status="REFUSED:SKIPPED_HIGHER_HIGH")
    d.loc[1, "status"] = "REFUSED:SKIPPED_CONFIRM"
    got = W.sleeve_cohort(d, "FLATBASE_SHORT")
    assert sorted(got.status) == ["REFUSED:SKIPPED_HIGHER_HIGH"] * 2       # armed filter blocks stay; the core confirm skip is out
    assert W.is_holdout_only(dict(var="j_dist_rsi1h")) and W.excluded_var("j_bl_sole")


def test_sleeve_daily_robust_and_partial_day():
    H = W._prep(_sig(30, "2026-12-07 01:00", per_day=3), "out_lbl_worked")
    assert W.sleeve_daily(H, pd.DataFrame(), "breadth_up_15").empty
    days = sorted(H.day.unique())
    md = pd.DataFrame({"day": days, "complete": ["True"] * len(days), "s0_breadth_up_15": 1.0})
    assert len(W.sleeve_daily(H, md, "breadth_up_15")) == len(days)        # "True" strings read as complete
    assert W.sleeve_daily(H, md, "breadth_up_15", last_day=days[3]).day.max() == days[3]
