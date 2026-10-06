"""🌊 GVOL_BLOCKED + 🪜 VWAP_STOP scout lines (observe-only, registered 2026-10-06; review fixes applied) — the frozen bars, the pure walkers,
and passes of each tracker on stubbed data (no network, state files in tmp_path)."""
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_frenzy_exits as X  # noqa: E402

T0 = 1_790_000_000_000 // 300_000 * 300_000
DAY0 = int(pd.Timestamp("2026-10-07", tz="UTC").value // 1_000_000)
LATER = DAY0 + 30 * 86_400_000
tv = lambda ps, t=T0: (np.array([t + i * X.MIN for i in range(len(ps))]), np.array(ps, float))


def test_frozen_constants():
    assert (X.GVB_N, X.GVB_DAYS, X.GVB_CONFIRM, X.GVB_DAY_MAX) == (20, 10, -0.20, 50.0)
    assert (X.GVB_LONG_LEV, X.GVB_LONG_LEV_STRONG, X.GVB_WIDE_LEV, X.HG_STREAK) == (0.32, 0.5, 0.2, 12.0)
    assert (X.VWS_N, X.VWS_TOP, X.VWS_SLEEVE_MIN) == (20, 50.0, 5)
    assert (X.VWS_K, X.VWS_LINE, X.VWS_FLOOR) == (0.5, -3.0, -12.0)
    assert X.EPISODE_MERGE_MS == 30 * X.MIN and X.LOCK_FROM == "2026-10-05T16:00:00"
    assert set(X.GVB_HIGH) == {"FRENZY_GVOL_HIGH", "FRENZY_WIDE_GVOL_HIGH"}
    assert set(X.GVB_UNREAD) == {"FRENZY_GVOL_UNREAD", "FRENZY_WIDE_GVOL_UNREAD"}
    assert "+400" in X.VWS_YEAR and "−395" in X.VWS_YEAR and "+0.03" in X.VWS_YEAR


def test_hold_green_calls_the_engine_function_at_the_frozen_streak(monkeypatch):
    seen = {}

    def spy(ep, code, th):
        seen.update(streak=th.frenzy_wide_hold_green_streak, code=code)
        return None
    monkeypatch.setattr(X, "frenzy_wide_hold_green_block", spy)
    th = SimpleNamespace(frenzy_wide_hold_green_streak=0.0, frenzy_wide_above_share_min=0.0)   # the live switch off
    ep = dict(fresh_on=True, above_streak=13, bar_ret_pct=0.2, above_share=80.0)
    assert X.gvb_take("WIDE", ep, "FRENZY_GREEN_BAR", 2.0, th) == (True, "")
    assert seen == dict(streak=12.0, code="FRENZY_GREEN_BAR") and th.frenzy_wide_hold_green_streak == 0.0   # the live th is not mutated
    monkeypatch.setattr(X, "frenzy_wide_hold_green_block", lambda ep, code, th: "FRENZY_WIDE_RECLAIM")
    assert X.gvb_take("WIDE", ep, "FRENZY_GREEN_BAR", 2.0, th) == (False, "FRENZY_WIDE_RECLAIM")


def test_gvb_bar_days_and_day_concentration():
    x = [0.4] * 30
    assert X.gvb_check(x, ["d1"] * 30, 0.0)[0] == "collecting"                      # 30 signals on 1 day = one observation-day
    assert X.gvb_check(x, [f"d{i % 10}" for i in range(30)], 0.0)[0] == "review"
    assert X.gvb_check(x, [f"d{i % 10}" for i in range(30)], None)[0] == "inconclusive"
    conc = [6.0] + [-0.1] * 19 + [0.1] * 10                                          # day d0 carries 6.0 of a +5.1 net
    st, tx = X.gvb_check(conc, [f"d{i % 10}" for i in range(30)], -1.0)
    assert st == "inconclusive" and "window leg fails" in tx
    neg = [-3.0] + [-0.2] * 29                                                       # a confirm read carried by one day also fails
    assert X.gvb_check(neg, ["d0"] + [f"d{1 + i % 12}" for i in range(29)], 0.0)[0] == "confirmed"
    assert X.gvb_check([-9.0] + [-0.05] * 29, ["d0"] + [f"d{1 + i % 12}" for i in range(29)], 0.0)[0] == "inconclusive"


def test_episode_merge_30_min_and_in_green_clock():
    ek = X.episode_keys(pd.DataFrame(dict(pair=["AIN", "AIN", "AIN", "AIN"], spike_at=["2026-10-04T14:25:00", "2026-10-04T14:30:00",
                                                                                          "2026-10-04T14:55:00", "2026-10-04T15:40:00"])))
    assert ek[0] == ek[1] == ek[2] == "AIN|2026-10-04T14:25:00" and ek[3] == "AIN|2026-10-04T15:40:00"   # chained ≤ 30 min
    gc = pd.DataFrame(dict(k=["2026-10-08T01:00:00", "2026-10-08T02:00:00"], pair="X", spike_at=["2026-10-07T14:25:00", "2026-10-07T14:30:00"],
                           v2=True, eligible=True, gvol="pass", cohort=True))
    assert list(X.gc_counted(gc)) == [True, False]                                    # replay drift no longer double-counts a spike


def test_bp_rule_floor_slip_and_order():
    tt, pp = tv([100, 98, 96.9, 96, 95.5, 95])
    r = X.vwap_shadow(tt, pp, 100, T0, T0 + 2 * X.MIN, 99.0, 2.0, [T0 + 3 * X.MIN], [96.0])
    assert r[1] == T0 + 3 * X.MIN and abs(r[0] - (-4.0 - X.FEE - X.SLIP)) < 1e-9    # both legs: ≤ −3 net ∧ < 98.01; slip charged once
    assert X.vwap_shadow(tt, pp, 100, T0, T0 + 2 * X.MIN, 99.0, 2.0, [T0 + 3 * X.MIN], [97.5])[2] == "open"    # −2.59: not ≤ −3
    assert X.vwap_shadow(tt, pp, 100, T0, T0 + 2 * X.MIN, 96.5, 2.0, [T0 + 3 * X.MIN], [96.0])[2] == "open"    # above 95.54
    fl = X.vwap_shadow(*tv([100, 97, 92, 87.5, 86]), 100, T0, T0 + X.MIN, 99.0, 2.0, [], [])
    assert fl[2] == "−12 floor" and abs(fl[0] - (-12.5 - X.FEE - X.SLIP)) < 1e-9
    # OHLC pseudo prints: the minute that arms the lock (high first) then dips — the low cannot escape the trail
    m1 = [[T0, 100, 100, 96.5, 96.6, 1], [T0 + X.MIN, 96.6, 104.5, 98.0, 99.0, 1]]
    t1, p1 = X.m1_prints(m1)
    assert list(p1) == [100, 100, 96.5, 96.6, 96.6, 104.5, 98.0, 99.0]
    r = X.vwap_shadow(t1, p1, 100, T0, T0 + 30_000, 99.0, 2.0, [], [], gap=True)
    assert r[2] == "floor / trail" and abs(r[0] - (2.41 - X.SLIP)) < 1e-9            # armed by the high (+4.41), out at the trail line on the low


def test_replica_parity_on_fills_live_did_not_stop():
    assert X.replica_stop(*tv([100, 98, 96.5, 99, 104]), 100, T0, T0 + 4 * X.MIN) == (True, T0 + 2 * X.MIN)
    assert X.replica_stop(*tv([100, 98, 99, 104, 96]), 100, T0, T0 + 4 * X.MIN) == (False, None)   # the −3 print after the live exit does not count


def _F(rows):
    cols = ["opened_at", "k", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "close_reason", "closed_at",
            "entry_frenzy_vwap", "entry_frenzy_vs_vwap_pct", "entry_atr_pct", "entry_frenzy_spike_at"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows], columns=cols)


def _fill(o, pair, strat, act, reason, closed, vwap=95.0, spike=None):
    return (X._iso(o), X._iso(o), pair, "LONG", strat, "CLOSED", 100.0, act, reason, X._iso(closed), vwap, 5.0, 2.0, spike)


def test_vws_run_stopped_parity_and_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "VWS_CSV", str(tmp_path / "vws.csv"))
    o1, o2, o3, o4 = DAY0 + X.H, DAY0 + 2 * X.H, DAY0 + 3 * X.H, DAY0 + 4 * X.H
    F = _F([_fill(o1, "XUSDT", "FRENZY_LONG", -3.0, "STOP_LOSS", o1 + 2 * X.MIN),
            _fill(o2, "YUSDT", "FRENZY_WIDE", 2.5, "RUNNER_TRAIL", o2 + 4 * X.MIN),
            _fill(o3, "ZUSDT", "FRENZY_LONG", 2.5, "RUNNER_TRAIL", o3 + 4 * X.MIN),
            _fill(o4, "VUSDT", "FRENZY_LONG", -3.0, "STOP_LOSS", o4 + 2 * X.MIN, vwap=float("nan"))])
    paths = {"XUSDT": [100, 98, 96.9, 101, 104, 101.5], "YUSDT": [100, 101, 103.5, 106, 104.4], "ZUSDT": [100, 96.5, 103.5, 106, 104.4]}
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[T, 100.0, 100.0, 100.0, 100.0, 1.0] for T in range(s // X.BAR * X.BAR, e, X.BAR)])
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", *tv(paths[pair], t0)))
    L = X.vws_run(LATER, F)
    v = pd.read_csv(X.VWS_CSV).set_index("pair")
    assert v.loc["XUSDT", "shadow_how"] == "floor / trail" and abs(v.loc["XUSDT", "delta"] - (1.41 - X.SLIP + 3.0)) < 1e-9
    assert v.loc["YUSDT", "delta"] == 0.0 and not v.loc["YUSDT", "replica_stop"]
    assert v.loc["ZUSDT", "delta"] == 0.0 and v.loc["ZUSDT", "replica_stop"]          # the replica stops a fill live rode through: counted
    assert v.loc["VUSDT", "excluded"] and v.loc["VUSDT", "final"] and v.loc["VUSDT", "excl_reason"] == "no entry_frenzy_vwap stamp"
    txt = "\n".join(L)
    assert "2 final fills, 1 replica stops" in txt and "⚠ the replica stops" in txt and "Excluded (stopped fill without a VWAP stamp / live P&L, stored once): 1" in txt
    assert "+400" in txt and "BP k 0.5" in txt
    monkeypatch.setattr(X, "_vws_price", lambda *a: pytest.fail("a final row was re-priced"))
    X.vws_run(LATER, F)                                                             # every row final (excluded once) → nothing re-priced


def test_vws_reprices_stored_rows_without_the_export(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "VWS_CSV", str(tmp_path / "vws.csv"))
    o1 = DAY0 + X.H
    pd.DataFrame([dict(k=X._iso(o1), opened_at=X._iso(o1), pair="XUSDT", sleeve="LONG", entry=100.0, vwap=95.0, atr=2.0, vs_vwap=5.0, actual=-3.0,
                       close_reason="STOP_LOSS", stopped=True, exit_live_at=X._iso(o1 + 2 * X.MIN), day="2026-10-07", cohort=True, ver=X.VWS_VER,
                       prelock=False, excluded=False, final=False, delta=-0.5, shadow=-3.5, px_src="1m")]).to_csv(X.VWS_CSV, index=False)
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[T, 100.0, 100.0, 100.0, 100.0, 1.0] for T in range(s // X.BAR * X.BAR, e, X.BAR)])
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", *tv([100, 98, 96.9, 101, 104, 101.5], t0)))
    X.vws_run(LATER, _F([]))                                                        # the export is gone
    v = pd.read_csv(X.VWS_CSV)
    assert len(v) == 1 and v.final.all() and v.px_src.iloc[0] == "tick" and abs(v.delta.iloc[0] - (1.41 - X.SLIP + 3.0)) < 1e-9


def test_vws_quarantines_bad_rows_before_saving(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "VWS_CSV", str(tmp_path / "vws.csv"))
    good = dict(k="2026-10-07T01:00:00", pair="A", sleeve="LONG", cohort=True, ver=X.VWS_VER, stopped=False, final=True, delta=0.0, replica_stop=False)
    bad = dict(good, k="2026-10-07T02:00:00", pair="B", delta=0.4)
    pd.DataFrame([good, bad]).to_csv(X.VWS_CSV, index=False)
    L = X.vws_run(DAY0, _F([]))
    saved = pd.read_csv(X.VWS_CSV)
    q = glob.glob(str(tmp_path / "vws.csv.*.bad"))
    assert list(saved.pair) == ["A"] and len(q) == 1 and list(pd.read_csv(q[0]).pair) == ["B"]
    assert any("quarantined" in x for x in L)
    X.vws_run(DAY0, _F([]))                                                         # a second quarantine never overwrites the first
    assert len(glob.glob(str(tmp_path / "vws.csv.*.bad"))) == 1                       # (nothing bad left the second time)
    p1 = X._bad_path(X.VWS_CSV); open(p1, "w").close()
    assert X._bad_path(X.VWS_CSV) != p1


def test_vws_gate_sleeve_min_five():
    k = [f"2026-10-{8 + i // 10:02d}T{i % 10:02d}:00:00" for i in range(20)]
    w = pd.DataFrame(dict(k=k, pair="P", final=True, sleeve=["LONG"] * 17 + ["WIDE"] * 3, delta=[1.0] * 17 + [-1.0] * 3))
    st, tx = X.vws_check(w)
    assert st == "candidate" and "leg not applied" in tx
    assert X.vws_check(w.assign(sleeve=["LONG"] * 15 + ["WIDE"] * 5, delta=[1.0] * 15 + [-1.0] * 5))[0] == "close"


def _gvb_stubs(monkeypatch, streak, code, in_state=None):
    monkeypatch.setattr(X, "_kl", lambda sym, tf, start, end: [[start + i * {"1m": X.MIN, "5m": X.BAR, "1h": X.H}[tf], 100.0, 100.5, 99.5, 100.0, 1000.0]
                                                              for i in range(min(int((end - start) // {"1m": X.MIN, "5m": X.BAR, "1h": X.H}[tf]) + 1, 1500))])
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", np.array([t0 + 13_000, t0 + 60_000, t0 + 120_000]),
                                                                   np.array([100.0, 104.0, 101.5] if pair != "WUSDT" else [100.0, 96.0, 95.0])))
    monkeypatch.setattr(X, "normal_hour_usd", lambda h1, ms: 1.0)
    sig_of = lambda bars: bars[-1][0] + X.BAR
    monkeypatch.setattr(X, "frenzy_walk", lambda bars, nh, th: dict(
        spike_ts=DAY0 - 5 * X.H + (5 * X.MIN if sig_of(bars) % (2 * X.H) else 0), hours=5.0, above_streak=streak[sig_of(bars)], above_share=80.0,
        vs_vwap_pct=2.0, bar_ret_pct=0.4, verified=True, in_state=(in_state or {}).get(sig_of(bars), True), fresh_on=True, _sig=sig_of(bars)))
    monkeypatch.setattr(X, "frenzy_flagged", lambda ep, th: True)
    monkeypatch.setattr(X, "frenzy_long_status", lambda ep, atr, v, th: (False, code[ep["_sig"]], ""))


def test_gvb_run_same_ruler_double_count_and_catchup(tmp_path, monkeypatch):
    """XUSDT: a LONG gvol refusal (READY) then a WIDE one of the same spike (5 min replay drift) → one counted; YUSDT WIDE reclaim → not taken;
    WUSDT blocked but its episode also had a live fill → excluded (listed); QUSDT replay FRENZY_ON in state → catch-up line; a pre-floor
    refusal → reference; the let-through fill priced with _lock_shadow (its live actual display-only)."""
    monkeypatch.setattr(X, "GVB_CSV", str(tmp_path / "gvb.csv"))
    s1, s2, s3, pre, s5, s6 = DAY0 + X.H, DAY0 + 2 * X.H, DAY0 + 3 * X.H, DAY0 - 2 * X.H, DAY0 + 5 * X.H, DAY0 + 6 * X.H
    streak = {s1: 20, s2: 20, s3: 8, pre: 20, s5: 20, s6: 20, DAY0 + 7 * X.H: 20}
    code = {s1: "FRENZY_READY", s2: "FRENZY_GREEN_BAR", s3: "FRENZY_GREEN_BAR", pre: "FRENZY_READY", s5: "FRENZY_READY", s6: "FRENZY_ON",
            DAY0 + 7 * X.H: "FRENZY_READY"}
    _gvb_stubs(monkeypatch, streak, code)
    rows = [(s1, "XUSDT", "FRENZY_GVOL_HIGH"), (s2, "XUSDT", "FRENZY_WIDE_GVOL_HIGH"), (s3, "YUSDT", "FRENZY_WIDE_GVOL_HIGH"),
            (pre, "XUSDT", "FRENZY_GVOL_HIGH"), (s5, "WUSDT", "FRENZY_GVOL_HIGH"), (s6, "QUSDT", "FRENZY_GVOL_HIGH")]
    J = pd.DataFrame([(X._iso(t), "BLOCK", p, g, "") for t, p, g in rows], columns=["t", "e", "pair", "gate", "strategy"])
    fo = DAY0 + 7 * X.H + 8_000                                                       # a live WUSDT fill of the same spike, opened 8 s after its close
    F = _F([_fill(fo, "WUSDT", "FRENZY_LONG", -3.0, "STOP_LOSS", fo + 10 * X.MIN, spike=X._iso(DAY0 - 5 * X.H + 5 * X.MIN))])
    th = SimpleNamespace(frenzy_wide_hold_green_streak=12.0, frenzy_wide_above_share_min=0.0)
    L = X.gvb_run(LATER, th, J, pd.DataFrame(), F)
    g = pd.read_csv(X.GVB_CSV).set_index(["k", "pair"])
    assert g.loc[(X._iso(s1), "XUSDT"), "counted"] and not g.loc[(X._iso(s2), "XUSDT"), "counted"]
    assert not g.loc[(X._iso(s3), "YUSDT"), "would_take"] and g.loc[(X._iso(s3), "YUSDT"), "why"] == "FRENZY_WIDE_RECLAIM"
    assert g.loc[(X._iso(s5), "WUSDT"), "double"] and not g.loc[(X._iso(s5), "WUSDT"), "counted"]
    assert g.loc[(X._iso(s6), "QUSDT"), "catchup"] and not g.loc[(X._iso(s6), "QUSDT"), "counted"]
    p = g.loc[(X._iso(DAY0 + 7 * X.H), "WUSDT")]
    assert p.gate == X.GVB_PASSED and p.passed_counted and abs(p.LOCK - (-4.0 - X.FEE - X.SLIP)) < 1e-9 and p.actual == -3.0
    txt = "\n".join(L)
    assert "1/20 signals · 1/10 days" in txt and "let-through mean -4.19 % (same ruler)" in txt
    assert "live actual (display only" in txt and "-3.00 %" in txt
    assert "also had a live fill (no double count): 1 (W " in txt and "Catch-up / not replayable at t" in txt and ": 1 (Q " in txt
    # a second run re-prices nothing final
    monkeypatch.setattr(X, "_lock_shadow", lambda *a: pytest.fail("a final row was re-priced"))
    X.gvb_run(LATER, th, J, pd.DataFrame(), F)


def test_green_clock_lists_catchup_lines(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "GC_CSV", str(tmp_path / "gc.csv"))
    s1 = DAY0 + X.H
    _gvb_stubs(monkeypatch, {s1: 20}, {s1: "FRENZY_ON"})
    J = pd.DataFrame([(X._iso(s1), "BLOCK", "XUSDT", "FRENZY_GREEN_BAR", ""), (X._iso(s1), "BLOCK", "XUSDT", "FRENZY_WIDE_CHOPPY", "")],
                     columns=["t", "e", "pair", "gate", "strategy"])
    F = pd.DataFrame(columns=["opened_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "closed_at", "k"])
    th = SimpleNamespace(frenzy_max_slots=2, frenzy_max_entries_per_pair_day=3, frenzy_long_lev_mult=0.32, frenzy_wide_lev_mult=0.2)
    txt = "\n".join(X.gc_run(LATER, th, F, J, pd.DataFrame()))
    assert "Catch-up / not replayable at t" in txt and ": 1 (X " in txt and "catch-up 1" in txt and "not eligible 0" in txt


def test_prelock_label(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "VWS_CSV", str(tmp_path / "vws.csv"))
    o = int(pd.Timestamp("2026-10-04T05:05:00", tz="UTC").value // 1_000_000)
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[T, 100.0, 100.0, 100.0, 100.0, 1.0] for T in range(s // X.BAR * X.BAR, e, X.BAR)])
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", *tv([100, 98, 96.9, 101, 104, 101.5], t0)))
    txt = "\n".join(X.vws_run(LATER, _F([_fill(o, "SANDUSDT", "FRENZY_LONG", -3.0, "STOP_LOSS", o + 2 * X.MIN)])))
    assert "(ref · † pre-lock exit regime)" in txt and "1 from the pre-lock exit regime" in txt
