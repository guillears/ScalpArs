"""🛟 HYBRID_EXIT per-fill shadow (observe-only, 2026-10-06; V3 of reports/FRENZY_EXIT_LOCK_VS_BULLRUN_2026-10-06.md) — the frozen walker
and bar, the saved / cut anatomy, hyb_run on stubbed data, and the HYB column in the exit table (no network; state in tmp_path)."""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_frenzy_exits as X  # noqa: E402

DAY0 = int(pd.Timestamp("2026-10-07", tz="UTC").value // 1_000_000)
LATER = DAY0 + 30 * 86_400_000
tk = lambda ps, t0, dt=1000: (np.array([t0 + i * dt for i in range(len(ps))]), np.array(ps, float))


def test_frozen_constants():
    assert (X.HYB_ARM, X.HYB_FLOOR, X.HYB_N, X.HYB_DAYS, X.HYB_MIN_D) == (1.0, 0.2, 40, 20, 0.30)
    assert "Δ −0.18 [−0.47, +0.11]" in X.HYB_YEAR and "SAVED 45 (+144)" in X.HYB_YEAR and X.GC_FROM == "2026-10-07T00:00:00"


def test_hyb_equals_the_lock_when_the_peak_never_reaches_plus_1_or_runs_straight_past_3():
    t0 = DAY0
    for ps in ([100, 100.8, 99, 96.8], [100, 100.5, 100.9, 101.0, 97.0], [100, 104, 107, 104.8]):
        h, l = X.walk_ticks_hyb(*tk(ps, t0), 100, t0), X._walk_ticks(*tk(ps, t0), 100, t0)
        assert h[:2] == l[:2]
    # the prior-print rule: the print that first reaches +1 cannot itself exit at the +0.2 floor
    h = X.walk_ticks_hyb(*tk([100, 99.0, 101.2], t0), 100, t0)
    assert h[2] == "open"


def test_orca_1805_shape_hyb_saves_the_stopped_lock():
    """live ORCA 10-06 18:05: peak +1.38 net, then down through −3 (live −3.01) — HYB exits at the +0.2 floor print."""
    t0 = DAY0
    ps = [100, 100.8, 101.47, 100.6, 100.25, 99.0, 96.9]
    h, l = X.walk_ticks_hyb(*tk(ps, t0), 100, t0), X._walk_ticks(*tk(ps, t0), 100, t0)
    assert h[2] == "+0.2 floor" and abs(h[0] - (0.25 - X.FEE)) < 1e-9 and l[2] == "stop" and l[0] < -3


def _allr(rows):
    return pd.DataFrame([dict(k=X._iso(o), pair=p, sleeve=s, entry=100.0, LOCK2=None, final=True) for o, p, s in rows])


def test_hyb_run_ticks_split_anatomy_and_no_reprice(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "HYB_CSV", str(tmp_path / "hyb.csv"))
    o1, o2, o3, o4 = DAY0 + X.H, DAY0 + 26 * X.H, DAY0 + 50 * X.H, DAY0 - 3 * X.H
    paths = {"SAVUSDT": [100, 101.5, 100.25, 96.0], "CUTUSDT": [100, 101.5, 100.1, 104, 106, 104.0], "WUSDT": [100, 96.5], "REFUSDT": [100, 96.5]}
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", *tk(paths[pair], t0, 60_000)))
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s + i * X.MIN, 100.0, 100.0, 100.0, 100.0, 1.0] for i in range(min(int((e - s) // X.MIN), 1500))])
    allr = _allr([(o1, "SAVUSDT", "LONG"), (o2, "CUTUSDT", "LONG"), (o3, "WUSDT", "WIDE"), (o4, "REFUSDT", "LONG")])
    F = pd.DataFrame(dict(k=allr.k, pair=allr.pair, entry_frenzy_adx_delta=[7.1, -0.5, 6.0, 1.0], entry_frenzy_di_spread=[36.5, 9.2, 39.0, 2.0]))
    hmap, L = X.hyb_run(LATER, F, allr)
    h = pd.read_csv(X.HYB_CSV).set_index("pair")
    assert h.loc["SAVUSDT"].HYB_how == "+0.2 floor" and abs(h.loc["SAVUSDT"].HYB_R - (0.25 - X.FEE)) < 1e-9 and h.loc["SAVUSDT"].LOCK2_R < -3
    assert h.loc["CUTUSDT"].HYB_how == "+0.2 floor" and abs(h.loc["CUTUSDT"].LOCK2_R - (4.0 - X.FEE)) < 1e-9          # the lock's runner is cut
    assert h.loc["SAVUSDT"].strong and not h.loc["CUTUSDT"].strong and pd.isna(h.loc["WUSDT"].strong)                    # strong = LONG stamps only
    assert all(h.px_src == "tick") and all(h.final) and abs(h.loc["SAVUSDT"].d - (h.loc["SAVUSDT"].HYB_R - h.loc["SAVUSDT"].LOCK2_R)) < 1e-12
    lg = h[(h.sleeve == "LONG") & h.cohort.astype(bool)]
    assert X.hyb_check(lg)[0] == "collecting" and X.hyb_anatomy(lg)[0] == 1 and X.hyb_anatomy(lg)[2] == 1   # SAVED SAV · CUT CUT
    assert "2/40 fills · 2/20 days" in "\n".join(L)
    assert hmap[(X._iso(o1), "SAVUSDT")][0] is not None                                  # the 1m column for the table
    monkeypatch.setattr(X, "_hyb_price", lambda *a: pytest.fail("a final row was re-priced"))
    X.hyb_run(LATER, F, allr)


def test_hyb_provisional_on_1m_before_the_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "HYB_CSV", str(tmp_path / "hyb.csv"))
    o = DAY0 + X.H
    m1 = [[o, 100, 101.5, 100, 101.2, 1], [o + X.MIN, 101.2, 101.3, 99.0, 99.5, 1]]
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [b for b in m1 if b[0] < e])
    monkeypatch.setattr(X, "_ticks", lambda *a: pytest.fail("no ticks before the 12 h horizon"))
    hmap, L = X.hyb_run(o + 10 * X.MIN, None, _allr([(o, "AUSDT", "LONG")]))
    r = pd.read_csv(X.HYB_CSV).iloc[0]
    assert not r.final and r.px_src == "1m" and r.HYB_how == "+0.2 floor" and abs(r.HYB_R - 0.2) < 1e-9 and r.LOCK2_1m == r.LOCK2_R


def test_hyb_validation_quarantines(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "HYB_CSV", str(tmp_path / "hyb.csv"))
    pd.DataFrame([dict(k=X._iso(DAY0), pair="A", sleeve="LONG", final=True, ver=1, HYB_R=0.11, LOCK2_R=-3.09, d=1.0, HYB_how="+0.2 floor", HYB_1m=0.11)]
                 ).to_csv(X.HYB_CSV, index=False)
    _, L = X.hyb_run(LATER, None, pd.DataFrame())
    assert len(glob.glob(str(tmp_path / "hyb.csv.*.bad"))) == 1 and "quarantined" in "\n".join(L)


def test_hyb_column_in_the_exit_table(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "CSV", str(tmp_path / "exits.csv")); monkeypatch.setattr(X, "HYB_CSV", str(tmp_path / "hyb.csv"))
    o = DAY0 + X.H
    F = pd.DataFrame(dict(opened_at=[X._iso(o)], k=[X._iso(o)], pair=["ORCAUSDT"], direction=["LONG"], entry_strategy=["FRENZY_LONG"], status=["CLOSED"],
                          entry_price=[100.0], pnl_percentage=[-3.01], entry_frenzy_adx_delta=[7.1], entry_frenzy_di_spread=[36.5]))
    monkeypatch.setattr(X, "_fills", lambda: F)
    monkeypatch.setattr(X, "_journal", lambda now: None)
    monkeypatch.setattr(X, "_cfg", lambda: X.SimpleNamespace())
    monkeypatch.setattr(X, "_extras", lambda *a: [])
    monkeypatch.setattr(X, "_price", lambda r, th, now, first, J=None: dict(k=r.k, pair=r.pair, sleeve="LONG", day=r.k[:10], entry=100.0, first=True,
                        ver=X.VER, closed=True, actual=-3.01, atr_entry=1.0, vs_vwap=1.0, above_share=None, LOCK2=-3.09, LOCK3=-3.09, EMA20=-3.09,
                        EMA50=-3.09, FIX3=-3.0, atr_chg30=None, code=None, final=True))
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s, 100, 101.5, 100, 101.2, 1], [s + X.MIN, 101.2, 101.3, 96.0, 96.5, 1]])
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", *tk([100, 101.5, 100.25, 96.0], t0, 60_000)))
    txt = "\n".join(X.run(LATER))
    assert "| LOCK2 | HYB | LOCK3 |" in txt and "| -3.09 | +0.20 | -3.09 |" in txt and "HYBRID_EXIT (HYB = V3" in txt


def test_hyb_rate_limit_stops_and_long_cohort_first(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "HYB_CSV", str(tmp_path / "hyb.csv"))
    seen = []

    def boom(src, now, b):
        seen.append(src["pair"])
        raise X.urllib.error.HTTPError("u", 429, "Too Many Requests", None, None)
    monkeypatch.setattr(X, "_hyb_price", boom)
    allr = _allr([(DAY0 - 3 * X.H, "REFUSDT", "LONG"), (DAY0 + X.H, "WUSDT", "WIDE"), (DAY0 + 2 * X.H, "L1USDT", "LONG"), (DAY0 + 3 * X.H, "L2USDT", "LONG")])
    _, L = X.hyb_run(LATER, None, allr)
    assert seen == ["L2USDT"] and "rate limit — stopped" in "\n".join(L) and "3 fill(s) left for the next run" in "\n".join(L)
    order = []
    monkeypatch.setattr(X, "_hyb_price", lambda src, now, b: (order.append(src["pair"]), (_ for _ in ()).throw(ValueError("x")))[1])
    X.hyb_run(LATER, None, allr)
    assert order == ["L2USDT", "L1USDT", "WUSDT", "REFUSDT"]                     # the FRENZY_LONG cohort first, newest first


def test_hyb_tick_path_still_open_falls_through_and_needs_both_exits(tmp_path, monkeypatch):
    """ticks that end with HYB out but the lock still open → never final on ticks; the 1m path decides and finalises by age."""
    o = DAY0 + X.H
    monkeypatch.setattr(X, "_ticks", lambda pair, t0, t1, now, b: ("ok", *tk([100, 101.5, 100.1, 100.5], t0, 60_000)))
    m1 = [[o + i * X.MIN, 100.0, 100.0, 100.0, 100.0, 1.0] for i in range(800)]
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [b for b in m1 if s <= b[0] < e])
    r = X._hyb_price(dict(k=X._iso(o), pair="AUSDT", entry=100.0, sleeve="LONG"), o + 13 * X.H, {})
    assert not r["final"] and r["tick_state"] == "open" and r["px_src"] == "1m"
    r = X._hyb_price(dict(k=X._iso(o), pair="AUSDT", entry=100.0, sleeve="LONG"), LATER, {})
    assert r["final"] and r["px_src"].startswith("1m") and r["HYB_how"] == "12 h cap"
