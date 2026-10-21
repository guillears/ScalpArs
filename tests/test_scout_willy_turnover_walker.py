"""🚶 DECISION_LOG 336 (operator-approved): the WILLY turnover v2 gate compares WALKER vs WALKER — the blocked entries and the let-through
fills are both priced under TODAY's live WILLY exit (+tp / −stop / cap, late TP DL 297, resting TP DL 296) with
scout_willy_timecap.evaluate_late(book="live"), the walker build_willy_exit_cf.py uses. The frozen sample and the bar are unchanged; the
old as-booked read is printed for information only."""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
sys.path.insert(0, HERE)
import scout_frenzy_exits as SF  # noqa: E402

MIN = 60_000
WR = dict(tp=1.25, sl=4.0, cap=90, late=(30.0, 0.25), rtp=True, label="+1.25 / −4 / 90 min · late +0.25 from min 30 · resting TP")


def _blk(n, pct, wpct, days=10, wfinal=True):
    return pd.DataFrame([dict(t=f"2026-10-{10 + (i % days):02d}T17:00:{i % 60:02d}", pair=f"P{i}USDT", gate="FRENZY_WILLY_TURNOVER",
                              pct=pct, how="take profit", final=True, wpct=wpct, wfinal=wfinal) for i in range(n)])


def _lt(n, pct, wpct, wfinal=True):
    return pd.DataFrame([dict(t=f"2026-10-{10 + (i % 10):02d}T16:00:00", pair=f"L{i}USDT", pct=pct, booked=pct, wpct=wpct, wfinal=wfinal,
                              wwhy="take profit") for i in range(n)])


# ─────────────── the verdict reads the walker columns on BOTH sides ───────────────
def test_verdict_is_walker_vs_walker_not_booked():
    """the deep-study shape: as booked +0.216 vs +0.103 would REVERT; walker +0.199 vs +0.263 is the hold zone."""
    s = SF.tb2_step({}, _blk(30, 0.216, 0.199), _lt(36, 0.103, 0.263))
    assert s.get("held_at_30") and not s.get("verdict") and s["ruler"] == SF.TB_RULER
    assert abs(s["at30"]["blocked"] - 0.199) < 1e-9 and abs(s["at30"]["let_through"] - 0.263) < 1e-9 and s["at30"]["n_lt"] == 36


def test_bar_unchanged_on_walker_prices():
    assert SF.TB2_N == 30 and SF.TB2_N2 == 60 and SF.TB2_DAYS == 10 and SF.TB2_MARGIN == 0.20
    assert SF.tb2_step({}, _blk(30, -9.0, 0.30), _lt(36, 9.0, 0.30))["verdict"].startswith("REVERT")      # ≥ (tie) → REVERT
    assert SF.tb2_step({}, _blk(30, 9.0, 0.05), _lt(36, -9.0, 0.30))["verdict"] == "filter confirmed"    # ≤ lt − 0.20 (0.05 ≤ 0.10)
    assert SF.tb2_step({}, _blk(29, 9.0, 0.9), _lt(36, 0.0, 0.0)) == {}                                   # < 30 waits
    assert SF.tb2_step({}, _blk(30, 9.0, 0.9, days=9), _lt(36, 0.0, 0.0)) == {}                           # < 10 days waits


def test_provisional_walker_price_on_either_side_holds_the_read():
    b = _blk(30, 0.5, 0.5)
    b.loc[3, "wfinal"] = False
    assert SF.tb2_step({}, b, _lt(36, 0.1, 0.1)) == {}
    lt = _lt(36, 0.1, 0.1)
    lt.loc[5, "wfinal"] = False
    assert SF.tb2_step({}, _blk(30, 0.5, 0.5), lt) == {}
    lt2 = _lt(36, 0.1, 0.1)
    lt2.loc[5, "wpct"] = np.nan                                   # walked-but-unpriced let-through → wait, never a partial mean
    assert SF.tb2_step({}, _blk(30, 0.5, 0.5), lt2) == {}
    b2 = _blk(30, 0.5, 0.5)
    b2.loc[0, "wpct"] = np.nan
    assert SF.tb2_step({}, b2, _lt(36, 0.1, 0.1)) == {}


def test_unpriceable_let_through_is_listed_apart_not_counted():
    lt = pd.concat([_lt(36, 0.1, 0.1), pd.DataFrame([dict(t="2026-10-12T19:00:00", pair="ZUSDT", pct=-4.0, booked=-4.0, wpct=np.nan,
                                                            wfinal=np.nan, wwhy="UNPRICEABLE")])], ignore_index=True)
    s = SF.tb2_step({}, _blk(30, 0.5, 0.5), lt)
    assert s["verdict"].startswith("REVERT") and s["n_lt"] == 36


def test_legacy_hold_is_reread_legacy_verdict_stays_frozen():
    legacy_hold = dict(held_at_30=True, at30=dict(n=30, days=10, blocked=0.0, let_through=0.1))   # pre-336 ruler, no 'ruler' key
    s = SF.tb2_step(legacy_hold, _blk(30, 0.0, 0.5), _lt(36, 0.1, 0.1))
    assert s["verdict"].startswith("REVERT") and s["ruler"] == SF.TB_RULER and s["n"] == 30   # read at 30 again, not at 60
    legacy_verdict = dict(verdict="filter confirmed", n=30, days=10, blocked=-1.0, let_through=0.1)
    assert SF.tb2_step(legacy_verdict, _blk(30, 0.0, 5.0), _lt(36, 0.1, 0.1)) == legacy_verdict


# ─────────────── the walker rule + evaluation (build_willy_exit_cf parity) ───────────────
def test_walker_rule_reads_the_live_exit():
    th = SimpleNamespace(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_willy_late_tp_minutes=30.0,
                         frenzy_willy_late_tp_pct=0.25, resting_tp_willy_enabled=True)
    wr = SF.tb_walker_rule(th)
    assert (wr["tp"], wr["sl"], wr["cap"], wr["late"], wr["rtp"]) == (1.25, 4.0, 90, (30.0, 0.25), True)
    off = SF.tb_walker_rule(SimpleNamespace(**{**vars(th), "frenzy_willy_late_tp_pct": 0.0, "resting_tp_willy_enabled": False}))
    assert off["late"] is None and not off["rtp"] and "late" not in off["label"]
    real = SF.tb_walker_rule()                                     # trading_config.json — the live exit, never hard-coded
    import json
    th_real = json.load(open(os.path.join(HERE, "trading_config.json")))["thresholds"]
    assert real["tp"] == float(th_real["frenzy_willy_tp_pct"]) and real["cap"] == int(th_real["frenzy_willy_max_hold_minutes"])
    assert real["rtp"] == bool(th_real.get("resting_tp_willy_enabled"))


def _P(mins, prices, te=0):
    t = np.array([te + int(m * MIN) for m in mins], dtype="int64")
    p = np.array(prices, dtype=float)
    return dict(t=t, hi=p, lo=p, cl=p, op=np.full(len(t), np.nan), bar=np.zeros(len(t), bool), end_ms=te + 91 * MIN, n_tick=len(t),
                n_bar=0, te=te, tick_end=te + 91 * MIN, lim=te + 91 * MIN)


def test_walker_eval_resting_tp_late_tp_and_print_booking():
    import scout_frenzy_willy_exit as FWX
    P = _P([1, 5, 6], [0.2, 1.31, 0.5])                            # +1.31 print through the +1.25 line at minute 5
    assert SF.tb_walker_eval(P, WR)["pct"] == 1.25                 # resting maker TP books exactly the line (DL 296)
    assert abs(FWX.evaluate(P, 1.25, 4.0, 90)["pct"] - 1.31) < 1e-12   # the scout pricer: first print through (old read)
    L = _P([1, 10, 29, 31, 40], [0.1, 0.2, 0.1, 0.30, 0.9])         # first event after min 30 is already ≥ +0.25 → that print (market close)
    ev = SF.tb_walker_eval(L, WR)
    assert ev["why"] == "LATE_TP" and abs(ev["pct"] - 0.30) < 1e-12
    L2 = _P([1, 31, 35], [0.1, 0.1, 0.27])                         # a later touch of the late line → exactly the line
    assert SF.tb_walker_eval(L2, WR)["pct"] == 0.25
    S = _P([1, 2], [-1.0, -4.05])
    assert SF.tb_walker_eval(S, WR)["why"] == "STOP" and abs(SF.tb_walker_eval(S, WR)["pct"] + 4.05) < 1e-12
    plain = dict(WR, late=None, rtp=False)                         # late + resting off → the print-booking walker
    assert abs(SF.tb_walker_eval(P, plain)["pct"] - 1.31) < 1e-12


def test_tb_walk_entry_mode_walks_the_booked_entry(monkeypatch):
    """a let-through fill: E / te given → no refusal-print search; the same path gives the scout price AND the walker price."""
    import scout_frenzy_willy_exit as FWX
    seen = {}
    P = _P([1, 5], [0.3, 1.31], te=1_792_000_000_000)

    def fake_path(pair, te, E, hz, tcache=None, k1m=None):
        seen.update(pair=pair, te=te, E=E, hz=hz)
        return P
    monkeypatch.setattr(FWX, "load_path", fake_path)
    monkeypatch.setattr(FWX.WT, "_tick_file", lambda pair, d: "on-disk")
    monkeypatch.setattr(FWX.WT, "_load_day", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no refusal search in entry mode")))
    r = SF.tb_walk("XUSDT", P["te"], (1.25, -4.0, 90), P["te"] + 200 * MIN, {"dl": 0}, lambda a, b: None, {}, wr=WR, entry=(0.5, P["te"]))
    assert seen == dict(pair="XUSDT", te=P["te"], E=0.5, hz=90)
    assert r["wpct"] == 1.25 and r["wwhy"] == "take profit" and r["wsrc"] == "ticks" and r["wfinal"] is True
    assert abs(r["pct"] - 1.31) < 1e-12 and r["final"] is True


# ─────────────── stores ───────────────
def test_tb_price_fills_walker_columns_without_rewriting_a_final_scout_price(monkeypatch):
    calls = []
    monkeypatch.setattr(SF, "tb_walk", lambda *a, **k: calls.append(k.get("wr")) or dict(pct=9.9, how="cap", src="1m", final=False, entry=1.0,
                                                                                       wpct=0.25, wwhy="late TP", wsrc="ticks", wfinal=True))
    rows = pd.DataFrame([dict(t="2026-10-12T10:30:07", pair="GTCUSDT", gate="FRENZY_WILLY_TURNOVER", trig="A", R=1.9, price=0.3, pct=1.2563,
                              how="take profit", tries=0, src="ticks", final=True, entry=0.3)])
    out = SF.tb_price(rows, SF._ms("2026-10-21T00:00:00"), budget={"dl": 0}, wr=WR)
    r = out.iloc[0]
    assert r.pct == 1.2563 and r.how == "take profit" and r.src == "ticks"          # the final scout price is never rewritten
    assert r.wpct == 0.25 and r.wwhy == "late TP" and r.wfinal is True and r.wrule == WR["label"]
    calls.clear()
    SF.tb_price(out, SF._ms("2026-10-21T00:00:00"), budget={"dl": 0}, wr=WR)
    assert calls == []                                                                # final on both rulers → never walked again
    SF.tb_price(out, SF._ms("2026-10-21T00:00:00"), budget={"dl": 0}, wr=dict(WR, tp=1.5, label="other exit"))
    assert len(calls) == 1                                                            # the live exit changed → walked again


def test_tb_lt_price_store_reuse_rule_change_fallback_and_unpriceable(monkeypatch, tmp_path):
    store = str(tmp_path / "lt.csv")
    calls = []

    def fake_walk(pair, t, rule, now_ms, budget, kl, tcache=None, wr=None, entry=None):
        calls.append((pair, entry))
        return dict(entry=entry[0], wpct=-4.0 if pair == "STOPUSDT" else 1.25, wwhy="stop" if pair == "STOPUSDT" else "take profit",
                    wsrc="ticks", wfinal=True)
    monkeypatch.setattr(SF, "tb_walk", fake_walk)
    w = pd.DataFrame([dict(k="2026-10-09T08:45:09", pair="KAIAUSDT", closed=True, actual=1.01, reason="FRENZY_TP", entry=0.11),
                      dict(k="2026-10-12T09:00:00", pair="STOPUSDT", closed=True, actual=-3.9, reason="STOP_LOSS", entry=np.nan),
                      dict(k="2026-10-13T09:00:00", pair="NOENTRYUSDT", closed=True, actual=0.5, reason="FRENZY_TP", entry=np.nan),
                      dict(k="2026-10-14T09:00:00", pair="OPENUSDT", closed=False, actual=np.nan, reason="", entry=1.0),
                      dict(k="2026-10-07T09:00:00", pair="EARLYUSDT", closed=True, actual=1.0, reason="FRENZY_TP", entry=1.0)])
    ent = {("2026-10-12T09:00:00", "STOPUSDT"): 2.0}                                   # the archive fallback (BASELINE*_orders)
    now = SF._ms("2026-10-21T00:00:00")
    out = SF.tb_lt_price(w, now, budget={"dl": 0}, wr=WR, store=store, entries=ent)
    assert list(out.pair) == ["KAIAUSDT", "STOPUSDT", "NOENTRYUSDT"]                  # closed, ≥ TB2_FROM only
    assert ("STOPUSDT", (2.0, SF._ms("2026-10-12T09:00:00"))) in calls                # walked from the archived entry price
    assert out.set_index("pair").wpct.to_dict()["STOPUSDT"] == -4.0
    assert np.isnan(out.set_index("pair").wpct["NOENTRYUSDT"]) and out.set_index("pair").tries["NOENTRYUSDT"] == 1
    calls.clear()
    SF.tb_lt_price(w, now, budget={"dl": 0}, wr=WR, store=store, entries=ent)
    out3 = SF.tb_lt_price(w, now, budget={"dl": 0}, wr=WR, store=store, entries=ent)
    assert calls == []                                                                # final walks reused; no entry → never walked
    assert out3.set_index("pair").wwhy["NOENTRYUSDT"] == "UNPRICEABLE"                # 3 failed runs → listed apart
    SF.tb_lt_price(w, now, budget={"dl": 0}, wr=dict(WR, label="other exit"), store=store, entries=ent)
    assert {c[0] for c in calls} == {"KAIAUSDT", "STOPUSDT"}                          # exit changed → walked again


def test_lt_parity_and_report_lines():
    ltw = pd.DataFrame([dict(t="2026-10-16T00:05:09", pair="ICNTUSDT", booked=-3.86, reason="STOP_LOSS", wpct=-4.06, wwhy="stop", wfinal=True),
                        dict(t="2026-10-17T16:55:09", pair="BATUSDT", booked=0.27, reason="WILLY_LATE_TP", wpct=0.32, wwhy="late TP", wfinal=True),
                        dict(t="2026-10-19T18:45:09", pair="PHAUSDT", booked=1.26, reason="FRENZY_TP", wpct=1.25, wwhy="take profit", wfinal=True),
                        dict(t="2026-10-10T08:00:00", pair="OLDUSDT", booked=-2.44, reason="MAX_HOLD_TIME", wpct=0.25, wwhy="late TP", wfinal=True)])
    par = SF.tb_lt_parity(ltw, SF._ms("2026-10-15T16:25:00"))
    assert par["n"] == 3 and par["same"] == 3 and abs(par["mad"] - (0.20 + 0.05 + 0.01) / 3) < 1e-9
    assert SF.tb_lt_parity(ltw, None) is None
    rows = _blk(3, 0.216, 0.199).assign(trig="A", R=2.0, price=1.0, src="ticks", entry=1.0)
    w = pd.DataFrame([dict(k=t, pair=p, actual=b, mcap=np.nan) for t, p, b in zip(ltw.t, ltw.pair, ltw.booked)])
    txt = "\n".join(SF.tb_lines(rows, w, {}, ltw=ltw, par=par, wr_label=WR["label"]))
    assert "Walker read (the verdict)" in txt and "Old as-booked read (info only" in txt and "same exit class 3/3" in txt
    assert "blocked 3 · 100 % · +0.199 %" in txt and "let-through 4 · 75 % · -0.560 %" in txt
    assert "N per side: blocked 3" in txt


def test_willy_rows_carry_the_entry_and_merge_never_wipes_it():
    o = pd.DataFrame([dict(opened_at="2026-10-09T08:45:09", pair="KAIAUSDT", direction="LONG", entry_strategy="FRENZY_WILLY", status="CLOSED",
                           pnl_percentage=1.01, pnl=125.5, close_reason="FRENZY_TP", entry_price=0.11)])
    r = SF.willy_rows(o)
    assert r.entry.iloc[0] == 0.11
    newer = r.assign(entry=np.nan)                                 # an export without the column
    assert SF.willy_merge(r, newer).entry.iloc[0] == 0.11


def test_tb_lt_price_inside_window_is_pending_not_a_try(monkeypatch, tmp_path):
    """review fix: a let-through fill still inside its exit window (max hold) is PENDING — runs then never count toward UNPRICEABLE."""
    store = str(tmp_path / "lt.csv")
    monkeypatch.setattr(SF, "tb_walk", lambda *a, **k: (_ for _ in ()).throw(AssertionError("never walked inside the window")))
    w = pd.DataFrame([dict(k="2026-10-20T23:30:00", pair="FRESHUSDT", closed=True, actual=0.3, reason="FRENZY_TP", entry=1.0),
                      dict(k="2026-10-20T23:30:00", pair="FRESHNOENTRYUSDT", closed=True, actual=0.3, reason="FRENZY_TP", entry=np.nan)])
    now = SF._ms("2026-10-21T00:00:00")                                               # 30 min after the fill, window = 90 (+1) min
    for _ in range(SF.TB_MAX_TRIES + 2):
        out = SF.tb_lt_price(w, now, budget={"dl": 0}, wr=WR, store=store, entries={})
    o = out.set_index("pair")
    for p in ("FRESHUSDT", "FRESHNOENTRYUSDT"):
        assert o.wwhy[p] != "UNPRICEABLE" and (pd.isna(o.tries[p]) or int(o.tries[p]) == 0)
    # once the window has passed, failed runs count again (no entry → UNPRICEABLE after TB_MAX_TRIES)
    later = now + (WR["cap"] + 2) * MIN
    for _ in range(SF.TB_MAX_TRIES):
        out = SF.tb_lt_price(w[w.pair == "FRESHNOENTRYUSDT"], later, budget={"dl": 0}, wr=WR, store=store, entries={})
    assert out.set_index("pair").wwhy["FRESHNOENTRYUSDT"] == "UNPRICEABLE"


def test_willy_deploy_ms_not_found_is_loud_never_now(monkeypatch):
    """review fix (docstring): a missing DL 251 floor → NOT_FOUND_MS + the loud NOT-EVALUABLE warning, never NOW."""
    import scout_revert_gates as RG
    monkeypatch.setattr(RG, "deploy_ms", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no commit")))
    warned = []
    monkeypatch.setattr(SF._GK, "warn_not_evaluable", lambda label, dl=None, stream=None: warned.append((label, dl)))
    assert SF.willy_deploy_ms() == SF.NOT_FOUND_MS
    assert warned and warned[0][1] == 251
    assert "NOT_FOUND_MS" in SF.willy_deploy_ms.__doc__ and "never NOW" in SF.willy_deploy_ms.__doc__
