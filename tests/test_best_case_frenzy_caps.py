"""⏱ Best-case ①c (DECISION_LOG 497): the FRENZY shorter-cap pricing step (scripts/best_case/frenzy_caps.py) — pure walk math."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
import frenzy_caps as FC  # noqa: E402
import scout_frenzy_duration as FD  # noqa: E402

MIN = 60_000
T0 = 1_800_000_000_000 - (1_800_000_000_000 % MIN)
TE = T0 + 20_000                       # entry 20 s into a minute → the walk starts at the next full minute
FEE = FD.TAKER


def bars(rows, start=T0 + MIN):
    """rows = [(h, l, c)] per minute from `start` (the first full minute after TE)."""
    return pd.DataFrame({"open_time": [start + i * MIN for i in range(len(rows))],
                         "h": [r[0] for r in rows], "l": [r[1] for r in rows], "c": [r[2] for r in rows]})


def px(net, exit_fee):
    return 100.0 * FC.ratio_for(net, FEE, exit_fee)


def test_ratio_for_inverts_the_engine_net():
    for net, ef in ((3.0, FD.MAKER), (-4.0, FD.TAKER), (0.25, FD.MAKER)):
        r = FC.ratio_for(net, FEE, ef)
        assert (r - 1) * 100 - FEE - ef * r == pytest.approx(net)


def test_walk_tp_stop_and_same_bar_conservative():
    tp_px, st_px = px(3.0, FD.MAKER), px(-3.0, FD.TAKER)
    b = bars([(100.5, 99.5, 100.0), (tp_px + 0.01, 99.8, 101.0)] + [(100.2, 99.9, 100.0)] * 20)
    assert FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 15)[::2] == (3.0, "TP")
    b = bars([(100.5, st_px - 0.01, 99.0)] + [(100.2, 99.9, 100.0)] * 20)
    assert FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 15)[::2] == (-3.0, "STOP")
    b = bars([(tp_px + 1, st_px - 1, 100.0)] + [(100.2, 99.9, 100.0)] * 20)     # one bar touches both → the stop
    assert FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 15)[2] == "STOP"


def test_walk_cap_closes_at_the_last_bar_closed_by_the_cap_and_missing_minute_is_none():
    b = bars([(100.3, 99.9, 100.0 + i * 0.01) for i in range(40)])
    pct, hold, why = FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 15)
    fire = FD.cap_fires_ms(TE, 15)
    assert why == "CAP" and hold == 15 and pct == pytest.approx(FD.net_pct(100.0, FD.price_at(b, fire), "LONG", FEE))
    assert FC.walk(b.drop(index=5), TE, 100.0, FEE, 3.0, 3.0, 15) is None      # a hole before the cap → unpriced, never guessed
    assert FC.walk(None, TE, 100.0, FEE, 3.0, 3.0, 15) is None


def test_walk_late_tp_line_after_late_minutes_only():
    late_px = px(0.25, FD.MAKER)
    rows = [(late_px + 0.01, 99.9, 100.0)] * 60                                 # touches +0.25 every minute, never the +5 target
    r = FC.walk(bars(rows), TE, 100.0, FEE, 5.0, 4.0, 90, late_min=30, late_tp=0.25)
    assert r[2] == "LATE_TP" and r[0] == 0.25 and r[1] > 29
    r = FC.walk(bars(rows), TE, 100.0, FEE, 5.0, 4.0, 15, late_min=30, late_tp=0.25)
    assert r[2] == "CAP"                                                        # a 15-min cap fires before the late line exists


def test_paired_cap_only_moves_fills_still_open():
    """the same path walked to the live cap and to X: a fill whose walk exits before X is unchanged (Δ 0)."""
    tp_px = px(3.0, FD.MAKER)
    b = bars([(100.2, 99.9, 100.0)] * 5 + [(tp_px + 0.01, 99.9, 103.0)] + [(100.2, 99.9, 100.0)] * 800)
    base = FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 720)
    assert base[2] == "TP"
    assert FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 15)[0] - base[0] == 0           # TP at minute ~6 < 15: unchanged
    assert FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 2)[2] == "CAP"                 # a 2-min cap would have cut it


def test_match_and_willy_target():
    assert FC.match(-3.0, "STOP", -3.43, 3.0) and FC.match(1.5, "TP", 1.52, 4.0)
    assert not FC.match(-4.0, "STOP", 1.5, 4.0) and not FC.match(3.0, "TP", -0.63, 3.0)
    # (review) a known hold under today's exit must match too: a TP the 1m walk finds 20 min late is a miss
    assert FC.match(2.0, "TP", 2.0, 4.0, walk_hold=3, ref_hold=2) and not FC.match(2.0, "TP", 2.0, 4.0, walk_hold=22, ref_hold=2)
    k = FC.MX.key("2026-10-11T18:10:08", "WUSDT")
    wcf = {k: (2.63, 24.7)}
    assert FC.willy_target({"frenzy_willy_tp_target_pct": 2.4, "opened_at": "2026-10-11T18:10:08", "pair": "WUSDT"}, wcf) == (2.4, 24.7)
    assert FC.willy_target({"opened_at": "2026-10-11 18:10:08", "pair": "WUSDT"}, wcf) == (2.63, 24.7)      # the master re-walk target
    assert FC.willy_target({"opened_at": "2026-10-11T18:10:08", "pair": "WUSDT"}, {k: (float("nan"), 12.0)}) == (float("inf"), 12.0)
    assert FC.willy_target({"opened_at": "2026-10-12T00:00:00", "pair": "XUSDT"}, wcf) == (None, None)     # no target anywhere → unpriced


def test_late_line_only_on_bars_starting_after_the_late_minute():
    late_px = px(0.25, FD.MAKER)
    te = T0                                                                     # entry on a minute boundary → first bar starts at T0 + 1 min
    rows = [(100.1, 99.9, 100.0)] * 28 + [(late_px + 0.01, 99.9, 100.0)] + [(100.1, 99.9, 100.0)] * 70   # bar 28 opens at minute 29
    r = FC.walk(bars(rows), te, 100.0, FEE, 5.0, 4.0, 90, late_min=30, late_tp=0.25)
    assert r[2] == "CAP"                                                        # the high sat in the bar [29, 30) min → still the +5 target


def test_exit_spec_stop0_guard(tmp_path):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"thresholds": {"frenzy_stop_pct": 0, "frenzy_long_stop_pct": 0, "frenzy_willy_stop_pct": 0,
                                            "frenzy_lite_tp_pct": 3, "frenzy_tp_pct": 1.5}}))
    assert FC.exit_spec("FRENZY_LITE", str(p))["sl"] == 3.0 and FC.exit_spec("FRENZY_LONG", str(p))["sl"] == 3.0
    assert FC.exit_spec("FRENZY_WILLY", str(p))["sl"] == 4.0


def test_exit_spec_reads_the_live_config():
    import json
    with open(FD.CONFIG) as f:
        th = json.load(f)["thresholds"]
    assert FC.exit_spec("FRENZY_LITE")["tp"] == float(th["frenzy_lite_tp_pct"])
    lg = FC.exit_spec("FRENZY_LONG")
    assert lg["tp"] == float(th["frenzy_tp_pct"]) and lg["sl"] == float(th.get("frenzy_long_stop_pct") or th["frenzy_stop_pct"])
    w = FC.exit_spec("FRENZY_WILLY")
    assert w["tp"] is None and w["sl"] == float(th["frenzy_willy_stop_pct"]) and w["late_tp"] == float(th["frenzy_willy_late_tp_pct"])


def test_best_dcr_and_runner_wire_the_section():
    src = open(os.path.join(ROOT, "scripts", "best_case", "best_dcr.py")).read()
    assert "frenzy_caps.csv" in src and "best_caps.csv" in src and "unpriced FRENZY fills" in src
    run = open(os.path.join(ROOT, "scripts", "best_case", "run_best_case.py")).read()
    assert 'step("frenzy_caps.py", exp, "--fetch")' in run and "L += caps_section()" in run and "L += late_section()" in run
    assert "FCm.B0" in src and "FCm.PARITY_MIN" in src and "m_changed > 0" in src and "DLT[col] = da" in src and "best_late.csv" in src
    assert run.index('step("frenzy_caps.py"') < run.index('step("best_dcr.py"')


def test_price_rows_real_walk_miss_and_known_exit(monkeypatch):
    """price_rows end to end on a synthetic path: REAL keeps a fill closed before X; WALK = base + (walk_X − walk_base); a walk that does
    not reproduce the base is kept as booked; a WILLY fill whose KNOWN exit (re-walk cf_min) came before X keeps Δ 0 (deep review)."""
    import scout_willy_timecap as WT
    flat = bars([(100.05 + i * 0.002, 99.95 + i * 0.002, 100.0 + i * 0.002) for i in range(900)])   # a slow drift up, no line touched
    monkeypatch.setattr(WT, "load_1m", lambda pair: flat)
    op = pd.Timestamp(TE, unit="ms")
    def row(sl, method, base, hold_min, **kw):
        return dict(sl=sl, method=method, pct=base, pair="AUSDT", direction="LONG", opened_at=op.isoformat(),
                    closed_at=(op + pd.Timedelta(minutes=hold_min)).isoformat(), entry_price=100.0, fee_in=FEE, **kw)
    caps = {s: 720 for s in FD.SLEEVES} | {"FRENZY_WILLY": 90, "FRENZY_WILLY_HT": 90}
    F = pd.DataFrame([row("FRENZY_LITE", "REAL", 2.9, 10),                         # closed at 10 min: CAP15 = booked
                      row("FRENZY_LITE", "WALK", D_flat := FC.walk(flat, TE, 100.0, FEE, 3.0, 3.0, 720)[0], 720),   # walk reproduces: priced
                      row("FRENZY_LITE", "WALK", 3.0, 30)])                         # base +3 TP but the flat walk caps → PARITY MISS
    out = FC.price_rows(F, "MASTER", caps, False, {}, [], TE + 2000 * MIN)
    assert out[0]["CAP15"] == 2.9 and out[0]["method"] == "REAL"
    w15 = FC.walk(flat, TE, 100.0, FEE, 3.0, 3.0, 15)[0]
    assert out[1]["status"] == "PRICED" and out[1]["CAP15"] == pytest.approx(D_flat + (w15 - D_flat))
    assert out[2]["status"] == "PARITY MISS (kept as booked)" and out[2]["CAP15"] == 3.0
    k = FC.MX.key(op.isoformat(), "AUSDT")
    W = pd.DataFrame([row("FRENZY_WILLY", "WALK", FC.walk(flat, TE, 100.0, FEE, 5.0, 4.0, 90, 30, 0.25)[0], 90)])
    ow = FC.price_rows(W, "MASTER", caps, False, {}, [], TE + 2000 * MIN, {k: (5.0, 90.0)})
    assert ow[0]["status"] == "PRICED" and ow[0]["CAP15"] != ow[0]["base"]          # open past 15 min: the cap moves it
    ow = FC.price_rows(W, "MASTER", caps, False, {}, [], TE + 2000 * MIN, {k: (5.0, 12.0)})
    assert ow[0]["status"] == "PARITY MISS (kept as booked)"                           # walk exits at 90 vs a known 12 min → miss


def test_late_variants_only_on_the_long_cap_and_below_the_tp(tmp_path):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"thresholds": {"frenzy_stop_pct": 3, "frenzy_long_stop_pct": 4, "frenzy_lite_tp_pct": 3, "frenzy_tp_pct": 1.5}}))
    lv = FC.late_variants("FRENZY_LITE", 720, str(p))
    assert (60, 2.0) in lv and all(l_ < 3 for _, l_ in lv) and len(lv) == len(FC.LATE_XS) * 5
    assert all(l_ < 1.5 for _, l_ in FC.late_variants("FRENZY_LONG", 720, str(p)))
    assert FC.late_variants("FRENZY_WILLY", 720, str(p)) == [] and FC.late_variants("FRENZY_LITE", 90, str(p)) == []
    assert FC.late_col(120, 2.0) == "LT120_2" and FC.late_col(60, 0.25) == "LT60_0.25"


def test_late_real_books_the_lower_line_only_after_x_and_before_the_real_close():
    l2 = px(2.0, FD.MAKER)
    rows = [(100.1, 99.9, 100.0)] * 70 + [(l2 + 0.01, 99.9, 100.0)] + [(100.1, 99.9, 100.0)] * 300
    b = bars(rows)                                                              # the +2 touch sits in the bar opening at minute 71
    close = TE + 300 * MIN
    assert FC.late_real(b, TE, close, 100.0, FEE, 3.0, 60, 2.0, -3.0) == 2.0    # armed at 60 → a loser rescued at +2
    assert FC.late_real(b, TE, close, 100.0, FEE, 3.0, 120, 2.0, -3.0) == -3.0  # armed after the touch → booked
    assert FC.late_real(b, TE, TE + 50 * MIN, 100.0, FEE, 3.0, 60, 2.0, 3.0) == 3.0   # closed before X → booked
    assert FC.late_real(b, TE, TE + 71 * MIN, 100.0, FEE, 3.0, 60, 2.0, 0.4) == 0.4   # the touch bar not closed before the real exit
    assert FC.late_real(b.drop(index=65), TE, close, 100.0, FEE, 3.0, 60, 2.0, 1.0) is None   # a hole in the span → unpriced
    both = bars([(100.1, 99.9, 100.0)] * 70 + [(l2 + 0.01, px(-3.0, FD.TAKER) - 0.01, 100.0)] + [(100.1, 99.9, 100.0)] * 300)
    assert FC.late_real(both, TE, close, 100.0, FEE, 3.0, 60, 2.0, 1.0) == 1.0      # live rode that wick: min(base, L), never the replica stop
    assert FC.late_real(both, TE, close, 100.0, FEE, 3.0, 60, 2.0, 2.9) == 2.0


def test_walk_late_line_rescues_a_capped_fill_and_cuts_a_later_full_tp():
    l2, tp = px(2.0, FD.MAKER), px(3.0, FD.MAKER)
    rows = [(100.1, 99.9, 100.0)] * 90 + [(l2 + 0.01, 99.9, 100.0)] + [(100.1, 99.9, 100.0)] * 50 + [(tp + 0.01, 99.9, 100.0)] + [(100.1, 99.9, 100.0)] * 700
    b = bars(rows)
    assert FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 720)[2] == "TP"
    r = FC.walk(b, TE, 100.0, FEE, 3.0, 3.0, 720, 60, 2.0)
    assert r[2] == "LATE_TP" and r[0] == 2.0                                    # the winner gives 1 point back


def test_price_rows_late_columns_real_walk_miss_and_no_willy(monkeypatch):
    """(review) price_rows end to end for LT<X>_<L>: REAL → late_real on the real path; WALK → base + (late walk − base walk); a parity miss
    keeps base; WILLY rows get no LT column; a late hole sets late_status only (①c status stays PRICED)."""
    import scout_willy_timecap as WT
    l2 = px(2.0, FD.MAKER)
    path = bars([(100.1, 99.9, 100.0)] * 70 + [(l2 + 0.01, 99.9, 100.0)] + [(100.1, 99.9, 100.0)] * 800)
    monkeypatch.setattr(WT, "load_1m", lambda pair: path)
    op = pd.Timestamp(TE, unit="ms")
    def row(sl, method, base, hold_min):
        return dict(sl=sl, method=method, pct=base, pair="AUSDT", direction="LONG", opened_at=op.isoformat(),
                    closed_at=(op + pd.Timedelta(minutes=hold_min)).isoformat(), entry_price=100.0, fee_in=FEE)
    caps = {s: 720 for s in FD.SLEEVES} | {"FRENZY_WILLY": 90, "FRENZY_WILLY_HT": 90}
    ex = FC.exit_spec("FRENZY_LITE")
    wb = FC.walk(path, TE, 100.0, FEE, ex["tp"], ex["sl"], 720)
    F = pd.DataFrame([row("FRENZY_LITE", "REAL", -3.0, 300), row("FRENZY_LITE", "WALK", wb[0], 720), row("FRENZY_LITE", "WALK", 3.0, 30)])
    out = FC.price_rows(F, "MASTER", caps, False, {}, [], TE + 2000 * MIN)
    assert out[0]["LT60_2"] == 2.0 and out[0]["LT120_2"] == -3.0               # touch at minute ~71: armed at 60, not at 120
    assert out[1]["status"] == "PRICED" and out[1]["LT60_2"] == pytest.approx(2.0) and out[1]["LT120_2"] == pytest.approx(wb[0])
    assert out[2]["status"].startswith("PARITY MISS") and out[2]["LT60_2"] == 3.0
    W = pd.DataFrame([row("FRENZY_WILLY", "REAL", 1.0, 20)])
    assert not any(k.startswith("LT") for k in FC.price_rows(W, "MASTER", caps, False, {}, [], TE + 2000 * MIN)[0])
    hole = path.drop(index=150)                                                # minute ~151: inside the X = 120 span, no cap bar
    monkeypatch.setattr(WT, "load_1m", lambda pair: hole)
    o = FC.price_rows(F.iloc[:1], "MASTER", caps, False, {}, [], TE + 2000 * MIN)[0]
    assert o["status"] == "PRICED" and str(o.get("late_status", "")).startswith("UNPRICED")


def test_late_real_exit_after_x_above_the_lower_line_books_it():
    """(deep review) the exit bar is not read: a real TP (or cap close) after X above L means price crossed L on the way → L filled."""
    flat = bars([(100.1, 99.9, 100.0)] * 400)
    assert FC.late_real(flat, TE, TE + 100 * MIN, 100.0, FEE, 3.0, 60, 2.0, 3.0) == 2.0     # TP at minute 100, armed at 60
    assert FC.late_real(flat, TE, TE + 100 * MIN, 100.0, FEE, 3.0, 120, 2.0, 3.0) == 3.0    # exited before X → booked
    assert FC.late_real(flat, TE, TE + 100 * MIN, 100.0, FEE, 3.0, 60, 2.0, 1.7) == 1.7     # exited below L → booked
    assert FC.late_real(flat, TE, TE + 60 * MIN + 30_000, 100.0, FEE, 3.0, 60, 2.0, 3.0) == 2.0   # no full bar after X, exit above L


def test_late_section_renders_not_read_and_empty(tmp_path, monkeypatch):
    import run_best_case as RB
    monkeypatch.setattr(RB, "OUT", str(tmp_path))
    (tmp_path / "best_late.csv").write_text("\n")
    assert "nothing to read" in "\n".join(RB.late_section())
    pd.DataFrame([dict(sleeve="FRENZY_LITE", line="", after=float("nan"), late_tp=float("nan"), tp=3.0, live_cap=720,
                       b_read="not read (1m bars missing — re-run frenzy_caps.py --fetch)", m_read="–")]).to_csv(tmp_path / "best_late.csv", index=False)
    assert "not read (1m bars missing" in "\n".join(RB.late_section())
