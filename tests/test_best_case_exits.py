"""Best-case view on TODAY's exits (DECISION_LOG 448): FRENZY_LONG on the live stop (−3 since DL 524; −4 re-walk only while −4 runs), current-batch LITE C +2
take-profits on +3, current-batch Flatbase opened before the +5 / −10 switch on its walk — fills that ran today's exit keep their real
result, and a fill the re-walk does not cover stops the run (never kept on the old exit)."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
import master_exits as MX  # noqa: E402


def _touches(tmp_path, rows):
    p = tmp_path / "touches.csv"
    pd.DataFrame(rows, columns=["block", "sleeve", "key", "why4"]).to_csv(p, index=False)
    return str(p)


def _M(rows):
    M = pd.DataFrame(rows, columns=["opened_at", "pair", "sl", "pct", "close_reason", "entry_frenzy_stop_pct"])
    M["o"] = pd.to_datetime(M.opened_at)
    return M


def test_long_stop4_tp_and_deeper_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(MX, "TOUCHES", _touches(tmp_path, [
        ("REAL", "FRENZY_LONG", "2026-10-18T00:30:09|AGTUSDT|LONG", "TP"),
        ("REAL", "FRENZY_LONG", "2026-10-03T03:36:19|ENJUSDT|LONG", "STOP"),
        ("SHADOW", "FRENZY_LONG", "2026-10-03T03:36:19|ENJUSDT|LONG", "TP"),     # a non-REAL row never decides
        ("REAL", "FRENZY_WILLY", "2026-10-17T19:40:10|ALTUSDT|LONG", "STOP")]))
    M = _M([("2026-10-18 00:30:09", "AGTUSDT", "FRENZY_LONG", -2.950589, "STOP_LOSS", None),
            ("2026-10-03 03:36:19", "ENJUSDT", "FRENZY_LONG", -3.018583, "STOP_LOSS", 3.0),
            ("2026-10-04 11:15:11", "AINUSDT", "FRENZY_LONG", 1.5, "FRENZY_TP", None),
            ("2026-10-29 00:55:09", "EVAAUSDT", "FRENZY_LONG", -4.004819, "STOP_LOSS", 4.0),
            ("2026-10-12 10:00:00", "TIMEUSDT", "FRENZY_LONG", -1.2, "TIME_EXIT", None),          # not a stop: never re-walked
            ("2026-10-17 19:40:10", "ALTUSDT", "FRENZY_WILLY", -4.0, "STOP_LOSS", None)])
    out, n = MX.reprice_long_stop4(M, live_stop=4.0)
    assert n == 2
    assert out.pct.tolist()[:3] == [1.5, pytest.approx(-4.018583), 1.5]   # TP books the line · STOP keeps its own slip, one point deeper
    assert out.pct.tolist()[3:] == [pytest.approx(-4.004819), -1.2, -4.0]  # −4 era, a non-stop loss, other sleeves: untouched
    assert M.pct.iloc[0] == -2.950589                                     # input not mutated


def test_long_stop4_missing_rewalk_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(MX, "TOUCHES", _touches(tmp_path, [("REAL", "FRENZY_LONG", "2026-10-18T00:30:09|AGTUSDT|LONG", "TP")]))
    with pytest.raises(SystemExit):
        MX.reprice_long_stop4(_M([("2026-10-15 01:00:00", "XUSDT", "FRENZY_LONG", -3.0, "STOP_LOSS", 3.0)]), live_stop=4.0)


def test_long_stop4_real_master_pinned():
    """the real master: 9 −3-era stops move, FRENZY_LONG 29 · 86 % · +0.74 (a master rebuild that changes the cohort fails here)."""
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[(M.stack_keep == True) & (M.is_probe != True)].copy()   # noqa: E712
    M["o"] = pd.to_datetime(M.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    M["pct"] = pd.to_numeric(M.stack_pct, errors="coerce")
    M["sl"] = M.entry_strategy.astype(str)
    out, n = MX.reprice_long_stop4(M, live_stop=4.0)
    g = out[out.sl == "FRENZY_LONG"].pct
    assert n == 9 and len(g) == 29
    assert round((g > 0).mean(), 3) == 0.862 and round(g.mean(), 2) == 0.74
    out3, n3 = MX.reprice_long_stop4(M, live_stop=3.0)                     # (DL 524) live −3: the master keeps its traded −3 stops
    g3 = out3[out3.sl == "FRENZY_LONG"].pct
    assert n3 == 0 and round((g3 > 0).mean(), 3) == 0.69 and round(g3.mean(), 2) == 0.13


def test_live_long_stop_reads_config(tmp_path):
    import json
    p = tmp_path / "c.json"
    for th, want in (({"frenzy_long_stop_pct": 0.0, "frenzy_stop_pct": 3.0}, 3.0), ({"frenzy_long_stop_pct": 4.0, "frenzy_stop_pct": 3.0}, 4.0)):
        p.write_text(json.dumps({"thresholds": th}))
        assert MX.live_long_stop(str(p)) == want


def test_b25_long_minus4_repriced_to_minus3(tmp_path, monkeypatch):
    """(DL 524) a −4-era FRENZY_LONG fill on the live −3 stop: its own touch3 stamp decides — no touch = as traded · touch after the +1.5 TP
    a −4 stop close = booked + 1 (own slip kept, even after a +1.5 peak — no TP credit) · touched then recovered = −3 + the median −4 stop slip ·
    a −4 stop without a stamp stops."""
    rows = [("2026-11-01T02:10:08", "2026-11-01T03:00:00", "FRENZY_TP", "POWRUSDT", "FRENZY_LONG", 1.51, 1.0, 100.0, 4.0, None, None),
            ("2026-11-03T13:20:07", "2026-11-03T14:00:00", "STOP_LOSS", "BMTUSDT", "FRENZY_LONG", -4.02, -4.0, 100.0, 4.0, "2026-11-03T13:30:59", 0.21),
            ("2026-11-06T23:30:07", "2026-11-07T01:00:00", "FRENZY_TP", "ONEUSDT", "FRENZY_LONG", 1.51, 1.5, 100.0, 4.0, "2026-11-06T23:32:03", 0.27),
            ("2026-11-07T10:00:00", "2026-11-07T11:00:00", "STOP_LOSS", "XUSDT", "FRENZY_LONG", -4.01, -4.0, 100.0, 4.0, "2026-11-07T10:30:00", 1.6)]
    cols = ["opened_at", "closed_at", "close_reason", "pair", "sl", "pct", "u", "notional_value", "entry_frenzy_stop_pct", "touch3_at", "touch3_peak_before"]
    A = pd.DataFrame(rows, columns=cols); A["o"] = pd.to_datetime(A.opened_at)
    cf = tmp_path / "cf.csv"; pd.DataFrame(columns=["opened_at", "pair", "entry_strategy", "booked", "cf_pct"]).to_csv(cf, index=False)
    fb = tmp_path / "fb.csv"; pd.DataFrame(columns=["k", "pair", "now510", "h_now510"]).to_csv(fb, index=False)
    monkeypatch.setattr(MX, "LITE_CF", str(cf))
    out, moved = MX.reprice_b25_exits(A, str(fb), live_stop=3.0)
    assert out.pct.tolist() == [1.51, pytest.approx(-3.02), pytest.approx(-3.015), pytest.approx(-3.01)] and moved == {"FRENZY_LONG": 3}   # slip = median(−0.02, −0.01)
    assert out.u.tolist()[1] == pytest.approx(-3.02)                                       # $ follows the new % on its notional
    same, mv = MX.reprice_b25_exits(A, str(fb), live_stop=4.0)                              # live −4: every −4 fill as traded
    assert same.pct.tolist() == A.pct.tolist() and mv == {}
    bad = A.iloc[[1]].assign(touch3_at=None)
    with pytest.raises(SystemExit):
        MX.reprice_b25_exits(bad, str(fb), live_stop=3.0)


def _b25(tmp_path, monkeypatch, A, cf_rows, fb_rows):
    cf = tmp_path / "cf.csv"
    pd.DataFrame(cf_rows, columns=["opened_at", "pair", "entry_strategy", "booked", "cf_pct"]).to_csv(cf, index=False)
    fb = tmp_path / "fb.csv"
    pd.DataFrame(fb_rows, columns=["k", "pair", "now510", "h_now510"]).to_csv(fb, index=False)
    monkeypatch.setattr(MX, "LITE_CF", str(cf))
    return MX.reprice_b25_exits(A, str(fb))


def _A(rows):
    A = pd.DataFrame(rows, columns=["opened_at", "closed_at", "close_reason", "pair", "sl", "pct", "u", "notional_value"])
    A["o"] = pd.to_datetime(A.opened_at, format="mixed")
    return A


def test_b25_lite_and_flatbase_repriced(tmp_path, monkeypatch):
    A = _A([("2026-10-27T13:10:07", "2026-10-27T14:00:00", "FRENZY_TP", "REZUSDT", "FRENZY_LITE_RETRY", 2.0056, 20.0, 1000.0),  # → −3
            ("2026-10-26 04:55:09", "2026-10-26T05:00:00", "FRENZY_TP", "ERAUSDT", "FRENZY_LITE", 2.0077, 20.0, 1000.0),    # → +3
            ("2026-10-27T11:20:10", "2026-10-27T11:30:00", "STOP_LOSS", "REZUSDT", "FRENZY_LITE", -3.0145, -30.0, 1000.0),  # stop: kept
            ("2026-11-01T17:30:08", "2026-11-01T18:00:00", "FRENZY_TP", "SOPHUSDT", "FRENZY_LITE", 3.0362, 30.0, 1000.0),   # +3 TP: kept
            ("2026-10-24T08:00:13", "2026-10-24T20:00:00", "STOP_LOSS", "KATUSDT", "FLATBASE_SHORT", -15.14, -75.7, 500.0), # → walk
            ("2026-11-01T12:00:09", "2026-11-01T13:00:00", "TP", "GUSDT", "FLATBASE_SHORT", 4.957, 24.785, 500.0),          # ran +5/−10: kept
            ("2026-10-24T09:00:00", "2026-10-24T10:00:00", "TP", "ABCUSDT", "MOMENTUM_LONG", 2.0, 2.0, 100.0)])
    out, moved = _b25(tmp_path, monkeypatch, A,
                      [("2026-10-27T13:10:07", "REZUSDT", "FRENZY_LITE_RETRY", 2.0056, -3.0),
                       ("2026-10-26T04:55:09", "ERAUSDT", "FRENZY_LITE", 2.0077, 3.0)],
                      [("2026-10-24T08:00:13", "KATUSDT", 4.9379, 90.0), ("2026-11-01T12:00:09", "GUSDT", 4.9379, 30.0)])
    assert out.pct.tolist() == [-3.0, 3.0, -3.0145, 3.0362, 4.9379, 4.957, 2.0]
    assert out.u.tolist() == [-30.0, 30.0, -30.0, 30.0, pytest.approx(24.6895), 24.785, 2.0]   # $ follows the new % on its notional
    assert pd.Timestamp(out.closed_at.iloc[4]) == pd.Timestamp("2026-10-24 09:30:13")          # Flatbase closes at the walk's exit
    assert out.closed_at.iloc[0] == "2026-10-27T14:00:00"                                     # LITE keeps its close (documented caveat)
    assert moved == {"FRENZY_LITE_RETRY": 1, "FRENZY_LITE": 1, "FLATBASE_SHORT": 1}


@pytest.mark.parametrize("row", [("2026-10-28T10:05:08", "2026-10-28T11:00:00", "FRENZY_TP", "INITUSDT", "FRENZY_LITE_RETRY", 2.0018, 20.0, 1000.0),
                                 ("2026-10-24T16:00:14", "2026-10-24T20:00:00", "STOP_LOSS", "ORDERUSDT", "FLATBASE_SHORT", -15.11, -75.0, 500.0)])
def test_b25_uncovered_fill_stops(tmp_path, monkeypatch, row):
    with pytest.raises(SystemExit):
        _b25(tmp_path, monkeypatch, _A([row]), [], [("2026-10-24T16:00:14", "ORDERUSDT", float("nan"), float("nan"))])


def test_cut_times():
    assert MX.FB_EXIT_UTC == pd.Timestamp("2026-10-28 21:37")     # DL 408 commit 18:27 −03:00 + 10 min
    assert MX.LONG_STOP4_UTC == pd.Timestamp("2026-10-19 17:00")  # DL 317 deploy (scout_long_stop4 cohort start)
    assert MX.key("2026-10-24 08:00:13.123456", "X") == MX.key("2026-10-24T08:00:13", "X")
