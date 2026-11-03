"""Best-case view on TODAY's exits (DECISION_LOG 448): master FRENZY_LONG −3-era stops on the live −4 stop, current-batch LITE C +2
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
    out, n = MX.reprice_long_stop4(M)
    assert n == 2
    assert out.pct.tolist()[:3] == [1.5, pytest.approx(-4.018583), 1.5]   # TP books the line · STOP keeps its own slip, one point deeper
    assert out.pct.tolist()[3:] == [pytest.approx(-4.004819), -1.2, -4.0]  # −4 era, a non-stop loss, other sleeves: untouched
    assert M.pct.iloc[0] == -2.950589                                     # input not mutated


def test_long_stop4_missing_rewalk_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(MX, "TOUCHES", _touches(tmp_path, [("REAL", "FRENZY_LONG", "2026-10-18T00:30:09|AGTUSDT|LONG", "TP")]))
    with pytest.raises(SystemExit):
        MX.reprice_long_stop4(_M([("2026-10-15 01:00:00", "XUSDT", "FRENZY_LONG", -3.0, "STOP_LOSS", 3.0)]))


def test_long_stop4_real_master_pinned():
    """the real master: 9 −3-era stops move, FRENZY_LONG 29 · 86 % · +0.74 (a master rebuild that changes the cohort fails here)."""
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[(M.stack_keep == True) & (M.is_probe != True)].copy()   # noqa: E712
    M["o"] = pd.to_datetime(M.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    M["pct"] = pd.to_numeric(M.stack_pct, errors="coerce")
    M["sl"] = M.entry_strategy.astype(str)
    out, n = MX.reprice_long_stop4(M)
    g = out[out.sl == "FRENZY_LONG"].pct
    assert n == 9 and len(g) == 29
    assert round((g > 0).mean(), 3) == 0.862 and round(g.mean(), 2) == 0.74


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
