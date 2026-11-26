"""🪜 DECISION_LOG 647 — EXH_RUNNER_F6: the EXH runner with a +6-net profit floor (observe only) — floor price, the floored walk inside
price_fill, its own forward floor (the DL 647 commit) and frozen read; +5 shown as info, never deciding."""
import json
import os
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_tp_runner_636 as S        # noqa: E402

MIN, DAYMS = 60_000, 86_400_000
T0 = int(pd.Timestamp("2026-12-01 12:00").value // 1_000_000)


def bars(prices, start=T0 + MIN):
    return [[start + i * MIN, *p] for i, p in enumerate(prices)]


def hist(px=93.0, rng=0.93, n=300):
    return bars([(px, px + rng / 2, px - rng / 2, px)] * n, start=T0 - n * MIN)


def _fill(opened, pair="XUSDT", touch=T0):
    return SimpleNamespace(pair=pair, entry_price=100.0, exit_price=93.0, closed_at=str(pd.Timestamp(touch, unit="ms")),
                           entry_exh_sl_pct=6.0, o=pd.Timestamp(opened))


def test_exh_floor_price_is_plus6_net_after_taker_both_sides():
    assert S.exh_floor_px(100.0, 6.0, fee=0.05) == pytest.approx(100 * 93.95 / 100.05)
    assert S.EXH_FLOOR_NET == 6.0 and S.EXH_FLOOR_INFO == 5.0 and S.F6_DL == 647


def test_floor_binds_on_a_bounce_the_plain_runner_rides():
    # short TP 93, ATR 1 % → trail 4 % = 96.72; a bounce to 94.5 hits the +6 floor (≈ 93.90) but not the trail → plain runner rides it,
    # then a run to 80 and a bounce to 84 → plain trail 83.2 exits there
    p = [(93, 94.5, 92.8, 94)] + [(94, 94, 80, 80)] + [(80, 84, 80, 84)]
    r = S.price_fill(_fill("2026-12-01 10:00"), -1, T0 + 3 * DAYMS, None, bars_fn=lambda *a: hist() + bars(p), funding_fn=lambda *a: 0.0)
    assert r["why_f6"] == "floor" and r["d_f6"] < 0                       # gave back ≈ 1 pt to the floor
    assert r["why"] == "trail" and r["d"] > 5                              # the plain runner caught the run
    assert r["why_f5"] in ("floor", "trail") and r["d_f5"] is not None
    assert r["d_f6"] == pytest.approx(S.delta_pp(-1, 100.0, 93.0, S.exh_floor_px(100.0, 6.0), 0.0))


def test_floor_binds_before_the_trail_on_a_spike():
    r = S.price_fill(_fill("2026-12-01 10:00"), -1, T0 + 3 * DAYMS, None,
                     bars_fn=lambda *a: hist() + bars([(93, 108, 92, 107)]), funding_fn=lambda *a: 0.0)
    assert r["why_f6"] == "floor" and r["why"] == "trail"                  # floor ≈ 93.9 < trail 96.72 < own stop 106: the tightest binds
    assert r["d_f6"] > r["d"]                                              # the floor caps the spike loss the trail takes


def test_floor_rows_drop_unpriced_walks():
    rows = [dict(d=1.0, day=1, o=1, d_f6=-1.0, why_f6="floor", fund_f6=0.0), dict(d=2.0, day=2, o=2, d_f6=None, why_f6=None, fund_f6=None)]
    assert S.floor_rows(rows) == [dict(d=-1.0, why="floor", fund=0.0, day=1, o=1)]


def _exh_frame(opens):
    a = pd.DataFrame([dict(opened_at=o, pair=f"P{i}USDT", pnl_percentage=6.9, close_reason="FRENZY_TP", entry_price=100, exit_price=93.0,
                           closed_at=str(pd.Timestamp(T0 + i * DAYMS, unit="ms")), entry_exh_tp_pct=7.0, entry_exh_atr_pct=1.0,
                           entry_exh_sl_pct=6.0) for i, o in enumerate(opens)]).reindex(columns=list(S.COLS))
    a["entry_strategy"], a["status"], a["direction"] = S.EXH, "CLOSED", "SHORT"
    a["o"] = pd.to_datetime(a.opened_at)
    return a


def _empty_ht():
    a = pd.DataFrame(columns=list(S.COLS) + ["o"])
    return a


def test_report_f6_counts_only_from_its_own_floor_and_freezes_once(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "EXH_N", 2)
    monkeypatch.setattr(S, "EXH_DAYS", 2)
    since636 = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    since647 = int(pd.Timestamp("2026-12-01 05:00").value // 1_000_000)
    exh = _exh_frame(["2026-12-01 01:00", "2026-12-01 06:00", "2026-12-01 07:00"])     # the first is before the 647 floor

    def bars_fn(pair, t, now):                    # every hit runs far, no bounce → floored runner trails out above TP (Δ > 0)
        h = bars([(93, 93.5, 92.5, 93)] * 300, start=t // MIN * MIN - 300 * MIN)
        return h + bars([(93, 93.2, 70, 70), (70, 75, 70, 74)], start=t // MIN * MIN + MIN)
    st = tmp_path / "s.json"
    L = "\n".join(S.report(frames={S.HT: _empty_ht(), S.EXH: exh}, since_ms=since636, era_ms={S.HT: since636, S.EXH: since636},
                           state_path=str(st), now_ms=T0 + 5 * DAYMS, bars_fn=bars_fn, funding_fn=lambda *a: 0.0,
                           newest=pd.Timestamp("2026-12-09"), f6_since_ms=since647))
    assert "EXH_RUNNER_F6" in L and "IN-SAMPLE (before the DL 647 commit; never deciding): N 1 " in L
    assert "FORWARD (opened ≥ 2026-12-01 05:00 UTC): N 2 " in L and "info — floor +5" in L
    rec = json.loads(st.read_text())["EXH_RUNNER_F6"]
    assert rec["verdict"] == "ARM_CANDIDATE" and "first 2 TP hits" in rec["detail"]
    L2 = "\n".join(S.report(frames={S.HT: _empty_ht(), S.EXH: exh}, since_ms=since636, era_ms={S.HT: since636, S.EXH: since636},
                            state_path=str(st), now_ms=T0 + 6 * DAYMS, bars_fn=bars_fn, funding_fn=lambda *a: 0.0,
                            newest=pd.Timestamp("2026-12-09"), f6_since_ms=since647))
    assert "EXH_RUNNER_F6: **FROZEN" in L2 and json.loads(st.read_text())["EXH_RUNNER_F6"]["run_at"] == rec["run_at"]   # freeze once


def test_f6_not_evaluable_before_the_647_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    L = "\n".join(S.report(frames={S.HT: _empty_ht(), S.EXH: _exh_frame([])}, since_ms=since, era_ms={S.HT: since, S.EXH: since},
                           state_path=str(tmp_path / "s.json"), now_ms=T0, bars_fn=lambda *a: [], funding_fn=lambda *a: 0.0,
                           newest=pd.Timestamp("2026-12-09")))
    assert "EXH_RUNNER_F6: ⚠ NOT EVALUABLE" in L


def test_tp_below_the_floor_never_runs_delta_zero():
    # (review) a low-ATR TP of +5.5 (exit 94.5) sits below the +6 floor → the floored rule sells at TP: Δ 0, not a fake instant 'floor' exit
    f = _fill("2026-12-01 10:00")
    f.exit_price = 94.5
    r = S.price_fill(f, -1, T0 + 3 * DAYMS, None, bars_fn=lambda *a: hist(94.5) + bars([(94.5, 94.7, 80, 80), (80, 84, 80, 84)]),
                     funding_fn=lambda *a: 0.0)
    assert r["why_f6"] == "tp<floor" and r["d_f6"] == 0.0 and r["fund_f6"] == 0.0
    assert r["why_f5"] != "tp<floor"                                       # +5 floor ≈ 94.9 sits above TP 94.5 → it runs


def test_f6_not_evaluable_count_ignores_fills_before_its_floor(tmp_path, monkeypatch):
    # (review) NOT_EVALUABLE fills opened before the DL 647 floor must not trip F6's > 10 % hold
    monkeypatch.setattr(S, "EXH_N", 2)
    monkeypatch.setattr(S, "EXH_DAYS", 2)
    since636 = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    since647 = int(pd.Timestamp("2026-12-01 05:00").value // 1_000_000)
    exh = _exh_frame(["2026-12-01 01:00", "2026-12-01 02:00", "2026-12-01 06:00", "2026-12-01 07:00"])
    exh.loc[[0, 1], "entry_exh_sl_pct"] = None                            # → NOT_EVALUABLE: no entry_exh_sl_pct stamp (pre-647)

    def bars_fn(pair, t, now):
        h = bars([(93, 93.5, 92.5, 93)] * 300, start=t // MIN * MIN - 300 * MIN)
        return h + bars([(93, 93.2, 70, 70), (70, 75, 70, 74)], start=t // MIN * MIN + MIN)
    st = tmp_path / "s.json"
    S.report(frames={S.HT: _empty_ht(), S.EXH: exh}, since_ms=since636, era_ms={S.HT: since636, S.EXH: since636}, state_path=str(st),
             now_ms=T0 + 5 * DAYMS, bars_fn=bars_fn, funding_fn=lambda *a: 0.0, newest=pd.Timestamp("2026-12-09"), f6_since_ms=since647)
    assert json.loads(st.read_text())["EXH_RUNNER_F6"]["verdict"] == "ARM_CANDIDATE"
