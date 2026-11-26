"""🪢 DECISION_LOG 647 — DIST_TRAIL_SHADOW: Exit B + the DL 645 ATR trail (arm +2 % best low, 0.5 × entry ATR) on real DIST fills (observe)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_dist_tp_shadow as S        # noqa: E402

MIN, DAYMS = 60_000, 86_400_000
TE = int(pd.Timestamp("2026-12-01 12:00:30").value // 1_000_000)


def bars(prices, start=None):
    start = (TE // MIN + 1) * MIN if start is None else start
    return [[start + i * MIN, *p] for i, p in enumerate(prices)]


def test_constants_match_the_dl645_cell():
    assert (S.TR_DL, S.TR_ARM, S.TR_K, S.TR_KEY) == (647, 2.0, 0.5, "DIST_TRAIL_SHADOW")


def test_trail_arms_only_after_a_prior_bar_reached_plus2_and_exits_as_taker():
    # E 100, ATR 2 % → trail = best × 1.01. Bar 1 low 97.5 (+2.5 %) → armed from bar 2: best 97.5 → stop 98.475; bar 2 high 99 → TRAIL
    w, v, _ = S.walk_trail(bars([(100, 100.2, 97.5, 98), (98, 99, 97.8, 98.8)]), 100.0, TE, 10.0, 3.0, 2.0)
    assert w == "TRAIL" and v == pytest.approx(-(98.475 / 100 - 1) * 100 - S.FEE_IN - S.FEE_TK - 0 - S.EXIT_SLIP_MIN)
    # the same bar that makes the low never arms itself: low 97.5 and high 99 in ONE bar → no trail, only the own stop (103)
    w2, _, _ = S.walk_trail(bars([(100, 99, 97.5, 98.3)] + [(98.3, 98.4, 98.2, 98.3)] * 1500), 100.0, TE, 10.0, 3.0, 2.0)
    assert w2 == "TIME"


def test_unarmed_path_equals_exit_b():
    p = [(100, 101, 99, 100)] * 30 + [(100, 104, 99.5, 103)]            # never +2 %, then the own stop 103
    assert S.walk_trail(bars(p), 100.0, TE, 10.0, 3.0, 2.0)[:2] == S.walk_x(bars(p), 100.0, TE, 10.0, 3.0)[:2]
    tp = [(100, 100.1, 89.0, 89.5)]                                       # straight to the TP
    assert S.walk_trail(bars(tp), 100.0, TE, 10.0, 3.0, 2.0)[0] == "TP"


def test_no_atr_stamp_is_unpriced():
    assert S.walk_trail(bars([(100, 100, 99, 99)]), 100.0, TE, 10.0, 3.0, None)[0] == "UNPRICED"


def _frame(n, start="2026-12-01"):
    rows = []
    for i in range(n):
        o = pd.Timestamp(start) + pd.Timedelta(hours=12 * i)
        rows.append(dict(opened_at=str(o), pair=f"P{i}USDT", direction="SHORT", status="CLOSED", entry_strategy="DISTRIBUTION_SHORT",
                         pnl=0.0, pnl_percentage=0.0, close_reason="STOP_LOSS", notional_value=1000, entry_price=100.0, entry_dist_tp_pct=10.0,
                         entry_dist_sl_pct=3.0, entry_dist_model_slip_pct=0.0, entry_depth_mode=None, entry_depth_walk_pct=None,
                         entry_dist_atr_pct=2.0))
    a = pd.DataFrame(rows)
    a["o"] = pd.to_datetime(a.opened_at)
    return a


def _bars_fn(pair, te, now):                 # dips 3 % then rallies through the own stop → Exit B stops (−3), the trail exits ≈ −(−1.5)
    s = (te // MIN + 1) * MIN
    return [[s, 100, 100.2, 97.0, 97.5], [s + MIN, 97.5, 104, 97.4, 103.5]] + [[s + (2 + i) * MIN, 103, 103, 103, 103] for i in range(1500)]


def test_report_trail_line_freezes_once_on_its_own_key(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "N_MIN", 4)
    monkeypatch.setattr(S, "DAYS_MIN", 2)
    a = _frame(6)
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    trs = int(pd.Timestamp("2026-12-01 06:00").value // 1_000_000)          # the first fill (00:00) is before the trail floor
    st = tmp_path / "s.json"
    L = "\n".join(S.report(frame=a, since_ms=since, state_path=str(st), now_ms=TE + 10 * DAYMS, bars_fn=_bars_fn,
                           newest=pd.Timestamp("2026-12-20"), tr_since_ms=trs))
    assert "DIST_TRAIL_SHADOW" in L and "priced 5 on" in L
    cur = json.loads(st.read_text())
    assert "DIST_TP_SHADOW" in cur                          # (deep review) both keys survive the two writes
    rec = cur["DIST_TRAIL_SHADOW"]
    assert rec["state"] == "SHIP CANDIDATE" and rec["k"] == 4 and "TRAIL" in rec["risk"]
    L2 = "\n".join(S.report(frame=a, since_ms=since, state_path=str(st), now_ms=TE + 10 * DAYMS, bars_fn=_bars_fn,
                            newest=pd.Timestamp("2026-12-20"), tr_since_ms=trs))
    assert "Frozen read" in L2 and json.loads(st.read_text())["DIST_TRAIL_SHADOW"]["run_at"] == rec["run_at"]


def test_report_trail_not_evaluable_before_commit(tmp_path, monkeypatch):
    real = S._GK.dl_deploy_ms
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda dl, **k: None if dl == 647 else real(dl, **k))
    L = "\n".join(S.report(frame=_frame(1), since_ms=0, state_path=str(tmp_path / "s.json"), now_ms=TE, bars_fn=_bars_fn,
                           newest=pd.Timestamp("2026-12-20")))
    assert "DIST_TRAIL_SHADOW" in L and "NOT EVALUABLE" in L


def test_damaged_tp_record_does_not_hide_the_trail_line(tmp_path, monkeypatch):
    # (review) a damaged DIST_TP_SHADOW record used to return early → the trail read must still run and freeze on its own key
    monkeypatch.setattr(S, "N_MIN", 4)
    monkeypatch.setattr(S, "DAYS_MIN", 2)
    st = tmp_path / "s.json"
    st.write_text(json.dumps({"DIST_TP_SHADOW": {"junk": 1}}))
    L = "\n".join(S.report(frame=_frame(6), since_ms=0, state_path=str(st), now_ms=TE + 10 * DAYMS, bars_fn=_bars_fn,
                           newest=pd.Timestamp("2026-12-20"), tr_since_ms=0))
    assert "record damaged" in L and "DIST_TRAIL_SHADOW" in L
    cur = json.loads(st.read_text())
    assert cur["DIST_TP_SHADOW"] == {"junk": 1} and cur["DIST_TRAIL_SHADOW"]["state"] == "SHIP CANDIDATE"


def test_missing_atr_stamps_hold_the_trail_freeze(tmp_path, monkeypatch):
    # (review) > 10 % of the prefix without the ATR stamp → not evaluable, and the freeze waits (no silently filtered sample)
    monkeypatch.setattr(S, "N_MIN", 4)
    monkeypatch.setattr(S, "DAYS_MIN", 2)
    a = _frame(6)
    a.loc[a.index[0], "entry_dist_atr_pct"] = None
    st = tmp_path / "s.json"
    L = "\n".join(S.report(frame=a, since_ms=0, state_path=str(st), now_ms=TE + 10 * DAYMS, bars_fn=_bars_fn,
                           newest=pd.Timestamp("2026-12-20"), tr_since_ms=0))
    assert "1 not evaluable" in L and "NOT EVALUABLE inside the prefix" in L
    assert "DIST_TRAIL_SHADOW" not in (json.loads(st.read_text()) if st.exists() else {})
