"""📏 FRENZY_LONG RE-SIZE read in the scout (DECISION_LOG 351): the cohort (live −4 stamp only, CLOSED, LONG, operator-closed out, dedup
across exports), the locked DL 317 gate called verbatim (same verdict as services.frenzy.frenzy_long_stop4_gate on the same rows), the
info-only confidence line, and the opportunity-scout wiring. Synthetic exports — no network."""
import datetime as dt
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_long_resize as S  # noqa: E402
from services.frenzy import frenzy_long_stop4_gate  # noqa: E402

WIN, LOSS = 1.44, -4.12


def _rows(n_days, per_day, losers, t0=dt.datetime(2026, 10, 24, 1, 0)):
    """per_day fills on each of n_days UTC days, every pair distinct; the first fill of the first `losers` days loses."""
    out = []
    for d in range(n_days):
        for k in range(per_day):
            o = t0 + dt.timedelta(days=d, hours=2 * k)
            x = LOSS if (k == 0 and d < losers) else WIN
            out.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), pair=f"P{d}X{k}USDT", direction="LONG", status="CLOSED",
                            entry_strategy="FRENZY_LONG", pnl=x, pnl_percentage=x, close_reason="FRENZY_TP" if x > 0 else "STOP_LOSS",
                            is_paper=False, entry_frenzy_stop_pct=4.0, touch3_at=None, touch3_recovered_tp=None))
    return out


def _csv(tmp_path, rows, name="a.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def test_cohort_keeps_only_closed_live4_long_and_drops_operator_closes(tmp_path):
    base = _rows(1, 1, 0)[0]
    rows = [base,
            {**base, "opened_at": "2026-10-24T03:00:00", "entry_frenzy_stop_pct": 3.0},            # pre-317 −3 fill
            {**base, "opened_at": "2026-10-24T04:00:00", "entry_frenzy_stop_pct": None},           # unstamped
            {**base, "opened_at": "2026-10-24T05:00:00", "entry_strategy": "FRENZY_LITE"},         # other sleeve
            {**base, "opened_at": "2026-10-24T06:00:00", "status": "OPEN"},                        # still open
            {**base, "opened_at": "2026-10-24T07:00:00", "close_reason": "MANUAL"}]                # operator-closed
    old = {**base, "status": "OPEN", "pnl_percentage": None}                                        # same key, older OPEN export row
    c, ox = S.cohort(S.load([_csv(tmp_path, [old], "old.csv"), _csv(tmp_path, rows, "new.csv")]))
    assert list(c.opened_at) == [base["opened_at"]]
    assert len(ox) == 1


def test_closed_wins_whatever_the_export_order_and_timestamp_format(tmp_path):
    base = _rows(1, 1, 0)[0]
    newer_open = {**base, "opened_at": base["opened_at"].replace("T", " ") + ".123456", "status": "OPEN", "pnl_percentage": None}
    a = S.load([_csv(tmp_path, [base], "old.csv"), _csv(tmp_path, [newer_open], "new.csv")])
    assert len(a) == 1 and a.status.iloc[0] == "CLOSED"


def test_unstamped_fills_are_shown_not_counted(tmp_path):
    base = _rows(1, 1, 0)[0]
    out = "\n".join(S.run(paths=[_csv(tmp_path, [{**base, "entry_frenzy_stop_pct": None}])]))
    assert "not counted: 1 " in out and "no live −4 fills yet" in out


def test_locked_gate_is_the_dashboard_function_and_resizes(tmp_path):
    c, _ = S.cohort(S.load([_csv(tmp_path, _rows(12, 2, 4))]))
    v, txt = frenzy_long_stop4_gate(S.gate_rows(c))
    assert v == "RESIZE" and "PROPOSE one size step" in txt
    out = "\n".join(S.run(paths=[_csv(tmp_path, _rows(12, 2, 4), "b.csv")]))
    assert f": {v} — {txt}" in out and "Locked gate (DL 317, same function as the dashboard" in out
    assert "Confidence (DL 351, info only" in out and "not yet" in out and "N 24/30" in out   # locked read passes before the info bars


def test_locked_gate_collects_below_twenty_fills_on_eight_days(tmp_path):
    c, _ = S.cohort(S.load([_csv(tmp_path, _rows(6, 3, 0))]))       # 18 fills, 6 days
    assert frenzy_long_stop4_gate(S.gate_rows(c))[0] == "COLLECTING"


def test_confidence_strong_and_breakeven_from_the_cohort(tmp_path):
    c, _ = S.cohort(S.load([_csv(tmp_path, _rows(12, 3, 4))]))       # 36 fills · 12 days · 32 wins
    ok, txt = S.confidence(c)
    assert ok, txt
    assert "breakeven 74 %" in txt                                    # 4.12 / (1.44 + 4.12)


def test_confidence_fails_when_the_cohort_loses(tmp_path):
    c, _ = S.cohort(S.load([_csv(tmp_path, _rows(12, 3, 12))]))      # one loser per day: avg (2·1.44 − 4.12) / 3 < 0
    ok, txt = S.confidence(c)
    assert not ok and "✗ avg" in txt and "✗ Σ $" in txt


def test_touch_rows_reach_the_revert_gate(tmp_path):
    rows = _rows(10, 1, 10)                                           # 10 −3 touches, none recovered → REVERT
    for r in rows:
        r["touch3_at"] = r["opened_at"]
        r["touch3_recovered_tp"] = False
    c, _ = S.cohort(S.load([_csv(tmp_path, rows)]))
    assert frenzy_long_stop4_gate(S.gate_rows(c))[0] == "REVERT"


def test_recovered_tri_state_parses_from_csv(tmp_path):
    rows = _rows(10, 1, 10)                                           # 10 touches: 3 True · 6 False · 1 unknown ('') → 3/9 known, no REVERT
    for i, r in enumerate(rows):
        r["touch3_at"] = r["opened_at"]
        r["touch3_recovered_tp"] = True if i < 3 else ("" if i == 9 else False)
    c, _ = S.cohort(S.load([_csv(tmp_path, rows)]))
    g = S.gate_rows(c)
    assert [z[5] for z in g].count(True) == 3 and [z[5] for z in g].count(None) == 1
    v, txt = frenzy_long_stop4_gate(g)
    assert v != "REVERT" and "1 unknown outcome" in txt


def test_empty_exports_never_raise():
    out = "\n".join(S.run(paths=[]))
    assert "COLLECTING" in out and "no live −4 fills yet" in out


def test_wired_into_the_opportunity_scout():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_long_resize as _lrs" in src and "_rg_sec += _lrs.run(now_ms)" in src


def test_old_export_without_the_stamp_columns_never_raises(tmp_path):
    old = {k: v for k, v in _rows(1, 1, 0)[0].items() if k not in ("entry_frenzy_stop_pct", "touch3_at", "touch3_recovered_tp")}
    out = "\n".join(S.run(paths=[_csv(tmp_path, [old], "old.csv"), _csv(tmp_path, _rows(1, 1, 0, dt.datetime(2026, 10, 25)), "new.csv")]))
    assert "not counted: 1 " in out and "(live 1 · paper 0)" in out
