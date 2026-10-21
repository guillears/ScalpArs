"""🔓 scripts/scout_willy_hold_off.py — the DL 340 revert-gate tracker (gate (a) day loss vs account · gate (b) freed cohort, first 8 frozen).
Synthetic exports only: no network, no git (since_ms passed), state in tmp_path."""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_hold_off as S  # noqa: E402

SINCE = pd.Timestamp("2026-10-22T00:00:00")
SINCE_MS = int(SINCE.value // 1_000_000)
NOW_MS = int(pd.Timestamp("2026-10-25T00:00:00").value // 1_000_000)


def _row(o, c, pair, es, pct, usd, paper=False, status="CLOSED", why="FRENZY_TP", d="LONG"):
    return dict(opened_at=o, closed_at=c, pair=pair, direction=d, status=status, entry_strategy=es, pnl=usd, pnl_percentage=pct,
                close_reason=why, is_paper=paper)


def _csv(tmp_path, rows, name="scalpars_orders_live_x.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return [str(p)]


def test_freed_cohort_needs_same_book_overlap_and_post_deploy(tmp_path):
    rows = [_row("2026-10-22T10:00:00", "2026-10-22T11:00:00", "WUSDT", "FRENZY_WILLY", 1.25, 30),
            _row("2026-10-22T10:30:00", "2026-10-22T12:00:00", "AUSDT", "FRENZY_LITE", -3.0, -60),        # inside → freed
            _row("2026-10-22T11:00:00", "2026-10-22T12:00:00", "BUSDT", "MOMENTUM", 1.0, 10),             # at the WILLY close → not
            _row("2026-10-22T10:40:00", "2026-10-22T12:00:00", "CUSDT", "MOMENTUM", 1.0, 10, paper=True),  # other book → not
            _row("2026-10-22T10:45:00", "2026-10-22T12:00:00", "DUSDT", "MANUAL", 1.0, 10),               # manual → not
            _row("2026-10-21T10:30:00", "2026-10-21T12:00:00", "EUSDT", "FRENZY_LONG", 1.0, 10),          # before the deploy → not
            _row("2026-10-21T10:00:00", "2026-10-21T11:00:00", "VUSDT", "FRENZY_WILLY", 1.25, 30)]
    a = S.load(_csv(tmp_path, rows))
    b = S.freed(a, SINCE, pd.Timestamp(NOW_MS, unit="ms"))
    assert list(b.pair) == ["AUSDT"] and list(b.willy_pair) == ["WUSDT"]


def test_dedup_closed_wins(tmp_path):
    rows = [_row("2026-10-22T10:00:00", None, "WUSDT", "FRENZY_WILLY", None, None, status="OPEN"),
            _row("2026-10-22T10:00:00", "2026-10-22T11:00:00", "WUSDT", "FRENZY_WILLY", 1.25, 30)]
    a = S.load(_csv(tmp_path, rows))
    assert len(a) == 1 and a.status.iloc[0] == "CLOSED"


def test_gate_a_counts_only_days_with_a_freed_fill(tmp_path):
    rows = [_row("2026-10-22T10:00:00", "2026-10-22T11:00:00", "WUSDT", "FRENZY_WILLY", -4.0, -100),
            _row("2026-10-22T09:00:00", "2026-10-22T12:00:00", "AUSDT", "MOMENTUM", -1.5, -30),             # opened BEFORE the WILLY: never makes a day count
            _row("2026-10-23T10:00:00", "2026-10-23T11:00:00", "VUSDT", "FRENZY_WILLY", -4.0, -100),
            _row("2026-10-23T10:30:00", "2026-10-23T12:00:00", "BUSDT", "FRENZY_LITE", -1.0, -20),          # freed fill → 10-23 counts
            _row("2026-10-23T20:00:00", "2026-10-23T21:00:00", "CUSDT", "MOMENTUM", 1.0, 5)]                # same day: in the day's P&L
    a = S.load(_csv(tmp_path, rows))
    now = pd.Timestamp(NOW_MS, unit="ms")
    out = S.gate_a(a, SINCE, now, account=1000.0)
    assert [(r["day"], r["n_freed"], r["n"], round(r["usd"], 2), r["alert"]) for r in out] == [("2026-10-23", 1, 3, -115.0, True)]
    out = S.gate_a(a, SINCE, now, account=2000.0)
    assert out[0]["alert"] is False and round(out[0]["pct"], 2) == -5.75
    L = S.run(NOW_MS, _csv(tmp_path, rows), since_ms=SINCE_MS, account=2000.0, state_path=str(tmp_path / "s.json"))
    ga = [x for x in L if x.startswith("- 2026-")]
    assert ga and all("ASSUMED account $2,000 — pass --account" in x for x in ga)


def _freed_rows(pcts):
    rows = [_row("2026-10-22T00:00:00", "2026-10-23T23:00:00", "WUSDT", "FRENZY_WILLY", 1.25, 30)]
    for i, p in enumerate(pcts):
        rows.append(_row(f"2026-10-22T{1 + i:02d}:00:00", f"2026-10-22T{1 + i:02d}:30:00", f"P{i}USDT", "FRENZY_LITE", p, p * 10))
    return rows


def test_gate_b_collecting_then_alert_and_frozen(tmp_path):
    st = str(tmp_path / "st.json")
    L = S.run(NOW_MS, _csv(tmp_path, _freed_rows([-1.0] * 7)), since_ms=SINCE_MS, state_path=st)
    assert any("collecting (7/8" in x for x in L) and not os.path.exists(st)
    L = S.run(NOW_MS, _csv(tmp_path, _freed_rows([-1.0] * 8 + [5.0, 5.0])), since_ms=SINCE_MS, state_path=st)
    assert any("🔴 ALERT — first 8: avg -1.000 %" in x for x in L)
    assert len(json.load(open(st))["frozen_keys"]) == 8
    L = S.run(NOW_MS, _csv(tmp_path, _freed_rows([-1.0] * 8 + [9.0] * 5)), since_ms=SINCE_MS, state_path=st)   # later fills never re-fit
    assert any("🔴 ALERT — first 8" in x for x in L)


def test_gate_b_bar_needs_both_conditions(tmp_path):
    cases = (([-0.4] * 8, False),             # avg < 0 but Σ −3.2 > −4 → no alert
             ([3.0] + [-1.0] * 7, True),       # avg −0.5 < 0 and Σ exactly −4.0 ≤ −4 → alert
             ([-0.5] * 8, True),               # avg −0.5 · Σ −4.0 → alert
             ([12.0] + [-1.0] * 7, False))     # Σ +5 · avg > 0 → no alert
    for k, (pcts, alert) in enumerate(cases):
        L = S.run(NOW_MS, _csv(tmp_path, _freed_rows(pcts)), since_ms=SINCE_MS, state_path=str(tmp_path / f"st{k}.json"))
        assert ("🔴 ALERT" in "\n".join(L)) is alert, pcts


def test_gate_b_operator_closed_excluded(tmp_path):
    rows = _freed_rows([-1.0] * 8)
    rows[1]["close_reason"] = "MANUAL"
    L = S.run(NOW_MS, _csv(tmp_path, rows), since_ms=SINCE_MS, state_path=str(tmp_path / "s.json"))
    txt = "\n".join(L)
    assert "collecting (7/8" in txt and "operator-closed: 1" in txt


def test_not_evaluable_without_deploy(monkeypatch, tmp_path):
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    L = S.run(NOW_MS, _csv(tmp_path, _freed_rows([-1.0] * 8)), state_path=str(tmp_path / "s.json"))
    assert any("NOT EVALUABLE" in x for x in L)


def test_wired_into_opportunity_scout_and_docs():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_hold_off as _who" in src and "_rg_sec += _who.run(now_ms)" in src
    cs = open(os.path.join(ROOT, "CLAUDE_CURRENT_STATE.md"), encoding="utf-8").read()
    assert "scripts/scout_willy_hold_off.py" in cs
    plan = open(os.path.join(ROOT, "reports", "NEXT_BATCH_REVIEW_PLAN.md"), encoding="utf-8").read()
    assert "scripts/scout_willy_hold_off.py" in plan
