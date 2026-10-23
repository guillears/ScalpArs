"""🕰🧭 LITE_LATE_BTCFALL observe line (DECISION_LOG 353): the frozen cell (≥ 12 h ∧ BTC 1 h slope ≤ 0; missing stamp = unscored, never
rest), master (today's stack, stack_keep) + forward exports (keys not in the master) pooled, operator-closed out, and the frozen gate
(COLLECTING → RETIRE / ARM_CANDIDATE / KEEP_OBSERVING at N ≥ 15 on ≥ 8 days, decided once on the crossing prefix). Synthetic CSVs — no network."""
import datetime as dt
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_lite_late_btcfall as S  # noqa: E402

T0 = dt.datetime(2026, 10, 24, 1, 0)


def _row(i, h, s, y, day=None, strat="FRENZY_LITE", reason=None, pair=None, **kw):
    o = T0 + dt.timedelta(days=i if day is None else day, minutes=i)
    r = dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), pair=pair or f"P{i}USDT", direction="LONG", status="CLOSED", entry_strategy=strat,
             pnl=y * 10, pnl_percentage=y, close_reason=reason or ("FRENZY_TP" if y > 0 else "STOP_LOSS"), is_paper=False,
             entry_frenzy_hours=h, entry_btc_1h_slope=s)
    r.update(kw)
    return r


def _csv(tmp_path, rows, name):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


def _master(tmp_path, rows):
    for r in rows:
        r.setdefault("stack_keep", True); r.setdefault("stack_pct", r["pnl_percentage"]); r.setdefault("stack_pnl", r["pnl"])
    return _csv(tmp_path, rows, "master.csv")


def test_cell_cut_and_unscored():
    df = pd.DataFrame(dict(h=[12.0, 11.99, 15, 15, None, 13], s=[0.0, -0.1, 0.01, -0.2, -0.1, None]))
    assert list(S.classify(df)) == ["cell", "rest", "rest", "cell", "unscored", "unscored"]


def test_master_uses_stack_and_forward_skips_master_keys(tmp_path):
    m = [_row(0, 13, -0.1, -3.0, stack_pct=-2.5, stack_pnl=-25.0), _row(1, 2, 0.1, 2.0, stack_keep=False),
         _row(2, 13, -0.1, 0.1, reason="MANUAL")]
    mp = _master(tmp_path, m)
    fwd = [dict(m[0]), _row(3, 14, -0.2, -3.0), _row(4, 14, -0.2, 2.0, status="OPEN"), _row(5, 3, 0.1, 2.0, strat="FRENZY_LONG"),
           _row(6, 14, -0.2, -0.5, reason="EXTERNAL_CLOSE"), _row(7, 14, -0.2, 2.0, strat="FRENZY_LITE_RETRY")]
    out = "\n".join(S.run(paths=[_csv(tmp_path, fwd, "f.csv")], master_path=mp, state_path=None))
    assert "master (today's stack) 1 + forward exports 2" in out and "operator-closed out (master 1, forward 1)" in out
    fwd2 = [dict(m[1]), dict(m[2])]                                   # a stack-blocked + an operator-closed MASTER row seen again in an export
    out2 = "\n".join(S.run(paths=[_csv(tmp_path, fwd2, "f2.csv")], master_path=mp, state_path=None))
    assert "forward exports 0" in out2 and "forward 0)" in out2
    m_, _, _ = S.load_master(mp)
    assert list(m_.y) == [-2.5]                                       # today's stack, not as traded


def test_collecting_below_fifteen_on_eight_days(tmp_path):
    rows = [_row(i, 13, -0.1, -3.0) for i in range(14)]
    v, txt = S.gate(_pool(tmp_path, rows))
    assert v == "COLLECTING" and "14/15" in txt


def _pool(tmp_path, fwd, master=()):
    m, _, keys = S.load_master(_master(tmp_path, list(master)) if master else os.path.join(str(tmp_path), "none.csv"))
    f, _ = S.load_forward(keys, [_csv(tmp_path, fwd, "fw.csv")])
    p = pd.concat([m, f], ignore_index=True)
    p["grp"] = S.classify(p)
    return p


def test_arm_candidate_when_the_cell_loses(tmp_path):
    cell = [_row(i, 13, -0.1, -3.0 if i % 4 else 2.0, day=i // 2) for i in range(16)]       # 16 fills · 8 days · 25 % WR
    rest = [_row(100 + i, 3, 0.1, 2.0 if i % 3 else -3.0, day=i // 3) for i in range(30)]
    v, txt = S.gate(_pool(tmp_path, cell + rest))
    assert v == "ARM_CANDIDATE", txt


def test_retire_when_the_cell_beats_rest_and_master_counts(tmp_path):
    master = [_row(i, 13, -0.1, 2.0, day=i) for i in range(8)]                             # master history counts toward the read
    cell = [_row(20 + i, 13, -0.1, 2.0 if i % 2 else -3.0, day=10 + i) for i in range(8)]
    rest = [_row(100 + i, 3, 0.1, -3.0 if i % 2 else 2.0, day=i) for i in range(20)]
    v, txt = S.gate(_pool(tmp_path, cell + rest, master))
    assert v == "RETIRE" and "read at 15 cell fills" in txt


def test_keep_observing_when_one_day_carries_the_loss(tmp_path):
    cell = [_row(i, 13, -0.1, -0.2, day=i) for i in range(14)] + [_row(50, 13, -0.1, -10.0, day=20)]   # all lose · one day = 78 % of it
    rest = [_row(100 + i, 3, 0.1, 2.0 if i % 3 else -3.0, day=i) for i in range(30)]
    v, txt = S.gate(_pool(tmp_path, cell + rest))
    assert v == "KEEP_OBSERVING" and "WR 0 %" in txt and "P(avg < 0) 1.00" in txt and "worst day 78%" in txt


def test_master_can_only_veto_never_rush_an_arm(tmp_path):
    # forward alone: the cell beats its forward rest → no arm; a weak master REST must not turn that into an arm (deep-review case)
    cell = [_row(i, 13, -0.1, -3.0 if i % 3 else 2.0, day=i // 2) for i in range(16)]                 # forward cell avg −1.33
    frest = [_row(100 + i, 3, 0.1, -3.0 if i % 4 else 2.0, day=i // 2) for i in range(16)]            # forward rest avg −1.75
    master = [_row(300 + i, 3, 0.1, 2.0 if i % 2 else -2.0, day=-40 + i) for i in range(40)]          # master rest avg 0.00
    v, txt = S.gate(_pool(tmp_path, cell + frest, master))
    assert v != "ARM_CANDIDATE", txt
    assert S.gate(_pool(tmp_path, cell + frest))[0] == "RETIRE"


def test_master_only_cell_never_arms(tmp_path):
    master = [_row(i, 13, -0.1, -3.0 if i % 4 else 2.0, day=i // 2) for i in range(16)] + \
             [_row(100 + i, 3, 0.1, 2.0 if i % 3 else -3.0, day=i // 3) for i in range(30)]
    v, txt = S.gate(_pool(tmp_path, [], master))
    assert v == "KEEP_OBSERVING" and "forward-only: cell 0" in txt


def test_decided_once_on_the_crossing_prefix(tmp_path):
    cell = [_row(i, 13, -0.1, -3.0 if i % 4 else 2.0, day=i // 2) for i in range(16)]
    late_winners = [_row(200 + i, 13, -0.1, 3.0, day=30 + i) for i in range(30)]          # after the read: never change it
    a, _ = S.gate(_pool(tmp_path, cell))
    b, _ = S.gate(_pool(tmp_path, cell + late_winners))
    assert a == b


def test_empty_inputs_never_raise(tmp_path):
    out = "\n".join(S.run(paths=[], master_path=os.path.join(str(tmp_path), "missing.csv"), state_path=None))
    assert "COLLECTING" in out and "AVAUSDT" in out


def test_wired_into_the_opportunity_scout():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_lite_late_btcfall as _llb" in src and "_rg_sec += _llb.run(now_ms)" in src


def test_verdict_is_persisted_and_never_moves(tmp_path):
    st = str(tmp_path / "gate.json")
    assert S.frozen_verdict(st, "COLLECTING", "x")[:2] == ("COLLECTING", "x") and not os.path.exists(st)
    v, t, now = S.frozen_verdict(st, "ARM_CANDIDATE", "read at 15")
    assert now and v == "ARM_CANDIDATE"
    v2, t2, now2 = S.frozen_verdict(st, "RETIRE", "later data")
    assert v2 == "ARM_CANDIDATE" and t2.startswith("read at 15") and "frozen" in t2 and not now2
