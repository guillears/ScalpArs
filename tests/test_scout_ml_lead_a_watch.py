"""👁 (DECISION_LOG 481) ML_LEAD_A scout line — frozen cell, history + forward split, probes out, the verdict ladder, hook + registry."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_ml_lead_a_watch as W  # noqa: E402


def _export(tmp_path, rows, name="scalpars_orders_paper_2026-11-20_10-00-00.csv"):
    base = dict(direction="LONG", entry_strategy="MOMENTUM", status="CLOSED", close_reason="RUNNER_TRAIL", cell_multiplier_source="UNMATCHED",
                pattern_cell_source="UNMATCHED", entry_ema20_slope=0.2, entry_rsi_prev=55.0)
    p = tmp_path / name
    pd.DataFrame([{**base, **r} for r in rows]).to_csv(p, index=False)
    return str(p)


def test_cell_is_frozen_and_unscored_never_counts():
    d = pd.DataFrame({"entry_ema20_slope": [0.1482, 0.1481, 0.3, np.nan], "entry_rsi_prev": [60.9, 50, 61.0, 50]})
    assert list(W.cell(d)[:3]) == [True, False, False] and pd.isna(W.cell(d).iloc[3])


def test_history_and_forward_split_probes_out(tmp_path):
    rows = [dict(opened_at="2026-10-25T10:00:00", closed_at="2026-10-25T10:30:00", pair="AAAUSDT", pnl_percentage=0.4),
            dict(opened_at="2026-11-10T10:00:00", closed_at="2026-11-10T10:30:00", pair="BBBUSDT", pnl_percentage=0.4),
            dict(opened_at="2026-11-10T11:00:00", closed_at="2026-11-10T11:30:00", pair="CCCUSDT", pnl_percentage=0.4, cell_multiplier_source="MOM_PROBE")]
    M = pd.DataFrame(columns=["opened_at", "pair", "pct", "entry_ema20_slope", "entry_rsi_prev"])
    F, _, _ = W.cohort([_export(tmp_path, rows)], since=pd.Timestamp("2026-11-06"), master=M)
    assert dict(F.block.value_counts()) == {"B25": 1, "FORWARD": 1} and (F.cell == True).all()   # noqa: E712


def _nh(cell_pct, rest_pct, n_cell=30, newest=(0, 0.0, 0.0), days=20):
    rows = [dict(block="MASTER", day=f"2026-09-{i % days + 1:02d}", pair=f"H{i}", pct=cell_pct + (i % 3) * 0.1, cell=True) for i in range(n_cell)]
    rows += [dict(block="MASTER", day=f"2026-09-{i % days + 1:02d}", pair=f"R{i}", pct=rest_pct + (i % 3) * 0.1, cell=False) for i in range(60)]
    rows += [dict(block="B25", day=f"2026-10-{i % 8 + 24:02d}", pair=f"N{i}", pct=newest[1], cell=True) for i in range(newest[0])]
    rows += [dict(block="B25", day=f"2026-10-{i % 8 + 24:02d}", pair=f"M{i}", pct=newest[2], cell=False) for i in range(newest[0])]
    d = pd.DataFrame(rows)
    d["cell"] = d.cell.astype(object)
    return d


def test_verdict_reads_history_and_batch_pooled():
    assert W.verdict(_nh(0.8, 0.0))[0] == W.CANDIDATE                       # history alone can make the call (operator 11-06)
    assert W.verdict(_nh(0.0, 0.1))[0] == "RETIRE"                          # pooled cell not above the rest
    assert W.verdict(_nh(0.8, 0.0, n_cell=20))[0] == "KEEP OBSERVING"       # Pattern W needs N ≥ 30
    assert W.verdict(_nh(0.8, 0.0, newest=(6, -0.5, 0.3)))[0] == "KEEP OBSERVING"   # the newest trades contradict
    assert W.verdict(_nh(0.8, 0.0, newest=(4, -0.5, 0.3)))[0] == W.CANDIDATE        # < 5 newest cell fills: not yet a contradiction


def test_run_end_to_end_with_unscored_breadth(tmp_path):
    # 🌡 (DL 489) a fill past the 11-06 study cache with no engine stamp: the cache-only rebuild cannot score it → the injected (failed) read decides
    rows = [dict(opened_at="2027-01-10T10:00:00", closed_at="2027-01-10T10:30:00", pair="ZZZUSDT", pnl_percentage=0.4)]
    M = pd.DataFrame(columns=["opened_at", "pair", "pct", "entry_ema20_slope", "entry_rsi_prev"])
    out = "\n".join(W.run(now_ms=1_800_000_000_000, paths=[_export(tmp_path, rows)], since_ms=1_790_000_000_000,
                          breadth_fn=lambda t, n: (pd.Series(dtype=float), False), master=M))
    assert "Market-hot read incomplete" in out and "| _unscored (Market-hot or a cell leg missing) — never counted_ | 1 |" in out


def test_hook_and_registry():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_ml_lead_a_watch as _mla")
    assert "_mla.run(now_ms)" in src[j:j + 200] and "except Exception as _mla_e" in src[j:j + 400]
    import observe_registry as OR
    assert OR.BOOKED["scout_ml_lead_a_watch.py"] == OR.EXIT["scout_ml_fixed_tp_watch.py"]["walks"]
    row = [r for r in OR.status() if r[0] == "scout_ml_lead_a_watch.py"][0]
    assert row[4] == "OK", row[5]
