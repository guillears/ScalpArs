"""👁🎯 (DECISION_LOG 479) ML_FIXED_TP scout line — history + forward (operator 11-05), probes out, the 4-level luck search, the verdict, hook + registry."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_ml_fixed_tp_watch as W  # noqa: E402


def _export(tmp_path, rows, name="scalpars_orders_paper_2026-11-20_10-00-00.csv"):
    base = dict(direction="LONG", entry_strategy="MOMENTUM", status="CLOSED", close_reason="RUNNER_TRAIL", cell_multiplier_source="UNMATCHED",
                pattern_cell_source="UNMATCHED", peak_pnl=0.8, trough_pnl=-0.1, peak_reached_at=None, trough_reached_at=None, post_exit_peak_pnl=1.3,
                post_exit_trough_pnl=0.2, post_exit_peak_minutes=10, post_exit_trough_minutes=1, post_exit_final_pnl=1.0, notional_value=1000.0)
    d = pd.DataFrame([{**base, **r} for r in rows])
    p = tmp_path / name
    d.to_csv(p, index=False)
    return str(p)


def _master():
    M, _ = W.RT.load_master()
    return M.iloc[0:0]


def test_history_and_forward_split_probes_out_and_settling(tmp_path):
    rows = [dict(opened_at="2026-10-25T10:00:00", closed_at="2026-10-25T10:30:00", pair="AAAUSDT", pnl_percentage=0.4),        # B25
            dict(opened_at="2026-11-10T10:00:00", closed_at="2026-11-10T10:30:00", pair="BBBUSDT", pnl_percentage=0.4),        # FORWARD
            dict(opened_at="2026-11-10T11:00:00", closed_at="2026-11-10T11:30:00", pair="CCCUSDT", pnl_percentage=0.4,
                 cell_multiplier_source="MOM_PROBE"),                                                                           # probe → out
            dict(opened_at="2026-11-20T09:30:00", closed_at="2026-11-20T09:40:00", pair="DDDUSDT", pnl_percentage=0.4)]        # < 60 min → waits
    p = _export(tmp_path, rows)
    F, newest, waiting, _ = W.cohort([p], since=pd.Timestamp("2026-11-06"), master=_master(), sc_ship_ms=None, sc_delta={})
    assert dict(F.block.value_counts()) == {"B25": 1, "FORWARD": 1} and waiting == 1
    assert (F["TP1"] == 1.0).all() and (F["TP1_how"] == "TP after exit").all()


def test_luck_search_prices_the_best_of_four():
    D = pd.DataFrame({"a": [1.0] * 12, "b": [0.5] * 12})
    p = W.luck_search(D, [f"d{i}" for i in range(12)])
    assert p["a"] <= p["b"] and p["a"] < 0.01


def test_verdict_needs_history_and_forward_to_agree():
    n = 40
    g = pd.DataFrame({"before": [0.1] * n, "day": [f"2026-09-{i % 28 + 1:02d}" for i in range(n)], "pair": [f"P{i}" for i in range(n)],
                      "block": ["MASTER"] * 30 + ["FORWARD"] * 10})
    for x in W.TPS:
        g[f"TP{x:g}"] = g.before
    g["TP3"] = g.before + 0.3
    v = W.verdict(g)
    assert v["TP3"][0] == W.CANDIDATE and v["TP1"][0] == "NO EDGE"
    g.loc[g.block == "FORWARD", "TP3"] = g.before - 0.3                     # forward disagrees → never a candidate
    assert W.verdict(g)["TP3"][0] == "KEEP OBSERVING"
    assert W.verdict(g.iloc[:10])["TP3"][0] == "COLLECTING"


def test_hook_and_registry():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_ml_fixed_tp_watch as _mft")
    assert "_mft.run(now_ms)" in src[j:j + 200] and "except Exception as _mft_e" in src[j:j + 400]
    import observe_registry as OR
    assert OR.EXIT["scout_ml_fixed_tp_watch.py"]["dl"] == "479 · 534"
    row = [r for r in OR.status() if r[0] == "scout_ml_fixed_tp_watch.py"][0]
    # 🔁 (DL 534) re-based on the second-chance exit (DL 532) → matches today's config again; any drift fails here
    assert row[4] == "OK", row[5]


def test_master_keys_are_raw_so_a_rebuilt_master_never_double_counts():
    import study_ml_fixed_tp_recorded as RT
    raw = RT.master_keys()
    kept = RT.master_keys(RT.load_master()[0])
    assert kept < raw                                                   # unkept / probe master fills are still known to the dedup


def test_newest_export_wins_by_its_stamp_not_file_time(tmp_path):
    old = _export(tmp_path, [dict(opened_at="2026-10-25T10:00:00", closed_at="2026-10-25T10:30:00", pair="AAAUSDT", pnl_percentage=-0.5)],
                  name="scalpars_orders_paper_2026-11-10_10-00-00.csv")
    new = _export(tmp_path, [dict(opened_at="2026-10-25T10:00:00", closed_at="2026-10-25T10:30:00", pair="AAAUSDT", pnl_percentage=0.4)],
                  name="scalpars_orders_paper_2026-11-20_10-00-00.csv")
    os.utime(new, (1, 1))                                               # the newer export copied earlier on disk
    d, _ = W.load_exports([old, new])
    assert len(d) == 1 and float(d.pnl_percentage.iloc[0]) == 0.4
