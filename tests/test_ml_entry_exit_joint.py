"""🎯×🚪 (DECISION_LOG 477) momentum-long entry × exit joint read — scripts/study_ml_entry_exit_joint.py (+ worker). Hermetic: synthetic
paths / fills; the replica evaluator is exercised on a hand-made path, no cache, no network."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
import study_ml_entry_exit_joint as J  # noqa: E402


def test_fixed_tp_levels_are_the_operators_and_pre_declared():
    assert J.FTP_LEVELS == (0.5, 1.0, 2.0, 3.0) and J.FTP_R_LEVELS == (1.0, 2.0, 3.0)
    V = lambda **kw: dict(base=True, **kw)
    v = J.ftp_variants(V)
    assert set(v) == {"FTP0.5", "FTP1", "FTP2", "FTP3", "FTP1_R", "FTP2_R", "FTP3_R"}
    assert v["FTP1"]["tp"] == 1.0 and v["FTP1"]["ladder"] == [] and v["FTP1"]["run_arm"] > 50          # plain: no ladder, no runner
    assert v["FTP2_R"]["tp"] == 2.0 and v["FTP2_R"]["ladder"] == [] and "run_arm" not in v["FTP2_R"]    # runner kept
    assert J.exit_columns()[:5] == list(J.REGISTERED.values())                                          # the DL 380 lines first


def test_replica_takes_profit_at_the_fixed_level_and_base_is_unchanged():
    import study_ml_exit_b25_run as RUN
    n = 200
    t = 1_700_000_000_000 + np.arange(n) * 60_000
    net = np.r_[np.linspace(0, 1.6, 100), np.linspace(1.6, -0.9, 100)]                 # up to +1.6 then down through the stop
    P = dict(t=t, net=net, p=100 * (1 + net / 100), e13=np.zeros(n), flip=np.zeros(n, bool), sig=np.zeros(n, bool), src="1m")
    fx = dict(o_ms=int(t[0]), atr=0.6, rsi_entry=None, E=100.0)
    V = RUN.V
    k, why, pct = RUN.evaluate(P, fx, J.ftp_variants(V)["FTP1"])
    assert why == "FIXED_TP" and 1.0 <= pct < 1.05
    k0, why0, pct0 = RUN.evaluate(P, fx, RUN.VARIANTS["BASE"])
    assert why0 != "FIXED_TP"                                                           # today's exit never takes a fixed TP
    k3, why3, pct3 = RUN.evaluate(P, fx, J.ftp_variants(V)["FTP3"])
    assert why3 != "FIXED_TP" and pct3 < 0                                              # +3 never reached → today's stop


def test_entry_cohorts_keep_unscored_out_of_the_kept_rows():
    F = pd.DataFrame({"ML_MKT_HOT": [True, False, np.nan, False], "ML_W4": [False, False, False, True],
                      "STOP_COOLDOWN": [False, False, False, False], "PAIR_1H_DOWNTREND": [False, np.nan, False, False]})
    c = J.entry_cohorts(F)
    assert c["ALL fills"].sum() == 4
    assert list(c["kept by ML_MKT_HOT"]) == [False, True, False, True]                 # the NaN fill is not "kept"
    assert list(c["kept by PAIR_1H_DOWNTREND"]) == [True, False, True, True]
    assert list(c["kept by EVERY line"]) == [False, False, False, False]


def test_summary_deltas_are_against_the_replica_base_and_split_by_block():
    rng = np.random.default_rng(3)
    n = 40
    F = pd.DataFrame({"BASE": rng.normal(0.1, 0.5, n), "day": [f"d{i // 2}" for i in range(n)],
                      "block": ["MASTER"] * 30 + ["B25"] * 10})
    for c in J.exit_columns():
        F[c] = F.BASE
    F["FTP1"] = F.BASE + 0.2
    T = J.summarise(F, {"ALL fills": pd.Series(True, index=F.index)}, J.exit_columns(), r=200)
    r = T[T.exit == "FTP1"].iloc[0]
    assert r.sum_d == pytest.approx(8.0) and r.sum_d_master == pytest.approx(6.0) and r.sum_d_b25 == pytest.approx(2.0)
    assert r.better == 100 and r.p_alone <= 0.05
    z = T[T.exit == "W_LAD_off"].iloc[0]
    assert z.sum_d == 0 and z.better == 0 and z.worse == 0


def test_worker_prices_exactly_the_reported_exits():
    src = open(os.path.join(ROOT, "scripts", "study_ml_entry_exit_joint_worker.py"), encoding="utf-8").read()
    assert 'J.ftp_variants(RUN.V)' in src and "J.REGISTERED.values()" in src and '"BASE": RUN.VARIANTS["BASE"]' in src
    main = open(os.path.join(ROOT, "scripts", "study_ml_entry_exit_joint.py"), encoding="utf-8").read()
    assert "_patch_b25(export)" in main and "LT.gain_line" in main and "entry × exit combinations were looked at" in main


def test_fixed_tp_is_booked_at_the_level_never_the_bar_high_and_end_of_data_is_out():
    import study_ml_entry_exit_joint_worker as W
    assert W.booked("FTP1", "FIXED_TP", 1.37) == 1.0                                   # 1m bar high overshoot → the limit price
    assert W.booked("FTP2_R", "RH_FIXED_TP", 2.4) == 2.0
    assert W.booked("FTP1", "STOP_LOSS", -0.7) == -0.7
    assert np.isnan(W.booked("FTP3", "END_OF_DATA", 0.4)) and np.isnan(W.booked("BASE", "ERR:x", 0.1))


def test_path_quality_split_counts_new_stops_and_tick_vs_1m():
    F = pd.DataFrame({"BASE": [0.5, 0.2, -0.7], "path_src": ["tick", "1m", "tick"],
                      "BASE_why": ["RUNNER_TRAIL", "RUNNER_TRAIL", "STOP_LOSS"]})
    for c in J.exit_columns():
        F[c] = F.BASE
        F[c + "_why"] = F.BASE_why
    F["FTP2"] = [0.5, -0.7, -0.7]
    F["FTP2_why"] = ["RUNNER_TRAIL", "STOP_LOSS", "STOP_LOSS"]
    X = J.split_rows(F, J.exit_columns()).set_index("exit")
    r = X.loc["FTP2"]
    assert r.new_stops == 1 and r.new_stops_1m == 1 and r.new_stops_tick == 0
    assert r.sum_d_new_stops == pytest.approx(-0.9) and r.sum_d_1m == pytest.approx(-0.9) and r.sum_d_tick == 0
