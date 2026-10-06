"""📊 Oct-6 momentum-short pair-volume ceiling shadow (scripts/ms_pvr_shadow.py + the MS_PVR_BLOCKED gate in
scripts/scout_revert_gates.py). Pinned: the engine's PVR formula (EWM span 5 / SMA 20 incl. the forming bar), the refusal
classification (A = what a revert to 1.0 re-admits), the frozen kept-side revert bar (< 70 % on N ≥ 15) and reading rule, and the
momentum-short exit replica's order of exits. Pure — no network."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import ms_pvr_shadow as S  # noqa: E402
import scout_revert_gates as R  # noqa: E402

MIN, B5 = S.MIN, S.B5


def test_selftests_pass():
    S.selftest()
    R.selftest()


def test_pvr_matches_indicators_formula():
    rng = np.random.default_rng(3)
    vols = list(rng.uniform(50, 150, 99))
    part = 80.0
    v = pd.Series(vols + [part])
    want = v.ewm(span=5, adjust=False).mean().iloc[-1] / v.rolling(20).mean().iloc[-1]
    assert abs(S.pvr_from_volumes(vols, part) - want) < 1e-12


def test_refusal_classes_bound_the_readmit_zone():
    assert S.classify_pvr([0.86, 0.99])[0] == "A"           # 0.86 inclusive = blocked today; < 1.0 = re-admitted by the revert
    assert S.classify_pvr([0.855, 0.859])[0] == "unk"       # never blocked by a 0.86 ceiling
    assert S.classify_pvr([1.0])[0] == "B"                  # still blocked after a revert to 1.0


def test_overridden_gate_replaced_by_frozen_226_gates():
    assert not hasattr(S, "kept_gate_state") and not hasattr(S, "reading_rule")      # the Sep-18 70 %/15 revert is overridden
    assert (S.REVERT_N, S.KEPT_MIN_FRESH, S.REVIEW_N, S.BREAKEVEN_WR, S.PVR_CEIL, S.PVR_OLD) == (15, 5, 20, 59.0, 0.86, 1.0)
    assert (S.KEPT_FROM, S.SHADOW_FROM) == ("2026-09-18 12:00", "2026-10-06 18:00")


def test_gate1_revert_insufficient_then_mean_vs_kept():
    assert S.revert_gate([], -0.3) == "insufficient"
    assert S.revert_gate([1.0] * 14, -0.3) == "insufficient"             # never 'consistent' / 'holds' by default under 15
    assert S.revert_gate([-0.2] * 15, -0.3) == "fired"                   # A mean ≥ kept mean → revert to 1.0
    assert S.revert_gate([-0.3] * 15, -0.3) == "fired"                   # ≥ is inclusive
    assert S.revert_gate([-0.4] * 15, -0.3) == "holds"
    assert S.revert_gate([-0.4] * 15 + [5.0] * 10, -0.3) == "holds"      # only the first 15 signals


def test_gate1_kept_reference_falls_back_below_5_fresh_fills():
    t = ["2026-09-20T10:00:00", "2026-09-21T10:00:00", "2026-10-07T10:00:00"]
    A = pd.DataFrame(dict(opened_at=t, pair=["XUSDT"] * 3, status=["CLOSED"] * 3, pnl_percentage=[1.0, -0.5, 0.2], pnl=[1, -1, 1],
                          pvr=[0.5] * 3, stack_keep=[True] * 3))
    A["o_ms"] = S._ms_series(A.opened_at).astype("int64")
    st, lab = S.kept_ref(A)
    assert st["n"] == 3 and "09-18" in lab                               # 1 fresh fill < 5 → the kept side since 09-18 12:00
    B = pd.concat([A] + [A.iloc[[2]]] * 4, ignore_index=True)
    st, lab = S.kept_ref(B)
    assert st["n"] == 5 and lab.startswith("kept since 10-06")


def test_gate2_sleeve_review_at_20():
    assert S.review_gate([1.0] * 19) == "collecting"
    assert S.review_gate([1.0] * 11 + [-1.0] * 9) == "review"           # 55 % < 59 %
    assert S.review_gate([1.0] * 12 + [-1.0] * 8) == "holds"            # 60 %
    first, full = R.ms_review_first([("a", "CLOSED", 1.0)] * 25, None, 20)
    assert full and len(first) == 20
    assert R.ms_review_first([("z", "CLOSED", -9.0)] * 25, [list(x) for x in first], 20)[0] == first   # frozen keys win


def test_kept_tally_dedup_and_filters():
    A = pd.DataFrame(dict(opened_at=["2026-09-17T10:00:00", "2026-09-20T10:00:00", "2026-09-21T10:00:00", "2026-09-22T10:00:00",
                                     "2026-09-23T10:00:00"],
                          pair=["XUSDT"] * 5, status=["CLOSED"] * 5, pnl_percentage=[1.0, 0.5, -0.7, 0.3, -0.2],
                          pnl=[10, 5, -7, 3, -2], pvr=[0.5, 0.5, 0.7, 0.9, 0.6], stack_keep=[True, True, True, True, False]))
    A["o_ms"] = S._ms_series(A.opened_at).astype("int64")
    f, st = S.kept_tally(A)
    assert list(f.pnl_percentage) == [0.5, -0.7]             # before the floor / PVR ≥ 0.86 / not stack-kept are out
    assert st["n"] == 2 and st["wins"] == 1
    assert S.kept_tally(A, stack_only=False)[1]["n"] == 3


def test_ema13_cross_is_suppressed_once_the_runner_armed():
    # EMA13 at 99.5 below a 100 short entry; EMA5 > EMA8 (stack flipped). Print 1 (99.45) is below EMA13 and lifts the peak to ≈ +0.46;
    # print 2 (99.55) is above EMA13 → the cross. Runner floor ≈ 0.46 − min(0.5·5, 0.35·0.46) ≈ +0.30 < the +0.36 print, ladder needs 1.0,
    # K-trail needs stretch ≤ half its peak (0.65 vs 0.75) → nothing else can close it.
    t = np.arange(3, dtype=np.int64) * 1000
    px = np.array([99.45, 99.55, 99.55])
    sc = dict(ts=np.array([0], np.int64), px=np.array([100.0]), e5=np.array([100.2]), e8=np.array([100.1]), e13=np.array([99.5]),
              e20=np.array([100.0]))
    r = S.simulate_ms(t, px, 100.0, 5.0, sc)
    assert (r["reason"], r["i"]) == ("OPEN_END", 2)                      # previous peak ≥ 0.40 → the cross is suppressed
    r = S.simulate_ms(t, px, 100.0, 5.0, sc, cfg=dict(run_arm=0.60))
    assert (r["reason"], r["i"]) == ("EMA13_CROSS_EXIT", 1)               # unarmed control: the same print closes on the cross


def test_scan_table_uses_last_print_at_or_before_the_scan():
    k5_ot = np.arange(-120, 0, dtype=np.int64) * B5
    k5_c = np.full(120, 100.0)
    t = np.array([0, 1000, 2000], np.int64)
    sc = S.scan_table(k5_ot, k5_c, np.array([1500], np.int64), t, np.array([100.0, 101.0, 102.0]))
    assert sc["px"][0] == 101.0                                          # never the 102 print after the scan
