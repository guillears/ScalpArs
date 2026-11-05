"""🎯 (DECISION_LOG 479) momentum-long fixed TPs from the recorded trade data — scripts/study_ml_fixed_tp_recorded.tp_outcome. Hermetic."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import study_ml_fixed_tp_recorded as S  # noqa: E402

NAN = np.nan


def r(before, peak, kind, pk=NAN, tr=NAN, pm=NAN, tm=NAN, fin=NAN):
    return dict(before=before, peak_pnl=peak, exit_kind=kind, post_exit_peak_pnl=pk, post_exit_trough_pnl=tr,
                post_exit_peak_minutes=pm, post_exit_trough_minutes=tm, post_exit_final_pnl=fin)


def test_peak_in_trade_books_the_tp_even_on_a_stopped_fill():
    assert S.tp_outcome(r(-0.7, 1.2, "STOP_LOSS"), 1.0) == (1.0, "TP in trade")
    assert S.tp_outcome(r(1.5, 2.1, "RUNNER_TRAIL"), 1.0) == (1.0, "TP in trade")


def test_loss_and_time_exits_are_unchanged_when_the_tp_was_not_reached():
    assert S.tp_outcome(r(-0.7, 0.4, "STOP_LOSS", pk=3.0, tr=0.1), 1.0) == (-0.7, "same exit")
    assert S.tp_outcome(r(0.1, 0.4, "NO_EXPANSION"), 1.0) == (0.1, "same exit")


def test_profit_exit_continues_on_the_45_min_record():
    assert S.tp_outcome(r(0.4, 0.8, "RUNNER_TRAIL", pk=1.3, tr=0.2), 1.0) == (1.0, "TP after exit")
    assert S.tp_outcome(r(0.4, 0.8, "RUNNER_TRAIL", pk=0.9, tr=-0.8), 1.0) == (S.STOP, "NEW STOP after exit")
    assert S.tp_outcome(r(0.4, 0.8, "RUNNER_TRAIL", pk=0.9, tr=0.1, fin=0.6), 1.0) == (0.6, "open at 45 min")


def test_both_hit_after_exit_always_books_the_stop():
    """extremes' times are not first touches — order unknown → the stop, whichever extreme came first (deep review 479)."""
    for pm, tm in ((20, 5), (5, 20), (NAN, NAN)):
        assert S.tp_outcome(r(0.4, 0.8, "RUNNER_TRAIL", pk=1.3, tr=-0.8, pm=pm, tm=tm), 1.0) == (S.STOP, "NEW STOP after exit")


def test_todays_stop_before_the_peak_credits_no_tp():
    """older eras' wider stops: a fill that fell to −0.70 before its peak would be stopped today first → compared as booked."""
    import pandas as pd
    F = pd.DataFrame({"trough_pnl": [-0.9, -0.9, -0.3, -0.9], "trough_reached_at": ["2026-08-05T10:00", "2026-08-05T12:00", "2026-08-05T10:00", None],
                      "peak_reached_at": ["2026-08-05T11:00", "2026-08-05T11:00", "2026-08-05T11:00", "2026-08-05T11:00"]})
    assert list(S.stop_before_peak(F)) == [True, False, False, False]
    row = dict(r(-0.9, 1.5, "STOP_LOSS"), stop_before_peak=True)
    assert S.tp_outcome(row, 1.0) == (-0.9, "same exit (today's stop first)")


def test_missing_post_exit_data_is_unpriced_not_zero():
    a, how = S.tp_outcome(r(0.4, 0.8, "RUNNER_TRAIL"), 1.0)
    assert np.isnan(a) and how == "no post-exit data"


def test_profit_exits_and_levels_are_the_operators():
    assert S.TPS == (0.5, 1.0, 2.0, 3.0) and S.STOP == -0.70
    assert set(S.PROFIT_EXITS) == {"RUNNER_TRAIL", "HARD_TP_LADDER", "HARD_TP", "TRAILING_STOP"}
