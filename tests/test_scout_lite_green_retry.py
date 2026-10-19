"""👁 LITE_GREEN_RETRY observe line (DECISION_LOG 315): next-red-bar selection on the engine's own frenzy_lite_status (green, green, red →
red · stretch broken → none · outside 16.75 h → none · episode ended · bars run out → pending), LITE C TP choice (and parity with the
engine rule), the export-rebuilt open checks, 24 h volume, the frozen gate verdicts. No network."""
import os
import sys
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
import scout_lite_green_retry as LG  # noqa: E402
from services.frenzy import frenzy_lite_status, frenzy_lite_c_tp  # noqa: E402

M5 = 300_000
SPIKE = 1_800_000_000_000
T0 = SPIKE + 5 * 3_600_000                                    # the refused bar, 5 h into the episode


def _ep(k, red=True, streak0=14, hours=None, vol=40.0, spike=SPIKE, in_state=False, reset=False):
    """the engine ep for the k-th bar after the refused one (k = 0 = the refused bar)."""
    last = T0 + k * M5
    s = (1 if reset else streak0 + k)
    return dict(verified=True, hours=(hours if hours is not None else (last - spike) / 3_600_000 + 5 / 60), in_state=in_state,
                above_streak=s, vol_mult=vol, last_bar_ts=last, spike_ts=spike, bar_red=red, bar_ret_pct=(-0.2 if red else 0.3))


def _st(ep):
    return ep, frenzy_lite_status(ep, LG.TH, 50e6, None)[1]


def _sid0():
    from services.frenzy import frenzy_lite_stretch_id
    return frenzy_lite_stretch_id(_ep(0, red=False))


def test_refused_bar_rebuilds_as_green():
    assert _st(_ep(0, red=False))[1] == "FRENZY_LITE_GREEN_BAR"


def test_green_green_red_picks_the_red():
    seq = [_st(_ep(1, red=False)), _st(_ep(2, red=False)), _st(_ep(3, red=True)), _st(_ep(4, red=True))]
    assert LG.next_red(seq, _sid0(), SPIKE) == (2, "READY")


def test_vol_high_and_in_state_bars_are_skipped():
    seq = [_st(_ep(1, red=True, vol=120.0)), _st(_ep(2, red=True, in_state=True)), _st(_ep(3, red=True))]
    assert LG.next_red(seq, _sid0(), SPIKE) == (2, "READY")


def test_stretch_broken_gives_no_fill():
    seq = [_st(_ep(1, red=False)), _st(_ep(2, red=True, reset=True)), _st(_ep(3, red=True))]
    assert LG.next_red(seq, _sid0(), SPIKE) == (1, "BROKEN")


def test_outside_the_lite_window_gives_no_fill():
    seq = [_st(_ep(1, red=False)), _st(_ep(2, red=True, hours=16.8))]
    assert seq[1][1] == "FRENZY_LITE_TOO_LATE"
    assert LG.next_red(seq, _sid0(), SPIKE) == (1, "WINDOW_OVER")


def test_new_episode_or_unflagged_ends_it():
    assert LG.next_red([_st(_ep(1, spike=SPIKE + M5))], _sid0(), SPIKE) == (0, "EPISODE_ENDED")
    assert LG.next_red([(None, "FRENZY_LITE_NOT_FLAGGED")], _sid0(), SPIKE) == (0, "EPISODE_ENDED")


def test_bars_run_out_is_pending():
    assert LG.next_red([_st(_ep(1, red=False))], _sid0(), SPIKE) == (None, "PENDING")
    assert LG.next_red([], _sid0(), SPIKE) == (None, "PENDING")


# ─────────────────────────── LITE C ───────────────────────────
@pytest.mark.parametrize("vm,sl,tp", [(20.0, 0.1, 3.0), (19.9, 5.0, 2.0), (50.0, 0.0, 2.0), (50.0, -1.0, 2.0), (None, 1.0, 2.0),
                                      (float("nan"), 1.0, 2.0), (30.0, float("nan"), 2.0)])
def test_lite_c_tp(vm, sl, tp):
    assert LG.lite_c_tp(vm, sl) == tp
    th = NS(frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0, frenzy_lite_c_enabled=True, frenzy_lite_c_vol_mult=20.0,
            frenzy_lite_c_btc_slope_min=0.0, frenzy_lite_c_tp_pct=2.0)
    assert frenzy_lite_c_tp(th, vm, sl)[0] == tp                  # the engine rule agrees at the frozen values


# ─────────────────────────── open checks / volume ───────────────────────────
def _orders(rows):
    o = pd.DataFrame(rows, columns=["pair", "entry_strategy", "_o", "_x"])
    return o


def test_open_checks():
    te = T0 + 3 * M5
    big = np.iinfo("int64").max
    assert LG.open_check(_orders([("AUSDT", "FRENZY_LONG", te - M5, big)]), "AUSDT", te) == "PAIR_HELD"
    assert LG.open_check(_orders([("AUSDT", "FRENZY_LONG", te - 2 * M5, te - M5)]), "AUSDT", te) is None
    two = _orders([("BUSDT", "FRENZY_LITE", te - M5, te + M5), ("CUSDT", "FRENZY_LITE", te - M5, te + M5)])
    assert LG.open_check(two, "AUSDT", te) == "SLOTS"
    day0 = te // 86_400_000 * 86_400_000
    three = _orders([("AUSDT", s, day0 + i, day0 + i + 1) for i, s in enumerate(("FRENZY_LONG", "FRENZY_WILLY", "FRENZY_LITE"))])
    assert LG.open_check(three, "AUSDT", te) == "PAIR_DAY_CAP"
    assert LG.willy_hold(_orders([("ZUSDT", "FRENZY_WILLY", te - M5, te + M5)]), te)
    assert not LG.willy_hold(_orders([("ZUSDT", "FRENZY_LONG", te - M5, te + M5)]), te)


def test_vol24():
    rows = [[i * M5, 1, 2.0, 1.0, 3.0, 10.0] for i in range(300)]
    assert LG.vol24_at(rows, 286) is None
    assert LG.vol24_at(rows, 299) == pytest.approx(288 * 10.0 * 2.0)


# ─────────────────────────── gate ───────────────────────────
def _fills(n_days, per_day, pct, pairs=None):
    rows = []
    for d in range(n_days):
        for k in range(per_day):
            rows.append(dict(day=f"2026-11-{d + 1:02d}", pair=(pairs[k % len(pairs)] if pairs else f"P{d}_{k}"),
                             pct=(pct(d, k) if callable(pct) else pct)))
    return pd.DataFrame(rows)


def test_verdict_collecting():
    assert LG.verdict(_fills(9, 3, 1.0), 0.1)[0] == "collecting (27/20 on 9/10 days)"
    assert LG.verdict(_fills(10, 1, 1.0), 0.1)[0].startswith("collecting (10/20")


def test_verdict_candidate():
    v, det = LG.verdict(_fills(10, 3, lambda d, k: 2.94 if k < 2 else -3.12), 0.2)   # 67 % WR ≥ the 58 % LITE C breakeven
    assert v.startswith("CANDIDATE") and det["p"] >= 0.9


def test_verdict_needs_breakeven_win_rate():
    """(315 deep dive) a positive mean at 50 % WR (< the 58 % LITE C breakeven) is never a CANDIDATE."""
    v, det = LG.verdict(_fills(10, 2, lambda d, k: 2.94 if k == 0 else -0.5), 0.2)
    assert v.startswith("hold") and det["wr"] < LG.BE_WR


def test_verdict_retire():
    v, _ = LG.verdict(_fills(10, 2, lambda d, k: 2.94 if k == 0 else -3.12), 0.2)
    assert v.startswith("RETIRE")


def test_verdict_hold_below_live_lite_or_concentrated():
    good = _fills(10, 2, lambda d, k: 2.94 if k == 0 else -0.5)
    assert LG.verdict(good, 5.0)[0].startswith("hold")                       # below the same-period live LITE mean
    assert LG.verdict(good, float("nan"))[0].startswith("hold")              # no live comparison yet
    conc = _fills(10, 2, lambda d, k: (20.0 if d == 0 else 0.3) if k == 0 else -0.25)
    assert LG.verdict(conc, 0.0)[0].startswith("hold")                       # one day ≥ 50 % of the gain
    pc = _fills(10, 2, lambda d, k: 2.94 if k == 0 else -0.5, pairs=["AUSDT", "BUSDT"])
    assert LG.verdict(pc, 0.0)[0].startswith("hold")                         # one pair ≥ 50 % of the gain


def test_orders_frame_ignores_non_fills_and_bounds_open_rows(monkeypatch):
    """deep review (315): SIGNAL_EXPIRED rows never hold a pair; an OPEN row is open ≤ 12 h (never forever)."""
    rows = pd.DataFrame([
        dict(opened_at="2026-05-01T10:00:00", closed_at=None, pair="PUMPUSDT", direction="LONG", status="SIGNAL_EXPIRED", entry_strategy="MOMENTUM", pnl_percentage=None, is_paper=True, _k="2026-05-01T10:00:00"),
        dict(opened_at="2026-10-19T17:10:00", closed_at=None, pair="AUSDT", direction="LONG", status="OPEN", entry_strategy="FRENZY_LITE", pnl_percentage=None, is_paper=False, _k="2026-10-19T17:10:00"),
    ])
    monkeypatch.setattr(LG.LS, "load_orders", lambda cols=None: rows)
    o = LG.orders_frame()
    assert list(o.pair) == ["AUSDT"]
    t_open = int(pd.Timestamp("2026-10-19T18:00:00").value // 1_000_000)
    t_late = int(pd.Timestamp("2026-10-20T06:00:00").value // 1_000_000)
    assert LG.open_check(o, "AUSDT", t_open) == "PAIR_HELD" and LG.open_check(o, "AUSDT", t_late) is None
    assert LG.open_check(o, "PUMPUSDT", t_open) is None
