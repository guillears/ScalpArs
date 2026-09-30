"""🧩 Sep-30 — the opportunity scout's missed-move classes (DECISION_LOG 151): an untraded move is judged by the CLOSEST full
gate set the bot refused it with (FAILS lines). Precedence: CAPACITY (a capacity gate fired) › EXECUTION (maker expired) ›
UNIVERSE › PRE_FAILS (window older than the first FAILS line) › FILTER_NEAR (≤2 gates) › FILTER_FAR (≥3) › SLEEVE."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import opportunity_scout as S  # noqa: E402

BAR = S.BAR
T0 = 1_790_000_000_000


def _df(rows, cols):
    d = pd.DataFrame(rows, columns=cols)
    d["ms"] = d["t"]
    return d


def _dec(fails=(), blocks=(), expired=()):
    fl = _df([], ["t", "pair", "dir", "strategy"])
    pos = _df([], ["t", "pair", "dir", "strategy", "closed_ms", "exp_ms"])
    bl = _df(list(blocks), ["t", "pair", "dir", "gate", "n"])
    fs = _df(list(fails), ["t", "pair", "dir", "gate", "n", "src"])
    ex = _df(list(expired), ["t", "pair", "dir"])
    return (fl, bl, pos, ex, [[T0 - 20 * BAR, T0 + 60 * BAR]], fs)


ROW = dict(type="ALT_SPIKE", pair="QNTUSDT", side="UP", bar_ts=T0, start_ts=T0, end_ts=T0 + 2 * BAR, in_universe=True)


def _cls(dec, row=ROW, book_max=4):
    S.FAILS_FROM[0] = T0 - 10 * BAR
    return S.bot_check(row, None, [], book_max, dec)


def test_closest_set_decides_near_vs_far():
    r = _cls(_dec(fails=[(T0, "QNTUSDT", "LONG", "A+B+C", 5, "MOMENTUM"), (T0 + BAR, "QNTUSDT", "LONG", "A+B", 1, "MOMENTUM")]))
    assert len(r) == 7 and r[4] == "FILTER_NEAR" and r[6] == "A+B"
    r = _cls(_dec(fails=[(T0, "QNTUSDT", "LONG", "A+B+C", 5, "MOMENTUM")]))
    assert r[4] == "FILTER_FAR"


def test_macro_only_single_and_disabled_sleeve_ignored():
    r = _cls(_dec(fails=[(T0, "QNTUSDT", "LONG", "MACRO:BTC_ADX_GATE_LOW", 2, "MOMENTUM")]))
    assert r[4] == "FILTER_NEAR" and r[6] == "MACRO:BTC_ADX_GATE_LOW"
    r = _cls(_dec(fails=[(T0, "QNTUSDT", "LONG", "FLIP_LONG_DISABLED", 3, "FLIP:FAN_RATIO_GATE")]))
    assert r[4] == "SLEEVE"


def test_precedence_capacity_execution_prefails():
    r = _cls(_dec(blocks=[(T0, "QNTUSDT", "LONG", "BOOK_FULL", 1)], fails=[(T0, "QNTUSDT", "LONG", "A", 1, "MOMENTUM")]))
    assert r[4] == "CAPACITY"
    r = _cls(_dec(expired=[(T0 + BAR, "QNTUSDT", "LONG")], fails=[(T0, "QNTUSDT", "LONG", "A", 1, "MOMENTUM")]))
    assert r[4] == "EXECUTION"
    S.FAILS_FROM[0] = T0 + 100 * BAR
    assert S.bot_check(ROW, None, [], 4, _dec(blocks=[(T0, "QNTUSDT", "LONG", "EMA_STACK", 1)]))[4] == "PRE_FAILS"


def test_ladder_pass_engine_chain_block_is_near():
    r = _cls(_dec(blocks=[(T0, "QNTUSDT", "LONG", "LONG_HEAT", 2)]))
    assert r[4] == "FILTER_NEAR" and "LONG_HEAT" in r[6]


def _path(closes, t0):
    idx = [t0 + i * BAR for i in range(len(closes))]
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"o": c, "h": c, "l": c, "c": c, "v": 1.0})


def test_exit_shape_fields_long_and_short():
    """📐 4 h exit-design fields: best / worst over 240 min, minutes to the FIRST bar at the best, deepest dip up to and INCLUDING
    that bar (a dip AFTER the peak never counts); needs the full 48 bars."""
    closes = [100.0] + [99.0, 98.0] + [101.0] * 3 + [105.0] + [90.0] * 20 + [105.0] + [90.0] * 24   # tie at 105 later: first wins
    d = _path(closes, T0)
    d.loc[T0 + 6 * BAR, "l"] = 97.5                                                   # the peak bar's own low counts (conservative)
    o = S._series_outcome(d, T0, +1)
    assert abs(o["mfe240"] - 5.0) < 1e-9 and abs(o["mae240"] + 10.0) < 1e-9
    assert o["t_peak_min"] == 30 and abs(o["mae_before_peak"] + 2.5) < 1e-9
    assert abs(o["mae_before_peak_x"] + 2.0) < 1e-9                                   # the shallow end excludes the peak bar's wick
    m = _path([200.0 - c for c in closes], T0)                                       # the mirror path for a SHORT
    s = S._series_outcome(m, T0, -1)
    assert s["t_peak_min"] == 30 and abs(s["mae_before_peak"] + 2.0) < 1e-9          # worst high before the low: 102 vs 100 → −2 %
    assert abs(s["mfe240"] - 5.0) < 1e-9 and abs(s["mae240"] + 10.0) < 1e-9          # low 95 → +5 %, high 110 → −10 %
    assert "mfe240" in S._series_outcome(_path(closes[:49], T0), T0, +1)            # 48 window bars → the 4 h fields
    assert "mfe240" not in S._series_outcome(_path(closes[:48], T0), T0, +1)        # 47 → none (never a partial 4 h read)
