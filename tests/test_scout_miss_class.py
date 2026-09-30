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
