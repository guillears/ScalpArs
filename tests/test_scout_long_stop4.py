"""👁 LONG_STOP4 observe line (DECISION_LOG 315): the study walker copy (−3 stop saved by −4 then TP · deeper · non-stopped Δ = 0 ·
same-bar stop + TP = stop · gap through the stop), the tick → 1 s / 1m path builder, the aggTrades parser, the frozen gate verdicts and
the walker's parity with the untracked study source (skipped when absent). No network."""
import ast
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_long_stop4 as LS  # noqa: E402

MIN = 60_000
TE = 1_800_000_000_000                                   # a whole minute
BASE_STOP = -3 - 0.03                                     # 🔧 (355) net −3 stop + slippage, like live
SH_STOP = -4 - 0.03
TP_NET = 1.5                                              # a TP books exactly the net target (live)


def _path(bars, te=TE):
    """bars = [(o, h, l, c) gross %] → 1-minute path dict (t = bar end)."""
    b = np.asarray(bars, float)
    t = te + MIN * np.arange(1, len(b) + 1)
    P = dict(t=t.astype("int64"), op=b[:, 0], hi=b[:, 1], lo=b[:, 2], cl=b[:, 3], te=te, end_ms=int(t[-1]), n_bar=len(b))
    P["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(P["hi"])[:-1]]
    return P


def _full(bars, fill=(0.5, 0.6, 0.4, 0.5)):
    return _path(list(bars) + [fill] * (721 - len(bars)))


# ─────────────────────────── walker cases ───────────────────────────
def test_minus3_stop_saved_by_minus4_then_tp():
    s = LS.shadow(_full([(0, 0.2, -3.5, -3.2), (-3.2, 0.0, -3.4, -0.5), (-0.5, 1.7, -0.6, 1.6)]))
    assert s["base_why"] == "STOP" and s["base"] == pytest.approx(BASE_STOP)
    assert s["shadow_why"] == "TP" and s["shadow"] == pytest.approx(TP_NET)
    assert s["delta"] == pytest.approx(TP_NET - BASE_STOP)


def test_deeper_stop_costs_one_point():
    s = LS.shadow(_full([(0, 0.2, -3.5, -3.2), (-3.2, -3.0, -4.6, -4.4)]))
    assert (s["base_why"], s["shadow_why"]) == ("STOP", "STOP")
    assert s["shadow"] == pytest.approx(SH_STOP)
    assert s["delta"] == pytest.approx(SH_STOP - BASE_STOP)          # −1


def test_non_stopped_fill_delta_is_exactly_zero():
    for bars in ([(0, 1.6, -1.0, 1.5)], [(0, 0.5, -2.9, -2.0), (-2.0, 1.6, -2.1, 1.5)]):
        s = LS.shadow(_full(bars))
        assert s["base_why"] == s["shadow_why"] == "TP"
        assert s["delta"] == 0.0
    s = LS.shadow(_full([], fill=(0.1, 0.4, -2.5, 0.2)))             # neither line in 12 h → CAP, identical
    assert s["base_why"] == s["shadow_why"] == "CAP" and s["delta"] == 0.0


def test_same_bar_stop_and_tp_is_stop():
    s = LS.shadow(_full([(0, 1.8, -3.2, 1.0)]))
    assert s["base_why"] == "STOP" and s["base"] == pytest.approx(BASE_STOP)
    s4 = LS.shadow(_full([(0, 1.8, -4.2, 1.0)]))
    assert s4["shadow_why"] == "STOP"


def test_bar_opening_through_the_stop_fills_at_its_open():
    b, why, _ = LS.walk(_full([(0, 0.1, -0.1, 0.0), (-3.6, -3.5, -3.7, -3.6)]), tp=1.5, sl=3.0, cap=720)
    assert why == "STOP" and b == pytest.approx(-3.6 - 0.045 - 0.045 * (1 - 0.036) - 0.03)


def test_incomplete_window_is_unres():
    b, why, _ = LS.walk(_path([(0, 0.4, -0.4, 0.1)] * 30), tp=1.5, sl=4.0, cap=720)
    assert why == "UNRES"


# ─────────────────────────── path builder ───────────────────────────
def test_assemble_path_ticks_then_1m_tail():
    E = 100.0
    te = TE + 30_500                                                 # fill mid-minute
    tt = np.array([te - 10, te, te + 200, te + 900, te + 1_500, te + 59_000], dtype="int64")
    pp = np.array([99.0, 99.5, 100.0, 101.0, 99.0, 100.5])
    cov = TE + 2 * MIN                                               # ticks cover up to here
    k1 = [[TE + 2 * MIN, 100.5, 101.0, 100.0, 100.2], [TE + 3 * MIN, 100.2, 100.3, 96.0, 97.0], [TE + 5 * MIN, 1, 1, 1, 1]]
    P = LS.assemble_path(te, E, 720, ticks=(tt, pp, cov), k1m=k1)
    # prints strictly after te: 3 one-second buckets (te+200 & te+900 share the second after te? te ends in .500 → buckets by (t−1)//1000)
    assert P["n_tick_s"] >= 2 and P["n_bar"] == 2                    # the 1m tail stops at the first gap
    assert P["t"][-1] == TE + 4 * MIN and P["end_ms"] == TE + 4 * MIN
    assert P["lo"].min() == pytest.approx(-4.0)
    assert P["hi"][:P["n_tick_s"]].max() == pytest.approx(1.0)
    b, why, _ = LS.walk(P, tp=1.5, sl=3.0, cap=720)
    assert why == "STOP"


def test_assemble_path_without_ticks_starts_at_the_next_whole_minute():
    te = TE + 12_000
    k1 = [[TE, 1, 1, 1, 1], [TE + MIN, 100, 101, 99, 100]]
    P = LS.assemble_path(te, 100.0, 720, ticks=None, k1m=k1)
    assert P["n_bar"] == 1 and P["t"][0] == TE + 2 * MIN and P["n_tick_s"] == 0


def test_rebase_reprices_to_the_entry():
    P = _path([(0, 2.0, -1.0, 1.0)])
    Q = LS.rebase(P, 1.01)
    assert Q["hi"][0] == pytest.approx((1.02 / 1.01 - 1) * 100)


def test_parse_agg_zip(tmp_path):
    csv = "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n2,1.5,3,1,1,1700000000200,true\n1,1.4,3,1,1,1700000000100,false\n"
    z = tmp_path / "X-aggTrades-2026-10-19.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("X-aggTrades-2026-10-19.csv", csv)
    t, p = LS.parse_agg_zip(str(z))
    assert list(t) == [1700000000100, 1700000000200] and list(p) == [1.4, 1.5]


def test_klines_cache_only_mode_never_fetches(tmp_path):
    bud = LS.Budget()
    assert LS.fapi_klines("XUSDT", "1m", TE, 10, bud, str(tmp_path), now=TE + 3600_000, fetch=False) is None
    assert bud.requests == 0


# ─────────────────────────── gate ───────────────────────────
def _rows(n_days, per_day, saved_per_day, delta_saved=4.56, delta_deeper=-1.0, extra_ok=0, viol=False):
    rows = []
    for d in range(n_days):
        day = f"2026-11-{d + 1:02d}"
        for k in range(per_day):
            sv = k < saved_per_day
            rows.append(dict(day=day, pair=f"P{k}", live_stopped=True, base_why="STOP", shadow_why="TP" if sv else "STOP",
                             delta=delta_saved if sv else delta_deeper))
        for _ in range(extra_ok):
            rows.append(dict(day=day, pair="Q", live_stopped=False, base_why="TP", shadow_why="TP", delta=0.0))
    if viol:
        rows.append(dict(day="2026-11-01", pair="V", live_stopped=False, base_why="STOP", shadow_why="TP", delta=4.56))
    return pd.DataFrame(rows)


def test_verdict_collecting():
    v, _ = LS.verdict(_rows(7, 3, 2))
    assert v.startswith("collecting (21/15 stops on 7/8 days)")
    v, _ = LS.verdict(_rows(14, 1, 1)[:14])
    assert v.startswith("collecting (14/15")


def test_verdict_candidate():
    v, det = LS.verdict(_rows(8, 2, 1, extra_ok=2))
    assert v.startswith("CANDIDATE") and det["saved"] == pytest.approx(0.5) and det["violations"] == 0


def test_verdict_retire_on_saved_share():
    v, det = LS.verdict(_rows(8, 4, 1, delta_saved=20.0))          # 25 % saved, mean Δ still > 0
    assert det["saved"] == pytest.approx(0.25) and v.startswith("RETIRE")


def test_verdict_retire_on_mean():
    v, det = LS.verdict(_rows(8, 2, 1, delta_saved=0.5, delta_deeper=-1.0))
    assert det["mean"] <= 0 and v.startswith("RETIRE")


def test_verdict_hold_on_integrity_violation():
    v, det = LS.verdict(_rows(8, 2, 1, extra_ok=2, viol=True))
    assert det["violations"] == 1 and v.startswith("hold")


def test_live_stopped_definition():
    assert LS.live_stopped("STOP_LOSS", -3.0) and not LS.live_stopped("FRENZY_TP", -3.0)
    assert LS.live_stopped(None, -2.5) and not LS.live_stopped(float("nan"), 1.0)


# ─────────────────────────── parity with the study ───────────────────────────
def test_walker_is_the_study_walker():
    src = os.path.join(ROOT, "scripts", "study_frenzy_realfills_common.py")
    if not os.path.exists(src):
        pytest.skip("study source not present")
    tree = ast.parse(open(src).read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "walk")
    ns = dict(np=np, MIN=MIN, ENTRY=0.045, TAKER=0.045, MAKER=0.018, SLIP=0.03)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), src, "exec"), ns)
    rng = np.random.default_rng(3)
    for _ in range(40):
        steps = rng.normal(0, 0.6, 721).cumsum()
        bars = [(c - 0.1, c + abs(rng.normal(0, 0.5)), c - abs(rng.normal(0, 0.5)), c) for c in steps]
        P = _path(bars)
        for sl in (3.0, 4.0):
            assert LS.walk(P, tp=1.5, sl=sl, cap=720) == ns["walk"](P, tp=1.5, sl=sl, cap=720)


def test_walk_net_books_the_live_net_targets():
    """🔧 (DECISION_LOG 355) the live bot judges TP / stop on NET pnl — a TP win books +1.5 (not ≈ +1.437), a stop −sl − slippage."""
    b, why, _ = LS.walk_net(_full([(0, 1.6, -0.1, 1.5)]), tp=1.5, sl=3.0)
    assert why == "TP" and b == pytest.approx(1.5)
    b, why, _ = LS.walk_net(_full([(0, 1.55, -0.1, 1.5)]), tp=1.5, sl=3.0)      # gross +1.55 does not yet book net +1.5
    assert why != "TP"
    b, why, _ = LS.walk_net(_full([(0, 0.1, -2.95, -2.9)]), tp=1.5, sl=3.0)     # gross −2.95 is already net ≤ −3
    assert why == "STOP" and b == pytest.approx(-3.0 - 0.03)
    b, why, _ = LS.walk_net(_full([(0, 0.1, -2.85, -2.8)]), tp=1.5, sl=3.0)
    assert why != "STOP"
