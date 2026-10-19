"""WILLY_AVGDOWN scout (DECISION_LOG 313): the shadow walker (= the study's AD1 walker at −2.00) on synthetic 1m bars + the frozen gate.
No network: the 1m frame is built in memory and the pair has no local tick archive."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_willy_avgdown as S  # noqa: E402

M1 = 60_000
T0 = 1_790_000_040_000 - (1_790_000_040_000 % M1)   # a whole minute
TE = T0 + 10_000                                    # entry 10 s into the minute → the first bar is T0 + 1 min
PAIR = "ZZTESTNOTICKUSDT"


def _bars(base=(0.99, 0.992, 0.988, 0.99), events=None, n=S.HORIZON + 1):
    """n 1m bars (o, h, l, c) from the first whole minute after entry; events = {bar index: (o, h, l, c)}. Entry price = 1.0."""
    rows = []
    for i in range(n):
        o, h, l, c = (events or {}).get(i, base)
        rows.append((T0 + M1 * (i + 1), o, h, l, c))
    return pd.DataFrame(rows, columns=["open_time", "o", "h", "l", "c"])


def _shadow(k):
    return S.shadow(S.path_of(PAIR, TE, 1.0, k), TE)


def test_no_touch_no_add():
    s = _shadow(_bars(base=(1.0, 1.002, 0.998, 1.0), events={5: (1.0, 1.02, 0.999, 1.01)}))
    assert s["base_why"] == "TP" and s["base"] == pytest.approx(1.25, abs=1e-6)
    assert not s["trig"] and s["leg"] == pytest.approx(0.0)
    assert s["merged"] == pytest.approx(s["base"])


def test_touch_then_tp_from_new_average():
    # bar 1 dips to −2 net (add), bar 5 rallies to 1.005: above the merged TP (~1.0038) but far below the original TP (1.0134)
    s = _shadow(_bars(events={1: (0.99, 0.992, 0.980, 0.985), 5: (0.99, 1.005, 0.989, 1.0)}))
    assert s["trig"] and s["merged_why"] == "TP"
    assert s["merged"] == pytest.approx(2.5, abs=0.05)        # both legs ≈ +1.25 of the new average, in % of the original notional
    assert s["base_why"] == "CAP" and s["base"] < 0           # the original alone rides to the 90-min cap at ~0.99
    assert s["leg"] == pytest.approx(s["merged"] - s["base"]) and s["leg"] > 3


def test_touch_then_stop_at_original_price_doubles_loss():
    s = _shadow(_bars(events={1: (0.99, 0.992, 0.980, 0.985), 3: (0.985, 0.986, 0.95, 0.955)}))
    assert s["base_why"] == "STOP" and s["base"] == pytest.approx(-4.0, abs=1e-6)
    assert s["trig"] and s["merged_why"] == "STOP"
    assert s["leg"] < -2.0                                    # the add loses −2 more + fees / slip on the same stop price
    assert s["merged"] == pytest.approx(s["base"] + s["leg"])


def test_late_tp_after_minute_30():
    # no add: the original's late TP (+0.25 from minute 30) is hit at bar 40, never the +1.25
    s = _shadow(_bars(base=(1.0, 1.002, 0.998, 1.0), events={40: (1.0, 1.004, 0.999, 1.003)}))
    assert not s["trig"] and s["base_why"] == "TP" and s["base"] == pytest.approx(0.25, abs=1e-6)
    # with an add: the merged late TP is measured from the NEW average (~0.9939), hit at bar 40 (high 0.995)
    s = _shadow(_bars(events={1: (0.99, 0.992, 0.980, 0.985), 40: (0.99, 0.995, 0.989, 0.99)}))
    assert s["trig"] and s["add_min"] < 30 and s["merged_why"] == "TP"
    assert s["base_why"] == "CAP" and s["merged"] == pytest.approx(0.5, abs=0.05)


def test_same_bar_stop_and_touch_is_stop():
    s = _shadow(_bars(events={1: (0.99, 0.992, 0.95, 0.955)}))   # one bar touches the add line AND the stop → add, then stop
    assert s["base_why"] == "STOP" and s["trig"] and s["merged_why"] == "STOP" and s["merged"] < s["base"] - 2
    s = _shadow(_bars(base=(1.0, 1.002, 0.998, 1.0), events={1: (1.0, 1.02, 0.95, 1.0)}))   # TP and stop on one bar → stop
    assert s["base_why"] == "STOP"


def _adds(n=30, days=8, pairs=6, leg=0.5, worst_merged=-2.0):
    r = [dict(day=f"2026-11-{1 + i % days:02d}", pair=f"P{i % pairs}USDT", leg=leg + 0.1 * (i % 3), leg_usd=(leg + 0.1 * (i % 3)) * 23,
              merged=(worst_merged if i == 0 else 1.0)) for i in range(n)]
    return pd.DataFrame(r)


def test_verdict_collecting():
    assert S.verdict(_adds(n=12))[0] == "collecting (12/30 on 8/8 days)"
    assert S.verdict(_adds(n=40, days=5))[0].startswith("collecting (40/30 on 5/8")


def test_verdict_discuss():
    v, det = S.verdict(_adds())
    assert v.startswith("DISCUSS AD1") and det["p"] >= 0.95 and det["pair_share"] < 0.5


def test_verdict_drop_cases():
    assert S.verdict(_adds(leg=-0.6))[0].startswith("DROP")                       # mean < 0
    assert S.verdict(_adds(pairs=1))[0].startswith("DROP")                        # one pair = 100 % of the gain
    assert S.verdict(_adds(worst_merged=-6.5))[0].startswith("DROP")              # merged −6.5 % of notional → −12.2 % of the account
    assert S.verdict(_adds(worst_merged=-6.2))[0].startswith("DISCUSS")           # a plain add-then-stop (≈ −6.2 %) still passes
    d = _adds()
    d["leg"] = np.where(np.arange(30) % 2 == 0, 3.0, -2.9)                        # mean ≈ +0.05, day-bootstrap not confident
    d["day"] = [f"2026-11-{1 + (i // 4) % 8:02d}" for i in range(30)]
    d.loc[d.day.isin(["2026-11-01", "2026-11-02"]), "leg"] = -2.9
    assert S.verdict(d)[0].startswith("DROP")


def test_day_boot_p_bounds():
    assert S.day_boot_p([1.0, 2.0], ["a", "b"]) == 1.0
    assert S.day_boot_p([-1.0, -2.0], ["a", "b"]) == 0.0


def test_worst_check_is_sizing_invariant():
    """deep review: the worst check reads merged % × the reference notional / account — not the fill's own $ (no drift with the balance)."""
    a, b = _adds(worst_merged=-6.0), _adds(worst_merged=-6.0)
    b["merged_usd"] = -6.0 * 9_000 / 100                                        # a much bigger live notional must not change the verdict
    assert S.verdict(a)[0] == S.verdict(b)[0] and S.verdict(a)[1]["worst_acct"] == pytest.approx(-6.0 * 2300 / 1225)


def test_gap_open_through_add_line_fills_at_open():
    s = _shadow(_bars(events={1: (0.975, 0.976, 0.974, 0.975)}))                 # bar 1 opens below the −2 % line
    assert s["trig"] and s["add_min"] is not None


def test_fills_cohort_long_only_and_dedup(monkeypatch, tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame([
        dict(opened_at="2026-10-19T14:00:00", pair="AUSDT", direction="LONG", status="OPEN", entry_strategy="FRENZY_WILLY", entry_price=1, pnl_percentage=None, pnl=None, notional_value=2300, is_paper=False),
        dict(opened_at="2026-10-19T14:00:00.5", pair="AUSDT", direction="LONG", status="CLOSED", entry_strategy="FRENZY_WILLY", entry_price=1, pnl_percentage=1.2, pnl=27, notional_value=2300, is_paper=False),
        dict(opened_at="2026-10-19T12:59:59", pair="BUSDT", direction="LONG", status="CLOSED", entry_strategy="FRENZY_WILLY", entry_price=1, pnl_percentage=1.2, pnl=27, notional_value=2300, is_paper=False),
        dict(opened_at="2026-10-19T15:00:00", pair="CUSDT", direction="SHORT", status="CLOSED", entry_strategy="FRENZY_WILLY", entry_price=1, pnl_percentage=1.2, pnl=27, notional_value=2300, is_paper=False),
        dict(opened_at="2026-10-19T15:00:00", pair="DUSDT", direction="LONG", status="CLOSED", entry_strategy="FRENZY_LITE", entry_price=1, pnl_percentage=1.2, pnl=27, notional_value=2300, is_paper=False),
    ]).to_csv(f, index=False)
    monkeypatch.setattr(S.EG, "glob_exports", lambda pattern: [str(f)])
    a = S.fills()
    assert list(a.pair) == ["AUSDT"] and a.status.tolist() == ["CLOSED"]
