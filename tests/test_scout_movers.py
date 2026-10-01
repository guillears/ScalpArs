"""🔭 Scout movers: 24 h volume + feature anchor (pure helpers, no network)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))
import opportunity_scout as S  # noqa: E402

BAR = S.BAR


def _frame(n=400, t0=1_700_000_000_000 // BAR * BAR):
    idx = [t0 + i * BAR for i in range(n)]
    return pd.DataFrame({"o": 1.0, "h": 1.0, "l": 1.0, "c": 2.0, "v": 10.0, "tb": 5.0}, index=idx)


def test_anchor_is_the_bar_before_the_window():
    assert S.mover_anchor(10 * BAR) == 9 * BAR


def test_q24_sums_288_closed_bars_ending_at_the_anchor():
    d = _frame(); a = int(d.index[300])
    assert S.mover_q24(d, a) == 288 * 2.0 * 10.0
    d2 = d.copy(); d2.loc[d2.index[301]:, "v"] = 1e9          # bars AFTER the anchor never leak in
    assert S.mover_q24(d2, a) == 288 * 2.0 * 10.0


def test_q24_none_when_history_does_not_reach():
    d = _frame()
    assert S.mover_q24(None, int(d.index[300])) is None
    assert S.mover_q24(d, int(d.index[100])) is None          # fewer than 288 bars before
    assert S.mover_q24(d, int(d.index[-1]) + 5 * BAR) is None  # anchor bar itself missing


def test_stamp_movers_fills_q24_and_never_raises_without_market_data():
    d = _frame(); st = int(d.index[301])
    mv = pd.DataFrame([dict(pair="AAAUSDT", side="UP", move_4h=1.0, start_ts=st, end_ts=st + 47 * BAR, q24=np.nan),
                       dict(pair="ZZZUSDT", side="DOWN", move_4h=-2.0, start_ts=st, end_ts=st + 47 * BAR, q24=np.nan)])
    out = S.stamp_movers(mv, {"AAAUSDT": d}, {}, None, set(), int(d.index[-1]))
    assert out.q24.iloc[0] == 288 * 20.0 and pd.isna(out.q24.iloc[1]) and len(out) == 2


def _stub(calls):
    def stamp(view, btc_full, alts, in_now, last_closed):
        calls.append(view.copy())
        fv = pd.to_numeric(view["feat_v"], errors="coerce") if "feat_v" in view else pd.Series(np.nan, index=view.index)
        v = view.copy()
        for c in ("feat_v", "feat_bar_ts", "entry_rsi", "entry_pair_volume_24h_usd"):
            if c not in v.columns:
                v[c] = np.nan
            v[c] = v[c].astype(object)
        need = v.index[fv.isna()]
        v.loc[need, "feat_v"] = 2; v.loc[need, "feat_bar_ts"] = v.loc[need, "bar_ts"]
        v.loc[need, "entry_rsi"] = 55.0; v.loc[need, "entry_pair_volume_24h_usd"] = v.loc[need, "qvol24_event"]
        return v
    return stamp


def test_big_movers_are_stamped_on_the_right_rows_and_slides_restamp(monkeypatch):
    d = _frame(); st = int(d.index[301]); calls = []
    monkeypatch.setattr(S, "stamp_features", _stub(calls))
    mv = pd.DataFrame([dict(pair="AAAUSDT", side="UP", move_4h=1.0, start_ts=st, end_ts=st + 47 * BAR, q24=None),
                       dict(pair="AAAUSDT", side="DOWN", move_4h=-9.0, start_ts=st + BAR, end_ts=st + 48 * BAR, q24=None),
                       dict(pair="BBBUSDT", side="UP", move_4h=7.0, start_ts=st, end_ts=st + 47 * BAR, q24=None)], index=[7, 3, 5])
    alts = {"AAAUSDT": d, "BBBUSDT": d}
    out = S.stamp_movers(mv, alts, {"AAAUSDT": 4}, None, set(), int(d.index[-1]))
    assert list(out.pair) == ["AAAUSDT", "AAAUSDT", "BBBUSDT"] and list(out.start_ts) == [st, st + BAR, st]     # rows / start untouched
    assert pd.isna(out.feat_v.iloc[0]) and out.feat_v.iloc[1] == 2 and out.feat_v.iloc[2] == 2                    # only the ≥ 5 % rows
    assert out.feat_bar_ts.iloc[1] == st and out.feat_bar_ts.iloc[2] == st - BAR                                  # anchor = start − 1 bar
    v = calls[0]; assert (v.bar_ts == v.start_ts).all() and set(v.type) == {"MOVER"}                              # no look-ahead view
    assert "type" not in out.columns and "bar_ts" not in out.columns and "rank" not in out.columns
    n = len(calls); out2 = S.stamp_movers(out, alts, {}, None, set(), int(d.index[-1]))
    assert out2.feat_v.iloc[1] == 2 and out2.entry_rsi.iloc[1] == 55.0 and len(calls) == n + 1                    # second run: nothing new
    slid = out2.copy(); slid.loc[1, "start_ts"] = st + 11 * BAR; d2 = d.copy(); d2.loc[d2.index[305]:, "v"] = 20.0
    out3 = S.stamp_movers(slid, {"AAAUSDT": d2, "BBBUSDT": d2}, {}, None, set(), int(d.index[-1]))
    assert out3.feat_bar_ts.iloc[1] == st + 10 * BAR                                                              # re-stamped at the new anchor
    assert out3.q24.iloc[1] == S.mover_q24(d2, st + 10 * BAR) != out2.q24.iloc[1]                                 # volume refreshed too
    assert out3.entry_pair_volume_24h_usd.iloc[1] == out3.q24.iloc[1]


def test_failure_returns_the_input_untouched(monkeypatch):
    d = _frame(); st = int(d.index[301])
    monkeypatch.setattr(S, "stamp_features", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    mv = pd.DataFrame([dict(pair="AAAUSDT", side="UP", move_4h=9.0, start_ts=st, end_ts=st + 47 * BAR, q24=None)])
    out = S.stamp_movers(mv, {"AAAUSDT": d}, {}, None, set(), int(d.index[-1]))
    assert out is mv


def test_pair_day_first_collapses_repeats_to_the_first_event():
    day = 86_400_000; t0 = 1_790_000_000_000 // day * day
    g = pd.DataFrame([dict(pair="MOVRUSDT", side="UP", type="TREND", bar_ts=t0 + 5 * BAR, day="d1", tbs2="STOP", closest_set="A"),
                      dict(pair="MOVRUSDT", side="UP", type="ALT_SPIKE", bar_ts=t0 + 5 * BAR, day="d1", tbs2="TARGET", closest_set="A"),
                      dict(pair="MOVRUSDT", side="UP", type="TREND", bar_ts=t0 + 90 * BAR, day="d1", tbs2="TARGET", closest_set="B"),
                      dict(pair="MOVRUSDT", side="DOWN", type="TREND", bar_ts=t0 + 95 * BAR, day="d1", tbs2="STOP", closest_set="A"),
                      dict(pair="MOVRUSDT", side="UP", type="TREND", bar_ts=t0 + 300 * BAR, day="d2", tbs2="STOP", closest_set="A")])
    u = S.pair_day_first(g)
    assert len(u) == 3 and list(u.sort_values("bar_ts").tbs2) == ["TARGET", "STOP", "STOP"]     # same bar: ALT_SPIKE sorts first
    assert len(S.pair_day_first(g, extra=("closest_set",))) == 4                                 # per gate set: A-up-d1, B-up-d1, A-down-d1, A-up-d2
    assert S.pair_day_first(g.iloc[0:0]) is not None and len(S.pair_day_first(g.iloc[0:0])) == 0


def test_diagnosis_lines_carry_pair_day_counts():
    day = 86_400_000; t0 = 1_790_000_000_000 // day * day
    rows = [dict(type="TREND", pair="AAAUSDT", side="UP", bar_ts=t0 + i * 60 * BAR, miss_class="FILTER_NEAR", tbs2=o, closest_set="PAIR_ADX_MAX")
            for i, o in enumerate(["STOP", "TARGET", "TARGET"])]
    allv = pd.DataFrame(rows)
    L = S.diagnosis_lines(allv)
    txt = "\n".join(L)
    assert "Pair-days: n · good · bad" in txt and "| FILTER_NEAR | 3 | 2 | 1 | 0 | 1 · 0 · 1 |" in txt     # 3 moves = ONE pair-day, first was a stop
    assert "| PAIR_ADX_MAX | 3 | 2 | 1 | 1 | 0 | 1 |" in txt
