"""👁 Oct-30: the ML_MKT_HOT watch line (scripts/scout_ml_mkt_hot_watch.py, DECISION_LOG 416) — frozen constants, breadth, cohort, the operator's bar, the deferrals, freeze once, damaged state, the hook and the registry.
Hermetic: temp exports and state, injected breadth, no network, no git."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_ml_mkt_hot_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-01 00:00")
SINCE = int(T0.value // 10**6)
NOW = int((T0 + pd.Timedelta(days=60)).value // 10**6)


def _fills(rows):
    """rows = [(open time, pair, pct)] → CLOSED full-size momentum longs."""
    return pd.DataFrame([dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair, direction="LONG", entry_strategy="MOMENTUM",
                              status="CLOSED", pnl_percentage=pct, pnl=pct * 10, close_reason="RUNNER_TRAIL" if pct > 0 else "STOP_LOSS L1", cell_multiplier_source="UNMATCHED") for o, pair, pct in rows])


def _write(tmp_path, df, stamp="2026-12-30_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _breadth():
    """0.8 on odd-hour bars (the entries of hour 1:xx read bar 01:00), 0.2 on even-hour bars."""
    idx = pd.date_range(T0 - pd.Timedelta(days=5), T0 + pd.Timedelta(days=40), freq="1h")
    return pd.Series(np.where(idx.hour % 2 == 1, 0.8, 0.2), index=idx)


def _zr(zone, rest=0.5, days=8):
    """zone fills at hour 1:xx (breadth 0.8), rest fills at hour 2:xx (breadth 0.2)."""
    rows = [(T0 + pd.Timedelta(days=i % days, hours=1, minutes=i), f"Z{i}USDT", p) for i, p in enumerate(zone)]
    rows += [(T0 + pd.Timedelta(days=i % days, hours=2, minutes=i), f"R{i}USDT", rest) for i in range(20)]
    return _fills(rows)


def _cohort(tmp_path, d):
    c, _ = W.load(_write(tmp_path, d))
    s = W.score(c, _breadth())
    return s[s.zone == "zone"], s[s.zone == "rest"]


def test_frozen_constants():
    assert (W.DL, W.HOT_CUT, W.MIN_COINS, W.N_MIN, W.DAYS_MIN, W.AVG_MAX, W.P_MIN, W.SHARE_MAX, W.WAIT_MIN) == (416, 0.50, 30, 15, 8, 0.05, 0.90, 0.50, 1200)
    assert not hasattr(W, "UNIVERSE") and W.MB.TOP_N == 100 and W.MB.EXCLUDED == {"BTCUSDT", "ETHUSDT"}   # 🌡 (DL 489) V2 replaced the frozen 108
    assert "reports/SCOUT_ML_MKT_HOT.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_breadth_share_and_min_coins():
    idx = pd.date_range("2026-11-01", periods=6, freq="1h")
    up = pd.DataFrame({f"C{i}": [1, 1, 1, 1, 2, 2] for i in range(30)}, index=idx)
    dn = pd.DataFrame({f"D{i}": [2, 2, 2, 2, 1, 1] for i in range(10)}, index=idx)
    b = W.breadth_series(pd.concat([up, dn], axis=1))
    assert list(b.dropna().index) == list(idx[4:]) and abs(b.dropna().iloc[0] - 0.75) < 1e-9
    assert b.iloc[:4].isna().all()                                            # bars with data but no 4 h-earlier close: the rule's NULL, kept as NaN
    assert W.breadth_series(dn).dropna().empty                                # too few coins


def test_score_boundary_closed_bar_stale_and_unscored(tmp_path):
    br = pd.Series([0.50, 0.51, 0.2, 0.9], index=pd.date_range(T0, periods=4, freq="1h"))
    d = _fills([(T0 + pd.Timedelta(minutes=30), "AUSDT", 1.0), (T0 + pd.Timedelta(hours=1, minutes=59), "BUSDT", 1.0),
                (T0 - pd.Timedelta(hours=1), "CUSDT", 1.0), (T0 + pd.Timedelta(hours=6), "DUSDT", 1.0)])
    c, _ = W.load(_write(tmp_path, d))
    s = W.score(c, br)
    assert dict(zip(s.pair, s.zone)) == {"AUSDT": "rest", "BUSDT": "zone", "CUSDT": "unscored", "DUSDT": "unscored"}   # 0.50 itself = rest; no bar yet / stale → unscored


def test_cohort_excludes_probes_shorts_other_sleeves(tmp_path):
    d = _fills([(T0, s, 1.0) for s in ("AUSDT", "PUSDT", "SUSDT", "LUSDT")])
    d.loc[1, "cell_multiplier_source"] = "GAPFLAT_PROBE"
    d.loc[2, "direction"] = "SHORT"
    d.loc[3, "entry_strategy"] = "FRENZY_LONG"
    assert W.load(_write(tmp_path, d))[0].pair.tolist() == ["AUSDT"]


def test_verdict_outcomes(tmp_path):
    st, det = W.verdict(*_cohort(tmp_path, _zr([0.3] * 12 + [-1.3] * 3)))
    assert st == W.CANDIDATE and "WR 80%" in det, det                               # breakeven at a high WR qualifies: the operator's bar has no breakeven leg
    assert W.verdict(*_cohort(tmp_path, _zr([0.6] * 15)))[0] == "RETIRED"           # zone ≥ rest
    assert W.verdict(*_cohort(tmp_path, _zr([0.2] * 15)))[0] == "KEEP OBSERVING"    # behind rest but above +0.05
    rows = _zr([-0.8] * 10 + [0.3] * 5)
    rows.loc[:9, "pair"] = "SAMEUSDT"
    st, det = W.verdict(*_cohort(tmp_path, rows))
    assert st == "KEEP OBSERVING" and "top pair 100%" in det, det
    assert W.verdict(*_cohort(tmp_path, _zr([-0.8] * 15, days=1)))[0] == "COLLECTING"


def test_freeze_once_deferrals_and_damaged_state(tmp_path):
    d = _zr([-0.8] * 12 + [0.3] * 3)
    last = pd.to_datetime(d.opened_at).max()
    st = tmp_path / "w.json"
    fn = lambda *a: (_breadth(), True)
    early = _write(tmp_path, d, (last + pd.Timedelta(minutes=30)).strftime("%Y-%m-%d_%H-%M-%S"))
    assert "newest export < the crossing fill + 1200 min" in "\n".join(W.run(NOW, early, str(st), SINCE, fn)) and not st.exists()
    late = _write(tmp_path, d, (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    assert "the Market-hot read stopped early" in "\n".join(W.run(NOW, late, str(st), SINCE, lambda *a: (_breadth(), False))) and not st.exists()
    out = "\n".join(W.run(NOW, late, str(st), SINCE, fn))
    assert "Frozen verdict" in out and "Before → after at the crossing: before" in out and json.load(open(st))["ML"]["state"] == W.CANDIDATE
    W.run(NOW, _write(tmp_path, _zr([0.6] * 15), "2027-01-05_00-00-00"), str(st), SINCE, fn)
    assert json.load(open(st))["ML"]["state"] == W.CANDIDATE                          # never re-decided
    for bad in ('{"ML": "x"}', '{"ML": {"state": "x"}}'):
        st.write_text(bad)
        o = "\n".join(W.run(NOW, late, str(st), SINCE, fn))
        assert "ML: frozen record damaged" in o and "Frozen verdict" not in o and st.read_text() == bad


def test_unscored_fill_in_prefix_holds_the_freeze(tmp_path):
    d = _zr([-0.8] * 12 + [0.3] * 3)
    gap = _fills([(T0 + pd.Timedelta(hours=1, seconds=30), "GAPUSDT", -0.8)])
    gap["opened_at"] = (T0 - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S")    # after the floor, before the first breadth bar
    br = _breadth()
    br = br[br.index >= T0]
    since = int((T0 - pd.Timedelta(hours=2)).value // 10**6)
    last = pd.to_datetime(d.opened_at).max()
    st = tmp_path / "w.json"
    p = _write(tmp_path, pd.concat([d, gap]), (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, p, str(st), since, lambda *a: (br, True)))
    assert "an unscored fill sits inside the prefix" in out and not st.exists()


def test_sleeves_decide_apart_and_lite_never_freezes(tmp_path):
    d = _zr([-0.8] * 12 + [0.3] * 3)
    w = d.copy(); w["entry_strategy"] = "FRENZY_WILLY"; w["pnl_percentage"] = 0.9; w["pnl"] = 9.0     # WILLY: the zone earns as much as rest
    lite = d.copy(); lite["entry_strategy"] = "FRENZY_LITE"
    other = d.copy(); other["entry_strategy"] = "FRENZY_LONG"
    last = pd.to_datetime(d.opened_at).max()
    st = tmp_path / "w.json"
    p = _write(tmp_path, pd.concat([d, w, lite, other]), (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, p, str(st), SINCE, lambda *a: (_breadth(), True)))
    rec = json.load(open(st))
    assert rec["ML"]["state"] == W.CANDIDATE and rec["FRENZY_WILLY"]["state"] == "RETIRED" and "FRENZY_LITE" not in rec and "FRENZY_LONG" not in out
    assert "### FRENZY_LITE (display only, never decided)" in out and "### FRENZY_WILLY_HT" in out


def _v2_fakes(monkeypatch, members, cover_ms=0):
    days = {}
    monkeypatch.setattr(W, "v2_universes", lambda ds, bud=None, now_ms=None, fetch=True: ({pd.Timestamp(d).normalize(): list(members) for d in ds}, True))
    monkeypatch.setattr(W, "_study_1h_cover", lambda: cover_ms)
    monkeypatch.setattr(W, "_study_1h", lambda s: {})
    return days


def test_fetch_v2_reads_late_days_per_member_and_stops_on_errors(monkeypatch):
    mem = [f"M{i}USDT" for i in range(40)]
    _v2_fakes(monkeypatch, mem)
    calls = []

    def fake(sym, iv, start, lim, bud, now, fetch=True):
        calls.append((sym, iv, start, lim))
        return [[start + k * W.HOUR, 1, 1, 1, 1.0 + (k >= 5) * (0.1 if int(sym[1:-4]) < 30 else -0.1), 1] for k in range(lim)]
    monkeypatch.setattr(W, "_raw_klines", fake)
    closes, complete = W.fetch_closes(SINCE, SINCE + 2 * W.DAY, days=[T0])
    assert complete and calls[0] == ("M0USDT", "1h", SINCE - 5 * W.HOUR, W.DAY_PAGE) and len(calls) == 40      # one read per (member, day)
    assert closes.index[0] == pd.Timestamp(SINCE - 4 * W.HOUR, unit="ms") and closes.attrs["universe"] == {T0: mem}   # stamped at the bar CLOSE
    br = W.breadth_series(closes)
    assert abs(br[T0 + pd.Timedelta(hours=1)] - 0.75) < 1e-9 and closes.attrs["valid_until"] is None
    bud = W.LS.Budget()

    def flaky(sym, iv, start, lim, b, now, fetch=True):
        if sym == "M3USDT":
            b.errors += 1
            return []
        return fake(sym, iv, start, lim, b, now)
    monkeypatch.setattr(W, "_raw_klines", flaky)
    closes, complete = W.fetch_closes(SINCE, SINCE + 2 * W.DAY, bud, days=[T0])
    br = W.breadth_series(closes)
    assert complete is False and closes.attrs["valid_until"] == pd.Timestamp(0, unit="ms")
    assert br[br.index > pd.Timestamp(0, unit="ms")].isna().all() and len(br) >= 24               # every hour of the read day unscored, never a stale bar
    monkeypatch.setattr(W, "_raw_klines", lambda *a, **k: None)
    assert W.fetch_closes(SINCE, SINCE + 2 * W.DAY, days=[T0])[1] is False


def test_v2_universes_history_is_the_engine_rule(monkeypatch):
    prev = SINCE - W.DAY
    vol = {f"S{i:03d}USDT": {prev: 1e6 - i} for i in range(120)}
    vol["BTCUSDT"] = {prev: 9e12}
    vol["USDCUSDT"] = {prev: 8e12}
    monkeypatch.setattr(W, "_study_daily", lambda: (vol, {}, {"USDCUSDT": "USDC"}, prev))
    W._MEMO.pop(("uni", SINCE), None)
    U, ok = W.v2_universes([T0], now_ms=NOW)
    W._MEMO.pop(("uni", SINCE), None)
    assert ok and U[T0] == W.MB.select_universe({s: v[prev] for s, v in vol.items()}, None, {"USDCUSDT": "USDC"}) and U[T0][0] == "S000USDT"
    assert len(U[T0]) == 100 and "BTCUSDT" not in U[T0] and "USDCUSDT" not in U[T0]


def test_damaged_record_blocks_only_its_sleeve_and_unscored_is_per_sleeve(tmp_path):
    d = _zr([-0.8] * 12 + [0.3] * 3)
    w = d.copy(); w["entry_strategy"] = "FRENZY_WILLY"
    ht = _fills([(T0 - pd.Timedelta(hours=1), "GAPUSDT", -0.8)]); ht["entry_strategy"] = "FRENZY_WILLY_HT"   # an unscored HT fill must not hold the others
    last = pd.to_datetime(d.opened_at).max()
    st = tmp_path / "w.json"
    st.write_text('{"FRENZY_WILLY": "x"}')
    br = _breadth()
    br = br[br.index >= T0]
    p = _write(tmp_path, pd.concat([d, w, ht]), (last + pd.Timedelta(hours=21)).strftime("%Y-%m-%d_%H-%M-%S"))
    out = "\n".join(W.run(NOW, p, str(st), int((T0 - pd.Timedelta(hours=2)).value // 10**6), lambda *a: (br, True)))
    rec = json.load(open(st))
    assert rec["ML"]["state"] == W.CANDIDATE and rec["FRENZY_WILLY"] == "x" and "FRENZY_WILLY: frozen record damaged" in out
    assert "an unscored fill" not in out


def test_before_after_daily_compound(tmp_path):
    c, _ = W.load(_write(tmp_path, _zr([-1.0] * 4, days=2)))
    txt = W.before_after(W.score(c, _breadth()))
    assert txt.startswith("before 24 ·") and "→ after 20 · 100% · +0.500 %" in txt


def test_floor_operator_closed_and_empty(tmp_path):
    a = _zr([-0.8] * 2)
    a.loc[1, "close_reason"] = "MANUAL"
    old = _fills([(T0 - pd.Timedelta(days=3), "OLDUSDT", -0.8)])
    fn = lambda *x: (_breadth(), True)
    out = "\n".join(W.run(NOW, _write(tmp_path, pd.concat([a, old])), None, SINCE, fn))
    assert out.count("| **zone (market hot, > 50% up)** | 1 |") == 1
    assert "| **zone (market hot, > 50% up)** | 0 |" in "\n".join(W.run(NOW, [], None, SINCE, fn))


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(W.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_ml_mkt_hot_watch as _mhw")
    assert "_mhw.run(now_ms)" in src[j:j + 200] and "except Exception as _mhw_e" in src[j:j + 400]
    import observe_registry as OR
    assert "scout_ml_mkt_hot_watch.py" in OR.BOOKED and OR.BOOKED["scout_ml_mkt_hot_watch.py"] is None
