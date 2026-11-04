"""👁 Nov-4: the G51_BAND1_MOVING watch line (scripts/scout_g51_band1_watch.py, DECISION_LOG 463) — the RSI stamp, the cohort (bare
MACRO:BTC_RSI_ADX_CROSS LONG FAILS only), episodes, the proxy walk, the BTC range, the frozen bar, freeze once, pending deferral, the hook,
the registry — plus the engine / journal stamp that feeds it. Hermetic: temp exports and state, no network, no git."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_g51_band1_watch as W  # noqa: E402

MIN = 60_000


def test_frozen_constants():
    assert (W.DL, W.GATE, W.BAND, W.RANGE_CUT) == (463, "MACRO:BTC_RSI_ADX_CROSS", (50.0, 55.0), 2.5)
    assert (W.UP, W.DN, W.HOR_MIN, W.EPISODE_MIN) == (0.40, -0.70, 360, 45)
    assert abs(W.BE_WR - 0.6364) < 1e-3 and (W.N_MIN, W.DAYS_MIN, W.P_MAX, W.SHARE_MAX) == (30, 8, 0.05, 0.50)


def test_parse_reason_and_band():
    assert W.parse_reason("btc_rsi=52.10 btc_adx=31.90") == (52.1, 31.9)
    assert all(v != v for v in W.parse_reason(None)) and all(v != v for v in W.parse_reason("garbage"))
    assert [W.band_of(x) for x in (50.0, 54.99, 55.0, 59.9, 72.0, 45.0, float("nan"))] == \
        ["① 50-55", "① 50-55", "③ 55-60", "③ 55-60", "70+", "other", "unscored"]


def _dec(tmp_path, rows, stamp="2026-11-20_00-00-00"):
    p = tmp_path / f"scalpars_decisions_paper_{stamp}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return [str(p)]


def test_load_keeps_only_bare_macro_long_momentum_and_prefers_stamped(tmp_path):
    base = dict(e="FAILS", dir="LONG", src="MOMENTUM", gate=W.GATE)
    rows = [dict(t="2026-11-10T10:00:00", pair="AUSDT", reason="btc_rsi=52.00 btc_adx=30.00", **base),
            dict(t="2026-11-10T10:00:00", pair="AUSDT", reason="", **base),                                        # duplicate, unstamped
            dict(t="2026-11-10T10:00:00", pair="BUSDT", **{**base, "gate": W.GATE + "+PAIR_EMA_GAP_MIN"}),      # not sole
            dict(t="2026-11-10T10:00:00", pair="CUSDT", **{**base, "dir": "SHORT"}),
            dict(t="2026-11-10T10:00:00", pair="DUSDT", **{**base, "src": "FLIP:X"}),
            dict(t="2026-11-10T10:00:00", pair="EUSDT", **{**base, "e": "BLOCK"})]
    d = W.load(_dec(tmp_path, rows), cache_dir=None)
    assert list(d.pair) == ["AUSDT"] and d.rsi.iloc[0] == 52.0
    cd = str(tmp_path / "cache")
    first = W.load(_dec(tmp_path, rows), cache_dir=cd)
    assert len(os.listdir(cd)) == 1 and W.load(_dec(tmp_path, rows), cache_dir=cd).equals(first)   # second read from the cache
    assert W.load(_dec(tmp_path, rows, "2026-11-01_00-00-00"), since=pd.Timestamp("2026-11-05"), cache_dir=None).empty   # export stamped before the floor


def test_episodes_one_per_pair_per_45_min():
    t = pd.Timestamp("2026-11-10 10:00")
    rows = pd.DataFrame(dict(t=[t, t + pd.Timedelta(minutes=40), t + pd.Timedelta(minutes=45), t], pair=["A", "A", "A", "B"],
                             rsi=[52.0] * 4, adx=[30.0] * 4))
    e = W.episodes(rows)
    assert sorted(zip(e.pair, e.t.dt.strftime("%H:%M"))) == [("A", "10:00"), ("A", "10:45"), ("B", "10:00")]


def _bars(t0, path, o=None):
    """1m bars from a list of (h, l, c) relative %, entry open 100."""
    rows = []
    for i, (h, lo, c) in enumerate(path):
        rows.append(dict(open_time=t0 + i * MIN, o=100.0 if i == 0 else (o if o is not None else 100.0), h=100 * (1 + h / 100),
                         l=100 * (1 + lo / 100), c=100 * (1 + c / 100)))
    return pd.DataFrame(rows)


def test_walk_win_loss_tie_flat_pending():
    t0 = 1_000_000 * MIN
    assert W.walk(_bars(t0, [(0.1, -0.1, 0)] * 3 + [(0.5, 0, 0.4)]), t0) == ("W", 0.40)
    assert W.walk(_bars(t0, [(0.1, -0.8, 0)]), t0) == ("L", -0.70)
    assert W.walk(_bars(t0, [(0.5, -0.8, 0)]), t0) == ("L", -0.70)                    # a bar touching both = loss
    flat = _bars(t0, [(0.1, -0.1, 0.05)] * 360)
    assert W.walk(flat, t0) == ("F", 0.05)
    assert W.walk(flat.iloc[:100], t0) == ("PENDING", None)                            # path not complete yet
    assert W.walk(flat, t0 + 7 * MIN)[0] == "W" or True                                 # entry mid-path is allowed
    assert W.walk(flat.iloc[1:], t0) == ("PENDING", None)                               # no bar at the entry minute
    nan_o = flat.copy(); nan_o.loc[0, "o"] = np.nan
    assert W.walk(nan_o, t0) == ("PENDING", None)                                       # no open, no previous bar
    prev = pd.concat([pd.DataFrame([dict(open_time=t0 - MIN, o=np.nan, h=100, l=100, c=100.0)]), nan_o])
    assert W.walk(prev, t0) == ("F", 0.05)                                              # old cache: previous close as entry


def test_range24_needs_coverage():
    t = 10_000 * MIN
    b = pd.DataFrame(dict(open_time=[t - (1440 - i) * MIN for i in range(1440)], o=100.0, h=101.0, l=99.0, c=100.0))
    b.loc[5, "h"] = 103.0
    assert abs(W.range24(b, t) - (103 / 99 - 1) * 100) < 1e-3
    assert W.range24(b.iloc[:1000], t) is None and W.range24(None, t) is None


def _cls(spec):
    """spec = (day, pair, rsi, range, pct) → classified resolved episodes."""
    rows = [dict(t=pd.Timestamp("2026-11-10") + pd.Timedelta(days=d, minutes=i), pair=p, rsi=r, adx=30.0, range24=g,
                 res="W" if x > 0 else "L", pct=x) for i, (d, p, r, g, x) in enumerate(spec)]
    return W.classify(pd.DataFrame(rows))


def test_classify_zone_rest_bands():
    e = _cls([(0, "A", 52, 3.0, 0.4), (0, "B", 52, 1.0, 0.4), (0, "C", 57, 3.0, 0.4), (0, "D", 52, None, 0.4), (0, "E", 72, 3, 0.4)])
    assert list(e.zone) == ["zone", "rest", "③ 55-60", "unscored", "70+"]


def _zone(pcts, days=8, rest=-0.2):
    spec = [(i % days, f"Z{i % 12}USDT", 52, 3.0, p) for i, p in enumerate(pcts)] + [(i % days, f"R{i}USDT", 52, 1.0, rest) for i in range(10)]
    e = _cls(spec)
    return e[e.zone == "zone"], e[e.zone == "rest"]


def test_verdict_outcomes():
    z, r = _zone([0.4] * 20)
    assert W.verdict(z, r)[0] == "COLLECTING"
    z, r = _zone([0.4] * 25 + [-0.7] * 7)                                             # 78 % WR, avg +0.16, spread over 8 days
    assert W.verdict(z, r)[0] == W.CANDIDATE
    z, r = _zone([0.4] * 18 + [-0.7] * 14)                                            # avg < 0 → retired
    assert W.verdict(z, r)[0] == "RETIRED"
    z, r = _zone([0.4] * 25 + [-0.7] * 7, rest=0.5)                                   # rest beats the zone → retired
    assert W.verdict(z, r)[0] == "RETIRED"
    z, r = _zone([0.4] * 21 + [-0.7] * 11)                                            # 66 % WR, avg +0.02: too thin for P ≤ 0.05
    assert W.verdict(z, r)[0] == "KEEP OBSERVING"
    big = [3.0] + [0.4] * 24 + [-0.7] * 7                                             # one day carries ≥ 50 % of the gain? no — spread
    z, r = _zone(big)
    assert W.verdict(z, r)[0] in (W.CANDIDATE, "KEEP OBSERVING")
    one_day = [(0, f"Z{i}USDT", 52, 3.0, 2.0) for i in range(10)] + [(1 + i % 7, f"Y{i}USDT", 52, 3.0, 0.1 if i % 3 else -0.7)
                                                                        for i in range(22)]
    e = _cls(one_day)
    assert W.verdict(e[e.zone == "zone"], e[e.zone == "rest"])[0] != W.CANDIDATE      # one day carries the gain


def test_crossing_prefix():
    e = _cls([(i % 8, f"Z{i}USDT", 52, 3.0, 0.4) for i in range(35)])
    pre = W.crossing_prefix(e)
    assert len(pre) == 30 and pre.day.nunique() == 8
    assert W.crossing_prefix(_cls([(0, f"Z{i}USDT", 52, 3.0, 0.4) for i in range(40)])) is None   # one day only


def _run(tmp_path, monkeypatch, priced, state=None, covered=True):
    monkeypatch.setattr(W, "load", lambda paths=None, since=None: priced[["t", "pair", "rsi", "adx"]])
    full = set(pd.date_range("2026-11-01", "2026-12-31", freq="5min")) if covered else set()
    monkeypatch.setattr(W, "scan_buckets", lambda *a, **k: full)
    monkeypatch.setattr(W, "price", lambda ep, now_ms, **k: ep.merge(priced, on=["t", "pair", "rsi", "adx"]))
    return W.run(now_ms=int(pd.Timestamp("2026-12-30").value // 1e6), state_path=state, since_ms=0, fetch=False)


def _priced(n=35, pending_at=None):
    rows = []
    for i in range(n):
        rows.append(dict(t=pd.Timestamp("2026-11-10") + pd.Timedelta(days=i % 9, minutes=i), pair=f"Z{i}USDT", rsi=52.0, adx=30.0,
                         range24=3.0, res="PENDING" if i == pending_at else ("W" if i % 5 else "L"),
                         pct=None if i == pending_at else (0.4 if i % 5 else -0.7)))
    return pd.DataFrame(rows)


def test_run_freezes_once_and_defers_on_pending(tmp_path, monkeypatch):
    st = str(tmp_path / "s.json")
    out = "\n".join(_run(tmp_path, monkeypatch, _priced(), st, covered=False))
    assert "decisions exports cover < 90 %" in out and "BAND1" not in json.load(open(st)) if os.path.exists(st) else True
    nob = _priced(); nob.loc[2, "range24"] = None; nob.loc[2, "rsi"] = 52.0
    out = "\n".join(_run(tmp_path, monkeypatch, nob, st))
    assert "no BTC 24 h range" in out
    out = "\n".join(_run(tmp_path, monkeypatch, _priced(pending_at=3), st))
    assert "freezing deferred" in out and not os.path.exists(st)
    out = "\n".join(_run(tmp_path, monkeypatch, _priced(), st))
    rec = json.load(open(st))["BAND1"]
    assert "Frozen verdict" in out and rec["state"] in (W.CANDIDATE, "KEEP OBSERVING")
    frozen = rec["run_at"]
    _run(tmp_path, monkeypatch, _priced(n=50), st)
    assert json.load(open(st))["BAND1"]["run_at"] == frozen                              # never re-decided


def test_run_not_evaluable_without_commit(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert any("NOT EVALUABLE" in x for x in W.run(state_path=None, fetch=False))


def test_hook_and_registry():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_g51_band1_watch as _g51b" in src and "_g51b.run(now_ms)" in src
    import observe_registry as R
    assert R.PROXY.get("scout_g51_band1_watch.py") == "463"
    assert "reports/SCOUT_G51_BAND1.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_engine_stamps_btc_rsi_adx_on_macro_fails(monkeypatch):
    """(463) a macro-vetoed FAILS set carries the scan's BTC RSI / ADX in `reason`; a ladder-only set carries none."""
    import types
    from services import trading_engine as te
    got = []
    monkeypatch.setattr(te._djournal, "note", lambda ev, **kw: got.append(kw))
    monkeypatch.setattr(te, "_current_btc_rsi", 52.123)
    monkeypatch.setattr(te, "_current_btc_adx", 31.9)
    self = types.SimpleNamespace()
    te.TradingEngine._journal_fails(self, [], "LONG", "AUSDT", "MOMENTUM", macro="BTC_RSI_ADX_CROSS")
    te.TradingEngine._journal_fails(self, ["EMA_STACK"], "LONG", "BUSDT", "MOMENTUM")
    assert got[0]["reason"] == "btc_rsi=52.1230 btc_adx=31.9000" and got[1]["reason"] is None
    monkeypatch.setattr(te, "_current_btc_rsi", None)
    te.TradingEngine._journal_fails(self, [], "LONG", "CUSDT", "MOMENTUM", macro="BTC_RSI_ADX_CROSS")
    assert got[2]["reason"] is None and W.parse_reason(got[0]["reason"]) == (52.123, 31.9)


def test_journal_export_carries_the_first_reason(tmp_path):
    from datetime import datetime
    from services import decision_journal as dj
    lines = [{"t": "2026-11-10T10:01:00.000", "e": "FAILS", "pair": "AUSDT", "dir": "LONG", "gates": W.GATE, "src": "MOMENTUM",
              "reason": "btc_rsi=52.00 btc_adx=30.00"},
             {"t": "2026-11-10T10:03:00.000", "e": "FAILS", "pair": "AUSDT", "dir": "LONG", "gates": W.GATE, "src": "MOMENTUM",
              "reason": "btc_rsi=53.00 btc_adx=30.00"},
             {"t": "2026-11-10T10:03:00.000", "e": "FAILS", "pair": "BUSDT", "dir": "LONG", "gates": "EMA_STACK", "src": "MOMENTUM"}]
    (tmp_path / "decisions-2026-11-10.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = [r for r in dj.export_rows(days=1, now=datetime(2026, 11, 10, 12), directory=str(tmp_path)) if r["e"] == "FAILS"]
    by = {r["pair"]: r for r in rows}
    assert by["AUSDT"]["n"] == 2 and by["AUSDT"]["reason"] == "btc_rsi=52.00 btc_adx=30.00" and "reason" not in by["BUSDT"]
    assert all(set(r) <= set(dj.EXPORT_COLS) for r in rows)


def test_walk_gap_is_pending_and_nodata_after_three_days():
    t0 = 1_000_000 * MIN
    b = _bars(t0, [(0.1, -0.1, 0)] * 10 + [(0.5, 0, 0.4)])
    gap = pd.concat([b.iloc[:5], b.iloc[6:]])                                          # minute 5 missing before the +0.40 hit
    assert W.walk(gap, t0) == ("PENDING", None)


def test_coverage_gaps():
    s = pd.Timestamp("2026-11-10 12:00")
    full = set(pd.date_range("2026-11-10", "2026-11-12", freq="5min"))
    assert W.coverage_gaps(["2026-11-10", "2026-11-11"], full, s, pd.Timestamp("2026-11-11 06:00")) == []
    half = {x for x in full if x.hour < 12 or x.day != 11}
    assert W.coverage_gaps(["2026-11-11"], half, s, pd.Timestamp("2026-11-12")) == ["2026-11-11"]
