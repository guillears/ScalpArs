"""🚫🔭 DECISION_LOG 630 — the EXH_NO_SPOT forward watch (scripts/scout_exh_nospot_630.py): observe-only pricing of the no-spot EXHAUSTION
setups that pass every other rule. No network: every Binance read is a fake (`net`), features() is stubbed where the engine read is not
the point. Covers the walk, funding, the "passes every other rule" check, the stage cache (features → entry → price), the 2-day entry
window, the walk starting after the fill, a delisted coin (never a permanent pause), an unreadable cache (never overwritten), the prefix
pause + entry capture, one shadow per coin (incl. unpriced would-trade setups), the frozen read-once gate, the journal loader (WIDE +
spike), the registry, the scout hook and the gitignore."""
import io
import json
import os
import sys
import urllib.error
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_exh_nospot_630 as S  # noqa: E402

MIN = 60_000
DAY = 86_400_000
HOLD = 3
TH = NS(exhaustion_min_vol_usd=20e6, exhaustion_wide_min_vol_usd=5e6, exhaustion_vol_decay_min=0.6366, exhaustion_tp_atr_mult=20.0,
        exhaustion_tp_cap_pct=7.0, exhaustion_sl_atr_mult=1.5, exhaustion_sl_cap_pct=6.0, exhaustion_max_hold_hours=24.0)
T0 = 1_795_000_500_000 // (15 * MIN) * (15 * MIN)
T1 = T0 + MIN                                   # the first whole minute after the fill at T0 + 8 s


def _feat(**kw):
    return dict(dict(ok=True, cross=True, vol24=50e6, vol_decay=0.9, atr_pct=2.0), **kw)


# ── ① pure pieces ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_passes_else_mirrors_decide_order_without_spot():
    assert S.passes_else(_feat(), "FRENZY", TH) == (True, None)
    assert S.passes_else(_feat(ok=False), "FRENZY", TH) == (False, "OUT_NO_DATA")
    assert S.passes_else(None, "FRENZY", TH) == (False, "OUT_NO_DATA")
    assert S.passes_else(_feat(cross=False), "FRENZY", TH) == (False, "OUT_NO_SIGNAL")
    assert S.passes_else(_feat(vol24=19.9e6), "FRENZY", TH) == (False, "OUT_VOL24")
    assert S.passes_else(_feat(vol24=19.9e6), "WIDE", TH) == (True, None)                 # WIDE floor $5M
    assert S.passes_else(_feat(vol24=4.9e6), "WIDE", TH) == (False, "OUT_VOL24")
    assert S.passes_else(_feat(vol24=19.9e6), float("nan"), TH) == (False, "OUT_VOL24")    # NaN universe → FRENZY floor
    assert S.passes_else(_feat(vol24=None), None, TH) == (False, "OUT_VOL24")
    assert S.passes_else(_feat(vol_decay=0.6), "FRENZY", TH) == (False, "OUT_VOL_DECAY")
    assert S.passes_else(_feat(vol_decay=None), "FRENZY", TH) == (False, "OUT_VOL_DECAY")
    assert S.passes_else(_feat(vol_decay=0.6366), "FRENZY", TH) == (True, None)           # ≥ the cut passes, as decide()
    assert S.hold_minutes(TH) == 1440 and S._uni(float("nan")) == "FRENZY" and S._uni("wide") == "WIDE"


def test_walk_stop_first_tp_through_and_time_exit():
    e = 100.0
    tk, mk, sl = S.XO.TAKER, S.XO.MAKER, S.SLIP
    pct, why, i = S.walk(e, [(100, 101, 99, 100), (100, 106.5, 92, 100)], 7.0, 6.0, hold=5)   # both in one bar → the stop
    assert why == "SL" and i == 1 and pct == pytest.approx(-6.0 - 2 * tk - sl)
    pct, why, _ = S.walk(e, [(108, 109, 107, 108)], 7.0, 6.0, hold=5)                          # gap over the stop → the open
    assert why == "SL" and pct == pytest.approx(-8.0 - 2 * tk - sl)
    tp_px = e * (1 - 7.0 / 100.0)
    assert S.walk(e, [(100, 100, tp_px, 94)], 7.0, 6.0, hold=5)[1] == "OPEN"                   # touching the TP exactly is not a fill
    pct, why, i = S.walk(e, [(100, 100, tp_px * 0.9999, 94)], 7.0, 6.0, hold=1)                # through → maker TP
    assert why == "TP" and i == 0 and pct == pytest.approx(7.0 - tk - mk)
    pct, why, i = S.walk(e, [(100, 101, 99, 98.0)] * 3, 7.0, 6.0, hold=3)                      # time exit at bar hold−1's close
    assert why == "TIME" and i == 2 and pct == pytest.approx(2.0 - 2 * tk - sl)
    assert S.walk(None, [(1, 1, 1, 1)], 7, 6, 3)[1] == "NO_DATA" and S.walk(1.0, [], 7, 6, 3)[1] == "NO_DATA"


def test_funding_sign_window_and_exit_boundary():
    rates = [(1000, 0.0001), (2000, -0.002), (3000, 0.0005), (4000, float("nan")), (5000, None)]
    assert S.funding_pct(rates, 1000, 3000) == pytest.approx((-0.002 + 0.0005) * 100)          # (entry, until]; a short pays a negative rate
    assert S.funding_pct(rates, 0, 99_999) == pytest.approx((0.0001 - 0.002 + 0.0005) * 100)   # bad rates skipped
    assert S.funding_pct([], 0, 1) == 0
    assert S.funding_until(T1, 4, "SL") == T1 + 4 * MIN                                        # stop / TP: the exit minute's START
    assert S.funding_until(T1, 4, "TP") == T1 + 4 * MIN
    assert S.funding_until(T1, 4, "TIME") == T1 + 5 * MIN + 1_000                              # time exit: its END + 1 s (fundingTime ms jitter)


def test_gate_prefix_days_and_frozen_direction():
    neg = [(f"2026-12-{d:02d}", -1.0 + (k % 3) * 0.1) for d in range(1, 11) for k in range(3)]
    v, txt, n = S.gate(neg)
    assert v == "KEEP" and n == 30 and "block stands" in txt
    pos = [(f"2026-12-{d:02d}", 2.0 + (k % 3) * 0.5) for d in range(1, 11) for k in range(3)]
    assert S.gate(pos)[0] == "REOPEN"
    assert S.gate([("2026-12-01", 1.0)] * 40)[0] == "COLLECTING"                              # 40 setups on ONE day → still collecting
    assert S.gate([])[0] == "COLLECTING"
    assert S.gate(neg + pos)[2] == 30                                                         # the FIRST qualifying prefix decides
    xs = [(f"d{i}", float(i)) for i in range(10)]
    assert S.lower_bound(xs) <= sum(x for _, x in xs) / len(xs)


# ── ② the stage cache + the 2-day entry window + the walk start ─────────────────────────────────────────────────────────────────────

class FakeNet(dict):
    def __init__(self, entry=1.0, bars=None, funding=(), rows=True, gone=()):
        self.calls = []

        def f5(p, T):
            self.calls.append(("5m", p))
            if p in gone:
                raise urllib.error.HTTPError("u", 400, "Invalid symbol", {}, io.BytesIO(b""))
            return [[0] * 8] if rows else []
        super().__init__(
            fetch_5m=f5,
            fetch_entry=lambda p, t: (self.calls.append(("agg", p, t)) or entry),
            fetch_1m=lambda p, t, n: (self.calls.append(("1m", p, t, n)) or [(t + i * MIN, *b) for i, b in enumerate(bars or [])]),
            fetch_funding=lambda p, a, b: (self.calls.append(("fund", p)) or list(funding)))


def _row(pair="NSUSDT", t=T0, uni="FRENZY"):
    return dict(pair=pair, _t=t, _spike=t - 3_600_000, exh_universe=uni)


@pytest.fixture
def feat(monkeypatch):
    box = {"f": _feat()}
    monkeypatch.setattr(S.X, "features", lambda rows, spot, T, on, has_spot, th: dict(box["f"]))
    monkeypatch.setattr(S, "hold_minutes", lambda th: HOLD)
    return box


def test_out_setups_are_listed_and_cached(feat):
    feat["f"] = _feat(vol_decay=0.2)
    net, cache = FakeNet(), {}
    c = S.price_setup(_row(), TH, cache, T0 + 10 * MIN, net)
    assert c["status"] == "OUT_VOL_DECAY" and net.calls == [("5m", "NSUSDT")]
    S.price_setup(_row(), TH, cache, T0 + 20 * MIN, net)                                       # cached: no second read
    assert len(net.calls) == 1
    assert S.price_setup(_row(), TH, {}, T0, FakeNet(rows=False))["status"] == "OUT_NO_DATA"


def test_entry_captured_early_priced_after_the_path_from_the_minute_after_the_fill(feat):
    net, cache = FakeNet(entry=100.0, bars=[(100, 100.5, 99, 99.5)] * HOLD, funding=[(T1 + 2 * MIN, 0.0003), (T1 + 3 * MIN + 500, 0.0001)]), {}
    assert S.price_setup(_row(), TH, cache, T0 + 30_000, net) is None                          # path open → wait; entry already cached
    k = f"NSUSDT|{T0}"
    assert cache[k]["entry"] == pytest.approx(100.0 * (1 - S.SLIP / 100)) and ("agg", "NSUSDT", T0 + S.LAT_MS) in net.calls
    c = S.price_setup(_row(), TH, cache, T0 + 10 * MIN, net)
    assert ("1m", "NSUSDT", T1, HOLD) in net.calls                                             # the walk starts at the minute AFTER the fill
    e = 100.0 * (1 - S.SLIP / 100)
    assert c["status"] == "PRICED" and c["exit"] == "TIME" and c["exit_ms"] == T1 + HOLD * MIN
    assert c["pct_ex_funding"] == pytest.approx((1 - 99.5 / e) * 100 - 2 * S.XO.TAKER - S.SLIP)
    assert c["funding"] == pytest.approx(0.04) and c["pct"] == pytest.approx(c["pct_ex_funding"] + 0.04)   # both (≤ end + 1 s)
    assert sum(1 for x in net.calls if x[0] == "agg") == 1                                     # the entry is fetched once


def test_first_seen_after_two_days_is_no_entry_never_a_proxy(feat):
    net = FakeNet()
    c = S.price_setup(_row(), TH, {}, T0 + 2 * DAY, net)
    assert c["status"] == "NO_ENTRY" and not any(x[0] == "agg" for x in net.calls)
    assert S.price_setup(_row(), TH, {}, T0 + MIN, FakeNet(entry=None))["status"] == "NO_ENTRY"   # no print after T + 8 s


def test_gapped_path_retried_then_no_data(feat):
    net, cache = FakeNet(entry=100.0, bars=[(100, 100.5, 99, 99.5)]), {}                       # 1 bar of a 3-bar hold
    end = T1 + HOLD * MIN
    assert S.price_setup(_row(), TH, cache, end + 4 * 3_600_000, net) is None                  # inside the 12 h grace → retried
    assert S.price_setup(_row(), TH, cache, end + S.GAP_GRACE_MS + MIN, net)["status"] == "NO_DATA"


# ── ③ the report: prefix, shadows, delisted coin, unreadable cache, capture after a pause, frozen gate ────────────────────────────

def _journal(rows):
    return pd.DataFrame([dict(e="EXHAUSTION", pair=p, exh_signal_at=pd.Timestamp(t, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"),
                              exh_spike_at=pd.Timestamp(t - 3_600_000, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"),
                              exh_block=b, exh_universe="FRENZY") for p, t, b in rows])


def _rep(j, net, tmp_path, now, **kw):
    return S.report(j, None, now_ms=now, state_path=str(tmp_path / "st.json"), floor_ms=T0, th=TH, net=net,
                    cache_path=str(tmp_path / "c.json"), **kw)


def test_report_prefix_shadow_and_capture(feat, tmp_path):
    j = _journal([("AUSDT", T0, "EXH_NO_SPOT"), ("AUSDT", T0 + MIN, "EXH_NO_SPOT"),          # 2nd inside A's shadow
                  ("BUSDT", T0 + 15 * MIN, "EXH_VOL24_LOW"),                                   # not a no-spot row
                  ("CUSDT", T0 + 30 * MIN, "EXH_NO_SPOT"), ("DUSDT", T0 + 31 * MIN, "EXH_NO_SPOT")])
    out = _rep(j, FakeNet(entry=100.0, bars=[(100, 100.2, 99.9, 100)] * HOLD), tmp_path, T0 + 32 * MIN)
    cache = json.load(open(tmp_path / "c.json"))
    assert "SHADOWED" in out and "AUSDT 2026" in out.split("SHADOWED")[1]
    assert cache[f"AUSDT|{T0}"]["status"] == "PRICED"
    assert "path not complete" in out                                                         # C's path open → the prefix pauses
    d = cache[f"DUSDT|{T0 + 31 * MIN}"]
    assert "entry" in d and "pct" not in d                                                    # D's entry captured despite the pause
    assert "BUSDT" not in json.dumps(cache)


def test_delisted_coin_is_no_data_never_a_permanent_pause(feat, tmp_path):
    j = _journal([("DEADUSDT", T0, "EXH_NO_SPOT"), ("LIVEUSDT", T0 + MIN, "EXH_NO_SPOT")])
    net = FakeNet(entry=100.0, bars=[(100, 100.2, 99.9, 100)] * HOLD, gone=("DEADUSDT",))
    out = _rep(j, net, tmp_path, T0 + 30 * MIN)
    cache = json.load(open(tmp_path / "c.json"))
    assert cache[f"DEADUSDT|{T0}"]["status"] == "NO_DATA" and cache[f"LIVEUSDT|{T0 + MIN}"]["status"] == "PRICED"
    assert "paused" not in out
    out2 = _rep(j, net, tmp_path, T0 + 40 * MIN)                                              # the next run does not re-read it
    assert sum(1 for x in net.calls if x[:2] == ("5m", "DEADUSDT")) == 1 and "paused" not in out2


def test_transient_error_pauses_without_caching(feat, tmp_path):
    j = _journal([("AUSDT", T0, "EXH_NO_SPOT")])
    net = FakeNet()
    net["fetch_5m"] = lambda p, T: (_ for _ in ()).throw(urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"")))
    out = _rep(j, net, tmp_path, T0 + DAY)
    assert "paused at AUSDT" in out and "HTTP 503" in out
    assert json.load(open(tmp_path / "c.json")) == {f"AUSDT|{T0}": {}}                        # nothing final cached → retried next run


def test_unreadable_cache_is_never_overwritten(feat, tmp_path):
    (tmp_path / "c.json").write_text("{not json")
    out = _rep(_journal([("AUSDT", T0, "EXH_NO_SPOT")]), FakeNet(), tmp_path, T0 + DAY)
    assert "NOT EVALUABLE" in out and "unreadable" in out
    assert (tmp_path / "c.json").read_text() == "{not json"


def test_cache_saved_even_when_the_walk_crashes(feat, tmp_path, monkeypatch):
    j = _journal([("AUSDT", T0, "EXH_NO_SPOT")])

    def boom(*a, **k):
        raise KeyboardInterrupt                                                              # not an Exception: escapes everything
    real = S.price_setup

    def first_then_boom(r, th, cache, now, net=None, capture_only=False):
        real(r, th, cache, now, net, capture_only=True)                                       # captures features + entry
        boom()
    monkeypatch.setattr(S, "price_setup", first_then_boom)
    with pytest.raises(KeyboardInterrupt):
        _rep(j, FakeNet(entry=100.0), tmp_path, T0 + MIN)
    assert "entry" in json.load(open(tmp_path / "c.json"))[f"AUSDT|{T0}"]                    # finally: saved


def test_unpriced_would_trade_setup_holds_its_coin_but_an_out_one_does_not(feat, tmp_path):
    j = _journal([("AUSDT", T0, "EXH_NO_SPOT"), ("AUSDT", T0 + 2 * MIN, "EXH_NO_SPOT")])     # 2nd inside A's hold (3 min + 1)
    out = _rep(j, FakeNet(entry=None), tmp_path, T0 + 20 * MIN)                               # A: no print → NO_ENTRY, but it WOULD trade
    assert "NO_ENTRY (1" in out and "SHADOWED" in out                                         # the 2nd A sits inside its unknown life
    feat["f"] = _feat(vol_decay=0.1)                                                          # an OUT setup never holds the coin
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    out = _rep(j, FakeNet(entry=None), tmp2, T0 + 20 * MIN)
    assert "OUT_VOL_DECAY (2" in out and "SHADOWED" not in out


def test_gate_freezes_on_the_first_qualifying_run_and_never_moves(feat, tmp_path):
    rows = [(f"P{k}USDT", T0 + d * DAY + k * MIN, "EXH_NO_SPOT") for d in range(10) for k in range(3)]
    j = _journal(rows)
    for d in range(10):                                                                       # one run per day, as the 4-h hook would
        out = _rep(j, FakeNet(entry=100.0, bars=[(100, 100.0, 99.0, 99.0)] * HOLD), tmp_path, T0 + d * DAY + 30 * MIN)
        if d < 9:
            assert not (tmp_path / "st.json").exists()                                        # < 10 days → nothing frozen
    st = json.load(open(tmp_path / "st.json"))
    assert st["n"] == 30 and st["verdict"] in ("KEEP", "REOPEN") and "FROZEN" in out
    (tmp_path / "st.json").write_text(json.dumps(dict(st, verdict="KEEP", detail="pinned")))
    out2 = _rep(j, FakeNet(entry=100.0, bars=[(100, 100.0, 99.0, 99.0)] * HOLD), tmp_path, T0 + 11 * DAY)
    assert "KEEP — pinned" in out2


def test_not_evaluable_before_the_commit():
    assert "NOT EVALUABLE" in S.report(_journal([]), None, floor_ms=None, th=TH, net=FakeNet())


def test_netstop_pauses_cleanly(feat, tmp_path):
    net = FakeNet()
    net["fetch_5m"] = lambda p, T: (_ for _ in ()).throw(S.XO.NetStop("HTTP 429"))
    out = _rep(_journal([("AUSDT", T0, "EXH_NO_SPOT")]), net, tmp_path, T0 + DAY)
    assert "paused at AUSDT" in out and "HTTP 429" in out


# ── ④ the engine's features + the journal loader ─────────────────────────────────────────────────────────────────────────────────

def test_real_engine_features_no_cross_is_out_no_signal():
    """the scout calls the ENGINE's features(): a choppy window (real ATR, RSI ≈ 50, no cross below 70) → OUT_NO_SIGNAL."""
    T = T0
    rows = []
    for i in range(1500):
        c = 100.0 + (0.5 if i % 2 else -0.5)
        rows.append([T - (1500 - i) * 300_000, 100.0, c + 0.3, c - 0.3, c, 1, 0, 30e6 / 288])
    f = S.X.features(rows, [], T, T - 3_600_000 - 300_000, False, TH)
    assert f["ok"] and f["atr_pct"] > 0 and not f["cross"]
    assert S.passes_else(f, "FRENZY", TH) == (False, "OUT_NO_SIGNAL")


def test_journal_loader_keeps_wide_and_the_spike_time(tmp_path):
    p = tmp_path / "scalpars_decisions_paper_x.csv"
    pd.DataFrame([dict(e="EXHAUSTION", pair="AUSDT", exh_signal_at="2026-12-01T10:00:00", exh_spike_at="2026-12-01T08:50:00",
                       exh_status="BLOCKED", exh_block="EXH_NO_SPOT", exh_universe="WIDE"),
                  dict(e="EXHAUSTION", pair="BUSDT", exh_signal_at="2026-12-01T10:15:00", exh_spike_at="2026-12-01T09:00:00",
                       exh_status="BLOCKED", exh_block="EXH_NO_SPOT", exh_universe="FRENZY"),
                  dict(e="DISTRIBUTION", pair="CUSDT", exh_signal_at=None, exh_spike_at=None, exh_status=None, exh_block=None,
                       exh_universe=None)]).to_csv(p, index=False)
    j = S.load_journal([str(p)])
    assert sorted(j.pair) == ["AUSDT", "BUSDT"] and "exh_spike_at" in j
    coh = S.cohort(j, 0)
    a = next(r for r in coh if r["pair"] == "AUSDT")
    assert a["_spike"] == S._ms("2026-12-01T08:50:00") and S._uni(a["exh_universe"]) == "WIDE"


# ── ⑤ registry · scout hook · gitignore ────────────────────────────────────────────────────────────────────────────────────────────

def test_registry_hook_and_gitignore():
    import observe_registry as R
    assert "scout_exh_nospot_630.py" in R.EXIT
    w = R.EXIT["scout_exh_nospot_630.py"]["walks"]
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert all(cfg.get(k) == v for k, v in w.items())                                         # pinned to the LIVE exit today (else ⚠ STALE)
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "import scout_exh_nospot_630" in src and "_ns630.run(now_ms)" in src
    gi = open(os.path.join(ROOT, ".gitignore")).read()
    assert "reports/.exh_nospot_630_cache.json" in gi and "reports/SCOUT_EXH_NOSPOT_630.json" in gi
