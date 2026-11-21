"""📉📊 DECISION_LOG 596 — three OBSERVE-only lines + their stamps:
EXH_AFTER_DROP (entry_exh_ret_1h_pct, EXHAUSTION journal ret_1h_pct) · WILLY_HT PRE_VOL_SURGE (entry_frenzy_pre_vol15_vs60, post-fill
backfill) · WILLY_HT BTC5M_UP (entry_btc_trend_gap_pct, already stamped). Pure maths, wiring by source, the scouts' frozen bars."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

from services import exhaustion as X              # noqa: E402
from services.frenzy import pre_vol15_vs60        # noqa: E402
from tests.test_exhaustion_short import _row, _th, BAR   # noqa: E402
import scout_exh_after_drop as A                  # noqa: E402
import scout_willy_ht_lines as W                  # noqa: E402
import observe_registry as OR                     # noqa: E402


def _src(p):
    with open(os.path.join(ROOT, p)) as f:
        return f.read()


# ── EXH ret_1h ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", range(5))
def test_exh_ret_1h_is_the_signal_close_vs_the_5m_close_one_hour_earlier(k):
    pair, T, on, sp, rows, spot, m, o_next = _row(k)
    f = X.features(rows, spot, T, on, True, _th())
    assert f['ok']
    by_open = {int(r[0]): float(r[4]) for r in rows}
    c_now, c_1h = by_open[T - BAR], by_open[T - BAR - 12 * BAR]       # the bar ending at T and the one ending at T − 60 min
    assert f['close'] == c_now
    assert f['ret_1h'] == pytest.approx((c_now / c_1h - 1) * 100, rel=1e-12)


def test_exh_ret_1h_none_on_a_failed_read():
    f = X.features([], [], 0, 0, True, _th())
    assert f['ok'] is False and f['ret_1h'] is None


def test_exh_ret_1h_wired_to_order_journal_and_export():
    eng = _src("services/trading_engine.py")
    assert "entry_exh_ret_1h_pct=(round(f['ret_1h'], 4) if f.get('ret_1h') is not None else None))" in eng      # the open call
    assert "entry_exh_ret_1h_pct: Optional[float] = None" in eng and "entry_exh_ret_1h_pct=entry_exh_ret_1h_pct)" in eng   # kw → Order
    assert "ret_1h_pct=(round(f['ret_1h'], 4) if f.get('ret_1h') is not None else None)" in eng                  # the journal row
    from models import Order, ExhaustionSignal
    assert "entry_exh_ret_1h_pct" in Order.__table__.columns and "ret_1h_pct" in ExhaustionSignal.__table__.columns
    db = _src("database.py")
    assert "'entry_exh_ret_1h_pct'" in db and "'shadow_notional_usd', 'ret_1h_pct'" in db
    mn = _src("main.py")
    assert '"exh_ret_1h_pct"]' in mn and "exh_ret_1h_pct=getattr(b, 'ret_1h_pct', None)" in mn
    from services.trading_engine import TradingEngine
    assert TradingEngine.stamp_class("entry_exh_ret_1h_pct") == "sleeve"
    assert TradingEngine.stamp_class("entry_frenzy_pre_vol15_vs60") == "sleeve"


# ── pre_vol15_vs60 ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _k1(vols, t0=1_700_000_000_000 // 60_000 * 60_000):
    return [[t0 + i * 60_000, 1, 1, 1, 1, float(v)] for i, v in enumerate(vols)]


def test_pre_vol15_vs60_matches_the_study_definition():
    vols = [10.0] * 60 + [30.0] * 15 + [999.0]            # 75 closed bars, then the fill's own minute (excluded)
    rows = _k1(vols)
    fill_ms = rows[75][0] + 20_000                         # 20 s into minute 75
    assert pre_vol15_vs60(rows, fill_ms) == pytest.approx(3.0)
    assert pre_vol15_vs60(list(reversed(rows)), fill_ms) == pytest.approx(3.0)   # order-independent
    extra = _k1([1.0] * 5 + vols, t0=rows[0][0] - 5 * 60_000)   # older bars beyond the 75 are ignored
    assert pre_vol15_vs60(extra, fill_ms) == pytest.approx(3.0)


def test_pre_vol15_vs60_fails_soft():
    rows = _k1([10.0] * 74 + [5.0])
    assert pre_vol15_vs60(rows, rows[-1][0] + 60_000) is not None
    assert pre_vol15_vs60(rows, rows[-1][0]) is None                   # only 74 closed bars before the fill minute
    assert pre_vol15_vs60(_k1([0.0] * 60 + [5.0] * 15), 10 ** 13) is None   # zero base volume
    assert pre_vol15_vs60(None, 10 ** 13) is None and pre_vol15_vs60([["x"]], 10 ** 13) is None
    assert pre_vol15_vs60(_k1([float("nan")] * 75), 10 ** 13) is None


def test_pre_vol_backfill_wired_for_frenzy_only():
    eng = _src("services/trading_engine.py")
    i = eng.index("async def _stamp_backfill(")
    body = eng[i:eng.index("def stamp_missing", i)]
    assert '_want1m = es.startswith("FRENZY_") and opened_at is not None' in body
    assert "binance_service.get_ohlcv(symbol, '1m', 90), 8.0) if _want1m else asyncio.sleep(0)" in body   # inside the bounded gather
    assert "st['entry_frenzy_pre_vol15_vs60'] = _pv" in body
    from models import Order
    assert "entry_frenzy_pre_vol15_vs60" in Order.__table__.columns
    assert "'entry_frenzy_pre_vol15_vs60'" in _src("database.py")


# ── EXH_AFTER_DROP scout ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_after_drop_zone_band_inclusive():
    assert A.zone_of(-5.0) == "zone" and A.zone_of(-10.0) == "zone" and A.zone_of(-7.3) == "zone"
    assert A.zone_of(-4.99) == "rest" and A.zone_of(-10.01) == "rest" and A.zone_of(3.0) == "rest"
    assert A.zone_of(None) == "unscored" and A.zone_of("x") == "unscored" and A.zone_of(float("nan")) == "unscored"


def _exh_csv(tmp_path, rows):
    base = dict(direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT", close_reason="FRENZY_TP", notional_value=1000.0,
                entry_exh_model_slip_pct=0.0, entry_exh_q5_usd=1e6, entry_exh_tp_pct=9.0, entry_exh_sl_pct=5.0, entry_exh_universe="FRENZY",
                closed_at="2026-12-01T00:00:00", pnl=1.0)
    df = pd.DataFrame([dict(base, **r) for r in rows])
    p = tmp_path / "scalpars_orders_paper_x.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def test_after_drop_counts_forward_stamped_fills_and_freezes_once(tmp_path):
    rows = []
    for d in range(9):                                          # 18 zone losers on 9 days + 9 rest winners
        for h in (1, 2):
            rows.append(dict(opened_at=f"2026-12-{d + 1:02d}T0{h}:00:00", pair=f"Z{d}{h}USDT", pnl_percentage=-2.0, entry_exh_ret_1h_pct=-7.0))
        rows.append(dict(opened_at=f"2026-12-{d + 1:02d}T05:00:00", pair=f"R{d}USDT", pnl_percentage=4.0, entry_exh_ret_1h_pct=1.0))
    rows.append(dict(opened_at="2026-12-20T05:00:00", pair="LATEUSDT", pnl_percentage=1.0, entry_exh_ret_1h_pct=None))   # newest export ≥ +24 h
    rows.append(dict(opened_at="2026-11-01T05:00:00", pair="OLDUSDT", pnl_percentage=-9.0, entry_exh_ret_1h_pct=-6.0))   # before the floor
    paths = _exh_csv(tmp_path, rows)
    st = tmp_path / "state.json"
    since = int(pd.Timestamp("2026-11-30").value // 1_000_000)
    out = "\n".join(A.run(paths=paths, state_path=str(st), since_ms=since, now_ms=since + 40 * 86_400_000))
    assert "| zone (would block: −10 … −5 % hour) | 18 | 9 | 0% | -2.000 % |" in out
    assert "| UNSCORED (no stamp) — never counted | 1 |" in out and "OLDUSDT" not in out
    assert "Frozen verdict" in out and A.CANDIDATE in out
    out2 = "\n".join(A.run(paths=paths, state_path=str(st), since_ms=since, now_ms=since + 41 * 86_400_000))
    assert out2.count("Frozen verdict") == 1


def test_after_drop_verdict_retires_a_winning_zone():
    z = pd.DataFrame(dict(pct=[1.0] * 15, day=[f"d{i % 8}" for i in range(15)], pair=["P"] * 15))
    r = pd.DataFrame(dict(pct=[0.5] * 5, day=["d0"] * 5, pair=["Q"] * 5))
    assert A.verdict(z, r)[0] == "RETIRED"
    assert A.verdict(z.iloc[:5], r)[0] == "COLLECTING"


def test_after_drop_not_evaluable_without_the_commit(monkeypatch):
    monkeypatch.setattr(A._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert any("NOT EVALUABLE" in x for x in A.run(state_path=None))


# ── WILLY_HT PRE_VOL_SURGE · BTC5M_UP ──────────────────────────────────────────────────────────────────────────────────────────────

def test_ht_596_zones_and_frozen_cuts():
    assert W.PRE_VOL_CUT == 1.60 and W.BTC_GAP_CUT == 0.0 and set(W.LINES_596) <= set(W.LINES)
    assert W.zone_above(1.61, 1.6) == "zone" and W.zone_above(1.6, 1.6) == "rest" and W.zone_above(None, 1.6) == "unscored"
    assert W.zone_above(0.001, 0.0) == "zone" and W.zone_above(0.0, 0.0) == "rest" and W.zone_above(-0.2, 0.0) == "rest"
    assert {"entry_frenzy_pre_vol15_vs60", "entry_btc_trend_gap_pct"} <= set(W.EXTRA)


def test_ht_596_lines_ht_only_and_forward(monkeypatch):
    c = pd.DataFrame(dict(pair=["AUSDT", "BUSDT"], o=pd.to_datetime(["2026-12-01", "2026-12-02"]),
                          entry_frenzy_pre_vol15_vs60=[2.0, 1.0], entry_btc_trend_gap_pct=[0.1, -0.1]))
    sc, _ = W.score(c, 0, fetch=False)
    assert list(sc.z_PRE_VOL_SURGE) == ["zone", "rest"] and list(sc.z_BTC5M_UP) == ["zone", "rest"]
    src = _src("scripts/scout_willy_ht_lines.py")
    assert 'if es != "FRENZY_WILLY_HT":' in src and "s_ln = s[s.o >= pd.Timestamp(int(floor_596), unit=\"ms\")]" in src
    assert "s = s_ln" not in src                                   # the per-line cohort never leaks into the next line


def test_registry_and_scout_hooks():
    assert "scout_exh_after_drop.py" in OR.classified()
    assert OR.BOOKED["scout_exh_after_drop.py"]["exhaustion_tp_cap_pct"] == 7.0   # 🔁 (609)
    op = _src("scripts/opportunity_scout.py")
    assert "import scout_exh_after_drop as _ead" in op and "_rg_sec += _ead.run(now_ms)" in op
    assert "reports/SCOUT_EXH_AFTER_DROP_609.json" in _src(".gitignore")   # 🔁 (609) re-based state file


def test_btc5m_up_counts_window_units():
    assert W.LINE_DAYS_MIN["BTC5M_UP"] == 15 and "PRE_VOL_SURGE" not in W.LINE_DAYS_MIN
    z = pd.DataFrame(dict(pct=[-1.0] * 20, day=[f"d{i % 10}" for i in range(20)], pair=[f"P{i}" for i in range(20)], o=pd.date_range("2026-12-01", periods=20, freq="h")))
    assert W.verdict(z, z.iloc[:0], 0.444, 15)[0] == "COLLECTING"            # 20 fills on 10 days: enough for a fill line, not a market line
    assert W.verdict(z, z.iloc[:0], 0.444)[0] != "COLLECTING"
    sc = z.assign(z_X=["zone"] * 20)
    assert W.crossing_prefix(sc, "z_X", 15) is None and W.crossing_prefix(sc, "z_X") is not None


# ── WILLY_HT BURST_REJECTION (DECISION_LOG 618) ──────────────────────────────────────────────────────────────────────────────────

def _k(t0, rows):
    """1m klines from t0: rows = [(high, low, close)] → [open_ms, o, h, l, c, v]."""
    return [[t0 + i * 60_000, c, h, l, c, 1.0] for i, (h, l, c) in enumerate(rows)]


def test_ht_618_cloc15_pure():
    t0 = 1_800_000_000_000 // 60_000 * 60_000
    fill = t0 + 15 * 60_000 + 7_000                                          # 7 s into the 16th minute → the 15 bars before it count
    bars = _k(t0, [(10.0, 9.0, 9.5)] * 14 + [(10.0, 9.0, 9.2)]) + _k(t0 + 15 * 60_000, [(99.0, 1.0, 50.0)])   # the fill-minute bar is ignored
    assert W.cloc15(bars, fill) == pytest.approx(0.2)
    assert W.cloc15(bars[:14], fill) is None                                 # fewer than 15 closed bars → unscored
    assert W.cloc15(_k(t0, [(5.0, 5.0, 5.0)] * 15), fill) is None             # flat range
    assert W.cloc15(None, fill) is None and W.cloc15([["x"]], fill) is None
    assert W.CLOC_CUT == 0.50 and W.CLOC_BARS == 15 and "BURST_REJECTION" in W.LINES and W.LINES_618 == ("BURST_REJECTION",)


def test_ht_618_zone_rules():
    z = W.zone_burst_rejection
    assert z(2.0, 0.49) == "zone" and z(2.0, 0.50) == "rest" and z(2.0, None) == "unscored"
    assert z(1.60, None) == "rest" and z(1.0, 0.1) == "rest"                  # no burst → rest without cloc
    assert z(None, 0.1) == "unscored"


def test_ht_618_score_reads_cloc_only_for_ht_bursts(monkeypatch):
    calls = []
    monkeypatch.setattr(W, "_load_cache", lambda: {})
    monkeypatch.setattr(W, "_save_cache", lambda c: None)
    monkeypatch.setattr(W, "ratios", lambda *a, **k: (1.0, 1.0))
    def fake_get(path, params):
        calls.append((path, params.get("symbol"), params))
        if path == "/fapi/v1/klines":
            t0 = (params["endTime"] + 1) - 15 * 60_000
            return _k(t0, [(10.0, 9.0, 9.5)] * 14 + [(10.0, 9.0, 9.1)])
        return []
    monkeypatch.setattr(W, "_get", fake_get)
    c = pd.DataFrame(dict(pair=["AUSDT", "BUSDT", "CUSDT"], entry_strategy=["FRENZY_WILLY_HT", "FRENZY_WILLY_HT", "FRENZY_WILLY"],
                          o=pd.to_datetime(["2026-12-01 10:00:30", "2026-12-02 00:00:00", "2026-12-03 00:00:00"]),
                          entry_frenzy_pre_vol15_vs60=[2.0, 1.0, 3.0], entry_btc_trend_gap_pct=[0.1, -0.1, 0.0]))
    sc, _ = W.score(c, int(pd.Timestamp("2026-12-04").value // 10**6), fetch=True)
    assert list(sc.z_BURST_REJECTION) == ["zone", "rest", "unscored"]                    # plain WILLY: never fetched → unscored
    assert [k[:2] for k in calls if k[0] == "/fapi/v1/klines"] == [("/fapi/v1/klines", "AUSDT")]   # only the HT burst fill is fetched
    p = [k[2] for k in calls if k[0] == "/fapi/v1/klines"][0]
    m0 = int(pd.Timestamp("2026-12-01 10:00:00").value // 10**6)
    assert p["endTime"] == m0 - 1 and p["limit"] == 20 and p["interval"] == "1m"          # only bars opened before the fill minute
    assert sc.cloc15.iloc[0] == pytest.approx(0.1) and sc.attrs.get("br_stop") is None


def test_ht_618_unreadable_burst_is_unscored_and_blocks_the_freeze(monkeypatch):
    monkeypatch.setattr(W, "_load_cache", lambda: {})
    monkeypatch.setattr(W, "_save_cache", lambda c: None)
    def boom(path, params):
        raise W.urllib.error.URLError("network down")                       # a network failure (not an answered-but-short window)
    monkeypatch.setattr(W, "_get", boom)
    monkeypatch.setattr(W, "ratios", lambda *a, **k: (1.0, 1.0))
    c = pd.DataFrame(dict(pair=["AUSDT"], entry_strategy=["FRENZY_WILLY_HT"], o=pd.to_datetime(["2026-12-01"]),
                          entry_frenzy_pre_vol15_vs60=[2.0], entry_btc_trend_gap_pct=[0.1]))
    sc, _ = W.score(c, int(pd.Timestamp("2026-12-04").value // 10**6), fetch=True)
    assert list(sc.z_BURST_REJECTION) == ["unscored"] and "not readable" in sc.attrs["br_stop"]
    src = _src("scripts/scout_willy_ht_lines.py")
    assert 'and not (ln == "BURST_REJECTION" and br_stop)' in src
    assert 's_ln = s[s.o >= pd.Timestamp(int(floor_618), unit="ms")]' in src


def test_ht_618_not_evaluable_before_the_commit(monkeypatch):
    real = W._GK.dl_deploy_ms
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda dl, *a, **k: None if dl == 618 else real(dl, *a, **k))
    c = pd.DataFrame(dict(pair=["AUSDT"], entry_strategy=["FRENZY_WILLY_HT"], o=pd.to_datetime(["2026-12-01"]), status=["CLOSED"],
                          pct=[1.0], entry_frenzy_pre_vol15_vs60=[1.0], entry_btc_trend_gap_pct=[0.1]))
    monkeypatch.setattr(W, "load", lambda paths=None: (c.assign(day=c.o.dt.floor("D")), pd.Timestamp("2026-12-30")))
    monkeypatch.setattr(W._GK, "split_operator_closed", lambda d: (d, d.iloc[0:0]))
    monkeypatch.setattr(W._GK, "operator_closed_line", lambda d: "0 operator-closed")
    out = "\n".join(W.run(now_ms=int(pd.Timestamp("2026-12-31").value // 10**6), state_path=None, fetch=False))
    assert "**BURST_REJECTION** — ⚠ NOT EVALUABLE until the DECISION_LOG 618 commit is in git." in out


def test_ht_618_cache_hit_short_marker_fetch_false_and_floor(monkeypatch):
    ms = int(pd.Timestamp("2026-12-01").value // 10**6)
    calls = []
    monkeypatch.setattr(W, "_save_cache", lambda c: None)
    monkeypatch.setattr(W, "_get", lambda path, params: calls.append(path) or [])
    monkeypatch.setattr(W, "ratios", lambda *a, **k: (1.0, 1.0))                  # CROWDED_LONGS reads are not under test here
    mk = lambda pairs, days: pd.DataFrame(dict(pair=pairs, entry_strategy=["FRENZY_WILLY_HT"] * len(pairs), o=pd.to_datetime(days),
                                              entry_frenzy_pre_vol15_vs60=[2.0] * len(pairs), entry_btc_trend_gap_pct=[0.1] * len(pairs)))
    # cache hit → no call, no stop · terminal SHORT → unscored, no stop
    monkeypatch.setattr(W, "_load_cache", lambda: {f"cloc|AUSDT|{ms}": 0.3, f"cloc|BUSDT|{ms}": "SHORT"})
    sc, _ = W.score(mk(["AUSDT", "BUSDT"], ["2026-12-01", "2026-12-01"]), ms + 10**9, fetch=True)
    assert list(sc.z_BURST_REJECTION) == ["zone", "unscored"] and sc.attrs["br_stop"] is None and calls == []
    # an answered-but-short window → cached SHORT, unscored, no stop
    cache = {}
    monkeypatch.setattr(W, "_load_cache", lambda: cache)
    sc, _ = W.score(mk(["CUSDT"], ["2026-12-01"]), ms + 10**9, fetch=True)
    assert list(sc.z_BURST_REJECTION) == ["unscored"] and sc.attrs["br_stop"] is None and cache[f"cloc|CUSDT|{ms}"] == "SHORT"
    # fetch=False with nothing cached → unscored AND the freeze is blocked
    monkeypatch.setattr(W, "_load_cache", lambda: {})
    sc, _ = W.score(mk(["DUSDT"], ["2026-12-01"]), ms + 10**9, fetch=False)
    assert list(sc.z_BURST_REJECTION) == ["unscored"] and "not cached" in sc.attrs["br_stop"]
    # fills before the DL 618 floor are never fetched and never block the freeze
    calls.clear()
    sc, _ = W.score(mk(["EUSDT"], ["2026-12-01"]), ms + 10**9, fetch=True, br_floor_ms=ms + 1)
    assert calls == [] and sc.attrs["br_stop"] is None and list(sc.z_BURST_REJECTION) == ["unscored"]


def test_ht_618_delisted_symbol_is_terminal_not_a_freeze_block(monkeypatch):
    cache = {}
    monkeypatch.setattr(W, "_load_cache", lambda: cache)
    monkeypatch.setattr(W, "_save_cache", lambda c: None)
    monkeypatch.setattr(W, "ratios", lambda *a, **k: (1.0, 1.0))
    def bad(path, params):
        raise W.urllib.error.HTTPError("u", 400, "Invalid symbol.", None, None)
    monkeypatch.setattr(W, "_get", bad)
    ms = int(pd.Timestamp("2026-12-01").value // 10**6)
    c = pd.DataFrame(dict(pair=["ZZZUSDT"], entry_strategy=["FRENZY_WILLY_HT"], o=pd.to_datetime(["2026-12-01"]),
                          entry_frenzy_pre_vol15_vs60=[2.0], entry_btc_trend_gap_pct=[0.1]))
    sc, _ = W.score(c, ms + 10**9, fetch=True)
    assert list(sc.z_BURST_REJECTION) == ["unscored"] and sc.attrs["br_stop"] is None and cache[f"cloc|ZZZUSDT|{ms}"] == "SHORT"
