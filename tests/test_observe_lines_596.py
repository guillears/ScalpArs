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
