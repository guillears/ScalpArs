"""🚦 (DECISION_LOG 660) execution refusals → blocked_signals rows (EXEC_DISLOC / EXEC_LATE / EXEC_CATCHUP_MOVED): the engine helper, its
wiring at every refusal site, the pricer's separation (filter readers never see EXEC_ rows; the exec section prices them on its own) and the
CHEAPER / DEARER side. Hermetic: temp exports, pricers stubbed."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import scout_blocked_filters as S  # noqa: E402

T0 = pd.Timestamp("2026-11-30 00:00")
SINCE = int(T0.value // 10**6)


def _src(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()


def _eng(monkeypatch):
    from services.trading_engine import TradingEngine
    eng = TradingEngine.__new__(TradingEngine)
    spawned = []
    monkeypatch.setattr(TradingEngine, "is_paper_mode", property(lambda self: True), raising=False)
    eng._wh_spawn = lambda coro: (spawned.append(coro), coro.close())
    rows = []
    real = TradingEngine._blocked_record
    eng._blocked_record = lambda *a, **k: (rows.append((a, k)), real(eng, *a, **k))[1]
    return eng, rows, spawned


def test_exec_refusal_row_shape(monkeypatch):
    eng, rows, spawned = _eng(monkeypatch)
    assert eng._exec_refusal("FRENZY_WILLY_HT", "LONG", "AIOTUSDT", "DISLOC", 0.09137, live_px=0.08853, stage="order", max_v=1.0,
                             fields={"entry_frenzy_spike_at": 1_700_000_000_000}, bar_ms=1_700_000_100_000) is True
    (sleeve, d, pair, px, lines, ef), kw = rows[0]
    assert (sleeve, d, pair, lines) == ("FRENZY_WILLY_HT", "LONG", "AIOTUSDT", ["EXEC_DISLOC"])
    assert px == pytest.approx(0.08853)                                   # the price the order would have met
    assert ef["exec_move_pct"] == pytest.approx(-3.1082, abs=1e-3) and ef["exec_stage"] == "order" and ef["exec_max"] == 1.0
    assert ef["entry_frenzy_spike_at"] == 1_700_000_000_000 and kw["bar_ms"] == 1_700_000_100_000 and len(spawned) == 1


def test_exec_refusal_late_uses_decision_price_and_never_raises(monkeypatch):
    eng, rows, _ = _eng(monkeypatch)
    assert eng._exec_refusal("FRENZY_WILLY_HT", "LONG", "COOKIEUSDT", "LATE", 0.0099, stage="scan", late_s=149.2, max_v=120) is True
    (_, _, _, px, lines, ef), _ = rows[0]
    assert px == pytest.approx(0.0099) and lines == ["EXEC_LATE"] and ef["exec_move_pct"] is None and ef["exec_late_s"] == 149.2
    assert eng._exec_refusal("S", "LONG", "P", "DISLOC", object(), live_px="x") is False          # no price at all → not queued, no raise
    assert eng._exec_refusal("S", "LONG", "P", "LATE", float("nan")) is False


def test_fz_exec_fields_carry_pricer_stamps():
    from services.trading_engine import TradingEngine
    f = TradingEngine._fz_exec_fields({"spike_ts": 1, "peak": 2.0, "vs_vwap_pct": -1.0, "vol_mult": 5, "hours": 0.5, "volume_24h": 9})
    assert f["entry_frenzy_spike_at"] == 1 and f["entry_frenzy_willy_spike_high"] == 2.0
    assert TradingEngine._fz_exec_fields(None)["entry_frenzy_spike_at"] is None


def test_every_execution_refusal_site_records():
    src = _src("services/trading_engine.py")
    for needle in ('"LATE", price, stage="scan"', '"CATCHUP_MOVED", _close, live_px=price', '"DISLOC", _close, live_px=price',
                   '"LATE", current_price, live_px=_lt_px, stage="order", late_s=_fz_late2', 'late_s=_ex_late', 'late_s=_ds_late', 'late_s=_fb_late',
                   '"DISLOC", current_price, live_px=_sg_live_px', '"CATCHUP_MOVED", frenzy_catchup_ref_price, live_px=_cu_px'):
        assert src.count(needle) == 1, needle
    # each call sits right before its refusal's return (never changes the decision)
    i = src.index('"DISLOC", current_price, live_px=_sg_live_px')
    assert "return None" in src[i:i + 400] and "_record_filter_block(f\"{_fz_bk}_DISLOC\"" in src[i - 1200:i]


def _dec(tmp_path, rows, name="scalpars_decisions_paper_2026-12-01_00-00-00.csv"):
    out = []
    for i, (pair, sleeve, lines, px, dr, snap) in enumerate(rows):
        t = T0 + pd.Timedelta(minutes=i * 7)
        out.append(dict(t=t.strftime("%Y-%m-%dT%H:%M:%S.000"), e="BLOCKED", pair=pair, dir=dr, price=px, gate=lines, strategy=sleeve,
                        bl_sleeve=sleeve, bl_bar_at=t.strftime("%Y-%m-%dT%H:%M:00.000"), bl_lines=lines, bl_sole=True, bl_is_paper=True,
                        bl_entry_fields=json.dumps(snap)))
    p = tmp_path / name
    pd.DataFrame(out).to_csv(p, index=False)
    return [str(p)]


def test_filter_readers_never_see_exec_rows(tmp_path):
    paths = _dec(tmp_path, [("AUSDT", "FRENZY_WILLY_HT", "WILLY_FAILED_SPIKE", 1.0, "LONG", {}),
                            ("BUSDT", "FRENZY_WILLY_HT", "EXEC_DISLOC", 2.0, "LONG", {"exec_move_pct": -2.4}),
                            ("CUSDT", "FRENZY_WILLY_HT", "EXEC_LATE", 3.0, "LONG", {})])
    B = S.load_blocked(paths, SINCE)
    assert list(B.pair) == ["AUSDT"] and not B.key.str.startswith("X|").any()
    X = S.load_blocked(paths, SINCE, exec_rows=True)
    assert sorted(X.pair) == ["BUSDT", "CUSDT"] and X.key.str.startswith("X|").all()
    assert S.is_exec_lines("EXEC_DISLOC") and not S.is_exec_lines("WILLY_FAILED_SPIKE") and not S.is_exec_lines(None)


@pytest.mark.parametrize("d,m,side", [("LONG", -2.4, "CHEAPER"), ("LONG", 3.1, "DEARER"), ("SHORT", 1.5, "CHEAPER"),
                                      ("SHORT", -1.5, "DEARER"), ("LONG", None, "–"), ("LONG", 0, "–")])
def test_exec_move_side(d, m, side):
    assert S.exec_move_side(d, m) == side


def test_exec_section_prices_and_splits(tmp_path, monkeypatch):
    paths = _dec(tmp_path, [("BUSDT", "FRENZY_WILLY_HT", "EXEC_DISLOC", 2.0, "LONG", {"exec_move_pct": -2.4}),
                            ("DUSDT", "FRENZY_WILLY_HT", "EXEC_DISLOC", 2.0, "LONG", {"exec_move_pct": 1.8}),
                            ("EUSDT", "EXHAUSTION_SHORT", "EXEC_LATE", 2.0, "SHORT", {})])

    def fake_price(B, now_ms, fetch=True, store_path=None):
        B = B.copy()
        B["pct"] = [6.0 if p == "BUSDT" else (-4.0 if p == "DUSDT" else None) for p in B.pair]
        B["why"] = ["TP" if p == "BUSDT" else ("STOP" if p == "DUSDT" else "UNKNOWN_SLEEVE") for p in B.pair]
        B["status"] = ["PRICED" if p in ("BUSDT", "DUSDT") else "NOT_EVALUABLE" for p in B.pair]
        B["exit_ms"] = [int(t) + 60_000 for t in B.te]
        return B
    monkeypatch.setattr(S, "price_all", fake_price)
    L = "\n".join(S.exec_section(paths, pd.DataFrame(columns=["sleeve", "status", "pct"]), SINCE + 86_400_000, fetch=False, store_path=None,
                                 since_ms=SINCE))
    assert "FRENZY_WILLY_HT · DISLOC · CHEAPER | 1 | 1 |" in L and "🟢 +6.000" in L
    assert "FRENZY_WILLY_HT · DISLOC · DEARER | 1 | 1 |" in L and "🔴 -4.000" in L
    assert "EXHAUSTION_SHORT · LATE · – | 1 | 0 |" in L and "not evaluable 1" in L


def test_exec_section_before_commit_and_run_never_breaks(monkeypatch):
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    L = S.exec_section([], None, SINCE, fetch=False, store_path=None)
    assert any("Not evaluable yet" in x for x in L)


@pytest.mark.parametrize("ref,px,d,out", [(100, 101.5, "LONG", True), (100, 97.0, "LONG", False), (100, 100.9, "LONG", False),
                                          (100, 101.0, "LONG", False), (100, 98.5, "SHORT", True), (100, 103, "SHORT", False),
                                          (100, 90, "LONG", False), (0, 1, "LONG", False), (100, None, "LONG", False), ("x", 1, "LONG", False)])
def test_entry_moved_against_one_sided(ref, px, d, out):
    from services.frenzy import entry_moved_against
    assert entry_moved_against(1.0, ref, px, d) is out
    assert entry_moved_against(0, 100, 150, "LONG") is False and entry_moved_against(None, 100, 150, "LONG") is False   # off


def test_engine_uses_the_one_sided_guard():
    src = _src("services/trading_engine.py")
    assert "if _sg_live_px and entry_moved_against(_sg_max, current_price, _sg_live_px, direction):" in src
    assert 'entry_moved_against(_dmax, _close, price, "LONG")' in src and "abs(price / _close - 1) * 100 > _dmax" not in src
    assert "bullrun_disloc_exceeded(_sg_max" not in src                       # Bull-run keeps its own two-sided 0.3 % guard (DL 87)


def test_order_stage_late_without_live_price_is_counted_not_priced(tmp_path, monkeypatch):
    paths = _dec(tmp_path, [("LUSDT", "FRENZY_WILLY_HT", "EXEC_LATE", 2.0, "LONG", {"exec_stage": "order", "exec_live_px": None}),
                            ("MUSDT", "FRENZY_WILLY_HT", "EXEC_LATE", 2.0, "LONG", {"exec_stage": "order", "exec_live_px": 2.01})])
    seen = []

    def fake_price(B, now_ms, fetch=True, store_path=None):
        seen.extend(B.pair)
        return B.assign(pct=1.0, why="TP", status="PRICED", exit_ms=[int(t) + 60_000 for t in B.te])
    monkeypatch.setattr(S, "price_all", fake_price)
    L = "\n".join(S.exec_section(paths, None, SINCE + 86_400_000, fetch=False, store_path=None, since_ms=SINCE))
    assert seen == ["MUSDT"] and "not evaluable 1" in L and "priced 1" in L


def _kept(rows):
    return pd.DataFrame([dict(sleeve=sl, status="PRICED", t_ms=SINCE + i * 86_400_000, te=SINCE + i * 86_400_000, day=f"2026-12-{i + 1:02d}",
                              pct=p, booked=p, snap={"entry_book_drift_pct": d}) for i, (sl, p, d) in enumerate(rows)])


def test_one_sided_revert_read(monkeypatch):
    K = _kept([("FRENZY_WILLY_HT", -2.0, -1.5)] * 10 + [("FRENZY_WILLY_HT", 1.0, 0.1)] * 3)
    K["day"] = [f"2026-12-{(i % 6) + 1:02d}" for i in range(len(K))]
    txt = " ".join(S.exec_admitted_lines(K, SINCE, max_pct=1.0))
    assert "10 cheaper-admitted fills" in txt and "REVERT alert" in txt
    K2 = _kept([("FRENZY_WILLY_HT", 6.0, -1.5)] * 3)
    assert "COLLECTING (3/10" in " ".join(S.exec_admitted_lines(K2, SINCE, max_pct=1.0))
    assert "0 kept fills yet" in " ".join(S.exec_admitted_lines(None, SINCE, max_pct=1.0))
    assert "guard off" in " ".join(S.exec_admitted_lines(K2, SINCE, max_pct=0))


def test_bullrun_fallback_leg_one_sided_placement_unchanged():
    src = _src("services/trading_engine.py")
    # 🚦 (DL 661) both fallback legs (live maker wait + paper sim) refuse only a price that ran AGAINST the trade
    assert src.count("entry_moved_against(max_disloc_pct, _mid, _px2, direction)") == 1
    assert src.count("entry_moved_against(max_disloc_pct, current_price, fallback_price, direction)") == 1
    # the PLACEMENT legs (a resting limit > max % from the mid = a gapped book) stay two-sided on purpose
    assert src.count("bullrun_disloc_exceeded(max_disloc_pct, _mid, limit_price)") == 1
    assert src.count("bullrun_disloc_exceeded(max_disloc_pct, current_price, limit_price)") == 1
    # both BULLRUN_DISLOC refusals now write an EXEC_ row
    assert src.count('self._exec_refusal(_wh_es, direction, pair, "DISLOC", (result.get(\'price\') or current_price), live_px=result.get(\'live_px\')') == 2
    assert src.count("'leg': 'placement'") == 2 and src.count("'leg': 'fallback'") == 2
