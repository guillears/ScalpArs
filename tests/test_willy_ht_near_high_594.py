"""🧱 (DECISION_LOG 594) WILLY_HT_NEAR_HIGH — FRENZY_WILLY_HT refused when its decision price is < frenzy_willy_ht_min_dip_pct % below the
episode's spike high H (frenzy_walk's peak = entry_frenzy_willy_spike_high). The rule (services/watch_filters), the engine at RUNTIME (the real
_frenzy_open / _frenzy_willy_ht_try on the SQLite harness of tests/test_willy_global_hold_switch, and the real open_position for the stamp),
D11 wiring, and the scout's absolute revert gate. Hermetic: no network."""
import asyncio
import json
import os
import pathlib
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from services import watch_filters as WF
from tests.test_willy_global_hold_switch import _off, _hold_db, BAR_OPEN

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = NS(watch_block_willy_ht_failed_spike=False, watch_block_willy_failed_spike=False, watch_block_willy_gap=False, frenzy_willy_ht_min_dip_pct=1.4)


# ── ① the pure rule ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_rule_block_allow_off_and_unreadable():
    assert WF.ht_dip_pct(98.0, 100.0) == pytest.approx(2.0) and WF.ht_dip_pct(101.0, 100.0) == pytest.approx(-1.0)
    assert WF.ht_dip_pct(None, 100.0) is None and WF.ht_dip_pct(1.0, 0) is None and WF.ht_dip_pct(1.0, float("nan")) is None
    blk = lambda p, h, th=ON: WF.frenzy_blocks(th, "FRENZY_WILLY_HT", vs_vwap_pct=1.0, price=p, spike_high=h)
    assert blk(99.0, 100.0) == ["WILLY_HT_NEAR_HIGH"]                    # dip 1.0 % < 1.4 → refused
    assert blk(98.61, 100.0) == ["WILLY_HT_NEAR_HIGH"]                   # 1.39 %
    assert blk(98.6, 100.0) == []                                        # exactly 1.4 % → allowed (block only < X)
    assert blk(97.0, 100.0) == []                                        # 3 %
    assert blk(100.5, 100.0) == ["WILLY_HT_NEAR_HIGH"]                   # above the high (dip < 0) → refused
    assert blk(99.0, None) == [] and blk(None, 100.0) == []              # unreadable → never blocks (fail-open)
    for x in (0, None, -1, float("nan"), 51, "x"):                       # 0 / unreadable / out of range = OFF
        assert blk(99.0, 100.0, NS(**{**vars(ON), "frenzy_willy_ht_min_dip_pct": x})) == []
    assert WF.ht_min_dip(NS()) == 0.0                                    # field missing → off
    for sl in ("FRENZY_WILLY", "FRENZY_LITE", "FRENZY_LONG", "FRENZY_LITE_RETRY"):   # plain WILLY and the other sleeves untouched
        assert "WILLY_HT_NEAR_HIGH" not in WF.frenzy_blocks(ON, sl, vs_vwap_pct=1.0, price=99.0, spike_high=100.0)
    both = NS(**{**vars(ON), "watch_block_willy_ht_failed_spike": True})
    assert WF.frenzy_blocks(both, "FRENZY_WILLY_HT", vs_vwap_pct=-0.5, price=99.0, spike_high=100.0) == ["WILLY_FAILED_SPIKE", "WILLY_HT_NEAR_HIGH"]


def test_rule_agrees_with_the_registered_line():
    """the armed rule and the DL 580 NEAR_SPIKE_HIGH line share ONE zone: off_peak = −dip, zone ⇔ off_peak > −1.4 ⇔ dip < 1.4."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_willy_ht_lines as L
    assert L.OFF_PEAK_CUT == -ON.frenzy_willy_ht_min_dip_pct and L.DL_ARMED_NEAR_HIGH == 594
    for p in (95.0, 98.0, 98.59, 98.6, 98.61, 99.5, 100.0, 101.0):
        assert (L.zone_near_high(p, 100.0) == "zone") == bool(WF.frenzy_blocks(ON, "FRENZY_WILLY_HT", price=p, spike_high=100.0)), p
        assert L.off_peak(p, 100.0) == pytest.approx(-WF.ht_dip_pct(p, 100.0))


# ── ② the engine at runtime ───────────────────────────────────────────────────────────────────────────────────────────────────────────
def _run(monkeypatch, peak, x=1.4, live=1.0, via_try=False):
    import services.trading_engine as TE
    _off(monkeypatch, TE)
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**12)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", 0.5)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", 1.0)
    th = TE.config.trading_config.thresholds
    monkeypatch.setattr(th, "frenzy_willy_ht_min_dip_pct", x, raising=False)
    monkeypatch.setattr(th, "watch_block_willy_ht_failed_spike", True)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        blocks, opened, spawned, rec, ev = [], [], [], [], []
        e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
        e._flip_entry_fields = lambda *a, **k: {"entry_ema_gap_8_13": 0.5}
        e._sanitize_open_kwargs = lambda ef, s_, d: ef
        e._wh_spawn = lambda coro: (spawned.append(coro), coro.close())
        _br = e._blocked_record
        e._blocked_record = lambda *a, **k: (rec.append((a, k)), _br(*a, **k))[1]
        e._willy_event = lambda gate, pair, reason=None, price=None, tally=True: ev.append((gate, pair, reason, price, tally))

        async def gvv(sig, wait=True): return 0.5
        e._frenzy_gvol_value = gvv; e._frenzy_gvol_catchup_value = gvv

        async def fake_open(**k): opened.append(k); return object()
        e.open_position = fake_open
        flag = dict(pair="FOOUSDT", spike_ts=BAR_OPEN - 600_000, hours=5.0, vwap=0.9, vs_vwap_pct=1.0, vol_mult=400.0, run_pct=8.0, atr_pct=1.0,
                    volume_24h=5e7, price=1.0, live_price=live, adx_delta=1.0, di_spread=2.0, above_streak=14, above_share=90.0,
                    bar_ret_pct=-0.1, peak=peak)
        async with SL() as db:
            if via_try:
                pd_ = dict(trig="A", armed=BAR_OPEN - 600_000)
                r = await e._frenzy_willy_ht_try(db, "FOOUSDT", dict(bar_ret_pct=-0.1), flag, {"adx": 30.0, "rsi": 55.0}, BAR_OPEN, pd_, 0.5, "x")
            else:
                r = await e._frenzy_open(db, flag, {"adx": 30.0, "rsi": 55.0}, BAR_OPEN, willy="A", willy_ht=True)
        await eng.dispose()
        return r, blocks, opened, spawned, rec, ev, flag
    return asyncio.run(go())


def test_engine_blocks_near_the_high(monkeypatch):
    r, blocks, opened, spawned, rec, ev, flag = _run(monkeypatch, peak=1.01)          # dip 0.99 % < 1.4
    assert r is None and opened == []
    assert [b for b in blocks if "_BLOCK_" in b] == ["FRENZY_WILLY_HT_BLOCK_WILLY_HT_NEAR_HIGH"]   # one counter
    assert len(rec) == 1 and len(spawned) == 1                                        # one blocked_signals row queued
    (sl, d, pair, px, lines, ef), kw = rec[0]
    assert (sl, d, pair, px, lines) == ("FRENZY_WILLY_HT", "LONG", "FOOUSDT", 1.0, ["WILLY_HT_NEAR_HIGH"])
    assert ef["entry_frenzy_willy_spike_high"] == 1.01 and ef["entry_frenzy_willy_dip_pct"] == pytest.approx(0.9901, abs=1e-4)
    assert ef["entry_frenzy_willy_trigger"] == "A" and ef["entry_frenzy_spike_at"] == BAR_OPEN - 600_000   # the pricer's snapshot
    assert len(ev) == 1 and ev[0][0] == "WILLY_HT_NEAR_HIGH" and ev[0][4] is False and "dip 0.99 %" in ev[0][2]   # WILLY journal line
    assert "armed filter WILLY_HT_NEAR_HIGH" in flag["willy_ht_last"]


def test_engine_allows_a_real_dip_and_stamps_it(monkeypatch):
    r, blocks, opened, spawned, rec, ev, _ = _run(monkeypatch, peak=1.05)             # dip 4.76 %
    assert r is True and len(opened) == 1 and rec == [] and ev == [] and not [b for b in blocks if "NEAR_HIGH" in b]
    k = opened[0]
    assert k["frenzy_willy_ht"] is True and k["entry_frenzy_willy_spike_high"] == 1.05
    assert k["entry_frenzy_willy_dip_pct"] == pytest.approx(4.7619, abs=1e-4)


def test_engine_judges_the_decision_price(monkeypatch):
    """the live price (the decision price), not the signal close: close 1.0 is 1.96 % below H, live 1.008 is 1.18 % below → refused."""
    r, blocks, opened, *_ = _run(monkeypatch, peak=1.02, live=1.008)
    assert r is None and opened == [] and "FRENZY_WILLY_HT_BLOCK_WILLY_HT_NEAR_HIGH" in blocks


def test_engine_off_at_zero(monkeypatch):
    r, blocks, opened, _, rec, ev, _ = _run(monkeypatch, peak=1.001, x=0)
    assert r is True and len(opened) == 1 and rec == [] and not [b for b in blocks if "NEAR_HIGH" in b]


def test_engine_fail_open_on_unreadable_high(monkeypatch):
    r, blocks, opened, _, rec, ev, _ = _run(monkeypatch, peak=None)
    assert r is True and len(opened) == 1 and rec == [] and "WILLY_HT_NEAR_HIGH_UNREAD" in blocks
    r, blocks, *_ = _run(monkeypatch, peak=None, x=0)
    assert "WILLY_HT_NEAR_HIGH_UNREAD" not in blocks                                   # line off → no unread count


def test_ht_pending_entry_ends_on_the_block(monkeypatch):
    """HT's one-shot rule: an armed-filter refusal ends the pending entry ("ended"), exactly like the other HT refusals."""
    r, blocks, opened, *_ = _run(monkeypatch, peak=1.01, via_try=True)
    assert r == "ended" and opened == [] and "FRENZY_WILLY_HT_BLOCK_WILLY_HT_NEAR_HIGH" in blocks
    r, _, opened, *_ = _run(monkeypatch, peak=1.05, via_try=True)
    assert r == "opened" and len(opened) == 1


def test_open_position_writes_the_stamps(monkeypatch):
    from tests.test_frenzy_willy_ht import _open
    r, _, _ = _open(monkeypatch, "FOOUSDT", frenzy_willy=True, frenzy_willy_ht=True, entry_frenzy_willy_spike_high=1.5, entry_frenzy_willy_dip_pct=33.3333)
    assert r is not None and r.entry_strategy == "FRENZY_WILLY_HT"
    assert r.entry_frenzy_willy_spike_high == 1.5 and r.entry_frenzy_willy_dip_pct == pytest.approx(33.3333)


# ── ③ D11 / D13 wiring ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_config_json_ui_and_stamp_contract():
    import config
    import models
    from services.trading_engine import TradingEngine
    mf = type(config.trading_config.thresholds).model_fields
    assert mf["frenzy_willy_ht_min_dip_pct"].default == 0.0                           # code default OFF
    src = (ROOT / "config.py").read_text()
    assert "DECISION_LOG 594" in src and "first 15 priced blocked" in src             # evidence + revert gate comment
    assert json.loads((ROOT / "trading_config.json").read_text())["thresholds"]["frenzy_willy_ht_min_dip_pct"] == 1.4
    html = (ROOT / "templates/index.html").read_text()
    assert html.count('id="config-fz-willy-ht-min-dip"') == 1
    assert "['config-fz-willy-ht-min-dip', 'frenzy_willy_ht_min_dip_pct', 0]" in html  # FRENZY_NUM_FIELDS: the load AND the save walk it
    assert "HT: min dip below spike high % (0 = off)" in html and "HT near-high block ${" in html
    assert "entry_frenzy_willy_dip_pct" in models.Order.__table__.columns           # rides the orders CSV
    assert "('entry_frenzy_willy_dip_pct', 'FLOAT')" in (ROOT / "database.py").read_text()   # migration
    assert TradingEngine.STAMP_SLEEVE_PREFIXES[0] == "entry_frenzy_"                  # D13: a sleeve stamp by prefix
    assert 'spike_high=_num(r, "entry_frenzy_willy_spike_high")' in (ROOT / "scripts/build_master_pool.py").read_text()   # master stack parity


# ── ④ the scout: the line priced + the absolute revert gate ───────────────────────────────────────────────────────────────────────────
def _S():
    sys.path.insert(0, str(ROOT / "scripts"))
    import scout_blocked_filters as S
    return S


def _B(pcts, days=None, pend=0, sole=True, line="WILLY_HT_NEAR_HIGH"):
    days = days if days is not None else list(range(len(pcts)))
    rows = [dict(te=i, lines=line, status="PRICED", pct=p, day=d, sleeve="FRENZY_WILLY_HT", sole=sole) for i, (p, d) in enumerate(zip(pcts, days))]
    rows += [dict(te=-1, lines=line, status="PENDING", pct=None, day=0, sleeve="FRENZY_WILLY_HT", sole=sole)] * pend
    return pd.DataFrame(rows)


def test_scout_line_and_gate():
    S = _S()
    assert "WILLY_HT_NEAR_HIGH" in S.LINES["FRENZY_WILLY_HT"] and S.ABS_GATE["WILLY_HT_NEAR_HIGH"] == (15, 8)
    assert S.LINES["FRENZY_LITE_RETRY"] == ("RETRY_BTC1H",) and S.LINES["FRENZY_WILLY"] == ("WILLY_GAP", "WILLY_FAILED_SPIKE")   # review: no line lost
    assert "WILLY_HT_NEAR_HIGH" in S.SOLE_GATED and S.GATE_KEY["FRENZY_WILLY_HT"] == "FRENZY_WILLY_HT@587"
    K = pd.DataFrame(columns=["te", "status", "pct", "sleeve"])                       # no kept fills needed (absolute gate)
    g = lambda B, **k: S.gate_read(B, K, "WILLY_HT_NEAR_HIGH", **k)[0]
    assert g(_B([-1.0] * 14)) == "COLLECTING"
    assert g(_B([-1.0] * 15)) == "KEEP"
    assert g(_B([-1.0] * 15, days=[i // 2 for i in range(15)])) == "KEEP"             # 15 on 8 days
    assert g(_B([-1.0] * 15, days=[i // 3 for i in range(15)])) == "COLLECTING"       # 15 on 5 days → waits for the 8th day
    assert g(_B([-1.0] * 15 + [-1.0] * 9, days=[i // 3 for i in range(24)])) == "KEEP"  # decided at the first prefix with 8 days
    assert g(_B([1.0] * 7 + [-1.0] * 8)) == "KEEP"                                    # avg −0.07 < 0 → keep
    assert g(_B([1.0] * 8 + [-1.0] * 7)).startswith("REVERT")                         # avg +0.07 ≥ 0 → revert
    assert g(_B([1.0] * 5 + [-1.0] * 5 + [0.0] * 5)).startswith("REVERT")             # avg exactly 0 → revert (≥ 0)
    assert g(_B([-1.0] * 15, pend=1)) == "COLLECTING"                                 # an earlier refusal still pending
    assert g(_B([-1.0] * 15, sole=False), sole=True) == "COLLECTING"                  # the sole-line case: shared refusals not counted
    assert g(_B([1.0] * 15, sole=True), sole=True).startswith("REVERT")


def test_scout_run_freezes_both_keys(tmp_path, monkeypatch):
    S = _S()
    T0 = pd.Timestamp("2026-11-20 00:00"); since = int(T0.value // 10**6)
    out = []
    for i in range(16):
        t = T0 + pd.Timedelta(days=i // 2, minutes=i * 7)
        out.append(dict(t=t.strftime("%Y-%m-%dT%H:%M:%S.000"), e="BLOCKED", pair=f"P{i}USDT", dir="LONG", price=1.0, gate="WILLY_HT_NEAR_HIGH",
                        strategy="FRENZY_WILLY_HT", bl_sleeve="FRENZY_WILLY_HT", bl_bar_at=t.strftime("%Y-%m-%dT%H:%M:00.000"),
                        bl_lines="WILLY_HT_NEAR_HIGH", bl_sole=True, bl_is_paper=True, bl_entry_fields="{}"))
    p = tmp_path / "scalpars_decisions_paper_2026-12-01_00-00-00.csv"
    pd.DataFrame(out).to_csv(p, index=False)
    monkeypatch.setattr(S, "price_all", lambda B, now_ms, fetch=True, store_path=None: B.assign(pct=-1.0, why="STOP", status="PRICED", exit_ms=B.te + 60_000))
    st = tmp_path / "g.json"
    txt = "\n".join(S.run(now_ms=since + 30 * 86_400_000, dec_paths=[str(p)], ord_paths=[], state_path=str(st), store_path=None, since_ms=since))
    g = json.load(open(st))
    assert g["FRENZY_WILLY_HT@587:WILLY_HT_NEAR_HIGH"]["state"] == "KEEP" and g["FRENZY_WILLY_HT@587:WILLY_HT_NEAR_HIGH|sole"]["state"] == "KEEP"
    assert "FRENZY_WILLY_HT · WILLY_HT_NEAR_HIGH (sole)" in txt and "ABSOLUTE gate" in txt


def test_config_validator_bounds_min_dip():
    """594 review: a hand-edited value outside 0 / (0, 50] is refused by the config validator (it would silently switch the filter off)."""
    from types import SimpleNamespace as NS
    from services.frenzy import frenzy_willy_ht_config_errors as E
    base = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_ht_tp_pct=5.0)
    ok = lambda v: not [e for e in E(NS(**base, frenzy_willy_ht_min_dip_pct=v)) if "min_dip" in e]
    assert ok(0) and ok(1.4) and ok(50) and not ok(60) and not ok(-1)
