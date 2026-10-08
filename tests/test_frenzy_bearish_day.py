"""🐻 Oct-8 (DECISION_LOG 250, operator declared overrides): FRENZY_LONG / FRENZY_WIDE / FRENZY_LITE bearish-day entry block
(frenzy_bearish_day_block; bearish = BTC last closed daily return < 0 ∧ BTC 5m trend gap < 0, the DECISION_LOG 239 definition), the ATR cap
3.0 in the JSON and the exit back on the fixed +3 TP (frenzy_lock_arm_pct 0). Pure rule + engine wiring per sleeve + D11 / D12 surfaces."""
import asyncio
import datetime as dt
import json
import math
import os
import re
import sys
import time
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.frenzy import BAR_MS, frenzy_bearish_block, frenzy_bearish_day, frenzy_exit_for  # noqa: E402

ON = NS(frenzy_bearish_day_block=True)


# ── pure rule ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_bearish_needs_both_legs_negative():
    assert frenzy_bearish_day(-0.5, -0.1) is True
    assert frenzy_bearish_day(-0.5, 0.0) is False and frenzy_bearish_day(0.0, -0.1) is False      # 0 is not < 0 (either leg)
    assert frenzy_bearish_day(0.3, -0.1) is False and frenzy_bearish_day(-0.3, 0.2) is False and frenzy_bearish_day(1, 1) is False


def test_unreadable_legs_are_undecidable_unless_the_other_leg_decides():
    assert frenzy_bearish_day(None, -0.1) is None and frenzy_bearish_day(-0.5, None) is None and frenzy_bearish_day(None, None) is None
    assert frenzy_bearish_day(float("nan"), -0.1) is None and frenzy_bearish_day(-0.5, math.inf) is None and frenzy_bearish_day("x", -1) is None
    assert frenzy_bearish_day(None, 0.2) is False and frenzy_bearish_day(0.4, None) is False                # a readable leg ≥ 0 decides: not bearish


def test_block_rule_fail_open_and_switch():
    assert frenzy_bearish_block(-0.5, -0.1, ON) == "BEARISH_DAY"
    assert frenzy_bearish_block(-0.5, 0.1, ON) is None and frenzy_bearish_block(0.5, -0.1, ON) is None
    assert frenzy_bearish_block(None, -0.1, ON) == "BEARISH_UNREAD" and frenzy_bearish_block(None, None, ON) == "BEARISH_UNREAD"   # caller fails OPEN
    assert frenzy_bearish_block(-0.5, -0.1, NS(frenzy_bearish_day_block=False)) is None and frenzy_bearish_block(-0.5, -0.1, NS()) is None   # off / missing


# ── engine: _frenzy_open per sleeve (the one gate site for FRENZY fresh ON + catch-up, WIDE, LITE) ─────────────────────────────────

def _run_open(monkeypatch, ret1d, gap, wide=False, lite=False, catchup_now=None, block_on=True, ef=None, zone_age_s=60, zone_on=True):
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models, config as C
    import services.trading_engine as TE
    th = C.trading_config.thresholds
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**6)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", ret1d)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", gap)
    monkeypatch.setattr(TE, "_zone_stamps_at", time.time() - zone_age_s)
    monkeypatch.setattr(th, "frenzy_bearish_day_block", block_on)
    monkeypatch.setattr(C.trading_config, "entry_zone_stamps_enabled", zone_on, raising=False)
    monkeypatch.setattr(th, "frenzy_gvol_max", 1.0)
    bar_open = int(time.time() // 300) * 300_000

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        blocks, opened = [], []
        try:
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
                e._record_filter_block = lambda name, d, had_room=True: blocks.append(name)
                e._flip_entry_fields = lambda *a, **k: dict(ef or {})
                e._sanitize_open_kwargs = lambda x, s, d: x

                async def gvv(sig, wait=True):
                    return 0.5
                e._frenzy_gvol_value = gvv
                e._frenzy_gvol_catchup_value = gvv

                async def fake_open(**kw):
                    opened.append(kw); return None
                e.open_position = fake_open
                flag = dict(pair="FOOUSDT", spike_ts=bar_open - 7_200_000, hours=3.0, vwap=1.0, vs_vwap_pct=1.0, vol_mult=150.0, run_pct=20.0,
                            atr_pct=1.5, volume_24h=5e7, price=1.0, live_price=1.0, code="FRENZY_GREEN_BAR", above_streak=20, bar_ret_pct=0.1,
                            above_share=90.0)
                await e._frenzy_open(db, flag, {"rsi": 50} if ef is not None else {}, bar_open, wide=wide, lite=lite, catchup_now=catchup_now)
                await e._frenzy_open(db, flag, {"rsi": 50} if ef is not None else {}, bar_open, wide=wide, lite=lite, catchup_now=catchup_now)   # same bar again
                return blocks, opened, flag.get("last_fire") or "", e
        finally:
            await eng.dispose()
    return asyncio.run(run())


@pytest.mark.parametrize("wide,lite,code,tag", [(False, False, "FRENZY_BEARISH_DAY", ""), (True, False, "FRENZY_WIDE_BEARISH_DAY", "WIDE "),
                                                (False, True, "FRENZY_LITE_BEARISH_DAY", "LITE ")])
def test_bearish_day_refuses_each_sleeve(monkeypatch, wide, lite, code, tag):
    b, o, lf, _ = _run_open(monkeypatch, -1.76, -0.12, wide=wide, lite=lite)
    assert b == [code, code] and o == []                                                     # counted per refusal, never opened
    assert f"{tag}refused: bearish day (BTC day −1.76%, trend gap −0.12%)" in lf


def test_bearish_day_refuses_a_catchup(monkeypatch):
    bar = int(time.time() // 300) * 300_000
    b, o, lf, _ = _run_open(monkeypatch, -0.4, -0.3, catchup_now=bar + 300_000)
    assert "FRENZY_BEARISH_DAY" in b and o == [] and "refused: bearish day" in lf


def test_not_bearish_or_switch_off_opens(monkeypatch):
    for ret, gap, on in ((0.5, -0.3, True), (-0.5, 0.3, True), (-0.5, -0.3, False)):
        b, o, _, _ = _run_open(monkeypatch, ret, gap, block_on=on)
        assert not [x for x in b if "BEARISH" in x] and len(o) == 2, (ret, gap, on)


def test_unread_fails_open_counted_once_per_bar(monkeypatch):
    b, o, _, e = _run_open(monkeypatch, None, -0.3)
    bb = lambda b: [x for x in b if "BEARISH" in x]
    assert bb(b) == ["FRENZY_BEARISH_UNREAD"] and len(o) == 2                                # both attempts open; one count for the bar
    b, o, _, _ = _run_open(monkeypatch, -0.5, -0.3, zone_age_s=4000)                       # stale zone stamp (> 30 min) = unread, like the stamp
    assert bb(b) == ["FRENZY_BEARISH_UNREAD"] and len(o) == 2


def test_parity_reads_the_value_the_fill_is_stamped_with(monkeypatch):
    # _ef carries entry_btc_1d_ret_pct → that is what open_position stamps → that is what the gate reads (the global is ignored)
    b, o, _, _ = _run_open(monkeypatch, -0.9, -0.3, ef={"entry_btc_1d_ret_pct": 0.4})
    assert not [x for x in b if "BEARISH" in x] and len(o) == 2 and o[0]["entry_btc_1d_ret_pct"] == 0.4
    b, o, lf, _ = _run_open(monkeypatch, 0.9, -0.3, ef={"entry_btc_1d_ret_pct": -0.2})
    assert b[0] == "FRENZY_BEARISH_DAY" and o == [] and "BTC day −0.20%" in lf


def test_stale_or_disabled_zone_stamps_never_block(monkeypatch, caplog):
    """Caveman review: a frozen negative daily return (stale zone stamps / switch off) must never block — UNREAD, fail-open, and the
    stale value is dropped from the fill's stamp too (parity: the stamp is None, the gate read None)."""
    import logging
    bb = lambda b: [x for x in b if "BEARISH" in x]
    b, o, _, _ = _run_open(monkeypatch, -0.9, -0.3, ef={"entry_btc_1d_ret_pct": -0.9}, zone_age_s=4000)
    assert bb(b) == ["FRENZY_BEARISH_UNREAD"] and len(o) == 2 and all(x.get("entry_btc_1d_ret_pct") is None for x in o)
    caplog.set_level(logging.WARNING); caplog.clear()
    b, o, _, _ = _run_open(monkeypatch, -0.9, -0.3, ef={"entry_btc_1d_ret_pct": -0.9}, zone_on=False)
    assert bb(b) == ["FRENZY_BEARISH_UNREAD"] and len(o) == 2 and all(x.get("entry_btc_1d_ret_pct") is None for x in o)
    assert sum("bearish-day gate UNREAD" in r.getMessage() for r in caplog.records) == 1          # the warning is once per bar like the counter


def test_gate_sits_last_in_frenzy_open():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("async def _frenzy_open(")
    body = eng[i:eng.index("async def _maybe_open_surge(", i)]
    g = body.index("_bb = frenzy_bearish_block(_b1d, _btg, th)")
    for earlier in ("_MAX_SLOTS", "_PAIR_DAY_CAP", "frenzy_gvol_block(", "frenzy_wide_choppy(", "frenzy_wide_hold_green_block(", "FRENZY_ENTRY_MAX_LATE_S", "_DISLOC"):
        assert body.index(earlier) < g, earlier
    assert g < body.index("order = await self.open_position(")
    assert 'self._record_filter_block(f"{_bk}_BEARISH_DAY", "LONG")' in body and 'self._record_filter_block("FRENZY_BEARISH_UNREAD", "LONG")' in body


# ── LITE: the bearish refusal comes AFTER the stretch is judged (no retry on the next bar) ─────────────────────────────────────────

def test_lite_bearish_refusal_ends_the_stretch(monkeypatch):
    import services.trading_engine as TE
    from tests.test_frenzy_lite import _ep, _th, _engine, _DB, NEXT, SID, LAST, NORM
    th = _th(frenzy_bearish_day_block=True, frenzy_lite_max_slots=2, frenzy_max_entries_per_pair_day=3, frenzy_max_entry_dislocation_pct=1.0)
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=th))
    monkeypatch.setattr(TE, "FRENZY_ENTRY_MAX_LATE_S", 10**9)
    monkeypatch.setattr(TE, "_current_btc_1d_ret_pct", -0.8)
    monkeypatch.setattr(TE, "_current_btc_trend_gap_pct", -0.05)
    monkeypatch.setattr(TE, "_zone_stamps_at", time.time())
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import models
    bars = [[LAST - (299 - i) * BAR_MS, 1, 1.01, 0.99, 1.0, 10.0] for i in range(300)]

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        blocks, opened = [], []
        e = _engine(TE, blocks)
        e._frenzy_lite_prev_missed = lambda *a, **k: False

        async def done_db(db, pair, sid):
            return False
        e._frenzy_lite_done_db = done_db

        async def gvv(sig, wait=True):
            return 0.5
        e._frenzy_gvol_value = gvv

        async def fake_open(**kw):
            opened.append(kw); return object()
        e.open_position = fake_open
        out = []
        try:
            async with async_sessionmaker(eng, expire_on_commit=False)() as db:
                for ep in (_ep(), _ep(**NEXT)):
                    del blocks[:]
                    flag = dict(ep, pair="FOOUSDT", volume_24h=50e6, price=1.0, live_price=1.0, spike_ts=LAST - 5 * 3_600_000, vwap=1.0,
                                vs_vwap_pct=1.0, run_pct=10.0, text="held above 1 h · volume 63× < 100×")
                    await e._frenzy_lite_eval(db, "FOOUSDT", ep, flag, {}, bars, int(ep["last_bar_ts"]) + BAR_MS, norm_hour=NORM)
                    out.append((list(blocks), flag))
        finally:
            await eng.dispose()
        return out, opened, e
    (r1, r2), opened, e = asyncio.run(run())
    assert r1[0] == ["FRENZY_LITE_BEARISH_DAY"] and opened == [] and e._fz_lite_done["FOOUSDT"] == SID      # judged, then refused
    assert "LITE refused: bearish day (BTC day −0.80%, trend gap −0.05%)" in r1[1]["last_fire"]
    assert r2[0] == [] and r2[1]["lite_code"] == "FRENZY_LITE_STRETCH_DONE" and "bearish day" in r2[1]["text"]   # next bar: no retry


# ── config / exit / surfaces ───────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_values_d11():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert mf["frenzy_bearish_day_block"].default is False and mf["frenzy_max_atr_pct"].default == 2.5 and mf["frenzy_lock_arm_pct"].default == 0.0
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_bearish_day_block"] is True and cfg["frenzy_max_atr_pct"] == 3.0 and cfg["frenzy_lock_arm_pct"] == 0.0 and cfg["frenzy_tp_pct"] == 3.0
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 250" in src and "FRENZY_ATR_CAP_STUDY_2026-10-08.md" in src and "$10,254 → $11,055" in src


def test_exit_is_the_fixed_tp_for_every_sleeve_with_the_json():
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    th = NS(**cfg)
    assert frenzy_exit_for(3.0, 3.0, th, use_tp=True) == (True, "FRENZY_TP", 3.0)            # +3 closes (no lock)
    assert frenzy_exit_for(2.9, 2.9, th, use_tp=True)[0] is False
    assert frenzy_exit_for(-3.0, 1.0, th, use_tp=True)[:2] == (True, "STOP_LOSS")
    assert frenzy_exit_for(2.0, 3.4, th, use_tp=True) == (True, "FRENZY_TP", 3.0)            # a missed +3 peak closes at once, never rides down
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert 'FRENZY_STRATEGIES = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY")' in eng
    assert eng.count("in FRENZY_STRATEGIES:   # 🔥 its own stop + trailing exit (services.frenzy)") == 2          # candle + realtime paths
    assert eng.count("_rh_backstop_floor(getattr(self, 'is_paper_mode', True)), use_tp=True)") == 2            # both with use_tp (all 3 sleeves)
    assert eng.count('short=(order.direction == "SHORT"), use_tp=True)') == 1 and eng.count('short=(direction == "SHORT"), use_tp=True)') == 1   # manual FRENZY mode


def test_ui_load_save_and_both_exports():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fz-bearish-block"') == 1
    assert "getElementById('config-fz-bearish-block'); if (_e) _e.checked = config.thresholds.frenzy_bearish_day_block === true" in html
    assert "frenzy_bearish_day_block: document.getElementById('config-fz-bearish-block')?.checked ?? false" in html
    assert "bearish-day block ${_bt.frenzy_bearish_day_block === true ? 'ON" in html                           # the config line …
    assert html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2              # … rides BOTH exports
    assert "bearish-day block ${m.bearish_block ? 'ON' : 'OFF'}" in html                                       # the monitor line (dashboard + both exports)
    assert "['config-fz-max-atr', 'frenzy_max_atr_pct', 3.0]" in html and "['config-fz-lock-arm', 'frenzy_lock_arm_pct', 0]" in html
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '"bearish_block": bool(getattr(_th, \'frenzy_bearish_day_block\', False))' in main


def test_scouts_wired():
    rg = open(os.path.join(ROOT, "scripts", "scout_revert_gates.py"), encoding="utf-8").read()
    fx = open(os.path.join(ROOT, "scripts", "scout_frenzy_exits.py"), encoding="utf-8").read()
    assert '"FRENZY_OCT8": ("grep:(DECISION_LOG 250)"' in rg and all(f'"{c}"' in rg for c in ("BEARISH_BLOCKED", "ATR_RAISE", "TP3_VS_LOCK"))
    assert "def bb_run(" in fx and "out += bb_run(now_ms, th, Je, _fills((\"FRENZY_LONG\", \"FRENZY_WIDE\", \"FRENZY_LITE\")))" in fx
    assert re.search(r"BB_GATES = \{\"FRENZY_BEARISH_DAY\": \"LONG\", \"FRENZY_WIDE_BEARISH_DAY\": \"WIDE\", \"FRENZY_LITE_BEARISH_DAY\": \"LITE\"\}", fx)
