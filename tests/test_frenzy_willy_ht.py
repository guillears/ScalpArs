"""🎲¼ FRENZY_WILLY_HT (DECISION_LOG 341, operator option B): a QUARTER-size WILLY for the setups the turnover filter ALONE refuses.
Entry = WILLY's (red candle after the trigger) only when the full-size WILLY was refused SOLELY by a measured turnover ratio · ¼ size at
WILLY's leverage · WILLY's exit · its OWN slot (pre-check + hard guard in open_position) · never a WILLY for the WILLY slot / hard guard /
global hold · alert-only gates · its own reduced-size line in the analysis scripts. No network (exchange reads stubbed)."""
import asyncio
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_frenzy_willy import BAR_OPEN, LAST, _DB, _th as _wth  # noqa: E402
from tests.test_willy_global_hold_switch import _hold_db, _offline  # noqa: E402


# ── entry: only on a turnover-ONLY refusal ─────────────────────────────────────────────────────────────────────────────────────────

def _eng(TE, R=2.0, held=False, ht_full=False, ht_result=True, ht_marker=None):
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    e.blocks, e.opens, e.events = [], [], []
    e._record_filter_block = lambda name, d, had_room=True: e.blocks.append(name)
    e._willy_event = lambda gate, pair, reason=None, price=None, tally=True: e.events.append(gate)
    e._fz_judged_loaded = True
    e._willy_turnover = lambda pair, vol: R

    async def fake_open(db, fl, ind, bar_open, wide=False, catchup_now=None, lite=False, willy=None, willy_note=None, willy_wait_bars=None,
                        lite_retry=None, willy_ht=False):
        e.opens.append(dict(willy=willy, ht=willy_ht, note=willy_note))
        if willy_ht and not ht_result and ht_marker:
            fl[ht_marker] = True                      # what the real open path sets on a hold / pair-held / slot / crash refusal
        return ht_result if willy_ht else True
    e._frenzy_open = fake_open

    async def mark(pair, sp):
        e.__dict__.setdefault('_fz_willy_seen', {})[pair] = sp
    e._frenzy_willy_mark = mark

    async def save():
        pass
    e._frenzy_willy_save = save

    async def opened(db, pair, bar_open):
        return False
    e._frenzy_willy_sleeve_opened = opened

    async def done(db, pair, sp, trig):
        return False
    e._frenzy_willy_done_db = done

    async def hold_state(db=None):
        return (True, 9, "ZZZUSDT", "OPEN") if held else (False, None, None, None)
    e._willy_hold_state = hold_state

    async def htf(db):
        return ht_full
    e._willy_ht_slots_full = htf
    return e


def _run(e, TE, bar_open=BAR_OPEN, prev=None):
    del e.opens[:]
    ep = dict(spike_ts=LAST, bar_ret_pct=-0.2)
    flag = dict(ep, pair="FOOUSDT", volume_24h=5e7, price=1.0)
    asyncio.run(e._frenzy_willy_eval(_DB(), "FOOUSDT", ep, flag, {}, bar_open, prev, True,
                                     dict(on=True, ready=False, code="FRENZY_ATR_HIGH", text="x"), pair_warm=True))
    return list(e.opens), flag


def _cfg(monkeypatch, TE, **kw):
    base = dict(frenzy_willy_max_vol_mcap_ratio=1.0, frenzy_willy_ht_enabled=True, frenzy_willy_ht_max_slots=1)
    base.update(kw)
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=_wth(**base)))


def test_turnover_only_refusal_opens_ht_and_ends_the_pending(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _eng(TE, R=2.0)
    o, f = _run(e, TE)
    assert o == [dict(willy="A", ht=True, note=o[0]["note"])] and "turnover 2.00× mcap → quarter size" in o[0]["note"]
    assert "FRENZY_WILLY_TURNOVER" in e.blocks                                                    # the turnover shadow record is unchanged
    assert "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {}) and "FRENZY_WILLY_HT_TAKEN" in e.events


def test_below_the_cap_opens_the_full_willy_not_ht(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _eng(TE, R=0.5)
    o, _ = _run(e, TE)
    assert [(x["willy"], x["ht"]) for x in o] == [("A", False)]


def test_no_ht_on_any_other_refusal(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    for e in (_eng(TE, R=None),                       # unreadable turnover = not "refused solely by a measured ratio"
              _eng(TE, R=2.0, held=True)):            # the hold / WILLY slot / pair-held check refused first
        o, _ = _run(e, TE)
        assert not any(x["ht"] for x in o), o
    _cfg(monkeypatch, TE, frenzy_willy_ht_enabled=False)
    e = _eng(TE, R=2.0)
    o, _ = _run(e, TE)
    assert o == [] and "FOOUSDT" in e._fz_willy_pending                                            # switch off: WILLY stays armed as before


def test_ht_slot_full_stays_armed_counted_once_then_opens(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _eng(TE, R=2.0, ht_full=True)
    o, f = _run(e, TE)
    assert o == [] and "FOOUSDT" in e._fz_willy_pending and "HT slot full" in f["willy_ht_last"]
    o, _ = _run(e, TE, bar_open=BAR_OPEN + 300_000, prev=LAST)
    assert o == [] and e.blocks.count("FRENZY_WILLY_HT_MAX_SLOTS") == 1                           # once per pending entry

    async def free(db):
        return False
    e._willy_ht_slots_full = free
    o, _ = _run(e, TE, bar_open=BAR_OPEN + 600_000, prev=LAST)
    assert [x["ht"] for x in o] == [True] and "FOOUSDT" not in e._fz_willy_pending


def test_ht_refusal_semantics_mirror_willy(monkeypatch):
    """(341 review) HT's own lateness / dislocation / bare open_position refusal ENDS the pending entry (the full WILLY's rule); only a
    hold, pair held, HT slot full or our own crash keep it armed. The open path's markers never leak into WILLY's handling."""
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)
    e = _eng(TE, R=2.0, ht_result=False)                                                           # late / disloc / bare refusal
    o, f = _run(e, TE)
    assert [x["ht"] for x in o] == [True] and "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {})
    for mk in ("_willy_hold", "_willy_pair_held", "_willy_slots_full", "_willy_open_failed"):
        e = _eng(TE, R=2.0, ht_result=False, ht_marker=mk)
        o, f = _run(e, TE)
        assert [x["ht"] for x in o] == [True] and "FOOUSDT" in e._fz_willy_pending, mk
        assert not any(k in f for k in ("_willy_hold", "_willy_pair_held", "_willy_slots_full", "_willy_open_failed")), mk


# ── open_position: ¼ size, own slot, hard guards, hold ─────────────────────────────────────────────────────────────────────────────

def _open(monkeypatch, pair, open_rows=(), hold_on=False, tiers=None, **kw):
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    _offline(monkeypatch, TE)
    monkeypatch.setattr(TE.TradingEngine, "_willy_global_hold_on", lambda self: hold_on)
    if tiers is not None:
        async def _brk(*a, **k):
            return {pair: list(tiers)}
        monkeypatch.setattr(TE.binance_service, "get_leverage_brackets", _brk)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        async with SL() as s_:
            for p, es in open_rows:
                s_.add(models.Order(pair=p, direction="LONG", status="OPEN", is_paper=True, entry_strategy=es, confidence="STRONG_BUY",
                                    entry_price=1.0, quantity=1.0, investment=1.0, leverage=10, notional_value=10.0, opened_at=dt.datetime(2026, 10, 21)))
            await s_.commit()
        e = TE.TradingEngine(); e.is_running = True; e.is_paper_mode = True
        blocks = []
        _orig = e._record_filter_block
        e._record_filter_block = lambda n, d, had_room=True: (blocks.append(n), _orig(n, d, had_room))

        async def bal(db):
            return 5000.0
        e.get_available_balance = bal
        async with SL() as db:
            r = await e.open_position(db=db, pair=pair, direction="LONG", confidence="STRONG_BUY", current_price=1.0,
                                      entry_frenzy_willy_trigger="A", **kw)
        async with SL() as s_:
            rows = [(o.pair, o.entry_strategy, o.investment, o.leverage, o.notional_value, o.entry_frenzy_willy_trigger, o.entry_willy_ht_size_frac)
                    for o in (await s_.execute(select(models.Order).where(models.Order.status == "OPEN"))).scalars().all()]
        await eng.dispose()
        return r, blocks, rows
    return asyncio.run(go())


def test_ht_is_a_quarter_of_willy_at_the_same_leverage(monkeypatch):
    r1, _, w = _open(monkeypatch, "FOOUSDT", frenzy_willy=True)
    r2, _, h = _open(monkeypatch, "FOOUSDT", frenzy_willy=True, frenzy_willy_ht=True)
    assert r1 is not None and r2 is not None
    (wp, wes, winv, wlev, wnot, wtr, wfr), (hp, hes, hinv, hlev, hnot, htr, hfr) = w[0], h[0]
    assert wes == "FRENZY_WILLY" and hes == "FRENZY_WILLY_HT" and htr == "A"
    assert wfr is None and hfr == 0.25                                                                  # 📈 (365) the HT size stamp
    assert hlev == wlev and abs(hinv / winv - 0.25) < 1e-6 and abs(hnot / wnot - 0.25) < 1e-3


def test_ht_own_slot_hard_guard_and_willy_guards_ignore_ht(monkeypatch):
    r, blocks, rows = _open(monkeypatch, "FOOUSDT", open_rows=[("BARUSDT", "FRENZY_WILLY_HT")], frenzy_willy=True, frenzy_willy_ht=True)
    assert r is None and "FRENZY_WILLY_HT_MAX_SLOTS" in blocks and len(rows) == 1                   # HT slot full → refused (hard guard)
    r, blocks, rows = _open(monkeypatch, "FOOUSDT", open_rows=[("BARUSDT", "FRENZY_WILLY_HT")], frenzy_willy=True)
    assert r is not None and "FRENZY_WILLY_MAX_SLOTS" not in blocks                                   # an open HT never fills WILLY's slot
    r, blocks, rows = _open(monkeypatch, "FOOUSDT", open_rows=[("BARUSDT", "FRENZY_WILLY")], frenzy_willy=True, frenzy_willy_ht=True)
    assert r is not None and "FRENZY_WILLY_HT_MAX_SLOTS" not in blocks                                # an open WILLY never fills HT's slot
    r, blocks, rows = _open(monkeypatch, "BARUSDT", open_rows=[("BARUSDT", "FRENZY_WILLY")], frenzy_willy=True, frenzy_willy_ht=True)
    assert r is None and "PAIR_HELD" in blocks                                                         # never on a held pair


def test_hold_generic_rule_and_ht_is_not_a_hold_trigger(monkeypatch):
    r, blocks, _ = _open(monkeypatch, "FOOUSDT", open_rows=[("BARUSDT", "FRENZY_WILLY")], hold_on=True, frenzy_willy=True, frenzy_willy_ht=True)
    assert r is None and "FRENZY_WILLY_HOLD" in blocks                                                 # hold ON: HT is a blocked sleeve
    r, blocks, _ = _open(monkeypatch, "FOOUSDT", open_rows=[("BARUSDT", "FRENZY_WILLY_HT")], hold_on=True)
    assert r is not None and "FRENZY_WILLY_HOLD" not in blocks                                         # an open HT never holds the book


# ── exit = WILLY's ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_exit_resting_tp_stop_and_time_are_willys():
    import config as C
    import services.trading_engine as TE
    from services.resting_tp import rtp_target, rtp_enabled, RTP_SWITCHES, rtp_reason_for_limit
    from services.frenzy import frenzy_willy_levels
    th = C.trading_config.thresholds
    assert rtp_enabled(th, "FRENZY_WILLY_HT") == rtp_enabled(th, "FRENZY_WILLY")
    for m in (5, 45):
        assert rtp_target(th, "FRENZY_WILLY_HT", m) == rtp_target(th, "FRENZY_WILLY", m)               # TP then the late TP @30
    assert len(RTP_SWITCHES) == len(set(RTP_SWITCHES))
    assert TE.bot_own_stop_pct(th, "FRENZY_WILLY_HT") == TE.bot_own_stop_pct(th, "FRENZY_WILLY") == abs(frenzy_willy_levels(th)[1])   # backstop
    assert rtp_reason_for_limit(th, "FRENZY_WILLY_HT", "LONG", 1.0, 0.05, 1.0, 0.02) == rtp_reason_for_limit(th, "FRENZY_WILLY", "LONG", 1.0, 0.05, 1.0, 0.02)
    assert "FRENZY_WILLY_HT" in TE.FRENZY_STRATEGIES and TE._ema13_cross_exit_applies("FRENZY_WILLY_HT") is False
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    for s_ in ('elif (order.entry_strategy or "") in WILLY_EXIT_CLASS:   # 🎲¼ (341) + HT · 🎲 Oct-8 (251): its OWN time cap',
               'elif (order.entry_strategy or "") in WILLY_EXIT_CLASS:   # 🎲¼ (341) + HT · 🎲 Oct-8 (251): its OWN exit',
               "elif (order_info.get('entry_strategy') or '') in WILLY_EXIT_CLASS:",
               "if st.get('es') in WILLY_EXIT_CLASS and not st.get('why')",                               # late-TP limit swap
               "'stop_loss': (frenzy_willy_levels(config.trading_config.thresholds)[1] if (_frenzy and _fz_es in WILLY_EXIT_CLASS)",
               "if (order.entry_strategy or '') in WILLY_EXIT_CLASS   # 🎲 None = no WILLY stop",
               "if (_rtp_es in WILLY_EXIT_CLASS or"):
        assert s_ in eng, s_
    # WILLY-only sites stay WILLY-only (slot, hard guard, hold, hold flush)
    for s_ in ('Order.entry_strategy == "FRENZY_WILLY",\n', 'if _wh_es == "FRENZY_WILLY":', "if (_oi.get('entry_strategy') or '') == \"FRENZY_WILLY\":"):
        assert s_ in eng, s_


def test_ht_exit_closes_like_willy():
    """the realtime exit branch WILLY rows run is the one HT rows run: the same function, the same inputs."""
    import config as C
    from services.frenzy import frenzy_willy_exit_for, frenzy_willy_levels
    th = C.trading_config.thresholds
    tp, sl, mh = frenzy_willy_levels(th)
    assert frenzy_willy_exit_for(sl - 0.1, 0.0, th, 5, None, resting_tp=False)[0] is True        # the −4 stop
    assert frenzy_willy_exit_for(0.1, 0.2, th, mh + 1, None, resting_tp=False)[0] is True        # the time exit


# ── config / UI / exports / analysis ───────────────────────────────────────────────────────────────────────────────────────────────

def test_config_json_ui_and_exports():
    import config as C
    mf = type(C.trading_config.thresholds).model_fields
    assert mf["frenzy_willy_ht_enabled"].default is True and mf["frenzy_willy_ht_size_frac"].default == 0.25 and mf["frenzy_willy_ht_max_slots"].default == 1
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_willy_ht_enabled"] is True and cfg["frenzy_willy_ht_size_frac"] == 0.25 and cfg["frenzy_willy_ht_max_slots"] == 1
    assert "DECISION_LOG 341" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for _id in ("config-fz-willy-ht-enabled", "config-fz-willy-ht-size-frac", "config-fz-willy-ht-slots"):
        assert html.count(f'id="{_id}"') == 1, _id
    assert "['config-fz-willy-ht-size-frac', 'frenzy_willy_ht_size_frac', 0.25]" in html and "['config-fz-willy-ht-slots', 'frenzy_willy_ht_max_slots', 1]" in html
    assert "(_key === 'frenzy_willy_ht_max_slots' || _key === 'frenzy_max_slots'" in html                 # an int on save
    assert "getElementById('config-fz-willy-ht-enabled'); if (_e) _e.checked = config.thresholds.frenzy_willy_ht_enabled !== false;" in html
    assert "frenzy_willy_ht_enabled: document.getElementById('config-fz-willy-ht-enabled')?.checked ?? true," in html
    assert "🎲¼ WILLY_HT ${m.willy_ht_enabled === false ? 'OFF' : 'ON'}" in html                         # monitor line → dashboard + both exports
    assert "🎲¼ WILLY_HT ${_bt.frenzy_willy_ht_enabled === false ? 'OFF' : 'ON'}" in html                 # config block → both exports
    assert html.count("lines.push(...frenzyReportLines(perf, hr2));") == 2 and html.count("lines.push(..._buildConfigLines(cfg, changelog, hr, hr2, status));") == 2
    assert '<option value="FRENZY_WILLY_HT">' in html and "['FRENZY_WILLY', 'FRENZY_WILLY_HT'].includes(o.entry_strategy || '')" in html
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '("FRENZY_WILLY_HT", "FRENZY_WILLY_HT (¼ size' in main and "return 'Frenzy-Willy-HT'" in main   # FRENZY table row (UI + exports) · strategy row
    assert '("FRENZY_WILLY_HT", "LONG"): "FRENZY WILLY HT Long (¼)"' in main and '"willy_ht_enabled": bool(' in main


def test_monitor_payload_runs():
    import main as M
    p = M._frenzy_monitor_payload()
    assert p.get("willy_ht_enabled") is True and p.get("willy_ht_size_frac") == 0.25 and p.get("willy_ht_max_slots") == 1


def test_size_frac_reader_and_gate():
    from services.frenzy import frenzy_willy_ht_frac, frenzy_willy_ht_gate
    for v, want in ((0.25, 0.25), (0.5, 0.5), (0, 0.25), (-1, 0.25), (1.5, 0.25), ("x", 0.25), (None, 0.25)):
        assert frenzy_willy_ht_frac(NS(frenzy_willy_ht_size_frac=v)) == want, v
    assert frenzy_willy_ht_frac(NS()) == 0.25
    d0 = dt.datetime(2026, 10, 22)
    day = lambda k: d0 + dt.timedelta(days=k // 3, hours=k % 3)
    assert frenzy_willy_ht_gate([], [])[0] == "COLLECTING"
    assert frenzy_willy_ht_gate([(day(i), -2.0) for i in range(4)], [])[0] == "KILL"                       # Σ −8
    assert frenzy_willy_ht_gate([(day(i), -0.5) for i in range(10)], [])[0] == "FIRST"                     # Σ −5, first 10 avg < 0
    ht = [(day(i), 0.3) for i in range(30)]                                                                  # 30 fills on 10 days
    assert frenzy_willy_ht_gate(ht, [(day(i), 0.2) for i in range(30)])[0] == "REMOVE_FILTER"
    assert frenzy_willy_ht_gate(ht, [(day(i), 0.6) for i in range(30)])[0] == "CONFIRMED"
    assert frenzy_willy_ht_gate(ht, [(day(i), 0.4) for i in range(30)])[0] == "REREAD"
    assert frenzy_willy_ht_gate([(day(i), 0.3) for i in range(29)], [])[0] == "COLLECTING"
    ht60 = [(day(i), 0.3) for i in range(60)]
    assert frenzy_willy_ht_gate(ht60, [(day(i), 0.4) for i in range(60)])[0] == "INCONCLUSIVE"


def test_analysis_scripts_keep_ht_off_the_full_size_lines():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import archive_batch as AB
    d = pd.DataFrame(dict(entry_strategy=["FRENZY_WILLY", "FRENZY_WILLY_HT", "MOMENTUM"], cell_multiplier_source=["", "", ""],
                          pattern_cell_source=["", "", "UNMATCHED"]))
    assert list(AB._probe(d)) == [False, True, False]
    src = open(os.path.join(ROOT, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    assert "df['is_probe'] = df['is_probe'] | (df.entry_strategy.astype(str) == 'FRENZY_WILLY_HT')" in src
    import screen_pool as SP
    assert SP.sleeve(dict(status="CLOSED", pair="FOOUSDT", entry_strategy="FRENZY_WILLY_HT", direction="LONG", pattern_cell_source="UNMATCHED")) is None


def test_scout_tracker(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_willy_ht as S
    rows = []
    for i in range(12):
        rows.append(dict(opened_at=f"2026-10-2{2 + i % 3}T{10 + i:02d}:00:00", closed_at=f"2026-10-2{2 + i % 3}T{10 + i:02d}:30:00", pair=f"H{i}USDT",
                         direction="LONG", status="CLOSED", entry_strategy="FRENZY_WILLY_HT", pnl=-1.0, pnl_percentage=-0.5,
                         close_reason=("MANUAL" if i == 0 else "STOP_LOSS"), is_paper=False))
    rows.append(dict(opened_at="2026-10-22T09:00:00", closed_at="2026-10-22T09:30:00", pair="WUSDT", direction="LONG", status="CLOSED",
                     entry_strategy="FRENZY_WILLY", pnl=5.0, pnl_percentage=1.25, close_reason="FRENZY_TP", is_paper=False))
    p = tmp_path / "scalpars_orders_live_x.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    since = int(pd.Timestamp("2026-10-22T00:00:00").value // 1_000_000)
    txt = "\n".join(S.run(paths=[str(p)], since_ms=since))
    assert "FRENZY_WILLY_HT (¼): N 11" in txt and "operator-closed: 1" in txt and "Gate verdict: FIRST" in txt
    import scout_gatekit as GK
    orig = GK.dl_deploy_ms
    try:
        GK.dl_deploy_ms = lambda *a, **k: None
        assert "NOT EVALUABLE" in "\n".join(S.run(paths=[str(p)]))
    finally:
        GK.dl_deploy_ms = orig
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_ht as _wht" in src and "_rg_sec += _wht.run(now_ms)" in src


def test_api_performance_has_the_ht_row_and_rows_sum_to_total():
    """(341 review) the sleeve table's fixed order list once dropped Frenzy-Willy-HT — the row vanished from the UI and both exports."""
    import models
    import main as M
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        paper = M.trading_engine.is_paper_mode
        async with SL() as s:
            for i, (es, p) in enumerate([("FRENZY_WILLY", 1.2), ("FRENZY_WILLY_HT", -0.5), ("FRENZY_WILLY_HT", 1.2), ("MOMENTUM", 0.4)]):
                t = dt.datetime(2026, 10, 22, 10 + i)
                s.add(models.Order(pair=f"P{i}USDT", direction="LONG", status="CLOSED", is_paper=paper, entry_strategy=es, confidence="STRONG_BUY",
                                   entry_price=1.0, exit_price=1.0 + p / 100, quantity=100.0, investment=10.0, leverage=10, notional_value=100.0,
                                   opened_at=t, closed_at=t + dt.timedelta(minutes=30), pnl=p, pnl_percentage=p, close_reason="FRENZY_TP"))
            await s.commit()
        async with SL() as s:
            r = await M.get_performance(db=s)
        await eng.dispose()
        return r if isinstance(r, dict) else json.loads(r.body)
    d = asyncio.run(go())
    rows = {r["sleeve"]: r for r in d["sleeve_performance"]}
    assert rows["Frenzy-Willy-HT"]["n"] == 2 and rows["Frenzy-Willy"]["n"] == 1
    parts = [r for k, r in rows.items() if k != "Total"]
    assert sum(r["n"] for r in parts) == rows["Total"]["n"] == 4
    assert abs(sum(r["net_usd"] for r in parts) - rows["Total"]["net_usd"]) < 0.01


def test_bracket_cap_never_restores_ht_to_a_full_margin(monkeypatch):
    """(341 review) DL 298's bracket keep (full margin at a lower leverage) must never apply to HT: when the brackets bind, HT's margin
    stays ≤ ¼ of WILLY's and its leverage ≤ WILLY's."""
    _, _, w = _open(monkeypatch, "FOOUSDT", frenzy_willy=True)
    winv, wlev, wnot = w[0][2], w[0][3], w[0][4]
    cap = 0.6 * wnot / 4 / 0.95                                   # the HT notional does not fit at WILLY's leverage (5 % headroom undone)
    tiers = [(cap, wlev), (wnot * 10, 2)]                          # a lower tier that WOULD carry the reference notional with a full margin
    r, blocks, h = _open(monkeypatch, "FOOUSDT", tiers=tiers, frenzy_willy=True, frenzy_willy_ht=True)
    assert r is not None, blocks
    hinv, hlev = h[0][2], h[0][3]
    assert hinv <= winv / 4 + 1e-6 and hlev <= wlev


def test_gate_review_fixes_and_config_validation():
    from services.frenzy import frenzy_willy_ht_gate as g, frenzy_willy_ht_config_errors as ce
    d0 = dt.datetime(2026, 10, 22)
    day = lambda k: d0 + dt.timedelta(days=k // 3, hours=k % 3)
    v, t = g([(day(i), -2.0) for i in range(4)], [])
    assert v == "KILL" and "main read: ⏳ 4/30" in t                                                   # the main status is never hidden
    v, t = g([(day(i), 0.3) for i in range(30)], [(day(i), 0.2) for i in range(9)])
    assert v == "REREAD" and "comparator too thin (9 < 10" in t                                       # ≥ 10 WILLY comparator fills
    v, t = g([(day(i), 0.3) for i in range(59)], [(day(i), 0.4) for i in range(59)])
    assert v == "REREAD" and "(59/60)" in t                                                            # the re-read needs ≥ 60 HT fills
    assert ce(NS(frenzy_willy_ht_enabled=True, frenzy_willy_ht_size_frac=0.25, frenzy_willy_ht_max_slots=1)) == []
    for bad in (0, -0.1, 1.5, None, "x"):
        assert ce(NS(frenzy_willy_ht_enabled=True, frenzy_willy_ht_size_frac=bad, frenzy_willy_ht_max_slots=1)), bad
    assert ce(NS(frenzy_willy_ht_enabled=True, frenzy_willy_ht_size_frac=0.25, frenzy_willy_ht_max_slots=0))
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "if (!(Number.isFinite(_hx) && _hx > 0 && _hx <= 1)) { _hf.value = '0.25'; showToast(" in html
    assert html.count('data-ss="fzyht"') == 1 and "S.fzyht = { info:" in html
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_cfg_errs.extend(_ht_cfg_errors(new_config.thresholds))" in main
    assert "'Frenzy-Willy', 'Frenzy-Willy-HT', 'Flatbase-Short', 'Manual']" in main   # 🩳 (348)


# ─────────────────────────── 📈 (DECISION_LOG 365) the HT re-size ladder in scripts/scout_willy_ht.py ───────────────────────────
def _ladder_frame(ht_rows, w_rows):
    """ht_rows / w_rows: lists of (day index, pnl %, size frac or None); 2 fills a day max; days from 2026-11-01."""
    import pandas as pd
    rows = []
    for es, src in (("FRENZY_WILLY_HT", ht_rows), ("FRENZY_WILLY", w_rows)):
        for k, (d, p, f) in enumerate(src):
            rows.append(dict(o=pd.Timestamp("2026-11-01") + pd.Timedelta(days=d, minutes=k), status="CLOSED", es=es, entry_strategy=es,
                             pct=p, close_reason="FRENZY_TP", entry_willy_ht_size_frac=f))
    return pd.DataFrame(rows)


def test_ht_resize_ladder():
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import scout_willy_ht as H
    import pandas as pd
    V2B = int(pd.Timestamp("2026-10-30").value // 1_000_000)
    STAMP = int(pd.Timestamp("2026-10-31").value // 1_000_000)
    W = [(d, 1.0, None) for d in range(12) for _ in range(2)]                  # 24 full-size WILLY fills at +1.0
    j = lambda ht, frac=0.25, w=W: " ".join(H.resize_lines(_ladder_frame(ht, w), V2B, STAMP, frac_now=frac))
    assert H.LADDER == (0.25, 0.5, 1.0) and H.STEP_GAP == 0.10 and H.PRE_STAMP_FRAC == 0.25
    t = j([(d, 0.8, 0.25) for d in range(5)])
    assert "now 0.25 · current run at this size: 5 fills on 5 days" in t and "next step 0.25 → 0.5" in t and "CANDIDATE" not in t
    t = j([(d % 10, 0.95, 0.25) for d in range(30)])                           # 30 on 10 days: +0.95 vs WILLY +1.0 → within 0.10
    assert "✅ STEP-UP candidate 0.25 → 0.5" in t and "REREAD" in t
    t = j([(d % 10, 0.85, 0.25) for d in range(30)])                           # +0.85 vs +1.0: between −0.20 and −0.10 → no step
    assert "no step: frozen 30 HT avg +0.850 %" in t
    assert "full size by the DL 341 rule" in j([(d % 10, 1.2, 0.25) for d in range(30)])                    # HT ≥ WILLY
    assert "🛑 STEP-DOWN candidate to 0.25" in j([(d % 10, 0.2, 0.5) for d in range(30)], 0.5)            # CONFIRMED at 0.5
    assert "DL 341 rule applies" in j([(d % 10, 0.2, 0.25) for d in range(30)])                            # CONFIRMED at 0.25
    assert "🛑 STEP-DOWN" in j([(d, -1.0, 0.5) for d in range(10)], 0.5)                                    # first 10 avg < 0
    # unstamped: before the stamp deploy = 0.25 · after it = unknown, counted aloud
    pre = _ladder_frame([(-2, 0.8, None)], W)                                                              # 10-30 ≥ V2B, < STAMP
    assert "current run at this size: 1 fills" in " ".join(H.resize_lines(pre, V2B, STAMP, frac_now=0.25))
    post = _ladder_frame([(3, 0.8, None)], W)
    assert "1 HT fill(s) with unknown size not counted" in " ".join(H.resize_lines(post, V2B, STAMP, frac_now=0.25))
    # only the current run at a size; fills before the V2b deploy never count
    run = [(0, 1.0, 0.5), (1, 1.0, 0.25), (2, 1.0, 0.5), (3, 1.0, 0.5)]
    assert "current run at this size: 2 fills on 2 days" in j(run, 0.5)
    old = _ladder_frame([(-5, 1.0, 0.25)], W)
    assert "current run at this size: 0 fills" in " ".join(H.resize_lines(old, V2B, STAMP, frac_now=0.25))
    assert "OFF the ladder" in j([(0, 1.0, 0.3)], 0.3) and "sync the config" in j([(0, 1.0, 0.5)], 0.25)
    assert "top of the ladder" in j([(0, 1.0, 1.0)], 1.0)
    assert "NOT evaluable" in " ".join(H.resize_lines(_ladder_frame([], W), None, None, frac_now=0.25))
    assert "no exports" in " ".join(H.resize_lines(None, V2B, STAMP, frac_now=0.25))                     # never raises
    assert "no exports" in " ".join(H.resize_lines(pd.DataFrame(), V2B, STAMP, frac_now=0.25))
    assert "ADVISORY" in j([(d % 10, 1.2, 0.25) for d in range(30)])                                      # HT ≥ WILLY + Pattern-W
    Wn = [(d, -0.1, None) for d in range(30) for _ in range(2)]
    sixty = [(d % 10, 1.0, 0.5) for d in range(30)] + [(10 + d % 10, -1.2, 0.5) for d in range(30)]       # frozen 30 fine, first 60 < 0
    assert "🛑 STEP-DOWN candidate to 0.25" in j(sixty, 0.5, Wn)
    assert "unreadable" in " ".join(H.resize_lines(_ladder_frame([(0, 1.0, 0.25)], W).drop(columns=["pct"]), V2B, STAMP, frac_now=0.25))
    assert (H.STEP_MIN_AVG, H.STEP_MIN_WR) == (0.10, 0.70)
    # the Pattern-W bar: avg +0.60 ≥ WILLY +0.5 − 0.10 but WR 60 % < 70 % → no step
    W5 = [(d, 0.5, None) for d in range(12) for _ in range(2)]
    mixed = [(d % 10, (2.0 if d % 5 < 3 else -1.5), 0.25) for d in range(30)]
    assert "no step: frozen 30 HT avg +0.600 % · WR 60%" in j(mixed, 0.25, [(d, 0.65, None) for d in range(12) for _ in range(2)])
    assert "the size bar fails" in j(mixed, 0.25, W5)                                                    # HT ≥ WILLY but WR 60 %
    # 0.5 → 1.0 also re-decides the HT slot cap
    assert "ALSO re-decide frenzy_willy_ht_max_slots" in j([(d % 10, 0.95, 0.5) for d in range(30)], 0.5)
    # KILL (Σ ≤ −8) at 0.5 → step down · fewer than 10 same-period WILLY comparators → no step yet
    assert "🛑 STEP-DOWN candidate to 0.25" in j([(0, -4.1, 0.5), (1, -4.1, 0.5)], 0.5)
    t = j([(d % 10, 0.95, 0.25) for d in range(30)], 0.25, [(d, 1.0, None) for d in range(3)])
    assert "CANDIDATE" not in t and "next step 0.25 → 0.5" in t
    # REMOVE_FILTER but HT itself loses (WILLY worse) → the size follows HT's expectancy: step down
    assert "🛑 STEP-DOWN candidate to 0.25" in j([(d % 10, -0.1, 0.5) for d in range(30)], 0.5, [(d, -0.5, None) for d in range(12) for _ in range(2)])
    # a NaN pct never enters the frozen sample (the gate drops it too)
    t = j([(d % 10, 0.95, 0.25) for d in range(30)] + [(10, float("nan"), 0.25)])
    assert "current run at this size: 30 fills" in t and "✅ STEP-UP" in t
