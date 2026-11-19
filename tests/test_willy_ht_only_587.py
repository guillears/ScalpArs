"""🎲 DECISION_LOG 587 (operator 11-18, PAPER, operator-declared) — plain FRENZY_WILLY entries OFF, FRENZY_WILLY_HT ON at 0.5 × 5×, HT take
profit +5 net (stop −4 unchanged). The WILLY machinery keeps running for HT; a setup that passes turnover is refused WILLY_PLAIN_OFF.
No network (exchange reads stubbed)."""
import json
import os
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tests.test_frenzy_willy_ht import _eng, _run, _cfg, _open  # noqa: E402


# ── ① the switch: plain WILLY refused, HT still opens ────────────────────────────────────────────────────────────────────────────────

def test_plain_off_refuses_a_setup_that_passes_turnover(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_plain_entries_enabled=False)
    e = _eng(TE, R=0.5)                                                                      # turnover 0.5 < 1.0: a plain WILLY setup
    o, flag = _run(e, TE)
    assert o == [] and e.blocks.count("WILLY_PLAIN_OFF") == 1 and "WILLY_PLAIN_OFF" in e.events
    assert "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {})                    # the entry ends (WILLY's one-shot rule)
    assert "plain WILLY entries OFF" in flag.get("willy_last", "")


def test_plain_off_still_opens_ht_on_a_turnover_refusal(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_plain_entries_enabled=False)
    e = _eng(TE, R=2.0)                                                                      # turnover 2.0 ≥ 1.0: the HT setup
    o, _ = _run(e, TE)
    assert [(x["willy"], x["ht"]) for x in o] == [("A", True)] and "WILLY_PLAIN_OFF" not in e.blocks


def test_plain_on_is_unchanged(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE)                                                                    # the field absent → plain entries ON (code default)
    e = _eng(TE, R=0.5)
    o, _ = _run(e, TE)
    assert [(x["willy"], x["ht"]) for x in o] == [("A", False)] and "WILLY_PLAIN_OFF" not in e.blocks
    from services.frenzy import frenzy_willy_plain_on
    assert frenzy_willy_plain_on(NS()) is True and frenzy_willy_plain_on(NS(frenzy_willy_plain_entries_enabled=False)) is False


# ── ② HT size = 0.5 × 5× (the EXHAUSTION / Flatbase size) ─────────────────────────────────────────────────────────────────────────────

def test_ht_size_is_half_margin_at_5x(monkeypatch):
    import config as C
    th = C.trading_config.thresholds
    assert (th.frenzy_willy_invest_mult, th.frenzy_willy_lev_mult, th.frenzy_willy_ht_size_frac) == (0.5, 0.25, 1.0)   # the live JSON
    r1, _, w = _open(monkeypatch, "FOOUSDT", frenzy_willy=True)
    r2, _, h = _open(monkeypatch, "FOOUSDT", frenzy_willy=True, frenzy_willy_ht=True)
    (_, wes, winv, wlev, wnot, _, _), (_, hes, hinv, hlev, hnot, _, hfr) = w[0], h[0]
    assert hes == "FRENZY_WILLY_HT" and hfr == 1.0 and hlev == wlev == 5                     # 20 × 0.25 = 5×
    assert hinv == pytest.approx(winv) and hnot == pytest.approx(wnot, rel=1e-3)             # HT = WILLY's full 0.5 × 5× ticket
    ratio = hnot / 5000.0                                                                    # the test account: 5,000 available, nothing else open
    assert ratio == pytest.approx(0.61, abs=0.02), ratio                                     # ≈ 0.61 × equity notional (the sleeve-set size)


# ── ③ HT take profit +5 net, stop −4: what the exit does ──────────────────────────────────────────────────────────────────────────────

def test_ht_tp5_target_and_the_exit_it_runs():
    from services.frenzy import (frenzy_willy_ht_tp, frenzy_willy_ht_target, frenzy_willy_exit_for, frenzy_willy_runner_k_for,
                                 frenzy_willy_ht_config_errors)
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    th = NS(**cfg)
    assert cfg["frenzy_willy_ht_tp_pct"] == 5.0 and cfg["frenzy_willy_stop_pct"] == 4.0 and frenzy_willy_ht_tp(th) == 5.0
    assert not frenzy_willy_ht_config_errors(th)
    assert frenzy_willy_ht_target(1.25, 5.0) == 5.0 and frenzy_willy_ht_target(6.8, 5.0) == 6.8   # a spike-high target above +5 (≤ +8) is kept
    row = dict(frenzy_willy_tp_target_pct=5.0)
    assert frenzy_willy_exit_for(5.0, 5.0, th, held_minutes=5, row=row)[:2] == (True, "FRENZY_TP")
    assert frenzy_willy_exit_for(-4.0, 0.0, th, held_minutes=5, row=row)[:2] == (True, "STOP_LOSS")
    # an HT row under the live JSON: no late TP, 24 h cap (the final DL 587 exit — section ⑤ covers it in full)
    hrow = dict(row, entry_strategy="FRENZY_WILLY_HT")
    assert frenzy_willy_exit_for(0.3, 0.3, th, held_minutes=31, row=hrow)[0] is False
    assert frenzy_willy_exit_for(-1.0, 1.0, th, held_minutes=1440, row=hrow)[:2] == (True, "MAX_HOLD_TIME")
    assert frenzy_willy_runner_k_for(th, "FRENZY_WILLY_HT", -1.0, 1.0) is None                # the runner never applies to HT
    from services.resting_tp import rtp_target
    assert rtp_target(th, "FRENZY_WILLY_HT", held_minutes=5, willy_tp=5.0)[0] == 5.0         # the resting limit rests at the row's +5
    assert rtp_target(th, "FRENZY_WILLY_HT", held_minutes=31, willy_tp=5.0) == (5.0, "FRENZY_TP")    # … and stays there (HT late TP off)


# ── ④ config / UI / exports / monitor / registry ─────────────────────────────────────────────────────────────────────────────────────

def test_config_json_ui_exports_and_registry():
    import config as C
    assert C.SignalThresholds.model_fields["frenzy_willy_plain_entries_enabled"].default is True
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_willy_plain_entries_enabled"] is True and cfg["frenzy_willy_enabled"] is True and cfg["frenzy_willy_ht_enabled"] is True   # plain WILLY back ON (DL 595)
    assert "DECISION_LOG 587" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fz-willy-plain-enabled"') == 1
    assert "config.thresholds.frenzy_willy_plain_entries_enabled !== false" in html
    assert "frenzy_willy_plain_entries_enabled: document.getElementById('config-fz-willy-plain-enabled')?.checked ?? true" in html
    assert "🎲 WILLY plain entries (DECISION_LOG 587)" in html and html.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2
    assert "m.willy_plain_enabled === false ? 'plain entries OFF (HT only, DL 587)'" in html
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '"willy_plain_enabled": getattr(_th, \'frenzy_willy_plain_entries_enabled\', True) is not False' in main
    import main as M
    assert M._sleeve_enabled(NS(frenzy_willy_enabled=True, frenzy_willy_plain_entries_enabled=False), "FRENZY_WILLY") is False
    assert M._sleeve_enabled(NS(frenzy_willy_enabled=True, frenzy_willy_ht_enabled=True, frenzy_willy_plain_entries_enabled=False), "FRENZY_WILLY_HT") is True
    from services.frenzy import frenzy_willy_ht_config_errors
    assert any("frenzy_willy_plain_entries_enabled" in x for x in frenzy_willy_ht_config_errors(NS(**dict(cfg, frenzy_willy_plain_entries_enabled="no"))))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import observe_registry as OR
    assert "scout_willy_ht_exit_lines.py" in OR.NONE and "scout_willy_ht_exit_lines.py" not in OR.EXIT       # retired (it priced the +2 exit)
    assert OR.BOOKED["scout_willy_ht_lines.py"]["frenzy_willy_ht_tp_pct"] == 5.0                              # re-based
    row = [r for r in OR.status() if r[0] == "scout_willy_ht_lines.py"][0]
    assert row[4] == "OK", row[5]
    assert "import scout_willy_ht_exit_lines" not in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    import scout_willy_ht_lines as W
    assert W.BE_WR["FRENZY_WILLY_HT"] == pytest.approx(4 / 9, abs=1e-3) and W.DL_HT == 587 and W.STATE_KEY["FRENZY_WILLY_HT"].endswith("@587")


# ── ⑤ the FINAL HT exit (operator 11-18): flat TP +5 · stop −4 · no late TP · 24 h — WILLY unchanged ────────────────────────────────

def _live_th(**kw):
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    cfg.update(kw)
    return NS(**cfg)


def test_ht_exit_is_a_plain_tp5_sl4_24h_and_willy_keeps_its_exit():
    from services.frenzy import frenzy_willy_exit_for, frenzy_willy_levels, frenzy_willy_late_tp
    from services.resting_tp import rtp_target, rtp_reason_for_limit
    th = _live_th()
    assert (th.frenzy_willy_ht_late_tp_enabled, th.frenzy_willy_ht_high_target_enabled, th.frenzy_willy_ht_max_hold_minutes) == (False, False, 1440.0)
    ht = dict(entry_strategy="FRENZY_WILLY_HT", frenzy_willy_tp_target_pct=5.0)
    wl = dict(entry_strategy="FRENZY_WILLY", frenzy_willy_tp_target_pct=5.0)
    assert frenzy_willy_levels(th, "FRENZY_WILLY_HT")[2] == 1440 and frenzy_willy_levels(th, "FRENZY_WILLY")[2] == 90
    assert frenzy_willy_late_tp(th, "FRENZY_WILLY_HT") is None and frenzy_willy_late_tp(th, "FRENZY_WILLY") == (30.0, 0.25)
    assert frenzy_willy_exit_for(0.3, 0.3, th, held_minutes=31, row=ht)[0] is False                  # HT: no late TP
    assert frenzy_willy_exit_for(-1.0, 1.0, th, held_minutes=600, row=ht)[0] is False                # HT: still open after 90 min
    assert frenzy_willy_exit_for(-1.0, 1.0, th, held_minutes=1440, row=ht)[:2] == (True, "MAX_HOLD_TIME")
    assert frenzy_willy_exit_for(5.0, 5.0, th, held_minutes=600, row=ht)[:2] == (True, "FRENZY_TP")
    assert frenzy_willy_exit_for(-4.0, 0.0, th, held_minutes=600, row=ht)[:2] == (True, "STOP_LOSS")
    assert frenzy_willy_exit_for(0.3, 0.3, th, held_minutes=31, row=wl)[:2] == (True, "WILLY_LATE_TP")   # plain WILLY (if re-enabled): unchanged
    assert frenzy_willy_exit_for(-1.0, 1.0, th, held_minutes=90, row=wl)[:2] == (True, "MAX_HOLD_TIME")
    assert rtp_target(th, "FRENZY_WILLY_HT", held_minutes=600, willy_tp=5.0) == (5.0, "FRENZY_TP")    # the resting limit never re-prices
    assert rtp_target(th, "FRENZY_WILLY", held_minutes=31, willy_tp=5.0)[1] == "WILLY_LATE_TP"
    assert rtp_reason_for_limit(th, "FRENZY_WILLY_HT", "LONG", 1.0, 0.05, 1.003, 0.00018, willy_tp=5.0) == "FRENZY_TP"
    on = _live_th(frenzy_willy_ht_late_tp_enabled=True, frenzy_willy_ht_max_hold_minutes=0)          # the code defaults = today's behaviour
    assert frenzy_willy_exit_for(0.3, 0.3, on, held_minutes=31, row=ht)[:2] == (True, "WILLY_LATE_TP") and frenzy_willy_levels(on, "FRENZY_WILLY_HT")[2] == 90


@pytest.mark.parametrize("high_on,want", [(False, 5.0), (True, 8.0)])
def test_open_path_stores_the_flat_target_and_rests_the_limit_there(monkeypatch, high_on, want):
    import config as C
    monkeypatch.setattr(C.trading_config.thresholds, "frenzy_willy_ht_high_target_enabled", high_on, raising=False)
    r, _, _ = _open(monkeypatch, "FOOUSDT", frenzy_willy=True, frenzy_willy_ht=True, entry_frenzy_willy_spike_high=1.5)   # spike high +50 %
    assert r is not None and r.entry_strategy == "FRENZY_WILLY_HT" and r.frenzy_willy_tp_target_pct == pytest.approx(want)
    if getattr(r, "resting_tp_price", None):
        assert r.resting_tp_price > r.entry_price * (1 + want / 100.0 * 0.95)                          # the maker limit rests at the stored target


def test_cache_rebuild_keeps_the_ht_exit(monkeypatch):
    import asyncio
    import datetime as dt
    import models
    import services.trading_engine as TE
    from tests.test_frenzy_willy import _hold_db
    from services.frenzy import frenzy_willy_exit_for
    for k, v in vars(_live_th()).items():
        monkeypatch.setattr(TE.config.trading_config.thresholds, k, v, raising=False)

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        async with SL() as s:
            s.add(models.Order(pair="HTRUSDT", direction="LONG", status="OPEN", is_paper=True, entry_strategy="FRENZY_WILLY_HT", confidence="STRONG_BUY",
                               entry_price=1.0, quantity=100.0, investment=20.0, leverage=5, notional_value=100.0, entry_fee=0.05,
                               opened_at=dt.datetime.utcnow(), frenzy_willy_tp_target_pct=5.0))
            await s.commit()
        e = TE.TradingEngine(); e.is_paper_mode = True
        async with SL() as db:
            await e.update_orders_cache(db)
        c = TE._open_orders_cache["HTRUSDT"][0]
        await eng.dispose()
        return c
    c = asyncio.run(go())
    th = TE.config.trading_config.thresholds
    assert c['entry_strategy'] == "FRENZY_WILLY_HT" and c['frenzy_willy_tp_target_pct'] == 5.0
    assert frenzy_willy_exit_for(0.3, 0.3, th, held_minutes=31, row=c)[0] is False and frenzy_willy_exit_for(-1.0, 0.0, th, held_minutes=1440, row=c)[1] == "MAX_HOLD_TIME"
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "max_hold = frenzy_willy_levels(config.trading_config.thresholds, order.entry_strategy)[2]" in src   # the max-hold sweep
    assert TE.order_backstop_pct(th, "FRENZY_WILLY_HT", leverage=5) > 4.0                                         # backstop beyond the −4 stop at 5×


def test_ht_exit_config_ui_exports():
    import config as C
    mf = C.SignalThresholds.model_fields
    assert (mf["frenzy_willy_ht_late_tp_enabled"].default, mf["frenzy_willy_ht_high_target_enabled"].default, mf["frenzy_willy_ht_max_hold_minutes"].default) == (True, True, 0.0)
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for i in ("config-fz-willy-ht-late-tp", "config-fz-willy-ht-high-target", "config-fz-willy-ht-max-hold"):
        assert html.count(f'id="{i}"') == 1, i
    assert "['config-fz-willy-ht-max-hold', 'frenzy_willy_ht_max_hold_minutes', 0]" in html
    for k in ("frenzy_willy_ht_late_tp_enabled", "frenzy_willy_ht_high_target_enabled"):
        assert f"config.thresholds.{k} !== false" in html and f"{k}: document.getElementById(" in html
    assert "HT late TP ${_bt.frenzy_willy_ht_late_tp_enabled === false ? 'OFF'" in html
    from services.frenzy import frenzy_willy_ht_config_errors
    bad = _live_th(frenzy_willy_ht_max_hold_minutes=0.5, frenzy_willy_ht_late_tp_enabled="no")
    errs = " ".join(frenzy_willy_ht_config_errors(bad))
    assert "frenzy_willy_ht_max_hold_minutes" in errs and "frenzy_willy_ht_late_tp_enabled" in errs and not frenzy_willy_ht_config_errors(_live_th())


# ── ⑥ FRENZY_LONG off: the FRENZY pass still runs for EXHAUSTION / DISTRIBUTION / WILLY_HT ──────────────────────────────────────────

def test_frenzy_pass_runs_with_frenzy_long_off(monkeypatch):
    import asyncio
    import services.trading_engine as TE
    from tests.test_frenzy_willy import _hold_db
    assert json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]["frenzy_long_enabled"] is False
    th = _live_th(frenzy_long_enabled=False, frenzy_short_observe=False, frenzy_wide_enabled=False, frenzy_lite_enabled=False,
                  frenzy_willy_enabled=True, frenzy_willy_plain_entries_enabled=False, exhaustion_short_enabled=True, distribution_short_enabled=True,
                  flatbase_short_enabled=False, exhaustion_wide_enabled=False)
    for k, v in vars(th).items():
        monkeypatch.setattr(TE.config.trading_config.thresholds, k, v, raising=False)
    bar_open = 1_763_500_500_000 // 900_000 * 900_000
    monkeypatch.setattr(TE, "_leash_time", NS(time=lambda: (bar_open + 6_000) / 1000.0))
    monkeypatch.setattr(TE, "_frenzy_flags", {}); monkeypatch.setattr(TE, "_frenzy_status", {}); monkeypatch.setattr(TE, "_frenzy_kl_cache", {})
    monkeypatch.setitem(TE._frenzy_norm_cache, "PUMPUSDT", (bar_open, 1e6, 10 ** 12))
    rows = [[bar_open - 300_000 * (400 - i), 1.0, 1.01, 0.99, 1.0, 1000.0] for i in range(401)]   # 400 closed 5m bars + the forming one

    async def top(*a, **k):
        return [dict(pair="PUMPUSDT", symbol="PUMP/USDT:USDT", volume_24h=9e7, change_24h=40.0, range_24h=45.0, price=1.0)]

    async def ohlcv(sym, tf, limit=100):
        return rows if tf == "5m" else []
    monkeypatch.setattr(TE.binance_service, "get_top_futures_pairs", top)
    monkeypatch.setattr(TE.binance_service, "get_ohlcv", ohlcv)
    monkeypatch.setattr(TE.binance_service.public_exchange, "markets", {}, raising=False)
    sp = bar_open - 3 * 3_600_000
    ep = dict(spike_ts=sp, verified=True, hours=3.0, in_state=False, fresh_on=False, above_hour=True, vol_mult=30.0, vs_vwap_pct=1.0,
              above_streak=12, bar_red=True, bar_ret_pct=-0.1, peak=1.2, price=1.0, base=0.8, last_bar_ts=bar_open - 300_000, on_bar_ts=None)
    monkeypatch.setattr(TE, "frenzy_walk", lambda bars, norm, th_: dict(ep))
    calls = {}

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        e = TE.TradingEngine(); e.is_paper_mode = True; e.is_running = True
        e._frenzy_gvol_start = lambda *a, **k: None

        async def willy(db, pair, ep_, flag, ind, bar_open_, prev, on, fz, pair_warm=False, closed=None):
            calls['willy'] = (pair, on)
        e._frenzy_willy_eval = willy

        async def lite_retry_sweep(*a, **k):
            return None
        e._frenzy_lite_retry_sweep = lite_retry_sweep
        async with SL() as db:
            await e._update_frenzy_pass(db)
        st = dict(e._exh_state())
        await eng.dispose()
        return e, st
    e, st = asyncio.run(go())
    assert "PUMPUSDT" in TE._frenzy_flags                                                     # the flag is computed with FRENZY_LONG off
    assert ("PUMPUSDT", sp) in st and st[("PUMPUSDT", sp)]["universe"] == "FRENZY"            # EXHAUSTION / DISTRIBUTION track the episode
    assert calls.get('willy') == ("PUMPUSDT", True)                                           # the WILLY machinery (HT arming) runs


# ── DL 587 review 8 / 9 / 12 ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_review8_plain_off_with_no_turnover_cap_is_refused():
    from services.frenzy import frenzy_willy_ht_config_errors
    bad = _live_th(frenzy_willy_plain_entries_enabled=False, frenzy_willy_max_vol_mcap_ratio=0)
    assert any("can never open anything" in x for x in frenzy_willy_ht_config_errors(bad))
    assert not any("can never open anything" in x for x in frenzy_willy_ht_config_errors(_live_th()))
    ok = _live_th(frenzy_willy_plain_entries_enabled=True, frenzy_willy_max_vol_mcap_ratio=0)
    assert not any("can never open anything" in x for x in frenzy_willy_ht_config_errors(ok))


def test_review9_plain_off_held_book_records_plain_off_not_a_hold(monkeypatch):
    """the WILLY hold is held and the setup passes turnover: no FRENZY_WILLY_HOLD / PAIR_HELD record — it could never open (plain off)."""
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_plain_entries_enabled=False)
    e = _eng(TE, R=0.5, held=True)
    o, flag = _run(e, TE)
    assert o == [] and e.blocks.count("WILLY_PLAIN_OFF") == 1
    assert not any(b in ("FRENZY_WILLY_HOLD", "PAIR_HELD") for b in e.blocks + e.events)
    assert "FOOUSDT" not in (getattr(e, "_fz_willy_pending", None) or {})


def test_review12_registry_pins_the_ht_exit_fields():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import observe_registry as R
    assert all(k in R.BOOKED["scout_willy_ht_lines.py"] for k in ("frenzy_willy_ht_max_hold_minutes", "frenzy_willy_ht_late_tp_enabled",
                                                                  "frenzy_willy_ht_high_target_enabled"))
    txt = open(os.path.join(ROOT, "scripts", "observe_registry.py"), encoding="utf-8").read()
    seg = txt[txt.index("scout_willy_ht_lines"):][:3000]
    for k in ("frenzy_willy_ht_max_hold_minutes", "frenzy_willy_ht_late_tp_enabled", "frenzy_willy_ht_high_target_enabled"):
        assert k in seg, k
    assert R is not None


def test_review17_frenzy_loop_runs_for_flatbase_exhaustion_distribution():
    import main as M
    off = {k: False for k in M._FRENZY_LOOP_SWITCHES}
    assert M._frenzy_loop_on(NS(**off)) is False
    for k in ("flatbase_short_enabled", "exhaustion_short_enabled", "distribution_short_enabled", "frenzy_willy_enabled"):
        assert M._frenzy_loop_on(NS(**dict(off, **{k: True}))) is True, k
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "if not _frenzy_loop_on(_th):" in src


def test_rereview2_unreadable_turnover_with_held_stays_armed_and_pop_journals(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_plain_entries_enabled=False)
    e = _eng(TE, R=None, held=True)                                                           # turnover UNREADABLE + held
    o, flag = _run(e, TE)
    assert "WILLY_PLAIN_OFF" not in e.blocks and "FOOUSDT" in e._fz_willy_pending            # not popped: the normal held branch, armed
    e2 = _eng(TE, R=0.5, held=True)                                                           # passes turnover + held → popped, journaled
    _run(e2, TE)
    assert "WILLY_PLAIN_OFF" in e2.blocks and "WILLY_PLAIN_OFF" in e2.events


def test_rereview3_plain_off_ignores_willys_own_slot_cap_for_ht(monkeypatch):
    import services.trading_engine as TE
    _cfg(monkeypatch, TE, frenzy_willy_plain_entries_enabled=False, frenzy_willy_global_hold=False)
    e = _eng(TE, R=2.0)

    async def full(db=None):
        return True
    e._willy_slots_full = full                                                                # WILLY's own cap full
    o, _ = _run(e, TE)
    assert [(x["willy"], x["ht"]) for x in o] == [("A", True)]                                # HT still tried (its own slot judged there)
    _cfg(monkeypatch, TE, frenzy_willy_plain_entries_enabled=True, frenzy_willy_global_hold=False)
    e = _eng(TE, R=2.0)
    e._willy_slots_full = full
    o, _ = _run(e, TE)
    assert not any(x["ht"] for x in o)                                                        # plain on: WILLY's cap still applies (unchanged)


def test_rereview10_frenzy_loop_uses_the_sleeve_helpers():
    import main as M
    off = {k: False for k in M._FRENZY_LOOP_SWITCHES}
    assert M._frenzy_loop_on(NS(**dict(off, exhaustion_short_enabled="yes"))) is False            # the helpers: anything but True = off
    assert M._frenzy_loop_on(NS(**dict(off, distribution_short_enabled=True))) is True
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "flatbase_enabled(th) or exhaustion_enabled(th) or distribution_enabled(th)" in src
    assert 'open_es[p_] = es_ or "MOMENTUM"' in src                                              # (re-review 9)
