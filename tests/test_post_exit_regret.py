"""🧭 Oct-30: the restored Post-Exit Regret Deep Dive (DECISION_LOG 419) — every exit name gets a row or is listed, every tracking path is whitelist-free,
the table sits below Winning Trades Drawdown on the dashboard and in both text reports, and the Frenzy Fills clean-up holds."""
import datetime as dt
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services.post_exit_regret import post_exit_regret_rows, post_exit_untracked  # noqa: E402
from models import Order  # noqa: E402


def _o(reason, peak=0.5, direction="LONG"):
    t = dt.datetime(2026, 10, 30)
    kw = {c.name: None for c in Order.__table__.columns}
    kw.update(close_reason=reason, status="CLOSED", direction=direction, opened_at=t, closed_at=t + dt.timedelta(minutes=30), pnl_percentage=-1.0,
              post_exit_peak_pnl=peak, post_exit_trough_pnl=-2.0 if peak is not None else None, post_exit_final_pnl=0.1 if peak is not None else None,
              post_exit_peak_minutes=5.0, post_exit_trough_minutes=9.0)
    return NS(**kw)


def _html():
    return open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()


def test_every_exit_name_gets_its_row_and_the_numbers():
    names = ["FRENZY_TP", "WILLY_LATE_TP", "MAX_HOLD_TIME", "MANUAL", "EXTERNAL_CLOSE", "BR_STOP_LOSS", "A_BRAND_NEW_EXIT"]
    rows = post_exit_regret_rows([_o(n) for n in names] + [_o("FRENZY_TP", peak=1.5, direction="SHORT")])
    assert [r["reason"] for r in rows] == sorted(names)
    r = next(x for x in rows if x["reason"] == "FRENZY_TP")
    assert (r["count"], r["longs"], r["shorts"], r["avg_post_peak"], r["avg_duration"], r["peak_first_pct"], r["recovery_010_pct"]) == (2, 1, 1, 1.0, "00:30:00", 100.0, 100.0)
    assert r["avg_sig_lost_min"] is None and r["avg_strpk_pct"] is None      # not stamped → dash, never zero


def test_untracked_exits_are_listed_never_silent():
    assert post_exit_regret_rows([_o("MAX_HOLD_TIME", peak=None)]) == []
    assert post_exit_untracked([_o("MAX_HOLD_TIME", peak=None), _o("MAX_HOLD_TIME", peak=None), _o("FRENZY_TP")]) == [{"reason": "MAX_HOLD_TIME", "n": 2}]
    h = _html()
    assert h.count('id="post-exit-untracked"') == 1 and h.count("Closed with no post-exit data yet") == 3 and h.count("(perf.post_exit_untracked && perf.post_exit_untracked.length > 0)) {") == 2   # UI + both exports, also with no tracked row


def test_no_tracking_whitelist_on_either_path():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    a = src.index("def _register_post_exit_tracking"); assert '_reason_base.startswith("STOP_LOSS")' not in src[a:a + 1500]
    b = src.index("async def _recover_post_exit_tracking"); assert '_reason_base.startswith("STOP_LOSS")' not in src[b:b + 4000]


def test_tracking_registers_on_the_reconciler_after_booking_and_on_close_fallback():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    i = main.index("_register_post_exit_tracking(order, order.close_reason)")
    assert main.count("_register_post_exit_tracking(") == 1 and i > main.index("order.total_fee = round((order.entry_fee or 0) + order.exit_fee, 4)")
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert src.count("self._register_post_exit_tracking(order, order.close_reason)") == 1 and src.count("self._register_post_exit_tracking(order, reason)") == 1


def test_register_keeps_a_booked_row_and_skips_one_without_exit_price():
    from services.trading_engine import TradingEngine
    import config
    e = TradingEngine.__new__(TradingEngine)
    e._post_exit_tracking = {}
    cfg = {c.name: None for c in Order.__table__.columns}
    cfg.update(id=1, pair="ZZUSDT", exit_price=1.0, entry_price=1.0, quantity=1.0, direction="LONG", entry_fee=0.0, pnl_percentage=1.0, peak_pnl=1.0, close_reason="X")
    TradingEngine._register_post_exit_tracking(e, NS(**cfg), "A_BRAND_NEW_EXIT")
    assert 1 in e._post_exit_tracking and e._post_exit_tracking[1]["close_reason"] == "A_BRAND_NEW_EXIT"
    TradingEngine._register_post_exit_tracking(e, NS(**dict(cfg, id=2, exit_price=None)), "BACKSTOP_STOP")
    assert 2 not in e._post_exit_tracking and config.trading_config is not None


def test_table_is_below_winning_drawdown_on_every_surface():
    h = _html()
    assert h.count('id="post-exit-regret-body"') == 1 and h.count("getElementById('post-exit-regret-body')") == 1
    w = h.index('id="winning-drawdown-body"')
    p = h.index('id="post-exit-regret-body"')
    assert w < p < h.index("<!-- 🩹 Recovery Hold (Oct 2")
    assert h.count("## Post-Exit Regret Deep Dive") == 2
    wd = [i for i in range(len(h)) if h.startswith("## Winning Trades Drawdown", i)]
    pe = [i for i in range(len(h)) if h.startswith("## Post-Exit Regret Deep Dive", i)]
    rh = [i for i in range(len(h)) if h.startswith("if (perf.recovery_hold && perf.recovery_hold.n > 0)", i)]
    assert len(wd) == len(pe) == len(rh) == 2 and all(a < b < c for a, b, c in zip(wd, pe, rh))
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert main.count('"post_exit_regret_deep_dive"') == 3 and main.count('"post_exit_untracked"') == 3   # payload + both fallbacks


def test_frenzy_clean_up():
    h = _html()
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "frenzy_breaks" not in h and "_frenzy_break_rows" not in main       # the table is gone
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "[FRENZY_SHORT_OBS]" in eng                                        # the recording stays
    for stale in ("−8 % TP / +15 % stop", "· TP +1 / no SL / 120 min · global hold", "· same exit)", "💪 sized-up: ADX rising"):
        assert stale not in main, stale
    assert '_fz_lbl + _fz_exit_label(_fz_es) + ")"' in main and '"coin": _fz_es' in main and "FLATBASE_EXIT_408_AT" in main
    assert "fzToggleCoins" in h and h.count("r.coin && !_open ? ' hidden'") == 1
