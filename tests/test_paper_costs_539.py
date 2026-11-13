"""💸 DECISION_LOG 539 — paper = live costs: paper market exits at the live book (exit slippage recorded), funding accrued on open trades and
folded into pnl (kept on its own in funding_fee_usd, never added twice), the M1 shadow on the BANK line."""
import os
import sys
from types import SimpleNamespace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from services.paper_costs import (accrue_funding, exit_slip_pct, fold_funding, funding_outside_pnl, funding_payment_usd,  # noqa: E402
                                  paper_exit_book_price, pnl_with_funding)


def _src(p):
    return open(os.path.join(ROOT, p)).read()


def test_book_side_and_slip_sign():
    assert paper_exit_book_price("LONG", 99.9, 100.1) == 99.9      # closing a long sells into the bid
    assert paper_exit_book_price("SHORT", 99.9, 100.1) == 100.1    # closing a short buys the ask
    assert paper_exit_book_price("SHORT", 99.9, None) is None
    assert exit_slip_pct("LONG", 100, 99.9) == 0.1                  # sold lower = worse = +
    assert exit_slip_pct("SHORT", 100, 100.1) == 0.1                # bought higher = worse = +
    assert exit_slip_pct("SHORT", 100, 99.95) == -0.05
    assert exit_slip_pct("LONG", 0, 1) is None


def test_funding_payment_sign():
    assert funding_payment_usd("LONG", 10, 0.001, 100) == -1.0       # a long pays a positive rate
    assert funding_payment_usd("SHORT", 10, 0.001, 100) == 1.0       # a short receives it
    assert funding_payment_usd("SHORT", 10, -0.00095, 100) == -0.95  # negative rate: the short pays (the PRL case)
    assert funding_payment_usd("SHORT", 0, 0.001, 100) is None


def test_accrue_window_and_fallback_mark():
    rows = [(1000, 0.001, 100.0), (2000, 0.001, None), (3000, 0.001, 100.0), (500, 0.5, 1.0)]
    tot, last, skip = accrue_funding("LONG", 1, rows, 999, 2000, fallback_mark=200.0)
    assert last == 2000 and skip == 0 and abs(tot - (-0.1 - 0.2)) < 1e-12   # (999, 2000]; the mark-less row uses the fallback
    assert accrue_funding("LONG", 1, rows, 3000, 4000) == (0.0, None, 0)       # the boundary row itself was already counted
    assert accrue_funding("LONG", 1, [(2000, 0.001, None)], 0, 9999)[2] == 1   # no mark, no fallback → skipped, never guessed


def test_never_added_twice():
    assert funding_outside_pnl(-5.0, True) == 0.0
    assert funding_outside_pnl(-5.0, None) == -5.0
    assert funding_outside_pnl(None, None) == 0.0
    assert pnl_with_funding(100.0, -5.0, True) == 100.0
    assert pnl_with_funding(100.0, -5.0, False) == 95.0
    assert fold_funding(100.0, 1.0, 10000.0, -5.0) == (95.0, 0.95)
    assert fold_funding(None, 1.0, 10000.0, -5.0) == (None, 1.0)


def test_flatbase_gate_reader_respects_flag():
    from services.flatbase import net_of_funding_pct
    assert net_of_funding_pct(-10.0, -50.0, 1.0, 10000.0) == (-10.5, True)       # legacy row: funding added
    assert net_of_funding_pct(-10.5, -50.0, 1.0, 10000.0, True) == (-10.5, True)  # 539 row: already inside
    assert net_of_funding_pct(-10.0, None, 1.0, 10000.0) == (-10.0, False)


def test_accounting_row_backs_funding_out():
    from services.accounting_export import trade_row
    from datetime import datetime
    base = dict(entry_fee=1.0, exit_fee=1.0, notional_value=1000.0, entry_price=1.0, quantity=1000.0, leverage=10.0, pair="XUSDT",
                direction="SHORT", exit_price=0.95, opened_at=datetime(2026, 11, 13), closed_at=datetime(2026, 11, 14))
    old = trade_row(SimpleNamespace(pnl=48.0, funding_fee_usd=-3.0, funding_in_pnl=None, **base))
    new = trade_row(SimpleNamespace(pnl=45.0, funding_fee_usd=-3.0, funding_in_pnl=True, **base))
    assert old["Net P&L (USDT)"] == new["Net P&L (USDT)"] == 45.0
    assert old["Gross P&L (USDT)"] == new["Gross P&L (USDT)"] == 50.0


def test_period_stats_funding_only_from_539_rows():
    import importlib
    main = importlib.import_module("main")
    o = lambda pnl, f, fip: SimpleNamespace(direction="SHORT", pnl=pnl, pnl_percentage=1.0, total_fee=1.0, investment=100.0,  # noqa: E731
                                            notional_value=1000.0, exit_slippage_pct=None, entry_slippage_pct=None,
                                            funding_fee_usd=f, funding_in_pnl=fip)
    r = main._period_stats("x", [o(10, -2.0, True), o(10, -7.0, None)])
    assert r["total_funding"] == -2.0


def test_engine_wiring():
    s = _src("services/trading_engine.py")
    assert "paper_exit_book_price(order.direction" in s and "_slippage_pct = _xpx, _xsl" in s
    assert "abs(_xsl) <= PAPER_EXIT_SLIP_SANE_PCT" in s
    assert "_fund_usd, _fund_last = await self._funding_at_close(order, actual_exit_price)" in s
    assert "order.funding_in_pnl = True" in s
    assert "_close_pct = pnl_data.get('pct_price', pnl_data['pnl_percentage'])" in s   # best / worst stay price-only
    assert "if _db_commit_success:\n            try:\n                _ft = asyncio.create_task(self._funding_reconcile_async(" in s
    assert "_stamp_funding_async" not in s
    m = _src("main.py")
    assert "_funding_accrual_loop(lambda: should_stop)" in m and "globals().get('_funding_task')" in m
    assert '"funding_usd": round(float(getattr(o, \'funding_fee_usd\', None) or 0.0), 4)' in m


def test_reporting_parity_funding_column():
    h = _src("templates/index.html")
    assert h.count("${'Funding'.padEnd(10)}") == 2                         # both text exports (D12)
    assert h.count("${fmtFunding(r.total_funding).padEnd(10)}") == 2
    assert ">Funding</th>" in h and "fmtFunding(r.total_funding)}</td>" in h
    assert h.count("function fmtFunding(") == 1


def test_model_and_migration():
    assert "funding_in_pnl = Column(Boolean" in _src("models.py") and "funding_accrued_ms = Column(BigInteger" in _src("models.py")
    d = _src("database.py")
    assert "ADD COLUMN funding_in_pnl BOOLEAN" in d and "ADD COLUMN funding_accrued_ms BIGINT" in d


def _bars(t0, closes, step=300_000):
    return [[t0 + i * step, c, c * 1.001, c * 0.999, c] for i, c in enumerate(closes)]


def test_m1_shadow_skip_and_enter():
    import scout_flatbase_confirm_watch as W
    H4, M5 = W.H4, W.M5
    s4_close_ms = 10 * H4
    t0 = s4_close_ms - 48 * M5                       # bars from 4 h before the S4 close
    n = 48 + 48 + 6 * 48 + 900
    px = [100.0] * n
    s4_i = 47                                        # the bar CLOSING at the S4 boundary
    px[s4_i] = 100.0
    conf_i = s4_i + 48                               # the shallow confirm: 0.5 % below S4 at the next 4-h close
    px[conf_i] = 99.5
    te = s4_close_ms + H4                            # the fill = the confirm close
    # never ≥ 1 % below inside 24 h → SKIP
    why, pct = W.price_m1(_bars(t0, px), 100.0, te, 10**15)
    assert (why, pct) == ("SKIP", 0.0)
    # a 4-h close 1.5 % below 8 h after S4 → enters at the next open, then a 6 % drop → TP
    px2 = list(px)
    k = s4_i + 96
    px2[k] = 98.5
    for j in range(k + 1, k + 30):
        px2[j] = 98.5
    for j in range(k + 30, k + 60):
        px2[j] = 92.0
    why2, pct2 = W.price_m1(_bars(t0, px2), 100.0, te, 10**15)
    assert why2 == "TP" and pct2 == W.TP_NET
    assert W.price_m1(_bars(t0, px), 120.0, te, 10**15)[0] == "UNPRICED"   # no S4 bar matches the stamp → never guessed


def test_topup_only_across_an_hour_boundary():
    from services.paper_costs import funding_topup_needed
    H = 3_600_000
    assert not funding_topup_needed(5 * H + 1, 5 * H + 30 * 60_000)   # a 30-min scalp inside one hour: no settlement possible, no read
    assert funding_topup_needed(5 * H - 1, 5 * H + 1)                  # crossed the hour
    assert funding_topup_needed(None, 5 * H)                            # unknown → read


def test_live_income_foreign_guard():
    from services.paper_costs import live_income_foreign
    assert not live_income_foreign(-1.0, -0.95)          # Binance income ≈ this position's estimate
    assert live_income_foreign(-5.0, -1.0)               # far beyond → another position on the coin: book the estimate
    assert not live_income_foreign(-0.04, 0.0)           # dust within the slack
    assert not live_income_foreign(None, -1.0)


def test_review_fixes_wired():
    s = _src("services/trading_engine.py")
    assert "funding_topup_needed(_since, _now)" in s and "asyncio.wait_for(_read(), 1.0)" in s
    assert "_row.opened_at != opened_at" in s and "live_income_foreign(_fund, _est)" in s
    assert "flag_modified(order, _fm)" in s
    assert "_tw_pct = pnl_data.get('pct_price', order.pnl_percentage)" in s
    assert "self._touch_finalize(order, pnl_data.get('pct_price', pnl_data['pnl_percentage']), reason)" in s
    assert "await self._recalculate_paper_balance(_fdb)" not in s           # no unlocked balance recalc in the background task
    assert "order.funding_fee_usd, order.funding_in_pnl = round(_fu_fb, 6), True" in s   # the live fallback close folds too
    m = _src("main.py")
    assert "order.funding_fee_usd, order.funding_in_pnl = round(_fu_rc, 6), True" in m   # reconciled closes fold too
    assert "pnl_pct, _pct_shown," in m
    assert "fin = lambda o: f(o.funding_fee_usd) if getattr(o, 'funding_in_pnl', None) else 0.0" in _src("services/track_record.py")
    assert "if getattr(o, 'funding_in_pnl', None) else 0.0)   # 💸 (539) ex funding" in _src("services/app_trades.py")
    a = _src("services/funding_accrual.py")
    assert "_LAST_READ" in a and "PUBLISH_LAG_MS" not in a


def test_m1_never_takes_the_confirm_bar_as_s4():
    import scout_flatbase_confirm_watch as W
    H4, M5 = W.H4, W.M5
    s4_close_ms = 10 * H4
    t0 = s4_close_ms - 48 * M5
    px = [100.0] * (48 + 48 + 6 * 48 + 900)
    px[47 + 48] = 99.97                              # confirm only 0.03 % below S4 — within the 0.05 % match band
    te = s4_close_ms + H4
    # the S4 bar is still found (the confirm bar is excluded) → the 24 h window runs from S4, no ≥ 1 % close → SKIP
    assert W.price_m1(_bars(t0, px), 100.0, te, 10**15) == ("SKIP", 0.0)
