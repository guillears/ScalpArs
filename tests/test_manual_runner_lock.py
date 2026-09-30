"""🩹 Sep-30 two armed manual QNT runners (peak +0.62 / +0.40, entry ATR 2.2 %) rode back to the −0.70 stop: the manual exits
were blind to the entry ATR, which disabled the armed-runner floor and its +0.10 break-even lock. Now the profit side reads
the real ATR; only the STOP widening stays hidden for manual fills (leverage / liquidation safety)."""
import os, sys
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
import config
from services.indicators import check_exit_conditions, calculate_pnl

E, LEV, INV = 306.71, 20.0, 800.0                       # the first QNT trade (id 2), as booked


def _check(price, peak, atr, manual=True):
    tc = config.trading_config
    fee = INV * LEV * float(getattr(tc, 'taker_fee', tc.trading_fee))
    kw = dict(sl_atr_pct=None, sl_atr_same=False) if manual else {}
    return check_exit_conditions("LONG", E, price, LEV, "STRONG_BUY", peak_pnl=peak, quantity=INV * LEV / E, entry_fee=fee,
                                 investment=INV, current_tp_level=1, dynamic_tp_target=0.4, signal_active=False,
                                 tp_trailing_enabled=True, entry_atr_pct=atr, **kw)


def _pnl(price):
    r = _check(price, 0.0, None)
    return r.get("pnl_pct", r.get("pnl_percentage"))


def test_live_config_is_the_one_these_trades_ran_on():
    th = config.trading_config.thresholds
    assert th.runner_trail_enabled and th.runner_trail_be_ratchet_enabled and abs(th.runner_trail_be_lock_pct - 0.10) < 1e-9
    assert abs(th.runner_trail_arm_peak - 0.40) < 1e-9 and th.runner_trail_atr_min == 0.0


def test_the_qnt_runners_now_bank_the_break_even_lock():
    # a price just under the lock and one comfortably above it, found on the checker's own (fee-net) pnl
    under = E * 1.0015                                   # ≈ +0.15 raw − ~0.09 fees ≈ +0.06 net
    over = E * 1.0030                                    # ≈ +0.30 raw ≈ +0.21 net
    for peak in (0.62, 0.397):                           # the two trades
        r = _check(under, peak, 2.2)
        assert r["should_close"] and r["reason"].startswith("RUNNER_TRAIL"), (peak, r)
        r = _check(over, peak, 2.2)
        assert not r["should_close"], (peak, r)          # above the lock: rides


def test_an_armed_runner_without_any_atr_still_keeps_the_lock():
    r = _check(E * 1.0015, 0.62, None)
    assert r["should_close"] and r["reason"].startswith("RUNNER_TRAIL")
    r = _check(E * 1.0015, 0.62, 0.0)                    # 0 counts as missing
    assert r["should_close"] and r["reason"].startswith("RUNNER_TRAIL")


def test_manual_stop_is_never_widened_by_the_atr():
    price = E * (1 - 0.0070)                             # ≈ −0.79 net: past the −0.70 stop, inside a widened one
    r = _check(price, 0.10, 2.2, manual=True)
    assert r["should_close"] and "STOP" in r["reason"], r
    r = _check(price, 0.10, 2.2, manual=False)           # a bot fill with the same ATR: stop widened (≤ −1.2) → holds
    assert not (r["should_close"] and "STOP" in r["reason"]), r


def test_wiring():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def update_orders_cache"); j = eng.find("\n    async def ", i + 10); cache = eng[i:(j if j > 0 else len(eng))]
    assert "'entry_atr_pct': getattr(order, 'entry_atr_pct', None)," in cache
    assert "'sl_entry_atr_pct': exit_entry_atr_pct(order.entry_strategy, getattr(order, 'entry_atr_pct', None))" in cache
    assert "_entry_atr_pct = order_info.get('sl_entry_atr_pct', order_info.get('entry_atr_pct'))" in eng
    assert "sl_atr_pct=exit_entry_atr_pct(order.entry_strategy, getattr(order, 'entry_atr_pct', None)), sl_atr_same=False" in eng
    assert "_rl_lock_only = (not _rl_has_atr and _rl_amin <= 0" in eng                     # realtime mirror of the no-ATR lock
