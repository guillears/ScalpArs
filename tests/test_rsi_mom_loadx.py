"""🧭 Sep-29 LOW-ADX RSI-MOMENTUM gate (DECISION_LOG 126) — pure-rule invariants (fail-OPEN), live/default parity, engine +
pool-builder + ledger + UI wiring."""
import json
import os
from types import SimpleNamespace

from services.indicators import rsi_mom_loadx_block

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TH = SimpleNamespace(long_rsi_momentum_adx_max=21.0)


def test_blocks_only_falling_rsi_on_weak_adx():
    assert rsi_mom_loadx_block(TH, 58.07, 61.06, 20.76) is True      # B14 TAO: RSI fell, ADX 20.8
    assert rsi_mom_loadx_block(TH, 54.38, 59.62, 16.53) is True      # B14 VIRTUAL
    assert rsi_mom_loadx_block(TH, 60.83, 55.00, 20.0) is False      # RSI rising → allowed
    assert rsi_mom_loadx_block(TH, 58.0, 61.0, 21.0) is False        # ADX == max → allowed (strict <)
    assert rsi_mom_loadx_block(TH, 58.0, 61.0, 28.64) is False       # strong trend → allowed (B14 INJ winner)
    assert rsi_mom_loadx_block(TH, 61.0, 61.0, 18.0) is False        # flat RSI → allowed (strict <)


def test_fail_open_and_off():
    for a, b, c in ((None, 61.0, 18.0), (58.0, None, 18.0), (58.0, 61.0, None), (float("nan"), 61.0, 18.0), ("x", 61.0, 18.0)):
        assert rsi_mom_loadx_block(TH, a, b, c) is False
    assert rsi_mom_loadx_block(SimpleNamespace(long_rsi_momentum_adx_max=0), 58.0, 61.0, 18.0) is False
    assert rsi_mom_loadx_block(SimpleNamespace(), 58.0, 61.0, 18.0) is False
    assert rsi_mom_loadx_block(SimpleNamespace(long_rsi_momentum_adx_max=None), 58.0, 61.0, 18.0) is False


def test_live_json_armed_default_off_and_wiring():
    import config
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["long_rsi_momentum_adx_max"] == 21.0
    assert th["rsi_momentum_filter_enabled"] is False                 # the old unscoped leg stays off
    assert config.SignalThresholds.model_fields["long_rsi_momentum_adx_max"].default == 0.0
    assert config.load_trading_config().thresholds.long_rsi_momentum_adx_max == 21.0
    ind = open(os.path.join(ROOT, "services", "indicators.py")).read()
    assert 'if rsi_mom_loadx_block(th, rsi, rsi_prev2, adx):' in ind and '_l_fails.append("PAIR_RSI_MOMENTUM_LOADX")' in ind
    assert ind.count('_fails.append("PAIR_RSI_MOMENTUM_LOADX")') == 1  # LONG leg only — never appended on the SHORT side
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count("rsi_prev2=indicators.get('rsi_prev2')") >= 3      # scan loop + revalidate + flip path all feed rsi_prev2 (deep review: the scan never did)
    assert config.SignalThresholds.model_fields["rsi_momentum_filter_enabled"].default is False   # old unscoped leg: default == live
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert 'STACK_VERSION = "2026-09-29a"' in bld and "rsi_mom_loadx_block(_LOADX_TH, r.get('entry_rsi'), r.get('entry_rsi_prev'), r.get('entry_adx'))" in bld
    led = open(os.path.join(ROOT, "scripts", "current_stack_ledger.py")).read()
    assert "rsi_mom_loadx_block(_th, a, b, c)" in led and "!= 21.0" in led
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-long-rsi-momentum-adx-max") == 3           # input + load + save
    assert "Low-ADX RSI-momentum gate (Sep 29)" in ui                  # report line (feeds both exports)


def _long_candidate(rsi, rsi_prev2, adx, adx_max, fails):
    """Drive the REAL get_signal with a textbook LONG stack and capture every LONG fail name."""
    from services.indicators import get_signal
    import config
    th = config.SignalThresholds(long_rsi_momentum_adx_max=adx_max, rsi_momentum_filter_enabled=False)
    get_signal(ema5=1.030, ema8=1.020, ema13=1.010, ema20=1.000, rsi=rsi, adx=adx, volume=2000.0, avg_volume=1000.0, price=1.035,
               ema20_prev3=0.990, ema50=0.980, ema50_prev12=0.960, rsi_prev3=rsi_prev2, rsi_prev2=rsi_prev2,
               ema5_prev1=1.025, ema8_prev1=1.018, ema5_prev2=1.020, ema8_prev2=1.016, ema13_prev1=1.008, ema13_prev2=1.006,
               adx_prev1=adx - 0.5, high_20=1.05, low_20=0.95,
               block_recorder=lambda name, d: None, multi_block_recorder=lambda names, d: fails.extend(names) if d == "LONG" else None,
               config=th)


def test_get_signal_records_the_new_fail_only_when_both_legs_hold():
    f = []; _long_candidate(58.0, 61.0, 18.0, 21.0, f); assert "PAIR_RSI_MOMENTUM_LOADX" in f
    f = []; _long_candidate(62.0, 61.0, 18.0, 21.0, f); assert "PAIR_RSI_MOMENTUM_LOADX" not in f     # RSI rising
    f = []; _long_candidate(58.0, 61.0, 26.0, 21.0, f); assert "PAIR_RSI_MOMENTUM_LOADX" not in f     # strong trend
    f = []; _long_candidate(58.0, 61.0, 18.0, 0.0, f);  assert "PAIR_RSI_MOMENTUM_LOADX" not in f     # gate off
    f = []; _long_candidate(58.0, None, 18.0, 21.0, f); assert "PAIR_RSI_MOMENTUM_LOADX" not in f     # rsi_prev2 missing → fail-open
    assert "PAIR_RSI_MOMENTUM" not in f                                                             # the old unscoped leg stays off
