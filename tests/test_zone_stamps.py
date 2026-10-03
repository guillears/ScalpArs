"""🧭 Sep-29 ZONE STAMPS (DECISION_LOG 127) — closed-bar semantics, fail-safety, observe-only, D11/D12 parity."""
import json
import os

import numpy as np
import pandas as pd

from services.indicators import closed_ema_gap_pct, last_closed_bar_ret_pct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _bars(closes):
    return [[i * 300000, c, c, c, c, 1.0] for i, c in enumerate(closes)]


def test_closed_bars_only_and_matches_pandas():
    rng = np.random.default_rng(3); closes = list(84000 * np.exp(np.cumsum(rng.normal(0, 0.002, 300))))
    got = closed_ema_gap_pct(_bars(closes), 50, 100)
    s = pd.Series(closes[:-1]); exp = (s.ewm(span=50, adjust=False).mean().iloc[-1] / s.ewm(span=100, adjust=False).mean().iloc[-1] - 1) * 100
    assert abs(got - round(exp, 4)) < 1e-9
    spiked = _bars(closes[:-1] + [closes[-1] * 5])                       # a wild FORMING print changes nothing
    assert closed_ema_gap_pct(spiked, 50, 100) == got
    assert last_closed_bar_ret_pct(_bars([100, 101, 102, 999])) == round((102 / 101 - 1) * 100, 4)


def test_fail_safe():
    assert closed_ema_gap_pct(None, 50, 100) is None and closed_ema_gap_pct([], 50, 100) is None
    assert closed_ema_gap_pct(_bars([1.0] * 50), 50, 100) is None            # too short for the slow EMA
    assert closed_ema_gap_pct([["x"] * 6] * 200, 50, 100) is None
    assert last_closed_bar_ret_pct(_bars([1.0, 2.0])) is None and last_closed_bar_ret_pct(None) is None
    assert last_closed_bar_ret_pct(_bars([0.0, 1.0, 2.0, 3.0])) == 100.0    # previous-close guard only trips on <= 0


def test_observe_only_and_parity():
    import config
    from models import Order
    cols = [c.name for c in Order.__table__.columns]
    for c in ("entry_btc_ema50_100_gap_pct", "entry_eth_5m_ret1_pct", "entry_btc_1d_ret_pct", "entry_pair_1h_ema20_200_gap_pct"):
        assert c in cols
    db = open(os.path.join(ROOT, "database.py")).read(); assert "entry_pair_1h_ema20_200_gap_pct" in db and "ADD COLUMN {_zc} FLOAT" in db
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count("entry_pair_1h_ema20_200_gap_pct=_z_pair_gap") == 1 and "get_ohlcv('ETH/USDT:USDT', '5m', 5)" in eng
    # deep review B3: the per-fill pair fetch sits AFTER the exchange order (between the fill and the Order row)
    _z = "'1h', 260), 8.0), 20, 200)"   # Oct-3: the read is bounded (the bot open lane is held there)
    assert eng.index(_z) > eng.index("binance_order_id = None") and eng.index(_z) < eng.index("        order = Order(\n            binance_order_id=binance_order_id,")
    assert "entry_btc_ema50_100_gap_pct=_g.get('_current_btc_ema50_100_gap_pct')" in eng and "put('entry_btc_1d_ret_pct'" in eng
    assert eng.count("getattr(config.trading_config, 'entry_zone_stamps_enabled', True)") == 2   # scan block + open_position only
    # observe-only: no line that touches a zone reading is a gate / sizing / signal statement
    for c in ("_current_btc_ema50_100_gap_pct", "_current_eth_5m_ret1_pct", "_current_btc_1d_ret_pct", "_z_pair_gap", "entry_pair_1h_ema20_200_gap_pct"):
        for ln in eng.splitlines():
            if c in ln and not ln.strip().startswith("#"):
                assert not any(k in ln for k in ("_record_filter_block", "signal = ", "cell_mult", "investment", "leverage", "return None")), ln
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json"))); assert cfg["entry_zone_stamps_enabled"] is True
    assert config.TradingConfig.model_fields["entry_zone_stamps_enabled"].default is True
    main = open(os.path.join(ROOT, "main.py")).read()
    assert "entry_zone_stamps_enabled: Optional[bool] = None" in main and main.count('"entry_pair_1h_ema20_200_gap_pct": getattr(') == 2
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-entry-zone-stamps-enabled") == 3 and "Zone stamps (Sep 29)" in ui


def test_every_open_position_call_site_uses_only_real_parameters():
    """Regression for the 'unexpected kwarg' species-kill class (deep review B1/B2): every keyword passed at any
    `open_position(` call site must be a real parameter of TradingEngine.open_position."""
    import ast, inspect, os
    os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
    from services.trading_engine import TradingEngine
    params = set(inspect.signature(TradingEngine.open_position).parameters)
    tree = ast.parse(open(os.path.join(ROOT, "services", "trading_engine.py")).read())
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "open_position":
            for kw in node.keywords:
                if kw.arg is not None and kw.arg not in params:
                    bad.append((node.lineno, kw.arg))
    assert not bad, bad
    for c in ("entry_btc_ema50_100_gap_pct", "entry_eth_5m_ret1_pct", "entry_btc_1d_ret_pct"):
        assert c in params                                       # the flip/spike/door paths pass them explicitly
