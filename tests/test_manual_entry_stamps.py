"""📐 Sep-29 manual fills record every entry stamp a momentum fill records (operator request) — same formulas as the scan,
exits unchanged, and the signed EMA gaps on every fill."""
import asyncio, os, re, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
import services.trading_engine as T
from services.indicators import calculate_indicators, gap_expand_marginal

# stamped only by a sleeve path, the liquidity cap, or set outside Order(): NULL on a momentum fill as on a manual one
NOT_A_MOMENTUM_STAMP = {
    "entry_br_r72", "entry_br_above", "entry_br_eff", "entry_br_off24h", "entry_br_door", "entry_br_door_age_min",
    "entry_br_bull_pct_top10", "entry_br_bear_pct_top10", "entry_br_top10_n",
    "entry_bear_r24", "entry_bear_below24", "entry_bear_eff24", "entry_bear_off24lo", "entry_bear_bypass",
    "entry_pair_1d_ndi", "entry_btc_4h_ema50_200_gap_pct", "entry_desired_notional", "entry_liquidity_cap_notional",
    "entry_btc_regime_started_at",
    "entry_surge_btc_move_pct", "entry_surge_pair_move_pct", "entry_surge_trigger_at", "entry_surge_gvol",   # ⚡ SURGE trigger stamps (sleeve-only)
    "entry_frenzy_spike_at", "entry_frenzy_hours", "entry_frenzy_vwap", "entry_frenzy_vs_vwap_pct", "entry_frenzy_vol_mult",
    "entry_frenzy_run_pct", "entry_frenzy_stop_atr", "entry_frenzy_bar_ret_pct", "entry_frenzy_di_spread", "entry_frenzy_adx_delta", "entry_frenzy_vol_trend", "entry_frenzy_gvol", "entry_frenzy_above_share", "entry_frenzy_above_streak", "entry_frenzy_catchup", "entry_frenzy_catchup_bars", "entry_frenzy_catchup_move_pct",          # 🔥 FRENZY episode stamps (sleeve-only)
    "entry_price", "entry_fee", "entry_order_type", "entry_strategy",   # set explicitly by open_manual_position
    "entry_bracket_max_leverage", "entry_bracket_cap_notional",   # 🪜 Oct-1: set explicitly in both open paths (exchange leverage brackets)
    "entry_slippage_pct",   # a manual paper fill IS the clicked price: left NULL so it never enters the slippage averages
    "entry_chop_burst_prior_fill_s",   # 🌀👥 Oct-4: the LONG_CHOP_BURST gate's own input — a bot decision stamp (DB lookup); MANUAL is never gated
}


def _bars(n, seed=3, drift=0.0):
    rng = np.random.default_rng(seed); c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.003, n)))
    h = c * (1 + rng.uniform(0, 0.002, n)); l = c * (1 - rng.uniform(0, 0.002, n)); o = np.r_[c[0], c[:-1]]
    return [[1_700_000_000_000 + i * 300_000, float(o[i]), float(h[i]), float(l[i]), float(c[i]), float(1000 + 40 * i)] for i in range(n)]


def _old_scan_formulas(indicators, signal):
    """The scan's inline expressions as they were before pair_entry_stamps (verbatim), for the parity check."""
    entry_gap = None
    if indicators.get('ema5') and indicators.get('ema20') and indicators['price'] > 0:
        entry_gap = round(abs((indicators['ema5'] - indicators['ema20']) / indicators['price'] * 100), 4)
    entry_ema_gap_5_8 = None
    if indicators.get('ema5') and indicators.get('ema8') and indicators['ema8'] > 0:
        entry_ema_gap_5_8 = round(abs((indicators['ema5'] - indicators['ema8']) / indicators['ema8'] * 100), 4)
    entry_ema_gap_8_13 = None
    if indicators.get('ema8') and indicators.get('ema13') and indicators['ema13'] > 0:
        entry_ema_gap_8_13 = round(abs((indicators['ema8'] - indicators['ema13']) / indicators['ema13'] * 100), 4)
    entry_ema5_stretch = None
    entry_price_vs_ema5_pct = None
    if indicators.get('ema5') and indicators['price'] > 0:
        entry_ema5_stretch = round(abs(indicators['price'] - indicators['ema5']) / indicators['price'] * 100, 4)
        entry_price_vs_ema5_pct = round((indicators['price'] - indicators['ema5']) / indicators['ema5'] * 100, 4)
    entry_rsi = indicators.get('rsi'); entry_rsi_prev = indicators.get('rsi_prev2')
    entry_adx = indicators.get('adx'); entry_adx_prev = indicators.get('adx_prev1')
    pair_ema20_slope_pct = None
    pair_ema20 = indicators.get('ema20'); pair_ema20_prev3 = indicators.get('ema20_prev3')
    if pair_ema20 and pair_ema20_prev3 and pair_ema20_prev3 != 0:
        pair_ema20_slope_pct = round(((pair_ema20 - pair_ema20_prev3) / pair_ema20_prev3) * 100, 4)
    _entry_atr_pct = None
    _atr = indicators.get('atr')
    if _atr is not None and indicators.get('price') and indicators['price'] > 0:
        _entry_atr_pct = round((_atr / indicators['price']) * 100, 4)
    _entry_ema50_slope = None
    _ema50 = indicators.get('ema50'); _ema50_prev12 = indicators.get('ema50_prev12')
    if _ema50 is not None and _ema50_prev12 is not None and _ema50_prev12 != 0:
        _entry_ema50_slope = round(((_ema50 - _ema50_prev12) / _ema50_prev12) * 100, 4)
    _gap13_50 = None
    _ema13_val = indicators.get('ema13')
    if _ema13_val is not None and _ema50 is not None and _ema50 != 0:
        _gap13_50 = round((_ema13_val - _ema50) / _ema50 * 100, 4)
    _dist13 = None
    _entry_price = indicators.get('price')
    if _ema13_val is not None and _entry_price is not None and _ema13_val != 0:
        _dist13 = round((_entry_price - _ema13_val) / _ema13_val * 100, 4)
    _pv = indicators.get('volume') or 0; _pa = indicators.get('avg_volume') or 0
    _pvr = round(_pv / _pa, 4) if _pa > 0 else 1.0
    return dict(
        entry_gap=entry_gap, entry_ema_gap_5_8=entry_ema_gap_5_8, entry_ema_gap_8_13=entry_ema_gap_8_13, entry_ema5_stretch=entry_ema5_stretch,
        entry_rsi=round(entry_rsi, 2) if entry_rsi is not None else None,
        entry_rsi_prev=round(entry_rsi_prev, 2) if entry_rsi_prev is not None else None,
        entry_adx=round(entry_adx, 4) if entry_adx is not None else None,
        entry_adx_prev=round(entry_adx_prev, 4) if entry_adx_prev is not None else None,
        entry_ema20_slope=pair_ema20_slope_pct, entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
        entry_pair_volume_ratio=round(_pvr, 4),
        entry_range_position=round(((indicators['price'] - indicators['low_20']) / (indicators['high_20'] - indicators['low_20'])) * 100, 1) if indicators.get('high_20') and indicators.get('low_20') and indicators['high_20'] != indicators['low_20'] else None,
        entry_adx_delta=round(entry_adx - entry_adx_prev, 4) if entry_adx is not None and entry_adx_prev is not None else None,
        entry_pos_di=indicators.get('pos_di'), entry_neg_di=indicators.get('neg_di'), entry_atr_pct=_entry_atr_pct,
        entry_ema50_slope=_entry_ema50_slope, entry_pair_ema20_ema50_gap_pct=_gap13_50, entry_dist_from_ema13_pct=_dist13,
        entry_gap_expand_marginal=gap_expand_marginal(indicators, signal),
    )


def test_pair_stamps_equal_the_scan_formulas_they_replaced():
    """The scan now passes **pair_entry_stamps(...) — every value must equal what the inline expressions passed before."""
    n = 0
    for seed in range(40):
        for drift in (-0.001, 0.0, 0.001):
            ind = calculate_indicators(_bars(100, seed, drift))
            for sig in ("LONG", "SHORT"):
                new = T.pair_entry_stamps(ind, sig); old = _old_scan_formulas(ind, sig)
                for k, v in old.items():
                    assert new[k] == v, (seed, drift, sig, k, new[k], v)
                n += 1
    assert n == 240
    # degenerate inputs never raise
    ind = calculate_indicators(_bars(100)); ind.update(high_20=ind['low_20'], atr=None, ema50_prev12=0.0, avg_volume=0)
    assert T.pair_entry_stamps(ind, "LONG")['entry_range_position'] is None and T.pair_entry_stamps(ind, "LONG")['entry_pair_volume_ratio'] == 1.0
    assert all(v is None for k, v in T.pair_entry_stamps({}, "LONG").items() if k != 'entry_pair_volume_ratio')


def test_signed_gaps_keep_the_sign_the_absolute_stamps_lose():
    ind = dict(price=100.0, ema5=99.0, ema8=98.5, ema13=98.8, ema20=99.4, ema5_prev1=98.8, ema20_prev1=99.45)
    st = T.pair_entry_stamps(ind, "LONG")
    assert st['entry_gap'] == 0.4 and st['entry_gap_5_20_signed_pct'] == -0.4          # EMA5 under EMA20: the early-turn case
    assert st['entry_gap_5_20_prev_signed_pct'] == -0.65                                # one bar earlier: more negative → rising
    assert st['entry_gap_5_8_signed_pct'] == round((99.0 - 98.5) / 98.5 * 100, 4) > 0
    # the scan's open call and the Order carry them for bot fills too
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("**pair_entry_stamps(indicators, signal),") == 1
    assert "entry_gap_5_20_signed_pct=entry_gap_5_20_signed_pct, entry_gap_5_20_prev_signed_pct=entry_gap_5_20_prev_signed_pct," in eng
    models = open(os.path.join(ROOT, "models.py"), encoding="utf-8").read(); db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    for c in ("entry_gap_5_20_signed_pct", "entry_gap_5_20_prev_signed_pct", "entry_gap_5_8_signed_pct"):
        assert f"{c} = Column(Float" in models and f"'{c}'" in db
    assert "'ema20_prev1':" in open(os.path.join(ROOT, "services", "indicators.py"), encoding="utf-8").read()


def test_manual_fill_records_every_field_a_momentum_fill_records(monkeypatch):
    import models
    bars5, bars1h = _bars(100, 11, 0.0005), _bars(260, 12, 0.0005)

    async def _ohlcv(symbol, tf, n): return (bars5 if tf == '5m' else bars1h)[-n:]
    async def _fr(symbol): return 0.00012345
    monkeypatch.setattr(T.binance_service, "get_ohlcv", _ohlcv)
    monkeypatch.setattr(T.binance_service, "fetch_funding_rate", _fr)
    from services import mcap_service
    monkeypatch.setattr(mcap_service, "get", lambda p: (1.5e9, 88))
    for k, v in dict(_current_btc_adx=24.123456, _current_btc_adx_prev=23.9, _current_btc_rsi=57.26, _current_btc_rsi_prev=56.04,
                     _current_btc_rsi_prev6=54.0, _current_btc_atr_pct=0.18, _current_btc_rsi_1h=55.0, _current_btc_rsi_1h_prev=54.2,
                     _btc_ema20_slope_pct=0.031, _btc_ema20_slope_pct_raw=0.031, _current_btc_1h_slope=0.05, _current_btc_trend_gap_pct=0.12, _market_bull_pct=61.0,
                     _market_bear_pct=22.0, _global_volume_ratio=1.123456, _current_btc_ema13=100.0, _current_btc_price=100.3,
                     _current_btc_regime="BULLISH", _current_btc_ema50_100_gap_pct=0.2, _current_eth_5m_ret1_pct=0.05,
                     _current_btc_1d_ret_pct=1.1, _zone_stamps_at=time.time(),
                     _current_btc_rsi_closed=61.5, _current_btc_rsi_closed_bar_ts=int(time.time() * 1000) - 300_000).items():   # 🩹 Oct-2 recovery-hold ruler
        monkeypatch.setattr(T, k, v, raising=False)
    now = T._leash_time.time()
    for k, v in dict(off24h=-0.8, r72=2.1, eff=0.4, above=63.0, updated_at=now, off30d=-6.0, off30d_at=now).items():
        monkeypatch.setitem(T._bullrun_monitor, k, v)
    for k, v in dict(off24lo=1.9, updated_at=now).items():
        monkeypatch.setitem(T._bearrun_monitor, k, v)
    eng = T.TradingEngine.__new__(T.TradingEngine); eng.is_paper_mode = True
    eng._scan_pair_meta = {"QNTUSDT": (2.5e8, 17, 900.0)}; eng._scan_pair_meta_at = time.time()
    st = asyncio.run(eng._manual_entry_stamps("QNTUSDT", "QNT/USDT:USDT", "LONG", 268.0))

    cols = [c.name for c in models.Order.__table__.columns if c.name.startswith("entry_")]
    expected = [c for c in cols if c not in NOT_A_MOMENTUM_STAMP]
    missing = [c for c in expected if c not in st]
    assert not missing, f"manual fill does not record: {missing}"
    empty = [c for c in expected if st[c] is None and c != "entry_gap_expand_marginal"]   # the tag is None by design when undefined
    assert not empty, f"recorded but empty with full data: {empty}"
    assert not [c for c in NOT_A_MOMENTUM_STAMP if c in st], "sleeve-only fields must stay NULL on a manual fill"
    # the scan's rounding
    assert st['entry_btc_adx'] == 24.1235 and st['entry_btc_rsi'] == 57.3 and st['entry_global_volume_ratio'] == 1.1235
    assert st['entry_pair_rank'] == 17 and st['entry_funding_rate'] == 0.000123 and st['entry_cmc_rank'] == 88
    assert st['exit_btc_regime'] == st['entry_btc_regime'] and 'entry_slippage_pct' not in st
    models.Order(**st)                                                                    # every key is a real column

    # fail-soft: no klines, no funding, no scan meta → the pair fields are empty, nothing raises, market fields still land
    async def _none(*a, **k): return None
    async def _boom(*a, **k): raise RuntimeError("exchange down")
    monkeypatch.setattr(T.binance_service, "get_ohlcv", _boom); monkeypatch.setattr(T.binance_service, "fetch_funding_rate", _none)
    eng._scan_pair_meta = {}
    st2 = asyncio.run(eng._manual_entry_stamps("QNTUSDT", "QNT/USDT:USDT", "LONG", 268.0))
    assert st2.get('entry_rsi') is None and st2['entry_btc_adx'] == 24.1235 and st2.get('entry_pair_rank') is None


def test_manual_stops_never_see_the_entry_atr():
    """Recording the ATR must not WIDEN a manual trade's stop (ATR stop widening / quiet-pair stop): at 30–50× a widened
    stop could sit past the leverage-aware floor. The profit side (runner floor, trails) reads the real ATR — hiding it there
    disabled the armed-runner lock (🩹 Sep-30, see test_manual_runner_lock.py)."""
    assert T.exit_entry_atr_pct("MANUAL", 0.9) is None
    assert T.exit_entry_atr_pct("MOMENTUM", 0.9) == 0.9 and T.exit_entry_atr_pct(None, 0.9) == 0.9 and T.exit_entry_atr_pct("SPIKE_FADE", 1.2) == 1.2
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def update_open_positions"); j = eng.index("\n    async def ", i + 10); body = eng[i:j]
    assert "sl_atr_pct=exit_entry_atr_pct(order.entry_strategy, getattr(order, 'entry_atr_pct', None))" in body          # candle stop
    assert "quiet_sl_pct=_quiet_sl_for(order.direction, order.entry_strategy,\n                                           exit_entry_atr_pct(" in body
    assert "_entry_atr_pct = order_info.get('sl_entry_atr_pct', order_info.get('entry_atr_pct'))" in eng                  # realtime stop

def test_every_pair_stamp_is_an_open_position_parameter():
    """A key added to pair_entry_stamps but not to open_position's signature would TypeError every scan open."""
    import inspect
    params = set(inspect.signature(T.TradingEngine.open_position).parameters)
    keys = set(T.pair_entry_stamps(calculate_indicators(_bars(100)), "LONG"))
    assert keys <= params, keys - params
    assert set(T.pair_entry_stamps({}, "LONG")) == keys                    # same key set whatever the inputs


def test_manual_stamp_reads_run_together_and_use_the_click_price(monkeypatch):
    bars5, bars1h = _bars(100, 21), _bars(260, 22)

    async def _ohlcv(symbol, tf, n):
        await asyncio.sleep(0.3); return (bars5 if tf == '5m' else bars1h)[-n:]
    async def _fr(symbol):
        await asyncio.sleep(0.3); return 0.0001
    monkeypatch.setattr(T.binance_service, "get_ohlcv", _ohlcv)
    monkeypatch.setattr(T.binance_service, "fetch_funding_rate", _fr)
    eng = T.TradingEngine.__new__(T.TradingEngine); eng.is_paper_mode = True
    t0 = time.time(); st = asyncio.run(eng._manual_entry_stamps("QNTUSDT", "QNT/USDT:USDT", "LONG", 123.0))
    assert time.time() - t0 < 0.8                                          # three 0.3 s reads: concurrent, not 0.9 s
    ind = calculate_indicators(bars5)
    assert st['entry_price_vs_ema5_pct'] == round((123.0 - ind['ema5']) / ind['ema5'] * 100, 4)   # priced at the click
    assert st['entry_funding_rate'] == 0.0001 and st['entry_pair_1h_ema20_200_gap_pct'] is not None
    # no pair indicators → no macro-trend reading (never an invented NEUTRAL)
    assert T.market_entry_stamps({}, "LONG", None, False, T.config.trading_config.thresholds)['entry_macro_trend'] is None


def test_manual_open_rechecks_the_pair_right_before_the_insert():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def open_manual_position"); j = eng.index("\n    async def ", i + 10); body = eng[i:j]
    a = body.index("_st, _obm = await asyncio.gather(self._manual_entry_stamps("); b = body.index("was opened by the bot while this manual entry was being prepared"); c = body.index("order = Order(")
    assert a < b < c and "opened_at=_clicked_at" in body


def test_one_hung_read_does_not_cancel_the_others(monkeypatch):
    """Deep review: under one gather-wide timeout a hung funding read wiped the pair stamps too."""
    bars5, bars1h = _bars(100, 31), _bars(260, 32)
    async def _ohlcv(symbol, tf, n): return (bars5 if tf == '5m' else bars1h)[-n:]
    async def _hang(symbol): await asyncio.sleep(30)
    monkeypatch.setattr(T.binance_service, "get_ohlcv", _ohlcv)
    monkeypatch.setattr(T.binance_service, "fetch_funding_rate", _hang)
    real_wait_for = asyncio.wait_for
    monkeypatch.setattr(T.asyncio, "wait_for", lambda coro, timeout: real_wait_for(coro, 0.3 if timeout < 8 else 2.0))   # per-read cap 0.3 s, overall 2 s
    eng = T.TradingEngine.__new__(T.TradingEngine); eng.is_paper_mode = True
    st = asyncio.run(eng._manual_entry_stamps("QNTUSDT", "QNT/USDT:USDT", "LONG", 100.0))
    assert st['entry_rsi'] is not None and st['entry_atr_pct'] is not None and st['entry_pair_1h_ema20_200_gap_pct'] is not None
    assert st['entry_funding_rate'] is None
    eng_src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng_src.index("    async def open_manual_position"); j = eng_src.index("\n    async def ", i + 10); body = eng_src[i:j]
    k = body.index("_st, _obm = await asyncio.gather(self._manual_entry_stamps(")
    assert "manual positions cap reached while this entry was being prepared" in body[k:] and "exceeds the available balance" in body[k:]


def test_manual_balance_check_counts_the_fee_the_bnb_reserve_cannot_pay():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    i = eng.index("    async def open_manual_position"); body = eng[i:eng.index("\n    async def ", i + 10)]
    assert "_fee_from_usdt = max(0.0, entry_fee - max(0.0, float(await self._recalculate_paper_bnb(db))))" in body
    assert "if investment + _fee_from_usdt > _avail_final:" in body


def test_manual_stamps_prefer_the_scans_raw_btc_slope():
    th = T.config.trading_config.thresholds
    g = dict(_btc_ema20_slope_pct=0.0, _btc_ema20_slope_pct_raw=None, _current_btc_adx=24.0, _current_btc_rsi=55.0)
    assert T.market_entry_stamps(g, "LONG", None, False, th)['entry_btc_ema20_slope'] is None      # unknown stays unknown
    g["_btc_ema20_slope_pct_raw"] = 0.05
    assert T.market_entry_stamps(g, "LONG", None, False, th)['entry_btc_ema20_slope'] == 0.05
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "_btc_ema20_slope_pct_raw = btc_ema20_slope_pct" in eng


def test_manual_fee_aware_balance_refuses_only_when_the_fee_overflows(monkeypatch):
    """Behavioural: $1,000 × 40 = $40k notional → ~$20 taker fee. Free USDT $1,010: fits when BNB pays the fee, refused when
    the BNB reserve is empty (the fee would come out of USDT)."""
    e = T.TradingEngine.__new__(T.TradingEngine); e.is_paper_mode = True
    async def bal(db): return 1_010.0
    e.get_available_balance = bal
    class _Res:
        def scalar(self): return 0
    class _DB:
        async def execute(self, *a, **k): return _Res()
    class Past(Exception): pass
    async def boom(*a, **k): raise Past()
    monkeypatch.setattr(T.binance_service, "get_current_price", boom)
    monkeypatch.setattr(T.websocket_tracker, "get_tracker", lambda p: None)
    def attempt(bnb):
        async def _bnb(db): return bnb
        e._recalculate_paper_bnb = _bnb
        try:
            asyncio.run(e.open_manual_position(_DB(), pair="QNTUSDT", direction="LONG", investment=1000, leverage=40,
                                               exit_mode="FIXED", sl_pct=1.0))
        except ValueError as err:
            return str(err)
        except Past:
            return "passed the balance check"
    assert attempt(100.0) == "passed the balance check"
    r = attempt(0.0); assert "fee not covered by BNB" in r and "exceeds available balance" in r
