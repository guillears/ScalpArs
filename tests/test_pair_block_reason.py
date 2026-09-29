"""Sep-29: every recorded filter block stamps the pair in context (Top Pairs 'Block Reason' = the gate that refused the setup)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")


def _eng():
    import services.trading_engine as T
    e = T.TradingEngine.__new__(T.TradingEngine)
    e._filter_block_counts = {}; e._last_pair_block_reason = T.PairReasonStash(); e._journal_ctx = None
    e._last_pair_block_reason.cur_seq = 1
    return e


def test_recorded_block_stamps_the_pair_in_context():
    e = _eng(); e._journal_pair = "QNTUSDT"; e._last_pair_block_reason["QNTUSDT"] = "No EMA Stack"
    e._record_filter_block("BTC_ADX_GATE_LOW", "LONG")
    assert e._last_pair_block_reason["QNTUSDT"] == "BTC_ADX_GATE_LOW"
    assert sum(v for k, v in e._filter_block_counts.items() if k[0] == "BTC_ADX_GATE_LOW") == 1


def test_no_pair_in_context_stamps_nothing_and_never_raises():
    e = _eng(); e._journal_pair = None
    e._record_filter_block("SPIKE_FADE_LAGGARD", "SHORT")
    assert e._last_pair_block_reason == {}
    e2 = _eng(); del e2._last_pair_block_reason; e2._journal_pair = "X"
    e2._record_filter_block("ANY_GATE", "LONG")           # missing stash must not raise
    e._record_filter_block("", "LONG")                    # empty name = no-op
    assert e._last_pair_block_reason == {}


def test_ui_marks_a_rated_but_blocked_setup():
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert "p.entry_ready === false" in ui and "🚫</span>' + _confSafe" in ui and "whitespace-nowrap ${_confBlocked" in ui and "const _brSafe = _escAttr(p.block_reason)" in ui


def test_first_decisive_gate_wins_and_other_sleeves_never_stamp():
    """Caveman review: a counter recorded LATER in the same pair iteration (flip / bull-run / spike / open failure) must not
    rename the gate that refused the rated setup; another sleeve's refusal never names the momentum reason."""
    import services.trading_engine as T
    e = _eng(); e._journal_pair = "QNTUSDT"; e._last_pair_block_reason["QNTUSDT"] = T.PAIR_REASON_PLACEHOLDER
    e._record_filter_block("BULLRUN_BTC_EMA13", "LONG")                       # Phase-1 bull-run hook on a no-stack pair
    assert e._last_pair_block_reason["QNTUSDT"] == T.PAIR_REASON_PLACEHOLDER
    e._record_filter_block("FAN_RATIO_GATE", "LONG")                          # the decisive momentum gate
    for later in ("FLIP_SHORT_QUALITY", "OPEN_FAILED_FLIP", "SPIKE_FADE_LAGGARD", "OPEN_FAILED_BOUNCE_LONG", "REDEPLOY_OPEN", "GROSS_CAP_SKIP", "BTC_ADX_GATE_LOW"):
        e._record_filter_block(later, "SHORT")
    assert e._last_pair_block_reason["QNTUSDT"] == "FAN_RATIO_GATE"
    assert sum(e._filter_block_counts.values()) == 9                          # every counter still counted
    assert T.pair_reason_stampable("BTC_ADX_GATE_LOW") and not T.pair_reason_stampable("BR_PAIR_BLACKLIST") and not T.pair_reason_stampable("")
    assert T.pair_reason_stampable("SPIKE_GUARD") and not T.pair_reason_stampable("SPIKE_FADE_BRSI") and not T.pair_reason_stampable("LONG_HEAT_FAILOPEN")


def test_sleeve_scope_and_scan_sequence():
    """A counter recorded while open_position works for another sleeve never stamps; the previous scan's reason stays on display
    until this scan stamps; a reason belongs to the verdict only when stamped in the verdict's scan (late gates included)."""
    import services.trading_engine as T
    e = _eng(); st = e._last_pair_block_reason; e._journal_pair = "QNTUSDT"
    e._open_ctx_momentum = False; e._record_filter_block("COOLDOWN", "SHORT")          # a flip's open_position hit the cooldown
    assert "QNTUSDT" not in st
    e._open_ctx_momentum = True; e._record_filter_block("BTC_ADX_GATE_LOW", "LONG"); st.mark_verdict("QNTUSDT")
    assert st["QNTUSDT"] == "BTC_ADX_GATE_LOW" and st.reason_for_verdict("QNTUSDT") == "BTC_ADX_GATE_LOW"
    st.cur_seq += 1                                                                     # next scan begins
    assert st["QNTUSDT"] == "BTC_ADX_GATE_LOW" and not st.stamped_this_scan("QNTUSDT")  # still displayed, no placeholder window
    e._record_filter_block("FAN_RATIO_GATE", "LONG")                                    # new scan's first decisive gate replaces it
    assert st["QNTUSDT"] == "FAN_RATIO_GATE"
    assert st.reason_for_verdict("QNTUSDT") is None                                     # verdict of THIS scan not written yet
    st.mark_verdict("QNTUSDT"); assert st.reason_for_verdict("QNTUSDT") == "FAN_RATIO_GATE"
    st.cur_seq += 1; st.mark_verdict("QNTUSDT")                                         # a later scan admits the pair: old reason no longer applies
    assert st.reason_for_verdict("QNTUSDT") is None
    st["SOLUSDT"] = T.PAIR_REASON_PLACEHOLDER; st.mark_verdict("SOLUSDT"); assert st.reason_for_verdict("SOLUSDT") is None


def test_pairs_route_marks_late_gate_refusals(monkeypatch):
    """/api/pairs: a rated setup refused by a gate that runs after the PairData write is not entry-ready and carries the gate."""
    import asyncio, types, main, services.trading_engine as T
    from datetime import datetime
    st = T.PairReasonStash(); st.cur_seq = 5
    st["AAAUSDT"] = "LONG_HEAT_BLOCK"; st.mark_verdict("AAAUSDT")                       # late gate, same scan as the verdict
    st.mark_verdict("BBBUSDT")                                                          # clean admit
    st["CCCUSDT"] = "BTC_ADX_GATE_LOW"; st.mark_verdict("CCCUSDT")
    monkeypatch.setattr(T, "trading_engine", types.SimpleNamespace(_last_pair_block_reason=st), raising=False)
    def row(pair, signal, conf):
        return types.SimpleNamespace(pair=pair, price=1.0, ema5=1.0, ema8=1.0, ema13=1.0, ema20=1.0, rsi=50.0, adx=20.0, signal=signal, confidence=conf,
                                     macro_regime="NEUTRAL", volume_24h=1e8, updated_at=datetime.utcnow(), volume_ratio=1.0, avg_volume=1.0)
    rows = [row("AAAUSDT", "LONG", "VERY_STRONG"), row("BBBUSDT", "LONG", "STRONG_BUY"), row("CCCUSDT", "NO_TRADE", "STRONG_BUY")]

    class _R:
        def __init__(self, v): self.v = v
        def scalars(self): return self
        def all(self): return self.v
        def __iter__(self): return iter(self.v)

    class _DB:
        def __init__(self): self.n = 0
        async def execute(self, *a, **k):
            self.n += 1
            return _R(rows) if self.n == 1 else _R([])
    out = asyncio.run(main.get_pairs(db=_DB(), limit=50))
    by = {r["pair"]: r for r in out}
    assert by["AAAUSDT"]["entry_ready"] is False and by["AAAUSDT"]["block_reason"] == "LONG_HEAT_BLOCK"
    assert by["BBBUSDT"]["entry_ready"] is True and by["BBBUSDT"]["block_reason"] is None
    assert by["CCCUSDT"]["entry_ready"] is False and by["CCCUSDT"]["block_reason"] == "BTC_ADX_GATE_LOW"
    assert "setup_side" in by["AAAUSDT"]


def test_dashboard_analytics_never_include_manual_rows():
    """Behavioural (verification review): compute the performance payload on a scratch in-memory DB with 8 systematic fills, add 3
    MANUAL fills carrying BTC/breadth stamps and the STRONG_BUY label, recompute — only ACCOUNT-LEVEL keys may change. Any
    analytic table moving = a manual fill leaked into a read of the systematic book."""
    import asyncio, json, datetime as dt
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    import main, models, services.trading_engine as T

    ACCOUNT_LEVEL = {"total_trades", "total_longs", "total_shorts", "total_wins", "total_losses", "win_rate", "win_rate_longs", "win_rate_shorts",
                     "avg_win", "avg_win_long", "avg_win_short", "avg_loss", "avg_loss_long", "avg_loss_short", "best_win_long", "best_win_short",
                     "worst_loss_long", "worst_loss_short", "total_pnl", "total_pnl_percentage", "total_pnl_notional_percentage",
                     "total_investment_notional", "total_investment_value", "total_investment_long_notional", "total_investment_long_value",
                     "total_investment_short_notional", "total_investment_short_value", "total_fees", "avg_duration", "avg_duration_long",
                     "avg_duration_short", "avg_leverage", "return_multiple", "daily_compound_return", "runtime_days",
                     "avg_win_pct", "avg_win_long_pct", "avg_win_short_pct", "avg_loss_pct", "avg_loss_long_pct", "avg_loss_short_pct", "expectancy",
                     "period_performance", "equity_curve", "sleeve_performance", "strategy_performance", "hourly_performance", "daily_performance",
                     "day_time_heatmap", "performance_over_time", "pair_performance"}

    def order(i, strat, direction, pnl_pct, **kw):
        t0 = dt.datetime(2026, 9, 20, 10, 0, 0) + dt.timedelta(hours=3 * i)
        base = dict(pair=f"P{i}USDT", direction=direction, status="CLOSED", entry_price=100.0, exit_price=100.0 * (1 + pnl_pct / 100), investment=500.0,
                    leverage=20.0, notional_value=10000.0, quantity=100.0, confidence="STRONG_BUY", entry_strategy=strat, is_paper=True,
                    pnl=pnl_pct * 100.0, pnl_percentage=pnl_pct, peak_pnl=max(pnl_pct, 0.1), trough_pnl=min(pnl_pct, -0.1), entry_fee=4.5, total_fee=9.0,
                    opened_at=t0, closed_at=t0 + dt.timedelta(minutes=17), close_reason=("RUNNER_TRAIL L1" if pnl_pct > 0 else "STOP_LOSS L1"),
                    entry_btc_rsi=55.0 + i, entry_btc_adx=24.0 + i % 3, entry_btc_ema20_slope=0.03, entry_btc_atr_pct=0.15, entry_btc_rsi_1h=52.0,
                    entry_btc_1h_slope=0.03, entry_bull_pct=62.0, entry_bear_pct=20.0, entry_macro_trend="BULLISH", entry_btc_trend_gap_pct=0.1,
                    cell_multiplier=1.0, cell_lev_multiplier=1.0)
        base.update(kw); return models.Order(**base)

    async def run():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        Session = async_sessionmaker(eng, expire_on_commit=False)
        main.trading_engine.is_paper_mode = True
        async with Session() as db:
            for i, (d, p) in enumerate([("LONG", 0.5), ("LONG", -0.7), ("SHORT", 0.4), ("LONG", 0.3), ("SHORT", -0.7), ("LONG", 0.6), ("SHORT", 0.2), ("LONG", -0.4)]):
                db.add(order(i, "MOMENTUM", d, p, entry_rsi=55.0, entry_adx=25.0, entry_gap=0.3, entry_ema_gap_5_8=0.08, entry_ema_gap_8_13=0.09,
                             entry_atr_pct=0.5, entry_range_position=60.0, entry_btc_regime="HEALTHY_BULL", cell_multiplier_source="UNMATCHED"))
            await db.commit()
            before = json.loads(json.dumps(await main._compute_performance(db), default=str))
            for j, (d, p) in enumerate([("LONG", 0.4), ("SHORT", 0.3), ("LONG", -0.6)]):
                db.add(order(20 + j, "MANUAL", d, p, manual_exit_mode="MOMENTUM", manual_block_reason="ATR_GAP_LONG", manual_setup_rating="STRONG_BUY",
                             manual_setup_side="LONG", manual_pair_rsi=54.0, manual_pair_adx=17.0, manual_gap_5_20=0.33))
            await db.commit()
            after = json.loads(json.dumps(await main._compute_performance(db), default=str))
        await eng.dispose()
        return before, after
    before, after = asyncio.run(run())
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed, "the manual fills must at least move the account totals"
    leaks = sorted(changed - ACCOUNT_LEVEL)
    assert not leaks, f"manual fills leaked into systematic analytics: {leaks}"
    assert {"total_trades", "sleeve_performance", "strategy_performance"} <= changed
