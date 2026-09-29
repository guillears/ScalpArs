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
    assert "p.entry_ready === false" in ui and "_confBlocked ? '🚫 ' + _confSafe" in ui and "const _brSafe = _escAttr(p.block_reason)" in ui


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
