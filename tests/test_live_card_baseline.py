"""🏦 Live portfolio card: fixed baseline maths + wiring (pure, no network)."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _fn():
    src = open(os.path.join(ROOT, "main.py")).read()
    a = src.index("def live_card_numbers("); b = src.index("\n\n\n", a)
    ns = {}; exec(src[a:b], ns); return ns["live_card_numbers"], src


def _split():
    src = open(os.path.join(ROOT, "main.py")).read()
    a = src.index("def split_flow_rows("); b = src.index("\n\n\n", a)
    ns = {}; exec(src[a:b], ns); return ns["split_flow_rows"]


def test_deposits_and_withdrawals_are_never_pnl():
    f, _ = _fn()
    assert f(3000.0, 3000.0, 0.0) == (3000.0, 0.0)
    assert f(3500.0, 3000.0, 500.0) == (3000.0, 0.0)            # a $500 deposit: equity up, P&L flat
    assert f(2700.0, 3000.0, -300.0) == (3000.0, 0.0)           # a $300 withdrawal
    assert f(3620.0, 3000.0, 500.0) == (3000.0, 120.0)          # deposit + $120 made
    assert f(2337.26, 3000.0, 0.0) == (3000.0, -662.74)


def test_missing_inputs_hide_the_lines():
    f, _ = _fn()
    assert f(None, 3000.0, 0.0) == (None, None) and f(3000.0, None, 0.0) == (None, None) and f(3000.0, 3000.0, None) == (None, None)


def test_wiring_model_migration_reset():
    _, src = _fn()
    mdl = open(os.path.join(ROOT, "models.py")).read(); dbs = open(os.path.join(ROOT, "database.py")).read()
    assert "live_initial_total_usd = Column(Float, nullable=True)" in mdl and "live_baseline_at = Column(DateTime, nullable=True)" in mdl
    assert "ADD COLUMN live_initial_total_usd FLOAT" in dbs and "ADD COLUMN live_baseline_at DATETIME" in dbs
    assert src.count("await _live_baseline(db, total, _live_open)") == 1
    lb = src[src.index("async def _live_baseline("):src.index("_LIVE_FLOW_MEMO = {}")]
    assert "await locked_commit(db)" in lb and "await db.commit()" not in lb and "if not (base > 1.0):" in lb
    assert "_bs.live_initial_total_usd = None; _bs.live_baseline_at = None" in src        # only a LIVE full reset clears it
    assert src.index("_bs.live_initial_total_usd = None") > src.index("if is_paper:\n            trading_engine.paper_balance = config.trading_config.paper_balance")


def test_flow_rows_are_stored_once_and_new_ones_stay_pending():
    f = _split(); rows = [(100, 500.0), (200, -300.0), (900, 50.0)]
    assert f(rows, 0, 500) == (200.0, 200, 50.0)                 # two settled, the newest still pending
    assert f(rows, 200, 500) == (0.0, 200, 50.0)                 # already stored rows are never added twice
    assert f(rows, 200, 1000) == (50.0, 900, 0.0)                # the pending one settles later, once
    assert f([], 200, 1000) == (0.0, 200, 0.0) and f(None, None, 10) == (0.0, 0, 0.0)
    s1, c1, p1 = f(rows, 0, 500); s2, c2, p2 = f(rows, c1, 1000)
    assert s1 + s2 == sum(a for _, a in rows) and p2 == 0.0      # stored in two steps == the full sum


def test_flows_wiring():
    src = open(os.path.join(ROOT, "main.py")).read()
    mdl = open(os.path.join(ROOT, "models.py")).read(); dbs = open(os.path.join(ROOT, "database.py")).read()
    assert "live_net_flows_usd = Column(Float, nullable=True)" in mdl and "live_flows_cursor_ms = Column(Integer, nullable=True)" in mdl
    assert "ADD COLUMN live_net_flows_usd FLOAT" in dbs and "ADD COLUMN live_flows_cursor_ms INTEGER" in dbs
    assert src.count("_live_flows = await _live_net_flows(db, _base_at)") == 1
    assert "_bs.live_net_flows_usd = None; _bs.live_flows_cursor_ms = None" in src
    nf = src[src.index("async def _live_net_flows("):src.index("async def _live_baseline(")]
    assert "await locked_commit(db)" in nf and "await db.commit()" not in nf
    assert "BotState.live_baseline_at == base_at" in nf and "BotState.live_flows_cursor_ms == old" in nf and "res.rowcount != 1" in nf   # compare-and-swap
    assert "now_ms - 89 * 86_400_000" in nf
    import re
    assert int(re.search(r"LIVE_FLOW_SETTLE_MS = (\d+) \* 60 \* 1000", src).group(1)) * 60 > 600      # settle lag > the 600 s transfer cache
