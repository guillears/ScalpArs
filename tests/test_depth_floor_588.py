"""📖💵 DECISION_LOG 588 — the depth guard's $ FLOOR mode (entry_depth_guard_min_usd > 0): a guarded entry shrinks to what the book fills
(thinner side binds) and is skipped only below the $ floor / the exchange minimum; min_usd 0 = the DL 551 / 583 fraction rule unchanged."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import entry_depth as D  # noqa: E402
from tests.test_depth_impact_584 import _open, _plan, CAPS  # noqa: E402


def test_pure_rule():
    assert D.depth_guard_size(30_500, 7_000, 0.25, 200) == (D.SHRINK, 7_000)          # a $50k account at 0.61 × equity on a $7k book → $7k
    assert D.depth_guard_size(30_500, 7_000, 0.25, 0) == (D.SKIP, 7_000)               # the fraction rule would skip it (7k < 0.25 × 30.5k)
    assert D.depth_guard_size(30_500, 150, 0.25, 200) == (D.SKIP, 150)                 # below the $ floor → skipped
    assert D.depth_guard_size(30_500, 40_000, 0.25, 200) == (D.FULL, 30_500)
    assert D.depth_guard_size(1_000, 600, 0.25, 0) == (D.SHRINK, 600) and D.depth_guard_size(1_000, 200, 0.25, 0) == (D.SKIP, 200)   # unchanged
    assert D.depth_guard_size(1_000, None, 0.25, 200) == (D.FULL, 1_000)               # unreadable book: fail-open as before
    assert D.exchange_min_notional({'limits': {'cost': {'min': 5}, 'amount': {'min': 10}}}, 2.0) == 20.0
    assert D.exchange_min_notional(None, 1.0) == 5.0


def test_open_path_shrinks_below_a_quarter_instead_of_skipping(monkeypatch):
    n = _plan(monkeypatch)
    deep = [[1.0, 1e12]]
    assert n > 1_000
    asks = [[1.0001, 0.2 * n / 1.0001], [1.05, 1e12]]                                    # the exit side holds 0.2 × the plan (> $200, < 0.25 ×)
    r, blocks = _open(monkeypatch, deep, asks=asks, mode="sweep", min_usd=200.0)
    assert r is not None and "DEPTH_GUARD_SKIP" not in blocks
    assert r.notional_value == pytest.approx(r.entry_cap_exit_side_usd, rel=1e-6) and r.entry_cap_binding_side == "EXIT"
    assert r.entry_cap_frac == pytest.approx(r.notional_value / n, rel=1e-3) and r.entry_cap_frac < 0.25
    assert CAPS[-1]['cap_action'] == "SHRINK"
    r0, blocks0 = _open(monkeypatch, deep, asks=asks, mode="sweep", min_usd=0.0)        # min_usd 0 → the old fraction rule: skipped
    assert r0 is None and "DEPTH_GUARD_SKIP" in blocks0


def test_open_path_entry_side_binds_too(monkeypatch):
    n = _plan(monkeypatch)
    bids = [[1.0, n * 0.2], [0.98, 1e12]]                                                # the ENTRY side (a SHORT sells the bids) holds 0.2 × plan
    r, blocks = _open(monkeypatch, bids, mode="sweep", min_usd=200.0)
    assert r is not None and r.entry_cap_binding_side == "ENTRY" and r.notional_value == pytest.approx(n * 0.2, rel=1e-6)


def test_open_path_skips_below_the_floor_and_says_why(monkeypatch):
    n = _plan(monkeypatch)
    asks = [[1.0001, 150 / 1.0001], [1.05, 1e12]]                                        # the exit side fills $150 < the $200 floor
    r, blocks = _open(monkeypatch, [[1.0, 1e12]], asks=asks, mode="sweep", min_usd=200.0)
    assert r is None and "DEPTH_GUARD_SKIP" in blocks
    assert CAPS[-1]['cap_action'] == "SKIP" and CAPS[-1]['cap_final_usd'] == 0.0


def test_no_min_margin_reskip_in_floor_mode(monkeypatch):
    """a $300 shrink at 5× = $60 margin < min_investment_size $100: the fraction mode re-skips it ('below min margin'), the $ floor mode trades it."""
    n = _plan(monkeypatch)
    asks = [[1.0001, 300 / 1.0001], [1.05, 1e12]]
    r, blocks = _open(monkeypatch, [[1.0, 1e12]], asks=asks, mode="sweep", min_usd=200.0, min_frac=0.0)
    assert r is not None and r.notional_value == pytest.approx(300, rel=1e-3)
    r2, blocks2 = _open(monkeypatch, [[1.0, 1e12]], asks=asks, mode="sweep", min_usd=0.0, min_frac=0.0)
    assert r2 is None and "DEPTH_GUARD_SKIP" in blocks2


def test_config_json_ui_exports_and_scout(tmp_path):
    import config as C
    assert C.InvestmentConfig.model_fields["entry_depth_guard_min_usd"].default == 0.0 if hasattr(C, "InvestmentConfig") else True
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    inv = cfg.get("investment", cfg)
    assert inv["entry_depth_guard_min_usd"] == 200.0 and "DECISION_LOG 588" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-depth-guard-min-usd"') == 1 and "config.investment.entry_depth_guard_min_usd ?? 0" in html
    assert "entry_depth_guard_min_usd: (() => {" in html and "shrink to the book, skip only below $" in html
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import pandas as pd
    import scout_depth_band_583 as S
    t0 = pd.Timestamp("2026-11-20")
    rows = [dict(opened_at=str(t0 + pd.Timedelta(hours=i)), pair=f"P{i}USDT", direction="SHORT", status="CLOSED", entry_strategy="EXHAUSTION_SHORT",
                 pnl=-1.0, pnl_percentage=-0.4, close_reason="STOP_LOSS", notional_value=500.0, entry_depth_shrunk_from=5000.0, entry_slippage_pct=0.0,
                 exit_slippage_pct=0.0, is_paper=True, entry_depth_mode="impact", entry_cap_frac=0.1) for i in range(22)]
    fr = pd.DataFrame(rows); fr["o"] = pd.to_datetime(fr.opened_at)
    txt = S.report_588(frame=fr, since_ms=int(t0.value // 10**6), state_path=str(tmp_path / "s.json"))
    assert "FROZEN" in txt and "REVERT" in txt and json.load(open(tmp_path / "s.json"))["verdict"] == "REVERT"
    assert S.gate_588([(0.2, 0.05)] * 20)[0] == "KEEP" and S.gate_588([(0.2, 0.3)] * 20)[0] == "REVERT" and S.gate_588([(0.2, None)] * 5)[0] == "COLLECTING"
    assert "_db583.report_588(now_ms=now_ms)" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "reports/SCOUT_DEPTH_FLOOR_588.json" in open(os.path.join(ROOT, ".gitignore")).read()


# ── review 14 / 15 (DL 588): floor = min($ floor, plan × min_frac) · the quantity step + limits.market min before the exchange-minimum check ──
def test_floor_is_min_of_usd_and_frac_and_small_plans_still_shrink():
    assert D.depth_floor_usd(244, 0.25, 200) == pytest.approx(61.0)                   # a $244 plan: floor = 0.25 × 244, not $200
    assert D.depth_floor_usd(30_500, 0.25, 200) == 200.0
    assert D.depth_guard_size(244, 100, 0.25, 200) == (D.SHRINK, 100)                  # shrinks as before the $ floor existed
    assert D.depth_guard_size(244, 50, 0.25, 200) == (D.SKIP, 50)


def test_step_and_market_min():
    m = {'precision': {'amount': 0.1}, 'limits': {'amount': {'min': 1}, 'market': {'min': 50}, 'cost': {'min': 5}}}
    assert D.step_floor_notional(m, 300, 1.03) == pytest.approx(299.936)              # 291.26… units → 291.2 × 1.03
    assert D.step_floor_notional({}, 300, 1.03) == 300                                 # no step known → unchanged
    assert D.exchange_min_notional(m, 2.0) == 100.0                                     # limits.market min (50) × price wins over amount.min


class _Mk(dict):
    def __init__(self, m):
        super().__init__(x=1)   # non-empty, like the loaded markets
        self._m = m

    def get(self, k, d=None):
        return self._m


def test_open_path_skips_below_the_exchange_minimum_and_names_it(monkeypatch):
    import services.trading_engine as TE
    n = _plan(monkeypatch)
    asks = [[1.0001, 300 / 1.0001], [1.05, 1e12]]                                        # shrinks to ~$300 (above the $200 floor)
    mk = _Mk({'precision': {'amount': 1}, 'limits': {'market': {'min': 400}}})          # the market order minimum: 400 units ≈ $400
    monkeypatch.setattr(TE.binance_service.public_exchange, "markets", mk, raising=False)
    msgs = []
    monkeypatch.setattr(TE.logger, "info", lambda m, *a, **k: msgs.append(str(m)))
    monkeypatch.setattr(TE.logger, "warning", lambda m, *a, **k: msgs.append(str(m)))
    r, blocks = _open(monkeypatch, [[1.0, 1e12]], asks=asks, mode="sweep", min_usd=200.0)
    assert r is None and "DEPTH_GUARD_SKIP" in blocks and n > 400
    assert any("exchange minimum" in m for m in msgs)


@pytest.mark.parametrize("step,qty,want", [(1.0, 10.5, 10.0), (10.0, 105, 100.0), (0.001, 1.23456, 1.234), (10.0, 9.9, 0.0)])
def test_amount_step_is_a_tick_size_step(step, qty, want):
    """ccxt binanceusdm = TICK_SIZE mode: precision.amount 1.0 / 10.0 / 0.001 is the STEP (not decimals)."""
    m = {'precision': {'amount': step}}
    assert D._amount_step(m) == step
    assert D.step_floor_notional(m, qty * 2.0, 2.0) == pytest.approx(want * 2.0)


def test_588_line_drops_probes_and_states_its_unit(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import pandas as pd
    import scout_depth_band_583 as S
    t0 = pd.Timestamp("2026-11-20")
    base = dict(direction="SHORT", status="CLOSED", pnl=1.0, close_reason="FRENZY_TP", notional_value=500.0, entry_depth_shrunk_from=5000.0,
                entry_slippage_pct=0.0, exit_slippage_pct=0.0, is_paper=True, entry_depth_mode="impact", entry_cap_frac=0.1)
    rows = [dict(base, opened_at=str(t0 + pd.Timedelta(hours=i)), pair=f"P{i}USDT", entry_strategy="EXHAUSTION_SHORT", pnl_percentage=0.5)
            for i in range(19)]
    rows += [dict(base, opened_at=str(t0 + pd.Timedelta(hours=30 + i)), pair=f"Q{i}USDT", entry_strategy="MOMENTUM", pnl_percentage=-5.0,
                  is_probe=True) for i in range(3)]                                       # probes: would flip the mean if counted
    rows += [dict(base, opened_at=str(t0 + pd.Timedelta(hours=40)), pair="RUSDT", entry_strategy="MOMENTUM", pnl_percentage=-5.0,
                  cell_multiplier_source="C1_PROBE")]
    fr = pd.DataFrame(rows); fr["o"] = pd.to_datetime(fr.opened_at)
    txt = S.report_588(frame=fr, since_ms=int(t0.value // 10**6), state_path=str(tmp_path / "s.json"))
    assert "19/20" in txt and "FROZEN" not in txt                                         # 4 probe fills dropped
    assert "unleveraged price %" in S.gate_588([(0.1, None)] * 20)[1]
