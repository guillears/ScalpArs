"""📉🌡 DECISION_LOG 651 — DIST_CRASH_FUNDING: the DISTRIBUTION_SHORT crash-market × funding filter + its frozen revert gate.
Pure rule (services/distribution.btc_dd30 / crash_funding_block), the REAL engine judge path (tests/test_distribution_short._run), config /
JSON / UI (D11), the scout gate, hooks."""
import json
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services import distribution as D  # noqa: E402
from tests import test_distribution_short as TD  # noqa: E402
import scout_dist_crash_funding_651 as S  # noqa: E402

DAY = 86_400_000
ON = NS(distribution_crash_funding_block_enabled=True)


def _daily(closes_highs, start=0):
    return [[start + i * DAY, 1.0, h, 1.0, c, 1.0] for i, (c, h) in enumerate(closes_highs)]


def test_btc_dd30_uses_the_last_30_closed_days_only():
    rows = _daily([(100.0, 100.0)] * 29 + [(70.0, 200.0)] + [(80.0, 80.0)])     # 31 days; the last one still open at now
    now = 30 * DAY + 5
    assert D.btc_dd30(rows, now) == pytest.approx((70 / 200 - 1) * 100)        # the forming day is ignored
    assert D.btc_dd30(rows, 31 * DAY) == pytest.approx((80 / 200 - 1) * 100)     # once closed it counts (high 200 still in the 30)
    assert D.btc_dd30(_daily([(1, 1)] * 29), 40 * DAY) is None                   # < 30 closed days
    assert D.btc_dd30(None, 1) is None and D.btc_dd30([["x"]], 1) is None
    gap = _daily([(100.0, 100.0)] * 31)
    del gap[10]                                                                 # a missing day → the 30 rows span 31 days → None
    assert D.btc_dd30(gap, 31 * DAY) is None


def test_rule_cuts_switch_and_fail_open():
    assert D.crash_funding_block(-20.0, 0.00005, ON)[0] is True                # both cuts inclusive (≤ −20 · ≥ base)
    assert D.crash_funding_block(-19.99, 0.01, ON)[0] is False                 # not a crash
    assert D.crash_funding_block(-30.0, 0.0000499, ON)[0] is False             # below base (a squeeze) → kept
    assert D.crash_funding_block(-30.0, -0.002, ON)[0] is False
    assert D.crash_funding_block(None, 0.01, ON)[:2] == (False, True)          # drawdown unread → fail-open, flagged
    assert D.crash_funding_block(-30.0, None, ON)[:2] == (False, True)         # crash but funding unread → fail-open, flagged
    assert D.crash_funding_block(-10.0, None, ON)[:2] == (False, False)        # no crash → funding not needed
    assert D.crash_funding_block(-30.0, 0.01, NS()) == (False, False, "")      # code default OFF
    assert D.crash_funding_block("x", object(), ON)[0] is False                # never raises
    assert D.crash_funding_block(-30.0, object(), ON)[:2] == (False, True)     # an odd input → fail-open, counted
    p = D.distribution_params(NS(distribution_crash_btc_dd_pct=5, distribution_crash_funding_min=1))
    assert p["crash_dd"] == -20.0 and p["crash_fund_min"] == 0.00005           # a bad value → the study cut
    errs = " ".join(D.distribution_config_errors(NS(**dict(TD._th().__dict__, distribution_crash_btc_dd_pct=3,
                                                                distribution_crash_funding_block_enabled="yes")), 20.0))
    assert "distribution_crash_btc_dd_pct" in errs and "distribution_crash_funding_block_enabled" in errs


def _go(dd_rows, funding, th_kw=None):
    import services.trading_engine as TE
    mp = pytest.MonkeyPatch()
    try:
        async def ohlcv(sym, tf="5m", limit=100):
            assert sym == "BTC/USDT:USDT" and tf == "1d"
            return dd_rows
        mp.setattr(TE.binance_service, "get_ohlcv", ohlcv)
        out = TD._run(mp, k=0, th_kw=dict(dict(distribution_crash_funding_block_enabled=True), **(th_kw or {})))

        return out
    finally:
        mp.undo()


def _crash_rows(T, dd=-25.0):
    d0 = T // DAY * DAY - 31 * DAY
    return _daily([(100.0, 100.0)] * 30 + [(100.0 * (1 + dd / 100), 100.0)], start=d0)


def test_engine_blocks_in_a_crash_with_paying_longs(monkeypatch):
    pair, T, *_ = TD._row(0)
    e, opens, sig, calls, _ = _go(_crash_rows(T), None)                       # fixture funding = +0.0001 ≥ base
    assert opens == [] and sig[0].block == "DIST_CRASH_FUNDING" and "DIST_CRASH_FUNDING" in e.blocks
    assert "longs still pay" in sig[0].reason and sig[0].tp_pct and sig[0].sl_pct and sig[0].decision_price


def test_engine_keeps_outside_a_crash_and_when_off(monkeypatch):
    pair, T, *_ = TD._row(0)
    e, opens, sig, calls, _ = _go(_crash_rows(T, dd=-5.0), None)
    assert len(opens) == 1 and sig[0].status == "OPENED"
    e, opens, sig, calls, _ = _go(_crash_rows(T), None, th_kw=dict(distribution_crash_funding_block_enabled=False))
    assert len(opens) == 1


def test_engine_unreadable_drawdown_fails_open_and_counts(monkeypatch):
    e, opens, sig, calls, _ = _go([], None)
    assert len(opens) == 1 and "DIST_CRASH_FUNDING_UNREAD" in e.blocks


def test_config_json_ui_wiring():
    import config as C
    assert C.SignalThresholds.model_fields["distribution_crash_funding_block_enabled"].default is False
    t = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert t["distribution_crash_funding_block_enabled"] is True and t["distribution_crash_btc_dd_pct"] == -20.0
    assert t["distribution_crash_funding_min"] == 5e-05
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert html.count("'config-dist-crash-funding-enabled'") == 2 and 'id="config-dist-crash-funding-enabled"' in html
    for i in ("config-dist-crash-dd", "config-dist-crash-funding-min"):
        assert f'id="{i}"' in html and f"['{i}'," in html
    assert "distribution_crash_funding_enabled" in open(os.path.join(ROOT, "main.py")).read()


def test_scout_gate_and_hooks():
    rows = [(f"2026-12-0{1 + i % 3}", -1.0) for i in range(14)]
    assert S.gate(rows)[0] == "COLLECTING"
    assert S.gate(rows + [("2026-12-03", -1.0)])[:3:2] == ("KEEP", 15)
    assert S.gate([(f"2026-12-0{1 + i % 3}", 0.5) for i in range(15)])[0] == "REVERT"
    assert S.gate([(f"2026-12-01", -1.0)] * 20)[0] == "COLLECTING"                 # < 3 days
    assert "NOT EVALUABLE" in S.report(journal=pd.DataFrame(), frame=pd.DataFrame(), floor_ms=None, state_path=None)
    assert "scout_dist_crash_funding_651" in open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "scout_dist_crash_funding_651.py" in open(os.path.join(ROOT, "scripts", "observe_registry.py")).read()
    assert "reports/SCOUT_DIST_CRASH_FUNDING_651.json" in open(os.path.join(ROOT, ".gitignore")).read()


def test_dd30_cache_retry_ban_and_yesterday_fallback(monkeypatch):
    import asyncio
    import services.trading_engine as TE
    e = TD._engine(TE)
    now = 100 * DAY + 3_600_000
    monkeypatch.setattr(TE, "_leash_time", NS(time=lambda: now / 1000.0))
    calls = []
    good = _daily([(100.0, 100.0)] * 29 + [(70.0, 100.0)], start=70 * DAY)

    async def ohlcv(sym, tf="5m", limit=100):
        calls.append(1)
        return out_rows[0]
    monkeypatch.setattr(TE.binance_service, "get_ohlcv", ohlcv)
    monkeypatch.setattr(TE, "_bs_ban_active", lambda: False)
    out_rows = [[]]
    assert asyncio.run(e._dist_btc_dd30()) is None and len(calls) == 1          # unreadable → None, not cached as a value
    out_rows[0] = good
    assert asyncio.run(e._dist_btc_dd30()) == pytest.approx(-30.0) and len(calls) == 2   # retried
    assert asyncio.run(e._dist_btc_dd30()) == pytest.approx(-30.0) and len(calls) == 2   # cached for the day
    now += DAY                                                                    # next day, Binance banned → yesterday's value
    monkeypatch.setattr(TE, "_bs_ban_active", lambda: True)
    assert asyncio.run(e._dist_btc_dd30()) == pytest.approx(-30.0) and len(calls) == 2


def test_one_per_coin_drops_reblocks_inside_the_hold():
    j = pd.DataFrame(dict(dist_block=["DIST_CRASH_FUNDING"] * 3 + ["DIST_RSI1H"], pair=["A", "A", "A", "A"],
                          dist_signal_at=["2026-12-01 00:00", "2026-12-01 06:00", "2026-12-02 01:00", "2026-12-01 07:00"]))
    k = S.one_per_coin(j)
    assert list(k[k.dist_block == "DIST_CRASH_FUNDING"].dist_signal_at) == ["2026-12-01 00:00", "2026-12-02 01:00"]
    assert (k.dist_block == "DIST_RSI1H").sum() == 1
