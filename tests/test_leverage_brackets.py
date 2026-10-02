"""🪜 Exchange leverage brackets (DECISION_LOG 170) — pure sizing rule + wiring."""
import json
import os

import config
import services.trading_engine as TE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOVR = [(5_000, 25), (25_000, 20), (100_000, 10), (500_000, 5), (1_000_000, 2)]      # shape of a small pair's table
f = TE.leverage_bracket_limit


def test_inside_the_bracket_nothing_changes():
    assert f(MOVR, 20, 200) == (20.0, 4_000.0, 25.0, 25_000.0)          # $4k at 20× is allowed as asked
    assert f(MOVR, 25, 200) == (25.0, 5_000.0, 25.0, 5_000.0)           # exactly the 25× cap


def test_leverage_above_the_pair_maximum_is_brought_down():
    lev, n, mx, cap = f(MOVR, 50, 100)                                    # the operator's paper case: 50× does not exist on MOVR
    assert (lev, mx, cap) == (25.0, 25.0, 0.0) and n == 2_500.0


def test_position_over_the_cap_takes_the_best_the_margin_can_carry():
    lev, n, mx, cap = f(MOVR, 20, 1_400)                                  # the bot's $28k at 20× on a small pair
    assert cap == 25_000.0 and (lev, n) == (20.0, 25_000.0)               # 20× up to $25k beats 10× × $1,400 = $14k
    lev, n, _, _ = f(MOVR, 25, 4_000)                                     # $100k at 25×: 25× → $5k · 20× → $25k · 10× → $40k
    assert (lev, n) == (10.0, 40_000.0)
    lev, n, _, _ = f(MOVR, 20, 10_000)                                    # $200k: 10× carries $100k (its cap) — more than 20× ($25k)
    assert (lev, n) == (10.0, 100_000.0)


def test_never_more_margin_and_never_above_any_cap():
    for lev_req in (2, 5, 10, 20, 25, 50, 125):
        for m in (10, 200, 1_400, 4_000, 50_000, 5_000_000):
            lev, n, mx, cap = f(MOVR, lev_req, m)
            assert lev <= lev_req and lev <= mx and n <= m * lev + 1e-6          # margin used = n / lev ≤ m
            assert n <= max(c for c, l in MOVR if l >= lev) + 1e-6               # inside the bracket of the leverage it ends on
            assert n <= m * lev_req + 1e-6                                       # never a bigger position than asked


def test_missing_or_bad_table_means_no_cap():
    assert f(None, 20, 100) is None and f([], 20, 100) is None and f([("x", 1)], 20, 100) is None
    assert f(MOVR, 0, 100) is None and f(MOVR, 20, 0) is None and f(MOVR, float("nan"), 100) is None
    assert f([(0, 20), (-5, 10)], 20, 100) is None


def test_tie_keeps_the_higher_leverage():
    assert f([(1_000, 20), (1_000, 10)], 20, 500)[:2] == (20.0, 1_000.0)


def test_wiring_d11_d12():
    inv = json.load(open(os.path.join(ROOT, "trading_config.json")))["investment"]
    assert inv["leverage_bracket_cap_enabled"] is True
    assert config.InvestmentConfig.model_fields["leverage_bracket_cap_enabled"].default is True
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert html.count("config-leverage-bracket-cap-enabled") == 3                 # input + load + save
    assert "leverage_bracket_cap_enabled: !!document.getElementById('config-leverage-bracket-cap-enabled')?.checked" in html
    assert "leverage-brackets-status" in html and "exchange leverage brackets ${inv.leverage_bracket_cap_enabled" in html
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert eng.count("leverage_bracket_limit(") == 3 and "_record_filter_block('BRACKET_CAP_SKIP', direction)" in eng   # def + bot + manual
    assert eng.count("entry_bracket_max_leverage=_brk_max_lev") == 2
    mdl = open(os.path.join(ROOT, "models.py")).read(); dbs = open(os.path.join(ROOT, "database.py")).read()
    for c in ("entry_bracket_max_leverage", "entry_bracket_cap_notional", "bracket_capped"):
        assert f"    {c} = Column(" in mdl and f"ADD COLUMN {c} " in dbs


def _svc(tiers=None, fail=False, keys=True):
    import asyncio, types
    import services.binance_service as B
    s = B.BinanceService.__new__(B.BinanceService); calls = {"n": 0}

    async def load_markets():
        return None

    async def fetch_leverage_tiers():
        calls["n"] += 1
        if fail:
            raise RuntimeError("boom")
        return tiers
    s.load_markets = load_markets; s.exchange = types.SimpleNamespace(fetch_leverage_tiers=fetch_leverage_tiers)
    B.settings.binance_api_key, B.settings.binance_api_secret = ("k", "s") if keys else ("", "")
    return s, calls, asyncio, B


def test_bracket_table_parse_cache_and_failure(monkeypatch):
    raw = {"MOVR/USDT:USDT": [{"maxNotional": 25000, "maxLeverage": 20}, {"maxNotional": 5000, "maxLeverage": 25}],
           "1000PEPE/USDT:USDT": [{"maxNotional": 50000, "maxLeverage": 50}], "BTC/USDC:USDC": [{"maxNotional": 1, "maxLeverage": 5}],
           "BAD/USDT:USDT": [{"maxNotional": None, "maxLeverage": 5}]}
    s, calls, asyncio, B = _svc(raw)
    k0, s0 = B.settings.binance_api_key, B.settings.binance_api_secret
    try:
        out = asyncio.run(s.get_leverage_brackets())
        assert out == {"MOVRUSDT": [(5000.0, 25.0), (25000.0, 20.0)], "1000PEPEUSDT": [(50000.0, 50.0)]}      # sorted, USDC / bad rows dropped
        assert asyncio.run(s.get_leverage_brackets()) is out and calls["n"] == 1                               # cached: no second call
        s2, c2, _, _ = _svc(fail=True); s2._lev_brackets = {"at": -1e18, "ok": True, "data": out}
        assert asyncio.run(s2.get_leverage_brackets()) == out and c2["n"] == 1                                 # failure keeps the last table
        assert asyncio.run(s2.get_leverage_brackets()) == out and c2["n"] == 1                                 # …and is not retried at once
        s3, c3, _, _ = _svc(raw, keys=False)
        assert asyncio.run(s3.get_leverage_brackets()) == {} and c3["n"] == 0                                  # no keys → no call, no cap
        s4, c4, _, _ = _svc(raw); monkeypatch.setattr(B, "_ban_until", B.time.time() + 3600)
        assert asyncio.run(s4.get_leverage_brackets()) == {} and c4["n"] == 0                                  # banned → returns at once, never sleeps
    finally:
        B.settings.binance_api_key, B.settings.binance_api_secret = k0, s0


def test_manual_panel_shown_in_live_since_oct2():
    """The panel was hidden in live while manual entry was paper-only; live entry shipped Oct-2 (DECISION_LOG 181) — only cap 0 hides it."""
    html = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert "_mp.classList.toggle('hidden', Number(data.manual_max_open_positions) <= 0);" in html
    assert "Number(data.manual_max_open_positions) <= 0 || data.is_paper === false" not in html
