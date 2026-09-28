"""💰 Sep-28 market-cap cache (DECISION_LOG 124) — symbol mapping, parsing, fail-safety, replay isolation, D11/D12 parity."""
import json
import os
import time

from services import mcap_service as M

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_base_symbol_mapping():
    assert M.base_symbol("BTCUSDT") == "btc"
    assert M.base_symbol("1000PEPEUSDT") == "pepe"
    assert M.base_symbol("1000SHIBUSDT") == "shib"
    assert M.base_symbol("1000000MOGUSDT") == "mog"
    assert M.base_symbol("1INCHUSDT") == "1inch"          # a leading digit that is NOT a multiplier stays
    assert M.base_symbol("ETHBTC") is None and M.base_symbol("USDT") is None and M.base_symbol(None) is None


def test_parse_detail():
    assert M.parse_detail({"data": {"mc": 1680329458021.29, "rk": 1}}) == (1680329458021.29, 1)
    assert M.parse_detail({"data": {"mc": None, "rk": None}}) == (None, None)       # futures-only / unknown
    assert M.parse_detail({"data": {"mc": 0, "rk": 5}}) == (None, None)              # no cap → no stamp at all
    assert M.parse_detail({"data": {"mc": 5e8, "rk": "x"}}) == (5e8, None)             # bad rank never discards a good cap
    assert M.parse_detail({"data": {"alias": "CAT", "mc": 1e9, "rk": 9}}, expect="1000cat") == (None, None)   # wrong coin refused
    assert M.parse_detail({"data": {"alias": "1000CAT", "mc": 1e8, "rk": 700}}, expect="1000cat") == (1e8, 700)


def test_lookup_candidates_full_symbol_first():
    assert M.lookup_candidates("1000CATUSDT") == ["1000cat", "cat"]
    assert M.lookup_candidates("1000PEPEUSDT") == ["1000pepe", "pepe"]
    assert M.lookup_candidates("BTCUSDT") == ["btc"]
    assert M.lookup_candidates("1INCHUSDT") == ["1inch"]
    assert M.lookup_candidates("ETHBTC") == []
    assert M.parse_detail({"data": {"mc": float("nan")}}) == (None, None)
    assert M.parse_detail({"data": None}) == (None, None)
    assert M.parse_detail(None) == (None, None)
    assert M.parse_detail({"data": {"mc": "x"}}) == (None, None)


def test_get_is_cache_only_and_expires():
    M._cache.clear()
    assert M.get("BTCUSDT") == (None, None)
    M._cache["BTCUSDT"] = (1.68e12, 1, time.time())
    assert M.get("BTCUSDT") == (1.68e12, 1)
    M._cache["BTCUSDT"] = (1.68e12, 1, time.time() - 10 * 3600)                   # older than 3× the refresh interval
    assert M.get("BTCUSDT") == (None, None)
    M._cache.clear()


def test_replay_never_fetches_and_no_loop_never_raises(monkeypatch):
    monkeypatch.setenv("SCALPARS_REPLAY", "1")
    assert M._enabled() is False
    monkeypatch.delenv("SCALPARS_REPLAY")
    M._last_refresh = 0.0
    M._running = False
    M.ensure_refresh(["BTCUSDT"])            # no running event loop here → swallowed, flag released
    assert M._running is False


def test_d11_d12_parity():
    import config
    with open(os.path.join(ROOT, "trading_config.json")) as f:
        cfg = json.load(f)
    assert cfg["mcap_fetch_enabled"] is True and cfg["mcap_refresh_minutes"] == 30.0
    f = config.TradingConfig.model_fields
    assert f["mcap_fetch_enabled"].default is True and f["mcap_refresh_minutes"].default == 30.0
    main = open(os.path.join(ROOT, "main.py")).read()
    assert "mcap_fetch_enabled: Optional[bool] = None" in main and "mcap_refresh_minutes: Optional[float] = None" in main
    assert '_mc = _mcap_get(p.pair)' in main and '"mcap_usd": _mc[0]' in main
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert ui.count("config-mcap-fetch-enabled") == 3 and ui.count("config-mcap-refresh-minutes") == 3
    assert "Market cap fetch (Sep 28)" in ui and ">Mcap</th>" in ui
    from models import Order
    cols = [c.name for c in Order.__table__.columns]
    assert "entry_mcap_usd" in cols and "entry_cmc_rank" in cols
    db = open(os.path.join(ROOT, "database.py")).read()
    assert "ADD COLUMN entry_mcap_usd FLOAT" in db and "ADD COLUMN entry_cmc_rank INTEGER" in db
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "_mcs.get(pair)" in eng and "_mcs.ensure_refresh(" in eng and "entry_mcap_usd=_mcap_usd" in eng
