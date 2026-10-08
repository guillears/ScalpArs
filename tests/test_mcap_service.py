"""💰 Sep-28 market-cap cache (DECISION_LOG 124) — symbol mapping, parsing, fail-safety, replay isolation, D11/D12 parity."""
import json
import os
import time

import pytest

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


@pytest.mark.mcap_real
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


def test_refresh_covers_frenzy_flagged_pairs_and_runs_after_the_frenzy_pass():
    """Oct-2: a FRENZY-flagged pair outside the Top-N had no market cap (APE). The refresh list includes the flagged pairs and is
    requested AFTER the FRENZY pass, so the first refresh after a deploy already sees the rebuilt flags."""
    import os
    eng = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("_mcs.ensure_refresh(") == 1 and "_mcs.ensure_refresh(_mcap_pairs + list(_frenzy_flags))" in eng
    assert eng.index("_mcap_pairs = [p.get('pair') for p in top_pairs]") < eng.index("await self._update_frenzy(db, wait=False)") < eng.index("_mcs.ensure_refresh(")


# ── 🗄 Oct-8 persistence: the cache survives a restart / deploy (RLCUSDT 18:24 "turnover unreadable" seconds after a deploy) ──────────

import asyncio  # noqa: E402



@pytest.fixture
def _iso(monkeypatch, tmp_path):
    """isolated cache + file path; fetching cannot reach the network (no test here starts a fetch)."""
    p = tmp_path / "mcap_cache.json"
    monkeypatch.setattr(M, "_path", lambda: str(p))
    monkeypatch.setattr(M, "_cache", {})
    monkeypatch.setattr(M, "_save_lock", None)
    monkeypatch.setattr(M, "_interval_s", lambda: 1800.0)
    monkeypatch.delenv("SCALPARS_REPLAY", raising=False)
    return p


def test_persist_round_trip_and_get_without_network(_iso, monkeypatch):
    now = time.time()
    M._cache.update({"RLCUSDT": (1.2e8, 310, now - 60), "BTCUSDT": (1.6e12, 1, now - 10), "OLDUSDT": (5e7, 900, now - 4 * 1800)})
    assert asyncio.run(M.save_persisted()) is True
    stored = json.load(open(_iso))["cache"]
    assert set(stored) == {"RLCUSDT", "BTCUSDT"}                                   # a stale entry is never written
    assert not [f for f in os.listdir(_iso.parent) if f.endswith(".tmp")]           # atomic: no temp file left behind
    monkeypatch.setattr(M, "_cache", {})                                            # ← the restart
    import httpx

    def _no_net(*a, **k):
        raise AssertionError("network touched")
    monkeypatch.setattr(httpx, "AsyncClient", _no_net)
    assert M.load_persisted() == 2
    assert M.get("RLCUSDT") == (1.2e8, 310) and M.get("BTCUSDT") == (1.6e12, 1) and M.get("OLDUSDT") == (None, None)


def test_load_applies_the_staleness_rule(_iso):
    now = time.time()
    _iso.write_text(json.dumps({"v": 1, "cache": {
        "FRESHUSDT": [2e8, 50, now - 1800 * 2.9],          # inside 3 × refresh → loaded
        "STALEUSDT": [2e8, 50, now - 1800 * 3 - 5],         # older than 3 × refresh → never served
        "FUTUREUSDT": [2e8, 50, now + 3600],                # clock nonsense → dropped
        "ZEROUSDT": [0, 50, now], "NANUSDT": ["nan", 1, now], "BADUSDT": "x", "SHORTUSDT": [1e8]}}))
    assert M.load_persisted() == 1
    assert set(M._cache) == {"FRESHUSDT"} and M.get("STALEUSDT") == (None, None)
    M._cache["FRESHUSDT"] = (3e8, 40, now)                   # a fresher in-memory value is never overwritten by the file
    assert M.load_persisted() == 0 and M.get("FRESHUSDT") == (3e8, 40)


def test_load_corrupt_or_missing_file_is_ignored(_iso, caplog):
    assert M.load_persisted() == 0 and M._cache == {}       # missing → quiet, nothing loaded
    _iso.write_text("{not json")
    with caplog.at_level("WARNING"):
        assert M.load_persisted() == 0
    assert M._cache == {} and "persisted cache ignored" in caplog.text
    _iso.write_text(json.dumps([1, 2, 3]))
    assert M.load_persisted() == 0 and M._cache == {}
    _iso.write_text(json.dumps({"cache": [1]}))
    assert M.load_persisted() == 0 and M._cache == {}


def test_save_failure_never_raises_and_replay_never_touches_the_file(_iso, monkeypatch):
    M._cache["XUSDT"] = (1e8, 10, time.time())
    monkeypatch.setattr(M, "_path", lambda: str(_iso.parent / "missing_dir" / "mcap_cache.json"))
    assert asyncio.run(M.save_persisted()) is False                                 # I/O error → warning, False
    monkeypatch.setattr(M, "_path", lambda: str(_iso))
    monkeypatch.setenv("SCALPARS_REPLAY", "1")
    assert asyncio.run(M.save_persisted()) is False and not _iso.exists()
    _iso.write_text(json.dumps({"cache": {"YUSDT": [1e8, 1, time.time()]}}))
    assert M.load_persisted() == 0


def test_refresh_persists_in_the_background(_iso, monkeypatch):
    async def fake_fetch(client, pair):
        return (4.2e8, 120) if pair == "GTCUSDT" else (None, None)
    monkeypatch.setattr(M, "_fetch_pair", fake_fetch)

    async def no_sleep(_s):
        return None
    monkeypatch.setattr(M.asyncio, "sleep", no_sleep)
    asyncio.run(M._refresh(["GTCUSDT", "NOPEUSDT"]))
    assert json.load(open(_iso))["cache"]["GTCUSDT"][:2] == [4.2e8, 120]


def test_conftest_isolates_mcap(tmp_path):
    """the autouse conftest fixture: the persist path is a tmp file, request_pair / ensure_refresh are no-ops (no network, no repo write)."""
    assert M._path() == str(tmp_path / "mcap_cache.json")
    assert M.request_pair("FOOUSDT") is False and M.ensure_refresh(["FOOUSDT"]) is None


def test_startup_loads_before_the_first_scan():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert main.index("_mcs_boot.load_persisted()") < main.index("await start_background_tasks()")
    src = open(os.path.join(ROOT, "services", "mcap_service.py"), encoding="utf-8").read()
    assert "'/opt/scalpars-data' if os.path.isdir('/opt/scalpars-data') else '.'" in src and "os.replace(tmp, path)" in src
    assert "await asyncio.to_thread(_write_file" in src


def test_write_fsyncs_the_directory_best_effort(_iso, monkeypatch):
    synced = []
    real_fsync = os.fsync
    monkeypatch.setattr(M.os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    M._write_file(str(_iso), {"AUSDT": (1e8, 1, time.time())})
    assert len(synced) == 2 and json.load(open(_iso))["cache"]["AUSDT"][0] == 1e8      # the file, then its directory
    real_open = os.open

    def no_dir_fd(p, flags, *a):
        if os.path.isdir(p):
            raise OSError("no directory fds here")
        return real_open(p, flags, *a)
    monkeypatch.setattr(M.os, "open", no_dir_fd)
    M._write_file(str(_iso), {"BUSDT": (2e8, 2, time.time())})                         # ignored, the write still lands
    assert set(json.load(open(_iso))["cache"]) == {"BUSDT"}


def test_load_keeps_a_valid_cap_with_a_bad_rank(_iso):
    now = time.time()
    _iso.write_text(json.dumps({"cache": {"AUSDT": [3e8, "x", now - 60], "BUSDT": [4e8, -5, now - 60], "CUSDT": [5e8, None, now - 60]}}))
    assert M.load_persisted() == 3
    assert M.get("AUSDT") == (3e8, None) and M.get("BUSDT") == (4e8, None) and M.get("CUSDT") == (5e8, None)


def test_load_sweeps_old_tmp_orphans(_iso):
    d = _iso.parent
    old, new, other = d / ".mcap_cache.abc.tmp", d / ".mcap_cache.def.tmp", d / "keep.tmp"
    for f in (old, new, other):
        f.write_text("partial")
    t = time.time() - 11 * 60
    os.utime(old, (t, t)); os.utime(other, (t, t))
    assert M.load_persisted() == 0                                                      # no cache file: still sweeps, never raises
    assert not old.exists() and new.exists() and other.exists()                         # only OUR orphans older than 10 min
    assert M._sweep_orphans(str(d / "nope")) == 0                                       # missing dir → 0, no raise
