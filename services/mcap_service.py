"""💰 Sep-28 market-cap cache (DECISION_LOG 124) — display column on Top Pairs + entry_mcap_usd / entry_cmc_rank stamps.

Source: the endpoint behind Binance's own "Info" panel (futures + spot), data provided by CoinMarketCap:
    https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/tardingPair/detail?symbol=<base, lowercase>
    → data.mc (market cap USD, circulating) · data.rk (CMC rank) · data.cs (circulating supply)
It is UNDOCUMENTED, so everything here is fail-safe by construction:
  · nothing on the trading path ever waits on the network — the engine only reads the in-memory cache (`get`);
  · refreshes run as a background task, at most one at a time, every `mcap_refresh_minutes`;
  · any error / missing field / non-positive value → the pair simply has no value (UI shows "–", stamps stay NULL);
  · values older than 3 × the refresh interval are treated as missing (a dead endpoint never serves stale caps forever).
🗄 Oct-8 persistence (a deploy restart left the cache empty → FRENZY_WILLY's turnover filter read "unreadable" seconds after the restart,
RLCUSDT 18:24 UTC): after every successful refresh / one-pair fetch the cache is written ATOMICALLY (temp file + os.replace, off the event
loop) to mcap_cache.json in the persistent data dir — /opt/scalpars-data on the server (outside /var/app/current, so an EB deploy never wipes
it; the same dir as the SQLite DB, trading_config.json and the decision journal), the repo dir locally. `load_persisted()` runs once at
startup before the first scan and applies the SAME staleness rule (a value older than 3 × the refresh interval is never loaded, and `get`
re-checks it). Unreadable / corrupt / missing file → a warning, nothing loaded, startup never blocked.
"""
import asyncio
import json
import logging
import math
import os
import re
import tempfile
import time
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger(__name__)

URL = "https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/tardingPair/detail"
_MULT_PREFIX = re.compile(r"^(1000000|100000|10000|1000|100)(?=[A-Z])")

_cache: Dict[str, Tuple[Optional[float], Optional[int], float]] = {}   # pair → (mcap_usd, cmc_rank, fetched_at)
_last_refresh = 0.0
_running = False
_fail_logged_at = 0.0
_task = None   # strong reference — asyncio keeps only weak refs to tasks (a GC'd refresh would leave _running stuck)


def base_symbol(pair: str) -> Optional[str]:
    """PEPEUSDT → 'pepe' · 1000PEPEUSDT → 'pepe' · 1000000MOGUSDT → 'mog' · BTCUSDT → 'btc'. None when not a USDT pair."""
    if not isinstance(pair, str):
        return None
    p = pair.upper().strip()
    if not p.endswith("USDT") or len(p) <= 4:
        return None
    b = _MULT_PREFIX.sub("", p[:-4])
    return b.lower() if b else None


def lookup_candidates(pair: str):
    """Symbols to query, most specific first (deep review Sep-28): the FULL base (1000cat = Simons Cat, 1000sats) before the
    multiplier-stripped one (pepe for 1000PEPE) — stripping first loses real coins and could land on a different coin."""
    if not isinstance(pair, str):
        return []
    p = pair.upper().strip()
    if not p.endswith("USDT") or len(p) <= 4:
        return []
    full = p[:-4].lower()
    out = [full]
    b = base_symbol(pair)
    if b and b != full:
        out.append(b)
    return out


def parse_detail(payload, expect: Optional[str] = None) -> Tuple[Optional[float], Optional[int]]:
    """(mcap_usd, cmc_rank) from the endpoint JSON; (None, None) on anything unexpected. When `expect` is given, the
    response's own symbol (data.alias) must equal it — a different coin is NEVER accepted."""
    try:
        d = (payload or {}).get("data") or {}
    except AttributeError:
        return None, None
    if expect is not None and str(d.get("alias") or "").upper() != expect.upper():
        return None, None
    try:
        mc = d.get("mc")
        mc = float(mc) if mc is not None else None
        if mc is None or not math.isfinite(mc) or mc <= 0:
            mc = None
    except (TypeError, ValueError):
        mc = None
    try:
        rk = d.get("rk")
        rk = int(rk) if rk is not None and int(rk) > 0 else None
    except (TypeError, ValueError):
        rk = None
    return (mc, rk) if mc is not None else (None, None)


def _interval_s() -> float:
    try:
        import config
        return max(5.0, float(getattr(config.trading_config, "mcap_refresh_minutes", 30.0) or 30.0)) * 60.0
    except Exception:
        return 1800.0


def _enabled() -> bool:
    import os
    if os.environ.get("SCALPARS_REPLAY") == "1":   # the engine replay must never touch the network
        return False
    try:
        import config
        return bool(getattr(config.trading_config, "mcap_fetch_enabled", True))
    except Exception:
        return False


def get(pair: str) -> Tuple[Optional[float], Optional[int]]:
    """Cached (mcap_usd, cmc_rank) for a pair — never touches the network. (None, None) when unknown or stale."""
    v = _cache.get(pair)
    if not v:
        return None, None
    mc, rk, ts = v
    if time.time() - ts > 3 * _interval_s():
        return None, None
    return mc, rk


PERSIST_NAME = "mcap_cache.json"
_save_lock = None   # asyncio.Lock, created lazily on the running loop: one writer at a time (the newest snapshot always lands last)


def _path() -> str:
    """the persisted cache file: /opt/scalpars-data (survives EB deploys — the established persistent-state dir) when it exists, else '.'."""
    base = '/opt/scalpars-data' if os.path.isdir('/opt/scalpars-data') else '.'
    return os.path.join(base, PERSIST_NAME)


def _clean_entry(v, now: float, max_age: float):
    """a stored [mcap_usd, cmc_rank, fetched_at] → the cache tuple, or None (bad / non-positive / non-finite / stale / from the future)."""
    try:
        mc, rk, ts = v
        mc = float(mc); ts = float(ts)
        if not math.isfinite(mc) or mc <= 0 or not math.isfinite(ts) or ts > now + 60 or now - ts > max_age:
            return None
    except (TypeError, ValueError):
        return None
    try:   # a bad rank never discards a valid cap (same rule as parse_detail)
        rk = int(rk) if rk is not None and int(rk) > 0 else None
    except (TypeError, ValueError, OverflowError):
        rk = None
    return (mc, rk, ts)


def _write_file(path: str, snapshot: Dict[str, Tuple[Optional[float], Optional[int], float]]) -> None:
    """atomic write (temp file in the same dir + fsync + os.replace, then a best-effort fsync of the directory) — a crash mid-write never
    leaves a half file. Raises on I/O errors of the write itself."""
    d = os.path.dirname(path) or '.'
    fd, tmp = tempfile.mkstemp(prefix='.mcap_cache.', suffix='.tmp', dir=d)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({"v": 1, "saved_at": time.time(), "cache": {k: [mc, rk, ts] for k, (mc, rk, ts) in snapshot.items()}}, f, separators=(',', ':'))
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:   # best-effort: make the rename itself durable (no directory fds on some platforms — ignored)
        dfd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except (OSError, AttributeError):
        pass


async def save_persisted() -> bool:
    """write the current (non-stale) cache to disk in a worker thread — never on the trading path, never raises. → True when written."""
    global _save_lock
    try:
        if os.environ.get("SCALPARS_REPLAY") == "1":
            return False
        now, max_age = time.time(), 3 * _interval_s()
        snap = {k: v for k, v in dict(_cache).items() if v and v[0] is not None and now - v[2] <= max_age}   # snapshot on the loop thread
        if _save_lock is None:
            _save_lock = asyncio.Lock()
        async with _save_lock:
            await asyncio.to_thread(_write_file, _path(), snap)
        return True
    except Exception as e:
        logger.warning(f"[MCAP] cache not persisted (fail-safe, trading unaffected): {str(e)[:120]}")
        return False


def _sweep_orphans(d: str, max_age_s: float = 600.0) -> int:
    """best-effort: delete `.mcap_cache.*.tmp` files (a write killed between mkstemp and os.replace) older than 10 min. Never raises."""
    n = 0
    try:
        now = time.time()
        for name in os.listdir(d):
            if name.startswith('.mcap_cache.') and name.endswith('.tmp'):
                try:
                    fp = os.path.join(d, name)
                    if now - os.path.getmtime(fp) > max_age_s:
                        os.unlink(fp); n += 1
                except OSError:
                    pass
    except Exception:
        pass
    return n


def load_persisted() -> int:
    """startup, before the first scan: merge the persisted cache into memory under the SAME staleness rule (older than 3 × the refresh
    interval → not loaded) — a fresher in-memory value is never overwritten. Missing file → 0 quietly; unreadable / corrupt → a warning, 0.
    Never raises, never touches the network. → the number of pairs loaded."""
    try:
        if os.environ.get("SCALPARS_REPLAY") == "1":
            return 0
        p = _path()
        _sweep_orphans(os.path.dirname(p) or '.')
        if not os.path.isfile(p):
            return 0
        with open(p, encoding='utf-8') as f:
            raw = json.load(f)
        items = (raw or {}).get("cache") if isinstance(raw, dict) else None
        if not isinstance(items, dict):
            raise ValueError("no 'cache' map")
        now, max_age = time.time(), 3 * _interval_s()
        n = stale = 0
        for k, v in items.items():
            e = _clean_entry(v, now, max_age) if isinstance(k, str) and k else None
            if e is None:
                stale += 1
                continue
            cur = _cache.get(k)
            if cur is None or cur[2] < e[2]:
                _cache[k] = e
                n += 1
        logger.info(f"[MCAP] loaded {n} persisted market caps ({stale} stale / invalid skipped)")
        return n
    except Exception as e:
        logger.warning(f"[MCAP] persisted cache ignored (unreadable: {str(e)[:120]}) — caps refill on the first refresh")
        return 0


async def _fetch_pair(client, pair: str) -> Tuple[Optional[float], Optional[int]]:
    """one pair's (mcap_usd, cmc_rank) from the endpoint (every lookup candidate, polite pacing) — (None, None) on any failure."""
    mc, rk = None, None
    for sym in lookup_candidates(pair):
        try:
            r = await client.get(URL, params={"symbol": sym})
            mc, rk = parse_detail(r.json(), expect=sym) if r.status_code == 200 else (None, None)
        except Exception:
            mc, rk = None, None
        if mc is not None:
            break
        await asyncio.sleep(0.15)
    return mc, rk


_one_req: Dict[str, float] = {}   # pair → last one-pair request time (dedupe)
_one_tasks: set = set()
ONE_PAIR_DEDUPE_S = 600.0


def request_pair(pair: str) -> bool:
    """🔄 Oct-8 (DECISION_LOG 251): fire-and-forget ONE-pair fetch for a pair whose cap is missing / stale (FRENZY_WILLY arms an entry on a
    pair the periodic refresh may not have read yet — the turnover filter fails closed without it). At most once per pair per 10 min, never
    while fetching is disabled, never awaited, never raises. → True when a fetch was started."""
    try:
        if not pair or not _enabled():
            return False
        v = _cache.get(pair)
        if v and time.time() - v[2] <= _interval_s():
            return False   # fresh already
        if time.time() - _one_req.get(pair, 0.0) < ONE_PAIR_DEDUPE_S:
            return False
        _one_req[pair] = time.time()

        async def _one():
            try:
                import httpx
                async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
                    mc, rk = await _fetch_pair(client, pair)
                if mc is not None:
                    _cache[pair] = (mc, rk, time.time())
                    await save_persisted()   # 🗄 background: a restart right after still has this cap
            except Exception:
                pass
        t = asyncio.get_running_loop().create_task(_one())
        _one_tasks.add(t); t.add_done_callback(_one_tasks.discard)
        return True
    except Exception:
        return False


async def _refresh(pairs: Iterable[str]) -> None:
    global _running, _last_refresh, _fail_logged_at
    try:
        import httpx
        ok = fail = 0
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            for pair in pairs:
                mc, rk = await _fetch_pair(client, pair)
                if mc is not None:
                    _cache[pair] = (mc, rk, time.time())
                    ok += 1
                else:
                    fail += 1
                await asyncio.sleep(0.15)          # ~50 pairs ≈ 10 s per refresh — polite to the endpoint
        if ok:
            await save_persisted()   # 🗄 background task (never the trading path), atomic, fail-safe
        if ok == 0 and fail and time.time() - _fail_logged_at > 3600:
            _fail_logged_at = time.time()
            logger.warning(f"[MCAP] refresh returned no market caps ({fail} pairs) — column shows '–' until it recovers")
        else:
            logger.info(f"[MCAP] refreshed {ok} market caps ({fail} without data)")
    except Exception as e:
        if time.time() - _fail_logged_at > 3600:
            _fail_logged_at = time.time()
            logger.warning(f"[MCAP] refresh failed (fail-safe, trading unaffected): {e}")
    finally:
        _last_refresh = time.time()
        _running = False


def ensure_refresh(pairs: Iterable[str]) -> None:
    """Fire-and-forget: start a background refresh when due. Safe to call every scan; never raises, never blocks."""
    global _running, _task
    try:
        if _running and _task is not None and _task.done():      # watchdog: a finished/cancelled task never pins the flag
            _running = False
        if not _enabled() or _running or time.time() - _last_refresh < _interval_s():
            return
        pairs = [p for p in dict.fromkeys(pairs) if p]
        if not pairs:
            return
        _running = True
        _task = asyncio.get_running_loop().create_task(_refresh(pairs))
    except Exception:
        _running = False
