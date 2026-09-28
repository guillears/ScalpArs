"""💰 Sep-28 market-cap cache (DECISION_LOG 124) — display column on Top Pairs + entry_mcap_usd / entry_cmc_rank stamps.

Source: the endpoint behind Binance's own "Info" panel (futures + spot), data provided by CoinMarketCap:
    https://www.binance.com/bapi/apex/v1/friendly/apex/marketing/tardingPair/detail?symbol=<base, lowercase>
    → data.mc (market cap USD, circulating) · data.rk (CMC rank) · data.cs (circulating supply)
It is UNDOCUMENTED, so everything here is fail-safe by construction:
  · nothing on the trading path ever waits on the network — the engine only reads the in-memory cache (`get`);
  · refreshes run as a background task, at most one at a time, every `mcap_refresh_minutes`;
  · any error / missing field / non-positive value → the pair simply has no value (UI shows "–", stamps stay NULL);
  · values older than 3 × the refresh interval are treated as missing (a dead endpoint never serves stale caps forever).
"""
import asyncio
import logging
import math
import re
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


async def _refresh(pairs: Iterable[str]) -> None:
    global _running, _last_refresh, _fail_logged_at
    try:
        import httpx
        ok = fail = 0
        async with httpx.AsyncClient(timeout=8.0, headers={"User-Agent": "Mozilla/5.0"}) as client:
            for pair in pairs:
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
                if mc is not None:
                    _cache[pair] = (mc, rk, time.time())
                    ok += 1
                else:
                    fail += 1
                await asyncio.sleep(0.15)          # ~50 pairs ≈ 10 s per refresh — polite to the endpoint
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
