"""⏳ The shared Binance cooldown file — unit-robust reader (DECISION_LOG 306).

reports/backtest_cache/.binance_cooldown_until holds an epoch until which NO research script may call Binance (a 418 / 429 ban is per IP).
Writers have used two units: the scouts (scout_b1h_negflank._arm_cooldown) write integer MILLISECONDS; several study fetchers write float
SECONDS. Every reader must accept both, or a ban armed in one unit is silently ignored (or blocks forever) in the other. Rule: a value below
1e11 is seconds (1e11 s ≈ year 5138; 1e11 ms ≈ 1973), anything else is milliseconds. Unreadable / missing / expired → 0 (no cooldown).
Never raises.
"""
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = os.path.join(_ROOT, "reports", "backtest_cache", ".binance_cooldown_until")


def to_ms(v):
    """a raw file value (str / number, seconds or ms) → epoch ms (0 when unreadable or ≤ 0)."""
    try:
        x = float(str(v).strip() or 0)
    except (TypeError, ValueError):
        return 0
    if not (0 < x < float("inf")):
        return 0
    return int(x * 1000) if x < 1e11 else int(x)


def read_until_ms(path=None):
    """the cooldown end in epoch ms from one file (default: the shared file); 0 when missing / unreadable."""
    try:
        with open(path or SHARED) as f:
            return to_ms(f.read())
    except Exception:
        return 0


def active(path=None, now_ms=None):
    """True while the cooldown in `path` (default shared) is still in force."""
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    return read_until_ms(path) > now
