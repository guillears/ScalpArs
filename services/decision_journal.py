"""📓 Decision journal (Sep-18) — what the bot SAW and DECIDED, one JSON line per event.

Why: the engine-replay harness has to RECONSTRUCT the forming candle and the scan second, which is
where it diverges from live (it reproduced live winners 2-3x more often than live losers). Recording
the decision inputs at the moment of the decision makes every future period exactly checkable and
gives the harness real known answers to calibrate against. It changes NOTHING about how the bot trades.

Design constraints (this bot has had SQLite write-lock incidents — the journal never touches the DB):
  * append-only JSONL files, one per UTC day: <dir>/decisions-YYYY-MM-DD.jsonl
    dir = /opt/scalpars-data/journal on the server (survives deploys), ./journal locally
  * events are BUFFERED in memory and written once per scan (flush at the next SCAN header, or when the
    buffer reaches _MAX_BUFFER) — a handful of small writes per minute
  * every failure is swallowed: the journal can never raise into the trading path
  * on day rollover yesterday's file is gzipped and files older than the retention are deleted
  * off switch: thresholds.decision_journal_enabled=false, or env SCALPARS_JOURNAL_OFF=1 (the replay
    harness sets it — a replay must not write a journal)

Events: SCAN (BTC/market state + macro vetoes + monitor), BLOCK (gate + pair + the pair's indicator
snapshot), ADMIT (a gate waived, e.g. CROSS_OB_OPEN), OPEN (a fill), EXPIRED (maker window lapsed).
"""
import gzip
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_MAX_BUFFER = 400
_buffer = []
_state = {'day': None, 'last_error_at': 0.0}

# the pair-level inputs the entry ladder reads (a BLOCK line with a full snapshot ≈ 650 bytes, without ≈ 110)
CTX_KEYS = (
    'price', 'rsi', 'rsi_prev1', 'rsi_prev2', 'adx', 'adx_prev1', 'pos_di', 'neg_di', 'atr', 'atr_pct',
    'ema5', 'ema8', 'ema13', 'ema20', 'ema50', 'ema5_prev1', 'ema8_prev1', 'ema13_prev1', 'ema20_prev3', 'ema50_prev12',
    'high_20', 'low_20', 'volume', 'avg_volume', 'candle_open', 'candle_volume_raw', 'candle_avg_volume_20',
)


def _dir():
    base = '/opt/scalpars-data' if os.path.isdir('/opt/scalpars-data') else '.'
    return os.path.join(base, 'journal')


def _enabled():
    if os.environ.get('SCALPARS_JOURNAL_OFF'):
        return False
    try:
        import config
        return bool(getattr(config.trading_config.thresholds, 'decision_journal_enabled', False))
    except Exception:
        return False


def _num(v):
    """JSON-safe, compact: floats rounded to 4 dp, NaN/inf → None, everything exotic → str."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    try:
        f = float(v)
        if f != f or f in (float('inf'), float('-inf')):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return str(v)[:80]


def snapshot(indicators):
    """Compact copy of the pair's indicator dict (only CTX_KEYS that are present)."""
    try:
        if not isinstance(indicators, dict):
            return None
        return {k: _num(indicators.get(k)) for k in CTX_KEYS if indicators.get(k) is not None}
    except Exception:
        return None


def note(event, **fields):
    """Buffer one event. Never raises."""
    try:
        if not _enabled():
            return
        rec = {'t': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], 'e': event}
        for k, v in fields.items():
            if v is None:
                continue
            rec[k] = {str(kk): _num(vv) for kk, vv in v.items()} if isinstance(v, dict) else _num(v)
        _buffer.append(json.dumps(rec, separators=(',', ':'), ensure_ascii=False))
        if len(_buffer) >= _MAX_BUFFER:
            flush()
    except Exception:
        pass


def flush():
    """Write the buffer to today's file. Never raises; on failure the buffer is dropped (bounded memory)."""
    if not _buffer:
        return
    lines, _buffer[:] = list(_buffer), []
    try:
        d = _dir()
        os.makedirs(d, exist_ok=True)
        day = datetime.utcnow().strftime('%Y-%m-%d')
        with open(os.path.join(d, f'decisions-{day}.jsonl'), 'a', encoding='utf-8', errors='replace') as f:
            f.write('\n'.join(lines) + '\n')
        if _state['day'] != day:
            _state['day'] = day
            _rollover(d, day)
    except Exception as e:
        if time.time() - _state['last_error_at'] > 3600:      # at most one warning an hour
            _state['last_error_at'] = time.time()
            logger.warning(f"[DECISION_JOURNAL] write failed ({e}) — events dropped, trading unaffected")


def _rollover(d, today):
    """gzip finished days, delete files past the retention. Best-effort."""
    try:
        import config
        keep = int(getattr(config.trading_config.thresholds, 'decision_journal_retention_days', 45) or 45)
    except Exception:
        keep = 45
    cutoff = (datetime.utcnow() - timedelta(days=max(1, keep))).strftime('%Y-%m-%d')
    for name in sorted(os.listdir(d)):
        if not name.startswith('decisions-'):
            continue
        day = name[len('decisions-'):len('decisions-') + 10]
        path = os.path.join(d, name)
        try:
            if day < cutoff:
                os.remove(path)
            elif day < today and name.endswith('.jsonl'):
                with open(path, 'rb') as src, gzip.open(path + '.gz', 'wb', compresslevel=5) as dst:   # level 5: ~3× faster, the event loop waits on this once a day
                    shutil.copyfileobj(src, dst)
                os.remove(path)
        except Exception:
            continue
