"""🗄 TRACK RECORD (DECISION_LOG 436, operator 2026-11-02: "build 1, 2 and 3" — the proof a buyer / auditor asks for; no trading change).

① BACKUP — once per UTC day (after 00:10) a consistent SQLite snapshot of the bot database (sqlite3 online-backup API: a consistent copy
   while the bot keeps writing, WAL included) is checked (PRAGMA quick_check), gzip-compressed and uploaded to the S3 bucket
   `backup_s3_bucket` (server-side encrypted, the instance's own AWS role — no keys in code) as scalpars/db/YYYY-MM/scalpars_YYYY-MM-DD.db.gz,
   with trading_config.json beside it; the 1st of each month is also copied to scalpars/db-monthly/ (meant to be kept for good — the
   bucket's lifecycle rule ages out db/). Empty bucket → the snapshot is still made and kept on the server (the last BACKUP_KEEP_LOCAL
   copies, next to the database) and the run is recorded as LOCAL_ONLY. Every attempt is a BackupRun row (ok / error / size / key).
② EXCHANGE RECORD — once per UTC day, for the day just ended (and any missed day up to EXCHANGE_BACKFILL_DAYS back, while Binance still
   serves it): every /fapi/v1/income row of the account (all types, raw, deduped) is stored in exchange_income; Binance's own balance
   (USDT wallet, margin balance, unrealized, BNB) is snapshotted; and an ExchangeDay row compares Binance with the bot's booked LIVE
   orders: realized P&L by close day (Binance REALIZED_PNL is gross of fees = the bot's pnl + its booked fees), commissions by fill day
   (entry fee on the open day, exit fee on the close day; BNB fees at the BNB 1m close of their minute), funding (informational — the bot
   stamps funding on the close day, Binance pays it while a position is held), and the WALLET CHECK: the USDT wallet moved between two
   snapshots by exactly Σ the USDT income rows between them (Binance's own consistency, independent of the bot). Status OK / CHECK
   (a difference above the tolerance) / NO_SNAPSHOT (a backfilled day: income only) / NO_ACTIVITY. Runs in paper and live — the record is
   the live ACCOUNT's (operator app trades included); the bot side counts is_paper = False rows only. Shares the Binance-app recorder's
   request budget machinery (its own WeightBudget instance, ban check, used-weight header pause).
③ CODE VERSION — code_version() = a 12-hex fingerprint of the runtime source (top-level *.py, services/*.py, live_terminal/*.py,
   templates/*.html), the default of Order.code_version (every insert path stamps it; not an entry_* market stamp) and of every BackupRun / ExchangeDay row.
   scripts/code_version.py maps a fingerprint back to its git commit (same file rule, same hash).
"""
import asyncio
import glob
import gzip
import hashlib
import logging
import os
import shutil
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_GLOBS = ("*.py", "services/*.py", "live_terminal/*.py", "templates/*.html")
BACKUP_AFTER_MIN = 10                 # the UTC day's backup / exchange record runs from 00:10 UTC (the day's last trades closed and booked)
BACKUP_RETRY_MIN = 60                 # a failed attempt is retried after this long
BACKUP_KEEP_LOCAL = 2                 # snapshots kept on the server next to the database
BACKUP_PREFIX = "scalpars/"           # the object-key prefix inside the bucket
EXCHANGE_BACKFILL_DAYS = 7            # missed days re-read (income only) — Binance keeps the income history ~3 months
TR_FIRST_DELAY_S, TR_INTERVAL_S = 180, 600
DAY_MS = 86_400_000


# ── ③ code version ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def code_paths(names: Iterable[str]) -> List[str]:
    """the runtime-source subset of repo-relative paths (posix separators), sorted. PURE — the same rule for a server checkout and a
    git tree (scripts/code_version.py)."""
    import fnmatch
    out = set()
    for n in names:
        n = str(n).replace(os.sep, "/")
        if "__pycache__" in n:
            continue
        for g in CODE_GLOBS:
            if fnmatch.fnmatchcase(n, g) and n.count("/") == g.count("/"):
                out.add(n)
    return sorted(out)


def code_fingerprint(read: Callable[[str], bytes], paths: Iterable[str]) -> str:
    """sha256 over (path, length, bytes) of each path in order → the first 12 hex chars. PURE on its inputs."""
    h = hashlib.sha256()
    for p in paths:
        b = read(p)
        h.update(p.encode() + b"\0" + str(len(b)).encode() + b"\0" + b)
    return h.hexdigest()[:12]


_CODE_VERSION: Optional[str] = None


def code_version(root: str = _ROOT) -> str:
    """the running code's fingerprint (computed once per process). Never raises → 'unknown'."""
    global _CODE_VERSION
    if _CODE_VERSION is None or root != _ROOT:
        try:
            names = [os.path.relpath(f, root) for g in CODE_GLOBS for f in glob.glob(os.path.join(root, g))]

            def _read(p):
                with open(os.path.join(root, p), "rb") as fh:
                    return fh.read()
            v = code_fingerprint(_read, code_paths(names))
        except Exception as e:
            logger.warning(f"[TRACK_RECORD] code version unreadable: {e}")
            v = "unknown"
        if root != _ROOT:
            return v
        _CODE_VERSION = v
    return _CODE_VERSION


# ── ① backup ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def db_file_path() -> Optional[str]:
    """the SQLite file the app writes (None for a non-file database)."""
    try:
        from database import engine
        p = engine.url.database
        return os.path.abspath(p) if p and p != ":memory:" else None
    except Exception:
        return None


def snapshot_db(src: str, dst_gz: str) -> int:
    """consistent copy of src (sqlite3 online backup) → quick_check → gzip at dst_gz. → compressed bytes. Raises on any failure (the
    partial file is removed)."""
    tmp = dst_gz + ".tmp.db"
    try:
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        s = sqlite3.connect(src, timeout=30)                  # a normal connection: the backup API only reads (mode=ro + WAL needs a writable -shm)
        d = sqlite3.connect(tmp)
        try:
            s.backup(d)
            chk = d.execute("PRAGMA quick_check").fetchone()
            if not chk or str(chk[0]).lower() != "ok":
                raise RuntimeError(f"snapshot quick_check failed: {chk}")
        finally:
            d.close()
            s.close()
        with open(tmp, "rb") as fi, gzip.open(dst_gz, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, 1 << 20)
        return os.path.getsize(dst_gz)
    except Exception:
        if os.path.exists(dst_gz):
            os.remove(dst_gz)
        raise
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def s3_keys(prefix: str, day: str) -> Dict[str, str]:
    """object keys of one day's backup. PURE. prefix '' or ending in '/'."""
    p = (prefix or "").strip().lstrip("/")
    if p and not p.endswith("/"):
        p += "/"
    out = {"db": f"{p}db/{day[:7]}/scalpars_{day}.db.gz", "config": f"{p}db/{day[:7]}/trading_config_{day}.json"}
    if day.endswith("-01"):
        out["monthly"] = f"{p}db-monthly/scalpars_{day}.db.gz"
    return out


def _s3_upload(bucket: str, pairs: List[Tuple[str, str]]):
    """upload [(local path, key)] to bucket, SSE-AES256, the instance role's credentials. Raises."""
    import boto3                                              # lazy: only the backup needs it (requirements.txt)
    cli = boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "eu-west-1")
    for path, key in pairs:
        cli.upload_file(path, bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})


def _prune_local(folder: str, keep: int = BACKUP_KEEP_LOCAL):
    files = sorted(glob.glob(os.path.join(folder, "scalpars_*.db.gz")))
    for f in files[:-keep] if keep > 0 else files:
        try:
            os.remove(f)
        except OSError:
            pass


def valid_bucket(name: str) -> bool:
    """an S3 bucket name (3–63 of a-z 0-9 . -, starts / ends alphanumeric). PURE."""
    import re
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", (name or "").strip()))


def space_needed(src: str, folder: str) -> int:
    """bytes a snapshot may need: the database + its WAL (the uncompressed copy) + the largest kept gzip + 1 GB of headroom for the live
    database. PURE on the file system it reads."""
    db = os.path.getsize(src) + (os.path.getsize(src + "-wal") if os.path.exists(src + "-wal") else 0)
    gz = max([os.path.getsize(f) for f in glob.glob(os.path.join(folder, "scalpars_*.db.gz"))] or [0])
    return db + gz + (1 << 30)


def run_backup_sync(day: str, bucket: str, prefix: str = "scalpars/", src: Optional[str] = None, config_path: Optional[str] = None,
                    upload=_s3_upload, free_bytes=None) -> dict:
    """one day's backup (blocking — call through asyncio.to_thread). → dict(status OK / LOCAL_ONLY, bytes, key, local). Raises.
    (436 review) a leftover *.tmp.db of a killed run is removed first · the snapshot is skipped (ERROR) when the disk lacks space_needed —
    the live database must never hit a full disk · a snapshot of the same day already on disk is reused (a failed upload retries the
    upload only) · a bad bucket name fails before any copy."""
    src = src or db_file_path()
    if not src or not os.path.exists(src):
        raise RuntimeError(f"database file not found ({src})")
    bucket = (bucket or "").strip()
    if bucket and not valid_bucket(bucket):
        raise ValueError(f"invalid S3 bucket name {bucket!r}")
    folder = os.path.join(os.path.dirname(src), "backups")
    os.makedirs(folder, exist_ok=True)
    for f in glob.glob(os.path.join(folder, "*.tmp.db")):
        try:
            os.remove(f)
        except OSError:
            pass
    local = os.path.join(folder, f"scalpars_{day}.db.gz")
    if os.path.exists(local) and os.path.getsize(local) > 0:
        size = os.path.getsize(local)
    else:
        free = (free_bytes if free_bytes is not None else shutil.disk_usage(folder).free)
        need = space_needed(src, folder)
        if free < need:
            raise RuntimeError(f"not enough disk for a snapshot: free {free / 1e9:.2f} GB < needed {need / 1e9:.2f} GB")
        size = snapshot_db(src, local)
    _prune_local(folder)
    keys = s3_keys(prefix, day)
    if not bucket:
        return dict(status="LOCAL_ONLY", bytes=size, key=None, local=local)
    pairs = [(local, keys["db"])]
    cfg = config_path or os.path.join(_ROOT, "trading_config.json")
    if os.path.exists(cfg):
        pairs.append((cfg, keys["config"]))
    if "monthly" in keys:
        pairs.append((local, keys["monthly"]))
    upload(bucket, pairs)
    return dict(status="OK", bytes=size, key=keys["db"], local=local)


# ── ② exchange record (pure core) ───────────────────────────────────────────────────────────────────────────────────────────────────
def day_bounds(day: str) -> Tuple[int, int]:
    t0 = int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    return t0, t0 + DAY_MS - 1


def income_key(r: dict) -> str:
    """(329) tranId repeats across income types → the key is (type, tranId, symbol, asset)."""
    return f"{r.get('incomeType')}|{r.get('tranId')}|{r.get('symbol') or ''}|{r.get('asset') or ''}"


def income_sums(rows: Iterable[dict], rate, lo_ms: int, hi_ms: int, strict: bool = True) -> dict:
    """Σ per type inside [lo, hi] in USDT (commission / non-USDT converted by rate(asset, t) → (price, estimated?)). An unconvertible asset
    → ValueError when strict, else left out and listed in `unknown` (the day is recorded CHECK with a note, never retried forever — 436
    review). PURE. → dict(realized, commission, funding, transfer, other, n, n_est[, unknown])."""
    s = dict(realized=0.0, commission=0.0, funding=0.0, transfer=0.0, other=0.0, n=0, n_est=0)
    unknown = set()
    for r in rows:
        t = int(r.get("time") or 0)
        if t < lo_ms or t > hi_ms:
            continue
        x = float(r.get("income") or 0.0)
        a = str(r.get("asset") or "USDT").upper()
        if a != "USDT":
            rr = rate(a, t)
            if rr is None:
                if strict:
                    raise ValueError(f"income asset {a} cannot be converted to USDT")
                unknown.add(a)
                continue
            x *= rr[0]
            s["n_est"] += 1 if rr[1] else 0
        typ = str(r.get("incomeType") or "")
        k = {"REALIZED_PNL": "realized", "COMMISSION": "commission", "FUNDING_FEE": "funding", "TRANSFER": "transfer"}.get(typ, "other")
        s[k] += x
        s["n"] += 1
    out = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in s.items()}
    if unknown:
        out["unknown"] = sorted(unknown)
    return out


def usdt_income_between(rows: Iterable[dict], lo_ms: int, hi_ms: int) -> float:
    """Σ USDT-asset income with lo < time ≤ hi — what the USDT wallet must have moved by between two snapshots. PURE."""
    return sum(float(r.get("income") or 0.0) for r in rows
               if str(r.get("asset") or "USDT").upper() == "USDT" and lo_ms < int(r.get("time") or 0) <= hi_ms)


def tolerance(equity: Optional[float]) -> float:
    """$ difference treated as rounding: max($1, 0.1 % of the account). PURE."""
    return max(1.0, 0.001 * abs(float(equity or 0.0)))


def exchange_day_verdict(sums: dict, bot: dict, snap: Optional[dict], prev_snap: Optional[dict], wallet_income: Optional[float]) -> dict:
    """PURE. sums = income_sums of the day · bot = dict(gross, fees, funding, closed, opened) of the bot's live rows · snap / prev_snap =
    dict(at_ms, wallet, …) or None · wallet_income = Σ USDT income between the two snapshots (None when either is missing).
    → the ExchangeDay fields: diffs + status."""
    eq = (snap or {}).get("margin_balance") or (snap or {}).get("wallet") or (prev_snap or {}).get("wallet")
    tol = tolerance(eq)
    d_real = round(sums["realized"] - bot["gross"], 4)
    d_fee = round(-sums["commission"] - bot["fees"], 4)                 # Binance commission is negative (paid); bot fees positive
    w_chk = None
    if snap and prev_snap and wallet_income is not None:
        w_chk = round(snap["wallet"] - prev_snap["wallet"] - wallet_income, 4)
    notes = []
    if abs(d_real) > tol:
        notes.append(f"realized P&L Binance {sums['realized']:+.2f} vs bot {bot['gross']:+.2f}")
    if abs(d_fee) > tol:
        notes.append(f"fees Binance {-sums['commission']:.2f} vs bot {bot['fees']:.2f}")
    if w_chk is not None and abs(w_chk) > tol:
        notes.append(f"wallet moved {snap['wallet'] - prev_snap['wallet']:+.2f} vs income {wallet_income:+.2f}")
    if sums.get("unknown"):
        notes.append("income in " + "/".join(sums["unknown"]) + " not converted to USDT (left out)")
    if sums["n"] == 0 and not bot["closed"] and not bot["opened"] and not sums.get("unknown"):
        status = "NO_ACTIVITY" if (w_chk is None or abs(w_chk) <= tol) else "CHECK"
    else:
        status = "CHECK" if notes else ("OK" if snap else "NO_SNAPSHOT")
    return dict(diff_realized=d_real, diff_fees=d_fee, wallet_check_diff=w_chk, tolerance=round(tol, 2), status=status,
                note="; ".join(notes)[:300] if notes else None)


# ── ② exchange record (I/O) and the loop ────────────────────────────────────────────────────────────────────────────────────────────
def _cfg():
    import config
    return config.trading_config


def _keys_configured() -> bool:
    try:
        from config import settings
        return bool(settings.binance_api_key and settings.binance_api_secret)
    except Exception:
        return False


async def _bot_side(db, day: str) -> dict:
    """the bot's LIVE rows for one UTC day: realized gross (pnl + both booked fees) and funding of rows CLOSED that day; fees by fill day
    (entry fee of rows OPENED that day, exit fee of rows CLOSED that day)."""
    from sqlalchemy import select, and_
    from models import Order
    lo = datetime.strptime(day, "%Y-%m-%d")
    hi = lo + timedelta(days=1)
    closed = (await db.execute(select(Order).where(and_(Order.is_paper == False, Order.status == "CLOSED",   # noqa: E712
                                                        Order.closed_at >= lo, Order.closed_at < hi)))).scalars().all()
    opened = (await db.execute(select(Order).where(and_(Order.is_paper == False, Order.opened_at >= lo,       # noqa: E712
                                                        Order.opened_at < hi)))).scalars().all()
    f = lambda v: float(v or 0.0)
    return dict(gross=round(sum(f(o.pnl) + f(o.entry_fee) + f(o.exit_fee) for o in closed), 4),
                fees=round(sum(f(o.entry_fee) for o in opened) + sum(f(o.exit_fee) for o in closed), 4),
                funding=round(sum(f(o.funding_fee_usd) for o in closed), 4), closed=len(closed), opened=len(opened))


class TrackRecorder:
    """one pass = ① today's backup if due · ② the exchange record of the day(s) just ended. Own sessions; never touches trading state."""

    def __init__(self, session_factory=None, commit=None, now: Callable[[], datetime] = None, backup_fn=None, app_recorder=None,
                 balance_fn=None, keys_configured: Callable[[], bool] = None):
        if session_factory is None:
            from database import AsyncSessionLocal as session_factory
        if commit is None:
            from database import locked_commit as commit
        self.sf, self.commit = session_factory, commit
        self.now = now or datetime.utcnow
        self.backup_fn = backup_fn or run_backup_sync
        self._app = app_recorder
        self.balance_fn = balance_fn
        self.keys_configured = keys_configured or _keys_configured

    def _rec(self):
        if self._app is None:
            from services.app_trades import AppTradeRecorder
            self._app = AppTradeRecorder()          # its budget / ban / header discipline — a SEPARATE instance (own budget)
        return self._app

    async def run_pass(self) -> dict:
        now = self.now()
        out = {}
        if now.hour == 0 and now.minute < BACKUP_AFTER_MIN:
            return out
        cfg = _cfg()
        if getattr(cfg, "backup_enabled", True):
            out["backup"] = await self._backup_if_due(now)
        if getattr(cfg, "exchange_record_enabled", True) and self.keys_configured():
            out["exchange"] = await self._exchange_if_due(now)
            try:
                out["rechecked"] = await self._recheck_recent(now)
            except Exception as e:
                logger.warning(f"[EXCHANGE_RECORD] recheck skipped: {e}")
        return out

    async def _backup_if_due(self, now: datetime) -> Optional[str]:
        from sqlalchemy import select
        from models import BackupRun
        day = now.strftime("%Y-%m-%d")
        async with self.sf() as db:
            rows = (await db.execute(select(BackupRun).where(BackupRun.day == day).order_by(BackupRun.id.desc()))).scalars().all()
        bucket_now = (getattr(_cfg(), "backup_s3_bucket", "") or "").strip()
        if any(r.ok and (r.status != "LOCAL_ONLY" or not bucket_now) for r in rows):
            return None                                         # done (a LOCAL_ONLY day re-runs once a bucket is set the same day)
        if rows and rows[0].started_at and now - rows[0].started_at < timedelta(minutes=BACKUP_RETRY_MIN):
            return None
        cfg = _cfg()
        t0 = time.monotonic()
        rec = dict(day=day, started_at=now, ok=False)
        try:
            res = await asyncio.to_thread(self.backup_fn, day, getattr(cfg, "backup_s3_bucket", "") or "", BACKUP_PREFIX)
            rec.update(ok=True, status=res["status"], bytes=res.get("bytes"), s3_key=res.get("key"))
            logger.info(f"[BACKUP] {day}: {res['status']} · {res.get('bytes', 0) / 1e6:.1f} MB · {res.get('key') or res.get('local')}")
        except Exception as e:
            rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:300])
            logger.error(f"[BACKUP] {day}: FAILED — {rec['error']} (retried in {BACKUP_RETRY_MIN} min)")
        rec["seconds"] = round(time.monotonic() - t0, 1)
        async with self.sf() as db:
            db.add(BackupRun(**rec))
            await self.commit(db)
        return rec["status"]

    async def _recheck_recent(self, now: datetime, days: int = 3) -> int:
        """(436 review) a live app trade imported after the day's pass, or a late booking, must not leave a stale CHECK: the bot side and the
        verdict of the last `days` recorded days are recomputed in place. Binance's stored sums / snapshot are unchanged, and the two
        findings a recompute cannot see are kept: a wallet-check difference and an unconverted income asset. → rows changed."""
        from sqlalchemy import select
        from models import ExchangeDay
        since = (now.date() - timedelta(days=days)).strftime("%Y-%m-%d")
        changed = 0
        async with self.sf() as db:
            for r in (await db.execute(select(ExchangeDay).where(ExchangeDay.day >= since))).scalars().all():
                bot = await _bot_side(db, r.day)
                sums = dict(realized=r.inc_realized or 0.0, commission=r.inc_commission or 0.0, funding=r.inc_funding or 0.0,
                            transfer=r.inc_transfer or 0.0, other=r.inc_other or 0.0, n=r.inc_rows or 0, n_est=r.inc_est_rows or 0)
                snap = dict(at_ms=r.snap_at_ms, wallet=r.wallet_usdt, margin_balance=r.margin_balance_usdt) if r.wallet_usdt is not None else None
                v = exchange_day_verdict(sums, bot, snap, None, None)
                keep = [x for x in (r.note or "").split("; ") if x.startswith("wallet moved") or "not converted" in x]
                notes = ([v["note"]] if v["note"] else []) + keep
                new = dict(bot_gross=bot["gross"], bot_fees=bot["fees"], bot_funding=bot["funding"], bot_closed=bot["closed"],
                           bot_opened=bot["opened"], diff_realized=v["diff_realized"], diff_fees=v["diff_fees"],
                           status=("CHECK" if (keep or v["status"] == "CHECK") else v["status"]), note=("; ".join(notes)[:300] or None))
                if any(getattr(r, k) != val for k, val in new.items()):
                    for k, val in new.items():
                        setattr(r, k, val)
                    changed += 1
            if changed:
                await self.commit(db)
        return changed

    async def _exchange_if_due(self, now: datetime) -> List[str]:
        from sqlalchemy import select
        from models import ExchangeDay
        today = now.date()
        want = [(today - timedelta(days=k)).strftime("%Y-%m-%d") for k in range(EXCHANGE_BACKFILL_DAYS, 0, -1)]
        async with self.sf() as db:
            have = set((await db.execute(select(ExchangeDay.day).where(ExchangeDay.day.in_(want)))).scalars().all())
        done = []
        for day in want:
            if day in have:
                continue
            try:
                await self._exchange_day(day, take_snapshot=(day == want[-1]), now=now)
                done.append(day)
            except Exception as e:
                from services.app_trades import AppTradeStop
                logger.error(f"[EXCHANGE_RECORD] {day}: not recorded — {type(e).__name__}: {e}")
                if isinstance(e, AppTradeStop):
                    break                                       # a ban / rate limit: stop the pass, retried next cycle
        return done

    async def _snapshot(self) -> Optional[dict]:
        rec = self._rec()
        fn = self.balance_fn
        if fn is None:
            from services.binance_service import binance_service
            fn = binance_service.get_balance
        b = await rec._call(5, fn)
        if not b or not b.get("ok"):
            return None
        return dict(at_ms=int(self.now().replace(tzinfo=timezone.utc).timestamp() * 1000), wallet=float(b.get("usdt_total") or 0.0),
                    margin_balance=float(b.get("usdt_margin_balance") or 0.0), unrealized=float(b.get("usdt_unrealized") or 0.0),
                    bnb=float(b.get("bnb_total") or 0.0))

    async def _exchange_day(self, day: str, take_snapshot: bool, now: datetime):
        from sqlalchemy import select
        from models import ExchangeDay, ExchangeIncome
        rec = self._rec()
        rec._bnb_missing, rec._bnb_done, rec._bnb_now = set(), set(), None   # (436 review) a block read before the day ended is re-read
        lo, hi = day_bounds(day)
        snap = await self._snapshot() if take_snapshot else None
        if take_snapshot and snap is None:                     # (436 review) never a permanent no-balance day: retried next cycle
            raise RuntimeError("Binance balance unreadable — the day is retried next cycle")
        async with self.sf() as db:
            prev = (await db.execute(select(ExchangeDay).where(ExchangeDay.snap_at_ms.isnot(None))
                                     .order_by(ExchangeDay.snap_at_ms.desc()).limit(1))).scalars().first()
        prev_snap = dict(at_ms=prev.snap_at_ms, wallet=prev.wallet_usdt) if prev is not None and prev.wallet_usdt is not None else None
        w_lo = min(lo, prev_snap["at_ms"]) if (snap and prev_snap and prev_snap["at_ms"] < lo) else lo
        w_hi = max(hi, snap["at_ms"]) if snap else hi
        if snap and prev_snap and snap["at_ms"] - prev_snap["at_ms"] > (EXCHANGE_BACKFILL_DAYS + 2) * DAY_MS:
            prev_snap = None                                     # too old a pair: the window would leave Binance's paging bounds
            w_lo = lo
        rows = await rec._income(w_lo, w_hi)
        await rec.prepare_rates({"BNB": [int(r.get("time") or 0) for r in rows if str(r.get("asset") or "").upper() == "BNB"]})
        sums = income_sums(rows, rec.rate, lo, hi, strict=False)
        w_inc = usdt_income_between(rows, prev_snap["at_ms"], snap["at_ms"]) if (snap and prev_snap) else None
        async with self.sf() as db:
            bot = await _bot_side(db, day)
            v = exchange_day_verdict(sums, bot, snap, prev_snap, w_inc)
            known = set((await db.execute(select(ExchangeIncome.key).where(ExchangeIncome.key.in_([income_key(r) for r in rows])))).scalars().all())
            for r in rows:
                k = income_key(r)
                if k in known:
                    continue
                known.add(k)
                db.add(ExchangeIncome(key=k, time_ms=int(r.get("time") or 0), income_type=str(r.get("incomeType") or "")[:32],
                                      asset=str(r.get("asset") or "")[:12], income=float(r.get("income") or 0.0),
                                      symbol=(str(r.get("symbol")) if r.get("symbol") else None), tran_id=str(r.get("tranId") or "")[:40],
                                      trade_id=(str(r.get("tradeId")) if r.get("tradeId") else None), info=(str(r.get("info"))[:120] if r.get("info") else None)))
            db.add(ExchangeDay(day=day, recorded_at=now, snap_at_ms=(snap or {}).get("at_ms"), wallet_usdt=(snap or {}).get("wallet"),
                               margin_balance_usdt=(snap or {}).get("margin_balance"), unrealized_usdt=(snap or {}).get("unrealized"),
                               bnb_total=(snap or {}).get("bnb"), inc_realized=sums["realized"], inc_commission=sums["commission"],
                               inc_funding=sums["funding"], inc_transfer=sums["transfer"], inc_other=sums["other"], inc_rows=sums["n"],
                               inc_est_rows=sums["n_est"], bot_gross=bot["gross"], bot_fees=bot["fees"], bot_funding=bot["funding"],
                               bot_closed=bot["closed"], bot_opened=bot["opened"], diff_realized=v["diff_realized"], diff_fees=v["diff_fees"],
                               wallet_check_diff=v["wallet_check_diff"], tolerance=v["tolerance"], status=v["status"], note=v["note"]))
            await self.commit(db)
        logger.info(f"[EXCHANGE_RECORD] {day}: {v['status']} · Binance realized {sums['realized']:+.2f} fees {-sums['commission']:.2f} "
                    f"funding {sums['funding']:+.2f} ({sums['n']} rows) · bot gross {bot['gross']:+.2f} fees {bot['fees']:.2f} · wallet check "
                    f"{v['wallet_check_diff'] if v['wallet_check_diff'] is not None else 'n/a'}" + (f" · {v['note']}" if v['note'] else ""))


async def track_record_loop(should_stop: Callable[[], bool], recorder: Optional[TrackRecorder] = None):
    """Background task (started with the app's background tasks). Every TR_INTERVAL_S: run_pass. Never raises out."""
    rec = recorder or TrackRecorder()
    await asyncio.sleep(TR_FIRST_DELAY_S)
    while not should_stop():
        try:
            await rec.run_pass()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[TRACK_RECORD] pass failed: {type(e).__name__}: {e}", exc_info=True)
        for _ in range(TR_INTERVAL_S):
            if should_stop():
                return
            await asyncio.sleep(1)


async def status(db, n_backups: int = 7, n_days: int = 10) -> dict:
    """/api/track-record payload."""
    from sqlalchemy import select
    from models import BackupRun, ExchangeDay
    cfg = _cfg()
    bk = (await db.execute(select(BackupRun).order_by(BackupRun.id.desc()).limit(n_backups))).scalars().all()
    xd = (await db.execute(select(ExchangeDay).order_by(ExchangeDay.day.desc()).limit(n_days))).scalars().all()
    f = lambda d: d.strftime("%Y-%m-%d %H:%M") if d else None
    return dict(code_version=code_version(), backup_enabled=bool(getattr(cfg, "backup_enabled", True)),
                backup_bucket=getattr(cfg, "backup_s3_bucket", "") or "", exchange_record_enabled=bool(getattr(cfg, "exchange_record_enabled", True)),
                keys_configured=_keys_configured(),
                backups=[dict(day=r.day, started_at=f(r.started_at), ok=bool(r.ok), status=r.status, mb=(round(r.bytes / 1e6, 1) if r.bytes else None),
                              key=r.s3_key, error=r.error, code_version=r.code_version) for r in bk],
                days=[dict(day=r.day, status=r.status, wallet=r.wallet_usdt, equity=r.margin_balance_usdt, realized=r.inc_realized,
                           commission=r.inc_commission, funding=r.inc_funding, transfer=r.inc_transfer, rows=r.inc_rows, bot_gross=r.bot_gross,
                           bot_fees=r.bot_fees, diff_realized=r.diff_realized, diff_fees=r.diff_fees, wallet_check=r.wallet_check_diff,
                           tolerance=r.tolerance, note=r.note) for r in xd])
