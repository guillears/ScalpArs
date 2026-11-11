"""🗄 (DECISION_LOG 436) track record: the code fingerprint (and its git mirror), the consistent SQLite backup (gzip, S3 keys, local
pruning, LOCAL_ONLY), the pure exchange-day math (income sums, wallet check, bot side, statuses), the recorder's schedule (once a day,
retry spacing, backfill, a stop on a rate limit), the order stamp, and the D11 / D12 wiring. Hermetic: temp files, fakes, no network."""
import asyncio
import datetime as dt
import gzip
import json
import os
import sqlite3
import sys
from types import SimpleNamespace as NS

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import track_record as TR  # noqa: E402


# ── ③ code version ──
def test_code_paths_rule_and_fingerprint():
    names = ["main.py", "config.py", "services/a.py", "services/sub/b.py", "services/__pycache__/a.cpython.pyc", "scripts/x.py",
             "tests/t.py", "templates/index.html", "templates/x/y.html", "live_terminal/h.py", "reports/r.py", "README.md"]
    assert TR.code_paths(names) == ["config.py", "live_terminal/h.py", "main.py", "services/a.py", "templates/index.html"]
    files = {"a.py": b"x = 1\n", "b.py": b"y = 2\n"}
    f1 = TR.code_fingerprint(files.__getitem__, ["a.py", "b.py"])
    assert len(f1) == 12 and f1 == TR.code_fingerprint(files.__getitem__, ["a.py", "b.py"])
    assert f1 != TR.code_fingerprint(dict(files, **{"a.py": b"x = 2\n"}).__getitem__, ["a.py", "b.py"])          # content
    assert f1 != TR.code_fingerprint({"a.py": b"x = 1\ny = 2\n", "b.py": b""}.__getitem__, ["a.py", "b.py"])      # boundaries count
    assert TR.code_version() == TR.code_version() and len(TR.code_version()) == 12


def test_code_version_of_a_tree(tmp_path):
    (tmp_path / "services").mkdir(); (tmp_path / "templates").mkdir(); (tmp_path / "scripts").mkdir()
    (tmp_path / "main.py").write_bytes(b"print(1)\n"); (tmp_path / "services" / "s.py").write_bytes(b"s\n")
    (tmp_path / "templates" / "index.html").write_bytes(b"<html>")
    v = TR.code_version(str(tmp_path))
    (tmp_path / "scripts" / "study.py").write_bytes(b"research only\n")                      # not runtime code → same version
    assert TR.code_version(str(tmp_path)) == v
    (tmp_path / "services" / "s.py").write_bytes(b"s2\n")
    assert TR.code_version(str(tmp_path)) != v


# ── ① backup ──
def _db(path, rows=3):
    c = sqlite3.connect(path)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    c.executemany("INSERT INTO t (v) VALUES (?)", [(f"r{i}",) for i in range(rows)])
    c.commit()
    return c


def test_snapshot_is_a_consistent_gzip_copy(tmp_path):
    src = str(tmp_path / "scalpars.db")
    live = _db(src)                                   # the writer stays open (WAL), like the app
    live.execute("INSERT INTO t (v) VALUES ('uncommitted')")   # an open write transaction is NOT in the snapshot
    out = str(tmp_path / "snap.db.gz")
    n = TR.snapshot_db(src, out)
    assert n == os.path.getsize(out) > 0
    raw = tmp_path / "back.db"
    raw.write_bytes(gzip.open(out).read())
    assert [r[0] for r in sqlite3.connect(str(raw)).execute("SELECT v FROM t ORDER BY id")] == ["r0", "r1", "r2"]
    live.rollback(); live.close()
    with pytest.raises(Exception):
        TR.snapshot_db(str(tmp_path / "missing.db"), str(tmp_path / "x.db.gz"))
    assert not os.path.exists(tmp_path / "x.db.gz") and not os.path.exists(tmp_path / "x.db.gz.tmp.db")


def test_s3_keys():
    assert TR.s3_keys("scalpars/", "2026-11-02") == {"db": "scalpars/db/2026-11/scalpars_2026-11-02.db.gz",
                                                     "config": "scalpars/db/2026-11/trading_config_2026-11-02.json"}
    k = TR.s3_keys("x", "2026-12-01")
    assert k["monthly"] == "x/db-monthly/scalpars_2026-12-01.db.gz" and k["db"].startswith("x/db/2026-12/")


def test_run_backup_local_only_upload_and_pruning(tmp_path):
    src = str(tmp_path / "scalpars.db"); _db(src).close()
    cfg = tmp_path / "trading_config.json"; cfg.write_text("{}")
    r = TR.run_backup_sync("2026-11-01", "", src=src, config_path=str(cfg))
    assert r["status"] == "LOCAL_ONLY" and r["key"] is None and os.path.exists(r["local"])
    sent = []
    r = TR.run_backup_sync("2026-12-01", " my-bucket ", "scalpars/", src=src, config_path=str(cfg), upload=lambda b, p: sent.append((b, p)))
    assert r["status"] == "OK" and sent[0][0] == "my-bucket"
    keys = [k for _, k in sent[0][1]]
    assert keys == ["scalpars/db/2026-12/scalpars_2026-12-01.db.gz", "scalpars/db/2026-12/trading_config_2026-12-01.json",
                    "scalpars/db-monthly/scalpars_2026-12-01.db.gz"]
    TR.run_backup_sync("2026-12-02", "", src=src, config_path=str(cfg))
    assert sorted(os.listdir(tmp_path / "backups")) == ["scalpars_2026-12-01.db.gz", "scalpars_2026-12-02.db.gz"]   # keep the last 2

    def boom(b, p):
        raise RuntimeError("AccessDenied")
    with pytest.raises(RuntimeError):
        TR.run_backup_sync("2026-12-03", "bkt-1", src=src, config_path=str(cfg), upload=boom)
    assert os.path.exists(tmp_path / "backups" / "scalpars_2026-12-03.db.gz")                    # the snapshot stays for the retry
    sent2 = []
    n0 = os.path.getmtime(tmp_path / "backups" / "scalpars_2026-12-03.db.gz")
    TR.run_backup_sync("2026-12-03", "bkt-1", src=src, config_path=str(cfg), upload=lambda b, p: sent2.append(p))
    assert sent2 and os.path.getmtime(tmp_path / "backups" / "scalpars_2026-12-03.db.gz") == n0  # retry = upload only, no new copy


def test_backup_guards_disk_tmp_and_bucket_name(tmp_path):
    src = str(tmp_path / "scalpars.db"); _db(src).close()
    (tmp_path / "backups").mkdir()
    (tmp_path / "backups" / "scalpars_2026-11-01.db.gz.tmp.db").write_bytes(b"x" * 100)            # a killed run's leftover
    with pytest.raises(RuntimeError, match="not enough disk"):
        TR.run_backup_sync("2026-11-02", "", src=src, free_bytes=10)
    assert not list((tmp_path / "backups").glob("*.tmp.db")) and not list((tmp_path / "backups").glob("*.gz"))
    with pytest.raises(ValueError, match="invalid S3 bucket"):
        TR.run_backup_sync("2026-11-02", "My_Bucket", src=src)
    assert TR.valid_bucket("my-scalpars-backups") and TR.valid_bucket("a.b-c") and not TR.valid_bucket("s3://x") and not TR.valid_bucket("ab")
    assert TR.space_needed(src, str(tmp_path / "backups")) >= os.path.getsize(src) + (1 << 30)


# ── ② exchange record (pure) ──
T0, _ = TR.day_bounds("2026-11-01")


def _inc(t_min, typ, x, asset="USDT", tid=None, sym="ABCUSDT"):
    return {"time": T0 + int(t_min * 60_000), "incomeType": typ, "income": str(x), "asset": asset, "tranId": tid or f"{typ}{t_min}", "symbol": sym}


RATE = lambda a, t: (600.0, False) if a == "BNB" else None


def test_income_sums_and_wallet_income():
    rows = [_inc(10, "REALIZED_PNL", 50), _inc(10, "COMMISSION", -2), _inc(11, "COMMISSION", -0.005, "BNB"),
            _inc(480, "FUNDING_FEE", -1.5), _inc(600, "TRANSFER", 1000, sym=None), _inc(700, "INSURANCE_CLEAR", 0.3),
            _inc(1500, "REALIZED_PNL", 99)]                                  # the next day — outside
    s = TR.income_sums(rows, RATE, *TR.day_bounds("2026-11-01"))
    assert s == dict(realized=50.0, commission=-5.0, funding=-1.5, transfer=1000.0, other=0.3, n=6, n_est=0)
    with pytest.raises(ValueError):
        TR.income_sums([_inc(5, "COMMISSION", -1, "XYZ")], RATE, *TR.day_bounds("2026-11-01"))
    lax = TR.income_sums([_inc(5, "COMMISSION", -1, "XYZ"), _inc(6, "REALIZED_PNL", 3)], RATE, *TR.day_bounds("2026-11-01"), strict=False)
    assert lax["unknown"] == ["XYZ"] and lax["realized"] == 3.0 and lax["n"] == 1
    v = TR.exchange_day_verdict(lax, dict(gross=3.0, fees=0, funding=0, closed=1, opened=0), None, None, None)
    assert v["status"] == "CHECK" and "XYZ not converted" in v["note"]
    assert TR.usdt_income_between(rows, T0, T0 + 1000 * 60_000) == pytest.approx(50 - 2 - 1.5 + 1000 + 0.3)   # the BNB fee is not USDT
    assert TR.income_key(rows[0]) == "REALIZED_PNL|REALIZED_PNL10|ABCUSDT|USDT"


def test_verdict_statuses():
    s = dict(realized=50.0, commission=-5.0, funding=-1.5, transfer=0.0, other=0.0, n=4, n_est=0)
    bot = dict(gross=50.2, fees=5.1, funding=-1.0, closed=2, opened=2)
    snap, prev = dict(at_ms=2, wallet=1043.5, margin_balance=1050.0), dict(at_ms=1, wallet=1000.0)
    v = TR.exchange_day_verdict(s, bot, snap, prev, 43.5)
    assert v["status"] == "OK" and v["wallet_check_diff"] == 0 and v["diff_realized"] == pytest.approx(-0.2) and v["tolerance"] == 1.05
    v = TR.exchange_day_verdict(s, dict(bot, gross=40.0), snap, prev, 43.5)
    assert v["status"] == "CHECK" and "realized P&L Binance +50.00 vs bot +40.00" in v["note"]
    v = TR.exchange_day_verdict(s, bot, snap, prev, 30.0)
    assert v["status"] == "CHECK" and "wallet moved +43.50 vs income +30.00" in v["note"]
    assert TR.exchange_day_verdict(s, bot, None, None, None)["status"] == "NO_SNAPSHOT"
    z = dict(realized=0.0, commission=0.0, funding=0.0, transfer=0.0, other=0.0, n=0, n_est=0)
    assert TR.exchange_day_verdict(z, dict(gross=0, fees=0, funding=0, closed=0, opened=0), snap, dict(at_ms=1, wallet=1043.5), 0.0)["status"] == "NO_ACTIVITY"
    assert TR.tolerance(None) == 1.0 and TR.tolerance(50_000) == 50.0


# ── ② the recorder (DB + schedule) ──
def _mem_db():
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from database import Base
    import models  # noqa: F401
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")

    async def init():
        async with eng.begin() as c:
            await c.run_sync(Base.metadata.create_all)
    return eng, async_sessionmaker(eng, expire_on_commit=False), init


class _App:
    """stands in for AppTradeRecorder: _call / _income / prepare_rates / rate."""
    def __init__(self, rows, stop=False):
        self.rows, self.stop, self.calls = rows, stop, []

    async def _call(self, w, fn, *a):
        return await fn(*a)

    async def _income(self, lo, hi):
        self.calls.append((lo, hi))
        if self.stop:
            from services.app_trades import AppTradeStop
            raise AppTradeStop("429")
        return [r for r in self.rows if lo <= r["time"] <= hi]

    async def prepare_rates(self, x):
        pass

    def rate(self, a, t):
        return RATE(a, t)


async def _commit(db):
    await db.commit()


def test_recorder_schedule_backup_and_exchange_days(monkeypatch):
    import models
    from sqlalchemy import select
    eng, SL, init = _mem_db()
    monkeypatch.setattr(TR, "_cfg", lambda: NS(backup_enabled=True, backup_s3_bucket="b", exchange_record_enabled=True))
    clock = {"now": dt.datetime(2026, 11, 2, 0, 5)}
    calls = []

    def backup_fn(day, bucket, prefix):
        calls.append((day, bucket, prefix))
        if len(calls) == 1:
            raise RuntimeError("AccessDenied")
        return dict(status="OK", bytes=1234, key=f"{prefix}db/{day}")

    async def bal():
        return dict(ok=True, usdt_total=1043.5, usdt_margin_balance=1050.0, usdt_unrealized=6.5, bnb_total=0.1)
    rows = [_inc(10, "REALIZED_PNL", 50), _inc(10, "COMMISSION", -2)]
    app = _App(rows)
    rec = TR.TrackRecorder(session_factory=SL, commit=_commit, now=lambda: clock["now"], backup_fn=backup_fn, app_recorder=app,
                           balance_fn=bal, keys_configured=lambda: True)

    async def go():
        await init()
        async with SL() as db:   # a live bot row closed on 11-01: pnl 47 net of 1 + 1 fees → gross 49
            db.add(models.Order(pair="ABCUSDT", direction="LONG", status="CLOSED", is_paper=False, entry_strategy="MOMENTUM", confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0, pnl=47.0, entry_fee=1.0, exit_fee=1.0,
                                opened_at=dt.datetime(2026, 11, 1, 0, 1), closed_at=dt.datetime(2026, 11, 1, 0, 10)))
            db.add(models.Order(pair="P", direction="LONG", status="CLOSED", is_paper=True, entry_strategy="MOMENTUM", confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0, pnl=999.0,
                                opened_at=dt.datetime(2026, 11, 1, 1), closed_at=dt.datetime(2026, 11, 1, 2)))   # paper: never the bot side
            await db.commit()
        r0 = await rec.run_pass()                                    # 00:05 → nothing before 00:10
        clock["now"] = dt.datetime(2026, 11, 2, 0, 12)
        r1 = await rec.run_pass()                                    # backup fails · 7 days recorded (6 income-only + yesterday with a snapshot)
        clock["now"] = dt.datetime(2026, 11, 2, 0, 40)
        r2 = await rec.run_pass()                                    # < 60 min after the failure → no retry; nothing left to record
        clock["now"] = dt.datetime(2026, 11, 2, 1, 15)
        r3 = await rec.run_pass()                                    # retry → OK
        r4 = await rec.run_pass()                                    # done for today
        async with SL() as db:
            bk = (await db.execute(select(models.BackupRun).order_by(models.BackupRun.id))).scalars().all()
            xd = (await db.execute(select(models.ExchangeDay).order_by(models.ExchangeDay.day))).scalars().all()
            inc = (await db.execute(select(models.ExchangeIncome))).scalars().all()
            o = (await db.execute(select(models.Order).limit(1))).scalars().first()
        await eng.dispose()
        return r0, r1, r2, r3, r4, bk, xd, inc, o
    r0, r1, r2, r3, r4, bk, xd, inc, o = asyncio.run(go())
    assert r0 == {} and r1["backup"] == "ERROR" and r2.get("backup") is None and r3["backup"] == "OK" and r4.get("backup") is None
    assert [b.status for b in bk] == ["ERROR", "OK"] and bk[1].s3_key == "scalpars/db/2026-11-02" and bk[1].code_version == TR.code_version()
    assert calls[0] == ("2026-11-02", "b", "scalpars/")
    assert len(xd) == 7 and xd[-1].day == "2026-11-01" and r1["exchange"][-1] == "2026-11-01" and r2["exchange"] == []
    y = xd[-1]
    assert y.wallet_usdt == 1043.5 and y.inc_realized == 50.0 and y.inc_commission == -2.0 and y.bot_gross == 49.0 and y.bot_fees == 2.0
    assert y.diff_realized == pytest.approx(1.0) and y.status == "OK" and y.wallet_check_diff is None     # first snapshot: no pair yet
    assert all(x.status in ("NO_ACTIVITY",) and x.snap_at_ms is None for x in xd[:-1])
    assert len(inc) == 2 and o.code_version == TR.code_version()                                 # ③ the model default stamps it


def test_rate_limit_stops_the_pass_and_keys_gate(monkeypatch):
    eng, SL, init = _mem_db()
    monkeypatch.setattr(TR, "_cfg", lambda: NS(backup_enabled=False, backup_s3_bucket="", exchange_record_enabled=True))
    app = _App([], stop=True)
    rec = TR.TrackRecorder(session_factory=SL, commit=_commit, now=lambda: dt.datetime(2026, 11, 2, 3), app_recorder=app,
                           balance_fn=None, keys_configured=lambda: True)

    async def go():
        await init()
        r = await rec.run_pass()
        rec.keys_configured = lambda: False
        r2 = await rec.run_pass()
        await eng.dispose()
        return r, r2
    r, r2 = asyncio.run(go())
    assert r["exchange"] == [] and len(app.calls) == 1 and "exchange" not in r2 and "backup" not in r      # one stop ends the pass


# ── wiring (D11 / D12 / D13) ──
def test_config_ui_exports_and_stamp_contract():
    import config as C
    mf = type(C.trading_config).model_fields
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))
    for k, v in (("backup_enabled", True), ("backup_s3_bucket", ""), ("exchange_record_enabled", True)):
        assert mf[k].default == v and cfg[k] == v, k
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for i in ("config-backup-enabled", "config-backup-s3-bucket", "config-exchange-record-enabled"):
        assert html.count(f'id="{i}"') == 1 and html.count(f"getElementById('{i}')") >= 2, i      # one input · load + save
    assert html.count("lines.push(...trackRecordReportLines(hr2));") == 2 and 'id="track-record-status"' in html
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "backup_s3_bucket: Optional[str] = None" in main and '@app.get("/api/track-record")' in main
    assert 'e="EXCHANGE_DAY"' in main and 'e="BACKUP"' in main and "+ _mr_cols + _bl_cols + _xd_cols" in main
    assert "track_record_loop as _track_record_loop" in main
    import models
    from services.trading_engine import TradingEngine
    assert "code_version" in [c.name for c in models.Order.__table__.columns] and not [c for c in models.Order.__table__.columns if c.name == "entry_code_version"]
    for c in main.split("TRACK_RECORD_EXPORT_COLS = [", 1)[1].split("]", 1)[0].replace('"', "").replace("\n", " ").split(","):
        c = c.strip()
        if c.startswith("xd_"):
            assert c[3:] in [x.name for x in models.ExchangeDay.__table__.columns], c
    assert "boto3" in open(os.path.join(ROOT, "requirements.txt")).read()
    assert "ADD COLUMN code_version VARCHAR(16)" in open(os.path.join(ROOT, "database.py")).read()


def test_lookup_script_matches_the_runtime_rule():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import code_version as CV
    assert CV.code_paths is TR.code_paths and CV.code_fingerprint is TR.code_fingerprint


def test_snapshot_failure_retries_wallet_pair_bnb_reset_and_recheck(monkeypatch):
    """(436 deep review) a failed balance read is retried, never a permanent no-balance day · two snapshots → the wallet check · the BNB
    rate caches reset per day · a later app-trade booking re-checks the recent days in place · a LOCAL_ONLY day re-runs once a bucket
    is set the same day."""
    import models
    from sqlalchemy import select
    eng, SL, init = _mem_db()
    cfgv = NS(backup_enabled=True, backup_s3_bucket="", exchange_record_enabled=True)
    monkeypatch.setattr(TR, "_cfg", lambda: cfgv)
    clock = {"now": dt.datetime(2026, 11, 2, 0, 12)}
    bal = {"ok": False}

    async def balance():
        return dict(ok=bal["ok"], usdt_total=bal.get("w", 0), usdt_margin_balance=bal.get("w", 0), usdt_unrealized=0, bnb_total=0)
    D1, _ = TR.day_bounds("2026-11-02")
    rows = [_inc(10, "REALIZED_PNL", 50), _inc(10, "COMMISSION", -2),
            dict(time=D1 + 30 * 60_000, incomeType="REALIZED_PNL", income="20", asset="USDT", tranId="x2", symbol="ABCUSDT")]
    app = _App(rows)
    app._bnb_missing, app._bnb_done, app._bnb_now = {1}, {2}, 9.9
    bk = []
    rec = TR.TrackRecorder(session_factory=SL, commit=_commit, now=lambda: clock["now"], app_recorder=app, balance_fn=balance,
                           keys_configured=lambda: True, backup_fn=lambda d, b, p: bk.append(b) or dict(status=("OK" if b else "LOCAL_ONLY"), bytes=1, key=b or None))

    async def go():
        await init()
        await rec.run_pass()                                          # balance unreadable → 11-01 NOT recorded (6 older days are)
        async with SL() as db:
            d1 = (await db.execute(select(models.ExchangeDay.day))).scalars().all()
        assert "2026-11-01" not in d1 and app._bnb_now is None and app._bnb_missing == set()     # the BNB caches were reset
        bal.update(ok=True, w=1000.0)
        clock["now"] = dt.datetime(2026, 11, 2, 0, 22)
        await rec.run_pass()                                          # retried → recorded with a snapshot
        cfgv.backup_s3_bucket = "bkt-1"
        await rec.run_pass()                                          # the LOCAL_ONLY day re-runs once a bucket is set
        async with SL() as db:   # an app trade booked later for 11-01 (gross 48) → the recheck updates the day in place
            db.add(models.Order(pair="ABCUSDT", direction="LONG", status="CLOSED", is_paper=False, entry_strategy="MANUAL", confidence="STRONG_BUY",
                                entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0, pnl=46.0, entry_fee=1.0, exit_fee=1.0,
                                opened_at=dt.datetime(2026, 11, 1, 0, 2), closed_at=dt.datetime(2026, 11, 1, 0, 9)))
            await db.commit()
        r = await rec.run_pass()
        async with SL() as db:
            y = (await db.execute(select(models.ExchangeDay).where(models.ExchangeDay.day == "2026-11-01"))).scalars().first()
            y_status, y_gross, y_note = y.status, y.bot_gross, y.note
        clock["now"] = dt.datetime(2026, 11, 3, 0, 15)               # the next day: a second snapshot → the wallet check
        bal.update(w=1020.0)
        await rec.run_pass()
        async with SL() as db:
            z = (await db.execute(select(models.ExchangeDay).where(models.ExchangeDay.day == "2026-11-02"))).scalars().first()
            zw = z.wallet_check_diff
        await eng.dispose()
        return r, y_status, y_gross, y_note, zw
    r, y_status, y_gross, y_note, zw = asyncio.run(go())
    assert bk == ["", "bkt-1"]                                         # one LOCAL_ONLY, then the same day uploaded
    assert r["rechecked"] >= 1 and y_gross == 48.0 and y_status == "CHECK"   # Binance +50 vs bot 48 → diff 2 > $1
    assert zw == 0.0                                                   # 1020 − 1000 = the +20 income between the snapshots
