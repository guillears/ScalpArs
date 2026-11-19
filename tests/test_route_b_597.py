"""📡 DECISION_LOG 597 — Route B recorder: hot-list selection (bot Frenzy list first), surge detector, file rotation / closed-hour pick,
mark compaction, the health verdict, the bot-side publisher payload. Pure; no network."""
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b"))
sys.path.insert(0, ROOT)

import recorder as R          # noqa: E402
import health_check as H      # noqa: E402

M = 60_000


def test_surge_flags_a_30_minute_move_and_expires():
    s = R.Surge()
    t0 = 1_700_000_000_000 // M * M
    for k in range(31):
        s.feed("AUSDT", 100.0 * (1 + 0.001 * k), t0 + k * M)      # +3.0 % over 30 min
        s.feed("BUSDT", 100.0, t0 + k * M)
    assert s.active(t0 + 30 * M) == ["AUSDT"]
    s.feed("AUSDT", 103.0, t0 + 30 * M + 5_000)                     # same minute → replaces, no double sample
    assert len(s.px["AUSDT"]) == 31
    assert s.active(t0 + 30 * M + 5 * 3_600_000) == ["AUSDT"]      # still hot 5 h later (units: ms vs s)
    assert s.active(t0 + 30 * M + R.SURGE_KEEP_S * 1000 + 1) == []  # expired after 6 h
    s.feed("CUSDT", "x", t0)
    s.feed("CUSDT", 0, t0)
    assert "CUSDT" not in s.px


def test_surge_needs_a_sample_at_least_30_minutes_old():
    s = R.Surge()
    t0 = 1_700_000_000_000 // M * M
    for k in (0, 1, 2, 31):                                          # minute 1 is exactly 30 min before minute 31 → a valid reference
        s.feed("AUSDT", 100.0 + 10 * (k == 31), t0 + k * M)
    assert s.active(t0 + 31 * M) == ["AUSDT"]
    s2 = R.Surge()
    s2.feed("BUSDT", 100.0, t0)
    s2.feed("BUSDT", 120.0, t0 + 20 * M)                             # only 20 min of history → no call yet
    assert s2.active(t0 + 20 * M) == []


def _tick(sym, ch, qv):
    return {"symbol": sym, "priceChangePercent": str(ch), "quoteVolume": str(qv)}


def test_pick_hot_priority_bot_first_then_sticky_surge_movers_volume():
    tick = [_tick("AUSDT", 30, 6e6), _tick("BUSDT", 50, 1e6), _tick("CUSDT", 5, 9e9), _tick("DUSDT", -40, 7e6), _tick("EUSD", 90, 9e9)]
    tradable = {"BTCUSDT", "ETHUSDT", "AUSDT", "BUSDT", "CUSDT", "DUSDT", "FZUSDT", "OLDUSDT", "SGUSDT"}
    ms = 10 ** 13
    out = R.pick_hot(tick, tradable, ["SGUSDT", "NOTLISTED"], ["OLDUSDT"], {"OLDUSDT": ms - 60_000}, ms, 50, bot=["FZUSDT", "ZZZ"])
    assert out[:5] == ["BTCUSDT", "ETHUSDT", "FZUSDT", "OLDUSDT", "SGUSDT"]
    assert out[5:7] == ["DUSDT", "AUSDT"]                             # movers by |24 h %| with ≥ $5M (BUSDT too thin)
    assert "CUSDT" in out and "EUSD" not in out and "ZZZ" not in out and "NOTLISTED" not in out
    assert R.pick_hot(tick, tradable, [], [], {}, ms, 3, bot=["FZUSDT"]) == ["BTCUSDT", "ETHUSDT", "FZUSDT"]   # cap keeps the priority
    assert "OLDUSDT" in R.pick_hot(tick, tradable, [], ["OLDUSDT"], {"OLDUSDT": ms - 59 * 60_000}, ms, 50)   # 59 min in → still sticky
    old = R.pick_hot(tick, tradable, [], ["OLDUSDT"], {"OLDUSDT": ms - R.STICKY_S * 1000 - 1}, ms, 50)
    assert "OLDUSDT" not in old                                       # sticky only for 60 min


def test_bot_pairs_order_and_staleness():
    ms = 10 ** 13
    doc = {"at": ms - 60_000, "open": ["XUSDT"], "flagged": {"AUSDT": "2026-11-19T10:00:00", "XUSDT": None},
           "tracked": ["BUSDT"], "followed": ["CUSDT", "AUSDT"]}
    assert R.bot_pairs(doc, ms) == ["XUSDT", "AUSDT", "BUSDT", "CUSDT"]
    assert R.bot_pairs(dict(doc, at=ms - R.BOT_MAX_AGE_S * 1000 - 1), ms) == []
    assert R.bot_pairs(None, ms) == [] and R.bot_pairs({"at": "x"}, ms) == [] and R.bot_pairs({"at": ms, "open": 5}, ms) == []


def test_compact_mark_and_hot_streams():
    rows = R.compact_mark([{"s": "AUSDT", "p": "1.0", "i": "1.01", "r": "0.0001", "T": 5}, {"x": 1}, None])
    assert rows == [["AUSDT", "1.0", "1.01", "0.0001", 5]]
    assert R.hot_streams(["AUSDT"], "@aggTrade") == ["ausdt@aggTrade"]
    assert R.HOT_CONNS["book"][0].endswith("/public/stream") and R.HOT_CONNS["trade"][0].endswith("/market/stream")


def test_writer_rotates_by_hour_and_closed_files_picks_only_past_hours(tmp_path):
    w = R.Writer(str(tmp_path), tag=111)
    h0 = 1_795_000_000_000 // 3_600_000 * 3_600_000
    w.write("book", {"a": 1}, h0 + 10)
    w.write("book", {"a": 2}, h0 + 3_600_000 + 10)                     # next hour → the first file is closed
    w.write("liq", {"a": 3}, h0 + 3_600_000 + 20)
    d0, hh0 = R.hour_key(h0)
    d1, hh1 = R.hour_key(h0 + 3_600_000)
    got = R.closed_files(str(tmp_path), d1, hh1)
    assert [(k, d, f) for k, d, f, _ in got] == [("book", d0, f"{hh0}-111.jsonl.gz")]
    assert got[0][3] not in w.open_paths() and len(w.open_paths()) == 2
    w.close()
    with gzip.open(got[0][3], "rt") as f:
        assert [json.loads(x) for x in f] == [{"a": 1}]
    assert R.file_hour("07-123.jsonl.gz") == "07" and R.file_hour("07.jsonl.gz") == "07" and R.file_hour("x7-1.jsonl.gz") is None


def test_a_killed_process_never_corrupts_the_next_ones_file(tmp_path):
    """C1: a process killed mid-hour leaves ITS file truncated; the restart writes its own file, fully readable."""
    import subprocess
    code = (f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'scripts', 'route_b')!r}); import recorder as R, os\n"
            f"w = R.Writer({str(tmp_path)!r}, tag=1)\n"
            "for i in range(5000): w.write('trade', {'i': i}, 1795000000000 + i)\n"
            "w.flush(); os._exit(9)\n")
    subprocess.run([sys.executable, "-c", code], check=False)
    w2 = R.Writer(str(tmp_path), tag=2)
    for i in range(100):
        w2.write("trade", {"i": i}, 1_795_000_000_000 + 5000 + i)
    w2.close()
    d, h = R.hour_key(1_795_000_000_000)
    with gzip.open(os.path.join(tmp_path, "trade", d, f"{h}-2.jsonl.gz"), "rt") as f:
        assert len(f.readlines()) == 100
    assert len(os.listdir(os.path.join(tmp_path, "trade", d))) == 2


def test_writer_pauses_on_low_disk():
    w = R.Writer("/nonexistent-never-written", tag=1)
    w.paused = True
    w.write("book", {"a": 1})
    assert w.dropped == 1 and w.fh == {}


def test_backoff_and_surge_tolerates_a_missing_minute():
    assert R.backoff_s(1) == 2 and R.backoff_s(5) == 6 and R.backoff_s(100) == 30
    s = R.Surge()
    t0 = 1_700_000_000_000 // M * M
    for k in range(32):
        if k in (1, 15):                                                 # two missing minutes
            continue
        s.feed("AUSDT", 100.0 * (1 + 0.001 * k), t0 + k * M)
    assert s.active(t0 + 31 * M) == ["AUSDT"]
    s2 = R.Surge()
    s2.feed("BUSDT", 100.0, t0)
    s2.feed("BUSDT", 110.0, t0 + 50 * M)                                 # the only older sample is 50 min old (> 40) → no call
    assert s2.active(t0 + 50 * M) == []


def test_status_carries_every_field_the_health_check_reads(tmp_path):
    rec = R.Recorder(str(tmp_path), "", 10)
    st = rec.status()
    for k in ("now", "started", "host", "age_s", "hot_n", "bot_n", "day_hot_unique", "reconnects", "day_gap_s", "day_n", "uploaded",
              "upload_fail", "disk_free_gb", "writes_paused", "writes_dropped", "upload_backlog_age_s"):
        assert k in st, k
    assert st["upload_backlog_age_s"] == 0


def test_health_verdict():
    now = 10 ** 13
    ok = {"now": now - 60_000, "age_s": {"mark": 1, "book": 2, "trade": 3, "liq": 900}, "bot_n": 40, "disk_free_gb": 20}
    assert H.verdict(ok, now)[0] == 0
    assert H.verdict(None, now)[0] == 2
    assert H.verdict(dict(ok, now=now - 16 * 60_000), now)[0] == 2
    assert H.verdict(dict(ok, age_s={"mark": 1, "book": 500, "trade": 3}), now)[0] == 2
    assert H.verdict(dict(ok, age_s={"mark": 1, "book": None, "trade": 3}), now)[0] == 2
    assert H.verdict(dict(ok, bot_n=0), now)[0] == 1
    assert H.verdict(dict(ok, disk_free_gb=1.0), now)[0] == 1
    assert H.verdict(dict(ok, upload_backlog_age_s=3 * 3600), now)[0] == 1
    assert H.verdict(dict(ok, writes_paused=True), now)[0] == 1


def test_deploy_script_is_safe_by_construction():
    with open(os.path.join(ROOT, "scripts", "route_b", "deploy_aws.sh")) as f:
        s = f.read()
    assert "HttpTokens=required" in s and "BlockPublicAcls=true" in s and "authorize-security-group-ingress" not in s
    assert '"Resource\\":\\"arn:aws:s3:::$BUCKET/bot/*\\"' in s                # the bot may write ONLY bot/*
    assert '\\"Action\\":[\\"s3:PutObject\\"],\\"Resource\\":[\\"arn:aws:s3:::$BUCKET/data/*\\",\\"arn:aws:s3:::$BUCKET/health/*\\"]' in s   # recorder: no code/* write
    assert "AbortIncompleteMultipartUpload" in s and "aws:SecureTransport" in s and "KillSignal=SIGTERM" in s
    assert "elasticbeanstalk" not in s.replace("aws-elasticbeanstalk-ec2-role", "")   # never touches the EB environment
    assert H.BUCKET in s


# ── the bot side (services/route_b_publish.py) ─────────────────────────────────────────────────────────────────────────────────────

def test_publish_payload_shape():
    from services import route_b_publish as P
    d = P.payload(123, True, ["BUSDT", "AUSDT", "AUSDT"], {"XUSDT": {"spike_ts": 1_795_000_000_000}, "YUSDT": {}, "ZUSDT": None},
                  {"CUSDT"}, ["XUSDT", "DUSDT"])
    assert d["at"] == 123 and d["paper"] is True and d["open"] == ["AUSDT", "BUSDT"] and d["tracked"] == ["CUSDT"]
    assert d["flagged"]["XUSDT"].startswith("2026-") and d["flagged"]["YUSDT"] is None and d["flagged"]["ZUSDT"] is None
    assert d["followed"] == ["XUSDT", "DUSDT"]
    assert R.bot_pairs(dict(d, at=10 ** 13), 10 ** 13) == ["AUSDT", "BUSDT", "XUSDT", "YUSDT", "ZUSDT", "CUSDT", "DUSDT"]   # the recorder reads it


def test_publish_off_under_pytest_and_never_raises():
    from services import route_b_publish as P
    assert P.enabled() is False and P.due() is False and P.maybe_publish(False, None, {}, [], []) is False   # PYTEST_CURRENT_TEST → off
    assert P.payload(1, False, 5, "x", None, None)["open"] == []                                         # junk in → empty lists, no raise


def test_publish_enabled_path_schedules_once_and_waits_for_the_thread(monkeypatch):
    import asyncio
    import threading
    from services import route_b_publish as P
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(P, "_state", {"at": 0.0, "task": None, "fut": None, "fails": 0})
    sent, gate = [], threading.Event()

    def fake_put(body):
        gate.wait(5)
        sent.append(json.loads(body))
    monkeypatch.setattr(P, "_put", fake_put)

    async def opens():
        return ["OUSDT"]

    async def go():
        assert P.maybe_publish(True, opens, {"AUSDT": {"spike_ts": 1_795_000_000_000}}, ["TUSDT"], ["AUSDT"], now_s=1000.0) is True
        assert P.maybe_publish(True, opens, {}, [], [], now_s=1001.0) is False           # interval
        await asyncio.sleep(0.2)
        assert P.due(now_s=1000.0 + P.PUBLISH_EVERY_S + 1) is False                       # the S3 thread is still busy
        gate.set()
        await P._state["task"]
        assert P.due(now_s=1000.0 + P.PUBLISH_EVERY_S + 1) is True

        async def boom():
            raise RuntimeError("db down")
        assert P.maybe_publish(False, boom, {}, [], [], now_s=1000.0 + P.PUBLISH_EVERY_S + 1) is True
        await P._state["task"]
    asyncio.run(go())
    assert sent[0]["open"] == ["OUSDT"] and sent[0]["flagged"]["AUSDT"].startswith("2026-") and sent[0]["tracked"] == ["TUSDT"]
    assert sent[1].get("open_unread") is True and sent[1]["open"] == []                    # an unreadable DB still publishes
    if P._pool is not None:
        P._pool.shutdown(wait=False)
        P._pool = None


def test_engine_hook_is_guarded_and_bounded():
    with open(os.path.join(ROOT, "services", "trading_engine.py")) as f:
        s = f.read()
    i = s.index("if _route_b.due():")
    blk = s[i - 200:i + 700]
    assert "except Exception as _rb_err:" in blk and "self._route_b_open_pairs" in blk
    call = blk[blk.index("_route_b.maybe_publish("):blk.index("except Exception as _rb_err:")]
    assert "await" not in call.split("# never awaits")[0]                       # the pass never waits on Route B
    assert "async def _route_b_open_pairs(self)" in s
    from services import route_b_publish as P
    assert P.BUCKET == H.BUCKET and P.KEY == "bot/frenzy_list.json" and P.PUBLISH_EVERY_S <= R.HOT_REFRESH_S
