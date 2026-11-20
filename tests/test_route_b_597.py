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
    assert '\\"Action\\":[\\"s3:PutObject\\"],\\"Resource\\":[\\"arn:aws:s3:::$BUCKET/data/*\\",\\"arn:aws:s3:::$BUCKET/health/*\\",\\"arn:aws:s3:::$BUCKET/features/*\\"]' in s   # recorder: no code/* write
    assert "AbortIncompleteMultipartUpload" in s and "aws:SecureTransport" in s and "KillSignal=SIGTERM" in s
    assert "elasticbeanstalk" not in s.replace("aws-elasticbeanstalk-ec2-role", "")   # never touches the EB environment
    assert H.BUCKET in s


# ── the bot side (services/route_b_publish.py) ─────────────────────────────────────────────────────────────────────────────────────

def test_publish_payload_shape():
    from services import route_b_publish as P
    import datetime as dt
    d = P.payload(123, True, [("BUSDT", "EXHAUSTION_SHORT", "SHORT", dt.datetime(2026, 11, 20, 1, 2, 3)), "AUSDT"],
                  {"XUSDT": {"spike_ts": 1_795_000_000_000}, "YUSDT": {}, "ZUSDT": None}, {"CUSDT"}, ["XUSDT", "DUSDT"],
                  sleeves={"EXHAUSTION_SHORT": True})
    assert d["at"] == 123 and d["paper"] is True and d["tracked"] == ["CUSDT"] and d["sleeves"] == {"EXHAUSTION_SHORT": True}
    assert d["open"]["BUSDT"] == {"s": "EXHAUSTION_SHORT", "e": "EXHAUSTION_SHORT", "d": "SHORT", "o": "2026-11-20T01:02:03"}
    assert d["open"]["AUSDT"]["s"] is None and d["recent"] == {}
    assert d["flagged"]["XUSDT"].startswith("2026-") and d["flagged"]["YUSDT"] is None and d["flagged"]["ZUSDT"] is None
    assert d["followed"] == ["XUSDT", "DUSDT"]
    assert R.bot_pairs(dict(d, at=10 ** 13), 10 ** 13) == ["BUSDT", "AUSDT", "XUSDT", "YUSDT", "ZUSDT", "CUSDT", "DUSDT"]   # the recorder reads it


def test_sleeves_on_reads_the_live_switches():
    from types import SimpleNamespace as NS
    from services import route_b_publish as P
    th = NS(exhaustion_short_enabled=True, frenzy_willy_enabled=True, frenzy_willy_ht_enabled=True, frenzy_willy_plain_entries_enabled=False)
    on = P.sleeves_on(th)
    assert on["EXHAUSTION_SHORT"] and on["FRENZY_WILLY_HT"] and not on["FRENZY_WILLY"] and not on["MOMENTUM_LONG"]
    assert set(on) == {r[0] for r in P.SLEEVE_SWITCHES}
    import json as _j
    cfg = _j.load(open(os.path.join(ROOT, "trading_config.json")))
    t = cfg.get("thresholds", cfg)
    assert all(k in t for r in P.SLEEVE_SWITCHES for k in r[1])        # every switch name exists in the live config


def test_sleeve_of_matches_the_engine_strategy_names():
    from services import route_b_publish as P
    assert P.sleeve_of("MOMENTUM", "LONG") == "MOMENTUM_LONG" and P.sleeve_of("MOMENTUM", "SHORT") == "MOMENTUM_SHORT"
    assert P.sleeve_of("FLIP:FAN", "SHORT") == "FLIP" and P.sleeve_of("BULLRUN_LONG", "LONG") == "BULL_RUN"
    assert P.sleeve_of("BEARRUN_SHORT", "SHORT") == "BEAR_RUN" and P.sleeve_of("FRENZY_WILLY_HT", "LONG") == "FRENZY_WILLY_HT"
    assert P.sleeve_of("FRENZY_WILLY", "LONG") == "FRENZY_WILLY" and P.sleeve_of("MANUAL", "LONG") == "MANUAL"
    assert P.sleeve_of("WHATEVER", "LONG") == "OTHER" and P.sleeve_of(None) == "OTHER"


def test_recent_closes_ride_the_list():
    import datetime as dt
    from services import route_b_publish as P
    rows = [("AUSDT", "FRENZY_WILLY", "LONG", dt.datetime(2026, 11, 20, 1, 0), "OPEN", None),
            ("BUSDT", "MOMENTUM", "SHORT", dt.datetime(2026, 11, 20, 1, 1), "CLOSED", dt.datetime(2026, 11, 20, 1, 3))]
    d = P.payload(1, False, rows, {}, [], [])
    assert list(d["open"]) == ["AUSDT"]
    assert d["recent"] == {"BUSDT|2026-11-20T01:01:00": {"p": "BUSDT", "s": "MOMENTUM_SHORT", "e": "MOMENTUM", "d": "SHORT",
                                                         "o": "2026-11-20T01:01:00", "c": "2026-11-20T01:03:00"}}


def test_publish_off_under_pytest_and_never_raises():
    from services import route_b_publish as P
    assert P.enabled() is False and P.due() is False and P.maybe_publish(False, None, {}, [], []) is False   # PYTEST_CURRENT_TEST → off
    assert P.payload(1, False, 5, "x", None, None)["open"] == {}                                         # junk in → empty, no raise


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
        return [("OUSDT", "FRENZY_WILLY_HT", "LONG", None)]

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
    assert sent[0]["open"] == {"OUSDT": {"s": "FRENZY_WILLY_HT", "e": "FRENZY_WILLY_HT", "d": "LONG", "o": None}}
    assert sent[0]["flagged"]["AUSDT"].startswith("2026-")
    assert sent[0]["tracked"] == ["TUSDT"]
    assert sent[1].get("open_unread") is True and sent[1]["open"] == {}                    # an unreadable DB still publishes
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


# ── hourly summaries + the daily report ────────────────────────────────────────────────────────────────────────────────────────

def _hour_files(tmp_path, tag, h0):
    w = R.Writer(str(tmp_path), tag=tag)
    for i in range(130):
        t = h0 + i * 1000
        w.write("book", {"r": t, "s": "AUSDT", "E": t, "T": t, "b": [["99.9", "100"], ["99.5", "100"]], "a": [["100.1", "100"], ["100.6", "100"]]}, t)
        w.write("trade", {"r": t, "d": {"s": "AUSDT", "p": "100", "q": "1", "T": t, "m": i % 3 == 0}}, t)
    w.write("liq", {"r": h0, "d": {"o": {"s": "AUSDT", "S": "SELL", "ap": "100", "z": "5", "q": "5", "T": h0}}}, h0)
    w.write("mark", {"r": h0, "E": h0, "m": [["AUSDT", "100", "100.01", "0.0001", 0]]}, h0)
    w.close()


def test_features_from_several_files_per_hour(tmp_path):
    import features as F
    h0 = 1_795_000_000_000 // 3_600_000 * 3_600_000
    _hour_files(tmp_path, 1, h0)
    _hour_files(tmp_path, 2, h0 + 200_000)                      # a restart in the same hour → a second file per kind
    d, h = R.hour_key(h0)
    g = R.hour_groups(R.closed_files(str(tmp_path), "2999-01-01", "00"))
    assert set(g) == {(d, h)} and len(g[(d, h)]["book"]) == 2
    rows = F.build(g[(d, h)]["book"], g[(d, h)]["trade"], g[(d, h)]["liq"], g[(d, h)]["mark"])
    r0 = [r for r in rows if r["minute"] == h0][0]
    assert r0["n_book"] == 60 and r0["n_trades"] == 60 and abs(r0["spread_bps"] - 20.0) < 1e-6
    assert abs(r0["imb_05"] - (19940 - 10010) / 29950) < 1e-3          # 0.5 % band: both bids, only the first ask (100.6 is outside)
    assert r0["liq_long_usd"] == 500.0 and r0["mark"] == "100"
    assert abs(r0["sell_usd"] - 2000.0) < 1e-6 and abs(r0["buy_usd"] - 4000.0) < 1e-6   # every 3rd trade a taker sell
    assert r0["cost_buy_2500"] != "" and r0["cost_buy_10000"] != ""
    out = tmp_path / "f.csv.gz"
    F.write(rows, str(out))
    import pandas as pd
    assert len(pd.read_csv(out, compression="gzip")) == len(rows)


def test_walk_cost():
    import features as F
    asks = [["100.1", "10"], ["100.2", "100"]]
    assert abs(F.walk_cost_bps(asks, 1001, 100.0, "buy") - 10.0) < 1e-6          # exactly the first level
    assert F.walk_cost_bps(asks, 10 ** 9, 100.0, "buy") != F.walk_cost_bps(asks, 10 ** 9, 100.0, "buy")   # not enough → NaN


def test_recorder_builds_the_summary_once_before_raw_files_go(tmp_path):
    import asyncio
    h0 = 1_795_000_000_000 // 3_600_000 * 3_600_000
    _hour_files(tmp_path, 1, h0)
    rec = R.Recorder(str(tmp_path), "", 10)
    d, h = R.hour_key(h0)
    g = R.hour_groups(R.closed_files(str(tmp_path), "2999-01-01", "00"))
    assert asyncio.run(rec._features((d, h), g[(d, h)])) is True
    assert os.path.exists(os.path.join(tmp_path, "_features", d, f"{h}.done")) and rec.feat_done == 1
    assert asyncio.run(rec._features((d, h), g[(d, h)])) is True and rec.feat_done == 1      # once
    assert R.hour_groups([("events", d, "05-1.jsonl.gz", "x")]) == {(d, "05"): {"events": ["x"]}}
    assert asyncio.run(rec._features((d, "05"), {"events": ["x"]})) is True                   # events-only hour: nothing to build


def test_recorder_gives_up_after_two_failed_summaries(tmp_path, monkeypatch):
    import asyncio
    rec = R.Recorder(str(tmp_path), "", 10)
    bad = {"book": [str(tmp_path / "missing.jsonl.gz")]}
    monkeypatch.setattr(R.sys, "executable", "/nonexistent-python")
    assert asyncio.run(rec._features(("2026-01-01", "00"), bad)) is False       # first failure → raw files held
    assert asyncio.run(rec._features(("2026-01-01", "00"), bad)) is True        # second → give up (raw stays in S3)
    assert rec.feat_fail == 2


def test_daily_report_positions_and_entry_book():
    import pandas as pd
    import daily_report as DR
    ev = [{"ev": "bot", "r": 1, "open": {"AUSDT": {"s": "EXHAUSTION_SHORT", "d": "SHORT", "o": "2026-11-20T01:02:30"}}, "sleeves": {"EXHAUSTION_SHORT": True}},
          {"ev": "bot", "r": 2, "open": {"AUSDT": {"s": "EXHAUSTION_SHORT", "d": "SHORT", "o": "2026-11-20T01:02:30"},
                                         "BUSDT": {"s": "FRENZY_WILLY_HT", "d": "LONG", "o": "2026-11-20T03:00:00"}}, "flagged": ["BUSDT"]},
          {"ev": "hot", "r": 3}]
    p = DR.positions(ev)
    assert len(p) == 2 and set(p.sleeve) == {"EXHAUSTION_SHORT", "FRENZY_WILLY_HT"}
    m = int(pd.Timestamp("2026-11-20T01:02:00", tz="UTC").value // 1_000_000)
    feats = pd.DataFrame([dict(sym="AUSDT", minute=m, n_book=100, cost_buy_2500=5.0, cost_sell_2500=6.0, cost_buy_10000=12.0, cost_sell_10000=14.0)])
    e = DR.entry_book(p, feats).set_index("pair")
    assert bool(e.loc["AUSDT", "recorded"]) and e.loc["AUSDT", "cost25"] == 6.0 and e.loc["AUSDT", "cost10"] == 14.0   # a SHORT sells in
    assert not bool(e.loc["BUSDT", "recorded"])
    assert DR.last_sleeves(ev) == {"EXHAUSTION_SHORT": True}


def test_health_flags_lagging_summaries():
    now = 10 ** 13
    base = {"now": now - 60_000, "age_s": {"mark": 1, "book": 1, "trade": 1}, "bot_n": 5, "disk_free_gb": 20, "started": now - 10 * 3_600_000,
            "features_done": 0, "features_last": None}
    assert H.verdict(base, now)[0] == 1
    assert H.verdict(dict(base, started=now - 3_600_000), now)[0] == 0                # young process: not judged yet


def test_daily_colours_follow_fixed_rules():
    import daily_report as DR
    G, Y, R_, W = DR.GREEN, DR.YELLOW, DR.RED, DR.WHITE
    assert DR.coverage_colour(0, 10, 10) == G and DR.coverage_colour(0, 10, 9) == G          # ≥ 90 %
    assert DR.coverage_colour(0, 10, 8) == Y and DR.coverage_colour(0, 10, 5) == Y           # 50–90 %
    assert DR.coverage_colour(0, 10, 4) == R_                                                 # < 50 %
    assert DR.coverage_colour(1, 10, 10) == Y                                                 # recorder degraded
    assert DR.coverage_colour(2, 10, 10) == R_ and DR.coverage_colour(2, 0, 0) == R_          # recorder down
    assert DR.coverage_colour(0, 0, 0) == W                                                   # no trades
    assert DR.summaries_colour(24) == G and DR.summaries_colour(21) == Y and DR.summaries_colour(5) == R_
    assert DR.summaries_colour(3, day_complete=False) == W
    assert DR.summaries_colour(20) == Y and DR.summaries_colour(19) == R_


def test_day_health_uses_that_days_record():
    import daily_report as DR
    d0 = 1795046400000                                                   # 2026-11-19 00:00 UTC
    ok = {"day_gap_s": {"market": 0, "book": 10, "trade": 0}, "started": d0 - 1000, "bot_n": 40}
    assert DR.day_health("2026-11-19", "2026-11-20", ok, 2)[0] == 0                    # healthy that day, down now → still 🟢
    assert DR.day_health("2026-11-19", "2026-11-20", None, 0)[0] == 2                  # no daily record
    assert DR.day_health("2026-11-19", "2026-11-20", dict(ok, day_gap_s={"book": 4000}), 0)[0] == 2
    assert DR.day_health("2026-11-19", "2026-11-20", dict(ok, day_gap_s={"book": 400}), 0)[0] == 1
    assert DR.day_health("2026-11-19", "2026-11-20", dict(ok, started=d0 + 3_600_000), 0)[0] == 1   # restarted during the day
    assert DR.day_health("2026-11-19", "2026-11-20", dict(ok, bot_n=0), 0)[0] == 1
    assert DR.day_health("2026-11-20", "2026-11-20", None, 1) == (1, "live")             # today → the live verdict


def test_one_report_market_sections(tmp_path):
    """the daily report carries the market-data analysis (first_view.sections) — one report, DECISION_LOG 598."""
    import numpy as np
    import pandas as pd
    import daily_report as DR
    import first_view as FV
    assert FV.bot_sets({"open": {"A": 1}, "flagged": ["B"], "tracked": ["C"]}) == ({"A"}, {"B"}, {"C"}, set())
    assert FV.bot_sets({"open": ["A"], "flagged": {"B": {}}}) == ({"A"}, {"B"}, set(), set())
    d = int(pd.Timestamp("2026-11-20", tz="UTC").value // 1_000_000)
    ev = [{"ev": "bot", "r": d + 1000, "open": {"AUSDT": {"s": "X"}}, "flagged": ["BUSDT"], "tracked": ["CUSDT"],
           "recent": {"k": {"p": "EUSDT", "s": "X", "o": "2026-11-20T00:00:30"}}},
          {"ev": "bot", "r": d + 2000, "flagged": ["DUSDT"]},
          {"ev": "bot", "r": d - 1000, "flagged": ["OLDUSDT"]}]                     # previous day → not counted
    doc = DR.day_bot_doc(ev, "2026-11-20")
    assert doc["open"] == ["AUSDT", "EUSDT"] and "followed" not in doc and doc["flagged"] == ["BUSDT", "DUSDT"] and doc["tracked"] == ["CUSDT"]
    rng = np.random.default_rng(1)
    rows = []
    for s in ("AUSDT", "BUSDT", "CUSDT"):
        mid = 1.0
        for i in range(90):
            mid *= 1 + rng.normal(0, 1e-3)
            rows.append(dict(sym=s, minute=d + i * 60_000, n_book=120, mid=mid, spread_bps=3.0, bid_usd_05=2e4, ask_usd_05=2e4,
                             imb_05=rng.uniform(-1, 1), cost_buy_2500=5.0, cost_sell_2500=5.0, cost_buy_10000=12.0, cost_sell_10000=12.0,
                             n_trades=10, buy_usd=rng.uniform(1e3, 1e4), sell_usd=rng.uniform(1e3, 1e4), liq_long_usd=0.0, liq_short_usd=0.0))
    f = tmp_path / "00.csv.gz"
    pd.DataFrame(rows).to_csv(f, index=False, compression="gzip")
    png = tmp_path / "day.png"
    L = FV.sections(doc, [str(f)], str(png))
    txt = "\n".join(L)
    for h in ("## ① What was recorded", "## ② What it costs", "## ③ Does the book", "## ④ Liquidations", "## Data and caveats"):
        assert h in txt
    assert "Covered in full detail: 3 of 5" in txt          # DUSDT + EUSDT (recent close) not recorded
    assert png.exists()
    assert "chart skipped" in "\n".join(FV.sections(doc, [str(f)], None))         # no chart path → no file, still a report


def test_non_overlap_counts_possibilities_not_events():
    import numpy as np
    import first_view as FV
    m = np.array([0, 1, 2, 3, 4, 5, 6, 20, 21]) * 60_000          # 9 signal minutes on one coin
    k = FV.non_overlap(m, 5)
    assert list(m[k] // 60_000) == [0, 5, 20]                      # 3 trades, not 9 events
    assert sorted(m[::-1][FV.non_overlap(m[::-1], 15)] // 60_000) == [0, 20]   # input order does not matter


def test_wl_ready_flag():
    import daily_report as DR
    assert DR.wl_flag(5) == "5/30"
    assert DR.wl_flag(29) == "29/30"
    assert DR.wl_flag(30).startswith("🟢 30/30") and "ready" in DR.wl_flag(30)
    assert DR.wl_flag(41).startswith("🟢 41/30")


def test_split_minute_merged_not_double_counted():
    import pandas as pd
    import first_view as FV
    a = pd.DataFrame([dict(sym="AUSDT", minute=0, n_book=80, mid=1.0, n_trades=3, buy_usd=10.0, sell_usd=5.0, liq_long_usd=0.0, liq_short_usd=0.0, max_trade_usd=4.0),
                      dict(sym="AUSDT", minute=0, n_book=40, mid=2.0, n_trades=2, buy_usd=1.0, sell_usd=1.0, liq_long_usd=7.0, liq_short_usd=0.0, max_trade_usd=9.0),
                      dict(sym="AUSDT", minute=60_000, n_book=120, mid=1.1, n_trades=1, buy_usd=1.0, sell_usd=0.0, liq_long_usd=0.0, liq_short_usd=0.0, max_trade_usd=1.0)])
    m = FV.merge_split_minutes(a).sort_values("minute").set_index("minute")
    assert len(m) == 2
    r = m.loc[0]
    assert r.mid == 1.0 and r.n_book == 80                       # book from the fuller row
    assert r.n_trades == 5 and r.buy_usd == 11.0 and r.sell_usd == 6.0 and r.liq_long_usd == 7.0 and r.max_trade_usd == 9.0
