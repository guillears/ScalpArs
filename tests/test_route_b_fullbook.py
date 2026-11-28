"""📡 Route B — full order book for the bot's priority coins (scripts/route_b/fullbook.py + the recorder / summary / health wiring).
Binance futures diff-depth sync (U / u / pu), resync on gaps, trusted edges, refresh without a gap, walk cost + bands, priority set
and caps, REST budget, the fb_* summary columns, the health block. Pure; no network."""
import gzip
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b"))

import fullbook as FB         # noqa: E402
import recorder as R          # noqa: E402
import health_check as H      # noqa: E402


def ev(U, u, pu, b=(), a=(), E=0):
    return {"e": "depthUpdate", "E": E, "s": "XUSDT", "U": U, "u": u, "pu": pu,
            "b": [[str(p), str(q)] for p, q in b], "a": [[str(p), str(q)] for p, q in a]}


SNAP = {"lastUpdateId": 100, "bids": [["99", "10"], ["98", "20"]], "asks": [["101", "10"], ["102", "20"]]}


def synced_book():
    bk = FB.LocalBook("XUSDT")
    bk.on_event(ev(95, 99, 94))                                    # older than the snapshot → dropped at sync
    bk.on_event(ev(100, 103, 99, b=[(99, 5)]))                     # U ≤ 100 ≤ u → first applied
    assert bk.on_snapshot(SNAP, 1000) == "synced"
    return bk


# ── the sync procedure ───────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_sync_drops_old_events_and_applies_the_bridging_one():
    bk = synced_book()
    assert bk.synced and bk.last_u == 103 and bk.bids[99.0] == 5.0 and bk.asks == {101.0: 10.0, 102.0: 20.0}
    assert bk.on_event(ev(104, 105, 103, a=[(101, 0)])) == "applied"      # qty 0 removes the level
    assert 101.0 not in bk.asks and bk.last_u == 105
    assert bk.on_event(ev(104, 105, 103)) == "old"                         # a repeat is ignored


def test_snapshot_older_than_the_stream_is_stale_and_asked_again():
    bk = FB.LocalBook("XUSDT")
    bk.on_event(ev(150, 155, 149))                                 # first buffered event starts AFTER the snapshot's id
    assert bk.on_snapshot(SNAP, 1000) == "stale" and not bk.synced and bk.want


def test_snapshot_newer_than_the_stream_waits_for_it():
    bk = FB.LocalBook("XUSDT")
    bk.on_event(ev(90, 95, 89))
    assert bk.on_snapshot(SNAP, 1000) == "waiting" and not bk.synced
    assert bk.on_event(ev(96, 101, 95, b=[(98, 0)])) == "synced"   # 96 ≤ 100 ≤ 101
    assert 98.0 not in bk.bids and bk.last_u == 101


def test_live_gap_unsyncs_and_a_new_snapshot_resyncs():
    bk = synced_book()
    assert bk.on_event(ev(110, 112, 109)) == "gap"                 # pu 109 ≠ last u 103
    assert not bk.synced and bk.bids == {} and bk.want and len(bk.buf) == 1
    assert bk.needs_snapshot(10 ** 7)
    bk.mark_requested(10 ** 7)
    assert not bk.needs_snapshot(10 ** 7)                          # one request at a time
    bk.on_event(ev(113, 115, 112, b=[(97, 1)]))
    snap2 = {"lastUpdateId": 111, "bids": [["99", "1"]], "asks": [["101", "1"]]}
    assert bk.on_snapshot(snap2, 10 ** 7) == "synced" and bk.last_u == 115 and bk.bids == {99.0: 1.0, 97.0: 1.0}


def test_a_gap_inside_the_buffer_keeps_what_follows_and_asks_again():
    bk = FB.LocalBook("XUSDT")
    bk.on_event(ev(100, 103, 99))
    bk.on_event(ev(110, 112, 109))                                 # gap between buffered events
    assert bk.on_snapshot(SNAP, 1000) == "gap" and not bk.synced and bk.want
    assert [e["u"] for e in bk.buf] == [112]


def test_trusted_edges_ignore_levels_beyond_a_full_snapshot():
    bk = FB.LocalBook("XUSDT")
    bk.on_event(ev(100, 101, 99))
    full = {"lastUpdateId": 100, "bids": [["99", "1"], ["98", "1"]], "asks": [["101", "1"], ["102", "1"]]}
    assert bk.on_snapshot(full, 0, limit=2) == "synced"            # 2 levels = the limit → both sides partial
    assert bk.edge_bid == 98.0 and bk.edge_ask == 102.0
    bk.on_event(ev(102, 102, 101, b=[(90, 50)], a=[(110, 50)]))    # beyond the edges: never half-known
    assert 90.0 not in bk.bids and 110.0 not in bk.asks
    m = FB.book_metrics(bk.bids, bk.asks, bk.edge_bid, bk.edge_ask)
    assert abs(m["rb"] - 2.0) < 1e-9 and abs(m["ra"] - 2.0) < 1e-9 # reach 2 % each side
    bk2 = synced_book()                                            # 2 levels < 1,000 → complete sides
    assert bk2.edge_bid == 0.0 and bk2.edge_ask == FB.INF
    assert FB.book_metrics(bk2.bids, bk2.asks)["rb"] == 100.0


def test_refresh_swaps_a_new_snapshot_in_without_a_gap():
    bk = synced_book()
    bk.begin_refresh(5000)
    assert bk.refreshing and bk.synced and bk.needs_snapshot(10 ** 7) is False  # nothing buffered yet
    assert bk.on_event(ev(104, 106, 103, b=[(99, 7)])) == "applied" and bk.bids[99.0] == 7.0   # still served live
    assert bk.needs_snapshot(10 ** 7)
    snap = {"lastUpdateId": 105, "bids": [["99", "6"], ["97", "3"]], "asks": [["101", "10"]]}
    assert bk.on_snapshot(snap, 6000) == "refreshed"
    assert bk.synced and not bk.refreshing and bk.bids == {99.0: 7.0, 97.0: 3.0} and bk.last_u == 106


def test_refresh_gives_up_after_three_stale_snapshots_and_keeps_the_live_book():
    bk = synced_book()
    bk.begin_refresh(0)
    for i in range(3):
        bk.on_event(ev(104 + 2 * i, 105 + 2 * i, 103 + 2 * i))
        assert bk.on_snapshot({"lastUpdateId": 50, "bids": [], "asks": []}, 0) in ("stale", "waiting")
        bk.buf.clear() if i == 2 else None
    assert bk.synced and not bk.refreshing


def test_refresh_due_after_an_hour_or_a_drift_toward_the_edge():
    bk = synced_book()
    bk.synced_ms, bk.reach0 = 0, 4.0
    assert not bk.refresh_due(60_000, 3.9)
    assert bk.refresh_due(FB.FB_REFRESH_S * 1000, 3.9)
    assert bk.refresh_due(FB.FB_REFRESH_MIN_GAP_S * 1000, 1.5)    # reach fell under half
    assert not bk.refresh_due(1000, 1.5)                           # … but not within 5 min of the sync


def test_buffer_overflow_is_counted():
    bk = FB.LocalBook("XUSDT", max_buffer=3)
    for i in range(5):
        bk.on_event(ev(i, i, i - 1))
    assert len(bk.buf) == 3 and bk.overflow == 2


def test_bad_events_never_raise():
    bk = FB.LocalBook("XUSDT")
    assert bk.on_event({"s": "XUSDT"}) == "bad"
    assert bk.on_snapshot({"bids": []}) == "bad"


# ── the derived numbers ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_bands_and_walk_cost():
    bids = {99.95: 10.0, 99.0: 100.0, 96.0: 1000.0}
    asks = {100.05: 10.0, 101.0: 100.0, 104.0: 1000.0}
    m = FB.book_metrics(bids, asks)
    assert m["mid"] == 100.0 and abs(m["sp"] - 10.0) < 1e-9
    b = dict(zip(FB.FB_BANDS, m["b"]))
    assert abs(b[0.1] - 999.5) < 1e-6 and abs(b[1.0] - (999.5 + 9900)) < 1e-6 and abs(b[5.0] - (999.5 + 9900 + 96000)) < 1e-6
    c = dict(zip(FB.FB_SIZES, m["cb"]))
    assert abs(c[1000] - 5.0) < 1e-6                               # all at 100.05 → 5 bps
    exp = 3000 / (1000.5 / 100.05 + 1999.5 / 101.0)                # 1,000.5 $ at 100.05 + the rest at 101
    assert abs(c[3000] - (exp / 100 - 1) * 1e4) < 1e-6
    assert c[25000] is not None and FB.walk_cost_bps([(100.05, 1.0)], 10 ** 6, 100.0, "buy") is None
    rec = FB.fbook_record(FB.LocalBook("XUSDT"), m, 1)
    assert len(rec["b"]) == len(FB.FB_BANDS) and len(rec["cs"]) == len(FB.FB_SIZES)
    json.dumps(rec)


def test_crossed_book_is_rejected():
    assert FB.book_metrics({101.0: 1.0}, {100.0: 1.0}) is None and FB.crossed({101.0: 1.0}, {100.0: 1.0})
    assert FB.book_metrics({}, {100.0: 1.0}) is None and not FB.crossed({}, {100.0: 1.0})


def test_fsnap_keeps_only_the_window():
    bk = synced_book()
    bk.bids[50.0] = 1.0
    m = FB.book_metrics(bk.bids, bk.asks, with_levels=True)
    rec = FB.fsnap_record(bk, m, 1)
    assert [p for p, _ in rec["b"]] == [99.0, 98.0] and rec["eb"] is None and rec["ea"] is None


# ── priority set, budget ─────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_pick_full_priority_cap_sticky_and_staleness():
    ms = 10 ** 13
    doc = {"at": ms - 60_000, "open": {"OPENUSDT": {}}, "flagged": {"OLDUSDT": "2026-11-20T01:00:00", "NEWUSDT": "2026-11-21T01:00:00"},
           "tracked": ["AUSDT", "BUSDT", "CUSDT", "OPENUSDT"], "followed": ["FOLLOWUSDT"]}
    seen = {}
    new, n = FB.pick_full(doc, ms, ["CUSDT"], seen)
    assert new == ["OPENUSDT", "NEWUSDT", "OLDUSDT", "CUSDT", "AUSDT", "BUSDT"] and n == 6    # followed is not priority
    new, n = FB.pick_full(doc, ms, [], {}, cap=3)
    assert new == ["OPENUSDT", "NEWUSDT", "OLDUSDT"] and n == 6
    doc2 = dict(doc, at=ms + 60_000, tracked=[])                   # A / B / C leave the list …
    new, _ = FB.pick_full(doc2, ms + 60_000, ["OPENUSDT", "AUSDT", "CUSDT"], seen)
    assert new[:3] == ["OPENUSDT", "NEWUSDT", "OLDUSDT"] and set(new[3:]) == {"AUSDT", "CUSDT"}       # … sticky
    new, _ = FB.pick_full(dict(doc2, at=ms + FB.FULL_STICKY_S * 1000), ms + FB.FULL_STICKY_S * 1000, new, seen)
    assert "AUSDT" not in new                                     # sticky expired
    new, _ = FB.pick_full(doc, ms, ["OPENUSDT"], {}, tradable={"NEWUSDT"})
    assert new == ["NEWUSDT"]                                     # delisted / not tradable → out at once
    assert FB.pick_full(dict(doc, at=ms - 31 * 60_000), ms, [], {})[0] == []   # stale bot list
    assert FB.pick_full(None, ms, [], {})[0] == [] and FB.pick_full({"at": "x"}, ms, [], {})[0] == []


def test_weight_bucket_and_rest_pause():
    wb = FB.WeightBucket(per_min=60, burst=40, now_s=0)
    assert wb.take(20, 0) and wb.take(20, 0) and not wb.take(20, 0)
    assert wb.take(20, 20)                                        # 1 weight / s refills
    assert FB.rest_pause_s(200, {"X-MBX-USED-WEIGHT-1M": "100"}, 0) == 0
    assert FB.rest_pause_s(200, {"x-mbx-used-weight-1m": "900"}, 30) == 31
    assert FB.rest_pause_s(429, {}, 0) == 120 and FB.rest_pause_s(429, {"Retry-After": "7"}, 0) == 7
    assert FB.rest_pause_s(418, {}, 0) == 600 and FB.rest_pause_s(500, {}, 0) == 0   # a 5xx / 400 is per coin, never global
    assert FB.pause_reason(429, {}) == 429 and FB.pause_reason(200, {"x-mbx-used-weight-1m": "900"}) == "weight" and FB.pause_reason(400, {}) is None


# ── recorder wiring ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

def _rec_with_synced_book(tmp_path):
    rec = R.Recorder(str(tmp_path), "", 10)
    rec.set_full(["XUSDT"])
    rec.hot_ws["fbook"] = object()                                 # "connected"
    bk = rec.books["XUSDT"]
    rec.on_fb_msg(json.dumps({"stream": "xusdt@depth@100ms", "data": ev(100, 103, 99, b=[(99, 5)])}))
    rec._fb_result(bk.on_snapshot(SNAP, 0))
    return rec, bk


def test_recorder_routes_diffs_and_writes_fbook_and_fsnap(tmp_path):
    rec, bk = _rec_with_synced_book(tmp_path)
    assert bk.synced and rec.fb_tot["syncs"] == 1
    t0 = 1_795_000_000_000
    assert rec.fb_tick(t0) == 1 and rec.fb_tick(t0 + 1000) == 1
    rec.set_full(["XUSDT", "YUSDT"])                               # a second coin not synced yet
    rec.fb_tick(t0 + 2000)
    st = rec.full_status(t0 + 2000)
    assert st["synced_n"] == 1 and st["n"] == 2 and st["not_synced"] == ["YUSDT"] and st["day"]["want_s"] == 4
    assert st["coverage_day_pct"] == 75.0
    rec.on_fb_msg(json.dumps({"stream": "xusdt@depth@100ms", "data": ev(200, 201, 150)}))   # a gap
    assert rec.fb_tot["gaps"] == 1 and not bk.synced
    rec.w.close()
    d, h = R.hour_key(t0)
    fb = [json.loads(x) for f in os.listdir(tmp_path / "fbook" / d) for x in gzip.open(tmp_path / "fbook" / d / f, "rt")]
    fs = [json.loads(x) for f in os.listdir(tmp_path / "fsnap" / d) for x in gzip.open(tmp_path / "fsnap" / d / f, "rt")]
    assert len(fb) == 3 and len(fs) == 1 and fb[0]["s"] == "XUSDT" and fb[0]["bb"] == 99.0 and fs[0]["b"][0] == [99.0, 5.0]


def test_recorder_unsyncs_a_crossed_book_and_never_emits_while_disconnected(tmp_path):
    rec, bk = _rec_with_synced_book(tmp_path)
    rec.hot_ws.pop("fbook")
    assert rec.fb_tick(1_795_000_000_000) == 0
    rec.hot_ws["fbook"] = object()
    bk.bids[200.0] = 1.0
    assert rec.fb_tick(1_795_000_000_000) == 0 and not bk.synced and rec.fb_tot["crossed"] == 1
    rec.w.close()


def test_recorder_full_off_and_kinds(tmp_path):
    assert "fbook" in R.KINDS and "fsnap" in R.KINDS and "fbook" in R.FEATURE_KINDS and "fsnap" not in R.FEATURE_KINDS
    rec = R.Recorder(str(tmp_path), "", 10, full_max=0)
    assert not rec.full_on and rec.status()["full"]["on"] is False


# ── the hourly summary ───────────────────────────────────────────────────────────────────────────────────────────────────────────

OLD_COLS = ["sym", "minute", "n_book", "mid", "mid_open", "mid_hi", "mid_lo", "spread_bps",
            "bid_usd_02", "ask_usd_02", "bid_usd_05", "ask_usd_05", "bid_usd_1", "ask_usd_1", "imb_05", "reach_bid_pct", "reach_ask_pct",
            "cost_buy_2500", "cost_sell_2500", "cost_buy_10000", "cost_sell_10000",
            "n_trades", "buy_usd", "sell_usd", "max_trade_usd", "vwap", "liq_long_usd", "liq_short_usd", "mark", "index", "funding"]


def test_summary_fb_columns(tmp_path):
    import features as F
    assert F.COLS[:len(OLD_COLS)] == OLD_COLS                       # old columns untouched, new ones appended
    assert "fb_bid_usd_05" in F.COLS and "fb_cost_sell_25000" in F.COLS and "n_fbook" in F.COLS
    m0 = 1_795_000_000_000 // 60_000 * 60_000
    p = tmp_path / "fb.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for i, (b05, c25) in enumerate([(1000, 30.0), (3000, None)]):
            f.write(json.dumps({"r": m0 + i * 1000, "s": "XUSDT", "sp": 10.0, "rb": 2.0, "ra": 100.0,
                                "b": [1, 2, b05, 4, 5, 6], "a": [1, 2, 3, 4, 5, 6], "cb": [1, 2, 3, 4, c25], "cs": [1, 2, 3, 4, 50.0]}) + "\n")
    rows = F.build([], fbook=[str(p)])
    assert len(rows) == 1
    r = rows[0]
    assert r["n_fbook"] == 2 and r["fb_bid_usd_05"] == 2000.0 and r["fb_cost_buy_25000"] == 30.0 and r["fb_cost_na_25000"] == 1
    assert r["fb_reach_ask_pct"] == 100.0 and r["n_book"] == 0
    out = tmp_path / "f.csv.gz"
    F.write(rows, str(out))
    import pandas as pd
    df = pd.read_csv(out, compression="gzip")
    assert list(df.columns) == F.COLS and df.fb_cost_sell_25000.iloc[0] == 50.0


def test_split_minute_merge_keeps_the_full_book_row():
    import pandas as pd
    import first_view as FV
    a = pd.DataFrame([{"sym": "X", "minute": 0, "n_book": 100, "n_trades": 1, "n_fbook": None, "fb_bid_usd_05": None},
                      {"sym": "X", "minute": 0, "n_book": 2, "n_trades": 1, "n_fbook": 60, "fb_bid_usd_05": 5000.0}])
    m = FV.merge_split_minutes(a)
    assert len(m) == 1 and m.n_book.iloc[0] == 100 and m.fb_bid_usd_05.iloc[0] == 5000.0 and m.n_trades.iloc[0] == 2


# ── health ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_health_full_book_block():
    now = 10 ** 13
    full = {"on": True, "cap": 30, "n_want": 35, "n": 30, "synced_n": 29, "not_synced": ["AUSDT"], "conn": True,
            "day": {"want_s": 7200, "synced_s": 7100, "gaps": 2, "stale": 0}, "totals": {}, "coverage_day_pct": 98.6,
            "rest": {"paused_s": 0, "n418": 0, "n429": 0}}
    ok = {"now": now - 60_000, "started": now - 3600_000, "age_s": {"mark": 1, "book": 2, "trade": 3, "fbook": 1}, "bot_n": 40,
          "disk_free_gb": 20, "full": full}
    assert H.verdict(ok, now)[0] == 0
    assert H.verdict(dict(ok, full=dict(full, synced_n=10)), now)[0] == 1
    assert H.verdict(dict(ok, full=dict(full, conn=False)), now)[0] == 1
    assert H.verdict(dict(ok, full=dict(full, coverage_day_pct=80.0)), now)[0] == 1
    assert H.verdict(dict(ok, full=dict(full, rest={"paused_s": 300, "pause_why": 418, "n418": 1, "n429": 0})), now)[0] == 1
    assert H.verdict(dict(ok, full=dict(full, rest={"paused_s": 30, "pause_why": "weight", "n418": 1, "n429": 0})), now)[0] == 0   # old 418
    assert H.verdict(dict(ok, full={"on": False, "requested": 30, "module": False}), now)[0] == 1       # fullbook.py failed to load
    assert H.verdict(dict(ok, full={"on": False, "requested": 0, "module": True}), now)[0] == 0         # FULL_MAX 0 = off on purpose
    assert H.verdict(dict(ok, started=now - 1800_000, full=dict(full, coverage_day_pct=50.0)), now)[0] == 0   # coverage only after 1 h up
    assert H.verdict(dict(ok, started=now - 60_000, full=dict(full, synced_n=0)), now)[0] == 0      # warm-up
    assert H.verdict(dict(ok, full=None), now)[0] == 0                                             # older recorder
    assert H.verdict(dict(ok, full=dict(full, synced_n=0)), now)[0] == 1                           # never ❌ for the full book
    lines = H.full_lines(full)
    assert "29 / 30" in lines[0] and "5 left out by the cap" in lines[0] and "gaps → resyncs 2" in lines[1]
    assert "not reported" in H.full_lines(None)[0]


def test_deploy_ships_fullbook():
    with open(os.path.join(ROOT, "scripts", "route_b", "deploy_aws.sh")) as f:
        s = f.read()
    assert 's3://$BUCKET/code/fullbook.py' in s and '"$HERE/fullbook.py"' in s and "Environment=FULL_MAX=$FULL_MAX" in s
    assert not math.isnan(FB.FB_WEIGHT_PER_MIN) and FB.FB_WEIGHT_PER_MIN + 40 < 1200   # our ≤ 1,200 / min rule, ticker included


# ── ⑤ open interest + positioning (scripts/route_b/positioning.py) ─────────────────────────────────────────────────────────────

import positioning as PS      # noqa: E402


def test_oi_track_changes_tolerate_one_missed_sweep():
    t = PS.OiTrack()
    assert t.feed("X", 100.0, 10) == (None, None)
    assert t.feed("X", 110.0, 11) == (10.0, None)
    for m, v in ((12, 110.0), (13, 110.0), (14, 110.0)):
        t.feed("X", v, m)
    c1, c5 = t.feed("X", 99.0, 15)                                   # 5 min earlier = minute 10 (100)
    assert abs(c1 - (-10.0)) < 1e-9 and abs(c5 - (-1.0)) < 1e-9
    c1, c5 = t.feed("X", 99.0, 17)                                   # minute 16 missed → minute 15 used (one older, never more)
    assert c1 == 0.0 and abs(c5 - (99.0 / 110 - 1) * 100) < 1e-3    # 5 min → minute 12 (11 not needed)
    assert t.feed("X", 99.0, 30)[0] is None                          # a long hole → no change, never a stale one
    t.prune({"Y"})
    assert t.d == {}


def test_order_and_ratio_budget():
    assert PS.order_coins(["A", "B"], ["B", "C", "D"], 3) == ["A", "B", "C"]
    coins = [f"C{i}" for i in range(150)]
    take, skipped = PS.ratio_coins(coins, 0)
    assert len(take) == 120 and skipped == 30                       # 480 requests per sweep / 4 endpoints
    take, skipped = PS.ratio_coins(coins, 480)                      # the previous sweep is still in the 300 s window: no starvation
    assert len(take) == 120
    take, skipped = PS.ratio_coins(coins, 940)                      # the rolling guard (960) still binds
    assert len(take) == 5 and take == coins[:5]
    assert PS.ratio_coins(coins, 1000) == ([], 150)
    r = PS.Rolling(300)
    r.add(0, 3)
    assert r.count(299) == 3 and r.count(300) == 0
    assert "period=5m&limit=1" in PS.ratio_path(PS.RATIO_ENDPOINTS[0][1], "XUSDT")
    assert len(PS.RATIO_ENDPOINTS) == 4


def test_parsers():
    assert PS.parse_oi({"openInterest": "123.5", "symbol": "X", "time": 7}) == (123.5, 7)
    assert PS.parse_oi({"code": -1121}) is None and PS.parse_oi(None) is None
    body = [{"symbol": "X", "longShortRatio": "1.5", "timestamp": 1}, {"symbol": "X", "longShortRatio": "2.0", "timestamp": 2}]
    assert PS.parse_ratio(body, "longShortRatio")[:2] == (2, 2.0)
    assert PS.parse_ratio([], "longShortRatio") is None and PS.parse_ratio({"code": 1}, "x") is None
    tk = PS.parse_ratio([{"buySellRatio": "0.8", "buyVol": "10", "sellVol": "12.5", "timestamp": 3}], "buySellRatio")
    row = PS.ratio_row("X", {"tla": (2, 2.0, {}), "tpa": None, "gla": (2, 1.1, {}), "tkr": tk})
    assert row == ["X", 3, 2.0, None, 1.1, 0.8, 10.0, 12.5]
    assert PS.ratio_row("X", {}) is None


class _FakeRest:
    """→ canned (status, headers, body) per path prefix; records every call."""

    def __init__(self, oi_status=200, ratio_status=200):
        self.calls, self.oi_status, self.ratio_status = [], oi_status, ratio_status

    def __call__(self, path):
        self.calls.append(path)
        if path.startswith("/fapi/v1/openInterest"):
            sym = path.split("=")[1]
            return self.oi_status, {"X-MBX-USED-WEIGHT-1M": "42"}, ({"openInterest": "1000", "symbol": sym, "time": 5}
                                                                    if self.oi_status == 200 else None)
        field = "buySellRatio" if "taker" in path else "longShortRatio"
        return self.ratio_status, ({"Retry-After": "30"} if self.ratio_status == 429 else {}), \
            ([{field: "1.25", "timestamp": 9, "buyVol": "1", "sellVol": "2"}] if self.ratio_status == 200 else None)


def _pos_rec(tmp_path, fake):
    import asyncio
    rec = R.Recorder(str(tmp_path), "", 10, full_max=0)
    rec.hot, rec.full = ["AUSDT", "BUSDT"], []
    rec._rest = fake
    PS_GAP, PS.REQ_GAP_S = PS.REQ_GAP_S, 0
    return rec, asyncio, PS_GAP


def test_oi_and_ratio_sweeps_write_rows(tmp_path):
    fake = _FakeRest()
    rec, asyncio, gap = _pos_rec(tmp_path, fake)
    try:
        t0 = 1_795_000_000_000 // 60_000 * 60_000
        assert asyncio.run(rec.oi_sweep(t0)) == 2 and rec.fb_rest["used_1m"] == "42"
        assert asyncio.run(rec.ratio_sweep(t0)) == 2 and len(fake.calls) == 2 + 8
        st = rec.status()["pos"]
        assert st["oi_ok"] == 2 and st["ratio_ok"] == 2 and st["oi_coverage_day_pct"] == 100.0 and st["ratio_5m"] == 8
        rec.w.close()
        import glob
        oi = [json.loads(x) for f in glob.glob(str(tmp_path / "oi" / "*" / "*.gz")) for x in gzip.open(f, "rt")]
        ra = [json.loads(x) for f in glob.glob(str(tmp_path / "ratio" / "*" / "*.gz")) for x in gzip.open(f, "rt")]
        assert oi[0]["r"] == t0 and R.hour_key(t0)[0] not in os.listdir(tmp_path / "oi")   # filed by write time, "r" = the sweep
        assert oi[0]["m"][0] == ["AUSDT", 1000.0, 5, None, None] and ra[0]["m"][0] == ["AUSDT", 9, 1.25, 1.25, 1.25, 1.25, 1.0, 2.0]
    finally:
        PS.REQ_GAP_S = gap


def test_429_and_418_stop_the_sweeps(tmp_path):
    fake = _FakeRest(oi_status=429, ratio_status=429)
    rec, asyncio, gap = _pos_rec(tmp_path, fake)
    try:
        asyncio.run(rec.oi_sweep(0))
        assert len(fake.calls) <= PS.CONCURRENCY and rec.fb_rest["n429"] >= 1 and rec.fb_rest["paused_until"] > R.now_ms()   # requests in flight finish; no new ones
        rec.fb_rest["paused_until"] = 0
        asyncio.run(rec.ratio_sweep(0))
        assert rec.pos["ratio_n429"] >= 1 and rec.ratio_paused_until > R.now_ms() + 200_000
        assert rec.fb_rest["paused_until"] == 0                      # /futures/data 429 pauses only the ratios
        fake2 = _FakeRest(ratio_status=418)
        rec2, _, _ = _pos_rec(tmp_path / "b", fake2)
        asyncio.run(rec2.ratio_sweep(0))
        assert len(fake2.calls) <= PS.CONCURRENCY and rec2.fb_rest["n418"] >= 1 and rec2.fb_rest["paused_until"] > R.now_ms() + 500_000
        rec.w.close()
        rec2.w.close()
    finally:
        PS.REQ_GAP_S = gap


def test_summary_oi_and_ratio_columns(tmp_path):
    import features as F
    assert F.COLS[:len(OLD_COLS)] == OLD_COLS and F.COLS[-len(F.POS_COLS):] == F.POS_COLS
    m0 = 1_795_000_000_000 // 3_600_000 * 3_600_000
    po, pr, pm = tmp_path / "oi.jsonl.gz", tmp_path / "ratio.jsonl.gz", tmp_path / "mark.jsonl.gz"
    with gzip.open(po, "wt") as f:
        f.write(json.dumps({"r": m0 + 2000, "m": [["XUSDT", 1000.0, 1, -1.5, None]]}) + "\n")
    with gzip.open(pr, "wt") as f:
        f.write(json.dumps({"r": m0 + 60_000 + 20_000, "m": [["XUSDT", 99, 1.1, 1.2, 1.3, 0.9, 5.0, 6.0]]}) + "\n")
    with gzip.open(pm, "wt") as f:
        for k in range(4):
            f.write(json.dumps({"r": m0 + k * 60_000, "E": m0 + k * 60_000, "m": [["XUSDT", "1", "1", "0", 0]]}) + "\n")
    rows = {r["minute"]: r for r in F.build([], mark=[str(pm)], oi=[str(po)], ratio=[str(pr)])}
    assert rows[m0]["oi"] == 1000.0 and rows[m0]["oi_chg_1m"] == -1.5 and rows[m0]["oi_chg_5m"] == ""
    assert "top_ls_acct" not in rows[m0]                             # before the first ratio read: blank
    assert rows[m0 + 60_000]["top_ls_acct"] == 1.1 and rows[m0 + 180_000]["taker_bs_ratio"] == 0.9   # carried forward
    assert rows[m0 + 180_000]["ratio_ts"] == 99 and "oi" not in rows[m0 + 180_000]


def test_health_pos_block():
    now = 10 ** 13
    pos = {"on": True, "oi_age_s": 30, "oi_coins": 120, "oi_ok": 120, "ratio_age_s": 100, "ratio_coins": 150, "ratio_skipped": 25,
           "ratio_ok": 120, "ratio_paused_s": 0, "ratio_n429": 0, "day": {}}
    ok = {"now": now - 60_000, "started": now - 3600_000, "age_s": {"mark": 1, "book": 2, "trade": 3}, "bot_n": 40,
          "disk_free_gb": 20, "pos": pos}
    assert H.verdict(ok, now)[0] == 0
    assert H.verdict(dict(ok, pos=dict(pos, oi_age_s=400)), now)[0] == 1
    assert H.verdict(dict(ok, pos=dict(pos, oi_ok=50)), now)[0] == 1
    assert H.verdict(dict(ok, pos=dict(pos, ratio_ok=90)), now)[0] == 1           # 90 / 125 tried
    assert H.verdict(dict(ok, pos=dict(pos, ratio_paused_s=100, ratio_n429=1)), now)[0] == 1
    assert H.verdict(dict(ok, started=now - 60_000, pos=dict(pos, oi_age_s=None)), now)[0] == 0   # warm-up
    assert "budget skipped 25" in H.pos_lines(pos)[1] and "not reported" in H.pos_lines(None)[0]


def test_deploy_ships_positioning():
    with open(os.path.join(ROOT, "scripts", "route_b", "deploy_aws.sh")) as f:
        s = f.read()
    assert 's3://$BUCKET/code/positioning.py' in s and '"$HERE/positioning.py"' in s
    # weight per minute at the caps: OI 150 + depth snapshots 300 + 24 h ticker 40 / 5 min (+ ratios ≤ 100 / min if Binance counts them)
    assert PS.OI_MAX_COINS + FB.FB_WEIGHT_PER_MIN + 8 + PS.RATIO_MAX_PER_SWEEP / 5 < 1200 / 2


# ── review fixes (2026-11-21): memory, starvation, late writes, backoffs, robustness, staleness ────────────────────────────────────

def _book_chain(bk, n, start=100, hold=False, now=0):
    for i in range(n):
        bk.on_event(ev(start + i, start + i, start + i - 1), now, hold=hold)


def test_buffer_is_small_and_holds_one_event_while_no_snapshot_can_come():
    assert FB.FB_MAX_BUFFER <= 300
    bk = FB.LocalBook("XUSDT")
    _book_chain(bk, 1000)
    assert len(bk.buf) == FB.FB_MAX_BUFFER and bk.overflow == 1000 - FB.FB_MAX_BUFFER
    bk2 = FB.LocalBook("XUSDT")
    _book_chain(bk2, 1000, hold=True)                              # a REST pause: nothing useful to keep
    assert len(bk2.buf) == 1 and bk2.needs_snapshot(10 ** 7)        # … but one event, so the request can start at once
    bk3 = FB.LocalBook("XUSDT")                                     # a snapshot already pending keeps buffering (it needs the bridge)
    bk3.on_event(ev(10, 20, 9))
    assert bk3.on_snapshot(SNAP, 0) == "waiting"
    bk3.on_event(ev(100, 103, 20, b=[(99, 5)]), 0, hold=True)
    assert bk3.synced and bk3.bids[99.0] == 5.0


def test_ratio_sweeps_never_starve_every_other_time():
    w, out = PS.Rolling(300), []
    for k in range(8):                                              # 150 coins, sweeps every 300 s, 0.1 s per request
        t = t0 = 20 + 300 * k
        take, _ = PS.ratio_coins(list(range(150)), w.count(t0))
        for _ in range(len(take) * 4):
            w.add(t)
            t += 0.1
        out.append(len(take))
    assert out == [120] * 8


def test_ratio_last_ms_only_when_coins_were_read_and_health_flags_empty_sweeps(tmp_path):
    fake = _FakeRest(ratio_status=500)
    rec, asyncio, gap = _pos_rec(tmp_path, fake)
    try:
        assert asyncio.run(rec.ratio_sweep(1000)) == 0
        assert rec.pos["ratio_last_ms"] is None and rec.pos["ratio_tried_ms"] == 1000 and rec.pos["ratio_unread"] == 2
        rec.w.close()
    finally:
        PS.REQ_GAP_S = gap
    now = 10 ** 13
    pos = {"on": True, "oi_age_s": 30, "oi_coins": 120, "oi_ok": 120, "ratio_age_s": 100, "ratio_tried_ms": now, "ratio_coins": 120,
           "ratio_skipped": 0, "ratio_ok": 120, "ratio_paused_s": 0, "ratio_n429": 0}
    ok = {"now": now - 60_000, "started": now - 3600_000, "age_s": {"mark": 1, "book": 2, "trade": 3}, "bot_n": 40, "disk_free_gb": 20}
    assert H.verdict(dict(ok, pos=pos), now)[0] == 0
    assert H.verdict(dict(ok, pos=dict(pos, ratio_ok=0)), now)[0] == 1                       # read 0 coins
    assert H.verdict(dict(ok, pos=dict(pos, ratio_coins=150, ratio_skipped=31, ratio_ok=119)), now)[0] == 1   # skipped > 20 %
    assert H.verdict(dict(ok, pos=dict(pos, ratio_coins=150, ratio_skipped=30, ratio_ok=120)), now)[0] == 0
    assert H.verdict(dict(ok, pos={"on": False, "module": False, "off": False}), now)[0] == 1   # positioning.py failed to load
    assert H.verdict(dict(ok, pos={"on": False, "module": True, "off": True}), now)[0] == 0    # POS_OFF=1 on purpose


def test_a_late_sweep_never_reopens_an_uploaded_hour(tmp_path):
    w = R.Writer(str(tmp_path), tag="T")
    old = R.now_ms() - 2 * 3_600_000                                # a sweep that STARTED two hours ago …
    w.write("oi", {"r": old, "m": [["A", 3]]})                      # … is filed under NOW (the recorder passes no ms)
    w.close()
    d, h = R.hour_key(R.now_ms())
    assert os.listdir(tmp_path / "oi") == [d] and os.listdir(tmp_path / "oi" / d) == [f"{h}-T.jsonl.gz"]


def test_refresh_gives_up_then_backs_off_and_hourly_counts_from_the_last_attempt():
    bk = synced_book()
    bk.synced_ms = 0
    assert bk.begin_refresh(1000)
    for i in range(3):
        bk.on_event(ev(104 + 2 * i, 105 + 2 * i, 103 + 2 * i), 1000)
        bk.on_snapshot({"lastUpdateId": 50, "bids": [], "asks": []}, 1000)
    assert not bk.refreshing and bk.refresh_next_ms == 1000 + FB.FB_REFRESH_MIN_GAP_S * 1000
    assert not bk.begin_refresh(2000) and not bk.refresh_due(2000, 0.0)                       # no retry storm
    assert not bk.refresh_due(1000 + FB.FB_REFRESH_MIN_GAP_S * 1000 - 1, 0.0)
    assert bk.refresh_due(1000 + FB.FB_REFRESH_MIN_GAP_S * 1000, 0.0)                         # the drift trigger, after the backoff
    assert not bk.refresh_due(1000 + FB.FB_REFRESH_S * 1000 - 1, None)
    assert bk.refresh_due(1000 + FB.FB_REFRESH_S * 1000, None)      # hourly from the last attempt, not the old sync
    bk.last_refresh_ms = 5000
    assert not bk.refresh_due(1000 + FB.FB_REFRESH_S * 1000, None)


def test_failed_snapshot_backs_off_per_coin_and_round_robin(tmp_path):
    bk = FB.LocalBook("XUSDT")
    bk.on_event(ev(1, 1, 0))
    bk.request_failed(10_000)
    assert not bk.needs_snapshot(14_000) and bk.needs_snapshot(15_000)
    bk.request_failed(15_000)
    assert not bk.needs_snapshot(24_000) and bk.needs_snapshot(25_000)     # 5, 10, 20 … ≤ 300 s
    for _ in range(10):
        bk.request_failed(0)
    assert bk.next_req_ms == 300_000
    assert bk.on_snapshot(SNAP, 0) in ("stale", "waiting", "synced") and bk.req_fails == 0
    rec = R.Recorder(str(tmp_path), "", 10)
    rec.set_full(["AUSDT", "BUSDT", "CUSDT"])
    for s in rec.full:
        rec.books[s].on_event(ev(1, 1, 0))
    rec.books["AUSDT"].last_req_ms = 50_000
    assert rec.next_snapshot(100_000) == "BUSDT"                    # A was asked most recently → B first
    rec.books["BUSDT"].mark_requested(100_000)
    assert rec.next_snapshot(100_000) == "CUSDT"
    rec.w.close()


def test_one_bad_event_unsyncs_that_coin_only(tmp_path):
    rec, bk = _rec_with_synced_book(tmp_path)
    rec.set_full(["XUSDT", "YUSDT"])
    other = rec.books["YUSDT"]
    other.on_event(ev(100, 103, 99), 0)
    other.on_snapshot(SNAP, 0)
    assert other.synced
    rec.on_fb_msg(json.dumps({"stream": "xusdt@depth@100ms", "data": ev(104, 105, 103, b=[("abc", "1")])}))
    rec.on_fb_msg("{not json")
    rec.on_fb_msg(json.dumps({"stream": "xusdt@depth@100ms", "data": {"s": "XUSDT", "U": 1, "u": 2, "pu": 0, "b": 5}}))
    assert not bk.synced and other.synced and rec.fb_tot["bad"] >= 2
    rec.w.close()


def test_a_silent_stream_is_unsynced_and_status_counts_only_fresh_books(tmp_path):
    rec, bk = _rec_with_synced_book(tmp_path)
    t = bk.last_ev_ms
    assert rec.fb_tick(t + 1000) == 1
    assert rec.full_status(t + 5000)["synced_n"] == 1 and rec.full_status(t + 20_000)["synced_n"] == 0   # no line in 10 s
    assert rec.fb_tick(t + (FB.FB_STALE_S + 1) * 1000) == 0 and not bk.synced and rec.fb_tot["silent"] == 1
    rec.w.close()


def test_an_empty_side_resnapshots_with_backoff(tmp_path):
    rec, bk = _rec_with_synced_book(tmp_path)
    bk.bids.clear()                                                 # price fell below the trusted edge
    t = bk.last_ev_ms
    assert rec.fb_tick(t) == 0 and bk.refreshing and rec.fb_tot["empty"] == 1
    assert rec.fb_tick(t + 1000) == 0 and rec.fb_tot["empty"] == 1  # one refresh in flight, not one per second
    rec.w.close()


def test_oi_400_on_one_coin_never_pauses_the_rest(tmp_path):
    class Fake(_FakeRest):
        def __call__(self, path):
            if "AUSDT" in path:
                self.calls.append(path)
                return 400, {}, None
            return super().__call__(path)
    fake = Fake()
    rec, asyncio, gap = _pos_rec(tmp_path, fake)
    try:
        assert asyncio.run(rec.oi_sweep(0)) == 1
        assert rec.fb_rest["paused_until"] == 0 and rec.pos["oi_err"] == 1 and rec.pos["oi_skipped"] == 0
        rec.w.close()
    finally:
        PS.REQ_GAP_S = gap


def test_429_pause_is_persisted_and_honoured_at_start(tmp_path):
    rec = R.Recorder(str(tmp_path), "", 10)
    rec._rest_result(429, {"Retry-After": "600"})
    assert rec.fb_rest["pause_why"] == 429 and os.path.exists(tmp_path / R.PAUSE_FILE)
    rec2 = R.Recorder(str(tmp_path), "", 10)
    assert rec2.fb_rest["paused_until"] == rec.fb_rest["paused_until"] and rec2.fb_rest["pause_why"] == 429
    import asyncio
    rec3 = R.Recorder(str(tmp_path / "z"), "", 10)
    rec3._rest = lambda path: (418, {}, None)
    try:
        asyncio.run(rec3._rest_json("/fapi/v1/ticker/24hr"))
        raise AssertionError("must raise")
    except RuntimeError:
        pass
    assert rec3.fb_rest["pause_why"] == 418 and rec3.fb_rest["paused_until"] > R.now_ms() + 500_000
    for r in (rec, rec2, rec3):
        r.w.close()


def test_bot_list_read_once_per_minute_for_both_managers(tmp_path):
    import asyncio
    rec = R.Recorder(str(tmp_path), "", 10)
    n = [0]

    async def fake():
        n[0] += 1
        return {"at": R.now_ms()}
    rec._bot_list = fake
    asyncio.run(rec._bot_doc())
    asyncio.run(rec._bot_doc())
    assert n[0] == 1
    rec.bot_doc_ms -= 60_000                                        # no boto3 → the aws CLI at most every 5 min
    asyncio.run(rec._bot_doc())
    assert n[0] == 1
    rec.bot_doc_ms -= 250_000
    asyncio.run(rec._bot_doc())
    assert n[0] == 2
    rec.s3 = object()                                               # boto3 in-process → once a minute
    rec.bot_doc_ms -= 60_000
    asyncio.run(rec._bot_doc())
    assert n[0] == 3
    rec.w.close()


def test_local_book_matches_the_exchange_under_drops_and_late_snapshots():
    import random
    for seed in (1, 2, 3):                                          # the deep review's fuzz: dropped events, duplicates, late snapshots
        random.seed(seed)
        tb, ta = {p: 1.0 for p in range(9000, 10000)}, {p: 1.0 for p in range(10001, 11000)}
        u, bk, pend, bad, checks = 100, FB.LocalBook("X"), [], 0, 0
        for t in range(4000):
            U, nu = u + 1, u + random.randint(1, 4)
            mid = (max(tb) + min(ta)) / 2
            b, a = [], []
            for _ in range(random.randint(1, 6)):
                p, q = int(mid + random.gauss(0, 30)), random.choice([0.0, 0.0, 1.0, 2.5])
                if random.random() < 0.5 and p < min(ta):
                    b.append((str(p), str(q)))
                    tb.pop(p, None) if q == 0 else tb.__setitem__(p, q)
                elif p > max(tb):
                    a.append((str(p), str(q)))
                    ta.pop(p, None) if q == 0 else ta.__setitem__(p, q)
            e = {"U": U, "u": nu, "pu": u, "b": b, "a": a, "E": nu}
            u = nu
            if random.random() < 0.002:
                continue                                            # a dropped event
            bk.on_event(e, t * 100)
            if bk.needs_snapshot(t * 100 + 10 ** 7) or (random.random() < 0.002 and bk.synced and not bk.refreshing):
                if bk.synced and not bk.refreshing:
                    bk.begin_refresh(t * 100)
                bk.mark_requested(t * 100)
                snap = {"lastUpdateId": u, "bids": [[str(p), str(q)] for p, q in sorted(tb.items(), reverse=True)[:300]],
                        "asks": [[str(p), str(q)] for p, q in sorted(ta.items())[:300]]}
                pend.append((t + random.randint(0, 30) - random.randint(0, 5), snap))
            for item in [x for x in pend if x[0] <= t]:
                pend.remove(item)
                bk.on_snapshot(item[1], t * 100, limit=300)
            if bk.synced:
                checks += 1
                if {p: q for p, q in tb.items() if p >= bk.edge_bid} != {int(p): q for p, q in bk.bids.items()} or \
                        {p: q for p, q in ta.items() if p <= bk.edge_ask} != {int(p): q for p, q in bk.asks.items()}:
                    bad += 1
        assert checks > 1000 and bad == 0, (seed, checks, bad)


def test_deploy_lifecycle_and_off_switch():
    with open(os.path.join(ROOT, "scripts", "route_b", "deploy_aws.sh")) as f:
        s = f.read()
    assert '"ID":"raw-data-to-deep-archive-30d"' in s and '"StorageClass":"DEEP_ARCHIVE"' in s and '"Days":30' in s
    assert "STANDARD_IA" not in s and "abort-mpu-7d" in s
    assert "POS_OFF=${POS_OFF:-0}" in s and "Environment=POS_OFF=$POS_OFF" in s and "FULL_MAX=${FULL_MAX:-40}" in s
    assert "rm -rf /var/lib/scalpars-recorder/{fbook,fsnap,oi,ratio}" in s


# ── review round 2 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_empty_priority_set_never_reconnects(monkeypatch, tmp_path):
    import asyncio
    import contextlib

    class WS:
        async def recv(self):
            await asyncio.sleep(3600)

        async def send(self, m):
            pass

    @contextlib.asynccontextmanager
    async def fake_connect(url, **kw):
        yield WS()
    monkeypatch.setattr(R, "ws_connect", fake_connect)
    monkeypatch.setattr(R, "READ_TIMEOUT_S", 0.1)
    monkeypatch.setattr(R, "backoff_s", lambda n: 0.01)
    rec = R.Recorder(str(tmp_path), "", 10)

    async def main():
        t = asyncio.create_task(rec.fullconn())
        await asyncio.sleep(0.3)                                    # empty set: not even connected
        assert "fbook" not in rec.hot_ws
        assert rec.reconnects["fbook"] == 0
        rec.set_full(["XUSDT"])                                     # a coin, but a silent socket → a real read timeout reconnects
        await asyncio.sleep(2.5)
        rec.set_full([])                                            # emptied: read timeouts are normal from now on
        await asyncio.sleep(0.3)
        n = rec.reconnects["fbook"]
        await asyncio.sleep(1.0)
        assert rec.reconnects["fbook"] == n and "fbook" in rec.hot_ws
        rec.stop.set()
        await t
    asyncio.run(main())
    rec.w.close()


def test_pause_values_are_finite_and_clamped(tmp_path):
    assert FB.rest_pause_s(418, {"Retry-After": "inf"}, 0) == 600 and FB.rest_pause_s(418, {"Retry-After": "nan"}, 0) == 600
    assert FB.rest_pause_s(418, {"Retry-After": "1e30"}, 0) == FB.MAX_PAUSE_S
    for content in ['{', '[]', '{"until":"abc"}', '{"until": null}', '{"until": 1e30, "why": 418}', '{"until": Infinity}', '\x00']:
        d = tmp_path / str(abs(hash(content)))
        d.mkdir()
        (d / R.PAUSE_FILE).write_text(content)
        r = R.Recorder(str(d), "", 10)
        assert r.fb_rest["paused_until"] == 0 and r.ratio_paused_until == 0, content
        r.w.close()
    rec = R.Recorder(str(tmp_path / "x"), "", 10)
    rec._rest_result(418, {"Retry-After": "1e300"})
    assert rec.fb_rest["paused_until"] <= R.now_ms() + R.MAX_PAUSE_MS
    rec.w.close()


def test_hot_list_built_during_a_pause_without_rest(tmp_path):
    import asyncio
    rec = R.Recorder(str(tmp_path), "", 10)
    rec.fb_rest["paused_until"] = R.now_ms() + 600_000
    calls = []
    rec._rest = lambda path: calls.append(path) or (200, {}, [])

    async def doc():
        return {"at": R.now_ms(), "open": {"OPENUSDT": {}}}
    rec._bot_doc = doc

    async def main():
        t = asyncio.create_task(rec.hot_manager())
        await asyncio.sleep(0.2)
        rec.stop.set()
        await t
    asyncio.run(main())
    assert calls == [] and rec.hot[:3] == ["BTCUSDT", "ETHUSDT", "OPENUSDT"]
    rec.w.close()


def test_fsnap_is_staggered_per_coin():
    offs = {FB.snap_offset_ms(f"C{i}USDT") // 1000 for i in range(30)}
    assert len(offs) >= 15                                          # 30 coins spread over ≥ 15 distinct seconds
    bk = FB.LocalBook("EVAAUSDT")
    t = 1_795_000_000_000
    assert bk.snap_due(t)                                           # the first one right after a sync
    bk.mark_snap(t)
    nxt = [x for x in range(t, t + 31_000, 1000) if bk.snap_due(x)]
    assert len(nxt) == 1 and (nxt[0] - bk.snap_off) % 30_000 < 1000


def test_writer_never_reopens_a_closed_hour(tmp_path):
    w = R.Writer(str(tmp_path), tag="T")
    H0 = 1_795_000_000_000 // 3_600_000 * 3_600_000
    w.write("oi", {"r": 1}, H0 + 1000)
    w.roll(*R.hour_key(H0 + 3_600_000))                             # hour closed (uploader may take it)
    w.write("oi", {"r": 2}, H0 + 2000)                              # a late line for it
    w.close()
    d0, h0 = R.hour_key(H0)
    import glob
    assert len(gzip.open(tmp_path / "oi" / d0 / f"{h0}-T.jsonl.gz", "rt").read().splitlines()) == 1
    assert len(glob.glob(str(tmp_path / "oi" / "*" / "*.gz"))) == 2   # the late line went to the current hour's file


def test_rest_keeps_one_connection_alive(monkeypatch, tmp_path):
    made = []

    class Resp:
        status, will_close = 200, False

        def read(self):
            return b'{"openInterest": "1", "time": 1}'

        def getheaders(self):
            return [("X-MBX-USED-WEIGHT-1M", "3")]

    class Conn:
        def __init__(self, host, timeout=None):
            made.append(host)
            self.n = 0

        def request(self, *a, **k):
            self.n += 1
            if len(made) == 1 and self.n == 3:
                raise ConnectionResetError("stale keep-alive")      # → one fresh retry

        def getresponse(self):
            return Resp()

        def close(self):
            pass
    monkeypatch.setattr(R.http.client, "HTTPSConnection", Conn)
    monkeypatch.setattr(R._tls, "conn", None, raising=False)
    rec = R.Recorder(str(tmp_path), "", 10)
    out = [rec._rest("/fapi/v1/openInterest?symbol=A") for _ in range(4)]
    assert all(o[0] == 200 and o[2]["openInterest"] == "1" for o in out) and len(made) == 2
    R._tls.conn = None
    rec.w.close()


def test_gather_exception_counts_that_coin_and_the_rest_go_on(tmp_path):
    fake = _FakeRest()

    def boom(path):
        if "AUSDT" in path:
            raise RuntimeError("worker died")
        return fake(path)
    rec, asyncio, gap = _pos_rec(tmp_path, boom)
    try:
        assert asyncio.run(rec.oi_sweep(0)) == 1 and rec.pos["oi_err"] == 1
        assert asyncio.run(rec.ratio_sweep(0)) == 1 and rec.pos["ratio_err"] == 1
        rec.w.close()
    finally:
        PS.REQ_GAP_S = gap


def test_ratio_reads_ignore_a_fapi_weight_pause_but_persist_their_own(tmp_path):
    fake = _FakeRest()
    rec, asyncio, gap = _pos_rec(tmp_path, fake)
    try:
        rec.fb_rest.update(paused_until=R.now_ms() + 60_000, pause_why="weight")
        assert asyncio.run(rec.ratio_sweep(0)) == 2                 # a separate limit: not blocked
        rec.fb_rest.update(pause_why=418)
        assert asyncio.run(rec.ratio_sweep(0)) == 0                 # an IP ban blocks everything
        rec.fb_rest.update(paused_until=0, pause_why=None)
        rec._rest = _FakeRest(ratio_status=429)
        asyncio.run(rec.ratio_sweep(0))
        rec2 = R.Recorder(str(tmp_path), "", 10)
        assert rec2.ratio_paused_until == rec.ratio_paused_until > R.now_ms()
        rec.w.close()
        rec2.w.close()
    finally:
        PS.REQ_GAP_S = gap


def test_health_rest_pause_reported_with_full_book_off_and_slow_oi():
    now = 10 ** 13
    ok = {"now": now - 60_000, "started": now - 3600_000, "age_s": {"mark": 1, "book": 2, "trade": 3}, "bot_n": 40, "disk_free_gb": 20,
          "full": {"on": False, "requested": 0, "module": True, "rest": {"paused_s": 100, "pause_why": 418}}}
    c, _, why = H.verdict(ok, now)
    assert c == 1 and "418" in why[0]
    pos = {"on": True, "oi_age_s": 30, "oi_coins": 120, "oi_ok": 120, "oi_sweep_s": 50, "ratio_age_s": 100, "ratio_tried_ms": now,
           "ratio_coins": 120, "ratio_skipped": 0, "ratio_ok": 120, "ratio_paused_s": 0}
    ok2 = dict(ok, full=None, pos=pos)
    assert H.verdict(ok2, now)[0] == 1 and H.verdict(dict(ok2, pos=dict(pos, oi_sweep_s=20)), now)[0] == 0


def test_listing_split_by_kind_and_age():
    now = 1_800_000_000
    old = time_str(now - 40 * 86400)
    new = time_str(now - 86400)
    out = (f"{old} 1000 data/book/2026-01-01/00-1.jsonl.gz\n{new} 200 data/fbook/2026-02-01/00-1.jsonl.gz\n"
           f"{new} 50 data/oi/2026-02-01/00-1.jsonl.gz\ngarbage line\n")
    n, size, kinds, older = H.parse_listing(out, now)
    assert (n, size, older) == (3, 1250, 1000) and kinds["fbook"] == (1, 200) and kinds["oi"] == (1, 50)


def time_str(t):
    import time as _t
    return _t.strftime("%Y-%m-%d %H:%M:%S", _t.gmtime(t))


def test_ratio_columns_carried_across_hours():
    import pandas as pd
    import first_view as FV
    m = 1_795_000_000_000 // 3_600_000 * 3_600_000
    a = pd.DataFrame([{"sym": "X", "minute": m - 60_000, "top_ls_acct": 1.5, "ratio_ts": m - 300_000, "taker_bs_ratio": 0.9},
                      {"sym": "X", "minute": m, "top_ls_acct": None, "ratio_ts": None, "taker_bs_ratio": None},
                      {"sym": "X", "minute": m + 20 * 60_000, "top_ls_acct": None, "ratio_ts": None, "taker_bs_ratio": None},
                      {"sym": "Y", "minute": m, "top_ls_acct": None, "ratio_ts": None, "taker_bs_ratio": None}])
    f = FV.ffill_ratios(a).set_index(["sym", "minute"])
    assert f.loc[("X", m), "top_ls_acct"] == 1.5 and f.loc[("X", m), "taker_bs_ratio"] == 0.9
    assert pd.isna(f.loc[("X", m + 20 * 60_000), "top_ls_acct"]) and pd.isna(f.loc[("Y", m), "top_ls_acct"])   # too old / other coin


def test_pending_timeout_checked_even_with_an_empty_buffer():
    bk = FB.LocalBook("X")
    bk.on_event(ev(10, 20, 9))
    assert bk.on_snapshot(SNAP, 0) == "waiting" and not bk.buf
    assert not bk.needs_snapshot(FB.FB_PENDING_TIMEOUT_S * 1000 + 1)   # no event yet → nothing to bridge …
    assert bk.pending is None and bk.want                               # … but the stale pending snapshot is dropped


def test_bad_env_falls_back(monkeypatch):
    monkeypatch.setenv("FULL_MAX", "thirty")
    R.BAD_ENV.clear()
    assert R.env_int("FULL_MAX", None) is None and R.BAD_ENV and R.BAD_ENV[0]["name"] == "FULL_MAX"
    monkeypatch.setenv("FULL_MAX", "12")
    assert R.env_int("FULL_MAX", None) == 12
    monkeypatch.delenv("FULL_MAX")
    assert R.env_int("FULL_MAX", 30) == 30
    R.BAD_ENV.clear()


def test_featurer_closes_the_past_hour_before_listing(tmp_path):
    import asyncio
    rec = R.Recorder(str(tmp_path), "", 10)
    H0 = (R.now_ms() // 3_600_000 - 1) * 3_600_000
    rec.w.write("oi", {"r": H0}, H0 + 1000)                         # a past hour still held open by the writer
    seen = []

    async def fake_features(key, kinds):
        seen.append(kinds)
        rec.stop.set()
        return True
    rec._features = fake_features
    asyncio.run(rec.featurer())
    assert seen and "oi" in seen[0]
    rec.w.close()
