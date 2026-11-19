"""🧭 the pairs panel's BLOCK REASON column redesign (operator 11-18): services/sleeve_status.build (pure) per state + the UI / export wiring."""
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import sleeve_status as S  # noqa: E402

NOW = 1_763_500_000_000
TH = NS(exhaustion_short_enabled=True, exhaustion_wide_enabled=True, distribution_short_enabled=True, flatbase_short_enabled=True,
        frenzy_willy_enabled=True, frenzy_willy_ht_enabled=True, frenzy_long_enabled=False, distribution_rsi1h_min=74.07)


def _states(lines):
    return [(l["sleeve"], l["state"], l["rank"]) for l in lines]


def test_tracking_and_waiting_lines():
    L = S.build("AUSDT", NOW, TH, exh_eps=[(NOW - 3_600_000, "FRENZY")], exh_rsi=(NOW, 78.4))
    assert _states(L) == [("EXH", "tracking", 3), ("DIST", "waiting", 3)]
    assert L[0]["reason"] == "tracking · RSI15 78 · waits cross ↓70" and L[1]["reason"] == "waits red ≥5× candle"


def test_wide_episode_has_no_dist_line_and_wide_off_hides_it():
    L = S.build("BUSDT", NOW, TH, exh_eps=[(NOW - 3_600_000, "WIDE")])
    assert _states(L) == [("WIDE", "tracking", 3)]
    off = NS(**dict(vars(TH), exhaustion_wide_enabled=False))
    assert S.build("BUSDT", NOW, off, exh_eps=[(NOW - 3_600_000, "WIDE")]) == []


def test_blocks_signals_and_order():
    sp = NOW - 7_200_000
    L = S.build("CUSDT", NOW, TH, exh_eps=[(sp, "FRENZY")],
                exh_last=dict(at_ms=NOW - 600_000, status="BLOCKED", block="EXH_NO_SPOT", reason="no Binance SPOT", spike_ms=sp, universe="FRENZY"),
                dist_last=dict(at_ms=NOW - 3 * 3_600_000, status="BLOCKED", block="DIST_RSI1H", reason="1h RSI 68.2 < 74.07", rsi1h=68.2, spike_ms=sp))
    assert _states(L) == [("EXH", "blocked", 1), ("DIST", "blocked", 2)]                       # the last hour first, then the older block
    assert L[0]["reason"] == "no spot" and L[1]["reason"] == "RSI1h 68 < 74.07"
    L2 = S.build("CUSDT", NOW, TH, exh_eps=[(sp, "FRENZY")], dist_last=dict(at_ms=NOW - 60_000, status="OPENED", block=None, reason="opened #9", spike_ms=sp))
    assert _states(L2)[0] == ("DIST", "signal", 1)


def test_open_trade_first_and_its_sleeve_not_repeated():
    L = S.build("DUSDT", NOW, TH, open_es="EXHAUSTION_SHORT", exh_eps=[(NOW - 3_600_000, "FRENZY")])
    assert _states(L) == [("EXH", "open", 0), ("DIST", "waiting", 3)]


def test_flatbase_and_ht_and_long():
    L = S.build("EUSDT", NOW, TH, fb_rec=dict(status="TRACKING"), willy_pend=dict(trig="A", armed=NOW - 300_000, exp=NOW + 25 * 60_000))
    assert ("FB", "tracking", 3) in _states(L) and ("HT", "armed", 3) in _states(L)
    assert [l["reason"] for l in L if l["sleeve"] == "HT"] == ["armed A, waits red ≤ 25 min"]
    L = S.build("EUSDT", NOW, TH, fb_rec=dict(status="TRACKING", confirm=dict(px=1.0)))
    assert L[0]["reason"].startswith("confirm wait")
    L = S.build("EUSDT", NOW, TH, fb_last=dict(at_ms=NOW - 1_800_000, status="SKIPPED", reason="NEAR_BASE — confirm close +1.2 %"))
    assert _states(L) == [("FB", "blocked", 1)] and "near_base" in L[0]["reason"].lower()
    flag = dict(text="below its average price", willy_ht_last="11-14 10:05 WILLY A red bar refused: turnover 0.8× mcap")
    assert _states(S.build("EUSDT", NOW, TH, flag=flag)) == [("HT", "blocked", 2)]                   # FRENZY_LONG is off → no LONG line
    on = NS(**dict(vars(TH), frenzy_long_enabled=True))
    assert ("LONG", "waiting", 3) in _states(S.build("EUSDT", NOW, on, flag=flag))


def test_switched_off_sleeves_never_show_and_bad_input_is_safe():
    off = NS(**{k: False for k in vars(TH)})
    assert S.build("FUSDT", NOW, off, exh_eps=[(NOW, "FRENZY")], fb_rec=dict(status="TRACKING"), willy_pend=dict(trig="A")) == []
    assert S.build("FUSDT", "x", TH, exh_eps=[("bad", None)]) == []


def test_ui_and_both_exports():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "function sleeveStatusShort(l)" in html and "function sleeveStatusFull(l)" in html
    assert 'id="sleeve-status-${_escAttr(p.pair)}"' in html and "p.sleeve_status.map(sleeveStatusFull)" in html
    assert 'data-momentum-gate="1">momentum: ${_brSafe}</div>' in html and "WILLY: 'WILLY'" in html                    # (review 7) the gate stays visible
    assert html.count("## 🧭 Sleeve status per pair") == 1 and html.count("...frenzyReportLines(perf, hr2)") == 2   # one shared block, both exports
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '"sleeve_status": _ss_map.get(p.pair) or []' in main and '"sleeve_status_rows":' in main and "async def _sleeve_status_map(db=None, pairs=None)" in main
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "_exh_rsi_last" in eng


def test_status_map_own_session_open_priority_and_bot_row_first(monkeypatch):
    import asyncio
    import datetime as dt
    import models
    import main as M
    import services.trading_engine as TE
    from tests.test_frenzy_willy import _hold_db
    for k, v in vars(TH).items():
        monkeypatch.setattr(M.config.trading_config.thresholds, k, v, raising=False)

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        monkeypatch.setattr(M, "AsyncSessionLocal", SL)                                          # its OWN session (never the caller's)
        async with SL() as s_:
            for es in ("MANUAL", "EXHAUSTION_SHORT"):
                s_.add(models.Order(pair="OPNUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy=es, confidence="STRONG_BUY",
                                    entry_price=1.0, quantity=1.0, investment=1.0, leverage=5, notional_value=5.0, opened_at=dt.datetime(2026, 11, 18)))
            await s_.commit()
        monkeypatch.setattr(M.trading_engine, "is_paper_mode", True, raising=False)
        r = await M._sleeve_status_map(None, [f"P{i:03d}USDT" for i in range(400)])               # 400 shortlist pairs: the open pair is still read
        await eng.dispose()
        return r
    r = asyncio.run(go())
    assert r["OPNUSDT"][0]["state"] == "open" and r["OPNUSDT"][0]["sleeve"] == "EXH"             # the bot row wins over MANUAL


def test_status_map_failure_is_isolated_and_logged(monkeypatch, caplog):
    import asyncio
    import main as M

    class Boom:
        async def __aenter__(self):
            raise RuntimeError("db down")

        async def __aexit__(self, *a):
            return False
    monkeypatch.setattr(M, "AsyncSessionLocal", lambda: Boom())
    monkeypatch.setattr(M, "_SLEEVE_STATUS_WARN_AT", [0.0])
    import logging
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(M._sleeve_status_map(None, ["AUSDT"])) == {}
    assert any("SLEEVE_STATUS" in r.message for r in caplog.records)


# ── review fixes (11-18) ──────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_fb_opened_and_shadow_missed_expired():
    L = S.build("GUSDT", NOW, TH, fb_last=dict(at_ms=NOW - 600_000, status="OPENED", reason="opened #3"))
    assert _states(L) == [("FB", "signal", 1)] and L[0]["reason"] == "opened"
    for st in ("SHADOW", "MISSED", "EXPIRED"):
        L = S.build("GUSDT", NOW, TH, fb_last=dict(at_ms=NOW - 5 * 3_600_000, status=st, reason="x"))
        assert _states(L) == [("FB", "blocked", 2)] and L[0]["reason"] == S.FB_STATE_TEXT[st]


def test_journal_rows_of_an_older_episode_or_other_universe_are_ignored():
    sp = NOW - 3_600_000
    old = dict(at_ms=NOW - 600_000, status="BLOCKED", block="EXH_NO_SPOT", reason="x", spike_ms=sp - 80 * 3_600_000, universe="FRENZY")
    assert _states(S.build("HUSDT", NOW, TH, exh_eps=[(sp, "FRENZY")], exh_last=old))[0] == ("EXH", "tracking", 3)
    wide = dict(old, spike_ms=sp, universe="WIDE")
    assert _states(S.build("HUSDT", NOW, TH, exh_eps=[(sp, "FRENZY")], exh_last=wide))[0] == ("EXH", "tracking", 3)
    dold = dict(at_ms=NOW - 600_000, status="BLOCKED", block="DIST_POS_CHG", reason="x", spike_ms=sp - 1)
    assert ("DIST", "waiting", 3) in _states(S.build("HUSDT", NOW, TH, exh_eps=[(sp, "FRENZY")], dist_last=dold))


def test_stale_rsi_is_dropped():
    sp = NOW - 3_600_000
    L = S.build("IUSDT", NOW, TH, exh_eps=[(sp, "FRENZY")], exh_rsi=(NOW - 31 * 60_000, 81.0))
    assert L[0]["reason"] == "tracking · waits RSI15 cross ↓70"


def test_ht_newer_text_prefix_and_plain_off_tail():
    import datetime as dt
    now = int(dt.datetime(2026, 11, 18, 12, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    flag = dict(willy_ht_last="11-18 09:00 WILLY A red bar refused: turnover 0.8× mcap",
                willy_last="11-18 11:30 WILLY A red bar passed turnover → refused: plain WILLY entries OFF (HT only)")
    L = S.build("JUSDT", now, TH, flag=flag)
    # (587 re-review 11) the newer text is a plain-WILLY-off refusal — a WILLY outcome, never an HT block → the HT text is shown
    assert L[0]["sleeve"] == "HT" and L[0]["reason"] == "turnover 0.8× mcap" and L[0]["rank"] == 2
    assert L[0]["at_ms"] == int(dt.datetime(2026, 11, 18, 9, 0, tzinfo=dt.timezone.utc).timestamp() * 1000)
    assert not L[0]["detail"].startswith("11-18") and "plain WILLY" not in L[0]["detail"]          # the UTC prefix is converted, not shown
    assert S.build("JUSDT", now, TH, flag=dict(willy_last=flag["willy_last"])) == []                # only a plain-off text → no HT line
    assert S.utc_prefix_ms("01-02 03:04 x", int(dt.datetime(2026, 12, 31, tzinfo=dt.timezone.utc).timestamp() * 1000))[1] == "x"


def test_open_plain_willy_is_labelled_willy():
    L = S.build("KUSDT", NOW, TH, open_es="FRENZY_WILLY", flag=dict(willy_last="11-18 10:00 x refused: y"))
    assert _states(L) == [("WILLY", "open", 0)]


def test_build_keeps_lines_on_an_error(monkeypatch):
    calls = []
    monkeypatch.setattr(S, "utc_prefix_ms", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    L = S.build("KUSDT", NOW, TH, open_es="EXHAUSTION_SHORT", flag=dict(willy_last="11-18 10:00 x refused: y"))
    assert [d["sleeve"] for d in L] == ["EXH"]                                                     # the open line survives the failure


def test_dist_not_vsw_shows_the_multiple():
    """(operator 11-18) a DIST near-miss shows how close the red candle came."""
    from services import sleeve_status as SS
    last = dict(at_ms=1_000, status="BLOCKED", block="DIST_NOT_VSW",
                reason="exchange quote volume 4.72× the 20-candle mean (need ≥ 5×, red) — the pre-screen's proxy passed")
    ln = SS._decision_line("DIST", last, 2_000, SS.DIST_REASON)
    assert ln["reason"] == "red candle 4.72× (needs 5×)"
    ln2 = SS._decision_line("DIST", dict(last, reason="garbled"), 2_000, SS.DIST_REASON)
    assert ln2["reason"] == "volume < 5×"
    ln3 = SS._decision_line("DIST", dict(last, reason="exchange quote volume 7.10× the 20-candle mean (need ≥ 5×, red)"), 2_000, SS.DIST_REASON)
    assert ln3["reason"] == "candle not red (7.10×)"
    ln4 = SS._decision_line("DIST", dict(last, reason="exchange quote volume 5.00× the 20-candle mean (need ≥ 5×, red)"), 2_000, SS.DIST_REASON)
    assert ln4["reason"] == "red candle 5.00× (needs 5×)"   # rounded at 2 dp: a red candle just under the cut, not "not red"
