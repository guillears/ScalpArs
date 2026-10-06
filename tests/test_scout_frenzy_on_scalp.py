"""⚡ ON_SCALP scout line (observe-only, registered 2026-10-06; reports/FRENZY_ON_SCALP_STUDY_2026-10-06.md) — the frozen rule and bar, the
60-s "new information" features, and full ons_run passes on stubbed data (no network; state files in tmp_path)."""
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_frenzy_exits as X  # noqa: E402

DAY0 = int(pd.Timestamp("2026-10-07", tz="UTC").value // 1_000_000)
LATER = DAY0 + 30 * 86_400_000
STEP = {"1m": X.MIN, "5m": X.BAR, "1h": X.H}
TH = SimpleNamespace(frenzy_max_hours=96.0)


def test_frozen_constants():
    assert (X.ONS_TP, X.ONS_T_MIN, round(X.ONS_COST, 10)) == (3.0, 120, 0.19)
    assert (X.ONS_N, X.ONS_DAYS, X.ONS_RETIRE_N, X.ONS_DD_MAX) == (30, 15, 60, 50.0)
    assert (X.ONS_BOOK0, X.ONS_NOTIONAL, X.ONS_LIQ) == (3000.0, 0.94, 23.75)          # PREREG "Book" factors at lev 0.2
    assert X.ONS_FLOW_MS == 60_000 and X.ENTRY_LAG_MS == 12_000 and X.GC_FROM == "2026-10-07T00:00:00"
    assert "502 episodes on 229 days" in X.ONS_YEAR and "+0.105" in X.ONS_YEAR and "[−0.33, +0.51]" in X.ONS_YEAR and "67.5 %" in X.ONS_YEAR
    assert (X.ONS_P3, X.ONS_HAIRCUT, X.ONS_FLOW_PAGES, X.ONS_FLOW_GIVEUP_D, X.ONS_PRE_MS) == (70.0, (0.30, 0.50), 200, 30, 12_000)
    assert "pair-EPISODE" in X.__doc__ and "893 bars" in X.__doc__
    assert "forceOrder" in X.ONS_LIQ_NOTE and "forceOrder" in X.__doc__ and "not built" in X.ONS_LIQ_NOTE


def test_no_stop_rides_through_the_flush():
    """GRIFFAIN 10-06 shape: −4 % at 6 s, then the pop — a no-stop TP +3 wins it; the −3 lock would have stopped (study anecdote)."""
    t0 = DAY0 + X.H
    tt = np.array([t0 + 12_100, t0 + 18_000, t0 + 37_000, t0 + 7 * X.MIN]); pp = np.array([100.0, 96.0, 102.0, 104.5])
    w = X.onscalp_walk(tt, pp, 100.0, t0 + 12_100)
    assert w["how"] == "TP +3" and abs(w["pnl"] - (4.5 - 0.19)) < 1e-9 and abs(w["mae"] - (-4.19)) < 1e-9 and w["hit"]
    assert X.walk_ticks(tt, pp, 100.0, t0 + 12_100)[2] == "stop"                     # the live lock on the same prints


def test_flow_window_is_close_to_close_plus_60s():
    t0 = DAY0
    T = [t0 - 1, t0, t0 + 59_999, t0 + 60_000]
    f = X.onscalp_flow(T, [50.0, 100.0, 100.0, 200.0], [1, 1, 1, 1], [False, True, False, False], [1, 1, 1, 1], t0, 100.0, None, None)
    assert f["n_agg_60"] == 2 and f["usd_60"] == 200.0 and f["buy_share_60"] == 0.5 and f["vol_x_24h"] is None and f["vol_x_norm"] is None
    assert f["px12_vs_close"] == 0.0                                                   # the 59.999 s print is the first ≥ +12 s
    empty = X.onscalp_flow([], [], [], [], [], t0, 100.0, 10.0, 600.0)
    assert empty["usd_60"] == 0.0 and empty["buy_share_60"] is None and empty["mdd_60"] is None


def _stubs(monkeypatch, scen, ticks, flows=None):
    """scen: sig → dict(fresh, in_state, on, spike, adx, di, code); bars carry their sig through the last open."""
    def kl(sym, tf, start, end):
        st = STEP[tf]
        n = min(int((end - start) // st) + 1, 1500)
        return [[start + i * st, 100.0, 100.5, 99.5, 100.0, 1000.0] for i in range(n)]
    monkeypatch.setattr(X, "_kl", kl)
    monkeypatch.setattr(X, "normal_hour_usd", lambda h1, ms: 6e6)
    sg = lambda bars: bars[-1][0] + X.BAR
    monkeypatch.setattr(X, "frenzy_walk", lambda bars, nh, th: None if sg(bars) not in scen else dict(
        spike_ts=scen[sg(bars)]["spike"], hours=5.0, above_streak=14, vs_vwap_pct=2.0, bar_ret_pct=0.3, vol_mult=150.0, verified=True,
        in_state=scen[sg(bars)].get("in_state", True), fresh_on=scen[sg(bars)].get("fresh", True), on_bar_ts=scen[sg(bars)].get("on"),
        last_bar_ts=bars[-1][0], _sig=sg(bars)))
    monkeypatch.setattr(X, "frenzy_flagged", lambda ep, th: True)
    monkeypatch.setattr(X, "frenzy_long_status", lambda ep, atr, v, th: (False, scen[ep["_sig"]].get("code", "FRENZY_ATR_HIGH"), ""))
    seen = []

    def adx(b):
        seen.append((len(b), b[-1][0]))
        return scen[b[-1][0] + X.BAR]["adx"]
    monkeypatch.setattr(X, "frenzy_adx_delta", adx)
    monkeypatch.setattr(X, "frenzy_di_spread", lambda b: scen[b[-1][0] + X.BAR]["di"])
    def tks(pair, t0, t1, now, b):
        tt, pp = ticks[pair](t0)
        m = tt <= t1
        return "ok", tt[m], pp[m]
    monkeypatch.setattr(X, "_ticks", tks)
    fl = flows or {}
    tr = lambda sym, a: ("ok", *fl.get(sym, (np.array([a + 1_000, a + 20_000]), np.array([100.0, 99.0]), np.array([2.0, 1.0]),
                                              np.array([False, True]), np.array([1.0, 3.0]))), False)
    monkeypatch.setattr(X, "_aggtrades_rest", lambda sym, a, b, d: tr(sym, a))
    monkeypatch.setattr(X, "_aggtrades_archive", lambda sym, a, b, now, bud: tr(sym, a))
    monkeypatch.setattr(X, "_ons_book_scan", lambda need, now: {})
    return seen


def _paths(up=True):
    if up:
        return lambda t0: (np.array([t0 + 13_000, t0 + 60_000, t0 + 5 * X.MIN, t0 + 3 * X.H]), np.array([100.0, 98.0, 103.5, 90.0]))
    return lambda t0: (np.array([t0 + 13_000, t0 + 60 * X.MIN, t0 + 121 * X.MIN, t0 + 4 * X.H]), np.array([100.0, 92.0, 97.0, 120.0]))


def _J(rows):
    return pd.DataFrame([(X._iso(t), e, p, g, s) for t, e, p, g, s in rows], columns=["t", "e", "pair", "gate", "strategy"])


def _F(rows):
    cols = ["opened_at", "k", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "closed_at"]
    return pd.DataFrame([(X._iso(o), X._iso(o), p, "LONG", s, "CLOSED", 100.0, a, X._iso(o + X.H)) for o, p, s, a in rows], columns=cols)


def test_ons_run_cohort_sources_control_catchup_and_parity(tmp_path, monkeypatch):
    """A: journal ATR_HIGH refusal, strong → cohort row. B: a READY fill (no refusal line), strong, same episode as a later strong ON bar →
    one counted. C: non-strong ON → the control file. D: a catch-up OPEN line 3 bars after its ON bar → priced at the ON bar. E: a live
    (non-catch-up) journal line the replay says is not in state → kept as a live-only ON bar (‡, parity False). G: a catch-up line the replay
    cannot place → a parity note, never priced. P: a pre-floor strong ON → reference."""
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    a, b, b2, c, d_on, e, pre = (DAY0 + X.H, DAY0 + 2 * X.H, DAY0 + 2 * X.H + 20 * X.MIN, DAY0 + 3 * X.H, DAY0 + 4 * X.H, DAY0 + 5 * X.H,
                                 DAY0 - 3 * X.H)
    d_cu = d_on + 3 * X.BAR
    scen = {a: dict(spike=a - 3 * X.H, adx=2.0, di=10.0), b: dict(spike=b - 3 * X.H, adx=1.0, di=5.0, code="FRENZY_READY"),
            b2: dict(spike=b - 3 * X.H + 10 * X.MIN, adx=3.0, di=8.0), c: dict(spike=c - 3 * X.H, adx=-1.0, di=4.0),
            d_on: dict(spike=d_on - 3 * X.H, adx=0.5, di=1.0), d_cu: dict(spike=d_on - 3 * X.H, fresh=False, on=d_on - X.BAR, adx=0.5, di=1.0),
            pre: dict(spike=pre - 3 * X.H, adx=1.0, di=1.0)}
    g = DAY0 + 6 * X.H
    del_e = {e: dict(spike=e - 3 * X.H, fresh=False, in_state=False, adx=1.0, di=1.0), g: dict(spike=g - 3 * X.H, fresh=False, in_state=False, adx=1.0, di=1.0)}
    scen.update(del_e)
    seen = _stubs(monkeypatch, scen, {p: _paths(p != "CUSDT") for p in ("AUSDT", "BUSDT", "CUSDT", "DUSDT", "EUSDT", "PUSDT")})
    J = _J([(a, "BLOCK", "AUSDT", "FRENZY_ATR_HIGH", ""), (a, "BLOCK", "AUSDT", "FRENZY_WIDE_ATR_HIGH", ""), (b2, "BLOCK", "BUSDT", "FRENZY_GREEN_BAR", ""),
            (c, "BLOCK", "CUSDT", "FRENZY_GREEN_BAR", ""), (d_cu + 9_000, "OPEN", "DUSDT", "", "FRENZY_LONG"), (e, "BLOCK", "EUSDT", "FRENZY_GVOL_HIGH", ""),
            (pre, "BLOCK", "PUSDT", "FRENZY_ATR_HIGH", ""), (g, "BLOCK", "GUSDT", "FRENZY_CATCHUP_STALE", "")])
    F = _F([(b + 8_000, "BUSDT", "FRENZY_LONG", -3.0)])
    txt = "\n".join(X.ons_run(LATER, TH, J, F))
    s = pd.read_csv(X.ONS_CSV).set_index(["k", "pair"]); ctl = pd.read_csv(X.ONS_CTRL_CSV).set_index(["k", "pair"])
    ra = s.loc[(X._iso(a), "AUSDT")]
    assert ra.strong and ra.counted and ra.final and ra.exit_how == "TP +3" and abs(ra.pnl - (3.5 - 0.19)) < 1e-9 and abs(ra.mae - (-2.19)) < 1e-9
    assert ra.replay_code == "FRENZY_ATR_HIGH" and "FRENZY_ATR_HIGH" in ra.live_gates and ra.src == "journal" and ra.px_src == "tick" and ra.univ == "ok"
    assert ra.flow_state == "ok" and abs(ra.buy_share_60 - 200 / 299) < 1e-9 and ra.n_trades_60 == 4 and abs(ra.px12_vs_close - (-1.0)) < 1e-9
    assert abs(ra.vol_x_24h - 299 / 100_000) < 1e-9 and abs(ra.vol_x_norm - 299 / 100_000) < 1e-9 and not ra.flow_trunc
    assert ra.n_agg_pre == 1 and ra.buy_share_pre == 1.0 and abs(ra.move_pre) < 1e-12                  # [close, +12 s): one buy at 100
    assert "not available" in ra.liq_flow
    rb = s.loc[(X._iso(b), "BUSDT")]
    assert rb.counted and rb.src == "fill" and "LONG" in rb.live_fill and not s.loc[(X._iso(b2), "BUSDT")].counted   # same episode (10-min drift)
    rc = ctl.loc[(X._iso(c), "CUSDT")]
    assert not rc.strong and rc.exit_how == "2 h" and abs(rc.pnl - (-3.19)) < 1e-9     # no stop: held through −8 to the 2 h print
    rd = s.loc[(X._iso(d_on), "DUSDT")]
    assert rd.src == "journal→catch-up" and X._iso(d_cu) in rd.cand_keys and rd.counted and rd.parity and ra.parity
    re_ = s.loc[(X._iso(e), "EUSDT")]
    assert re_.is_on and not re_.parity and re_.replay_code == "LIVE_ONLY" and "replay: not in state" in re_.parity_why and re_.counted
    rg = ctl.loc[(X._iso(g), "GUSDT")]
    assert not rg.is_on and rg.not_on_reason == "not in state" and pd.isna(rg.get("pnl"))
    assert not s.loc[(X._iso(pre), "PUSDT")].counted and not s.loc[(X._iso(pre), "PUSDT")].cohort
    assert sorted(s.index[s.counted.astype(bool)].get_level_values(1)) == ["AUSDT", "BUSDT", "DUSDT", "EUSDT"]
    assert all(n == 300 and last + X.BAR in scen for n, last in seen)                # strong read on closed[-300:] ending AT the ON bar
    assert "4/30 fires · 1/15 days" in txt and "‡ live-only 1 of 4" in txt
    # a second run prices nothing final and replays no resolved candidate
    monkeypatch.setattr(X, "_ons_shadow", lambda *a_: pytest.fail("a final row was re-priced"))
    monkeypatch.setattr(X, "_ons_replay", lambda *a_: pytest.fail("a resolved candidate was replayed"))
    X.ons_run(LATER, TH, J, F)
    assert len(pd.read_csv(X.ONS_CSV)) == len(s)


def test_provisional_on_1m_then_final_on_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    a = DAY0 + X.H
    _stubs(monkeypatch, {a: dict(spike=a - 3 * X.H, adx=2.0, di=10.0)}, {"AUSDT": _paths(True)})
    J = _J([(a, "BLOCK", "AUSDT", "FRENZY_ATR_HIGH", "")])
    txt = "\n".join(X.ons_run(a + 30 * X.MIN, TH, J, None))                             # 30 min after: 1m flat bars, still open
    r = pd.read_csv(X.ONS_CSV).iloc[0]
    assert not r.final and r.px_src == "1m" and r.exit_how == "open" and "ᵖ" in txt
    X.ons_run(LATER, TH, pd.DataFrame(columns=J.columns), None)                         # the journal has rolled off: re-priced from the row
    r = pd.read_csv(X.ONS_CSV).iloc[0]
    assert r.final and r.px_src == "tick" and r.exit_how == "TP +3"


def test_validation_quarantines_a_bad_row(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    a = DAY0 + X.H
    _stubs(monkeypatch, {a: dict(spike=a - 3 * X.H, adx=2.0, di=10.0)}, {"AUSDT": _paths(True)})
    X.ons_run(LATER, TH, _J([(a, "BLOCK", "AUSDT", "FRENZY_ATR_HIGH", "")]), None)
    d = pd.read_csv(X.ONS_CSV)
    d.loc[0, "pnl"] = 1.0                                                              # a TP row below +3 must never be saved
    d.to_csv(X.ONS_CSV, index=False)
    txt = "\n".join(X.ons_run(LATER, TH, pd.DataFrame(columns=["t", "e", "pair", "gate", "strategy"]), None))
    bad = glob.glob(str(tmp_path / "ons.csv.*.bad"))
    assert len(bad) == 1 and "TP exit at 1.0" in pd.read_csv(bad[0]).bad_reason.iloc[0] and "quarantined" in txt
    assert len(pd.read_csv(X.ONS_CSV)) == 0
    # the same failing row on later runs never spawns another .bad (one per row key)
    monkeypatch.setattr(X, "onscalp_validate", lambda r, s_: (_ for _ in ()).throw(ValueError("still bad")) if r.get("is_on") else None)
    X.ons_run(LATER, TH, _J([(a, "BLOCK", "AUSDT", "FRENZY_ATR_HIGH", "")]), None)
    X.ons_run(LATER, TH, _J([(a, "BLOCK", "AUSDT", "FRENZY_ATR_HIGH", "")]), None)
    assert len(glob.glob(str(tmp_path / "ons.csv.*.bad"))) == 1


def test_unreadable_state_moves_to_bad_and_starts_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    (tmp_path / "ons.csv").write_text("garbage,cols\n1,2\n")
    _stubs(monkeypatch, {}, {})
    X.ons_run(LATER, TH, pd.DataFrame(columns=["t", "e", "pair", "gate", "strategy"]), None)
    assert len(glob.glob(str(tmp_path / "ons.csv.*.bad"))) == 1


def test_book_snapshot_closest_within_60s(tmp_path, monkeypatch):
    a = DAY0 + X.H
    df = pd.DataFrame(dict(k=[X._iso(a), X._iso(a + X.H)], pair=["AUSDT", "BUSDT"], is_on=True))
    got = {(X._iso(a), "AUSDT"): dict(ob_imb_05=0.3), (X._iso(a + 60_000), "AUSDT"): dict(ob_imb_05=-0.9),
           (X._iso(a + X.H + 60_000), "BUSDT"): dict(ob_imb_05=0.1), (X._iso(a + X.H + 120_000), "BUSDT"): dict(ob_imb_05=0.7)}
    monkeypatch.setattr(X, "_ons_book_scan", lambda need, now: {k: v for k, v in got.items() if k in need})
    out = X._ons_attach_book(df, a + X.H)
    assert out.loc[0, "ob_imb_05"] == 0.3 and out.loc[0, "book_lag_s"] == 0 and out.loc[0, "book_level"] == "minute"
    assert out.loc[1, "ob_imb_05"] == 0.1 and out.loc[1, "book_lag_s"] == 60                      # the +120 s snapshot is outside the window


def test_book_scan_reads_journal_book_lines(tmp_path, monkeypatch):
    hdr = "t,e,pair,dir,gate,n,no_room,strategy,price,conf,cell,via,reason,closed,src," + ",".join(X.ONS_BOOK_COLS)
    vals = ",".join(str(i / 10) for i in range(len(X.ONS_BOOK_COLS)))
    f = tmp_path / "scalpars_decisions_paper_2026-10-07_05-00-00.csv"
    f.write_text(hdr + "\n" + f"2026-10-07T01:00:00,BOOK,AUSDT,,,,,,,,,,,,FRENZY,{vals}\n" + f"2026-10-07T01:00:00,BOOK,ZUSDT,,,,,,,,,,,,FRENZY,{vals}\n")
    monkeypatch.setattr(X.os.path, "expanduser", lambda p: str(tmp_path / os.path.basename(p)))
    got = X._ons_book_scan({("2026-10-07T01:00:00", "AUSDT")}, int(os.path.getmtime(f) * 1000))
    assert list(got) == [("2026-10-07T01:00:00", "AUSDT")] and got[("2026-10-07T01:00:00", "AUSDT")]["ob_imb_05"] == 0.3


def test_aggtrades_paging(monkeypatch):
    t0 = DAY0
    pages = [[dict(a=i, p="1.0", q="2", f=i, l=i, T=t0 + i, m=False) for i in range(1000)],
             [dict(a=1000 + i, p="1.0", q="2", f=0, l=1, T=t0 + 1000 + i * 100, m=True) for i in range(600)]]
    calls = []

    class R:
        def __init__(self, b):
            self.b = b

        def read(self):
            import json
            return json.dumps(self.b).encode()
    monkeypatch.setattr(X.urllib.request, "urlopen", lambda url, timeout: (calls.append(url), R(pages[len(calls) - 1]))[1])
    monkeypatch.setattr(X.time, "sleep", lambda s: None)
    st, T, p, q, m, nraw, trunc = X._aggtrades_rest("AUSDT", t0, t0 + 59_999, float("inf"))
    assert st == "ok" and len(calls) == 2 and "fromId=1000" in calls[1] and not trunc
    assert len(T) == 1000 + 590 and T.max() <= t0 + 59_999 and nraw[-1] == 2 and m[-1]           # prints after the window are dropped


def test_rate_limit_stops_the_tracker_for_this_run(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    calls = []

    def boom(sym, sig, th, live=False):
        calls.append(sym)
        raise X.urllib.error.HTTPError("u", 429, "Too Many Requests", None, None)
    monkeypatch.setattr(X, "_ons_replay", boom)
    J = _J([(DAY0 + i * X.H, "BLOCK", f"P{i}USDT", "FRENZY_ATR_HIGH", "") for i in range(1, 5)])
    txt = "\n".join(X.ons_run(LATER, TH, J, None))
    assert len(calls) == 1 and "Binance rate limit — stopped" in txt and "3 candidate(s) / row(s) left for the next run" in txt
    assert not os.path.exists(X.ONS_CSV) or len(pd.read_csv(X.ONS_CSV)) == 0


def test_flow_source_rest_recent_archive_older_and_400_fallback(monkeypatch):
    sig = DAY0
    used = []
    ok = lambda *a: ("ok", np.array([sig + 1_000]), np.array([100.0]), np.array([1.0]), np.array([False]), np.array([1.0]), False)
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s + i * X.MIN, 100.0, 100.0, 100.0, 100.0, 1.0] for i in range(1440)])
    monkeypatch.setattr(X, "_aggtrades_rest", lambda *a: (used.append("rest"), ok())[1])
    monkeypatch.setattr(X, "_aggtrades_archive", lambda *a: (used.append("archive"), ok())[1])
    b = {"dl": 2, "deadline": float("inf")}
    assert X._ons_flow("A", sig, 100.0, 6000.0, sig + X.H, b)["flow_src"] == "rest"
    assert X._ons_flow("A", sig, 100.0, 6000.0, sig + 3 * 86_400_000, b)["flow_src"] == "archive"
    def r400(*a):
        raise X.urllib.error.HTTPError("u", 400, "Search window is restricted to recent 2 days only", None, None)
    monkeypatch.setattr(X, "_aggtrades_rest", r400)
    assert X._ons_flow("A", sig, 100.0, 6000.0, sig + X.H, b)["flow_src"] == "archive" and used == ["rest", "archive", "archive"]


def _zip(csv_bytes, path):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("X-aggTrades-2026-10-07.csv", csv_bytes)
    return path


def test_read_archive_streams_the_window_with_qty_and_side(tmp_path):
    hdr = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    body = b"".join(b"%d,0.5,10,5,7,%d,%s\n" % (i, 1000 * i, b"true" if i % 2 else b"false") for i in range(1, 50))
    T, p, q, m, n = X._read_aggtrades_zip(_zip(hdr + body, tmp_path / "a.zip"), 3000, 6000, chunk=4)
    assert list(T) == [3000, 4000, 5000, 6000] and list(m) == [True, False, True, False] and list(n) == [3] * 4 and list(q) == [10] * 4
    T, *_ = X._read_aggtrades_zip(_zip(body, tmp_path / "b.zip"), 1000, 1000)
    assert list(T) == [1000]                                                        # no header line


def test_archive_parse_error_is_pending_and_only_an_old_404_is_missing(tmp_path, monkeypatch):
    X._arch_clear()
    bad = tmp_path / "bad.zip"; bad.write_bytes(b"not a zip")
    X._ARCH[("AUSDT", "2026-10-07")] = str(bad)
    b = {"dl": 2, "deadline": float("inf")}
    assert X._aggtrades_archive("AUSDT", DAY0, DAY0 + 59_999, LATER, b)[0] == "pending"
    X._ARCH.clear()

    def e404(url, timeout):
        raise X.urllib.error.HTTPError(url, 404, "Not Found", None, None)
    monkeypatch.setattr(X.urllib.request, "urlopen", e404)
    assert X._aggtrades_archive("AUSDT", DAY0, DAY0 + 59_999, DAY0 + 10 * 86_400_000, b)[0] == "pending"      # < 30 days: retried
    assert X._aggtrades_archive("AUSDT", DAY0, DAY0 + 59_999, DAY0 + 31 * 86_400_000, b)[0] == "missing"
    X._arch_clear()
    monkeypatch.setattr(X, "_aggtrades_archive", lambda *a: ("pending",) + (None,) * 6)
    monkeypatch.setattr(X, "_aggtrades_rest", lambda *a: (_ for _ in ()).throw(ValueError("garbled json")))
    assert X._ons_flow("A", DAY0, 100.0, 6000.0, DAY0 + X.H, b)["flow_state"] == "pending"


def test_archive_zip_downloaded_once_per_pair_day(tmp_path, monkeypatch):
    X._arch_clear()
    body = b"".join(b"%d,1.0,1,1,1,%d,false\n" % (i, DAY0 + 1000 * i) for i in range(1, 200))
    raw = open(_zip(body, tmp_path / "z.zip"), "rb").read()
    calls = []

    class R:
        def __init__(self):
            self.done = False

        def read(self, n=-1):
            if self.done:
                return b""
            self.done = True
            return raw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(X.urllib.request, "urlopen", lambda url, timeout: (calls.append(url), R())[1])
    b = {"dl": 5, "deadline": float("inf")}
    r1 = X._aggtrades_archive("AUSDT", DAY0 + 1000, DAY0 + 5000, LATER, b)
    r2 = X._aggtrades_archive("AUSDT", DAY0 + 100_000, DAY0 + 101_000, LATER, b)
    assert r1[0] == r2[0] == "ok" and len(calls) == 1 and b["dl"] == 4 and list(r2[1]) == [DAY0 + 100_000, DAY0 + 101_000]
    paths = [v for v in X._ARCH.values()]
    X._arch_clear()
    assert not any(os.path.exists(p_) for p_ in paths)                              # temp zips removed at the end of the run


def test_truncated_rest_read_is_never_stored_as_ok(monkeypatch):
    sig = DAY0
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s + i * X.MIN, 100.0, 100.0, 100.0, 100.0, 1.0] for i in range(1440)])
    tr = ("ok", np.array([sig + 1_000]), np.array([100.0]), np.array([1.0]), np.array([False]), np.array([1.0]), True)
    monkeypatch.setattr(X, "_aggtrades_rest", lambda *a: tr)
    monkeypatch.setattr(X, "_aggtrades_archive", lambda *a: ("pending",) + (None,) * 6)
    b = {"dl": 2, "deadline": float("inf")}
    f = X._ons_flow("A", sig, 100.0, 6000.0, sig + X.H, b)
    assert f["flow_state"] == "pending" and f["flow_trunc"]
    monkeypatch.setattr(X, "_aggtrades_archive", lambda *a: tr[:-1] + (False,))
    f = X._ons_flow("A", sig, 100.0, 6000.0, sig + X.H, b)
    assert f["flow_state"] == "ok" and f["flow_src"] == "archive" and f["flow_trunc"] is False
    with pytest.raises(ValueError):
        X.onscalp_validate(dict(is_on=True, strong=False, adx_delta=-1, di_spread=1, flow_state="ok", flow_trunc=True, usd_60=1.0), False)


def test_tick_window_reanchored_at_the_actual_entry_and_stuck_rows_finalise(tmp_path, monkeypatch):
    """the first print ≥ close + 12 s comes at +5 min → the 2 h exit is at entry + 2 h, past the close-anchored window (re-fetched); a tick
    path that stops before any exit falls through to 1m and finalises by age."""
    sig = DAY0 + X.H
    paths = {"AUSDT": (np.array([sig + 5 * X.MIN, sig + 100 * X.MIN, sig + 125 * X.MIN]), np.array([100.0, 101.0, 99.0])),
             "BUSDT": (np.array([sig + 13_000, sig + 60 * X.MIN]), np.array([100.0, 101.0]))}
    fetched = []

    def tks(pair, t0, t1, now, b):
        fetched.append(t1)
        tt, pp = paths[pair]
        return "ok", tt[tt <= t1], pp[tt <= t1]
    monkeypatch.setattr(X, "_ticks", tks)
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s + i * X.MIN, 100.0, 100.0, 100.0, 100.0, 1.0] for i in range(int((e - s) // X.MIN))])
    e, t_e, w, px, st, fin = X._ons_shadow("AUSDT", sig, LATER, {})
    assert fin and px == "tick" and w["how"] == "2 h" and t_e == sig + 5 * X.MIN and w["exit_ms"] == sig + 125 * X.MIN and len(fetched) == 2
    e, t_e, w, px, st, fin = X._ons_shadow("BUSDT", sig, LATER, {})
    assert fin and px.startswith("1m") and st == "open" and w["how"] == "2 h"
    e, t_e, w, px, st, fin = X._ons_shadow("BUSDT", sig, sig + 3 * X.H, {})         # not stale / given up yet → provisional
    assert not fin


def test_pnl_final_counts_before_the_flow_and_the_flow_fills_in_later(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    a = DAY0 + X.H
    _stubs(monkeypatch, {a: dict(spike=a - 3 * X.H, adx=2.0, di=10.0)}, {"AUSDT": _paths(True)})
    monkeypatch.setattr(X, "_aggtrades_rest", lambda *a_: ("pending",) + (None,) * 6)
    monkeypatch.setattr(X, "_aggtrades_archive", lambda *a_: ("pending",) + (None,) * 6)
    X.ons_run(LATER, TH, _J([(a, "BLOCK", "AUSDT", "FRENZY_ATR_HIGH", "")]), None)
    r = pd.read_csv(X.ONS_CSV).iloc[0]
    assert r.final and r.counted and r.flow_state == "pending"
    _stubs(monkeypatch, {a: dict(spike=a - 3 * X.H, adx=2.0, di=10.0)}, {"AUSDT": _paths(True)})
    monkeypatch.setattr(X, "_ons_shadow", lambda *a_: pytest.fail("a final P&L was re-priced"))
    X.ons_run(LATER, TH, pd.DataFrame(columns=["t", "e", "pair", "gate", "strategy"]), None)
    r = pd.read_csv(X.ONS_CSV).iloc[0]
    assert r.final and r.flow_state == "ok" and abs(r.buy_share_60 - 200 / 299) < 1e-9


def test_universe_exclusion_vol24_low_and_blacklist(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "ONS_CSV", str(tmp_path / "ons.csv")); monkeypatch.setattr(X, "ONS_CTRL_CSV", str(tmp_path / "ctrl.csv"))
    a, b, c = DAY0 + X.H, DAY0 + 2 * X.H, DAY0 + 3 * X.H
    scen = {a: dict(spike=a - 3 * X.H, adx=2.0, di=10.0, code="FRENZY_VOL24_LOW"), b: dict(spike=b - 3 * X.H, adx=2.0, di=10.0),
            c: dict(spike=c - 3 * X.H, adx=2.0, di=10.0)}
    _stubs(monkeypatch, scen, {p: _paths(True) for p in ("AUSDT", "BUSDT", "CUSDT")})
    J = _J([(a, "BLOCK", "AUSDT", "FRENZY_VOL24_LOW", ""), (b, "BLOCK", "BUSDT", "FRENZY_ATR_HIGH", ""), (c, "BLOCK", "CUSDT", "FRENZY_ATR_HIGH", "")])
    txt = "\n".join(X.ons_run(LATER, SimpleNamespace(frenzy_max_hours=96.0, frenzy_pair_blacklist="BUSDT"), J, None))
    s = pd.read_csv(X.ONS_CSV).set_index("pair")
    assert s.loc["AUSDT"].univ == "VOL24_LOW" and s.loc["BUSDT"].univ == "BLACKLIST" and s.loc["CUSDT"].univ == "ok"
    assert list(s.counted.astype(bool)) == [False, False, True] and "Outside the study's universe" in txt and "1/30 fires" in txt


def test_per_episode_year_reference_rederives_from_the_study_file():
    f = os.path.join(X.ROOT, "reports", "FRENZY_ON_SCALP_SIGNALS_2026-10-06.csv")
    if not os.path.exists(f):
        pytest.skip("study signal file not on disk")
    d = pd.read_csv(f)
    s = d[(d.adx_delta > 0) & (d.di_spread > 0)].copy()
    s["pnl"] = np.where(s.tX3 <= 120, 3.0, s.at120)
    s["hit"] = s.tX3 <= 120
    assert len(s) == 893 and abs(s.pnl.mean() - 0.136) < 0.001                     # the study's per-bar line
    s["spike_at"] = (pd.to_datetime(s.signal_utc) - pd.to_timedelta(s.hours, unit="h")).dt.round("5min").dt.strftime("%Y-%m-%dT%H:%M:%S")
    s["k"] = pd.to_datetime(s.signal_utc).dt.strftime("%Y-%m-%dT%H:%M:%S")
    s = s.reset_index(drop=True)
    e = s.sort_values(["k", "pair"], kind="stable").assign(_ep=X.episode_keys(s)).drop_duplicates("_ep")
    ci = X.day_ci(e.pnl.values, e.day.values)
    assert (len(e), e.day.nunique(), round(e.pnl.mean(), 3), round(e.hit.mean() * 100, 1)) == (502, 229, 0.105, 67.5)
    assert round(ci[0], 2) == -0.33 and round(ci[1], 2) == 0.51


def test_seventy_percent_leg_and_additions_labelled():
    w = pd.DataFrame(dict(day=[f"d{i % 16}" for i in range(32)], pair="P", pnl=[3.0, 3.0, -1.0, 3.0] * 8, mae=-1.0,
                          hit=[True, True, False, True] * 8, entry_at=[f"2026-10-08T{i % 24:02d}:00:00" for i in range(32)]))
    assert X.onscalp_check(w)[0] == "candidate"
    st, tx = X.onscalp_check(w.assign(hit=[True, False, False, True] * 8))            # 50 % < 70 %: every other leg passes
    assert st == "collecting" and "P(+3 within 2 h) 50 %" in tx
    assert "(addition)" in X.onscalp_check(w.assign(pnl=[1.0, -2.0] * 16))[1]


def test_live_override_on_no_episode_and_short_1h_fetch_raises(monkeypatch):
    sig = DAY0 + X.H
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s + i * STEP[tf], 100.0, 100.5, 99.5, 100.0, 1.0]
                                                       for i in range(min(int((e - s) // STEP[tf]) + 1, 1500))])
    monkeypatch.setattr(X, "frenzy_walk", lambda bars, nh, th: None)
    monkeypatch.setattr(X, "normal_hour_usd", lambda h1, ms: 6e6)
    monkeypatch.setattr(X, "frenzy_adx_delta", lambda b: 1.0); monkeypatch.setattr(X, "frenzy_di_spread", lambda b: 1.0)
    assert X._ons_replay("A", sig, TH)["kind"] == "not_on"
    r = X._ons_replay("A", sig, TH, live=True)
    assert r["kind"] == "on" and not r["parity"] and r["replay_code"] == "LIVE_ONLY" and r["spike_at"] is None and "NO_EPISODE" in r["parity_why"]
    monkeypatch.setattr(X, "normal_hour_usd", lambda h1, ms: None)
    with pytest.raises(ValueError):
        X._ons_replay("A", sig, TH)                                                 # a full-length 1h window without a normal hour = a bad read
    monkeypatch.setattr(X, "_kl", lambda sym, tf, s, e: [[s + i * STEP[tf] + (5 * X.H if tf == "1h" else 0), 100.0, 100.5, 99.5, 100.0, 1.0]
                                                       for i in range(min(int((e - s) // STEP[tf]) + 1, 1500))])
    assert X._ons_replay("A", sig, TH)["why"].startswith("NO_NORMAL_HOUR")          # the listing starts inside the window: genuinely young

