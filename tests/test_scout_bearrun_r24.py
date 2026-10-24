"""🐻⏬ BEARRUN_R24_25 shadow scout (DECISION_LOG 369): T0 vs V25 on the engine's own state machine (the study's run_variant), extra-window
extraction + merge + kind labels (EARLIER / TAIL / EXTRA), the live-config reading, parity vs live fills, pricing through the V2 basket
with injected klines, the frozen read (CANDIDATE / KEEP_4 / WAITING / COLLECTING / decided once), the NOT-evaluable branch, the 5m page
cache and the network stops. Synthetic data only — real network blocked, no write outside tmp_path."""
import json
import os
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import scout_bearrun_r24 as S  # noqa: E402
import scout_bearrun_v2 as V2  # noqa: E402

H = S.HOUR
M5 = S.FIVE
T_BASE = int(pd.Timestamp("2026-11-02T00:00:00Z").value // 1_000_000)
KW = dict(r24_on=4.0, r24_off=3.0, below_on=50.0, below_off=47.0, eff_on=0.15, eff_off=0.15, latch_r6h=3.0, latch_ema50=True)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """no real network, no write to the real reports/ state or caches."""
    def _no_net(*a, **k):
        raise AssertionError("real network call in a test")
    monkeypatch.setattr(urllib.request, "urlopen", _no_net)
    monkeypatch.setattr(V2, "CACHE", str(tmp_path / "v2cache"))
    monkeypatch.setattr(S, "CACHE", str(tmp_path / "r24cache"))
    monkeypatch.setattr(S, "STATE", str(tmp_path / "state.json"))
    monkeypatch.setattr(V2, "STATE", str(tmp_path / "v2state.json"))
    yield


# ─────────────────────────── helpers ───────────────────────────
def _R(r24, below=60.0, eff=0.5, r6=-1.0, step=M5, t0=T_BASE):
    """hand-made readings (one row per step): px 100, e50h 200 (no EMA50 latch)."""
    n = len(r24)
    t = pd.to_datetime([t0 + i * step for i in range(n)], unit="ms")
    f = lambda v: (list(v) if hasattr(v, "__len__") else [v] * n)
    return pd.DataFrame({"t": t, "px": 100.0, "hi": 100.0, "lo": 99.0, "r24": f(r24), "r12": 0.0, "r6": f(r6), "below24": f(below),
                         "eff24": f(eff), "dd24": 0.0, "off24lo": 0.0, "effH": 0.0, "hi24": 100.0, "e50h": 200.0})


def _net(get, **kw):
    return V2.Net(http_get=get, pace_s=0, cooldown_path=kw.pop("cooldown_path", None) or os.devnull + ".none", **kw)


class Server:
    """public-fapi stub: BTCUSDT 5m klines from a frame, alt 1 h klines from a path function, universe calls; counts requests."""

    def __init__(self, btc5=None, alt=None, now_ms=None, code=200, bad=()):
        self.btc5, self.alt, self.now, self.code, self.calls, self.bad = btc5, alt, now_ms, code, [], set(bad)

    def __call__(self, url):
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        self.calls.append((u.path, q))
        hdr = {"X-MBX-USED-WEIGHT-1M": str(5 * len(self.calls))}
        if self.code != 200:
            return self.code, {"Retry-After": "60"}, ""
        if u.path == "/fapi/v1/klines" and q.get("interval") == "5m" and q["symbol"] == "BTCUSDT":
            st, lim = int(q["startTime"]), int(q["limit"])
            b = self.btc5[(self.btc5.open_time >= st)].head(lim)
            rows = [[int(r.open_time), str(r.o), str(r.h), str(r.l), str(r.c), str(r.vol), 0, "0"] for r in b.itertuples()]
            return 200, hdr, json.dumps(rows)
        if u.path == "/fapi/v1/klines":
            if q["symbol"] in self.bad:
                return 400, hdr, ""
            st, lim = int(q["startTime"]), int(q["limit"])
            step = M5 if q.get("interval") == "5m" else H
            rows = []
            for k in range(lim):
                o = st + k * step
                if o + step > self.now:
                    break
                c0, c1 = self.alt(q["symbol"], o), self.alt(q["symbol"], o + step)
                rows.append([o, str(c0), str(max(c0, c1) * 1.001), str(min(c0, c1) * 0.999), str(c1), "1", 0, "1000000"])
            return 200, hdr, json.dumps(rows)
        return 404, hdr, ""


# ─────────────────────────── T0 vs V25 + extra windows ───────────────────────────
def test_v25_is_superset_and_tail_earlier_labels():
    # 0 → −3 (V25 only) → −4.5 (both) → −3.5 (both stay) → −2 (T0 off, V25 stays: TAIL) → 0 (both off)
    seq = [0.0] * 5 + [-3.0] * 6 + [-4.5] * 4 + [-3.5] * 4 + [-2.0] * 5 + [0.0] * 5
    R = _R(seq)
    t0, v25 = S.states(R, KW)
    assert not (t0 & ~v25).any()                                     # V25 ⊇ T0
    assert list(np.nonzero(v25)[0]) == list(range(5, 24))
    assert list(np.nonzero(t0)[0]) == list(range(11, 19))
    X = S.extra_windows(R, t0, v25, 180)
    assert [(w["i0"], w["i1"], w["end_reason"], w["kind"]) for w in X] == [(5, 11, "T0 ON", "EARLIER"), (19, 24, "V25 OFF", "TAIL")]
    assert X[0]["t0_on_ms"] == T_BASE + 11 * M5 and X[1]["t0_on_ms"] is None


def test_live_thresholds_change_t0_not_the_shadow_leg():
    seq = [0.0] * 3 + [-3.2] * 4 + [0.0] * 3
    R = _R(seq)
    t0, v25 = S.states(R, dict(KW, r24_on=3.0, r24_off=2.0))      # a live config at 3 / 2 → T0 fires on −3.2 too
    assert t0[3:7].all() and v25[3:7].all()
    assert S.extra_windows(R, t0, v25, 180) == []
    t0b, v25b = S.states(R, dict(KW, below_on=70.0))                # every other leg follows the live config in BOTH variants
    assert not v25b.any() and not t0b.any()


def test_merge_gap_rules_and_standalone_extra():
    step = 30 * 60_000                                              # 30-min readings
    seq = [0.0] * 2 + [-3.0] * 2 + [0.0] * 4 + [-3.0] * 2 + [0.0] * 8 + [-3.0] * 2 + [0.0] * 2
    R = _R(seq, step=step)
    t0, v25 = S.states(R, KW)
    X = S.extra_windows(R, t0, v25, 180)
    # gap 1 = 4 readings × 30 min = 120 min ≤ 180 → merged; gap 2 = 240 min → a new window
    assert [(w["i0"], w["i1"], w["stretches"], w["kind"]) for w in X] == [(2, 10, 2, "EXTRA"), (18, 20, 1, "EXTRA")]
    assert all(w["end_reason"] == "V25 OFF" for w in X)


def test_no_merge_across_t0_on_and_earlier_after_v25_off():
    step = 30 * 60_000
    # extra (−3) → V25 OFF (eff fails) 1 h → both ON (−5): the extra window ends on V25 OFF but T0 turns ON ≤ 180 min later → EARLIER
    seq = [0.0] * 2 + [-3.0] * 2 + [-3.0] * 2 + [-5.0] * 3 + [0.0] * 2
    eff = [0.5] * 4 + [0.05] * 2 + [0.5] * 3 + [0.5] * 2
    R = _R(seq, eff=eff, step=step)
    t0, v25 = S.states(R, KW)
    X = S.extra_windows(R, t0, v25, 180)
    assert [(w["i0"], w["i1"], w["end_reason"], w["kind"]) for w in X] == [(2, 4, "V25 OFF", "EARLIER")]
    assert X[0]["t0_on_ms"] == T_BASE + 6 * step
    # T0 ON between two extra stretches < 180 min apart → NOT merged (tail + earlier labels)
    seq2 = [0.0] + [-3.0] * 2 + [-4.5] * 1 + [-2.0] * 2 + [-4.5] * 1 + [-2.0] * 2 + [0.0] * 2
    R2 = _R(seq2, step=step)
    a, b = S.states(R2, KW)
    X2 = S.extra_windows(R2, a, b, 180)
    assert [w["kind"] for w in X2] == ["EARLIER", "TAIL→EARLIER", "TAIL"]


def test_open_window_at_the_last_reading():
    R = _R([0.0] * 3 + [-3.0] * 3)
    t0, v25 = S.states(R, KW)
    X = S.extra_windows(R, t0, v25, 180)
    assert len(X) == 1 and X[0]["end_ms"] is None and X[0]["end_reason"] == "open" and X[0]["i1"] == 6


def test_live_engine_kw_reads_config_and_defaults(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"thresholds": {"bearrun_r24_on": 4.0, "bearrun_r24_off": 3.0, "bearrun_below_on": 55.0,
                                            "bearrun_latch_ema50": False, "bearrun_window_merge_minutes": 120}}))
    kw, merge = S.live_engine_kw(str(p))
    assert kw["below_on"] == 55.0 and kw["latch_ema50"] is False and kw["eff_on"] == 0.15 and merge == 120.0
    assert S.v25_kw(kw)["r24_on"] == 2.5 and S.v25_kw(kw)["r24_off"] == 1.5 and S.v25_kw(kw)["below_on"] == 55.0
    kw2, merge2 = S.live_engine_kw(str(tmp_path / "missing.json"))
    assert kw2 == KW and merge2 == 180.0
    kw3, _ = S.live_engine_kw()                                     # the real file is read as the engine reads it
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert kw3["r24_on"] == th.get("bearrun_r24_on", 4.0) and kw3["r24_off"] == th.get("bearrun_r24_off", 3.0)


def test_study_t0_engine_kw_default_and_override():
    R = _R([0.0] * 3 + [-4.2] * 3 + [-3.1] * 2 + [-2.9] * 2 + [-2.0] * 2)
    exp_default = [False] * 3 + [True] * 5 + [False] * 4                   # ON at −4.2, stays at −3.1, OFF at −2.9 (> −3)
    assert list(S.TT.run_variant(R, "T0")) == exp_default                  # the study's own T0 (engine defaults) — unchanged
    assert list(S.TT.run_variant(R, "T0", engine_kw=KW)) == exp_default
    exp_v25 = [False] * 3 + [True] * 9                                     # 2.5 / 1.5: stays through −2.9 and −2.0
    assert list(S.TT.run_variant(R, "T0", engine_kw=S.v25_kw(KW))) == exp_v25
    assert list(S.TT.run_variant(R, "T0", engine_kw=dict(KW, latch_r6h=0.5))) == exp_default   # r6 −1 < 0.5: latch idle


# ─────────────────────────── parity ───────────────────────────
def test_parity_match_rate():
    n = 8 * 288 + 40
    seq = [0.0] * n
    for i in range(8 * 288 + 10, 8 * 288 + 20):
        seq[i] = -4.5
    R = _R(seq)
    t0, _ = S.states(R, KW)
    on_ms = T_BASE + (8 * 288 + 12) * M5
    F = pd.DataFrame({"o_ms": [on_ms, on_ms + 60_000, T_BASE + (8 * 288 + 35) * M5, T_BASE], "pair": ["A", "B", "C", "D"], "src": "x"})
    P = S.parity(R, t0, F, 180, 15)
    assert P["n_fills"] == 3                                         # the warm-up fill is out of range
    assert P["matched"] == 2 and P["n_live_w"] == 1 and P["live_w_matched"] == 1
    assert len(P["replica_w"]) == 1 and P["replica_w_with_fill"] == 1
    assert "2/3" in S._parity_line(P) and "containment" in S._parity_line(P)


def test_load_fills_dedups_journal_and_orders(tmp_path):
    d = tmp_path / "scalpars_decisions_paper_x.csv"
    pd.DataFrame({"t": ["2026-10-22T17:16:24.403", "2026-10-22T17:16:50", "2026-10-22T18:00:00"], "e": ["OPEN", "OPEN", "BLOCK"],
                  "pair": ["SOLUSDT", "SOLUSDT", "APTUSDT"], "strategy": ["BEARRUN_SHORT", "BEARRUN_SHORT", "BEARRUN_SHORT"]}).to_csv(d, index=False)
    o = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame({"opened_at": ["2026-10-22 17:16:30", "2026-10-07 14:49:11"], "closed_at": ["", ""], "pair": ["SOLUSDT", "UNIUSDT"],
                  "direction": ["SHORT", "SHORT"], "status": ["CLOSED", "CLOSED"], "entry_strategy": ["BEARRUN_SHORT", "BEARRUN_SHORT"],
                  "pnl": [1, 2], "pnl_percentage": [0.1, 0.2], "close_reason": ["TP", "TP"], "entry_bear_bypass": ["", ""],
                  "is_paper": [True, True]}).to_csv(o, index=False)
    F = S.load_fills([str(d)], [str(o)])
    assert sorted(F.pair) == ["SOLUSDT", "UNIUSDT"]                 # journal 17:16:24 / 17:16:50 + orders 17:16:30 = ONE fill (±2 min)


# ─────────────────────────── frozen read ───────────────────────────
def _rec(k, a, cls="clean_slide", weeks_apart=True):
    on = T_BASE + k * (8 * 24 * H if weeks_apart else 6 * H)
    return dict(on_ms=on, a_mean=a, d_mean=a, cls=cls, kind="EXTRA")


def test_gate_collecting_waiting_candidate_keep():
    W = [_rec(k, 0.3 + 0.01 * k) for k in range(19)]
    assert S.gate_read(W)[0] == "COLLECTING"
    W20 = W + [_rec(19, 0.2)]
    assert S.gate_read(W20, pending=[W20[5]["on_ms"] + 1])[0] == "WAITING"
    v, txt, n = S.gate_read(W20)
    assert v == "CANDIDATE" and n == 20 and "operator decides" in txt
    W_fs = [_rec(k, 0.3) for k in range(10)] + [_rec(10 + k, -0.25, "false_start") for k in range(10)]
    v2, txt2, _ = S.gate_read(W_fs)
    assert v2 == "KEEP_4" and "④" in txt2 and "①" not in txt2
    W_neg = [_rec(k, -0.1) for k in range(20)]
    assert S.gate_read(W_neg)[0] == "KEEP_4"
    W_conc = [_rec(0, 10.0)] + [_rec(k, 0.01) for k in range(1, 20)]
    v3, txt3, _ = S.gate_read(W_conc)
    assert v3 == "KEEP_4" and "③" in txt3
    same_week = [_rec(k, 0.3, weeks_apart=False) for k in range(25)]
    assert S.gate_read(same_week)[0] == "COLLECTING"                 # ≥ 20 windows but < 6 ISO weeks


def test_gate_prefix_is_the_shortest_and_decided_once():
    W = [_rec(k, 0.3) for k in range(20)] + [_rec(20 + k, -5.0) for k in range(5)]
    st = S._fresh()
    v, txt, now_ = V2.decide_once(st, "gate_read", S.gate_read, W, T_BASE)
    assert v == "CANDIDATE" and now_ and st["gate_read"]["prefix_n"] == 20
    v2, txt2, now2 = V2.decide_once(st, "gate_read", S.gate_read, [_rec(k, -1.0) for k in range(25)], T_BASE + H)
    assert v2 == "CANDIDATE" and not now2 and "frozen" in txt2


def test_interim_line():
    assert "not reached" in S.interim([_rec(k, 0.1) for k in range(7)])
    assert "NON-TERMINAL" in S.interim([_rec(k, 0.1) for k in range(8)])


# ─────────────────────────── end to end (synthetic 5m BTC + injected alt klines) ───────────────────────────
def _btc5(days_flat=9, decline_h=24, drop=3.0, flat_after_h=40):
    n0, n1, n2 = days_flat * 288, decline_h * 12, flat_after_h * 12
    c = [100.0] * n0 + list(100.0 * (1 - drop / 100 * np.arange(1, n1 + 1) / n1)) + [100.0 * (1 - drop / 100)] * n2
    c = np.array(c) * 1000
    t0 = T_BASE - (n0 + n1 + n2) * M5
    t0 -= t0 % S.PAGE_MS                                            # page-aligned start
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open_time": t0 + np.arange(len(c)) * M5, "o": o, "h": np.maximum(o, c) * 1.0001,
                         "l": np.minimum(o, c) * 0.9999, "c": c, "vol": 1.0})


def _alt(sym, ms):
    return 10.0 * (1 - 0.0004 * ((ms - T_BASE) / H)) if sym != "ZZZUSDT" else 10.0


NOFILLS = pd.DataFrame(columns=["o_ms", "pair", "src"])


def _run(tmp_path, b, srv, deploy="first", syms=("AAAUSDT", "ZZZUSDT"), st_path=None, cfg=None, now=None):
    now = now or int(b.open_time.iloc[-1]) + M5 + 60_000
    if cfg is None:
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"thresholds": {}}))
    srv.now = now
    dep = int(b.open_time.iloc[0]) if deploy == "first" else deploy
    return "\n".join(S.run(now_ms=now, net=_net(srv), state_path=st_path, deploy_ms=dep, btc5=b, syms=list(syms), fills=NOFILLS,
                           cfg_path=str(cfg), cache_dir=str(tmp_path / "c")))


def test_run_end_to_end_persists_and_prices(tmp_path):
    b = _btc5()
    srv = Server(alt=_alt)
    st_path = str(tmp_path / "st.json")
    txt = _run(tmp_path, b, srv, st_path=st_path)
    assert "Persisted 1 newly final" in txt, txt
    st = json.load(open(st_path))
    assert len(st["windows"]) == 1 and st["seen"] == {}
    w = st["windows"][0]
    assert w["status"] == "final" and w["kind"] == "EXTRA" and w["end_reason"] == "V25 OFF" and w["d_exit"] == "window end"
    assert w["cls"] in ("clean_slide", "true_other", "false_start")
    assert w["a_n"] == 2 and w["a_mean"] is not None and w["d_n"] == 2
    assert "COLLECTING" in txt and "Parity (containment): no live Bear-Run fill" in txt
    assert any(q.get("interval") == "5m" and q["symbol"] == "AAAUSDT" and int(q["startTime"]) == w["on_ms"] - M5 for _, q in srv.calls)
    n_calls = len(srv.calls)                                         # a second run appends nothing and refetches nothing
    txt2 = _run(tmp_path, b, srv, st_path=st_path)
    assert "newly final" not in txt2 and len(srv.calls) == n_calls
    assert len(json.load(open(st_path))["windows"]) == 1


def test_state_default_resolved_at_runtime(tmp_path, monkeypatch):
    p = tmp_path / "patched_state.json"
    monkeypatch.setattr(S, "STATE", str(p))
    _run(tmp_path, _btc5(), Server(alt=_alt))                         # state_path None → S.STATE read at call time
    assert p.exists() and len(json.load(open(p))["windows"]) == 1


def test_not_evaluable_persists_nothing(tmp_path):
    p = tmp_path / "st.json"
    txt = _run(tmp_path, _btc5(), Server(alt=_alt), deploy=None, st_path=str(p))
    assert "NOT evaluable" in txt and "Context (last 7 days" in txt and not p.exists()
    assert "context totals (priced): 1 windows" in txt                # pre-commit context windows are priced once final


def test_context_not_priced_once_cohort_evaluable(tmp_path):
    b = _btc5()
    srv = Server(alt=_alt)
    txt = _run(tmp_path, b, srv, deploy=int(b.open_time.iloc[-1]), st_path=str(tmp_path / "s.json"))
    assert "Context (last 7 days" in txt and not [q for _, q in srv.calls if q.get("symbol") != "BTCUSDT"]


def test_window_not_final_is_pending_and_seen(tmp_path):
    b = _btc5(flat_after_h=14)                                       # ended, but 24 h since the start not reached
    p = tmp_path / "st.json"
    srv = Server(alt=_alt)
    txt = _run(tmp_path, b, srv, st_path=str(p))
    assert "Open / not yet final cohort extra windows" in txt and "newly final" not in txt
    assert not [q for _, q in srv.calls if q.get("symbol") != "BTCUSDT"]          # only final windows are priced
    st = json.load(open(p))
    assert st["windows"] == [] and len(st["seen"]) == 1


def test_seen_window_that_scrolls_out_is_lost(tmp_path):
    p = tmp_path / "st.json"
    st = S._fresh()
    st["seen"] = {str(T_BASE - 40 * 24 * H): "09-23 00:00"}
    V2.save_state(st, str(p))
    txt = _run(tmp_path, _btc5(), Server(alt=_alt), deploy=T_BASE - 41 * 24 * H, st_path=str(p))
    st2 = json.load(open(p))
    lost = [w for w in st2["windows"] if w["status"] == "lost"]
    assert len(lost) == 1 and "LOST" in txt and "lost 1" in txt and st2["seen"] == {}


def test_unpriceable_window_is_persisted_not_dropped(tmp_path):
    p = tmp_path / "st.json"
    txt = _run(tmp_path, _btc5(), Server(alt=_alt, bad=("AAAUSDT", "ZZZUSDT")), st_path=str(p))
    w = json.load(open(p))["windows"]
    assert len(w) == 1 and w[0]["status"] == "unpriceable" and "UNPRICEABLE" in txt and "unpriceable 1" in txt
    assert S._final(w) == []


def test_r24_guard_skips_the_read(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"thresholds": {"bearrun_r24_on": 2.5, "bearrun_r24_off": 1.5}}))
    p = tmp_path / "st.json"
    srv = Server(alt=_alt)
    txt = _run(tmp_path, _btc5(), srv, st_path=str(p), cfg=cfg)
    assert "GUARD" in txt and "SKIPPED" in txt and not p.exists() and not srv.calls
    assert not S.superset_ok(dict(KW, r24_on=4.0, r24_off=1.0)) and S.superset_ok(KW)


def _bars_alt(sym, start, n, f):
    return [[start - M5 + k * M5, f(start - M5 + (k - 1) * M5), 0, 0, f(start - M5 + k * M5), 1.0] for k in range(n)]


def test_ten_minute_window_enters_inside_and_exits_at_its_end(tmp_path):
    start = T_BASE + 7 * M5                                            # not on an hour boundary
    w = dict(start_ms=start, end_ms=start + 2 * M5, kind="EXTRA", v25_end_ms=start + 2 * M5)
    price = lambda ms: 10.0 * (1 - 0.001 * ((ms - start) / M5))     # falls 0.1 % per 5m bar
    srv = Server(alt=lambda s, ms: price(ms), now_ms=start + 25 * H)
    P = S.price_5m(w, _net(srv), start + 25 * H, ["AAAUSDT"], cache_dir=str(tmp_path / "c"))
    assert P["d_exit"] == "window end" and P["d_exit_ms"] == start + 2 * M5
    k5 = [q for _, q in srv.calls if q.get("interval") == "5m"]
    assert int(k5[0]["startTime"]) == start - M5                       # entry = the close of the bar closing AT the start
    en, ex = price(start), price(start + 2 * M5)
    assert P["d"]["mean"] == pytest.approx((1 - ex / en) * 100 - V2.FEE_RT, abs=1e-4)
    rk = [q for _, q in srv.calls if q.get("interval") == "1h"]
    assert int(rk[0]["startTime"]) == (start // H) * H - 24 * H        # ranking = the 24 closed 1 h bars before the start


def test_earlier_d_holds_to_v25_off():
    s = T_BASE
    assert S.d_exit_ms(dict(start_ms=s, end_ms=s + H, kind="EARLIER", v25_end_ms=s + 5 * H)) == (s + 5 * H, "V25 OFF")
    assert S.d_exit_ms(dict(start_ms=s, end_ms=s + H, kind="TAIL→EARLIER", v25_end_ms=None)) == (s + 24 * H, "24 h cap")
    assert S.d_exit_ms(dict(start_ms=s, end_ms=s + H, kind="EXTRA", v25_end_ms=s + H)) == (s + H, "window end")
    assert S.d_exit_ms(dict(start_ms=s, end_ms=s + 30 * H, kind="TAIL", v25_end_ms=None)) == (s + 24 * H, "window end")


def test_drop_pct_excludes_the_first_non_extra_bar():
    R = _R([0.0] * 3 + [-3.0] * 3 + [0.0] * 3)
    R.loc[6, "lo"] = 50.0                                              # the first non-extra bar has a deep wick
    t0, v25 = S.states(R, KW)
    w = S.extra_windows(R, t0, v25, 180)[0]
    assert S.base_record(w, R)["drop_pct"] == pytest.approx(-1.0)


def test_parity_dedup_per_pair_two_minutes():
    F = pd.DataFrame({"o_ms": [0, 90_000, 200_000, 30_000], "pair": ["A", "A", "A", "B"], "src": "x"})
    D = S.dedup_fills(F)
    assert list(zip(D.pair, D.o_ms)) == [("A", 0), ("B", 30_000), ("A", 200_000)]


def test_engine_parity_of_the_readings():
    """rebuild the engine's _update_bullrun_monitor readings (999 closed 5m closes, EMA20 seeded at the first; 149 closed 1 h closes,
    EMA50 seeded at the first) at one bar and compare with compute_readings there."""
    rng = np.random.default_rng(3)
    n = 4000
    c = 80_000 * np.exp(np.cumsum(rng.normal(-0.00005, 0.002, n)))
    t0 = T_BASE - T_BASE % H
    o = np.r_[c[0], c[:-1]]
    b = pd.DataFrame({"open_time": t0 + np.arange(n) * M5, "o": o, "h": np.maximum(o, c) * 1.0005, "l": np.minimum(o, c) * 0.9995,
                      "c": c, "vol": 1.0})
    R = S.TT.compute_readings(b)
    i = n - 6                                                        # a reading at hh:35 (not on an hour boundary)
    closes = list(c[i - 998:i + 1])
    ema, k, above = closes[0], 2.0 / 21.0, []
    for x in closes:
        ema = x * k + ema * (1 - k)
        above.append(x > ema)
    w24 = closes[-288:]
    r24 = (w24[-1] / w24[0] - 1) * 100
    eff24 = abs(w24[-1] - w24[0]) / sum(abs(w24[j] - w24[j - 1]) for j in range(1, 288))
    below24 = 100 - 100 * sum(above[-288:]) / 288
    r6 = (closes[-1] / closes[-72] - 1) * 100
    t_read = int(b.open_time.iloc[i]) + M5
    hb = b[b.open_time < t_read - (t_read % H)].copy()               # 5m bars of the 1 h bars CLOSED by the reading
    hc = hb.assign(hr=hb.open_time // H).groupby("hr").c.last().values[-149:]
    e = hc[0]
    for x in hc:
        e = x * (2.0 / 51.0) + e * (1 - 2.0 / 51.0)
    assert R.r24.values[i] == pytest.approx(r24, abs=1e-9)
    assert R.r6.values[i] == pytest.approx(r6, abs=1e-9)
    assert R.eff24.values[i] == pytest.approx(eff24, abs=1e-9)
    assert R.below24.values[i] == pytest.approx(below24, abs=0.5)    # EMA20 seed 999 bars back vs the whole series
    assert R.e50h.values[i] == pytest.approx(e, rel=1e-4)            # EMA50 seed 149 h back vs the whole series


# ─────────────────────────── 5m pages + network stops ───────────────────────────
def test_btc5m_pages_cache_complete_pages_only(tmp_path):
    b = _btc5()
    now = int(b.open_time.iloc[-1]) + M5 + 1
    srv = Server(btc5=b, now_ms=now)
    got = S.btc_5m(_net(srv), now, days=3, cache_dir=str(tmp_path / "c"))
    assert got is not None and got.open_time.is_monotonic_increasing and int(got.open_time.iloc[-1]) == int(b.open_time.iloc[-1])
    assert all(q["limit"] == "1000" and int(q["startTime"]) % S.PAGE_MS == 0 for _, q in srv.calls)
    n1 = len(srv.calls)
    S.btc_5m(_net(srv), now, days=3, cache_dir=str(tmp_path / "c"))
    assert len(srv.calls) - n1 == 1                                  # only the incomplete last page is refetched


@pytest.mark.parametrize("code", [418, 429, 403])
def test_network_stops(tmp_path, code):
    cd = tmp_path / "cooldown"
    net = V2.Net(http_get=Server(code=code), pace_s=0, cooldown_path=str(cd))
    L = S.run(now_ms=T_BASE, net=net, state_path=str(tmp_path / "s.json"), deploy_ms=None, syms=[],
              fills=pd.DataFrame(columns=["o_ms", "pair", "src"]), cfg_path=str(tmp_path / "x.json"), cache_dir=str(tmp_path / "c"))
    txt = "\n".join(L)
    assert "unreadable this run" in txt and net.stop and len(net.get.calls) == 1
    assert cd.exists() == (code in (418, 429))


def test_budget_stop_before_request(tmp_path):
    srv = Server(btc5=_btc5(), now_ms=T_BASE)
    net = V2.Net(http_get=srv, pace_s=0, cooldown_path=str(tmp_path / "cd"), weight_stop=0)
    assert S.btc_5m(net, T_BASE, days=3, cache_dir=str(tmp_path / "c")) is None and net.stop and not srv.calls


def test_two_stretch_earlier_window_d_holds_to_v25_off_after_the_end():
    """verification fix: a merged EARLIER window's gap is V25-off too — V25 OFF is searched from the window END, never inside the gap."""
    import numpy as np
    R = _R([0.0] * 20)
    v25 = np.zeros(20, bool); v25[3:6] = True; v25[8:16] = True          # stretch 3-5, gap 6-7, stretch 8-15
    t0 = np.zeros(20, bool); t0[11:14] = True                             # T0 joins at 11 → window 3 … 11 is EARLIER
    w = S.extra_windows(R, t0, v25, 180)[0]
    assert w["stretches"] == 2 and "EARLIER" in w["kind"] and w["i1"] == 11
    tms = S._tms(R)
    assert w["v25_end_ms"] == int(tms[16])                                # not the gap at 6
    assert S.d_exit_ms(w) == (int(tms[16]), "V25 OFF")
