"""🐻⏩ BEARRUN_V2_SHADOW scout (DECISION_LOG 359): the frozen V2-C1 state machine on closed 1 h bars (ON legs, the three OFF legs, the
later-hour re-entry rule), window classes, the basket proxies (exit d: OFF close / +2 % stop / 24 h cap · exit a': TP / stop / both-in-bar /
6 h cap), C2 label, append-only persistence + finality, the frozen read outcomes (decided once), the NOT-evaluable path, and the network
budget / cooldown stops. Synthetic klines only — no network."""
import json
import os
import sys
import urllib.parse

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_bearrun_v2 as S  # noqa: E402

H = S.HOUR
T_BASE = int(pd.Timestamp("2026-11-01T00:00:00Z").value // 1_000_000)


# ─────────────────────────── helpers ───────────────────────────
def _frame(n, **cols):
    """a hand-made indicator frame: defaults = no signal (slope +, +DI > −DI, r6 0)."""
    D = pd.DataFrame({"open": [T_BASE + i * H for i in range(n)]})
    D["t"] = D.open + H
    for k, v in dict(c=100.0, h=100.2, l=99.8, slope3=0.1, pdi=30.0, ndi=20.0, r6=0.0).items():
        D[k] = v
    for k, v in cols.items():
        D[k] = v
    run, prev = [], 0
    for g in (D.pdi > D.ndi).values:
        prev = prev + 1 if g else 0
        run.append(prev)
    D["pdi_run"] = run
    return D


def _set(D, i, **kw):
    for k, v in kw.items():
        D.loc[i, k] = v
    run, prev = [], 0
    for g in (D.pdi > D.ndi).values:
        prev = prev + 1 if g else 0
        run.append(prev)
    D["pdi_run"] = run


def _bear(D, i):
    _set(D, i, slope3=-0.1, pdi=15.0, ndi=30.0, r6=-1.5)


def _btc_rows(closes, t0=T_BASE, wick=0.001):
    out = []
    for i, c in enumerate(closes):
        prev = closes[i - 1] if i else c
        out.append([t0 + i * H, prev, max(prev, c) * (1 + wick), min(prev, c) * (1 - wick), c, 1e9])
    return out


class FakeHTTP:
    """serves /fapi/v1/klines from per-symbol synthetic rows; counts calls; optional forced status codes."""

    def __init__(self, series=None, codes=None, weight=10, info=None, tick=None):
        self.series, self.codes, self.weight, self.calls = series or {}, list(codes or []), weight, []
        self.info, self.tick = info, tick

    def __call__(self, url):
        self.calls.append(url)
        hdr = {"X-MBX-USED-WEIGHT-1M": str(self.weight)}
        if self.codes:
            code = self.codes.pop(0)
            if code != 200:
                return code, {"Retry-After": "60"}, ""
        u = urllib.parse.urlparse(url)
        q = dict(urllib.parse.parse_qsl(u.query))
        if u.path.endswith("/klines"):
            rows = self.series.get(q["symbol"], [])
            st, lim = int(q["startTime"]), int(q["limit"])
            sel = [r for r in rows if r[0] >= st][:lim]
            return 200, hdr, json.dumps([[r[0], r[1], r[2], r[3], r[4], 0, r[0] + H - 1, r[5]] for r in sel])
        if u.path.endswith("/exchangeInfo"):
            return 200, hdr, json.dumps(self.info)
        if u.path.endswith("/ticker/24hr"):
            return 200, hdr, json.dumps(self.tick)
        return 404, hdr, ""


@pytest.fixture(autouse=True)
def _iso(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(S, "PACE_S", 0)
    cd = tmp_path / "cooldown"
    monkeypatch.setattr(S.BC, "SHARED", str(cd))
    yield


def _net(http, tmp_path, **kw):
    return S.Net(http_get=http, cooldown_path=str(tmp_path / "cooldown"), pace_s=0, **kw)


# ─────────────────────────── state machine ───────────────────────────
def test_on_needs_all_three_legs():
    D = _frame(80)
    _set(D, 60, slope3=-0.1, pdi=15.0, ndi=30.0, r6=-0.9)            # r6 > −1 → no
    _set(D, 61, slope3=0.0, pdi=15.0, ndi=30.0, r6=-1.5)             # slope = 0 → no (strict < 0)
    _set(D, 62, slope3=-0.1, pdi=30.0, ndi=30.0, r6=-1.5)            # −DI = +DI → no
    assert S.v2_windows(D) == []
    _bear(D, 63)
    W = S.v2_windows(D)
    assert len(W) == 1 and W[0]["on_i"] == 63 and W[0]["on_ms"] == T_BASE + 64 * H


def test_warmup_bars_never_fire():
    D = _frame(80)
    _bear(D, S.WARM_BARS - 1)
    assert S.v2_windows(D) == []


def test_off_on_slope_and_fragment_merge():
    D = _frame(80)
    for i in range(60, 70):
        _bear(D, i)
    _set(D, 64, slope3=0.0)                                          # OFF at 64 (slope ≥ 0), re-ON at 65 (a later hour) …
    W = S.v2_windows(D)
    assert len(W) == 1                                               # … ≤ 180 min after the OFF → the SAME window (study merge)
    assert W[0]["on_i"] == 60 and W[0]["fragments"] == 2
    assert W[0]["x_off_i"] == 64 and W[0]["off_i"] == 70            # basket exit d = first OFF; window OFF = last OFF (bar 70 slope +)
    D2 = _frame(70)
    for i in range(60, 70):
        _bear(D2, i)
    w2 = S.v2_windows(D2)[0]
    assert w2["off_i"] is None and w2["off_ms"] is None and w2["x_off_i"] is None   # still ON at the last bar


def test_reon_after_merge_gap_is_a_new_window():
    D = _frame(90)
    for i in range(60, 64):
        _bear(D, i)                                                  # OFF at 64 (slope +0.1 default)
    for i in range(68, 72):
        _bear(D, i)                                                  # re-ON at 68: 4 h after the OFF > 180 min → new window
    W = S.v2_windows(D)
    assert [w["on_i"] for w in W] == [60, 68] and W[0]["off_i"] == 64 and W[0]["fragments"] == 1
    D = _frame(90)
    for i in range(60, 64):
        _bear(D, i)
    for i in range(67, 72):
        _bear(D, i)                                                  # re-ON at 67: exactly 180 min after → merged
    assert [w["on_i"] for w in S.v2_windows(D)] == [60]


def test_off_on_two_consecutive_pdi_bars():
    D = _frame(80)
    _bear(D, 60)
    for i in range(61, 80):
        _set(D, i, slope3=-0.1, pdi=15.0, ndi=30.0, r6=0.0)
    _set(D, 65, pdi=31.0)                                            # one +DI bar → still ON
    _set(D, 70, pdi=31.0); _set(D, 71, pdi=31.0)                     # two consecutive → OFF at 71
    W = S.v2_windows(D)
    assert W[0]["off_i"] == 71 and "+DI" in W[0]["off_reason"]


def test_off_on_bounce_high_vs_earlier_low_and_close():
    D = _frame(80)
    _bear(D, 60)
    for i in range(61, 80):
        _set(D, i, slope3=-0.1, pdi=15.0, ndi=30.0, r6=0.0, c=100.0, h=100.5, l=99.5)
    _set(D, 60, c=100.0)
    _set(D, 62, l=98.0, c=98.5, h=99.0)                              # running low 98
    _set(D, 63, h=99.6, l=98.6, c=99.0)                              # 99.6 < 98 × 1.015 = 99.47? no: 99.6 > 99.47 → OFF
    W = S.v2_windows(D)
    assert W[0]["off_i"] == 63 and "off the low" in W[0]["off_reason"]
    # a bar whose own low is the new low and whose high is +1.5 % above it does NOT trigger on the high (order unknown) …
    D2 = _frame(80)
    _bear(D2, 60)
    for i in range(61, 80):
        _set(D2, i, slope3=-0.1, pdi=15.0, ndi=30.0, r6=0.0, c=99.0, h=99.4, l=98.8)
    _set(D2, 62, l=97.0, h=99.4, c=97.5)                             # high 99.4 vs earlier low 98.8 × 1.015 = 100.28 → no; close 97.5 vs 97 → no
    W2 = S.v2_windows(D2)
    assert W2[0]["off_i"] != 62
    # … but its CLOSE ≥ +1.5 % above the low including itself does
    _set(D2, 62, l=97.0, h=99.4, c=98.6)                             # 98.6 > 97 × 1.015 = 98.455 → OFF
    assert S.v2_windows(D2)[0]["off_i"] == 62


# ─────────────────────────── classes ───────────────────────────
def _cls_frame(path_c, path_h=None, path_l=None, pre_high=101.0):
    n = 60 + len(path_c)
    D = _frame(n)
    D.loc[30:59, "h"] = 100.2
    D.loc[45, "h"] = pre_high
    for k, c in enumerate(path_c):
        i = 60 + k
        D.loc[i, "c"] = c
        D.loc[i, "h"] = (path_h[k] if path_h else c * 1.001)
        D.loc[i, "l"] = (path_l[k] if path_l else c * 0.999)
    D.loc[59, "c"] = 100.0
    return D


def test_class_clean_slide():
    D = _cls_frame([99.5, 99.0, 98.5, 98.0, 97.5, 96.8] + [96.8] * 20)
    assert S.classify(D, 59)[:2] == ("clean_slide", "")


def test_class_false_start_up_first():
    D = _cls_frame([100.5, 101.5, 102.5] + [97.0] * 23)            # above the 24 h high (101) first
    cls, sub, final = S.classify(D, 59)
    assert (cls, sub, final) == ("false_start", "F1", True)


def test_class_false_unresolved_and_pending():
    D = _cls_frame([99.8] * 26)
    assert S.classify(D, 59) == ("false_start", "F1_unresolved", True)
    D = _cls_frame([99.8] * 10)                                     # not final yet
    assert S.classify(D, 59) == ("pending", "", False)


def test_class_true_other():
    # bounce 1.5 % off the low (kills the clean slide) but no new high / +2 %, then −3 % from the 24 h high
    D = _cls_frame([99.0, 98.0, 99.6, 99.0, 97.5] + [97.5] * 21, pre_high=100.5)
    assert S.classify(D, 59)[0] == "true_other"


# ─────────────────────────── basket walks ───────────────────────────
def _bars(cl, hi=None, lo=None):
    return [[i, (cl[i - 1] if i else c), (hi[i] if hi else max(c, cl[i - 1] if i else c)), (lo[i] if lo else min(c, cl[i - 1] if i else c)),
             c, 1.0] for i, c in enumerate(cl)]


def test_walk_d_stop_exit_cap():
    b = _bars([100, 99, 98, 97], hi=[100, 100, 102.1, 97])
    assert S.walk_d(b, 0, 3) == (pytest.approx(-2.0 - S.FEE_RT), "STOP")
    b = _bars([100, 99, 98, 97])
    p, r = S.walk_d(b, 0, 3)
    assert r == "EXIT" and p == pytest.approx(3.0 - S.FEE_RT)
    b = _bars([100, 101.5, 101.9])                                    # loss < 2 % → exits at the close, no stop
    assert S.walk_d(b, 0, 2)[0] == pytest.approx(-1.9 - S.FEE_RT)


def test_walk_a_tp_stop_both_and_cap():
    assert S.walk_a(_bars([100, 99.4]), 0, 1) == (pytest.approx(0.5 - S.FEE_RT), "TP")
    assert S.walk_a(_bars([100, 101.2]), 0, 1) == (pytest.approx(-1.0 - S.FEE_RT), "STOP")
    b = _bars([100, 100], hi=[100, 101.1], lo=[100, 99.0])
    assert S.walk_a(b, 0, 1)[1] == "STOP"                             # both in one bar → stop first
    p, r = S.walk_a(_bars([100] + [99.8] * 8), 0, 8)
    assert r == "CAP" and p == pytest.approx(0.2 - S.FEE_RT)


# ─────────────────────────── price_window (fake net) ───────────────────────────
def _coin_series(on_ms, n_coins=45, drop=0.0, qv_base=1e6):
    """coins with bars from ON − 25 h to ON + 24 h; coin k has quote vol (k+1)·base; half below their 24 h-earlier price."""
    out = {}
    start = on_ms - 25 * H
    for k in range(n_coins):
        rows = []
        for j in range(49):
            if j <= 24:
                c = 100.0 - (1.0 if (k % 2 == 0 and j == 24) else 0.0)
            else:
                c = 100.0 * (1 - drop * (j - 24) / 24)
            rows.append([start + j * H, c, c * 1.001, c * 0.999, c, (k + 1) * qv_base])
        out[f"C{k}USDT"] = rows
    return out


def test_price_window_basket_c2_and_cache(tmp_path):
    on_ms = T_BASE + 100 * H
    ser = _coin_series(on_ms, n_coins=45, drop=0.03)
    http = FakeHTTP(series=ser)
    net = _net(http, tmp_path)
    w = dict(on_i=60, on_ms=on_ms, off_i=70, off_ms=on_ms + 10 * H, x_off_i=70, x_off_ms=on_ms + 10 * H)
    now = on_ms + 30 * H
    P = S.price_window(w, None, net, now, list(ser), final=True)
    assert P["basket"] == [f"C{k}USDT" for k in range(44, 24, -1)]  # top-20 by quote vol at ON
    assert P["c2_pct"] is not None and P["c2"] is True and P["c2_pct"] >= 50   # 23 of 45 even coins below
    # exit d = OFF close (10 h → −1.25 % move → +1.25 − 0.10 per coin)
    x = 100.0 * (1 - 0.03 * 10 / 24)                                 # every coin's close at the OFF hour
    exp = np.mean([(1 - x / (99.0 if k % 2 == 0 else 100.0)) * 100 for k in range(44, 24, -1)]) - S.FEE_RT
    assert P["d"]["n"] == 20 and P["d"]["mean"] == pytest.approx(exp, abs=1e-3)
    assert P["priced_full"] is True
    n_calls = len(http.calls)
    S.price_window(w, None, net, now, list(ser), final=True)
    assert len(http.calls) == n_calls                                 # final windows come from the cache


def test_c2_na_when_few_coins(tmp_path):
    on_ms = T_BASE + 100 * H
    ser = _coin_series(on_ms, n_coins=30)
    P = S.price_window(dict(on_i=60, on_ms=on_ms, off_i=None), None, _net(FakeHTTP(series=ser), tmp_path), on_ms + 30 * H,
                       list(ser), final=True)
    assert P["c2_pct"] is None and P["c2"] is None
    assert P["d"]["n"] == 20                                          # still ON at +24 h → the 24 h cap


# ─────────────────────────── budget / cooldown ───────────────────────────
def test_cooldown_active_blocks_without_calling(tmp_path):
    (tmp_path / "cooldown").write_text(str(T_BASE + 10 * H))
    http = FakeHTTP()
    net = _net(http, tmp_path)
    assert net.json("/fapi/v1/klines", {"symbol": "X", "startTime": 0, "limit": 1}, T_BASE) is None
    assert net.stop and "cooldown" in net.why and http.calls == []


@pytest.mark.parametrize("code", [418, 429])
def test_rate_limit_arms_cooldown_and_stops(tmp_path, code):
    http = FakeHTTP(codes=[code])
    net = _net(http, tmp_path)
    assert net.json("/fapi/v1/klines", {"symbol": "X", "startTime": 0, "limit": 1}, T_BASE) is None
    assert net.stop and str(code) in net.why
    assert S.BC.read_until_ms(str(tmp_path / "cooldown")) == T_BASE + 60_000
    assert net.json("/fapi/v1/klines", {"symbol": "X", "startTime": 0, "limit": 1}, T_BASE) is None
    assert len(http.calls) == 1


def test_403_and_weight_budget_stop(tmp_path):
    net = _net(FakeHTTP(codes=[403]), tmp_path)
    net.json("/fapi/v1/exchangeInfo", None, T_BASE)
    assert net.stop and "403" in net.why
    http = FakeHTTP(series={"X": [[0, 1, 1, 1, 1, 1]]}, weight=950)
    net = _net(http, tmp_path)
    assert net.json("/fapi/v1/klines", {"symbol": "X", "startTime": 0, "limit": 1}, T_BASE) is not None
    assert net.used == 950
    assert net.json("/fapi/v1/klines", {"symbol": "X", "startTime": 0, "limit": 1}, T_BASE) is None
    assert net.stop and "weight" in net.why and len(http.calls) == 1


# ─────────────────────────── persistence + frozen read ───────────────────────────
def _w(day, d_mean, cls="clean_slide", hour=3):
    on = int(pd.Timestamp(f"2026-11-{day:02d}T{hour:02d}:00:00Z").value // 1_000_000)
    return dict(on_ms=on, on_utc="", off_ms=None, off_utc="—", off_reason="", btc_on=1.0, btc_low=1.0, drop_pct=0.0, cls=cls,
                cls_sub="", r6=-1.0, slope3=-0.1, di_gap=1.0, c2_pct=None, c2=None, d_n=20, d_mean=d_mean, d_wr=50.0,
                d_stops=0, a_n=20, a_mean=0.0, a_wr=50.0, a_stops=0)


def test_append_only(tmp_path):
    st = S.load_state(str(tmp_path / "s.json"))
    assert S.append_windows(st, [_w(1, 1.0)]) == 1
    assert S.append_windows(st, [_w(1, -9.0), _w(2, 2.0)]) == 1       # the same ON is never rewritten
    assert [w["d_mean"] for w in st["windows"]] == [1.0, 2.0]
    assert S.save_state(st, str(tmp_path / "s.json"))
    assert len(S.load_state(str(tmp_path / "s.json"))["windows"]) == 2


def _wk(k, d_mean, cls="clean_slide", per_week=3):
    """window k placed so that windows spread over ISO weeks (per_week windows per week, distinct days)."""
    base = pd.Timestamp("2026-11-02T03:00:00Z")                       # a Monday
    t = base + pd.Timedelta(weeks=k // per_week, days=k % per_week)
    w = _w(1, d_mean, cls)
    w["on_ms"] = int(t.value // 1_000_000)
    return w


def test_interim_checkpoint_is_non_terminal():
    assert "not reached" in S.interim_read([_w(d, 1.0) for d in range(1, 8)])
    txt = S.interim_read([_w(d, -1.0, "false_start") for d in range(1, 9)])
    assert "NON-TERMINAL" in txt and "not on track" in txt and "stays off" not in txt.lower()
    st = S.load_state("/nonexistent/x.json")
    assert "interim" not in st and st["arm_read"] is None             # never frozen


def test_arm_read_collecting_on_weeks():
    W = [_wk(k, 1.0, per_week=7) for k in range(25)]                 # 25 windows but only 4 ISO weeks
    assert S.arm_read(W)[0] == "COLLECTING"


def _arm_set(fs_each=-0.2):
    W = [_wk(k, 2.0 + 0.01 * k, "clean_slide") for k in range(12)] + [_wk(k, fs_each, "false_start") for k in range(12, 16)] + \
        [_wk(k, 0.1, "true_other") for k in range(16, 20)]
    return W


def test_arm_read_wf_condition_blocks_today(monkeypatch):
    v, txt, n = S.arm_read(_arm_set())
    assert v == "NO_ARM" and n == 20 and "⑤" in txt and "②" not in txt.split("failed:")[1]
    monkeypatch.setattr(S, "WF_HELDOUT_SUM", 5.0)
    assert S.arm_read(_arm_set())[0] == "ARM_CANDIDATE"


def test_arm_read_fs_ratio_and_concentration(monkeypatch):
    monkeypatch.setattr(S, "WF_HELDOUT_SUM", 5.0)
    W = _arm_set(fs_each=-4.0)                                       # FS lose 16 vs clean earn ≈ 24.7 → 65 % > 60 %
    v, txt, _ = S.arm_read(W)
    assert v == "NO_ARM" and "④" in txt
    m = S.summary(W[:20])
    assert m["fs_ratio"] == pytest.approx(16.0 / sum(2.0 + 0.01 * k for k in range(12)))
    W = [_wk(k, -0.1, "clean_slide") for k in range(19)] + [_wk(19, 30.0, "clean_slide")]
    v, txt, _ = S.arm_read(W)
    assert v == "NO_ARM" and "③" in txt


def test_arm_read_shortest_prefix_and_pending(monkeypatch):
    monkeypatch.setattr(S, "WF_HELDOUT_SUM", 5.0)
    W = _arm_set() + [_wk(25, -500.0, "false_start")]                 # the 21st window never counts
    assert S.arm_read(W)[0] == "ARM_CANDIDATE"
    W20 = _arm_set()
    pend = [W20[10]["on_ms"] + H]                                  # a cohort window inside the prefix is not final yet
    assert S.arm_read(W20, pend)[0] == "WAITING"
    assert S.arm_read(W20, [W20[-1]["on_ms"] + 100 * H])[0] == "ARM_CANDIDATE"   # later pending windows do not block


def test_drop_read():
    W = [_wk(k, 1.0, "clean_slide") for k in range(10)] + [_wk(k, -1.5, "false_start") for k in range(10, 30)]
    v, txt, n = S.drop_read(W)
    assert v == "DROP_SHADOW" and n == 30
    W = [_wk(k, 1.0, "clean_slide") for k in range(30)]
    assert S.drop_read(W)[0] == "KEEP_SHADOW"
    assert S.drop_read(W[:29])[0] == "COLLECTING"


def test_decided_once(tmp_path, monkeypatch):
    st = S.load_state(str(tmp_path / "x.json"))
    v, _, now = S.decide_once(st, "arm_read", S.arm_read, _arm_set(), T_BASE)
    assert v == "NO_ARM" and now
    monkeypatch.setattr(S, "WF_HELDOUT_SUM", 5.0)
    v2, txt, now2 = S.decide_once(st, "arm_read", S.arm_read, _arm_set(), T_BASE)
    assert v2 == "NO_ARM" and not now2 and "frozen" in txt
    v3, _, now3 = S.decide_once(st, "drop_read", S.drop_read, _arm_set(), T_BASE)
    assert v3 == "COLLECTING" and not now3 and st["drop_read"] is None


# ─────────────────────────── run(): finality, persistence, not-evaluable, T0 ───────────────────────────
def _scenario():
    """BTC flat 100 bars, then −0.5 %/h for 14 h (V2 ON), then flat-low: a clean slide; coins drop 3 % over 24 h."""
    closes = [100.0 + 0.01 * (i % 3) for i in range(100)]
    for k in range(14):
        closes.append(closes[-1] * 0.995)
    closes += [closes[-1]] * 40
    rows = _btc_rows(closes)
    D = S.btc_frame(rows)
    W = S.v2_windows(D)
    assert W, "synthetic decline must trigger V2"
    on_ms = W[0]["on_ms"]
    return rows, D, W, on_ms


def test_run_not_evaluable_prints_context_and_persists_nothing(tmp_path):
    rows, D, W, on_ms = _scenario()
    ser = _coin_series(on_ms, n_coins=45, drop=0.03)
    st = str(tmp_path / "state.json")
    now = int(D.t.values[-1]) + 60_000
    L = S.run(now, net=_net(FakeHTTP(series=ser), tmp_path), paths=[], state_path=st, deploy_ms=None, btc_rows=rows, syms=list(ser))
    txt = "\n".join(L)
    assert "NOT evaluable" in txt and "Context" in txt and "ON " in txt
    assert not os.path.exists(st)


def test_run_finality_and_t0(tmp_path):
    rows, D, W, on_ms = _scenario()
    ser = _coin_series(on_ms, n_coins=45, drop=0.03)
    st = str(tmp_path / "state.json")
    exp = tmp_path / "orders.csv"
    pd.DataFrame([dict(opened_at=pd.Timestamp(on_ms + 5 * H, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"), closed_at="", pair="SOLUSDT",
                       direction="SHORT", status="CLOSED", entry_strategy="BEARRUN_SHORT", pnl=-20.0, pnl_percentage=-0.8,
                       close_reason="STOP_LOSS", entry_bear_bypass="BTC_1H_RSI_MIN_GATE", is_paper=False),
                  dict(opened_at=pd.Timestamp(on_ms + 6 * H, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"), closed_at="", pair="XUSDT",
                       direction="SHORT", status="CLOSED", entry_strategy="MOMENTUM", pnl=1, pnl_percentage=0.1, close_reason="TP",
                       entry_bear_bypass="", is_paper=False),                     # blank bypass (NaN in the CSV) → never a Bear-Run fill
                  dict(opened_at=pd.Timestamp(on_ms + 7 * H, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"), closed_at="", pair="YUSDT",
                       direction="SHORT", status="CLOSED", entry_strategy="MANUAL", pnl=1, pnl_percentage=0.1, close_reason="APP_CLOSE",
                       entry_bear_bypass="", is_paper=False)]
                 ).to_csv(exp, index=False)
    # 1) before ON + 24 h: shown as open, nothing persisted
    rows_early = rows[: W[0]["on_i"] + 10]
    now = rows_early[-1][0] + H + 60_000
    L = S.run(now, net=_net(FakeHTTP(series=ser), tmp_path), paths=[str(exp)], state_path=st, deploy_ms=on_ms - H,
              btc_rows=rows_early, syms=list(ser))
    assert "Open / not yet final" in "\n".join(L)
    assert not os.path.exists(st) or S.load_state(st)["windows"] == []
    # 2) after ON + 24 h: persisted once, T0 fill listed inside the window
    now = rows[-1][0] + H + 60_000
    L = S.run(now, net=_net(FakeHTTP(series=ser), tmp_path), paths=[str(exp)], state_path=st, deploy_ms=on_ms - H, btc_rows=rows,
              syms=list(ser))
    txt = "\n".join(L)
    s1 = S.load_state(st)
    assert len(s1["windows"]) == 1 and s1["windows"][0]["cls"] == "clean_slide"
    assert s1["windows"][0]["d_n"] == 20 and s1["windows"][0]["c2_pct"] is not None
    assert s1["windows"][0]["universe_at"] is not None                # the universe snapshot taken at the first pricing run
    assert os.path.exists(str(tmp_path / "state.json") + ".lock")      # load / append / save ran under the flock
    assert "SOLUSDT" in txt and "inside" in txt and "-0.80 %" in txt and "XUSDT" not in txt and "YUSDT" not in txt
    assert "COLLECTING" in txt and "interim checkpoint" in txt and "FS-loss ratio" in txt and "ISO weeks" in txt
    # 3) rerun: nothing re-appended, record unchanged
    rec = json.dumps(s1["windows"][0], sort_keys=True)
    S.run(now + H, net=_net(FakeHTTP(series=ser), tmp_path), paths=[str(exp)], state_path=st, deploy_ms=on_ms - H, btc_rows=rows,
          syms=list(ser))
    s2 = S.load_state(st)
    assert len(s2["windows"]) == 1 and json.dumps(s2["windows"][0], sort_keys=True) == rec


def test_run_budget_stop_does_not_persist(tmp_path):
    rows, D, W, on_ms = _scenario()
    ser = _coin_series(on_ms, n_coins=45, drop=0.03)
    st = str(tmp_path / "state.json")
    now = rows[-1][0] + H + 60_000
    http = FakeHTTP(series=ser, codes=[200] * 5 + [429])
    L = S.run(now, net=_net(http, tmp_path), paths=[], state_path=st, deploy_ms=on_ms - H, btc_rows=rows, syms=list(ser))
    assert "429" in "\n".join(L)
    assert not os.path.exists(st) or S.load_state(st)["windows"] == []
    assert len(http.calls) == 6


def test_btc_unreadable_path(tmp_path):
    (tmp_path / "cooldown").write_text(str(T_BASE + 1000 * H))
    L = S.run(T_BASE, net=_net(FakeHTTP(), tmp_path), paths=[], state_path=str(tmp_path / "s.json"), deploy_ms=None)
    assert "unreadable" in "\n".join(L)


def test_wilder_copy_matches_study():
    sp = os.path.join(ROOT, "scripts", "study_bearrun_early_trigger.py")
    if not os.path.exists(sp):
        pytest.skip("study file not present")
    import study_bearrun_early_trigger as ST
    rng = np.random.default_rng(3)
    c = pd.Series(100 + np.cumsum(rng.normal(0, 0.5, 300)))
    h, l = c + rng.uniform(0, 0.5, 300), c - rng.uniform(0, 0.5, 300)
    for a, b in zip(S.wilder(h, l, c), ST.wilder(h, l, c)):
        assert np.allclose(a.values, b.values, equal_nan=True)


def test_walk_d_gap_stop_books_the_open():
    b = _bars([100, 99, 103], hi=[100, 99.5, 104], lo=[100, 98.5, 102.5])
    b[2][1] = 103.0                                                   # bar 2 OPENS at 103 (> the 102 stop) → filled at 103
    p, r = S.walk_d(b, 0, 2)
    assert r == "STOP_GAP" and p == pytest.approx(-3.0 - S.FEE_RT)


def test_per_symbol_4xx_drops_coin_not_unread(tmp_path):
    on_ms = T_BASE + 100 * H
    ser = _coin_series(on_ms, n_coins=45, drop=0.03)

    class Bad(FakeHTTP):
        def __call__(self, url):
            if "symbol=C44USDT" in url:
                self.calls.append(url)
                return 400, {"X-MBX-USED-WEIGHT-1M": "5"}, '{"code":-1121,"msg":"Invalid symbol."}'
            return super().__call__(url)
    net = _net(Bad(series=ser), tmp_path)
    w = dict(on_i=60, on_ms=on_ms, off_i=70, off_ms=on_ms + 10 * H, x_off_i=70, x_off_ms=on_ms + 10 * H)
    P = S.price_window(w, None, net, on_ms + 30 * H, list(ser), final=True)
    assert P["dropped_4xx"] == ["C44USDT"] and P["coins_unread"] == 0 and not net.stop
    assert P["priced_full"] is True and "C44USDT" not in P["basket"] and len(P["basket"]) == 20
    # a 5xx is NOT a drop: the coin stays unread → the window is not final (retry next run)
    class Down(FakeHTTP):
        def __call__(self, url):
            if "symbol=C44USDT" in url:
                return 503, {}, ""
            return super().__call__(url)
    P = S.price_window(w, None, _net(Down(series=ser), tmp_path), on_ms + 30 * H, list(ser), final=True)
    assert P["coins_unread"] == 1 and P["priced_full"] is False


def test_low_capped_at_24h():
    D = _frame(100)
    D.loc[60, "c"] = 100.0
    D.loc[61:84, "l"] = 99.0
    D.loc[90, "l"] = 90.0                                            # after ON + 24 h → not the window's low
    w = dict(on_i=60, on_ms=0, off_i=95, off_ms=1, off_reason="x", x_off_i=95, r6=-1.0, slope3=-0.1, di_gap=1.0)
    P = dict(d=S._stats([]), a=S._stats([]))
    rec = S.build_record(w, D, P, "true_other", "")
    assert rec["btc_low"] == 99.0 and rec["drop_pct"] == pytest.approx(-1.0)


def test_persisted_reads_wait_for_pending_and_do_not_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "WF_HELDOUT_SUM", 5.0)
    st = S.load_state(str(tmp_path / "s.json"))
    S.append_windows(st, _arm_set())
    early = st["windows"][5]["on_ms"] + H
    L = S._persisted_lines(st, 0, T_BASE, None, str(tmp_path / "s.json"), pending={early})
    assert any("WAITING" in x for x in L) and st["arm_read"] is None
    L = S._persisted_lines(st, 0, T_BASE, None, str(tmp_path / "s.json"), pending=set())
    assert st["arm_read"]["verdict"] == "ARM_CANDIDATE"
    assert S.load_state(str(tmp_path / "s.json"))["arm_read"]["verdict"] == "ARM_CANDIDATE"
    assert "_dirty" not in json.load(open(tmp_path / "s.json"))


def test_scout_wiring_present():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    assert "import scout_bearrun_v2" in src
