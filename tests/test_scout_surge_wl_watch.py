"""👁 Nov-1: SURGE_WL watch lines (DECISION_LOG 427) and the exit study's missing-archive fallback. Hermetic: temp exports and state, no network, no git."""
import io
import os
import sys
import urllib.error

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_surge_wl_watch as W  # noqa: E402

T0 = pd.Timestamp("2026-11-02 00:00")
NOW, SINCE = int((T0 + pd.Timedelta(days=60)).value // 10**6), int(T0.value // 10**6)


def _fills(spec):
    """spec = [(window index, pair, rsi, volume, pct)]."""
    rows = []
    for i, (w, pair, rsi, vol, pct) in enumerate(spec):
        o = T0 + pd.Timedelta(days=w, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=9)).strftime("%Y-%m-%dT%H:%M:%S"), pair=pair, entry_strategy="SURGE_LONG", status="CLOSED",
                         pnl_percentage=pct, close_reason="BR_STOP_LOSS" if pct < 0 else "BR_TRAILING_STOP", entry_surge_trigger_at=f"T{w}", entry_btc_rsi=rsi, entry_pair_volume_24h_usd=vol))
    return pd.DataFrame(rows)


def _write(tmp_path, df, stamp="2027-02-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def _z(pcts, rest=2.0):
    return [(i % 8, f"Z{i}USDT", 85.0, 500e6, p) for i, p in enumerate(pcts)] + [(i % 8, f"R{i}USDT", 70.0, 500e6, rest) for i in range(10)]


def _groups(tmp_path, spec):
    c, _ = W.load(_write(tmp_path, _fills(spec)))
    c = c.assign(zone=W.zone_of(c, "entry_btc_rsi", "ge", W.RSI_CUT))
    return c[c.zone == "zone"], c[c.zone == "rest"]


def test_frozen_constants():
    assert (W.RSI_CUT, W.VOL_CUT, W.N_MIN, W.WINDOWS_MIN, W.BE_WR, W.DL) == (78.0, 150e6, 15, 8, 0.315, 427)


def test_zone_edges_and_unscored():
    d = pd.DataFrame(dict(a=[78.0, 77.99, None], v=[150e6, 149.9e6, None]))
    assert W.zone_of(d, "a", "ge", 78.0).tolist() == ["zone", "rest", "unscored"]
    assert W.zone_of(d, "v", "lt", 150e6).tolist() == ["rest", "zone", "unscored"]


def test_verdict_outcomes(tmp_path):
    assert W.verdict(*_groups(tmp_path, _z([-1.2] * 14)))[0] == "COLLECTING"
    assert W.verdict(*_groups(tmp_path, [(0, f"Z{i}USDT", 85.0, 500e6, -1.2) for i in range(15)]))[0] == "COLLECTING"       # 15 fills, one window
    assert W.verdict(*_groups(tmp_path, _z([-1.2] * 16)))[0] == W.CANDIDATE
    assert W.verdict(*_groups(tmp_path, _z([3.0] * 16)))[0] == "RETIRED"
    assert W.verdict(*_groups(tmp_path, _z([-1.2] * 16, rest=-2.0)))[0] == "RETIRED"
    assert W.verdict(*_groups(tmp_path, _z([1.5] * 6 + [-2.0] * 10)))[0] == "KEEP OBSERVING"                                   # 38 % wins: above breakeven


def test_forward_only_operator_closed_out_and_each_line_apart(tmp_path):
    old = _fills([(-5, "OLDUSDT", 85.0, 50e6, -1.2)])
    op = _fills([(1, "OPUSDT", 85.0, 50e6, -1.2)]).assign(close_reason="MANUAL_CLOSE")
    hot = _fills([(2, "HOTUSDT", 85.0, 500e6, -1.2)])
    thin = _fills([(3, "THINUSDT", 70.0, 50e6, -1.2)])
    none = _fills([(4, "NAUSDT", None, None, 2.0)])
    out = "\n".join(W.run(NOW, _write(tmp_path, pd.concat([old, op, hot, thin, none])), None, SINCE))
    a, b = out.split("### SURGE_THIN")
    assert "| **zone** | 1 | 1 | 0% | -1.200 %" in a and "| rest | 1 |" in a and "UNSCORED (stamp missing) — never counted | 1 |" in a
    assert "| **zone** | 1 | 1 | 0% | -1.200 %" in b 
    assert "| rest | 1 | 1 | 0% |" in b                                                                               # HOT is rest on the volume line, NA stays unscored


def test_freezes_once_and_defers_on_a_young_export(tmp_path):
    state = str(tmp_path / "s.json")
    f = _fills(_z([-1.2] * 16))
    young = "\n".join(W.run(NOW, _write(tmp_path, f, stamp="2026-11-05_00-00-00"), state, SINCE))
    assert "freezing deferred" in young and not os.path.exists(state)
    out = "\n".join(W.run(NOW, _write(tmp_path, f), state, SINCE))
    assert f"**Frozen verdict" in out and W.CANDIDATE in out.split("Evidence:")[0]
    assert os.path.exists(state) and len(__import__("json").load(open(state))["SURGE_BTC_HOT"]["keys"]) >= 16
    better = _fills(_z([-1.2] * 16) + [(j, f"X{j}USDT", 85.0, 500e6, 9.0) for j in range(8)])
    again = "\n".join(W.run(NOW + 1, _write(tmp_path, better, stamp="2027-02-02_00-00-00"), state, SINCE))
    assert W.CANDIDATE in again.split("Live (information")[0] and "Live (information only, never re-decides): RETIRED" in again


def test_hook_registry_gitignore():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_surge_wl_watch as _swl" in src and "_rg_sec += _swl.run(now_ms)" in src
    assert '"scout_surge_wl_watch.py": {"surge_long_exit_no_lock": True' in open(os.path.join(ROOT, "scripts", "observe_registry.py"), encoding="utf-8").read()
    assert "reports/SCOUT_SURGE_WL.json" in open(os.path.join(ROOT, ".gitignore")).read()


# ── the exit study: a day missing from Binance's archive falls back to 1m bars; a fill that cannot be priced is listed


class _Resp(io.BytesIO):
    headers = {"X-MBX-USED-WEIGHT-1M": "1"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _study(tmp_path, monkeypatch, klines_ok=True):
    import json
    import study_surge_exit_grid as G
    monkeypatch.setattr(G, "AGG_REAL", str(tmp_path))
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    G.PATH_SRC.clear()
    t0 = int(pd.Timestamp("2026-10-28 08:30:27").value // 10**6)

    def fake(url, timeout=0):
        if "data.binance.vision" in str(url):
            raise urllib.error.HTTPError(str(url), 404, "Not Found", None, None)
        if not klines_ok:
            raise urllib.error.URLError("down")
        s = int(str(url).split("startTime=")[1].split("&")[0])
        return _Resp(json.dumps([[s + 60000 * i, "100", "102", "99", "101"] for i in range(3)] if s < t0 else []).encode())
    monkeypatch.setattr(G.urllib.request, "urlopen", fake)
    return G, t0


def test_missing_archive_day_falls_back_to_1m_and_is_not_cached(tmp_path, monkeypatch):
    G, t0 = _study(tmp_path, monkeypatch)
    r = G.fetch_agg_public("HAEDALUSDT", t0, t0 + 240 * 60000)
    assert r is not None and G.PATH_SRC[("HAEDALUSDT", t0)] == "1m" and not os.path.exists(tmp_path / f"HAEDALUSDT_{t0}.csv")
    ts, px = r
    assert px[0] == 100.0 and px[20] == 99.0 and px[40] == 102.0 and px[60] == 101.0 and (ts[1:61] > ts[:60]).all()   # green bar: open, low, high, close
    assert abs(px[1] - px[0]) < 0.1                                                                                   # a stop fills near its line, never at the bar's low


def test_only_the_missing_day_falls_back(tmp_path, monkeypatch):
    G, t0 = _study(tmp_path, monkeypatch)
    late = int(pd.Timestamp("2026-10-27 22:00:00").value // 10**6)     # 4 h hold crosses into the 28th; the 27th's archive is on disk
    day = pd.DataFrame([[1, 2.0, 1, 1, 1, late + 1000, 0]])
    day.to_csv(tmp_path / "BTWUSDT-aggTrades-2026-10-27.zip.csv", index=False, header=False)
    import zipfile
    with zipfile.ZipFile(tmp_path / "BTWUSDT-aggTrades-2026-10-27.zip", "w") as z:
        z.write(tmp_path / "BTWUSDT-aggTrades-2026-10-27.zip.csv", "x.csv")
    assert G.fetch_agg_public("BTWUSDT", late, late + 240 * 60000) is not None
    assert G.PATH_SRC[("BTWUSDT", late)] == "aggTrades+1m" and not os.path.exists(tmp_path / f"BTWUSDT_{late}.csv")


def test_no_klines_either_returns_none(tmp_path, monkeypatch):
    G, t0 = _study(tmp_path, monkeypatch, klines_ok=False)
    assert G.fetch_agg_public("HAEDALUSDT", t0, t0 + 240 * 60000) is None


def test_report_lists_skipped_fills():
    src = open(os.path.join(ROOT, "scripts", "study_surge_exit_grid.py"), encoding="utf-8").read()
    assert "SKIPPED.append(" in src and "fills could NOT be priced" in src
