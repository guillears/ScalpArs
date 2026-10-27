"""🩸🌙 OBS-F1 late-tape weak spike (DECISION_LOG 384): the late-tape gap on the last CLOSED daily bar (≥ 400 bars), the frozen zone /
control / early / unscored split (+DI ≤ 48.5 ∧ Ext% ≤ 1.80, boundaries inclusive), the read (COLLECTING → RETIRE / FILTER_CANDIDATE /
KEEP_OBSERVING at N ≥ 15 on ≥ 8 UTC days, HOLD on an open fade), decide-once persistence, the not-evaluable branch, and no fetch when
bars are injected. Synthetic data — no network."""
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_fade_late_weak as S  # noqa: E402

DAY = S.DAY_MS
T0 = dt.datetime(2026, 10, 27, 1, 0)
SINCE = int(pd.Timestamp(T0).value // 1_000_000)


def _bars(n=600, rising=True, end=None):
    last = (SINCE + 89 * DAY) // DAY * DAY if end is None else int(end.value // 1_000_000)   # the day before the runs' now
    t0 = last - (n - 1) * DAY
    c = np.linspace(50_000, 100_000, n) if rising else np.linspace(100_000, 50_000, n)
    return [[t0 + i * DAY, 0, 0, 0, float(x)] for i, x in enumerate(c)]


def _row(i, y, pdi=40.0, ext=1.0, day=None, pair=None, status="CLOSED", reason=None):
    o = T0 + dt.timedelta(days=i if day is None else day, minutes=i)
    return dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), pair=pair or f"P{i}USDT", direction="SHORT", status=status,
                entry_strategy="SPIKE_FADE", pnl=y * 100.0, pnl_percentage=y, close_reason=reason or ("RUNNER_TRAIL" if y > 0 else "STOP_LOSS L1"),
                is_paper=True, entry_pos_di=pdi, entry_dist_from_ema13_pct=ext)


def _csv(tmp_path, rows):
    p = tmp_path / "o.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return [str(p)]


def _run(tmp_path, rows, bars=None, state=None):
    return S.run(SINCE + 90 * DAY, paths=_csv(tmp_path, rows), since_ms=SINCE, state_path=state or str(tmp_path / "st.json"),
                 bars=bars if bars is not None else _bars(), fetch=False)


def _verdict(L):
    """the verdict word of the Read line (its fixed header names every verdict, so never match the whole line)."""
    x = [l for l in L if l.lstrip().startswith("Read (")]
    assert len(x) == 1
    return x[0].split("): ", 1)[1].split(" —")[0]


def test_late_gap_uses_last_closed_bar_and_needs_400():
    b = _bars(n=500)
    k, _ = S.load_btc_1d(b[-1][0] + DAY, bars=b, fetch=False)
    last_open = pd.Timestamp(b[-1][0], unit="ms")
    g = S.late_gap(k, [last_open + pd.Timedelta(hours=23), last_open + pd.Timedelta(days=1)])
    c = k.c.values
    e = lambda n: pd.Series(c).ewm(span=n, adjust=False).mean().values
    assert np.isclose(g[0], (e(50)[-2] / e(200)[-2] - 1) * 100)          # the still-open last bar is not used
    assert np.isclose(g[1], (e(50)[-1] / e(200)[-1] - 1) * 100)
    short, _ = S.load_btc_1d(b[-1][0] + DAY, bars=_bars(n=399), fetch=False)
    assert np.isnan(S.late_gap(short, [last_open + pd.Timedelta(days=1)])[0])
    exact, _ = S.load_btc_1d(b[-1][0] + DAY, bars=_bars(n=400), fetch=False)
    assert np.isfinite(S.late_gap(exact, [last_open + pd.Timedelta(days=1)])[0])
    assert np.isnan(S.late_gap(k, [last_open + pd.Timedelta(days=3)])[0])      # stale bars → unscored, never an older bar
    assert np.isnan(S.late_gap(None, [last_open])[0])


def test_side_frozen_boundaries():
    df = pd.DataFrame(dict(entry_pos_di=[48.5, 48.51, 40, 40, None, 40], entry_dist_from_ema13_pct=[1.80, 1.0, 1.81, 1.0, 1.0, 1.0]))
    s = S.side(df, np.array([1.0, 1.0, 1.0, -0.1, 1.0, np.nan]))
    assert list(s) == ["zone", "control", "control", "early", "unscored", "nobar"]


def test_missing_bar_waits_and_never_persists(tmp_path):
    rows = [_row(i, -1.0 if i % 3 else 0.3, day=i // 2) for i in range(16)]
    st = str(tmp_path / "st.json")
    stale = _bars()[:-85]                                                     # cache ends ~day 4 of the fills → later fills have no bar
    L = S.run(SINCE + 90 * DAY, paths=_csv(tmp_path, rows), since_ms=SINCE, state_path=st, bars=stale, fetch=False)
    assert _verdict(L) == "COLLECTING" and not os.path.exists(st) and any("waiting" in x for x in L)


def test_collecting_then_filter_candidate_and_frozen(tmp_path):
    rows = [_row(i, -1.0 if i % 3 else 0.3, day=i // 2) for i in range(16)] + [_row(40, 0.6, pdi=60)]
    st = str(tmp_path / "st.json")
    L = _run(tmp_path, rows[:10], state=st)
    assert _verdict(L) == "COLLECTING" and not os.path.exists(st)
    L = _run(tmp_path, rows, state=st)
    assert _verdict(L) == "FILTER_CANDIDATE"
    assert json.load(open(st))[S.KEY]["verdict"] == "FILTER_CANDIDATE"
    L2 = _run(tmp_path, [_row(i, 1.0, day=i // 2) for i in range(16)], state=st)    # frozen: later data never re-decides
    assert _verdict(L2) == "FILTER_CANDIDATE" and any("frozen" in x for x in L2)
    L3 = S.run(SINCE + 90 * DAY, paths=_csv(tmp_path, rows), since_ms=SINCE, state_path=st, bars=[], fetch=False)
    assert _verdict(L3) == "FILTER_CANDIDATE"                                   # frozen verdict shown even without BTC bars


def test_retire_when_zone_pays(tmp_path):
    L = _run(tmp_path, [_row(i, 0.5 if i % 4 else -0.2, day=i // 2) for i in range(16)])
    assert _verdict(L) == "RETIRE"


def test_concentration_clause(tmp_path):
    # 16 zone fades on 8 days, WR 50 %, every loser −1.0 → passes WR / avg / P; only the day share differs
    def rows(spread):
        out = []
        for i in range(16):
            win = i % 2 == 0
            day = i // 2 if (spread or win) else 0                         # concentrated: every loser on day 0
            out.append(_row(i, 0.3 if win else -1.0, day=day))
        return out
    assert _verdict(_run(tmp_path, rows(True), state=str(tmp_path / "a.json"))) == "FILTER_CANDIDATE"
    assert _verdict(_run(tmp_path, rows(False), state=str(tmp_path / "b.json"))) == "KEEP_OBSERVING"


def test_breakeven_uses_every_kept_fade(tmp_path):
    zone = [_row(i, -1.0 if i % 3 else 0.3, day=i // 2) for i in range(16)]
    other = [_row(50 + i, 2.0, pdi=None, day=i // 2) for i in range(6)]       # unscored, still in the prefix breakeven
    g = pd.DataFrame(dict(o=pd.to_datetime([r["opened_at"] for r in zone + other]), pair=[r["pair"] for r in zone + other],
                          y=[r["pnl_percentage"] for r in zone + other], side=["zone"] * 16 + ["unscored"] * 6))
    v, txt = S.read(g)
    pre = g[g.o <= g[g.side == "zone"].o.iloc[14]]                          # the prefix ends at the 15th zone fill
    w, l = pre.y[pre.y > 0], pre.y[pre.y < 0]
    be = 100 * abs(l.mean()) / (w.mean() + abs(l.mean()))
    assert (pre.side == "unscored").sum() > 0 and be < 60                    # the unscored +2.0 wins pull the breakeven down
    assert "fallback" not in txt and f"breakeven {be:.0f} %" in txt and v == "FILTER_CANDIDATE"


def test_early_tape_never_counts(tmp_path):
    L = _run(tmp_path, [_row(i, -1.0, day=i // 2) for i in range(20)], bars=_bars(rising=False))
    assert any("PAUSED" in x for x in L) and _verdict(L) == "COLLECTING" and any("zone 0/15" in x for x in L)


def test_hold_on_open_fade(tmp_path):
    rows = [_row(i, -1.0, day=i // 2) for i in range(16)] + [_row(1, 0.0, day=0, status="OPEN", pair="OPENUSDT")]
    st = str(tmp_path / "st.json")
    L = _run(tmp_path, rows, state=st)
    assert _verdict(L) == "HOLD" and not os.path.exists(st)


def test_not_evaluable_and_no_bars(tmp_path, monkeypatch):
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    st = str(tmp_path / "st.json")
    L = S.run(SINCE, paths=_csv(tmp_path, [_row(0, -1.0)]), state_path=st, bars=_bars(), fetch=False)
    assert any("NOT EVALUABLE" in x for x in L) and not os.path.exists(st)
    monkeypatch.setattr(S, "KCACHE", str(tmp_path / "none.json"))
    L = S.run(SINCE + 90 * DAY, paths=_csv(tmp_path, [_row(i, -1.0, day=i // 2) for i in range(16)]), since_ms=SINCE, state_path=st,
              fetch=False)
    assert any("unavailable" in x for x in L) and not os.path.exists(st)


def test_floor_is_the_study_cutoff_once_deployed(monkeypatch):
    pin = int(pd.Timestamp(S.FLOOR_PIN).value // 1_000_000)
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: pin + DAY)        # deploy after the cutoff: the cutoff still rules
    assert S.floor_ms() == pin
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert S.floor_ms() is None


def test_registration_block_is_informational_and_never_persisted(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "REG_FROM", (T0 - dt.timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%S"))
    reg = [_row(100 + i, -1.0 if i % 3 else 0.3, day=-9 + i // 2) for i in range(16)]      # 16 in-sample zone fades before SINCE
    fwd = [_row(i, 0.3, pdi=60, day=i) for i in range(3)]                                  # gate cohort: 3 control fades only
    st = str(tmp_path / "st.json")
    L = _run(tmp_path, reg + fwd, state=st)
    assert _verdict(L) == "COLLECTING" and not os.path.exists(st)                         # the gate never counts the registration fades
    info = [x for x in L if x.lstrip().startswith("In-sample read WITH")]
    assert len(info) == 1 and "would be FILTER_CANDIDATE" in info[0] and "🚨" not in info[0]
    assert any(x.startswith("**Registration block") and "zone N 16" in x for x in L)


def test_fetch_caches_only_closed_bars_and_refetches_next_day(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "KCACHE", str(tmp_path / "c" / "k.json"))
    day0 = int(pd.Timestamp("2026-12-31").value // 1_000_000)
    calls = []

    def fake_get(url, bud):
        calls.append(url)
        n = 450 + len(calls)                                               # each call one more bar (the open one included)
        return [[day0 - (n - 2 - i) * DAY, 0, 0, 0, 100.0 + i] for i in range(n)]
    monkeypatch.setattr(S._MRL, "_get", fake_get)
    now = day0 + DAY // 2                                                  # mid-day: the bar opened at day0 is still open
    k, _ = S.load_btc_1d(now)
    assert int(k.t.max()) == day0 - DAY and len(calls) == 1
    assert max(r[0] for r in json.load(open(S.KCACHE))) == day0 - DAY       # the open bar was not cached
    S.load_btc_1d(now)
    assert len(calls) == 1                                                 # cache fresh → no second GET
    k2, _ = S.load_btc_1d(now + DAY)
    assert len(calls) == 2 and int(k2.t.max()) == day0                     # next day: refetch, the now-closed bar arrives
    monkeypatch.setattr(S._MRL, "_get", lambda url, bud: None)
    monkeypatch.setattr(S, "KCACHE", str(tmp_path / "none.json"))
    k3, note = S.load_btc_1d(now)
    assert k3 is None and "unavailable" in note
