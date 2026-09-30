"""⚡ Sep-30 SURGE sleeves — pure-rule invariants, live/default parity and engine wiring (DECISION_LOG 146–148).

Bull-run lessons pinned here: stateless trigger (closed bar only, fail-closed on short data), sign-slip normalisation of the move,
no look-ahead in the pair selection, the exit override is scoped to SURGE_LONG only, the kill bar, and every new config field on
all D11 surfaces."""
import json
import os
import re
from types import SimpleNamespace

import services.trading_engine as TE
from services.surge import surge_entry_open, surge_pair_pick, surge_trigger, surge_tripwire, wilder_atr_pct

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BAR = 300_000


def _th(**kw):
    d = dict(surge_btc_move_pct=1.0, surge_btc_vol_mult=3.0, surge_long_require_24h_high=True, surge_short_require_24h_low=False,
             surge_atr_min_pct=1.5, surge_long_require_leader=True, surge_short_require_leader=False,
             surge_long_entry_delay_min=0.0, surge_short_entry_delay_min=20.0, surge_entry_window_min=5.0)
    d.update(kw)
    return SimpleNamespace(**d)


def _btc(last_move_pct, n=300, vol_last=10.0, base=100.0):
    """n rows [ts,o,h,l,c,v], flat at `base` with volume 1, then the last CLOSED bar (index n-2) jumps; row n-1 = forming bar."""
    rows = [[i * BAR, base, base * 1.001, base * 0.999, base, 1.0] for i in range(n)]
    c = base * (1 + last_move_pct / 100.0)
    rows[n - 2] = [(n - 2) * BAR, base, max(base, c), min(base, c), c, vol_last]
    rows[n - 1] = [(n - 1) * BAR, c, c, c, c * 0.5, 999.0]   # forming bar: must be ignored
    return rows


def test_long_trigger_fires_on_the_last_closed_bar():
    t = surge_trigger(_btc(1.2), _th(), "LONG")
    assert t is not None and t["bar_ts"] == 298 * BAR and t["close_ts"] == 299 * BAR
    assert abs(t["btc_move_pct"] - 1.2) < 1e-6


def test_trigger_needs_move_volume_and_breakout():
    assert surge_trigger(_btc(0.9), _th(), "LONG") is None                        # move below the bar
    assert surge_trigger(_btc(1.2, vol_last=2.0), _th(), "LONG") is None          # volume 2× < 3×
    assert surge_trigger(_btc(1.2, vol_last=2.0), _th(surge_btc_vol_mult=0), "LONG") is not None   # 0 = volume leg off
    rows = _btc(1.2); rows[100][2] = 200.0                                         # an earlier 24 h high above the close
    assert surge_trigger(rows, _th(), "LONG") is None
    assert surge_trigger(rows, _th(surge_long_require_24h_high=False), "LONG") is not None


def test_short_trigger_and_sign_slip():
    assert surge_trigger(_btc(-1.2), _th(), "SHORT") is not None
    assert surge_trigger(_btc(-1.2), _th(surge_btc_move_pct=-1.0), "SHORT") is not None   # sign slip read as a magnitude
    assert surge_trigger(_btc(1.2), _th(surge_btc_move_pct=-1.0), "LONG") is not None
    assert surge_trigger(_btc(1.2), _th(), "SHORT") is None
    assert surge_trigger(_btc(-1.2), _th(), "LONG") is None


def test_trigger_fails_closed_on_short_or_bad_data():
    assert surge_trigger(_btc(1.2)[:200], _th(), "LONG") is None
    assert surge_trigger(None, _th(), "LONG") is None
    assert surge_trigger(_btc(1.2), _th(surge_btc_move_pct=0), "LONG") is None
    assert surge_trigger(_btc(1.2), _th(), "SIDEWAYS") is None


def test_entry_window_per_side():
    close = 1_000_000_000
    assert surge_entry_open(close, close, _th(), "LONG")
    assert not surge_entry_open(close + 5 * 60_000, close, _th(), "LONG")          # window [0, 5) min
    assert not surge_entry_open(close + 19 * 60_000, close, _th(), "SHORT")        # SHORT waits 20 min
    assert surge_entry_open(close + 20 * 60_000, close, _th(), "SHORT")
    assert not surge_entry_open(close + 25 * 60_000, close, _th(), "SHORT")


def _pair(atr_pct_target, move_pct, n=60, base=10.0):
    half = base * atr_pct_target / 100.0 / 2
    rows = [[i * BAR, base, base + half, base - half, base, 1.0] for i in range(n)]
    c = base * (1 + move_pct / 100.0)
    rows[n - 1] = [(n - 1) * BAR, base, max(base, c) + half, min(base, c) - half, c, 1.0]
    return rows


def test_pair_pick_atr_leader_and_no_lookahead():
    rows = _pair(2.0, 2.0); tb = rows[-1][0]
    ok, why, atr, pm = surge_pair_pick(rows, tb, 1.2, _th(), "LONG")
    assert ok and why is None and atr > 1.5 and abs(pm - 2.0) < 1e-9
    assert surge_pair_pick(rows, tb, 2.5, _th(), "LONG")[1] == "SURGE_NOT_LEADER"          # BTC outran the pair
    assert surge_pair_pick(_pair(0.5, 2.0), tb, 1.2, _th(), "LONG")[1] == "SURGE_ATR_LOW"
    assert surge_pair_pick(rows, tb, -1.2, _th(), "SHORT")[0] is True                       # SHORT: leader test off by default
    # bars AFTER the trigger bar are ignored; a missing trigger bar = no data (never judged on the wrong bar)
    later = rows + [[tb + BAR, 99.0, 99.0, 99.0, 99.0, 1.0]]
    assert surge_pair_pick(later, tb, 1.2, _th(), "LONG")[:2] == (True, None)
    assert surge_pair_pick(rows[:-1], tb, 1.2, _th(), "LONG")[1] == "SURGE_NO_DATA"
    assert wilder_atr_pct(rows[:10]) is None


def test_kill_bar():
    assert surge_tripwire([0.5] * 9, "LONG") is None                                        # judged only at 10
    assert surge_tripwire([1.0] * 3 + [-0.1] * 7, "LONG") is not None                       # 3 winners ≤ 3
    assert surge_tripwire([1.0] * 4 + [-0.1] * 6, "LONG") is None
    assert surge_tripwire([0.1] * 5 + [-1.0] * 5, "LONG") is not None                       # mean −0.45 ≤ −0.30
    assert surge_tripwire([1.0] * 4 + [-0.1] * 6, "SHORT") is not None                      # SHORT bar is 4 winners
    assert surge_tripwire([0.3] * 5 + [-0.72] * 5, "SHORT") is not None                     # mean −0.21 ≤ −0.20
    assert surge_tripwire([0.3] * 5 + [-0.65] * 5, "SHORT") is None                         # mean −0.175 > −0.20


def test_exit_override_scoped_to_surge_long():
    assert TE._surge_trail_override("SURGE_LONG") == 1.0
    for s in ("BULLRUN_LONG", "SURGE_SHORT", "MOMENTUM", None, ""):
        assert TE._surge_trail_override(s) is None
    # peak +2.0, ATR 1.0, now +0.5: GREEN door trails 2×ATR → line 0.2 (lock) holds; the 1×ATR override → line 1.0 → closes
    assert TE._bullrun_exit_for(0.5, 2.0, 1.0, "GREEN")[0] is False
    close, reason, line = TE._bullrun_exit_for(0.5, 2.0, 1.0, "GREEN", trail_mult_override=1.0)
    assert close is True and abs(line - 1.0) < 1e-9
    assert TE._bullrun_exit_for(0.5, 2.0, 1.0, "GREEN", trail_mult_override=None)[0] is False


def test_pair_reason_excludes_surge_counters():
    assert any("SURGE_ATR_LOW".startswith(p) for p in TE._PAIR_REASON_EXCLUDED_PREFIXES)


def test_config_parity_and_d11_surfaces():
    from config import SignalThresholds
    fields = set(SignalThresholds.model_fields) if hasattr(SignalThresholds, "model_fields") else set(SignalThresholds.__fields__)
    live = json.load(open(os.path.join(ROOT, "trading_config.json")))
    live_th = live.get("thresholds", live)
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    keys = sorted(f for f in fields if f.startswith("surge_"))
    assert len(keys) == 24, keys
    for k in keys:
        assert k in live_th, f"{k} missing from trading_config.json"
        assert html.count(k) >= 2, f"{k} not wired to the UI load + save handlers"
    ids = set(re.findall(r'id="(config-sg-[a-z0-9-]+)"', html))
    assert len(ids) == 22
    for i in ids:
        assert html.count(f"'{i}'") >= 1, f"{i} has an input but no handler"


def test_surge_table_on_ui_and_both_exports():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="surge-body"') == 1
    assert html.count("## ⚡ SURGE Sleeves") == 2          # clipboard copy + saved-file export


def test_open_position_accepts_every_surge_kwarg():
    import inspect
    ps = inspect.signature(TE.TradingEngine.open_position).parameters
    for k in ("surge_dir", "entry_surge_btc_move_pct", "entry_surge_pair_move_pct", "entry_surge_trigger_at"):
        assert k in ps
    from models import Order
    cols = {c.name for c in Order.__table__.columns}
    for k in ("entry_surge_btc_move_pct", "entry_surge_pair_move_pct", "entry_surge_trigger_at"):
        assert k in cols


def test_catch_up_view_judges_an_older_closed_bar():
    """The engine judges every bar closed since the last scan through views bars[:len-k] (review: a slow scan must not skip a
    trigger bar). A spike on the bar BEFORE the latest closed one is found with k=1 and not with k=0 (its 30-min window moved)."""
    rows = _btc(1.2)
    n = len(rows)
    base = rows[0][4]
    late = [r[:] for r in rows]
    late.insert(n - 1, [(n - 1) * BAR, late[n - 2][4], late[n - 2][4], late[n - 2][4], late[n - 2][4], 1.0])   # one more closed bar
    for i, r in enumerate(late):
        r[0] = i * BAR
    t1 = surge_trigger(late[:len(late) - 1], _th(), "LONG")
    assert t1 is not None and t1["bar_ts"] == (len(late) - 3) * BAR
    assert base > 0 and surge_trigger(late, _th(), "LONG") is None   # k=0 judges the newer, low-volume bar → no trigger


def test_kill_verdict_fields_are_engine_owned():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "surge_long_kill_verdict:" not in html and "surge_short_kill_verdict:" not in html   # never posted by the UI save
    assert "window._sgLoadedEnabled" in html                                                  # stale page can't re-enable a killed side


def test_every_catch_up_view_can_fire_at_the_engine_fetch_size():
    """Engine fetches 310 bars and judges views bars[:len-k] for k = 5..0 (deep review: at 300 bars the k=5 view had only 294
    closed bars and could never fire). A spike on the bar that is 'last closed' in view k must be found for every k."""
    for k in range(6):
        rows = _btc(1.2, n=310)
        spike = rows[308]; rows[308] = [308 * BAR, 100.0, 100.1, 99.9, 100.0, 1.0]
        idx = 308 - k
        rows[idx] = [idx * BAR] + spike[1:]
        for j in range(idx + 1, 310):                      # later bars hold the spike level with low volume (quiet)
            rows[j] = [j * BAR, spike[4], spike[4], spike[4], spike[4], 1.0]
        view = rows[:len(rows) - k] if k else rows
        t = surge_trigger(view, _th(), "LONG")
        assert t is not None and t["bar_ts"] == idx * BAR, k
