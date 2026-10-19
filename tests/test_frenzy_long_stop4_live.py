"""🛑 2026-10-19 FRENZY_LONG stop −4 LIVE (operator DECLARED OVERRIDE, DECISION_LOG 317) + 👁 the observe-only touch snapshots at the first −2 / −3
touch (services/touch_snapshot.py). LONG uses frenzy_long_stop_pct (4); WIDE / LITE / LITE_RETRY / the manual FRENZY exit stay on frenzy_stop_pct
(3); the LONG backstop rests beyond −4 by the standard margin; 0 = the old behaviour."""
import datetime as dt
import json
import os
import sys
from types import SimpleNamespace as NS

from services.frenzy import (frenzy_exit_for, frenzy_long_stop4_gate, frenzy_long_stop_config_errors, frenzy_stop_for)
from services.touch_snapshot import (TOUCH_FIELDS, TOUCH_LEVELS, touch_columns, touch_features, touch_mark, touch_outcome, touch_snapshot)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = NS(frenzy_stop_pct=3.0, frenzy_long_stop_pct=4.0, frenzy_tp_pct=1.5, frenzy_lite_tp_pct=3.0, frenzy_lock_arm_pct=0.0, broker_backstop_pct=2.5,
        frenzy_trail_arm_pct=5.0, frenzy_trail_giveback_pct=1.5)


# ── the stop per sleeve ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_long_uses_4_others_stay_3():
    assert frenzy_stop_for(TH, "FRENZY_LONG") == 4.0
    for s in ("FRENZY_WIDE", "FRENZY_LITE", "FRENZY_LITE_RETRY", None, "MANUAL"):
        assert frenzy_stop_for(TH, s) == 3.0, s
    for off in (0, 0.0, None, "x", -2, float("nan")):
        assert frenzy_stop_for(NS(**dict(vars(TH), frenzy_long_stop_pct=off)), "FRENZY_LONG") == 3.0, off   # 0 / bad = the shared stop
    assert frenzy_stop_for(NS(frenzy_stop_pct=0), "FRENZY_LITE") == 3.0                                       # shared 0 → 3 (as before)


def test_exit_lines():
    c, why, line = frenzy_exit_for(-3.5, 0.2, TH, None, use_tp=True, sleeve="FRENZY_LONG")
    assert not c and line == -4.0                                                                             # LONG rides through −3
    assert frenzy_exit_for(-4.01, 0.2, TH, None, use_tp=True, sleeve="FRENZY_LONG")[:2] == (True, "STOP_LOSS")
    for s in ("FRENZY_LITE", "FRENZY_WIDE", "FRENZY_LITE_RETRY"):
        assert frenzy_exit_for(-3.01, 0.2, TH, None, use_tp=True, sleeve=s)[:2] == (True, "STOP_LOSS"), s
    assert frenzy_exit_for(-3.01, 0.2, TH, None, use_tp=True)[:2] == (True, "STOP_LOSS")                     # the manual FRENZY exit (no sleeve)
    old = NS(**dict(vars(TH), frenzy_long_stop_pct=0.0))
    assert frenzy_exit_for(-3.01, 0.2, old, None, use_tp=True, sleeve="FRENZY_LONG")[:2] == (True, "STOP_LOSS")   # switch 0 = −3


def test_backstop_beyond_the_long_stop():
    import services.trading_engine as TE
    assert TE.bot_own_stop_pct(TH, "FRENZY_LONG") == 4.0 and TE.bot_own_stop_pct(TH, "FRENZY_LITE") == 3.0
    bk = TE.order_backstop_pct(TH, "FRENZY_LONG", leverage=1)
    assert bk == 4.5 and bk - 4.0 >= TE.MANUAL_BACKSTOP_MIN_GAP_PCT                                            # fires only beyond −4
    assert TE.order_backstop_pct(TH, "FRENZY_LITE", leverage=8) == 3.5 and TE.order_backstop_pct(TH, "FRENZY_WIDE", leverage=1) == 3.5
    assert TE.order_backstop_pct(NS(**dict(vars(TH), frenzy_long_stop_pct=0)), "FRENZY_LONG", leverage=1) == 3.5   # 0 = old
    assert TE.order_backstop_pct(TH, "FRENZY_LONG", leverage=20) < 5.0                                       # liquidation cap still applies
    assert TE.order_backstop_pct(TH, "MANUAL", "FRENZY", None, 1) == 3.5                                     # manual FRENZY keeps −3


def test_entry_stamp(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=NS(**dict(vars(TH), frenzy_willy_stop_pct=4.0, frenzy_willy_tp_pct=1.25,
                                                                               frenzy_willy_max_hold_minutes=90))))
    assert TE.TradingEngine._frenzy_entry_stop_pct("FRENZY_LONG") == 4.0
    assert TE.TradingEngine._frenzy_entry_stop_pct("FRENZY_LITE") == 3.0
    assert TE.TradingEngine._frenzy_entry_stop_pct("FRENZY_WILLY") == 4.0
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "entry_frenzy_stop_pct=(self._frenzy_entry_stop_pct(_fz_es) if _frenzy else None)" in eng
    assert "_stop = ((abs(_wsl) if _wsl is not None else None) if _wl else frenzy_stop_for(th, _es))" in eng   # stop/ATR stamp uses it too


def test_revert_and_resize_gate():
    d = lambda i, h=0: dt.datetime(2026, 10, 20) + dt.timedelta(days=i, hours=h)
    ten_bad = [(d(i), "P%d" % i, -4.05, 4.0, d(i, 1), i < 2) for i in range(10)]                             # 2/10 recovered
    assert frenzy_long_stop4_gate(ten_bad)[0] == "REVERT"
    ok10 = [(d(i), "P%d" % i, 1.5 if i < 3 else -4.05, 4.0, d(i, 1), i < 3) for i in range(10)]
    assert frenzy_long_stop4_gate(ok10)[0] == "COLLECTING"
    pre317 = [(d(i), "P", -3.0, None, d(i, 1), False) for i in range(10)]                                     # no stamp = −3 era: ignored
    assert frenzy_long_stop4_gate(pre317)[0] == "COLLECTING"
    neg = [(d(i // 2), "P%d" % i, (1.5 if i < 3 else -4.05), 4.0, d(i // 2, 1 + i % 2), i < 3) for i in range(20)]   # 3/10 pass · Σ(live − (−3)) < 0
    assert frenzy_long_stop4_gate(neg)[0] == "REVERT"
    good = [(d(i // 3), "P%d" % i, 1.5, 4.0, None, None) for i in range(24)]                                 # 24 wins on 8 days, never touched
    assert frenzy_long_stop4_gate(good)[0] == "RESIZE"
    one_pair = [(d(i // 3), "SAME", 1.5, 4.0, None, None) for i in range(24)]
    assert frenzy_long_stop4_gate(one_pair)[0] == "HOLD"                                                       # one pair = 100 % of the gain


# ── touch snapshots (observe only) ─────────────────────────────────────────────────────────────────────────────────────────────────

def test_touch_mark_once_per_level():
    mk = {}
    assert touch_mark(mk, -1.0, 0.8, 1000, 1.0) == []
    assert touch_mark(mk, -2.2, 0.8, 2000, 0.98) == [2] and mk[2]["ms"] == 2000 and mk[2]["peak"] == 0.8
    assert touch_mark(mk, -2.5, 0.8, 3000, 0.97) == []                                                        # never re-recorded
    assert touch_mark(mk, -3.1, 0.8, 4000, 0.96) == [3] and mk[2]["min"] == -3.1
    touch_mark(mk, 1.6, 1.6, 5000, 1.02)
    assert mk[3]["max"] == 1.6 and mk[3]["min"] == -3.1 and mk[2]["ms"] == 2000 and mk[3]["ms"] == 4000
    assert touch_mark(mk, "x", None, 6000, None) == [] and touch_mark(None, -5, 0, 1, 1) == []                # never raises


def _rows(n=120, entry_bar=60):
    """5m rows rising 0.1 % a bar, the entry bar volume 1000, the last bar 2500."""
    rows = []
    for i in range(n):
        c = 100 * (1.001 ** i)
        rows.append([i * 300_000, c * 0.999, c * 1.002, c * 0.997, c, 1000.0 if i == entry_bar else (2500.0 if i == n - 1 else 800.0)])
    return rows


def test_touch_features_on_a_synthetic_path():
    rows = _rows()
    now = 120 * 300_000 + 1000; entry_ms = 61 * 300_000
    f = touch_features(rows, 95.0, now, entry_ms, vwap=100.0, btc_now=60_600.0, btc_entry=60_000.0, btc_1h_slope=-0.02)
    assert f["vs_ema50_pct"] < 0 and f["vs_vwap_pct"] == -5.0 and f["btc_since_entry_pct"] == 1.0 and f["btc_1h_slope"] == -0.02
    assert f["vol_ratio"] == 2.5 and 0 < f["atr_pct"] < 1 and f["rsi"] > 60 and f["adx"] is not None
    assert touch_features(None, 95.0, now, entry_ms)["vs_ema50_pct"] is None                                  # not in memory → None
    assert touch_features(rows, None, now, entry_ms)["rsi"] is None
    forming = rows + [[120 * 300_000, 1, 1, 1, 1, 99999.0]]                                                  # a forming bar is ignored
    assert touch_features(forming, 95.0, now, entry_ms)["vol_ratio"] == 2.5


def test_touch_snapshot_and_outcome():
    oa = dt.datetime(2026, 10, 19, 12, 0)
    m = dict(ms=int((oa + dt.timedelta(minutes=7)).replace(tzinfo=dt.timezone.utc).timestamp() * 1000), peak=0.6, price=0.97, min=-3.1, max=1.6)
    snap = touch_snapshot(m, oa, dict(rsi=40.0))
    assert snap["min"] == 7.0 and snap["peak_before"] == 0.6 and snap["rsi"] == 40.0
    assert touch_snapshot(m, oa.replace(tzinfo=dt.timezone.utc), {})["min"] == 7.0                             # tz-aware entry
    assert touch_outcome(m, 1.5, "FRENZY_TP", 1.5, -3.1) == (True, -3.1)
    assert touch_outcome(dict(m, max=0.4), -4.05, "STOP_LOSS", 1.5, -4.05) == (False, -4.05)
    assert touch_outcome(dict(m, max=1.7), 0.2, "MAX_HOLD_TIME", 1.5, -3.1)[0] is True                       # reached the TP after the touch
    assert touch_outcome(None, -1.0, "MAX_HOLD_TIME", 1.5, -3.3) == (False, -3.3)                            # markers lost (restart)
    assert set(touch_columns()) == {f"touch{lv}_{f}" for lv in TOUCH_LEVELS for f in TOUCH_FIELDS} and len(touch_columns()) == 26


def test_engine_writes_once_and_finalizes(monkeypatch):
    import services.trading_engine as TE
    monkeypatch.setattr(TE.config, "trading_config", NS(thresholds=TH))
    monkeypatch.setattr(TE, "_current_btc_price", 60_600.0)
    monkeypatch.setattr(TE, "_current_btc_1h_slope", 0.01)
    monkeypatch.setitem(TE._frenzy_kl_cache, "FOOUSDT", dict(rows=_rows(), keep=1500))
    monkeypatch.setitem(TE._frenzy_flags, "FOOUSDT", dict(vwap=100.0))
    e = object.__new__(TE.TradingEngine)
    o = NS(id=7, pair="FOOUSDT", entry_strategy="FRENZY_LONG", opened_at=dt.datetime(2026, 10, 19, 12, 0), entry_btc_price=60_000.0,
           peak_pnl=0.5, trough_pnl=-3.4, exit_price=None, frenzy_lite_tp_used=None, **{c: None for c in touch_columns()})
    e._touch_note(7, -2.1, 0.5, 98.0); e._touch_write(o)
    assert o.touch2_at is not None and o.touch2_peak_before == 0.5 and o.touch2_btc_since_entry_pct == 1.0 and o.touch3_at is None
    first = o.touch2_at
    e._touch_note(7, -2.4, 0.5, 97.0); e._touch_write(o)
    assert o.touch2_at == first                                                                                 # written once
    e._touch_note(7, -3.4, 0.5, 96.0)                                                                         # touched, monitor never saw it
    o.exit_price = 101.5
    e._touch_finalize(o, 1.52, "FRENZY_TP")                                                                    # the close writes it + the outcome
    assert o.touch3_at is not None and o.touch3_recovered_tp is True and o.touch3_min_after == -3.4 and o.touch2_recovered_tp is True
    assert 7 not in e._touch_mk
    lite = NS(**dict(vars(o), id=8, entry_strategy="MOMENTUM", **{c: None for c in touch_columns()}))
    e._touch_note(8, -3.5, 0, 1); e._touch_finalize(lite, -3.5, "STOP_LOSS")
    assert lite.touch3_at is None                                                                               # FRENZY family only
    e._touch_write(NS(id=None)); e._touch_finalize(NS(), None, None)                                           # never raises


def test_hooks_are_observe_only_and_wired():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("self._touch_note(") >= 2 and "self._touch_write(order)" in eng
    assert eng.count("self._touch_finalize(order,") == 2                                                       # both booking paths
    assert "entry_btc_price=(globals().get('_current_btc_price') if _frenzy else None)" in eng
    import models
    cols = {c.name for c in models.Order.__table__.columns}
    assert set(touch_columns()) | {"entry_frenzy_stop_pct", "entry_btc_price"} <= cols                         # ride the orders CSV
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "from services.touch_snapshot import TOUCH_LEVELS as _TL, TOUCH_FIELDS as _TF" in db and "'entry_frenzy_stop_pct'" in db
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert 'f"  👁 touched −{_tl}% (first touch snapshot)"' in main and "_l4g(_l4r)[1]" in main                 # FRENZY Fills rows → both exports


# ── config / UI parity ─────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_ui_parity():
    import config as C
    assert C.SignalThresholds.model_fields["frenzy_long_stop_pct"].default == 0.0
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert cfg["frenzy_long_stop_pct"] == 4.0 and cfg["frenzy_stop_pct"] == 3.0
    assert "DECISION_LOG 317" in open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert frenzy_long_stop_config_errors(NS(frenzy_long_stop_pct=4.0)) == [] and frenzy_long_stop_config_errors(NS(frenzy_long_stop_pct=0)) == []
    assert frenzy_long_stop_config_errors(NS(frenzy_long_stop_pct=0.5)) and frenzy_long_stop_config_errors(NS(frenzy_long_stop_pct=12))
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_cfg_errs.extend(_ls_cfg_errors(new_config.thresholds, _sb_lev))" in main and '"long_stop": _fz_stop_for(_th, "FRENZY_LONG")' in main
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="config-fz-long-stop"') == 1 and html.count("['config-fz-long-stop', 'frenzy_long_stop_pct', 0]") == 1
    assert "(FRENZY_LONG −' + _bt.frenzy_long_stop_pct + '%, DECISION_LOG 317)" in html and "(LONG −' + m.long_stop + '%)" in html
    assert "const _fzSl = ((o.entry_strategy || '') === 'FRENZY_LONG'" in html


def test_long_stop4_scout_reads_live_minus_4():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import scout_long_stop4 as LS
    assert LS.is_live4(4.0) and not LS.is_live4(3.0) and not LS.is_live4(None) and not LS.is_live4("nan")
    assert LS.cohort_stopped(True, "FRENZY_TP", 1.5, "STOP") is True                                          # live −4: touched −3 = in the cohort
    assert LS.cohort_stopped(True, "STOP_LOSS", -4.05, "TP") is False
    assert LS.cohort_stopped(False, "STOP_LOSS", -3.05, "TP") is True                                         # pre-317: the booked stop
    assert "entry_frenzy_stop_pct" in LS.COLS and "LIVE −4 NOTE" in open(LS.__file__, encoding="utf-8").read()
    assert (LS.TP, LS.SL_LIVE, LS.SL_SHADOW, LS.GATE_STOPS, LS.GATE_DAYS) == (1.5, 3.0, 4.0, 15, 8)            # frozen gate unchanged


# ── dual review ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_unknown_outcome_excluded_and_frozen_main_sample():
    d = lambda i, h=1: dt.datetime(2026, 10, 20) + dt.timedelta(days=i, hours=h)
    unk = [(d(i), "P%d" % i, -4.05, 4.0, d(i), None) for i in range(10)]                                      # outcome unknown ×10
    v, txt = frenzy_long_stop4_gate(unk)
    assert v == "COLLECTING" and "10 unknown outcome" in txt and "recovered 0/0" in txt
    mixed = unk + [(d(10 + i), "Q%d" % i, -4.05, 4.0, d(10 + i), i < 3) for i in range(10)]                    # 3 of the first 10 KNOWN
    v, txt = frenzy_long_stop4_gate(mixed)
    assert v == "REVERT" and "first 20 touches on 20 days" in txt and "recovered to the TP (<" not in txt   # first-10 (known) passed 3/10
    # frozen main sample: 20 touches on 4 days (recovered ≥ 3 of the first 10) → not judged yet; the read waits for 8 days
    burst = [(d(i // 5), "B%d" % i, (1.5 if i < 4 else -4.05), 4.0, d(i // 5, 1 + i % 5), i < 4) for i in range(20)]
    assert frenzy_long_stop4_gate(burst)[0] == "COLLECTING"
    later = burst + [(d(4 + i), "L%d" % i, -4.05, 4.0, d(4 + i), False) for i in range(4)]                     # 8 days reached at touch 24
    v, txt = frenzy_long_stop4_gate(later)
    assert v == "REVERT" and "first 24 touches on 8 days" in txt


def test_reconciler_finalizes_the_touch_outcome():
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    i = main.index("order.close_reason = \"EXTERNAL_CLOSE\""); j = main.index("tx = Transaction(", i)
    assert "trading_engine._touch_finalize(order, order.pnl_percentage, order.close_reason)" in main[i:j]


def test_touch_markers_pruned_and_skipped_while_closing(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine)
    e._touch_note(1, -2.5, 0, 1.0); e._touch_note(2, -2.5, 0, 1.0)
    e._touch_prune({2})
    assert set(e._touch_mk) == {2}
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "in FRENZY_STRATEGIES and not order_info.get('_closing_in_progress'):" in eng
    assert "self._touch_prune({_o.id for _o in open_orders})" in eng


def test_stale_cache_gives_no_pair_fields():
    rows = _rows()
    late = 120 * 300_000 + 11 * 60_000                                                                          # last bar closed 11 min before
    f = touch_features(rows, 95.0, late, 61 * 300_000, vwap=100.0, btc_now=1.0, btc_entry=1.0)
    assert f["vs_ema50_pct"] is None and f["rsi"] is None and f["vol_ratio"] is None and f["vs_vwap_pct"] == -5.0


def test_config_refuses_a_stop_inside_the_backstop():
    base = dict(frenzy_stop_pct=3.0, broker_backstop_pct=2.5, frenzy_long_lev_mult=0.05, frenzy_long_lev_mult_strong=0.0)
    assert frenzy_long_stop_config_errors(NS(**base, frenzy_long_stop_pct=4.0)) == []                         # 1×: backstop 4.5
    errs = frenzy_long_stop_config_errors(NS(**dict(base, frenzy_long_lev_mult_strong=1.0), frenzy_long_stop_pct=4.0))   # strong 20×: 4.14
    assert errs and "too wide" in errs[0]
    assert frenzy_long_stop_config_errors(NS(**dict(base, frenzy_long_lev_mult=1.0), frenzy_long_stop_pct=4.0), base_leverage=10) == []   # 10×: 4.5
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert "_cfg_errs.extend(_ls_cfg_errors(new_config.thresholds, _sb_lev))" in main
