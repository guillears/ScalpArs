"""Route B weekly-scan PREREG build item 3 — the WORKED / REACH labels, the path decomposition, the S5 no-spot label. No network."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "route_b_signals"))
import labels as LB      # noqa: E402

M = 60_000


def _bars(fav_hi, adv_lo, entry=100.0, side=-1):
    """per-minute favourable / adverse % → h / l arrays for a SHORT (fav = down) or LONG."""
    fav_hi, adv_lo = np.asarray(fav_hi, float), np.asarray(adv_lo, float)
    if side < 0:
        l = entry * (1 - fav_hi / 100)
        h = entry * (1 - adv_lo / 100)
    else:
        h = entry * (1 + fav_hi / 100)
        l = entry * (1 + adv_lo / 100)
    return h, l


@pytest.mark.parametrize("side", [1, -1])
def test_worked_before_stop(side):
    h, l = _bars([0.3, 1.2, 0.1], [-0.2, -0.5, -3.5], side=side)
    r = LB.worked_reach(h, l, 100.0, side, sl_pct=3.0, hold_min=3)
    assert r["lbl_path"] == "WORKED" and r["lbl_worked"] == 1.0 and r["lbl_pass_min"] == 2.0
    assert r["lbl_final"] and r["lbl_reach_pct"] == pytest.approx(1.2)          # up to and including the stop minute


def test_stop_first_is_not_worked():
    h, l = _bars([0.3, 0.4, 2.0], [-0.2, -3.1, -0.1])
    r = LB.worked_reach(h, l, 100.0, -1, 3.0, 10)
    assert r["lbl_path"] == "NEVER" and r["lbl_worked"] == 0.0 and r["lbl_reach_pct"] == pytest.approx(0.4) and r["lbl_final"]


def test_same_minute_both_is_ambig_zero():
    h, l = _bars([0.2, 1.5], [-0.1, -3.2])
    r = LB.worked_reach(h, l, 100.0, -1, 3.0, 10)
    assert r["lbl_path"] == "AMBIG" and r["lbl_worked"] == 0.0 and r["lbl_ambig"]


def test_flat_needs_the_full_hold():
    h, l = _bars([0.2] * 5, [-0.2] * 5)
    a = LB.worked_reach(h, l, 100.0, -1, 3.0, 10)                                # path shorter than the hold → unknown
    assert a["lbl_path"] == "PENDING" and math.isnan(a["lbl_worked"]) and not a["lbl_final"]
    b = LB.worked_reach(h, l, 100.0, -1, 3.0, 5)
    assert b["lbl_path"] == "FLAT" and b["lbl_worked"] == 0.0 and b["lbl_final"] and b["lbl_reach_pct"] == pytest.approx(0.2)


def test_worked_known_while_reach_still_open():
    h, l = _bars([1.1, 0.5], [-0.1, -0.2])
    r = LB.worked_reach(h, l, 100.0, -1, 3.0, 60)
    assert r["lbl_worked"] == 1.0 and not r["lbl_final"] and math.isnan(r["lbl_reach_pct"])


def test_path_beyond_hold_is_ignored():
    h, l = _bars([0.2, 0.2, 5.0], [-0.1, -0.1, -0.1])
    r = LB.worked_reach(h, l, 100.0, -1, 3.0, 2)
    assert r["lbl_path"] == "FLAT" and r["lbl_reach_pct"] == pytest.approx(0.2)


def test_no_levels():
    assert LB.worked_reach(np.ones(3), np.ones(3), 100.0, -1, None, 10)["lbl_path"] == "NO_LEVELS"


def test_decomposition_groups():
    t = np.arange(300) * M
    # loser that never worked, then recovered +1 % within 120 min after the exit
    h, l = _bars(np.r_[[0.2] * 10, [1.5] * 290], [-3.2] * 300)
    d = LB.decomposition(h, l, t, 100.0, -1, exit_ms=0, booked_pct=-3.3, lbl_path="NEVER")
    assert d["lbl_group"] == "STOPPED_RECOVERED" and d["lbl_recovered"] is True
    d2 = LB.decomposition(h, l, t, 100.0, -1, exit_ms=0, booked_pct=-3.3, lbl_path="WORKED")
    assert d2["lbl_group"] == "GAVE_BACK"
    h3, l3 = _bars([5.0] * 300, [-0.1] * 300)
    d3 = LB.decomposition(h3, l3, t, 100.0, -1, exit_ms=0, booked_pct=1.0, lbl_path="WORKED")
    assert d3["lbl_group"] == "LEFT_ON_TABLE"
    d4 = LB.decomposition(h3[:30], l3[:30], t[:30], 100.0, -1, exit_ms=0, booked_pct=1.0, lbl_path="WORKED")
    assert d4["lbl_group"] == "PENDING"                                            # bars do not reach exit + 45 min yet


# ───────────────────────────── item 3 wiring: price_signal + the S5 no-spot label
import json  # noqa: E402

import build as BLD      # noqa: E402
import common as C       # noqa: E402
import sources as SRC    # noqa: E402
from tests.test_route_b_signals import T0, _crash_klines, _day_frame, _sig_row  # noqa: E402


def _long_crash():
    k = _day_frame(T0 - 40 * 60 * M, n=40 * 60 + 1700)
    k["h"] = k.o + 0.01
    k.loc[k.open_ms >= T0 + 40 * M, "l"] = 1.0
    return k


def test_price_signal_carries_labels_on_the_same_path(monkeypatch):
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    k = _long_crash()                                                          # the crash: a short works, TP hit
    o = BLD.price_signal(_sig_row(), sp, k, None, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_reason"] == "TP" and o["out_lbl_path"] == "WORKED" and o["out_lbl_worked"] == 1.0
    assert o["out_lbl_group"] in ("CAPTURED", "LEFT_ON_TABLE") and o["out_final"]


def test_label_open_keeps_pricing_final_but_blocks_the_carry(monkeypatch):
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    k = _crash_klines()
    k = k[k.open_ms < T0 + 120 * M].reset_index(drop=True)                     # TP inside, the 24-h label hold not covered
    o = BLD.price_signal(_sig_row(), sp, k, None, T0 + 2_000 * M, T0 + 1_900 * M)
    assert o["out_reason"] == "TP" and o["out_final"] and not o["out_lbl_done"]  # pricing finality is unchanged (no loser tilt)
    key = BLD.carry_key(sp, None, "BLOCKED:EXH_VOL_DECAY")
    prev = dict(o, code_version=BLD.CODE_VERSION, carry_key=key)
    assert not BLD.can_carry(prev, key) and BLD.can_carry(dict(prev, out_lbl_done=True), key)
    late = BLD.price_signal(_sig_row(), sp, k, None, T0 + (1440 + 180) * M + BLD.LABEL_GRACE_MS + 31 * M, T0 + 9_000 * M)
    assert late["out_lbl_done"] and late["out_lbl_path"] == "WORKED" and late["out_lbl_group"] == "LEFT_ON_TABLE"   # known label + group kept; reach stays NaN


def test_exact_label_and_group_on_the_crash(monkeypatch):
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    o = BLD.price_signal(_sig_row(), sp, _long_crash(), None, T0 + 10_000 * M, T0 + 9_000 * M)
    assert o["out_lbl_path"] == "WORKED" and o["out_lbl_pass_min"] == 10.0 and o["out_lbl_group"] == "LEFT_ON_TABLE" and o["out_lbl_done"]
    assert o["out_lbl_reach_pct"] == pytest.approx((1 - 1.0 / float(o["out_entry_px"])) * 100)


def _flat(n, start, px=100.0):
    t = start + np.arange(n) * M
    return pd.DataFrame(dict(open_ms=t, o=px, h=px * 1.001, l=px * 0.999, c=px, qv=1.0))


def test_real_stop_before_the_label_passage_is_never_for_the_decomposition(monkeypatch):
    """FLATBASE's stop counts funding paid (DL 607): the walk stops at sl − paid, the price label has not passed −sl yet, then the price
    falls +1 % our way → lbl_worked stays the price label (WORKED), the decomposition reads the REAL stop: NEVER (review H3)."""
    T = T0 + 30 * M
    k = _flat(6000, T0 - 600 * M)
    i = (k.open_ms >= T + 60 * M) & (k.open_ms < T + 70 * M)
    k.loc[i, ["h", "c"]] = [100 * 1.065, 100 * 1.06]                          # +6.5 % against the short (stop 7 − funding 1.5 = 5.5)
    k.loc[k.open_ms >= T + 70 * M, ["l", "c", "o", "h"]] = [98.5, 98.6, 98.6, 98.7]
    sp = dict(C.exit_specs(C.load_thresholds())["FLATBASE_SHORT"], tp=20.0, sl=7.0, stop_counts_funding=True)
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([(T + 50 * M, -0.015, 100.0)], True))
    r = _sig_row(sleeve="FLATBASE_SHORT", signal_time=pd.Timestamp(T, unit="ms", tz="UTC"))
    o = BLD.price_signal(r, sp, k, None, T0 + 100_000 * M, T0 + 90_000 * M)
    assert o["out_reason"] == "STOP_LOSS" and o["out_lbl_path"] == "WORKED"
    assert o["out_lbl_decomp_path"] == "NEVER" and o["out_lbl_group"] in ("NEVER_WORKED", "STOPPED_RECOVERED")


def test_decomposition_label_rule():
    assert LB.decomposition_label("WORKED", 5.0, "STOP_LOSS", 10) == "WORKED"
    assert LB.decomposition_label("WORKED", 12.0, "STOP_LOSS", 10) == "NEVER"
    assert LB.decomposition_label("FLAT", math.nan, "STOP_LOSS", 10) == "NEVER"
    assert LB.decomposition_label("FLAT", math.nan, "MAX_HOLD_TIME", 10) == "FLAT"


def test_decomposition_pending_while_walk_not_final():
    t = np.arange(300) * M
    h, l = _bars([5.0] * 300, [-0.1] * 300)
    assert LB.decomposition(h, l, t, 100.0, -1, 0, 1.0, "WORKED", walk_final=False)["lbl_group"] == "PENDING"


@pytest.mark.parametrize("t_off,group", [(44, "PENDING"), (45, "LEFT_ON_TABLE")])
def test_decomposition_best_window_boundary(t_off, group):
    """best = bars opening before the exit minute's end + 45 min: the bar at +45 is the first one needed for coverage."""
    t = np.arange(t_off + 1) * M
    fav = np.zeros(t_off + 1)
    fav[-1] = 5.0
    h, l = _bars(fav, [-0.1] * (t_off + 1))
    d = LB.decomposition(h, l, t, 100.0, -1, exit_ms=0, booked_pct=1.0, lbl_path="WORKED")
    assert d["lbl_group"] == group


@pytest.mark.parametrize("bar,recovered", [(0, False), (1, True), (120, True), (121, False)])
def test_decomposition_recovery_window(bar, recovered):
    """recovered = +1 % in the bars AFTER the exit minute and before its end + 120 min."""
    n = 200
    t = np.arange(n) * M
    fav = np.zeros(n)
    fav[bar] = 1.5
    h, l = _bars(fav, [-0.1] * n)
    d = LB.decomposition(h, l, t, 100.0, -1, exit_ms=0, booked_pct=-3.0, lbl_path="NEVER")
    assert d["lbl_recovered"] is recovered and d["lbl_group"] == ("STOPPED_RECOVERED" if recovered else "NEVER_WORKED")


def test_exact_stop_touch_counts():
    h, l = _bars([0.2, 0.3], [-0.1, -3.0])
    assert LB.worked_reach(h, l, 100.0, -1, 3.0, 10)["lbl_path"] == "NEVER"


def test_taken_fill_pessimistic_minute_feeds_the_label(monkeypatch):
    """the fill minute's favourable extreme is capped (pre-fill prints) — a +2 % low inside the fill minute must not make it WORKED."""
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    sp = C.exit_specs(C.load_thresholds())["EXHAUSTION_SHORT"]
    k = _flat(3000, T0 - 600 * M)
    fm = T0 + 31 * M
    k.loc[k.open_ms == fm, "l"] = 98.0                                          # a pre-fill print 2 % lower in the fill minute
    fill = pd.Timestamp(fm + 20_000, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
    r = _sig_row(status="OPENED", fill_entry_price=100.0, fill_opened_at=fill)
    o = BLD.price_signal(r, sp, k, None, T0 + 100_000 * M, T0 + 90_000 * M)
    assert o["out_fill_minute"] == "pessimistic" and o["out_lbl_path"] == "FLAT"


def test_ht_label_uses_the_gross_stop(monkeypatch):
    th = C.load_thresholds()
    th = dict(th, frenzy_willy_ht_late_tp_enabled=False, frenzy_willy_ht_high_target_enabled=False, frenzy_willy_ht_tp_pct=3.0)
    sp = C.exit_specs(th)["FRENZY_WILLY_HT"]
    monkeypatch.setattr(SRC, "funding_history", lambda *a, **kw: ([], True))
    k = _flat(3000, T0 - 600 * M)
    i = k.open_ms >= T0 + 40 * M
    k.loc[i, "l"] = 100 * (1 - (sp["sl_gross"] - 0.01) / 100)                  # just inside the GROSS stop: never touched
    r = _sig_row(sleeve="FRENZY_WILLY_HT", direction="LONG")
    o = BLD.price_signal(r, sp, k, None, T0 + 100_000 * M, T0 + 90_000 * M)
    assert o["out_sl_pct"] == pytest.approx(sp["sl_gross"]) and o["out_lbl_path"] != "NEVER"


def _s5_setup(tmp_path, monkeypatch, k, T):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({f"KUSDT|{T}": {"entry": 100.0, "sl": 2.0, "tp": 3.0, "ok": True}}))
    monkeypatch.setattr(BLD, "klines_for", lambda sym, a, b, now: k[(k.open_ms >= a) & (k.open_ms < b)].reset_index(drop=True))
    S = pd.DataFrame([dict(_sig_row(status="BLOCKED:EXH_NO_SPOT", signal_time=pd.Timestamp(T, unit="ms", tz="UTC"))),
                      dict(_sig_row(status="BLOCKED:EXH_VOL24_LOW", signal_time=pd.Timestamp(T, unit="ms", tz="UTC")))])
    return S, str(p)


def test_s5_walks_from_the_first_whole_minute_after_t_plus_8s(tmp_path, monkeypatch):
    T = T0 + 30 * M                                                              # t1 = T + 1 min (T + 8 s floors to T)
    k = _flat(3000, T0 - 600 * M)
    k.loc[k.open_ms == T, "l"] = 98.0                                            # inside the fill minute: NOT walked
    k.loc[k.open_ms == T + 5 * M, "h"] = 102.5                                   # minute 5 of the walk: the stop (2 %)
    S, p = _s5_setup(tmp_path, monkeypatch, k, T)
    sp = C.exit_specs(C.load_thresholds())
    out = BLD.s5_labels(S, sp, T0 + 100_000 * M, cache_path=p)
    assert out.s5_lbl_path.iloc[0] == "NEVER" and out.s5_lbl_pass_min.iloc[0] == 5.0 and out.s5_lbl_final.iloc[0] is True
    assert out.s5_entry_px.iloc[1] != out.s5_entry_px.iloc[1]                    # another gate: NaN
    assert BLD.s5_labels(S, sp, T0, cache_path=str(tmp_path / "missing.json")).s5_entry_px.isna().all()


def test_s5_gap_is_pending_then_no_data(tmp_path, monkeypatch):
    T = T0 + 30 * M
    k = _flat(3000, T0 - 600 * M)
    k = k[k.open_ms != T + 200 * M].reset_index(drop=True)                       # one missing minute inside the hold
    S, p = _s5_setup(tmp_path, monkeypatch, k, T)
    sp = C.exit_specs(C.load_thresholds())
    assert BLD.s5_labels(S, sp, T + 2000 * M, cache_path=p).s5_lbl_path.iloc[0] == "PENDING"
    late = BLD.s5_labels(S, sp, T + 1500 * M + BLD.LABEL_GRACE_MS, cache_path=p)
    assert late.s5_lbl_path.iloc[0] == "NO_DATA" and late.s5_lbl_final.iloc[0] is True


def test_s5_hold_is_the_scouts():
    th = C.load_thresholds()
    sp = C.exit_specs(th)["EXHAUSTION_SHORT"]
    assert BLD.s5_hold(th, sp) == sp["hold_min"]
