"""🎲² tests for scripts/scout_willy_exit_v2.py (DECISION_LOG 356) — synthetic paths / exports, no network, state in tmp."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_willy_exit_v2 as S  # noqa: E402
import willy_exit_era as WEra  # noqa: E402

TE0 = 1_761_000_000_000
MIN = 60_000


def _P(te, ev, bar=False, cap=90):
    return S._P(te, ev, cap=cap, bar=bar)


def test_selftest(capsys):
    assert S.selftest() and "selftest WILLY_EXIT_V2 OK" in capsys.readouterr().out


def test_pinned_shadow_and_constants():
    assert (S.OLD_TP, S.OLD_STOP, S.OLD_CAP, S.OLD_LATE_MIN, S.OLD_LATE_TP) == (1.25, 4.0, 90, 30.0, 0.25)
    assert (S.ARM, S.FLOOR) == (1.25, 1.0)
    assert (S.EARLY_N, S.EARLY_GAP, S.MAIN_N, S.MAIN_DAYS, S.FINAL_N) == (10, 0.15, 15, 6, 25)
    assert S.STATE.endswith(os.path.join("reports", ".willy_exit_v2_gate.json"))
    assert set(WEra.V2_COLS) <= set(S.COLS) and "close_reason" in S.COLS and "entry_btc_1h_slope" in S.COLS


def test_walk_v2_equals_old_exit_when_target_is_125():
    rng = np.random.default_rng(7)
    for k in range(200):
        n = 40
        mins = np.sort(rng.uniform(0.1, 95, n))
        x = np.cumsum(rng.normal(0, 0.6, n))
        P = _P(TE0, [(m, v, v) for m, v in zip(mins, x)])
        a, b = S.walk_v2(P, 1.25), S.shadow_today(P)
        assert a["why"] == b["why"], (k, a, b)
        assert (np.isnan(a["pct"]) and np.isnan(b["pct"])) or abs(a["pct"] - b["pct"]) < 1e-9


def test_walk_v2_runner_floor_cap_and_no_stop_once_armed():
    up = _P(TE0, [(2, 1.3, 1.3), (10, 2.0, 2.0), (20, 3.6, 3.6)])
    assert S.walk_v2(up, 3.5) == {"why": "TP", "pct": 3.5, "armed": True}
    fl = _P(TE0, [(2, 1.3, 1.3), (10, 0.95, 0.95)])
    w = S.walk_v2(fl, 3.5)
    assert w["why"] == "FLOOR" and abs(w["pct"] - 0.95) < 1e-12
    # armed then a crash far below the stop on a 1m bar: books the floor / the bar open — never the −4 stop
    crash = dict(t=np.array([TE0 + 2 * MIN, TE0 + 10 * MIN]), hi=np.array([1.4, 0.5]), lo=np.array([1.2, -5.0]), cl=np.array([1.3, -4.8]),
                 op=np.array([1.25, 0.6]), bar=np.array([True, True]), end_ms=TE0 + 100 * MIN, te=TE0)
    w = S.walk_v2(crash, 3.5)
    assert w["why"] == "FLOOR" and abs(w["pct"] - 0.6) < 1e-12
    # armed: the late TP no longer applies (35 min at +1.1 > late +0.25 → still open), the cap closes
    late = _P(TE0, [(2, 1.3, 1.3), (35, 1.1, 1.1), (60, 1.2, 1.2)])
    assert S.walk_v2(late, 3.5)["why"] == "CAP"
    # not armed (never reached +1.25): the late TP fires as today
    nl = _P(TE0, [(2, 0.8, 0.8), (35, 0.4, 0.4)])
    assert S.walk_v2(nl, 3.5)["why"] == "LATE_TP"
    # not armed: the −4 stop as today
    st = _P(TE0, [(2, 0.5, 0.5), (10, -4.1, -4.1)])
    assert S.walk_v2(st, 3.5)["why"] == "STOP"


def _exports(rows):
    df = pd.DataFrame(rows)
    df["_m"] = 1.0
    df["_batch"] = "scalpars_orders_paper_2026-10-30_00-00-00.csv"
    return df


def _row(i, day, tp_t, pnl, reason, peak, slope=0.1, es="FRENZY_WILLY", status="CLOSED", armed_at=None):
    return dict(opened_at=f"2026-10-{day:02d} {10 + i % 10:02d}:{i % 60:02d}:00", closed_at="x", pair=f"P{i}USDT", direction="LONG",
                entry_strategy=es, status=status, entry_price=1.0, pnl_percentage=pnl, close_reason=reason, peak_pnl=peak,
                entry_btc_1h_slope=slope, frenzy_willy_tp_target_pct=tp_t, willy_floor_armed_at=armed_at, entry_frenzy_willy_spike_high=1.05)


def _const_paths(shadow_pct):
    """path_fn: a path on which the OLD exit books exactly `shadow_pct` per pair (dict pair → pct) and the V2 walk mirrors the booked."""
    def fn(r):
        v = shadow_pct[r["pair"]]
        tp = r.get("frenzy_willy_tp_target_pct")
        live = r["pnl_percentage"]
        if v == 1.25 and tp and tp > 1.25:   # runner: +1.3 (arm, shadow books +1.25) then the booked exit
            ev = [(1, 1.3, 1.3), (5, live, live)] if live >= tp - 1e-9 else [(1, 1.3, 1.3), (5, live, live)]
        else:
            ev = [(1, v, v)]
        return _P(TE0, ev)
    return fn


def test_run_ledger_parity_era_and_frozen_gates(tmp_path):
    rows, sh = [], {}
    for i in range(16):   # 16 armed runner fills on 8 days: 8 runners at +3.5, 8 floor exits at +1.0
        tp_t = 3.5
        if i % 2 == 0:
            rows.append(_row(i, 20 + i // 2, tp_t, 3.5, "FRENZY_TP", 3.6, slope=(-0.1 if i % 4 == 0 else 0.2)))
        else:
            rows.append(_row(i, 20 + i // 2, tp_t, 1.0, "WILLY_FLOOR", 1.4, slope=0.2))
        sh[f"P{i}USDT"] = 1.25
    rows.append(_row(30, 29, 1.25, 1.25, "FRENZY_TP", 1.3))   # V2 era, target 1.25: old exit exactly (not armed)
    sh["P30USDT"] = 1.25
    rows.append(_row(31, 29, None, -4.0, "STOP_LOSS", 0.2))   # pre-V2 (no stamp) — inside the counted window by time
    sh["P31USDT"] = -4.0
    rows.append(_row(32, 29, 3.0, 0.0, "MANUAL", 0.5))          # operator-closed → out
    sh["P32USDT"] = 0.0
    state = str(tmp_path / "g.json")
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=0, path_fn=_const_paths(sh), fetch=False, state_path=state))
    assert "exit era (WILLY + WILLY_HT, all closed in the exports): pre-V2 1 · V2 18 (runner target / floor 17)" in out
    assert "operator-closed: 1" in out
    assert "armed: 16" in out
    assert "parity (V2 re-walk vs booked, ±0.15): 18/18 = 100 %" in out
    # ledger A: runners +3.5 − 1.25 = +2.25 each ×8 = +18 · floor 1.0 − 1.25 = −0.25 ×8 = −2 → Σ +16
    assert "Ledger A (target + floor): 16 armed priced fills · Σ(live − shadow) +16.00 pts" in out
    assert "runners (TP above +1.25): 8" in out and "floor exits (WILLY_FLOOR): 8" in out
    assert "| ALL | falling |" in out and "| ALL | rising |" in out
    assert "EARLY: first 10 armed" in out and "no alert" in out
    assert "MAIN: main read (15 armed on 8 days): Σ(live − shadow) +16.25 pts → PASS (keep V2)" in out
    st = json.load(open(state))
    assert {"EARLY", "MAIN"} <= set(st) and st["MAIN"]["verdict"] == "PASS" and "FINAL" not in st
    assert set(st) - {"EARLY", "MAIN"} == {f"LADDER_{k}" for k in S.LADDERS}                      # 🪜 (357) ladder reads freeze at 15 armed too
    # frozen: a later run with a terrible tape keeps the decided verdicts
    rows2 = [dict(r, pnl_percentage=(-4.0 if r["frenzy_willy_tp_target_pct"] == 3.5 else r["pnl_percentage"])) for r in rows]
    out2 = "\n".join(S.run(raw=_exports(rows2), floor_ms=0, path_fn=_const_paths(sh), fetch=False, state_path=state))
    assert "no verdict (parity)" in out2 or "(frozen " in out2


def test_main_read_reverts_and_early_alert(tmp_path):
    rows, sh = [], {}
    for i in range(15):   # every armed fill gave back to the floor: 15 × (1.0 − 1.25) = −3.75 < 0 → REVERT
        rows.append(_row(i, 20 + i // 2, 3.0, 1.0, "WILLY_FLOOR", 1.4))
        sh[f"P{i}USDT"] = 1.25
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=0, path_fn=_const_paths(sh), fetch=False, state_path=str(tmp_path / "g.json")))
    assert "MAIN: main read (15 armed on 8 days): Σ(live − shadow) -3.75 pts → 🛑 REVERT to +1.25" in out
    assert "EARLY: first 10 armed: live +1.000 vs shadow +1.250 %/fill (alert if < shadow − 0.15) → ⚠ ALERT: review WILLY exit V2" in out


def test_parity_fail_blocks_freeze(tmp_path):
    rows, sh = [], {}
    for i in range(12):
        rows.append(_row(i, 20 + i, 3.0, 2.0, "FRENZY_TP", 3.1))   # booked +2.0 but the path says +3.0 TP → parity fails
        sh[f"P{i}USDT"] = 1.25

    def fn(r):
        return _P(TE0, [(1, 1.3, 1.3), (5, 3.1, 3.1)])
    state = str(tmp_path / "g.json")
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=0, path_fn=fn, fetch=False, state_path=state))
    assert "→ no verdict (parity)" in out and "EARLY: no verdict (parity)" in out
    assert not os.path.exists(state)


def test_not_evaluable_without_deploy():
    out = "\n".join(S.run(raw=_exports([_row(1, 20, 3.0, 1.0, "WILLY_FLOOR", 1.4)]), floor_ms=None, fetch=False, state_path=None))
    assert "NOT evaluable yet" in out and "exit era" in out


def test_dedup_closed_wins_and_ht_counted():
    a = _row(1, 20, 3.0, 1.0, "WILLY_FLOOR", 1.4, es="FRENZY_WILLY_HT")
    b = dict(a, status="OPEN", pnl_percentage=None)
    raw = pd.DataFrame([dict(b, _m=2.0, _batch="new.csv"), dict(a, _m=1.0, _batch="old.csv")])
    o = S.dedup(raw)
    assert len(o) == 1 and o.iloc[0].status == "CLOSED" and o.iloc[0].batch == "old.csv"


def test_wired_into_opportunity_scout():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_exit_v2 as _wv2" in src and "_rg_sec += _wv2.run(now_ms)" in src


def test_ladder_walk_raises_the_floor_from_earlier_peaks_only():
    """🪜 (357) INIT 10-23 shape: arm, peak +2.85, fall → flat books +1.0; TRAIL10 books +1.85; STEPS +1.5; the peak event itself never sets its own floor."""
    te = 1_760_000_000_000
    P = _P(te, [(1, 1.3, 1.3), (2, 2.85, 2.85), (3, 0.9, 0.9), (20, 7.5, 7.5)])
    assert S.walk_v2(P, 7.42)["pct"] == 0.9                                                       # flat: the 0.9 print (tick)
    w = S.walk_v2(P, 7.42, floor_fn=S.LADDERS["TRAIL10"])
    assert w["why"] == "FLOOR" and w["pct"] == 0.9                                               # a tick books the print below the floor
    P2 = _P(te, [(1, 1.3, 1.3), (2, 2.85, 2.85), (3, 1.9, 1.6), (20, 7.5, 7.5)], bar=True)
    P2["op"] = np.array([1.3, 2.85, 1.9, 7.5])
    assert S.walk_v2(P2, 7.42, floor_fn=S.LADDERS["TRAIL10"])["pct"] == 1.85                     # bar: min(floor 1.85, open 1.9)
    assert S.walk_v2(P2, 7.42, floor_fn=S.LADDERS["STEPS"])["pct"] == 7.42                      # STEPS floor +1.5 < the 1.6 low → rides to TP
    assert S.walk_v2(P2, 7.42)["why"] == "TP"                                                     # flat +1.0 rides the 1.6 dip to the target
    one = _P(te, [(1, 1.3, 1.3), (2, 3.0, 1.0)], bar=True)
    one["op"] = np.array([1.3, 2.0])
    assert S.walk_v2(one, 7.42, floor_fn=S.LADDERS["TRAIL10"])["pct"] == 1.0                     # the bar setting the 3.0 peak judges the OLD floor


def test_ladder_reads_candidate_and_keep_flat():
    import pandas as pd
    ts = pd.to_datetime([f"2026-11-{1 + i // 2:02d}T10:{i:02d}:00" for i in range(16)])
    base = pd.DataFrame(dict(ts=ts, pair=[f"P{i}" for i in range(16)], armed=True, v2=1.0))
    for k in S.LADDERS:
        base[f"lad_{k}"] = 1.0; base[f"ladd_{k}"] = 0.0
    base["ladd_TRAIL10"] = 0.2; base["lad_TRAIL10"] = 1.2
    r = S.ladder_reads(base)
    assert r["LADDER_TRAIL10"][0] == "CANDIDATE" and r["LADDER_STEPS"][0] == "KEEP_FLAT"
    one = base.copy(); one["ladd_HALF"] = [5.0] + [0.0] * 15
    assert S.ladder_reads(one)["LADDER_HALF"][0] == "KEEP_FLAT"                                   # one fill carries the gain
