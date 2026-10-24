"""🎲²ᵇ tests for scripts/scout_willy_exit_v2.py (DECISION_LOG 358 — V2b, the floor removed) — synthetic paths / exports, no network,
state in tmp."""
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
DEPLOY = pd.Timestamp("2026-10-24 00:00:00")                 # the DL 358 deploy + 10 min (the counted floor) in these tests
FLOOR_MS = int(DEPLOY.value // 1_000_000)


def _P(te, ev, bar=False, cap=90):
    return S._P(te, ev, cap=cap, bar=bar)


def test_selftest(capsys):
    assert S.selftest() and "selftest WILLY_EXIT_V2b OK" in capsys.readouterr().out


def test_pinned_constants_and_fresh_state():
    assert (S.OLD_TP, S.OLD_STOP, S.OLD_CAP, S.OLD_LATE_MIN, S.OLD_LATE_TP) == (1.25, 4.0, 90, 30.0, 0.25)
    assert (S.DL, S.DL_V2, S.LAG_MIN) == (358, 356, 10)
    assert (S.ARM, S.FLOOR, S.FLOOR_SLIP) == (1.25, 1.0, 0.05)
    assert (S.EARLY_N, S.EARLY_GAP, S.MAIN_N, S.MAIN_DAYS, S.FINAL_N) == (10, 0.15, 15, 6, 25)
    assert (S.SH_N, S.SH_DAYS, S.SH_MIN_GAIN, S.SH_SHARE) == (15, 6, 0.10, 0.50)
    assert S.SHADOWS == ("FLOOR10", "STEPS", "TRAIL10", "TRAIL15", "HALF", "NOLATE", "TGT15", "TGT20", "LATE45", "LATE60")
    assert S.LATE_MINS == {"LATE45": 45.0, "LATE60": 60.0}
    assert S.TARGET_KS == {"TGT15": 1.5, "TGT20": 2.0} and (S.LIVE_K, S.TGT_MAX, S.MAKER_FEE_DEF) == (1.0, 8.0, 0.00018)
    assert {"entry_fee", "quantity"} <= set(S.COLS)
    assert S.STATE.endswith(os.path.join("reports", ".willy_exit_v2b_gate.json")) and S.KEY == "V2B_"
    assert set(WEra.V2_COLS) <= set(S.COLS) and "close_reason" in S.COLS and "closed_at" in S.COLS


# ─────────────────────────── walkers ───────────────────────────
def test_walk_v2_default_is_the_old_exit_at_the_target():
    rng = np.random.default_rng(7)
    for k in range(200):
        n = 40
        mins = np.sort(rng.uniform(0.1, 95, n))
        x = np.cumsum(rng.normal(0, 0.6, n))
        P = _P(TE0, [(m, v, v) for m, v in zip(mins, x)])
        a, b = S.walk_v2(P, 1.25), S.shadow_today(P)
        assert a["why"] == b["why"], (k, a, b)
        assert (np.isnan(a["pct"]) and np.isnan(b["pct"])) or abs(a["pct"] - b["pct"]) < 1e-9
        assert not S.walk_v2(P, 3.5)["armed"] and S.walk_v2(P, 3.5)["why"] != "FLOOR"     # V2b: never a floor


def test_walk_v2b_runner_falls_to_the_stop_late_tp_and_cap():
    fall = _P(TE0, [(2, 1.3, 1.3), (5, 2.85, 2.85), (8, 1.0, 1.0), (12, -4.1, -4.1)])
    assert S.walk_v2(fall, 7.42) == {"why": "STOP", "pct": -4.1, "armed": False}
    late = _P(TE0, [(2, 1.3, 1.3), (35, 1.1, 1.1)])
    assert S.walk_v2(late, 3.5)["why"] == "LATE_TP"                                           # late TP for every trade
    cap = _P(TE0, [(2, 1.3, 1.3), (35, 0.1, 0.1), (60, 0.2, 0.2)])
    assert S.walk_v2(cap, 3.5)["why"] == "CAP"
    up = _P(TE0, [(2, 1.3, 1.3), (10, 2.0, 2.0), (20, 3.6, 3.6)])
    assert S.walk_v2(up, 3.5) == {"why": "TP", "pct": 3.5, "armed": False}


def test_shadow_walks_b_c_d():
    dip = _P(TE0, [(1, 1.3, 1.3), (2, 2.85, 2.85), (3, 0.9, 0.9), (20, 7.5, 7.5)])
    assert S.walk_v2(dip, 7.42)["why"] == "TP"                                                # live V2b rides the dip
    assert abs(S.shadow_walk(dip, 7.42, "FLOOR10") - (0.9 - S.FLOOR_SLIP)) < 1e-12           # (b) the DL 356 floor + slip
    assert abs(S.shadow_walk(dip, 7.42, "TRAIL10") - (0.9 - S.FLOOR_SLIP)) < 1e-12           # (c) a tick books the print
    assert S.shadow_walk(dip, 7.42, "NOLATE") == 7.42                                          # (d)
    lt = _P(TE0, [(2, 2.0, 2.0), (35, 0.3, 0.3), (60, 3.6, 3.6)])
    assert S.walk_v2(lt, 3.5)["why"] == "LATE_TP" and S.shadow_walk(lt, 3.5, "NOLATE") == 3.5
    assert S.shadow_walk(_P(TE0, [(1, 0.5, 0.5), (10, -4.2, -4.2)]), 3.5, "FLOOR10") == -4.2   # never armed: the stop, no slip


def test_ladder_walk_raises_the_floor_from_earlier_peaks_only():
    te = 1_760_000_000_000
    P2 = _P(te, [(1, 1.3, 1.3), (2, 2.85, 2.85), (3, 1.9, 1.6), (20, 7.5, 7.5)], bar=True)
    P2["op"] = np.array([1.3, 2.85, 1.9, 7.5])
    assert S.walk_v2(P2, 7.42, floor_fn=S.LADDERS["TRAIL10"])["pct"] == 1.85                     # bar: min(floor 1.85, open 1.9)
    assert S.walk_v2(P2, 7.42, floor_fn=S.LADDERS["STEPS"])["pct"] == 7.42
    assert S.walk_v2(P2, 7.42, floor=S.FLOOR)["why"] == "TP"
    one = _P(te, [(1, 1.3, 1.3), (2, 3.0, 1.0)], bar=True)
    one["op"] = np.array([1.3, 2.0])
    assert S.walk_v2(one, 7.42, floor_fn=S.LADDERS["TRAIL10"])["pct"] == 1.0


# ─────────────────────────── cohort / eras ───────────────────────────
def _exports(rows):
    df = pd.DataFrame(rows)
    df["_m"] = 1.0
    df["_batch"] = "scalpars_orders_paper_2026-11-05_00-00-00.csv"
    return df


def _row(pair, opened, tp_t, pnl, reason, peak=1.3, closed="x", slope=0.1, es="FRENZY_WILLY", status="CLOSED", armed_at=None):
    return dict(opened_at=opened, closed_at=closed, pair=pair, direction="LONG", entry_strategy=es, status=status, entry_price=1.0,
                pnl_percentage=pnl, close_reason=reason, peak_pnl=peak, entry_btc_1h_slope=slope, frenzy_willy_tp_target_pct=tp_t,
                willy_floor_armed_at=armed_at, entry_frenzy_willy_spike_high=1.05, entry_fee=0.4, quantity=1000.0)


TP_PATH = [(1, 1.3, 1.3), (5, 3.6, 3.6)]                          # V2b +3.5 TP · shadow (a) +1.25 · every floor shadow +3.5 (Δ 0)
STOP_PATH = [(1, 1.3, 1.3), (5, 0.9, 0.9), (10, -4.1, -4.1)]      # V2b −4.1 stop · (a) +1.25 · FLOOR10 / ladders 0.85 · NOLATE −4.1
INIT_PATH = [(1, 1.3, 1.3), (2, 2.85, 2.85), (3, 0.98, 0.98), (20, 7.5, 7.5)]   # floored (booked +1.06), then ran through +7.42


def _fn(paths):
    return lambda r: _P(TE0, paths[r["pair"]])


def _book(n_tp_first=None, stop_every=2, n=16):
    rows, paths = [], {}
    rows.append(_row("INITUSDT", "2026-10-23 13:00:00", 7.42, 1.06, "WILLY_FLOOR", peak=2.85, closed="2026-10-23 13:05:00",
                     armed_at="2026-10-23 13:01:00"))
    paths["INITUSDT"] = INIT_PATH
    rows.append(_row("TRNUSDT", "2026-10-23 23:40:00", 3.0, 3.0, "FRENZY_TP", closed="2026-10-24 00:20:00"))   # transition → not counted
    paths["TRNUSDT"] = TP_PATH
    for i in range(n):
        p = f"R{i}USDT"
        day = 25 + i // 2
        stop = (i % stop_every == 1) if stop_every else False
        rows.append(_row(p, f"2026-10-{day:02d} {10 + i % 2:02d}:00:00" if day <= 31 else f"2026-11-{day - 31:02d} {10 + i % 2:02d}:00:00",
                         3.5, (-4.1 if stop else 3.5), ("STOP_LOSS" if stop else "FRENZY_TP"), slope=(-0.1 if i % 4 == 0 else 0.2)))
        paths[p] = STOP_PATH if stop else TP_PATH
    rows.append(_row("BASEUSDT", "2026-11-04 12:00:00", 1.25, 1.25, "FRENZY_TP"))       # V2b, target +1.25: not a runner
    paths["BASEUSDT"] = [(1, 1.3, 1.3)]
    rows.append(_row("OPUSDT", "2026-11-04 13:00:00", 3.0, 0.0, "MANUAL"))              # operator-closed → out
    paths["OPUSDT"] = TP_PATH
    return rows, paths


def test_cohort_splits_v2_era_transition_and_v2b():
    rows, _ = _book()
    co = S.cohort(_exports(rows), FLOOR_MS)
    assert list(co["v2"].pair) == ["INITUSDT"] and list(co["trans"].pair) == ["TRNUSDT"]
    assert "INITUSDT" not in set(co["kept"].pair) and "TRNUSDT" not in set(co["kept"].pair)
    assert len(co["kept"]) == 17 and list(co["ox"].pair) == ["OPUSDT"]


def test_run_mixed_book_reverts_and_floor_shadows_become_candidates(tmp_path):
    rows, paths = _book()
    state = str(tmp_path / "g.json")
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=FLOOR_MS, path_fn=_fn(paths), fetch=False, state_path=state))
    assert "exit era (WILLY + WILLY_HT, all closed in the exports)" in out
    assert "operator-closed: 1" in out and "transition (opened before the counted floor = the DL 358 deploy + 10 min, closed after the deploy — not counted): 1" in out
    # the V2 (floor) era, apart: INIT floored at +1.06; V2b would have booked its +7.42 target; the +1.25 exit +1.25
    assert "V2 (floor, DL 356) era fills — closed before the DL 358 deploy, shown APART (never counted): 1 · floor exits (WILLY_FLOOR) 1" in out
    assert "booked avg +1.060 · what V2b would have booked +7.420 · today's +1.25 exit +1.250" in out
    assert "parity (V2-floor re-walk vs booked, ±0.15): 1/1 = 100 %" in out
    assert "runner fills (target > +1.25): 16" in out
    assert "parity (V2b re-walk vs booked, ±0.15): 17/17 = 100 %" in out
    # gates on runner fills: first 10 = 5 TP + 5 stops: live −0.300 vs shadow +1.250 → ALERT · 15 on 8 days: 8×2.25 + 7×(−5.35) = −19.45
    assert "V2B_EARLY: first 10 runner fills: live -0.300 vs shadow +1.250 %/fill (alert if < shadow − 0.15) → ⚠ ALERT: review WILLY exit V2b" in out
    assert "V2B_MAIN: main read (15 runner fills on 8 days): Σ(live − shadow) -19.45 pts → 🛑 REVERT to +1.25 (frenzy_willy_high_target_enabled false)" in out
    assert "V2B_FINAL: final read: 16/25 runner fills" in out
    # shadows: a stop fill saved by a floor at +0.85 → Δ +4.95 × 7 over 15 = +2.310; NOLATE = live (Δ 0) → KEEP
    for k in ("FLOOR10", "STEPS", "TRAIL10", "TRAIL15", "HALF"):
        assert f"V2B_SH_{k}: 15 runner fills on 8 days: avg Δ (shadow − live V2b) +2.310 %/fill · top fill 14% of the gain → CANDIDATE" in out, k
    assert "V2B_SH_NOLATE: 15 runner fills on 8 days: avg Δ (shadow − live V2b) +0.000 %/fill · top fill 100% of the gain → keep V2b" in out
    # (e) a lower target books less on the 8 TP fills (the 7 stops never reach either target) → KEEP
    for k in S.TARGET_KS:
        d = (S.target_at_k(rows[2], S.TARGET_KS[k]) - 3.5) * 8 / 15
        assert f"V2B_SH_{k}: 15 runner fills on 8 days: avg Δ (shadow − live V2b) {d:+.3f} %/fill" in out and d < 0, k
    assert "(e) TGT15: avg" in out and "(e) target recompute check (k 1 vs the stored target, ±0.02): 0/16 ⚠" in out   # fixture targets are hand-set
    assert "(b) FLOOR10: avg" in out and "(c) STEPS" in out and "(d) NOLATE" in out and "(a) TODAY +1.25 exit (the gate shadow)" in out
    assert out.count("per fill (last 20): ") == 1 and out.split("per fill (last 20): ")[1].split("\n")[0].count(" | a ") == 16
    assert "| ALL | falling |" in out and "| ALL | rising |" in out
    st = json.load(open(state))
    assert all(k.startswith("V2B_") for k in st) and st["V2B_MAIN"]["verdict"] == "REVERT" and "V2B_FINAL" not in st
    assert {f"V2B_SH_{k}" for k in S.SHADOWS} <= set(st) and st["V2B_SH_NOLATE"]["verdict"] == "KEEP"
    # frozen: a later all-TP book keeps the decided verdicts
    rows2, paths2 = _book(stop_every=0)
    out2 = "\n".join(S.run(raw=_exports(rows2), floor_ms=FLOOR_MS, path_fn=_fn(paths2), fetch=False, state_path=state))
    assert "V2B_MAIN: main read (15 runner fills on 8 days): Σ(live − shadow) -19.45 pts" in out2 and "(frozen " in out2


def test_run_all_runners_win_passes_and_shadows_keep(tmp_path):
    rows, paths = _book(stop_every=0, n=26)
    state = str(tmp_path / "g.json")
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=FLOOR_MS, path_fn=_fn(paths), fetch=False, state_path=state))
    assert "V2B_EARLY: first 10 runner fills: live +3.500 vs shadow +1.250" in out and "no alert" in out
    assert "V2B_MAIN: main read (15 runner fills on 8 days): Σ(live − shadow) +33.75 pts → PASS (keep V2b)" in out
    assert "V2B_FINAL: final read (first 25 runner fills): Σ(live − shadow) +56.25 pts → PASS (keep V2b)" in out
    for k in S.SHADOWS:
        if k in S.TARGET_KS:      # (e) every TP fill books the lower target instead of +3.5
            d = S.target_at_k(rows[2], S.TARGET_KS[k]) - 3.5
            assert f"V2B_SH_{k}: 15 runner fills on 8 days: avg Δ (shadow − live V2b) {d:+.3f}" in out and d < 0
        else:
            assert f"V2B_SH_{k}: 15 runner fills on 8 days: avg Δ (shadow − live V2b) +0.000" in out
    assert "keep V2b" in out


def test_parity_fail_blocks_freeze(tmp_path):
    rows, paths = _book(stop_every=0)
    rows = [dict(r, pnl_percentage=2.0) if str(r["pair"]).startswith("R") else r for r in rows]   # booked +2.0, the path says +3.5
    state = str(tmp_path / "g.json")
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=FLOOR_MS, path_fn=_fn(paths), fetch=False, state_path=state))
    assert "→ no verdict (parity)" in out and "V2B_EARLY: no verdict (parity)" in out and "V2B_SH_FLOOR10: no verdict (parity)" in out
    assert not os.path.exists(state)


def test_unpriced_runner_holds_the_reads(tmp_path):
    rows, paths = _book(stop_every=0)
    fn = _fn(paths)
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=FLOOR_MS, path_fn=lambda r: None if r["pair"] == "R3USDT" else fn(r),
                          fetch=False, state_path=str(tmp_path / "g.json")))
    assert "V2B_MAIN: main read sample (15): 1 unpriced — held" in out and "V2B_SH_FLOOR10: sample (15): 1 unpriced" in out


def test_not_evaluable_without_deploy():
    rows, _ = _book()
    out = "\n".join(S.run(raw=_exports(rows), floor_ms=None, fetch=False, state_path=None))
    assert "NOT evaluable yet — no DECISION_LOG 358 commit found" in out and "exit era" in out


def test_shadow_reads_candidate_keep_and_concentration():
    ts = pd.to_datetime([f"2026-11-{1 + i // 2:02d}T10:{i:02d}:00" for i in range(16)])
    base = pd.DataFrame(dict(ts=ts, pair=[f"P{i}" for i in range(16)], runner=True, v2b=1.0))
    for k in S.SHADOWS:
        base[f"sh_{k}"] = 1.0; base[f"d_{k}"] = 0.0
    base["d_TRAIL10"] = 0.2
    r = S.shadow_reads(base)
    assert r["V2B_SH_TRAIL10"][0] == "CANDIDATE" and r["V2B_SH_STEPS"][0] == "KEEP"
    one = base.copy(); one["d_NOLATE"] = [5.0] + [0.0] * 15
    assert S.shadow_reads(one)["V2B_SH_NOLATE"][0] == "KEEP"                                     # one fill carries the gain
    few = base.head(8)
    assert S.shadow_reads(few)["V2B_SH_FLOOR10"][0] == "COLLECTING"


def test_dedup_closed_wins_and_ht_counted():
    a = _row("XUSDT", "2026-10-25 10:00:00", 3.0, 1.0, "FRENZY_TP", es="FRENZY_WILLY_HT")
    b = dict(a, status="OPEN", pnl_percentage=None)
    raw = pd.DataFrame([dict(b, _m=2.0, _batch="new.csv"), dict(a, _m=1.0, _batch="old.csv")])
    o = S.dedup(raw)
    assert len(o) == 1 and o.iloc[0].status == "CLOSED" and o.iloc[0].batch == "old.csv"


def test_wired_into_opportunity_scout():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_willy_exit_v2 as _wv2" in src and "_rg_sec += _wv2.run(now_ms)" in src


def test_era_helper_still_classifies_v2b_and_floor_rows():
    df = pd.DataFrame([dict(frenzy_willy_tp_target_pct=None, close_reason="STOP_LOSS"),
                       dict(frenzy_willy_tp_target_pct=1.25, close_reason="FRENZY_TP"),          # V2b at +1.25: the old exit exactly
                       dict(frenzy_willy_tp_target_pct=3.5, close_reason="WILLY_LATE_TP"),       # V2b runner
                       dict(frenzy_willy_tp_target_pct=7.42, close_reason="WILLY_FLOOR")])       # V2-era floor row (INIT)
    assert WEra.counts(df) == (1, 3, 2)


def test_target_shadows_recompute_the_engine_target():
    """(e) TGT15 / TGT20 (operator 2026-10-23): the target at H × (1 − k %) on the engine's fee convention — the k 1.0 recompute reproduces a
    real stored target (AVAUSDT 10-24: entry 0.4544, H 0.4847, entry fee $3.2579 on $7,239.72 → stored +5.5374)."""
    ava = dict(entry_price=0.4544, entry_frenzy_willy_spike_high=0.4847, entry_fee=3.2579, quantity=7239.72 / 0.4544)
    assert S.target_at_k(ava, 1.0, maker=0.00018) == 5.5374
    assert S.target_at_k(ava, 1.5, maker=0.00018) == 5.0042 and S.target_at_k(ava, 2.0, maker=0.00018) == 4.471
    assert S.target_at_k(dict(ava, entry_frenzy_willy_spike_high=0.46), 2.0, maker=0.00018) == 1.25        # floor +1.25
    assert S.target_at_k(dict(ava, entry_frenzy_willy_spike_high=0.6), 1.5, maker=0.00018) == 8.0          # cap +8
    for bad in (dict(ava, entry_fee=None), dict(ava, quantity=0), dict(ava, entry_frenzy_willy_spike_high=None), {}):
        assert S.target_at_k(bad, 1.5) is None
    P = _P(TE0, [(1, 1.3, 1.3), (5, 5.2, 5.2)])                     # reaches +5.2: k 1.5 / 2.0 fill, the live +5.5374 does not
    assert S.shadow_walk(P, 5.5374, "TGT15", ava) == 5.0042 and S.shadow_walk(P, 5.5374, "TGT20", ava) == 4.471
    assert S.walk_v2(P, 5.5374)["why"] == "CAP"                                                         # the live +5.5374 never fills
    assert np.isnan(S.shadow_walk(P, 5.5374, "TGT15", None))                                            # unpriceable → unresolved


def test_late_minute_shadows():
    """(f) LATE45 / LATE60 (operator 2026-10-24): the +0.25 late line from minute 45 / 60 instead of 30, everything else live V2b."""
    p = _P(TE0, [(2, 1.3, 1.3), (35, 0.6, 0.6), (50, 0.7, 0.7), (55, 3.6, 3.6)])
    assert S.walk_v2(p, 3.5)["why"] == "LATE_TP" and abs(S.walk_v2(p, 3.5)["pct"] - 0.6) < 1e-9       # live: the first tick past 30 min books its print
    assert abs(S.shadow_walk(p, 3.5, "LATE45") - 0.7) < 1e-9                                            # 45: the 50-min print
    assert S.shadow_walk(p, 3.5, "LATE60") == 3.5                                                       # 60: rides to the +3.5 target at 55 min
    stop = _P(TE0, [(2, 1.3, 1.3), (35, 0.6, 0.6), (40, -4.1, -4.1)])
    assert S.walk_v2(stop, 3.5)["pct"] == 0.6 and S.shadow_walk(stop, 3.5, "LATE45") == -4.1           # waiting longer can also lose
