"""🔂 DECISION_LOG 534 — the three momentum-long watch lines re-based on today's exit incl. the second chance (scripts/ml_sc_history.py)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import ml_sc_history as SCH                 # noqa: E402
import observe_registry as OR               # noqa: E402

SHIP = int(pd.Timestamp("2026-11-11 20:00").value // 10**6)


def test_rebase_rule():
    delta = {"2026-10-01T10:00:00|AAAUSDT": (0.5, 0.42), "2026-10-01T11:00:00|BBBUSDT": (0.0, 0.3)}
    pct = [0.4, 0.3, 0.6, -0.7, 0.8, 0.2]
    o = ["2026-10-01 10:00:00.123", "2026-10-01T11:00:00", "2026-10-02T09:00:00", "2026-10-02T10:00:00", "2026-10-02T11:00:00", "2026-11-12T01:00:00"]
    pr = ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT"]
    adx = [-1.0, -0.5, -0.2, -0.3, 0.4, -1.0]
    cr = ["RUNNER_TRAIL", "HARD_TP_LADDER L2", "RH_RUNNER_TRAIL", "STOP_LOSS", "RUNNER_TRAIL", "RUNNER_TRAIL"]
    p, n_ch, n_un, n_dis = SCH.rebase(pct, o, pr, adx, cr, SHIP, delta)
    assert abs(p[0] - 0.9) < 1e-12                  # pre-ship, delta row, replica reproduced the booked exit → booked + delta
    assert p[1] == 0.3                              # delta 0 → unchanged, not counted as changed
    assert np.isnan(p[2])                           # in zone + runner exit, no delta row → UNPRICED (never silently booked)
    assert p[3] == -0.7 and p[4] == 0.8             # a stop / ADX rising → the second chance could not touch it
    assert p[5] == 0.2                              # opened after the ship → booked (the engine ran it)
    assert (n_ch, n_un, n_dis) == (1, 1, 0)


def test_replica_disagreeing_with_the_booked_exit_gets_no_credit():
    # review 534: booked at its stop (−1.10), the replica ran it to a runner (+0.33) → its +delta must NOT be added
    p, n_ch, n_un, n_dis = SCH.rebase([-1.10], ["2026-10-02T01:57:30"], ["ZROUSDT"], [-0.5], ["STOP_LOSS_WIDE L1"], SHIP,
                                      {"2026-10-02T01:57:30|ZROUSDT": (0.8, 0.33)})
    assert p[0] == -1.10 and (n_ch, n_un, n_dis) == (0, 0, 1)


def test_rebase_missing_columns_fail_closed_and_no_ship_is_not_evaluable():
    p, n_ch, n_un, n_dis = SCH.rebase([0.4], ["2026-10-01T10:00:00"], ["ZZZUSDT"], None, None, SHIP, {})
    assert p[0] == 0.4 and (n_ch, n_un, n_dis) == (0, 0, 0)  # no ADX stamp → never in the zone
    p, *n = SCH.rebase([0.4], ["2026-10-01T10:00:00"], ["AAAUSDT"], [-1.0], ["RUNNER_TRAIL"], None, {"2026-10-01T10:00:00|AAAUSDT": (0.5, 0.4)})
    assert p[0] == 0.4 and n == [0, 0, 0]                   # DL 532 not in git → no re-base at all (never "every fill pre-ship")
    import scout_ml_fixed_tp_watch as FT
    G = FT.sc_rebase(pd.DataFrame(dict(opened_at=["2026-10-01T10:00:00"], pair=["AAAUSDT"], before=[0.4], close_reason=["RUNNER_TRAIL"])), "before", None, {})
    assert G.attrs["sc"][3] is False and "NOT EVALUABLE" in FT.sc_line(G.attrs["sc"])


def test_frozen_delta_file_matches_the_532_study():
    d = pd.read_csv(SCH.DELTA)
    assert len(d) == 123 and int((d.delta.abs() > 1e-12).sum()) == 38
    assert abs(d.delta.sum() - 11.774) < 0.01      # DL 532: master +33.1 → +44.9 (Δ +11.77)
    assert d.sc_reason[d.delta.abs() > 1e-12].str.startswith("SC_").all()


def test_registry_lines_pinned_to_the_shipped_knobs_and_not_stale():
    th = OR.thresholds()
    for s in ("scout_ml_fixed_tp_watch.py", "scout_ml_matched_refusals.py"):
        w = OR.EXIT[s]["walks"]
        assert all(w[k] == v for k, v in SCH.KNOBS.items()), s
        assert not OR.diffs(w, th), s
    w = OR.BOOKED["scout_ml_lead_a_watch.py"]
    assert all(w[k] == v for k, v in SCH.KNOBS.items()) and not OR.diffs(w, th)
    moved = dict(th, ml_second_chance_floor_pct=0.4)
    assert OR.diffs(OR.EXIT["scout_ml_fixed_tp_watch.py"]["walks"], moved)        # a knob change → ⚠ STALE again


def test_fixed_tp_sc_rebase_before_drops_unpriced_and_counts():
    import scout_ml_fixed_tp_watch as FT
    F = pd.DataFrame(dict(opened_at=["2026-10-01T10:00:00", "2026-10-02T09:00:00", "2026-11-12T01:00:00"], pair=["AAAUSDT", "CCCUSDT", "FFFUSDT"],
                          before=[0.4, 0.6, 0.2], entry_adx_delta=[-1.0, -0.2, -1.0], close_reason=["RUNNER_TRAIL", "RUNNER_TRAIL", "RUNNER_TRAIL"]))
    G = FT.sc_rebase(F, "before", SHIP, {"2026-10-01T10:00:00|AAAUSDT": (0.5, 0.4)})
    assert list(G.pair) == ["AAAUSDT", "FFFUSDT"] and abs(G.before.iloc[0] - 0.9) < 1e-12 and G.attrs["sc"] == (1, 1, 0, True)
    assert F.attrs == {}                                    # the caller's frame is never touched


def test_matched_refusals_rows_carry_adx_and_fresh_store():
    import scout_ml_matched_refusals as MM
    assert MM.STORE.endswith("SCOUT_ML_MATCHED_REFUSALS_SC.csv")
    R = pd.DataFrame(dict(o_ms=[SHIP], mr_entry_fields=['{"entry_atr_pct": 0.5, "entry_adx_delta": -0.4}'], price=[1.0], mr_post_gate=[""],
                          mr_labels=["W2"], mr_btc_regime=["STRONG_BULL"], pair=["AAAUSDT"]))
    S, _ = MM.shadow_cohort(R, SHIP - 1)
    assert S.adx.iloc[0] == -0.4
    assert "entry_adx_delta" in MM.ORD_COLS


def test_apply_second_chance_fails_closed_without_adx_or_knobs():
    import scout_ml_matched_refusals as MM
    base = (5, "RUNNER_TRAIL", 0.4, 1_000, "1m")
    prm = dict(enabled=True, adx_delta_max=0.0, floor=0.3, trail_mult=1.0, clock_min=180.0)
    assert MM.apply_second_chance(None, {}, base, None, prm) == base           # no ADX stamp → today's BASE price
    assert MM.apply_second_chance(None, {}, base, float("nan"), prm) == base
    assert MM.apply_second_chance(None, {}, base, -1.0, None) == base          # knobs off / unreadable
    assert MM.apply_second_chance(None, {}, base, 0.5, prm) == base            # ADX rising → out of zone
    assert MM.apply_second_chance(None, {}, (5, "STOP_LOSS", -0.7, 1_000, "1m"), -1.0, prm)[1] == "STOP_LOSS"   # not a runner / ladder


def test_matched_refusals_prices_nothing_new_when_the_knobs_are_off(monkeypatch, tmp_path):
    import scout_ml_matched_refusals as MM
    monkeypatch.setattr(MM, "_live_sc_params", lambda: None)
    called = []
    monkeypatch.setattr(MM, "price_replica", lambda fills, now: called.append(1) or {})
    monkeypatch.setattr(MM, "_rp", lambda: None)
    R = pd.DataFrame(dict(t=[SHIP], e=["ML_MATCHED"], o_ms=[SHIP], mr_entry_fields=['{"entry_adx_delta": -0.4}'], price=[1.0], mr_post_gate=[""],
                          mr_labels=["W2"], mr_btc_regime=["STRONG_BULL"], pair=["AAAUSDT"], mr_family=["W"]))
    out = "\n".join(MM.run(now_ms=SHIP + 10**8, refusals=R, orders=pd.DataFrame(), floor_ms=SHIP - 1, fetch=False,
                           state_path=str(tmp_path / "s.json"), store_path=None))
    assert not called and "NOT priced this run" in out
