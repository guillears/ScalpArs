"""🌊 master pool: BULLRUN_LONG screened by today's full Bull-Run gate chain (DECISION_LOG 518, 11-10).

Before: the builder applied only the pair blacklist → stack_keep kept 99 v2 fills where today's engine admits 24 (DL 104).
Pins: the frozen thresholds equal trading_config.json; each gate in engine order on synthetic rows; the 24-of-v2 count on the pool."""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import build_master_pool as B  # noqa: E402

BASE = dict(entry_br_door="GREEN", entry_br_r72=12.0, entry_br_eff=0.12, entry_br_off24h=-0.5, entry_btc_dist_from_ema13_pct=0.05,
            entry_btc_1h_slope=0.1, entry_bull_pct=55.0, entry_bear_pct=10.0, entry_pair_volume_ratio=0.9, entry_slippage_pct=0.02,
            opened_at="2026-09-20T17:00:00", era="B10")


def test_frozen_thresholds_match_live_config():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    # the door approximation assumes these (caveman review 518) — a retune must revisit bullrun_stack_block
    assert (th["bullrun_green_r72_off"], th["bullrun_green_eff_on"], th["bullrun_green_above_on"], th["bullrun_rearm_flap_merge_min"]) == (8.0, 0.1, 56.0, 30.0)
    assert th["spike_chase_enabled"] is B.SPIKE_CHASE_ENABLED_FROZEN and th["spike_bounce_enabled"] is B.SPIKE_BOUNCE_ENABLED_FROZEN
    for k, v in B.BULLRUN_STACK_FROZEN.items():
        assert th[k] == v, f"{k}: live {th[k]} vs frozen {v} — update BULLRUN_STACK_FROZEN + bump STACK_VERSION, rebuild the pool"


def test_clean_row_admitted_and_each_gate_in_engine_order():
    assert B.bullrun_stack_block(BASE) == ""
    cases = [(dict(entry_br_r72=9.0), "BULLRUN_DOOR_GREEN"), (dict(entry_br_eff=0.09), "BULLRUN_DOOR_GREEN"),
             (dict(entry_br_off24h=-2.0), "BULLRUN_BTC_OFF24H"), (dict(entry_btc_dist_from_ema13_pct=-0.01), "BULLRUN_BTC_EMA13"),
             (dict(entry_btc_1h_slope=0.0), "BULLRUN_BTC_1H_SLOPE"), (dict(entry_bull_pct=39.9), "BULLRUN_BREADTH"),
             (dict(entry_pair_volume_ratio=1.21), "BULLRUN_PVR_MAX"), (dict(entry_slippage_pct=-0.31), "BULLRUN_DISLOC"),
             (dict(entry_br_above=52.9), "BULLRUN_DOOR_GREEN"), (dict(opened_at="2026-08-21T17:06:00"), "BULLRUN_V1")]
    for over, why in cases:
        assert B.bullrun_stack_block({**BASE, **over}) == why, (over, why)
    assert B.bullrun_stack_block({**BASE, "entry_slippage_pct": 0.3}) == ""        # engine: strict > (bullrun_disloc_exceeded)
    # first gate wins (engine order): a door miss AND a breadth miss → the door
    assert B.bullrun_stack_block({**BASE, "entry_br_r72": 9.0, "entry_bull_pct": 10.0}) == "BULLRUN_DOOR_GREEN"


NO_GREEN_CHECK = ([], None)   # cache span None → the today-GREEN check fails open; the door / age legs are tested on their own


def test_rearm_needs_a_today_green_within_48h():
    t0, t1 = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")
    ends = ([pd.Timestamp("2026-09-19T10:00")], (t0, t1))
    rearm = {**BASE, "entry_br_door": "REARM", "entry_br_door_age_min": 10.0}
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-20T09:00:00"}, green_ends=ends) == ""                      # 23 h after
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-21T11:00:00"}, green_ends=ends) == "BULLRUN_REARM_NO_GREEN"  # 49 h after
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-21T11:00:00"}, green_ends=([], None)) == ""               # no cache → fail-open
    assert B.bullrun_stack_block({**BASE, "opened_at": "2026-09-21T11:00:00"}, green_ends=ends) == ""                      # GREEN never checked


def test_green_rebuild_on_a_synthetic_run(tmp_path):
    # a straight run-up long enough for r72 ≥ 10 / above ≥ 56 / eff ≥ 0.10, then a steady slide → exactly one GREEN episode that ends
    n = 864 * 3
    top = 100.0 * (1 + 0.0002) ** n
    px = [100.0 * (1 + 0.0002) ** i for i in range(n)] + [top * (1 - 0.0003) ** i for i in range(1, 1500)]
    t0 = 1767052800000
    syn = tmp_path / "btc5m.csv"
    pd.DataFrame({"open_time": [t0 + i * 300000 for i in range(len(px))], "c": px}).to_csv(syn, index=False)
    ends, span = B.bullrun_green_ends(str(syn))
    assert len(ends) == 1 and span is not None
    turn = pd.to_datetime(t0 + n * 300000, unit="ms")
    assert turn < ends[0] < turn + pd.Timedelta(hours=72)          # ends during the slide, not before the top


def test_door_bands_pinned_to_live_config():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert B.BULLRUN_GREEN_ON == (th["bullrun_green_r72_on"], th["bullrun_green_above_on"], th["bullrun_green_eff_on"])
    assert B.BULLRUN_GREEN_OFF == (th["bullrun_green_r72_off"], th["bullrun_green_above_off"], th["bullrun_green_eff_off"])
    assert B.BULLRUN_REARM_AFTER_GREEN_H == th["bullrun_rearm_after_green_hours"]


def test_rearm_window_measured_from_the_door_open():
    ends = ([pd.Timestamp("2026-09-19T10:00")], (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")))
    late = {**BASE, "entry_br_door": "REARM", "opened_at": "2026-09-21T11:00:00", "entry_br_door_age_min": 120.0}   # door opened 09:00, 47 h after
    assert B.bullrun_stack_block({**late, "entry_br_door_age_min": 55.0}, green_ends=ends) == "BULLRUN_REARM_NO_GREEN"   # door 10:05 = 48.1 h
    assert B.bullrun_door_open_at(late) == pd.Timestamp("2026-09-21T09:00")
    assert B.bullrun_rearm_has_green(B.bullrun_door_open_at(late), ends) is True


def test_rearm_door_skips_green_legs_and_ages_from_the_log_episodes(monkeypatch):
    monkeypatch.setattr(B, "bullrun_green_ends", lambda *a, **k: NO_GREEN_CHECK)
    rearm = {**BASE, "entry_br_door": "REARM", "entry_br_r72": 1.0, "entry_br_off24h": -5.0}
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-21T09:30:00"}) == ""                    # 48 min into episode #44
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-21T14:29:00"}) == "BULLRUN_REARM_STALE"  # 347 min (DL 104's burst)
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-21T14:29:00", "entry_br_door_age_min": 20.0}) == ""   # the stamp wins
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-08-25T10:00:00"}) == ""                    # no known episode → fail-open
    assert B.bullrun_door_age({"opened_at": "2026-09-20T19:00:00"}) is None                              # between #40's end and #42
    assert abs(B.bullrun_door_age({"opened_at": "2026-09-20T17:13:14"}) - 60.33) < 0.01                  # ENA: seconds decide it
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-20T17:13:14"}) == "BULLRUN_REARM_STALE"
    assert B.bullrun_door({"entry_br_door": None, "era": "B4"}) == "REARM" and B.bullrun_door({"era": "B3"}) == "GREEN"
    # REARM warm-up (bull = bear = 0) fails CLOSED, GREEN fails open — the engine's bullrun_breadth_ok split
    assert B.bullrun_stack_block({**rearm, "opened_at": "2026-09-21T09:00:00", "entry_bull_pct": 0, "entry_bear_pct": 0}) == "BULLRUN_BREADTH"
    assert B.bullrun_stack_block({**BASE, "entry_bull_pct": 0, "entry_bear_pct": 0}) == ""


def test_missing_stamps_fail_open(monkeypatch):
    monkeypatch.setattr(B, "bullrun_green_ends", lambda *a, **k: NO_GREEN_CHECK)
    row = {k: None for k in BASE}
    row.update(opened_at="2026-09-20T17:00:00", era="B10")
    assert B.bullrun_stack_block(row) == ""
    assert B.bullrun_stack_block({**row, "entry_br_door": "REARM", "opened_at": "2026-09-21T09:00:00"}) == "BULLRUN_BREADTH"   # REARM: fail-closed


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")), reason="pool not built")
def test_pool_keeps_dl104_cohort():
    m = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    b = m[m.entry_strategy == "BULLRUN_LONG"]                       # ALL eras: the v1 floor is inside the gate chain (deep review 518)
    k = b[b.stack_keep.astype(bool)]
    assert len(k) == 10 and round(k.pnl.sum()) == 2426 and (k.opened_at >= "2026-08-21T19:16").all()   # DL 521: GREEN only
    assert (b.loc[b.entry_br_door.eq("REARM"), "stack_block_reason"] != "").all()
    assert not m[m.entry_strategy.isin(["SPIKE_BOUNCE", "SPIKE_CHASE"]) & ~m.is_probe.fillna(False).astype(bool)].stack_keep.astype(bool).any()
    assert set(b.loc[~b.stack_keep.astype(bool), "stack_block_reason"]) <= {
        "SLEEVE_PAIR_BLACKLIST", "BULLRUN_DOOR_GREEN", "BULLRUN_REARM_STALE", "BULLRUN_BTC_1H_SLOPE", "BULLRUN_BTC_EMA13",
        "BULLRUN_BREADTH", "BULLRUN_PVR_MAX", "BULLRUN_DISLOC", "BULLRUN_BTC_OFF24H", "BULLRUN_V1", "BULLRUN_REARM_NO_GREEN"}
