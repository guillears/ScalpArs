"""🌀👥 Oct-4 MOMENTUM-LONG CHOP ∧ BURST BLOCK (operator ARMED override, DECISION_LOG 201).

Rule: refuse a MOMENTUM long when BTC 72 h efficiency (entry_btc_eff72) ≤ long_chop_burst_eff72_max (0.007) AND another bot fill
(non-MANUAL, non-*_PROBE, this mode, open or closed) opened ≤ long_chop_burst_window_s (120) s before the decision.

Invariants pinned here:
  · switch off / threshold ≤ 0 / unreadable = off; boundaries inclusive; unknown eff72 or gap FAILS OPEN.
  · the prior-fill lookup ignores MANUAL rows, *_PROBE fires, the other mode and fills older than the 10-min stamp horizon.
  · engine gate: momentum-long guard identical to the heat / mega-cap guard; counter LONG_CHOP_BURST; stamp on the Order.
  · config parity: code default OFF, trading_config.json ON 0.007 / 120, builder freeze == JSON; UI input + load + save + report line.
  · master pool: exactly LIT Jul-10 (BASE) and WLD Oct-1 (B15) refused.
"""
import inspect
import json
import os
import re
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from services.trading_engine import long_chop_burst_block, chop_burst_prior_fill_s

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _th(on=True, emax=0.007, win=120.0):
    return SimpleNamespace(long_chop_burst_block_enabled=on, long_chop_burst_eff72_max=emax, long_chop_burst_window_s=win)


def test_blocks_only_chop_and_burst_inclusive():
    assert long_chop_burst_block(_th(), 0.006, 2) is True           # WLD Oct-1: eff 0.006, ENA 2 s earlier
    assert long_chop_burst_block(_th(), 0.007, 120) is True         # both boundaries inclusive
    assert long_chop_burst_block(_th(), 0.0, 0.0) is True           # eff 0.000 is the deepest chop, not "missing"
    assert long_chop_burst_block(_th(), 0.008, 2) is False          # not chop
    assert long_chop_burst_block(_th(), 0.003, 120.1) is False      # alone
    assert long_chop_burst_block(_th(), 0.04, 10) is False          # burst in a trending tape (net winners on the master)


def test_off_and_fail_open():
    assert long_chop_burst_block(_th(on=False), 0.001, 5) is False
    assert long_chop_burst_block(_th(emax=0), 0.0, 5) is False
    assert long_chop_burst_block(_th(win=0), 0.001, 0) is False
    assert long_chop_burst_block(SimpleNamespace(), 0.001, 5) is False
    for bad in (None, "", "x", float("nan"), float("inf"), -0.001):
        assert long_chop_burst_block(_th(), bad, 5) is False        # stale / unknown eff72 → no block
        assert long_chop_burst_block(_th(), 0.001, bad) is False    # unknown gap → no block
    assert long_chop_burst_block(_th(emax="0.007", win="120"), "0.005", "60") is True


@pytest.mark.asyncio
async def test_prior_fill_lookup_scope(db):
    from models import Order
    now = datetime(2026, 10, 4, 12, 0, 0)

    def o(sec_ago, strat="MOMENTUM", paper=True, cms=None, pcs=None, pair="AAAUSDT"):
        return Order(pair=pair, direction="LONG", status="CLOSED", entry_price=1.0, current_price=1.0, investment=10.0,
                     leverage=1, notional_value=10.0, quantity=10.0, confidence="STRONG_BUY", is_paper=paper, entry_strategy=strat,
                     cell_multiplier_source=cms, pattern_cell_source=pcs, opened_at=now - timedelta(seconds=sec_ago))
    assert await chop_burst_prior_fill_s(db, True, now) is None                     # empty book
    db.add_all([o(5, strat="MANUAL"), o(6, paper=False), o(7, cms="GAP_PROBE"), o(8, pcs="DEADBAND_PROBE"), o(700)])
    await db.commit()
    assert await chop_burst_prior_fill_s(db, True, now) is None                     # manual / live / probes / > 10 min all ignored
    db.add(o(90, strat="FLIP:FAN_RATIO_GATE", pair="BBBUSDT")); await db.commit()
    assert await chop_burst_prior_fill_s(db, True, now) == 90.0                     # any sleeve counts
    db.add(o(40, strat=None, pair="CCCUSDT", cms="UNMATCHED")); await db.commit()
    assert await chop_burst_prior_fill_s(db, True, now) == 40.0                     # NULL strategy = legacy momentum, counted; most recent wins
    assert await chop_burst_prior_fill_s(db, False, now) == 6.0                     # the other mode sees only its own rows


def _src():
    return open(os.path.join(ROOT, "services", "trading_engine.py")).read()


def _guard(src, anchor):
    i = src.index(anchor)
    start = src.rindex("if (", 0, i)
    end = src.index("):", i) + 2
    return src[start:end]


def test_engine_guard_matches_megacap_guard_and_counts():
    from services.trading_engine import TradingEngine
    src = inspect.getsource(TradingEngine.open_position)
    norm = lambda g: re.sub(r"\s+", " ", re.sub(r"\s*and long_(megacap|chop_burst)_block\([^)]*\)", "", g)).strip()
    mega = _guard(src, "and long_megacap_block(")
    cb = _guard(src, "and long_chop_burst_block(")
    assert norm(mega) == norm(cb)
    assert "long_chop_burst_block(config.trading_config.thresholds, _cb_eff, _cb_prior_s)" in cb
    i = src.index("and long_chop_burst_block(")
    assert src.index('self._record_filter_block("LONG_CHOP_BURST", "LONG")', i) < src.index("return None", i)
    assert src.index("and long_megacap_block(") < i                                 # last of the momentum-long gates
    # eff72 is read with the SAME freshness rule as the stamp; the stamp rides every momentum fill
    assert "_cb_eff = monitor_entry_stamps(_bullrun_monitor, _bearrun_monitor, _leash_time.time()).get('entry_btc_eff72')" in src
    assert "entry_chop_burst_prior_fill_s=(_cb_prior_s if _cb_momentum else None)" in src
    from services.trading_engine import pair_reason_stampable
    assert pair_reason_stampable("LONG_CHOP_BURST")                                  # Top Pairs block reason names it


def test_order_column_and_migration():
    import models
    assert "entry_chop_burst_prior_fill_s" in {c.name for c in models.Order.__table__.columns}
    dbsrc = open(os.path.join(ROOT, "database.py")).read()
    assert "ALTER TABLE orders ADD COLUMN entry_chop_burst_prior_fill_s FLOAT" in dbsrc


def test_config_parity_ui_and_builder_freeze():
    import config
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    assert th["long_chop_burst_block_enabled"] is True and th["long_chop_burst_eff72_max"] == 0.007 and th["long_chop_burst_window_s"] == 120.0
    f = config.SignalThresholds.model_fields
    assert f["long_chop_burst_block_enabled"].default is False                       # code default OFF, shipped via JSON
    assert f["long_chop_burst_eff72_max"].default == 0.007 and f["long_chop_burst_window_s"].default == 120.0
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    m = re.search(r"_CB_TH = SimpleNamespace\(long_chop_burst_block_enabled=True, long_chop_burst_eff72_max=([0-9.]+), long_chop_burst_window_s=([0-9.]+)\)", bld)
    assert m and float(m.group(1)) == th["long_chop_burst_eff72_max"] and float(m.group(2)) == th["long_chop_burst_window_s"]
    assert 'STACK_VERSION = "2026-10-04b"  # 10-04b: LONG_CHOP_BURST' in bld
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    for i in ("config-long-chop-burst-block-enabled", "config-long-chop-burst-eff72-max", "config-long-chop-burst-window-s"):
        assert ui.count(f'id="{i}"') == 1 and ui.count(i) == 3                         # input + load + save
    assert "Chop ∧ Burst Block (Oct 4)" in ui                                         # _buildConfigLines → both exports


def test_builder_pass_sequential_and_scoped():
    """chop_burst_pass on a synthetic era: 2nd+ fill in chop refused; a refused fill is no longer a neighbour; probes / MANUAL /
    blocked rows / other eras are not neighbours; non-chop bursts and sleeves are never refused."""
    from scripts.build_master_pool import chop_burst_pass
    rows = [  # (era, opened_at, strat, dir, probe, keep, eff)
        ("E1", "2026-07-10T17:00:00", "MOMENTUM", "LONG", False, True, 0.001),    # 0 first of the burst — kept
        ("E1", "2026-07-10T17:01:00", "MOMENTUM", "LONG", False, True, 0.001),    # 1 60 s after #0 → refused
        ("E1", "2026-07-10T17:02:30", "MOMENTUM", "LONG", False, True, 0.001),    # 2 150 s after #0, 90 s after REFUSED #1 → kept
        ("E1", "2026-07-10T17:03:00", "MOMENTUM", "LONG", False, True, 0.020),    # 3 30 s after #2 but trending → kept
        ("E1", "2026-07-10T18:00:00", "MOMENTUM", "LONG", True, True, 0.001),     # 4 probe — not a neighbour
        ("E1", "2026-07-10T18:00:30", "MANUAL", "LONG", False, False, 0.001),     # 5 manual — not a neighbour
        ("E1", "2026-07-10T18:01:00", "MOMENTUM", "LONG", False, True, 0.001),    # 6 only probe/manual before → kept
        ("E1", "2026-07-10T19:00:00", "SPIKE_FADE", "SHORT", False, False, 0.001),  # 7 stack-blocked fill — not a neighbour
        ("E1", "2026-07-10T19:00:40", "MOMENTUM", "LONG", False, True, 0.001),    # 8 → kept
        ("E1", "2026-07-10T20:00:00", "SPIKE_FADE", "SHORT", False, True, 0.001),  # 9 kept fade — IS a neighbour, never refused itself
        ("E1", "2026-07-10T20:02:00", "MOMENTUM", "LONG", False, True, 0.007),    # 10 exactly 120 s, eff at the max → refused
        ("E2", "2026-07-10T20:02:30", "MOMENTUM", "LONG", False, True, 0.001),    # 11 other era → kept
        ("E1", "2026-07-10T21:00:00", "MOMENTUM", "SHORT", False, True, 0.001),   # 12 momentum short (neighbour)
        ("E1", "2026-07-10T21:00:10", "MOMENTUM", "SHORT", False, True, 0.001),   # 13 shorts never refused
    ]
    df = pd.DataFrame(rows, columns=["era", "opened_at", "entry_strategy", "direction", "is_probe", "keep", "eff"])
    got = chop_burst_pass(df, list(df.keep), df.eff, _th(), long_chop_burst_block, 120.0)
    assert got == {1, 10}


def test_master_pool_refuses_exactly_lit_and_wld():
    p = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
    if not os.path.exists(p):
        pytest.skip("pool not built")
    d = pd.read_csv(p, low_memory=False, usecols=["pair", "opened_at", "stack_block_reason"])
    blk = d[d.stack_block_reason == "LONG_CHOP_BURST"]
    assert sorted(zip(blk.pair, blk.opened_at.astype(str).str[:19])) == [("LITUSDT", "2026-07-10T17:02:16"), ("WLDUSDT", "2026-10-01T01:25:05")]



def test_master_sleeve_sizes_frozen_match_live_json():
    """Review (Oct-4): the builder freezes today's sleeve sizes with STACK_VERSION — they must equal trading_config.json at ship time."""
    import json, re as _re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bld = open(os.path.join(root, "scripts", "build_master_pool.py"), encoding="utf-8").read()
    th = json.load(open(os.path.join(root, "trading_config.json")))["thresholds"]
    m = _re.search(r"_th = dict\(([^)]*)\)", bld, _re.S)
    frozen = {k: float(v) for k, v in _re.findall(r"(\w+)=([0-9.]+)", m.group(1))}
    for k, v in frozen.items():
        assert abs(float(th[k]) - v) < 1e-9, k
