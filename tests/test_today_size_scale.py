"""📏 TODAY's cell sizing — ONE rule, scripts/build_master_pool.today_size_rule (master STACK 2026-10-06c, DECISION_LOG 233), used by
the master's stack_pnl AND scripts/screen_pool.pnl_current (SCREENED_BASELINE `pnl_current_sizing`). Engine order: FLIP / NONEXP_CALM3D /
momentum SHORT (C1, W2+W1) → 1× · UNMATCHED long: crowd-sprint → 1×, else pair-vol ≥ 0.90 → 1×, else min(cell, 1.5×) (206).
History: v19 (2026-10-06) found the screen still pricing UNMATCHED longs at 2× and W2+W1 shorts at 2×; 10-06c then found the master
pricing the same 3+1 W2+W1 shorts at 2× and the 5 PVR ≥ 0.90 longs at 1.5× — two copies of one rule drift, so there is one copy now."""
import csv, json, os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scripts import build_master_pool as B
from scripts.build_master_pool import today_size_rule, today_size_scale

MOM, FLIP = "MOMENTUM", "FLIP:FAN_RATIO_GATE"


@pytest.mark.parametrize("args,want,tag", [
    ((MOM, "SHORT", "C1", 2.0), 0.5, "SHORT_1X"),                              # C1 short de-mux (Jun-29)
    ((MOM, "SHORT", "C1+C6", 1.5), 1 / 1.5, "SHORT_1X"),                       # any multiplied momentum short → 1×
    ((MOM, "SHORT", "W1+W2+W1", 2.0), 0.5, "SHORT_1X"),                        # W2+W1 short de-muxed 2026-07-30
    ((MOM, "SHORT", "W1+W2+W1", 2.0, None, None, None, 1.5), 1 / 3, "SHORT_1X"),  # SOL 06-18: 2× at 30× lev → 1× at 20×
    ((MOM, "SHORT", "W1", 1.0, None, None, None, 1.5), 1 / 1.5, "SHORT_1X"),      # a lev-only boost is stripped too
    ((MOM, "LONG", "UNMATCHED", 1.0, 0.50, 0.01, 0.50, 1.5), 1 / 1.5, "UNMATCHED_INV"),
    ((MOM, "LONG", "ADXMAX_PROBE", 0.5, None, None, None, 0.05), 1.0, ""),       # probes never re-priced
    ((MOM, "SHORT", "UNMATCHED", 2.0), 0.5, "SHORT_1X"),                       # not the 1.5× long rule
    ((MOM, "SHORT", "W1", 1.0), 1.0, ""),
    ((MOM, "LONG", "NONEXP_CALM3D", 2.0), 0.5, "CALM3D_1X"),                   # DECISION_LOG 225
    ((MOM, "LONG", "NONEXP_CALM3D", 1.0), 1.0, ""),
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE[NEGDI15]×2", 2.0), 0.25, "FLIP_FAN_LEV"),  # 220 size → 1× · 252 FAN lev 20× → 10×
    (("nan", "SHORT", "FLIP:FAN_RATIO_GATE×2", 2.0), 0.25, "FLIP_FAN_LEV"),          # flip known by its source alone
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0), 0.5, "FLIP_FAN_LEV"),              # a 1× FAN fill at 20× → 10× (252)
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 10), 1.0, "FLIP_FAN_LEV"),   # already bracket-capped at 10× → unchanged
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 20), 0.5, "FLIP_FAN_LEV"),
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE", 1.0, None, None, None, 1.0, 15), 10 / 15, "FLIP_FAN_LEV"),  # schedule-capped 15× → 10×
    ((FLIP, "SHORT", "FLIP:FAN_RATIO_GATE+FGP_QS", 0.5, None, None, None, 0.05, 1), 1.0, "FLIP_FAN_LEV"),  # probe 1× stays 1× (0.05 floor)
    (("FLIP:PAIR_RSI_OB", "SHORT", "FLIP:PAIR_RSI_OB×2", 2.0), 0.5, "FLIP_1X"),      # other flip sources: 220 rule unchanged
    (("FLIP:PAIR_RSI_OB", "SHORT", "FLIP:PAIR_RSI_OB", 1.0), 1.0, ""),
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.50, 0.01, 0.50), 0.75, "UNMATCHED_INV"),  # DECISION_LOG 206 — 2× → 1.5×
    ((MOM, "LONG", "UNMATCHED", 2.5, 0.50, 0.01, 0.50), 0.6, "UNMATCHED_INV"),   # old 2.5× quiet boost → 1.5×
    ((MOM, "LONG", "UNMATCHED", 1.5, 0.50, 0.01, 0.50), 1.0, "UNMATCHED_INV"),   # already at today's size
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.80, 0.08, 0.50), 0.5, "SPRINT_DEMUX"),    # crowd-sprint → 1×
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.80, 0.08, 1.20), 0.5, "SPRINT_DEMUX"),    # sprint first (engine order)
    ((MOM, "LONG", "UNMATCHED", 1.5, 0.80, 0.08, 0.50), 1 / 1.5, "SPRINT_DEMUX"),
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.74, 0.08, 0.50), 0.75, "UNMATCHED_INV"),  # strict > on global vol
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.80, 0.07, 0.50), 0.75, "UNMATCHED_INV"),  # strict > on BTC slope
    ((MOM, "LONG", "UNMATCHED", 2.0, None, 0.08, 0.50), 0.75, "UNMATCHED_INV"),  # missing stamp = fail-open
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.50, 0.01, 0.90), 0.5, "PVR_DEMUX"),       # crowded entry (≥)
    ((MOM, "LONG", "UNMATCHED", 2.0, 0.50, 0.01, float("nan")), 0.75, "UNMATCHED_INV"),
    ((MOM, "LONG", "UNMATCHED", 1.0, 0.80, 0.08, 1.20), 1.0, ""),               # a 1× fill is never re-priced
    (("SPIKE_FADE", "SHORT", "SPIKE_FADE", 2.0), 1.0, ""),                      # fades: ticket scale, not a cell
    (("FRENZY_LONG", "LONG", "FRENZY_LONG", 2.0), 1.0, ""),                     # sleeves: re-priced by _today
    (("BEARRUN_SHORT", "SHORT", "", 2.0), 1.0, ""),
])
def test_today_size_rule(args, want, tag):
    f, t = today_size_rule(*args)
    assert f == pytest.approx(want) and t == tag
    assert today_size_scale(*args) == pytest.approx(want)


def test_frozen_constants_match_live_json():
    """Frozen with STACK_VERSION — must equal the live config at ship time (a retune = new STACK_VERSION + these values)."""
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = th.get("thresholds", th)
    assert th["long_unmatched_sprint_demux_gvr_min"] == B.UNMATCHED_SPRINT_GVR_MIN_FROZEN
    assert th["long_unmatched_sprint_demux_b20slope_min"] == B.UNMATCHED_SPRINT_B20SLOPE_MIN_FROZEN
    assert th["long_unmatched_mult_pvr_max"] == B.UNMATCHED_PVR_MAX_FROZEN
    assert th["long_unmatched_demux_inv_mult"] == 1.0                           # the PVR de-mux target is 1×
    assert th["long_unmatched_quiet_mult"] <= B.UNMATCHED_LONG_INV_FROZEN       # the quiet boost never sizes above the cell
    rules = th["pattern_cell_rules"]
    um = [r for r in rules if r.get("pattern") == "UNMATCHED" and r.get("direction") == "LONG"][0]
    assert float(um["inv_mult"]) == B.UNMATCHED_LONG_INV_FROZEN and float(um["lev_mult"]) == B.UNMATCHED_LONG_LEV_FROZEN
    assert th["long_unmatched_demux_lev_mult"] == 1.0 and th["long_unmatched_quiet_lev_mult"] <= B.UNMATCHED_LONG_LEV_FROZEN
    assert th["nonexp_calm3d_lev_mult"] <= 1.0
    assert th["flip_short_negdi_lev_mult"] <= 1.0 and th["flip_short_tg_shallow_lev_mult"] <= 1.0
    hot = [r["pattern"] for r in rules if r.get("direction") == "SHORT"
           and max(float(r.get("inv_mult") or 1.0), float(r.get("lev_mult") or 1.0)) > 1.0]
    assert not hot, f"a SHORT cell sizes above 1× again ({hot}) — today_size_rule de-muxes every momentum short; re-scope it"
    assert th["nonexp_calm3d_invest_mult"] <= 1.0
    assert th["flip_short_negdi_mult"] <= 1.0 and th["flip_short_tg_shallow_mult"] <= 1.0


def test_screen_uses_the_shared_rule():
    import inspect
    from scripts import screen_pool
    assert screen_pool.today_size_scale is today_size_scale
    body = inspect.getsource(screen_pool.pnl_current)
    assert "today_size_scale(" in body and "cell_lev_multiplier" in body and "getattr(" not in body


def _rows(name):
    p = os.path.join(ROOT, "reports", name)
    if not os.path.exists(p):
        pytest.skip(f"{name} not built")
    return list(csv.DictReader(open(p, encoding="utf-8")))


def test_frozen_baseline_is_at_todays_sizing():
    """A sizing change without a screen_pool.py re-run leaves the frozen column stale."""
    from scripts.screen_pool import pnl_current
    stale = [(r["pair"], r["opened_at"]) for r in _rows("SCREENED_BASELINE.csv")
             if abs(float(r["pnl_current_sizing"]) - pnl_current(r)) > 0.01]
    assert not stale, f"SCREENED_BASELINE.csv not re-frozen at today's sizing: {stale[:5]}"


def test_screen_and_master_agree_on_kept_rows():
    """Every screened row the master keeps carries the same today's-sizing $ (±1 cent of rounding)."""
    m = {(r["opened_at"][:19], r["pair"], r["direction"]): r for r in _rows("MASTER_POOL_stacked.csv")}
    assert next(iter(m.values())).get("stack_version") == B.STACK_VERSION, "master not rebuilt after a STACK_VERSION bump — run scripts/build_master_pool.py"
    diff = []
    for r in _rows("SCREENED_BASELINE.csv"):
        mm = m.get((r["opened_at"][:19], r["pair"], r["direction"]))
        assert mm is not None, f"screened row missing from the master: {r['pair']} {r['opened_at']}"
        if mm["stack_keep"] == "True" and abs(float(mm["stack_pnl"]) - float(r["pnl_current_sizing"])) > 0.011:
            diff.append((r["pair"], r["opened_at"], r["pnl_current_sizing"], mm["stack_pnl"]))
    assert not diff, f"screen vs master sizing drift: {diff[:5]}"
