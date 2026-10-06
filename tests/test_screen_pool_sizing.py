"""SCREENED_BASELINE `pnl_current_sizing` = today's sizing (2026-10-06, v19). Pins scripts/screen_pool.pnl_current against the
master builder's re-price rules (STACK 2026-10-06b) plus the engine's short cells: every momentum SHORT cell (C1, W2+W1) → 1× · NONEXP_CALM3D → 1× (DECISION_LOG 225) · every FLIP above 1×
→ 1× (220) · UNMATCHED long: crowd-sprint de-mux → 1×, else PVR ≥ 0.90 → 1×, else 2× → 1.5× (206). Before v19 the screen still
priced UNMATCHED longs at 2× and skipped the sprint de-mux (ML +$3,635 instead of +$2,612)."""
import csv, os, re, sys
from types import SimpleNamespace
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import screen_pool as S

TH = SimpleNamespace(long_unmatched_sprint_demux_gvr_min=0.74, long_unmatched_sprint_demux_b20slope_min=0.07,
                     long_unmatched_mult_pvr_max=0.90)


def _row(src, mult, direction="LONG", pnl=100.0, gvr=0.50, slope=0.01, pvr=0.50):
    return {"pnl": str(pnl), "cell_multiplier_source": src, "cell_multiplier": str(mult), "direction": direction,
            "entry_global_volume_ratio": "" if gvr is None else str(gvr), "entry_btc_ema20_slope": "" if slope is None else str(slope),
            "entry_pair_volume_ratio": "" if pvr is None else str(pvr)}


@pytest.mark.parametrize("row,want", [
    (_row("C1", 2.0, "SHORT"), 50.0),                                  # C1 short de-mux (Jun-29)
    (_row("C1+C6", 1.5, "SHORT"), 100.0 / 1.5),                        # any multiplied short cell → 1×
    (_row("NONEXP_CALM3D", 2.0), 50.0),                                # DECISION_LOG 225
    (_row("NONEXP_CALM3D", 1.0), 100.0),
    (_row("FLIP:FAN_RATIO_GATE", 2.0, "SHORT"), 50.0),                 # DECISION_LOG 220 — no flip cell above 1×
    (_row("FLIP:FAN_RATIO_GATE×2", 2.0, "SHORT"), 50.0),
    (_row("UNMATCHED", 2.0), 75.0),                                    # DECISION_LOG 206 — 2× → 1.5×
    (_row("UNMATCHED", 2.0, pnl=-80.0), -60.0),                        # losers scale the same way
    (_row("UNMATCHED", 2.0, gvr=0.80, slope=0.08), 50.0),              # crowd-sprint de-mux → 1×
    (_row("UNMATCHED", 2.0, gvr=0.80, slope=0.08, pvr=1.20), 50.0),    # sprint takes precedence (still 1×)
    (_row("UNMATCHED", 2.0, gvr=0.74, slope=0.08), 75.0),              # strict > on global vol
    (_row("UNMATCHED", 2.0, gvr=0.80, slope=0.07), 75.0),              # strict > on BTC slope
    (_row("UNMATCHED", 2.0, gvr=None, slope=0.08), 75.0),              # missing stamp = fail-open (no de-mux)
    (_row("UNMATCHED", 2.0, pvr=0.90), 50.0),                          # crowded-PVR de-mux (≥)
    (_row("UNMATCHED", 2.0, pvr=None), 75.0),
    (_row("UNMATCHED", 1.5), 100.0),                                   # already at today's size
    (_row("UNMATCHED", 1.0, gvr=0.80, slope=0.08), 100.0),             # a 1× fill is never re-priced
    (_row("UNMATCHED", 2.0, "SHORT"), 50.0),                           # not the 1.5× long rule — a multiplied short → 1×
    (_row("W1+W2+W1", 2.0, "SHORT"), 50.0),                            # W2+W1 short de-muxed 2026-07-30
    (_row("W1", 1.0, "SHORT"), 100.0),
])
def test_pnl_current_rules(row, want):
    assert S.pnl_current(row, TH) == pytest.approx(want)


def test_frozen_constants_match_builder_and_live():
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert float(re.search(r"UNMATCHED_LONG_INV_FROZEN = ([0-9.]+)", bld).group(1)) == S.UNMATCHED_LONG_INV
    assert "r.entry_global_volume_ratio > 0.74 and r.entry_btc_ema20_slope > 0.07" in bld   # builder's frozen sprint legs
    import json
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = th.get("thresholds", th)
    assert th["long_unmatched_sprint_demux_gvr_min"] == 0.74 and th["long_unmatched_sprint_demux_b20slope_min"] == 0.07
    assert th["long_unmatched_quiet_mult"] <= S.UNMATCHED_LONG_INV       # the quiet boost no longer sizes above the cell
    assert th["nonexp_calm3d_invest_mult"] <= 1.0
    assert th["flip_short_negdi_mult"] <= 1.0 and th["flip_short_tg_shallow_mult"] <= 1.0
    rules = th.get("pattern_cell_rules") or json.load(open(os.path.join(ROOT, "trading_config.json"))).get("pattern_cell_rules") or []
    assert rules, "pattern_cell_rules not found"
    hot = [x["pattern"] for x in rules if x.get("direction") == "SHORT" and float(x.get("inv_mult") or 1.0) > 1.0]
    assert not hot, f"a SHORT cell sizes above 1× again ({hot}) — pnl_current de-muxes every short; re-scope it and re-freeze"


def test_frozen_baseline_is_at_todays_sizing():
    """A sizing change without a re-run of screen_pool.py leaves the frozen column stale — catch it here."""
    p = os.path.join(ROOT, "reports", "SCREENED_BASELINE.csv")
    if not os.path.exists(p):
        pytest.skip("baseline not frozen")
    import json
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    live = SimpleNamespace(**th.get("thresholds", th))   # explicit, not S.th — config.py loads the JSON cwd-relative (deep review)
    rows = list(csv.DictReader(open(p)))
    stale = [(r["pair"], r["opened_at"]) for r in rows if abs(float(r["pnl_current_sizing"]) - S.pnl_current(r, live)) > 0.01]
    assert not stale, f"SCREENED_BASELINE.csv not re-frozen at today's sizing: {stale[:5]}"
