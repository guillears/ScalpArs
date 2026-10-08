"""DECISION_LOG 220 (2026-10-06): the FAN flip 2× cells NEGDI15 + TG_SHALLOW are back to 1× live,
and the master pool re-prices every kept flip above 1× to 1×."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_live_json_flip_cells_at_1x():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = th.get("thresholds", th)
    assert th["flip_short_negdi_mult"] <= 1.0 and th["flip_short_tg_shallow_mult"] <= 1.0
    assert th["flip_short_negdi_lev_mult"] <= 1.0 and th["flip_short_tg_shallow_lev_mult"] <= 1.0
    # the master re-prices EVERY kept flip above 1× → no other flip sizing source may exceed 1× either
    size, lev = (float(x) for x in th["flip_fan_qs_cell"].split(":")[3:5])
    assert size <= 1.0 and lev <= 1.0
    for spec in str(th["flip_entry_sources"]).split(","):
        assert all(float(x) <= 1.0 for x in spec.split(":")[1:]), spec


def test_master_builder_reprices_flips_to_1x():
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert 'STACK_VERSION = "2026-10-08a"' in bld
    from scripts.build_master_pool import today_size_scale   # 10-06c: one shared sizing rule (tests/test_today_size_scale.py)
    for src in ("FLIP:FAN_RATIO_GATE[NEGDI15]×2", "FLIP:FAN_RATIO_GATE[TG_SHALLOW]×2", "FLIP:FAN_RATIO_GATE×2"):
        assert today_size_scale("FLIP:FAN_RATIO_GATE", "SHORT", src, 2.0) == 0.5
