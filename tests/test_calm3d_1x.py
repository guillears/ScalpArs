"""DECISION_LOG 225 (2026-10-06): the NONEXP_CALM3D door sizes at 1× (its 2× cell verdict fired). Pins the live JSON, the code
default, the engine fallback (a missing key or a 0 must never re-arm 2×), the UI save fallback, and the master re-price."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_calm3d_sizes_1x_everywhere():
    th = json.load(open(os.path.join(ROOT, "trading_config.json")))
    th = th.get("thresholds", th)
    assert th["nonexp_calm3d_invest_mult"] <= 1.0
    import config as C
    assert C.SignalThresholds.model_fields["nonexp_calm3d_invest_mult"].default == 1.0
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "getattr(_th_sp, 'nonexp_calm3d_invest_mult', 1.0) or 1.0" in eng and "'nonexp_calm3d_invest_mult', 2.0" not in eng
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert "config-nonexp-calm3d-invest-mult')?.value || 1.0" in ui and "_ssNum('config-nonexp-calm3d-invest-mult', 1.0) || 1.0" in ui


def test_master_builder_reprices_calm3d():
    bld = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert 'STACK_VERSION = "2026-10-06c"' in bld
    from scripts.build_master_pool import today_size_scale   # 10-06c: one shared sizing rule
    assert today_size_scale("MOMENTUM", "LONG", "NONEXP_CALM3D", 2.0) == 0.5
    assert today_size_scale("MOMENTUM", "LONG", "NONEXP_CALM3D", 1.0) == 1.0
