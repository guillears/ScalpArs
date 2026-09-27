"""Sep-27 — the builder's FADE_FRESHBREAK is a STAMP PROXY (entry_rsi_prev = rsi_prev2) for a live gate that reads
rsi_prev1 at trigger. It may only screen fills opened BEFORE the gate shipped; a post-ship fill already passed the
real gate, and re-screening it on the wrong bar removed six live winners from the master pool. Pin the scope + the
thresholds to the live config: a retune means post-ship fills opened under the old values need re-screening, so it
must fail here and force the builder to be revisited.
"""
import importlib.util
import json
import pathlib

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("build_master_pool", _ROOT / "scripts" / "build_master_pool.py")
bmp = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(bmp)
blk = bmp.fade_freshbreak_stamp_block


def test_pre_ship_stamp_proxy_blocks_fresh_breakout():
    assert blk("2026-08-10T10:03:58", 40.0, 0.10)          # TAUSDT-class pre-ship fill
    assert not blk("2026-08-10T10:03:58", 44.0, 0.10)      # strict < on rsi_prev
    assert not blk("2026-08-10T10:03:58", 40.0, -0.40)     # strict > on pgap
    assert not blk("2026-08-10T10:03:58", float("nan"), 0.10)   # fail-open on unstamped
    assert not blk("2026-08-10T10:03:58", 40.0, None)


def test_post_ship_fills_defer_to_live_gate():
    assert not blk(bmp.FADE_FB_SHIP_UTC, 40.0, 0.10)       # boundary: shipped → live gate owns it
    assert not blk("2026-09-23T09:04:05", 30.0, 0.50)      # ZILUSDT-class: stamp fails, live gate passed
    assert blk("2026-08-10T13:48:18", 40.0, 0.10)          # one second before ship


def test_thresholds_match_live_config():
    th = json.loads((_ROOT / "trading_config.json").read_text())["thresholds"]
    assert float(th["spike_fade_fb_rsi_prev_min"]) == bmp.FADE_FB_RSI_PREV_MIN
    assert float(th["spike_fade_fb_pgap_min"]) == bmp.FADE_FB_PGAP_MIN
