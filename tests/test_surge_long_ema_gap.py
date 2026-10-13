"""📐 Oct-13 (DECISION_LOG 287): SURGE_LONG EMA13 − EMA50 gap gate — pure rule (services.surge.surge_long_ema_gap_block), D11 surfaces,
engine wiring (gate sits in the SURGE_LONG entry path, judged on the stamped value) and the master-builder freeze."""
import json
import os
import re
from types import SimpleNamespace

from services.surge import SURGE_LONG_EMA_GAP, SURGE_LONG_EMA_GAP_FAILOPEN, surge_long_ema_gap_block

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def th(v):
    return SimpleNamespace(surge_long_min_ema13_50_gap_pct=v)


# ── pure rule ────────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_blocks_strictly_below_the_threshold():
    assert surge_long_ema_gap_block(th(0.0), -0.0001) == SURGE_LONG_EMA_GAP
    assert surge_long_ema_gap_block(th(0.0), -2.5) == SURGE_LONG_EMA_GAP
    assert surge_long_ema_gap_block(th(0.0), 0.0) is None          # equal passes (strict <)
    assert surge_long_ema_gap_block(th(0.0), 0.37) is None
    assert surge_long_ema_gap_block(th(0.5), 0.49) == SURGE_LONG_EMA_GAP
    assert surge_long_ema_gap_block(th(-0.3), -0.2) is None
    assert surge_long_ema_gap_block(th(0.0), "-0.1") == SURGE_LONG_EMA_GAP   # numeric string stamp


def test_unreadable_gap_fails_open():
    for g in (None, float("nan"), float("inf"), float("-inf"), "x", True):
        assert surge_long_ema_gap_block(th(0.0), g) == SURGE_LONG_EMA_GAP_FAILOPEN, g


def test_off_settings_never_block_or_count():
    for v in (None, "", "  ", "abc", float("nan")):
        assert surge_long_ema_gap_block(th(v), -5.0) is None, v
        assert surge_long_ema_gap_block(th(v), None) is None, v
    assert surge_long_ema_gap_block(SimpleNamespace(), -5.0) is None   # missing key = off


# ── D11 parity ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_config_default_is_off_and_live_json_arms_at_zero():
    from config import SignalThresholds
    assert SignalThresholds().surge_long_min_ema13_50_gap_pct is None
    assert SignalThresholds(surge_long_min_ema13_50_gap_pct=0.0).surge_long_min_ema13_50_gap_pct == 0.0
    live = json.loads(_read("trading_config.json"))
    assert live["thresholds"]["surge_long_min_ema13_50_gap_pct"] == 0.0
    cfg = _read("config.py")
    assert "DECISION_LOG 287" in cfg and "REVERT GATE" in cfg and "WR ≥ 31.5 %" in cfg


def test_ui_input_load_save_and_export_line():
    html = _read("templates", "index.html")
    assert html.count('id="config-sg-long-min-gap"') == 1
    assert html.count("getElementById('config-sg-long-min-gap')") == 2          # load + save
    assert "surge_long_min_ema13_50_gap_pct: (function(){ const v = String(document.getElementById('config-sg-long-min-gap')" in html
    assert "_e.value = (_v === null || _v === undefined) ? '' : _v; } }   // 📐 Oct-13 (287)" in html
    start = html.index("function _buildConfigLines(")
    assert "_bt.surge_long_min_ema13_50_gap_pct" in html[start:start + 60000]   # the shared config block both text exports use


def test_window_ledger_counts_the_refusals_on_every_surface():
    from models import SurgeTrigger
    assert "refused_ema_gap" in {c.name for c in SurgeTrigger.__table__.columns}
    assert "ADD COLUMN refused_ema_gap INTEGER DEFAULT 0" in _read("database.py")
    assert "'ref_gap': getattr(t, 'refused_ema_gap', 0) or 0" in _read("main.py")
    html = _read("templates", "index.html")
    assert html.count("${r.ref_open}/${r.ref_gap ?? 0}") == 3       # UI cell + clipboard copy + saved-file export
    assert html.count("'Skip A/L/D/S/O/G'") == 2


# ── engine wiring ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_engine_gate_sits_in_the_surge_long_entry_path():
    src = _read("services", "trading_engine.py")
    body = src[src.index("async def _maybe_open_surge("):src.index("async def _maybe_open_bullrun_long(")]
    i_ef = body.index("_ef = dict(self._flip_entry_fields(indicators, flip_dir=side)")
    i_gate = body.index("surge_long_ema_gap_block(th, _gap_v)")
    i_open = body.index("order = await self.open_position(")
    assert i_ef < i_gate < i_open
    gate = body[i_gate - 400:i_open]
    assert 'if side == "LONG":' in gate
    assert "_gap_v = _ef.get('entry_pair_ema20_ema50_gap_pct')" in gate          # the stamped value (engine parity)
    assert "self._record_filter_block(SURGE_LONG_EMA_GAP, \"LONG\")" in gate
    assert "self._record_filter_block(SURGE_LONG_EMA_GAP_FAILOPEN, \"LONG\")" in gate
    assert "[SURGE_LONG_EMA_GAP]" in gate and "would-be entry {price" in gate and "trigger bar close" in gate
    assert re.search(r"await self\._surge_ledger_counts\(db, side, st\)\s+continue", gate)
    # the stamp the gate reads is the EMA13 − EMA50 gap
    assert "put('entry_pair_ema20_ema50_gap_pct', lambda: round((e13 - e50) / e50 * 100, 4))" in src
    # the refused pair is never re-stamped as a momentum Block Reason
    import services.trading_engine as TE
    assert not TE.pair_reason_stampable(SURGE_LONG_EMA_GAP) and not TE.pair_reason_stampable(SURGE_LONG_EMA_GAP_FAILOPEN)


# ── master builder freeze ────────────────────────────────────────────────────────────────────────────────────────────────────
def test_builder_frozen_gate_matches_live_config():
    bld = _read("scripts", "build_master_pool.py")
    assert "_SURGE_GAP_TH = SimpleNamespace(surge_long_min_ema13_50_gap_pct=0.0)" in bld
    assert "surge_long_ema_gap_block(_SURGE_GAP_TH, r.get(G)) == 'SURGE_LONG_EMA_GAP'" in bld
    live = json.loads(_read("trading_config.json"))["thresholds"]
    assert live["surge_long_min_ema13_50_gap_pct"] == 0.0
