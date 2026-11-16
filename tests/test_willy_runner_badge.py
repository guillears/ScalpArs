"""🏃 Nov-16 (operator: BARD was a WILLY runner but the Open Orders badge said "TP +1.25%") — a runner row shows its trail, not a TP."""
import pathlib

HTML = (pathlib.Path(__file__).resolve().parents[1] / "templates/index.html").read_text()
MAIN = (pathlib.Path(__file__).resolve().parents[1] / "main.py").read_text()


def test_runner_badge_replaces_the_tp_badge():
    i = HTML.index("['FRENZY_WILLY', 'FRENZY_WILLY_HT'].includes(o.entry_strategy || '')")
    blk = HTML[i:HTML.index("} else if (['FRENZY_LONG', 'FRENZY_WIDE', 'FRENZY_LITE']", i)]
    assert "const _rk = Number(o.frenzy_willy_runner_k), _ra = Number(o.entry_atr_pct);" in blk
    assert "if (_rk > 0 && _ra > 0) {" in blk and "'🏃 exit ≤ +' + _rLine.toFixed(2)" in blk and "'🏃 +' + _wTp + '↗')" in blk
    assert blk.index("'🎯 +' + _wTp") < blk.index("'🏃 exit ≤ +'")                  # the runner badge overrides the TP one
    assert "Math.max(Math.max(_tpPeak, _wTp) - _rk * _ra, _rFl)" in blk                # the engine's exit line (frenzy_willy_runner_line)
    assert "const _rArmed = _tpPeak >= _wTp - _rGap;" in blk                           # arms at T − fee gap (frenzy_willy_runner_arm_at)
    assert "(_wLateOn && _wHeldMin >= _wLateMin ? '⏱ +' + _wLatePct" in blk            # an unarmed runner shows the late TP once active


def test_open_orders_payload_carries_the_runner_fields():
    i = MAIN.index('@app.get("/api/orders/open")')
    body = MAIN[i:i + 20000]
    assert '"frenzy_willy_runner_k": getattr(o, \'frenzy_willy_runner_k\', None)' in body and '"entry_atr_pct"' in body


def test_fixed_exit_badges_are_short_tp_and_sl():
    """📱 Nov-16 (operator): fixed-exit sleeves keep one short badge with TP · SL; trail sleeves only when armed (unchanged)."""
    assert '">🎯 +${_fbTp} · −${_fbSl}</span>`;' in HTML                                  # Flatbase (P&L terms)
    assert '">🎯 +${_fzTp} · −${_fzSl}</span>`;' in HTML                                  # LITE / RETRY / FRENZY_LONG
    assert "${_wLateNow ? '⏱ +' + _wLatePct : '🎯 +' + _wTp} · ${_wSlN > 0 ? '−' + _wSlN : 'no SL'}" in HTML   # WILLY / HT (+ late TP)
    assert "const _wLateNow = _wLateOn && _wHeldMin >= _wLateMin && _tpPeak < _wTp;" in HTML
    for old in ("🩳 TP +${_fbTp}% · SL", "🎯 TP +${_wTp}% · ${_wSl} · ${_wMh}m", "trail ${_rW} pts · ${_wSl}"):
        assert old not in HTML
