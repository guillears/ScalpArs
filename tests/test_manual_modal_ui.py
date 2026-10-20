"""🖐 Oct-20 (operator, DECISION_LOG 320): the manual-entry form moved from a full-width bar to a pop-up opened by a 🖐 button in the
Top Pairs header; the Note input is gone; a live risk line sits above the Open button. Static pins on templates/index.html."""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UI = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()


def _modal_html():
    a = UI.index('<div id="manual-entry-modal"')
    b = UI.index('<div class="glass rounded-xl overflow-hidden">', a)
    return UI[a:b]


def test_hand_button_sits_in_the_top_pairs_header():
    a = UI.index("Top Pairs by Volume</h2>")
    b = UI.index('<div id="manual-entry-modal"', a)
    header = UI[a:b]
    assert 'id="refresh-btn"' in header
    assert re.search(r'<button[^>]*id="manual-entry-btn"[^>]*onclick="openManualModal\(\)"[^>]*title="Manual entry"[^>]*>🖐</button>', header)
    assert UI.count('id="manual-entry-btn"') == 1


def test_modal_holds_the_inputs_hints_risk_line_and_open_button():
    m = _modal_html()
    for i in ("manual-pair", "manual-direction", "manual-investment", "manual-leverage", "manual-exit-mode", "manual-sl", "manual-tp",
              "manual-limit-hint", "manual-pair-hint", "manual-risk-line", "manual-open-btn", "manual-entry-close"):
        assert f'id="{i}"' in m, i
    assert m.index('id="manual-risk-line"') < m.index('id="manual-open-btn"')   # the risk line sits just above the Open button
    assert 'role="dialog"' in m and 'onclick="manualModalBackdropClick(event)"' in m


def test_no_manual_note_input_and_note_not_sent():
    assert 'id="manual-note"' not in UI and "g('manual-note')" not in UI
    body = UI[UI.index("async function openManualPosition()"):]
    body = body[:body.index("JSON.stringify(body)")]
    assert "note:" not in body
    assert "o.manual_note" in UI   # historic notes still shown in the MANUAL badge hover


def test_risk_line_function_and_wiring():
    assert "function manualRiskLine()" in UI
    assert 'oninput="manualRiskLine()"' in _modal_html()
    fn = UI[UI.index("function manualRiskLine()"):]
    fn = fn[:fn.index("\n        }\n")]
    assert "window._lastTotalPortfolio" in fn and "text-amber-300" in fn and "text-red-400" in fn
    assert "z >= 10" in fn and "z >= 5" in fn and "stop set by the" in fn


def test_modal_open_close_and_esc():
    assert "function openManualModal()" in UI and "function closeManualModal()" in UI
    assert re.search(r"e\.key === 'Escape' && !e\.defaultPrevented && manualModalIsOpen\(\)\) \{ e\.preventDefault\(\); closeManualModal\(\);", UI)
    ok = UI[UI.index("if (res.ok) {", UI.index("async function openManualPosition()")):]
    assert ok.index("closeManualModal();") < ok.index("} else {")   # closes after a successful open
    assert "document.getElementById('manual-entry-btn'); if (_mp && data.manual_max_open_positions != null)" in UI   # cap 0 hides the 🖐 button
