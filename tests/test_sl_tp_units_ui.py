"""⇄ (DECISION_LOG 326, operator 2026-10-20) "% | $" unit switch for the stop / target inputs.

The server stays in %: the engine's SL / TP are P&L % NET of fees on the position notional (pnl_percentage = net pnl /
(entry × qty) × 100), so $ = notional × % / 100 exactly; a $ typed is converted to the % (4 dp, as set_exit_override stores it)
before anything is sent. Pinned here: the toggle in the three places (🖐 manual entry · ✎ Set exit editor · ➕ Add pop-up shows
both units), the shared conversion helper is what every place uses, the submitted value is the %, and the helper math (node).
"""
import json
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html")).read()


def _span(start, end):
    i = HTML.index(start)
    return HTML[i:HTML.index(end, i)]


# ── source wiring ────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_manual_modal_toggle_and_ids_unique():
    modal = _span('id="manual-entry-modal"', 'id="add-position-modal"')
    for i in ("manual-sltp-unit", "manual-sltp-unit-pct", "manual-sltp-unit-usd", "manual-sl-unit-lbl", "manual-tp-unit-lbl",
              "manual-sl-conv", "manual-tp-conv", "manual-sl", "manual-tp"):
        assert HTML.count(f'id="{i}"') == 1, i
        assert f'id="{i}"' in modal, i
    assert "manualSetSlTpUnit('pct')" in modal and "manualSetSlTpUnit('usd')" in modal


def test_manual_submit_sends_pct_and_risk_line_uses_helper():
    sub = _span("async function openManualPosition()", "body: JSON.stringify(body)")
    assert "sl_pct: (g('manual-sl')?.disabled || (g('manual-sl')?.value === '' && !g('manual-sl')?.validity?.badInput)) ? null : manualFieldPct('manual-sl')" in sub
    assert "tp_pct: (g('manual-tp')?.disabled || (g('manual-tp')?.value === '' && !g('manual-tp')?.validity?.badInput)) ? null : manualFieldPct('manual-tp')" in sub
    assert "parseFloat(g('manual-sl').value)" not in sub and "parseFloat(g('manual-tp').value)" not in sub
    fpct = _span("function manualFieldPct(id)", "function manualSetSlTpUnit")
    assert "slTpPctFromUsd(x, n)" in fpct and "manualNotional()" in fpct
    risk = _span("function manualRiskLine()", "try { manualExitModeChanged(); } catch (e) {}   // default exit")
    assert "manualFieldPct('manual-sl')" in risk                       # a $ stop is read through the same conversion as the submit
    assert "const y = slTpUsdFromPct(stop, pos);" in risk              # the risk $ = the helper's $ → agrees with the typed $ exactly
    assert "manualSlTpConv()" in risk                                  # the "= …" line refreshes on every input of the pop-up
    assert "slTpFmtUsd(slTpUsdFromPct(_lf, pos))" in risk              # the too-wide refusal carries both units
    assert "manualSlTpPaint()" in _span("function openManualModal()", "function closeManualModal()")


def test_exit_editor_toggle_and_pct_submitted():
    ed = _span("function closeChoiceNear(anchor, label, curSl, curTp, isOverride, pnlNow, notional)", "async function closeOrder(")
    for i in ("exit-ovr-unit", "exit-ovr-unit-pct", "exit-ovr-unit-usd", "exit-ovr-sl", "exit-ovr-tp", "exit-ovr-sl-conv", "exit-ovr-tp-conv"):
        assert f"'{i}'" in ed, i
    assert "slTpPctFromUsd(x, n)" in ed and "slTpConvertTyped(" in ed and "slTpGetUnit('exit')" in ed and "slTpSaveUnit('exit', u)" in ed
    assert "finish({ action: 'edit', sl, tp, unit, notional: n, slUsd:" in ed
    assert "fieldPct(slI)" in ed and "fieldPct(tpI)" in ed
    co = _span("async function closeOrder(", "// ----- Investor Portfolio -----")
    assert "Number(_oRow.entry_price) * Number(_oRow.quantity)" in co    # notional = entry × qty, the engine's pnl base
    assert "closeChoiceNear(btn, label, curSl, curTp, isOverride, pnlNow, _oNotional)" in co
    assert "JSON.stringify({ order_id: orderId, sl_pct: choice.sl, tp_pct: choice.tp })" in co   # only the % reaches the API
    assert "_usdNote" in co                                             # a server refusal shows both units when typed in $


def test_add_popup_shows_both_units():
    op = _span("function openAddModal(orderId, btn)", "function closeAddModal()")
    assert "slTpUsdFromPct(o.pattern_fixed_sl_pct, Number(o.entry_price) * Number(o.quantity))" in op
    assert "slTpUsdFromPct(o.pattern_fixed_tp_pct, Number(o.entry_price) * Number(o.quantity))" in op
    pv = _span("async function addPreview()", "async function submitAddPosition()")
    assert "Number(d.new_qty) * Number(d.new_avg_entry)" in pv and 'id="add-pos-sltp-usd"' in pv
    assert "slTpUsdFromPct(d.stop_pct, _nN)" in pv and "slTpUsdFromPct(d.tp_pct, _nN)" in pv


def test_unit_storage_is_guarded_and_defaults_to_pct():
    blk = _span("function slTpGetUnit(form)", "function slTpExactPct(")
    assert "try { return localStorage.getItem('sltpUnit.' + form) === 'usd' ? 'usd' : 'pct'; } catch (e) { return 'pct'; }" in blk
    assert "try { localStorage.setItem(" in blk


def test_engine_pnl_base_is_entry_times_qty():
    """The $ = notional × % / 100 identity holds only while the engine's pnl % is net pnl over entry × quantity."""
    ind = open(os.path.join(ROOT, "services", "indicators.py")).read()
    assert "notional = quantity * entry_price" in ind and "pnl_percentage = (pnl / notional) * 100" in ind
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read()
    assert "pattern_fixed_sl_pct=round(sl, 4)" in eng                   # the 4 dp the $ → % conversion mirrors


# ── conversion math (node) ───────────────────────────────────────────────────────────────────────────────────────────────────

def _helpers():
    i = HTML.index("        // ⇄ SLTP-UNITS BEGIN")
    j = HTML.index("        // 🖐 Oct-20 (operator, DECISION_LOG 320): the manual form is a pop-up", i)
    return HTML[i:j]


_STUB = r"""
const _els = {};
function mkEl(v) { return { value: v ?? '', disabled: false, dataset: {}, textContent: '', step: '', placeholder: '', style: {},
                           setAttribute() {}, validity: {} }; }
['manual-investment', 'manual-leverage', 'manual-sl', 'manual-tp', 'manual-sl-conv', 'manual-tp-conv', 'manual-sl-unit-lbl',
 'manual-tp-unit-lbl', 'manual-sltp-unit-pct', 'manual-sltp-unit-usd'].forEach(id => { _els[id] = mkEl(''); });
const document = { getElementById: id => _els[id] || null };
let _toasts = []; function showToast(m) { _toasts.push(m); }
let _store = {}; let _storeThrows = false;
const localStorage = { getItem: k => { if (_storeThrows) throw new Error('blocked'); return _store[k] ?? null; },
                       setItem: (k, v) => { if (_storeThrows) throw new Error('blocked'); _store[k] = v; } };
function manualRiskLine() { manualSlTpConv(); }
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_conversion_math_and_manual_form_round_trip():
    prog = _STUB + _helpers() + r"""
const out = {};
out.usd = slTpUsdFromPct(1.5, 10000); out.pct = slTpPctFromUsd(90, 10000);
out.neg = slTpPctFromUsd(-90, 6000); out.lock = slTpPctFromUsd(30, 6000); out.dp4 = slTpPctFromUsd(33.333, 10000);
out.zeroN = slTpPctFromUsd(90, 0); out.badN = slTpUsdFromPct(1.5, null); out.negZero = Object.is(slTpPctFromUsd(-0.0000001, 10000), -0);
out.fmt = [slTpFmtUsd(-90, true), slTpFmtUsd(30, true), slTpFmtPct(-1.5, true), slTpFmtPct(0.9)];
// manual pop-up: $500 × 20 = $10,000 position
_els['manual-investment'].value = '500'; _els['manual-leverage'].value = '20'; _els['manual-sl'].value = '1.5'; _els['manual-tp'].value = '';
out.unit0 = _manualSlTpUnit;
manualSetSlTpUnit('usd');
out.slUsd = _els['manual-sl'].value; out.slConv = _els['manual-sl-conv'].textContent; out.slPctUsd = manualFieldPct('manual-sl');
out.lbl = _els['manual-sl-unit-lbl'].textContent; out.saved = _store['sltpUnit.manual'];
_els['manual-sl'].value = '90'; manualSlTpConv();
out.typed = manualFieldPct('manual-sl'); out.typedConv = _els['manual-sl-conv'].textContent;
manualSetSlTpUnit('pct'); out.back = _els['manual-sl'].value;
// a notional where the cents round: 33.3 × 10 = $333 → 1.5 % = $4.995 shown as $5 → back to exactly 1.5 (nothing jumps)
_els['manual-investment'].value = '33.3'; _els['manual-leverage'].value = '10'; _els['manual-sl'].value = '1.5';
manualSetSlTpUnit('usd'); out.rUsd = _els['manual-sl'].value; out.rPct = manualFieldPct('manual-sl');
manualSetSlTpUnit('pct'); out.rBack = _els['manual-sl'].value;
// $ shown, then Size changed → the $ typed rules (no stale exact %)
manualSetSlTpUnit('usd'); _els['manual-investment'].value = '66.6'; out.afterSize = manualFieldPct('manual-sl');
// $ mode without Size × Lev → not convertible, and the toggle refuses to switch into $
manualSetSlTpUnit('pct'); _els['manual-investment'].value = ''; manualSetSlTpUnit('usd'); out.refused = _manualSlTpUnit; out.toasts = _toasts.length;
_storeThrows = true; out.blocked = slTpGetUnit('manual'); slTpSaveUnit('manual', 'usd');
console.log(JSON.stringify(out));
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r["usd"] == 150 and r["pct"] == 0.9 and r["neg"] == -1.5 and r["lock"] == 0.5 and r["dp4"] == 0.3333
    assert r["zeroN"] is None and r["badN"] is None and r["negZero"] is False
    assert r["fmt"] == ["−$90.00", "+$30.00", "−1.50 %", "0.90 %"]
    assert r["unit0"] == "pct"
    assert r["slUsd"] == "150" and r["slConv"] == "= 1.50 %" and r["slPctUsd"] == 1.5 and r["lbl"] == "SL $" and r["saved"] == "usd"
    assert r["typed"] == 0.9 and r["typedConv"] == "= 0.90 %" and r["back"] == "0.9"
    assert r["rUsd"] == "5" and r["rPct"] == 1.5 and r["rBack"] == "1.5"
    assert r["afterSize"] == pytest.approx(5 / 666 * 100, abs=1e-4)
    assert r["refused"] == "pct" and r["toasts"] == 1
    assert r["blocked"] == "pct"


def test_inline_ids_referenced_exist():
    for i in ("manual-sltp-unit-pct", "manual-sltp-unit-usd", "manual-sl-unit-lbl", "manual-tp-unit-lbl", "manual-sl-conv", "manual-tp-conv"):
        assert len(re.findall(re.escape(f"'{i}'"), HTML)) >= 1, i                  # used by the JS


_DOM = r"""
const _byId = {};
function mkNode(tag) {
  const n = { tag, style: {}, dataset: {}, children: [], value: '', textContent: '', placeholder: '', step: '', type: '', id: '',
    disabled: false, validity: {}, _ls: {}, offsetHeight: 200, className: '', title: '',
    setAttribute(k, v) { this[k] = v; }, append(...c) { this.children.push(...c); }, replaceChildren(...c) { this.children = c; },
    addEventListener(t, f) { (this._ls[t] = this._ls[t] || []).push(f); }, contains() { return true; }, remove() {},
    getBoundingClientRect() { return { right: 500, bottom: 100, top: 80 }; } };
  return n;
}
function walk(n) { if (n.id) _byId[n.id] = n; (n.children || []).forEach(walk); }
const document = { createElement: mkNode, body: { appendChild(b) { walk(b); } }, addEventListener() {}, removeEventListener() {},
                   getElementById: id => _byId[id] || null };
const window = { innerWidth: 1200, innerHeight: 900 };
let _toasts = []; function showToast(m) { _toasts.push(m); }
let _store = {};
const localStorage = { getItem: k => _store[k] ?? null, setItem: (k, v) => { _store[k] = v; } };
let _nearConfirmCancel = null;
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_exit_editor_signed_usd_round_trip():
    i = HTML.index("        function closeChoiceNear(anchor, label, curSl, curTp, isOverride, pnlNow, notional) {")
    j = HTML.index("        async function closeOrder(", i)
    helpers = _span("        // ⇄ SLTP-UNITS BEGIN", "        // 🖐 manual entry: SL / TP typed in % or $")
    prog = _DOM + helpers + HTML[i:j] + r"""
(async () => {
  const out = {};
  // $6,000 position, current override stop −1.5 % / target +2 %, the browser remembers $
  _store['sltpUnit.exit'] = 'usd';
  let p = closeChoiceNear(mkNode('button'), 'X LONG', -1.5, 2, true, 0.3, 6000);
  out.prefSl = _byId['exit-ovr-sl'].value; out.prefTp = _byId['exit-ovr-tp'].value;
  out.conv0 = _byId['exit-ovr-sl-conv'].textContent;
  _byId['exit-ovr-sl'].value = '30'; _byId['exit-ovr-sl']._ls.input[0]();     // +$30 = lock +0.50 %
  out.conv1 = _byId['exit-ovr-sl-conv'].textContent;
  out.hasUnit = !!_byId['exit-ovr-unit'];
  // flip to % → typed values convert, then back to $ → the prefilled target restores its exact %
  _byId['exit-ovr-unit-pct'].onclick(); out.pctSl = _byId['exit-ovr-sl'].value; out.pctTp = _byId['exit-ovr-tp'].value;
  out.saved = _store['sltpUnit.exit'];
  _byId['exit-ovr-unit-usd'].onclick(); out.usdTp = _byId['exit-ovr-tp'].value;
  _nearConfirmCancel(); out.cancel = await p;
  // unknown notional → % only, the $ button is disabled
  let q = closeChoiceNear(mkNode('button'), 'Y SHORT', null, null, false, null, null);
  out.noN = _byId['exit-ovr-unit-usd'].disabled; out.noNph = _byId['exit-ovr-sl'].placeholder;
  _nearConfirmCancel(); await q;
  console.log(JSON.stringify(out));
})();
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r["prefSl"] == "-90" and r["prefTp"] == "120" and r["conv0"] == "= −1.50 %"
    assert r["conv1"] == "= +0.50 %" and r["hasUnit"] is True
    assert r["pctSl"] == "0.5" and r["pctTp"] == "2" and r["saved"] == "pct"
    assert r["usdTp"] == "120" and r["cancel"] is None
    assert r["noN"] is True and r["noNph"] == "Stop P&L %"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_exit_editor_set_sends_signed_pct():
    i = HTML.index("        function closeChoiceNear(anchor, label, curSl, curTp, isOverride, pnlNow, notional) {")
    j = HTML.index("        async function closeOrder(", i)
    helpers = _span("        // ⇄ SLTP-UNITS BEGIN", "        // 🖐 manual entry: SL / TP typed in % or $")
    src = HTML[i:j].replace("set.onclick = doSet;", "set.onclick = doSet; window._doSet = doSet;")
    assert src != HTML[i:j]
    prog = _DOM + helpers + src + r"""
(async () => {
  _store['sltpUnit.exit'] = 'usd';
  const p = closeChoiceNear(mkNode('button'), 'X LONG', null, null, false, 0.3, 6000);
  _byId['exit-ovr-sl'].value = '-90'; _byId['exit-ovr-tp'].value = '33.333';
  window._doSet();
  console.log(JSON.stringify(await p));
})();
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r == {"action": "edit", "sl": -1.5, "tp": 0.5556, "unit": "usd", "notional": 6000, "slUsd": -90, "tpUsd": 33.333}


# ── review round (326 I1 + M items) ──────────────────────────────────────────────────────────────────────────────────────────

def _cut(start, end):
    i = HTML.index(start)
    return HTML[i:HTML.index(end, i)]


_FORM = r"""
const _els = {};
function mkEl(v) { return { value: v ?? '', disabled: false, dataset: {}, textContent: '', step: '', placeholder: '', style: {},
                           setAttribute() {}, validity: {}, classList: { toggle() {} }, getBoundingClientRect() { return {}; } }; }
['manual-investment', 'manual-leverage', 'manual-sl', 'manual-tp', 'manual-sl-conv', 'manual-tp-conv', 'manual-sl-unit-lbl',
 'manual-tp-unit-lbl', 'manual-sltp-unit-pct', 'manual-sltp-unit-usd', 'manual-risk-line', 'manual-pair', 'manual-direction',
 'manual-exit-mode', 'manual-open-btn'].forEach(id => { _els[id] = mkEl(''); });
_els['manual-pair'].value = 'BTCUSDT'; _els['manual-direction'].value = 'LONG'; _els['manual-exit-mode'].value = 'FIXED';
const document = { getElementById: id => _els[id] || null };
const window = { _cachedConfig: { thresholds: { manual_floor_sl_pct: -3 } }, _lastTotalPortfolio: 0, _lastStatusIsPaper: true };
let _toasts = []; function showToast(m) { _toasts.push(m); }
let _store = {};
const localStorage = { getItem: k => _store[k] ?? null, setItem: (k, v) => { _store[k] = v; } };
let _sent = []; const API_BASE = '';
async function fetch(url, o) { _sent.push(o.body); return { ok: true, json: async () => ({}) }; }
async function confirmNear() { return true; }
function closeManualModal() {} function loadOpenOrders() {} function loadTransactions() {} function loadBalance() {}
function manualPairKey() { return 'BTCUSDT'; } function manualModalIsOpen() { return true; } function openManualModal() {} function manualLimitHint() {}
"""


def _form_prog(body):
    return (_FORM + _helpers()
            + _cut("        function manualRiskLine() {", "        try { manualExitModeChanged(); } catch (e) {}   // default exit")
            + _cut("        function manualApplyFix(f, pair) {", "        function manualExitModeChanged() {")
            + _cut("        async function openManualPosition() {", "        // Close order manually")
            + _cut("        const MANUAL_LIQ_SAFETY_UI", "        function renderManualStopTable() {")
            + "(async () => { const out = {};\n" + body + "\nconsole.log(JSON.stringify(out)); })();")


def _run(prog):
    return json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_i1_unit_switch_refused_without_notional_both_directions():
    r = _run(_form_prog(r"""
// the reviewers' repro: $2 stop at 500 × 1, clear Size, click % → must NOT relabel "2" as 2 %
_els['manual-investment'].value = '500'; _els['manual-leverage'].value = '1';
manualSetSlTpUnit('usd'); _els['manual-sl'].value = '2'; _els['manual-tp'].value = '7';
_els['manual-investment'].value = '';
manualSetSlTpUnit('pct');
out.unitAfter = _manualSlTpUnit; out.slAfter = _els['manual-sl'].value; out.tpAfter = _els['manual-tp'].value; out.t1 = _toasts.slice(-1)[0];
// submit with no notional in $ mode → refused, nothing sent
await openManualPosition(); out.sent1 = _sent.length; out.t2 = _toasts.slice(-1)[0];
// retype 500 × 20 → the $2 stays $2 = 0.02 % (not 2 %)
_els['manual-investment'].value = '500'; _els['manual-leverage'].value = '20';
await openManualPosition(); out.body = JSON.parse(_sent[0]);
// % → $ without notional is refused too
manualSetSlTpUnit('pct'); _els['manual-leverage'].value = '0'; manualSetSlTpUnit('usd'); out.unit2 = _manualSlTpUnit; out.sl2 = _els['manual-sl'].value;
"""))
    assert r["unitAfter"] == "usd" and r["slAfter"] == "2" and r["tpAfter"] == "7" and "Type Size and Lev first" in r["t1"]
    assert r["sent1"] == 0 and "Type Size and Lev first" in r["t2"]
    assert r["body"]["sl_pct"] == 0.02 and r["body"]["tp_pct"] == 0.07
    assert r["unit2"] == "pct" and r["sl2"] == "0.02"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_bad_input_refused_not_dropped():
    r = _run(_form_prog(r"""
_els['manual-investment'].value = '500'; _els['manual-leverage'].value = '20';
_els['manual-sl'].value = '1.5'; _els['manual-tp'].value = ''; _els['manual-tp'].validity = { badInput: true };   // '1,5' typed in TP
await openManualPosition(); out.sent = _sent.length; out.t = _toasts.slice(-1)[0];
manualSetSlTpUnit('usd'); out.unit = _manualSlTpUnit;   // a switch with a mistyped field is refused too
"""))
    assert r["sent"] == 0 and r["t"] == "SL / TP must be numbers" and r["unit"] == "pct"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_pct_mode_payload_identical_to_pre_326():
    r = _run(_form_prog(r"""
_els['manual-investment'].value = '500'; _els['manual-leverage'].value = '20';
const cases = [['FIXED', '1.5', ''], ['FIXED', '0.25', '3'], ['FLOOR', '', '2.75'], ['BULLRUN_SL', '1', ''], ['FIXED', '2', '0.1']];
out.same = []; let k = 0;
for (const [mode, sl, tp] of cases) {
  _els['manual-exit-mode'].value = mode; _els['manual-sl'].value = sl; _els['manual-tp'].value = tp;
  _els['manual-sl'].disabled = (mode !== 'FIXED' && mode !== 'BULLRUN_SL'); _els['manual-tp'].disabled = (mode === 'BULLRUN_SL');
  const g = id => _els[id];
  const pre = JSON.stringify({ pair: 'BTCUSDT', direction: 'LONG', investment: 500, leverage: 20, exit_mode: mode,   // the pre-326 expressions
      sl_pct: (g('manual-sl')?.disabled || g('manual-sl')?.value === '') ? null : parseFloat(g('manual-sl').value),
      tp_pct: (g('manual-tp')?.disabled || g('manual-tp')?.value === '') ? null : parseFloat(g('manual-tp').value) });
  await openManualPosition(); out.same.push(_sent[k++] === pre);
}
"""))
    assert r["same"] == [True] * 5


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_limit_fix_in_usd_mode_keeps_usd_and_recomputes_pct():
    r = _run(_form_prog(r"""
_els['manual-investment'].value = '500'; _els['manual-leverage'].value = '20'; _els['manual-sl'].value = '1.5';
manualSetSlTpUnit('usd'); out.usd0 = _els['manual-sl'].value;                     // $150 on $10,000
manualApplyFix({ investment: 250, leverage: 10, label: 'x' }, 'BTCUSDT');         // ManualLimitError one-click fix → $2,500
out.usd1 = _els['manual-sl'].value; out.pct1 = manualFieldPct('manual-sl'); out.risk = _els['manual-risk-line'].textContent;
out.conv = _els['manual-sl-conv'].textContent;
"""))
    assert r["usd0"] == "150" and r["usd1"] == "150" and r["pct1"] == 6
    assert "$150.00" in r["risk"] and "= −6.00 %" in r["risk"] and "wider than the −3.00 % (−$75.00 on this position)" in r["risk"]
    assert r["conv"] == "= 6.00 %"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_usd_to_pct_rounding_non_round_notional_and_fmt_null():
    r = _run(_STUB + _helpers() + r"""
console.log(JSON.stringify({ p: slTpPctFromUsd(17.29, 61.728 * 20), back: slTpUsdFromPct(slTpPctFromUsd(17.29, 1234.56), 1234.56),
  n1: slTpFmtUsd(null), n2: slTpFmtUsd(NaN), n3: slTpFmtPct(undefined), n4: slTpFmtUsd(''), z: slTpFmtUsd(0) }));
""")
    assert r["p"] == round(17.29 / 1234.56 * 100, 4) == 1.4005
    assert abs(r["back"] - 17.29) < 0.005                                            # 4 dp keeps the $ to the cent on this size
    assert r["n1"] == r["n2"] == r["n3"] == r["n4"] == "—" and r["z"] == "$0.00"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_exit_editor_refuses_when_size_changed_since_typed():
    co = _cut("        async function closeOrder(", "        // ----- Investor Portfolio -----")
    prog = _STUB + _helpers() + r"""
const window = { _openOrdersById: { 7: { entry_price: 100, quantity: 60 } } };   // $6,000
let _sent = []; const API_BASE = '';
async function fetch(url, o) { _sent.push(o.body); return { ok: true, json: async () => ({ pair: 'X', direction: 'LONG', sl_pct: -1.5, tp_pct: null }) }; }
let _choice; async function closeChoiceNear() { return _choice; }
function loadOpenOrders() {} function loadClosedOrders() {} function loadTransactions() {} function loadPerformance() {} function loadBalance() {}
""" + co + r"""
(async () => {
  const out = {}; const btn = { getBoundingClientRect() { return {}; } };
  _choice = { action: 'edit', sl: -1.5, tp: null, unit: 'usd', notional: 6000, slUsd: -90, tpUsd: null };
  window._openOrdersById[7] = { entry_price: 100, quantity: 75 };               // an ➕ add landed: $7,500
  await closeOrder(7, btn, 'X LONG', null, null, false, 0.1);
  out.sent1 = _sent.length; out.t1 = _toasts.slice(-1)[0];
  window._openOrdersById[7] = { entry_price: 100, quantity: 60 };               // unchanged → sent, the % only
  await closeOrder(7, btn, 'X LONG', null, null, false, 0.1);
  out.body = JSON.parse(_sent[0]);
  _choice = { action: 'edit', sl: -1.5, tp: null, unit: 'pct', notional: 6000, slUsd: null, tpUsd: null };   // % typed → no size check
  window._openOrdersById[7] = { entry_price: 100, quantity: 75 };
  await closeOrder(7, btn, 'X LONG', null, null, false, 0.1); out.sent3 = _sent.length;
  console.log(JSON.stringify(out));
})();
"""
    r = _run(prog)
    assert r["sent1"] == 0 and "position size changed" in r["t1"] and "check the stop again" in r["t1"]
    assert "−$90.00 is now −1.20 %" in r["t1"] and "$6,000.00 → $7,500.00" in r["t1"]
    assert r["body"] == {"order_id": 7, "sl_pct": -1.5, "tp_pct": None}
    assert r["sent3"] == 2


def test_usd_inputs_take_cents():
    assert "si.step = usd ? '0.01' : '0.1'" in HTML and "ti.step = usd ? '0.01' : '0.1'" in HTML
    assert "slI.step = tpI.step = unit === 'usd' ? '0.01' : '0.1'" in HTML
