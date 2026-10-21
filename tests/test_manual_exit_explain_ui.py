"""📖 (DECISION_LOG 338, operator 2026-10-21) 🖐 Manual entry pop-up: a plain-language line under the Exit dropdown explains the chosen
exit's stop / take profit / max hold in the active unit (% or $, notional = Size × Lev, the DL 326 slTp* helpers), from the bot's LIVE saved
config (window._cachedConfig = GET /api/config) — and the SL / TP boxes a mode ignores are greyed with who sets them, never sent.

Pinned: source wiring; the builder run in node for every mode against a FAKE config (every number moves with the config — nothing hard-coded);
the leverage floor (the same manualStopForLeverage the risk line uses) and the broker-backstop refusal match the ENGINE's refusals
(manual_exit_levels / manual_momentum_refusal) at every leverage 1–125; the $ figures equal slTpUsdFromPct; disabled boxes are not
submitted; the UI backstop constants equal the engine's; every src-less inline script passes node --check."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
NODE = shutil.which("node") is None


def _span(start, end):
    i = HTML.index(start)
    return HTML[i:HTML.index(end, i)]


def _builder_js():
    return (_span("        // ⇄ SLTP-UNITS BEGIN", "        // ⇄ SLTP-UNITS END")
            + _span("        const MANUAL_LIQ_SAFETY_UI", "        function renderManualStopTable() {")
            + _span("        // 📖 MANUAL EXIT EXPLAIN BEGIN", "        // paints the line"))


def _node(body):
    r = subprocess.run(["node", "-e", _builder_js() + body], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    return json.loads(r.stdout)


# A FAKE config: deliberately different from trading_config.json, so a hard-coded number cannot pass.
FAKE = {
    "investment": {"max_holding_time_minutes": 300, "no_expansion_minutes": 45},
    "confidence_levels": {"STRONG_BUY": {"stop_loss": -0.55, "signal_active_sl": -0.85, "tp_min": 0.35,
                                         "be_level1_trigger": 99, "be_level1_offset": 99}},
    "thresholds": {
        "manual_floor_sl_pct": -2.5, "resting_tp_manual_enabled": True, "broker_backstop_enabled": False, "broker_backstop_pct": 2.5,
        "frenzy_stop_pct": 2.2, "frenzy_tp_pct": 1.7, "frenzy_lock_arm_pct": 0, "frenzy_trail_arm_pct": 4.5, "frenzy_trail_giveback_pct": 1.1,
        "frenzy_max_hold_minutes": 240,
        "bullrun_base_sl_pct": -0.6, "sl_atr_multiplier": 1.4, "sl_atr_widen_floor_pct": -1.1, "bullrun_be_arm_pct": 0.9,
        "bullrun_be_lock_pct": 0.15, "bullrun_trail_atr_mult": 2.5, "bullrun_ladder": "3.0:2.6, 6.0:5.2",
        "runner_trail_enabled": True, "runner_trail_arm_peak": 0.33, "runner_trail_atr_mult": 0.8, "runner_trail_atr_min": 0,
        "runner_trail_be_ratchet_enabled": True, "runner_trail_be_lock_pct": 0.07, "runner_trail_giveback_frac": 0,
        "runner_trail_short_enabled": True, "runner_trail_short_arm_peak": 0.36, "runner_trail_short_atr_mult": 0.6,
        "runner_trail_short_giveback_frac": 0.3, "runner_trail_short_be_ratchet_enabled": False, "runner_trail_short_atr_min": 0,
        "hard_tp_enabled": True, "hard_tp_ladder_long": "1.1:0.3,2.0:0.4", "hard_tp_ladder_short": "0.9:0.2",
        "ema13_cross_exit_enabled": True, "ema13_cross_exit_long_enabled": False, "ema13_cross_exit_short_enabled": True,
        "ema13_cross_requires_stack_flip": True, "fl1_wide_sl_backstop": -0.95,
    },
}


# ── source wiring ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_line_sits_under_the_exit_row_inside_the_modal():
    m = _span('id="manual-entry-modal"', 'id="add-position-modal"')
    assert HTML.count('id="manual-exit-explain"') == 1 and 'id="manual-exit-explain"' in m
    assert m.index('id="manual-exit-mode"') < m.index('id="manual-tp"') < m.index('id="manual-exit-explain"') < m.index('id="manual-risk-line"')
    risk = _span("function manualRiskLine()", "function manualExitFieldWhy")
    assert "try { manualExitExplainRender(); } catch (e) {}" in risk          # refreshed on every input / change of the pop-up
    rend = _span("function manualExitExplainRender()", "// 📖 MANUAL EXIT EXPLAIN END")
    assert "cfg: window._cachedConfig || {}" in rend and "unit: _manualSlTpUnit" in rend and "manualFieldPct(id)" in rend
    assert "if (!el2 || el2.disabled) return null;" in rend                   # a greyed box never feeds the line


def test_greyed_boxes_carry_their_reason_and_submit_skips_them():
    ch = _span("function manualExitModeChanged()", "async function openManualPosition()")
    assert "sl.disabled = (m !== 'FIXED' && m !== 'BULLRUN_SL')" in ch      # unchanged rule (DL 320 / Bull-Run pins)
    assert "tp.disabled = (m === 'MOMENTUM' || m === 'FRENZY' || m === 'BULLRUN' || m === 'BULLRUN_SL')" in ch
    assert "manualExitFieldWhy(m, k)" in ch and "el.dataset.why = w" in ch and "el.title = w ?" in ch and "el.style.opacity" in ch
    conv = _span("function manualSlTpConv()", "// 🖐 Oct-20 (operator, DECISION_LOG 320)")
    assert "if (el.disabled) { c.textContent = el.dataset && el.dataset.why ? el.dataset.why : ''; return; }" in conv
    sub = _span("async function openManualPosition()", "body: JSON.stringify(body)")
    assert "g('manual-sl')?.disabled ||" in sub and "g('manual-tp')?.disabled ||" in sub


def test_ui_backstop_constants_match_the_engine():
    import services.trading_engine as TE
    m = re.search(r"const MANUAL_BACKSTOP_MARGIN_PCT_UI = ([0-9.]+), MANUAL_BACKSTOP_MIN_GAP_PCT_UI = ([0-9.]+), MANUAL_BACKSTOP_LIQ_SHARE_UI = ([0-9.]+);", HTML)
    assert m
    assert float(m.group(1)) == TE.MANUAL_BACKSTOP_MARGIN_PCT and float(m.group(2)) == TE.MANUAL_BACKSTOP_MIN_GAP_PCT
    assert float(m.group(3)) == TE.MANUAL_BACKSTOP_LIQ_SHARE


# ── the builder in node ──────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(NODE, reason="node not installed")
def test_every_mode_reads_the_fake_config_pct_unit():
    r = _node("const cfg = " + json.dumps(FAKE) + r""";
const o = (extra) => Object.assign({ cfg, size: 500, lev: 20, dir: 'LONG', unit: 'pct' }, extra || {});
const out = {};
for (const m of ['FIXED', 'FLOOR', 'MOMENTUM', 'FRENZY', 'BULLRUN', 'BULLRUN_SL']) out[m] = manualExitExplainText(m, o(m === 'FIXED' || m === 'BULLRUN_SL' ? { sl: 1.5, tp: m === 'FIXED' ? 3 : null } : (m === 'FLOOR' ? { tp: 2 } : {})));
out.MS = manualExitExplainText('MOMENTUM', o({ dir: 'SHORT' }));
out.F0 = manualExitExplainText('FIXED', o());
out.fz = slTpUsdFromPct(-2.2, 10000); out.fl = manualStopForLeverage(20, -2.5).stop;
console.log(JSON.stringify(out));""")
    fx, fl, mo, fz, br, bs, ms = (r[k]["text"] for k in ("FIXED", "FLOOR", "MOMENTUM", "FRENZY", "BULLRUN", "BULLRUN_SL", "MS"))
    # FIXED: typed SL / TP, resting maker TP (resting_tp_manual_enabled), global max hold 300 min = 5 h
    assert fx.startswith("Stop −1.50 % (≈ −$150.00 on this $10,000.00 position)") and "TP +3.00 % (≈ +$300.00) as a resting maker limit" in fx
    assert "no other exit" in fx and fx.endswith("max hold 5 h") and r["FIXED"]["why"] == {"sl": None, "tp": None}
    assert "Stop: type your SL (widest −2.50 % at 20×)" in r["F0"]["text"] and "no TP" in r["F0"]["text"]
    # FLOOR: the config floor −2.5 binds at 20× (liquidation 80 % = −3.6)
    assert r["fl"] == -2.5 and fl.startswith("Stop −2.50 % (≈ −$250.00 on this $10,000.00 position) = the leverage floor at 20×")
    assert "TP +2.00 % (≈ +$200.00)" in fl and r["FLOOR"]["why"] == {"sl": "set by the floor stop", "tp": None}
    # MOMENTUM LONG: STRONG_BUY stops, runner trail, hard-TP ladder floor = trigger − offset, EMA13 long off, no-expansion 45 min
    assert mo.startswith("Stop −0.55 % (≈ −$55.00 on this $10,000.00 position), −0.85 % (≈ −$85.00) while the pair's LONG signal is still on (no ATR widening on manual trades)")
    assert "from a +0.33 % peak it trails 0.8× the entry ATR under the peak, never below +0.0700 % (≈ +$7.00)" in mo
    assert "ladder: a +1.10 % peak locks +0.80 % (2 rungs)" in mo and "EMA13-cross exit off for LONGs" in mo
    assert "at 45 min it closes, in profit or not, unless the pair's signal is still on (the clock then restarts)" in mo and mo.endswith("max hold 5 h")
    assert r["MOMENTUM"]["why"] == {"sl": "set by the Momentum exit", "tp": "set by the Momentum exit"}
    # MOMENTUM SHORT: its own runner settings + ladder; EMA13 exit on with the stack-flip condition
    assert "from a +0.36 % peak it trails 0.6× the entry ATR under the peak (at most 30 % of the peak)" in ms and "never below" not in ms
    assert "ladder: a +0.90 % peak locks +0.70 %" in ms and "closes when price crosses EMA13 with the EMA5/8 stack flipped until the runner arms (+0.36 %)" in ms
    assert "until the runner arms" not in mo                                     # the LONG side has no EMA13 exit to suppress
    # FRENZY: frenzy_stop_pct, fixed frenzy_tp_pct (lock off; the trail arm sits above the TP → never reached → not shown), max hold min(300, 240)
    assert fz.startswith("Stop −2.20 % (≈ −$220.00 on this $10,000.00 position)") and r["fz"] == -220
    assert "TP +1.70 % (≈ +$170.00) (software, every tick)" in fz and "trails" not in fz and fz.endswith("max hold 4 h")
    # BULLRUN: base widened by ATR mult up to the cap; BE arm → lock; trail mult; ladder rungs peak:floor
    assert br.startswith("Stop −0.60 % (≈ −$60.00 on this $10,000.00 position) widened to 1.4× the entry ATR, at most −1.10 % (≈ −$110.00)")
    assert "break-even: at a +0.90 % peak the stop moves to +0.15 % (≈ +$15.00), then trails 2.5× the entry ATR under the peak" in br
    assert "ladder: a +3.00 % peak locks +2.60 % … +6.00 % locks +5.20 % (2 rungs)" in br and "no fixed TP" in br and br.endswith("max hold 5 h")
    # BULLRUN_SL: the typed stop until the arm, then the Bull-Run profit side; TP box greyed
    assert bs.startswith("Your stop −1.50 % (≈ −$150.00 on this $10,000.00 position) until the peak reaches +0.90 %") and "break-even:" in bs
    assert r["BULLRUN_SL"]["why"] == {"sl": None, "tp": "set by the Bull-Run exit"}
    assert not any(r[k]["warn"] for k in ("FIXED", "FLOOR", "MOMENTUM", "FRENZY", "BULLRUN", "BULLRUN_SL", "MS"))


@pytest.mark.skipif(NODE, reason="node not installed")
def test_usd_unit_matches_slTpUsdFromPct_and_frenzy_lock_variant():
    cfg = json.loads(json.dumps(FAKE))
    cfg["thresholds"].update(frenzy_lock_arm_pct=3.0, frenzy_lock_floor_pct=2.0, frenzy_lock_trail_pct=1.5, resting_tp_manual_enabled=False)
    r = _node("const cfg = " + json.dumps(cfg) + r""";
const out = {};
out.fx = manualExitExplainText('FIXED', { cfg, size: 333, lev: 7, dir: 'LONG', unit: 'usd', sl: 1.2345, tp: 0.5 }).text;
out.u1 = slTpFmtUsd(slTpUsdFromPct(-1.2345, 333 * 7), true); out.u2 = slTpFmtUsd(slTpUsdFromPct(0.5, 333 * 7), true);
out.fz = manualExitExplainText('FRENZY', { cfg, size: 500, lev: 20, dir: 'LONG', unit: 'usd' }).text;
out.noPos = manualExitExplainText('FLOOR', { cfg, size: '', lev: 20, dir: 'LONG', unit: 'usd' }).text;
console.log(JSON.stringify(out));""")
    assert r["fx"].startswith(f"Stop {r['u1']} (−1.23 %)") and f"TP {r['u2']} (+0.50 %) (software, every tick)" in r["fx"]
    assert r["fz"].startswith("Stop −$220.00 (−2.20 %)") and "at a +3.00 % peak it locks +$200.00 (+2.00 %), then trails 1.50 points under the peak (no fixed TP)" in r["fz"]
    assert "TP" not in r["fz"].replace("no fixed TP", "")
    assert r["noPos"].startswith("Stop −2.50 % = the leverage floor")       # no Size → % only, never a made-up $


@pytest.mark.skipif(NODE, reason="node not installed")
def test_refusals_match_the_engine_at_every_leverage():
    """The ⚠ "the open will be refused" note appears exactly when the engine refuses the open (manual_exit_levels — leverage floor +
    broker-backstop gap — and, for MOMENTUM, manual_momentum_refusal's stop check), at every leverage 1–125, LONG and SHORT, for the FAKE
    config with the backstop off and on AND for the live trading_config.json (338 review M4)."""
    import services.trading_engine as TE
    from unittest import mock
    levs = list(range(1, 126))
    cases = [("FIXED", 1.5), ("FIXED", 0.9), ("FLOOR", None), ("FRENZY", None), ("BULLRUN", None), ("BULLRUN_SL", 1.5), ("BULLRUN_SL", 0.4), ("MOMENTUM", None)]
    live = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
    cfgs = []
    for bk in (False, True):
        c = json.loads(json.dumps(FAKE)); c["thresholds"]["broker_backstop_enabled"] = bk; cfgs.append(("fake", bk, c))
    cfgs.append(("live", live["thresholds"].get("broker_backstop_enabled"), live))
    for bk_name, bk, cfg in cfgs:
      for side in ("LONG", "SHORT"):
        r = _node("const side = " + json.dumps(side) + ";\n"
                  "const cfg = " + json.dumps(cfg) + ", levs = " + json.dumps(levs) + ", cases = " + json.dumps(cases) + r""";
const out = {};
for (const [m, sl] of cases) out[m + ':' + sl] = levs.map(lev => manualExitExplainText(m, { cfg, size: 100, lev, dir: side, unit: 'pct', sl }).warn);
console.log(JSON.stringify(out));""")
        th = SimpleNamespace(**cfg["thresholds"])
        sb = SimpleNamespace(**cfg["confidence_levels"]["STRONG_BUY"])
        fake_cfg = SimpleNamespace(trading_config=SimpleNamespace(confidence_levels={"STRONG_BUY": sb}, thresholds=th))
        with mock.patch.object(TE, "config", fake_cfg):
            for m, sl in cases:
                eng = []
                for lev in levs:
                    try:
                        TE.manual_exit_levels(th, m, sl, None, lev)
                        bad = m == "MOMENTUM" and TE.manual_momentum_refusal(th, side, 1.0, None, lev) is not None
                    except ValueError:
                        bad = True
                    eng.append(bad)
                k = f"{m}:{'null' if sl is None else sl}"
                assert r[k] == eng, (bk_name, bk, side, m, sl, [l for l, a, b in zip(levs, r[k], eng) if a != b])


@pytest.mark.skipif(NODE, reason="node not installed")
def test_mode_switch_greys_and_submit_never_sends_ignored_values():
    prog = r"""
const _els = {};
function mkEl(v) { return { value: v ?? '', disabled: false, dataset: {}, textContent: '', step: '', placeholder: '', style: {}, title: '',
                           parentElement: { style: {} }, setAttribute() {}, validity: {}, classList: { toggle() {} }, getBoundingClientRect() { return {}; } }; }
['manual-investment', 'manual-leverage', 'manual-sl', 'manual-tp', 'manual-sl-conv', 'manual-tp-conv', 'manual-sl-unit-lbl', 'manual-tp-unit-lbl',
 'manual-sltp-unit-pct', 'manual-sltp-unit-usd', 'manual-risk-line', 'manual-pair', 'manual-direction', 'manual-exit-mode', 'manual-open-btn',
 'manual-exit-explain'].forEach(id => { _els[id] = mkEl(''); });
const document = { getElementById: id => _els[id] || null };
const window = { _cachedConfig: """ + json.dumps(FAKE) + r""", _lastTotalPortfolio: 0, _lastStatusIsPaper: true };
let _toasts = []; function showToast(m) { _toasts.push(m); }
const localStorage = { getItem: () => null, setItem() {} };
let _sent = []; const API_BASE = '';
async function fetch(url, o) { _sent.push(JSON.parse(o.body)); return { ok: true, json: async () => ({}) }; }
async function confirmNear() { return true; }
function closeManualModal() {} function loadOpenOrders() {} function loadTransactions() {} function loadBalance() {}
function manualPairKey() { return 'BTCUSDT'; }
""" + _span("        // ⇄ SLTP-UNITS BEGIN", "        // 🖐 Oct-20 (operator, DECISION_LOG 320)") \
        + _span("        function manualRiskLine() {", "        try { manualExitModeChanged(); } catch (e) {}   // default exit") \
        + _span("        function manualExitModeChanged() {", "        // Close order manually") \
        + _span("        const MANUAL_LIQ_SAFETY_UI", "        function renderManualStopTable() {") + r"""
(async () => {
  const out = { modes: {} };
  _els['manual-pair'].value = 'BTCUSDT'; _els['manual-direction'].value = 'LONG'; _els['manual-investment'].value = '500'; _els['manual-leverage'].value = '20';
  for (const m of ['FIXED', 'MOMENTUM', 'FLOOR', 'FRENZY', 'BULLRUN', 'BULLRUN_SL']) {
    _els['manual-sl'].value = '1.5'; _els['manual-tp'].value = '2';      // typed in BOTH boxes before every switch
    _els['manual-exit-mode'].value = m; manualExitModeChanged();
    await openManualPosition();
    const b = _sent[_sent.length - 1];
    out.modes[m] = { sl: b.sl_pct, tp: b.tp_pct, slDis: _els['manual-sl'].disabled, tpDis: _els['manual-tp'].disabled,
                     slConv: _els['manual-sl-conv'].textContent, tpTitle: _els['manual-tp'].title, slOp: _els['manual-sl'].style.opacity,
                     line: _els['manual-exit-explain'].textContent, why: _els['manual-tp'].dataset.why || null };
  }
  console.log(JSON.stringify(out));
})();"""
    r = subprocess.run(["node", "-e", prog], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    o = json.loads(r.stdout)["modes"]
    assert o["FIXED"]["sl"] == 1.5 and o["FIXED"]["tp"] == 2 and not o["FIXED"]["slDis"] and o["FIXED"]["slOp"] == ""
    assert o["FLOOR"]["sl"] is None and o["FLOOR"]["tp"] == 2 and o["FLOOR"]["slConv"] == "set by the floor stop"
    assert o["BULLRUN_SL"]["sl"] == 1.5 and o["BULLRUN_SL"]["tp"] is None and o["BULLRUN_SL"]["why"] == "set by the Bull-Run exit"
    for m, who in (("MOMENTUM", "Momentum"), ("FRENZY", "Frenzy"), ("BULLRUN", "Bull-Run")):
        assert o[m]["sl"] is None and o[m]["tp"] is None and o[m]["slDis"] and o[m]["tpDis"], m
        assert o[m]["slConv"] == f"set by the {who} exit" and o[m]["tpTitle"].startswith(f"Not used: set by the {who} exit") and o[m]["slOp"] == "0.4"
    for m in o:
        assert o[m]["line"].startswith("📖 Stop") or o[m]["line"].startswith("📖 Your stop"), m
    assert "TP +2.00 %" not in o["MOMENTUM"]["line"] and "−1.50 %" not in o["FRENZY"]["line"]   # an ignored box never feeds the line


@pytest.mark.skipif(NODE, reason="node not installed")
def test_inline_scripts_pass_node_check():
    blocks = re.findall(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', HTML, re.S)
    assert blocks
    for i, b in enumerate(blocks):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(b); fn = f.name
        try:
            r = subprocess.run(["node", "--check", fn], capture_output=True, text=True)
            assert r.returncode == 0, f"script block {i}: {r.stderr[:400]}"
        finally:
            os.unlink(fn)


# ── 338 review ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_i1_save_refreshes_the_cached_config_and_the_lines():
    sv = _span("async function saveConfig()", "// (Backtest engine removed)")
    ok = sv[sv.index("if (res.ok) {"):sv.index("} else {", sv.index("if (res.ok) {"))]
    assert "const _cr = await fetch(`${API_BASE}/api/config`);" in ok
    assert "window._cachedConfig = await _cr.json();" in ok and "manualRiskLine();" in ok
    assert ok.index("showToast('Configuration saved successfully')") < ok.index("window._cachedConfig = await _cr.json();")
    assert "re-read at page load and after every successful Save" in HTML     # the tooltip claim, now true


@pytest.mark.skipif(NODE, reason="node not installed")
def test_i2_no_config_no_numbers():
    r = _node(r"""
const out = {};
for (const [k, cfg] of [['none', undefined], ['empty', {}], ['noTh', { investment: { max_holding_time_minutes: 60 } }]])
  out[k] = ['FIXED', 'FLOOR', 'MOMENTUM', 'FRENZY', 'BULLRUN', 'BULLRUN_SL'].map(m => manualExitExplainText(m, { cfg, size: 500, lev: 125, dir: 'LONG', unit: 'usd', sl: 9 }));
console.log(JSON.stringify(out));""")
    for k in ("none", "empty", "noTh"):
        for x in r[k]:
            assert x["text"] == "settings not loaded yet — reopen in a moment" and x["warn"] is False and not re.search(r"\d", x["text"])
    assert r["none"][2]["why"] == {"sl": "set by the Momentum exit", "tp": "set by the Momentum exit"}   # the greyed note still says who sets it


@pytest.mark.skipif(NODE, reason="node not installed")
def test_m3_refusal_advice_per_mode():
    cfg = json.loads(json.dumps(FAKE)); cfg["thresholds"]["broker_backstop_enabled"] = True
    r = _node("const cfg = " + json.dumps(cfg) + r""";
const t = (m, lev, sl) => manualExitExplainText(m, { cfg, size: 100, lev, dir: 'LONG', unit: 'pct', sl }).text;
console.log(JSON.stringify({ fx: t('FIXED', 50, 1.5), bs: t('BULLRUN_SL', 50, 1.5), mo: t('MOMENTUM', 50), fz: t('FRENZY', 50), fl: t('FLOOR', 50) }));""")
    assert r["fx"].endswith("the open will be refused (lower the leverage or use a tighter stop)")
    assert r["bs"].endswith("the open will be refused (lower the leverage or use a tighter stop)")
    assert "⚠ the momentum stop's −1.10 % cap is checked against the exchange safety stop, which can rest at most −1.38 % from the entry at 50×" in r["mo"]
    assert r["fz"].endswith("(lower the leverage or use Custom SL/TP)") and "too close to its −1.20 % stop" in r["fl"]   # FAKE: sl_atr_widen_floor_pct −1.10 → the momentum cap reads from config
