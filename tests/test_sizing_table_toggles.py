"""⚖️ Sleeve Sizing table — the HT / EXHAUSTION / EXH WIDE / DISTRIBUTION / FLATBASE rows carry REAL controls (operator 11-19).

They are PROXIES of the switches / mults that live in each sleeve's own box: no id, no config key — they write the source input (the
one loadConfig + saveConfig already walk) and re-dispatch the same event; every render copies the source back. ONE source of truth,
the save payload is unchanged. Plus the 👁 "Hide inactive sleeves" view preference (localStorage, default checked, never saved).
"""
import json
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()

# row → (switch column, toggle source, parent, inv source, lev source)   None = stays a gray computed mirror
ROWS = {
    "fzyht": ("long", "config-fz-willy-ht-enabled", "config-fz-willy-enabled", None, None),
    "exh": ("short", "config-exh-enabled", None, "config-exh-invest-mult", "config-exh-lev-mult"),
    "exhw": ("short", "config-exh-wide-enabled", "config-exh-enabled", None, None),
    "dist": ("short", "config-dist-enabled", None, "config-dist-invest-mult", "config-dist-lev-mult"),
    "fbs": ("short", "config-fbs-enabled", None, "config-fbs-invest-mult", "config-fbs-lev-mult"),
}


def _row(key):
    i = HTML.index(f'<tr data-ss="{key}"')
    return HTML[i:HTML.index("</tr>", i)]


def _cells(row):
    return re.findall(r"<td\b.*?</td>", row, flags=re.S)


def _fn_body(name):
    i = HTML.index(f"async function {name}()")
    j = HTML.index("\n        }\n", i)
    return HTML[i:j]


SAVE = _fn_body("saveConfig")
LOAD = _fn_body("loadConfig")


def _saved(src_id):
    """the id is serialized by saveConfig: directly, or through a *_NUM_FIELDS list that saveConfig maps."""
    if f"getElementById('{src_id}')" in SAVE:
        return True
    for lst in re.findall(r"\b([A-Z_]+_NUM_FIELDS)\.map\(", SAVE):
        decl = HTML[HTML.index(f"const {lst} = ["):]
        decl = decl[:decl.index("];")]
        if f"['{src_id}'," in decl:
            return lst in LOAD
    return False


@pytest.mark.parametrize("key", list(ROWS))
def test_row_has_real_toggle_bound_to_saved_source(key):
    col, src, parent, _inv, _lev = ROWS[key]
    row = _row(key)
    cells = _cells(row)
    assert "ss-ro-sw" not in row, f"{key}: still the read-only on/off text"
    sw_cell = cells[1] if col == "long" else cells[2]
    other = cells[2] if col == "long" else cells[1]
    m = re.search(r'<input type="checkbox" class="ss-proxy sr-only peer" data-src="([^"]+)"( data-parent="([^"]+)")?', sw_cell)
    assert m, f"{key}: no proxy toggle in the {col} column"
    assert m.group(1) == src and (m.group(3) or None) == parent
    assert "peer-checked:bg-emerald-500" in sw_cell                                  # same switch look as the other rows
    assert 'class="text-gray-600">–</span>' in other                                  # the other direction stays a dash
    assert HTML.count(f'id="{src}"') == 1 and _saved(src), src                        # bound to ONE existing, saved source
    if parent:
        assert HTML.count(f'id="{parent}"') == 1


@pytest.mark.parametrize("key", list(ROWS))
def test_inv_lev_editable_or_gray_mirror(key):
    _col, _src, _p, inv, lev = ROWS[key]
    cells = _cells(_row(key))
    for cell, src, cls in ((cells[3], inv, "ss-ro-inv"), (cells[4], lev, "ss-ro-lev")):
        if src is None:
            assert cls in cell and "<input" not in cell                               # HT / WIDE: no size of their own → gray computed
        else:
            m = re.search(r'<input type="number" class="ss-proxy [^"]*" data-src="([^"]+)"', cell)
            assert m and m.group(1) == src and cls not in cell
            assert HTML.count(f'id="{src}"') == 1 and _saved(src), src
    if key == "fzyht":
        assert "roInvTxt: `×${_htf} of WILLY`, roLevTxt: '= WILLY'" in HTML and "S.fzyht = { info:" in HTML


def test_proxies_add_no_id_and_no_config_key():
    proxies = re.findall(r"<input[^>]*\bss-proxy\b[^>]*>", HTML)
    assert len(proxies) == 11                                                         # 5 toggles + 3 × (inv, lev)
    assert all(" id=" not in p for p in proxies)                                      # invisible to load / save (they walk ids)
    for tok in ("ss-proxy", "ss-hide-inactive", "ss_hide_inactive", "data-src"):
        assert tok not in SAVE and tok not in LOAD
    import config as C
    fields = type(C.trading_config.thresholds).model_fields
    cfgj = json.load(open(os.path.join(ROOT, "trading_config.json")))
    assert "ss_hide_inactive" not in fields and "ss_hide_inactive" not in json.dumps(cfgj)


def test_render_greys_and_hides_with_one_condition():
    assert "input[type=checkbox]:not(.ss-proxy)" in HTML                              # proxies judged with their parent, not alone
    assert "tr.querySelectorAll('.ss-ro-sw, input.ss-proxy[type=checkbox]')" in HTML
    assert "tr.classList.toggle('opacity-50', rowOff);" in HTML
    assert "tr.classList.toggle('hidden', hideOff && rowOff && !_ssStickyRows.has(k));" in HTML
    assert "hdr.classList.toggle('hidden', hideOff && !anyShown)" in HTML             # empty sections hide
    assert "if (tr.id === 'ss-totals') return;" in HTML                               # totals always shown
    assert "_ssSyncProxies(body);" in HTML


def test_hide_inactive_checkbox_defaults_checked_and_is_local_only():
    assert HTML.count('id="ss-hide-inactive"') == 1
    tag = re.search(r'<input type="checkbox" id="ss-hide-inactive"[^>]*>', HTML).group(0)
    assert " checked" in tag
    i = HTML.index("function _ssHideInit()"); blk = HTML[i:HTML.index("function renderSleeveSizing()", i)]
    assert "let v = true;" in blk and "if (s === '0') v = false;" in blk
    assert "try { const s = localStorage.getItem(SS_HIDE_KEY)" in blk and "try { localStorage.setItem(SS_HIDE_KEY" in blk and "localStorage.getItem(SS_HIDE_KEY)" in blk and "localStorage.setItem(SS_HIDE_KEY" in blk
    assert "fetch(" not in blk


def _proxy_js():
    i = HTML.index("        // 🔁 Proxy controls")
    return HTML[i:HTML.index("        function renderSleeveSizing() {", i)]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_two_way_sync_and_hide_pref_in_node():
    prog = r"""
const listeners = {};
function el(o) { const e = Object.assign({ dataset: {}, classList: { contains: c => (e.cls || []).includes(c) }, events: [], _l: {},
  addEventListener(t, f) { (this._l[t] = this._l[t] || []).push(f); },
  dispatchEvent(ev) { this.events.push(ev.type); (listeners[ev.type] || []).forEach(f => f({ type: ev.type, target: this })); } }, o); return e; }
const src = { 'config-exh-enabled': el({ id: 'config-exh-enabled', type: 'checkbox', checked: false }),
              'config-exh-wide-enabled': el({ id: 'config-exh-wide-enabled', type: 'checkbox', checked: true }),
              'config-exh-invest-mult': el({ id: 'config-exh-invest-mult', type: 'number', value: '0.5', step: '0.01' }),
              'ss-hidden-n': el({ id: 'ss-hidden-n' }),
              'ss-hide-inactive': el({ id: 'ss-hide-inactive', type: 'checkbox', checked: true }) };
const pSw = el({ type: 'checkbox', cls: ['ss-proxy'], dataset: { src: 'config-exh-enabled' }, checked: false });
const pW = el({ type: 'checkbox', cls: ['ss-proxy'], dataset: { src: 'config-exh-wide-enabled', parent: 'config-exh-enabled' }, checked: false });
const pInv = el({ type: 'number', cls: ['ss-proxy'], dataset: { src: 'config-exh-invest-mult' }, value: '', step: '0.05' });
const trExh = { dataset: { ss: 'exh' } }, trW = { dataset: { ss: 'exhw' } };
const tbody = { querySelectorAll: q => q === '[data-parent="config-exh-enabled"]' ? [pW] : [] };
pSw.closest = q => q === '#sleeve-sizing-body' ? tbody : trExh; pW.closest = q => q === '#sleeve-sizing-body' ? tbody : trW;
let store = {}, storageThrows = false;
globalThis.localStorage = { getItem: k => { if (storageThrows) throw new Error('x'); return k in store ? store[k] : null; },
                            setItem: (k, v) => { if (storageThrows) throw new Error('x'); store[k] = v; } };
globalThis.Event = class { constructor(t) { this.type = t; } };
globalThis.document = { activeElement: null, getElementById: id => src[id] || null,
  addEventListener(t, f) { (listeners[t] = listeners[t] || []).push(f); } };
""" + _proxy_js() + r"""
const body = { querySelectorAll: () => [pSw, pW, pInv] };
const out = {};
_ssSyncProxies(body);
out.sync0 = [pSw.checked, pW.checked, pW.disabled, pInv.value, pInv.step];
// toggle the proxy: a click fires input then change
pSw.checked = true; listeners.input.forEach(f => f({ type: 'input', target: pSw })); listeners.change.forEach(f => f({ type: 'change', target: pSw }));
out.fwd = [src['config-exh-enabled'].checked, src['config-exh-enabled'].events.join(',')];
out.sticky = [..._ssStickyRows].sort();   // the toggled row + its child stay visible
_ssSyncProxies(body); out.childEnabled = pW.disabled;
// type in the number proxy (focused → never overwritten by the sync)
pInv.value = '0.3'; document.activeElement = pInv; listeners.input.forEach(f => f({ type: 'input', target: pInv }));
out.num = [src['config-exh-invest-mult'].value, src['config-exh-invest-mult'].events.join(',')];
// source box changed elsewhere → table follows
document.activeElement = null; src['config-exh-invest-mult'].value = '0.7'; src['config-exh-enabled'].checked = false; _ssSyncProxies(body);
out.back = [pInv.value, pSw.checked, pW.disabled];
// blur after a rejected entry: the proxy shows the source again
pInv.value = ''; listeners.focusout.forEach(f => f({ type: 'focusout', target: pInv })); out.blur = pInv.value;
// hide preference: default checked, uncheck remembered, storage failure → default checked
_ssHideInit(); out.hideDefault = src['ss-hide-inactive'].checked;
src['ss-hide-inactive'].checked = false; src['ss-hide-inactive']._l.change[0](); out.stored = store.ss_hide_inactive;
out.stickyCleared = _ssStickyRows.size;
// "N hidden — show": unchecks the box, which saves + dispatches change
src['ss-hide-inactive'].checked = true; store = {}; src['ss-hide-inactive'].dispatchEvent = function (ev) { this.events.push(ev.type); this._l.change.forEach(f => f()); };
src['ss-hidden-n']._l.click[0](); out.show = [src['ss-hide-inactive'].checked, store.ss_hide_inactive, src['ss-hide-inactive'].events.join(',')];
delete src['ss-hide-inactive'].dataset.ready; src['ss-hide-inactive'].checked = true; _ssHideInit(); out.remembered = src['ss-hide-inactive'].checked;
storageThrows = true; delete src['ss-hide-inactive'].dataset.ready; _ssHideInit(); out.throwDefault = src['ss-hide-inactive'].checked;
console.log(JSON.stringify(out));
"""
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert r["sync0"] == [False, True, True, "0.5", "0.01"]         # child disabled while EXHAUSTION is off · step = the source's
    assert r["fwd"] == [True, "input,change"]                       # one forward per click, same events on the source
    assert r["childEnabled"] is False
    assert r["sticky"] == ["exh", "exhw"]                           # toggled row + its child never vanish under the cursor
    assert r["blur"] == "0.7" and r["stickyCleared"] == 0
    assert r["show"] == [False, "0", "change"]
    assert r["num"] == ["0.3", "input"]
    assert r["back"] == ["0.7", False, True]
    assert r["hideDefault"] is True and r["stored"] == "0" and r["remembered"] is False and r["throwDefault"] is True


def test_review_fixes_wired():
    assert HTML.count('id="ss-hidden-n"') == 1
    assert HTML.index('id="ss-hidden-n"') - HTML.index('id="ss-hide-inactive"') < 200       # right after the checkbox
    assert "hn.textContent = hideOff && nHidden > 0 ? `· ${nHidden} hidden — show` : '';" in HTML
    assert "tr.classList.toggle('hidden', hideOff && rowOff && !_ssStickyRows.has(k));" in HTML
    assert "_ssStickyRows.clear();" in HTML
    assert "document.addEventListener('input', _ssMarkSticky, true);" in HTML and "document.addEventListener('change', _ssMarkSticky, true);" in HTML
    assert "document.addEventListener('focusout'," in HTML and "if (src.step && p.step !== src.step) p.step = src.step;" in HTML
