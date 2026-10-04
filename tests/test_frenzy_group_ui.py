"""🔥 Oct-4 collapsible FRENZY group in "Top Pairs by Volume" (operator): the pinned FRENZY rows fold under one header row.

Pinned: ALWAYS starts collapsed on every page load (no storage read/write — page-session toggle only); the header is highlighted, never
auto-expanded, when a FRENZY pair is READY / just opened a FRENZY or WIDE trade; non-FRENZY rows are untouched; ids unique.
"""
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HTML = open(os.path.join(ROOT, "templates", "index.html")).read()


def test_starts_collapsed_and_never_persisted():
    assert "var _frenzyGroupExpanded = false;" in HTML
    i = HTML.index("var _frenzyGroupExpanded"); j = HTML.index("let _refreshCooldown = null;")
    block = HTML[i:j]
    assert "localStorage" not in block and "sessionStorage" not in block
    # the render only ever reads the flag; nothing outside the toggle assigns it
    assert len(re.findall(r"_frenzyGroupExpanded\s*=[^=]", HTML)) == 2          # the declaration + the toggle
    assert "frenzyMark(_fz)" in HTML                                             # the 🔥 badge on each FRENZY row stays


def test_header_markup_ids_unique():
    for i in ("frenzy-group-header", "frenzy-group-chevron"):
        assert HTML.count(f'id="{i}"') == 1
    assert 'colspan="16"' in HTML[HTML.index('id="frenzy-group-header"'):HTML.index('id="frenzy-group-chevron"')]


def _js():
    s = HTML
    i = s.index("                const _fzIdx = data.map"); j = s.index("                _setHTMLIfChanged(tbody, _rowsHtml);", i)
    k = s.index("        var _frenzyGroupExpanded"); m = s.index("        let _refreshCooldown = null;")
    return s[k:m], s[i:j]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_render_groups_highlights_without_expanding():
    fn, grp = _js()
    prog = fn + r"""
function pad(n){return String(n).padStart(2,'0')}
const d = new Date(Date.now() - 3 * 60000);
const lf = pad(d.getUTCMonth()+1)+'-'+pad(d.getUTCDate())+' '+pad(d.getUTCHours())+':'+pad(d.getUTCMinutes())+' WIDE opened';
function render(data){ const _rowsArr = data.map(p => '<tr' + (p.frenzy ? ' class="frenzy-group-row' + (_frenzyGroupExpanded === true ? '' : ' hidden') + '"' : '') + '>' + p.pair + '</tr>');
""" + grp + r"""
return _rowsHtml; }
const out = {
  calm: render([{pair:'AUSDT', frenzy:{in_state:true, ready:false}}, {pair:'CUSDT'}]),
  ready: render([{pair:'AUSDT', frenzy:{in_state:true, ready:true}}, {pair:'CUSDT'}]),
  just: render([{pair:'AUSDT', frenzy:{in_state:false, ready:false, last_fire: lf}}, {pair:'CUSDT'}]),
  none: render([{pair:'CUSDT'}, {pair:'DUSDT'}]),
  old: _frenzyJustOpened({last_fire: '01-01 00:00 opened'}),
  refused: _frenzyJustOpened({last_fire: lf.replace('WIDE opened', 'refused: slots')}),
};
console.log(JSON.stringify(out));
"""
    import json
    r = json.loads(subprocess.run(["node", "-e", prog], capture_output=True, text=True, check=True).stdout)
    assert '🔥 FRENZY pairs (1) · 1 ON · 0 READY' in r["calm"] and 'bg-orange-500/10' in r["calm"]
    assert r["calm"].index('frenzy-group-header') < r["calm"].index('AUSDT') < r["calm"].index('CUSDT')
    assert '<tr class="frenzy-group-row hidden">AUSDT</tr>' in r["calm"] and '<tr>CUSDT</tr>' in r["calm"]
    for k in ("ready", "just"):                                                   # highlighted, still collapsed
        assert 'bg-emerald-500/15' in r[k] and 'frenzy-group-row hidden' in r[k] and 'aria-expanded="false"' in r[k]
    assert '1 just opened' in r["just"]
    assert r["none"] == '<tr>CUSDT</tr><tr>DUSDT</tr>'                           # no FRENZY pair → no header, rows unchanged
    assert r["old"] is False and r["refused"] is False
