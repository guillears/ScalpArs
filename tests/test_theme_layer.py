"""🎨 Oct-22 dashboard visual refresh (DECISION_LOG 345) — LOOK ONLY.

Pins the theme layer (tokens, dark default, light under data-theme), the header light/dark button, the early no-flash head
script, the Total Portfolio Value sparkline (ids, period keys, <2-point guard, no own request) and — the load-bearing part —
that the refresh removed nothing: every element id in HEAD's index.html is still present, and login.html keeps its form
contract (action, method, field names, the ids its script and the ?error=1 hook use)."""
import os
import re
import subprocess

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX = os.path.join(ROOT, "templates", "index.html")
LOGIN = os.path.join(ROOT, "templates", "login.html")


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def _head(path):
    try:
        return subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except Exception:
        pytest.skip("git HEAD not available")


def _ids(src):
    return set(re.findall(r'\bid="([^"$`{}]+)"', src))


def _theme_block(src):
    m = re.search(r'<style id="scalpars-theme">(.*?)</style>', src, re.S)
    assert m, "theme layer <style id=\"scalpars-theme\"> missing"
    return m.group(1)


# ── index.html ──────────────────────────────────────────────────────────────

def test_theme_block_present_after_existing_styles_with_tokens():
    src = _read(INDEX)
    css = _theme_block(src)
    first_style = src.index("<style>")
    assert src.index('<style id="scalpars-theme">') > first_style, "theme layer must come after the existing styles"
    for tok in ("--bg", "--surface", "--surface-2", "--line", "--fg", "--muted", "--accent", "--pos", "--neg", "--warn",
                "--info", "--violet", "--pos-soft", "--neg-soft", "--warn-soft", "--info-soft", "--violet-soft"):
        assert tok + ":" in css, tok
    assert ':root[data-theme="light"]' in css           # light only when chosen
    assert "body.lt-mode" in css                         # the Live Terminal stays on the dark tokens
    assert "prefers-reduced-motion" in css
    assert "IBM+Plex+Sans" in src and "JetBrains+Mono" in src


def test_theme_layer_maps_core_tailwind_utilities():
    css = _theme_block(_read(INDEX))
    for cls in (r".text-white", r".text-gray-400", r".bg-white\/5", r".border-white\/10", r".text-emerald-400",
                r".text-red-400", r".text-amber-400", r".text-sky-400", r".text-purple-400", r".bg-emerald-500\/20", r".glass"):
        assert ":root " + cls in css, cls


def test_theme_button_in_header_before_logout():
    src = _read(INDEX)
    btn = src.index('id="theme-toggle-btn"')
    assert src.index("<header") < btn < src.index('href="/logout"') < src.index("</header>")
    assert "localStorage.setItem(THEME_KEY" in src and "const THEME_KEY = 'scalpars-theme'" in src


def test_early_head_script_default_dark_no_flash():
    src = _read(INDEX)
    m = re.search(r'<script id="scalpars-theme-early">(.*?)</script>', src, re.S)
    assert m, "early theme script missing"
    assert src.index('id="scalpars-theme-early"') < src.index("</head>")
    assert src.index('id="scalpars-theme-early"') < src.index("cdn.tailwindcss.com")   # before first paint
    body = m.group(1)
    assert "localStorage.getItem('scalpars-theme') === 'light'" in body   # only an explicit light choice changes it
    assert "try" in body and "catch" in body


def test_sparkline_ids_periods_and_guard():
    src = _read(INDEX)
    card = src[src.index('id="total-portfolio"'):src.index('id="total-trades-count"')]
    for i in ("portfolio-spark-wrap", "portfolio-spark", "portfolio-spark-empty", "portfolio-spark-periods", "portfolio-spark-meta"):
        assert f'id="{i}"' in card, i
    for p in ("7D", "30D", "ALL"):
        assert f'data-period="{p}"' in card
    assert 'data-period="24H"' not in card       # only daily history exists → no 24H (operator rule)
    assert "aria-pressed" in card and "not enough history yet" in card
    js = re.search(r'<script id="scalpars-theme-js">(.*?)</script>', src, re.S).group(1)
    assert "const PERIOD_KEY = 'scalpars-spark-period'" in js
    assert "data.length < 2" in js                # guard: fewer than 2 points → hide the chart, keep the buttons
    assert "'7D': 'last 7 days'" in js and "fill=\"' + col" not in js and "<circle" not in js   # style= colours, no stretched dot
    # 🎨 DECISION_LOG 386: return % vs a dashed zero line, a ±2 % minimum range, the hollow now-dot, tooltip, both text exports
    assert "const MIN_SPAN = 4" in js and "stroke-dasharray:3 3" in js and "nowR" in js and "portfolio-spark-tip" in card
    assert src.count("window.scalparsSparkReturns ? window.scalparsSparkReturns()") == 2
    assert "fetch(" not in js                     # reads the calendar data already loaded — no request of its own
    assert "pnlCalData" in js
    assert js.count("try {") >= 4                 # never throws into the dashboard


# ids removed on purpose by a FEATURE removal (not the restyle) — the guard compares against git HEAD, so they show up until committed
INTENTIONALLY_REMOVED = {"config-fz-willy-floor", "config-fz-willy-floor-arm"}   # 🎲²ᵇ DECISION_LOG 358: the WILLY V2 floor removed


def test_no_existing_id_removed():
    head = _ids(_head("templates/index.html"))
    now = _ids(_read(INDEX))
    missing = head - now - INTENTIONALLY_REMOVED
    assert not missing, f"ids removed by the restyle: {sorted(missing)[:20]}"


def test_section_order_unchanged():
    def order(s):
        return [m for m in re.findall(r'<h2[^>]*>(.*?)</h2>', s, re.S)]
    assert order(_head("templates/index.html")) == order(_read(INDEX))


def test_paper_live_switch_reloads_calendar_and_sparkline():
    src = _read(INDEX)
    h = src[src.index("document.getElementById('paper-toggle').addEventListener('change'"):]
    h = h[:h.index("showToast(enabled ? 'Paper trading mode enabled'")]
    assert "loadPerformance();" in h and "loadPnlCalendar();" in h


def test_theme_switch_suppresses_transitions_and_tinted_rows_keep_hover_tint():
    for p in (INDEX, LOGIN):
        src = _read(p)
        assert "html.scalpars-theme-switching *" in _theme_block(src)
        assert "classList.add('scalpars-theme-switching')" in src and "classList.remove('scalpars-theme-switching')" in src
    assert ':root tbody tr:not([class*="bg-"]):hover' in _theme_block(_read(INDEX))


# ── login.html ──────────────────────────────────────────────────────────────

def _form(src):
    m = re.search(r"<form\b[^>]*>", src)
    assert m
    return m.group(0)


def test_login_form_contract_unchanged():
    head, now = _head("templates/login.html"), _read(LOGIN)
    assert _form(head) == _form(now)
    assert 'method="POST"' in _form(now) and 'action="/login"' in _form(now)
    assert re.findall(r'name="([^"]+)"', head) == re.findall(r'name="([^"]+)"', now)
    pw = re.search(r"<input\b[^>]*>", now, re.S).group(0)
    assert 'type="password"' in pw and 'name="password"' in pw and 'id="password"' in pw
    for hook in ('id="error-box"', 'id="error-msg"', "params.get('error') === '1'", "errorBox.classList.remove('hidden')",
                 "sessionStorage.setItem('jarvis_play_welcome', '1')"):
        assert hook in head and hook in now, hook
    assert _ids(head) <= _ids(now)


def test_login_theme_layer_and_toggle():
    src = _read(LOGIN)
    css = _theme_block(src)
    assert ':root[data-theme="light"]' in css and "--accent" in css
    assert src.index('id="scalpars-theme-early"') < src.index("</head>")
    assert "localStorage.getItem('scalpars-theme') === 'light'" in src
    assert 'id="theme-toggle-btn"' in src and "localStorage.setItem('scalpars-theme'" in src
    assert 'SCALP<span class="scalpars-logo-accent">ARS</span>' in src
