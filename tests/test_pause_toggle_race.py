"""Operator 2026-10-22: after Pause the button flipped back to RUNNING (the bot WAS paused — two POST /api/pause in the log). A status poll
sent before the click carried is_running=true. Pins the guard: every Pause/Start click bumps _runToggleSeq before and after its request,
and loadStatus applies the run state only when no click happened while it was in flight."""
import os
import re

HTML = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()


def _fn(name):
    i = HTML.index(f"async function {name}(")
    return HTML[i:HTML.index("\n        }\n", i)]


def test_status_poll_skips_run_state_after_a_click():
    src = _fn("loadStatus")
    assert src.index("const _seq0 = _runToggleSeq;") < src.index("await fetch(`${API_BASE}/api/status`)")
    guard = src.index("if (_seq0 === _runToggleSeq) {")
    assert guard < src.index("isRunning = data.is_running;") < src.index("runtimeSeconds = data.runtime_seconds;")
    assert len(re.findall(r"isRunning = data\.is_running", HTML)) == 1          # the only place the poll sets the run state


def test_toggle_bumps_the_counter_around_its_request():
    src = _fn("toggleBot")
    a = src.index("_runToggleSeq++;")
    f = src.index("await fetch(`${API_BASE}${endpoint}`, { method: 'POST' });")
    b = src.index("_runToggleSeq++;", f)
    assert a < f < b < src.index("isRunning = (endpoint === '/api/start');")
    assert "isRunning = !isRunning" not in src                                   # never a blind flip


def test_second_click_while_pending_is_ignored():
    i = HTML.index("async function toggleBot(")
    src = HTML[i:HTML.index("async function resetBot(", i)]
    assert src.index("if (_toggleInFlight) return;") < src.index("_toggleInFlight = true;") < src.index("await fetch(")
    assert "} finally {\n                _toggleInFlight = false;" in src
