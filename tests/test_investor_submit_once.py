"""Operator 2026-10-23: removed Guille, added Guille again → an "Investor 'Guille' already exists" error, yet Guille was added. CloudWatch:
four POST /api/investors in one second (two submits × preview + real) — the second submit hit the 409. Pins the guard (DECISION_LOG 352):
every investor-modal button goes through _investorSubmitOnce, which drops a click while a submit is pending (Deposit / Withdraw / Transfer
would otherwise register the money twice) and always releases (finally). The guard is executed in node when available."""
import json
import os
import re
import shutil
import subprocess

import pytest

HTML = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()
SUBMITS = ("submitAddInvestor", "submitEditInvestor", "submitDeposit", "submitTransfer", "submitWithdraw", "submitRemoveInvestor")


def test_every_investor_button_goes_through_the_guard():
    for f in SUBMITS:
        assert HTML.count(f'onclick="_investorSubmitOnce({f})"') == 1, f
        assert f'onclick="{f}()"' not in HTML, f
        assert len(re.findall(rf"(?<!function )(?<![.\w]){f}\(\)", HTML)) == 0, f"{f} called outside the guard"


def _guard_src():
    i = HTML.index("let _investorSubmitBusy = false;")
    return HTML[i:HTML.index("\n        }\n", i) + len("\n        }\n")]


def test_guard_shape():
    src = _guard_src()
    assert src.index("if (_investorSubmitBusy) { showToast(") < src.index("_investorSubmitBusy = true;") < src.index("await fn();")
    assert "finally { _investorSubmitBusy = false; }" in src


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_double_click_posts_once_and_the_guard_releases_after_an_error():
    js = "const showToast = () => {};\n" + _guard_src() + """
    (async () => {
        let posts = 0;
        const slow = () => new Promise(r => setTimeout(() => { posts++; r(); }, 20));
        await Promise.all([_investorSubmitOnce(slow), _investorSubmitOnce(slow)]);   // double click while pending
        const a = posts;
        await _investorSubmitOnce(async () => { throw new Error('x'); }).catch(() => {});
        await _investorSubmitOnce(slow);                                            // released after the throw
        console.log(JSON.stringify({a, b: posts}));
    })();"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=20)
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip()) == {"a": 1, "b": 2}
