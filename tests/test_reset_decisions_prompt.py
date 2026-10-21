"""Operator 2026-10-21: the reset no longer downloads the Decisions CSV automatically — it ASKS first (OK = download, Cancel = skip)."""
import os

HTML = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()


def _reset_src():
    i = HTML.index("async function _resetBotInner(direction)")
    return HTML[i:HTML.index("// Close reset dropdown when clicking outside", i)]


def test_download_is_asked_not_automatic():
    src = _reset_src()
    ask = src.index("const _saveDec = confirm('Save a copy of the Decisions CSV first?")
    assert src.index("if (!confirm(confirmMsg)) return;") < ask          # the reset confirm comes first
    dl = src.index("await _downloadDecisionsCsv(15e3)")
    assert ask < src.index("if (_saveDec) {") < dl                        # the download only runs inside the operator's yes
    assert src.count("_downloadDecisionsCsv(") == 1


def test_failed_download_still_asks_before_resetting():
    src = _reset_src()
    assert "Decisions CSV was NOT saved" in src and "Reset anyway?" in src
