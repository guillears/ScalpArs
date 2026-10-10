"""🎯 Oct-9: tests for the scout WILLY_TP125 exit shadow (scripts/scout_willy_tp125.py, DECISION_LOG 265). Wraps the module's hermetic
selftest (no network, temp tree) and pins the hook order (after WILLY_TIMECAP's fetch pass) and the gitignored state file."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_willy_tp125 as W  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    W.selftest()
    assert "selftest WILLY_TP125 OK" in capsys.readouterr().out
    assert WT._NET_BLOCKED is False and W.STATE.endswith(os.path.join("reports", "SCOUT_WILLY_TP125.json"))   # globals restored
    assert WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap"))
    assert W.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_tp125")) and WT._K1 == {}


def test_frozen_rule_pinned():
    assert (W.TP_ALT, W.BASE_TP, W.REVERT_N, W.N_MIN, W.DAYS_MIN, W.REREAD_N, W.P_MIN, W.SHARE_MAX, W.BOOT_N, W.BOOT_SEED) == (
        1.25, 1.00, 15, 20, 8, 40, 0.90, 0.50, 4000, 7)


def test_scout_hook_after_timecap_and_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_willy_timecap as _wt"), src.index("import scout_willy_tp125 as _w125")
    assert i < j                                                     # cache-only line runs after the line that fetches
    blk = src[j:j + 400]
    assert "except Exception as _w125_e" in blk and "Unavailable this run" in blk


def test_no_network_in_module():
    src = open(os.path.join(HERE, "scripts", "scout_willy_tp125.py")).read()
    assert "urllib" not in src and "ensure_data" not in src and "_fetch" not in src


def test_state_file_gitignored():
    assert "reports/SCOUT_WILLY_TP125.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
