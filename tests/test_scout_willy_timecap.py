"""⏱ Oct-9: tests for the scout WILLY_TIMECAP exit shadow (scripts/scout_willy_timecap.py, DECISION_LOG 264). Wraps the module's hermetic
selftest (synthetic exports, tick / 1m caches and state in a temp dir, network blocked, no repo state written) and pins the frozen rule."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    WT.selftest()
    assert "selftest WILLY_TIMECAP OK" in capsys.readouterr().out
    assert WT._NET_BLOCKED is False and WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap"))   # globals restored
    assert WT.STATE.endswith(os.path.join("reports", "SCOUT_WILLY_TIMECAP.json"))


def test_frozen_rule_pinned():
    assert (WT.CAPS, WT.VERDICT_CAPS, WT.N_MIN, WT.DAYS_MIN, WT.REREAD_N, WT.P_MIN, WT.SHARE_MAX, WT.BOOT_N, WT.BOOT_SEED) == \
        ((15, 20, 30, 60), (15, 20), 20, 8, 40, 0.90, 0.50, 4000, 7)
    assert (WT.TAKER, WT.PARITY_TOL, WT.MAX_REQ, WT.WEIGHT_STOP, WT.BUDGET_S, WT.DEPLOY_GREP) == (0.045, 0.05, 2, 900, 25.0, "(DECISION_LOG 251)")


def test_scout_hook_is_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i = src.index("import scout_willy_timecap as _wt")
    assert "try:" in src[i - 400:i] and "except Exception as _wt_e" in src[i:i + 400]


def test_state_file_gitignored():
    assert "reports/SCOUT_WILLY_TIMECAP.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
