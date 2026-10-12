"""🎯↩ Oct-12: tests for the scout FRENZY_TP_REVERT revert gate (scripts/scout_frenzy_tp_revert.py, DECISION_LOG 283 — FRENZY TP +3 → +1.5).
Wraps the hermetic selftest (temp tick files, network blocked, no repo state written) and pins the frozen bar, the counted floor, the hook
order (after FRENZY_WILLY_TP's fetch pass) and the gitignored state file; also the FRENZY_WILLY_TP re-base to the DL 283 floor."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_frenzy_tp_revert as TR  # noqa: E402
import scout_frenzy_willy_exit as FX  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    TR.selftest()
    assert "selftest FRENZY_TP_REVERT OK" in capsys.readouterr().out
    assert WT._NET_BLOCKED is False and WT._K1 == {} and WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap"))
    assert TR.STATE.endswith(os.path.join("reports", "SCOUT_FRENZY_TP_REVERT.json"))


def test_frozen_bar_pinned():
    assert (TR.OLD_EXIT, TR.N_MIN, TR.DAYS_MIN, TR.P_MIN, TR.SHARE_MAX, TR.PARITY_MIN, TR.BOOT_N, TR.BOOT_SEED) == \
        ((3.0, 3.0, 720), 20, 8, 0.90, 0.50, 0.90, 4000, 7)
    assert TR.GROUPS == ("POOLED", "FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
    assert TR.DEPLOY_GREP == "(DECISION_LOG 283)" and TR.LAG_MIN == 10
    assert TR.deploy_ts() >= pd.Timestamp("2026-10-12 00:10")


def test_frenzy_willy_tp_rebased_to_dl283():
    assert FX.DEPLOY_GREP == "(DECISION_LOG 283)" and FX.DEPLOY_LAG_MIN == 10 and FX.KEEP_LIVE == "KEEP today's exit"
    assert FX.deploy_ts() >= pd.Timestamp("2026-10-12 00:10")
    assert "KEEP +3 / −3" not in open(os.path.join(HERE, "scripts", "scout_frenzy_willy_exit.py"), encoding="utf-8").read()


def test_hook_after_frenzy_willy_tp_and_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_frenzy_willy_exit as _fwx"), src.index("import scout_frenzy_tp_revert as _ftr")
    assert i < j and "try:" in src[j - 400:j] and "except Exception as _ftr_e" in src[j:j + 400]


def test_state_file_gitignored():
    assert "reports/SCOUT_FRENZY_TP_REVERT.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
