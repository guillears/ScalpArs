"""🛑 Oct-10: tests for the scout WILLY_STOP exit shadow (scripts/scout_willy_stop.py, DECISION_LOG 268). Wraps the module's hermetic selftest
and pins the frozen rule, the hook order (after WILLY_TIMECAP, whose cache it reads) and the gitignored state file."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_willy_stop as WS  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    WS.selftest()
    assert "selftest WILLY_STOP OK" in capsys.readouterr().out
    assert WT._NET_BLOCKED is False and WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap")) and WT._K1 == {}
    assert WS.STATE.endswith(os.path.join("reports", "SCOUT_WILLY_STOP.json"))


def test_frozen_rule_pinned():
    assert (WS.SL_MAIN, WS.SL_CONTEXT, WS.N_MIN, WS.DAYS_MIN, WS.REREAD_N, WS.HIT_MIN, WS.HIT_REREAD, WS.P_MIN, WS.SHARE_MAX, WS.PARITY_MIN,
            WS.BOOT_N, WS.BOOT_SEED, WS.COMBO) == (4.0, (2.0, 3.0, 5.0, 8.0), 20, 8, 40, 8, 16, 0.90, 0.50, 0.90, 4000, 7, ("T125_S4", 1.25, 4.0))


def test_hook_after_timecap_guarded_and_offline():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_willy_timecap as _wt"), src.index("import scout_willy_stop as _wst")
    assert i < j and "except Exception as _wst_e" in src[j:j + 400]
    mod = open(os.path.join(HERE, "scripts", "scout_willy_stop.py")).read()
    assert "ensure_data" not in mod and "urllib" not in mod


def test_state_file_gitignored():
    assert "reports/SCOUT_WILLY_STOP.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
