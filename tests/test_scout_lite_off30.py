"""🪶 Oct-8: tests for the scout LITE_OFF30H3 observe line (scripts/scout_lite_off30.py, DECISION_LOG 259). Wraps the module's hermetic
selftest (synthetic exports + caches in a temp dir, network blocked, no repo state written; the study-parity step runs only when
reports/LITE_ENTRY_SIGNS_STUDY_2026-10-08.csv + the k5m_full cache exist, else it is skipped inside the selftest) and pins the frozen rule."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_lite_off30 as LO  # noqa: E402


def test_hermetic_selftest(capsys):
    LO.selftest()
    assert "selftest LITE_OFF30H3 OK" in capsys.readouterr().out
    assert LO._NET_BLOCKED is False and LO.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_lite_off30"))   # globals restored
    assert LO.STATE.endswith(os.path.join("reports", "SCOUT_LITE_OFF30H3.json"))


def test_frozen_rule_pinned():
    assert (LO.OFF_PCT, LO.WIN_BARS, LO.N_MIN, LO.DAYS_MIN, LO.BE_REF, LO.BE_MIN_FILLS, LO.BOOT_N, LO.BOOT_SEED) == \
        (3.0, 6, 30, 15, 49.9, 30, 4000, 7)
    assert LO.START == "2026-10-07 00:00" and LO.MAX_REQ == 2 and LO.WEIGHT_STOP == 900 and LO.BUDGET_S == 25.0


def test_scout_hook_is_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i = src.index("import scout_lite_off30 as _lo")
    assert "try:" in src[i - 400:i] and "except Exception as _lo_e" in src[i:i + 400]


def test_state_file_gitignored():
    assert "reports/SCOUT_LITE_OFF30H3.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
