"""🔁📏 Oct-9: tests for the scout FRENZY_REENTRY_AFTER_WIN + FRENZY_STRETCHED observe lines (scripts/scout_frenzy_entry_lines.py). Wraps the
module's hermetic selftest (synthetic exports + state in a temp dir, sockets blocked, no repo state written; the cut re-derivation reads
reports/MASTER_POOL_stacked.csv only when present) and pins the frozen rule incl. the stretch cut."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_frenzy_entry_lines as FE  # noqa: E402


def test_hermetic_selftest(capsys):
    FE.selftest()
    assert "selftest FRENZY entry lines OK" in capsys.readouterr().out
    assert FE.EXPORT_GLOB == os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")   # globals restored
    assert FE.STATES["REENTRY"].endswith(os.path.join("reports", "SCOUT_FRENZY_REENTRY.json"))
    assert FE.STATES["STRETCHED"].endswith(os.path.join("reports", "SCOUT_FRENZY_STRETCHED.json"))


def test_frozen_rule_pinned():
    assert (FE.STRETCH_CUT, FE.STRETCH_CUT_N) == (13.0, 15)          # P75 of master FRENZY_LONG + WIDE vs-VWAP (12.984, N 15), distribution only
    assert (FE.N_MIN, FE.DAYS_MIN, FE.REREAD_N, FE.BE_REF, FE.BE_MIN_FILLS, FE.BOOT_N, FE.BOOT_SEED) == (15, 8, 30, 51.5, 30, 4000, 7)
    assert FE.COUNTED == ("FRENZY_LONG", "FRENZY_WIDE") and FE.REF_STRAT == "FRENZY_LITE"
    assert set(FE.FAMILY) == {"FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY"} and FE.DEPLOY_GREP == "(DECISION_LOG 250)"


def test_scout_hook_is_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i = src.index("import scout_frenzy_entry_lines as _fe")
    assert "try:" in src[i - 400:i] and "except Exception as _fe_e" in src[i:i + 400]


def test_state_files_gitignored():
    lines = open(os.path.join(HERE, ".gitignore")).read().split("\n")
    assert "reports/SCOUT_FRENZY_REENTRY.json" in lines and "reports/SCOUT_FRENZY_STRETCHED.json" in lines
