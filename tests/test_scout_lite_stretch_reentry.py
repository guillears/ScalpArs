"""📐🔁 Oct-12: tests for the scout LITE_STRETCH_REENTRY observe line (scripts/scout_lite_stretch_reentry.py, DECISION_LOG 272). Wraps the
module's hermetic selftest (synthetic exports + tick files in a temp dir, network blocked, no repo state written) and pins the frozen rule,
the hook order (after FRENZY_WILLY_TP, whose fetch pass fills the tick / 1m caches it reads) and the gitignored state file."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_lite_stretch_reentry as LS  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    LS.selftest()
    assert "selftest LITE_STRETCH_REENTRY OK" in capsys.readouterr().out
    assert WT._NET_BLOCKED is False and WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap")) and WT._K1 == {}
    assert LS._CACHE.endswith(os.path.join("reports", "backtest_cache")) and LS.EXPORT_GLOB.endswith("scalpars_orders_paper_*.csv")
    assert LS.STATE.endswith(os.path.join("reports", "SCOUT_LITE_STRETCH_REENTRY.json"))


def test_frozen_rule_pinned():
    assert (LS.GAP_CUT, LS.K_MIN, LS.N_MIN, LS.DAYS_MIN, LS.P_MIN, LS.SHARE_MAX, LS.BOOT_N, LS.BOOT_SEED, LS.EMA_BARS) == \
        (1.87, 2, 15, 8, 0.95, 0.50, 4000, 7, 300)
    assert LS.STRATEGY == "FRENZY_LITE" and LS.NON_WILLY == ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE")
    assert "N ≥ 15 on ≥ 8 distinct days" in LS.BAR and "RETIRED" in LS.BAR and "never armed automatically" in LS.BAR


def test_hook_after_frenzy_willy_tp_and_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_frenzy_willy_exit as _fwx"), src.index("import scout_lite_stretch_reentry as _lsr")
    assert i < j and "try:" in src[j - 400:j] and "except Exception as _lsr_e" in src[j:j + 400]


def test_state_file_gitignored():
    assert "reports/SCOUT_LITE_STRETCH_REENTRY.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
