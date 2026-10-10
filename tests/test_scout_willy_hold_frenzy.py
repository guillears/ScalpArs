"""🔓 Oct-10: tests for the scout HOLD_FRENZY_EXEMPT line (scripts/scout_willy_hold_frenzy.py, DECISION_LOG 266) and the WILLY_HOLD tracker's
dtype fix. Wraps both hermetic selftests and pins the hook order (the exempt line reads the tracker's store, so it runs after it)."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_willy_hold as WH  # noqa: E402
import scout_willy_hold_frenzy as HF  # noqa: E402


def test_hermetic_selftests(capsys):
    HF.selftest()
    WH.selftest()
    out = capsys.readouterr().out
    assert "selftest HOLD_FRENZY_EXEMPT OK" in out and "selftest OK" in out
    assert HF.EXPORT_GLOB.endswith("scalpars_orders_paper_*.csv") and HF.STATE.endswith(os.path.join("reports", "SCOUT_HOLD_FRENZY_EXEMPT.json"))


def test_frozen_rule_pinned():
    assert (HF.FRENZY3, HF.N_MIN, HF.DAYS_MIN, HF.REREAD_N, HF.P_MIN, HF.SHARE_MAX, HF.BOOT_N, HF.BOOT_SEED) == (
        ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"), 15, 8, 30, 0.90, 0.50, 4000, 7)


def test_hook_after_hold_tracker_and_guarded():
    src = open(os.path.join(HERE, "scripts", "scout_frenzy_exits.py")).read()
    i, j = src.index("import scout_willy_hold as _WH"), src.index("import scout_willy_hold_frenzy as _WHF")
    assert i < j and "except Exception as ex" in src[j:j + 300] and "Unavailable this run" in src[j:j + 300]


def test_no_network_in_exempt_line():
    src = open(os.path.join(HERE, "scripts", "scout_willy_hold_frenzy.py")).read()
    assert "urllib" not in src and "klines_1m" not in src and "price_rows" not in src


def test_state_file_gitignored():
    assert "reports/SCOUT_HOLD_FRENZY_EXEMPT.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
