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


def test_s4_cells_pinned():
    """DECISION_LOG 278: CAP60 / CAP90 on today's −4 stop — own cohort (99b9048 + 10 min), own keys, bar text word for word."""
    assert (WT.S4_CELLS, WT.S4_PARITY_MIN, WT.S4_LAG_MIN, WT.S4_DEPLOY_GREP, WT.S4_DEPLOY_PIN) == \
        ((("CAP60", None, 60), ("CAP90", None, 90), ("T125C60", 1.25, 60), ("T125C90", 1.25, 90)), 0.90, 10, "(DECISION_LOG 271)",
         "2026-10-12 01:49:51")
    assert WT.S4_KEY == {"CAP60": "S4_CAP60", "CAP90": "S4_CAP90", "T125C60": "S4_T125C60", "T125C90": "S4_T125C90"}
    assert WT.S4_KEEP == {"CAP60": "KEEP 120", "CAP90": "KEEP 120", "T125C60": "KEEP TODAY", "T125C90": "KEEP TODAY"}
    for frag in ("N ≥ 20 counted fills on ≥ 8 days", "day-bootstrap P ≥ 0.90", "no fill > 50 % of Σ Δ", "untouched fills Δ = 0",
                 "walker parity ≥ 90 %", "mean Δ ≤ 0 → keep today", "Compared against today's +1 / −4 / 120",
                 "hold the freeze while their tick archive / 1m page is still due", "WALKER PARITY / INTEGRITY FAILED"):
        assert frag in WT.S4_BAR_T125
    assert WT.S4_LABEL == "pre-registered 2026-10-12, DECISION_LOG 278"
    for frag in ("at the first prefix with N ≥ 20 counted fills on ≥ 8 distinct UTC days, decided ONCE",
                 "CANDIDATE (operator decides; never automatic)", "day-bootstrap P(Δ > 0) ≥ 0.90 (4,000 resamples, seed 7)",
                 "no single fill carries > 50 % of Σ Δ", "fills the cap never touches show Δ = 0 (walker integrity)",
                 "walker parity vs live ≥ 90 %", "else, mean Δ ≤ 0 → KEEP 120; else KEEP OBSERVING",
                 "Provisional (1m-part) or unpriced fills in the prefix hold the freeze while their tick archive / 1m page is still due.",
                 "WALKER PARITY / INTEGRITY FAILED"):
        assert frag in WT.S4_BAR
    assert set(WT.S4_KEY.values()).isdisjoint({"first", "reread"})          # never collides with the CAP15 / CAP20 frozen keys
