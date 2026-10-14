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
        ((("CAP60", 1.0, 60), ("CAP90", 1.0, 90), ("T125C60", 1.25, 60), ("T125C90", 1.25, 90)), 0.90, 10, "(DECISION_LOG 281)",
         "2026-10-12 12:00:00")
    assert WT.S4_KEY == {"CAP60": "S4_CAP60", "CAP90": "S4_CAP90", "T125C60": "S4_T125C60", "T125C90": "S4_T125C90"}
    assert WT.S4_KEEP == {"CAP60": "KEEP TODAY", "CAP90": "KEEP TODAY", "T125C60": "KEEP TODAY", "T125C90": "KEEP TODAY"}   # (281)
    for frag in ("N ≥ 20 counted fills on ≥ 8 days", "day-bootstrap P ≥ 0.90", "no fill > 50 % of Σ Δ", "untouched fills Δ = 0",
                 "walker parity ≥ 90 %", "mean Δ ≤ 0 → keep today", "Compared against today's exit (trading_config.json; +1.25 / −4 / 90 from DECISION_LOG 281",
                 "hold the freeze while their tick archive / 1m page is still due", "WALKER PARITY / INTEGRITY FAILED"):
        assert frag in WT.S4_BAR_T125
    for b in (WT.S4_BAR, WT.S4_BAR_T125):                                       # (281) cohort sentence + n/a rule
        assert "Counted from the DECISION_LOG 281 deploy (today = +1.25 / −4 / 90)" in b and "identical to today's exit, is n/a" in b
    assert WT.S4_LABEL == "pre-registered 2026-10-12, DECISION_LOG 278"
    for frag in ("at the first prefix with N ≥ 20 counted fills on ≥ 8 distinct UTC days, decided ONCE",
                 "CANDIDATE (operator decides; never automatic)", "day-bootstrap P(Δ > 0) ≥ 0.90 (4,000 resamples, seed 7)",
                 "no single fill carries > 50 % of Σ Δ", "fills the cap never touches show Δ = 0 (walker integrity)",
                 "walker parity vs live ≥ 90 %", "else, mean Δ ≤ 0 → KEEP 120; else KEEP OBSERVING",
                 "Provisional (1m-part) or unpriced fills in the prefix hold the freeze while their tick archive / 1m page is still due.",
                 "WALKER PARITY / INTEGRITY FAILED"):
        assert frag in WT.S4_BAR
    assert set(WT.S4_KEY.values()).isdisjoint({"first", "reread"})          # never collides with the CAP15 / CAP20 frozen keys


def test_old_exit_revert_gate_pinned():
    """DECISION_LOG 281: OLD_EXIT (+1 / −4 / 120) vs the new live exit, counted from the DL 281 deploy; frozen revert bar."""
    assert (WT.OLD_EXIT, WT.OLD_KEY, WT.OLD_LABEL, WT.OLD_REVERT, WT.OLD_KEEP) == \
        ((1.0, 4.0, 120), "OLD_EXIT", "pre-registered 2026-10-12, DECISION_LOG 281", "REVERT CANDIDATE (operator decides)", "KEEP NEW")
    for frag in ("FROZEN REVERT BAR (pre-registered 2026-10-12, DECISION_LOG 281)", "N ≥ 20 counted fills on ≥ 8 distinct UTC days",
                 "mean paired Δ (old − new) > 0 with day-bootstrap P(Δ > 0) ≥ 0.90 (4,000, seed 7)", "no fill > 50 % of ΣΔ",
                 "walker parity ≥ 90 %", "the new exit's average (walked) < 0", "Else KEEP NEW", "Not frozen on failed parity or integrity",
                 "hold the freeze while their tick archive / 1m page is still due"):
        assert frag in WT.OLD_BAR, frag
    assert WT.s4_na("T125C90", 1.25, 90) == "= today" and WT.s4_na("CAP90", 1.25, 90) == "" and WT.s4_na("CAP90", 1.25, 75) == "cap > hold"
    assert WT.s4_na("OLD_EXIT", 1.0, 120, 4.0) == "= today" and WT.s4_na("OLD_EXIT", 1.25, 90, 4.0) == ""
    assert WT.dl281_floor is WT.s4_deploy_ts and WT.S4_DEPLOY_GREP == "(DECISION_LOG 281)"
    g = WT._git_ct("(DECISION_LOG 281)")
    assert WT.s4_deploy_ts() == (WT.pd.Timestamp("2026-10-12 12:00:00") if g is None else g) + WT.pd.Timedelta(minutes=10)


def test_s285_cells_pinned():
    """DECISION_LOG 285: T125C30 / LT30_025 / LT30_03 vs today's exit — own floor (DL 285 commit + 10 min), own keys, bar text pinned."""
    assert WT.S285_CELLS == (("T125C30", 1.25, 30, None, None), ("LT30_025", 1.25, None, 30, 0.25), ("LT30_03", 1.25, None, 30, 0.30))
    assert WT.S285_KEY == {"T125C30": "S4_T125C30", "LT30_025": "S4_LT30_025", "LT30_03": "S4_LT30_03"}
    assert (WT.S285_LABEL, WT.S285_DEPLOY_GREP, WT.S285_DEPLOY_PIN) == \
        ("pre-registered 2026-10-13, DECISION_LOG 285", "(DECISION_LOG 285)", "2026-10-13 04:05:00")
    assert set(WT.S285_KEY.values()).isdisjoint(set(WT.S4_KEY.values()) | {"first", "reread", WT.OLD_KEY})
    for frag in ("FROZEN BAR (pre-registered 2026-10-13, DECISION_LOG 285)", "N ≥ 20 counted fills on ≥ 8 distinct UTC days, decided ONCE",
                 "P(mean Δ > 0) ≥ 0.90 (4,000 resamples, seed 7)", "no single fill carries > 50 % of Σ Δ", "walker parity vs live ≥ 90 %",
                 "mean Δ ≤ 0 → KEEP TODAY", "else KEEP OBSERVING", "Never frozen on failed parity or integrity",
                 "Counted from the DECISION_LOG 285 commit + 10 min"):
        assert frag in WT.S285_BAR, frag
    g = WT._git_ct("(DECISION_LOG 285)")
    assert WT.s285_deploy_ts() == (WT.pd.Timestamp("2026-10-13 04:05:00") if g is None else g) + WT.pd.Timedelta(minutes=10)
    assert all(WT.s285_na(n, 1.25, 90) == "" for n in WT.S285_NAMES)


def test_s290_cells_pinned():
    """DECISION_LOG 290: T130C90 … T150C90 (higher WILLY TP on today's −4 stop + hold) — own floor (DL 290 commit + 10 min), own keys,
    bar text pinned (family of 5, operator decides)."""
    assert WT.S290_CELLS == (("T130C90", 1.30), ("T135C90", 1.35), ("T140C90", 1.40), ("T145C90", 1.45), ("T150C90", 1.50))
    assert WT.S290_KEY == {"T130C90": "S4_T130C90", "T135C90": "S4_T135C90", "T140C90": "S4_T140C90", "T145C90": "S4_T145C90",
                           "T150C90": "S4_T150C90"}
    assert (WT.S290_LABEL, WT.S290_DEPLOY_GREP, WT.S290_DEPLOY_PIN) == \
        ("pre-registered 2026-10-14, DECISION_LOG 290", "(DECISION_LOG 290)", "2026-10-14 05:30:00")
    assert set(WT.S290_KEY.values()).isdisjoint(set(WT.S4_KEY.values()) | set(WT.S285_KEY.values()) | {"first", "reread", WT.OLD_KEY})
    for frag in ("FROZEN BAR (pre-registered 2026-10-14, DECISION_LOG 290)", "N ≥ 20 counted fills on ≥ 8 distinct UTC days, decided ONCE",
                 "P(mean Δ > 0) ≥ 0.90 (4,000 resamples, seed 7)", "no single fill carries > 50 % of Σ Δ", "walker parity vs live ≥ 90 %",
                 "mean Δ ≤ 0 → KEEP TODAY", "else KEEP OBSERVING", "FAMILY OF 5", "no cell is armed without the operator",
                 "Never frozen on failed parity or integrity", "Counted from the DECISION_LOG 290 commit + 10 min"):
        assert frag in WT.S290_BAR, frag
    g = WT._git_ct("(DECISION_LOG 290)")
    assert WT.s290_deploy_ts() == (WT.pd.Timestamp("2026-10-14 05:30:00") if g is None else g) + WT.pd.Timedelta(minutes=10)
    assert all(WT.s290_na(n, 1.25, 90) == "" for n in WT.S290_NAMES) and WT.s290_na("T130C90", 1.30, 90) == "= today"
    assert WT.s290_spec("T150C90", 90) == (1.50, 90)
