"""🧭🎲 Oct-12: tests for the scout WILLY_BTC1H observe line (scripts/scout_willy_btc1h.py, DECISION_LOG 275). Wraps the module's
hermetic selftest (synthetic BTC 5m series, exports and tick files in a temp dir, network blocked, no repo state written) and pins the
frozen constants and the bar text."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_willy_btc1h as WB  # noqa: E402
import scout_b1h_negflank as NF  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    study_ran = WB.selftest()
    assert "selftest WILLY_BTC1H OK" in capsys.readouterr().out
    if os.path.exists(os.path.join(HERE, "scripts", "study_b19_port_conc_regime.py")):
        assert study_ran, "slope parity vs the study's btc_feats must run when the study module is present"
    assert WT._NET_BLOCKED is False and NF._NET_BLOCKED is False and WT._K1 == {}
    assert WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap"))
    assert WB._CACHE.endswith(os.path.join("reports", "backtest_cache")) and WT.EXPORT_GLOB.endswith("scalpars_orders_paper_*.csv")
    assert WB.STATE.endswith(os.path.join("reports", "SCOUT_WILLY_BTC1H.json"))
    assert WB.ROWS_CSV.endswith(os.path.join("reports", "SCOUT_WILLY_BTC1H.csv"))


def test_frozen_constants_pinned():
    assert (WB.N_MIN, WB.DAYS_MIN, WB.P_MIN, WB.SHARE_MAX, WB.BOOT_N, WB.BOOT_SEED, WB.EMA_SPAN, WB.WARM_BARS) == \
        (15, 8, 0.95, 0.50, 4000, 7, 20, 100)
    assert WB.COUNT_FROM == "the DECISION_LOG 281 deploy" and WB.STRATEGY == "FRENZY_WILLY"     # (281) was COUNT_START 2026-10-12 00:00
    assert WB.count_start() == WT.dl281_floor()
    for s in (WB.BAR, WB.SUB_BAR, WB.__doc__):
        assert "Counted from the DECISION_LOG 281 deploy (one live exit: today's config, +1.25 / −4 / 90)" in s.replace("\n", " ").replace("               ", "")
    assert WB.ARM_STATE == "ARM CANDIDATE (operator decides; never automatic)"


def test_bar_text_pinned():
    for s in (WB.BAR, WB.__doc__):
        assert "pre-registered 2026-10-12, DECISION_LOG 275" in s
    for frag in ("BLOCKED side reaches N ≥ 15 on ≥ 8 distinct UTC days", "blocked avg ≥ kept avg → RETIRED",
                 "ARM CANDIDATE (operator decides; never automatic)", "P(mean < 0) ≥ 0.95 (4,000 resamples, seed 7)",
                 "no single day and no single pair carries ≥ 50 % of the blocked side's gross loss", "KEEP OBSERVING",
                 "Deferred while any fill ≤ the prefix end is OPEN, unpriced, or provisional"):
        assert frag in WB.BAR, frag


def test_dip_sub_pinned():
    assert (WB.DIP_COL, WB.DIP_CUT, WB.SUB_KEY) == ("entry_frenzy_bar_ret_pct", -1.16, "dip_sub")
    assert WB.SUB_STATE == "CANDIDATE (operator decides; never automatic)"
    for s in (WB.SUB_BAR, WB.__doc__):
        assert "pre-registered 2026-10-12, DECISION_LOG 279" in s
    for frag in ("EACH side reaches N ≥ 15 on ≥ 8 distinct UTC days", "deep avg ≤ shallow avg → RETIRED",
                 "P(deep mean − shallow mean > 0) ≥ 0.95 (4,000 resamples, seed 7)", "≥ 50 % of the deep side's gross gain", "missing stamp → unscored"):
        assert frag in WB.SUB_BAR, frag
    assert WB.DIP_COL in WB.COLS and set(WB.WT.COLS) <= set(WB.COLS)


def test_hook_after_lite_stretch_and_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_lite_stretch_reentry as _lsr"), src.index("import scout_willy_btc1h as _wb1h")
    assert i < j and src.index("except Exception as _lsr_e") < j
    assert "try:" in src[src.index("except Exception as _lsr_e"):j] and "except Exception as _wb1h_e" in src[j:j + 400] and "_wb1h.run(now_ms)" in src[j:j + 200]


def test_state_file_gitignored():
    assert "reports/SCOUT_WILLY_BTC1H.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
