"""🧭📊 Oct-13: tests for the scout LITE_BTC1H_FALLING + LONG_NONTOP50 observe lines (scripts/scout_frenzy_wl_lines.py, DECISION_LOG 288).
Wraps the hermetic selftest (synthetic exports / volumes / state in a temp dir, sockets blocked, injected fetcher, no repo state written) and
pins the frozen bar, the zone rules, the counted floor, the guarded hook and the gitignored state file."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_frenzy_wl_lines as WL  # noqa: E402
import scout_b1h_negflank as NF  # noqa: E402


def test_hermetic_selftest(capsys):
    WL.selftest()
    assert "selftest FRENZY WL lines OK" in capsys.readouterr().out
    assert WL.EXPORT_GLOB == os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")          # globals restored
    assert NF.SHARED_COOLDOWN.endswith(os.path.join("backtest_cache", ".binance_cooldown_until")) and len(NF._cooldown_paths()) == 3
    assert WL.STATE.endswith(os.path.join("reports", "SCOUT_FRENZY_WL_LINES.json"))


def test_frozen_bar_pinned():
    assert (WL.SLOPE_CUT, WL.TOP_N, WL.N_MIN, WL.DAYS_MIN, WL.P_MIN, WL.SHARE_MAX, WL.BOOT_N, WL.BOOT_SEED, WL.PENDING_DAYS) == \
        (0.0, 50, 20, 8, 0.95, 0.50, 4000, 7, 7)
    assert WL.LINES == ("LITE_BTC1H_FALLING", "LONG_NONTOP50")
    assert WL.SLEEVE == {"LITE_BTC1H_FALLING": "FRENZY_LITE", "LONG_NONTOP50": "FRENZY_LONG"}
    assert round(WL.breakeven(3.0, 3.0), 1) == 50.0 and round(WL.breakeven(1.5, 3.0), 1) == 66.7
    assert WL.DEPLOY_GREP == "(DECISION_LOG 288)" and WL.LAG_MIN == 10 and WL.deploy_ts() >= pd.Timestamp("2026-10-13 00:10")


def test_zone_rules():
    assert list(WL.lite_group([0.0, -0.01, 0.01, float("nan")])) == ["zone", "zone", "rest", "unscored"]
    csv = {"2026-10-11": {"AUSDT"}, "2026-10-12": {"BUSDT"}}
    assert WL.top50_status("AUSDT", "2026-10-12", csv, {}) is True          # prior-day list counts
    assert WL.top50_status("ZUSDT", "2026-10-12", csv, {}) is False
    assert WL.top50_status("ZUSDT", "2026-10-13", csv, {}) is None          # 10-13 list (10-12 volumes) unknown → unscored


def test_hook_after_tp_revert_and_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_frenzy_tp_revert as _ftr"), src.index("import scout_frenzy_wl_lines as _fwl")
    assert i < j and "try:" in src[j - 400:j] and "except Exception as _fwl_e" in src[j:j + 400]


def test_state_file_gitignored():
    assert "reports/SCOUT_FRENZY_WL_LINES.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")
