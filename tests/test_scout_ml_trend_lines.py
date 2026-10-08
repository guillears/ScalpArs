"""🧭 Oct-8: tests for the scout PAIR_1H_DOWNTREND / TREND_ALIGNED observe lines (scripts/scout_ml_trend_lines.py). Wraps the module's
hermetic selftest (synthetic caches in a temp dir, network blocked, no repo state written; the study-parity step runs only when
reports/study_ml_checklist_frame.pkl + the k5m_full cache exist, else it is skipped inside the selftest) and pins the shared constants."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_b1h_negflank as NF  # noqa: E402
import scout_ml_trend_lines as MT  # noqa: E402


def test_hermetic_selftest(capsys):
    MT.selftest()
    assert "selftest ML trend lines OK" in capsys.readouterr().out
    assert MT._NET_BLOCKED is False and MT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_ml_trend"))   # globals restored


def test_constants_pinned_to_shared_bar():
    assert (MT.N_MIN, MT.DAYS_MIN, MT.REREAD_N, MT.BE_REF, MT.BOOT_N, MT.BOOT_SEED) == \
        (NF.N_MIN, NF.DAYS_MIN, NF.DU_REREAD_N, NF.DU_BE_REF, NF.DU_BOOT_N, NF.DU_BOOT_SEED) == (15, 8, 30, 61.8, 4000, 7)
    assert MT.P1_START == "2026-09-30 00:00" and MT.TA_START == "2026-10-07 00:00" and MT.KEEP_MIN == 0.05 and MT.MAX_REQ == 2


def test_scout_hook_is_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i = src.index("import scout_ml_trend_lines as _mt")
    assert "try:" in src[i - 400:i] and "except Exception as _mt_e" in src[i:i + 400]
