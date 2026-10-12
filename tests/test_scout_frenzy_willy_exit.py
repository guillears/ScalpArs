"""🎲➡🔥 Oct-10: tests for the scout FRENZY_WILLY_TP exit shadow (scripts/scout_frenzy_willy_exit.py, DECISION_LOG 267). Wraps the module's
hermetic selftest and pins the frozen rule, the hook order (after WILLY_TIMECAP, whose cache it shares) and the gitignored state file."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import scout_frenzy_willy_exit as FX  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402


def test_hermetic_selftest(capsys):
    FX.selftest()
    assert "selftest FRENZY_WILLY_EXIT OK" in capsys.readouterr().out
    assert WT._NET_BLOCKED is False and WT.MY_CACHE.endswith(os.path.join("backtest_cache", "scout_willy_timecap")) and WT._K1 == {}
    assert FX.STATE.endswith(os.path.join("reports", "SCOUT_FRENZY_WILLY_EXIT.json"))


def test_frozen_rule_pinned():
    assert (FX.SLEEVES, FX.GRID, FX.N_MIN, FX.DAYS_MIN, FX.REREAD_N, FX.P_MIN, FX.SHARE_MAX, FX.PARITY_MIN, FX.BOOT_N, FX.BOOT_SEED) == (
        ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE"),
        (("T1_S2", None, 2.0), ("T1_S25", None, 2.5), ("T1_S3", None, 3.0), ("T125_S2", 1.25, 2.0), ("T125_S25", 1.25, 2.5), ("T125_S3", 1.25, 3.0),
         ("T125_S4", 1.25, 4.0)),
        15, 8, 30, 0.95, 0.50, 0.90, 4000, 7)


def test_hook_after_timecap_and_guarded():
    src = open(os.path.join(HERE, "scripts", "opportunity_scout.py")).read()
    i, j = src.index("import scout_willy_timecap as _wt"), src.index("import scout_frenzy_willy_exit as _fwx")
    assert i < j and "except Exception as _fwx_e" in src[j:j + 400]


def test_state_file_gitignored():
    assert "reports/SCOUT_FRENZY_WILLY_EXIT.json" in open(os.path.join(HERE, ".gitignore")).read().split("\n")


def test_ensure_data_fetches_1m_page_for_undecided_cell(monkeypatch):
    """(DECISION_LOG 272 review I2) a scored 'ticks+1m' fill with an undecided cell (need_1m) gets its 1m page; a decided one does not.
    Hermetic: the fetcher, the ledger and the cooldown are stubbed — no network, no repo state written."""
    import pandas as pd
    calls = []
    monkeypatch.setattr(WT, "load_attempts", lambda: {"done": {}, "fail": {}})
    monkeypatch.setattr(WT, "save_attempts", lambda att: None)
    monkeypatch.setattr(WT, "_have_day", lambda *a, **k: True)                  # no archive job
    monkeypatch.setattr(WT, "_append_1m", lambda *a, **k: None)
    monkeypatch.setattr(WT.NF, "_cooldown_until", lambda: 0)
    monkeypatch.setattr(WT, "_fetch_1m", lambda sym, start, n, now, timeout_s=10.0: (calls.append(sym), (pd.DataFrame(), 1))[1])
    te = 1_791_600_000_000
    f = pd.DataFrame(dict(pair=["AUSDT", "BUSDT"], te=[te, te], ts=pd.to_datetime([te, te], unit="ms"), src=["ticks+1m", "ticks+1m"],
                          need_1m=[True, False]))
    WT.ensure_data(f, 720, te + 2 * 86_400_000, fetch=True)
    assert calls == ["AUSDT"]
