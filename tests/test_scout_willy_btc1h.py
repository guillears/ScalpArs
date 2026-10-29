"""🧭🎲 Oct-12: tests for the scout WILLY_BTC1H observe line (scripts/scout_willy_btc1h.py, DECISION_LOG 275). Wraps the module's
hermetic selftest (synthetic BTC 5m series, exports and tick files in a temp dir, network blocked, no repo state written) and pins the
frozen constants and the bar text."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import pandas as pd  # noqa: E402
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
    assert WB.STATE.endswith(os.path.join("reports", "SCOUT_WILLY_BTC1H_V2B.json"))   # 🔁 (409) fresh file for the re-based read
    assert WB.ROWS_CSV.endswith(os.path.join("reports", "SCOUT_WILLY_BTC1H.csv"))


def test_frozen_constants_pinned():
    assert (WB.N_MIN, WB.DAYS_MIN, WB.P_MIN, WB.SHARE_MAX, WB.BOOT_N, WB.BOOT_SEED, WB.EMA_SPAN, WB.WARM_BARS) == \
        (15, 8, 0.95, 0.50, 4000, 7, 20, 100)
    assert WB.COUNT_FROM == "the DECISION_LOG 281 deploy" and WB.STRATEGY == "FRENZY_WILLY"     # (281) was COUNT_START 2026-10-12 00:00
    assert WB.count_start() == WT.dl281_floor()
    for s in (WB.BAR, WB.SUB_BAR, WB.__doc__):
        assert "Counted from the DECISION_LOG 281 deploy; priced under today's live exit V2b since DECISION_LOG 409" in s.replace("\n", " ").replace("               ", "")
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
    gi = open(os.path.join(HERE, ".gitignore")).read().split("\n")
    assert "reports/SCOUT_WILLY_BTC1H.json" in gi and "reports/SCOUT_WILLY_BTC1H_V2B.json" in gi


# ── 🔁 RE-BASE to today's live exit V2b (DECISION_LOG 409) ───────────────────────────────────────────────────────────────────────
def _row(**k):
    import types
    base = dict(pair="AUSDT", te=1_761_000_000_000, E=1.0, pct=1.25, why="FRENZY_TP", entry_fee=0.045, quantity=100.0,
                entry_frenzy_spike_at=None, frenzy_willy_tp_target_pct=None, entry_frenzy_willy_spike_high=None)
    base.update(k)
    return types.SimpleNamespace(**base)


def test_v2b_target_stored_stamped_rebuilt_missing():
    """stored target first; else the engine's target from the stamped H; else from a rebuilt H; else UNPRICED with the reason."""
    import scout_willy_exit_v2 as V2
    assert WB.v2b_target(_row(frenzy_willy_tp_target_pct=3.5)) == (3.5, "stored")
    t, src = WB.v2b_target(_row(entry_frenzy_willy_spike_high=1.05))
    assert src == "stamped H" and t == V2.target_at_k({"entry_price": 1.0, "entry_frenzy_willy_spike_high": 1.05, "entry_fee": 0.045,
                                                        "quantity": 100.0}, 1.0) and 1.25 < t < 8
    assert WB.v2b_target(_row(entry_frenzy_willy_spike_high=1.001))[0] == 1.25           # below +1.25 → the +1.25 floor
    assert WB.v2b_target(_row(entry_frenzy_willy_spike_high=2.0))[0] == 8.0              # capped at +8
    seen = {}
    def hf(pair, sp_ms, red_ms, tc):
        seen.update(sp=sp_ms, red=red_ms)
        return 1.05, "ticks"
    r = _row(entry_frenzy_spike_at="2025-10-20T22:35:00", te=int(pd.Timestamp("2025-10-20 22:50:08").value // 10**6))
    t, src = WB.v2b_target(r, high_fn=hf)
    assert src == "rebuilt H (ticks)" and t == WB.v2b_target(_row(entry_frenzy_willy_spike_high=1.05))[0]
    assert seen == dict(sp=int(pd.Timestamp("2025-10-20 22:35").value // 10**6), red=int(pd.Timestamp("2025-10-20 22:50").value // 10**6))
    assert WB.v2b_target(r, high_fn=lambda *a: (None, None)) == (None, "no spike high (path data missing)")
    assert WB.v2b_target(_row()) == (None, "no spike time")
    assert WB.v2b_target(_row(entry_frenzy_willy_spike_high=1.05, entry_fee=None)) == (None, "no fee / quantity")


def test_ran_v2b_and_registry_matches_live_exit():
    """parity is read only on fills that ran V2b; the registry walks = today's V2b config (not STALE)."""
    ms = 1_761_200_000_000
    assert WB.ran_v2b(_row(te=ms, frenzy_willy_tp_target_pct=2.0), ms)
    assert not WB.ran_v2b(_row(te=ms - 1, frenzy_willy_tp_target_pct=2.0), ms)            # before the DL 358 deploy
    assert not WB.ran_v2b(_row(te=ms, frenzy_willy_tp_target_pct=None), ms)                # pre-V2 fill
    assert not WB.ran_v2b(_row(te=ms, frenzy_willy_tp_target_pct=2.0, why="WILLY_FLOOR"), ms)
    assert not WB.ran_v2b(_row(te=ms, frenzy_willy_tp_target_pct=2.0), None)
    import observe_registry as OR
    th = OR.thresholds()
    assert OR.diffs(OR.EXIT["scout_willy_btc1h.py"]["walks"], th) == []
    assert (WB.V2B_K, WB.V2B_CAP) == (float(th["frenzy_willy_high_k_pct"]), int(th["frenzy_willy_max_hold_minutes"]))


def test_price_v2b_walks_the_target_and_reads_parity_only_when_it_ran(monkeypatch):
    """price_v2b end to end on a synthetic net-% path: the stored target is hit → TP at the target; parity True only for a fill that RAN
    V2b; no target → NO_TARGET, unpriced."""
    import numpy as np
    te = 1_761_200_000_000
    P = dict(t=np.array([te + 60_000, te + 120_000]), hi=np.array([1.0, 3.0]), lo=np.array([-0.5, 0.5]), cl=np.array([0.5, 2.0]),
             op=np.array([np.nan, np.nan]), bar=np.array([False, False]), end_ms=te + 200 * 60_000, te=te, n_tick=2, n_bar=0, tick_end=te + 200 * 60_000, lim=te + 91 * 60_000)
    monkeypatch.setattr(WB.FWX, "load_path", lambda *a, **k: P)
    r = _row(te=te, frenzy_willy_tp_target_pct=2.5, pct=2.5)
    out = WB.price_v2b(r, None, te)
    assert (out["walk"], out["walk_why"], out["tgt"], out["tgt_src"], out["par_ok"]) == (2.5, "TP", 2.5, "stored", True)
    assert WB.price_v2b(r, None, te + 1)["par_ok"] is False                            # opened before the V2b deploy → no parity case
    nt = WB.price_v2b(_row(te=te), None, te)
    assert nt["walk_why"] == "NO_TARGET" and nt["walk"] != nt["walk"] and nt["src"] is None


def test_price_v2b_target_invariant_or_held(monkeypatch):
    """409 deep review: a fill without H is priced when its walk is the same at +1.25 and +8 (it stopped before reaching +1.25 —
    ICNT 10-16 −4.06); a target-dependent one stays NO_TARGET. A late entry (> 60 s after a bar close) never rebuilds H."""
    import numpy as np
    te = 1_761_000_008_000                                                             # 8 s after a bar close
    def mk(hi):
        return dict(t=np.array([te + 60_000, te + 120_000]), hi=np.array([hi, 0.5]), lo=np.array([-0.5, -4.5]), cl=np.array([0.0, -4.5]),
                    op=np.array([np.nan, np.nan]), bar=np.array([False, False]), end_ms=te + 200 * 60_000, te=te, n_tick=2, n_bar=0,
                    tick_end=te + 200 * 60_000, lim=te + 91 * 60_000)
    monkeypatch.setattr(WB, "spike_high", lambda *a, **k: (None, None))
    r = _row(te=te, entry_frenzy_spike_at="2025-10-23T06:10:00")
    monkeypatch.setattr(WB.FWX, "load_path", lambda *a, **k: mk(0.9))                 # never touches +1.25, then the stop
    out = WB.price_v2b(r, None, te)
    assert out["walk_why"] == "STOP" and out["tgt_src"].startswith("target-invariant") and out["walk"] < -4
    monkeypatch.setattr(WB.FWX, "load_path", lambda *a, **k: mk(2.0))                 # touches +1.25 (and +2) → target matters
    out = WB.price_v2b(r, None, te)
    assert out["walk_why"] == "NO_TARGET" and out["tgt_src"] == "no spike high (path data missing)"
    assert WB.v2b_target(_row(te=te + 60_000, entry_frenzy_spike_at="2025-10-23T06:10:00"))[1].startswith("no spike high (entry > 60 s")
