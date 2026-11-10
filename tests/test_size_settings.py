"""⚙️ best-case re-size → settings (scripts/best_case/size_settings.py, operator 11-10; reviews DL 516)."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "best_case"))
import size_settings as SS  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = {"confidence_levels": {"STRONG_BUY": {"leverage": 20.0}}, "leverage_balance_schedule": "0:20, 25000:15",
       "max_open_positions": 4, "rsi_adx_multiplier_hard_cap": 2.5,
       "frenzy_lite_retry_invest_mult": 1.0, "frenzy_lite_retry_lev_mult": 0.2,
       "frenzy_lite_invest_mult": 1.0, "frenzy_lite_lev_mult": 0.4,
       "flatbase_invest_mult": 0.2, "flatbase_lev_mult": 0.25,
       "momentum_long_invest_mult": 0.2, "momentum_long_lev_mult": 0.25,
       "frenzy_willy_invest_mult": 0.5, "frenzy_willy_lev_mult": 0.25, "frenzy_willy_ht_size_frac": 0.5,
       "frenzy_long_invest_mult": 0.2, "frenzy_long_lev_mult": 0.05}


def test_real_config_shape_loads(tmp_path):
    """the real file nests: sleeve fields under thresholds, schedule / slots under investment, leverage under confidence_levels."""
    c = SS.load_cfg(os.path.join(ROOT, "trading_config.json"))
    assert SS.base_leverage(c) >= 1 and SS.slots(c) >= 1 and SS.schedule_ceilings(c)
    for sl in SS.FIELDS:
        SS.today(c, sl)
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"confidence_levels": {"STRONG_BUY": {"leverage": 10}}, "investment": {"leverage_balance_schedule": "0:20, 25000:15",
                                                                                                    "max_open_positions": 5},
                             "thresholds": {"flatbase_invest_mult": 0.2}}))
    c = SS.load_cfg(str(p))
    assert SS.base_leverage(c) == 10 and SS.slots(c) == 5 and SS.schedule_ceilings(c) == [(0, 20), (25000, 15)]
    assert SS.base_leverage({}) == 20


def test_keep_leverage_scale_investment_rounded_down_to_005():
    r = SS.recommend(CFG, "FLATBASE_SHORT", 2.05)                    # 0.41 → 0.40
    assert (r["new"]["inv"], r["new"]["lev"]) == (0.4, 5) and r["size_x"] <= 2.05
    r = SS.recommend(CFG, "FRENZY_LITE", 0.86)                       # step DOWN: 0.86 → 0.85
    assert (r["new"]["inv"], r["new"]["lev"]) == (0.85, 8)


def test_within_5pct_rounds_down_to_one_slot():
    r = SS.recommend(CFG, "FRENZY_WILLY", 2.05)                      # 1.025 → 1.0 at 5× (the 11-09 call)
    assert (r["new"]["inv"], r["new"]["lev"], r["new"]["lev_set"]) == (1.0, 5, 0.25)
    r = SS.recommend(CFG, "FRENZY_WILLY", 2.10)                      # exactly 1.05 → still one slot at 5×
    assert (r["new"]["inv"], r["new"]["lev"]) == (1.0, 5)


def test_above_one_slot_raises_leverage_never_over_target():
    r = SS.recommend(CFG, "FRENZY_LITE_RETRY", 1.70)                 # 6.8 → 7×, 0.97 → 0.95
    assert (r["new"]["inv"], r["new"]["lev"], r["new"]["lev_set"]) == (0.95, 7, 0.35)
    assert r["size_x"] <= 1.70 and not r["flag"]
    for f in (0.5, 0.86, 1.3, 1.7, 2.05, 3.3, 4.09):
        for sl in ("FRENZY_LITE_RETRY", "FRENZY_LITE", "FLATBASE_SHORT", "FRENZY_WILLY"):
            assert SS.recommend(CFG, sl, f)["size_x"] <= f + 1e-9


def test_base_leverage_cap_and_engine_clamp_are_flagged():
    r = SS.recommend(CFG, "FRENZY_LITE_RETRY", 30)                   # 120 → 20×, investment 6 → engine clamps to 2.5
    assert r["new"]["lev"] == 20 and r["new"]["inv"] == 2.5 and "clamps" in r["flag"]
    r = SS.recommend(CFG, "FRENZY_LONG", 0.3)                        # 0.06 → below the engine floor 0.1
    assert r["new"]["inv"] == 0.1 and "clamps" in r["flag"]


def test_momentum_long_moves_leverage_only():
    r = SS.recommend(CFG, "MOMENTUM_LONG", 5)
    assert (r["new"]["inv"], r["new"]["lev"], r["new"]["lev_set"]) == (1.0, 5, 0.25) and r["size_x"] == 5
    r = SS.recommend(CFG, "MOMENTUM_LONG", 10)
    assert (r["new"]["inv"], r["new"]["lev"], r["new"]["lev_set"]) == (1.0, 10, 0.5)
    assert "1.5 slots" in r["flag"]


def test_ht_solved_against_willys_new_settings():
    wn = SS.recommend(CFG, "FRENZY_WILLY", 2.05)["new"]
    r = SS.recommend(CFG, "FRENZY_WILLY_HT", 4.09, willy_new=wn)
    assert r["new"]["inv_set"] == 1.0 and r["new"]["lev"] == 5 and abs(r["size_x"] - 4.0) < 1e-9 and not r["flag"]
    r = SS.recommend(CFG, "FRENZY_WILLY_HT", 8.0, willy_new=wn)
    assert r["new"]["inv_set"] == 1.0 and "capped" in r["flag"]
    r = SS.recommend(CFG, "FRENZY_WILLY_HT", 1.5)                    # no WILLY in the row → today's WILLY: 0.75 frac
    assert r["new"]["inv_set"] == 0.75
    r = SS.recommend(CFG, "FRENZY_WILLY_HT", 0.01)                   # never 0 (the engine reads ≤ 0 as 0.25)
    assert r["new"]["inv_set"] == 0.05


def test_table_rows_flags_and_stops():
    rows = [("C1", {"FLATBASE_SHORT": 2.05, "SPIKE_FADE": 1.4}), ("C5", {"FRENZY_WILLY": 2.05, "FRENZY_WILLY_HT": 4.09}),
            ("C4", {"MOMENTUM_LONG": 10.0})]
    t = SS.table(CFG, rows, {"FLATBASE_SHORT": 2.4, "FRENZY_WILLY_HT": 1.2, "MOMENTUM_LONG": 0.17})
    assert len(t) == 2 + 4                                           # unknown sleeve skipped
    assert "0.4× inv · 5× lev" in t[2] and "2.4 → 4.8" in t[2]
    assert "`frenzy_willy_ht_size_frac` 1" in t[4] and "1.2 → 4.8" in t[4]
    assert "0.17 → 1.70" in t[5] and "⚠" in t[5]


def test_run_best_case_parses_the_resize_lines():
    sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
    import run_best_case as R
    a = ("RESIZE FACTORS (5 % cap, one stop = 5 % of the account): FRENZY_LITE_RETRY ×1.70 (stop now 2.9 %) · FRENZY_LITE ×0.86 (stop now 5.8 %) "
         "· FLATBASE_SHORT ×2.05 (stop now 2.4 %)\n"
         "RESIZE FACTORS 2 (DL 504): MOMENTUM_LONG ×5 / ×10 (one stop now ≈ 0.17 % of the account · ×5 ≈ 0.86 % · ×10 ≈ 1.73 %)\n"
         "RESIZE FACTORS 3 (DL 514): FRENZY_WILLY ×2.05 (stop now 2.4 % → 5 %; ×1.46 after its coin caps) · FRENZY_WILLY_HT ×4.09 "
         "(stop now 1.2 % → 5 %; ×3.27 after its coin caps)")
    f = lambda p: re.search(p, a, re.M)  # noqa: E731
    out = "\n".join(R._settings_block(f(r"^RESIZE FACTORS \(.*?: (.*)$"), f(r"^RESIZE FACTORS 2 \(DL 504\): (.*)$"),
                                      f(r"^RESIZE FACTORS 3 \(DL 514\): (.*)$")))
    rows = [ln for ln in out.split("\n") if ln.startswith("| C")]
    assert [ln.split(" | ")[1] for ln in rows] == ["FRENZY_LITE_RETRY", "FRENZY_LITE", "FLATBASE_SHORT", "MOMENTUM_LONG", "MOMENTUM_LONG",
                                                   "FRENZY_WILLY", "FRENZY_WILLY_HT"]
    assert "¾" not in out and "÷ " in out
