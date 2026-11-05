"""🎲 (DECISION_LOG 473) best-case LUCK TEST per watch line — scripts/best_case/luck_test.py (pure), its wiring into best_dcr.py,
confidence.py (table ③) and run_best_case.py (table ② luck-checked columns). Hermetic: synthetic cohorts, no network."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "best_case"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import luck_test as LT  # noqa: E402


def _cohort(n=80, n_noise=12, effect=0.0, nz=30, seed=1, per_day=2):
    rng = np.random.default_rng(seed)
    zone = np.zeros(n, bool)
    zone[rng.choice(n, nz, replace=False)] = True
    pct = rng.normal(0.5, 1.0, n) - effect * zone
    d = {"pct": pct, "day": [f"2026-10-{1 + i // per_day:02d}" for i in range(n)], "entry_line_var": np.where(zone, 1.0, 0.0) + rng.normal(0, 0.01, n)}
    for j in range(n_noise):
        d[f"entry_noise_{j}"] = rng.normal(0, 1, n)
    return pd.DataFrame(d), zone


def test_verdict_bands():
    assert LT.verdict(0.05) == LT.REAL and LT.verdict(0.051) == LT.UNCLEAR and LT.verdict(0.20) == LT.UNCLEAR
    assert LT.verdict(0.21) == LT.LUCK and LT.verdict(None) == LT.FEW and LT.verdict(float("nan")) == LT.FEW


def test_day_block_perms_keep_same_day_fills_together():
    days = np.array(["a", "a", "b", "c", "c", "c"])
    for perm in LT.day_block_perms(days, r=30, seed=3):
        assert sorted(perm) == list(range(6))
        src_days = days[perm]                          # the day each position's result came from
        for d in ("a", "b", "c"):                      # a day's results land on consecutive time positions, never split
            pos = np.flatnonzero(src_days == d)
            assert (np.diff(pos) == 1).all()


def test_planted_effect_beats_the_search_and_noise_does_not():
    df, zone = _cohort(effect=1.6)
    r = LT.block_line(df, zone, r=200)
    assert r["verdict"] == LT.REAL and r["p_search"] <= 0.05 and r["delta"] > 1.0 and r["k"] > 100
    df0, zone0 = _cohort(effect=0.0, seed=5)
    r0 = LT.block_line(df0, zone0, r=200)
    assert r0["verdict"] != LT.REAL and r0["p_search"] >= r0["p_single"]           # the search can only make luck likelier


def test_post_fill_columns_are_out_and_1d_p_sits_between():
    df, zone = _cohort(effect=1.0, seed=9)
    df["entry_fee"] = df.pct * 3                                  # a booking column that would leak the result
    df["entry_bracket_cap"] = df.pct
    assert "entry_fee" not in LT.candidate_cols(df) and "entry_bracket_cap" not in LT.candidate_cols(df)
    r = LT.block_line(df, zone, r=300)
    assert r["p_single"] <= r["p_1d"] <= r["p_search"]


def test_too_few_and_unscored_is_not_rest():
    df, zone = _cohort(n=12, nz=6)
    assert LT.block_line(df, zone)["verdict"] == LT.FEW
    d = pd.DataFrame({"entry_x": [1.0, 2.0, 3.0, 4.0, np.nan, np.nan], "pct": 0.0})
    Z, V = LT.family(d, ["entry_x"])
    assert not Z[4:].any() and not V[4:].any()          # unscored fills sit in neither side of any candidate
    assert ((V & ~Z) | Z).sum(axis=0).max() == 4


def test_t_score_does_not_let_tiny_cells_win_by_size():
    """a raw-Δ max over thousands of 5-fill cells beat every real line (the first 11-05 run read p 1.0 everywhere); t keeps a strong
    28-fill line testable."""
    df, zone = _cohort(n=60, nz=28, effect=1.4, n_noise=25, seed=11)
    r = LT.block_line(df, zone, r=200)
    assert r["p_search"] < 0.5 and r["t"] > 3


def test_gain_line_sign_flip_by_day():
    days = np.repeat([f"d{i}" for i in range(20)], 2)
    good = pd.DataFrame({"A": np.full(40, 0.8) + np.linspace(-0.1, 0.1, 40), "B": np.zeros(40)})
    r = LT.gain_line(good, days, "A")
    assert r["verdict"] == LT.REAL and r["p_search"] <= 0.05
    rng = np.random.default_rng(2)
    noise = pd.DataFrame({"A": rng.normal(0, 1, 40), "B": rng.normal(0, 1, 40), "C": rng.normal(0, 1, 40)})
    rn = LT.gain_line(noise, days, "A")
    assert rn["p_search"] >= rn["p_single"]
    assert LT.gain_line(good.iloc[:10], days[:10], "A")["verdict"] == LT.FEW


def test_wiring_best_dcr_confidence_and_report():
    src = open(os.path.join(ROOT, "scripts", "best_case", "best_dcr.py"), encoding="utf-8").read()
    assert "import luck_test as LT" in src and 'luck.csv' in src and "LUCK TOTAL B:" in src and "LUCK CALLS:" in src
    assert "!= LT.LUCK" in src                                                    # only NOT ABOVE LUCK lines are switched off
    assert "if not any(c == LT.LUCK for (s_, _l), c in LCALL.items() if s_ == sl)" in src   # a re-size is off when its sleeve has one
    assert LT.R >= 2000
    import confidence as CF
    CF.LUCK.clear()
    assert CF.luck_cell("X", "Y") == "not run"
    CF.LUCK[("FRENZY_WILLY", "WILLY_GAP")] = dict(verdict="NOT ABOVE LUCK", p_single=0.005, p_search=0.45, n=53)
    CF.LUCK[("SURGE_LONG", "SURGE_BTC_HOT")] = dict(verdict="TOO FEW", p_single=float("nan"), p_search=float("nan"), n=6)
    assert "NOT ABOVE LUCK" in CF.luck_cell("FRENZY_WILLY", "WILLY_GAP") and "full search p 0.45" in CF.luck_cell("FRENZY_WILLY", "WILLY_GAP")
    CF.LUCK[("FRENZY_LONG", "LONG_ADX35")] = dict(verdict="NOT ABOVE LUCK", p_single=0.01, p_1d=0.30, p_search=1.0, n=33)
    assert "1D search p 0.30" in CF.luck_cell("FRENZY_LONG", "LONG_ADX35")
    assert CF.luck_cell("SURGE_LONG", "SURGE_BTC_HOT") == "too few to test (6 trades before the freeze)"
    assert "28,545" in CF.luck_cell("FRENZY_WILLY", "WILLY_GAP")
    assert set(CF.THEORY) == set(CF.LINES)                                        # every line states when its theory was written
    CF.LUCK.clear()
    import run_best_case as RB
    txt = "\n".join(["LUCK CALLS: FRENZY_WILLY:WILLY_GAP NOT ABOVE LUCK",
                     "LUCK TOTAL B: B25 real path -3.850 → -1.000 %/day · master intended size 1.250 → 1.300"] +
                    [f"LUCK {k}: B25 total 0.100 %/day | master total 0.200 · re-sized: none" for k, _ in RB.SCEN if k != "TOTAL B"])
    t = RB.clean(txt, "LUCK")
    assert t["today"] == (-3.85, 1.25) and t["TOTAL B"] == (-1.0, 1.3) and "WILLY_GAP" in t["calls"]


def test_fb_confirm_luck_prices_the_registration_search():
    """(DECISION_LOG 474, operator: "It doesnt make sense that flatbase is REAL with so few trades") — the FB_CONFIRM luck test must
    penalise every entry its registration study tried (waits 1–24 h, selling 1–7 % higher), not 2 alternatives."""
    fb = open(os.path.join(ROOT, "scripts", "best_case", "fb_combo.py"), encoding="utf-8").read()
    assert "WAITS_H = (1, 2, 4, 8, 12, 24)" in fb and "HIGHER_PCT = (1, 2, 3, 4, 5, 6, 7)" in fb
    assert 'rec[f"w{h}"]' in fb and 'rec[f"hi{x}"]' in fb and "if r.te < SHIP_MS:" in fb
    src = open(os.path.join(ROOT, "scripts", "best_case", "best_dcr.py"), encoding="utf-8").read()
    assert '(c[:1] == "w" and c[1:].isdigit()) or (c[:2] == "hi" and c[2:].isdigit())' in src
    assert "lacks the FB_CONFIRM search alternatives" in src                              # a stale csv stops, never the easy test
    d = pd.DataFrame({"A": [1.0] * 20, "B": [np.nan] + [0.0] * 19})
    assert LT.gain_line(d, np.repeat([f"d{i}" for i in range(10)], 2), "A")["dropped"] == 1
    days = np.repeat([f"d{i}" for i in range(10)], 2)     # 15 alternatives of pure noise, the best picked: calibrated over 40 seeds
    hits_single = hits_search = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        alts = pd.DataFrame({f"a{j}": rng.normal(0, 3, 20) for j in range(15)})
        r = LT.gain_line(alts, days, alts.mean().idxmax(), r=500, seed=seed)
        assert r["p_search"] >= r["p_single"]
        hits_single += r["p_single"] <= 0.05
        hits_search += r["p_search"] <= 0.05
    assert hits_search <= 6 and hits_single >= 3 * max(hits_search, 1)   # the search penalty is what keeps a picked best near 5 % REAL
