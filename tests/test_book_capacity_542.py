"""🧺 DECISION_LOG 542 follow-up — review / study sims model the SHARED book (count ceiling 10, 4 tickets of margin), not the old per-sleeve
slot caps (scripts/book_capacity.py)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import book_capacity as BK  # noqa: E402

LIM = dict(count=10, margin=4.0)


def _src(p):
    return open(os.path.join(ROOT, p)).read()


def test_limits_follow_the_engine():
    assert BK.limits({"investment": {"max_open_positions": 4, "max_open_positions_hard": 10, "redeploy_leftover_enabled": True}}) == LIM
    assert BK.limits({"investment": {"max_open_positions": 4, "max_open_positions_hard": 10, "redeploy_leftover_enabled": False}})["count"] == 4
    assert BK.limits() == LIM, "pinned to today's trading_config.json (4 margin slots, hard 10, redeploy on) — update if the operator changes them"


def test_tickets():
    th = {"frenzy_long_invest_mult": 0.2, "frenzy_willy_invest_mult": 1.0, "frenzy_willy_ht_size_frac": 0.5, "surge_long_invest_mult": 1.0,
          "surge_short_invest_mult": 0.5, "momentum_long_invest_mult": 0.6}
    assert BK.ticket("FRENZY_LONG", th) == 0.2
    assert BK.ticket("FRENZY_WILLY_HT", th) == 0.5
    assert BK.ticket("SURGE_SHORT", th) == 0.5 and BK.ticket("SURGE", th, "SHORT") == 0.5 and BK.ticket("SURGE_LONG", th) == 1.0
    assert BK.ticket("MOMENTUM", th, "LONG") == 0.6 and BK.ticket("MOMENTUM", th, "SHORT") == 1.0 and BK.ticket(None, th) == 1.0
    assert BK.ticket("SOMETHING_NEW", th) == 1.0


def test_tickets_engine_bounds_review():
    th = {"spike_fade_invest_mult": 2.0, "frenzy_long_invest_mult": 0, "surge_long_invest_mult": 5.0, "rsi_adx_multiplier_hard_cap": 2.0,
          "bearrun_invest_mult": 0, "frenzy_willy_ht_size_frac": 7, "momentum_long_invest_mult": 0}
    assert BK.ticket("SPIKE_FADE", th) == 2.0 and BK.ticket("SPIKE_CHASE", {}) == 2.0 and BK.ticket("SPIKE_BOUNCE", {}) == 1.0
    assert BK.ticket("FRENZY_LONG", th) == 0.1                  # an explicit 0 = the 0.1 floor (the engine's sleeve block)
    assert BK.ticket("SURGE_LONG", th) == 2.0                   # ≤ the hard cap
    assert BK.ticket("BEARRUN_SHORT", th) == 1.0                # Bear-Run: 0 / blank → its 1.0 default (`or 1.0`)
    assert BK.ticket("FRENZY_WILLY_HT", th) == 0.25             # an out-of-range HT fraction → 0.25 (frenzy_willy_ht_frac)
    assert BK.ticket(float("nan"), th, "LONG") == 1.0 and BK.ticket("", {"momentum_long_invest_mult": 0.5}, "LONG") == 0.5   # legacy NULL = MOMENTUM


def test_check_margin_count_and_partial():
    assert BK.check([], 1.0, LIM) == (1.0, None)
    assert BK.check([1.0] * 4, 1.0, LIM) == (0.0, "NO_BALANCE")
    assert BK.check([1.0, 1.0, 1.0, 0.6], 1.0, LIM) == (0.4, None)            # a leftover fill is partial ("invest all available")
    assert BK.check([0.2] * 10, 0.2, LIM) == (0.0, "BOOK_FULL")               # cheap fills hit the count with margin free
    assert BK.check([1.0, 1.0, 1.0, 0.95], 1.0, LIM, min_frac=0.1) == (0.0, "NO_BALANCE")


def test_fills_on_empty_book():
    assert BK.fills_on_empty_book(1.0, LIM) == 4                  # a SURGE trigger: margin stops it at 4, not the sleeve cap 10
    assert BK.fills_on_empty_book(0.2, LIM) == 10                 # FRENZY_LONG: the count ceiling
    assert BK.fills_on_empty_book(0.85, LIM) == 5                 # LITE: 4 × 0.85 = 3.4, then a 0.6 leftover fill
    assert BK.fills_on_empty_book(1.0, LIM, cap=2) == 2


def test_scripts_wired():
    s = _src("scripts/surge_bearrun_review.py")
    assert "BK.check(book, BK.ticket(\"SURGE_\" + side, TH), BK.limits(CFG))" in s and "n_open >= int(TH[\"surge_max_slots\"])" not in s
    assert "\"REFUSED_BOOK_FULL\", \"REFUSED_NO_BALANCE\"" in s
    m = _src("scripts/study_surge_missed_opps.py")
    assert "BK.check([o[2] for o in open_pos], TICKET, BOOK)" in m and "len(open_pos) >= SLOTS" not in m and "[:BOOK_N]" in m
    f = _src("scripts/scout_frenzy_exits.py")
    assert "and _book_ok(book_n, book_used, th))" in f and "book_open=book_n, book_used=round(book_used, 4)" in f
    w = _src("scripts/scout_willy_hold_frenzy.py")
    assert "why = _BK.check(used, th[\"tickets\"].get(sl, 1.0), th[\"book\"])[1]" in w and "book=_BK.limits()" in w
    p = _src("scripts/study_tp34_portfolio.py")
    assert "if len(openp) >= COUNT:" in p and "len(openp) >= MAXPOS" not in p and "BK.ticket(SLEEVE_OF.get(r.S, r.S), TH)" in p
    assert "len(openp) >= P.COUNT:" in _src("scripts/study_port7_portfolio.py")
    v = _src("scripts/study_surge_missed_live.py")
    assert "M.BK.check([o[2] for o in open_pos], M.TICKET, M.BOOK)" in v and "[:M.BOOK_N]" in v and "len(open_pos) >= M.SLOTS" not in v


def test_frenzy_green_clock_book_gate():
    import pandas as pd
    import scout_frenzy_exits as F
    th = {"frenzy_long_invest_mult": 0.2}
    B = pd.DataFrame(dict(k=["2026-11-13T10:00:00"] * 4, closed_at=[None] * 4, entry_strategy=["MOMENTUM"] * 4, direction=["SHORT"] * 4))
    n, used = F._book_at("2026-11-13T11:00:00", th, B)
    assert (n, used) == (4, 4.0) and not F._book_ok(n, used, th)        # 4 full tickets → no margin
    assert F._book_ok(9, 1.8, th) and not F._book_ok(10, 2.0, th)       # count ceiling 10
    assert F._book_ok(float("nan"), float("nan"), th)                   # an old stored row without the reading passes
    assert F._book_at("2026-11-13T09:00:00", th, B) == (0, 0.0)         # nothing open before the fills
