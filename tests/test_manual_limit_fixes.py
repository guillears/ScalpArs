"""🪜 Manual order over the exchange's position limit → one-click corrections (DECISION_LOG 185)."""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import services.trading_engine as TE  # noqa: E402

f = TE.manual_bracket_fixes
MOVR = [(5_000, 20), (10_000, 10), (60_000, 5), (100_000, 4)]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_the_operators_case_500_at_20x_on_movr():
    a, b = f(MOVR, 20, 500)
    assert (a["kind"], a["investment"], a["leverage"], a["notional"]) == ("KEEP_LEVERAGE", 250.0, 20.0, 5000.0)
    assert (b["kind"], b["investment"], b["leverage"], b["notional"]) == ("KEEP_POSITION", 1000.0, 10.0, 10000.0)
    assert a["label"] == "Keep 20× → size $250 (position $5,000)" and b["label"] == "Keep the $10,000 position → 10×, size $1,000"


def test_inside_the_limit_or_no_table_gives_no_fix():
    assert f(MOVR, 20, 250) == [] and f(MOVR, 10, 1000) == [] and f(MOVR, 4, 25_000) == []
    assert f(None, 20, 500) == [] and f([], 20, 500) == [] and f(MOVR, 0, 500) == [] and f(MOVR, 20, 0) == []
    assert f(MOVR, "x", 500) == [] and f(MOVR, 20, float("nan")) == [] and f([("a", 1)], 20, 500) == []


def test_every_fix_is_accepted_by_the_limit_rule_and_never_grows_the_trade_beyond_the_ask():
    for tiers in (MOVR, [(50_000, 75), (250_000, 50), (1_000_000, 20)], [(3_333, 25), (7_777, 12)]):
        for lev in (2, 3, 4, 5, 7, 10, 12, 20, 25, 50, 75, 125):
            for m in (3.7, 55, 250, 251, 500, 999.99, 5_000, 40_000):
                want = m * lev
                for x in f(tiers, lev, m):
                    lim = TE.leverage_bracket_limit(tiers, x["leverage"], x["investment"])
                    assert x["leverage"] <= lim[2] and x["investment"] * x["leverage"] <= lim[3] + 0.01      # the engine would accept it
                    assert x["investment"] >= 1 and x["notional"] <= want + 0.01                             # never a bigger position
                    if x["kind"] == "KEEP_LEVERAGE":
                        assert x["investment"] <= m and x["leverage"] <= lev                                 # never more margin
                    else:
                        assert x["leverage"] < lev and abs(x["notional"] - want) <= x["leverage"] * 0.01 + 0.01


def test_leverage_above_the_pair_maximum_offers_the_maximum():
    a, b = f(MOVR, 50, 100)                                    # $5,000 position asked at 50×; the pair's maximum is 20×
    assert a["kind"] == "KEEP_LEVERAGE" and a["leverage"] == 20 and a["investment"] == 100 and a["label"] == "Use 20× (this pair's maximum) → size $100 (position $2,000)"
    assert (b["kind"], b["investment"], b["leverage"]) == ("KEEP_POSITION", 250.0, 20.0)                  # same $5,000 position, more margin
    assert [x["kind"] for x in f(MOVR, 50, 100, available=200)] == ["KEEP_LEVERAGE"]
    a, b = f(MOVR, 50, 40)                                     # $2,000 position: 20× keeps the size; 20× with $100 keeps the position
    assert (a["investment"], a["leverage"]) == (40, 20) and (b["kind"], b["investment"], b["leverage"]) == ("KEEP_POSITION", 100.0, 20.0)


def test_keep_position_is_dropped_when_the_margin_is_not_available_or_no_bracket_holds_it():
    assert [x["kind"] for x in f(MOVR, 20, 500, available=800)] == ["KEEP_LEVERAGE"]
    assert [x["kind"] for x in f(MOVR, 20, 500, available=1000)] == ["KEEP_LEVERAGE", "KEEP_POSITION"]
    assert [x["kind"] for x in f(MOVR, 20, 6_000)] == ["KEEP_LEVERAGE"]            # $120,000: above every bracket
    assert f([(5, 20)], 20, 0.5) == []                                              # under $1 → nothing worth offering


def test_the_budget_is_the_balance_net_of_the_entry_fee():
    # $1,002 available, $5 fee on the $10,000 position → $997 for margin: the "$1,000 at 10×" fix would be refused → not offered
    assert [x["kind"] for x in f(MOVR, 20, 500, available=1002 - 5)] == ["KEEP_LEVERAGE"]
    assert [x["kind"] for x in f(MOVR, 20, 500, available=1005 - 5)] == ["KEEP_LEVERAGE", "KEEP_POSITION"]
    assert [x["kind"] for x in f(MOVR, 20, 500, available=float("nan"))] == ["KEEP_LEVERAGE"] and f(MOVR, 20, 500, available="x") == []


def test_labels_read_cleanly():
    assert f([(150, 20)], 20, 9)[0]["label"] == "Keep 20× → size $7.5 (position $150)"
    assert f([(50_000, 20), (2_000_000, 10)], 20, 50_000)[1]["label"] == "Keep the $1,000,000 position → 10×, size $100,000"


def test_small_sizes_round_down_to_cents():
    (a,) = f([(150, 20)], 20, 9, available=0)
    assert a["investment"] == 7.5 and math.isclose(a["notional"], 150.0)


def test_wiring_error_endpoint_and_dashboard():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py")).read(); mn = open(os.path.join(ROOT, "main.py")).read()
    ui = open(os.path.join(ROOT, "templates", "index.html")).read()
    assert issubclass(TE.ManualLimitError, ValueError) and TE.ManualLimitError("x", [{"a": 1}]).fixes == [{"a": 1}]
    assert eng.count("raise ManualLimitError(") == 2 and eng.count("manual_bracket_fixes(_tiers, leverage, investment, available - _fee_usdt_est)") == 2
    assert mn.index("except ManualLimitError as e:") < mn.index("except ValueError as e:", mn.index("except ManualLimitError as e:"))
    assert 'content={"detail": str(e), "fixes": e.fixes}' in mn and '@app.get("/api/manual/limits")' in mn
    assert 'id="toast-actions"' in ui and "function manualApplyFix(f, pair)" in ui and 'id="manual-limit-hint"' in ui
    assert ui.count('oninput="manualLimitHint()"') == 2 and 'oninput="manualPairHint(); manualLimitHint()"' in ui
    assert "/api/manual/limits?pair=" in ui and "onClick: () => manualApplyFix(f, _fxPair)" in ui
    body = ui[ui.index("function manualApplyFix(f, pair)"):ui.index("function manualExitModeChanged()")]
    assert "openManualPosition" not in body and "fetch(" not in body                 # a fix only fills the inputs
