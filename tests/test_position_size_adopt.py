"""📐 (DECISION_LOG 572) SIZE ADOPTION — a tracked LIVE position whose size was changed in the Binance app: the row adopts Binance's size
and average entry after 2 consecutive matching sightings, and its resting take profit keeps the SAME LIMIT PRICE for the full new quantity
(operator: "same price, the full quantity")."""
import asyncio, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)

from services.position_size_adopt import size_differs, size_tolerance, confirm_pass, adopt_values, SIZE_TOL_FRAC, pin_target_column
from tests.test_resting_tp import _setup, TICK

MAN_PX = 7.20   # the operator's resting buy limit on a MANUAL SHORT (entry 10.0)


def test_tolerance_and_difference():
    assert size_tolerance(1000, 0.1) == 1000 * SIZE_TOL_FRAC and size_tolerance(10, 1.0) == 1.0
    assert not size_differs(1000, 1004, 0.1) and size_differs(1000, 1006, 0.1) and size_differs(1000, 900, 0.1)
    assert not size_differs(None, 10, 1) and not size_differs(10, 0, 1) and not size_differs("x", 10, 1)


def test_two_pass_confirmation_on_the_same_size_and_gap():
    seen = {}
    assert not confirm_pass(seen, 1, 1500, 1, 1000, 0.0)              # first sighting
    assert not confirm_pass(seen, 1, 1500, 1, 1000, 5.0)              # same size but < 10 s later (a racing run) → not counted
    assert confirm_pass(seen, 1, 1500, 1, 1000, 15.0) and 1 not in seen   # ≥ 10 s later, same size → adopt, cleared
    seen = {}
    assert not confirm_pass(seen, 1, 1500, 1, 1000, 0.0)
    assert not confirm_pass(seen, 1, 1800, 1, 1000, 20.0)            # the size moved again → the count restarts
    assert confirm_pass(seen, 1, 1800, 1, 1000, 40.0)


def test_pin_target_columns():
    assert pin_target_column("MANUAL") == "pattern_fixed_tp_pct" and pin_target_column("FRENZY_LITE") == "frenzy_lite_tp_used"
    assert pin_target_column("FLIP:FAN_RATIO_GATE") == "flip_fan_tp_pct" and pin_target_column("FRENZY_WILLY") is None
    assert pin_target_column("FRENZY_WILLY", exit_override_at="x") == "pattern_fixed_tp_pct"   # an operator override owns the target


def test_adopt_values_use_binance_average_and_row_leverage():
    v = adopt_values(1000, 10.0, 5, 1500, 9.8)
    assert v["quantity"] == 1500 and v["entry_price"] == 9.8 and abs(v["notional_value"] - 14700) < 1e-9 and abs(v["investment"] - 2940) < 1e-9
    assert v["delta_qty"] == 500 and v["from_qty"] == 1000
    assert adopt_values(1000, 10.0, 5, 1500, 0)["entry_price"] == 10.0   # unreadable Binance entry → the row's own
    assert adopt_values(0, 10, 5, 1500, 9.8) is None and adopt_values(1000, 10, 0, 1500, 9.8) is None


def _manual_row(**kw):
    d = dict(pair="HUMAUSDT", direction="SHORT", entry_strategy="MANUAL", manual_exit_mode="NOSTOP", entry_price=10.0, current_price=9.0,
             quantity=1000.0, notional_value=10000.0, investment=2000.0, leverage=5.0, entry_fee=4.5, pattern_fixed_tp_pct=27.0,
             protective_tp_order_id="T1", resting_tp_price=MAN_PX, resting_tp_filled_qty=0.0, backstop_algo_id=None)
    d.update(kw)
    return d


def _pos(contracts, entry=9.8, side="SHORT", sym="HUMA/USDT:USDT"):
    return [{"symbol": sym, "side": side, "contracts": contracts, "entry_price": entry}]


def _run(monkeypatch, row, passes, paper=False, gap_s=20.0, fund=None, fetch=None):
    import services.trading_engine as TE_
    clock = {"t": 1000.0}
    def _mono():
        return clock["t"]   # advanced by gap_s between reconcile passes only
    monkeypatch.setattr(TE_.time, "monotonic", _mono)
    T, eng, engine, S, seed, read, log, blocks = _setup(
        monkeypatch, paper=paper, rows=[row], th_kw=dict(resting_tp_manual_enabled=True, resting_tp_willy_enabled=True),
        fetch=fetch or (lambda oid: {"status": "canceled", "filled": 0.0, "amount": row.get("quantity"), "average": 0.0}),
        place=lambda pair, price: {"id": "T2", "price": price})

    async def _fund(o, px):
        return fund if fund is not None else (float(o.funding_fee_usd or 0), o.funding_accrued_ms)
    eng._funding_at_close = _fund

    async def go():
        await seed()
        out = []
        for p in passes:
            clock["t"] += gap_s
            async with S() as s:
                out.append(await eng.adopt_position_sizes(s, p))
        r = await read(); await engine.dispose(); return out, r
    out, r = asyncio.run(go())
    return out, r[1], log, blocks


def test_added_in_the_app_same_price_new_qty(monkeypatch):
    out, o, log, blocks = _run(monkeypatch, _manual_row(), [_pos(1500.0), _pos(1500.0)])
    assert out == [[], [1]]                                            # adopted on the 2nd matching sighting only
    assert o.quantity == 1500.0 and o.entry_price == 9.8 and abs(o.investment - 1500 * 9.8 / 5) < 1e-6
    assert o.size_adopted_from_qty == 1000.0 and o.size_adopted_at is not None and o.manual_add_count == 1
    assert o.resting_tp_pinned_price == MAN_PX and o.resting_tp_price == MAN_PX and o.protective_tp_order_id == "T2"
    places = [x for x in log if x[0] == "place"]
    assert [x[0] for x in log].index("cancel") < [x[0] for x in log].index("place")   # the old limit comes off first
    assert places == [("place", "HUMAUSDT", "SHORT", 1500.0, MAN_PX)]  # SAME price, the FULL new quantity
    assert o.pattern_fixed_tp_pct is not None and abs(o.pattern_fixed_tp_pct - 27.0) > 0.1   # net % re-derived at the same price
    assert "POSITION_SIZE_ADOPTED" in blocks


def test_reduced_in_the_app(monkeypatch):
    out, o, log, blocks = _run(monkeypatch, _manual_row(), [_pos(600.0, entry=10.0), _pos(600.0, entry=10.0)])
    assert out == [[], [1]] and o.quantity == 600.0 and o.manual_add_count in (None, 0)   # a reduction is not an "add"
    assert [x for x in log if x[0] == "place"] == [("place", "HUMAUSDT", "SHORT", 600.0, MAN_PX)]


def test_idempotent_after_adoption_and_within_tolerance(monkeypatch):
    out, o, log, _ = _run(monkeypatch, _manual_row(), [_pos(1500.0), _pos(1500.0), _pos(1500.0), _pos(1500.0)])
    assert out == [[], [1], [], []] and len([x for x in log if x[0] == "place"]) == 1   # once adopted, nothing more
    out, o, log, _ = _run(monkeypatch, _manual_row(), [_pos(1003.0), _pos(1003.0)])   # within 0.5 % → never touched
    assert out == [[], []] and not [x for x in log if x[0] in ("cancel", "place")] and o.quantity == 1000.0


def test_skips_side_mismatch_closing_paper_and_partial_tp(monkeypatch):
    out, o, log, _ = _run(monkeypatch, _manual_row(), [_pos(1500.0, side="LONG"), _pos(1500.0, side="LONG")])
    assert out == [[], []] and o.quantity == 1000.0                    # a flipped position is the UNTRACKED logic's, never adopted
    out, o, log, _ = _run(monkeypatch, _manual_row(closing_in_progress=True), [_pos(1500.0), _pos(1500.0)])
    assert out == [[], []] and o.quantity == 1000.0
    out, o, log, _ = _run(monkeypatch, _manual_row(resting_tp_filled_qty=200.0), [_pos(1500.0), _pos(1500.0)])
    assert out == [[], []] and o.quantity == 1000.0                    # a partially filled TP: never re-sized automatically
    out, o, log, _ = _run(monkeypatch, _manual_row(is_paper=True), [_pos(1500.0), _pos(1500.0)], paper=True)
    assert out == [[], []] and o.quantity == 1000.0 and not log          # paper: untouched, no exchange call


def test_willy_rows_are_not_price_pinned(monkeypatch):
    row = _manual_row(entry_strategy="FRENZY_WILLY", manual_exit_mode=None, direction="LONG", pair="WILUSDT", entry_price=100.0,
                      quantity=10.0, notional_value=1000.0, investment=125.0, leverage=8.0, entry_fee=0.45, pattern_fixed_tp_pct=None,
                      resting_tp_price=101.40)
    out, o, log, _ = _run(monkeypatch, row, [_pos(15.0, entry=100.5, side="LONG", sym="WIL/USDT:USDT")] * 2)
    assert out == [[], [1]] and o.quantity == 15.0 and o.resting_tp_pinned_price is None   # WILLY re-places at its own sleeve rule


def test_wiring():
    import models
    src = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for c in ("size_adopted_at", "size_adopted_from_qty", "resting_tp_pinned_price"):
        assert c in models.Order.__table__.columns
        assert f"('{c}'" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "await trading_engine.adopt_position_sizes(db, binance_positions)" in src and src.count('"size_adopted_at":') == 2
    assert "resting_tp_pinned_price=None)   # 📐 (572) the operator's new target replaces an adopted price pin" in eng
    assert "resting_tp_pinned_price=None)   # 📐 (572) an operator edit sets a new target" in eng
    assert "function sizeAdoptBadgeHtml(o)" in html and "📐 size from Binance" in html


def test_racing_runs_within_10s_never_adopt(monkeypatch):
    out, o, log, _ = _run(monkeypatch, _manual_row(), [_pos(1500.0), _pos(1500.0)], gap_s=2.0)
    assert out == [[], []] and o.quantity == 1000.0 and not [x for x in log if x[0] in ("cancel", "place")]


def test_partial_tp_fill_during_withdraw_aborts(monkeypatch):
    part = lambda oid: {"status": "canceled", "filled": 200.0, "amount": 1000.0, "average": MAN_PX}   # the limit partly filled meanwhile
    out, o, log, _ = _run(monkeypatch, _manual_row(), [_pos(1500.0), _pos(1500.0)], fetch=part)
    assert out == [[], []] and o.quantity == 1000.0 and float(o.resting_tp_filled_qty or 0) == 200.0 and o.size_adopted_at is None


def test_peak_trough_rebased_and_funding_settled_at_old_size(monkeypatch):
    row = _manual_row(high_price_since_entry=10.2, low_price_since_entry=8.5, peak_pnl=14.0, trough_pnl=-2.0, funding_fee_usd=-1.0)
    out, o, log, _ = _run(monkeypatch, row, [_pos(1500.0), _pos(1500.0)], fund=(-6.5, 1794855600000))
    assert out == [[], [1]]
    # SHORT: peak at the low 8.5, trough at the high 10.2 — re-read against the NEW average 9.8 (not the old 10.0)
    assert 12.5 < o.peak_pnl < 13.5 and -4.5 < o.trough_pnl < -3.5
    assert o.funding_fee_usd == -6.5 and o.funding_accrued_ms == 1794855600000   # settled at the OLD size up to the adoption


def test_lite_row_pins_and_rederives_its_stored_target(monkeypatch):
    row = _manual_row(entry_strategy="FRENZY_LITE", manual_exit_mode=None, direction="LONG", pair="LTXUSDT", entry_price=100.0,
                      quantity=10.0, notional_value=1000.0, investment=125.0, leverage=8.0, entry_fee=0.45, pattern_fixed_tp_pct=None,
                      frenzy_lite_tp_used=3.0, resting_tp_price=103.15)
    out, o, log, _ = _run(monkeypatch, row, [_pos(15.0, entry=99.0, side="LONG", sym="LTX/USDT:USDT")] * 2)
    assert out == [[], [1]] and o.resting_tp_pinned_price == 103.15
    assert [x for x in log if x[0] == "place"] == [("place", "LTXUSDT", "LONG", 15.0, 103.15)]
    assert o.frenzy_lite_tp_used > 3.0   # the same price is a higher net % on the lower average → the software TP sits on it too


def test_bot_add_drops_the_pin():
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "resting_tp_pinned_price=None)   # 📐 (572 review 5) the add re-bases the target" in eng
