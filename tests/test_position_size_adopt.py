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


def _run(monkeypatch, row, passes, paper=False, gap_s=20.0, fund=None, fetch=None, fills=None, forced=None, fills_error=False):
    import services.trading_engine as TE_
    clock = {"t": 1000.0}
    calls = {"fills": 0}

    async def _fills(sym, a, b):   # 💥 (574) Binance userTrades stub — offline
        calls["fills"] += 1
        if fills_error:
            raise RuntimeError("read failed")
        now = int(__import__("time").time() * 1000)
        return [dict(r, time=(now - 15_000 if r.get("time") is None else r["time"])) for r in (fills or [])]

    async def _force(sym, a, b):
        return [{"orderId": i} for i in (forced or [])]

    async def _bnb():
        return 600.0
    monkeypatch.setattr(TE_.binance_service, "fetch_reduce_fills", _fills)
    monkeypatch.setattr(TE_.binance_service, "fetch_force_orders", _force)
    monkeypatch.setattr(TE_.binance_service, "get_bnb_price", _bnb)
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
            await asyncio.gather(*list(eng.__dict__.get('_pricing_tasks', {}).values()), return_exceptions=True)   # 💥 (574) background pricing
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



# ─────────────────────────── 💥 (DECISION_LOG 574) a reduction's realised part ───────────────────────────
import datetime as _dt
_OPEN = _dt.datetime.utcnow() - _dt.timedelta(hours=2)   # the row opened before the reduction's fills
from services.position_size_adopt import summarize_reduction, fold_partial, reduce_side, PARTIAL_KIND_LIQ, PARTIAL_KIND_OPERATOR, PARTIAL_KIND_MIXED


def _fill(side, qty, rp, fee, oid, t=None, asset="USDT"):
    return {"side": side, "qty": str(qty), "realizedPnl": str(rp), "commission": str(fee), "commissionAsset": asset, "orderId": oid, "time": t}


def test_classification_and_sums():
    assert reduce_side("SHORT") == "BUY" and reduce_side("LONG") == "SELL" and reduce_side("x") is None
    f = [_fill("BUY", 300, -120.0, 1.2, 9), _fill("BUY", 200, -80.0, 0.8, 9), _fill("SELL", 999, 0, 0, 1)]   # the SELL opened it: never counted
    s = summarize_reduction(f, "SHORT", force_order_ids=[9])
    assert s["qty"] == 500 and s["realised"] == -200.0 and abs(s["fee_usd"] - 2.0) < 1e-9 and s["liq_qty"] == 500 and s["kind"] == PARTIAL_KIND_LIQ
    s = summarize_reduction(f, "SHORT", force_order_ids=[])
    assert s["kind"] == PARTIAL_KIND_OPERATOR and s["liq_qty"] == 0
    s = summarize_reduction([_fill("BUY", 300, -1, 0, 9), _fill("BUY", 300, -1, 0, 7)], "SHORT", force_order_ids=[9])
    assert s["kind"] == PARTIAL_KIND_MIXED and s["liq_qty"] == 300
    s = summarize_reduction([_fill("BUY", 10, -1, 0.01, 5, asset="BNB")], "SHORT", bnb_usdt=600)
    assert abs(s["fee_usd"] - 6.0) < 1e-9
    assert summarize_reduction([_fill("BUY", 10, -1, 0.01, 5, asset="BNB")], "SHORT")["fee_estimated"] is True
    s = summarize_reduction([_fill("BUY", 10, -1, 0, 5, t=50), _fill("BUY", 10, -1, 0, 5, t=150)], "SHORT", since_ms=100, until_ms=200)   # explicit times
    assert s["qty"] == 10 and s["last_ms"] == 150                                     # the window is (since, until]
    assert summarize_reduction(f, "x") is None


def test_fold_partial_on_the_original_notional():
    p, pc = fold_partial(-100.0, 600.0, 10.0, -250.0, 400.0)               # 600 left + 400 closed earlier, entry 10 → base 10,000
    assert p == -350.0 and pc == -3.5
    assert fold_partial(-100.0, 600.0, 10.0, 0, 0) == (-100.0, None)         # nothing partial → unchanged


def test_liquidation_priced_and_badged_on_adoption(monkeypatch):
    f = [_fill("BUY", 400, -400.0, 2.0, 77)]
    out, o, log, blocks = _run(monkeypatch, _manual_row(opened_at=_OPEN), [_pos(600.0, entry=10.0), _pos(600.0, entry=10.0)], fills=f, forced=[77])
    assert out == [[], [1]] and o.quantity == 600.0
    assert o.partial_closed_qty == 400.0 and o.liquidated_qty == 400.0 and o.partial_realized_pnl_usd == -400.0 and o.partial_fee_usd == 2.0
    assert (o.partial_unpriced_qty or 0) == 0 and o.partial_priced_until_ms is not None and "LIQUIDATION" in (o.partial_close_note or "")
    assert "POSITION_LIQUIDATED" in blocks and "POSITION_PARTIAL_CLOSED" not in blocks


def test_operator_partial_close_priced(monkeypatch):
    f = [_fill("BUY", 400, 300.0, 1.5, 55)]
    out, o, log, blocks = _run(monkeypatch, _manual_row(opened_at=_OPEN), [_pos(600.0, entry=10.0), _pos(600.0, entry=10.0)], fills=f, forced=[])
    assert o.partial_closed_qty == 400.0 and (o.liquidated_qty or 0) == 0 and o.partial_realized_pnl_usd == 300.0
    assert "POSITION_PARTIAL_CLOSED" in blocks and "POSITION_LIQUIDATED" not in blocks


def test_unpriced_then_retried_never_double_counted(monkeypatch):
    out, o, log, blocks = _run(monkeypatch, _manual_row(opened_at=_OPEN), [_pos(600.0, entry=10.0), _pos(600.0, entry=10.0)], fills_error=True)
    assert o.quantity == 600.0 and o.partial_unpriced_qty == 400.0 and o.partial_closed_qty in (None, 0) and "POSITION_PARTIAL_UNPRICED" in blocks
    # the retry prices it once; further passes (> 60 s apart) find nothing new after the marker → no second booking
    f = [_fill("BUY", 400, -400.0, 2.0, 77)]
    out, o, log, blocks = _run(monkeypatch, _manual_row(opened_at=_OPEN, quantity=600.0, partial_unpriced_qty=400.0, size_adopted_at=__import__("datetime").datetime.utcnow(),
                                                        size_adopted_from_qty=1000.0),
                               [_pos(600.0, entry=10.0)] * 3, gap_s=70.0, fills=f, forced=[77])
    assert o.partial_closed_qty == 400.0 and o.partial_realized_pnl_usd == -400.0 and (o.partial_unpriced_qty or 0) == 0


def test_final_close_includes_the_partial_and_wiring():
    import services.trading_engine as TE_
    from types import SimpleNamespace as NS
    row = NS(quantity=600.0, entry_price=10.0, partial_realized_pnl_usd=-400.0, partial_fee_usd=2.0, partial_closed_qty=400.0)
    p, pc = TE_.TradingEngine._fold_partial_close(row, -100.0, -1.67)
    assert p == -502.0 and pc == -5.02                                         # −100 − 400 − 2 on 10,000
    assert TE_.TradingEngine._fold_partial_close(NS(quantity=600.0, entry_price=10.0), -100.0, -1.67) == (-100.0, -1.67)
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert eng.count("self._fold_partial_close(") >= 2   # the main close + the fallback external close
    assert "trading_engine._fold_partial_close(order" in open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()   # reconciled close
    import models
    for c in ("partial_realized_pnl_usd", "partial_fee_usd", "partial_closed_qty", "liquidated_qty", "partial_unpriced_qty",
              "partial_priced_until_ms", "partial_close_note"):
        assert c in models.Order.__table__.columns and c in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert "function partialCloseBadgeHtml(o)" in html and "💥 liquidated" in html



def test_row_closed_before_pricing_is_folded_exactly_once(monkeypatch):
    """(574 review I3 / B1) a HUMA #25-like row: reduced by a liquidation, adopted (572-era: never priced), then CLOSED by hand before the
    deploy — the bot's close leaves closing_in_progress set. With the account FLAT (no Binance position, review I2) the closed-row sweep
    prices (opened_at, size_adopted_at] and folds it into the booked pnl ONCE."""
    now = _dt.datetime.utcnow()
    row = _manual_row(opened_at=_OPEN, quantity=600.0, status="CLOSED", closing_in_progress=True, closed_at=now, pnl=-100.0,
                      pnl_percentage=-1.6667, total_fee=10.0, size_adopted_at=now, size_adopted_from_qty=1000.0,
                      protective_tp_order_id=None, resting_tp_price=None)
    f = [_fill("BUY", 400, -400.0, 2.0, 77)]
    out, o, log, blocks = _run(monkeypatch, row, [[], [], []], gap_s=70.0, fills=f, forced=[77])
    assert o.status == "CLOSED" and o.partial_closed_qty == 400.0 and o.liquidated_qty == 400.0
    assert o.pnl == -502.0 and o.pnl_percentage == -5.02 and abs(o.total_fee - 12.0) < 1e-9   # −100 − 400 − 2 on the original 10,000
    assert "POSITION_LIQUIDATED" in blocks and blocks.count("POSITION_LIQUIDATED") == 1        # three sweeps, folded exactly once


def test_resting_tp_fills_excluded_not_skipped(monkeypatch):
    f = [_fill("BUY", 100, 50.0, 0.1, "T1"), _fill("BUY", 300, -300.0, 1.0, 88)]   # T1 = the bot's own resting TP — excluded
    row = _manual_row(opened_at=_OPEN, quantity=600.0, resting_tp_filled_qty=100.0, protective_tp_order_id="T1",
                      size_adopted_at=_dt.datetime.utcnow(), size_adopted_from_qty=1000.0, partial_unpriced_qty=300.0)
    out, o, log, blocks = _run(monkeypatch, row, [_pos(600.0, entry=10.0)], fills=f, forced=[])
    assert o.partial_closed_qty == 300.0 and o.partial_realized_pnl_usd == -300.0 and "POSITION_PARTIAL_CLOSED" in blocks


def test_force_window_over_7_days_is_unclassified_but_priced(monkeypatch):
    old_open = _dt.datetime.utcnow() - _dt.timedelta(days=9)
    f = [_fill("BUY", 400, -400.0, 2.0, 77)]
    out, o, log, blocks = _run(monkeypatch, _manual_row(opened_at=old_open), [_pos(600.0, entry=10.0), _pos(600.0, entry=10.0)], fills=f, forced=[77])
    assert o.partial_closed_qty == 400.0 and o.partial_realized_pnl_usd == -400.0           # P&L still booked
    assert "unclassified" in (o.partial_close_note or "")                                   # forceOrders covers ≤ 7 days → UNCLASSIFIED
