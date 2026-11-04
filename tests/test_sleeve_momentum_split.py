"""📈📉 DECISION_LOG 456: the Sleeve select splits momentum into Momentum Long / Momentum Short (flip_source MOMENTUM_LONG / _SHORT = the
pure-momentum book on one side); the "Filter by Momentum cell" dropdown is gone from the dashboard (the API keeps cell_source)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cell_filter_origin_switch import _html, _perf, ROWS  # noqa: E402

SIDES = [("MOMENTUM", "UNMATCHED", "LONG"), ("MOMENTUM", "W1", "SHORT"), ("MOMENTUM", None, "SHORT"), ("FRENZY_WILLY", "FRENZY_WILLY", "LONG"),
         ("FLIP:FAN_RATIO_GATE", "FLIPCELL", "SHORT"), ("SPIKE_FADE", "SPIKE_FADE", "SHORT")]


def test_momentum_long_short_split():
    import asyncio  # noqa: F401
    from test_cell_filter_origin_switch import _order  # noqa: F401
    rows = [(es, cell) for es, cell, _ in SIDES]
    dirs = [d for _, _, d in SIDES]
    import test_cell_filter_origin_switch as T
    orig = T._order

    def _o(models, i, strat, cell, **kw):
        return orig(models, i, strat, cell, direction=dirs[i], **kw)
    T._order = _o
    try:
        allm = _perf(rows, flip_source="MOMENTUM")["total_trades"]
        lng = _perf(rows, flip_source="MOMENTUM_LONG")["total_trades"]
        sht = _perf(rows, flip_source="MOMENTUM_SHORT")["total_trades"]
    finally:
        T._order = orig
    assert (allm, lng, sht) == (3, 1, 2)                     # flips, FRENZY and fades never count as momentum


def test_dashboard_options():
    h = _html()
    assert '<option value="MOMENTUM_LONG">📈 Momentum Long</option>' in h and '<option value="MOMENTUM_SHORT">📉 Momentum Short</option>' in h
    assert 'id="cell-source-filter"' not in h and "Filter by Momentum cell" not in h
    assert '<option value="MOMENTUM">Non-flip (momentum)</option>' not in h
