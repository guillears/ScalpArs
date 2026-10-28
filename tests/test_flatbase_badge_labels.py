"""Operator 2026-10-23: the FLATBASE_SHORT open-row badge read "−8% / +15% / 72h" (price moves) — "TP is +8% or +15%?". Every other sleeve
badge is in P&L terms, so the short's badge / export config line / hover now say TP +8 % · SL −15 % with the price levels in brackets
(DECISION_LOG 354). Display only."""
import os

HTML = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()


def test_open_row_badge_is_in_pnl_terms():
    assert ">🩳 TP +${_fbTp}% · SL −${_fbSl}% · ${_fbMh}h</span>" in HTML
    assert ">🩳 −${_fbTp}% / +${_fbSl}%" not in HTML
    assert "take profit at −${_fbTp}%" in HTML and "stop at +${_fbSl}%" in HTML     # the hover keeps the price levels


def test_export_config_line_and_hover_name_both_terms():
    assert "· exit TP +${_bt.flatbase_tp_pct ?? 5}% (price −${_bt.flatbase_tp_pct ?? 5}%)" in HTML
    assert "SL −${_bt.flatbase_stop_pct ?? 10}% (price +${_bt.flatbase_stop_pct ?? 10}%)" in HTML
    assert "· exit: TP +${_n(_fb.flatbase_tp_pct, 5)} % (price −${_n(_fb.flatbase_tp_pct, 5)} %)" in HTML


def test_monitor_line_and_hover_sl_half():
    assert "· −${p.tp ?? 5}% / +${p.stop ?? 10}%" not in HTML
    assert "· TP +${p.tp ?? 5}% (price −${p.tp ?? 5}%) / SL −${p.stop ?? 10}% (price +${p.stop ?? 10}%) / " in HTML
    assert "SL −${_n(_fb.flatbase_stop_pct, 10)} % (price +${_n(_fb.flatbase_stop_pct, 10)} %)" in HTML

