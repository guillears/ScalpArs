"""🧭 Oct-5: open positions per sleeve under the "USDT in Open Orders" card (main._open_by_sleeve)."""
import os
from types import SimpleNamespace as S

os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")


def test_open_by_sleeve_labels_and_order():
    import main
    rows = main._open_by_sleeve([S(entry_strategy="SPIKE_FADE", direction="SHORT"), S(entry_strategy=None, direction="LONG"),
                                 S(entry_strategy="FLIP:FAN_RATIO_GATE", direction="SHORT"), S(entry_strategy="SPIKE_FADE", direction="SHORT"),
                                 S(entry_strategy="MANUAL", direction="LONG"), S(entry_strategy="FRENZY_WIDE", direction="LONG")])
    assert rows[0] == ["Spike Fade Short", 2]
    assert ["Momentum Long", 1] in rows and ["Flip Short", 1] in rows and ["Manual Long", 1] in rows and ["FRENZY WIDE Long", 1] in rows
    assert main._open_by_sleeve([]) == [] and main._open_by_sleeve(None) == []


def test_card_wired():
    html = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates", "index.html"), encoding="utf-8").read()
    assert html.count('id="usdt-orders-sleeves"') == 1 and "data.open_by_sleeve" in html
