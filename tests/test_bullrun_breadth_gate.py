"""🌊 Sep-19 bull-run breadth minimum (57c, operator-found) — the pure rule the sleeve entry calls.
Invariants: blank/None/0 threshold = gate OFF (always admits); warm-up state (bull AND bear both
0/None = breadth not computed since boot) fails OPEN — the sleeve must never freeze on a deploy;
a real market with bull below the min blocks; at/above the min admits (inclusive); a market that
is genuinely 0% bull but has bear readings (bull=0, bear>0) is a REAL reading and blocks; the
function never raises (garbage input admits).
"""
from types import SimpleNamespace

from services.trading_engine import bullrun_breadth_ok


def _th(v=60.0, on=True):
    return SimpleNamespace(bullrun_breadth_enabled=on, bullrun_breadth_min=v)


def test_blocks_below_min():
    # tonight's losing episode: fills at breadth 38-58 while BTC held the monitor GREEN
    assert bullrun_breadth_ok(_th(), 57.8, 26.7) is False
    assert bullrun_breadth_ok(_th(), 40.0, 44.4) is False


def test_admits_at_and_above_min_inclusive():
    assert bullrun_breadth_ok(_th(), 60.0, 20.0) is True
    assert bullrun_breadth_ok(_th(), 77.8, 17.8) is True


def test_toggle_off_admits_everything():
    # ships OFF (operator Sep-19): review at the weekend with the journal breadth series
    assert bullrun_breadth_ok(_th(on=False), 10.0, 80.0) is True
    assert bullrun_breadth_ok(SimpleNamespace(bullrun_breadth_min=60.0), 10.0, 80.0) is True  # flag absent = off


def test_off_when_blank_zero_or_none():
    assert bullrun_breadth_ok(_th(None), 10.0, 80.0) is True
    assert bullrun_breadth_ok(_th(''), 10.0, 80.0) is True
    assert bullrun_breadth_ok(_th(0), 10.0, 80.0) is True
    assert bullrun_breadth_ok(SimpleNamespace(), 10.0, 80.0) is True  # field absent entirely


def test_warmup_fails_open_for_green():
    # B4 restart artifact: fills stamped bull 0.0 / bear 0.0 minutes after a deploy
    assert bullrun_breadth_ok(_th(), 0.0, 0.0) is True
    assert bullrun_breadth_ok(_th(), None, None) is True
    assert bullrun_breadth_ok(_th(), 0.0, None) is True
    assert bullrun_breadth_ok(_th(), 0.0, 0.0, door='GREEN') is True


def test_warmup_fails_closed_for_rearm():
    # Sep-19 fix: 5 of B4's 7 REARM fills entered blind post-restart, 4 lost — the aggressive
    # door waits one scan cycle instead of trading an unmeasured market
    assert bullrun_breadth_ok(_th(), 0.0, 0.0, door='REARM') is False
    assert bullrun_breadth_ok(_th(), None, None, door='REARM') is False
    # with a REAL reading, REARM is judged on the floor like anyone else
    assert bullrun_breadth_ok(_th(v=40.0), 45.0, 30.0, door='REARM') is True
    assert bullrun_breadth_ok(_th(v=40.0), 27.9, 62.8, door='REARM') is False  # the B4 bear-majority fill
    assert bullrun_breadth_ok(_th(v=40.0), 38.0, 30.0, door='GREEN') is False  # sub-40 zone: no winning history
    # toggle off / garbage still admit regardless of door
    assert bullrun_breadth_ok(_th(on=False), 0.0, 0.0, door='REARM') is True
    assert bullrun_breadth_ok(_th('x'), 0.0, 0.0, door='REARM') is True


def test_genuine_zero_bull_with_bear_reading_blocks():
    # bear readings prove the scan ran — bull 0 is then a real (terrible) market, not warm-up
    assert bullrun_breadth_ok(_th(), 0.0, 46.7) is False


def test_never_raises_on_garbage():
    assert bullrun_breadth_ok(_th('sixty'), 50.0, 30.0) is True
    assert bullrun_breadth_ok(_th(), 'x', 'y') is True
