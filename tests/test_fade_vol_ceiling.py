"""Sep-14 fade 24h-volume ceiling (operator discipline-override, DECISION_LOG 55). Pins the pure gate:
≥ ceiling blocks, below passes, 0/None ceiling = off, missing/zero volume fails OPEN (parity with the
other fade gates). Falsifiable: flipping the fail-open, the boundary, or the off-sentinel fails."""
from services.trading_engine import _fade_vol_blocked as blocked


def test_boundary_and_direction():
    assert blocked(20_000_000, 20_000_000) is True       # at the line → blocked
    assert blocked(311_401_660, 20_000_000) is True      # VTHO Sep-14
    assert blocked(189_870_969, 20_000_000) is True      # 龙虾 Sep-14
    assert blocked(19_999_999, 20_000_000) is False
    assert blocked(2_646_236, 20_000_000) is False       # SYRUP (scanner micro-cap) passes


def test_off_sentinel_and_fail_open():
    assert blocked(500_000_000, 0) is False              # 0 = off
    assert blocked(500_000_000, None) is False
    assert blocked(None, 20_000_000) is False            # missing volume → fade fires (fail-open)
    assert blocked(0, 20_000_000) is False
    assert blocked("garbage", 20_000_000) is False       # never raises
