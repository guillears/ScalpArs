"""Sep 16 — BTC 24h-range dimension buckets: every value lands in exactly one half-open bucket; boundaries; None."""
from main import _range_bucket, _off24h_bucket, BTC_OFF24H_BUCKETS, BTC_OFF24LO_BUCKETS


def _exactly_one(v, buckets):
    return sum(1 for lbl, lo, hi in buckets if lo <= v < hi)


def test_every_reading_lands_in_exactly_one_bucket():
    for v in [-12.0, -4.0, -3.99, -2.0, -1.99, -0.8, -0.79, 0.0]:
        assert _exactly_one(-v, BTC_OFF24H_BUCKETS) == 1   # stamped value is ≤0; bucketed on the distance below the high
    for v in [0.0, 0.49, 0.5, 0.99, 1.0, 1.99, 2.0, 3.99, 4.0, 9.0]:
        assert _exactly_one(v, BTC_OFF24LO_BUCKETS) == 1


def test_gate_boundaries_are_where_the_sleeves_put_them():
    assert _off24h_bucket(-2.0) == '2 to 4% below'                          # at the bull gate (≤ −2 refused): refused side
    assert _off24h_bucket(-1.99) == '0.8 to 2% below'                       # inside the gate
    assert _range_bucket(2.0, BTC_OFF24LO_BUCKETS) == '2 to 4%'            # at the bear gate: refused side
    assert _range_bucket(1.99, BTC_OFF24LO_BUCKETS) == '1 to 2%'


def test_none_is_not_bucketed():
    assert _range_bucket(None, BTC_OFF24H_BUCKETS) is None and _off24h_bucket(None) is None
