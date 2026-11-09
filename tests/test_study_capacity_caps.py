"""💧 capacity study (DECISION_LOG 505) — the pure impact functions."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import study_capacity_caps as S  # noqa: E402

BOOK = [(0.0, 1000.0), (0.25, 5000.0), (0.5, 10000.0), (1.0, 20000.0), (2.0, 40000.0)]


def test_walk_impact_inside_the_first_band_is_half_spread():
    assert abs(S.walk_impact(500, 0.02, BOOK) - 0.01) < 1e-12        # within the top level: ½ spread, nothing walked


def test_walk_impact_grows_with_size_and_extrapolates():
    a, b, c = S.walk_impact(5000, 0.0, BOOK), S.walk_impact(40000, 0.0, BOOK), S.walk_impact(60000, 0.0, BOOK)
    assert 0 < a < b < c and c == c


def test_walk_impact_edge_cases():
    assert S.walk_impact(0, 0.02, BOOK) == 0.0
    assert S.walk_impact(1000, 0.02, [(0.0, 1000.0)]) != S.walk_impact(1000, 0.02, [(0.0, 1000.0)])        # < 2 points → NaN
    flat = BOOK[:-1] + [(2.0, 20000.0)]
    x = S.walk_impact(60000, 0.0, flat)
    assert x != x                                                       # a flat last band never explodes — NaN → the √ law takes it


def test_sqrt_impact_monotonic():
    v = S.sqrt_impact(np.array([1e3, 1e4, 1e5]), 5e7, 26.0)
    assert np.all(np.diff(v) > 0)
