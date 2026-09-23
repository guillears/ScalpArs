"""🏦 Sep-23 MEGA-CAP EXCLUSION for momentum longs — the pure rule the engine gate calls.

Evidence (current stack, 116 momentum longs): eligible-universe rank ≤10 = 10·50%·−0.17%·−$131 over 9 windows
(HYPE 5, SOL 3, ADA 1, XRP 1; ex-ADA 9·44%·−$373) vs rank 11–15 = 15·93%·+$1,920. Declared operator override
at N=10 (< the N≥30 bar). Thesis: a 5m EMA-stack signal on a BTC-beta mega-cap is BTC noise; the unmatched
thesis is idiosyncratic flow. The bull-run sleeve KEEPS the top 10 — that is its universe.

Invariants pinned here:
  · 0 / missing threshold = off.
  · rank is the RAW eligible-universe rank (`entry_pair_rank`, stamped before blacklist removal) — the number the
    evidence was computed on — never br_rank.
  · boundary inclusive: rank == N blocks, N+1 passes.
  · a missing/garbage rank FAILS OPEN (a block never fires on data it does not have).
"""
from types import SimpleNamespace

from services.trading_engine import long_megacap_block


def _th(n=10):
    return SimpleNamespace(long_megacap_rank_max=n)


def test_blocks_top_ten_inclusive():
    assert long_megacap_block(_th(), 1) is True
    assert long_megacap_block(_th(), 5) is True          # XRP B12
    assert long_megacap_block(_th(), 10) is True         # ADA B2 sat exactly at 10
    assert long_megacap_block(_th(), 11) is False        # first pair of the 11-15 sweet spot
    assert long_megacap_block(_th(), 28) is False        # WIF


def test_zero_or_missing_threshold_disables():
    assert long_megacap_block(_th(0), 1) is False
    assert long_megacap_block(_th(None), 1) is False
    assert long_megacap_block(SimpleNamespace(), 1) is False


def test_missing_or_garbage_rank_fails_open():
    for bad in (None, 0, -1, "", "nonsense", float("nan")):      # NaN = an unstamped pool row
        assert long_megacap_block(_th(), bad) is False


def test_threshold_is_read_as_int():
    assert long_megacap_block(SimpleNamespace(long_megacap_rank_max="10"), 10) is True
    assert long_megacap_block(SimpleNamespace(long_megacap_rank_max=10.0), 10) is True
    assert long_megacap_block(SimpleNamespace(long_megacap_rank_max="nonsense"), 10) is False


# ───────────────────────────────────────────── guard parity (deep review, 2026-09-23)
# The engine gate's exemption list is copied by hand from the LONG_HEAT_BLOCK guard. No test calls
# open_position for either gate, so a drifted copy (e.g. accidentally exempting CALM3D doors, or the
# bull-run sleeve losing its exemption) would pass every pure-function test. Pin the two guards to each
# other at the source level — the cheapest protection that would actually fail.
import inspect
import re


def _guard(src, anchor):
    i = src.index(anchor)
    start = src.rindex("if (", 0, i)
    end = src.index("):", i) + 2
    return src[start:end]


def _norm(g):
    g = g.replace("_lh_block and ", "")
    g = re.sub(r"\s*and long_megacap_block\([^)]*\)", "", g)
    return re.sub(r"\s+", " ", g).strip()


def test_megacap_guard_matches_heat_block_guard_token_for_token():
    from services.trading_engine import TradingEngine
    src = inspect.getsource(TradingEngine.open_position)
    heat = _guard(src, '_lh_block and direction == "LONG"')
    mega = _guard(src, "and long_megacap_block(")
    assert _norm(heat) == _norm(mega)
    for g in (heat, mega):                       # the momentum-long DOORS must stay gated
        assert "nonexp_calm3d" not in g
        assert "cross_ob_open" not in g
        assert "not bullrun_long" in g           # the bull-run sleeve keeps the top 10
