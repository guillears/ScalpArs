"""🔤 DECISION_LOG 589 — EXHAUSTION's spot market check maps a perp symbol with a size-multiplier prefix (1000PEPEUSDT, 1000000BOBUSDT …)
to its spot coin (PEPEUSDT) — the exact symbol first, the prefix-stripped coin only when spot lists it. The spot share uses QUOTE volumes
(unit-free); a mapped coin must price alike (spot close × multiplier ≈ perp close). No network."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from services import exhaustion as X  # noqa: E402
from tests import test_exhaustion_short as T  # noqa: E402

SPOT = {"PEPEUSDT", "BONKUSDT", "SHIBUSDT", "FLOKIUSDT", "LUNCUSDT", "1000SATSUSDT", "1MBABYDOGEUSDT", "1INCHUSDT", "BOBUSDT", "RATSUSDT"}


@pytest.mark.parametrize("perp,want", [
    ("1000PEPEUSDT", ("PEPEUSDT", 1000)), ("1000BONKUSDT", ("BONKUSDT", 1000)), ("1000SHIBUSDT", ("SHIBUSDT", 1000)),
    ("1000FLOKIUSDT", ("FLOKIUSDT", 1000)), ("1000LUNCUSDT", ("LUNCUSDT", 1000)), ("1000RATSUSDT", ("RATSUSDT", 1000)),
    ("1000000BOBUSDT", ("BOBUSDT", 1_000_000)),
    ("1000SATSUSDT", ("1000SATSUSDT", 1)), ("1MBABYDOGEUSDT", ("1MBABYDOGEUSDT", 1)),     # spot lists these WITH the prefix → exact first
    ("1INCHUSDT", ("1INCHUSDT", 1)), ("PEPEUSDT", ("PEPEUSDT", 1)), ("1000PEPE/USDT:USDT", ("PEPEUSDT", 1000)),
    ("FOOUSDT", (None, 1)), ("1000FOOUSDT", (None, 1)), ("10000LADYSUSDT", (None, 1)),  # no spot coin → stays no-spot
    ("100PEPEUSDT", (None, 1)), ("1000USDT", (None, 1)),                                   # not a multiplier prefix / nothing left
])
def test_mapping(perp, want):
    assert X.spot_symbol_for(perp, SPOT) == want


def test_mapping_never_raises_and_price_check():
    assert X.spot_symbol_for(None, SPOT) == (None, 1) and X.spot_symbol_for("1000PEPEUSDT", None) == (None, 1)
    perp, spot = [[0, 0, 0, 0, 0.0039045]], [[0, 0, 0, 0, 0.00000391]]          # 1000PEPEUSDT vs PEPEUSDT, live 11-18 (parity run)
    assert X.spot_price_consistent(perp, spot, 1000) is True and X.spot_price_consistent(perp, spot, 1) is False
    assert X.spot_price_consistent(perp, [[0, 0, 0, 0, 0]], 1000) is None and X.spot_price_consistent([], spot, 1000) is None


def test_spot_share_is_unit_free():
    """the rule reads QUOTE volume (kline field 7) only — a 1000× base unit changes nothing."""
    T0 = 100 * X.HOUR_MS
    rows = [[T0 - 25 * X.HOUR_MS + i * X.M15_MS, 0, 0, 0, 0, 1e9, 0, 1000.0 + i] for i in range(100)]
    rows_scaled = [r[:5] + [r[5] / 1000] + r[6:] for r in rows]                            # base volume ÷ 1000: ignored
    assert X.spot_share_chg(rows, T0, 5e4, 2e6) == X.spot_share_chg(rows_scaled, T0, 5e4, 2e6)


def _prefixed(monkeypatch):
    orig = T._row

    def row(k):
        r = list(orig(k))
        r[0] = "1000" + r[0]
        return tuple(r)
    monkeypatch.setattr(T, "_row", row)
    return orig(1)[0]


def test_engine_1000_prefix_passes_the_spot_check_and_reads_the_spot_coin(monkeypatch):
    base = _prefixed(monkeypatch)
    import services.trading_engine as TE
    monkeypatch.setattr(TE._exh_rules, "spot_price_consistent", lambda *a, **k: True)   # the fixture's spot rows carry no close (qv only)
    e, opens, sig, _, calls = T._run(monkeypatch, k=1, spot_syms={base})
    spt = [c for c in calls if len(c) == 5 and c[4]]
    assert spt and spt[0][0] == base                                                          # the spot klines are read on the spot coin
    assert len(opens) == 1 and opens[0]['entry_exh_spot_symbol'] == base and sig[0].status == "OPENED"
    assert f"spot {base} ×1,000" in sig[0].reason and "EXH_SPOT_MAPPED" not in e.blocks and e._exh_spot_mapped == {"1000" + base: 1}
    assert "EXH_NO_SPOT" not in e.blocks and "EXH_SPOT_MISMATCH" not in e.blocks


def test_engine_no_spot_stays_no_spot(monkeypatch):
    _prefixed(monkeypatch)
    e, opens, sig, _, calls = T._run(monkeypatch, k=1, spot_syms={"BTCUSDT"})
    assert opens == [] and sig[0].block == "EXH_NO_SPOT" and not [c for c in calls if len(c) == 5 and c[4]]
    assert "EXH_SPOT_MAPPED" not in e.blocks


def test_engine_mapped_price_mismatch_fails_closed(monkeypatch):
    base = _prefixed(monkeypatch)
    import services.trading_engine as TE
    monkeypatch.setattr(TE._exh_rules, "spot_price_consistent", lambda *a, **k: False)
    e, opens, sig, _, calls = T._run(monkeypatch, k=1, spot_syms={base})
    assert opens == [] and sig[0].block == "EXH_SPOT_MISMATCH" and X.B_SPOT_MISMATCH_TXT in sig[0].reason and "×1,000" in sig[0].reason
    assert "EXH_SPOT_MISMATCH" in e.blocks


def test_engine_mapped_unreadable_close_blocks(monkeypatch):
    """the fixture's spot klines carry close 0 → spot_price_consistent None → a MAPPED coin is blocked (fail-closed); exact symbols never check."""
    base = _prefixed(monkeypatch)
    e, opens, sig, _, calls = T._run(monkeypatch, k=1, spot_syms={base})
    assert opens == [] and sig[0].block == "EXH_SPOT_MISMATCH"


def test_sleeve_status_label():
    from services import sleeve_status as SS
    assert SS.EXH_REASON["EXH_SPOT_MISMATCH"] == "spot mismatch"


def test_exact_symbol_unchanged(monkeypatch):
    e, opens, sig, _, calls = T._run(monkeypatch, k=1)
    pair = T._row(1)[0]
    assert opens[0]['entry_exh_spot_symbol'] == pair and "EXH_SPOT_MAPPED" not in e.blocks


def test_stamp_column_and_class():
    import models
    import services.trading_engine as TE
    assert "entry_exh_spot_symbol" in [c.name for c in models.Order.__table__.columns]
    assert TE.TradingEngine.stamp_class("entry_exh_spot_symbol") == "sleeve"
    assert "entry_exh_spot_symbol VARCHAR(30)" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()


def test_open_path_stores_the_spot_symbol(monkeypatch):
    import datetime as dt
    import time
    r, *_ = T._e2e(monkeypatch, n_open=0, exhaustion_short=True, exhaustion_bar_close_ms=int(time.time() * 1000),
                   entry_frenzy_spike_at=dt.datetime(2026, 11, 17, 8, 0), entry_exh_tp_pct=3.6, entry_exh_sl_pct=1.8,
                   entry_exh_decision_price=1.0, entry_exh_q5_usd=2e6, entry_exh_spot_symbol="PEPEUSDT")
    assert r is not None and r.entry_exh_spot_symbol == "PEPEUSDT"
