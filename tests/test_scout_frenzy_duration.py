"""⏱ FRENZY max-duration read (DECISION_LOG 496): the pure math the B25 per-sleeve cap recommendation rests on."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_frenzy_duration as D  # noqa: E402

MIN = 60_000
T0 = 1_800_000_000_000 - (1_800_000_000_000 % MIN)   # a minute boundary


def test_caps_read_live_config_and_engine_fallbacks(tmp_path):
    p = tmp_path / "c.json"
    def cfg(fz, wl, glob=1200):
        p.write_text(json.dumps({"investment": {"max_holding_time_minutes": glob},
                                 "thresholds": {"frenzy_max_hold_minutes": fz, "frenzy_willy_max_hold_minutes": wl}}))
        return D.caps(str(p))
    c = cfg(300, 45)
    assert c["FRENZY_LITE"] == c["FRENZY_LITE_RETRY"] == c["FRENZY_LONG"] == 300
    assert c["FRENZY_WILLY"] == c["FRENZY_WILLY_HT"] == 45
    # engine rules (services/trading_engine.py max-hold): FRENZY = min(global, own), 0 → the other; WILLY round(), < 1 → 120, never the global
    assert cfg(0, 0)["FRENZY_LITE"] == 1200 and cfg(0, 0)["FRENZY_WILLY"] == 120
    assert cfg(720, 90, glob=600)["FRENZY_LONG"] == 600 and cfg(720, 90, glob=600)["FRENZY_WILLY"] == 90
    assert cfg(720, 90, glob=0)["FRENZY_LONG"] == 720
    assert cfg(720, 89.6)["FRENZY_WILLY"] == 90
    live = D.caps()
    assert set(live) == set(D.SLEEVES) and all(v > 0 for v in live.values())


def test_net_pct_matches_the_engine_fee_formula_both_directions():
    # bot net = gross − entry fee − taker × (exit / entry)  (+ the slip ruler)
    assert D.net_pct(100, 103, "LONG", D.TAKER, slip=0) == pytest.approx(3.0 - 0.045 - 0.045 * 1.03)
    assert D.net_pct(100, 97, "SHORT", D.MAKER, slip=0) == pytest.approx(3.0 - 0.018 - 0.045 * 0.97)
    assert D.net_pct(100, 100, "LONG", D.TAKER) == pytest.approx(-0.045 - 0.045 - D.SLIP)
    assert D.entry_fee("MAKER") == D.MAKER and D.entry_fee("MAKER+TAKER") == D.TAKER and D.entry_fee(None) == D.TAKER


def test_price_at_is_the_last_bar_closed_by_t_never_later():
    bars = pd.DataFrame({"open_time": [T0, T0 + MIN, T0 + 2 * MIN], "c": [1.0, 2.0, 3.0]})
    assert D.price_at(bars, T0 + MIN + D.LAT_MS) == 1.0        # the bar [T0, T0+1m) closed by then; [T0+1m, T0+2m) had not
    assert D.price_at(bars, T0 + 2 * MIN) == 2.0               # a bar closing exactly at t counts
    assert D.price_at(bars, T0 + 6 * MIN) is None
    assert D.price_at(None, T0) is None


def test_cap_at_only_changes_fills_still_open_when_the_cap_fires():
    assert D.cap_at(2.5, 10, 15, 999.0, 100, "LONG", D.TAKER) == 2.5          # closed by X → booked, whatever the price
    # (deep review I2) closed 3 s after X — before the cap's own X + 8 s → its real TP / stop came first: booked, never re-priced
    assert D.cap_at(0.25, 15 + 3 / 60, 15, 999.0, 100, "LONG", D.TAKER) == 0.25
    assert D.open_at_cap(15 + 9 / 60, 15) and not D.open_at_cap(15 + 8 / 60, 15)
    assert D.cap_at(-3.0, 40, 15, 101.0, 100, "LONG", D.TAKER) == pytest.approx(D.net_pct(100, 101, "LONG", D.TAKER))
    assert D.cap_at(-3.0, 40, 15, None, 100, "LONG", D.TAKER) is None          # unpriced, never guessed
    assert D.cap_at(-3.0, 40, 15, float("nan"), 100, "LONG", D.TAKER) is None   # a NaN-blanked master row is unpriced too
    assert D.cap_at(1.0, None, 15, 101.0, 100, "LONG", D.TAKER) is None


def test_boot_p_resamples_days_and_is_deterministic():
    assert D.boot_p([1.0, 2.0, 0.5], ["a", "b", "c"]) == 1.0
    assert D.boot_p([-1.0, -2.0], ["a", "b"]) == 0.0
    assert np.isnan(D.boot_p([1.0, 2.0], ["a", "a"]))                          # one day = one observation → no read
    assert D.boot_p([1, -3, 2], ["a", "b", "c"]) == D.boot_p([1, -3, 2], ["a", "b", "c"])


def _frame():
    op = [pd.Timestamp(T0, unit="ms") + pd.Timedelta(hours=h) for h in (0, 1, 30)]
    hold = [10, 50, 300]
    return pd.DataFrame({
        "pair": ["AUSDT", "BUSDT", "CUSDT"], "direction": ["LONG"] * 3, "id": [1, 2, 3],
        "opened_at": [t.isoformat() for t in op],
        "closed_at": [(t + pd.Timedelta(minutes=m)).isoformat() for t, m in zip(op, hold)],
        "entry_price": [100.0] * 3, "exit_price": [103.0, 97.0, 97.0], "entry_order_type": ["MARKET"] * 3,
        # the two stops are booked by the engine's own fee formula (parity must read 0)
        "pnl_percentage": [2.9, D.net_pct(100, 97, "LONG", D.TAKER, slip=0), D.net_pct(100, 97, "LONG", D.TAKER, slip=0)],
        "close_reason": ["FRENZY_TP", "STOP_LOSS", "STOP_LOSS"],
        "notional_value": [1000.0] * 3, "pnl": [29.0, -30.0, -30.0], "status": ["CLOSED"] * 3})


def test_profile_counts_cap_and_open_at():
    f = D._norm(_frame(), "pnl_percentage")
    p = D.profile(f, 720)
    assert p["n"] == 3 and p["open_at"][15] == 2 and p["open_at"][60] == 1 and p["open_at"][240] == 1 and p["open_at"][360] == 0
    assert p["w_med"] == 10 and p["l_med"] == 175 and p["n_cap"] == 0
    assert D.hold_minutes("2026-11-07 08:20:00", "2026-11-07T13:20:00+00:00") == 300


def test_sleeve_block_end_to_end_with_account_dcr():
    f = D._norm(_frame(), "pnl_percentage")
    for x in D.XS:
        f[f"px_{x}"] = [None if not D.open_at_cap(h, x) else 101.0 for h in f.hold]
    acc = _frame()
    acc["k"] = D.fill_key(acc)
    acc["usd"] = acc.pnl
    keys = acc.k.tolist()
    L = D.sleeve_block(f, 720, "FRENZY_LITE", (acc, pd.Timestamp(T0, unit="ms"), pd.Timestamp(T0, unit="ms") + pd.Timedelta(days=2), keys))
    txt = "\n".join(L)
    assert "| 15 min | 2 |" in txt and "| 4.0 h | 1 |" in txt and "| 6.0 h | 0 |" in txt
    assert "→" in txt and "nan" not in txt.lower()
    assert "CUSDT #3" in txt and "market close at 4 h" in txt
    assert "parity — fee ruler on 2 market-closed fill(s), slip 0: median |Δ| 0.000" in txt and "RULER OFF" not in txt


def test_parity_flags_a_wrong_fee_or_price_ruler():
    f = D._norm(_frame(), "pnl_percentage")
    assert D.parity_line(f)[1] is True
    f.loc[2, "pct"] = -5.0
    f.loc[1, "pct"] = -5.0
    t, ok = D.parity_line(f)
    assert not ok and "RULER OFF" in t
    g = D._norm(_frame(), "pnl_percentage")
    g["px_close"] = [None, None, 97.0]
    assert D.parity_line(g)[1] and "price ruler on 1 cap close" in D.parity_line(g)[0]
    g["px_close"] = [None, None, 90.0]
    assert not D.parity_line(g)[1]


def test_entry_fee_prefers_the_fills_own_fee():
    assert D.entry_fee("MARKET", 0.45, 1000) == pytest.approx(0.045)
    assert D.entry_fee("MARKET", None, 1000) == D.TAKER and D.entry_fee("MAKER", float("nan"), 1000) == D.MAKER


def test_haircut_only_shrinks_a_favourable_delta():
    assert D.haircut_text(-0.2) == "n/a (Δ ≤ 0)" and D.haircut_text(0.2) == "+0.100 … +0.140"


def test_dcr_join_miss_raises_instead_of_printing_after_equals_before():
    f = D._norm(_frame(), "pnl_percentage")
    for x in D.XS:
        f[f"px_{x}"] = [None if not D.open_at_cap(h, x) else 101.0 for h in f.hold]
    acc = _frame()
    acc["k"] = D.fill_key(acc)
    acc["usd"] = acc.pnl
    with pytest.raises(ValueError, match="DCR join matched"):
        D.sleeve_block(f, 720, "X", (acc.iloc[:2], pd.Timestamp(T0, unit="ms"), pd.Timestamp(T0, unit="ms") + pd.Timedelta(days=2), D.fill_key(f)))


def test_short_end_to_end_and_master_profile_only():
    f = _frame().assign(direction="SHORT", exit_price=[97.0, 103.0, 103.0])
    f["pnl_percentage"] = [2.9] + [D.net_pct(100, 103, "SHORT", D.TAKER, slip=0)] * 2
    f = D._norm(f, "pnl_percentage")
    for x in D.XS:
        f[f"px_{x}"] = [None if not D.open_at_cap(h, x) else 99.0 for h in f.hold]   # price fell: a short gains
    txt = "\n".join(D.sleeve_block(f, 720, "S"))
    assert "RULER OFF" not in txt and "| 15 min | 2 |" in txt
    row = [l for l in txt.splitlines() if l.startswith("| 15 min")][0]
    assert float(row.split("|")[6]) > 0                         # Δ Σ positive for the short
    m = "\n".join(D.sleeve_block(f, 720, "M", priced=False))
    assert "② cap at" not in m and "not priced (master)" in m


def test_registered_in_the_observe_registry():
    import observe_registry as OR
    assert "scout_frenzy_duration.py" in OR.classified()
