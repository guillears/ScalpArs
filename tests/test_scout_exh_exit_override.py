"""🎯 DECISION_LOG 579 — EXHAUSTION_SHORT exit override (paper, declared): the paired-pricing revert gate + the re-based DL 578 gate."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)

import scout_exh_exit_override as X          # noqa: E402
import scout_exhaustion_short as SC          # noqa: E402
import observe_registry as OR                # noqa: E402


def test_levels_cap_and_bad_atr():
    assert X.levels(3.9, X.SHIPPED) == (5.0, 5.0)
    tp, sl = X.levels(3.9, X.OVERRIDE)
    assert tp == 10.0 and abs(sl - 5.85) < 1e-9
    assert X.levels(2.0, X.OVERRIDE) == (10.0, 3.0)
    assert X.levels(0.3, X.OVERRIDE)[0] == 6.0                 # 20 × 0.3
    for bad in (None, 0, -1, float("nan"), "x"):
        assert X.levels(bad, X.SHIPPED) is None


def test_price_short_tp_through_not_touch():
    e = 100.0
    touch = [(100, 100.5, 95.0, 96)] * 3                      # low == the TP price → no fill
    net, why = X.price_short(e, touch, 5.0, 5.0, 3)
    assert why == "TIME"
    through = [(100, 100.5, 94.99, 96)]
    net, why = X.price_short(e, through, 5.0, 5.0, 3)
    assert why == "TP" and abs(net - (5.0 - X.TAKER - X.MAKER)) < 1e-9


def test_price_short_stop_first_and_gap():
    e = 100.0
    both = [(100, 106.0, 90.0, 95)]                           # stop and TP in one bar → the stop
    net, why = X.price_short(e, both, 5.0, 5.0, 10)
    assert why == "SL" and abs(net - (-5.0 - 2 * X.TAKER)) < 1e-9
    gap = [(100, 101, 99, 100), (108.0, 109.0, 107.0, 108)]  # gaps over the stop → filled at the open; slip on the taker EXIT only
    net, why = X.price_short(e, gap, 5.0, 5.0, 10, slip_pct=0.1)
    assert why == "SL" and abs(net - (-8.0 - 2 * X.TAKER - 0.1)) < 1e-9


def test_price_short_time_open_and_bad_entry():
    e = 100.0
    bars = [(100, 101, 99, 98.0)] * 5
    net, why = X.price_short(e, bars, 5.0, 5.0, 5)
    assert why == "TIME" and abs(net - (2.0 - 2 * X.TAKER)) < 1e-9
    assert X.price_short(e, bars[:3], 5.0, 5.0, 5) == (None, "OPEN")
    assert X.price_short(e, [], 5.0, 5.0, 5) == (None, "NO_DATA")
    for bad in (float("nan"), None, 0, -1, "x"):
        assert X.price_short(bad, bars, 5.0, 5.0, 5) == (None, "NO_DATA")


def test_path_bars_needs_every_minute():
    t1 = 1_000_000 * 60_000
    rows = [(t1 + i * 60_000, 1, 2, 0.5, 1) for i in range(5)]
    assert len(X.path_bars(rows, t1, 5)) == 5
    assert X.path_bars(rows[:2] + rows[3:], t1, 5) is None   # a missing minute → None (never a shifted time exit)


def test_gate_states():
    assert X.gate([])[0] == "COLLECTING"
    few = [("d%d" % (i % 9), 1.0) for i in range(45)]
    assert X.gate(few)[0] == "COLLECTING"                     # 9 days < 10 → no checkpoint reached
    keep = [("d%d" % (i % 12), 0.5 + (i % 3) - 1) for i in range(40)]
    v, txt = X.gate(keep)
    assert v == "KEEP" and "checkpoint 40" in txt and "next read at 80" in txt
    worse = [("d%d" % (i % 12), -3.0 + (i % 5) * 0.5) for i in range(40)]
    assert X.gate(worse)[0] == "REVERT"                       # clearly worse → the upper bound < 0
    flat_noise = [("d%d" % (i % 12), (4.8 if i % 2 else -4.8)) for i in range(40)]
    assert X.gate(flat_noise)[0] == "KEEP"                    # a mean ≈ 0 with big noise is NOT shown worse
    # read once per checkpoint: fills 41..79 never change the 40-fill verdict
    late = worse + [("d%d" % (i % 12), 50.0) for i in range(30)]
    assert X.gate(late)[0] == "REVERT"
    nan = keep + [("d99", float("nan"))]
    assert X.gate(nan)[0] == "KEEP"                           # a NaN Δ is dropped
    assert X.upper_bound([("a", 1.0), ("b", 1.0)]) == 1.0


def test_price_fill_gating_and_cache(monkeypatch):
    t1 = int(1_763_454_660_000)                               # a whole minute
    opened = __import__("pandas").Timestamp(t1 - 60_000, unit="ms").strftime("%Y-%m-%dT%H:%M:%S")
    rows = [(t1 + i * 60_000, 100.0, 100.2, 99.9, 100.0) for i in range(X.HOLD_MIN)]
    rows[30] = (t1 + 30 * 60_000, 100.0, 100.0, 89.0, 90.0)  # a 10 %+ drop at minute 30 → both TPs fill
    calls = []
    monkeypatch.setattr(X, "fetch_1m", lambda pair, t, n: calls.append(t) or rows)
    cache = {}
    assert X.price_fill("AUSDT", opened, 100.0, 4.0, 0.0, cache, now_ms=t1 + 3_600_000) is None and calls == []   # path not over yet
    out = X.price_fill("AUSDT", opened, 100.0, 4.0, 0.0, cache, now_ms=t1 + 26 * 3_600_000)
    assert calls == [t1] and out["override"][1] == "TP" and out["shipped"][1] == "TP"
    assert abs(out["override"][0] - (10 - X.TAKER - X.MAKER)) < 1e-9 and abs(out["shipped"][0] - (5 - X.TAKER - X.MAKER)) < 1e-9
    assert isinstance(cache["AUSDT|%d" % t1], dict)            # the priced results are cached, not the bars
    assert X.price_fill("AUSDT", opened, 100.0, 4.0, 0.0, cache, now_ms=t1 + 26 * 3_600_000) == out and calls == [t1]
    monkeypatch.setattr(X, "fetch_1m", lambda pair, t, n: rows[:100])
    assert X.price_fill("BUSDT", opened, 100.0, 4.0, 0.0, {}, now_ms=t1 + 25 * 3_600_000) is None          # short path, recent → wait
    c2 = {}
    assert X.price_fill("BUSDT", opened, 100.0, 4.0, 0.0, c2, now_ms=t1 + 40 * 3_600_000) == "NO_DATA" and c2


def test_report_prefix_halt_and_frozen_checkpoints(monkeypatch, tmp_path):
    import pandas as pd
    base = pd.Timestamp("2026-11-20")
    fills = pd.DataFrame([dict(pair=f"P{i}USDT", opened_at=str(base + pd.Timedelta(hours=7 * i)), entry_price=100.0, entry_exh_atr_pct=4.0,
                               entry_exh_tp_pct=10.0, slip=0.0, pnl_percentage=1.0, o=base + pd.Timedelta(hours=7 * i)) for i in range(45)])
    monkeypatch.setattr(X, "_load_cache", lambda: {})
    monkeypatch.setattr(X, "_save_cache", lambda c: None)
    n = [0]
    def fake(pair, opened_at, entry, atr, slip, cache, now_ms=None):
        n[0] += 1
        if pair == "P5USDT" and fake.fail:
            raise X.urllib.error.URLError("down")
        return {"override": (1.0, "TP"), "shipped": (0.5, "TP")}
    fake.fail = True
    monkeypatch.setattr(X, "price_fill", fake)
    st = tmp_path / "st.json"
    txt = X.report(fills, now_ms=0, state_path=str(st))
    assert "pricing halted at P5USDT" in txt and "COLLECTING" in txt and n[0] == 6     # stops at the first failure → a prefix
    fake.fail = False
    txt = X.report(fills, now_ms=0, state_path=str(st))
    assert "FROZEN checkpoint 40" in txt and "KEEP" in txt
    import json
    rec = json.load(open(st))["40"]
    monkeypatch.setattr(X, "price_fill", lambda *a, **k: {"override": (-9.0, "SL"), "shipped": (0.5, "TP")})
    txt = X.report(fills, now_ms=0, state_path=str(st))
    assert json.load(open(st))["40"] == rec and "FROZEN checkpoint 40" in txt              # never recomputed


def test_era_split_with_rounded_stamps():
    import random
    from services.exhaustion import exit_era_579
    rng = random.Random(579)
    for _ in range(20000):
        a = rng.uniform(0.02, 8.0)
        assert not exit_era_579(round(min(3 * a, 5), 6), round(a, 6))        # shipped stamps never read as the override
        assert exit_era_579(round(min(20 * a, 10), 6), round(a, 6))          # override stamps always do


def test_era_split():
    from services.exhaustion import exit_era_579
    assert exit_era_579(10.0, 3.9) and exit_era_579(5.5, 0.4)
    assert exit_era_579(4.0, 0.2)                             # override at ATR 0.2 % stamps 20 × 0.2 = 4 % (≤ 5) → still the override
    assert not exit_era_579(5.0, 3.9) and not exit_era_579(0.6, 0.2) and not exit_era_579(4.5, 1.5)
    assert not exit_era_579(None, 1.0) and not exit_era_579("x", 1.0) and not exit_era_579(4.0, None)
    assert X.is_override_era(10.0, 3.9) and not X.is_override_era(5.0, 4.5)


def _csv(tmp_path, rows):
    import pandas as pd
    p = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return [str(p)]


def _row(**k):
    base = dict(opened_at="2026-11-18T08:30:11", closed_at="2026-11-18T09:25:30", pair="ZKPUSDT", direction="SHORT", status="CLOSED",
                entry_strategy="EXHAUSTION_SHORT", pnl=1.0, pnl_percentage=4.94, close_reason="FRENZY_TP", notional_value=4893.4,
                entry_exh_model_slip_pct=0.0, entry_exh_q5_usd=486652.0, entry_exh_tp_pct=5.0, entry_exh_sl_pct=5.0, entry_slippage_pct=0.13,
                entry_price=0.03838, entry_exh_atr_pct=3.908822)
    base.update(k)
    return base


def test_load_extra_columns_and_rebased_report(tmp_path):
    paths = _csv(tmp_path, [_row(), _row(opened_at="2026-11-19T10:00:00", pair="AAAUSDT", entry_exh_tp_pct=10.0, entry_exh_sl_pct=5.86, entry_exh_atr_pct=3.9,
                                         pnl_percentage=-5.9, close_reason="STOP_LOSS")])
    f = X.load(paths)
    assert {"entry_price", "entry_exh_atr_pct"} <= set(f.columns) and len(f) == 2
    plain = SC.load_fills(paths)
    assert "entry_price" not in plain.columns                 # the default loader is unchanged
    txt = SC.report(plain, SC.load_journal([]))
    # 🔁 (609) the 578 gate now counts the TP-7 era: the TP-5 (ZKP) and TP-10 (AAA) fills are listed, not counted
    assert "earlier exit era (TP 5 / TP 10), not counted" in txt and "ZKPUSDT" in txt and "AAAUSDT" in txt


def test_registry_and_config_pins():
    # ⛔ (DECISION_LOG 609) the DL 579 gate is RETIRED (the sleeve runs TP 7): classified NONE, unhooked; the module stays as the shared library
    assert "scout_exh_exit_override.py" in OR.classified() and "scout_exh_exit_override.py" in OR.NONE and "scout_exh_exit_override.py" not in OR.EXIT
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    rows = {r[0]: r for r in OR.status(cfg)}
    assert rows["scout_exhaustion_short.py"][4] == "OK" and rows["scout_exh_tp7_609.py"][4] == "OK"
    for k, v in (("exhaustion_tp_atr_mult", 20.0), ("exhaustion_tp_cap_pct", 7.0), ("exhaustion_sl_atr_mult", 1.5), ("exhaustion_sl_cap_pct", 6.0),
                 ("exhaustion_max_hold_hours", 24.0)):
        assert cfg[k] == v
    assert X.OVERRIDE == dict(tp_atr=20.0, tp_cap=10.0, sl_atr=1.5, sl_cap=6.0, hold_h=24.0)   # the TP-10 levels (history; the library constant)
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 579" in src and "DECISION_LOG 609" in src
    sc = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "_exo.report(" not in sc and "import scout_exh_tp7_609 as _tp7" in sc


def test_config_accepts_the_override():
    from types import SimpleNamespace as NS
    import config as C
    from services.exhaustion import exhaustion_config_errors, exhaustion_params
    th = C.SignalThresholds()
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    t = NS(**dict(th.model_dump(), **{k: v for k, v in cfg.items() if k.startswith("exhaustion_")}))
    assert exhaustion_config_errors(t) == []
    p = exhaustion_params(t)
    assert p["tp_atr"] == 20.0 and p["tp_cap"] == 7.0 and p["sl_cap"] == 6.0   # 🎯 (609)


def test_replica_check_and_booked_kind():
    assert X.booked_kind("FRENZY_TP") == "TP" and X.booked_kind("STOP_LOSS") == "SL" and X.booked_kind("MAX_HOLD") == "TIME"
    good = [(9.93, "TP", 9.9, "FRENZY_TP")] * 9 + [(-6.1, "SL", -6.2, "STOP_LOSS")]
    m, g, ok = X.replica_check(good)
    assert ok and m == 1.0 and g < 0.1
    bad = [(9.93, "TP", -6.0, "STOP_LOSS")] * 5 + good[:5]
    assert X.replica_check(bad)[2] is False
    assert X.replica_check([]) == (None, None, False)


def test_unreliable_replica_freezes_no_verdict(monkeypatch, tmp_path):
    import json
    import pandas as pd
    base = pd.Timestamp("2026-11-20")
    fills = pd.DataFrame([dict(pair=f"P{i}USDT", opened_at=str(base + pd.Timedelta(hours=7 * i)), entry_price=100.0, entry_exh_atr_pct=4.0,
                               entry_exh_tp_pct=10.0, slip=0.0, pnl_percentage=-6.0, close_reason="STOP_LOSS", o=base + pd.Timedelta(hours=7 * i))
                          for i in range(41)])
    monkeypatch.setattr(X, "_load_cache", lambda: {})
    monkeypatch.setattr(X, "_save_cache", lambda c: None)
    monkeypatch.setattr(X, "price_fill", lambda *a, **k: {"override": (9.93, "TP"), "shipped": (4.93, "TP")})
    st = tmp_path / "st.json"
    txt = X.report(fills, now_ms=0, state_path=str(st))
    assert json.load(open(st))["40"]["verdict"] == "UNRELIABLE" and "NOT RELIABLE" in txt


def test_skipped_checkpoint_is_reported(monkeypatch, tmp_path):
    import pandas as pd
    base = pd.Timestamp("2026-11-20")
    fills = pd.DataFrame([dict(pair=f"P{i}USDT", opened_at=str(base + pd.Timedelta(minutes=10 * i)), entry_price=100.0, entry_exh_atr_pct=4.0,
                               entry_exh_tp_pct=10.0, slip=0.0, pnl_percentage=9.9, close_reason="FRENZY_TP", o=base + pd.Timedelta(minutes=10 * i))
                          for i in range(41)])
    monkeypatch.setattr(X, "_load_cache", lambda: {})
    monkeypatch.setattr(X, "_save_cache", lambda c: None)
    monkeypatch.setattr(X, "price_fill", lambda *a, **k: {"override": (9.93, "TP"), "shipped": (4.93, "TP")})
    assert "checkpoint 40 SKIPPED for good" in X.report(fills, now_ms=0, state_path=str(tmp_path / "s.json"))


def test_booked_price_pct_strips_funding():
    from types import SimpleNamespace as NS
    assert abs(X.booked_price_pct(NS(pnl_percentage=4.9, funding_fee_usd=-10.0, notional_value=5000.0)) - 5.1) < 1e-9
    assert X.booked_price_pct(NS(pnl_percentage=4.9, funding_fee_usd=None, notional_value=None)) == 4.9
    assert X.booked_price_pct(NS(pnl_percentage="x")) is None
