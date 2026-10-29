"""🚪 Oct-29: EXIT_SHADOWS — FLIP only (scripts/scout_exit_shadows.py, DECISION_LOG 411): the sleeve, the per-fill live stop, the
short/long net path with the real entry fee, the fixed-TP walk, path parity, the frozen verdict (luck-corrected over 3 exits), the
per-fill result cache + signature re-walk, the gap retry, misses on distinct days, the export-age deferral, NOT EVALUABLE, the scout hook
and the registry. Hermetic: synthetic paths, no network, temp state."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_exit_shadows as X  # noqa: E402

MIN = 60_000
TE = 1_761_000_008_000


def test_frozen_constants_sleeve_and_stops():
    assert (X.DL, X.N_MIN, X.DAYS_MIN, X.MIN_GAIN, X.ALPHA, X.SHARE_MAX, X.PARITY_MIN) == (411, 20, 8, 0.10, 0.10, 0.50, 0.90)
    assert X.SLEEVES == {"FLIP": (("FLIP:", None), ("TP1.5", "TP2", "TP3"))}   # operator 10-29: flips only, +1.5 / +2 / +3
    assert X.sleeve_of("FLIP:FAN_RATIO_GATE", "SHORT") == "FLIP" and X.sleeve_of("FLIP:FAN_RATIO_GATE", "LONG") == "FLIP"
    for es, d in (("SPIKE_FADE", "SHORT"), ("MOMENTUM", "LONG"), ("MOMENTUM", "SHORT"), ("FRENZY_LITE", "LONG")):
        assert X.sleeve_of(es, d) is None
    # the engine's stop: −0.7 widened to 1.5 × ATR, capped −1.2 (B25 flips: ATR 1.48 → −1.20)
    assert X.stop_of("FLIP", 0.256) == 0.7 and abs(X.stop_of("FLIP", 0.602) - 0.903) < 1e-9
    assert X.stop_of("FLIP", 1.48) == 1.2 and X.stop_of("FLIP", None) == 0.7 and X.stop_of("FLIP", float("nan")) == 0.7
    assert "reports/SCOUT_EXIT_SHADOWS.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def _gross(pts, n_tick=0):
    """[(minute, hi, lo, cl)] gross-long % events (bars open at the entry level) → an assemble_path-like dict."""
    t = np.array([TE + int(m * MIN) for m, *_ in pts], "int64")
    hi, lo, cl = (np.array([p[i] for p in pts], float) for i in (1, 2, 3))
    return dict(t=t, hi=hi, lo=lo, op=np.zeros(len(t)), cl=cl, end_ms=int(t[-1]) if len(t) else TE, te=TE, n_tick_s=n_tick)


def test_to_net_short_mirrors_long_and_uses_the_real_entry_fee():
    g = _gross([(1, 2.0, -1.0, 0.5)])
    L, S = X.to_net(g, "LONG"), X.to_net(g, "SHORT")
    assert abs(L["hi"][0] - (2.0 - 0.045 - 0.045 * 1.02)) < 1e-9 and abs(L["lo"][0] - (-1.0 - 0.045 - 0.045 * 0.99)) < 1e-9
    assert abs(S["hi"][0] - (1.0 - 0.045 - 0.045 * 0.99)) < 1e-9          # a short's best = the LOW price
    assert abs(S["lo"][0] - (-2.0 - 0.045 - 0.045 * 1.02)) < 1e-9          # its worst = the HIGH price
    assert abs(X.to_net(g, "SHORT", fin=0.018)["hi"][0] - (1.0 - 0.018 - 0.045 * 0.99)) < 1e-9   # maker entry fee (deep review H1)
    assert S["bar"].tolist() == [True] and X.to_net(_gross([(1, 0, 0, 0)], n_tick=1), "SHORT")["bar"].tolist() == [False]
    g2 = _gross([(1, 0.0, -3.0, -2.5)]); g2["op"] = np.array([-2.0])        # a 1m-only first bar "opening" through the stop
    assert X.to_net(g2, "LONG")["op"][0] == X.to_net(_gross([(1, 0, 0, 0)]), "LONG")["op"][0]   # → opens AT the entry (no fake gap)


def test_fixed_walk_and_exits_on():
    P = X.to_net(_gross([(5, 0.5, -0.2, 0.3), (40, 3.5, 0.2, 3.0), (100, 3.6, 2.0, 2.5)]), "LONG")
    assert abs(X.walk_fixed(P, 3.0, 0.7) - 3.0) < 1e-9                    # TP at the line (resting)
    stop = X.to_net(_gross([(5, 0.1, -1.0, -0.9)] + [(m, 0, 0, 0) for m in range(6, 722)]), "LONG")
    assert abs(X.walk_fixed(stop, 3.0, 0.7) - (-0.7 - X.SLIP)) < 1e-9      # the stop at its line (no slip either side)
    slow = X.to_net(_gross([(m, 0.4, -0.1, 0.1) for m in range(1, 800)]), "LONG")
    r = type("R", (), dict(sleeve="FLIP", sl=1.2, pct=0.3, tc=TE + 120 * MIN))()
    out = X.exits_on(slow, r)
    assert set(out) == {"TP1.5", "TP2", "TP3"} and abs(out["TP2"] - slow["cl"][718]) < 1e-9   # never +2 → the 12 h cap
    assert np.isnan(X.walk_fixed(X.to_net(_gross([(5, 0, 0, 0)]), "LONG"), 2.0, 1.2))       # path too short → undecided


def test_parity_rules():
    P = X.to_net(_gross([(5, 0.2, -1.0, -0.8), (20, 0.5, -0.5, 0.3)]), "LONG")
    assert X.parity_ok(P, 0.7, -0.7, "FLIP_STOP_LOSS L1", TE + 6 * MIN) is True
    assert X.parity_ok(P, 0.7, 0.3, "FLIP_RUNNER_TRAIL", TE + 20 * MIN) is False   # the path says the stop was hit, the bot rode on
    assert X.parity_ok(P, 1.2, 0.3, "FLIP_RUNNER_TRAIL", TE + 20 * MIN) is True    # the same fill on a widened −1.2 stop is consistent
    assert X.parity_ok(P, 0.7, -0.72, "EMA13_CROSS_EXIT L1", TE + 20 * MIN) is True
    calm = X.to_net(_gross([(5, 0.3, -0.1, 0.1)]), "LONG")
    assert X.parity_ok(calm, 0.7, -0.7, "FLIP_STOP_LOSS L1", TE + 5 * MIN) is False
    assert X.parity_ok(calm, 0.7, 0.1, "FLIP_RUNNER_TRAIL", TE + 50 * MIN) is None
    assert X.is_stop_close("FLIP_STOP_LOSS L1") and X.is_stop_close("PATTERN_FIXED_SL L1") and not X.is_stop_close("FLIP_TRAILING_STOP")


def _rows(deltas, days=8, booked=0.1, parity=True):
    n = len(deltas)
    return pd.DataFrame(dict(pct=[booked] * n, day=[f"2026-11-{1 + i % days:02d}" for i in range(n)],
                             exits=[{"TP2": booked + d, "TP3": booked} for d in deltas], parity=[parity] * n))


def test_verdict_candidate_keep_share_parity_and_crossing():
    v, ps = X.verdict(_rows([0.4] * 20), ("TP2", "TP3"))
    assert v["TP2"][0].startswith("CANDIDATE") and v["TP3"][0].startswith("KEEP") and ps == 1.0
    assert X.verdict(_rows([0.05] * 20), ("TP2",))[0]["TP2"][0].startswith("KEEP")              # < +0.10 / fill
    assert X.verdict(_rows([5.0] + [0.0] * 19), ("TP2",))[0]["TP2"][0].startswith("KEEP")       # one fill carries the gain
    assert X.verdict(_rows([0.4] * 20, parity=False), ("TP2",))[0]["TP2"][0].startswith("KEEP")  # path not trusted
    r = _rows([0.4] * 20); r.at[0, "exits"] = {"TP2": np.nan}
    assert "N 19" in X.verdict(r, ("TP2",))[0]["TP2"][1]                                        # an unpriced exit is out of its N
    assert X.crossing(_rows([0.1] * 25)) == 20 and X.crossing(_rows([0.1] * 25, days=3)) is None


def _export(tmp_path, n, start="2026-11-01", stamp="2026-12-01_00-00-00"):
    rows = []
    for i in range(n):
        o = pd.Timestamp(start) + pd.Timedelta(days=i % 8, minutes=i)
        rows.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=20)).strftime("%Y-%m-%dT%H:%M:%S"),
                         pair=f"P{i}USDT", direction="LONG", entry_strategy="FLIP:FAN_RATIO_GATE", status="CLOSED", entry_price=1.0,
                         pnl_percentage=0.1, pnl=1.0, close_reason="FLIP_RUNNER_TRAIL", cell_multiplier_source="UNMATCHED",
                         entry_atr_pct=0.3, entry_fee=0.18, notional_value=1000.0))
    rows.append(dict(rows[0], pair="PROBEUSDT", cell_multiplier_source="GAPFLAT_PROBE"))
    rows.append(dict(rows[0], pair="MOMUSDT", entry_strategy="MOMENTUM"))
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return [str(p)]


def _good(r):
    g = _gross([(5, 2.2, 0.0, 2.0)] + [(m, 2.0, 1.9, 2.0) for m in range(6, 760)])
    sh = int(r.te) - TE
    return dict(g, te=int(r.te), t=g["t"] + sh, end_ms=g["end_ms"] + sh)


def test_run_freezes_once_caches_and_waits_for_a_late_export(tmp_path, monkeypatch):
    since = int(pd.Timestamp("2026-11-01").value // 10**6)
    now = int(pd.Timestamp("2026-12-01").value // 10**6)
    calls = []
    monkeypatch.setattr(X, "raw_path", lambda r, *a, **k: (calls.append(r.pair) or _good(r), "ok"))
    st = tmp_path / "x.json"
    early = _export(tmp_path, 20, stamp="2026-11-08_12-00-00")              # the newest export is < the prefix end + 1200 min → wait
    out0 = "\n".join(X.run(now, early, str(st), since, fetch=True))
    assert "newest export < the prefix end + 1200 min" in out0 and "Verdict (prefix" not in out0
    assert json.load(open(st))["_fills"]                                    # results cached even while the freeze waits
    n_calls = len(calls)
    paths = _export(tmp_path, 20)
    out = "\n".join(X.run(now, paths, str(st), since, fetch=True))
    assert len(calls) == n_calls                                            # cached → never re-walked
    assert "FLIP — counted" in out and "PROBEUSDT" not in out and "MOMUSDT" not in out and "Verdict (prefix" in out
    assert json.load(open(st))["FLIP"]["exits"]["TP2"][0].startswith("CANDIDATE")
    monkeypatch.setattr(X, "raw_path", lambda *a, **k: (None, "empty"))
    assert "Frozen verdict" in "\n".join(X.run(now, paths, str(st), since, fetch=True))   # never re-decided
    monkeypatch.setattr(X, "raw_path", lambda r, *a, **k: (_good(r), "ok"))
    dry = "\n".join(X.run(now, paths, None, since, fetch=False))
    assert "NOT FROZEN" in dry and "Frozen verdict" not in dry              # a run without state never claims a frozen verdict


def test_signature_rewalk_gap_retry_and_misses(tmp_path, monkeypatch):
    since = int(pd.Timestamp("2026-11-01").value // 10**6)
    now = int(pd.Timestamp("2026-12-01").value // 10**6)
    calls = []
    monkeypatch.setattr(X, "raw_path", lambda r, *a, **k: (calls.append(r.pair) or _good(r), "ok"))
    paths, st = _export(tmp_path, 3), str(tmp_path / "s.json")
    X.run(now, paths, st, since, fetch=True); X.run(now, paths, st, since, fetch=True)
    assert len(calls) == 3                                                  # cached
    monkeypatch.setattr(X, "FIX_CAP", 600)                                  # an exit constant changes → every fill re-walked
    X.run(now, paths, st, since, fetch=True)
    assert len(calls) == 6
    p2 = _export(tmp_path, 3, stamp="2026-12-02_00-00-00")
    df = pd.read_csv(p2[0]); df.loc[0, "pnl_percentage"] = 0.2; df.to_csv(p2[0], index=False)   # a corrected booked % → that fill only
    X.run(now, paths + p2, st, since, fetch=True)
    assert len(calls) == 7
    monkeypatch.setattr(X, "raw_path", lambda r, *a, **k: (_good(r), "gap"))
    st2 = str(tmp_path / "g.json")
    assert "priced 0 · pending 3" in "\n".join(X.run(now, paths, st2, since, fetch=True))       # a short page: retry another day first
    assert "priced 3 · pending 0" in "\n".join(X.run(now + 86_400_000, paths, st2, since, fetch=True))
    monkeypatch.setattr(X, "raw_path", lambda *a, **k: (None, "empty"))
    st3 = str(tmp_path / "m.json")
    for day in ("2026-12-01", "2026-12-01", "2026-12-02"):
        out = "\n".join(X.run(int(pd.Timestamp(day).value // 10**6), paths, st3, since, fetch=True))
    assert "unpriceable 0" in out                                           # 2 distinct days so far → still pending
    out = "\n".join(X.run(int(pd.Timestamp("2026-12-03").value // 10**6), paths, st3, since, fetch=True))
    assert "unpriceable 3" in out                                           # the 3rd distinct day → UNPRICEABLE
    assert "priced 0 · pending 3" in "\n".join(X.run(since + 60 * MIN, paths, None, since, fetch=False))   # window open → pending


def test_not_evaluable_hook_and_registry(monkeypatch):
    monkeypatch.setattr(X._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(X.run(now_ms=0, paths=[], state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py")).read()
    j = src.index("import scout_exit_shadows as _exs")
    assert "_exs.run(now_ms)" in src[j:j + 200] and "except Exception as _exs_e" in src[j:j + 400]
    import observe_registry as OR
    e = OR.EXIT["scout_exit_shadows.py"]
    assert e["sleeves"] == "FLIP"
    assert {k: e["walks"][k] for k in ("sl_atr_multiplier", "sl_atr_widen_floor_pct", "flip_fan_runner_strpk")} == \
        {"sl_atr_multiplier": 1.5, "sl_atr_widen_floor_pct": -1.2, "flip_fan_runner_strpk": True}
    assert "runner_trail_short_arm_peak" in e["walks"] and "momentum_long_sl_atr_threshold" not in e["walks"]   # the FAN flip exit pinned
    assert OR.diffs(e["walks"], OR.thresholds()) == []


def test_load_entry_fee_and_a_short_flip_walk(tmp_path):
    """review: the real entry fee (entry_fee / notional; missing or out of range → taker 0.045), and a SHORT flip reaching TP2 on a drop."""
    p = _export(tmp_path, 2)
    df = pd.read_csv(p[0]); df.loc[1, "entry_fee"] = np.nan; df.to_csv(p[0], index=False)
    d, _ = X.load(p)
    assert dict(zip(d.pair, d.fin.round(3))) == {"P0USDT": 0.018, "P1USDT": 0.045}
    g = _gross([(5, 0.3, -2.5, -2.2)] + [(m, -2.0, -2.2, -2.1) for m in range(6, 730)])   # price falls 2.5 % → a short's +2
    P = X.to_net(g, "SHORT", 0.018)
    r = type("R", (), dict(sleeve="FLIP", sl=1.2, pct=0.3, tc=TE + 30 * MIN))()
    assert X.exits_on(P, r)["TP2"] == 2.0 and abs(X.exits_on(P, r)["TP1.5"] - 1.5) < 1e-9
