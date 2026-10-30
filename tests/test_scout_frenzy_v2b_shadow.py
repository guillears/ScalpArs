"""🎲🪶 tests for scripts/scout_frenzy_v2b_shadow.py (DECISION_LOG 379 — LITE / LONG on WILLY's V2b exit + fixed higher TPs; observe-only).
Synthetic exports / paths / H, no network, state in tmp, no CSV written."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_frenzy_v2b_shadow as S  # noqa: E402
import scout_willy_exit_v2 as SV     # noqa: E402

MIN, HOUR, DAY = 60_000, 3_600_000, 86_400_000
T0 = int(pd.Timestamp("2026-11-01 00:00:00").value // 1_000_000)
ERA = {250: T0 - 40 * DAY, 283: T0 - 30 * DAY, 284: T0 - 29 * DAY, 301: T0 - 20 * DAY, 317: T0 - 10 * DAY}


def _iso(ms):
    return pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%dT%H:%M:%S")


def _row(te, strat="FRENZY_LONG", pnl=1.5, status="CLOSED", pair="AAAUSDT", **kw):
    r = dict(opened_at=_iso(te), closed_at=_iso(te + 30 * MIN), pair=pair, direction="LONG", entry_strategy=strat, status=status,
             entry_price=1.0, quantity=100.0, entry_fee=0.045, pnl_percentage=pnl, close_reason="FRENZY_TP",
             frenzy_lite_tp_used=None, entry_frenzy_spike_at=_iso(te - 2 * HOUR), entry_frenzy_vol_mult=150.0,
             entry_frenzy_stop_pct=4.0 if strat == "FRENZY_LONG" else 3.0, entry_btc_1h_slope=0.1)
    r.update(kw)
    return r


def _P(te, ev):
    return SV._P(te, ev, cap=S.CAP_LIVE)


RUN_UP = [(5, 1.6, 1.6), (20, 3.5, 3.5)]          # +1.5 replica TP, then on to +3.5 (V2B T ≈ +3.36 hits)


def _run(rows, tmp_path, path_ev=None, H=1.045, floor_ms=T0, now=None, **kw):
    raw = pd.DataFrame(rows)
    ev = path_ev or (lambda r: RUN_UP)
    return S.run(now_ms=now or (T0 + 60 * DAY), raw=raw, floor_ms=floor_ms, path_fn=lambda r: _P(int(r["te"]), ev(r)),
                 h_fn=lambda r: ((H, "test") if H else (None, "none")), fetch=False, state_path=str(tmp_path / "g.json"), csv_path=None,
                 era=ERA, **kw)


def test_pinned_constants():
    assert (S.DL, S.LAG_MIN, S.CAP_LIVE) == (379, 10, 720)
    assert (S.V2B_STOP, S.V2B_CAP, S.V2B_LATE_MIN, S.V2B_LATE_TP) == (4.0, 90, 30.0, 0.25)
    assert (S.V2B_TH.frenzy_willy_tp_pct, S.V2B_TH.frenzy_willy_high_k_pct, S.V2B_TH.frenzy_willy_high_tp_max_pct) == (1.25, 1.0, 8.0)
    assert S.SHADOWS == {"LITE": ("V2B", "V2B_OWN", "TP4", "TP5", "TP6"), "LONG": ("V2B", "TP2", "TP3", "TP4")}
    assert S.OWN_STOP == {"LITE": 3.0, "LONG": 4.0}
    assert (S.GATE_N, S.GATE_DAYS, S.MIN_GAIN, S.ALPHA, S.DAY_SHARE, S.PARITY_TOL, S.PARITY_MIN) == (30, 10, 0.10, 0.05, 0.50, 0.15, 0.90)
    assert S.DIST_EDGES == (3.0, 6.0) and S.WEIGHT_STOP_5M <= 600 and S.MAX_REQ_5M <= 20
    assert S.STATE.startswith(os.path.join(ROOT, "reports")) and S.CSV_OUT.startswith(os.path.join(ROOT, "reports"))


def test_live_exit_replica_rules():
    te = T0 + DAY
    assert S.live_exit(_row(te), ERA) == (1.5, 4.0, "era")
    assert S.live_exit(_row(ERA[283] - HOUR, entry_frenzy_stop_pct=None), ERA)[:2] == (3.0, 3.0)        # pre-283 LONG +3 / pre-317 −3
    assert S.live_exit(_row(ERA[317] + HOUR, entry_frenzy_stop_pct=None), ERA)[:2] == (1.5, 4.0)
    assert S.live_exit(_row(te, "FRENZY_LITE", frenzy_lite_tp_used=2.0), ERA) == (2.0, 3.0, "stored")
    assert S.live_exit(_row(te, "FRENZY_LITE_RETRY", entry_frenzy_vol_mult=25, entry_btc_1h_slope=0.1), ERA)[:2] == (3.0, 3.0)   # zone
    assert S.live_exit(_row(te, "FRENZY_LITE", entry_frenzy_vol_mult=10, entry_btc_1h_slope=0.1), ERA)[:2] == (2.0, 3.0)        # out
    assert S.live_exit(_row(ERA[284] - HOUR, "FRENZY_LITE"), ERA)[0] == 1.5
    assert S.lite_zone(_row(te, entry_frenzy_vol_mult=25)) == "zone" and S.lite_zone(_row(te, entry_btc_1h_slope=None)) == "unread"


def test_v2b_target_and_buckets():
    r = _row(T0)
    T, raw = S.v2b_target(r, 1.045, maker=0.00018)
    assert raw == pytest.approx((1.045 * 0.99 - 1) * 100 - 0.045 - 0.018 * 1.045 * 0.99, abs=1e-3) and T == pytest.approx(raw, abs=1e-3)
    assert S.v2b_target(r, 1.0, maker=0.00018)[0] == 1.25                     # floor +1.25
    assert S.v2b_target(r, 1.5, maker=0.00018)[0] == 8.0                      # cap +8
    assert S.v2b_target(r, None) == (None, None) and S.v2b_target(dict(r, entry_fee=None), 1.05) == (None, None)
    assert [S.dist_bucket(x) for x in (-1, 2.99, 3.0, 5.99, 6.0, None)] == ["near<3", "near<3", "mid3-6", "mid3-6", "far>=6", "noH"]


def test_fixed_tp_shadow_equals_replica_when_the_replica_stops():
    te = T0
    P = _P(te, [(5, 1.0, 1.0), (10, -3.1, -3.1), (30, 5.0, 5.0)])
    rep = S.walk_live(P, 1.5, 3.0)
    assert rep["why"] == "STOP"
    for k in ("TP4", "TP5", "TP6"):
        assert S.shadow_pct(P, k, "LITE", None, 3.0) == pytest.approx(rep["pct"])
    up = _P(te, [(5, 1.6, 1.6), (30, 4.2, 4.2), (40, -3.2, -3.2)])
    assert S.walk_live(up, 1.5, 4.0)["pct"] == 1.5
    assert S.shadow_pct(up, "TP4", "LONG", None) == 4.0 and S.shadow_pct(up, "TP5", "LONG", None) != S.shadow_pct(up, "TP5", "LONG", None)
    assert S.shadow_pct(up, "TP2", "LITE", None) != S.shadow_pct(up, "TP2", "LITE", None)        # not a LITE shadow → nan
    # V2B: stop −4 vs V2B_OWN −3; the late TP +0.25 after 30 min; cap 90
    dip = _P(te, [(5, -3.2, -3.2), (40, 0.3, 0.3)])
    assert S.shadow_pct(dip, "V2B_OWN", "LITE", 3.0) == pytest.approx(-3.2)
    assert S.shadow_pct(dip, "V2B", "LITE", 3.0) == pytest.approx(0.3)                         # late line already through at 40 min


def test_day_bootstrap_and_bonferroni():
    rng = np.random.default_rng(1)
    d = rng.normal(0.3, 1.0, 60)
    day = np.repeat(np.arange(12), 5)
    b = S.day_boot(d, day)
    assert len(b) == S.BOOT_N and abs(b.mean() - d.mean()) < 0.05
    assert S.bonf_lb(d, day, 5) < S.bonf_lb(d, day, 1)
    assert len(S.day_boot(d[:5], day[:5])) == 0 and S.bonf_lb(d[:5], day[:5], 3) != S.bonf_lb(d[:5], day[:5], 3)
    assert S.top_share([1.0, 1.0, 2.0], ["a", "b", "a"]) == pytest.approx(0.75)
    assert S.top_share([-1.0, 0.5], ["a", "b"]) != S.top_share([-1.0, 0.5], ["a", "b"])        # Σ ≤ 0 → nan


def _gate_rows(n_days=10, per_day=3, strat="FRENZY_LONG"):
    return [_row(T0 + HOUR + d * DAY + j * 2 * HOUR, strat, pair=f"P{d}{j}USDT") for d in range(n_days) for j in range(per_day)]


def test_gate_candidate_is_operator_call_and_frozen(tmp_path):
    L = _run(_gate_rows(), tmp_path)
    txt = "\n".join(L)
    assert "LONG_V2B: CANDIDATE" in txt and "OPERATOR CALL (never automatic)" in txt
    assert "LONG_TP2: CANDIDATE" in txt and "LONG_TP3: CANDIDATE" in txt
    assert "LONG_TP4: CANDIDATE" in txt                                # +3.5 never reaches +4 → 12 h cap at +3.5: Δ +2.0
    st = json.load(open(tmp_path / "g.json"))
    assert st["LONG_V2B"]["verdict"] == "CANDIDATE" and "LITE_V2B" not in st
    assert "LITE_V2B: COLLECTING" in txt and "THEORY T1" in txt and "THEORY T2" in txt
    L2 = _run(_gate_rows(), tmp_path, path_ev=lambda r: [(5, 1.6, 1.6), (20, -4.5, -4.5)])
    assert "frozen" in "\n".join(L2) and "LONG_V2B: CANDIDATE" in "\n".join(L2)             # decided ONCE


def test_gate_hold_collecting_parity_and_reference(tmp_path):
    rows = _gate_rows()
    op = _row(T0 + HOUR + 2 * DAY + 30 * MIN, status="OPEN", pair="OPENUSDT")
    txt = "\n".join(_run(rows + [op], tmp_path))
    assert "LONG_V2B: HOLD" in txt and "1 still open" in txt
    (tmp_path / "b").mkdir()
    txt = "\n".join(_run(_gate_rows(n_days=9, per_day=4), tmp_path / "b"))
    assert "LONG_V2B: COLLECTING — 36/30 complete fills on 9/10 days" in txt
    bad = [dict(r, pnl_percentage=-3.0) for r in _gate_rows()]
    (tmp_path / "c").mkdir()
    txt = "\n".join(_run(bad, tmp_path / "c"))
    assert "parity (era replica vs booked, ±0.15): 0/30 = 0 %" in txt and "LONG_V2B: NO_VERDICT" in txt
    assert json.load(open(tmp_path / "c" / "g.json"))["LONG_V2B"]["verdict"] == "NO_VERDICT"            # NO_VERDICT is a decision
    ref = [_row(T0 - 5 * DAY + j * HOUR, "FRENZY_LITE", pnl=2.0, frenzy_lite_tp_used=2.0, pair=f"R{j}USDT") for j in range(4)]
    pre250 = _row(ERA[250] - DAY, pair="OLDUSDT")
    (tmp_path / "d").mkdir()
    txt = "\n".join(_run(ref + [pre250], tmp_path / "d"))
    assert "MASTER / ARCHIVED REFERENCE — IN-SAMPLE" in txt and "LITE: 4 fills" in txt and "trail-era fills (exit not replicable" in txt
    assert "): 1" in txt and not os.path.exists(tmp_path / "d" / "g.json")


def test_not_evaluable_without_commit_and_without_path(tmp_path):
    txt = "\n".join(_run(_gate_rows(), tmp_path, floor_ms=None))
    assert "GATE cohort: NOT evaluable yet — no DECISION_LOG 379 commit" in txt and "not evaluable (no DL 379 commit yet)" in txt
    raw = pd.DataFrame(_gate_rows(n_days=1, per_day=2))
    L = S.run(now_ms=T0 + 2 * HOUR, raw=raw, floor_ms=T0, path_fn=lambda r: None, h_fn=lambda r: (None, "none"), fetch=False,
              state_path=str(tmp_path / "n.json"), csv_path=None, era=ERA)
    assert "not evaluable (no path) 2" in "\n".join(L) and "[NOT_EVALUABLE]" in "\n".join(L)


def test_dedup_and_operator_closed():
    rows = [_row(T0, status="OPEN"), _row(T0, pnl=1.5), _row(T0 + HOUR, close_reason="MANUAL_CLOSE"), _row(T0 + 2 * HOUR, "SPIKE_FADE")]
    co = S.cohort(pd.DataFrame(rows), T0)
    assert len(co["gate"]) == 1 and len(co["ox"]) == 1 and len(co["open"]) == 0


def test_writes_only_under_reports(tmp_path):
    c = pd.DataFrame([dict(pair="X")])
    with pytest.raises(ValueError):
        S.write_csv(c, str(tmp_path / "x.csv"))


def test_wired_into_the_opportunity_scout():
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_frenzy_v2b_shadow as _fv2b" in src and "_fv2b.run(now_ms)" in src


# ─────────────────────────── review fixes (dual review, DL 379) ───────────────────────────
def test_delta_is_vs_today_not_the_era_replica(tmp_path):
    """a pre-283 LONG fill ran +3 / −3: parity uses that era exit, Δ uses TODAY's +1.5 / −4 and the shadows today's −4."""
    te = ERA[283] - DAY
    r = _row(te, pnl=-3.0, close_reason="STOP_LOSS", entry_frenzy_stop_pct=None)
    ev = [(5, 1.6, 1.6), (10, -3.2, -3.2), (30, 2.5, 2.5)]                 # TODAY: +1.5 TP · era (+3 / −3): stop −3.2
    c = S.price(S.cohort(pd.DataFrame([r]), None)["ref"], T0, path_fn=lambda x: _P(int(x["te"]), ev), h_fn=lambda x: (1.045, "t"),
                fetch=False, era=ERA)
    x = c.iloc[0]
    assert (x.tp_live, x.stop_live, x.tp_today, x.stop_today) == (3.0, 3.0, 1.5, 4.0)
    assert x.rep_era == pytest.approx(-3.2) and x.rep == pytest.approx(1.5)
    assert x.sh_TP2 == pytest.approx(2.0) and x.d_TP2 == pytest.approx(0.5)            # TP2 at −4 survives the −3.2 dip
    assert S.today_exit(_row(te, "FRENZY_LITE", entry_frenzy_vol_mult=25)) == (3.0, 3.0)
    assert S.today_exit(_row(te, "FRENZY_LITE", entry_frenzy_vol_mult=5)) == (3.0, 3.0)   # LITE C off since DL 420
    assert S.today_exit(_row(te, "FRENZY_LITE_RETRY", frenzy_lite_tp_used=3.0, entry_frenzy_vol_mult=5)) == (3.0, 3.0)


def test_no_verdict_never_flips(tmp_path):
    bad = [dict(r, pnl_percentage=-3.0) for r in _gate_rows()]
    assert "LONG_V2B: NO_VERDICT" in "\n".join(_run(bad, tmp_path))
    good = _gate_rows(n_days=14)                                             # same prefix, now parity-perfect and more fills
    txt = "\n".join(_run(good, tmp_path))
    assert "LONG_V2B: NO_VERDICT" in txt and "frozen" in txt and "LONG_V2B: CANDIDATE" not in txt


def test_parity_is_read_on_the_prefix_only(tmp_path):
    rows = _gate_rows() + [dict(r, pnl_percentage=-3.0) for r in _gate_rows(n_days=20)[30:]]   # later fills fail parity
    txt = "\n".join(_run(rows, tmp_path))
    assert "prefix parity 100 %" in txt and "LONG_V2B: CANDIDATE" in txt


def test_shortest_prefix_later_losers_cannot_change_the_read(tmp_path):
    rows = _gate_rows() + [_row(T0 + 20 * DAY + j * HOUR, pair=f"L{j}USDT") for j in range(40)]
    lose = lambda r: RUN_UP if int(r["te"]) < T0 + 15 * DAY else [(5, 1.6, 1.6), (20, -4.5, -4.5)]      # noqa: E731
    txt = "\n".join(_run(rows, tmp_path, path_ev=lose))
    assert "LONG_V2B: CANDIDATE — 30 fills on 10 days" in txt


def test_candidate_with_non_identical_deltas(tmp_path):
    def ev(r):
        j = int((int(r["te"]) - T0) // HOUR) % 7
        v = 2.5 + 0.2 * j                                                   # some below T ≈ +3.36 → V2B caps at v (varied Δ)
        return [(5, 1.6, 1.6), (20, v, v)] if j != 3 else [(5, 1.6, 1.6), (20, 0.5, 0.5)]   # one loser shape (Δ −1.0)
    txt = "\n".join(_run(_gate_rows(), tmp_path, path_ev=ev))
    line = [x for x in txt.split("\n") if x.startswith("  LONG_V2B:")][0]
    assert "CANDIDATE" in line
    c = S.price(S.cohort(pd.DataFrame(_gate_rows()), T0)["gate"], T0 + 60 * DAY, path_fn=lambda r: _P(int(r["te"]), ev(r)),
                h_fn=lambda r: (1.045, "t"), fetch=False, era=ERA)
    assert c.d_V2B.nunique() > 2


def test_hold_on_a_young_incomplete_fill_and_not_evaluable_when_old(tmp_path):
    rows = _gate_rows() + [_row(T0 + 9 * DAY + 30 * MIN, pair="NOHUSDT")]
    hfn = lambda r: (None, "none") if r["pair"] == "NOHUSDT" else (1.045, "t")     # noqa: E731
    raw = pd.DataFrame(rows)
    run = lambda now, sp: S.run(now_ms=now, raw=raw, floor_ms=T0, path_fn=lambda r: _P(int(r["te"]), RUN_UP), h_fn=hfn, fetch=False,  # noqa: E731
                                state_path=sp, csv_path=None, era=ERA)
    young = "\n".join(run(T0 + 10 * DAY + 12 * HOUR, str(tmp_path / "y.json")))
    assert "LONG_V2B: HOLD" in young and "LONG_TP2: HOLD" in young and "1 incomplete < 3 days old" in young
    old = "\n".join(run(T0 + 60 * DAY, str(tmp_path / "o.json")))
    assert "LONG_V2B: CANDIDATE" in old and "1 not evaluable (incomplete ≥ 3 days: NOHUSDT" in old
    assert "LONG_TP2: CANDIDATE" in old and "1 not evaluable" in [x for x in old.split("\n") if x.startswith("  LONG_TP2")][0]


def test_unreadable_state_file_is_never_overwritten(tmp_path, capsys):
    sp = tmp_path / "g.json"
    sp.write_text("{not json")
    txt = "\n".join(_run(_gate_rows(), tmp_path))
    assert "state file unreadable — not frozen" in txt and sp.read_text() == "{not json"
    assert "left untouched" in capsys.readouterr().err
    v, t = S.frozen(str(tmp_path / "new.json"), "X", "KEEP", "t", True)
    assert json.load(open(tmp_path / "new.json"))["X"]["verdict"] == "KEEP" and not list(tmp_path.glob("*.tmp"))


def test_resolve_h_uses_the_h_window_without_look_ahead(monkeypatch):
    import build_willy_exit_cf as CF
    import scout_willy_latejoin as LJ
    te = T0 + 3 * HOUR + 7_000                                             # fill 7 s after a 5m close
    sp = T0 + HOUR                                                          # spike bar CLOSE
    b0, last = CF.h_window(sp, te)
    assert b0 == sp - S.BAR5 and last == (te // S.BAR5) * S.BAR5 - S.BAR5
    monkeypatch.setattr(CF, "reconstruct_high", lambda *a, **k: (None, "none"))
    seen = {}

    def fake_chunk(pair, start, limit, now_ms, bud):
        seen.update(start=start, limit=limit)
        rows = [[b0 + i * S.BAR5, 1, 1.01 + 0.001 * i, 0.99, 1, 0, 0, 1000] for i in range(limit)]
        return rows + [[b0 + limit * S.BAR5, 1, 9.0, 1, 1, 0, 0, 1000]]                       # the bar AFTER the window: look-ahead
    monkeypatch.setattr(LJ, "_fetch_chunk", fake_chunk)
    r = _row(te, entry_frenzy_spike_at=_iso(sp))
    H, src = S.resolve_h(r, T0 + DAY, S._Bud(True), {})
    n = (last - b0) // S.BAR5 + 1
    assert seen == {"start": b0, "limit": n} and src == "5m_fetch" and H == pytest.approx(1.01 + 0.001 * (n - 1))
    assert S.resolve_h(dict(r, entry_frenzy_spike_at=None), T0 + DAY, S._Bud(True), {}) == (None, "no_spike_stamp")


def test_h_fetch_budget_weight_stop_and_cooldown(monkeypatch, tmp_path):
    import scout_willy_latejoin as LJ
    import scout_b1h_negflank as NF
    monkeypatch.setattr(LJ, "_NET_BLOCKED", True)                         # any real request would raise
    monkeypatch.setattr(LJ, "CACHE", str(tmp_path / "5m"))
    monkeypatch.setattr(LJ, "BAD_SYMBOLS", str(tmp_path / "bad.json"))
    b = S._Bud(True)
    b.used = S.WEIGHT_STOP_5M                                              # weight stop → no request, cache only
    assert S._h_from_5m("ZZTESTUSDT", T0, T0 + S.BAR5, T0 + DAY, b) is None
    assert b.stop.startswith("used weight") and b.fetch is False and b.n == 0
    b = S._Bud(True)
    monkeypatch.setattr(NF, "_cooldown_until", lambda: T0 + 10 * DAY)    # shared cooldown active → no request
    assert S._h_from_5m("ZZTESTUSDT", T0, T0 + S.BAR5, T0 + DAY, b) is None and "cooldown" in b.stop and b.n == 0
    b = S._Bud(True)
    monkeypatch.setattr(NF, "_cooldown_until", lambda: 0)
    b.n = b.max_req                                                        # request cap
    assert S._h_from_5m("ZZTESTUSDT", T0, T0 + S.BAR5, T0 + DAY, b) is None and "request cap" in b.stop
    assert S._h_from_5m("ZZTESTUSDT", T0, T0 + (LJ.KL_MAX + 5) * S.BAR5, T0 + DAY, S._Bud(True)) is None     # window > one page
