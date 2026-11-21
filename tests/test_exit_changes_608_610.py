"""DECISION_LOG 608 (WILLY_HT TP +6) · 609 (EXHAUSTION TP cap 7) · 610 (Open Orders stop line in plain words) — config, era rule, both
revert gates (pure + report freeze / hold), registry, hooks, UI wiring."""
import json
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _cfg():
    return json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]


def _src(p):
    return open(os.path.join(ROOT, p), encoding="utf-8").read()


# ── config ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_config_values_and_evidence_comments():
    import config as C
    c = _cfg()
    assert c["frenzy_willy_ht_tp_pct"] == 6.0 and c["exhaustion_tp_cap_pct"] == 7.0
    assert c["exhaustion_tp_atr_mult"] == 20.0 and c["exhaustion_sl_cap_pct"] == 6.0 and c["frenzy_willy_stop_pct"] == 4.0   # unchanged
    th = C.SignalThresholds()
    assert th.frenzy_willy_ht_tp_pct == 0.0 and th.exhaustion_tp_cap_pct == 5.0                    # code defaults unchanged
    s = _src("config.py")
    assert "DL 608" in s and "scripts/scout_ht_tp6_608.py" in s and "DECISION_LOG 609" in s and "scripts/scout_exh_tp7_609.py" in s
    from services.exhaustion import exhaustion_config_errors, exhaustion_params, exit_levels
    t = NS(**dict(th.model_dump(), **c))
    assert exhaustion_config_errors(t) == [] and exhaustion_params(t)["tp_cap"] == 7.0
    assert exit_levels(3.0, t) == (7.0, 4.5) and exit_levels(0.2, t) == (4.0, 0.3)                  # min(20 × ATR, 7) / min(1.5 × ATR, 6)
    from services.frenzy import frenzy_willy_ht_tp
    assert frenzy_willy_ht_tp(t) == 6.0


def test_open_exh_row_stamped_tp10_runs_the_new_cap():
    """a row OPEN at the deploy (stamp 10) is clamped to today's cap by row_levels → the resting-TP target moves to 7 (the sweep re-prices the
    maker limit). The engine runs EXH with resting_tp=True: the software check never books the TP itself (the maker limit does). Known gap
    (DL 609 deep review #3, documented): a short already > 7 % in profit at the deploy cannot re-price (the new limit would cross) → it keeps
    the old −10 % limit until price comes back above the 7 % level."""
    import config as C
    from services.exhaustion import row_levels, exhaustion_exit_for, rtp_net_tp
    from services.resting_tp import _exhaustion_row_target
    t = NS(**dict(C.SignalThresholds().model_dump(), **_cfg()))
    assert row_levels(dict(entry_exh_tp_pct=10.0, entry_exh_sl_pct=6.0), t) == (7.0, 6.0)
    assert row_levels(dict(entry_exh_tp_pct=5.5, entry_exh_sl_pct=4.0), t) == (5.5, 4.0)
    row = dict(entry_exh_tp_pct=10.0, entry_exh_sl_pct=6.0)
    assert exhaustion_exit_for(100.0, 92.9, 60, t, row=row, resting_tp=True)[0] is False          # the maker limit books it, not software
    assert exhaustion_exit_for(100.0, 106.1, 60, t, row=row, resting_tp=True)[:2] == (True, "STOP_LOSS")
    r = NS(entry_strategy="EXHAUSTION_SHORT", entry_price=100.0, quantity=10.0, entry_fee=0.45, entry_exh_tp_pct=10.0, entry_exh_sl_pct=6.0)
    tgt = _exhaustion_row_target(t, r)
    assert tgt is not None and tgt[0] == pytest.approx(rtp_net_tp(100.0, 0.045, 0.00018, 7.0), abs=1e-6)   # the limit target = the 7 % level


def test_exit_era_609():
    from services.exhaustion import exit_era_609, exit_era_579
    assert exit_era_609(7.0, 3.0) and exit_era_609(6.2, 0.31)                                     # TP-7 stamps (cap / 20 × ATR)
    assert not exit_era_609(10.0, 3.0) and exit_era_579(10.0, 3.0)                                 # the TP-10 era
    assert not exit_era_609(5.0, 3.7) and not exit_era_609(4.9, 3.0)                               # the shipped TP-5 era (≤ 3 × ATR)
    assert not exit_era_609(None, 3.0) and not exit_era_609(float("nan"), 3.0)
    assert exit_era_609(7.0, 3.0, "2026-11-22T01:00:00") and not exit_era_609(6.2, 0.31, "2026-11-20 10:00:00")   # the date floor (review)


# ── HT_TP6 gate ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_ht_tp6_classify_and_verdict():
    import scout_ht_tp6_608 as H
    assert H.FEE_GAP == pytest.approx((0.00045 - 0.00018) * 100)                                     # taker − maker exit fee, % pts
    assert H.classify(-4.1, "STOP_LOSS", 4.975) == ("REACHED", pytest.approx(-9.1))                  # a +5 limit fill reads peak ≈ 4.974 (deep review)
    assert H.classify(4.966, "FRENZY_TP", 4.974)[0] == "REACHED"
    assert H.classify(6.0, "FRENZY_TP", 6.03) == ("REACHED", pytest.approx(1.0))
    assert H.classify(-4.1, "STOP_LOSS", 5.2) == ("REACHED", pytest.approx(-9.1))                 # reached +5, then the stop
    assert H.classify(-4.1, "STOP_LOSS", 4.95) == ("NOT_REACHED", 0.0)                             # ENSO-like: never +5
    assert H.classify(1.26, "FRENZY_TP", 1.26) == ("NOT_REACHED", 0.0)                             # an older era's low TP never counts
    assert H.classify(-4.1, "STOP_LOSS", None) == ("UNSCORED", None) and H.classify(None, "STOP_LOSS", 6.0)[0] == "UNSCORED"
    assert H.verdict([1.0] * 9)[0] == "COLLECTING"
    assert H.verdict([1.0] * 9 + [-9.0])[0] == "KEEP"                                             # Σ 0 → keep (only < 0 reverts)
    assert H.verdict([1.0] * 8 + [-9.0, -9.0])[0] == "REVERT"                                      # two misses fire it
    assert H.verdict([1.0] * 10 + [-9.0] * 5)[0] == "KEEP"                                         # only the first 10 count


def _ht_rows(spec, t0="2026-11-25"):
    T0 = pd.Timestamp(t0)
    return pd.DataFrame([dict(opened_at=(T0 + pd.Timedelta(hours=3 * i)).strftime("%Y-%m-%dT%H:%M:%S"), pair=f"P{i}USDT", direction="LONG",
                              status="CLOSED", entry_strategy="FRENZY_WILLY_HT", pnl_percentage=p, close_reason=r, peak_pnl=pk)
                         for i, (p, r, pk) in enumerate(spec)]).assign(o=lambda d: pd.to_datetime(d.opened_at))


def test_ht_tp6_report_freezes_once_and_waits_for_the_export(tmp_path):
    import scout_ht_tp6_608 as H
    spec = [(6.0, "FRENZY_TP", 6.1)] * 8 + [(-4.1, "STOP_LOSS", 5.3)] * 2 + [(-4.1, "STOP_LOSS", 2.0)]
    f = _ht_rows(spec)
    st = tmp_path / "s.json"
    since = int(pd.Timestamp("2026-11-24").value // 10**6)
    last = f.o.iloc[9]
    out = "\n".join(H.report(frame=f, since_ms=since, state_path=str(st), now_ms=since, newest=last + pd.Timedelta(hours=2)))
    assert "⏸ freezing deferred" in out and not st.exists()                                       # newest export < the 10th fill + 24 h
    out = "\n".join(H.report(frame=f, since_ms=since, state_path=str(st), now_ms=since, newest=last + pd.Timedelta(hours=30)))
    assert "FROZEN" in out and json.load(open(st))["HT_TP6"]["verdict"] == "REVERT" and "1 never reached +5" in out
    f2 = _ht_rows([(6.0, "FRENZY_TP", 6.1)] * 10)
    out = "\n".join(H.report(frame=f2, since_ms=since, state_path=str(st), now_ms=since, newest=last + pd.Timedelta(days=9)))
    assert "REVERT" in out                                                                         # frozen once — never re-decided


def test_ht_tp6_cohort_needs_the_plus6_stored_target(tmp_path):
    import scout_ht_tp6_608 as H
    f = _ht_rows([(-4.1, "STOP_LOSS", 5.3)] * 3).assign(frenzy_willy_tp_target_pct=[5.0, 6.0, 6.0])
    out = "\n".join(H.report(frame=f, since_ms=int(pd.Timestamp("2026-11-24").value // 10**6), state_path=str(tmp_path / "s.json"),
                             newest=pd.Timestamp("2026-12-30")))
    assert "2 reached +5" in out and "1 opened inside the deploy lag with the old +5 target (out)" in out


def test_ht_tp6_not_evaluable_before_the_commit(tmp_path, monkeypatch):
    import scout_ht_tp6_608 as H
    monkeypatch.setattr(H._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(H.report(frame=_ht_rows([(6.0, "FRENZY_TP", 6.1)]), state_path=str(tmp_path / "s.json")))


# ── EXH_TP7 gate ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_exh_tp7_hold_watches_the_last_fill_of_the_cut():
    import scout_exh_tp7_609 as X
    rows = [(0, -3.0)] * 10 + [(1, -3.0)] * 10 + [(d, -3.0) for d in (2, 3, 4)]                  # 20 fills on 2 days, the cut needs 23 (5 days)
    st, _, k = X.gate_k(rows)
    assert st == "REVERT" and k == 23                                                             # the freeze hold must watch fill 23, not fill 20
    assert X.gate_k(rows[:20])[2] is None


def test_exh_tp7_funding_read():
    """💸 (DECISION_LOG 613) booked funding as % of the entry notional + the hold — information only."""
    import scout_exh_tp7_609 as X
    pct, h = X.funding_read(-87.38, 0.01563, 338877.0, "2026-11-21T14:30:40", "2026-11-21T21:19:04.1")
    assert pct == pytest.approx(-87.38 / (0.01563 * 338877.0) * 100, abs=0.006) and h == pytest.approx(6.807, abs=0.01)
    assert X.funding_read(None, 1.0, 1.0, "2026-11-21T00:00:00", None) == (None, None)
    assert X.funding_read(5.0, 0.0, 1.0, "x", "y") == (None, None)
    assert X.funding_read(-1.0, 1.0, 100.0, "2026-11-21T00:00:00", pd.NaT)[1] is None                # NaT closed_at
    assert X.funding_read(-1.0, 1.0, 100.0, "2026-11-21T05:00:00", "2026-11-21T00:00:00")[1] is None  # closed before opened
    assert X.funding_read(-1.0, 1.0, 50.0, "a", "b", 50.0)[0] == pytest.approx(-1.0)                  # the partial-close base


def _exh_fill_rows(n, funding):
    T0 = pd.Timestamp("2026-11-25")
    return pd.DataFrame([dict(opened_at=(T0 + pd.Timedelta(hours=5 * i)).strftime("%Y-%m-%dT%H:%M:%S"),
                              closed_at=(T0 + pd.Timedelta(hours=5 * i + 2)).strftime("%Y-%m-%dT%H:%M:%S"), pair=f"P{i}USDT", direction="SHORT",
                              status="CLOSED", entry_strategy="EXHAUSTION_SHORT", pnl_percentage=6.9, close_reason="FRENZY_TP", entry_price=100.0,
                              entry_exh_tp_pct=7.0, entry_exh_sl_pct=6.0, entry_exh_atr_pct=3.0, entry_exh_model_slip_pct=0.1, entry_depth_mode=None,
                              entry_depth_walk_pct=None, funding_fee_usd=funding, quantity=50.0) for i in range(n)]).assign(o=lambda d: pd.to_datetime(d.opened_at))


def test_exh_tp7_funding_never_changes_the_verdict(tmp_path):
    """💸 (613) the funding read is information only — same verdict / frozen record / state file whatever the funding column holds."""
    import json
    import re
    import config as C
    import scout_exh_tp7_609 as X
    th = NS(**dict(C.SignalThresholds().model_dump(), **_cfg()))
    since = int(pd.Timestamp("2026-11-24").value // 10**6)
    outs, states = [], []
    for j, fu in enumerate((-10.0, float("nan"), +25.0)):
        sp = tmp_path / f"s{j}.json"
        out = X.report(frame=_exh_fill_rows(24, fu), since_ms=since, state_path=str(sp), now_ms=since + 60 * 86_400_000,
                       bars_fn=lambda p, te, n: _bars(te, [(0.0, -7.2), (0.0, -10.2)]), newest=pd.Timestamp("2026-12-30"), th=th)
        outs.append([re.sub(r" · (funding|held) [^·]*?(?= ·|$)", "", x) for x in out if "💸" not in x])
        states.append(json.loads(sp.read_text())[X.KEY] if sp.exists() else None)
    assert outs[0] == outs[1] == outs[2] and states[0] == states[1] == states[2]
    assert any(x.startswith("**FROZEN") or x.startswith("Now: REVERT") or x.startswith("Now: KEEP") for x in outs[0])   # a verdict is reached


def test_exh_tp7_report_shows_funding(tmp_path):
    import config as C
    import scout_exh_tp7_609 as X
    th = NS(**dict(C.SignalThresholds().model_dump(), **_cfg()))
    T0 = pd.Timestamp("2026-11-25")
    f = pd.DataFrame([dict(opened_at="2026-11-25T00:00:00", closed_at="2026-11-25T02:00:00", pair="P0USDT", direction="SHORT", status="CLOSED",
                           entry_strategy="EXHAUSTION_SHORT", pnl_percentage=6.9, close_reason="FRENZY_TP", entry_price=100.0, entry_exh_tp_pct=7.0,
                           entry_exh_sl_pct=6.0, entry_exh_atr_pct=3.0, entry_exh_model_slip_pct=0.1, entry_depth_mode=None, entry_depth_walk_pct=None,
                           funding_fee_usd=-10.0, quantity=50.0)]).assign(o=lambda d: pd.to_datetime(d.opened_at))
    out = "\n".join(X.report(frame=f, since_ms=int(pd.Timestamp("2026-11-24").value // 10**6), state_path=str(tmp_path / "s.json"),
                             now_ms=int(pd.Timestamp("2026-12-30").value // 10**6), bars_fn=lambda p, te, n: _bars(te, [(0.0, -7.2), (0.0, -10.2)]),
                             newest=T0 + pd.Timedelta(days=9), th=th))
    assert "funding -0.20 %" in out and "held 2.0 h" in out and "avg -0.20 % of notional per fill" in out and "avg hold 2.0 h" in out


def test_exh_tp7_tp_old_and_gate():
    import scout_exh_tp7_609 as X
    assert X.tp_old(3.0) == 10.0 and X.tp_old(0.3) == 6.0 and X.tp_old(None) is None and X.tp_old(-1) is None
    assert X.gate([(d, 0.0) for d in range(19)])[0] == "COLLECTING"
    assert X.gate([(0, -1.0)] * 25)[0] == "COLLECTING"                                             # 25 fills on 1 day: waits for 5 days
    assert X.gate([(i // 4, -3.0) for i in range(20)])[0] == "REVERT"                              # shown worse
    assert X.gate([(i // 4, (-3.0 if i % 2 else 3.0)) for i in range(20)])[0] == "KEEP"            # a tie is never a revert


def _bars(te, path):
    """1m closed bars from the minute after te: path = list of (high, low) in % of 100."""
    m0 = te // 60_000 * 60_000 + 60_000
    return [[m0 + i * 60_000, 100.0, 100.0 * (1 + h / 100), 100.0 * (1 + l / 100), 100.0] for i, (h, l) in enumerate(path)]


def test_exh_tp7_report_prices_both_tps_and_reads_the_row_levels(tmp_path):
    import config as C
    import scout_exh_tp7_609 as X
    th = NS(**dict(C.SignalThresholds().model_dump(), **_cfg()))
    T0 = pd.Timestamp("2026-11-25")
    rows = []
    for i in range(3):
        rows.append(dict(opened_at=(T0 + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%S"), pair=f"P{i}USDT", direction="SHORT", status="CLOSED",
                         entry_strategy="EXHAUSTION_SHORT", pnl_percentage=6.9, close_reason="FRENZY_TP", entry_price=100.0, entry_exh_tp_pct=7.0,
                         entry_exh_sl_pct=6.0, entry_exh_atr_pct=3.0, entry_exh_model_slip_pct=0.1, entry_depth_mode=None, entry_depth_walk_pct=None))
    f = pd.DataFrame(rows).assign(o=lambda d: pd.to_datetime(d.opened_at))
    paths = {"P0USDT": [(0.0, -7.2), (0.0, -10.2)],          # TP 7 then TP 10 → Δ −3
             "P1USDT": [(0.0, -7.2), (6.5, 0.0)],            # TP 7, then the stop → TP 10 stops → Δ ≈ +13
             "P2USDT": [(6.5, 0.0)]}                          # stop first → both stop → Δ 0
    bars_fn = lambda pair, te, now: _bars(te, paths[pair])
    since = int(pd.Timestamp("2026-11-24").value // 10**6)
    out = "\n".join(X.report(frame=f, since_ms=since, state_path=str(tmp_path / "s.json"), now_ms=since + 40 * 86_400_000,
                             bars_fn=bars_fn, newest=T0 + pd.Timedelta(days=9), th=th))
    assert "P0USDT" in out and "TP 7 TP" in out and "TP 10 TP" in out and "Δ -3.00" in out
    assert "TP 10 STOP" in out and "Δ +0.00" in out and "COLLECTING" in out and "priced 3 on 1 days" in out


# ── registry · hooks · gitignore · UI ──────────────────────────────────────────────────────────────────────────────────────────────

def test_registry_hooks_gitignore():
    import observe_registry as OR
    assert OR.EXIT["scout_exh_tp7_609.py"]["walks"]["exhaustion_tp_cap_pct"] == 7.0 and OR.BOOKED["scout_ht_tp6_608.py"]["frenzy_willy_ht_tp_pct"] == 6.0
    rows = {r[0]: r for r in OR.status(_cfg())}
    for f in ("scout_exh_tp7_609.py", "scout_ht_tp6_608.py", "scout_exhaustion_short.py", "scout_exh_after_drop.py", "scout_exh_wide_585.py",
              "scout_exh_ssc_oivol.py", "scout_willy_ht_lines.py", "scout_willy_failed_spike_watch.py", "scout_blocked_filters.py"):
        assert rows[f][4] == "OK", (f, rows[f])
    assert "scout_exh_exit_override.py" in OR.NONE
    sc = _src("scripts/opportunity_scout.py")
    assert "_tp7.report(now_ms=now_ms)" in sc and "_ht6.report(now_ms=now_ms)" in sc and "_exo.report(" not in sc
    gi = _src(".gitignore").split()
    for f in ("reports/SCOUT_EXH_TP7_609.json", "reports/SCOUT_HT_TP6_608.json", "reports/SCOUT_EXH_SSC_OIVOL_609.json",
              "reports/SCOUT_EXH_AFTER_DROP_609.json", "reports/SCOUT_EXH_WIDE_585_609.json"):
        assert f in gi
    import scout_exh_after_drop as AD, scout_exh_wide_585 as WD, scout_exh_ssc_oivol as SO
    assert AD.BE_WR == 0.422 and AD.DL_EXIT == 609 and WD.DL_EXIT == 609 and SO.DL_EXIT == 609
    assert AD.STATE.endswith("SCOUT_EXH_AFTER_DROP_609.json") and WD.STATE.endswith("SCOUT_EXH_WIDE_585_609.json") and SO.STATE.endswith("SCOUT_EXH_SSC_OIVOL_609.json")


def test_dashboard_gates_read_the_tp7_era():
    m = _src("main.py")
    assert "exit_era_609 as _exera" in m and '" · counts DL 609 (TP 7) exit-era fills only"' in m
    assert "for a, p_, r, s_, dm_, wk_, tp_, at_ in (await db.execute(_wq)).all()" in m and "if _exera(tp_, at_, a)   # 🔁 (609) the WIDE gate" in m and "if _exera(tp_, at_, a)   # 🔁 (609) stamp + the era" in m


def test_open_orders_stop_line_ui_and_exports():
    h = _src("templates/index.html")
    assert "TP cap 7 % (DECISION_LOG 609" in h and "exit override DL 579 (TP ≈ 10 %" not in h
    assert "_exTp = Number(o.exit_tp_pct) > 0 ? Number(o.exit_tp_pct) : (Number(o.entry_exh_tp_pct) > 0 && Number(o.entry_exh_tp_pct) <= _exCap + 1e-9)" in h    # the badge shows the clamped level
    m = _src("main.py")
    i = m.index("_stop_at_pct = None")
    blk = m[i:i + 2200]
    assert "flatbase_stop_pct" in blk and "_exit_lv = _exh_rl(o, _th_sa)" in blk and "_exit_lv = _dist_rl(o, _th_sa)" in blk and "frenzy_willy_stop_pct" in blk


def test_closed_orders_funding_pct_same_base_as_pnl_pct():
    """💸 (DECISION_LOG 613) Closed Orders Funding % on the pnl_percentage base — entry × (quantity + partial_closed_qty)."""
    from services.paper_costs import funding_pct_of
    assert funding_pct_of(-87.38, 0.01563, 338877.0) == pytest.approx(-1.65, abs=0.005)        # the RARE 11-21 row: 7 % TP → 5.31 % booked
    assert funding_pct_of(-10.0, 100.0, 5.0, 5.0) == pytest.approx(-1.0)                       # partial closes widen the base (DL 574)
    assert funding_pct_of(0.0, 100.0, 1.0) == 0.0
    for bad in [(None, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, None, 1.0), (1.0, 1.0, 0.0), ("x", 1.0, 1.0), (float("nan"), 1.0, 1.0)]:
        assert funding_pct_of(*bad) is None
    assert funding_pct_of(-10.0, 100.0, 0.0, 5.0) == pytest.approx(-2.0)                       # fully reduced before the close: the partial base
    assert funding_pct_of(-10.0, 100.0, 5.0, float("nan")) == funding_pct_of(-10.0, 100.0, 5.0, -3.0) == pytest.approx(-2.0)


def test_closed_orders_funding_pct_is_the_funding_share_of_pnl_pct():
    """the engine's own fold path (fold_funding on entry × qty, then fold_partial re-bases on entry × (qty + pq)) moves pnl_percentage
    by exactly funding_pct_of — partial rows included."""
    from services.paper_costs import fold_funding, funding_pct_of
    from services.position_size_adopt import fold_partial
    E, q, pq, fu, raw = 2.0, 300.0, 200.0, -7.0, 40.0
    for pq_, net in ((0.0, 0.0), (pq, 12.0)):
        def final(fund):
            p, pc = fold_funding(raw, raw / (E * q) * 100, E * q, fund)
            p2, pc2 = fold_partial(p, q, E, net, pq_)
            return pc2 if pc2 is not None else pc
        assert final(fu) - final(0.0) == pytest.approx(funding_pct_of(fu, E, q, pq_), abs=0.006)


def test_closed_orders_funding_pct_wired():
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    m = (root / "main.py").read_text()
    assert ('"funding_pct": (_funding_pct_of(getattr(o, \'funding_fee_usd\', None), o.entry_price, o.quantity, '
            'getattr(o, \'partial_closed_qty\', None))') in m
    assert "if getattr(o, 'funding_in_pnl', None) else None)," in m      # older rows (funding not inside the P&L) → no %
    assert "Number(o.funding_pct).toFixed(2)" in (root / "templates" / "index.html").read_text()


def test_open_orders_send_the_rows_own_exh_dist_levels_615():
    """🎯 (DECISION_LOG 615) the Open Orders payload carries exit_tp_pct / exit_sl_pct = the server's row_levels (the badge reads these
    first) plus the raw EXH / DIST stamps; the JS clamp of the stamps to today's caps is only the fallback (ENSO 11-21: the badge showed
    the −6 cap while the engine's stop was −5.21)."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    m = (root / "main.py").read_text()
    i = m.index('@app.get("/api/orders/open")'); j = m.index("@app.", i + 10)
    for k in ("entry_exh_tp_pct", "entry_exh_sl_pct", "entry_dist_tp_pct", "entry_dist_sl_pct"):
        assert f'"{k}": getattr(o, \'{k}\', None)' in m[i:j]
    assert '"exit_tp_pct": (round(float(_exit_lv[0]), 2)' in m[i:j] and '"exit_sl_pct": (round(float(_exit_lv[1]), 2)' in m[i:j]
    assert "_exit_lv = _exh_rl(o, _th_sa)" in m[i:j] and "_exit_lv = _dist_rl(o, _th_sa)" in m[i:j]
    import services.exhaustion as SE, services.distribution as SD                   # the JS fallback keys = the engine's FIELDS
    assert SE.FIELDS['sl_cap'] == 'exhaustion_sl_cap_pct' and SE.FIELDS['tp_cap'] == 'exhaustion_tp_cap_pct'
    assert SD.FIELDS['sl_cap'] == 'distribution_sl_cap_pct' and SD.FIELDS['tp_cap'] == 'distribution_tp_cap_pct'
    h = (root / "templates" / "index.html").read_text()
    assert h.count("Number(o.exit_tp_pct) > 0 ? Number(o.exit_tp_pct)") == 2 and h.count("Number(o.exit_sl_pct) > 0 ? Number(o.exit_sl_pct)") == 2
    assert "Number(o.entry_exh_sl_pct) <= _exSlCap + 1e-9" in h
    assert "Number(o.entry_dist_tp_pct) <= _dsTpCap + 1e-9" in h and "Number(o.entry_dist_sl_pct) <= _dsSlCap + 1e-9" in h
