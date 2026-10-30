"""🎲 tests for the master WILLY / WILLY_HT V2b re-price (DECISION_LOG 377, STACK 2026-10-26a): scripts/build_willy_exit_cf.py (V2b CF on
synthetic paths: the target from the spike high H, late TP, stop, cap, no floor; the H window / rebuild; the no-H rule) and its use by
scripts/build_master_pool.py. Synthetic data + the committed reports; no network."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(1, ROOT)
import build_willy_exit_cf as C  # noqa: E402
import scout_willy_exit_v2 as S  # noqa: E402

TE0 = 1_761_000_000_000
MIN = 60_000
TH = dict(frenzy_willy_tp_pct=1.25, frenzy_willy_stop_pct=4.0, frenzy_willy_max_hold_minutes=90, frenzy_willy_late_tp_minutes=30.0,
          frenzy_willy_late_tp_pct=0.25, resting_tp_willy_enabled=True, frenzy_willy_high_target_enabled=True, frenzy_willy_high_k_pct=1.0,
          frenzy_willy_high_tp_max_pct=8.0, _maker_fee=0.00018)


def _P(ev, bar=False):
    return S._P(TE0, ev, cap=90, bar=bar)


def _row(H, e=0.1, q=10_000.0, fee_pct=0.045, **kw):
    r = dict(entry_price=e, quantity=q, entry_fee=e * q * fee_pct / 100, entry_frenzy_willy_spike_high=H, pair="TESTUSDT",
             opened_at="2026-10-23T16:30:09", entry_frenzy_spike_at="2026-10-23T16:05:00", pnl_percentage=1.0, entry_strategy="FRENZY_WILLY")
    r.update(kw)
    return r


def test_rule_strings():
    assert C.rule_pre356(TH) == "+1.25/-4/90/late30@+0.25/rtp"                      # the STACK ≤ 2026-10-25a rule, reproducible
    assert C.rule_v2b(TH) == "V2b:T=max(1.25,min(8,net@H*0.99))/-4/90/late30@+0.25/rtp/nofloor"
    assert C.rule_v2b(dict(TH, frenzy_willy_high_target_enabled=False)) == C.rule_pre356(TH)   # switch off = the old exit exactly


def test_target_from_h_is_the_engine_function():
    from services.exec_shadow import net_pct_at
    r = _row(0.106)                                                                  # H 6 % above the entry
    tu, traw = C.target_from_h(TH, r, 0.106)
    assert traw == round(net_pct_at("LONG", 0.1, 0.106 * 0.99, 0.045, 0.00018), 4) and tu == round(max(1.25, min(8.0, traw)), 4)
    assert C.target_from_h(TH, r, 0.1005)[0] == 1.25                                 # H barely above → the base +1.25
    assert C.target_from_h(TH, r, 0.2)[0] == 8.0                                     # capped at +8
    assert abs(C.entry_fee_pct(r) - 0.045) < 1e-12 and C.entry_fee_pct(dict(r, quantity=0)) is None


def test_v2b_books_the_target_from_h():
    T = C.target_from_h(TH, _row(0.106), 0.106)[0]
    assert 4.5 < T < 5.0
    up = _P([(2, 1.3, 1.3), (6, 3.0, 2.0), (11, T + 0.4, T - 0.2)])
    w = C.walk_v2b(up, T, TH)
    assert w["why"] == "TP" and w["pct"] == T and abs(w["xmin"] - 11) < 1e-9       # resting maker TP books exactly T
    assert C.walk_pre356(up, TH)["why"] == "TP" and C.walk_pre356(up, TH)["pct"] == 1.25   # the old exit takes +1.25 on the first event


def test_v2b_late_tp_for_every_trade_and_stop_without_floor():
    T = 4.0
    late = _P([(2, 2.85, 2.85), (20, 1.0, 1.0), (31, 0.6, 0.3), (40, 0.1, 0.1)])
    w = C.walk_v2b(late, T, TH)
    assert w["why"] == "LATE_TP" and abs(w["pct"] - 0.6) < 1e-9 and abs(w["xmin"] - 31) < 1e-9   # first tick past 30 min books its print
    # INIT shape: peak +2.85 then −4.1 — the removed V2 floor (+1.0) would have closed it; V2b rides to the −4 stop
    fall = _P([(2, 1.3, 1.3), (5, 2.85, 2.85), (8, 1.0, 1.0), (12, -4.1, -4.1)])
    w = C.walk_v2b(fall, 7.42, TH)
    assert w["why"] == "STOP" and w["pct"] == -4.1 and abs(w["xmin"] - 12) < 1e-9
    assert S.walk_v2(fall, 7.42, floor=1.0)["why"] == "FLOOR"                       # (the floor shadow differs — proof the CF has none)


def test_v2b_cap_and_unres():
    flat = _P([(m, 0.1, -0.1) for m in (5, 20, 29)] + [(m, 0.2, 0.1) for m in (35, 60, 89.5)])
    w = C.walk_v2b(flat, 3.0, TH)
    assert w["why"] == "CAP" and w["xmin"] == 90.0 and abs(w["pct"] - 0.15) < 1e-9   # last event ≤ the cap
    short = dict(_P([(5, 0.5, 0.4)]), end_ms=TE0 + 10 * MIN)
    assert C.walk_v2b(short, 3.0, TH)["why"] == "UNRES"


def test_v2b_equals_old_exit_at_the_base_target():
    rng = np.random.default_rng(11)
    for k in range(150):
        mins = np.sort(rng.uniform(0.1, 95, 40))
        x = np.cumsum(rng.normal(0, 0.6, 40))
        P = _P([(m, v, v) for m, v in zip(mins, x)])
        a, b = C.walk_v2b(P, 1.25, TH), C.walk_pre356(P, TH)
        assert a["why"] == b["why"], (k, a, b)
        assert (np.isnan(a["pct"]) and np.isnan(b["pct"])) or abs(a["pct"] - b["pct"]) < 1e-9
        if a["why"] not in ("UNRES",):
            assert abs(a["xmin"] - b["xmin"]) < 1e-6, (k, a, b)


def test_h_window_is_frenzy_walks_peak_window():
    sp = int(pd.Timestamp("2026-10-23 16:05:00").value // 1_000_000)               # spike bar CLOSE time (frenzy_walk spike_ts)
    te = int(pd.Timestamp("2026-10-23 16:30:09").value // 1_000_000)
    b0, last = C.h_window(sp, te)
    assert pd.Timestamp(b0, unit="ms") == pd.Timestamp("2026-10-23 16:00") and pd.Timestamp(last, unit="ms") == pd.Timestamp("2026-10-23 16:25")
    assert C.h_window(te, te) is None                                                # spike after the last closed bar → no window
    te_b = int(pd.Timestamp("2026-10-23 16:30:00").value // 1_000_000)              # fill exactly on a bar boundary: the 16:30 bar is not closed
    assert pd.Timestamp(C.h_window(sp, te_b)[1], unit="ms") == pd.Timestamp("2026-10-23 16:25")


def test_reconstruct_high_from_5m_cache(monkeypatch):
    sp = int(pd.Timestamp("2026-10-23 16:05:00").value // 1_000_000)
    te = int(pd.Timestamp("2026-10-23 16:30:09").value // 1_000_000)
    o = np.arange(sp - 10 * 300_000, sp + 10 * 300_000, 300_000, dtype="int64")
    h = np.where((o >= sp - 300_000) & (o <= sp + 4 * 300_000), 1.0, 0.5)
    h[o == sp + 300_000] = 1.7                                                       # inside the window
    h[o == sp + 6 * 300_000] = 9.9                                                   # the bar AFTER the fill — never in H
    h[o == sp - 2 * 300_000] = 5.0                                                   # before the spike bar — never in H
    monkeypatch.setitem(C._K5, "ZZZUSDT", pd.DataFrame(dict(open_time=o, h=h)))
    assert C.reconstruct_high("ZZZUSDT", sp, te) == (1.7, "5m")
    gap = pd.DataFrame(dict(open_time=o[o != sp + 300_000], h=h[o != sp + 300_000]))  # a missing bar → never trusted as 5m
    monkeypatch.setitem(C._K5, "ZZZUSDT", gap)
    monkeypatch.setattr(C.FX.WT, "load_1m", lambda pair: pd.DataFrame(dict(open_time=np.array([], "int64"), h=np.array([], float))))
    monkeypatch.setattr(C.FX.WT, "_load_day", lambda pair, d, cache: None)
    assert C.reconstruct_high("ZZZUSDT", sp, te) == (None, "none")


def test_resolve_target_prefers_the_stamped_target():
    r = _row(0.106, frenzy_willy_tp_target_pct=3.3)
    g = C.resolve_target(TH, r)
    assert g["T"] == 3.3 and g["t_src"] == "stamp" and g["h_src"] == "stamp" and g["T_calc"] == C.target_from_h(TH, r, 0.106)[0]
    g = C.resolve_target(TH, _row(0.106))
    assert g["t_src"] == "rebuilt" and g["T"] == C.target_from_h(TH, _row(0.106), 0.106)[0]


def test_no_h_is_priced_only_when_the_exit_is_target_invariant(monkeypatch):
    monkeypatch.setattr(C, "reconstruct_high", lambda *a, **k: (None, "none"))
    stop = _P([(2, 0.5, 0.5), (6, -4.2, -4.2)])
    tp = _P([(2, 1.3, 1.3), (6, -4.2, -4.2)])
    late = _P([(2, 0.5, 0.5), (20, 1.0, 1.0), (31, 0.6, 0.3), (40, 0.1, 0.1)])          # never touches +1.25 → late TP at any T ≥ 1.25
    cap = _P([(m, 0.1, -0.1) for m in (5, 20, 29)] + [(m, 0.2, 0.1) for m in (35, 60, 89.5)])
    for P, why, pct in ((stop, "STOP", -4.2), (late, "LATE_TP", 0.6), (cap, "CAP", 0.15), (tp, "NO_H", None)):
        P = dict(P, n_bar=0, n_tick=len(P["t"]))
        monkeypatch.setattr(C.FX, "load_path", lambda *a, _P=P, **k: _P)
        out = C.price_row(_row(float("nan"), opened_at=pd.Timestamp(TE0, unit="ms").isoformat()), TH, {}, {}, C.rule_v2b(TH))
        assert out["cf_why"] == why and (abs(out["cf_pct"] - pct) < 1e-9 if pct is not None else out["cf_pct"] is None)
        assert out["t_src"] == ("invariant" if why != "NO_H" else "none")


# ─────────────────────────── the committed master uses it ───────────────────────────
def test_master_prices_willy_and_ht_from_the_v2b_cf():
    import build_master_pool as B
    src = open(os.path.join(ROOT, "scripts", "build_master_pool.py")).read()
    assert B.STACK_VERSION == "2026-10-30a"
    assert f'WILLY_EXIT_FROZEN = "{C.rule_v2b(C.load_th())}"' in src                 # the frozen rule = today's config rule
    assert 'WILLY_CF_STRATS = ("FRENZY_WILLY", "FRENZY_WILLY_HT")' in src and 'strat == "FRENZY_WILLY_HT" and bool(r.stack_keep)' in src
    cf = pd.read_csv(os.path.join(ROOT, "reports", "WILLY_EXIT_CF.csv"))
    assert cf.exit_rule.eq(C.rule_v2b(C.load_th())).all() and set(cf.entry_strategy) <= {"FRENZY_WILLY", "FRENZY_WILLY_HT"}
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    k = M[(M.stack_keep == True) & M.entry_strategy.astype(str).isin(["FRENZY_WILLY", "FRENZY_WILLY_HT"])]
    m = {(a, p): c for a, p, c in zip(cf.opened_at, cf.pair, cf.cf_pct) if pd.notna(c)}
    assert len(k) >= 38 and all(abs(r.stack_pct - m[(str(r.opened_at)[:19], r.pair)]) < 1e-9 for r in k.itertuples())
    assert (M.stack_version == "2026-10-30a").all()


def test_pre356_cf_file_reproduces_the_old_stack():
    p = os.path.join(ROOT, "reports", "WILLY_EXIT_CF_PRE356.csv")
    if not os.path.exists(p):
        pytest.skip("pre-356 table not built")
    old = pd.read_csv(p)
    cf = pd.read_csv(os.path.join(ROOT, "reports", "WILLY_EXIT_CF.csv"))
    assert old.exit_rule.eq("+1.25/-4/90/late30@+0.25/rtp").all()
    j = cf.merge(old, on=["opened_at", "pair"], suffixes=("", "_old"))
    assert len(j) == len(old) and np.allclose(j.pre356_pct, j.cf_pct_old, equal_nan=True)   # the V2b table's "before" column = the old CF
