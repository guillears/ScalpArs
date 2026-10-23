"""🎲² DECISION_LOG 356 (WILLY exit V2) — group-3 WILLY scouts: frozen gates keep their registered (pre-V2) exit, V2 runner / floor fills are
split out of old-exit comparisons (a V2 fill with target +1.25 stays in), and every scout prints willy_exit_era.era_line once.
Synthetic, hermetic: temp files only, every walker / fetch stubbed, sockets blocked — no network, no repo state written."""
import json
import os
import socket
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import willy_exit_era as WEra  # noqa: E402
import scout_willy_btc1h as WB  # noqa: E402
import scout_willy_ht as SHT  # noqa: E402
import scout_willy_nomiss as SNM  # noqa: E402
import scout_willy_hold_off as SHO  # noqa: E402
import scout_willy_hold as WH  # noqa: E402
import scout_willy_hold_frenzy as SHF  # noqa: E402
import scout_willy_tp125 as STP  # noqa: E402
import scout_willy_stop as SST  # noqa: E402
import scout_willy_avgdown as SAD  # noqa: E402
import scout_willy_mktdip as SMD  # noqa: E402
import scout_willy_timecap as WT  # noqa: E402

ERA = "🎲² exit era ("
TP, ARM, HIGH = WEra.TP_COL, WEra.ARM_COL, WEra.HIGH_COL


@pytest.fixture(autouse=True)
def _no_net(monkeypatch):
    def _boom(*a, **k):
        raise OSError("network blocked (test)")
    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)


def _order(o, pair, es="FRENZY_WILLY", pct=1.25, why="FRENZY_TP", tp=np.nan, c=None, usd=10.0, paper=False, **kw):
    return dict(dict(opened_at=o, closed_at=c or o, pair=pair, direction="LONG", status="CLOSED", entry_strategy=es, entry_price=100.0,
                     exit_price=101.0, notional_value=1000.0, pnl=usd, pnl_percentage=pct, close_reason=why, is_paper=paper,
                     entry_frenzy_willy_trigger="A", **{TP: tp, HIGH: (np.nan if tp != tp else 105.0), ARM: np.nan}), **kw)


# the three exit eras every test uses: pre-V2 · V2 at the old target (+1.25 → old exit exactly) · V2 runner (+3.5, floor / TP +3.5)
def _three(day="2026-10-24", es="FRENZY_WILLY"):
    return [_order(f"{day}T01:00:00", "OLDUSDT", es=es, pct=1.25),
            _order(f"{day}T02:00:00", "V125USDT", es=es, pct=1.25, tp=1.25),
            _order(f"{day}T03:00:00", "RUNUSDT", es=es, pct=1.0, why="WILLY_FLOOR", tp=3.5)]


def _csv(tmp_path, rows, name="scalpars_orders_paper_x.csv"):
    p = tmp_path / name
    pd.DataFrame(rows).to_csv(p, index=False)
    return str(p)


# ─────────────────────────── WILLY_BTC1H (DL 275 / 279, frozen) ───────────────────────────
def test_btc1h_exit_pinned_and_config_change_does_not_move_it(tmp_path, monkeypatch):
    th = {"frenzy_willy_tp_pct": 2.0, "frenzy_willy_stop_pct": 2.5, "frenzy_willy_max_hold_minutes": 60}
    (tmp_path / "trading_config.json").write_text(json.dumps({"thresholds": th}))
    monkeypatch.setattr(WH, "ROOT", str(tmp_path))                     # the old reader now sees the hypothetical config …
    assert WH.exit_rule("FRENZY_WILLY")[0] == 2.0                       # … and would have moved the walk
    assert WB.willy_rule() == WB.willy_rule(th) == WB.PINNED_RULE == (1.25, 4.0, 90)   # pinned: it never does


def _btc1h_run(tmp_path, monkeypatch, rows):
    seen = []

    def fake_price(r, rule, tcache=None):
        seen.append(rule)
        runner = WEra.is_v2_changed({TP: getattr(r, TP), "close_reason": r.why})
        return dict(walk=1.25, walk_why="TP", walk_min=5.0, src="ticks", par_ok=not runner)   # a runner can never match the old exit
    monkeypatch.setattr(WB, "price", fake_price)
    monkeypatch.setattr(WB, "pending", lambda *a, **k: False)
    monkeypatch.setattr(WB, "load_btc5m", lambda: pd.DataFrame({"open_time": pd.Series(dtype="int64"), "c": pd.Series(dtype=float)}))
    monkeypatch.setattr(WT, "load_attempts", lambda: {"done": set()})
    monkeypatch.setattr(WT, "ticks_pending", lambda *a, **k: False)
    now = int(pd.Timestamp("2026-10-26").value // 10**6)
    out = "\n".join(WB.run(now, orders=pd.DataFrame(rows), fetch=False, state_path=str(tmp_path / "s.json"),
                           rows_path=str(tmp_path / "r.csv"), floor="2026-10-01", cstart="2026-10-01"))
    return out, seen


def test_btc1h_walk_pinned_era_line_and_runner_out_of_parity(tmp_path, monkeypatch):
    (tmp_path / "trading_config.json").write_text(json.dumps({"thresholds": {"frenzy_willy_tp_pct": 2.0}}))
    monkeypatch.setattr(WH, "ROOT", str(tmp_path))
    out, seen = _btc1h_run(tmp_path, monkeypatch, _three())
    assert seen and set(seen) == {(1.25, 4.0, 90)}, seen                 # every fill walked under the pinned exit, config 2.0 ignored
    assert out.count(ERA + "WILLY_BTC1H)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out
    par = [l for l in out.split("\n") if l.startswith("Walker parity")][0]
    assert ": 2/2" in par and "1 V2 runner / floor fill(s) out" in par and "RUNUSDT target +3.5 WILLY_FLOOR" in par, par
    assert "V125USDT" not in par.split("out")[-1]                        # the +1.25 V2 fill stays a parity case
    assert "+1.25 / −4 / 90 min, pinned — DECISION_LOG 356" in out


def test_btc1h_reader_asks_for_v2_columns():
    assert set(WEra.V2_COLS) <= set(WB.COLS) and "close_reason" in WB.COLS


# ─────────────────────────── WILLY_HT (DL 341) ───────────────────────────
def test_ht_runner_kept_out_of_both_sides_and_era_line(tmp_path):
    rows = _three(es="FRENZY_WILLY_HT") + _three(day="2026-10-25")
    p = _csv(tmp_path, rows)
    since = pd.Timestamp("2026-10-23")
    ht, w, _ = SHT.cohorts(SHO.load([p]), since)
    assert sorted(ht.pair) == ["OLDUSDT", "V125USDT"] and sorted(w.pair) == ["OLDUSDT", "V125USDT"]
    out = "\n".join(SHT.run(0, paths=[p], since_ms=int(since.value // 10**6)))
    assert out.count(ERA + "WILLY_HT)") == 1 and "pre-V2 2 · V2 4 (runner target / floor 2)" in out
    assert "kept OUT of both sides" in out and out.count("RUNUSDT") == 2
    assert "FRENZY_WILLY_HT (¼): N 2" in out and "FRENZY_WILLY (full): N 2" in out


# ─────────────────────────── WILLY_NOMISS (DL 350) ───────────────────────────
def test_nomiss_runner_kept_out_of_both_sides_and_era_line(tmp_path):
    rows = [dict(r, entry_frenzy_willy_nomiss=True) for r in _three()] + _three(day="2026-10-25")
    p = _csv(tmp_path, rows)
    since = pd.Timestamp("2026-10-23")
    nm, normal, _ = SNM.cohorts(SNM.load([p]), since)
    assert sorted(nm.pair) == ["OLDUSDT", "V125USDT"] and sorted(normal.pair) == ["OLDUSDT", "V125USDT"]
    out = "\n".join(SNM.run(0, paths=[p], dec_paths=[], since_ms=int(since.value // 10**6)))
    assert out.count(ERA + "WILLY_NOMISS)") == 1 and "pre-V2 2 · V2 4 (runner target / floor 2)" in out
    assert "kept OUT of both sides" in out and "RUNUSDT R_NOMISS +1.00 %" in out
    assert set(WEra.V2_COLS) <= set(SNM.COLS)


# ─────────────────────────── WILLY_HOLD_OFF (DL 340) ───────────────────────────
def test_hold_off_era_line_and_reader_columns(tmp_path):
    p = _csv(tmp_path, _three())
    out = "\n".join(SHO.run(int(pd.Timestamp("2026-10-26").value // 10**6), paths=[p], since_ms=int(pd.Timestamp("2026-10-23").value // 10**6),
                            state_path=str(tmp_path / "st.json")))
    assert out.count(ERA + "WILLY_HOLD_OFF)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out
    assert set(WEra.V2_COLS) <= set(SHO.COLS) and TP in SHO.load([p]).columns


# ─────────────────────────── WILLY_TP125 (DL 265) / WILLY_STOP (DL 268) — pinned exit, runner out ───────────────────────────
def _shadow_env(monkeypatch, tmp_path):
    monkeypatch.setattr(WT, "levels", lambda th=None: (2.0, 2.5, 60))     # a hypothetical config: must not reach a production run
    monkeypatch.setattr(WT, "deploy_ts", lambda: pd.Timestamp("2026-10-01"))
    monkeypatch.setattr(WT, "load_attempts", lambda: {"done": set()})
    monkeypatch.setattr(WT, "ticks_pending", lambda *a, **k: False)
    monkeypatch.setattr(STP, "score_pair", lambda *a, **k: (None, None))
    monkeypatch.setattr(SST, "score_fill", lambda *a, **k: (None, None))


def test_tp125_exit_pinned_runner_excluded_target125_kept(tmp_path, monkeypatch):
    _shadow_env(monkeypatch, tmp_path)
    out = "\n".join(STP.run(int(pd.Timestamp("2026-10-26").value // 10**6), orders=pd.DataFrame(_three()),
                            state_path=str(tmp_path / "tp.json"), study=False))
    assert "today's config takes +1.25" in out and "+2" not in out.split("\n")[0]       # pinned +1.25, never the patched config
    assert out.count(ERA + "WILLY_TP125)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out
    ex = [l for l in out.split("\n") if "V2 runner / floor fill(s)" in l][0]
    assert ex.startswith("Excluded: 1 V2") and "RUNUSDT target +3.5 WILLY_FLOOR" in ex and "V125USDT" not in ex


def test_tp125_v2_split_pure():
    o = WT._prep(pd.DataFrame(_three() + [_order("2026-10-24T04:00:00", "TPUSDT", pct=3.5, tp=3.5)]))
    keep, out = STP._v2_split(o)
    assert sorted(keep.pair) == ["OLDUSDT", "V125USDT"] and sorted(out.pair) == ["RUNUSDT", "TPUSDT"]   # FRENZY_TP +3.5 runner out too


def test_stop_exit_pinned_runner_excluded_target125_kept(tmp_path, monkeypatch):
    _shadow_env(monkeypatch, tmp_path)
    rows = _three() + [_order("2026-10-24T04:00:00", "TPUSDT", pct=3.5, tp=3.5)]
    out = "\n".join(SST.run(int(pd.Timestamp("2026-10-26").value // 10**6), orders=pd.DataFrame(rows), state_path=str(tmp_path / "st.json")))
    assert "TP +1.25 and the 90-min cap" in out and "a −4 stop" in out                   # pinned (+1.25 / −4 / 90), not 2.0 / 2.5 / 60
    assert out.count(ERA + "WILLY_STOP)") == 1 and "pre-V2 1 · V2 3 (runner target / floor 2)" in out
    ex = [l for l in out.split("\n") if "V2 runner / floor fill(s)" in l][0]
    assert ex.startswith("Excluded: 2 V2") and "RUNUSDT" in ex and "TPUSDT target +3.5 FRENZY_TP +3.50" in ex and "V125USDT" not in ex
    uns = [l for l in out.split("\n") if l.startswith("UNSCORED")][0]
    assert "V125USDT" in uns and "OLDUSDT" in uns and "RUNUSDT" not in uns


def test_shadow_harness_th_still_overrides():
    # an explicit th is the selftest harness (unchanged); None = the pinned production path — both selftests still pass
    assert STP.selftest() is None and SST.selftest() is None


# ─────────────────────────── WILLY_AVGDOWN (DL 313) — runner out of the parity read ───────────────────────────
def test_avgdown_runner_out_of_parity(monkeypatch, capsys):
    rows = [_order("2026-10-24T01:00:00", "OLDUSDT", pct=1.25), _order("2026-10-24T02:00:00", "V125USDT", pct=1.25, tp=1.25),
            _order("2026-10-24T03:00:00", "RUNUSDT", pct=3.5, tp=3.5)]
    monkeypatch.setattr(SAD, "fills", lambda: pd.DataFrame(rows))
    monkeypatch.setattr(SAD, "kl", lambda *a, **k: SAD._EMPTY_K.copy())
    monkeypatch.setattr(SAD, "path_of", lambda *a, **k: {})
    # walker base −1 on every fill: the old-exit fills disagree in sign with their booked +1.25 (flagged); the runner would too, but is out
    monkeypatch.setattr(SAD, "shadow", lambda P, te: dict(base=-1.0, base_why="CAP", merged=-1.0, merged_why="CAP", trig=False,
                                                          add_min=np.nan, leg=np.nan))
    SAD.main()
    out = capsys.readouterr().out
    assert out.count(ERA + "WILLY_AVGDOWN)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out
    assert "sign disagreements 2/2 · 1 V2 runner / floor fill(s) out" in out and "RUNUSDT" not in out


# ─────────────────────────── WILLY_MKTDIP (DL 305) — booked avg on old-exit fills only ───────────────────────────
def test_mktdip_booked_avg_excludes_runner(monkeypatch, capsys):
    rows = [_order("2026-10-24T01:00:00", "OLDUSDT", pct=1.0, entry_ema_gap_8_13=2.0),
            _order("2026-10-24T02:00:00", "V125USDT", pct=1.5, tp=1.25, entry_ema_gap_8_13=2.0),
            _order("2026-10-24T03:00:00", "RUNUSDT", pct=8.0, tp=8.0, entry_ema_gap_8_13=2.0)]
    monkeypatch.setattr(SMD, "fills", lambda: pd.DataFrame(rows))
    monkeypatch.setattr(SMD, "breadth", lambda te, bud: 0.1)
    monkeypatch.setattr(SMD, "path", lambda *a, **k: (True, True))
    SMD.main()
    out = capsys.readouterr().out
    assert out.count(ERA + "WILLY_MKTDIP)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out
    assert "avg booked +1.25 % (old-exit fills; 1 V2 runner / floor out)" in out              # (1.0 + 1.5) / 2, the +8 runner out
    assert "ZONE      fills   3" in out                                                       # the gate (touch / +2 on the path) keeps all


# ─────────────────────────── WILLY_HOLD (DL 251) / HOLD_FRENZY_EXEMPT (DL 266) — era line ───────────────────────────
def test_willy_hold_era_line(tmp_path, monkeypatch):
    monkeypatch.setattr(WH, "STORE", str(tmp_path / "store.csv"))
    monkeypatch.setattr(WH, "floor_ms", lambda: 0)
    monkeypatch.setattr(WH, "export_rows", lambda: pd.DataFrame(columns=WH.COLS))
    monkeypatch.setattr(WH, "orders", lambda: pd.DataFrame(_three()))
    out = "\n".join(WH.run(int(pd.Timestamp("2026-10-26").value // 10**6)))
    assert out.count(ERA + "WILLY_HOLD)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out and "BOOKED money" in out
    assert not os.path.exists(tmp_path / "store.csv")


def test_hold_frenzy_era_line_and_reader(tmp_path, monkeypatch):
    p = _csv(tmp_path, _three())
    monkeypatch.setattr(SHF, "EXPORT_GLOB", os.path.join(str(tmp_path), "scalpars_orders_paper_*.csv"))
    monkeypatch.setattr(SHF._EG, "glob_exports", lambda pat: [p])
    o = SHF.load_orders()
    assert TP in o.columns and "close_reason" in o.columns
    dec = pd.DataFrame(columns=["t", "e", "pair", "gate"])
    dec.attrs["newest_ms"] = 0
    out = "\n".join(SHF.run(int(pd.Timestamp("2026-10-26").value // 10**6), rows=pd.DataFrame(columns=WH.COLS), orders=o,
                            state_path=str(tmp_path / "hf.json"), ctx={"dec": dec, "m5": None, "d1": None}))
    assert out.count(ERA + "HOLD_FRENZY_EXEMPT)") == 1 and "pre-V2 1 · V2 2 (runner target / floor 1)" in out
