"""🅰️ Nov-4 (DECISION_LOG 464): the Alpha block per sleeve group — config fields (D11), the three engine call sites, the entry stamp,
the UI switches, and the ALPHA_SLEEVES watch line (scripts/scout_alpha_sleeves_watch.py). Hermetic: temp exports and state, no network."""
import json
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_alpha_sleeves_watch as W  # noqa: E402

FIELDS = ("alpha_filter_momentum", "alpha_filter_frenzy", "alpha_filter_spike")


def test_config_fields_alpha_to_frenzy_only():
    import config
    tc = config.TradingConfig()
    want = dict(alpha_filter_momentum=True, alpha_filter_frenzy=False, alpha_filter_spike=True)   # operator 11-04: Alpha to FRENZY only
    assert {f: getattr(tc, f) for f in FIELDS} == want and tc.alpha_subtype_filter_enabled is True
    j = json.load(open(os.path.join(ROOT, "trading_config.json")))
    assert {f: j[f] for f in FIELDS} == want
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert all(re.search(rf"^\s+{f}: Optional\[bool\] = None", main, re.M) for f in FIELDS)


def test_each_universe_reads_its_own_switch():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "alpha_subtype_filter_enabled=getattr(tc, 'alpha_filter_frenzy', False)" in src                        # FRENZY loop
    assert "alpha_subtype_filter_enabled=getattr(config.trading_config, 'alpha_filter_spike', True)" in src        # spike scanner
    assert "_alpha_filter = getattr(config.trading_config, 'alpha_filter_momentum', True)" in src                  # main scan
    assert src.count("'alpha_subtype_filter_enabled', True)") == 2                                                # the SURGE + FRENZY gvol reads only


def test_ui_switches_load_and_save():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for f in FIELDS:
        i = "config-" + f.replace("_", "-")
        assert html.count(i) == 3, i                                                    # input + load + save
        assert f"{f}: !!document.getElementById('{i}')?.checked" in html


def test_alpha_flag_and_stamp_class():
    from services import trading_engine as te
    assert te.binance_alpha_flag(["Alpha", "Crypto"]) is True and te.binance_alpha_flag("Alpha") is True
    assert te.binance_alpha_flag(["DeFi"]) is False and te.binance_alpha_flag(None) is None
    assert "entry_pair_alpha" in te.TradingEngine.STAMP_UNIVERSAL
    import models
    assert "entry_pair_alpha" in models.Order.__table__.columns
    assert "ADD COLUMN entry_pair_alpha BOOLEAN" in open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()


def test_group_and_flag_parsing():
    assert [W.group_of(s) for s in ("FRENZY_LITE", "FLATBASE_SHORT", "SPIKE_FADE", "SURGE_LONG", "MOMENTUM", "BEARRUN_SHORT")] == \
        ["alpha_filter_frenzy", "alpha_filter_frenzy", "alpha_filter_spike", "alpha_filter_momentum", "alpha_filter_momentum", "alpha_filter_momentum"]
    assert [W._flag(v) for v in (True, "True", 1, "0", False, float("nan"), None, "x")] == [True, True, True, False, False, None, None, None]


def _rows(spec, t0="2026-11-10 00:00"):
    out = []
    for i, (sleeve, alpha, pct, status) in enumerate(spec):
        o = pd.Timestamp(t0) + pd.Timedelta(hours=i)
        out.append(dict(opened_at=o.strftime("%Y-%m-%dT%H:%M:%S"), closed_at=(o + pd.Timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S"),
                        pair=f"P{i}USDT", direction="LONG", entry_strategy=sleeve, status=status, pnl_percentage=pct, pnl=pct * 10,
                        close_reason="TP" if pct > 0 else "STOP_LOSS", cell_multiplier_source=sleeve, entry_pair_alpha=alpha))
    return pd.DataFrame(rows := out)


def _write(tmp_path, df, stamp="2026-12-01_00-00-00"):
    p = tmp_path / f"scalpars_orders_paper_{stamp}.csv"
    df.to_csv(p, index=False)
    return [str(p)]


def test_breakeven_and_verdict():
    c = pd.DataFrame(dict(pct=[1.0] * 6 + [-1.0] * 4))
    assert abs(W.breakeven_wr(c) - 0.5) < 1e-9 and W.breakeven_wr(c.iloc[:9]) is None
    a = pd.DataFrame(dict(pct=[1.0] * 5 + [-1.0] * 10))
    assert W.verdict(a, c)[0] == W.REBLOCK                                              # WR 33 % < 50 %, Σ < 0
    assert W.verdict(pd.DataFrame(dict(pct=[1.0] * 10 + [-1.0] * 5)), c)[0] == "KEEP OPEN"
    assert W.verdict(a, c.iloc[:3])[0] == W.REBLOCK                                     # < 10 controls → Σ alone
    assert W.verdict(pd.DataFrame(dict(pct=[3.0] * 4 + [-0.5] * 11)), c)[0] == "KEEP OPEN"   # low WR, Σ > 0


def test_run_freezes_per_sleeve_once_and_defers_on_open(tmp_path):
    spec = [("FRENZY_WILLY", True, -1.0 if i % 3 else 1.0, "CLOSED") for i in range(15)] + \
           [("FRENZY_WILLY", False, 1.0 if i % 2 else -0.8, "CLOSED") for i in range(12)] + \
           [("FRENZY_LITE", True, 2.0, "CLOSED") for _ in range(5)] + [("MOMENTUM", None, 0.3, "CLOSED")]
    df = pd.concat([_rows(spec), _rows([("FRENZY_WILLY", True, -1.0, "CLOSED")], "2026-11-20 00:00")], ignore_index=True)
    df.loc[3, "status"] = "OPEN"                                                         # an early ALPHA fill still open
    st = str(tmp_path / "s.json")
    out = "\n".join(W.run(now_ms=0, paths=_write(tmp_path, df), state_path=st, since_ms=0))
    assert "freezing deferred" in out and not os.path.exists(st)
    df2 = df.copy(); df2.loc[3, "status"] = "CLOSED"
    out = "\n".join(W.run(now_ms=0, paths=_write(tmp_path, df2, "2026-12-02_00-00-00"), state_path=st, since_ms=0))
    rec = json.load(open(st))
    assert rec["FRENZY_WILLY"]["state"] == W.REBLOCK and rec["FRENZY_WILLY"]["switch"] == "alpha_filter_frenzy"
    assert "FRENZY_LITE" not in rec and "⏳ 5/15" in out and "| MOMENTUM |" in out
    first = rec["FRENZY_WILLY"]["run_at"]
    W.run(now_ms=10 ** 12, paths=_write(tmp_path, df2, "2026-12-02_00-00-00"), state_path=st, since_ms=0)
    assert json.load(open(st))["FRENZY_WILLY"]["run_at"] == first                      # never re-decided


def test_probes_and_pre_floor_fills_are_out(tmp_path):
    df = _rows([("FRENZY_LITE", True, 1.0, "CLOSED"), ("FRENZY_LITE", True, 1.0, "CLOSED")])
    df.loc[0, "cell_multiplier_source"] = "FRENZY_LITE_PROBE"
    d = W.load(_write(tmp_path, df))
    assert len(d) == 1
    out = "\n".join(W.run(now_ms=0, paths=_write(tmp_path, df), state_path=None,
                          since_ms=int(pd.Timestamp("2026-12-01").value // 1e6)))
    assert "| FRENZY_LITE |" not in out


def test_not_evaluable_and_hook_registry(monkeypatch):
    monkeypatch.setattr(W._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert any("NOT EVALUABLE" in x for x in W.run(state_path=None))
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_alpha_sleeves_watch as _alps" in src and "_alps.run(now_ms)" in src
    import observe_registry as R
    assert "scout_alpha_sleeves_watch.py" in R.BOOKED
    assert "reports/SCOUT_ALPHA_SLEEVES.json" in open(os.path.join(ROOT, ".gitignore")).read().split("\n")


def test_gvol_universe_keeps_the_calibrated_universe():
    from services import trading_engine as te
    m = {"A/USDT:USDT": {"info": {"underlyingSubType": ["Alpha"]}}, "B/USDT:USDT": {"info": {"underlyingSubType": ["DeFi"]}}}
    ps = [dict(symbol="A/USDT:USDT"), dict(symbol="B/USDT:USDT"), dict(symbol="C/USDT:USDT")]
    assert [p["symbol"] for p in te.gvol_universe(ps, m, True)] == ["B/USDT:USDT", "C/USDT:USDT"]
    assert len(te.gvol_universe(ps, m, False)) == 3
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert src.count("self._frenzy_gvol_start(gvol_universe(") == 1 and src.count("self._frenzy_gvol_catchup_start(gvol_universe(") == 1


def test_stamp_since_exempts_older_fills():
    from datetime import datetime
    from services import trading_engine as te
    E = te.TradingEngine
    full = {c: 1 for c in E.STAMP_UNIVERSAL}
    old = dict(full, entry_pair_alpha=None, opened_at=datetime(2026, 11, 1))
    new = dict(full, entry_pair_alpha=None, opened_at=datetime(2026, 11, 6))
    assert E.stamp_missing(E, "FRENZY_LITE", old) == [] and E.stamp_missing(E, "FRENZY_LITE", new) == ["entry_pair_alpha"]
    assert E.stamp_missing(E, "FRENZY_LITE", dict(full, entry_pair_alpha=None)) == ["entry_pair_alpha"]   # no open time → required


def test_frenzy_shortlist_never_loses_a_non_alpha_coin_to_an_alpha_one():
    from services import trading_engine as te
    m, allp = {}, []
    for i in range(30):                                                                 # 30 non-Alpha movers, 10 bigger Alpha movers
        allp.append(dict(symbol=f"N{i}/USDT:USDT", pair=f"N{i}USDT", change_24h=16 + i, range_24h=0, volume_24h=5e7))
        m[f"N{i}/USDT:USDT"] = {"info": {"underlyingSubType": ["DeFi"]}}
    for i in range(10):
        allp.append(dict(symbol=f"A{i}/USDT:USDT", pair=f"A{i}USDT", change_24h=90 + i, range_24h=0, volume_24h=5e7))
        m[f"A{i}/USDT:USDT"] = {"info": {"underlyingSubType": ["Alpha"]}}
    old = te.frenzy_shortlist([p for p in allp if p["pair"].startswith("N")], set(), 15.0, 2e7)
    new = te.frenzy_shortlist_split(allp, set(), 15.0, 2e7, m)
    assert new[:25] == old and sorted(new[25:]) == sorted(f"A{i}USDT" for i in range(10))
    assert te.frenzy_shortlist_split([p for p in allp if p["pair"].startswith("N")], set(), 15.0, 2e7, m) == old   # block ON → identical
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "names = frenzy_shortlist_split(allp, skip, chg, vmin, binance_service.public_exchange.markets or {})" in src
