"""🅰️ Nov-4 (DECISION_LOG 464): the Alpha block per sleeve group — config fields (D11), the three engine call sites, the entry stamp,
the UI switches, and the ALPHA_SLEEVES watch line (scripts/scout_alpha_sleeves_watch.py). Hermetic: temp exports and state, no network."""
import json
import os
import re
import sys

import numpy as np
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
    assert "import scout_alpha_sleeves_watch as _alps" in src and "_alps.run(now_ms, market=True)" in src
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
    assert "_sl_b, _sl_a, _q_b, _q_a = frenzy_shortlist_parts(allp, skip, chg, vmin, binance_service.public_exchange.markets or {})" in src


def test_alpha_shortlist_never_crowds_out_a_followed_coin():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "names = frenzy_read_order(_sl_b, _sl_a, _follow, _frenzy_flags, by, skip," in src
    from services import trading_engine as te
    m = {"A/USDT:USDT": {"info": {"underlyingSubType": ["Alpha"]}}}
    ps = [dict(symbol="A/USDT:USDT", pair="AUSDT", change_24h=50, range_24h=0, volume_24h=5e7),
          dict(symbol="B/USDT:USDT", pair="BUSDT", change_24h=20, range_24h=0, volume_24h=5e7)]
    assert te.frenzy_shortlist_split(ps, set(), 15.0, 2e7, m) == ["BUSDT", "AUSDT"]
    assert te.frenzy_shortlist_split(ps, set(), 15.0, 2e7, m, alpha_cap=0) == ["BUSDT"]


def test_shortlist_parts_count_qualifying_before_the_cap():
    from services import trading_engine as te
    m, allp = {}, []
    for i in range(30):
        allp.append(dict(symbol=f"N{i}/USDT:USDT", pair=f"N{i}USDT", change_24h=16 + i, range_24h=0, volume_24h=5e7))
    for i in range(3):
        allp.append(dict(symbol=f"A{i}/USDT:USDT", pair=f"A{i}USDT", change_24h=40, range_24h=0, volume_24h=5e7))
        m[f"A{i}/USDT:USDT"] = {"info": {"underlyingSubType": ["Alpha"]}}
    allp.append(dict(symbol="L/USDT:USDT", pair="LUSDT", change_24h=40, range_24h=0, volume_24h=1e6))   # too thin → never qualifies
    b, a, qb, qa = te.frenzy_shortlist_parts(allp, set(), 15.0, 2e7, m)
    assert (len(b), len(a), qb, qa) == (25, 3, 30, 3)
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "[FRENZY_CAP] shortlist cap cut coins" in src


def _pre464_read(sl, follow, flags, by, skip):
    """the follow loop exactly as it was before DL 464"""
    names = list(sl)
    for k in follow:
        if k in by and k.upper() not in skip and k not in names and (len(names) < 40 or k in flags):
            names.append(k)
    return names


def test_read_order_non_alpha_part_identical_to_before_464():
    from services import trading_engine as te
    import random
    rnd = random.Random(7)
    for _ in range(200):
        base = [f"N{i}USDT" for i in range(60)]
        alpha = {f"A{i}USDT" for i in range(30)}
        by = {k: {} for k in base + sorted(alpha)}
        sl_b = rnd.sample(base, rnd.randint(0, 25))
        sl_a = rnd.sample(sorted(alpha), rnd.randint(0, 25))
        follow = rnd.sample(base + sorted(alpha), rnd.randint(0, 40)) + rnd.sample(base, 3)
        flags = set(rnd.sample(base + sorted(alpha), 5))
        skip = {"N1USDT"}
        got = te.frenzy_read_order(sl_b, sl_a, follow, flags, by, skip, lambda k: k in alpha)
        want_base = _pre464_read(sl_b, [k for k in follow if k not in alpha], flags, by, skip)
        assert got[:len(want_base)] == want_base                                              # non-Alpha part: same coins, same order
        assert all(k in alpha for k in got[len(want_base):])                                 # Alpha only after it
        assert [k for k in got if k in alpha][:len(sl_a)] == sl_a                            # Alpha shortlist before followed Alpha
        assert len(got) == len(set(got))
    assert te.frenzy_read_order(["A1"], [], ["A2"], set(), {"A1": {}, "A2": {}}, set(), lambda k: False) == ["A1", "A2"]


# ── 🧭 (DECISION_LOG 471) Alpha or another watch line? — the shared zones and the confound note
def test_watch_zones_unscored_market_lines_are_none_and_shared_by_best_case():
    import watch_zones as WZ
    d = pd.DataFrame({"entry_adx": [60.0, 30.0, np.nan], "entry_ema_gap_8_13": [2.0, 0.5, np.nan]})
    z = WZ.zones(d)
    assert z["FRENZY_WILLY"]["MKT_HOT"] is None and z["MOMENTUM_LONG"]["STOP_COOLDOWN"] is None
    assert z["FRENZY_WILLY"]["WILLY_GAP"].tolist() == [True, True, False]          # NaN inputs → never in the zone
    assert WZ.zones(d, hot=["zone", "rest", None])["FRENZY_WILLY"]["MKT_HOT"].tolist() == [True, False, False]
    assert WZ.sleeve_name("MOMENTUM", "LONG") == "MOMENTUM_LONG" and WZ.sleeve_name(np.nan, "SHORT") == "MOMENTUM_SHORT"
    assert WZ.sleeve_name("FRENZY_LITE", "LONG") == "FRENZY_LITE"
    for f in ("best_dcr.py", "best_dcr_noflip.py"):                                # one definition: the best case reads the same zones
        src = open(os.path.join(ROOT, "scripts", "best_case", f), encoding="utf-8").read()
        assert "WZ.zones(d, hot=d.hot, late=d.late, cool=d.cool)" in src and '"WILLY_GAP": (ax' not in src


def test_confound_note_splits_alpha_fills_inside_vs_outside_lines(tmp_path):
    df = _rows([("FRENZY_WILLY", True, -4.0, "CLOSED"), ("FRENZY_WILLY", True, 1.2, "CLOSED"),
                ("FRENZY_WILLY", False, -1.0, "CLOSED"), ("FLATBASE_SHORT", True, 5.0, "CLOSED")])
    df["entry_adx"] = [60.0, 30.0, 70.0, np.nan]
    df["entry_ema_gap_8_13"] = [2.0, 2.0, 2.0, np.nan]
    out = "\n".join(W.run(now_ms=0, paths=_write(tmp_path, df), state_path=None, since_ms=0))
    assert "Alpha or another watch line?" in out
    assert "| FRENZY_WILLY | 1 · -4.00 % · $" in out and "| 1 · 100% · +1.20 % · $" in out   # the control fill (ADX 70) is never counted
    assert "| MKT_HOT |" in out                                                        # market line listed as not scored offline
    assert "FRENZY_WILLY -4.00 % → WILLY_GAP" in out and "FRENZY_WILLY +1.20 % → outside every line" in out
    assert "| FLATBASE_SHORT | no block line for this sleeve |" in out


def test_confound_note_scores_mkt_hot_once_when_market_on(tmp_path, monkeypatch):
    df = _rows([("FRENZY_WILLY", True, 1.0, "CLOSED"), ("FRENZY_WILLY_HT", True, 2.0, "CLOSED"), ("FRENZY_LITE", True, -3.0, "CLOSED")])
    df["entry_adx"], df["entry_ema_gap_8_13"] = 30.0, 2.0
    calls = []
    monkeypatch.setattr(W, "market_hot", lambda o, now: calls.append(len(o)) or np.array(["zone"] * len(o), dtype=object))
    out = "\n".join(W.run(now_ms=0, paths=_write(tmp_path, df), state_path=None, since_ms=0, market=True))
    assert calls == [2]                                                               # one read for WILLY + HT, none for LITE
    assert "FRENZY_WILLY +1.00 % → MKT_HOT" in out and "FRENZY_WILLY_HT +2.00 % → MKT_HOT" in out
    monkeypatch.setattr(W, "market_hot", lambda o, now: None)                          # incomplete read → not scored, never 'outside'
    out = "\n".join(W.run(now_ms=0, paths=_write(tmp_path, df), state_path=None, since_ms=0, market=True))
    assert "| FRENZY_WILLY | 0 | 1 · 100% · +1.00 % · $+10 | MKT_HOT |" in out


def test_every_column_a_zone_reads_is_loaded_by_the_scout():
    """(471 deep review) a new line whose column is missing from ZONE_COLS would read all-NaN (= never in the zone) in the scout."""
    import watch_zones as WZ
    src = open(os.path.join(ROOT, "scripts", "watch_zones.py"), encoding="utf-8").read()
    body = src[src.index("def zones("):src.index("ZONE_COLS =")]
    read = set(re.findall(r'_num\(d, "([a-z0-9_]+)"\)', body)) | set(re.findall(r'd\.get\("([a-z0-9_]+)"', body)) | \
        set(re.findall(r'd\.([a-z0-9_]+)\.astype', body)) | set(re.findall(r'"([a-z0-9_]+)" in d', body))
    assert read and read <= set(WZ.ZONE_COLS), read - set(WZ.ZONE_COLS)
    assert set(WZ.ZONE_COLS) <= set(W.COLS)
