"""🧯 DECISION_LOG 582 — EXH_SSC_OIVOL: the EXHAUSTION_SHORT entry filter (declared override) + its frozen shadow revert gate.

Pure rule (services/exhaustion.ssc_oivol_block / oi_pick), the REAL engine judge path (tests/test_exhaustion_short._run: SQLite, every
exchange read stubbed), config / UI / both exports (D11 / D12), the stamps (D13), the scout (pricing on a mocked 1m path, prefix halt,
frozen read-once gate), the observe registry, the opportunity-scout hook and the gitignore."""
import asyncio
import json
import os
import sys
from types import SimpleNamespace as NS

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from services import exhaustion as X  # noqa: E402
from tests import test_exhaustion_short as TX  # noqa: E402
import scout_exh_ssc_oivol as S  # noqa: E402

MIN = 60_000


def _th(**kw):
    return TX._th(**dict(dict(exhaustion_ssc_oivol_block_enabled=True), **kw))


# ── ① the pure rule ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────

def test_block_needs_both_conditions_with_strict_cuts():
    th = _th()
    v = 100e6
    assert X.ssc_oivol_block(-0.02, 0.2 * v, v, th)[:2] == (True, False)                     # both → block
    assert X.ssc_oivol_block(-0.05, 0.2 * v, v, th)[:2] == (False, False)                    # spot share falls a lot → keep
    assert X.ssc_oivol_block(-0.02, 0.1 * v, v, th)[:2] == (False, False)                    # small OI → keep
    assert X.ssc_oivol_block(-0.0366, 0.2 * v, v, th)[0] is False                            # strict: ssc == the cut → keep
    assert X.ssc_oivol_block(-0.02, 0.128 * v, v, th)[0] is False                            # strict: oi_vol == the cut → keep
    assert X.ssc_oivol_block(-0.0365, 0.12801 * v, v, th)[0] is True
    b, u, ov = X.ssc_oivol_block(-0.02, 0.2 * v, v, th)
    assert ov == pytest.approx(0.2)
    for oi, vol, ssc in ((None, v, -0.02), (0.0, v, -0.02), (-5.0, v, -0.02), (0.2 * v, None, -0.02), (0.2 * v, 0.0, -0.02),
                         (float("nan"), v, -0.02), (0.2 * v, v, None), (0.2 * v, v, float("nan")), ("x", v, -0.02)):
        assert X.ssc_oivol_block(ssc, oi, vol, th)[:2] == (False, True), (oi, vol, ssc)  # unreadable → FAIL-OPEN, flagged unread
    th2 = _th(exhaustion_ssc_oivol_ssc_min=-0.01, exhaustion_ssc_oivol_oivol_min=0.5)          # the config cuts are the ones used
    assert X.ssc_oivol_block(-0.02, 0.6 * v, v, th2)[0] is False and X.ssc_oivol_block(-0.005, 0.6 * v, v, th2)[0] is True
    bad = _th(exhaustion_ssc_oivol_ssc_min=7, exhaustion_ssc_oivol_oivol_min=-3)              # a bad value → the study's cut
    p = X.exhaustion_params(bad)
    assert p["oivol_ssc_min"] == -0.0366 and p["oivol_min"] == 0.128
    assert X.ssc_oivol_enabled(_th()) is True and X.ssc_oivol_enabled(_th(exhaustion_ssc_oivol_block_enabled="yes")) is False
    assert X.ssc_oivol_enabled(NS()) is False                                                 # code default OFF


def test_oi_row_pick_is_lag_safe():
    T = 1_763_000_100_000 // (15 * MIN) * (15 * MIN)
    r = lambda ts, v: {"timestamp": ts, "sumOpenInterestValue": str(v)}
    rows = [r(T - 15 * MIN, 1.0), r(T - 10 * MIN, 2.0), r(T - 5 * MIN, 3.0), r(T, 4.0), r(T + 5 * MIN, 99.0)]
    assert X.oi_pick(rows, T) == (3.0, T - 5 * MIN)                                          # timestamp = create_time → the row ≤ T − 5 min
    assert X.oi_pick(list(reversed(rows[:3])), T) == (3.0, T - 5 * MIN)                      # any order
    assert X.oi_pick([r(T - 30 * MIN, 5.0)], T) == (5.0, T - 30 * MIN)                       # = the study's create_time ≥ T − 30 min
    assert X.oi_pick([r(T - 30 * MIN - 1, 5.0)], T) == (None, None)                          # older → stale → unreadable
    assert X.oi_pick([r(T - 5 * MIN + 1, 5.0)], T) == (None, None)                           # newer than T − 5 min → never used
    assert X.oi_pick([r(T, 5.0)], T) == (None, None)
    assert X.oi_pick([r(T, 0), r(T - 5 * MIN, "x"), {"timestamp": None}, "junk"], T) == (None, None)
    assert X.oi_pick([r(T - 5 * MIN, 0), r(T - 10 * MIN, 7.0)], T) == (7.0, T - 10 * MIN)    # a zero row is skipped, not trusted
    assert X.oi_pick(None, T) == (None, None) and X.oi_pick([], T) == (None, None)
    assert X.oi_vol(10.0, 100.0) == pytest.approx(0.1) and X.oi_vol(None, 1) is None and X.oi_vol(1, 0) is None


# ── ② the REAL engine judge path ─────────────────────────────────────────────────────────────────────────────────────────────────────

def _oi(v):
    return lambda T: [{"timestamp": T - 5 * MIN, "sumOpenInterestValue": str(v)}]


def _hook(notional):
    def h(e):
        async def sh(db):
            return notional
        e._exh_shadow_notional = sh
    return h


def _go(oi_rows, th_kw=None, notional=5000.0, k=1):
    mp = pytest.MonkeyPatch()
    try:
        return TX._run(mp, k=k, th_kw=dict(dict(exhaustion_ssc_oivol_block_enabled=True), **(th_kw or {})), oi_rows=oi_rows,
                       engine_hook=_hook(notional))
    finally:
        mp.undo()


def test_engine_block_counts_journals_and_never_orders():
    # spot-share cut −1 → condition (a) always true on the fixture row; a huge OI → condition (b) true
    e, opens, sig, _, calls = _go(_oi(1e15), th_kw=dict(exhaustion_ssc_oivol_ssc_min=-1.0))
    assert opens == [] and e.blocks == ["EXH_SSC_OIVOL"]
    s = sig[0]
    assert s.status == "BLOCKED" and s.block == "EXH_SSC_OIVOL" and "OI/vol24" in s.reason and "spot_share_chg" in s.reason
    assert s.oi_usd == 1e15 and s.oi_vol == pytest.approx(1e15 / s.vol24_usd)
    assert s.decision_price and s.atr_pct and s.tp_pct and s.sl_pct and s.q5_usd                     # what the shadow scout prices with
    assert (s.tp_pct, s.sl_pct) == X.exit_levels(s.atr_pct, _th())
    assert s.shadow_notional_usd == 5000.0 and s.model_slip_pct == pytest.approx(X.model_slip_pct(5000.0, s.q5_usd))
    assert [c for c in calls if c[0] == "oi"]


def test_engine_pass_stamps_the_oi():
    e, opens, sig, _, _ = _go(_oi(1.0), th_kw=dict(exhaustion_ssc_oivol_ssc_min=-1.0))              # tiny OI → (b) false → trade
    assert len(opens) == 1 and sig[0].status == "OPENED" and e.blocks == []
    o = opens[0]
    assert o["entry_exh_oi_usd"] == 1.0 and o["entry_exh_oi_vol"] == round(1.0 / o["entry_pair_volume_24h_usd"], 6)
    assert sig[0].oi_usd == 1.0 and sig[0].model_slip_pct is None


@pytest.mark.parametrize("rows", [[], None, [{"timestamp": 0, "sumOpenInterestValue": "5"}], "stale"])
def test_engine_unreadable_oi_fails_open_and_counts(rows):
    rr = (lambda T: [{"timestamp": T - 31 * MIN, "sumOpenInterestValue": "1e15"}]) if rows == "stale" else rows
    e, opens, sig, _, _ = _go(rr, th_kw=dict(exhaustion_ssc_oivol_ssc_min=-1.0))
    assert len(opens) == 1 and sig[0].status == "OPENED" and e.blocks == ["EXH_SSC_OIVOL_UNREAD"]
    assert opens[0]["entry_exh_oi_usd"] is None and opens[0]["entry_exh_oi_vol"] is None


def test_engine_switch_off_trades_and_still_stamps():
    e, opens, sig, _, calls = _go(_oi(1e15), th_kw=dict(exhaustion_ssc_oivol_ssc_min=-1.0, exhaustion_ssc_oivol_block_enabled=False))
    assert len(opens) == 1 and e.blocks == [] and opens[0]["entry_exh_oi_usd"] == 1e15             # data for the W/L, filter off
    e, opens, sig, _, _ = _go([], th_kw=dict(exhaustion_ssc_oivol_block_enabled=False))
    assert len(opens) == 1 and e.blocks == []                                                      # off: no UNREAD counter either


def test_engine_rule_refusal_never_reads_oi():
    e, opens, sig, _, calls = _go(_oi(1e15), th_kw=dict(exhaustion_vol_decay_min=0.9), k=2)   # ROSE: vol_decay 0.80
    assert opens == [] and sig[0].block == "EXH_VOL_DECAY" and not [c for c in calls if c[0] == "oi"]   # OI read LAST


def test_oi_read_and_shadow_notional_are_total(monkeypatch):
    import services.trading_engine as TE
    e = TX._engine(TE)

    async def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(TE.binance_service, "fetch_exh_open_interest", boom)
    assert asyncio.run(e._exh_oi_read("AUSDT", 900_000 * 2000)) is None

    async def ok(pid, end_ms, limit=12):
        return [{"timestamp": end_ms - 5 * MIN, "sumOpenInterestValue": "123.5"}]
    monkeypatch.setattr(TE.binance_service, "fetch_exh_open_interest", ok)
    assert asyncio.run(e._exh_oi_read("AUSDT", 900_000 * 2000)) == 123.5

    class DB:
        async def rollback(self):
            pass
    e.get_available_balance = boom
    assert asyncio.run(e._exh_shadow_notional(DB())) is None


def test_binance_open_interest_read(monkeypatch):
    from services import binance_service as BS
    svc = BS.binance_service
    seen = []

    class C:
        async def fapiDataGetOpenInterestHist(self, p):
            seen.append(p)
            return [{"timestamp": 1, "sumOpenInterestValue": "2"}]
    monkeypatch.setattr(svc, "exh_exchange", C())
    monkeypatch.setattr(svc, "exh_limited_until", 0.0, raising=False)
    monkeypatch.setattr(BS, "_ban_until", 0.0)
    assert asyncio.run(svc.fetch_exh_open_interest("ABC/USDT:USDT", 123, 12)) == [{"timestamp": 1, "sumOpenInterestValue": "2"}]
    assert seen == [{"symbol": "ABCUSDT", "period": "5m", "endTime": 123, "limit": 12}]

    class Bad:
        async def fapiDataGetOpenInterestHist(self, p):
            raise RuntimeError("HTTP 500")
    monkeypatch.setattr(svc, "exh_exchange", Bad())
    assert asyncio.run(svc.fetch_exh_open_interest("ABCUSDT", 123)) is None
    monkeypatch.setattr(BS, "_ban_until", 4e12)
    assert asyncio.run(svc.fetch_exh_open_interest("ABCUSDT", 123)) is None                   # a ban → no call


# ── ③ config / UI / exports / stamps / journal export (D11 · D12 · D13) ─────────────────────────────────────────────────────────────

def test_config_defaults_json_ui_exports_and_validation():
    import config as C
    mf = C.SignalThresholds.model_fields
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    for k, d, j in (("exhaustion_ssc_oivol_block_enabled", False, True), ("exhaustion_ssc_oivol_ssc_min", -0.0366, -0.0366),
                    ("exhaustion_ssc_oivol_oivol_min", 0.128, 0.128)):
        assert mf[k].default == d and cfg[k] == j, k
    src = open(os.path.join(ROOT, "config.py"), encoding="utf-8").read()
    assert "DECISION_LOG 582" in src and "DECLARED OVERRIDE" in src
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    for i in ("config-exh-ssc-oivol-enabled", "config-exh-ssc-oivol-ssc-min", "config-exh-ssc-oivol-oivol-min"):
        assert html.count(f'id="{i}"') == 1, i
    a, b = html.index("<!-- ===== 🪫 EXHAUSTION_SHORT BOX"), html.index("<!-- ===== EXHAUSTION_SHORT BOX END ===== -->")
    assert 'id="config-exh-ssc-oivol-enabled"' in html[a:b]                                     # inside the EXHAUSTION box
    assert "['config-exh-ssc-oivol-ssc-min', 'exhaustion_ssc_oivol_ssc_min', -0.0366]" in html     # EXHAUSTION_NUM_FIELDS = load AND save
    assert "['config-exh-ssc-oivol-oivol-min', 'exhaustion_ssc_oivol_oivol_min', 0.128]" in html
    assert "config.thresholds.exhaustion_ssc_oivol_block_enabled" in html
    assert "exhaustion_ssc_oivol_block_enabled: document.getElementById('config-exh-ssc-oivol-enabled')" in html
    # both text exports: the shared config-line builder (🪫 line) and the FRENZY monitor line (exhaustionMonitorStr)
    assert html.count("..._buildConfigLines(cfg, changelog, hr, hr2, status)") == 2
    assert "🧯 EXH_SSC_OIVOL (DL 582, declared override): ${_bt.exhaustion_ssc_oivol_block_enabled === true" in html
    f = html[html.index("function exhaustionMonitorStr(m)"):]
    f = f[:f.index("\n        }")]
    assert "🧯 EXH_SSC_OIVOL ${m.exhaustion_ssc_oivol_enabled ? 'ON' : 'OFF'}" in f and "p.oivol_min" in f and "p.oivol_ssc_min" in f
    main = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    assert '"exhaustion_ssc_oivol_enabled": getattr(_th, \'exhaustion_ssc_oivol_block_enabled\', False) is True' in main
    for c in ("exh_oi_usd", "exh_oi_vol", "exh_model_slip_pct", "exh_shadow_notional_usd"):
        assert f'"{c}"' in main and f"{c}=getattr(b, " in main, c
    import config
    th = config.trading_config.thresholds
    assert X.exhaustion_config_errors(th, 20.0) == []
    bad = NS(**dict(th.model_dump(), exhaustion_ssc_oivol_block_enabled="on", exhaustion_ssc_oivol_ssc_min=-2.0,
                    exhaustion_ssc_oivol_oivol_min=-0.1))
    errs = " | ".join(X.exhaustion_config_errors(bad, 20.0))
    for k in ("exhaustion_ssc_oivol_block_enabled", "exhaustion_ssc_oivol_ssc_min", "exhaustion_ssc_oivol_oivol_min"):
        assert k in errs, k


def test_stamps_columns_and_migration():
    import models
    import services.trading_engine as TE
    cols = {c.name for c in models.Order.__table__.columns}
    assert {"entry_exh_oi_usd", "entry_exh_oi_vol"} <= cols
    for c in ("entry_exh_oi_usd", "entry_exh_oi_vol"):
        assert TE.TradingEngine.stamp_class(c) == "sleeve"
    jc = {c.name for c in models.ExhaustionSignal.__table__.columns}
    assert {"oi_usd", "oi_vol", "model_slip_pct", "shadow_notional_usd"} <= jc
    db = open(os.path.join(ROOT, "database.py"), encoding="utf-8").read()
    assert "'entry_exh_oi_usd', 'entry_exh_oi_vol'" in db and "ALTER TABLE exhaustion_signals ADD COLUMN" in db
    eng = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert "entry_exh_oi_usd=entry_exh_oi_usd, entry_exh_oi_vol=entry_exh_oi_vol" in eng      # a source on the open path


# ── ④ the scout ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

T0 = int(pd.Timestamp("2026-11-20").value // 1_000_000)


def _path(drop_at=None, rise_at=None, n=S.HOLD_MIN, t1=T0):
    rows = [(t1 + i * MIN, 100.0, 100.2, 99.9, 100.0) for i in range(n)]
    if drop_at is not None:
        rows[drop_at] = (t1 + drop_at * MIN, 100.0, 100.0, 80.0, 81.0)
    if rise_at is not None:
        rows[rise_at] = (t1 + rise_at * MIN, 100.0, 120.0, 99.0, 119.0)
    return rows


def test_price_signal_live_exit_with_slip_on_the_entry(monkeypatch):
    calls = []
    monkeypatch.setattr(S.XO, "fetch_1m", lambda pair, t, n: calls.append(t) or _path(drop_at=30))
    cache = {}
    assert S.price_signal("AUSDT", T0, 100.0, 10.0, 6.0, 0.2, cache, now_ms=T0 + 3_600_000) is None and calls == []   # path not over
    net, why = S.price_signal("AUSDT", T0, 100.0, 10.0, 6.0, 0.2, cache, now_ms=T0 + 26 * 3_600_000)
    assert why == "TP" and net == pytest.approx(10.0 - S.XO.TAKER - S.XO.MAKER - 0.2) and calls == [T0]   # maker TP: slip on the entry only
    assert S.price_signal("AUSDT", T0, 100.0, 10.0, 6.0, 0.2, cache, now_ms=T0 + 26 * 3_600_000) == (net, why) and calls == [T0]   # cached
    monkeypatch.setattr(S.XO, "fetch_1m", lambda pair, t, n: _path(rise_at=10))
    net, why = S.price_signal("BUSDT", T0, 100.0, 10.0, 6.0, 0.2, {}, now_ms=T0 + 26 * 3_600_000)
    assert why == "SL" and net == pytest.approx(-6.0 - 2 * S.XO.TAKER - 0.2 - 0.2)            # taker stop: slip on both legs
    monkeypatch.setattr(S.XO, "fetch_1m", lambda pair, t, n: _path()[:100])
    assert S.price_signal("CUSDT", T0, 100.0, 10.0, 6.0, 0.0, {}, now_ms=T0 + 25 * 3_600_000) is None
    c = {}
    assert S.price_signal("CUSDT", T0, 100.0, 10.0, 6.0, 0.0, c, now_ms=T0 + 40 * 3_600_000) == "NO_DATA" and c
    assert S.price_signal("DUSDT", T0, None, 10.0, 6.0, 0.0, {}, now_ms=T0 + 40 * 3_600_000) == "NO_DATA"
    assert S.row_levels(10.0, 6.0, 1.0) == (10.0, 6.0) and S.row_levels(None, 6.0, 2.0) == (7.0, 3.0) and S.row_levels(None, None, None) is None   # 🔁 (609) the TP-7 cap


def test_gate_read_rule():
    assert S.gate([])[0] == "COLLECTING"
    assert S.gate([("d%d" % (i % 9), 1.0) for i in range(40)])[0] == "COLLECTING"           # 9 days → not yet
    v, txt, n = S.gate([("d%d" % (i % 10), 1.0) for i in range(30)])
    assert v == "REVERT" and n == 30 and "lower bound" in txt                                  # clearly winners → the filter blocked winners
    v, txt, n = S.gate([("d%d" % (i % 10), -0.5) for i in range(30)])
    assert v == "KEEP" and n == 30
    rows = [("d0", 2.0)] * 30 + [("d%d" % i, 2.0) for i in range(1, 10)]                       # 10 days reached only at row 39
    v, txt, n = S.gate(rows)
    assert n == 39 and v == "REVERT"
    noisy = [("d%d" % (i % 12), (5.6 if i % 2 else -5.4)) for i in range(30)]                  # mean +0.1 with SD ≈ 5.6 → NOT shown to be winners
    assert S.gate(noisy)[0] == "KEEP"
    v, _, n = S.gate([("d%d" % (i % 10), 0.0) for i in range(30)])
    assert v == "KEEP"                                                                         # mean exactly 0 → not > 0 → KEEP


def _journal(n, block="EXH_SSC_OIVOL", start="2026-11-20T00:15:00", step_h=8):
    b = pd.Timestamp(start)
    return pd.DataFrame([dict(e="EXHAUSTION", pair=f"P{i}USDT", exh_signal_at=(b + pd.Timedelta(hours=step_h * i)).strftime("%Y-%m-%dT%H:%M:%S"),
                              exh_status="BLOCKED", exh_block=block, exh_reason="x", exh_decision_price=100.0, exh_atr_pct=2.0,
                              exh_tp_pct=10.0, exh_sl_pct=3.0, exh_model_slip_pct=0.1, exh_oi_vol=0.2, exh_spot_share_chg=-0.02)
                         for i in range(n)])


def test_report_prefix_halt_frozen_once_and_revert(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "CACHE", str(tmp_path / "c.json"))
    j = pd.concat([_journal(35), _journal(3, block="EXH_VOL_DECAY", start="2026-11-20T01:00:00")], ignore_index=True)
    early = _journal(2, start="2026-11-01T00:15:00")                                             # before the deploy floor → not counted
    j = pd.concat([j, early], ignore_index=True)
    floor = int(pd.Timestamp("2026-11-19").value // 1_000_000)
    n = [0]

    def fake(pair, t_ms, entry, tp, sl, slip, cache, now_ms=None):
        n[0] += 1
        if pair == "P5USDT" and fake.fail:
            raise S.urllib.error.URLError("down")
        return (fake.val, "TP")
    fake.fail, fake.val = True, 1.0
    monkeypatch.setattr(S, "price_signal", fake)
    st = tmp_path / "st.json"
    txt = S.report(j, None, now_ms=0, state_path=str(st), floor_ms=floor)
    assert "pricing halted at P5USDT" in txt and "COLLECTING" in txt and n[0] == 6 and not st.exists()
    fake.fail = False
    txt = S.report(j, None, now_ms=0, state_path=str(st), floor_ms=floor)
    rec = json.load(open(st))
    assert rec["verdict"] == "REVERT" and rec["n"] == 30 and "FROZEN" in txt and "running: N 35" in txt
    fake.val = -3.0                                                                              # later prices never re-judge a frozen read
    txt = S.report(j, None, now_ms=0, state_path=str(st), floor_ms=floor)
    assert json.load(open(st)) == rec and "FROZEN" in txt and "REVERT" in txt
    st2 = tmp_path / "st2.json"
    S.report(j, None, now_ms=0, state_path=str(st2), floor_ms=floor)
    assert json.load(open(st2))["verdict"] == "KEEP"
    txt = S.report(j, None, now_ms=0, state_path=str(tmp_path / "x.json"), floor_ms=None)
    assert "NOT EVALUABLE" in txt


def test_report_kept_fills_information(monkeypatch, tmp_path):
    monkeypatch.setattr(S, "CACHE", str(tmp_path / "c.json"))
    monkeypatch.setattr(S, "price_signal", lambda *a, **k: (-1.0, "SL"))
    fills = pd.DataFrame([dict(opened_at="2026-11-20T03:00:00", pnl_percentage=2.0), dict(opened_at="2026-12-30T03:00:00", pnl_percentage=-9.0)])
    txt = S.report(_journal(3), fills, now_ms=0, state_path=None, floor_ms=0)
    assert "kept EXHAUSTION fills over the same window: N 1 · mean +2.000" in txt and "running: N 3 · WR 0 % · mean -1.000" in txt


def test_load_journal_carries_the_pricing_columns(tmp_path):
    import scout_exhaustion_short as SC
    p = tmp_path / "scalpars_decisions_x.csv"
    _journal(2).to_csv(p, index=False)
    j = SC.load_journal([str(p)])
    for c in ("exh_decision_price", "exh_atr_pct", "exh_tp_pct", "exh_sl_pct", "exh_model_slip_pct", "exh_oi_vol", "exh_spot_share_chg"):
        assert c in j.columns, c
    assert len(S.cohort(j, 0)) == 2


# ── ⑤ registry · hook · gitignore ──────────────────────────────────────────────────────────────────────────────────────────────────

def test_registry_hook_and_gitignore():
    import observe_registry as OR
    assert "scout_exh_ssc_oivol.py" in OR.classified()
    cfg = json.load(open(os.path.join(ROOT, "trading_config.json")))["thresholds"]
    rows = {r[0]: r for r in OR.status(cfg)}
    assert rows["scout_exh_ssc_oivol.py"][4] == "OK"
    w = OR.EXIT["scout_exh_ssc_oivol.py"]["walks"]
    assert w == {"exhaustion_tp_atr_mult": 20.0, "exhaustion_tp_cap_pct": 7.0, "exhaustion_sl_atr_mult": 1.5, "exhaustion_sl_cap_pct": 6.0,
                 "exhaustion_max_hold_hours": 24.0}
    assert S.LIVE_EXIT == dict(tp_atr=20.0, tp_cap=7.0, sl_atr=1.5, sl_cap=6.0, hold_h=24.0) and S.DL_EXIT == 609
    src = open(os.path.join(ROOT, "scripts", "opportunity_scout.py"), encoding="utf-8").read()
    assert "import scout_exh_ssc_oivol as _sso" in src and "_rg_sec += _sso.run(now_ms)" in src
    gi = open(os.path.join(ROOT, ".gitignore"), encoding="utf-8").read().split()
    assert "reports/.exh_ssc_oivol_cache.json" in gi and "reports/SCOUT_EXH_SSC_OIVOL_609.json" in gi
    assert S.DL == 582 and S.GATE_N == 30 and S.GATE_DAYS == 10


def test_config_refuses_a_cut_that_can_never_block():
    from types import SimpleNamespace as NS
    import config as C
    th = C.SignalThresholds()
    bad = NS(**dict(th.model_dump(), exhaustion_ssc_oivol_ssc_min=-0.001, exhaustion_spot_share_chg_max=-0.0066))
    assert any("exhaustion_ssc_oivol_ssc_min" in e for e in X.exhaustion_config_errors(bad))
