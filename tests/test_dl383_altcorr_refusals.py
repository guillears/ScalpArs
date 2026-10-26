"""📈🧷 (DECISION_LOG 383) OBSERVE-ONLY additions — ① the alt-correlation regime stamp (orders.entry_alt_corr24 / entry_alt_beta24 = the frozen
OBS-R1 / R1b of ML_REGIME_WL_2026-10-26 §8) on every fill, ② the matched momentum-long refusal recorder (ml_matched_refusals → e=ML_MATCHED rows
of the Decisions CSV) + its scout. No network: every exchange read is a fake."""
import asyncio
import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from services import alt_corr as AC  # noqa: E402

BAR, DAY = 300_000, 86_400_000
T0 = int(pd.Timestamp("2026-10-20 00:00:00").value // 1_000_000)


# ─────────────────────────────── fixture market (seeded, deterministic) ───────────────────────────────
def _market(n_bars=2100, seed=383):
    """21 alts + BTC: alt log returns = beta_i · btc + idio; a few alts broken on purpose (stale / short history / window gap)."""
    rng = np.random.default_rng(seed)
    t = T0 + np.arange(n_bars, dtype=np.int64) * BAR
    btc_r = rng.normal(0, 0.002, n_bars)
    bars = {"BTCUSDT": (t, 60000 * np.exp(np.cumsum(btc_r)))}
    names = [f"A{i:02d}USDT" for i in range(21)]
    for i, p in enumerate(names):
        r = (0.6 + 0.08 * i) * btc_r + rng.normal(0, 0.0015 + 0.0001 * i, n_bars)
        bars[p] = (t.copy(), (1 + i) * np.exp(np.cumsum(r)))
    # A03 stale: its last 9 bars missing (> 30 min)
    tt, cc = bars["A03USDT"]; bars["A03USDT"] = (tt[:-9], cc[:-9])
    # A05 short history: only the last 1500 bars exist
    tt, cc = bars["A05USDT"]; bars["A05USDT"] = (tt[-1500:], cc[-1500:])
    # A07 gap inside the 288-bar window
    tt, cc = bars["A07USDT"]; keep = np.ones(len(tt), bool); keep[-50] = False; bars["A07USDT"] = (tt[keep], cc[keep])
    return bars, names


def _rows(t, c):
    return [[int(a), float(x), float(x), float(x), float(x), 1.0] for a, x in zip(t, c)]


# ─────────────────────────────── ① parity with the study's own code ───────────────────────────────
def test_corr_beta_equal_study_breadth_feats_on_fixture(monkeypatch, tmp_path):
    import study_ml_regime_wl_build as RB
    bars, names = _market()
    order = names[:20]                                                           # A20 is #21 → never back-filled
    T = int(bars["BTCUSDT"][0][-1]) + BAR + 123_456                              # mid-bar entry: the last bar closed ≤ T
    day = (T // DAY) * DAY

    def load5(pair):
        if pair not in bars:
            return None
        t, c = bars[pair]
        return pd.DataFrame(dict(t=t.astype(np.int64), h=c, l=c, c=c, qv=np.ones(len(t))))
    monkeypatch.setattr(RB, "load5", load5)
    monkeypatch.setattr(RB, "universe", lambda: {day - DAY: order})          # the list DATED the previous UTC day (study reading)
    monkeypatch.setattr(RB, "FUND", str(tmp_path / "no_funding"))
    F = RB.breadth_feats(np.array([T], dtype=np.int64))
    s_corr, s_beta = float(F.br_corr24h_top20.iloc[0]), float(F.br_beta_btc_24h.iloc[0])

    pair_rows = {p: _rows(*bars[p]) for p in names}
    listed = {p: int(bars[p][0][0]) for p in names}                             # onboard = the first bar (the study's cache start)
    corr, beta, n = AC.alt_corr_beta(pair_rows, _rows(*bars["BTCUSDT"]), order, T, listed)
    assert n == 17                                                                # 20 − stale A03 − short A05 − gap A07
    assert np.isfinite(s_corr) and abs(corr - s_corr) < 1e-12
    assert np.isfinite(s_beta) and abs(beta - s_beta) < 1e-12
    # the engine fetch only holds the last 290 bars: same value (history judged by onboard date, not by the rows fetched)
    short = {p: r[-290:] for p, r in pair_rows.items()}
    c2, b2, n2 = AC.alt_corr_beta(short, _rows(*bars["BTCUSDT"])[-290:], order, T, listed)
    assert n2 == 17 and abs(c2 - s_corr) < 1e-12 and abs(b2 - s_beta) < 1e-12


def test_universe_reads_day_x_minus_2_volume_like_the_study(monkeypatch, tmp_path):
    """the study reads the list DATED X − 1, which ranks day X − 2's volume (backtest_phase1_universe) — the engine ranks day X − 2."""
    import study_ml_regime_wl_build as RB
    rng = np.random.default_rng(1)
    syms = [f"S{i:02d}USDT" for i in range(60)] + ["BTCUSDT", "USDCUSDT", "龙虾USDT", "1000PEPEUSDT"]
    days = [T0 + k * DAY for k in range(4)]
    vols = {d: {s: float(rng.uniform(1, 100)) for s in syms} for d in days}
    vols[days[1]]["BTCUSDT"] = vols[days[1]]["USDCUSDT"] = vols[days[1]]["龙虾USDT"] = 1e9   # in the top 50, then filtered
    vols[days[1]]["1000PEPEUSDT"] = 5e8
    rows = []
    for d in days[1:]:                                                            # the builder: day d ranked by day d − 1's volume
        for r, (qv, s) in enumerate(sorted(((v, s) for s, v in vols[d - DAY].items()), reverse=True)[:50], 1):
            rows.append(dict(date_ms=d, rank=r, pair=s, quote_vol=qv))
    f = tmp_path / "u.csv"; pd.DataFrame(rows).to_csv(f, index=False)
    monkeypatch.setattr(RB, "TOP50", [str(f)])
    U = RB.universe()
    T = days[3] + 3 * 3_600_000                                                   # an entry on day X = days[3]
    di = max(d for d in U if d <= (T // DAY) * DAY - DAY)                         # breadth_feats' choice
    study20 = U[di][:20]
    assert AC.universe_day_ms(T) == days[1]
    eng20 = AC.universe_top20(vols[AC.universe_day_ms(T)])
    assert eng20 == study20 and eng20[0] == "1000PEPEUSDT"
    assert not {"BTCUSDT", "USDCUSDT", "龙虾USDT"} & set(eng20)


# ─────────────────────────────── tracker: once per bar, stamp freshness, fail-safe ───────────────────────────────
def _fake_fetchers(bars, names, fail=None):
    calls = {"k": 0, "u": 0}

    async def kl(pair, interval, limit):
        calls["k"] += 1
        if fail == "klines":
            raise RuntimeError("boom")
        if interval == "1d":
            day = AC.universe_day_ms(int(bars["BTCUSDT"][0][-1]) + BAR)
            vol = {p: 1000.0 - i for i, p in enumerate(names)}
            return [[day, 0, 0, 0, 0, 0, day + DAY - 1, vol.get(pair, 1.0)]]
        t, c = bars[pair]
        return _rows(t, c)[-limit:]

    async def uni():
        calls["u"] += 1
        if fail == "universe":
            return None
        extra = {f"X{i:02d}USDT": 1e3 for i in range(40)}                       # ≥ 50 day volumes needed (a thin read is refused)
        return (dict({p: 1e6 for p in names + ["BTCUSDT"]}, **extra), {p: int(bars[p][0][0]) for p in names})
    return kl, uni, calls


def test_tracker_computes_once_per_bar_and_stamps_only_fresh_values():
    bars, names = _market()
    kl, uni, calls = _fake_fetchers(bars, names)
    tr = AC.AltCorrTracker(fetch_klines=kl, fetch_universe=uni)
    now = int(bars["BTCUSDT"][0][-1]) + BAR + 6_000
    v = asyncio.run(tr.refresh(now))
    assert v["corr"] is not None and v["beta"] is not None and v["n"] >= 8
    n_calls = calls["k"]
    asyncio.run(tr.refresh(now + 60_000))                                         # same bar → no new reads
    assert calls["k"] == n_calls and calls["u"] == 1
    c, b = tr.stamp(now + 60_000)
    assert c == round(v["corr"], 6) and b == round(v["beta"], 6)
    assert tr.stamp(now + 2 * BAR) != (None, None)                               # ≤ 2 bars old → still used
    assert tr.stamp(now + 3 * BAR) == (None, None)                               # older → NULL, never a stale value
    assert AC.AltCorrTracker().stamp(now) == (None, None)                        # nothing computed yet


@pytest.mark.parametrize("fail", ["klines", "universe"])
def test_tracker_failures_give_null_and_never_raise(fail):
    bars, names = _market()
    kl, uni, _ = _fake_fetchers(bars, names, fail=fail)
    tr = AC.AltCorrTracker(fetch_klines=kl, fetch_universe=uni)
    now = int(bars["BTCUSDT"][0][-1]) + BAR + 6_000
    asyncio.run(tr.refresh(now))
    assert tr.stamp(now) == (None, None)


def test_too_few_scored_pairs_is_null():
    bars, names = _market()
    rows = {p: _rows(*bars[p]) for p in names[:7]}
    T = int(bars["BTCUSDT"][0][-1]) + BAR
    assert AC.alt_corr_beta(rows, _rows(*bars["BTCUSDT"]), names[:7], T, None)[:2] == (None, None)
    assert AC.alt_corr_beta(None, None, None, T) == (None, None, 0)              # garbage in → NULL, no raise


def test_engine_stamp_helper_is_null_when_the_tracker_raises(monkeypatch):
    import services.trading_engine as TE

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(TE.alt_corr_tracker, "stamp", boom)
    assert TE.alt_corr_stamp() == {"entry_alt_corr24": None, "entry_alt_beta24": None}


# ─────────────────────────────── real open_position: fill stamp + refusal recorder ───────────────────────────────
def _open_momentum(monkeypatch, matched=None, recorder_boom=False, corr=(0.42, 1.5)):
    """the REAL open_position (paper, in-memory SQLite, offline) for a momentum LONG on FOOUSDT."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    from tests.test_willy_global_hold_switch import _hold_db, _offline
    _offline(monkeypatch, TE)
    monkeypatch.setattr(TE.TradingEngine, "_willy_global_hold_on", lambda self: False)
    now_ms = int(pd.Timestamp.now("UTC").value // 1_000_000)
    monkeypatch.setattr(TE.alt_corr_tracker, "value", dict(bar_close_ms=(now_ms // BAR) * BAR, corr=corr[0], beta=corr[1], n=20, at=0))
    if matched:
        monkeypatch.setattr(TE.config.trading_config.thresholds, "long_unmatched_only", True)
        monkeypatch.setattr(TE.config.trading_config.thresholds, "long_w2_reenable_1h_min", 99.0, raising=False)
        monkeypatch.setattr(TE, "_compute_pattern_c_match", lambda *a, **k: (False,) * 10)
        monkeypatch.setattr(TE, "_compute_pattern_w_match", lambda *a, **k: (False, True, False, False, False, False, True))

        async def _noflip(*a, **k):
            return None
        monkeypatch.setattr(TE.TradingEngine, "_maybe_open_flip", _noflip)
    if recorder_boom:
        def _boom(self, **k):
            raise RuntimeError("recorder down")
        monkeypatch.setattr(TE.TradingEngine, "_ml_refusal_record", _boom)

    async def go():
        eng, SL = await _hold_db(monkeypatch, open_willy=False)()
        e = TE.TradingEngine(); e.is_running = True; e.is_paper_mode = True
        blocks = []
        _orig = e._record_filter_block
        e._record_filter_block = lambda n, d, had_room=True: (blocks.append(n), _orig(n, d, had_room))

        async def bal(db):
            return 5000.0
        e.get_available_balance = bal
        async with SL() as db:
            r = await e.open_position(db=db, pair="FOOUSDT", direction="LONG", confidence="STRONG_BUY", current_price=1.0,
                                      entry_rsi=55.0, entry_adx=25.0, entry_atr_pct=0.8, entry_btc_rsi=58.0, entry_btc_regime="STRONG_BULL")
        await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            orders = (await s_.execute(select(models.Order))).scalars().all()
            refusals = (await s_.execute(select(models.MatchedLongRefusal))).scalars().all()
        await eng.dispose()
        return r, blocks, orders, refusals, e
    return asyncio.run(go())


def test_every_fill_carries_the_stamp(monkeypatch):
    r, blocks, orders, refusals, _ = _open_momentum(monkeypatch)
    assert r is not None and len(orders) == 1 and not refusals
    assert orders[0].entry_alt_corr24 == 0.42 and orders[0].entry_alt_beta24 == 1.5


def test_fill_stamp_null_when_the_value_is_missing(monkeypatch):
    import services.trading_engine as TE
    r, _, orders, _, _ = _open_momentum(monkeypatch, corr=(None, None))
    assert r is not None and orders[0].entry_alt_corr24 is None and orders[0].entry_alt_beta24 is None


def test_refusal_is_recorded_with_labels_snapshot_and_slots(monkeypatch):
    r, blocks, orders, refusals, e = _open_momentum(monkeypatch, matched=True)
    assert r is None and "LONG_UNMATCHED_ONLY" in blocks and not orders                  # the decision is unchanged: refused
    assert len(refusals) == 1
    m = refusals[0]
    assert (m.pair, m.labels, m.family, m.c_any, m.w_any, m.btc_regime, m.confidence) == ("FOOUSDT", "W2", "W", False, True, "STRONG_BULL", "STRONG_BUY")
    assert m.room is True and m.open_count == 0 and m.max_positions >= 1 and m.price == 1.0 and m.is_paper is True
    _sec = (m.at - dt.datetime(1970, 1, 1)).total_seconds()
    assert m.bar_at == dt.datetime(1970, 1, 1) + dt.timedelta(seconds=(int(_sec) // 300) * 300)          # the 5m bar the refusal fell in
    ef = json.loads(m.entry_fields_json)
    assert ef["entry_rsi"] == 55.0 and ef["entry_atr_pct"] == 0.8 and ef["entry_btc_regime"] == "STRONG_BULL" and ef["entry_cohort"] == "W2"
    assert m.alt_corr24 == 0.42 and m.alt_beta24 == 1.5
    # a second refusal of the same pair in the same 5m bar is not recorded again (in-memory dedupe)
    assert e._ml_refusal_record(pair="FOOUSDT", price=1.1, confidence="STRONG_BUY", labels="W2", c_any=False, w_any=True,
                                entry_fields={}) is False


def test_recorder_failure_never_changes_the_refusal(monkeypatch):
    r, blocks, orders, refusals, _ = _open_momentum(monkeypatch, matched=True, recorder_boom=True)
    assert r is None and "LONG_UNMATCHED_ONLY" in blocks and not orders and not refusals


def test_recorder_dedupe_across_restart_and_bounded_memory(monkeypatch):
    """unique (pair, bar_at, is_paper): a restarted process that refuses the same pair in the same bar writes nothing new."""
    import models
    import services.trading_engine as TE
    from sqlalchemy import select
    from tests.test_willy_global_hold_switch import _hold_db

    async def go():
        eng, SL = await _hold_db(monkeypatch)()
        out = []
        t = T0 + 7 * BAR + 10_000
        for proc in range(2):                                                         # two "processes" (fresh in-memory dedupe)
            e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
            out.append(e._ml_refusal_record(pair="FOOUSDT", price=1.0, confidence="HIGH", labels="C6+W2", c_any=True, w_any=True,
                                            entry_fields={"x": float("nan"), "y": np.float64(2.0)}, now_ms=t + proc * 1000))
            await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
        e.ML_REFUSAL_SEEN_MAX = 3
        for k in range(10):
            e._ml_refusal_record(pair=f"P{k}USDT", price=1.0, confidence="HIGH", labels="W2", c_any=False, w_any=True, entry_fields={},
                                 now_ms=t + k * 3_600_000)
        await asyncio.gather(*list(getattr(e, "_wh_tasks", set())))
        async with SL() as s_:
            rows = (await s_.execute(select(models.MatchedLongRefusal).where(models.MatchedLongRefusal.pair == "FOOUSDT"))).scalars().all()
        await eng.dispose()
        return out, rows, len(e._mr_seen)
    out, rows, nseen = asyncio.run(go())
    assert out == [True, True] and len(rows) == 1 and rows[0].family == "C+W"       # the second insert found the row → skipped
    assert json.loads(rows[0].entry_fields_json) == {"x": None, "y": 2.0}            # NaN never reaches the JSON
    assert nseen <= 4                                                                 # pruned to the last hour beyond the cap


# ─────────────────────────────── migration ───────────────────────────────
def test_migration_adds_the_columns_and_the_table(tmp_path, monkeypatch):
    import models
    import database
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy import text
    eng = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'old.db'}")

    async def go():
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
            for col in ("entry_alt_corr24", "entry_alt_beta24"):
                await c.execute(text(f"ALTER TABLE orders DROP COLUMN {col}"))
            await c.execute(text("DROP TABLE ml_matched_refusals"))
        monkeypatch.setattr(database, "engine", eng)
        await database.init_db()
        async with eng.begin() as c:
            ocols = [r[1] for r in (await c.execute(text("PRAGMA table_info(orders)"))).all()]
            tabs = [r[0] for r in (await c.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))).all()]
        await eng.dispose()
        return ocols, tabs
    ocols, tabs = asyncio.run(go())
    assert {"entry_alt_corr24", "entry_alt_beta24"} <= set(ocols) and "ml_matched_refusals" in tabs
    assert {"entry_alt_corr24", "entry_alt_beta24"} <= {c.name for c in models.Order.__table__.columns}   # D12: the orders CSV dumps every column


# ─────────────────────────────── exports / dashboard (D12) ───────────────────────────────
def _seed_perf(models, paper):
    out = []
    for i, (p, c, b) in enumerate([(0.8, 0.30, 1.2), (-0.5, 0.45, 1.6), (0.3, None, None)]):
        t = dt.datetime(2026, 10, 27, 10 + i)
        out.append(models.Order(pair=f"P{i}USDT", direction="LONG", status="CLOSED", is_paper=paper, entry_strategy="MOMENTUM", confidence="STRONG_BUY",
                                entry_price=1.0, exit_price=1.0 + p / 100, quantity=100.0, investment=10.0, leverage=10, notional_value=100.0,
                                opened_at=t, closed_at=t + dt.timedelta(minutes=30), pnl=p, pnl_percentage=p, close_reason="STOP_LOSS",
                                entry_alt_corr24=c, entry_alt_beta24=b))
    return out


def test_api_performance_carries_alt_columns_on_both_wl_tables():
    import models
    import main as M
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        async with SL() as s:
            for o in _seed_perf(models, M.trading_engine.is_paper_mode):
                s.add(o)
            await s.commit()
        async with SL() as s:
            r = await M.get_performance(db=s)
        await eng.dispose()
        return r if isinstance(r, dict) else json.loads(r.body)
    d = asyncio.run(go())
    assert d["total_trades"] == 3                                                     # not the all-zeros fallback
    eco = {(r["outcome"], r["direction"]): r for r in d["entry_conditions_by_outcome"]}
    assert eco[("Winners", "LONG")]["avg_alt_corr24"] == 0.3 and eco[("Winners", "LONG")]["avg_alt_beta24"] == 1.2   # the NULL fill is skipped
    assert eco[("Losers", "LONG")]["avg_alt_corr24"] == 0.45
    so = {(r["strategy"], r["outcome"]): r for r in d["entry_conditions_by_strategy_outcome"]}
    assert so[("MOMENTUM", "Losers")]["avg_alt_beta24"] == 1.6 and so[("MOMENTUM", "Winners")]["avg_alt_corr24"] == 0.3


def test_d12_grep_ui_header_and_both_text_exports():
    html = open(os.path.join(ROOT, "templates", "index.html"), encoding="utf-8").read()
    assert html.count(">AltCorr</th>") == 1 and html.count(">AltBeta</th>") == 1                  # the by-Outcome thead (cloned into both strategy tables)
    assert html.count("${r.avg_alt_corr24 != null ? safe(r.avg_alt_corr24).toFixed(3) : '-'}") == 1   # by-Outcome UI row
    assert html.count("${r.avg_alt_corr24!=null?safe(r.avg_alt_corr24).toFixed(3):'-'}") == 1      # by-Strategy UI row renderer
    assert html.count("${'AltCorr'.padEnd(8)} ${'AltBeta'}") == 4                                 # headers: outcome + strategy × BOTH exports
    assert html.count("r.avg_alt_corr24 != null ? r.avg_alt_corr24.toFixed(3) : '-'") == 2        # outcome rows × BOTH exports
    assert html.count("${g(r.avg_alt_corr24,3).padEnd(8)} ${g(r.avg_alt_beta24,3)}") == 2         # strategy rows × BOTH exports
    i1 = html.index("## Entry Conditions by Outcome (Winners vs Losers)\\n")
    i2 = html.index("## Entry Conditions by Outcome (Winners vs Losers)\\n", i1 + 10)
    assert "AltCorr" in html[i1:i1 + 6000] and "AltCorr" in html[i2:i2 + 6000]                  # one per export function


def test_decisions_export_carries_ml_matched_rows(monkeypatch, tmp_path):
    import models
    import main as M
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from services import decision_journal as DJ
    monkeypatch.setattr(DJ, "export_rows", lambda days, *a, **k: [])

    async def go():
        eng = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with eng.begin() as c:
            await c.run_sync(models.Base.metadata.create_all)
        SL = async_sessionmaker(eng, expire_on_commit=False)
        paper = M.trading_engine.is_paper_mode
        now = dt.datetime.utcnow()
        async with SL() as s:
            s.add(models.MatchedLongRefusal(pair="FOOUSDT", at=now - dt.timedelta(days=10), bar_at=now - dt.timedelta(days=10), price=1.5,
                                            confidence="HIGH", labels="W2+W4", family="W", c_any=False, w_any=True, btc_regime="STRONG_BULL",
                                            open_count=1, max_positions=4, room=True, post_gate="", entry_fields_json='{"entry_atr_pct":0.7}',
                                            is_paper=paper))
            s.add(models.MatchedLongRefusal(pair="OLDUSDT", at=now - dt.timedelta(days=60), bar_at=now - dt.timedelta(days=60), is_paper=paper))
            await s.commit()

        async def _init(db):
            return None
        monkeypatch.setattr(M.trading_engine, "initialize", _init)
        async with SL() as s:
            resp = await M.export_decisions_csv(days=3, db=s)
            body = b"".join([x if isinstance(x, bytes) else x.encode() async for x in resp.body_iterator]).decode()
        await eng.dispose()
        return body
    body = asyncio.run(go())
    import io
    d = pd.read_csv(io.StringIO(body))
    m = d[d.e == "ML_MATCHED"]
    assert list(m.pair) == ["FOOUSDT"]                                                # 10 days old rides a 3-day export; 60 days old does not
    r = m.iloc[0]
    assert r.mr_labels == "W2+W4" and r.mr_btc_regime == "STRONG_BULL" and float(r.price) == 1.5 and json.loads(r.mr_entry_fields)["entry_atr_pct"] == 0.7
    assert set(M.ML_MATCHED_EXPORT_COLS) <= set(d.columns)


# ─────────────────────────────── the scout (no network, no replica data: an injected pricer) ───────────────────────────────
def _scout_frames(n_days=12, per_day=3, pct=0.5, label="W2", regime="STRONG_BULL"):
    rows, ords = [], []
    for d in range(n_days):
        for k in range(per_day):
            t = T0 + d * DAY + k * 3_600_000
            rows.append(dict(t=pd.Timestamp(t, unit="ms").strftime("%Y-%m-%dT%H:%M:%S.000"), e="ML_MATCHED", pair=f"P{k}USDT", price=1.0,
                             conf="HIGH", mr_labels=label, mr_family="W", mr_btc_regime=regime, mr_post_gate="",
                             mr_entry_fields=json.dumps({"entry_atr_pct": 0.8})))
        ords.append(dict(opened_at=pd.Timestamp(T0 + d * DAY + 1800_000, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"), closed_at="x", pair="CTLUSDT",
                         direction="LONG", status="CLOSED", entry_strategy="MOMENTUM", entry_price=1.0, entry_atr_pct=0.5, pnl_percentage=0.1,
                         close_reason="STOP_LOSS", cell_multiplier_source="", pattern_cell_source="UNMATCHED",
                         entry_pattern_c_any_match=False, entry_pattern_w_any_match=False, is_paper=True))
    rows.append(dict(rows[0], pair="GATEDUSDT", mr_post_gate="LONG_HEAT_BLOCK"))       # would have been refused anyway → out
    rows.append(dict(rows[0], t="2026-10-01T00:00:00", pair="OLDUSDT"))                # before the floor → out
    return pd.DataFrame(rows), pd.DataFrame(ords)


def _pricer(shadow_pct, ctl_pct=0.05, pending=()):
    def f(fills, now_ms):
        out = {}
        for x in fills:
            st = "PENDING" if any(p in x["key"] for p in pending) else "PRICED"
            out[x["key"]] = dict(pct=(ctl_pct if x["key"].startswith("C:") else shadow_pct(x)), reason="X", exit_ms=x["o_ms"] + 600_000,
                                 src="tick", status=st)
        return out
    return f


def _scout_run(tmp_path, R, O, pricer, **kw):
    import scout_ml_matched_refusals as S
    dec = tmp_path / "dec.csv"; R.to_csv(dec, index=False)
    ordp = tmp_path / "ord.csv"; O.to_csv(ordp, index=False)
    return "\n".join(S.run(now_ms=T0 + 40 * DAY, refusals=S.read_refusals([str(dec)]), orders=S.read_orders([str(ordp)]), floor_ms=T0,
                           price_fn=pricer, fetch=False, state_path=str(tmp_path / "gate.json"), store_path=None, **kw))


def test_scout_operator_call_and_freeze_once(tmp_path):
    R, O = _scout_frames(pct=0.5)
    rng = np.random.default_rng(3)
    txt = _scout_run(tmp_path, R, O, _pricer(lambda x: 0.6 if rng.random() < 0.8 else -0.3))
    assert "excluded post-gate 1" in txt and "W2_CFREE: OPERATOR_CALL" in txt and "W2_CFREE_SBULL: OPERATOR_CALL" in txt
    st = json.load(open(tmp_path / "gate.json"))
    assert st["W2_CFREE"]["verdict"] == "OPERATOR_CALL"
    # decided once: a later run with losing prices keeps the frozen verdict
    txt2 = _scout_run(tmp_path, R, O, _pricer(lambda x: -1.0))
    assert "W2_CFREE: OPERATOR_CALL" in txt2 and "(frozen" in txt2


def test_scout_decided_no_below_control_collecting_and_hold(tmp_path):
    R, O = _scout_frames()
    txt = _scout_run(tmp_path, R, O, _pricer(lambda x: 0.3 if hash(x["key"]) % 10 < 7 else -0.1, ctl_pct=0.9))
    assert "W2_CFREE: DECIDED_NO" in txt and "< same-days control" in txt
    R2, O2 = _scout_frames(n_days=5)
    d2 = tmp_path / "b"; d2.mkdir()
    txt = _scout_run(d2, R2, O2, _pricer(lambda x: 0.5))
    assert "W2_CFREE: COLLECTING" in txt and not os.path.exists(d2 / "gate.json")
    d3 = tmp_path / "c"; d3.mkdir()
    R3, O3 = _scout_frames(n_days=15)
    txt = _scout_run(d3, R3, O3, _pricer(lambda x: 0.5, pending=("P0USDT|" + str(T0 + 2 * DAY),)))   # an early refusal still pending
    assert "W2_CFREE: HOLD" in txt


def test_scout_c_labels_and_regime_split_cells(tmp_path):
    import scout_ml_matched_refusals as S
    assert S.cell_flags("C6+W2", "STRONG_BULL") == {"W2_CFREE": False, "W2_CFREE_SBULL": False}
    assert S.cell_flags("W2+W4", "HEALTHY_BULL") == {"W2_CFREE": True, "W2_CFREE_SBULL": False}
    assert S.cell_flags("W4", "STRONG_BULL") == {"W2_CFREE": False, "W2_CFREE_SBULL": False}


def test_scout_is_exception_safe_and_not_evaluable_without_commit(tmp_path, monkeypatch):
    import scout_ml_matched_refusals as S
    monkeypatch.setattr(S._GK, "dl_deploy_ms", lambda *a, **k: None)
    assert "NOT EVALUABLE" in "\n".join(S.run(refusals=pd.DataFrame(), orders=pd.DataFrame(), fetch=False, state_path=None, store_path=None))

    def boom(*a, **k):
        raise RuntimeError("bad export")
    monkeypatch.setattr(S, "read_refusals", boom)
    out = "\n".join(S.run(floor_ms=T0, fetch=False, state_path=None, store_path=None))
    assert "Unavailable this run" in out


def test_scout_episodes_one_per_pair_and_pending_blocks_the_pair():
    import scout_ml_matched_refusals as S
    df = pd.DataFrame(dict(pair=["A", "A", "A", "B"], o_ms=[0, 100, 900_000, 50], status=["PRICED", "PRICED", "PRICED", "PENDING"],
                           exit_ms=[600_000, 700_000, 1_000_000, None]))
    e = S.episodes(df).set_index(["pair", "o_ms"])                                    # returned in open order
    assert [bool(e.loc[k, "episode"]) for k in (("A", 0), ("A", 100), ("A", 900_000), ("B", 50))] == [True, False, True, False]
    assert [bool(e.loc[k, "waiting"]) for k in (("A", 0), ("A", 100), ("A", 900_000), ("B", 50))] == [False, False, False, True]


# ─────────────────────────────── dual-review fixes (383 review) ───────────────────────────────
class _FakeRegimeClient:
    def __init__(self, fail=None, weight="10", retry_after=None):
        self.calls, self.fail, self.last_response_headers = 0, fail, {"x-mbx-used-weight-1m": weight}
        if retry_after is not None:
            self.last_response_headers["Retry-After"] = str(retry_after)

    async def fapiPublicGetKlines(self, params):
        self.calls += 1
        if self.fail:
            raise self.fail
        return [[0, "1", "1", "1", "1", "1", 0, "5"]]

    async def fapiPublicGetExchangeInfo(self):
        self.calls += 1
        if self.fail:
            raise self.fail
        return {"symbols": [{"symbol": "AUSDT", "contractType": "PERPETUAL", "status": "TRADING", "onboardDate": 1}]}

    async def fapiPublicGetTicker24hr(self):
        self.calls += 1
        return [{"symbol": "AUSDT", "quoteVolume": "100"}]


def _regime_service(monkeypatch, client):
    import services.binance_service as BS
    svc = object.__new__(BS.BinanceService)
    svc.regime_exchange, svc.regime_limited_until, svc.regime_used_weight = client, 0.0, None
    monkeypatch.setattr(BS, "_ban_until", 0)
    return BS, svc


def test_regime_client_backs_off_after_a_rate_limit_and_never_calls_during_it(monkeypatch):
    from ccxt.base.errors import RateLimitExceeded
    cl = _FakeRegimeClient(fail=RateLimitExceeded("429 Too many requests"), retry_after=300)
    BS, svc = _regime_service(monkeypatch, cl)
    t0 = __import__("time").time()
    assert asyncio.run(svc.fetch_regime_klines("AUSDT")) is None and cl.calls == 1
    assert svc.regime_limited() and svc.regime_limited_until >= t0 + 299                      # max(Retry-After 300, 120)
    assert asyncio.run(svc.fetch_regime_klines("AUSDT")) is None and asyncio.run(svc.fetch_regime_universe_inputs()) is None
    assert cl.calls == 1                                                                       # nothing sent during the back-off
    cl2 = _FakeRegimeClient(fail=RuntimeError("binance -1003 way too many requests"))
    BS, svc2 = _regime_service(monkeypatch, cl2)
    asyncio.run(svc2.fetch_regime_universe_inputs())
    assert svc2.regime_limited_until >= t0 + 119                                               # the 120 s floor


def test_regime_client_skips_while_a_ban_is_recorded_and_pauses_on_high_weight(monkeypatch):
    cl = _FakeRegimeClient()
    BS, svc = _regime_service(monkeypatch, cl)
    monkeypatch.setattr(BS, "_ban_until", 1.0)                                                 # an expired ban not yet cleared (POST_BAN_COOLDOWN)
    assert asyncio.run(svc.fetch_regime_klines("AUSDT")) is None and cl.calls == 0
    monkeypatch.setattr(BS, "_ban_until", 0)
    assert asyncio.run(svc.fetch_regime_klines("AUSDT")) and not svc.regime_limited()
    cl.last_response_headers["x-mbx-used-weight-1m"] = "1800"
    assert asyncio.run(svc.fetch_regime_klines("AUSDT"))                                       # this read lands …
    assert svc.regime_limited() and svc.regime_used_weight == 1800                             # … the next ones stand down
    assert asyncio.run(svc.fetch_regime_universe_inputs()) is None and cl.calls == 2


def test_tracker_stops_the_bar_when_the_client_becomes_limited():
    bars, names = _market()
    kl, uni, calls = _fake_fetchers(bars, names)
    state = {"limited": False, "n": 0}

    async def kl2(pair, interval, limit):
        if interval == "5m":
            state["n"] += 1
            if state["n"] >= 5:                          # a 429 on the 5th 5m read
                state["limited"] = True
                return None
        return await kl(pair, interval, limit)
    tr = AC.AltCorrTracker(fetch_klines=kl2, fetch_universe=uni, concurrency=2, is_limited=lambda: state["limited"])
    now = int(bars["BTCUSDT"][0][-1]) + BAR + 1_000
    assert asyncio.run(tr.refresh(now)) is None and tr.value is None
    assert state["n"] <= 6 and tr.skipped_limited >= 15                                       # no fan-out after the 429 (≤ 1 read in flight)
    # limited before the bar → nothing is read at all; the universe rebuild is not attempted either
    tr2 = AC.AltCorrTracker(fetch_klines=kl, fetch_universe=uni, is_limited=lambda: True)
    n0 = calls["k"]; u0 = calls["u"]
    assert asyncio.run(tr2.refresh(now)) is None and calls["k"] == n0 and calls["u"] == u0


def test_universe_rebuild_aborts_when_limited_mid_rebuild_and_keeps_the_old_list(caplog):
    bars, names = _market()
    kl, uni, _ = _fake_fetchers(bars, names)
    st = {"n": 0, "lim": False}

    async def kl2(pair, interval, limit):
        st["n"] += 1
        if st["n"] == 10:
            st["lim"] = True
            return None
        return await kl(pair, interval, limit)
    tr = AC.AltCorrTracker(fetch_klines=kl2, fetch_universe=uni, concurrency=1, is_limited=lambda: st["lim"])
    now = int(bars["BTCUSDT"][0][-1]) + BAR + 1_000
    old = dict(day_ms=AC.universe_day_ms(now) - 3 * DAY, pairs=names[:20], listed={})
    tr.universe = dict(old)
    import logging
    with caplog.at_level(logging.WARNING):
        assert asyncio.run(tr.ensure_universe(now)) == names[:20]                             # the old list stays in use
    assert st["n"] == 10 and tr.universe["day_ms"] == old["day_ms"]                           # stopped at the 429, nothing rebuilt
    assert "universe list is stale" in caplog.text


def test_failed_refusal_insert_frees_its_key_but_a_unique_violation_does_not(monkeypatch):
    import services.trading_engine as TE

    class _Boom:
        def __init__(self, msg): self.msg = msg
        async def __aenter__(self): raise RuntimeError(self.msg)
        async def __aexit__(self, *a): return False
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    import collections
    for msg, kept in (("database is locked", False), ("UNIQUE constraint failed", True)):
        e._mr_seen = collections.OrderedDict({("FOOUSDT", 0): 1})
        monkeypatch.setattr(TE, "AsyncSessionLocal", lambda m=msg: _Boom(m))
        assert asyncio.run(e._ml_refusal_insert(dict(pair="FOOUSDT", bar_at=None, is_paper=True), ("FOOUSDT", 0))) is False
        assert (("FOOUSDT", 0) in e._mr_seen) is kept


def test_refusal_dedupe_is_an_ordered_cap_dropping_the_oldest(monkeypatch):
    import services.trading_engine as TE
    e = object.__new__(TE.TradingEngine); e.is_paper_mode = True
    e.ML_REFUSAL_SEEN_MAX = 3
    monkeypatch.setattr(TE.TradingEngine, "_wh_spawn", lambda self, coro: coro.close())
    for k in range(5):
        assert e._ml_refusal_record(pair=f"P{k}USDT", price=1.0, confidence="HIGH", labels="W2", c_any=False, w_any=True, entry_fields={},
                                    now_ms=T0 + k * BAR)
    assert list(e._mr_seen) == [("P2USDT", T0 + 2 * BAR), ("P3USDT", T0 + 3 * BAR), ("P4USDT", T0 + 4 * BAR)]


def test_scout_persists_not_evaluable_and_never_reprices_it(tmp_path, monkeypatch):
    import scout_ml_matched_refusals as S
    monkeypatch.setattr(S, "REPORTS", str(tmp_path))
    R, O = _scout_frames(n_days=2)
    dec = tmp_path / "dec.csv"; R.to_csv(dec, index=False)
    ordp = tmp_path / "ord.csv"; O.to_csv(ordp, index=False)
    seen = []

    def pf(fills, now_ms):
        seen.extend(f["key"] for f in fills)
        return {f["key"]: dict(pct=None, reason="NO_DATA", exit_ms=None, src="none", status="NOT_EVALUABLE") for f in fills}
    store = str(tmp_path / "store.csv")
    kw = dict(now_ms=T0 + 40 * DAY, refusals=S.read_refusals([str(dec)]), orders=S.read_orders([str(ordp)]), floor_ms=T0, price_fn=pf,
              fetch=False, state_path=str(tmp_path / "g.json"), store_path=store)
    txt = "\n".join(S.run(**kw))
    n1 = len(seen)
    assert n1 > 0 and "not evaluable" in txt and (pd.read_csv(store).status == "NOT_EVALUABLE").all()
    S.run(**kw)
    assert len(seen) == n1                                                                    # terminal: never fetched / priced again


def test_scout_cooldown_never_shortens_an_existing_one(tmp_path):
    import scout_ml_matched_refusals as S
    f = tmp_path / "cd"
    now = 1_000_000_000_000
    f.write_text(str(now + 3_600_000))
    assert S.arm_cooldown(15 * 60_000, path=str(f), now_ms=now) == now + 3_600_000           # the longer existing one wins
    f.write_text(str(now - 1))
    assert S.arm_cooldown(15 * 60_000, path=str(f), now_ms=now) == now + 15 * 60_000
