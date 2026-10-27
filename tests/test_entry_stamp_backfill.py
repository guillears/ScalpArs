"""🧾 DECISION_LOG 389 — every bot fill records every shared entry stamp (operator 10-27: "make sure all current active sleeves are collecting
ALL data"). After the fill + commit ONE background pass fills the shared entry_* columns a sleeve path left NULL: never overwrites a sleeve's
own value (COALESCE), never the columns an open position's exit / hold logic reads, never MANUAL / replays, never raises. In-memory SQLite,
no network (klines / funding / daily −DI patched). Also pins the FLATBASE scout's frozen OBS-FB1 observe line (zone split)."""
import asyncio
import datetime as dt
import os
import re
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

from test_fade_mcap_stamp import _with_db, _engine  # noqa: E402


def _klines(n=300, p0=1.0):
    t0 = 1_790_000_000_000
    out, p = [], p0
    for i in range(n):
        o = p; p = p * (1.0 + (0.002 if i % 3 else -0.001))
        out.append([t0 + i * 300_000, o, max(o, p) * 1.001, min(o, p) * 0.999, p, 1000.0 + i])
    return out


def _fb_row(**kw):
    import models
    base = dict(pair="BANKUSDT", direction="SHORT", status="OPEN", is_paper=True, entry_strategy="FLATBASE_SHORT", confidence="STRONG_BUY",
                entry_price=1.3, quantity=100.0, investment=100.0, leverage=5, notional_value=500.0,
                opened_at=dt.datetime(2026, 10, 26, 20, 0, 10), entry_atr_pct=None, entry_rsi=None, entry_funding_rate=None,
                entry_frenzy_run_pct=23.5)
    base.update(kw)
    return models.Order(**base)


def _patch_reads(monkeypatch, kl=None, fr=0.00012, ndi=31.5):
    import services.trading_engine as TE

    async def ohlcv(symbol, tf, limit):
        return kl if kl is not None else _klines()

    async def funding(symbol, max_age_s=None):
        return fr

    async def lag(symbol):
        return ndi
    monkeypatch.setattr(TE.binance_service, "get_ohlcv", ohlcv)
    monkeypatch.setattr(TE.binance_service, "fetch_funding_rate", funding)
    monkeypatch.setattr(TE, "_fade_pair_1d_ndi", lag)
    monkeypatch.setattr(TE, "_current_btc_rsi", 55.5)
    monkeypatch.setattr(TE, "_current_btc_adx", 22.0)
    monkeypatch.setattr(TE, "_current_btc_price", 101_000.0)


def _run(monkeypatch, row):
    e = _engine()

    async def fn(ids):
        row.id = ids[0]
        assert e._stamp_backfill_schedule(row) is True
        await asyncio.gather(*list(e._stamp_backfill_tasks))
    _, (got,) = asyncio.run(_with_db(monkeypatch, [row])(fn))
    return got


def test_flatbase_fill_gets_pair_market_and_frenzy_stamps(monkeypatch):
    _patch_reads(monkeypatch)
    got = _run(monkeypatch, _fb_row())
    for c in ("entry_rsi", "entry_adx", "entry_gap", "entry_ema_gap_5_8", "entry_dist_from_ema13_pct", "entry_pos_di", "entry_neg_di",
              "entry_gap_5_20_signed_pct", "entry_pair_volume_ratio", "entry_quality_score"):
        assert getattr(got, c) is not None, c
    assert got.entry_btc_rsi == 55.5 and got.entry_btc_adx == 22.0 and got.entry_btc_price == 101_000.0
    assert got.entry_funding_rate == 0.00012 and got.entry_pair_1d_ndi == 31.5
    assert got.entry_frenzy_di_spread is not None and got.entry_frenzy_vol_trend is not None
    assert got.entry_frenzy_run_pct == 23.5                                  # the sleeve's own stamp untouched


def test_never_overwrites_and_never_touches_exit_columns(monkeypatch):
    _patch_reads(monkeypatch)
    got = _run(monkeypatch, _fb_row(entry_rsi=12.34, entry_funding_rate=-0.0005))
    assert got.entry_rsi == 12.34 and got.entry_funding_rate == -0.0005      # COALESCE: a sleeve value always wins
    assert got.entry_atr_pct is None and got.entry_btc_rsi_closed is None   # exit / hold rulers never appear mid-trade
    assert (got.entry_price, got.quantity, got.investment, got.leverage, got.status) == (1.3, 100.0, 100.0, 5, "OPEN")


def test_reads_fail_soft(monkeypatch):
    import services.trading_engine as TE
    _patch_reads(monkeypatch)

    async def boom(*a, **k):
        raise RuntimeError("418")
    monkeypatch.setattr(TE.binance_service, "get_ohlcv", boom)
    monkeypatch.setattr(TE.binance_service, "fetch_funding_rate", boom)
    got = _run(monkeypatch, _fb_row())
    assert got.entry_rsi is None and got.entry_funding_rate is None          # unreadable → NULL
    assert got.entry_btc_rsi == 55.5                                         # the market snapshot still lands


def test_skips_manual_and_replay(monkeypatch):
    e = _engine()
    assert e._stamp_backfill_schedule(_fb_row(id=1, entry_strategy="MANUAL")) is False
    monkeypatch.setenv("SCALPARS_REPLAY", "1")
    assert e._stamp_backfill_schedule(_fb_row(id=1)) is False


def test_never_set_covers_exit_readers():
    import services.trading_engine as TE
    assert {"entry_atr_pct", "entry_btc_rsi_closed", "entry_btc_regime", "entry_br_door", "entry_price"} <= TE.TradingEngine.STAMP_BACKFILL_NEVER
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    assert src.count("self._stamp_backfill_schedule(order)") == 1           # wired once, in open_position after the commit


def test_backfill_block_is_observe_only():
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    k = src.index("    # 🧾 (DECISION_LOG 389) every bot fill records every entry stamp"); k2 = src.index("    async def _rtp_arm(", k)
    allowed = ("STAMP_BACKFILL_NEVER", "getattr(order, 'entry_price', None)", "entry_price, snap", "if ind and entry_price and float(entry_price) > 0",
               "price=float(entry_price)", "'entry_price', 'entry_fee'")
    for ln in src[k:k2].splitlines():
        if ln.strip().startswith("#"):
            continue
        code = re.sub(r"'entry_[a-z0-9_]+'", "''", ln)              # the STAMP_* column-name lists are data, not trading code
        assert not any(x in code for x in ("_record_filter_block", "investment", "leverage", "quantity")), ln
        if re.search(r"\bentry_price\b(?!_)", ln):
            assert any(a in ln for a in allowed), ln
    assert "await self._stamp_backfill_schedule" not in src and "await self._stamp_backfill(" not in src   # never awaited on the open path


def test_obs_fb1_zone_frozen():
    import scout_flatbase_short as S
    assert (S.OBS_PK_RUN_MAX, S.OBS_PK_OFFPK_MIN, S.OBS_PK_SHADOW) == (67.82, -27.21, "TP1_SL5")
    assert S.obs_pk_zone(67.82, -17.99) == "zone"          # BTR (winner)
    assert S.obs_pk_zone(70.44, -27.21) == "out"           # BMT
    assert S.obs_pk_zone(195.79, -34.77) == "out"          # ORDER
    assert S.obs_pk_zone(None, -5) == "unscored" and S.obs_pk_zone(float("nan"), -5) == "unscored"


# ── 🧾🔒 DECISION_LOG 391: the stamp contract ──────────────────────────────────────────────────────────────────────────────────────

def test_every_entry_column_is_classified():
    """a new entry_* column (or a new sleeve's stamp) cannot ship without deciding: universal / sleeve / booking / legacy."""
    import models
    import services.trading_engine as TE
    E = TE.TradingEngine
    cols = [c.name for c in models.Order.__table__.columns if c.name.startswith("entry_")]
    unc = [c for c in cols if E.stamp_class(c) == "unclassified"]
    assert not unc, f"classify these in TradingEngine.STAMP_* (DECISION_LOG 391): {unc}"
    assert set(E.STAMP_UNIVERSAL) <= set(cols) and set(E.STAMP_BEST_EFFORT) <= set(E.STAMP_UNIVERSAL)
    assert not set(E.STAMP_UNIVERSAL) & (set(E.STAMP_SLEEVE_ONLY) | set(E.STAMP_BOOKING) | set(E.STAMP_LEGACY))


# universal columns the open path's Order(...) constructor stamps for EVERY fill (verified 100 % per sleeve on 10-12 → 10-27 exports);
# a new universal column must be added here or produced by the backfill — the test forces that decision
CTOR_ALWAYS = {
    'entry_pattern_c1_match', 'entry_pattern_c2_match', 'entry_pattern_c3_match', 'entry_pattern_c4_match', 'entry_pattern_c5_match',
    'entry_pattern_c6_match', 'entry_pattern_c7_match', 'entry_pattern_c8_match', 'entry_pattern_c9_match', 'entry_pattern_c_any_match',
    'entry_pattern_w1_match', 'entry_pattern_w2_match', 'entry_pattern_w3_match', 'entry_pattern_w4_match', 'entry_pattern_w5_match',
    'entry_pattern_w6_match', 'entry_pattern_w_any_match', 'entry_slippage_pct', 'entry_desired_notional', 'entry_liquidity_cap_notional',
    'entry_bracket_max_leverage', 'entry_bracket_cap_notional', 'entry_mcap_usd', 'entry_cmc_rank', 'entry_vol_mcap_ratio',
    'entry_alt_corr24', 'entry_alt_beta24', 'entry_btc_off30d_high_pct', 'entry_pair_1h_ema20_200_gap_pct', 'entry_pair_volume_24h_usd',
    'entry_atr_pct', 'entry_btc_regime', 'entry_btc_rsi_closed', 'entry_btc_r72_pct', 'entry_btc_eff72',   # the never-backfilled five
}
BACKFILL_EXPLICIT = {'entry_funding_rate', 'entry_pair_1d_ndi', 'entry_quality_score', 'entry_pair_age_days', 'entry_btc_price',
                     'entry_btc_4h_ema50_200_gap_pct', 'entry_btc_ema50_100_gap_pct', 'entry_eth_5m_ret1_pct', 'entry_btc_1d_ret_pct'}


def test_universal_columns_have_a_source_on_every_path():
    """each universal column has an explicit source: the post-fill backfill (any sleeve; minus the never-written columns) or the open
    path's constructor for every fill (CTOR_ALWAYS, checked as a whole-word kwarg). The runtime truth is scripts/scout_stamp_coverage.py
    + the [STAMP_COVERAGE] log — this pins that no universal column is left without a decided source."""
    import services.trading_engine as TE
    E = TE.TradingEngine

    class T:
        pass
    bf = (set(TE.pair_entry_stamps({}, "LONG")) | set(TE.market_entry_stamps({}, "LONG", None, False, T()))
          | set(TE.monitor_entry_stamps({}, {}, 0)) | BACKFILL_EXPLICIT) - set(E.STAMP_BACKFILL_NEVER) - {'entry_macro_trend'}
    src = open(os.path.join(ROOT, "services", "trading_engine.py"), encoding="utf-8").read()
    k = src.index("    def _stamp_backfill_schedule("); k2 = src.index("    async def _rtp_arm(", k)
    for c in BACKFILL_EXPLICIT:   # the explicit backfill keys are really assigned in the backfill block
        assert re.search(rf"['\"]{c}['\"]\]\s*=|{c}=", src[k:k2]) or c in ("entry_btc_ema50_100_gap_pct", "entry_eth_5m_ret1_pct",
                                                                             "entry_btc_1d_ret_pct"), c
    i = src.index("        order = Order(\n            binance_order_id=binance_order_id,"); j = src.index("        db.add(order)", i)
    mon = set(TE.monitor_entry_stamps({}, {}, 0)) if "**monitor_entry_stamps(" in src[i:j] else set()   # splatted into the constructor
    for c in CTOR_ALWAYS:
        assert c in mon or re.search(rf"\b{c}=(?!=)", src[i:j]), f"{c} is not a kwarg of open_position's Order(...)"
    missing = [c for c in E.STAMP_UNIVERSAL if c not in bf and c not in CTOR_ALWAYS]
    assert not missing, f"universal columns with no decided source: {missing}"


def test_stamp_missing_and_by_design(monkeypatch):
    import services.trading_engine as TE
    e = _engine()
    full = {c: 1 for c in TE.TradingEngine.STAMP_UNIVERSAL}
    assert e.stamp_missing("FRENZY_LITE", full) == []
    assert e.stamp_missing("FRENZY_LITE", dict(full, entry_rsi=None)) == ["entry_rsi"]
    monkeypatch.setattr(TE.TradingEngine, "STAMP_NULL_BY_DESIGN", {"entry_rsi": ("FRENZY_",)})
    assert e.stamp_missing("FRENZY_LITE", dict(full, entry_rsi=None)) == []
    assert e.stamp_missing("SPIKE_FADE", dict(full, entry_rsi=None)) == ["entry_rsi"]


def test_coverage_scout_flags_gaps_and_unclassified(tmp_path):
    import pandas as pd
    import services.trading_engine as TE
    import scout_stamp_coverage as S
    U = TE.TradingEngine.STAMP_UNIVERSAL
    rows = []
    for i in range(4):
        r = {c: 1.0 for c in U}
        r.update(opened_at=f"2026-10-27T0{i}:00:00", pair=f"P{i}USDT", direction="SHORT", entry_strategy="FLATBASE_SHORT", status="CLOSED")
        if i:
            r["entry_rsi"] = None                                              # 25 % coverage → alert
        rows.append(r)
    rows.append(dict({c: 1.0 for c in U}, opened_at="2026-10-27T05:00:00", pair="MUSDT", direction="LONG", entry_strategy="MANUAL", entry_rsi=None))
    for i, (st, cf) in enumerate((("SIGNAL_EXPIRED", "STRONG_BUY"), ("SIGNAL_EXPIRED", "STRONG_BUY"), ("OPEN", "RECOVERED"))):
        rows.append(dict(opened_at=f"2026-10-27T06:0{i}:00", pair=f"X{i}USDT", direction="LONG", status=st, confidence=cf))   # no strategy, no stamps
    p = tmp_path / "scalpars_orders_paper_x.csv"
    pd.DataFrame(rows).assign(entry_new_thing=1).to_csv(p, index=False)
    out = "\n".join(S.run(paths=[str(p)], since_ms=int(pd.Timestamp("2026-10-26").value // 1_000_000)))
    assert "🚨 FLATBASE_SHORT: N 4 · 1 column(s) below the bar: rsi 25%" in out
    assert "UNCLASSIFIED entry_* column(s) in the export — the contract does not know them: entry_new_thing" in out
    assert "- MANUAL" not in out and "- 🚨 MANUAL" not in out and "Verdict: 🚨 ALERT" in out   # MANUAL fills never counted
    assert "MOMENTUM" not in out                                                              # expired / adopted rows are not fills
    assert "① Current batch BEFORE the fix" in out and "② Since the fix" in out                # the pre-fix fills are shown, never hidden
