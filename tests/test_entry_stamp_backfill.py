"""🧾 DECISION_LOG 389 — every bot fill records every shared entry stamp (operator 10-27: "make sure all current active sleeves are collecting
ALL data"). After the fill + commit ONE background pass fills the shared entry_* columns a sleeve path left NULL: never overwrites a sleeve's
own value (COALESCE), never the columns an open position's exit / hold logic reads, never MANUAL / replays, never raises. In-memory SQLite,
no network (klines / funding / daily −DI patched). Also pins the FLATBASE scout's frozen OBS-FB1 observe line (zone split)."""
import asyncio
import datetime as dt
import os
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
        assert not any(x in ln for x in ("_record_filter_block", "investment", "leverage", "quantity")), ln
        if "entry_price" in ln:
            assert any(a in ln for a in allowed), ln
    assert "await self._stamp_backfill_schedule" not in src and "await self._stamp_backfill(" not in src   # never awaited on the open path


def test_obs_fb1_zone_frozen():
    import scout_flatbase_short as S
    assert (S.OBS_PK_RUN_MAX, S.OBS_PK_OFFPK_MIN, S.OBS_PK_SHADOW) == (67.82, -27.21, "TP1_SL5")
    assert S.obs_pk_zone(67.82, -17.99) == "zone"          # BTR (winner)
    assert S.obs_pk_zone(70.44, -27.21) == "out"           # BMT
    assert S.obs_pk_zone(195.79, -34.77) == "out"          # ORDER
    assert S.obs_pk_zone(None, -5) == "unscored" and S.obs_pk_zone(float("nan"), -5) == "unscored"
