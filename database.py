"""
SCALPARS Trading Platform - Database
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings
import asyncio

# Aug-24 (33): single-process FAIR write queue for SQLite. The Aug-24 22:00 incident: a live
# close burned 5x5s busy-timeouts ("database is locked") while the 1s monitor loop streamed
# small commits — SQLite's busy handler POLLS, so a waiting writer loses every re-acquire race
# to a high-frequency writer. asyncio.Lock wakes waiters FIFO, so routing every commit through
# locked_commit() makes the queue fair: a close waits at most one in-flight commit, never 35s.
# NEVER call locked_commit while holding db_write_lock (it is not reentrant) — the lock must
# only ever be taken inside this helper.
db_write_lock = asyncio.Lock()

async def locked_commit(session) -> None:
    """Commit under the process-wide fair write lock. Use for EVERY commit on the SQLite DB."""
    import time as _t
    _w0 = _t.monotonic()
    async with db_write_lock:
        _wait = _t.monotonic() - _w0
        if _wait > 1.0:
            import logging
            logging.getLogger(__name__).warning(f"[DB_QUEUE_SLOW] commit waited {_wait:.2f}s in the fair write queue")
        await session.commit()

# Create async engine with SQLite concurrency settings.
#
# busy_timeout=5s is an intentional BOUND on how long a writer will wait for the
# SQLite write lock before failing.  The previous value of 60s was chosen to
# maximise the odds of eventually getting through, but it turned lock starvation
# into multi-minute silent stalls because close_position could wait up to
# 60s * 5 attempts = 5 minutes while scan_loop kept hammering the lock with
# tiny per-pair commits (see Apr 11 NEARUSDT incident).
#
# 5s strikes a balance: most commits still complete on the first attempt because
# typical contention windows are sub-second, but worst-case close_position stalls
# are now bounded to ~25s (5 attempts * 5s) instead of ~5 minutes.
#
# Must match the PRAGMA busy_timeout in init_db.
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    connect_args={"timeout": 5},
)

# Aug-25 (36): WRITE-TRANSACTION HOLD TELEMETRY. Three live closes starved 35s+ behind a
# single long-lived write transaction; the commit queue can't see it (the SQLite file lock is
# taken at first WRITE, i.e. flush time, and held until that session commits). These passive
# listeners stamp each connection's first write statement and log [TXN_HOLD] when the write→
# commit span exceeds 3s — naming the holder instead of guessing. Fail-open everywhere.
from sqlalchemy import event as _sa_event
import time as _tx_time
import logging as _tx_logging
_txn_first_write = {}

def _install_txn_telemetry(sync_engine):
    @_sa_event.listens_for(sync_engine, "after_cursor_execute")
    def _stmt(conn, cursor, statement, parameters, context, executemany):
        try:
            _st = (statement or "").lstrip()[:10].upper()
            if _st.startswith(("INSERT", "UPDATE", "DELETE", "REPLACE")):
                k = id(conn)
                if k not in _txn_first_write:
                    _txn_first_write[k] = (_tx_time.monotonic(), (statement or "")[:110])
        except Exception:
            pass

    def _txn_end(conn, kind):
        try:
            v = _txn_first_write.pop(id(conn), None)
            if v is not None:
                _dur = _tx_time.monotonic() - v[0]
                if _dur > 3.0:
                    _tx_logging.getLogger("database").warning(
                        f"[TXN_HOLD] write transaction held {_dur:.1f}s before {kind} — first write: {v[1]}")
        except Exception:
            pass

    @_sa_event.listens_for(sync_engine, "commit")
    def _cm(conn):
        _txn_end(conn, "COMMIT")

    @_sa_event.listens_for(sync_engine, "rollback")
    def _rb(conn):
        _txn_end(conn, "ROLLBACK")

_install_txn_telemetry(engine.sync_engine)

# Async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()


async def init_db():
    """Initialize database tables"""
    # Enable WAL mode for better concurrent read/write performance
    # WAL allows readers and writers to operate simultaneously without blocking
    async with engine.begin() as conn:
        from sqlalchemy import text as _wal_text
        await conn.execute(_wal_text("PRAGMA journal_mode=WAL"))
        await conn.execute(_wal_text("PRAGMA busy_timeout=5000"))
        await conn.execute(_wal_text("PRAGMA synchronous=NORMAL"))
        await conn.execute(_wal_text("PRAGMA wal_autocheckpoint=1000"))
        import logging
        logging.getLogger("database").info("[DB] SQLite WAL mode enabled, busy_timeout=5s, synchronous=NORMAL")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Jul 1, 2026 — one-time cleanup: purge retired phantom-flip sources (idempotent). Only the
    # allowlisted phantoms (LONG_UNMATCHED_ONLY + MOMENTUM_SHORT_W1_REGIME) are collected/displayed
    # now; delete the historical rows of the retired sources so the DB matches the display. Runs every
    # boot but only touches rows once (nothing left to delete after the first run).
    async with engine.begin() as conn:
        from sqlalchemy import text as _pf_text, inspect as _pf_inspect
        def _has_pf(connection):
            return 'phantom_flips' in _pf_inspect(connection).get_table_names()
        try:
            if await conn.run_sync(_has_pf):
                # Jul 5 — REBUILT after the PASS-phantom loss (operator invariant: phantoms die
                # ONLY on /api/reset, NEVER on a redeploy). The old "DELETE ... NOT IN allowlist"
                # was fail-DANGEROUS: any source added to the engine but not to the hardcoded copy
                # was executed on the next boot (Jul-5 incident: 4 PASS phantoms). Now we delete
                # ONLY sources explicitly listed in models.PHANTOM_RETIRED_SOURCES — an unknown or
                # newly-added source ALWAYS survives a deploy, even if every other registry is
                # forgotten. Fail-safe by construction.
                from models import PHANTOM_RETIRED_SOURCES as _pf_retired
                _pf_in = ", ".join(f"'{s}'" for s in _pf_retired)  # trusted literals from models.py
                _res = await conn.execute(_pf_text(
                    f"DELETE FROM phantom_flips WHERE source_filter IN ({_pf_in})"))
                if getattr(_res, 'rowcount', 0):
                    import logging
                    logging.getLogger("database").info(f"[DB] purged {_res.rowcount} retired phantom_flip rows")
        except Exception as _e:
            import logging
            logging.getLogger("database").warning(f"[DB] phantom_flip cleanup skipped: {_e}")

    # Auto-migrate: add new columns to existing tables if missing
    async with engine.begin() as conn:
        from sqlalchemy import text, inspect as sa_inspect
        
        def _migrate(connection):
            inspector = sa_inspect(connection)
            if 'orders' in inspector.get_table_names():
                columns = [c['name'] for c in inspector.get_columns('orders')]
                if 'entry_gap' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_gap FLOAT"))
                if 'backstop_algo_id' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN backstop_algo_id VARCHAR(30)"))
                if 'entry_rsi' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_rsi FLOAT"))
                if 'trough_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN trough_pnl FLOAT DEFAULT 0.0"))
                if 'no_expansion_last_check' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN no_expansion_last_check DATETIME"))
                if 'entry_adx' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_adx FLOAT"))
                if 'entry_macro_trend' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_macro_trend VARCHAR(10)"))
                if 'entry_ema_gap_5_8' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_ema_gap_5_8 FLOAT"))
                if 'entry_ema_gap_8_13' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_ema_gap_8_13 FLOAT"))
                if 'peak_ema5_gap' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN peak_ema5_gap FLOAT DEFAULT 0.0"))
                if 'peak_ema5_dist_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN peak_ema5_dist_pct FLOAT"))
                if 'peak_ema5_slope_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN peak_ema5_slope_pct FLOAT"))
                if 'entry_ema5_stretch' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_ema5_stretch FLOAT"))
                if 'entry_order_type' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_order_type VARCHAR(15) DEFAULT 'TAKER'"))
                if 'exit_order_type' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_order_type VARCHAR(15) DEFAULT 'TAKER'"))
                if 'entry_ema20_slope' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_ema20_slope FLOAT"))
                if 'entry_btc_ema20_slope' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_ema20_slope FLOAT"))
                if 'post_exit_peak_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_peak_pnl FLOAT"))
                if 'post_exit_trough_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_trough_pnl FLOAT"))
                if 'post_exit_peak_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_peak_minutes FLOAT"))
                if 'post_exit_trough_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_trough_minutes FLOAT"))
                if 'post_exit_signal_lost_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_signal_lost_minutes FLOAT"))
                if 'post_exit_pnl_at_signal_lost' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_signal_lost FLOAT"))
                if 'post_exit_final_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_final_pnl FLOAT"))
                if 'post_exit_peak_before_signal_lost' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_peak_before_signal_lost FLOAT"))
                if 'signal_active_at_close' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN signal_active_at_close BOOLEAN"))
                if 'post_exit_rsi_exit_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_rsi_exit_minutes FLOAT"))
                if 'post_exit_rsi_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_rsi_exit_pnl FLOAT"))
                if 'post_exit_rsi3_exit_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_rsi3_exit_minutes FLOAT"))
                if 'post_exit_rsi3_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_rsi3_exit_pnl FLOAT"))
                if 'post_exit_ema13_cross_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_ema13_cross_minutes FLOAT"))
                if 'post_exit_ema13_cross_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_ema13_cross_pnl FLOAT"))
                # May 23: post-exit regime-flip tracker
                if 'post_exit_regime_flip_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_regime_flip_at DATETIME"))
                if 'post_exit_regime_flip_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_regime_flip_pnl_pct FLOAT"))
                if 'post_arm_min_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_arm_min_pnl_pct FLOAT"))
                if 'post_arm_min_pnl_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_arm_min_pnl_at DATETIME"))
                if 'first_rsi2_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_rsi2_pnl FLOAT"))
                if 'first_rsi2_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_rsi2_minutes FLOAT"))
                if 'first_rsi3_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_rsi3_pnl FLOAT"))
                if 'first_rsi3_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_rsi3_minutes FLOAT"))
                if 'entry_adx_prev' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_adx_prev FLOAT"))
                if 'post_exit_signal_regained_minutes' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_signal_regained_minutes FLOAT"))
                if 'post_exit_pnl_at_signal_regained' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_signal_regained FLOAT"))
                if 'post_exit_floor_before_signal_regain' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_floor_before_signal_regain FLOAT"))
                # May 12 LATE PM: time-bucketed post-exit P&L snapshots
                if 'post_exit_pnl_at_1min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_1min FLOAT"))
                if 'post_exit_pnl_at_2min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_2min FLOAT"))
                if 'post_exit_pnl_at_5min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_5min FLOAT"))
                if 'post_exit_pnl_at_15min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_15min FLOAT"))
                if 'post_exit_pnl_at_30min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_pnl_at_30min FLOAT"))
                if 'phantom_be_l1_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l1_triggered_at DATETIME"))
                if 'phantom_be_l1_would_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l1_would_exit_pnl FLOAT"))
                if 'phantom_be_l2_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l2_triggered_at DATETIME"))
                if 'phantom_be_l2_would_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l2_would_exit_pnl FLOAT"))
                if 'phantom_regime_change_exit_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_regime_change_exit_triggered_at DATETIME"))
                if 'phantom_regime_change_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_regime_change_exit_pnl FLOAT"))
                if 'phantom_tick_a_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_a_triggered_at DATETIME"))
                if 'phantom_tick_a_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_a_pnl FLOAT"))
                if 'phantom_tick_b_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_b_triggered_at DATETIME"))
                if 'phantom_tick_b_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_b_pnl FLOAT"))
                if 'phantom_tick_c_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_c_triggered_at DATETIME"))
                if 'phantom_tick_c_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_c_pnl FLOAT"))
                if 'entry_btc_rsi' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_rsi FLOAT"))
                if 'phantom_tick_d_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_d_triggered_at DATETIME"))
                if 'phantom_tick_d_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_d_pnl FLOAT"))
                if 'phantom_tick_e_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_e_triggered_at DATETIME"))
                if 'phantom_tick_e_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_e_pnl FLOAT"))
                if 'phantom_tick_f_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_f_triggered_at DATETIME"))
                if 'phantom_tick_f_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_f_pnl FLOAT"))
                if 'phantom_tick_g_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_g_triggered_at DATETIME"))
                if 'phantom_tick_g_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_tick_g_pnl FLOAT"))
                if 'exit_price_vs_ema5_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_price_vs_ema5_pct FLOAT"))
                if 'exit_ema5_slope_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_ema5_slope_pct FLOAT"))
                if 'exit_ema5_crossed' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_ema5_crossed BOOLEAN"))
                if 'entry_btc_rsi_prev' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_rsi_prev FLOAT"))
                # May 15: pair RSI prev for Pair RSI Direction analytics
                if 'entry_rsi_prev' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_rsi_prev FLOAT"))
                # May 15: BTC RSI 6 candles prior (~30min) for sustained-momentum analytics
                if 'entry_btc_rsi_prev6' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_rsi_prev6 FLOAT"))
                # May 15 PM: BTC Volatility Regime + BTC 1h RSI Direction
                if 'entry_btc_atr_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_atr_pct FLOAT"))
                if 'entry_btc_rsi_1h' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_rsi_1h FLOAT"))
                if 'entry_btc_rsi_1h_prev' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_rsi_1h_prev FLOAT"))
                # May 19, 2026 — Pattern C Tracker (observation-only, 4 patterns × 2 directions)
                if 'entry_pattern_c1_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c1_match BOOLEAN"))
                if 'entry_pattern_c2_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c2_match BOOLEAN"))
                if 'entry_pattern_c3_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c3_match BOOLEAN"))
                if 'entry_pattern_c4_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c4_match BOOLEAN"))
                if 'entry_pattern_c5_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c5_match BOOLEAN"))
                if 'entry_pattern_c6_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c6_match BOOLEAN"))
                if 'entry_pattern_c7_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c7_match BOOLEAN"))
                if 'entry_pattern_c8_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c8_match BOOLEAN"))
                if 'entry_pattern_c9_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c9_match BOOLEAN"))
                if 'entry_pattern_c_any_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_c_any_match BOOLEAN"))
                # May 21: Pattern W computed at entry (lifted from main.py post-hoc helper).
                if 'entry_pattern_w1_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w1_match BOOLEAN"))
                if 'entry_pattern_w2_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w2_match BOOLEAN"))
                if 'entry_pattern_w3_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w3_match BOOLEAN"))
                if 'entry_pattern_w4_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w4_match BOOLEAN"))
                if 'entry_pattern_w5_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w5_match BOOLEAN"))
                if 'entry_pattern_w6_match' not in columns:
                    # May 21 (late) — W6 added: BTC ADX 22-26 + Pair Gap ≤+0.20% (LONG) / BTC ADX ≥32 (SHORT)
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w6_match BOOLEAN"))
                if 'entry_pattern_w_any_match' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pattern_w_any_match BOOLEAN"))
                # May 21: Pattern Cell Ship rule per-trade attribution + override exits.
                if 'pattern_cell_source' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN pattern_cell_source VARCHAR(60)"))
                if 'pattern_fixed_tp_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN pattern_fixed_tp_pct FLOAT"))
                if 'pattern_fixed_sl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN pattern_fixed_sl_pct FLOAT"))
                if 'peak_reached_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN peak_reached_at DATETIME"))
                if 'trough_reached_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN trough_reached_at DATETIME"))
                if 'trough_ema5_dist_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN trough_ema5_dist_pct FLOAT"))
                if 'ema5_went_negative' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN ema5_went_negative VARCHAR(20)"))
                if 'entry_price_vs_ema5_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_price_vs_ema5_pct FLOAT"))
                if 'signal_lost_flagged' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN signal_lost_flagged BOOLEAN DEFAULT 0"))
                if 'signal_lost_flag_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN signal_lost_flag_pnl FLOAT"))
                if 'signal_lost_flagged_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN signal_lost_flagged_at DATETIME"))
                if 'regime_neutral_hit_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN regime_neutral_hit_at DATETIME"))
                if 'regime_neutral_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN regime_neutral_pnl FLOAT"))
                if 'regime_comeback_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN regime_comeback_at DATETIME"))
                if 'regime_comeback_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN regime_comeback_pnl FLOAT"))
                if 'regime_opposite_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN regime_opposite_at DATETIME"))
                if 'regime_opposite_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN regime_opposite_pnl FLOAT"))
                if 'entry_global_volume_ratio' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_global_volume_ratio FLOAT"))
                if 'entry_pair_volume_ratio' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pair_volume_ratio FLOAT"))
                if 'entry_bull_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_bull_pct FLOAT"))
                if 'entry_bear_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_bear_pct FLOAT"))
                if 'entry_range_position' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_range_position FLOAT"))
                if 'entry_adx_delta' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_adx_delta FLOAT"))
                if 'entry_quality_score' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_quality_score FLOAT"))
                if 'entry_btc_regime' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_regime VARCHAR(20)"))
                if 'exit_btc_regime' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_btc_regime VARCHAR(20)"))
                if 'exit_slippage_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_slippage_pct FLOAT"))
                if 'fl1_origin' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN fl1_origin VARCHAR(20)"))
                if 'fl2_flagged' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN fl2_flagged BOOLEAN DEFAULT 0"))
                if 'fl2_flagged_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN fl2_flagged_at DATETIME"))
                if 'fl2_flag_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN fl2_flag_pnl FLOAT"))
                if 'closing_in_progress' not in columns:
                    # Reconciler race guard (Apr 16 — SUIUSDT incident).  Set to
                    # True by the bot before initiating a close on Binance; the
                    # monitor reconciler honours the flag to avoid overwriting
                    # in-flight bot closes with EXTERNAL_CLOSE.
                    connection.execute(text("ALTER TABLE orders ADD COLUMN closing_in_progress BOOLEAN NOT NULL DEFAULT 0"))
                if 'close_initiated_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN close_initiated_at DATETIME"))
                if 'protective_sl_order_id' not in columns:
                    # Broker-side protective stops (Apr 17 — system-down insurance).
                    # Binance order IDs for the STOP_MARKET + TAKE_PROFIT_MARKET
                    # reduceOnly orders placed at position open.  Fire only if
                    # bot's own exits don't run.
                    connection.execute(text("ALTER TABLE orders ADD COLUMN protective_sl_order_id VARCHAR(50)"))
                if 'protective_tp_order_id' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN protective_tp_order_id VARCHAR(50)"))

                # Premium Multiplier (May 4, 2026) — Phase 3 Position Multiplier per CLAUDE.md May 3.
                # Tracks which RSI×ADX cell rule fired and what multiplier was applied.
                # cell_multiplier_capped flags trades where balance forced sub-target investment.
                if 'cell_multiplier' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN cell_multiplier FLOAT NOT NULL DEFAULT 1.0"))
                if 'cell_lev_multiplier' not in columns:
                    # May 21: leverage-side multiplier — added when "both" apply mode shipped.
                    connection.execute(text("ALTER TABLE orders ADD COLUMN cell_lev_multiplier FLOAT NOT NULL DEFAULT 1.0"))
                if 'cell_multiplier_source' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN cell_multiplier_source VARCHAR(40)"))
                if 'cell_multiplier_capped' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN cell_multiplier_capped BOOLEAN NOT NULL DEFAULT 0"))
                # EMA13 strict-mode tracking (May 8): pnl% at first hold
                if 'ema13_strict_held_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN ema13_strict_held_pnl_pct FLOAT"))
                # Post-exit tracking running state (May 8): persisted continuously
                # so a bot restart mid-window doesn't reset captured peak/trough.
                if 'post_exit_running_high' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_running_high FLOAT"))
                if 'post_exit_running_low' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_running_low FLOAT"))
                if 'post_exit_running_peak_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_running_peak_at DATETIME"))
                if 'post_exit_running_trough_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN post_exit_running_trough_at DATETIME"))
                # Trailing pullback confirmation tracking (May 9)
                if 'trailing_first_pullback_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN trailing_first_pullback_pnl_pct FLOAT"))
                if 'trailing_pullback_resets' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN trailing_pullback_resets INTEGER DEFAULT 0"))
                if 'trailing_confirmed_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN trailing_confirmed_at DATETIME"))
                # Regime stability instrumentation (May 5)
                if 'entry_btc_regime_started_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_regime_started_at DATETIME"))
                # BTC Trend Filter diagnostic (May 5)
                if 'entry_btc_trend_gap_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_trend_gap_pct FLOAT"))

                # Exploration Analytics (Apr 28) — observation-only fields for next-batch analysis.
                if 'entry_pos_di' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pos_di FLOAT"))
                if 'entry_neg_di' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_neg_di FLOAT"))
                if 'entry_atr_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_atr_pct FLOAT"))
                if 'entry_ema50_slope' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_ema50_slope FLOAT"))
                if 'entry_funding_rate' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_funding_rate FLOAT"))
                if 'entry_pair_ema20_ema50_gap_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pair_ema20_ema50_gap_pct FLOAT"))
                if 'entry_dist_from_ema13_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_dist_from_ema13_pct FLOAT"))
                if 'entry_btc_dist_from_ema13_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_dist_from_ema13_pct FLOAT"))
                # May 14: BTC 1h EMA20 slope at entry (higher-TF discriminator candidate)
                if 'entry_btc_1h_slope' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_btc_1h_slope FLOAT"))
                # May 14: Phantom BE @ 0.20/0.05 (aggressive observation tracker)
                if 'phantom_be_aggr_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_aggr_triggered_at DATETIME"))
                if 'phantom_be_aggr_would_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_aggr_would_exit_pnl FLOAT"))
                if 'entry_pair_volume_24h_usd' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pair_volume_24h_usd FLOAT"))
                if 'entry_pair_rank' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pair_rank INTEGER"))
                if 'entry_pair_age_days' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_pair_age_days FLOAT"))
                # Jul 27 spike full ship: option-D L2 state columns
                if 'spike_rsi_max' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN spike_rsi_max FLOAT"))
                if 'spike_armed' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN spike_armed BOOLEAN DEFAULT 0"))
                # Jul 28 BE-LOCK SHADOW: per-threshold first-touch minute + post-touch trough
                if 'belock_t15_min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN belock_t15_min FLOAT"))
                if 'belock_tr15' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN belock_tr15 FLOAT"))
                if 'belock_t20_min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN belock_t20_min FLOAT"))
                if 'belock_tr20' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN belock_tr20 FLOAT"))
                if 'belock_t30_min' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN belock_t30_min FLOAT"))
                if 'belock_tr30' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN belock_tr30 FLOAT"))
                if 'exit_pair_ema20_ema50_gap_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_pair_ema20_ema50_gap_pct FLOAT"))
                if 'exit_btc_trend_gap_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_btc_trend_gap_pct FLOAT"))
                # Phase 1 shadow tracking — counterfactual exit at price-vs-EMA cross (May 6)
                if 'first_cross_ema13_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_cross_ema13_at DATETIME"))
                if 'first_cross_ema13_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_cross_ema13_pnl_pct FLOAT"))
                if 'confirmed_cross_ema13_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN confirmed_cross_ema13_at DATETIME"))
                if 'confirmed_cross_ema13_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN confirmed_cross_ema13_pnl_pct FLOAT"))
                if 'first_cross_ema20_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_cross_ema20_at DATETIME"))
                if 'first_cross_ema20_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN first_cross_ema20_pnl_pct FLOAT"))
                if 'confirmed_cross_ema20_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN confirmed_cross_ema20_at DATETIME"))
                if 'confirmed_cross_ema20_pnl_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN confirmed_cross_ema20_pnl_pct FLOAT"))
                # ===== LEASH SHADOW START (May 30) — observation-only virtual leashes =====
                for _sc in ('shadow_tight_pnl','shadow_wide_pnl','shadow_tierA_pnl','shadow_tierB_pnl',
                            'shadow_strpk_pnl','shadow_stren_pnl','shadow_peak_stretch',
                            'shadow_tight_min','shadow_wide_min','shadow_tierA_min','shadow_tierB_min',
                            'shadow_strpk_min','shadow_stren_min',
                            'shadow_strpk04_pnl','shadow_strpk04_min','shadow_strpk03_pnl','shadow_strpk03_min',
                            'shadow_strpk_signed_pnl','shadow_strpk_signed_min',  # Jun 1: signed-stretch leash variant
                            'runner_peak_stretch',  # Jun 1: live runner stretch-trail peak tracking
                            'shadow_peak_stretch_at_close',  # Jun 16: shadow peak snapshotted at live close (sampling vs post-exit diagnostic)
                            'shadow_atr05_pnl','shadow_atr05_min',  # Jun 16: ATR-floor give-back shadows (chandelier) N=0.5/1.0/1.5
                            'shadow_atr10_pnl','shadow_atr10_min',
                            'shadow_atr15_pnl','shadow_atr15_min',
                            'shadow_cap025_pnl','shadow_cap025_min',  # Jun 17 PM: give-back-cap shadows (frac 0.25/0.35/0.50)
                            'shadow_cap035_pnl','shadow_cap035_min',
                            'shadow_cap050_pnl','shadow_cap050_min',
                            'shadow_arm035_pnl','shadow_arm035_min',  # Jul 6: arm-level shadows (arm 0.35/0.40, trail 0.25, pre-0.45 tracking)
                            'shadow_arm040_pnl','shadow_arm040_min'):
                    if _sc not in columns:
                        connection.execute(text(f"ALTER TABLE orders ADD COLUMN {_sc} FLOAT"))
                for _sc in ('shadow_tight_reason','shadow_wide_reason','shadow_tierA_reason','shadow_tierB_reason',
                            'shadow_strpk_reason','shadow_stren_reason',
                            'shadow_strpk04_reason','shadow_strpk03_reason','shadow_strpk_signed_reason',
                            'runner_trail_bound'):  # Jun 17: lock/atr/stretch — which mechanism bound the runner-trail exit
                    if _sc not in columns:
                        connection.execute(text(f"ALTER TABLE orders ADD COLUMN {_sc} VARCHAR(15)"))
                # ===== LEASH SHADOW END =====
                # Liquidity-aware sizing observability (Jun 2, 2026)
                if 'entry_desired_notional' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_desired_notional FLOAT"))
                if 'entry_liquidity_cap_notional' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_liquidity_cap_notional FLOAT"))
                if 'liquidity_capped' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN liquidity_capped BOOLEAN DEFAULT 0"))
                if 'entry_slippage_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_slippage_pct FLOAT"))
                # Aug 21 gate 57: Bull-Run Monitor readings at entry (BULLRUN_LONG fills)
                if 'entry_br_r72' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_br_r72 FLOAT"))
                if 'entry_br_above' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_br_above FLOAT"))
                if 'entry_br_eff' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_br_eff FLOAT"))
                if 'entry_br_off24h' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_br_off24h FLOAT"))
                if 'entry_br_door' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_br_door VARCHAR(8)"))
                if 'funding_fee_usd' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN funding_fee_usd FLOAT"))
                # Jun 7: phantom EMA13 cross (records would-have-exited pnl when EMA13
                # cross exit is DISABLED for that direction — observation CF).
                if 'phantom_ema13_cross_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_ema13_cross_pnl FLOAT"))
                if 'phantom_ema13_cross_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_ema13_cross_at DATETIME"))
                # Jun 8: trailing min-profit gate phantom CF
                if 'phantom_trail_suppress_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_trail_suppress_pnl FLOAT"))
                if 'phantom_trail_suppress_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_trail_suppress_at DATETIME"))
                # Jun 8: gap-expanding relaxation A/B cohort tag
                if 'entry_gap_expand_marginal' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_gap_expand_marginal BOOLEAN"))
                # Jun 14: Flip Entry sleeve — strategy tag (MOMENTUM vs FLIP:<SOURCE>)
                if 'entry_strategy' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_strategy VARCHAR(40)"))
                # Jul 22: HARD_TP mechanism shadow (leash A / ladder B) — tick-honest CF
                if 'hard_tp_shadow_leash_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN hard_tp_shadow_leash_pnl FLOAT"))
                if 'hard_tp_shadow_leash_fired' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN hard_tp_shadow_leash_fired BOOLEAN"))
                if 'hard_tp_shadow_ladder_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN hard_tp_shadow_ladder_pnl FLOAT"))
                if 'hard_tp_shadow_ladder_fired' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN hard_tp_shadow_ladder_fired BOOLEAN"))

            # Aug 21 gate 57: monitor_periods ledger — last_update (added after the table shipped)
            if 'monitor_periods' in inspector.get_table_names():
                _mp_cols = [c['name'] for c in inspector.get_columns('monitor_periods')]
                if 'last_update' not in _mp_cols:
                    connection.execute(text("ALTER TABLE monitor_periods ADD COLUMN last_update DATETIME"))
                if 'blocked_off24h' not in _mp_cols:
                    connection.execute(text("ALTER TABLE monitor_periods ADD COLUMN blocked_off24h INTEGER DEFAULT 0"))
                if 'blocked_ema13' not in _mp_cols:
                    connection.execute(text("ALTER TABLE monitor_periods ADD COLUMN blocked_ema13 INTEGER DEFAULT 0"))
                if 'blocked_pvr' not in _mp_cols:
                    connection.execute(text("ALTER TABLE monitor_periods ADD COLUMN blocked_pvr INTEGER DEFAULT 0"))
                if 'blocked_1h' not in _mp_cols:
                    connection.execute(text("ALTER TABLE monitor_periods ADD COLUMN blocked_1h INTEGER DEFAULT 0"))
            # Aug-22: investors.eth_wallet (optional payout address)
            if 'investors' in inspector.get_table_names():
                _inv_cols = [c['name'] for c in inspector.get_columns('investors')]
                if 'eth_wallet' not in _inv_cols:
                    connection.execute(text("ALTER TABLE investors ADD COLUMN eth_wallet VARCHAR(42)"))
            if 'transactions' in inspector.get_table_names():
                tx_columns = [c['name'] for c in inspector.get_columns('transactions')]
                if 'order_type' not in tx_columns:
                    connection.execute(text("ALTER TABLE transactions ADD COLUMN order_type VARCHAR(15) DEFAULT 'TAKER'"))

            if 'pair_data' in inspector.get_table_names():
                pd_columns = [c['name'] for c in inspector.get_columns('pair_data')]
                if 'ema5_prev3' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN ema5_prev3 FLOAT"))
                if 'ema20_prev3' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN ema20_prev3 FLOAT"))
                if 'rsi_prev1' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN rsi_prev1 FLOAT"))
                if 'rsi_prev2' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN rsi_prev2 FLOAT"))
                if 'macro_regime' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN macro_regime VARCHAR(10)"))
                if 'volume_ratio' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN volume_ratio FLOAT"))
                if 'ema50' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN ema50 FLOAT"))

            if 'bot_state' in inspector.get_table_names():
                bs_columns = [c['name'] for c in inspector.get_columns('bot_state')]
                if 'ban_until' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN ban_until FLOAT DEFAULT 0.0"))
                if 'paper_bnb_balance_usd' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN paper_bnb_balance_usd FLOAT DEFAULT 500.0"))
                if 'runtime_initial_total_usd' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN runtime_initial_total_usd FLOAT"))
                if 'current_btc_regime' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN current_btc_regime VARCHAR(20)"))
                if 'btc_regime_started_at' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN btc_regime_started_at DATETIME"))
                if 'filter_block_counts_json' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN filter_block_counts_json TEXT"))
                if 'filter_funnel_v2_json' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN filter_funnel_v2_json TEXT"))
                if 'last_bnb_check_at' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN last_bnb_check_at DATETIME"))

            # Jun 14 — Phantom Flip cohort sub-division (matched-long fade by C/W family)
            if 'phantom_flips' in inspector.get_table_names():
                pf_columns = [c['name'] for c in inspector.get_columns('phantom_flips')]
                if 'entry_cohort' not in pf_columns:
                    connection.execute(text("ALTER TABLE phantom_flips ADD COLUMN entry_cohort VARCHAR(8)"))
                # Jun 15 — full entry-indicator capture on phantoms (RSI/ATR/fan-ratio/regime
                # parity with the flip Order) so the phantom POOL is analyzable cross-batch.
                _pf_float = [
                    'entry_gap',
                    'entry_rsi', 'entry_rsi_prev', 'entry_adx', 'entry_adx_prev', 'entry_adx_delta',
                    'entry_pos_di', 'entry_neg_di', 'entry_ema_gap_5_8', 'entry_ema_gap_8_13',
                    'entry_ema5_stretch', 'entry_price_vs_ema5_pct', 'entry_atr_pct',
                    'entry_pair_ema20_ema50_gap_pct', 'entry_dist_from_ema13_pct', 'entry_range_position',
                    'entry_btc_adx', 'entry_btc_rsi', 'entry_btc_ema20_slope', 'entry_btc_1h_slope',
                    'entry_btc_dist_from_ema13_pct',
                    'entry_btc_trend_gap_pct',  # Jul 8 — flip depth gate (flip_short_btc_trend_gap_min) surface
                    # full-parity round 2 (Jun 15)
                    'entry_ema20_slope', 'entry_ema50_slope', 'entry_global_volume_ratio',
                    'entry_pair_volume_ratio', 'entry_bull_pct', 'entry_bear_pct', 'entry_pair_volume_24h_usd',
                    'entry_quality_score',  # Float to match Order.entry_quality_score
                    # full-parity round 3 (Jun 15) — BTC prev/higher-TF companions so flips feed
                    # the "by BTC ... Direction / Volatility / 1h RSI" tables (they compare cur vs prev)
                    'entry_btc_adx_prev', 'entry_btc_rsi_prev', 'entry_btc_rsi_prev6',
                    'entry_btc_atr_pct', 'entry_btc_rsi_1h', 'entry_btc_rsi_1h_prev',
                ]
                for _col in _pf_float:
                    if _col not in pf_columns:
                        connection.execute(text(f"ALTER TABLE phantom_flips ADD COLUMN {_col} FLOAT"))
                for _col in ('entry_pair_rank',):
                    if _col not in pf_columns:
                        connection.execute(text(f"ALTER TABLE phantom_flips ADD COLUMN {_col} INTEGER"))
                for _col in ('entry_macro_trend', 'entry_btc_regime'):
                    if _col not in pf_columns:
                        connection.execute(text(f"ALTER TABLE phantom_flips ADD COLUMN {_col} VARCHAR(20)"))

            if 'investors' not in inspector.get_table_names():
                connection.execute(text("""
                    CREATE TABLE investors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name VARCHAR(100) NOT NULL UNIQUE,
                        shares FLOAT NOT NULL DEFAULT 0.0,
                        total_deposited FLOAT NOT NULL DEFAULT 0.0,
                        total_withdrawn FLOAT NOT NULL DEFAULT 0.0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
            # Jun 1, 2026 — dated cash-flow ledger per investor
            if 'investor_ledger' not in inspector.get_table_names():
                connection.execute(text("""
                    CREATE TABLE investor_ledger (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        investor_id INTEGER NOT NULL,
                        type VARCHAR(12) NOT NULL,
                        amount FLOAT NOT NULL,
                        nav_at_time FLOAT,
                        shares_delta FLOAT,
                        note VARCHAR(200),
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_investor_ledger_investor_id ON investor_ledger (investor_id)"
                ))
            # Jul 3, 2026 — investors table: enforce AUTOINCREMENT (never reuse a deleted
            # investor's id — recycled ids inherited the old id's ledger history). Rebuild
            # once (SQLite can't ALTER-in AUTOINCREMENT), then seed sqlite_sequence past
            # every id ever seen in investor_ledger so no historical ledger row can ever
            # re-attach to a future investor.
            if 'investors' in inspector.get_table_names():
                _inv_sql = connection.execute(text(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='investors'"
                )).scalar() or ''
                if 'AUTOINCREMENT' not in _inv_sql.upper():
                    connection.execute(text("""
                        CREATE TABLE investors_new (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name VARCHAR(100) NOT NULL UNIQUE,
                            shares FLOAT NOT NULL DEFAULT 0.0,
                            total_deposited FLOAT NOT NULL DEFAULT 0.0,
                            total_withdrawn FLOAT NOT NULL DEFAULT 0.0,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            eth_wallet VARCHAR(42)
                        )
                    """))
                    # eth_wallet is guaranteed present here: its ALTER guard runs earlier in this function (Aug-22)
                    connection.execute(text(
                        "INSERT INTO investors_new (id, name, shares, total_deposited, total_withdrawn, created_at, eth_wallet) "
                        "SELECT id, name, shares, total_deposited, total_withdrawn, created_at, eth_wallet FROM investors"
                    ))
                    connection.execute(text("DROP TABLE investors"))
                    connection.execute(text("ALTER TABLE investors_new RENAME TO investors"))
                    _max_seen = connection.execute(text(
                        "SELECT MAX(m) FROM (SELECT COALESCE(MAX(id),0) AS m FROM investors "
                        "UNION ALL SELECT COALESCE(MAX(investor_id),0) FROM investor_ledger)"
                    )).scalar() or 0
                    connection.execute(text("DELETE FROM sqlite_sequence WHERE name='investors'"))
                    connection.execute(text(
                        "INSERT INTO sqlite_sequence (name, seq) VALUES ('investors', :s)"
                    ), {"s": int(_max_seen)})
            # Jul 2, 2026 — daily NAV/share snapshots (deposit-proof performance record)
            if 'nav_snapshots' not in inspector.get_table_names():
                connection.execute(text("""
                    CREATE TABLE nav_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date VARCHAR(10) NOT NULL UNIQUE,
                        portfolio_value FLOAT NOT NULL,
                        total_shares FLOAT NOT NULL,
                        nav_per_share FLOAT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                connection.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_nav_snapshots_date ON nav_snapshots (date)"
                ))

        await conn.run_sync(_migrate)

    # ────────────────────────────────────────────────────────────────────
    # One-off data migration (May 17, 2026): rename BREAKEVEN_SL_L* close_reason
    # values to BREAKEVEN_EXIT_L* (the close is a BE floor exit, not a stop loss).
    # Idempotent: matches only the legacy prefix.
    # ────────────────────────────────────────────────────────────────────
    async with engine.begin() as conn:
        def _rename_breakeven(connection):
            inspector = sa_inspect(connection)
            if 'orders' in inspector.get_table_names():
                result = connection.execute(text(
                    "UPDATE orders SET close_reason = 'BREAKEVEN_EXIT' || substr(close_reason, 13) "
                    "WHERE close_reason LIKE 'BREAKEVEN_SL%'"
                ))
                if result.rowcount > 0:
                    print(f"[DB_MIGRATE] Renamed {result.rowcount} BREAKEVEN_SL_L* rows -> BREAKEVEN_EXIT_L*")
        await conn.run_sync(_rename_breakeven)

    # ────────────────────────────────────────────────────────────────────
    # One-off data migration: backfill legacy CHOPPY rows with the new
    # CHOPPY_WEAK / CHOPPY_FLAT sub-labels.  Idempotent: skips if any row
    # already has a sub-label (meaning we've already backfilled once).
    #
    # For entry_btc_regime we have the stored entry_btc_adx and
    # entry_btc_ema20_slope, so reclassification is ground-truth.
    #
    # For exit_btc_regime we do NOT store exit ADX/slope historically —
    # so we use the entry values as a proxy.  This is reasonable because
    # the bot's trades are short (typically <1h) and BTC rarely flips
    # sub-regime within that window.  Slightly inaccurate for long-held
    # trades, but keeps within-trade consistency so the Regime Transition
    # Impact analysis doesn't treat every legacy CHOPPY trade as a shift.
    # ────────────────────────────────────────────────────────────────────
    async with engine.begin() as conn:
        from sqlalchemy import text

        def _backfill_choppy_split(connection):
            import logging
            _log = logging.getLogger("database")

            # Idempotency guard
            already = connection.execute(text(
                "SELECT COUNT(*) FROM orders "
                "WHERE entry_btc_regime IN ('CHOPPY_WEAK', 'CHOPPY_FLAT') "
                "   OR exit_btc_regime IN ('CHOPPY_WEAK', 'CHOPPY_FLAT')"
            )).scalar()
            if already and already > 0:
                _log.info(f"[MIGRATION] CHOPPY split backfill: already migrated ({already} sub-label rows present), skipping")
                return

            rows = connection.execute(text(
                "SELECT id, entry_btc_adx, entry_btc_ema20_slope, "
                "       entry_btc_regime, exit_btc_regime "
                "FROM orders "
                "WHERE (entry_btc_regime = 'CHOPPY' OR exit_btc_regime = 'CHOPPY') "
                "  AND entry_btc_adx IS NOT NULL "
                "  AND entry_btc_ema20_slope IS NOT NULL"
            )).fetchall()

            if not rows:
                _log.info("[MIGRATION] CHOPPY split backfill: no legacy CHOPPY rows to update")
                return

            updated = 0
            skipped = 0
            for row in rows:
                adx = row.entry_btc_adx
                slope = row.entry_btc_ema20_slope

                # Mirror classify_btc_regime's CHOPPY logic
                if adx < 18:
                    new_label = "CHOPPY_WEAK"
                elif abs(slope) < 0.02:
                    new_label = "CHOPPY_FLAT"
                else:
                    # Row is tagged CHOPPY but the values don't fit either
                    # sub-bucket — shouldn't happen given the old classifier,
                    # but skip defensively.
                    skipped += 1
                    continue

                new_entry = new_label if row.entry_btc_regime == "CHOPPY" else row.entry_btc_regime
                new_exit = new_label if row.exit_btc_regime == "CHOPPY" else row.exit_btc_regime

                connection.execute(
                    text("UPDATE orders SET entry_btc_regime = :entry, exit_btc_regime = :exit WHERE id = :oid"),
                    {"entry": new_entry, "exit": new_exit, "oid": row.id}
                )
                updated += 1

            _log.info(
                f"[MIGRATION] CHOPPY split backfill: updated {updated} orders "
                f"(skipped {skipped} defensively; exit_btc_regime uses entry ADX/slope as proxy)"
            )

        await conn.run_sync(_backfill_choppy_split)

        # Aug-25 one-shot repair (operator-approved "ship the fix"): live DOGE order 5's close
        # was booked with YESTERDAY'S fill (0.08995) by the unbounded fill lookup during the
        # 02:50 lock cascade — the real fill was 0.09219 (EXIT_SLIPPAGE log line). pnl −160.08
        # → −35.51 (same fee formula: gross −32.27 − entry 0.9291 − exit 2.3082). The strict
        # WHERE (all the old wrong values) makes this a no-op everywhere except that exact row,
        # and a no-op again on every later boot.
        from sqlalchemy import text as _sql_text
        await conn.execute(_sql_text(
            "UPDATE orders SET exit_price=0.09219, exit_fee=2.3082, total_fee=3.2373, "
            "pnl=-35.51, pnl_percentage=-0.69 "
            "WHERE id=5 AND pair='DOGEUSDT' AND is_paper=0 AND exit_price=0.08995 "
            "AND pnl < -159.0 AND pnl > -161.0"))

        # Aug-25 one-shot repair #2 (operator-approved "ok fix it"): live PROM order 7. The
        # 02:50 failed open left a 1,355-unit GHOST leg (filled on Binance, DB insert died in
        # the lock cascade); the reconciler closed its remainder at 03:00:27 for -83.80,
        # invisible to the books. Binance income (REALIZED_PNL, full episode): -74.62 vs booked
        # +16.17. Row updated to the full-episode truth; pct = -74.62/(112.18*20) = -3.33%.
        # Also DOGE order 5's phantom +2.4086% exit-slippage reading (computed vs the stale
        # fill) becomes the real -0.0216. Guarded on old wrong values = no-op after first boot.
        await conn.execute(_sql_text(
            "UPDATE orders SET pnl=-74.62, pnl_percentage=-3.33, "
            "notes='Aug-25 repair: includes ghost leg (failed-open duplicate; reconciler-closed 03:00:27 at -83.80; Binance income ground truth)' "
            "WHERE id=7 AND pair='PROMUSDT' AND is_paper=0 "
            "AND pnl > 16.0 AND pnl < 16.3"))
        await conn.execute(_sql_text(
            "UPDATE orders SET exit_slippage_pct=-0.0216 "
            "WHERE id=5 AND pair='DOGEUSDT' AND is_paper=0 "
            "AND exit_slippage_pct > 2.0"))

        # Aug-25 one-shot repair #3: live HYPE order 16 — maker entry partially filled (1.0 of
        # 77.31, remainder canceled at the 20s timeout) but investment/notional booked the
        # INTENDED size; quantity was already correct so P&L math was never wrong. Real:
        # notional = 1.0 x 81.023 = 81.02, investment = 81.02/20 = 4.05. Guarded on the old
        # wrong values (qty 1.0, investment > 300) = no-op after first boot.
        await conn.execute(_sql_text(
            "UPDATE orders SET investment=4.0512, notional_value=81.023, entry_desired_notional=81.023 "
            "WHERE id=16 AND pair='HYPEUSDT' AND is_paper=0 AND quantity=1.0 AND investment > 300"))


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await locked_commit(session)  # per-request implicit commit joins the fair write queue
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
