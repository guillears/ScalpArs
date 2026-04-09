"""
SCALPARS Trading Platform - Database
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

# Create async engine with SQLite concurrency settings
# busy_timeout=30000 (30s) makes SQLite wait for locks instead of failing immediately
# This prevents "database is locked" errors when scan loop and real-time monitor write simultaneously
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
    connect_args={"timeout": 30},
)

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
        await conn.execute(_wal_text("PRAGMA busy_timeout=30000"))
        import logging
        logging.getLogger("database").info("[DB] SQLite WAL mode enabled, busy_timeout=30s")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Auto-migrate: add new columns to existing tables if missing
    async with engine.begin() as conn:
        from sqlalchemy import text, inspect as sa_inspect
        
        def _migrate(connection):
            inspector = sa_inspect(connection)
            if 'orders' in inspector.get_table_names():
                columns = [c['name'] for c in inspector.get_columns('orders')]
                if 'entry_gap' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_gap FLOAT"))
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
                if 'phantom_be_l1_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l1_triggered_at DATETIME"))
                if 'phantom_be_l1_would_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l1_would_exit_pnl FLOAT"))
                if 'phantom_be_l2_triggered_at' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l2_triggered_at DATETIME"))
                if 'phantom_be_l2_would_exit_pnl' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN phantom_be_l2_would_exit_pnl FLOAT"))
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
                if 'exit_slippage_pct' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN exit_slippage_pct FLOAT"))

            if 'transactions' in inspector.get_table_names():
                tx_columns = [c['name'] for c in inspector.get_columns('transactions')]
                if 'order_type' not in tx_columns:
                    connection.execute(text("ALTER TABLE transactions ADD COLUMN order_type VARCHAR(15) DEFAULT 'TAKER'"))

            if 'pair_data' in inspector.get_table_names():
                pd_columns = [c['name'] for c in inspector.get_columns('pair_data')]
                if 'ema5_prev3' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN ema5_prev3 FLOAT"))
                if 'rsi_prev1' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN rsi_prev1 FLOAT"))
                if 'rsi_prev2' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN rsi_prev2 FLOAT"))
                if 'macro_regime' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN macro_regime VARCHAR(10)"))
                if 'volume_ratio' not in pd_columns:
                    connection.execute(text("ALTER TABLE pair_data ADD COLUMN volume_ratio FLOAT"))

            if 'bot_state' in inspector.get_table_names():
                bs_columns = [c['name'] for c in inspector.get_columns('bot_state')]
                if 'ban_until' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN ban_until FLOAT DEFAULT 0.0"))
                if 'paper_bnb_balance_usd' not in bs_columns:
                    connection.execute(text("ALTER TABLE bot_state ADD COLUMN paper_bnb_balance_usd FLOAT DEFAULT 500.0"))

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
        
        await conn.run_sync(_migrate)


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
