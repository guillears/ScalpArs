"""
SCALPARS Trading Platform - Database
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True
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
                if 'entry_ema5_stretch' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_ema5_stretch FLOAT"))
                if 'entry_order_type' not in columns:
                    connection.execute(text("ALTER TABLE orders ADD COLUMN entry_order_type VARCHAR(15) DEFAULT 'TAKER'"))

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
