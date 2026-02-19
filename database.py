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
