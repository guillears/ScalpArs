"""
SCALPARS Trading Platform - Main Application
"""
import asyncio
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from pydantic import BaseModel
from sqlalchemy import select, and_, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database import init_db, get_db, AsyncSessionLocal
from models import Order, Transaction, BotState, PairData, ConfigChangeLog
import config
from config import (
    trading_config, save_trading_config, load_trading_config,
    TradingConfig, ConfidenceConfig, SignalThresholds, InvestmentConfig
)
from services.binance_service import binance_service, set_ban_until, set_ban_persist_callback, get_ban_status
from services.trading_engine import trading_engine, realtime_stop_loss_callback
from services.indicators import calculate_indicators, get_signal
from services.websocket_tracker import websocket_tracker

import os
import logging

# Setup logging - configure root logger to show all our messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Set level for all our loggers
logging.getLogger("services").setLevel(logging.INFO)
logger = logging.getLogger("scalpars")

# Background task control
_scan_task = None
_monitor_task = None
should_stop = False
_scan_lock = asyncio.Lock()


async def monitor_loop():
    """Fast loop (1s) for SL/TP checks and cache updates.
    
    Runs independently of scan_loop so stop-loss monitoring is never
    blocked by slow API calls in scan_and_trade.
    """
    global should_stop
    logger.info("[MONITOR] Monitor loop started (1s cycle)")
    while not should_stop:
        try:
            async with AsyncSessionLocal() as db:
                await trading_engine.initialize(db)
                await trading_engine.update_open_positions(db)
                await trading_engine.update_orders_cache(db)
                await trading_engine.update_post_exit_tracking(db)
        except Exception as e:
            logger.error(f"[ERROR] Error in monitor loop: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(1)


async def scan_loop():
    """Slow loop for scanning pairs and opening new positions.
    
    Runs in its own task so it never blocks the fast monitor_loop.
    Uses exponential backoff on failures to avoid hammering Binance
    after rate-limit bans.
    """
    global should_stop
    logger.info("[SCAN_LOOP] Waiting 10s before first scan (post-deploy cooldown)...")
    await asyncio.sleep(10)
    logger.info("[SCAN_LOOP] Scan loop started")
    scan_backoff = 1
    while not should_stop:
        try:
            async with AsyncSessionLocal() as db:
                await trading_engine.initialize(db)
                if trading_engine.is_running:
                    async with _scan_lock:
                        await trading_engine.scan_and_trade(db)
                    scan_backoff = 1
        except Exception as e:
            logger.error(f"[ERROR] Error in scan loop: {e}")
            import traceback
            traceback.print_exc()
            scan_backoff = min(scan_backoff * 2, 300)
            logger.warning(f"[SCAN_LOOP] Backing off {scan_backoff}s after error")
        await asyncio.sleep(scan_backoff)


async def start_background_tasks():
    """Start background trading tasks"""
    global _scan_task, _monitor_task, should_stop
    should_stop = False
    
    # Start WebSocket tracker for real-time price tracking
    await websocket_tracker.start()
    logger.info("[STARTUP] WebSocket price tracker started")
    
    # Register real-time stop loss callback
    websocket_tracker.set_price_callback(realtime_stop_loss_callback)
    logger.info("[STARTUP] Real-time stop loss callback registered")

    async def has_open_orders() -> bool:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(func.count(Order.id)).where(Order.status == "OPEN"))
            return (result.scalar() or 0) > 0

    websocket_tracker.set_open_orders_callback(has_open_orders)
    logger.info("[STARTUP] Open-orders callback registered for WS staleness check")
    
    _monitor_task = asyncio.create_task(monitor_loop())
    _scan_task = asyncio.create_task(scan_loop())
    logger.info("[STARTUP] Monitor and scan loops started independently")


async def stop_background_tasks():
    """Stop background trading tasks"""
    global _scan_task, _monitor_task, should_stop
    should_stop = True
    
    # Stop WebSocket tracker
    await websocket_tracker.stop()
    logger.info("[SHUTDOWN] WebSocket price tracker stopped")
    
    for task in (_monitor_task, _scan_task):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("[STARTUP] Initializing database...")
    await init_db()
    
    # Clear any stale ban state from DB on startup (fresh deploy = fresh IP)
    async with AsyncSessionLocal() as db:
        ban_row = await db.execute(select(BotState).limit(1))
        bot_state = ban_row.scalar_one_or_none()
        if bot_state and bot_state.ban_until and bot_state.ban_until > 0:
            bot_state.ban_until = 0
            await db.commit()
            logger.info("[STARTUP] Cleared stale ban state from DB (fresh deploy)")
    
    # Register callback to persist ban state to DB when detected
    def _persist_ban(ban_epoch: float):
        import asyncio as _aio
        async def _save():
            async with AsyncSessionLocal() as _db:
                result = await _db.execute(select(BotState).limit(1))
                state = result.scalar_one_or_none()
                if state:
                    state.ban_until = ban_epoch
                    await _db.commit()
                    logger.info(f"[BINANCE] Ban state persisted to DB: expires at {ban_epoch:.0f}")
        try:
            loop = _aio.get_running_loop()
            loop.create_task(_save())
        except RuntimeError:
            pass
    set_ban_persist_callback(_persist_ban)
    
    logger.info("[STARTUP] Starting background tasks...")
    await start_background_tasks()
    
    # Subscribe to WebSocket for any existing open orders and restore bot state
    async with AsyncSessionLocal() as db:
        # Initialize trading engine state from database
        await trading_engine.initialize(db)
        
        # If bot was running before shutdown, set fresh started_at for accurate timing
        if trading_engine.is_running and trading_engine.started_at is None:
            trading_engine.started_at = datetime.utcnow()
            await trading_engine.save_state(db)
            logger.info(f"[STARTUP] Resumed running bot with saved runtime: {trading_engine.total_runtime_seconds}s")
        
        result = await db.execute(
            select(Order).where(Order.status == "OPEN")
        )
        open_orders = result.scalars().all()
        for order in open_orders:
            # Subscribe without resetting - preserve any existing tracking data
            # The order's low_price_since_entry/high_price_since_entry are stored in DB
            # We don't want to reset WebSocket tracker on server restart
            await websocket_tracker.subscribe_pair(order.pair)  # No initial_price = no reset
        if open_orders:
            logger.info(f"[STARTUP] Subscribed to {len(open_orders)} open order pairs for WebSocket tracking (preserved existing tracking)")
    
    logger.info("[STARTUP] SCALPARS Trading Platform started")
    yield
    # Shutdown
    logger.info("[SHUTDOWN] Stopping background tasks...")
    await stop_background_tasks()
    
    # Save runtime before shutdown (so it persists across server restarts)
    async with AsyncSessionLocal() as db:
        if trading_engine.is_running and trading_engine.started_at:
            # Add elapsed time to total and clear started_at
            elapsed = (datetime.utcnow() - trading_engine.started_at).total_seconds()
            trading_engine.total_runtime_seconds += int(elapsed)
            trading_engine.started_at = None
            await trading_engine.save_state(db)
            logger.info(f"[SHUTDOWN] Saved runtime: {trading_engine.total_runtime_seconds}s")
    
    await binance_service.close()
    logger.info("[SHUTDOWN] SCALPARS Trading Platform stopped")


# Create FastAPI app
app = FastAPI(
    title="SCALPARS Trading Platform",
    description="Automated crypto futures trading bot",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# ============== Pydantic Models ==============

class ConfigUpdate(BaseModel):
    trading_fee: Optional[float] = None
    maker_fee: Optional[float] = None
    taker_fee: Optional[float] = None
    maker_entry_enabled: Optional[bool] = None
    maker_timeout_seconds: Optional[int] = None
    maker_offset_ticks: Optional[int] = None
    maker_exit_enabled: Optional[bool] = None
    maker_exit_timeout_seconds: Optional[int] = None
    maker_exit_offset_ticks: Optional[int] = None
    paper_trading: Optional[bool] = None
    paper_balance: Optional[float] = None
    investment: Optional[Dict] = None
    thresholds: Optional[Dict] = None
    confidence_levels: Optional[Dict] = None


class ManualCloseRequest(BaseModel):
    order_id: int


# ============== API Routes ==============

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main page"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


# ----- Bot Status -----

@app.get("/api/status")
async def get_status(db: AsyncSession = Depends(get_db)):
    """Get bot status"""
    await trading_engine.initialize(db)
    return trading_engine.get_status()


@app.post("/api/start")
async def start_bot(db: AsyncSession = Depends(get_db)):
    """Start the trading bot"""
    result = await trading_engine.start(db)
    ban = get_ban_status()
    if ban["banned"]:
        result["ban_warning"] = f"Binance IP ban active, {ban['remaining_seconds']}s remaining. Bot will wait before scanning."
    return result


@app.post("/api/pause")
async def pause_bot(db: AsyncSession = Depends(get_db)):
    """Pause the trading bot"""
    result = await trading_engine.pause(db)
    return result


@app.post("/api/paper-mode")
async def set_paper_mode(enabled: bool, db: AsyncSession = Depends(get_db)):
    """Toggle paper trading mode"""
    result = await trading_engine.set_paper_mode(enabled, db)
    return result


@app.post("/api/reset")
async def reset_paper_trading(db: AsyncSession = Depends(get_db)):
    """Reset paper trading - clear all data and start fresh with $10,000"""
    import config
    
    # Stop the bot first if running
    if trading_engine.is_running:
        await trading_engine.pause(db)
    
    # Delete all paper trading orders
    await db.execute(
        delete(Order).where(Order.is_paper == True)
    )
    
    # Delete all paper trading transactions
    await db.execute(
        delete(Transaction).where(Transaction.is_paper == True)
    )
    
    # Reset bot state
    trading_engine.paper_balance = config.trading_config.paper_balance  # Default 10,000
    trading_engine.total_runtime_seconds = 0
    trading_engine.started_at = None
    trading_engine.is_running = False
    
    # Clear any ban state
    set_ban_until(0)
    
    # Save the reset state
    await trading_engine.save_state(db)
    
    # Clear ban in DB
    result_state = await db.execute(select(BotState).limit(1))
    state_row = result_state.scalar_one_or_none()
    if state_row:
        state_row.ban_until = 0
    
    await db.commit()
    
    logger.info(f"[RESET] Paper trading reset. Balance: ${trading_engine.paper_balance}, Timer: 00:00:00")
    
    return {
        "success": True,
        "message": "Paper trading reset successfully",
        "paper_balance": trading_engine.paper_balance,
        "runtime": "00:00:00"
    }


# ----- Balance -----

@app.get("/api/balance")
async def get_balance(db: AsyncSession = Depends(get_db)):
    """Get account balance"""
    await trading_engine.initialize(db)
    
    if trading_engine.is_paper_mode:
        # Always recalculate from DB to prevent stale/inflated values
        balance = await trading_engine._recalculate_paper_balance(db)
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        open_orders = result.scalars().all()
        used_margin = sum(o.investment for o in open_orders)
        
        return {
            "usdt_balance": balance,
            "bnb_balance": 0,
            "usdt_in_orders": used_margin,
            "total_portfolio": balance + used_margin,
            "is_paper": True
        }
    else:
        balance = await binance_service.get_balance()
        return {
            "usdt_balance": balance['usdt_free'],
            "bnb_balance": balance['bnb_total'],
            "usdt_in_orders": balance['usdt_used'],
            "total_portfolio": balance['total_portfolio'],
            "is_paper": False
        }


# ----- Market Data -----

@app.get("/api/pairs")
async def get_pairs(db: AsyncSession = Depends(get_db), limit: int = 50):
    """Get top pairs with indicators"""
    # Validate limit
    limit = min(max(limit, 5), 100)
    
    # Get pairs from cache
    result = await db.execute(
        select(PairData).order_by(desc(PairData.volume_24h)).limit(limit)
    )
    pairs = result.scalars().all()
    
    # Get open position counts per pair
    positions_result = await db.execute(
        select(Order.pair, Order.direction, func.count(Order.id))
        .where(and_(Order.status == "OPEN", Order.is_paper == trading_engine.is_paper_mode))
        .group_by(Order.pair, Order.direction)
    )
    positions = {(row[0], row[1]): row[2] for row in positions_result}
    
    pairs_data = []
    for p in pairs:
        long_count = positions.get((p.pair, "LONG"), 0)
        short_count = positions.get((p.pair, "SHORT"), 0)
        
        # Use real-time WebSocket price instead of stale OHLCV close
        ws_tracker = websocket_tracker.trackers.get(p.pair)
        if ws_tracker and ws_tracker.last_price and ws_tracker.last_price > 0:
            display_price = ws_tracker.last_price
        else:
            display_price = p.price
        
        # Calculate gap using OHLCV price (matches indicator logic)
        gap = None
        if p.ema5 and p.ema20 and p.price and p.price > 0:
            gap = round(((p.ema5 - p.ema20) / p.price) * 100, 4)
        
        pairs_data.append({
            "pair": p.pair,
            "price": display_price,
            "ema5": round(p.ema5, 2) if p.ema5 else None,
            "ema8": round(p.ema8, 2) if p.ema8 else None,
            "ema13": round(p.ema13, 2) if p.ema13 else None,
            "ema20": round(p.ema20, 2) if p.ema20 else None,
            "gap": gap,
            "rsi": round(p.rsi, 2) if p.rsi else None,
            "adx": round(p.adx, 2) if p.adx else None,
            "signal": p.signal,
            "confidence": p.confidence,
            "macro_regime": p.macro_regime,
            "volume_24h": p.volume_24h,
            "open_positions": {
                "long": long_count,
                "short": short_count
            }
        })
    
    return pairs_data


@app.post("/api/pairs/refresh")
async def refresh_pairs(db: AsyncSession = Depends(get_db)):
    """Force refresh pair data. Skips Binance calls if data is already fresh."""
    from services.trading_engine import OHLCV_BATCH_SIZE, OHLCV_BATCH_DELAY

    freshness = await db.execute(
        select(func.max(PairData.updated_at))
    )
    latest_update = freshness.scalar_one_or_none()
    if latest_update and (datetime.utcnow() - latest_update).total_seconds() < 30:
        count_result = await db.execute(select(func.count(PairData.pair)))
        cached_count = count_result.scalar() or 0
        logger.info(f"[REFRESH] Data is fresh ({(datetime.utcnow() - latest_update).total_seconds():.0f}s old), skipping Binance API calls")
        return {"status": "cached", "count": cached_count}

    if _scan_lock.locked():
        logger.info("[REFRESH] Scan already in progress, skipping duplicate API calls")
        return {"status": "scan_in_progress", "count": 0}

    async with _scan_lock:
        top_pairs = await binance_service.get_top_futures_pairs(config.trading_config.trading_pairs_limit)

        for batch_start in range(0, len(top_pairs), OHLCV_BATCH_SIZE):
            batch = top_pairs[batch_start:batch_start + OHLCV_BATCH_SIZE]

            for pair_info in batch:
                symbol = pair_info['symbol']
                pair = pair_info['pair']
                volume_24h = pair_info['volume_24h']

                ohlcv = await binance_service.get_ohlcv(symbol, '5m', 100)
                if not ohlcv:
                    continue

                indicators = calculate_indicators(ohlcv)
                if not indicators:
                    continue

                signal, confidence = get_signal(
                    ema5=indicators.get('ema5'),
                    ema8=indicators.get('ema8'),
                    ema13=indicators.get('ema13'),
                    ema20=indicators.get('ema20'),
                    rsi=indicators.get('rsi'),
                    adx=indicators.get('adx'),
                    volume=indicators.get('volume'),
                    avg_volume=indicators.get('avg_volume'),
                    price=indicators.get('close')
                )

                await trading_engine.update_pair_data(db, pair, indicators, signal, confidence, volume_24h)

            if batch_start + OHLCV_BATCH_SIZE < len(top_pairs):
                await asyncio.sleep(OHLCV_BATCH_DELAY)

    return {"status": "refreshed", "count": len(top_pairs)}


def _compute_be_level(order) -> dict:
    """Determine which BE protection tier is active based on peak P&L and confidence config."""
    tc = config.trading_config
    conf = order.confidence or "LOW"
    conf_config = tc.confidence_levels.get(conf, tc.confidence_levels.get("LOW"))

    peak = order.peak_pnl or 0
    l5_trigger = getattr(conf_config, 'be_level5_trigger', 999)
    l4_trigger = getattr(conf_config, 'be_level4_trigger', 999)
    l3_trigger = getattr(conf_config, 'be_level3_trigger', 999)
    l2_trigger = getattr(conf_config, 'be_level2_trigger', 999)
    l1_trigger = getattr(conf_config, 'be_level1_trigger', 999)

    if peak >= l5_trigger:
        return {"be_level": 5, "be_stop": getattr(conf_config, 'be_level5_offset', 0)}
    elif peak >= l4_trigger:
        return {"be_level": 4, "be_stop": getattr(conf_config, 'be_level4_offset', 0)}
    elif peak >= l3_trigger:
        return {"be_level": 3, "be_stop": getattr(conf_config, 'be_level3_offset', 0)}
    elif peak >= l2_trigger:
        return {"be_level": 2, "be_stop": getattr(conf_config, 'be_level2_offset', 0)}
    elif peak >= l1_trigger:
        return {"be_level": 1, "be_stop": getattr(conf_config, 'be_level1_offset', 0)}
    return {"be_level": 0, "be_stop": 0}


# ----- Orders -----

@app.get("/api/orders/open")
async def get_open_orders(db: AsyncSession = Depends(get_db)):
    """Get all open orders"""
    await trading_engine.initialize(db)
    
    result = await db.execute(
        select(Order)
        .where(and_(Order.status == "OPEN", Order.is_paper == trading_engine.is_paper_mode))
        .order_by(desc(Order.opened_at))
    )
    orders = result.scalars().all()
    
    orders_data = []
    for o in orders:
        # Use real-time WebSocket price instead of stale DB value
        ws_tracker = websocket_tracker.trackers.get(o.pair)
        if ws_tracker and ws_tracker.last_price and ws_tracker.last_price > 0:
            current_price = ws_tracker.last_price
        else:
            current_price = o.current_price
        
        # Calculate current P&L (including both entry and estimated exit fees)
        if current_price and o.entry_price:
            # Estimate exit fee based on current notional value
            current_notional = current_price * o.quantity
            entry_notional = o.entry_price * o.quantity
            estimated_exit_fee = current_notional * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
            total_fees = o.entry_fee + estimated_exit_fee
            
            if o.direction == "LONG":
                pnl = (current_price - o.entry_price) * o.quantity - total_fees
                # P&L percentage as % of notional (not investment)
                pnl_pct = (pnl / entry_notional) * 100
            else:
                pnl = (o.entry_price - current_price) * o.quantity - total_fees
                # P&L percentage as % of notional (not investment)
                pnl_pct = (pnl / entry_notional) * 100
        else:
            pnl = 0
            pnl_pct = 0
            estimated_exit_fee = 0
        
        # Calculate duration
        duration_seconds = (datetime.utcnow() - o.opened_at).total_seconds()
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        # Calculate PRICE drop from peak (this is what triggers the pullback/trailing stop)
        if o.direction == "LONG" and o.high_price_since_entry and o.high_price_since_entry > 0:
            drop_from_peak = ((o.high_price_since_entry - (current_price or o.entry_price)) / 
                            o.high_price_since_entry * 100)
        elif o.direction == "SHORT" and o.low_price_since_entry and o.low_price_since_entry > 0:
            drop_from_peak = (((current_price or o.entry_price) - o.low_price_since_entry) / 
                            o.low_price_since_entry * 100)
        else:
            drop_from_peak = 0
        
        orders_data.append({
            "id": o.id,
            "pair": o.pair,
            "direction": o.direction,
            "confidence": o.confidence,
            "investment": round(o.investment, 2),
            "leverage": o.leverage,
            "notional_value": round(o.notional_value, 2),
            "quantity": o.quantity,
            "entry_price": o.entry_price,
            "current_price": current_price,
            "entry_fee": round(o.entry_fee, 4),
            "estimated_exit_fee": round(estimated_exit_fee, 4),
            "total_fees": round(o.entry_fee + estimated_exit_fee, 4),
            "high_since_entry": o.high_price_since_entry,
            "low_since_entry": o.low_price_since_entry,
            "drop_from_peak": round(drop_from_peak, 2),
            "peak_pnl": round(o.peak_pnl, 2),
            "pnl": round(pnl, 2),
            "pnl_percentage": round(pnl_pct, 2),
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "duration_seconds": duration_seconds,
            "opened_at": o.opened_at.isoformat(),
            "tp_level": o.current_tp_level or 1,
            "tp_target": o.dynamic_tp_target or 0,
            **_compute_be_level(o),
            "entry_order_type": getattr(o, 'entry_order_type', None) or "TAKER",
            "exit_order_type": getattr(o, 'exit_order_type', None) or "TAKER"
        })

    return orders_data


@app.get("/api/orders/closed")
async def get_closed_orders(db: AsyncSession = Depends(get_db)):
    """Get all closed orders"""
    await trading_engine.initialize(db)
    
    result = await db.execute(
        select(Order)
        .where(and_(Order.status == "CLOSED", Order.is_paper == trading_engine.is_paper_mode))
        .order_by(desc(Order.closed_at))
        .limit(100)
    )
    orders = result.scalars().all()
    
    orders_data = []
    for o in orders:
        duration_seconds = (o.closed_at - o.opened_at).total_seconds() if o.closed_at else 0
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        orders_data.append({
            "id": o.id,
            "pair": o.pair,
            "direction": o.direction,
            "confidence": o.confidence,
            "investment": round(o.investment, 2),
            "leverage": o.leverage,
            "notional_value": round(o.notional_value, 2),
            "quantity": o.quantity,
            "entry_price": o.entry_price,
            "exit_price": o.exit_price,
            "entry_fee": round(o.entry_fee, 4),
            "exit_fee": round(o.exit_fee or 0, 4),
            "total_fee": round(o.total_fee or 0, 4),
            "pnl": round(o.pnl or 0, 2),
            "pnl_percentage": round(o.pnl_percentage or 0, 2),
            "peak_pnl": round(o.peak_pnl or 0, 2),
            "close_reason": o.close_reason,
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "opened_at": o.opened_at.isoformat(),
            "closed_at": o.closed_at.isoformat() if o.closed_at else None,
            "tp_level": o.current_tp_level or 1,
            **_compute_be_level(o),
            "entry_order_type": getattr(o, 'entry_order_type', None) or "TAKER",
            "exit_order_type": getattr(o, 'exit_order_type', None) or "TAKER",
            "post_exit_peak_minutes": o.post_exit_peak_minutes,
            "post_exit_trough_minutes": o.post_exit_trough_minutes,
            "post_exit_final_pnl": o.post_exit_final_pnl,
            "post_exit_signal_lost_minutes": o.post_exit_signal_lost_minutes,
        })

    return orders_data


@app.get("/api/transactions")
async def get_transactions(db: AsyncSession = Depends(get_db)):
    """Get all transactions"""
    await trading_engine.initialize(db)
    
    # Join with Order to get confidence
    result = await db.execute(
        select(Transaction, Order.confidence)
        .outerjoin(Order, Transaction.order_id == Order.id)
        .where(Transaction.is_paper == trading_engine.is_paper_mode)
        .order_by(desc(Transaction.timestamp))
        .limit(200)
    )
    rows = result.all()
    
    return [{
        "id": t.id,
        "order_id": t.order_id,
        "binance_order_id": t.binance_order_id,
        "pair": t.pair,
        "action": t.action,
        "price": t.price,
        "quantity": t.quantity,
        "investment": round(t.investment, 2),
        "leverage": t.leverage,
        "notional_value": round(t.notional_value, 2),
        "fee": round(t.fee, 4),
        "order_type": getattr(t, 'order_type', None) or "TAKER",
        "timestamp": t.timestamp.isoformat(),
        "confidence": confidence
    } for t, confidence in rows]


@app.post("/api/orders/close")
async def close_order(request: ManualCloseRequest, db: AsyncSession = Depends(get_db)):
    """Manually close an order"""
    result = await db.execute(
        select(Order).where(Order.id == request.order_id)
    )
    order = result.scalar_one_or_none()
    
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    if order.status != "OPEN":
        raise HTTPException(status_code=400, detail="Order is not open")
    
    # Get current price
    symbol = order.pair.replace('USDT', '/USDT:USDT')
    current_price = await binance_service.get_current_price(symbol)
    
    if current_price <= 0:
        raise HTTPException(status_code=500, detail="Could not fetch current price")
    
    closed_order = await trading_engine.close_position(db, order, current_price, "MANUAL")
    
    if not closed_order:
        raise HTTPException(status_code=500, detail="Failed to close order")
    
    return {"status": "closed", "pnl": closed_order.pnl}


# ----- Performance Metrics -----

@app.get("/api/performance")
async def get_performance(regime: str = None, db: AsyncSession = Depends(get_db)):
    """Get closed orders performance metrics, optionally filtered by macro trend regime"""
    await trading_engine.initialize(db)
    
    try:
        return await _compute_performance(db, regime=regime)
    except Exception as e:
        logger.error(f"[PERF] Unhandled error in get_performance: {e}\n{traceback.format_exc()}")
        return {
            "total_trades": 0, "total_longs": 0, "total_shorts": 0,
            "total_wins": 0, "total_losses": 0,
            "win_rate": 0, "win_rate_longs": 0, "win_rate_shorts": 0,
            "avg_win": 0, "avg_win_long": 0, "avg_win_short": 0,
            "avg_loss": 0, "avg_loss_long": 0, "avg_loss_short": 0,
            "expectancy": 0,
            "best_win_long": 0, "best_win_short": 0,
            "worst_loss_long": 0, "worst_loss_short": 0,
            "total_pnl": 0, "total_pnl_percentage": 0,
            "total_pnl_notional_percentage": 0,
            "total_investment_notional": 0, "total_investment_value": 0,
            "total_investment_long_notional": 0, "total_investment_long_value": 0,
            "total_investment_short_notional": 0, "total_investment_short_value": 0,
            "total_fees": 0,
            "avg_duration": "00:00:00", "avg_duration_long": "00:00:00", "avg_duration_short": "00:00:00",
            "avg_leverage": 0, "return_multiple": 0, "daily_compound_return": 0,
            "runtime_days": 0,
            "by_confidence": {}, "by_macro_trend": {}, "outcome_distribution": [],
            "gap_performance": [], "ema58_gap_performance": [], "rsi_performance": [], "adx_performance": [], "adx_direction_performance": [], "stretch_performance": [],
            "pair_slope_performance": [], "btc_slope_performance": [], "btc_adx_performance": [], "btc_adx_direction_performance": [], "adx_dir_crosstab": [], "btc_slope_adx_crosstab": [],
            "by_close_reason": {},
            "stop_loss_deep_dive": {"total_sl_trades": 0, "be_was_active": {"count": 0}, "positive_no_be": {"count": 0}, "never_positive": {"count": 0}, "avg_peak_all_sl": 0},
            "winning_trades_drawdown": [], "trough_recovery": [],
            "never_positive_deep_dive": [],
            "performance_over_time": [],
            "post_exit_regret_deep_dive": [],
            "hold_time_expectancy": [],
            "entry_conditions_by_reason": [],
            "_error": str(e)
        }


def _compute_entry_type_stats(orders):
    """Compute performance breakdown by entry order type (MAKER / TAKER / TAKER_FALLBACK)."""
    taker_fee_rate = getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
    groups = {}
    for o in orders:
        etype = getattr(o, 'entry_order_type', None) or "TAKER"
        if etype not in groups:
            groups[etype] = {"trades": [], "pnl_sum": 0, "fee_sum": 0, "wins": 0, "by_confidence": {}}
        groups[etype]["trades"].append(o)
        groups[etype]["pnl_sum"] += (o.pnl or 0)
        groups[etype]["fee_sum"] += (o.total_fee or 0)
        if (o.pnl or 0) > 0:
            groups[etype]["wins"] += 1
        conf = getattr(o, 'confidence', 'UNKNOWN') or 'UNKNOWN'
        groups[etype]["by_confidence"][conf] = groups[etype]["by_confidence"].get(conf, 0) + 1

    total_trades = len(orders)
    result = {}
    for etype, data in groups.items():
        count = len(data["trades"])
        avg_entry_fee = sum(o.entry_fee or 0 for o in data["trades"]) / count if count else 0
        # Fee savings: difference vs what fees would be if entry was also taker
        if etype == "MAKER":
            hypothetical_taker_entry_fees = sum(
                (o.entry_price * o.quantity * taker_fee_rate) for o in data["trades"]
            )
            actual_entry_fees = sum(o.entry_fee or 0 for o in data["trades"])
            fee_savings = hypothetical_taker_entry_fees - actual_entry_fees
        else:
            fee_savings = 0

        result[etype] = {
            "trades": count,
            "pct_of_total": round(count / total_trades * 100, 1) if total_trades else 0,
            "win_rate": round(data["wins"] / count * 100, 1) if count else 0,
            "avg_pnl_pct": round(sum(o.pnl_percentage or 0 for o in data["trades"]) / count, 2) if count else 0,
            "avg_pnl_usd": round(data["pnl_sum"] / count, 2) if count else 0,
            "total_pnl": round(data["pnl_sum"], 2),
            "avg_entry_fee": round(avg_entry_fee, 4),
            "total_fees": round(data["fee_sum"], 2),
            "fee_savings": round(fee_savings, 2),
            "by_confidence": data["by_confidence"],
        }
    return result


def _compute_exit_type_stats(orders):
    """Compute performance breakdown by exit order type (MAKER / TAKER / TAKER_FALLBACK) with close reason."""
    taker_fee_rate = getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
    groups = {}
    for o in orders:
        etype = getattr(o, 'exit_order_type', None) or "TAKER"
        if etype not in groups:
            groups[etype] = {"trades": [], "pnl_sum": 0, "fee_sum": 0, "wins": 0, "close_reasons": {}, "by_confidence": {}}
        groups[etype]["trades"].append(o)
        groups[etype]["pnl_sum"] += (o.pnl or 0)
        groups[etype]["fee_sum"] += (o.total_fee or 0)
        if (o.pnl or 0) > 0:
            groups[etype]["wins"] += 1
        cr = (o.close_reason or "UNKNOWN").split(" ")[0]
        groups[etype]["close_reasons"][cr] = groups[etype]["close_reasons"].get(cr, 0) + 1
        conf = getattr(o, 'confidence', 'UNKNOWN') or 'UNKNOWN'
        groups[etype]["by_confidence"][conf] = groups[etype]["by_confidence"].get(conf, 0) + 1

    total_trades = len(orders)
    result = {}
    for etype, data in groups.items():
        count = len(data["trades"])
        avg_exit_fee = sum(o.exit_fee or 0 for o in data["trades"]) / count if count else 0
        if etype == "MAKER":
            hypothetical_taker_exit_fees = sum(
                ((o.exit_price or o.entry_price) * o.quantity * taker_fee_rate) for o in data["trades"]
            )
            actual_exit_fees = sum(o.exit_fee or 0 for o in data["trades"])
            fee_savings = hypothetical_taker_exit_fees - actual_exit_fees
        else:
            fee_savings = 0

        result[etype] = {
            "trades": count,
            "pct_of_total": round(count / total_trades * 100, 1) if total_trades else 0,
            "win_rate": round(data["wins"] / count * 100, 1) if count else 0,
            "avg_pnl_pct": round(sum(o.pnl_percentage or 0 for o in data["trades"]) / count, 2) if count else 0,
            "avg_pnl_usd": round(data["pnl_sum"] / count, 2) if count else 0,
            "total_pnl": round(data["pnl_sum"], 2),
            "avg_exit_fee": round(avg_exit_fee, 4),
            "total_fees": round(data["fee_sum"], 2),
            "fee_savings": round(fee_savings, 2),
            "by_close_reason": data["close_reasons"],
            "by_confidence": data["by_confidence"],
        }
    return result


def _compute_time_buckets(orders, bucket_minutes=15):
    """Group closed trades into time buckets and compute per-bucket stats."""
    from datetime import timezone
    UTC_MINUS_3 = timezone(timedelta(hours=-3))
    try:
        timed = [(o, o.closed_at.replace(tzinfo=timezone.utc) if o.closed_at and o.closed_at.tzinfo is None else o.closed_at)
                 for o in orders if o.closed_at is not None]
        if not timed:
            return []
        timed.sort(key=lambda x: x[1])
        first_time = timed[0][1]
        buckets = {}
        for o, ct in timed:
            mins_since_start = (ct - first_time).total_seconds() / 60
            bucket_idx = int(mins_since_start // bucket_minutes)
            bucket_key = first_time + timedelta(minutes=bucket_idx * bucket_minutes)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(o)
        result = []
        cumulative_pnl = 0
        for bk in sorted(buckets.keys()):
            bucket_orders = buckets[bk]
            count = len(bucket_orders)
            pnl_sum = sum(o.pnl or 0 for o in bucket_orders)
            cumulative_pnl += pnl_sum
            wins = sum(1 for o in bucket_orders if (o.pnl or 0) > 0)
            longs = sum(1 for o in bucket_orders if o.direction == "LONG")
            shorts = count - longs
            rsis = [o.entry_rsi for o in bucket_orders if o.entry_rsi is not None]
            gaps = [o.entry_gap for o in bucket_orders if o.entry_gap is not None]
            adxs = [o.entry_adx for o in bucket_orders if o.entry_adx is not None]
            gaps58 = [o.entry_ema_gap_5_8 for o in bucket_orders if o.entry_ema_gap_5_8 is not None]
            local_time = bk.astimezone(UTC_MINUS_3)
            result.append({
                "time": local_time.strftime("%H:%M"),
                "trades": count,
                "longs": longs,
                "shorts": shorts,
                "wins": wins,
                "win_rate": round(wins / count * 100, 1) if count > 0 else 0,
                "pnl": round(pnl_sum, 2),
                "avg_pnl": round(pnl_sum / count, 2) if count > 0 else 0,
                "cumulative_pnl": round(cumulative_pnl, 2),
                "avg_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "avg_gap58": round(sum(gaps58) / len(gaps58), 4) if gaps58 else None,
                "avg_adx": round(sum(adxs) / len(adxs), 1) if adxs else None
            })
        return result
    except Exception as e:
        logger.error(f"[PERF] Error computing time buckets: {e}\n{traceback.format_exc()}")
        return []


async def _compute_performance(db: AsyncSession, regime: str = None):
    result = await db.execute(
        select(Order)
        .where(and_(Order.status == "CLOSED", Order.is_paper == trading_engine.is_paper_mode))
    )
    all_orders = result.scalars().all()
    
    # Compute by_macro_trend summary from ALL orders (before filtering)
    macro_trend_performance = {}
    for trend in ['BULLISH', 'BEARISH', 'NEUTRAL']:
        trend_orders = [o for o in all_orders if (o.entry_macro_trend or 'NEUTRAL') == trend]
        trend_longs = [o for o in trend_orders if o.direction == "LONG"]
        trend_shorts = [o for o in trend_orders if o.direction == "SHORT"]
        trend_long_pnl = sum(o.pnl or 0 for o in trend_longs)
        trend_short_pnl = sum(o.pnl or 0 for o in trend_shorts)
        trend_long_inv = sum(o.investment for o in trend_longs)
        trend_short_inv = sum(o.investment for o in trend_shorts)
        trend_long_wins = len([o for o in trend_longs if (o.pnl or 0) > 0])
        trend_short_wins = len([o for o in trend_shorts if (o.pnl or 0) > 0])
        trend_total_pnl = trend_long_pnl + trend_short_pnl
        trend_total_wins = trend_long_wins + trend_short_wins
        trend_total_wr = round(trend_total_wins / len(trend_orders) * 100, 2) if trend_orders else 0
        t_wins_list = [o for o in trend_orders if (o.pnl or 0) > 0]
        t_losses_list = [o for o in trend_orders if (o.pnl or 0) <= 0]
        t_avg_win = sum(o.pnl for o in t_wins_list) / len(t_wins_list) if t_wins_list else 0
        t_avg_loss = sum(o.pnl for o in t_losses_list) / len(t_losses_list) if t_losses_list else 0
        t_rr = round(t_avg_win / abs(t_avg_loss), 2) if t_avg_loss != 0 else 0
        macro_trend_performance[trend] = {
            "total_trades": len(trend_orders),
            "long_trades": len(trend_longs),
            "short_trades": len(trend_shorts),
            "total_pnl": round(trend_total_pnl, 2),
            "total_win_rate": trend_total_wr,
            "risk_reward": t_rr,
            "long_pnl": round(trend_long_pnl, 2),
            "short_pnl": round(trend_short_pnl, 2),
            "long_pnl_pct": round(trend_long_pnl / trend_long_inv * 100, 2) if trend_long_inv > 0 else 0,
            "short_pnl_pct": round(trend_short_pnl / trend_short_inv * 100, 2) if trend_short_inv > 0 else 0,
            "long_win_rate": round(trend_long_wins / len(trend_longs) * 100, 2) if trend_longs else 0,
            "short_win_rate": round(trend_short_wins / len(trend_shorts) * 100, 2) if trend_shorts else 0,
        }
    
    # Apply regime filter if requested
    if regime and regime in ('BULLISH', 'BEARISH', 'NEUTRAL'):
        orders = [o for o in all_orders if (o.entry_macro_trend or 'NEUTRAL') == regime]
    else:
        orders = all_orders
    
    if not orders:
        early_open_result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == trading_engine.is_paper_mode)
            )
        )
        early_used_margin = sum(o.investment for o in early_open_result.scalars().all())
        early_portfolio = trading_engine.paper_balance + early_used_margin
        early_return_multiple = round(early_portfolio / config.trading_config.paper_balance, 4) if config.trading_config.paper_balance > 0 else 0
        return {
            "total_trades": 0,
            "total_longs": 0,
            "total_shorts": 0,
            "total_wins": 0,
            "total_losses": 0,
            "win_rate": 0,
            "win_rate_longs": 0,
            "win_rate_shorts": 0,
            "avg_win": 0,
            "avg_win_long": 0,
            "avg_win_short": 0,
            "avg_loss": 0,
            "avg_loss_long": 0,
            "avg_loss_short": 0,
            "best_win_long": 0,
            "best_win_short": 0,
            "worst_loss_long": 0,
            "worst_loss_short": 0,
            "total_pnl": 0,
            "total_pnl_percentage": 0,
            "total_pnl_notional_percentage": 0,
            "total_investment_notional": 0,
            "total_investment_value": 0,
            "total_investment_long_notional": 0,
            "total_investment_long_value": 0,
            "total_investment_short_notional": 0,
            "total_investment_short_value": 0,
            "total_fees": 0,
            "avg_duration": "00:00:00",
            "avg_duration_long": "00:00:00",
            "avg_duration_short": "00:00:00",
            "avg_leverage": 0,
            "return_multiple": early_return_multiple,
            "daily_compound_return": 0,
            "runtime_days": round(trading_engine.get_runtime_seconds() / 86400.0, 2),
            "by_confidence": {},
            "by_macro_trend": macro_trend_performance,
            "outcome_distribution": [],
            "gap_performance": [],
            "ema58_gap_performance": [],
            "rsi_performance": [],
            "adx_performance": [],
            "stretch_performance": [],
            "pair_slope_performance": [],
            "btc_slope_performance": [],
            "btc_adx_performance": [], "btc_adx_direction_performance": [], "adx_dir_crosstab": [], "btc_slope_adx_crosstab": [],
            "by_close_reason": {},
            "stop_loss_deep_dive": {"total_sl_trades": 0, "be_was_active": {"count": 0}, "positive_no_be": {"count": 0}, "never_positive": {"count": 0}, "avg_peak_all_sl": 0},
            "winning_trades_drawdown": [], "trough_recovery": [],
            "never_positive_deep_dive": [],
            "performance_over_time": [],
            "post_exit_regret_deep_dive": [],
            "hold_time_expectancy": [],
            "entry_conditions_by_reason": []
        }
    
    # Separate longs and shorts
    longs = [o for o in orders if o.direction == "LONG"]
    shorts = [o for o in orders if o.direction == "SHORT"]
    
    # Winning trades
    wins = [o for o in orders if (o.pnl or 0) > 0]
    losses = [o for o in orders if (o.pnl or 0) <= 0]
    
    long_wins = [o for o in longs if (o.pnl or 0) > 0]
    short_wins = [o for o in shorts if (o.pnl or 0) > 0]
    
    # Calculate metrics
    total_trades = len(orders)
    total_longs = len(longs)
    total_shorts = len(shorts)
    
    win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
    win_rate_longs = (len(long_wins) / total_longs * 100) if total_longs > 0 else 0
    win_rate_shorts = (len(short_wins) / total_shorts * 100) if total_shorts > 0 else 0
    
    # Average wins
    avg_win = sum(o.pnl for o in wins) / len(wins) if wins else 0
    avg_win_long = sum(o.pnl for o in long_wins) / len(long_wins) if long_wins else 0
    avg_win_short = sum(o.pnl for o in short_wins) / len(short_wins) if short_wins else 0
    avg_loss = sum(o.pnl for o in losses) / len(losses) if losses else 0
    long_losses = [o for o in longs if (o.pnl or 0) <= 0]
    short_losses = [o for o in shorts if (o.pnl or 0) <= 0]
    avg_loss_long = sum(o.pnl for o in long_losses) / len(long_losses) if long_losses else 0
    avg_loss_short = sum(o.pnl for o in short_losses) / len(short_losses) if short_losses else 0
    
    # Expectancy per trade: E = WR * AvgWin - (1 - WR) * |AvgLoss|
    wr = win_rate / 100
    expectancy = (wr * avg_win) - ((1 - wr) * abs(avg_loss)) if total_trades > 0 else 0
    
    # Best/worst - Best win should only count winning trades (pnl > 0)
    long_wins_pnls = [o.pnl for o in longs if (o.pnl or 0) > 0]
    short_wins_pnls = [o.pnl for o in shorts if (o.pnl or 0) > 0]
    long_loss_pnls = [o.pnl for o in longs if (o.pnl or 0) <= 0]
    short_loss_pnls = [o.pnl for o in shorts if (o.pnl or 0) <= 0]
    
    best_win_long = max(long_wins_pnls) if long_wins_pnls else 0
    best_win_short = max(short_wins_pnls) if short_wins_pnls else 0
    worst_loss_long = min(long_loss_pnls) if long_loss_pnls else 0
    worst_loss_short = min(short_loss_pnls) if short_loss_pnls else 0
    
    # Totals
    total_pnl = sum(o.pnl or 0 for o in orders)
    total_investment_value = sum(o.investment for o in orders)
    total_investment_notional = sum(o.notional_value for o in orders)
    total_pnl_percentage = (total_pnl / total_investment_value * 100) if total_investment_value > 0 else 0
    
    total_investment_long_value = sum(o.investment for o in longs)
    total_investment_long_notional = sum(o.notional_value for o in longs)
    total_investment_short_value = sum(o.investment for o in shorts)
    total_investment_short_notional = sum(o.notional_value for o in shorts)
    
    total_fees = sum(o.total_fee or 0 for o in orders)
    avg_leverage = sum(o.leverage for o in orders) / total_trades if total_trades > 0 else 0
    
    # Return Multiple & Daily Compound Return
    initial_balance = config.trading_config.paper_balance
    open_result = await db.execute(
        select(Order).where(
            and_(Order.status == "OPEN", Order.is_paper == trading_engine.is_paper_mode)
        )
    )
    open_orders_for_balance = open_result.scalars().all()
    used_margin = sum(o.investment for o in open_orders_for_balance)
    current_balance = trading_engine.paper_balance + used_margin
    runtime_days = trading_engine.get_runtime_seconds() / 86400.0
    return_multiple = current_balance / initial_balance if initial_balance > 0 else 0
    if runtime_days >= 0.5 and return_multiple > 0:
        daily_compound_return = (return_multiple ** (1 / runtime_days) - 1) * 100
    else:
        daily_compound_return = 0
    
    # Durations
    def calc_avg_duration(order_list):
        if not order_list:
            return "00:00:00"
        total_seconds = sum(
            (o.closed_at - o.opened_at).total_seconds() 
            for o in order_list if o.closed_at
        )
        avg_seconds = total_seconds / len(order_list)
        hours = int(avg_seconds // 3600)
        minutes = int((avg_seconds % 3600) // 60)
        seconds = int(avg_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    # Performance by confidence level
    confidence_performance = {}
    for conf in ['VERY_STRONG', 'STRONG_BUY', 'LOW', 'MEDIUM', 'HIGH', 'EXTREME']:
        conf_orders = [o for o in orders if o.confidence == conf]
        conf_longs = [o for o in conf_orders if o.direction == "LONG"]
        conf_shorts = [o for o in conf_orders if o.direction == "SHORT"]
        
        conf_long_pnl = sum(o.pnl or 0 for o in conf_longs)
        conf_short_pnl = sum(o.pnl or 0 for o in conf_shorts)
        conf_long_investment = sum(o.investment for o in conf_longs)
        conf_short_investment = sum(o.investment for o in conf_shorts)
        
        conf_long_wins = len([o for o in conf_longs if (o.pnl or 0) > 0])
        conf_short_wins = len([o for o in conf_shorts if (o.pnl or 0) > 0])
        
        # Total P&L, Win Rate, and Risk/Reward
        conf_total_pnl = conf_long_pnl + conf_short_pnl
        conf_total_wins = conf_long_wins + conf_short_wins
        conf_total_win_rate = round(conf_total_wins / len(conf_orders) * 100, 2) if conf_orders else 0
        
        conf_wins_list = [o for o in conf_orders if (o.pnl or 0) > 0]
        conf_losses_list = [o for o in conf_orders if (o.pnl or 0) <= 0]
        conf_avg_win = sum(o.pnl for o in conf_wins_list) / len(conf_wins_list) if conf_wins_list else 0
        conf_avg_loss = sum(o.pnl for o in conf_losses_list) / len(conf_losses_list) if conf_losses_list else 0
        conf_risk_reward = round(conf_avg_win / abs(conf_avg_loss), 2) if conf_avg_loss != 0 else 0
        
        # Average entry gap
        conf_gaps = [o.entry_gap for o in conf_orders if o.entry_gap is not None]
        conf_avg_gap = round(sum(conf_gaps) / len(conf_gaps), 4) if conf_gaps else None
        
        confidence_performance[conf] = {
            "total_trades": len(conf_orders),
            "long_trades": len(conf_longs),
            "short_trades": len(conf_shorts),
            "total_pnl": round(conf_total_pnl, 2),
            "total_win_rate": conf_total_win_rate,
            "risk_reward": conf_risk_reward,
            "long_pnl": round(conf_long_pnl, 2),
            "short_pnl": round(conf_short_pnl, 2),
            "long_pnl_pct": round(conf_long_pnl / conf_long_investment * 100, 2) if conf_long_investment > 0 else 0,
            "short_pnl_pct": round(conf_short_pnl / conf_short_investment * 100, 2) if conf_short_investment > 0 else 0,
            "long_win_rate": round(conf_long_wins / len(conf_longs) * 100, 2) if conf_longs else 0,
            "short_win_rate": round(conf_short_wins / len(conf_shorts) * 100, 2) if conf_shorts else 0,
            "avg_entry_gap": conf_avg_gap
        }
    
    # Outcome Distribution - trades grouped by P&L percentage ranges
    outcome_ranges = [
        ("> +1.0%", lambda pct: pct > 1.0),
        ("+0.5% to +1.0%", lambda pct: 0.5 < pct <= 1.0),
        ("+0.3% to +0.5%", lambda pct: 0.3 < pct <= 0.5),
        ("+0.15% to +0.3%", lambda pct: 0.15 < pct <= 0.3),
        ("0% to +0.15%", lambda pct: 0 < pct <= 0.15),
        ("-0.25% to 0%", lambda pct: -0.25 < pct <= 0),
        ("-0.40% to -0.25%", lambda pct: -0.40 < pct <= -0.25),
        ("< -0.40%", lambda pct: pct <= -0.40)
    ]
    
    outcome_distribution = []
    for range_name, condition in outcome_ranges:
        matching_orders = [o for o in orders if condition(o.pnl_percentage or 0)]
        count = len(matching_orders)
        avg_pnl_usd = sum(o.pnl or 0 for o in matching_orders) / count if count > 0 else 0
        outcome_distribution.append({
            "range": range_name,
            "count": count,
            "avg_pnl_usd": round(avg_pnl_usd, 2)
        })
    
    # Performance by Entry Gap / RSI / ADX (split by direction)
    gap_performance = []
    ema58_gap_performance = []
    rsi_performance = []
    adx_performance = []
    adx_direction_performance = []
    stretch_performance = []
    pair_slope_performance = []
    btc_slope_performance = []
    btc_adx_performance = []
    btc_adx_direction_performance = []
    adx_dir_crosstab = []
    btc_slope_adx_crosstab = []
    try:
        gap_ranges = [
            ("0.12 - 0.15%", 0.12, 0.15),
            ("0.15 - 0.20%", 0.15, 0.20),
            ("0.20 - 0.25%", 0.20, 0.25),
            ("0.25 - 0.30%", 0.25, 0.30),
            ("0.30 - 0.35%", 0.30, 0.35),
            ("0.35 - 0.40%", 0.35, 0.40),
            ("0.40 - 0.50%", 0.40, 0.50),
            ("0.50 - 0.60%", 0.50, 0.60),
            ("0.60 - 0.70%", 0.60, 0.70),
            ("0.70 - 0.80%", 0.70, 0.80),
            ("0.80 - 1.00%", 0.80, 1.00),
            ("1.00 - 1.25%", 1.00, 1.25),
            ("1.25 - 1.50%", 1.25, 1.50),
            ("> 1.50%", 1.50, 999),
        ]
        gap_orders = [o for o in orders if o.entry_gap is not None]
        for range_name, gap_min, gap_max in gap_ranges:
            range_orders = [o for o in gap_orders if gap_min <= o.entry_gap < gap_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                gap_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        # Performance by Entry Gap EMA5-EMA8 (momentum gap)
        ema58_ranges = [
            ("0.00 - 0.01%", 0.00, 0.01),
            ("0.01 - 0.02%", 0.01, 0.02),
            ("0.02 - 0.05%", 0.02, 0.05),
            ("0.05 - 0.10%", 0.05, 0.10),
            ("0.10 - 0.20%", 0.10, 0.20),
            ("0.20 - 0.30%", 0.20, 0.30),
            ("0.30 - 0.40%", 0.30, 0.40),
            ("> 0.40%", 0.40, 999),
        ]
        ema58_gap_orders = [o for o in orders if o.entry_ema_gap_5_8 is not None]
        for range_name, gap_min, gap_max in ema58_ranges:
            range_orders = [o for o in ema58_gap_orders if gap_min <= o.entry_ema_gap_5_8 < gap_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                ema58_gap_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        rsi_ranges = [
            ("20 - 30", 20, 30),
            ("30 - 35", 30, 35),
            ("35 - 40", 35, 40),
            ("40 - 45", 40, 45),
            ("45 - 50", 45, 50),
            ("50 - 55", 50, 55),
            ("55 - 60", 55, 60),
            ("60 - 65", 60, 65),
            ("65 - 70", 65, 70),
            ("70 - 80", 70, 80),
        ]
        rsi_orders = [o for o in orders if o.entry_rsi is not None]
        for range_name, rsi_min, rsi_max in rsi_ranges:
            range_orders = [o for o in rsi_orders if rsi_min <= o.entry_rsi < rsi_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                rsi_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        adx_ranges = [
            ("10 - 15", 10, 15),
            ("15 - 20", 15, 20),
            ("20 - 25", 20, 25),
            ("25 - 30", 25, 30),
            ("30 - 35", 30, 35),
            ("35 - 40", 35, 40),
            ("40 - 50", 40, 50),
            ("> 50", 50, 999),
        ]
        adx_orders = [o for o in orders if o.entry_adx is not None]
        for range_name, adx_min, adx_max in adx_ranges:
            range_orders = [o for o in adx_orders if adx_min <= o.entry_adx < adx_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                adx_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })
        # Performance by ADX Direction (Rising vs Falling at entry)
        adx_dir_orders = [o for o in orders if o.entry_adx is not None and o.entry_adx_prev is not None]
        for adx_dir_label in ["Rising", "Falling"]:
            if adx_dir_label == "Rising":
                dir_pool = [o for o in adx_dir_orders if o.entry_adx > o.entry_adx_prev]
            else:
                dir_pool = [o for o in adx_dir_orders if o.entry_adx <= o.entry_adx_prev]
            if not dir_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                d_orders = [o for o in dir_pool if (o.direction or "LONG") == direction]
                count = len(d_orders)
                if count == 0:
                    continue
                d_wins = len([o for o in d_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in d_orders)
                avg_dur_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in d_orders if o.closed_at) / count if count > 0 else 0
                dur_h, dur_m, dur_s = int(avg_dur_secs // 3600), int((avg_dur_secs % 3600) // 60), int(avg_dur_secs % 60)
                conf_breakdown = {}
                for o in d_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                adx_direction_performance.append({
                    "adx_direction": adx_dir_label,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(d_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "avg_duration": f"{dur_h:02d}:{dur_m:02d}:{dur_s:02d}",
                    "by_confidence": conf_breakdown
                })
        # Performance by Entry EMA5 Stretch
        stretch_ranges = [
            ("0.00 - 0.04%", 0.00, 0.04),
            ("0.04 - 0.08%", 0.04, 0.08),
            ("0.08 - 0.12%", 0.08, 0.12),
            ("0.12 - 0.16%", 0.12, 0.16),
            ("0.16 - 0.20%", 0.16, 0.20),
            ("0.20 - 0.25%", 0.20, 0.25),
            ("0.25 - 0.30%", 0.25, 0.30),
            ("> 0.30%", 0.30, 999),
        ]
        stretch_orders = [o for o in orders if o.entry_ema5_stretch is not None]
        for range_name, s_min, s_max in stretch_ranges:
            range_orders = [o for o in stretch_orders if s_min <= o.entry_ema5_stretch < s_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                stretch_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        # Performance by Pair EMA20 Slope (absolute value, 0.02% buckets)
        slope_ranges = [
            ("0.00 - 0.02%", 0.00, 0.02),
            ("0.02 - 0.04%", 0.02, 0.04),
            ("0.04 - 0.06%", 0.04, 0.06),
            ("0.06 - 0.08%", 0.06, 0.08),
            ("0.08 - 0.10%", 0.08, 0.10),
            ("0.10 - 0.12%", 0.10, 0.12),
            ("0.12 - 0.14%", 0.12, 0.14),
            ("0.14 - 0.16%", 0.14, 0.16),
            ("0.16 - 0.18%", 0.16, 0.18),
            ("0.18 - 0.20%", 0.18, 0.20),
            ("0.20 - 0.25%", 0.20, 0.25),
            ("0.25 - 0.30%", 0.25, 0.30),
            ("0.30 - 0.40%", 0.30, 0.40),
            ("0.40 - 0.60%", 0.40, 0.60),
            ("> 0.60%", 0.60, 999),
        ]
        pair_slope_orders = [o for o in orders if o.entry_ema20_slope is not None]
        for range_name, s_min, s_max in slope_ranges:
            range_orders = [o for o in pair_slope_orders if s_min <= abs(o.entry_ema20_slope) < s_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                avg_rsi = round(sum(o.entry_rsi or 0 for o in dir_orders) / count, 1) if any(o.entry_rsi for o in dir_orders) else None
                avg_adx = round(sum(o.entry_adx or 0 for o in dir_orders) / count, 1) if any(o.entry_adx for o in dir_orders) else None
                avg_gap = round(sum(o.entry_gap or 0 for o in dir_orders) / count, 4) if any(o.entry_gap for o in dir_orders) else None
                pair_slope_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap
                })

        # Performance by BTC EMA20 Slope (absolute value, 0.02% buckets)
        btc_slope_orders = [o for o in orders if o.entry_btc_ema20_slope is not None]
        for range_name, s_min, s_max in slope_ranges:
            range_orders = [o for o in btc_slope_orders if s_min <= abs(o.entry_btc_ema20_slope) < s_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                avg_rsi = round(sum(o.entry_rsi or 0 for o in dir_orders) / count, 1) if any(o.entry_rsi for o in dir_orders) else None
                avg_adx = round(sum(o.entry_adx or 0 for o in dir_orders) / count, 1) if any(o.entry_adx for o in dir_orders) else None
                avg_gap = round(sum(o.entry_gap or 0 for o in dir_orders) / count, 4) if any(o.entry_gap for o in dir_orders) else None
                btc_slope_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap
                })

        # Performance by BTC ADX at entry
        btc_adx_ranges = [
            ("10-15", 10, 15), ("15-20", 15, 20), ("20-25", 20, 25),
            ("25-30", 25, 30), ("30-35", 30, 35), ("35-40", 35, 40), ("40+", 40, 999),
        ]
        btc_adx_orders = [o for o in orders if o.entry_btc_adx is not None]
        for range_name, a_min, a_max in btc_adx_ranges:
            range_orders = [o for o in btc_adx_orders if a_min <= o.entry_btc_adx < a_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                avg_rsi = round(sum(o.entry_rsi or 0 for o in dir_orders) / count, 1) if any(o.entry_rsi for o in dir_orders) else None
                avg_adx = round(sum(o.entry_adx or 0 for o in dir_orders) / count, 1) if any(o.entry_adx for o in dir_orders) else None
                avg_gap = round(sum(o.entry_gap or 0 for o in dir_orders) / count, 4) if any(o.entry_gap for o in dir_orders) else None
                btc_adx_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap
                })

        # Performance by BTC ADX Direction (Rising vs Falling at entry)
        btc_adx_dir_orders = [o for o in orders if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None]
        for btc_dir_label in ["Rising", "Falling"]:
            if btc_dir_label == "Rising":
                btc_dir_pool = [o for o in btc_adx_dir_orders if o.entry_btc_adx > o.entry_btc_adx_prev]
            else:
                btc_dir_pool = [o for o in btc_adx_dir_orders if o.entry_btc_adx <= o.entry_btc_adx_prev]
            if not btc_dir_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                bd_orders = [o for o in btc_dir_pool if (o.direction or "LONG") == direction]
                bd_count = len(bd_orders)
                if bd_count == 0:
                    continue
                bd_wins = len([o for o in bd_orders if (o.pnl or 0) > 0])
                bd_pnl_sum = sum(o.pnl or 0 for o in bd_orders)
                bd_avg_dur = sum((o.closed_at - o.opened_at).total_seconds() for o in bd_orders if o.closed_at) / bd_count if bd_count > 0 else 0
                bd_h, bd_m, bd_s = int(bd_avg_dur // 3600), int((bd_avg_dur % 3600) // 60), int(bd_avg_dur % 60)
                bd_conf = {}
                for o in bd_orders:
                    c = o.confidence or "UNKNOWN"
                    bd_conf[c] = bd_conf.get(c, 0) + 1
                btc_adx_direction_performance.append({
                    "btc_adx_direction": btc_dir_label,
                    "direction": direction,
                    "count": bd_count,
                    "win_rate": round(bd_wins / bd_count * 100, 1),
                    "avg_pnl_usd": round(bd_pnl_sum / bd_count, 2),
                    "total_pnl_usd": round(bd_pnl_sum, 2),
                    "avg_duration": f"{bd_h:02d}:{bd_m:02d}:{bd_s:02d}",
                    "by_confidence": bd_conf
                })

        # Pair ADX Dir x BTC ADX Dir Cross-Tab
        adx_ct_orders = [o for o in orders if o.entry_adx is not None and o.entry_adx_prev is not None
                         and o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None]
        for pair_dir in ["Rising", "Falling"]:
            for btc_dir in ["Rising", "Falling"]:
                pool = [o for o in adx_ct_orders
                        if (o.entry_adx > o.entry_adx_prev) == (pair_dir == "Rising")
                        and (o.entry_btc_adx > o.entry_btc_adx_prev) == (btc_dir == "Rising")]
                if not pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    dc_orders = [o for o in pool if (o.direction or "LONG") == direction]
                    dc_count = len(dc_orders)
                    if dc_count == 0:
                        continue
                    dc_wins = len([o for o in dc_orders if (o.pnl or 0) > 0])
                    dc_pnl = sum(o.pnl or 0 for o in dc_orders)
                    dc_conf = {}
                    for o in dc_orders:
                        c = o.confidence or "UNKNOWN"
                        dc_conf[c] = dc_conf.get(c, 0) + 1
                    adx_dir_crosstab.append({
                        "pair_adx_dir": pair_dir,
                        "btc_adx_dir": btc_dir,
                        "direction": direction,
                        "trades": dc_count,
                        "win_rate": round(dc_wins / dc_count * 100, 1),
                        "avg_pnl": round(dc_pnl / dc_count, 2),
                        "total_pnl": round(dc_pnl, 2),
                        "by_confidence": dc_conf,
                    })

        # BTC Slope x BTC ADX Cross-Tab
        ct_slope_ranges = [
            ("<0.06%", 0.00, 0.06), ("0.06-0.10%", 0.06, 0.10), ("0.10-0.16%", 0.10, 0.16),
            ("0.16-0.25%", 0.16, 0.25), ("0.25-0.40%", 0.25, 0.40), (">0.40%", 0.40, 999),
        ]
        ct_adx_ranges_bt = [
            ("20-25", 20, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35+", 35, 999),
        ]
        ct_orders = [o for o in orders if o.entry_btc_ema20_slope is not None and o.entry_btc_adx is not None]
        for sr_name, sr_min, sr_max in ct_slope_ranges:
            for ar_name, ar_min, ar_max in ct_adx_ranges_bt:
                bucket = [o for o in ct_orders if sr_min <= abs(o.entry_btc_ema20_slope) < sr_max and ar_min <= o.entry_btc_adx < ar_max]
                if not bucket:
                    continue
                ct_wins = len([o for o in bucket if (o.pnl or 0) > 0])
                ct_pnl_sum = sum(o.pnl or 0 for o in bucket)
                ct_count = len(bucket)
                ct_long_count = len([o for o in bucket if (o.direction or "LONG") == "LONG"])
                ct_short_count = ct_count - ct_long_count
                btc_slope_adx_crosstab.append({
                    "slope_range": sr_name,
                    "adx_range": ar_name,
                    "trades": ct_count,
                    "direction": f"{ct_long_count}L/{ct_short_count}S",
                    "win_rate": round(ct_wins / ct_count * 100, 1),
                    "avg_pnl": round(ct_pnl_sum / ct_count, 2),
                    "total_pnl": round(ct_pnl_sum, 2),
                })

    except Exception as e:
        logger.error(f"[PERF] Error computing gap/rsi/adx/stretch performance: {e}\n{traceback.format_exc()}")
        gap_performance = []
        ema58_gap_performance = []
        rsi_performance = []
        adx_performance = []
        adx_direction_performance = []
        stretch_performance = []
        pair_slope_performance = []
        btc_slope_performance = []
        btc_adx_performance = []
        btc_adx_direction_performance = []
        adx_dir_crosstab = []
        btc_slope_adx_crosstab = []
    
    # By Close Reason - group by reason with L4+ aggregation
    by_close_reason = {}
    try:
        close_reason_stats = {}
        for o in orders:
            reason = o.close_reason or "UNKNOWN"
            
            if " L" in reason:
                parts = reason.split(" L")
                base_reason = parts[0]
                try:
                    level = int(parts[1].replace("+", ""))
                    if level >= 6:
                        reason = f"{base_reason} L6+"
                except ValueError:
                    pass
            
            if reason not in close_reason_stats:
                close_reason_stats[reason] = {"trades": [], "pnl_sum": 0, "pnl_pct_sum": 0, "by_confidence": {}, "by_direction": {"LONG": 0, "SHORT": 0}}
            
            close_reason_stats[reason]["trades"].append(o)
            close_reason_stats[reason]["pnl_sum"] += o.pnl or 0
            close_reason_stats[reason]["pnl_pct_sum"] += o.pnl_percentage or 0
            
            direction = o.direction or "UNKNOWN"
            if direction in close_reason_stats[reason]["by_direction"]:
                close_reason_stats[reason]["by_direction"][direction] += 1
            
            conf = o.confidence or "UNKNOWN"
            close_reason_stats[reason]["by_confidence"][conf] = close_reason_stats[reason]["by_confidence"].get(conf, 0) + 1
        
        def _price_drop_pct(o):
            if o.direction == "LONG" and o.high_price_since_entry and o.exit_price and o.high_price_since_entry > 0:
                return (o.high_price_since_entry - o.exit_price) / o.high_price_since_entry * 100
            elif o.direction == "SHORT" and o.low_price_since_entry and o.exit_price and o.low_price_since_entry > 0:
                return (o.exit_price - o.low_price_since_entry) / o.low_price_since_entry * 100
            return 0

        for reason, data in close_reason_stats.items():
            count = len(data["trades"])
            drops = [_price_drop_pct(o) for o in data["trades"]]
            avg_drop = sum(drops) / count if count > 0 else 0
            peaks = [o.peak_pnl or 0 for o in data["trades"]]
            avg_peak = sum(peaks) / count if count > 0 else 0
            sig_active = sum(1 for o in data["trades"] if o.signal_active_at_close is True)
            sig_inactive = sum(1 for o in data["trades"] if o.signal_active_at_close is False)
            gaps = [o.entry_gap for o in data["trades"] if o.entry_gap is not None]
            rsis = [o.entry_rsi for o in data["trades"] if o.entry_rsi is not None]
            post_exit_peaks = [o.post_exit_peak_pnl for o in data["trades"] if o.post_exit_peak_pnl is not None]
            post_exit_troughs = [o.post_exit_trough_pnl for o in data["trades"] if o.post_exit_trough_pnl is not None]

            rsi2_fired = [o for o in data["trades"] if o.first_rsi2_pnl is not None]
            rsi3_fired = [o for o in data["trades"] if o.first_rsi3_pnl is not None]

            by_close_reason[reason] = {
                "trades": count,
                "avg_pnl_pct": round(data["pnl_pct_sum"] / count, 2) if count > 0 else 0,
                "avg_pnl_usd": round(data["pnl_sum"] / count, 2) if count > 0 else 0,
                "total_pnl_usd": round(data["pnl_sum"], 2),
                "by_confidence": data["by_confidence"],
                "by_direction": data["by_direction"],
                "avg_price_drop": round(avg_drop, 4),
                "avg_peak_pnl_pct": round(avg_peak, 4),
                "avg_post_exit_peak_pnl": round(sum(post_exit_peaks) / len(post_exit_peaks), 4) if post_exit_peaks else None,
                "avg_post_exit_trough_pnl": round(sum(post_exit_troughs) / len(post_exit_troughs), 4) if post_exit_troughs else None,
                "signal_active": sig_active,
                "signal_inactive": sig_inactive,
                "avg_entry_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "avg_entry_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_duration": calc_avg_duration(data["trades"]),
                "rsi2_fire_pct": round(len(rsi2_fired) / count * 100, 1) if count > 0 else 0,
                "avg_rsi2_pnl": round(sum(o.first_rsi2_pnl for o in rsi2_fired) / len(rsi2_fired), 4) if rsi2_fired else None,
                "avg_rsi2_min": round(sum(o.first_rsi2_minutes for o in rsi2_fired) / len(rsi2_fired), 1) if rsi2_fired else None,
                "rsi3_fire_pct": round(len(rsi3_fired) / count * 100, 1) if count > 0 else 0,
                "avg_rsi3_pnl": round(sum(o.first_rsi3_pnl for o in rsi3_fired) / len(rsi3_fired), 4) if rsi3_fired else None,
                "avg_rsi3_min": round(sum(o.first_rsi3_minutes for o in rsi3_fired) / len(rsi3_fired), 1) if rsi3_fired else None,
            }
    except Exception as e:
        logger.error(f"[PERF] Error computing close reason stats: {e}\n{traceback.format_exc()}")
        by_close_reason = {}
    
    # Entry Conditions by Close Reason
    entry_conditions_by_reason = []
    try:
        ecr_groups = {}
        for o in orders:
            reason = o.close_reason or "UNKNOWN"
            if " L" in reason:
                parts = reason.split(" L")
                base_reason = parts[0]
                try:
                    level = int(parts[1].replace("+", ""))
                    if level >= 6:
                        reason = f"{base_reason} L6+"
                except ValueError:
                    pass
            ecr_groups.setdefault(reason, []).append(o)

        for reason in sorted(ecr_groups.keys()):
            group = ecr_groups[reason]
            count = len(group)
            if count == 0:
                continue

            ec_longs = sum(1 for o in group if o.direction == "LONG")
            ec_shorts = count - ec_longs

            ec_conf = {}
            for o in group:
                c = o.confidence or "UNKNOWN"
                ec_conf[c] = ec_conf.get(c, 0) + 1

            rsis = [o.entry_rsi for o in group if o.entry_rsi is not None]
            adxs = [o.entry_adx for o in group if o.entry_adx is not None]
            gaps = [o.entry_gap for o in group if o.entry_gap is not None]
            gaps58 = [o.entry_ema_gap_5_8 for o in group if o.entry_ema_gap_5_8 is not None]
            stretches = [o.entry_ema5_stretch for o in group if o.entry_ema5_stretch is not None]
            slopes = [abs(o.entry_ema20_slope) for o in group if o.entry_ema20_slope is not None]
            btc_slopes = [abs(o.entry_btc_ema20_slope) for o in group if o.entry_btc_ema20_slope is not None]
            btc_adxs = [o.entry_btc_adx for o in group if o.entry_btc_adx is not None]

            adx_rising = sum(1 for o in group if o.entry_adx is not None and o.entry_adx_prev is not None and o.entry_adx > o.entry_adx_prev)
            adx_falling = sum(1 for o in group if o.entry_adx is not None and o.entry_adx_prev is not None and o.entry_adx <= o.entry_adx_prev)
            btc_adx_rising = sum(1 for o in group if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None and o.entry_btc_adx > o.entry_btc_adx_prev)
            btc_adx_falling = sum(1 for o in group if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None and o.entry_btc_adx <= o.entry_btc_adx_prev)

            peaks = [o.peak_pnl or 0 for o in group]
            pnls = [o.pnl_percentage or 0 for o in group]

            avg_dur_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in group if o.closed_at and o.opened_at) / count
            hours, remainder = divmod(int(avg_dur_secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            avg_dur_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            entry_conditions_by_reason.append({
                "reason": reason,
                "trades": count,
                "longs": ec_longs,
                "shorts": ec_shorts,
                "by_confidence": ec_conf,
                "avg_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_adx": round(sum(adxs) / len(adxs), 1) if adxs else None,
                "adx_rising": adx_rising,
                "adx_falling": adx_falling,
                "avg_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "avg_gap58": round(sum(gaps58) / len(gaps58), 4) if gaps58 else None,
                "avg_stretch": round(sum(stretches) / len(stretches), 4) if stretches else None,
                "avg_ema20_slope": round(sum(slopes) / len(slopes), 4) if slopes else None,
                "avg_btc_slope": round(sum(btc_slopes) / len(btc_slopes), 4) if btc_slopes else None,
                "avg_btc_adx": round(sum(btc_adxs) / len(btc_adxs), 1) if btc_adxs else None,
                "btc_adx_rising": btc_adx_rising,
                "btc_adx_falling": btc_adx_falling,
                "avg_peak_pct": round(sum(peaks) / count, 4),
                "avg_pnl_pct": round(sum(pnls) / count, 4),
                "avg_duration": avg_dur_str,
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing entry conditions by reason: {e}\n{traceback.format_exc()}")

    # Stop Loss Deep Dive + Winning Trades Drawdown
    stop_loss_deep_dive = {"total_sl_trades": 0, "be_was_active": {"count": 0}, "positive_no_be": {"count": 0}, "never_positive": {"count": 0}, "avg_peak_all_sl": 0}
    winning_trades_drawdown = []
    never_positive_trades = []
    try:
        def _price_drop_pct_sl(o):
            if o.direction == "LONG" and o.high_price_since_entry and o.exit_price and o.high_price_since_entry > 0:
                return (o.high_price_since_entry - o.exit_price) / o.high_price_since_entry * 100
            elif o.direction == "SHORT" and o.low_price_since_entry and o.exit_price and o.low_price_since_entry > 0:
                return (o.exit_price - o.low_price_since_entry) / o.low_price_since_entry * 100
            return 0

        tc = config.trading_config
        sl_orders = [o for o in orders if o.close_reason and (o.pnl or 0) <= 0 and (
            o.close_reason.startswith("STOP_LOSS") or
            o.close_reason.startswith("MOMENTUM_EXIT") or
            o.close_reason.startswith("PNL_TRAILING") or
            o.close_reason.startswith("SLOPE_EXIT") or
            o.close_reason.startswith("SIGNAL_LOST") or
            o.close_reason.startswith("BREAKEVEN_SL")
        )]
        
        be_active_trades = []
        positive_no_be_trades = []
        never_positive_trades = []
        
        for o in sl_orders:
            conf = o.confidence or "LOW"
            conf_config = tc.confidence_levels.get(conf, tc.confidence_levels.get("LOW"))
            trigger = conf_config.be_level1_trigger
            peak = o.peak_pnl or 0
            
            if peak >= trigger:
                be_active_trades.append(o)
            elif peak > 0:
                positive_no_be_trades.append(o)
            else:
                never_positive_trades.append(o)
        
        def _sl_group_stats(group_orders):
            count = len(group_orders)
            empty = {
                "count": 0, "avg_peak_pnl": 0, "avg_trough_pnl": 0, "worst_trough_pnl": 0,
                "avg_close_pnl": 0, "total_pnl_usd": 0,
                "by_confidence": {}, "by_direction": {"LONG": 0, "SHORT": 0},
                "avg_entry_gap": None, "avg_entry_rsi": None, "avg_price_drop": 0,
                "avg_duration": "00:00:00",
                "signal_active": 0, "signal_inactive": 0
            }
            if count == 0:
                return empty
            avg_peak = sum(o.peak_pnl or 0 for o in group_orders) / count
            troughs = [o.trough_pnl or 0 for o in group_orders]
            avg_trough = sum(troughs) / count
            worst_trough = min(troughs)
            avg_close = sum(o.pnl_percentage or 0 for o in group_orders) / count
            total_pnl = sum(o.pnl or 0 for o in group_orders)
            drops = [_price_drop_pct_sl(o) for o in group_orders]
            avg_drop = sum(drops) / count
            by_conf = {}
            for o in group_orders:
                c = o.confidence or "LOW"
                by_conf[c] = by_conf.get(c, 0) + 1
            by_dir = {"LONG": 0, "SHORT": 0}
            for o in group_orders:
                d = o.direction or "LONG"
                by_dir[d] = by_dir.get(d, 0) + 1
            gaps = [o.entry_gap for o in group_orders if o.entry_gap is not None]
            rsis = [o.entry_rsi for o in group_orders if o.entry_rsi is not None]
            sig_active = sum(1 for o in group_orders if o.signal_active_at_close is True)
            sig_inactive = sum(1 for o in group_orders if o.signal_active_at_close is False)
            return {
                "count": count,
                "avg_peak_pnl": round(avg_peak, 4),
                "avg_trough_pnl": round(avg_trough, 4),
                "worst_trough_pnl": round(worst_trough, 4),
                "avg_close_pnl": round(avg_close, 4),
                "total_pnl_usd": round(total_pnl, 2),
                "by_confidence": by_conf,
                "by_direction": by_dir,
                "avg_entry_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
                "avg_entry_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_price_drop": round(avg_drop, 4),
                "avg_duration": calc_avg_duration(group_orders),
                "signal_active": sig_active,
                "signal_inactive": sig_inactive
            }
        
        all_sl_by_conf = {}
        for o in sl_orders:
            c = o.confidence or "LOW"
            all_sl_by_conf[c] = all_sl_by_conf.get(c, 0) + 1
        all_sl_by_dir = {"LONG": 0, "SHORT": 0}
        for o in sl_orders:
            d = o.direction or "LONG"
            all_sl_by_dir[d] = all_sl_by_dir.get(d, 0) + 1
        all_sl_gaps = [o.entry_gap for o in sl_orders if o.entry_gap is not None]
        all_sl_rsis = [o.entry_rsi for o in sl_orders if o.entry_rsi is not None]
        all_sl_drops = [_price_drop_pct_sl(o) for o in sl_orders]
        all_sl_count = len(sl_orders)

        stop_loss_deep_dive = {
            "total_sl_trades": all_sl_count,
            "be_was_active": _sl_group_stats(be_active_trades),
            "positive_no_be": _sl_group_stats(positive_no_be_trades),
            "never_positive": _sl_group_stats(never_positive_trades),
            "avg_peak_all_sl": round(sum(o.peak_pnl or 0 for o in sl_orders) / all_sl_count, 4) if sl_orders else 0,
            "all_avg_trough_pnl": round(sum(o.trough_pnl or 0 for o in sl_orders) / all_sl_count, 4) if sl_orders else 0,
            "all_worst_trough_pnl": round(min((o.trough_pnl or 0) for o in sl_orders), 4) if sl_orders else 0,
            "all_avg_close_pnl": round(sum(o.pnl_percentage or 0 for o in sl_orders) / all_sl_count, 4) if sl_orders else 0,
            "all_avg_price_drop": round(sum(all_sl_drops) / all_sl_count, 4) if sl_orders else 0,
            "all_total_pnl_usd": round(sum(o.pnl or 0 for o in sl_orders), 2),
            "all_by_confidence": all_sl_by_conf,
            "all_by_direction": all_sl_by_dir,
            "all_avg_entry_gap": round(sum(all_sl_gaps) / len(all_sl_gaps), 2) if all_sl_gaps else None,
            "all_avg_entry_rsi": round(sum(all_sl_rsis) / len(all_sl_rsis), 1) if all_sl_rsis else None,
            "all_avg_duration": calc_avg_duration(sl_orders),
            "all_signal_active": sum(1 for o in sl_orders if o.signal_active_at_close is True),
            "all_signal_inactive": sum(1 for o in sl_orders if o.signal_active_at_close is False)
        }
        
        winning_orders = [o for o in orders if o.pnl and o.pnl > 0]
        win_by_reason = {}
        for o in winning_orders:
            reason = (o.close_reason or "UNKNOWN").split(" L")[0]
            if reason not in win_by_reason:
                win_by_reason[reason] = []
            win_by_reason[reason].append(o)
        
        for reason, group in sorted(win_by_reason.items()):
            count = len(group)
            troughs = [o.trough_pnl or 0 for o in group]
            avg_trough = sum(troughs) / count
            worst_trough = min(troughs)
            peaks = [o.peak_pnl or 0 for o in group]
            avg_peak = sum(peaks) / count
            avg_close_pnl = sum(o.pnl_percentage or 0 for o in group) / count
            total_pnl_usd = sum(o.pnl or 0 for o in group)
            by_dir = {"LONG": 0, "SHORT": 0}
            for o in group:
                by_dir[o.direction or "LONG"] += 1
            by_conf = {}
            for o in group:
                c = o.confidence or "LOW"
                by_conf[c] = by_conf.get(c, 0) + 1
            drops = [_price_drop_pct_sl(o) for o in group]
            avg_drop = sum(drops) / count
            gaps = [o.entry_gap for o in group if o.entry_gap is not None]
            rsis = [o.entry_rsi for o in group if o.entry_rsi is not None]
            sig_active = sum(1 for o in group if o.signal_active_at_close is True)
            sig_inactive = sum(1 for o in group if o.signal_active_at_close is False)
            post_exit_peaks = [o.post_exit_peak_pnl for o in group if o.post_exit_peak_pnl is not None]
            post_exit_troughs = [o.post_exit_trough_pnl for o in group if o.post_exit_trough_pnl is not None]
            winning_trades_drawdown.append({
                "close_reason": reason,
                "count": count,
                "by_direction": by_dir,
                "by_confidence": by_conf,
                "avg_entry_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
                "avg_entry_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_trough_pnl": round(avg_trough, 4),
                "worst_trough_pnl": round(worst_trough, 4),
                "avg_peak_pnl": round(avg_peak, 4),
                "avg_close_pnl": round(avg_close_pnl, 4),
                "total_pnl_usd": round(total_pnl_usd, 2),
                "avg_price_drop": round(avg_drop, 4),
                "avg_duration": calc_avg_duration(group),
                "signal_active": sig_active,
                "signal_inactive": sig_inactive,
                "avg_post_exit_peak_pnl": round(sum(post_exit_peaks) / len(post_exit_peaks), 4) if post_exit_peaks else None,
                "avg_post_exit_trough_pnl": round(sum(post_exit_troughs) / len(post_exit_troughs), 4) if post_exit_troughs else None
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing SL deep dive / winning drawdown: {e}\n{traceback.format_exc()}")

    # Trough Recovery Analysis: cumulative — for each depth, how many trades dipped that far and recovered?
    trough_recovery = []
    try:
        trough_levels = [-0.05, -0.10, -0.15, -0.20, -0.25, -0.30, -0.40, -0.50]
        for level in trough_levels:
            reached = [o for o in orders if (o.trough_pnl or 0) <= level]
            count = len(reached)
            if count == 0:
                continue
            recovered = [o for o in reached if (o.pnl or 0) > 0]
            rec_count = len(recovered)
            avg_final_pct = sum(o.pnl_percentage or 0 for o in reached) / count
            avg_final_usd = sum(o.pnl or 0 for o in reached) / count
            sig_active = sum(1 for o in reached if o.signal_active_at_close is True)
            sig_inactive = sum(1 for o in reached if o.signal_active_at_close is False)
            by_dir = {"LONG": 0, "SHORT": 0}
            for o in reached:
                by_dir[o.direction or "LONG"] += 1
            reason_counts = {}
            for o in reached:
                r = (o.close_reason or "UNKNOWN").split(" L")[0]
                reason_counts[r] = reason_counts.get(r, 0) + 1
            top_reason = max(reason_counts, key=reason_counts.get) if reason_counts else "-"
            top_reason_n = reason_counts.get(top_reason, 0)
            trough_recovery.append({
                "level": level,
                "count": count,
                "recovered": rec_count,
                "recovery_pct": round(rec_count / count * 100, 1),
                "avg_final_pct": round(avg_final_pct, 4),
                "avg_final_usd": round(avg_final_usd, 2),
                "signal_active_pct": round(sig_active / count * 100, 1) if count > 0 else 0,
                "by_direction": by_dir,
                "top_reason": f"{top_reason} ({top_reason_n})",
                "avg_duration": calc_avg_duration(reached)
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing trough recovery: {e}\n{traceback.format_exc()}")

    rsi_adx_crosstab = []
    try:
        ct_rsi_ranges = [
            ("20-30", 20, 30), ("30-35", 30, 35), ("35-40", 35, 40),
            ("40-45", 40, 45), ("45-50", 45, 50), ("50-55", 50, 55),
            ("55-60", 55, 60), ("60-65", 60, 65), ("65-70", 65, 70), ("70-80", 70, 80),
        ]
        ct_adx_ranges = [
            ("10-15", 10, 15), ("15-20", 15, 20), ("20-25", 20, 25),
            ("25-30", 25, 30), ("30-35", 30, 35), ("35-40", 35, 40),
            ("40-50", 40, 50), ("50+", 50, 999),
        ]
        ct_orders = [o for o in orders if o.entry_rsi is not None and o.entry_adx is not None]
        for direction in ["LONG", "SHORT"]:
            dir_ct = [o for o in ct_orders if (o.direction or "LONG") == direction]
            for rsi_name, rsi_lo, rsi_hi in ct_rsi_ranges:
                for adx_name, adx_lo, adx_hi in ct_adx_ranges:
                    bucket = [o for o in dir_ct if rsi_lo <= o.entry_rsi < rsi_hi and adx_lo <= o.entry_adx < adx_hi]
                    if not bucket:
                        continue
                    ct_count = len(bucket)
                    ct_wins = sum(1 for o in bucket if (o.pnl or 0) > 0)
                    ct_pnl = sum(o.pnl or 0 for o in bucket)
                    rsi_adx_crosstab.append({
                        "direction": direction,
                        "rsi_range": rsi_name,
                        "adx_range": adx_name,
                        "trades": ct_count,
                        "win_rate": round(ct_wins / ct_count * 100, 1),
                        "total_pnl": round(ct_pnl, 2),
                        "avg_pnl": round(ct_pnl / ct_count, 2)
                    })
    except Exception as e:
        logger.error(f"[PERF] Error computing RSI x ADX cross-tab: {e}\n{traceback.format_exc()}")

    never_positive_deep_dive = []
    try:
        np_trades = never_positive_trades
        if np_trades:
            def _np_bucket_stats(bucket_orders, label, dimension_value, direction=None, total_in_range=0):
                count = len(bucket_orders)
                if count == 0:
                    return None
                total_pnl = sum(o.pnl or 0 for o in bucket_orders)
                avg_close = sum(o.pnl_percentage or 0 for o in bucket_orders) / count
                gaps = [o.entry_gap for o in bucket_orders if o.entry_gap is not None]
                gaps58 = [o.entry_ema_gap_5_8 for o in bucket_orders if o.entry_ema_gap_5_8 is not None]
                rsis = [o.entry_rsi for o in bucket_orders if o.entry_rsi is not None]
                adxs = [o.entry_adx for o in bucket_orders if o.entry_adx is not None]
                by_conf = {}
                for o in bucket_orders:
                    c = o.confidence or "LOW"
                    by_conf[c] = by_conf.get(c, 0) + 1
                by_reason = {}
                for o in bucket_orders:
                    r = (o.close_reason or "UNKNOWN").split(" L")[0]
                    by_reason[r] = by_reason.get(r, 0) + 1
                sig_active = sum(1 for o in bucket_orders if o.signal_active_at_close is True)
                sig_inactive = sum(1 for o in bucket_orders if o.signal_active_at_close is False)
                pct = round(count / total_in_range * 100, 1) if total_in_range > 0 else 0
                return {
                    "dimension": label,
                    "value": dimension_value,
                    "direction": direction,
                    "count": count,
                    "total_in_range": total_in_range,
                    "pct_of_total": pct,
                    "avg_close_pnl": round(avg_close, 4),
                    "total_pnl_usd": round(total_pnl, 2),
                    "avg_pnl_usd": round(total_pnl / count, 2),
                    "avg_entry_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                    "avg_entry_gap58": round(sum(gaps58) / len(gaps58), 4) if gaps58 else None,
                    "avg_entry_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                    "avg_entry_adx": round(sum(adxs) / len(adxs), 1) if adxs else None,
                    "by_confidence": by_conf,
                    "by_close_reason": by_reason,
                    "signal_active": sig_active,
                    "signal_inactive": sig_inactive,
                    "avg_duration": calc_avg_duration(bucket_orders)
                }

            for direction in ["LONG", "SHORT"]:
                dir_trades = [o for o in np_trades if (o.direction or "LONG") == direction]
                all_dir = len([o for o in orders if (o.direction or "LONG") == direction])
                row = _np_bucket_stats(dir_trades, "Direction", direction, direction, all_dir)
                if row:
                    never_positive_deep_dive.append(row)

            for conf in ["VERY_STRONG", "STRONG_BUY"]:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_trades if (o.confidence or "LOW") == conf and (o.direction or "LONG") == direction]
                    all_in = len([o for o in orders if (o.confidence or "LOW") == conf and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Confidence", conf, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_adx_ranges = [("25-30", 25, 30), ("30-35", 30, 35), ("35-40", 35, 40), ("40-50", 40, 50)]
            np_adx_trades = [o for o in np_trades if o.entry_adx is not None]
            all_adx_trades = [o for o in orders if o.entry_adx is not None]
            for rng, lo, hi in np_adx_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_adx_trades if lo <= o.entry_adx < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_adx_trades if lo <= o.entry_adx < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "ADX", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_rsi_ranges = [("30-35", 30, 35), ("35-40", 35, 40), ("40-45", 40, 45), ("45-50", 45, 50), ("50-55", 50, 55), ("55-60", 55, 60), ("60-65", 60, 65), ("65-70", 65, 70), ("70-80", 70, 80)]
            np_rsi_trades = [o for o in np_trades if o.entry_rsi is not None]
            all_rsi_trades = [o for o in orders if o.entry_rsi is not None]
            for rng, lo, hi in np_rsi_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_rsi_trades if lo <= o.entry_rsi < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_rsi_trades if lo <= o.entry_rsi < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "RSI", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_gap58_ranges = [("0.02-0.05%", 0.02, 0.05), ("0.05-0.10%", 0.05, 0.10), ("0.10-0.20%", 0.10, 0.20), (">0.20%", 0.20, 999)]
            np_gap58_trades = [o for o in np_trades if o.entry_ema_gap_5_8 is not None]
            all_gap58_trades = [o for o in orders if o.entry_ema_gap_5_8 is not None]
            for rng, lo, hi in np_gap58_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_gap58_trades if lo <= o.entry_ema_gap_5_8 < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_gap58_trades if lo <= o.entry_ema_gap_5_8 < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "EMA5-EMA8 Gap", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_gap_ranges = [("0.15-0.25%", 0.15, 0.25), ("0.25-0.35%", 0.25, 0.35), ("0.35-0.50%", 0.35, 0.50), ("0.50-0.80%", 0.50, 0.80), (">0.80%", 0.80, 999)]
            np_gap_trades = [o for o in np_trades if o.entry_gap is not None]
            all_gap_trades = [o for o in orders if o.entry_gap is not None]
            for rng, lo, hi in np_gap_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_gap_trades if lo <= o.entry_gap < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_gap_trades if lo <= o.entry_gap < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Entry Gap", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_stretch_ranges = [("0.04-0.08%", 0.04, 0.08), ("0.08-0.12%", 0.08, 0.12), ("0.12-0.16%", 0.12, 0.16), ("0.16-0.20%", 0.16, 0.20), ("0.20-0.25%", 0.20, 0.25), ("0.25-0.30%", 0.25, 0.30), (">0.30%", 0.30, 999)]
            np_stretch_trades = [o for o in np_trades if o.entry_ema5_stretch is not None]
            all_stretch_trades = [o for o in orders if o.entry_ema5_stretch is not None]
            for rng, lo, hi in np_stretch_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_stretch_trades if lo <= o.entry_ema5_stretch < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_stretch_trades if lo <= o.entry_ema5_stretch < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "EMA5 Stretch", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_slope_ranges = [("0.06-0.08%", 0.06, 0.08), ("0.08-0.10%", 0.08, 0.10), ("0.10-0.12%", 0.10, 0.12), ("0.12-0.14%", 0.12, 0.14), ("0.14-0.16%", 0.14, 0.16), ("0.16-0.18%", 0.16, 0.18), ("0.18-0.20%", 0.18, 0.20), ("0.20-0.25%", 0.20, 0.25), ("0.25-0.30%", 0.25, 0.30), ("0.30-0.40%", 0.30, 0.40), ("0.40-0.60%", 0.40, 0.60), (">0.60%", 0.60, 999)]
            np_slope_trades = [o for o in np_trades if o.entry_ema20_slope is not None]
            all_slope_trades = [o for o in orders if o.entry_ema20_slope is not None]
            for rng, lo, hi in np_slope_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_slope_trades if lo <= abs(o.entry_ema20_slope) < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_slope_trades if lo <= abs(o.entry_ema20_slope) < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "EMA20 Slope", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_btc_slope_ranges = [("0.06-0.10%", 0.06, 0.10), ("0.10-0.14%", 0.10, 0.14), ("0.14-0.18%", 0.14, 0.18), ("0.18-0.25%", 0.18, 0.25), (">0.25%", 0.25, 999)]
            np_btc_slope_trades = [o for o in np_trades if o.entry_btc_ema20_slope is not None]
            all_btc_slope_trades = [o for o in orders if o.entry_btc_ema20_slope is not None]
            for rng, lo, hi in np_btc_slope_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_slope_trades if lo <= abs(o.entry_btc_ema20_slope) < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_slope_trades if lo <= abs(o.entry_btc_ema20_slope) < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC EMA20 Slope", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_btc_adx_ranges = [("20-25", 20, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35+", 35, 999)]
            np_btc_adx_trades = [o for o in np_trades if o.entry_btc_adx is not None]
            all_btc_adx_trades = [o for o in orders if o.entry_btc_adx is not None]
            for rng, lo, hi in np_btc_adx_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_adx_trades if lo <= o.entry_btc_adx < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_adx_trades if lo <= o.entry_btc_adx < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC ADX", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            for np_adx_dir_label in ["Rising", "Falling"]:
                np_adx_dir_trades = [o for o in np_trades if o.entry_adx is not None and o.entry_adx_prev is not None]
                all_adx_dir_trades = [o for o in orders if o.entry_adx is not None and o.entry_adx_prev is not None]
                if np_adx_dir_label == "Rising":
                    np_adx_dir_pool = [o for o in np_adx_dir_trades if o.entry_adx > o.entry_adx_prev]
                    all_adx_dir_pool = [o for o in all_adx_dir_trades if o.entry_adx > o.entry_adx_prev]
                else:
                    np_adx_dir_pool = [o for o in np_adx_dir_trades if o.entry_adx <= o.entry_adx_prev]
                    all_adx_dir_pool = [o for o in all_adx_dir_trades if o.entry_adx <= o.entry_adx_prev]
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_adx_dir_pool if (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_adx_dir_pool if (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "ADX Direction", np_adx_dir_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            for np_btc_dir_label in ["Rising", "Falling"]:
                np_btc_dir_trades = [o for o in np_trades if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None]
                all_btc_dir_trades = [o for o in orders if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None]
                if np_btc_dir_label == "Rising":
                    np_btc_dir_pool = [o for o in np_btc_dir_trades if o.entry_btc_adx > o.entry_btc_adx_prev]
                    all_btc_dir_pool = [o for o in all_btc_dir_trades if o.entry_btc_adx > o.entry_btc_adx_prev]
                else:
                    np_btc_dir_pool = [o for o in np_btc_dir_trades if o.entry_btc_adx <= o.entry_btc_adx_prev]
                    all_btc_dir_pool = [o for o in all_btc_dir_trades if o.entry_btc_adx <= o.entry_btc_adx_prev]
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_dir_pool if (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_dir_pool if (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC ADX Direction", np_btc_dir_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            for reason_key in ["STOP_LOSS", "STOP_LOSS_WIDE", "MOMENTUM_EXIT", "PNL_TRAILING", "SLOPE_EXIT", "SIGNAL_LOST"]:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_trades if o.close_reason and o.close_reason.startswith(reason_key) and (o.direction or "LONG") == direction]
                    all_in = len([o for o in orders if o.close_reason and o.close_reason.startswith(reason_key) and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Close Reason", reason_key, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)
    except Exception as e:
        logger.error(f"[PERF] Error computing Never Positive deep dive: {e}\n{traceback.format_exc()}")

    # Post-Exit Regret Deep Dive
    post_exit_regret_deep_dive = []
    try:
        pe_orders = [o for o in orders if o.post_exit_peak_pnl is not None and o.close_reason and (o.close_reason.startswith("BREAKEVEN_SL") or o.close_reason.startswith("SIGNAL_LOST") or o.close_reason.startswith("TICK_MOMENTUM_EXIT") or o.close_reason.startswith("RSI_MOMENTUM_EXIT") or o.close_reason.startswith("STOP_LOSS"))]
        if pe_orders:
            reason_groups = {}
            for o in pe_orders:
                reason_groups.setdefault(o.close_reason, []).append(o)

            for reason in sorted(reason_groups.keys()):
                group = reason_groups[reason]
                count = len(group)
                pe_longs = sum(1 for o in group if o.direction == "LONG")
                pe_shorts = count - pe_longs

                avg_duration_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in group if o.closed_at) / count if count > 0 else 0
                dur_h, dur_m, dur_s = int(avg_duration_secs // 3600), int((avg_duration_secs % 3600) // 60), int(avg_duration_secs % 60)

                avg_close_pnl = sum(o.pnl_percentage or 0 for o in group) / count
                avg_post_peak = sum(o.post_exit_peak_pnl or 0 for o in group) / count
                avg_peak_min = sum(o.post_exit_peak_minutes or 0 for o in group) / count
                avg_post_trough = sum(o.post_exit_trough_pnl or 0 for o in group) / count
                avg_trough_min = sum(o.post_exit_trough_minutes or 0 for o in group) / count
                avg_final = sum(o.post_exit_final_pnl or 0 for o in group) / count

                peak_first_count = sum(1 for o in group if (o.post_exit_peak_minutes or 0) < (o.post_exit_trough_minutes or 0))
                peak_first_pct = round(peak_first_count / count * 100, 1) if count > 0 else 0

                sig_lost_orders = [o for o in group if o.post_exit_signal_lost_minutes is not None]
                sig_lost_pct = round(len(sig_lost_orders) / count * 100, 1) if count > 0 else 0
                avg_sig_lost_min = sum(o.post_exit_signal_lost_minutes for o in sig_lost_orders) / len(sig_lost_orders) if sig_lost_orders else None
                avg_pnl_at_sig_lost = sum(o.post_exit_pnl_at_signal_lost or 0 for o in sig_lost_orders) / len(sig_lost_orders) if sig_lost_orders else None
                reachable_peak_orders = [o for o in group if o.post_exit_peak_before_signal_lost is not None]
                avg_reachable_peak = sum(o.post_exit_peak_before_signal_lost for o in reachable_peak_orders) / len(reachable_peak_orders) if reachable_peak_orders else None

                rsi_exit_orders = [o for o in group if o.post_exit_rsi_exit_minutes is not None]
                rsi_exit_pct = round(len(rsi_exit_orders) / count * 100, 1) if count > 0 else 0
                avg_rsi_exit_min = sum(o.post_exit_rsi_exit_minutes for o in rsi_exit_orders) / len(rsi_exit_orders) if rsi_exit_orders else None
                avg_rsi_exit_pnl = sum(o.post_exit_rsi_exit_pnl or 0 for o in rsi_exit_orders) / len(rsi_exit_orders) if rsi_exit_orders else None

                rsi3_exit_orders = [o for o in group if o.post_exit_rsi3_exit_minutes is not None]
                rsi3_exit_pct = round(len(rsi3_exit_orders) / count * 100, 1) if count > 0 else 0
                avg_rsi3_exit_min = sum(o.post_exit_rsi3_exit_minutes for o in rsi3_exit_orders) / len(rsi3_exit_orders) if rsi3_exit_orders else None
                avg_rsi3_exit_pnl = sum(o.post_exit_rsi3_exit_pnl or 0 for o in rsi3_exit_orders) / len(rsi3_exit_orders) if rsi3_exit_orders else None

                rec_005 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= 0.05)
                rec_010 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= 0.10)

                sig_regained_orders = [o for o in group if o.post_exit_signal_regained_minutes is not None]
                sig_regained_pct = round(len(sig_regained_orders) / count * 100, 1) if count > 0 else 0
                avg_sig_regained_min = sum(o.post_exit_signal_regained_minutes for o in sig_regained_orders) / len(sig_regained_orders) if sig_regained_orders else None
                avg_pnl_at_sig_regained = sum(o.post_exit_pnl_at_signal_regained or 0 for o in sig_regained_orders) / len(sig_regained_orders) if sig_regained_orders else None

                post_exit_regret_deep_dive.append({
                    "reason": reason,
                    "count": count,
                    "longs": pe_longs,
                    "shorts": pe_shorts,
                    "avg_duration": f"{dur_h:02d}:{dur_m:02d}:{dur_s:02d}",
                    "avg_close_pnl": round(avg_close_pnl, 4),
                    "avg_post_peak": round(avg_post_peak, 4),
                    "avg_peak_min": round(avg_peak_min, 1),
                    "avg_post_trough": round(avg_post_trough, 4),
                    "avg_trough_min": round(avg_trough_min, 1),
                    "peak_first_pct": peak_first_pct,
                    "avg_final_pnl": round(avg_final, 4),
                    "sig_lost_pct": sig_lost_pct,
                    "avg_sig_lost_min": round(avg_sig_lost_min, 1) if avg_sig_lost_min is not None else None,
                    "avg_pnl_at_sig_lost": round(avg_pnl_at_sig_lost, 4) if avg_pnl_at_sig_lost is not None else None,
                    "avg_reachable_peak": round(avg_reachable_peak, 4) if avg_reachable_peak is not None else None,
                    "rsi_exit_pct": rsi_exit_pct,
                    "avg_rsi_exit_min": round(avg_rsi_exit_min, 1) if avg_rsi_exit_min is not None else None,
                    "avg_rsi_exit_pnl": round(avg_rsi_exit_pnl, 4) if avg_rsi_exit_pnl is not None else None,
                    "rsi3_exit_pct": rsi3_exit_pct,
                    "avg_rsi3_exit_min": round(avg_rsi3_exit_min, 1) if avg_rsi3_exit_min is not None else None,
                    "avg_rsi3_exit_pnl": round(avg_rsi3_exit_pnl, 4) if avg_rsi3_exit_pnl is not None else None,
                    "recovery_005_pct": round(rec_005 / count * 100, 1),
                    "recovery_010_pct": round(rec_010 / count * 100, 1),
                    "sig_regained_pct": sig_regained_pct,
                    "avg_sig_regained_min": round(avg_sig_regained_min, 1) if avg_sig_regained_min is not None else None,
                    "avg_pnl_at_sig_regained": round(avg_pnl_at_sig_regained, 4) if avg_pnl_at_sig_regained is not None else None,
                })
    except Exception as e:
        logger.error(f"[PERF] Error computing Post-Exit Regret deep dive: {e}\n{traceback.format_exc()}")

    # Hold-Time Expectancy
    hold_time_expectancy = []
    try:
        ht_buckets = [
            ("<3m", 0, 3), ("3-8m", 3, 8), ("8-15m", 8, 15),
            ("15-30m", 15, 30), ("30-60m", 30, 60), ("60m+", 60, 999999),
        ]
        ht_orders = [o for o in orders if o.opened_at and o.closed_at]
        for label, lo_min, hi_min in ht_buckets:
            bucket = [o for o in ht_orders
                      if lo_min <= (o.closed_at - o.opened_at).total_seconds() / 60.0 < hi_min]
            ht_count = len(bucket)
            if ht_count == 0:
                continue
            ht_longs = sum(1 for o in bucket if o.direction == "LONG")
            ht_shorts = ht_count - ht_longs
            ht_winners = [o for o in bucket if (o.pnl or 0) > 0]
            ht_losers = [o for o in bucket if (o.pnl or 0) <= 0]
            ht_wr = len(ht_winners) / ht_count
            ht_avg_win = sum(o.pnl or 0 for o in ht_winners) / len(ht_winners) if ht_winners else 0
            ht_avg_loss = sum(o.pnl or 0 for o in ht_losers) / len(ht_losers) if ht_losers else 0
            ht_expectancy = ht_wr * ht_avg_win + (1 - ht_wr) * ht_avg_loss
            ht_peaks = [o.peak_pnl for o in bucket if o.peak_pnl is not None]
            ht_avg_peak = sum(ht_peaks) / len(ht_peaks) if ht_peaks else 0
            ht_avg_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket) / ht_count
            ht_total_pnl = sum(o.pnl or 0 for o in bucket)
            ht_conf = {}
            for o in bucket:
                c = o.confidence or "UNKNOWN"
                ht_conf[c] = ht_conf.get(c, 0) + 1
            hold_time_expectancy.append({
                "duration": label,
                "trades": ht_count,
                "longs": ht_longs,
                "shorts": ht_shorts,
                "win_rate": round(ht_wr * 100, 1),
                "avg_win": round(ht_avg_win, 2),
                "avg_loss": round(ht_avg_loss, 2),
                "expectancy": round(ht_expectancy, 2),
                "avg_peak_pct": round(ht_avg_peak, 4),
                "avg_pnl_pct": round(ht_avg_pnl_pct, 4),
                "total_pnl": round(ht_total_pnl, 2),
                "by_confidence": ht_conf,
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing Hold-Time Expectancy: {e}\n{traceback.format_exc()}")

    return {
        "total_trades": total_trades,
        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "total_wins": len(wins),
        "total_losses": len(losses),
        "win_rate": round(win_rate, 2),
        "win_rate_longs": round(win_rate_longs, 2),
        "win_rate_shorts": round(win_rate_shorts, 2),
        "avg_win": round(avg_win, 2),
        "avg_win_long": round(avg_win_long, 2),
        "avg_win_short": round(avg_win_short, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_loss_long": round(avg_loss_long, 2),
        "avg_loss_short": round(avg_loss_short, 2),
        "expectancy": round(expectancy, 2),
        "best_win_long": round(best_win_long, 2),
        "best_win_short": round(best_win_short, 2),
        "worst_loss_long": round(worst_loss_long, 2),
        "worst_loss_short": round(worst_loss_short, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_percentage": round(total_pnl_percentage, 2),
        "total_pnl_notional_percentage": round(total_pnl / total_investment_notional * 100, 2) if total_investment_notional > 0 else 0,
        "total_investment_notional": round(total_investment_notional, 2),
        "total_investment_value": round(total_investment_value, 2),
        "total_investment_long_notional": round(total_investment_long_notional, 2),
        "total_investment_long_value": round(total_investment_long_value, 2),
        "total_investment_short_notional": round(total_investment_short_notional, 2),
        "total_investment_short_value": round(total_investment_short_value, 2),
        "total_fees": round(total_fees, 2),
        "avg_duration": calc_avg_duration(orders),
        "avg_duration_long": calc_avg_duration(longs),
        "avg_duration_short": calc_avg_duration(shorts),
        "avg_leverage": round(avg_leverage, 2),
        "return_multiple": round(return_multiple, 4),
        "daily_compound_return": round(daily_compound_return, 4),
        "runtime_days": round(runtime_days, 2),
        "by_confidence": confidence_performance,
        "by_macro_trend": macro_trend_performance,
        "outcome_distribution": outcome_distribution,
        "gap_performance": gap_performance,
        "ema58_gap_performance": ema58_gap_performance,
        "rsi_performance": rsi_performance,
        "adx_performance": adx_performance,
        "adx_direction_performance": adx_direction_performance,
        "stretch_performance": stretch_performance,
        "pair_slope_performance": pair_slope_performance,
        "btc_slope_performance": btc_slope_performance,
        "btc_adx_performance": btc_adx_performance,
        "btc_adx_direction_performance": btc_adx_direction_performance,
        "adx_dir_crosstab": adx_dir_crosstab,
        "btc_slope_adx_crosstab": btc_slope_adx_crosstab,
        "by_close_reason": by_close_reason,
        "stop_loss_deep_dive": stop_loss_deep_dive,
        "winning_trades_drawdown": winning_trades_drawdown,
        "trough_recovery": trough_recovery,
        "rsi_adx_crosstab": rsi_adx_crosstab,
        "never_positive_deep_dive": never_positive_deep_dive,
        "performance_over_time": _compute_time_buckets(orders),
        "by_entry_type": _compute_entry_type_stats(orders),
        "by_exit_type": _compute_exit_type_stats(orders),
        "post_exit_regret_deep_dive": post_exit_regret_deep_dive,
        "hold_time_expectancy": hold_time_expectancy,
        "entry_conditions_by_reason": entry_conditions_by_reason
    }


# ----- Configuration -----

@app.get("/api/config")
async def get_config():
    """Get current trading configuration"""
    config = load_trading_config()
    return config.model_dump()


@app.put("/api/config")
async def update_config(config_update: ConfigUpdate):
    """Update trading configuration"""
    global trading_config
    
    current_config = load_trading_config()
    update_data = config_update.model_dump(exclude_none=True)
    
    # Update nested configs
    if 'investment' in update_data:
        current_investment = current_config.investment.model_dump()
        current_investment.update(update_data['investment'])
        update_data['investment'] = InvestmentConfig(**current_investment)
    
    if 'thresholds' in update_data:
        current_thresholds = current_config.thresholds.model_dump()
        current_thresholds.update(update_data['thresholds'])
        update_data['thresholds'] = SignalThresholds(**current_thresholds)
    
    if 'confidence_levels' in update_data:
        current_levels = {k: v.model_dump() for k, v in current_config.confidence_levels.items()}
        for level, values in update_data['confidence_levels'].items():
            if level in current_levels:
                current_levels[level].update(values)
            else:
                current_levels[level] = values
        update_data['confidence_levels'] = {k: ConfidenceConfig(**v) for k, v in current_levels.items()}
    
    # Create new config
    new_config_data = current_config.model_dump()
    new_config_data.update(update_data)
    new_config = TradingConfig(**new_config_data)
    
    # Detect changes between old and new config
    def flatten_config(cfg_dict, prefix=""):
        flat = {}
        for k, v in cfg_dict.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(flatten_config(v, f"{key}."))
            else:
                flat[key] = v
        return flat

    old_flat = flatten_config(current_config.model_dump())
    new_flat = flatten_config(new_config.model_dump())
    changes = []
    for key in set(old_flat) | set(new_flat):
        old_val = old_flat.get(key)
        new_val = new_flat.get(key)
        if old_val != new_val:
            changes.append({"field": key, "old": str(old_val), "new": str(new_val)})

    # Save
    if save_trading_config(new_config):
        # Log changes to DB
        if changes:
            try:
                async with AsyncSessionLocal() as db:
                    for ch in changes:
                        db.add(ConfigChangeLog(
                            field=ch["field"],
                            old_value=ch["old"],
                            new_value=ch["new"]
                        ))
                    await db.commit()
                print(f"Config updated with {len(changes)} change(s) logged.")
            except Exception as e:
                print(f"Config saved but failed to log changes: {e}")

        # Reload global config
        import config
        config.trading_config = new_config
        return {"status": "success", "message": f"Configuration saved ({len(changes)} change(s))"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save configuration")


@app.get("/api/config/changelog")
async def get_config_changelog(limit: int = 50):
    """Get recent config change history"""
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(ConfigChangeLog)
            .order_by(ConfigChangeLog.changed_at.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [{
            "field": r.field,
            "old_value": r.old_value,
            "new_value": r.new_value,
            "changed_at": r.changed_at.isoformat() if r.changed_at else None
        } for r in rows]


@app.put("/api/config/pairs-limit")
async def update_pairs_limit(data: dict):
    """Update trading pairs limit (how many top pairs to trade)"""
    import config
    
    limit = data.get('limit', 50)
    # Validate limit
    if limit not in [5, 10, 20, 50, 100]:
        raise HTTPException(status_code=400, detail="Limit must be 5, 10, 20, 50, or 100")
    
    current_config = load_trading_config()
    current_config.trading_pairs_limit = limit
    
    if save_trading_config(current_config):
        config.trading_config = current_config
        return {"success": True, "trading_pairs_limit": limit}
    else:
        raise HTTPException(status_code=500, detail="Failed to save configuration")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

