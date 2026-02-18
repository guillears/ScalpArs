"""
SCALPARS Trading Platform - Main Application
"""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
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
from models import Order, Transaction, BotState, PairData
import config
from config import (
    trading_config, save_trading_config, load_trading_config,
    TradingConfig, ConfidenceConfig, SignalThresholds, InvestmentConfig
)
from services.binance_service import binance_service
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
background_task = None
monitor_task = None
should_stop = False


async def trading_loop():
    """Main trading loop that runs in background"""
    global should_stop
    logger.info("[LOOP] Trading loop started")
    while not should_stop:
        try:
            async with AsyncSessionLocal() as db:
                await trading_engine.initialize(db)
                
                logger.info(f"[LOOP] Cycle - is_running: {trading_engine.is_running}, is_paper: {trading_engine.is_paper_mode}")
                
                if trading_engine.is_running:
                    # Scan and open new positions
                    await trading_engine.scan_and_trade(db)
                
                # Always update and check open positions (even when paused)
                await trading_engine.update_open_positions(db)
                
                # Update orders cache for real-time stop loss checking
                await trading_engine.update_orders_cache(db)
        except Exception as e:
            logger.error(f"[ERROR] Error in trading loop: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait 1 second before next iteration (real-time price tracking)
        await asyncio.sleep(1)


async def start_background_tasks():
    """Start background trading tasks"""
    global background_task, should_stop
    should_stop = False
    
    # Start WebSocket tracker for real-time price tracking
    await websocket_tracker.start()
    logger.info("[STARTUP] WebSocket price tracker started")
    
    # Register real-time stop loss callback
    websocket_tracker.set_price_callback(realtime_stop_loss_callback)
    logger.info("[STARTUP] Real-time stop loss callback registered")
    
    background_task = asyncio.create_task(trading_loop())


async def stop_background_tasks():
    """Stop background trading tasks"""
    global background_task, should_stop
    should_stop = True
    
    # Stop WebSocket tracker
    await websocket_tracker.stop()
    logger.info("[SHUTDOWN] WebSocket price tracker stopped")
    
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("[STARTUP] Initializing database...")
    await init_db()
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
    return templates.TemplateResponse("index.html", {"request": request})


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
    
    # Save the reset state
    await trading_engine.save_state(db)
    
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
        # Calculate paper trading balance
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        open_orders = result.scalars().all()
        used_margin = sum(o.investment for o in open_orders)
        
        return {
            "usdt_balance": trading_engine.paper_balance,
            "bnb_balance": 0,
            "usdt_in_orders": used_margin,
            "total_portfolio": trading_engine.paper_balance + used_margin,
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
    limit = min(max(limit, 5), 50)
    
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
            "volume_24h": p.volume_24h,
            "open_positions": {
                "long": long_count,
                "short": short_count
            }
        })
    
    return pairs_data


@app.post("/api/pairs/refresh")
async def refresh_pairs(db: AsyncSession = Depends(get_db)):
    """Force refresh pair data"""
    top_pairs = await binance_service.get_top_futures_pairs(50)
    
    for pair_info in top_pairs:
        symbol = pair_info['symbol']
        pair = pair_info['pair']
        volume_24h = pair_info['volume_24h']  # Get 24h volume from tickers
        
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
        
        # Pass the 24h volume from tickers, not from OHLCV
        await trading_engine.update_pair_data(db, pair, indicators, signal, confidence, volume_24h)
    
    return {"status": "refreshed", "count": len(top_pairs)}


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
            estimated_exit_fee = current_notional * config.trading_config.trading_fee
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
            # Dynamic TP tracking
            "tp_level": o.current_tp_level or 1,
            "tp_target": o.dynamic_tp_target or 0
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
            "close_reason": o.close_reason,
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "opened_at": o.opened_at.isoformat(),
            "closed_at": o.closed_at.isoformat() if o.closed_at else None,
            # TP Level reached
            "tp_level": o.current_tp_level or 1
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
async def get_performance(db: AsyncSession = Depends(get_db)):
    """Get closed orders performance metrics"""
    await trading_engine.initialize(db)
    
    result = await db.execute(
        select(Order)
        .where(and_(Order.status == "CLOSED", Order.is_paper == trading_engine.is_paper_mode))
    )
    orders = result.scalars().all()
    
    if not orders:
        return {
            "total_trades": 0,
            "total_longs": 0,
            "total_shorts": 0,
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
            "return_multiple": round(trading_engine.paper_balance / config.trading_config.paper_balance, 4) if config.trading_config.paper_balance > 0 else 0,
            "daily_compound_return": 0,
            "runtime_days": round(trading_engine.get_runtime_seconds() / 86400.0, 2),
            "by_confidence": {},
            "outcome_distribution": [],
            "gap_performance": [],
            "by_close_reason": {}
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
    current_balance = trading_engine.paper_balance
    runtime_days = trading_engine.get_runtime_seconds() / 86400.0
    return_multiple = current_balance / initial_balance if initial_balance > 0 else 0
    if runtime_days >= 0.01 and return_multiple > 0:
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
    for conf in ['LOW', 'MEDIUM', 'HIGH', 'EXTREME']:
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
        ("> +0.8%", lambda pct: pct > 0.8),
        ("+0.4% to +0.8%", lambda pct: 0.4 < pct <= 0.8),
        ("0% to +0.4%", lambda pct: 0 < pct <= 0.4),
        ("-0.8% to 0%", lambda pct: -0.8 < pct <= 0),
        ("-1% to -0.8%", lambda pct: -1 < pct <= -0.8),
        ("-1.2% to -1%", lambda pct: -1.2 < pct <= -1),
        ("-1.5% to -1.2%", lambda pct: -1.5 < pct <= -1.2),
        ("< -1.5%", lambda pct: pct <= -1.5)
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
    
    # Performance by Entry Gap Range
    gap_ranges = [
        ("0.06 - 0.10%", 0.06, 0.10),
        ("0.10 - 0.15%", 0.10, 0.15),
        ("0.15 - 0.20%", 0.15, 0.20),
        ("0.20 - 0.25%", 0.20, 0.25),
        ("0.25 - 0.30%", 0.25, 0.30),
        ("0.30 - 0.35%", 0.30, 0.35),
        ("0.35 - 0.40%", 0.35, 0.40),
        ("0.40 - 0.50%", 0.40, 0.50),
        ("0.50 - 0.60%", 0.50, 0.60),
        ("0.60 - 0.70%", 0.60, 0.70),
        ("0.70 - 0.80%", 0.70, 0.80),
        ("> 0.80%", 0.80, 999),
    ]
    
    # Only include orders with entry_gap tracked
    gap_orders = [o for o in orders if o.entry_gap is not None]
    
    gap_performance = []
    for range_name, gap_min, gap_max in gap_ranges:
        range_orders = [o for o in gap_orders if gap_min <= o.entry_gap < gap_max]
        count = len(range_orders)
        if count == 0:
            gap_performance.append({
                "range": range_name,
                "count": 0,
                "win_rate": 0,
                "avg_pnl_usd": 0,
                "by_confidence": {}
            })
            continue
        
        range_wins = len([o for o in range_orders if (o.pnl or 0) > 0])
        range_pnl_sum = sum(o.pnl or 0 for o in range_orders)
        
        # Confidence breakdown
        conf_breakdown = {}
        for o in range_orders:
            conf = o.confidence or "UNKNOWN"
            conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
        
        gap_performance.append({
            "range": range_name,
            "count": count,
            "win_rate": round(range_wins / count * 100, 1),
            "avg_pnl_usd": round(range_pnl_sum / count, 2),
            "total_pnl_usd": round(range_pnl_sum, 2),
            "by_confidence": conf_breakdown
        })
    
    # By Close Reason - group by reason with L4+ aggregation
    close_reason_stats = {}
    for o in orders:
        reason = o.close_reason or "UNKNOWN"
        
        # Normalize reason: group L4, L5, L6... into L4+
        if " L" in reason:
            parts = reason.split(" L")
            base_reason = parts[0]
            try:
                level = int(parts[1].replace("+", ""))
                if level >= 4:
                    reason = f"{base_reason} L4+"
                # Keep L1, L2, L3 as-is
            except ValueError:
                pass  # Keep original if parsing fails
        
        if reason not in close_reason_stats:
            close_reason_stats[reason] = {"trades": [], "pnl_sum": 0, "pnl_pct_sum": 0, "by_confidence": {}}
        
        close_reason_stats[reason]["trades"].append(o)
        close_reason_stats[reason]["pnl_sum"] += o.pnl or 0
        close_reason_stats[reason]["pnl_pct_sum"] += o.pnl_percentage or 0
        
        # Track confidence breakdown per close reason
        conf = o.confidence or "UNKNOWN"
        close_reason_stats[reason]["by_confidence"][conf] = close_reason_stats[reason]["by_confidence"].get(conf, 0) + 1
    
    by_close_reason = {}
    for reason, data in close_reason_stats.items():
        count = len(data["trades"])
        by_close_reason[reason] = {
            "trades": count,
            "avg_pnl_pct": round(data["pnl_pct_sum"] / count, 2) if count > 0 else 0,
            "avg_pnl_usd": round(data["pnl_sum"] / count, 2) if count > 0 else 0,
            "total_pnl_usd": round(data["pnl_sum"], 2),
            "by_confidence": data["by_confidence"]
        }
    
    return {
        "total_trades": total_trades,
        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "win_rate": round(win_rate, 2),
        "win_rate_longs": round(win_rate_longs, 2),
        "win_rate_shorts": round(win_rate_shorts, 2),
        "avg_win": round(avg_win, 2),
        "avg_win_long": round(avg_win_long, 2),
        "avg_win_short": round(avg_win_short, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_loss_long": round(avg_loss_long, 2),
        "avg_loss_short": round(avg_loss_short, 2),
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
        "outcome_distribution": outcome_distribution,
        "gap_performance": gap_performance,
        "by_close_reason": by_close_reason
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
    
    # Save
    if save_trading_config(new_config):
        # Reload global config - need to update the module-level variable
        import config
        config.trading_config = new_config
        print(f"Config updated. LOW enabled: {new_config.confidence_levels['LOW'].enabled}")
        return {"status": "success", "message": "Configuration saved"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save configuration")


@app.put("/api/config/pairs-limit")
async def update_pairs_limit(data: dict):
    """Update trading pairs limit (how many top pairs to trade)"""
    import config
    
    limit = data.get('limit', 50)
    # Validate limit
    if limit not in [5, 10, 20, 50]:
        raise HTTPException(status_code=400, detail="Limit must be 5, 10, 20, or 50")
    
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

