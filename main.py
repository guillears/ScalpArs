"""
SCALPARS Trading Platform - Main Application
"""
import asyncio
import math
import time
import traceback
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from sqlalchemy import select, and_, func, desc, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database import init_db, get_db, AsyncSessionLocal
from models import Order, Transaction, BotState, PairData, ConfigChangeLog, BnbSwapLog, Investor
import config
from config import (
    trading_config, save_trading_config, load_trading_config,
    TradingConfig, ConfidenceConfig, SignalThresholds, InvestmentConfig
)
from services.binance_service import binance_service, set_ban_until, set_ban_persist_callback, get_ban_status
from services.trading_engine import trading_engine, realtime_stop_loss_callback, _open_orders_cache, _cache_lock
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

# Live Terminal — attach the logging handler that captures every LogRecord
# into a thread-safe ring buffer for the /api/terminal/stream SSE endpoint.
# Purely additive: it sits alongside the existing stdout StreamHandler.
# Never raises (see live_terminal/handler.py).
from live_terminal.handler import install as _install_terminal_handler
from live_terminal.stream import (
    start_heartbeat as _start_terminal_heartbeat,
    stop_heartbeat as _stop_terminal_heartbeat,
    event_stream_generator as _terminal_event_stream,
)
_install_terminal_handler()

# Background task control
_scan_task = None
_monitor_task = None
_bnb_swap_task = None
should_stop = False
_scan_lock = asyncio.Lock()


async def monitor_loop():
    """Fast loop (1s) for SL/TP checks and cache updates.
    
    Runs independently of scan_loop so stop-loss monitoring is never
    blocked by slow API calls in scan_and_trade.
    Also runs periodic reconciliation every ~60s (live mode only).
    """
    global should_stop
    logger.info("[MONITOR] Monitor loop started (1s cycle)")
    _reconcile_counter = 0
    _RECONCILE_INTERVAL = 60
    while not should_stop:
        try:
            async with AsyncSessionLocal() as db:
                await trading_engine.initialize(db)
                await trading_engine.update_open_positions(db)
                await trading_engine.update_orders_cache(db)
                await trading_engine.update_post_exit_tracking(db)

                _reconcile_counter += 1
                if _reconcile_counter >= _RECONCILE_INTERVAL and not trading_engine.is_paper_mode:
                    _reconcile_counter = 0
                    try:
                        closed = await _reconcile_open_orders(db)
                        if closed:
                            logger.info(f"[MONITOR_RECONCILE] Auto-reconciled {len(closed)} orphan order(s)")
                    except Exception as e:
                        logger.error(f"[MONITOR_RECONCILE] Error during auto-reconciliation: {e}")
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


async def bnb_swap_loop():
    """Periodic loop to check BNB balance and auto-swap USDT to BNB for fee coverage."""
    global should_stop
    interval = config.trading_config.bnb_check_interval_hours * 3600
    logger.info(f"[BNB_LOOP] BNB swap loop started (every {config.trading_config.bnb_check_interval_hours}h)")
    await asyncio.sleep(60)
    while not should_stop:
        try:
            if config.trading_config.bnb_swap_enabled:
                async with AsyncSessionLocal() as db:
                    await trading_engine.initialize(db)
                    await trading_engine.bnb_scheduled_check(db)
        except Exception as e:
            logger.error(f"[BNB_LOOP] Error in BNB swap loop: {e}")
            import traceback
            traceback.print_exc()
        await asyncio.sleep(interval)


async def start_background_tasks():
    """Start background trading tasks"""
    global _scan_task, _monitor_task, _bnb_swap_task, should_stop
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
    _bnb_swap_task = asyncio.create_task(bnb_swap_loop())
    logger.info("[STARTUP] Monitor, scan, and BNB swap loops started independently")


async def stop_background_tasks():
    """Stop background trading tasks"""
    global _scan_task, _monitor_task, _bnb_swap_task, should_stop
    should_stop = True
    
    # Stop WebSocket tracker
    await websocket_tracker.stop()
    logger.info("[SHUTDOWN] WebSocket price tracker stopped")
    
    for task in (_monitor_task, _scan_task, _bnb_swap_task):
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
    
    # Live Terminal — start background heartbeat task once event loop is up
    _start_terminal_heartbeat()

    logger.info("[STARTUP] SCALPARS Trading Platform started")
    yield
    # Shutdown
    logger.info("[SHUTDOWN] Stopping background tasks...")
    _stop_terminal_heartbeat()
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

LOGIN_PASSWORD = os.environ.get("LOGIN_PASSWORD", "guille86")

class AuthMiddleware(BaseHTTPMiddleware):
    OPEN_PATHS = {"/login", "/static", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if any(path == p or path.startswith(p + "/") for p in self.OPEN_PATHS):
            return await call_next(request)
        if not request.session.get("authenticated"):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "Not authenticated"}, status_code=401)
            return RedirectResponse(url="/login", status_code=302)
        return await call_next(request)

app.add_middleware(AuthMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SESSION_SECRET", "scalpars-s3cr3t-k3y-ch4ng3-m3"),
    max_age=60 * 60 * 24 * 30,
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
    revalidate_on_taker_fallback: Optional[bool] = None
    maker_exit_enabled: Optional[bool] = None
    maker_exit_timeout_seconds: Optional[int] = None
    maker_exit_offset_ticks: Optional[int] = None
    paper_trading: Optional[bool] = None
    paper_balance: Optional[float] = None
    # Top-level fields previously missing from this schema — the UI was
    # sending them in save payloads but Pydantic was silently dropping them
    # (Pydantic v2 default: extras ignored).  Adding them here makes UI
    # updates to these fields actually persist.  Backfilled Apr 17.
    pair_blacklist: Optional[str] = None
    trading_pairs_limit: Optional[int] = None
    new_listing_filter_days: Optional[int] = None
    alpha_subtype_filter_enabled: Optional[bool] = None
    bnb_swap_enabled: Optional[bool] = None
    bnb_check_interval_hours: Optional[int] = None
    bnb_runway_hours: Optional[int] = None
    paper_bnb_initial_usd: Optional[float] = None
    investment: Optional[Dict] = None
    thresholds: Optional[Dict] = None
    confidence_levels: Optional[Dict] = None


class ManualCloseRequest(BaseModel):
    order_id: int


# ============== Auth Routes ==============

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page"""
    if request.session.get("authenticated"):
        return RedirectResponse(url="/", status_code=302)
    with open("templates/login.html", "r") as f:
        return HTMLResponse(content=f.read())


@app.post("/login")
async def login(request: Request, password: str = Form(...)):
    """Validate password and create session"""
    if password == LOGIN_PASSWORD:
        request.session["authenticated"] = True
        return RedirectResponse(url="/", status_code=302)
    return RedirectResponse(url="/login?error=1", status_code=302)


@app.get("/logout")
async def logout(request: Request):
    """Clear session and redirect to login"""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=302)


# ============== API Routes ==============

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main page"""
    with open("templates/index.html", "r") as f:
        return HTMLResponse(content=f.read())


# ----- Live Terminal (UI-only observability) -----

@app.get("/api/terminal/stream")
async def terminal_event_stream(request: Request):
    """Server-Sent Events stream for the Live Terminal view.

    Replays the current ring buffer on connect, then streams new log
    events as they arrive. See live_terminal/stream.py for details.
    Purely additive — does not touch trading logic.
    """
    return StreamingResponse(
        _terminal_event_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx/proxy buffering
            "Connection": "keep-alive",
        },
    )


# ----- Bot Status -----

@app.get("/api/status")
async def get_status(db: AsyncSession = Depends(get_db)):
    """Get bot status"""
    await trading_engine.initialize(db)
    # May 22: refresh BNB burn rate every status poll (cheap SQL aggregate).
    # Previously the metric only recomputed every bnb_check_interval_hours (6h),
    # leaving Runway/Burn empty for hours after a reset until the swap loop fired.
    # See CLAUDE.md May 11 — burn-rate recompute was already decoupled from swap
    # gate; this just makes it UI-poll-driven so the runway stays current.
    try:
        await trading_engine._recompute_bnb_burn_rate(db)
    except Exception as e:
        logger.debug(f"[STATUS] burn rate recompute skipped: {e}")
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
async def reset_trading(direction: str = "ALL", db: AsyncSession = Depends(get_db)):
    """Reset trading data. direction=ALL|LONG|SHORT"""
    import config

    is_paper = trading_engine.is_paper_mode
    mode_label = "Paper" if is_paper else "Live"
    direction = direction.upper()

    if direction == "ALL":
        # Full reset — original behavior
        if trading_engine.is_running:
            await trading_engine.pause(db)

        await db.execute(
            delete(Order).where(Order.is_paper == is_paper)
        )
        await db.execute(
            delete(Transaction).where(Transaction.is_paper == is_paper)
        )
        await db.execute(
            delete(BnbSwapLog).where(BnbSwapLog.is_paper == is_paper)
        )

        if is_paper:
            trading_engine.paper_balance = config.trading_config.paper_balance
            trading_engine.paper_bnb_balance_usd = config.trading_config.paper_bnb_initial_usd

        trading_engine.total_runtime_seconds = 0
        trading_engine.started_at = None
        trading_engine.is_running = False
        trading_engine._bnb_emergency_threshold = 0.0
        trading_engine._bnb_projected_need = 0.0
        trading_engine._bnb_burn_rate = 0.0
        trading_engine._last_bnb_check = None

        # Reset Filter Block counters (in-memory + persisted)
        trading_engine._filter_block_counts = {}

        set_ban_until(0)
        await trading_engine.save_state(db)

        result_state = await db.execute(select(BotState).limit(1))
        state_row = result_state.scalar_one_or_none()
        if state_row:
            state_row.ban_until = 0
            state_row.filter_block_counts_json = None

        await db.commit()

        balance = trading_engine.paper_balance if is_paper else 0
        logger.info(f"[RESET] {mode_label} ALL trading reset. {'Balance: $' + str(balance) + ', ' if is_paper else ''}Timer: 00:00:00")

        return {
            "success": True,
            "message": f"{mode_label} trading reset successfully (ALL)",
            "paper_balance": trading_engine.paper_balance if is_paper else None,
            "mode": "paper" if is_paper else "live",
            "runtime": "00:00:00",
            "direction": "ALL"
        }

    elif direction in ("LONG", "SHORT"):
        # Partial reset — only delete orders/transactions for the specified direction
        # Get order IDs to delete matching transactions
        order_ids_result = await db.execute(
            select(Order.id).where(
                and_(Order.is_paper == is_paper, Order.direction == direction)
            )
        )
        order_ids = [row[0] for row in order_ids_result.fetchall()]

        if order_ids:
            await db.execute(
                delete(Transaction).where(Transaction.order_id.in_(order_ids))
            )
            await db.execute(
                delete(Order).where(Order.id.in_(order_ids))
            )

        await db.commit()

        # Recalculate paper balance from remaining trades
        if is_paper:
            await trading_engine._recalculate_paper_balance(db)
            await trading_engine.save_state(db)

        logger.info(f"[RESET] {mode_label} {direction} trades reset. Deleted {len(order_ids)} orders.")

        return {
            "success": True,
            "message": f"{mode_label} {direction} trades reset ({len(order_ids)} orders deleted)",
            "paper_balance": trading_engine.paper_balance if is_paper else None,
            "mode": "paper" if is_paper else "live",
            "direction": direction,
            "deleted_count": len(order_ids)
        }

    else:
        return {"success": False, "message": f"Invalid direction: {direction}. Use ALL, LONG, or SHORT."}


# ----- Balance -----

@app.get("/api/balance")
async def get_balance(db: AsyncSession = Depends(get_db)):
    """Get account balance"""
    await trading_engine.initialize(db)
    
    if trading_engine.is_paper_mode:
        balance = await trading_engine._recalculate_paper_balance(db)
        await trading_engine._recalculate_paper_bnb(db)
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        open_orders = result.scalars().all()
        used_margin = sum(o.investment for o in open_orders)
        bnb_usd = trading_engine.paper_bnb_balance_usd
        
        return {
            "usdt_balance": balance,
            "bnb_balance": round(bnb_usd, 2),
            "bnb_balance_is_usd": True,
            "usdt_in_orders": used_margin,
            "total_portfolio": balance + used_margin + bnb_usd,
            "is_paper": True
        }
    else:
        balance = await binance_service.get_balance()
        bnb_price = await binance_service.get_bnb_price()
        bnb_usd = balance['bnb_total'] * bnb_price if bnb_price > 0 else 0
        usdt_free = balance['usdt_free']
        total = balance['usdt_total'] + bnb_usd
        return {
            "usdt_balance": usdt_free,
            "bnb_balance": balance['bnb_total'],
            "bnb_balance_usd": round(bnb_usd, 2),
            "bnb_balance_is_usd": False,
            "usdt_in_orders": balance['usdt_used'],
            "total_portfolio": round(total, 2),
            "is_paper": False
        }


@app.get("/api/bnb-swaps")
async def get_bnb_swaps(db: AsyncSession = Depends(get_db)):
    """Get BNB swap history and current status"""
    result = await db.execute(
        select(BnbSwapLog).order_by(desc(BnbSwapLog.timestamp)).limit(50)
    )
    swaps = result.scalars().all()
    return {
        "swaps": [
            {
                "timestamp": s.timestamp.isoformat() if s.timestamp else None,
                "swap_type": s.swap_type,
                "amount_usdt": s.amount_usdt,
                "bnb_price": s.bnb_price,
                "amount_bnb": s.amount_bnb,
                "pre_bnb_usd": s.pre_bnb_usd,
                "post_bnb_usd": s.post_bnb_usd,
                "burn_rate": s.burn_rate,
                "is_paper": s.is_paper,
            }
            for s in swaps
        ],
        "status": {
            "bnb_swap_enabled": config.trading_config.bnb_swap_enabled,
            "burn_rate_per_hour": round(trading_engine._bnb_burn_rate, 2),
            "projected_need": round(trading_engine._bnb_projected_need, 2),
            "emergency_threshold": round(trading_engine._bnb_emergency_threshold, 2),
            "data_mature": getattr(trading_engine, '_bnb_data_mature', False),
            # May 7 — emit timezone-aware ISO so JS interprets as UTC unambiguously
            # (datetime.utcnow() produces naive datetime; without TZ info, JS Date()
            # parsing varies by browser and the displayed time can be off by hours).
            "last_check": (trading_engine._last_bnb_check.replace(tzinfo=timezone.utc).isoformat()
                           if trading_engine._last_bnb_check else None),
            "check_interval_hours": config.trading_config.bnb_check_interval_hours,
            "runway_hours": config.trading_config.bnb_runway_hours,
        }
    }


@app.post("/api/bnb-swaps/manual")
async def manual_bnb_buy(data: dict, db: AsyncSession = Depends(get_db)):
    """Manually buy BNB with a specified USDT amount"""
    amount = data.get("amount", 0)
    if amount <= 0:
        raise HTTPException(400, "Amount must be positive")

    await trading_engine.initialize(db)

    if trading_engine.is_paper_mode:
        bnb_price = await binance_service.get_bnb_price()
        if bnb_price <= 0:
            bnb_price = 600.0
        pre_bnb = trading_engine.paper_bnb_balance_usd
        pre_usdt = trading_engine.paper_balance
        trading_engine.paper_bnb_balance_usd += amount
        swap_log = BnbSwapLog(
            swap_type="manual",
            amount_usdt=amount,
            bnb_price=bnb_price,
            amount_bnb=round(amount / bnb_price, 6),
            pre_bnb_usd=pre_bnb,
            post_bnb_usd=trading_engine.paper_bnb_balance_usd,
            pre_usdt=pre_usdt,
            post_usdt=pre_usdt - amount,
            burn_rate=trading_engine._bnb_burn_rate,
            is_paper=True
        )
        db.add(swap_log)
        await db.commit()
        await trading_engine._recalculate_paper_balance(db)
        await trading_engine.save_state(db)
        return {"ok": True, "bnb_amount": round(amount / bnb_price, 6), "bnb_price": round(bnb_price, 2), "cost_usdt": round(amount, 2)}
    else:
        pre_balance = await binance_service.get_balance()
        pre_bnb_price = await binance_service.get_bnb_price()
        pre_bnb_usd = pre_balance['bnb_total'] * pre_bnb_price if pre_bnb_price > 0 else 0
        pre_usdt = pre_balance['usdt_free']

        result = await binance_service.buy_bnb(amount)
        if not result:
            raise HTTPException(500, "BNB purchase failed — check Binance API logs")
        new_balance = await binance_service.get_balance()
        bnb_price = result['price']
        swap_log = BnbSwapLog(
            swap_type="manual",
            amount_usdt=result['cost_usdt'],
            bnb_price=bnb_price,
            amount_bnb=result['bnb_amount'],
            pre_bnb_usd=pre_bnb_usd,
            post_bnb_usd=new_balance['bnb_total'] * bnb_price,
            pre_usdt=pre_usdt,
            post_usdt=new_balance['usdt_free'],
            burn_rate=trading_engine._bnb_burn_rate,
            is_paper=False
        )
        db.add(swap_log)
        await db.commit()
        return {"ok": True, "bnb_amount": result['bnb_amount'], "bnb_price": round(bnb_price, 2), "cost_usdt": round(result['cost_usdt'], 2)}


@app.post("/api/bnb-swaps/manual-sell")
async def manual_bnb_sell(data: dict, db: AsyncSession = Depends(get_db)):
    """Manually sell BNB for a specified USDT amount equivalent. May 25 — mirror of manual_bnb_buy."""
    amount = data.get("amount", 0)
    if amount <= 0:
        raise HTTPException(400, "Amount must be positive")

    await trading_engine.initialize(db)

    if trading_engine.is_paper_mode:
        # Validate sufficient BNB balance
        available_bnb = trading_engine.paper_bnb_balance_usd or 0
        if available_bnb < amount:
            raise HTTPException(400, f"Insufficient BNB balance: available ${available_bnb:.2f}, requested ${amount:.2f}")

        bnb_price = await binance_service.get_bnb_price()
        if bnb_price <= 0:
            bnb_price = 600.0
        pre_bnb = trading_engine.paper_bnb_balance_usd
        pre_usdt = trading_engine.paper_balance
        trading_engine.paper_bnb_balance_usd -= amount
        if trading_engine.paper_bnb_balance_usd < 0:
            trading_engine.paper_bnb_balance_usd = 0
        # Note: paper_balance is reverse-derived from DB via _recalculate_paper_balance
        # which subtracts total_bnb_swaps. We log this as a NEGATIVE swap (amount_usdt
        # negative) so the reverse-derived balance increases by `amount`.
        swap_log = BnbSwapLog(
            swap_type="manual_sell",
            amount_usdt=-amount,  # negative = USDT inflow (BNB → USDT)
            bnb_price=bnb_price,
            amount_bnb=round(amount / bnb_price, 6),
            pre_bnb_usd=pre_bnb,
            post_bnb_usd=trading_engine.paper_bnb_balance_usd,
            pre_usdt=pre_usdt,
            post_usdt=pre_usdt + amount,
            burn_rate=trading_engine._bnb_burn_rate,
            is_paper=True
        )
        db.add(swap_log)
        await db.commit()
        await trading_engine._recalculate_paper_balance(db)
        await trading_engine.save_state(db)
        return {"ok": True, "bnb_amount": round(amount / bnb_price, 6), "bnb_price": round(bnb_price, 2), "proceeds_usdt": round(amount, 2)}
    else:
        pre_balance = await binance_service.get_balance()
        pre_bnb_price = await binance_service.get_bnb_price()
        pre_bnb_usd = pre_balance['bnb_total'] * pre_bnb_price if pre_bnb_price > 0 else 0
        pre_usdt = pre_balance['usdt_free']

        if pre_bnb_usd < amount:
            raise HTTPException(400, f"Insufficient BNB balance: available ${pre_bnb_usd:.2f}, requested ${amount:.2f}")

        result = await binance_service.sell_bnb(amount)
        if not result:
            raise HTTPException(500, "BNB sell failed — check Binance API logs")
        new_balance = await binance_service.get_balance()
        bnb_price = result['price']
        swap_log = BnbSwapLog(
            swap_type="manual_sell",
            amount_usdt=-result['proceeds_usdt'],  # negative = USDT inflow
            bnb_price=bnb_price,
            amount_bnb=result['bnb_amount'],
            pre_bnb_usd=pre_bnb_usd,
            post_bnb_usd=new_balance['bnb_total'] * bnb_price,
            pre_usdt=pre_usdt,
            post_usdt=new_balance['usdt_free'],
            burn_rate=trading_engine._bnb_burn_rate,
            is_paper=False
        )
        db.add(swap_log)
        await db.commit()
        return {"ok": True, "bnb_amount": result['bnb_amount'], "bnb_price": round(bnb_price, 2), "proceeds_usdt": round(result['proceeds_usdt'], 2)}


# ----- Market Data -----

@app.get("/api/pairs")
async def get_pairs(db: AsyncSession = Depends(get_db), limit: int = 50):
    """Get top pairs with indicators.

    May 8 fix: filter by `updated_at` recency. PairData rows accumulate over
    time — pairs that were once in the top-50 but have since dropped out
    keep their stale volume_24h value forever (the bot's update path only
    touches pairs currently in the scan list, never deletes old rows).
    Without this filter the dashboard's "Top Pairs by Volume" table was
    showing historical pumps (e.g., HIGHUSDT at $1.35B from a past
    launchpad spike, when current real volume was $38M). The actual
    trading list is unaffected — `scan_and_trade` always pulls fresh data
    from CCXT — but the UI display was misleading.
    """
    # Validate limit
    limit = min(max(limit, 5), 100)

    # Only show pairs updated in the last 10 minutes — covers normal scan
    # cadence (~60s per cycle) with comfortable margin for slow scans.
    _stale_cutoff = datetime.utcnow() - timedelta(minutes=10)

    # Get pairs from cache, filtered to recently-scanned only.
    result = await db.execute(
        select(PairData)
        .where(PairData.updated_at >= _stale_cutoff)
        .order_by(desc(PairData.volume_24h))
        .limit(limit)
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

        # Calculate EMA5-EMA8 gap
        gap_5_8 = None
        if p.ema5 and p.ema8 and p.ema8 > 0:
            gap_5_8 = round(abs((p.ema5 - p.ema8) / p.ema8) * 100, 4)

        # Determine block reason for NO_TRADE / NOTHING signals
        # Block reason: read from engine's per-pair stash (May 26).
        # Single source of truth — every filter that fires _record_filter_block
        # also stamps _last_pair_block_reason[pair] = tag. No UI enumeration drift.
        block_reason = None
        if p.signal in (None, "NOTHING", "NO_TRADE"):
            import services.trading_engine as _te
            engine = getattr(_te, 'trading_engine', None)
            if engine and hasattr(engine, '_last_pair_block_reason'):
                block_reason = engine._last_pair_block_reason.get(p.pair)
            if not block_reason:
                block_reason = "Awaiting scan (no pair data yet)"

        pairs_data.append({
            "pair": p.pair,
            "price": display_price,
            "ema5": round(p.ema5, 2) if p.ema5 else None,
            "ema8": round(p.ema8, 2) if p.ema8 else None,
            "ema13": round(p.ema13, 2) if p.ema13 else None,
            "ema20": round(p.ema20, 2) if p.ema20 else None,
            "gap": gap,
            "gap_5_8": gap_5_8,
            "rsi": round(p.rsi, 2) if p.rsi else None,
            "adx": round(p.adx, 2) if p.adx else None,
            "signal": p.signal,
            "confidence": p.confidence,
            "macro_regime": p.macro_regime,
            "volume_24h": p.volume_24h,
            "block_reason": block_reason,
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
                    price=indicators.get('close'),
                    adx_prev1=indicators.get('adx_prev1'),
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

        cached_flagged = False
        cached_fl2_flagged = False
        cached_fl1_origin = None
        # Cache peak/trough are updated by _realtime_callback on every WS tick,
        # while DB peak/trough only get updated on monitor-loop polls. The cache
        # is therefore fresher — use it preferentially for the open-orders display.
        cached_peak_pnl = None
        cached_trough_pnl = None
        for ci in _open_orders_cache.get(o.pair, []):
            if ci['id'] == o.id:
                cached_flagged = ci.get('signal_lost_flagged', False)
                cached_fl2_flagged = ci.get('fl2_flagged', False)
                cached_fl1_origin = ci.get('fl1_origin')
                cached_peak_pnl = ci.get('peak_pnl')
                cached_trough_pnl = ci.get('trough_pnl')
                break
        # Fall back to DB values if cache missing (rare — cache can lag briefly)
        if not cached_flagged and getattr(o, 'signal_lost_flagged', False):
            cached_flagged = True
        if not cached_fl2_flagged and getattr(o, 'fl2_flagged', False):
            cached_fl2_flagged = True
        if cached_fl1_origin is None:
            cached_fl1_origin = getattr(o, 'fl1_origin', None)
        
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
            "cell_multiplier": getattr(o, 'cell_multiplier', None),
            "cell_lev_multiplier": getattr(o, 'cell_lev_multiplier', None),
            "cell_multiplier_source": getattr(o, 'cell_multiplier_source', None),
            "pattern_cell_source": getattr(o, 'pattern_cell_source', None),
            "pattern_fixed_tp_pct": getattr(o, 'pattern_fixed_tp_pct', None),
            "pattern_fixed_sl_pct": getattr(o, 'pattern_fixed_sl_pct', None),
            "quantity": o.quantity,
            "entry_price": o.entry_price,
            "current_price": current_price,
            "entry_fee": round(o.entry_fee, 4),
            "estimated_exit_fee": round(estimated_exit_fee, 4),
            "total_fees": round(o.entry_fee + estimated_exit_fee, 4),
            "high_since_entry": o.high_price_since_entry,
            "low_since_entry": o.low_price_since_entry,
            "drop_from_peak": round(drop_from_peak, 2),
            # Display peak P&L: prefer cache (fresher than DB by up to one monitor-loop
            # interval), then enforce invariant peak >= current. Without the clamp the UI
            # can briefly show peak < current when WS price moves between monitor polls.
            # Apr 29 — display-only fix; DB state unchanged, monitor loop still owns
            # persistence. See CLAUDE.md "Peak/Trough P&L Invariant Bug + Option A Fix".
            "peak_pnl": round(max(
                cached_peak_pnl if cached_peak_pnl is not None else (o.peak_pnl or 0),
                pnl_pct,
            ), 2),
            "pnl": round(pnl, 2),
            "pnl_percentage": round(pnl_pct, 2),
            "duration": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "duration_seconds": duration_seconds,
            "opened_at": o.opened_at.isoformat(),
            "tp_level": o.current_tp_level or 1,
            "tp_target": o.dynamic_tp_target or 0,
            **_compute_be_level(o),
            "entry_order_type": getattr(o, 'entry_order_type', None) or "TAKER",
            "exit_order_type": getattr(o, 'exit_order_type', None) or "TAKER",
            "signal_lost_flagged": cached_flagged,
            "fl2_flagged": cached_fl2_flagged,
            "fl1_origin": cached_fl1_origin,
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
            "cell_multiplier": getattr(o, 'cell_multiplier', None),
            "cell_lev_multiplier": getattr(o, 'cell_lev_multiplier', None),
            "cell_multiplier_source": getattr(o, 'cell_multiplier_source', None),
            "pattern_cell_source": getattr(o, 'pattern_cell_source', None),
            "pattern_fixed_tp_pct": getattr(o, 'pattern_fixed_tp_pct', None),
            "pattern_fixed_sl_pct": getattr(o, 'pattern_fixed_sl_pct', None),
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
            "exit_slippage_pct": o.exit_slippage_pct,
            "post_exit_peak_minutes": o.post_exit_peak_minutes,
            "post_exit_trough_minutes": o.post_exit_trough_minutes,
            "post_exit_final_pnl": o.post_exit_final_pnl,
            "post_exit_signal_lost_minutes": o.post_exit_signal_lost_minutes,
            "entry_quality_score": o.entry_quality_score,
            "entry_btc_regime": o.entry_btc_regime,
            "exit_btc_regime": o.exit_btc_regime,
        })

    return orders_data


@app.get("/api/orders/export.csv")
async def export_orders_csv(db: AsyncSession = Depends(get_db)):
    """Export ALL orders (CLOSED + SIGNAL_EXPIRED) for current paper/live mode as CSV.

    No row limit. Includes every column on the Order model so downstream analysis
    (Winner Exit / BE Layer counterfactuals, cross-sample re-binning, Aborted
    distribution analysis) can run from a single download.
    """
    import csv
    import io

    await trading_engine.initialize(db)

    # May 7: invalidate any session-cached state so we read the latest committed
    # rows from disk. Without this, the export was occasionally missing trades
    # closed in the brief window between trading_engine.initialize() and this
    # query — the session's transaction snapshot pre-dated those commits.
    await db.commit()
    db.expunge_all()

    result = await db.execute(
        select(Order)
        .where(and_(
            Order.status.in_(["CLOSED", "SIGNAL_EXPIRED"]),
            Order.is_paper == trading_engine.is_paper_mode,
        ))
        .order_by(desc(Order.opened_at))
    )
    orders = result.scalars().all()

    # Use Order's table column order as the canonical schema — futureproof against
    # new columns being added (no need to update this endpoint).
    column_names = [c.name for c in Order.__table__.columns]

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(column_names)

    for o in orders:
        row = []
        for col in column_names:
            val = getattr(o, col, None)
            if isinstance(val, datetime):
                row.append(val.isoformat())
            elif val is None:
                row.append("")
            else:
                row.append(val)
        writer.writerow(row)

    buffer.seek(0)
    mode = "paper" if trading_engine.is_paper_mode else "live"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"scalpars_orders_{mode}_{timestamp}.csv"

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            # May 7: prevent browser/proxy from serving cached CSV when
            # operator clicks Export multiple times in a session.
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/transactions")
async def get_transactions(db: AsyncSession = Depends(get_db)):
    """Get all transactions"""
    await trading_engine.initialize(db)
    
    # Join with Order to get confidence + cell_multiplier (for UI coloring)
    result = await db.execute(
        select(Transaction, Order.confidence, Order.cell_multiplier, Order.cell_multiplier_source)
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
        "cell_multiplier": cell_multiplier,
        "cell_multiplier_source": cell_multiplier_source,
        "fee": round(t.fee, 4),
        "order_type": getattr(t, 'order_type', None) or "TAKER",
        "timestamp": t.timestamp.isoformat(),
        "confidence": confidence
    } for t, confidence, cell_multiplier, cell_multiplier_source in rows]


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


@app.post("/api/recover-positions")
async def recover_positions(db: AsyncSession = Depends(get_db)):
    """Import orphan Binance positions that have no matching Order in the database."""
    await trading_engine.initialize(db)
    if trading_engine.is_paper_mode:
        raise HTTPException(400, "Position recovery only works in live mode")

    positions = await binance_service.get_open_positions()
    if positions is None:
        raise HTTPException(503, "Could not fetch Binance positions — API error")
    if not positions:
        return {"recovered": 0, "positions": []}

    recovered = []
    for pos in positions:
        pair = pos['symbol'].replace('/USDT:USDT', 'USDT')
        existing = await db.execute(
            select(Order).where(
                and_(Order.pair == pair, Order.status == "OPEN", Order.is_paper == False)
            )
        )
        if existing.scalar_one_or_none():
            continue

        investment = pos['margin']
        leverage = pos['leverage']
        notional = pos['notional']
        quantity = pos['contracts']
        entry_price = pos['entry_price']
        direction = pos['side']

        order = Order(
            pair=pair,
            direction=direction,
            status="OPEN",
            entry_price=entry_price,
            current_price=pos['mark_price'],
            investment=abs(investment),
            leverage=leverage,
            notional_value=abs(notional),
            quantity=quantity,
            confidence="RECOVERED",
            entry_fee=0.0,
            entry_order_type="TAKER",
            peak_pnl=0.0,
            trough_pnl=0.0,
            high_price_since_entry=entry_price if direction == "LONG" else None,
            low_price_since_entry=entry_price if direction == "SHORT" else None,
            is_paper=False,
            current_tp_level=1,
            dynamic_tp_target=0.0
        )
        db.add(order)
        await db.flush()

        transaction = Transaction(
            order_id=order.id,
            pair=pair,
            action=f"OPEN_{direction}",
            price=entry_price,
            quantity=quantity,
            investment=abs(investment),
            leverage=leverage,
            notional_value=abs(notional),
            fee=0.0,
            order_type="TAKER",
            is_paper=False
        )
        db.add(transaction)
        recovered.append({"pair": pair, "direction": direction, "entry_price": entry_price, "quantity": quantity})

    await db.commit()
    return {"recovered": len(recovered), "positions": recovered}


async def _get_actual_fill_price(order) -> float:
    """Try to fetch the actual fill price from Binance trade history for an order.
    Falls back to current WebSocket price or entry price if trade history unavailable."""
    symbol = order.pair.replace('USDT', '/USDT:USDT')
    try:
        trades = await binance_service.fetch_my_trades(symbol, limit=10)
        if trades:
            close_side = 'sell' if order.direction == 'LONG' else 'buy'
            relevant = [t for t in trades if t['side'] == close_side]
            if relevant:
                latest = relevant[-1]
                logger.info(
                    f"[FILL_PRICE] {order.pair}: Found actual fill @ {latest['price']} "
                    f"(side={latest['side']}, time={latest['datetime']})"
                )
                return latest['price']
    except Exception as e:
        logger.warning(f"[FILL_PRICE] {order.pair}: Could not fetch trade history: {e}")

    fallback = order.current_price or order.entry_price
    logger.warning(f"[FILL_PRICE] {order.pair}: Using fallback price {fallback} (no trade history match)")
    return fallback


_reconcile_skip_counter = 0

# Reconciler race guard (Apr 16 — SUIUSDT incident).
# The bot sets Order.closing_in_progress=True + close_initiated_at=NOW() before
# sending a close to Binance.  Reconciler skips such rows if the intent is
# younger than CLOSE_INTENT_STALE_SECONDS, so the bot's own close path can
# commit the real exit reason (TRAILING_STOP, BREAKEVEN_EXIT, FL_*, ...) without
# being overwritten by EXTERNAL_CLOSE.  Older intents are treated as stale
# (close path likely crashed) so the row is still reconciled eventually.
#
# Sizing: worst-case bot close flow =
#   3 exit retries x 2s sleep                     = 6s
#   3 exit retries x 5s DB busy_timeout           = 15s
#   1s post-close verify sleep + API call         = ~1.5s
#   5 DB commit retries x up to 5s each           = ~25s
# Total realistic ceiling ~45-50s.  120s chosen as safety margin; beyond
# that we assume the close path crashed and let the reconciler recover.
CLOSE_INTENT_STALE_SECONDS = 120

async def _reconcile_open_orders(db: AsyncSession) -> list:
    """Shared reconciliation: close DB orders that are OPEN but gone from Binance.

    Returns list of dicts describing each closed order, or empty list.
    When the bulk get_open_positions() fails, falls back to per-symbol checks
    after 5 consecutive failures.
    """
    global _reconcile_skip_counter

    binance_positions = await binance_service.get_open_positions()

    if binance_positions is None:
        _reconcile_skip_counter += 1
        if _reconcile_skip_counter >= 5:
            logger.critical(
                f"[RECONCILE] get_open_positions() failed {_reconcile_skip_counter} consecutive times — "
                f"falling back to per-symbol position checks"
            )
            return await _reconcile_per_symbol(db)
        logger.warning(f"[RECONCILE] Skipped — Binance API error (consecutive skips: {_reconcile_skip_counter})")
        return []

    _reconcile_skip_counter = 0

    binance_pairs = set()
    for pos in binance_positions:
        pair = pos['symbol'].replace('/USDT:USDT', 'USDT')
        binance_pairs.add(pair)

    return await _close_orphan_orders(db, binance_pairs)


async def _reconcile_per_symbol(db: AsyncSession) -> list:
    """Fallback reconciliation: check each DB open order individually against Binance."""
    result = await db.execute(
        select(Order).where(and_(Order.status == "OPEN", Order.is_paper == False))
    )
    open_orders = result.scalars().all()
    if not open_orders:
        return []

    still_open_pairs = set()
    for order in open_orders:
        symbol = order.pair.replace('USDT', '/USDT:USDT')
        try:
            pos = await binance_service.get_position_for_symbol(symbol)
            if pos is not None:
                still_open_pairs.add(order.pair)
            else:
                logger.info(f"[RECONCILE_FALLBACK] {order.pair}: no position found on Binance")
        except Exception as e:
            logger.error(f"[RECONCILE_FALLBACK] {order.pair}: per-symbol check failed: {e}, keeping as OPEN")
            still_open_pairs.add(order.pair)

    return await _close_orphan_orders(db, still_open_pairs)


async def _close_orphan_orders(db: AsyncSession, binance_pairs: set) -> list:
    """Close DB orders whose pair is not in binance_pairs (not open on Binance).

    Each order is wrapped in its own try/except so a failure on one
    order never prevents the others from being reconciled.
    """
    result = await db.execute(
        select(Order).where(and_(Order.status == "OPEN", Order.is_paper == False))
    )
    open_orders = result.scalars().all()

    closed = []
    for order in open_orders:
        if order.pair not in binance_pairs:
            # Reconciler race guard: the bot publishes closing_in_progress=True
            # + close_initiated_at=NOW() before calling Binance.  If the flag is
            # fresh, skip — the bot's own close path is in flight and will
            # write the real exit reason.  Stale flags (>CLOSE_INTENT_STALE_SECONDS)
            # are ignored so a crashed close path can still be reconciled.
            if order.closing_in_progress and order.close_initiated_at is not None:
                _age = (datetime.utcnow() - order.close_initiated_at).total_seconds()
                if _age < CLOSE_INTENT_STALE_SECONDS:
                    logger.info(
                        f"[RECONCILE_SKIP] {order.pair} {order.direction}: bot close in progress "
                        f"(age={_age:.1f}s < {CLOSE_INTENT_STALE_SECONDS}s) — skipping EXTERNAL_CLOSE"
                    )
                    continue
                else:
                    logger.warning(
                        f"[RECONCILE_STALE_INTENT] {order.pair} {order.direction}: "
                        f"close-intent stale (age={_age:.1f}s > {CLOSE_INTENT_STALE_SECONDS}s) — "
                        f"proceeding as EXTERNAL_CLOSE (close path likely crashed)"
                    )
            try:
                exit_price = await _get_actual_fill_price(order)

                # Reconciler simply marks orphan orders as EXTERNAL_CLOSE.
                # BROKER_SL / BROKER_TP labels were removed Apr 17 with the
                # protective-stops feature (the feature was never functional
                # for this account — see CLAUDE.md).
                order.status = "CLOSED"
                order.close_reason = "EXTERNAL_CLOSE"
                order.closed_at = datetime.utcnow()
                order.exit_price = exit_price
                if order.direction == "LONG":
                    raw = (order.exit_price - order.entry_price) * order.quantity
                else:
                    raw = (order.entry_price - order.exit_price) * order.quantity
                fee = (order.entry_fee or 0) + order.exit_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                order.pnl = round(raw - fee, 4)
                notional = order.entry_price * order.quantity if order.quantity else 1
                order.pnl_percentage = round(((raw - fee) / notional) * 100, 4)
                order.exit_fee = round(order.exit_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee), 4)
                order.total_fee = round((order.entry_fee or 0) + order.exit_fee, 4)

                tx = Transaction(
                    order_id=order.id, pair=order.pair,
                    action=f"CLOSE_{order.direction}", price=order.exit_price,
                    quantity=order.quantity, investment=order.investment,
                    leverage=order.leverage, notional_value=order.notional_value,
                    fee=order.exit_fee, order_type="EXTERNAL", is_paper=False
                )
                db.add(tx)
                closed.append({"pair": order.pair, "direction": order.direction, "pnl": order.pnl})
                logger.warning(f"[RECONCILE] {order.pair} {order.direction}: closed as EXTERNAL_CLOSE @ {order.exit_price} (not found on Binance)")
            except Exception as e:
                logger.error(f"[RECONCILE] {order.pair} {order.direction}: failed to close orphan order {order.id}: {e}")

    if closed:
        await db.commit()
        async with _cache_lock:
            for info in closed:
                pair = info["pair"]
                _open_orders_cache[pair] = [
                    o for o in _open_orders_cache.get(pair, [])
                    if not (o.get('direction') == info["direction"])
                ]
    return closed


@app.post("/api/reconcile-positions")
async def reconcile_positions(db: AsyncSession = Depends(get_db)):
    """Close DB orders that are OPEN but no longer exist on Binance (orphan detection)."""
    await trading_engine.initialize(db)
    if trading_engine.is_paper_mode:
        raise HTTPException(400, "Reconciliation only works in live mode")

    closed = await _reconcile_open_orders(db)
    return {"closed": len(closed), "orders": closed}


# ----- Performance Metrics -----

@app.get("/api/performance")
async def get_performance(regime: str = None, window_hours: int = None,
                          from_date: str = None, to_date: str = None,
                          db: AsyncSession = Depends(get_db)):
    """Get closed orders performance metrics, optionally filtered by macro trend regime and/or time window.

    window_hours: int — restrict to trades closed within the last N hours (e.g., 24=last day, 168=last week).
                  None / 0 / negative = no time filter (default).
    from_date / to_date: str (ISO YYYY-MM-DD) — restrict to trades closed within [from_date 00:00 UTC,
                  to_date 23:59:59 UTC]. Both required for the date range to apply; takes precedence
                  over window_hours when both provided.
    """
    await trading_engine.initialize(db)

    try:
        return await _compute_performance(db, regime=regime, window_hours=window_hours,
                                          from_date=from_date, to_date=to_date)
    except Exception as e:
        logger.error(f"[PERF] Unhandled error in get_performance: {e}\n{traceback.format_exc()}")
        return {
            "total_trades": 0, "total_longs": 0, "total_shorts": 0,
            "total_wins": 0, "total_losses": 0,
            "win_rate": 0, "win_rate_longs": 0, "win_rate_shorts": 0,
            "avg_win": 0, "avg_win_long": 0, "avg_win_short": 0,
            "avg_loss": 0, "avg_loss_long": 0, "avg_loss_short": 0,
            "expectancy": 0, "expectancy_pct": 0,
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
            "gap_performance": [], "ema58_gap_performance": [],
            "ema813_gap_performance": [], "ema_fan_accel_performance": [], "rsi_performance": [], "range_position_performance": [], "adx_delta_performance": [], "adx_performance": [], "adx_direction_performance": [], "rsi_direction_performance": [], "stretch_performance": [],
            "pair_slope_performance": [], "btc_slope_performance": [], "pair_ema20_ema50_gap_performance": [], "btc_ema20_ema50_gap_performance": [], "btc_adx_performance": [], "btc_adx_direction_performance": [], "btc_rsi_direction_performance": [], "btc_rsi_direction_30m_performance": [], "btc_volatility_performance": [], "btc_rsi_1h_direction_performance": [], "btc_vol_adx_crosstab": [], "btc_rsi_1h_5m_crosstab": [], "adx_dir_crosstab": [], "rsi_dir_crosstab": [], "btc_rsi_30m_5m_crosstab": [], "range_pos_btc_rsi_dir_crosstab": [], "range_pos_pair_rsi_dir_crosstab": [], "pair_slope_adx_crosstab": [], "btc_slope_adx_crosstab": [], "adx_delta_btc_adx_crosstab": [], "btc_gap_btc_adx_crosstab": [], "pair_gap_pair_adx_crosstab": [],
            "btc_rsi_performance": [], "btc_rsi_adx_crosstab": [], "quality_score_performance": [],
            "regime_performance": [], "regime_transition_performance": [],
            "by_close_reason": {},
            "stop_loss_deep_dive": {"total_sl_trades": 0, "be_was_active": {"count": 0}, "positive_no_be": {"count": 0}, "never_positive": {"count": 0}, "avg_peak_all_sl": 0},
            "winning_trades_drawdown": [], "trough_recovery": [],
            "never_positive_deep_dive": [],
            "performance_over_time": [],
            "post_exit_regret_deep_dive": [],
            "hold_time_expectancy": [],
            "entry_conditions_by_reason": [],
            "entry_conditions_by_outcome": [],
            "multiplier_cell_performance": {"longs": [], "shorts": [], "summary": {}},
            "pattern_cell_performance": {"rules": [], "summary": {}},
            "extension_multiplier_performance": {"rules": [], "summary": {}},
            "btc_1h_slope_btc_adx_multiplier_performance": {"rules": [], "summary": {}},
            "pattern_4cohort_coverage": {"cohorts": [], "total": {}},
            "pattern_c_combo_tracker": {"rows": [], "tracker": "C"},
            "pattern_w_combo_tracker": {"rows": [], "tracker": "W"},
            "flagged_exits": [],
            "period_performance": [],
            "equity_curve": [],
            "pnl_distribution": [],
            "hourly_performance": [],
            "daily_performance": [],
            "day_time_heatmap": [],
            "_error": str(e)
        }


def _compute_entry_type_stats(orders, signal_expired_orders=None):
    """Compute performance breakdown by entry order type (MAKER / TAKER / TAKER_FALLBACK).

    Amendment #7 (Apr 18): if signal_expired_orders is provided, include a
    SIGNAL_EXPIRED row showing aborted-entry count by confidence. These trades
    have pnl=0 by definition (never entered), so WR/PnL fields stay at 0.
    """
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

    # Include SIGNAL_EXPIRED (aborted entries) as a synthetic row
    if signal_expired_orders:
        groups["SIGNAL_EXPIRED"] = {"trades": [], "pnl_sum": 0, "fee_sum": 0, "wins": 0, "by_confidence": {}}
        for o in signal_expired_orders:
            groups["SIGNAL_EXPIRED"]["trades"].append(o)
            conf = getattr(o, 'confidence', 'UNKNOWN') or 'UNKNOWN'
            groups["SIGNAL_EXPIRED"]["by_confidence"][conf] = groups["SIGNAL_EXPIRED"]["by_confidence"].get(conf, 0) + 1

    # Total used as denominator for pct_of_total. SIGNAL_EXPIRED rows aren't in
    # `orders` (synthetic row), so include their count in the denom — otherwise
    # SIGNAL_EXPIRED count / closed-only total can exceed 100% in regime slices
    # where aborts > closes (the historical 315.4% bug from May 1). Including
    # them keeps all pct values internally consistent and bounded ≤100%.
    total_trades = len(orders) + (len(signal_expired_orders) if signal_expired_orders else 0)
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


def _compute_signal_expired_breakdown(signal_expired_orders):
    """Aggregate SIGNAL_EXPIRED rows by reason category.

    Amendment #7 (Apr 18) persists each aborted entry as an Order with
    close_reason='SIGNAL_EXPIRED:<reason>'. This function buckets those
    by reason category (signal_flipped / confidence_lost / btc_adx_direction
    / btc_adx_out_of_range / btc_rsi_out_of_range / other) and includes
    direction + confidence breakdowns per category.

    Returns list of dicts sorted by count descending, with each row:
      { category, count, pct_of_total, longs, shorts, by_confidence,
        sample_reasons (top 3 specific reason strings in this category) }

    Used at the 200-trade analysis to identify the dominant cause of
    signal expirations and decide which filter or timeout to adjust.
    """
    if not signal_expired_orders:
        return []

    def _categorize(raw_reason: str) -> str:
        # raw_reason example: "signal_flipped_LONG_to_NO_TRADE", "btc_adx_out_of_range_27.3"
        if not raw_reason:
            return "Other"
        if raw_reason.startswith("signal_flipped_"):
            return "Signal Flipped"
        if raw_reason == "confidence_lost":
            return "Confidence Lost"
        if raw_reason.startswith("btc_adx_direction_"):
            return "BTC ADX Direction Flipped"
        if raw_reason.startswith("btc_adx_out_of_range"):
            return "BTC ADX Out of Range"
        if raw_reason.startswith("btc_rsi_out_of_range"):
            return "BTC RSI Out of Range"
        return "Other"

    total = len(signal_expired_orders)
    buckets = {}

    for o in signal_expired_orders:
        # close_reason = "SIGNAL_EXPIRED:<actual_reason>"
        cr = o.close_reason or ""
        raw_reason = cr.split(":", 1)[1] if ":" in cr else cr
        category = _categorize(raw_reason)

        if category not in buckets:
            buckets[category] = {
                "count": 0,
                "longs": 0,
                "shorts": 0,
                "by_confidence": {},
                "reason_counts": {},
                "wait_seconds": [],  # May 2: per-row wait time before re-validation killed entry
            }
        b = buckets[category]
        b["count"] += 1
        if (o.direction or "LONG") == "LONG":
            b["longs"] += 1
        else:
            b["shorts"] += 1
        conf = o.confidence or "UNKNOWN"
        b["by_confidence"][conf] = b["by_confidence"].get(conf, 0) + 1
        b["reason_counts"][raw_reason] = b["reason_counts"].get(raw_reason, 0) + 1
        # May 2: capture wait_seconds = closed_at - opened_at. Pre-enrichment rows
        # have opened_at == closed_at (wait=0); skip them so historical zeros don't
        # pollute the median.
        if o.opened_at and o.closed_at:
            wait_s = (o.closed_at - o.opened_at).total_seconds()
            if wait_s > 0:
                b["wait_seconds"].append(wait_s)

    rows = []
    for category, b in buckets.items():
        # Numeric-suffixed reasons (btc_rsi_out_of_range_X.X, btc_adx_out_of_range_X.X)
        # have a unique value embedded per trade — top-3 only captures the rare
        # repeats. Detect those, parse the numeric, show count + range + avg
        # for the whole prefix instead of cherry-picking 3 values out of 18.
        # Discrete reasons (signal_flipped_*, confidence_lost, btc_adx_direction_*)
        # repeat exactly, so top-3 captures their distribution well.
        numeric_groups = {}  # prefix -> list of values
        discrete_counts = {}
        for raw_reason, n in b["reason_counts"].items():
            # Try to detect "<prefix>_<float>" pattern
            parts = raw_reason.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    val = float(parts[1])
                    prefix = parts[0]
                    numeric_groups.setdefault(prefix, []).extend([val] * n)
                    continue
                except ValueError:
                    pass
            discrete_counts[raw_reason] = discrete_counts.get(raw_reason, 0) + n

        sample_reasons = []
        # Numeric groups: show prefix with N + range + avg
        for prefix, vals in numeric_groups.items():
            n = len(vals)
            vmin = min(vals)
            vmax = max(vals)
            vavg = sum(vals) / n
            sample_reasons.append(f"{prefix} (N={n}, range {vmin:.1f}-{vmax:.1f}, avg {vavg:.1f})")
        # Discrete reasons: top 3 by count
        top_discrete = sorted(discrete_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        for r, n in top_discrete:
            sample_reasons.append(f"{r} ({n})")

        # May 2: wait-time distribution (median / p90 / max) — None when no
        # post-enrichment rows in this bucket yet.
        waits = sorted(b.get("wait_seconds", []))
        if waits:
            n = len(waits)
            median_wait = waits[n // 2] if n % 2 == 1 else (waits[n // 2 - 1] + waits[n // 2]) / 2
            p90_idx = max(0, min(n - 1, int(round(0.9 * (n - 1)))))
            p90_wait = waits[p90_idx]
            max_wait = waits[-1]
            wait_n = n
        else:
            median_wait = p90_wait = max_wait = None
            wait_n = 0

        rows.append({
            "category": category,
            "count": b["count"],
            "pct_of_total": round(b["count"] / total * 100, 1) if total else 0,
            "longs": b["longs"],
            "shorts": b["shorts"],
            "by_confidence": b["by_confidence"],
            "sample_reasons": sample_reasons,
            "wait_n": wait_n,
            "median_wait": round(median_wait, 1) if median_wait is not None else None,
            "p90_wait": round(p90_wait, 1) if p90_wait is not None else None,
            "max_wait": round(max_wait, 1) if max_wait is not None else None,
        })

    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows


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


def _compute_period_performance(orders):
    """Compute performance summary for rolling time periods."""
    from datetime import timezone
    now = datetime.utcnow().replace(tzinfo=timezone.utc)

    periods = [
        ("Last 6h", timedelta(hours=6)),
        ("Last 12h", timedelta(hours=12)),
        ("1 Day", timedelta(days=1)),
        ("2 Days", timedelta(days=2)),
        ("3 Days", timedelta(days=3)),
        ("5 Days", timedelta(days=5)),
        ("7 Days", timedelta(days=7)),
        ("15 Days", timedelta(days=15)),
        ("30 Days", timedelta(days=30)),
    ]

    timed = []
    for o in orders:
        if o.closed_at:
            ca = o.closed_at.replace(tzinfo=timezone.utc) if o.closed_at.tzinfo is None else o.closed_at
            timed.append((o, ca))

    result = []
    for label, delta in periods:
        cutoff = now - delta
        bucket = [o for o, ca in timed if ca >= cutoff]
        result.append(_period_stats(label, bucket))

    result.append(_period_stats("Total", [o for o, _ in timed]))
    return result


def _period_stats(label, trades):
    count = len(trades)
    if count == 0:
        return {"period": label, "count": 0, "longs": 0, "shorts": 0,
                "win_rate": 0, "avg_pnl_pct": 0, "total_pnl": 0, "profit_factor": 0,
                "total_fees": 0, "total_investment": 0, "total_notional": 0,
                "pnl_over_inv": 0, "pnl_over_not": 0}
    longs = sum(1 for o in trades if o.direction == "LONG")
    shorts = count - longs
    wins = [o for o in trades if (o.pnl or 0) > 0]
    win_rate = round(len(wins) / count * 100, 1)
    avg_pnl_pct = round(sum(o.pnl_percentage or 0 for o in trades) / count, 4)
    total_pnl = round(sum(o.pnl or 0 for o in trades), 2)
    total_fees = round(sum(o.total_fee or 0 for o in trades), 2)
    total_investment = round(sum(o.investment or 0 for o in trades), 2)
    total_notional = round(sum(o.notional_value or 0 for o in trades), 2)
    pnl_over_inv = round(total_pnl / total_investment * 100, 2) if total_investment > 0 else 0
    pnl_over_not = round(total_pnl / total_notional * 100, 4) if total_notional > 0 else 0
    total_wins_usd = sum(o.pnl for o in trades if (o.pnl or 0) > 0)
    total_losses_usd = abs(sum(o.pnl for o in trades if (o.pnl or 0) < 0))
    profit_factor = round(total_wins_usd / total_losses_usd, 2) if total_losses_usd > 0 else (999 if total_wins_usd > 0 else 0)
    # Slippage stats (only for trades that have slippage data)
    slippage_trades = [o for o in trades if o.exit_slippage_pct is not None]
    avg_slippage_pct = round(sum(o.exit_slippage_pct for o in slippage_trades) / len(slippage_trades), 4) if slippage_trades else None
    return {
        "period": label, "count": count, "longs": longs, "shorts": shorts,
        "win_rate": win_rate, "avg_pnl_pct": avg_pnl_pct, "total_pnl": total_pnl,
        "profit_factor": profit_factor, "total_fees": total_fees,
        "total_investment": total_investment, "total_notional": total_notional,
        "pnl_over_inv": pnl_over_inv, "pnl_over_not": pnl_over_not,
        "avg_slippage_pct": avg_slippage_pct, "slippage_count": len(slippage_trades),
    }


def _compute_equity_curve(orders):
    """Compute cumulative P&L curve trade by trade."""
    from datetime import timezone
    UTC_MINUS_3 = timezone(timedelta(hours=-3))
    timed = [(o, o.closed_at.replace(tzinfo=timezone.utc) if o.closed_at.tzinfo is None else o.closed_at)
             for o in orders if o.closed_at is not None]
    if not timed:
        return []
    timed.sort(key=lambda x: x[1])
    result = []
    cum_all = 0
    cum_long = 0
    cum_short = 0
    for i, (o, ct) in enumerate(timed):
        pnl = o.pnl or 0
        cum_all += pnl
        if o.direction == "LONG":
            cum_long += pnl
        else:
            cum_short += pnl
        local_time = ct.astimezone(UTC_MINUS_3)
        result.append({
            "trade_num": i + 1,
            "pnl": round(pnl, 2),
            "cumulative": round(cum_all, 2),
            "cum_long": round(cum_long, 2),
            "cum_short": round(cum_short, 2),
            "direction": o.direction,
            "timestamp": local_time.strftime("%m/%d %H:%M"),
        })
    return result


def _compute_pnl_distribution(orders):
    """Compute histogram of trade P&L percentages."""
    closed = [o for o in orders if o.pnl_percentage is not None]
    if not closed:
        return []
    edges = [-1.5, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5]
    buckets = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f"{lo:+.2f}%"
        in_bucket = [o for o in closed if lo <= (o.pnl_percentage or 0) < hi]
        buckets.append({
            "label": label,
            "lo": lo, "hi": hi,
            "count": len(in_bucket),
            "count_long": sum(1 for o in in_bucket if o.direction == "LONG"),
            "count_short": sum(1 for o in in_bucket if o.direction == "SHORT"),
        })
    below = [o for o in closed if (o.pnl_percentage or 0) < edges[0]]
    above = [o for o in closed if (o.pnl_percentage or 0) >= edges[-1]]
    if below:
        buckets.insert(0, {
            "label": f"<{edges[0]:+.1f}%", "lo": -999, "hi": edges[0],
            "count": len(below),
            "count_long": sum(1 for o in below if o.direction == "LONG"),
            "count_short": sum(1 for o in below if o.direction == "SHORT"),
        })
    if above:
        buckets.append({
            "label": f"≥{edges[-1]:+.1f}%", "lo": edges[-1], "hi": 999,
            "count": len(above),
            "count_long": sum(1 for o in above if o.direction == "LONG"),
            "count_short": sum(1 for o in above if o.direction == "SHORT"),
        })
    return buckets


def _compute_hourly_performance(orders):
    """Compute win rate and avg P&L by hour of day (UTC-3)."""
    from datetime import timezone
    UTC_MINUS_3 = timezone(timedelta(hours=-3))
    hourly = {h: [] for h in range(24)}
    for o in orders:
        if o.closed_at is None:
            continue
        ct = o.closed_at.replace(tzinfo=timezone.utc) if o.closed_at.tzinfo is None else o.closed_at
        hour = ct.astimezone(UTC_MINUS_3).hour
        hourly[hour].append(o)
    result = []
    for h in range(24):
        trades = hourly[h]
        count = len(trades)
        if count == 0:
            result.append({"hour": h, "label": f"{h:02d}:00", "trades": 0, "wins": 0,
                           "win_rate": 0, "avg_pnl_pct": 0, "longs": 0, "shorts": 0})
            continue
        wins = sum(1 for o in trades if (o.pnl or 0) > 0)
        longs = sum(1 for o in trades if o.direction == "LONG")
        avg_pnl = round(sum(o.pnl_percentage or 0 for o in trades) / count, 4)
        result.append({
            "hour": h, "label": f"{h:02d}:00", "trades": count, "wins": wins,
            "win_rate": round(wins / count * 100, 1),
            "avg_pnl_pct": avg_pnl,
            "longs": longs, "shorts": count - longs,
        })
    return result


_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def _compute_daily_performance(orders):
    """Compute win rate and avg P&L by day of week (UTC-3)."""
    from datetime import timezone
    UTC_MINUS_3 = timezone(timedelta(hours=-3))
    daily = {d: [] for d in range(7)}
    for o in orders:
        if o.closed_at is None:
            continue
        ct = o.closed_at.replace(tzinfo=timezone.utc) if o.closed_at.tzinfo is None else o.closed_at
        weekday = ct.astimezone(UTC_MINUS_3).weekday()
        daily[weekday].append(o)
    result = []
    for d in range(7):
        trades = daily[d]
        count = len(trades)
        if count == 0:
            result.append({"day": d, "label": _DAY_LABELS[d], "trades": 0, "wins": 0,
                           "win_rate": 0, "avg_pnl_pct": 0, "longs": 0, "shorts": 0})
            continue
        wins = sum(1 for o in trades if (o.pnl or 0) > 0)
        longs = sum(1 for o in trades if o.direction == "LONG")
        avg_pnl = round(sum(o.pnl_percentage or 0 for o in trades) / count, 4)
        result.append({
            "day": d, "label": _DAY_LABELS[d], "trades": count, "wins": wins,
            "win_rate": round(wins / count * 100, 1),
            "avg_pnl_pct": avg_pnl,
            "longs": longs, "shorts": count - longs,
        })
    return result


_TIME_BLOCK_LABELS = ["00-04", "04-08", "08-12", "12-16", "16-20", "20-24"]

_VOL_BINS = [
    # May 12: refined 5 → 8 bins. Split "< 0.95" into three (very-low/low/edge)
    # and "> 1.25" into two (high/extreme) — both zones are where bleed lives.
    # May 19: further split "< 0.70" into "< 0.50" and "0.50-0.70" to surface
    # extreme-low-volume entries in finer detail (PROMUSDT NP today landed in <0.70).
    # Empty cells in the 2D cross-tab auto-drop.
    ("< 0.50", 0.0, 0.50),
    ("0.50-0.70", 0.50, 0.70),
    ("0.70-0.85", 0.70, 0.85),
    ("0.85-0.95", 0.85, 0.95),
    ("0.95-1.05", 0.95, 1.05),
    ("1.05-1.10", 1.05, 1.10),
    ("1.10-1.25", 1.10, 1.25),
    ("1.25-1.50", 1.25, 1.50),
    ("> 1.50", 1.50, 999.0),
]

def _compute_ttp_ratio(o):
    """Time-to-Peak Ratio (Apr 28, Trade Quality Metric).

    Where in the hold did the trade reach its peak P&L?
    - Ratio 0.0 = peaked at entry (instantaneous)
    - Ratio 0.5 = peaked at midpoint of hold
    - Ratio 1.0 = peaked at close (no give-back, but maybe just lucky on exit timing)

    Returns None for trades that:
    - Never went positive (peak_pnl <= 0)
    - Don't have peak_reached_at recorded (older trades pre-tracking)
    - Are still open (closed_at is None)
    - Closed instantaneously (zero duration)
    """
    if o.closed_at is None or o.opened_at is None:
        return None
    if o.peak_pnl is None or o.peak_pnl <= 0:
        return None
    if o.peak_reached_at is None:
        return None
    total_secs = (o.closed_at - o.opened_at).total_seconds()
    if total_secs <= 0:
        return None
    peak_secs = (o.peak_reached_at - o.opened_at).total_seconds()
    if peak_secs < 0:
        return None
    ratio = peak_secs / total_secs
    return max(0.0, min(1.0, ratio))


def _vol_bin_label(ratio):
    if ratio is None:
        return None
    for label, lo, hi in _VOL_BINS:
        if lo <= ratio < hi:
            return label
    return _VOL_BINS[-1][0]


def _btc_adx_bucket(o):
    v = getattr(o, 'entry_btc_adx', None)
    if v is None:
        return None
    if v < 18: return '<18'
    if v < 25: return '18-25'
    if v < 30: return '25-30'
    if v < 35: return '30-35'
    return '35+'


def _ema50_alignment_bucket(o):
    v = getattr(o, 'entry_ema50_slope', None)
    if v is None:
        return None
    if abs(v) < 0.04:
        return 'Flat'
    if (v > 0 and (o.direction or 'LONG') == 'LONG') or (v < 0 and (o.direction or 'LONG') == 'SHORT'):
        return 'Aligned'
    return 'Opposite'


def _pair_adx_bucket(o):
    v = getattr(o, 'entry_adx', None)
    if v is None:
        return None
    if v < 18: return '<18'
    if v < 22: return '18-22'
    if v < 25: return '22-25'
    if v < 28: return '25-28'
    if v < 30: return '28-30'
    if v < 33: return '30-33'
    return '33+'


def _di_spread_bucket(o):
    p = getattr(o, 'entry_pos_di', None)
    n = getattr(o, 'entry_neg_di', None)
    if p is None or n is None:
        return None
    spread = abs(p - n)
    if spread < 2: return '< 2'
    if spread < 5: return '2 - 5'
    if spread < 10: return '5 - 10'
    return '> 10'


def _btc_rsi_bucket(o):
    v = getattr(o, 'entry_btc_rsi', None)
    if v is None:
        return None
    if v < 20: return '<20'
    if v < 25: return '20-25'
    if v < 30: return '25-30'
    if v < 35: return '30-35'
    if v < 40: return '35-40'
    if v < 45: return '40-45'
    if v < 50: return '45-50'
    if v < 55: return '50-55'
    if v < 60: return '55-60'
    if v < 65: return '60-65'
    if v < 70: return '65-70'
    return '70+'


def _funding_rate_bucket(o):
    v = getattr(o, 'entry_funding_rate', None)
    if v is None:
        return None
    v_pct = v * 100
    if v_pct < -0.05: return '<-0.05%'
    if v_pct < -0.02: return '-0.05 to -0.02%'
    if v_pct < 0.02: return '-0.02 to +0.02%'
    if v_pct < 0.05: return '+0.02 to +0.05%'
    return '>+0.05%'


def _compute_atr_bucket_performance(orders):
    """Performance by Pair ATR(14)% (entry volatility) — May 9.

    Tests the hypothesis that high-volatility pairs systematically lose money.
    Buckets trades by entry_atr_pct and reports per-bucket WR / Avg P&L%.
    Replaces the manual blacklist approach if a clean breakpoint emerges.
    """
    closed = [o for o in orders if o.status == "CLOSED" and o.entry_atr_pct is not None and o.pnl is not None]
    if not closed:
        return []
    # May 9: finer 0.10% buckets in the discriminating zone (0.25-0.85), with
    # wider 0.85-1.00 and ≥1.00 catchalls since data is thin at high ATR.
    # May 15 PM: split ≥1.00% into 1.00-1.25 / 1.25-1.50 / 1.50-2.00 / >2.00
    # to expose the LONG-ATR-cap zone (cross-sample 1.25-1.50 LONG = killer).
    buckets = [
        ("<0.25%", 0.0, 0.25),
        ("0.25-0.35%", 0.25, 0.35),
        ("0.35-0.45%", 0.35, 0.45),
        ("0.45-0.55%", 0.45, 0.55),
        ("0.55-0.65%", 0.55, 0.65),
        ("0.65-0.75%", 0.65, 0.75),
        ("0.75-0.85%", 0.75, 0.85),
        ("0.85-1.00%", 0.85, 1.00),
        ("1.00-1.25%", 1.00, 1.25),
        ("1.25-1.50%", 1.25, 1.50),
        ("1.50-2.00%", 1.50, 2.00),
        (">2.00%", 2.00, 999),
    ]
    rows = []
    for label, lo, hi in buckets:
        bucket = [o for o in closed if lo <= o.entry_atr_pct < hi]
        if not bucket:
            continue
        n = len(bucket)
        longs = sum(1 for o in bucket if o.direction == "LONG")
        shorts = n - longs
        wins = sum(1 for o in bucket if o.pnl > 0)
        total_pnl = sum(o.pnl for o in bucket)
        avg_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket) / n
        avg_atr = sum(o.entry_atr_pct for o in bucket) / n
        avg_rsi = sum(o.entry_rsi or 0 for o in bucket) / n
        avg_adx = sum(o.entry_adx or 0 for o in bucket) / n
        rows.append({
            "range": label,
            "trades": n,
            "longs": longs,
            "shorts": shorts,
            "win_rate": round(wins / n * 100, 1),
            "avg_pnl": round(total_pnl / n, 2),
            "avg_pnl_pct": round(avg_pnl_pct, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_atr_pct": round(avg_atr, 3),
            "avg_rsi": round(avg_rsi, 1),
            "avg_adx": round(avg_adx, 1),
        })
    return rows


def _compute_volume_intersection_crosstab(orders):
    """May 10 evening: 2D cross-tab of Global Vol Ratio (rows) × Pair Vol USD (cols).

    Validates the intersection-style rescue clause (Global<0.95 AND Pair Vol $
    <$100M → block; otherwise allow). Per-direction so future asymmetric tuning
    can be data-driven.

    Bucket boundaries match existing 1D tables for cross-reference consistency:
    - Global Vol Ratio rows: matches existing Volume Cross-Tab (Global x Pair) Global axis
    - Pair Vol USD cols: matches Performance by Pair 24h Volume table

    Empty cells (N=0) are dropped from output to keep the table readable.

    Pre-deploy trades have NULL entry_pair_volume_24h_usd and are excluded.
    """
    M = 1_000_000.0
    B = 1_000_000_000.0

    # Bucket boundaries — match existing individual tables.
    # May 18: Global Vol rows use the same _VOL_BINS as Volume Cross-Tab (8-bucket
    # granular scheme) so the two tables align side-by-side. Empty cells still drop.
    global_buckets = list(_VOL_BINS)
    pair_buckets = [
        ("<$30M", 0, 30 * M),
        ("$30-50M", 30 * M, 50 * M),
        ("$50-80M", 50 * M, 80 * M),
        ("$80-100M", 80 * M, 100 * M),
        ("$100-150M", 100 * M, 150 * M),
        ("$150-250M", 150 * M, 250 * M),
        ("$250-500M", 250 * M, 500 * M),
        ("$500M-1B", 500 * M, 1 * B),
        (">$1B", 1 * B, 1e15),
    ]

    def _bucket(value, buckets):
        for name, lo, hi in buckets:
            if lo <= value < hi:
                return name
        return None

    def _direction_rows(direction):
        rows = []
        for r in orders:
            if r.status != "CLOSED" or r.pnl is None:
                continue
            d = (r.direction.value if hasattr(r.direction, 'value') else r.direction) or ""
            if d != direction:
                continue
            gv = r.entry_global_volume_ratio
            pvu = getattr(r, 'entry_pair_volume_24h_usd', None)
            if gv is None or pvu is None:
                continue
            gb = _bucket(gv, global_buckets)
            pb = _bucket(pvu, pair_buckets)
            if gb is None or pb is None:
                continue
            rows.append({'gb': gb, 'pb': pb, 'pnl': r.pnl, 'pnl_pct': r.pnl_percentage or 0.0})

        from collections import defaultdict
        cells = defaultdict(list)
        for rec in rows:
            cells[(rec['gb'], rec['pb'])].append(rec)

        out = []
        for gname, _, _ in global_buckets:
            for pname, _, _ in pair_buckets:
                key = (gname, pname)
                cell = cells.get(key, [])
                if not cell:
                    continue
                n = len(cell)
                wins = sum(1 for c in cell if c['pnl'] > 0)
                total_pnl = sum(c['pnl'] for c in cell)
                avg_pct = sum(c['pnl_pct'] for c in cell) / n
                out.append({
                    'global_bucket': gname,
                    'pair_bucket': pname,
                    'n': n,
                    'win_rate': round(100 * wins / n, 1),
                    'avg_pct': round(avg_pct, 3),
                    'total_pnl': round(total_pnl, 2),
                })
        return out

    return {
        'long': _direction_rows('LONG'),
        'short': _direction_rows('SHORT'),
        'global_buckets': [b[0] for b in global_buckets],
        'pair_buckets': [b[0] for b in pair_buckets],
    }


def _compute_pair_volume_bucket_performance(orders):
    """May 10: Bucket trades by absolute pair 24h USD volume at entry.

    Goal: find the structural threshold below which pairs systematically
    underperform — instead of blacklisting pairs one-by-one. Once 2-sample
    evidence shows a clean breakpoint, can promote to a min_pair_volume
    filter.

    Buckets (in USD): <30M, 30-50M, 50-80M, 80-100M, 100-150M, 150-250M,
    250-500M, 500M-1B, >1B. Granular at low end (kill-zone hypothesis),
    wider at high end (liquid mid-large caps).

    Pre-deploy trades have NULL entry_pair_volume_24h_usd and are excluded.
    """
    M = 1_000_000.0
    B = 1_000_000_000.0
    buckets = [
        ("<$30M",      0,        30 * M),
        ("$30-50M",    30 * M,   50 * M),
        ("$50-80M",    50 * M,   80 * M),
        ("$80-100M",   80 * M,   100 * M),
        ("$100-150M",  100 * M,  150 * M),
        ("$150-250M",  150 * M,  250 * M),
        ("$250-500M",  250 * M,  500 * M),
        ("$500M-1B",   500 * M,  1 * B),
        (">$1B",       1 * B,    1e15),
    ]
    closed = [o for o in orders if o.status == "CLOSED" and o.pnl is not None]
    rows = []
    for name, lo, hi in buckets:
        b = [o for o in closed
             if getattr(o, 'entry_pair_volume_24h_usd', None) is not None
             and lo <= o.entry_pair_volume_24h_usd < hi]
        if not b:
            continue
        n = len(b)
        wins = sum(1 for o in b if o.pnl > 0)
        total_pnl = sum(o.pnl for o in b)
        avg_pct = sum(o.pnl_percentage or 0 for o in b) / n
        avg_vol_usd = sum(o.entry_pair_volume_24h_usd for o in b) / n
        # direction split
        n_long = sum(1 for o in b if (o.direction.value if hasattr(o.direction, 'value') else o.direction) == "LONG")
        n_short = n - n_long
        # average RSI/ADX for context
        rsis = [o.entry_rsi for o in b if o.entry_rsi is not None]
        adxs = [o.entry_adx for o in b if o.entry_adx is not None]
        rows.append({
            "bucket": name,
            "n": n,
            "n_long": n_long,
            "n_short": n_short,
            "win_rate": round(100 * wins / n, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / n, 2),
            "avg_pct": round(avg_pct, 3),
            "avg_volume_usd": round(avg_vol_usd, 0),
            "avg_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
            "avg_adx": round(sum(adxs) / len(adxs), 2) if adxs else None,
        })
    return rows


def _compute_pair_performance(orders):
    """Per-pair performance: one row per pair with L/S breakdown"""
    closed = [o for o in orders if o.status == "CLOSED" and o.pnl is not None]
    from collections import defaultdict
    buckets = defaultdict(list)
    for o in closed:
        buckets[o.pair].append(o)
    rows = []
    for pair, trades in buckets.items():
        n = len(trades)
        longs = sum(1 for o in trades if o.direction == "LONG")
        shorts = n - longs
        wins = sum(1 for o in trades if o.pnl > 0)
        total_pnl = sum(o.pnl for o in trades)
        hold_hours = []
        for o in trades:
            if o.closed_at and o.opened_at:
                delta = (o.closed_at - o.opened_at).total_seconds() / 3600
                hold_hours.append(delta)
        avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0
        pnl_pct_sum = sum(o.pnl_percentage or 0 for o in trades)
        slippage_trades = [o for o in trades if o.exit_slippage_pct is not None]
        avg_slippage = round(sum(o.exit_slippage_pct for o in slippage_trades) / len(slippage_trades), 4) if slippage_trades else None
        # May 9: AvgATR% per pair — testing hypothesis "high volatility pairs drive losses"
        atr_trades = [o.entry_atr_pct for o in trades if o.entry_atr_pct is not None]
        avg_atr = round(sum(atr_trades) / len(atr_trades), 3) if atr_trades else None
        rows.append({
            "pair": pair,
            "longs": longs,
            "shorts": shorts,
            "trades": n,
            "win_rate": round(wins / n * 100, 1),
            "avg_pnl": round(total_pnl / n, 2),
            "avg_pnl_pct": round(pnl_pct_sum / n, 4),
            "total_pnl": round(total_pnl, 2),
            "avg_hold_hours": avg_hold,
            "avg_slippage_pct": avg_slippage,
            "avg_atr_pct": avg_atr,
        })
    rows.sort(key=lambda r: r["total_pnl"], reverse=True)
    return rows


def _compute_volume_crosstab(orders):
    """Cross-tab: Direction x GlobalVolBin x PairVolBin -> #Trades, WinRate, AvgP&L, TotalP&L"""
    rows = []
    closed = [o for o in orders if o.status == "CLOSED" and o.pnl is not None]
    for direction in ("LONG", "SHORT"):
        dir_orders = [o for o in closed if o.direction == direction]
        for g_label, g_lo, g_hi in _VOL_BINS:
            for p_label, p_lo, p_hi in _VOL_BINS:
                bucket = [
                    o for o in dir_orders
                    if o.entry_global_volume_ratio is not None
                    and o.entry_pair_volume_ratio is not None
                    and g_lo <= o.entry_global_volume_ratio < g_hi
                    and p_lo <= o.entry_pair_volume_ratio < p_hi
                ]
                if not bucket:
                    continue
                n = len(bucket)
                wins = sum(1 for o in bucket if o.pnl > 0)
                total_pnl = sum(o.pnl for o in bucket)
                hold_hours = []
                for o in bucket:
                    if o.closed_at and o.opened_at:
                        hold_hours.append((o.closed_at - o.opened_at).total_seconds() / 3600)
                avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in bucket)
                rows.append({
                    "direction": direction,
                    "global_vol": g_label,
                    "pair_vol": p_label,
                    "trades": n,
                    "win_rate": round(wins / n * 100, 1),
                    "avg_pnl": round(total_pnl / n, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / n, 4),
                    "total_pnl": round(total_pnl, 2),
                    "avg_hold_hours": avg_hold,
                })
    return rows


# May 25 — refactored from 11 narrow 5% bins to 7 wider purpose-built bins.
# OLD: <25 / 25-30 / 30-35 / 35-40 / 40-45 / 45-50 / 50-55 / 55-60 / 60-65 / 65-70 / 70%+
# NEW: <30 / 30-50 / 50-60 / 60-70 / 70-80 / 80-85 / 85%+
# Rationale: old layout lumped everything ≥70% into one bucket, masking the
# SHORT 85%+ capitulation cliff (-$764 cross-batch). New layout collapses
# rarely-populated low end (25-50% was 89 trades across 5 narrow bins) into
# 30-50 single bucket, and SPLITS the formerly-lumped tail into 70-80, 80-85,
# 85%+ so the cliff is visible. LONG worst zone (60-70%, -$1,428 cross-batch)
# gets its own row. Same boundaries for both directions (LONG uses Bull%,
# SHORT uses Bear%, same bin definitions).
_BREADTH_BINS = [
    ("< 30%", 0, 30),
    ("30-50%", 30, 50),
    ("50-60%", 50, 60),
    ("60-70%", 60, 70),
    ("70-80%", 70, 80),
    ("80-85%", 80, 85),
    ("85%+", 85, 101),
]


def _compute_breadth_crosstab(orders):
    """Cross-tab: Direction x Breadth% bin -> #Trades, WinRate, AvgP&L, TotalP&L
    For LONGs: binned by entry_bull_pct; for SHORTs: binned by entry_bear_pct."""
    rows = []
    closed = [o for o in orders if o.status == "CLOSED" and o.pnl is not None]
    for direction in ("LONG", "SHORT"):
        dir_orders = [o for o in closed if o.direction == direction]
        for b_label, b_lo, b_hi in _BREADTH_BINS:
            if direction == "LONG":
                bucket = [o for o in dir_orders if o.entry_bull_pct is not None and b_lo <= o.entry_bull_pct < b_hi]
            else:
                bucket = [o for o in dir_orders if o.entry_bear_pct is not None and b_lo <= o.entry_bear_pct < b_hi]
            if not bucket:
                continue
            n = len(bucket)
            wins = sum(1 for o in bucket if o.pnl > 0)
            total_pnl = sum(o.pnl for o in bucket)
            hold_hours = []
            for o in bucket:
                if o.closed_at and o.opened_at:
                    hold_hours.append((o.closed_at - o.opened_at).total_seconds() / 3600)
            avg_hold = sum(hold_hours) / len(hold_hours) if hold_hours else 0
            pnl_pct_sum = sum(o.pnl_percentage or 0 for o in bucket)
            rows.append({
                "direction": direction,
                "breadth_bin": b_label,
                "trades": n,
                "win_rate": round(wins / n * 100, 1),
                "avg_pnl": round(total_pnl / n, 2),
                "avg_pnl_pct": round(pnl_pct_sum / n, 4),
                "total_pnl": round(total_pnl, 2),
                "avg_hold_hours": avg_hold,
            })
    return rows


def _compute_day_time_heatmap(orders):
    """Compute win rate and avg P&L by day-of-week x 4-hour block (UTC-3)."""
    from datetime import timezone
    UTC_MINUS_3 = timezone(timedelta(hours=-3))
    grid = {}
    for d in range(7):
        for b in range(6):
            grid[(d, b)] = []
    for o in orders:
        if o.closed_at is None:
            continue
        ct = o.closed_at.replace(tzinfo=timezone.utc) if o.closed_at.tzinfo is None else o.closed_at
        local = ct.astimezone(UTC_MINUS_3)
        weekday = local.weekday()
        block = local.hour // 4
        grid[(weekday, block)].append(o)
    result = []
    for d in range(7):
        for b in range(6):
            trades = grid[(d, b)]
            count = len(trades)
            if count == 0:
                result.append({"day": d, "day_label": _DAY_LABELS[d],
                               "block": b, "block_label": _TIME_BLOCK_LABELS[b],
                               "trades": 0, "wins": 0, "win_rate": 0, "avg_pnl_pct": 0})
                continue
            wins = sum(1 for o in trades if (o.pnl or 0) > 0)
            avg_pnl = round(sum(o.pnl_percentage or 0 for o in trades) / count, 4)
            result.append({
                "day": d, "day_label": _DAY_LABELS[d],
                "block": b, "block_label": _TIME_BLOCK_LABELS[b],
                "trades": count, "wins": wins,
                "win_rate": round(wins / count * 100, 1),
                "avg_pnl_pct": avg_pnl,
            })
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
            gaps813 = [getattr(o, 'entry_ema_gap_8_13', None) for o in bucket_orders if getattr(o, 'entry_ema_gap_8_13', None) is not None]
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
                "avg_gap813": round(sum(gaps813) / len(gaps813), 4) if gaps813 else None,
                "avg_adx": round(sum(adxs) / len(adxs), 1) if adxs else None
            })
        return result
    except Exception as e:
        logger.error(f"[PERF] Error computing time buckets: {e}\n{traceback.format_exc()}")
        return []


async def _compute_performance(db: AsyncSession, regime: str = None, window_hours: int = None,
                                from_date: str = None, to_date: str = None):
    result = await db.execute(
        select(Order)
        .where(and_(Order.status == "CLOSED", Order.is_paper == trading_engine.is_paper_mode))
    )
    all_orders = result.scalars().all()

    # Amendment #7 (Apr 18): also fetch SIGNAL_EXPIRED rows for Entry Type Performance.
    # These are aborted entry attempts where the signal went stale during the maker wait.
    # They are NOT real trades (no PnL, no fill) so they are excluded from every
    # aggregation except the entry-type breakdown.
    #
    # Split-report fix (May 3, supersedes May 1):
    # SIGNAL_EXPIRED rows don't capture entry_macro_trend (no fill = no regime
    # context). Earlier code zeroed them out entirely for regime sections,
    # which fixed the 315.4% bug (SIGNAL_EXPIRED count / closed-shorts) but
    # also dropped the Aborted row from Entry Conditions by Outcome AND the
    # entire Signal Expired Breakdown table from split reports.
    #
    # Fix: always load all SIGNAL_EXPIRED orders. For regime-filtered calls,
    # filter by DIRECTION so BULLISH section sees LONG aborts only and
    # BEARISH section sees SHORT aborts only. This is correct because the
    # split report itself is direction-segregated (BULLISH = LONG-only,
    # BEARISH = SHORT-only). The 315.4% bug is fixed separately in
    # _compute_entry_type_stats by computing pct_of_total against an
    # aborts-inclusive denominator.
    signal_expired_result = await db.execute(
        select(Order)
        .where(and_(Order.status == "SIGNAL_EXPIRED", Order.is_paper == trading_engine.is_paper_mode))
    )
    _all_signal_expired = signal_expired_result.scalars().all()
    if regime is None:
        signal_expired_orders = _all_signal_expired
    elif regime == 'BULLISH':
        signal_expired_orders = [o for o in _all_signal_expired if (o.direction or 'LONG') == 'LONG']
    elif regime == 'BEARISH':
        signal_expired_orders = [o for o in _all_signal_expired if (o.direction or 'LONG') == 'SHORT']
    else:
        # NEUTRAL or any other regime — no clean direction mapping
        signal_expired_orders = []
    
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
        trend_avg_pnl_pct = round(sum(o.pnl_percentage or 0 for o in trend_orders) / len(trend_orders), 4) if trend_orders else 0
        macro_trend_performance[trend] = {
            "total_trades": len(trend_orders),
            "long_trades": len(trend_longs),
            "short_trades": len(trend_shorts),
            "total_pnl": round(trend_total_pnl, 2),
            "total_win_rate": trend_total_wr,
            "avg_pnl_pct": trend_avg_pnl_pct,
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

    # Apply time-window filter if requested (May 15 PM).
    # window_hours = N restricts to trades closed within the last N hours.
    # from_date / to_date (ISO YYYY-MM-DD, May 18) restrict to a date range and
    # take precedence over window_hours when both provided.
    # Applied AFTER the regime filter so the two compose. Same window applied
    # to signal_expired_orders so SIGNAL_EXPIRED rows in Entry Conditions by
    # Outcome / Signal Expired Breakdown reflect the same recency cut.
    if from_date and to_date:
        try:
            _start = datetime.strptime(from_date, "%Y-%m-%d")
            _end = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)  # inclusive end of day
            orders = [o for o in orders if o.closed_at and _start <= o.closed_at < _end]
            signal_expired_orders = [o for o in signal_expired_orders if o.closed_at and _start <= o.closed_at < _end]
        except ValueError:
            logger.warning(f"[PERF] Invalid date range from_date={from_date} to_date={to_date}, skipping filter")
    elif window_hours and window_hours > 0:
        _cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        orders = [o for o in orders if o.closed_at and o.closed_at >= _cutoff]
        signal_expired_orders = [o for o in signal_expired_orders if o.closed_at and o.closed_at >= _cutoff]

    if not orders:
        logger.warning("[PERF] No closed orders found for is_paper=%s (regime=%s), returning empty performance",
                       trading_engine.is_paper_mode, regime)
        early_open_result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == trading_engine.is_paper_mode)
            )
        )
        early_used_margin = sum(o.investment for o in early_open_result.scalars().all())
        if trading_engine.is_paper_mode:
            # Same reverse-derive approach as the main calc (CLAUDE.md May 5).
            # No closed orders here → total_pnl is 0 → return_multiple is 1.0x by construction.
            _early_bnb = (trading_engine.paper_bnb_balance_usd or 0)
            early_return_multiple = 1.0  # No realized P&L exists in this branch
        else:
            early_return_multiple = 1.0
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
            "ema813_gap_performance": [],
            "ema_fan_accel_performance": [],
            "rsi_performance": [],
            "range_position_performance": [],
            "adx_delta_performance": [],
            "adx_performance": [],
            "stretch_performance": [],
            "pair_slope_performance": [],
            "btc_slope_performance": [],
            "pair_ema20_ema50_gap_performance": [],
            "btc_ema20_ema50_gap_performance": [],
            "btc_adx_performance": [], "btc_adx_direction_performance": [], "btc_rsi_direction_performance": [], "btc_rsi_direction_30m_performance": [], "btc_volatility_performance": [], "btc_rsi_1h_direction_performance": [], "btc_vol_adx_crosstab": [], "btc_rsi_1h_5m_crosstab": [], "adx_dir_crosstab": [], "rsi_dir_crosstab": [], "btc_rsi_30m_5m_crosstab": [], "range_pos_btc_rsi_dir_crosstab": [], "range_pos_pair_rsi_dir_crosstab": [], "pair_slope_adx_crosstab": [], "btc_slope_adx_crosstab": [], "adx_delta_btc_adx_crosstab": [], "btc_gap_btc_adx_crosstab": [], "pair_gap_pair_adx_crosstab": [], "rsi_direction_performance": [],
            "btc_rsi_performance": [], "btc_rsi_adx_crosstab": [], "quality_score_performance": [],
            "regime_performance": [], "regime_transition_performance": [],
            "by_close_reason": {},
            "stop_loss_deep_dive": {"total_sl_trades": 0, "be_was_active": {"count": 0}, "positive_no_be": {"count": 0}, "never_positive": {"count": 0}, "avg_peak_all_sl": 0},
            "winning_trades_drawdown": [], "trough_recovery": [],
            "never_positive_deep_dive": [],
            "performance_over_time": [],
            "post_exit_regret_deep_dive": [],
            "hold_time_expectancy": [],
            "entry_conditions_by_reason": [],
            "entry_conditions_by_outcome": [],
            "multiplier_cell_performance": {"longs": [], "shorts": [], "summary": {}},
            "pattern_cell_performance": {"rules": [], "summary": {}},
            "extension_multiplier_performance": {"rules": [], "summary": {}},
            "btc_1h_slope_btc_adx_multiplier_performance": {"rules": [], "summary": {}},
            "pattern_4cohort_coverage": {"cohorts": [], "total": {}},
            "pattern_c_combo_tracker": {"rows": [], "tracker": "C"},
            "pattern_w_combo_tracker": {"rows": [], "tracker": "W"},
            "flagged_exits": [],
            "period_performance": [],
            "equity_curve": [],
            "pnl_distribution": [],
            "hourly_performance": [],
            "daily_performance": [],
            "day_time_heatmap": [],
        }

    # Separate longs and shorts
    longs = [o for o in orders if o.direction == "LONG"]
    shorts = [o for o in orders if o.direction == "SHORT"]
    
    # Winning trades (use all_ prefix to avoid shadowing by inner loops)
    all_wins = [o for o in orders if (o.pnl or 0) > 0]
    all_losses = [o for o in orders if (o.pnl or 0) <= 0]
    
    long_wins = [o for o in longs if (o.pnl or 0) > 0]
    short_wins = [o for o in shorts if (o.pnl or 0) > 0]
    
    # Calculate metrics
    total_trades = len(orders)
    total_longs = len(longs)
    total_shorts = len(shorts)
    
    win_rate = (len(all_wins) / total_trades * 100) if total_trades > 0 else 0
    win_rate_longs = (len(long_wins) / total_longs * 100) if total_longs > 0 else 0
    win_rate_shorts = (len(short_wins) / total_shorts * 100) if total_shorts > 0 else 0
    
    # Average wins
    avg_win = sum(o.pnl for o in all_wins) / len(all_wins) if all_wins else 0
    avg_win_long = sum(o.pnl for o in long_wins) / len(long_wins) if long_wins else 0
    avg_win_short = sum(o.pnl for o in short_wins) / len(short_wins) if short_wins else 0
    avg_loss = sum(o.pnl for o in all_losses) / len(all_losses) if all_losses else 0
    long_losses = [o for o in longs if (o.pnl or 0) <= 0]
    short_losses = [o for o in shorts if (o.pnl or 0) <= 0]
    avg_loss_long = sum(o.pnl for o in long_losses) / len(long_losses) if long_losses else 0
    avg_loss_short = sum(o.pnl for o in short_losses) / len(short_losses) if short_losses else 0
    
    # Expectancy per trade: E = WR * AvgWin - (1 - WR) * |AvgLoss|
    wr = win_rate / 100
    expectancy = (wr * avg_win) - ((1 - wr) * abs(avg_loss)) if total_trades > 0 else 0

    # Expectancy in % (leverage-/invest-invariant — preferred for cross-batch comparison)
    avg_win_pct = sum(o.pnl_percentage or 0 for o in all_wins) / len(all_wins) if all_wins else 0
    avg_loss_pct = sum(o.pnl_percentage or 0 for o in all_losses) / len(all_losses) if all_losses else 0
    avg_win_long_pct = sum(o.pnl_percentage or 0 for o in long_wins) / len(long_wins) if long_wins else 0
    avg_win_short_pct = sum(o.pnl_percentage or 0 for o in short_wins) / len(short_wins) if short_wins else 0
    avg_loss_long_pct = sum(o.pnl_percentage or 0 for o in long_losses) / len(long_losses) if long_losses else 0
    avg_loss_short_pct = sum(o.pnl_percentage or 0 for o in short_losses) / len(short_losses) if short_losses else 0
    expectancy_pct = (wr * avg_win_pct) - ((1 - wr) * abs(avg_loss_pct)) if total_trades > 0 else 0

    # Best/worst - Best win should only count winning trades (pnl > 0)
    long_wins_pnls = [o.pnl for o in longs if (o.pnl or 0) > 0]
    short_wins_pnls = [o.pnl for o in shorts if (o.pnl or 0) > 0]
    long_loss_pnls = [o.pnl for o in longs if (o.pnl or 0) <= 0]
    short_loss_pnls = [o.pnl for o in shorts if (o.pnl or 0) <= 0]

    best_win_long = max(long_wins_pnls) if long_wins_pnls else 0
    best_win_short = max(short_wins_pnls) if short_wins_pnls else 0
    worst_loss_long = min(long_loss_pnls) if long_loss_pnls else 0
    worst_loss_short = min(short_loss_pnls) if short_loss_pnls else 0

    # Best/worst as % — pick the % corresponding to the best/worst $ trade.
    # Using the same trade's pnl_percentage (not max-of-percentages) because
    # operator's mental model is "this best $ trade returned Y %".
    def _pct_of(orders_pnl_list, target_pnl):
        for o in orders_pnl_list:
            if (o.pnl or 0) == target_pnl:
                return o.pnl_percentage or 0
        return 0

    best_win_long_pct = _pct_of([o for o in longs if (o.pnl or 0) > 0], best_win_long) if long_wins_pnls else 0
    best_win_short_pct = _pct_of([o for o in shorts if (o.pnl or 0) > 0], best_win_short) if short_wins_pnls else 0
    worst_loss_long_pct = _pct_of([o for o in longs if (o.pnl or 0) <= 0], worst_loss_long) if long_loss_pnls else 0
    worst_loss_short_pct = _pct_of([o for o in shorts if (o.pnl or 0) <= 0], worst_loss_short) if short_loss_pnls else 0
    
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
    open_result = await db.execute(
        select(Order).where(
            and_(Order.status == "OPEN", Order.is_paper == trading_engine.is_paper_mode)
        )
    )
    open_orders_for_balance = open_result.scalars().all()
    used_margin = sum(o.investment for o in open_orders_for_balance)
    runtime_days = trading_engine.get_runtime_seconds() / 86400.0

    if trading_engine.is_paper_mode:
        # Paper mode: reverse-derive initial_balance the same way live mode does.
        # current_balance = free USDT + locked margin + BNB equivalent.
        # initial_balance = current_balance - total_realized_pnl.
        # By construction: return_multiple = 1 + total_pnl/initial. Always
        # internally consistent with the displayed P&L. Doesn't drift when
        # config paper_balance is edited. Same approach as live (line below).
        # See CLAUDE.md May 5 — Return Multiple fix.
        _bnb_now = (trading_engine.paper_bnb_balance_usd or 0)
        current_balance = trading_engine.paper_balance + used_margin + _bnb_now
        initial_balance = current_balance - total_pnl
    else:
        balance = await binance_service.get_balance()
        bnb_price = await binance_service.get_bnb_price()
        bnb_usd = balance['bnb_total'] * bnb_price if bnb_price > 0 else 0
        current_balance = balance['usdt_total'] + bnb_usd
        initial_balance = current_balance - total_pnl

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
        
        conf_avg_pnl_pct = round(sum(o.pnl_percentage or 0 for o in conf_orders) / len(conf_orders), 4) if conf_orders else 0
        confidence_performance[conf] = {
            "total_trades": len(conf_orders),
            "long_trades": len(conf_longs),
            "short_trades": len(conf_shorts),
            "total_pnl": round(conf_total_pnl, 2),
            "total_win_rate": conf_total_win_rate,
            "avg_pnl_pct": conf_avg_pnl_pct,
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
    ema813_gap_performance = []
    ema_fan_accel_performance = []
    rsi_performance = []
    range_position_performance = []
    adx_delta_performance = []
    adx_performance = []
    adx_direction_performance = []
    rsi_direction_performance = []  # May 15: pair RSI direction at entry
    stretch_performance = []
    pair_slope_performance = []
    btc_slope_performance = []
    pair_ema20_ema50_gap_performance = []
    btc_ema20_ema50_gap_performance = []
    btc_adx_performance = []
    btc_adx_direction_performance = []
    btc_rsi_direction_performance = []  # May 15: BTC RSI direction at entry
    btc_rsi_direction_30m_performance = []  # May 15: BTC RSI direction over 30min (sustained-momentum)
    btc_volatility_performance = []  # May 15 PM: BTC Volatility Regime (ATR%)
    btc_rsi_1h_direction_performance = []  # May 15 PM: BTC 1h RSI Direction (1h timeframe momentum)
    btc_vol_adx_crosstab = []  # May 15 PM: BTC Volatility × BTC ADX cross-tab (catches "violent chop" cell)
    btc_rsi_1h_5m_crosstab = []  # May 15 PM: BTC 1h RSI Dir × BTC 5m RSI Dir cross-tab (multi-TF alignment)
    adx_dir_crosstab = []
    rsi_dir_crosstab = []  # May 15: Pair RSI Dir × BTC RSI Dir cross-tab
    btc_rsi_30m_5m_crosstab = []  # May 15: BTC RSI 30m × BTC RSI 5m cross-tab
    range_pos_btc_rsi_dir_crosstab = []  # May 15: Range Position × BTC RSI Direction (5m)
    range_pos_pair_rsi_dir_crosstab = []  # May 15: Range Position × Pair RSI Direction (vs prev2)
    btc_slope_adx_crosstab = []
    pair_slope_adx_crosstab = []
    btc_rsi_performance = []
    btc_rsi_adx_crosstab = []
    quality_score_performance = []
    regime_performance = []
    regime_transition_performance = []
    try:
        gap_ranges = [
            ("0.00 - 0.05%", 0.00, 0.05),
            ("0.05 - 0.08%", 0.05, 0.08),
            ("0.08 - 0.10%", 0.08, 0.10),
            ("0.10 - 0.12%", 0.10, 0.12),
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
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
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        # Performance by Entry Gap EMA5-EMA8 (momentum gap)
        # May 2: split low-end buckets (0.02-0.06) into 0.01-wide sub-buckets
        # to match the granularity already done on the slope tables. Most data
        # clusters in the 0.02-0.06 range so finer resolution there is high-EV.
        ema58_ranges = [
            ("0.00 - 0.02%", 0.00, 0.02),
            ("0.02 - 0.03%", 0.02, 0.03),
            ("0.03 - 0.04%", 0.03, 0.04),
            ("0.04 - 0.05%", 0.04, 0.05),
            ("0.05 - 0.06%", 0.05, 0.06),
            ("0.06 - 0.08%", 0.06, 0.08),
            ("0.08 - 0.10%", 0.08, 0.10),
            ("0.10 - 0.12%", 0.10, 0.12),
            ("0.12 - 0.14%", 0.12, 0.14),
            ("0.14 - 0.16%", 0.14, 0.16),
            ("0.16 - 0.18%", 0.16, 0.18),
            ("0.18 - 0.20%", 0.18, 0.20),
            ("> 0.20%", 0.20, 999),
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                ema58_gap_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
                })

        # Performance by Entry Gap EMA8-EMA13 (May 27 — momentum stretch vs trend ref).
        # Complement to EMA5-EMA8 — measures distance from EMA8 to bot's key trend reference
        # (EMA13 is what EMA13_CROSS_EXIT fires against). Larger 8-13 gap = smoothed momentum
        # farther past trend ref. Same bucket structure as EMA5-EMA8 for direct visual
        # comparison. Pre-deploy trades have NULL on entry_ema_gap_8_13 — excluded silently.
        ema813_ranges = [
            ("0.00 - 0.02%", 0.00, 0.02),
            ("0.02 - 0.03%", 0.02, 0.03),
            ("0.03 - 0.04%", 0.03, 0.04),
            ("0.04 - 0.05%", 0.04, 0.05),
            ("0.05 - 0.06%", 0.05, 0.06),
            ("0.06 - 0.08%", 0.06, 0.08),
            ("0.08 - 0.10%", 0.08, 0.10),
            ("0.10 - 0.12%", 0.10, 0.12),
            ("0.12 - 0.14%", 0.12, 0.14),
            ("0.14 - 0.16%", 0.14, 0.16),
            ("0.16 - 0.18%", 0.16, 0.18),
            ("0.18 - 0.20%", 0.18, 0.20),
            ("> 0.20%", 0.20, 999),
        ]
        ema813_gap_orders = [o for o in orders if getattr(o, 'entry_ema_gap_8_13', None) is not None]
        for range_name, gap_min, gap_max in ema813_ranges:
            range_orders = [o for o in ema813_gap_orders if gap_min <= o.entry_ema_gap_8_13 < gap_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                ema813_gap_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
                })

        # EMA Fan Acceleration (May 28) — observation-only, no schema change.
        # fan_ratio = (EMA5-EMA8 gap) / (EMA8-EMA13 gap). Spatial proxy for whether the
        # EMA fan is accelerating at entry: >1 = fast EMAs pulling away from trend ref
        # (momentum accelerating); <1 = front of fan compressing (momentum decelerating /
        # trend maturing). Uses existing entry_ema_gap_5_8 + entry_ema_gap_8_13 — both are
        # post-May-27 only (entry_ema_gap_8_13 capture date), so this inherits that window.
        # Requires gap_8_13 > 0 to avoid divide-by-zero (near-zero 8-13 gaps skipped).
        # Buckets re-bucketed May 29 to ALIGN with the shipped fan_ratio dead-zone filter
        # (SHORT block [1.02,1.65) active, LONG block [0.85,1.70) observation-only). The
        # MID-fan band is a clean loser dead-zone in BOTH directions (mature/late trend = no
        # edge); the tails win (pullback entry <0.85 / fresh burst >1.65). Boundaries are kept
        # ROUND (not snapped to the exact 1.02/1.65/0.85/1.70 edges) so the table stays an
        # at-a-glance DRIFT DETECTOR — if losers migrate into <0.85 or >1.65, the band needs to
        # move. Precise band validation is done from the raw CSV each checkpoint, not this table.
        #   <0.85       = LONG keep-low winner tail (pullback)
        #   0.85-1.00   = LONG dead-zone start (SHORT keep-below)
        #   1.00-1.35   = shared dead-zone lower half (loser)
        #   1.35-1.65   = shared dead-zone upper half (loser; DOT/PEPE capitulation sat ~1.46)
        #   1.65-2.00   = keep-high winner tail start
        #   >2.00       = strong acceleration (winner)
        # May 29: split the >2.00 tail into 2-3 / 3-5 / >5 — the LONG keep-high tail
        # (fan>1.70, kept by the filter) is losing in the extreme (avg ratio 5.2, 25% WR)
        # while 1.65-2.00 wins. Splitting locates where the upper edge should be tightened.
        ema_fan_ranges = [
            ("<0.85", -999, 0.85),
            ("0.85-1.00", 0.85, 1.00),
            ("1.00-1.35", 1.00, 1.35),
            ("1.35-1.65", 1.35, 1.65),
            ("1.65-2.00", 1.65, 2.00),
            ("2.00-3.00", 2.00, 3.00),
            ("3.00-5.00", 3.00, 5.00),
            (">5.00", 5.00, 999),
        ]
        ema_fan_orders = [
            o for o in orders
            if getattr(o, 'entry_ema_gap_5_8', None) is not None
            and getattr(o, 'entry_ema_gap_8_13', None) is not None
            and (o.entry_ema_gap_8_13 or 0) > 0
        ]
        for range_name, r_min, r_max in ema_fan_ranges:
            range_orders = [
                o for o in ema_fan_orders
                if r_min <= (o.entry_ema_gap_5_8 / o.entry_ema_gap_8_13) < r_max
            ]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                ratio_sum = sum((o.entry_ema_gap_5_8 / o.entry_ema_gap_8_13) for o in dir_orders)
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                ema_fan_accel_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_ratio": round(ratio_sum / count, 2),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
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
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        adx_ranges = [
            ("15-18", 15, 18), ("18-22", 18, 22), ("22-25", 22, 25),
            ("25-28", 25, 28), ("28-30", 28, 30), ("30-33", 30, 33), ("33-35", 33, 35), ("35+", 35, 999),
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
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
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in d_orders)
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
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "avg_duration": f"{dur_h:02d}:{dur_m:02d}:{dur_s:02d}",
                    "by_confidence": conf_breakdown
                })
        # Performance by RSI Direction (Rising vs Falling at entry) — May 15
        rsi_dir_orders = [o for o in orders if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None]
        for rsi_dir_label in ["Rising", "Falling"]:
            if rsi_dir_label == "Rising":
                r_dir_pool = [o for o in rsi_dir_orders if o.entry_rsi > o.entry_rsi_prev]
            else:
                r_dir_pool = [o for o in rsi_dir_orders if o.entry_rsi <= o.entry_rsi_prev]
            if not r_dir_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                rd_orders = [o for o in r_dir_pool if (o.direction or "LONG") == direction]
                rd_count = len(rd_orders)
                if rd_count == 0:
                    continue
                rd_wins = len([o for o in rd_orders if (o.pnl or 0) > 0])
                rd_pnl_sum = sum(o.pnl or 0 for o in rd_orders)
                rd_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in rd_orders)
                rd_avg_dur = sum((o.closed_at - o.opened_at).total_seconds() for o in rd_orders if o.closed_at) / rd_count if rd_count > 0 else 0
                rd_h, rd_m, rd_s = int(rd_avg_dur // 3600), int((rd_avg_dur % 3600) // 60), int(rd_avg_dur % 60)
                rd_conf = {}
                for o in rd_orders:
                    c = o.confidence or "UNKNOWN"
                    rd_conf[c] = rd_conf.get(c, 0) + 1
                rsi_direction_performance.append({
                    "rsi_direction": rsi_dir_label,
                    "direction": direction,
                    "count": rd_count,
                    "win_rate": round(rd_wins / rd_count * 100, 1),
                    "avg_pnl_usd": round(rd_pnl_sum / rd_count, 2),
                    "avg_pnl_pct": round(rd_pnl_pct_sum / rd_count, 4),
                    "total_pnl_usd": round(rd_pnl_sum, 2),
                    "avg_duration": f"{rd_h:02d}:{rd_m:02d}:{rd_s:02d}",
                    "by_confidence": rd_conf
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
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
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        # Performance by Entry Range Position (price position in 20-candle high-low range)
        range_pos_ranges = [
            ("0-2%", 0, 2), ("2-5%", 2, 5), ("5-10%", 5, 10),
            ("10-15%", 10, 15), ("15-25%", 15, 25),
            ("25-50%", 25, 50), ("50-75%", 50, 75),
            ("75-85%", 75, 85), ("85-90%", 85, 90),
            ("90-95%", 90, 95), ("95-98%", 95, 98), ("98-100%", 98, 100.1),
        ]
        range_pos_orders = [o for o in orders if o.entry_range_position is not None]
        for range_name, rp_min, rp_max in range_pos_ranges:
            range_orders = [o for o in range_pos_orders if rp_min <= o.entry_range_position < rp_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                # May 15: avg duration column — tests "low-range losers ride longer" hypothesis
                avg_dur_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in dir_orders if o.closed_at) / count if count > 0 else 0
                dh, dm, ds = int(avg_dur_secs // 3600), int((avg_dur_secs % 3600) // 60), int(avg_dur_secs % 60)
                range_position_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "avg_duration": f"{dh:02d}:{dm:02d}:{ds:02d}",
                    "by_confidence": conf_breakdown
                })

        # Range Position × BTC RSI Direction Cross-Tab (May 15)
        # Tests: does macro RSI direction discriminate within Range Position buckets?
        # Specifically: low-range SHORTs (chasing) — do they fail more when BTC RSI Rising?
        rp_btcrsi_orders = [o for o in orders if o.entry_range_position is not None
                            and o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None]
        for rp_name, rp_min, rp_max in range_pos_ranges:
            for btc_rsi_dir in ["Rising", "Falling"]:
                if btc_rsi_dir == "Rising":
                    pool = [o for o in rp_btcrsi_orders
                            if rp_min <= o.entry_range_position < rp_max
                            and o.entry_btc_rsi > o.entry_btc_rsi_prev]
                else:
                    pool = [o for o in rp_btcrsi_orders
                            if rp_min <= o.entry_range_position < rp_max
                            and o.entry_btc_rsi <= o.entry_btc_rsi_prev]
                if not pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    cell = [o for o in pool if (o.direction or "LONG") == direction]
                    c_count = len(cell)
                    if c_count == 0:
                        continue
                    c_wins = len([o for o in cell if (o.pnl or 0) > 0])
                    c_pnl = sum(o.pnl or 0 for o in cell)
                    c_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in cell)
                    c_conf = {}
                    for o in cell:
                        c = o.confidence or "UNKNOWN"
                        c_conf[c] = c_conf.get(c, 0) + 1
                    range_pos_btc_rsi_dir_crosstab.append({
                        "range_position": rp_name,
                        "btc_rsi_dir": btc_rsi_dir,
                        "direction": direction,
                        "trades": c_count,
                        "win_rate": round(c_wins / c_count * 100, 1),
                        "avg_pnl": round(c_pnl / c_count, 2),
                        "avg_pnl_pct": round(c_pnl_pct_sum / c_count, 4),
                        "total_pnl": round(c_pnl, 2),
                        "by_confidence": c_conf,
                    })

        # Range Position × Pair RSI Direction Cross-Tab (May 15)
        # Tests: does pair-level RSI direction discriminate within Range Position buckets?
        # Pair RSI direction uses rsi_prev2 (~10min, matches RSI Momentum Filter logic).
        # Post-deploy trades only — entry_rsi_prev is NULL for historical orders.
        rp_pairrsi_orders = [o for o in orders if o.entry_range_position is not None
                             and o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None]
        for rp_name, rp_min, rp_max in range_pos_ranges:
            for pair_rsi_dir in ["Rising", "Falling"]:
                if pair_rsi_dir == "Rising":
                    pool = [o for o in rp_pairrsi_orders
                            if rp_min <= o.entry_range_position < rp_max
                            and o.entry_rsi > o.entry_rsi_prev]
                else:
                    pool = [o for o in rp_pairrsi_orders
                            if rp_min <= o.entry_range_position < rp_max
                            and o.entry_rsi <= o.entry_rsi_prev]
                if not pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    cell = [o for o in pool if (o.direction or "LONG") == direction]
                    c_count = len(cell)
                    if c_count == 0:
                        continue
                    c_wins = len([o for o in cell if (o.pnl or 0) > 0])
                    c_pnl = sum(o.pnl or 0 for o in cell)
                    c_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in cell)
                    c_conf = {}
                    for o in cell:
                        c = o.confidence or "UNKNOWN"
                        c_conf[c] = c_conf.get(c, 0) + 1
                    range_pos_pair_rsi_dir_crosstab.append({
                        "range_position": rp_name,
                        "pair_rsi_dir": pair_rsi_dir,
                        "direction": direction,
                        "trades": c_count,
                        "win_rate": round(c_wins / c_count * 100, 1),
                        "avg_pnl": round(c_pnl / c_count, 2),
                        "avg_pnl_pct": round(c_pnl_pct_sum / c_count, 4),
                        "total_pnl": round(c_pnl, 2),
                        "by_confidence": c_conf,
                    })

        # Performance by ADX Delta (adx - adx_prev at entry)
        adx_delta_ranges = [
            ("< -2.0", -999, -2.0), ("-2.0 to -1.0", -2.0, -1.0), ("-1.0 to -0.5", -1.0, -0.5),
            ("-0.5 to -0.3", -0.5, -0.3), ("-0.3 to -0.1", -0.3, -0.1), ("-0.1 to 0.0", -0.1, 0.0),
            ("0.0 to 0.05", 0.0, 0.05), ("0.05 to 0.1", 0.05, 0.1), ("0.1 to 0.3", 0.1, 0.3),
            ("0.3 to 0.5", 0.3, 0.5), ("0.5 to 1.0", 0.5, 1.0), ("1.0 to 2.0", 1.0, 2.0), ("> 2.0", 2.0, 999),
        ]
        adx_delta_orders = [o for o in orders if o.entry_adx_delta is not None]
        for range_name, d_min, d_max in adx_delta_ranges:
            range_orders = [o for o in adx_delta_orders if d_min <= o.entry_adx_delta < d_max]
            if not range_orders:
                continue
            for direction in ["LONG", "SHORT"]:
                dir_orders = [o for o in range_orders if (o.direction or "LONG") == direction]
                count = len(dir_orders)
                if count == 0:
                    continue
                dir_wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                conf_breakdown = {}
                for o in dir_orders:
                    conf = o.confidence or "UNKNOWN"
                    conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                adx_delta_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown
                })

        # Performance by Pair EMA20 Slope — SIGNED buckets (May 12).
        # Previously absolute-value bucketing collapsed downtrend and uptrend slopes
        # into the same row. Splitting by sign exposes direction-specific patterns
        # (e.g., a deep negative pair slope behaves differently for LONG vs SHORT
        # entries than a deep positive slope of the same magnitude).
        slope_ranges = [
            ("< -0.60%", -999, -0.60),
            ("-0.60 to -0.40%", -0.60, -0.40),
            ("-0.40 to -0.30%", -0.40, -0.30),
            ("-0.30 to -0.25%", -0.30, -0.25),
            ("-0.25 to -0.20%", -0.25, -0.20),
            ("-0.20 to -0.18%", -0.20, -0.18),
            ("-0.18 to -0.16%", -0.18, -0.16),
            ("-0.16 to -0.14%", -0.16, -0.14),
            ("-0.14 to -0.12%", -0.14, -0.12),
            ("-0.12 to -0.10%", -0.12, -0.10),
            ("-0.10 to -0.08%", -0.10, -0.08),
            ("-0.08 to -0.06%", -0.08, -0.06),
            ("-0.06 to -0.04%", -0.06, -0.04),
            ("-0.04 to -0.03%", -0.04, -0.03),
            ("-0.03 to -0.02%", -0.03, -0.02),
            ("-0.02 to -0.01%", -0.02, -0.01),
            ("-0.01 to 0%", -0.01, 0.0),
            ("0 to +0.01%", 0.0, 0.01),
            ("+0.01 to +0.02%", 0.01, 0.02),
            ("+0.02 to +0.03%", 0.02, 0.03),
            ("+0.03 to +0.04%", 0.03, 0.04),
            ("+0.04 to +0.06%", 0.04, 0.06),
            ("+0.06 to +0.08%", 0.06, 0.08),
            ("+0.08 to +0.10%", 0.08, 0.10),
            ("+0.10 to +0.12%", 0.10, 0.12),
            ("+0.12 to +0.14%", 0.12, 0.14),
            ("+0.14 to +0.16%", 0.14, 0.16),
            ("+0.16 to +0.18%", 0.16, 0.18),
            ("+0.18 to +0.20%", 0.18, 0.20),
            ("+0.20 to +0.25%", 0.20, 0.25),
            ("+0.25 to +0.30%", 0.25, 0.30),
            ("+0.30 to +0.40%", 0.30, 0.40),
            ("+0.40 to +0.60%", 0.40, 0.60),
            ("> +0.60%", 0.60, 999),
        ]
        pair_slope_orders = [o for o in orders if o.entry_ema20_slope is not None]
        for range_name, s_min, s_max in slope_ranges:
            range_orders = [o for o in pair_slope_orders if s_min <= o.entry_ema20_slope < s_max]
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                pair_slope_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
                })

        # Performance by BTC EMA20 Slope (absolute value, 0.02% buckets)
        btc_slope_orders = [o for o in orders if o.entry_btc_ema20_slope is not None]
        for range_name, s_min, s_max in slope_ranges:
            range_orders = [o for o in btc_slope_orders if s_min <= o.entry_btc_ema20_slope < s_max]
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                btc_slope_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
                })

        # Performance by Pair EMA13-EMA50 Gap at entry (May 5, observation-only).
        # May 12: refined to 24 signed buckets — fine 0.05% granularity in the
        # ±0.50% activity zone, coarser tails. Empty rows auto-drop. Same scheme
        # used by BTC version + NP Deep Dive gap dimension for apples-to-apples
        # cross-table comparison.
        pair_ema_gap_ranges = [
            ("< -1.00%", -999, -1.00),
            ("-1.00 to -0.80%", -1.00, -0.80),
            ("-0.80 to -0.60%", -0.80, -0.60),
            ("-0.60 to -0.50%", -0.60, -0.50),
            ("-0.50 to -0.40%", -0.50, -0.40),
            ("-0.40 to -0.30%", -0.40, -0.30),
            ("-0.30 to -0.25%", -0.30, -0.25),
            ("-0.25 to -0.20%", -0.25, -0.20),
            ("-0.20 to -0.15%", -0.20, -0.15),
            ("-0.15 to -0.10%", -0.15, -0.10),
            ("-0.10 to -0.05%", -0.10, -0.05),
            ("-0.05 to 0%", -0.05, 0.0),
            ("0 to +0.05%", 0.0, 0.05),
            ("+0.05 to +0.10%", 0.05, 0.10),
            ("+0.10 to +0.15%", 0.10, 0.15),
            ("+0.15 to +0.20%", 0.15, 0.20),
            ("+0.20 to +0.25%", 0.20, 0.25),
            ("+0.25 to +0.30%", 0.25, 0.30),
            ("+0.30 to +0.40%", 0.30, 0.40),
            ("+0.40 to +0.50%", 0.40, 0.50),
            ("+0.50 to +0.60%", 0.50, 0.60),
            ("+0.60 to +0.80%", 0.60, 0.80),
            ("+0.80 to +1.00%", 0.80, 1.00),
            ("> +1.00%", 1.00, 999),
        ]
        pair_ema20_ema50_gap_performance = []
        pair_ema_gap_orders = [o for o in orders if getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) is not None]
        for range_name, g_min, g_max in pair_ema_gap_ranges:
            range_orders = [o for o in pair_ema_gap_orders if g_min <= o.entry_pair_ema20_ema50_gap_pct < g_max]
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                pair_ema20_ema50_gap_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
                    "avg_duration": calc_avg_duration(dir_orders),
                })

        # Performance by BTC EMA20-EMA50 Gap at entry (May 6, observation-only)
        # Mirrors pair version. Uses entry_btc_trend_gap_pct (BTC EMA20 vs EMA50, ~4hr context).
        btc_ema20_ema50_gap_performance = []
        btc_ema_gap_orders = [o for o in orders if getattr(o, 'entry_btc_trend_gap_pct', None) is not None]
        for range_name, g_min, g_max in pair_ema_gap_ranges:
            range_orders = [o for o in btc_ema_gap_orders if g_min <= o.entry_btc_trend_gap_pct < g_max]
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                never_positive = sum(1 for o in dir_orders if (o.peak_pnl or 0) <= 0)
                btc_ema20_ema50_gap_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap,
                    "never_positive": never_positive,
                    "never_positive_pct": round(never_positive / count * 100, 1) if count else 0,
                    "avg_duration": calc_avg_duration(dir_orders),
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                btc_adx_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
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
                bd_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in bd_orders)
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
                    "avg_pnl_pct": round(bd_pnl_pct_sum / bd_count, 4),
                    "total_pnl_usd": round(bd_pnl_sum, 2),
                    "avg_duration": f"{bd_h:02d}:{bd_m:02d}:{bd_s:02d}",
                    "by_confidence": bd_conf
                })

        # Performance by BTC RSI Direction (Rising vs Falling at entry) — May 15
        btc_rsi_dir_orders = [o for o in orders if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None]
        for brsi_dir_label in ["Rising", "Falling"]:
            if brsi_dir_label == "Rising":
                br_dir_pool = [o for o in btc_rsi_dir_orders if o.entry_btc_rsi > o.entry_btc_rsi_prev]
            else:
                br_dir_pool = [o for o in btc_rsi_dir_orders if o.entry_btc_rsi <= o.entry_btc_rsi_prev]
            if not br_dir_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                brd_orders = [o for o in br_dir_pool if (o.direction or "LONG") == direction]
                brd_count = len(brd_orders)
                if brd_count == 0:
                    continue
                brd_wins = len([o for o in brd_orders if (o.pnl or 0) > 0])
                brd_pnl_sum = sum(o.pnl or 0 for o in brd_orders)
                brd_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in brd_orders)
                brd_avg_dur = sum((o.closed_at - o.opened_at).total_seconds() for o in brd_orders if o.closed_at) / brd_count if brd_count > 0 else 0
                brd_h, brd_m, brd_s = int(brd_avg_dur // 3600), int((brd_avg_dur % 3600) // 60), int(brd_avg_dur % 60)
                brd_conf = {}
                for o in brd_orders:
                    c = o.confidence or "UNKNOWN"
                    brd_conf[c] = brd_conf.get(c, 0) + 1
                btc_rsi_direction_performance.append({
                    "btc_rsi_direction": brsi_dir_label,
                    "direction": direction,
                    "count": brd_count,
                    "win_rate": round(brd_wins / brd_count * 100, 1),
                    "avg_pnl_usd": round(brd_pnl_sum / brd_count, 2),
                    "avg_pnl_pct": round(brd_pnl_pct_sum / brd_count, 4),
                    "total_pnl_usd": round(brd_pnl_sum, 2),
                    "avg_duration": f"{brd_h:02d}:{brd_m:02d}:{brd_s:02d}",
                    "by_confidence": brd_conf
                })

        # Performance by BTC RSI Direction (30min — sustained momentum) — May 15
        btc_rsi_30m_dir_orders = [o for o in orders if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None]
        for brsi30_dir_label in ["Rising", "Falling"]:
            if brsi30_dir_label == "Rising":
                br30_dir_pool = [o for o in btc_rsi_30m_dir_orders if o.entry_btc_rsi > o.entry_btc_rsi_prev6]
            else:
                br30_dir_pool = [o for o in btc_rsi_30m_dir_orders if o.entry_btc_rsi <= o.entry_btc_rsi_prev6]
            if not br30_dir_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                br30d_orders = [o for o in br30_dir_pool if (o.direction or "LONG") == direction]
                br30d_count = len(br30d_orders)
                if br30d_count == 0:
                    continue
                br30d_wins = len([o for o in br30d_orders if (o.pnl or 0) > 0])
                br30d_pnl_sum = sum(o.pnl or 0 for o in br30d_orders)
                br30d_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in br30d_orders)
                br30d_avg_dur = sum((o.closed_at - o.opened_at).total_seconds() for o in br30d_orders if o.closed_at) / br30d_count if br30d_count > 0 else 0
                br30d_h, br30d_m, br30d_s = int(br30d_avg_dur // 3600), int((br30d_avg_dur % 3600) // 60), int(br30d_avg_dur % 60)
                br30d_conf = {}
                for o in br30d_orders:
                    c = o.confidence or "UNKNOWN"
                    br30d_conf[c] = br30d_conf.get(c, 0) + 1
                btc_rsi_direction_30m_performance.append({
                    "btc_rsi_direction_30m": brsi30_dir_label,
                    "direction": direction,
                    "count": br30d_count,
                    "win_rate": round(br30d_wins / br30d_count * 100, 1),
                    "avg_pnl_usd": round(br30d_pnl_sum / br30d_count, 2),
                    "avg_pnl_pct": round(br30d_pnl_pct_sum / br30d_count, 4),
                    "total_pnl_usd": round(br30d_pnl_sum, 2),
                    "avg_duration": f"{br30d_h:02d}:{br30d_m:02d}:{br30d_s:02d}",
                    "by_confidence": br30d_conf
                })

        # BTC RSI 30m × BTC RSI 5m Cross-Tab — May 15
        # Diagnoses sustained-momentum vs flicker. Both Falling = clean bear; mixed = transition zones.
        btc_rsi_ct_orders = [o for o in orders if o.entry_btc_rsi is not None
                             and o.entry_btc_rsi_prev is not None
                             and getattr(o, 'entry_btc_rsi_prev6', None) is not None]
        for rsi30_dir in ["Rising", "Falling"]:
            for rsi5_dir in ["Rising", "Falling"]:
                pool = [o for o in btc_rsi_ct_orders
                        if (o.entry_btc_rsi > o.entry_btc_rsi_prev6) == (rsi30_dir == "Rising")
                        and (o.entry_btc_rsi > o.entry_btc_rsi_prev) == (rsi5_dir == "Rising")]
                if not pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    p_orders = [o for o in pool if (o.direction or "LONG") == direction]
                    p_count = len(p_orders)
                    if p_count == 0:
                        continue
                    p_wins = len([o for o in p_orders if (o.pnl or 0) > 0])
                    p_pnl = sum(o.pnl or 0 for o in p_orders)
                    p_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in p_orders)
                    p_conf = {}
                    for o in p_orders:
                        c = o.confidence or "UNKNOWN"
                        p_conf[c] = p_conf.get(c, 0) + 1
                    btc_rsi_30m_5m_crosstab.append({
                        "btc_rsi_30m_dir": rsi30_dir,
                        "btc_rsi_5m_dir": rsi5_dir,
                        "direction": direction,
                        "trades": p_count,
                        "win_rate": round(p_wins / p_count * 100, 1),
                        "avg_pnl": round(p_pnl / p_count, 2),
                        "avg_pnl_pct": round(p_pnl_pct_sum / p_count, 4),
                        "total_pnl": round(p_pnl, 2),
                        "by_confidence": p_conf,
                    })

        # Performance by BTC Volatility Regime — May 15 PM
        # ATR/price × 100. Distinguishes "violent chop" (high ATR + low ADX) from
        # "clean trend" (mid ATR + high ADX) — a class of regime ADX alone can't see.
        # NOTE: pre-deploy trades have entry_btc_atr_pct = NULL (excluded automatically).
        btc_vol_buckets = [
            ("< 0.10%", 0.0, 0.10),
            ("0.10 - 0.15%", 0.10, 0.15),
            ("0.15 - 0.20%", 0.15, 0.20),
            ("0.20 - 0.30%", 0.20, 0.30),
            ("0.30 - 0.45%", 0.30, 0.45),
            ("> 0.45%", 0.45, 999.0),
        ]
        btc_vol_orders = [o for o in orders if getattr(o, 'entry_btc_atr_pct', None) is not None]
        for bv_label, bv_lo, bv_hi in btc_vol_buckets:
            bv_pool = [o for o in btc_vol_orders if bv_lo <= o.entry_btc_atr_pct < bv_hi]
            if not bv_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                bvd_orders = [o for o in bv_pool if (o.direction or "LONG") == direction]
                bvd_count = len(bvd_orders)
                if bvd_count == 0:
                    continue
                bvd_wins = len([o for o in bvd_orders if (o.pnl or 0) > 0])
                bvd_pnl_sum = sum(o.pnl or 0 for o in bvd_orders)
                bvd_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in bvd_orders)
                bvd_avg_dur = sum((o.closed_at - o.opened_at).total_seconds() for o in bvd_orders if o.closed_at) / bvd_count if bvd_count > 0 else 0
                bvd_h, bvd_m, bvd_s = int(bvd_avg_dur // 3600), int((bvd_avg_dur % 3600) // 60), int(bvd_avg_dur % 60)
                bvd_conf = {}
                for o in bvd_orders:
                    c = o.confidence or "UNKNOWN"
                    bvd_conf[c] = bvd_conf.get(c, 0) + 1
                btc_volatility_performance.append({
                    "btc_volatility": bv_label,
                    "direction": direction,
                    "count": bvd_count,
                    "win_rate": round(bvd_wins / bvd_count * 100, 1),
                    "avg_pnl_usd": round(bvd_pnl_sum / bvd_count, 2),
                    "avg_pnl_pct": round(bvd_pnl_pct_sum / bvd_count, 4),
                    "total_pnl_usd": round(bvd_pnl_sum, 2),
                    "avg_duration": f"{bvd_h:02d}:{bvd_m:02d}:{bvd_s:02d}",
                    "by_confidence": bvd_conf
                })

        # Performance by BTC 1h RSI Direction — May 15 PM
        # 1h timeframe momentum slice. Adds to 5m (vs prev1) and 30m (vs prev6) family.
        # Direction = sign of (entry_btc_rsi_1h - entry_btc_rsi_1h_prev).
        # Flat treated as Rising (>=) for parity with other RSI Direction tables.
        btc_rsi_1h_dir_orders = [o for o in orders if getattr(o, 'entry_btc_rsi_1h', None) is not None
                                  and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None]
        for brsi1h_dir_label in ["Rising", "Falling"]:
            if brsi1h_dir_label == "Rising":
                br1h_pool = [o for o in btc_rsi_1h_dir_orders if o.entry_btc_rsi_1h > o.entry_btc_rsi_1h_prev]
            else:
                br1h_pool = [o for o in btc_rsi_1h_dir_orders if o.entry_btc_rsi_1h <= o.entry_btc_rsi_1h_prev]
            if not br1h_pool:
                continue
            for direction in ["LONG", "SHORT"]:
                br1h_orders = [o for o in br1h_pool if (o.direction or "LONG") == direction]
                br1h_count = len(br1h_orders)
                if br1h_count == 0:
                    continue
                br1h_wins = len([o for o in br1h_orders if (o.pnl or 0) > 0])
                br1h_pnl_sum = sum(o.pnl or 0 for o in br1h_orders)
                br1h_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in br1h_orders)
                br1h_avg_dur = sum((o.closed_at - o.opened_at).total_seconds() for o in br1h_orders if o.closed_at) / br1h_count if br1h_count > 0 else 0
                br1h_h, br1h_m, br1h_s = int(br1h_avg_dur // 3600), int((br1h_avg_dur % 3600) // 60), int(br1h_avg_dur % 60)
                br1h_conf = {}
                for o in br1h_orders:
                    c = o.confidence or "UNKNOWN"
                    br1h_conf[c] = br1h_conf.get(c, 0) + 1
                btc_rsi_1h_direction_performance.append({
                    "btc_rsi_1h_direction": brsi1h_dir_label,
                    "direction": direction,
                    "count": br1h_count,
                    "win_rate": round(br1h_wins / br1h_count * 100, 1),
                    "avg_pnl_usd": round(br1h_pnl_sum / br1h_count, 2),
                    "avg_pnl_pct": round(br1h_pnl_pct_sum / br1h_count, 4),
                    "total_pnl_usd": round(br1h_pnl_sum, 2),
                    "avg_duration": f"{br1h_h:02d}:{br1h_m:02d}:{br1h_s:02d}",
                    "by_confidence": br1h_conf
                })

        # BTC Volatility × BTC ADX Cross-Tab — May 15 PM
        # Decomposes "violent chop" (high ATR + low ADX = big swings, no direction)
        # vs "clean trend" (mid ATR + high ADX = ride-able momentum).
        # 3 vol buckets × 3 ADX buckets × direction.
        btc_vol_adx_orders = [o for o in orders if getattr(o, 'entry_btc_atr_pct', None) is not None
                              and o.entry_btc_adx is not None]
        vol_buckets_ct = [("Low <0.15%", 0.0, 0.15), ("Mid 0.15-0.30%", 0.15, 0.30), ("High >0.30%", 0.30, 999.0)]
        adx_buckets_ct = [("Low <20", 0.0, 20.0), ("Mid 20-30", 20.0, 30.0), ("High >30", 30.0, 999.0)]
        for v_lbl, v_lo, v_hi in vol_buckets_ct:
            for a_lbl, a_lo, a_hi in adx_buckets_ct:
                ct_pool = [o for o in btc_vol_adx_orders
                           if v_lo <= o.entry_btc_atr_pct < v_hi
                           and a_lo <= o.entry_btc_adx < a_hi]
                if not ct_pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    ct_orders = [o for o in ct_pool if (o.direction or "LONG") == direction]
                    ct_count = len(ct_orders)
                    if ct_count == 0:
                        continue
                    ct_wins = len([o for o in ct_orders if (o.pnl or 0) > 0])
                    ct_pnl = sum(o.pnl or 0 for o in ct_orders)
                    ct_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in ct_orders)
                    ct_conf = {}
                    for o in ct_orders:
                        c = o.confidence or "UNKNOWN"
                        ct_conf[c] = ct_conf.get(c, 0) + 1
                    btc_vol_adx_crosstab.append({
                        "btc_volatility": v_lbl,
                        "btc_adx": a_lbl,
                        "direction": direction,
                        "trades": ct_count,
                        "win_rate": round(ct_wins / ct_count * 100, 1),
                        "avg_pnl": round(ct_pnl / ct_count, 2),
                        "avg_pnl_pct": round(ct_pnl_pct_sum / ct_count, 4),
                        "total_pnl": round(ct_pnl, 2),
                        "by_confidence": ct_conf,
                    })

        # BTC 1h RSI × BTC 5m RSI Cross-Tab — May 15 PM
        # Multi-timeframe alignment: both Rising = clean uptrend, both Falling = clean
        # downtrend, mixed = transition zones (5m blip during 1h trend, or 1h reversal
        # showing first on 5m). Aligned cells should outperform mixed cells if the
        # multi-TF hypothesis is right.
        btc_rsi_mtf_orders = [o for o in orders if getattr(o, 'entry_btc_rsi_1h', None) is not None
                              and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None
                              and o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None]
        for r1h_dir in ["Rising", "Falling"]:
            for r5m_dir in ["Rising", "Falling"]:
                mtf_pool = [o for o in btc_rsi_mtf_orders
                            if (o.entry_btc_rsi_1h > o.entry_btc_rsi_1h_prev) == (r1h_dir == "Rising")
                            and (o.entry_btc_rsi > o.entry_btc_rsi_prev) == (r5m_dir == "Rising")]
                if not mtf_pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    mtf_orders = [o for o in mtf_pool if (o.direction or "LONG") == direction]
                    mtf_count = len(mtf_orders)
                    if mtf_count == 0:
                        continue
                    mtf_wins = len([o for o in mtf_orders if (o.pnl or 0) > 0])
                    mtf_pnl = sum(o.pnl or 0 for o in mtf_orders)
                    mtf_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in mtf_orders)
                    mtf_conf = {}
                    for o in mtf_orders:
                        c = o.confidence or "UNKNOWN"
                        mtf_conf[c] = mtf_conf.get(c, 0) + 1
                    btc_rsi_1h_5m_crosstab.append({
                        "btc_rsi_1h_dir": r1h_dir,
                        "btc_rsi_5m_dir": r5m_dir,
                        "direction": direction,
                        "trades": mtf_count,
                        "win_rate": round(mtf_wins / mtf_count * 100, 1),
                        "avg_pnl": round(mtf_pnl / mtf_count, 2),
                        "avg_pnl_pct": round(mtf_pnl_pct_sum / mtf_count, 4),
                        "total_pnl": round(mtf_pnl, 2),
                        "by_confidence": mtf_conf,
                    })

        # Pair RSI Dir × BTC RSI Dir Cross-Tab — May 15
        rsi_ct_orders = [o for o in orders if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None
                         and o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None]
        for pair_rdir in ["Rising", "Falling"]:
            for btc_rdir in ["Rising", "Falling"]:
                rct_pool = [o for o in rsi_ct_orders
                            if (o.entry_rsi > o.entry_rsi_prev) == (pair_rdir == "Rising")
                            and (o.entry_btc_rsi > o.entry_btc_rsi_prev) == (btc_rdir == "Rising")]
                if not rct_pool:
                    continue
                for direction in ["LONG", "SHORT"]:
                    rct_orders = [o for o in rct_pool if (o.direction or "LONG") == direction]
                    rct_count = len(rct_orders)
                    if rct_count == 0:
                        continue
                    rct_wins = len([o for o in rct_orders if (o.pnl or 0) > 0])
                    rct_pnl = sum(o.pnl or 0 for o in rct_orders)
                    rct_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in rct_orders)
                    rct_conf = {}
                    for o in rct_orders:
                        c = o.confidence or "UNKNOWN"
                        rct_conf[c] = rct_conf.get(c, 0) + 1
                    rsi_dir_crosstab.append({
                        "pair_rsi_dir": pair_rdir,
                        "btc_rsi_dir": btc_rdir,
                        "direction": direction,
                        "trades": rct_count,
                        "win_rate": round(rct_wins / rct_count * 100, 1),
                        "avg_pnl": round(rct_pnl / rct_count, 2),
                        "avg_pnl_pct": round(rct_pnl_pct_sum / rct_count, 4),
                        "total_pnl": round(rct_pnl, 2),
                        "by_confidence": rct_conf,
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
                    dc_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dc_orders)
                    adx_dir_crosstab.append({
                        "pair_adx_dir": pair_dir,
                        "btc_adx_dir": btc_dir,
                        "direction": direction,
                        "trades": dc_count,
                        "win_rate": round(dc_wins / dc_count * 100, 1),
                        "avg_pnl": round(dc_pnl / dc_count, 2),
                        "avg_pnl_pct": round(dc_pnl_pct_sum / dc_count, 4),
                        "total_pnl": round(dc_pnl, 2),
                        "by_confidence": dc_conf,
                    })

        # BTC Slope x BTC ADX Cross-Tab AND Pair Slope x Pair ADX Cross-Tab
        # (May 2): both refactored to Dir-first format, matching the EMA50/DI/Funding
        # cross-tabs. BTC slope buckets split at the low end (was single <0.06% lump,
        # now <0.02 / 0.02-0.04 / 0.04-0.06) to surface activity in the dominant zone.
        ct_slope_ranges = [
            ("<0.02%", 0.00, 0.02), ("0.02-0.04%", 0.02, 0.04), ("0.04-0.06%", 0.04, 0.06),
            ("0.06-0.10%", 0.06, 0.10), ("0.10-0.16%", 0.10, 0.16),
            ("0.16-0.25%", 0.16, 0.25), ("0.25-0.40%", 0.25, 0.40), (">0.40%", 0.40, 999),
        ]
        ct_adx_ranges_bt = [
            ("10-15", 10, 15), ("15-20", 15, 20), ("20-25", 20, 25),
            ("25-30", 25, 30), ("30-35", 30, 35), ("35+", 35, 999),
        ]
        # Pair-level ADX buckets match the existing "Performance by Entry ADX" cadence
        # (15-18, 18-22, 22-25, 25-28, 28-30, 30-33, 33+) plus a <15 catch-all. Tighter
        # than the BTC bins because pair ADX clusters more narrowly under current filters.
        ct_adx_ranges_pair = [
            ("<15", 0, 15), ("15-18", 15, 18), ("18-22", 18, 22), ("22-25", 22, 25),
            ("25-28", 25, 28), ("28-30", 28, 30), ("30-33", 30, 33), ("33+", 33, 999),
        ]

        def _build_slope_adx_crosstab(orders_pool, slope_attr, adx_attr, adx_ranges):
            """Build a Dir-first cross-tab over (slope, ADX) with per-direction rows.
            Returns a list of dicts shaped like the other cross-tab tables.
            """
            rows = []
            pool = [o for o in orders_pool
                    if getattr(o, slope_attr, None) is not None
                    and getattr(o, adx_attr, None) is not None]
            for direction in ["LONG", "SHORT"]:
                for sr_name, sr_min, sr_max in ct_slope_ranges:
                    for ar_name, ar_min, ar_max in adx_ranges:
                        bucket = [
                            o for o in pool
                            if (o.direction or "LONG") == direction
                            and sr_min <= abs(getattr(o, slope_attr)) < sr_max
                            and ar_min <= getattr(o, adx_attr) < ar_max
                        ]
                        if not bucket:
                            continue
                        b_count = len(bucket)
                        b_wins = len([o for o in bucket if (o.pnl or 0) > 0])
                        b_pnl = sum(o.pnl or 0 for o in bucket)
                        b_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket)
                        # Never Positive count per cell — same definition as Performance by RSI x ADX
                        b_never_positive = sum(1 for o in bucket if (o.peak_pnl or 0) <= 0)
                        b_conf = {}
                        for o in bucket:
                            c = o.confidence or "UNKNOWN"
                            b_conf[c] = b_conf.get(c, 0) + 1
                        rows.append({
                            "direction": direction,
                            "slope_range": sr_name,
                            "adx_range": ar_name,
                            "trades": b_count,
                            "win_rate": round(b_wins / b_count * 100, 1),
                            "avg_pnl": round(b_pnl / b_count, 2),
                            "avg_pnl_pct": round(b_pnl_pct / b_count, 4),
                            "total_pnl": round(b_pnl, 2),
                            "never_positive": b_never_positive,
                            "never_positive_pct": round(b_never_positive / b_count * 100, 1) if b_count else 0,
                            "by_confidence": b_conf,
                        })
            return rows

        # Pair Slope × Pair ADX (NEW, May 2) — placed before the BTC version in render
        pair_slope_adx_crosstab = _build_slope_adx_crosstab(
            orders, 'entry_ema20_slope', 'entry_adx', ct_adx_ranges_pair
        )
        # BTC Slope × BTC ADX (refactored)
        btc_slope_adx_crosstab = _build_slope_adx_crosstab(
            orders, 'entry_btc_ema20_slope', 'entry_btc_adx', ct_adx_ranges_bt
        )

        # ADX Delta × BTC ADX Cross-Tab (May 11 — pooled-data finding, see CLAUDE.md
        # May 11 deep review).  Tracks the regime-conditional ADX Delta pattern:
        # pair-level ADX spike (delta 1.0-2.0) is catastrophic when BTC ADX is in the
        # mid-strength 18-25 zone, but profitable when BTC ADX is 25-35.  Same dimension,
        # opposite outcomes by macro context.  Cell breakdown enables data-driven
        # 2D filter rules (block specific delta×BTC-ADX combos).
        adx_delta_ranges = [
            ("<0", -999, 0),
            ("0.0-0.1", 0, 0.1),
            ("0.1-0.3", 0.1, 0.3),
            ("0.3-0.5", 0.3, 0.5),
            ("0.5-1.0", 0.5, 1.0),
            ("1.0-2.0", 1.0, 2.0),
            (">2.0", 2.0, 999),
        ]
        btc_adx_ranges_for_delta = [
            ("<18", 0, 18),
            ("18-25", 18, 25),
            ("25-30", 25, 30),
            ("30-35", 30, 35),
            ("≥35", 35, 999),
        ]
        adx_delta_btc_adx_crosstab = []
        _adx_delta_pool = [o for o in orders
                           if getattr(o, 'entry_adx_delta', None) is not None
                           and getattr(o, 'entry_btc_adx', None) is not None]
        for direction in ["LONG", "SHORT"]:
            for dr_name, dr_min, dr_max in adx_delta_ranges:
                for ar_name, ar_min, ar_max in btc_adx_ranges_for_delta:
                    bucket = [
                        o for o in _adx_delta_pool
                        if (o.direction or "LONG") == direction
                        and dr_min <= o.entry_adx_delta < dr_max
                        and ar_min <= o.entry_btc_adx < ar_max
                    ]
                    if not bucket:
                        continue
                    b_count = len(bucket)
                    b_wins = len([o for o in bucket if (o.pnl or 0) > 0])
                    b_pnl = sum(o.pnl or 0 for o in bucket)
                    b_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket)
                    b_never_positive = sum(1 for o in bucket if (o.peak_pnl or 0) <= 0)
                    b_conf = {}
                    for o in bucket:
                        c = o.confidence or "UNKNOWN"
                        b_conf[c] = b_conf.get(c, 0) + 1
                    adx_delta_btc_adx_crosstab.append({
                        "direction": direction,
                        "delta_range": dr_name,
                        "btc_adx_range": ar_name,
                        "trades": b_count,
                        "win_rate": round(b_wins / b_count * 100, 1),
                        "avg_pnl": round(b_pnl / b_count, 2),
                        "avg_pnl_pct": round(b_pnl_pct / b_count, 4),
                        "total_pnl": round(b_pnl, 2),
                        "never_positive": b_never_positive,
                        "never_positive_pct": round(b_never_positive / b_count * 100, 1) if b_count else 0,
                        "by_confidence": b_conf,
                    })

        # BTC EMA13-EMA50 Gap × BTC ADX Cross-Tab (May 11 UTC-3 — see CLAUDE.md).
        # Surfaces whether BTC over-extension (gap > +0.10%) is uniformly bad across
        # BTC ADX magnitudes, or concentrated in a specific BTC ADX × gap intersection.
        # May 19: re-bucketed to use the same 24-bucket fine grid as the 1D BTC Gap
        # table (pair_ema_gap_ranges) so 1D and 2D views are directly comparable.
        # Surfaces the +0.10-0.20% loser sub-zone with finer resolution (+0.10-0.15
        # vs +0.15-0.20) — critical for tuning the May 19 cross-filter rules.
        btc_gap_btc_adx_crosstab = []
        _btc_gap_pool = [o for o in orders
                         if getattr(o, 'entry_btc_trend_gap_pct', None) is not None
                         and getattr(o, 'entry_btc_adx', None) is not None]
        for direction in ["LONG", "SHORT"]:
            for gr_name, gr_min, gr_max in pair_ema_gap_ranges:
                for ar_name, ar_min, ar_max in btc_adx_ranges_for_delta:
                    bucket = [
                        o for o in _btc_gap_pool
                        if (o.direction or "LONG") == direction
                        and gr_min <= o.entry_btc_trend_gap_pct < gr_max
                        and ar_min <= o.entry_btc_adx < ar_max
                    ]
                    if not bucket:
                        continue
                    b_count = len(bucket)
                    b_wins = len([o for o in bucket if (o.pnl or 0) > 0])
                    b_pnl = sum(o.pnl or 0 for o in bucket)
                    b_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket)
                    b_never_positive = sum(1 for o in bucket if (o.peak_pnl or 0) <= 0)
                    b_conf = {}
                    for o in bucket:
                        c = o.confidence or "UNKNOWN"
                        b_conf[c] = b_conf.get(c, 0) + 1
                    btc_gap_btc_adx_crosstab.append({
                        "direction": direction,
                        "gap_range": gr_name,
                        "btc_adx_range": ar_name,
                        "trades": b_count,
                        "win_rate": round(b_wins / b_count * 100, 1),
                        "avg_pnl": round(b_pnl / b_count, 2),
                        "avg_pnl_pct": round(b_pnl_pct / b_count, 4),
                        "total_pnl": round(b_pnl, 2),
                        "never_positive": b_never_positive,
                        "never_positive_pct": round(b_never_positive / b_count * 100, 1) if b_count else 0,
                        "by_confidence": b_conf,
                    })

        # Pair EMA13-EMA50 Gap × Pair ADX Cross-Tab (May 24 — pair-level counterpart of BTC version).
        # Reuses pair_ema_gap_ranges (24 fine buckets) for direct 1D/2D comparison and
        # ct_adx_ranges_pair for finer pair-ADX resolution. Tests whether pair over-extension
        # (gap > +0.10%) at a given pair ADX is uniformly bad, or concentrated in cells.
        # Mirror of the BTC version per CLAUDE.md May 19 design — surfaces sub-cell structure
        # in the pair's own multi-hour trend.
        pair_gap_pair_adx_crosstab = []
        _pair_gap_pool = [o for o in orders
                          if getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) is not None
                          and getattr(o, 'entry_adx', None) is not None]
        for direction in ["LONG", "SHORT"]:
            for gr_name, gr_min, gr_max in pair_ema_gap_ranges:
                for ar_name, ar_min, ar_max in ct_adx_ranges_pair:
                    bucket = [
                        o for o in _pair_gap_pool
                        if (o.direction or "LONG") == direction
                        and gr_min <= o.entry_pair_ema20_ema50_gap_pct < gr_max
                        and ar_min <= o.entry_adx < ar_max
                    ]
                    if not bucket:
                        continue
                    b_count = len(bucket)
                    b_wins = len([o for o in bucket if (o.pnl or 0) > 0])
                    b_pnl = sum(o.pnl or 0 for o in bucket)
                    b_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket)
                    b_never_positive = sum(1 for o in bucket if (o.peak_pnl or 0) <= 0)
                    b_conf = {}
                    for o in bucket:
                        c = o.confidence or "UNKNOWN"
                        b_conf[c] = b_conf.get(c, 0) + 1
                    pair_gap_pair_adx_crosstab.append({
                        "direction": direction,
                        "gap_range": gr_name,
                        "pair_adx_range": ar_name,
                        "trades": b_count,
                        "win_rate": round(b_wins / b_count * 100, 1),
                        "avg_pnl": round(b_pnl / b_count, 2),
                        "avg_pnl_pct": round(b_pnl_pct / b_count, 4),
                        "total_pnl": round(b_pnl, 2),
                        "never_positive": b_never_positive,
                        "never_positive_pct": round(b_never_positive / b_count * 100, 1) if b_count else 0,
                        "by_confidence": b_conf,
                    })

        # Performance by BTC Entry RSI
        btc_rsi_ranges = [
            ("<20", 0, 20), ("20-25", 20, 25), ("25-30", 25, 30),
            ("30-35", 30, 35), ("35-40", 35, 40), ("40-45", 40, 45), ("45-50", 45, 50),
            ("50-55", 50, 55), ("55-60", 55, 60), ("60-65", 60, 65), ("65-70", 65, 70), ("70+", 70, 999),
        ]
        btc_rsi_orders = [o for o in orders if o.entry_btc_rsi is not None]
        for range_name, r_min, r_max in btc_rsi_ranges:
            range_orders = [o for o in btc_rsi_orders if r_min <= o.entry_btc_rsi < r_max]
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
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                btc_rsi_performance.append({
                    "range": range_name,
                    "direction": direction,
                    "count": count,
                    "win_rate": round(dir_wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": conf_breakdown,
                    "avg_rsi": avg_rsi,
                    "avg_adx": avg_adx,
                    "avg_gap": avg_gap
                })

        # BTC RSI x BTC ADX Cross-Tab
        ct_btc_rsi_ranges = [
            ("<20", 0, 20), ("20-25", 20, 25), ("25-30", 25, 30),
            ("30-35", 30, 35), ("35-40", 35, 40), ("40-45", 40, 45), ("45-50", 45, 50),
            ("50-55", 50, 55), ("55-60", 55, 60), ("60-65", 60, 65), ("65-70", 65, 70), ("70+", 70, 999),
        ]
        ct_btc_adx_ranges = [
            ("10-15", 10, 15), ("15-20", 15, 20), ("20-25", 20, 25),
            ("25-30", 25, 30), ("30-35", 30, 35), ("35+", 35, 999),
        ]
        ct_btc_rsi_orders = [o for o in orders if o.entry_btc_rsi is not None and o.entry_btc_adx is not None]
        for direction in ["LONG", "SHORT"]:
            dir_ct = [o for o in ct_btc_rsi_orders if (o.direction or "LONG") == direction]
            for rsi_name, rsi_lo, rsi_hi in ct_btc_rsi_ranges:
                for adx_name, adx_lo, adx_hi in ct_btc_adx_ranges:
                    bucket = [o for o in dir_ct if rsi_lo <= o.entry_btc_rsi < rsi_hi and adx_lo <= o.entry_btc_adx < adx_hi]
                    if not bucket:
                        continue
                    ct_wins = len([o for o in bucket if (o.pnl or 0) > 0])
                    ct_pnl_sum = sum(o.pnl or 0 for o in bucket)
                    ct_count = len(bucket)
                    ct_pnl_pct_sum = sum(o.pnl_percentage or 0 for o in bucket)
                    # May 2: Never Positive count per cell — see Performance by RSI x ADX
                    ct_never_positive = sum(1 for o in bucket if (o.peak_pnl or 0) <= 0)
                    btc_rsi_adx_crosstab.append({
                        "direction": direction,
                        "btc_rsi_range": rsi_name,
                        "btc_adx_range": adx_name,
                        "trades": ct_count,
                        "win_rate": round(ct_wins / ct_count * 100, 1),
                        "avg_pnl": round(ct_pnl_sum / ct_count, 2),
                        "avg_pnl_pct": round(ct_pnl_pct_sum / ct_count, 4),
                        "total_pnl": round(ct_pnl_sum, 2),
                        "never_positive": ct_never_positive,
                        "never_positive_pct": round(ct_never_positive / ct_count * 100, 1) if ct_count else 0,
                    })

        # Quality Score Performance
        try:
            qs_orders = [o for o in orders if o.entry_quality_score is not None]
            for score_val in range(7):  # 0-6
                score_orders = [o for o in qs_orders if int(o.entry_quality_score) == score_val]
                if not score_orders:
                    continue
                for direction in ["LONG", "SHORT"]:
                    dir_orders = [o for o in score_orders if (o.direction or "LONG") == direction]
                    if not dir_orders:
                        continue
                    count = len(dir_orders)
                    wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                    pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                    pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                    conf_breakdown = {}
                    for o in dir_orders:
                        conf = o.confidence or "UNKNOWN"
                        conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                    quality_score_performance.append({
                        "range": str(score_val),
                        "direction": direction,
                        "count": count,
                        "win_rate": round(wins / count * 100, 1),
                        "avg_pnl_usd": round(pnl_sum / count, 2),
                        "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                        "total_pnl_usd": round(pnl_sum, 2),
                        "by_confidence": conf_breakdown
                    })
        except Exception as e:
            logger.error(f"[PERF] Error computing quality score performance: {e}")

        # BTC Regime Performance — group by entry_btc_regime
        try:
            regime_orders = [o for o in orders if o.entry_btc_regime is not None]
            # CHOPPY is the legacy single-bucket label (pre-split historical rows).
            # CHOPPY_WEAK / CHOPPY_FLAT are the new split labels used by
            # classify_btc_regime for all new trades.
            regime_labels = ["CHOPPY", "CHOPPY_WEAK", "CHOPPY_FLAT",
                             "HEALTHY_BULL", "STRONG_BULL", "BULL_EXHAUSTED",
                             "HEALTHY_BEAR", "STRONG_BEAR", "BEAR_EXHAUSTED", "UNKNOWN"]
            for regime in regime_labels:
                bucket = [o for o in regime_orders if o.entry_btc_regime == regime]
                if not bucket:
                    continue
                for direction in ["LONG", "SHORT"]:
                    dir_orders = [o for o in bucket if (o.direction or "LONG") == direction]
                    if not dir_orders:
                        continue
                    count = len(dir_orders)
                    wins = len([o for o in dir_orders if (o.pnl or 0) > 0])
                    pnl_sum = sum(o.pnl or 0 for o in dir_orders)
                    pnl_pct_sum = sum(o.pnl_percentage or 0 for o in dir_orders)
                    peaks = [o.peak_pnl or 0 for o in dir_orders]
                    avg_peak = round(sum(peaks) / count, 4) if peaks else 0
                    conf_breakdown = {}
                    for o in dir_orders:
                        conf = o.confidence or "UNKNOWN"
                        conf_breakdown[conf] = conf_breakdown.get(conf, 0) + 1
                    regime_performance.append({
                        "range": regime,
                        "direction": direction,
                        "count": count,
                        "win_rate": round(wins / count * 100, 1),
                        "avg_pnl_usd": round(pnl_sum / count, 2),
                        "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                        "total_pnl_usd": round(pnl_sum, 2),
                        "avg_peak_pct": avg_peak,
                        "by_confidence": conf_breakdown
                    })

            # Regime Transition Performance — entry vs exit regime
            transition_orders = [o for o in regime_orders if o.exit_btc_regime is not None]
            same_regime = [o for o in transition_orders if o.entry_btc_regime == o.exit_btc_regime]
            shifted_regime = [o for o in transition_orders if o.entry_btc_regime != o.exit_btc_regime]
            for label, group in [("SAME_REGIME", same_regime), ("REGIME_SHIFT", shifted_regime)]:
                if not group:
                    continue
                count = len(group)
                wins = len([o for o in group if (o.pnl or 0) > 0])
                pnl_sum = sum(o.pnl or 0 for o in group)
                pnl_pct_sum = sum(o.pnl_percentage or 0 for o in group)
                regime_transition_performance.append({
                    "range": label,
                    "direction": "ALL",
                    "count": count,
                    "win_rate": round(wins / count * 100, 1),
                    "avg_pnl_usd": round(pnl_sum / count, 2),
                    "avg_pnl_pct": round(pnl_pct_sum / count, 4),
                    "total_pnl_usd": round(pnl_sum, 2),
                    "by_confidence": {}
                })
        except Exception as e:
            logger.error(f"[PERF] Error computing regime performance: {e}")

    except Exception as e:
        logger.error(f"[PERF] Error computing gap/rsi/adx/stretch performance: {e}\n{traceback.format_exc()}")
        gap_performance = []
        ema58_gap_performance = []
        rsi_performance = []
        range_position_performance = []
        adx_delta_performance = []
        adx_performance = []
        adx_direction_performance = []
        rsi_direction_performance = []  # May 15
        stretch_performance = []
        pair_slope_performance = []
        btc_slope_performance = []
        pair_ema20_ema50_gap_performance = []
        btc_ema20_ema50_gap_performance = []
        btc_adx_performance = []
        btc_adx_direction_performance = []
        btc_rsi_direction_performance = []  # May 15
        btc_rsi_direction_30m_performance = []  # May 15
        btc_volatility_performance = []  # May 15 PM
        btc_rsi_1h_direction_performance = []  # May 15 PM
        btc_vol_adx_crosstab = []  # May 15 PM
        btc_rsi_1h_5m_crosstab = []  # May 15 PM
        adx_dir_crosstab = []
        rsi_dir_crosstab = []  # May 15
        btc_rsi_30m_5m_crosstab = []  # May 15
        range_pos_btc_rsi_dir_crosstab = []  # May 15
        range_pos_pair_rsi_dir_crosstab = []  # May 15
        btc_slope_adx_crosstab = []
        pair_slope_adx_crosstab = []
        btc_rsi_performance = []
        btc_rsi_adx_crosstab = []
        quality_score_performance = []
        regime_performance = []
        regime_transition_performance = []
    
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
            # Trough tracking (May 6 — added at user request). trough_pnl is signed
            # (negative for trades that went under water, 0 for trades that never went red).
            troughs = [o.trough_pnl or 0 for o in data["trades"]]
            avg_trough = sum(troughs) / count if count > 0 else 0
            worst_trough = min(troughs) if troughs else 0  # Most negative = worst
            sig_active = sum(1 for o in data["trades"] if o.signal_active_at_close is True)
            sig_inactive = sum(1 for o in data["trades"] if o.signal_active_at_close is False)
            gaps = [o.entry_gap for o in data["trades"] if o.entry_gap is not None]
            rsis = [o.entry_rsi for o in data["trades"] if o.entry_rsi is not None]
            post_exit_peaks = [o.post_exit_peak_pnl for o in data["trades"] if o.post_exit_peak_pnl is not None]
            post_exit_troughs = [o.post_exit_trough_pnl for o in data["trades"] if o.post_exit_trough_pnl is not None]
            # May 17: post-arm minimum (BE-floor counterfactual support)
            post_arm_mins = [getattr(o, 'post_arm_min_pnl_pct', None) for o in data["trades"] if getattr(o, 'post_arm_min_pnl_pct', None) is not None]

            rsi2_fired = [o for o in data["trades"] if o.first_rsi2_pnl is not None]
            ema5_dists = [o.exit_price_vs_ema5_pct for o in data["trades"] if o.exit_price_vs_ema5_pct is not None]
            # Exit-time BTC trend gap (May 6, pair-side dropped May 7)
            exit_btc_trend_gaps = [getattr(o, 'exit_btc_trend_gap_pct', None) for o in data["trades"] if getattr(o, 'exit_btc_trend_gap_pct', None) is not None]

            by_close_reason[reason] = {
                "trades": count,
                "avg_pnl_pct": round(data["pnl_pct_sum"] / count, 2) if count > 0 else 0,
                "avg_pnl_usd": round(data["pnl_sum"] / count, 2) if count > 0 else 0,
                "total_pnl_usd": round(data["pnl_sum"], 2),
                "by_confidence": data["by_confidence"],
                "by_direction": data["by_direction"],
                "avg_price_drop": round(avg_drop, 4),
                "avg_peak_pnl_pct": round(avg_peak, 4),
                "avg_trough_pnl_pct": round(avg_trough, 4),
                "worst_trough_pnl_pct": round(worst_trough, 4),
                "avg_post_exit_peak_pnl": round(sum(post_exit_peaks) / len(post_exit_peaks), 4) if post_exit_peaks else None,
                "avg_post_exit_trough_pnl": round(sum(post_exit_troughs) / len(post_exit_troughs), 4) if post_exit_troughs else None,
                "avg_post_arm_min_pct": round(sum(post_arm_mins) / len(post_arm_mins), 4) if post_arm_mins else None,
                "post_arm_min_n": len(post_arm_mins),
                "signal_active": sig_active,
                "signal_inactive": sig_inactive,
                "avg_entry_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "avg_entry_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_duration": calc_avg_duration(data["trades"]),
                "rsi2_fire_pct": round(len(rsi2_fired) / count * 100, 1) if count > 0 else 0,
                "avg_rsi2_pnl": round(sum(o.first_rsi2_pnl for o in rsi2_fired) / len(rsi2_fired), 4) if rsi2_fired else None,
                "avg_rsi2_min": round(sum(o.first_rsi2_minutes for o in rsi2_fired) / len(rsi2_fired), 1) if rsi2_fired else None,
                "avg_exit_ema5_pct": round(sum(ema5_dists) / len(ema5_dists), 4) if ema5_dists else None,
                "avg_exit_btc_trend_gap": round(sum(exit_btc_trend_gaps) / len(exit_btc_trend_gaps), 4) if exit_btc_trend_gaps else None,
            }
    except Exception as e:
        logger.error(f"[PERF] Error computing close reason stats: {e}\n{traceback.format_exc()}")
        by_close_reason = {}
    
    # Entry Conditions by Close Reason
    # EMA Fan Acceleration helper (May 28) — avg of per-trade fan_ratio (5-8 gap / 8-13 gap)
    # for a group. Shared by Entry Conditions by Close Reason + by Outcome. gap_8_13 > 0 guard.
    def _avg_fan(grp):
        fr = [o.entry_ema_gap_5_8 / o.entry_ema_gap_8_13 for o in grp
              if o.entry_ema_gap_5_8 is not None
              and getattr(o, 'entry_ema_gap_8_13', None) and o.entry_ema_gap_8_13 > 0]
        return round(sum(fr) / len(fr), 2) if fr else None

    entry_conditions_by_reason = []
    try:
        # Same trigger lookup used by Stop Loss Deep Dive (line ~3870) so the
        # SL Profile column is consistent with that table's classification.
        _ecr_tc = config.trading_config
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
            direction = o.direction or "UNKNOWN"
            ecr_groups.setdefault((reason, direction), []).append(o)

        for (reason, direction) in sorted(ecr_groups.keys()):
            group = ecr_groups[(reason, direction)]
            count = len(group)
            if count == 0:
                continue

            ec_conf = {}
            for o in group:
                c = o.confidence or "UNKNOWN"
                ec_conf[c] = ec_conf.get(c, 0) + 1

            rsis = [o.entry_rsi for o in group if o.entry_rsi is not None]
            adxs = [o.entry_adx for o in group if o.entry_adx is not None]
            gaps = [o.entry_gap for o in group if o.entry_gap is not None]
            gaps58 = [o.entry_ema_gap_5_8 for o in group if o.entry_ema_gap_5_8 is not None]
            gaps813 = [getattr(o, 'entry_ema_gap_8_13', None) for o in group if getattr(o, 'entry_ema_gap_8_13', None) is not None]
            stretches = [o.entry_ema5_stretch for o in group if o.entry_ema5_stretch is not None]
            slopes = [abs(o.entry_ema20_slope) for o in group if o.entry_ema20_slope is not None]
            btc_slopes = [abs(o.entry_btc_ema20_slope) for o in group if o.entry_btc_ema20_slope is not None]
            btc_adxs = [o.entry_btc_adx for o in group if o.entry_btc_adx is not None]
            btc_rsis = [o.entry_btc_rsi for o in group if o.entry_btc_rsi is not None]

            adx_rising = sum(1 for o in group if o.entry_adx is not None and o.entry_adx_prev is not None and o.entry_adx > o.entry_adx_prev)
            adx_falling = sum(1 for o in group if o.entry_adx is not None and o.entry_adx_prev is not None and o.entry_adx <= o.entry_adx_prev)
            btc_adx_rising = sum(1 for o in group if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None and o.entry_btc_adx > o.entry_btc_adx_prev)
            btc_adx_falling = sum(1 for o in group if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None and o.entry_btc_adx <= o.entry_btc_adx_prev)
            btc_rsi_rising = sum(1 for o in group if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None and o.entry_btc_rsi > o.entry_btc_rsi_prev)
            btc_rsi_falling = sum(1 for o in group if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None and o.entry_btc_rsi <= o.entry_btc_rsi_prev)
            # May 15: Pair RSI direction (matches RSI Momentum Filter — rsi vs rsi_prev2)
            rsi_rising = sum(1 for o in group if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None and o.entry_rsi > o.entry_rsi_prev)
            rsi_falling = sum(1 for o in group if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None and o.entry_rsi <= o.entry_rsi_prev)
            # May 15: BTC RSI 30min sustained-momentum direction
            btc_rsi_30m_rising = sum(1 for o in group if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None and o.entry_btc_rsi > o.entry_btc_rsi_prev6)
            btc_rsi_30m_falling = sum(1 for o in group if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None and o.entry_btc_rsi <= o.entry_btc_rsi_prev6)
            # May 15 PM: BTC Volatility (ATR%) — avg over the group
            btc_atrs = [getattr(o, 'entry_btc_atr_pct', None) for o in group if getattr(o, 'entry_btc_atr_pct', None) is not None]
            # May 15 PM: BTC 1h RSI Direction (rsi_1h vs rsi_1h_prev)
            btc_rsi_1h_rising = sum(1 for o in group if getattr(o, 'entry_btc_rsi_1h', None) is not None and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None and o.entry_btc_rsi_1h > o.entry_btc_rsi_1h_prev)
            btc_rsi_1h_falling = sum(1 for o in group if getattr(o, 'entry_btc_rsi_1h', None) is not None and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None and o.entry_btc_rsi_1h <= o.entry_btc_rsi_1h_prev)
            ema5_dists = [o.entry_price_vs_ema5_pct for o in group if o.entry_price_vs_ema5_pct is not None]
            adx_deltas = [o.entry_adx_delta for o in group if o.entry_adx_delta is not None]
            range_positions = [o.entry_range_position for o in group if o.entry_range_position is not None]
            # Exploration Analytics fields (Apr 28) — observation-only
            pos_dis = [o.entry_pos_di for o in group if getattr(o, 'entry_pos_di', None) is not None]
            neg_dis = [o.entry_neg_di for o in group if getattr(o, 'entry_neg_di', None) is not None]
            atr_pcts = [o.entry_atr_pct for o in group if getattr(o, 'entry_atr_pct', None) is not None]
            ema50_slopes = [o.entry_ema50_slope for o in group if getattr(o, 'entry_ema50_slope', None) is not None]
            funding_rates = [o.entry_funding_rate for o in group if getattr(o, 'entry_funding_rate', None) is not None]
            # Derived: DI Spread (|+DI - -DI|) per trade, then averaged
            di_spreads = [
                abs(o.entry_pos_di - o.entry_neg_di)
                for o in group
                if getattr(o, 'entry_pos_di', None) is not None and getattr(o, 'entry_neg_di', None) is not None
            ]
            # Derived: EMA50 Alignment counts (Aligned / Opposite / Flat) — flat threshold |slope| < 0.04%
            ema50_aligned = 0
            ema50_opposite = 0
            ema50_flat = 0
            for o in group:
                v = getattr(o, 'entry_ema50_slope', None)
                if v is None:
                    continue
                if abs(v) < 0.04:
                    ema50_flat += 1
                elif (v > 0 and (o.direction or 'LONG') == 'LONG') or (v < 0 and (o.direction or 'LONG') == 'SHORT'):
                    ema50_aligned += 1
                else:
                    ema50_opposite += 1
            # Trade Quality (Apr 28): Time-to-Peak Ratio per group
            ttp_ratios = []
            for o in group:
                r = _compute_ttp_ratio(o)
                if r is not None:
                    ttp_ratios.append(r)
            # BTC Trend Filter gap at entry (May 5) — diagnostic for rollback validation
            btc_trend_gaps = [o.entry_btc_trend_gap_pct for o in group if getattr(o, 'entry_btc_trend_gap_pct', None) is not None]
            # Pair EMA20 vs EMA50 gap at entry (May 5, observation-only)
            pair_ema20_ema50_gaps = [getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) for o in group if getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) is not None]
            # Exit-time BTC trend gap (May 6, pair-side dropped May 7)
            exit_btc_trend_gaps_e = [getattr(o, 'exit_btc_trend_gap_pct', None) for o in group if getattr(o, 'exit_btc_trend_gap_pct', None) is not None]
            # Volume context (May 10 evening) — Global Vol Ratio + Pair 24h $ Volume at entry
            global_vol_ratios = [o.entry_global_volume_ratio for o in group if getattr(o, 'entry_global_volume_ratio', None) is not None]
            pair_vol_usds = [getattr(o, 'entry_pair_volume_24h_usd', None) for o in group if getattr(o, 'entry_pair_volume_24h_usd', None) is not None]
            # Entry Extension / Late Entry Risk (May 13 PM) — direction-aware (positive = late within move)
            ext_vals_ecr = []
            for _o in group:
                _d = getattr(_o, 'entry_dist_from_ema13_pct', None)
                if _d is None:
                    continue
                ext_vals_ecr.append(_d if direction == 'LONG' else -_d)
            # BTC Market Extension / BTC Late Regime Risk (May 14) — same direction-flip rule
            btc_ext_vals_ecr = []
            for _o in group:
                _d = getattr(_o, 'entry_btc_dist_from_ema13_pct', None)
                if _d is None:
                    continue
                btc_ext_vals_ecr.append(_d if direction == 'LONG' else -_d)
            # BTC 1h Slope (May 14) — higher-TF context, raw signed value
            btc_1h_slopes_ecr = [getattr(o, 'entry_btc_1h_slope', None) for o in group if getattr(o, 'entry_btc_1h_slope', None) is not None]
            # Breadth: LONGs use Bull%, SHORTs use Bear%
            if direction == "LONG":
                breadths = [o.entry_bull_pct for o in group if o.entry_bull_pct is not None]
            else:
                breadths = [o.entry_bear_pct for o in group if o.entry_bear_pct is not None]

            peaks = [o.peak_pnl or 0 for o in group]
            troughs = [o.trough_pnl for o in group if o.trough_pnl is not None]
            pnls = [o.pnl_percentage or 0 for o in group]
            total_pnl_usd = sum(o.pnl or 0 for o in group)

            # SL Profile classification — same logic as Stop Loss Deep Dive (line ~3879).
            # For losing trades only (pnl <= 0), classify by peak P&L vs the confidence's
            # be_level1_trigger:
            #   P (positive_no_be)  : 0 < peak < trigger — went green but never armed BE
            #   N (never_positive)  : peak <= 0          — never went green at all
            #   BE (be_was_active)  : peak >= trigger    — BE armed but trade still lost
            # Winners (pnl > 0) are excluded — they're positive by definition.
            # Caveat: classification depends on the configured be_level1_trigger AT REPORT
            # TIME. If triggers are changed across a sample, historical row classifications
            # shift. Currently both VERY_STRONG and STRONG_BUY use the same trigger so this
            # is uniform.
            sl_p_count = 0  # Positive, No BE
            sl_n_count = 0  # Never Positive
            sl_be_count = 0  # BE Was Active (rare under current config)
            for _o in group:
                if (_o.pnl or 0) > 0:
                    continue
                _conf_cfg = _ecr_tc.confidence_levels.get(
                    _o.confidence or "LOW",
                    _ecr_tc.confidence_levels.get("LOW")
                )
                _trigger = _conf_cfg.be_level1_trigger if _conf_cfg else 0.15
                _peak = _o.peak_pnl or 0
                if _peak >= _trigger:
                    sl_be_count += 1
                elif _peak > 0:
                    sl_p_count += 1
                else:
                    sl_n_count += 1

            avg_dur_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in group if o.closed_at and o.opened_at) / count
            hours, remainder = divmod(int(avg_dur_secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            avg_dur_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            entry_conditions_by_reason.append({
                "reason": reason,
                "direction": direction,
                "trades": count,
                "avg_fan_ratio": _avg_fan(group),
                "by_confidence": ec_conf,
                "avg_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_adx": round(sum(adxs) / len(adxs), 1) if adxs else None,
                "adx_rising": adx_rising,
                "adx_falling": adx_falling,
                "avg_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "avg_gap58": round(sum(gaps58) / len(gaps58), 4) if gaps58 else None,
                "avg_gap813": round(sum(gaps813) / len(gaps813), 4) if gaps813 else None,
                "avg_stretch": round(sum(stretches) / len(stretches), 4) if stretches else None,
                "avg_ema20_slope": round(sum(slopes) / len(slopes), 4) if slopes else None,
                "avg_btc_slope": round(sum(btc_slopes) / len(btc_slopes), 4) if btc_slopes else None,
                "avg_btc_rsi": round(sum(btc_rsis) / len(btc_rsis), 1) if btc_rsis else None,
                "avg_btc_adx": round(sum(btc_adxs) / len(btc_adxs), 1) if btc_adxs else None,
                "btc_adx_rising": btc_adx_rising,
                "btc_adx_falling": btc_adx_falling,
                "btc_rsi_rising": btc_rsi_rising,
                "btc_rsi_falling": btc_rsi_falling,
                "rsi_rising": rsi_rising,
                "rsi_falling": rsi_falling,
                "btc_rsi_30m_rising": btc_rsi_30m_rising,
                "btc_rsi_30m_falling": btc_rsi_30m_falling,
                "avg_btc_atr_pct": round(sum(btc_atrs) / len(btc_atrs), 4) if btc_atrs else None,
                "btc_rsi_1h_rising": btc_rsi_1h_rising,
                "btc_rsi_1h_falling": btc_rsi_1h_falling,
                "avg_ema5_dist": round(sum(ema5_dists) / len(ema5_dists), 4) if ema5_dists else None,
                "avg_adx_delta": round(sum(adx_deltas) / len(adx_deltas), 4) if adx_deltas else None,
                "avg_range_position": round(sum(range_positions) / len(range_positions), 1) if range_positions else None,
                "avg_breadth": round(sum(breadths) / len(breadths), 1) if breadths else None,
                # Exploration Analytics (Apr 28)
                "avg_pos_di": round(sum(pos_dis) / len(pos_dis), 1) if pos_dis else None,
                "avg_neg_di": round(sum(neg_dis) / len(neg_dis), 1) if neg_dis else None,
                "avg_di_spread": round(sum(di_spreads) / len(di_spreads), 1) if di_spreads else None,
                "avg_atr_pct": round(sum(atr_pcts) / len(atr_pcts), 4) if atr_pcts else None,
                "avg_ema50_slope": round(sum(ema50_slopes) / len(ema50_slopes), 4) if ema50_slopes else None,
                "ema50_aligned": ema50_aligned,
                "ema50_opposite": ema50_opposite,
                "ema50_flat": ema50_flat,
                "avg_funding_rate": round(sum(funding_rates) / len(funding_rates), 6) if funding_rates else None,
                "avg_ttp_ratio": round(sum(ttp_ratios) / len(ttp_ratios), 2) if ttp_ratios else None,
                # BTC Trend Filter gap at entry (May 5)
                "avg_btc_trend_gap": round(sum(btc_trend_gaps) / len(btc_trend_gaps), 4) if btc_trend_gaps else None,
                # Pair EMA20 vs EMA50 gap at entry (May 5, observation-only)
                "avg_pair_ema20_ema50_gap": round(sum(pair_ema20_ema50_gaps) / len(pair_ema20_ema50_gaps), 4) if pair_ema20_ema50_gaps else None,
                # Exit-time BTC trend gap (May 6, pair-side dropped May 7)
                "avg_exit_btc_trend_gap": round(sum(exit_btc_trend_gaps_e) / len(exit_btc_trend_gaps_e), 4) if exit_btc_trend_gaps_e else None,
                # Volume context (May 10 evening)
                "avg_global_vol_ratio": round(sum(global_vol_ratios) / len(global_vol_ratios), 3) if global_vol_ratios else None,
                "avg_pair_vol_usd": round(sum(pair_vol_usds) / len(pair_vol_usds), 0) if pair_vol_usds else None,
                # Entry Extension / Late Entry Risk (May 13 PM) — direction-aware: positive = late
                "avg_ext_pct": round(sum(ext_vals_ecr) / len(ext_vals_ecr), 4) if ext_vals_ecr else None,
                # BTC Market Extension / BTC Late Regime Risk (May 14)
                "avg_btc_ext_pct": round(sum(btc_ext_vals_ecr) / len(btc_ext_vals_ecr), 4) if btc_ext_vals_ecr else None,
                # BTC 1h EMA20 slope (May 14) — higher-TF context, raw signed
                "avg_btc_1h_slope": round(sum(btc_1h_slopes_ecr) / len(btc_1h_slopes_ecr), 4) if btc_1h_slopes_ecr else None,
                "avg_peak_pct": round(sum(peaks) / count, 4),
                "avg_trough_pct": round(sum(troughs) / len(troughs), 4) if troughs else None,
                "worst_trough_pct": round(min(troughs), 4) if troughs else None,
                "avg_pnl_pct": round(sum(pnls) / count, 4),
                "total_pnl_usd": round(total_pnl_usd, 2),
                "avg_duration": avg_dur_str,
                # SL Profile counts for losing trades — render as "P:n N:n" (BE column
                # included for completeness but typically zero under current config).
                "sl_positive_no_be": sl_p_count,
                "sl_never_positive": sl_n_count,
                "sl_be_was_active": sl_be_count,
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing entry conditions by reason: {e}\n{traceback.format_exc()}")

    # Entry Conditions by Outcome (May 2 — Winner-vs-Loser comparison view).
    # Same column set as Entry Conditions by Close Reason but collapsed to 4 buckets:
    # Winners L, Losers L, Winners S, Losers S. Purpose: enforce the Apr 17
    # methodological rule — when a pattern shows up in losers, compare against
    # winners on the same dimensions before treating it as discriminative.
    # Higher N per row makes cell-level statistics meaningful.
    # "Winner" = pnl > 0 after fees; "Loser" = pnl <= 0.
    entry_conditions_by_outcome = []
    try:
        _eco_tc = config.trading_config
        # Outcome buckets: 'Winners' (closed pnl>0), 'Losers' (closed pnl<=0),
        # 'Aborted' (SIGNAL_EXPIRED rows from Amendment #7 — entries killed at
        # re-validation, never opened). May 2: aborted rows now carry full
        # entry-indicator data so they can be compared against winners/losers
        # on the same dimensions.
        eco_groups = {
            ('LONG', 'Winners'): [],
            ('LONG', 'Losers'): [],
            ('LONG', 'Aborted'): [],
            ('SHORT', 'Winners'): [],
            ('SHORT', 'Losers'): [],
            ('SHORT', 'Aborted'): [],
        }
        for o in orders:
            direction = o.direction or 'LONG'
            if direction not in ('LONG', 'SHORT'):
                continue
            outcome_label = 'Winners' if (o.pnl or 0) > 0 else 'Losers'
            eco_groups[(direction, outcome_label)].append(o)
        # Aborted bucket — signal_expired_orders is loaded earlier in this
        # function (line ~2580) when regime is None. Empty list when
        # per-regime computation is requested.
        for o in (signal_expired_orders or []):
            direction = o.direction or 'LONG'
            if direction not in ('LONG', 'SHORT'):
                continue
            eco_groups[(direction, 'Aborted')].append(o)

        # Order rows: Winners L, Losers L, Aborted L, Winners S, Losers S, Aborted S
        ordered_keys = [
            ('LONG', 'Winners'), ('LONG', 'Losers'), ('LONG', 'Aborted'),
            ('SHORT', 'Winners'), ('SHORT', 'Losers'), ('SHORT', 'Aborted'),
        ]
        for key in ordered_keys:
            group = eco_groups[key]
            count = len(group)
            if count == 0:
                continue
            direction, outcome = key
            is_winner = outcome == 'Winners'

            ec_conf = {}
            for o in group:
                c = o.confidence or "UNKNOWN"
                ec_conf[c] = ec_conf.get(c, 0) + 1

            rsis = [o.entry_rsi for o in group if o.entry_rsi is not None]
            adxs = [o.entry_adx for o in group if o.entry_adx is not None]
            gaps = [o.entry_gap for o in group if o.entry_gap is not None]
            gaps58 = [o.entry_ema_gap_5_8 for o in group if o.entry_ema_gap_5_8 is not None]
            gaps813 = [getattr(o, 'entry_ema_gap_8_13', None) for o in group if getattr(o, 'entry_ema_gap_8_13', None) is not None]
            stretches = [o.entry_ema5_stretch for o in group if o.entry_ema5_stretch is not None]
            slopes = [abs(o.entry_ema20_slope) for o in group if o.entry_ema20_slope is not None]
            btc_slopes = [abs(o.entry_btc_ema20_slope) for o in group if o.entry_btc_ema20_slope is not None]
            btc_adxs = [o.entry_btc_adx for o in group if o.entry_btc_adx is not None]
            btc_rsis = [o.entry_btc_rsi for o in group if o.entry_btc_rsi is not None]

            adx_rising = sum(1 for o in group if o.entry_adx is not None and o.entry_adx_prev is not None and o.entry_adx > o.entry_adx_prev)
            adx_falling = sum(1 for o in group if o.entry_adx is not None and o.entry_adx_prev is not None and o.entry_adx <= o.entry_adx_prev)
            btc_adx_rising = sum(1 for o in group if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None and o.entry_btc_adx > o.entry_btc_adx_prev)
            btc_adx_falling = sum(1 for o in group if o.entry_btc_adx is not None and o.entry_btc_adx_prev is not None and o.entry_btc_adx <= o.entry_btc_adx_prev)
            btc_rsi_rising = sum(1 for o in group if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None and o.entry_btc_rsi > o.entry_btc_rsi_prev)
            btc_rsi_falling = sum(1 for o in group if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None and o.entry_btc_rsi <= o.entry_btc_rsi_prev)
            # May 15: Pair RSI direction (matches RSI Momentum Filter — rsi vs rsi_prev2)
            rsi_rising = sum(1 for o in group if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None and o.entry_rsi > o.entry_rsi_prev)
            rsi_falling = sum(1 for o in group if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None and o.entry_rsi <= o.entry_rsi_prev)
            # May 15: BTC RSI 30min sustained-momentum direction
            btc_rsi_30m_rising = sum(1 for o in group if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None and o.entry_btc_rsi > o.entry_btc_rsi_prev6)
            btc_rsi_30m_falling = sum(1 for o in group if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None and o.entry_btc_rsi <= o.entry_btc_rsi_prev6)
            # May 15 PM: BTC Volatility (ATR%) — avg over the group
            btc_atrs = [getattr(o, 'entry_btc_atr_pct', None) for o in group if getattr(o, 'entry_btc_atr_pct', None) is not None]
            # May 15 PM: BTC 1h RSI Direction (rsi_1h vs rsi_1h_prev)
            btc_rsi_1h_rising = sum(1 for o in group if getattr(o, 'entry_btc_rsi_1h', None) is not None and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None and o.entry_btc_rsi_1h > o.entry_btc_rsi_1h_prev)
            btc_rsi_1h_falling = sum(1 for o in group if getattr(o, 'entry_btc_rsi_1h', None) is not None and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None and o.entry_btc_rsi_1h <= o.entry_btc_rsi_1h_prev)
            ema5_dists = [o.entry_price_vs_ema5_pct for o in group if o.entry_price_vs_ema5_pct is not None]
            adx_deltas = [o.entry_adx_delta for o in group if o.entry_adx_delta is not None]
            range_positions = [o.entry_range_position for o in group if o.entry_range_position is not None]
            pos_dis = [o.entry_pos_di for o in group if getattr(o, 'entry_pos_di', None) is not None]
            neg_dis = [o.entry_neg_di for o in group if getattr(o, 'entry_neg_di', None) is not None]
            atr_pcts = [o.entry_atr_pct for o in group if getattr(o, 'entry_atr_pct', None) is not None]
            ema50_slopes = [o.entry_ema50_slope for o in group if getattr(o, 'entry_ema50_slope', None) is not None]
            funding_rates = [o.entry_funding_rate for o in group if getattr(o, 'entry_funding_rate', None) is not None]
            di_spreads = [
                abs(o.entry_pos_di - o.entry_neg_di)
                for o in group
                if getattr(o, 'entry_pos_di', None) is not None and getattr(o, 'entry_neg_di', None) is not None
            ]
            ema50_aligned = 0
            ema50_opposite = 0
            ema50_flat = 0
            for o in group:
                v = getattr(o, 'entry_ema50_slope', None)
                if v is None:
                    continue
                if abs(v) < 0.04:
                    ema50_flat += 1
                elif (v > 0 and direction == 'LONG') or (v < 0 and direction == 'SHORT'):
                    ema50_aligned += 1
                else:
                    ema50_opposite += 1
            ttp_ratios = []
            for o in group:
                r = _compute_ttp_ratio(o)
                if r is not None:
                    ttp_ratios.append(r)
            # BTC Trend Filter gap at entry (May 5) — diagnostic for rollback validation
            btc_trend_gaps = [o.entry_btc_trend_gap_pct for o in group if getattr(o, 'entry_btc_trend_gap_pct', None) is not None]
            # Pair EMA20 vs EMA50 gap at entry (May 5, observation-only)
            pair_ema20_ema50_gaps = [getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) for o in group if getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) is not None]
            # Exit-time BTC trend gap (May 6, pair-side dropped May 7)
            exit_btc_trend_gaps_e = [getattr(o, 'exit_btc_trend_gap_pct', None) for o in group if getattr(o, 'exit_btc_trend_gap_pct', None) is not None]
            # Volume context (May 10 evening)
            global_vol_ratios = [o.entry_global_volume_ratio for o in group if getattr(o, 'entry_global_volume_ratio', None) is not None]
            pair_vol_usds = [getattr(o, 'entry_pair_volume_24h_usd', None) for o in group if getattr(o, 'entry_pair_volume_24h_usd', None) is not None]
            # Entry Extension / Late Entry Risk (May 13 PM) — direction-aware (positive = late within move)
            ext_vals_eco = []
            for _o in group:
                _d = getattr(_o, 'entry_dist_from_ema13_pct', None)
                if _d is None:
                    continue
                ext_vals_eco.append(_d if direction == 'LONG' else -_d)
            # BTC Market Extension / BTC Late Regime Risk (May 14)
            btc_ext_vals_eco = []
            for _o in group:
                _d = getattr(_o, 'entry_btc_dist_from_ema13_pct', None)
                if _d is None:
                    continue
                btc_ext_vals_eco.append(_d if direction == 'LONG' else -_d)
            # BTC 1h Slope (May 14)
            btc_1h_slopes_eco = [getattr(o, 'entry_btc_1h_slope', None) for o in group if getattr(o, 'entry_btc_1h_slope', None) is not None]
            if direction == "LONG":
                breadths = [o.entry_bull_pct for o in group if o.entry_bull_pct is not None]
            else:
                breadths = [o.entry_bear_pct for o in group if o.entry_bear_pct is not None]

            peaks = [o.peak_pnl or 0 for o in group]
            troughs = [o.trough_pnl for o in group if o.trough_pnl is not None]
            pnls = [o.pnl_percentage or 0 for o in group]
            total_pnl_usd = sum(o.pnl or 0 for o in group)

            # SL Profile only meaningful for losers — same classification rule as
            # Stop Loss Deep Dive / Entry Conditions by Close Reason.
            sl_p_count = 0
            sl_n_count = 0
            sl_be_count = 0
            # SL Profile only meaningful for Losers (closed at a loss).
            # Aborted rows never opened — peak/trough are always 0, classification
            # would be misleading. Winners are positive by definition.
            if outcome == 'Losers':
                for _o in group:
                    if (_o.pnl or 0) > 0:
                        continue
                    _conf_cfg = _eco_tc.confidence_levels.get(
                        _o.confidence or "LOW",
                        _eco_tc.confidence_levels.get("LOW")
                    )
                    _trigger = _conf_cfg.be_level1_trigger if _conf_cfg else 0.15
                    _peak = _o.peak_pnl or 0
                    if _peak >= _trigger:
                        sl_be_count += 1
                    elif _peak > 0:
                        sl_p_count += 1
                    else:
                        sl_n_count += 1

            avg_dur_secs = sum((o.closed_at - o.opened_at).total_seconds() for o in group if o.closed_at and o.opened_at) / count
            hours, remainder = divmod(int(avg_dur_secs), 3600)
            minutes, seconds = divmod(remainder, 60)
            avg_dur_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            entry_conditions_by_outcome.append({
                "outcome": outcome,
                "direction": direction,
                "trades": count,
                "avg_fan_ratio": _avg_fan(group),
                "by_confidence": ec_conf,
                "avg_rsi": round(sum(rsis) / len(rsis), 1) if rsis else None,
                "avg_adx": round(sum(adxs) / len(adxs), 1) if adxs else None,
                "adx_rising": adx_rising,
                "adx_falling": adx_falling,
                "avg_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
                "avg_gap58": round(sum(gaps58) / len(gaps58), 4) if gaps58 else None,
                "avg_gap813": round(sum(gaps813) / len(gaps813), 4) if gaps813 else None,
                "avg_stretch": round(sum(stretches) / len(stretches), 4) if stretches else None,
                "avg_ema20_slope": round(sum(slopes) / len(slopes), 4) if slopes else None,
                "avg_btc_slope": round(sum(btc_slopes) / len(btc_slopes), 4) if btc_slopes else None,
                "avg_btc_rsi": round(sum(btc_rsis) / len(btc_rsis), 1) if btc_rsis else None,
                "avg_btc_adx": round(sum(btc_adxs) / len(btc_adxs), 1) if btc_adxs else None,
                "btc_adx_rising": btc_adx_rising,
                "btc_adx_falling": btc_adx_falling,
                "btc_rsi_rising": btc_rsi_rising,
                "btc_rsi_falling": btc_rsi_falling,
                "rsi_rising": rsi_rising,
                "rsi_falling": rsi_falling,
                "btc_rsi_30m_rising": btc_rsi_30m_rising,
                "btc_rsi_30m_falling": btc_rsi_30m_falling,
                "avg_btc_atr_pct": round(sum(btc_atrs) / len(btc_atrs), 4) if btc_atrs else None,
                "btc_rsi_1h_rising": btc_rsi_1h_rising,
                "btc_rsi_1h_falling": btc_rsi_1h_falling,
                "avg_ema5_dist": round(sum(ema5_dists) / len(ema5_dists), 4) if ema5_dists else None,
                "avg_adx_delta": round(sum(adx_deltas) / len(adx_deltas), 4) if adx_deltas else None,
                "avg_range_position": round(sum(range_positions) / len(range_positions), 1) if range_positions else None,
                "avg_breadth": round(sum(breadths) / len(breadths), 1) if breadths else None,
                "avg_pos_di": round(sum(pos_dis) / len(pos_dis), 1) if pos_dis else None,
                "avg_neg_di": round(sum(neg_dis) / len(neg_dis), 1) if neg_dis else None,
                "avg_di_spread": round(sum(di_spreads) / len(di_spreads), 1) if di_spreads else None,
                "avg_atr_pct": round(sum(atr_pcts) / len(atr_pcts), 4) if atr_pcts else None,
                "avg_ema50_slope": round(sum(ema50_slopes) / len(ema50_slopes), 4) if ema50_slopes else None,
                "ema50_aligned": ema50_aligned,
                "ema50_opposite": ema50_opposite,
                "ema50_flat": ema50_flat,
                "avg_funding_rate": round(sum(funding_rates) / len(funding_rates), 6) if funding_rates else None,
                "avg_ttp_ratio": round(sum(ttp_ratios) / len(ttp_ratios), 2) if ttp_ratios else None,
                # BTC Trend Filter gap at entry (May 5)
                "avg_btc_trend_gap": round(sum(btc_trend_gaps) / len(btc_trend_gaps), 4) if btc_trend_gaps else None,
                # Pair EMA20 vs EMA50 gap at entry (May 5, observation-only)
                "avg_pair_ema20_ema50_gap": round(sum(pair_ema20_ema50_gaps) / len(pair_ema20_ema50_gaps), 4) if pair_ema20_ema50_gaps else None,
                # Exit-time BTC trend gap (May 6, pair-side dropped May 7)
                "avg_exit_btc_trend_gap": round(sum(exit_btc_trend_gaps_e) / len(exit_btc_trend_gaps_e), 4) if exit_btc_trend_gaps_e else None,
                # Volume context (May 10 evening)
                "avg_global_vol_ratio": round(sum(global_vol_ratios) / len(global_vol_ratios), 3) if global_vol_ratios else None,
                "avg_pair_vol_usd": round(sum(pair_vol_usds) / len(pair_vol_usds), 0) if pair_vol_usds else None,
                # Entry Extension / Late Entry Risk (May 13 PM) — direction-aware: positive = late
                "avg_ext_pct": round(sum(ext_vals_eco) / len(ext_vals_eco), 4) if ext_vals_eco else None,
                # BTC Market Extension / BTC Late Regime Risk (May 14)
                "avg_btc_ext_pct": round(sum(btc_ext_vals_eco) / len(btc_ext_vals_eco), 4) if btc_ext_vals_eco else None,
                # BTC 1h EMA20 slope (May 14) — higher-TF context, raw signed
                "avg_btc_1h_slope": round(sum(btc_1h_slopes_eco) / len(btc_1h_slopes_eco), 4) if btc_1h_slopes_eco else None,
                "avg_peak_pct": round(sum(peaks) / count, 4),
                "avg_trough_pct": round(sum(troughs) / len(troughs), 4) if troughs else None,
                "worst_trough_pct": round(min(troughs), 4) if troughs else None,
                "avg_pnl_pct": round(sum(pnls) / count, 4),
                "total_pnl_usd": round(total_pnl_usd, 2),
                "avg_duration": avg_dur_str,
                "sl_positive_no_be": sl_p_count,
                "sl_never_positive": sl_n_count,
                "sl_be_was_active": sl_be_count,
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing entry conditions by outcome: {e}\n{traceback.format_exc()}")
        entry_conditions_by_outcome = []

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
        # Close-reason prefixes that represent losing exits for the SL Deep Dive.
        # The outer `pnl <= 0` filter ensures only losers are counted, so adding
        # reasons that can close in profit (TRAILING_STOP, FL_RECOVERED) is safe —
        # those trades are filtered by pnl anyway. We include both the direct and
        # FL_-prefixed variants of each reason. Reasons without a non-FL variant
        # (DEEP_STOP, EMERGENCY_SL, NO_EXPANSION) are covered by the FL_ branch;
        # the direct branch is a harmless no-op for those.
        _sl_reason_prefixes = [
            "STOP_LOSS", "MOMENTUM_EXIT", "PNL_TRAILING", "SLOPE_EXIT",
            "SIGNAL_LOST", "BREAKEVEN_EXIT",
            "DEEP_STOP",          # FL_DEEP_STOP at -1% — modern stop-loss for flagged trades
            "EMERGENCY_SL",       # FL1 WIDE_SL backstop at -1.2%
            "RSI_MOMENTUM_EXIT",  # RSI faded, trade cut on losing side
            "NO_EXPANSION",       # Trade timed out without profit expansion
            "REGIME_CHANGE",      # BTC regime flipped against the trade
            "MAX_HOLD_TIME",      # Hit max duration (losing side only via pnl<=0 filter)
            # May 8: added EMA13_CROSS_EXIT and EMA_STACK_CROSS_EXIT — these became
            # the dominant losing-exit mechanism but were missing from this whitelist,
            # leaving Stop Loss Deep Dive structurally blank when most losers cut via
            # EMA13 cross before hitting -0.9% SL.
            "EMA13_CROSS_EXIT",
            "EMA_STACK_CROSS_EXIT",
        ]
        _cr_match = lambda cr: any(cr.startswith(p) or cr.startswith(f"FL_{p}") for p in _sl_reason_prefixes)
        sl_orders = [o for o in orders if o.close_reason and (o.pnl or 0) <= 0 and _cr_match(o.close_reason)]
        
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
            # May 17: post-arm minimum P&L (BE-floor counterfactual)
            post_arm_mins = [getattr(o, 'post_arm_min_pnl_pct', None) for o in group_orders if getattr(o, 'post_arm_min_pnl_pct', None) is not None]
            # May 22: ATR aggregation for sl_atr_multiplier diagnostic.
            atrs_sl = [o.entry_atr_pct for o in group_orders if o.entry_atr_pct is not None and o.entry_atr_pct > 0]
            avg_atr_sl = sum(atrs_sl) / len(atrs_sl) if atrs_sl else None
            atr_sl_15x_sl = -(avg_atr_sl * 1.5) if avg_atr_sl is not None else None
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
                "signal_inactive": sig_inactive,
                "avg_post_arm_min_pct": round(sum(post_arm_mins) / len(post_arm_mins), 4) if post_arm_mins else None,
                "post_arm_min_n": len(post_arm_mins),
                "avg_atr_pct": round(avg_atr_sl, 4) if avg_atr_sl is not None else None,
                "atr_sl_15x": round(atr_sl_15x_sl, 4) if atr_sl_15x_sl is not None else None,
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
            "all_signal_inactive": sum(1 for o in sl_orders if o.signal_active_at_close is False),
            # May 18: post-arm-min aggregate for BE-floor counterfactual context
            "all_avg_post_arm_min_pct": round(sum(p for p in (getattr(o, 'post_arm_min_pnl_pct', None) for o in sl_orders) if p is not None) / max(1, sum(1 for o in sl_orders if getattr(o, 'post_arm_min_pnl_pct', None) is not None)), 4) if any(getattr(o, 'post_arm_min_pnl_pct', None) is not None for o in sl_orders) else None,
            "all_post_arm_min_n": sum(1 for o in sl_orders if getattr(o, 'post_arm_min_pnl_pct', None) is not None),
        }
        
        winning_orders = [o for o in orders if o.pnl and o.pnl > 0]
        win_by_reason = {}
        for o in winning_orders:
            reason = o.close_reason or "UNKNOWN"
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
            # May 17: post-arm minimum (BE-floor counterfactual)
            post_arm_mins = [getattr(o, 'post_arm_min_pnl_pct', None) for o in group if getattr(o, 'post_arm_min_pnl_pct', None) is not None]
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
                "avg_post_exit_trough_pnl": round(sum(post_exit_troughs) / len(post_exit_troughs), 4) if post_exit_troughs else None,
                "avg_post_arm_min_pct": round(sum(post_arm_mins) / len(post_arm_mins), 4) if post_arm_mins else None,
                "post_arm_min_n": len(post_arm_mins),
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
            ("15-18", 15, 18), ("18-22", 18, 22), ("22-25", 22, 25),
            ("25-28", 25, 28), ("28-30", 28, 30), ("30-33", 30, 33), ("33-35", 33, 35), ("35+", 35, 999),
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
                    ct_pnl_pct = sum(o.pnl_percentage or 0 for o in bucket)
                    # May 2: Never Positive count per cell — distinguishes
                    # bad-entry cells (high NP%) from bad-exit cells (low NP%, lots of losers).
                    ct_never_positive = sum(1 for o in bucket if (o.peak_pnl or 0) <= 0)
                    rsi_adx_crosstab.append({
                        "direction": direction,
                        "rsi_range": rsi_name,
                        "adx_range": adx_name,
                        "trades": ct_count,
                        "win_rate": round(ct_wins / ct_count * 100, 1),
                        "total_pnl": round(ct_pnl, 2),
                        "avg_pnl": round(ct_pnl / ct_count, 2),
                        "avg_pnl_pct": round(ct_pnl_pct / ct_count, 4),
                        "never_positive": ct_never_positive,
                        "never_positive_pct": round(ct_never_positive / ct_count * 100, 1) if ct_count else 0,
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
                gaps813 = [getattr(o, 'entry_ema_gap_8_13', None) for o in bucket_orders if getattr(o, 'entry_ema_gap_8_13', None) is not None]
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
                    "avg_entry_gap813": round(sum(gaps813) / len(gaps813), 4) if gaps813 else None,
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

            np_adx_ranges = [("15-18", 15, 18), ("18-22", 18, 22), ("22-25", 22, 25), ("25-28", 25, 28), ("28-30", 28, 30), ("30-33", 30, 33), ("33-35", 33, 35), ("35+", 35, 999)]
            np_adx_trades = [o for o in np_trades if o.entry_adx is not None]
            all_adx_trades = [o for o in orders if o.entry_adx is not None]
            for rng, lo, hi in np_adx_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_adx_trades if lo <= o.entry_adx < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_adx_trades if lo <= o.entry_adx < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "ADX", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Entry Extension / Late Entry Risk (May 13 PM) — direction-aware buckets.
            # extension = entry_dist_from_ema13_pct for LONG, -entry_dist_from_ema13_pct for SHORT
            # so "positive extension = late within the move" for both sides.
            np_ext_ranges = [
                ("< -0.20%", -99, -0.20),
                ("-0.20 to -0.10%", -0.20, -0.10),
                ("-0.10 to 0%", -0.10, 0.0),
                ("0 to +0.10%", 0.0, 0.10),
                ("+0.10 to +0.20%", 0.10, 0.20),
                ("+0.20 to +0.40%", 0.20, 0.40),
                ("+0.40 to +0.60%", 0.40, 0.60),
                ("> +0.60%", 0.60, 99),
            ]
            np_ext_trades = [o for o in np_trades if getattr(o, 'entry_dist_from_ema13_pct', None) is not None]
            all_ext_trades = [o for o in orders if getattr(o, 'entry_dist_from_ema13_pct', None) is not None]
            for rng, lo, hi in np_ext_ranges:
                for direction in ["LONG", "SHORT"]:
                    def _ext_of(o, d=direction):
                        v = o.entry_dist_from_ema13_pct
                        return v if d == "LONG" else -v
                    bucket = [o for o in np_ext_trades if (o.direction or "LONG") == direction and lo <= _ext_of(o) < hi]
                    all_in = len([o for o in all_ext_trades if (o.direction or "LONG") == direction and lo <= _ext_of(o) < hi])
                    row = _np_bucket_stats(bucket, "Extension", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # BTC Market Extension / BTC Late Regime Risk (May 14) — same direction-aware
            # bucketing as pair extension. Positive = BTC stretched within the move.
            np_btc_ext_ranges = np_ext_ranges  # share bucket definitions
            np_btc_ext_trades = [o for o in np_trades if getattr(o, 'entry_btc_dist_from_ema13_pct', None) is not None]
            all_btc_ext_trades = [o for o in orders if getattr(o, 'entry_btc_dist_from_ema13_pct', None) is not None]
            for rng, lo, hi in np_btc_ext_ranges:
                for direction in ["LONG", "SHORT"]:
                    def _btc_ext_of(o, d=direction):
                        v = o.entry_btc_dist_from_ema13_pct
                        return v if d == "LONG" else -v
                    bucket = [o for o in np_btc_ext_trades if (o.direction or "LONG") == direction and lo <= _btc_ext_of(o) < hi]
                    all_in = len([o for o in all_btc_ext_trades if (o.direction or "LONG") == direction and lo <= _btc_ext_of(o) < hi])
                    row = _np_bucket_stats(bucket, "BTC Ext", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_rsi_ranges = [("20-30", 20, 30), ("30-35", 30, 35), ("35-40", 35, 40), ("40-45", 40, 45), ("45-50", 45, 50), ("50-55", 50, 55), ("55-60", 55, 60), ("60-65", 60, 65), ("65-70", 65, 70), ("70+", 70, 999)]
            np_rsi_trades = [o for o in np_trades if o.entry_rsi is not None]
            all_rsi_trades = [o for o in orders if o.entry_rsi is not None]
            for rng, lo, hi in np_rsi_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_rsi_trades if lo <= o.entry_rsi < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_rsi_trades if lo <= o.entry_rsi < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "RSI", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_gap58_ranges = [
                ("0.00-0.02%", 0.00, 0.02), ("0.02-0.04%", 0.02, 0.04), ("0.04-0.06%", 0.04, 0.06),
                ("0.06-0.08%", 0.06, 0.08), ("0.08-0.10%", 0.08, 0.10), ("0.10-0.12%", 0.10, 0.12),
                ("0.12-0.14%", 0.12, 0.14), ("0.14-0.16%", 0.14, 0.16), ("0.16-0.18%", 0.16, 0.18),
                ("0.18-0.20%", 0.18, 0.20), (">0.20%", 0.20, 999)
            ]
            np_gap58_trades = [o for o in np_trades if o.entry_ema_gap_5_8 is not None]
            all_gap58_trades = [o for o in orders if o.entry_ema_gap_5_8 is not None]
            for rng, lo, hi in np_gap58_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_gap58_trades if lo <= o.entry_ema_gap_5_8 < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_gap58_trades if lo <= o.entry_ema_gap_5_8 < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "EMA5-EMA8 Gap", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # EMA Fan Acceleration (May 28) — fan_ratio = 5-8 gap / 8-13 gap. Buckets match
            # the standalone Fan Acceleration table (recentered on ~0.6 steady). Post-May-27 only.
            def _np_fan_of(o):
                if o.entry_ema_gap_5_8 is None:
                    return None
                g813 = getattr(o, 'entry_ema_gap_8_13', None)
                if not g813 or g813 <= 0:
                    return None
                return o.entry_ema_gap_5_8 / g813
            # May 29: aligned with the fan_ratio filter dead-zone buckets (see main fan table);
            # >2.00 tail split into 2-3 / 3-5 / >5 to locate the upper-edge tightening point.
            np_fan_ranges = [
                ("<0.85", -999, 0.85),
                ("0.85-1.00", 0.85, 1.00),
                ("1.00-1.35", 1.00, 1.35),
                ("1.35-1.65", 1.35, 1.65),
                ("1.65-2.00", 1.65, 2.00),
                ("2.00-3.00", 2.00, 3.00),
                ("3.00-5.00", 3.00, 5.00),
                (">5.00", 5.00, 999),
            ]
            np_fan_trades = [o for o in np_trades if _np_fan_of(o) is not None]
            all_fan_trades = [o for o in orders if _np_fan_of(o) is not None]
            for rng, lo, hi in np_fan_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_fan_trades if (o.direction or "LONG") == direction and lo <= _np_fan_of(o) < hi]
                    all_in = len([o for o in all_fan_trades if (o.direction or "LONG") == direction and lo <= _np_fan_of(o) < hi])
                    row = _np_bucket_stats(bucket, "Fan Accel", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_gap_ranges = [("<0.15%", 0.00, 0.15), ("0.15-0.25%", 0.15, 0.25), ("0.25-0.35%", 0.25, 0.35), ("0.35-0.50%", 0.35, 0.50), ("0.50-0.80%", 0.50, 0.80), (">0.80%", 0.80, 999)]
            np_gap_trades = [o for o in np_trades if o.entry_gap is not None]
            all_gap_trades = [o for o in orders if o.entry_gap is not None]
            for rng, lo, hi in np_gap_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_gap_trades if lo <= o.entry_gap < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_gap_trades if lo <= o.entry_gap < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Entry Gap", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_stretch_ranges = [("<0.04%", 0.00, 0.04), ("0.04-0.08%", 0.04, 0.08), ("0.08-0.12%", 0.08, 0.12), ("0.12-0.16%", 0.12, 0.16), ("0.16-0.20%", 0.16, 0.20), ("0.20-0.25%", 0.20, 0.25), ("0.25-0.30%", 0.25, 0.30), (">0.30%", 0.30, 999)]
            np_stretch_trades = [o for o in np_trades if o.entry_ema5_stretch is not None]
            all_stretch_trades = [o for o in orders if o.entry_ema5_stretch is not None]
            for rng, lo, hi in np_stretch_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_stretch_trades if lo <= o.entry_ema5_stretch < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_stretch_trades if lo <= o.entry_ema5_stretch < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "EMA5 Stretch", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # May 12: signed buckets, matching Performance by Pair/BTC EMA20 Slope tables.
            np_slope_ranges = [
                ("< -0.60%", -999, -0.60),
                ("-0.60 to -0.40%", -0.60, -0.40),
                ("-0.40 to -0.30%", -0.40, -0.30),
                ("-0.30 to -0.25%", -0.30, -0.25),
                ("-0.25 to -0.20%", -0.25, -0.20),
                ("-0.20 to -0.18%", -0.20, -0.18),
                ("-0.18 to -0.16%", -0.18, -0.16),
                ("-0.16 to -0.14%", -0.16, -0.14),
                ("-0.14 to -0.12%", -0.14, -0.12),
                ("-0.12 to -0.10%", -0.12, -0.10),
                ("-0.10 to -0.08%", -0.10, -0.08),
                ("-0.08 to -0.06%", -0.08, -0.06),
                ("-0.06 to -0.04%", -0.06, -0.04),
                ("-0.04 to -0.03%", -0.04, -0.03),
                ("-0.03 to -0.02%", -0.03, -0.02),
                ("-0.02 to -0.01%", -0.02, -0.01),
                ("-0.01 to 0%", -0.01, 0.0),
                ("0 to +0.01%", 0.0, 0.01),
                ("+0.01 to +0.02%", 0.01, 0.02),
                ("+0.02 to +0.03%", 0.02, 0.03),
                ("+0.03 to +0.04%", 0.03, 0.04),
                ("+0.04 to +0.06%", 0.04, 0.06),
                ("+0.06 to +0.08%", 0.06, 0.08),
                ("+0.08 to +0.10%", 0.08, 0.10),
                ("+0.10 to +0.12%", 0.10, 0.12),
                ("+0.12 to +0.14%", 0.12, 0.14),
                ("+0.14 to +0.16%", 0.14, 0.16),
                ("+0.16 to +0.18%", 0.16, 0.18),
                ("+0.18 to +0.20%", 0.18, 0.20),
                ("+0.20 to +0.25%", 0.20, 0.25),
                ("+0.25 to +0.30%", 0.25, 0.30),
                ("+0.30 to +0.40%", 0.30, 0.40),
                ("+0.40 to +0.60%", 0.40, 0.60),
                ("> +0.60%", 0.60, 999),
            ]
            np_slope_trades = [o for o in np_trades if o.entry_ema20_slope is not None]
            all_slope_trades = [o for o in orders if o.entry_ema20_slope is not None]
            for rng, lo, hi in np_slope_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_slope_trades if lo <= o.entry_ema20_slope < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_slope_trades if lo <= o.entry_ema20_slope < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "EMA20 Slope", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # May 12: BTC EMA20 Slope NP deep dive uses same signed buckets as Pair.
            np_btc_slope_ranges = np_slope_ranges
            np_btc_slope_trades = [o for o in np_trades if o.entry_btc_ema20_slope is not None]
            all_btc_slope_trades = [o for o in orders if o.entry_btc_ema20_slope is not None]
            for rng, lo, hi in np_btc_slope_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_slope_trades if lo <= o.entry_btc_ema20_slope < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_slope_trades if lo <= o.entry_btc_ema20_slope < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC EMA20 Slope", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_btc_adx_ranges = [("10-15", 10, 15), ("15-20", 15, 20), ("20-25", 20, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35+", 35, 999)]
            np_btc_adx_trades = [o for o in np_trades if o.entry_btc_adx is not None]
            all_btc_adx_trades = [o for o in orders if o.entry_btc_adx is not None]
            for rng, lo, hi in np_btc_adx_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_adx_trades if lo <= o.entry_btc_adx < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_adx_trades if lo <= o.entry_btc_adx < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC ADX", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            np_btc_rsi_ranges = [("<20", 0, 20), ("20-25", 20, 25), ("25-30", 25, 30), ("30-35", 30, 35), ("35-40", 35, 40), ("40-45", 40, 45), ("45-50", 45, 50), ("50-55", 50, 55), ("55-60", 55, 60), ("60-65", 60, 65), ("65-70", 65, 70), ("70+", 70, 999)]
            np_btc_rsi_trades = [o for o in np_trades if o.entry_btc_rsi is not None]
            all_btc_rsi_trades = [o for o in orders if o.entry_btc_rsi is not None]
            for rng, lo, hi in np_btc_rsi_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_rsi_trades if lo <= o.entry_btc_rsi < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_rsi_trades if lo <= o.entry_btc_rsi < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC RSI", rng, direction, all_in)
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

            np_adx_delta_ranges = [
                ("< -2.0", -999, -2.0), ("-2.0 to -1.0", -2.0, -1.0), ("-1.0 to -0.5", -1.0, -0.5),
                ("-0.5 to -0.3", -0.5, -0.3), ("-0.3 to -0.1", -0.3, -0.1), ("-0.1 to 0.0", -0.1, 0.0),
                ("0.0 to 0.05", 0.0, 0.05), ("0.05 to 0.1", 0.05, 0.1), ("0.1 to 0.3", 0.1, 0.3),
                ("0.3 to 0.5", 0.3, 0.5), ("0.5 to 1.0", 0.5, 1.0), ("1.0 to 2.0", 1.0, 2.0), ("> 2.0", 2.0, 999),
            ]
            np_adx_delta_trades = [o for o in np_trades if o.entry_adx_delta is not None]
            all_adx_delta_trades = [o for o in orders if o.entry_adx_delta is not None]
            for rng, lo, hi in np_adx_delta_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_adx_delta_trades if lo <= o.entry_adx_delta < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_adx_delta_trades if lo <= o.entry_adx_delta < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "ADX Delta", rng, direction, all_in)
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

            # Pair RSI Direction (May 15) — matches RSI Momentum Filter comparison (rsi vs rsi_prev2)
            for np_rsi_dir_label in ["Rising", "Falling"]:
                np_rsi_dir_trades = [o for o in np_trades if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None]
                all_rsi_dir_trades = [o for o in orders if o.entry_rsi is not None and getattr(o, 'entry_rsi_prev', None) is not None]
                if np_rsi_dir_label == "Rising":
                    np_rsi_dir_pool = [o for o in np_rsi_dir_trades if o.entry_rsi > o.entry_rsi_prev]
                    all_rsi_dir_pool = [o for o in all_rsi_dir_trades if o.entry_rsi > o.entry_rsi_prev]
                else:
                    np_rsi_dir_pool = [o for o in np_rsi_dir_trades if o.entry_rsi <= o.entry_rsi_prev]
                    all_rsi_dir_pool = [o for o in all_rsi_dir_trades if o.entry_rsi <= o.entry_rsi_prev]
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_rsi_dir_pool if (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_rsi_dir_pool if (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "RSI Direction", np_rsi_dir_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # BTC RSI Direction (May 15) — uses entry_btc_rsi vs entry_btc_rsi_prev (1 candle)
            for np_btc_rsi_dir_label in ["Rising", "Falling"]:
                np_btc_rsi_dir_trades = [o for o in np_trades if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None]
                all_btc_rsi_dir_trades = [o for o in orders if o.entry_btc_rsi is not None and o.entry_btc_rsi_prev is not None]
                if np_btc_rsi_dir_label == "Rising":
                    np_btc_rsi_dir_pool = [o for o in np_btc_rsi_dir_trades if o.entry_btc_rsi > o.entry_btc_rsi_prev]
                    all_btc_rsi_dir_pool = [o for o in all_btc_rsi_dir_trades if o.entry_btc_rsi > o.entry_btc_rsi_prev]
                else:
                    np_btc_rsi_dir_pool = [o for o in np_btc_rsi_dir_trades if o.entry_btc_rsi <= o.entry_btc_rsi_prev]
                    all_btc_rsi_dir_pool = [o for o in all_btc_rsi_dir_trades if o.entry_btc_rsi <= o.entry_btc_rsi_prev]
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_rsi_dir_pool if (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_rsi_dir_pool if (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC RSI Direction", np_btc_rsi_dir_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # BTC RSI Direction 30min (May 15) — sustained-momentum window (vs btc_rsi_prev6)
            for np_btc_rsi30_dir_label in ["Rising", "Falling"]:
                np_btc_rsi30_dir_trades = [o for o in np_trades if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None]
                all_btc_rsi30_dir_trades = [o for o in orders if o.entry_btc_rsi is not None and getattr(o, 'entry_btc_rsi_prev6', None) is not None]
                if np_btc_rsi30_dir_label == "Rising":
                    np_btc_rsi30_dir_pool = [o for o in np_btc_rsi30_dir_trades if o.entry_btc_rsi > o.entry_btc_rsi_prev6]
                    all_btc_rsi30_dir_pool = [o for o in all_btc_rsi30_dir_trades if o.entry_btc_rsi > o.entry_btc_rsi_prev6]
                else:
                    np_btc_rsi30_dir_pool = [o for o in np_btc_rsi30_dir_trades if o.entry_btc_rsi <= o.entry_btc_rsi_prev6]
                    all_btc_rsi30_dir_pool = [o for o in all_btc_rsi30_dir_trades if o.entry_btc_rsi <= o.entry_btc_rsi_prev6]
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btc_rsi30_dir_pool if (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btc_rsi30_dir_pool if (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC RSI Direction 30m", np_btc_rsi30_dir_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # BTC 1h RSI Direction (May 15 PM) — 1h timeframe sustained momentum
            for np_btc_rsi1h_dir_label in ["Rising", "Falling"]:
                np_b1h_dir_trades = [o for o in np_trades if getattr(o, 'entry_btc_rsi_1h', None) is not None and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None]
                all_b1h_dir_trades = [o for o in orders if getattr(o, 'entry_btc_rsi_1h', None) is not None and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None]
                if np_btc_rsi1h_dir_label == "Rising":
                    np_b1h_pool = [o for o in np_b1h_dir_trades if o.entry_btc_rsi_1h > o.entry_btc_rsi_1h_prev]
                    all_b1h_pool = [o for o in all_b1h_dir_trades if o.entry_btc_rsi_1h > o.entry_btc_rsi_1h_prev]
                else:
                    np_b1h_pool = [o for o in np_b1h_dir_trades if o.entry_btc_rsi_1h <= o.entry_btc_rsi_1h_prev]
                    all_b1h_pool = [o for o in all_b1h_dir_trades if o.entry_btc_rsi_1h <= o.entry_btc_rsi_1h_prev]
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_b1h_pool if (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_b1h_pool if (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC 1h RSI Direction", np_btc_rsi1h_dir_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # BTC Volatility (May 15 PM) — ATR/price × 100 buckets
            _np_btc_vol_bins = [
                ("< 0.10%", 0.0, 0.10),
                ("0.10 - 0.15%", 0.10, 0.15),
                ("0.15 - 0.20%", 0.15, 0.20),
                ("0.20 - 0.30%", 0.20, 0.30),
                ("0.30 - 0.45%", 0.30, 0.45),
                ("> 0.45%", 0.45, 999.0),
            ]
            for direction in ["LONG", "SHORT"]:
                dir_np_vol = [o for o in np_trades if (o.direction or "LONG") == direction]
                dir_all_vol = [o for o in orders if (o.direction or "LONG") == direction]
                for v_label, v_lo, v_hi in _np_btc_vol_bins:
                    bucket = [o for o in dir_np_vol if getattr(o, 'entry_btc_atr_pct', None) is not None and v_lo <= o.entry_btc_atr_pct < v_hi]
                    all_in = len([o for o in dir_all_vol if getattr(o, 'entry_btc_atr_pct', None) is not None and v_lo <= o.entry_btc_atr_pct < v_hi])
                    row = _np_bucket_stats(bucket, "BTC Volatility", v_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Market Breadth at Entry — LONGs binned by Bull%, SHORTs by Bear%
            for direction in ["LONG", "SHORT"]:
                dir_np = [o for o in np_trades if (o.direction or "LONG") == direction]
                dir_all = [o for o in orders if (o.direction or "LONG") == direction]
                for b_label, b_lo, b_hi in _BREADTH_BINS:
                    if direction == "LONG":
                        bucket = [o for o in dir_np if o.entry_bull_pct is not None and b_lo <= o.entry_bull_pct < b_hi]
                        all_in = len([o for o in dir_all if o.entry_bull_pct is not None and b_lo <= o.entry_bull_pct < b_hi])
                    else:
                        bucket = [o for o in dir_np if o.entry_bear_pct is not None and b_lo <= o.entry_bear_pct < b_hi]
                        all_in = len([o for o in dir_all if o.entry_bear_pct is not None and b_lo <= o.entry_bear_pct < b_hi])
                    row = _np_bucket_stats(bucket, "Breadth", b_label, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Pair EMA13-EMA50 Gap (signed) — May 12: synced to 24-bucket scheme
            # matching Performance by Pair EMA13-EMA50 Gap for apples-to-apples
            # cross-table comparison.
            np_pgap_ranges = [
                ("< -1.00%", -999, -1.00),
                ("-1.00 to -0.80%", -1.00, -0.80),
                ("-0.80 to -0.60%", -0.80, -0.60),
                ("-0.60 to -0.50%", -0.60, -0.50),
                ("-0.50 to -0.40%", -0.50, -0.40),
                ("-0.40 to -0.30%", -0.40, -0.30),
                ("-0.30 to -0.25%", -0.30, -0.25),
                ("-0.25 to -0.20%", -0.25, -0.20),
                ("-0.20 to -0.15%", -0.20, -0.15),
                ("-0.15 to -0.10%", -0.15, -0.10),
                ("-0.10 to -0.05%", -0.10, -0.05),
                ("-0.05 to 0%", -0.05, 0.00),
                ("0 to +0.05%", 0.00, 0.05),
                ("+0.05 to +0.10%", 0.05, 0.10),
                ("+0.10 to +0.15%", 0.10, 0.15),
                ("+0.15 to +0.20%", 0.15, 0.20),
                ("+0.20 to +0.25%", 0.20, 0.25),
                ("+0.25 to +0.30%", 0.25, 0.30),
                ("+0.30 to +0.40%", 0.30, 0.40),
                ("+0.40 to +0.50%", 0.40, 0.50),
                ("+0.50 to +0.60%", 0.50, 0.60),
                ("+0.60 to +0.80%", 0.60, 0.80),
                ("+0.80 to +1.00%", 0.80, 1.00),
                ("> +1.00%", 1.00, 999),
            ]
            np_pgap_trades = [o for o in np_trades if o.entry_pair_ema20_ema50_gap_pct is not None]
            all_pgap_trades = [o for o in orders if o.entry_pair_ema20_ema50_gap_pct is not None]
            for rng, lo, hi in np_pgap_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_pgap_trades if lo <= o.entry_pair_ema20_ema50_gap_pct < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_pgap_trades if lo <= o.entry_pair_ema20_ema50_gap_pct < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Pair EMA13-EMA50 Gap", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # BTC EMA13-EMA50 Gap (signed) — May 12: synced to same 24-bucket
            # scheme as Pair version for apples-to-apples cross-table comparison.
            np_btcgap_ranges = np_pgap_ranges
            np_btcgap_trades = [o for o in np_trades if o.entry_btc_trend_gap_pct is not None]
            all_btcgap_trades = [o for o in orders if o.entry_btc_trend_gap_pct is not None]
            for rng, lo, hi in np_btcgap_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_btcgap_trades if lo <= o.entry_btc_trend_gap_pct < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_btcgap_trades if lo <= o.entry_btc_trend_gap_pct < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "BTC EMA13-EMA50 Gap", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # ATR(14) % — pair volatility regime at entry
            np_atr_ranges = [
                ("<0.20%", 0.00, 0.20), ("0.20-0.25%", 0.20, 0.25), ("0.25-0.30%", 0.25, 0.30),
                ("0.30-0.40%", 0.30, 0.40), ("0.40-0.50%", 0.40, 0.50), ("0.50-0.65%", 0.50, 0.65),
                ("0.65-0.85%", 0.65, 0.85), ("0.85-1.00%", 0.85, 1.00), ("≥1.00%", 1.00, 999),
            ]
            np_atr_trades = [o for o in np_trades if o.entry_atr_pct is not None]
            all_atr_trades = [o for o in orders if o.entry_atr_pct is not None]
            for rng, lo, hi in np_atr_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_atr_trades if lo <= o.entry_atr_pct < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_atr_trades if lo <= o.entry_atr_pct < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "ATR%", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Pair 24h Volume USD — liquidity tier at entry
            np_pvol_ranges = [
                ("<$50M", 0, 50_000_000), ("$50-80M", 50_000_000, 80_000_000),
                ("$80-100M", 80_000_000, 100_000_000), ("$100-150M", 100_000_000, 150_000_000),
                ("$150-250M", 150_000_000, 250_000_000), ("$250-500M", 250_000_000, 500_000_000),
                ("$500M-1B", 500_000_000, 1_000_000_000), (">$1B", 1_000_000_000, 9.9e18),
            ]
            np_pvol_trades = [o for o in np_trades if o.entry_pair_volume_24h_usd is not None]
            all_pvol_trades = [o for o in orders if o.entry_pair_volume_24h_usd is not None]
            for rng, lo, hi in np_pvol_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_pvol_trades if lo <= o.entry_pair_volume_24h_usd < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_pvol_trades if lo <= o.entry_pair_volume_24h_usd < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Pair 24h Volume", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Global Volume Ratio at Entry (May 12) — slicing dimension using shared
            # _VOL_BINS (refined to 8 buckets the same day). Surfaces NP concentration
            # in very-low / extreme gvol zones (e.g., SUI's gvol<0.70 LONG cell).
            np_gvol_trades = [o for o in np_trades if o.entry_global_volume_ratio is not None]
            all_gvol_trades = [o for o in orders if o.entry_global_volume_ratio is not None]
            for rng, lo, hi in _VOL_BINS:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_gvol_trades if lo <= o.entry_global_volume_ratio < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_gvol_trades if lo <= o.entry_global_volume_ratio < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Global Vol Ratio", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Range Position at Entry — same fine buckets as Performance by Range Position
            np_rp_ranges = [
                ("0-2%", 0, 2), ("2-5%", 2, 5), ("5-10%", 5, 10),
                ("10-15%", 10, 15), ("15-25%", 15, 25),
                ("25-50%", 25, 50), ("50-75%", 50, 75),
                ("75-85%", 75, 85), ("85-90%", 85, 90),
                ("90-95%", 90, 95), ("95-98%", 95, 98), ("98-100%", 98, 100.1),
            ]
            np_rp_trades = [o for o in np_trades if o.entry_range_position is not None]
            all_rp_trades = [o for o in orders if o.entry_range_position is not None]
            for rng, lo, hi in np_rp_ranges:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_rp_trades if lo <= o.entry_range_position < hi and (o.direction or "LONG") == direction]
                    all_in = len([o for o in all_rp_trades if lo <= o.entry_range_position < hi and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Range Position", rng, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)

            # Derive the set of close-reason bases from the actual never-positive
            # trades instead of hand-picking a whitelist that always gets stale.
            # Strip any " L1"/"L2"/... level suffix so "STOP_LOSS L1" and
            # "STOP_LOSS L2" collapse into one "STOP_LOSS" bucket.
            def _np_reason_base(cr: str) -> str:
                return cr.rsplit(" L", 1)[0] if " L" in cr else cr

            np_reason_bases = sorted({
                _np_reason_base(o.close_reason)
                for o in np_trades if o.close_reason
            })

            def _matches_reason(o, base: str) -> bool:
                if not o.close_reason:
                    return False
                # Exact match OR exact base followed by a level suffix.
                # Guards against false positives like "STOP_LOSS" matching
                # "STOP_LOSS_WIDE L1" via naive startswith.
                return o.close_reason == base or o.close_reason.startswith(base + " L")

            for reason_key in np_reason_bases:
                for direction in ["LONG", "SHORT"]:
                    bucket = [o for o in np_trades if _matches_reason(o, reason_key) and (o.direction or "LONG") == direction]
                    all_in = len([o for o in orders if _matches_reason(o, reason_key) and (o.direction or "LONG") == direction])
                    row = _np_bucket_stats(bucket, "Close Reason", reason_key, direction, all_in)
                    if row:
                        never_positive_deep_dive.append(row)
    except Exception as e:
        logger.error(f"[PERF] Error computing Never Positive deep dive: {e}\n{traceback.format_exc()}")

    # Post-Exit Regret Deep Dive
    post_exit_regret_deep_dive = []
    try:
        # Include every close reason that has post-exit tracking data.
        # The old hand-picked whitelist (BREAKEVEN_EXIT / SIGNAL_LOST / STOP_LOSS
        # / TICK_MOMENTUM_EXIT / RSI_MOMENTUM_EXIT / REGIME_CHANGE) was silently
        # dropping the majority of trades, including TRAILING_STOP (the single
        # most useful regret signal — "did we exit too early?"), the new FL2
        # family (FL_RECOVERED, FL_DEEP_STOP), FL1[WIDE_SL]'s FL_EMERGENCY_SL,
        # and FL_NO_EXPANSION.  Every row that has post_exit_peak_pnl tracked
        # is a candidate for regret analysis.
        pe_orders = [o for o in orders if o.post_exit_peak_pnl is not None and o.close_reason]
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

                ema13_cross_orders = [o for o in group if getattr(o, 'post_exit_ema13_cross_minutes', None) is not None]
                ema13_cross_pct = round(len(ema13_cross_orders) / count * 100, 1) if count > 0 else 0
                avg_ema13_cross_min = sum(o.post_exit_ema13_cross_minutes for o in ema13_cross_orders) / len(ema13_cross_orders) if ema13_cross_orders else None
                avg_ema13_cross_pnl = sum(o.post_exit_ema13_cross_pnl or 0 for o in ema13_cross_orders) / len(ema13_cross_orders) if ema13_cross_orders else None

                # May 23: post-exit regime-flip aggregation. RegFlipMin =
                # minutes from close until BTC regime first stopped supporting
                # trade direction. P&L@RegFlip = running P&L at that moment.
                # NULL/dash = regime never flipped in tracking window (= original
                # exit was structurally correct; holding would have ridden into
                # the post-exit trough). The comparison column for the
                # "EMA13 vs regime exit" question.
                regime_flip_orders = []
                for o in group:
                    _rf_at = getattr(o, 'post_exit_regime_flip_at', None)
                    _exit_at = getattr(o, 'closed_at', None)
                    if _rf_at is not None and _exit_at is not None:
                        regime_flip_orders.append(o)
                regime_flip_pct = round(len(regime_flip_orders) / count * 100, 1) if count > 0 else 0
                if regime_flip_orders:
                    _rf_minutes = []
                    _rf_pnls = []
                    for o in regime_flip_orders:
                        try:
                            _dt = (o.post_exit_regime_flip_at - o.closed_at).total_seconds() / 60.0
                            _rf_minutes.append(_dt)
                        except Exception:
                            pass
                        _pnl = getattr(o, 'post_exit_regime_flip_pnl_pct', None)
                        if _pnl is not None:
                            _rf_pnls.append(_pnl)
                    avg_regime_flip_min = sum(_rf_minutes) / len(_rf_minutes) if _rf_minutes else None
                    avg_regime_flip_pnl = sum(_rf_pnls) / len(_rf_pnls) if _rf_pnls else None
                else:
                    avg_regime_flip_min = None
                    avg_regime_flip_pnl = None

                rec_neg020 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= -0.20)
                rec_neg010 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= -0.10)
                rec_neg005 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= -0.05)
                rec_005 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= 0.05)
                rec_010 = sum(1 for o in group if (o.post_exit_peak_pnl or 0) >= 0.10)

                sig_regained_orders = [o for o in group if o.post_exit_signal_regained_minutes is not None]
                sig_regained_pct = round(len(sig_regained_orders) / count * 100, 1) if count > 0 else 0
                avg_sig_regained_min = sum(o.post_exit_signal_regained_minutes for o in sig_regained_orders) / len(sig_regained_orders) if sig_regained_orders else None
                avg_pnl_at_sig_regained = sum(o.post_exit_pnl_at_signal_regained or 0 for o in sig_regained_orders) / len(sig_regained_orders) if sig_regained_orders else None

                floor_orders = [o for o in sig_regained_orders if o.post_exit_floor_before_signal_regain is not None]
                avg_floor_before_sig_regain = sum(o.post_exit_floor_before_signal_regain for o in floor_orders) / len(floor_orders) if floor_orders else None

                ema5_dists_pe = [o.exit_price_vs_ema5_pct for o in group if o.exit_price_vs_ema5_pct is not None]
                ema5_slopes_pe = [o.exit_ema5_slope_pct for o in group if o.exit_ema5_slope_pct is not None]
                ema5_crossed_pe = sum(1 for o in group if o.exit_ema5_crossed is True)

                # May 22: ATR aggregation for sl_atr_multiplier diagnostic.
                # AvgATR% shows volatility character of the bucket.
                # ATR-SL@1.5× shows what wider SL would have been; compare against AvgClose
                # to gauge whether ATR widening would have rescued the trade.
                atrs_pe = [o.entry_atr_pct for o in group if o.entry_atr_pct is not None and o.entry_atr_pct > 0]
                avg_atr_pe = sum(atrs_pe) / len(atrs_pe) if atrs_pe else None
                atr_sl_widened_pe = -(avg_atr_pe * 1.5) if avg_atr_pe is not None else None

                # May 31: stretch-fade recoverable-regret band (Leash Shadow strpk/stren).
                # strpk = stretch-trail (aggressive ceiling), stren = stretch-to-entry
                # (conservative floor). Both run the ACTUAL post-exit path (respect the
                # sequence-trap, unlike PostPeak%). Read Close% → [stren, strpk] → PostPeak%.
                # Only armed post-deploy trades have shadow data; NULL/dash otherwise.
                strpk_pe = [o.shadow_strpk_pnl for o in group if getattr(o, 'shadow_strpk_pnl', None) is not None]
                stren_pe = [o.shadow_stren_pnl for o in group if getattr(o, 'shadow_stren_pnl', None) is not None]
                avg_strpk_pe = sum(strpk_pe) / len(strpk_pe) if strpk_pe else None
                avg_stren_pe = sum(stren_pe) / len(stren_pe) if stren_pe else None
                # May 31: looser-K stretch-trail variants (0.4 / 0.3) — holds runners longer
                strpk04_pe = [o.shadow_strpk04_pnl for o in group if getattr(o, 'shadow_strpk04_pnl', None) is not None]
                strpk03_pe = [o.shadow_strpk03_pnl for o in group if getattr(o, 'shadow_strpk03_pnl', None) is not None]
                avg_strpk04_pe = sum(strpk04_pe) / len(strpk04_pe) if strpk04_pe else None
                avg_strpk03_pe = sum(strpk03_pe) / len(strpk03_pe) if strpk03_pe else None
                # fire-minute (from open); compare to Duration → pre/post-close
                strpk_min_pe = [o.shadow_strpk_min for o in group if getattr(o, 'shadow_strpk_min', None) is not None]
                stren_min_pe = [o.shadow_stren_min for o in group if getattr(o, 'shadow_stren_min', None) is not None]
                avg_strpk_min = sum(strpk_min_pe) / len(strpk_min_pe) if strpk_min_pe else None
                avg_stren_min = sum(stren_min_pe) / len(stren_min_pe) if stren_min_pe else None

                no_regain = [o for o in group if o.post_exit_signal_regained_minutes is None]
                nr_count = len(no_regain)
                nr_rec_neg020 = sum(1 for o in no_regain if (o.post_exit_peak_pnl or 0) >= -0.20) if nr_count else 0
                nr_rec_neg010 = sum(1 for o in no_regain if (o.post_exit_peak_pnl or 0) >= -0.10) if nr_count else 0
                nr_rec_neg005 = sum(1 for o in no_regain if (o.post_exit_peak_pnl or 0) >= -0.05) if nr_count else 0

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
                    "ema13_cross_pct": ema13_cross_pct,
                    "avg_ema13_cross_min": round(avg_ema13_cross_min, 1) if avg_ema13_cross_min is not None else None,
                    "avg_ema13_cross_pnl": round(avg_ema13_cross_pnl, 4) if avg_ema13_cross_pnl is not None else None,
                    # May 23: post-exit regime-flip diagnostic (EMA13 vs regime exit comparison)
                    "regime_flip_pct": regime_flip_pct,
                    "avg_regime_flip_min": round(avg_regime_flip_min, 1) if avg_regime_flip_min is not None else None,
                    "avg_regime_flip_pnl": round(avg_regime_flip_pnl, 4) if avg_regime_flip_pnl is not None else None,
                    "recovery_neg020_pct": round(rec_neg020 / count * 100, 1),
                    "recovery_neg010_pct": round(rec_neg010 / count * 100, 1),
                    "recovery_neg005_pct": round(rec_neg005 / count * 100, 1),
                    "recovery_005_pct": round(rec_005 / count * 100, 1),
                    "recovery_010_pct": round(rec_010 / count * 100, 1),
                    "sig_regained_pct": sig_regained_pct,
                    "avg_sig_regained_min": round(avg_sig_regained_min, 1) if avg_sig_regained_min is not None else None,
                    "avg_pnl_at_sig_regained": round(avg_pnl_at_sig_regained, 4) if avg_pnl_at_sig_regained is not None else None,
                    "avg_floor_before_sig_regain": round(avg_floor_before_sig_regain, 4) if avg_floor_before_sig_regain is not None else None,
                    "nr_count": nr_count,
                    "nr_rec_neg020_pct": round(nr_rec_neg020 / nr_count * 100, 1) if nr_count > 0 else None,
                    "nr_rec_neg010_pct": round(nr_rec_neg010 / nr_count * 100, 1) if nr_count > 0 else None,
                    "nr_rec_neg005_pct": round(nr_rec_neg005 / nr_count * 100, 1) if nr_count > 0 else None,
                    "avg_exit_ema5_dist": round(sum(ema5_dists_pe) / len(ema5_dists_pe), 4) if ema5_dists_pe else None,
                    "avg_exit_ema5_slope": round(sum(ema5_slopes_pe) / len(ema5_slopes_pe), 4) if ema5_slopes_pe else None,
                    "exit_ema5_crossed_pct": round(ema5_crossed_pe / count * 100, 1) if count > 0 else None,
                    "avg_atr_pct": round(avg_atr_pe, 4) if avg_atr_pe is not None else None,
                    "atr_sl_15x": round(atr_sl_widened_pe, 4) if atr_sl_widened_pe is not None else None,
                    # May 31: stretch-fade recoverable-regret band (observation-only)
                    "avg_strpk_pct": round(avg_strpk_pe, 4) if avg_strpk_pe is not None else None,
                    "avg_strpk04_pct": round(avg_strpk04_pe, 4) if avg_strpk04_pe is not None else None,
                    "avg_strpk03_pct": round(avg_strpk03_pe, 4) if avg_strpk03_pe is not None else None,
                    "avg_stren_pct": round(avg_stren_pe, 4) if avg_stren_pe is not None else None,
                    "avg_strpk_min": round(avg_strpk_min, 1) if avg_strpk_min is not None else None,
                    "avg_stren_min": round(avg_stren_min, 1) if avg_stren_min is not None else None,
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

    # Flagged Exits — trades that hit signal lost but were kept open via the flag system
    flagged_exits = []
    try:
        flagged_orders = [o for o in orders if getattr(o, 'signal_lost_flagged', False)]
        if flagged_orders:
            total_flagged = len(flagged_orders)
            fg_groups = {}
            for o in flagged_orders:
                reason = o.close_reason or "UNKNOWN"
                fg_groups.setdefault(reason, []).append(o)

            for reason in sorted(fg_groups.keys()):
                group = fg_groups[reason]
                count = len(group)
                fg_longs = sum(1 for o in group if o.direction == "LONG")
                fg_shorts = count - fg_longs
                by_conf = {}
                for o in group:
                    c = o.confidence or "UNKNOWN"
                    by_conf[c] = by_conf.get(c, 0) + 1
                # FL Origin breakdown: SIGNAL_LOST vs WIDE_SL
                fl_origin_counts = {}
                for o in group:
                    origin = getattr(o, 'fl1_origin', None) or "SIGNAL_LOST"  # legacy rows default to SIGNAL_LOST
                    fl_origin_counts[origin] = fl_origin_counts.get(origin, 0) + 1
                pct = round(count / total_flagged * 100, 1)
                avg_pnl_pct = round(sum(o.pnl_percentage or 0 for o in group) / count, 4)
                avg_pnl_usd = round(sum(o.pnl or 0 for o in group) / count, 2)
                peak_pnls = [o.peak_pnl for o in group if o.peak_pnl is not None]
                avg_peak_pnl = round(sum(peak_pnls) / len(peak_pnls), 4) if peak_pnls else None
                avg_pullback = round(avg_peak_pnl - avg_pnl_pct, 4) if avg_peak_pnl is not None else None
                flag_pnls = [o.signal_lost_flag_pnl for o in group if o.signal_lost_flag_pnl is not None]
                avg_pnl_at_sl = round(sum(flag_pnls) / len(flag_pnls), 4) if flag_pnls else None
                net_recover = round(avg_pnl_pct - avg_pnl_at_sl, 4) if avg_pnl_at_sl is not None else None
                net_recover_usds = [
                    (o.pnl or 0) - (o.signal_lost_flag_pnl / 100 * o.notional_value)
                    for o in group if o.signal_lost_flag_pnl is not None and o.notional_value
                ]
                net_recover_usd = round(sum(net_recover_usds) / len(net_recover_usds), 2) if net_recover_usds else None
                dur_open_flag = []
                dur_flag_close = []
                for o in group:
                    if o.opened_at and getattr(o, 'signal_lost_flagged_at', None):
                        dur_open_flag.append((o.signal_lost_flagged_at - o.opened_at).total_seconds() / 60.0)
                    if getattr(o, 'signal_lost_flagged_at', None) and o.closed_at:
                        dur_flag_close.append((o.closed_at - o.signal_lost_flagged_at).total_seconds() / 60.0)
                avg_dur_open_flag = round(sum(dur_open_flag) / len(dur_open_flag), 1) if dur_open_flag else None
                avg_dur_flag_close = round(sum(dur_flag_close) / len(dur_flag_close), 1) if dur_flag_close else None

                flagged_exits.append({
                    "reason": reason, "longs": fg_longs, "shorts": fg_shorts, "by_confidence": by_conf,
                    "fl_origin": fl_origin_counts,
                    "count": count, "pct": pct,
                    "avg_pnl_pct": avg_pnl_pct, "avg_pnl_usd": avg_pnl_usd,
                    "avg_peak_pnl": avg_peak_pnl, "avg_pullback": avg_pullback,
                    "avg_pnl_at_sl": avg_pnl_at_sl, "net_recover": net_recover, "net_recover_usd": net_recover_usd,
                    "avg_dur_open_flag": avg_dur_open_flag, "avg_dur_flag_close": avg_dur_flag_close,
                })

            # ALL summary row
            all_avg_pnl_pct = round(sum(o.pnl_percentage or 0 for o in flagged_orders) / total_flagged, 4)
            all_avg_pnl_usd = round(sum(o.pnl or 0 for o in flagged_orders) / total_flagged, 2)
            all_peaks = [o.peak_pnl for o in flagged_orders if o.peak_pnl is not None]
            all_avg_peak = round(sum(all_peaks) / len(all_peaks), 4) if all_peaks else None
            all_avg_pullback = round(all_avg_peak - all_avg_pnl_pct, 4) if all_avg_peak is not None else None
            all_flag_pnls = [o.signal_lost_flag_pnl for o in flagged_orders if o.signal_lost_flag_pnl is not None]
            all_avg_at_sl = round(sum(all_flag_pnls) / len(all_flag_pnls), 4) if all_flag_pnls else None
            all_net_recover = round(all_avg_pnl_pct - all_avg_at_sl, 4) if all_avg_at_sl is not None else None
            all_nr_usds = [
                (o.pnl or 0) - (o.signal_lost_flag_pnl / 100 * o.notional_value)
                for o in flagged_orders if o.signal_lost_flag_pnl is not None and o.notional_value
            ]
            all_net_recover_usd = round(sum(all_nr_usds), 2) if all_nr_usds else None
            all_dur_of = []
            all_dur_fc = []
            for o in flagged_orders:
                if o.opened_at and getattr(o, 'signal_lost_flagged_at', None):
                    all_dur_of.append((o.signal_lost_flagged_at - o.opened_at).total_seconds() / 60.0)
                if getattr(o, 'signal_lost_flagged_at', None) and o.closed_at:
                    all_dur_fc.append((o.closed_at - o.signal_lost_flagged_at).total_seconds() / 60.0)
            all_longs = sum(1 for o in flagged_orders if o.direction == "LONG")
            all_shorts = total_flagged - all_longs
            all_by_conf = {}
            for o in flagged_orders:
                c = o.confidence or "UNKNOWN"
                all_by_conf[c] = all_by_conf.get(c, 0) + 1
            all_fl_origin = {}
            for o in flagged_orders:
                origin = getattr(o, 'fl1_origin', None) or "SIGNAL_LOST"
                all_fl_origin[origin] = all_fl_origin.get(origin, 0) + 1
            flagged_exits.append({
                "reason": "ALL", "longs": all_longs, "shorts": all_shorts, "by_confidence": all_by_conf,
                "fl_origin": all_fl_origin,
                "count": total_flagged, "pct": 100.0,
                "avg_pnl_pct": all_avg_pnl_pct, "avg_pnl_usd": all_avg_pnl_usd,
                "avg_peak_pnl": all_avg_peak, "avg_pullback": all_avg_pullback,
                "avg_pnl_at_sl": all_avg_at_sl, "net_recover": all_net_recover, "net_recover_usd": all_net_recover_usd,
                "avg_dur_open_flag": round(sum(all_dur_of) / len(all_dur_of), 1) if all_dur_of else None,
                "avg_dur_flag_close": round(sum(all_dur_fc) / len(all_dur_fc), 1) if all_dur_fc else None,
            })
    except Exception as e:
        logger.error(f"[PERF] Error computing Flagged Exits: {e}\n{traceback.format_exc()}")


    return {
        "total_trades": total_trades,
        "total_longs": total_longs,
        "total_shorts": total_shorts,
        "total_wins": len(all_wins),
        "total_losses": len(all_losses),
        "win_rate": round(win_rate, 2),
        "win_rate_longs": round(win_rate_longs, 2),
        "win_rate_shorts": round(win_rate_shorts, 2),
        "avg_win": round(avg_win, 2),
        "avg_win_long": round(avg_win_long, 2),
        "avg_win_short": round(avg_win_short, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_loss_long": round(avg_loss_long, 2),
        "avg_loss_short": round(avg_loss_short, 2),
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_win_long_pct": round(avg_win_long_pct, 2),
        "avg_win_short_pct": round(avg_win_short_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "avg_loss_long_pct": round(avg_loss_long_pct, 2),
        "avg_loss_short_pct": round(avg_loss_short_pct, 2),
        "expectancy": round(expectancy, 2),
        "expectancy_pct": round(expectancy_pct, 2),
        "best_win_long": round(best_win_long, 2),
        "best_win_short": round(best_win_short, 2),
        "worst_loss_long": round(worst_loss_long, 2),
        "worst_loss_short": round(worst_loss_short, 2),
        "best_win_long_pct": round(best_win_long_pct, 2),
        "best_win_short_pct": round(best_win_short_pct, 2),
        "worst_loss_long_pct": round(worst_loss_long_pct, 2),
        "worst_loss_short_pct": round(worst_loss_short_pct, 2),
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
        "ema813_gap_performance": ema813_gap_performance,
        "ema_fan_accel_performance": ema_fan_accel_performance,
        "rsi_performance": rsi_performance,
        "adx_performance": adx_performance,
        "adx_direction_performance": adx_direction_performance,
        "rsi_direction_performance": rsi_direction_performance,
        "stretch_performance": stretch_performance,
        "range_position_performance": range_position_performance,
        "adx_delta_performance": adx_delta_performance,
        "pair_slope_performance": pair_slope_performance,
        "btc_slope_performance": btc_slope_performance,
        "pair_ema20_ema50_gap_performance": pair_ema20_ema50_gap_performance,
        "btc_ema20_ema50_gap_performance": btc_ema20_ema50_gap_performance,
        "btc_adx_performance": btc_adx_performance,
        "btc_adx_direction_performance": btc_adx_direction_performance,
        "btc_rsi_direction_performance": btc_rsi_direction_performance,
        "btc_rsi_direction_30m_performance": btc_rsi_direction_30m_performance,
        "btc_volatility_performance": btc_volatility_performance,
        "btc_rsi_1h_direction_performance": btc_rsi_1h_direction_performance,
        "btc_vol_adx_crosstab": btc_vol_adx_crosstab,
        "btc_rsi_1h_5m_crosstab": btc_rsi_1h_5m_crosstab,
        "adx_dir_crosstab": adx_dir_crosstab,
        "rsi_dir_crosstab": rsi_dir_crosstab,
        "btc_rsi_30m_5m_crosstab": btc_rsi_30m_5m_crosstab,
        "range_pos_btc_rsi_dir_crosstab": range_pos_btc_rsi_dir_crosstab,
        "range_pos_pair_rsi_dir_crosstab": range_pos_pair_rsi_dir_crosstab,
        "pair_slope_adx_crosstab": pair_slope_adx_crosstab,
        "btc_slope_adx_crosstab": btc_slope_adx_crosstab,
        "adx_delta_btc_adx_crosstab": adx_delta_btc_adx_crosstab,
        "btc_gap_btc_adx_crosstab": btc_gap_btc_adx_crosstab,
        "pair_gap_pair_adx_crosstab": pair_gap_pair_adx_crosstab,
        "btc_rsi_performance": btc_rsi_performance,
        "btc_rsi_adx_crosstab": btc_rsi_adx_crosstab,
        "quality_score_performance": quality_score_performance,
        "regime_performance": regime_performance,
        "regime_transition_performance": regime_transition_performance,
        "by_close_reason": by_close_reason,
        "stop_loss_deep_dive": stop_loss_deep_dive,
        "winning_trades_drawdown": winning_trades_drawdown,
        "trough_recovery": trough_recovery,
        "rsi_adx_crosstab": rsi_adx_crosstab,
        "never_positive_deep_dive": never_positive_deep_dive,
        "performance_over_time": _compute_time_buckets(orders),
        "by_entry_type": _compute_entry_type_stats(orders, signal_expired_orders=signal_expired_orders),
        "signal_expired_breakdown": _compute_signal_expired_breakdown(signal_expired_orders),
        "by_exit_type": _compute_exit_type_stats(orders),
        "post_exit_regret_deep_dive": post_exit_regret_deep_dive,
        "hold_time_expectancy": hold_time_expectancy,
        "entry_conditions_by_reason": entry_conditions_by_reason,
        "entry_conditions_by_outcome": entry_conditions_by_outcome,
        "flagged_exits": flagged_exits,
        "period_performance": _compute_period_performance(orders),
        "equity_curve": _compute_equity_curve(orders),
        "pnl_distribution": _compute_pnl_distribution(orders),
        "hourly_performance": _compute_hourly_performance(orders),
        "daily_performance": _compute_daily_performance(orders),
        "day_time_heatmap": _compute_day_time_heatmap(orders),
        "volume_crosstab": _compute_volume_crosstab(orders),
        # May 10 evening: 2D Global Vol Ratio × Pair Vol USD cross-tab. Buckets
        # match existing Volume Cross-Tab (Global axis) and Performance by Pair
        # 24h Volume (Pair axis) tables for cross-reference consistency.
        "volume_intersection_crosstab": _compute_volume_intersection_crosstab(orders),
        "breadth_crosstab": _compute_breadth_crosstab(orders),
        "pair_performance": _compute_pair_performance(orders),
        # May 10: pair 24h USD volume bucket performance — find structural size threshold
        "pair_volume_bucket_performance": _compute_pair_volume_bucket_performance(orders),
        # May 9: ATR bucket performance — tests "high volatility = loss driver" hypothesis
        "atr_bucket_performance": _compute_atr_bucket_performance(orders),
        # Premium Multiplier Cell Performance (May 4, 2026 — Phase 3 Position Multiplier per CLAUDE.md May 3)
        "multiplier_cell_performance": _compute_multiplier_cell_performance(orders),
        # Pattern Cell Ship Performance (May 21, NEW — pattern-based rules per CLAUDE.md May 21 ship)
        "pattern_cell_performance": _compute_pattern_cell_performance(orders),
        "extension_multiplier_performance": _compute_extension_multiplier_performance(orders),
        "btc_1h_slope_btc_adx_multiplier_performance": _compute_btc_1h_slope_btc_adx_multiplier_performance(orders),
        # 4-Cohort Pattern Coverage (May 21 late — split Unm.L/Unm.W into Both/C-only/W-only/Truly)
        "pattern_4cohort_coverage": _compute_pattern_4cohort_coverage(orders),
        # Pattern Combo Trackers (May 26 evening — surface within-tracker combos like C2+C4 or W2+W4)
        "pattern_c_combo_tracker": _compute_pattern_combo_tracker(orders, tracker='C'),
        "pattern_w_combo_tracker": _compute_pattern_combo_tracker(orders, tracker='W'),
        # EMA13 strict-mode performance (May 8, 2026 — tracks impact of ema13_cross_requires_stack_flip)
        "ema13_strict_performance": _compute_ema13_strict_performance(orders),
        # Trailing pullback confirmation performance (May 9, 2026)
        "trailing_confirmation_performance": _compute_trailing_confirmation_performance(orders),
        # Post-exit P&L snapshots for EMA13_CROSS_EXIT and STOP_LOSS_WIDE (May 12 LATE PM)
        "post_exit_snapshots_by_reason": _compute_post_exit_snapshots_by_reason(orders),
        # Entry Extension / Late Entry Risk (May 13 PM) — tests whether bad trades
        # are timing failures (late entries) vs signal-quality failures.
        "ema13_extension_performance": _compute_ema13_extension_performance(orders),
        "ema13_extension_pvol_crosstab": _compute_ema13_extension_pvol_crosstab(orders),
        "ema13_extension_adxdelta_crosstab": _compute_ema13_extension_adxdelta_crosstab(orders),
        "ema13_extension_pair_adx_crosstab": _compute_ema13_extension_pair_adx_crosstab(orders),
        # BTC Market Extension / BTC Late Regime Risk (May 14) — macro counterpart of
        # pair extension. Tests whether losses cluster when BTC itself is stretched
        # (price far from BTC EMA13), and especially when both BTC and pair are
        # extended simultaneously (double-stretch).
        "btc_extension_performance": _compute_btc_extension_performance(orders),
        "btc_extension_globalvol_crosstab": _compute_btc_extension_globalvol_crosstab(orders),
        "btc_extension_pair_extension_crosstab": _compute_btc_extension_pair_extension_crosstab(orders),
        # BTC 1h Slope (May 14) — higher-TF macro context. Discriminator candidate
        # after May 4 finding that every 5m-timeframe dimension showed identical
        # winner/loser signatures. Tests "5m bearish blip during 1h uptrend" hypothesis.
        "btc_1h_slope_performance": _compute_btc_1h_slope_performance(orders),
        "btc_5m_1h_slope_alignment_crosstab": _compute_btc_5m_1h_slope_alignment_crosstab(orders),
        "btc_1h_30m_rsi_direction_crosstab": _compute_btc_1h_30m_rsi_direction_crosstab(orders),
        "btc_1h_slope_adx_crosstab": _compute_btc_1h_slope_adx_crosstab(orders),
        # BTC ATR × BTC ADX cross-tab (May 22, 2026). Surfaces the BTC ATR<0.10
        # × BTC ADX ≥30 SHORT killer cell and the asymmetric LONG mirror.
        "btc_atr_adx_crosstab": _compute_btc_atr_adx_crosstab(orders),
        # Pattern C Tracker validation (May 19, observation-only)
        "pattern_c_validation": _compute_pattern_c_validation(orders),
        # Pattern C batch coverage + unmatched-losers deep dive (May 20 late)
        "pattern_c_batch_coverage": _compute_pattern_c_batch_coverage(orders),
        "pattern_c_unmatched_losers": _compute_unmatched_losers(orders, limit=20),
        # Pattern W tracker (May 20 latest+2 — observation-only, mirror of Pattern C for WINNERS)
        "pattern_w_validation": _compute_pattern_w_validation(orders),
        # Pattern W batch coverage + unmatched-winners deep dive (May 20 latest+3)
        "pattern_w_batch_coverage": _compute_pattern_w_batch_coverage(orders),
        "pattern_w_unmatched_winners": _compute_unmatched_winners(orders, limit=20),
        "leash_shadow": _compute_leash_shadow(orders),  # LEASH SHADOW (May 30, observation-only)
        # Fast-exit counterfactual grid (May 13 — Option A analytics).
        # Tests: "what if we exited at +X% the moment P&L reached it within N min?"
        # Compares real outcome vs hypothetical fast-exit across (threshold × window) grid.
        # Uses peak_pnl + peak_reached_at — conservative (only catches trades whose
        # peak happens within window; misses trades that crossed threshold before peak).
        "fast_exit_counterfactual": _compute_fast_exit_counterfactual(orders),
    }


def _compute_trailing_confirmation_performance(orders):
    """Trailing pullback confirmation tracking (May 9, 2026 — TP-level breakdown added).

    For trades where the trailing confirmation timer captured the
    counterfactual (`trailing_first_pullback_pnl_pct IS NOT NULL`),
    compares would-have-been-immediate-fire P&L vs the actual close.
    Positive delta = confirmation helped (price recovered after wick,
    trade ran further). Negative delta = confirmation hurt.

    Breakdown: rows for (direction × TP level) with totals per direction.
    L3+ pools L3, L4, L5 since each is rare. This lets us see if
    confirmation helps L1 (smaller winners, tighter trailing — sensitive
    to wicks) but hurts L3+ (bigger winners — may want to lock profits).
    """
    # Bucket trades by direction × tp_level
    by_dir_level = {}  # (direction, tp_label) -> list of records
    for o in orders:
        if getattr(o, 'status', None) != "CLOSED":
            continue
        first_pullback = getattr(o, 'trailing_first_pullback_pnl_pct', None)
        if first_pullback is None:
            continue
        d = (o.direction.value if hasattr(o.direction, 'value') else o.direction) or "LONG"
        if d not in ("LONG", "SHORT"):
            continue
        tp = getattr(o, 'current_tp_level', 1) or 1
        if tp <= 1:
            tp_label = "L1"
        elif tp == 2:
            tp_label = "L2"
        else:
            tp_label = "L3+"
        close_pct = o.pnl_percentage if o.pnl_percentage is not None else 0.0
        delta_pct = close_pct - first_pullback
        if close_pct and abs(close_pct) > 1e-9 and o.pnl is not None:
            dollar_delta = o.pnl * (delta_pct / close_pct)
        else:
            inv = o.investment or 0.0
            lev = o.leverage or 1
            dollar_delta = (delta_pct / 100.0) * inv * lev
        rec = {
            'first_pullback_pct': first_pullback,
            'close_pct': close_pct,
            'peak_pct': o.peak_pnl if o.peak_pnl is not None else None,  # May 12: peak during trade
            'delta_pct': delta_pct,
            'dollar_delta': dollar_delta,
            'pnl': o.pnl or 0.0,
            'resets': getattr(o, 'trailing_pullback_resets', 0) or 0,
            # May 12 LATE PM: time-bucketed post-exit P&L snapshots
            'pnl_at_1min': getattr(o, 'post_exit_pnl_at_1min', None),
            'pnl_at_2min': getattr(o, 'post_exit_pnl_at_2min', None),
            'pnl_at_5min': getattr(o, 'post_exit_pnl_at_5min', None),
            'pnl_at_15min': getattr(o, 'post_exit_pnl_at_15min', None),
            'pnl_at_30min': getattr(o, 'post_exit_pnl_at_30min', None),
            # May 23: ATR-trailing diagnostic. atr_pct lets us see which trades
            # were eligible for the wider 0.50× trailing (atr ≥ 0.60%).
            'atr_pct': getattr(o, 'entry_atr_pct', None),
        }
        by_dir_level.setdefault((d, tp_label), []).append(rec)

    def _verdict(rows):
        if not rows or len(rows) < 5:
            return "⚠ Low N"
        n = len(rows)
        avg_delta = sum(r['delta_pct'] for r in rows) / n
        total_dollar = sum(r['dollar_delta'] for r in rows)
        if avg_delta >= 0.05 and total_dollar > 0:
            return "★ HELPING"
        if avg_delta <= -0.05 or total_dollar < 0:
            return "⚠ HURTING"
        return "✓ Marginal"

    def _avg_skip_none(rs, key):
        """Avg of non-None values for `key`. Returns (avg, n_with_data) or (None, 0)."""
        vals = [r[key] for r in rs if r.get(key) is not None]
        if not vals:
            return None, 0
        return round(sum(vals) / len(vals), 4), len(vals)

    def _row_from(rs, label_suffix=""):
        if not rs:
            return None
        n = len(rs)
        n_with_resets = sum(1 for r in rs if r['resets'] > 0)
        # May 12 LATE PM: time-bucketed averages.
        # n_at_Xmin tells operator how many trades had the snapshot captured
        # (vs NULL = tracking ended before reaching that threshold).
        avg_1m, n_1m = _avg_skip_none(rs, 'pnl_at_1min')
        avg_2m, n_2m = _avg_skip_none(rs, 'pnl_at_2min')
        avg_5m, n_5m = _avg_skip_none(rs, 'pnl_at_5min')
        avg_15m, n_15m = _avg_skip_none(rs, 'pnl_at_15min')
        avg_30m, n_30m = _avg_skip_none(rs, 'pnl_at_30min')
        # May 12 LATE PM: avg peak during trade (reference for pullback decisions)
        avg_peak_pct, _ = _avg_skip_none(rs, 'peak_pct')
        avg_close_pct = round(sum(r['close_pct'] for r in rs) / n, 4)
        # Pullback used = peak - close (positive number = how much retraced before exit)
        avg_pullback_used = round(avg_peak_pct - avg_close_pct, 4) if avg_peak_pct is not None else None
        # May 12 LATE PM (v2): PB needed to reach each +Nmin snapshot
        # = max(0, peak - +Nmin%). Compare to avg_pullback_used:
        #   below = recovery zone (wider PB might capture)
        #   above = decay (wider PB would catch worse price)
        def _pb_needed(avg_snap):
            if avg_snap is None or avg_peak_pct is None:
                return None
            return round(max(0, avg_peak_pct - avg_snap), 4)
        return {
            'n': n,
            'n_with_resets': n_with_resets,
            'avg_peak_pct': avg_peak_pct,
            'avg_first_pullback_pct': round(sum(r['first_pullback_pct'] for r in rs) / n, 4),
            'avg_close_pct': avg_close_pct,
            'avg_pullback_used': avg_pullback_used,
            'avg_delta_pct': round(sum(r['delta_pct'] for r in rs) / n, 4),
            'total_dollar_delta': round(sum(r['dollar_delta'] for r in rs), 2),
            'avg_pnl_at_1min': avg_1m, 'n_at_1min': n_1m,
            'avg_pnl_at_2min': avg_2m, 'n_at_2min': n_2m,
            'avg_pnl_at_5min': avg_5m, 'n_at_5min': n_5m,
            'avg_pnl_at_15min': avg_15m, 'n_at_15min': n_15m,
            'avg_pnl_at_30min': avg_30m, 'n_at_30min': n_30m,
            'pb_to_1min': _pb_needed(avg_1m),
            'pb_to_2min': _pb_needed(avg_2m),
            'pb_to_5min': _pb_needed(avg_5m),
            'pb_to_15min': _pb_needed(avg_15m),
            'pb_to_30min': _pb_needed(avg_30m),
            # May 23: ATR-trailing diagnostic columns.
            # avg_atr_pct: sanity check on ATR distribution for this tier.
            # atr_active_pct: % of trades where atr × 0.50 > 0.30 (i.e., atr ≥ 0.60).
            #   If 0%, the wider trailing is dormant on this row.
            # cf_30_close_pct: simulated close if trailing_atr_multiplier were 0.30
            #   (the previous value). For atr < 0.60: same as actual close (no diff).
            #   For atr ≥ 0.60: approx = first_pullback_pnl (= moment 0.30% retrace hit).
            # cf_30_delta_pp: avg_close − cf_30_close. Positive = wider trailing
            #   captured more (ship working); negative = wider gave back more (revert).
            'avg_atr_pct': (lambda v: round(v, 4) if v is not None else None)(_avg_skip_none(rs, 'atr_pct')[0]),
            'atr_active_pct': round(100.0 * sum(1 for r in rs if (r.get('atr_pct') or 0) >= 0.60) / n, 1) if n else 0.0,
            'cf_30_close_pct': round(sum(
                (r['first_pullback_pct'] if (r.get('atr_pct') or 0) >= 0.60 else r['close_pct'])
                for r in rs
            ) / n, 4) if n else None,
            'cf_30_delta_pp': round(avg_close_pct - (sum(
                (r['first_pullback_pct'] if (r.get('atr_pct') or 0) >= 0.60 else r['close_pct'])
                for r in rs
            ) / n), 4) if n else None,
            'verdict': _verdict(rs),
        }

    # Build ordered output: per direction emit rows for L1, L2, L3+, then TOTAL
    rows_out = []
    grand_total_n = 0
    grand_total_resets = 0
    grand_sum_dollar = 0.0
    grand_sum_delta_weighted = 0.0
    for d in ("LONG", "SHORT"):
        all_dir = []
        for tp_label in ("L1", "L2", "L3+"):
            rs = by_dir_level.get((d, tp_label), [])
            if rs:
                row = _row_from(rs)
                row['direction'] = d
                row['tp_level'] = tp_label
                rows_out.append(row)
                all_dir.extend(rs)
        if all_dir:
            tot = _row_from(all_dir)
            tot['direction'] = d
            tot['tp_level'] = "TOTAL"
            rows_out.append(tot)
            grand_total_n += tot['n']
            grand_total_resets += tot['n_with_resets']
            grand_sum_dollar += tot['total_dollar_delta']
            grand_sum_delta_weighted += tot['avg_delta_pct'] * tot['n']

    summary = {
        'n': grand_total_n,
        'n_with_resets': grand_total_resets,
        'avg_delta_pct': round(grand_sum_delta_weighted / grand_total_n, 4) if grand_total_n > 0 else 0.0,
        'total_dollar_delta': round(grand_sum_dollar, 2),
    }
    return {'rows': rows_out, 'summary': summary}


def _compute_post_exit_snapshots_by_reason(orders):
    """Post-exit P&L snapshots for EMA13_CROSS_EXIT and STOP_LOSS_WIDE.

    Mirror of Trailing Confirmation's time-bucket columns, but for two
    different exit mechanisms answering different questions:

    EMA13_CROSS_EXIT — "did we exit too early on a wick that recovered?"
      ★ EARLY EXIT  = avg_pnl_at_5min ≥ avg_close + 0.20pp (trade kept going)
      ✓ CORRECT     = avg_pnl_at_5min ≤ avg_close (no recovery)
      ⚠ AMBIGUOUS   = anything in between

    STOP_LOSS_WIDE — "did SL fire correctly or on a wick that recovered?"
      ⚠ SL ON NOISE = avg_pnl_at_5min > -0.50% (substantial recovery — SL
                       fired on a wick that recovered; suggests widen SL)
      ★ SL CORRECT  = avg_pnl_at_5min ≤ -0.70% (price kept dropping)
      ✓ AMBIGUOUS   = anything in between

    Rows: (direction × close_reason). One row per direction per reason if
    data exists. Snapshot cells show "avg(N)" — N is non-null trades that
    actually reached that time threshold.
    """
    by_dir_reason = {}  # (direction, reason_base) -> list of records
    REASONS_OF_INTEREST = {'EMA13_CROSS_EXIT', 'STOP_LOSS_WIDE'}

    for o in orders:
        if getattr(o, 'status', None) != 'CLOSED':
            continue
        cr = o.close_reason or ''
        # Strip " L1" / " L2" suffix
        reason_base = cr.rsplit(' L', 1)[0] if ' L' in cr else cr
        if reason_base not in REASONS_OF_INTEREST:
            continue
        d = (o.direction.value if hasattr(o.direction, 'value') else o.direction) or 'LONG'
        if d not in ('LONG', 'SHORT'):
            continue
        rec = {
            'close_pct': o.pnl_percentage if o.pnl_percentage is not None else 0.0,
            'peak_pct': o.peak_pnl if o.peak_pnl is not None else None,  # May 12: peak during trade
            'pnl_at_1min': getattr(o, 'post_exit_pnl_at_1min', None),
            'pnl_at_2min': getattr(o, 'post_exit_pnl_at_2min', None),
            'pnl_at_5min': getattr(o, 'post_exit_pnl_at_5min', None),
            'pnl_at_15min': getattr(o, 'post_exit_pnl_at_15min', None),
            'pnl_at_30min': getattr(o, 'post_exit_pnl_at_30min', None),
        }
        by_dir_reason.setdefault((d, reason_base), []).append(rec)

    def _avg_skip_none(rs, key):
        vals = [r[key] for r in rs if r.get(key) is not None]
        if not vals:
            return None, 0
        return round(sum(vals) / len(vals), 4), len(vals)

    def _verdict_ema13(rs, avg_close):
        avg_5m, n_5m = _avg_skip_none(rs, 'pnl_at_5min')
        if n_5m < 5:
            return '⚠ Low N'
        delta_5m = avg_5m - avg_close
        if delta_5m >= 0.20:
            return '★ EXIT TOO EARLY'
        if delta_5m <= 0:
            return '✓ EXIT CORRECT'
        return '⚠ AMBIGUOUS'

    def _verdict_sl(rs):
        avg_5m, n_5m = _avg_skip_none(rs, 'pnl_at_5min')
        if n_5m < 5:
            return '⚠ Low N'
        if avg_5m > -0.50:
            return '⚠ SL ON NOISE'  # recovered substantially → SL fired on wick
        if avg_5m <= -0.70:
            return '★ SL CORRECT'  # kept dropping → SL saved us
        return '✓ AMBIGUOUS'

    rows_out = []
    for reason_base in ('EMA13_CROSS_EXIT', 'STOP_LOSS_WIDE'):
        for d in ('LONG', 'SHORT'):
            rs = by_dir_reason.get((d, reason_base), [])
            if not rs:
                continue
            n = len(rs)
            avg_close = round(sum(r['close_pct'] for r in rs) / n, 4)
            # May 12 LATE PM: peak during trade (reference for context)
            avg_peak, _ = _avg_skip_none(rs, 'peak_pct')
            avg_1m, n_1m = _avg_skip_none(rs, 'pnl_at_1min')
            avg_2m, n_2m = _avg_skip_none(rs, 'pnl_at_2min')
            avg_5m, n_5m = _avg_skip_none(rs, 'pnl_at_5min')
            avg_15m, n_15m = _avg_skip_none(rs, 'pnl_at_15min')
            avg_30m, n_30m = _avg_skip_none(rs, 'pnl_at_30min')
            verdict = _verdict_ema13(rs, avg_close) if reason_base == 'EMA13_CROSS_EXIT' else _verdict_sl(rs)
            # May 12 LATE PM (v2): PB needed to reach each +Nmin from peak.
            # For SL: this translates to "SL widening needed" (same math).
            def _pb_needed_pes(avg_snap):
                if avg_snap is None or avg_peak is None:
                    return None
                return round(max(0, avg_peak - avg_snap), 4)
            rows_out.append({
                'close_reason': reason_base,
                'direction': d,
                'n': n,
                'avg_peak_pct': avg_peak,
                'avg_close_pct': avg_close,
                'avg_pnl_at_1min': avg_1m, 'n_at_1min': n_1m,
                'avg_pnl_at_2min': avg_2m, 'n_at_2min': n_2m,
                'avg_pnl_at_5min': avg_5m, 'n_at_5min': n_5m,
                'avg_pnl_at_15min': avg_15m, 'n_at_15min': n_15m,
                'avg_pnl_at_30min': avg_30m, 'n_at_30min': n_30m,
                'pb_to_1min': _pb_needed_pes(avg_1m),
                'pb_to_2min': _pb_needed_pes(avg_2m),
                'pb_to_5min': _pb_needed_pes(avg_5m),
                'pb_to_15min': _pb_needed_pes(avg_15m),
                'pb_to_30min': _pb_needed_pes(avg_30m),
                'verdict': verdict,
            })

    return {'rows': rows_out}


# =================================================================
# Entry Extension / Late Entry Risk (May 13 PM)
# Hypothesis: bad trades may be late entries (timing failures) rather
# than wrong signals (quality failures). Distance from EMA13 at entry
# measures how late within the move the entry fired.
#
# LONG: signed_dist > 0 = price above EMA13 = chasing/late
# SHORT: signed_dist < 0 = price below EMA13 = late after capitulation
# =================================================================

def _compute_ema13_extension_performance(orders):
    """Single-dim bucketing of EMA13 extension by direction.

    For LONG: extension = signed_dist (positive = above EMA13 = late)
    For SHORT: extension = -signed_dist (positive = below EMA13 = late)

    So in both cases, "extension > 0" means the entry was late within the
    move. "extension < 0" means we entered on a pullback (ideal).
    """
    closed = [o for o in orders if (o.status == 'CLOSED') and o.entry_dist_from_ema13_pct is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    # Buckets sized for typical 5m crypto entries — 0.10% granularity
    # Refined May 24: previous coarse 3-bucket view at >+0.20% was hiding
    # non-monotonic structure. Cross-batch shows +0.50-0.70% is WINNER zone
    # while +0.70-0.80% is DISASTER cell. Fine granularity surfaces it.
    # Negative buckets = pullback entry (ideal); positive = late
    buckets = [
        ('< -0.20%', -99, -0.20, 'pullback'),
        ('-0.20 to -0.10%', -0.20, -0.10, 'pullback'),
        ('-0.10 to 0%', -0.10, 0.0, 'near mean'),
        ('0 to +0.10%', 0.0, 0.10, 'mild late'),
        ('+0.10 to +0.20%', 0.10, 0.20, 'late (blocked LONG)'),
        ('+0.20 to +0.30%', 0.20, 0.30, 'extended'),
        ('+0.30 to +0.40%', 0.30, 0.40, 'extended'),
        ('+0.40 to +0.50%', 0.40, 0.50, 'very extended'),
        ('+0.50 to +0.60%', 0.50, 0.60, 'very extended'),
        ('+0.60 to +0.70%', 0.60, 0.70, 'extreme'),
        ('+0.70 to +0.80%', 0.70, 0.80, 'extreme'),
        ('+0.80 to +1.00%', 0.80, 1.00, 'extreme'),
        ('> +1.00%', 1.00, 99, 'parabolic'),
    ]

    def _bucket_for_direction(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for label, lo, hi, tag in buckets:
            # For LONG, extension = entry_dist (positive = chasing)
            # For SHORT, extension = -entry_dist (positive = late)
            if direction == 'LONG':
                sub = [o for o in dir_orders if lo <= o.entry_dist_from_ema13_pct < hi]
            else:
                sub = [o for o in dir_orders if lo <= -o.entry_dist_from_ema13_pct < hi]
            if not sub:
                continue
            n = len(sub)
            wins = sum(1 for o in sub if (o.pnl or 0) > 0)
            total_pnl = sum(o.pnl or 0 for o in sub)
            avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
            avg_peak = sum(o.peak_pnl or 0 for o in sub) / n
            doa = sum(1 for o in sub if (o.peak_pnl or 0) <= 0.10)
            np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
            conf = {}
            for o in sub:
                c = o.confidence or 'UNKNOWN'
                conf[c] = conf.get(c, 0) + 1
            rows.append({
                'range': label,
                'tag': tag,
                'count': n,
                'win_rate': round(wins / n * 100, 1),
                'avg_pnl_pct': round(avg_pnl_pct, 4),
                'avg_pnl_usd': round(total_pnl / n, 2),
                'total_pnl_usd': round(total_pnl, 2),
                'avg_peak_pct': round(avg_peak, 4),
                'doa_count': doa,
                'doa_pct': round(doa / n * 100, 1),
                'np_count': np,
                'np_pct': round(np / n * 100, 1),
                'by_confidence': conf,
            })
        return rows

    return {
        'longs': _bucket_for_direction('LONG'),
        'shorts': _bucket_for_direction('SHORT'),
        'pool_size': len(closed),
    }


def _compute_ema13_extension_pvol_crosstab(orders):
    """Cross-tab: Entry Extension (directional, positive = late) × Pair Vol Ratio.

    Per CLAUDE.md May 13 PM hypothesis: high extension + high pvol = exhaustion.
    """
    closed = [o for o in orders if (o.status == 'CLOSED') and o.entry_dist_from_ema13_pct is not None and o.entry_pair_volume_ratio is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    ext_buckets = [
        ('< 0%', -99, 0.0),
        ('0 to +0.20%', 0.0, 0.20),
        ('+0.20 to +0.40%', 0.20, 0.40),
        ('+0.40 to +0.60%', 0.40, 0.60),
        ('> +0.60%', 0.60, 99),
    ]
    pvol_buckets = [
        ('< 0.95', 0.0, 0.95),
        ('0.95-1.10', 0.95, 1.10),
        ('1.10-1.25', 1.10, 1.25),
        ('> 1.25', 1.25, 99),
    ]

    def _crosstab_for(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for ext_label, ext_lo, ext_hi in ext_buckets:
            for pv_label, pv_lo, pv_hi in pvol_buckets:
                # Direction-aware extension
                if direction == 'LONG':
                    sub = [o for o in dir_orders if ext_lo <= o.entry_dist_from_ema13_pct < ext_hi and pv_lo <= o.entry_pair_volume_ratio < pv_hi]
                else:
                    sub = [o for o in dir_orders if ext_lo <= -o.entry_dist_from_ema13_pct < ext_hi and pv_lo <= o.entry_pair_volume_ratio < pv_hi]
                if not sub:
                    continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                total_pnl = sum(o.pnl or 0 for o in sub)
                avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'extension': ext_label,
                    'pvol_ratio': pv_label,
                    'count': n,
                    'win_rate': round(wins / n * 100, 1),
                    'avg_pnl_pct': round(avg_pnl_pct, 4),
                    'avg_pnl_usd': round(total_pnl / n, 2),
                    'total_pnl_usd': round(total_pnl, 2),
                    'np_count': np,
                    'np_pct': round(np / n * 100, 1),
                })
        return rows

    return {
        'longs': _crosstab_for('LONG'),
        'shorts': _crosstab_for('SHORT'),
        'pool_size': len(closed),
    }


def _compute_ema13_extension_adxdelta_crosstab(orders):
    """Cross-tab: Entry Extension × ADX Delta.

    Tests: late entries (high extension) combined with accelerating
    momentum (high ADX delta) = pure exhaustion signature.
    """
    closed = [o for o in orders if (o.status == 'CLOSED') and o.entry_dist_from_ema13_pct is not None and o.entry_adx_delta is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    ext_buckets = [
        ('< 0%', -99, 0.0),
        ('0 to +0.20%', 0.0, 0.20),
        ('+0.20 to +0.40%', 0.20, 0.40),
        ('+0.40 to +0.60%', 0.40, 0.60),
        ('> +0.60%', 0.60, 99),
    ]
    adxd_buckets = [
        ('< 0.3', -99, 0.3),
        ('0.3-0.7', 0.3, 0.7),
        ('0.7-1.2', 0.7, 1.2),
        ('1.2-1.8', 1.2, 1.8),
        ('> 1.8', 1.8, 99),
    ]

    def _crosstab_for(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for ext_label, ext_lo, ext_hi in ext_buckets:
            for adxd_label, adxd_lo, adxd_hi in adxd_buckets:
                if direction == 'LONG':
                    sub = [o for o in dir_orders if ext_lo <= o.entry_dist_from_ema13_pct < ext_hi and adxd_lo <= o.entry_adx_delta < adxd_hi]
                else:
                    sub = [o for o in dir_orders if ext_lo <= -o.entry_dist_from_ema13_pct < ext_hi and adxd_lo <= o.entry_adx_delta < adxd_hi]
                if not sub:
                    continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                total_pnl = sum(o.pnl or 0 for o in sub)
                avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'extension': ext_label,
                    'adx_delta': adxd_label,
                    'count': n,
                    'win_rate': round(wins / n * 100, 1),
                    'avg_pnl_pct': round(avg_pnl_pct, 4),
                    'avg_pnl_usd': round(total_pnl / n, 2),
                    'total_pnl_usd': round(total_pnl, 2),
                    'np_count': np,
                    'np_pct': round(np / n * 100, 1),
                })
        return rows

    return {
        'longs': _crosstab_for('LONG'),
        'shorts': _crosstab_for('SHORT'),
        'pool_size': len(closed),
    }


def _compute_ema13_extension_pair_adx_crosstab(orders):
    """Cross-tab: Entry Extension × Pair ADX (May 24).

    Tests whether the extension failure modes are conditional on
    pair-level trend strength. Hypothesis: high extension + low pair ADX
    = chasing weak trend (high failure); high extension + high pair ADX
    = confirmed momentum (better odds).
    """
    closed = [o for o in orders
              if (o.status == 'CLOSED')
              and o.entry_dist_from_ema13_pct is not None
              and o.entry_adx is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    ext_buckets = [
        ('< 0%', -99, 0.0),
        ('0 to +0.20%', 0.0, 0.20),
        ('+0.20 to +0.40%', 0.20, 0.40),
        ('+0.40 to +0.60%', 0.40, 0.60),
        ('+0.60 to +0.80%', 0.60, 0.80),
        ('+0.80 to +1.00%', 0.80, 1.00),
        ('> +1.00%', 1.00, 99),
    ]
    # Pair ADX bins — match standard analytics
    adx_buckets = [
        ('< 15', 0, 15),
        ('15-18', 15, 18),
        ('18-22', 18, 22),
        ('22-25', 22, 25),
        ('25-28', 25, 28),
        ('28-30', 28, 30),
        ('30-33', 30, 33),
        ('> 33', 33, 99),
    ]

    def _crosstab_for(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for ext_label, ext_lo, ext_hi in ext_buckets:
            for adx_label, adx_lo, adx_hi in adx_buckets:
                if direction == 'LONG':
                    sub = [o for o in dir_orders
                           if ext_lo <= o.entry_dist_from_ema13_pct < ext_hi
                           and adx_lo <= o.entry_adx < adx_hi]
                else:
                    sub = [o for o in dir_orders
                           if ext_lo <= -o.entry_dist_from_ema13_pct < ext_hi
                           and adx_lo <= o.entry_adx < adx_hi]
                if not sub:
                    continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                total_pnl = sum(o.pnl or 0 for o in sub)
                avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'extension': ext_label,
                    'pair_adx': adx_label,
                    'count': n,
                    'win_rate': round(wins / n * 100, 1),
                    'avg_pnl_pct': round(avg_pnl_pct, 4),
                    'avg_pnl_usd': round(total_pnl / n, 2),
                    'total_pnl_usd': round(total_pnl, 2),
                    'np_count': np,
                    'np_pct': round(np / n * 100, 1),
                })
        return rows

    return {
        'longs': _crosstab_for('LONG'),
        'shorts': _crosstab_for('SHORT'),
        'pool_size': len(closed),
    }


# BTC Market Extension / BTC Late Regime Risk (May 14) — macro counterpart of pair extension.
# Tests whether losses cluster when BTC itself is stretched (price far from BTC EMA13),
# and especially when both BTC and pair are extended simultaneously (double-stretch).

def _compute_btc_extension_performance(orders):
    """Single-dim bucketing of BTC Market Extension by direction.

    Direction-aware (same as pair extension):
    - LONG: extension = entry_btc_dist_from_ema13_pct (positive = BTC above EMA13)
    - SHORT: extension = -entry_btc_dist_from_ema13_pct (positive = BTC below EMA13)

    Positive extension = BTC stretched within the move = market late-entry risk.
    """
    closed = [o for o in orders if (o.status == 'CLOSED') and getattr(o, 'entry_btc_dist_from_ema13_pct', None) is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    buckets = [
        ('< -0.20%', -99, -0.20, 'pullback'),
        ('-0.20 to -0.10%', -0.20, -0.10, 'pullback'),
        ('-0.10 to 0%', -0.10, 0.0, 'near mean'),
        ('0 to +0.10%', 0.0, 0.10, 'mild late'),
        ('+0.10 to +0.20%', 0.10, 0.20, 'late'),
        ('+0.20 to +0.40%', 0.20, 0.40, 'extended'),
        ('+0.40 to +0.60%', 0.40, 0.60, 'very extended'),
        ('> +0.60%', 0.60, 99, 'extreme'),
    ]

    def _bucket_for_direction(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for label, lo, hi, tag in buckets:
            if direction == 'LONG':
                sub = [o for o in dir_orders if lo <= o.entry_btc_dist_from_ema13_pct < hi]
            else:
                sub = [o for o in dir_orders if lo <= -o.entry_btc_dist_from_ema13_pct < hi]
            if not sub:
                continue
            n = len(sub)
            wins = sum(1 for o in sub if (o.pnl or 0) > 0)
            total_pnl = sum(o.pnl or 0 for o in sub)
            avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
            avg_peak = sum(o.peak_pnl or 0 for o in sub) / n
            doa = sum(1 for o in sub if (o.peak_pnl or 0) <= 0.10)
            np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
            conf = {}
            for o in sub:
                c = o.confidence or 'UNKNOWN'
                conf[c] = conf.get(c, 0) + 1
            rows.append({
                'range': label,
                'tag': tag,
                'count': n,
                'win_rate': round(wins / n * 100, 1),
                'avg_pnl_pct': round(avg_pnl_pct, 4),
                'avg_pnl_usd': round(total_pnl / n, 2),
                'total_pnl_usd': round(total_pnl, 2),
                'avg_peak_pct': round(avg_peak, 4),
                'doa_count': doa,
                'doa_pct': round(doa / n * 100, 1),
                'np_count': np,
                'np_pct': round(np / n * 100, 1),
                'by_confidence': conf,
            })
        return rows

    return {
        'longs': _bucket_for_direction('LONG'),
        'shorts': _bucket_for_direction('SHORT'),
        'pool_size': len(closed),
    }


def _compute_btc_extension_globalvol_crosstab(orders):
    """Cross-tab: BTC Market Extension × Global Volume Ratio.

    Market climax detector: BTC stretched + global volume spiking = whole market
    may be at exhaustion. Macro version of Pair Extension × Pair Volume Ratio.
    """
    closed = [o for o in orders if (o.status == 'CLOSED')
              and getattr(o, 'entry_btc_dist_from_ema13_pct', None) is not None
              and getattr(o, 'entry_global_volume_ratio', None) is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    ext_buckets = [
        ('< 0%', -99, 0.0),
        ('0 to +0.20%', 0.0, 0.20),
        ('+0.20 to +0.40%', 0.20, 0.40),
        ('+0.40 to +0.60%', 0.40, 0.60),
        ('> +0.60%', 0.60, 99),
    ]
    gvol_buckets = [
        ('< 0.85', 0.0, 0.85),
        ('0.85-1.00', 0.85, 1.00),
        ('1.00-1.20', 1.00, 1.20),
        ('> 1.20', 1.20, 99),
    ]

    def _crosstab_for(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for ext_label, ext_lo, ext_hi in ext_buckets:
            for gv_label, gv_lo, gv_hi in gvol_buckets:
                if direction == 'LONG':
                    sub = [o for o in dir_orders if ext_lo <= o.entry_btc_dist_from_ema13_pct < ext_hi and gv_lo <= o.entry_global_volume_ratio < gv_hi]
                else:
                    sub = [o for o in dir_orders if ext_lo <= -o.entry_btc_dist_from_ema13_pct < ext_hi and gv_lo <= o.entry_global_volume_ratio < gv_hi]
                if not sub:
                    continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                total_pnl = sum(o.pnl or 0 for o in sub)
                avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'btc_extension': ext_label,
                    'global_vol': gv_label,
                    'count': n,
                    'win_rate': round(wins / n * 100, 1),
                    'avg_pnl_pct': round(avg_pnl_pct, 4),
                    'avg_pnl_usd': round(total_pnl / n, 2),
                    'total_pnl_usd': round(total_pnl, 2),
                    'np_count': np,
                    'np_pct': round(np / n * 100, 1),
                })
        return rows

    return {
        'longs': _crosstab_for('LONG'),
        'shorts': _crosstab_for('SHORT'),
        'pool_size': len(closed),
    }


def _compute_btc_extension_pair_extension_crosstab(orders):
    """Cross-tab: BTC Extension × Pair Extension — DOUBLE-STRETCH DETECTOR.

    The highest-information-value cell of the new BTC dimension: are losses
    concentrated when BOTH market and pair are stretched simultaneously?
    """
    closed = [o for o in orders if (o.status == 'CLOSED')
              and getattr(o, 'entry_btc_dist_from_ema13_pct', None) is not None
              and getattr(o, 'entry_dist_from_ema13_pct', None) is not None]
    if not closed:
        return {"longs": [], "shorts": [], "pool_size": 0}

    btc_ext_buckets = [
        ('< 0%', -99, 0.0),
        ('0 to +0.20%', 0.0, 0.20),
        ('+0.20 to +0.40%', 0.20, 0.40),
        ('> +0.40%', 0.40, 99),
    ]
    pair_ext_buckets = [
        ('< 0%', -99, 0.0),
        ('0 to +0.20%', 0.0, 0.20),
        ('+0.20 to +0.40%', 0.20, 0.40),
        ('+0.40 to +0.60%', 0.40, 0.60),
        ('+0.60 to +0.80%', 0.60, 0.80),
        ('+0.80 to +1.00%', 0.80, 1.00),
        ('> +1.00%', 1.00, 99),
    ]

    def _crosstab_for(direction):
        rows = []
        dir_orders = [o for o in closed if o.direction == direction]
        for btc_label, btc_lo, btc_hi in btc_ext_buckets:
            for pair_label, pair_lo, pair_hi in pair_ext_buckets:
                if direction == 'LONG':
                    sub = [o for o in dir_orders
                           if btc_lo <= o.entry_btc_dist_from_ema13_pct < btc_hi
                           and pair_lo <= o.entry_dist_from_ema13_pct < pair_hi]
                else:
                    sub = [o for o in dir_orders
                           if btc_lo <= -o.entry_btc_dist_from_ema13_pct < btc_hi
                           and pair_lo <= -o.entry_dist_from_ema13_pct < pair_hi]
                if not sub:
                    continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                total_pnl = sum(o.pnl or 0 for o in sub)
                avg_pnl_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'btc_extension': btc_label,
                    'pair_extension': pair_label,
                    'count': n,
                    'win_rate': round(wins / n * 100, 1),
                    'avg_pnl_pct': round(avg_pnl_pct, 4),
                    'avg_pnl_usd': round(total_pnl / n, 2),
                    'total_pnl_usd': round(total_pnl, 2),
                    'np_count': np,
                    'np_pct': round(np / n * 100, 1),
                })
        return rows

    return {
        'longs': _crosstab_for('LONG'),
        'shorts': _crosstab_for('SHORT'),
        'pool_size': len(closed),
    }


# BTC 1h Slope (May 14) — higher-TF macro context. Three analytics surfaces:
# 1. Single-dim performance by 1h slope (signed buckets)
# 2. 5m × 1h slope alignment cross-tab (the diagnostic — Aligned/Opposite/Flat)
# 3. 1h slope × BTC ADX cross-tab

def _compute_btc_1h_slope_performance(orders):
    closed = [o for o in orders if o.status == 'CLOSED' and getattr(o, 'entry_btc_1h_slope', None) is not None]
    if not closed:
        return {'longs': [], 'shorts': [], 'pool_size': 0}
    # Finer granularity in both negative and positive ranges to surface the
    # actual inflection point (May 14 — earlier table lumped <-0.20% which
    # hid potential signal in the deep-bear zone).
    buckets = [
        ('< -0.40%', -99, -0.40),
        ('-0.40 to -0.30%', -0.40, -0.30),
        ('-0.30 to -0.20%', -0.30, -0.20),
        ('-0.20 to -0.15%', -0.20, -0.15),
        ('-0.15 to -0.10%', -0.15, -0.10),
        ('-0.10 to -0.05%', -0.10, -0.05),
        ('-0.05 to 0%', -0.05, 0.0),
        ('0 to +0.05%', 0.0, 0.05),
        ('+0.05 to +0.10%', 0.05, 0.10),
        ('+0.10 to +0.15%', 0.10, 0.15),
        ('+0.15 to +0.20%', 0.15, 0.20),
        ('+0.20 to +0.30%', 0.20, 0.30),
        ('+0.30 to +0.40%', 0.30, 0.40),
        ('> +0.40%', 0.40, 99),
    ]
    def _for(direction):
        rows = []
        dir_o = [o for o in closed if o.direction == direction]
        for lbl, lo, hi in buckets:
            sub = [o for o in dir_o if lo <= o.entry_btc_1h_slope < hi]
            if not sub: continue
            n = len(sub)
            wins = sum(1 for o in sub if (o.pnl or 0) > 0)
            tot = sum(o.pnl or 0 for o in sub)
            avg_pct = sum(o.pnl_percentage or 0 for o in sub) / n
            avg_peak = sum(o.peak_pnl or 0 for o in sub) / n
            np_ct = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
            rows.append({
                'range': lbl, 'count': n, 'win_rate': round(wins/n*100, 1),
                'avg_pnl_pct': round(avg_pct, 4), 'avg_pnl_usd': round(tot/n, 2),
                'total_pnl_usd': round(tot, 2), 'avg_peak_pct': round(avg_peak, 4),
                'np_count': np_ct, 'np_pct': round(np_ct/n*100, 1),
            })
        return rows
    return {'longs': _for('LONG'), 'shorts': _for('SHORT'), 'pool_size': len(closed)}


def _compute_btc_5m_1h_slope_alignment_crosstab(orders):
    """5m × 1h BTC slope alignment cross-tab.

    Categories:
    - Aligned: 5m and 1h slope same sign AND both magnitudes >= flat_threshold
    - Opposite: 5m and 1h slope opposite signs (both magnitudes >= flat_threshold)
    - 5m flat: |5m slope| < flat_threshold (1h drives direction)
    - 1h flat: |1h slope| < flat_threshold (5m drives, no macro context)
    - Both flat: both within flat_threshold

    flat_threshold = 0.02% (matches existing macro_trend_flat_threshold).
    """
    closed = [o for o in orders if o.status == 'CLOSED'
              and getattr(o, 'entry_btc_1h_slope', None) is not None
              and getattr(o, 'entry_btc_ema20_slope', None) is not None]
    if not closed:
        return {'longs': [], 'shorts': [], 'pool_size': 0}
    flat = 0.02

    def _classify(s5, s1):
        f5 = abs(s5) < flat
        f1 = abs(s1) < flat
        if f5 and f1: return 'Both flat'
        if f5: return f'5m flat / 1h {"up" if s1 > 0 else "down"}'
        if f1: return f'5m {"up" if s5 > 0 else "down"} / 1h flat'
        # Both directional
        if (s5 > 0 and s1 > 0): return 'Aligned UP'
        if (s5 < 0 and s1 < 0): return 'Aligned DOWN'
        if s5 < 0 and s1 > 0: return '5m DOWN / 1h UP (counter-trend)'
        return '5m UP / 1h DOWN (counter-trend)'

    def _for(direction):
        rows = []
        dir_o = [o for o in closed if o.direction == direction]
        buckets = {}
        for o in dir_o:
            cat = _classify(o.entry_btc_ema20_slope, o.entry_btc_1h_slope)
            buckets.setdefault(cat, []).append(o)
        # Preserve a consistent display order
        order = ['Aligned UP', 'Aligned DOWN', '5m UP / 1h DOWN (counter-trend)',
                 '5m DOWN / 1h UP (counter-trend)', '5m flat / 1h up', '5m flat / 1h down',
                 '5m up / 1h flat', '5m down / 1h flat', 'Both flat']
        for cat in order:
            sub = buckets.get(cat, [])
            if not sub: continue
            n = len(sub)
            wins = sum(1 for o in sub if (o.pnl or 0) > 0)
            tot = sum(o.pnl or 0 for o in sub)
            avg_pct = sum(o.pnl_percentage or 0 for o in sub) / n
            np_ct = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
            rows.append({
                'alignment': cat, 'count': n, 'win_rate': round(wins/n*100, 1),
                'avg_pnl_pct': round(avg_pct, 4), 'total_pnl_usd': round(tot, 2),
                'avg_pnl_usd': round(tot/n, 2),
                'np_count': np_ct, 'np_pct': round(np_ct/n*100, 1),
            })
        return rows
    return {'longs': _for('LONG'), 'shorts': _for('SHORT'), 'pool_size': len(closed)}


def _compute_btc_1h_30m_rsi_direction_crosstab(orders):
    """BTC 1h × 30m RSI Direction 2D cross-tab (May 26, 2026 evening).

    Surfaces multi-timeframe BTC RSI direction patterns invisible in
    single-dim tables. Per CLAUDE.md May 26 watchlist:
    - SHORT 1h Falling × 30m Rising = cleanest winner cell
    - SHORT 1h Falling × 30m Falling = dominant losing cohort (R:R asymmetry)
    - LONG 1h Rising × 30m Rising = dominant LONG losing cohort

    Direction classification: simple Rising/Falling (current > prev) with
    Flat only when exactly equal (rare). Matches dashboard convention.
    """
    closed = [o for o in orders if o.status == 'CLOSED'
              and getattr(o, 'entry_btc_rsi', None) is not None
              and getattr(o, 'entry_btc_rsi_prev6', None) is not None
              and getattr(o, 'entry_btc_rsi_1h', None) is not None
              and getattr(o, 'entry_btc_rsi_1h_prev', None) is not None]
    if not closed:
        return {'longs': [], 'shorts': [], 'pool_size': 0}

    def _cls(curr, prev):
        if curr > prev: return 'Rising'
        if curr < prev: return 'Falling'
        return 'Flat'

    def _for(direction):
        rows = []
        dir_o = [o for o in closed if o.direction == direction]
        buckets = {}
        for o in dir_o:
            d30 = _cls(o.entry_btc_rsi, o.entry_btc_rsi_prev6)
            d1h = _cls(o.entry_btc_rsi_1h, o.entry_btc_rsi_1h_prev)
            key = (d1h, d30)
            buckets.setdefault(key, []).append(o)
        # Display order: 1h Rising/Falling/Flat × 30m Rising/Falling/Flat
        order = [
            ('Rising', 'Rising'), ('Rising', 'Falling'), ('Rising', 'Flat'),
            ('Falling', 'Rising'), ('Falling', 'Falling'), ('Falling', 'Flat'),
            ('Flat', 'Rising'), ('Flat', 'Falling'), ('Flat', 'Flat'),
        ]
        for (d1h, d30) in order:
            sub = buckets.get((d1h, d30), [])
            if not sub: continue
            n = len(sub)
            wins = sum(1 for o in sub if (o.pnl or 0) > 0)
            tot = sum(o.pnl or 0 for o in sub)
            avg_pct = sum(o.pnl_percentage or 0 for o in sub) / n
            np_ct = sum(1 for o in sub if (o.peak_pnl or 0) <= 0.005)
            rows.append({
                'dir_1h': d1h, 'dir_30m': d30, 'count': n,
                'win_rate': round(wins/n*100, 1),
                'avg_pnl_pct': round(avg_pct, 4),
                'total_pnl_usd': round(tot, 2),
                'avg_pnl_usd': round(tot/n, 2),
                'np_count': np_ct, 'np_pct': round(np_ct/n*100, 1),
            })
        return rows
    return {'longs': _for('LONG'), 'shorts': _for('SHORT'), 'pool_size': len(closed)}


def _compute_btc_atr_adx_crosstab(orders):
    """BTC ATR × BTC ADX 2D cross-tab (May 22, 2026).
    Surfaces the SHORT killer cell (BTC ATR <0.10 × BTC ADX ≥30 = 3t/33% WR/-$159
    cross-batch) vs adjacent winner zones, and reveals the asymmetric LONG mirror
    (same cell wins for LONGs)."""
    closed = [o for o in orders if o.status == 'CLOSED'
              and getattr(o, 'entry_btc_atr_pct', None) is not None
              and o.entry_btc_adx is not None]
    if not closed:
        return {'longs': [], 'shorts': [], 'pool_size': 0}
    atr_bk = [('<0.10%', 0.0, 0.10), ('0.10-0.15%', 0.10, 0.15),
              ('0.15-0.20%', 0.15, 0.20), ('0.20-0.30%', 0.20, 0.30),
              ('≥0.30%', 0.30, 99)]
    adx_bk = [('<18', 0, 18), ('18-22', 18, 22), ('22-25', 22, 25),
              ('25-30', 25, 30), ('30-35', 30, 35), ('≥35', 35, 999)]
    def _for(direction):
        rows = []
        dir_o = [o for o in closed if o.direction == direction]
        for atrl, atrlo, atrhi in atr_bk:
            for axl, axlo, axhi in adx_bk:
                sub = [o for o in dir_o
                       if atrlo <= (o.entry_btc_atr_pct or 0) < atrhi
                       and axlo <= (o.entry_btc_adx or 0) < axhi]
                if not sub: continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                tot = sum(o.pnl or 0 for o in sub)
                avg_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np_ct = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'btc_atr': atrl, 'btc_adx': axl, 'count': n,
                    'win_rate': round(wins/n*100, 1),
                    'avg_pnl_pct': round(avg_pct, 4),
                    'total_pnl_usd': round(tot, 2),
                    'np_count': np_ct, 'np_pct': round(np_ct/n*100, 1),
                })
        return rows
    return {'longs': _for('LONG'), 'shorts': _for('SHORT'), 'pool_size': len(closed)}


def _compute_btc_1h_slope_adx_crosstab(orders):
    closed = [o for o in orders if o.status == 'CLOSED'
              and getattr(o, 'entry_btc_1h_slope', None) is not None
              and o.entry_btc_adx is not None]
    if not closed:
        return {'longs': [], 'shorts': [], 'pool_size': 0}
    slope_bk = [('< -0.10%', -99, -0.10), ('-0.10 to 0%', -0.10, 0.0),
                ('0 to +0.10%', 0.0, 0.10), ('+0.10 to +0.20%', 0.10, 0.20),
                ('> +0.20%', 0.20, 99)]
    adx_bk = [('15-20', 15, 20), ('20-25', 20, 25), ('25-30', 25, 30),
              ('30-35', 30, 35), ('≥35', 35, 99)]
    def _for(direction):
        rows = []
        dir_o = [o for o in closed if o.direction == direction]
        for sl, slo, shi in slope_bk:
            for al, alo, ahi in adx_bk:
                sub = [o for o in dir_o if slo <= o.entry_btc_1h_slope < shi and alo <= o.entry_btc_adx < ahi]
                if not sub: continue
                n = len(sub)
                wins = sum(1 for o in sub if (o.pnl or 0) > 0)
                tot = sum(o.pnl or 0 for o in sub)
                avg_pct = sum(o.pnl_percentage or 0 for o in sub) / n
                np_ct = sum(1 for o in sub if (o.peak_pnl or 0) <= 0)
                rows.append({
                    'slope_1h': sl, 'btc_adx': al, 'count': n,
                    'win_rate': round(wins/n*100, 1),
                    'avg_pnl_pct': round(avg_pct, 4),
                    'total_pnl_usd': round(tot, 2),
                    'np_count': np_ct, 'np_pct': round(np_ct/n*100, 1),
                })
        return rows
    return {'longs': _for('LONG'), 'shorts': _for('SHORT'), 'pool_size': len(closed)}


def _compute_pattern_c_validation(orders):
    """Pattern C Tracker validation (May 19, 2026 — observation-only).

    For each (pattern, direction) bucket, count matched trades and report
    WR / Avg P&L % / Total $. The locked promotion gate is N≥30 per
    pattern with WR ≤40% AND Avg P&L % ≤ -0.20% — at that point the
    pattern becomes a filter candidate.

    Patterns (mirror SHORT vs LONG):
      C1: Capitulation chase  — extreme range pos + deep pair gap + ADX accelerating
      C2: Macro counter-trend — BTC trend opposite trade direction
      C3: Stretch exhaustion  — high EMA5 stretch + high pair ADX + range extreme
      C4: Low-vol chop        — low BTC ATR + low BTC ADX + low pair ADX
      C5: Slow Climber Death  — weak ADX + slow accel + flat slope (May 19)
      C6: Macro over-extended — BTC same-direction climactic (May 19)
      C7: Pair Countertrend Bounce — pair stretched + slope-confirmed + mid-range (May 20)
      C8: Oversold/Overbought Chop — range extreme + sharp ADXΔ + no pair trend
          + low BTC vol (May 20-late, hypothesis from C4 deep-dive)
      C9: Low-vol Countertrend Chop — C4 base + MILD countertrend pair_gap
          (May 20-latest, "tight C4-LOSS" sub-pattern from EDEN deep-dive)

    Returns per-pattern rows + ANY-match row + per-direction TOTAL row.
    """
    patterns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c_any']
    # Batch P&L reference (May 20 latest+8): total $ of every closed trade in the batch
    # regardless of pattern match. Lets per-row "Batch If Shipped" column show what the
    # WHOLE batch P&L would become if THIS pattern's TP+SL fix were shipped alone.
    batch_actual_total_usd = sum((o.pnl or 0) for o in orders if o.status == 'CLOSED')
    pattern_labels = {
        'c1': 'C1 Capitulation chase',
        'c2': 'C2 Macro counter-trend',
        'c3': 'C3 Stretch exhaustion',
        'c4': 'C4 Low-vol chop',
        'c5': 'C5 Slow Climber Death',
        'c6': 'C6 Macro over-extended',
        'c7': 'C7 Pair Countertrend Bounce',
        'c8': 'C8 Oversold/Overbought Chop',
        'c9': 'C9 Low-vol Countertrend Chop',
        'c_any': 'ANY (C1∨…∨C9)',
    }
    # Helper: simulate fixed-TP outcome on a cohort.
    # For each trade: if peak >= tp_threshold, exit at tp_threshold (net %);
    # otherwise use actual pnl_percentage. Returns (sim_total$, fires, saves, kills).
    # Peak_pnl is net of fees per engine convention, so tp_threshold is net %.
    def _sim_tp_cohort(cohort, tp):
        sim_d = 0.0
        actual_d = 0.0
        fires = saves = kills = 0
        for o in cohort:
            nt = o.notional_value or ((o.investment or 0) * (o.leverage or 1))
            actual_pct = o.pnl_percentage or 0
            peak = o.peak_pnl
            actual_d += actual_pct / 100.0 * nt
            if peak is not None and peak >= tp:
                sim_pct = tp
                fires += 1
                if actual_pct <= 0:
                    saves += 1
                elif actual_pct > tp:
                    kills += 1
            else:
                sim_pct = actual_pct
            sim_d += sim_pct / 100.0 * nt
        return actual_d, sim_d, fires, saves, kills

    # Combined TP+SL counterfactual (May 20 latest+7): apply BOTH caps.
    # Uses peak_reached_at vs trough_reached_at to model which fires first.
    # If both would fire, sequence by timestamps; if no timestamps, favor TP
    # (conservative — TP firing is the safer assumption since it's a winner).
    # Returns (actual$, sim$, tp_fires, sl_fires, cut_winners).
    def _sim_combined_cohort(cohort, tp, sl):
        sim_d = 0.0
        actual_d = 0.0
        tp_fires = sl_fires = cut_winners = 0
        for o in cohort:
            nt = o.notional_value or ((o.investment or 0) * (o.leverage or 1))
            actual_pct = o.pnl_percentage or 0
            peak = o.peak_pnl
            trough = o.trough_pnl
            peak_at = getattr(o, 'peak_reached_at', None)
            trough_at = getattr(o, 'trough_reached_at', None)
            actual_d += actual_pct / 100.0 * nt
            tp_would_fire = peak is not None and peak >= tp
            sl_would_fire = trough is not None and trough <= -sl
            if tp_would_fire and sl_would_fire:
                # Both caps eligible — use timestamps to determine which fires first
                if peak_at and trough_at:
                    if peak_at <= trough_at:
                        sim_pct = tp
                        tp_fires += 1
                    else:
                        sim_pct = -sl
                        sl_fires += 1
                        if actual_pct > 0:
                            cut_winners += 1
                else:
                    # No timestamp data — assume TP fires (favorable assumption)
                    sim_pct = tp
                    tp_fires += 1
            elif tp_would_fire:
                sim_pct = tp
                tp_fires += 1
            elif sl_would_fire:
                sim_pct = -sl
                sl_fires += 1
                if actual_pct > 0:
                    cut_winners += 1
            else:
                sim_pct = actual_pct
            sim_d += sim_pct / 100.0 * nt
        return actual_d, sim_d, tp_fires, sl_fires, cut_winners

    # SL counterfactual (May 20 latest+6): cap losses at -sl_threshold% net.
    # For each trade with actual_pct < -sl_threshold → exit at -sl_threshold.
    # Cut-winner detection: trade whose TROUGH crossed -sl_threshold but actual
    # P&L > 0 — tighter SL would have stopped it before recovery. Mirrors the
    # kills metric on TP counterfactual.
    # Returns (actual$, sim$, fires, saves, cut_winners).
    def _sim_sl_cohort(cohort, sl_threshold):
        sim_d = 0.0
        actual_d = 0.0
        fires = saves = cut_winners = 0
        for o in cohort:
            nt = o.notional_value or ((o.investment or 0) * (o.leverage or 1))
            actual_pct = o.pnl_percentage or 0
            trough = o.trough_pnl
            actual_d += actual_pct / 100.0 * nt
            # Cut-winner: ended positive but trough touched the new SL
            if trough is not None and trough <= -sl_threshold and actual_pct > 0:
                cut_winners += 1
                sim_pct = -sl_threshold
                fires += 1
            elif actual_pct < -sl_threshold:
                # Loser worse than new SL → capped at new SL
                sim_pct = -sl_threshold
                fires += 1
                saves += 1
            else:
                sim_pct = actual_pct
            sim_d += sim_pct / 100.0 * nt
        return actual_d, sim_d, fires, saves, cut_winners

    rows = []
    for direction in ('LONG', 'SHORT'):
        for p in patterns:
            attr = f'entry_pattern_{p}_match'
            matched = [o for o in orders
                       if o.direction == direction
                       and getattr(o, attr, None) is True
                       and o.status == 'CLOSED']
            n = len(matched)
            if n == 0:
                continue
            wins = sum(1 for o in matched if (o.pnl or 0) > 0)
            wr = (wins / n * 100) if n > 0 else 0
            # Loser-precision: % of matches that were losers.
            # High loser_precision = pattern is a clean loser detector → filter candidate.
            # Low loser_precision = pattern mostly catches winners → blocking would kill them.
            loser_count = n - wins
            loser_precision_pct = (loser_count / n * 100) if n > 0 else 0
            total_usd = sum((o.pnl or 0) for o in matched)
            # R:R ratio (May 20 latest+5): avg loss $ / avg win $.
            # Lower = better. <2 green, 2-4 amber, >4 red.
            # Reveals the structural asymmetry — high WR with terrible R:R
            # can still net negative (e.g., 80% WR with 1:5 R:R = trap).
            win_sum = sum((o.pnl or 0) for o in matched if (o.pnl or 0) > 0)
            loss_sum_abs = sum(abs(o.pnl or 0) for o in matched if (o.pnl or 0) <= 0)
            avg_win_usd = (win_sum / wins) if wins > 0 else 0
            avg_loss_usd = (loss_sum_abs / loser_count) if loser_count > 0 else 0
            # rr_ratio = avg_loss / avg_win (1:X format — X is the multiple).
            # If no losses, RR is "perfect" (display as "1:0").
            # If no wins, RR is undefined (display as "—").
            if wins == 0:
                rr_ratio = None
            elif loser_count == 0:
                rr_ratio = 0.0  # pure winners — best possible R:R
            else:
                rr_ratio = avg_loss_usd / avg_win_usd if avg_win_usd > 0 else None
            pnls = [o.pnl_percentage for o in matched if o.pnl_percentage is not None]
            avg_pct = (sum(pnls) / len(pnls)) if pnls else 0
            avg_peak = None
            peaks = [o.peak_pnl for o in matched if o.peak_pnl is not None]
            if peaks:
                avg_peak = sum(peaks) / len(peaks)
            np_count = sum(1 for o in matched
                           if o.peak_pnl is not None and o.peak_pnl < 0.05)
            np_rate = (np_count / n * 100) if n > 0 else 0

            # TP counterfactual columns (May 20 — observation-only).
            # Shows what total $ would be if a fixed TP cap was applied to this
            # cohort at 0.05% and 0.10%. NP trades (peak <0.05%) cannot be
            # rescued by TP — only positive-peak trades that retraced get saved.
            # CLAUDE.md May 20-late entry locks decision matrix: ship TP when
            # cohort N≥30, TP-Δ$ ≥ +$80, AND kills ≤ saves/2.
            tp10_actual, tp10_sim, tp10_fires, tp10_saves, tp10_kills = _sim_tp_cohort(matched, 0.10)
            tp10_delta = tp10_sim - tp10_actual
            # Combined TP 0.10 + SL 0.50 counterfactual (May 20 latest+7)
            cb_actual, cb_sim, cb_tp_fires, cb_sl_fires, cb_cut = _sim_combined_cohort(matched, 0.10, 0.50)
            cb_delta = cb_sim - cb_actual

            # SL counterfactual columns (May 20 latest+6 — observation-only).
            # Shows what total $ would be if SL was tightened to -0.50% or -0.60%.
            # Mirror of TP counterfactual:
            #   fires = trades where new SL would activate
            #   saves = losers worse than new SL that get capped earlier
            #   cut_winners = winners whose trough crossed new SL (would have been killed pre-recovery)
            # Decision: ship SL tightening when cohort N≥30, SL-Δ$ ≥ +$100, AND
            # cut_winners ≤ saves/10 (low winner-kill collateral).
            sl50_actual, sl50_sim, sl50_fires, sl50_saves, sl50_cut = _sim_sl_cohort(matched, 0.50)
            sl50_delta = sl50_sim - sl50_actual
            # tp10_sim IS the new total (sum of sim$ across cohort)

            # Verdict per locked gates (CLAUDE.md May 20-latest+4 — BUG FIX:
            # all winner-verdicts now also require positive Avg P&L % AND positive
            # Total $. Previously a cohort with high WR but tiny wins and a few
            # big losses (e.g., 80% WR / +0.008% / -$28) was being labeled
            # "★ Winners cohort" — wrong because a multiplier on that would
            # amplify the net loss. Multipliers can't ship from cohorts that
            # lose money in absolute terms, regardless of how many trades won.
            #
            # The Pattern C tracker identifies BOTH directions of ship action:
            #   - High loser-precision + losing avg → FILTER candidate
            #   - High win-precision + WINNING avg + WINNING $ → MULTIPLIER candidate
            #   - Mixed or "high WR but net losing" → not actionable
            # Order matters: check the strict gates first, then weaker ones.
            if n >= 30 and wr >= 70 and avg_pct >= 0.10 and total_usd > 0:
                # May 20 latest+5: lowered Avg threshold from 0.20% → 0.10%.
                # With BE 0.10 floor the cohort avg can't mechanically reach
                # 0.20% — even 100% WR + avg win 0.10% only gives 0.10% avg.
                verdict = '★ MULTIPLIER CANDIDATE — meets winner ship gate (N≥30, WR≥70%, Avg≥+0.10%, Total$>0)'
            elif n >= 30 and wr <= 40 and avg_pct <= -0.20:
                verdict = '⚠ FILTER CANDIDATE — meets loser ship gate (N≥30, WR≤40%, Avg≤-0.20%)'
            elif n >= 10 and wr >= 60 and avg_pct > 0 and total_usd > 0:
                # Weak winner signal — high WR AND making money. Wait for N≥30.
                verdict = '★ Winners cohort — wait for N≥30 to promote to MULTIPLIER'
            elif n >= 10 and wr >= 60 and (avg_pct <= 0 or total_usd <= 0):
                # NEW verdict (May 20 latest+4): high WR but net losing.
                # Wins are too small or a few large losses dominate.
                # NOT a multiplier candidate — a multiplier would amplify the bleed.
                verdict = '⚠ High WR but net losing — wins too small or losses dominate (NOT MULTIPLIER candidate)'
            elif n >= 10 and wr <= 45 and avg_pct <= -0.15:
                verdict = '⚠ Warning — trending toward FILTER (wait for N≥30)'
            elif n < 10:
                verdict = '⚠ Low N'
            else:
                verdict = '✓ Inconclusive'
            rows.append({
                'direction': direction,
                'pattern': pattern_labels[p],
                'pattern_key': p,
                'n': n,
                'wr': round(wr, 1),
                'avg_pct': round(avg_pct, 3),
                'total_usd': round(total_usd, 2),
                'avg_peak_pct': round(avg_peak, 3) if avg_peak is not None else None,
                'np_count': np_count,
                'np_rate': round(np_rate, 1),
                'loser_count': loser_count,
                'loser_precision_pct': round(loser_precision_pct, 1),
                'avg_win_usd': round(avg_win_usd, 2),
                'avg_loss_usd': round(avg_loss_usd, 2),
                'rr_ratio': round(rr_ratio, 2) if rr_ratio is not None else None,
                # TP counterfactual columns
                'tp10_new_total_usd': round(tp10_sim, 2),
                'tp10_delta_usd': round(tp10_delta, 2),
                'tp10_fires': tp10_fires,
                'tp10_saves': tp10_saves,
                'tp10_kills': tp10_kills,
                # Combined TP+SL counterfactual (May 20 latest+7)
                'combined_new_total_usd': round(cb_sim, 2),
                'combined_delta_usd': round(cb_delta, 2),
                'combined_tp_fires': cb_tp_fires,
                'combined_sl_fires': cb_sl_fires,
                'combined_cut_winners': cb_cut,
                # SL counterfactual columns (May 20 latest+6)
                'sl50_new_total_usd': round(sl50_sim, 2),
                'sl50_delta_usd': round(sl50_delta, 2),
                'sl50_fires': sl50_fires,
                'sl50_saves': sl50_saves,
                'sl50_cut_winners': sl50_cut,
                # Batch P&L reference (May 20 latest+8): single-pattern ship projection.
                # batch_actual is constant on every row (whole-batch P&L); batch_if_shipped
                # shows what whole-batch P&L would become if this pattern's TP+SL fix
                # shipped alone (using combined_delta as the realistic dual-cap delta).
                'batch_actual_total_usd': round(batch_actual_total_usd, 2),
                'batch_if_shipped_usd': round(batch_actual_total_usd + cb_delta, 2),
                'verdict': verdict,
            })
    return rows


def _compute_pattern_w_match(o):
    """Pattern W (winner tracker) signature matching — computed at report time.

    Unlike Pattern C which stores per-trade match flags on Order, Pattern W
    evaluates signatures from already-captured entry features. This is
    LIGHTER than Pattern C's design (no schema, no engine touchpoints, no
    migration). When a Pattern W signature graduates to multiplier ship, we'd
    add per-trade capture for that specific signature at that point.

    Pattern W signatures — designed FROM cross-batch winner analysis (May 20),
    targeting the multiplier-candidate framework:
      W1: HighConv trend continuation — strong ADX + accel + stretch
      W2: Macro tailwind — BTC RSI in sweet spot + BTC ADX committed + gap aligned
      W3: Energetic volatility breakout — BTC ATR high + above-avg pair vol + stretch
      W4: Pullback entry aligned — mid-range + pair gap aligned + ADX not declining
      W5: Confluence — multiple "sweet spot" cells all true simultaneously

    Cross-batch baseline (May 20 latest+2, deduped pool May 4 → today):
      LONG winners: 136 / 252 (54.0% baseline WR)
      SHORT winners: 155 / 258 (60.1% baseline WR)
      Strongest LONG: W2 mod (BTC RSI 55-65 + BTC ADX 22-30) → 71.2% / N=66 ★
      Strongest SHORT: W4 (pullback aligned) → 71.4% / N=14 (small)

    Returns (w1, w2, w3, w4, w5, w_any) booleans.
    Direction-aware: LONG and SHORT use mirrored thresholds.
    """
    direction = o.direction
    if direction not in ('LONG', 'SHORT'):
        return (False, False, False, False, False, False)

    # Safe extraction
    rsi = o.entry_rsi
    adx = o.entry_adx
    adx_delta = getattr(o, 'entry_adx_delta', None)
    stretch = o.entry_ema5_stretch
    rng_pos = getattr(o, 'entry_range_position', None)
    pair_gap = getattr(o, 'entry_pair_ema20_ema50_gap_pct', None)
    btc_rsi = o.entry_btc_rsi
    btc_adx = o.entry_btc_adx
    btc_atr = getattr(o, 'entry_btc_atr_pct', None)
    btc_gap = getattr(o, 'entry_btc_trend_gap_pct', None)
    pair_vol = getattr(o, 'entry_pair_volume_ratio', None)

    def _and(*conds):
        return all(c is True for c in conds)

    if direction == 'LONG':
        w1 = _and(
            adx is not None and adx >= 22,
            adx_delta is not None and adx_delta >= 0.5,
            stretch is not None and stretch >= 0.16,
        )
        w2 = _and(
            btc_rsi is not None and 50 <= btc_rsi <= 65,
            btc_adx is not None and btc_adx >= 22,
            btc_gap is not None and btc_gap >= 0.10,
        )
        w3 = _and(
            btc_atr is not None and btc_atr >= 0.20,
            pair_vol is not None and pair_vol >= 1.20,
            stretch is not None and stretch >= 0.20,
        )
        w4 = _and(
            rng_pos is not None and 40 <= rng_pos <= 75,
            pair_gap is not None and pair_gap >= 0.10,
            adx_delta is not None and adx_delta >= 0,
        )
        w5 = _and(
            btc_adx is not None and 22 <= btc_adx <= 30,
            btc_rsi is not None and 55 <= btc_rsi <= 65,
            adx is not None and 22 <= adx <= 30,
            stretch is not None and 0.16 <= stretch <= 0.25,
        )
    else:  # SHORT
        w1 = _and(
            adx is not None and adx >= 22,
            adx_delta is not None and adx_delta >= 0.5,
            stretch is not None and stretch >= 0.20,
        )
        w2 = _and(
            btc_rsi is not None and 30 <= btc_rsi <= 45,
            btc_adx is not None and btc_adx >= 22,
            btc_gap is not None and btc_gap <= -0.10,
        )
        w3 = _and(
            btc_atr is not None and btc_atr >= 0.20,
            pair_vol is not None and pair_vol >= 1.20,
            stretch is not None and stretch >= 0.25,
        )
        w4 = _and(
            rng_pos is not None and 25 <= rng_pos <= 60,
            pair_gap is not None and pair_gap <= -0.10,
            adx_delta is not None and adx_delta >= 0,
        )
        w5 = _and(
            btc_adx is not None and 22 <= btc_adx <= 30,
            btc_rsi is not None and 30 <= btc_rsi <= 40,
            adx is not None and 22 <= adx <= 30,
            stretch is not None and 0.20 <= stretch <= 0.30,
        )

    # W6 added May 21 (late) — "Mature BTC Bear" SHORT / "Healthy BTC Tailwind" LONG.
    # Cross-batch validated 100% WR on N=25 SHORT / N=14 LONG (post-May-15 pool).
    # Mirror of engine helper in services/trading_engine.py::_compute_pattern_w_match.
    if direction == 'LONG':
        w6 = _and(
            btc_adx is not None and 22 <= btc_adx < 26,
            pair_gap is not None and pair_gap < 0.20,
        )
    else:
        w6 = _and(
            btc_adx is not None and btc_adx >= 32,
        )

    return (w1, w2, w3, w4, w5, w6, (w1 or w2 or w3 or w4 or w5 or w6))


def _compute_pattern_w_validation(orders):
    """Pattern W tracker analytics (May 20 latest+2 — observation-only).

    Symmetric to Pattern C but for WINNER patterns. Returns per (pattern × direction)
    rows with the same column structure as Pattern C tracker:
      - N, WR, AvgP&L%, Total$, AvgPeak%, NP rate
      - Winner-precision (% of matches that won) — mirrored from Loser %
      - Same verdict tiers as Pattern C but optimized for multiplier promotion

    Locked promotion gates (CLAUDE.md May 20 latest+1):
      ★ MULTIPLIER CANDIDATE: N≥30 AND WR≥70% AND Avg P&L%≥+0.20%
      (Cross-batch stability check required before actual ship — see CLAUDE.md)
    """
    patterns = ['w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w_any']
    pattern_labels = {
        'w1': 'W1 HighConv trend',
        'w2': 'W2 Macro tailwind',
        'w3': 'W3 Energetic volatility',
        'w4': 'W4 Pullback aligned',
        'w5': 'W5 Confluence',
        'w6': 'W6 BTC Tailwind/Bear',
        'w_any': 'ANY (W1∨…∨W6)',
    }
    # Batch P&L reference (May 20 latest+8): mirrors Pattern C — total $ of every
    # closed trade in the batch. Per-row "Batch If Shipped" = batch_actual +
    # mult20_delta (single-row 2.0× multiplier ship projection).
    batch_actual_total_usd = sum((o.pnl or 0) for o in orders if o.status == 'CLOSED')

    # Cache match results per order (compute once)
    order_matches = {}
    for o in orders:
        if o.status == 'CLOSED' and o.direction in ('LONG', 'SHORT'):
            w1, w2, w3, w4, w5, w6, w_any = _compute_pattern_w_match(o)
            order_matches[id(o)] = {
                'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w_any': w_any
            }

    rows = []
    for direction in ('LONG', 'SHORT'):
        for p in patterns:
            matched = [o for o in orders
                       if o.direction == direction
                       and o.status == 'CLOSED'
                       and id(o) in order_matches
                       and order_matches[id(o)][p]]
            n = len(matched)
            if n == 0:
                continue
            wins = sum(1 for o in matched if (o.pnl or 0) > 0)
            losses = n - wins
            wr = (wins / n * 100) if n > 0 else 0
            # Winner-precision: % of matches that won (the metric that matters for
            # multiplier candidates). This is just WR by another name but kept
            # explicit for parity with Pattern C's Loser %.
            winner_count = wins
            winner_precision_pct = (winner_count / n * 100) if n > 0 else 0
            # Loser % column (May 20 latest+5) added to Pattern W for symmetry
            # with Pattern C — although mathematically just 100 - Win%, having
            # both visible side by side helps operator at a glance.
            loser_precision_pct = 100 - winner_precision_pct if n > 0 else 0
            total_usd = sum((o.pnl or 0) for o in matched)
            # R:R ratio (May 20 latest+5): same logic as Pattern C
            win_sum = sum((o.pnl or 0) for o in matched if (o.pnl or 0) > 0)
            loss_sum_abs = sum(abs(o.pnl or 0) for o in matched if (o.pnl or 0) <= 0)
            avg_win_usd = (win_sum / wins) if wins > 0 else 0
            avg_loss_usd = (loss_sum_abs / losses) if losses > 0 else 0
            if wins == 0:
                rr_ratio = None
            elif losses == 0:
                rr_ratio = 0.0
            else:
                rr_ratio = avg_loss_usd / avg_win_usd if avg_win_usd > 0 else None
            pnls = [o.pnl_percentage for o in matched if o.pnl_percentage is not None]
            avg_pct = (sum(pnls) / len(pnls)) if pnls else 0
            avg_peak = None
            peaks = [o.peak_pnl for o in matched if o.peak_pnl is not None]
            if peaks:
                avg_peak = sum(peaks) / len(peaks)
            np_count = sum(1 for o in matched
                           if o.peak_pnl is not None and o.peak_pnl < 0.05)
            np_rate = (np_count / n * 100) if n > 0 else 0

            # Multiplier counterfactual (May 20 latest+3): at 2.0×, what's the new
            # cohort total $? Multiplier scales every trade in cohort linearly.
            # Mirror of Pattern C's TP CF columns but with no fires/saves/kills
            # concept — multiplier always applies to every match, just scales them.
            # Chose 2.0× over 1.5× because 2.0× is the hard cap value already in
            # the multiplier system — shows the maximum ship-decision scenario.
            mult_20 = 2.0
            mult20_new_total_usd = sum((o.pnl or 0) * mult_20 for o in matched)
            mult20_delta_usd = mult20_new_total_usd - total_usd

            # Verdict per locked gate (May 20 latest+5: lowered Avg threshold from
            # 0.20% to 0.10% because with BE 0.10 floor the cohort avg can never
            # mechanically reach 0.20% — even 100% WR with avg win 0.10% would
            # only give 0.10% avg. 0.10% is the realistic ceiling for "winner
            # cohort makes money" given current exit stack.)
            # May 20 latest+4 BUG FIX still holds: all winner verdicts require
            # positive Avg AND positive Total $.
            if n >= 30 and wr >= 70 and avg_pct >= 0.10 and total_usd > 0:
                verdict = '★ MULTIPLIER CANDIDATE — meets winner ship gate (N≥30, WR≥70%, Avg≥+0.10%, Total$>0)'
            elif n >= 10 and wr >= 60 and avg_pct > 0 and total_usd > 0:
                # Real winner signal: high WR AND making money.
                verdict = '★ Winners cohort — wait for N≥30 to promote'
            elif n >= 10 and wr >= 60 and (avg_pct <= 0 or total_usd <= 0):
                # NEW: high WR but net-losing cohort. Wins are tiny or losses
                # dominate. NOT a multiplier candidate.
                verdict = '⚠ High WR but net losing — wins too small or losses dominate (NOT MULTIPLIER candidate)'
            elif n >= 10 and wr <= 45:
                verdict = '✗ Anti-pattern (winners cohort showing losses) — drop from tracker'
            elif n < 10:
                verdict = '⚠ Low N'
            else:
                verdict = '✓ Inconclusive'

            rows.append({
                'direction': direction,
                'pattern': pattern_labels[p],
                'pattern_key': p,
                'n': n,
                'wr': round(wr, 1),
                'winner_count': winner_count,
                'loser_count': losses,
                'winner_precision_pct': round(winner_precision_pct, 1),
                'loser_precision_pct': round(loser_precision_pct, 1),
                'avg_win_usd': round(avg_win_usd, 2),
                'avg_loss_usd': round(avg_loss_usd, 2),
                'rr_ratio': round(rr_ratio, 2) if rr_ratio is not None else None,
                'avg_pct': round(avg_pct, 3),
                'total_usd': round(total_usd, 2),
                'avg_peak_pct': round(avg_peak, 3) if avg_peak is not None else None,
                'np_count': np_count,
                'np_rate': round(np_rate, 1),
                # Multiplier counterfactual (2.0×)
                'mult20_new_total_usd': round(mult20_new_total_usd, 2),
                'mult20_delta_usd': round(mult20_delta_usd, 2),
                # Batch P&L reference (May 20 latest+8): single-pattern multiplier ship.
                # batch_actual constant on every row; batch_if_shipped = what whole-batch
                # P&L becomes if this single pattern's 2.0× multiplier shipped alone.
                'batch_actual_total_usd': round(batch_actual_total_usd, 2),
                'batch_if_shipped_usd': round(batch_actual_total_usd + mult20_delta_usd, 2),
                'verdict': verdict,
            })
    return rows


def _compute_pattern_w_batch_coverage(orders):
    """Pattern W batch-level winner coverage (May 20 latest+3 — observation).

    Symmetric to Pattern C batch coverage but for WINNERS.

    For each direction, compute:
      - Total CLOSED trades
      - Winners + Losers count
      - How many winners match ANY W1..W5 signature (covered)
      - How many winners are OUTSIDE all W signatures (gap — discovery surface)
      - Coverage %

    Coverage answers: "Is Pattern W catching most winners, or are we missing
    winning signatures?" If coverage drops materially below current, the
    Unmatched Winners Deep Dive table surfaces what's outside.
    """
    out = {}
    for direction in ('LONG', 'SHORT'):
        closed = [o for o in orders if o.direction == direction and o.status == 'CLOSED']
        winners = [o for o in closed if (o.pnl or 0) > 0]
        losers = [o for o in closed if (o.pnl or 0) <= 0]

        def has_any_w(o):
            _, _, _, _, _, _, w_any = _compute_pattern_w_match(o)
            return w_any

        winners_in = [o for o in winners if has_any_w(o)]
        winners_out = [o for o in winners if not has_any_w(o)]

        coverage = (100 * len(winners_in) / max(len(winners), 1)) if winners else None

        out[direction] = {
            'total': len(closed),
            'winners': len(winners),
            'losers': len(losers),
            'winners_in_w': len(winners_in),
            'winners_outside_w': len(winners_out),
            'coverage_pct': round(coverage, 1) if coverage is not None else None,
        }
    return out


def _compute_unmatched_winners(orders, limit=20):
    """Unmatched Winners Deep Dive (May 20 latest+3 — observation, mirror of unmatched losers).

    Lists CLOSED winners that did NOT match any W1..W5 signature, with their
    entry conditions. These are winning trades our W framework is BLIND to.

    When ≥3 unmatched winners in a batch share a recognizable entry signature
    (e.g., all in a specific RngPos×BTC ADX combination), that's the discovery
    signal: define a new pattern (W6, W7, ...) and add to tracker.

    Sorted by $ gain magnitude (largest winners first). Capped at `limit` rows.
    """
    rows = []
    for o in orders:
        if o.status != 'CLOSED':
            continue
        if (o.pnl or 0) <= 0:
            continue  # losers excluded — this is the WINNERS table

        # Skip if any W pattern matched
        _, _, _, _, _, _, w_any = _compute_pattern_w_match(o)
        if w_any:
            continue

        rows.append({
            'pair': o.pair,
            'direction': o.direction,
            'opened_at': o.opened_at.isoformat() if o.opened_at else None,
            'pnl': round(o.pnl or 0, 2),
            'peak_pnl': round(o.peak_pnl, 3) if o.peak_pnl is not None else None,
            'pnl_percentage': round(o.pnl_percentage, 3) if o.pnl_percentage is not None else None,
            'entry_rsi': round(o.entry_rsi, 1) if o.entry_rsi is not None else None,
            'entry_adx': round(o.entry_adx, 1) if o.entry_adx is not None else None,
            'entry_adx_delta': round(o.entry_adx_delta, 2) if getattr(o, 'entry_adx_delta', None) is not None else None,
            'entry_ema5_stretch': round(o.entry_ema5_stretch, 3) if o.entry_ema5_stretch is not None else None,
            'entry_range_position': round(o.entry_range_position, 1) if getattr(o, 'entry_range_position', None) is not None else None,
            'entry_pair_ema20_ema50_gap_pct': round(o.entry_pair_ema20_ema50_gap_pct, 3) if getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) is not None else None,
            'entry_btc_rsi': round(o.entry_btc_rsi, 1) if o.entry_btc_rsi is not None else None,
            'entry_btc_adx': round(o.entry_btc_adx, 1) if o.entry_btc_adx is not None else None,
            'entry_btc_atr_pct': round(o.entry_btc_atr_pct, 3) if getattr(o, 'entry_btc_atr_pct', None) is not None else None,
            'entry_btc_trend_gap_pct': round(o.entry_btc_trend_gap_pct, 3) if getattr(o, 'entry_btc_trend_gap_pct', None) is not None else None,
            'close_reason': o.close_reason or '',
        })

    # Sort by $ gain (best first), then cap
    rows.sort(key=lambda r: -r['pnl'])
    return rows[:limit]


def _compute_pattern_c_batch_coverage(orders):
    """Pattern C batch-level loser coverage (May 20 late — observation).

    For each direction, compute:
      - Total CLOSED trades
      - Winners + Losers count
      - How many losers match ANY C1..C8 signature (covered)
      - How many losers are OUTSIDE all C signatures (gap)
      - Coverage %

    Coverage answers the question: "Is Pattern C catching most losers, or
    are we missing failure modes?" If coverage drops below ~70% across
    a batch, that's the signal to define new C9/C10 signatures.

    Returns {LONG: {...}, SHORT: {...}}.
    """
    out = {}
    for direction in ('LONG', 'SHORT'):
        closed = [o for o in orders if o.direction == direction and o.status == 'CLOSED']
        winners = [o for o in closed if (o.pnl or 0) > 0]
        losers = [o for o in closed if (o.pnl or 0) <= 0]

        def has_any_c(o):
            for k in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'):
                if getattr(o, f'entry_pattern_{k}_match', None) is True:
                    return True
            return False

        losers_in = [o for o in losers if has_any_c(o)]
        losers_out = [o for o in losers if not has_any_c(o)]

        coverage = (100 * len(losers_in) / max(len(losers), 1)) if losers else None

        out[direction] = {
            'total': len(closed),
            'winners': len(winners),
            'losers': len(losers),
            'losers_in_c': len(losers_in),
            'losers_outside_c': len(losers_out),
            'coverage_pct': round(coverage, 1) if coverage is not None else None,
        }
    return out


def _compute_unmatched_losers(orders, limit=20):
    """Unmatched-Losers Deep Dive (May 20 late — observation).

    Lists CLOSED losers that did NOT match any C1..C8 signature, with
    their entry conditions. These are losers our current pattern framework
    is BLIND to.

    When ≥3 unmatched losers in a batch share a recognizable entry
    signature (e.g., all small-cap + high stretch + RngPos extreme),
    that's the discovery signal: define a new pattern (C9, C10, ...).

    Sorted by $ loss magnitude (worst first). Capped at `limit` rows.
    """
    rows = []
    for o in orders:
        if o.status != 'CLOSED':
            continue
        if (o.pnl or 0) > 0:
            continue  # winners excluded

        # Skip if any C pattern matched
        has_any = False
        for k in ('c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'):
            if getattr(o, f'entry_pattern_{k}_match', None) is True:
                has_any = True
                break
        if has_any:
            continue

        rows.append({
            'pair': o.pair,
            'direction': o.direction,
            'opened_at': o.opened_at.isoformat() if o.opened_at else None,
            'pnl': round(o.pnl or 0, 2),
            'peak_pnl': round(o.peak_pnl, 3) if o.peak_pnl is not None else None,
            'pnl_percentage': round(o.pnl_percentage, 3) if o.pnl_percentage is not None else None,
            'entry_rsi': round(o.entry_rsi, 1) if o.entry_rsi is not None else None,
            'entry_adx': round(o.entry_adx, 1) if o.entry_adx is not None else None,
            'entry_adx_delta': round(o.entry_adx_delta, 2) if getattr(o, 'entry_adx_delta', None) is not None else None,
            'entry_ema5_stretch': round(o.entry_ema5_stretch, 3) if o.entry_ema5_stretch is not None else None,
            'entry_range_position': round(o.entry_range_position, 1) if getattr(o, 'entry_range_position', None) is not None else None,
            'entry_pair_ema20_ema50_gap_pct': round(o.entry_pair_ema20_ema50_gap_pct, 3) if getattr(o, 'entry_pair_ema20_ema50_gap_pct', None) is not None else None,
            'entry_btc_rsi': round(o.entry_btc_rsi, 1) if o.entry_btc_rsi is not None else None,
            'entry_btc_adx': round(o.entry_btc_adx, 1) if o.entry_btc_adx is not None else None,
            'entry_btc_atr_pct': round(o.entry_btc_atr_pct, 3) if getattr(o, 'entry_btc_atr_pct', None) is not None else None,
            'entry_btc_trend_gap_pct': round(o.entry_btc_trend_gap_pct, 3) if getattr(o, 'entry_btc_trend_gap_pct', None) is not None else None,
            'close_reason': o.close_reason or '',
        })

    # Sort by $ loss magnitude (worst first), then cap
    rows.sort(key=lambda r: r['pnl'])
    return rows[:limit]




# ===================== LEASH SHADOW START (May 30, 2026 — observation-only) =====================
def _compute_leash_shadow(orders):
    """Leash Shadow Tracker report. Compares virtual trailing leashes (tight/wide/
    tierA/tierB) vs the actual exit on the runner-profile cohort, sliced by
    Direction x stretch-bucket. Observation-only — see CLAUDE.md May 30.
    TO REMOVE: delete this fenced block + the payload line + the UI block."""
    ACT = 0.45  # armed threshold — matches live tp_min=0.45 (was 0.5, stale)
    LEASHES = [
        ('actual', None, None, None),
        ('tight', 'shadow_tight_pnl', 'shadow_tight_reason', 'shadow_tight_min'),
        ('wide', 'shadow_wide_pnl', 'shadow_wide_reason', 'shadow_wide_min'),
        ('tierA', 'shadow_tierA_pnl', 'shadow_tierA_reason', 'shadow_tierA_min'),
        ('tierB', 'shadow_tierB_pnl', 'shadow_tierB_reason', 'shadow_tierB_min'),
        ('strpk', 'shadow_strpk_pnl', 'shadow_strpk_reason', 'shadow_strpk_min'),  # stretch-trail K=0.5
        ('strpk04', 'shadow_strpk04_pnl', 'shadow_strpk04_reason', 'shadow_strpk04_min'),  # K=0.4 (looser)
        ('strpk03', 'shadow_strpk03_pnl', 'shadow_strpk03_reason', 'shadow_strpk03_min'),  # K=0.3 (loosest)
        ('stren', 'shadow_stren_pnl', 'shadow_stren_reason', 'shadow_stren_min'),  # stretch-to-entry
    ]
    # armed + shadow-populated cohort
    rows = [o for o in orders
            if getattr(o, 'status', None) == 'CLOSED'
            and (getattr(o, 'peak_pnl', 0) or 0) >= ACT
            and getattr(o, 'shadow_tight_pnl', None) is not None]
    BUCKETS = [
        ('LONG ≥0.25 (GATE)', 'LONG', 0.25, 999.0, True, False),
        ('LONG <0.25 (control)', 'LONG', -999.0, 0.25, False, False),
        ('SHORT ≥0.25', 'SHORT', 0.25, 999.0, False, False),
        ('SHORT <0.25', 'SHORT', -999.0, 0.25, False, False),
        (' LONG 0.25-0.40', 'LONG', 0.25, 0.40, False, True),
        (' LONG 0.40-0.60', 'LONG', 0.40, 0.60, False, True),
        (' LONG 0.60+', 'LONG', 0.60, 999.0, False, True),
    ]
    slices = []
    drill = []
    for label, dir_, lo, hi, is_gate, is_sub in BUCKETS:
        coh = [o for o in rows if o.direction == dir_
               and getattr(o, 'entry_ema5_stretch', None) is not None
               and lo <= o.entry_ema5_stretch < hi]
        if not coh:
            continue
        actual_total = sum((o.pnl_percentage or 0) for o in coh)
        actual_avg_close = round(actual_total / len(coh), 3) if coh else 0.0
        # actual-row "Min" = avg trade duration (open→real close). Each leash's fire-min vs
        # this = pre-close (fired earlier) if <, post-close (held the runner) if >.
        _durs = [(o.closed_at - o.opened_at).total_seconds() / 60.0 for o in coh if o.closed_at and o.opened_at]
        actual_dur_min = round(sum(_durs) / len(_durs), 1) if _durs else None
        leash_rows = []
        for name, pcol, rcol, mcol in LEASHES:
            evals = []
            for o in coh:
                ev = (o.pnl_percentage if name == 'actual' else getattr(o, pcol, None))
                if ev is not None:
                    evals.append(ev)
            n = len(evals)
            total = sum(evals)
            avg = total / n if n else 0.0
            clean = trap = slconv = 0
            cwins, ctraps, evs, mps = [], [], [], []
            for o in coh:
                ev = (o.pnl_percentage if name == 'actual' else getattr(o, pcol, None))
                av = o.pnl_percentage
                if name != 'actual' and ev is not None and av is not None:
                    d = ev - av
                    if d > 0.1:
                        clean += 1; cwins.append(d)
                    elif d < -0.1:
                        trap += 1; ctraps.append(d)
                    if (getattr(o, rcol, '') or '') == 'hard_sl':
                        slconv += 1
                mp = max(o.peak_pnl or 0, o.post_exit_peak_pnl or 0)
                if mp > 0 and ev is not None:
                    mps.append(mp); evs.append(ev)
            # $ counterfactual: cf_$ = (leash% / 100) × notional_value (matches pnl=pct×notional/100)
            # matched_actual_pct = actual P&L% over ONLY the trades this leash covers — so an
            # under-covered leash (fewer trades than the slice, e.g. a post-deploy-only variant on a
            # mixed cohort) is compared apples-to-apples, not vs the full-slice actual (which produced
            # phantom negatives / "✗ hurts"). Consistent with how delta_usd is already matched.
            cf_usd = 0.0
            act_usd_m = 0.0
            matched_actual_pct = 0.0
            for o in coh:
                ev = (o.pnl_percentage if name == 'actual' else getattr(o, pcol, None))
                if ev is None:
                    continue
                if name == 'actual':
                    cf_usd += (o.pnl or 0.0)
                else:
                    cf_usd += (ev / 100.0) * (getattr(o, 'notional_value', 0) or 0.0)
                act_usd_m += (o.pnl or 0.0)
                matched_actual_pct += (o.pnl_percentage or 0.0)
            delta = total - matched_actual_pct
            delta_usd = cf_usd - act_usd_m
            # avg fire-minute (from open); actual row = avg trade duration
            if name == 'actual':
                fire_min = actual_dur_min
            else:
                _fm = [getattr(o, mcol, None) for o in coh if getattr(o, mcol, None) is not None]
                fire_min = round(sum(_fm) / len(_fm), 1) if _fm else None
            post_close = (name != 'actual' and fire_min is not None
                          and actual_dur_min is not None and fire_min > actual_dur_min)
            pct_max = (sum(evs) / sum(mps) * 100) if mps else 0.0
            # no_data: leash has no populated trades in this slice (e.g. post-deploy-only
            # variant whose cohort trades predate it) — don't score it as a real verdict.
            no_data = (name != 'actual' and n == 0)
            if name == 'actual':
                verdict = 'baseline'
            elif no_data:
                verdict = '⏳ no data'
            elif name == 'tight':
                verdict = '✓ sim valid' if abs(delta) <= max(0.5, 0.1 * n) else '⚠ sim off'
            elif delta > 0.5 and clean >= 2 * max(trap, 1):
                verdict = '★ helps'
            elif delta < -0.5:
                verdict = '✗ hurts'
            else:
                verdict = '⚠ marginal'
            leash_rows.append({
                'leash': name, 'n': n, 'avg': round(avg, 3), 'total': round(total, 2),
                'actual_avg_close': actual_avg_close, 'no_data': no_data,
                'fire_min': fire_min, 'post_close': post_close,
                'act_usd': round(act_usd_m, 2), 'cf_usd': round(cf_usd, 2), 'delta_usd': round(delta_usd, 2),
                'delta': round(delta, 2), 'clean': clean, 'trap': trap,
                'avg_clean': round(sum(cwins) / len(cwins), 2) if cwins else 0.0,
                'avg_trap': round(sum(ctraps) / len(ctraps), 2) if ctraps else 0.0,
                'sl_conv': slconv, 'pct_max': round(pct_max, 0), 'verdict': verdict,
            })
        slices.append({'slice': label, 'is_gate': is_gate, 'is_sub': is_sub, 'rows': leash_rows})
        # per-trade drill for the gate slice only
        if is_gate:
            for o in sorted(coh, key=lambda x: -(x.post_exit_peak_pnl or 0))[:15]:
                drill.append({
                    'pair': o.pair, 'stretch': round(o.entry_ema5_stretch or 0, 2),
                    'pkstr': round(o.shadow_peak_stretch, 2) if o.shadow_peak_stretch is not None else None,
                    'actual': round(o.pnl_percentage or 0, 2),
                    'tierA': round(o.shadow_tierA_pnl, 2) if o.shadow_tierA_pnl is not None else None,
                    'tierB': round(o.shadow_tierB_pnl, 2) if o.shadow_tierB_pnl is not None else None,
                    'strpk': round(o.shadow_strpk_pnl, 2) if o.shadow_strpk_pnl is not None else None,
                    'stren': round(o.shadow_stren_pnl, 2) if o.shadow_stren_pnl is not None else None,
                    'pxpk': round(o.post_exit_peak_pnl or 0, 2),
                })
    return {'slices': slices, 'drill': drill, 'cohort_n': len(rows)}
# ====================== LEASH SHADOW END ======================


# Fast-exit counterfactual thresholds + windows (May 13 — Option A analytics).
# (Restored May 31 — these module-level constants were accidentally swallowed
#  when the adjacent Phantom-BE / Time-to-L1 functions were retired.)
_FAST_EXIT_THRESHOLDS = [0.20, 0.30, 0.40]   # P&L % triggers to test (May 17: shifted from [0.10,0.15,0.20] now that 0.20 is live)
_FAST_EXIT_WINDOWS = [1, 2, 5]               # minutes from entry (3min dropped May 13)
_FAST_EXIT_DEFAULT_CELL = (0.20, 2)          # cell for close-reason breakdown
_FAST_EXIT_FEE_PCT = 0.063                   # taker round-trip fee approx


def _compute_fast_exit_counterfactual(orders):
    """Fast-exit counterfactual grid (May 13).

    For each (threshold, window) pair: identify trades whose peak P&L reached
    >= threshold within `window` minutes of entry. For those trades, the
    hypothetical 'fast-exit' would have locked in `threshold - fee`. Compare
    portfolio swing vs actual outcomes.

    Returns three sections:
      - grid_rows: 3 thresholds x 4 windows aggregate metrics
      - direction_split: same grid but split LONG/SHORT to surface asymmetry
      - close_reason_breakdown: for the default cell, what close reasons would fire

    May 18 extension: when a trade closed early (e.g. live FAST_EXIT fired at
    0.20%), in-trade peak is capped at that close. To honestly evaluate higher
    thresholds (0.30%, 0.40%), we now consult post-exit snapshots
    (`post_exit_pnl_at_{1,2,5,15,30}min`) and `post_exit_peak_pnl` — but only
    the portion that falls inside `window_min` from ENTRY, not from close. This
    keeps each (threshold, window) cell directly comparable.

    Pool caveat: spans multiple configs across the date range. Net% is leverage-
    invariant; raw $ is approximate. Use Net% for cross-batch comparison.
    """
    # Build a working list with timing
    pool = []
    for o in orders:
        if getattr(o, 'status', None) != 'CLOSED':
            continue
        peak = getattr(o, 'peak_pnl', None)
        peak_at = getattr(o, 'peak_reached_at', None)
        opened = getattr(o, 'opened_at', None)
        if peak is None or peak_at is None or opened is None:
            continue
        try:
            peak_min = (peak_at - opened).total_seconds() / 60.0
        except Exception:
            continue
        # May 18: capture post-exit data to extend synthetic peak past live close.
        closed_at = getattr(o, 'closed_at', None)
        hold_min = None
        if closed_at is not None and opened is not None:
            try:
                hold_min = (closed_at - opened).total_seconds() / 60.0
            except Exception:
                hold_min = None
        post_exit_snaps = []  # list of (offset_min_from_close, pnl_pct)
        for snap_min in (1, 2, 5, 15, 30):
            val = getattr(o, f'post_exit_pnl_at_{snap_min}min', None)
            if val is not None:
                post_exit_snaps.append((float(snap_min), float(val)))
        post_exit_peak_pnl = getattr(o, 'post_exit_peak_pnl', None)
        post_exit_peak_min = getattr(o, 'post_exit_peak_minutes', None)
        direction = (o.direction.value if hasattr(o.direction, 'value') else o.direction) or 'LONG'
        pool.append({
            'pair': o.pair,
            'direction': direction,
            'pnl_pct': o.pnl_percentage or 0.0,
            'pnl_usd': o.pnl or 0.0,
            'peak_pct': peak,
            'peak_min': peak_min,
            'hold_min': hold_min,
            'post_exit_snaps': post_exit_snaps,
            'post_exit_peak_pnl': post_exit_peak_pnl,
            'post_exit_peak_min': post_exit_peak_min,
            'investment': o.investment or 0.0,
            'leverage': o.leverage or 1,
            'close_reason': (o.close_reason or 'UNKNOWN').split(' L')[0],
        })

    if not pool:
        return {'grid_rows': [], 'direction_split': [], 'close_reason_breakdown': None,
                'caveat': 'No trades with timing data available.'}

    n_total = len(pool)

    def synthetic_peak_in_window(t, window_min):
        """Return the highest P&L observed within window_min of ENTRY,
        consulting in-trade peak plus post-exit snapshots inside the budget.

        Returns the synthetic peak %. Always includes in-trade peak if it
        occurred within the window (which it always does — peak is between
        open and close, so peak_min <= hold_min).
        """
        candidates = []
        # In-trade peak — only counts if reached within window
        if t['peak_min'] is not None and t['peak_min'] <= window_min:
            candidates.append(t['peak_pct'])
        # If trade closed before window expired, consult post-exit data
        hold_min = t.get('hold_min')
        if hold_min is None or hold_min >= window_min:
            return max(candidates) if candidates else None
        budget_remaining = window_min - hold_min
        # post_exit_peak_pnl: post-exit window highest, with timing
        pe_peak = t.get('post_exit_peak_pnl')
        pe_peak_min = t.get('post_exit_peak_min')
        if pe_peak is not None and pe_peak_min is not None and pe_peak_min <= budget_remaining:
            candidates.append(pe_peak)
        # Snapshots at fixed offsets — include those inside budget
        for offset, val in t.get('post_exit_snaps') or []:
            if offset <= budget_remaining:
                candidates.append(val)
        return max(candidates) if candidates else None

    def cell_metrics(threshold, window_min, sub_pool):
        """Compute fired-trades metrics for a single (threshold, window) cell.

        May 18: uses synthetic_peak_in_window so trades that exceeded threshold
        AFTER live close (within window from entry) are correctly counted.
        """
        fired = []
        for t in sub_pool:
            syn = synthetic_peak_in_window(t, window_min)
            if syn is not None and syn >= threshold:
                fired.append(t)
        winners = [t for t in fired if t['pnl_pct'] > 0]
        losers = [t for t in fired if t['pnl_pct'] <= 0]
        # Pct-based metrics: leverage-invariant
        give_up_pct = sum(t['pnl_pct'] - threshold for t in winners)  # negative-ish ($)
        saved_pct = sum(threshold - t['pnl_pct'] for t in losers)     # positive
        net_pct = saved_pct - give_up_pct
        # $-based: approximate using investment * leverage and (threshold - fee)
        cf_per_trade_usd = lambda t: t['investment'] * t['leverage'] * (threshold - _FAST_EXIT_FEE_PCT) / 100.0
        real_dollars = sum(t['pnl_usd'] for t in fired)
        cf_dollars = sum(cf_per_trade_usd(t) for t in fired)
        return {
            'fired': fired,
            'winners': winners,
            'losers': losers,
            'give_up_pct': give_up_pct,
            'saved_pct': saved_pct,
            'net_pct': net_pct,
            'real_dollars': real_dollars,
            'cf_dollars': cf_dollars,
            'delta_dollars': cf_dollars - real_dollars,
        }

    # Pool-wide totals — denominators for Win Fire% and Los Fire%
    total_winners = sum(1 for t in pool if t['pnl_pct'] > 0)
    total_losers = sum(1 for t in pool if t['pnl_pct'] <= 0)

    # Build grid (aggregate across direction)
    grid_rows = []
    for thr in _FAST_EXIT_THRESHOLDS:
        for win in _FAST_EXIT_WINDOWS:
            m = cell_metrics(thr, win, pool)
            n_fire = len(m['fired'])
            n_winners_cut = len(m['winners'])
            n_losers_saved = len(m['losers'])
            pct_fire = n_fire / n_total * 100 if n_total else 0
            # New: hit-rate against each population. Stable denominators
            # (pool-wide totals), so values are comparable across rows.
            win_fire_pct = n_winners_cut / total_winners * 100 if total_winners else 0
            los_fire_pct = n_losers_saved / total_losers * 100 if total_losers else 0
            # Verdict
            if m['delta_dollars'] > 0:
                verdict = 'positive'
            elif m['delta_dollars'] < 0:
                verdict = 'negative'
            else:
                verdict = 'neutral'
            grid_rows.append({
                'threshold_pct': thr,
                'window_min': win,
                'n_fire': n_fire,
                'pct_fire': round(pct_fire, 1),  # kept for backwards-compat
                'win_fire_pct': round(win_fire_pct, 1),
                'los_fire_pct': round(los_fire_pct, 1),
                'n_winners_cut': n_winners_cut,
                'n_losers_saved': n_losers_saved,
                'give_up_pct': round(-m['give_up_pct'], 2),   # display as negative
                'saved_pct': round(m['saved_pct'], 2),
                'net_pct': round(m['net_pct'], 2),
                'real_dollars': round(m['real_dollars'], 2),
                'cf_dollars': round(m['cf_dollars'], 2),
                'delta_dollars': round(m['delta_dollars'], 2),
                'verdict': verdict,
            })

    # Mark best Net% and best Δ$ cells
    if grid_rows:
        max_net = max(r['net_pct'] for r in grid_rows)
        max_delta = max(r['delta_dollars'] for r in grid_rows)
        for r in grid_rows:
            r['is_best_net'] = (r['net_pct'] == max_net)
            r['is_best_dollars'] = (r['delta_dollars'] == max_delta)

    # Direction split
    longs = [t for t in pool if t['direction'] == 'LONG']
    shorts = [t for t in pool if t['direction'] == 'SHORT']
    direction_split = []
    for thr in _FAST_EXIT_THRESHOLDS:
        for win in _FAST_EXIT_WINDOWS:
            ml = cell_metrics(thr, win, longs)
            ms = cell_metrics(thr, win, shorts)
            # Verdict per-direction
            long_pos = ml['delta_dollars'] > 0
            short_pos = ms['delta_dollars'] > 0
            if long_pos and short_pos:
                v = 'OK both'
            elif long_pos and not short_pos and len(ms['fired']) > 0:
                v = 'LONG only'
            elif long_pos and len(ms['fired']) == 0:
                v = 'LONG only (no SHORT data)'
            elif not long_pos and short_pos:
                v = 'SHORT only'
            else:
                v = 'Negative both'
            direction_split.append({
                'threshold_pct': thr,
                'window_min': win,
                'long_n': len(ml['fired']),
                'long_delta_dollars': round(ml['delta_dollars'], 2),
                'short_n': len(ms['fired']),
                'short_delta_dollars': round(ms['delta_dollars'], 2),
                'verdict': v,
            })

    # Close-reason breakdown for ALL cells (lets UI swap which cell is shown
    # without re-fetching). Keyed by f"{thr}_{win}". Empty cells still listed
    # with rows=[] so the UI doesn't show "no data" confusingly.
    def _breakdown_for_cell(thr, win):
        m = cell_metrics(thr, win, pool)
        by_cr = {}
        for t in m['fired']:
            cr = t['close_reason']
            by_cr.setdefault(cr, []).append(t)
        cr_rows = []
        cf_per_trade_usd = lambda t: t['investment'] * t['leverage'] * (thr - _FAST_EXIT_FEE_PCT) / 100.0
        for cr, trs in sorted(by_cr.items(), key=lambda kv: -len(kv[1])):
            n = len(trs)
            wins = sum(1 for t in trs if t['pnl_pct'] > 0)
            avg_close = sum(t['pnl_pct'] for t in trs) / n
            avg_peak = sum(t['peak_pct'] for t in trs) / n
            wr = wins / n * 100
            real_dollars = sum(t['pnl_usd'] for t in trs)
            cf_dollars = sum(cf_per_trade_usd(t) for t in trs)
            delta_dollars = cf_dollars - real_dollars
            if wr >= 50:
                effect = 'GIVES UP'
                per_trade_pct = round(-(avg_close - thr), 3)
            else:
                effect = 'SAVES'
                per_trade_pct = round(thr - avg_close, 3)
            cr_rows.append({
                'close_reason': cr,
                'n': n,
                'actual_wr': round(wr, 0),
                'avg_close_pct': round(avg_close, 3),
                'avg_peak_pct': round(avg_peak, 3),
                'effect': effect,
                'per_trade_pct': per_trade_pct,
                'real_dollars': round(real_dollars, 2),
                'cf_dollars': round(cf_dollars, 2),
                'delta_dollars': round(delta_dollars, 2),
            })
        return cr_rows

    breakdowns = {}
    for thr in _FAST_EXIT_THRESHOLDS:
        for win in _FAST_EXIT_WINDOWS:
            key = f"{thr:.2f}_{win}"
            breakdowns[key] = _breakdown_for_cell(thr, win)

    default_thr, default_win = _FAST_EXIT_DEFAULT_CELL
    default_key = f"{default_thr:.2f}_{default_win}"

    return {
        'grid_rows': grid_rows,
        'direction_split': direction_split,
        'close_reason_breakdowns': breakdowns,            # all cells
        'default_breakdown_key': default_key,
        'available_thresholds': _FAST_EXIT_THRESHOLDS,
        'available_windows': _FAST_EXIT_WINDOWS,
        'pool_total': n_total,
        'pool_total_winners': total_winners,
        'pool_total_losers': total_losers,
        # Backwards-compat: keep the default breakdown under the old name
        'close_reason_breakdown': {
            'threshold_pct': default_thr,
            'window_min': default_win,
            'rows': breakdowns.get(default_key, []),
        },
        'caveat': 'May 18 extension — consults post-exit snapshots (post_exit_pnl_at_*min, post_exit_peak_pnl) capped at window-from-entry. Pre-May-13 trades lack post-exit data and use in-trade peak only. Net% is leverage-invariant; raw $ approximate.',
    }


def _compute_ema13_strict_performance(orders):
    """EMA13 strict-mode counterfactual tracking (May 8, 2026).

    For trades where strict mode held at least one EMA13 cross exit
    (`ema13_strict_held_pnl_pct IS NOT NULL`), compares the P&L at the
    first hold point vs the actual close P&L. Positive delta = strict
    mode helped (price recovered after the wick); negative delta = strict
    mode hurt (price kept declining).

    Returns rows per direction + a TOTAL row with verdict.
    """
    by_dir = {"LONG": [], "SHORT": []}
    for o in orders:
        if getattr(o, 'status', None) != "CLOSED":
            continue
        held = getattr(o, 'ema13_strict_held_pnl_pct', None)
        if held is None:
            continue
        d = (o.direction.value if hasattr(o.direction, 'value') else o.direction) or "LONG"
        if d not in by_dir:
            continue
        close_pct = o.pnl_percentage if o.pnl_percentage is not None else 0.0
        delta_pct = close_pct - held
        # $ delta: ratio close $ / close % × delta %, falls back to 0 when close% is 0
        if close_pct and abs(close_pct) > 1e-9 and o.pnl is not None:
            dollar_delta = o.pnl * (delta_pct / close_pct)
        else:
            # Fallback: approximate via investment * leverage
            inv = o.investment or 0.0
            lev = o.leverage or 1
            dollar_delta = (delta_pct / 100.0) * inv * lev
        by_dir[d].append({
            'held_pct': held,
            'close_pct': close_pct,
            'delta_pct': delta_pct,
            'dollar_delta': dollar_delta,
            'pnl': o.pnl or 0.0,
        })

    def _verdict(rows):
        if not rows:
            return "⚠ Low N"
        n = len(rows)
        avg_delta = sum(r['delta_pct'] for r in rows) / n
        total_dollar = sum(r['dollar_delta'] for r in rows)
        if n < 5:
            return "⚠ Low N"
        if avg_delta >= 0.05 and total_dollar > 0:
            return "★ HELPING"
        if avg_delta <= -0.05 or total_dollar < 0:
            return "⚠ HURTING"
        return "✓ Marginal"

    rows_out = []
    total_delta_pct = 0.0
    total_dollar = 0.0
    total_n = 0
    for d in ("LONG", "SHORT"):
        rs = by_dir[d]
        n = len(rs)
        if n == 0:
            continue
        avg_held = sum(r['held_pct'] for r in rs) / n
        avg_close = sum(r['close_pct'] for r in rs) / n
        avg_delta = sum(r['delta_pct'] for r in rs) / n
        sum_dollar = sum(r['dollar_delta'] for r in rs)
        rows_out.append({
            'direction': d,
            'n': n,
            'avg_held_pct': round(avg_held, 4),
            'avg_close_pct': round(avg_close, 4),
            'avg_delta_pct': round(avg_delta, 4),
            'total_dollar_delta': round(sum_dollar, 2),
            'verdict': _verdict(rs),
        })
        total_delta_pct += avg_delta * n
        total_dollar += sum_dollar
        total_n += n
    summary = {
        'n': total_n,
        'avg_delta_pct': round(total_delta_pct / total_n, 4) if total_n > 0 else 0.0,
        'total_dollar_delta': round(total_dollar, 2),
    }
    return {'rows': rows_out, 'summary': summary}


def _compute_multiplier_cell_performance(orders):
    """Premium Multiplier tracking — per CLAUDE.md May 3 design.

    Groups CLOSED orders by cell_multiplier_source and reports per-source:
      Source / Multi / N / WR / Avg P&L% / Total$ / Expect$/tr / PF / BL Avg% / Δ vs BL / Capped / Verdict

    Baseline (BL) = the same direction's overall Avg P&L% from trades that did NOT
    fire a multiplier rule (i.e., cell_multiplier == 1.0). This isolates the
    cell's effect from baseline regime drag.

    Verdict (per CLAUDE.md May 3 framework):
      ★ WORKING — cell Avg P&L% ≥ baseline AND total $ positive AND N ≥ 5
      ⚠ Low N — N < 5 (insufficient data, no verdict)
      ✓ Marginal — within ±0.10pp of baseline
      ⚠ DRAG — materially below baseline (cell hurt under leverage)
      ✗ HARMFUL — total $ negative (cell broke under leverage; revert immediately)
    """
    closed = [o for o in orders if (o.status == 'CLOSED')]
    if not closed:
        return {"longs": [], "shorts": [], "summary": {}}

    # May 13: parse CURRENT config multipliers so we can flag cells that have been
    # demoted (historical trades stamped 2.0× but config now 1.0×).
    def _parse_rules(s, prefix):
        out = {}
        if not s: return out
        for rule in s.split(','):
            rule = rule.strip()
            if not rule: continue
            parts = rule.split(':')
            if prefix in ('STRETCH', 'SCORE') and len(parts) == 2:
                # 1D rule format: "<lo>-<hi>:<multiplier>" (no second dimension)
                # STRETCH retired May 15 PM; SCORE shipped May 18 (new dimension).
                out[f"{prefix}_{parts[0]}"] = float(parts[1])
            elif len(parts) == 3:
                # 2D rule format (legacy 3-part): "<dim1_lo>-<dim1_hi>:<dim2_lo>-<dim2_hi>:<invest_mult>"
                # → leverage_mult implicit 1.0 under old configs.
                out[f"{prefix}_{parts[0]}_{parts[1]}"] = float(parts[2])
            elif len(parts) == 4:
                # 2D rule format (May 21 extended): adds leverage_mult as 4th field.
                # For current-config diff we still key on the invest_mult value (the
                # historical-cell-perf table compares against past trades which only
                # stored cell_multiplier=invest side pre-May-21). Leverage side is
                # tracked separately via cell_lev_multiplier on the Order row.
                out[f"{prefix}_{parts[0]}_{parts[1]}"] = float(parts[2])
        return out

    try:
        th = config.trading_config.thresholds
        cur_pair_long = _parse_rules(getattr(th, 'rsi_adx_multiplier_long', ''), 'PAIR')
        cur_pair_short = _parse_rules(getattr(th, 'rsi_adx_multiplier_short', ''), 'PAIR')
        cur_btc_long = _parse_rules(getattr(th, 'btc_rsi_adx_multiplier_long', ''), 'BTC')
        cur_btc_short = _parse_rules(getattr(th, 'btc_rsi_adx_multiplier_short', ''), 'BTC')
        # SCORE-based multipliers retired May 21 (see CLAUDE.md removal entry).
        # STRETCH-based multiplier retired May 15 PM. Historical SCORE_* / STRETCH_*
        # sources in cell_multiplier_source still render in the Multiplier Cell
        # Performance table from past-config diffs — they show with no "current
        # config" map (i.e., as historical-only rows).
    except Exception:
        cur_pair_long = cur_pair_short = cur_btc_long = cur_btc_short = {}

    def _direction_baseline(direction):
        """Avg P&L% of NON-multiplied trades for this direction."""
        base_trades = [o for o in closed
                       if o.direction == direction and (o.cell_multiplier or 1.0) == 1.0
                       and o.pnl_percentage is not None]
        if not base_trades:
            return None
        return sum(o.pnl_percentage for o in base_trades) / len(base_trades)

    def _bucket_for_direction(direction):
        baseline = _direction_baseline(direction)
        # Pick the appropriate current-config map for this direction
        if direction == 'LONG':
            current_map = {**cur_pair_long, **cur_btc_long}
        else:
            current_map = {**cur_pair_short, **cur_btc_short}
        # Group by source (NULL = baseline 1.0×).
        # May 21: also track cell_lev_multiplier alongside cell_multiplier (invest side).
        # Effective multiplier (used for Δ$ math) = inv × lev.
        # Legacy trades pre-May-21 have cell_lev_multiplier=1.0 (default).
        groups = {}
        for o in closed:
            if o.direction != direction:
                continue
            inv_mult = o.cell_multiplier if o.cell_multiplier is not None else 1.0
            lev_mult = getattr(o, 'cell_lev_multiplier', None)
            if lev_mult is None:
                lev_mult = 1.0
            src = o.cell_multiplier_source or '[Default 1.0x]'
            groups.setdefault(src, {'inv_multi': inv_mult, 'lev_multi': lev_mult, 'orders': []})['orders'].append(o)

        rows = []
        for src, data in groups.items():
            os_ = data['orders']
            n = len(os_)
            if n == 0:
                continue
            wins = [o for o in os_ if (o.pnl or 0) > 0]
            losses = [o for o in os_ if (o.pnl or 0) <= 0]
            wr = 100.0 * len(wins) / n
            avg_pct = sum((o.pnl_percentage or 0) for o in os_) / n
            total_d = sum((o.pnl or 0) for o in os_)
            expect_d = total_d / n
            win_d = sum((o.pnl or 0) for o in wins)
            loss_d = abs(sum((o.pnl or 0) for o in losses))
            pf = (win_d / loss_d) if loss_d > 0 else (float('inf') if win_d > 0 else 0)
            capped = sum(1 for o in os_ if getattr(o, 'cell_multiplier_capped', False))

            is_default = (src == '[Default 1.0x]')
            inv_mult = data['inv_multi']
            lev_mult = data['lev_multi']
            # May 21: effective multiplier for Δ$ math = inv × lev.
            # In "investment" mode, lev=1.0 so effective = inv (no change vs pre-May-21).
            # In "leverage" mode, inv=1.0 so effective = lev.
            # In "both" mode, effective = inv × lev (compounding).
            effective_mult = inv_mult * lev_mult
            # Δ$ vs Baseline (May 4 redesign per user feedback): dollar impact of the
            # multiplier vs same trades at baseline (1.0× effective multiplier).  Formula:
            #   delta_dollars = total_$ × (1 − 1/effective_multiplier)
            # This isolates the multiplier's $-contribution.  Independent of confidence-
            # level leverage — both actual and counterfactual use the same leverage.
            # Positive Δ$ = multiplier extracted positive boost.  Negative Δ$ = multiplier
            # amplified losses.
            if is_default or effective_mult == 1.0:
                delta_dollars = 0.0
            else:
                delta_dollars = total_d * (1.0 - 1.0 / effective_mult)

            if is_default:
                verdict = '(baseline reference)'
            else:
                if n < 5:
                    verdict = '⚠ Low N'
                elif total_d < 0:
                    verdict = '✗ HARMFUL'
                elif delta_dollars > 1.0:
                    verdict = '★ WORKING'
                elif abs(delta_dollars) <= 1.0:
                    verdict = '✓ Marginal'
                else:
                    verdict = '⚠ DRAG'

            # May 13: current-config multiplier (for demotion flag) — keyed on
            # invest-side value (matches the historical cell_multiplier column).
            current_mult = current_map.get(src) if not is_default else None
            is_demoted = (current_mult is not None and current_mult < inv_mult)
            rows.append({
                'source': src,
                # Backward-compat: 'multiplier' field still present (= inv side) so older
                # UI/exports don't break.  New 'inv_multiplier' / 'lev_multiplier' / 'effective_multiplier'
                # added May 21 — use these in updated table renderers.
                'multiplier': round(inv_mult, 2),
                'inv_multiplier': round(inv_mult, 2),
                'lev_multiplier': round(lev_mult, 2),
                'effective_multiplier': round(effective_mult, 2),
                'current_config_multiplier': round(current_mult, 2) if current_mult is not None else None,
                'is_demoted': is_demoted,
                'n': n,
                'wr_pct': round(wr, 1),
                'avg_pnl_pct': round(avg_pct, 4),
                'total_dollars': round(total_d, 2),
                'expect_per_trade': round(expect_d, 3),
                'profit_factor': round(pf, 2) if pf != float('inf') else 999.99,
                'baseline_avg_pct': round(baseline, 4) if baseline is not None else None,
                'delta_vs_baseline_dollars': round(delta_dollars, 2),
                'capped_count': capped,
                'verdict': verdict,
            })
        # Sort: rules first (by effective_multiplier desc), default last
        rows.sort(key=lambda r: (r['source'] == '[Default 1.0x]', -r['effective_multiplier']))
        return rows

    longs = _bucket_for_direction('LONG')
    shorts = _bucket_for_direction('SHORT')

    # Summary uplift line — per CLAUDE.md May 3 spec.  May 4 redesign:
    # rename "1x" → "baseline" terminology to avoid confusion with leverage
    # (multiplier is independent of confidence-level leverage).
    def _uplift(rows):
        # May 21: a row is "boosted" if its EFFECTIVE multiplier (inv × lev) ≠ 1.0.
        # This correctly captures "both" mode rows where inv=1 but lev=2 (or vice versa).
        def _eff(r):
            return r.get('effective_multiplier') or r.get('multiplier') or 1.0
        boosted = [r for r in rows if r['source'] != '[Default 1.0x]' and _eff(r) != 1.0]
        actual_total = sum(r['total_dollars'] for r in boosted)
        # Counterfactual at baseline (effective_mult=1.0) = total / effective_mult (linear scaling).
        # "Baseline" here means "what the same trades would have made if cell multiplier
        # was 1.0 on both sides", regardless of confidence-level leverage.
        sim_baseline = sum(r['total_dollars'] / _eff(r) for r in boosted if _eff(r))
        return {
            'multiplied_trades_n': sum(r['n'] for r in boosted),
            'actual_total_dollars': round(actual_total, 2),
            'simulated_baseline_dollars': round(sim_baseline, 2),
            'uplift_dollars': round(actual_total - sim_baseline, 2),
        }

    return {
        'longs': longs,
        'shorts': shorts,
        'summary': {'longs': _uplift(longs), 'shorts': _uplift(shorts)},
    }


def _compute_pattern_4cohort_coverage(orders):
    """4-Cohort Pattern Coverage (May 21 late ship per CLAUDE.md).

    Bucket every CLOSED trade by C-match × W-match into 4 cohorts:
      1. BOTH (matches at least one C AND at least one W) — crossed cohort
      2. C only (matches at least one C but no W) — pure loser-shape signature
      3. W only (matches at least one W but no C) — pure winner-shape signature
      4. TRULY UNMATCHED (matches NO C and NO W) — outside the entire taxonomy

    The framing fix: the original "Unmatched Loser" / "Unmatched Winner"
    classification (loser without C, winner without W) conflated TRULY
    unmatched trades with crossed trades. The crossover analysis (cross-batch
    pool) showed:
      - 50% of Unm.L cross-batch ALSO match a W pattern → they're "Both"
      - 63% of Unm.W cross-batch ALSO match a C pattern → they're "Both"

    This table separates those concerns cleanly. The TRULY UNMATCHED cohort
    is the genuine residual outside the C1-C9 + W1-W6 taxonomy.

    Returns dict with 'cohorts' (list of 4 cohort rows) + 'total' summary.
    Each cohort row includes N, W, L, WR%, Total $, Avg $/tr, and
    direction split.
    """
    closed = [o for o in orders if o.status == 'CLOSED' and o.pnl is not None]
    if not closed:
        return {'cohorts': [], 'total': {}}

    def _c_flags(o):
        s = set()
        for i in range(1, 10):
            if getattr(o, f'entry_pattern_c{i}_match', None) is True:
                s.add(f'C{i}')
        return s

    def _w_flags(o):
        # DB flags (only populated for trades opened after Phase 1 engine ship,
        # CLAUDE.md May 21 late evening). For older trades, fall back to
        # post-hoc computation via _compute_pattern_w_match so the cohort
        # bucketing stays correct across the entire historical window.
        s = set()
        any_populated = any(
            getattr(o, f'entry_pattern_w{i}_match', None) is True
            for i in range(1, 7)
        )
        if any_populated:
            for i in range(1, 7):
                if getattr(o, f'entry_pattern_w{i}_match', None) is True:
                    s.add(f'W{i}')
        else:
            # post-hoc fallback: compute from raw entry features
            try:
                w1, w2, w3, w4, w5, w6, _ = _compute_pattern_w_match(o)
                if w1: s.add('W1')
                if w2: s.add('W2')
                if w3: s.add('W3')
                if w4: s.add('W4')
                if w5: s.add('W5')
                if w6: s.add('W6')
            except Exception:
                pass
        return s

    cohorts = {'both': [], 'c_only': [], 'w_only': [], 'truly_unmatched': []}
    for o in closed:
        cm = _c_flags(o)
        wm = _w_flags(o)
        if cm and wm:
            cohorts['both'].append(o)
        elif cm:
            cohorts['c_only'].append(o)
        elif wm:
            cohorts['w_only'].append(o)
        else:
            cohorts['truly_unmatched'].append(o)

    def stat(cohort):
        n = len(cohort)
        if n == 0:
            return {'n': 0, 'w': 0, 'l': 0, 'wr_pct': 0.0, 'total_dollars': 0.0,
                    'avg_dollars_per_trade': 0.0, 'pct_of_batch': 0.0,
                    'long_n': 0, 'long_w': 0, 'long_total': 0.0,
                    'short_n': 0, 'short_w': 0, 'short_total': 0.0}
        w = sum(1 for o in cohort if (o.pnl or 0) > 0)
        tot = sum((o.pnl or 0) for o in cohort)
        long_sub = [o for o in cohort if o.direction == 'LONG']
        short_sub = [o for o in cohort if o.direction == 'SHORT']
        return {
            'n': n,
            'w': w,
            'l': n - w,
            'wr_pct': round(100*w/n, 1),
            'total_dollars': round(tot, 2),
            'avg_dollars_per_trade': round(tot/n, 2),
            'pct_of_batch': round(100*n/len(closed), 1),
            'long_n': len(long_sub),
            'long_w': sum(1 for o in long_sub if (o.pnl or 0) > 0),
            'long_total': round(sum((o.pnl or 0) for o in long_sub), 2),
            'short_n': len(short_sub),
            'short_w': sum(1 for o in short_sub if (o.pnl or 0) > 0),
            'short_total': round(sum((o.pnl or 0) for o in short_sub), 2),
        }

    return {
        'cohorts': [
            {'key': 'both', 'label': 'Both C and W match (crossed)', **stat(cohorts['both'])},
            {'key': 'c_only', 'label': 'C only (no W match)', **stat(cohorts['c_only'])},
            {'key': 'w_only', 'label': 'W only (no C match)', **stat(cohorts['w_only'])},
            {'key': 'truly_unmatched', 'label': 'TRULY UNMATCHED (no C, no W)', **stat(cohorts['truly_unmatched'])},
        ],
        'total': {
            'n': len(closed),
            'w': sum(1 for o in closed if (o.pnl or 0) > 0),
            'l': sum(1 for o in closed if (o.pnl or 0) <= 0),
            'total_dollars': round(sum((o.pnl or 0) for o in closed), 2),
        },
    }


def _compute_pattern_combo_tracker(orders, tracker='C'):
    """Pattern combo tracker (May 26 evening ship).

    Groups CLOSED trades by their sorted+joined pattern signature within ONE
    tracker (C or W). Surfaces combos like "C2+C4" or "W2+W4" that fire on
    the same trade — currently invisible in single-pattern trackers.

    Triggered by today's RENDERUSDT/IOUSDT LONG observation: both trades
    matched C2+C4 simultaneously but the combo wasn't a row in any table.

    Per-row metrics: N, W/L, WR%, Avg P&L %, Total $, NP%, AvgPeak%, R:R,
    TP+SL combined Δ$ (per Pattern C verdict logic), MULT 2× Δ$ (per
    Pattern W verdict logic), Verdict.

    Combos shown for both directions. Sorted by Total $ ascending (worst
    losers first). Trades with NO match in this tracker get combo='(none)'
    — useful baseline reference.

    No schema change — reads entry_pattern_{c1-9, w1-6}_match columns.
    Falls back to post-hoc Pattern W computation for trades without
    populated w flags (mirrors _compute_pattern_4cohort_coverage logic).
    """
    closed = [o for o in orders if o.status == 'CLOSED' and o.pnl is not None]
    if not closed:
        return {'rows': [], 'tracker': tracker}

    def _c_flags_list(o):
        out = []
        for i in range(1, 10):
            if getattr(o, f'entry_pattern_c{i}_match', None) is True:
                out.append(f'C{i}')
        return out

    def _w_flags_list(o):
        any_populated = any(
            getattr(o, f'entry_pattern_w{i}_match', None) is True
            for i in range(1, 7)
        )
        out = []
        if any_populated:
            for i in range(1, 7):
                if getattr(o, f'entry_pattern_w{i}_match', None) is True:
                    out.append(f'W{i}')
        else:
            try:
                w1, w2, w3, w4, w5, w6, _ = _compute_pattern_w_match(o)
                for idx, flag in enumerate([w1, w2, w3, w4, w5, w6], start=1):
                    if flag:
                        out.append(f'W{idx}')
            except Exception:
                pass
        return out

    flags_fn = _c_flags_list if tracker == 'C' else _w_flags_list

    # Group by (combo_signature, direction)
    groups = {}
    for o in closed:
        flags = flags_fn(o)
        combo = '+'.join(flags) if flags else '(none)'
        direction = o.direction or 'UNKNOWN'
        key = (combo, direction)
        groups.setdefault(key, []).append(o)

    # Per-cohort TP+SL counterfactual (mirror Pattern C tracker logic)
    def _sim_combined(cohort, tp=0.10, sl=0.50):
        sim_d = 0.0
        actual_d = 0.0
        tp_fires = sl_fires = cut_winners = 0
        for o in cohort:
            nt = o.notional_value or ((o.investment or 0) * (o.leverage or 1))
            actual_pct = o.pnl_percentage or 0
            peak = o.peak_pnl
            trough = o.trough_pnl
            peak_at = getattr(o, 'peak_reached_at', None)
            trough_at = getattr(o, 'trough_reached_at', None)
            actual_d += actual_pct / 100.0 * nt
            tp_would_fire = peak is not None and peak >= tp
            sl_would_fire = trough is not None and trough <= -sl
            if tp_would_fire and sl_would_fire:
                if peak_at and trough_at:
                    if peak_at <= trough_at:
                        sim_pct, tp_fires = tp, tp_fires + 1
                    else:
                        sim_pct, sl_fires = -sl, sl_fires + 1
                        if actual_pct > 0: cut_winners += 1
                else:
                    sim_pct, tp_fires = tp, tp_fires + 1
            elif tp_would_fire:
                sim_pct, tp_fires = tp, tp_fires + 1
            elif sl_would_fire:
                sim_pct, sl_fires = -sl, sl_fires + 1
                if actual_pct > 0: cut_winners += 1
            else:
                sim_pct = actual_pct
            sim_d += sim_pct / 100.0 * nt
        return actual_d, sim_d, tp_fires, sl_fires, cut_winners

    # Per-cohort MULT 2× counterfactual
    def _sim_mult2x(cohort):
        actual_d = sum((o.pnl_percentage or 0) / 100.0 *
                       (o.notional_value or ((o.investment or 0) * (o.leverage or 1)))
                       for o in cohort)
        return actual_d  # at 2× returns 2 × actual; Δ$ = actual

    rows = []
    for (combo, direction), cohort in groups.items():
        n = len(cohort)
        if n < 1:
            continue
        winners = [o for o in cohort if (o.pnl or 0) > 0]
        losers = [o for o in cohort if (o.pnl or 0) <= 0]
        nps = [o for o in cohort if (o.peak_pnl or 0) <= 0]
        total_pnl = sum(o.pnl or 0 for o in cohort)
        avg_pct = sum(o.pnl_percentage or 0 for o in cohort) / n
        avg_peak = sum(o.peak_pnl or 0 for o in cohort) / n
        wr = (len(winners) / n * 100) if n else 0
        np_pct = (len(nps) / n * 100) if n else 0
        loser_pct = (len(losers) / n * 100) if n else 0
        avg_win = (sum(o.pnl for o in winners) / len(winners)) if winners else 0
        avg_loss = abs(sum(o.pnl for o in losers) / len(losers)) if losers else 0
        rr = (avg_loss / avg_win) if avg_win > 0 else (float('inf') if losers else 0)
        rr_str = '∞' if rr == float('inf') else ('—' if not winners else f'{rr:.1f}')

        actual_d, sim_d, tp_fires, sl_fires, cw = _sim_combined(cohort)
        combined_delta = sim_d - actual_d

        mult_delta = _sim_mult2x(cohort)  # Δ$ at 2× = original total (linear)

        # Verdict (mirror Pattern C/W logic)
        verdict = ''
        if n < 5:
            verdict = '⚠ Low N'
        elif n >= 30 and wr <= 40 and avg_pct <= -0.20 and total_pnl < 0:
            verdict = '⚠ FILTER CANDIDATE'
        elif n >= 30 and wr >= 70 and avg_pct >= 0.10 and total_pnl > 0:
            verdict = '★ MULTIPLIER CANDIDATE'
        elif n >= 10 and wr <= 45 and avg_pct <= -0.15 and total_pnl < 0:
            verdict = '⚠ Warning (filter trend)'
        elif n >= 10 and wr >= 60 and avg_pct > 0 and total_pnl > 0:
            verdict = '★ Winner trend — wait for N≥30'
        elif n >= 10 and wr >= 60 and (avg_pct <= 0 or total_pnl <= 0):
            verdict = '⚠ High WR but net losing'
        else:
            verdict = '✓ Inconclusive'

        rows.append({
            'combo': combo,
            'direction': direction,
            'n': n,
            'w': len(winners),
            'l': len(losers),
            'wr_pct': round(wr, 1),
            'avg_pnl_pct': round(avg_pct, 3),
            'total_dollars': round(total_pnl, 2),
            'np_count': len(nps),
            'np_rate_pct': round(np_pct, 1),
            'loser_pct': round(loser_pct, 1),
            'avg_peak_pct': round(avg_peak, 3),
            'rr_ratio': rr_str,
            'combined_delta_usd': round(combined_delta, 2),
            'combined_fires_tp': tp_fires,
            'combined_fires_sl': sl_fires,
            'combined_cut_winners': cw,
            'mult2x_delta_usd': round(mult_delta, 2),
            'verdict': verdict,
        })

    # Sort by total_dollars ascending (worst losers first), but pull "(none)" baseline rows to bottom
    rows.sort(key=lambda r: (r['combo'] == '(none)', r['total_dollars']))

    return {'rows': rows, 'tracker': tracker}


def _compute_pattern_cell_performance(orders):
    """Pattern Cell Ship Performance (May 21 Phase 2) — sister to Multiplier Cell
    Performance but for the NEW pattern-based shipping mechanism.

    Groups closed trades by pattern_cell_source (e.g., "C4", "W2+W4") and computes
    per-cell verdicts. Distinct from Multiplier Cell Performance because:
      - Source is a pattern label (C4/C8/W1/W2/W4) not a RSI×ADX coordinate
      - Pattern C cells can have fixed_tp/fixed_sl overrides (active exit-side effect
        even when inv_mult / lev_mult = 1.0×)
      - Verdict considers both sizing AND exit effects

    Returns dict with 'rules' (list of per-cell rows) and 'summary' (aggregate uplift).
    """
    closed = [o for o in orders if o.status == 'CLOSED' and o.pnl is not None]
    if not closed:
        return {'rules': [], 'summary': {}}

    # Pull current config rules so we can flag demoted cells AND surface configured
    # TP/SL even when a cell hasn't traded yet (informational).
    try:
        cfg_rules = getattr(config.trading_config.thresholds, 'pattern_cell_rules', []) or []
    except Exception:
        cfg_rules = []

    # Brief description map mirrors the UI labels — keep in sync if patterns added.
    PATTERN_BRIEF = {
        'C1': 'Capitulation chase', 'C2': 'Macro counter-trend', 'C3': 'Stretch exhaustion',
        'C4': 'Low-vol chop', 'C5': 'Slow Climber Death', 'C6': 'Macro over-extended same dir',
        'C7': 'Pair Countertrend Bounce', 'C8': 'Oversold/Overbought Chop',
        'C9': 'Low-vol Countertrend Chop',
        'W1': 'HighConv trend', 'W2': 'Macro tailwind', 'W3': 'Energetic vol breakout',
        'W4': 'Pullback aligned', 'W5': 'Confluence',
        'W6': 'BTC Tailwind/Bear',
    }

    # Build rule-lookup keyed by (pattern, direction). Pattern Cell rules apply at
    # entry time, but a trade may carry multiple matched patterns in its source
    # label (e.g., "C4+C8"). For attribution we surface ALL trades matching by
    # exact source string as one row, AND also synthesize "no shipped trade yet"
    # rows for cells configured but not yet triggered.
    rule_index = {}
    for r in cfg_rules:
        try:
            p = r.get('pattern', '')
            d = r.get('direction', '')
            if not p or not d:
                continue
            rule_index[(p, d)] = r
        except Exception:
            continue

    # Compute direction baseline = avg P&L% on non-multiplied non-pattern trades.
    def _direction_baseline(direction):
        base = [o for o in closed if o.direction == direction
                and (o.cell_multiplier or 1.0) == 1.0
                and (getattr(o, 'pattern_cell_source', None) is None)
                and o.pnl_percentage is not None]
        if not base:
            return None
        return sum(o.pnl_percentage for o in base) / len(base)

    bl_long = _direction_baseline('LONG')
    bl_short = _direction_baseline('SHORT')

    # Group closed trades by (direction, pattern_cell_source)
    groups = {}
    for o in closed:
        src = getattr(o, 'pattern_cell_source', None)
        if not src:
            continue
        key = (o.direction, src)
        groups.setdefault(key, []).append(o)

    rows = []
    for (direction, src), os_ in groups.items():
        n = len(os_)
        wins = [o for o in os_ if (o.pnl or 0) > 0]
        losses = [o for o in os_ if (o.pnl or 0) <= 0]
        wr = 100.0 * len(wins) / n if n else 0.0
        avg_pct = sum((o.pnl_percentage or 0) for o in os_) / n if n else 0
        total_d = sum((o.pnl or 0) for o in os_)
        expect_d = total_d / n if n else 0
        win_d = sum((o.pnl or 0) for o in wins)
        loss_d = abs(sum((o.pnl or 0) for o in losses))
        pf = (win_d / loss_d) if loss_d > 0 else (float('inf') if win_d > 0 else 0)
        capped = sum(1 for o in os_ if getattr(o, 'cell_multiplier_capped', False))

        # Aggregate inv/lev/effective mult — use mode (most common value) across
        # the cohort (config can change mid-batch; we report what trades actually used)
        inv_mults = [o.cell_multiplier for o in os_ if o.cell_multiplier is not None]
        lev_mults = [getattr(o, 'cell_lev_multiplier', None) or 1.0 for o in os_]
        avg_inv = sum(inv_mults) / max(len(inv_mults), 1)
        avg_lev = sum(lev_mults) / max(len(lev_mults), 1)
        avg_eff = avg_inv * avg_lev

        # Fixed TP/SL — surface from any trade in the group (consistent if rule unchanged)
        fixed_tp = next((getattr(o, 'pattern_fixed_tp_pct', None) for o in os_ if getattr(o, 'pattern_fixed_tp_pct', None) is not None), None)
        fixed_sl = next((getattr(o, 'pattern_fixed_sl_pct', None) for o in os_ if getattr(o, 'pattern_fixed_sl_pct', None) is not None), None)

        # Count close_reasons for diagnostic (how often did fixed TP/SL fire?)
        n_fixed_tp = sum(1 for o in os_ if o.close_reason and 'PATTERN_FIXED_TP' in o.close_reason)
        n_fixed_sl = sum(1 for o in os_ if o.close_reason and 'PATTERN_FIXED_SL' in o.close_reason)

        # Δ$ vs baseline = total_$ × (1 - 1/effective_mult). For pure-fixed-exit
        # cells (eff=1.0×) this is zero — they're shipped via exits, not sizing.
        # The exit-side effect on $ is implicit in total_d itself (already capped).
        if avg_eff == 1.0:
            delta_dollars = 0.0
        else:
            delta_dollars = total_d * (1.0 - 1.0 / avg_eff)

        # Brief description — derived from src label. For single-pattern (e.g. "W1")
        # use the brief directly; for combo (e.g., "C4+C8") build comma list.
        parts = src.split('+')
        brief_parts = [PATTERN_BRIEF.get(p, p) for p in parts]
        brief = ', '.join(brief_parts)

        # Verdict — for sized cells use Phase 3 matrix; for pure-exit cells use simpler logic
        if n < 5:
            verdict = '⚠ Low N'
        elif avg_eff > 1.0:
            # Sizing cell — standard verdict
            if total_d < 0:
                verdict = '✗ HARMFUL'
            elif wr >= 70 and delta_dollars > 1.0:
                verdict = '★ WORKING'
            elif delta_dollars > 1.0:
                verdict = '✓ Marginal'
            elif delta_dollars < -1.0:
                verdict = '⚠ DRAG'
            else:
                verdict = '✓ Marginal'
        else:
            # Pure exit cell (C-rule with fixed TP/SL, mult=1.0)
            # Verdict based on fixed exits firing + total $ outcome
            if total_d > 0:
                verdict = '★ EXITS WORKING'
            elif total_d > -10:
                verdict = '✓ Marginal (exits)'
            elif n_fixed_tp + n_fixed_sl >= n * 0.5:
                # At least half the cohort hit fixed TP or SL
                verdict = '✓ Exits firing'
            else:
                verdict = '⚠ Exits not firing'

        # Per-direction baseline for context
        baseline = bl_long if direction == 'LONG' else bl_short

        rows.append({
            'source': src,
            'brief': brief,
            'direction': direction,
            'inv_multiplier': round(avg_inv, 2),
            'lev_multiplier': round(avg_lev, 2),
            'effective_multiplier': round(avg_eff, 2),
            'fixed_tp_pct': fixed_tp,
            'fixed_sl_pct': fixed_sl,
            'n': n,
            'wr_pct': round(wr, 1),
            'avg_pnl_pct': round(avg_pct, 4),
            'total_dollars': round(total_d, 2),
            'expect_per_trade': round(expect_d, 3),
            'profit_factor': round(pf, 2) if pf != float('inf') else 999.99,
            'baseline_avg_pct': round(baseline, 4) if baseline is not None else None,
            'delta_vs_baseline_dollars': round(delta_dollars, 2),
            'capped_count': capped,
            'n_fixed_tp_fires': n_fixed_tp,
            'n_fixed_sl_fires': n_fixed_sl,
            'verdict': verdict,
        })

    # Synthesize rows for configured cells that haven't traded yet — informational
    # (operator can see all configured rules + their settings even at zero N).
    traded_sources = {r['source'] for r in rows}
    for (p, d), rule in rule_index.items():
        # A configured single-pattern cell may map to a multi-pattern source label
        # if the trade matched multiple patterns. Skip if the EXACT pattern label
        # (no combos) has been traded.
        if p in traded_sources:
            continue
        # Check if a combo with this pattern exists
        if any(p in s.split('+') for s in traded_sources):
            continue
        inv = float(rule.get('inv_mult', 1.0) or 1.0)
        lev = float(rule.get('lev_mult', 1.0) or 1.0)
        rows.append({
            'source': p,
            'brief': PATTERN_BRIEF.get(p, p),
            'direction': d,
            'inv_multiplier': round(inv, 2),
            'lev_multiplier': round(lev, 2),
            'effective_multiplier': round(inv * lev, 2),
            'fixed_tp_pct': rule.get('fixed_tp_pct'),
            'fixed_sl_pct': rule.get('fixed_sl_pct'),
            'n': 0,
            'wr_pct': 0.0,
            'avg_pnl_pct': 0.0,
            'total_dollars': 0.0,
            'expect_per_trade': 0.0,
            'profit_factor': 0,
            'baseline_avg_pct': None,
            'delta_vs_baseline_dollars': 0.0,
            'capped_count': 0,
            'n_fixed_tp_fires': 0,
            'n_fixed_sl_fires': 0,
            'verdict': '⏳ No trades yet',
        })

    # Sort: by direction, then by N desc, then by source
    rows.sort(key=lambda r: (r['direction'], -r['n'], r['source']))

    # Summary uplift across all cells
    def _eff_for_row(r):
        return r.get('effective_multiplier') or 1.0
    boosted = [r for r in rows if r['n'] > 0 and _eff_for_row(r) != 1.0]
    actual_total = sum(r['total_dollars'] for r in boosted)
    sim_baseline = sum(r['total_dollars'] / _eff_for_row(r) for r in boosted if _eff_for_row(r))
    fixed_cells = [r for r in rows if r['n'] > 0 and _eff_for_row(r) == 1.0 and (r['fixed_tp_pct'] is not None or r['fixed_sl_pct'] is not None)]
    fixed_total = sum(r['total_dollars'] for r in fixed_cells)
    fixed_n = sum(r['n'] for r in fixed_cells)
    summary = {
        'boosted_trades_n': sum(r['n'] for r in boosted),
        'boosted_actual_dollars': round(actual_total, 2),
        'boosted_baseline_dollars': round(sim_baseline, 2),
        'boosted_uplift_dollars': round(actual_total - sim_baseline, 2),
        'fixed_exit_trades_n': fixed_n,
        'fixed_exit_total_dollars': round(fixed_total, 2),
    }

    return {'rules': rows, 'summary': summary}


def _compute_extension_multiplier_performance(orders):
    """Extension Multiplier Performance (May 24, 2026) — sister to Pattern Cell Ship
    Performance but for the Pair Distance from EMA13 multiplier dimension.

    Groups closed trades by cell_multiplier_source starting with "EXT_" (e.g., "EXT_L1b",
    "EXT_L1b+L2a") and computes per-cell verdicts using the same logic as the RSI×ADX
    multiplier surface.

    Returns dict with 'rules' (list of per-cell rows) and 'summary' (aggregate uplift).
    """
    closed = [o for o in orders if o.status == 'CLOSED' and o.pnl is not None]
    if not closed:
        return {'rules': [], 'summary': {}}

    try:
        cfg_rules = getattr(config.trading_config.thresholds, 'extension_multiplier_rules', []) or []
    except Exception:
        cfg_rules = []

    def _rule_brief(rule):
        parts = [f"Ext {rule.get('ext_min', 0):+.2f} to {rule.get('ext_max', 0):+.2f}%"]
        if rule.get('pair_vol_max') is not None:
            parts.append(f"× PairVol<{rule['pair_vol_max']}")
        if rule.get('adx_delta_max') is not None:
            parts.append(f"× ADXΔ<{rule['adx_delta_max']}")
        return ' '.join(parts)

    rule_index = {}
    for r in cfg_rules:
        try:
            name = r.get('name', '')
            d = r.get('direction', '')
            if not name or not d:
                continue
            rule_index[(name, d)] = r
        except Exception:
            continue

    def _direction_baseline(direction):
        base = [o for o in closed if o.direction == direction
                and (o.cell_multiplier or 1.0) == 1.0
                and not (getattr(o, 'cell_multiplier_source', '') or '').startswith('EXT_')
                and (getattr(o, 'pattern_cell_source', None) is None)
                and o.pnl_percentage is not None]
        if not base:
            return None
        return sum(o.pnl_percentage for o in base) / len(base)

    bl_long = _direction_baseline('LONG')
    bl_short = _direction_baseline('SHORT')

    groups = {}
    for o in closed:
        src = getattr(o, 'cell_multiplier_source', '') or ''
        if not src.startswith('EXT_'):
            continue
        key = (o.direction, src)
        groups.setdefault(key, []).append(o)

    rows = []
    for (direction, src), os_ in groups.items():
        n = len(os_)
        wins = [o for o in os_ if (o.pnl or 0) > 0]
        losses = [o for o in os_ if (o.pnl or 0) <= 0]
        wr = 100.0 * len(wins) / n if n else 0.0
        avg_pct = sum((o.pnl_percentage or 0) for o in os_) / n if n else 0
        total_d = sum((o.pnl or 0) for o in os_)
        expect_d = total_d / n if n else 0
        win_d = sum((o.pnl or 0) for o in wins)
        loss_d = abs(sum((o.pnl or 0) for o in losses))
        pf = (win_d / loss_d) if loss_d > 0 else (float('inf') if win_d > 0 else 0)
        capped = sum(1 for o in os_ if getattr(o, 'cell_multiplier_capped', False))

        inv_mults = [o.cell_multiplier for o in os_ if o.cell_multiplier is not None]
        lev_mults = [getattr(o, 'cell_lev_multiplier', None) or 1.0 for o in os_]
        avg_inv = sum(inv_mults) / max(len(inv_mults), 1)
        avg_lev = sum(lev_mults) / max(len(lev_mults), 1)
        avg_eff = avg_inv * avg_lev

        peaks = [o.peak_pnl for o in os_ if o.peak_pnl is not None]
        avg_peak = sum(peaks) / max(len(peaks), 1) if peaks else 0.0
        loser_pks = [o.peak_pnl or 0 for o in losses]
        loser_pk20 = sum(1 for p in loser_pks if p >= 0.20)

        delta_dollars = total_d * (1.0 - 1.0 / avg_eff) if avg_eff != 1.0 else 0.0

        bare = src[4:] if src.startswith('EXT_') else src
        names = bare.split('+')
        rule_briefs = []
        for nm in names:
            r = rule_index.get((nm, direction))
            if r:
                rule_briefs.append(f"{nm}: {_rule_brief(r)}")
            else:
                rule_briefs.append(nm)
        brief = ' | '.join(rule_briefs)

        if n < 5:
            verdict = '⚠ Low N'
        elif total_d < 0:
            verdict = '✗ HARMFUL'
        elif wr >= 70 and delta_dollars > 1.0:
            verdict = '★ WORKING'
        elif delta_dollars > 1.0:
            verdict = '✓ Marginal'
        elif delta_dollars < -1.0:
            verdict = '⚠ DRAG'
        else:
            verdict = '✓ Marginal'

        baseline = bl_long if direction == 'LONG' else bl_short

        rows.append({
            'source': src,
            'brief': brief,
            'direction': direction,
            'inv_multiplier': round(avg_inv, 2),
            'lev_multiplier': round(avg_lev, 2),
            'effective_multiplier': round(avg_eff, 2),
            'n': n,
            'wr_pct': round(wr, 1),
            'avg_pnl_pct': round(avg_pct, 4),
            'avg_peak_pct': round(avg_peak, 3),
            'total_dollars': round(total_d, 2),
            'expect_per_trade': round(expect_d, 3),
            'profit_factor': round(pf, 2) if pf != float('inf') else 999.99,
            'baseline_avg_pct': round(baseline, 4) if baseline is not None else None,
            'delta_vs_baseline_dollars': round(delta_dollars, 2),
            'capped_count': capped,
            'loser_count': len(losses),
            'loser_peak_ge_20_count': loser_pk20,
            'verdict': verdict,
        })

    traded_sources = {r['source'] for r in rows}
    for (nm, d), rule in rule_index.items():
        bare = f"EXT_{nm}"
        if bare in traded_sources:
            continue
        if any(nm in (s[4:].split('+') if s.startswith('EXT_') else []) for s in traded_sources):
            continue
        inv = float(rule.get('inv_mult', 1.0) or 1.0)
        lev = float(rule.get('lev_mult', 1.0) or 1.0)
        rows.append({
            'source': bare,
            'brief': f"{nm}: {_rule_brief(rule)}",
            'direction': d,
            'inv_multiplier': round(inv, 2),
            'lev_multiplier': round(lev, 2),
            'effective_multiplier': round(inv * lev, 2),
            'n': 0,
            'wr_pct': 0.0,
            'avg_pnl_pct': 0.0,
            'avg_peak_pct': 0.0,
            'total_dollars': 0.0,
            'expect_per_trade': 0.0,
            'profit_factor': 0,
            'baseline_avg_pct': None,
            'delta_vs_baseline_dollars': 0.0,
            'capped_count': 0,
            'loser_count': 0,
            'loser_peak_ge_20_count': 0,
            'verdict': '⏳ No trades yet',
        })

    rows.sort(key=lambda r: (r['direction'], -r['n'], r['source']))

    boosted = [r for r in rows if r['n'] > 0 and (r.get('effective_multiplier') or 1.0) != 1.0]
    actual_total = sum(r['total_dollars'] for r in boosted)
    sim_baseline = sum(r['total_dollars'] / (r.get('effective_multiplier') or 1.0) for r in boosted)
    summary = {
        'boosted_trades_n': sum(r['n'] for r in boosted),
        'boosted_actual_dollars': round(actual_total, 2),
        'boosted_baseline_dollars': round(sim_baseline, 2),
        'boosted_uplift_dollars': round(actual_total - sim_baseline, 2),
    }

    return {'rules': rows, 'summary': summary}


def _compute_btc_1h_slope_btc_adx_multiplier_performance(orders):
    """BTC 1h Slope × BTC ADX Multiplier Performance (May 24 evening, 2026).

    Sister to Extension Multiplier Performance. Groups closed trades by
    cell_multiplier_source starting with "BTC1H_" and computes per-cell verdicts.
    """
    closed = [o for o in orders if o.status == 'CLOSED' and o.pnl is not None]
    if not closed:
        return {'rules': [], 'summary': {}}

    try:
        cfg_rules = getattr(config.trading_config.thresholds,
                            'btc_1h_slope_btc_adx_multiplier_rules', []) or []
    except Exception:
        cfg_rules = []

    def _rule_brief(rule):
        sm = rule.get('slope_min', 0)
        sx = rule.get('slope_max', 0)
        am = rule.get('adx_min', 0)
        ax = rule.get('adx_max', 0)
        return f"Slope {sm:+.2f} to {sx:+.2f}% × ADX {am:.0f}-{ax:.0f}"

    rule_index = {}
    for r in cfg_rules:
        try:
            name = r.get('name', '')
            d = r.get('direction', '')
            if not name or not d:
                continue
            rule_index[(name, d)] = r
        except Exception:
            continue

    def _direction_baseline(direction):
        base = [o for o in closed if o.direction == direction
                and (o.cell_multiplier or 1.0) == 1.0
                and not (getattr(o, 'cell_multiplier_source', '') or '').startswith('BTC1H_')
                and (getattr(o, 'pattern_cell_source', None) is None)
                and o.pnl_percentage is not None]
        if not base:
            return None
        return sum(o.pnl_percentage for o in base) / len(base)

    bl_long = _direction_baseline('LONG')
    bl_short = _direction_baseline('SHORT')

    groups = {}
    for o in closed:
        src = getattr(o, 'cell_multiplier_source', '') or ''
        if not src.startswith('BTC1H_'):
            continue
        key = (o.direction, src)
        groups.setdefault(key, []).append(o)

    rows = []
    for (direction, src), os_ in groups.items():
        n = len(os_)
        wins = [o for o in os_ if (o.pnl or 0) > 0]
        losses = [o for o in os_ if (o.pnl or 0) <= 0]
        wr = 100.0 * len(wins) / n if n else 0.0
        avg_pct = sum((o.pnl_percentage or 0) for o in os_) / n if n else 0
        total_d = sum((o.pnl or 0) for o in os_)
        expect_d = total_d / n if n else 0
        win_d = sum((o.pnl or 0) for o in wins)
        loss_d = abs(sum((o.pnl or 0) for o in losses))
        pf = (win_d / loss_d) if loss_d > 0 else (float('inf') if win_d > 0 else 0)
        capped = sum(1 for o in os_ if getattr(o, 'cell_multiplier_capped', False))

        inv_mults = [o.cell_multiplier for o in os_ if o.cell_multiplier is not None]
        lev_mults = [getattr(o, 'cell_lev_multiplier', None) or 1.0 for o in os_]
        avg_inv = sum(inv_mults) / max(len(inv_mults), 1)
        avg_lev = sum(lev_mults) / max(len(lev_mults), 1)
        avg_eff = avg_inv * avg_lev

        peaks = [o.peak_pnl for o in os_ if o.peak_pnl is not None]
        avg_peak = sum(peaks) / max(len(peaks), 1) if peaks else 0.0
        loser_pks = [o.peak_pnl or 0 for o in losses]
        loser_pk20 = sum(1 for p in loser_pks if p >= 0.20)

        delta_dollars = total_d * (1.0 - 1.0 / avg_eff) if avg_eff != 1.0 else 0.0

        bare = src[6:] if src.startswith('BTC1H_') else src
        names = bare.split('+')
        rule_briefs = []
        for nm in names:
            r = rule_index.get((nm, direction))
            if r:
                rule_briefs.append(f"{nm}: {_rule_brief(r)}")
            else:
                rule_briefs.append(nm)
        brief = ' | '.join(rule_briefs)

        if n < 5:
            verdict = '⚠ Low N'
        elif total_d < 0:
            verdict = '✗ HARMFUL'
        elif wr >= 70 and delta_dollars > 1.0:
            verdict = '★ WORKING'
        elif delta_dollars > 1.0:
            verdict = '✓ Marginal'
        elif delta_dollars < -1.0:
            verdict = '⚠ DRAG'
        else:
            verdict = '✓ Marginal'

        baseline = bl_long if direction == 'LONG' else bl_short

        rows.append({
            'source': src,
            'brief': brief,
            'direction': direction,
            'inv_multiplier': round(avg_inv, 2),
            'lev_multiplier': round(avg_lev, 2),
            'effective_multiplier': round(avg_eff, 2),
            'n': n,
            'wr_pct': round(wr, 1),
            'avg_pnl_pct': round(avg_pct, 4),
            'avg_peak_pct': round(avg_peak, 3),
            'total_dollars': round(total_d, 2),
            'expect_per_trade': round(expect_d, 3),
            'profit_factor': round(pf, 2) if pf != float('inf') else 999.99,
            'baseline_avg_pct': round(baseline, 4) if baseline is not None else None,
            'delta_vs_baseline_dollars': round(delta_dollars, 2),
            'capped_count': capped,
            'loser_count': len(losses),
            'loser_peak_ge_20_count': loser_pk20,
            'verdict': verdict,
        })

    # Synthesize rows for configured rules that haven't traded yet
    traded_sources = {r['source'] for r in rows}
    for (nm, d), rule in rule_index.items():
        bare = f"BTC1H_{nm}"
        if bare in traded_sources:
            continue
        if any(nm in (s[6:].split('+') if s.startswith('BTC1H_') else []) for s in traded_sources):
            continue
        inv = float(rule.get('inv_mult', 1.0) or 1.0)
        lev = float(rule.get('lev_mult', 1.0) or 1.0)
        rows.append({
            'source': bare,
            'brief': f"{nm}: {_rule_brief(rule)}",
            'direction': d,
            'inv_multiplier': round(inv, 2),
            'lev_multiplier': round(lev, 2),
            'effective_multiplier': round(inv * lev, 2),
            'n': 0,
            'wr_pct': 0.0,
            'avg_pnl_pct': 0.0,
            'avg_peak_pct': 0.0,
            'total_dollars': 0.0,
            'expect_per_trade': 0.0,
            'profit_factor': 0,
            'baseline_avg_pct': None,
            'delta_vs_baseline_dollars': 0.0,
            'capped_count': 0,
            'loser_count': 0,
            'loser_peak_ge_20_count': 0,
            'verdict': '⏳ No trades yet',
        })

    rows.sort(key=lambda r: (r['direction'], -r['n'], r['source']))

    boosted = [r for r in rows if r['n'] > 0 and (r.get('effective_multiplier') or 1.0) != 1.0]
    actual_total = sum(r['total_dollars'] for r in boosted)
    sim_baseline = sum(r['total_dollars'] / (r.get('effective_multiplier') or 1.0) for r in boosted)
    summary = {
        'boosted_trades_n': sum(r['n'] for r in boosted),
        'boosted_actual_dollars': round(actual_total, 2),
        'boosted_baseline_dollars': round(sim_baseline, 2),
        'boosted_uplift_dollars': round(actual_total - sim_baseline, 2),
    }

    return {'rules': rows, 'summary': summary}


# ----- Investor Portfolio -----

class InvestorCreate(BaseModel):
    name: str

class InvestorDeposit(BaseModel):
    investor_id: int
    amount: float

class InvestorWithdraw(BaseModel):
    investor_id: int
    amount: float

class InvestorRename(BaseModel):
    name: str


async def _get_portfolio_value(db: AsyncSession) -> float:
    """Return the total USDT portfolio value (balance + open positions margin)."""
    await trading_engine.initialize(db)
    if trading_engine.is_paper_mode:
        balance = await trading_engine._recalculate_paper_balance(db)
        result = await db.execute(
            select(Order).where(and_(Order.status == "OPEN", Order.is_paper == True))
        )
        open_orders = result.scalars().all()
        used_margin = sum(o.investment for o in open_orders)
        bnb_usd = trading_engine.paper_bnb_balance_usd
        return balance + used_margin + bnb_usd
    else:
        bal = await binance_service.get_balance()
        bnb_price = await binance_service.get_bnb_price()
        bnb_usd = bal['bnb_total'] * bnb_price if bnb_price > 0 else 0
        return bal['usdt_total'] + bnb_usd


async def _get_total_shares(db: AsyncSession) -> float:
    result = await db.execute(select(func.coalesce(func.sum(Investor.shares), 0.0)))
    return result.scalar()


async def _get_nav_per_share(db: AsyncSession) -> float:
    total_shares = await _get_total_shares(db)
    if total_shares <= 0:
        return 1.0
    portfolio = await _get_portfolio_value(db)
    return portfolio / total_shares


@app.get("/api/investors")
async def list_investors(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Investor).order_by(Investor.id))
    investors = result.scalars().all()

    total_shares = await _get_total_shares(db)
    portfolio_value = await _get_portfolio_value(db)
    nav = portfolio_value / total_shares if total_shares > 0 else 1.0

    rows = []
    for inv in investors:
        pct = (inv.shares / total_shares * 100) if total_shares > 0 else 0.0
        value = inv.shares * nav
        rows.append({
            "id": inv.id,
            "name": inv.name,
            "shares": round(inv.shares, 6),
            "ownership_pct": round(pct, 2),
            "value_usd": round(value, 2),
            "total_deposited": round(inv.total_deposited, 2),
            "total_withdrawn": round(inv.total_withdrawn, 2),
            "created_at": inv.created_at.isoformat() if inv.created_at else None,
        })

    return {
        "investors": rows,
        "total_shares": round(total_shares, 6),
        "nav_per_share": round(nav, 6),
        "portfolio_value": round(portfolio_value, 2),
    }


@app.post("/api/investors")
async def add_investor(body: InvestorCreate, db: AsyncSession = Depends(get_db)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Name is required")

    existing = await db.execute(select(Investor).where(Investor.name == name))
    if existing.scalar():
        raise HTTPException(409, f"Investor '{name}' already exists")

    inv = Investor(name=name, shares=0.0, total_deposited=0.0, total_withdrawn=0.0)
    db.add(inv)
    await db.flush()
    return {"ok": True, "id": inv.id, "name": inv.name}


@app.post("/api/investors/deposit")
async def investor_deposit(body: InvestorDeposit, db: AsyncSession = Depends(get_db)):
    if body.amount <= 0:
        raise HTTPException(400, "Amount must be positive")

    inv = await db.get(Investor, body.investor_id)
    if not inv:
        raise HTTPException(404, "Investor not found")

    nav = await _get_nav_per_share(db)
    new_shares = body.amount / nav
    inv.shares += new_shares
    inv.total_deposited += body.amount
    await db.flush()

    return {"ok": True, "new_shares": round(new_shares, 6), "nav": round(nav, 6)}


@app.post("/api/investors/withdraw")
async def investor_withdraw(body: InvestorWithdraw, db: AsyncSession = Depends(get_db)):
    if body.amount <= 0:
        raise HTTPException(400, "Amount must be positive")

    inv = await db.get(Investor, body.investor_id)
    if not inv:
        raise HTTPException(404, "Investor not found")

    nav = await _get_nav_per_share(db)
    shares_needed = body.amount / nav

    if shares_needed > inv.shares + 1e-9:
        max_value = inv.shares * nav
        raise HTTPException(400, f"Insufficient shares. Max withdrawal: ${max_value:.2f}")

    inv.shares = max(0.0, inv.shares - shares_needed)
    inv.total_withdrawn += body.amount
    await db.flush()

    return {"ok": True, "shares_removed": round(shares_needed, 6), "nav": round(nav, 6)}


@app.patch("/api/investors/{investor_id}")
async def rename_investor(investor_id: int, body: InvestorRename, db: AsyncSession = Depends(get_db)):
    inv = await db.get(Investor, investor_id)
    if not inv:
        raise HTTPException(404, "Investor not found")
    new_name = (body.name or "").strip()
    if not new_name:
        raise HTTPException(400, "Name cannot be empty")
    inv.name = new_name
    await db.flush()
    return {"ok": True, "id": investor_id, "name": new_name}


@app.delete("/api/investors/{investor_id}")
async def remove_investor(investor_id: int, db: AsyncSession = Depends(get_db)):
    inv = await db.get(Investor, investor_id)
    if not inv:
        raise HTTPException(404, "Investor not found")

    if inv.shares > 1e-9:
        nav = await _get_nav_per_share(db)
        remaining_value = inv.shares * nav
        inv.total_withdrawn += remaining_value
        inv.shares = 0.0
        await db.flush()

    await db.execute(delete(Investor).where(Investor.id == investor_id))
    await db.flush()

    return {"ok": True}


# ----- Server Info -----

_server_initial_ip = None
_server_current_ip = None
_server_ip_last_check = 0

async def _fetch_outbound_ip() -> str:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get("https://api.ipify.org")
            return r.text.strip()
    except Exception:
        return "unavailable"

@app.get("/api/server-ip")
async def get_server_ip():
    global _server_initial_ip, _server_current_ip, _server_ip_last_check
    now = time.time()
    if _server_initial_ip is None:
        _server_initial_ip = await _fetch_outbound_ip()
        _server_current_ip = _server_initial_ip
        _server_ip_last_check = now
    elif now - _server_ip_last_check > 300:
        _server_current_ip = await _fetch_outbound_ip()
        _server_ip_last_check = now
    return {
        "ip": _server_current_ip,
        "initial_ip": _server_initial_ip,
        "changed": _server_current_ip != _server_initial_ip and _server_current_ip != "unavailable"
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
        # Float-safe comparison: numerically equal floats (e.g. 0.00018 vs 0.018/100
        # which differ in IEEE-754 representation but represent the same intended value)
        # should NOT log as a change. Use math.isclose for numeric pairs.
        if isinstance(old_val, float) and isinstance(new_val, float):
            if math.isclose(old_val, new_val, rel_tol=1e-9, abs_tol=1e-12):
                continue
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

