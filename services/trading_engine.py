"""
SCALPARS Trading Platform - Trading Engine
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy import select, update, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import Order, Transaction, BotState, PairData
from database import AsyncSessionLocal
import config
from config import save_trading_config, TradingConfig
from services.binance_service import binance_service
from services.indicators import calculate_indicators, get_signal, check_exit_conditions, calculate_pnl
from services.websocket_tracker import websocket_tracker

logger = logging.getLogger(__name__)

# Cache for open orders to enable fast real-time stop loss checks
_open_orders_cache: Dict[str, List[Dict]] = {}  # pair -> list of order info
_cache_lock = asyncio.Lock()


class TradingEngine:
    """Main trading engine that manages positions and executes trades"""
    
    def __init__(self):
        self.is_running = False
        self.is_paper_mode = True
        self.paper_balance = config.trading_config.paper_balance
        self.started_at: Optional[datetime] = None
        self.total_runtime_seconds = 0
        self._task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._last_scan_time: float = 0
    
    async def initialize(self, db: AsyncSession):
        """Initialize engine state from database"""
        result = await db.execute(select(BotState).limit(1))
        state = result.scalar_one_or_none()
        
        if state:
            self.is_running = state.is_running
            self.is_paper_mode = state.is_paper_mode
            self.paper_balance = state.paper_balance
            self.total_runtime_seconds = state.total_runtime_seconds
            if state.is_running and state.started_at:
                self.started_at = state.started_at
        else:
            # Create initial state
            state = BotState(
                is_running=False,
                is_paper_mode=True,
                paper_balance=config.trading_config.paper_balance,
                total_runtime_seconds=0
            )
            db.add(state)
            await db.commit()
    
    async def save_state(self, db: AsyncSession):
        """Save engine state to database"""
        result = await db.execute(select(BotState).limit(1))
        state = result.scalar_one_or_none()
        
        if state:
            state.is_running = self.is_running
            state.is_paper_mode = self.is_paper_mode
            state.paper_balance = self.paper_balance
            state.total_runtime_seconds = self.total_runtime_seconds
            state.started_at = self.started_at
            state.updated_at = datetime.utcnow()
        else:
            state = BotState(
                is_running=self.is_running,
                is_paper_mode=self.is_paper_mode,
                paper_balance=self.paper_balance,
                total_runtime_seconds=self.total_runtime_seconds,
                started_at=self.started_at
            )
            db.add(state)
        
        await db.commit()
    
    async def start(self, db: AsyncSession):
        """Start the trading bot"""
        self.is_running = True
        self.started_at = datetime.utcnow()
        await self.save_state(db)
        return {"status": "running", "message": "Bot started"}
    
    async def pause(self, db: AsyncSession):
        """Pause the trading bot (can still close positions)"""
        if self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            self.total_runtime_seconds += int(elapsed)
        
        self.is_running = False
        self.started_at = None
        await self.save_state(db)
        return {"status": "paused", "message": "Bot paused - will still close open positions"}
    
    async def set_paper_mode(self, enabled: bool, db: AsyncSession):
        """Toggle paper trading mode"""
        self.is_paper_mode = enabled
        if enabled:
            self.paper_balance = config.trading_config.paper_balance
        await self.save_state(db)
        return {"paper_mode": enabled}
    
    def get_runtime_seconds(self) -> int:
        """Get total runtime in seconds"""
        if self.is_running and self.started_at:
            elapsed = (datetime.utcnow() - self.started_at).total_seconds()
            return self.total_runtime_seconds + int(elapsed)
        return self.total_runtime_seconds
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        runtime = self.get_runtime_seconds()
        hours = runtime // 3600
        minutes = (runtime % 3600) // 60
        seconds = runtime % 60
        
        return {
            "is_running": self.is_running,
            "is_paper_mode": self.is_paper_mode,
            "paper_balance": self.paper_balance,
            "runtime_seconds": runtime,
            "runtime_formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        }
    
    async def get_available_balance(self, db: AsyncSession) -> float:
        """Get available balance for trading.
        
        For paper trading: self.paper_balance already represents free cash
        (reduced on open, restored on close), so no need to subtract used_margin.
        """
        if self.is_paper_mode:
            return self.paper_balance
        else:
            balance = await binance_service.get_balance()
            return balance['usdt_free']
    
    def calculate_position_size(self, available_balance: float, confidence: str) -> Tuple[float, float]:
        """
        Calculate position size and leverage based on config
        
        Returns:
            Tuple of (investment_amount, leverage)
        """
        tc = config.trading_config
        conf_level = tc.confidence_levels.get(confidence)
        
        if not conf_level or not conf_level.enabled:
            return 0, 0
        
        # Calculate safe reserve
        if tc.investment.reserve_mode == "percentage":
            reserve = available_balance * (tc.investment.reserve_percentage / 100)
        else:
            reserve = tc.investment.reserve_fixed
        
        # Available after reserve
        tradeable = max(0, available_balance - reserve)
        
        # Calculate base investment
        if tc.investment.mode == "percentage":
            investment = tradeable * (tc.investment.percentage / 100)
        else:
            investment = min(tc.investment.fixed_amount, tradeable)
        
        # Apply investment multiplier for higher confidence levels
        multiplier = getattr(conf_level, 'investment_multiplier', 1.0)
        investment = investment * multiplier
        
        # Ensure investment doesn't exceed tradeable balance
        investment = min(investment, tradeable)
        
        # Clamp investment to min/max size limits
        investment = max(investment, tc.investment.min_investment_size)
        investment = min(investment, tc.investment.max_investment_size)
        
        # If clamped min exceeds available tradeable balance, skip the trade
        if investment > tradeable:
            logger.warning(f"Min investment size ({tc.investment.min_investment_size}) exceeds tradeable balance ({tradeable:.2f}), skipping")
            return 0, 0
        
        # Get leverage from config
        leverage = conf_level.leverage
        
        return investment, leverage
    
    async def open_position(
        self,
        db: AsyncSession,
        pair: str,
        direction: str,
        confidence: str,
        current_price: float,
        entry_gap: float = None
    ) -> Optional[Order]:
        """Open a new position"""
        if not self.is_running:
            logger.warning(f"[SKIP] {pair}: Bot not running")
            return None
        
        # Check if confidence level is enabled
        conf_config = config.trading_config.confidence_levels.get(confidence)
        if not conf_config or not conf_config.enabled:
            logger.warning(f"[SKIP] {pair}: {confidence} confidence not enabled")
            return None
        
        # Check max open positions limit
        total_open = await db.execute(
            select(func.count(Order.id)).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        if total_open.scalar() >= config.trading_config.investment.max_open_positions:
            logger.warning(f"[SKIP] {pair}: Max open positions ({config.trading_config.investment.max_open_positions}) reached")
            return None
        
        # Check if we already have a position for this pair
        result = await db.execute(
            select(Order).where(
                and_(
                    Order.pair == pair,
                    Order.status == "OPEN",
                    Order.is_paper == self.is_paper_mode
                )
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            logger.info(f"[SKIP] {pair}: Already have open position")
            return None  # Already have position
        
        # Check cooldown after loss - don't re-enter same pair too quickly after a loss
        cooldown_minutes = config.trading_config.investment.cooldown_after_loss_minutes
        if cooldown_minutes > 0:
            cooldown_threshold = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
            result = await db.execute(
                select(Order).where(
                    and_(
                        Order.pair == pair,
                        Order.status == "CLOSED",
                        Order.is_paper == self.is_paper_mode,
                        Order.pnl < 0,  # Only check losing trades
                        Order.closed_at >= cooldown_threshold
                    )
                ).order_by(desc(Order.closed_at)).limit(1)
            )
            recent_loss = result.scalar_one_or_none()
            if recent_loss:
                time_since_loss = (datetime.utcnow() - recent_loss.closed_at).total_seconds() / 60
                logger.info(f"[COOLDOWN] {pair}: Recent loss {time_since_loss:.1f} mins ago, waiting {cooldown_minutes} mins")
                return None
        
        # Calculate position size
        available = await self.get_available_balance(db)
        investment, leverage = self.calculate_position_size(available, confidence)
        logger.info(f"[TRADE] {pair}: {direction} {confidence} - Investment: ${investment:.2f}, Leverage: {leverage}x")
        
        if investment <= 0:
            return None
        
        # Calculate notional and quantity
        notional_value = investment * leverage
        quantity = notional_value / current_price
        
        # Calculate entry fee
        entry_fee = notional_value * config.trading_config.trading_fee
        
        # Execute trade
        binance_order_id = None
        actual_price = current_price
        
        if not self.is_paper_mode:
            # Real trade
            symbol = pair.replace('USDT', '/USDT:USDT')
            side = 'buy' if direction == 'LONG' else 'sell'
            
            result = await binance_service.create_market_order(
                symbol=symbol,
                side=side,
                amount=quantity,
                leverage=int(leverage)
            )
            
            if result:
                binance_order_id = result['id']
                actual_price = result['price']
                entry_fee = result.get('fee', entry_fee)
        else:
            # Paper trade - deduct from paper balance
            self.paper_balance -= investment
        
        # Create order record
        order = Order(
            binance_order_id=binance_order_id,
            pair=pair,
            direction=direction,
            status="OPEN",
            entry_price=actual_price,
            current_price=actual_price,
            investment=investment,
            leverage=leverage,
            notional_value=notional_value,
            quantity=quantity,
            confidence=confidence,
            entry_gap=entry_gap,
            entry_fee=entry_fee,
            peak_pnl=0.0,
            high_price_since_entry=actual_price if direction == "LONG" else None,
            low_price_since_entry=actual_price if direction == "SHORT" else None,
            is_paper=self.is_paper_mode,
            # Initialize dynamic TP tracking
            current_tp_level=1,
            dynamic_tp_target=conf_config.tp_min
        )
        db.add(order)
        await db.flush()  # Flush to get the order ID
        
        # Create transaction record
        transaction = Transaction(
            order_id=order.id,
            binance_order_id=binance_order_id,
            pair=pair,
            action=f"OPEN_{direction}",
            price=actual_price,
            quantity=quantity,
            investment=investment,
            leverage=leverage,
            notional_value=notional_value,
            fee=entry_fee,
            is_paper=self.is_paper_mode
        )
        db.add(transaction)
        
        await db.commit()
        await db.refresh(order)
        
        # Save paper balance to database (critical - was missing before!)
        if self.is_paper_mode:
            await self.save_state(db)
        
        # Force reset WebSocket tracking for new order (fresh start from entry price)
        # This ensures we track high/low from the actual entry, not from previous orders
        websocket_tracker.force_reset_tracking(pair, actual_price)
        await websocket_tracker.subscribe_pair(pair, actual_price)
        
        logger.info(f"[ORDER CREATED] {pair}: {direction} {confidence} - ID={order.id}, Investment=${investment:.2f}")
        
        return order
    
    async def close_position(
        self,
        db: AsyncSession,
        order: Order,
        current_price: float,
        reason: str = "MANUAL"
    ) -> Optional[Order]:
        """Close an existing position"""
        if order.status != "OPEN":
            return None
        
        # CRITICAL: Never close with invalid price - this would cause -100% P&L
        if current_price is None or current_price <= 0:
            logger.error(f"[CLOSE_BLOCKED] {order.pair}: Attempted to close with invalid price={current_price}, reason={reason}")
            return None
        
        # Calculate exit fee
        notional_at_close = order.quantity * current_price
        exit_fee = notional_at_close * config.trading_config.trading_fee
        total_fee = order.entry_fee + exit_fee
        
        # Calculate P&L
        pnl_data = calculate_pnl(
            direction=order.direction,
            entry_price=order.entry_price,
            current_price=current_price,
            quantity=order.quantity,
            leverage=order.leverage,
            entry_fee=order.entry_fee,
            exit_fee=exit_fee
        )
        
        # Execute trade
        if not self.is_paper_mode:
            symbol = order.pair.replace('USDT', '/USDT:USDT')
            result = await binance_service.close_position(
                symbol=symbol,
                side=order.direction,
                amount=order.quantity
            )
            if not result:
                return None
        else:
            # Paper trade - return investment + P&L
            self.paper_balance += order.investment + pnl_data['pnl']
        
        # Update order
        order.status = "CLOSED"
        order.exit_price = current_price
        order.exit_fee = exit_fee
        order.total_fee = total_fee
        order.pnl = pnl_data['pnl']
        order.pnl_percentage = pnl_data['pnl_percentage']
        order.closed_at = datetime.utcnow()
        order.close_reason = reason
        
        # Create transaction record
        transaction = Transaction(
            order_id=order.id,
            binance_order_id=order.binance_order_id,
            pair=order.pair,
            action=f"CLOSE_{order.direction}",
            price=current_price,
            quantity=order.quantity,
            investment=order.investment,
            leverage=order.leverage,
            notional_value=notional_at_close,
            fee=exit_fee,
            is_paper=order.is_paper
        )
        db.add(transaction)
        
        await db.commit()
        await db.refresh(order)
        
        # Save paper balance
        if self.is_paper_mode:
            await self.save_state(db)
        
        # Keep WebSocket subscription active for all top pairs (real-time price display)
        # Pairs are subscribed in scan_and_trade() and stay subscribed
        
        return order
    
    async def update_open_positions(self, db: AsyncSession) -> List[Dict]:
        """Update all open positions with current prices and check exit conditions"""
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        open_orders = result.scalars().all()
        
        updates = []
        
        for order in open_orders:
            # Get current price
            symbol = order.pair.replace('USDT', '/USDT:USDT')
            current_price = await binance_service.get_current_price(symbol)
            
            if current_price <= 0:
                continue
            
            # Update current price
            order.current_price = current_price
            
            # Get real-time high/low from WebSocket tracker (more accurate)
            ws_high, ws_low = websocket_tracker.get_high_low(order.pair)
            
            # Also update WebSocket tracker with current price (fallback/sync)
            websocket_tracker.update_price(order.pair, current_price)
            
            # Use the best of WebSocket tracking and order tracking
            if order.direction == "LONG":
                # For LONG, track highest price
                old_high = order.high_price_since_entry
                
                # DEFENSIVE: If high_price is 0, None, or invalid, initialize to entry price
                # This fixes corrupted orders from race conditions during creation
                if order.high_price_since_entry is None or order.high_price_since_entry <= 0:
                    order.high_price_since_entry = order.entry_price
                    logger.warning(f"[TRACKING_FIX] {order.pair} LONG: Initialized high_price from {old_high} to entry {order.entry_price}")
                    old_high = order.high_price_since_entry  # Update old_high for comparison
                
                # Apply normal tracking logic - only update if new price is HIGHER
                if ws_high is not None and ws_high > 0:
                    if ws_high > order.high_price_since_entry:
                        order.high_price_since_entry = ws_high
                if current_price > 0 and current_price > order.high_price_since_entry:
                    order.high_price_since_entry = current_price
                    
                # Log if high_price was updated
                if order.current_tp_level and order.current_tp_level >= 2 and old_high != order.high_price_since_entry:
                    logger.info(f"[TRACKING] {order.pair} LONG L{order.current_tp_level}: HIGH updated {old_high} -> {order.high_price_since_entry} (ws_high={ws_high})")
            else:
                # For SHORT, track lowest price
                old_low = order.low_price_since_entry
                
                # DEFENSIVE: If low_price is 0, None, or invalid, initialize to entry price
                # This fixes corrupted orders from race conditions during creation
                if order.low_price_since_entry is None or order.low_price_since_entry <= 0:
                    order.low_price_since_entry = order.entry_price
                    logger.warning(f"[TRACKING_FIX] {order.pair} SHORT: Initialized low_price from {old_low} to entry {order.entry_price}")
                    old_low = order.low_price_since_entry  # Update old_low for comparison
                
                # Apply normal tracking logic - only update if new price is LOWER
                if ws_low is not None and ws_low > 0:
                    if ws_low < order.low_price_since_entry:
                        order.low_price_since_entry = ws_low
                if current_price > 0 and current_price < order.low_price_since_entry:
                    order.low_price_since_entry = current_price
                    
                # Log if low_price was updated
                if order.current_tp_level and order.current_tp_level >= 2 and old_low != order.low_price_since_entry:
                    logger.info(f"[TRACKING] {order.pair} SHORT L{order.current_tp_level}: LOW updated {old_low} -> {order.low_price_since_entry} (ws_low={ws_low})")
            
            # Get cached indicator data for this pair
            pair_result = await db.execute(
                select(PairData).where(PairData.pair == order.pair)
            )
            pair_data = pair_result.scalar_one_or_none()
            
            # Extract EMA values for trend check
            ema5 = pair_data.ema5 if pair_data else None
            ema8 = pair_data.ema8 if pair_data else None
            ema13 = pair_data.ema13 if pair_data else None
            ema20 = pair_data.ema20 if pair_data else None
            
            # Check max holding time
            max_hold = config.trading_config.investment.max_holding_time_minutes
            if max_hold > 0 and order.opened_at:
                from datetime import timezone
                opened = order.opened_at.replace(tzinfo=timezone.utc) if order.opened_at.tzinfo is None else order.opened_at
                age_minutes = (datetime.now(timezone.utc) - opened).total_seconds() / 60
                if age_minutes >= max_hold:
                    logger.info(f"[MAX_HOLD_TIME] {order.pair} {order.direction}: held {age_minutes:.0f}min >= limit {max_hold}min, force closing")
                    closed_order = await self.close_position(db, order, current_price, "MAX_HOLD_TIME")
                    if closed_order:
                        updates.append({
                            "order_id": closed_order.id,
                            "pair": closed_order.pair,
                            "action": "CLOSED",
                            "reason": "MAX_HOLD_TIME",
                            "pnl": closed_order.pnl,
                            "tp_level": order.current_tp_level or 1
                        })
                    continue
            
            # Check exit conditions (including fees for accurate SL/TP)
            exit_result = check_exit_conditions(
                direction=order.direction,
                entry_price=order.entry_price,
                current_price=current_price,
                leverage=order.leverage,
                confidence=order.confidence,
                peak_pnl=order.peak_pnl,
                quantity=order.quantity,
                entry_fee=order.entry_fee,
                investment=order.investment,
                high_price=order.high_price_since_entry,
                low_price=order.low_price_since_entry,
                # Pass indicators for dynamic TP
                ema5=ema5,
                ema8=ema8,
                ema13=ema13,
                ema20=ema20,
                current_tp_level=order.current_tp_level or 1,
                dynamic_tp_target=order.dynamic_tp_target
            )
            
            order.peak_pnl = exit_result.get("peak_pnl", order.peak_pnl)
            reason = exit_result.get("reason")
            
            if exit_result.get("should_close"):
                closed_order = await self.close_position(db, order, current_price, reason)
                if closed_order:
                    updates.append({
                        "order_id": closed_order.id,
                        "pair": closed_order.pair,
                        "action": "CLOSED",
                        "reason": reason,
                        "pnl": closed_order.pnl,
                        "tp_level": exit_result.get("tp_level", 1)
                    })
            elif reason == "EXTEND_TP":
                # Extend TP target - update order fields
                new_tp_level = exit_result.get("new_tp_level", order.current_tp_level + 1)
                new_tp_target = exit_result.get("new_tp_target")
                
                logger.info(f"[EXTEND_TP] {order.pair} {order.direction}: L{order.current_tp_level} -> L{new_tp_level} (target: {new_tp_target:.4f}%)")
                
                order.current_tp_level = new_tp_level
                order.dynamic_tp_target = new_tp_target
                
                # NOTE: Do NOT reset high/low tracking when extending TP!
                # We want to keep the best price ever seen for trailing stop calculation.
                # Otherwise, if price reverses after extension, we lose the profit reference.
                
                await db.commit()
                
                updates.append({
                    "order_id": order.id,
                    "pair": order.pair,
                    "action": "EXTEND_TP",
                    "new_level": new_tp_level,
                    "new_target": new_tp_target
                })
            else:
                await db.commit()
        
        return updates
    
    async def scan_and_trade(self, db: AsyncSession) -> List[Dict]:
        """Scan top pairs and open positions based on signals"""
        if not self.is_running:
            return []
        
        import time
        now = time.time()
        if now - self._last_scan_time < 30:
            return []
        
        logger.info(f"[SCAN] Starting scan_and_trade cycle...")
        actions = []
        
        # Get top pairs based on config limit
        pairs_limit = config.trading_config.trading_pairs_limit
        top_pairs = await binance_service.get_top_futures_pairs(pairs_limit)
        
        # Subscribe all top pairs to WebSocket for real-time price streaming
        for pair_info in top_pairs:
            await websocket_tracker.subscribe_pair(pair_info['pair'])
        
        for pair_info in top_pairs:
            pair = pair_info['pair']
            symbol = pair_info['symbol']
            volume_24h = pair_info['volume_24h']  # Get 24h volume from tickers
            
            # Get OHLCV data
            ohlcv = await binance_service.get_ohlcv(symbol, '5m', 100)
            if not ohlcv:
                continue
            
            # Calculate indicators
            indicators = calculate_indicators(ohlcv)
            if not indicators:
                continue
            
            # Skip pairs with degenerate data (no price movement)
            # RSI of exactly 0 or 100 means all candles moved in one direction (or not at all)
            # Null ADX means not enough price variation to compute trend strength
            rsi_val = indicators.get('rsi')
            adx_val = indicators.get('adx')
            if rsi_val is not None and (rsi_val >= 99.9 or rsi_val <= 0.1):
                logger.debug(f"[SKIP] {pair}: Degenerate RSI={rsi_val:.1f} (no price variation)")
                continue
            if adx_val is None:
                logger.debug(f"[SKIP] {pair}: ADX is null (insufficient price data)")
                continue
            
            # Get signal
            signal, confidence = get_signal(
                ema5=indicators.get('ema5'),
                ema8=indicators.get('ema8'),
                ema13=indicators.get('ema13'),
                ema20=indicators.get('ema20'),
                rsi=indicators.get('rsi'),
                adx=indicators.get('adx'),
                volume=indicators.get('volume'),
                avg_volume=indicators.get('avg_volume'),
                price=indicators.get('price')
            )
            
            # Log signal for debugging (only log tradeable signals)
            if signal in ["LONG", "SHORT"]:
                logger.info(f"[SIGNAL-FOUND] {pair}: {signal} {confidence} - RSI={indicators.get('rsi'):.1f}, ADX={indicators.get('adx')}")
            
            # Update pair data cache with 24h volume from tickers
            await self.update_pair_data(db, pair, indicators, signal, confidence, volume_24h)
            
            # Open position if we have a valid signal
            if signal in ["LONG", "SHORT"] and confidence and confidence != "NO_TRADE":
                logger.info(f"[SIGNAL] {pair}: {signal} with {confidence} confidence - Opening position...")
                # Compute entry gap for tracking
                entry_gap = None
                if indicators.get('ema5') and indicators.get('ema20') and indicators['price'] > 0:
                    entry_gap = round(abs((indicators['ema5'] - indicators['ema20']) / indicators['price'] * 100), 4)
                order = await self.open_position(
                    db=db,
                    pair=pair,
                    direction=signal,
                    confidence=confidence,
                    current_price=indicators['price'],
                    entry_gap=entry_gap
                )
                
                if order:
                    actions.append({
                        "pair": pair,
                        "action": f"OPENED_{signal}",
                        "confidence": confidence,
                        "price": indicators['price']
                    })
            
            # Delay between API calls to avoid Binance rate limits
            # (0.3s per pair = ~15s for 50 pairs, well within limits)
            await asyncio.sleep(0.3)
        
        self._last_scan_time = time.time()
        return actions
    
    async def update_pair_data(
        self,
        db: AsyncSession,
        pair: str,
        indicators: Dict,
        signal: str,
        confidence: Optional[str],
        volume_24h: Optional[float] = None
    ):
        """Update pair data cache in database"""
        result = await db.execute(
            select(PairData).where(PairData.pair == pair)
        )
        pair_data = result.scalar_one_or_none()
        
        # Use provided 24h volume, or fall back to candle volume
        actual_volume_24h = volume_24h if volume_24h is not None else indicators.get('volume', 0)
        
        if pair_data:
            pair_data.price = indicators.get('price', 0)
            pair_data.ema5 = indicators.get('ema5')
            pair_data.ema8 = indicators.get('ema8')
            pair_data.ema13 = indicators.get('ema13')
            pair_data.ema20 = indicators.get('ema20')
            pair_data.rsi = indicators.get('rsi')
            pair_data.adx = indicators.get('adx')
            pair_data.volume_24h = actual_volume_24h
            pair_data.avg_volume = indicators.get('avg_volume')
            pair_data.signal = signal
            pair_data.confidence = confidence
            pair_data.updated_at = datetime.utcnow()
        else:
            pair_data = PairData(
                pair=pair,
                price=indicators.get('price', 0),
                ema5=indicators.get('ema5'),
                ema8=indicators.get('ema8'),
                ema13=indicators.get('ema13'),
                ema20=indicators.get('ema20'),
                rsi=indicators.get('rsi'),
                adx=indicators.get('adx'),
                volume_24h=actual_volume_24h,
                avg_volume=indicators.get('avg_volume'),
                signal=signal,
                confidence=confidence
            )
            db.add(pair_data)
        
        await db.commit()
    
    async def check_realtime_stop_loss(self, pair: str, current_price: float):
        """
        Real-time stop loss AND trailing stop check called by WebSocket on each price update.
        This provides instant protection instead of waiting for polling cycles.
        - Stop loss / break-even SL: triggers when P&L drops below threshold.
        - Trailing stop: triggers when price pulls back X% from high/low (only in post-TP zone).
        """
        global _open_orders_cache
        
        # CRITICAL: Never process invalid prices
        if current_price is None or current_price <= 0:
            return
        
        # Check cache first for fast lookup
        async with _cache_lock:
            cached_orders = _open_orders_cache.get(pair, [])
        
        if not cached_orders:
            return  # No open orders for this pair
        
        # Check each cached order
        for order_info in cached_orders:
            order_id = order_info['id']
            direction = order_info['direction']
            entry_price = order_info['entry_price']
            stop_loss_pct = order_info['stop_loss']
            quantity = order_info['quantity']
            entry_fee = order_info['entry_fee']
            cached_peak_pnl = order_info.get('peak_pnl', 0.0)
            breakeven_trigger = order_info.get('breakeven_trigger', 999)
            breakeven_offset = order_info.get('breakeven_offset', 0.0)
            pullback_trigger = order_info.get('pullback_trigger', 0.04)
            
            # Skip if entry data is invalid
            if entry_price <= 0 or quantity <= 0:
                logger.warning(f"[REALTIME_SL] {pair}: Invalid cache data - entry_price={entry_price}, quantity={quantity}")
                continue
            
            # Track high/low prices in real-time (updated on every tick)
            if direction == "LONG":
                cached_high = order_info.get('high_price', entry_price)
                if current_price > cached_high:
                    order_info['high_price'] = current_price
                high_price = order_info.get('high_price', entry_price)
            else:
                cached_low = order_info.get('low_price', entry_price)
                if current_price < cached_low:
                    order_info['low_price'] = current_price
                low_price = order_info.get('low_price', entry_price)
            
            # Calculate current P&L with fees
            entry_notional = entry_price * quantity
            current_notional = current_price * quantity
            exit_fee = current_notional * config.trading_config.trading_fee
            total_fees = entry_fee + exit_fee
            
            if direction == "LONG":
                pnl = (current_price - entry_price) * quantity - total_fees
            else:
                pnl = (entry_price - current_price) * quantity - total_fees
            
            pnl_pct = (pnl / entry_notional) * 100
            
            # Track peak P&L in real-time for break-even decisions
            current_peak = max(cached_peak_pnl, pnl_pct) if pnl_pct > 0 else cached_peak_pnl
            order_info['peak_pnl'] = current_peak
            
            # Get TP target to determine if trailing stop would be active
            tp_level = order_info.get('current_tp_level', 1)
            conf = config.trading_config.confidence_levels.get(
                order_info.get('confidence', 'LOW'))
            tp_min = conf.tp_min if conf else 0.1
            effective_tp_target = tp_level * tp_min if tp_level > 1 else tp_min
            
            # For REAL-TIME checks, trailing stop only activates at L2+ (TP already extended).
            # At L1, the polling loop handles TP extension first, then trailing stop.
            # This prevents the real-time trailing stop from firing before TP can be extended.
            trailing_stop_would_be_active = tp_level >= 2
            
            # Apply break-even logic ONLY in pre-TP zone (trailing stop not active)
            effective_sl = stop_loss_pct
            breakeven_active = False
            if not trailing_stop_would_be_active and current_peak >= breakeven_trigger:
                breakeven_active = True
                effective_sl = breakeven_offset
            
            # Check if stop loss triggered
            if pnl_pct <= effective_sl:
                reason_prefix = "BREAKEVEN_SL" if breakeven_active else "STOP_LOSS"
                logger.warning(f"[REALTIME_{reason_prefix}] {pair} {direction}: pnl={pnl_pct:.4f}% <= effective_sl={effective_sl}% (original_sl={stop_loss_pct}%, peak={current_peak:.4f}%) - CLOSING NOW!")
                
                # Close the order immediately using a new database session
                try:
                    async with AsyncSessionLocal() as db:
                        # Re-fetch the order to ensure it's still open
                        result = await db.execute(
                            select(Order).where(
                                and_(Order.id == order_id, Order.status == "OPEN")
                            )
                        )
                        order = result.scalar_one_or_none()
                        
                        if order:
                            # Close the order
                            await self.close_position(
                                db, order, current_price, 
                                f"{reason_prefix} L{order.current_tp_level}"
                            )
                            logger.info(f"[REALTIME_{reason_prefix}] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                            
                            # Remove from cache
                            async with _cache_lock:
                                _open_orders_cache[pair] = [
                                    o for o in _open_orders_cache.get(pair, []) 
                                    if o['id'] != order_id
                                ]
                except Exception as e:
                    logger.error(f"[REALTIME_SL] Error closing {pair}: {e}")
                continue  # Already handled, skip trailing stop check
            
            # Real-time trailing stop check (only when trailing stop is active)
            if trailing_stop_would_be_active:
                should_close_trailing = False
                tp_level = order_info.get('current_tp_level', 1)
                
                if direction == "LONG" and high_price and high_price > 0:
                    price_drop_pct = ((high_price - current_price) / high_price) * 100
                    if price_drop_pct >= pullback_trigger:
                        # SAFEGUARD: Never close at a loss via trailing stop
                        if pnl_pct >= 0:
                            should_close_trailing = True
                            logger.warning(f"[REALTIME_TRAILING] {pair} LONG L{tp_level}: high={high_price:.6f}, current={current_price:.6f}, drop={price_drop_pct:.4f}% >= {pullback_trigger}%, pnl={pnl_pct:.4f}% - CLOSING NOW!")
                        else:
                            logger.info(f"[REALTIME_TRAILING_BLOCKED] {pair} LONG L{tp_level}: Would close at loss ({pnl_pct:.4f}%), waiting for recovery or SL")
                
                elif direction == "SHORT" and low_price and low_price > 0:
                    price_rise_pct = ((current_price - low_price) / low_price) * 100
                    if price_rise_pct >= pullback_trigger:
                        # SAFEGUARD: Never close at a loss via trailing stop
                        if pnl_pct >= 0:
                            should_close_trailing = True
                            logger.warning(f"[REALTIME_TRAILING] {pair} SHORT L{tp_level}: low={low_price:.6f}, current={current_price:.6f}, rise={price_rise_pct:.4f}% >= {pullback_trigger}%, pnl={pnl_pct:.4f}% - CLOSING NOW!")
                        else:
                            logger.info(f"[REALTIME_TRAILING_BLOCKED] {pair} SHORT L{tp_level}: Would close at loss ({pnl_pct:.4f}%), waiting for recovery or SL")
                
                if should_close_trailing:
                    try:
                        async with AsyncSessionLocal() as db:
                            result = await db.execute(
                                select(Order).where(
                                    and_(Order.id == order_id, Order.status == "OPEN")
                                )
                            )
                            order = result.scalar_one_or_none()
                            
                            if order:
                                await self.close_position(
                                    db, order, current_price,
                                    f"TRAILING_STOP L{order.current_tp_level}"
                                )
                                logger.info(f"[REALTIME_TRAILING] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                
                                # Remove from cache
                                async with _cache_lock:
                                    _open_orders_cache[pair] = [
                                        o for o in _open_orders_cache.get(pair, [])
                                        if o['id'] != order_id
                                    ]
                    except Exception as e:
                        logger.error(f"[REALTIME_TRAILING] Error closing {pair}: {e}")
    
    async def update_orders_cache(self, db: AsyncSession):
        """Update the open orders cache for real-time stop loss checking.
        Includes peak_pnl and breakeven config for break-even SL logic."""
        global _open_orders_cache
        
        result = await db.execute(
            select(Order).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        orders = result.scalars().all()
        
        # Build new cache
        new_cache: Dict[str, List[Dict]] = {}
        for order in orders:
            # Get config for this order's confidence level
            conf_config = config.trading_config.confidence_levels.get(order.confidence)
            if not conf_config:
                continue
            
            order_info = {
                'id': order.id,
                'direction': order.direction,
                'entry_price': order.entry_price,
                'quantity': order.quantity,
                'entry_fee': order.entry_fee,
                'confidence': order.confidence,
                'stop_loss': conf_config.stop_loss,
                'current_tp_level': order.current_tp_level,
                'peak_pnl': order.peak_pnl or 0.0,
                'breakeven_trigger': conf_config.breakeven_trigger,
                'breakeven_offset': conf_config.breakeven_offset,
                'high_price': order.high_price_since_entry or order.entry_price,
                'low_price': order.low_price_since_entry or order.entry_price,
                'pullback_trigger': conf_config.pullback_trigger
            }
            
            if order.pair not in new_cache:
                new_cache[order.pair] = []
            new_cache[order.pair].append(order_info)
        
        async with _cache_lock:
            _open_orders_cache = new_cache
        
        logger.debug(f"[CACHE] Updated orders cache: {len(orders)} orders across {len(new_cache)} pairs")


# Global trading engine instance
trading_engine = TradingEngine()


async def realtime_stop_loss_callback(pair: str, price: float):
    """Callback function for WebSocket price updates to check stop loss in real-time"""
    await trading_engine.check_realtime_stop_loss(pair, price)
