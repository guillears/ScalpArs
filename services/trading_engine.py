"""
SCALPARS Trading Platform - Trading Engine
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sqlalchemy import select, update, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from models import Order, Transaction, BotState, PairData, BnbSwapLog
from database import AsyncSessionLocal
import config
from config import save_trading_config, TradingConfig
from services.binance_service import binance_service
from services.indicators import calculate_indicators, get_signal, check_exit_conditions, calculate_pnl, determine_macro_regime, is_signal_direction_active
from services.websocket_tracker import websocket_tracker

logger = logging.getLogger(__name__)

OHLCV_BATCH_SIZE = 10
OHLCV_BATCH_DELAY = 5.0

# Cache for open orders to enable fast real-time stop loss checks
_open_orders_cache: Dict[str, List[Dict]] = {}  # pair -> list of order info
_cache_lock = asyncio.Lock()
_close_lock = asyncio.Lock()

# Orders whose exit failed but whose position still exists on Binance.
# Maps order_id -> attempt count. Retried each monitor cycle until success,
# EXTERNAL_CLOSE, or max retries reached.
_exit_retry_queue: Dict[int, int] = {}
_EXIT_RETRY_MAX = 30

# Current BTC macro regime, updated by scan_and_trade, read by update_open_positions
_current_btc_regime: str = "NEUTRAL"

# Global volume ratio: sum(current volumes) / sum(20-bar avg volumes) across top pairs.
# Computed at end of each scan, used by next scan cycle as a market regime gate.
_global_volume_ratio: float = 1.0

# Phantom Tick Momentum shadow configs: (label, windows, delta_or_deltas)
# delta_or_deltas: float = uniform delta for all windows, list = per-window deltas
_SHADOW_TICK_CONFIGS = [
    ('a', [15, 30, 45], 0.15),
    ('b', [30, 45, 60], 0.12),
    ('c', [30, 45, 60], 0.15),
    ('d', [30, 60, 90], 0.12),
    ('e', [30, 60, 90], 0.15),
    ('f', [30, 60, 90], [0.08, 0.12, 0.18]),
    ('g', [60, 90, 120], 0.15),
]


def _check_tick_momentum_fade(tick_buf, now, windows, per_window_deltas, direction):
    """Check if all windows confirm momentum fading. Returns True if confirmed."""
    min_window = min(windows) if windows else 15
    if len(tick_buf) < 5 or (now - tick_buf[0][0]) < min_window:
        return False

    smooth_cutoff = now - 5.0
    smooth_prices = [p for t, p in tick_buf if t >= smooth_cutoff]
    smoothed = sum(smooth_prices) / len(smooth_prices) if smooth_prices else tick_buf[-1][1]

    for w, delta in zip(windows, per_window_deltas):
        target_time = now - w
        best_tick = None
        best_diff = float('inf')
        for t, p in tick_buf:
            diff = abs(t - target_time)
            if diff < best_diff:
                best_diff = diff
                best_tick = p
        if best_tick is None or best_diff > w * 0.5:
            return False
        price_change_pct = ((smoothed - best_tick) / best_tick) * 100
        if direction == "LONG" and price_change_pct > -delta:
            return False
        elif direction == "SHORT" and price_change_pct < delta:
            return False
    return True


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
        self._initialized = False
        self._post_exit_tracking: Dict[int, dict] = {}
        self._rsi3_history: Dict[int, list] = {}  # per-order RSI history for 3-drop detection
        # BNB fee management
        self.paper_bnb_balance_usd: float = config.trading_config.paper_bnb_initial_usd
        self._bnb_emergency_threshold: float = 0.0
        self._bnb_projected_need: float = 0.0
        self._bnb_burn_rate: float = 0.0
        self._last_bnb_check: Optional[datetime] = None
    
    async def initialize(self, db: AsyncSession):
        """Initialize engine state from database (only on first call)"""
        if self._initialized:
            return
        
        result = await db.execute(select(BotState).limit(1))
        state = result.scalar_one_or_none()
        
        if state:
            self.is_running = state.is_running
            self.is_paper_mode = state.is_paper_mode
            self.paper_balance = state.paper_balance
            self.paper_bnb_balance_usd = getattr(state, 'paper_bnb_balance_usd', None) or config.trading_config.paper_bnb_initial_usd
            self.total_runtime_seconds = state.total_runtime_seconds
            if state.is_running and state.started_at:
                self.started_at = state.started_at
        else:
            # Create initial state
            state = BotState(
                is_running=False,
                is_paper_mode=True,
                paper_balance=config.trading_config.paper_balance,
                paper_bnb_balance_usd=config.trading_config.paper_bnb_initial_usd,
                total_runtime_seconds=0
            )
            db.add(state)
            await db.commit()
        
        # Recalculate paper_balance from orders to self-heal any accumulated drift
        if self.is_paper_mode:
            await self._recalculate_paper_balance(db)
            await self.save_state(db)
        
        self._initialized = True
    
    async def save_state(self, db: AsyncSession):
        """Save engine state to database"""
        result = await db.execute(select(BotState).limit(1))
        state = result.scalar_one_or_none()
        
        if state:
            state.is_running = self.is_running
            state.is_paper_mode = self.is_paper_mode
            state.paper_balance = self.paper_balance
            state.paper_bnb_balance_usd = self.paper_bnb_balance_usd
            state.total_runtime_seconds = self.total_runtime_seconds
            state.started_at = self.started_at
            state.updated_at = datetime.utcnow()
        else:
            state = BotState(
                is_running=self.is_running,
                is_paper_mode=self.is_paper_mode,
                paper_balance=self.paper_balance,
                paper_bnb_balance_usd=self.paper_bnb_balance_usd,
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
            "paper_bnb_balance_usd": round(self.paper_bnb_balance_usd, 2),
            "bnb_burn_rate": round(self._bnb_burn_rate, 2),
            "bnb_emergency_threshold": round(self._bnb_emergency_threshold, 2),
            "bnb_last_check": self._last_bnb_check.isoformat() if self._last_bnb_check else None,
            "runtime_seconds": runtime,
            "runtime_formatted": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
            "global_volume_ratio": round(_global_volume_ratio, 4)
        }
    
    async def _recalculate_paper_balance(self, db: AsyncSession) -> float:
        """Recalculate paper_balance from DB as source of truth.
        
        Formula: initial + closed_pnl + closed_fees - open_margin - bnb_swaps
        
        Fees are paid from BNB, not USDT. Since Order.pnl is net of fees,
        we add back closed fees to avoid double-counting (fees already tracked
        in the BNB balance via _deduct_fee_from_bnb).
        """
        initial = config.trading_config.paper_balance
        closed_pnl_result = await db.execute(
            select(func.coalesce(func.sum(Order.pnl), 0)).where(
                and_(Order.status == "CLOSED", Order.is_paper == True)
            )
        )
        total_closed_pnl = closed_pnl_result.scalar() or 0
        closed_fees_result = await db.execute(
            select(func.coalesce(func.sum(Order.total_fee), 0)).where(
                and_(Order.status == "CLOSED", Order.is_paper == True)
            )
        )
        total_closed_fees = closed_fees_result.scalar() or 0
        open_margin_result = await db.execute(
            select(func.coalesce(func.sum(Order.investment), 0)).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        total_open_margin = open_margin_result.scalar() or 0
        bnb_swap_result = await db.execute(
            select(func.coalesce(func.sum(BnbSwapLog.amount_usdt), 0)).where(
                BnbSwapLog.is_paper == True
            )
        )
        total_bnb_swaps = bnb_swap_result.scalar() or 0
        correct_balance = initial + total_closed_pnl + total_closed_fees - total_open_margin - total_bnb_swaps
        if abs(correct_balance - self.paper_balance) > 0.01:
            logger.warning(
                f"[BALANCE_SYNC] Correcting drift: "
                f"in_memory={self.paper_balance:.2f}, db_correct={correct_balance:.2f}, "
                f"diff={self.paper_balance - correct_balance:.2f}"
            )
        self.paper_balance = correct_balance
        return correct_balance

    async def _recalculate_paper_bnb(self, db: AsyncSession) -> float:
        """Recalculate paper BNB balance from DB as source of truth.
        
        Formula: initial_bnb + sum(swap inflows) - sum(all fees from paper orders)
        """
        initial = config.trading_config.paper_bnb_initial_usd
        swap_result = await db.execute(
            select(func.coalesce(func.sum(BnbSwapLog.amount_usdt), 0)).where(
                BnbSwapLog.is_paper == True
            )
        )
        total_swaps = swap_result.scalar() or 0
        fee_result = await db.execute(
            select(func.coalesce(func.sum(Order.total_fee), 0)).where(
                and_(Order.status == "CLOSED", Order.is_paper == True)
            )
        )
        total_closed_fees = fee_result.scalar() or 0
        open_entry_fee_result = await db.execute(
            select(func.coalesce(func.sum(Order.entry_fee), 0)).where(
                and_(Order.status == "OPEN", Order.is_paper == True)
            )
        )
        total_open_entry_fees = open_entry_fee_result.scalar() or 0
        correct = initial + total_swaps - total_closed_fees - total_open_entry_fees
        self.paper_bnb_balance_usd = correct
        return correct

    async def _deduct_fee_from_bnb(self, fee_usd: float, db: AsyncSession):
        """Deduct a fee (in USDT) from the paper BNB balance and check emergency threshold."""
        if not self.is_paper_mode or fee_usd <= 0:
            return
        self.paper_bnb_balance_usd -= fee_usd
        if self.paper_bnb_balance_usd < 0:
            self.paper_bnb_balance_usd = 0
        if config.trading_config.bnb_swap_enabled and self._bnb_emergency_threshold > 0:
            if self.paper_bnb_balance_usd < self._bnb_emergency_threshold:
                await self._execute_bnb_swap(db, swap_type="emergency")

    async def _execute_bnb_swap(self, db: AsyncSession, swap_type: str = "scheduled"):
        """Execute a USDT→BNB swap (paper or live)."""
        tc = config.trading_config
        if not tc.bnb_swap_enabled:
            return
        
        target = self._bnb_projected_need if self._bnb_projected_need > 0 else tc.paper_bnb_initial_usd * 0.4
        
        if self.is_paper_mode:
            current_bnb = self.paper_bnb_balance_usd
            if current_bnb >= target:
                return
            shortfall = target - current_bnb
            available_usdt = await self._recalculate_paper_balance(db)
            min_investment = tc.investment.min_investment_size
            if available_usdt - shortfall < min_investment:
                shortfall = max(0, available_usdt - min_investment)
            if shortfall <= 0:
                logger.warning(f"[BNB_SWAP] Cannot swap: insufficient USDT (available={available_usdt:.2f})")
                return
            
            bnb_price = await binance_service.get_bnb_price()
            if bnb_price <= 0:
                bnb_price = 600.0  # fallback for paper mode
            
            pre_usdt = self.paper_balance
            pre_bnb = self.paper_bnb_balance_usd
            self.paper_bnb_balance_usd += shortfall
            
            swap_log = BnbSwapLog(
                swap_type=swap_type,
                amount_usdt=shortfall,
                bnb_price=bnb_price,
                amount_bnb=round(shortfall / bnb_price, 6),
                pre_bnb_usd=pre_bnb,
                post_bnb_usd=self.paper_bnb_balance_usd,
                pre_usdt=pre_usdt,
                post_usdt=pre_usdt - shortfall,
                burn_rate=self._bnb_burn_rate,
                is_paper=True
            )
            db.add(swap_log)
            await db.commit()
            
            await self._recalculate_paper_balance(db)
            await self.save_state(db)
            logger.info(
                f"[BNB_SWAP] Paper {swap_type}: swapped ${shortfall:.2f} USDT → "
                f"{shortfall/bnb_price:.4f} BNB @ ${bnb_price:.2f}. "
                f"BNB: ${pre_bnb:.2f} → ${self.paper_bnb_balance_usd:.2f}"
            )
        else:
            balance = await binance_service.get_balance()
            bnb_price = await binance_service.get_bnb_price()
            if bnb_price <= 0:
                return
            current_bnb_usd = balance['bnb_total'] * bnb_price
            if current_bnb_usd >= target:
                return
            shortfall = target - current_bnb_usd
            available_usdt = balance['usdt_free']
            min_investment = tc.investment.min_investment_size
            if available_usdt - shortfall < min_investment:
                shortfall = max(0, available_usdt - min_investment)
            if shortfall <= 5:
                return
            
            result = await binance_service.buy_bnb(shortfall)
            if not result:
                return
            
            new_balance = await binance_service.get_balance()
            swap_log = BnbSwapLog(
                swap_type=swap_type,
                amount_usdt=result['cost_usdt'],
                bnb_price=result['price'],
                amount_bnb=result['bnb_amount'],
                pre_bnb_usd=current_bnb_usd,
                post_bnb_usd=new_balance['bnb_total'] * result['price'],
                pre_usdt=available_usdt,
                post_usdt=new_balance['usdt_free'],
                burn_rate=self._bnb_burn_rate,
                is_paper=False
            )
            db.add(swap_log)
            await db.commit()
            logger.info(
                f"[BNB_SWAP] Live {swap_type}: bought {result['bnb_amount']:.4f} BNB "
                f"for ${result['cost_usdt']:.2f} @ ${result['price']:.2f}"
            )

    async def bnb_scheduled_check(self, db: AsyncSession):
        """Scheduled BNB balance check: compute burn rate, project needs, swap if necessary."""
        tc = config.trading_config
        if not tc.bnb_swap_enabled:
            return
        
        self._last_bnb_check = datetime.utcnow()
        
        now = datetime.utcnow()
        cutoff_24h = now - timedelta(hours=24)
        cutoff_12h = now - timedelta(hours=12)
        
        result_24h = await db.execute(
            select(
                func.coalesce(func.sum(Order.total_fee), 0),
                func.count(Order.id)
            ).where(
                and_(Order.status == "CLOSED", Order.closed_at >= cutoff_24h)
            )
        )
        row_24h = result_24h.one()
        fees_24h = float(row_24h[0] or 0)
        count_24h = int(row_24h[1] or 0)
        
        result_12h = await db.execute(
            select(func.coalesce(func.sum(Order.total_fee), 0)).where(
                and_(Order.status == "CLOSED", Order.closed_at >= cutoff_12h)
            )
        )
        fees_12h = float(result_12h.scalar() or 0)
        
        hours_elapsed = 24.0 if count_24h > 0 else 0
        self._bnb_burn_rate = fees_24h / hours_elapsed if hours_elapsed > 0 else 0
        self._bnb_projected_need = self._bnb_burn_rate * tc.bnb_runway_hours
        self._bnb_emergency_threshold = fees_12h
        
        if self._bnb_projected_need <= 0:
            logger.info("[BNB_CHECK] No fee history yet, skipping swap check")
            return
        
        if self.is_paper_mode:
            await self._recalculate_paper_bnb(db)
            current_bnb = self.paper_bnb_balance_usd
        else:
            balance = await binance_service.get_balance()
            bnb_price = await binance_service.get_bnb_price()
            current_bnb = balance['bnb_total'] * bnb_price if bnb_price > 0 else 0
        
        logger.info(
            f"[BNB_CHECK] Burn rate: ${self._bnb_burn_rate:.2f}/hr | "
            f"Projected need ({tc.bnb_runway_hours}h): ${self._bnb_projected_need:.2f} | "
            f"Emergency threshold (12h fees): ${self._bnb_emergency_threshold:.2f} | "
            f"Current BNB: ${current_bnb:.2f}"
        )
        
        if current_bnb < self._bnb_projected_need:
            await self._execute_bnb_swap(db, swap_type="scheduled")

    async def get_available_balance(self, db: AsyncSession) -> float:
        """Get available balance for trading.
        
        For paper trading: always recalculate from DB to prevent drift.
        """
        if self.is_paper_mode:
            return await self._recalculate_paper_balance(db)
        else:
            balance = await binance_service.get_balance()
            return balance['usdt_free']
    
    def calculate_position_size(self, available_balance: float, confidence: str, total_portfolio: float = None) -> Tuple[float, float]:
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
        elif tc.investment.mode == "equal_split":
            max_pos = tc.investment.max_open_positions or 5
            base = total_portfolio if total_portfolio else available_balance
            if tc.investment.reserve_mode == "percentage":
                reserve_from_total = base * (tc.investment.reserve_percentage / 100)
            else:
                reserve_from_total = tc.investment.reserve_fixed
            investment = max(0, base - reserve_from_total) / max_pos
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

    async def _try_maker_entry(
        self, symbol: str, side: str, amount: float, leverage: int,
        direction: str, pair: str, notional_value: float,
        maker_fee_rate: float, taker_fee_rate: float
    ) -> Optional[Dict]:
        """Attempt a maker (limit) entry, falling back to taker (market) on timeout."""
        tc = config.trading_config
        timeout = getattr(tc, 'maker_timeout_seconds', 15)
        offset_ticks = getattr(tc, 'maker_offset_ticks', 2)

        ob = await binance_service.fetch_orderbook(symbol)
        if not ob:
            logger.warning(f"[MAKER_ENTRY] {pair}: Orderbook unavailable, falling back to taker")
            result = await binance_service.create_market_order(symbol, side, amount, leverage)
            if not result:
                return None
            return {
                'id': result['id'], 'price': result['price'],
                'amount': result.get('amount', amount),
                'entry_fee': result.get('fee', notional_value * taker_fee_rate),
                'entry_order_type': 'TAKER_FALLBACK',
            }

        tick_size = await binance_service.get_tick_size(symbol)
        if direction == 'LONG':
            limit_price = ob['best_bid'] - (offset_ticks * tick_size)
        else:
            limit_price = ob['best_ask'] + (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_ENTRY] {pair}: Placing limit {side} @ {limit_price} "
                     f"(bid={ob['best_bid']}, ask={ob['best_ask']}, offset={offset_ticks} ticks)")

        limit_result = await binance_service.create_limit_order(
            symbol=symbol, side=side, amount=amount, price=limit_price, leverage=leverage
        )
        if not limit_result:
            logger.warning(f"[MAKER_ENTRY] {pair}: Limit order failed, falling back to taker")
            result = await binance_service.create_market_order(symbol, side, amount, leverage)
            if not result:
                return None
            return {
                'id': result['id'], 'price': result['price'],
                'amount': result.get('amount', amount),
                'entry_fee': result.get('fee', notional_value * taker_fee_rate),
                'entry_order_type': 'TAKER_FALLBACK',
            }

        order_id = limit_result['id']
        polls = max(1, timeout // 2)
        filled = False

        for i in range(polls):
            await asyncio.sleep(2)
            status = await binance_service.fetch_order_status(symbol, order_id)
            if not status:
                continue
            if status['status'] == 'closed':
                filled = True
                fill_price = status['average'] or limit_price
                fill_amount = status['filled'] or amount
                fill_fee = status.get('fee', 0) or (fill_amount * fill_price * maker_fee_rate)
                logger.info(f"[MAKER_ENTRY] {pair}: Limit FILLED @ {fill_price} after {(i+1)*2}s")
                return {
                    'id': order_id, 'price': fill_price,
                    'amount': fill_amount, 'entry_fee': fill_fee,
                    'entry_order_type': 'MAKER',
                }

        # Timeout -- cancel and check for partial fill
        logger.info(f"[MAKER_ENTRY] {pair}: Timeout after {timeout}s, cancelling limit order")
        await binance_service.cancel_order(symbol, order_id)
        await asyncio.sleep(0.5)

        final_status = await binance_service.fetch_order_status(symbol, order_id)
        filled_qty = final_status['filled'] if final_status else 0

        if filled_qty and filled_qty > 0:
            fill_price = final_status['average'] or limit_price
            fill_fee = final_status.get('fee', 0) or (filled_qty * fill_price * maker_fee_rate)
            logger.info(f"[MAKER_ENTRY] {pair}: Partial fill {filled_qty}/{amount} @ {fill_price}")
            return {
                'id': order_id, 'price': fill_price,
                'amount': filled_qty, 'entry_fee': fill_fee,
                'entry_order_type': 'MAKER',
            }

        # No fill at all -- fall back to market order
        logger.info(f"[MAKER_ENTRY] {pair}: No fill, falling back to market order")
        result = await binance_service.create_market_order(symbol, side, amount, leverage)
        if not result:
            return None
        return {
            'id': result['id'], 'price': result['price'],
            'amount': result.get('amount', amount),
            'entry_fee': result.get('fee', notional_value * taker_fee_rate),
            'entry_order_type': 'TAKER_FALLBACK',
        }

    async def _simulate_maker_entry_paper(
        self, pair: str, direction: str, current_price: float,
        notional_value: float, maker_fee_rate: float, taker_fee_rate: float
    ) -> Dict:
        """Simulate maker entry for paper trading using WebSocket prices."""
        tc = config.trading_config
        timeout = getattr(tc, 'maker_timeout_seconds', 15)
        offset_ticks = getattr(tc, 'maker_offset_ticks', 2)

        # Estimate tick size from price magnitude
        if current_price >= 10000:
            tick_size = 0.10
        elif current_price >= 100:
            tick_size = 0.01
        elif current_price >= 1:
            tick_size = 0.001
        else:
            tick_size = 0.0001

        if direction == 'LONG':
            limit_price = current_price - (offset_ticks * tick_size)
        else:
            limit_price = current_price + (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_PAPER] {pair}: Simulating limit {direction} @ {limit_price} "
                     f"(current={current_price}, offset={offset_ticks} ticks)")

        # Monitor WebSocket prices for the timeout window
        polls = max(1, timeout // 2)
        for i in range(polls):
            await asyncio.sleep(2)
            tracker = websocket_tracker.get_tracker(pair)
            if not tracker or not tracker.last_price:
                continue

            ws_price = tracker.last_price
            if direction == 'LONG' and ws_price <= limit_price:
                logger.info(f"[MAKER_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price,
                    'entry_fee': notional_value * maker_fee_rate,
                    'entry_order_type': 'MAKER',
                }
            elif direction == 'SHORT' and ws_price >= limit_price:
                logger.info(f"[MAKER_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price,
                    'entry_fee': notional_value * maker_fee_rate,
                    'entry_order_type': 'MAKER',
                }

        # No fill -- fallback to current price as taker
        tracker = websocket_tracker.get_tracker(pair)
        fallback_price = tracker.last_price if tracker and tracker.last_price else current_price
        logger.info(f"[MAKER_PAPER] {pair}: No fill after {timeout}s, taker fallback @ {fallback_price}")
        return {
            'price': fallback_price,
            'entry_fee': notional_value * taker_fee_rate,
            'entry_order_type': 'TAKER_FALLBACK',
        }

    async def _fetch_actual_fill_price(self, order, fallback_price: float) -> float:
        """Fetch the actual fill price from Binance trade history for an externally closed order."""
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

        logger.warning(f"[FILL_PRICE] {order.pair}: Using fallback price {fallback_price}")
        return fallback_price

    async def _try_maker_exit(
        self, symbol: str, side: str, amount: float,
        pair: str, direction: str, current_price: float
    ) -> Dict:
        """Attempt a maker (limit) exit, falling back to taker (market) on timeout.
        For LONG exits: sell at best_ask + offset (higher = better).
        For SHORT exits: buy at best_bid - offset (lower = better)."""
        tc = config.trading_config
        timeout = getattr(tc, 'maker_exit_timeout_seconds', 10)
        offset_ticks = getattr(tc, 'maker_exit_offset_ticks', 2)
        maker_fee_rate = getattr(tc, 'maker_fee', 0.00018)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)
        close_side = 'sell' if direction == 'LONG' else 'buy'

        ob = await binance_service.fetch_orderbook(symbol)
        if not ob:
            logger.warning(f"[MAKER_EXIT] {pair}: Orderbook unavailable, falling back to taker")
            try:
                result = await binance_service.close_position(symbol, direction, amount)
                if not result:
                    logger.error(f"[MAKER_EXIT] {pair}: Taker fallback ALSO failed — position NOT closed on Binance")
                    return None
                return {
                    'price': result['price'], 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK',
                }
            except Exception as e:
                logger.critical(
                    f"[MAKER_EXIT] {pair}: Taker fallback CRASHED (orderbook unavailable path): {e}. "
                    f"Returning fallback with current_price={current_price}."
                )
                return {
                    'price': current_price, 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK_RECOVERED',
                }

        tick_size = await binance_service.get_tick_size(symbol)
        if direction == 'LONG':
            limit_price = ob['best_ask'] + (offset_ticks * tick_size)
        else:
            limit_price = ob['best_bid'] - (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_EXIT] {pair}: Placing limit {close_side} @ {limit_price} "
                     f"(bid={ob['best_bid']}, ask={ob['best_ask']}, offset={offset_ticks} ticks)")

        limit_result = await binance_service.create_limit_order(
            symbol=symbol, side=close_side, amount=amount, price=limit_price, leverage=1
        )
        if not limit_result:
            logger.warning(f"[MAKER_EXIT] {pair}: Limit order failed, falling back to taker")
            try:
                result = await binance_service.close_position(symbol, direction, amount)
                if not result:
                    logger.error(f"[MAKER_EXIT] {pair}: Taker fallback ALSO failed — position NOT closed on Binance")
                    return None
                return {
                    'price': result['price'], 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK',
                }
            except Exception as e:
                logger.critical(
                    f"[MAKER_EXIT] {pair}: Taker fallback CRASHED (limit order failed path): {e}. "
                    f"Returning fallback with current_price={current_price}."
                )
                return {
                    'price': current_price, 'fee_rate': taker_fee_rate,
                    'exit_order_type': 'TAKER_FALLBACK_RECOVERED',
                }

        order_id = limit_result['id']
        polls = max(1, timeout // 2)

        for i in range(polls):
            await asyncio.sleep(2)
            status = await binance_service.fetch_order_status(symbol, order_id)
            if not status:
                continue
            if status['status'] == 'closed':
                fill_price = status['average'] or limit_price
                logger.info(f"[MAKER_EXIT] {pair}: Limit FILLED @ {fill_price} after {(i+1)*2}s")
                return {
                    'price': fill_price, 'fee_rate': maker_fee_rate,
                    'exit_order_type': 'MAKER',
                }

        logger.info(f"[MAKER_EXIT] {pair}: Timeout after {timeout}s, cancelling limit order")
        await binance_service.cancel_order(symbol, order_id)
        await asyncio.sleep(0.5)

        final_status = await binance_service.fetch_order_status(symbol, order_id)
        filled_qty = final_status['filled'] if final_status else 0

        if filled_qty and filled_qty > 0:
            fill_price = final_status['average'] or limit_price
            logger.info(f"[MAKER_EXIT] {pair}: Partial fill {filled_qty}/{amount} @ {fill_price}, market closing remainder")
            remainder = amount - filled_qty
            if remainder > 0:
                await binance_service.close_position(symbol, direction, remainder)
            return {
                'price': fill_price, 'fee_rate': maker_fee_rate,
                'exit_order_type': 'MAKER',
            }

        logger.info(f"[MAKER_EXIT] {pair}: No fill, falling back to market order")
        try:
            result = await binance_service.close_position(symbol, direction, amount)
            if not result:
                logger.error(f"[MAKER_EXIT] {pair}: Taker fallback ALSO failed — position NOT closed on Binance")
                return None
            return {
                'price': result['price'], 'fee_rate': taker_fee_rate,
                'exit_order_type': 'TAKER_FALLBACK',
            }
        except Exception as e:
            logger.critical(
                f"[MAKER_EXIT] {pair}: Taker fallback CRASHED after market order likely executed on Binance: {e}. "
                f"Returning fallback result with current_price={current_price} to allow DB closure."
            )
            return {
                'price': current_price, 'fee_rate': taker_fee_rate,
                'exit_order_type': 'TAKER_FALLBACK_RECOVERED',
            }

    async def _simulate_maker_exit_paper(
        self, pair: str, direction: str, current_price: float
    ) -> Dict:
        """Simulate maker exit for paper trading using WebSocket prices."""
        tc = config.trading_config
        timeout = getattr(tc, 'maker_exit_timeout_seconds', 10)
        offset_ticks = getattr(tc, 'maker_exit_offset_ticks', 2)
        maker_fee_rate = getattr(tc, 'maker_fee', 0.00018)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)

        if current_price >= 10000:
            tick_size = 0.10
        elif current_price >= 100:
            tick_size = 0.01
        elif current_price >= 1:
            tick_size = 0.001
        else:
            tick_size = 0.0001

        if direction == 'LONG':
            limit_price = current_price + (offset_ticks * tick_size)
        else:
            limit_price = current_price - (offset_ticks * tick_size)

        limit_price = round(limit_price / tick_size) * tick_size

        logger.info(f"[MAKER_EXIT_PAPER] {pair}: Simulating limit exit {direction} @ {limit_price} "
                     f"(current={current_price}, offset={offset_ticks} ticks)")

        polls = max(1, timeout // 2)
        for i in range(polls):
            await asyncio.sleep(2)
            tracker = websocket_tracker.get_tracker(pair)
            if not tracker or not tracker.last_price:
                continue

            ws_price = tracker.last_price
            if direction == 'LONG' and ws_price >= limit_price:
                logger.info(f"[MAKER_EXIT_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price, 'fee_rate': maker_fee_rate,
                    'exit_order_type': 'MAKER',
                }
            elif direction == 'SHORT' and ws_price <= limit_price:
                logger.info(f"[MAKER_EXIT_PAPER] {pair}: Simulated FILL @ {limit_price} after {(i+1)*2}s "
                             f"(ws_price={ws_price})")
                return {
                    'price': limit_price, 'fee_rate': maker_fee_rate,
                    'exit_order_type': 'MAKER',
                }

        tracker = websocket_tracker.get_tracker(pair)
        fallback_price = tracker.last_price if tracker and tracker.last_price else current_price
        logger.info(f"[MAKER_EXIT_PAPER] {pair}: No fill after {timeout}s, taker fallback @ {fallback_price}")
        return {
            'price': fallback_price, 'fee_rate': taker_fee_rate,
            'exit_order_type': 'TAKER_FALLBACK',
        }

    async def open_position(
        self,
        db: AsyncSession,
        pair: str,
        direction: str,
        confidence: str,
        current_price: float,
        entry_gap: float = None,
        entry_ema_gap_5_8: float = None,
        entry_ema5_stretch: float = None,
        entry_rsi: float = None,
        entry_adx: float = None,
        entry_adx_prev: float = None,
        entry_macro_trend: str = None,
        entry_ema20_slope: float = None,
        entry_btc_ema20_slope: float = None,
        entry_btc_adx: float = None,
        entry_btc_adx_prev: float = None,
        entry_btc_rsi: float = None,
        entry_btc_rsi_prev: float = None,
        entry_price_vs_ema5_pct: float = None,
        entry_global_volume_ratio: float = None,
        entry_pair_volume_ratio: float = None
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
        
        # Check cooldown - don't re-enter same pair too quickly after any close
        cooldown_minutes = config.trading_config.investment.cooldown_after_loss_minutes
        if cooldown_minutes > 0:
            cooldown_threshold = datetime.utcnow() - timedelta(minutes=cooldown_minutes)
            result = await db.execute(
                select(Order).where(
                    and_(
                        Order.pair == pair,
                        Order.status == "CLOSED",
                        Order.is_paper == self.is_paper_mode,
                        Order.closed_at >= cooldown_threshold
                    )
                ).order_by(desc(Order.closed_at)).limit(1)
            )
            recent_close = result.scalar_one_or_none()
            if recent_close:
                time_since_close = (datetime.utcnow() - recent_close.closed_at).total_seconds() / 60
                logger.info(f"[COOLDOWN] {pair}: Recent exit {time_since_close:.1f} mins ago (pnl={recent_close.pnl:.2f}), waiting {cooldown_minutes} mins")
                return None
        
        # Calculate position size
        available = await self.get_available_balance(db)
        open_margin_result = await db.execute(
            select(func.coalesce(func.sum(Order.investment), 0)).where(
                and_(Order.status == "OPEN", Order.is_paper == self.is_paper_mode)
            )
        )
        total_portfolio = available + (open_margin_result.scalar() or 0)
        investment, leverage = self.calculate_position_size(available, confidence, total_portfolio=total_portfolio)
        logger.info(f"[TRADE] {pair}: {direction} {confidence} - Investment: ${investment:.2f}, Leverage: {leverage}x")
        
        if investment <= 0:
            return None
        
        # Calculate notional and quantity
        notional_value = investment * leverage
        quantity = notional_value / current_price
        
        # Determine fee rate and entry type
        tc = config.trading_config
        maker_enabled = tc.maker_entry_enabled
        maker_fee_rate = getattr(tc, 'maker_fee', tc.trading_fee)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)

        entry_order_type = "TAKER"
        entry_fee = notional_value * taker_fee_rate
        
        # Execute trade
        binance_order_id = None
        actual_price = current_price
        
        if not self.is_paper_mode:
            symbol = pair.replace('USDT', '/USDT:USDT')
            side = 'buy' if direction == 'LONG' else 'sell'

            if maker_enabled:
                # --- Maker entry flow ---
                result = await self._try_maker_entry(
                    symbol=symbol, side=side, amount=quantity,
                    leverage=int(leverage), direction=direction, pair=pair,
                    notional_value=notional_value,
                    maker_fee_rate=maker_fee_rate, taker_fee_rate=taker_fee_rate
                )
                if result:
                    binance_order_id = result['id']
                    actual_price = result['price']
                    entry_fee = result['entry_fee']
                    entry_order_type = result['entry_order_type']
                    quantity = result.get('amount', quantity)
                else:
                    logger.error(f"[MAKER_ENTRY] {pair}: Both maker and fallback failed")
                    return None
            else:
                result = await binance_service.create_market_order(
                    symbol=symbol, side=side, amount=quantity, leverage=int(leverage)
                )
                if result:
                    binance_order_id = result['id']
                    actual_price = result['price']
                    entry_fee = result.get('fee', entry_fee)
                    entry_order_type = "TAKER"
        else:
            # Paper trade -- simulate maker fill if enabled
            if maker_enabled:
                result = await self._simulate_maker_entry_paper(
                    pair=pair, direction=direction, current_price=current_price,
                    notional_value=notional_value,
                    maker_fee_rate=maker_fee_rate, taker_fee_rate=taker_fee_rate
                )
                actual_price = result['price']
                entry_fee = result['entry_fee']
                entry_order_type = result['entry_order_type']
                quantity = notional_value / actual_price
            else:
                entry_order_type = "TAKER"
        
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
            entry_ema_gap_5_8=entry_ema_gap_5_8,
            entry_ema5_stretch=entry_ema5_stretch,
            entry_rsi=entry_rsi,
            entry_adx=entry_adx,
            entry_adx_prev=entry_adx_prev,
            entry_macro_trend=entry_macro_trend,
            entry_ema20_slope=entry_ema20_slope,
            entry_btc_ema20_slope=entry_btc_ema20_slope,
            entry_btc_adx=entry_btc_adx,
            entry_btc_adx_prev=entry_btc_adx_prev,
            entry_btc_rsi=entry_btc_rsi,
            entry_btc_rsi_prev=entry_btc_rsi_prev,
            entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
            entry_global_volume_ratio=entry_global_volume_ratio,
            entry_pair_volume_ratio=entry_pair_volume_ratio,
            entry_fee=entry_fee,
            entry_order_type=entry_order_type,
            peak_pnl=0.0,
            trough_pnl=0.0,
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
            order_type="MAKER" if entry_order_type == "MAKER" else "TAKER",
            is_paper=self.is_paper_mode
        )
        db.add(transaction)
        
        await db.commit()
        await db.refresh(order)
        
        # Recalculate paper balance from DB (source of truth) and save
        if self.is_paper_mode:
            await self._recalculate_paper_balance(db)
            await self._deduct_fee_from_bnb(entry_fee, db)
            await self.save_state(db)
        
        # Force reset WebSocket tracking for new order (fresh start from entry price)
        # This ensures we track high/low from the actual entry, not from previous orders
        websocket_tracker.force_reset_tracking(pair, actual_price)
        await websocket_tracker.subscribe_pair(pair, actual_price)
        
        # Fetch current EMA5 data so the WebSocket tick loop can capture
        # peak EMA5 metrics immediately (before update_orders_cache runs).
        _pair_data_row = await db.execute(
            select(PairData.ema5, PairData.ema5_prev3).where(PairData.pair == pair)
        )
        _pair_data = _pair_data_row.first()
        _cached_ema5 = _pair_data.ema5 if _pair_data else None
        _cached_ema5_prev3 = _pair_data.ema5_prev3 if _pair_data else None

        # Immediately add to real-time cache so the WebSocket SL callback can
        # protect this order right away (without waiting for update_orders_cache).
        async with _cache_lock:
            order_cache_entry = {
                'id': order.id,
                'direction': direction,
                'entry_price': actual_price,
                'quantity': quantity,
                'entry_fee': entry_fee,
                'confidence': confidence,
                'stop_loss': conf_config.stop_loss,
                'current_tp_level': 1,
                'peak_pnl': 0.0,
                'trough_pnl': 0.0,
                'be_levels_enabled': getattr(conf_config, 'be_levels_enabled', True),
                'be_level1_trigger': conf_config.be_level1_trigger,
                'be_level1_offset': conf_config.be_level1_offset,
                'be_level2_trigger': conf_config.be_level2_trigger,
                'be_level2_offset': conf_config.be_level2_offset,
                'be_level3_trigger': conf_config.be_level3_trigger,
                'be_level3_offset': conf_config.be_level3_offset,
                'be_level4_trigger': conf_config.be_level4_trigger,
                'be_level4_offset': conf_config.be_level4_offset,
                'be_level5_trigger': conf_config.be_level5_trigger,
                'be_level5_offset': conf_config.be_level5_offset,
                'high_price': actual_price,
                'low_price': actual_price,
                'pullback_trigger': conf_config.pullback_trigger,
                'tp_trailing_enabled': conf_config.tp_trailing_enabled,
                'cached_ema5': _cached_ema5,
                'cached_ema5_prev3': _cached_ema5_prev3,
                'peak_ema5_dist_pct': None,
                'peak_ema5_slope_pct': None,
                'peak_reached_at': None,
                'trough_reached_at': None,
                'trough_ema5_dist_pct': None,
                'ema5_ever_negative': False,
                'signal_lost_flagged': False,
                'signal_lost_flag_pnl': None,
                'signal_lost_flagged_at': None,
                'tick_prices': [],
                'phantom_be_l1_triggered': False,
                'phantom_be_l1_triggered_at': None,
                'phantom_be_l1_would_exit_pnl': None,
                'phantom_be_l2_triggered': False,
                'phantom_be_l2_triggered_at': None,
                'phantom_be_l2_would_exit_pnl': None,
                'phantom_tick_a_triggered': False,
                'phantom_tick_a_triggered_at': None,
                'phantom_tick_a_pnl': None,
                'phantom_tick_b_triggered': False,
                'phantom_tick_b_triggered_at': None,
                'phantom_tick_b_pnl': None,
                'phantom_tick_c_triggered': False,
                'phantom_tick_c_triggered_at': None,
                'phantom_tick_c_pnl': None,
                'phantom_tick_d_triggered': False,
                'phantom_tick_d_triggered_at': None,
                'phantom_tick_d_pnl': None,
                'phantom_tick_e_triggered': False,
                'phantom_tick_e_triggered_at': None,
                'phantom_tick_e_pnl': None,
                'phantom_tick_f_triggered': False,
                'phantom_tick_f_triggered_at': None,
                'phantom_tick_f_pnl': None,
                'phantom_tick_g_triggered': False,
                'phantom_tick_g_triggered_at': None,
                'phantom_tick_g_pnl': None,
                'regime_neutral_hit': False,
                'regime_neutral_hit_at': None,
                'regime_neutral_pnl': None,
                'regime_comeback_at': None,
                'regime_comeback_pnl': None,
                'regime_opposite_at': None,
                'regime_opposite_pnl': None,
            }
            if pair not in _open_orders_cache:
                _open_orders_cache[pair] = []
            _open_orders_cache[pair].append(order_cache_entry)
        
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
        async with _close_lock:
            return await self._close_position_locked(db, order, current_price, reason)

    async def _close_position_locked(
        self,
        db: AsyncSession,
        order: Order,
        current_price: float,
        reason: str = "MANUAL"
    ) -> Optional[Order]:
        """Internal close logic, must be called under _close_lock."""
        if order.status != "OPEN":
            return None
        
        # Re-verify from DB to prevent race between polling loop and real-time monitor
        fresh_check = await db.execute(
            select(Order.status).where(Order.id == order.id)
        )
        db_status = fresh_check.scalar_one_or_none()
        if db_status != "OPEN":
            logger.warning(f"[CLOSE_RACE_PREVENTED] {order.pair}: Order {order.id} already {db_status}, skipping duplicate close (reason={reason})")
            return None
        
        # CRITICAL: Never close with invalid price - this would cause -100% P&L
        if current_price is None or current_price <= 0:
            logger.error(f"[CLOSE_BLOCKED] {order.pair}: Attempted to close with invalid price={current_price}, reason={reason}")
            return None
        
        # Attempt maker exit if enabled, otherwise use taker
        tc = config.trading_config
        maker_exit_enabled = getattr(tc, 'maker_exit_enabled', False)
        taker_fee_rate = getattr(tc, 'taker_fee', tc.trading_fee)
        exit_order_type = 'TAKER'
        actual_exit_price = current_price

        if not self.is_paper_mode:
            # --- Live mode: exit with bounded retry ---
            max_exit_retries = 3
            exit_result = None

            _urgent_exit = any(reason.startswith(p) for p in (
                "STOP_LOSS", "BREAKEVEN_SL", "FL_SIGNAL_LOST", "FL_REGIME_CHANGE", "FL_TICK_MOMENTUM",
            ))

            for attempt in range(1, max_exit_retries + 1):
                if maker_exit_enabled and reason != "MANUAL" and not _urgent_exit:
                    symbol = order.pair.replace('USDT', '/USDT:USDT')
                    exit_result = await self._try_maker_exit(
                        symbol=symbol, side=order.direction, amount=order.quantity,
                        pair=order.pair, direction=order.direction, current_price=current_price
                    )
                else:
                    symbol = order.pair.replace('USDT', '/USDT:USDT')
                    result = await binance_service.close_position(
                        symbol=symbol, side=order.direction, amount=order.quantity
                    )
                    if result:
                        exit_result = {
                            'price': current_price,
                            'fee_rate': taker_fee_rate,
                            'exit_order_type': 'TAKER',
                        }
                    else:
                        exit_result = None

                if exit_result is not None:
                    break

                if attempt < max_exit_retries:
                    logger.warning(
                        f"[EXIT_RETRY] {order.pair}: Attempt {attempt}/{max_exit_retries} failed, retrying in 2s..."
                    )
                    await asyncio.sleep(2)
                    _retry_tracker = websocket_tracker.get_tracker(order.pair)
                    fresh_price = _retry_tracker.last_price if _retry_tracker else None
                    if fresh_price and fresh_price > 0:
                        current_price = fresh_price

            if exit_result is None:
                try:
                    positions = await binance_service.get_open_positions()
                    if positions is None:
                        raise RuntimeError("Binance API error — cannot determine position state")
                    binance_pairs = {p['symbol'].replace('/USDT:USDT', 'USDT') for p in positions}
                    if order.pair not in binance_pairs:
                        actual_price = await self._fetch_actual_fill_price(order, current_price)
                        logger.warning(f"[EXTERNAL_CLOSE] {order.pair}: position gone from Binance — closing in DB @ {actual_price}")
                        order.status = "CLOSED"
                        order.close_reason = "EXTERNAL_CLOSE"
                        order.closed_at = datetime.utcnow()
                        order.exit_price = actual_price
                        taker_fee = getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                        notional_at_close = order.quantity * actual_price
                        exit_fee = notional_at_close * taker_fee
                        if order.direction == "LONG":
                            raw_pnl = (actual_price - order.entry_price) * order.quantity
                        else:
                            raw_pnl = (order.entry_price - actual_price) * order.quantity
                        order.pnl = round(raw_pnl - (order.entry_fee or 0) - exit_fee, 4)
                        _notional = order.entry_price * order.quantity if order.quantity else 1
                        order.pnl_percentage = round(((raw_pnl - (order.entry_fee or 0) - exit_fee) / _notional) * 100, 4)
                        order.exit_fee = round(exit_fee, 4)
                        order.total_fee = round((order.entry_fee or 0) + exit_fee, 4)
                        order.exit_order_type = "EXTERNAL"
                        tx = Transaction(
                            order_id=order.id, pair=order.pair,
                            action=f"CLOSE_{order.direction}", price=actual_price,
                            quantity=order.quantity, investment=order.investment,
                            leverage=order.leverage, notional_value=order.notional_value,
                            fee=order.exit_fee, order_type="EXTERNAL",
                            is_paper=False
                        )
                        db.add(tx)
                        _exit_retry_queue.pop(order.id, None)
                        async with _cache_lock:
                            _open_orders_cache[order.pair] = [
                                o for o in _open_orders_cache.get(order.pair, []) if o['id'] != order.id
                            ]
                        return order
                except Exception as e:
                    logger.error(f"[EXTERNAL_CLOSE] {order.pair}: reconcile check failed: {e}")
                _exit_retry_queue.setdefault(order.id, 0)
                logger.critical(
                    f"[EXIT_FAILED] {order.pair}: All {max_exit_retries} exit attempts failed — "
                    f"added to retry queue (attempt {_exit_retry_queue[order.id]}/{_EXIT_RETRY_MAX})"
                )
                return None

            actual_exit_price = exit_result['price']
            exit_fee_rate = exit_result['fee_rate']
            exit_order_type = exit_result['exit_order_type']
            notional_at_close = order.quantity * actual_exit_price
            exit_fee = notional_at_close * exit_fee_rate
        else:
            # --- Paper mode: no retry needed ---
            _urgent_exit_paper = any(reason.startswith(p) for p in (
                "STOP_LOSS", "BREAKEVEN_SL", "FL_SIGNAL_LOST", "FL_REGIME_CHANGE", "FL_TICK_MOMENTUM",
            ))
            if maker_exit_enabled and reason != "MANUAL" and not _urgent_exit_paper:
                exit_result = await self._simulate_maker_exit_paper(
                    pair=order.pair, direction=order.direction, current_price=current_price
                )
                actual_exit_price = exit_result['price']
                exit_fee_rate = exit_result['fee_rate']
                exit_order_type = exit_result['exit_order_type']
                notional_at_close = order.quantity * actual_exit_price
                exit_fee = notional_at_close * exit_fee_rate
            else:
                notional_at_close = order.quantity * current_price
                exit_fee = notional_at_close * taker_fee_rate

        total_fee = (order.entry_fee or 0) + exit_fee

        # Calculate P&L
        pnl_data = calculate_pnl(
            direction=order.direction,
            entry_price=order.entry_price,
            current_price=actual_exit_price,
            quantity=order.quantity,
            leverage=order.leverage,
            entry_fee=order.entry_fee or 0,
            exit_fee=exit_fee
        )

        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Essential close — commit to DB immediately so the
        # order is never left as a zombie if optional metadata fails.
        # ═══════════════════════════════════════════════════════════════
        order.status = "CLOSED"
        order.exit_price = actual_exit_price
        order.exit_fee = exit_fee
        order.total_fee = total_fee
        order.exit_order_type = exit_order_type
        order.pnl = pnl_data['pnl']
        order.pnl_percentage = pnl_data['pnl_percentage']
        order.closed_at = datetime.utcnow()
        order.close_reason = reason

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
            order_type="TAKER",
            is_paper=order.is_paper
        )
        db.add(transaction)

        await db.commit()
        await db.refresh(order)
        logger.info(f"[CLOSE_COMMITTED] {order.pair} {order.direction}: essential close saved (reason={reason}, pnl={pnl_data['pnl']:.4f})")

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Optional metadata — failures here must NEVER revert
        # the close above.  A second commit persists the extras.
        # ═══════════════════════════════════════════════════════════════
        try:
            # Persist phantom shadow data, peak EMA5 metrics, and signal-lost flag from real-time cache
            for cached in _open_orders_cache.get(order.pair, []):
                if cached['id'] == order.id:
                    if cached.get('signal_lost_flagged'):
                        order.signal_lost_flagged = True
                        order.signal_lost_flag_pnl = cached.get('signal_lost_flag_pnl')
                        order.signal_lost_flagged_at = cached.get('signal_lost_flagged_at')
                        if not reason.startswith("FL_"):
                            order.close_reason = f"FL_{reason}"
                    order.phantom_be_l1_triggered_at = cached.get('phantom_be_l1_triggered_at')
                    order.phantom_be_l1_would_exit_pnl = cached.get('phantom_be_l1_would_exit_pnl')
                    order.phantom_be_l2_triggered_at = cached.get('phantom_be_l2_triggered_at')
                    order.phantom_be_l2_would_exit_pnl = cached.get('phantom_be_l2_would_exit_pnl')
                    for _lbl in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                        setattr(order, f'phantom_tick_{_lbl}_triggered_at', cached.get(f'phantom_tick_{_lbl}_triggered_at'))
                        setattr(order, f'phantom_tick_{_lbl}_pnl', cached.get(f'phantom_tick_{_lbl}_pnl'))
                    if cached.get('peak_ema5_dist_pct') is not None:
                        order.peak_ema5_dist_pct = cached['peak_ema5_dist_pct']
                    if cached.get('peak_ema5_slope_pct') is not None:
                        order.peak_ema5_slope_pct = cached['peak_ema5_slope_pct']
                    if cached.get('peak_reached_at') is not None:
                        order.peak_reached_at = cached['peak_reached_at']
                    if cached.get('trough_reached_at') is not None:
                        order.trough_reached_at = cached['trough_reached_at']
                    if cached.get('trough_ema5_dist_pct') is not None:
                        order.trough_ema5_dist_pct = cached['trough_ema5_dist_pct']
                    order.regime_neutral_hit_at = cached.get('regime_neutral_hit_at')
                    order.regime_neutral_pnl = cached.get('regime_neutral_pnl')
                    order.regime_comeback_at = cached.get('regime_comeback_at')
                    order.regime_comeback_pnl = cached.get('regime_comeback_pnl')
                    order.regime_opposite_at = cached.get('regime_opposite_at')
                    order.regime_opposite_pnl = cached.get('regime_opposite_pnl')
                    order._ema5_ever_negative = cached.get('ema5_ever_negative', False)
                    break

            pd = None
            try:
                pair_data_result = await db.execute(
                    select(PairData).where(PairData.pair == order.pair)
                )
                pd = pair_data_result.scalar_one_or_none()
                if pd:
                    order.signal_active_at_close = is_signal_direction_active(
                        order.direction, pd.ema5, pd.ema8, pd.ema20, pd.price
                    )
                else:
                    order.signal_active_at_close = None
            except Exception:
                order.signal_active_at_close = None

            if pd and pd.ema5 and actual_exit_price:
                if order.direction == "LONG":
                    order.exit_price_vs_ema5_pct = round((actual_exit_price - pd.ema5) / actual_exit_price * 100, 4)
                else:
                    order.exit_price_vs_ema5_pct = round((pd.ema5 - actual_exit_price) / actual_exit_price * 100, 4)
                if pd.ema5_prev3 and pd.ema5:
                    order.exit_ema5_slope_pct = round((pd.ema5 - pd.ema5_prev3) / pd.ema5 * 100, 4)

            try:
                ohlcv_data = await binance_service.get_ohlcv(order.pair, '5m', 100)
                if ohlcv_data and len(ohlcv_data) >= 51 and order.opened_at:
                    import pandas as _pd_lib
                    from ta.trend import EMAIndicator as _EMA
                    df = _pd_lib.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['close'] = df['close'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['timestamp'] = _pd_lib.to_datetime(df['timestamp'], unit='ms')
                    ema5_series = _EMA(close=df['close'], window=5).ema_indicator()
                    entry_ts = order.opened_at
                    if entry_ts.tzinfo:
                        entry_ts = entry_ts.replace(tzinfo=None)
                    mask = df['timestamp'] >= _pd_lib.Timestamp(entry_ts)
                    crossed = False
                    for idx in df.index[mask]:
                        e5 = ema5_series.iloc[idx]
                        if _pd_lib.isna(e5):
                            continue
                        if order.direction == "LONG" and df.at[idx, 'low'] <= e5:
                            crossed = True
                            break
                        elif order.direction == "SHORT" and df.at[idx, 'high'] >= e5:
                            crossed = True
                            break
                    order.exit_ema5_crossed = crossed
            except Exception:
                pass

            ema5_ever_neg = getattr(order, '_ema5_ever_negative', False)
            if not ema5_ever_neg:
                order.ema5_went_negative = "NEVER"
            elif order.exit_price_vs_ema5_pct is not None and order.exit_price_vs_ema5_pct >= 0:
                order.ema5_went_negative = "RECOVERED"
            else:
                order.ema5_went_negative = "ENDED_NEG"

            await db.commit()
        except Exception as _meta_err:
            logger.warning(f"[CLOSE_METADATA] {order.pair}: Optional metadata failed (order already closed safely): {_meta_err}")
            try:
                await db.rollback()
            except Exception:
                pass
        
        # Recalculate paper balance from DB (source of truth) and save
        if self.is_paper_mode:
            await self._recalculate_paper_balance(db)
            await self._deduct_fee_from_bnb(exit_fee, db)
            await self.save_state(db)

        self._register_post_exit_tracking(order, reason)
        self._rsi3_history.pop(order.id, None)

        return order

    def _register_post_exit_tracking(self, order: Order, reason: str):
        """Register a BE or Signal Lost exit trade for post-exit price tracking (regret metric)."""
        tc = config.trading_config
        if not getattr(tc, 'post_exit_tracking_enabled', False):
            return
        _reason_base = reason[3:] if reason.startswith("FL_") else reason
        if not (_reason_base.startswith("BREAKEVEN_SL") or _reason_base.startswith("SIGNAL_LOST") or _reason_base.startswith("TICK_MOMENTUM_EXIT") or _reason_base.startswith("RSI_MOMENTUM_EXIT") or _reason_base.startswith("STOP_LOSS") or _reason_base.startswith("REGIME_CHANGE")):
            return
        minutes = getattr(tc, 'post_exit_tracking_minutes', 45)
        tracker = websocket_tracker.get_tracker(order.pair)
        initial_price = tracker.last_price if tracker else order.exit_price
        now = datetime.utcnow()
        # Carry tick buffer and phantom tick states from cache
        cached_tick_buf = []
        phantom_tick_states = {}
        for cached in _open_orders_cache.get(order.pair, []):
            if cached['id'] == order.id:
                cached_tick_buf = cached.get('tick_prices', [])
                for _lbl in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                    phantom_tick_states[f'phantom_tick_{_lbl}_triggered'] = cached.get(f'phantom_tick_{_lbl}_triggered', False)
                    phantom_tick_states[f'phantom_tick_{_lbl}_triggered_at'] = cached.get(f'phantom_tick_{_lbl}_triggered_at')
                    phantom_tick_states[f'phantom_tick_{_lbl}_pnl'] = cached.get(f'phantom_tick_{_lbl}_pnl')
                break

        _pe_notional = order.entry_price * order.quantity if order.quantity else 1
        _pe_fee_drag = (((order.entry_fee or 0) + _pe_notional * getattr(tc, 'taker_fee', tc.trading_fee)) / _pe_notional) * 100

        self._post_exit_tracking[order.id] = {
            "order_id": order.id,
            "pair": order.pair,
            "entry_price": order.entry_price,
            "direction": order.direction,
            "fee_drag_pct": _pe_fee_drag,
            "exit_time": now,
            "tracking_until": now + timedelta(minutes=minutes),
            "post_high": initial_price or order.exit_price,
            "post_low": initial_price or order.exit_price,
            "peak_at": now,
            "trough_at": now,
            "signal_lost_at": None,
            "pnl_at_signal_lost": None,
            "peak_before_signal_lost": 0.0,
            "rsi_exit_at": None,
            "rsi_exit_pnl": None,
            "rsi3_exit_at": None,
            "rsi3_exit_pnl": None,
            "rsi_history": [],
            "signal_regained_at": None,
            "pnl_at_signal_regained": None,
            "running_min_pnl": None,
            "floor_before_signal_regain": None,
            "close_reason": reason,
            "tick_prices": cached_tick_buf,
            **phantom_tick_states,
        }
        logger.info(f"[POST_EXIT] Registered {order.pair} order {order.id} ({reason}) for {minutes}min tracking")

    async def update_post_exit_tracking(self, db: AsyncSession):
        """Check prices for recently closed BE trades and update peak/trough/timing. Called from monitor loop.

        Uses isolated DB sessions for all queries and writes so that failures
        never corrupt the shared monitor-loop session / connection pool.
        """
        if not self._post_exit_tracking:
            return

        now = datetime.utcnow()
        completed = []

        for order_id in list(self._post_exit_tracking.keys()):
            info = self._post_exit_tracking[order_id]
            tracker = websocket_tracker.get_tracker(info["pair"])
            if not tracker or not tracker.last_price or tracker.last_price <= 0:
                continue

            price = tracker.last_price
            entry = info["entry_price"]
            direction = info["direction"]

            if price > info["post_high"]:
                info["post_high"] = price
                info["peak_at"] = now
            if price < info["post_low"]:
                info["post_low"] = price
                info["trough_at"] = now

            # Current P&L for tracking calculations (net of fees, consistent with pnl_percentage)
            if direction == "LONG":
                current_pnl = ((price - entry) / entry) * 100 - info["fee_drag_pct"]
            else:
                current_pnl = ((entry - price) / entry) * 100 - info["fee_drag_pct"]

            # Track running minimum P&L (from entry) for floor-before-recovery analysis
            if info["running_min_pnl"] is None or current_pnl < info["running_min_pnl"]:
                info["running_min_pnl"] = current_pnl

            # Read pair_data for signal-lost, signal-regained, and RSI momentum checks (isolated session)
            pair_data = None
            if info["signal_lost_at"] is None or info["signal_regained_at"] is None or info["rsi_exit_at"] is None:
                try:
                    async with AsyncSessionLocal() as pe_read_db:
                        pd_result = await pe_read_db.execute(
                            select(PairData).where(PairData.pair == info["pair"])
                        )
                        pair_data = pd_result.scalar_one_or_none()
                except Exception:
                    pass

            # Signal-lost detection
            if info["signal_lost_at"] is None and pair_data:
                if not is_signal_direction_active(
                    direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
                ):
                    info["signal_lost_at"] = now
                    info["pnl_at_signal_lost"] = current_pnl

            # Signal-regained detection (for SIGNAL_LOST exits: did the signal come back?)
            _cr = info.get("close_reason", "")
            if info["signal_regained_at"] is None and pair_data and ("SIGNAL_LOST" in _cr):
                if is_signal_direction_active(
                    direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
                ):
                    info["signal_regained_at"] = now
                    info["pnl_at_signal_regained"] = current_pnl
                    info["floor_before_signal_regain"] = info["running_min_pnl"]

            # RSI momentum exit simulation (2-drop and 3-drop)
            if pair_data and pair_data.rsi is not None:
                _rsi = pair_data.rsi
                _rsi1 = pair_data.rsi_prev1
                _rsi2 = pair_data.rsi_prev2

                # 2-drop check
                if info["rsi_exit_at"] is None and _rsi1 is not None and _rsi2 is not None:
                    rsi_triggered = False
                    if direction == "LONG" and _rsi < _rsi1 < _rsi2:
                        rsi_triggered = True
                    elif direction == "SHORT" and _rsi > _rsi1 > _rsi2:
                        rsi_triggered = True
                    if rsi_triggered:
                        info["rsi_exit_at"] = now
                        info["rsi_exit_pnl"] = current_pnl

                # 3-drop check: maintain RSI history buffer
                history = info["rsi_history"]
                if not history or history[-1] != _rsi:
                    history.append(_rsi)
                    if len(history) > 4:
                        history.pop(0)
                if info["rsi3_exit_at"] is None and len(history) >= 4:
                    if direction == "LONG" and history[-1] < history[-2] < history[-3] < history[-4]:
                        info["rsi3_exit_at"] = now
                        info["rsi3_exit_pnl"] = current_pnl
                    elif direction == "SHORT" and history[-1] > history[-2] > history[-3] > history[-4]:
                        info["rsi3_exit_at"] = now
                        info["rsi3_exit_pnl"] = current_pnl

            # Track reachable peak (best P&L while signal still active)
            if info["signal_lost_at"] is None:
                if current_pnl > info["peak_before_signal_lost"]:
                    info["peak_before_signal_lost"] = current_pnl

            # Post-exit phantom tick momentum checks
            tick_exit_min_profit = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_profit', 0.05)
            pe_tick_buf = info.get("tick_prices")
            if pe_tick_buf is not None:
                now_ts = time.time()
                pe_tick_buf.append((now_ts, price))
                pe_tick_buf[:] = [(t, p) for t, p in pe_tick_buf if t >= now_ts - 125]
                if current_pnl > tick_exit_min_profit:
                    for _lbl, _swin, _sdelta in _SHADOW_TICK_CONFIGS:
                        _tk = f'phantom_tick_{_lbl}_triggered'
                        if not info.get(_tk):
                            _sdeltas = _sdelta if isinstance(_sdelta, list) else [_sdelta] * len(_swin)
                            if _check_tick_momentum_fade(pe_tick_buf, now_ts, _swin, _sdeltas, direction):
                                info[_tk] = True
                                info[f'phantom_tick_{_lbl}_triggered_at'] = now
                                info[f'phantom_tick_{_lbl}_pnl'] = current_pnl

            if now >= info["tracking_until"]:
                _fd = info["fee_drag_pct"]
                if direction == "LONG":
                    peak_pnl = ((info["post_high"] - entry) / entry) * 100 - _fd
                    trough_pnl = ((info["post_low"] - entry) / entry) * 100 - _fd
                    final_pnl = ((price - entry) / entry) * 100 - _fd
                else:
                    peak_pnl = ((entry - info["post_low"]) / entry) * 100 - _fd
                    trough_pnl = ((entry - info["post_high"]) / entry) * 100 - _fd
                    final_pnl = ((entry - price) / entry) * 100 - _fd

                exit_time = info["exit_time"]
                peak_minutes = (info["peak_at"] - exit_time).total_seconds() / 60.0
                trough_minutes = (info["trough_at"] - exit_time).total_seconds() / 60.0
                sig_lost_minutes = None
                if info["signal_lost_at"]:
                    sig_lost_minutes = (info["signal_lost_at"] - exit_time).total_seconds() / 60.0
                rsi_exit_minutes = None
                if info["rsi_exit_at"]:
                    rsi_exit_minutes = (info["rsi_exit_at"] - exit_time).total_seconds() / 60.0
                rsi3_exit_minutes = None
                if info["rsi3_exit_at"]:
                    rsi3_exit_minutes = (info["rsi3_exit_at"] - exit_time).total_seconds() / 60.0
                sig_regained_minutes = None
                if info["signal_regained_at"]:
                    sig_regained_minutes = (info["signal_regained_at"] - exit_time).total_seconds() / 60.0

                try:
                    async with AsyncSessionLocal() as pe_write_db:
                        await pe_write_db.execute(
                            update(Order)
                            .where(Order.id == order_id)
                            .values(
                                post_exit_peak_pnl=round(peak_pnl, 4),
                                post_exit_trough_pnl=round(trough_pnl, 4),
                                post_exit_peak_minutes=round(peak_minutes, 2),
                                post_exit_trough_minutes=round(trough_minutes, 2),
                                post_exit_signal_lost_minutes=round(sig_lost_minutes, 2) if sig_lost_minutes is not None else None,
                                post_exit_pnl_at_signal_lost=round(info["pnl_at_signal_lost"], 4) if info["pnl_at_signal_lost"] is not None else None,
                                post_exit_final_pnl=round(final_pnl, 4),
                                post_exit_peak_before_signal_lost=round(info["peak_before_signal_lost"], 4) if info["signal_lost_at"] is not None else None,
                                post_exit_rsi_exit_minutes=round(rsi_exit_minutes, 2) if rsi_exit_minutes is not None else None,
                                post_exit_rsi_exit_pnl=round(info["rsi_exit_pnl"], 4) if info["rsi_exit_pnl"] is not None else None,
                                post_exit_rsi3_exit_minutes=round(rsi3_exit_minutes, 2) if rsi3_exit_minutes is not None else None,
                                post_exit_rsi3_exit_pnl=round(info["rsi3_exit_pnl"], 4) if info["rsi3_exit_pnl"] is not None else None,
                                post_exit_signal_regained_minutes=round(sig_regained_minutes, 2) if sig_regained_minutes is not None else None,
                                post_exit_pnl_at_signal_regained=round(info["pnl_at_signal_regained"], 4) if info["pnl_at_signal_regained"] is not None else None,
                                post_exit_floor_before_signal_regain=round(info["floor_before_signal_regain"], 4) if info["floor_before_signal_regain"] is not None else None,
                                phantom_tick_a_triggered_at=info.get("phantom_tick_a_triggered_at"),
                                phantom_tick_a_pnl=round(info["phantom_tick_a_pnl"], 4) if info.get("phantom_tick_a_pnl") is not None else None,
                                phantom_tick_b_triggered_at=info.get("phantom_tick_b_triggered_at"),
                                phantom_tick_b_pnl=round(info["phantom_tick_b_pnl"], 4) if info.get("phantom_tick_b_pnl") is not None else None,
                                phantom_tick_c_triggered_at=info.get("phantom_tick_c_triggered_at"),
                                phantom_tick_c_pnl=round(info["phantom_tick_c_pnl"], 4) if info.get("phantom_tick_c_pnl") is not None else None,
                                phantom_tick_d_triggered_at=info.get("phantom_tick_d_triggered_at"),
                                phantom_tick_d_pnl=round(info["phantom_tick_d_pnl"], 4) if info.get("phantom_tick_d_pnl") is not None else None,
                                phantom_tick_e_triggered_at=info.get("phantom_tick_e_triggered_at"),
                                phantom_tick_e_pnl=round(info["phantom_tick_e_pnl"], 4) if info.get("phantom_tick_e_pnl") is not None else None,
                                phantom_tick_f_triggered_at=info.get("phantom_tick_f_triggered_at"),
                                phantom_tick_f_pnl=round(info["phantom_tick_f_pnl"], 4) if info.get("phantom_tick_f_pnl") is not None else None,
                                phantom_tick_g_triggered_at=info.get("phantom_tick_g_triggered_at"),
                                phantom_tick_g_pnl=round(info["phantom_tick_g_pnl"], 4) if info.get("phantom_tick_g_pnl") is not None else None,
                            )
                        )
                        await pe_write_db.commit()
                    sig_info = f", sig_lost={sig_lost_minutes:.1f}min" if sig_lost_minutes is not None else ""
                    rsi_info = f", rsi_exit={rsi_exit_minutes:.1f}min@{info['rsi_exit_pnl']:.4f}%" if rsi_exit_minutes is not None else ""
                    rsi3_info = f", rsi3_exit={rsi3_exit_minutes:.1f}min@{info['rsi3_exit_pnl']:.4f}%" if rsi3_exit_minutes is not None else ""
                    logger.info(
                        f"[POST_EXIT] {info['pair']} order {order_id}: "
                        f"peak={peak_pnl:.4f}%@{peak_minutes:.1f}min trough={trough_pnl:.4f}%@{trough_minutes:.1f}min "
                        f"final={final_pnl:.4f}%{sig_info}{rsi_info}{rsi3_info}"
                    )
                except Exception as e:
                    logger.error(f"[POST_EXIT] Error saving order {order_id}: {e}")

                completed.append(order_id)

        for order_id in completed:
            del self._post_exit_tracking[order_id]

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
            # Use WebSocket price only -- no REST fallback to avoid rate-limit bans.
            # Open orders are always subscribed to WebSocket; if no price yet
            # (e.g. first seconds after startup), just skip and retry next cycle.
            tracker = websocket_tracker.get_tracker(order.pair)
            current_price = tracker.last_price if tracker else None

            if not current_price or current_price <= 0:
                continue
            
            order.current_price = current_price
            
            ws_high, ws_low = websocket_tracker.get_high_low(order.pair)
            
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
            
            # Merge realtime peak/trough from cache (may differ from DB if a
            # price spike occurred between polling cycles)
            realtime_peak = order.peak_pnl or 0
            realtime_trough = order.trough_pnl or 0
            realtime_peak_ema5_gap = order.peak_ema5_gap or 0
            async with _cache_lock:
                for cached in _open_orders_cache.get(order.pair, []):
                    if cached['id'] == order.id:
                        realtime_peak = max(realtime_peak, cached.get('peak_pnl', 0))
                        realtime_trough = min(realtime_trough, cached.get('trough_pnl', 0))
                        realtime_peak_ema5_gap = max(realtime_peak_ema5_gap, cached.get('peak_ema5_gap', 0))
                        break
            
            # Check NO_EXPANSION: close stale trades that never expanded
            no_exp_minutes = config.trading_config.investment.no_expansion_minutes
            if no_exp_minutes > 0 and order.opened_at:
                from datetime import timezone
                # Use last reset time if available, otherwise use opened_at
                ref_time = order.no_expansion_last_check or order.opened_at
                ref_time = ref_time.replace(tzinfo=timezone.utc) if ref_time.tzinfo is None else ref_time
                age_minutes = (datetime.now(timezone.utc) - ref_time).total_seconds() / 60
                if age_minutes >= no_exp_minutes:
                    conf_config = config.trading_config.confidence_levels.get(order.confidence)
                    if conf_config:
                        be_l1_trigger = conf_config.be_level1_trigger
                        be_l1_offset = conf_config.be_level1_offset
                        if order.direction == "LONG":
                            raw_pnl = (current_price - order.entry_price) * order.quantity
                        else:
                            raw_pnl = (order.entry_price - current_price) * order.quantity
                        est_exit_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                        net_pnl = raw_pnl - (order.entry_fee or 0) - est_exit_fee
                        entry_notional = order.entry_price * order.quantity if order.quantity > 0 else 1
                        cur_pnl_pct = (net_pnl / entry_notional) * 100
                        if realtime_peak < be_l1_trigger and cur_pnl_pct < be_l1_offset:
                            # Re-check if buy signal is still active before closing
                            if pair_data and pair_data.signal == order.direction:
                                order.no_expansion_last_check = datetime.now(timezone.utc)
                                logger.info(f"[NO_EXPANSION_RESET] {order.pair} {order.direction}: signal still {order.direction}, resetting timer (was {age_minutes:.0f}min)")
                                continue
                            logger.info(f"[NO_EXPANSION] {order.pair} {order.direction}: {age_minutes:.0f}min, peak={realtime_peak:.4f}% < BE_L1={be_l1_trigger}%, cur={cur_pnl_pct:.4f}% < BE_L1_off={be_l1_offset}%")
                            closed_order = await self.close_position(db, order, current_price, "NO_EXPANSION")
                            if closed_order:
                                updates.append({
                                    "order_id": closed_order.id,
                                    "pair": closed_order.pair,
                                    "action": "CLOSED",
                                    "reason": "NO_EXPANSION",
                                    "pnl": closed_order.pnl,
                                    "tp_level": order.current_tp_level or 1
                                })
                            continue

            # Compute current P&L % for exit checks
            if order.direction == "LONG":
                _raw_pnl = (current_price - order.entry_price) * order.quantity
            else:
                _raw_pnl = (order.entry_price - current_price) * order.quantity
            _est_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
            _net_pnl = _raw_pnl - (order.entry_fee or 0) - _est_fee
            _notional = order.entry_price * order.quantity if order.quantity > 0 else 1
            pnl_pct = (_net_pnl / _notional) * 100

            # In-trade RSI pattern tracking (first occurrence, no P&L threshold)
            if pair_data and pair_data.rsi is not None:
                _trk_rsi = pair_data.rsi
                _trk_rsi1 = pair_data.rsi_prev1
                _trk_rsi2 = pair_data.rsi_prev2
                from datetime import timezone as _tz
                _trk_opened = order.opened_at.replace(tzinfo=_tz.utc) if order.opened_at and order.opened_at.tzinfo is None else order.opened_at
                _trk_age = (datetime.now(_tz.utc) - _trk_opened).total_seconds() / 60 if _trk_opened else 0

                # 2-drop detection
                if order.first_rsi2_pnl is None and _trk_rsi1 is not None and _trk_rsi2 is not None:
                    rsi2_fired = False
                    if order.direction == "LONG" and _trk_rsi < _trk_rsi1 < _trk_rsi2:
                        rsi2_fired = True
                    elif order.direction == "SHORT" and _trk_rsi > _trk_rsi1 > _trk_rsi2:
                        rsi2_fired = True
                    if rsi2_fired:
                        order.first_rsi2_pnl = round(pnl_pct, 4)
                        order.first_rsi2_minutes = round(_trk_age, 2)

                # 3-drop detection via rolling history buffer
                oid = order.id
                if oid not in self._rsi3_history:
                    self._rsi3_history[oid] = []
                hist = self._rsi3_history[oid]
                if not hist or hist[-1] != _trk_rsi:
                    hist.append(_trk_rsi)
                    if len(hist) > 4:
                        hist.pop(0)
                if order.first_rsi3_pnl is None and len(hist) >= 4:
                    if order.direction == "LONG" and hist[-1] < hist[-2] < hist[-3] < hist[-4]:
                        order.first_rsi3_pnl = round(pnl_pct, 4)
                        order.first_rsi3_minutes = round(_trk_age, 2)
                    elif order.direction == "SHORT" and hist[-1] > hist[-2] > hist[-3] > hist[-4]:
                        order.first_rsi3_pnl = round(pnl_pct, 4)
                        order.first_rsi3_minutes = round(_trk_age, 2)

            # Regime Neutral tracking: record when regime goes NEUTRAL, comes back, or goes opposite
            _favorable_regime = "BULLISH" if order.direction == "LONG" else "BEARISH"
            _opposite_regime = "BEARISH" if order.direction == "LONG" else "BULLISH"
            if _current_btc_regime == "NEUTRAL" and not cached.get('regime_neutral_hit'):
                cached['regime_neutral_hit'] = True
                cached['regime_neutral_hit_at'] = datetime.utcnow()
                cached['regime_neutral_pnl'] = round(pnl_pct, 4)
                logger.info(f"[REGIME_NEUTRAL] {order.pair} {order.direction}: regime went NEUTRAL (pnl={pnl_pct:.4f}%)")
            elif cached.get('regime_neutral_hit'):
                if _current_btc_regime == _favorable_regime and not cached.get('regime_comeback_at'):
                    cached['regime_comeback_at'] = datetime.utcnow()
                    cached['regime_comeback_pnl'] = round(pnl_pct, 4)
                    logger.info(f"[REGIME_COMEBACK] {order.pair} {order.direction}: regime back to {_favorable_regime} (pnl={pnl_pct:.4f}%)")
                elif _current_btc_regime == _opposite_regime and not cached.get('regime_opposite_at'):
                    cached['regime_opposite_at'] = datetime.utcnow()
                    cached['regime_opposite_pnl'] = round(pnl_pct, 4)
                    logger.info(f"[REGIME_OPPOSITE] {order.pair} {order.direction}: regime went {_opposite_regime} (pnl={pnl_pct:.4f}%)")

            # REGIME_CHANGE: close when BTC macro regime flips against trade direction
            regime_exit_enabled = getattr(config.trading_config.thresholds, 'regime_change_exit_enabled', True)
            if regime_exit_enabled and _current_btc_regime != "NEUTRAL":
                regime_conflicts = (
                    (order.direction == "LONG" and _current_btc_regime == "BEARISH") or
                    (order.direction == "SHORT" and _current_btc_regime == "BULLISH")
                )
                if regime_conflicts:
                    tp_level = order.current_tp_level or 1
                    logger.info(f"[REGIME_CHANGE] {order.pair} {order.direction} L{tp_level}: BTC regime now {_current_btc_regime}, closing (pnl={pnl_pct:.4f}%)")
                    closed_order = await self.close_position(db, order, current_price, f"REGIME_CHANGE L{tp_level}")
                    if closed_order:
                        updates.append({
                            "order_id": closed_order.id,
                            "pair": closed_order.pair,
                            "action": "CLOSED",
                            "reason": f"REGIME_CHANGE L{tp_level}",
                            "pnl": closed_order.pnl,
                            "tp_level": tp_level
                        })
                    continue

            # RSI Momentum Exit: two consecutive RSI drops (LONG) or rises (SHORT) within P&L range
            rsi_exit_enabled = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_enabled', False)
            rsi_exit_min_profit = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_min_profit', 0.05)
            rsi_exit_max_profit = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_max_profit', 999.0)
            if rsi_exit_enabled and pair_data and pnl_pct > rsi_exit_min_profit and pnl_pct < rsi_exit_max_profit:
                _rsi = pair_data.rsi
                _rsi1 = pair_data.rsi_prev1
                _rsi2 = pair_data.rsi_prev2
                if _rsi is not None and _rsi1 is not None and _rsi2 is not None:
                    rsi_fading = False
                    if order.direction == "LONG" and _rsi < _rsi1 < _rsi2:
                        rsi_fading = True
                    elif order.direction == "SHORT" and _rsi > _rsi1 > _rsi2:
                        rsi_fading = True
                    if rsi_fading:
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[RSI_MOMENTUM_EXIT] {order.pair} {order.direction} L{tp_level}: RSI fading ({_rsi2:.1f}->{_rsi1:.1f}->{_rsi:.1f}), pnl={pnl_pct:.4f}% (range {rsi_exit_min_profit}% to {rsi_exit_max_profit}%)")
                        closed_order = await self.close_position(db, order, current_price, f"RSI_MOMENTUM_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"RSI_MOMENTUM_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # P&L trailing stop: only MOMENTUM_EXIT (signal lost). Skipped when signal active + RSI exit enabled.
            pnl_trigger = getattr(config.trading_config.thresholds, 'pnl_trailing_trigger', 0.0)
            pnl_ratio = getattr(config.trading_config.thresholds, 'pnl_trailing_ratio', 0.0)
            if pnl_trigger > 0 and pnl_ratio > 0 and realtime_peak >= pnl_trigger:
                signal_active = pair_data and is_signal_direction_active(
                    order.direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
                )
                if signal_active and rsi_exit_enabled:
                    pass  # RSI momentum exit handles signal-active exits
                else:
                    pnl_exit_level = realtime_peak * pnl_ratio
                    if pnl_pct <= pnl_exit_level:
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[MOMENTUM_EXIT] {order.pair} {order.direction} L{tp_level}: pnl={pnl_pct:.4f}% <= peak={realtime_peak:.4f}%*{pnl_ratio}(no-signal)={pnl_exit_level:.4f}%")
                        closed_order = await self.close_position(db, order, current_price, f"MOMENTUM_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"MOMENTUM_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # SLOPE_EXIT: EMA5 slope reversal
            ema5_slope_enabled = getattr(config.trading_config.thresholds, 'ema5_slope_exit_enabled', False)
            if ema5_slope_enabled and pair_data and pair_data.ema5 is not None:
                if pair_data.ema5_prev3 is not None and pair_data.ema5_prev3 != 0:
                    ema5_slope_pct = ((pair_data.ema5 - pair_data.ema5_prev3) / pair_data.ema5_prev3) * 100
                    slope_threshold = getattr(config.trading_config.thresholds, 'ema5_slope_threshold', 0.0)
                    if (order.direction == "LONG" and ema5_slope_pct <= slope_threshold) or \
                       (order.direction == "SHORT" and ema5_slope_pct >= -slope_threshold):
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[SLOPE_EXIT] {order.pair} {order.direction} L{tp_level}: slope={ema5_slope_pct:.4f}% (threshold={slope_threshold}%)")
                        closed_order = await self.close_position(db, order, current_price, f"SLOPE_EXIT L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"SLOPE_EXIT L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

            # SIGNAL_LOST: full signal no longer matches entry direction
            # Flag system: instead of exiting in primary range, flag the trade and let it run.
            # Security gap at [-0.9, -0.7] is the hard exit for flagged trades.
            signal_lost_enabled = getattr(config.trading_config.thresholds, 'signal_lost_exit_enabled', True)
            signal_dir_active = pair_data and is_signal_direction_active(
                order.direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
            )
            if signal_lost_enabled and pair_data and not signal_dir_active:
                signal_lost_min = getattr(config.trading_config.thresholds, 'signal_lost_min_profit', 0.03)
                signal_lost_max = getattr(config.trading_config.thresholds, 'signal_lost_max_profit', 999.0)
                if order.direction == "LONG":
                    sl_raw_pnl = (current_price - order.entry_price) * order.quantity
                else:
                    sl_raw_pnl = (order.entry_price - current_price) * order.quantity
                sl_exit_fee = current_price * order.quantity * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
                sl_net_pnl = sl_raw_pnl - (order.entry_fee or 0) - sl_exit_fee
                sl_notional = order.entry_price * order.quantity if order.quantity > 0 else 1
                sl_pnl_pct = (sl_net_pnl / sl_notional) * 100
                conf_config = config.trading_config.confidence_levels.get(order.confidence)
                sl_tp_target = order.dynamic_tp_target if order.dynamic_tp_target is not None else (conf_config.tp_min if conf_config else 0.2)

                _flag_enabled = getattr(config.trading_config.thresholds, 'signal_lost_flag_enabled', True)

                # Check if trade is already flagged (from cache)
                _is_flagged = False
                async with _cache_lock:
                    for _ci in _open_orders_cache.get(order.pair, []):
                        if _ci['id'] == order.id:
                            _is_flagged = _ci.get('signal_lost_flagged', False)
                            break

                if sl_pnl_pct >= signal_lost_min and sl_pnl_pct <= signal_lost_max and sl_pnl_pct < sl_tp_target:
                    if _flag_enabled and not _is_flagged:
                        # Flag system ON: flag the trade instead of exiting
                        tp_level = order.current_tp_level or 1
                        flag_time = datetime.utcnow()
                        async with _cache_lock:
                            for _ci in _open_orders_cache.get(order.pair, []):
                                if _ci['id'] == order.id:
                                    _ci['signal_lost_flagged'] = True
                                    _ci['signal_lost_flag_pnl'] = round(sl_pnl_pct, 4)
                                    _ci['signal_lost_flagged_at'] = flag_time
                                    break
                        order.signal_lost_flagged = True
                        order.signal_lost_flag_pnl = round(sl_pnl_pct, 4)
                        order.signal_lost_flagged_at = flag_time
                        await db.commit()
                        logger.info(f"[SIGNAL_LOST_FLAG] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% — FLAGGED (persisted to DB), signal='{pair_data.signal}'")
                        continue
                    elif not _flag_enabled:
                        # Flag system OFF: original behavior — exit immediately
                        tp_level = order.current_tp_level or 1
                        logger.info(f"[SIGNAL_LOST] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% >= min {signal_lost_min}%, signal now '{pair_data.signal}' != '{order.direction}'")
                        closed_order = await self.close_position(db, order, current_price, f"SIGNAL_LOST L{tp_level}")
                        if closed_order:
                            updates.append({
                                "order_id": closed_order.id,
                                "pair": closed_order.pair,
                                "action": "CLOSED",
                                "reason": f"SIGNAL_LOST L{tp_level}",
                                "pnl": closed_order.pnl,
                                "tp_level": tp_level
                            })
                        continue

                # Security gap: flagged trade with signal still lost
                security_gap_min = getattr(config.trading_config.thresholds, 'signal_lost_flag_security_min', -0.9)
                security_gap_max = getattr(config.trading_config.thresholds, 'signal_lost_flag_security_max', -0.7)
                if _is_flagged and sl_pnl_pct >= security_gap_min and sl_pnl_pct <= security_gap_max:
                    tp_level = order.current_tp_level or 1
                    logger.info(f"[FL_SIGNAL_LOST] {order.pair} {order.direction} L{tp_level}: pnl={sl_pnl_pct:.4f}% hit security gap [{security_gap_min}, {security_gap_max}]")
                    closed_order = await self.close_position(db, order, current_price, f"FL_SIGNAL_LOST L{tp_level}")
                    if closed_order:
                        updates.append({
                            "order_id": closed_order.id,
                            "pair": closed_order.pair,
                            "action": "CLOSED",
                            "reason": f"FL_SIGNAL_LOST L{tp_level}",
                            "pnl": closed_order.pnl,
                            "tp_level": tp_level
                        })
                    continue

            # Check exit conditions (including fees for accurate SL/TP)
            is_signal_active = (pair_data and is_signal_direction_active(
                order.direction, pair_data.ema5, pair_data.ema8, pair_data.ema20, pair_data.price
            )) if pair_data else False
            exit_conf_config = config.trading_config.confidence_levels.get(order.confidence)
            exit_result = check_exit_conditions(
                direction=order.direction,
                entry_price=order.entry_price,
                current_price=current_price,
                leverage=order.leverage,
                confidence=order.confidence,
                peak_pnl=realtime_peak,
                trough_pnl=realtime_trough,
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
                dynamic_tp_target=order.dynamic_tp_target,
                signal_active=is_signal_active,
                tp_trailing_enabled=exit_conf_config.tp_trailing_enabled if exit_conf_config else True
            )
            
            order.peak_pnl = exit_result.get("peak_pnl", order.peak_pnl)
            order.trough_pnl = exit_result.get("trough_pnl", order.trough_pnl)
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
                
                # Sync cache so real-time WebSocket exits use the correct level
                async with _cache_lock:
                    for cached_order in _open_orders_cache.get(order.pair, []):
                        if cached_order['id'] == order.id:
                            cached_order['current_tp_level'] = new_tp_level
                            break
                
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

        # --- Process exit retry queue (live mode only) ---
        if not self.is_paper_mode and _exit_retry_queue:
            retry_ids = list(_exit_retry_queue.keys())
            for order_id in retry_ids:
                attempt = _exit_retry_queue.get(order_id, 0) + 1
                _exit_retry_queue[order_id] = attempt

                if attempt > _EXIT_RETRY_MAX:
                    logger.critical(
                        f"[EXIT_RETRY_EXHAUSTED] Order {order_id}: Gave up after {_EXIT_RETRY_MAX} retries"
                    )
                    del _exit_retry_queue[order_id]
                    continue

                retry_result = await db.execute(
                    select(Order).where(Order.id == order_id)
                )
                retry_order = retry_result.scalar_one_or_none()
                if not retry_order or retry_order.status != "OPEN":
                    _exit_retry_queue.pop(order_id, None)
                    continue

                tracker = websocket_tracker.get_tracker(retry_order.pair)
                price = tracker.last_price if tracker else None
                if not price or price <= 0:
                    continue

                logger.info(
                    f"[EXIT_RETRY_QUEUE] {retry_order.pair}: Retry {attempt}/{_EXIT_RETRY_MAX}"
                )
                async with _close_lock:
                    closed = await self._close_position_locked(
                        db, retry_order, price, reason=retry_order.close_reason or "EXIT_RETRY"
                    )
                if closed:
                    _exit_retry_queue.pop(order_id, None)
                    logger.info(f"[EXIT_RETRY_QUEUE] {retry_order.pair}: Successfully closed on retry {attempt}")

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
        global _global_volume_ratio
        actions = []
        _scan_vol_sum = 0.0
        _scan_avg_vol_sum = 0.0
        
        # Get top pairs based on config limit
        pairs_limit = config.trading_config.trading_pairs_limit
        top_pairs = await binance_service.get_top_futures_pairs(pairs_limit)
        _blacklist_str = getattr(config.trading_config, 'pair_blacklist', '')
        _blacklist = set(p.strip() for p in _blacklist_str.split(',') if p.strip())
        if _blacklist:
            top_pairs = [p for p in top_pairs if p['pair'] not in _blacklist]
            logger.info(f"[SCAN] Blacklist active: excluded {len(_blacklist)} pairs ({', '.join(sorted(_blacklist))})")
        logger.info(f"[SCAN] Fetched {len(top_pairs)} pairs from Binance (limit={pairs_limit})")
        
        if not top_pairs:
            logger.warning("[SCAN] No pairs returned from Binance - skipping scan cycle")
            self._last_scan_time = time.time()
            return []
        
        # Subscribe all top pairs to WebSocket in a single batch (one reconnection)
        await websocket_tracker.subscribe_pairs_batch([p['pair'] for p in top_pairs])
        
        # BTC global regime filter: fetch BTC data once before processing all pairs
        btc_global_enabled = getattr(config.trading_config.thresholds, 'btc_global_filter_enabled', False)
        btc_ema20 = None
        btc_ema20_prev6 = None
        btc_regime = "NEUTRAL"
        btc_ema20_slope_pct = None
        btc_adx = None
        btc_adx_prev = None
        btc_rsi = None
        btc_rsi_prev = None
        if btc_global_enabled:
            btc_ohlcv = await binance_service.get_ohlcv('BTC/USDT:USDT', '5m', 100)
            if btc_ohlcv:
                btc_indicators = calculate_indicators(btc_ohlcv)
                if btc_indicators:
                    btc_ema20 = btc_indicators.get('ema20')
                    btc_ema20_prev6 = btc_indicators.get('ema20_prev6')
                    btc_adx = btc_indicators.get('adx')
                    btc_adx_prev = btc_indicators.get('adx_prev1')
                    btc_rsi = btc_indicators.get('rsi')
                    btc_rsi_prev = btc_indicators.get('rsi_prev1')
                    flat_th = config.trading_config.thresholds.macro_trend_flat_threshold
                    btc_regime = determine_macro_regime(btc_ema20, btc_ema20_prev6, flat_th)
                    if btc_ema20 and btc_ema20_prev6 and btc_ema20_prev6 != 0:
                        btc_ema20_slope_pct = round(((btc_ema20 - btc_ema20_prev6) / btc_ema20_prev6) * 100, 4)
            global _current_btc_regime
            _current_btc_regime = btc_regime
            logger.info(f"[SCAN] BTC Global Filter: regime={btc_regime} (ema20={btc_ema20}, prev6={btc_ema20_prev6}, adx={btc_adx})")
        
        for batch_start in range(0, len(top_pairs), OHLCV_BATCH_SIZE):
            batch = top_pairs[batch_start:batch_start + OHLCV_BATCH_SIZE]
            batch_num = batch_start // OHLCV_BATCH_SIZE + 1
            total_batches = (len(top_pairs) + OHLCV_BATCH_SIZE - 1) // OHLCV_BATCH_SIZE
            logger.info(f"[SCAN] Processing batch {batch_num}/{total_batches} ({len(batch)} pairs)")

            for pair_info in batch:
                pair = pair_info['pair']
                symbol = pair_info['symbol']
                volume_24h = pair_info['volume_24h']

                ohlcv = await binance_service.get_ohlcv(symbol, '5m', 100)
                if not ohlcv:
                    continue

                _pair_vol_bars = getattr(config.trading_config.thresholds, 'pair_volume_lookback_bars', 20)
                _global_vol_bars = getattr(config.trading_config.thresholds, 'global_volume_lookback_bars', 48)
                indicators = calculate_indicators(ohlcv, pair_volume_bars=_pair_vol_bars, global_volume_bars=_global_vol_bars)
                if not indicators:
                    continue

                rsi_val = indicators.get('rsi')
                adx_val = indicators.get('adx')
                if rsi_val is not None and (rsi_val >= 99.9 or rsi_val <= 0.1):
                    logger.debug(f"[SKIP] {pair}: Degenerate RSI={rsi_val:.1f} (no price variation)")
                    continue
                if adx_val is None:
                    logger.debug(f"[SKIP] {pair}: ADX is null (insufficient price data)")
                    continue

                _pair_vol = indicators.get('volume') or 0
                _pair_avg_vol = indicators.get('avg_volume') or 0
                _pair_volume_ratio = round(_pair_vol / _pair_avg_vol, 4) if _pair_avg_vol > 0 else 1.0
                _pair_avg_vol_global = indicators.get('avg_volume_global') or 0
                _scan_vol_sum += _pair_vol
                _scan_avg_vol_sum += _pair_avg_vol_global if _pair_avg_vol_global > 0 else _pair_avg_vol

                signal, confidence = get_signal(
                    ema5=indicators.get('ema5'),
                    ema8=indicators.get('ema8'),
                    ema13=indicators.get('ema13'),
                    ema20=indicators.get('ema20'),
                    rsi=indicators.get('rsi'),
                    adx=indicators.get('adx'),
                    volume=indicators.get('volume'),
                    avg_volume=indicators.get('avg_volume'),
                    price=indicators.get('price'),
                    ema20_prev6=indicators.get('ema20_prev6'),
                    ema50=indicators.get('ema50'),
                    ema50_prev12=indicators.get('ema50_prev12'),
                    rsi_prev3=indicators.get('rsi_prev3'),
                    ema5_prev1=indicators.get('ema5_prev1'),
                    ema8_prev1=indicators.get('ema8_prev1')
                )

                if signal in ["LONG", "SHORT"]:
                    logger.info(f"[SIGNAL-FOUND] {pair}: {signal} {confidence} - RSI={indicators.get('rsi'):.1f}, ADX={indicators.get('adx')}")

                if signal in ["LONG", "SHORT"] and btc_global_enabled:
                    flat_th = config.trading_config.thresholds.macro_trend_flat_threshold
                    pair_regime = determine_macro_regime(
                        indicators.get('ema20'), indicators.get('ema20_prev6'), flat_th
                    )
                    neutral_mode = getattr(config.trading_config.thresholds, 'macro_trend_neutral_mode', 'both')
                    btc_blocks = False
                    if btc_regime == "NEUTRAL" and neutral_mode != "both":
                        btc_blocks = True
                    elif btc_regime == "BULLISH" and signal != "LONG":
                        btc_blocks = True
                    elif btc_regime == "BEARISH" and signal != "SHORT":
                        btc_blocks = True

                    pair_blocks = (pair_regime != btc_regime)

                    _th = config.trading_config.thresholds
                    btc_rsi_blocks = False
                    if btc_rsi is not None:
                        if signal == "LONG":
                            _rsi_lo = getattr(_th, 'btc_rsi_min_long', 0)
                            _rsi_hi = getattr(_th, 'btc_rsi_max_long', 100)
                        else:
                            _rsi_lo = getattr(_th, 'btc_rsi_min_short', 0)
                            _rsi_hi = getattr(_th, 'btc_rsi_max_short', 100)
                        if (_rsi_lo > 0 and btc_rsi < _rsi_lo) or (_rsi_hi < 100 and btc_rsi > _rsi_hi):
                            btc_rsi_blocks = True

                    btc_adx_range_blocks = False
                    if btc_adx is not None:
                        if signal == "LONG":
                            _adx_lo = getattr(_th, 'btc_adx_min_long', 0)
                            _adx_hi = getattr(_th, 'btc_adx_max_long', 100)
                        else:
                            _adx_lo = getattr(_th, 'btc_adx_min_short', 0)
                            _adx_hi = getattr(_th, 'btc_adx_max_short', 100)
                        if (_adx_lo > 0 and btc_adx < _adx_lo) or (_adx_hi < 100 and btc_adx > _adx_hi):
                            btc_adx_range_blocks = True

                    btc_adx_dir_blocks = False
                    if btc_adx is not None and btc_adx_prev is not None:
                        _adx_dir_cfg = getattr(_th, f'btc_adx_dir_{signal.lower()}', 'both')
                        if _adx_dir_cfg == 'rising' and btc_adx <= btc_adx_prev:
                            btc_adx_dir_blocks = True
                        elif _adx_dir_cfg == 'falling' and btc_adx >= btc_adx_prev:
                            btc_adx_dir_blocks = True

                    pair_adx_dir_blocks = False
                    _pair_adx = indicators.get('adx')
                    _pair_adx_prev = indicators.get('adx_prev1')
                    if _pair_adx is not None and _pair_adx_prev is not None:
                        _pair_adx_dir_cfg = getattr(_th, f'adx_dir_{signal.lower()}', 'both')
                        if _pair_adx_dir_cfg == 'rising' and _pair_adx <= _pair_adx_prev:
                            pair_adx_dir_blocks = True
                        elif _pair_adx_dir_cfg == 'falling' and _pair_adx >= _pair_adx_prev:
                            pair_adx_dir_blocks = True

                    btc_cross_blocks = False
                    btc_cross_reason = ""
                    if btc_rsi is not None and btc_adx is not None:
                        _cf_key = 'btc_rsi_adx_filter_long' if signal == 'LONG' else 'btc_rsi_adx_filter_short'
                        _cf_str = getattr(_th, _cf_key, '')
                        if _cf_str and _cf_str.strip():
                            for _cf_rule in _cf_str.split(','):
                                _cf_rule = _cf_rule.strip()
                                if not _cf_rule or ':' not in _cf_rule:
                                    continue
                                try:
                                    _cf_rsi_part, _cf_min_adx = _cf_rule.split(':')
                                    _cf_rsi_min, _cf_rsi_max = map(float, _cf_rsi_part.split('-'))
                                    if _cf_rsi_min <= btc_rsi < _cf_rsi_max:
                                        if btc_adx < float(_cf_min_adx):
                                            btc_cross_blocks = True
                                            btc_cross_reason = f"BTC RSI {btc_rsi:.1f} in [{_cf_rsi_min}-{_cf_rsi_max}) requires ADX>={_cf_min_adx}, got {btc_adx:.1f}"
                                        break
                                except (ValueError, TypeError):
                                    continue

                    if btc_blocks or pair_blocks or btc_rsi_blocks or btc_adx_range_blocks or btc_adx_dir_blocks or pair_adx_dir_blocks or btc_cross_blocks:
                        if btc_cross_blocks:
                            reason = btc_cross_reason
                        elif pair_adx_dir_blocks:
                            _pd_label = "Rising" if _pair_adx > _pair_adx_prev else "Falling"
                            _pd_want = getattr(_th, f'adx_dir_{signal.lower()}', 'both')
                            reason = f"Pair ADX {_pd_label} ({_pair_adx:.1f} vs prev {_pair_adx_prev:.1f}), {signal} requires {_pd_want}"
                        elif btc_adx_dir_blocks:
                            _adx_dir_label = "Rising" if btc_adx > btc_adx_prev else "Falling"
                            _adx_dir_want = getattr(_th, f'btc_adx_dir_{signal.lower()}', 'both')
                            reason = f"BTC ADX {_adx_dir_label} ({btc_adx:.1f} vs prev {btc_adx_prev:.1f}), {signal} requires {_adx_dir_want}"
                        elif btc_rsi_blocks:
                            reason = f"BTC RSI {btc_rsi:.1f} out of {signal} range [{_rsi_lo}-{_rsi_hi}]"
                        elif btc_adx_range_blocks:
                            reason = f"BTC ADX {btc_adx:.1f} out of {signal} range [{_adx_lo}-{_adx_hi}]"
                        elif btc_blocks:
                            reason = f"BTC={btc_regime}"
                        else:
                            reason = f"pair={pair_regime} vs BTC={btc_regime}"
                        logger.info(f"[BTC-GATE] {pair}: {signal} blocked — {reason}")
                        signal = "NO_TRADE"

                if signal in ["LONG", "SHORT"]:
                    _th = config.trading_config.thresholds
                    global_vol_blocks = False
                    if getattr(_th, 'global_volume_filter_enabled', False):
                        _gv_thresh = getattr(_th, f'global_volume_threshold_{signal.lower()}', 1.05)
                        if _global_volume_ratio < _gv_thresh:
                            global_vol_blocks = True

                    pair_vol_blocks = False
                    if getattr(_th, 'pair_volume_filter_enabled', False):
                        _pv_thresh = getattr(_th, f'pair_volume_threshold_{signal.lower()}', 1.10)
                        if _pair_volume_ratio < _pv_thresh:
                            pair_vol_blocks = True

                    if global_vol_blocks or pair_vol_blocks:
                        if global_vol_blocks:
                            reason = f"Global Vol {_global_volume_ratio:.2f} < {_gv_thresh} for {signal}"
                        else:
                            reason = f"Pair Vol {_pair_volume_ratio:.2f} < {_pv_thresh} for {signal}"
                        logger.info(f"[VOL-GATE] {pair}: {signal} blocked — {reason}")
                        signal = "NO_TRADE"

                await self.update_pair_data(db, pair, indicators, signal, confidence, volume_24h, _pair_volume_ratio)

                if signal in ["LONG", "SHORT"] and confidence and confidence != "NO_TRADE":
                    logger.info(f"[SIGNAL] {pair}: {signal} with {confidence} confidence - Opening position...")
                    entry_gap = None
                    if indicators.get('ema5') and indicators.get('ema20') and indicators['price'] > 0:
                        entry_gap = round(abs((indicators['ema5'] - indicators['ema20']) / indicators['price'] * 100), 4)
                    entry_ema_gap_5_8 = None
                    if indicators.get('ema5') and indicators.get('ema8') and indicators['ema8'] > 0:
                        entry_ema_gap_5_8 = round(abs((indicators['ema5'] - indicators['ema8']) / indicators['ema8'] * 100), 4)
                    entry_ema5_stretch = None
                    entry_price_vs_ema5_pct = None
                    if indicators.get('ema5') and indicators['price'] > 0:
                        entry_ema5_stretch = round(abs(indicators['price'] - indicators['ema5']) / indicators['price'] * 100, 4)
                        entry_price_vs_ema5_pct = round((indicators['price'] - indicators['ema5']) / indicators['ema5'] * 100, 4)
                    entry_rsi = indicators.get('rsi')
                    entry_adx = indicators.get('adx')
                    entry_adx_prev = indicators.get('adx_prev1')
                    if btc_global_enabled:
                        entry_regime = btc_regime
                    else:
                        flat_th = config.trading_config.thresholds.macro_trend_flat_threshold
                        entry_regime = determine_macro_regime(
                            indicators.get('ema20'), indicators.get('ema20_prev6'), flat_th
                        )
                    pair_ema20_slope_pct = None
                    pair_ema20 = indicators.get('ema20')
                    pair_ema20_prev6 = indicators.get('ema20_prev6')
                    if pair_ema20 and pair_ema20_prev6 and pair_ema20_prev6 != 0:
                        pair_ema20_slope_pct = round(((pair_ema20 - pair_ema20_prev6) / pair_ema20_prev6) * 100, 4)
                    order = await self.open_position(
                        db=db,
                        pair=pair,
                        direction=signal,
                        confidence=confidence,
                        current_price=indicators['price'],
                        entry_gap=entry_gap,
                        entry_ema_gap_5_8=entry_ema_gap_5_8,
                        entry_ema5_stretch=entry_ema5_stretch,
                        entry_rsi=round(entry_rsi, 2) if entry_rsi is not None else None,
                        entry_adx=round(entry_adx, 1) if entry_adx is not None else None,
                        entry_adx_prev=round(entry_adx_prev, 1) if entry_adx_prev is not None else None,
                        entry_macro_trend=entry_regime,
                        entry_ema20_slope=pair_ema20_slope_pct,
                        entry_btc_ema20_slope=btc_ema20_slope_pct,
                        entry_btc_adx=round(btc_adx, 1) if btc_adx is not None else None,
                        entry_btc_adx_prev=round(btc_adx_prev, 1) if btc_adx_prev is not None else None,
                        entry_btc_rsi=round(btc_rsi, 1) if btc_rsi is not None else None,
                        entry_btc_rsi_prev=round(btc_rsi_prev, 1) if btc_rsi_prev is not None else None,
                        entry_price_vs_ema5_pct=entry_price_vs_ema5_pct,
                        entry_global_volume_ratio=round(_global_volume_ratio, 4),
                        entry_pair_volume_ratio=round(_pair_volume_ratio, 4)
                    )

                    if order:
                        actions.append({
                            "pair": pair,
                            "action": f"OPENED_{signal}",
                            "confidence": confidence,
                            "price": indicators['price']
                        })

            if batch_start + OHLCV_BATCH_SIZE < len(top_pairs):
                await asyncio.sleep(OHLCV_BATCH_DELAY)
            
        if _scan_avg_vol_sum > 0:
            _global_volume_ratio = round(_scan_vol_sum / _scan_avg_vol_sum, 4)
            logger.info(f"[GLOBAL_VOL] ratio={_global_volume_ratio:.4f} (sum_vol={_scan_vol_sum:.0f}, sum_avg={_scan_avg_vol_sum:.0f})")

        self._last_scan_time = time.time()
        elapsed = self._last_scan_time - now
        logger.info(f"[SCAN] Completed in {elapsed:.1f}s - {len(top_pairs)} pairs processed, {len(actions)} positions opened")
        return actions
    
    async def update_pair_data(
        self,
        db: AsyncSession,
        pair: str,
        indicators: Dict,
        signal: str,
        confidence: Optional[str],
        volume_24h: Optional[float] = None,
        volume_ratio: Optional[float] = None
    ):
        """Update pair data cache in database"""
        result = await db.execute(
            select(PairData).where(PairData.pair == pair)
        )
        pair_data = result.scalar_one_or_none()
        
        # Use provided 24h volume, or fall back to candle volume
        actual_volume_24h = volume_24h if volume_24h is not None else indicators.get('volume', 0)
        
        flat_th = config.trading_config.thresholds.macro_trend_flat_threshold
        regime = determine_macro_regime(
            indicators.get('ema20'), indicators.get('ema20_prev6'), flat_th
        )
        
        if pair_data:
            pair_data.price = indicators.get('price', 0)
            pair_data.ema5 = indicators.get('ema5')
            pair_data.ema5_prev3 = indicators.get('ema5_prev3')
            pair_data.ema8 = indicators.get('ema8')
            pair_data.ema13 = indicators.get('ema13')
            pair_data.ema20 = indicators.get('ema20')
            pair_data.rsi = indicators.get('rsi')
            pair_data.rsi_prev1 = indicators.get('rsi_prev1')
            pair_data.rsi_prev2 = indicators.get('rsi_prev2')
            pair_data.adx = indicators.get('adx')
            pair_data.volume_24h = actual_volume_24h
            pair_data.avg_volume = indicators.get('avg_volume')
            pair_data.signal = signal
            pair_data.confidence = confidence
            pair_data.macro_regime = regime
            pair_data.volume_ratio = volume_ratio
            pair_data.updated_at = datetime.utcnow()
        else:
            pair_data = PairData(
                pair=pair,
                price=indicators.get('price', 0),
                ema5=indicators.get('ema5'),
                ema5_prev3=indicators.get('ema5_prev3'),
                ema8=indicators.get('ema8'),
                ema13=indicators.get('ema13'),
                ema20=indicators.get('ema20'),
                rsi=indicators.get('rsi'),
                rsi_prev1=indicators.get('rsi_prev1'),
                rsi_prev2=indicators.get('rsi_prev2'),
                adx=indicators.get('adx'),
                volume_24h=actual_volume_24h,
                avg_volume=indicators.get('avg_volume'),
                signal=signal,
                confidence=confidence,
                macro_regime=regime,
                volume_ratio=volume_ratio
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
            cached_trough_pnl = order_info.get('trough_pnl', 0.0)
            be_l1_trigger = order_info.get('be_level1_trigger', 999)
            be_l1_offset = order_info.get('be_level1_offset', 0.0)
            be_l2_trigger = order_info.get('be_level2_trigger', 999)
            be_l2_offset = order_info.get('be_level2_offset', 0.0)
            be_l3_trigger = order_info.get('be_level3_trigger', 999)
            be_l3_offset = order_info.get('be_level3_offset', 0.0)
            be_l4_trigger = order_info.get('be_level4_trigger', 999)
            be_l4_offset = order_info.get('be_level4_offset', 0.0)
            be_l5_trigger = order_info.get('be_level5_trigger', 999)
            be_l5_offset = order_info.get('be_level5_offset', 0.0)
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
            exit_fee = current_notional * getattr(config.trading_config, 'taker_fee', config.trading_config.trading_fee)
            total_fees = entry_fee + exit_fee
            
            if direction == "LONG":
                pnl = (current_price - entry_price) * quantity - total_fees
            else:
                pnl = (entry_price - current_price) * quantity - total_fees
            
            pnl_pct = (pnl / entry_notional) * 100
            
            # Track peak P&L in real-time for break-even decisions
            current_peak = max(cached_peak_pnl, pnl_pct) if pnl_pct > 0 else cached_peak_pnl
            if pnl_pct > cached_peak_pnl and pnl_pct > 0:
                order_info['peak_reached_at'] = datetime.utcnow()
                _ema5_val = order_info.get('cached_ema5')
                if _ema5_val and _ema5_val > 0:
                    if direction == 'LONG':
                        order_info['peak_ema5_dist_pct'] = round((current_price - _ema5_val) / current_price * 100, 4)
                    else:
                        order_info['peak_ema5_dist_pct'] = round((_ema5_val - current_price) / current_price * 100, 4)
                    _ema5_prev3 = order_info.get('cached_ema5_prev3')
                    if _ema5_prev3 and _ema5_prev3 > 0:
                        order_info['peak_ema5_slope_pct'] = round((_ema5_val - _ema5_prev3) / _ema5_val * 100, 4)
            order_info['peak_pnl'] = current_peak
            
            current_trough = min(cached_trough_pnl, pnl_pct) if pnl_pct < 0 else cached_trough_pnl
            if pnl_pct < cached_trough_pnl and pnl_pct < 0:
                order_info['trough_reached_at'] = datetime.utcnow()
                _ema5_val_t = order_info.get('cached_ema5')
                if _ema5_val_t and _ema5_val_t > 0:
                    if direction == 'LONG':
                        order_info['trough_ema5_dist_pct'] = round((current_price - _ema5_val_t) / current_price * 100, 4)
                    else:
                        order_info['trough_ema5_dist_pct'] = round((_ema5_val_t - current_price) / current_price * 100, 4)
            order_info['trough_pnl'] = current_trough

            # Track if EMA5 distance ever went unfavorable
            if not order_info.get('ema5_ever_negative'):
                _ema5_neg = order_info.get('cached_ema5')
                if _ema5_neg and _ema5_neg > 0:
                    if direction == 'LONG' and current_price < _ema5_neg:
                        order_info['ema5_ever_negative'] = True
                    elif direction == 'SHORT' and current_price > _ema5_neg:
                        order_info['ema5_ever_negative'] = True
            
            # Shadow BE tracking: record phantom triggers using original L1/L2 values
            _SHADOW_BE = [(1, 0.50, 0.20), (2, 1.00, 0.50)]
            for _sl, _strig, _soff in _SHADOW_BE:
                _tk = f'phantom_be_l{_sl}_triggered'
                _ek = f'phantom_be_l{_sl}_would_exit_pnl'
                _ak = f'phantom_be_l{_sl}_triggered_at'
                if not order_info.get(_tk) and current_peak >= _strig:
                    order_info[_tk] = True
                    order_info[_ak] = datetime.utcnow()
                if order_info.get(_tk) and order_info.get(_ek) is None and pnl_pct <= _soff:
                    order_info[_ek] = pnl_pct

            # Get TP target to determine if trailing stop would be active
            tp_level = order_info.get('current_tp_level', 1)
            conf = config.trading_config.confidence_levels.get(
                order_info.get('confidence', 'LOW'))
            tp_min = conf.tp_min if conf else 0.1
            effective_tp_target = tp_level * tp_min if tp_level > 1 else tp_min
            
            # Trailing stop activates once peak reaches TP target or at L2+.
            trailing_stop_would_be_active = current_peak >= effective_tp_target or tp_level >= 2
            
            # Apply 3-level break-even logic (highest level wins)
            effective_sl = stop_loss_pct
            signal_still_active = order_info.get('signal_active', False)
            breakeven_active = False
            be_level = 0
            be_enabled = order_info.get('be_levels_enabled', True)

            if be_enabled and current_peak >= be_l5_trigger:
                breakeven_active = True
                be_level = 5
                effective_sl = be_l5_offset
            elif be_enabled and current_peak >= be_l4_trigger:
                breakeven_active = True
                be_level = 4
                effective_sl = be_l4_offset
            elif be_enabled and current_peak >= be_l3_trigger:
                breakeven_active = True
                be_level = 3
                effective_sl = be_l3_offset
            elif be_enabled and current_peak >= be_l2_trigger:
                breakeven_active = True
                be_level = 2
                effective_sl = be_l2_offset
            elif be_enabled and current_peak >= be_l1_trigger:
                breakeven_active = True
                be_level = 1
                effective_sl = be_l1_offset
            elif signal_still_active:
                effective_sl = order_info.get('signal_active_sl', stop_loss_pct)

            # Check if stop loss triggered
            if pnl_pct <= effective_sl:
                if breakeven_active:
                    close_reason = f"BREAKEVEN_SL_L{be_level}"
                elif signal_still_active:
                    close_reason = f"STOP_LOSS_WIDE L{tp_level}"
                else:
                    close_reason = f"STOP_LOSS L{tp_level}"
                
                logger.warning(f"[REALTIME_{close_reason}] {pair} {direction}: pnl={pnl_pct:.4f}% <= effective_sl={effective_sl}% (original_sl={stop_loss_pct}%, peak={current_peak:.4f}%) - CLOSING NOW!")
                
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
                                close_reason
                            )
                            logger.info(f"[REALTIME_{close_reason}] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                            
                            # Remove from cache
                            async with _cache_lock:
                                _open_orders_cache[pair] = [
                                    o for o in _open_orders_cache.get(pair, []) 
                                    if o['id'] != order_id
                                ]
                except Exception as e:
                    logger.error(f"[REALTIME_SL] Error closing {pair}: {e}")
                continue  # Already handled, skip trailing stop check
            
            # Real-time RSI Momentum Exit: two consecutive RSI drops/rises within P&L range
            rt_rsi_exit_enabled = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_enabled', False)
            rt_rsi_exit_min = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_min_profit', 0.05)
            rt_rsi_exit_max = getattr(config.trading_config.thresholds, 'rsi_momentum_exit_max_profit', 999.0)
            if rt_rsi_exit_enabled and pnl_pct > rt_rsi_exit_min and pnl_pct < rt_rsi_exit_max:
                _rt_rsi = order_info.get('rsi')
                _rt_rsi1 = order_info.get('rsi_prev1')
                _rt_rsi2 = order_info.get('rsi_prev2')
                if _rt_rsi is not None and _rt_rsi1 is not None and _rt_rsi2 is not None:
                    rt_rsi_fading = False
                    if direction == "LONG" and _rt_rsi < _rt_rsi1 < _rt_rsi2:
                        rt_rsi_fading = True
                    elif direction == "SHORT" and _rt_rsi > _rt_rsi1 > _rt_rsi2:
                        rt_rsi_fading = True
                    if rt_rsi_fading:
                        tp_level = order_info.get('current_tp_level', 1)
                        logger.warning(f"[REALTIME_RSI_MOMENTUM_EXIT] {pair} {direction} L{tp_level}: RSI fading ({_rt_rsi2:.1f}->{_rt_rsi1:.1f}->{_rt_rsi:.1f}), pnl={pnl_pct:.4f}% (range {rt_rsi_exit_min}% to {rt_rsi_exit_max}%) - CLOSING NOW!")
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
                                        f"RSI_MOMENTUM_EXIT L{tp_level}"
                                    )
                                    logger.info(f"[REALTIME_RSI_MOMENTUM_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [
                                            o for o in _open_orders_cache.get(pair, [])
                                            if o['id'] != order_id
                                        ]
                        except Exception as e:
                            logger.error(f"[REALTIME_RSI_MOMENTUM_EXIT] Error closing {pair}: {e}")
                        continue

            # Real-time Tick Momentum Exit: multi-window price velocity check
            tick_exit_enabled = getattr(config.trading_config.thresholds, 'tick_momentum_exit_enabled', False)
            _is_trade_flagged = order_info.get('signal_lost_flagged', False)
            tick_exit_min_profit = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_profit', 0.05)
            if _is_trade_flagged:
                tick_exit_min_profit = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_profit_flagged', -0.10)
            now = time.time()
            tick_buf = order_info.get('tick_prices', [])
            tick_buf.append((now, current_price))
            cutoff = now - 125
            tick_buf[:] = [(t, p) for t, p in tick_buf if t >= cutoff]
            order_info['tick_prices'] = tick_buf

            if tick_exit_enabled and pnl_pct > tick_exit_min_profit:
                tick_min_delta_fallback = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_delta', 0.05)
                deltas_str = getattr(config.trading_config.thresholds, 'tick_momentum_exit_min_deltas', '')
                windows_str = getattr(config.trading_config.thresholds, 'tick_momentum_exit_windows', '15,30,60')
                try:
                    windows = [int(w.strip()) for w in windows_str.split(',') if w.strip()]
                except (ValueError, AttributeError):
                    windows = [15, 30, 60]

                per_window_deltas = None
                if deltas_str and deltas_str.strip():
                    try:
                        parsed = [float(d.strip()) for d in deltas_str.split(',') if d.strip()]
                        if len(parsed) == len(windows):
                            per_window_deltas = parsed
                    except (ValueError, AttributeError):
                        pass
                if per_window_deltas is None:
                    per_window_deltas = [tick_min_delta_fallback] * len(windows)

                all_windows_confirm = _check_tick_momentum_fade(tick_buf, now, windows, per_window_deltas, direction)

                # Shadow tick momentum: check phantom configs
                for _lbl, _swin, _sdelta in _SHADOW_TICK_CONFIGS:
                    _tk = f'phantom_tick_{_lbl}_triggered'
                    if not order_info.get(_tk):
                        _sdeltas = _sdelta if isinstance(_sdelta, list) else [_sdelta] * len(_swin)
                        if _check_tick_momentum_fade(tick_buf, now, _swin, _sdeltas, direction):
                            order_info[_tk] = True
                            order_info[f'phantom_tick_{_lbl}_triggered_at'] = datetime.utcnow()
                            order_info[f'phantom_tick_{_lbl}_pnl'] = pnl_pct

                if all_windows_confirm:
                    tp_level = order_info.get('current_tp_level', 1)
                    deltas_info = '/'.join(f"{d:.2f}" for d in per_window_deltas)
                    logger.warning(f"[REALTIME_TICK_MOMENTUM_EXIT] {pair} {direction} L{tp_level}: tick momentum fading across {windows}s windows (deltas={deltas_info}%), pnl={pnl_pct:.4f}% > min={tick_exit_min_profit}% - CLOSING NOW!")
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
                                    f"TICK_MOMENTUM_EXIT L{tp_level}"
                                )
                                logger.info(f"[REALTIME_TICK_MOMENTUM_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%")
                                async with _cache_lock:
                                    _open_orders_cache[pair] = [
                                        o for o in _open_orders_cache.get(pair, [])
                                        if o['id'] != order_id
                                    ]
                    except Exception as e:
                        logger.error(f"[REALTIME_TICK_MOMENTUM_EXIT] Error closing {pair}: {e}")
                    continue

            # Real-time P&L trailing: only MOMENTUM_EXIT (signal lost). Skipped when signal active + RSI exit enabled.
            pnl_trigger = getattr(config.trading_config.thresholds, 'pnl_trailing_trigger', 0.0)
            pnl_ratio = getattr(config.trading_config.thresholds, 'pnl_trailing_ratio', 0.0)
            if pnl_trigger > 0 and pnl_ratio > 0 and cached_peak_pnl >= pnl_trigger:
                rt_signal_active = order_info.get('signal_active', False)
                if rt_signal_active and rt_rsi_exit_enabled:
                    pass  # RSI momentum exit handles signal-active exits
                else:
                    pnl_exit_level = cached_peak_pnl * pnl_ratio
                    if pnl_pct <= pnl_exit_level:
                        tp_level = order_info.get('current_tp_level', 1)
                        logger.warning(f"[REALTIME_MOMENTUM_EXIT] {pair} {direction} L{tp_level}: pnl={pnl_pct:.4f}% <= peak={cached_peak_pnl:.4f}%*{pnl_ratio}(no-signal)={pnl_exit_level:.4f}%, price={current_price:.6f} - CLOSING NOW!")
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
                                        f"MOMENTUM_EXIT L{tp_level}"
                                    )
                                    logger.info(f"[REALTIME_MOMENTUM_EXIT] {pair} closed at {current_price} with pnl={pnl_pct:.4f}%, peak was {cached_peak_pnl:.4f}%")
                                    async with _cache_lock:
                                        _open_orders_cache[pair] = [
                                            o for o in _open_orders_cache.get(pair, [])
                                            if o['id'] != order_id
                                        ]
                        except Exception as e:
                            logger.error(f"[REALTIME_MOMENTUM_EXIT] Error closing {pair}: {e}")
                        continue
            
            # Real-time trailing stop check (only when trailing stop is active and TP/trailing enabled)
            if trailing_stop_would_be_active and order_info.get('tp_trailing_enabled', False):
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
        
        # Fetch current EMA values for each pair with open orders
        pair_names = list({o.pair for o in orders})
        pair_emas: Dict[str, Dict] = {}
        pair_ema5s: Dict[str, float] = {}
        if pair_names:
            sig_result = await db.execute(
                select(PairData.pair, PairData.ema5, PairData.ema8, PairData.ema20, PairData.price,
                       PairData.rsi, PairData.rsi_prev1, PairData.rsi_prev2,
                       PairData.ema5_prev3).where(PairData.pair.in_(pair_names))
            )
            for row in sig_result:
                pair_emas[row.pair] = {
                    'ema5': row.ema5, 'ema8': row.ema8,
                    'ema20': row.ema20, 'price': row.price,
                    'rsi': row.rsi, 'rsi_prev1': row.rsi_prev1, 'rsi_prev2': row.rsi_prev2,
                    'ema5_prev3': row.ema5_prev3,
                }
                if row.ema5 is not None:
                    pair_ema5s[row.pair] = row.ema5
        
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
                'signal_active_sl': conf_config.signal_active_sl,
                'signal_active': is_signal_direction_active(
                    order.direction,
                    pair_emas.get(order.pair, {}).get('ema5'),
                    pair_emas.get(order.pair, {}).get('ema8'),
                    pair_emas.get(order.pair, {}).get('ema20'),
                    pair_emas.get(order.pair, {}).get('price')
                ),
                'current_tp_level': order.current_tp_level,
                'peak_pnl': order.peak_pnl or 0.0,
                'trough_pnl': order.trough_pnl or 0.0,
                'be_levels_enabled': getattr(conf_config, 'be_levels_enabled', True),
                'be_level1_trigger': conf_config.be_level1_trigger,
                'be_level1_offset': conf_config.be_level1_offset,
                'be_level2_trigger': conf_config.be_level2_trigger,
                'be_level2_offset': conf_config.be_level2_offset,
                'be_level3_trigger': conf_config.be_level3_trigger,
                'be_level3_offset': conf_config.be_level3_offset,
                'be_level4_trigger': conf_config.be_level4_trigger,
                'be_level4_offset': conf_config.be_level4_offset,
                'be_level5_trigger': conf_config.be_level5_trigger,
                'be_level5_offset': conf_config.be_level5_offset,
                'high_price': order.high_price_since_entry or order.entry_price,
                'low_price': order.low_price_since_entry or order.entry_price,
                'pullback_trigger': conf_config.pullback_trigger,
                'tp_trailing_enabled': conf_config.tp_trailing_enabled,
                'cached_ema5': pair_ema5s.get(order.pair),
                'cached_ema5_prev3': pair_emas.get(order.pair, {}).get('ema5_prev3'),
                'peak_ema5_gap': order.peak_ema5_gap or 0.0,
                'peak_ema5_dist_pct': order.peak_ema5_dist_pct,
                'peak_ema5_slope_pct': order.peak_ema5_slope_pct,
                'peak_reached_at': order.peak_reached_at,
                'trough_reached_at': order.trough_reached_at,
                'trough_ema5_dist_pct': order.trough_ema5_dist_pct,
                'ema5_ever_negative': order.ema5_went_negative in ("RECOVERED", "ENDED_NEG") if order.ema5_went_negative else False,
                'signal_lost_flagged': bool(order.signal_lost_flagged) if order.signal_lost_flagged else False,
                'signal_lost_flag_pnl': order.signal_lost_flag_pnl,
                'signal_lost_flagged_at': order.signal_lost_flagged_at,
                'rsi': pair_emas.get(order.pair, {}).get('rsi'),
                'rsi_prev1': pair_emas.get(order.pair, {}).get('rsi_prev1'),
                'rsi_prev2': pair_emas.get(order.pair, {}).get('rsi_prev2'),
                'tick_prices': [],
                'phantom_be_l1_triggered': order.phantom_be_l1_triggered_at is not None,
                'phantom_be_l1_triggered_at': order.phantom_be_l1_triggered_at,
                'phantom_be_l1_would_exit_pnl': order.phantom_be_l1_would_exit_pnl,
                'phantom_be_l2_triggered': order.phantom_be_l2_triggered_at is not None,
                'phantom_be_l2_triggered_at': order.phantom_be_l2_triggered_at,
                'phantom_be_l2_would_exit_pnl': order.phantom_be_l2_would_exit_pnl,
                'phantom_tick_a_triggered': order.phantom_tick_a_triggered_at is not None,
                'phantom_tick_a_triggered_at': order.phantom_tick_a_triggered_at,
                'phantom_tick_a_pnl': order.phantom_tick_a_pnl,
                'phantom_tick_b_triggered': order.phantom_tick_b_triggered_at is not None,
                'phantom_tick_b_triggered_at': order.phantom_tick_b_triggered_at,
                'phantom_tick_b_pnl': order.phantom_tick_b_pnl,
                'phantom_tick_c_triggered': order.phantom_tick_c_triggered_at is not None,
                'phantom_tick_c_triggered_at': order.phantom_tick_c_triggered_at,
                'phantom_tick_c_pnl': order.phantom_tick_c_pnl,
                'phantom_tick_d_triggered': order.phantom_tick_d_triggered_at is not None,
                'phantom_tick_d_triggered_at': order.phantom_tick_d_triggered_at,
                'phantom_tick_d_pnl': order.phantom_tick_d_pnl,
                'phantom_tick_e_triggered': order.phantom_tick_e_triggered_at is not None,
                'phantom_tick_e_triggered_at': order.phantom_tick_e_triggered_at,
                'phantom_tick_e_pnl': order.phantom_tick_e_pnl,
                'phantom_tick_f_triggered': order.phantom_tick_f_triggered_at is not None,
                'phantom_tick_f_triggered_at': order.phantom_tick_f_triggered_at,
                'phantom_tick_f_pnl': order.phantom_tick_f_pnl,
                'phantom_tick_g_triggered': order.phantom_tick_g_triggered_at is not None,
                'phantom_tick_g_triggered_at': order.phantom_tick_g_triggered_at,
                'phantom_tick_g_pnl': order.phantom_tick_g_pnl,
                'regime_neutral_hit': order.regime_neutral_hit_at is not None,
                'regime_neutral_hit_at': order.regime_neutral_hit_at,
                'regime_neutral_pnl': order.regime_neutral_pnl,
                'regime_comeback_at': order.regime_comeback_at,
                'regime_comeback_pnl': order.regime_comeback_pnl,
                'regime_opposite_at': order.regime_opposite_at,
                'regime_opposite_pnl': order.regime_opposite_pnl,
            }
            
            if order.pair not in new_cache:
                new_cache[order.pair] = []
            new_cache[order.pair].append(order_info)
        
        async with _cache_lock:
            # Preserve realtime-tracked peaks that the DB may not have yet.
            # The realtime callback updates peak_pnl/high_price/low_price in
            # the cache between polling cycles; a naive overwrite would lose them.
            for pair, new_orders in new_cache.items():
                old_orders = _open_orders_cache.get(pair, [])
                for new_info in new_orders:
                    for old_info in old_orders:
                        if old_info['id'] == new_info['id']:
                            new_info['peak_pnl'] = max(new_info['peak_pnl'], old_info.get('peak_pnl', 0))
                            new_info['trough_pnl'] = min(new_info['trough_pnl'], old_info.get('trough_pnl', 0))
                            new_info['peak_ema5_gap'] = max(new_info['peak_ema5_gap'], old_info.get('peak_ema5_gap', 0))
                            if old_info.get('peak_pnl', 0) >= new_info.get('peak_pnl', 0):
                                new_info['peak_ema5_dist_pct'] = old_info.get('peak_ema5_dist_pct')
                                new_info['peak_ema5_slope_pct'] = old_info.get('peak_ema5_slope_pct')
                                new_info['peak_reached_at'] = old_info.get('peak_reached_at')
                            if old_info.get('trough_pnl', 0) <= new_info.get('trough_pnl', 0):
                                new_info['trough_reached_at'] = old_info.get('trough_reached_at')
                                new_info['trough_ema5_dist_pct'] = old_info.get('trough_ema5_dist_pct')
                            if old_info.get('ema5_ever_negative'):
                                new_info['ema5_ever_negative'] = True
                            if old_info.get('signal_lost_flagged'):
                                new_info['signal_lost_flagged'] = True
                                new_info['signal_lost_flag_pnl'] = old_info.get('signal_lost_flag_pnl')
                                new_info['signal_lost_flagged_at'] = old_info.get('signal_lost_flagged_at')
                            if new_info['direction'] == 'LONG':
                                new_info['high_price'] = max(new_info['high_price'], old_info.get('high_price', 0))
                            else:
                                new_info['low_price'] = min(new_info['low_price'], old_info.get('low_price', float('inf')))
                            new_info['tick_prices'] = old_info.get('tick_prices', [])
                            for _lvl in [1, 2]:
                                for _key in [f'phantom_be_l{_lvl}_triggered', f'phantom_be_l{_lvl}_triggered_at', f'phantom_be_l{_lvl}_would_exit_pnl']:
                                    if old_info.get(_key) is not None:
                                        new_info[_key] = old_info[_key]
                            for _lbl in ['a', 'b', 'c', 'd', 'e', 'f', 'g']:
                                for _key in [f'phantom_tick_{_lbl}_triggered', f'phantom_tick_{_lbl}_triggered_at', f'phantom_tick_{_lbl}_pnl']:
                                    if old_info.get(_key) is not None:
                                        new_info[_key] = old_info[_key]
                            if old_info.get('regime_neutral_hit'):
                                new_info['regime_neutral_hit'] = True
                                new_info['regime_neutral_hit_at'] = old_info.get('regime_neutral_hit_at')
                                new_info['regime_neutral_pnl'] = old_info.get('regime_neutral_pnl')
                            if old_info.get('regime_comeback_at') is not None:
                                new_info['regime_comeback_at'] = old_info['regime_comeback_at']
                                new_info['regime_comeback_pnl'] = old_info.get('regime_comeback_pnl')
                            if old_info.get('regime_opposite_at') is not None:
                                new_info['regime_opposite_at'] = old_info['regime_opposite_at']
                                new_info['regime_opposite_pnl'] = old_info.get('regime_opposite_pnl')
                            break
            _open_orders_cache.clear()
            _open_orders_cache.update(new_cache)
        
        logger.debug(f"[CACHE] Updated orders cache: {len(orders)} orders across {len(new_cache)} pairs")


# Global trading engine instance
trading_engine = TradingEngine()


async def realtime_stop_loss_callback(pair: str, price: float):
    """Callback function for WebSocket price updates to check stop loss in real-time"""
    await trading_engine.check_realtime_stop_loss(pair, price)
