"""
SCALPARS Trading Platform - WebSocket Price Tracker

Real-time price tracking via Binance Futures WebSocket for accurate high/low tracking.
"""
import asyncio
import json
import logging
from typing import Dict, Optional, Set
from datetime import datetime
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)


class PriceTracker:
    """Tracks high/low prices for a single pair"""
    
    def __init__(self, pair: str, initial_price: float = None):
        self.pair = pair
        self.high_price: Optional[float] = initial_price
        self.low_price: Optional[float] = initial_price
        self.last_price: Optional[float] = initial_price
        self.last_update: Optional[datetime] = datetime.utcnow() if initial_price else None
        self.trade_count: int = 0
    
    def update(self, price: float):
        """Update tracking with new price"""
        self.last_price = price
        self.last_update = datetime.utcnow()
        self.trade_count += 1
        
        # Track new highs (important for LONG trailing stops)
        if self.high_price is None or price > self.high_price:
            old_high = self.high_price
            self.high_price = price
            # Log significant high updates (first 5 trades or new highs)
            if self.trade_count <= 5 or (old_high and price > old_high * 1.0001):
                logger.debug(f"[TRACKER] {self.pair} NEW HIGH: {old_high} -> {price}")
        
        # Track new lows (important for SHORT trailing stops)
        if self.low_price is None or price < self.low_price:
            old_low = self.low_price
            self.low_price = price
            # Log significant low updates (first 5 trades or new lows)
            if self.trade_count <= 5 or (old_low and price < old_low * 0.9999):
                logger.debug(f"[TRACKER] {self.pair} NEW LOW: {old_low} -> {price}")
    
    def reset(self, initial_price: float = None):
        """Reset tracking, but preserve better prices if they exist
        
        This prevents destroying good tracking data on server restarts or reconnections.
        """
        if initial_price is None:
            # Full reset - clear everything
            self.high_price = None
            self.low_price = None
        else:
            # Smart reset - only update if initial is better or no existing value
            # This preserves good tracking (e.g., low_price for SHORT orders)
            if self.high_price is None or initial_price > self.high_price:
                self.high_price = initial_price
            if self.low_price is None or initial_price < self.low_price:
                self.low_price = initial_price
        
        self.last_price = initial_price
        self.last_update = datetime.utcnow() if initial_price else None
        self.trade_count = 0
    
    def force_reset(self, initial_price: float = None):
        """Force complete reset - use only when starting fresh (new order on pair)
        
        Unlike reset(), this always overwrites existing tracking data.
        """
        self.high_price = initial_price
        self.low_price = initial_price
        self.last_price = initial_price
        self.last_update = datetime.utcnow() if initial_price else None
        self.trade_count = 0


class WebSocketTracker:
    """
    Real-time price tracker using Binance Futures WebSocket.
    
    Connects to Binance WebSocket and tracks high/low prices for subscribed pairs.
    This allows accurate trailing stop calculations even between polling intervals.
    Also supports real-time stop loss callbacks for instant order protection.
    """
    
    BINANCE_WS_URL = "wss://fstream.binance.com/stream"
    
    def __init__(self):
        self.trackers: Dict[str, PriceTracker] = {}
        self.subscribed_pairs: Set[str] = set()
        self.websocket = None
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1  # Start with 1 second
        self._max_reconnect_delay = 60  # Max 60 seconds
        self._price_callback = None  # Callback for real-time price updates (e.g., stop loss checking)
    
    def set_price_callback(self, callback):
        """Set callback function to be called on each price update.
        
        Callback signature: async def callback(pair: str, price: float)
        Used for real-time stop loss checking.
        """
        self._price_callback = callback
        logger.info("[WS_TRACKER] Price callback registered for real-time stop loss")
        self._lock = asyncio.Lock()
    
    async def start(self):
        """Start the WebSocket tracker"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._run_forever())
        logger.info("[WS_TRACKER] WebSocket tracker started")
    
    async def stop(self):
        """Stop the WebSocket tracker"""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info("[WS_TRACKER] WebSocket tracker stopped")
    
    async def subscribe_pair(self, pair: str, initial_price: float = None):
        """Subscribe to price updates for a pair
        
        Note: This does NOT reset existing tracking data. Use force_reset_tracking()
        if you need to start fresh (e.g., when opening a new order).
        """
        async with self._lock:
            # Normalize pair name (e.g., BTCUSDT -> btcusdt)
            pair_lower = pair.lower()
            
            if pair not in self.subscribed_pairs:
                self.subscribed_pairs.add(pair)
                self.trackers[pair] = PriceTracker(pair, initial_price)
                logger.info(f"[WS_TRACKER] Subscribed to {pair} (initial: {initial_price})")
                
                # Reconnect to update subscriptions
                if self.websocket and self.running:
                    await self._reconnect()
            elif initial_price:
                # Already subscribed - just update with new price, preserving better high/low
                # This prevents destroying good tracking data on reconnections/restarts
                self.trackers[pair].update(initial_price)
                logger.debug(f"[WS_TRACKER] Updated {pair} price: {initial_price} (preserved tracking)")
    
    async def unsubscribe_pair(self, pair: str):
        """Unsubscribe from price updates for a pair"""
        async with self._lock:
            if pair in self.subscribed_pairs:
                self.subscribed_pairs.discard(pair)
                if pair in self.trackers:
                    del self.trackers[pair]
                logger.info(f"[WS_TRACKER] Unsubscribed from {pair}")
                
                # Reconnect to update subscriptions
                if self.websocket and self.running:
                    await self._reconnect()
    
    def get_high_low(self, pair: str) -> tuple[Optional[float], Optional[float]]:
        """Get tracked high and low prices for a pair"""
        tracker = self.trackers.get(pair)
        if tracker:
            return tracker.high_price, tracker.low_price
        return None, None
    
    def get_tracker(self, pair: str) -> Optional[PriceTracker]:
        """Get the price tracker for a pair"""
        return self.trackers.get(pair)
    
    def reset_tracking(self, pair: str, initial_price: float = None):
        """Reset high/low tracking for a pair, preserving better prices
        
        This is a soft reset - it won't overwrite better existing tracking.
        Use force_reset_tracking() for a complete reset.
        """
        tracker = self.trackers.get(pair)
        if tracker:
            tracker.reset(initial_price)
            logger.info(f"[WS_TRACKER] Soft reset tracking for {pair} (initial: {initial_price})")
    
    def force_reset_tracking(self, pair: str, initial_price: float = None):
        """Force complete reset of tracking for a pair
        
        Use this when opening a NEW order - we need fresh tracking from entry price.
        Unlike reset_tracking(), this always overwrites existing data.
        Also ensures the pair is in subscribed_pairs for proper WebSocket tracking.
        """
        tracker = self.trackers.get(pair)
        if tracker:
            tracker.force_reset(initial_price)
            logger.info(f"[WS_TRACKER] Force reset tracking for {pair} (initial: {initial_price})")
        else:
            # Create new tracker if doesn't exist
            self.trackers[pair] = PriceTracker(pair, initial_price)
            logger.info(f"[WS_TRACKER] Created new tracker for {pair} (initial: {initial_price})")
        
        # Ensure pair is in subscribed_pairs (needed for WebSocket to track it)
        if pair not in self.subscribed_pairs:
            self.subscribed_pairs.add(pair)
            logger.info(f"[WS_TRACKER] Added {pair} to subscribed_pairs")
    
    def update_price(self, pair: str, price: float):
        """Manually update price (fallback when WebSocket not available)"""
        tracker = self.trackers.get(pair)
        if tracker:
            tracker.update(price)
    
    async def _reconnect(self):
        """Reconnect WebSocket with updated subscriptions"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None
    
    def _build_ws_url(self) -> str:
        """Build WebSocket URL with all subscribed streams"""
        if not self.subscribed_pairs:
            return None
        
        # Build stream names (e.g., btcusdt@trade)
        streams = [f"{pair.lower()}@trade" for pair in self.subscribed_pairs]
        stream_param = "/".join(streams)
        
        return f"{self.BINANCE_WS_URL}?streams={stream_param}"
    
    async def _run_forever(self):
        """Main WebSocket loop with auto-reconnect"""
        while self.running:
            try:
                url = self._build_ws_url()
                
                if not url:
                    # No pairs subscribed, wait and retry
                    await asyncio.sleep(1)
                    continue
                
                logger.info(f"[WS_TRACKER] Connecting to WebSocket ({len(self.subscribed_pairs)} pairs)...")
                
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self.websocket = ws
                    self._reconnect_delay = 1  # Reset delay on successful connection
                    logger.info(f"[WS_TRACKER] Connected! Tracking: {', '.join(self.subscribed_pairs)}")
                    
                    async for message in ws:
                        if not self.running:
                            break
                        
                        try:
                            data = json.loads(message)
                            await self._handle_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"[WS_TRACKER] Invalid JSON: {message[:100]}")
                        except Exception as e:
                            logger.error(f"[WS_TRACKER] Error handling message: {e}")
                
            except ConnectionClosed as e:
                logger.warning(f"[WS_TRACKER] Connection closed: {e}")
            except Exception as e:
                logger.error(f"[WS_TRACKER] WebSocket error: {e}")
            
            self.websocket = None
            
            if self.running:
                # Exponential backoff for reconnection
                logger.info(f"[WS_TRACKER] Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
    
    async def _handle_message(self, data: dict):
        """Handle incoming WebSocket message"""
        # Binance combined stream format: {"stream": "btcusdt@trade", "data": {...}}
        if "stream" in data and "data" in data:
            stream = data["stream"]
            trade_data = data["data"]
            
            # Extract pair from stream name (e.g., "btcusdt@trade" -> "BTCUSDT")
            pair = stream.split("@")[0].upper()
            
            # Extract price from trade data
            if "p" in trade_data:
                price = float(trade_data["p"])
                
                # CRITICAL: Never process invalid prices
                if price <= 0:
                    logger.warning(f"[WS_TRACKER] Received invalid price {price} for {pair}, skipping")
                    return
                
                # Update tracker
                tracker = self.trackers.get(pair)
                if tracker:
                    tracker.update(price)
                
                # Call price callback for real-time stop loss checking
                if self._price_callback:
                    try:
                        await self._price_callback(pair, price)
                    except Exception as e:
                        # Don't let callback errors break the WebSocket loop
                        logger.error(f"[WS_TRACKER] Price callback error for {pair}: {e}")


# Global WebSocket tracker instance
websocket_tracker = WebSocketTracker()
