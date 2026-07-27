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
        # Stamped ONLY by real WS messages and explicit tick=True updates
        # (REST watchdog fallback). The monitor loop echoes update_price()
        # every cycle with the tracker's own stale price, which resets
        # last_update — so last_update can NEVER measure stream silence.
        # last_tick can. (Jul-27 HBAR frozen-price incident.)
        self.last_tick: Optional[datetime] = datetime.utcnow() if initial_price else None
        self.trade_count: int = 0

    def update(self, price: float, tick: bool = False):
        """Update tracking with new price. tick=True only for real WS
        messages / REST watchdog fallback — it feeds the silence clock."""
        self.last_price = price
        self.last_update = datetime.utcnow()
        if tick:
            self.last_tick = self.last_update
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
        # Entry/reset price counts as a real observation → grace period
        # before the silence watchdog treats the pair as stale
        self.last_tick = self.last_update
        self.trade_count = 0

    def force_reset(self, initial_price: float = None):
        """Force complete reset - use only when starting fresh (new order on pair)

        Unlike reset(), this always overwrites existing tracking data.
        """
        self.high_price = initial_price
        self.low_price = initial_price
        self.last_price = initial_price
        self.last_update = datetime.utcnow() if initial_price else None
        self.last_tick = self.last_update
        self.trade_count = 0


class WebSocketTracker:
    """
    Real-time price tracker using Binance Futures WebSocket.
    
    Connects to Binance WebSocket and tracks high/low prices for subscribed pairs.
    This allows accurate trailing stop calculations even between polling intervals.
    Also supports real-time stop loss callbacks for instant order protection.
    """
    
    BINANCE_WS_URL = "wss://fstream.binance.com/stream"
    
    ZERO_PRICE_THRESHOLD = 50
    # Connection-level dead-stream watchdog: with dozens-to-hundreds of @trade
    # streams multiplexed, total silence for this long means the connection is
    # dead even if TCP/pings look alive. Added after the Jul-27 frozen-price
    # incident (staleness detection only counted received messages, so a
    # silent stream was invisible to it).
    SILENCE_TIMEOUT = 60

    def __init__(self):
        self.trackers: Dict[str, PriceTracker] = {}
        self.subscribed_pairs: Set[str] = set()
        self.websocket = None
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._reconnect_delay = 1  # Start with 1 second
        self._max_reconnect_delay = 60  # Max 60 seconds
        self._price_callback = None
        self._open_orders_callback = None
        self._consecutive_zero_prices = 0
        self._lock = asyncio.Lock()
        # Pair set the LIVE connection's stream URL was built from.
        # Reconnect decisions must compare against this, NOT subscribed_pairs:
        # force_reset_tracking() adds a pair to subscribed_pairs before
        # subscribe_pair() runs, which made subscribe_pair skip the reconnect
        # for genuinely-new pairs — spike-scanner pairs (outside the top-50
        # stream) were then permanently absent from the connection
        # (Jul-27 EVAA/FLOCK/HBAR frozen-price incidents).
        self._connected_pairs: Set[str] = set()
    
    def set_price_callback(self, callback):
        """Set callback function to be called on each price update.
        
        Callback signature: async def callback(pair: str, price: float)
        Used for real-time stop loss checking.
        """
        self._price_callback = callback
        logger.info("[WS_TRACKER] Price callback registered for real-time stop loss")

    def set_open_orders_callback(self, callback):
        """Set callback to check for open orders before forcing reconnect.

        Callback signature: async def callback() -> bool
        Returns True if there are open orders (reconnect should be skipped).
        """
        self._open_orders_callback = callback
        logger.info("[WS_TRACKER] Open-orders callback registered for staleness check")

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
            elif initial_price:
                # Already subscribed - just update with new price, preserving better high/low
                # This prevents destroying good tracking data on reconnections/restarts
                self.trackers[pair].update(initial_price)
                logger.debug(f"[WS_TRACKER] Updated {pair} price: {initial_price} (preserved tracking)")

            # Reconnect when the pair is missing from the LIVE connection's
            # stream list — regardless of how it got into subscribed_pairs.
            # (Checking subscribed_pairs membership here was the bug: a pair
            # pre-added by force_reset_tracking never triggered a reconnect
            # and stayed permanently silent.)
            if self.websocket and self.running and pair not in self._connected_pairs:
                logger.info(f"[WS_TRACKER] {pair} not in live stream — reconnecting to add it")
                await self._reconnect()
    
    async def subscribe_pairs_batch(self, pairs: list):
        """Subscribe multiple pairs at once with a single reconnection.
        
        Much more efficient than calling subscribe_pair() in a loop when
        adding many pairs (e.g. during scan_and_trade), because the
        WebSocket only reconnects once instead of once per new pair.
        """
        new_pairs = []
        async with self._lock:
            for pair in pairs:
                if pair not in self.subscribed_pairs:
                    self.subscribed_pairs.add(pair)
                    self.trackers[pair] = PriceTracker(pair)
                    new_pairs.append(pair)
            # Reconnect if ANY requested pair is missing from the live
            # connection (not only brand-new subscriptions) — heals pairs
            # that were pre-added without ever entering the stream.
            missing_live = [p for p in pairs if p not in self._connected_pairs]
            if missing_live and self.websocket and self.running:
                await self._reconnect()
        if new_pairs:
            logger.info(f"[WS_TRACKER] Batch subscribed {len(new_pairs)} new pairs (total: {len(self.subscribed_pairs)})")

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
    
    def update_price(self, pair: str, price: float, tick: bool = False):
        """Manually update price. tick=True ONLY for genuinely fresh prices
        (REST watchdog fallback) — it feeds the silence clock. The monitor
        loop's per-cycle echo of the tracker's own price must stay tick=False
        or the watchdog can never detect a dead stream."""
        tracker = self.trackers.get(pair)
        if tracker:
            tracker.update(price, tick=tick)
        else:
            # Create the tracker so fallback updates are recorded (and the
            # watchdog's 90s silence clock works) even if subscription lagged
            self.trackers[pair] = PriceTracker(pair, price)
    
    async def _staleness_reconnect(self):
        """Force reconnect to recover a stale stream.

        Reconnects UNCONDITIONALLY — including with open orders. Reconnects do
        NOT destroy tracking data (high/low/last_price survive; only
        force_reset clears them), so a stale stream must always be recovered:
        an open position on a dead stream is price-blind and its realtime SL
        is starved (Jul-27 incident: FLOCK/EVAA frozen at entry for ~1h
        because the old code skipped reconnect while orders were open).
        """
        if self._open_orders_callback:
            try:
                has_open = await self._open_orders_callback()
                if has_open:
                    logger.warning(
                        "[WS_TRACKER] Stale stream with OPEN ORDERS — reconnecting "
                        "(tracking data is preserved across reconnects)"
                    )
            except Exception as e:
                logger.error(f"[WS_TRACKER] Error checking open orders: {e}")

        logger.info("[WS_TRACKER] Forcing reconnect to recover stale stream")
        self._consecutive_zero_prices = 0
        await self._reconnect()

    async def force_reconnect(self, reason: str = ""):
        """Public forced reconnect (e.g. engine watchdog found a silent open-order pair)."""
        logger.warning(f"[WS_TRACKER] Forced reconnect requested{': ' + reason if reason else ''}")
        self._consecutive_zero_prices = 0
        await self._reconnect()

    def is_pair_streamed(self, pair: str) -> bool:
        """True if the LIVE connection's stream list includes this pair.
        False = the pair can never tick on this connection (reconnect needed)."""
        return pair in self._connected_pairs

    def pair_silence_seconds(self, pair: str) -> Optional[float]:
        """Seconds since the last REAL tick for a pair (None = never ticked).

        Reads last_tick, not last_update — the monitor loop echoes the
        tracker's own stale price back via update_price() every cycle,
        which refreshes last_update and would blind this clock (the bug
        that kept the Jul-27 watchdog from ever firing on HBAR).
        """
        tracker = self.trackers.get(pair)
        if not tracker or tracker.last_tick is None:
            return None
        return (datetime.utcnow() - tracker.last_tick).total_seconds()

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
                # Snapshot the pair set the URL is built from. Pairs subscribed
                # while the connection is being established skip _reconnect()
                # (self.websocket is still None) and would otherwise be
                # permanently missing from this connection's stream list —
                # the Jul-27 FLOCK/EVAA frozen-price race.
                pairs_snapshot = set(self.subscribed_pairs)
                url = self._build_ws_url()

                if not url:
                    # No pairs subscribed, wait and retry
                    await asyncio.sleep(1)
                    continue

                logger.info(f"[WS_TRACKER] Connecting to WebSocket ({len(pairs_snapshot)} pairs)...")

                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5
                ) as ws:
                    self.websocket = ws
                    self._connected_pairs = pairs_snapshot  # what this connection actually streams
                    self._reconnect_delay = 1  # Reset delay on successful connection
                    logger.info(f"[WS_TRACKER] Connected! Tracking: {', '.join(self.subscribed_pairs)}")

                    missing = self.subscribed_pairs - pairs_snapshot
                    if missing:
                        logger.warning(
                            f"[WS_TRACKER] {len(missing)} pair(s) subscribed during connect "
                            f"({', '.join(sorted(missing))}) — reconnecting with full stream list"
                        )
                        continue  # exits the context manager, which closes ws

                    while self.running:
                        try:
                            message = await asyncio.wait_for(ws.recv(), timeout=self.SILENCE_TIMEOUT)
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"[WS_WATCHDOG] No messages for {self.SILENCE_TIMEOUT}s on a "
                                f"{len(pairs_snapshot)}-stream connection — dead stream, reconnecting"
                            )
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

                if price <= 0:
                    self._consecutive_zero_prices += 1
                    if self._consecutive_zero_prices % self.ZERO_PRICE_THRESHOLD == 0:
                        logger.warning(
                            f"[WS_TRACKER] {self._consecutive_zero_prices} consecutive zero-price "
                            f"messages (latest: {pair}). Checking if safe to reconnect..."
                        )
                        await self._staleness_reconnect()
                    return

                self._consecutive_zero_prices = 0

                tracker = self.trackers.get(pair)
                if tracker:
                    tracker.update(price, tick=True)

                if self._price_callback:
                    try:
                        await self._price_callback(pair, price)
                    except Exception as e:
                        logger.error(f"[WS_TRACKER] Price callback error for {pair}: {e}")


# Global WebSocket tracker instance
websocket_tracker = WebSocketTracker()
