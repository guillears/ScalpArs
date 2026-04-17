"""
SCALPARS Trading Platform - Binance Service
"""
import ccxt.async_support as ccxt
from ccxt.base.errors import RateLimitExceeded, DDoSProtection, ExchangeNotAvailable
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
import re
import time
from config import settings, trading_config
from datetime import datetime

logger = logging.getLogger(__name__)

_ban_until: float = 0
_ban_persist_callback = None
POST_BAN_COOLDOWN = 60

_leverage_blocked_pairs: set = set()


def set_ban_persist_callback(callback):
    """Register a callback that persists ban_until to the database."""
    global _ban_persist_callback
    _ban_persist_callback = callback


def set_ban_until(value: float):
    """Set the ban expiry from external code (e.g. loaded from DB at startup)."""
    global _ban_until
    _ban_until = value
    if value > 0:
        wait = value - time.time()
        if wait > 0:
            logger.warning(f"[BINANCE] Ban state restored from DB, {wait:.0f}s remaining")
        else:
            logger.info("[BINANCE] Ban state from DB already expired, clearing")
            _ban_until = 0


def get_ban_status() -> dict:
    """Return current ban state for API consumers."""
    if _ban_until > 0:
        remaining = _ban_until - time.time()
        if remaining > 0:
            return {"banned": True, "remaining_seconds": int(remaining)}
    return {"banned": False, "remaining_seconds": 0}


class BinanceService:
    """Service for interacting with Binance Futures API"""
    
    def __init__(self):
        # Public exchange for market data (no auth needed)
        self.public_exchange = ccxt.binanceusdm({
            'enableRateLimit': True,
            'sandbox': False,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Private exchange for trading (requires API keys)
        self.exchange = ccxt.binanceusdm({
            'apiKey': settings.binance_api_key,
            'secret': settings.binance_api_secret,
            'enableRateLimit': True,
            'sandbox': False,
            'options': {
                'defaultType': 'future',
                'adjustForTimeDifference': True
            }
        })
        
        # Spot exchange for BNB purchases (same API keys)
        self.spot_exchange = ccxt.binance({
            'apiKey': settings.binance_api_key,
            'secret': settings.binance_api_secret,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True
            }
        })
        self._markets_loaded = False
        self._public_markets_loaded = False
        self._spot_markets_loaded = False
    
    async def load_markets(self):
        """Load market data for private exchange"""
        if not self._markets_loaded:
            await self.exchange.load_markets()
            self._markets_loaded = True
    
    async def load_public_markets(self):
        """Load market data for public exchange"""
        if not self._public_markets_loaded:
            await self.public_exchange.load_markets()
            self._public_markets_loaded = True
    
    async def _check_ban(self):
        """If Binance has IP-banned us, sleep until the ban expires + cooldown buffer."""
        global _ban_until
        if _ban_until > 0:
            now = time.time()
            if now < _ban_until:
                wait = _ban_until - now + 2
                logger.warning(f"[BINANCE] IP banned, waiting {wait:.0f}s until ban expires")
                await asyncio.sleep(wait)
            logger.info(f"[BINANCE] Ban expired, waiting {POST_BAN_COOLDOWN}s cooldown before resuming API calls")
            await asyncio.sleep(POST_BAN_COOLDOWN)
            _ban_until = 0
            if _ban_persist_callback:
                try:
                    _ban_persist_callback(0)
                except Exception:
                    pass

    @staticmethod
    def _detect_ban(error):
        """Check if an error is an IP ban and record the expiry."""
        global _ban_until
        match = re.search(r'banned until (\d+)', str(error))
        if match:
            _ban_until = int(match.group(1)) / 1000
            logger.error(f"[BINANCE] IP ban detected, expires at {_ban_until:.0f} (epoch)")
            if _ban_persist_callback:
                try:
                    _ban_persist_callback(_ban_until)
                except Exception as cb_err:
                    logger.error(f"[BINANCE] Failed to persist ban state: {cb_err}")

    async def _load_spot_markets(self):
        """Load market data for spot exchange"""
        if not self._spot_markets_loaded:
            await self.spot_exchange.load_markets()
            self._spot_markets_loaded = True

    async def get_bnb_price(self) -> float:
        """Get current BNB/USDT price from spot market"""
        try:
            await self.load_public_markets()
            ticker = await self.public_exchange.fetch_ticker('BNB/USDT:USDT')
            return float(ticker.get('last', 0))
        except Exception:
            try:
                await self._load_spot_markets()
                ticker = await self.spot_exchange.fetch_ticker('BNB/USDT')
                return float(ticker.get('last', 0))
            except Exception as e:
                logger.error(f"[BINANCE] Error fetching BNB price: {e}")
                return 0.0

    async def buy_bnb(self, amount_usdt: float) -> Optional[Dict]:
        """Buy BNB with USDT via transfer-to-spot + spot market buy + transfer-back.
        Four steps; all non-time-sensitive, no expiring quotes."""
        try:
            await self.load_markets()
            await self._load_spot_markets()

            # Step 1: Transfer USDT from futures wallet to spot wallet
            logger.info(f"[BNB_SWAP] Step 1/4: Transferring {amount_usdt} USDT futures → spot")
            await self.spot_exchange.transfer('USDT', amount_usdt, 'future', 'spot')

            # Step 2: Buy BNB on spot using quoteOrderQty (spend exact USDT amount)
            logger.info(f"[BNB_SWAP] Step 2/4: Buying BNB with {amount_usdt} USDT on spot")
            order = await self.spot_exchange.create_order(
                'BNB/USDT', 'market', 'buy', None, None,
                {'quoteOrderQty': round(amount_usdt, 2)}
            )

            avg_price = float(order.get('average', order.get('price', 0)))
            cost = float(order.get('cost', amount_usdt))
            order_id = order.get('id', 'spot_buy')

            # Step 3: Fetch actual spot balances and transfer BNB back to futures
            spot_bal = await self.spot_exchange.fetch_balance()
            actual_bnb = float(spot_bal.get('BNB', {}).get('free', 0))
            if actual_bnb <= 0:
                logger.error("[BNB_SWAP] No BNB on spot after buy. Check order status.")
                return None

            if avg_price <= 0 and actual_bnb > 0:
                avg_price = cost / actual_bnb

            logger.info(f"[BNB_SWAP] Step 3/4: Transferring {actual_bnb} BNB spot → futures")
            await self.spot_exchange.transfer('BNB', actual_bnb, 'spot', 'future')

            # Step 4: Return any leftover USDT (from lot-size rounding) back to futures
            leftover_usdt = float(spot_bal.get('USDT', {}).get('free', 0))
            if leftover_usdt > 0.01:
                logger.info(f"[BNB_SWAP] Step 4/4: Returning {leftover_usdt:.2f} leftover USDT spot → futures")
                await self.spot_exchange.transfer('USDT', leftover_usdt, 'spot', 'future')

            logger.info(f"[BNB_SWAP] Complete: {cost:.2f} USDT → {actual_bnb} BNB @ {avg_price:.2f}")
            return {
                'bnb_amount': actual_bnb,
                'price': avg_price,
                'cost_usdt': cost,
                'order_id': str(order_id)
            }
        except Exception as e:
            logger.error(f"[BNB_SWAP] Failed: {e}. If transfer already happened, check spot wallet for stranded funds.")
            return None

    async def close(self):
        """Close exchange connections"""
        await self.exchange.close()
        await self.public_exchange.close()
        await self.spot_exchange.close()
    
    async def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            await self.load_markets()
            balance = await self.exchange.fetch_balance()
            
            usdt_balance = balance.get('USDT', {})
            bnb_balance = balance.get('BNB', {})
            
            # Extract stable Wallet Balance and USDT-only Available Balance
            # from raw Binance response (the CCXT 'free' field is account-wide,
            # which includes BNB value — we want USDT-only).
            usdt_wallet = float(usdt_balance.get('total', 0))
            usdt_free = float(usdt_balance.get('free', 0))
            raw_info = balance.get('info', {})
            for asset in raw_info.get('assets', []):
                if asset.get('asset') == 'USDT':
                    usdt_wallet = float(asset.get('walletBalance', usdt_wallet))
                    usdt_free = float(asset.get('maxWithdrawAmount', usdt_free))
                    break
            
            return {
                'usdt_free': usdt_free,
                'usdt_used': float(usdt_balance.get('used', 0)),
                'usdt_total': usdt_wallet,
                'bnb_free': float(bnb_balance.get('free', 0)),
                'bnb_total': float(bnb_balance.get('total', 0)),
                'total_portfolio': float(balance.get('total', {}).get('USDT', 0))
            }
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching balance: {e}")
            return {
                'usdt_free': 0,
                'usdt_used': 0,
                'usdt_total': 0,
                'bnb_free': 0,
                'bnb_total': 0,
                'total_portfolio': 0
            }
    
    async def get_top_futures_pairs(
        self,
        limit: int = 20,
        new_listing_filter_days: int = 0,
    ) -> List[Dict]:
        """Get top USDT-perpetual futures pairs by 24h volume.

        When ``new_listing_filter_days`` > 0, pairs whose Binance onboardDate
        (from exchangeInfo, surfaced via CCXT ``markets[symbol]['info']``) is
        within the last N days are excluded *before* the top-N-by-volume cut.
        This filters out Binance's Seed Tag / Monitoring Tag pairs — low
        liquidity, manipulation-prone, poor fit for 5m-EMA strategy.  See
        CLAUDE.md Apr 17 analysis for the RAVEUSDT blow-up that motivated this.
        """
        try:
            await self._check_ban()
            await self.load_public_markets()

            tickers = None
            for attempt in range(3):
                try:
                    tickers = await self.public_exchange.fetch_tickers()
                    break
                except (RateLimitExceeded, DDoSProtection, ExchangeNotAvailable) as e:
                    self._detect_ban(e)
                    if _ban_until > 0:
                        await self._check_ban()
                        continue
                    wait = (attempt + 1) * 5
                    logger.warning(f"[BINANCE] Rate limited fetching tickers, waiting {wait}s (attempt {attempt + 1}/3): {e}")
                    await asyncio.sleep(wait)
                except Exception as e:
                    if 'Too many requests' in str(e) or '1003' in str(e) or '429' in str(e) or 'banned' in str(e).lower():
                        self._detect_ban(e)
                        if _ban_until > 0:
                            await self._check_ban()
                            continue
                        wait = (attempt + 1) * 5
                        logger.warning(f"[BINANCE] Rate limited fetching tickers, waiting {wait}s (attempt {attempt + 1}/3): {e}")
                        await asyncio.sleep(wait)
                    else:
                        raise

            if tickers is None:
                logger.error("[BINANCE] Failed to fetch tickers after 3 attempts")
                return []

            # Filter USDT perpetual futures and sort by volume
            futures_pairs = []
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT:USDT'):  # USDT perpetual futures
                    # Safe conversion with defaults for None values
                    last_price = ticker.get('last')
                    quote_volume = ticker.get('quoteVolume')
                    percentage = ticker.get('percentage')

                    # Skip if no price data
                    if last_price is None:
                        continue

                    futures_pairs.append({
                        'symbol': symbol,
                        'pair': symbol.replace('/USDT:USDT', 'USDT'),
                        'price': float(last_price) if last_price is not None else 0.0,
                        'volume_24h': float(quote_volume) if quote_volume is not None else 0.0,
                        'change_24h': float(percentage) if percentage is not None else 0.0
                    })

            # New-listing filter: drop pairs listed within the last N days,
            # based on Binance's onboardDate in market metadata.  Applied
            # BEFORE the top-N-by-volume cut so "top 50" stays "top 50 of
            # eligible pairs."  Fails open: pairs without a parseable
            # onboardDate are kept (conservative — don't accidentally block
            # established pairs due to missing metadata).
            if new_listing_filter_days > 0:
                import time as _time
                cutoff_ms = int((_time.time() - new_listing_filter_days * 86400) * 1000)
                markets = self.public_exchange.markets or {}
                before_count = len(futures_pairs)
                filtered_pairs = []
                filtered_out_names = []
                for p in futures_pairs:
                    market = markets.get(p['symbol'], {})
                    info = market.get('info', {}) if isinstance(market, dict) else {}
                    onboard_raw = info.get('onboardDate')
                    if onboard_raw is None:
                        # No metadata -> keep (fail open)
                        filtered_pairs.append(p)
                        continue
                    try:
                        onboard_ms = int(onboard_raw)
                    except (ValueError, TypeError):
                        filtered_pairs.append(p)
                        continue
                    if onboard_ms >= cutoff_ms:
                        # Listed within the filter window -> skip
                        filtered_out_names.append(p['pair'])
                    else:
                        filtered_pairs.append(p)
                futures_pairs = filtered_pairs
                if filtered_out_names:
                    _preview = ', '.join(sorted(filtered_out_names)[:8])
                    _extra = f" +{len(filtered_out_names) - 8} more" if len(filtered_out_names) > 8 else ""
                    logger.info(
                        f"[BINANCE] New-listing filter ({new_listing_filter_days}d): "
                        f"excluded {len(filtered_out_names)}/{before_count} pairs "
                        f"({_preview}{_extra})"
                    )

            # Sort by volume descending
            futures_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)

            return futures_pairs[:limit]
        except Exception as e:
            self._detect_ban(e)
            logger.error(f"[BINANCE] Error fetching top pairs: {e}", exc_info=True)
            return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> List:
        """Get OHLCV data for indicator calculation"""
        try:
            await self._check_ban()
            await self.load_public_markets()
        except Exception as e:
            self._detect_ban(e)
            logger.error(f"[BINANCE] Error initializing for OHLCV {symbol}: {e}")
            return []
        for attempt in range(3):
            try:
                ohlcv = await self.public_exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                return ohlcv
            except (RateLimitExceeded, DDoSProtection, ExchangeNotAvailable) as e:
                self._detect_ban(e)
                if _ban_until > 0:
                    await self._check_ban()
                    continue
                wait = (attempt + 1) * 5
                logger.warning(f"[BINANCE] Rate limited fetching OHLCV for {symbol}, waiting {wait}s (attempt {attempt + 1}/3): {e}")
                await asyncio.sleep(wait)
            except Exception as e:
                if 'Too many requests' in str(e) or '1003' in str(e) or '429' in str(e) or 'banned' in str(e).lower():
                    self._detect_ban(e)
                    if _ban_until > 0:
                        await self._check_ban()
                        continue
                    wait = (attempt + 1) * 5
                    logger.warning(f"[BINANCE] Rate limited fetching OHLCV for {symbol}, waiting {wait}s (attempt {attempt + 1}/3): {e}")
                    await asyncio.sleep(wait)
                else:
                    logger.error(f"[BINANCE] Error fetching OHLCV for {symbol}: {e}")
                    return []
        logger.error(f"[BINANCE] Failed to fetch OHLCV for {symbol} after 3 attempts")
        return []
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        try:
            await self._check_ban()
            await self.load_public_markets()
            ticker = await self.public_exchange.fetch_ticker(symbol)
            return float(ticker.get('last', 0))
        except (DDoSProtection, RateLimitExceeded) as e:
            self._detect_ban(e)
            logger.error(f"[BINANCE] Rate limited fetching price for {symbol}: {e}")
            return 0.0
        except Exception as e:
            self._detect_ban(e)
            logger.error(f"[BINANCE] Error fetching price for {symbol}: {e}")
            return 0.0
    
    async def set_leverage(self, symbol: str, leverage: int) -> int:
        """Set leverage for a symbol. Returns actual leverage applied, or 0 on failure."""
        try:
            await self.load_markets()
            result = await self.exchange.set_leverage(leverage, symbol)
            actual = int(result.get('leverage', leverage)) if isinstance(result, dict) else leverage
            if actual != leverage:
                logger.warning(f"[BINANCE] Leverage for {symbol}: requested {leverage}x but got {actual}x")
            return actual
        except Exception as e:
            logger.warning(f"[BINANCE] set_leverage({symbol}, {leverage}x) failed: {e}")
            try:
                pos = await self.get_position(symbol)
                if pos and pos.get('leverage'):
                    actual = int(pos['leverage'])
                    logger.info(f"[BINANCE] Current leverage for {symbol}: {actual}x (from position)")
                    return actual
            except Exception:
                pass
            logger.error(f"[BINANCE] Cannot determine actual leverage for {symbol}")
            return 0
    
    async def create_market_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        amount: float,
        leverage: int = 1,
        is_close: bool = False
    ) -> Optional[Dict]:
        """Create a market order"""
        try:
            await self.load_markets()

            if not is_close:
                actual_leverage = await self.set_leverage(symbol, leverage)
                if actual_leverage == 0:
                    logger.error(f"[LEVERAGE_MISMATCH] {symbol}: Cannot determine leverage, skipping order")
                    _leverage_blocked_pairs.add(symbol)
                    return None
                if actual_leverage != leverage:
                    logger.warning(f"[LEVERAGE_MISMATCH] {symbol}: Binance leverage {actual_leverage}x != configured {leverage}x — blocking pair")
                    _leverage_blocked_pairs.add(symbol)
                    return None
            
            # reduceOnly prevents position flip on close orders
            params = {'reduceOnly': True} if is_close else {}
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params=params
            )
            
            return {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': float(order.get('amount') or amount),
                'price': float(order.get('average') or order.get('price') or 0),
                'cost': float(order.get('cost') or 0),
                'fee': float((order.get('fee') or {}).get('cost', 0)),
                'timestamp': order.get('timestamp', datetime.now().timestamp() * 1000)
            }
        except Exception as e:
            logger.error(f"[BINANCE] Error creating order for {symbol}: {e}")
            return None
    
    async def get_tick_size(self, symbol: str) -> float:
        """Get the minimum price increment (tick size) for a symbol"""
        try:
            await self.load_markets()
            market = self.exchange.market(symbol)
            tick = market.get('precision', {}).get('price')
            if tick is not None:
                if isinstance(tick, int):
                    return 10 ** (-tick)
                return float(tick)
            return 0.01
        except Exception as e:
            logger.error(f"[BINANCE] Error getting tick size for {symbol}: {e}")
            return 0.01

    async def fetch_orderbook(self, symbol: str, limit: int = 5) -> Optional[Dict]:
        """Get best bid/ask from orderbook"""
        try:
            await self.load_markets()
            ob = await self.exchange.fetch_order_book(symbol, limit)
            if ob and ob.get('bids') and ob.get('asks'):
                return {
                    'best_bid': float(ob['bids'][0][0]),
                    'best_ask': float(ob['asks'][0][0]),
                    'bid_qty': float(ob['bids'][0][1]),
                    'ask_qty': float(ob['asks'][0][1]),
                }
            return None
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching orderbook for {symbol}: {e}")
            return None

    async def create_limit_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        leverage: int = 1,
        is_close: bool = False
    ) -> Optional[Dict]:
        """Place a limit (maker) order"""
        try:
            await self.load_markets()
            if not is_close:
                actual_leverage = await self.set_leverage(symbol, leverage)
                if actual_leverage == 0:
                    logger.error(f"[LEVERAGE_MISMATCH] {symbol}: Cannot determine leverage, skipping limit order")
                    _leverage_blocked_pairs.add(symbol)
                    return None
                if actual_leverage != leverage:
                    logger.warning(f"[LEVERAGE_MISMATCH] {symbol}: Binance leverage {actual_leverage}x != configured {leverage}x — blocking pair")
                    _leverage_blocked_pairs.add(symbol)
                    return None

            # reduceOnly prevents position flip on close orders
            params = {'reduceOnly': True} if is_close else {}
            order = await self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=side,
                amount=amount,
                price=price,
                params=params
            )

            return {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': float(order.get('amount') or amount),
                'price': float(order.get('price') or price),
                'status': order.get('status', 'open'),
                'filled': float(order.get('filled') or 0),
                'remaining': float(order.get('remaining') or amount),
                'fee': float((order.get('fee') or {}).get('cost', 0)),
                'timestamp': order.get('timestamp', datetime.now().timestamp() * 1000)
            }
        except Exception as e:
            logger.error(f"[BINANCE] Error creating limit order for {symbol}: {e}")
            return None

    async def fetch_order_status(self, symbol: str, order_id: str) -> Optional[Dict]:
        """Check the fill status of an order"""
        try:
            await self.load_markets()
            order = await self.exchange.fetch_order(order_id, symbol)
            return {
                'id': order['id'],
                'status': order.get('status', 'unknown'),
                'filled': float(order.get('filled') or 0),
                'remaining': float(order.get('remaining') or 0),
                'average': float(order.get('average') or order.get('price') or 0),
                'fee': float((order.get('fee') or {}).get('cost', 0)),
            }
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching order status {order_id} for {symbol}: {e}")
            return None

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an unfilled or partially filled order"""
        try:
            await self.load_markets()
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.error(f"[BINANCE] Error cancelling order {order_id} for {symbol}: {e}")
            return False

    async def close_position(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Close a position"""
        # To close a LONG, we sell. To close a SHORT, we buy.
        close_side = 'sell' if side == 'LONG' else 'buy'
        return await self.create_market_order(symbol, close_side, amount, is_close=True)

    async def place_protective_stops(
        self,
        symbol: str,
        direction: str,         # "LONG" or "SHORT" — the position direction
        quantity: float,        # position size (base asset units)
        entry_price: float,
        sl_pct: float,          # e.g. 1.5 for -1.5% SL from entry
        tp_pct: float,          # e.g. 5.0 for +5.0% TP from entry
    ) -> Dict[str, Optional[str]]:
        """Place broker-side protective STOP_MARKET + TAKE_PROFIT_MARKET orders
        for an existing position.  Both use reduceOnly=true with explicit
        quantity (matches CCXT's reliable path on Binance Futures).  IMPORTANT:
        we intentionally do NOT set closePosition=true — Binance rejects that
        with error -4120 "Order type not supported for this endpoint" and
        routes it to the Algo Order API which CCXT doesn't handle cleanly.
        Using reduceOnly+quantity on the standard /fapi/v1/order endpoint is
        the well-tested path.  Apr 17 hotfix after -4120 production failures.

        Because we lose the closePosition=true auto-cancel behavior, the
        close path (services/trading_engine._close_position_locked) MUST
        explicitly call cancel_protective_stops after a bot-initiated close
        to avoid orphan orders triggering on future positions on the same
        pair.  reduceOnly prevents flip-into-new-position from orphans, but
        they could still close a fresh position at a stale trigger price if
        not cancelled.

        Fires ONLY if the bot's own exit logic doesn't run — system-down
        insurance (Apr 11 scenario).  SL is placed 0.3% below the bot's
        -1.2% FL_EMERGENCY_SL backstop; TP is placed well above normal
        trailing-stop exits.  See CLAUDE.md "Broker-side Protective Stops".

        Fails open: if either order fails to place, returns the IDs of
        whichever succeeded (or None).  Never raises — a protective-order
        failure must NOT block the main trade.

        Returns {"sl_order_id": str|None, "tp_order_id": str|None}.
        """
        out: Dict[str, Optional[str]] = {"sl_order_id": None, "tp_order_id": None}

        if direction not in ("LONG", "SHORT"):
            logger.error(f"[PROTECTIVE_STOPS] {symbol}: invalid direction={direction}")
            return out
        if entry_price is None or entry_price <= 0:
            logger.error(f"[PROTECTIVE_STOPS] {symbol}: invalid entry_price={entry_price}")
            return out
        if sl_pct <= 0 or tp_pct <= 0:
            logger.warning(
                f"[PROTECTIVE_STOPS] {symbol}: sl_pct={sl_pct} tp_pct={tp_pct} — "
                f"non-positive values disable that leg"
            )

        # Close side is the opposite of the position direction
        close_side = 'sell' if direction == "LONG" else 'buy'

        # Compute trigger prices.  For LONG:  SL is BELOW entry,  TP is ABOVE.
        # For SHORT: SL is ABOVE entry, TP is BELOW.
        if direction == "LONG":
            sl_price = entry_price * (1 - sl_pct / 100.0)
            tp_price = entry_price * (1 + tp_pct / 100.0)
        else:
            sl_price = entry_price * (1 + sl_pct / 100.0)
            tp_price = entry_price * (1 - tp_pct / 100.0)

        try:
            await self.load_markets()
        except Exception as _e:
            logger.error(f"[PROTECTIVE_STOPS] {symbol}: load_markets failed ({_e}); skipping protective stops")
            return out

        # Round to Binance tick size so the exchange accepts the trigger prices
        try:
            _tick = await self.get_tick_size(symbol)
        except Exception:
            _tick = 0.0
        def _round(px: float) -> float:
            if not _tick or _tick <= 0:
                return px
            import math
            return round(math.floor(px / _tick) * _tick, 10) if direction == "LONG" else round(math.ceil(px / _tick) * _tick, 10)

        # Use MARK_PRICE so wicks / flash-crash false-triggers don't fire the stops.
        # IMPORTANT: we intentionally do NOT use closePosition=true here.
        # That flag routes STOP_MARKET / TAKE_PROFIT_MARKET through Binance's
        # Algo Order API endpoint which CCXT's create_order does not reliably
        # hit — observed in production with error -4120 "Order type not
        # supported for this endpoint. Please use the Algo Order API endpoints
        # instead." on both DOGE and BCH.  Using reduceOnly=true + explicit
        # quantity keeps us on the standard /fapi/v1/order endpoint with a
        # well-tested CCXT path.
        #
        # Trade-off: no auto-cancel when the position closes by other means.
        # Caller (trading_engine._close_position_locked) MUST call
        # cancel_protective_stops after a successful bot close to avoid
        # orphan orders triggering against future positions on the same pair.
        common_params = {
            'reduceOnly': True,       # Can only close, not flip direction
            'workingType': 'MARK_PRICE',
            'timeInForce': 'GTE_GTC',
        }

        # SL — STOP_MARKET
        if sl_pct > 0:
            try:
                _sl_trigger = _round(sl_price)
                sl_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side=close_side,
                    amount=quantity,
                    price=None,
                    params={**common_params, 'stopPrice': _sl_trigger},
                )
                out["sl_order_id"] = str(sl_order.get('id')) if sl_order and sl_order.get('id') else None
                logger.info(
                    f"[PROTECTIVE_STOPS] {symbol} {direction}: SL placed @ {_sl_trigger} "
                    f"(-{sl_pct}% from entry {entry_price}, qty={quantity}) id={out['sl_order_id']}"
                )
            except Exception as _e:
                logger.error(
                    f"[PROTECTIVE_STOPS] {symbol} {direction}: SL placement FAILED "
                    f"({str(_e)[:200]}) — position is NOT protected on broker side"
                )

        # TP — TAKE_PROFIT_MARKET
        if tp_pct > 0:
            try:
                _tp_trigger = _round(tp_price)
                tp_order = await self.exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=close_side,
                    amount=quantity,
                    price=None,
                    params={**common_params, 'stopPrice': _tp_trigger},
                )
                out["tp_order_id"] = str(tp_order.get('id')) if tp_order and tp_order.get('id') else None
                logger.info(
                    f"[PROTECTIVE_STOPS] {symbol} {direction}: TP placed @ {_tp_trigger} "
                    f"(+{tp_pct}% from entry {entry_price}, qty={quantity}) id={out['tp_order_id']}"
                )
            except Exception as _e:
                logger.error(
                    f"[PROTECTIVE_STOPS] {symbol} {direction}: TP placement FAILED "
                    f"({str(_e)[:200]}) — upside protection NOT in place"
                )

        return out

    async def cancel_protective_stops(
        self,
        symbol: str,
        sl_order_id: Optional[str] = None,
        tp_order_id: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Defensive cancel of protective orders.  Normally NOT needed because
        Binance auto-cancels closePosition=true orders when the position
        closes by other means.  Called on bot startup to clean up any orphans
        left by a crash mid-close, and as a belt-and-braces after manual close.

        Ignores Binance "unknown order id" errors — those are expected when
        Binance has already auto-cancelled.
        """
        out: Dict[str, bool] = {"sl_cancelled": False, "tp_cancelled": False}
        for label, oid, key in (("SL", sl_order_id, "sl_cancelled"), ("TP", tp_order_id, "tp_cancelled")):
            if not oid:
                continue
            try:
                await self.load_markets()
                await self.exchange.cancel_order(oid, symbol)
                out[key] = True
                logger.debug(f"[PROTECTIVE_STOPS] {symbol}: cancelled {label} order {oid}")
            except Exception as _e:
                _msg = str(_e).lower()
                # These error substrings mean the order was already gone — expected case
                _benign = any(s in _msg for s in (
                    'unknown order', 'does not exist', 'order does not exist',
                    '-2011', 'cancelrejected', 'order not exist'
                ))
                if _benign:
                    logger.debug(f"[PROTECTIVE_STOPS] {symbol}: {label} order {oid} already gone (auto-cancelled) — OK")
                    out[key] = True  # Treat as success: the order IS gone, which is what we wanted
                else:
                    logger.warning(f"[PROTECTIVE_STOPS] {symbol}: {label} cancel failed for {oid} ({_e})")
        return out
    
    async def get_open_positions(self) -> Optional[List[Dict]]:
        """Get all open positions from Binance. Returns None on API error (distinct from empty list)."""
        try:
            await self.load_markets()
            positions = await self.exchange.fetch_positions()
            
            open_positions = []
            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if contracts != 0:
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'side': 'LONG' if pos.get('side') == 'long' else 'SHORT',
                        'contracts': abs(contracts),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'mark_price': float(pos.get('markPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                        'leverage': int(pos.get('leverage') or 1),
                        'notional': float(pos.get('notional', 0)),
                        'margin': float(pos.get('initialMargin', 0))
                    })
            
            return open_positions
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching positions: {e}")
            return None

    async def get_position_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Lightweight check: fetch position for a single symbol.
        Returns dict with position info if open, None if no position or on error."""
        try:
            await self.load_markets()
            positions = await self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get('contracts', 0))
                if contracts != 0:
                    return {
                        'symbol': pos['symbol'],
                        'side': 'LONG' if pos.get('side') == 'long' else 'SHORT',
                        'contracts': abs(contracts),
                        'entry_price': float(pos.get('entryPrice', 0)),
                        'mark_price': float(pos.get('markPrice', 0)),
                        'unrealized_pnl': float(pos.get('unrealizedPnl', 0)),
                        'leverage': int(pos.get('leverage') or 1),
                    }
            return None
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching position for {symbol}: {e}")
            return None

    async def fetch_my_trades(self, symbol: str, limit: int = 5) -> Optional[List[Dict]]:
        """Fetch recent trades for a symbol from Binance.
        Returns list of trade dicts, or None on error."""
        try:
            await self.load_markets()
            trades = await self.exchange.fetch_my_trades(symbol, limit=limit)
            return [
                {
                    'price': float(t.get('price', 0)),
                    'amount': float(t.get('amount', 0)),
                    'cost': float(t.get('cost', 0)),
                    'side': t.get('side', ''),
                    'timestamp': t.get('timestamp'),
                    'datetime': t.get('datetime'),
                    'fee': t.get('fee', {}),
                }
                for t in trades
            ]
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching trades for {symbol}: {e}")
            return None


# Global service instance
binance_service = BinanceService()
