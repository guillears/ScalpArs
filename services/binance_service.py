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
        self._markets_loaded = False
        self._public_markets_loaded = False
    
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
        """If Binance has IP-banned us, sleep until the ban expires."""
        global _ban_until
        if _ban_until > 0:
            now = time.time()
            if now < _ban_until:
                wait = _ban_until - now + 2
                logger.warning(f"[BINANCE] IP banned, waiting {wait:.0f}s until ban expires")
                await asyncio.sleep(wait)
            _ban_until = 0

    @staticmethod
    def _detect_ban(error):
        """Check if an error is an IP ban and record the expiry."""
        global _ban_until
        match = re.search(r'banned until (\d+)', str(error))
        if match:
            _ban_until = int(match.group(1)) / 1000
            logger.error(f"[BINANCE] IP ban detected, expires at {_ban_until:.0f} (epoch)")

    async def close(self):
        """Close exchange connections"""
        await self.exchange.close()
        await self.public_exchange.close()
    
    async def get_balance(self) -> Dict:
        """Get account balance"""
        try:
            await self.load_markets()
            balance = await self.exchange.fetch_balance()
            
            usdt_balance = balance.get('USDT', {})
            bnb_balance = balance.get('BNB', {})
            
            return {
                'usdt_free': float(usdt_balance.get('free', 0)),
                'usdt_used': float(usdt_balance.get('used', 0)),
                'usdt_total': float(usdt_balance.get('total', 0)),
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
    
    async def get_top_futures_pairs(self, limit: int = 20) -> List[Dict]:
        """Get top futures pairs by 24h volume"""
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
            
            # Sort by volume descending
            futures_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
            
            return futures_pairs[:limit]
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching top pairs: {e}", exc_info=True)
            return []
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> List:
        """Get OHLCV data for indicator calculation"""
        await self._check_ban()
        await self.load_public_markets()
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
    
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol"""
        try:
            await self.load_markets()
            await self.exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.error(f"[BINANCE] Error setting leverage for {symbol}: {e}")
            return False
    
    async def create_market_order(
        self, 
        symbol: str, 
        side: str,  # 'buy' or 'sell'
        amount: float,
        leverage: int = 1
    ) -> Optional[Dict]:
        """Create a market order"""
        try:
            await self.load_markets()
            
            # Set leverage first
            await self.set_leverage(symbol, leverage)
            
            # Create order
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount
            )
            
            return {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': float(order.get('amount', amount)),
                'price': float(order.get('average', order.get('price', 0))),
                'cost': float(order.get('cost', 0)),
                'fee': float(order.get('fee', {}).get('cost', 0)),
                'timestamp': order.get('timestamp', datetime.now().timestamp() * 1000)
            }
        except Exception as e:
            logger.error(f"[BINANCE] Error creating order for {symbol}: {e}")
            return None
    
    async def close_position(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Close a position"""
        # To close a LONG, we sell. To close a SHORT, we buy.
        close_side = 'sell' if side == 'LONG' else 'buy'
        return await self.create_market_order(symbol, close_side, amount)
    
    async def get_open_positions(self) -> List[Dict]:
        """Get all open positions from Binance"""
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
                        'leverage': int(pos.get('leverage', 1)),
                        'notional': float(pos.get('notional', 0)),
                        'margin': float(pos.get('initialMargin', 0))
                    })
            
            return open_positions
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching positions: {e}")
            return []


# Global service instance
binance_service = BinanceService()
