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

_leverage_blocked_pairs: dict = {}  # symbol -> blocked_at epoch. Sep-7 full-review: was a set with no
# expiry — one transient set_leverage failure benched a pair until process restart. TTL below.
_LEVERAGE_BLOCK_TTL_S = 3600.0


def is_leverage_blocked(symbol: str) -> bool:
    """TTL-aware check (Sep-7): a block older than 1h expires — a REAL config/exchange mismatch
    re-blocks on the next attempt within seconds, so expiry only costs one retried call."""
    _ts = _leverage_blocked_pairs.get(symbol)
    if _ts is None:
        return False
    if time.time() - _ts > _LEVERAGE_BLOCK_TTL_S:
        _leverage_blocked_pairs.pop(symbol, None)
        logger.info(f"[LEVERAGE_BLOCK] {symbol}: 1h TTL expired — pair eligible again")
        return False
    return True


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


def _range_24h_pct(high, low) -> float:
    """24 h low→high swing in % from a ccxt ticker's high / low; 0.0 when either is missing or not positive (never raises)."""
    try:
        h, l = float(high), float(low)
        return (h / l - 1) * 100 if h > 0 and l > 0 and h >= l else 0.0
    except (TypeError, ValueError):
        return 0.0



OHLCV_CCXT_CAP = 1000   # 🔧 Oct-6 (DECISION_LOG 230): newer ccxt caps fetch_ohlcv here; larger reads go to the raw klines endpoint

class BinanceService:
    """Service for interacting with Binance Futures API"""
    
    def __init__(self):
        # 📖 Oct-3: research reads (order-book snapshots) get their OWN client and throttle — never queued in front of orders (trading
        # client) nor of the scan / FRENZY kline reads (public client). Created lazily on first use.
        self.research_exchange = None
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
        self._commission_cache = None  # (monotonic_ts, rates) -- fee auto-sync (Sep-3)
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
        # Funding rate cache (Apr 28, Exploration Analytics): {symbol: (rate, fetched_at_unix_seconds)}
        # Funding intervals are 8h on Binance, so caching for ~8h saves API calls.
        self._funding_rate_cache: Dict[str, Tuple[float, float]] = {}
        self._funding_rate_ttl_seconds = 8 * 60 * 60  # 8 hours

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

    async def get_capital_flows(self, start_ms: int):
        """Aug-27 flow-reconciliation alarm: TRUE external flows (on-chain/fiat) from the
        capital endpoints — deposits (status 1=success? Binance: 1=success for hisrec... we
        keep completed only) and withdrawals (status 6=completed). These land SPOT-side; the
        futures TRANSFER feed is the primary ledger — this is the cross-check. Returns
        [(ts_ms, +amount|for deposits / -amount|for withdrawals, coin)]. 10-min dict cache.
        Fail-open: raises to caller."""
        import time as _t
        _cache = getattr(self, '_cap_flow_cache', None) or {}
        _hit = _cache.get(start_ms)
        if _hit and (_t.monotonic() - _hit[0]) < 600:
            return _hit[1]
        out = []
        _now_ms = int(_t.time() * 1000)
        # capital endpoints cap the window at 90d — clamp
        _s = max(start_ms, _now_ms - 89 * 86400000)
        deps = await self.spot_exchange.sapiGetCapitalDepositHisrec({'startTime': _s, 'endTime': _now_ms, 'limit': 1000})
        for r in deps or []:
            if str(r.get('status')) in ('1', '6'):  # review: 6 = credited-but-cannot-withdraw is already usable capital
                out.append((int(r.get('insertTime') or 0), abs(float(r.get('amount') or 0)), r.get('coin')))
        wds = await self.spot_exchange.sapiGetCapitalWithdrawHistory({'startTime': _s, 'endTime': _now_ms, 'limit': 1000})
        for r in wds or []:
            if str(r.get('status')) == '6':
                _ts = r.get('completeTime') or r.get('applyTime')
                try:
                    _tms = int(_ts) if str(_ts).isdigit() else int(__import__('datetime').datetime.fromisoformat(str(_ts)).replace(tzinfo=__import__('datetime').timezone.utc).timestamp() * 1000)
                except Exception:
                    _tms = 0
                out.append((_tms, -abs(float(r.get('amount') or 0)), r.get('coin')))
        _cache[start_ms] = (_t.monotonic(), out)
        if len(_cache) > 8:  # Sep-7 hygiene: keys accumulate as the era window shifts — keep newest
            for _k in sorted(_cache, key=lambda k: _cache[k][0])[:-8]:
                _cache.pop(_k, None)
        self._cap_flow_cache = _cache
        return out

    async def get_transfer_rows(self, start_ms: int):
        """Aug-27 phase-2: raw external USDT transfer rows [(ts_ms, amount)] since start_ms
        (deposits +, withdrawals −). 10-min cache keyed on start_ms."""
        import time as _t
        _cache = getattr(self, '_tx_rows_cache', None) or {}
        _hit = _cache.get(start_ms)
        if _hit and (_t.monotonic() - _hit[0]) < 600:
            return _hit[1]
        out = []
        cursor = start_ms
        for _page in range(10):
            rows = await self.exchange.fapiPrivateGetIncome({
                'incomeType': 'TRANSFER', 'startTime': int(cursor),
                'endTime': int(_t.time() * 1000), 'limit': 1000})
            if not rows:
                break
            for r in rows:
                if r.get('asset') == 'USDT':
                    out.append((int(r.get('time') or 0), float(r.get('income') or 0), str(r.get('tranId') or '')))
            if len(rows) < 1000:
                break
            # Sep-7 hygiene: page on the LAST timestamp itself (not +1) so same-millisecond rows
            # split across a page boundary are not dropped; the overlap dedups below.
            _next = int(rows[-1].get('time') or 0)
            if _next <= cursor:  # no forward progress (pathological) — stop rather than loop
                break
            cursor = _next
        _seen = set()
        # Sep-7 review: dedup on tranId (unique per income row) so two GENUINE same-ms same-amount
        # transfers survive; rows lacking tranId fall back to the (ts, amount) identity.
        out = [r for r in out if not ((r[2] or r[:2]) in _seen or _seen.add(r[2] or r[:2]))]
        out = [(t, a) for t, a, _ in out]  # callers consume (ts, amount)
        _cache[start_ms] = (_t.monotonic(), out)
        if len(_cache) > 8:  # Sep-7 hygiene: same cap as _cap_flow_cache
            for _k in sorted(_cache, key=lambda k: _cache[k][0])[:-8]:
                _cache.pop(_k, None)
        self._tx_rows_cache = _cache
        return out

    async def get_spot_balance_usd(self) -> Optional[Dict]:
        """Spot wallet snapshot in USD (Aug-24, operator: Futures/Spot split on the portfolio card
        to catch stranded-funds errors like the $77.64 BNB-swap incident). 60s TTL cache — the UI
        polls /api/balance every few seconds; spot balances only move during swaps. Fail-open None."""
        try:
            import time as _t
            _c = getattr(self, '_spot_bal_cache', None)
            if _c and (_t.monotonic() - _c[0]) < 60:
                return _c[1]
            await self._load_spot_markets()
            bal = await self.spot_exchange.fetch_balance()
            usdt = float((bal.get('USDT') or {}).get('total') or 0)
            bnb = float((bal.get('BNB') or {}).get('total') or 0)
            bnb_usd = 0.0
            if bnb > 0:
                px = await self.get_bnb_price()
                bnb_usd = bnb * px if px > 0 else 0.0
            out = {'usdt': round(usdt, 2), 'bnb': bnb, 'bnb_usd': round(bnb_usd, 2),
                   'total': round(usdt + bnb_usd, 2)}
            self._spot_bal_cache = (_t.monotonic(), out)
            return out
        except Exception as e:
            logger.warning(f"[SPOT_BAL] fetch failed: {e}")
            return None

    async def buy_bnb(self, amount_usdt: float) -> Optional[Dict]:
        """Buy BNB with USDT via transfer-to-spot + spot market buy + transfer-back.
        Four steps; all non-time-sensitive, no expiring quotes."""
        try:
            await self.load_markets()
            await self._load_spot_markets()

            # Aug-24 live incident: the raw float shortfall (77.637613796537) TRANSFERRED fine
            # (ccxt truncates), crediting spot 77.6376 — but step 2's buy asked round(...,2)=77.64,
            # more than spot held → 'insufficient balance' AFTER the debit. Floor to 2 dp up front
            # so the transfer and the buy use the identical amount.
            amount_usdt = int(amount_usdt * 100) / 100.0
            # Sep-7 full-review (2 reviewers): spot BNB/USDT min order is ~$10 — a $5-10 request
            # previously TRANSFERRED to spot (step 1) then the buy rejected, stranding the USDT.
            # Refuse BEFORE any money moves (exact mirror of the Aug-24 sell-path check).
            try:
                _bm = self.spot_exchange.market('BNB/USDT')
                _min_cost = float(((_bm.get('limits') or {}).get('cost') or {}).get('min') or 10.0)
            except Exception:
                _min_cost = 10.0
            _floor = max(1.0, _min_cost)
            if amount_usdt < _floor:
                logger.warning(f"[BNB_SWAP] Refused: ${amount_usdt:.2f} is below the spot minimum order of ${_floor:.0f} — nothing transferred")
                return None

            # Step 1: Transfer USDT from futures wallet to spot wallet
            logger.info(f"[BNB_SWAP] Step 1/4: Transferring {amount_usdt} USDT futures → spot")
            await self.spot_exchange.transfer('USDT', amount_usdt, 'future', 'spot')

            # Step 2: Buy BNB on spot using quoteOrderQty (spend exact USDT amount)
            logger.info(f"[BNB_SWAP] Step 2/4: Buying BNB with {amount_usdt} USDT on spot")
            try:
                order = await self.spot_exchange.create_order(
                    'BNB/USDT', 'market', 'buy', None, None,
                    {'quoteOrderQty': round(amount_usdt, 2)}
                )
            except Exception as _buy_err:
                # Sep-7 full-review: the USDT is already ON SPOT — roll it back instead of
                # stranding it silently (sell path got this rollback Aug-24; buy was missed).
                logger.error(f"[BNB_SWAP] Spot buy failed ({_buy_err}) — rolling {amount_usdt} USDT back to futures")
                try:
                    await self.spot_exchange.transfer('USDT', amount_usdt, 'spot', 'future')
                    logger.info(f"[BNB_SWAP] Rollback complete: {amount_usdt} USDT returned to futures")
                except Exception as _rb_err:
                    logger.error(f"[BNB_SWAP] ROLLBACK FAILED ({_rb_err}) — {amount_usdt} USDT STRANDED ON SPOT; manual sweep needed (a later successful buy also sweeps it)")
                return None

            avg_price = float(order.get('average') or order.get('price') or 0)
            cost = float(order.get('cost') or amount_usdt)
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

    async def sell_bnb(self, amount_usdt: float) -> Optional[Dict]:
        """Sell BNB for USDT via transfer-to-spot + spot market sell + transfer-back.
        Mirror of buy_bnb (May 25). Spec:
        - Input: amount_usdt = approximate USDT value of BNB to sell.
        - Convert to BNB quantity at current price.
        - Transfer that BNB from futures wallet → spot wallet.
        - Market-sell on BNB/USDT spot pair.
        - Transfer received USDT spot → futures.
        - Return any leftover BNB (lot-size remainder) back to futures.
        """
        try:
            await self.load_markets()
            await self._load_spot_markets()

            # Step 0: Convert USDT amount to BNB units at current price
            bnb_price = await self.get_bnb_price()
            if bnb_price <= 0:
                logger.error("[BNB_SWAP_SELL] Failed: could not get BNB price")
                return None
            bnb_to_sell = round(amount_usdt / bnb_price, 4)
            if bnb_to_sell <= 0:
                logger.error(f"[BNB_SWAP_SELL] Computed BNB qty too small: {bnb_to_sell} at price {bnb_price}")
                return None

            # Aug-24 fix (live -1013 NOTIONAL + stranded funds): check the spot min-notional BEFORE any transfer.
            try:
                _bm = self.spot_exchange.market('BNB/USDT')
                _min_cost = float(((_bm.get('limits') or {}).get('cost') or {}).get('min') or 10.0)
            except Exception:
                _min_cost = 10.0
            if amount_usdt < _min_cost:
                logger.error(f"[BNB_SWAP_SELL] Refused: ${amount_usdt:.2f} is below the spot minimum order of ${_min_cost:.0f} — nothing transferred. Enter the USD value to sell (≥ ${_min_cost:.0f}).")
                return None

            # Sweep any BNB already stranded on spot (e.g. from a previously failed sell) into this sell.
            try:
                _sb = await self.spot_exchange.fetch_balance()
                _stranded = float((_sb.get('BNB') or {}).get('free') or 0)
                if _stranded > 0.0005:
                    logger.info(f"[BNB_SWAP_SELL] Sweeping {_stranded} stranded spot BNB into this sell")
                    bnb_to_sell = round(bnb_to_sell, 4)
            except Exception:
                _stranded = 0.0

            # Step 1: Transfer BNB from futures wallet to spot wallet
            bnb_to_sell = int(bnb_to_sell * 1e8 + 0.5) / 1e8  # review: half-up to 8 dp — a pure floor can land 1e-8 below the 4-dp value while _sell_qty rounds back up over the transferred balance
            logger.info(f"[BNB_SWAP_SELL] Step 1/4: Transferring {bnb_to_sell} BNB futures → spot")
            await self.spot_exchange.transfer('BNB', bnb_to_sell, 'future', 'spot')

            # Step 2: Sell BNB on spot for USDT (market order) — on failure, transfer the BNB BACK (no stranding)
            _sell_qty = round(bnb_to_sell + (_stranded if _stranded > 0.0005 else 0.0), 4)
            logger.info(f"[BNB_SWAP_SELL] Step 2/4: Selling {_sell_qty} BNB on spot")
            try:
                order = await self.spot_exchange.create_order(
                    'BNB/USDT', 'market', 'sell', _sell_qty, None
                )
            except Exception as _sell_err:
                logger.error(f"[BNB_SWAP_SELL] Spot sell failed ({_sell_err}) — transferring {bnb_to_sell} BNB back to futures")
                try:
                    await self.spot_exchange.transfer('BNB', bnb_to_sell, 'spot', 'future')
                    logger.info("[BNB_SWAP_SELL] Rollback transfer completed — no stranded funds")
                except Exception as _rb_err:
                    logger.critical(f"[BNB_SWAP_SELL] ROLLBACK FAILED ({_rb_err}) — {bnb_to_sell} BNB stranded on SPOT; move it back manually")
                return None

            avg_price = float(order.get('average') or order.get('price') or 0)  # Sep-7: 'average': None must not TypeError post-trade
            cost = float(order.get('cost') or 0)  # USDT received
            order_id = order.get('id', 'spot_sell')

            # Step 3: Fetch actual spot balances and transfer USDT back to futures
            spot_bal = await self.spot_exchange.fetch_balance()
            actual_usdt = float(spot_bal.get('USDT', {}).get('free', 0))
            if actual_usdt <= 0:
                logger.error("[BNB_SWAP_SELL] No USDT on spot after sell. Check order status.")
                return None

            if avg_price <= 0 and actual_usdt > 0 and bnb_to_sell > 0:
                avg_price = actual_usdt / bnb_to_sell

            logger.info(f"[BNB_SWAP_SELL] Step 3/4: Transferring {actual_usdt:.2f} USDT spot → futures")
            await self.spot_exchange.transfer('USDT', actual_usdt, 'spot', 'future')

            # Step 4: Return any leftover BNB (lot-size rounding remainder) back to futures
            leftover_bnb = float(spot_bal.get('BNB', {}).get('free', 0))
            if leftover_bnb > 0.0001:
                logger.info(f"[BNB_SWAP_SELL] Step 4/4: Returning {leftover_bnb} leftover BNB spot → futures")
                await self.spot_exchange.transfer('BNB', leftover_bnb, 'spot', 'future')

            logger.info(f"[BNB_SWAP_SELL] Complete: {bnb_to_sell} BNB → {actual_usdt:.2f} USDT @ {avg_price:.2f}")
            return {
                'bnb_amount': bnb_to_sell,
                'price': avg_price,
                'proceeds_usdt': actual_usdt,
                'order_id': str(order_id)
            }
        except Exception as e:
            logger.error(f"[BNB_SWAP_SELL] Failed: {e}. If transfer already happened, check spot wallet for stranded funds.")
            return None

    async def close(self):
        """Close exchange connections"""
        await self.exchange.close()
        await self.public_exchange.close()
        await self.spot_exchange.close()
        if getattr(self, 'research_exchange', None) is not None:
            await self.research_exchange.close()
    
    @property
    def _last_balance_payload(self):
        """Last successful get_balance payload if fetched ≤5 s ago, else None (status-poll reuse)."""
        p = getattr(self, '_last_balance_payload_raw', None)
        return p if (p and time.time() - p.get('_ts', 0) <= 5.0) else None

    @_last_balance_payload.setter
    def _last_balance_payload(self, v):
        self._last_balance_payload_raw = v

    def invalidate_flow_caches(self):
        """Sep-7 review: called by the fund deposit/withdraw endpoints so a just-registered
        flow is visible to the return metrics immediately (not after the 10-min TTL)."""
        self._tx_rows_cache = {}
        self._cap_flow_cache = {}

    async def get_commission_rates(self) -> Dict:
        """Sep-3 (operator): the account's ACTUAL futures fee rates + BNB-discount state.
        /fapi/v1/commissionRate returns the BASE tier rate (BNB discount NOT included);
        /fapi/v1/feeBurn says whether the futures 10% pay-in-BNB discount is switched on.
        Effective rate = base x 0.9 when feeBurn AND BNB fuel exists (caller's job).
        Raises to caller (fail-open there). 1h cache -- rates change at most daily."""
        _now = time.monotonic()
        if self._commission_cache and _now - self._commission_cache[0] < 3600:
            return self._commission_cache[1]
        cr = await self.exchange.fapiPrivateGetCommissionRate({'symbol': 'BTCUSDT'})
        fb = await self.exchange.fapiPrivateGetFeeBurn()
        out = {
            'maker': float(cr.get('makerCommissionRate') or 0),
            'taker': float(cr.get('takerCommissionRate') or 0),
            'fee_burn': str(fb.get('feeBurn')).lower() == 'true',
        }
        if out['maker'] <= 0 or out['taker'] <= 0:
            raise ValueError(f"commissionRate returned non-positive rates: {cr}")
        self._commission_cache = (_now, out)
        return out

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
            
            # Aug-22: Binance "Margin Ratio" inputs (cross wallet): totalMaintMargin / totalMarginBalance
            # (liquidation at 100%). Surfaced as-is; the dashboard computes the ratio.
            try:
                _maint = float(raw_info.get('totalMaintMargin', 0) or 0)
                _mbal = float(raw_info.get('totalMarginBalance', 0) or 0)
            except Exception:
                _maint, _mbal = 0.0, 0.0
            _payload = {
                'ok': True,
                'usdt_free': usdt_free,
                'usdt_used': float(usdt_balance.get('used', 0)),
                'usdt_total': usdt_wallet,
                'bnb_free': float(bnb_balance.get('free', 0)),
                'bnb_total': float(bnb_balance.get('total', 0)),
                'total_portfolio': float(balance.get('total', {}).get('USDT', 0)),
                'maint_margin': _maint,
                'margin_balance': _mbal,
            }
            # Aug-22 review: short-lived copy for same-poll consumers (status margin ratio) — 5 s TTL
            self._last_balance_payload = dict(_payload, _ts=time.time())
            return _payload
        except Exception as e:
            logger.error(f"[BINANCE] Error fetching balance: {e}")
            # 'ok': False lets callers distinguish a FAILED fetch from a genuinely
            # empty account — zeros must never be persisted as NAV or used to price
            # share issuance (Jul 2 review C1).
            return {
                'ok': False,
                'usdt_free': 0,
                'usdt_used': 0,
                'usdt_total': 0,
                'bnb_free': 0,
                'bnb_total': 0,
                'total_portfolio': 0,
                'maint_margin': 0,
                'margin_balance': 0,
            }
    
    async def get_top_futures_pairs(
        self,
        limit: int = 20,
        new_listing_filter_days: int = 0,
        alpha_subtype_filter_enabled: bool = False,
        coin_underlying_only: bool = False,
    ) -> List[Dict]:
        """Get top USDT-perpetual futures pairs by 24h volume.

        When ``new_listing_filter_days`` > 0, pairs whose Binance onboardDate
        (from exchangeInfo, surfaced via CCXT ``markets[symbol]['info']``) is
        within the last N days are excluded *before* the top-N-by-volume cut.
        This filters out Binance's Seed Tag / Monitoring Tag pairs — low
        liquidity, manipulation-prone, poor fit for 5m-EMA strategy.  See
        CLAUDE.md Apr 17 analysis for the RAVEUSDT blow-up that motivated this.

        When ``alpha_subtype_filter_enabled`` is True, pairs whose Binance
        ``underlyingSubType`` contains "Alpha" are excluded.  This is Binance's
        launchpad / Innovation Zone tier — pairs that carry the "high
        volatility" UI warning regardless of listing age.  Catches pairs like
        LABUSDT and RAVEUSDT proactively before they hit the bot.  See
        CLAUDE.md May 5 entry on Alpha subtype filter.
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
                        'change_24h': float(percentage) if percentage is not None else 0.0,
                        # 🔥 Oct-3: 24 h low→high swing (%) — FRENZY's shortlist also admits pairs that dumped then pumped back (MOVR: change −0.6 %, range 28 %)
                        'range_24h': _range_24h_pct(ticker.get('high'), ticker.get('low'))
                    })

            # New-listing filter: drop pairs listed within the last N days,
            # based on Binance's onboardDate in market metadata.  Applied
            # BEFORE the top-N-by-volume cut so "top 50" stays "top 50 of
            # eligible pairs."  Fails open: pairs without a parseable
            # onboardDate are kept (conservative — don't accidentally block
            # established pairs due to missing metadata).
            # Jul 14: crypto-only universe — allowlist Binance `underlyingType == "COIN"`.
            # ONE condition excludes tokenized stocks (MU/INTC/SPY/NVDA...), commodities
            # (NATGAS/COPPER/XPT), indexes and leveraged ETFs — 132+ EQUITY/TradFi perps —
            # BEFORE the top-N cut. Signals are calibrated to crypto microstructure; equity
            # perps trade synthetically while the underlying market is closed (MU 07-14 short,
            # our first-ever equity trade, fired at 03:26 UTC). Fail-open: missing/unknown
            # underlyingType keeps the pair (never drop an established crypto on a metadata gap).
            if coin_underlying_only:
                _cu_mkts = self.public_exchange.markets or {}
                _cu_before = len(futures_pairs)
                _cu_kept, _cu_dropped = [], []
                for p in futures_pairs:
                    _ut = (((_cu_mkts.get(p['symbol'], {}) or {}).get('info', {}) or {}).get('underlyingType'))
                    if _ut is None or _ut == 'COIN':
                        _cu_kept.append(p)
                    else:
                        _cu_dropped.append(p['pair'])
                futures_pairs = _cu_kept
                if _cu_dropped:
                    _cu_prev = ', '.join(sorted(_cu_dropped)[:8])
                    _cu_extra = f" +{len(_cu_dropped) - 8} more" if len(_cu_dropped) > 8 else ""
                    logger.info(
                        f"[BINANCE] Crypto-only filter: excluded {len(_cu_dropped)}/{_cu_before} "
                        f"non-COIN perps ({_cu_prev}{_cu_extra})"
                    )

            # Jul 13: stamp listing age (days since Binance onboardDate) on EVERY pair —
            # threaded scan→order as entry_pair_age_days (read gate for the 180→90-day
            # new-listing step-down). None when metadata is missing (fail-open, like the filter).
            import time as _time
            _now_ms = _time.time() * 1000
            _age_mkts = self.public_exchange.markets or {}
            for p in futures_pairs:
                try:
                    _ob = ((_age_mkts.get(p['symbol'], {}) or {}).get('info', {}) or {}).get('onboardDate')
                    p['age_days'] = round((_now_ms - int(_ob)) / 86400000.0, 1) if _ob is not None else None
                except (ValueError, TypeError):
                    p['age_days'] = None

            if new_listing_filter_days > 0:
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

            # Alpha-subtype filter (May 5, 2026): drop pairs whose Binance
            # `underlyingSubType` contains "Alpha".  This is the launchpad /
            # Innovation Zone tier — pairs flagged with the "high volatility"
            # UI warning, elevated triggerProtect (0.15 vs 0.05 for liquid
            # pairs), and historically the "never-positive + emergency-SL"
            # failure pattern.  Runs alongside the new-listing filter; together
            # they catch different failure modes.  Fails open when subtype
            # metadata is missing.
            if alpha_subtype_filter_enabled:
                markets = self.public_exchange.markets or {}
                before_count_alpha = len(futures_pairs)
                kept_pairs = []
                alpha_excluded = []
                for p in futures_pairs:
                    market = markets.get(p['symbol'], {})
                    info = market.get('info', {}) if isinstance(market, dict) else {}
                    subtype = info.get('underlyingSubType')
                    # Be defensive: subtype may be a list, a string, or None.
                    is_alpha = False
                    if isinstance(subtype, list):
                        is_alpha = any(s == "Alpha" for s in subtype)
                    elif isinstance(subtype, str):
                        is_alpha = subtype == "Alpha"
                    if is_alpha:
                        alpha_excluded.append(p['pair'])
                    else:
                        kept_pairs.append(p)
                futures_pairs = kept_pairs
                if alpha_excluded:
                    _preview = ', '.join(sorted(alpha_excluded)[:8])
                    _extra = f" +{len(alpha_excluded) - 8} more" if len(alpha_excluded) > 8 else ""
                    logger.info(
                        f"[BINANCE] Alpha-subtype filter: "
                        f"excluded {len(alpha_excluded)}/{before_count_alpha} pairs "
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
                if limit and int(limit) > OHLCV_CCXT_CAP:
                    # 🔧 Oct-6 (DECISION_LOG 230): ccxt ≥ 4.5.40 caps fetch_ohlcv at 1000 bars although Binance futures serves up to 1500, so
                    # the live FRENZY window was 999 closed bars while every study used 1499 (a spike > ~58 h old dropped out of "verified").
                    # Read the raw endpoint directly — same rows ([open ms, o, h, l, c, v]), independent of the installed ccxt version.
                    raw = await self.public_exchange.fapiPublicGetKlines({
                        'symbol': self.public_exchange.market(symbol)['id'], 'interval': timeframe, 'limit': min(int(limit), 1500)})
                    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in raw or []]
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
    
    async def get_leverage_brackets(self) -> Dict[str, list]:
        """🪜 Oct-1: the exchange's leverage brackets per pair → {"MOVRUSDT": [(notional_cap, max_leverage), …]} sorted by cap
        (e.g. MOVR: up to $5k at 25×, then lower leverage for bigger positions). Account-level endpoint (needs the API keys);
        cached 6 h, a failure keeps the last table and is retried after 10 min. {} when unavailable (no keys / never fetched)
        — callers then apply NO bracket cap. Called from the order path and the status poll, so it NEVER sleeps out a ban,
        is single-flight (concurrent callers get the last table at once) and bounded at 6 s. Never raises."""
        now = time.monotonic(); c = getattr(self, '_lev_brackets', None) or {"at": -1e18, "ok": False, "data": {}}
        if now - c["at"] < (6 * 3600 if c["ok"] else 600):
            return c["data"]
        # stamp BEFORE any await: the next 10 min of callers return the last table instead of queueing behind this fetch
        self._lev_brackets = {"at": now, "ok": False, "data": c["data"]}
        if not (settings.binance_api_key and settings.binance_api_secret):
            if not getattr(self, '_lev_brackets_nokey_logged', False):
                self._lev_brackets_nokey_logged = True
                logger.info("[BRACKETS] no exchange API keys — leverage brackets unavailable, no bracket cap is applied")
            return c["data"]
        if _ban_until > time.time():                                    # a ban is active: do not call, do not wait
            return c["data"]
        try:
            async def _fetch():
                await self.load_markets()
                return await self.exchange.fetch_leverage_tiers()
            raw = await asyncio.wait_for(_fetch(), timeout=6.0)
            out = {}
            for sym, tiers in (raw or {}).items():
                if not str(sym).endswith(":USDT"):
                    continue
                rows = []
                for t in tiers or []:
                    cap, lev = t.get("maxNotional"), t.get("maxLeverage")
                    if cap and lev and float(cap) > 0 and float(lev) >= 1:
                        rows.append((float(cap), float(lev)))
                if rows:
                    out[str(sym).split("/")[0] + "USDT"] = sorted(rows)
            if not out:
                raise RuntimeError("empty bracket table")
            self._lev_brackets = {"at": now, "ok": True, "data": out}
            logger.info(f"[BRACKETS] leverage brackets loaded for {len(out)} pairs")
            return out
        except Exception as e:
            try:
                self._detect_ban(e)
            except Exception:
                pass
            logger.warning(f"[BRACKETS] leverage bracket fetch failed ({str(e)[:80] or type(e).__name__}) — keeping the last table ({len(c['data'])} pairs), retry in 10 min")
            return c["data"]

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
        is_close: bool = False,
        status: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Create a market order. `status` (optional dict): status['sent'] is set True right before the order request leaves, so
        a caller that gets None back can tell "never sent" (refused before the request) from "sent, outcome unknown"."""
        try:
            await self._check_ban()  # Aug-24 M3: private calls respect an active IP ban (close_position delegates here too)
            await self.load_markets()

            if not is_close:
                actual_leverage = await self.set_leverage(symbol, leverage)
                if actual_leverage == 0:
                    logger.error(f"[LEVERAGE_MISMATCH] {symbol}: Cannot determine leverage, skipping order")
                    _leverage_blocked_pairs[symbol] = time.time()
                    return None
                if actual_leverage != leverage:
                    logger.warning(f"[LEVERAGE_MISMATCH] {symbol}: Binance leverage {actual_leverage}x != configured {leverage}x — blocking pair")
                    _leverage_blocked_pairs[symbol] = time.time()
                    return None
            
            # Aug-24 M5: client-side precision + MIN_NOTIONAL check (entries only) — a -1013/-4164 reject
            # used to surface as a silent None (the flip-bug class). Rounded amount, then min-cost gate.
            try:
                _amt_p = float(self.exchange.amount_to_precision(symbol, amount))
                if _amt_p > 0:
                    amount = _amt_p
                if not is_close:
                    _mkt = self.exchange.market(symbol)
                    _min_amt = ((_mkt.get('limits') or {}).get('amount') or {}).get('min')
                    _min_cost = ((_mkt.get('limits') or {}).get('cost') or {}).get('min') or 5.0
                    if _min_amt and amount < float(_min_amt):
                        logger.error(f"[MIN_NOTIONAL_BLOCK] {symbol}: amount {amount} < exchange min {_min_amt} — order not sent")
                        return None
                    _tk = await self.exchange.fetch_ticker(symbol)
                    _last = float(_tk.get('last') or 0)
                    if _last > 0 and amount * _last < float(_min_cost):
                        logger.error(f"[MIN_NOTIONAL_BLOCK] {symbol}: notional {amount * _last:.2f} < min {_min_cost} — order not sent")
                        return None
            except Exception as _m5e:
                logger.warning(f"[MIN_NOTIONAL_CHECK] {symbol}: pre-check failed ({_m5e}) — proceeding (exchange will validate)")
            # reduceOnly prevents position flip on close orders
            params = {'reduceOnly': True} if is_close else {}
            if status is not None:
                status['sent'] = True
            order = await self.exchange.create_order(
                symbol=symbol,
                type='market',
                side=side,
                amount=amount,
                params=params
            )
            
            # Aug-24 LIVE M2 fix: a market-order ACK can lack average/price → a 0 used to stamp through and the
            # realtime SL then SKIPS the position (entry_price<=0 guard) = live position with NO stop (GRASS incident).
            _px = float(order.get('average') or order.get('price') or 0)
            _filled_ref = None   # the filled size from a refetch (the first ACK often reports 0 before the fill is in)
            if _px <= 0:
                # Aug-25: RETRY the refetch (3x, 0.4s apart) — the single instant refetch kept
                # finding avgPrice still 0 (fill not yet reported) and control fell through to
                # the ticker/WS fallbacks, booking the DECISION price as the fill (BTC id 3:
                # exit off by $35/0.044%, slip recorded 0.0). Fills report within ~1s.
                for _m2_try in range(3):
                    try:
                        if _m2_try:
                            await asyncio.sleep(0.4)
                        _ref = await self.exchange.fetch_order(order['id'], symbol)
                        _px = float((_ref or {}).get('average') or (_ref or {}).get('price') or 0)
                        if (_ref or {}).get('filled'):
                            _filled_ref = _ref.get('filled')
                        if _px > 0:
                            break
                    except Exception as _re:
                        logger.warning(f"[BINANCE] fill-price refetch failed for {symbol} (try {_m2_try+1}/3): {_re}")
            if _px <= 0:
                try:
                    _tk = await self.exchange.fetch_ticker(symbol)
                    _px = float(_tk.get('last') or 0)
                    logger.critical(f"[BINANCE] {symbol}: fill price unavailable from order+refetch — stamped TICKER last {_px} (M2 fallback)")
                except Exception:
                    pass
            if _px <= 0:  # review M2: both fallbacks failed — never be silent (the [M2_REPAIR] scan pass is the last net)
                logger.critical(f"[BINANCE] {symbol}: fill price STILL 0 after refetch + ticker — order {order.get('id')} returns price 0; scan repair must fix it")
            return {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': float(order.get('amount') or amount),
                'filled': float(_filled_ref or order.get('filled') or 0),   # 0 = not reported (callers fall back to 'amount')
                'price': _px,
                'cost': float(order.get('cost') or 0),
                'fee': float((order.get('fee') or {}).get('cost') or 0),
                'timestamp': order.get('timestamp', datetime.now().timestamp() * 1000)
            }
        except Exception as e:
            logger.error(f"[BINANCE] Error creating order for {symbol}: {e}")
            return None
    
    # ── Aug-11 🛡 BROKER BACKSTOP (Algo Order API) ────────────────────────────
    # -4120 ROOT CAUSE SOLVED: Binance MANDATORILY migrated conditional orders
    # (STOP_MARKET / TAKE_PROFIT_MARKET / TRAILING_STOP_MARKET) from /fapi/v1/order
    # to POST /fapi/v1/algoOrder (algoType=CONDITIONAL), account waves through
    # 2025-12-09 (error -4120 = STOP_ORDER_SWITCH_ALGO). The Apr-17 "removal after
    # 4 failed hotfixes" retried params on the deprecated endpoint — never an
    # account defect. closePosition=true makes these orders structurally incapable
    # of creating exposure (flatten-only). DECISION_LOG 2026-08-11 (6).
    async def place_backstop_stop(self, pair: str, direction: str, trigger_price: float):
        """Place a resting CONDITIONAL STOP_MARKET (close-all). Returns algoId or None."""
        try:
            await self._check_ban()
            side = 'SELL' if direction == 'LONG' else 'BUY'
            try:
                _trig = self.public_exchange.price_to_precision(pair.replace('USDT', '/USDT:USDT'), trigger_price)
            except Exception:
                _trig = f"{trigger_price:.6g}"
            res = await self.exchange.fapiPrivatePostAlgoOrder({
                'algoType': 'CONDITIONAL',
                'symbol': pair,
                'side': side,
                'type': 'STOP_MARKET',
                'triggerPrice': _trig,
                'closePosition': 'true',
                'workingType': 'MARK_PRICE',
                'priceProtect': 'TRUE',
            })
            _aid = res.get('algoId') if isinstance(res, dict) else None
            return str(_aid) if _aid else None
        except Exception as e:
            # Aug-25 live (PROM): -4130 'an open stop already exists' = an ORPHAN backstop from a
            # failed open attempt (placed, then the DB insert died). Backstops are the ONLY algo
            # orders this bot uses, so cancel-all-for-symbol is safe: clear the orphan, retry once.
            if '-4130' in str(e):
                try:
                    await self.exchange.fapiPrivateDeleteAlgoOpenOrders({'symbol': pair})
                    logger.warning(f"[BINANCE] {pair}: cleared orphan algo order(s) after -4130 — retrying backstop place")
                    res = await self.exchange.fapiPrivatePostAlgoOrder({
                        'algoType': 'CONDITIONAL', 'symbol': pair,
                        'side': 'SELL' if direction == 'LONG' else 'BUY',
                        'type': 'STOP_MARKET', 'triggerPrice': _trig,
                        'closePosition': 'true', 'workingType': 'MARK_PRICE', 'priceProtect': 'TRUE',
                    })
                    _aid = res.get('algoId') if isinstance(res, dict) else None
                    if _aid:
                        return str(_aid)
                except Exception as e2:
                    logger.error(f"[BINANCE] backstop -4130 recovery failed for {pair}: {e2}")
            self._detect_ban(e)
            logger.error(f"[BINANCE] backstop place failed for {pair}: {e}")
            return None

    async def cancel_backstop_stop(self, pair: str, algo_id: str) -> bool:
        """Cancel a resting backstop. Already-gone/triggered orders count as success."""
        try:
            await self._check_ban()
            await self.exchange.fapiPrivateDeleteAlgoOrder({'algoId': algo_id})
            return True
        except Exception as e:
            _m = str(e)
            if any(t in _m for t in ('-2011', 'Unknown order', 'not exist', 'NOT_FOUND', 'CANCELED', 'TRIGGERED')):
                return True  # nothing resting = already done
            self._detect_ban(e)
            logger.error(f"[BINANCE] backstop cancel failed for {pair} algoId={algo_id}: {e}")
            return False

    async def query_backstop_stop(self, algo_id: str):
        """Query a backstop algo order (dict incl. algoStatus) or None."""
        try:
            await self._check_ban()
            return await self.exchange.fapiPrivateGetAlgoOrder({'algoId': algo_id})
        except Exception as e:
            self._detect_ban(e)
            logger.error(f"[BINANCE] backstop query failed algoId={algo_id}: {e}")
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

    async def fetch_orderbook_depth(self, symbol: str, limit: int = 500) -> Optional[Dict]:
        """📖 Oct-3: the raw book (bids / asks, best first) for research stamps — {'bids': [[p, q], …], 'asks': […]}. Public endpoint (weight
        10 at 500 levels). None on any error (never raises); never sleeps out a ban (the caller bounds it with wait_for)."""
        try:
            if _ban_until > time.time():
                return None
            if self.research_exchange is None:
                self.research_exchange = ccxt.binanceusdm({'enableRateLimit': True, 'options': {'defaultType': 'future', 'adjustForTimeDifference': True}})
            if not getattr(self.research_exchange, 'markets', None):
                await self.research_exchange.load_markets()
            ob = await self.research_exchange.fetch_order_book(symbol, limit)   # its own client + throttle (deep review)
            if ob and ob.get('bids') and ob.get('asks'):
                return {'bids': ob['bids'], 'asks': ob['asks']}
            return None
        except Exception as e:
            self._detect_ban(e)
            logger.debug(f"[BINANCE] depth read failed for {symbol}: {e}")
            return None

    async def fetch_ohlcv_research(self, symbol: str, timeframe: str = '5m', limit: int = 60) -> Optional[List]:
        """🌊 Oct-3: klines on the RESEARCH client (its own throttle — FRENZY's market-wide volume read at the bar close never queues behind the
        scan / FRENZY / order reads). One attempt, None on any error (never raises, never sleeps out a ban; the caller bounds it with wait_for)."""
        try:
            if _ban_until > time.time():
                return None
            if self.research_exchange is None:
                self.research_exchange = ccxt.binanceusdm({'enableRateLimit': True, 'options': {'defaultType': 'future', 'adjustForTimeDifference': True}})
            if not getattr(self.research_exchange, 'markets', None):
                await self.research_exchange.load_markets()
            return await self.research_exchange.fetch_ohlcv(symbol, timeframe, limit=limit) or None
        except Exception as e:
            self._detect_ban(e)
            logger.debug(f"[BINANCE] research kline read failed for {symbol}: {e}")
            return None

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
            await self._check_ban()  # Aug-24 M3
            await self.load_markets()
            if not is_close:
                actual_leverage = await self.set_leverage(symbol, leverage)
                if actual_leverage == 0:
                    logger.error(f"[LEVERAGE_MISMATCH] {symbol}: Cannot determine leverage, skipping limit order")
                    _leverage_blocked_pairs[symbol] = time.time()
                    return None
                if actual_leverage != leverage:
                    logger.warning(f"[LEVERAGE_MISMATCH] {symbol}: Binance leverage {actual_leverage}x != configured {leverage}x — blocking pair")
                    _leverage_blocked_pairs[symbol] = time.time()
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
                'fee': float((order.get('fee') or {}).get('cost') or 0),
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
                'fee': float((order.get('fee') or {}).get('cost') or 0),
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
        Returns the position dict if open, None ONLY when Binance answered and there is no position. A FAILED read RAISES
        (Oct-2, DECISION_LOG 183): it used to return None as well, so "could not read" looked like "position gone" — the close
        path then booked a still-open trade as closed (its safety stop already cancelled), and the reconciler's per-symbol
        fallback, which only runs when the exchange is failing, closed every live row as EXTERNAL. Both callers already treat an
        exception as "still open"."""
        try:
            await self.load_markets()
            positions = await self.exchange.fetch_positions([symbol])
            for pos in positions:
                contracts = float(pos.get('contracts') or 0)
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
            self._detect_ban(e)
            logger.error(f"[BINANCE] Error fetching position for {symbol}: {e}")
            raise

    async def get_funding_fees_usd(self, pair: str, start_ms: int, end_ms: int) -> Optional[float]:
        """Aug-24 M6: Σ FUNDING_FEE income for one symbol over [start_ms, end_ms] (negative = paid). None on error."""
        try:
            if _ban_until > 0 and time.time() < _ban_until:  # review I1: optional metadata — never sleep out a ban for it
                return None
            await self.load_markets()
            _sym = pair.replace('USDT', '/USDT:USDT') if '/' not in pair else pair
            _mid = self.exchange.market_id(_sym)
            rows = await self.exchange.fapiPrivateGetIncome({
                'symbol': _mid, 'incomeType': 'FUNDING_FEE',
                'startTime': int(start_ms), 'endTime': int(end_ms), 'limit': 1000,
            })
            return float(sum(float(r.get('income', 0) or 0) for r in (rows or [])))
        except Exception as e:
            logger.warning(f"[FUNDING] income fetch failed for {pair}: {e}")
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

    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch the current funding rate for a futures symbol (Exploration Analytics, Apr 28).

        Returns the rate as a decimal (e.g., 0.0001 = 0.01%) or None on error.
        Cached per-symbol for 8h to match Binance's funding interval — first call
        per scan-cycle hits the API, subsequent same-symbol calls within 8h reuse.
        """
        try:
            now = time.time()
            cached = self._funding_rate_cache.get(symbol)
            if cached:
                rate, fetched_at = cached
                if (now - fetched_at) < self._funding_rate_ttl_seconds:
                    return rate

            await self.load_public_markets()
            data = await self.public_exchange.fetch_funding_rate(symbol)
            if data is None:
                return None
            rate = data.get('fundingRate')
            if rate is None:
                return None
            rate_f = float(rate)
            self._funding_rate_cache[symbol] = (rate_f, now)
            return rate_f
        except Exception as e:
            logger.warning(f"[BINANCE] Error fetching funding rate for {symbol}: {e}")
            return None


# Global service instance
binance_service = BinanceService()
