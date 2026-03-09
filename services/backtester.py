"""
SCALPARS Backtest Engine
Replays historical 5m candles through signal/exit logic for strategy comparison.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from config import TradingConfig
from services.indicators import get_signal, determine_macro_regime

logger = logging.getLogger(__name__)

PAIR_SYMBOLS = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "DOGE": "DOGE/USDT:USDT",
    "HYPE": "HYPE/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

async def fetch_historical_candles(
    binance_service, pair_keys: List[str], days: int
) -> Dict[str, List]:
    """Fetch paginated 5m candles from Binance for each pair."""
    await binance_service.load_public_markets()
    now_ms = int(time.time() * 1000)
    start_ms = now_ms - (days * 86_400_000)
    result: Dict[str, List] = {}

    symbols_needed = set()
    for key in pair_keys:
        symbols_needed.add(PAIR_SYMBOLS.get(key, f"{key}/USDT:USDT"))
    symbols_needed.add(PAIR_SYMBOLS["BTC"])

    for symbol in symbols_needed:
        candles: list = []
        since = start_ms
        while since < now_ms:
            try:
                batch = await binance_service.public_exchange.fetch_ohlcv(
                    symbol, "5m", since=since, limit=1500
                )
            except Exception as e:
                logger.error(f"[BACKTEST] fetch error {symbol}: {e}")
                break
            if not batch:
                break
            candles.extend(batch)
            since = batch[-1][0] + 1
            await asyncio.sleep(0.15)
        result[symbol] = candles
        logger.info(f"[BACKTEST] {symbol}: {len(candles)} candles fetched")

    return result


# ---------------------------------------------------------------------------
# Vectorised indicator pre-computation
# ---------------------------------------------------------------------------

def _sf(val) -> Optional[float]:
    """Safe float conversion (NaN -> None)."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def precompute_indicators(candles: List) -> List[Dict]:
    """Compute all indicators over the full candle history at once (O(n))."""
    if not candles or len(candles) < 60:
        return []

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)

    ema5 = EMAIndicator(close=df["close"], window=5).ema_indicator()
    ema8 = EMAIndicator(close=df["close"], window=8).ema_indicator()
    ema13 = EMAIndicator(close=df["close"], window=13).ema_indicator()
    ema20 = EMAIndicator(close=df["close"], window=20).ema_indicator()
    ema50 = EMAIndicator(close=df["close"], window=50).ema_indicator()
    rsi_s = RSIIndicator(close=df["close"], window=12).rsi()
    adx_s = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()
    avg_vol = df["volume"].rolling(window=20).mean()

    out: List[Dict] = []
    for i in range(56, len(df)):
        out.append({
            "ts": int(df["timestamp"].iloc[i]),
            "price": float(df["close"].iloc[i]),
            "open": float(df["open"].iloc[i]),
            "high": float(df["high"].iloc[i]),
            "low": float(df["low"].iloc[i]),
            "ema5": _sf(ema5.iloc[i]),
            "ema5_prev3": _sf(ema5.iloc[i - 3]),
            "ema8": _sf(ema8.iloc[i]),
            "ema13": _sf(ema13.iloc[i]),
            "ema20": _sf(ema20.iloc[i]),
            "ema20_prev6": _sf(ema20.iloc[i - 6]),
            "ema50": _sf(ema50.iloc[i]),
            "ema50_prev6": _sf(ema50.iloc[i - 6]),
            "rsi": _sf(rsi_s.iloc[i]),
            "adx": _sf(adx_s.iloc[i]),
            "vol": float(df["volume"].iloc[i]),
            "avg_vol": _sf(avg_vol.iloc[i]),
        })
    return out


# ---------------------------------------------------------------------------
# Position-size calculator
# ---------------------------------------------------------------------------

def _calc_size(balance: float, cfg: TradingConfig, conf) -> float:
    inv = cfg.investment
    reserve = (balance * inv.reserve_percentage / 100
               if inv.reserve_mode == "percentage" else inv.reserve_fixed)
    tradeable = max(0.0, balance - reserve)
    if tradeable <= 0:
        return 0.0

    if inv.mode == "percentage":
        base = tradeable * inv.percentage / 100
    elif inv.mode == "equal_split":
        base = (balance - reserve) / max(1, inv.max_open_positions)
    else:
        base = min(inv.fixed_amount, tradeable)

    amt = base * conf.investment_multiplier
    amt = max(inv.min_investment_size, min(amt, inv.max_investment_size, tradeable))
    return amt


# ---------------------------------------------------------------------------
# Exit checker (self-contained, no global state)
# ---------------------------------------------------------------------------

def _pnl_pct(direction, entry_price, price, quantity, entry_fee, trading_fee):
    raw = ((price - entry_price) if direction == "LONG" else (entry_price - price)) * quantity
    net = raw - entry_fee - (price * quantity * trading_fee)
    notional = entry_price * quantity
    return (net / notional) * 100 if notional else 0.0


def _check_exits(pos: dict, ind: dict, sig: str, cfg: TradingConfig) -> Optional[str]:
    """Return close_reason string or None."""
    conf = cfg.confidence_levels.get(pos["confidence"])
    if not conf:
        return None

    th = cfg.thresholds
    d = pos["direction"]
    ep = pos["entry_price"]
    qty = pos["quantity"]
    fee = cfg.trading_fee
    ef = pos["entry_fee"]
    tp_level = pos["tp_level"]

    # Track high/low with intra-candle extremes
    if d == "LONG":
        pos["high_price"] = max(pos["high_price"], ind["high"])
    else:
        pos["low_price"] = min(pos["low_price"], ind["low"])

    pnl = _pnl_pct(d, ep, ind["price"], qty, ef, fee)
    worst_price = ind["low"] if d == "LONG" else ind["high"]
    worst_pnl = _pnl_pct(d, ep, worst_price, qty, ef, fee)

    if pnl > pos["peak_pnl"]:
        pos["peak_pnl"] = pnl
    if worst_pnl < pos["trough_pnl"]:
        pos["trough_pnl"] = worst_pnl

    peak = pos["peak_pnl"]
    sig_active = sig == d

    # 1 — Stop loss (checked at intra-candle worst)
    eff_sl = conf.stop_loss
    be_active = False
    if peak >= conf.breakeven_trigger:
        eff_sl = conf.breakeven_offset
        be_active = True
    elif sig_active:
        eff_sl = conf.signal_active_sl

    if worst_pnl <= eff_sl:
        if be_active:
            return f"BREAKEVEN_SL L{tp_level}"
        return f"{'STOP_LOSS_WIDE' if sig_active else 'STOP_LOSS'} L{tp_level}"

    # 2 — P&L trailing
    if th.pnl_trailing_trigger > 0 and th.pnl_trailing_ratio > 0 and peak >= th.pnl_trailing_trigger:
        ratio = th.pnl_trailing_ratio_signal_active if sig_active else th.pnl_trailing_ratio
        if pnl <= peak * ratio:
            tag = "PNL_TRAILING" if sig_active else "MOMENTUM_EXIT"
            return f"{tag} L{tp_level}"

    # 3 — Slope exit
    if th.ema5_slope_exit_enabled:
        e5, e5p = ind.get("ema5"), ind.get("ema5_prev3")
        if e5 and e5p and e5p != 0:
            slope = ((e5 - e5p) / e5p) * 100
            if (d == "LONG" and slope <= th.ema5_slope_threshold) or \
               (d == "SHORT" and slope >= -th.ema5_slope_threshold):
                return f"SLOPE_EXIT L{tp_level}"

    # 4 — Signal lost
    if th.signal_lost_exit_enabled and sig != d:
        dyn_tp = pos.get("dynamic_tp") or conf.tp_min
        if th.signal_lost_min_profit <= pnl < dyn_tp:
            return f"SIGNAL_LOST L{tp_level}"

    # 5 — TP extension / trailing stop
    if conf.tp_trailing_enabled:
        eff_tp = pos.get("dynamic_tp") or conf.tp_min
        if pnl >= eff_tp:
            trend = False
            e5, e8, e13, e20 = ind.get("ema5"), ind.get("ema8"), ind.get("ema13"), ind.get("ema20")
            if all(v is not None for v in (e5, e8, e13, e20)):
                gap = ((e5 - e20) / ind["price"]) * 100
                trend = (e5 > e8 > e13 > e20 and gap > 0) if d == "LONG" else (e5 < e8 < e13 < e20 and gap < 0)
            if trend:
                new_lv = max(int(pnl / conf.tp_min), tp_level + 1)
                pos["tp_level"] = new_lv
                pos["dynamic_tp"] = new_lv * conf.tp_min
            else:
                return f"TRAILING_STOP L{tp_level}"

        trailing_on = peak >= eff_tp or tp_level >= 2
        if trailing_on:
            if d == "LONG" and pos["high_price"] > 0:
                drop = ((pos["high_price"] - ind["price"]) / pos["high_price"]) * 100
                if drop >= conf.pullback_trigger and pnl >= 0:
                    return f"TRAILING_STOP L{tp_level}"
            elif d == "SHORT" and pos["low_price"] > 0:
                rise = ((ind["price"] - pos["low_price"]) / pos["low_price"]) * 100
                if rise >= conf.pullback_trigger and pnl >= 0:
                    return f"TRAILING_STOP L{tp_level}"

    return None


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def run_backtest(
    indicators_by_pair: Dict[str, List[Dict]],
    btc_indicators: List[Dict],
    cfg: TradingConfig,
    initial_balance: float,
) -> Dict:
    """Run one strategy against pre-computed indicators. Returns aggregated results."""
    import config as config_module

    balance = initial_balance
    positions: Dict[str, dict] = {}
    trades: List[dict] = []
    cooldowns: Dict[str, int] = {}
    pair_sigs: Dict[str, str] = {}

    th = cfg.thresholds
    inv = cfg.investment
    fee = cfg.trading_fee

    btc_lk = {ind["ts"]: ind for ind in btc_indicators}
    pair_lk: Dict[str, Dict[int, Dict]] = {}
    for pair, inds in indicators_by_pair.items():
        pair_lk[pair] = {ind["ts"]: ind for ind in inds}

    all_ts = sorted({ind["ts"] for inds in indicators_by_pair.values() for ind in inds})

    for ts in all_ts:
        # BTC regime
        bi = btc_lk.get(ts)
        btc_e50 = bi["ema50"] if bi and th.btc_global_filter_enabled else None
        btc_e50p = bi["ema50_prev6"] if bi and th.btc_global_filter_enabled else None

        # --- exits ---
        for pair in list(positions):
            ind = pair_lk.get(pair, {}).get(ts)
            if not ind:
                continue
            reason = _check_exits(positions[pair], ind, pair_sigs.get(pair, "NOTHING"), cfg)
            if reason:
                pos = positions.pop(pair)
                p = ind["price"]
                d = pos["direction"]
                q = pos["quantity"]
                raw = ((p - pos["entry_price"]) if d == "LONG" else (pos["entry_price"] - p)) * q
                ef = p * q * fee
                net = raw - pos["entry_fee"] - ef
                notional = pos["entry_price"] * q
                pnl_p = (net / notional) * 100 if notional else 0

                trades.append({
                    "pair": pair, "direction": d, "confidence": pos["confidence"],
                    "entry_price": pos["entry_price"], "exit_price": p,
                    "investment": pos["investment"], "leverage": pos["leverage"],
                    "pnl": net, "pnl_pct": pnl_p,
                    "peak_pnl": pos["peak_pnl"], "trough_pnl": pos["trough_pnl"],
                    "close_reason": reason,
                    "opened_at": pos["opened_at"], "closed_at": ts,
                    "signal_active": pair_sigs.get(pair) == d,
                    "entry_rsi": pos.get("entry_rsi"),
                    "entry_adx": pos.get("entry_adx"),
                    "entry_gap": pos.get("entry_gap"),
                })
                balance += pos["investment"] + net
                if net < 0 and inv.cooldown_after_loss_minutes > 0:
                    cooldowns[pair] = ts + inv.cooldown_after_loss_minutes * 60_000

        # --- entries ---
        for pair, plk in pair_lk.items():
            ind = plk.get(ts)
            if not ind:
                continue

            re50 = btc_e50 if th.btc_global_filter_enabled else ind.get("ema50")
            re50p = btc_e50p if th.btc_global_filter_enabled else ind.get("ema50_prev6")

            orig = config_module.trading_config
            config_module.trading_config = cfg
            try:
                sig, conf_name = get_signal(
                    ema5=ind["ema5"], ema8=ind["ema8"], ema13=ind["ema13"], ema20=ind["ema20"],
                    rsi=ind["rsi"], adx=ind["adx"],
                    volume=ind["vol"], avg_volume=ind["avg_vol"],
                    price=ind["price"], ema20_prev6=ind["ema20_prev6"],
                    ema50=re50, ema50_prev6=re50p,
                )
            finally:
                config_module.trading_config = orig

            pair_sigs[pair] = sig
            if sig not in ("LONG", "SHORT") or not conf_name or conf_name == "NO_TRADE":
                continue
            if pair in positions:
                continue
            cc = cfg.confidence_levels.get(conf_name)
            if not cc or not cc.enabled:
                continue
            if len(positions) >= inv.max_open_positions:
                continue
            if pair in cooldowns and ts < cooldowns[pair]:
                continue

            amt = _calc_size(balance, cfg, cc)
            if amt <= 0:
                continue

            notional = amt * cc.leverage
            qty = notional / ind["price"]
            e_fee = notional * fee
            balance -= amt

            positions[pair] = {
                "direction": sig, "confidence": conf_name,
                "entry_price": ind["price"], "quantity": qty,
                "investment": amt, "leverage": cc.leverage, "entry_fee": e_fee,
                "high_price": ind["high"], "low_price": ind["low"],
                "peak_pnl": 0.0, "trough_pnl": 0.0,
                "tp_level": 1, "dynamic_tp": None,
                "opened_at": ts,
                "entry_rsi": ind.get("rsi"), "entry_adx": ind.get("adx"),
                "entry_gap": (abs((ind["ema5"] - ind["ema20"]) / ind["price"] * 100)
                              if ind["ema5"] and ind["ema20"] and ind["price"] else None),
            }

    # Force-close remaining positions
    for pair, pos in positions.items():
        inds = indicators_by_pair.get(pair, [])
        if not inds:
            continue
        last = inds[-1]
        p = last["price"]
        d = pos["direction"]
        q = pos["quantity"]
        raw = ((p - pos["entry_price"]) if d == "LONG" else (pos["entry_price"] - p)) * q
        ef = p * q * fee
        net = raw - pos["entry_fee"] - ef
        notional = pos["entry_price"] * q
        pnl_p = (net / notional) * 100 if notional else 0
        trades.append({
            "pair": pair, "direction": d, "confidence": pos["confidence"],
            "entry_price": pos["entry_price"], "exit_price": p,
            "investment": pos["investment"], "leverage": pos["leverage"],
            "pnl": net, "pnl_pct": pnl_p,
            "peak_pnl": pos["peak_pnl"], "trough_pnl": pos["trough_pnl"],
            "close_reason": "END_OF_DATA",
            "opened_at": pos["opened_at"], "closed_at": last["ts"],
            "signal_active": pair_sigs.get(pair) == d,
            "entry_rsi": pos.get("entry_rsi"), "entry_adx": pos.get("entry_adx"),
            "entry_gap": pos.get("entry_gap"),
        })
        balance += pos["investment"] + net

    return _aggregate(trades, initial_balance, balance)


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------

def _fmt_dur(minutes: float) -> str:
    m = int(minutes)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}"


def _aggregate(trades: List[dict], initial_bal: float, final_bal: float) -> Dict:
    empty = {
        "total_trades": 0, "winning": 0, "losing": 0,
        "win_rate": 0, "total_pnl": 0, "total_pnl_pct": 0,
        "avg_pnl_pct": 0, "avg_duration": "00:00",
        "final_balance": round(final_bal, 2),
        "by_close_reason": {}, "by_direction": {}, "by_confidence": {},
        "equity_curve": [],
    }
    if not trades:
        return empty

    total = len(trades)
    wins = [t for t in trades if t["pnl"] > 0]
    total_pnl = sum(t["pnl"] for t in trades)
    durs = [(t["closed_at"] - t["opened_at"]) / 60_000 for t in trades]

    # By close reason
    by_reason: Dict[str, dict] = {}
    for t in trades:
        rk = (t["close_reason"] or "UNKNOWN").split(" L")[0]
        if rk not in by_reason:
            by_reason[rk] = {"n": 0, "pnl": 0.0, "pcts": [], "durs": [], "dirs": {},
                             "sig_a": 0, "sig_i": 0}
        r = by_reason[rk]
        r["n"] += 1
        r["pnl"] += t["pnl"]
        r["pcts"].append(t["pnl_pct"])
        r["durs"].append((t["closed_at"] - t["opened_at"]) / 60_000)
        r["dirs"][t["direction"]] = r["dirs"].get(t["direction"], 0) + 1
        if t.get("signal_active"):
            r["sig_a"] += 1
        else:
            r["sig_i"] += 1

    by_close_reason = {}
    for rk, r in by_reason.items():
        by_close_reason[rk] = {
            "trades": r["n"],
            "avg_pnl_pct": round(sum(r["pcts"]) / len(r["pcts"]), 4),
            "total_pnl": round(r["pnl"], 2),
            "avg_duration": _fmt_dur(sum(r["durs"]) / len(r["durs"])),
            "by_direction": r["dirs"],
            "signal_active": r["sig_a"],
            "signal_inactive": r["sig_i"],
        }

    # By direction
    by_dir = {}
    for d in ("LONG", "SHORT"):
        dt = [t for t in trades if t["direction"] == d]
        if dt:
            by_dir[d] = {
                "trades": len(dt),
                "total_pnl": round(sum(t["pnl"] for t in dt), 2),
                "avg_pnl_pct": round(sum(t["pnl_pct"] for t in dt) / len(dt), 4),
                "win_rate": round(len([t for t in dt if t["pnl"] > 0]) / len(dt) * 100, 1),
            }

    # By confidence
    by_conf: Dict[str, dict] = {}
    for t in trades:
        c = t["confidence"]
        if c not in by_conf:
            by_conf[c] = {"trades": 0, "total_pnl": 0.0, "wins": 0}
        by_conf[c]["trades"] += 1
        by_conf[c]["total_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            by_conf[c]["wins"] += 1
    for c in by_conf:
        by_conf[c]["total_pnl"] = round(by_conf[c]["total_pnl"], 2)
        by_conf[c]["win_rate"] = round(by_conf[c]["wins"] / by_conf[c]["trades"] * 100, 1)

    # Equity curve
    sorted_t = sorted(trades, key=lambda t: t["closed_at"])
    eq: list = []
    cum = 0.0
    for t in sorted_t:
        cum += t["pnl"]
        eq.append({"ts": t["closed_at"], "pnl": round(cum, 2)})

    return {
        "total_trades": total,
        "winning": len(wins),
        "losing": total - len(wins),
        "win_rate": round(len(wins) / total * 100, 1),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / initial_bal * 100, 2) if initial_bal else 0,
        "avg_pnl_pct": round(sum(t["pnl_pct"] for t in trades) / total, 4),
        "avg_duration": _fmt_dur(sum(durs) / len(durs)),
        "final_balance": round(final_bal, 2),
        "by_close_reason": by_close_reason,
        "by_direction": by_dir,
        "by_confidence": by_conf,
        "equity_curve": eq,
    }
