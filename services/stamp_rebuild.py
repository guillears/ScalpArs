"""🧾⏪ (DECISION_LOG 393, 2026-10-27) RESEARCH ONLY — the candidate rebuild of the shared entry stamps of fills that opened BEFORE the
post-fill backfill (DL 389), used by scripts/study_stamp_rebuild_parity.py. Only the pair 1d −DI and the BTC 4h gap rebuild exactly
(30 / 30 each incl. the deep-review re-check; listing age 391 / 391); everything off the forming 5m bar does not (16–41 % exact) → the
ENGINE writes only those two + listing age, under its ambiguity rule
(TradingEngine._stamp_rebuild_exact). Never write this module's other outputs into the orders table.
Pure: klines in → stamps out, with the bot's OWN formulas on the bars the bot would have read at the fill:

  pair     the 100 5m bars whose open ≤ the fill (the scan's window, the forming bar included as the scan reads it) → calculate_indicators
           with price = the fill price → pair_entry_stamps (the momentum scan / MANUAL formulas)
  BTC 5m   the 100 5m bars whose open ≤ the fill → the scan's BTC block: adx / adx_prev1 / rsi / rsi_prev1 / rsi_prev6 / atr % / EMA20
           3-bar slope / EMA13−EMA50 trend gap / price vs EMA13 / price
  BTC 1h   the 100 1h bars whose open ≤ the fill → EMA20 3-bar slope, RSI, RSI one bar back
  BTC 4h   closed_ema_gap_pct(4h bars ≤ the fill, 50, 200) (the fade laggard gate's reading)
  pair 1d  closed_wilder_ndi(1d bars ≤ the fill) (the fade laggard gate's reading)
  funding  the last SETTLED funding rate at or before the fill (live stamps the premium index's current rate — close, not identical)
  age      days since Binance onboardDate
  FRENZY   di_spread / adx_delta / vol_trend on the CLOSED 5m bars before the fill — FLATBASE fills only (FRENZY_* stamp their own)

NOT rebuildable (scan-wide state never stored): market breadth (bull / bear %), the global volume ratio, the quality score (needs the
breadth), the BTC 72 h / 24 h monitors, alt-corr. Written with COALESCE — a value the fill already carries always wins.
"""
from typing import Optional

BAR5 = 300_000


def _upto(rows, end_ms, n):
    """the last n bars whose open ≤ end_ms (rows oldest first)."""
    r = [b for b in (rows or []) if int(b[0]) <= int(end_ms)]
    return r[-n:]


def _r(v, n):
    return round(float(v), n) if v is not None else None


def btc_5m_stamps(ind) -> dict:
    if not ind:
        return {}
    px, e13, e20, e20p3, e50, atr = (ind.get(k) for k in ('price', 'ema13', 'ema20', 'ema20_prev3', 'ema50', 'atr'))
    return dict(
        entry_btc_adx=_r(ind.get('adx'), 4), entry_btc_adx_prev=_r(ind.get('adx_prev1'), 4),
        entry_btc_rsi=_r(ind.get('rsi'), 1), entry_btc_rsi_prev=_r(ind.get('rsi_prev1'), 1), entry_btc_rsi_prev6=_r(ind.get('rsi_prev6'), 1),
        entry_btc_atr_pct=(round(atr / px * 100, 4) if (atr is not None and px) else None),
        entry_btc_ema20_slope=(round((e20 - e20p3) / e20p3 * 100, 4) if (e20 is not None and e20p3) else None),
        entry_btc_trend_gap_pct=(round((e13 - e50) / e50 * 100, 4) if (e13 is not None and e50) else None),
        entry_btc_dist_from_ema13_pct=(round((px - e13) / e13 * 100, 4) if (px is not None and e13) else None),
        entry_btc_price=px)


def btc_1h_stamps(ind) -> dict:
    if not ind:
        return {}
    e20, e20p3 = ind.get('ema20'), ind.get('ema20_prev3')
    return dict(entry_btc_1h_slope=(round((e20 - e20p3) / e20p3 * 100, 4) if (e20 is not None and e20p3) else None),
                entry_btc_rsi_1h=_r(ind.get('rsi'), 1), entry_btc_rsi_1h_prev=_r(ind.get('rsi_prev1'), 1))


def rebuild_stamps(direction, fill_ms, entry_price, *, pair_k5=None, btc_k5=None, btc_k1h=None, btc_k4h=None, pair_k1d=None,
                   funding=None, onboard_ms=None, frenzy=False) -> dict:
    """→ {entry_column: value} for one fill (None values dropped). funding = [(fundingTime_ms, rate)]. Never raises."""
    from services.indicators import calculate_indicators, closed_ema_gap_pct, closed_wilder_ndi
    from services.trading_engine import pair_entry_stamps
    out = {}

    def _ind(rows):
        try:
            return calculate_indicators(rows) if rows and len(rows) >= 30 else None
        except Exception:
            return None
    try:
        w = _upto(pair_k5, fill_ms, 100)
        ind = _ind(w)
        if ind:
            if entry_price and float(entry_price) > 0:
                ind = dict(ind, price=float(entry_price))
            out.update(pair_entry_stamps(ind, direction))
    except Exception:
        pass
    try:
        out.update(btc_5m_stamps(_ind(_upto(btc_k5, fill_ms, 100))))
    except Exception:
        pass
    try:
        out.update(btc_1h_stamps(_ind(_upto(btc_k1h, fill_ms, 100))))
    except Exception:
        pass
    try:
        if btc_k4h:
            out['entry_btc_4h_ema50_200_gap_pct'] = closed_ema_gap_pct(_upto(btc_k4h, fill_ms, 1000), 50, 200)
    except Exception:
        pass
    try:
        if pair_k1d:
            out['entry_pair_1d_ndi'] = closed_wilder_ndi(_upto(pair_k1d, fill_ms, 200))
    except Exception:
        pass
    try:
        fr = [(int(t), float(v)) for t, v in (funding or []) if int(t) <= int(fill_ms)]
        if fr:
            out['entry_funding_rate'] = round(max(fr)[1], 6)
    except Exception:
        pass
    try:
        if onboard_ms:
            out['entry_pair_age_days'] = round((int(fill_ms) - int(onboard_ms)) / 86400000.0, 1)
    except Exception:
        pass
    if frenzy:
        try:
            from services.frenzy import frenzy_di_spread, frenzy_adx_delta, frenzy_vol_trend
            cl = [b for b in (pair_k5 or []) if int(b[0]) + BAR5 <= int(fill_ms)]
            for col, fn, rows in (('entry_frenzy_di_spread', frenzy_di_spread, cl[-300:]), ('entry_frenzy_adx_delta', frenzy_adx_delta, cl[-300:]),
                                  ('entry_frenzy_vol_trend', frenzy_vol_trend, cl)):
                v = fn(rows)
                if v is not None:
                    out[col] = v
        except Exception:
            pass
    return {k: v for k, v in out.items() if v is not None}
