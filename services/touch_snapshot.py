"""🛑👁 2026-10-19 TOUCH SNAPSHOTS (DECISION_LOG 317 — OBSERVE ONLY; operator: "evaluate in LONGs speed of the drop, entry ADX, price vs 5m
EMA50 at the touch, BTC since entry — among ALL other points — at −2 % and at −3 %; goal: understand which recover and which not, so we can
split the SL in future").

Every FRENZY-family fill (FRENZY_LONG / WIDE / LITE / LITE_RETRY / WILLY) records ONCE, at its FIRST touch of −2 % and of −3 % net, a snapshot
of what the engine already holds in memory (no REST call anywhere on this path): the touch time and minutes from entry, the best net P&L
before the touch, the pair's 5m EMA50 / RSI / ADX / ATR % from the FRENZY kline cache (last CLOSED bar — ≤ 5 min old), price vs the spike
VWAP (FRENZY flag), the last closed 5m bar's volume vs the entry signal bar's (pair fields None when the cache's last closed bar closed
> 10 min before the touch), BTC since entry (scan-refreshed BTC price — scan-interval resolution — vs the stamped
entry_btc_price) and the BTC 1h EMA20 slope. At the close: recovered_tp (the trade reached its take profit after the touch) and min_after
(the worst net P&L after the touch). NaN / NULL when a value is not in memory. Never changes an exit decision. Pure helpers; never raise."""
import math
from datetime import datetime
from typing import Dict, List, Optional

TOUCH_LEVELS = (2, 3)
TOUCH_FIELDS = ("at", "min", "peak_before", "vs_ema50_pct", "rsi", "adx", "vs_vwap_pct", "vol_ratio", "atr_pct", "btc_since_entry_pct",
                "btc_1h_slope", "recovered_tp", "min_after")
TP_REASONS = ("FRENZY_TP", "WILLY_LATE_TP", "WILLY_FLOOR")   # 🎲² (356) a V2 runner that armed and floored at ≈ +1.0 recovered past the WILLY TP
BAR_MS = 300_000
STALE_MS = 600_000   # the last cached CLOSED bar must have closed ≤ 10 min before the touch, else the pair fields are None


def touch_columns() -> List[str]:
    return [f"touch{lv}_{f}" for lv in TOUCH_LEVELS for f in TOUCH_FIELDS]


def _fin(v) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def touch_mark(markers: Dict, pnl, peak_before, now_ms, price) -> List[int]:
    """Update the in-memory markers of ONE order with a new net P&L reading → the levels touched for the FIRST time by this reading.
    markers = {level: {ms, peak, price, min, max}}; min / max track the net P&L from the touch on (the outcome fields)."""
    out = []
    try:
        p = _fin(pnl)
        if p is None or markers is None:
            return out
        for lv, m in markers.items():
            m['min'] = min(m.get('min', p), p)
            m['max'] = max(m.get('max', p), p)
        for lv in TOUCH_LEVELS:
            if lv not in markers and p <= -float(lv):
                pk = _fin(peak_before)
                markers[lv] = dict(ms=int(now_ms), peak=(pk if pk is not None else None), price=_fin(price), min=p, max=p)
                out.append(lv)
    except Exception:
        return out
    return out


def _ema_last(vals, n):
    if len(vals) < n:
        return None
    k = 2.0 / (n + 1)
    e = sum(vals[:n]) / n
    for v in vals[n:]:
        e = v * k + e * (1 - k)
    return e


def _wilder(vals, n):
    if len(vals) < n:
        return None
    a = sum(vals[:n]) / n
    for v in vals[n:]:
        a = (a * (n - 1) + v) / n
    return a


def _rsi(closes, n=14):
    if len(closes) < n + 2:
        return None
    d = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    g = _wilder([max(x, 0.0) for x in d], n); l_ = _wilder([max(-x, 0.0) for x in d], n)
    if g is None or l_ is None:
        return None
    return 100.0 if l_ == 0 else 100.0 - 100.0 / (1.0 + g / l_)


def _adx_atr(rows, n=14):
    """(ADX, ATR) Wilder on closed rows [ts, o, h, l, c, v]."""
    if len(rows) < 2 * n + 2:
        return None, None
    tr, pdm, mdm = [], [], []
    for i in range(1, len(rows)):
        h, l_, pc = float(rows[i][2]), float(rows[i][3]), float(rows[i - 1][4])
        ph, pl = float(rows[i - 1][2]), float(rows[i - 1][3])
        up, dn = h - ph, pl - l_
        pdm.append(up if (up > dn and up > 0) else 0.0); mdm.append(dn if (dn > up and dn > 0) else 0.0)
        tr.append(max(h - l_, abs(h - pc), abs(l_ - pc)))
    atr_s, p_s, m_s = [sum(tr[:n])], [sum(pdm[:n])], [sum(mdm[:n])]
    for i in range(n, len(tr)):
        atr_s.append(atr_s[-1] - atr_s[-1] / n + tr[i]); p_s.append(p_s[-1] - p_s[-1] / n + pdm[i]); m_s.append(m_s[-1] - m_s[-1] / n + mdm[i])
    dx = []
    for a, p, m in zip(atr_s, p_s, m_s):
        if a <= 0:
            dx.append(0.0); continue
        pdi, mdi = 100 * p / a, 100 * m / a
        dx.append(0.0 if (pdi + mdi) == 0 else 100 * abs(pdi - mdi) / (pdi + mdi))
    adx = _wilder(dx, n)
    return adx, _wilder(tr, n)


def touch_features(rows_5m, price, now_ms, entry_ms, vwap=None, btc_now=None, btc_entry=None, btc_1h_slope=None) -> Dict:
    """The snapshot fields at a touch (except at / min / peak_before / outcome) from in-memory data. rows_5m = the pair's cached 5m klines
    (only CLOSED bars at now_ms are used). Missing data → None per field. Never raises."""
    out = dict(vs_ema50_pct=None, rsi=None, adx=None, vs_vwap_pct=None, vol_ratio=None, atr_pct=None, btc_since_entry_pct=None, btc_1h_slope=None)
    try:
        px = _fin(price)
        closed = [r for r in (rows_5m or []) if r and len(r) >= 6 and int(r[0]) + BAR_MS <= int(now_ms)][-300:]
        if closed and int(closed[-1][0]) + BAR_MS < int(now_ms) - STALE_MS:   # (dual review) a stale cache (pair left the shortlist) → no pair fields
            closed = []
        closes = [float(r[4]) for r in closed]
        if px and closes:
            e50 = _ema_last(closes, 50)
            out['vs_ema50_pct'] = round((px / e50 - 1) * 100, 4) if e50 else None
            r_ = _rsi(closes)
            out['rsi'] = round(r_, 2) if r_ is not None else None
            adx, atr = _adx_atr(closed)
            out['adx'] = round(adx, 2) if adx is not None else None
            out['atr_pct'] = round(atr / closes[-1] * 100, 4) if (atr is not None and closes[-1] > 0) else None
            ent = [r for r in closed if int(r[0]) + BAR_MS <= int(entry_ms)]
            if ent and float(ent[-1][5]) > 0:
                out['vol_ratio'] = round(float(closed[-1][5]) / float(ent[-1][5]), 4)
        vw = _fin(vwap)
        if px and vw and vw > 0:
            out['vs_vwap_pct'] = round((px / vw - 1) * 100, 4)
        bn, be = _fin(btc_now), _fin(btc_entry)
        if bn and be and be > 0:
            out['btc_since_entry_pct'] = round((bn / be - 1) * 100, 4)
        sl = _fin(btc_1h_slope)
        out['btc_1h_slope'] = round(sl, 6) if sl is not None else None
    except Exception:
        return out
    return out


def touch_snapshot(marker, opened_at, feats) -> Dict:
    """→ {field: value} (without the touchN_ prefix) for one level: at / min / peak_before from the marker + the features."""
    try:
        ms = int(marker['ms'])
        at = datetime.utcfromtimestamp(ms / 1000)
        oa = opened_at.replace(tzinfo=None) if (opened_at is not None and getattr(opened_at, 'tzinfo', None)) else opened_at   # naive UTC like the rows
        mins = round((at - oa).total_seconds() / 60.0, 2) if oa is not None else None
        return dict(at=at, min=mins, peak_before=marker.get('peak'), **(feats or {}))
    except Exception:
        return {}


def touch_outcome(marker, final_pnl, close_reason, tp, trough=None):
    """(recovered_tp, min_after) at the close for a level that was touched. recovered_tp = the close booked a TP-class reason, or the net
    P&L after the touch reached +tp (marker max — lost after a restart → the reason alone). min_after = the worst net P&L after the touch
    (marker min, the final P&L, and the order's trough — after a first touch the lifetime trough IS the post-touch minimum)."""
    try:
        r = str(close_reason or "").upper()
        rec = any(r.startswith(x) or x in r for x in TP_REASONS)
        mx, t = _fin((marker or {}).get('max')), _fin(tp)
        if not rec and mx is not None and t is not None and t > 0 and mx >= t:
            rec = True
        cands = [x for x in (_fin((marker or {}).get('min')), _fin(final_pnl), _fin(trough)) if x is not None]
        return rec, (round(min(cands), 4) if cands else None)
    except Exception:
        return None, None
