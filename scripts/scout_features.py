"""🧬 Sep-30 — feature stamps for opportunity-scout events (DECISION_LOG 152).

Every stored scout event gets, ONCE, at the bar a sleeve could have entered (the event's anchor bar `bar_ts` — the same bar its
outcomes f30/f60/tbs2 are measured from, so there is no look-ahead):
  ① the bot's OWN entry_* columns, same names, computed by the bot's own pure code (pair_entry_stamps, market_entry_stamps,
     _calculate_quality_score, long_heat_eval, _compute_pattern_c/w_match, closed_* helpers, the Bull/Bear-Run monitor formulas),
     so a missed move and a real fill sit in one table and scripts/sweep_separators.py runs on both;
  ② pre-move features the bot does not record (compression, volume build-up, taker-buy share, open interest, funding, distance to
     multi-day extremes, lead/lag vs BTC and ETH, session) — prefixed `pre_`.
Convention (as scripts/validate_against_master.py F2): indicators are read on CLOSED bars ending at the stamp bar; the live bot
reads the forming bar, which sits between that bar and the next — both alignments are accepted there.
Sleeve-only bot fields (bull-run door br_*, fade laggard gate, liquidity cap) and market cap / CMC rank (needs the CoinMarketCap
key) stay empty, as on a momentum fill. NOT pooled under bot names because the quantity differs from the live stamp: volume ratios
(live = partial forming bar → pre_*_volume_ratio_closed), funding (settled, not predicted → pre_funding_settled), Bear-Run readings
(live only on bear-bypass fills → pre_bear_*). Breadth uses TODAY's bot universe (incl. BTC/ETH, minus pair_blacklist, as the
scan) on past bars (≤ 40 h back — small drift). entry_pair_volume_24h_usd = Σ close×volume of the 288 bars before the event (the
ticker's 24 h quote volume at scan time, approximated). DIRECTION: stamps describe the WITH-MOVE trade (UP → LONG, DOWN → SHORT);
direction-dependent columns (macro-trend threshold, patterns, gap_expand_marginal) must not be pooled with fade/flip fills as-is.
Pattern flags re-read from the CSV are 'True'/'False' strings — coerce before a sweep. Pure over its inputs except `fetch_pair_extras` (public Binance reads, fail-soft)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")   # importing the engine must never touch a real database
import config  # noqa: E402
import services.trading_engine as TE  # noqa: E402
from services.indicators import (calculate_indicators, closed_ema_gap_pct, closed_wilder_ndi,  # noqa: E402
                                 determine_macro_regime, last_closed_bar_ret_pct)

BAR, HOUR, DAY = 300_000, 3_600_000, 86_400_000
FEAT_VERSION = 2   # v2: breadth incl. BTC/ETH, forming-hour BTC 1h, engine seed lengths, W3 volume input
LEGACY_COLS = ("entry_pair_volume_ratio", "entry_global_volume_ratio", "entry_funding_rate", "entry_bear_r24",
               "entry_bear_below24", "entry_bear_eff24", "entry_bear_off24lo")   # v1 names of quantities that differ from the live stamp
TH = config.trading_config.thresholds


def _rows(df, t, n=None):
    """Exchange-style rows [open_ms, o, h, l, c, v] of the bars with open ≤ t (closed at the stamp), the last `n`."""
    if df is None or not len(df):
        return []
    d = df[df.index <= t]
    if n:
        d = d.tail(n)
    return [[int(i), float(r.o), float(r.h), float(r.l), float(r.c), float(r.v)] for i, r in zip(d.index, d.itertuples())]


def _closed_tf(df, t, tf_ms, n):
    """Higher-timeframe bars CLOSED by the stamp time (bar t closes at t + 5 min)."""
    if df is None or not len(df):
        return []
    return _rows(df[df.index + tf_ms <= t + BAR], t, n)


def _with_forming_hour(df1h, df5, t, n):
    """The engine's get_ohlcv('1h', n): n−1 closed hours + the hour still forming at the stamp, rebuilt from the 5m bars ≤ t
    (open of the hour, max high, min low, close at t, summed volume) — the forming-bar read with no look-ahead."""
    closed = _closed_tf(df1h, t, HOUR, n - 1)
    h0 = (t + BAR) // HOUR * HOUR
    part = df5[(df5.index >= h0) & (df5.index <= t)] if df5 is not None else None
    if part is not None and len(part):
        closed = closed + [[int(h0), float(part.o.iloc[0]), float(part.h.max()), float(part.l.min()), float(part.c.iloc[-1]),
                            float(part.v.sum())]]
    elif closed:                                                    # the hour opened exactly at the stamp: a flat just-opened bar
        c = closed[-1][4]; closed = closed + [[int(h0), c, c, c, c, 0.0]]
    return closed


def _fx(rows):
    """closed_* helpers drop the LAST row as the forming candle — hand them the closed rows plus a dummy."""
    return rows + [rows[-1]] if rows else rows


def _r(v, n=4):
    try:
        return None if v is None or v != v else round(float(v), n)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────── market (BTC / ETH / breadth) ───────────────────────────────
def btc_monitor(btc5, t):
    """The Bull-Run / Bear-Run monitor readings at t, with the engine's formulas (_update_bullrun_monitor: 1000-bar fetch → 999
    closed bars, W=864 / W24=288, EMA20 seeded from the window's first close)."""
    rows = _rows(btc5, t, 999)
    if len(rows) < 939:
        return {}
    closes = [r[4] for r in rows]
    W, W24 = 864, 288
    win = closes[-W:]
    ema = closes[0]; k = 2.0 / 21.0; above_flags = []
    for c in closes:
        ema = c * k + ema * (1 - k)
        above_flags.append(c > ema)
    diffs = sum(abs(win[i] - win[i - 1]) for i in range(1, W))
    w24 = closes[-W24:]
    d24 = sum(abs(w24[i] - w24[i - 1]) for i in range(1, W24))
    hi24 = max(r[2] for r in rows[-288:]); lo24 = min(r[3] for r in rows[-288:])
    return dict(r72=(win[-1] / win[0] - 1) * 100.0, above=100.0 * sum(above_flags[-W:]) / W,
                eff=(abs(win[-1] - win[0]) / diffs) if diffs > 0 else 0.0,
                off24h=(win[-1] / hi24 - 1) * 100.0, off24lo=(closes[-1] / lo24 - 1) * 100.0,
                r24=(w24[-1] / w24[0] - 1) * 100.0, below24=100.0 - 100.0 * sum(above_flags[-W24:]) / W24,
                eff24=(abs(w24[-1] - w24[0]) / d24) if d24 > 0 else 0.0, px=closes[-1])


def market_context(t, btc5, btc1h, btc4h, btc1d, eth5, universe5):
    """The scan-level state the engine publishes as module globals, rebuilt at t → (g dict for market_entry_stamps, extras)."""
    g, ex = {}, {}
    ind = calculate_indicators(_rows(btc5, t, 100)) if btc5 is not None else {}
    if ind:
        px, e13, e20, e20p3, e50 = ind.get('price'), ind.get('ema13'), ind.get('ema20'), ind.get('ema20_prev3'), ind.get('ema50')
        g.update(_current_btc_adx=ind.get('adx'), _current_btc_adx_prev=ind.get('adx_prev1'), _current_btc_rsi=ind.get('rsi'),
                 _current_btc_rsi_prev=ind.get('rsi_prev1'), _current_btc_rsi_prev6=ind.get('rsi_prev6'),
                 _current_btc_ema13=e13, _current_btc_price=px,
                 _current_btc_atr_pct=_r(ind['atr'] / px * 100) if (ind.get('atr') is not None and px) else None,
                 _btc_ema20_slope_pct_raw=_r((e20 - e20p3) / e20p3 * 100) if (e20 and e20p3) else None,
                 _current_btc_trend_gap_pct=_r((e13 - e50) / e50 * 100) if (e13 is not None and e50) else None)
        flat = min(float(getattr(TH, 'macro_trend_flat_threshold_long', TH.macro_trend_flat_threshold)),
                   float(getattr(TH, 'macro_trend_flat_threshold_short', TH.macro_trend_flat_threshold)))
        g['_current_btc_regime'] = determine_macro_regime(e20, e20p3, flat)
    h1 = _with_forming_hour(btc1h, btc5, t, 100)
    i1 = calculate_indicators(h1) if len(h1) >= 51 else {}
    if i1:
        g['_current_btc_rsi_1h'] = _r(i1.get('rsi'), 1); g['_current_btc_rsi_1h_prev'] = _r(i1.get('rsi_prev1'), 1)
        if i1.get('ema20') and i1.get('ema20_prev3'):
            g['_current_btc_1h_slope'] = _r((i1['ema20'] - i1['ema20_prev3']) / i1['ema20_prev3'] * 100)
    # breadth + global volume over the bot's universe (engine Phase 2: breadth flat threshold, per-pair volume sums)
    nb = nr = n = 0; vs = avs = 0.0
    for d in (universe5 or {}).values():
        pi = calculate_indicators(_rows(d, t, 100), pair_volume_bars=int(getattr(TH, 'pair_volume_lookback_bars', 20) or 20),
                                  global_volume_bars=int(getattr(TH, 'global_volume_lookback_bars', 48) or 48)) if d is not None else {}
        if not pi or pi.get('adx') is None or (pi.get('rsi') is not None and (pi['rsi'] >= 99.9 or pi['rsi'] <= 0.1)):
            continue                                                  # the scan's own skips (null ADX / degenerate RSI) before breadth
        n += 1
        reg = determine_macro_regime(pi.get('ema20'), pi.get('ema20_prev3'), float(getattr(TH, 'market_breadth_flat_threshold', 0.03)))
        nb += reg == "BULLISH"; nr += reg == "BEARISH"
        vs += pi.get('volume') or 0
        avs += (pi.get('avg_volume_global') or 0) if (pi.get('avg_volume_global') or 0) > 0 else (pi.get('avg_volume') or 0)
    if n:
        g['_market_bull_pct'] = round(nb / n * 100, 1); g['_market_bear_pct'] = round(nr / n * 100, 1)
    if avs > 0:
        ex['global_volume_ratio_closed'] = round(vs / avs, 4)   # NOT entry_global_volume_ratio: live reads a partial forming bar
    ex['n_breadth'] = n
    mon = btc_monitor(btc5, t)
    ex['mon'] = mon
    k30 = _closed_tf(btc1h, t, HOUR, 720)
    if mon and len(k30) >= 600:
        ex['off30d'] = round((mon['px'] / max(r[2] for r in k30) - 1) * 100.0, 2)
    ex['btc_ema50_100_gap_pct'] = closed_ema_gap_pct(_fx(_rows(btc5, t, 299)), 50, 100)
    ex['eth_5m_ret1_pct'] = last_closed_bar_ret_pct(_fx(_rows(eth5, t, 5)))
    ex['btc_1d_ret_pct'] = last_closed_bar_ret_pct(_fx(_closed_tf(btc1d, t, DAY, 4)))
    ex['btc_4h_ema50_200_gap_pct'] = closed_ema_gap_pct(_fx(_closed_tf(btc4h, t, 4 * HOUR, 999)), 50, 200)
    return g, ex


# ─────────────────────────────── one event ───────────────────────────────
def _pctile(series, value):
    s = pd.Series(series).dropna()
    return None if not len(s) or value is None or value != value else round(float((s <= value).mean() * 100), 1)


def _atr_series_pct(d):
    tr = pd.concat([d.h - d.l, (d.h - d.c.shift()).abs(), (d.l - d.c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100


def pre_move(d, t0, start, btc5, eth5, extras):
    """Pre-move features (not in the bot): measured on bars ≤ the move's START (compression / build-up) or ≤ the stamp bar."""
    out = {}
    if d is None or not len(d):
        return out
    h = d[d.index <= start]
    if len(h) >= 60:
        atr = _atr_series_pct(h)
        out['pre_atr_pct'] = _r(atr.iloc[-1]); out['pre_atr_pctile_24h'] = _pctile(atr.tail(288), atr.iloc[-1])
        w = h.tail(24)
        out['pre_range_2h_pct'] = _r((w.h.max() - w.l.min()) / w.c.iloc[-1] * 100)
        m20, s20 = h.c.rolling(20).mean(), h.c.rolling(20).std(ddof=0)
        bbw = 4 * s20 / m20 * 100
        out['pre_bbw_pct'] = _r(bbw.iloc[-1]); out['pre_bbw_pctile_24h'] = _pctile(bbw.tail(288), bbw.iloc[-1])
        med = float(h.v.tail(288).median())
        out['pre_vol_build_1h'] = _r(float(h.v.tail(12).mean()) / med, 3) if med > 0 else None
        if 'tb' in h and h.tb.notna().tail(12).all():
            v12 = float(h.v.tail(12).sum())
            out['pre_taker_buy_1h'] = _r(float(h.tb.tail(12).sum()) / v12 * 100, 1) if v12 > 0 else None
        out['pre_ret_4h_pct'] = _r((h.c.iloc[-1] / h.c.iloc[-49] - 1) * 100) if len(h) >= 49 else None
        out['pre_ret_24h_pct'] = _r((h.c.iloc[-1] / h.c.iloc[-289] - 1) * 100) if len(h) >= 289 else None
    a = d[d.index <= t0]
    if len(a) >= 3 and 'tb' in a and a.tb.notna().tail(3).all():
        v3 = float(a.v.tail(3).sum())
        out['pre_taker_buy_at_entry'] = _r(float(a.tb.tail(3).sum()) / v3 * 100, 1) if v3 > 0 else None
    for name, ref in (("btc", btc5), ("eth", eth5)):          # lead/lag: the reference's 30-min return before the move started
        if ref is not None and start in ref.index and start - 6 * BAR in ref.index:
            out[f'pre_{name}_r30_before_start'] = _r((ref.c.loc[start] / ref.c.loc[start - 6 * BAR] - 1) * 100)
    if start in d.index and start - 6 * BAR in d.index and out.get('pre_btc_r30_before_start') is not None:
        own = (d.c.loc[start] / d.c.loc[start - 6 * BAR] - 1) * 100
        out['pre_r30_before_start_vs_btc'] = _r(own - out['pre_btc_r30_before_start'])
    k1h = extras.get('k1h')
    if k1h is not None and len(k1h):
        c1 = k1h[k1h.index + HOUR <= t0 + BAR]
        px = float(a.c.iloc[-1]) if len(a) else None
        cur = a[a.index >= (c1.index[-1] + HOUR if len(c1) else t0 + BAR)]   # the hour still forming at the stamp, from 5m bars
        for days in (3, 7):
            w = c1.tail(days * 24)
            if px and len(w) >= days * 24 * 0.9:
                hi = max(float(w.h.max()), float(cur.h.max()) if len(cur) else 0.0)
                lo = min(float(w.l.min()), float(cur.l.min()) if len(cur) else float("inf"))
                out[f'pre_off_{days}d_high_pct'] = _r((px / hi - 1) * 100)
                out[f'pre_off_{days}d_low_pct'] = _r((px / lo - 1) * 100)
    oi = extras.get('oi')
    if oi is not None and len(oi):
        o = oi[oi.index <= t0]
        if len(o) and t0 - o.index[-1] <= 15 * 60_000:                 # a stale snapshot is never used
            now_v = float(o.iloc[-1])
            for hh in (1, 4):
                p = o[o.index <= o.index[-1] - hh * HOUR]
                out[f'pre_oi_chg_{hh}h_pct'] = _r((now_v / float(p.iloc[-1]) - 1) * 100) if len(p) and float(p.iloc[-1]) > 0 else None
    ts = pd.Timestamp(t0 + BAR, unit="ms")
    out['pre_hour_utc'] = int(ts.hour); out['pre_weekday'] = int(ts.weekday())
    return out


def event_features(ev, alts, mctx_cache, market, extras_for):
    """All stamps for one event row (dict with type/pair/side/bar_ts/start_ts[/rank/qvol24_event]). `market` = (btc5, btc1h, btc4h,
    btc1d, eth5, universe5); `mctx_cache` memoises the market context per bar; `extras_for(pair)` → {'k1h','k1d','oi','funding'}."""
    t0 = int(ev["bar_ts"]); start = int(ev["start_ts"]) if pd.notna(ev.get("start_ts")) else t0
    direction = "LONG" if ev["side"] == "UP" else "SHORT"
    if t0 not in mctx_cache:
        mctx_cache[t0] = market_context(t0, *market)
    g, mx = mctx_cache[t0]
    btc5, _, _, _, eth5, _ = market
    st = {"feat_v": FEAT_VERSION, "feat_bar_ts": t0}
    pair_ev = ev["pair"] not in ("BTCUSDT", "UNIVERSE")
    d = alts.get(ev["pair"]) if pair_ev else None
    ind = None
    rows100 = _rows(d, t0, 100) if (d is not None and t0 in d.index) else []
    if len(rows100) == 100:                                            # the scan's 100-bar fetch (EMA/ADX seeding)
        ind = calculate_indicators(rows100, pair_volume_bars=int(getattr(TH, 'pair_volume_lookback_bars', 20) or 20),
                                   global_volume_bars=int(getattr(TH, 'global_volume_lookback_bars', 48) or 48))
        if ind:
            st.update(TE.pair_entry_stamps(ind, direction))
            st['pre_pair_volume_ratio_closed'] = st.pop('entry_pair_volume_ratio', None)   # live = partial forming bar → own name
    st.update(TE.market_entry_stamps(g, direction, ind, bool(getattr(TH, 'btc_global_filter_enabled', False)), TH))
    st.pop('entry_global_volume_ratio', None)
    st['pre_global_volume_ratio_closed'] = mx.get('global_volume_ratio_closed')
    if ind:
        st['entry_quality_score'] = TE._calculate_quality_score(direction, ind.get('rsi'), ind.get('adx'), st.get('entry_gap'),
                                                                g.get('_market_bull_pct'), g.get('_market_bear_pct'),
                                                                g.get('_current_btc_adx'), st.get('entry_ema20_slope'))
    mon = mx.get('mon') or {}
    if mon:
        st.update(entry_btc_off24h_pct=_r(mon['off24h'], 2), entry_btc_off24lo_pct=_r(mon['off24lo'], 2),
                  entry_btc_r72_pct=_r(mon['r72'], 2), entry_btc_eff72=int(mon['eff'] * 1000) / 1000.0,
                  entry_btc_above72_pct=_r(mon['above'], 1),
                  pre_bear_r24=_r(mon['r24'], 2), pre_bear_below24=_r(mon['below24'], 1), pre_bear_eff24=_r(mon['eff24'], 3))
    st['entry_btc_off30d_high_pct'] = mx.get('off30d')
    try:
        st['entry_long_heat_flags'] = TE.long_heat_eval(TH, st.get('entry_btc_ema20_slope'), st.get('entry_btc_rsi_prev'),
                                                        st.get('entry_bull_pct'), mx.get('off30d'))[0]
    except Exception:
        st['entry_long_heat_flags'] = None
    for k in ('btc_ema50_100_gap_pct', 'eth_5m_ret1_pct', 'btc_1d_ret_pct', 'btc_4h_ema50_200_gap_pct'):
        st['entry_' + k] = mx.get(k)
    if pair_ev:
        st['entry_pair_rank'] = ev.get("rank"); st['entry_pair_volume_24h_usd'] = ev.get("qvol24_event")
        ext = extras_for(ev["pair"]) or {}
        k1h, k1d = ext.get('k1h'), ext.get('k1d')
        st['entry_pair_1h_ema20_200_gap_pct'] = closed_ema_gap_pct(_fx(_closed_tf(k1h, t0, HOUR, 259)), 20, 200)
        st['entry_pair_1d_ndi'] = closed_wilder_ndi(_fx(_closed_tf(k1d, t0, DAY, 199)))
        fr = ext.get('funding')
        if fr is not None and len(fr):                                 # last SETTLED rate (live stamps the predicted one → own name)
            f = fr[fr.index <= t0]
            st['pre_funding_settled'] = _r(float(f.iloc[-1]), 6) if (len(f) and t0 - f.index[-1] <= 9 * HOUR) else None
        if ind:
            bg = g.get('_current_btc_trend_gap_pct')
            try:
                (st['entry_pattern_c1_match'], st['entry_pattern_c2_match'], st['entry_pattern_c3_match'], st['entry_pattern_c4_match'],
                 st['entry_pattern_c5_match'], st['entry_pattern_c6_match'], st['entry_pattern_c7_match'], st['entry_pattern_c8_match'],
                 st['entry_pattern_c9_match'], st['entry_pattern_c_any_match']) = TE._compute_pattern_c_match(
                    direction=direction, rng_pos=st.get('entry_range_position'), pair_gap=st.get('entry_pair_ema20_ema50_gap_pct'),
                    adx_delta=st.get('entry_adx_delta'), btc_rsi=st.get('entry_btc_rsi'), btc_rsi_prev=st.get('entry_btc_rsi_prev'),
                    btc_adx=st.get('entry_btc_adx'), btc_adx_prev=st.get('entry_btc_adx_prev'), btc_gap=bg,
                    stretch=st.get('entry_ema5_stretch'), pair_adx=st.get('entry_adx'), btc_atr=st.get('entry_btc_atr_pct'),
                    ema20_slope=st.get('entry_ema20_slope'), ema50_slope=st.get('entry_ema50_slope'))
                (st['entry_pattern_w1_match'], st['entry_pattern_w2_match'], st['entry_pattern_w3_match'], st['entry_pattern_w4_match'],
                 st['entry_pattern_w5_match'], st['entry_pattern_w6_match'], st['entry_pattern_w_any_match']) = TE._compute_pattern_w_match(
                    direction=direction, rsi=st.get('entry_rsi'), adx=st.get('entry_adx'), adx_delta=st.get('entry_adx_delta'),
                    stretch=st.get('entry_ema5_stretch'), rng_pos=st.get('entry_range_position'),
                    pair_gap=st.get('entry_pair_ema20_ema50_gap_pct'), btc_rsi=st.get('entry_btc_rsi'), btc_adx=st.get('entry_btc_adx'),
                    btc_atr=st.get('entry_btc_atr_pct'), btc_gap=bg, pair_vol_ratio=st.get('pre_pair_volume_ratio_closed'))   # closed-bar ratio (live: forming bar)
            except Exception:
                pass
        st.update(pre_move(d, t0, start, btc5, eth5, ext))
    else:
        st.update(pre_move(btc5, t0, start, btc5, eth5, {}) if ev["pair"] == "BTCUSDT" else
                  {'pre_hour_utc': int(pd.Timestamp(t0 + BAR, unit="ms").hour), 'pre_weekday': int(pd.Timestamp(t0 + BAR, unit="ms").weekday())})
    st['feat_n_breadth'] = mx.get('n_breadth')
    return st


# ─────────────────────────────── public reads (fail-soft) ───────────────────────────────
def fetch_pair_extras(ex, retry, pair, since_ms):   # `retry` kept for the call signature; each read is tried ONCE (a pair
    # whose OI endpoint errors must not burn 3 retries × 4 reads every hour)
    """Pair 1h (≥ 10 days, for the 1h EMA20/200 gap and 3/7-day extremes), 1d (200, daily −DI), open interest (5m) and settled
    funding since `since_ms`. Every piece independent: a failed read → that key missing (its stamps stay empty)."""
    sym = pair[:-4] + "/USDT:USDT"; out = {}

    def frame(rows):
        return pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t").set_index("t") if rows else None
    try:
        out['k1h'] = frame(ex.fetch_ohlcv(sym, "1h", limit=500))
    except Exception:
        pass
    try:
        out['k1d'] = frame(ex.fetch_ohlcv(sym, "1d", limit=210))
    except Exception:
        pass
    try:
        oi = ex.fetch_open_interest_history(sym, "5m", limit=500)   # latest ~41.7 h ⊇ the 40 h stamp window or []
        s = pd.Series({int(x["timestamp"]): float(x.get("openInterestAmount") or np.nan) for x in oi}).dropna().sort_index()
        out['oi'] = s if len(s) else None
    except Exception:
        pass
    try:
        fr = ex.fetch_funding_rate_history(sym, since=int(since_ms - 2 * DAY), limit=100) or []
        s = pd.Series({int(x["timestamp"]): float(x["fundingRate"]) for x in fr if x.get("fundingRate") is not None}).sort_index()
        out['funding'] = s if len(s) else None
    except Exception:
        pass
    return out
