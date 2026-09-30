"""🧬 Sep-30 — scout feature stamps (scripts/scout_features.py, DECISION_LOG 152): the bot's own entry_* columns + pre_* move
features at an event's anchor bar. Pinned: NO LOOK-AHEAD (bars after the stamp bar never change a stamp), the Bull/Bear-Run
monitor formulas, and fail-soft on missing data."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import scout_features as SF  # noqa: E402

BAR = SF.BAR


def _walk(n, seed, t_end, step=BAR, start=100.0):
    """n bars ending at t_end; the SAME bar timestamp always gets the same values whatever n is (bars are drawn per timestamp)."""
    idx = t_end - step * np.arange(n)[::-1]
    r = np.array([np.random.default_rng([seed, int(t // step)]).random(3) for t in idx])
    c = start * np.exp(np.cumsum((r[:, 0] - 0.5) * 0.006))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"o": o, "h": np.maximum(o, c) * 1.001, "l": np.minimum(o, c) * 0.999, "c": c,
                         "v": 100 + 100 * r[:, 1], "tb": 40 + 80 * r[:, 2]}, index=idx)


T_END = 1_790_000_000_000 // BAR * BAR


def _market(extra_after=0):
    """Every frame — 5m, BTC 1h/4h/1d, ETH, universe, pair extras — extends `extra_after` bars into the FUTURE of the same data."""
    e5 = extra_after * BAR
    btc = _walk(1500 + extra_after, 1, T_END + e5)
    alt = _walk(640 + extra_after, 2, T_END + e5, start=5.0)
    hr = lambda step, n, seed, start=100.0: _walk(n + extra_after, seed, (T_END // step) * step - step + extra_after * step, step=step, start=start)
    uni = {f"P{i}USDT": _walk(640 + extra_after, 10 + i, T_END + e5) for i in range(5)}
    k1h, k1d = hr(SF.HOUR, 500, 20, 5.0), hr(SF.DAY, 210, 21, 5.0)
    oi_idx = T_END + e5 - BAR * np.arange(500 + extra_after)[::-1]
    oi = pd.Series(1000 + (oi_idx // BAR % 97), index=oi_idx, dtype=float)
    fr_idx = (T_END // (8 * SF.HOUR)) * 8 * SF.HOUR + extra_after * 8 * SF.HOUR - 8 * SF.HOUR * np.arange(10 + extra_after)[::-1]
    fr = pd.Series((fr_idx // SF.HOUR % 7) * 1e-5, index=fr_idx, dtype=float)
    extras = {"k1h": k1h, "k1d": k1d, "oi": oi, "funding": fr}
    market = (btc, hr(SF.HOUR, 1000, 3), hr(4 * SF.HOUR, 1000, 5), hr(SF.DAY, 10, 6), _walk(640 + extra_after, 4, T_END + e5), uni)
    return btc, alt, market, extras


def _stamp(extra_after=0, extras=True):
    btc, alt, market, ext = _market(extra_after)
    t0 = T_END - 60 * BAR
    ev = dict(type="ALT_SPIKE", pair="QNTUSDT", side="UP", bar_ts=t0, start_ts=t0 - 6 * BAR, rank=12, qvol24_event=9e7)
    return SF.event_features(ev, {"QNTUSDT": alt}, {}, market, lambda p: ext if extras else {})


def _cut(obj, t0, tf=None):
    """What existed at the stamp: 5m bars ≤ t0, higher-timeframe bars CLOSED by t0 + 5 min, snapshots ≤ t0."""
    if obj is None:
        return None
    if tf is None:
        return obj[obj.index <= t0]
    return obj[obj.index + tf <= t0 + BAR]


def _stamp_frames(market, alt, ext, t0):
    ev = dict(type="ALT_SPIKE", pair="QNTUSDT", side="UP", bar_ts=t0, start_ts=t0 - 6 * BAR, rank=12, qvol24_event=9e7)
    return SF.event_features(ev, {"QNTUSDT": alt}, {}, market, lambda p: ext)


def _same(a, b):
    return [k for k in set(a) | set(b) if not (a.get(k) == b.get(k) or (isinstance(a.get(k), float) and isinstance(b.get(k), float)
                                                                          and np.isnan(a[k]) and np.isnan(b[k])))]


def test_no_look_ahead():
    """Stamps from the FULL data (60 bars of future after t0) == stamps from the data CUT at t0 (what existed then). Any read past
    the stamp — even one 5m bar — changes a value or empties it, so this fails."""
    btc, alt, market, ext = _market(0)
    t0 = T_END - 60 * BAR
    full = _stamp_frames(market, alt, ext, t0)
    b5, h1, h4, d1, eth, uni = market
    cut_market = (_cut(b5, t0), _cut(h1, t0, SF.HOUR), _cut(h4, t0, 4 * SF.HOUR), _cut(d1, t0, SF.DAY), _cut(eth, t0),
                  {k: _cut(v, t0) for k, v in uni.items()})
    cut_ext = {"k1h": _cut(ext["k1h"], t0, SF.HOUR), "k1d": _cut(ext["k1d"], t0, SF.DAY),
               "oi": ext["oi"][ext["oi"].index <= t0], "funding": ext["funding"][ext["funding"].index <= t0]}
    cut = _stamp_frames(cut_market, _cut(alt, t0), cut_ext, t0)
    assert _same(full, cut) == []
    # mutation guard: cutting ONE bar earlier must change something (the test really sees the stamp bar)
    early = _stamp_frames(cut_market, _cut(alt, t0 - BAR), cut_ext, t0)
    assert _same(full, early) != []


def test_bot_columns_present_and_fail_soft():
    st = _stamp()
    for c in ("entry_rsi", "entry_adx", "entry_gap", "entry_btc_rsi", "entry_btc_r72_pct", "entry_bull_pct", "entry_quality_score",
              "entry_pattern_c1_match", "entry_pattern_w1_match", "entry_btc_rsi_1h", "entry_pair_1d_ndi", "entry_btc_4h_ema50_200_gap_pct",
              "pre_atr_pctile_24h", "pre_taker_buy_1h", "pre_btc_r30_before_start", "pre_off_7d_high_pct", "pre_funding_settled",
              "pre_oi_chg_1h_pct", "pre_pair_volume_ratio_closed", "pre_global_volume_ratio_closed"):
        assert st.get(c) is not None, c
    for c in ("entry_pair_volume_ratio", "entry_global_volume_ratio", "entry_funding_rate", "entry_bear_r24"):
        assert c not in st, c                                          # different quantity from the live stamp → never under its name
    st = _stamp(extras=False)
    assert st["entry_pair_1d_ndi"] is None and "pre_oi_chg_1h_pct" not in st      # no extras → empty, never guessed
    assert st["feat_bar_ts"] == T_END - 60 * BAR and st["entry_pair_rank"] == 12


def test_monitor_matches_engine_formula():
    """A literal copy of the engine's _update_bullrun_monitor arithmetic (k5 = 1000 fetched bars, k5[:-1] closed)."""
    btc = _walk(1500, 7, T_END)
    m = SF.btc_monitor(btc, T_END)
    k5 = [[int(i), r.o, r.h, r.l, r.c, r.v] for i, r in zip(btc.index, btc.itertuples())][-999:] + [[0, 0, 0, 0, 0, 0]]
    closes = [float(r[4]) for r in k5[:-1]]
    W = 864; win = closes[-W:]
    ema = closes[0]; _k = 2.0 / 21.0; above_flags = []
    for c in closes:
        ema = c * _k + ema * (1 - _k); above_flags.append(c > ema)
    diffs = sum(abs(win[i] - win[i - 1]) for i in range(1, W))
    W24 = 288; w24 = closes[-W24:]; d24 = sum(abs(w24[i] - w24[i - 1]) for i in range(1, W24))
    want = dict(r72=(win[-1] / win[0] - 1) * 100.0, above=100.0 * sum(above_flags[-W:]) / W, eff=abs(win[-1] - win[0]) / diffs,
                off24h=(win[-1] / max(float(r[2]) for r in k5[-289:-1]) - 1) * 100.0,
                off24lo=(closes[-1] / min(float(r[3]) for r in k5[-289:-1]) - 1) * 100.0,
                r24=(w24[-1] / w24[0] - 1) * 100.0, below24=100.0 - 100.0 * sum(above_flags[-W24:]) / W24, eff24=abs(w24[-1] - w24[0]) / d24)
    for k, v in want.items():
        assert abs(m[k] - v) < 1e-9, k
    assert SF.btc_monitor(btc.head(500), T_END) == {}                 # too short → no reading
