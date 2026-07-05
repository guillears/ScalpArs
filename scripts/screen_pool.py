#!/usr/bin/env python3
"""
screen_pool.py — build the CANONICAL screened baseline from the raw COMBINED pool,
using the REAL engine filter functions (no hand-approximation).

WHY THIS EXISTS: every flip/short cross-batch analysis was repeatedly done on the RAW
(unscreened) pool, producing false signals (the "QS floor->3" mirage, the stale
46/74%/+$1187 reference, etc.). This script applies the LIVE filter stack ONCE and
freezes the survivors to reports/SCREENED_BASELINE.csv. Analyse THAT file — never the
raw 269-flip pool.

RE-RUN WHEN: a new batch is appended to the raw pool, OR any filter/config changes.
Then re-freeze and re-pin the checksum in CLAUDE_CURRENT_STATE.md.

VALIDATION ANCHOR (asserted below): MOM-long = 23 · 83% · +$2146 (independently exact).
Flip-short uses the real services.trading_engine._flip_filters with a field-audited `ind`.
"""
import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from services.trading_engine import _flip_filters

RAW = "reports/COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv"  # Jul 3: now spans 06-16..07-03 (batch appended; filename kept — all tooling points here)
OUT = "reports/SCREENED_BASELINE.csv"
th = config.trading_config.thresholds
BL = set("ALLOUSDT,BNBUSDT,EIGENUSDT,ENAUSDT,ESPORTSUSDT,FILUSDT,MUSDT,RAVEUSDT,SYNUSDT,TRUMPUSDT,VELVETUSDT,VVVUSDT,XAGUSDT,XAUUSDT,ZECUSDT".split(","))

def nf(x):
    try: return float(x)
    except: return None
def isflip(r): return 'FLIP' in (r.get('cell_multiplier_source') or r.get('entry_strategy') or '').upper()

def pnl_current(r):
    """Realized $ under the CURRENT sizing. The COMBINED pool's `pnl` reflects HISTORICAL
    multipliers; the only cell whose sizing CHANGED this era is C1 SHORT (de-muxed 2x->1x,
    2026-06-29). Unmatched-long (2x) and W2+W1 (2x) are unchanged. Without this, MOM-short
    shows -$332 (C1 at 2x) instead of the true -$122 (C1 at 1x). Memory: 'cross-batch re-sims
    MUST apply C1 demux.'"""
    p = nf(r.get('pnl')) or 0.0
    src_ = r.get('cell_multiplier_source') or ''
    mult = nf(r.get('cell_multiplier')) or 1.0
    if src_ == 'C1' and mult == 2.0:
        return p / 2.0
    # Jul 3 — flips run 2x live now (FAN_RATIO_GATE:2.0); the pinned flip baseline is at 1x, so
    # de-mux any multiplied FLIP row to 1x when future batches land (else 1x-historical mixes with
    # 2x-fresh and the flip net$ silently drifts — review I2; the FLIP anchor is count+$ so a miss fails the freeze).
    if src_.startswith('FLIP') and mult > 1.0:
        return p / mult
    return p

def flip_ind(r):  # field-audited against engine _ff_in (trading_engine.py:3414)
    a, b = nf(r.get('entry_ema_gap_5_8')), nf(r.get('entry_ema_gap_8_13'))
    return {'flip_dir':'SHORT','btc_regime':r.get('entry_btc_regime'),'adx':nf(r.get('entry_adx')),
        'pair_gap':nf(r.get('entry_pair_ema20_ema50_gap_pct')),'fan_ratio':(abs(a/b) if (a is not None and b) else None),
        'ema5_stretch':nf(r.get('entry_ema5_stretch')),'adx_delta':nf(r.get('entry_adx_delta')),
        'btc_rsi':nf(r.get('entry_btc_rsi')),'btc_rsi_prev6':nf(r.get('entry_btc_rsi_prev6')),
        'btc_adx':nf(r.get('entry_btc_adx')),'btc_atr_pct':nf(r.get('entry_btc_atr_pct')),
        'atr_pct':nf(r.get('entry_atr_pct')),'pair_rsi':nf(r.get('entry_rsi')),
        'quality_score':nf(r.get('entry_quality_score')),'bear_pct':nf(r.get('entry_bear_pct')),
        'range_position':nf(r.get('entry_range_position')),
        # Jul 3 — required by flip_short_btc_1h_slope_max (the BTC-1h regime gate); missing = fail-open
        'btc_1h_slope':nf(r.get('entry_btc_1h_slope'))}

def sleeve(r):
    """Return 'MOM_LONG' / 'MOM_SHORT' / 'FLIP_SHORT' if the row SURVIVES the current stack, else None."""
    if r.get('status') != 'CLOSED' or r['pair'] in BL: return None
    d = r['direction']
    # --- MOM-long: keep only unmatched longs (long_unmatched_only) ---
    if d == 'LONG' and not isflip(r):
        if r.get('pattern_cell_source') != 'UNMATCHED': return None
        # Jul 5 — LONG_BTC1H_DEADBAND parity: block when |BTC 1h slope| < long_btc_1h_deadband
        # (flat hourly = no carry / DOA). Missing slope = fail-open, mirrors the engine gate.
        _db = float(getattr(th, 'long_btc_1h_deadband', 0.0) or 0.0)
        _s1h = nf(r.get('entry_btc_1h_slope'))
        if _db > 0 and _s1h is not None and abs(_s1h) < _db: return None  # LONG_BTC1H_DEADBAND
        return 'MOM_LONG'
    # --- FLIP-short: the REAL engine flip filter ---
    if d == 'SHORT' and isflip(r):
        blocked, *_ = _flip_filters('FAN_RATIO_GATE', flip_ind(r))
        return None if blocked else 'FLIP_SHORT'
    # --- MOM-short: current momentum-short filters (Jun-28 BTC-ATR floor + RSI band + pair-ADX rising) ---
    if d == 'SHORT' and not isflip(r):
        batr = nf(r.get('entry_btc_atr_pct')); rsi = nf(r.get('entry_rsi'))
        adx, adxp = nf(r.get('entry_adx')), nf(r.get('entry_adx_prev'))
        if batr is not None and batr < float(getattr(th, 'momentum_short_btc_atr_min', 0.0) or 0.0): return None  # MOMENTUM_SHORT_LOATR
        if rsi is not None and not (float(getattr(th,'momentum_short_rsi_min',0)) <= rsi <= float(getattr(th,'momentum_short_rsi_max',100))): return None
        if adx is not None and adxp is not None and adx <= adxp: return None  # Pair ADX Dir S: rising
        # Jun 30: W1-regime block — current-stack parity with engine open_position
        # (momentum_short_w1_block_regimes). W1 mom-short drains in HEALTHY_BEAR (N=20/40%/-$650),
        # non-W1 control breakeven+ → block W1 in the listed regimes (STRONG_BEAR W1 exempt = wins).
        _w1blk = {s.strip() for s in (getattr(th, 'momentum_short_w1_block_regimes', '') or '').split(',') if s.strip()}
        if (r.get('entry_pattern_w1_match') or '').strip().lower() == 'true' and (r.get('entry_btc_regime') in _w1blk): return None  # MOMENTUM_SHORT_W1_REGIME (reverted → empty = off)
        # Jun 30: high-pair-volume block (momentum_short_pair_vol_max) — current-stack parity. Shorting into high
        # pair vol = climactic/exhaustion → bounce; the one separator robust across both periods.
        _pvmax = float(getattr(th, 'momentum_short_pair_vol_max', 0.0) or 0.0)
        _pv = nf(r.get('entry_pair_volume_ratio'))
        if _pvmax > 0 and _pv is not None and _pv >= _pvmax: return None  # MOMENTUM_SHORT_PAIRVOL
        # Jul 1: weak-capitulation block (momentum_short_weakcap_*, engine-live since Jun 28 but MISSING here
        # until now — parity gap caught by the Jul-1 deep dive: TAO 06-24 survived the screen yet the live
        # engine would block it). Block when range<15 AND pairATR<0.45 AND pairADX<28 = shorting a dead tape
        # at the bottom of its range with no trend strength.
        if getattr(th, 'momentum_short_weakcap_enabled', False):
            _wc_r = nf(r.get('entry_range_position')); _wc_a = nf(r.get('entry_atr_pct')); _wc_x = nf(r.get('entry_adx'))
            if (_wc_r is not None and _wc_a is not None and _wc_x is not None
                    and _wc_r < float(getattr(th, 'momentum_short_weakcap_range_max', 0.0) or 0.0)
                    and _wc_a < float(getattr(th, 'momentum_short_weakcap_atr_max', 0.0) or 0.0)
                    and _wc_x < float(getattr(th, 'momentum_short_weakcap_padx_max', 0.0) or 0.0)):
                return None  # MOMENTUM_SHORT_WEAKCAP
        return 'MOM_SHORT'
    return None

def main():
    rows = list(csv.DictReader(open(RAW)))
    kept = []
    agg = {}
    for r in rows:
        s = sleeve(r)
        if not s: continue
        r['screen_sleeve'] = s
        kept.append(r)
        agg.setdefault(s, []).append(r)
    # report + validate
    print(f"Raw rows: {len(rows)}  ->  screened survivors: {len(kept)}\n")
    print(f"{'sleeve':12}{'N':>4}{'WR%':>6}{'net$':>9}")
    order = ['MOM_LONG','MOM_SHORT','FLIP_SHORT']
    tot_n=tot_w=tot_d=0
    for s in order:
        rs = agg.get(s, [])
        if not rs: print(f"{s:12} N=0"); continue
        pls=[nf(x.get('pnl_percentage')) for x in rs]; d=[pnl_current(x) for x in rs]  # de-muxed (C1 1x)
        w=sum(1 for p in pls if p>0)
        print(f"{s:12}{len(rs):>4}{w/len(rs)*100:>5.0f}%{sum(d):>+9.0f}")
        tot_n+=len(rs); tot_w+=w; tot_d+=sum(d)
    print(f"{'TOTAL':12}{tot_n:>4}{tot_w/tot_n*100:>5.0f}%{tot_d:>+9.0f}   (net$ = current sizing, C1 de-muxed to 1x)")
    # hard validation anchors (both must hold — these are the verified current-stack truth)
    ml = agg.get('MOM_LONG', []);  ms = agg.get('MOM_SHORT', [])
    ml_net = sum(pnl_current(x) for x in ml);  ms_net = sum(pnl_current(x) for x in ms)
    # Anchors updated 2026-07-05 (v5): long_btc_1h_deadband=0.05 shipped (flat-1h DOA block) —
    # screens out 7 flat-zone MLs (4L/3W, -$256): ML 34->27/$2708->$2964. MS/FLIP unchanged.
    # v4 (2026-07-03): ① Jun30-Jul3 batch (27 trades) appended; ② flip_short_btc_1h_slope_max=0
    # shipped — screens out 26 slope>0 flips. ML 25->34/$2496->$2708, MS 15->17/$443->$661,
    # FLIP 46->36/$825 (1x — incl. de-mux of the Jun-17/18 legacy 2x flips, review I2).
    # v3 (2026-07-01): weakcap parity closed (MOM-short 16/$392 -> 15/$443).
    # Prior anchors archived in DECISION_LOG: 23/$2146+28/-$122 (pre-batch), 25/$2496+14/$310 (W1-block v1),
    # 25/$2496+16/$392 (v2), 25/$2496+15/$443 (v3).
    _pvmax = float(getattr(th, 'momentum_short_pair_vol_max', 0.0) or 0.0)
    _pv_surv = sum(1 for r in ms if _pvmax > 0 and nf(r.get('entry_pair_volume_ratio')) is not None and nf(r.get('entry_pair_volume_ratio')) >= _pvmax)
    assert _pv_surv == 0, f"FAIL: {_pv_surv} pair_vol>={_pvmax} mom-shorts survived — vol block not applied, NOT freezing"
    assert len(ml) == 27 and round(ml_net) == 2964, f"FAIL: MOM-long {len(ml)}/${ml_net:.0f} != 27/$2964 — 1h deadband applied? screen wrong, NOT freezing"
    assert len(ms) == 17 and round(ms_net) == 661, f"FAIL: MOM-short {len(ms)}/${ms_net:.0f} != 17/$661 (C1 de-mux + pair-vol + weakcap?) — NOT freezing"
    fl = agg.get('FLIP_SHORT', [])
    fl_net = sum(pnl_current(x) for x in fl)
    assert len(fl) == 36 and round(fl_net) == 825, f"FAIL: FLIP-short {len(fl)}/${fl_net:.0f} != 36/$825 — btc_1h_slope gate or flip de-mux not applied? NOT freezing"
    print("\n✅ VALIDATION PASSED (MOM-long 27/$2964 + MOM-short 17/$661 + FLIP 36 + 0 pair-vol survivors). Freezing.")
    # freeze — add a de-muxed P&L column so downstream analysis uses current-sizing $ directly
    cols = list(rows[0].keys()) + ['screen_sleeve', 'pnl_current_sizing']
    with open(OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in kept:
            r['pnl_current_sizing'] = round(pnl_current(r), 4)
            w.writerow(r)
    print(f"Wrote {len(kept)} screened rows -> {OUT}")

if __name__ == '__main__':
    main()
