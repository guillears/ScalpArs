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

RAW = "reports/COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv"
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
    if (r.get('cell_multiplier_source') or '') == 'C1' and (nf(r.get('cell_multiplier')) or 1.0) == 2.0:
        return p / 2.0
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
        'range_position':nf(r.get('entry_range_position'))}

def sleeve(r):
    """Return 'MOM_LONG' / 'MOM_SHORT' / 'FLIP_SHORT' if the row SURVIVES the current stack, else None."""
    if r.get('status') != 'CLOSED' or r['pair'] in BL: return None
    d = r['direction']
    # --- MOM-long: keep only unmatched longs (long_unmatched_only) ---
    if d == 'LONG' and not isflip(r):
        return 'MOM_LONG' if r.get('pattern_cell_source') == 'UNMATCHED' else None
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
    assert len(ml) == 23 and round(ml_net) == 2146, f"FAIL: MOM-long {len(ml)}/${ml_net:.0f} != 23/$2146 — screen wrong, NOT freezing"
    assert len(ms) == 28 and round(ms_net) == -122, f"FAIL: MOM-short {len(ms)}/${ms_net:.0f} != 28/-$122 (did you forget the C1 de-mux?) — NOT freezing"
    print("\n✅ VALIDATION PASSED (MOM-long 23/$2146 + MOM-short 28/-$122). Freezing.")
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
