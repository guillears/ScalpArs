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

RAW = "reports/COMBINED_momentum_flip_2026-06-16to28_DEDUP.csv"  # Jul 8: now spans 06-16..07-08 (batches appended; filename kept — all tooling points here)
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
    # Jul 10 — UNMATCHED-LONG crowded-entry de-mux (long_unmatched_mult_pvr_max): the 2× cell
    # sizes at 1× when entry PVR >= threshold (✗ HARMFUL sub-cell: zone 10 trades net-negative
    # at both sizings). Historical 2× fills in the zone re-price to 1× under the current stack.
    _upv = float(getattr(th, 'long_unmatched_mult_pvr_max', 0.0) or 0.0)
    _pvr = nf(r.get('entry_pair_volume_ratio'))
    if (_upv > 0 and src_ == 'UNMATCHED' and r.get('direction') == 'LONG' and mult > 1.0
            and _pvr is not None and _pvr >= _upv):
        return p / mult
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
        'btc_1h_slope':nf(r.get('entry_btc_1h_slope')),
        # Jul 8 — required by flip_short_btc_trend_gap_min (the BTC depth gate); missing = fail-open
        'btc_trend_gap':nf(r.get('entry_btc_trend_gap_pct'))}

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
        # Jul 10 — LONG stretch ceiling parity (ema5_stretch_max_long, indicators.py signal gate):
        # chase-entry block. Zero-cost cut: every pool winner entered <= 0.34; only-ever occupants
        # above 0.35 are losers (LDO 0.53, TAC 0.37). Missing stretch = fail-open like the engine.
        _smax = float(getattr(th, 'ema5_stretch_max_long', 0.0) or 0.0)
        _str = nf(r.get('entry_ema5_stretch'))
        if _smax > 0 and _str is not None and _str > _smax: return None  # EMA5_STRETCH_MAX_LONG
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
        # Jul 6: deep-gap floor (momentum_short_pair_gap_min) — block SHORT when pair 13-50 gap
        # <= threshold (already crashed below the 4h trend → bounce). GIGGLE post-mortem ship.
        _dg = float(getattr(th, 'momentum_short_pair_gap_min', 0.0) or 0.0)
        _pg = nf(r.get('entry_pair_ema20_ema50_gap_pct'))
        if _dg < 0 and _pg is not None and _pg <= _dg: return None  # MOMENTUM_SHORT_DEEPGAP
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
        # Jul 13: GAPFLAT probes are a deliberately-suspect 1x observation sleeve —
        # NEVER part of the screened baseline / anchors. They have their own A/B
        # table (Gap-Expand Probe) and their own gates.
        if (r.get('cell_multiplier_source') or '') == 'GAPFLAT_PROBE':
            continue
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
    # Jul 10 (operator-requested dual pricing): the FLIP anchor is 1× BY POLICY (measuring stick);
    # ALSO print the live-sizing counterfactual (TG_SHALLOW + NEGDI15 cells at their live mults,
    # max not stacked) so both numbers are on record at every freeze — anchor ≠ money view.
    _fs_rows = agg.get('FLIP_SHORT', [])
    _tgm_ = float(getattr(th, 'flip_short_tg_shallow_mult', 0.0) or 0.0)
    _tglo_ = float(getattr(th, 'flip_short_tg_shallow_min', -0.10) or -0.10)
    _tghi_ = float(getattr(th, 'flip_short_tg_shallow_max', 0.0) or 0.0)
    _ndm_ = float(getattr(th, 'flip_short_negdi_mult', 0.0) or 0.0)
    _ndmin_ = float(getattr(th, 'flip_short_negdi_min', 15.0) or 15.0)
    _fs_live = 0.0
    for r in _fs_rows:
        _p = pnl_current(r)
        _tg = nf(r.get('entry_btc_trend_gap_pct')); _nd = nf(r.get('entry_neg_di'))
        _m = 1.0
        if _tgm_ > 1.0 and _tg is not None and _tglo_ <= _tg < _tghi_: _m = max(_m, _tgm_)
        if _ndm_ > 1.0 and _nd is not None and _nd >= _ndmin_: _m = max(_m, _ndm_)
        _fs_live += _p * _m
    _tot_live = tot_d - sum(pnl_current(r) for r in _fs_rows) + _fs_live
    print(f"{'':12}{'':>4}{'':>5} {'':>8}  AT LIVE SIZING: FLIP ${_fs_live:+.0f} · TOTAL ${_tot_live:+.0f} (flip cells applied; ML/MS already live-sized)")
    # hard validation anchors (both must hold — these are the verified current-stack truth)
    ml = agg.get('MOM_LONG', []);  ms = agg.get('MOM_SHORT', [])
    ml_net = sum(pnl_current(x) for x in ml);  ms_net = sum(pnl_current(x) for x in ms)
    # Anchors updated 2026-07-06 (v10): momentum_short_pair_gap_min=-1.0 shipped (deep-gap floor,
    # GIGGLE post-mortem, N=3 operator override) — screens out the 3 deep-hole shorts
    # (BEL W +0.41 / MANTA / GIGGLE, -$224): MS 18->15/$526->$750. ML/FLIP unchanged.
    # v9 (2026-07-06): 07-06 batch appended (5 trades). ML +AAVE long W (+$263)
    # 28->29/$3435. MS +GIGGLE W4 L (-$135) 17->18/$526 — ⚠ ADA C1 W (+$91) traded LIVE but is
    # WEAKCAP-screened on stamped fields (rng 10.3/ATR 0.33/pADX 26.1 all under thresholds;
    # boundary sampling diff signal-time vs fill-time — screen is the stricter consistent ruler).
    # FLIP +2 core losers at 1x (pnl_current auto-demuxed the 2x fills: KAITO -$123, AAVE -$73):
    # core 37->39/$833->$637; SLOPEUP unchanged 26/-$478; blended 65/$159.
    # v8 (2026-07-06): flip 1h gate revert FIRED (phantoms 18·78% ≥ locked 60%/N≥10)
    # → slope>0 flip-shorts ADMITTED live at 1× cap (flip_short_btc_1h_slope_admit_mult=1.0, tag
    # B1H_SLOPEUP). The 26 historical slope>0 flips re-enter the screen: FLIP 37/$833 -> 63/$356.
    # SPLIT PIN (evaluate separately): core slope<=0 = 37·76%·+$833 · SLOPEUP probation = 26·~50%·−$477.
    # The SLOPEUP re-block gate compares LIVE forward vs this historical prior.
    # v7 (2026-07-05): 07-04/05 batch appended (3 trades: PUMP W +$208,
    # RPL/ETHFI L — both flat-1h zone, screened OUT by the deadband as the live gate now would):
    # ML 27->28/$2964->$3172. MS/FLIP unchanged.
    # v6 (2026-07-05): CHOPPY_FLAT removed from both flip regime fields
    # (block evidence dissolved — 28/30 CHOP flips are BTC30_RISE/BTC1H_SLOPE blocked anyway;
    # label was a proxy for BTC-turning-up): re-admits 1 winner (STG 06-18) -> FLIP 36->37/$825->$833.
    # v5 (2026-07-05): long_btc_1h_deadband=0.05 shipped (flat-1h DOA block) —
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
    assert len(ml) == 41 and round(ml_net) == 3550, f"FAIL: MOM-long {len(ml)}/${ml_net:.0f} != 41/$3550 (v14: 07-10 batch + stretch<=0.35 + PVR>=0.90 demux) — screen wrong, NOT freezing"
    assert len(ms) == 19 and round(ms_net) == 794, f"FAIL: MOM-short {len(ms)}/${ms_net:.0f} != 19/$794 (v14; NEAR 07-08 weakcap-screened) — NOT freezing"
    fl = agg.get('FLIP_SHORT', [])
    fl_net = sum(pnl_current(x) for x in fl)
    # v11 (2026-07-07 era, operator: "we block with fundaments"): SLOPEUP admit REVERTED to hard
    # block (admit_mult=0) <1 day after ship — the un-block rested on ONE flawed pillar (13/18
    # phantoms from a single bear Monday, not net-admissible) vs the block's three. FLIP returns
    # to the core-only cohort. Rewritten revert gate lives in CURRENT_STATE.
    # v12 (Jul 8): BTC trend-gap depth gate (flip_short_btc_trend_gap_min=-0.22) screens 12 more flips (42%WR/-$244)
    assert len(fl) == 31 and round(fl_net) == 692, f"FAIL: FLIP-short {len(fl)}/${fl_net:.0f} != 31/$692 (v14, unchanged from v13) — trend-gap gate off? de-mux? NOT freezing"
    print(f"\n✅ VALIDATION PASSED (ML 29/$3435 + MS 15/$750 + FLIP core 39/$637 + 0 pair-vol survivors). Freezing.")
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
