#!/usr/bin/env python3
"""
fade_exit_resim.py — path re-simulation of the SPIKE_FADE exit stack on REAL tick + 1m data (Sep-14, 2026;
operator: "make the analysis of ATR-normalised fade exits with all trades we have").

Per fade: aggTrades ticks for the first 10 min (sub-minute exits are the whole large-cap story) then 1m bars
to +4h. Replays the LIVE fade stack (fixed stop · short runner trail armed at +0.4 peak with 0.5×ATR give-back
capped at 0.35×peak and a +0.10 BE lock · HARD_TP ladder floors) and ATR-normalised variants. Bars are
resolved ADVERSE-FIRST (for a short: the high is tested before the low) — a deliberately pessimistic
convention, identical for every variant, so the comparison is fair even if absolute numbers are conservative.
Fees: 0.045% taker each side (the fade enters and exits taker). Sizing: the trade's own investment × leverage.

  ./venv/bin/python scripts/fade_exit_resim.py <paths_dir>   # dir with fades.csv + <key>.json per fade
"""
import sys, json, os, glob
import pandas as pd, numpy as np
D = sys.argv[1]
F = pd.read_csv(f'{D}/fades.csv')
FEE = 0.045  # % per side
LADDER = [(1.0, 0.25), (1.5, 0.30), (2.0, 0.40), (3.0, 0.60), (4.0, 0.80)]  # hard_tp_ladder_short


def load(key, entry):
    p = f"{D}/{key.replace('|','_').replace(':','')}.json"
    if not os.path.exists(p): return None
    j = json.load(open(p))
    ticks = j['ticks']
    # ENTRY ALIGNMENT (review of pass 1): opened_at is the signal time, the recorded entry_price is the fill;
    # on a spiking pair the first ticks after the signal can sit >0.5% from the fill and manufacture phantom
    # P&L. Start the path at the first tick within 0.15% of the fill (fill latency), else 2 s after the signal.
    if ticks:
        t0 = ticks[0][0]
        near = [i for i, (t, px) in enumerate(ticks) if abs(px / entry - 1) < 0.0015 and t - t0 <= 120000]
        start = near[0] if near else next((i for i, (t, px) in enumerate(ticks) if t - t0 >= 2000), 0)
        ticks = ticks[start:]
    ev = [(t, px, px) for t, px in ticks]                      # ticks: hi == lo == px
    t_last = ev[-1][0] if ev else 0
    for t, o, h, l, c in j['bars']:
        if t <= t_last: continue                                # bars after the tick window only
        ev.append((t, h, l))                                    # adverse-first: (hi, lo)
    return ev


def sim(entry, atr, ev, stop_pct, arm, gb_mult, gb_frac, lock, ladder, max_min=240):
    """SHORT. Returns (pnl_pct_net, reason, minutes). stop_pct negative."""
    peak = 0.0; t0 = ev[0][0]
    for t, hi, lo in ev:
        if (t - t0) / 60000 > max_min: break
        # adverse first
        adv = (entry - hi) / entry * 100 - 2 * FEE   # NET pnl at the bar high (engine stops/trails on fee-inclusive pnl)
        if adv <= stop_pct + 0.02:   # engine fires at ≈ the level (龙虾 filled at net −1.49 on a −1.5 stop)
            return stop_pct, 'STOP', (t - t0) / 60000
        # floor from CURRENT peak (before this bar's favourable move), then test adverse side against floor
        floor = None
        if peak >= arm - 0.005 and atr and atr > 0:
            gb = gb_mult * atr
            if gb_frac > 0 and gb_frac * peak < gb: gb = gb_frac * peak
            floor = max(peak - gb, lock)
        if ladder:
            lf = max([f for p_, f in ladder if peak >= p_], default=None)
            if lf is not None: floor = max(floor if floor is not None else -99, lf)
        if floor is not None and floor >= 0 and adv <= floor:
            return floor, 'TRAIL', (t - t0) / 60000
        fav = (entry - lo) / entry * 100 - 2 * FEE   # NET peak
        if fav > peak: peak = fav
    # timed out: mark at last price (use last lo/hi midpoint)
    t, hi, lo = ev[-1] if ev else (t0, entry, entry)
    return (entry - (hi + lo) / 2) / entry * 100 - 2 * FEE, 'TIMEOUT', (t - t0) / 60000


VARIANTS = {
    'LIVE  stop −1.5 · arm 0.4 · gb 0.5ATR': dict(stop=lambda a: -1.5, arm=lambda a: 0.4, gb=0.5, frac=0.35, lock=0.10, ladder=LADDER),
    'A  stop −max(1.5,1.5ATR)≤3 · rest live': dict(stop=lambda a: -min(3.0, max(1.5, 1.5 * a)), arm=lambda a: 0.4, gb=0.5, frac=0.35, lock=0.10, ladder=LADDER),
    'B  stop −max(1.5,2ATR)≤3 · rest live':   dict(stop=lambda a: -min(3.0, max(1.5, 2.0 * a)), arm=lambda a: 0.4, gb=0.5, frac=0.35, lock=0.10, ladder=LADDER),
    'C  stop A · arm max(0.4,0.5ATR)':        dict(stop=lambda a: -min(3.0, max(1.5, 1.5 * a)), arm=lambda a: max(0.4, 0.5 * a), gb=0.5, frac=0.35, lock=0.10, ladder=LADDER),
    'D  stop A · arm max(0.4,1ATR) · gb 1ATR': dict(stop=lambda a: -min(3.0, max(1.5, 1.5 * a)), arm=lambda a: max(0.4, 1.0 * a), gb=1.0, frac=0.0, lock=0.10, ladder=LADDER),
    'E  stop B · arm max(0.4,1ATR) · gb 1ATR': dict(stop=lambda a: -min(3.0, max(1.5, 2.0 * a)), arm=lambda a: max(0.4, 1.0 * a), gb=1.0, frac=0.0, lock=0.10, ladder=LADDER),
    'F  stop −2.5 flat · rest live':          dict(stop=lambda a: -2.5, arm=lambda a: 0.4, gb=0.5, frac=0.35, lock=0.10, ladder=LADDER),
    'G  live stop · gb 1ATR uncapped only':    dict(stop=lambda a: -1.5, arm=lambda a: 0.4, gb=1.0, frac=0.0, lock=0.10, ladder=LADDER),
    'H  live stop · arm max(0.4,1ATR) · gb 1ATR': dict(stop=lambda a: -1.5, arm=lambda a: max(0.4, 1.0 * a), gb=1.0, frac=0.0, lock=0.10, ladder=LADDER),
    'I  live stop · gb 0.5ATR uncapped (no 0.35 cap)': dict(stop=lambda a: -1.5, arm=lambda a: 0.4, gb=0.5, frac=0.0, lock=0.10, ladder=LADDER),
}

rows = []
for _, r in F.iterrows():
    ev = load(r.key, float(r.entry_price))
    if ev is None or len(ev) < 5: continue
    atr = float(r.entry_atr_pct) if pd.notna(r.entry_atr_pct) else 0.5
    dpp = float(r.investment) * float(r.leverage) / 100.0   # $ per 1% move
    rec = dict(key=r.key, era=r.era, pair=r.pair, atr=atr, vol_M=float(r.entry_pair_volume_24h_usd) / 1e6 if pd.notna(r.entry_pair_volume_24h_usd) else np.nan,
               screened=str(r.stack_keep).lower() != 'false', actual_pct=float(r.pnl_percentage), actual_usd=float(r.pnl), actual_reason=str(r.close_reason))
    for name, v in VARIANTS.items():
        p, why, mins = sim(float(r.entry_price), atr, ev, v['stop'](atr), v['arm'](atr), v['gb'], v['frac'], v['lock'], v['ladder'])
        rec[f'{name}|pct'] = p; rec[f'{name}|usd'] = p * dpp; rec[f'{name}|why'] = why; rec[f'{name}|min'] = mins
    rows.append(rec)
R = pd.DataFrame(rows)
R.to_csv(f'{D}/resim_results.csv', index=False)
R['large'] = R.vol_M >= 20
live = 'LIVE  stop −1.5 · arm 0.4 · gb 0.5ATR'
print(f"re-simulated {len(R)} fades ({R.screened.sum()} screened, {R.large.sum()} large-cap ≥$20M) on tick+1m paths, adverse-first")
print(f"\nVALIDATION — LIVE replica vs what actually happened (screened): sign agreement "
      f"{((R[R.screened][f'{live}|pct']>0)==(R[R.screened].actual_pct>0)).mean()*100:.0f}% · "
      f"Σ replica ${R[R.screened][f'{live}|usd'].sum():+.0f} vs actual ${R[R.screened].actual_usd.sum():+.0f}")
for label, mask in [('SCREENED all', R.screened), ('SCREENED micro-cap <$20M', R.screened & ~R.large), ('SCREENED large-cap ≥$20M', R.screened & R.large), ('RAW all 88', R.screened | ~R.screened)]:
    d = R[mask]
    if not len(d): continue
    print(f"\n== {label}: N={len(d)} ==")
    print(f"{'variant':44} {'WR':>4} {'avg%':>7} {'Σ$':>8} {'Δ$ vs LIVE':>11} {'stops':>5} {'worst%':>7}")
    base = d[f'{live}|usd'].sum()
    for name in VARIANTS:
        p = d[f'{name}|pct']; u = d[f'{name}|usd']
        print(f"{name:44} {100*(p>0).mean():3.0f}% {p.mean():+7.3f} {u.sum():+8.0f} {u.sum()-base:+11.0f} {(d[f'{name}|why']=='STOP').sum():5d} {p.min():+7.2f}")
print("\n== large-cap fills, per variant (pct) ==")
cols = ['pair', 'atr', 'vol_M', 'actual_pct'] + [f'{n}|pct' for n in VARIANTS]
print(R[R.large][cols].rename(columns=lambda c: c.split('|')[0][:6] if '|' in c else c).round(2).to_string(index=False))
