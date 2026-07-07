#!/usr/bin/env python3
"""
screen_phantoms.py — NET-ADMISSIBILITY join for PASS:* flip-short phantoms.

WHY: PASS phantoms are seeded at FIRST BLOCK inside `_flip_filters`. The 1h-slope
gate fires with six gates still downstream (HIATR, RSI_MIN, QUALITY, BEAR_MIN, ADXD,
PAIR_GAP, FAN sub-gates), so the phantom tracker's N/WR OVER-COUNT what a re-enable
would actually admit. The rewritten SLOPEUP revert gate (2026-07-06) and the per-regime
carve-out gates are defined on NET-ADMISSIBLE phantoms only — this script computes them
by replaying the REAL `_flip_filters` with ONLY the phantom's own blocking gate disabled.

USAGE:
  python3 scripts/screen_phantoms.py <phantoms.csv> [<phantoms2.csv> ...]
      [--source PASS:FLIP_SHORT_BTC1H_SLOPE]

  CSVs come from /api/phantom-flips/export.csv (download BEFORE every paper reset —
  phantoms are deleted on reset; stacking multiple exports here = stacking priors).
  Rows are deduped on (entry_at, pair, source_filter) so overlapping exports are safe.

SOURCES handled:
  PASS:FLIP_SHORT_BTC1H_SLOPE  -> re-run with flip_short_btc_1h_slope_max=99 (gate off)
  PASS:FLIP_SHORT_REGIME       -> re-run with both regime-block fields cleared

Output: per-BTC-regime net-admissible table (N / WR / avg% / Σ% / dates) + the
downstream-blocker census for the rows that would NOT actually trade.
"""
import csv, sys, os
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from services.trading_engine import _flip_filters

th = config.trading_config.thresholds

DEFAULT_SOURCE = "PASS:FLIP_SHORT_BTC1H_SLOPE"


def nf(x):
    try:
        return float(x)
    except Exception:
        return None


def phantom_ind(r):
    """PhantomFlip CSV row -> the engine `ind` dict (field-audited against
    screen_pool.flip_ind / trading_engine _ff_in). Phantom columns already use
    engine-native names except pair_rsi (entry_rsi)."""
    a, b = nf(r.get('entry_ema_gap_5_8')), nf(r.get('entry_ema_gap_8_13'))
    return {
        'flip_dir': (r.get('flip_direction') or 'SHORT').upper(),
        'btc_regime': r.get('entry_btc_regime') or None,
        'adx': nf(r.get('entry_adx')),
        'pair_gap': nf(r.get('entry_pair_ema20_ema50_gap_pct')),
        'fan_ratio': (abs(a / b) if (a is not None and b) else None),
        'ema5_stretch': nf(r.get('entry_ema5_stretch')),
        'adx_delta': nf(r.get('entry_adx_delta')),
        'btc_rsi': nf(r.get('entry_btc_rsi')),
        'btc_rsi_prev6': nf(r.get('entry_btc_rsi_prev6')),
        'btc_adx': nf(r.get('entry_btc_adx')),
        'btc_atr_pct': nf(r.get('entry_btc_atr_pct')),
        'atr_pct': nf(r.get('entry_atr_pct')),
        'pair_rsi': nf(r.get('entry_rsi')),
        'quality_score': nf(r.get('entry_quality_score')),
        'bear_pct': nf(r.get('entry_bear_pct')),
        'range_position': nf(r.get('entry_range_position')),
        'btc_1h_slope': nf(r.get('entry_btc_1h_slope')),
    }


def disable_own_gate(source):
    """Turn OFF exactly the gate that seeded this PASS source, so the replay
    evaluates only the DOWNSTREAM stack. Returns a restore function."""
    if source == 'PASS:FLIP_SHORT_BTC1H_SLOPE':
        old = getattr(th, 'flip_short_btc_1h_slope_max', 99.0)
        th.flip_short_btc_1h_slope_max = 99.0
        def restore():
            th.flip_short_btc_1h_slope_max = old
        return restore
    if source == 'PASS:FLIP_SHORT_REGIME':
        old1 = getattr(th, 'flip_short_regime_block_regimes', '')
        old2 = getattr(th, 'flip_short_regime_block_any_adxd_regimes', '')
        th.flip_short_regime_block_regimes = ''
        th.flip_short_regime_block_any_adxd_regimes = ''
        def restore():
            th.flip_short_regime_block_regimes = old1
            th.flip_short_regime_block_any_adxd_regimes = old2
        return restore
    raise SystemExit(f"Unsupported source {source!r} — add its gate-off mapping first.")


def main():
    args = [a for a in sys.argv[1:]]
    source = DEFAULT_SOURCE
    if '--source' in args:
        i = args.index('--source')
        source = args[i + 1]
        del args[i:i + 2]
    files = args
    if not files:
        raise SystemExit(__doc__)

    rows, seen = [], set()
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r.get('source_filter') != source:
                    continue
                if not r.get('pnl_pct'):        # still-open phantom — no outcome yet
                    continue
                key = (r.get('entry_at'), r.get('pair'), r.get('source_filter'))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(r)

    print(f"source={source}  closed phantoms loaded: {len(rows)} (deduped, {len(files)} file(s))")
    if not rows:
        return

    restore = disable_own_gate(source)
    try:
        admissible, blocked = [], []
        for r in rows:
            b, reason, *_ = _flip_filters('FAN_RATIO_GATE', phantom_ind(r))
            (blocked if b else admissible).append((r, reason))
    finally:
        restore()

    def stats(rs):
        n = len(rs)
        if not n:
            return "0"
        pnls = [nf(r.get('pnl_pct')) or 0.0 for r in rs]
        w = sum(1 for p in pnls if p > 0)
        dates = {(r.get('entry_at') or '')[:10] for r in rs}
        return (f"{n:>3} · {100*w/n:3.0f}% · avg {sum(pnls)/n:+.3f}% · "
                f"Σ {sum(pnls):+.2f}% · {len(dates)} date(s)")

    adm_rows = [r for r, _ in admissible]
    print(f"\nNET-ADMISSIBLE (pass every downstream gate): {stats(adm_rows)}")
    by_reg = defaultdict(list)
    for r, _ in admissible:
        by_reg[r.get('entry_btc_regime') or 'UNKNOWN'].append(r)
    for reg in sorted(by_reg, key=lambda k: -len(by_reg[k])):
        print(f"  {reg:<16} {stats(by_reg[reg])}")

    print(f"\nBLOCKED DOWNSTREAM (would NOT trade even with the gate open): {len(blocked)}")
    by_blk = defaultdict(list)
    for r, reason in blocked:
        by_blk[reason or '?'].append(r)
    for reason in sorted(by_blk, key=lambda k: -len(by_blk[k])):
        rs = by_blk[reason]
        print(f"  {reason:<26} {stats(rs)}")

    # Per-regime carve-out gate check (locked 2026-07-06): N>=10 · WR>=60% ·
    # avg>=+0.30 · net positive · >=3 distinct dates — on NET-ADMISSIBLE rows.
    print("\nCARVE-OUT GATE CHECK (net-admissible; N≥10 · WR≥60% · avg≥+0.30 · Σ>0 · ≥3 dates):")
    for reg in sorted(by_reg, key=lambda k: -len(by_reg[k])):
        rs = by_reg[reg]
        pnls = [nf(r.get('pnl_pct')) or 0.0 for r in rs]
        n, w = len(rs), sum(1 for p in pnls if p > 0)
        avg, tot = sum(pnls) / len(pnls), sum(pnls)
        dates = len({(r.get('entry_at') or '')[:10] for r in rs})
        ok = (n >= 10 and 100*w/n >= 60 and avg >= 0.30 and tot > 0 and dates >= 3)
        fails = []
        if n < 10: fails.append(f"N {n}/10")
        if n and 100*w/n < 60: fails.append(f"WR {100*w/n:.0f}%")
        if avg < 0.30: fails.append(f"avg {avg:+.2f}")
        if tot <= 0: fails.append("Σ≤0")
        if dates < 3: fails.append(f"dates {dates}/3")
        print(f"  {reg:<16} {'✅ PASSES' if ok else '❌ ' + ', '.join(fails)}")


if __name__ == '__main__':
    main()
