#!/usr/bin/env python3
"""
Cross-batch winner signature analysis (May 20 latest+2).

Identifies the 4-6 most common entry signatures shared by WINNERS in the
archived trade pool. Goal: validate whether Pattern W (winner tracker)
has structural signal — i.e., whether winners cluster into recognizable
multi-dimensional combinations, or are uniformly distributed (in which
case Pattern W would just track noise).

Methodology:
  1. Load deduped pool (May 4 → today, CLOSED only).
  2. Split into LONG winners and SHORT winners.
  3. For each direction's winners, find clusters by combining entry features.
  4. Score each candidate signature by: how many winners share it (recall)
     AND what fraction of matches are winners (precision).
  5. Report top signatures meeting: N ≥ 15 winners + ≥70% precision.

The output drives Pattern W signature definitions.
"""
import csv, glob, os
from collections import defaultdict

def f(s):
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None

def collect():
    files = sorted(glob.glob("reports/orders_*.csv"))
    live = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")))
    if live:
        files.append(live[-1])
    seen = set(); out = []
    for p in files:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") != "CLOSED":
                    continue
                op = r.get("opened_at", "")
                if not op or op < "2026-05-04":
                    continue
                k = (op, r.get("pair"), r.get("direction"))
                if k in seen:
                    continue
                seen.add(k); out.append(r)
    return out

rows = collect()
print(f"Pool: {len(rows)} trades\n")

# ============================================================
# Split winners / losers per direction
# ============================================================
longs = [r for r in rows if r.get("direction") == "LONG"]
shorts = [r for r in rows if r.get("direction") == "SHORT"]

long_W = [r for r in longs if (f(r.get("pnl")) or 0) > 0]
long_L = [r for r in longs if (f(r.get("pnl")) or 0) <= 0]
short_W = [r for r in shorts if (f(r.get("pnl")) or 0) > 0]
short_L = [r for r in shorts if (f(r.get("pnl")) or 0) <= 0]

print(f"LONG: {len(longs)} total ({len(long_W)} winners, {len(long_L)} losers)")
print(f"SHORT: {len(shorts)} total ({len(short_W)} winners, {len(short_L)} losers)")
print()

# ============================================================
# Each candidate signature is a function (r) -> bool.
# We score on:
#   - recall: matched_winners / total_winners
#   - precision: matched_winners / (matched_winners + matched_losers)
#   - lift: precision / baseline_winrate
# A good Pattern W has precision ≥ 70% (consistent winners)
# AND recall meaningful (catches a useful fraction of winners).
# ============================================================

def score_signature(label, predicate, W, L):
    mW = sum(1 for r in W if predicate(r))
    mL = sum(1 for r in L if predicate(r))
    total = mW + mL
    if total == 0:
        return None
    precision = 100 * mW / total
    recall = 100 * mW / max(len(W), 1)
    baseline = 100 * len(W) / max(len(W) + len(L), 1)
    lift = precision / baseline if baseline > 0 else 0
    return {
        "label": label, "matched_winners": mW, "matched_losers": mL,
        "total": total, "precision": precision, "recall": recall,
        "baseline_wr": baseline, "lift": lift,
    }

def report(direction, W, L, sigs):
    print("=" * 90)
    print(f"{direction} WINNER SIGNATURES — baseline WR = {100*len(W)/(len(W)+len(L)):.1f}%")
    print("=" * 90)
    print(f"  {'Signature':50} N    W    L    Prec%  Rec%   Lift  Verdict")
    results = []
    for label, pred in sigs:
        s = score_signature(label, pred, W, L)
        if s is None:
            continue
        results.append(s)
        verdict = ""
        if s["matched_winners"] >= 15 and s["precision"] >= 70:
            verdict = "★ PATTERN W candidate"
        elif s["precision"] >= 65:
            verdict = "✓ promising"
        elif s["precision"] < 50:
            verdict = "✗ no signal"
        print(f"  {label:50} {s['total']:>3}  {s['matched_winners']:>3}  {s['matched_losers']:>3}  "
              f"{s['precision']:>5.1f}  {s['recall']:>5.1f}  {s['lift']:>5.2f}  {verdict}")
    return results

# ============================================================
# Candidate signature library — multi-dimensional combinations
# Based on prior analysis observations across the pool:
#  - Strong trend continuation (high ADX + ADX still rising)
#  - Macro tailwind (BTC trend aligned, BTC ADX committed)
#  - Volatility breakout (energetic BTC + above-avg volume + stretch)
#  - Pullback entry (mid-range + pair gap aligned)
#  - Confluence (multiple "sweet spot" cells all true)
#
# Some are designed for LONG, some for SHORT, some symmetric.
# ============================================================

def has(r, k):
    return f(r.get(k))

# ============== LONG signatures ==============
long_sigs = [
    # Core mechanism candidates
    ("W1: HighConv trend (ADX≥22 + ADXΔ≥0.5 + stretch ≥0.16)",
        lambda r: (has(r, "entry_adx") or 0) >= 22
                  and (has(r, "entry_adx_delta") or 0) >= 0.5
                  and (has(r, "entry_ema5_stretch") or 0) >= 0.16),

    ("W2: Macro tailwind (BTC RSI 50-65 + BTC ADX ≥22 + BTC gap ≥+0.10)",
        lambda r: 50 <= (has(r, "entry_btc_rsi") or -1) <= 65
                  and (has(r, "entry_btc_adx") or 0) >= 22
                  and (has(r, "entry_btc_trend_gap_pct") or -99) >= 0.10),

    ("W3: Energetic vol (BTC ATR≥0.20 + pair vol ratio≥1.20 + stretch≥0.20)",
        lambda r: (has(r, "entry_btc_atr_pct") or 0) >= 0.20
                  and (has(r, "entry_pair_volume_ratio") or 0) >= 1.20
                  and (has(r, "entry_ema5_stretch") or 0) >= 0.20),

    ("W4: Pullback aligned (RngPos 40-75 + pair gap ≥+0.10 + ADXΔ≥0)",
        lambda r: 40 <= (has(r, "entry_range_position") or -1) <= 75
                  and (has(r, "entry_pair_ema20_ema50_gap_pct") or -99) >= 0.10
                  and (has(r, "entry_adx_delta") or -99) >= 0),

    ("W5: Confluence (BTC ADX 22-30 + BTC RSI 55-65 + Pair ADX 22-30 + stretch 0.16-0.25)",
        lambda r: 22 <= (has(r, "entry_btc_adx") or -1) <= 30
                  and 55 <= (has(r, "entry_btc_rsi") or -1) <= 65
                  and 22 <= (has(r, "entry_adx") or -1) <= 30
                  and 0.16 <= (has(r, "entry_ema5_stretch") or -1) <= 0.25),

    # Simpler 2D winners
    ("Sweet spot: BTC RSI 55-65 + BTC ADX 22-30 alone",
        lambda r: 55 <= (has(r, "entry_btc_rsi") or -1) <= 65
                  and 22 <= (has(r, "entry_btc_adx") or -1) <= 30),

    ("Sweet spot: stretch 0.16-0.25 alone",
        lambda r: 0.16 <= (has(r, "entry_ema5_stretch") or -1) <= 0.25),

    ("Sweet spot: pair gap positive AND BTC gap positive (both trends aligned)",
        lambda r: (has(r, "entry_pair_ema20_ema50_gap_pct") or -99) > 0
                  and (has(r, "entry_btc_trend_gap_pct") or -99) > 0),

    # Test C-pattern inverses (high WR among C-flagged might mean ANTI-pattern is winner)
    ("Non-C cohort (matches NO Pattern C signature)",
        lambda r: not any(r.get(f"entry_pattern_c{i}_match", "").lower() == "true" for i in range(1, 10))),
]

# ============== SHORT signatures (mirrored where appropriate) ==============
short_sigs = [
    ("W1: HighConv trend (ADX≥22 + ADXΔ≥0.5 + stretch ≥0.20)",
        lambda r: (has(r, "entry_adx") or 0) >= 22
                  and (has(r, "entry_adx_delta") or 0) >= 0.5
                  and (has(r, "entry_ema5_stretch") or 0) >= 0.20),

    ("W2: Macro tailwind (BTC RSI 30-45 + BTC ADX ≥22 + BTC gap ≤-0.10)",
        lambda r: 30 <= (has(r, "entry_btc_rsi") or -1) <= 45
                  and (has(r, "entry_btc_adx") or 0) >= 22
                  and (has(r, "entry_btc_trend_gap_pct") or 99) <= -0.10),

    ("W3: Energetic vol (BTC ATR≥0.20 + pair vol ratio≥1.20 + stretch≥0.25)",
        lambda r: (has(r, "entry_btc_atr_pct") or 0) >= 0.20
                  and (has(r, "entry_pair_volume_ratio") or 0) >= 1.20
                  and (has(r, "entry_ema5_stretch") or 0) >= 0.25),

    ("W4: Pullback aligned (RngPos 25-60 + pair gap ≤-0.10 + ADXΔ≥0)",
        lambda r: 25 <= (has(r, "entry_range_position") or -1) <= 60
                  and (has(r, "entry_pair_ema20_ema50_gap_pct") or 99) <= -0.10
                  and (has(r, "entry_adx_delta") or -99) >= 0),

    ("W5: Confluence (BTC ADX 22-30 + BTC RSI 30-40 + Pair ADX 22-30 + stretch 0.20-0.30)",
        lambda r: 22 <= (has(r, "entry_btc_adx") or -1) <= 30
                  and 30 <= (has(r, "entry_btc_rsi") or -1) <= 40
                  and 22 <= (has(r, "entry_adx") or -1) <= 30
                  and 0.20 <= (has(r, "entry_ema5_stretch") or -1) <= 0.30),

    ("Sweet spot: BTC RSI 30-40 + BTC ADX 22-30 alone",
        lambda r: 30 <= (has(r, "entry_btc_rsi") or -1) <= 40
                  and 22 <= (has(r, "entry_btc_adx") or -1) <= 30),

    ("Sweet spot: stretch 0.20-0.30 alone",
        lambda r: 0.20 <= (has(r, "entry_ema5_stretch") or -1) <= 0.30),

    ("Sweet spot: pair gap negative AND BTC gap negative (both trends aligned)",
        lambda r: (has(r, "entry_pair_ema20_ema50_gap_pct") or 99) < 0
                  and (has(r, "entry_btc_trend_gap_pct") or 99) < 0),

    ("Non-C cohort (matches NO Pattern C signature)",
        lambda r: not any(r.get(f"entry_pattern_c{i}_match", "").lower() == "true" for i in range(1, 10))),

    # Today's strong SHORT cohort
    ("Today's pattern: BTC ATR≤0.15 + Pair ADX 22-25 + RngPos 15-30",
        lambda r: (has(r, "entry_btc_atr_pct") or 99) <= 0.15
                  and 22 <= (has(r, "entry_adx") or -1) <= 25
                  and 15 <= (has(r, "entry_range_position") or -1) <= 30),
]

print()
report("LONG", long_W, long_L, long_sigs)
print()
report("SHORT", short_W, short_L, short_sigs)
