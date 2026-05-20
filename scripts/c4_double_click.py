#!/usr/bin/env python3
"""
C4 deep-dive: within C4-matched trades, find what separates winners from losers.

C4 signature (same for both directions):
    BTC ATR <= 0.15% AND BTC ADX <= 22 AND Pair ADX <= 25

Since entry_pattern_c4_match only populated post-May-19 deploy, we RE-COMPUTE
C4 match from raw entry_btc_atr_pct + entry_btc_adx + entry_adx so the full
cross-batch pool participates.

Then within the C4 cohort, split winners (pnl > 0) vs losers (pnl <= 0) and
compare distributions on every captured entry dimension.
"""
import csv, glob, os
from datetime import datetime
from collections import defaultdict
from statistics import mean, median, stdev

def parse_dt(s):
    if not s: return None
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f","%Y-%m-%dT%H:%M:%S","%Y-%m-%d %H:%M:%S.%f","%Y-%m-%d %H:%M:%S"):
        try: return datetime.strptime(s.strip(),fmt)
        except: pass
    return None

def f(s):
    if s is None or s=="": return None
    try: return float(s)
    except: return None

def collect():
    files = sorted(glob.glob("reports/orders_*.csv"))
    live = sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")))
    if live: files.append(live[-1])
    seen=set(); out=[]
    for p in files:
        with open(p) as fh:
            for r in csv.DictReader(fh):
                if r.get("status")!="CLOSED": continue
                op=r.get("opened_at","")
                if not op or op<"2026-05-04": continue
                k=(op,r.get("pair"),r.get("direction"))
                if k in seen: continue
                seen.add(k); out.append(r)
    return out

rows = collect()
print(f"Total deduped CLOSED trades (May 4+): {len(rows)}\n")

# Re-compute C4 match from raw fields (since entry_pattern_c4_match NULL pre-deploy)
C4_BTC_ATR_MAX = 0.15
C4_BTC_ADX_MAX = 22.0
C4_PAIR_ADX_MAX = 25.0

c4_trades = []
for r in rows:
    btc_atr = f(r.get("entry_btc_atr_pct"))
    btc_adx = f(r.get("entry_btc_adx"))
    pair_adx = f(r.get("entry_adx"))
    if btc_atr is None or btc_adx is None or pair_adx is None:
        continue  # can't determine match
    if btc_atr <= C4_BTC_ATR_MAX and btc_adx <= C4_BTC_ADX_MAX and pair_adx <= C4_PAIR_ADX_MAX:
        c4_trades.append(r)

print(f"C4-matched trades (recomputed from raw fields): {len(c4_trades)}")
print(f"  (entry_pattern_c4_match populated only on post-deploy trades — recompute covers full pool)\n")

if not c4_trades:
    print("No C4 trades in pool. Cannot proceed.")
    exit()

# Split winners / losers
winners = [r for r in c4_trades if (f(r.get("pnl")) or 0) > 0]
losers = [r for r in c4_trades if (f(r.get("pnl")) or 0) <= 0]

print(f"C4 winners: N={len(winners)}")
print(f"C4 losers:  N={len(losers)}")
print(f"C4 WR:      {100*len(winners)/len(c4_trades):.1f}%")
print()

# By direction
for dir_ in ["LONG","SHORT"]:
    sub = [r for r in c4_trades if r.get("direction")==dir_]
    w = [r for r in sub if (f(r.get("pnl")) or 0)>0]
    print(f"  {dir_}: N={len(sub)}  WR={100*len(w)/max(len(sub),1):.1f}%")
print()

# P&L stats
def stats(trades, key):
    vals = [f(r.get(key)) for r in trades]
    vals = [v for v in vals if v is not None]
    if not vals: return None
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "min": min(vals),
        "max": max(vals),
    }

# Compare dimensions winners vs losers
DIMS = [
    "entry_rsi",
    "entry_adx",
    "entry_adx_delta",
    "entry_adx_prev",
    "entry_ema_gap_5_8",
    "entry_ema5_stretch",
    "entry_ema20_slope",
    "entry_btc_rsi",
    "entry_btc_adx",
    "entry_btc_ema20_slope",
    "entry_btc_atr_pct",
    "entry_btc_rsi_1h",
    "entry_btc_trend_gap_pct",
    "entry_range_position",
    "entry_global_volume_ratio",
    "entry_pair_volume_ratio",
    "entry_pos_di",
    "entry_neg_di",
    "entry_funding_rate",
    "entry_pair_ema20_ema50_gap_pct",
    "entry_dist_from_ema13_pct",
    "entry_btc_1h_slope",
    "entry_pair_volume_24h_usd",
    "entry_quality_score",
    "peak_pnl",
    "trough_pnl",
    "pnl_percentage",
]

print("="*100)
print(f"{'Dimension':40} {'W_mean':>10} {'L_mean':>10} {'Δ(W-L)':>10} {'W_med':>10} {'L_med':>10}  Note")
print("="*100)

for d in DIMS:
    ws = stats(winners, d)
    ls = stats(losers, d)
    if not ws or not ls:
        continue
    delta = ws["mean"] - ls["mean"]
    # Flag big gaps
    note = ""
    if ws["mean"] != 0 and abs(delta) / max(abs(ws["mean"]), abs(ls["mean"]), 0.001) > 0.20:
        note = " ★ >20% gap"
    print(f"{d:40} {ws['mean']:>10.4f} {ls['mean']:>10.4f} {delta:>+10.4f} {ws['median']:>10.4f} {ls['median']:>10.4f}{note}")

print()
print("="*100)
print("Outcome distribution (peak/trough/close)")
print("="*100)

# Peak buckets
def bucket(v):
    if v is None: return "NULL"
    if v < 0: return "neg"
    if v < 0.05: return "[0, 0.05)"
    if v < 0.10: return "[0.05, 0.10)"
    if v < 0.20: return "[0.10, 0.20)"
    if v < 0.30: return "[0.20, 0.30)"
    if v < 0.50: return "[0.30, 0.50)"
    return ">= 0.50"

for label, subset in [("WINNERS", winners), ("LOSERS", losers)]:
    print(f"\n{label} (N={len(subset)})  Peak distribution:")
    cnt = defaultdict(int)
    for r in subset:
        cnt[bucket(f(r.get("peak_pnl")))] += 1
    for k in ["neg","[0, 0.05)","[0.05, 0.10)","[0.10, 0.20)","[0.20, 0.30)","[0.30, 0.50)",">= 0.50","NULL"]:
        if cnt[k]:
            print(f"  peak {k:18} N={cnt[k]:3} ({100*cnt[k]/len(subset):.0f}%)")

# Close reasons
print("\n\nClose reason distribution within C4:")
for label, subset in [("WINNERS", winners), ("LOSERS", losers)]:
    cnt = defaultdict(int)
    for r in subset:
        cnt[r.get("close_reason","UNKNOWN")] += 1
    print(f"\n{label}:")
    for k, v in sorted(cnt.items(), key=lambda x: -x[1]):
        print(f"  {k:30} N={v}")

# Pair-level analysis — which pairs dominate C4?
print("\n\n" + "="*100)
print("PER-PAIR DISTRIBUTION (C4 only)")
print("="*100)
pair_cnt = defaultdict(lambda: {"w": 0, "l": 0, "$": 0.0})
for r in c4_trades:
    p = r.get("pair","?")
    pnl = f(r.get("pnl")) or 0
    if pnl > 0:
        pair_cnt[p]["w"] += 1
    else:
        pair_cnt[p]["l"] += 1
    pair_cnt[p]["$"] += pnl

print(f"{'Pair':20} {'W':>4} {'L':>4} {'WR%':>6} {'$':>10}")
sorted_pairs = sorted(pair_cnt.items(), key=lambda x: -(x[1]["w"]+x[1]["l"]))
for pair, v in sorted_pairs[:30]:
    n = v["w"] + v["l"]
    wr = 100*v["w"]/n if n else 0
    print(f"{pair:20} {v['w']:>4} {v['l']:>4} {wr:>6.1f} {v['$']:>+10.2f}")

# Hour-of-day distribution
print("\n\n" + "="*100)
print("HOUR-OF-DAY (UTC) — C4 winners vs losers")
print("="*100)
hour_cnt = defaultdict(lambda: {"w": 0, "l": 0})
for r in c4_trades:
    ot = parse_dt(r.get("opened_at"))
    if ot is None: continue
    h = ot.hour
    pnl = f(r.get("pnl")) or 0
    if pnl > 0: hour_cnt[h]["w"] += 1
    else: hour_cnt[h]["l"] += 1
print(f"{'Hour':>5} {'W':>4} {'L':>4} {'WR%':>6}")
for h in sorted(hour_cnt.keys()):
    v = hour_cnt[h]
    n = v["w"] + v["l"]
    wr = 100*v["w"]/n
    print(f"{h:>5} {v['w']:>4} {v['l']:>4} {wr:>6.1f}")
