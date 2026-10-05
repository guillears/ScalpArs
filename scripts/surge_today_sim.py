#!/usr/bin/env python3
"""⚡ Oct-4 (operator: "show me what today's results would have looked like") — every SURGE_LONG trigger candidate on the last N hours of
LIVE public data, through the SAME code paths as the year grid: services.surge.surge_pair_pick for the picks, the market-volume reading of
the live gate's universe (scout_surge_obs._market over the scout's top-volume frames), entry 60 s after the bar closes (1m open), today's
SURGE_LONG exit replica (surge_bearrun_review.variants LIVE) on 1m bars, low before high, net of fees, 4 h hold. Read-only.
  venv/bin/python scripts/surge_today_sim.py [hours=6]
"""
import os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import opportunity_scout as OS
import scout_surge_obs as SO
from services.surge import surge_pair_pick
from types import SimpleNamespace

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 6
BAR, MIN = 300_000, 60_000
now_ms = int(time.time() * 1000); last_closed = (now_ms // BAR - 1) * BAR
cfg = OS.load_cfg(); c2 = {**cfg, **(cfg.get("thresholds") or {})}; th = SimpleNamespace(**c2)
btc = OS.k5("BTC/USDT:USDT", last_closed, 1500)
scan, rank, cutoff, limit = OS.bot_universe(cfg)
alts = {}
for p in scan:
    alts[p] = OS.k5(p[:-4] + "/USDT:USDT", last_closed); time.sleep(0.03)
in_now = [p for p in scan if rank.get(p, 999) <= limit]
b = btc.copy(); b["qv"] = b.c * b.v
b["r30"] = (b.c / b.c.shift(6) - 1) * 100
b["volx"] = b.qv / b.qv.shift(1).rolling(288).median()
b["hi"] = b.h.shift(1).rolling(288).max()
b["hi_x60"] = b.h.shift(13).rolling(276).max()
b["adx"] = SO._adx(b); b["d_adx"] = b.adx - b.adx.shift(6)
frames = {}
for p, d in list(alts.items()) + [("BTCUSDT", btc)]:
    if d is None or len(d) < 300:
        continue
    x = d.copy(); x["qv24"] = (x.c * x.v).rolling(288).sum(); x["e20"] = x.c.ewm(span=20, adjust=False).mean(); x["m48"] = x.v.rolling(48).mean()
    frames[p] = x
bl = {x.strip().upper() for x in str(c2.get("surge_long_pair_blacklist") or "").split(",") if x.strip()} | {"BTCUSDT", "ETHUSDT"}
loc = lambda t: (pd.to_datetime(int(t), unit="ms") - pd.Timedelta(hours=3)).strftime("%H:%M")   # operator's local time (UTC−3), bar OPEN


def picks_at(t, btc_move, taken):
    cand = sorted((p for p in in_now if p not in bl and p in frames and t in frames[p].index and np.isfinite(frames[p].at[t, "qv24"])),
                  key=lambda p: -frames[p].at[t, "qv24"])[:int(c2.get("surge_universe_size", 20) or 20)]
    out = []
    for p in cand:
        if p in taken:
            continue
        d = alts[p]
        rows = [[int(k), float(x.o), float(x.h), float(x.l), float(x.c), float(x.v)] for k, x in d[d.index <= t].tail(60).iterrows()]
        ok, why, atr, pm = surge_pair_pick(rows, int(t), float(btc_move), th, "LONG")
        if ok:
            out.append((p, atr, pm))
    return out


RULES = [  # name, move, volx, high ('strict' | 'x60'), market-vol min (before the cooldown; 0 = off), window bars (0 = the trigger bar only)
    ("TODAY (1 % · 3× · strict)", 1.0, 3.0, "strict", 0.0, 0),
    ("1 % + market vol ≥ 1", 1.0, 3.0, "strict", 1.0, 0),
    ("0.75 % + market vol ≥ 1", 0.75, 3.0, "strict", 1.0, 0),
    ("0.5 % + market vol ≥ 1", 0.5, 3.0, "strict", 1.0, 0),
    ("0.3 % · 5× + market vol ≥ 1", 0.3, 5.0, "strict", 1.0, 0),
    ("0.3 % · 5× · excl. last 1 h + market vol ≥ 1", 0.3, 5.0, "x60", 1.0, 0),
    ("0.3 % · 5× + market vol ≥ 1 · window 15 min", 0.3, 5.0, "strict", 1.0, 2),
    ("0.3 % · 5× + market vol ≥ 1 · window 30 min", 0.3, 5.0, "strict", 1.0, 5),
    ("0.3 % · 5× + market vol ≥ 1 · cooldown only after a fill", 0.3, 5.0, "strict", 1.0, 0, True),
    ("0.5 % + market vol ≥ 1 · cooldown only after a fill", 0.5, 3.0, "strict", 1.0, 0, True),
]
t0 = last_closed - int(HOURS * 3600_000)
mk = {}
out = []
for name, mv, vx, hr, gmin, wb, *_of in RULES:
    on_fill = bool(_of and _of[0])
    last, tot = -10**15, []
    lines = []
    for t in b.index[b.index > t0]:
        r = b.loc[t]
        hi = r.hi if hr == "strict" else r.hi_x60
        if not (r.r30 >= mv and r.volx >= vx and r.c >= hi):
            continue
        if t not in mk:
            mk[t] = SO._market(frames, t)
        br, gv = mk[t]
        if gmin > 0 and not (np.isfinite(gv) and gv >= gmin):
            lines.append(f"  {loc(t)}  BTC {r.r30:+.2f}% vol {r.volx:.1f}× mkt {gv:.2f} → refused (market volume)"); continue
        if t - last < 4 * 3600_000:
            lines.append(f"  {loc(t)}  BTC {r.r30:+.2f}% → in cooldown"); continue
        if not on_fill:
            last = t
        taken, fills = set(), []
        for k in range(0, wb + 1):
            bt = int(t) + k * BAR
            if bt not in b.index or len(fills) >= 4:
                break
            for p, atr, pm in picks_at(bt, b.at[bt, "r30"], taken):
                if len(fills) >= 4:
                    break
                taken.add(p)
                te = bt + BAR + MIN
                pnl, why, held = SO._walk(SO._m1(OS.EX, OS._retry, p, te - MIN), te, atr, now_ms)
                fills.append((loc(bt), p.replace("USDT", ""), atr, pm, pnl, why, held))
        if on_fill and fills:
            last = t
        mean = np.mean([f[4] for f in fills if f[4] is not None]) if any(f[4] is not None for f in fills) else None
        if mean is not None:
            tot.append(mean)
        lines.append(f"  {loc(t)}  BTC {r.r30:+.2f}% vol {r.volx:.1f}× mkt {gv:.2f} breadth {br:.0f}% ADXΔ {r.d_adx:+.1f} → TRIGGER · "
                     + (" · ".join(f"{f[1]} @{f[0]} (ATR {f[2]:.2f}, {f[3]:+.2f}%) → {('%+.2f%%' % f[4]) if f[4] is not None else 'open'} {f[5]} {f[6]}m"
                                   for f in fills) if fills else "no pair passed ATR ≥ 1.5 ∧ leading BTC"))
    out.append((name, lines, tot))
print(f"Last {HOURS:g} h, local time (UTC−3), bar OPEN times. Exit = today's SURGE_LONG exit on 1m bars, net of fees.\n")
for name, lines, tot in out:
    print(f"■ {name}: " + (f"{len(tot)} trigger(s) with fills · avg/trigger {np.mean(tot):+.2f}%" if tot else "no trigger with fills"))
    print("\n".join(lines) if lines else "  (no bar met the move / volume / high legs)")
    print()
