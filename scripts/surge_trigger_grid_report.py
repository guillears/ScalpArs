#!/usr/bin/env python3
"""⚡ Oct-4 (operator, watching BTC +2 % in two hours with SURGE silent: "1 % is too late", "the 24 h high should not count the
current move's own highs — everything before the last hour", "the 5-min entry window and the 4 h cooldown make no sense",
"check all together: ADX, breadth, market volume").

Read-only research. Reads the grid produced by scripts/surge_bearrun_review.py with SURGE_MOVE / SURGE_HIGH_TOL / SURGE_SPACING
(reports/backtest_cache/surge_review_m<move>_h<high>_s<spacing>/{events_picks,walk}.csv; baseline = surge_review/ = today's rule).
Every fill = the live SURGE_LONG pick rules (top-20, ATR ≥ 1.5, leader, ≤ 4) walked on 1-s aggTrade paths with the live Bull-Run
exit (LIVE column, net of fees). Units: TRIGGER (window) — one trigger's fills = one observation; day-block bootstrap.

Per trigger (closed bars at the trigger bar, no look-ahead): BTC 5m ADX(14) and its 30-min change, breadth = share of the top-50
by 24 h volume closing above their 5m EMA20 (and its 30-min change), market volume = the bot's global_volume_ratio (Σ bar volume ÷
Σ 48-bar mean, top-50), BTC bar volume × 288-bar median.
Out: reports/SURGE_TRIGGER_GRID_<date>.md
"""
import glob, os, re, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, "scripts")
import surge_bearrun_review as R
from services.frenzy import global_volume_ratio

CACHE = "reports/backtest_cache"; DATE = "2026-10-04"; NB = 3000
rng = np.random.default_rng(20261004)
SPLIT = R.SPLIT


def load_variant(d):
    w = os.path.join(d, "walk.csv"); e = os.path.join(d, "events_picks.csv")
    if not (os.path.exists(w) and os.path.exists(e)):
        return None
    W = pd.read_csv(w); W = W[(W.side == "LONG") & (W.kind == "PICK")]
    E = pd.read_csv(e); E = E[E.side == "LONG"]
    return W, E


variants = {}
BASE = ("1.0", "strict", "4", "3", "-", "0", "")   # (move, 24 h-high rule, cooldown h, BTC vol ×, market-vol gate before the cooldown, extra window bars)
base = load_variant(f"{CACHE}/surge_review")
if base:
    variants[BASE] = base
_HN = {"0": "strict", "x60": "excl. last 1 h", "off": "none"}
for d in sorted(glob.glob(f"{CACHE}/surge_review_m*_s*")):
    tail = os.path.basename(d)[len("surge_review_"):]
    parts = dict((x[0], x[1:]) for x in tail.split("_") if x)
    if not {"m", "h", "s"} <= set(parts):
        continue
    v = load_variant(d)
    if not v:
        continue
    num = lambda x: x.rstrip("0").rstrip(".") if "." in x else x
    variants[(parts["m"], _HN.get(parts["h"], parts["h"]), num(parts["s"]), num(parts.get("v", "3")), parts.get("g", "-"), parts.get("w", "0"),
              "f" if "f" in parts else "")] = v


def label(k):
    return (f"{k[0]} % · vol ≥ {k[3]}× · {k[1]} · {k[2]} h" + (f" · mkt vol ≥ {k[4]} (before cooldown)" if k[4] != "-" else "")
            + (f" · window {5 + 5 * int(k[5])} min" if k[5] != "0" else "") + (" · cooldown only after a fill" if k[6] else ""))


print(f"variants loaded: {len(variants)}")

# ── per-trigger market features (union of every variant's triggers) ──
trig_all = sorted({int(t) for W, E in variants.values() for t in E.trig.unique()})
btc = R._load5("BTCUSDT")
h, l, c = btc.h, btc.l, btc.c
up, dn = h.diff(), -l.diff()
pdm = np.where((up > dn) & (up > 0), up, 0.0); ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
pdi = 100 * pd.Series(pdm, index=btc.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
ndi = 100 * pd.Series(ndm, index=btc.index).ewm(alpha=1 / 14, adjust=False).mean() / atr
adx = (100 * (pdi - ndi).abs() / (pdi + ndi)).ewm(alpha=1 / 14, adjust=False).mean()
med = btc.qv.shift(1).rolling(288).median()
U = dict(R._universe_pairs()); D = {}
for p in U:
    try:
        d = R._load5(p)
    except Exception:
        continue
    d["v24"] = d.qv.rolling(288, min_periods=250).sum(); d["e20"] = d.c.ewm(span=20, adjust=False).mean()
    D[p] = d
print(f"universe loaded: {len(D)}")


def breadth(t):
    snap = [(d.at[t, "v24"], d.at[t, "c"] > d.at[t, "e20"]) for d in D.values() if t in d.index and np.isfinite(d.at[t, "v24"])]
    top = sorted(snap, key=lambda x: -x[0])[:50]
    return np.mean([x[1] for x in top]) * 100 if len(top) >= 30 else np.nan, top


rows = []
for t in trig_all:
    if t not in btc.index:
        continue
    i = btc.index.get_loc(t)
    b0, _ = breadth(t); b6, _ = breadth(t - 6 * R.BAR)
    snap = [(p, d.at[t, "v24"]) for p, d in D.items() if t in d.index and np.isfinite(d.at[t, "v24"])]
    top = [p for p, _ in sorted(snap, key=lambda x: -x[1])[:50]]
    bars = {}
    for p in top:
        d = D[p]; j = d.index.get_loc(t)
        if j >= 47:
            w = d.iloc[j - 47:j + 1]
            bars[p] = [[int(ix), 0, 0, 0, 0, float(v)] for ix, v in zip(w.index, w.vol)]
    rows.append(dict(trig=t, adx=adx.iloc[i], d_adx=adx.iloc[i] - adx.iloc[i - 6], di=pdi.iloc[i] - ndi.iloc[i],
                     breadth=b0, d_breadth=b0 - b6, gvol=R._gvol_at(int(t)), btc_volx=btc.qv.iloc[i] / med.iloc[i]))   # live-gate universe (validated)
F = pd.DataFrame(rows).set_index("trig")
print(f"features: {len(F)} triggers")


def per_trigger(W):
    g = W.groupby("trig").agg(m=("LIVE", "mean"), n=("LIVE", "size"), s=("LIVE", "sum")).reset_index()
    g["day"] = pd.to_datetime(g.trig, unit="ms").dt.strftime("%Y-%m-%d")
    return g.join(F, on="trig")


def ci(g):
    if len(g) < 3:
        return np.nan, np.nan
    days = g.day.unique(); idx = {d: i for i, d in enumerate(days)}; di = g.day.map(idx).values
    s = np.bincount(di, weights=g.m.values, minlength=len(days)); n = np.bincount(di, minlength=len(days)).astype(float)
    w = np.stack([np.bincount(rng.integers(0, len(days), len(days)), minlength=len(days)) for _ in range(NB)]).astype(float)
    mm = (w @ s) / np.maximum(w @ n, 1)
    return np.percentile(mm, 2.5), np.percentile(mm, 97.5)


def line(g, W, label):
    if not len(g):
        return f"| {label} | 0 | | | | | | | |"
    lo, hi = ci(g)
    h1 = g[g.trig < SPLIT].m.mean(); h2 = g[g.trig >= SPLIT].m.mean()
    mon = pd.to_datetime(g.trig, unit="ms").dt.strftime("%m"); mp = g.groupby(mon).m.mean()
    ww = W[W.trig.isin(g.trig)]
    return (f"| {label} | {len(g)} | {g.day.nunique()} | {len(ww)} | {(ww.LIVE > 0).mean() * 100:.0f}% | **{g.m.mean():+.3f}** | "
            f"[{lo:+.3f}, {hi:+.3f}] | {h1:+.3f} / {h2:+.3f} | {(mp > 0).sum()}/{len(mp)} |")


OUT = [f"# ⚡ SURGE LONG trigger grid — trigger size × 24 h-high rule × cooldown, with market conditions ({DATE})", "",
       "Each fill = today's SURGE_LONG pick rules (top-20 by volume, ATR ≥ 1.5 %, outrunning BTC, ≤ 4 per trigger) walked on 1-second "
       "trade prices with today's Bull-Run exit, net of fees, entry 60 s after the trigger bar. **avg/trigger** = the mean of each "
       "trigger's fills (one trigger = one observation); CI = 95 % day-block bootstrap. H1 = Jan–Apr, H2 = May–Oct. "
       "Not modelled: open-slot limits across overlapping triggers, the 0.3 % dislocation guard, the bot's real latency spread.", "",
       "## 1 · The grid", "",
       "| move ≥ · BTC volume · 24 h-high rule · cooldown | triggers | days | fills | fill WR | avg/trigger % | 95 % CI | H1 / H2 | months > 0 |",
       "|---|---|---|---|---|---|---|---|---|"]
order = sorted(variants, key=lambda k: (k[4] != "-", k[6], int(k[5]), -float(k[0]), -float(k[3]), ["strict", "excl. last 1 h", "none"].index(k[1]) if k[1] in ["strict", "excl. last 1 h", "none"] else 9, -float(k[2])))
G = {}
for k in order:
    W, E = variants[k]; g = per_trigger(W); G[k] = (g, W)
    tag = " ← today" if k == BASE else ""
    OUT.append(line(g, W, label(k) + tag))
OUT += ["", "## 2 · Market conditions at the trigger (every variant, split)", "",
        "ADX↑ = BTC 5m ADX higher than 30 min earlier · breadth ≥ 60 % of the top-50 above their EMA20 · breadth↑ = higher than "
        "30 min earlier · mkt vol ≥ 1 = the bot's market-volume ratio ≥ 1 · ALL = ADX↑ ∧ breadth↑ ∧ mkt vol ≥ 1.", ""]
conds = [("ADX↑", lambda g: g.d_adx > 0), ("ADX↓", lambda g: g.d_adx <= 0), ("breadth ≥ 60", lambda g: g.breadth >= 60),
         ("breadth < 60", lambda g: g.breadth < 60), ("breadth↑", lambda g: g.d_breadth > 0), ("breadth↓", lambda g: g.d_breadth <= 0),
         ("mkt vol ≥ 1", lambda g: g.gvol >= 1), ("mkt vol < 1", lambda g: g.gvol < 1),
         ("ALL", lambda g: (g.d_adx > 0) & (g.d_breadth > 0) & (g.gvol >= 1)),
         ("not ALL", lambda g: ~((g.d_adx > 0) & (g.d_breadth > 0) & (g.gvol >= 1)))]
for k in order:
    g, W = G[k]
    if len(g) < 20:
        continue
    OUT += [f"**{label(k)}**", "",
            "| condition | triggers | days | fills | fill WR | avg/trigger % | 95 % CI | H1 / H2 | months > 0 |", "|---|---|---|---|---|---|---|---|---|"]
    for name, fn in conds:
        OUT.append(line(g[fn(g).fillna(False).values], W, name))
    OUT.append("")

# lead time: how much earlier a variant fires on the moves today's rule catches
OUT += ["## 3 · Lead time on the moves today's rule catches", "",
        "For every trigger of today's rule, the earliest trigger of the variant in the 3 h before it (same move). "
        "BTC gain given away = BTC % from the variant's trigger close to today's trigger close.", "",
        "| variant | today's triggers matched | median minutes earlier | median BTC % already moved by today's trigger |", "|---|---|---|---|"]
if BASE in G:
    bt = G[BASE][0].trig.values
    for k in order:
        if k == BASE:
            continue
        vt = np.sort(G[k][0].trig.values); mins, gains = [], []
        for t in bt:
            cand = vt[(vt <= t) & (vt >= t - 3 * 3600_000)]
            if len(cand):
                t0 = cand[0]; mins.append((t - t0) / 60_000)
                gains.append((btc.c.get(t) / btc.c.get(t0) - 1) * 100 if t0 in btc.c.index and t in btc.c.index else np.nan)
        OUT.append(f"| {label(k)} | {len(mins)}/{len(bt)} | {np.median(mins) if mins else float('nan'):.0f} | "
                   f"{np.nanmedian(gains) if gains else float('nan'):+.2f} |")
OUT += ["", "## Blind spots", "",
        "- One exit (today's Bull-Run exit) — a faster trigger may want a different exit; not tuned here (one change at a time).",
        "- The entry window is modelled as 'enter 60 s after the trigger bar'; a 1 h cooldown is the proxy for 'stay active while the move "
        "lasts' (consecutive qualifying bars ≥ 1 h apart become new triggers). Overlapping open positions / slot caps are not modelled.",
        "- Grid = 18 cells on the same year: the best cell is optimistic (pick-the-best bias); a winner needs the haircut + fresh probe "
        "before any size."]
md = f"reports/SURGE_TRIGGER_GRID_{DATE}.md"
open(md, "w").write("\n".join(OUT) + "\n")
print("\n".join(OUT)); print("→", md)
