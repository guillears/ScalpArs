#!/usr/bin/env python3
"""🔥 HOT-STATE SCALP backtest on 1-second data (operator, 2026-10-01 — 18 manual MOVR longs in 20 minutes, +$1,729 at 20–50×:
"this is scalping: find a gem and do what I did"; "what called the attention was volume + ATR + overall direction ascending").

PRE-DECLARED (written before any result; MOVR 09-30 / 10-01 is the inspiration and lies AFTER the 5m cache end → not in the evidence):
  universe   every pair of the 5m cache (Jan-29 → Sep-27 2026) with ≥ 30 days of history, prior-24 h quote volume ≥ $20M, and a
             Binance SPOT 1-second feed under the same symbol (futures-only and 1000-prefixed pairs are skipped and COUNTED)
  HOT        from CLOSED 5m bars: ATR(14) ≥ 2 % of price ∧ RSI(14) ≥ 70, and the live price ≥ 3 % above the 5m EMA5
             (the operator's entries: ATR 3.4 %, RSI 72–81, stretch 1–5 %)
  GEM        HOT ∧ the pair is up ≥ 20 % over the prior 24 h ("ascending for a while") ∧ its prior-24 h volume ≥ 3× its average
             daily volume of the 7 days before that ("volume called the attention")
  episode    HOT 5m bars less than 30 min apart = one episode; 1s data is fetched from its first bar to 30 min past its last
  trades     SEQUENTIAL, one position per pair: LONG at the first HOT second; after an exit wait 5 s and re-enter while HOT
  exits      gross TP / SL %: A 0.59 / 1.11 (the operator's +0.5 / −1.2 net) · B 1.09 / 1.11 · C 2.09 / 1.11 · D 3.09 / 1.51 ·
             E 0.59 / 0.61 · F stop 1.5, once +1.5 a trail 1.0 below the peak · max hold 30 min; inside one second the stop
             is checked first
             + TIME-STOP exits (added 10-01 16:55 UTC, before any result of this run was read; the operator's 22 trades show
             winners closing in a median 16 s and the losers he cut by hand at a median 47 s — "if it does not go fast, get
             out"): G 0.59 / 1.11 out at market after 30 s · H 0.59 / 1.11 after 60 s · I 1.09 / 1.11 after 60 s
  costs      0.09 % fees + 0.02 % slippage per round trip (the bot's measured fills: master batch +0.01 %, the operator's tape
             replay 0.016 % — the first draft's 0.10 % was 5–10× too harsh); a stress column adds 0.10 % (thin pairs)
  control    the same episodes' windows: LONG every 60 s on NON-HOT seconds (independent trades, same exit) — does the HOT
             timing beat a random entry on the same pair in the same hours?
  units      UTC DAYS (mean over the day's trades; t-interval), halves Jan–Apr / May–Sep
  PASS bar   ≥ 30 days · day-mean > 0 in BOTH halves · one-sided Bonferroni bound (18 tests = 2 cohorts × 9 exits) > 0 ·
             HOT − control by day: 95 % interval > 0 · top-3 days < 50 % of the summed profit · no pair > 30 % of it
  caveats    spot prices (not the futures tape); only pairs still listed; stop / target fill at their level inside a second
Usage: venv/bin/python scripts/hot_scalp_backtest.py [--report-only]
       → reports/HOT_SCALP_TRADES_2026-10-01.csv (not for git) + reports/HOT_SCALP_BACKTEST_2026-10-01.md"""
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K5 = os.path.join(ROOT, "reports", "backtest_cache", "k5m_full")
TRADES = os.path.join(ROOT, "reports", "HOT_SCALP_TRADES_2026-10-01.csv")
DONE = os.path.join(ROOT, "reports", "backtest_cache", "hot_scalp_done.json")
OUT = os.path.join(ROOT, "reports", "HOT_SCALP_BACKTEST_2026-10-01.md")
BAR, HOLD, FEE, SLIP, STRESS, SPLIT = 300_000, 1800, 0.09, 0.02, 0.10, "2026-05-01"   # SLIP = measured (master 0.01 %, tape replay 0.016 %); STRESS = thin-pair line
EXITS = {"A +0.59 / −1.11": (0.59, 1.11, None), "B +1.09 / −1.11": (1.09, 1.11, None), "C +2.09 / −1.11": (2.09, 1.11, None),
         "D +3.09 / −1.51": (3.09, 1.51, None), "E +0.59 / −0.61": (0.59, 0.61, None), "F trail 1.0 after +1.5, stop 1.5": (None, 1.5, (1.5, 1.0)),
         "G +0.59 / −1.11, out after 30 s": (0.59, 1.11, 30), "H +0.59 / −1.11, out after 60 s": (0.59, 1.11, 60),
         "I +1.09 / −1.11, out after 60 s": (1.09, 1.11, 60)}


def get(url):
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(url, timeout=25))
        except urllib.error.HTTPError as e:
            if e.code == 400:
                return None
            time.sleep(3 + 5 * k)
        except Exception:
            time.sleep(3 + 5 * k)
    return None


def k1s(pair, s, e):
    rows = []
    while s < e:
        r = get(f"https://api.binance.com/api/v3/klines?symbol={pair}&interval=1s&startTime={s}&endTime={e}&limit=1000")
        if not r:
            break
        rows += [(int(x[0]), float(x[2]), float(x[3]), float(x[4])) for x in r]
        s = int(r[-1][0]) + 1000; time.sleep(0.04)
    if not rows:
        return None
    return pd.DataFrame(rows, columns=["t", "h", "l", "c"]).drop_duplicates("t").set_index("t")


def frame5(pair):
    """5m indicators usable DURING each bar (all from bars closed before it) + the episode list."""
    d = pd.read_csv(os.path.join(K5, pair + ".csv")).drop_duplicates("open_time").set_index("open_time").sort_index()
    if len(d) < 288 * 30:
        return None, []
    pc = d.c.shift(1); tr = np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs()))
    d["atr"] = (tr.ewm(alpha=1 / 14, adjust=False).mean() / d.c * 100).shift(1)
    d["ema5"] = d.c.ewm(span=5, adjust=False).mean().shift(1)
    dl = d.c.diff(); up = dl.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean(); dn = (-dl.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    d["rsi"] = (100 - 100 / (1 + up / dn.replace(0, np.nan))).shift(1)
    d["q24"] = d.qvol.rolling(288).sum().shift(1)
    d["r24"] = (d.c.shift(1) / d.c.shift(289) - 1) * 100
    d["q7"] = d.qvol.rolling(288 * 7).sum().shift(289) / 7                 # average daily volume of the 7 days BEFORE the last 24 h
    hot = (d.atr >= 2) & (d.rsi >= 70) & (d.h >= d.ema5 * 1.03) & (d.q24 >= 20e6)
    hot.iloc[:288 * 30] = False
    t = d.index.values[hot.values]
    eps = [] if not len(t) else [(int(e[0]), int(e[-1])) for e in np.split(t, np.where(np.diff(t) > 30 * 60_000)[0] + 1)]
    return d, eps


def walk(h, l, c, i, x, entry):
    """LONG from `entry` (the close of second i). Returns (gross %, exit index, how)."""
    tp, sl, tr = x; hold = tr if isinstance(tr, int) else HOLD
    if isinstance(tr, int):
        tr = None                                                       # an int in the third slot = a time stop in seconds
    j1 = min(i + 1 + hold, len(c)); H, Lo = h[i + 1:j1], l[i + 1:j1]
    if not len(H):
        return 0.0, i, "TIME"
    if tr is None:
        hs = np.nonzero(Lo <= entry * (1 - sl / 100))[0]; ht = np.nonzero(H >= entry * (1 + tp / 100))[0]
        a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
        if a == b == 10**9:
            return (c[j1 - 1] / entry - 1) * 100, j1 - 1, "TIME"
        return (-sl, i + 1 + a, "STOP") if a <= b else (tp, i + 1 + b, "TP")
    arm, dist = tr
    pk = np.maximum.accumulate(np.concatenate(([entry], H)))[:-1]         # the peak BEFORE each second
    lvl = np.where(pk >= entry * (1 + arm / 100), pk * (1 - dist / 100), entry * (1 - sl / 100))
    hs = np.nonzero(Lo <= lvl)[0]
    if not len(hs):
        return (c[j1 - 1] / entry - 1) * 100, j1 - 1, "TIME"
    k = hs[0]
    return (lvl[k] / entry - 1) * 100, i + 1 + k, ("STOP" if lvl[k] < entry else "TRAIL")


RAW = os.path.join(ROOT, "reports", "backtest_cache", "k1s_hot")       # one small npz per episode → re-analyses need no refetch
EXC = os.path.join(ROOT, "reports", "HOT_SCALP_EXCURSIONS_2026-10-01.csv")
TARGET = 0.59


def load_1s(pair, t0, t1):
    f = os.path.join(RAW, f"{pair}_{t0}.npz")
    if os.path.exists(f):
        z = np.load(f); return pd.DataFrame({"h": z["h"], "l": z["l"], "c": z["c"]}, index=z["t"])
    s = k1s(pair, t0, t1 + BAR + HOLD * 1000)
    if s is None or len(s) < 600:
        return None
    os.makedirs(RAW, exist_ok=True)
    np.savez_compressed(f, t=s.index.values.astype("int64"), h=s.h.values.astype("float64"), l=s.l.values.astype("float64"), c=s.c.values.astype("float64"))
    return s


def excursions(pair, eid, ts, h, l, c, hot, gem):
    """For HOT seconds 30 s apart: how long until +0.59 % (within 30 min) and how deep the price dipped first — in the first
    30 / 60 / 120 s and up to the target. The raw material for choosing a stop width and a time stop."""
    out = []; hi = np.nonzero(hot)[0]; last = -10**9
    for i in hi:
        if i - last < 30:
            continue
        last = i; e = c[i]; j1 = min(i + 1 + HOLD, len(c)); H, Lo = h[i + 1:j1], l[i + 1:j1]
        if len(H) < 120:
            continue
        ht = np.nonzero(H >= e * (1 + TARGET / 100))[0]; t = int(ht[0]) if len(ht) else -1
        dip = lambda n: round((Lo[:n].min() / e - 1) * 100, 4)
        cut = (t + 1) if t >= 0 else len(Lo)
        out.append((pair, eid, int(ts[i]), bool(gem[i]), bool(i == hi[0]), t, dip(min(30, cut)), dip(min(60, cut)), dip(min(120, cut)), dip(cut),
                    round((c[min(i + 30, len(c) - 1)] / e - 1) * 100, 4), round((c[min(i + 60, len(c) - 1)] / e - 1) * 100, 4)))
    return out


def run_episode(pair, d5, t0, t1, eid):
    s = load_1s(pair, t0, t1)
    if s is None or len(s) < 600:
        return None
    ts = s.index.values; bar = ts // BAR * BAR
    st = d5.reindex(bar)
    atr, rsi, ema5, q24, r24, q7 = (st[k].values for k in ("atr", "rsi", "ema5", "q24", "r24", "q7"))
    h, l, c = s.h.values, s.l.values, s.c.values
    hot = (atr >= 2) & (rsi >= 70) & (c >= ema5 * 1.03) & (q24 >= 20e6)
    gem = hot & (r24 >= 20) & (q24 >= 3 * q7)
    hot[len(c) - 60:] = False                                           # no entry in the last minute of the data
    hi = np.nonzero(hot)[0]; out = []
    ex = excursions(pair, eid, ts, h, l, c, hot, gem)
    if ex:
        pd.DataFrame(ex, columns=["pair", "eid", "t", "gem", "first", "secs_to_target", "dip30", "dip60", "dip120", "dip_to_target", "px30", "px60"]
                     ).to_csv(EXC, mode="a", header=not os.path.exists(EXC), index=False)
    if len(hi):
        for en, x in EXITS.items():
            i = hi[0]
            while True:
                r, j, how = walk(h, l, c, int(i), x, c[i])
                out.append((pair, eid, int(ts[i]), en, "HOT", bool(gem[i]), round(r, 4), how, int(j - i)))
                k = np.searchsorted(hi, j + 5)
                if k >= len(hi):
                    break
                i = hi[k]
    ci = np.arange(300, len(c) - 60, 60); ci = ci[~hot[ci] & np.isfinite(atr[ci])]
    for en, x in EXITS.items():
        for i in ci:
            r, j, how = walk(h, l, c, int(i), x, c[i])
            out.append((pair, eid, int(ts[i]), en, "CTRL", False, round(r, 4), how, int(j - i)))
    return out


def tq(p, n):                                                           # one-sided t quantile (Cornish–Fisher), no scipy
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p); g1 = (z**3 + z) / 4; g2 = (5 * z**5 + 16 * z**3 + 3 * z) / 96
    v = max(n - 1, 1); return z + g1 / v + g2 / v**2


def report():
    T = pd.read_csv(TRADES); meta = json.load(open(DONE))
    T["day"] = pd.to_datetime(T.t, unit="ms").dt.strftime("%Y-%m-%d"); T["net"] = T.r - FEE - SLIP
    L = ["# 🔥 Hot-state scalp backtest — 1-second data, all pairs, Jan → Sep 2026", "",
         f"Episodes detected: {meta['episodes']} on {meta['pairs_all']} pairs · with a spot 1s feed: {meta['fetched']} episodes on {meta['pairs_spot']} pairs "
         f"({meta['skipped_nospot']} episodes skipped: futures-only or 1000-prefixed pairs; {meta['skipped_nodata']} with no data). "
         "LONG, one position per pair, re-entering while the state lasts. Net = after 0.09 % fees and 0.02 % measured slippage; stress = a further 0.10 %. "
         "No-edge rate = SL ÷ (TP + SL). Day range = 95 % interval on day means; strict bound = one-sided Bonferroni over 18 tests.", ""]
    for coh, m in (("HOT (ATR ≥ 2 %, RSI ≥ 70, price ≥ 3 % above EMA5)", T.kind == "HOT"), ("GEM (HOT ∧ up ≥ 20 % in 24 h ∧ volume ≥ 3× its week)", (T.kind == "HOT") & T.gem)):
        L += [f"## {coh}", "",
              "| Exit | trades | episodes | pairs | days | hit rate | no-edge | breakeven | net % / trade | stress −0.10 | day mean [95 %] | strict bound | Jan–Apr / May–Sep | control net % | HOT − control by day [95 %] | top-3 days | top pair | PASS |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for en, x in EXITS.items():
            g = T[m & (T.exit == en)]
            if len(g) < 5:
                L.append(f"| {en} | {len(g)} | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – |"); continue
            dm = g.groupby("day").net.mean(); n = len(dm); se = dm.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            lo, hi = dm.mean() - tq(0.975, n) * se, dm.mean() + tq(0.975, n) * se; bon = dm.mean() - tq(1 - 0.05 / 18, n) * se
            h1, h2 = dm[dm.index < SPLIT].mean(), dm[dm.index >= SPLIT].mean()
            ctl = T[(T.kind == "CTRL") & (T.exit == en) & T.eid.isin(g.eid.unique())]
            cd = ctl.groupby("day").net.mean(); pr = (dm - cd).dropna(); pse = pr.std(ddof=1) / np.sqrt(len(pr)) if len(pr) > 1 else np.nan
            plo, phi = pr.mean() - tq(0.975, len(pr)) * pse, pr.mean() + tq(0.975, len(pr)) * pse
            ds = g.groupby("day").net.sum(); pos = ds[ds > 0].sum(); top3 = ds.sort_values(ascending=False).head(3).sum() / pos * 100 if pos > 0 else np.nan
            ps = g.groupby("pair").net.sum(); tp_ = ps.max() / ps[ps > 0].sum() * 100 if (ps > 0).any() else np.nan
            if x[0] is None or isinstance(x[2], int):
                hit = ((g.how == "TP").mean() if x[0] is not None else (g.r > 0).mean()) * 100; ne = be = "–"
            else:
                hit = (g.how == "TP").mean() * 100; ne = f"{x[1] / (x[0] + x[1]) * 100:.0f}%"; be = f"{(x[1] + FEE + SLIP) / (x[0] + x[1]) * 100:.0f}%"
            ok = (n >= 30 and h1 > 0 and h2 > 0 and bon > 0 and plo > 0 and g.net.sum() > 0 and top3 < 50 and tp_ <= 30)
            L.append(f"| {en} | {len(g):,} | {g.eid.nunique()} | {g.pair.nunique()} | {n} | {hit:.0f}% | {ne} | {be} | {g.net.mean():+.3f} | {g.net.mean() - STRESS:+.3f} | "
                     f"{dm.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {bon:+.3f} | {h1:+.3f} / {h2:+.3f} | {ctl.net.mean():+.3f} | {pr.mean():+.3f} [{plo:+.3f}, {phi:+.3f}] | "
                     f"{top3:.0f}% | {tp_:.0f}% | {'✅' if ok else '—'} |")
        L.append("")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        report(); sys.exit(0)
    info = get("https://api.binance.com/api/v3/exchangeInfo") or {}
    spot = {s["symbol"] for s in info.get("symbols", [])}
    if not spot:
        raise SystemExit("spot exchangeInfo failed")
    meta = json.load(open(DONE)) if os.path.exists(DONE) else dict(done=[], episodes=0, pairs_all=0, fetched=0, pairs_spot=0, skipped_nospot=0, skipped_nodata=0)
    done = set(meta["done"]); first = not os.path.exists(TRADES)
    files = sorted(glob.glob(os.path.join(K5, "*.csv"))); t_start = time.time()
    for n_, f in enumerate(files):
        pair = os.path.basename(f)[:-4]
        if pair in done or pair in ("BTCUSDT", "ETHUSDT"):
            continue
        d5, eps = frame5(pair)
        if eps:
            meta["episodes"] += len(eps); meta["pairs_all"] += 1
            if pair not in spot or pair.startswith("1000"):
                meta["skipped_nospot"] += len(eps)
            else:
                got = 0; rows = []
                for k, (t0, t1) in enumerate(eps):
                    r = run_episode(pair, d5, t0, t1, f"{pair}:{t0}")
                    if r is None:
                        meta["skipped_nodata"] += 1
                    else:
                        got += 1; rows += r
                if got:
                    meta["fetched"] += got; meta["pairs_spot"] += 1
                    pd.DataFrame(rows, columns=["pair", "eid", "t", "exit", "kind", "gem", "r", "how", "secs"]).to_csv(TRADES, mode="a", header=first, index=False); first = False
        done.add(pair); meta["done"] = sorted(done); json.dump(meta, open(DONE, "w"))
        print(f"[{n_ + 1}/{len(files)}] {pair}: {len(eps)} episodes · fetched so far {meta['fetched']} · {time.time() - t_start:.0f}s", flush=True)
    report()
