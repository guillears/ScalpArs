#!/usr/bin/env python3
"""🔬 One pair, 48 hours, every variable (operator, 2026-10-01: "analyse MOVR during all day, today and yesterday… high ATR +
abnormal volume + all other variables… how many entries and exits we could have… shorts and longs. In detail").

Data (public, cached in reports/backtest_cache/pair48/): futures 1m + 5m + 1d klines (with taker-buy volume) for the features,
spot 1-second klines for the exit path (a +0.6 % target is reached inside a minute at this ATR). Grid: every closed minute of
the last 48 h, LONG and SHORT evaluated at each. Features use CLOSED bars only.
  5m   EMA5/8/13/20/50 gaps and stack, price vs EMA20/EMA50, EMA20 slope, RSI14, ADX14, +DI, −DI, ATR %, bar volume vs its 20-bar
       mean, last-hour volume vs the prior 24 h, 24 h range position, 1 h / 4 h return
  1m   EMA5−EMA13 gap, RSI14, return 1 / 3 / 5 / 15 min, volume vs its 20-bar mean, taker-buy share 1 m / 5 m, stretch from
       EMA13, position in the last 15-min range, bar range
Signed features are direction-adjusted (× +1 for a LONG, × −1 for a SHORT) so "with the trade" reads the same on both sides.
Exits (gross): A +0.59 / −1.11 · B +1.09 / −1.11 · C +2.09 / −1.11 · D +3.09 / −1.51 · 30-min limit · stop first inside a
second · cost 0.09 % fees + 0.02 % slippage.
THIS IS IN-SAMPLE ON ONE PAIR. The only honesty check available is DAY 1 vs DAY 2: thresholds are cut on day 1 and a rule
counts only if it is positive on BOTH days. Whatever survives is a description of these 48 h, not a tested strategy.
Usage: venv/bin/python scripts/movr_48h_study.py [PAIR] [END_UTC] → reports/PAIR48_<PAIR>_<date>.md"""
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "reports", "backtest_cache", "pair48"); COST = 0.11; HOLD = 1800
EXITS = {"A +0.59/−1.11": (0.59, 1.11), "B +1.09/−1.11": (1.09, 1.11), "C +2.09/−1.11": (2.09, 1.11), "D +3.09/−1.51": (3.09, 1.51)}


def get(u):
    for k in range(5):
        try:
            return json.load(urllib.request.urlopen(u, timeout=25))
        except Exception:
            time.sleep(2 + 3 * k)
    raise SystemExit("fetch failed: " + u[:90])


def klines(base, pair, iv, s, e, lim):
    f = os.path.join(CACHE, f"{pair}_{'fut' if 'fapi' in base else 'spot'}_{iv}_{s}_{e}.csv")
    if os.path.exists(f):
        return pd.read_csv(f).set_index("t")
    rows = []; cur = s
    while cur < e:
        r = get(f"{base}?symbol={pair}&interval={iv}&startTime={cur}&endTime={e}&limit={lim}")
        if not r:
            break
        rows += [(int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7]), float(x[10])) for x in r]
        cur = int(r[-1][0]) + 1; time.sleep(0.08)
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "q", "tb"]).drop_duplicates("t")
    os.makedirs(CACHE, exist_ok=True); d.to_csv(f, index=False)
    return d.set_index("t")


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n=14):
    d = c.diff(); u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); v = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / v.replace(0, np.nan))


def adx(d, n=14):
    up, dn = d.h.diff(), -d.l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index); ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    pc = d.c.shift(1); tr = np.maximum(d.h - d.l, np.maximum((d.h - pc).abs(), (d.l - pc).abs()))
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / atr; ndi = 100 * ndm.ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, ndi, atr / d.c * 100


def walk(h, l, c, i, sg, tp, sl):
    e = c[i]; H, L = h[i + 1:i + 1 + HOLD], l[i + 1:i + 1 + HOLD]
    if not len(H):
        return np.nan
    if sg > 0:
        hs = np.nonzero(L <= e * (1 - sl / 100))[0]; ht = np.nonzero(H >= e * (1 + tp / 100))[0]
    else:
        hs = np.nonzero(H >= e * (1 + sl / 100))[0]; ht = np.nonzero(L <= e * (1 - tp / 100))[0]
    a = hs[0] if len(hs) else 10**9; b = ht[0] if len(ht) else 10**9
    if a == b == 10**9:
        return sg * (c[min(i + HOLD, len(c) - 1)] / e - 1) * 100
    return -sl if a <= b else tp


if __name__ == "__main__":
    pair = sys.argv[1] if len(sys.argv) > 1 else "MOVRUSDT"
    end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.utcnow().tz_localize(None).floor("h")
    e_ms = int(end.timestamp() * 1000); s_ms = e_ms - 48 * 3600_000; FAPI, SAPI = "https://fapi.binance.com/fapi/v1/klines", "https://api.binance.com/api/v3/klines"
    f5 = klines(FAPI, pair, "5m", s_ms - 5 * 86400_000, e_ms, 1500); f1 = klines(FAPI, pair, "1m", s_ms - 6 * 3600_000, e_ms, 1500)
    fd = klines(FAPI, pair, "1d", s_ms - 45 * 86400_000, e_ms, 100); s1 = klines(SAPI, pair, "1s", s_ms, e_ms, 1000)
    # ---------- 5m features, usable from the bar's close time
    a5, pdi, ndi, atr5 = adx(f5)
    F5 = pd.DataFrame({"gap5_20": (ema(f5.c, 5) / ema(f5.c, 20) - 1) * 100, "gap5_8": (ema(f5.c, 5) / ema(f5.c, 8) - 1) * 100,
                       "gap8_13": (ema(f5.c, 8) / ema(f5.c, 13) - 1) * 100, "px_vs_ema20": (f5.c / ema(f5.c, 20) - 1) * 100,
                       "px_vs_ema50": (f5.c / ema(f5.c, 50) - 1) * 100, "ema20_slope": (ema(f5.c, 20) / ema(f5.c, 20).shift(3) - 1) * 100,
                       "rsi5m": rsi(f5.c) - 50, "adx5m": a5, "di_diff": pdi - ndi, "atr5m": atr5,
                       "vol5m_ratio": f5.q / f5.q.rolling(20).mean().shift(1),
                       "vol1h_vs_24h": f5.q.rolling(12).sum() / (f5.q.rolling(288).sum().shift(12) / 24),
                       "range_pos24h": ((f5.c - f5.l.rolling(288).min()) / (f5.h.rolling(288).max() - f5.l.rolling(288).min()) * 100) - 50,
                       "ret_1h": (f5.c / f5.c.shift(12) - 1) * 100, "ret_4h": (f5.c / f5.c.shift(48) - 1) * 100})
    st = np.sign(ema(f5.c, 5) - ema(f5.c, 8)) + np.sign(ema(f5.c, 8) - ema(f5.c, 13)) + np.sign(ema(f5.c, 13) - ema(f5.c, 20))
    F5["ema_stack"] = st                                                # +3 = fully stacked up, −3 = fully stacked down
    F5.index = F5.index + 300_000                                       # known at the bar's CLOSE
    # ---------- 1m features, known at the bar's close
    F1 = pd.DataFrame({"gap1m_5_13": (ema(f1.c, 5) / ema(f1.c, 13) - 1) * 100, "rsi1m": rsi(f1.c) - 50,
                       "ret_1m": (f1.c / f1.c.shift(1) - 1) * 100, "ret_3m": (f1.c / f1.c.shift(3) - 1) * 100,
                       "ret_5m": (f1.c / f1.c.shift(5) - 1) * 100, "ret_15m": (f1.c / f1.c.shift(15) - 1) * 100,
                       "vol1m_ratio": f1.q / f1.q.rolling(20).mean().shift(1), "buy_share_1m": (f1.tb / f1.q.replace(0, np.nan) - 0.5) * 100,
                       "buy_share_5m": (f1.tb.rolling(5).sum() / f1.q.rolling(5).sum().replace(0, np.nan) - 0.5) * 100,
                       "stretch_1m_ema13": (f1.c / ema(f1.c, 13) - 1) * 100,
                       "pos_15m": ((f1.c - f1.l.rolling(15).min()) / (f1.h.rolling(15).max() - f1.l.rolling(15).min()) * 100) - 50,
                       "bar_range_1m": (f1.h / f1.l - 1) * 100})
    F1.index = F1.index + 60_000
    SIGNED = ["gap5_20", "gap5_8", "gap8_13", "px_vs_ema20", "px_vs_ema50", "ema20_slope", "rsi5m", "di_diff", "range_pos24h", "ret_1h", "ret_4h",
              "ema_stack", "gap1m_5_13", "rsi1m", "ret_1m", "ret_3m", "ret_5m", "ret_15m", "buy_share_1m", "buy_share_5m", "stretch_1m_ema13", "pos_15m"]
    UNSIGNED = ["adx5m", "atr5m", "vol5m_ratio", "vol1h_vs_24h", "vol1m_ratio", "bar_range_1m"]
    ts = s1.index.values; h, l, c = s1.h.values, s1.l.values, s1.c.values; pos = {int(t): i for i, t in enumerate(ts)}
    rows = []
    for t in range(s_ms + 60_000, e_ms - HOLD * 1000, 60_000):
        if t not in pos or t not in F1.index:
            continue
        i = pos[t] - 1                                                  # the last second of the closed minute
        b5 = F5.loc[:t].iloc[-1]; b1 = F1.loc[t]
        for sg in (1, -1):
            r = dict(t=t, side="LONG" if sg > 0 else "SHORT", day=pd.Timestamp(t, unit="ms").strftime("%m-%d"), px=c[i])
            for k in SIGNED:
                r[k] = sg * float(b5[k] if k in F5.columns else b1[k])
            for k in UNSIGNED:
                r[k] = float(b5[k] if k in F5.columns else b1[k])
            for en, (tp, sl) in EXITS.items():
                r[en] = walk(h, l, c, i, sg, tp, sl)
            rows.append(r)
    T = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan); T.to_csv(os.path.join(CACHE, f"{pair}_grid.csv"), index=False)
    days = sorted(T.day.unique()); mid = pd.Timestamp(s_ms + 24 * 3600_000, unit="ms"); T["half"] = np.where(T.t < s_ms + 24 * 3600_000, "first 24h", "last 24h")
    L = [f"# 🔬 {pair} — the last 48 hours, every variable ({pd.Timestamp(s_ms, unit='ms'):%m-%d %H:%M} → {end:%m-%d %H:%M} UTC)", ""]
    dv = fd.q; base = dv.iloc[:-3].tail(30).mean()
    L += ["## What made the pair special", "",
          f"- Price {c[0]:.4g} → {c[-1]:.4g} ({(c[-1] / c[0] - 1) * 100:+.0f} %), low {l.min():.4g}, high {h.max():.4g} (range {(h.max() / l.min() - 1) * 100:.0f} %).",
          f"- Daily futures volume: 30-day normal ${base / 1e6:,.0f}M → last three days " + " · ".join(f"${v / 1e6:,.0f}M ({v / base:.0f}×)" for v in dv.tail(3)),
          f"- 5-minute ATR: median {T.atr5m.median():.2f} % in these 48 h (first 24 h {T[T.half == 'first 24h'].atr5m.median():.2f} %, last 24 h {T[T.half == 'last 24h'].atr5m.median():.2f} %); "
          f"the bot's usual pair is ~0.46 %.",
          f"- Trend state by minute: EMA stack fully up {(T[T.side == 'LONG'].ema_stack == 3).mean() * 100:.0f} % of the time, fully down {(T[T.side == 'LONG'].ema_stack == -3).mean() * 100:.0f} %, mixed the rest.", ""]
    L += ["## Baseline — entering at EVERY minute (no rule)", "", "| Side | Half | minutes | " + " | ".join(f"{k}: win % · net %" for k in EXITS) + " |", "|---|---|---|" + "---|" * len(EXITS)]
    for sd in ("LONG", "SHORT"):
        for hf in ("first 24h", "last 24h"):
            g = T[(T.side == sd) & (T.half == hf)]
            L.append(f"| {sd} | {hf} | {len(g)} | " + " | ".join(f"{(g[k] > 0).mean() * 100:.0f}% · {g[k].mean() - COST:+.3f}" for k in EXITS) + " |")
    L += ["", "## Which variable separates winners from losers? (exit A; terciles cut on the first 24 h; LOW · MID · HIGH win %)", "",
          "Signed variables are in the direction of the trade (HIGH = strongly with the trade).", "",
          "| Variable | LONG first 24 h | LONG last 24 h | SHORT first 24 h | SHORT last 24 h | same best end on all four? |", "|---|---|---|---|---|---|"]
    A = "A +0.59/−1.11"; scores = []
    for f in SIGNED + UNSIGNED:
        cells = []; ends = []
        for sd in ("LONG", "SHORT"):
            g1 = T[(T.side == sd) & (T.half == "first 24h")]; ed = np.nanquantile(g1[f], [1 / 3, 2 / 3])
            for hf in ("first 24h", "last 24h"):
                g = T[(T.side == sd) & (T.half == hf)]; q = np.searchsorted(ed, g[f].values, side="right")
                w = [(g[A][q == k] > 0).mean() * 100 if (q == k).sum() >= 40 else np.nan for k in range(3)]
                cells.append(" · ".join("–" if np.isnan(x) else f"{x:.0f}" for x in w)); ends.append(int(np.nanargmax(w)) if not np.all(np.isnan(w)) else -1)
                scores.append((f, sd, hf, w))
        same = len(set(ends)) == 1 and ends[0] in (0, 2)
        L.append(f"| {f} | " + " | ".join(cells) + f" | {'YES (' + ('LOW' if ends[0] == 0 else 'HIGH') + ')' if same else 'no'} |")
    # ---------- the rules read off MOVR (2026-10-01), applied unchanged: one position at a time, 1 min pause after an exit
    hi = lambda n: f1.h.rolling(n).max(); lo = lambda n: f1.l.rolling(n).min()
    X = pd.DataFrame({"pb15": (f1.c / hi(15) - 1) * 100, "pb240": (f1.c / hi(240) - 1) * 100, "bn15": (f1.c / lo(15) - 1) * 100, "bn240": (f1.c / lo(240) - 1) * 100})
    X.index = X.index + 60_000; T = T.join(X, on="t")
    T["pull15"] = np.where(T.side == "LONG", -T.pb15, T.bn15); T["pull240"] = np.where(T.side == "LONG", -T.pb240, T.bn240)

    def seq(mask, side, tp, sl):
        g = T[mask & (T.side == side)].sort_values("t"); sg = 1 if side == "LONG" else -1; out = []; free = 0
        for t, hf in zip(g.t.values, g.half.values):
            if t < free:
                continue
            i = pos[int(t)] - 1; e = c[i]; H, Lo = h[i + 1:i + 1 + HOLD], l[i + 1:i + 1 + HOLD]
            if sg > 0:
                hs = np.nonzero(Lo <= e * (1 - sl / 100))[0]; ht = np.nonzero(H >= e * (1 + tp / 100))[0]
            else:
                hs = np.nonzero(H >= e * (1 + sl / 100))[0]; ht = np.nonzero(Lo <= e * (1 - tp / 100))[0]
            a_ = hs[0] if len(hs) else 10**9; b_ = ht[0] if len(ht) else 10**9
            r, secs = ((sg * (c[min(i + HOLD, len(c) - 1)] / e - 1) * 100, HOLD) if a_ == b_ == 10**9 else ((-sl, a_ + 1) if a_ <= b_ else (tp, b_ + 1)))
            out.append((hf, r - COST)); free = t + secs * 1000 + 60_000
        return pd.DataFrame(out, columns=["half", "net"])
    RULES = [("LONG after a ≥ 5 % drop from the 4-hour high", T.pull240 >= 5, "LONG", 3.09, 1.51), ("LONG after a ≥ 5 % drop from the 15-min high", T.pull15 >= 5, "LONG", 1.09, 1.11),
             ("LONG always in", T.px > 0, "LONG", 3.09, 1.51), ("LONG always in", T.px > 0, "LONG", 0.59, 1.11),
             ("SHORT after a ≥ 5 % bounce from the 4-hour low", T.pull240 >= 5, "SHORT", 3.09, 1.51), ("SHORT always in", T.px > 0, "SHORT", 3.09, 1.51), ("SHORT always in", T.px > 0, "SHORT", 0.59, 1.11)]
    L += ["", "## Rules read off MOVR, applied unchanged — one position at a time (trades · won · total % of position, after costs)", "",
          "| Rule | Exit | First 24 h | Last 24 h | 48 h total |", "|---|---|---|---|---|"]
    for nm, m, sd, tp, sl in RULES:
        q = seq(m, sd, tp, sl); cells = []
        for hf in ("first 24h", "last 24h"):
            g = q[q.half == hf]; cells.append(f"{len(g)} · {(g.net > 0).mean() * 100:.0f}% · {g.net.sum():+.1f}%" if len(g) else "0")
        L.append(f"| {nm} | +{tp} / −{sl} | {cells[0]} | {cells[1]} | {len(q)} · {q.net.sum():+.1f}% |")
    open(os.path.join(ROOT, "reports", f"PAIR48_{pair}_{end:%Y-%m-%d}.md"), "w").write("\n".join(L) + "\n"); print("\n".join(L))
