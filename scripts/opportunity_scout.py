#!/usr/bin/env python3
"""🔭 Opportunity scout (operator, 2026-09-30: "monitor BTC to detect things we are not seeing, like this morning's SURGE").

READ-ONLY research job — public Binance market data + the operator's order exports. It never talks to the bot and never places
orders (ccxt without keys; fetch_ohlcv / fetch_tickers only). Run hourly (scheduled task); each run looks back 26 h.

Pre-declared event types (deliberately LOOSER than the live sleeves, to surface near-misses):
  BTC_MOVE      BTC 30-min return |r| ≥ 0.8 % on a closed 5m bar. Flag "SURGE trigger held" = services.surge.surge_trigger's legs
                with the live trading_config values (move, volume multiple, LONG 24 h-high close) — trigger only, not spacing/entry.
  BREADTH_BURST ≥ 40 % of the top-40 alts (with data on that bar) moved ≥ 1 % the same way over 15 min
  ALT_SPIKE     one alt: 15-min return |r| ≥ 4 % with bar quote volume ≥ 5× its prior-24 h median
Clusters: consecutive qualifying bars of the same type + pair + direction within 2 h of the cluster's FIRST bar = ONE event (window
units); the cluster keeps its first bar across runs (seeded from SCOUT_EVENTS.csv), its last bar and bar count.

Follow-through (events ≥ 2 h old): reference = the close of the event's first bar; +30 / +60 / +120 min in the event direction —
top-20 alts' median / best for BTC_MOVE and BREADTH_BURST, the pair itself for ALT_SPIKE.
Bot cross-check: ALL ~/Downloads/scalpars_orders_*.csv merged (dedup opened_at + pair + direction); fills opened in
[first bar close, last bar close + 60 min] in the event direction (LONG for UP, SHORT for DOWN); MANUAL fills listed apart and never
count as bot activity; "unknown" when no export covers that window. "⭐ MISSED?" = bot opened nothing and median follow-through
≥ +0.5 % at 60 min — a candidate to backtest, NOT a trade signal. Read the base-rate table first.

Outputs (main project reports/): SCOUT_EVENTS.csv (one row per event; detection columns first-seen, outcome columns refreshed) ·
SCOUT_REPORT_latest.md (last 24 h) · SCOUT_REPORT_<date>.md.
Usage: venv/bin/python scripts/opportunity_scout.py
"""
import glob
import json
import os
import time
from datetime import datetime, timezone

import ccxt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS = os.path.join(ROOT, "reports")
EVENTS_CSV = os.path.join(REPORTS, "SCOUT_EVENTS.csv")
BAR = 300_000
MIN = 60_000
FETCH = 640                      # 26 h lookback (312) + a full 288-bar window + slack (review: 320 left ~22 h without a full window)
EX = ccxt.binanceusdm({"enableRateLimit": True})

BTC_MOVE_MIN = 0.8
BREADTH_SHARE = 0.40
BREADTH_MOVE = 1.0
SPIKE_MOVE = 4.0
SPIKE_VOL = 5.0
DEDUPE_MS = 2 * 3600_000
MISSED_MIN = 0.5
DETECT_COLS = ["move", "vol_mult", "note"]           # first-seen values survive later runs


def _retry(fn, *a, **kw):
    for attempt in range(3):
        try:
            return fn(*a, **kw)
        except Exception:
            time.sleep(1 + attempt)
    return None


def k5(sym, last_closed):
    rows = _retry(EX.fetch_ohlcv, sym, "5m", limit=FETCH)
    if not rows:
        return None
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t").set_index("t")
    return d[d.index <= last_closed]                  # every series cut at the SAME last closed bar (no forming-bar look-ahead)


def load_cfg():
    try:
        return json.load(open(os.path.join(ROOT, "trading_config.json")))
    except Exception:
        return {}


def universe(cfg, n=40):
    excl = {x.strip() for x in (cfg.get("pair_blacklist") or "").split(",") if x.strip()}
    excl |= {x.strip() for x in (cfg.get("no_trade_pairs") or "").split(",") if x.strip()}
    excl |= {"BTCUSDT", "ETHUSDT", "USDCUSDT"}
    tk = _retry(EX.fetch_tickers) or {}
    vol = sorted(((s.split("/")[0] + "USDT", v.get("quoteVolume") or 0) for s, v in tk.items() if s.endswith("/USDT:USDT")),
                 key=lambda x: -x[1])
    return [p for p, _ in vol if p not in excl][:n]


def load_exports():
    """All operator exports merged (review: the newest alone covers only minutes after a reset). Returns (orders, coverage)."""
    frames, cov = [], []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "direction", "entry_strategy"))
            _ts = pd.to_datetime(d["opened_at"], utc=True, format="ISO8601")
            d["opened_ms"] = (_ts - pd.Timestamp(0, tz="UTC")) // pd.Timedelta(milliseconds=1)   # unit-safe (ISO8601 parses to [s] here)
            if len(d):
                frames.append(d)
                cov.append((int(d.opened_ms.min()), int(os.path.getmtime(f) * 1000)))
        except Exception as e:
            print(f"[scout] export skipped ({os.path.basename(f)}): {e}")
    if not frames:
        return None, []
    o = pd.concat(frames, ignore_index=True).drop_duplicates(["opened_at", "pair", "direction"])
    return o, cov


def detect(btc, alts, now_ms, cfg):
    th = (cfg.get("thresholds") or {})
    s_move = abs(float(th.get("surge_btc_move_pct", 1.0) or 1.0))
    s_vol = float(th.get("surge_btc_vol_mult", 3.0) or 0.0)
    s_high = bool(th.get("surge_long_require_24h_high", True))
    raw = []
    since = now_ms - 26 * 3600_000
    r30 = (btc.c / btc.c.shift(6) - 1) * 100
    qv = btc.v * btc.c
    med = qv.shift(1).rolling(288, min_periods=288).median()
    hi24 = btc.h.shift(1).rolling(288, min_periods=288).max()
    for t in btc.index[(btc.index >= since) & (r30.abs() >= BTC_MOVE_MIN)]:
        side = "UP" if r30[t] > 0 else "DOWN"
        vm = qv[t] / med[t] if pd.notna(med[t]) and med[t] > 0 else None
        held = (abs(r30[t]) >= s_move and (s_vol <= 0 or (vm is not None and vm >= s_vol))
                and (side == "DOWN" or not s_high or (pd.notna(hi24[t]) and btc.c[t] >= hi24[t])))
        raw.append(dict(type="BTC_MOVE", pair="BTCUSDT", side=side, bar_ts=int(t), move=round(r30[t], 2),
                        vol_mult=round(vm, 1) if vm else None, held=bool(held)))
    r15 = {p: (d.c / d.c.shift(3) - 1) * 100 for p, d in alts.items() if d is not None and len(d) > 30}
    if r15:
        R = pd.DataFrame(r15)
        n = R.notna().sum(axis=1)                                  # review: only alts with data on that bar count
        up = (R >= BREADTH_MOVE).sum(axis=1) / n.where(n > 0)
        dn = (R <= -BREADTH_MOVE).sum(axis=1) / n.where(n > 0)
        for t in R.index[R.index >= since]:
            for side, sh in (("UP", up[t]), ("DOWN", dn[t])):
                if pd.notna(sh) and sh >= BREADTH_SHARE and n[t] >= 10:
                    raw.append(dict(type="BREADTH_BURST", pair="TOP40", side=side, bar_ts=int(t), move=round(sh * 100, 0),
                                    vol_mult=None, held=False))
    for p, d in alts.items():
        if d is None or len(d) < 300:
            continue
        m15 = (d.c / d.c.shift(3) - 1) * 100
        q = d.v * d.c
        qm = q.shift(1).rolling(288, min_periods=288).median()
        hit = (d.index >= since) & (m15.abs() >= SPIKE_MOVE) & (q >= SPIKE_VOL * qm)
        for t in d.index[hit]:
            raw.append(dict(type="ALT_SPIKE", pair=p, side="UP" if m15[t] > 0 else "DOWN", bar_ts=int(t), move=round(m15[t], 2),
                            vol_mult=round(q[t] / qm[t], 1), held=False))
    return raw


def cluster(raw, known):
    """Start-anchored 2 h clusters per (type, pair, side). `known` = {key: [first-bar ts of events already in SCOUT_EVENTS.csv]} —
    a bar inside a known event's 2 h window joins THAT event (review: the sliding 26 h window used to re-seed a cluster later)."""
    raw.sort(key=lambda e: e["bar_ts"])
    out = {}
    for e in raw:
        key = (e["type"], e["pair"], e["side"])
        anchor = next((k for k in known.get(key, []) if 0 <= e["bar_ts"] - k < DEDUPE_MS), None)
        if anchor is None:
            anchor = next((c for (kk, c) in out if kk == key and 0 <= e["bar_ts"] - c < DEDUPE_MS), None)
        if anchor is None:
            anchor = e["bar_ts"]
        ev = out.setdefault((key, anchor), dict(type=e["type"], pair=e["pair"], side=e["side"], bar_ts=anchor, move=e["move"],
                                                vol_mult=e["vol_mult"], held=False, end_ts=e["bar_ts"], n_bars=0))
        ev["end_ts"] = max(ev["end_ts"], e["bar_ts"]); ev["n_bars"] += 1
        ev["held"] = ev["held"] or e["held"]
        if abs(e["move"]) > abs(ev["move"]) and e["type"] != "BREADTH_BURST":
            ev["move"] = e["move"]                                     # the cluster's strongest bar
    evs = list(out.values())
    for ev in evs:
        ev["note"] = ("SURGE trigger held" if ev["held"] else "below SURGE trigger") if ev["type"] == "BTC_MOVE" else \
                     (f"{ev['move']:.0f}% of top-40 moved ≥{BREADTH_MOVE}%" if ev["type"] == "BREADTH_BURST" else "")
        if ev["n_bars"] > 1:
            ev["note"] = (ev["note"] + f" · {ev['n_bars']} bars to {datetime.fromtimestamp((ev['end_ts'] + BAR) / 1000, timezone.utc):%H:%M}").strip(" ·")
        del ev["held"]
    return evs


def follow_through(e, alts, top20, now_ms):
    if now_ms - (e["bar_ts"] + BAR) < 2 * 3600_000:
        return {}
    sgn = 1 if e["side"] == "UP" else -1
    series = [alts.get(e["pair"])] if e["type"] == "ALT_SPIKE" else [alts.get(p) for p in top20]
    res = {}
    for h in (30, 60, 120):
        mv = []
        for d in series:
            if d is None or e["bar_ts"] not in d.index:
                continue
            i = d.index.get_loc(e["bar_ts"])        # review: the EVENT bar's own close is the reference (was one bar late)
            j = i + h // 5                           # close of bar j = event close + h
            if j < len(d):
                mv.append(sgn * (d.c.iloc[j] / d.c.iloc[i] - 1) * 100)
        if mv:
            res[f"f{h}_med"] = round(float(np.median(mv)), 2)
            res[f"f{h}_best"] = round(float(np.max(mv)), 2)
    return res


def bot_activity(e, orders, cov):
    t0 = e["bar_ts"] + BAR
    t1 = e["end_ts"] + BAR + 60 * MIN
    if orders is None or not any(start <= t0 and t1 <= end for start, end in cov):
        return "unknown (no export covers this window)", ""
    g = orders[(orders.opened_ms >= t0) & (orders.opened_ms <= t1)]
    if e["type"] == "ALT_SPIKE":
        g = g[g.pair == e["pair"]]
    g = g[g.direction.astype(str).str.upper() == ("LONG" if e["side"] == "UP" else "SHORT")]
    man = g[g.entry_strategy.astype(str) == "MANUAL"]
    bot = g[g.entry_strategy.astype(str) != "MANUAL"]
    b = ", ".join(f"{s}×{n}" for s, n in bot.entry_strategy.fillna("MOMENTUM").astype(str).value_counts().items()) or "none"
    return b, (f"manual×{len(man)}" if len(man) else "")


def fmt(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.2f}%"


def main():
    now_ms = int(time.time() * 1000)
    last_closed = (now_ms // BAR - 1) * BAR
    cfg = load_cfg()
    btc = k5("BTC/USDT:USDT", last_closed)
    if btc is None or len(btc) < 400:
        print("BTC fetch failed or too short — no run"); return
    uni = universe(cfg, 40)
    alts = {}
    for p in uni:
        alts[p] = k5(p[:-4] + "/USDT:USDT", last_closed)
        time.sleep(0.05)
    top20 = uni[:20]
    orders, cov = load_exports()
    old = pd.read_csv(EVENTS_CSV) if os.path.exists(EVENTS_CSV) else pd.DataFrame()
    known = {}
    for r in old.itertuples() if len(old) else []:
        known.setdefault((r.type, r.pair, r.side), []).append(int(r.bar_ts))
    evs = cluster(detect(btc, alts, now_ms, cfg), known)
    for e in evs:
        e.update(follow_through(e, alts, top20, now_ms))
        e["bot"], e["manual"] = bot_activity(e, orders, cov)
        f60 = e.get("f60_med")
        e["missed"] = bool(f60 is not None and f60 >= MISSED_MIN and e["bot"] == "none")
        e["time_utc"] = datetime.fromtimestamp((e["bar_ts"] + BAR) / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
    new = pd.DataFrame(evs)
    if len(new):
        key = ["type", "pair", "side", "bar_ts"]
        if len(old):
            n2, o2 = new.set_index(key), old.set_index(key)
            merged = n2.combine_first(o2)                               # outcome columns: new wins unless it is missing
            both = n2.index.intersection(o2.index)
            for c in DETECT_COLS:                                       # detection columns: first-seen wins
                if c in o2.columns:
                    merged.loc[both, c] = o2.loc[both, c]
            allv = merged.reset_index()
        else:
            allv = new
        os.makedirs(REPORTS, exist_ok=True)
        allv.sort_values("bar_ts").to_csv(EVENTS_CSV, index=False)
    rep = new[new.bar_ts >= now_ms - 24 * 3600_000] if len(new) else new
    lines = [f"# 🔭 Opportunity scout — {datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d %H:%M} UTC (last 24 h)", "",
             "Read-only. Candidates to backtest, not trade signals. Time = the event's first bar close (UTC). f60 med = median move of "
             "the top-20 alts (or the spiking pair) in the event direction 60 min after that close; '—' = too recent. "
             "Read the base-rate table before any ⭐.", ""]
    covtxt = (f"order exports cover {datetime.fromtimestamp(min(s for s, _ in cov)/1000, timezone.utc):%m-%d %H:%M} → "
              f"{datetime.fromtimestamp(max(e for _, e in cov)/1000, timezone.utc):%m-%d %H:%M} UTC (gaps between exports = unknown)"
              if cov else "no order export found in ~/Downloads")
    lines += [f"Universe: top-40 tradeable alts by 24 h volume · {covtxt}", ""]
    if not len(rep):
        lines.append("No events in the last 24 h.")
    else:
        ready = rep[rep["f60_med"].notna()] if "f60_med" in rep else rep.iloc[0:0]
        if len(ready):
            lines += ["**Base rate per event type (all events with a 60-min read — not only the ⭐ ones):**", "",
                      "| Type | Events | Continued at 60 min | Avg f60 (event direction) | Avg f120 |", "|---|---|---|---|---|"]
            for ty, g in ready.groupby("type"):
                f120 = g["f120_med"].mean() if "f120_med" in g and g["f120_med"].notna().any() else None
                lines.append(f"| {ty} | {len(g)} | {(g.f60_med > 0).mean()*100:.0f}% | {g.f60_med.mean():+.2f}% | {fmt(f120)} |")
            lines.append("")
        lines += ["| Time UTC | Type | Pair | Dir | Move | Vol× | Note | f30 med | f60 med / best | f120 med | Bot | Manual | ⭐ |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, e in rep.sort_values("bar_ts").iterrows():
            mv = f"{e['move']:+.2f}%" if e["type"] != "BREADTH_BURST" else f"{e['move']:.0f}% of alts"
            vm = "" if pd.isna(e.get("vol_mult")) else e["vol_mult"]
            lines.append(f"| {e['time_utc'][5:]} | {e['type']} | {e['pair']} | {e['side']} | {mv} | {vm} | {e.get('note') or ''} | "
                         f"{fmt(e.get('f30_med'))} | {fmt(e.get('f60_med'))} / {fmt(e.get('f60_best'))} | {fmt(e.get('f120_med'))} | "
                         f"{e['bot']} | {e.get('manual') or ''} | {'⭐ MISSED?' if e['missed'] else ''} |")
        m = rep[rep.missed]
        lines += ["", f"**Summary:** {len(rep)} events · {int(rep.missed.sum())} flagged ⭐ MISSED? (bot did nothing in the event "
                      f"direction, ≥ +{MISSED_MIN}% median follow-through at 60 min)."]
        if len(m):
            lines.append("Flagged: " + "; ".join(f"{r.time_utc[11:]} {r.type} {r.pair} {r.side} (f60 {r.f60_med:+.2f}%)" for r in m.itertuples()))
    txt = "\n".join(lines) + "\n"
    os.makedirs(REPORTS, exist_ok=True)
    open(os.path.join(REPORTS, "SCOUT_REPORT_latest.md"), "w", encoding="utf-8").write(txt)
    open(os.path.join(REPORTS, f"SCOUT_REPORT_{datetime.fromtimestamp(now_ms/1000, timezone.utc):%Y-%m-%d}.md"), "w", encoding="utf-8").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
