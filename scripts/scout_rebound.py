#!/usr/bin/env python3
"""🩹 Scout — BTC REBOUND-window OBSERVATION (pre-registered 2026-10-05, operator; DECISION_LOG 204). OBSERVE only — never a trade.

Question: June's momentum longs won while BTC was 15–24 % below its 30-day high (real trades at ≤ −15 %: 20 · 100 % · +0.61 %), but the
full-year replay says "washed-out" alone loses (yr5: ≤ −15 % −0.107 %/trade, CI < 0 — the Jan–Apr declines). The candidate separator is
the TURN: washed-out AND rising. Called by scripts/opportunity_scout.py every run (never breaks it). Public Binance data + the bot's exports.

STATE (frozen): on the last CLOSED BTC 1h bar — ① close ≤ −15 % below the highest 1h high of the prior 30 days (720 bars) ② BTC 3-day
  return (close vs 72 h earlier) > 0 → REBOUND ON. A window = consecutive ON hours (gaps ≤ 6 h merge). Both legs are market-wide → the unit
  is the WINDOW (window-units rule), never the trade.
TALLY   every paper-bot fill opened inside a REBOUND window (orders exports in ~/Downloads, MANUAL / *_PROBE excluded), per sleeve; momentum
  longs are the read. Also tallied: the "still falling" twin (① ∧ 3-day ≤ 0) for contrast.
YEAR    yr5 replay (today's config, 3 seeds): REBOUND 53 ML fills/seed · 61 % · −0.022 %/trade · CI −0.15…+0.16 · 27 days (Feb −0.09,
  Mar +1.38 (n 2), Jun −0.07, Jul +0.72) — no edge proven; still-falling 116/seed · 55 % · −0.145 · CI −0.26…−0.03.
REVIEW  at ≥ 3 REBOUND windows with ≥ 10 momentum-long fills (a fill counts when the PRIOR closed hour was ON): candidate only if window means are
  > 0 in ≥ 2/3 of windows ∧ mean of window means > 0 ∧ no window ≥ 50 % of the positive gain. Never re-fit the −15 % / 3-day legs.
"""
import glob
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 3_600_000
CSV = os.path.join(ROOT, "reports", "SCOUT_REBOUND.csv")   # one row per ON hour (kind, hour_ms) — windows are derived; fills count only in ON hours
OFF_MAX, R72_MIN, MERGE_H, REVIEW_W, REVIEW_N = -15.0, 0.0, 6, 3, 10
YEAR_REF = ("yr5: REBOUND 53 ML/seed · 61 % · −0.022 %/trade (CI −0.15…+0.16, 27 days) · still falling 116/seed · 55 % · −0.145 "
            "(CI −0.26…−0.03)")


def btc_1h(EX, retry, last_closed_ms):
    """~1,700 closed BTC 1h bars ending before last_closed_ms (4 calls): 720 for the 30-day high + 72 for the 3-day return + ~37 days
    judged (review: 900 bars judged only ~8 days)."""
    out = {}
    since = last_closed_ms - (720 + 72 + 24 * 37) * H
    while since < last_closed_ms:
        r = retry(EX.fapiPublicGetKlines, {"symbol": "BTCUSDT", "interval": "1h", "startTime": int(since), "limit": 500}) or []
        if not r:
            break
        for x in r:
            t = int(x[0])
            if t + H <= last_closed_ms + 300_000:          # closed bars only
                out[t] = (float(x[2]), float(x[4]))
        nxt = int(r[-1][0]) + H
        if nxt <= since:
            break
        since = nxt
        time.sleep(0.05)
    d = pd.DataFrame([(t, h, c) for t, (h, c) in sorted(out.items())], columns=["t", "h", "c"]).set_index("t")
    d["hi30"] = d.h.shift(1).rolling(720, min_periods=720).max()   # exactly the prior 30 days (frozen definition)
    d["off30"] = (d.c / d.hi30 - 1) * 100
    d["r72"] = (d.c / d.c.shift(72) - 1) * 100
    return d


def windows(mask_index):
    """contiguous ON hours (gaps ≤ MERGE_H) → [(start_ms, end_ms)] (end = the last ON bar's close)."""
    ts = sorted(int(t) for t in mask_index)
    out = []
    for t in ts:
        if out and t - out[-1][1] <= MERGE_H * H:
            out[-1][1] = t + H
        else:
            out.append([t, t + H])
    return [tuple(x) for x in out]


def _orders():
    fr = []
    for f in glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("opened_at", "pair", "direction", "entry_strategy", "status",
                                                                          "pnl_percentage", "pnl", "cell_multiplier_source"))
            if len(d) and {"opened_at", "pnl_percentage"} <= set(d.columns):
                fr.append(d.assign(_m=os.path.getmtime(f)))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    o = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable")
    o["_k"] = o.opened_at.astype(str).str[:19]                       # exports mix 19- and 26-char stamps (review)
    o = o.drop_duplicates(["_k", "pair", "direction"], keep="last")
    o = o[o.status.astype(str).str.upper() == "CLOSED"]
    o = o[(o.entry_strategy.astype(str) != "MANUAL") & ~o.get("cell_multiplier_source", pd.Series("", index=o.index)).astype(str).str.endswith("_PROBE")]
    _t = pd.to_datetime(o._k, format="ISO8601", errors="coerce")   # one malformed row must not kill the section (review)
    o = o[_t.notna()].copy(); o["ms"] = (_t[_t.notna()] - pd.Timestamp(0)).dt.total_seconds() * 1000
    o["hour"] = (o.ms // H * H).astype("int64")
    o["sleeve"] = o.entry_strategy.fillna("MOMENTUM").astype(str).str.split(":").str[0] + " " + o.direction.astype(str)
    o["pct"] = pd.to_numeric(o.pnl_percentage, errors="coerce")
    return o


def run(EX, retry, last_closed_ms):
    """→ (lines, windows table). Never raises past the caller's try."""
    d = btc_1h(EX, retry, last_closed_ms)
    d = d[d.off30.notna() & d.r72.notna()]
    on = d[(d.off30 <= OFF_MAX) & (d.r72 > R72_MIN)]
    fall = d[(d.off30 <= OFF_MAX) & (d.r72 <= R72_MIN)]
    now = d.iloc[-1]
    state = "🟢 REBOUND ON" if (now.off30 <= OFF_MAX and now.r72 > R72_MIN) else ("🔻 still falling (washed-out, 3-day ≤ 0)" if now.off30 <= OFF_MAX else "off")
    # persist the ON HOURS (union across runs) — review: fills are tallied only inside ON hours, never in the merged gaps
    old = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame(columns=["kind", "hour_ms"])
    if len(old) and "hour_ms" not in old:   # an older window-format file → rebuild from this run
        old = pd.DataFrame(columns=["kind", "hour_ms"])
    new = pd.DataFrame([("REBOUND", int(t)) for t in on.index] + [("FALLING", int(t)) for t in fall.index], columns=["kind", "hour_ms"])
    HR = pd.concat([old, new], ignore_index=True).drop_duplicates(["kind", "hour_ms"]).sort_values(["kind", "hour_ms"])
    if len(HR):
        tmp = CSV + ".tmp"; HR.to_csv(tmp, index=False); os.replace(tmp, CSV)
    sets = {k: set(g.hour_ms.astype("int64")) for k, g in HR.groupby("kind")}
    W = pd.DataFrame([(k, a, b, sum(1 for t in sets[k] if a <= t < b)) for k in sets for a, b in windows(sets[k])],
                     columns=["kind", "start_ms", "end_ms", "on_hours"])
    o = _orders()
    L = ["## 🩹 BTC rebound-window observation (pre-registered, OBSERVE only — never a trade)", "",
         f"State (frozen): BTC ≤ {OFF_MAX:g} % below its 30-day high (1h) ∧ BTC 3-day return > 0 → REBOUND. Unit = the window. "
         f"Year reference: {YEAR_REF}. Review at ≥ {REVIEW_W} REBOUND windows with ≥ {REVIEW_N} momentum-long fills: window means > 0 in ≥ 2/3 of "
         "windows ∧ mean of window means > 0 ∧ no window ≥ 50 % of the positive gain. A fill counts when the PRIOR closed hour was ON.", "",
         f"**Now:** {state} · BTC {now.off30:+.1f} % vs its 30-day high · 3-day {now.r72:+.2f} %", ""]
    if len(W):
        L += ["| Window (UTC) | kind | ON hours | momentum longs: N · WR · avg % | all bot fills: N · avg % |", "|---|---|---|---|---|"]
        for r in W.sort_values("start_ms", ascending=False).head(12).itertuples():
            g = o[(o.hour - H).isin(sets[r.kind]) & (o.ms >= r.start_ms + H) & (o.ms < r.end_ms + H)] if len(o) else o   # the PRIOR closed hour was ON (no look-ahead)
            ml = g[g.sleeve == "MOMENTUM LONG"] if len(g) else g
            L.append(f"| {pd.to_datetime(r.start_ms, unit='ms'):%m-%d %H:%M} → {pd.to_datetime(r.end_ms, unit='ms'):%m-%d %H:%M} | {r.kind} | "
                     f"{r.on_hours} | " + (f"{len(ml)} · {(ml.pct > 0).mean() * 100:.0f}% · {ml.pct.mean():+.3f}" if len(ml) else "0")
                     + " | " + (f"{len(g)} · {g.pct.mean():+.3f}" if len(g) else "0") + " |")
        rb = W[W.kind == "REBOUND"]
        if len(o) and len(rb):
            per = []
            for r in rb.itertuples():
                ml = o[(o.hour - H).isin(sets["REBOUND"]) & (o.ms >= r.start_ms + H) & (o.ms < r.end_ms + H) & (o.sleeve == "MOMENTUM LONG")]
                if len(ml):
                    per.append((len(ml), ml.pct.mean(), ml.pct.sum()))
            n = sum(x[0] for x in per)
            if per:
                pos = [x[2] for x in per if x[2] > 0]
                L += ["", f"**Tally: {len(per)} REBOUND windows with momentum-long fills (of {REVIEW_W}) · {n} fills (of {REVIEW_N})** · mean of window "
                          f"means {np.mean([x[1] for x in per]):+.3f} % · pooled {sum(x[2] for x in per) / n:+.3f} %/trade · windows > 0: "
                          f"{sum(1 for x in per if x[1] > 0)}/{len(per)} (bar ≥ 2/3)"
                          + (f" · top window {max(pos) / sum(pos) * 100:.0f} % of the positive gain" if pos else "")
                          + (" · 📋 REVIEW DUE" if len(per) >= REVIEW_W and n >= REVIEW_N else "")]
    else:
        L.append("No washed-out hour (≤ −15 % below the 30-day high) recorded yet (each run judges the last ~37 days).")
    return L + [""]
