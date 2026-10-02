#!/usr/bin/env python3
"""🪜 Staircase watch for the opportunity scout (operator, 2026-10-02) — ALERT ONLY, never a trade signal.

Lists the futures pairs that right now look like MOVR (Sep-30 / Oct-1) and SAND (Oct-2): after a volume spike they keep closing
above the volume-weighted average price anchored at the spike while still trading a multiple of their normal volume. Same rule
as scripts/staircase_swing_test.py, whose year test is NOT established (1 trade in 5 wins; the total is carried by a few giants).

  leader   a 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour ∧ ≥ $2M
  episode  starts at a leader bar; anchored VWAP = Σ(typical price × volume) / Σ volume from that bar; it ends when 24 h pass
           without a state bar, and the next leader bar after that starts a new one (the research walk)
  state    ≥ 2 h (24 bars) after the onset ∧ every 5m close of the last hour ≥ the anchored VWAP ∧ last-hour volume ≥ 50× normal
           (★ = ≥ 100×, the tested bar). ⏳ = still ON ≥ 32 h after the onset.
Public market data only: one 24 h-ticker call to shortlist (up ≥ 15 % with ≥ $20M, plus pairs already being followed), then two
kline calls per pair. An onset too close to the start of the fetched window cannot be verified → the pair is not listed."""
import numpy as np
import pandas as pd

BAR = 300_000
MIN_CHANGE, MIN_QUOTE, SHORTLIST, EXTRA_MAX = 15.0, 20e6, 25, 10
LEAD_RET, LEAD_VOLX, LEAD_Q1H = 5.0, 20.0, 2e6
STATE_BARS, STATE_VOLX, TESTED_VOLX = 24, 50.0, 100.0
LATE_HOURS = 32.0             # post-hoc cut of the year test (not reviewed): entries this late did best; earlier ones lost on average
WINDOW_BARS, MAX_HOURS = 1500, 96.0   # 125 h of 5m bars: an onset ≤ 96 h old always has ≥ 24 h of visible history before it
GAP_MS = 24 * 3600_000


def staircase_state(d, norm_hour):
    """d = 5m frame (index = open time ms; columns h, l, c, q = quote volume), CLOSED bars only, oldest first.
    norm_hour = the pair's normal hourly quote volume. Walks the window like the research rule and returns the LIVE episode
    (the one still running at the last bar) or None: onset time, price before it, anchored VWAP, hours since, last-hour volume
    multiple, in_state, and onset_verified (False when the onset sits in the first 24 h of the window → cannot be trusted)."""
    if d is None or len(d) < 40 or not norm_hour or not (norm_hour > 0):
        return None
    t = d.index.values.astype("int64"); h, l, c, q = d.h.values, d.l.values, d.c.values, d.q.values
    n = len(c); q1h = pd.Series(q).rolling(12).sum().values; volx = q1h / norm_hour
    r30 = np.r_[[np.nan] * 6, (c[6:] / c[:-6] - 1) * 100]
    with np.errstate(invalid="ignore"):
        lead = np.nonzero((r30 >= LEAD_RET) & (volx >= LEAD_VOLX) & (q1h >= LEAD_Q1H))[0]
    i = 0
    for on in lead:
        on = int(on)
        if on < i:
            continue                                                   # inside the previous episode
        pv = (h[on:] + l[on:] + c[on:]) / 3 * q[on:]; vw = np.cumsum(pv) / np.maximum(np.cumsum(q[on:]), 1e-12)
        above = pd.Series(c[on:] >= vw).rolling(12).min().fillna(0).values >= 1          # every close of the last hour ≥ VWAP
        up = (c[on:] >= vw)[::-1]; streak = int(up.argmin()) if not up.all() else len(up)   # closes in a row ≥ VWAP, ending now
        with np.errstate(invalid="ignore"):
            state = above & (volx[on:] >= STATE_VOLX) & (np.arange(n - on) >= STATE_BARS)
        last = on; end = None
        for j in range(on + 1, n):                                     # the episode ends when 24 h pass without a state bar
            if t[j] - t[last] > GAP_MS:
                end = j; break
            if state[j - on]:
                last = j
        if end is not None:
            i = end; continue
        return dict(onset_ts=int(t[on]), base=float(c[on - 6]), price=float(c[-1]), vwap=float(vw[-1]),
                    hours=float((t[-1] - t[on]) / 3600_000.0), volx=float(volx[-1]), above_hour=bool(above[-1]), above_streak=streak,
                    above_share=float((c[on:] >= vw).mean()), in_state=bool(state[-1]),
                    onset_verified=bool(t[on] - t[0] >= GAP_MS + 12 * BAR))
    return None


def scan(ex, last_closed, retry=lambda fn, *a, **kw: fn(*a, **kw), extra=()):
    """Rows for pairs with a live, verifiable staircase episode, plus how many shortlisted pairs could not be read.
    Returns (rows, shortlisted, unreadable) or None when the shortlist itself could not be built. Never raises."""
    try:
        tick = retry(ex.fapiPublicGetTicker24hr)
        if not tick:
            return None
        ok = lambda x: str(x.get("symbol", "")).endswith("USDT") and str(x["symbol"]).isascii() and x["symbol"] not in ("BTCUSDT", "ETHUSDT")
        chg = {x["symbol"]: float(x.get("priceChangePercent") or 0) for x in tick if ok(x)}
        qv = {x["symbol"]: float(x.get("quoteVolume") or 0) for x in tick if ok(x)}
        cand = sorted((s for s in chg if chg[s] >= MIN_CHANGE and qv[s] >= MIN_QUOTE), key=lambda s: -chg[s])[:SHORTLIST]
        cand += [s for s in extra if s in chg and s not in cand][:EXTRA_MAX]      # pairs already followed stay in view while they run
    except Exception:
        return None
    rows, bad, streak = [], 0, 0
    for sym in cand:
        if streak >= 3:                                                # the exchange is not answering — stop, do not stall the run
            bad += 1; continue
        try:
            k1 = retry(ex.fapiPublicGetKlines, {"symbol": sym, "interval": "1h", "limit": 744})
            hq = [float(r[7]) for r in (k1 or []) if int(r[0]) + 3600_000 <= last_closed - GAP_MS]      # full hours, ending a day ago
            if k1 and len(hq) < 240:
                streak = 0; continue                                   # listed < ~10 days: no normal volume to compare with
            k5 = retry(ex.fapiPublicGetKlines, {"symbol": sym, "interval": "5m", "limit": WINDOW_BARS}) if k1 else None
            if not k1 or not k5:
                bad += 1; streak += 1; continue
            streak = 0
            d = pd.DataFrame([[int(r[0]), float(r[2]), float(r[3]), float(r[4]), float(r[7])] for r in k5], columns=["t", "h", "l", "c", "q"]).drop_duplicates("t").set_index("t")
            s = staircase_state(d[d.index <= last_closed], float(np.median(hq[-720:])))
            if s is not None and s["onset_verified"] and s["hours"] <= MAX_HOURS:
                rows.append(dict(pair=sym, chg24=chg[sym], **s))
        except Exception:
            bad += 1
    late = lambda r: r["in_state"] and r["hours"] >= LATE_HOURS
    return sorted(rows, key=lambda r: (not r["in_state"], not late(r), -r["volx"])), len(cand), bad


def lines(result):
    """Markdown section for the scout report. result = scan()'s return value."""
    L = ["## 🪜 Staircase watch — pairs holding above their spike's average price on heavy volume (alert only)", ""]
    if result is None:
        return L + ["Unavailable this run (the exchange ticker could not be read)."]
    rows, n, bad = result
    tail = f" {bad} of {n} shortlisted pairs could not be read this run." if bad else ""
    if not rows:
        return L + [f"No pair is in a live spike episode right now ({n} checked: up ≥ 15 % in 24 h with ≥ $20M traded, listed ≥ ~10 days).{tail}"]
    L += ["State = ≥ 2 h after the spike, every close of the last hour above the volume-weighted average price since the spike, last-hour volume ≥ 50× normal "
          "(★ = ≥ 100×, the level tested). Year test of buying this state and selling on a close below that average (100× level): 1 trade in 5 wins, average win "
          "+25 % / loss −5 %, by-day range spans zero, and without its best 5 % of trades it loses — NOT established. ⏳ = still ON ≥ 32 h after the spike: in an "
          "unreviewed after-the-fact cut those late entries did best (+2.7 %/trade at the 100× level, carried by a handful of trades) and earlier entries lost. "
          f"An alert, never a signal.{tail}", "",
          "| Pair | State | Spike (UTC) | Hours since | Gain since spike | Price vs average | Average price | Last-hour volume | Closes above average | 24 h |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        st = ((("★ ON" if r["volx"] >= TESTED_VOLX else "ON") + (" · ⏳" if r["hours"] >= LATE_HOURS else "")) if r["in_state"]
              else ("above for the hour, volume fading" if r["volx"] < STATE_VOLX else "above for the hour, < 2 h since the spike") if r["above_hour"]
              else f"back above, {min(r.get('above_streak') or 0, 11)} of 12 closes" if r["price"] >= r["vwap"] and (r.get("above_streak") or 0) > 0 else "below average")
        L.append(f"| {r['pair']} | {st} | {pd.Timestamp(r['onset_ts'] + BAR, unit='ms'):%m-%d %H:%M} | {r['hours']:.1f} | {(r['price'] / r['base'] - 1) * 100:+.0f}% | "
                 f"{(r['price'] / r['vwap'] - 1) * 100:+.1f}% | {r['vwap']:.6g} | {r['volx']:.0f}× | {r['above_share'] * 100:.0f}% | {r['chg24']:+.0f}% |")
    return L


def note_items(result, noted, now_ms):
    """(key, line) notes to write: one per pair per episode when it is first seen ON, one more when it is still ON ≥ 32 h.
    A key of the same pair within 96 h of this onset counts as the same episode (the onset can shift a little between runs)."""
    if not result:
        return []
    def seen(prefix, r):
        for k in noted:
            p = k.split("|")
            if len(p) == 3 and p[0] == prefix and p[1] == r["pair"]:
                try:
                    if abs(int(p[2]) - r["onset_ts"]) <= MAX_HOURS * 3600_000:
                        return True
                except ValueError:
                    pass
        return False
    out = []
    for r in result[0]:
        if not r["in_state"]:
            continue
        txt = (f"{r['pair']} +{(r['price'] / r['base'] - 1) * 100:.0f}% since the {pd.Timestamp(r['onset_ts'] + BAR, unit='ms'):%m-%d %H:%M} spike · "
               f"{(r['price'] / r['vwap'] - 1) * 100:+.1f}% vs its average price {r['vwap']:.6g} · volume {r['volx']:.0f}× normal (alert only)")
        if not seen("ST", r):
            out.append((f"ST|{r['pair']}|{r['onset_ts']}", "🪜 staircase ON: " + txt))
        if r["hours"] >= LATE_HOURS and not seen("ST32", r):
            out.append((f"ST32|{r['pair']}|{r['onset_ts']}", f"🪜⏳ staircase still ON after {r['hours']:.0f} h: " + txt))
    return out[:8]


def followed(noted, now_ms):
    """Pairs noted ON within the last 96 h — kept on the shortlist even when their 24 h change has cooled."""
    out = []
    for k, v in (noted or {}).items():
        p = k.split("|")
        if len(p) == 3 and p[0] == "ST" and now_ms - int(v or 0) <= MAX_HOURS * 3600_000 and p[1] not in out:
            out.append(p[1])
    return out
