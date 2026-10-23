#!/usr/bin/env python3
"""👁 Scout — LITE_GREEN_RETRY observe line (DECISION_LOG 315, operator 2026-10-19: "add LONG_STOP4 and LITE_GREEN_RETRY watch lines").
OBSERVE ONLY (low prior) — never changes config, never calls the bot API.
PARITY NOTE (2026-10-19, DECISION_LOG 316): live FRENZY_LITE_RETRY fills now exist (operator armed the 15-min rule at 4×) — at the read,
compare this shadow cohort vs the live FRENZY_LITE_RETRY fills (same refusals → same entry bar?). The frozen gate below is unchanged.

Origin: reports/FRENZY_REALFILLS_EXIT_RUNNER_2026-10-19.md §F2 / §G. FRENZY_LITE judges an above-VWAP stretch ONCE: a green signal candle
(FRENZY_LITE_GREEN_BAR) consumes the stretch. Replica (real era, T100): 92 of 184 stretches were consumed by a green bar; letting the next
red bar of the same stretch trade added 48 fills at +0.23 % (+$295, CI −$761…+$1,287, P 0.72). DECISION_LOG 243's 1,992-fill study refuted
"retry later bars" (+0.163 → −0.005 %/trade). Example: AGT 10-16 02:00 refused green → next red 02:15 → +2 by 02:22, +3 by 02:46 (1m).

FROZEN DEFINITION (pre-registered 2026-10-19 — never re-fit)
  cohort   every FRENZY_LITE stretch whose judged bar was refused for a green candle: decisions-journal exports (~/Downloads/
           scalpars_decisions_{paper,live}_*.csv via export_glob — the journal is mode-agnostic), e = BLOCK ∧ gate = FRENZY_LITE_GREEN_BAR,
           refusal time (the 5-min bucket = the judged bar's close) ≥ COHORT_START, dedup (bucket, pair). DAY units = UTC day of the shadow entry.
  episode  rebuilt with the ENGINE's own code (services.frenzy.frenzy_walk on the 1,499 closed 5m bars ending at each bar + normal_hour_usd on
           1h klines = the study's walk, study_frenzy_realfills_walk.py): spike anchor, spike-anchored VWAP, stretch id, vol ×, hours. The
           refused bar must rebuild to FRENZY_LITE_GREEN_BAR (bucket bar, else the one before = a late pass) — else REPLICA_MISMATCH (listed,
           never a shadow fill).
  shadow   the NEXT bar of the SAME stretch (same spike, same stretch id = every close since stays ≥ the spike VWAP) that
           frenzy_lite_status rates FRENZY_LITE_READY with no judged stretch (red / flat candle ∧ vol < 100× ∧ 2 h ≤ age ≤ 16.75 h ∧ ≥ 12 closes
           above the VWAP ∧ 24 h volume ≥ $20M); later green / vol-high / in-state / 24 h-volume bars are skipped (the replica's
           green_consumes=False). No shadow fill (counted by reason) when first: BROKEN (a close < VWAP → new stretch id) · EPISODE_ENDED ·
           WINDOW_OVER (> 16.75 h) — or the READY bar is refused by an open check below (the study replica: each consumes the stretch).
  open     reconstructible checks, in the replica's order: WILLY global hold (a FRENZY_WILLY open at the entry → not judged, next bar retries,
           ≤ 6 bars then HOLD_EXPIRED) · BEARISH_DAY (BTC last closed daily return < 0 ∧ BTC 5m EMA13 < EMA50, services.frenzy rule) ·
           PAIR_HELD (any export position on the pair open at the entry) · SLOTS (2 FRENZY_LITE open) · PAIR_DAY_CAP (3 FRENZY-family entries
           on the pair that UTC day) · DISLOCATION (first print after close + 8 s × 1.001 more than 1 % (+0.1 slack) from the close).
  priced   entry = the bar's close + 8 s, first print (ticks; else the next 1m open = provisional) × (1 + 0.10 % live slip) — the study's
           price_one; LITE C: TP +3 if vol × ≥ 20 ∧ BTC 1h EMA20 slope > 0 (last closed hour), else +2 · stop −3 · 12 h · resting maker TP at
           the line · the study walker on NET targets like live (scout_long_stop4.walk_net, DECISION_LOG 355; was .walk = gross lines − fees). $ at LITE's $2,450 notional (8×).
  NOT reconstructed (stated): the ≤ 25-pair FRENZY shortlist, the market-volume gate, the first-bar guarantee / restarts / pauses, the
           engine's 6–8 h normal-hour cache, the ticker 24 h volume (rebuilt as 288 × volume × typical price, the walk's convention),
           positions absent from the exports.
GATE (frozen, pre-registered 2026-10-19 — the report's §G)
  READ at ≥ 20 shadow fills on ≥ 10 UTC days. Below → "collecting (n/20 on d/10 days)".
  CANDIDATE (operator decides; nothing armed) only if: shadow mean > 0 with day-bootstrap P(mean > 0) ≥ 0.90 (4,000, seed 7)
    ∧ shadow mean ≥ the same-period live FRENZY_LITE mean (CLOSED fills opened ≥ COHORT_START, booked pnl_percentage, orders exports)
    ∧ no UTC day and no pair carries ≥ 50 % of the shadow gain ∧ shadow WR ≥ 58 % (BE_WR, LITE C breakeven — added pre-commit
    from reports/LITE_GREEN_RETRY_DEEP_2026-10-19.md).
  OPEN FILTERS on the shadow entry (frozen): LITE status, bearish-day block, pair-day cap, 1 % dislocation AND pair held / LITE slots /
    WILLY hold are REFUSALS (the live engine would refuse them — a shadow fill must be a trade the bot could have taken).
  RETIRE if the shadow mean ≤ 0 at the read. Otherwise "hold — keep observing".
BINANCE  per refusal: fapi 5m (1,500 back + ≤ 205 forward bars, weight 10 + 2), 1h (800 bars, weight 5); per READY bar: BTC 5m / 1h / 1d
  (weight ≤ 5) + the shadow path (aggTrades archive first — CDN, no weight — else one 1m request). Cache reports/backtest_cache/
  scout_lite_green_retry/. ≤ MAX_REFUSALS uncached refusals per run, paced, stop at X-MBX-USED-WEIGHT-1M ≥ 700, shared cooldown honoured /
  armed on 418-429, a 403 stops the run. A stretch whose bars / 12 h path are not complete yet stays PENDING.
Usage  venv/bin/python scripts/scout_lite_green_retry.py                (batch review)
       venv/bin/python scripts/scout_lite_green_retry.py --since ISO    (DIAGNOSTIC only — an earlier cohort start, never the frozen read)
"""
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_ROOT = os.path.dirname(_D)
_ROOT in sys.path or sys.path.insert(0, _ROOT)
import export_glob as EG                                     # noqa: E402
import scout_long_stop4 as LS                                # noqa: E402  (the study walker, tick-first paths, Binance budget, bootstrap)
import scout_gatekit as _GK                                  # noqa: E402  🧰 (DECISION_LOG 331) operator-closed fills out of the live comparison
from services.frenzy import (frenzy_walk, normal_hour_usd, frenzy_flagged, frenzy_lite_status, frenzy_lite_stretch_id,  # noqa: E402
                             frenzy_bearish_block, FRENZY_LITE_READY)

COHORT_START = "2026-10-19T17:00:00"
GATE = "FRENZY_LITE_GREEN_BAR"
GATE_N, GATE_DAYS, P_MIN, SHARE_MAX = 20, 10, 0.90, 0.50
# frozen engine thresholds (trading_config.json 2026-10-19) — the shadow never follows a later config change
TH = SimpleNamespace(frenzy_spike_ret_pct=5.0, frenzy_spike_vol_mult=20.0, frenzy_spike_min_hour_usd=2e6, frenzy_state_vol_mult=100.0,
                     frenzy_min_hours=2.0, frenzy_max_hours=96.0, frenzy_min_volume_usd=20e6, frenzy_long_skip_green_bar=True,
                     frenzy_lite_enabled=True, frenzy_lite_max_hours=16.75, frenzy_lite_min_above_closes=12, frenzy_bearish_day_block=True)
LITE_C_VOL, TP_HI, TP_LO, SL, CAP = 20.0, 3.0, 2.0, 3.0, 720
WIN, BACK, FWD, H1N = 1499, 1500, 205, 800
DELAY_MS, ENTRY_SLIP, DISLOC = 8_000, 1.001, 0.011             # the study's price_one (close + 8 s, +0.10 %, 1 % + 0.1 slack)
HOLD_MAX_BARS, LITE_SLOTS, PAIR_DAY_CAP = 6, 2, 3
NOTIONAL = 2450.0
FRENZY_FAMILY = ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY")
M5, MIN, HOUR, DAY = 300_000, 60_000, 3_600_000, 86_400_000
MAX_REFUSALS = 8
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_lite_green_retry")
NO_FILL = ("BROKEN", "EPISODE_ENDED", "WINDOW_OVER", "HOLD_EXPIRED", "BEARISH_DAY", "PAIR_HELD", "SLOTS", "PAIR_DAY_CAP", "DISLOCATION",
           "NO_PATH")


def _ms(s):
    return int(pd.Timestamp(str(s)[:19]).value // 1_000_000)


# ─────────────────────────── pure pieces (tested offline) ───────────────────────────
def lite_c_tp(vol_mult, btc_1h_slope):
    """LITE C (DL 301), frozen: +3 when vol × ≥ 20 ∧ BTC 1h EMA20 slope > 0, else +2 (missing → +2)."""
    try:
        vm, sl = float(vol_mult), float(btc_1h_slope)
    except (TypeError, ValueError):
        return TP_LO
    return TP_HI if (np.isfinite(vm) and np.isfinite(sl) and vm >= LITE_C_VOL and sl > 0) else TP_LO


def classify(ep, code, sid0, spike0):
    """one bar AFTER the refused bar → READY · SKIP (not a signal bar: green, vol ≥ 100×, in state, 24 h volume, unreadable) ·
    BROKEN · EPISODE_ENDED · WINDOW_OVER."""
    if not ep or not frenzy_flagged(ep, TH) or int(ep.get("spike_ts") or -1) != int(spike0):
        return "EPISODE_ENDED"
    if frenzy_lite_stretch_id(ep) != int(sid0):
        return "BROKEN"                                          # a close < the spike VWAP reset the streak → a new stretch
    if code == "FRENZY_LITE_TOO_LATE":
        return "WINDOW_OVER"
    if code == FRENZY_LITE_READY:
        return "READY"
    return "SKIP"


def next_red(states, sid0, spike0):
    """states = iterable of (ep, code) for the bars after the refused one, in order → (index, 'READY') for the first READY bar of the same
    stretch, (index, reason) when the stretch ends first, (None, 'PENDING') when the bars run out undecided."""
    for i, (ep, code) in enumerate(states):
        c = classify(ep, code, sid0, spike0)
        if c != "SKIP":
            return i, c
    return None, "PENDING"


def vol24_at(rows, j):
    """24 h quote volume of the 288 closed 5m bars ending at rows[j] (volume × typical price — the walk's convention); None if short."""
    if j < 287:
        return None
    a = np.asarray([r[5] * (r[2] + r[3] + r[4]) / 3.0 for r in rows[j - 287:j + 1]], float)
    return float(a.sum())


def ema(x, span):
    return pd.Series(np.asarray(x, float)).ewm(span=span, adjust=False).mean().values


def positions_at(orders, te):
    """orders (deduped exports) → the rows open at te (opened ≤ te < closed; OPEN rows = still open)."""
    if not len(orders):
        return orders
    o = orders["_o"].values; c = orders["_x"].values
    return orders[(o <= te) & (te < c)]


def open_check(orders, pair, te):
    """the replica's open-path refusals that the exports can rebuild → reason or None (WILLY hold handled by the caller)."""
    op = positions_at(orders, te)
    if len(op) and (op.pair.astype(str) == pair).any():
        return "PAIR_HELD"
    if len(op) and (op.entry_strategy.astype(str) == "FRENZY_LITE").sum() >= LITE_SLOTS:
        return "SLOTS"
    if len(orders):
        day0 = te // DAY * DAY
        same = orders[(orders.pair.astype(str) == pair) & orders.entry_strategy.astype(str).isin(FRENZY_FAMILY)
                      & (orders["_o"].values >= day0) & (orders["_o"].values < te)]
        if len(same) >= PAIR_DAY_CAP:
            return "PAIR_DAY_CAP"
    return None


def willy_hold(orders, te):
    op = positions_at(orders, te)
    return bool(len(op) and (op.entry_strategy.astype(str) == "FRENZY_WILLY").any())


def shares(d):
    """largest single UTC day / pair share of the positive shadow total (nan when the total ≤ 0)."""
    tot = float(d.pct.sum())
    if not tot > 0:
        return np.nan, np.nan
    return float(d.groupby("day").pct.sum().max() / tot), float(d.groupby("pair").pct.sum().max() / tot)


BE_WR = 0.58   # LITE C breakeven win rate (reports/LITE_GREEN_RETRY_DEEP_2026-10-19.md: 56 % retry WR vs 58 % breakeven on two years) — frozen


def verdict(d, live_mean):
    """d = shadow FILLS: day, pair, pct (net %) → (verdict string, detail dict)."""
    n, nd = len(d), (d.day.nunique() if len(d) else 0)
    if n < GATE_N or nd < GATE_DAYS:
        return f"collecting ({n}/{GATE_N} on {nd}/{GATE_DAYS} days)", {}
    mu, lo, hi, p = LS.day_boot(d.pct.values, d.day.values)
    sd, sp = shares(d)
    wr = float((d.pct > 0).mean())
    det = dict(mean=mu, lo=lo, hi=hi, p=p, live_mean=live_mean, day_share=sd, pair_share=sp, wr=wr)
    if mu <= 0:
        return "RETIRE the line (frozen gate: shadow mean ≤ 0 at the read)", det
    ok = (p >= P_MIN and np.isfinite(live_mean) and mu >= live_mean and np.isfinite(sd) and sd < SHARE_MAX and sp < SHARE_MAX
          and wr >= BE_WR)   # (315 deep dive) win rate ≥ LITE C breakeven, frozen
    if ok:
        return "CANDIDATE (operator decides; reviewed study + 30–50 % haircut before any change)", det
    return "hold — keep observing (read reached, CANDIDATE bar not met)", det


# ─────────────────────────── data ───────────────────────────
def refusals(start=COHORT_START):
    fr = []
    for f in sorted(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")), key=os.path.getmtime):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "dir", "gate")))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=["t", "pair"])
    a = pd.concat(fr, ignore_index=True)
    a = a[(a.e.astype(str) == "BLOCK") & (a.gate.astype(str) == GATE)].copy()
    a["t"] = a.t.astype(str).str[:19].str.replace(" ", "T")
    a = a[a.t >= start].drop_duplicates(["t", "pair"]).sort_values(["t", "pair"]).reset_index(drop=True)
    return a


def _ms_col(s, fill):
    x = pd.to_datetime(s, errors="coerce")
    return np.array([int(v.value // 1_000_000) if pd.notna(v) else fill for v in x], dtype="int64")


def orders_frame():
    o = LS.load_orders(cols=("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl_percentage", "is_paper",
                             "close_reason"))
    if not len(o):
        return o
    o = o.copy()
    big = np.iinfo("int64").max
    o["_o"] = _ms_col(o._k, -1)
    o["_x"] = _ms_col(o.closed_at.astype(str).str[:19].str.replace(" ", "T"), big)
    # deep review (315): only real positions — SIGNAL_EXPIRED / other non-fills never hold a pair; an OPEN row in an old export is open
    # at most the FRENZY sleeves' 12 h hold (never "open forever", which turned old rows into permanent PAIR_HELD refusals)
    st = o.status.astype(str)
    o = o[st.isin(["OPEN", "CLOSED"])].copy()
    op = o.status.astype(str) == "OPEN"
    o.loc[op, "_x"] = o.loc[op, "_o"] + 12 * 3_600_000
    return o[o._o > 0].reset_index(drop=True)


def episode_data(pair, bucket, bud, now, fetch=True):
    """5m rows ending ≥ the refusal + the forward window, and the 1h rows → (rows5, h1) or None (budget / cooldown)."""
    back = LS.fapi_klines(pair, "5m", bucket - BACK * M5, BACK, bud, CACHE, now, fetch=fetch)
    if back is None:
        return None
    fwd = LS.fapi_klines(pair, "5m", bucket, FWD, bud, CACHE, now, fetch=fetch) if now >= bucket + M5 else []
    if fwd is None:
        return None
    h1 = LS.fapi_klines(pair, "1h", bucket // HOUR * HOUR - H1N * HOUR, H1N, bud, CACHE, now, fetch=fetch)
    if h1 is None:
        return None
    rows = {int(r[0]): r for r in back + fwd}
    return [rows[k] for k in sorted(rows)], h1


def state_at(rows, j, h1):
    """the engine's read of bar rows[j] (just closed): (ep, code, vol24)."""
    if j < 300:
        return None, "FRENZY_LITE_NO_DATA", None
    sub = rows[max(0, j - WIN + 1):j + 1]
    nh = normal_hour_usd(h1, int(rows[j][0]))
    ep = frenzy_walk(sub, nh, TH) if nh else None
    v24 = vol24_at(rows, j)
    if not ep:
        return None, "FRENZY_LITE_NOT_FLAGGED", v24
    return ep, frenzy_lite_status(ep, TH, v24, None)[1], v24


def btc_ctx(tclose, bud, now, fetch=True):
    """(last closed daily return %, 5m EMA13−EMA50 gap %, 1h EMA20 slope) at a bar close; None on budget stop."""
    k5 = LS.fapi_klines("BTCUSDT", "5m", tclose - 300 * M5, 300, bud, CACHE, now, fetch=fetch)
    k1h = LS.fapi_klines("BTCUSDT", "1h", tclose // HOUR * HOUR - 150 * HOUR, 150, bud, CACHE, now, fetch=fetch)
    k1d = LS.fapi_klines("BTCUSDT", "1d", tclose // DAY * DAY - 2 * DAY, 2, bud, CACHE, now, fetch=fetch)
    if k5 is None or k1h is None or k1d is None:
        return None
    gap = np.nan
    c5 = [r[4] for r in k5 if r[0] + M5 <= tclose]
    if len(c5) >= 100:
        e13, e50 = ema(c5, 13)[-1], ema(c5, 50)[-1]
        gap = (e13 - e50) / e50 * 100
    d1 = [r for r in k1d if r[0] == tclose // DAY * DAY - DAY]
    ret = (d1[0][4] / d1[0][1] - 1) * 100 if d1 and d1[0][1] > 0 else np.nan
    ch = [r[4] for r in k1h if r[0] + HOUR <= tclose]
    slope = np.nan
    if len(ch) >= 40:
        e = ema(ch, 20)
        slope = float(e[-1] - e[-2])
    return ret, gap, slope


def shadow_one(r, orders, bud, now, fetch=True):
    """one refusal → dict(status FILL / NONE / PENDING / MISMATCH, reason, …)."""
    pair, bucket = str(r["pair"]), _ms(r["t"])
    base = dict(refused=r["t"], pair=pair)
    data = episode_data(pair, bucket, bud, now, fetch)
    if data is None:
        return dict(base, status="PENDING", reason=bud.why or ("budget" if fetch else "per-run cap — re-run"))
    rows, h1 = data
    idx = {int(x[0]): i for i, x in enumerate(rows)}
    j0 = None
    for bo in (bucket - M5, bucket - 2 * M5):                     # the judged bar closes at the bucket (a late pass: one bar earlier)
        j = idx.get(bo)
        if j is None:
            continue
        ep, code, _ = state_at(rows, j, h1)
        if code == GATE:
            j0, ep0 = j, ep
            break
    if j0 is None:
        return dict(base, status="MISMATCH", reason="the refused bar does not rebuild to GREEN_BAR")
    sid0, spike0 = frenzy_lite_stretch_id(ep0), int(ep0["spike_ts"])
    base.update(spike=pd.Timestamp(spike0, unit="ms").strftime("%m-%d %H:%M"), vwap=float(ep0["vwap"]),
                refused_hours=float(ep0["hours"]))
    held_first, k = None, j0 + 1
    while True:
        def gen(start):
            for jj in range(start, len(rows)):
                if rows[jj][0] + M5 > now:
                    return
                ep, code, _ = state_at(rows, jj, h1)
                yield ep, code
        i, out = next_red(gen(k), sid0, spike0)
        if out == "PENDING":
            return dict(base, status="PENDING", reason="stretch still open (bars not closed / fetched yet)")
        jj = k + i
        if out != "READY":
            return dict(base, status="NONE", reason=out, bars_after=jj - j0)
        ep, code, _ = state_at(rows, jj, h1)
        tclose = int(rows[jj][0]) + M5
        te = tclose + DELAY_MS
        if willy_hold(orders, te):                                # refused WITHOUT judging; the next bar retries (≤ 6 bars)
            held_first = jj if held_first is None else held_first
            if jj - held_first >= HOLD_MAX_BARS:
                return dict(base, status="NONE", reason="HOLD_EXPIRED", bars_after=jj - j0)
            k = jj + 1
            continue
        break
    btc = btc_ctx(tclose, bud, now, fetch)
    if btc is None:
        return dict(base, status="PENDING", reason=bud.why or "budget")
    d1, gap, slope = btc
    if frenzy_bearish_block(d1, gap, TH) == "BEARISH_DAY":
        return dict(base, status="NONE", reason="BEARISH_DAY", bars_after=jj - j0)
    why = open_check(orders, pair, te)
    if why:
        return dict(base, status="NONE", reason=why, bars_after=jj - j0)
    close = float(rows[jj][4])
    P, stt = LS.build_path(pair, te, close, CAP, bud, CACHE, now, fetch=fetch)
    if P is None:
        return dict(base, status="PENDING", reason=bud.why or ("budget" if stt == "budget" else "no path yet"))
    p0 = P["op"][0] if np.isfinite(P["op"][0]) else P["cl"][0]
    f = (1 + p0 / 100) * ENTRY_SLIP
    if abs(f - 1) > DISLOC:
        return dict(base, status="NONE", reason="DISLOCATION", bars_after=jj - j0)
    Q = LS.rebase(P, f)
    tp = lite_c_tp(ep["vol_mult"], slope)
    pct, xwhy, xmin = LS.walk_net(Q, tp=tp, sl=SL, cap=CAP)          # 🔧 (DECISION_LOG 355) net TP / stop like live LITE (no fee double count)
    if xwhy == "UNRES":
        return dict(base, status="PENDING", reason="12 h path not complete")
    return dict(base, status="FILL", reason="READY", bars_after=jj - j0, entry_at=pd.Timestamp(te, unit="ms").strftime("%Y-%m-%dT%H:%M:%S"),
                day=pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"), vol_mult=float(ep["vol_mult"]), hours=float(ep["hours"]),
                btc_slope=slope, tp=tp, pct=pct, why=xwhy, exit_min=xmin, usd=pct / 100 * NOTIONAL, prov=bool(Q["n_bar"] > 0))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    start, diag = COHORT_START, False
    if "--since" in argv:
        start, diag = str(argv[argv.index("--since") + 1])[:19], True
    R = refusals(start)
    tag = "DIAGNOSTIC (cohort start overridden — NOT the frozen read) · " if diag else ""
    print(f"LITE_GREEN_RETRY (DECISION_LOG 315, observe) · {tag}cohort {GATE} refusals (decisions journals) ≥ {start} UTC → {len(R)}"
          f" on {R.pair.nunique() if len(R) else 0} pairs")
    print(f"  shadow: the next red / flat READY bar of the same stretch · close + 8 s × 1.001 · LITE C (+3 if vol ≥ 20× ∧ BTC 1h slope > 0, "
          f"else +2) / −3 / 12 h maker TP · ${NOTIONAL:,.0f} · study walker on NET targets (DL 355) · ticks first, else fapi 1m = provisional")
    O = orders_frame()
    lt = O[(O.entry_strategy.astype(str) == "FRENZY_LITE") & (O.status.astype(str) == "CLOSED") & (O._k >= start)] if len(O) else O
    lt, lt_opx = _GK.split_operator_closed(lt) if len(lt) else (lt, lt)   # 🧰 (331) the comparison reads only exits the BOT took (PAIR_HELD keeps every position)
    live_mean = float(pd.to_numeric(lt.pnl_percentage, errors="coerce").mean()) if len(lt) else np.nan
    print((f"  same-period live FRENZY_LITE (booked): {len(lt)} fills · mean {live_mean:+.3f} %" if len(lt) else
           "  same-period live FRENZY_LITE (booked): 0 fills (no comparison yet)") + f" · {_GK.operator_closed_line(lt_opx)}")
    if not len(R):
        print(f"  verdict: collecting (0/{GATE_N} on 0/{GATE_DAYS} days)")
        return
    bud, out, net = LS.Budget(), [], 0
    now = int(time.time() * 1000)
    for r in R.to_dict("records"):
        n0 = bud.requests + bud.archives
        out.append(shadow_one(r, O, bud, now, fetch=(net < MAX_REFUSALS and not bud.stop)))
        net += int(bud.requests + bud.archives > n0)
    S = pd.DataFrame(out)
    F = S[S.status == "FILL"].copy() if "status" in S else pd.DataFrame()
    cnt = S.status.value_counts().to_dict()
    print(f"  refusals: shadow fills {cnt.get('FILL', 0)} · no fill {cnt.get('NONE', 0)} · pending {cnt.get('PENDING', 0)} · replica "
          f"mismatch {cnt.get('MISMATCH', 0)} · fapi requests {bud.requests} (weight {bud.used}) · archives {bud.archives}"
          f"{(' · stopped: ' + bud.why) if bud.why else ''}")
    nf = S[S.status == "NONE"]
    if len(nf):
        print("  no shadow fill: " + " · ".join(f"{k} {v}" for k, v in nf.reason.value_counts().items()))
    for x in S.itertuples():
        if x.status == "FILL":
            print(f"    ✓ {x.refused} {x.pair:12s} → entry {x.entry_at} (+{int(x.bars_after)} bars, {x.hours:.1f} h, vol {x.vol_mult:.0f}×) TP +{x.tp:g}"
                  f" → {x.pct:+.2f} % {x.why} @ {x.exit_min:.0f} min · ${x.usd:+.2f}{'  ᵖ' if x.prov else ''}")
        else:
            print(f"    · {x.refused} {x.pair:12s} {x.status} {x.reason}")
    if not len(F):
        print(f"  verdict: collecting (0/{GATE_N} on 0/{GATE_DAYS} days)")
        return
    print(f"  shadow fills {len(F)} on {F.day.nunique()} days · mean {F.pct.mean():+.3f} % · WR {(F.pct > 0).mean():.0%} · Σ ${F.usd.sum():+.2f}"
          f" · 1m-provisional {int(F.prov.sum())}")
    v, det = verdict(F, live_mean)
    if det:
        print(f"  gate: mean {det['mean']:+.3f} [{det['lo']:+.2f}, {det['hi']:+.2f}] day-boot P {det['p']:.3f} (≥ {P_MIN}) · vs live LITE "
              f"{det['live_mean']:+.3f} · top day {det['day_share']:.0%} · top pair {det['pair_share']:.0%} (< 50 %) · WR {det['wr']:.0%} (≥ {BE_WR:.0%})")
    print(f"  verdict: {v}")


if __name__ == "__main__":
    main()
