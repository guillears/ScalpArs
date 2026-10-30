#!/usr/bin/env python3
"""👁 Scout — FB_CONFIRM watch line (pre-registered 2026-10-30, DECISION_LOG 417; OBSERVE ONLY — never changes config, no bot API; public 5m klines only).

QUESTION   Would FLATBASE_SHORT do better entering after a 4-hour close below the signal price, instead of at the signal? Operator 10-30: the trades go negative a lot first. Theory: if the coin cannot even
           close below where we wanted to short it, the drop has not started.
RULE       From each real fill (signal price E = its entry price, time te): at each UTC 4-hour candle close within 24 h after te, first close < E → shadow short at the next 5m open. None → NO TRADE. Same exit as live: stop
           first on a bar touching both, stop gross +10 % → net −10.09 (a gap books the open), TP gross −5 % → net +4.94 (the live booked values), else market at 72 h after te. No funding (paper rows carry none).
EVIDENCE   (in-sample, NEVER counted) B25 16 signals re-priced at +5 / −10: before 16 · 62 % · −0.62 %, real-account daily compound −0.43 → after 12 finished · 92 % · +3.75 %, +1.19, gross targets (at the net constants −0.70 → +3.69) (3 skipped, 1 unresolved); it removes 5 of the 6 stops. Only 3 of the
           16 really ran on +5 / −10. AGAINST: 2026 backtest 535 → 461 · +1.25 → +1.30 %/trade, total +670 → +601 pts, median hold 4.4 → 7.3 h. Found on these same fills → observe only.
COHORT     CLOSED FLATBASE_SHORT fills from the orders exports (export_glob: Downloads + exports_archive/), dedup (opened_at[:19], pair) — CLOSED beats OPEN, newest export wins; operator-closed out (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 417
           commit + 10 min (forward only). A fill is priced once its shadow is decided: skipped, TP, stop or 72 h; undecided → PENDING (holds the freeze). No klines → UNPRICED (holds the freeze).
FROZEN BAR (operator's aim: well above breakeven, same style as DL 416), read per SIGNAL, a skip counts 0 %: at the first prefix (by open time) of N ≥ 15 priced signals on ≥ 8 UTC days, decided ONCE, Δ = shadow − booked: ① mean Δ ≤ 0 → RETIRED; ② ENTRY CANDIDATE (operator decides; never
           automatic) iff mean Δ ≥ +0.50 %/signal ∧ shadow mean > 0 ∧ day-clustered bootstrap P(mean Δ > 0) ≥ 0.90 (4,000 resamples, seed 7) ∧ no single signal ≥ 50 % of the positive Σ Δ; ③ else KEEP OBSERVING. Deferred until the newest export was downloaded ≥ the prefix end + 72 h (a long-held fill
           could be missing). The read shows before → after with the daily compound return. observe_registry pins today's exit (a change → ⚠ STALE). A candidate ships only after the 30–50 % in-sample haircut, real-sizing daily compound, leave-one-out and a pre-committed revert.
STATE      reports/SCOUT_FB_CONFIRM.json (gitignored).
Usage: venv/bin/python -I scripts/scout_flatbase_confirm_watch.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK        # noqa: E402  (operator-closed fills out · the DL commit time)
import scout_b1h_negflank as NF    # noqa: E402  (state I/O)
import scout_long_stop4 as LS      # noqa: E402  (budgeted, cached public fapi klines)

DL = 417
SLEEVE = "FLATBASE_SHORT"
CONFIRM_H, MAX_H = 24, 72
STOP_GROSS, TP_GROSS = 10.0, 5.0
STOP_NET, TP_NET, TIME_FEE = -10.09, 4.94, 0.09
N_MIN, DAYS_MIN = 15, 8
D_MIN, P_MIN, SHARE_MAX = 0.50, 0.90, 0.50
BOOT_N, BOOT_SEED = 4000, 7
WAIT_MIN = MAX_H * 60
M5, H4, HOUR = 300_000, 14_400_000, 3_600_000
PAGE_BARS = 900                    # 75 h of 5m bars
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_FB_CONFIRM.json")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_fb_confirm")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "pnl", "close_reason")
CANDIDATE = "ENTRY CANDIDATE (operator decides; never automatic)"
REGISTRATION = ("in-sample at registration (NEVER counted): B25 16 signals at +5 / −10 · −0.62 % → 12 finished · 92 % · +3.75 %, removes 5 of 6 stops; 2026 backtest +1.25 → +1.30 %/trade, total +670 → +601, holds longer")


def load(paths=None):
    """→ (closed FLATBASE fills — dedup, CLOSED beats OPEN, newest export wins —, the newest read export's download time or None)."""
    fr, read = [], []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
            read.append(f)
    newest = None
    for f in read:
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "pct", "te"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str) == SLEEVE].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c["E"] = pd.to_numeric(c.entry_price, errors="coerce")
    c = c[c.pct.notna() & c.E.notna()]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    c["te"] = c.o.values.astype("datetime64[ms]").astype("int64")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def price_shadow(bars, te, E, now_ms):
    """bars = [[open ms, o, h, l, c], …] 5m CLOSED, ascending, from te's bar. → (why, net %). why: SKIP (0 %) · TP · STOP · TIME · PENDING · UNPRICED. PURE."""
    if not len(bars):
        return "UNPRICED", np.nan
    b = np.asarray(bars, float)
    t, o, h, l, c = b[:, 0].astype("int64"), b[:, 1], b[:, 2], b[:, 3], b[:, 4]
    closes = (t + M5) % H4 == 0
    in_window = t + M5 <= te + CONFIRM_H * HOUR
    after = t + M5 > te
    hit = np.flatnonzero(closes & in_window & after & (c < E))
    last_close = int(t[-1]) + M5
    if not len(hit):
        return ("SKIP", 0.0) if last_close >= te + CONFIRM_H * HOUR else ("PENDING", np.nan)
    i0 = int(hit[0]) + 1
    if i0 >= len(t):
        return "PENDING", np.nan
    Ee = float(o[i0])
    for i in range(i0, len(t)):
        if t[i] >= te + MAX_H * HOUR:
            return "TIME", float(-(o[i] / Ee - 1) * 100 - TIME_FEE)
        line = Ee * (1 + STOP_GROSS / 100)
        if h[i] >= line:
            return "STOP", float(STOP_NET if o[i] < line else -(o[i] / Ee - 1) * 100 - TIME_FEE)
        if l[i] < Ee * (1 - TP_GROSS / 100):     # the paper resting TP needs a trade THROUGH the limit
            return "TP", TP_NET
    return "PENDING", np.nan


def fetch_bars(pair, te, now_ms, bud):
    """5m closed bars from te's bar for up to 75 h. → (bars or None on a budget / error stop)."""
    start = int(te) // M5 * M5
    pg = LS.fapi_klines(pair, "5m", start, PAGE_BARS, bud, CACHE, now_ms, fetch=True)
    if pg is None or getattr(bud, "errors", 0):
        return None
    return [q[:5] for q in pg]


def verdict(rows):
    """rows = priced signals (booked pct, shadow net, day, pair) → (state, detail). PURE."""
    n, nd = len(rows), rows.day.nunique() if len(rows) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"{n}/{N_MIN} priced signals · {nd}/{DAYS_MIN} days"
    d = (rows.shadow - rows.pct).values
    md, ms = float(d.mean()), float(rows.shadow.mean())
    days = rows.day.values
    uniq = pd.unique(days)
    g = [d[days == u] for u in uniq]
    rng = np.random.default_rng(BOOT_SEED)
    s = np.array([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() for _ in range(BOOT_N)])
    p = float((s > 0).mean())
    pos = d[d > 0].sum()
    top = float(d.max() / pos) if pos > 0 else 0.0
    det = (f"{n} signals · {nd} d · booked avg {rows.pct.mean():+.3f} % → shadow {ms:+.3f} % (skip = 0) · mean Δ {md:+.3f} · P(Δ > 0) {p:.2f} · top signal {top:.0%} of + Σ Δ · "
           f"skipped {(rows.why == 'SKIP').sum()}")
    if md <= 0:
        return "RETIRED", det
    if md >= D_MIN and ms > 0 and p >= P_MIN and top < SHARE_MAX:
        return CANDIDATE, det
    return "KEEP OBSERVING", det


def before_after(rows):
    """booked vs shadow: N · WR · avg · the daily compound return with each trade = the account (comparison only). PURE."""
    if not len(rows):
        return "no priced signals"
    span = pd.date_range(rows.o.min().normalize(), rows.o.max().normalize(), freq="D").strftime("%Y-%m-%d")

    def dcr(col):
        return (((1 + (rows.groupby("day")[col].sum() / 100).reindex(span).fillna(0)).prod()) ** (1 / len(span)) - 1) * 100
    k = rows[rows.why != "SKIP"]
    a = f"{len(k)} · {(k.shadow > 0).mean():.0%} · {k.shadow.mean():+.3f} %" if len(k) else "0 trades"
    return f"booked {len(rows)} · {(rows.pct > 0).mean():.0%} · {rows.pct.mean():+.3f} % · {dcr('pct'):+.3f} %/day → confirmed {a} · {dcr('shadow'):+.3f} %/day ({(rows.why == 'SKIP').sum()} skipped)"


def crossing_prefix(rows):
    seen = set()
    for k, d in enumerate(rows.day.values, 1):
        seen.add(d)
        if k >= N_MIN and len(seen) >= DAYS_MIN:
            return rows.iloc[:k]
    return None


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, bars_fn=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 417 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write. bars_fn(pair, te, now_ms) → bars or None; None → public klines."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 FB_CONFIRM — FLATBASE_SHORT entered after a 4-hour close below the signal, vs today's entry (pre-registered 2026-10-30, DECISION_LOG {DL}, OBSERVE only)", ""]
    since_given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="FB_CONFIRM watch line")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    bud = LS.Budget()
    why, sh, stopped = [], [], False
    for r in keep.itertuples():
        bars = bars_fn(r.pair, r.te, now_ms) if bars_fn else (None if stopped else fetch_bars(r.pair, r.te, now_ms, bud))
        if bars is None:
            stopped = True
            why.append("UNPRICED")
            sh.append(np.nan)
            continue
        w, v = price_shadow(bars, r.te, r.E, now_ms)
        why.append(w)
        sh.append(v)
    keep = keep.assign(why=why, shadow=sh) if len(keep) else keep.assign(why=pd.Series(dtype=object), shadow=pd.Series(dtype=float))
    priced = keep[keep.shadow.notna()]
    waiting = keep[keep.shadow.isna()]
    L += [f"Cohort: CLOSED {SLEEVE} fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if since_given is None else 'floor given'}) · {_GK.operator_closed_line(ox)} · priced {len(priced)} · "
          f"skipped {(priced.why == 'SKIP').sum()} · pending / unpriced {len(waiting)}" + (" (" + ", ".join(f"{a} {b:%m-%d %H:%M}" for a, b in zip(waiting.pair, waiting.o)) + ")" if len(waiting) else "") + ".", "", "Before → after now: " + before_after(priced) + ".", ""]
    if stopped:
        L += ["⚠ kline fetch stopped this run (budget, cooldown or a network error): the rest stay unpriced; nothing freezes.", ""]
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = st.get(SLEEVE) if ok else None
    damaged = rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec))
    rec = None if damaged else rec
    if damaged:
        L.append("⚠ frozen record damaged — operator restore needed; this line does not freeze until then.")
    hold = []
    if not rec and ok and not damaged:
        pre = crossing_prefix(priced) if len(priced) else None
        if pre is not None:
            last = pre.o.max()
            if (waiting.o <= last).any() or stopped:
                hold.append("a signal inside the prefix is not priced yet")
            elif newest is None or newest < last + pd.Timedelta(minutes=WAIT_MIN):
                hold.append(f"newest export < the crossing signal + {WAIT_MIN} min (a fill could still be open)")
            else:
                state, det = verdict(pre)
                rec = dict(state=state, detail=det, before_after=before_after(pre), at=f"{last:%Y-%m-%d %H:%M} UTC",
                           run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"),
                           keys=[f"{a:%Y-%m-%dT%H:%M:%S}|{b}|{w}" for a, b, w in zip(pre.o, pre.pair, pre.why)])
                if state_path:
                    st[SLEEVE] = rec
                    NF.du_save_state(st, state_path)
    live_state, live_det = verdict(priced)
    if rec and ok and state_path:
        L.append(f"**Frozen verdict (crossing at signal {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})")
        L.append(f"Before → after at the crossing: {rec.get('before_after', '–')}")
        L.append(f"Live (information only, never re-decides): {live_state} — {live_det}")
    else:
        L.append(f"Now: ⏳ {live_state} ({live_det}).")
    if hold:
        L.append("⏸ freezing deferred: " + " · ".join(hold) + " — re-checked next run.")
    if not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += ["", f"Evidence: {REGISTRATION}.",
          f"FROZEN bar (DL {DL}, per signal, skip = 0): at N ≥ {N_MIN} priced signals on ≥ {DAYS_MIN} UTC days, decided ONCE — mean Δ ≤ 0 → RETIRED; {CANDIDATE} iff mean Δ ≥ {D_MIN:+.2f} ∧ shadow mean > 0 ∧ P(Δ > 0) ≥ {P_MIN:.2f} ({BOOT_N:,}, seed {BOOT_SEED}) ∧ "
          f"no signal ≥ {SHARE_MAX:.0%} of + Σ Δ; else KEEP OBSERVING.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
