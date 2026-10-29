#!/usr/bin/env python3
"""🚪 Scout — EXIT_SHADOWS (FLIP only): today's flip exit vs fixed take profits other sleeves run, re-priced on each real flip's own
path (pre-registered 2026-10-29, DECISION_LOG 411, operator: "use the exit strategies we have in other sleeves" · "keep it only for flips
with +1.5 / +2 / +3"). OBSERVE ONLY — never changes config, no bot API.

WHY FLIPS ONLY (the 10-29 historical exercise on every full-size fill, master on today's stack + B25, real entry fees and live stops):
  SPIKE_FADE 88 — every fixed TP loses to today's trail (tick-priced −0.27 … −0.41 %/fill) → answered, kept.
  MOMENTUM LONG 177 — TP1.5 / TP2 tie (0.00 / −0.01), TP3 … TP6 −0.13 … −0.23, late TP @30 / 90-min cap −0.01 → answered, kept.
  MOMENTUM SHORT 45 — every fixed TP −0.18 … −0.54 → answered, kept.
  FLIP 40 — TP2 / TP3 / TP4 +0.40 / +0.54 / +0.69 %/fill (P 0.95 – 0.97, below the 3-exit luck bar), but carried by the Jun-17 → Jul-1
  washed-out window (TP4: 3 days ≈ 23 of 27 pts); since September TP3 / TP4 worse, TP2 mixed → the history cannot settle it → this
  forward read.

COHORT  FLIP:* fills (their own direction), FULL-SIZE only (*_PROBE out), operator-closed out. Stop = the engine's per fill: base −0.7,
        widened to sl_atr_multiplier 1.5 × entry_atr_pct, capped at sl_atr_widen_floor_pct −1.2 (B25 live flip stops −1.20; master 17 / 17
        flip stop closes on the line). Not modelled: the STRONG_BUY tier's signal_active_sl −1.0 (never fired on a flip so far — parity
        is the guard).
EXITS   TP1.5 (FRENZY_LONG) · TP2 / TP3 (LITE C) — fixed take profit, the fill's live stop, 12 h.
PRICING   net % of notional on the fill's own path from its entry price (the fill's REAL entry fee — maker 0.018 on about half
          the flips, from entry_fee / notional — + the taker 0.045 × exit/entry out: the bot's paper accounting), full-day ticks from the shared caches only (never an archive download), then public fapi 1m pages
          (scout_long_stop4.fapi_klines — budgeted, cached, cooldown-aware). A 1m bar touching both lines = the stop (adverse first).
          No slippage on either side (the paper comparator books its stops at the line); every exit pays the taker exit fee (TP
          included — conservative vs a maker TP by ≈ 0.027). On a 1m-only path the first bar is treated as opening AT the entry (the
          entry's own minute is not on the path — a stop inside it books at the line, never as a fake gap). Comparator = the BOOKED
          result (as traded).
          A fill is priced once its whole window (the longest exit, or its live hold) is over; its result is cached in the state and
          re-walked only when its signature changes (the exits, every exit constant, the fill's stop, booked % or close reason). An
          exit the path cannot decide (data gap / a short page) is retried on another UTC day, then accepted as unpriced (out of that
          exit's N). No path at all → retried; UNPRICEABLE (out, counted) only after misses on 3 different UTC days.
PARITY    a live STOP close shows the fill's stop line touched by the close; a live non-stop close booked clearly above the line never shows
          the line crossed ≥ 1 min before it. < 90 % agreement in a prefix → no verdict.
🔒 FROZEN VERDICT (per sleeve, decided ONCE, never re-fit): at the first prefix (by open time) of ≥ 20 counted fills on ≥ 8 UTC days,
   every prefix fill priced and the newest export downloaded ≥ the prefix end + the 1200-min max hold (the exports carry CLOSED rows
   only — a fill still open would be missing), per exit: Δ = exit − booked → CANDIDATE (operator decides an engine change) iff mean Δ ≥
   +0.10 %/fill ∧ day-clustered bootstrap P(mean Δ > 0) ≥ 1 − 0.10 / m (m = the sleeve's exits — the luck correction) ∧ no single fill
   ≥ 50 % of the positive Σ Δ ∧ parity ≥ 90 %; else KEEP (today's exit).
COUNTED   fills opened ≥ the DECISION_LOG 411 commit + 10 min. Context (NEVER counted): the latest 100 earlier fills per sleeve, cache
          only (results cached in their own state bucket).
STATE     reports/SCOUT_EXIT_SHADOWS.json (gitignored): frozen verdicts · per-fill results · fetch misses.
Usage: venv/bin/python -I scripts/scout_exit_shadows.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG                # noqa: E402  (Downloads + exports_archive/, DL 409)
import scout_gatekit as _GK              # noqa: E402  (operator-closed out · the DL commit time)
import scout_b1h_negflank as NF          # noqa: E402  (state I/O)
import scout_long_stop4 as LS            # noqa: E402  (budgeted public fapi klines + shared tick caches + path assembly)
import scout_willy_exit_v2 as V2         # noqa: E402  (the V2b walker — a generic net-path walker)

DL = 411
N_MIN, DAYS_MIN = 20, 8
MIN_GAIN, ALPHA, SHARE_MAX, PARITY_MIN = 0.10, 0.10, 0.50, 0.90
BOOT_N, BOOT_SEED = 4000, 7
TAKER, SLIP = 0.045, 0.0   # (deep review M3) no slip on either side: the paper comparator books its stops at the line
MIN, DAY = 60_000, 86_400_000
FIX_CAP, MAX_HOLD = 720, 1200
SL_BASE, SL_ATR_MULT, SL_CAP = 0.7, 1.5, 1.2   # the engine's base stop (STRONG / VERY_STRONG tiers) and its ATR widening
FLIP_EXITS = ("TP1.5", "TP2", "TP3")
SLEEVES = {   # name → ((entry_strategy match, direction or None), exits) — FLIP only (operator 10-29; the others answered by history)
    "FLIP": (("FLIP:", None), FLIP_EXITS),
}
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_EXIT_SHADOWS.json")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "scout_exit_shadows")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "pnl", "close_reason",
        "cell_multiplier_source", "entry_atr_pct", "entry_fee", "notional_value")
MAX_REQ_RUN, CONTEXT_MAX, MISS_DAYS = 20, 100, 3


# ─────────────────────────── cohort ───────────────────────────
def sleeve_of(es, d):
    es, d = str(es), str(d)
    for name, ((m, md), _) in SLEEVES.items():
        if (es.startswith(m) if m.endswith(":") else es == m) and (md is None or d == md):
            return name
    return None


def stop_of(sleeve, atr):
    """the fill's live stop (positive % net): −0.7 widened to 1.5 × entry ATR %, capped at −1.2. PURE."""
    try:
        a = float(atr)
    except (TypeError, ValueError):
        a = float("nan")
    return SL_BASE if not (a == a and a > 0) else float(min(SL_CAP, max(SL_BASE, SL_ATR_MULT * a)))


def load(paths=None):
    """→ (closed full-size FLIP fills — dedup (opened_at[:19], pair, direction), CLOSED beats OPEN, newest export wins —,
    the newest export's download time (naive UTC) or None)."""
    fr, read, files = [], [], list(_EG.glob_exports(EXPORT_GLOB) if paths is None else paths)
    for f in files:
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f)))
            read.append(f)
    newest = None
    for f in read:                                       # only exports that were actually read (a corrupt download never clears the wait)
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["sleeve", "o", "c", "te", "tc", "day", "pct", "E", "sl", "fin"]), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d["sleeve"] = [sleeve_of(a, b) for a, b in zip(d.entry_strategy, d.direction)]
    d = d[d.sleeve.notna() & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")
    d = d[d._c].copy()
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    d["c"] = pd.to_datetime(d.closed_at.astype(str).str.replace(" ", "T").str[:19], format="ISO8601", errors="coerce")
    d["pct"] = pd.to_numeric(d.pnl_percentage, errors="coerce")
    d["E"] = pd.to_numeric(d.entry_price, errors="coerce")
    d = d[d.o.notna() & d.c.notna() & d.pct.notna() & (d.E > 0)].copy()
    tf = pd.to_datetime(d.opened_at.astype(str).str.replace(" ", "T").str[:26], format="ISO8601", errors="coerce")
    d["te"] = np.where(tf.notna(), tf.values.astype("datetime64[ms]").astype("int64"), d.o.values.astype("datetime64[ms]").astype("int64"))
    d["tc"] = d.c.values.astype("datetime64[ms]").astype("int64")
    d["day"] = d.o.dt.strftime("%Y-%m-%d")
    d["sl"] = [stop_of(s, a) for s, a in zip(d.sleeve, d.entry_atr_pct)]
    fee = pd.to_numeric(d.entry_fee, errors="coerce") / pd.to_numeric(d.notional_value, errors="coerce") * 100
    d["fin"] = np.where(fee.notna() & (fee >= 0) & (fee < 0.2), fee, TAKER)   # (deep review H1) the fill's REAL entry fee % (maker 0.018 on about half the flips)
    return d.sort_values(["o", "pair"]).reset_index(drop=True), newest


# ─────────────────────────── path (net % for either side) ───────────────────────────
def to_net(P, side, fin=TAKER):
    """gross-long % path (scout_long_stop4.assemble_path) → the bot's NET % path for `side` (hi = the favourable extreme): gross − the
    fill's entry fee % `fin` − the taker exit fee × exit / entry (the paper accounting; exits always taker). PURE."""
    def net(x, short):
        r = 1 + np.asarray(x, float) / 100.0
        return (-(r - 1) * 100 if short else (r - 1) * 100) - fin - TAKER * r
    s = str(side).upper() == "SHORT"
    hi, lo = (P["lo"], P["hi"]) if s else (P["hi"], P["lo"])
    n_t = int(P.get("n_tick_s", 0))
    bar = np.zeros(len(P["t"]), bool)
    bar[n_t:] = True
    op = np.array(P["op"], float)
    if n_t == 0 and len(op):
        op[0] = 0.0          # (deep review M2) 1m-only path: the entry minute is off the path → the first bar opens AT the entry (no fake gap)
    return dict(t=np.asarray(P["t"], "int64"), hi=net(hi, s), lo=net(lo, s), op=net(op, s), cl=net(P["cl"], s), bar=bar,
                end_ms=int(P["end_ms"]), te=int(P["te"]), n_tick=n_t)


def raw_path(r, horizon, bud, now_ms, fetch=True):
    """gross-long path: full-day tick files from the SHARED caches only (never an archive download), then public 1m pages of ≤ 1000
    bars until the window end. → (P or None, 'ok' / 'budget' / 'empty')."""
    te, E = int(r.te), float(r.E)
    lim = te + (horizon + 1) * MIN
    ts, ps, cov = [], [], te
    for d in pd.date_range(pd.Timestamp(te, unit="ms").normalize(), pd.Timestamp(lim, unit="ms").normalize()):
        f = LS._local_tick_file(r.pair, f"{d:%Y-%m-%d}")
        if f is None:
            break
        try:
            with np.load(f) as z:
                if "w0" in z.files:
                    break
                t, p = z["t"].astype("int64"), z["p"].astype(float)
        except Exception:
            break
        o = np.argsort(t, kind="stable")
        ts.append(t[o]); ps.append(p[o])
        cov = int(d.value // 1_000_000) + DAY
    ticks = (np.concatenate(ts), np.concatenate(ps), min(cov, lim)) if ts else None
    cov = min(cov, lim) if ts else te
    k1, gap = None, False
    first = -(-cov // MIN) * MIN
    if cov < lim and int((lim - first) // MIN) >= 1:
        k1 = []
        while first + MIN <= lim:
            n = min(int((lim - first) // MIN), 1000)
            if n < 1:
                break
            page = LS.fapi_klines(r.pair, "1m", first, n, bud, CACHE, now_ms, fetch=fetch)
            if page is None:
                return None, "budget"
            if not page or len(page) < n:
                k1 += page or []
                gap = True                                # a short / empty page before the window end (network error, delisting)
                break
            k1 += page
            first += n * MIN
    P = LS.assemble_path(te, E, horizon, ticks, k1 or None)
    if P is None:
        return None, "empty"
    return P, ("gap" if gap else "ok")


# ─────────────────────────── exits ───────────────────────────
def _slip(w):
    return float(w["pct"]) - (SLIP if w["why"] in ("STOP", "CAP") else 0.0)


def walk_fixed(P, tp, sl, cap=FIX_CAP):
    """fixed TP / stop / cap on a net path (scout_willy_exit_v2.walk_v2, no late TP) → net % (nan = undecided)."""
    w = V2.walk_v2(P, tp, sl=sl, cap=cap, late=None)
    return np.nan if w["why"] == "UNRES" else _slip(w)


def is_stop_close(reason):
    """a live close on a stop line: STOP_LOSS* / FLIP_STOP_LOSS / PATTERN_FIXED_SL / HARD_STOP (never TRAILING_STOP — a trail in profit)."""
    u = str(reason or "").upper()
    return "STOP_LOSS" in u or "_SL" in u or "HARD_STOP" in u


def parity_ok(P, sl, booked, reason, tc):
    """the path agrees with the bot (None = path short):
    · a live STOP close must show the stop line (−sl, 0.15 tolerance) touched by the live close (+1 min);
    · a live NON-stop close that booked clearly above the line (> −sl + 0.15) must NOT show the line crossed ≥ 1 min before it
      (else the bot rode through a stop the path says was hit → the path is wrong). A non-stop close AT the line is consistent."""
    t, lo = P["t"], P["lo"]
    if P["end_ms"] < int(tc) - MIN:
        return None
    if is_stop_close(reason):
        return bool(((t <= int(tc) + MIN) & (lo <= -sl + 0.15)).any())
    if float(booked) <= -sl + 0.15:
        return True
    return not bool(((t <= int(tc) - MIN) & (lo <= -sl)).any())


def horizon_of(r):
    return max(FIX_CAP, int((int(r.tc) - int(r.te)) // MIN) + 2)


def exits_on(P, r):
    """→ {exit: net %} for one fill on its net path. PURE."""
    _, exits = SLEEVES[r.sleeve]
    out = {}
    for x in exits:
        if x.startswith("TP"):
            out[x] = walk_fixed(P, float(x[2:]), r.sl)
    return out


def fill_key(r):
    return f"{r.pair}|{int(r.te)}"


def signature(r):
    """everything a cached result depends on: the sleeve's exits + every exit constant + this fill's stop, booked % and close reason
    (Δ reads the booked %; parity reads the reason). A mismatch → re-walk (final review)."""
    _, exits = SLEEVES[r.sleeve]
    return json.dumps([list(exits), FIX_CAP, TAKER, SLIP, round(float(r.fin), 6),
                       round(float(r.E), 10), int(r.tc), round(float(r.sl), 6), round(float(r.pct), 6), str(r.close_reason)], default=str)


def price_fill(r, bud, now_ms, cache, fetch=True):
    """→ dict(status PRICED / PENDING / UNPRICEABLE, exits, parity[, err]). `cache` = the state's per-fill store (results + misses).
    Never raises."""
    k = fill_key(r)
    c = cache.get(k)
    sig = signature(r)
    if c and c.get("status") == "PRICED" and c.get("sig") != sig:
        c = cache[k] = {kk: v for kk, v in c.items() if kk in ("miss_days", "why")}   # rules or booked numbers changed → re-walk
    if c and c.get("status") in ("PRICED", "UNPRICEABLE"):
        return dict(status=c["status"], exits={x: (np.nan if v is None else v) for x, v in (c.get("exits") or {}).items()},
                    parity=c.get("parity"))
    try:
        if now_ms < int(r.te) + (horizon_of(r) + 2) * MIN:
            return dict(status="PENDING", exits={}, parity=None, err="window open")
        raw, st = raw_path(r, horizon_of(r), bud, now_ms, fetch=fetch)
        if raw is None:
            if st == "empty" and fetch:
                return _miss(cache, k, now_ms, "no path")
            return dict(status="PENDING", exits={}, parity=None, err=st)
        P = to_net(raw, r.direction, getattr(r, "fin", TAKER))
        out = exits_on(P, r)
        par = parity_ok(P, r.sl, r.pct, r.close_reason, r.tc)
        today = pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d")
        tdays = set((c or {}).get("try_days") or [])
        if any(v != v for v in out.values()) or st == "gap":
            tdays.add(today)
            if len(tdays) < 2:                         # retry on another UTC day (a network blip heals); then accept the gap
                if fetch:
                    cache[k] = dict(c or {}, try_days=sorted(tdays))
                return dict(status="PENDING", exits=out, parity=par, err="exit undecided / short page (data gap) — retry another day")
        res = dict(status="PRICED", exits=out, parity=par)
        cache[k] = dict(status="PRICED", sig=sig, exits={x: (None if v != v else float(v)) for x, v in out.items()}, parity=par)
        return res
    except Exception as e:
        return _miss(cache, k, now_ms, f"error {str(e)[:60]}") if fetch else dict(status="PENDING", exits={}, parity=None, err=str(e)[:60])


def _miss(cache, k, now_ms, why):
    """a fetch / pricing miss on this UTC day; UNPRICEABLE once misses fall on MISS_DAYS distinct days."""
    c = cache.setdefault(k, {})
    days = set(c.get("miss_days") or [])
    days.add(pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d"))
    c["miss_days"] = sorted(days)
    c["why"] = why
    if len(days) >= MISS_DAYS:
        c["status"] = "UNPRICEABLE"
        return dict(status="UNPRICEABLE", exits={}, parity=None, err=why)
    return dict(status="PENDING", exits={}, parity=None, err=f"{why} (miss {len(days)}/{MISS_DAYS})")


# ─────────────────────────── verdict ───────────────────────────
def p_pos(delta, days, n=BOOT_N, seed=BOOT_SEED):
    """day-clustered bootstrap P(mean Δ > 0). PURE."""
    g = pd.DataFrame(dict(x=np.asarray(delta, float), d=np.asarray(days))).groupby("d").x.agg(["sum", "count"])
    if len(g) < 2:
        return None
    idx = np.random.default_rng(seed).integers(0, len(g), size=(n, len(g)))
    s, c = g["sum"].values[idx].sum(1), g["count"].values[idx].sum(1)
    return float((s / c > 0).mean())


def verdict(rows, exits):
    """rows = priced counted fills (pct, day, exits dict, parity) of ONE sleeve → ({exit: (state, text)}, parity share). PURE."""
    par = [p for p in rows.parity if p is not None]
    pshare = (sum(par) / len(par)) if par else None
    out, m = {}, len(exits)
    for x in exits:
        v = np.array([e.get(x, np.nan) for e in rows.exits], float)
        okm = ~np.isnan(v)
        v, d, days = v[okm], v[okm] - rows.pct.values[okm], rows.day.values[okm]
        if not len(d):
            out[x] = ("KEEP today's exit", "N 0 (not priceable on this prefix)")
            continue
        pp = p_pos(d, days)
        pos = np.clip(d, 0, None)
        share = float(pos.max() / pos.sum()) if pos.sum() > 0 else 0.0
        txt = (f"N {len(d)} · {len(set(days))} d · WR {(v > 0).mean():.0%} · avg {v.mean():+.3f} vs today {rows.pct.values[okm].mean():+.3f}"
               f" · Δ {d.mean():+.3f} %/fill · P(Δ>0) {(pp if pp is not None else float('nan')):.2f} (bar {1 - ALPHA / m:.3f}) · top fill "
               f"{share:.0%} of the gain")
        ok = (d.mean() >= MIN_GAIN and pp is not None and pp >= 1 - ALPHA / m and share < SHARE_MAX
              and pshare is not None and pshare >= PARITY_MIN)
        out[x] = ("CANDIDATE (operator decides)" if ok else "KEEP today's exit", txt)
    return out, pshare


def crossing(rows):
    """the first prefix (by open time) of ≥ 20 fills on ≥ 8 UTC days → index end (exclusive) or None."""
    seen = set()
    for k, d in enumerate(rows.day.values, 1):
        seen.add(d)
        if k >= N_MIN and len(seen) >= DAYS_MIN:
            return k
    return None


# ─────────────────────────── report ───────────────────────────
def _table(rows, exits):
    if not len(rows):
        return ["| Exit | N | WR | avg % | Δ vs today | better / worse |", "|---|---|---|---|---|---|", "| today | 0 | – | – | – | – |"]
    L = ["| Exit | N | WR | avg % | Δ vs today | better / worse |", "|---|---|---|---|---|---|",
         f"| today (booked) | {len(rows)} | {(rows.pct > 0).mean():.0%} | {rows.pct.mean():+.3f} | – | – |"]
    for x in exits:
        v = np.array([e.get(x, np.nan) for e in rows.exits], float)
        ok = ~np.isnan(v)
        if not ok.any():
            L.append(f"| {x} | 0 | – | – | – | – |")
            continue
        d = v[ok] - rows.pct.values[ok]
        L.append(f"| {x} | {int(ok.sum())} | {(v[ok] > 0).mean():.0%} | {v[ok].mean():+.3f} | {d.mean():+.3f} | "
                 f"{int((d > 1e-9).sum())} / {int((d < -1e-9).sum())} |")
    return L


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None, fetch=True, max_req=MAX_REQ_RUN):
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🚪 EXIT_SHADOWS — today's exit vs the exits other sleeves run (pre-registered 2026-10-29, DECISION_LOG {DL}, OBSERVE only)", "",
         "FLIP fills only: fixed TP +1.5 / +2 / +3 (12 h, the fill's live stop −0.7 … −1.2 by ATR) vs today's flip exit. Δ = exit − the "
         "BOOKED result, per fill, same path. Full-size only. Fades / momentum longs / momentum shorts answered by the 10-29 historical "
         "exercise (DL 411) — today's exits kept, not tracked.", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="EXIT_SHADOWS")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    d, newest = load(paths)
    keep, ox = _GK.split_operator_closed(d) if len(d) else (d, d)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    cache = st.setdefault("_fills", {}) if ok else {}
    n0 = json.dumps([cache, st.get("_context", {})], sort_keys=True, default=str)
    bud = LS.Budget()
    dirty = False
    for name, (_, exits) in SLEEVES.items():
        s = keep[keep.sleeve == name]
        fw, cx = s[s.o >= since], s[s.o < since].tail(CONTEXT_MAX)
        res = []
        for r in fw.itertuples():
            if bud.requests + 2 > max_req:
                bud.halt(f"request cap {max_req}/run")
            res.append(price_fill(r, bud, now_ms, cache, fetch=fetch))
        fw = fw.assign(status=[x["status"] for x in res], exits=[x["exits"] for x in res], parity=[x["parity"] for x in res],
                       err=[x.get("err", "") for x in res])
        pr = fw[fw.status == "PRICED"]
        L += [f"### {name} — counted (opened ≥ {since:%Y-%m-%d %H:%M} UTC): {len(fw)} closed · priced {len(pr)} · pending "
              f"{int((fw.status == 'PENDING').sum())} · unpriceable {int((fw.status == 'UNPRICEABLE').sum())}", ""]
        L += _table(pr, exits) + [""]
        un = [f"{p.pair} {p.o:%m-%d %H:%M} ({(cache.get(fill_key(p)) or {}).get('why', '?')})" for p in fw[fw.status == "UNPRICEABLE"].head(5).itertuples()]
        if un:
            L.append("Unpriceable (out of the read): " + "; ".join(un))
        frozen = st.get(name) if ok else None
        cand = fw[fw.status != "UNPRICEABLE"]
        if frozen:
            L.append(f"**Frozen verdict (prefix to {frozen['at']}, parity {frozen['parity']}):** "
                     + " · ".join(f"{x} {v[0]}" for x, v in frozen["exits"].items()))
        else:
            k = crossing(cand)
            if k is None:
                L.append(f"⏳ COLLECTING — {len(cand)}/{N_MIN} fills on {cand.day.nunique()}/{DAYS_MIN} days.")
            else:
                pre = cand.iloc[:k]
                wait = newest is None or newest < pre.o.max() + pd.Timedelta(minutes=MAX_HOLD)
                if (pre.status == "PENDING").any() or wait:
                    pend = pre[pre.status == "PENDING"]
                    L.append("⏸ freezing deferred — " + ("; ".join(f"{p.pair} {p.o:%m-%d %H:%M} ({p.err})" for p in pend.head(5).itertuples())
                                                      if len(pend) else "")
                             + (f"{'; ' if len(pend) else ''}newest export < the prefix end + {MAX_HOLD} min (a fill could still be open)"
                                if wait else "") + " — re-checked next run.")
                else:
                    v, ps = verdict(pre, exits)
                    rec = dict(at=f"{pre.o.max():%Y-%m-%d %H:%M} UTC", parity=(f"{ps:.0%}" if ps is not None else "n/a"),
                               exits={x: list(t) for x, t in v.items()})
                    if ok and state_path:
                        st[name] = rec
                        dirty = True
                    L.append(f"**Verdict (prefix to {rec['at']}, parity {rec['parity']})"
                             + ("" if (ok and state_path) else " — NOT FROZEN (no state / state corrupt)") + ":**")
                    L += [f"- {x}: {t[0]} — {t[1]}" for x, t in v.items()]
        if len(cx):
            ctx = st.setdefault("_context", {}) if ok else {}
            cres = [price_fill(r, bud, now_ms, ctx, fetch=False) for r in cx.itertuples()]
            cx = cx.assign(status=[x["status"] for x in cres], exits=[x["exits"] for x in cres])
            cp = cx[cx.status == "PRICED"]
            L += ["", f"Context (before the DL {DL} commit, latest {CONTEXT_MAX}, cache only, NEVER counted): {len(cx)} fills · priced {len(cp)}", ""]
            L += _table(cp, exits) if len(cp) else []
        L.append("")
    if ok and state_path and (dirty or json.dumps([cache, st.get("_context", {})], sort_keys=True, default=str) != n0):
        NF.du_save_state(st, state_path)
    L.append(f"Operator-closed out: {_GK.operator_closed_line(ox)} · Binance requests this run {bud.requests}"
             + (f" ({bud.why})" if bud.why else "") + ".")
    L.append(f"🔒 Bar (DL {DL}, frozen): at ≥ {N_MIN} fills on ≥ {DAYS_MIN} days, once — CANDIDATE iff mean Δ ≥ +{MIN_GAIN:.2f} "
             f"%/fill ∧ day-bootstrap P(Δ > 0) ≥ 1 − {ALPHA:.2f}/m ∧ no fill ≥ {SHARE_MAX:.0%} of the gain ∧ path parity ≥ {PARITY_MIN:.0%}; else KEEP. "
             f"m = {len(FLIP_EXITS)} exits. A CANDIDATE is a proposal for a reviewed study (haircut, before → after per batch with the daily "
             "compound return, revert gate), never a ship.")
    if not ok:
        L.append("⚠ frozen state corrupt — operator restore needed; freezing skipped this run.")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(run()))
