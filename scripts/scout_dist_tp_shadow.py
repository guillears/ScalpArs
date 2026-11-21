#!/usr/bin/env python3
"""🎯👁 Scout — DIST_TP_SHADOW: DISTRIBUTION_SHORT at TP 5 % / TP 8 % vs today's Exit B TP (DECISION_LOG 605, operator 2026-11-20
"register the DIST TP 5 / 8 watch line too"; OBSERVE only — alert-only, never changes config, public 1m klines only).

QUESTION   reports/TP_GRID_2026-11-20_DIST.md (DL 603): at one size a lower TP loses per trade (backtest TP 5 −0.20 … −0.55, TP 8 −0.01 … −0.32
           %/fill vs TP 10), but at EQUAL drawdown it is a coin flip over 2023–26 and leans to TP 5 / 8 on HOLDOUT (P better 0.79–0.81).
           This line tracks both on real fills so the batch review can decide TP × size on fresh evidence.
RULE       every CLOSED DISTRIBUTION_SHORT fill opened ≥ the DL 605 commit + 10 min (orders exports, dedup, CLOSED wins, operator-closed out)
           is walked on its own 1m path from the REAL fill price (entry_price), from the minute AFTER the fill minute (its range holds
           pre-fill prints), for 24 h with the fill's own stamped stop (entry_dist_sl_pct, stop first on a bar touching both, a gap books
           the open) at three TPs: TODAY (entry_dist_tp_pct = min(20 × ATR, 10 %)), 5 % and 8 % (a TP lower than today's only; a resting
           maker TP fills only when a trade goes ≥ 0.05 % through). Net = gross − 0.05 entry fee − exit fee (maker 0.02 / taker 0.05)
           − the entry leg still owed (services.exhaustion.entry_leg_slip: an impact-mode fill already booked its walk) − on a taker exit
           max(the fill's modelled slip, 0.10). Levels = services.distribution.row_levels (stamps, else the caps). No funding (stated).
           Paired Δ5 = net(TP 5) − net(TODAY), Δ8 = net(TP 8) − net(TODAY); parity line = the TODAY walk vs the booked %.
🔒 READ    frozen ONCE (reports/SCOUT_DIST_TP_SHADOW.json, gitignored) at the first k ≥ 30 priced fills spanning ≥ 10 UTC days, per arm:
           SHIP CANDIDATE (operator decides, together with the DIST size) iff the one-sided day-cluster bootstrap P(mean Δ > 0) ≥ 0.95
           (4,000, seed 7) ∧ the mean Δ without the best 5 % of fills > 0 · else NO SHIP (keep today's TP). Shown beside it (never deciding):
           the RISK read at one size 0.61 × equity over the cohort's calendar days — daily compound, max drawdown, worst day per arm — and the
           equal-drawdown daily compound (each arm sized to TODAY's max drawdown; each arm bucketed by its own exit day). A fill whose own TP is
           ≤ the arm's level is out of that arm (Δ ≡ 0). Freezing waits while a kline fetch stopped this run or the
           newest orders export is < the read prefix's last fill + 24 h (an earlier-opened fill could still be open). A fill whose path is still
           incomplete 48 h after entry with the klines answering → NOT_EVALUABLE (out of the read; the read cannot stall).
Usage: venv/bin/python -I scripts/scout_dist_tp_shadow.py"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import scout_exh_wide_585 as _W    # noqa: E402  (shared orders loader)
import scout_long_stop4 as LS      # noqa: E402  (budgeted, cached public fapi klines)
from services.distribution import DISTRIBUTION_SHORT, row_levels   # noqa: E402
from services.exhaustion import entry_leg_slip       # noqa: E402

DL = 605
KEY = "DIST_TP_SHADOW"
STATE = os.path.join(_R, "reports", "SCOUT_DIST_TP_SHADOW.json")
CACHE = os.path.join(_R, "reports", "backtest_cache", "scout_dist_tp")
COLS = ("opened_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "notional_value",
        "entry_price", "entry_dist_tp_pct", "entry_dist_sl_pct", "entry_dist_model_slip_pct", "entry_depth_mode", "entry_depth_walk_pct")
ARMS = (5.0, 8.0)
THRU, FEE_IN, FEE_MK, FEE_TK = 0.05, 0.05, 0.02, 0.05
EXIT_SLIP_MIN = 0.10                     # taker exits pay at least the study's model slip (DL 603)
HOLD_H, MIN1, HOUR, DAY, PAGE = 24, 60_000, 3_600_000, 86_400_000, 900
N_MIN, DAYS_MIN, P_MIN, BOOT, SEED, TOP = 30, 10, 0.95, 4000, 7, 0.05
SIZE = 0.61


def _thresholds():
    try:
        with open(os.path.join(_R, "trading_config.json")) as fh:
            return (json.load(fh) or {}).get("thresholds") or {}
    except Exception:
        return {}


_TH = _thresholds()
STALE_H = 48


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def walk(bars, E, te, tp, sl, slip=0.0, entry_slip=None):
    """→ (why, net %) — see walk_x. PURE."""
    return walk_x(bars, E, te, tp, sl, slip, entry_slip)[:2]


def walk_x(bars, E, te, tp, sl, slip=0.0, entry_slip=None):
    """short from the real fill price E at te; 1m CLOSED bars [[open ms, o, h, l, c], …]; from the minute after the fill minute; stop
    sl % above first on a bar touching both (a gap books the open); TP tp % below needs THRU % through (maker, books the line); 24 h →
    the first bar opening ≥ te + 24 h, at its open (taker). → (why, net %, exit bar open ms); PENDING when the path ends first. PURE."""
    if not (E and E > 0 and tp and tp > 0 and sl and sl > 0):
        return "UNPRICED", np.nan, None
    b = [q for q in (bars or []) if int(q[0]) >= int(te) // MIN1 * MIN1 + MIN1]
    if not b:
        return "UNPRICED", np.nan, None
    s = max(float(slip or 0.0), 0.0)
    se = s if entry_slip is None else max(float(entry_slip), 0.0)   # the entry leg still owed (DL 584: impact fills booked their walk)
    sx = max(s, EXIT_SLIP_MIN)                                       # a taker exit pays ≥ the study's 0.10 model slip
    stop, tpl = E * (1 + sl / 100), E * (1 - (tp + THRU) / 100)
    taker = lambda px: float(-(px / E - 1) * 100 - FEE_IN - FEE_TK - se - sx)
    for q in b:
        if int(q[0]) >= te + HOLD_H * HOUR:
            return "TIME", taker(float(q[1])), int(q[0])
        if q[2] >= stop:
            return "STOP", taker(max(float(q[1]), stop)), int(q[0])
        if q[3] <= tpl:
            return "TP", float(tp - FEE_IN - FEE_MK - se), int(q[0])
    return "PENDING", np.nan, None


def fetch_bars(pair, te, now_ms, bud):
    """1m closed bars from te's minute for 24 h + 1 bar (2 pages). → bars or None on a budget / error stop (this fill's own errors only)."""
    start, end, out, e0 = int(te) // MIN1 * MIN1, int(te) + HOLD_H * HOUR + MIN1, [], getattr(bud, "errors", 0)
    while start <= min(end, now_ms):
        pg = LS.fapi_klines(pair, "1m", start, PAGE, bud, CACHE, now_ms, fetch=True)
        if pg is None or getattr(bud, "errors", 0) > e0:
            return None
        out += [q[:5] for q in pg]
        if len(pg) < PAGE:
            break
        start += PAGE * MIN1
    return out


def arm_verdict(deltas, days, seed=SEED):
    """one arm's frozen read on its paired Δ (fill order; NaN-free prefix) → (state, text). SHIP CANDIDATE iff the one-sided day-cluster
    bootstrap P(mean Δ > 0) ≥ P_MIN ∧ mean Δ without the best TOP share > 0; else NO SHIP. PURE."""
    d, w = np.asarray(deltas, float), np.asarray(days)
    u = np.unique(w)
    g = [d[w == x] for x in u]
    rng = np.random.default_rng(seed)
    m = np.array([np.concatenate([g[i] for i in rng.integers(0, len(u), len(u))]).mean() for _ in range(BOOT)])
    p = float((m > 0).mean())
    cut = max(1, int(np.ceil(TOP * len(d))))
    loo = float(np.sort(d)[:-cut].mean())
    ok = p >= P_MIN and loo > 0
    return ("SHIP CANDIDATE" if ok else "NO SHIP"), (f"mean Δ {d.mean():+.2f} pts/fill · P(Δ > 0) {p:.2f} (need ≥ {P_MIN:.2f}) · "
                                                     f"without the best {cut} {loo:+.2f} (need > 0)")


def prefix(rows):
    """rows = [(day, Δ5, Δ8, ...)] in fill order, NaN-free → the first k ≥ N_MIN spanning ≥ DAYS_MIN days, or None. PURE."""
    for k in range(N_MIN, len(rows) + 1):
        if len({r[0] for r in rows[:k]}) >= DAYS_MIN:
            return k
    return None


def risk_read(rows, d0, d1):
    """one-size daily compound / max drawdown / worst day per arm + the equal-drawdown daily compound (each arm sized to TODAY's max
    drawdown). rows = [((close day today, close day TP5, close day TP8), net_today, net5, net8)] — each arm bucketed by ITS OWN close day.
    Information only. PURE."""
    if not rows:
        return {}
    days = np.arange(d0, d1 + 1)
    out = {}
    def series(j):
        s = pd.Series([r[j] for r in rows]).groupby([r[0][j - 1] for r in rows]).sum()
        return s.reindex(days, fill_value=0.0).values
    def stats(r, x):
        g = np.maximum(1 + r * x / 100, 1e-9)
        eq = np.cumprod(g)
        return (eq[-1] ** (1 / len(g)) - 1) * 100, (eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1).min() * 100, (g.min() - 1) * 100
    R = {n: series(j) for n, j in (("TODAY", 1), ("TP5", 2), ("TP8", 3))}
    tdd = stats(R["TODAY"], SIZE)[1]
    for n, r in R.items():
        dcr, dd, wd = stats(r, SIZE)
        eq_dcr = None
        if tdd < 0:
            lo, hi = 0.0, 50.0
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if stats(r, mid)[1] > tdd else (lo, mid)
            eq_dcr = stats(r, lo)[0] if lo < 49.9 else None   # never reaches today's drawdown → n/a
        out[n] = dict(dcr=dcr, dd=dd, worst=wd, eq_dcr=eq_dcr)
    return out


def newest_export():
    t = None
    for pat in _W.PATTERNS:
        for f in _EG.glob_exports(pat):
            s = _EG.export_stamp(f)
            try:
                x = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
            except (OSError, ValueError):
                continue
            t = x if t is None or x > t else t
    return t


def report(frame=None, since_ms=None, state_path=STATE, now_ms=None, bars_fn=None, newest=None):
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 🎯👁 DIST_TP_SHADOW (DECISION_LOG {DL}, OBSERVE only) — DISTRIBUTION_SHORT at TP 5 % / 8 % vs today's TP, on real fills "
         "(1m walk from the real fill price from the next minute, the fill's own stop first, TP ≥ 0.05 % through, 24 h; modelled slip; no funding)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="DIST_TP_SHADOW")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    a = _W.load(sleeve=DISTRIBUTION_SHORT, cols=COLS) if frame is None else frame
    a = a[a.o >= pd.Timestamp(int(since_ms), unit="ms")] if len(a) else a
    try:
        with open(state_path) as fh:
            st = json.load(fh)
    except FileNotFoundError:
        st = {}
    except (OSError, ValueError):
        st = None
    if st is not None and not isinstance(st, dict):
        st = None
    rec = (st or {}).get(KEY)
    if rec is not None and not (isinstance(rec, dict) and {"run_at", "k", "arms"} <= set(rec)):
        return L + [f"⚠ frozen {KEY} record damaged — operator restore needed; not re-decided.", ""]
    bud, rows, par, stopped = LS.Budget(), [], [], False
    for r in a.itertuples():
        E, slip = _f(r.entry_price), _f(getattr(r, "entry_dist_model_slip_pct", None)) or 0.0
        tp0, sl = row_levels(r, _TH)                    # the stamps, else the caps — the levels the engine ran (services.distribution)
        es = entry_leg_slip(slip, getattr(r, "entry_depth_mode", None), _f(getattr(r, "entry_depth_walk_pct", None)))
        te = int(r.o.value // 1_000_000)
        bars = bars_fn(r.pair, te, now_ms) if bars_fn else fetch_bars(r.pair, te, now_ms, bud)
        stopped |= bars is None
        w0, v0, x0 = walk_x(bars, E, te, tp0, sl, slip, es) if bars else ("UNPRICED", np.nan, None)
        arms = [walk_x(bars, E, te, x, sl, slip, es) if (bars and tp0 and x < tp0) else (w0, v0, x0) for x in ARMS]
        vals = [v0] + [v for _, v, _ in arms]
        if any(v != v for v in vals):
            if bars is not None and now_ms > te + STALE_H * HOUR:
                L.append(f"  · {r.pair} {r.o:%m-%d %H:%M} → NOT_EVALUABLE (no complete price path) — out of the read")
                continue
            L.append(f"  · {r.pair} {r.o:%m-%d %H:%M} booked {_f(r.pnl_percentage) or 0:+.2f} → pending / unpriced (holds the read)")
            rows.append(None)
            continue
        if _f(r.pnl_percentage) is not None:
            par.append(abs(v0 - _f(r.pnl_percentage)))
        rows.append((te // DAY, v0, vals[1], vals[2], r.o, (x0 // DAY, arms[0][2] // DAY, arms[1][2] // DAY), tp0))
        L.append(f"  · {r.pair} {r.o:%m-%d %H:%M} booked {_f(r.pnl_percentage) or 0:+.2f} · today TP {tp0:g} {w0} {v0:+.2f} · "
                 f"TP 5 {arms[0][0]} {vals[1]:+.2f} (Δ {vals[1] - v0:+.2f}) · TP 8 {arms[1][0]} {vals[2]:+.2f} (Δ {vals[2] - v0:+.2f})")
    done = []
    for x in rows:                      # fill order; the read waits at the first unpriced fill
        if x is None:
            break
        done.append(x)
    if not len(a):
        L.append("  no new fills yet.")
    L.append(f"  cohort: CLOSED fills opened ≥ {pd.Timestamp(int(since_ms), unit='ms'):%Y-%m-%d %H:%M} UTC · priced {len(done)} on "
             f"{len({x[0] for x in done})} days" + (f" · parity today-walk vs booked: median |diff| {np.median(par):.2f}" if par else ""))
    k = prefix([(x[0],) for x in done])
    res = {}
    if k:
        pre = done[:k]
        for i, nm, X in ((2, "TP 5", 5.0), (3, "TP 8", 8.0)):
            sub = [x for x in pre if x[6] > X]           # a fill whose own TP is ≤ X is not a test of X (Δ ≡ 0) → out of that arm
            if len(sub) >= 2:
                sv, tx = arm_verdict([x[i] - x[1] for x in sub], [x[0] for x in sub])
                res[nm] = (sv, tx + f" · {len(sub)} of {k} fills had today's TP above {X:g} %")
            else:
                res[nm] = ("NO SHIP", f"only {len(sub)} of {k} fills had today's TP above {X:g} % — nothing to test")
    hold = []
    if stopped:
        hold.append("a kline fetch stopped this run")
    newest = newest_export() if newest is None else newest
    if k and (newest is None or newest < max(x[4] for x in done[:k]) + pd.Timedelta(hours=HOLD_H)):   # only fills up to the read's prefix matter
        hold.append("newest export < the last priced fill + 24 h (an earlier-opened fill could still be open)")
    if st is None:
        L.append("⚠ state file corrupt — operator restore needed; not frozen this run.")
    elif rec is None and res and not hold:
        pr = done[:k]
        rr = risk_read([(x[5], x[1], x[2], x[3]) for x in pr], int(min(min(x[5]) for x in pr)), int(max(max(x[5]) for x in pr)))
        rec = dict(run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"), k=k,
                   arms={n: dict(state=s, detail=t) for n, (s, t) in res.items()},
                   risk={n: {kk: (round(vv, 3) if vv is not None else None) for kk, vv in d.items()} for n, d in rr.items()})
        st[KEY] = rec
        try:
            tmp = state_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(st, fh, indent=1)
            os.replace(tmp, state_path)
        except OSError as e:
            L.append(f"⚠ state not saved ({e})")
    if rec:
        L.append(f"**Frozen read ({rec.get('run_at')}, first {rec.get('k')} fills):** " +
                 " · ".join(f"{n}: {d.get('state')} — {d.get('detail')}" for n, d in (rec.get("arms") or {}).items()))
        for n, d in (rec.get("risk") or {}).items():
            L.append(f"  risk at 0.61 × equity — {n}: DCR {d.get('dcr')} %/day · max DD {d.get('dd')} % · worst day {d.get('worst')} % · "
                     f"DCR at today's drawdown {d.get('eq_dcr') if d.get('eq_dcr') is not None else 'n/a'} %/day")
    elif res:
        L.append("Now: " + " · ".join(f"{n}: {s} — {t}" for n, (s, t) in res.items()))
        if hold:
            L.append("  ⏸ freezing deferred: " + " · ".join(hold) + " — re-checked next run.")
    else:
        L.append(f"Now: ⏳ COLLECTING — {len(done)}/{N_MIN} fills on {len({x[0] for x in done})}/{DAYS_MIN} days")
    L.append("  Evidence (DL 603, never counted): backtest 1,547 fills — same size TP 10 +0.93 vs 8 +0.81 vs 5 +0.60 %/fill; equal drawdown "
             "ALL P(better) 0.35–0.52 · HOLDOUT 0.75–0.81 (ranges include 0). A SHIP CANDIDATE is decided with the DIST size (pre-live list).")
    return L + [""]


if __name__ == "__main__":
    print("\n".join(report()))
