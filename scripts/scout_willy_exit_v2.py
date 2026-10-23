"""🎲²ᵇ WILLY EXIT V2b scout (DECISION_LOG 358 — the V2 floor REMOVED; observe-only, FROZEN revert gates, verdicts persisted).

History: DECISION_LOG 356 shipped WILLY exit V2 = a per-order take profit aimed just below the spike high (Order.frenzy_willy_tp_target_pct =
max(+1.25, min(+8, net at H × 0.99)), resting maker limit) + a profit FLOOR (+1.0 once the peak touched +1.25; stop and late TP skipped when
armed). DECISION_LOG 357 added ladder shadows of that floor. DECISION_LOG 358 (2026-10-23, operator) REMOVED the floor from the engine: the
live WILLY / WILLY_HT exit is now V2b = TP at the stored target · stop −4 throughout · the 30-min late TP +0.25 for EVERY trade (no armed
exception) · 90-min cap. Evidence: reports/WILLY_HIGHTARGET_2026-10-23.md variant H1_k1 (backtest 2026 Δ vs the +1.25 exit +0.10
[−0.03, +0.24], master +0.24, live +0.07) vs the floor (backtest −0.03 at F 1.0 / +0.02 at F 0.5, live negative at every F).

ERAS (from the orders exports; dedup (opened_at[:19], pair, direction), CLOSED wins; operator-closed out via scout_gatekit):
  V2 (floor) fills   — V2-stamped (target column set) and CLOSED before the DL 358 deploy (INIT 10-23) → shown APART, never counted.
  transition fills   — V2-stamped, opened before the DL 358 deploy + 10 min but closed after the deploy → listed by count, never counted.
  V2b counted cohort — opened ≥ the DL 358 deploy + 10 min (scout_gatekit.dl_deploy_ms(358, lag_min=10)).
  RUNNER fill        — a V2b fill whose stored target is > +1.25 (only those can differ from the old exit; no arming anymore).

Per fill (same tick → 1m hybrid path as the other WILLY scouts, scout_frenzy_willy_exit.load_path; caches + the budgeted public fetch helper
of scout_willy_timecap.ensure_data only):
  LIVE    = the booked pnl_percentage.
  PARITY  = re-walk of the LIVE V2b exit with the fill's stamped target (walk_v2, no floor) vs booked: < 90 % of priced fills within 0.15 →
            "no verdict (parity)" (nothing is frozen on a failed parity).
  SHADOWS on every RUNNER fill (same path, same convention; Δ = shadow − the live V2b re-walk, paired):
    (a) TODAY  — the pinned pre-356 exit: +1.25 resting maker TP / −4 / late +0.25 after 30 min / 90 cap (scout_willy_timecap.evaluate_late
                 (book="live")) — the GATE shadow, never read from config.
    (b) FLOOR10 — the DL 356 rule: flat floor +1.0 once +1.25 is touched (no stop / no late TP while armed; cap closes).
    (c) the four DL 357 ladders (floor = f(peak of the EARLIER events), armed at +1.25):
          STEPS    peak < 2 → +1.0 · < 3 → +1.5 · < 4 → +2.25 · < 5 → +3.0 · ≥ 5 → peak − 1.5
          TRAIL10  max(+1.0, peak − 1.0)      TRAIL15  max(+1.0, peak − 1.5)      HALF  max(+1.0, 50 % of peak)
    (d) NOLATE — no late TP: ride the target until the 90-min cap, stop −4.
    A FLOOR exit in (b) / (c) pays FLOOR_SLIP 0.05 (a taker market close; V2b never floors, so every shadow floor exit replaces a maker /
    stop / cap exit — the DL 357 convention).

🔒 FROZEN gates (decided ONCE each, persisted in reports/.willy_exit_v2b_gate.json under V2B_* keys — a fresh file, so nothing from the
DL 356 file can leak; operator decides — nothing automatic). Δ = live (booked) − shadow (a):
  V2B_EARLY — the first 10 runner fills: avg(live) < avg(shadow a) − 0.15 → ALERT (review)
  V2B_MAIN  — the shortest prefix of ≥ 15 runner fills on ≥ 6 UTC days: Σ Δ < 0 → REVERT to +1.25 (frenzy_willy_high_target_enabled
              false) · else PASS
  V2B_FINAL — the first 25 runner fills: Σ Δ < 0 → REVERT · else PASS
🔒 FROZEN read per shadow (b)(c)(d) (V2B_SH_<name>): at the shortest prefix of ≥ 15 runner fills on ≥ 6 UTC days → CANDIDATE (operator
decides an engine change) iff avg Δ (shadow − live re-walk) ≥ +0.10 %/fill ∧ no single fill carries ≥ 50 % of the positive Δ; else KEEP.
A runner fill without a priced path / with an unresolved walk holds the freeze of any read whose sample it belongs to."""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import willy_exit_era as WEra      # noqa: E402

DL, DL_V2 = 358, 356
LAG_MIN = 10
_ROOT = os.path.dirname(_HERE)
STATE = os.path.join(_ROOT, "reports", ".willy_exit_v2b_gate.json")   # fresh file (DL 358) — the DL 356 file is never read
KEY = "V2B_"                                                              # every persisted key carries it
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
STRATEGIES = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "close_reason",
        "peak_pnl", "entry_btc_1h_slope") + WEra.V2_COLS
# the PINNED pre-356 exit (shadow a) — DECISION_LOG 281 / 297; never read from config
OLD_TP, OLD_STOP, OLD_CAP, (OLD_LATE_MIN, OLD_LATE_TP) = WEra.OLD_TP, WEra.OLD_STOP, WEra.OLD_HOLD, WEra.OLD_LATE
ARM, FLOOR = 1.25, 1.0                    # the REMOVED DL 356 floor — shadow (b) / the V2-era re-walk only (never the live exit)
FLOOR_SLIP = 0.05                         # a shadow floor exit = a taker market close (DL 357 convention)
PARITY_TOL, PARITY_MIN = 0.15, 0.90
EARLY_N, EARLY_GAP = 10, 0.15
MAIN_N, MAIN_DAYS = 15, 6
FINAL_N = 25
MIN = 60_000
# 🪜 ladder shadows (DL 357) — frozen
LADDERS = {
    "STEPS": lambda p: 1.0 if p < 2 else 1.5 if p < 3 else 2.25 if p < 4 else 3.0 if p < 5 else p - 1.5,
    "TRAIL10": lambda p: max(1.0, p - 1.0),
    "TRAIL15": lambda p: max(1.0, p - 1.5),
    "HALF": lambda p: max(1.0, 0.5 * p),
}
SHADOWS = ("FLOOR10",) + tuple(LADDERS) + ("NOLATE",)    # (b) · (c) · (d)
SH_N, SH_DAYS, SH_MIN_GAIN, SH_SHARE = 15, 6, 0.10, 0.50
LIST_CAP = 20


# ─────────────────────────── walkers (net % path: t, hi, lo, cl, op, bar, end_ms, te) ───────────────────────────
def walk_v2(P, tp, sl=OLD_STOP, cap=OLD_CAP, late=OLD_LATE_MIN, ztp=OLD_LATE_TP, arm=ARM, floor=None, floor_fn=None):
    """a WILLY exit on a net path → dict(why TP / LATE_TP / STOP / FLOOR / CAP / UNRES, pct, armed).
    Default (floor None, floor_fn None) = the LIVE V2b exit (DL 358): scout_willy_timecap.evaluate_late(book="live") exactly with the
    stored target (stop first on an event touching both lines; resting TP books the line; a late line already through at the first late
    event books that print / bar open). late=None → no late TP (shadow d).
    SHADOW TOOL ONLY (the engine has no floor since DL 358): floor (flat) or floor_fn(peak of the EARLIER events) — exists only when tp > arm;
    it arms on the first event with hi ≥ arm (not judged on the arming event itself); armed: an event touching the floor → FLOOR (bar:
    min(floor, open); tick: the print) — checked before the TP (conservative); no stop, no late line; the cap still closes."""
    te = int(P["te"])
    t, hi, lo, op, bar = P["t"], P["hi"], P["lo"], P["op"], P["bar"]
    n = int(np.searchsorted(t, te + cap * MIN, side="right"))
    has_floor = (floor is not None or floor_fn is not None) and float(tp) > float(arm) + 1e-9
    use_late = late is not None and ztp is not None and float(ztp) < float(tp)
    i0 = None
    if use_late:
        idx = np.flatnonzero(t[:n] > te + late * MIN)
        i0 = int(idx[0]) if len(idx) else None
    armed = False
    pk = float("-inf")                    # 🪜 (357) peak of the EARLIER events — the ladder floor never uses the event that sets the peak
    for i in range(n):
        b, o = bool(bar[i]), float(op[i]) if np.isfinite(op[i]) else float("nan")
        if not armed:
            lt = use_late and t[i] > te + late * MIN
            line = float(ztp) if lt else float(tp)
            if sl and lo[i] <= -sl:
                v = (min(-sl, o) if np.isfinite(o) else -sl) if b else float(lo[i])
                return dict(why="STOP", pct=float(v), armed=False)
            if hi[i] >= line:
                if not lt or i != i0:
                    v = line
                elif b:
                    v = o if (np.isfinite(o) and o >= line) else line
                else:
                    v = float(hi[i])
                return dict(why=("LATE_TP" if lt else "TP"), pct=float(v), armed=False)
            if has_floor and hi[i] >= arm:
                armed = True
            pk = max(pk, float(hi[i]))
            continue
        fl = float(floor_fn(pk)) if floor_fn is not None else float(floor)
        if lo[i] <= fl:
            v = (min(fl, o) if np.isfinite(o) else fl) if b else float(lo[i])
            return dict(why="FLOOR", pct=float(v), armed=True)
        if hi[i] >= tp:
            return dict(why="TP", pct=float(tp), armed=True)
        pk = max(pk, float(hi[i]))
    if n == 0 or int(P["end_ms"]) < te + cap * MIN:
        return dict(why="UNRES", pct=float("nan"), armed=armed)
    return dict(why="CAP", pct=float(P["cl"][n - 1]), armed=armed)


def shadow_today(P):
    """(a) the PINNED old exit (+1.25 resting / −4 / late +0.25 @30 / 90) — scout_willy_timecap.evaluate_late(book='live')."""
    import scout_willy_timecap as WT
    r = WT.evaluate_late(P, OLD_TP, OLD_STOP, OLD_CAP, OLD_LATE_MIN, OLD_LATE_TP, book="live")
    return dict(why=r["why"], pct=float(r["pct"]) if r.get("pct") is not None else float("nan"))


def shadow_walk(P, tp, name):
    """(b) FLOOR10 · (c) a DL 357 ladder · (d) NOLATE → booked % (a FLOOR exit pays FLOOR_SLIP; UNRES → nan)."""
    if name == "NOLATE":
        w = walk_v2(P, tp, late=None)
    elif name == "FLOOR10":
        w = walk_v2(P, tp, floor=FLOOR)
    else:
        w = walk_v2(P, tp, floor_fn=LADDERS[name])
    return float(w["pct"]) - (FLOOR_SLIP if w["why"] == "FLOOR" else 0.0)


# ─────────────────────────── cohort ───────────────────────────
def read_exports(pattern=EXPORT_GLOB):
    fr = []
    for f in _EG.glob_exports(pattern):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            fr.append(d.assign(_m=os.path.getmtime(f), _batch=os.path.basename(f)))
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame(columns=list(COLS) + ["_m", "_batch"])


def dedup(raw):
    """dedup (opened_at[:19], pair, direction), a CLOSED row beats an OPEN one, newest export wins among equals; the fill's batch = the
    OLDEST export it appears in. WILLY + WILLY_HT LONG only."""
    o = raw.reindex(columns=list(dict.fromkeys(list(COLS) + list(raw.columns))))
    if not len(o):
        return o.assign(_k=pd.Series(dtype=object), batch=pd.Series(dtype=object))
    o = o[o.entry_strategy.astype(str).isin(STRATEGIES) & (o.direction.astype(str) == "LONG")].copy()
    o["_k"] = o.opened_at.astype(str).str[:19]
    if "_m" not in o:
        o["_m"] = 0.0
    if "_batch" not in o:
        o["_batch"] = "?"
    first = o.sort_values("_m", kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="first").set_index(["_k", "pair", "direction"])["_batch"]
    o["_c"] = o.status.astype(str).str.upper().eq("CLOSED")
    o = o.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "direction"], keep="last")
    o["batch"] = [first.get((k, p, d), "?") for k, p, d in zip(o._k, o.pair, o.direction)]
    return o


def cohort(raw, floor_ms):
    """→ dict(kept = counted V2b fills (opened ≥ floor_ms, operator-closed out) · ox = operator-closed · allc = every closed WILLY row
    (era line) · v2 = V2 (floor) era fills (stamped, closed before the DL 358 deploy = floor_ms − LAG_MIN) · trans = stamped fills opened
    before floor_ms but closed at / after the deploy)."""
    o = dedup(raw)
    c = o[o.status.astype(str).str.upper() == "CLOSED"].copy() if len(o) else o
    empty = c.iloc[0:0] if len(c) else c
    out = dict(kept=empty, ox=empty, allc=c, v2=empty, trans=empty)
    if not len(c):
        return out
    c["ts"] = pd.to_datetime(c._k, format="ISO8601", errors="coerce")
    c = c[c.ts.notna()].copy()
    cl = pd.to_datetime(c.closed_at.astype(str).str[:19], format="ISO8601", errors="coerce") if "closed_at" in c else pd.Series(pd.NaT, index=c.index)
    c["cts"] = cl.where(cl.notna(), c.ts)
    out["allc"] = c
    if floor_ms is None:
        out.update(kept=c.iloc[0:0], ox=c.iloc[0:0], v2=c.iloc[0:0], trans=c.iloc[0:0])
        return out
    floor_ts = pd.Timestamp(int(floor_ms), unit="ms")
    cut_ts = pd.Timestamp(int(floor_ms) - LAG_MIN * MIN, unit="ms")
    pre = c[c.ts < floor_ts]
    v2p, _ = WEra.masks(pre)
    old_v2 = pre[v2p.values & (pre.cts < cut_ts).values]
    trans = pre[v2p.values & (pre.cts >= cut_ts).values]
    k = c[c.ts >= floor_ts]
    kept, ox = _GK.split_operator_closed(k)
    out.update(kept=kept.sort_values(["ts", "pair"], kind="stable").reset_index(drop=True), ox=ox,
               v2=old_v2.sort_values(["ts", "pair"], kind="stable").reset_index(drop=True), trans=trans)
    return out


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def is_runner(r):
    """a RUNNER fill: its stored target is above the old +1.25 TP (the only fills whose V2b exit can differ from the old exit)."""
    tp = _f(r.get(WEra.TP_COL))
    return tp is not None and tp > OLD_TP + WEra.EPS


def btc_side(v):
    x = _f(v)
    return "unread" if x is None else ("falling" if x <= 0 else "rising")


def _price_cols():
    return (("live", "shadow", "shadow_why", "v2b", "v2b_why", "v2f", "v2f_why", "runner", "src", "side", "delta")
            + tuple(x for k_ in SHADOWS for x in (f"sh_{k_}", f"d_{k_}")))


def price(c, path_fn=None, fetch=False, now_ms=None):
    """→ c with the shadow (a) / V2b re-walk / V2-floor re-walk / shadow (b)(c)(d) columns. path_fn(row) → P or None (default: the hybrid
    tick → 1m loader on the shared caches)."""
    c = c.copy()
    if not len(c):
        for k in _price_cols():
            c[k] = pd.Series(dtype=object)
        return c
    c["live"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c["side"] = [btc_side(v) for v in c.get("entry_btc_1h_slope", pd.Series([None] * len(c)))]
    c["runner"] = [is_runner(r) for r in c.to_dict("records")]
    if path_fn is None:
        import scout_frenzy_willy_exit as FX
        import scout_willy_timecap as WT
        tcache = {}
        E = pd.to_numeric(c.entry_price, errors="coerce")
        te = pd.to_datetime(c.opened_at.astype(str), format="ISO8601", errors="coerce").values.astype("datetime64[ms]").astype("int64")
        c["E"], c["te"] = E.values, te
        if fetch:
            try:
                fr = c.assign(src=None)[["ts", "pair", "te", "src"]]
                WT.ensure_data(fr, OLD_CAP, now_ms or int(pd.Timestamp.now("UTC").value // 1_000_000), fetch=True)
            except Exception as e:
                print(f"  (WILLY_EXIT_V2b: fetch pass skipped: {str(e)[:80]})")

        def path_fn(r):
            if not (r["E"] and r["E"] > 0):
                return None
            P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), max(OLD_CAP, 120), tcache)
            P["_src"] = FX.path_src(P)
            return P
    cols = {k: [] for k in ("shadow", "shadow_why", "v2b", "v2b_why", "v2f", "v2f_why", "src")}
    sh = {k: [] for k in SHADOWS}
    for r in c.to_dict("records"):
        try:
            P = path_fn(r)
        except Exception:
            P = None
        if P is None or not len(P.get("t", [])):
            for k, v in (("shadow", np.nan), ("shadow_why", "UNPRICED"), ("v2b", np.nan), ("v2b_why", "UNPRICED"), ("v2f", np.nan),
                         ("v2f_why", "UNPRICED"), ("src", None)):
                cols[k].append(v)
            for k in SHADOWS:
                sh[k].append(np.nan)
            continue
        tp = _f(r.get(WEra.TP_COL)) or OLD_TP
        s = shadow_today(P)
        w = walk_v2(P, tp)                       # the LIVE V2b exit (DL 358)
        wf = walk_v2(P, tp, floor=FLOOR)         # the V2 (DL 356) exit — parity of the V2-era fills only
        for k, v in (("shadow", s["pct"]), ("shadow_why", s["why"]), ("v2b", w["pct"]), ("v2b_why", w["why"]), ("v2f", wf["pct"]),
                     ("v2f_why", wf["why"]), ("src", P.get("_src", "path"))):
            cols[k].append(v)
        for k in SHADOWS:
            sh[k].append(shadow_walk(P, tp, k))
    for k, v in cols.items():
        c[k] = v
    c["delta"] = c.live - c.shadow
    for k in SHADOWS:
        c[f"sh_{k}"] = sh[k]
        c[f"d_{k}"] = pd.to_numeric(c[f"sh_{k}"], errors="coerce") - pd.to_numeric(c.v2b, errors="coerce")
    return c


# ─────────────────────────── gates ───────────────────────────
def parity(c, col="v2b", label="V2b re-walk"):
    """→ (ok, text): share of priced fills whose re-walk is within PARITY_TOL of the booked %."""
    g = c[c[col].notna() & c.live.notna()] if len(c) else c
    if not len(g):
        return True, f"parity ({label} vs booked): no priced fill yet"
    within = (pd.to_numeric(g[col]) - g.live).abs() <= PARITY_TOL
    share = float(within.mean())
    ok = share >= PARITY_MIN
    return ok, (f"parity ({label} vs booked, ±{PARITY_TOL}): {int(within.sum())}/{len(g)} = {share * 100:.0f} %"
                + ("" if ok else f" < {PARITY_MIN * 100:.0f} % → no verdict (parity)"))


def _days(g):
    return g.ts.dt.strftime("%Y-%m-%d").nunique() if len(g) else 0


def _runners(c):
    return c[c.runner.astype(bool)].sort_values(["ts", "pair"], kind="stable") if len(c) else c


def _prefix(a, n, days):
    for k in range(len(a)):
        s = a.iloc[:k + 1]
        if k + 1 >= n and _days(s) >= days:
            return s
    return None


def reads(c):
    """the three FROZEN gate reads on the RUNNER fills (in open order) → {name: (verdict, text, sample_complete)}."""
    a = _runners(c)
    out = {}

    def _sample_ok(s):
        return bool(len(s)) and bool(s.shadow.notna().all())

    e = a.head(EARLY_N)
    if len(e) >= EARLY_N:
        lv, sv = e.live.mean(), e.shadow.mean()
        if not _sample_ok(e):
            out[KEY + "EARLY"] = ("HOLD", f"first {EARLY_N} runner fills: {int(e.shadow.isna().sum())} unpriced — held", False)
        else:
            bad = lv < sv - EARLY_GAP
            out[KEY + "EARLY"] = (("ALERT" if bad else "OK"), f"first {EARLY_N} runner fills: live {lv:+.3f} vs shadow {sv:+.3f} %/fill (alert if < "
                                  f"shadow − {EARLY_GAP}) → " + ("⚠ ALERT: review WILLY exit V2b" if bad else "no alert"), True)
    else:
        out[KEY + "EARLY"] = ("COLLECTING", f"early alert: {len(a)}/{EARLY_N} runner fills", False)
    m = _prefix(a, MAIN_N, MAIN_DAYS)
    if m is None:
        out[KEY + "MAIN"] = ("COLLECTING", f"main read: {len(a)}/{MAIN_N} runner fills on {_days(a)}/{MAIN_DAYS} UTC days", False)
    elif not _sample_ok(m):
        out[KEY + "MAIN"] = ("HOLD", f"main read sample ({len(m)}): {int(m.shadow.isna().sum())} unpriced — held", False)
    else:
        sd = float(m.delta.sum())
        out[KEY + "MAIN"] = (("REVERT" if sd < 0 else "PASS"), f"main read ({len(m)} runner fills on {_days(m)} days): Σ(live − shadow) {sd:+.2f} pts → "
                             + ("🛑 REVERT to +1.25 (frenzy_willy_high_target_enabled false)" if sd < 0 else "PASS (keep V2b)"), True)
    f = a.head(FINAL_N)
    if len(f) < FINAL_N:
        out[KEY + "FINAL"] = ("COLLECTING", f"final read: {len(a)}/{FINAL_N} runner fills", False)
    elif not _sample_ok(f):
        out[KEY + "FINAL"] = ("HOLD", f"final read sample: {int(f.shadow.isna().sum())} unpriced — held", False)
    else:
        sd = float(f.delta.sum())
        out[KEY + "FINAL"] = (("REVERT" if sd < 0 else "PASS"), f"final read (first {FINAL_N} runner fills): Σ(live − shadow) {sd:+.2f} pts → "
                              + ("🛑 REVERT to +1.25 (frenzy_willy_high_target_enabled false)" if sd < 0 else "PASS (keep V2b)"), True)
    return out


def shadow_reads(c):
    """per shadow (b)(c)(d): the frozen read on the runner fills (open order) → {V2B_SH_<name>: (verdict, text, complete)}."""
    a = _runners(c)
    m = _prefix(a, SH_N, SH_DAYS)
    out = {}
    for k in SHADOWS:
        name = f"{KEY}SH_{k}"
        if m is None:
            out[name] = ("COLLECTING", f"{len(a)}/{SH_N} runner fills on {_days(a)}/{SH_DAYS} days", False)
            continue
        d = pd.to_numeric(m[f"d_{k}"], errors="coerce")
        if d.isna().any():
            nv = int(pd.to_numeric(m.v2b, errors="coerce").isna().sum())
            out[name] = ("HOLD", f"sample ({len(m)}): {nv} unpriced / {int(d.isna().sum()) - nv} unresolved shadow walks — held", False)
            continue
        gain = d.clip(lower=0)
        share = float(gain.max() / gain.sum()) if gain.sum() > 0 else 1.0
        ok = float(d.mean()) >= SH_MIN_GAIN and share < SH_SHARE
        out[name] = (("CANDIDATE" if ok else "KEEP"),
                     f"{len(m)} runner fills on {_days(m)} days: avg Δ (shadow − live V2b) {d.mean():+.3f} %/fill · top fill {share:.0%} of the gain → "
                     + (f"CANDIDATE: operator decides an engine change to {k}" if ok else "keep V2b"), True)
    return out


def frozen(state_path, name, v, txt, complete, allow):
    """🔒 each read decided ONCE: the first complete non-COLLECTING verdict (with parity OK) is persisted under its name. → (v, txt)."""
    st = {}
    try:
        if state_path and os.path.exists(state_path):
            with open(state_path) as fh:
                st = json.load(fh) or {}
    except Exception:
        st = {}
    if name in st:
        return st[name]["verdict"], st[name]["text"] + f" (frozen {str(st[name].get('at', ''))[:16]} UTC)"
    if state_path and complete and allow and v not in ("COLLECTING", "HOLD"):
        try:
            st[name] = {"verdict": v, "text": txt, "at": pd.Timestamp.now("UTC").isoformat()}
            with open(state_path, "w") as fh:
                json.dump(st, fh, indent=1)
        except Exception:
            pass
    return v, txt


# ─────────────────────────── report ───────────────────────────
def _avg(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    return f"{s.mean():+.3f}" if len(s) else "n/a"


def _fmt(v):
    x = _f(v)
    return "n/a" if x is None else f"{x:+.2f}"


def v2_era_lines(v):
    """the V2 (floor, DL 356) fills — closed before the DL 358 deploy: shown APART, never in a gate or a shadow read."""
    if not len(v):
        return ["V2 (floor, DL 356) era fills — closed before the DL 358 deploy: 0"]
    rs = v.close_reason.astype(str)
    L = [f"V2 (floor, DL 356) era fills — closed before the DL 358 deploy, shown APART (never counted): {len(v)} · "
         f"floor exits (WILLY_FLOOR) {int(rs.str.endswith(WEra.FLOOR_REASON).sum())} · booked avg {_avg(v.live)} · "
         f"what V2b would have booked {_avg(v.v2b)} · today's +1.25 exit {_avg(v.shadow)}"]
    if v.v2f.notna().any():
        L.append("  " + parity(v, col="v2f", label="V2-floor re-walk")[1].replace(" → no verdict (parity)", " (informational — no read uses this era)"))
    L.append("  per fill: " + " · ".join(
        f"{r.pair} {str(r.opened_at)[5:16]} target {_fmt(getattr(r, WEra.TP_COL))} booked {_fmt(r.live)} {r.close_reason} | V2b {_fmt(r.v2b)} · "
        f"+1.25 exit {_fmt(r.shadow)}" for r in v.tail(LIST_CAP).itertuples()))
    return L


def shadow_ledger(c):
    a = c[c.runner.astype(bool) & c.v2b.notna()] if len(c) else c
    if not len(a):
        return ["Shadow lines: 0 priced runner fills"]
    L = [f"Shadow lines on {len(a)} priced runner fills (Δ = shadow − the live V2b re-walk, same path; booked live avg {_avg(a.live)} · "
         f"V2b re-walk avg {_avg(a.v2b)}):",
         f"  (a) TODAY +1.25 exit (the gate shadow): avg {_avg(a.shadow)} · Σ(live − shadow) {pd.to_numeric(a.delta, errors='coerce').sum():+.2f} · "
         f"Σ(shadow − re-walk) {(pd.to_numeric(a.shadow, errors='coerce') - pd.to_numeric(a.v2b, errors='coerce')).sum():+.2f}"]
    tag = {"FLOOR10": "(b)", "NOLATE": "(d)"}
    for k in SHADOWS:
        d = pd.to_numeric(a[f"d_{k}"], errors="coerce")
        L.append(f"  {tag.get(k, '(c)')} {k}: avg {_avg(a[f'sh_{k}'])} · Σ Δ {d.sum():+.2f} · better {int((d > 1e-9).sum())} / worse {int((d < -1e-9).sum())}")
    L.append(f"  per fill (last {LIST_CAP}): " + " · ".join(
        f"{r.pair} {str(r.opened_at)[5:16]} T {_fmt(getattr(r, WEra.TP_COL))} live {_fmt(r.live)} V2b {_fmt(r.v2b)} | a {_fmt(r.shadow)} "
        + " ".join(f"{k} {_fmt(getattr(r, f'sh_{k}'))}" for k in SHADOWS) for r in a.tail(LIST_CAP).itertuples()))
    return L


def batch_table(c):
    if not len(c):
        return []
    g = c[c.shadow.notna()]
    L = ["| batch | BTC side | N | runner | before: shadow (today's exit) %/fill | after: live V2b %/fill | Δ |", "|---|---|---|---|---|---|---|"]
    for (b, s), x in g.groupby(["batch", "side"], sort=True):
        L.append(f"| {b} | {s} | {len(x)} | {int(x.runner.sum())} | {_avg(x.shadow)} | {_avg(x.live)} | {x.delta.mean():+.3f} |")
    for s, x in g.groupby("side", sort=True):
        L.append(f"| ALL | {s} | {len(x)} | {int(x.runner.sum())} | {_avg(x.shadow)} | {_avg(x.live)} | {x.delta.mean():+.3f} |")
    L.append(f"| ALL | all | {len(g)} | {int(g.runner.sum())} | {_avg(g.shadow)} | {_avg(g.live)} | {g.delta.mean():+.3f} |")
    return L


def run(now_ms=None, paths=None, raw=None, floor_ms="auto", path_fn=None, fetch=True, state_path=STATE):
    L = [f"## 🎲²ᵇ WILLY_EXIT_V2b — TP at the spike high, no floor (DECISION_LOG {DL}; frozen gates on runner fills; shadow = the pinned "
         f"+1.25 / −4 / late 0.25@30 / 90 exit)", ""]
    if raw is None:
        raw = read_exports(paths or EXPORT_GLOB)
    if floor_ms == "auto":
        floor_ms = _GK.dl_deploy_ms(DL, lag_min=LAG_MIN, label="WILLY_EXIT_V2b")
    co = cohort(raw, floor_ms)
    L.append(WEra.era_line(co["allc"], "WILLY + WILLY_HT, all closed in the exports"))
    if floor_ms is None:
        L += [f"NOT evaluable yet — no DECISION_LOG {DL} commit found (the counted cohort starts at its deploy + {LAG_MIN} min; the V2 floor "
              f"era is split at the deploy).", ""]
        return L
    c = co["kept"]
    L.append(f"Cohort: {len(c)} closed WILLY / WILLY_HT fills since the DL {DL} deploy + {LAG_MIN} min · " + _GK.operator_closed_line(co["ox"])
             + f" · transition (opened before the counted floor = the DL {DL} deploy + 10 min, closed after the deploy — not counted): {len(co['trans'])}")
    both = pd.concat([c.assign(_era="V2B"), co["v2"].assign(_era="V2")], ignore_index=True) if (len(c) or len(co["v2"])) else c.assign(_era="V2B")
    both = price(both, path_fn=path_fn, fetch=fetch, now_ms=now_ms)
    c = both[both._era == "V2B"].reset_index(drop=True) if len(both) else both
    v = both[both._era == "V2"].reset_index(drop=True) if len(both) else both
    L += v2_era_lines(v)
    n_un = int(c.shadow.isna().sum()) if len(c) else 0
    L.append(f"Priced: {len(c) - n_un} · unpriced (no path yet): {n_un} · runner fills (target > +{OLD_TP}): {int(c.runner.sum()) if len(c) else 0}")
    pok, ptxt = parity(c)
    L.append(ptxt)
    L += shadow_ledger(c)
    L += [""] + batch_table(c) + [""]
    L.append("🔒 FROZEN gates + shadow reads (operator decides; persisted once decided):")
    for name, (vv, txt, complete) in list(reads(c).items()) + list(shadow_reads(c).items()):
        if not pok and complete:   # (review) a verdict frozen earlier still shows; the parity guard only blocks NEW freezes
            fv, ft = frozen(state_path, name, vv, txt, False, False)
            L.append(f"  {name}: {ft}" if "(frozen " in ft else f"  {name}: no verdict (parity) — {txt}")
            continue
        fv, ft = frozen(state_path, name, vv, txt, complete, pok)
        L.append(f"  {name}: {ft}")
    L.append("")
    return L


# ─────────────────────────── selftest (hermetic) ───────────────────────────
def _P(te, ev, cap=OLD_CAP, bar=False):
    """synthetic net path: ev = [(minutes after te, hi, lo)] — ticks unless bar=True."""
    t = np.array([te + int(m * MIN) for m, _, _ in ev], dtype="int64")
    hi = np.array([h for _, h, _ in ev], float)
    lo = np.array([lw for _, _, lw in ev], float)
    return dict(t=t, hi=hi, lo=lo, cl=(hi + lo) / 2, op=np.full(len(t), np.nan), bar=np.full(len(t), bar), end_ms=int(te + (cap + 5) * MIN), te=te)


def selftest():
    te = 1_760_000_000_000
    up = _P(te, [(1, 0.5, 0.5), (5, 1.3, 1.3), (8, 1.1, 1.1), (12, 3.6, 3.6)])
    assert walk_v2(up, 3.5)["why"] == "TP" and walk_v2(up, 3.5)["pct"] == 3.5
    assert shadow_today(up)["why"] == "TP" and abs(shadow_today(up)["pct"] - 1.25) < 1e-9
    dip = _P(te, [(5, 1.3, 1.3), (8, 0.9, 0.9), (12, 3.6, 3.6)])
    assert walk_v2(dip, 3.5)["why"] == "TP"                                   # V2b: no floor — rides the dip to the target
    assert walk_v2(dip, 3.5, floor=FLOOR)["why"] == "FLOOR"                   # the DL 356 floor (shadow b only)
    assert abs(shadow_walk(dip, 3.5, "FLOOR10") - (0.9 - FLOOR_SLIP)) < 1e-9
    same = _P(te, [(5, 0.7, 0.7), (35, 0.3, 0.3)])
    assert walk_v2(same, 1.25)["why"] == shadow_today(same)["why"] == "LATE_TP"
    lt = _P(te, [(5, 2.0, 2.0), (35, 0.3, 0.3), (60, 3.6, 3.6)])
    assert walk_v2(lt, 3.5)["why"] == "LATE_TP" and walk_v2(lt, 3.5, late=None)["why"] == "TP"   # (d) NOLATE rides past the late line
    print("selftest WILLY_EXIT_V2b OK")
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run(fetch="--no-fetch" not in sys.argv)))
