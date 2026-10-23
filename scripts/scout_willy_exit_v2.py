"""🎲² WILLY EXIT V2 scout (DECISION_LOG 356 — observe-only; FROZEN revert gates, verdict persisted).

What V2 changed (services/frenzy.py): a FRENZY_WILLY / FRENZY_WILLY_HT fill aims its take profit just below the episode's spike high
(Order.frenzy_willy_tp_target_pct = max(+1.25, min(+8, net at H × 0.99)), resting maker limit) and, when that target is ABOVE the floor arm
(+1.25), a profit floor arms once the peak reaches +1.25: then it closes at net ≤ +1.0 (WILLY_FLOOR), with no stop and no late-TP
downgrade; the 90-min cap still closes. A target of exactly +1.25 runs the old exit. The stop is unchanged (−4; the study's BTC-falling −2.5
stop was NOT shipped — the falling side is governed by the WILLY_BTC1H gate).

Cohort: CLOSED FRENZY_WILLY + FRENZY_WILLY_HT LONG fills from the orders exports (export_glob; dedup (opened_at[:19], pair, direction), CLOSED
wins; operator-closed out via scout_gatekit) opened ≥ the DL 356 deploy (scout_gatekit.dl_deploy_ms(356, lag_min=10)).

Per fill (same tick → 1m hybrid path as the other WILLY scouts, scout_frenzy_willy_exit.load_path; caches + the budgeted public fetch helper
of scout_willy_timecap.ensure_data only):
  SHADOW = TODAY's old exit on that path — +1.25 net resting maker TP / −4 / late +0.25 after 30 min / 90 cap
           (scout_willy_timecap.evaluate_late(book="live"), net convention) — PINNED here, never read from config.
  LIVE   = the booked pnl_percentage.
  PARITY = re-walk of the LIVE V2 exit with the fill's stamped target (walk_v2) vs booked: < 90 % of priced fills within 0.15 →
           "no verdict (parity)" (nothing is frozen on a failed parity).

Ledger A (target + floor): the ARMED fills (runner target > the arm and peak ≥ the arm: willy_floor_armed_at, or peak_pnl ≥ the arm) —
Σ(live − shadow) split into runners (TP above +1.25) · floor exits (WILLY_FLOOR) · other (cap / late). Per-batch before (shadow) → after
(live) table, split by BTC side at entry (entry_btc_1h_slope ≤ 0 = falling — analytics only).

🔒 FROZEN gates (decided ONCE each, persisted in reports/.willy_exit_v2_gate.json; operator decides — nothing automatic):
  EARLY  — the first 10 armed fills: avg(live) < avg(shadow) − 0.15 → ALERT (review)
  MAIN   — the shortest prefix of ≥ 15 armed fills on ≥ 6 UTC days: Σ(live − shadow) < 0 → REVERT to +1.25 (frenzy_willy_high_target_enabled
           false) · else PASS
  FINAL  — the first 25 armed fills: Σ(live − shadow) < 0 → REVERT · else PASS
An armed fill without a priced path holds the freeze of any read whose sample it belongs to.

🪜 LADDER SHADOWS (operator 2026-10-23 after INIT 10-23: peak +2.85 → floored at +1.06, then ran to +11.5 %; DECISION_LOG 357 — observe-only):
every priced V2 fill is ALSO re-walked with a rising floor (only runner targets can differ; the rest show Δ = 0) instead of the flat +1.0 (same path, same arm +1.25, the floor at
each event = f(peak of the EARLIER events) — never the event that sets the peak):
  STEPS    peak < 2 → +1.0 · < 3 → +1.5 · < 4 → +2.25 · < 5 → +3.0 · ≥ 5 → peak − 1.5
  TRAIL10  max(+1.0, peak − 1.0)      TRAIL15  max(+1.0, peak − 1.5)      HALF  max(+1.0, 50 % of peak)
Δ = ladder walk − the flat V2 re-walk of the SAME fill (paired, same convention; a ladder FLOOR exit where the flat walk did not floor pays
LADDER_FLOOR_SLIP 0.05 — the taker market close vs the maker TP it replaces (deep review); fills that never armed show Δ = 0 — a live-armed fill whose
re-walk never arms counts in the 15 with Δ = 0).
🔒 FROZEN read per ladder (decided once, persisted): at the shortest prefix of ≥ 15 armed fills on ≥ 6 UTC days → CANDIDATE (operator decides
an engine change) iff avg Δ ≥ +0.10 %/fill ∧ no single fill carries ≥ 50 % of the positive Δ; else KEEP_FLAT. Master evidence (36 fills,
10-23): flat +0.45 vs ladders +0.30 … +0.34 — the ladder sells runners that breathe on the way up (CFX, CAP)."""
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

DL = 356
_ROOT = os.path.dirname(_HERE)
STATE = os.path.join(_ROOT, "reports", ".willy_exit_v2_gate.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
STRATEGIES = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "entry_price", "pnl_percentage", "close_reason",
        "peak_pnl", "entry_btc_1h_slope") + WEra.V2_COLS
# the PINNED pre-356 exit (the shadow) — DECISION_LOG 281 / 297; never read from config
OLD_TP, OLD_STOP, OLD_CAP, (OLD_LATE_MIN, OLD_LATE_TP) = WEra.OLD_TP, WEra.OLD_STOP, WEra.OLD_HOLD, WEra.OLD_LATE
ARM, FLOOR = 1.25, 1.0                    # the shipped floor (DL 356; operator) — the parity re-walk uses these
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
LADDER_N, LADDER_DAYS, LADDER_MIN_GAIN, LADDER_SHARE = 15, 6, 0.10, 0.50
LADDER_FLOOR_SLIP = 0.05                  # a ladder floor exit (taker + slippage) that replaces a flat TP / cap / late exit


# ─────────────────────────── walkers (net % path: t, hi, lo, cl, op, bar, end_ms, te) ───────────────────────────
def walk_v2(P, tp, sl=OLD_STOP, cap=OLD_CAP, late=OLD_LATE_MIN, ztp=OLD_LATE_TP, arm=ARM, floor=FLOOR, floor_fn=None):
    """the LIVE V2 exit on a net path → dict(why TP / LATE_TP / STOP / FLOOR / CAP / UNRES, pct, armed). Before the floor arms it is
    scout_willy_timecap.evaluate_late(book="live") exactly (stop first on an event touching both lines; resting TP books the line; a late line
    already through at the first late event books that print / bar open). The floor exists only when tp > arm; it arms on the first event
    with hi ≥ arm (not judged on the arming event itself — the order inside a bar is unknown); armed: an event touching the floor → FLOOR
    (bar: min(floor, open); tick: the print) — checked before the TP (conservative); no stop, no late line; the cap still closes."""
    te = int(P["te"])
    t, hi, lo, op, bar = P["t"], P["hi"], P["lo"], P["op"], P["bar"]
    n = int(np.searchsorted(t, te + cap * MIN, side="right"))
    has_floor = floor is not None and float(tp) > float(arm) + 1e-9
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
    """the PINNED old exit (+1.25 resting / −4 / late +0.25 @30 / 90) — scout_willy_timecap.evaluate_late(book='live')."""
    import scout_willy_timecap as WT
    r = WT.evaluate_late(P, OLD_TP, OLD_STOP, OLD_CAP, OLD_LATE_MIN, OLD_LATE_TP, book="live")
    return dict(why=r["why"], pct=float(r["pct"]) if r.get("pct") is not None else float("nan"))


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
    """→ (closed counted fills, operator-closed excluded, all closed WILLY rows for the era line)."""
    o = dedup(raw)
    c = o[o.status.astype(str).str.upper() == "CLOSED"].copy() if len(o) else o
    if not len(c):
        return c, c, c
    c["ts"] = pd.to_datetime(c._k, format="ISO8601", errors="coerce")
    c = c[c.ts.notna()].copy()
    allc = c
    if floor_ms is None:
        return c.iloc[0:0], c.iloc[0:0], allc
    c = c[c.ts >= pd.Timestamp(int(floor_ms), unit="ms")]
    kept, ox = _GK.split_operator_closed(c)
    return kept.sort_values(["ts", "pair"], kind="stable").reset_index(drop=True), ox, allc


def _f(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def armed_live(r, arm=ARM):
    """the fill ARMED live: a runner target (> arm) and (willy_floor_armed_at recorded or peak_pnl ≥ arm)."""
    tp = _f(r.get(WEra.TP_COL))
    if tp is None or tp <= arm + 1e-9:
        return False
    if str(r.get(WEra.ARM_COL) or "").strip() not in ("", "nan", "NaT", "None"):
        return True
    pk = _f(r.get("peak_pnl"))
    return pk is not None and pk >= arm - 1e-9


def btc_side(v):
    x = _f(v)
    return "unread" if x is None else ("falling" if x <= 0 else "rising")


def price(c, path_fn=None, fetch=False, now_ms=None):
    """→ c with shadow / v2 walk columns. path_fn(row) → P or None (default: the hybrid tick → 1m loader on the shared caches)."""
    c = c.copy()
    if not len(c):
        for k in ("live", "shadow", "shadow_why", "v2", "v2_why", "armed", "src", "side", "delta") + tuple(
                x for k_ in LADDERS for x in (f"lad_{k_}", f"ladd_{k_}")):
            c[k] = pd.Series(dtype=object)
        return c
    c["live"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c["side"] = [btc_side(v) for v in c.get("entry_btc_1h_slope", pd.Series([None] * len(c)))]
    c["armed"] = [armed_live(r) for r in c.to_dict("records")]
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
                print(f"  (WILLY_EXIT_V2: fetch pass skipped: {str(e)[:80]})")

        def path_fn(r):
            if not (r["E"] and r["E"] > 0):
                return None
            P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), max(OLD_CAP, 120), tcache)
            P["_src"] = FX.path_src(P)
            return P
    sh, shw, v2, v2w, src = [], [], [], [], []
    lad = {k: [] for k in LADDERS}
    for r in c.to_dict("records"):
        try:
            P = path_fn(r)
        except Exception:
            P = None
        if P is None or not len(P.get("t", [])):
            sh.append(np.nan); shw.append("UNPRICED"); v2.append(np.nan); v2w.append("UNPRICED"); src.append(None)
            for k in LADDERS:
                lad[k].append(np.nan)
            continue
        s = shadow_today(P)
        w = walk_v2(P, _f(r.get(WEra.TP_COL)) or OLD_TP)
        sh.append(s["pct"]); shw.append(s["why"]); v2.append(w["pct"]); v2w.append(w["why"]); src.append(P.get("_src", "path"))
        for k, fn in LADDERS.items():   # 🪜 (357) same path, same target, rising floor
            lw = walk_v2(P, _f(r.get(WEra.TP_COL)) or OLD_TP, floor_fn=fn)
            lad[k].append(lw["pct"] - (LADDER_FLOOR_SLIP if (lw["why"] == "FLOOR" and w["why"] != "FLOOR") else 0.0))
    c["shadow"], c["shadow_why"], c["v2"], c["v2_why"], c["src"] = sh, shw, v2, v2w, src
    c["delta"] = c.live - c.shadow
    for k in LADDERS:
        c[f"lad_{k}"] = lad[k]
        c[f"ladd_{k}"] = c[f"lad_{k}"] - c.v2
    return c


# ─────────────────────────── gates ───────────────────────────
def parity(c):
    """→ (ok, text): share of priced fills whose V2 re-walk is within PARITY_TOL of the booked %."""
    g = c[c.v2.notna() & c.live.notna()] if len(c) else c
    if not len(g):
        return True, "parity: no priced fill yet"
    within = (g.v2 - g.live).abs() <= PARITY_TOL
    share = float(within.mean())
    ok = share >= PARITY_MIN
    return ok, (f"parity (V2 re-walk vs booked, ±{PARITY_TOL}): {int(within.sum())}/{len(g)} = {share * 100:.0f} %"
                + ("" if ok else f" < {PARITY_MIN * 100:.0f} % → no verdict (parity)"))


def _days(g):
    return g.ts.dt.strftime("%Y-%m-%d").nunique() if len(g) else 0


def reads(c):
    """the three FROZEN reads on the armed fills (in open order) → {name: (verdict, text, sample_complete)}. COLLECTING = not reached."""
    a = c[c.armed].sort_values(["ts", "pair"], kind="stable") if len(c) else c
    out = {}

    def _sample_ok(s):
        return bool(len(s)) and bool(s.shadow.notna().all())

    e = a.head(EARLY_N)
    if len(e) >= EARLY_N:
        lv, sv = e.live.mean(), e.shadow.mean()
        if not _sample_ok(e):
            out["EARLY"] = ("HOLD", f"first {EARLY_N} armed: {int(e.shadow.isna().sum())} unpriced — held", False)
        else:
            bad = lv < sv - EARLY_GAP
            out["EARLY"] = (("ALERT" if bad else "OK"), f"first {EARLY_N} armed: live {lv:+.3f} vs shadow {sv:+.3f} %/fill (alert if < shadow − "
                            f"{EARLY_GAP}) → " + ("⚠ ALERT: review WILLY exit V2" if bad else "no alert"), True)
    else:
        out["EARLY"] = ("COLLECTING", f"early alert: {len(a)}/{EARLY_N} armed", False)
    m = None
    for k in range(len(a)):
        s = a.iloc[:k + 1]
        if k + 1 >= MAIN_N and _days(s) >= MAIN_DAYS:
            m = s
            break
    if m is None:
        out["MAIN"] = ("COLLECTING", f"main read: {len(a)}/{MAIN_N} armed on {_days(a)}/{MAIN_DAYS} UTC days", False)
    elif not _sample_ok(m):
        out["MAIN"] = ("HOLD", f"main read sample ({len(m)}): {int(m.shadow.isna().sum())} unpriced — held", False)
    else:
        sd = float(m.delta.sum())
        out["MAIN"] = (("REVERT" if sd < 0 else "PASS"), f"main read ({len(m)} armed on {_days(m)} days): Σ(live − shadow) {sd:+.2f} pts → "
                       + ("🛑 REVERT to +1.25 (frenzy_willy_high_target_enabled false)" if sd < 0 else "PASS (keep V2)"), True)
    f = a.head(FINAL_N)
    if len(f) < FINAL_N:
        out["FINAL"] = ("COLLECTING", f"final read: {len(a)}/{FINAL_N} armed", False)
    elif not _sample_ok(f):
        out["FINAL"] = ("HOLD", f"final read sample: {int(f.shadow.isna().sum())} unpriced — held", False)
    else:
        sd = float(f.delta.sum())
        out["FINAL"] = (("REVERT" if sd < 0 else "PASS"), f"final read (first {FINAL_N} armed): Σ(live − shadow) {sd:+.2f} pts → "
                        + ("🛑 REVERT to +1.25" if sd < 0 else "PASS (keep V2)"), True)
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
    if complete and allow and v not in ("COLLECTING", "HOLD"):
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


def ledger_a(c):
    a = c[c.armed & c.delta.notna()] if len(c) else c
    if not len(a):
        return ["Ledger A (target + floor): 0 armed priced fills"]
    rs = a.close_reason.astype(str)
    run = a[rs.str.endswith("FRENZY_TP") & (a.live > OLD_TP + 0.05)]
    flo = a[rs.str.endswith(WEra.FLOOR_REASON)]
    oth = a[~a.index.isin(run.index) & ~a.index.isin(flo.index)]
    L = [f"Ledger A (target + floor): {len(a)} armed priced fills · Σ(live − shadow) {a.delta.sum():+.2f} pts · avg {a.delta.mean():+.3f}"]
    for lab, g in (("runners (TP above +1.25)", run), ("floor exits (WILLY_FLOOR)", flo), ("other (cap / late / stop)", oth)):
        L.append(f"  {lab}: {len(g)} · live {_avg(g.live)} vs shadow {_avg(g.shadow)} · Σ Δ {g.delta.sum():+.2f}" if len(g) else f"  {lab}: 0")
    return L


def ladder_reads(c):
    """🪜 (357) per ladder: the frozen read on the armed fills (open order) → {name: (verdict, text, complete)}."""
    a = c[c.armed].sort_values(["ts", "pair"], kind="stable") if len(c) else c
    m = None
    for k in range(len(a)):
        s = a.iloc[:k + 1]
        if k + 1 >= LADDER_N and _days(s) >= LADDER_DAYS:
            m = s
            break
    out = {}
    for k in LADDERS:
        col = f"ladd_{k}"
        if m is None:
            out[f"LADDER_{k}"] = ("COLLECTING", f"{len(a)}/{LADDER_N} armed on {_days(a)}/{LADDER_DAYS} days", False)
            continue
        d = pd.to_numeric(m[col], errors="coerce")
        if d.isna().any():
            nv = int(pd.to_numeric(m.v2, errors="coerce").isna().sum())
            out[f"LADDER_{k}"] = ("HOLD", f"sample ({len(m)}): {nv} unpriced / {int(d.isna().sum()) - nv} unresolved ladder walks — held", False)
            continue
        gain = d.clip(lower=0)
        share = float(gain.max() / gain.sum()) if gain.sum() > 0 else 1.0
        ok = float(d.mean()) >= LADDER_MIN_GAIN and share < LADDER_SHARE
        out[f"LADDER_{k}"] = (("CANDIDATE" if ok else "KEEP_FLAT"),
                              f"{len(m)} armed on {_days(m)} days: avg Δ vs flat {d.mean():+.3f} %/fill · top fill {share:.0%} of the gain → "
                              + ("🪜 CANDIDATE: operator decides an engine ladder" if ok else "keep the flat +1.0 floor"), True)
    return out


def ladder_ledger(c):
    a = c[c.armed & c.v2.notna()] if len(c) else c
    if not len(a):
        return ["🪜 Ladder shadows: 0 armed priced fills"]
    L = [f"🪜 Ladder shadows on {len(a)} armed fills (Δ = ladder − flat +1.0 re-walk, same path):"]
    for k in LADDERS:
        d = pd.to_numeric(a[f"ladd_{k}"], errors="coerce")
        L.append(f"  {k}: avg {_avg(a[f'lad_{k}'])} vs flat {_avg(a.v2)} · Σ Δ {d.sum():+.2f} · better {int((d > 1e-9).sum())} / worse {int((d < -1e-9).sum())}")
    L.append("  per fill (last 20): " + " · ".join(f"{r.pair} {str(r.opened_at)[5:16]} flat {r.v2:+.2f} | " + " ".join(f"{k} {getattr(r, f'lad_{k}'):+.2f}" for k in LADDERS)
                                                    for r in a.tail(20).itertuples()))
    return L


def batch_table(c):
    if not len(c):
        return []
    g = c[c.shadow.notna()]
    L = ["| batch | BTC side | N | armed | before: shadow (today's exit) %/fill | after: live V2 %/fill | Δ |", "|---|---|---|---|---|---|---|"]
    for (b, s), x in g.groupby(["batch", "side"], sort=True):
        L.append(f"| {b} | {s} | {len(x)} | {int(x.armed.sum())} | {_avg(x.shadow)} | {_avg(x.live)} | {x.delta.mean():+.3f} |")
    for s, x in g.groupby("side", sort=True):
        L.append(f"| ALL | {s} | {len(x)} | {int(x.armed.sum())} | {_avg(x.shadow)} | {_avg(x.live)} | {x.delta.mean():+.3f} |")
    L.append(f"| ALL | all | {len(g)} | {int(g.armed.sum())} | {_avg(g.shadow)} | {_avg(g.live)} | {g.delta.mean():+.3f} |")
    return L


def run(now_ms=None, paths=None, raw=None, floor_ms="auto", path_fn=None, fetch=True, state_path=STATE):
    L = [f"## 🎲² WILLY_EXIT_V2 — TP at the spike high + floor (DECISION_LOG {DL}, frozen gates; shadow = the pinned +1.25 / −4 / late 0.25@30 / 90 exit)", ""]
    if raw is None:
        raw = read_exports(paths or EXPORT_GLOB)
    if floor_ms == "auto":
        floor_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_EXIT_V2")
    c, ox, allc = cohort(raw, floor_ms)
    L.append(WEra.era_line(allc, "WILLY + WILLY_HT, all closed in the exports"))
    if floor_ms is None:
        L += ["NOT evaluable yet — no DECISION_LOG 356 commit found (the counted cohort starts at its deploy + 10 min).", ""]
        return L
    L.append(f"Cohort: {len(c)} closed WILLY / WILLY_HT fills since the DL 356 deploy + 10 min · " + _GK.operator_closed_line(ox))
    c = price(c, path_fn=path_fn, fetch=fetch, now_ms=now_ms)
    n_un = int(c.shadow.isna().sum()) if len(c) else 0
    L.append(f"Priced: {len(c) - n_un} · unpriced (no path yet): {n_un} · armed: {int(c.armed.sum()) if len(c) else 0} · "
             f"runner targets (> +{ARM}): {int((pd.to_numeric(c.get(WEra.TP_COL), errors='coerce') > ARM + 1e-9).sum()) if len(c) else 0}")
    pok, ptxt = parity(c)
    L.append(ptxt)
    L += ledger_a(c)
    L += ladder_ledger(c)
    L += [""] + batch_table(c) + [""]
    L.append("🔒 FROZEN gates (operator decides; persisted once decided):")
    for name, (v, txt, complete) in reads(c).items():
        if not pok and complete:
            L.append(f"  {name}: no verdict (parity) — {txt}")
            continue
        fv, ft = frozen(state_path, name, v, txt, complete, pok)
        L.append(f"  {name}: {ft}")
    for name, (v, txt, complete) in ladder_reads(c).items():   # 🪜 (357) observe-only reads, same persistence
        if not pok and complete:
            L.append(f"  {name}: no verdict (parity) — {txt}")
            continue
        fv, ft = frozen(state_path, name, v, txt, complete, pok)
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
    assert walk_v2(dip, 3.5)["why"] == "FLOOR" and abs(walk_v2(dip, 3.5)["pct"] - 0.9) < 1e-9
    same = _P(te, [(5, 0.7, 0.7), (35, 0.3, 0.3)])
    assert walk_v2(same, 1.25) == {**walk_v2(same, 1.25)} and walk_v2(same, 1.25)["why"] == shadow_today(same)["why"] == "LATE_TP"
    print("selftest WILLY_EXIT_V2 OK")
    return True


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("\n".join(run(fetch="--no-fetch" not in sys.argv)))
