#!/usr/bin/env python3
"""👁 Scout — LITE_C_REVERT tracker (DECISION_LOG 420, operator 2026-10-30: "keep tracking this … to re check if the revert was positive after all"; OBSERVE ONLY — never changes
config, no bot API, no network).

QUESTION   LITE C (+2 % target outside the zone) was switched off when its locked gate fired (Σ Δ −9.91 on 13 trades, 0 saved, 10 capped). Was going back to +3 % right?
METHOD     Every closed fill the old rule would have put on +2 (lite_c_zone not true, still stamped with the switch off) is priced both ways: booked = today's exit; shadow = +2 if its
           peak_pnl reached +2 (within 0.05), else the booked result (it never reached the old target). Δ = booked − shadow: +1 when it went on to +3 · −5 when it reached +2, then stopped (BTW type).
           Zone fills ran +3 under both rules. peak_pnl is the bot's own stamp, so a wick it did not record counts as not reached; no klines needed.
COHORT     CLOSED FRENZY_LITE and FRENZY_LITE_RETRY LONG fills, decided SEPARATELY, from the orders exports (export_glob), dedup (opened_at[:19], pair, entry_strategy) — CLOSED beats OPEN,
           newest export wins; operator-closed out (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 420 commit + 10 min, booked on the +3 target (a stored frenzy_lite_tp_used of 2 = an
           order from before the switch, out). A fill without a peak stamp → UNSCORED (listed, never counted).
FROZEN BAR (mirrors the DL 301 main read, never re-fit), per sleeve: at the first prefix where outside fills reach N ≥ 20 on ≥ 8 UTC days, decided ONCE — Σ Δ ≤ 0 → RE-ARM CANDIDATE (operator
           decides; never automatic) · Σ Δ > 0 ∧ no day and no pair ≥ 50 % of the positive Δ → REVERT CONFIRMED · else KEEP OBSERVING. Deferred until the newest export is ≥ the crossing fill + the
           max hold. Each read shows WR · avg · daily compound (each trade = account) booked vs shadow. Pinned to +3 / −3 / 12 h with LITE C off (a change → ⚠ STALE).
REGISTRATION (in-sample, NEVER counted) +3 vs LITE C: B25 LITE 12 · 75 % · +1.50 vs +0.83 (compound +2.19 vs +1.21) · master LITE 30 · 53 % · +0.20 vs 60 % · +0.30 (+0.21 vs +0.37) · B25 RETRY 16 · 50 % · 0.00 vs 56 % · 0.00.
STATE      reports/SCOUT_LITE_C_REVERT.json (gitignored).
Usage: venv/bin/python -I scripts/scout_lite_c_revert_watch.py
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import scout_b1h_negflank as NF    # noqa: E402  (state I/O)

DL = 420
SLEEVES = ("FRENZY_LITE", "FRENZY_LITE_RETRY")
OLD_TP = 2.0
PEAK_TOL = 0.05                    # a resting limit books +2 with the running peak just under it (REZ 10-27 1.9972, DUSK 10-17 1.9976): frozen
N_MIN, DAYS_MIN, SHARE_MAX = 20, 8, 0.50
CAP_DEFAULT = 720
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_LITE_C_REVERT.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "close_reason", "peak_pnl", "lite_c_zone", "frenzy_lite_tp_used")
CONFIRMED, REARM = "REVERT CONFIRMED", "RE-ARM CANDIDATE (operator decides; never automatic)"


def cap_minutes(path=None):
    try:
        with open(path or os.path.join(_ROOT, "trading_config.json")) as f:
            c = json.load(f)
        return float(c.get("thresholds", c).get("frenzy_max_hold_minutes", CAP_DEFAULT))
    except Exception:
        return float(CAP_DEFAULT)


def shadow_pct(booked, peak):
    """the old +2 rule's result on this fill. PURE."""
    return OLD_TP if peak >= OLD_TP - PEAK_TOL else booked


def load(paths=None):
    """→ (closed LITE / RETRY fills that ran +3 with zone, shadow, delta, scored; the newest read export's time or None)."""
    fr, newest = [], None
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if not {"opened_at", "pair", "direction", "entry_strategy", "status"} <= set(d.columns):
            continue
        fr.append(d.assign(_m=os.path.getmtime(f)))
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    names = list(COLS) + ["o", "day", "zone", "pct", "shadow", "delta", "scored"]
    if not fr:
        return pd.DataFrame(columns=names), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str).isin(SLEEVES) & (d.direction.astype(str) == "LONG")].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    tp = pd.to_numeric(c.frenzy_lite_tp_used, errors="coerce")
    c = c[c.pct.notna() & ~(tp == OLD_TP)]
    z = c.lite_c_zone.astype(str).str.lower()
    c["zone"] = z.isin(("true", "1", "1.0", "t"))
    pk = pd.to_numeric(c.peak_pnl, errors="coerce")
    c["scored"] = pk.notna() & c.lite_c_zone.notna()   # no zone stamp: cannot tell which rule it would have run
    c["shadow"] = [shadow_pct(b, p) if np.isfinite(p) else np.nan for b, p in zip(c.pct, pk)]
    c["delta"] = c.pct - c.shadow
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def _pos_share(delta, key):
    g = pd.Series(np.clip(np.asarray(delta, float), 0, None)).groupby(np.asarray(key)).sum()
    return float(g.max() / g.sum()) if g.sum() > 0 else 0.0


def verdict(o):
    """o = counted outside fills (pct, shadow, delta, day, pair) → (state, detail). PURE."""
    n, nd = len(o), o.day.nunique() if len(o) else 0
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", f"outside N {n}/{N_MIN} · {nd}/{DAYS_MIN} days · Σ Δ {float(o.delta.sum()) if n else 0.0:+.2f} pts"
    sd, sp, total = _pos_share(o.delta, o.day), _pos_share(o.delta, o.pair), float(o.delta.sum())
    det = (f"outside N {n} · {nd} d · Σ Δ {total:+.2f} pts · went on to +3 {int((o.delta > 0).sum())} · stopped after +2 {int((o.delta < 0).sum())} · "
           f"top day {sd:.0%} / top pair {sp:.0%} of the positive Δ")
    if total <= 0:
        return REARM, det
    return (CONFIRMED if sd < SHARE_MAX and sp < SHARE_MAX else "KEEP OBSERVING"), det


def crossing_prefix(sc):
    seen, n = set(), 0
    for k, (z, d) in enumerate(zip(sc.zone.values, sc.day.values), 1):
        if z:
            continue
        n += 1
        seen.add(d)
        if n >= N_MIN and len(seen) >= DAYS_MIN:
            return sc.iloc[:k]
    return None


def dcr(g, col, days):
    """daily compound %, each trade = account, over the calendar days given. PURE."""
    r = (g.groupby("day")[col].sum() / 100.0).reindex(days).fillna(0.0)
    return float(((1 + r).prod() ** (1 / len(days)) - 1) * 100) if len(days) else float("nan")


def _row(lab, g, col, days):
    if not len(g):
        return f"| {lab} | 0 | – | – | – |"
    return f"| {lab} | {len(g)} | {(g[col] > 0).mean():.0%} | {g[col].mean():+.3f} % | {dcr(g, col, days):+.3f} %/day |"


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 420 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 LITE_C_REVERT — was going back to +3 % right? (DECISION_LOG {DL}, OBSERVE only)", ""]
    given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="LITE_C_REVERT tracker")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    changed, damaged, cap = False, False, cap_minutes()
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · each sleeve decided separately · {_GK.operator_closed_line(ox)}.", ""]
    for es in SLEEVES:
        s = keep[keep.entry_strategy.astype(str) == es] if len(keep) else keep
        sc = s[s.scored.astype(bool)] if len(s) else s
        uns = len(s) - len(sc)
        o = sc[~sc.zone.astype(bool)] if len(sc) else sc
        days = [d.strftime("%Y-%m-%d") for d in pd.date_range(since.normalize(), pd.Timestamp(now_ms, unit="ms").normalize())]
        L += [f"### {es}", "", "| Outside fills | N | WR | avg % | daily compound (each trade = account) |", "|---|---|---|---|---|",
              _row("booked (+3 %, today)", o, "pct", days), _row("if LITE C were still on (+2 %)", o, "shadow", days), "",
              f"Zone fills (+3 % under both rules): {int(sc.zone.astype(bool).sum()) if len(sc) else 0} · unscored (no peak stamp): {uns}."]
        rec = st.get(es) if ok else None
        if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
            rec, ok, damaged = None, False, True
        why = []
        if not rec and ok:
            pre = crossing_prefix(sc) if len(sc) else None
            if pre is not None:
                last = pre.o.max()
                if newest is None or newest < last + pd.Timedelta(minutes=cap):
                    why.append(f"newest export < the crossing fill + {cap:.0f} min")
                else:
                    state, det = verdict(pre[~pre.zone.astype(bool)])
                    rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
                    if state_path:
                        st[es] = rec
                        changed = True
        ls, ld = verdict(o)
        if rec and ok and state_path:
            L += [f"**Frozen verdict (crossing at fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})", f"Live (information only, never re-decides): {ls} — {ld}"]
        else:
            L.append(f"Now: ⏳ {ls} ({ld}).")
        if why:
            L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
        L.append("")
    if changed and ok and state_path:
        NF.du_save_state(st, state_path)
    if damaged:
        L.append(f"⚠ frozen record damaged — fix {os.path.basename(state_path)} by hand; freezing skipped this run.")
    elif not ok:
        L.append(f"⚠ frozen state corrupt — operator restore needed ({os.path.basename(state_path)}.*.bad); freezing skipped this run.")
    L += [f"FROZEN bar (DL {DL}, per sleeve): at {N_MIN} outside fills on ≥ {DAYS_MIN} UTC days, decided ONCE — Σ Δ ≤ 0 → {REARM}; Σ Δ > 0 ∧ no day / pair ≥ {SHARE_MAX:.0%} of the positive Δ → {CONFIRMED}; else KEEP OBSERVING. "
          "Δ = booked − the +2 shadow. Registration (in-sample, never counted): B25 LITE +0.83 → +1.50, master LITE +0.30 → +0.20, B25 RETRY 0.00 → 0.00 %/fill.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
