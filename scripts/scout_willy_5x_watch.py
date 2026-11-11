#!/usr/bin/env python3
"""⛔ RETIRED 2026-11-11 (DECISION_LOG 530): decided — WILLY's size is set by C5 (DL 528). Unhooked from the opportunity scout; still runnable standalone.
👁 Scout — WILLY_5X gate (DECISION_LOG 425, operator 2026-10-31: "reduce the leverage of Willy until we solve the winners vs loosers"; OBSERVE ONLY — never changes config, no bot
API, no network).

QUESTION   WILLY went 10× → 5× as a bridge. When does it go back? Leverage changes no % result, so every WILLY watch line keeps counting; only the account's exposure halves.
COHORT     CLOSED FRENZY_WILLY fills, from the orders exports (export_glob), dedup (opened_at[:19], pair, entry_strategy) — CLOSED beats OPEN, newest export wins; operator-closed out
           (scout_gatekit). COUNTED: opened ≥ the DECISION_LOG 425 commit + 10 min, in open order. FRENZY_WILLY_HT runs the same leverage: shown on its own row, never decides.
FROZEN BAR (operator-approved with the change, never re-fit): at the first 15 counted fills, decided ONCE — avg < 0 → SLEEVE REVIEW (full kill checklist, no auto-off) · WR ≥ 69.0 % (the
           frozen V2b breakeven, DL 412) ∧ avg > 0 → RESTORE 10× CANDIDATE (operator decides; never automatic) · else STAY AT 5×. A filter passing its own gate and shipping also opens the
           10× question, at the operator's call. Deferred until the newest export is ≥ the 15th fill + WILLY's max hold. Pinned to lev 0.25 + the V2b exit keys (a change → ⚠ STALE).
NOTE (DL 487)  since 11-06 WILLY also runs at HALF margin (frenzy_willy_invest_mult 0.5) — "RESTORE 10×" means the LEVERAGE knob; the margin
           returns to 1.0 first, by DL 487's own revert gate (WILLY_GAP checkpoint). % results are size-free, so this read is unaffected.
REGISTRATION (never counted) B25 WILLY 19 · 42 % · −1.54 %/fill, daily compound on the real account path 10× −6.84 → 5× −3.16 %/day · master 38 · 82 % · +0.71, today's size model +3.66 → +1.90.
STATE      reports/SCOUT_WILLY_5X.json (gitignored).
Usage: venv/bin/python -I scripts/scout_willy_5x_watch.py
"""
import json
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402
import scout_b1h_negflank as NF    # noqa: E402  (state I/O)

DL = 425
SLEEVE, HT = "FRENZY_WILLY", "FRENZY_WILLY_HT"
N_MIN = 15
BREAKEVEN_WR = 69.0
CAP_DEFAULT = 90
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_WILLY_5X.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "closed_at", "pair", "entry_strategy", "status", "pnl_percentage", "close_reason", "leverage")
REVIEW, RESTORE, STAY = "SLEEVE REVIEW (kill checklist, no auto-off)", "RESTORE 10× CANDIDATE (operator decides; never automatic)", "STAY AT 5×"


def cap_minutes(path=None):
    try:
        with open(path or os.path.join(_ROOT, "trading_config.json")) as f:
            c = json.load(f)
        return float(c.get("thresholds", c).get("frenzy_willy_max_hold_minutes", CAP_DEFAULT))
    except Exception:
        return float(CAP_DEFAULT)


def load(paths=None):
    """→ (closed WILLY + HT fills in open order; the newest read export's time or None)."""
    fr, newest = [], None
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if not {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            continue
        fr.append(d.assign(_m=os.path.getmtime(f)))
        s = _EG.export_stamp(f)
        try:
            t = pd.Timestamp(s) if s else pd.Timestamp(os.path.getmtime(f), unit="s")
        except (OSError, ValueError):
            continue
        newest = t if newest is None or t > newest else newest
    names = list(COLS) + ["o", "day", "pct"]
    if not fr:
        return pd.DataFrame(columns=names), newest
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str).isin((SLEEVE, HT))].copy()
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values(["o", "pair"]).reset_index(drop=True), newest


def verdict(g):
    """g = the counted fills (pct) → (state, detail). PURE. Below N_MIN → COLLECTING."""
    n = len(g)
    if n < N_MIN:
        return "COLLECTING", f"{n}/{N_MIN} fills"
    wr, avg = float((g.pct > 0).mean() * 100), float(g.pct.mean())
    det = f"{n} fills · WR {wr:.0f} % · avg {avg:+.3f} %"
    if avg < 0:
        return REVIEW, det
    if wr >= BREAKEVEN_WR and avg > 0:
        return RESTORE, det
    return STAY, det


def dcr(g, days):
    """daily compound %, each trade = account, over the calendar days given. PURE."""
    r = (g.groupby("day").pct.sum() / 100.0).reindex(days).fillna(0.0)
    return float(((1 + r).prod() ** (1 / len(days)) - 1) * 100) if len(days) else float("nan")


def _row(lab, g, days):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – |"
    lev = pd.to_numeric(g.leverage, errors="coerce").median() if "leverage" in g else float("nan")
    return f"| {lab} | {len(g)} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.3f} % | {dcr(g, days):+.3f} %/day | {lev:.0f}× |"


def run(now_ms=None, paths=None, state_path=STATE, since_ms=None):
    """→ markdown lines. since_ms None → the DECISION_LOG 425 commit + 10 min (not in git → NOT EVALUABLE). state_path None → no write."""
    now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    L = [f"## 👁 WILLY_5X — WILLY at 5×: back to 10× or review? (DECISION_LOG {DL}, OBSERVE only)", ""]
    given = since_ms
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_5X gate")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    c, newest = load(paths)
    c = c[c.o >= since] if len(c) else c
    keep, ox = _GK.split_operator_closed(c) if len(c) else (c, c)
    w = keep[keep.entry_strategy.astype(str) == SLEEVE] if len(keep) else keep
    h = keep[keep.entry_strategy.astype(str) == HT] if len(keep) else keep
    days = [d.strftime("%Y-%m-%d") for d in pd.date_range(since.normalize(), pd.Timestamp(now_ms, unit="ms").normalize())]
    L += [f"Cohort: CLOSED fills opened ≥ {since:%Y-%m-%d %H:%M} UTC ({'DL ' + str(DL) + ' commit + 10 min' if given is None else 'floor given'}) · operator-closed: {len(ox)} — excluded.", "",
          "| Sleeve | N | WR | avg % | daily compound (each trade = account) | leverage (median) |", "|---|---|---|---|---|---|",
          _row(SLEEVE, w, days), _row(HT + " (information, never decides)", h, days), ""]
    st, ok = NF.du_load_state(state_path, now_ms) if state_path else ({}, True)
    rec = st.get(SLEEVE) if ok else None
    if rec is not None and not (isinstance(rec, dict) and {"state", "detail", "at", "run_at"} <= set(rec)):
        rec, ok = None, False
    why = []
    if not rec and ok and len(w) >= N_MIN:
        pre = w.iloc[:N_MIN]
        last, cap = pre.o.max(), cap_minutes()
        if newest is None or newest < last + pd.Timedelta(minutes=cap):
            why.append(f"newest export < the {N_MIN}th fill + {cap:.0f} min")
        else:
            state, det = verdict(pre)
            rec = dict(state=state, detail=det, at=f"{last:%Y-%m-%d %H:%M} UTC", run_at=pd.Timestamp(now_ms, unit="ms").strftime("%Y-%m-%d %H:%M UTC"))
            if state_path:
                st[SLEEVE] = rec
                NF.du_save_state(st, state_path)
    ls, ld = verdict(w)
    if rec and ok and state_path:
        L += [f"**Frozen verdict (15th fill {rec['at']}, frozen {rec['run_at']}): {rec['state']}** ({rec['detail']})", f"Live (information only, never re-decides): {ls} — {ld}"]
    else:
        L.append(f"Now: ⏳ {ls} ({ld}).")
    if why:
        L.append("⏸ freezing deferred: " + " · ".join(why) + " — re-checked next run.")
    if not ok and state_path and rec is None:
        L.append(f"⚠ frozen state corrupt or damaged — operator restore needed ({os.path.basename(state_path)}); freezing skipped this run.")
    L += ["", f"FROZEN bar (DL {DL}): at the first {N_MIN} counted WILLY fills, decided ONCE — avg < 0 → {REVIEW}; WR ≥ {BREAKEVEN_WR} % ∧ avg > 0 → {RESTORE}; else {STAY}. A filter that passes its own gate and ships also opens the 10× question (operator).",
          "Registration (never counted): B25 19 · 42 % · −1.54; daily compound, real path, 10× −6.84 → 5× −3.16. Master 38 · 82 % · +0.71; today's size model +3.66 → +1.90.", ""]
    return L


if __name__ == "__main__":
    print("\n".join(run()))
