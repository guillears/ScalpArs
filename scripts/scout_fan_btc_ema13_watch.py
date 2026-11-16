#!/usr/bin/env python3
"""👁 Scout — FAN_BTC_EMA13_DEEP observe line (DECISION_LOG 567, operator 2026-11-16 "ship it" for the observe line; OBSERVE ONLY — never
changes config, no bot API, no network).

QUESTION   Do FAN flips (FLIP:FAN_RATIO_GATE shorts) opened with BTC already far below its EMA13 lose? (a fresh BTC dump bounces and squeezes
           the new short — reports/FAN_FLIP_WL_2026-11-16.md: master 20 · 65 % · −0.11 vs rest 37 · 84 % · +0.27; day-boot P 0.79, shuffled
           null p 0.15, backtest 2 / 3 seeds → not armed.)
TAG        entry_btc_dist_from_ema13_pct ≤ −0.16 (BTC close vs its 5 m EMA13 at entry — the FLIP_FAN_BTC_EMA13 gate's own reading). FROZEN,
           never re-fit.
COHORT     CLOSED FAN flips from the orders exports (paper + live via export_glob; dedup (opened_at[:19], pair, entry_strategy), CLOSED beats
           OPEN, newest export wins), opened ≥ the DECISION_LOG 567 commit + 10 min (forward only — the study's fills never count), booked
           P&L (the live exit at the time: the DL 563 +0.5 % resting TP).
GATE       arm CANDIDATE (operator decides) once tagged fills reach N ≥ 15 on ≥ 8 UTC days AND the day-block bootstrap P(mean < 0) ≥ 0.95 AND
           tagged WR < untagged WR; else keep observing. Decided once at the first prefix that meets N / days (state in
           reports/SCOUT_FAN_BTC_EMA13.json).
Usage: venv/bin/python -I scripts/scout_fan_btc_ema13_watch.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as _EG          # noqa: E402
import scout_gatekit as _GK        # noqa: E402

DL = 567
CUT = -0.16
N_MIN, DAYS_MIN, P_MIN = 15, 8, 0.95
ES = "FLIP:FAN_RATIO_GATE"
_ROOT = os.path.dirname(_D)
STATE = os.path.join(_ROOT, "reports", "SCOUT_FAN_BTC_EMA13.json")
EXPORT_GLOB = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
COLS = ("opened_at", "pair", "direction", "entry_strategy", "status", "pnl_percentage", "close_reason", "entry_btc_dist_from_ema13_pct",
        "cell_multiplier_source", "pattern_cell_source", "is_probe")


def tagged(dist):
    """PURE: the frozen tag — BTC at least 0.16 % below its EMA13 at entry. Unreadable → False (never tagged)."""
    try:
        v = float(dist)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v) and v <= CUT)


def load(paths=None, since_ms=None):
    fr = []
    for f in (_EG.glob_exports(EXPORT_GLOB) if paths is None else paths):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS)
        except Exception:
            continue
        if not {"opened_at", "pair", "entry_strategy", "status"} <= set(d.columns):
            continue
        fr.append(d.assign(_m=os.path.getmtime(f)))
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "day", "pct", "tag"])
    d = pd.concat(fr, ignore_index=True).reindex(columns=list(COLS) + ["_m"])
    d = d[d.entry_strategy.astype(str).eq(ES) & d.direction.astype(str).eq("SHORT")].copy()
    _prb = pd.Series(False, index=d.index)   # (565 review) full-size only: probes out (is_probe / *_PROBE cell tags), operator closes out
    if "is_probe" in d:
        _prb |= d.is_probe.astype(str).str.lower().isin(("true", "1", "1.0"))
    for _pc in ("cell_multiplier_source", "pattern_cell_source"):
        if _pc in d:
            _prb |= d[_pc].astype(str).str.upper().str.contains("PROBE")
    if "close_reason" in d:
        _prb |= d.close_reason.astype(str).str.upper().str.startswith(("MANUAL", "OVERRIDE"))
    d = d[~_prb]
    d["_k"] = d.opened_at.astype(str).str.replace(" ", "T").str[:19]
    d["_c"] = d.status.astype(str).str.upper().eq("CLOSED")
    d = d.sort_values(["_c", "_m"], kind="stable").drop_duplicates(["_k", "pair", "entry_strategy"], keep="last")
    d["o"] = pd.to_datetime(d._k, format="ISO8601", errors="coerce")
    c = d[d._c & d.o.notna()].copy()
    if since_ms is not None:
        c = c[c.o >= pd.Timestamp(int(since_ms), unit="ms")]
    c["pct"] = pd.to_numeric(c.pnl_percentage, errors="coerce")
    c = c[c.pct.notna()]
    c["tag"] = [tagged(v) for v in c.entry_btc_dist_from_ema13_pct]
    c["day"] = c.o.dt.strftime("%Y-%m-%d")
    return c.sort_values("o").reset_index(drop=True)


def p_mean_neg(t, n_boot=5000, seed=567):
    """PURE: day-block bootstrap P(mean < 0) of the tagged fills' pct."""
    if not len(t):
        return float("nan")
    g = [v.pct.values for _, v in t.groupby("day")]
    rng = np.random.default_rng(seed)
    return float(np.mean([np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))]).mean() < 0 for _ in range(n_boot)]))


def verdict(c):
    """c = counted fills (pct, tag, day) → (state, detail). PURE."""
    t, r = c[c.tag], c[~c.tag]
    n, nd = len(t), t.day.nunique() if len(t) else 0
    wr_t = (t.pct > 0).mean() if n else float("nan")
    wr_r = (r.pct > 0).mean() if len(r) else float("nan")
    base = (f"tagged {n} · {nd} d · WR {wr_t:.0%} · avg {t.pct.mean() if n else float('nan'):+.3f} vs untagged {len(r)} · "
            f"WR {wr_r:.0%} · avg {r.pct.mean() if len(r) else float('nan'):+.3f}")
    if n < N_MIN or nd < DAYS_MIN:
        return "COLLECTING", base + f" (needs {N_MIN} on {DAYS_MIN} days)"
    p = p_mean_neg(t)
    ok = p >= P_MIN and (not len(r) or wr_t < wr_r)
    return ("ARM CANDIDATE (operator decides)" if ok else "NO ARM (keep observing)"), base + f" · P(mean<0) {p:.2f}"


def run(paths=None, state_path=STATE):
    since = _GK.dl_deploy_ms(DL, label="FAN_BTC_EMA13_DEEP")
    if since is None:
        return "## 👁 FAN_BTC_EMA13_DEEP (DL 567) — NOT EVALUABLE (deploy commit not found)"
    c = load(paths, since)
    st = {}
    try:
        st = json.load(open(state_path)) if state_path and os.path.exists(state_path) else {}
    except Exception:
        st = {}
    if st.get("state"):
        gate = f"**{st['state']}** (frozen {st['at']}: {st['detail']})"
    else:
        state, det = verdict(c)
        gate = f"{state} — {det}"
        if state != "COLLECTING" and state_path:
            json.dump(dict(state=state, detail=det, at=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")), open(state_path, "w"))
    lines = [f"## 👁 FAN_BTC_EMA13_DEEP — FAN flips with BTC ≤ {CUT} % below its EMA13 (DECISION_LOG 567, observe only)",
             f"Forward FAN flips since the DL {DL} deploy: {len(c)} · tagged {int(c.tag.sum()) if len(c) else 0}.", f"- Gate: {gate}"]
    for _, x in c.tail(12).iterrows():
        lines.append(f"  - {x.o:%m-%d %H:%M} {x.pair} {'🏷 tagged' if x.tag else 'untagged'} · BTC vs EMA13 {x.entry_btc_dist_from_ema13_pct} · {x.pct:+.2f} % ({x.close_reason})")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
