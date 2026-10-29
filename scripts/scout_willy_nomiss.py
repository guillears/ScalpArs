#!/usr/bin/env python3
"""🧭 Scout — WILLY entry A R_NOMISS tracker (DECISION_LOG 350, operator-approved fix 2026-10-23, ALERT-only — never changes config).

RULE (engine)   frenzy_willy_a_nomiss: a cold pair read PAST the WILLY A late window (frenzy_willy_a_cold_max_bars) still gets entry A when no
                red 5m bar (close < open) closed from the spike bar up to the bar it was first read on — armed as the on-time A (red wait
                from the spike), the 24 h-volume floor re-checked at the red bar, turnover routing (WILLY / WILLY_HT ¼) as usual. Fills carry
                entry_frenzy_willy_nomiss = True. Study (reports/WILLY_VOLUME_FLOOR_2026-10-23.md § R_NOMISS): ≈ 0.66 trades / week,
                69.4 % WR, Δ vs on-time WILLY −0.07 (day-boot CI −0.81 … +0.63).
COHORT          CLOSED WILLY-class fills (FRENZY_WILLY + FRENZY_WILLY_HT) from the orders exports (paper + live via export_glob; dedup
                (opened_at, pair, direction), CLOSED wins), opened ≥ the DL 350 deploy (git commit + 10 min; not in git → NOT EVALUABLE,
                nothing counted); operator-closed (MANUAL / MANUAL_* / EXTERNAL_CLOSE) excluded — scout_gatekit.
                R_NOMISS = entry_frenzy_willy_nomiss true · normal = every other WILLY-class fill of the same period.
GATE            🔒 FROZEN sample = the first 15 R_NOMISS fills (by opened_at). Comparator = normal WILLY-class fills opened between the first
                and the 15th R_NOMISS fill (≥ 10 needed, else NO_COMPARATOR — no verdict). R_NOMISS avg pnl_percentage < normal avg − 0.30 pts → ALERT: review switching
                frenzy_willy_a_nomiss off. Else → KEEP (no re-fit of the threshold).
OBSERVE         decisions-export journal (e = WILLY) events ≥ the deploy: FRENZY_WILLY_A_NOMISS (allowed) · _MISSED (a red bar was already
                missed → refused as before) · _LATE (no red missed, the wait from the spike was over) · _OFF (eligible while the switch was
                off — counted only, not priced) · _VOL24_LOW (refused at the red bar by the volume floor).
Usage           venv/bin/python scripts/scout_willy_nomiss.py           (batch review; no network, no bot API)
"""
import os
import sys
import time

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_R = os.path.dirname(_D)
_R in sys.path or sys.path.insert(0, _R)
import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402
import willy_exit_era as WEra                                # noqa: E402  🎲² (DECISION_LOG 356) exit-era split / header

DL = 350
GATE_N = 15                                                  # frozen first-N R_NOMISS fills
GATE_GAP = 0.30                                              # pts below the same-period normal WILLY avg → review
MIN_COMPARATOR = 10                                          # normal WILLY fills needed in the window, else NO_COMPARATOR (no verdict)
WILLY_CLASS = ("FRENZY_WILLY", "FRENZY_WILLY_HT")
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
DEC_PATTERNS = (os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"), os.path.join(_R, "reports", "BASELINE*_decisions.csv"))
COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "pnl", "pnl_percentage", "close_reason", "is_paper",
        "entry_frenzy_willy_nomiss") + tuple(WEra.V2_COLS)     # DL 356: the V2 stamps ride the reader
EVENTS = ("FRENZY_WILLY_A_NOMISS", "FRENZY_WILLY_A_NOMISS_MISSED", "FRENZY_WILLY_A_NOMISS_LATE", "FRENZY_WILLY_A_NOMISS_OFF",
          "FRENZY_WILLY_A_NOMISS_VOL24_LOW")


def _ts(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:19].str.replace(" ", "T"), errors="coerce", format="%Y-%m-%dT%H:%M:%S")


def _truthy(s):
    return pd.Series(s).astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def load(paths=None):
    """every export row (paper + live), dedup (opened_at, pair, direction) — a CLOSED row wins, then the newest export."""
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    fr = []
    for f in paths:
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=list(COLS) + ["o", "pct", "usd", "es", "nomiss"])
    a = pd.concat(fr, ignore_index=True)
    for c in COLS:
        if c not in a:
            a[c] = None
    a = a.assign(_c=(a.status.astype(str) == "CLOSED").astype(int)).reset_index(drop=True)
    a = a.assign(_i=range(len(a))).sort_values(["_c", "_i"]).drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a["o"] = _ts(a.opened_at).values
    a["pct"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a["usd"] = pd.to_numeric(a.pnl, errors="coerce")
    a["es"] = a.entry_strategy.astype(str)
    a["nomiss"] = _truthy(a.entry_frenzy_willy_nomiss).values
    return a[a.o.notna()].drop(columns=["_c", "_i"]).sort_values("o").reset_index(drop=True)


def cohorts(a, since):
    """→ (R_NOMISS kept, normal WILLY-class kept, operator-closed) — CLOSED, opened ≥ since, operator-closed out. DL 356: the gate (DL 350)
    was registered on the OLD WILLY exit → V2 runner / floor fills are kept OUT of both sides (listed by run)."""
    c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since) & a.es.isin(WILLY_CLASS)] if len(a) else a
    if not len(c):
        return c, c, c
    kept, ox = _GK.split_operator_closed(c)
    kept = kept[kept.pct.notna()]
    kept, _ = WEra.split(kept)
    return kept[kept.nomiss].sort_values("o"), kept[~kept.nomiss].sort_values("o"), ox


def gate(nm, normal):
    """🔒 the frozen gate → (verdict, text). nm / normal = frames with o (opened_at) and pct. Never raises."""
    try:
        n = len(nm)
        if n < GATE_N:
            return "COLLECTING", f"⏳ {n}/{GATE_N} R_NOMISS fills closed (alert-only · DECISION_LOG {DL})"
        s = nm.sort_values("o").head(GATE_N)
        t0, t1 = s.o.iloc[0], s.o.iloc[-1]
        w = normal[(normal.o >= t0) & (normal.o <= t1)]
        mavg = float(s.pct.mean())
        if len(w) < MIN_COMPARATOR:
            return "NO_COMPARATOR", (f"📋 first {GATE_N} R_NOMISS avg {mavg:+.3f} % — comparator too thin: {len(w)} < {MIN_COMPARATOR} normal "
                                     f"WILLY fills in the same period ({t0:%m-%d} → {t1:%m-%d}); no verdict, re-read when it fills")
        wavg = float(w.pct.mean())
        head = (f"📋 first {GATE_N} R_NOMISS avg {mavg:+.3f} % vs normal WILLY same period {wavg:+.3f} % ({len(w)} fills, "
                f"{t0:%m-%d} → {t1:%m-%d})")
        if mavg < wavg - GATE_GAP:
            return "ALERT", head + f" → 🚨 below WILLY − {GATE_GAP:.2f}: review switching frenzy_willy_a_nomiss OFF"
        return "KEEP", head + f" → within WILLY − {GATE_GAP:.2f}: keep (frozen, never re-fit)"
    except Exception as e:
        return "ERROR", f"gate unreadable ({str(e)[:80]})"


def _line(name, g):
    if not len(g):
        return f"- {name}: N 0"
    return (f"- {name}: N {len(g)} · {g.o.dt.date.nunique()} days · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · "
            f"Σ {g.pct.sum():+.2f} pts · ${g.usd.sum():+.2f}")


def read_events(paths=None, since=None):
    """{event: (N, days)} of the e = WILLY R_NOMISS journal events ≥ since — dedup (t, pair, gate) across the decisions exports."""
    if paths is None:
        paths = sorted({p for pat in DEC_PATTERNS for p in EG.glob_exports(pat)}, key=os.path.getmtime)   # 🗄 (409) + exports_archive/
    fr = []
    for f in paths:
        try:
            x = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "gate"))
        except Exception:
            continue
        if {"e", "gate", "t"} <= set(x.columns):
            fr.append(x[(x.e.astype(str) == "WILLY") & x.gate.astype(str).isin(EVENTS)])
    out = {k: (0, 0) for k in EVENTS}
    if not fr:
        return out
    d = pd.concat(fr, ignore_index=True).drop_duplicates(["t", "pair", "gate"])
    d["ts"] = _ts(d.t).values
    if since is not None:
        d = d[d.ts >= since]
    for k in EVENTS:
        g = d[d.gate.astype(str) == k]
        out[k] = (len(g), g.ts.dt.date.nunique() if len(g) else 0)
    return out


def run(now_ms=None, paths=None, dec_paths=None, since_ms=None):
    L = [f"## 🧭 WILLY A R_NOMISS — late read, no red bar missed (DECISION_LOG {DL}, ALERT-only gate)", ""]
    if since_ms is None:
        since_ms = _GK.dl_deploy_ms(DL, lag_min=10, label="WILLY_NOMISS")
    if since_ms is None:
        return L + [f"⚠ NOT EVALUABLE — the DECISION_LOG {DL} deploy commit is not in git yet; nothing is counted.", ""]
    since = pd.Timestamp(int(since_ms), unit="ms")
    a = load(paths)
    nm, normal, ox = cohorts(a, since)
    L.append(f"Cohort: CLOSED WILLY-class fills opened ≥ {since:%Y-%m-%d %H:%M} UTC (deploy + 10 min) · {_GK.operator_closed_line(ox)}")
    _c = a[(a.status.astype(str) == "CLOSED") & (a.o >= since) & a.es.isin(WILLY_CLASS)] if len(a) else a
    _c = _GK.split_operator_closed(_c)[0] if len(_c) else _c
    L.append(WEra.era_line(_c, "WILLY_NOMISS"))                                # DL 356 (c)
    _vx = WEra.split(_c)[1] if len(_c) else _c
    if len(_vx):
        L.append(f"V2 runner / floor fills kept OUT of both sides (DECISION_LOG 356 — the gate was registered on the old +{WEra.OLD_TP:g} exit): "
                 + " · ".join(f"{r.o:%m-%d %H:%M} {r.pair}{' R_NOMISS' if r.nomiss else ''} {r.pct:+.2f} %" for r in _vx.itertuples()))
    L.append(_line("R_NOMISS (entry_frenzy_willy_nomiss)", nm))
    L.append(_line("normal WILLY (same period, WILLY + HT)", normal))
    v, txt = gate(nm, normal)
    L.append(f"Gate verdict: {v} — {txt}")
    ev = read_events(dec_paths, since)
    _n = lambda k: f"{ev[k][0]} ({ev[k][1]}d)"
    L.append(f"Journal events ≥ deploy: allowed (A_NOMISS) {_n('FRENZY_WILLY_A_NOMISS')} · refused: missed {_n('FRENZY_WILLY_A_NOMISS_MISSED')} / "
             f"late {_n('FRENZY_WILLY_A_NOMISS_LATE')} / vol24 {_n('FRENZY_WILLY_A_NOMISS_VOL24_LOW')} / off {_n('FRENZY_WILLY_A_NOMISS_OFF')}"
             " (off = eligible while the switch was off; counted, not priced)")
    return L + [""]


def main():
    print("\n".join(run(int(time.time() * 1000))))


if __name__ == "__main__":
    main()
