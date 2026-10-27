#!/usr/bin/env python3
"""🧾🔒 Scout — STAMP COVERAGE per sleeve (DECISION_LOG 391, 2026-10-27 — operator: "make sure all sleeves have complete data … we can't
have strategies running without data again"). ALERT-only, never changes config.

CONTRACT   services.trading_engine.TradingEngine.STAMP_UNIVERSAL — the entry_* columns every bot fill must carry (when the source is
           readable); STAMP_BEST_EFFORT columns have an external source that can be missing for a small / new coin; STAMP_NULL_BY_DESIGN
           (column, sleeve) pairs are skipped. tests/test_entry_stamp_backfill.py fails on any unclassified entry_* column.
COHORT     bot fills (MANUAL out) opened ≥ the DL 391 deploy + 10 min (the DL 389 backfill + the 391 listing-age source both live;
           before them the sleeve paths each passed their own subset), every paper + live orders export, dedup (opened_at, pair, direction), the newest export wins.
           A fill opened < 2 min before its export may still be waiting for the background backfill → left out.
ALERT      per sleeve with N ≥ 3: any universal column below 90 % (best-effort below 60 %) → 🚨 named with its coverage; an entry_*
           column in an export that the contract does not classify → 🚨 (the model grew without the contract).
Usage      venv/bin/python scripts/scout_stamp_coverage.py
"""
import glob
import os
import sys

import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_D)
for _p in (_D, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import export_glob as EG                                     # noqa: E402
import scout_gatekit as _GK                                  # noqa: E402

DL, FLOOR_DL = 391, 391   # the cohort starts at THIS deploy (the 389 backfill + the 391 listing-age source both live)
PATTERN = os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")
MIN_N, BAR, BAR_BEST = 3, 0.90, 0.60
SETTLE_MS = 120_000
BATCH_START = "2026-10-23T18:30:00"                          # B25 (PAPER since 10-23 ~18:30 UTC) — move at each batch boundary                                          # the backfill runs 2–5 s after a fill; 2 min of margin before the export


def _contract():
    from services.trading_engine import TradingEngine as E
    return E


def load(paths=None):
    paths = paths if paths is not None else sorted(EG.glob_exports(PATTERN), key=os.path.getmtime)
    fr = []
    for i, f in enumerate(paths):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        _mt = None
        try:   # the export's own UTC instant from its filename (survives a copy / touch); mtime as the fallback
            _st = EG.export_stamp(f)
            _mt = (pd.Timestamp(_st) - pd.Timestamp(0)) / pd.Timedelta(milliseconds=1) if _st else os.path.getmtime(f) * 1000
        except Exception:
            _mt = None
        fr.append(d.assign(_i=i, _export_ms=_mt))
    if not fr:
        return pd.DataFrame()
    newest_cols = list(fr[-1].columns)
    a = pd.concat(fr, ignore_index=True)
    a = a.sort_values("_i", kind="stable").drop_duplicates(["opened_at", "pair", "direction"], keep="last").copy()
    a["o"] = pd.to_datetime(a.opened_at, errors="coerce", format="mixed")   # exports mix '…T…' and '… …' stamps
    a["es"] = (a["entry_strategy"] if "entry_strategy" in a else pd.Series(index=a.index, dtype=object)).fillna("MOMENTUM").astype(str)
    a = a[a.o.notna()].copy()
    a.attrs["newest_cols"] = newest_cols
    return a


def coverage(frame, since_ms, E=None, until_ms=None):
    """→ (rows [(sleeve, n, [(col, cov, bar)] below bar)], unclassified columns) for bot fills opened in [since, until). Pure over the frame."""
    E = E or _contract()
    unc = sorted(c for c in frame.columns if str(c).startswith("entry_") and E.stamp_class(c) == "unclassified")
    d = frame[(frame.es != "MANUAL") & (frame.o >= pd.Timestamp(int(since_ms), unit="ms"))]
    if "status" in d:   # real fills only: SIGNAL_EXPIRED rows (an aborted maker entry) and RECOVERED adoptions never ran the open path
        d = d[d.status.astype(str).isin(("OPEN", "CLOSED"))]
    if "confidence" in d:
        d = d[d.confidence.astype(str) != "RECOVERED"]
    if until_ms is not None:
        d = d[d.o < pd.Timestamp(int(until_ms), unit="ms")]
    if "_export_ms" in d:
        _o_ms = (d.o - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
        d = d[d._export_ms.isna() | (_o_ms <= d._export_ms - SETTLE_MS)]
    rows = []
    for es, g in d.groupby("es"):
        low = []
        for c in E.STAMP_UNIVERSAL:
            des = tuple(E.STAMP_NULL_BY_DESIGN.get(c) or ())
            if des and es.startswith(des):
                continue
            cov = float(g[c].notna().mean()) if c in g else 0.0
            bar = BAR_BEST if c in E.STAMP_BEST_EFFORT else BAR
            if cov < bar:
                low.append((c, cov, bar))
        rows.append((es, len(g), low))
    return rows, unc


def _lines(rows, E):
    out, alerts = [], 0
    for es, n, low in sorted(rows, key=lambda r: -r[1]):
        if n < MIN_N:
            out.append(f"- {es}: N {n} (< {MIN_N}) · " + ("complete" if not low else
                       "short so far: " + ", ".join(f"{c[6:]} {cov:.0%}" for c, cov, _ in low)))
            continue
        if low:
            alerts += 1
            out.append(f"- 🚨 {es}: N {n} · {len(low)} column(s) below the bar: " + ", ".join(f"{c[6:]} {cov:.0%}" for c, cov, _ in low))
        else:
            out.append(f"- ✅ {es}: N {n} · all {len(E.STAMP_UNIVERSAL)} universal columns at the bar")
    return out, alerts


def run(now_ms=None, paths=None, since_ms=None):
    L = [f"## 🧾🔒 STAMP COVERAGE per sleeve (DECISION_LOG {DL}, alert-only)", ""]
    try:
        E = _contract()
        now_ms = int(pd.Timestamp.now("UTC").value // 1_000_000) if now_ms is None else int(now_ms)
        if since_ms is None:
            since_ms = _GK.dl_deploy_ms(FLOOR_DL, lag_min=10, label="STAMP_BACKFILL")
        a = load(paths)
        if not len(a):
            return L + ["No orders export found.", ""]
        unc = sorted(c for c in a.attrs.get("newest_cols", a.columns)   # the newest export only: a column dropped from the model never alerts forever
                     if str(c).startswith("entry_") and E.stamp_class(c) == "unclassified")
        L.append(f"Contract: {len(E.STAMP_UNIVERSAL)} universal columns (bar {BAR:.0%}; best-effort {BAR_BEST:.0%}) · MANUAL out")
        if unc:
            L.append(f"🚨 UNCLASSIFIED entry_* column(s) in the export — the contract does not know them: {', '.join(unc)}")
        # ① every fill of the current batch BEFORE the fix — never hidden (operator 10-27): what each sleeve is missing, for the batch
        # review's W/L reads (those columns need a klines rebuild there; the live backfill only covers fills after the deploy)
        b0 = pd.Timestamp(BATCH_START)
        cut = since_ms if since_ms is not None else now_ms
        hist, _ = coverage(a, int((b0 - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)), E, until_ms=cut)
        L.append(f"① Current batch BEFORE the fix — fills {b0:%Y-%m-%d %H:%M} → {pd.Timestamp(int(cut), unit='ms'):%m-%d %H:%M} UTC, "
                 f"{sum(n for _, n, _ in hist)} bot fills (informational; these gaps need the klines rebuild at the batch review):")
        for es, n, low in sorted(hist, key=lambda r: -r[1]):
            L.append(f"- {es}: N {n} · " + ("complete" if not low else
                     f"{len(low)} column(s) short: " + ", ".join(f"{c[6:]} {cov:.0%}" for c, cov, _ in low)))
        # ② fills since the fix — the verdict
        if since_ms is None:
            L += [f"② Since the fix: ⚠ NOT EVALUABLE — the DECISION_LOG {FLOOR_DL} deploy commit is not in git yet.",
                  "Verdict: " + (f"🚨 ALERT — {len(unc)} unclassified column(s) (classify in TradingEngine.STAMP_*)" if unc else "⏳ not evaluable yet"), ""]
            return L
        rows, _ = coverage(a, since_ms, E)
        L.append(f"② Since the fix — bot fills opened ≥ {pd.Timestamp(int(since_ms), unit='ms'):%Y-%m-%d %H:%M} UTC (DL {FLOOR_DL} deploy + 10 min) — the verdict:")
        if not rows:
            L.append("No bot fill in the cohort yet.")
        out, alerts = _lines(rows, E)
        L += out
        _why = ([f"{alerts} sleeve(s) missing data (fix the path; [STAMP_COVERAGE] log lines name each fill)"] if alerts else []) \
            + ([f"{len(unc)} unclassified column(s) (classify in TradingEngine.STAMP_*)"] if unc else [])
        _ready = any(n >= MIN_N for _, n, _ in rows)
        L.append("Verdict: " + ("🚨 ALERT — " + " · ".join(_why) if _why else
                                ("✅ complete" if _ready else f"⏳ not evaluable yet (no sleeve at N ≥ {MIN_N})")))
        return L + [""]
    except Exception as e:
        return L + [f"Unavailable this run ({str(e)[:120]}).", ""]


def main():
    print("\n".join(run()))


if __name__ == "__main__":
    main()
