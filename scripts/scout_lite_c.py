#!/usr/bin/env python3
"""🎯 Scout — LITE C revert-gate tracker (DECISION_LOG 301, operator ARMED 2026-10-15, discipline override below the locked gates).

RULE (engine)   every FRENZY_LITE entry decides its TP once at entry: ZONE = entry_frenzy_vol_mult ≥ 20 ∧ entry_btc_1h_slope > 0 → TP +3
                (frenzy_lite_tp_pct); everything else (missing stamps included) → TP +2 (frenzy_lite_c_tp_pct). Stop −3 / 12 h unchanged.
COHORT          CLOSED FRENZY_LITE fills that STORED a LITE C TP (frenzy_lite_tp_used > 0 = the switch was ON at entry), paper + live exports (export_glob), dedup
                (opened_at, pair, direction), newest export wins. Only OUTSIDE-zone fills can differ from the old +3/−3 exit.
COUNTERFACTUAL  an outside fill that did NOT close at its +2 TP never reached +2, so it never reached +3 either → Δ = 0 (no fetch).
                An outside fill that closed at the +2 TP is walked on 1m klines from its close minute to entry + 12 h under +3 / −3:
                high ≥ entry × 1.0302 first → CAPPED (old exit +3.0) · low ≤ entry × 0.97 first (same minute = stop first, conservative
                against C) → SAVED (old exit −3.0) · neither → the 12 h close. Δ = actual − old. The tally is Σ Δ (pct points, 1×).
REVERT GATE     (pre-committed, never moved — CURRENT_STATE pin)
                · EARLY KILL any time: Σ Δ ≤ −4 points → set frenzy_lite_c_enabled false.
                · FIRST READ at 10 outside fills: Σ Δ ≤ 0 → revert.
                · MAIN READ at 20 outside fills on ≥ 8 UTC days: Σ Δ ≤ 0, OR one day / one pair carries ≥ 50 % of the positive Δ → revert;
                  else KEEP. Switching off does not touch open +2 orders (they run out at +2); new fills go back to +3.
BINANCE         public fapi/v1/klines only; cache-first (reports/backtest_cache/scout_lite_c/); ≤ 6 requests per run, weight-gated
                (stop once X-MBX-USED-WEIGHT-1M ≥ 900), the SHARED scout cooldown file is honoured and armed on 418 / 429.
Usage           venv/bin/python scripts/scout_lite_c.py            (batch review; never changes config, never calls the bot API)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import export_glob as EG                                     # noqa: E402
import scout_b1h_negflank as NF                              # noqa: E402  (shared cooldown helpers)
import scout_gatekit as _GK                                  # noqa: E402  🧰 (DECISION_LOG 331) operator-closed fills out of the gate

COHORT_START = "2026-10-15T00:00:00"                         # plus: only fills that STORED a LITE C TP (frenzy_lite_tp_used set = switch ON at entry)
VOL_CUT, SLOPE_CUT, TP_HI, TP_LO, STOP = 20.0, 0.0, 3.0, 2.0, 3.0
TP_HI_GROSS = 1.0302                                         # +3 net with the maker fee (resting TP), conservative
HOLD_MS, M1 = 12 * 3600_000, 60_000
MAX_REQ, WEIGHT_STOP = 6, 900
EARLY_KILL, FIRST_N, MAIN_N, MAIN_DAYS = -4.0, 10, 20, 8
CACHE = os.path.join(os.path.dirname(_D), "reports", "backtest_cache", "scout_lite_c")
COLS = ("opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "pnl_percentage", "close_reason",
        "peak_pnl", "entry_frenzy_vol_mult", "entry_btc_1h_slope", "frenzy_lite_tp_used", "lite_c_zone", "is_paper")


def _ms(s):
    return int(pd.Timestamp(str(s)).tz_localize(None).value // 1_000_000)


def fills():
    fr = []
    for f in sorted(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in COLS))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a = a[(a.entry_strategy.astype(str) == "FRENZY_LITE") & (a.opened_at.astype(str) >= COHORT_START)]
    a = a.assign(_c=(a.status == "CLOSED").astype(int)).sort_values("_c").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a = a[a.status == "CLOSED"]
    if "frenzy_lite_tp_used" not in a.columns:
        return a.iloc[0:0]
    return a[pd.to_numeric(a.frenzy_lite_tp_used, errors="coerce") > 0].reset_index(drop=True)   # switch-off / pre-301 fills never count


def in_zone(r):
    """the stored engine decision when present, else re-derived from the stamps (missing → outside, like the engine)."""
    tp = pd.to_numeric(pd.Series([r.get("frenzy_lite_tp_used")]), errors="coerce").iloc[0]
    if pd.notna(tp):
        return bool(tp >= TP_HI - 1e-9)
    v = pd.to_numeric(pd.Series([r.get("entry_frenzy_vol_mult")]), errors="coerce").iloc[0]
    s = pd.to_numeric(pd.Series([r.get("entry_btc_1h_slope")]), errors="coerce").iloc[0]
    return bool(pd.notna(v) and pd.notna(s) and v >= VOL_CUT and s > SLOPE_CUT)


class _Budget:
    n, used = 0, 0


def klines_1m(pair, start_ms, end_ms, bud):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{pair}_{start_ms}_{end_ms}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    now = int(time.time() * 1000)
    if end_ms > now - M1:
        return None                                          # window not finished yet → read next time
    if NF._cooldown_until() > now or bud.n >= MAX_REQ or bud.used >= WEIGHT_STOP:
        return None
    url = (f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1m&startTime={int(start_ms)}"
           f"&endTime={int(end_ms)}&limit=1000")
    bud.n += 1
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            rows = json.loads(r.read().decode())
            bud.used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            NF._arm_cooldown(e.code, (e.headers or {}).get("Retry-After"), now)
            bud.n = MAX_REQ
        return None
    except Exception:
        return None
    out = [[int(x[0]), float(x[2]), float(x[3]), float(x[4])] for x in rows]
    with open(path, "w") as f:
        json.dump(out, f)
    return out


def walk(bars, entry):
    """+3 / −3 from the +2 close onward → the old exit's result (pct, 1×)."""
    for _, h, l, c in bars:
        if l <= entry * (1 - STOP / 100):
            return -STOP, "SAVED"
        if h >= entry * TP_HI_GROSS:
            return TP_HI, "CAPPED"
    return (bars[-1][3] / entry - 1) * 100 if bars else None, "TIMEOUT"


def main():
    a = fills()
    a, opx = _GK.split_operator_closed(a) if len(a) else (a, a)   # 🧰 (331) operator-closed (MANUAL / EXTERNAL_CLOSE): not the bot's exit
    print("✅ RESOLVED 2026-10-30 (DECISION_LOG 420): this gate fired (EARLY KILL) and LITE C is off. Frozen history only; the live read is scripts/scout_lite_c_revert_watch.py.")
    print(f"LITE C revert gate (DECISION_LOG 301) · cohort FRENZY_LITE closed fills opened ≥ {COHORT_START} UTC → {len(a)}"
          f" · {_GK.operator_closed_line(opx)}")
    if not len(a):
        return
    a["zone"] = [in_zone(r) for r in a.to_dict("records")]
    a["pct"] = pd.to_numeric(a.pnl_percentage, errors="coerce")
    a["day"] = a.opened_at.astype(str).str[:10]
    for z, g in a.groupby("zone"):
        print(f"  {'ZONE (+3)' if z else 'OUTSIDE (+2)':13s} N {len(g):3d} · WR {(g.pct > 0).mean():.0%} · avg {g.pct.mean():+.3f} % · Σ {g.pct.sum():+.2f}")
    o = a[~a.zone].copy()
    bud, rows, pending = _Budget(), [], 0
    for r in o.to_dict("records"):
        tp_close = str(r.get("close_reason") or "").upper().startswith("FRENZY_TP") or r["pct"] >= TP_LO - 0.05
        if not tp_close:
            rows.append((r, 0.0, "SAME"))
            continue
        t0, t_open = _ms(r["closed_at"]) // M1 * M1, _ms(r["opened_at"])
        bars = klines_1m(r["pair"], t0, t_open + HOLD_MS, bud)
        if bars is None:
            pending += 1
            continue
        old, kind = walk(bars, float(r["entry_price"]))
        rows.append((r, (r["pct"] - old) if old is not None else 0.0, kind))
    if not rows:
        print(f"  outside fills priced 0 · pending {pending}")
        return
    d = pd.DataFrame([dict(day=r["day"], pair=r["pair"], delta=dl, kind=k) for r, dl, k in rows])
    tot = d.delta.sum()
    kinds = d.kind.value_counts().to_dict()
    print(f"  OUTSIDE priced {len(d)} (pending {pending}) on {d.day.nunique()} days · saved {kinds.get('SAVED', 0)} · capped "
          f"{kinds.get('CAPPED', 0)} · timeout {kinds.get('TIMEOUT', 0)} · same {kinds.get('SAME', 0)} · Σ Δ vs +3/−3 = {tot:+.2f} pts")
    pos = d[d.delta > 0]
    share = max(pos.groupby("day").delta.sum().max(), pos.groupby("pair").delta.sum().max()) / pos.delta.sum() if len(pos) else 0.0
    if tot <= EARLY_KILL:
        verdict = "🔴 EARLY KILL — revert (frenzy_lite_c_enabled false)"
    elif len(d) >= MAIN_N and d.day.nunique() >= MAIN_DAYS:
        verdict = "🔴 MAIN READ — revert" if (tot <= 0 or share >= 0.5) else "🟢 MAIN READ — keep"
    elif len(d) >= FIRST_N:
        verdict = "🔴 FIRST READ — revert" if tot <= 0 else f"🟢 FIRST READ passed — collecting to {MAIN_N} on {MAIN_DAYS} days"
    else:
        verdict = f"collecting ({len(d)}/{FIRST_N} for the first read)"
    print(f"  top day/pair share of the gain {share:.0%} · verdict: {verdict}")


if __name__ == "__main__":
    main()
