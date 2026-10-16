#!/usr/bin/env python3
"""👁 Scout — LITE_OB_THIN observe line (DECISION_LOG 302, operator 2026-10-16: "add the free watch line"). LONG SHOT, low prior.

Origin: reports/LITE_OB_2026-10-15.md + its adversarial review — the order book BEFORE the +1 touch is identical for dying vs running LITE
trades; the one post-touch "lead" (depth within 1 % shrinking between minute +1 and +2 after the touch) failed review (post-hoc mask, price
proxy within the dying group ρ 0.94, not ask-specific). Kept as ONE free line because the bot already records minute books (no engine change)
and the economic bar is low (flag precision ≥ 38 % vs a ~35 % base death rate among trades still open 2 min after the touch).

FROZEN DEFINITION (never re-fit)
  cohort   CLOSED FRENZY_LITE fills opened ≥ COHORT_START (orders exports, paper + live via export_glob), dedup (opened_at, pair, direction).
  touch    first 1m bar from the entry minute on whose HIGH ≥ entry × TOUCH_GROSS (+1 % net with taker fees) — public 1m klines.
           (known limitation, kept frozen: the entry-minute bar is included whole, so a high printed before the fill second counts.)
  frames   P1 / P2 = the minute BOOK snapshots (Decisions CSV rows e == "BOOK", taken ~20 s into the minute) labelled touch-minute + 1 / + 2.
  eligible the trade is still OPEN at the P2 snapshot (label + 20 s < closed_at) and both frames exist.
  feature  THIN = log10((bid$ + ask$ within 1 %) at P2 / at P1) < 0 ; also printed: ask-only change and the mid change P1→P2 (price proxy).
  outcome  DIED = closed ≤ −2.5 % · RAN = closed ≥ +1.9 % · else OTHER.
GATE (opens a rule STUDY, never a ship)  eligible N ≥ 20 on ≥ 8 UTC days ∧ precision (DIED among THIN) ≥ 38 % ∧ DIED-rate(THIN) −
  DIED-rate(not THIN) > 0 at day-bootstrap P ≥ 0.90. RETIRE at N 20 if DIED-rate(THIN) ≤ DIED-rate(not THIN).
DATA  the Decisions export carries only the last 3 days of BOOK rows (DB keeps 14) → keep the dashboard's auto Decisions download ON or
  download at least every 3 days. Binance: public klines only,
  ≤ 8 requests per run, weight-gated (stop at 900), shared scout cooldown honoured / armed on 418-429; cache reports/backtest_cache/scout_lite_ob/.
Usage  venv/bin/python scripts/scout_lite_ob_thin.py      (batch review; never changes config, never calls the bot API)
"""
import glob
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
import scout_b1h_negflank as NF                              # noqa: E402  (shared cooldown + day bootstrap helpers)

COHORT_START = "2026-10-16T00:00:00"
TOUCH_GROSS = 1.011
DIED, RAN = -2.5, 1.9
GATE_N, GATE_DAYS, PREC_MIN, BOOT_P = 20, 8, 0.38, 0.90
MAX_REQ, WEIGHT_STOP = 8, 900
CACHE = os.path.join(os.path.dirname(_D), "reports", "backtest_cache", "scout_lite_ob")
M1 = 60_000


def _ms(s):
    return int(pd.Timestamp(str(s)).tz_localize(None).value // 1_000_000)


def fills():
    fr = []
    for f in sorted(EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime):
        try:
            fr.append(pd.read_csv(f, low_memory=False, usecols=lambda c: c in (
                "opened_at", "closed_at", "pair", "direction", "status", "entry_strategy", "entry_price", "pnl_percentage")))
        except Exception:
            continue
    if not fr:
        return pd.DataFrame()
    a = pd.concat(fr, ignore_index=True)
    a = a[(a.entry_strategy.astype(str) == "FRENZY_LITE") & (a.opened_at.astype(str) >= COHORT_START)]
    a = a.assign(_c=(a.status == "CLOSED").astype(int)).sort_values("_c").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    return a[a.status == "CLOSED"].reset_index(drop=True)


def books():
    fr = []
    for f in EG.glob_exports(os.path.expanduser("~/Downloads/scalpars_decisions_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda c: c in ("t", "e", "pair", "ob_mid", "ob_bid_usd_1", "ob_ask_usd_1"))
            fr.append(d[d.e == "BOOK"])
        except Exception:
            continue
    if not fr:
        return pd.DataFrame(columns=["t", "pair", "ob_mid", "ob_bid_usd_1", "ob_ask_usd_1"])
    b = pd.concat(fr, ignore_index=True).drop_duplicates(["t", "pair"])
    if not {"ob_bid_usd_1", "ob_ask_usd_1", "ob_mid"} <= set(b.columns):
        return pd.DataFrame(columns=["t", "pair", "ob_mid", "ob_bid_usd_1", "ob_ask_usd_1"])
    b["ms"] = [_ms(x) for x in b.t]
    b = b.drop_duplicates(["pair", "ms"], keep="last")
    return b.set_index(["pair", "ms"]).sort_index()


class _Bud:
    n, used = 0, 0


def kl1m(pair, start, end, bud):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{pair}_{start}_{end}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    now = int(time.time() * 1000)
    if NF._cooldown_until() > now or bud.n >= MAX_REQ or bud.used >= WEIGHT_STOP:
        return None
    bud.n += 1
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1m&startTime={start}&endTime={end}&limit=1000"
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
    out = [[int(x[0]), float(x[2])] for x in rows]
    with open(p, "w") as f:
        json.dump(out, f)
    return out


def main():
    a = fills()
    print(f"LITE_OB_THIN (DECISION_LOG 302, long shot) · cohort FRENZY_LITE closed fills opened ≥ {COHORT_START} UTC → {len(a)}")
    if not len(a):
        return
    B, bud, rows, miss = books(), _Bud(), [], 0
    for r in a.to_dict("records"):
        t0, t1 = _ms(r["opened_at"]) // M1 * M1, _ms(r["closed_at"])
        k = kl1m(r["pair"], t0, t1 // M1 * M1, bud)
        if k is None:
            miss += 1
            continue
        hit = [t for t, h in k if h >= float(r["entry_price"]) * TOUCH_GROSS]
        if not hit:
            continue
        p1, p2 = hit[0] + M1, hit[0] + 2 * M1
        if p2 + 20_000 >= t1:
            continue                                         # not open at the P2 snapshot
        try:
            f1, f2 = B.loc[(r["pair"], p1)], B.loc[(r["pair"], p2)]
        except KeyError:
            miss += 1
            continue
        tot1, tot2 = f1.ob_bid_usd_1 + f1.ob_ask_usd_1, f2.ob_bid_usd_1 + f2.ob_ask_usd_1
        if not (tot1 > 0 and tot2 > 0):
            miss += 1
            continue
        pct = float(r["pnl_percentage"])
        rows.append(dict(day=str(r["opened_at"])[:10], pair=r["pair"], thin=np.log10(tot2 / tot1) < 0,
                         d_tot=np.log10(tot2 / tot1), d_ask=np.log10(f2.ob_ask_usd_1 / f1.ob_ask_usd_1) if f1.ob_ask_usd_1 > 0 else np.nan,
                         d_mid=(f2.ob_mid / f1.ob_mid - 1) * 100, out="DIED" if pct <= DIED else ("RAN" if pct >= RAN else "OTHER")))
    d = pd.DataFrame(rows)
    print(f"  eligible (touched +1, open at P2, both frames) {len(d)} · unscored {miss} (missing klines / books — keep the auto Decisions download on / download every ≤ 3 days)")
    if not len(d):
        return
    for th, g in d.groupby("thin"):
        print(f"  {'THIN' if th else 'not thin':9s} N {len(g):3d} · DIED {(g.out == 'DIED').sum()} · RAN {(g.out == 'RAN').sum()} · OTHER "
              f"{(g.out == 'OTHER').sum()} · median mid change P1→P2 {g.d_mid.median():+.2f} %")
    t, nt = d[d.thin], d[~d.thin]
    prec = (t.out == "DIED").mean() if len(t) else np.nan
    gap = prec - ((nt.out == "DIED").mean() if len(nt) else np.nan)
    p_pos = None
    if len(t) and len(nt) and d.day.nunique() >= 2:          # day-clustered bootstrap of the die-rate gap
        rng, days, gaps = np.random.default_rng(7), d.day.unique(), []
        by = {k: d[d.day == k] for k in days}
        for _ in range(4000):
            s = pd.concat([by[k] for k in rng.choice(days, len(days))])
            a1, a0 = s[s.thin], s[~s.thin]
            if len(a1) and len(a0):
                gaps.append((a1.out == "DIED").mean() - (a0.out == "DIED").mean())
        p_pos = float((np.array(gaps) > 0).mean()) if gaps else None
    n_ok, days = len(d) >= GATE_N, d.day.nunique()
    if n_ok and gap <= 0:
        verdict = "RETIRE (THIN does not die more often)"
    elif n_ok and days >= GATE_DAYS:
        if prec >= PREC_MIN and p_pos is not None and p_pos >= BOOT_P:
            verdict = "GATE MET → open a rule study (operator decides)"
        else:
            verdict = "keep observing (gate not met)"
    else:
        verdict = f"collecting ({len(d)}/{GATE_N} on {days}/{GATE_DAYS} days)"
    print(f"  precision {prec:.0%} (bar {PREC_MIN:.0%}) · die-rate gap {gap:+.0%} · P(gap>0) {p_pos if p_pos is None else round(p_pos, 2)} · {verdict}")


if __name__ == "__main__":
    main()
