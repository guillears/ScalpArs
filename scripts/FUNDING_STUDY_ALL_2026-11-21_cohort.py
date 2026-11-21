#!/usr/bin/env python3
"""FUNDING_STUDY_ALL_2026-11-21 — real-fill cohort (master on today's stack + the current batch), every sleeve, + the funding-rate fetch.

Research only (no bot code / config / git touched).
  MASTER = reports/MASTER_POOL_stacked.csv stack_keep, CLOSED, full size (is_probe False) — except FRENZY_WILLY_HT and FLATBASE_SHORT, whose
           today's-filter lists are reports/KEPT_LIST_{WILLY_HT,FLATBASE}_2026-11-21.csv (stack_keep + the later DLs; HT probes are the
           sleeve's real size in the master era, so the HT list is used as is — memory "Today-filters = pool stack_keep").
  BATCH  = the current export (paper reset 11-18): every CLOSED bot fill; HT / FB filtered by the same kept lists (block BATCH).
  Dedup key (opened_at, pair, direction).
  price-only % on today's stack: stack_pct (master) / pnl_percentage (batch) MINUS the booked funding % when funding_in_pnl is True
  (DL 539 convention: from 11-13 pnl includes funding) — so every fill carries price-only % + the MODELLED funding exactly once.
Funding rates: /fapi/v1/fundingRate (real settlement rows) per pair over the fills' windows, cached in reports/.funding_cache/rates_<PAIR>.csv.
Budget: fundingRate has its own 500 / 5 min IP limit (no weight header) → ≤ 1 call / 0.7 s, hard cap 400 calls; stop on 418 / 429 or an
active scripts/binance_cooldown.py cooldown.
Output: reports/FUNDING_STUDY_ALL_2026-11-21_data/real_cohort.csv
Usage: venv/bin/python -I scripts/FUNDING_STUDY_ALL_2026-11-21_cohort.py [--fetch]"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import binance_cooldown as BC  # noqa: E402

RP = os.path.join(ROOT, "reports")
OUT = os.path.join(RP, "FUNDING_STUDY_ALL_2026-11-21_data")
CACHE = os.path.join(RP, ".funding_cache")
EXPORT = next((p for p in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports_archive", "scalpars_orders_paper_2026-11-21_22-18-17.csv"), os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-21_22-18-17.csv")) if os.path.exists(p)), os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-21_22-18-17.csv"))
COLS = ["pair", "direction", "opened_at", "closed_at", "entry_price", "quantity", "partial_closed_qty", "investment", "leverage",
        "pnl", "pnl_percentage", "funding_fee_usd", "funding_in_pnl", "entry_funding_rate", "close_reason", "peak_pnl", "trough_pnl",
        "entry_strategy", "is_probe", "entry_exh_atr_pct", "entry_exh_tp_pct", "entry_exh_sl_pct", "entry_exh_universe", "exit_price"]


def sleeve_of(s):
    s = str(s)
    return "FLIP" if s.startswith("FLIP") else s


def kept_keys(fn):
    k = pd.read_csv(os.path.join(RP, fn))
    return {(p, pd.Timestamp(t).floor("s")) for p, t in zip(k.pair, k.k)}


def build():
    m = pd.read_csv(os.path.join(RP, "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[[c for c in COLS + ["stack_keep", "stack_pct", "era"] if c in m.columns]].copy()
    m["block"] = "MASTER"
    b = pd.read_csv(EXPORT, low_memory=False)
    b = b[[c for c in COLS if c in b.columns]].copy()
    b = b[b.status.eq("CLOSED")] if "status" in b.columns else b
    b["block"] = "BATCH"; b["era"] = "B28"; b["stack_keep"] = True; b["stack_pct"] = b.pnl_percentage
    X = pd.concat([m, b], ignore_index=True)
    X["sleeve"] = X.entry_strategy.map(sleeve_of)
    X["o"] = pd.to_datetime(X.opened_at, format="mixed").dt.floor("s")
    X["c"] = pd.to_datetime(X.closed_at, format="mixed")
    X = X.drop_duplicates(["o", "pair", "direction"], keep="last")
    ht, fb = kept_keys("KEPT_LIST_WILLY_HT_2026-11-21.csv"), kept_keys("KEPT_LIST_FLATBASE_2026-11-21.csv")
    key = list(zip(X.pair, X.o))
    keep = X.stack_keep.eq(True) & ~X.is_probe.fillna(False).astype(bool)
    keep = np.where(X.sleeve.eq("FRENZY_WILLY_HT"), [k in ht for k in key], keep)
    keep = np.where(X.sleeve.eq("FLATBASE_SHORT"), [k in fb for k in key], keep)
    X = X[keep & X.c.notna() & ~X.sleeve.isin(["MANUAL", "nan"])].copy()
    X["notional"] = X.entry_price * (X.quantity + X.partial_closed_qty.fillna(0))
    fb_pct = np.where(X.funding_in_pnl.eq(True), X.funding_fee_usd.fillna(0) / X.notional * 100, 0.0)
    X["booked_fund_pct"] = np.where(X.funding_in_pnl.eq(True), fb_pct, np.nan)
    X["price_pct"] = X.stack_pct - fb_pct
    X["hold_h"] = (X.c - X.o).dt.total_seconds() / 3600
    X["sgn"] = np.where(X.direction.str.upper().eq("SHORT"), 1.0, -1.0)   # a short RECEIVES a positive rate
    return X.reset_index(drop=True)


_n = {"n": 0}


def get(url):
    if BC.active():
        sys.exit("binance cooldown active — stop")
    if _n["n"] >= 400:
        sys.exit("call cap reached — stop")
    req = urllib.request.Request(url, headers={"User-Agent": "research-client/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            sys.exit(f"HTTP {e.code} — stop, not retrying")
        if e.code == 400:
            return []
        raise
    _n["n"] += 1
    time.sleep(0.7)
    return data


def rates(pair):
    f = os.path.join(CACHE, f"rates_{pair}.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def fetch(X):
    os.makedirs(CACHE, exist_ok=True)
    for p, g in X.groupby("pair"):
        a = int((g.o.min() - pd.Timedelta("9h")).value // 10**6); z = int((g.c.max() + pd.Timedelta("1h")).value // 10**6)
        old = rates(p)
        if old is not None and len(old) and old.attrs.get("x") is None:
            meta = os.path.join(CACHE, f"rates_{p}.span")
            if os.path.exists(meta):
                s0, s1 = map(int, open(meta).read().split())
                if s0 <= a and s1 >= z:
                    continue
        rows, s = [], a
        while s < z:
            d = get(f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={p}&startTime={s}&endTime={z}&limit=1000")
            if not d:
                break
            rows += [dict(t=int(j["fundingTime"]), rate=float(j["fundingRate"]),
                          mark=float(j["markPrice"]) if j.get("markPrice") not in (None, "") else np.nan) for j in d]
            s = int(d[-1]["fundingTime"]) + 1
            if len(d) < 1000:
                break
        pd.DataFrame(rows, columns=["t", "rate", "mark"]).drop_duplicates("t").to_csv(os.path.join(CACHE, f"rates_{p}.csv"), index=False)
        open(os.path.join(CACHE, f"rates_{p}.span"), "w").write(f"{a} {z}")
        print(p, len(rows), "calls so far", _n["n"], flush=True)


def price(X):
    """modelled funding per fill: Σ over settlements s with open < s ≤ close of sgn × rate × notional (entry notional; mark ratio applied
    when the settlement row carries a mark price: notional × mark / entry). Also entry-time facts (lag-safe: the last settled rate ≤ open)."""
    out = []
    for r in X.itertuples():
        R = rates(r.pair)
        d = dict(mod_fund_usd=np.nan, mod_fund_pct=np.nan, n_settle=0, interval_h=np.nan, last_rate=np.nan, next_rate=np.nan,
                 max_paid_rate=np.nan)
        if R is not None:
            t = R.t.values.astype("int64"); rt = R.rate.values.astype(float); mk = R.mark.values.astype(float)
            o_ms = int(r.o.value // 10**6); c_ms = int(r.c.value // 10**6)
            k = (t > o_ms) & (t <= c_ms)
            ratio = np.where(np.isfinite(mk[k]) & (mk[k] > 0), mk[k] / r.entry_price, 1.0)
            pct = (r.sgn * rt[k] * ratio * 100).sum()
            d.update(mod_fund_pct=pct, mod_fund_usd=pct / 100 * r.notional, n_settle=int(k.sum()))
            j = int(np.searchsorted(t, o_ms, "right"))
            if j >= 1:
                d["last_rate"] = rt[j - 1]
            if j >= 2:
                d["interval_h"] = round((t[j - 1] - t[j - 2]) / 3.6e6)
            if j < len(t):
                d["next_rate"] = rt[j]
            if k.any():
                d["max_paid_rate"] = float(np.max(-r.sgn * rt[k]))
        out.append(d)
    return pd.concat([X, pd.DataFrame(out, index=X.index)], axis=1)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    X = build()
    if "--fetch" in sys.argv:
        fetch(X)
    X = price(X)
    X.to_csv(os.path.join(OUT, "real_cohort.csv"), index=False)
    print(X.groupby(["block", "sleeve"]).size().to_string())
