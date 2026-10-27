#!/usr/bin/env python3
"""🧾⏪ PARITY of the stamp rebuild (DECISION_LOG 393) — before ANY historical stamp is written (operator: "300 % sure … 0 errors"),
rebuild every stamp of fills that ALREADY carry the live value and compare column by column.

COHORT   real bot fills (OPEN / CLOSED, MANUAL out) opened 2026-10-16 → now from the B22–B24 archives + every ~/Downloads paper export,
         dedup (opened_at, pair, direction), newest export wins. Each column is compared on the fills where the live value exists.
VARIANTS the forming 5m bar is the issue (live = a PARTIAL bar at decision time; history = the finished bar):
         FULL   = the finished 5m bar (what plain history shows)
         PART   = the forming bar rebuilt from the CLOSED 1m bars before the fill + the fill minute's 1m bar as of its OPEN (o=h=l=c=open)
         PARTC  = the forming bar from 1m bars with open ≤ the fill (the fill minute's full 1m bar included)
         The fill price replaces `price` in every variant (as the live paths stamp it).
OUTPUT   reports/STAMP_REBUILD_PARITY_2026-10-27.csv (per fill × column × variant: live, rebuilt, Δ) + _summary.csv (per column × variant:
         N · exact (|Δ| ≤ the column's rounding unit) · within 2 units · median |Δ| · p95 |Δ| · max |Δ|).
NETWORK  public fapi klines / fundingRate only (scout_ml_regime_lines Budget: paced, weight-aware, stops on 418 / 429, shared cooldown);
         every response cached under reports/backtest_cache/stamp_rebuild/.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_D)
for _p in (_D, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///./_x.db")

import scout_ml_regime_lines as _MRL                         # noqa: E402

CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "stamp_rebuild")
OUT = os.path.join(_ROOT, "reports", "STAMP_REBUILD_PARITY_2026-10-27")
FAPI = "https://fapi.binance.com"
SINCE = "2026-10-16"
M1, M5 = 60_000, 300_000

# rounding unit of each live stamp (the column's own rounding in the engine) → "exact" = |Δ| ≤ unit
UNIT = {"entry_rsi": 0.01, "entry_rsi_prev": 0.01, "entry_adx": 1e-4, "entry_adx_prev": 1e-4, "entry_adx_delta": 1e-4,
        "entry_gap": 1e-4, "entry_ema_gap_5_8": 1e-4, "entry_ema_gap_8_13": 1e-4, "entry_ema5_stretch": 1e-4, "entry_price_vs_ema5_pct": 1e-4,
        "entry_ema20_slope": 1e-4, "entry_ema50_slope": 1e-4, "entry_range_position": 0.1, "entry_pos_di": 1e-4, "entry_neg_di": 1e-4,
        "entry_pair_ema20_ema50_gap_pct": 1e-4, "entry_dist_from_ema13_pct": 1e-4, "entry_pair_volume_ratio": 1e-4,
        "entry_gap_5_20_signed_pct": 1e-4, "entry_gap_5_20_prev_signed_pct": 1e-4, "entry_gap_5_8_signed_pct": 1e-4,
        "entry_gap_expand_marginal": 1e-4, "entry_atr_pct": 1e-4,
        "entry_btc_adx": 1e-4, "entry_btc_adx_prev": 1e-4, "entry_btc_rsi": 0.1, "entry_btc_rsi_prev": 0.1, "entry_btc_rsi_prev6": 0.1,
        "entry_btc_atr_pct": 1e-4, "entry_btc_ema20_slope": 1e-4, "entry_btc_trend_gap_pct": 1e-4, "entry_btc_dist_from_ema13_pct": 1e-4,
        "entry_btc_price": 0.1, "entry_btc_1h_slope": 1e-4, "entry_btc_rsi_1h": 0.1, "entry_btc_rsi_1h_prev": 0.1,
        "entry_btc_4h_ema50_200_gap_pct": 1e-4, "entry_pair_1d_ndi": 1e-2, "entry_funding_rate": 1e-6, "entry_pair_age_days": 0.1,
        "entry_frenzy_di_spread": 1e-2, "entry_frenzy_adx_delta": 1e-2, "entry_frenzy_vol_trend": 1e-2}


def _cached(key, url, bud):
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, key + ".json")
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    d = _MRL._get(url, bud)
    if d is not None:
        with open(p, "w") as fh:
            json.dump(d, fh)
    return d


def klines(sym, tf, end_ms, limit, bud):
    d = _cached(f"{sym}_{tf}_{end_ms}_{limit}", f"{FAPI}/fapi/v1/klines?symbol={sym}&interval={tf}&endTime={int(end_ms)}&limit={limit}", bud)
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])] for r in (d or [])]


def bulk(sym, tf, t0, t1, step_ms, bud):
    """all bars of tf in [t0, t1] (1500 per call, cached per call)."""
    out, end = {}, int(t1)
    while end >= t0:
        rows = klines(sym, tf, end, 1500, bud)
        if not rows:
            break
        for r in rows:
            out[r[0]] = r
        end = rows[0][0] - 1
        if bud.stop:
            break
    return [out[k] for k in sorted(out) if k <= t1]


def funding_hist(sym, t_ms, bud):
    d = _cached(f"{sym}_funding_{t_ms // 86_400_000}", f"{FAPI}/fapi/v1/fundingRate?symbol={sym}&startTime={int(t_ms) - 3 * 86_400_000}"
                f"&endTime={int(t_ms) + 86_400_000}&limit=1000", bud)
    return [(int(r["fundingTime"]), float(r["fundingRate"])) for r in (d or [])]


def forming(k5, k1, fill_ms, mode):
    """the 5m window with its LAST bar replaced by the partial bar built from 1m data (mode PART / PARTC); FULL → unchanged."""
    if mode == "FULL" or not k5:
        return k5
    b0 = k5[-1][0]
    if mode == "PART":
        m = [r for r in k1 if b0 <= r[0] and r[0] + M1 <= fill_ms]
        cur = [r for r in k1 if r[0] <= fill_ms < r[0] + M1]
        if cur:
            o = cur[0][1]
            m = m + [[cur[0][0], o, o, o, o, 0.0]]
    else:
        m = [r for r in k1 if b0 <= r[0] <= fill_ms]
    if not m:
        return k5
    bar = [b0, m[0][1], max(r[2] for r in m), min(r[3] for r in m), m[-1][4], sum(r[5] for r in m)]
    return k5[:-1] + [bar]


def load_fills():
    from services.trading_engine import TradingEngine as E
    fs = sorted(glob.glob(os.path.join(_ROOT, "reports", "BASELINE2[2-4]*_orders.csv"))) + \
        sorted(glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")), key=os.path.getmtime)
    fr = []
    for i, f in enumerate(fs):
        try:
            fr.append(pd.read_csv(f, low_memory=False).assign(_i=i))
        except Exception:
            pass
    a = pd.concat(fr, ignore_index=True).sort_values("_i", kind="stable").drop_duplicates(["opened_at", "pair", "direction"], keep="last")
    a["o"] = pd.to_datetime(a.opened_at, format="mixed", errors="coerce")
    a = a[(a.o >= pd.Timestamp(SINCE)) & a.status.astype(str).isin(("OPEN", "CLOSED"))]
    a = a[a.entry_strategy.fillna("MOMENTUM").astype(str) != "MANUAL"].copy()
    a["es"] = a.entry_strategy.fillna("MOMENTUM").astype(str)
    a["fill_ms"] = ((a.o - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)).astype("int64")
    return a.sort_values("o").reset_index(drop=True)


def main():
    from services.stamp_rebuild import rebuild_stamps
    bud = _MRL.Budget(max_req=1500)
    a = load_fills()
    t0, t1 = int(a.fill_ms.min()) - 200 * 3_600_000, int(a.fill_ms.max()) + M5
    btc5 = bulk("BTCUSDT", "5m", t0, t1, M5, bud)
    btc1m = bulk("BTCUSDT", "1m", int(a.fill_ms.min()) - 600_000, t1, M1, bud)
    btc1h = bulk("BTCUSDT", "1h", t0 - 100 * 3_600_000, t1, 3_600_000, bud)
    btc4h = bulk("BTCUSDT", "4h", t0 - 1000 * 14_400_000, t1, 14_400_000, bud)
    rows = []
    for r in a.itertuples():
        if bud.stop:
            print("STOP:", bud.stop)
            break
        sym = r.pair
        fm = int(r.fill_ms)
        p5 = klines(sym, "5m", fm, 400, bud)
        p1 = klines(sym, "1m", fm, 10, bud)
        p1d = klines(sym, "1d", fm, 200, bud)
        fr = funding_hist(sym, fm, bud)
        b5 = [b for b in btc5 if b[0] <= fm][-100:]
        b1 = [b for b in btc1m if b[0] <= fm]
        for mode in ("FULL", "PART", "PARTC"):
            pw = forming([b for b in p5 if b[0] <= fm][-100:], p1, fm, mode)
            bw = forming(b5, b1, fm, mode)
            st = rebuild_stamps(r.direction, fm, float(r.entry_price), pair_k5=pw, btc_k5=bw,
                                btc_k1h=btc1h, btc_k4h=btc4h, pair_k1d=p1d, funding=fr, onboard_ms=None,
                                frenzy=str(r.es).startswith(("FRENZY_", "FLATBASE")))
            if str(r.es).startswith(("FRENZY_", "FLATBASE")):   # the FRENZY stamps read CLOSED bars — the full 400-bar history
                st2 = rebuild_stamps(r.direction, fm, float(r.entry_price), pair_k5=p5, frenzy=True)
                for c in ("entry_frenzy_di_spread", "entry_frenzy_adx_delta", "entry_frenzy_vol_trend"):
                    if c in st2:
                        st[c] = st2[c]
            for c, unit in UNIT.items():
                lv = getattr(r, c, None)
                if lv is None or (isinstance(lv, float) and np.isnan(lv)) or c not in st:
                    continue
                rows.append(dict(pair=sym, o=r.o, es=r.es, mode=mode, col=c, live=float(lv), rebuilt=float(st[c]),
                                 d=abs(float(st[c]) - float(lv)), unit=unit))
    d = pd.DataFrame(rows)
    d.to_csv(OUT + ".csv", index=False)
    g = d.groupby(["col", "mode"])
    s = pd.DataFrame(dict(N=g.size(), exact=g.apply(lambda x: (x.d <= x.unit * 1.0001).mean()),
                          within2=g.apply(lambda x: (x.d <= 2 * x.unit * 1.0001).mean()), med=g.d.median(),
                          p95=g.d.quantile(0.95), max=g.d.max())).reset_index()
    s.to_csv(OUT + "_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(s.to_string())
    print("budget:", bud.n, "requests", "stop:", bud.stop)


if __name__ == "__main__":
    main()
