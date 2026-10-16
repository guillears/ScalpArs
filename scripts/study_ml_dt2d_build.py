#!/usr/bin/env python3
"""ML_DOWNTREND_2D (2026-10-16) — feature builder. Research only (pre-registration reports/ML_DOWNTREND_2D_PREREG.txt).

Blocks (one row per fill):
  replay  ENGINE_REPLAY_YR6_2Y_TODAY_fills MOM-long, non-probe, CLOSED, 3 seeds (study_yr6wl_momentum_common.load_replay; y = pct_new)
  master  MASTER_POOL_stacked MOMENTUM LONG non-probe CLOSED (load_master_mom; y as traded = 1×; y_stack = today's stack, NaN blocked)
  fresh   the PAIR_1H_DOWNTREND line's own counted cohort (scout_ml_trend_lines.load_orders, ≥ 2026-09-30, ex-washed) with every
          entry_* column re-read from the same exports (export_glob, dedup (opened_at, pair, direction) keep newest export)
Every fill gets k_* features rebuilt from 5m klines (cache only + reports/ml_dt2d_fetch/5m top-up written by --fetch) and the
ML_WL_1015 market-impulse features where that file has the fill. Output: reports/ML_DT2D_features.pkl
    venv/bin/python -W ignore scripts/study_ml_dt2d_build.py [--fetch]
"""
import os as _os_bc, sys as _sys_bc; _sys_bc.path.insert(0, _os_bc.path.dirname(_os_bc.path.abspath(__file__))); import binance_cooldown as _BC  # noqa: E401,E402  (DL 306)
import glob
import json
import os
import sys
import time
import urllib.request
import urllib.error

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from study_yr6wl_momentum_common import load_replay, load_master_mom   # noqa: E402
from study_ml_wl_1015_common import add_derived                          # noqa: E402

H, D4, M5 = 3_600_000, 4 * 3_600_000, 300_000
OUT = os.path.join(ROOT, "reports/ML_DT2D_features.pkl")
FETCH = os.path.join(ROOT, "reports/ml_dt2d_fetch/5m")
SRC = [os.path.join(ROOT, "reports/backtest_cache_2025/k5m_full"), os.path.join(ROOT, "reports/backtest_cache/k5m_full"),
       os.path.join(ROOT, "reports/backtest_cache/k5m"), os.path.join(ROOT, "reports/backtest_cache/negflank2d_ext/5m"),
       os.path.join(ROOT, "reports/backtest_cache/scout_ml_trend/5m"), FETCH]
COOLDOWNS = [os.path.join(ROOT, "reports/backtest_cache/.binance_cooldown_until"),
             os.path.join(ROOT, "reports/backtest_cache/scout_ml_trend/.ratelimited_until")]
KEY = ["opened_at", "pair", "direction"]


def k5(sym):
    fr = []
    for s in SRC:
        f = os.path.join(s, f"{sym}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            for c in ("h", "l"):
                if c not in d:
                    d[c] = d["c"]
            fr.append(d[["open_time", "h", "l", "c"]])
    if not fr:
        return None
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    d["T"] = d.open_time.astype("int64") + M5
    return d


def htf_partial(closes, bucket, span, T, c):
    E = closes.ewm(span=span, adjust=False).mean()
    a = 2.0 / (span + 1)
    b = (T // bucket) * bucket
    return a * c + (1 - a) * E.reindex(b - bucket).values, E.reindex(b - 3 * bucket).values


def grid(sym):
    p = k5(sym)
    if p is None or len(p) < 3000:
        return None
    T, c = p["T"].values, p.c.values.astype(float)
    g = pd.DataFrame({"T": T})
    h1 = pd.Series(c, index=T).groupby(((T - M5) // H) * H).last()
    e20, e20p3 = htf_partial(h1, H, 20, T, c)
    e200, _ = htf_partial(h1, H, 200, T, c)
    g["g1h"] = (e20 / e200 - 1) * 100
    g["px_e200"] = (c / e200 - 1) * 100
    g["px_e20"] = (c / e20 - 1) * 100
    g["slope1h"] = (e20 - e20p3) / e20p3 * 100
    h4 = h1.groupby((h1.index // D4) * D4).last()
    e20_4, e20_4p3 = htf_partial(h4, D4, 20, T, c)
    e50_4, _ = htf_partial(h4, D4, 50, T, c)
    g["gap4h"] = (e20_4 / e50_4 - 1) * 100
    g["slope4h"] = (e20_4 - e20_4p3) / e20_4p3 * 100
    # history guard: ≥ 400 hourly closes before the bar (EMA200 seed) else NaN
    nh = np.searchsorted(h1.index.values, (T // H) * H, side="left")
    g.loc[nh < 400, ["g1h", "px_e200", "px_e20", "slope1h", "gap4h", "slope4h"]] = np.nan
    s = pd.Series(c, index=T)
    for lab, ms in (("4h", 4 * H), ("24h", 24 * H)):
        i = np.searchsorted(T, T - ms, side="right") - 1
        ok = (i >= 0) & (T - T[np.clip(i, 0, None)] <= ms + 3 * M5)
        g[f"ret{lab}"] = np.where(ok, (c / c[np.clip(i, 0, None)] - 1) * 100, np.nan)
    for lab, ms in (("1h", H), ("4h", 4 * H), ("24h", 24 * H)):
        i = np.searchsorted(T, T - ms, side="right") - 1
        ok = (i >= 0) & (T - T[np.clip(i, 0, None)] <= ms + 3 * M5)
        g[f"g1h_chg{lab}"] = np.where(ok, g.g1h.values - g.g1h.values[np.clip(i, 0, None)], np.nan)
    hh = pd.Series(p.h.values, index=pd.to_datetime(T, unit="ms")).rolling("24h").max().values
    ll = pd.Series(p.l.values, index=pd.to_datetime(T, unit="ms")).rolling("24h").min().values
    first = T - T[0] >= 24 * H
    g["off24h_high"] = np.where(first, (c / hh - 1) * 100, np.nan)
    g["above24h_low"] = np.where(first, (c / ll - 1) * 100, np.nan)
    return g


def lookup(g, t_ms, cols):
    idx = np.searchsorted(g["T"].values, t_ms, side="right") - 1
    out = np.full((len(t_ms), len(cols)), np.nan)
    ok = idx >= 0
    age = t_ms - np.where(ok, g["T"].values[np.clip(idx, 0, None)], 0)
    ok &= age <= 900_000
    out[ok] = g[cols].values[idx[ok]]
    return out


PC = ["g1h", "px_e200", "px_e20", "slope1h", "gap4h", "slope4h", "ret4h", "ret24h", "g1h_chg1h", "g1h_chg4h", "g1h_chg24h",
      "off24h_high", "above24h_low"]
BC = ["g1h", "slope1h", "slope4h", "gap4h", "g1h_chg4h", "ret24h"]


def fresh():
    import scout_ml_trend_lines as S
    from export_glob import glob_exports
    o = S.load_orders()
    o = o[(o.ts >= pd.Timestamp("2026-09-30")) & ~o.washed.astype(bool)].copy()
    o["_k"] = o.opened_at.astype(str).str[:19]
    fr = []
    for f in glob_exports(os.path.expanduser("~/Downloads/scalpars_orders_paper_*.csv")):
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if "opened_at" not in d:
            continue
        d["_k"] = d.opened_at.astype(str).str[:19]
        d = d[d.direction.astype(str) == "LONG"]
        d = d.merge(o[["_k", "pair"]], on=["_k", "pair"])
        if len(d):
            fr.append(d.assign(_m=os.path.getmtime(f), _src=os.path.basename(f)))
    a = pd.concat(fr, ignore_index=True).sort_values("_m", kind="stable").drop_duplicates(["_k", "pair"], keep="last")
    a = a.merge(o[["_k", "pair", "pct", "usd", "g1h"]].rename(columns={"g1h": "g1h_line"}), on=["_k", "pair"])
    assert len(a) == len(o), (len(a), len(o))
    a["ts"] = pd.to_datetime(a._k)
    a["day"] = a.ts.dt.floor("D")
    a["y"] = a.pct.astype(float)
    a["block"] = "fresh"
    a["is_live"] = a._src.str.contains("_live_")
    return a


def fetch_missing(D):
    """≤ 1 klines page (limit 1500) per symbol whose 5m series ends before a fill; shared cooldown honoured; stop at weight 900."""
    now = int(time.time() * 1000)
    for f in COOLDOWNS:
        try:
            if _BC.read_until_ms(f) > now:   # DL 306: unit-robust (ms)
                print("cooldown live — no fetch"); return
        except Exception:
            pass
    os.makedirs(FETCH, exist_ok=True)
    need = {}
    for sym, g in D.groupby("pair"):
        need[sym] = int(g.t_ms.max())
    need["BTCUSDT"] = int(D.t_ms.max())
    n = 0
    for sym, tmax in sorted(need.items()):
        p = k5(sym)
        end = int(p.open_time.max()) if p is not None else None
        if end is not None and end + M5 >= tmax:
            continue
        start = end + M5 if end is not None and tmax - end < 1400 * M5 else tmax - 1400 * M5
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval=5m&startTime={start}&limit=1500"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:
                rows = json.loads(r.read().decode()); used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
        except urllib.error.HTTPError as e:
            print(sym, "HTTP", e.code)
            if e.code in (418, 429):
                return
            continue
        n += 1
        d = pd.DataFrame([(int(x[0]), float(x[2]), float(x[3]), float(x[4])) for x in rows if int(x[0]) + M5 <= now],
                         columns=["open_time", "h", "l", "c"])
        f = os.path.join(FETCH, f"{sym}.csv")
        if os.path.exists(f):
            d = pd.concat([pd.read_csv(f), d]).drop_duplicates("open_time", keep="last").sort_values("open_time")
        d.to_csv(f, index=False)
        print(f"fetched {sym} {len(d)} bars · used weight {used}")
        if used >= 900:
            print("weight stop"); return
        time.sleep(0.3)
    print("requests:", n)


def main():
    R = load_replay(["MOM-long"])
    R["block"] = R.regime
    M = load_master_mom()
    M = M[M.sleeve == "MOM-long"].copy()
    M["block"] = "master"
    M["seed"] = 0
    Fr = fresh()
    Fr["seed"] = 0
    for d in (R, M, Fr):
        d["t_ms"] = d.ts.values.astype("datetime64[ms]").astype("int64")
    if "--fetch" in sys.argv:
        fetch_missing(Fr)
    keep = sorted((set(R.columns) | set(M.columns) | set(Fr.columns)))
    D = pd.concat([R, M, Fr], ignore_index=True, sort=False)
    D = add_derived(D)
    D["win"] = D.y > 0
    # k_* rebuild
    for c in PC:
        D[f"k_{c}"] = np.nan
    t0 = time.time()
    for i, (sym, idx) in enumerate(D.groupby("pair").groups.items()):
        g = grid(sym)
        if g is None:
            continue
        D.loc[idx, [f"k_{c}" for c in PC]] = lookup(g, D.loc[idx, "t_ms"].values, PC)
        if i % 50 == 0:
            print(f"  {i} pairs · {time.time() - t0:.0f}s", flush=True)
    gb = grid("BTCUSDT")
    D[[f"k_btc_{c}" for c in BC]] = lookup(gb, D.t_ms.values, BC)
    # zone variable: stamp first, rebuild where missing
    st = pd.to_numeric(D.entry_pair_1h_ema20_200_gap_pct, errors="coerce")
    D["g1h_stamp"] = st
    D["g1h_z"] = st.fillna(D.k_g1h)
    D["g1h_src"] = np.where(st.notna(), "stamp", np.where(D.k_g1h.notna(), "rebuild", "unscored"))
    D["DT"] = np.where(D.g1h_z.notna(), D.g1h_z <= 0, np.nan)
    # market-impulse features (ML_WL_1015)
    F = pd.read_csv(os.path.join(ROOT, "reports/ML_WL_1015_features.csv"))
    F["_k"] = F.opened_at.astype(str).str[:19]
    F["blk"] = np.where(F.block.isin(["master", "live"]), "real", F.block)
    D["_k"] = D.opened_at.astype(str).str[:19]
    D["blk"] = np.where(D.block.isin(["master", "fresh"]), "real", D.block)
    D["seed_"] = pd.to_numeric(D.seed, errors="coerce").fillna(0).astype(int)
    F["seed_"] = pd.to_numeric(F.seed, errors="coerce").fillna(0).astype(int)
    mk = [c for c in F.columns if c not in ("block", "pair", "opened_at", "seed", "_k", "blk", "seed_")
          and not c.startswith(("pnl_m", "e1_", "e3_", "e5_", "btc_d", "br_d")) and c not in ("univ_day_lag_d", "br_n")]
    F = F.drop_duplicates(["blk", "_k", "pair", "seed_"])
    D = D.merge(F[["blk", "_k", "pair", "seed_"] + mk].rename(columns={c: f"m_{c}" for c in mk}),
                on=["blk", "_k", "pair", "seed_"], how="left")
    D.to_pickle(OUT)
    print("wrote", OUT, D.shape)
    print(D.groupby("block").g1h_src.value_counts().unstack().fillna(0).astype(int).to_string())
    ok = D.g1h_stamp.notna() & D.k_g1h.notna()
    for b, g in D[ok].groupby("block"):
        print(f"rebuild vs stamp {b}: n {len(g)} sign agree {(np.sign(g.g1h_stamp) == np.sign(g.k_g1h)).mean():.3f} "
              f"r {g.g1h_stamp.corr(g.k_g1h):.3f} med|d| {(g.g1h_stamp - g.k_g1h).abs().median():.3f}")
    kc = [c for c in D.columns if c.startswith(("k_", "m_"))]
    print(D.groupby("block")[kc].apply(lambda x: x.notna().mean()).T.round(2).to_string())


if __name__ == "__main__":
    main()
