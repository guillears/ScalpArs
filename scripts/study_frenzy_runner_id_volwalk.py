#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 · operator add-on "what if FRENZY_LONG buys at 70x?" — engine-exact walk at LOWER state-volume bars.

Research only (read-only on code/config). Construction = scripts/frenzy_engine_cohort_build.py (the build the staircase study's build50
reused), with three changes, frozen before any number was read:
  · data   reports/backtest_cache_2025/k5m_full + reports/backtest_cache/k5m_full merged per pair (2025-01 → 2026-10-12), blacklists out
  · bars   every bar whose last-hour volume >= the LOWEST tested multiple (50x) × normal is walked once per threshold T in (50, 70, 85, 100)
           whose pre-filter it passes, with the REAL services.frenzy.frenzy_walk / frenzy_flagged / frenzy_long_status and today's
           trading_config thresholds except frenzy_state_vol_mult = T
  · output only FRESH-ON bars (the bars an entry can happen on) + their status code (FRENZY_READY = a FRENZY_LONG entry at that T)
Usage: SHARD=i/n venv/bin/python scripts/study_frenzy_runner_id_volwalk.py <outdir>
"""
import os, sys
import numpy as np, pandas as pd
R = "/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, R); os.chdir(R)
import config  # noqa: E402
from services.frenzy import frenzy_walk, normal_hour_usd, frenzy_flagged, frenzy_long_status  # noqa: E402
from services.surge import wilder_atr_pct  # noqa: E402

TH0 = config.trading_config.thresholds; TC = config.trading_config
TS = (50.0, 70.0, 85.0, 100.0)


class THX:
    def __init__(self, base, vm):
        self._b, self._vm = base, vm

    def __getattr__(self, k):
        if k == "frenzy_state_vol_mult":
            return self._vm
        return getattr(self._b, k)


THS = {t: THX(TH0, t) for t in TS}
H, GAP, WIN = 3_600_000, 24 * 3_600_000, 1499
T0 = int(pd.Timestamp("2025-01-01").value // 1_000_000)
SKIP = set()
for src in (getattr(TC, "pair_blacklist", ""), getattr(TC, "no_trade_pairs", ""), getattr(TH0, "frenzy_pair_blacklist", "")):
    SKIP |= {x.strip().upper() for x in str(src or "").split(",") if x.strip()}
C25, C26 = "reports/backtest_cache_2025/k5m_full", "reports/backtest_cache/k5m_full"


def load(pair):
    fr = [pd.read_csv(f"{c}/{pair}.csv") for c in (C25, C26) if os.path.exists(f"{c}/{pair}.csv")]
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    return d


def one(pair):
    d = load(pair)
    if len(d) < 3000:
        return []
    ot = d.open_time.values.astype("int64"); o, h, l, c, v, qv = (d[k].values.astype(float) for k in ("o", "h", "l", "c", "vol", "qvol"))
    rows = [[int(a), float(b), float(e), float(f), float(g), float(x)] for a, b, e, f, g, x in zip(ot, o, h, l, c, v)]
    q = v * (h + l + c) / 3.0; q1h = pd.Series(q).rolling(12).sum().values
    d["hr"] = d.open_time // H * H
    g = d.groupby("hr").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("vol", "sum"), n=("o", "size"))
    g = g[g.n == 12]
    H1 = [[int(k), r.o, r.h, r.l, r.c, r.v] for k, r in g.iterrows()]; hk = np.array([x[0] for x in H1], dtype="int64")
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    ncache = {}; out = []
    j0 = max(300, int(np.searchsorted(ot, T0)))
    for j in range(j0, len(ot)):
        sc = int(ot[j]) + 300_000
        k = int(np.searchsorted(hk, sc - H, side="right"))
        k0 = max(0, k - 744); m = int(np.searchsorted(hk[k0:k] + H, int(ot[j]) - GAP, side="right"))
        key = (k0, k0 + m)
        if key not in ncache:
            ncache[key] = normal_hour_usd(H1[k0:k], int(ot[j]))
        nh = ncache[key]
        if not nh or not np.isfinite(q1h[j]):
            continue
        ratio = q1h[j] / nh
        if ratio < TS[0]:
            continue
        sub = rows[max(0, j - WIN + 1):j + 1]
        atr = None
        for t in TS:
            if ratio < t:
                break
            ep = frenzy_walk(sub, nh, THS[t])
            if not (ep and frenzy_flagged(ep, THS[t]) and ep.get("in_state") and ep.get("fresh_on")):
                continue
            if atr is None:
                atr = wilder_atr_pct(sub[-300:])
            v24 = float(vol24[j]) if np.isfinite(vol24[j]) else None
            ready, code, _ = frenzy_long_status(ep, atr, v24, THS[t])
            out.append(dict(T=t, pair=pair, t_signal_close=sc, bar_open=int(ot[j]), ready=bool(ready), code=code,
                            spike_ts=int(ep["spike_ts"]), hours=ep["hours"], vs_vwap=ep["vs_vwap_pct"], above_share=ep["above_share"],
                            vol_mult=ep["vol_mult"], vwap=ep["vwap"], price=ep["price"], bar_red=bool(ep["bar_red"]),
                            bar_ret=ep["bar_ret_pct"], run_pct=ep["run_pct"], gain_pct=ep["gain_pct"], off_peak=ep["off_peak_pct"],
                            atr=atr, vol24=v24, norm_hour=nh, ratio=ratio))
    return out


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    pairs = sorted({f[:-4] for c in (C25, C26) for f in os.listdir(c) if f.endswith(".csv")} - SKIP)
    if os.environ.get("PAIRS"):
        pairs = os.environ["PAIRS"].split(",")
    i_, n_ = map(int, os.environ.get("SHARD", "0/1").split("/")); pairs = pairs[i_::n_]
    res = []
    for p in pairs:
        try:
            r = one(p); res += r
            print(p, len(r), sum(x["ready"] for x in r), flush=True)
        except Exception as e:
            print("FAIL", p, repr(e), flush=True)
    pd.DataFrame(res).to_pickle(f"{outdir}/vw_{i_}of{n_}.pkl"); print("done", len(res))
