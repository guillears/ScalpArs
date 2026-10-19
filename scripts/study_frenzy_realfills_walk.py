#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — engine-exact 5m episode walk over the REAL-FILL ERA (2026-10-02 → 2026-10-19 15:45 UTC), every
pair, at frenzy_state_vol_mult T in (50, 70, 85, 100). Parts C / D / E / F read this.

Same construction as scripts/study_frenzy_runner_id_volwalk.py (the REAL services.frenzy.frenzy_walk / normal_hour_usd / frenzy_flagged /
frenzy_long_status, today's trading_config thresholds, only frenzy_state_vol_mult = T), extended:
  · data   reports/backtest_cache/k5m_full (→ 10-12) + data.binance.vision daily 5m (10-12 → 10-18) + public fapi 5m for 10-19 (this study's
           fetcher) + the runner-ID fetch; merged per pair, closed bars only
  · rows   EVERY bar of a flagged episode (verified ∧ hours ≤ 96), not only fresh-ON bars, so the LITE status (not-in-state bars) and the
           episode timeline can be rebuilt; fields = the ep dict the engine judges + 5m ATR(14) (wilder, 300 bars) + 24 h quote volume
           (288 × volume × typical price) + the FRENZY_LONG status code at T
Usage: SHARD=i/n venv/bin/python scripts/study_frenzy_realfills_walk.py <outdir>
"""
import os, sys, glob
import numpy as np, pandas as pd
R = "/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, R); os.chdir(R)
import config  # noqa: E402
from services.frenzy import frenzy_walk, normal_hour_usd, frenzy_flagged, frenzy_long_status  # noqa: E402
from services.surge import wilder_atr_pct  # noqa: E402

SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
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
T0 = int(pd.Timestamp("2026-10-02").value // 1_000_000)
LEAD0 = int(pd.Timestamp("2026-09-27").value // 1_000_000)
SKIP = set()
for src in (getattr(TC, "pair_blacklist", ""), getattr(TC, "no_trade_pairs", ""), getattr(TH0, "frenzy_pair_blacklist", "")):
    SKIP |= {x.strip().upper() for x in str(src or "").split(",") if x.strip()}


def load5(pair):
    fr = []
    f = f"reports/backtest_cache/k5m_full/{pair}.csv"
    if os.path.exists(f):
        fr.append(pd.read_csv(f))
    fr += [pd.read_csv(x) for x in sorted(glob.glob(f"{SCR}/rf/dl/k5/{pair}/*.csv"))]
    for x in (f"{SCR}/rid/fetch/{pair}_5m.csv", f"{SCR}/rf/dl/k5api/{pair}.csv"):
        if os.path.exists(x):
            fr.append(pd.read_csv(x))
    if not fr:
        return None
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    return d[d.open_time >= T0 - 40 * 86_400_000].reset_index(drop=True)


def one(pair):
    d = load5(pair)
    if d is None or len(d) < 3000:
        return []
    ot = d.open_time.values.astype("int64"); o, h, l, c, v, qv = (d[k].values.astype(float) for k in ("o", "h", "l", "c", "vol", "qvol"))
    rows = [[int(a), float(b), float(e), float(f), float(g), float(x)] for a, b, e, f, g, x in zip(ot, o, h, l, c, v)]
    q = v * (h + l + c) / 3.0; q1h = pd.Series(q).rolling(12).sum().values
    d["hr"] = d.open_time // H * H
    g = d.groupby("hr").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("vol", "sum"), n=("o", "size"))
    g = g[g.n == 12]
    H1 = [[int(k), r.o, r.h, r.l, r.c, r.v] for k, r in g.iterrows()]; hk = np.array([x[0] for x in H1], dtype="int64")
    vol24 = pd.Series(q).rolling(288, min_periods=288).sum().values
    # spike-lead prefilter (the walk's own lead rule, vectorised with the per-bar normal hour): a bar can only belong to an episode if a
    # lead bar exists in the previous WIN bars
    ncache = {}

    def nh_at(j):
        sc = int(ot[j]) + 300_000
        k = int(np.searchsorted(hk, sc - H, side="right"))
        k0 = max(0, k - 744); m = int(np.searchsorted(hk[k0:k] + H, int(ot[j]) - GAP, side="right"))
        key = (k0, k0 + m)
        if key not in ncache:
            ncache[key] = normal_hour_usd(H1[k0:k], int(ot[j]))
        return ncache[key]
    r30 = np.r_[np.full(6, np.nan), c[6:] / c[:-6] - 1] * 100
    cand = np.flatnonzero((ot >= LEAD0) & (r30 >= float(TH0.frenzy_spike_ret_pct)) & (q1h >= float(TH0.frenzy_spike_min_hour_usd)))
    leads = [j for j in cand if (lambda nh: nh and q1h[j] / nh >= float(TH0.frenzy_spike_vol_mult))(nh_at(j))]
    if not leads:
        return []
    leads = np.array(leads)
    out = []
    j0 = max(300, int(np.searchsorted(ot, T0)))
    for j in range(j0, len(ot)):
        if not np.any((leads <= j) & (leads > j - WIN)):
            continue
        nh = nh_at(j)
        if not nh or not np.isfinite(q1h[j]):
            continue
        sub = rows[max(0, j - WIN + 1):j + 1]
        atr = None
        for t in TS:
            ep = frenzy_walk(sub, nh, THS[t])
            if not (ep and frenzy_flagged(ep, THS[t])):
                continue
            if atr is None:
                atr = wilder_atr_pct(sub[-300:])
            v24 = float(vol24[j]) if np.isfinite(vol24[j]) else None
            ready, code, _ = frenzy_long_status(ep, atr, v24, THS[t]) if ep.get("in_state") or True else (False, "", "")
            out.append(dict(T=t, pair=pair, bar_open=int(ot[j]), t_close=int(ot[j]) + 300_000, spike_ts=int(ep["spike_ts"]),
                            hours=ep["hours"], vol_mult=ep["vol_mult"], in_state=bool(ep["in_state"]), fresh_on=bool(ep["fresh_on"]),
                            above_streak=int(ep["above_streak"]), above_hour=bool(ep["above_hour"]), verified=bool(ep["verified"]),
                            bar_red=bool(ep["bar_red"]), bar_ret=ep["bar_ret_pct"], vs_vwap=ep["vs_vwap_pct"], above_share=ep["above_share"],
                            on_bar_ts=ep["on_bar_ts"], price=ep["price"], vwap=ep["vwap"], run_pct=ep["run_pct"], atr=atr, vol24=v24,
                            long_ready=bool(ready), long_code=code, norm_hour=nh, o=float(o[j]), c=float(c[j]), h=float(h[j]), l=float(l[j])))
    return out


if __name__ == "__main__":
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    pairs = sorted(({f[:-4] for f in os.listdir("reports/backtest_cache/k5m_full") if f.endswith(".csv")}
                    | set(os.listdir(f"{SCR}/rf/dl/k5"))) - SKIP)
    if os.environ.get("PAIRS"):
        pairs = os.environ["PAIRS"].split(",")
    i_, n_ = map(int, os.environ.get("SHARD", "0/1").split("/")); pairs = pairs[i_::n_]
    res = []
    for p in pairs:
        try:
            r = one(p); res += r
            if r:
                print(p, len(r), flush=True)
        except Exception as e:
            print("FAIL", p, repr(e), flush=True)
    pd.DataFrame(res).to_pickle(f"{outdir}/w_{i_}of{n_}.pkl"); print("done", len(res))
