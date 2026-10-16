#!/usr/bin/env python3
"""WILLY THEORY — step 1: FRENZY episode bars rebuilt with the ENGINE'S frenzy_walk / normal_hour_usd on every WILLY + LITE-cohort pair.
build_pair_w = study_lite_theory_episodes.build_pair copied VERBATIM except ONE line: the candidate lead bar itself is judged
(ev[L:…] instead of ev[L + 1:…]) — LITE never needed the spike bar, WILLY's entry A arms ON it (first parity run: 494 / 1,201 matched
fills had the replay's trigger one bar earlier). No outcome is read. Output reports/willy_theory/bars_w.pkl.
"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402
import study_lite_theory_common as T  # noqa: E402
from services.frenzy import frenzy_walk, normal_hour_usd  # noqa: E402

TH = T.th()
WIN_AFTER = 360


def build_pair_w(pair):
    d = T.load_5m(pair)
    if d is None or len(d) < 400:
        return None
    d = d[(d.open_time >= T.START - 45 * 86_400_000) & (d.open_time < T.END)].reset_index(drop=True)
    t = d.open_time.values.astype("int64"); o = d.o.values.astype(float); h = d.h.values.astype(float)
    l = d.l.values.astype(float); c = d.c.values.astype(float); v = d.vol.values.astype(float); qv = d.qvol.values.astype(float)
    n = len(t)
    q = v * (h + l + c) / 3.0
    q1h = pd.Series(q).rolling(12).sum().values
    cq = np.concatenate([[0.0], np.cumsum(qv)])
    # hourly bars (completed hours, the replay's 5m → 1h resampler)
    hh = (t // T.HOUR_MS) * T.HOUR_MS
    g = pd.DataFrame(dict(hh=hh, o=o, h=h, l=l, c=c, v=v)).groupby("hh").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"),
                                                                              c=("c", "last"), v=("v", "sum")).reset_index()
    H_open = g.hh.values.astype("int64")
    H_rows = g[["hh", "o", "h", "l", "c", "v"]].values.tolist()
    qh = g.v.values * (g.h.values + g.l.values + g.c.values) / 3.0
    med = pd.Series(qh).rolling(719, min_periods=240).median().shift(25).values
    hidx = np.searchsorted(H_open, hh)
    norm_apx = med[np.minimum(hidx, len(med) - 1)]
    ret6 = np.full(n, np.nan); ret6[6:] = (c[6:] / c[:-6] - 1) * 100
    with np.errstate(invalid="ignore", divide="ignore"):
        lead = (ret6 >= 5.0) & (q1h >= 1.8e6) & (q1h / norm_apx >= 15.0)
    lead[:11] = False
    li = np.flatnonzero(lead)
    if not len(li):
        return None
    ev = np.zeros(n, bool)
    for L in li:
        ev[L:min(n, L + WIN_AFTER + 1)] = True        # WILLY: the lead bar itself (the spike bar) is judged too
    ev &= (t >= T.START) & (np.arange(n) >= 300)
    rows = d[["open_time", "o", "h", "l", "c", "vol"]].values.tolist()
    for r in rows:
        r[0] = int(r[0])
    ncache = {}
    out = []
    for j in np.flatnonzero(ev):
        cur_open = ((t[j] + T.BAR_MS + T.T_LAG_MS) // T.HOUR_MS) * T.HOUR_MS
        if cur_open not in ncache:
            k = int(np.searchsorted(H_open, cur_open))      # completed hours before the scan's forming hour
            ncache[cur_open] = normal_hour_usd(H_rows[max(0, k - 743):k], int(t[j]))
        nh = ncache[cur_open]
        if not nh:
            continue
        win = rows[max(0, j - T.WIN + 1):j + 1]
        ep = frenzy_walk(win, nh, TH)
        if not ep or not ep.get("verified") or ep.get("hours") is None or ep["hours"] > 30.0 or ep["hours"] > 96.0:
            continue
        on = int(np.searchsorted(t, int(ep["spike_ts"]) - T.BAR_MS))
        if on >= n or t[on] != int(ep["spike_ts"]) - T.BAR_MS:
            continue
        hs = float(h[on + 1:j + 1].max()) if j > on else np.nan
        out.append(dict(pair=pair, T=int(t[j]), j=int(j), spike_ts=int(ep["spike_ts"]), hours=ep["hours"], streak=int(ep["above_streak"]),
                        vol_mult=ep["vol_mult"], in_state=bool(ep["in_state"]), vs_vwap=ep["vs_vwap_pct"], vwap=ep["vwap"],
                        bar_red=bool(ep["bar_red"]), bar_ret=ep["bar_ret_pct"], run_pct=ep["run_pct"], gain_pct=ep["gain_pct"],
                        off_peak=ep["off_peak_pct"], above_share=ep["above_share"], peak=ep["peak"], base=ep["base"],
                        o=o[j], h=h[j], l=l[j], c=c[j], norm=nh,
                        vol24=(float(cq[j + 1] - cq[j + 1 - 288]) if j + 1 >= 288 else np.nan),
                        spike_volx=(q1h[on] / nh if np.isfinite(q1h[on]) else np.nan),
                        spike_ret=((c[on] / c[on - 6] - 1) * 100 if on >= 6 and c[on - 6] > 0 else np.nan),
                        spike_high=h[on], hi_since=hs, gap_ok=bool(j - on == (t[j] - t[on]) // T.BAR_MS)))
    return pd.DataFrame(out) if out else None


def main():
    F = pd.read_csv(W.FILLS, low_memory=False, usecols=["sleeve", "pair"])
    K = pd.read_csv(os.path.join(W.REP, "LITE_PROTECT_rows.csv"), low_memory=False, usecols=["pair"])
    pairs = sorted(set(F[F.sleeve == "FRENZY_WILLY"].pair) | set(K.pair) | {"DEEPUSDT", "DUSKUSDT", "EPICUSDT", "ZKUSDT"})
    t0 = time.time(); res = []
    with Pool(int(os.environ.get("JOBS", "8"))) as P:
        for k, r in enumerate(P.imap_unordered(build_pair_w, pairs, chunksize=1)):
            if r is not None:
                res.append(r)
            if k % 40 == 0:
                print(f"{k}/{len(pairs)} · {time.time() - t0:.0f}s", flush=True)
    B = pd.concat(res, ignore_index=True)
    B.to_pickle(os.path.join(W.OUT, "bars_w.pkl"))
    print(f"bars {len(B):,} · pairs {B.pair.nunique()} · episodes {B.groupby(['pair', 'spike_ts']).ngroups:,} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
