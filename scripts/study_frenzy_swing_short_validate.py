#!/usr/bin/env python3
"""FRENZY swing-short study (2026-10-20) — PARITY check of the re-implemented walk against the REAL services.frenzy.frenzy_walk.
For sampled bars of sampled episodes (inside the flag, the off bar, and bars after it) the engine's own inputs are rebuilt (the last 1499
closed 5m bars, ≤ 744 complete 1h bars, services.frenzy.normal_hour_usd) and frenzy_walk + frenzy_flagged are called. Compared:
flagged (yes / no), spike time, anchored VWAP. Writes reports/FRENZY_SWING_SHORT_2026-10-20_parity.csv and prints the agreement.
Usage: venv/bin/python scripts/study_frenzy_swing_short_validate.py [episodes.pkl] [n_episodes]
"""
import os, pickle, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
import config  # noqa: E402
from services.frenzy import frenzy_walk, frenzy_flagged, normal_hour_usd  # noqa: E402
import study_frenzy_swing_short_walk as W  # noqa: E402

TH = config.trading_config.thresholds
fn = sys.argv[1] if len(sys.argv) > 1 else os.path.join(W.D, "episodes.pkl")
NE = int(sys.argv[2]) if len(sys.argv) > 2 else 120
eps = pickle.load(open(fn, "rb"))
rng = np.random.default_rng(11)
pick = rng.choice(len(eps), size=min(NE, len(eps)), replace=False)
by = {}
for k in pick:
    by.setdefault(eps[k]["pair"], []).append(eps[k])
rows = []
for pair, es in by.items():
    x = W.load(pair)
    t, o, h, l, c, v, qv, real = x
    df = pd.DataFrame({"t": t, "o": o, "h": h, "l": l, "c": c, "v": v, "real": real})
    df["hr"] = df.t // W.HR * W.HR
    g = df.groupby("hr").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"), v=("v", "sum"), n=("real", "sum"))
    g = g[g.n == 12]; ho = g.index.values.astype("int64"); H1 = g[["o", "h", "l", "c", "v"]].values
    pos = {int(a): i for i, a in enumerate(t)}
    for e in es:
        on, off = e["on_i"], e["off_i"]
        cand = sorted(set([on + 1, on + 24, on + 30, off - 1, off, off + 1, off + 12] + list(rng.integers(on, off + 1, 5))))
        for ci in cand:
            if ci < 0 or ci >= len(e["t"]):
                continue
            T = pos[int(e["t"][ci])]
            a = max(0, T - 1498)
            closed = [[int(t[i]), o[i], h[i], l[i], c[i], v[i]] for i in range(a, T + 1)]
            kh = np.searchsorted(ho, t[T], side="right")       # hours opened ≤ the bar's open (the engine fetches 744 incl. the forming one)
            hb = [[int(ho[i]), *H1[i]] for i in range(max(0, kh - 744), kh)]
            nh = normal_hour_usd(hb, int(t[T]))
            ep = frenzy_walk(closed, nh, TH) if nh else None
            eng_flag = frenzy_flagged(ep, TH)
            mine = bool(e["flag"][ci])
            rows.append(dict(pair=pair, spike_ts=e["spike_ts"], bar=int(t[T]), rel=ci - on, off_rel=off - on, mine=mine, engine=eng_flag,
                             eng_spike=(ep or {}).get("spike_ts"), same_spike=bool(ep and ep.get("spike_ts") == e["spike_ts"]),
                             vwap_mine=e["vwap"][ci], vwap_eng=(ep or {}).get("vwap")))
P = pd.DataFrame(rows)
P.to_csv(os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20_parity.csv"), index=False)
both = P[P.mine & P.engine]
print(f"bars {len(P)} · episodes {P.groupby(['pair','spike_ts']).ngroups} · flag agreement {(P.mine == P.engine).mean() * 100:.1f}% "
      f"(mine-only {(P.mine & ~P.engine).sum()}, engine-only {(~P.mine & P.engine).sum()}) · same spike when both flagged "
      f"{both.same_spike.mean() * 100:.1f}% · VWAP |rel diff| median {np.nanmedian(np.abs(both.vwap_mine / both.vwap_eng.astype(float) - 1)) * 100:.4f}% "
      f"p95 {np.nanpercentile(np.abs(both.vwap_mine / both.vwap_eng.astype(float) - 1), 95) * 100:.4f}%")
