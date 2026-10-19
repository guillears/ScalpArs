#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 — AGTUSDT 10-16 → 10-19 case at the 70x / 85x / 100x state-volume bars (same engine functions as the
volwalk; 5m = cache + public klines fetched by study_frenzy_runner_id_fetch.py; paths = ticks first, then the fetched 1m).
Usage: venv/bin/python scripts/study_frenzy_runner_id_agt.py <scratch>"""
import os, sys
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
S = sys.argv[1]
import study_frenzy_runner_id_volwalk as VW  # noqa: E402
sys.argv = [sys.argv[0], S]
import study_frenzy_runner_id_build as B  # noqa: E402

P = "AGTUSDT"
d = pd.concat([pd.read_csv(f"reports/backtest_cache/k5m_full/{P}.csv"), pd.read_csv(f"{S}/fetch/{P}_5m.csv")])
d = d.drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
d = d[d.open_time + 300_000 <= int(pd.Timestamp("2026-10-19 14:55").value // 1_000_000)].reset_index(drop=True)
VW.load = lambda pair: d
VW.T0 = int(pd.Timestamp("2026-10-15").value // 1_000_000)
rows = pd.DataFrame(VW.one(P))
rows["signal_close"] = pd.to_datetime(rows.t_signal_close, unit="ms")
k1 = pd.concat([B.WT.load_1m(P), pd.read_csv(f"{S}/fetch/{P}_1m.csv")]).drop_duplicates("open_time", keep="last").sort_values("open_time")
out = []
for _, r in rows.iterrows():
    te = int(r.t_signal_close) + 8_000
    Pp = B.FX.load_path(P, te, float(r.price), 720, {}, k1m=k1)
    m, tm, st, cen, _ = B.mfe_before_stop(Pp, 720)
    out.append(dict(T=r["T"], signal_close=r.signal_close, code=r.code, vol_mult=round(r.vol_mult, 1), price=r.price, hours=round(r.hours, 1),
                    vs_vwap=round(r.vs_vwap, 1), mfe_720_before_stop=round(m, 2), stopped=st, x0=round(B.walk(Pp, "FIX", tp=1.5)[0], 2),
                    r1=round(B.walk(Pp, "R1")[0], 2), r2=round(B.walk(Pp, "R2")[0], 2)))
O = pd.DataFrame(out)
O.to_csv("reports/FRENZY_RUNNER_ID_2026-10-19_agt_case.csv", index=False)
print(O.to_string())
