#!/usr/bin/env python3
"""🌐 EXHAUSTION WIDE universe (DECISION_LOG 585) — build the PARITY fixture tests/fixtures/exh_wide_parity.npz from the EXPAND_EDGES study's
own spike lists (reports/EXPAND_EDGES_2026-11-18_data/exh/X3_vol15/meta.csv = the 15× list the L4 cell used; BASE/meta.csv = the 20× list).
Picks spikes that are in the 15× list and NOT in the 20× list (the WIDE-only spikes) and stores, per spike, exactly what the live engine holds
one hour after it: the study's 5m series (study_frenzy_swing_short_walk.load — flat-filled grid, base volume) for the 1,500 bars ending 12
bars after the spike bar, and the study's normal hour (norm_per_bar) at the LAST bar (the engine reads normal_hour_usd once per window).
tests/test_exh_wide_585.py runs services.frenzy.frenzy_walk at the WIDE multiple (services.exhaustion.wide_walk_th) on it: it must find the
same spike (and the 20× walk must not). READ-ONLY on the bot. Usage: venv/bin/python scripts/exh_wide_parity_fixture.py"""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import study_frenzy_swing_short_walk as W   # noqa: E402
import exh_expand_build as XB               # noqa: E402
DX = os.path.join(R, "reports", "EXPAND_EDGES_2026-11-18_data", "exh")
BAR, N5 = 300_000, 1500


def main(n=6):
    W.SRC = XB.TB.K5M_SRC; W.T0 = pd.Timestamp("2023-05-01").value // 10**6   # expand_edges_exh.spikes_one's sources
    x3 = pd.read_csv(os.path.join(DX, "X3_vol15", "meta.csv")); base = pd.read_csv(os.path.join(DX, "BASE", "meta.csv"))
    bk = set(zip(base.pair, base.on_ms))
    only = x3[[(p, o) not in bk for p, o in zip(x3.pair, x3.on_ms)]].reset_index(drop=True)
    rng = np.random.default_rng(585)
    blob, k = {}, 0
    for i in rng.permutation(len(only)):
        if k >= n:
            break
        p, on = str(only.pair[i]), int(only.on_ms[i])
        z = W.load(p)
        if z is None:
            continue
        t, o, h, l, c, v, qv, real = z
        j = int(np.searchsorted(t, on))
        if j >= len(t) or t[j] != on or j + 12 >= len(t) or j + 12 - N5 < 0:
            continue
        e = j + 12
        nh = W.norm_per_bar(t, h, l, c, v, real)
        if not (nh[e] > 0):
            continue
        a = e - N5 + 1
        q = v * (h + l + c) / 3.0
        q1h = pd.Series(q).rolling(12).sum().values
        blob[f"r{k}_k5"] = np.c_[t[a:e + 1], o[a:e + 1], h[a:e + 1], l[a:e + 1], c[a:e + 1], v[a:e + 1]]
        blob[f"r{k}_meta"] = np.array([on, on + BAR, nh[e], q1h[j] / nh[j]], float)
        blob[f"r{k}_pair"] = np.array(p)
        print(p, pd.Timestamp(on, unit="ms"), "volx at spike (study norm)", round(q1h[j] / nh[j], 2))
        k += 1
    np.savez_compressed(os.path.join(R, "tests", "fixtures", "exh_wide_parity.npz"), **blob)


if __name__ == "__main__":
    main()
