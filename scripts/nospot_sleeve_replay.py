#!/usr/bin/env python3
"""NOSPOT_SLEEVE — operator-coin replays 2026-11-10 → now (PIPPIN, BLESS, JELLYJELLY, PHAROS; HUMA / ZKC / ZKP have a spot market → not in
the universe). Spike list rebuilt on cached + fresh public klines with the NOSPOT replay rule (nospot_mcap_replay.spikes), then the SAME
row build (nospot_sleeve_build.work) with the live metrics / premium / funding, then NS0 + every primary rule × every exit, one position
per pair. Trades whose exit horizon runs past the data end are OPEN (marked to the last bar).
Writes reports/NOSPOT_SLEEVE_2026-11-18_replays.csv"""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import nospot_sleeve_build as NB   # noqa: E402
import nospot_sleeve_lib as L   # noqa: E402
LD = os.path.join(R, "reports", "NOSPOT_SLEEVE_data", "live")
NB.TB.K5M_SRC.append(os.path.join(LD, "k5m")); NB.TB.FUND_SRC.append(os.path.join(LD, "funding"))
import nospot_mcap_replay as MR   # noqa: E402
MR.B.TB.K5M_SRC = NB.TB.K5M_SRC
PAIRS = ["PIPPINUSDT", "BLESSUSDT", "JELLYJELLYUSDT", "PHAROSUSDT"]
T0 = pd.Timestamp("2026-11-10").value // 10**6
EXH = {e: NB.EXITS[e][6] * 300_000 for e in NB.EXITS}


def main():
    import json
    import leads_deep_regimes as LR
    B = LR.btc5m(); btc = (B.open_time.values.astype("int64"), B.c.values.astype(float))
    SUP = json.load(open(os.path.join(R, "reports", "NOSPOT_MCAP_data", "supply_used.json")))
    MI = NB.met_index(sorted(set(NB.MET_DIRS + [os.path.join(LD, "metrics")])))
    PI = NB.met_index(sorted(set(NB.PREM_DIRS + [os.path.join(LD, "premium")])))
    od = os.path.join(LD, "rows"); os.makedirs(od, exist_ok=True)
    out = []
    for p in PAIRS:
        eps = MR.spikes(p)
        eps = eps[eps.spike_ms >= T0 - 3 * 86_400_000]
        print(p, "spikes:", [str(pd.Timestamp(s, unit="ms"))[:16] for s in eps.spike_ms], flush=True)
        if not len(eps):
            continue
        st = {(p, m): 0 for m in ("2026-10", "2026-11")}
        fo = os.path.join(od, p + ".pkl")
        if os.path.exists(fo):
            os.remove(fo)
        NB.work((p, eps, st, MI.get(p, []), PI.get(p, []), SUP.get(p), btc, od))
        if not os.path.exists(fo):
            continue
        X = pd.read_pickle(fo)
        X = X[X.entry_ms >= T0].reset_index(drop=True)
        if not len(X):
            continue
        X["half"] = "26-H2"; X["blk"] = "LIVE"; X["day"] = X.entry_ms // L.DAY; X["epi"] = 0
        X["slip"] = np.maximum(0.10, L.slip_model(X.q5.values.astype(float)))
        X["f_neg"] = (X.f_last < 0).astype(float); X["f_hourly"] = (X.f_int_h == 1).astype(float)
        for e in L.EXITS:
            X[f"{e}_net"] = L.net(X, e)
        last = NB.TB.load(p)[0][-1]
        RM = L.rule_masks(X)
        coh = {"NS0": L.ns0_mask(X), **{k: m for k, (m, _) in RM.items()}}
        for c, m in coh.items():
            for e in L.EXITS:
                for i in L.one_pos(X, np.nonzero(m)[0], e):
                    out.append(dict(pair=p, cohort=c, exit=e, entry=str(pd.Timestamp(int(X.entry_ms[i]), unit="ms"))[:16],
                                    kind={0: "TP", 1: "SL", 2: "TIME"}[int(X[f"{e}_k"][i])], open=bool(int(X[f"{e}_k"][i]) == 2 and X.entry_ms[i] + EXH[e] > last + 300_000),
                                    gross=round(float(X[f"{e}_r"][i]), 2), funding=round(float(X[f"{e}_f"][i]), 3), net=round(float(X[f"{e}_net"][i]), 2),
                                    dd_hi=round(float(X.dd_hi[i]), 1), oi_chg1h=round(float(X.oi_chg1h[i]), 2), glA=round(float(X.glA[i]), 2),
                                    f_last=round(float(X.f_last[i]), 4), f_int_h=X.f_int_h[i], maxup1h_24h=round(float(X.maxup1h_24h[i]), 1)))
        # path context: peak and low since 11-10
        t, o, h, l, c, v, qv = NB.TB.load(p)
        k0 = np.searchsorted(t, T0); ih = k0 + int(np.argmax(h[k0:]))
        out.append(dict(pair=p, cohort="_PATH", exit="", entry=str(pd.Timestamp(int(t[ih]), unit="ms"))[:16], kind="peak",
                        gross=round(float((1 - l[ih:].min() / h[ih]) * 100), 1), net=np.nan))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(R, "reports", "NOSPOT_SLEEVE_2026-11-18_replays.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 2000)
    print(df[(df.exit.isin(["XE", "F1012", ""]))].to_string())


if __name__ == "__main__":
    main()
