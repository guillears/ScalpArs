#!/usr/bin/env python3
"""EXHAUSTION_SHORT (DECISION_LOG 578) — build the PARITY fixture tests/fixtures/exh_parity.npz from the SPOT-SHARE DEEP study's own data.
Picks study rows (the RSI14X-15m SHORT signal bars of reports/SPOT_SHARE_DEEP_data/bars/*.pkl), and stores, per row, exactly the inputs a
live bot would hold at the signal close T: the study's 5m perp series (frenzy_ta_build.load — flat-filled grid, exchange quote volume) for
the 1,500 bars ending at T plus the entry bar (its open = the study's ATR denominator), the spot 15m quote volumes of the 26 h before T,
the episode's on / spike times, and the STUDY's values (atr_pct / vol_decay / spot_share_chg / q5 as stored; RSI from the study's own
rsi() on the FULL history). tests/test_exhaustion_short.py recomputes them with services/exhaustion.py. READ-ONLY on the bot.
Usage: venv/bin/python scripts/exh_parity_fixture.py"""
import glob, os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import ssd_bars as SB   # noqa: E402  (extends the k5m / funding sources exactly like the study)
import ta_wl_features as WF   # noqa: E402
TB = SB.TB
D = os.path.join(R, "reports", "SPOT_SHARE_DEEP_data")
M5, M15, H1 = 300_000, 900_000, 3_600_000
N5 = 1500
BIT = 2 * TB.FAMS.index("RSI14X") + 1


def eps_for(u, p):
    f = os.path.join(R, "reports", "FRENZY_TA_CATALOG_data", "episodes_meta.csv") if u == "U1" else os.path.join(D, "episodes_meta.csv")
    e = pd.read_csv(f)
    return e[e.pair == p].sort_values("on_ms").reset_index(drop=True)


def pick():
    out = []
    for f in sorted(glob.glob(os.path.join(D, "bars", "*.pkl"))):
        x = pd.read_pickle(f)
        m = (((x.bits.values >> BIT) & 1).astype(bool) & np.isfinite(x.spot_share_chg.values.astype(float))
             & (x.vol_decay.values >= 0.6366) & (x.spot_share_chg.values < -0.0066))
        if m.any():
            r = x[m].iloc[len(x[m]) // 2]
            out.append((r.u, r.pair, int(r.entry_ms), int(r.ep), float(r.atr_pct), float(r.vol_decay), float(r.spot_share_chg), float(r.q5)))
    return out


def main():
    rows = pick()
    rng = np.random.default_rng(578)
    sel = [rows[i] for i in sorted(rng.choice(len(rows), size=5, replace=False))]
    blob = {}
    for k, (u, p, T, ep, atrp, vd, ssc, q5) in enumerate(sel):
        t, o, h, l, c, v, qv = TB.load(p)
        J = int(np.searchsorted(t, T)); assert t[J] == T
        a = J - N5
        g15, o15, h15, l15, c15, q15 = WF.agg(t, o, h, l, c, qv, M15)
        k15 = int(WF.last_closed(g15, M15, np.array([T]))[0])
        r = TB.rsi(c15, 14)
        e = eps_for(u, p).iloc[ep]
        if u == "C":   # HOLDOUT-C's spot klines live in the study's own folder (ssd_data.py), the catalog's in EDD / TA_WL (ta_wl_features)
            WF.D = D
        sp, months = WF.spot_frame(p)
        WF.D = os.path.join(R, "reports", "TA_WL_DEEP_2026-11-17_data")
        sp = sp[(sp.open_time >= T - 26 * H1) & (sp.open_time < T)]
        blob[f"r{k}_k5"] = np.c_[t[a:J + 1], o[a:J + 1], h[a:J + 1], l[a:J + 1], c[a:J + 1], v[a:J + 1], qv[a:J + 1]]
        blob[f"r{k}_spot"] = sp[["open_time", "qv"]].values.astype(float)
        vol24 = float(pd.Series(qv).rolling(288, min_periods=288).sum().values[J - 1])
        blob[f"r{k}_meta"] = np.array([T, int(e.on_ms), int(e.spike_ms), atrp, vd, ssc, q5, r[k15 - 1], r[k15], vol24], float)
        blob[f"r{k}_pair"] = np.array(p)
        print(p, u, pd.Timestamp(T, unit="ms"), "rsi", round(r[k15 - 1], 3), round(r[k15], 3), "atr%", round(atrp, 4), "vd", round(vd, 4), "ssc", round(ssc, 5),
              "vol24", round(vol24 / 1e6, 2))
    np.savez_compressed(os.path.join(R, "tests", "fixtures", "exh_parity.npz"), **blob)


if __name__ == "__main__":
    main()
