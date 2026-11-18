#!/usr/bin/env python3
"""📉 DISTRIBUTION_SHORT (DECISION_LOG 586) — build the PARITY fixture tests/fixtures/dist_parity.npz from the UNIFIED_SHORT pool the L1 study
read (reports/UNIFIED_SHORT_data: unified_short_lib.load() rows, the builder's own 5m series and metrics). Picks VSW15 signal rows with a
readable top_pos_chg4h — some that pass the frozen cell (≤ −0.01097) and some that fail it — and stores, per row, what a live bot holds at
the signal close T: the 15m candles (the builder's 5m grid grouped by open // 15 min, quote volume summed) for the 499 candles ending at
T, the 5m rows (base volume) of the last 1,500 bars (the FRENZY window), the metrics sum_toptrader_long_short_ratio rows (create_time) of
the 6 h before T, and the STUDY's values (vmult15, bar_ret_pct, atr15_pct, the entry bar's open, vol24, top_pos_chg4h, spike, and the
observe-only stamps rsi1h / di_diff1h / gap15). tests/
test_distribution_short.py recomputes them with services/distribution.py. READ-ONLY on the bot.
Usage: venv/bin/python scripts/dist_parity_fixture.py"""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import unified_short_lib as L       # noqa: E402
import unified_short_build as B     # noqa: E402  (patches the k5m / metrics sources exactly like the builder)
import ta_wl_features as FW         # noqa: E402
TB = B.TB
M5, M15, H1 = 300_000, 900_000, 3_600_000


def main(n_pass=4, n_fail=3):
    X = L.load()
    m = X.g_VSW15.values & np.isfinite(X.top_pos_chg4h.values) & X.is15.values
    rng = np.random.default_rng(586)
    cand = X[m]
    sel = list(rng.choice(np.nonzero(cand.top_pos_chg4h.values <= -0.01097)[0], 30, replace=False)) + \
          list(rng.choice(np.nonzero(cand.top_pos_chg4h.values > -0.01097)[0], 30, replace=False))
    blob, k, npass, nfail = {}, 0, 0, 0
    for i in sel:
        r = cand.iloc[i]
        ok = r.top_pos_chg4h <= -0.01097
        if (ok and npass >= n_pass) or (not ok and nfail >= n_fail):
            continue
        p, T = str(r.pair), int(r.entry_ms)
        x = TB.load(p)
        if x is None:
            continue
        t, o, h, l, c, v, qv = x
        J = int(np.searchsorted(t, T))
        if J >= len(t) or t[J] != T or J < 1500:
            continue
        st, en, O15, H15, L15, C15, V15 = B.agg(t[:J], o[:J], h[:J], l[:J], c[:J], qv[:J], M15)
        g15 = t[:J][st]
        if g15[-1] + M15 != T:
            continue
        a = max(0, len(g15) - 499)
        me = FW.metrics_frame(p)
        if me is None:
            continue
        mm = me[(me.ct >= T - 6 * H1) & (me.ct <= T)]
        blob[f"r{k}_k15"] = np.c_[g15[a:], O15[a:], H15[a:], L15[a:], C15[a:], V15[a:]]
        blob[f"r{k}_k5"] = np.c_[t[J - 1500:J], o[J - 1500:J], h[J - 1500:J], l[J - 1500:J], c[J - 1500:J], v[J - 1500:J]]
        blob[f"r{k}_mx"] = np.c_[mm.ct.values.astype(float), mm.sum_toptrader_long_short_ratio.values.astype(float)]
        blob[f"r{k}_meta"] = np.array([T, float(r.vmult15), float(r.bar_ret_pct), float(r.atr15_pct), float(o[J]), float(r.vol24),
                                       float(r.top_pos_chg4h), float(r.spike_ms), float(r.rsi1h), float(r.di_diff1h), float(r.gap15)], float)
        blob[f"r{k}_pair"] = np.array(p)
        print(p, pd.Timestamp(T, unit="ms"), "vmult", round(float(r.vmult15), 2), "chg4h", round(float(r.top_pos_chg4h), 5), "pass" if ok else "fail")
        k += 1
        npass += ok; nfail += (not ok)
        if npass >= n_pass and nfail >= n_fail:
            break
    np.savez_compressed(os.path.join(R, "tests", "fixtures", "dist_parity.npz"), **blob)


if __name__ == "__main__":
    main()
