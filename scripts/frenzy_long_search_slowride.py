#!/usr/bin/env python3
"""FRENZY LONG SEARCH — the one surviving post-hoc lead, SLOW_RIDE (BASE_24h · hold ≤ 168 h · TP +20 % maker · no stop · 10-slot
book, first come): one-size DCR per block, drawdown, open-position profile. Descriptive only (post-hoc). Writes _slowride.csv."""
import os, sys
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R, "scripts"))
import frenzy_long_search_eval as EV   # noqa: E402
EX = "H168_Sx_T20"; DAY = EV.DAY


def main():
    df, _ = EV.load()
    d = EV.dedupe(df[(df.ck == 24) & df[EX + "_complete"]], EX)
    d = d.assign(n=EV.net(d, EX)).sort_values("entry_ms")
    ox = []; keep = []
    for e, x in zip(d.entry_ms.values, d[EX + "_x"].values):
        ox = [y for y in ox if y > e]
        if len(ox) < 10:
            ox.append(x); keep.append(True)
        else:
            keep.append(False)
    b = d[np.array(keep)]
    b.drop(columns=["path"]).to_csv(EV.OUT + "_slowride_trades.csv.gz", index=False)
    rows = []
    for blk, (a0, a1) in {"ALL": ("2023-01-01", "2026-11-12"), "HOLD-A": ("2023-01-01", "2024-12-31"), "DESIGN": ("2025-01-01", "2026-06-30"),
                          "HOLD-B": ("2026-07-01", "2026-11-12")}.items():
        d0 = pd.Timestamp(a0).value // 10**6 // DAY; d1 = pd.Timestamp(a1).value // 10**6 // DAY
        s = b[(b.day >= d0) & (b.day <= d1)]
        days = np.arange(d0, d1 + 1)
        for f in (0.05, 0.10, 0.18):
            pnl = pd.Series(s.n.values * f / 100).groupby(s[EX + "_x"].values // DAY).sum().reindex(days).fillna(0).values
            eq = np.cumprod(1 + pnl)
            rows.append(dict(block=blk, notional=f, N=len(s), trades_day=len(s) / len(days), avg=s.n.mean(),
                             dcr=(eq[-1] ** (1 / len(days)) - 1) * 100, maxdd=(eq / np.maximum.accumulate(eq) - 1).min() * 100,
                             worst_day=pnl.min() * 100))
    o = pd.DataFrame(rows); o.to_csv(EV.OUT + "_slowride.csv", index=False)
    print(o.round(3).to_string())


if __name__ == "__main__":
    main()
