#!/usr/bin/env python3
"""🔥 FRENZY_LONG — where should the ATR entry limit sit? Operator (2026-10-02): "ATR 2 % seems too low a cap for frenzy pairs, maybe 2.5 or 3".
Same entries, exit and strict ruler as scripts/frenzy_long_followup.py (stop 3 · trail 5/1.5 · 12 h; 1-minute bars, gap-aware fills, 0.10 %
slippage, costs, funding). The limit of 2.0 was found on the SEEN set (≥ $100M) — so the read that matters is the UNSEEN set ($20–100M),
where no ATR cut was ever fitted. Shown: each ATR BAND on its own (is 2–2.5 % good or bad?) and each CAP (everything up to X).
Usage: venv/bin/python scripts/frenzy_atr_cap_test.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_long_followup as F  # noqa: E402
sys.argv = _a
H, B, FS, ST = F.H, F.B, F.FS, F.ST; OUT = os.path.join(ST.ROOT, "reports", "FRENZY_ATR_CAP_TEST_2026-10-02.md")

if __name__ == "__main__":
    tr = []
    for f in sorted(glob.glob(os.path.join(ST.K5, "*.csv"))):
        pair = os.path.basename(f)[:-4]
        if pair in ("BTCUSDT", "ETHUSDT") or not pair.isascii():
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) >= 288 * 40:
            tr += F.triggers(pair, d)[0]
    TR = pd.DataFrame(tr).sort_values("t").reset_index(drop=True); TR["day"] = pd.to_datetime(TR.t, unit="ms").dt.strftime("%Y-%m-%d"); TR["set"] = np.where(TR.q24 >= 100e6, "SEEN", "UNSEEN")
    paths = {}
    for r in TR.itertuples():
        h, l, c = B.bars1m(r.pair, r.t)
        if len(c) >= 30:
            paths[r.Index] = (h, l, c)
    lo, hi = int(pd.Timestamp("2025-12-30").value // 10**6), int(pd.Timestamp("2026-10-02").value // 10**6); FR = {}
    for p in TR.pair.unique():
        try:
            FR[p] = FS.funding(p, lo, hi)
        except SystemExit:
            FR[p] = None

    def net(G, S=3.0, A=5.0, T=1.5):
        rows = []; free = {}
        for r in G.itertuples():
            if r.t < free.get(r.pair, 0) or r.Index not in paths:
                continue
            res, mins, how = H.walk(*paths[r.Index], r.entry, S, A, T, side=1, slip=0.10, gap=True); free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins))
        E = pd.DataFrame(rows, columns=["i", "r", "mins"]).set_index("i"); X = G.loc[E.index].copy(); X["mins"] = E.mins
        X["net"] = E.r - B.COST + np.array([0.0 if FR.get(r.pair) is None else -FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])
        return X

    def row(X):
        if len(X) < 30:
            return f"{len(X)} | – | – | – | – | –"
        bd = H.boot(X, "day", 1500); s = X.net.sort_values()
        run_ = max((len(z) for z in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
        return (f"{len(X)} | {(X.net > 0).mean() * 100:.0f}% | {X.net.mean():+.2f} | {X[X.day < B.SPLIT].net.mean():+.2f} / {X[X.day >= B.SPLIT].net.mean():+.2f} | "
                f"[{bd[0]:+.2f}, {bd[1]:+.2f}] | {run_}")

    L = ["# 🔥 FRENZY_LONG — the ATR entry limit: 2 %, 2.5 % or 3 %?", "",
         f"{len(TR):,} entries (SEEN ≥ $100M: {(TR.set == 'SEEN').sum():,} · UNSEEN $20–100M: {(TR.set == 'UNSEEN').sum():,}). Stop 3 · trail 5/1.5 · strict ruler. % of position at 1×.", ""]
    HEAD = ["| ATR | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | longest losing run |", "|---|---|---|---|---|---|---|"]
    for sname in ("SEEN", "UNSEEN", "BOTH"):
        G = TR if sname == "BOTH" else TR[TR.set == sname]
        L += [f"## {sname} — by ATR BAND (each band alone)", ""] + HEAD
        for a, b in ((0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 99)):
            L.append(f"| {a:g}–{b:g} % | ".replace("–99 %", "+ %") + row(net(G[(G.atr > a) & (G.atr <= b)])) + " |")
        L += ["", f"## {sname} — by CAP (every entry up to the limit; trades overlap across rows)", ""] + HEAD
        for cap in (1.5, 2.0, 2.5, 3.0, 4.0, 99):
            L.append(f"| ≤ {cap:g} % | ".replace("≤ 99 %", "no limit") + row(net(G[G.atr <= cap])) + " |")
        L.append("")
    L += ["## Also: does a WIDER stop suit the higher-ATR band? (BOTH sets, band 2–3 %)", "", "| Exit | trades | won | per trade | Jan–Apr / May–Sep | 95 % by day | longest losing run |", "|---|---|---|---|---|---|---|"]
    G = TR[(TR.atr > 2.0) & (TR.atr <= 3.0)]
    for S, A, T in ((3.0, 5.0, 1.5), (4.0, 5.0, 2.0), (4.0, 6.0, 2.0), (5.0, 7.5, 2.5)):
        L.append(f"| stop {S:g} · trail {A:g}/{T:g} | " + row(net(G, S, A, T)) + " |")
    L += ["", "## NOT tested", "", "- Delisted pairs, slippage beyond 0.10 %, exchange position limits; no fresh time period.", "- The 2.0 limit was chosen on the SEEN set; any limit chosen from this table is fitted on both sets — haircut applies."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
