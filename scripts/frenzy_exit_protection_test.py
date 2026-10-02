#!/usr/bin/env python3
"""🔥 FRENZY exit — is there a hole between the −3 % stop and the +5 % trail? Operator (2026-10-02), after a manual SCR long on the Frenzy exit
peaked at +1.7 % and closed at the −3 % stop: "the trailing starting at 5 % might be too risky, no protection in between?"
Same entries as the live sleeve (staircase ON, 24 h volume ≥ $20M, 5m ATR ≤ 2.5 %), strict ruler (1-minute bars, gap-aware fills, 0.10 %
slippage, costs, funding), 12 h cap. FROZEN variants, all with the −3 % stop:
  BASE      trail arms +5, gives back 1.5                                  (live)
  BE x      as BASE, plus: once the trade has been +x % (1.5 / 2 / 3), the stop moves to break-even (+0.1 %)
  STEP      as BASE, plus: at +2 % the stop moves to −1 %, at +3.5 % to +0.5 %
  HALF x    as BASE, plus: once +x % (2 / 3), the stop moves to −1.5 % (half the risk)
  TRAIL a/g the trail arms earlier: 3/1.5 · 3/2 · 4/1.5 · 4/2 · 2/2
Usage: venv/bin/python scripts/frenzy_exit_protection_test.py"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_a, sys.argv = sys.argv, [sys.argv[0]]
import frenzy_long_followup as F  # noqa: E402
sys.argv = _a
H, B, FS, ST = F.H, F.B, F.FS, F.ST; OUT = os.path.join(ST.ROOT, "reports", "FRENZY_EXIT_PROTECTION_TEST_2026-10-02.md"); SLIP = 0.10


def walk(h, l, c, e, steps, A, T, S=3.0):
    """LONG. steps = ((peak %, new stop %), …) ascending. Stop first (gap-aware), then the trail, both from the best price of PRIOR bars."""
    best = e
    for i in range(len(c)):
        prev = c[i - 1] if i else e; pk = (best / e - 1) * 100
        stop = -S
        for arm, lvl in steps:
            if pk >= arm:
                stop = lvl
        sp = e * (1 + stop / 100)
        if l[i] <= sp:
            return (min(sp, prev) / e - 1) * 100 - SLIP, i + 1, ("stop" if stop == -S else "lock")
        if pk >= A and l[i] <= best * (1 - T / 100):
            return (min(best * (1 - T / 100), prev) / e - 1) * 100 - SLIP, i + 1, "trail"
        best = max(best, h[i])
    return (c[-1] / e - 1) * 100, len(c), "cap"


VARIANTS = (("BASE (live): trail 5/1.5", (), 5.0, 1.5), ("BE at +1.5 %", ((1.5, 0.1),), 5.0, 1.5), ("BE at +2 %", ((2.0, 0.1),), 5.0, 1.5), ("BE at +3 %", ((3.0, 0.1),), 5.0, 1.5),
            ("STEP: +2 → −1 %, +3.5 → +0.5 %", ((2.0, -1.0), (3.5, 0.5)), 5.0, 1.5), ("HALF at +2 % (stop → −1.5 %)", ((2.0, -1.5),), 5.0, 1.5), ("HALF at +3 % (stop → −1.5 %)", ((3.0, -1.5),), 5.0, 1.5),
            ("TRAIL 4/1.5", (), 4.0, 1.5), ("TRAIL 4/2", (), 4.0, 2.0), ("TRAIL 3/1.5", (), 3.0, 1.5), ("TRAIL 3/2", (), 3.0, 2.0), ("TRAIL 2/2", (), 2.0, 2.0))

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
    TR = TR[TR.atr <= 2.5]
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

    def net(G, steps, A, T):
        rows = []; free = {}
        for r in G.itertuples():
            if r.t < free.get(r.pair, 0) or r.Index not in paths:
                continue
            res, mins, how = walk(*paths[r.Index], r.entry, steps, A, T); free[r.pair] = r.t + mins * 60_000; rows.append((r.Index, res, mins, how))
        E = pd.DataFrame(rows, columns=["i", "r", "mins", "how"]).set_index("i"); X = G.loc[E.index].copy(); X["mins"] = E.mins; X["how"] = E.how
        X["net"] = E.r - B.COST + np.array([0.0 if FR.get(r.pair) is None else -FR[r.pair][(FR[r.pair].t > r.t) & (FR[r.pair].t <= r.t + r.mins * 60_000)].rate.sum() * 100 for r in X.itertuples()])
        return X

    L = ["# 🔥 FRENZY exit — protection between the −3 % stop and the +5 % trail", "",
         f"{len(TR):,} entries with ATR ≤ 2.5 % (SEEN ≥ $100M: {(TR.set == 'SEEN').sum()} · UNSEEN $20–100M: {(TR.set == 'UNSEEN').sum()}). Stop −3 % in every variant. % of position at 1×, strict ruler.", ""]
    X0 = net(TR, (), 5.0, 1.5)
    pk = []
    for r in X0.itertuples():
        h, l, c = paths[r.Index]; n = int(r.mins); pk.append((max(h[:n].max(), r.entry) / r.entry - 1) * 100)
    X0["peak"] = pk; st = X0[X0.how == "stop"]
    L += ["## Where the full −3 % stops come from (live exit)", "",
          f"{len(st)} of {len(X0)} trades end at the full stop. Their best point before stopping: never above +0.5 % {((st.peak < 0.5).mean() * 100):.0f}% · +0.5–1.5 % {(((st.peak >= 0.5) & (st.peak < 1.5)).mean() * 100):.0f}% · "
          f"+1.5–3 % {(((st.peak >= 1.5) & (st.peak < 3)).mean() * 100):.0f}% · +3–5 % {(((st.peak >= 3) & (st.peak < 5)).mean() * 100):.0f}%. "
          f"So {((st.peak >= 1.5).mean() * 100):.0f}% of the full stops had been at least +1.5 % in profit first (the SCR case).", ""]
    for sname in ("BOTH", "SEEN", "UNSEEN"):
        G = TR if sname == "BOTH" else TR[TR.set == sname]
        L += [f"## {sname} — {len(G)} entries", "", "| Exit | trades | won | full stops | locks | per trade | Jan–Apr / May–Sep | 95 % by day | avg win / loss | best 5 % removed | longest losing run |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, steps, A, T in VARIANTS:
            X = net(G, steps, A, T); bd = H.boot(X, "day", 1500); s = X.net.sort_values()
            run_ = max((len(z) for z in "".join("L" if v <= 0 else "W" for v in X.sort_values("t").net).split("W")), default=0)
            L.append(f"| {name} | {len(X)} | {(X.net > 0).mean() * 100:.0f}% | {(X.how == 'stop').mean() * 100:.0f}% | {(X.how == 'lock').mean() * 100:.0f}% | {X.net.mean():+.2f} | {X[X.day < B.SPLIT].net.mean():+.2f} / {X[X.day >= B.SPLIT].net.mean():+.2f} | "
                     f"[{bd[0]:+.2f}, {bd[1]:+.2f}] | {X[X.net > 0].net.mean():+.1f} / {X[X.net <= 0].net.mean():+.1f} | {s.iloc[:-max(1, int(len(s) * 0.05))].mean():+.2f} | {run_} |")
        L.append("")
    L += ["## NOT tested", "", "- Delisted pairs, slippage beyond 0.10 %, exchange position limits; no fresh time period — 12 variants on the same data, so the best row is optimistic.",
          "- A lock at break-even is assumed to fill like a stop (gap-aware + 0.10 %)."]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
