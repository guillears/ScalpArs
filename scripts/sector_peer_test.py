#!/usr/bin/env python3
"""🧩 SECTOR PEERS — operator lead (2026-10-02): SAND spiked on 258× volume at 07:05 and MANA / AXS / ENJ / GALA all made their first
move 10–20 min later and continued for an hour. "When one pair of a sector spikes on huge volume, buy its sector peers on their
first move." One event on one day → tested on the year. PRE-DECLARED before any result:
  sectors   Binance futures underlyingSubType tags (TODAY's tags applied to history — a known approximation), except the generic
            Crypto / TradFi / Alpha / Index / Pre-IPO. NARROW = tags with ≤ 35 pairs (Metaverse, Gaming, NFT, Layer-2, PoW, Payment,
            Storage, RWA, Chinese …); BROAD = the rest (DeFi, Infrastructure, AI, Layer-1, Meme). A pair belongs to each of its tags.
  leader    a 5m close with 30-min return ≥ +5 % ∧ last-hour volume ≥ 20× the pair's normal hour (median hourly volume over the
            30 days ending a day earlier) ∧ last-hour volume ≥ $2M. First leader event per SECTOR per 12 h = one sector event.
  FIRST-MOVE entry   each OTHER pair of the sector: its first 5m close within 3 h after the leader bar with 30-min return ≥ +3 %
  IMMEDIATE entry    every other pair of the sector at the leader bar's close
  exits (5m bars, stop first inside a bar, cost 0.11 %):  A  +2.0 / −1.5, 2 h limit   ·   B  hold 60 min, −2.0 stop
  control   the SAME first-move rule on the same pairs at times with NO leader event in any of the pair's sectors in the prior 3 h
  PASS      a cell passes when, on sector-event means: > 0 in both halves ∧ 95 % interval above 0 ∧ above its control in both halves
Usage: venv/bin/python scripts/sector_peer_test.py → reports/SECTOR_PEER_TEST_2026-10-02.md"""
import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); BC = os.path.join(ROOT, "reports", "backtest_cache"); K5 = os.path.join(BC, "k5m_full")
OUT = os.path.join(ROOT, "reports", "SECTOR_PEER_TEST_2026-10-02.md"); BAR = 300_000; COST = 0.11; SPLIT = int(pd.Timestamp("2026-05-01").value // 10**6)
GENERIC = {"Crypto", "TradFi", "Alpha", "Index", "Pre-IPO"}


def tq(n):
    z = 1.959964; d = max(n - 1, 1); return z + (z**3 + z) / (4 * d) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * d * d)


def out_A(o, h, l, c, i):          # +2.0 / −1.5, 2 h
    e = c[i]
    for j in range(i + 1, min(i + 25, len(c))):
        if l[j] <= e * 0.985:
            return -1.5
        if h[j] >= e * 1.02:
            return 2.0
    return (c[min(i + 24, len(c) - 1)] / e - 1) * 100


def out_B(o, h, l, c, i):          # 60 min, −2.0 stop
    e = c[i]
    for j in range(i + 1, min(i + 13, len(c))):
        if l[j] <= e * 0.98:
            return -2.0
    return (c[min(i + 12, len(c) - 1)] / e - 1) * 100


if __name__ == "__main__":
    tags = json.load(open(os.path.join(BC, "sector_map.json"))); P = {}
    for f in sorted(glob.glob(os.path.join(K5, "*.csv"))):
        p = os.path.basename(f)[:-4]; tg = [t for t in tags.get(p, []) if t not in GENERIC]
        if not tg or p in ("BTCUSDT", "ETHUSDT"):
            continue
        d = pd.read_csv(f).drop_duplicates("open_time").sort_values("open_time")
        if len(d) < 288 * 40:
            continue
        t = d.open_time.values.astype("int64"); c = d.c.values; q1h = d.qvol.rolling(12).sum(); norm = q1h.shift(288).rolling(288 * 30).median()
        r30 = (d.c / d.c.shift(6) - 1).values * 100
        lead = (r30 >= 5) & ((q1h / norm).values >= 20) & (q1h.values >= 2e6)
        P[p] = dict(t=t, o=d.o.values, h=d.h.values, l=d.l.values, c=c, r30=r30, lead=lead, tags=tg, pos={int(x): k for k, x in enumerate(t)})
    sectors = {}
    for p, v in P.items():
        for g in v["tags"]:
            sectors.setdefault(g, []).append(p)
    narrow = {g for g, m in sectors.items() if sum(1 for _ in m) <= 35}
    ev = []                                                             # (sector, t0, leader)
    for g, mem in sectors.items():
        cand = sorted((int(P[p]["t"][i]), p) for p in mem for i in np.nonzero(P[p]["lead"])[0]); last = -10**18
        for t0, p in cand:
            if t0 - last >= 12 * 3600_000:
                ev.append((g, t0, p)); last = t0
    lead_times = {g: np.array(sorted(t for s, t, _ in ev if s == g)) for g in sectors}
    rows = []
    for g, t0, ld in ev:
        for p in sectors[g]:
            if p == ld:
                continue
            v = P[p]; i0 = v["pos"].get(t0)
            if i0 is None or i0 + 40 >= len(v["c"]):
                continue
            rows.append(dict(kind="IMMEDIATE", sector=g, t0=t0, pair=p, A=out_A(v["o"], v["h"], v["l"], v["c"], i0), B=out_B(v["o"], v["h"], v["l"], v["c"], i0)))
            w = np.nonzero(v["r30"][i0 + 1:i0 + 37] >= 3)[0]
            if len(w):
                i = i0 + 1 + int(w[0])
                rows.append(dict(kind="FIRST-MOVE", sector=g, t0=t0, pair=p, A=out_A(v["o"], v["h"], v["l"], v["c"], i), B=out_B(v["o"], v["h"], v["l"], v["c"], i), lag=(i - i0) * 5))
    for p, v in P.items():                                              # control: first moves with no sector leader in the prior 3 h
        idx = np.nonzero(v["r30"] >= 3)[0]; last = -10**9; lt = np.concatenate([lead_times[g] for g in v["tags"]]) if v["tags"] else np.array([])
        for i in idx:
            if i - last < 36 or i + 40 >= len(v["c"]):
                continue
            last = i; t = v["t"][i]
            if len(lt) and ((lt <= t) & (lt >= t - 3 * 3600_000)).any():
                continue
            for g in v["tags"]:
                rows.append(dict(kind="CONTROL", sector=g, t0=int(t) // (6 * 3600_000) * (6 * 3600_000), pair=p, A=out_A(v["o"], v["h"], v["l"], v["c"], i), B=out_B(v["o"], v["h"], v["l"], v["c"], i)))
    T = pd.DataFrame(rows); T["A"] -= COST; T["B"] -= COST; T["half"] = np.where(T.t0 < SPLIT, 1, 2); T["scope"] = np.where(T.sector.isin(narrow), "NARROW", "BROAD")
    T.to_csv(os.path.join(BC, "sector_peer_trades.csv"), index=False)
    L = ["# 🧩 SECTOR PEERS — buy a sector's other pairs after one of them spikes on huge volume", "",
         f"{len(ev)} sector events ({sum(1 for g, _, _ in ev if g in narrow)} in narrow sectors) on {len(sectors)} tags, {len(P)} pairs, Jan–Sep 2026. % of position after 0.11 % costs.",
         "Narrow sectors: " + ", ".join(f"{g} ({len(sectors[g])})" for g in sorted(narrow)) + " · broad: " + ", ".join(f"{g} ({len(sectors[g])})" for g in sorted(set(sectors) - narrow)), "",
         "| Scope | Entry | Exit | trades | sector events | won | per event Jan–Apr / May–Sep | by event [95 %] | control Jan–Apr / May–Sep | PASS |", "|---|---|---|---|---|---|---|---|---|---|"]
    for sc in ("NARROW", "BROAD"):
        for kind in ("FIRST-MOVE", "IMMEDIATE"):
            for ex, lab in (("A", "+2.0 / −1.5, 2 h"), ("B", "60 min, −2.0 stop")):
                g = T[(T.scope == sc) & (T.kind == kind)]; c = T[(T.scope == sc) & (T.kind == "CONTROL")]
                em = g.groupby(["sector", "t0"])[ex].mean(); h = g.groupby(["sector", "t0"]).half.first(); n = len(em)
                se = em.std(ddof=1) / np.sqrt(n); lo, hi = em.mean() - tq(n) * se, em.mean() + tq(n) * se
                a1, a2 = em[h == 1].mean(), em[h == 2].mean(); c1, c2 = c[c.half == 1][ex].mean(), c[c.half == 2][ex].mean()
                ok = a1 > 0 and a2 > 0 and lo > 0 and a1 > c1 and a2 > c2
                L.append(f"| {sc} | {kind} | {lab} | {len(g):,} | {n} | {(g[ex] > 0).mean() * 100:.0f}% | {a1:+.3f} / {a2:+.3f} | {em.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {c1:+.3f} / {c2:+.3f} | {'✅' if ok else '—'} |")
    g = T[(T.kind == "FIRST-MOVE")]; L += ["", "FIRST-MOVE, exit B, by sector: " + " · ".join(f"{s} {x.groupby('t0').B.mean().mean():+.2f} ({x.t0.nunique()} ev)" for s, x in g.groupby("sector")),
                                         "FIRST-MOVE by lag after the leader (exit B, per trade): " + " · ".join(f"{a}-{b} min {x.B.mean():+.2f} ({len(x)})" for a, b in ((5, 30), (35, 90), (95, 180)) for x in [g[(g.lag >= a) & (g.lag <= b)]])]
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L))
