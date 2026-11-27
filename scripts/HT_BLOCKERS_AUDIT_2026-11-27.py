#!/usr/bin/env python3
"""HT_BLOCKERS_AUDIT_2026-11-27 — the three armed WILLY_HT entry blockers on the 2-year backtest at TODAY's HT exit (research only).
(operator 2026-11-27: "why we are not having HT trades, the blockers, and the audit of the blockers in backtest")

PRIOR (re-used, not re-run): DL 527 / 628 FAILED_SPIKE armed + live revert gate frozen KEEP (blocked 23 · 13 % · −2.90 vs kept +1.00) ·
HT_FAILED_SPIKE_PARITY 11-19 (engine = rebuild on 47 / 47; the 1,142 unfiltered rebuild entries, priced only where the decision can flip) ·
DL 594 NEAR_HIGH (11-18 screen, partly in-sample, null not cleared) · DL 619 BURST armed (backtest burst zone 39 of 300) · HT_WILLY 11-20
(300 = after FAILED_SPIKE + dip) · HT_FILTER_BT 11-26 (CROWDED / PRE_VOL — not armed). NEW: every blocker judged on the SAME unfiltered
cohort at today's exit (TP +6 net / SL −4 / 24 h, funding priced) — alone, as the sole blocker, and the stack before → after.

COHORT   reports/HT_FAILED_SPIKE_PARITY_2026-11-19_fullhist.csv: 1,142 unfiltered rebuild HT entries 2025-01 → 2026-11-17 (pair, entry_ms,
         vs_vwap = the engine's FAILED_SPIKE input). Spike high H from the 11-20 cohort where known (416 FAILED_SPIKE-kept rows), else the
         max 1m high of the 6 h before the entry (review: on the 415 known rows the proxy catches only 63 of 117 true NEAR_HIGH blocks and
         flags 5 wrongly — it is used only on FAILED_SPIKE-blocked rows, so "NEAR_HIGH alone", "FS sole" and "today minus FAILED_SPIKE" carry
         a noisy NEAR_HIGH mask; the deep NEAR_HIGH read uses the true H on 372 / 373 rows). E = 1m open at the entry minute × 1.001 (as the cohort).
BLOCKERS FAILED_SPIKE vs_vwap ≤ 0 · NEAR_HIGH dip = (1 − open / H) × 100 < 1.4 · BURST pre_vol15_vs60 > 1.60 ∧ cloc15 < 0.50 (unreadable →
         never blocks, like the engine).
PRICE    scripts/STATE_EXIT_2026-11-21_common.walk on 1m (stop first inside a bar, 0.10 slip, taker / maker fees, funding). Parity: 299 of
         fills_out's 300 kept rows (tp6) re-priced identically; TUTUSDT 2025-10-10 (a +6 TP) is not in the 1,142 parity set — with it
         "today" moves Σ −5 → ≈ +1. One position per coin: a later entry while the coin's previous KEPT trade is open is
         skipped (per scenario).
READ     kept vs blocked per blocker: N · WR · avg · Σ · per half-year · DESIGN (< 2025-07) / HOLDOUT · day-bootstrap P(blocked mean < 0) ·
         sleeve before → after: avg, DCR (one size 0.61), max DD, worst day. HT breakeven WR at +6 / −4 ≈ 40 %.
Out: reports/HT_BLOCKERS_AUDIT_2026-11-27.csv · _summary.txt"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
import importlib
C = importlib.import_module("STATE_EXIT_2026-11-21_common")
from services.frenzy import pre_vol15_vs60, cloc15        # noqa: E402
from services.watch_filters import ht_burst_state        # noqa: E402

MIN, HR, DAY = 60_000, 3_600_000, 86_400_000
HO = int(pd.Timestamp("2025-07-01").value // 10**6)
SZ = 0.61
OUT = os.path.join(R, "reports", "HT_BLOCKERS_AUDIT_2026-11-27")


def ht_px(E, tp_net, sl_net):
    return E * (100 + tp_net + C.TAKER) / (100 - C.MAKER), E * (100 - sl_net + C.TAKER) / (100 - C.TAKER)


def one(r):
    p, te = r["pair"], int(r["te"])
    a = C.k1m(p, te - 6 * HR, te + 24 * HR + MIN)
    if a is None or not len(a["t"]):
        return None
    g = C.grid(a, te - 6 * HR, te + 24 * HR + MIN)
    T = g["t"]
    i0 = int(np.searchsorted(T, te))
    if i0 >= len(T) or not g["present"][i0]:
        return None
    live = float(g["o"][i0])
    E = live * 1.001
    H = r["H"] if r["H"] == r["H"] else float(np.nanmax(g["h"][:i0])) if i0 > 0 else np.nan
    Hp = float(np.nanmax(g["h"][:i0])) if i0 > 0 else np.nan
    dip = (1 - live / H) * 100 if H == H and H > 0 else np.nan
    rows = [[int(t), o, h, l, c, v] for t, o, h, l, c, v in zip(a["t"], a["o"], a["h"], a["l"], a["c"], a["v"]) if te - 80 * MIN <= int(t) < te // MIN * MIN]
    pv, cl = pre_vol15_vs60(rows, te), cloc15(rows, te)
    tp, sl = ht_px(E, 6.0, 4.0)
    end_i = min(i0 + 24 * 60, len(T))
    fs_t, fs_r = C.funding(p)
    F = C.fund_cum(T, g["o"], E, 1, fs_t, fs_r, te)
    w = C.walk(g, i0, end_i, E, 1, tp, sl, F, 0.10, C.TAKER, "price", 4.0)
    if w is None:
        return None
    return dict(live=live, E=E, H_used=H, H_proxy=Hp, dip=dip, pre_vol=pv, cloc=cl, burst=ht_burst_state(pv, cl),
                net=w["net"], why=w["why"], xt=int(T[w["j"]]), cover=float(g["present"][i0:end_i].mean()))


def one_per_coin(d):
    """keep the first entry per coin; skip later ones while its kept trade is still open. d sorted by te."""
    last = {}
    keep = []
    for p, te, xt in zip(d.pair, d.te, d.xt):
        if p in last and te < last[p]:
            keep.append(False); continue
        last[p] = xt; keep.append(True)
    return d[np.array(keep, bool)]


def book(d):
    """DCR (one size), max DD, worst day over every calendar day of the cohort span."""
    if not len(d):
        return 0.0, 0.0, 0.0
    day = pd.to_datetime(d.xt, unit="ms").dt.floor("D")
    days = pd.date_range(pd.to_datetime(FULL.te.min(), unit="ms").floor("D"), pd.to_datetime(FULL.xt.max(), unit="ms").floor("D"))
    r = (d.net * SZ / 100).groupby(day).sum().reindex(days, fill_value=0.0)
    eq = (1 + r).cumprod()
    return (eq.iloc[-1] ** (1 / len(days)) - 1) * 100, float((eq / eq.cummax() - 1).min() * 100), float(r.min() * 100)


def boot_neg(x, day, n=2000, seed=1127):
    x = np.asarray(x); day = np.asarray(day)
    u = np.unique(day)
    if len(u) < 2:
        return np.nan
    idx = {k: np.nonzero(day == k)[0] for k in u}
    rng = np.random.default_rng(seed)
    m = []
    for _ in range(n):
        pick = rng.choice(u, len(u))
        v = np.concatenate([x[idx[k]] for k in pick])
        m.append(v.mean())
    return float((np.array(m) < 0).mean())


def stat(g):
    if not len(g):
        return "0"
    return f"{len(g)} · {(g.net > 0).mean():.0%} · {g.net.mean():+.2f} · Σ {g.net.sum():+.0f}"


FULL = None


def main():
    global FULL
    f = pd.read_csv(os.path.join(R, "reports", "HT_FAILED_SPIKE_PARITY_2026-11-19_fullhist.csv"))
    k = pd.read_pickle(os.path.join(R, "reports", "HT_WILLY_2026-11-20_data", "cohorts.pkl"))["HT_nodip"][["pair", "ts", "H", "dip"]]
    A = f.rename(columns={"entry_ms": "te"}).merge(k.rename(columns={"ts": "te", "dip": "dip_known"}), on=["pair", "te"], how="left")
    need = sorted({(p, ds) for p, te in zip(A.pair, A.te) for ds in C.days_of(int(te) - 6 * HR, int(te) + 25 * HR)})
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(lambda x: C.fetch_kday(*x), need))
    print(f"klines: {len(need)} pair-days, {sum(got)} fetched", flush=True)
    rows = []
    for r in A.to_dict("records"):
        o = one(r)
        if o:
            rows.append({**r, **o})
    D = pd.DataFrame(rows).sort_values("te").reset_index(drop=True)
    D["FS"] = D.vs_vwap <= 0
    D["NH"] = D.dip < 1.4
    D["BU"] = D.burst == "zone"
    ts = pd.to_datetime(D.te, unit="ms")
    D["day"] = ts.dt.date
    D["half"] = ts.dt.year.astype(str) + np.where(ts.dt.month <= 6, "-H1", "-H2")
    D["blk"] = np.where(D.te < HO, "DESIGN", "HOLDOUT")
    FULL = D
    D.to_csv(OUT + ".csv", index=False)
    L = [f"HT blockers audit — {len(D)} of {len(A)} unfiltered rebuild HT entries priced at TP +6 / SL −4 / 24 h (1m, funding, 0.10 slip)"]
    # parity on the 300 kept rows (fills_out tp6)
    fo = pd.read_csv(os.path.join(R, "reports", "STATE_EXIT_2026-11-21_data", "fills_out.csv"), low_memory=False)
    fo = fo[(fo.kind == "BT") & (fo.sleeve == "HT")][["pair", "te", "tp6"]]
    P = D.merge(fo, on=["pair", "te"])
    L.append(f"parity vs fills_out tp6: {len(P)} matched · same sign {np.mean(np.sign(P.net) == np.sign(P.tp6)):.1%} · "
             f"|Δ| ≤ 0.05 {np.mean((P.net - P.tp6).abs() <= 0.05):.1%} · mean Δ {(P.net - P.tp6).mean():+.3f}")
    kn = D[D.dip_known.notna()]
    L.append(f"H proxy (6 h max) vs the known spike high on {len(kn)}: NEAR_HIGH decision agrees "
             f"{np.mean(((1 - kn.live / kn.H_proxy) * 100 < 1.4) == (kn.dip_known < 1.4)):.1%}")
    L.append(f"burst readable {(D.burst != 'unread').mean():.0%}")
    lines = [("FAILED_SPIKE", "FS"), ("NEAR_HIGH", "NH"), ("BURST", "BU")]
    allb = D.FS | D.NH | D.BU
    L += ["", "## Each blocker alone on the unfiltered cohort (no one-per-coin) — blocked | kept",
          "| blocker | blocked N · WR · avg · Σ | kept | DESIGN blocked / kept | HOLDOUT blocked / kept | P(blocked mean<0) day-boot | sole blocker |",
          "|---|---|---|---|---|---|---|"]
    for name, c in lines:
        b, kk = D[D[c]], D[~D[c]]
        others = [x for _, x in lines if x != c]
        sole = D[D[c] & ~D[others].any(axis=1)]
        L.append(f"| {name} | {stat(b)} | {stat(kk)} | {b[b.blk == 'DESIGN'].net.mean():+.2f} / {kk[kk.blk == 'DESIGN'].net.mean():+.2f} | "
                 f"{b[b.blk == 'HOLDOUT'].net.mean():+.2f} / {kk[kk.blk == 'HOLDOUT'].net.mean():+.2f} | {boot_neg(b.net, b.day):.2f} | {stat(sole)} |")
    L += ["", "## Half-years — blocked avg (N) per blocker | kept by all three"]
    for h, g in D.groupby("half"):
        L.append(f"{h}: " + " · ".join(f"{n} {g[g[c]].net.mean():+.2f} ({int(g[c].sum())})" for n, c in lines)
                 + f" · kept {g[~(g.FS | g.NH | g.BU)].net.mean():+.2f} ({int((~(g.FS | g.NH | g.BU)).sum())})")
    L += ["", "## Sleeve before → after (one position per coin, one size 0.61; DCR over every calendar day of the span)",
          "| stack | N · WR · avg · Σ | DCR %/day | max DD | worst day |", "|---|---|---|---|---|"]
    scen = [("no blockers", pd.Series(True, index=D.index)), ("FAILED_SPIKE only", ~D.FS), ("NEAR_HIGH only", ~D.NH), ("BURST only", ~D.BU),
            ("FAILED_SPIKE + NEAR_HIGH", ~(D.FS | D.NH)), ("today: all three", ~allb),
            ("all three minus FAILED_SPIKE", ~(D.NH | D.BU)), ("all three minus NEAR_HIGH", ~(D.FS | D.BU)), ("all three minus BURST", ~(D.FS | D.NH))]
    for name, m in scen:
        d = one_per_coin(D[m])
        dcr, dd, wd = book(d)
        L.append(f"| {name} | {stat(d)} | {dcr:+.3f} | {dd:.1f} % | {wd:.1f} % |")
    L += ["", "## 2026 only (the live year) — same stacks"]
    D26 = D[pd.to_datetime(D.te, unit="ms").dt.year == 2026]
    for name, m in scen:
        d = one_per_coin(D26[m.loc[D26.index]])
        L.append(f"{name}: {stat(d)}")
    open(OUT + "_summary.txt", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
