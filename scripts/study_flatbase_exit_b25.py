#!/usr/bin/env python3
"""FLATBASE_SHORT exit before → after, the full DECISION_LOG 406 table (operator 2026-10-28: "where is the avg duration before and after?").

Every frozen exit shadow of the FLATBASE scout (TP_SHADOWS, DL 367 / 370 / 390) vs the live exit, re-walked on each real fill's own 5m path
with the scout's own walker (scout_flatbase_short.tp_shadow_eval — fees, taker slip on stop / time, no funding; cache only), per block:
  ① current batch B25 (opened ≥ 2026-10-23 18:30) · ② master per batch (the archived FLATBASE fills — B24) · ③ backtest (the exit study's
  per-half Δ, the scout's tp_backtest_note).
Columns: N · days · WR · avg % · Σ pts · $ at each fill's ACTUAL notional · Δ$ · avg / median hold (HH:MM) · DCR on the real account path
(B25: $2,900 start, +$7,100 top-up 10-26 13:50) · day-block bootstrap P(Δ > 0) · two-sided split (live-stopped saved / unchanged / deeper ·
live-TP give-back / cut winner) · top pair share of the gain · 🧮 leave-one-out · the 30–50 % in-sample haircut on Δ.
Usage: venv/bin/python -I scripts/study_flatbase_exit_b25.py <current orders export> [--fetch]   (cache only by default)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_flatbase_short as FS  # noqa: E402
import review_math as RM  # noqa: E402

B25 = pd.Timestamp("2026-10-23 18:30")
B25_START, B25_FLOWS = 2900.0, [("2026-10-26T13:50:00", 7100.0)]


def walks(real, fetch=False):
    from scout_long_stop4 import Budget
    bud = Budget()
    now = int(pd.Timestamp.now("UTC").value // 1_000_000)
    out = []
    for r in real.itertuples():
        st, w = FS.tp_shadow_eval(r, bud, now, fetch=fetch)   # cache only unless --fetch (Binance request budget)
        if st != "PRICED":
            out.append(None)
            continue
        o_ms = int(pd.Timestamp(r.o).value // 1_000_000)
        lead = (FS.tp_path_start(o_ms)[0] - o_ms) / 60_000.0
        out.append({k: (float(v["net_pct"]), lead + (v["bars"] - 0.5) * 5.0, str(v.get("reason"))) for k, v in w.items()})
    return out


def boot_p(delta, days, n=4000, seed=7):
    """day-block bootstrap P(mean Δ > 0)."""
    rng = np.random.default_rng(seed)
    g = pd.Series(delta.values).groupby(days.values).apply(list).tolist()
    if len(g) < 2:
        return float("nan")
    hits = 0
    for _ in range(n):
        pick = [g[i] for i in rng.integers(0, len(g), len(g))]
        hits += np.mean([x for b in pick for x in b]) > 0
    return hits / n


def split(live_reason, live, cell):
    """two-sided split on the REAL close (review 2: the live-stopped cohort is the booked one, never the replica's reason)."""
    out = {}
    for lr, a, b in zip(live_reason, live, cell):
        kind = "STOP" if "STOP" in lr.upper() else ("TP" if "TP" in lr.upper() else "TIME")
        d = b - a
        tag = (("stopped→saved" if d > 1e-6 else "stopped→deeper" if d < -1e-6 else "stopped→unchanged") if kind == "STOP" else
               ("TP→cut winner" if b < 0 < a else "TP→give-back" if d < -1e-6 else "TP→unchanged") if kind == "TP" else "time")
        n, s = out.get(tag, (0, 0.0))
        out[tag] = (n + 1, s + d)
    return " · ".join(f"{k} {n} ({s:+.1f})" for k, (n, s) in out.items())


def block(name, rows, W, account=None):
    keep = [i for i, w in enumerate(W) if w is not None]
    miss = [f"{rows.pair.iloc[i]} {rows.o.iloc[i]:%m-%d %H:%M}" for i, w in enumerate(W) if w is None]
    if miss:
        print(f"\n⚠ {name}: {len(miss)} fill(s) NOT priced (path unreadable / window open): {', '.join(miss)}")
    if not keep:
        print(f"\n### {name}: 0 priced fills")
        return
    r = rows.iloc[keep].reset_index(drop=True)
    W = [W[i] for i in keep]
    notional = (pd.to_numeric(r.entry_price, errors="coerce") * pd.to_numeric(r.quantity, errors="coerce")).fillna(0).values   # the ACTUAL notional
    days = r.o.dt.date.astype(str)
    live = np.array([w["LIVE"][0] for w in W])
    print(f"\n### {name} — {len(r)} fills on {days.nunique()} days · booked avg {r.pct.mean():+.2f} % (re-walk parity: "
          f"{np.mean(np.abs(live - r.pct.values) <= 0.5):.0%} within ±0.5)")
    print("| Exit (TP / SL) | N | WR | avg % | Σ pts | $ (actual notional) | Δ$ vs today | avg hold | median hold | DCR %/day | "
          "P(Δ>0) day-boot | split (live-stopped / live-TP) | top pair share of Δ | 🧮 leave-one-out | Δ avg after 30–50 % haircut |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    base_usd = live / 100 * notional
    for k in ("LIVE",) + tuple(FS.TP_SHADOWS):
        tp, sl = FS._tp_live() if k == "LIVE" else FS.TP_SHADOWS[k]
        v = np.array([w[k][0] for w in W])
        h = np.array([w[k][1] for w in W])
        usd = v / 100 * notional
        d = v - live
        dcr = "–"
        if account is not None:
            acc, t0, t1 = account
            a = acc.copy()
            key = r.o.dt.strftime("%Y-%m-%dT%H:%M") + r.pair
            dmap = dict(zip(key, (v - live) / 100 * notional))
            assert int(a.k.isin(dmap).sum()) == len(r), f"account join matched {int(a.k.isin(dmap).sum())} of {len(r)} fills"
            a["u"] = a.usd_live + a.k.map(dmap).fillna(0.0)
            dcr = f"{RM.dcr_account(a, a.entry_strategy == 'FLATBASE_SHORT', 'u', B25_START, t0, t1, B25_FLOWS):+.3f}"
        lab = r.pair + " " + r.o.dt.strftime("%m-%d")
        pos = pd.Series(np.clip(d, 0, None)).groupby(r.pair.values).sum()
        share = f"{pos.max() / pos.sum():.0%} ({pos.idxmax()})" if pos.sum() > 0 else "–"
        p = boot_p(pd.Series(d), days) if k != "LIVE" else float("nan")
        hc = f"{d.mean() * 0.5:+.2f} … {d.mean() * 0.7:+.2f}" if k != "LIVE" else "–"
        print(f"| {'**TODAY**' if k == 'LIVE' else k} +{tp:g} / −{sl:g} | {len(v)} | {(v > 0).mean():.0%} | {v.mean():+.2f} | {v.sum():+.2f} | "
              f"{usd.sum():+,.0f} | {(usd - base_usd).sum():+,.0f} | {FS.hhmm(h.mean())} | {FS.hhmm(float(np.median(h)))} | {dcr} | "
              f"{'–' if np.isnan(p) else f'{p:.2f}'} | {'–' if k == 'LIVE' else split(list(r.close_reason.astype(str)), live, v)} | {share if k != 'LIVE' else '–'} | "
              f"{'–' if k == 'LIVE' else RM.loo_text(d, lab)} | {hc} |")


def main(argv):
    a = FS.load_orders(None)
    c = a[(a.status.astype(str) == "CLOSED") & (a.es == FS.FLATBASE_SHORT)]
    real, _ox = FS._GK.split_operator_closed(c)
    real = real.sort_values("o").drop_duplicates(["o", "pair"]).reset_index(drop=True)
    W = walks(real, fetch="--fetch" in argv)
    cur = next((a for a in argv[1:] if a.endswith(".csv")), None)
    acc = None
    if cur:
        x = pd.read_csv(cur, low_memory=False)
        x = x[(x.status == "CLOSED") & (pd.to_datetime(x.opened_at, format="mixed") >= B25)].copy()
        x["k"] = x.opened_at.astype(str).str.replace(" ", "T").str[:16] + x.pair
        x["usd_live"] = pd.to_numeric(x.pnl, errors="coerce").fillna(0.0)
        acc = (x, pd.to_datetime(x.opened_at, format="mixed").min(), pd.to_datetime(x.closed_at, format="mixed").max())
    b25 = real.o >= B25
    print("# FLATBASE_SHORT — exit before → after (DECISION_LOG 406 table; frozen scout shadows, same paths, fees + slip, no funding)")
    block("② BEFORE B25 (every pre-B25 FLATBASE fill in the exports = B24's one LIVE fill; the master holds the same 1)", real[~b25].reset_index(drop=True), [w for w, m in zip(W, b25) if not m])
    block("① CURRENT BATCH B25 (paper; DCR on the real account path)", real[b25].reset_index(drop=True), [w for w, m in zip(W, b25) if m], acc)
    print("\n### ③ BACKTEST (the exit study, S4 entry · 72 h, Δ per trade vs today's +8 / −15, no funding)")
    for k in FS.TP_SHADOWS:
        print(f"- {k}: {FS.tp_backtest_note(k)}")


if __name__ == "__main__":
    main(sys.argv)
