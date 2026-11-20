#!/usr/bin/env python3
"""📡 Route B — FIRST VIEW of the recorded data (DECISION_LOG 597; EXPLORATORY — 2 hours, one night, never a ship input).

Reads per-minute feature CSVs (scripts/route_b/features.py) + the bot's list, writes a markdown report + one chart:
  ① inventory · ② liquidity / cost at OUR size per coin (bot coins vs the rest) · ③ does the book or the tape predict the next minutes?
     (book imbalance within 0.5 % and taker buy share vs the next 1 / 5 / 15-min mid return — Spearman PER COIN, then the mean across
     coins with a coin-level t-stat; each hour apart = the consistency check; vs a shuffled control) · ④ liquidations.
Usage: python3 first_view.py <out.md> <bot.json> <feat-hour.csv> [<feat-hour.csv> …]"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 15)
MIN_ROWS = 40          # coin-minutes with book data needed for a coin's correlation


def load(paths):
    fr = []
    for p in paths:
        d = pd.read_csv(p)
        d["hour"] = os.path.basename(p)
        fr.append(d)
    a = pd.concat(fr, ignore_index=True)
    a = a[a.sym.astype(str).str.endswith("USDT")]          # USDT-M only (the liquidation stream also carries coin-margined contracts)
    return a.sort_values(["sym", "minute"]).reset_index(drop=True)


def bot_sets(doc):
    op = set((doc.get("open") or {}).keys()) if isinstance(doc.get("open"), dict) else set(doc.get("open") or [])
    return op, set((doc.get("flagged") or {}).keys()), set(doc.get("tracked") or []), set(doc.get("followed") or [])


def add_targets(b):
    """next-h-minute mid return (bps) per coin on CONSECUTIVE minutes only (a missing minute → NaN)."""
    b = b.sort_values(["sym", "minute"]).copy()
    g = b.groupby("sym")
    for h in HORIZONS:
        fut_mid = g["mid"].shift(-h)
        fut_min = g["minute"].shift(-h)
        ok = (fut_min - b["minute"]) == h * 60_000
        b[f"raw_{h}"] = np.where(ok, (fut_mid / b["mid"] - 1.0) * 1e4, np.nan)
        b[f"ret_{h}"] = b[f"raw_{h}"] - b.groupby("minute")[f"raw_{h}"].transform("mean")   # vs the market that same minute
    tot = b["buy_usd"] + b["sell_usd"]
    b["buy_share"] = np.where(tot > 0, b["buy_usd"] / tot, np.nan)
    return b


def coin_corr(b, feat, tgt, shuffle_seed=None):
    """Spearman(feat, tgt) per coin (≥ MIN_ROWS rows) → Series. shuffle_seed → the target shuffled WITHIN each coin (the control)."""
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    out = {}
    for s, g in b[[feat, tgt, "sym"]].dropna().groupby("sym"):
        if len(g) < MIN_ROWS:
            continue
        y = g[tgt].values
        if rng is not None:
            y = rng.permutation(y)
        r = pd.Series(g[feat].values).rank().corr(pd.Series(y).rank())
        if r == r:
            out[s] = r
    return pd.Series(out, dtype=float)


def tstat(x):
    x = x.dropna()
    return (x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if len(x) > 2 and x.std(ddof=1) > 0 else float("nan")


def fmt_usd(v):
    return "–" if v != v else (f"${v / 1e6:.2f}M" if v >= 1e6 else f"${v / 1e3:.0f}k")


def main(argv):
    out, botp, feats = argv[0], argv[1], argv[2:]
    doc = json.load(open(botp))
    op, fl, tr, fo = bot_sets(doc)
    a = load(feats)
    b = a[a.n_book > 0].copy()
    b = add_targets(b)
    syms = sorted(b.sym.unique())
    bot_all = op | fl | tr
    hours = sorted(a.hour.unique())
    L = ["# 📡 Route B — first view of the recorded data (2026-11-20, 00:00–02:00 UTC)", "",
         "**EXPLORATORY.** Two hours of one night, one market regime. Nothing here is a filter, a sleeve or a ship input — it shows what the "
         "data can answer and the first shapes worth testing properly once weeks are recorded. Raw data: s3 (private bucket); features: "
         "`scripts/route_b/features.py` (one row per coin per minute); this report: `scripts/route_b/first_view.py`.", "",
         "## ① What was recorded", "",
         f"- **Full detail (order book every 0.5 s + every trade):** {len(syms)} coins · {len(b):,} coin-minutes with book data · "
         f"{int(b.n_book.sum()):,} book snapshots · {int(a.n_trades.sum()):,} trades.",
         f"- **Whole market (mark price, funding, liquidations):** {a.sym.nunique()} coins.",
         f"- **The bot's own coins at report time:** {len(op)} open · {len(fl)} Frenzy-flagged · {len(tr)} tracked by EXHAUSTION / FLATBASE · "
         f"{len(fo)} followed by the Frenzy pass. Covered in full detail: "
         f"{len([s for s in bot_all if s in set(syms)])} of {len(bot_all)} open / flagged / tracked.", ""]

    # ② liquidity at our size
    per = b.groupby("sym").agg(spread=("spread_bps", "median"), bid05=("bid_usd_05", "median"), ask05=("ask_usd_05", "median"),
                               cb25=("cost_buy_2500", "median"), cs25=("cost_sell_2500", "median"),
                               cb10=("cost_buy_10000", "median"), cs10=("cost_sell_10000", "median"),
                               vol=("buy_usd", "sum"), vols=("sell_usd", "sum"))
    per["vol_usd_h"] = (per.vol + per.vols) / len(hours)
    per["bot"] = [("open" if s in op else "flagged" if s in fl else "tracked" if s in tr else "followed" if s in fo else "") for s in per.index]
    per["worst10"] = per[["cb10", "cs10"]].max(axis=1)
    L += ["## ② What it costs to trade at our size (the real order book)", "",
          "Cost = how far from the middle price a MARKET order of that size fills, walking the real book (one side; a round trip pays "
          "both sides + fees ≈ 0.10 %). Median over the two hours per coin.", "",
          "| Group | coins | spread (bps) | $ within 0.5 % (bid / ask) | cost $2.5k buy / sell (bps) | cost $10k buy / sell (bps) |",
          "|---|---|---|---|---|---|"]
    for lab, m in (("Bot coins (open / flagged / tracked)", per.bot.isin(["open", "flagged", "tracked"])),
                   ("Followed by the Frenzy pass", per.bot == "followed"), ("Other hot coins (movers / volume)", per.bot == "")):
        g = per[m]
        if not len(g):
            continue
        L.append(f"| {lab} | {len(g)} | {g.spread.median():.1f} | {fmt_usd(g.bid05.median())} / {fmt_usd(g.ask05.median())} | "
                 f"{g.cb25.median():.1f} / {g.cs25.median():.1f} | {g.cb10.median():.1f} / {g.cs10.median():.1f} |")
    thin = per[per.bot.isin(["open", "flagged", "tracked"])].sort_values("worst10", ascending=False).head(8)
    L += ["", "**Thinnest bot coins** (where our size moves the price most):", "",
          "| Coin | role | spread (bps) | $ within 0.5 % bid / ask | $10k buy / sell cost (bps) | traded $/hour |", "|---|---|---|---|---|---|"]
    for s, r in thin.iterrows():
        L.append(f"| {s} | {r.bot} | {r.spread:.1f} | {fmt_usd(r.bid05)} / {fmt_usd(r.ask05)} | {r.cb10:.1f} / {r.cs10:.1f} | {fmt_usd(r.vol_usd_h)} |")
    L += ["", "*Why it matters:* this is the slippage our model ESTIMATES today (EXH / DIST / Flatbase sizing, the depth guard); from now on "
          "it can be checked against the real book at every entry and every stop.", ""]

    # ③ predictive — per coin, then across coins
    L += ["## ③ Does the book or the trade flow predict the next minutes? (first look)", "",
          "For every coin separately: the rank correlation between a reading at the end of a minute and the price move over the next "
          "1 / 5 / 15 minutes, measured AGAINST THE MARKET that same minute (the night drifted down; a market-wide drift is not a "
         "signal). Then the average across coins (each coin = one vote) and how many coins agree. The control re-runs it with "
          "the future moves shuffled inside each coin (what pure luck looks like).", "",
          "| Reading | horizon | coins | mean correlation | coins positive | t (across coins) | luck control mean | hour 1 / hour 2 mean |",
          "|---|---|---|---|---|---|---|---|"]
    res = {}
    for feat, lab in (("imb_05", "Book imbalance within 0.5 % (more bids = +)"), ("buy_share", "Taker BUY share of the minute's trades")):
        for h in HORIZONS:
            c = coin_corr(b, feat, f"ret_{h}")
            ctl = coin_corr(b, feat, f"ret_{h}", shuffle_seed=597)
            per_h = [coin_corr(b[b.hour == hh], feat, f"ret_{h}").mean() for hh in hours]
            res[(feat, h)] = c
            L.append(f"| {lab} | {h} min | {len(c)} | {c.mean():+.3f} | {(c > 0).mean():.0%} | {tstat(c):+.1f} | {ctl.mean():+.3f} | "
                     + " / ".join(f"{x:+.3f}" for x in per_h) + " |")
    # size of the move: fifths built per coin, pooled; the move is vs the market that minute
    def fifths(feat):
        q_ = b.groupby("sym")[feat].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) if x.notna().sum() >= 25 else np.nan)
        return b.assign(q=q_).dropna(subset=["q", "ret_5", "ret_15"]).groupby("q").agg(n=("ret_5", "size"), r5=("ret_5", "mean"), r15=("ret_15", "mean"))
    qi, qb = fifths("imb_05"), fifths("buy_share")
    L += ["", "**How big is it?** Next 5 / 15-min move vs the market, by fifth of the reading (fifths built per coin, then pooled):", "",
          "| Fifth | Book imbalance: coin-minutes · 5 min · 15 min (bps) | Taker buy share: coin-minutes · 5 min · 15 min (bps) |", "|---|---|---|"]
    for k in range(5):
        ri = qi.loc[k] if k in qi.index else None
        rb = qb.loc[k] if k in qb.index else None
        lab = {0: "1 (most asks / most selling)", 4: "5 (most bids / most buying)"}.get(k, str(k + 1))
        L.append(f"| {lab} | " + (f"{int(ri.n):,} · {ri.r5:+.1f} · {ri.r15:+.1f}" if ri is not None else "–") + " | "
                 + (f"{int(rb.n):,} · {rb.r5:+.1f} · {rb.r15:+.1f}" if rb is not None else "–") + " |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(1, 6)
        for axi, qq, ttl in ((ax[0], qi, "Book imbalance (0.5 %)"), (ax[1], qb, "Taker BUY share of the minute")):
            axi.bar(x - 0.18, qq.r5.values, 0.36, label="next 5 min", color="#5b8def")
            axi.bar(x + 0.18, qq.r15.values, 0.36, label="next 15 min", color="#f0a33a")
            axi.axhline(0, color="#888", lw=0.8)
            axi.set_xticks(x)
            axi.set_xlabel("fifth (1 = lowest … 5 = highest)")
            axi.set_title(ttl)
            axi.set_ylabel("move vs market (bps)")
            axi.legend(frameon=False, fontsize=8)
        fig.suptitle("Route B first view — 2 hours, " + f"{b.sym.nunique()} coins (exploratory)", fontsize=10)
        fig.tight_layout()
        png = os.path.splitext(out)[0] + ".png"
        fig.savefig(png, dpi=110)
        L += ["", f"![first view]({os.path.basename(png)})"]
    except Exception as e:
        L += ["", f"(chart skipped: {e})"]
    L += ["", "*Reading it:* a correlation of a few hundredths with most coins agreeing and both hours the same sign is a real but SMALL "
          "effect; compare the bps move with the ≈ 10 bps a market round trip costs before calling anything tradable. Two hours can't "
          "separate a lasting effect from tonight's tape — this is the list of things to test properly, not a finding.", ""]

    # ④ liquidations
    lq = a.groupby("sym")[["liq_long_usd", "liq_short_usd"]].sum()
    tot_l, tot_s = lq.liq_long_usd.sum(), lq.liq_short_usd.sum()
    top = lq.assign(t=lq.sum(axis=1)).sort_values("t", ascending=False).head(6)
    L += ["## ④ Liquidations (a sample — Binance sends at most one per coin per second)", "",
          f"- Longs force-sold: {fmt_usd(tot_l)} · shorts force-bought: {fmt_usd(tot_s)} over the two hours.",
          "- Most liquidated: " + " · ".join(f"{s} {fmt_usd(r.t)} ({'longs' if r.liq_long_usd >= r.liq_short_usd else 'shorts'})" for s, r in top.iterrows()), "",
          "## What this first view says", "",
          "- **Recording works end to end** and covers every coin the bot holds, flags or tracks.",
          "- **Cost at our size is now measurable from the real book** (section ②) — the first practical use (slippage model, depth guard, "
          "thin-coin sizing).",
          "- **Book / flow → next minutes (section ③):** read the table as *direction of travel only*. The proper test needs weeks: "
          "per-coin + per-day consistency, the shuffled null, held-out days, and the cost of trading it.",
          "- **Per-sleeve reads** (book at entry, winners vs losers) start as fills accumulate — none can be read from two hours.", "",
          "## Data and caveats", "",
          "- Book = top 20 levels each side; for liquid coins 20 levels may not reach 1 % (the band is then a lower bound).",
          "- Mid-price moves vs the cross-coin average of the same minute, not tradable fills; no fees in section ③.",
          "- The coin-level t ignores that coins move together in the same minutes — it overstates confidence; weeks + day-blocks fix that.",
          "- One quiet night (00:00–02:00 UTC); the first 30 min of recording (23:36–24:00) are left out.", ""]
    open(out, "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1:])
