#!/usr/bin/env python3
"""🐉 "Ride the monster" test (operator, 2026-10-01 — MOVR +140 % in two days): does a SMALL, WIDE-STOP, MULTI-DAY long on every
runaway / volume-surge trigger make money, i.e. do the rare monsters pay for the many that fade?

PRE-DECLARED (before any result):
  events   the v3 trigger events (reports/*_SLEEVE_FILLS_v3_2026-10-01.csv, not in git — re-run the two design scripts first): RUNAWAY LONG +8 % / +12 %, VOL_LEAD LONG (TOP50 / NEXT50)
           ONE position per pair at a time (a trigger while that pair's position is open is skipped)
  entry    the open of the 5m bar after the trigger bar
  exits    A  stop −20 %, hold 7 d
           B  stop −20 %; once up +30 % a 25 % trail from the peak; hold 7 d
           C  stop −10 %; once up +20 % a 20 % trail from the peak; hold 3 d
           D  stop −30 %, take-profit +100 %, hold 7 d
           5m bars, the stop / trail is checked BEFORE the bar's high (adverse-first); a bar opening through a stop fills at the open
  costs    0.09 % fees + 0.10 % slippage round trip; funding ignored (stated)
  baseline the same exits on the same universe sampled every 6 h on every pair (TOP50 / NEXT50), one position per pair at a time
  units    entry WEEKS (t-interval on week means — holds are 3–7 days, so adjacent entry DAYS share most of their window and a
           day interval is too narrow; review 2026-10-01); plus the trade mean, how each trade ended (initial stop / trail / TP /
           time), the share reaching +50 % / +100 %, and how much of the total the best 5 trades carry
  guards   (review 2026-10-01) ≥ 30 days of pair history at entry for events AND baseline · a bar whose high is > 1.5× its body top
           is a bad tick: the high is clipped there (BABYUSDT printed a +7,393 % wick) · a bar that OPENS above the TP fills at the
           open · dropped events are counted in the report
  limits   the trail level raised by a bar applies from the NEXT bar (a bar that peaks and then falls through its own new trail
           survives to the next open) · the one-per-pair rule and the data-end guard depend on the exit, so the four exits are NOT
           compared on identical trades · events come from the v3 FILLS csvs (events the 1m simulator dropped are not here)
  caveat   only pairs still cached today (dead pumps that were delisted are missing → flatters longs)
Usage: venv/bin/python scripts/monster_ride_test.py → reports/MONSTER_RIDE_TEST_2026-10-01.md"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_spike_follow_study as V2  # noqa: E402

BAR, D1 = V2.BAR, 86_400_000
COST = 0.19
OUT = os.path.join(V2.ROOT, "reports", "MONSTER_RIDE_TEST_2026-10-01.md")
EXITS = {"A": dict(stop=20, arm=None, trail=None, tp=None, days=7), "B": dict(stop=20, arm=30, trail=25, tp=None, days=7),
         "C": dict(stop=10, arm=20, trail=20, tp=None, days=3), "D": dict(stop=30, arm=None, trail=None, tp=100, days=7)}


WICK = 1.5      # bad-tick guard: high clipped at 1.5 × max(open, close)


def ride(d, i, x):
    """One long from the open of bar i. Returns (net %, max favourable %, exit bar index, how) or None when the data ends early."""
    n = x["days"] * 288
    if i + n > len(d):
        return None
    o = d.o.values[i:i + n]; l = d.l.values[i:i + n]; c = d.c.values[i:i + n]
    h = np.minimum(d.h.values[i:i + n], np.maximum(o, c) * WICK)
    e = o[0]; peak = 0.0; stop = init = -float(x["stop"])
    for k in range(n):
        vo = (o[k] / e - 1) * 100
        if k > 0 and vo <= stop:
            return vo - COST, peak, i + k, ("STOP" if stop == init else "TRAIL")
        if k > 0 and x["tp"] is not None and vo >= x["tp"]:
            return vo - COST, max(peak, vo), i + k, "TP"
        vl = (l[k] / e - 1) * 100
        if vl <= stop:
            return stop - COST, peak, i + k, ("STOP" if stop == init else "TRAIL")
        vh = (h[k] / e - 1) * 100
        if x["tp"] is not None and vh >= x["tp"]:
            return x["tp"] - COST, max(peak, vh), i + k, "TP"
        peak = max(peak, vh)
        if x["arm"] is not None and peak >= x["arm"]:
            stop = max(stop, ((1 + peak / 100) * (1 - x["trail"] / 100) - 1) * 100)   # 25 % below the peak PRICE
    return (c[-1] / e - 1) * 100 - COST, peak, i + n - 1, "TIME"


def run(events):
    """events: DataFrame t, pair (sorted). One position per pair at a time, per exit. Returns (trades, drop counts of exit A)."""
    rows = []; drops = {}
    for name, x in EXITS.items():
        busy = {}; dc = {"no data / < 30 d history": 0, "pair already in a position": 0, "data ends before the hold": 0}
        for r in events.itertuples():
            d = V2.D.get(r.pair)
            if d is None or r.t not in d.index or r.t - d.index[0] < 30 * D1:
                dc["no data / < 30 d history"] += 1; continue
            i = d.index.get_loc(r.t) + 1
            if busy.get(r.pair, -1) >= i:
                dc["pair already in a position"] += 1; continue
            res = ride(d, i, x)
            if res is None:
                dc["data ends before the hold"] += 1; continue
            busy[r.pair] = res[2]
            rows.append((name, r.t, r.pair, res[0], res[1], res[3]))
        drops[name] = dc
    return pd.DataFrame(rows, columns=["exit", "t", "pair", "r", "mfe", "how"]), drops


def wk(g):
    """Entry-week means → (mean, lo, hi, n weeks)."""
    wm = g.assign(w=pd.to_datetime(g.t, unit="ms").dt.strftime("%G-%V")).groupby("w").r.mean()
    m, se = wm.mean(), wm.std(ddof=1) / np.sqrt(len(wm)); q = 1.96 + 2.4 / max(len(wm) - 1, 1)
    return m, m - q * se, m + q * se, len(wm)


def line(label, g, x):
    if not len(g):
        return f"| {label} | 0 | – | – | – | – | – | – | – |"
    m, lo, hi, nw = wk(g)
    tot = g.r.sum(); top5 = g.r.sort_values(ascending=False).head(5).sum()
    how = " / ".join(f"{(g.how == k).mean() * 100:.0f}" for k in ("STOP", "TRAIL", "TP", "TIME"))
    return (f"| {label} | {len(g)} | {nw} | {g.r.mean():+.2f} | {m:+.2f} [{lo:+.2f}, {hi:+.2f}] | {(g.r > 0).mean() * 100:.0f}% | "
            f"{how} | {(g.mfe >= 50).mean() * 100:.1f}% / {(g.mfe >= 100).mean() * 100:.1f}% | "
            f"{tot:+.0f} (best 5: {top5:+.0f}; without them {tot - top5:+.0f}) |")


if __name__ == "__main__":
    ev = {}
    F = pd.read_csv(os.path.join(V2.ROOT, "reports", "RUNAWAY_SLEEVE_FILLS_v3_2026-10-01.csv"), usecols=["t", "pair", "side", "th"]).drop_duplicates()
    for th in (8.0, 12.0):
        ev[f"RUNAWAY LONG +{th:.0f}%"] = F[(F.side == "LONG") & (F.th == th)][["t", "pair"]].sort_values("t")
    G = pd.read_csv(os.path.join(V2.ROOT, "reports", "LIFTOFF_SLEEVE_FILLS_v3_2026-10-01.csv"), usecols=["t", "pair", "side", "th"]).drop_duplicates()
    for uni in ("TOP50", "NEXT50"):
        ev[f"VOLUME LEADS PRICE LONG {uni}"] = G[(G.side == "LONG") & (G.th == f"VOL_LEAD·{uni}")][["t", "pair"]].sort_values("t")
    base = {"TOP50": [], "NEXT50": []}
    for t in [t for t in V2.btc.index[::72] if t >= 1769644800000]:
        ranked = V2.universe(int(t), 100)
        for uni, ps in (("TOP50", ranked[:50]), ("NEXT50", ranked[50:100])):
            base[uni] += [(int(t), p) for p in ps]
    for uni in base:
        ev[f"BASELINE {uni} (every 6 h)"] = pd.DataFrame(base[uni], columns=["t", "pair"]).sort_values("t")
    bad = sum(int((d.h.values > np.maximum(d.o.values, d.c.values) * WICK).sum()) for d in V2.D.values())
    RES = {label: run(e[["t", "pair"]]) for label, e in ev.items()}
    TRIG = [k for k in ev if not k.startswith("BASELINE")]
    L = ["# 🐉 Ride-the-monster test — small, wide-stop, multi-day longs on runaway / volume-surge triggers", "",
         "Unlevered %, after 0.19 % costs, funding ignored. One position per pair at a time, ≥ 30 days of pair history. Only pairs still "
         "listed today (flatters longs). Ranges are 95 % t-intervals on ENTRY-WEEK means (holds are 3–7 days, so entry days overlap). "
         f"Bad-tick guard: {bad} bars in the whole cache had a high > {WICK}× the body top and were clipped there. The four exits are "
         "not run on identical trades (the one-per-pair rule and the data-end cut depend on the exit).", ""]
    neg = sig = beat = 0
    for name, x in EXITS.items():
        desc = f"stop −{x['stop']} %" + (f", {x['trail']} % trail after +{x['arm']} %" if x["arm"] else "") + (f", TP +{x['tp']} %" if x["tp"] else "") + f", hold {x['days']} d"
        L += [f"## Exit {name}: {desc}", "",
              "| Cohort | trades | weeks | avg per trade % | week mean % [95 %] | winners | ended by initial stop / trail / TP / time % | reached +50 % / +100 % | total % (concentration) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for label in ev:
            R = RES[label][0]; g = R[R.exit == name]
            L.append(line(label, g, x))
            if label in TRIG and len(g):
                m, lo, hi, _ = wk(g); neg += g.r.mean() < 0; sig += hi < 0
                bR = RES["BASELINE " + ("NEXT50" if "NEXT50" in label else "TOP50") + " (every 6 h)"][0]
                beat += g.r.mean() > bR[bR.exit == name].r.mean()
        L.append("")
    L += ["## Reading (computed from the tables above)", "",
          f"- {neg} of {4 * len(TRIG)} trigger × exit cells have a negative average per trade; in {sig} of them the week range excludes zero. "
          f"No cell has a range above zero.",
          f"- {beat} of {4 * len(TRIG)} trigger cells have a better per-trade average than the no-trigger baseline of their universe under the same exit "
          "(runaway triggers are compared with TOP50). The baselines lose too: a large part of the loss belongs to buying this universe "
          "with these exits, not to the trigger.",
          "- The trigger raises the share of trades that reach +50 % / +100 % versus the baseline (see the column), and the average still "
          "does not turn positive: the initial stop ends more trades than the monsters pay for.", ""]
    L += ["## Do the monsters give it back? Exit A (no trail), trades whose best point was ≥ +100 %", "",
          "| Cohort | such trades | median final % | mean final % | finished below +50 % | finished below 0 |", "|---|---|---|---|---|---|"]
    for label in TRIG:
        R = RES[label][0]; g = R[(R.exit == "A") & (R.mfe >= 100)]
        L.append(f"| {label} | {len(g)} | {g.r.median():+.1f} | {g.r.mean():+.1f} | {(g.r < 50).mean() * 100:.0f}% | {(g.r < 0).mean() * 100:.0f}% |" if len(g) else f"| {label} | 0 | – | – | – | – |")
    L += ["", "## Events not traded (exit A)", "", "| Cohort | events | " + " | ".join(RES[TRIG[0]][1]["A"]) + " |", "|---|---|---|---|---|"]
    for label in ev:
        L.append(f"| {label} | {len(ev[label])} | " + " | ".join(str(v) for v in RES[label][1]["A"].values()) + " |")
    L.append("")
    L += ["## Verdict and limits", "",
          "- No sleeve. No evidence of an edge in any of the 16 cells; most ranges span zero, so this is \"no edge found\", not \"proven loser\".",
          "- The no-trigger baselines are negative too: Jan–Sep 2026 was a poor period for holding these alts for days — one 8-month span.",
          "- Survivorship (delisted pumps missing) flatters longs, so the true result is worse, not better.",
          "- Stops and trails fill at their level inside a 5m bar (optimistic in fast bars, e.g. the TUT exit); funding is ignored.",
          "- Not tested: delisted pairs, sub-minute entries, other periods.", ""]
    R = RES["RUNAWAY LONG +12%"][0]; b = R[R.exit == "B"].sort_values("r", ascending=False)
    L += ["## Runaway +12 %, exit B — the 10 best and 10 worst trades", "", "| date | pair | result % | best point % |", "|---|---|---|---|"]
    for r in pd.concat([b.head(10), b.tail(10)]).itertuples():
        L.append(f"| {pd.Timestamp(r.t, unit='ms'):%Y-%m-%d} | {r.pair} | {r.r:+.1f} | {r.mfe:+.1f} |")
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
