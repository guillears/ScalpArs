#!/usr/bin/env python3
"""Do the runaway / lift-off triggers raise the ODDS of a big move at all? (2026-10-01, overnight check on the v2 / v3 grids.)

The grids test specific exits (all fail). This asks the exit-free question: after each trigger, over the next 4 h / 24 h (5m bars,
entry = the next bar's open), how often does the pair run ≥ +5 % / +10 % in the trade direction BEFORE falling −3 % / −5 % against it,
and what is the plain 24 h return — versus an UNCONDITIONAL baseline: the same universes (top-50 / ranks 51-100 by 24 h volume) sampled
every 6 h on every pair (no trigger). Units: UTC days (day means, t-interval). Also the operator's COMBO: VOL_LEAD ∧ EMA200_LIFT on the
same pair within 2 h. Events are read from the newest FILLS csvs (v3, else v2; not in git — re-run the two design scripts first) — nothing is re-detected.
Usage: venv/bin/python scripts/trigger_forward_odds.py  → reports/TRIGGER_FORWARD_ODDS_2026-10-01.md"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_spike_follow_study as V2  # noqa: E402

BAR, H = V2.BAR, 3_600_000
OUT = os.path.join(V2.ROOT, "reports", "TRIGGER_FORWARD_ODDS_2026-10-01.md")


def FILE(name):
    """The newest FILLS csv of a test (v3 if it exists, else v2)."""
    for v in ("v3", "v2"):
        f = f"{name}_SLEEVE_FILLS_{v}_2026-10-01.csv"
        if os.path.exists(os.path.join(V2.ROOT, "reports", f)):
            return f
    raise SystemExit(f"no FILLS csv for {name}")


def fwd(pair, t, side):
    """Forward stats from the open of the bar after t. Returns dict or None."""
    d = V2.D.get(pair)
    if d is None or t not in d.index or t - d.index[0] < 30 * 86_400_000:
        return None
    i = d.index.get_loc(t)
    w = d.iloc[i + 1: i + 1 + 288]
    if len(w) < 280:
        return None
    e = float(w.o.iloc[0]); sg = 1 if side == "LONG" else -1
    fav = (w.h.values / e - 1) * 100 if sg > 0 else (1 - w.l.values / e) * 100
    adv = (w.l.values / e - 1) * 100 if sg > 0 else (1 - w.h.values / e) * 100
    out = {"r24": sg * (float(w.c.iloc[-1]) / e - 1) * 100, "r4": sg * (float(w.c.iloc[47]) / e - 1) * 100,
           "mfe24": float(fav.max()), "mae24": float(adv.min())}
    for tgt, stp in ((5, 3), (10, 5)):                         # target before stop (stop first on a shared bar)
        hit_t = np.argmax(fav >= tgt) if (fav >= tgt).any() else 10**9
        hit_s = np.argmax(adv <= -stp) if (adv <= -stp).any() else 10**9
        out[f"t{tgt}s{stp}"] = float(hit_t < hit_s)
        out[f"s{stp}first"] = float(hit_s <= hit_t and hit_s < 10**9)
    return out


def summarize(rows, label):
    """Day means with a t-interval. The target-vs-stop read is the CONDITIONAL ratio target / (target + stop): raw hit rates rise
    mechanically with volatility (events move ±11–14 % in 24 h, the baseline ±3.7 %), the ratio does not (review 2026-10-01)."""
    g = pd.DataFrame(rows)
    if not len(g):
        return f"| {label} | 0 | – | – | – | – | – | – |"
    g["day"] = pd.to_datetime(g.t, unit="ms").dt.date
    dm = g.groupby("day").mean(numeric_only=True)
    def ci(col):
        x = dm[col]
        if len(x) < 3:
            return f"{x.mean():+.2f}"
        m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); q = 1.96 + 2.4 / max(len(x) - 1, 1)
        return f"{m:+.2f} [{m - q * se:+.2f},{m + q * se:+.2f}]"
    ratio = lambda a, b: f"{g[a].sum() / max(g[a].sum() + g[b].sum(), 1) * 100:.0f}%"
    return (f"| {label} | {len(g)} | {len(dm)} | {ci('r4')} | {ci('r24')} | {ratio('t5s3', 's3first')} | {ratio('t10s5', 's5first')} | "
            f"{g.mfe24.median():+.1f} / {g.mae24.median():+.1f} |")


if __name__ == "__main__":
    L = ["# Do the triggers raise the odds of a big move? (exit-free, 5m bars, day units)", "",
         "Gross of fees. `target share of resolved` = among events that hit EITHER +5 % (target) or −3 % (stop) first, the share that hit the "
         "target first (volatility-neutral; same for +10 / −5; 5m bars, stop-first on a shared bar). Baseline = the same universe sampled every "
         "6 h on every pair with ≥ 30 days of history, no trigger.", "",
         "| Cohort | events | days | 4 h return % [95 %] | 24 h return % [95 %] | +5 vs −3: target share of resolved | +10 vs −5: target share of resolved | median best / worst 24 h |",
         "|---|---|---|---|---|---|---|---|"]
    # baseline
    base = {"TOP50": {"LONG": [], "SHORT": []}, "NEXT50": {"LONG": [], "SHORT": []}}
    ts = [t for t in V2.btc.index[::72] if t >= 1769644800000]          # every 6 h from 2026-01-29
    for t in ts:
        ranked = V2.universe(int(t), 100)
        for uni, ps in (("TOP50", ranked[:50]), ("NEXT50", ranked[50:100])):
            for p in ps:
                for side in ("LONG", "SHORT"):
                    r = fwd(p, int(t), side)
                    if r:
                        base[uni][side].append(dict(t=int(t), **r))
    for uni in base:
        for side in base[uni]:
            L.append(summarize(base[uni][side], f"BASELINE {uni} {side}"))
    ev = {}
    for name, f in (("RUNAWAY", FILE("RUNAWAY")), ("LIFTOFF", FILE("LIFTOFF"))):
        F = pd.read_csv(os.path.join(V2.ROOT, "reports", f), usecols=["t", "pair", "side", "th"]).drop_duplicates(["t", "pair", "side", "th"])
        for (side, th), g in F.groupby(["side", "th"]):
            rows = [dict(t=int(r.t), pair=r.pair, **x) for r in g.itertuples() if (x := fwd(r.pair, int(r.t), side))]
            ev[(name, side, str(th))] = rows
            L.append(summarize(rows, f"{name} {side} {th}"))
    # operator's combo: VOL_LEAD and EMA200_LIFT on the same pair within 2 h (either order), per universe/side
    for uni in ("TOP50", "NEXT50"):
        for side in ("LONG", "SHORT"):
            a = pd.DataFrame(ev.get(("LIFTOFF", side, f"VOL_LEAD·{uni}"), [])); b = pd.DataFrame(ev.get(("LIFTOFF", side, f"EMA200_LIFT·{uni}"), []))
            if not len(a) or not len(b):
                continue
            rows = []                                          # enter when the SECOND signal fires (no look-ahead)
            for p, ga in a.groupby("pair"):
                gb = b[b.pair == p]
                for first, second in ((ga, gb), (gb, ga)):
                    for r in second.itertuples():
                        if ((r.t - first.t >= 0) & (r.t - first.t <= 2 * H)).any():
                            rows.append(r._asdict())
            L.append(summarize(rows, f"COMBO VOL_LEAD ∧ EMA200_LIFT (≤2 h) {uni} {side}"))
    open(OUT, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print(f"\n→ {OUT}")
