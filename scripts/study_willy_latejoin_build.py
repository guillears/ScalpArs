#!/usr/bin/env python3
"""WILLY late joiners (2026-10-22) — COHORT step (no outcome computed here). READ-ONLY on the bot.
Pre-registration: reports/WILLY_LATE_JOIN_2026-10-22_PREREG.md.

Per FRENZY episode of reports/FRENZY_FLATBASE_SHORT_2026-10-22_data/episodes_compact.pkl (engine-parity walk, spike ≥ 2025-01-01):
  L = (first shortlisted-while-flagged bar − spike bar) / 5 min   (engine frenzy_willy_late_bars at the first cold pass)
  why late = which shortlist leg failed on the spike bar (24 h volume < $20M / move < 15 % / both)
  per rule (R0 / R1-30 / R1-60 / R2) the arm bar, the red bar (first close < open from the arm bar, close ≤ the rule's expiry), R at the red bar.
  NULL rows: on-time episodes (L ≤ 3) re-armed at spike + L' for L' = 4…12 (the lag-matched comparator), wait 60 min from arming.
Outputs: reports/WILLY_LATE_JOIN_2026-10-22_data/episodes.csv · entries.csv (one row per (episode, rule)) · need_days.csv (pair, day).
Usage: venv/bin/python scripts/study_willy_latejoin_build.py [--procs 8]
"""
import argparse, os, sys
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_latejoin_common as C  # noqa: E402
import study_flatbase_short_common as FBC  # noqa: E402

T0 = pd.Timestamp("2025-01-01").value // 10**6
VMIN, CHG = 20e6, 15.0
SUP = C.supply()


def red_after(o, c, a, last):
    for k in range(a, min(last, len(c) - 1) + 1):
        if c[k] < o[k]:
            return k
    return None


def one(pair, rows):
    full = FBC.load_full(pair)
    if full is None:
        return [], []
    t, o, h, l, c, v, qv, real = full
    vol24 = pd.Series(qv).rolling(288, min_periods=288).sum().values
    hi24 = pd.Series(h).rolling(288, min_periods=288).max().values; lo24 = pd.Series(l).rolling(288, min_periods=288).min().values
    c288 = np.r_[np.full(288, np.nan), c[:-288]]
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.maximum(np.abs(c / c288 - 1) * 100, (hi24 / lo24 - 1) * 100)
    eps, ent = [], []
    cs = SUP.get(pair)
    for on_t, fs_t in rows:
        on = int((on_t - t[0]) // C.BAR); j = int((fs_t - t[0]) // C.BAR)
        if on < 0 or on >= len(t) or t[on] != on_t:
            continue
        L = j - on
        vlo, mlo = bool(vol24[on] < VMIN), bool(move[on] < CHG)
        why = "vol" if vlo and not mlo else ("move" if mlo and not vlo else ("both" if vlo else "none"))
        pump = (h[on:on + 13].max() / c[max(0, on - 6)] - 1) * 100
        eps.append(dict(pair=pair, on_t=int(on_t), spike_close=int(t[on] + C.BAR), first_seen_t=int(fs_t), L=L, why_late=why,
                        vol24_spike=float(vol24[on]), vol24_join=float(vol24[j]) if j < len(t) else np.nan, move_spike=float(move[on]),
                        pump_1h=float(pump), supply=cs if cs else np.nan))
        specs = []
        if L <= C.COLD:
            specs.append(("R0", j, j + C.WAIT_BARS))
        elif L <= 12:
            if L <= 6:
                specs.append(("R1-30", j, j + C.WAIT_BARS))
            specs.append(("R1-60", j, j + C.WAIT_BARS))
            specs.append(("R2", j, on + C.WAIT_BARS))
        if L <= C.COLD:
            for Lp in range(4, 13):
                specs.append((f"NULL{Lp}", on + Lp, on + Lp + C.WAIT_BARS))
        for rule, a, e in specs:
            k = red_after(o, c, a, e) if a <= e else None
            d = dict(pair=pair, on_t=int(on_t), spike_close=int(t[on] + C.BAR), L=L, rule=rule, arm_close=int(t[min(a, len(t) - 1)] + C.BAR))
            if k is None:
                d.update(st="red_expired"); ent.append(d); continue
            rc = int(t[k] + C.BAR)
            Rv = (vol24[k] / (cs * c[k])) if (cs and np.isfinite(vol24[k])) else np.nan
            d.update(st="ok", red_close=rc, wait_bars=k - a, red_ret=(c[k] / o[k] - 1) * 100, vol24_red=float(vol24[k]),
                     mcap_red=(cs * c[k]) if cs else np.nan, R=Rv,
                     route=("UNREAD" if not np.isfinite(Rv) else ("FULL" if Rv < C.RCUT else "HT")),
                     min_spike_to_entry=(rc - (t[on] + C.BAR)) / C.MIN, run_from_base=(c[k] / c[max(0, on - 6)] - 1) * 100)
            ent.append(d)
    return eps, ent


def _w(args):
    try:
        return one(*args)
    except Exception as ex:  # noqa
        print("ERR", args[0], ex, flush=True)
        return [], []


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    E = pd.read_pickle(os.path.join(C.FB, "episodes_compact.pkl"))
    E = E[E.on_t >= T0]
    groups = [(p, list(g[["on_t", "first_seen_t"]].itertuples(index=False, name=None))) for p, g in E.groupby("pair")]
    eps, ent = [], []
    with mp.Pool(A.procs) as P:
        for k, (a, b) in enumerate(P.imap_unordered(_w, groups, chunksize=4)):
            eps += a; ent += b
            if k % 200 == 0:
                print(k, len(groups), flush=True)
    os.makedirs(C.D, exist_ok=True)
    ep = pd.DataFrame(eps).sort_values("on_t"); en = pd.DataFrame(ent).sort_values(["on_t", "rule"])
    ep.to_csv(os.path.join(C.D, "episodes.csv"), index=False); en.to_csv(os.path.join(C.D, "entries.csv"), index=False)
    ok = en[en.st == "ok"]
    need = set()
    for p, rc in zip(ok.pair, ok.red_close):
        for x in (rc, rc + (C.CAP + 2) * C.MIN):
            need.add((p, pd.Timestamp(int(x), unit="ms").strftime("%Y-%m-%d")))
    pd.DataFrame(sorted(need), columns=["pair", "day"]).to_csv(os.path.join(C.D, "need_days.csv"), index=False)
    print("episodes", len(ep), "entries", len(en), "ok", len(ok), "pair-days", len(need))
    print(ep.L.describe())
