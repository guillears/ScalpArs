#!/usr/bin/env python3
"""FRENZY flat-base short (2026-10-22) — EPISODES + PRE-SPIKE FEATURES (no outcome is computed here).

Per pair: the live-engine FRENZY episodes from study_frenzy_swing_short_walk.walk (unchanged; source list + the new data folder), then the
pre-registered flatness features measured on the window that ENDS at the base bar (the close 30 min before the spike bar):
  FLAT14 / FLAT30  max/min − 1 of the closes over 14 / 30 d (NaN when < 90 % real bars)
  RV14             std of 1 h log returns over 14 d (%)
  DEADVOL          mean daily quote volume over the 7 d before the base / median daily quote volume over the 60 d before those 7 d
  DIST90           base / 90-d high − 1 (%)
Outputs (data folder): episodes_compact.pkl (one row per verified + shortlisted episode) · episodes_sample.pkl (walk-format episodes for the
parity check: 150 random + every episode with a spike ≥ 2026-10-02).
Usage: venv/bin/python scripts/study_flatbase_short_walk.py [--procs 8]
"""
import argparse, glob, os, pickle, sys
import multiprocessing as mp
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_flatbase_short_common as C  # noqa: E402
W = C.W


def feats(full, on_t):
    t, o, h, l, c, v, qv, real = full
    on = int((on_t - t[0]) // C.BAR)
    b = on - 6
    out = dict(FLAT14=np.nan, FLAT30=np.nan, RV14=np.nan, DEADVOL=np.nan, DIST90=np.nan, hist_days=(b + 1) * C.BAR / C.DAY)
    if b < 0 or b >= len(t):
        return out
    for nm, nd in (("FLAT14", 14), ("FLAT30", 30)):
        n = nd * 288
        if b + 1 >= n:
            sl = slice(b + 1 - n, b + 1)
            if real[sl].mean() >= 0.9:
                out[nm] = (c[sl].max() / c[sl].min() - 1) * 100
    n = 14 * 288
    if b + 1 >= n and real[b + 1 - n: b + 1].mean() >= 0.9:
        hc = c[b + 1 - n: b + 1][11::12]
        out["RV14"] = float(np.std(np.diff(np.log(hc)), ddof=1) * 100)
    n = 67 * 288
    if b + 1 >= n and real[b + 1 - n: b + 1].mean() >= 0.9:
        dq = qv[b + 1 - n: b + 1].reshape(67, 288).sum(1)
        med = np.median(dq[:60])
        out["DEADVOL"] = float(dq[60:].mean() / med) if med > 0 else np.nan
    n = 90 * 288
    if b + 1 >= n and real[b + 1 - n: b + 1].mean() >= 0.9:
        out["DIST90"] = (c[b] / h[b + 1 - n: b + 1].max() - 1) * 100
    return out


def one(args):
    pair, skip, sample_all_after = args
    try:
        eps = W.walk(pair, skip)
    except Exception as e:
        print("ERR walk", pair, e, flush=True)
        return [], []
    if not eps:
        return [], []
    full = C.load_full(pair)
    rows, keep = [], []
    for e in eps:
        on_t = int(e["t"][e["on_i"]]); off_t = int(e["t"][e["off_i"]]) if e["off_i"] < len(e["t"]) else int(e["t"][-1]) + C.BAR
        seen = np.nonzero(e["seen"])[0]
        first_seen_t = int(e["t"][seen[0]]) if len(seen) else None
        r = dict(pair=pair, on_t=on_t, off_t=off_t, spike_ts=int(e["spike_ts"]), base=float(e["base"]), alive=bool(e["alive"]),
                 new_ep_at_end=bool(e["new_ep_at_end"]), first_seen_t=first_seen_t,
                 end_t=(int(e["t"][e["end_i"]]) if e["end_i"] is not None and e["end_i"] < len(e["t"]) else None))
        r.update(feats(full, on_t) if full is not None else {})
        rows.append(r)
        if on_t >= sample_all_after:
            keep.append(e)
    return rows, keep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
    skip = W.blacklist()
    pairs = sorted({os.path.basename(f)[:-4] for d in W.SRC for f in glob.glob(os.path.join(d, "*USDT.csv"))})
    import json
    S = json.load(open(os.path.join(C.D, "symbols.json")))
    tradfi = set(S.get("tradfi", []))
    pairs = [p for p in pairs if p not in ("BTCUSDT", "ETHUSDT") and p not in tradfi]
    cut = pd.Timestamp("2026-10-02").value // 10**6
    rows, keep = [], []
    with mp.Pool(A.procs) as pool:
        for k, (r, kk) in enumerate(pool.imap_unordered(one, [(p, skip, cut) for p in pairs], chunksize=4)):
            rows += r; keep += kk
            if k % 100 == 0:
                print(k, len(pairs), len(rows), flush=True)
    E = pd.DataFrame(rows).sort_values(["spike_ts", "pair"]).reset_index(drop=True)
    E["spike"] = pd.to_datetime(E.spike_ts, unit="ms")
    E.to_pickle(os.path.join(C.D, "episodes_compact.pkl"))
    # parity sample: every episode with a spike ≥ 10-02 + 150 random earlier ones (walk format), re-walked for the random ones
    rng = np.random.default_rng(22)
    early = E[E.spike < pd.Timestamp("2026-10-02")]
    pick = early.iloc[rng.choice(len(early), 150, replace=False)]
    for pair, g in pick.groupby("pair"):
        for e in W.walk(pair, skip):
            if int(e["t"][e["on_i"]]) in set(g.on_t):
                keep.append(e)
    pickle.dump(keep, open(os.path.join(C.D, "episodes_sample.pkl"), "wb"))
    print("episodes", len(E), "pairs", E.pair.nunique(), "sample", len(keep))
