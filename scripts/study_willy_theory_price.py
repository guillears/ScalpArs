#!/usr/bin/env python3
"""WILLY THEORY — step 3: pricing (pre-registration frozen first: reports/WILLY_THEORY_PREREG.txt).

Every window bar of every armed episode (alt_entries.pkl: WILLY entry, trigger bar, greens, later reds, refused R ≥ 1 reds …) is priced
as an entry at its close + 12 s under TODAY's WILLY exit, on two rulers:
  RH  scout_frenzy_willy_exit.load_path (ticks first, 1m tail) — verdict on fully tick-covered rows only
  R1  the same with ticks suppressed (1m only) — refute-only
Exit = scout_willy_timecap.evaluate_late(P, 1.25, 4, 90, 30, 0.25) booked as the LOWVWAP study (W.book). For bars of episodes WITH a WILLY
entry the W8 exit grid (FX.evaluate plain, TP × SL × cap) and path stats are added (RH path, 240-min horizon).
Output reports/willy_theory/priced.pkl.
"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402
import study_tick_maker_lock_common as C  # noqa: E402  (2025 loader patches, network blocked)

FX, WT = C.FX, C.WT
WT._NET_BLOCKED = True
MIN = W.MIN
GRID_TP = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
GRID_SL = (2.0, 3.0, 4.0, 6.0, 0.0)          # 0 = no stop
GRID_CAP = (30, 60, 90, 240)
HOR = 240

if not os.path.exists(W.PREREG):
    sys.exit("REFUSED: pre-registration missing")


class NoTicks(dict):
    """a tcache that answers 'no tick file' for every day → load_path walks 1m only (R1 ruler)."""
    def __contains__(self, k):
        return True

    def __getitem__(self, k):
        return None


def entry_price(pair, te, tc):
    """→ (E_raw, src): the last print ≤ te within 120 s when ticks exist, else the 1m open of te's minute."""
    z = C.raw_ticks(pair, te - 120_000, te, tc)
    if z is not None and len(z[0]):
        return float(z[1][-1]), "tick"
    k = WT.load_1m(pair)
    m = (te // MIN) * MIN
    i = int(np.searchsorted(k.open_time.values, m))
    if i < len(k) and int(k.open_time.values[i]) == m and np.isfinite(k.o.values[i]):
        return float(k.o.values[i]), "1m"
    return np.nan, "none"


def grid_book(ev, sl):
    w = ev["why"]
    if w == "TP":
        return float(ev["pct_line"])
    if w == "STOP":
        return min(float(ev["pct"]), -sl) - W.SLIP
    if w == "CAP":
        return float(ev["pct"])
    return np.nan


def price_row(r, tc, grid):
    pair = r["pair"]; te = int(r["T"]) + W.BAR_MS + W.T_LAG_MS
    e, esrc = entry_price(pair, te, tc)
    out = dict(pair=pair, T=int(r["T"]), esrc=esrc)
    if not np.isfinite(e):
        out["why"] = "NOENTRY"; return out
    E = e * (1 + W.ENTRY_SLIP / 100)
    out["E"] = E
    P = FX.load_path(pair, te, E, HOR, tc)
    ev = WT.evaluate_late(P, W.TP, W.SL, W.CAP, W.LATE, W.LTP)
    out.update(why=ev["why"], y=W.book(ev), xmin=ev["xmin"], src=FX.path_src(P), on_bar=ev["on_bar"])
    # tick coverage of the 90-min decision window only (the grid's 240-min horizon may run past the covered day)
    out["tick90"] = bool(P["n_tick"] > 0 and P["tick_end"] >= te + (W.CAP + 1) * MIN)
    P1 = FX.load_path(pair, te, E, W.CAP, NoTicks())
    e1 = WT.evaluate_late(P1, W.TP, W.SL, W.CAP, W.LATE, W.LTP)
    out.update(why1=e1["why"], y1=W.book(e1))
    if grid:
        n90 = int(np.searchsorted(P["t"], te + 90 * MIN, side="right"))
        if n90:
            hi, lo, t = P["hi"][:n90], P["lo"][:n90], P["t"][:n90]
            out["mfe90"] = float(hi.max()); out["mae90"] = float(lo.min())
            k = np.flatnonzero(hi >= W.TP)
            out["t_tp"] = (t[k[0]] - te) / MIN if len(k) else np.nan
            k = np.flatnonzero(lo <= -W.SL)
            out["t_sl"] = (t[k[0]] - te) / MIN if len(k) else np.nan
            out["cl90"] = float(P["cl"][n90 - 1]) if P["end_ms"] >= te + 90 * MIN else np.nan
            for m in (5, 15, 30, 60):
                nm = int(np.searchsorted(P["t"], te + m * MIN, side="right"))
                out[f"r{m}"] = float(P["cl"][nm - 1]) if nm and P["end_ms"] >= te + m * MIN else np.nan
        for tp in GRID_TP:
            for sl in GRID_SL:
                for cap in GRID_CAP:
                    out[f"g_{tp:g}_{sl:g}_{cap}"] = grid_book(FX.evaluate(P, tp, sl, cap), sl)
    return out


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        try:
            out.append(price_row(r, tc, r["grid"]))
        except Exception as e:  # noqa: BLE001
            out.append(dict(pair=r["pair"], T=int(r["T"]), why="ERR", err=str(e)[:160]))
    return out


def main():
    t0 = time.time()
    A = pd.read_pickle(W.ALT)
    S = pd.read_pickle(W.SIG)
    ent_eps = set(zip(S[S.T_ent.notna()].pair, S[S.T_ent.notna()].spike_ts.astype("int64")))
    A["grid"] = [(p, int(s)) in ent_eps for p, s in zip(A.pair, A.spike_ts)]
    U = A.sort_values("grid", ascending=False).drop_duplicates(["pair", "T"])[["pair", "T", "grid"]]
    print(f"bars to price {len(U):,} (grid on {int(U.grid.sum()):,})", flush=True)
    jobs = int(os.environ.get("JOBS", "8"))
    items = U.to_dict("records")
    pairs = sorted(U.pair.unique()); g = {p: k % (jobs * 6) for k, p in enumerate(pairs)}
    chunks = [sorted([r for r in items if g[r["pair"]] == k], key=lambda r: r["T"]) for k in range(jobs * 6)]
    with Pool(jobs) as P:
        res = sum(P.map(job, [c for c in chunks if c], chunksize=1), [])
    R = pd.DataFrame(res)
    R.to_pickle(os.path.join(W.OUT, "priced.pkl"))
    print(f"priced {len(R):,} · why {R.why.value_counts().to_dict()} · src {R.src.value_counts().to_dict()} · "
          f"tick90 {R.tick90.mean() * 100:.1f}% · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
