#!/usr/bin/env python3
"""BATCH28 deep review — EXHAUSTION_SHORT block (2026-11-27). Research only, no bot / config change.
Current batch (paper 11-18 → 11-27) = EXH's whole real history (EXH has no earlier master batch). Every closed fill re-priced on today's
exit (TP min(20×ATR, 7) maker / SL min(1.5×ATR, 6) / 24 h, 1m walk, real funding — the DL 649 machinery, parity reported), then each
watch line read before → after at ONE size (0.61 × equity): N · WR · avg · Σ · DCR · max DD · worst day; backtest (2 years, TP 7,
today's filters) for the same cut. Usage: venv/bin/python -I scripts/BATCH28_REVIEW_EXH_2026-11-27.py <orders.csv>
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R = os.path.dirname(HERE)
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-27_14-36-23.csv")
sys.argv = [sys.argv[0], EXPORT]
sp = importlib.util.spec_from_file_location("exh_deep", os.path.join(HERE, "EXH_EARLY_SPIKE_DEEP_2026-11-26.py"))
DP = importlib.util.module_from_spec(sp); sp.loader.exec_module(DP)
SA = DP.SA; DAY = DP.DAY; SIZE = SA.SIZE
OUT = os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data")


def stats(K, d0, d1, label):
    if not len(K):
        return dict(cut=label, N=0)
    p, _ = SA.path(K, d0, d1)
    return dict(cut=label, N=len(K), WR=(K.net > 0).mean() * 100, avg=K.net.mean(), sum=K.net.sum(), **{k: p[k] for k in ("DCR", "maxDD", "worst_day")})


def main():
    K, par, n_all = DP.real_fills()
    par.to_csv(os.path.join(OUT, "exh_parity.csv"), index=False)
    print(f"EXH closed fills {n_all}; parity exit kind {int((par.booked_kind == par.model_kind).sum())}/{len(par)} · median |px err| "
          f"{par.exit_px_err.abs().median():.3f} %")
    x = pd.read_csv(EXPORT, low_memory=False)
    x = x[(x.entry_strategy == "EXHAUSTION_SHORT") & (x.status == "CLOSED")]
    x["T"] = [int(pd.Timestamp(v).value // 10**6) for v in x.opened_at]
    cols = ["entry_bear_pct", "entry_exh_ret_1h_pct", "entry_exh_universe", "entry_exh_spot_symbol", "pnl", "quantity", "entry_price"]
    K = K.merge(x[["pair", "T"] + cols], on=["pair", "T"], how="left")
    K["booked_usd"] = K.pnl
    K.to_csv(os.path.join(OUT, "exh_batch_repriced.csv"), index=False)
    d0, d1 = int(K.day.min()), int(max(K.xm.max() // DAY, K.day.max()))
    kept = K[~K.blk_nospot & ~K.blk_ssc]      # today's armed filters (ZKP: spot stamp missing → shown both ways)
    print(f"kept by today's armed filters {len(kept)} of {len(K)}; universe {K.entry_exh_universe.value_counts(dropna=False).to_dict()}")
    print("booked avg (as traded) %.2f  · re-priced TP7 avg %.2f" % (K.booked_net.mean(), K.net.mean()))
    zones = {
        "EXH_AFTER_DROP (1h ret −10…−5)": lambda d: d.entry_exh_ret_1h_pct.between(-10, -5),
        "EXH_BEAR80 (bear % > 80)": lambda d: d.entry_bear_pct > 80,
        "EXH_EARLY_SPIKE (< 0.5 h)": lambda d: d.hrs < 0.5,
    }
    rows = [stats(K, d0, d1, "ALL closed (booked exit → TP7 re-priced)"), stats(kept, d0, d1, "kept by armed filters (before)")]
    allz = np.zeros(len(kept), bool)
    for name, f in zones.items():
        z = f(kept).fillna(False).to_numpy()
        allz |= z
        rows.append(stats(kept[z], d0, d1, f"  zone: {name}"))
        rows.append(stats(kept[~z], d0, d1, f"  after: block {name}"))
    rows.append(stats(kept[~allz], d0, d1, "  after: block all three lines"))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "exh_lines_batch.csv"), index=False)
    print(T.round(2).to_string(index=False))
    print(K[["pair", "T", "hrs", "entry_bear_pct", "entry_exh_ret_1h_pct", "booked_kind", "kind", "booked_net", "net"]].assign(
        T=lambda d: pd.to_datetime(d['T'], unit='ms')).round(2).to_string(index=False))


if __name__ == "__main__" and os.environ.get("BT") != "1":
    main()


def backtest_lines():
    """2-year backtest (TP_GRID F7 = today's exit, DL 582 applied, one position per pair) — BEAR80 / EARLY_SPIKE / both, before → after:
    N · avg · DCR · max DD · worst day per scope (ALL · DESIGN · HOLDOUT · each half-year). Breadth = EXH_BREADTH_2026-11-25 rebuild
    (from 2024-06 only — majors in the store); rows without a breadth read are kept on both sides."""
    D, _ = SA.load_bt()
    B = pd.read_csv(os.path.join(R, "reports", "EXH_BREADTH_2026-11-25_trades.csv"), usecols=["pair", "entry_ms", "bear"])
    D = D.merge(B, on=["pair", "entry_ms"], how="left")
    D = D[D.entry_ms >= B.entry_ms.min()].reset_index(drop=True)        # the breadth-covered span only (like-for-like)
    cuts = {"today (no block)": np.zeros(len(D), bool), "block BEAR80": (D.bear > 80).to_numpy(), "block EARLY < 0.5 h": (D.hrs < 0.5).to_numpy(),
            "block both": ((D.bear > 80) | (D.hrs < 0.5)).to_numpy()}
    rows = []
    for name, blk in cuts.items():
        K = D[SA.EV.one_pos(D.pair.values, D.entry_ms.values, D.xm.values, ~blk)].reset_index(drop=True)
        K["xd"] = K.xm // DAY
        scopes = [("ALL", K)] + [(b, K[K.blk == b]) for b in ("DESIGN", "HOLDOUT")] + [(h, K[K.half == h]) for h in sorted(K.half.unique())]
        for sc, Kc in scopes:
            if not len(Kc):
                continue
            p, _ = SA.path(Kc, int(Kc.xd.min()), int(Kc.xd.max()))
            rows.append(dict(cut=name, scope=sc, N=len(Kc), WR=(Kc.net > 0).mean() * 100, avg=Kc.net.mean(), DCR=p["DCR"], maxDD=p["maxDD"],
                             worst_day=p["worst_day"]))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "exh_lines_backtest.csv"), index=False)
    print(T.round(2).to_string(index=False))


if __name__ == "__main__" and os.environ.get("BT") == "1":
    backtest_lines()
