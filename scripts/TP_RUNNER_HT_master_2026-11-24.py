#!/usr/bin/env python3
"""HT master + batch on TODAY's filters and exit, before (TP +6 / −4 / 24 h) → after (3 × 15m ATR runner past the +6 TP, no floor).
Kept fills = reports/KEPT_LIST_WILLY_HT_2026-11-21.csv (dip ≥ 1.4 DL 594 ∧ FAILED_SPIKE DL 527) MASTER rows re-checked for the
DL 619 burst rule on their own 1m bars + the current batch's kept HT fills (BATCH_TODAY_RULES_2026-11-23, all of today's filters).
Walk from the minute after the real fill price; one size X 0.61 × $10k; DCR over every calendar day of each block."""
import os, sys, glob, importlib
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
T = importlib.import_module("TP_HOLD_2026-11-24"); C = T.C
from services.frenzy import pre_vol15_vs60, cloc15
from services.watch_filters import ht_burst_state
MIN, HR = 60_000, 3_600_000; X, ACC = 0.61, 10_000
TAKER, MAKER, SLIP = 0.05, 0.02, 0.10
MS = lambda s: int((pd.to_datetime(s, utc=True, format="mixed") - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1))


def exports():
    P = []
    for f in sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv"))):
        h = pd.read_csv(f, nrows=0).columns
        if "entry_strategy" not in h: continue
        x = pd.read_csv(f, low_memory=False, usecols=[c for c in ("opened_at", "pair", "entry_strategy", "entry_price", "status", "pnl", "quantity") if c in h])
        P.append(x[x.entry_strategy == "FRENZY_WILLY_HT"].assign(src=f))
    X_ = pd.concat(P); X_["k"] = X_.opened_at.astype(str).str[:19]
    return X_.sort_values("src").drop_duplicates(["k", "pair"], keep="last")


def burst(pair, te):
    m0 = te // MIN * MIN
    a = C.k1m(pair, m0 - 75 * MIN, m0)
    if a is None or not len(a["t"]): return "unread"
    rows = [[int(t), o, h, l, c, v] for t, o, h, l, c, v in zip(a["t"], a["o"], a["h"], a["l"], a["c"], a["v"])]
    return ht_burst_state(pre_vol15_vs60(rows, te), cloc15(rows, te))


def walk(pair, te, E):
    tp_px = E * (100 + 6.0 + TAKER) / (100 - MAKER); sl_px = E * (100 - 4.0 + TAKER) / (100 - TAKER)
    m0 = te // MIN * MIN
    a = C.k1m(pair, m0, m0 + 48 * HR + MIN)
    if a is None or len(a["t"]) < 2: return None
    g = C.grid(a, m0, min(m0 + 48 * HR + MIN, int(a["t"][-1]) + MIN))
    n24 = min(len(g["t"]) - 1, 24 * 60)
    net = lambda px, exit_fee: (px / E - 1) * 100 - TAKER - exit_fee
    for i in range(1, n24 + 1):
        if g["l"][i] <= sl_px:
            px = min(sl_px, g["o"][i]) * (1 - SLIP / 100); b = net(px, TAKER)
            return dict(before=b, after=b, why="SL")
        if g["h"][i] >= tp_px:
            b = 6.0
            atr = T.atr15_pct(pair, int(g["t"][i]))
            best = tp_px; ex = None
            for j in range(i + 1, min(len(g["t"]), i + 1 + 24 * 60)):
                st = best * (1 - 3.0 * atr / 100)
                if g["l"][j] <= st:
                    ex = min(st, g["o"][j]); break
                best = max(best, g["h"][j])
            if ex is None: ex = g["c"][min(len(g["t"]) - 1, i + 24 * 60)]
            return dict(before=b, after=net(ex * (1 - SLIP / 100), TAKER), why="TP", atr=atr)
    px = g["c"][n24] * (1 - SLIP / 100); b = net(px, TAKER)
    return dict(before=b, after=b, why="TIME")


def dcr(df, col, days):
    r = df.groupby("day")[col].sum() * X / 100
    r = r.reindex(days, fill_value=0.0)
    return (np.prod(1 + r.values) ** (1 / len(days)) - 1) * 100


def main():
    K = pd.read_csv(os.path.join(ROOT, "reports", "KEPT_LIST_WILLY_HT_2026-11-21.csv"))
    EX = exports()
    M = K[K.block == "MASTER"].merge(EX[["k", "pair", "entry_price"]], on=["k", "pair"], how="left")
    B = pd.read_csv(os.path.join(ROOT, "reports", "BATCH_TODAY_RULES_2026-11-23_fills.csv"))
    B = B[(B.sleeve == "HT") & B.kept.astype(bool)].copy()
    B["k"] = B.opened_at.astype(str).str[:16]
    EX["k16"] = EX.k.str[:16]
    B = B.merge(EX[["k16", "pair", "entry_price", "k"]].rename(columns={"k": "kfull"}), left_on=["k", "pair"], right_on=["k16", "pair"], how="left")
    B["k"] = B.kfull
    rows = []
    for blk, D in (("MASTER", M), ("BATCH", B)):
        for r in D.itertuples():
            te = MS(r.k)
            if blk == "MASTER":
                st = burst(r.pair, te)
                if st == "zone":
                    print("  burst-blocked (DL 619):", r.pair, r.k); continue
            w = walk(r.pair, te, float(r.entry_price))
            if w is None: print("  no path", r.pair, r.k); continue
            rows.append(dict(block=blk, pair=r.pair, k=r.k, day=str(r.k)[:10], **w))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(T.OUT, "ht_master_batch_runner.csv"), index=False)
    for blk, g in R.groupby("block", sort=False):
        days = pd.date_range(g.day.min(), g.day.max()).strftime("%Y-%m-%d") if blk == "MASTER" else pd.date_range("2026-11-18", "2026-11-23").strftime("%Y-%m-%d")
        for col, nm in (("before", "BEFORE (TP +6)"), ("after", "AFTER (3×ATR runner)")):
            print(f"{blk:6s} {nm:22s} N {len(g):2d} · WR {100*(g[col]>0).mean():3.0f}% · avg {g[col].mean():+.2f} · ${(g[col]*X*ACC/100).sum():+,.0f} · DCR {dcr(g, col, days):+.2f}%/day over {len(days)} days")
        tp = g[g.why == "TP"]
        print(f"        TP hits {len(tp)} · runner Δ per hit {(tp.after - tp.before).mean():+.2f} pp · better {100*((tp.after - tp.before) > 0).mean():.0f}% · exit mix {g.why.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
