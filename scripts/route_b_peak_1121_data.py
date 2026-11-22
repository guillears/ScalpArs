#!/usr/bin/env python3
"""ROUTE_B_PEAK_STUDY 2026-11-21 — data: cohorts (REAL fills + SHADOW priced signals) and per-trade minute panels (1m klines + Route B
per-minute summaries + REST open interest). Research only. Pre-registration: reports/ROUTE_B_PEAK_STUDY_2026-11-21_PREREG.md.

Reuses the DL 612 loader (scripts/route_b_signals: S3 per-minute summaries cached under reports/ROUTE_B_SIGNALS_data/, klines, orders).
Output: reports/ROUTE_B_SIGNALS_data/peak_1121/panels.pkl (gitignored folder).
Usage: venv/bin/python -I scripts/route_b_peak_1121_data.py"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "route_b_signals"))
import sources as S  # noqa: E402
from common import DATA, REC_START, MIN_MS  # noqa: E402

OUT = os.path.join(DATA, "peak_1121")
SLEEVES = ("EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FLATBASE_SHORT", "FRENZY_WILLY_HT", "FRENZY_WILLY")
PRE_MIN = 600          # klines warm-up before entry (5m RSI14 needs ≥ 70 bars)
CTX_MIN = 30           # Route B context ± 30 min around the life
BOOK_COLS = ["spread_bps", "bid_usd_02", "ask_usd_02", "bid_usd_05", "ask_usd_05", "bid_usd_1", "ask_usd_1", "imb_05",
             "cost_buy_2500", "cost_sell_2500", "cost_buy_10000", "cost_sell_10000", "mid", "mid_hi", "mid_lo"]
FLOW_COLS = ["n_trades", "buy_usd", "sell_usd", "max_trade_usd"]


def ms(ts):
    return int(pd.Timestamp(ts).value // 1_000_000)


def cohorts():
    O, nprobe = S.load_orders(since=pd.Timestamp("2026-11-19", tz="UTC"))
    O = O.copy()
    O["ot"] = pd.to_datetime(O.opened_at, utc=True, format="mixed")
    O["ct"] = pd.to_datetime(O.closed_at, utc=True, format="mixed", errors="coerce")
    allrec = O[O.ot >= REC_START]
    real = allrec[(allrec.status == "CLOSED") & allrec.ct.notna()].copy()
    rows = []
    for r in real.itertuples():
        rows.append(dict(cohort="REAL", sleeve=r.entry_strategy, pair=r.pair, side=1 if r.direction == "LONG" else -1,
                         entry_ms=ms(r.ot), exit_ms=ms(r.ct), entry_px=float(r.entry_price), net_pct=float(r.pnl_percentage),
                         exit_reason=str(r.close_reason), tp_pct=np.nan, key=f"{r.opened_at}|{r.pair}|{r.direction}"))
    sig = pd.read_pickle(os.path.join(DATA, "signals.pkl.gz"))
    pri = sig.out_priced.fillna(False).astype(bool) & sig.out_final.fillna(False).astype(bool)
    nod = ~sig.poss_dup.fillna(False).astype(bool)
    sh = sig[pri & nod & ~sig.status.astype(str).str.startswith("OPENED")]
    for r in sh.itertuples():
        rows.append(dict(cohort="SHADOW", sleeve=r.sleeve, pair=r.pair, side=1 if r.direction == "LONG" else -1,
                         entry_ms=int(r.out_entry_ms), exit_ms=int(r.out_exit_ms), entry_px=float(r.out_entry_px),
                         net_pct=float(r.out_net_pct), exit_reason=str(r.out_reason), tp_pct=float(r.out_tp_pct),
                         key=f"{r.signal_time}|{r.pair}|{r.status}"))
    # the REAL fill's TP under TODAY's exit for the reached-X split: the linked OPENED signal's priced TP when present
    op = sig[sig.status.astype(str).str.startswith("OPENED") & sig.fill_key.notna()]
    tpmap = {str(k): float(v) for k, v in zip(op.fill_key, op.out_tp_pct) if v == v}
    C = pd.DataFrame(rows)
    for i, r in C[C.cohort == "REAL"].iterrows():
        C.at[i, "tp_pct"] = tpmap.get(r.key.split("|")[0][:19] + "|" + r.pair + "|" + ("LONG" if r.side > 0 else "SHORT"), np.nan)
    open_n = int((allrec.status != "CLOSED").sum())
    return C, dict(probes=nprobe, open_at_export=open_n, real_all=len(allrec),
                   real_by_sleeve=allrec.groupby("entry_strategy").size().to_dict(),
                   orders_export_newest=str(O.src_file.iloc[-1]) if len(O) else "")


def oi_hist(sym, a_ms, b_ms, cache):
    """REST openInterestHist 5m (≤ 30 days back); rows (label_ms, oi). Cached per coin."""
    path = os.path.join(cache, f"oi_{sym}.pkl")
    if os.path.exists(path):
        d = pd.read_pickle(path)
        if len(d) and d.t.min() <= a_ms + 5 * MIN_MS and d.t.max() >= b_ms - 15 * MIN_MS:
            return d
    rows, s = [], a_ms
    while s < b_ms:
        js = S.limiter().get("/futures/data/openInterestHist", dict(symbol=sym, period="5m", startTime=s, endTime=b_ms, limit=500), 1)
        if not js:
            break
        rows += [(int(x["timestamp"]), float(x["sumOpenInterest"])) for x in js]
        if len(js) < 500:
            break
        s = int(js[-1]["timestamp"]) + 1
        time.sleep(0.2)
    d = pd.DataFrame(rows, columns=["t", "oi"]).drop_duplicates("t").sort_values("t")
    d.to_pickle(path)
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    C, meta = cohorts()
    C = C[C.sleeve.isin(SLEEVES)].reset_index(drop=True)
    keys = sorted(k for k in os.listdir(os.path.join(DATA, "minutes")))
    s3keys = [k.replace(".pkl.gz", ".csv.gz").replace("features_", "features/", 1) for k in keys]
    s3keys = [k[:len("features/2026-11-20")] + "/" + k[len("features/2026-11-20_"):] for k in s3keys]
    book, mark, liq, nh = S.load_minutes(s3keys)
    print(f"Route B minutes: {nh} hours · book rows {len(book):,} · mark {len(mark):,} · liq {len(liq):,}")
    bk = {s: g.set_index("minute") for s, g in book.groupby("sym")}
    mk = {s: g.set_index("minute") for s, g in mark.groupby("sym")}
    lq = {s: g.set_index("minute") for s, g in liq.groupby("sym")}
    rb_end = int(book.minute.max())
    now = int(time.time() * 1000)
    panels = []
    for r in C.itertuples():
        e0 = r.entry_ms - r.entry_ms % MIN_MS
        x0 = r.exit_ms - r.exit_ms % MIN_MS
        K = S.klines(r.pair, e0 - PRE_MIN * MIN_MS, x0 + (CTX_MIN + 1) * MIN_MS, now)
        if not len(K):
            panels.append(dict(meta=r._asdict(), ok=False, why="no klines"))
            continue
        P = K.rename(columns={"open_ms": "t"}).set_index("t")
        idx = P.index.values
        b = bk.get(r.pair)
        if b is not None:
            b = b.reindex(idx)
            for c in BOOK_COLS:              # a book row older than 2 minutes is not used
                P[c] = b[c].ffill(limit=2).values if c in b else np.nan
            for c in FLOW_COLS:
                P[c] = b[c].values if c in b else np.nan
            P["has_book"] = b["n_book"].notna().values
        else:
            for c in BOOK_COLS + FLOW_COLS:
                P[c] = np.nan
            P["has_book"] = False
        m = mk.get(r.pair)
        for c in ("mark", "index", "funding"):
            P[c] = m.reindex(idx)[c].ffill(limit=2).values if m is not None else np.nan
        q = lq.get(r.pair)
        for c in ("liq_long_usd", "liq_short_usd"):
            P[c] = q.reindex(idx)[c].fillna(0).values if q is not None else 0.0
        P.loc[P.index > rb_end, ["liq_long_usd", "liq_short_usd"]] = np.nan
        # OI: label T available at T + 10 min (DL 612 mislabel + margin) → as-of join on availability
        oi = oi_hist(r.pair, e0 - 120 * MIN_MS, x0 + 31 * MIN_MS, OUT) if not S.limiter().stopped else pd.DataFrame(columns=["t", "oi"])
        if len(oi):
            av = oi.t.values + 10 * MIN_MS
            j = np.searchsorted(av, idx, "right") - 1
            P["oi"] = np.where(j >= 0, oi.oi.values[np.maximum(j, 0)], np.nan)
        else:
            P["oi"] = np.nan
        panels.append(dict(meta=r._asdict(), ok=True, P=P.reset_index()))
    pd.to_pickle(dict(panels=panels, meta=meta, rb_end=rb_end, built=pd.Timestamp.utcnow().isoformat(),
                      used_weight=S.limiter().used, stopped=S.limiter().stopped), os.path.join(OUT, "panels.pkl"))
    print(meta)
    print(C.groupby(["sleeve", "cohort"]).size().to_string())
    print(f"panels {sum(p['ok'] for p in panels)} / {len(panels)} · binance stopped={S.limiter().stopped}")


if __name__ == "__main__":
    main()
