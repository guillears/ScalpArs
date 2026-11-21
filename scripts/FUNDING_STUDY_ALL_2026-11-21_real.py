#!/usr/bin/env python3
"""FUNDING_STUDY_ALL_2026-11-21 — REAL-fill re-walks on 1m with real settlements (research only).

  EXH  : the 8 closed current-batch fills + the OPEN ENSO fill (journal OPEN 2026-11-21 03:30:33 @ 0.480889, ATR 3.470864 %, SL 5.206 %;
         walked to the last complete bar fetched = an OPEN mark, flagged) under TODAY's exit: TP min(20 × stamped ATR, 7) / SL = the
         stamped SL re-based to min(1.5 × ATR, 6) / 24 h, price stop.
         PARITY walk = each fill's OWN era exit (stamped TP; RARE #32 clamped to 7 at the DL 609 deploy) vs the booked % (incl. funding).
  FB   : the two real FB fills whose funding can reach the smallest budget (T 10-25 paid 3.39 %, KERNEL 11-20 paid 1.60 %) — every other
         FB real fill paid < 0.5 % (Δ = 0 by construction); today's exit TP 5 / SL 10 counting funding paid (DL 607) / 72 h.
Walk = scripts/FUNDING_STUDY_ALL_2026-11-21_bt.walk_all (stop first on the HIGH, maker TP ≥ 0.05 % through, new market exits taker 0.05 %
+ 0.10 % slip), path from the minute AFTER the fill; entry = the booked fill price; entry cost = taker 0.05 % (the entry slip is in the fill).
Funding = Σ real settlements in (fill, exit], each rate × (1m open at the settlement ÷ fill) — the engine's rate × qty × mark (parity table).
Output: reports/FUNDING_STUDY_ALL_2026-11-21_data/real_walk.pkl"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "FUNDING_STUDY_ALL_2026-11-21_data")
CACHE = os.path.join(ROOT, "reports", ".funding_cache")
HR = 3_600_000


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


BT = _imp("FSB", "FUNDING_STUDY_ALL_2026-11-21_bt.py")
C = BT.C


def frate(pair):
    for f in (os.path.join(CACHE, f"funding_{pair}.csv"), os.path.join(CACHE, f"rates_{pair}.csv")):
        if os.path.exists(f):
            x = pd.read_csv(f).sort_values("t")
            return x.t.values.astype("int64"), x.rate.values.astype(float)
    return np.zeros(0, "int64"), np.zeros(0)


DEPLOY_609 = int(pd.Timestamp("2026-11-21T20:59:49").value // 10**6)   # b3e5a19d commit time (UTC): open TP-10 rows clamped to 7


def one(pair, te_ms, E, tp, sl, k, hold_h, paid_stop, time_cells=True, deploy_clamp=False):
    t, o, h, l, c = [k[x].values for x in ("t", "o", "h", "l", "c")]
    t = t.astype("int64")
    i0 = int(np.searchsorted(t, (te_ms // 60_000 + 1) * 60_000, "left"))
    ft, fr = frate(pair)
    F = C.settle_on_bars(t, o, E, te_ms, ft, fr, 1, markscale=True)
    end = te_ms + hold_h * HR
    if deploy_clamp and tp > 7.0 and DEPLOY_609 < min(end, int(t[-1]) + 60_000):   # parity: TP 10 until the DL 609 deploy, 7 after
        w1 = BT.walk_all(t, o, h, l, c, i0, E, tp, sl, DEPLOY_609, F, 0.10, 0.05, 0.05, 0.02, paid_stop, False)
        if w1 is not None and w1["base_why"] != "TIME":
            w1["open_mark"] = False
            return w1
        i1 = int(np.searchsorted(t, (DEPLOY_609 // 60_000 + 1) * 60_000, "left"))
        w = BT.walk_all(t, o, h, l, c, i1, E, 7.0, sl, min(end, int(t[-1]) + 60_000), F, 0.10, 0.05, 0.05, 0.02, paid_stop, False)
        w["base_hold"] += (int(t[i1]) - int(t[i0])) / HR
        w["open_mark"] = int(t[-1]) + 60_000 < end
        return w
    w = BT.walk_all(t, o, h, l, c, i0, E, tp, sl, min(end, int(t[-1]) + 60_000), F, 0.10, 0.05, 0.05, 0.02, paid_stop, time_cells)
    w["open_mark"] = int(t[-1]) + 60_000 < end
    k0 = w["base_hold"]
    return w


def main():
    ex = pd.read_csv(next((p for p in (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports_archive", "scalpars_orders_paper_2026-11-21_22-18-17.csv"), os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-21_22-18-17.csv")) if os.path.exists(p)), os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-21_22-18-17.csv")), low_memory=False)
    e = ex[ex.entry_strategy == "EXHAUSTION_SHORT"].copy()
    rows = []
    for r in e.itertuples():
        te = int(pd.Timestamp(r.opened_at).value // 10**6)
        kf = glob.glob(os.path.join(CACHE, f"k1m_{r.pair}_{te // 60_000 * 60_000}.csv"))
        k = pd.read_csv(kf[0])
        atr = float(r.entry_exh_atr_pct)
        sl_today = min(1.5 * atr, 6.0)
        tp_today = min(20 * atr, 7.0)
        par = one(r.pair, te, float(r.entry_price), float(r.entry_exh_tp_pct), float(r.entry_exh_sl_pct), k, 24, False, deploy_clamp=True)
        w = one(r.pair, te, float(r.entry_price), tp_today, sl_today, k, 24, False)
        w.update(sleeve="EXHAUSTION_SHORT", id=r.id, pair=r.pair, opened_at=r.opened_at, closed_at=r.closed_at, booked=r.pnl_percentage,
                 booked_reason=r.close_reason, booked_fund_usd=r.funding_fee_usd, notional=r.entry_price * r.quantity,
                 stamp_rate=r.entry_funding_rate, par_net=par["base_net"], par_why=par["base_why"], par_fund=par["base_fund"],
                 par_hold=par["base_hold"], tp_today=tp_today, sl_today=sl_today, peak_pnl=r.peak_pnl)
        ft, fr = frate(r.pair)
        j = int(np.searchsorted(ft, te, "right"))
        w.update(last_rate=fr[j - 1] if j else np.nan, next_rate=fr[j] if j < len(fr) else np.nan,
                 interval_h=C.interval_h(ft, te))
        rows.append(w)
    # the OPEN ENSO fill (not in the export)
    te = int(pd.Timestamp("2026-11-21T03:30:33").value // 10**6)
    k = pd.read_csv(glob.glob(os.path.join(CACHE, f"k1m_ENSOUSDT_{te // 60_000 * 60_000}.csv"))[0])
    atr = 3.470864
    w = one("ENSOUSDT", te, 0.480889, min(20 * atr, 7.0), min(1.5 * atr, 6.0), k, 24, False)
    par = one("ENSOUSDT", te, 0.480889, 10.0, min(1.5 * atr, 6.0), k, 24, False, deploy_clamp=True)
    w.update(par_net=par["base_net"], par_why=par["base_why"], par_fund=par["base_fund"], par_hold=par["base_hold"])
    ft, fr = frate("ENSOUSDT"); j = int(np.searchsorted(ft, te, "right"))
    w.update(sleeve="EXHAUSTION_SHORT", id="ENSO(open)", pair="ENSOUSDT", opened_at="2026-11-21T03:30:33", closed_at=None, booked=np.nan,
             booked_reason="OPEN", tp_today=min(20 * atr, 7.0), sl_today=min(1.5 * atr, 6.0), last_rate=fr[j - 1] if j else np.nan,
             next_rate=fr[j] if j < len(fr) else np.nan, interval_h=C.interval_h(ft, te), last_bar=pd.to_datetime(k.t.max(), unit="ms"))
    rows.append(w)
    # FB: every kept real fill (KEPT_LIST_FLATBASE ∩ FUNDING_STOP real_rows: today's-stack entry — MODEL_CONFIRM or REAL — E / te from there)
    fb = pd.read_csv(os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data", "real_rows.csv"))
    fb = fb.merge(pd.read_csv(os.path.join(ROOT, "reports", "KEPT_LIST_FLATBASE_2026-11-21.csv")), on=["pair", "k"])
    for r in fb.itertuples():
        fn = os.path.join(ROOT, "reports", "FUNDING_STOP_2026-11-21_data", "k1m_KERNELUSDT.csv") if r.pair == "KERNELUSDT" else \
            os.path.join(ROOT, "reports", "TP_GRID_2026-11-20_FB_data", "k1m", f"{r.pair}.csv")
        k = pd.read_csv(fn).rename(columns={"open_time": "t"})
        te = int(r.te)
        t = k.t.values.astype("int64"); o, h, l, c = [k[x].values.astype(float) for x in "ohlc"]
        i0 = int(np.searchsorted(t, (te // 60_000 + 1) * 60_000, "left"))
        ft, fr = frate(r.pair)
        F = C.settle_on_bars(t, o, r.E, te, ft, fr, 1, markscale=True)
        w = BT.walk_all(t, o, h, l, c, i0, float(r.E), 5.0, 10.0, min(te + 72 * HR, int(t[-1]) + 60_000), F, 0.25, 0.05, 0.05, 0.02, True,
                        time_cells=False)
        w.update(sleeve="FLATBASE_SHORT", id=f"{r.pair} {r.k[:10]}", pair=r.pair, opened_at=r.k, block=r.block, kind=r.kind,
                 ref_Bp=r.Bp_net, ref_Bp_fund=r.Bp_fund, ref_A=r.A_net, te=te, last_rate=r.rate_at_entry, interval_h=r.interval_h)
        rows.append(w)
    R = pd.DataFrame(rows)
    R.to_pickle(os.path.join(OUT, "real_walk.pkl"))
    pd.set_option("display.width", 250)
    print(R[["sleeve", "id", "pair", "booked", "booked_reason", "par_net", "par_why", "par_fund", "base_net", "base_why", "base_fund",
             "base_hold", "label", "reach", "open_mark"]].round(3).to_string())


if __name__ == "__main__":
    main()
