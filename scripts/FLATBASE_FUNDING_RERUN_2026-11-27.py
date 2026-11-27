#!/usr/bin/env python3
"""FLATBASE_FUNDING_RERUN_2026-11-27 — DL 616's FLATBASE funding rules re-run with the REAL fills incl. this batch + one new variant
(operator: "yes run the flatbase funding re-run with batch fills"; MOVE paid −4.18 % funding in 21 h). Research only.
PRIOR (DL 616, 11-21): backtest 545 — entry skip on adverse / hourly funding REFUTED (skipped fills better), in-trade budget REFUTED (−0.17),
NET stop (DL 607) kept. Real fills then: only 2 FB fills paid enough to reach a budget. NEW: (1) the real cohort on TODAY's filters = the
17 master fills of reports/KEPT_LIST_FLATBASE_2026-11-21.csv + the batch kept fills (KERNEL, MMT, PIPPIN, SLX, BIGTIME) + the open MOVE /
LAYER shorts (entered at the rebuild's confirm price, marked at the last bar = OPEN), walked on 5m with REAL Binance funding settlements;
(2) the never-tested HOURLY rule, pre-registered here: at the first settlement whose gap from the previous one is ≤ 1 h (the coin is on
hourly funding), exit at that bar's open if the short is not ≥ G % in our favour, G ∈ {0, 2} — ONE-SHOT: only that first hourly settlement
is checked (a later re-check is a different, untested rule); SWITCH = the same but only when the coin was NOT hourly at entry (N = 0 in both
sets → not testable). The export's batch rows are taken as kept: only SLX / BIGTIME are FB fills opened ≥ 11-21 (no probe / filter screen).
Real paid_rate = the last rate settled at the entry (entries sit on 4 h boundaries) — the backtest's DL 616 stamp may differ in definition. Cells: ENTRY skip paid rate ≥ X ∈ {0.05, 0.10, 0.25, 0.5, 1.0} % or hourly at entry (6) · BUDGET F × G (15, DL 616's
walker) · HOURLY_G0 / G2 · SWITCH_G0 / G2 (4). Backtest: DL 616's bt_FB.pkl for the old families + the 4 new cells walked here (same walker,
same funding table). Today's exit: TP −5 maker / stop +10 counting funding PAID / 72 h.
Usage: venv/bin/python -I scripts/FLATBASE_FUNDING_RERUN_2026-11-27.py <orders.csv>
"""
import importlib.util
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE); sys.path.insert(0, R)
EXPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-27_16-16-13.csv")
sys.argv = [sys.argv[0], EXPORT]


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn)); m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m); return m


BT = _imp("fsb", "FUNDING_STUDY_ALL_2026-11-21_bt.py")
ET = _imp("fbet", "FLATBASE_ENTRY_TIMING_2026-11-26.py")
C = BT.C
HR, BAR = 3_600_000, 300_000
OUT = os.path.join(R, "reports", "FLATBASE_FUNDING_RERUN_2026-11-27")
X_SIZE = 0.61


def hourly_exit(t, o, i0, k0, E, T_fill, fs_t, F, G, switch, slip, taker, entry_cost):
    """first hourly settlement after the fill (gap ≤ 1 h + 2 min) on or before the base exit bar k0 → forced exit at that bar's open if the
    short is < G % in favour. switch: only when the interval at entry was > 1 h. → (net, why) or None (rule not triggered)."""
    if fs_t is None or len(fs_t) < 2:
        return None
    if switch and C.interval_h(fs_t, T_fill) <= 1:
        return None
    gaps = np.diff(fs_t)
    for k in np.flatnonzero(gaps <= HR + 120_000) + 1:
        s = int(fs_t[k])
        if s <= T_fill:
            continue
        j = int(np.searchsorted(t, s - s % BAR, "left"))
        if j > k0:
            return None
        if j < i0:
            continue
        fav = (1 - o[j] / E) * 100
        if fav >= G:
            return None
        Xf = o[j] * (1 + slip / 100)
        return (1 - Xf / E) * 100 - entry_cost - taker * Xf / E + F[j], "HOURLY"
    return None


def fetch_funding(pair, a, b):
    rows, s = [], a
    while s < b:
        u = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={pair}&startTime={s}&endTime={b}&limit=1000"
        js = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-client/1.0"}), timeout=30))
        rows += js
        if len(js) < 1000:
            break
        s = int(js[-1]["fundingTime"]) + 1
        time.sleep(0.7)
    time.sleep(0.7)
    return np.array([int(r["fundingTime"]) for r in rows], dtype="int64"), np.array([float(r["fundingRate"]) for r in rows])


def real_cohort():
    S = pd.read_pickle(os.path.join(R, "reports", "FLATBASE_ENTRY_TIMING_2026-11-26_data", "sim.pkl"))
    L = S[(S["var"] == "LIVE") & S.traded & S.src.isin(["MASTER", "BATCH"])]
    rows = [dict(pair=r.pair, te=int(r.te), E=float(r.E), src=r.src, open=False) for r in L.itertuples()]
    o = pd.read_csv(EXPORT, low_memory=False)
    o = o[(o.entry_strategy == "FLATBASE_SHORT") & (o.status == "CLOSED") & (pd.to_datetime(o.opened_at) >= "2026-11-21")]
    rows += [dict(pair=r.pair, te=int(pd.Timestamp(r.opened_at).value // 10**6) // BAR * BAR, E=float(r.entry_price), src="BATCH", open=False)
             for r in o.itertuples()]
    W = pd.read_csv(os.path.join(R, "reports", "FLATBASE_REBUILD_2026-11-27_data", "window_episodes.csv"))
    rc = pd.read_pickle(os.path.join(R, "reports", "FLATBASE_REBUILD_2026-11-27_data", "recon_1127.pkl"))
    for p in ("MOVEUSDT", "LAYERUSDT"):
        e = rc[(rc.pair == p) & (rc.cf_entry_t >= pd.Timestamp("2026-11-26").value // 10**6)].iloc[0]
        rows.append(dict(pair=p, te=int(e.cf_entry_t), E=float(e.cf_entry_px), src="OPEN", open=True))
    return rows


def walk_real(rows):
    out = []
    now = int(time.time() * 1000) // BAR * BAR - BAR
    for r in rows:
        t, o, h, l, c, _ = ET.real_arrays(r["pair"], r["te"] - 2 * HR, min(r["te"] + 73 * HR, now))
        i0 = int(np.searchsorted(t, r["te"]))
        fs_t, fs_r = fetch_funding(r["pair"], r["te"] - 3 * 86_400_000, r["te"] + 73 * HR)
        F = C.settle_on_bars(t, o, r["E"], r["te"] + 10_000, fs_t, fs_r, 5, markscale=True)
        end_t = min(r["te"] + 72 * HR, int(t[-1]) + BAR)
        w = BT.walk_all(t, o, h, l, c, i0, r["E"], 5.0, 10.0, end_t, F, 0.10, 0.05, 0.05, 0.02, True, time_cells=False)
        if w is None:
            continue
        k0 = int(np.searchsorted(t, int(t[i0]) + int(round((w["base_hold"] - 1 / 60) * HR)), "left"))   # (review) by time, gap-safe
        for G in (0, 2):
            for sw in (False, True):
                hx = hourly_exit(t, o, i0, k0, r["E"], r["te"] + 10_000, fs_t, F, G, sw, 0.10, 0.05, 0.05)
                w[f"{'SWITCH' if sw else 'HOURLY'}_G{G}_net"] = hx[0] if hx else w["base_net"]
        w.update(pair=r["pair"], te=r["te"], src=r["src"], open=r["open"], paid_rate=-(float(C.last_rate(fs_t, fs_r, r["te"])) * 100),
                 interval_h=C.interval_h(fs_t, r["te"]), fund_paid=-w["base_fund"])
        out.append(w)
    return pd.DataFrame(out)


def bt_hourly():
    """the 4 new cells on the 545 backtest fills — DL 616's own FB walker and funding table."""
    BT.fb_setup()
    X = BT.fb_rows()
    OC, RC, Ft = BT.G["OC"], BT.G["RC"], BT.G["F"]
    res = []
    for pair, g in X.groupby("pair"):
        x = RC.W.load(pair)
        fx = Ft.get(pair)
        if x is None:
            continue
        t, o, h, l, c = [np.asarray(a) for a in x[:5]]; t = t.astype("int64")
        tidx = {int(tt): k for k, tt in enumerate(t)}
        for r in g.itertuples():
            i = tidx.get(int(r.cf_entry_t))
            if i is None:
                continue
            E = float(r.cf_entry_px); te = int(r.cf_entry_t) + 10_000; end_t = int(t[i]) + 72 * HR
            tot, miss = OC.fund(Ft, pair, te, end_t)
            fs_t = None if (miss or fx is None or not len(fx[0])) else fx[0]
            F = np.zeros(len(t)) if fs_t is None else C.settle_on_bars(t, o, E, te, fs_t, fx[3], 5, markscale=True)
            w = BT.walk_all(t, o, h, l, c, i, E, 5.0, 10.0, end_t, F, float(r.slip_x), RC.TAKER, RC.TAKER, RC.MAKER, True, time_cells=False)
            if w is None:
                continue
            k0 = int(np.searchsorted(t, int(t[i]) + int(round((w["base_hold"] - 1 / 60) * HR)), "left"))   # (review) by time, gap-safe
            d = dict(rid=r.rid, te=int(r.cf_entry_t), base_net=w["base_net"])
            for G in (0, 2):
                for sw in (False, True):
                    hx = hourly_exit(t, o, i, k0, E, te, fs_t, F, G, sw, float(r.slip_x), RC.TAKER, RC.TAKER)
                    d[f"{'SWITCH' if sw else 'HOURLY'}_G{G}_net"] = hx[0] if hx else w["base_net"]
                    d[f"{'SWITCH' if sw else 'HOURLY'}_G{G}_hit"] = hx is not None
            res.append(d)
    return pd.DataFrame(res)


def summarize(D, label, cells):
    rows = []
    b = D.base_net
    for nm, col in cells:
        v = D[col] if col in D else None
        if v is None:
            continue
        hit = (v - b).abs() > 1e-9
        rows.append(dict(set=label, rule=nm, N=len(D), changed=int(hit.sum()), avg_before=b.mean(), avg_after=v.mean(), delta=(v - b).mean(),
                         changed_before=b[hit].mean() if hit.any() else np.nan, changed_after=v[hit].mean() if hit.any() else np.nan))
    return rows


def main():
    os.makedirs(OUT + "_data", exist_ok=True)
    real = walk_real(real_cohort())
    real.to_pickle(OUT + "_data/real.pkl")
    print(real[["pair", "src", "open", "base_why", "base_net", "fund_paid", "paid_rate", "interval_h"]].round(2).to_string(index=False))
    budget = [(f"BUDGET F{F:g} G{G}", f"B{F:g}_{G}_net") for F in BT.F_GRID for G in BT.G_GRID]
    newc = [(f"{k} G{G}", f"{k}_G{G}_net") for k in ("HOURLY", "SWITCH") for G in (0, 2)]
    rows = summarize(real, "REAL (closed + open marks)", budget + newc)
    # entry skips on real
    for X in BT.X_GRID:
        m = real.paid_rate >= X
        aft = real.base_net.where(~m, 0.0).mean()            # (review) a skip counts as 0 % over N — comparable with the exit rows
        rows.append(dict(set="REAL (closed + open marks)", rule=f"ENTRY skip paid ≥ {X:g} %", N=len(real), changed=int(m.sum()),
                         avg_before=real.base_net.mean(), avg_after=aft, delta=aft - real.base_net.mean(),
                         changed_before=real.base_net[m].mean() if m.any() else np.nan, changed_after=0.0))
    m = real.interval_h <= 1
    aft = real.base_net.where(~m, 0.0).mean()
    rows.append(dict(set="REAL (closed + open marks)", rule="ENTRY skip hourly at entry", N=len(real), changed=int(m.sum()),
                     avg_before=real.base_net.mean(), avg_after=aft, delta=aft - real.base_net.mean(),
                     changed_before=real.base_net[m].mean() if m.any() else np.nan, changed_after=0.0))
    B = bt_hourly()
    B.to_pickle(OUT + "_data/bt_hourly.pkl")
    old = pd.read_pickle(os.path.join(R, "reports", "FUNDING_STUDY_ALL_2026-11-21_data", "bt_FB.pkl"))
    rows += summarize(old, "BACKTEST 545 (DL 616 walk)", budget)
    rows += summarize(B, "BACKTEST 545 (new cells)", newc)
    hol = int(pd.Timestamp("2025-07-01").value // 10**6)
    for k in ("HOURLY", "SWITCH"):
        for G in (0, 2):
            c = f"{k}_G{G}_net"
            for blk, mm in (("DESIGN", B.te < hol), ("HOLDOUT", B.te >= hol)):
                rows.append(dict(set=f"BACKTEST {blk}", rule=f"{k} G{G}", N=int(mm.sum()), changed=int(B[f"{k}_G{G}_hit"][mm].sum()),
                                 avg_before=B.base_net[mm].mean(), avg_after=B[c][mm].mean(), delta=(B[c] - B.base_net)[mm].mean()))
    T = pd.DataFrame(rows)
    T.to_csv(OUT + "_data/cells.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
