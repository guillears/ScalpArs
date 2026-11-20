#!/usr/bin/env python3
"""TP_GRID 2026-11-20 (EXHAUSTION_SHORT take-profit level) — STEP 1: the real paper fills of the current batch, re-walked.

Research only (no bot code / config change). Operator question: would a 6 / 7 / 8 % TP (instead of ≈ 10 %) raise the win rate and the
result? The current batch export holds 5 closed EXHAUSTION_SHORT fills (no probes; the master pool holds 0).

Method:
  * 1m futures klines from the Binance public futures API (fapi /klines, limit 1500 = weight 10) for 24 h after each entry, and the
    funding-rate history (fapi /fundingRate) for the same window. The IP-wide X-MBX-USED-WEIGHT-1M header is read after every call;
    the script waits for the minute to roll above 1,000 and stops on HTTP 418 / 429. Cached under
    reports/TP_GRID_2026-11-20_EXH_data/real/ (a second run reads the cache, 0 weight).
  * Each fill is walked from its REAL fill price (entry_price, which already carries the drift / walk) on the 1m bars from the entry
    minute on (the entry minute's bar is included). Per bar: the stop is checked first on the HIGH (fill = max(stop, bar open)),
    then the resting maker TP on the LOW, filled only when the low trades through the TP by ≥ 1 tick (stress: by ≥ 0.05 %).
    A bar touching both = stop. No touch in 24 h → time exit at the close of the last bar inside 24 h.
  * Costs (one size): entry taker 0.05 % · TP maker 0.02 % · stop / time taker 0.05 % + 0.10 % slip · funding = the real
    settlements between entry and the cell's own exit, at the fill notional (a short receives a positive rate).
  * Parity: the booked exit (each fill's OWN stamped TP / SL — ZKP was opened before DL 579, at TP 5 / SL 5) must reproduce the booked
    exit kind and price; the booked net (pnl ÷ notional) is shown beside the model net (booked fees 0.045 / 0.018 % and the real stop
    slip differ from the model's 0.05 / 0.02 / 0.10).
  * Grid: today's stop min(1.5 × ATR15, 6 %) (from entry_exh_atr_pct) and 24 h for every cell; TP fixed X ∈ {4 … 10} % (10 = today's
    min(20 × ATR, 10) for every fill here) + min(k × ATR, X).
Writes reports/TP_GRID_2026-11-20_EXH_real.csv (per fill × cell) and prints the tables used in the report.
Usage: venv/bin/python -I scripts/TP_GRID_2026-11-20_EXH_real.py"""
import json, os, sys, time
import numpy as np, pandas as pd
import requests

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RP = os.path.join(R, "reports")
DD = os.path.join(RP, "TP_GRID_2026-11-20_EXH_data", "real")
EXPORT = os.path.join(R, "exports_archive", "scalpars_orders_paper_2026-11-20_16-01-39.csv")
FAPI = "https://fapi.binance.com"
UA = {"User-Agent": "research-client/1.0"}
MIN, H1 = 60_000, 3_600_000
TAKER, MAKER, SLIP = 0.05, 0.02, 0.10
SIZE = 0.61


def get(path, params):
    for _ in range(3):
        r = requests.get(FAPI + path, params=params, headers=UA, timeout=20)
        if r.status_code in (418, 429):
            sys.exit(f"STOP: HTTP {r.status_code} from Binance ({path}) — not retrying")
        r.raise_for_status()
        w = int(r.headers.get("X-MBX-USED-WEIGHT-1M", "0"))
        print(f"  {path} {params.get('symbol')} used-weight-1m {w}", flush=True)
        if w > 1000:
            time.sleep(61 - time.time() % 60)
        return r.json()
    raise RuntimeError("no answer")


def klines(sym, t0):
    f = os.path.join(DD, f"{sym}_{t0}_1m.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    rows = []; s = t0
    end = t0 + 24 * H1 + MIN
    while s < end:
        js = get("/fapi/v1/klines", dict(symbol=sym, interval="1m", startTime=s, endTime=end, limit=1500))
        if not js:
            break
        rows += [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4])] for k in js]
        s = int(js[-1][0]) + MIN
        if len(js) < 1500:
            break
    x = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c"]).drop_duplicates("t")
    x.to_csv(f, index=False)
    return x


def funding(sym, t0):
    f = os.path.join(DD, f"{sym}_{t0}_funding.json")
    if os.path.exists(f):
        return json.load(open(f))
    js = get("/fapi/v1/fundingRate", dict(symbol=sym, startTime=t0, endTime=t0 + 25 * H1, limit=1000))
    out = [(int(j["fundingTime"]), float(j["fundingRate"])) for j in js]
    json.dump(out, open(f, "w"))
    return out


def tick_of(x):
    u = np.unique(np.r_[x.o, x.h, x.l, x.c]); d = np.diff(u); d = d[d > 1e-15]
    return float(d.min()) if len(d) else 0.0


def walk(x, T, E0, tp, sl, tick, fund, through=None):
    """short; returns (kind, exit price, exit ms, gross %, funding %)."""
    end = T + 24 * H1
    b = x[(x.t >= T // MIN * MIN) & (x.t < end)]
    S = E0 * (1 + sl / 100); P = E0 * (1 - tp / 100)
    thr = P - tick if through is None else P * (1 - through / 100)
    kind, X, xm = "TIME", float(b.c.iloc[-1]), int(b.t.iloc[-1]) + MIN
    for t, o, h, l in zip(b.t.values, b.o.values, b.h.values, b.l.values):
        if h >= S:
            kind, X, xm = "SL", max(S, o), int(t) + MIN; break
        if l <= thr:
            kind, X, xm = "TP", P, int(t) + MIN; break
    g = -(X / E0 - 1) * 100
    f = sum(rate for ft, rate in fund if T < ft <= xm) * 100   # short receives +rate
    return kind, X, xm, g, f


def net_of(kind, g, f):
    return g + f - TAKER - (MAKER if kind == "TP" else TAKER + SLIP)


def cells():
    C = [("TP4", None, 4), ("TP5", None, 5), ("TP6", None, 6), ("TP7", None, 7), ("TP8", None, 8), ("TP9", None, 9),
         ("TP10 (today)", 20, 10)]
    for k in (1.5, 2, 3):
        for cap in (6, 8, 10):
            C.append((f"min({k:g}ATR,{cap})", k, cap))
    return C


def main():
    os.makedirs(DD, exist_ok=True)
    O = pd.read_csv(EXPORT, low_memory=False)
    E = O[(O.entry_strategy == "EXHAUSTION_SHORT") & (O.status == "CLOSED")].copy()
    if "is_probe" in E:
        E = E[~E.is_probe.astype(bool)]
    E["T0"] = E.opened_at.map(lambda s: pd.Timestamp(s).value // 10**6)
    E = E.sort_values("T0")
    out = []; par = []
    for _, r in E.iterrows():
        sym = r.pair; T = int(r.T0); E0 = float(r.entry_price)
        x = klines(sym, T // MIN * MIN); fd = funding(sym, T); tick = tick_of(x)
        notional = r.quantity * E0
        booked = r.pnl / notional * 100
        bk = walk(x, T, E0, r.entry_exh_tp_pct, r.entry_exh_sl_pct, tick, fd)
        bkind = {"FRENZY_TP": "TP", "STOP_LOSS": "SL"}.get(r.close_reason, r.close_reason)
        par.append(dict(pair=sym, opened=r.opened_at[:16], tp_booked=r.entry_exh_tp_pct, sl_booked=round(r.entry_exh_sl_pct, 2),
                        booked_kind=bkind, model_kind=bk[0], booked_exit=r.exit_price, model_exit=round(bk[1], 7),
                        booked_close=str(r.closed_at)[:16], model_close=pd.to_datetime(bk[2], unit="ms").strftime("%Y-%m-%dT%H:%M"),
                        booked_fund=r.funding_fee_usd / notional * 100, model_fund=bk[4],
                        booked_net=booked, model_net=net_of(bk[0], bk[3], bk[4]),
                        booked_fee=r.total_fee / notional * 100,
                        real_stop_slip=(r.exit_price / (E0 * (1 + r.entry_exh_sl_pct / 100)) - 1) * 100 if bkind == "SL" else np.nan))
        sl_today = min(1.5 * r.entry_exh_atr_pct, 6.0)
        for nm, k, cap in cells():
            tp = cap if k is None else min(k * r.entry_exh_atr_pct, cap)
            for thr_nm, thr in (("tick", None), ("0.05", 0.05)):
                kind, X, xm, g, f = walk(x, T, E0, tp, sl_today, tick, fd, thr)
                out.append(dict(pair=sym, T=T, day=T // 86_400_000, cell=nm, through=thr_nm, tp=tp, sl=sl_today, kind=kind,
                                exit_ms=xm, hold_h=(xm - T) / H1, gross=g, fund=f, net=net_of(kind, g, f),
                                mfe24=(1 - x[(x.t >= T // MIN * MIN) & (x.t < T + 24 * H1)].l.min() / E0) * 100,
                                mae24=(x[(x.t >= T // MIN * MIN) & (x.t < T + 24 * H1)].h.max() / E0 - 1) * 100))
    P = pd.DataFrame(par); Z = pd.DataFrame(out)
    Z.to_csv(os.path.join(RP, "TP_GRID_2026-11-20_EXH_real.csv"), index=False)
    P.to_csv(os.path.join(RP, "TP_GRID_2026-11-20_EXH_real_parity.csv"), index=False)
    pd.set_option("display.width", 250)
    print(P.round(3).to_string())
    # per cell summary (tick fill), one size 0.61, calendar days of the batch window (first entry day → export day)
    d0 = int(Z.day.min()); d1 = int(pd.Timestamp("2026-11-20").value // 10**6 // 86_400_000)
    rows = []
    for (nm, thr), g in Z.groupby(["cell", "through"], sort=False):
        v = g.net.values
        r_d = pd.Series(v * SIZE / 100).groupby(g.exit_ms.values // 86_400_000).sum().reindex(range(d0, d1 + 1), fill_value=0)
        eq = np.cumprod(1 + r_d.values)
        rows.append(dict(cell=nm, through=thr, N=len(v), WR=(v > 0).mean() * 100, avg=v.mean(), sum=v.sum(),
                         dcr=(eq[-1] ** (1 / len(r_d)) - 1) * 100, maxdd=(1 - eq / np.maximum.accumulate(np.r_[1, eq])[1:]).max() * 100,
                         worst_day=r_d.min() * 100, kinds="".join(g.kind.str[0])))
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(RP, "TP_GRID_2026-11-20_EXH_real_summary.csv"), index=False)
    print(S.round(2).to_string())
    print(Z[Z.through == "tick"].pivot(index="pair", columns="cell", values="net").round(2).to_string())
    print(Z[(Z.through == "tick")].groupby("pair")[["mfe24", "mae24"]].first().round(2))


if __name__ == "__main__":
    main()
