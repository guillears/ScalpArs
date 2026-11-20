#!/usr/bin/env python3
"""HT_WILLY 2026-11-20 (PREREG reports/HT_WILLY_2026-11-20_PREREG.md) — REAL FILLS. Research only.
Current batch HT (the 2026-11-20 13:22 paper export — env HT_WILLY_EXPORT, probes excluded) and master WILLY (MASTER_POOL_stacked.csv,
stack_keep, full size, B19–B27). Each fill re-run through ht_willy_sim.one(): replica REF (parity vs the booked exit), the HT grid, and the
same random-entry controls (K = 50). Entry = the fill minute, E = the real entry price (live = entry ÷ 1.001). H = the stamped spike high,
else the max 1m high from the spike bar to the entry minute (B19–B23 lack the stamp). Data: data.binance.vision via fast_tp_data
(≤ 11-19); 11-20 from fapi /fapi/v1/klines (weight 5 / 1,000 bars, used-weight header read, sleep at 900, stop on 403 / 418 / 429).
Current-batch draws whose exit lies past the data end are MARKED at the last close (flag kind OPEN, counted in the report).
Output reports/HT_WILLY_2026-11-20_data/real.pkl
Usage: venv/bin/python -I scripts/ht_willy_real.py"""
import json, os, sys, time, urllib.request, urllib.error
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import ht_willy_sim as W  # noqa: E402
import fast_tp_data as FD  # noqa: E402
import binance_cooldown as BC  # noqa: E402   # the shared Binance ban file (other fetchers honour it too)
RP = W.RP
EXPORT = os.environ.get("HT_WILLY_EXPORT", os.path.join(os.path.dirname(HERE), "exports_archive",
                                                   "scalpars_orders_paper_2026-11-20_13-22-26.csv"))
FAPI = "https://fapi.binance.com/fapi/v1/klines"
UA = {"User-Agent": "research-client/1.0"}
MIN, DAY = 60_000, 86_400_000
FC = os.path.join(W.D, "k1m_fapi"); os.makedirs(FC, exist_ok=True)
_W = {"used": 0, "calls": 0}
NOW = int(time.time() * 1000)


def fapi_day(p, ds):
    f = os.path.join(FC, f"{p}_{ds}.npy")
    if os.path.exists(f):
        return np.load(f)
    t0 = int(pd.Timestamp(ds).value // 10**6); t1 = min(t0 + DAY, NOW - MIN)
    out = []; t = t0
    while t < t1:
        if _W["used"] > 900:
            time.sleep(61); _W["used"] = 0
        if BC.active():
            raise SystemExit("shared Binance cooldown active — stopping")
        url = f"{FAPI}?symbol={p}&interval=1m&startTime={t}&endTime={t1 - 1}&limit=1000"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                hdr = r.headers.get("X-MBX-USED-WEIGHT-1M")
                if hdr is None:
                    raise SystemExit("no used-weight header — stopping")
                _W["used"] = int(hdr); _W["calls"] += 1
                rows = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 418, 429):
                raise SystemExit(f"fapi HTTP {e.code} — stopping")
            raise
        if not rows:
            break
        out += [[float(x) for x in r_[:5]] for r_ in rows]
        t = int(rows[-1][0]) + MIN
        if len(rows) < 1000:
            break
    a = np.array(out).reshape(-1, 5)
    a = a[a[:, 0] < NOW - MIN]   # only closed bars
    if ds < pd.Timestamp(NOW, unit="ms").strftime("%Y-%m-%d"):
        np.save(f, a)
    return a


def get1m(p, t0, t1):
    parts = []
    for ds in FD.days_of(t0, min(t1, NOW)):
        a = FD.day_arr(p, ds)
        if a is None or not len(a):
            if pd.Timestamp(ds) <= pd.Timestamp("2026-11-19"):
                FD.fetch([(p, ds)]); a = FD.day_arr(p, ds)
            if a is None or not len(a):
                a = fapi_day(p, ds)
        if a is None or not len(a):
            return None
        parts.append(a)
    a = np.concatenate(parts); a = a[(a[:, 0] >= t0) & (a[:, 0] < t1)]
    _, ix = np.unique(a[:, 0], return_index=True); a = a[ix]
    return a[:, 0].astype(np.int64), a[:, 1], a[:, 2], a[:, 3], a[:, 4]


def ms(v):
    return int((pd.to_datetime(v, format="mixed") - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1))


def rows():
    e = pd.read_csv(EXPORT, low_memory=False)
    e = e[(e.entry_strategy == "FRENZY_WILLY_HT") & (e.status == "CLOSED")].copy()
    if "is_probe" in e:
        e = e[e.is_probe != True]  # noqa: E712
    e["src"] = "CURRENT"; e["era"] = "CURRENT"
    m = pd.read_csv(os.path.join(RP, "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[(m.entry_strategy == "FRENZY_WILLY") & (m.stack_keep == True) & (m.is_probe != True) & (m.status == "CLOSED")].copy()  # noqa: E712
    m["src"] = "MASTER"
    cols = ["src", "era", "pair", "entry_strategy", "opened_at", "closed_at", "entry_price", "exit_price", "pnl_percentage", "close_reason",
            "entry_frenzy_spike_at", "entry_frenzy_willy_spike_high", "frenzy_willy_runner_k", "entry_atr_pct", "notional_value"]
    cols_m = cols + [c for c in ("stack_pct",) if c in m]
    X = pd.concat([m[cols_m], e[cols]], ignore_index=True)
    return X.reset_index(drop=True)


def main():
    W.MARK_OPEN = True
    W.FD.get1m = get1m   # route every bar read through the fapi-aware loader
    import exh_exit2_sim  # noqa: F401
    X = rows()
    print("fills", X.groupby(["src", "entry_strategy"]).size().to_dict(), flush=True)
    out = []
    for i, r in X.iterrows():
        sl = "HT" if r.entry_strategy == "FRENZY_WILLY_HT" else "WILLY"
        T = ms(r.opened_at) // MIN * MIN; sp = ms(r.entry_frenzy_spike_at)
        H = float(r.entry_frenzy_willy_spike_high) if pd.notna(r.entry_frenzy_willy_spike_high) else np.nan
        if not np.isfinite(H):
            a = get1m(r.pair, sp, T)
            H = float(a[2].max()) if a is not None and len(a[0]) else np.nan
        live = float(r.entry_price) / 1.001
        rw = pd.Series(dict(pair=r.pair, entry_ms=T, spike_ms=sp, live=live, peak=H, dip=(H - live) / H * 100,
                            run_atr=(float(r.entry_atr_pct) if pd.notna(r.frenzy_willy_runner_k) else np.nan)), name=i)
        try:
            o = W.one(sl, rw, 777 + i)
        except SystemExit:
            raise
        except Exception as ex:  # noqa: BLE001
            print("ERR", r.pair, r.opened_at, repr(ex), flush=True); o = None
        if o is None:
            print("no path", r.pair, r.opened_at, flush=True); continue
        o.update(src=r.src, era=r.era, sleeve=sl, opened_at=r.opened_at, closed_at=r.closed_at, booked=float(r.pnl_percentage),
                 booked_reason=r.close_reason, booked_x=ms(r.closed_at), stack_pct=r.get("stack_pct", np.nan), H_src=("stamp" if pd.notna(r.entry_frenzy_willy_spike_high) else "1m max"))
        out.append(o)
    Z = pd.DataFrame(out)
    Z.to_pickle(os.path.join(W.D, "real.pkl"))
    print("priced", len(Z), "fapi calls", _W["calls"], "last used weight", _W["used"], flush=True)
    print(Z[["src", "pair", "opened_at", "booked", "booked_reason", "ref_net", "ref_kind", "stack_pct"]].to_string())


if __name__ == "__main__":
    main()
