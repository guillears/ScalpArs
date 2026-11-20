#!/usr/bin/env python3
"""FAST_TP (PREREG reports/FAST_TP_2026-11-20_PREREG.md) — REAL FILLS. Research only.
Real fills of the five sleeves: master (reports/MASTER_POOL_stacked.csv, stack_keep, full size: is_probe False / no *_PROBE) per batch
(era) + the CURRENT batch (newest export ~/Downloads/scalpars_orders_paper_2026-11-20_13-22-26.csv). Each fill priced on 1m klines
from its own entry (data.binance.vision zips; days not yet published → fapi /fapi/v1/klines, ≤ 1,200 weight/min, used-weight read,
stop on 418 / 429).
REF = as traded (pnl_percentage, closed_at), except FRENZY_WILLY_HT master fills which traded the OLD +2 exit → re-priced on 1m at
today's HT exit (TP 5 maker / SL 4 / 24 h) as REF (as-traded kept beside). The 33 cells are overlaid exactly as in the backtest:
trigger bar strictly before REF's exit bar; the entry-minute bar is EXCLUDED (it holds prices from before the fill — conservative);
sensitivity with it included is kept (…_b0 columns).
Output reports/FAST_TP_2026-11-20_real.csv (one row per fill, nets per cell) + _real_paths.pkl.
Usage: venv/bin/python -I scripts/fast_tp_real.py"""
import json, math, os, sys, time, urllib.request
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fast_tp_sim as S   # noqa: E402
import fast_tp_data as FD  # noqa: E402
RP = S.RP; OUT = os.path.join(RP, "FAST_TP_2026-11-20")
EXPORT = os.path.expanduser("~/Downloads/scalpars_orders_paper_2026-11-20_13-22-26.csv")
SLEEVES = {"EXHAUSTION_SHORT": "EXH", "DISTRIBUTION_SHORT": "DIST", "FLATBASE_SHORT": "FLATBASE", "FRENZY_WILLY": "WILLY", "FRENZY_WILLY_HT": "HT"}
MIN, H1, DAY = 60_000, 3_600_000, 86_400_000
FAPI = "https://fapi.binance.com/fapi/v1/klines"
UA = {"User-Agent": "research-client/1.0"}
_W = {"used": 0}
_FC = {}


def fapi_1m(p, t0, t1):
    """1m bars [t0, t1) from fapi (1,000 per call, weight 5); honours the used-weight header."""
    out = []; t = t0
    while t < t1:
        key = (p, t)
        if key in _FC:
            rows = _FC[key]
        else:
            if _W["used"] > 900:
                time.sleep(61); _W["used"] = 0
            url = f"{FAPI}?symbol={p}&interval=1m&startTime={t}&limit=1000"
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                    _W["used"] = int(r.headers.get("X-MBX-USED-WEIGHT-1M", _W["used"] + 5))
                    rows = json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (418, 429, 403):
                    raise SystemExit(f"fapi HTTP {e.code} — stopping")
                raise
            _FC[key] = rows
        if not rows:
            break
        out += rows
        t = int(rows[-1][0]) + MIN
        if len(rows) < 1000:
            break
    a = np.array([[float(x) for x in r_[:5]] for r_ in out if t0 <= int(r_[0]) < t1])
    return a


def bars(p, t0, t1):
    a = FD.get1m(p, t0, t1)
    if a is not None and len(a[0]) >= (t1 - t0) // MIN - 3:
        return a
    need = [(p, ds) for ds in FD.days_of(t0, t1) if pd.Timestamp(ds) < pd.Timestamp("2026-11-19")]
    if need:
        FD.fetch(need)
        a = FD.get1m(p, t0, t1)
        if a is not None and len(a[0]) >= (t1 - t0) // MIN - 3:
            return a
    now = int(time.time() * 1000)
    b = fapi_1m(p, t0, min(t1, now))
    if not len(b):
        return None
    return b[:, 0].astype(np.int64), b[:, 1], b[:, 2], b[:, 3], b[:, 4]


def fills():
    m = pd.read_csv(os.path.join(RP, "MASTER_POOL_stacked.csv"), low_memory=False)
    m = m[m.entry_strategy.isin(SLEEVES) & (m.stack_keep == True) & (m.is_probe != True) & (m.status == "CLOSED")].copy()  # noqa: E712
    m["src"] = "master"; m["batch"] = m.era.astype(str)
    e = pd.read_csv(EXPORT, low_memory=False)
    e = e[e.entry_strategy.isin(SLEEVES) & (e.status == "CLOSED") & ~e.entry_strategy.str.endswith("_PROBE")].copy()
    if "is_probe" in e:
        e = e[e.is_probe != True]  # noqa: E712
    e["src"] = "batch"; e["batch"] = "CURRENT(11-18→11-20)"
    cols = ["src", "batch", "pair", "entry_strategy", "direction", "opened_at", "closed_at", "entry_price", "exit_price", "pnl_percentage",
            "close_reason", "notional_value"]
    X = pd.concat([m[cols], e[cols]], ignore_index=True)
    X["open_ms"] = (pd.to_datetime(X.opened_at, format="mixed") - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    X["close_ms"] = (pd.to_datetime(X.closed_at, format="mixed") - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    X = X.drop_duplicates(["opened_at", "pair", "direction"]).reset_index(drop=True)
    return X


def one(rw):
    sl = SLEEVES[rw.entry_strategy]; d = -1 if rw.direction == "SHORT" else 1
    b0 = rw.open_ms // MIN * MIN
    hold = {"EXH": 24 * 60, "DIST": 24 * 60, "FLATBASE": 72 * 60, "WILLY": 90, "HT": 24 * 60}[sl]
    t1 = max(rw.close_ms, b0 + 31 * MIN, b0 + (hold + 2) * MIN if sl == "HT" else 0) + 2 * MIN
    a = bars(rw.pair, b0, t1)
    if a is None or a[0][0] != b0:
        return None
    t, o, h, l, c = a
    E = float(rw.entry_price); tick = S.tick_for(rw.pair, E)
    out = dict(rw._asdict()) if hasattr(rw, "_asdict") else rw.to_dict()
    out.update(sleeve=sl, dirn=d, asis_net=float(rw.pnl_percentage), asis_idx=int((rw.close_ms - b0) // MIN))
    # REF
    if sl == "HT" and rw.src == "master":
        y = S.willy_ref(t, o, h, l, c, b0, E / 1.001, None, None, S.HT_CUR, tick)
        fr = -S.fund_short(rw.pair, b0, int(t[y[1]]) + MIN) if False else 0.0
        out.update(ref_net=float(y[0]), ref_idx=int(y[1]), ref_kind={"FRENZY_TP": "TP", "STOP_LOSS": "SL"}.get(y[2], "TIME"), ref_tp=5.0,
                   ref_src="re-priced at today's HT exit (TP5/SL4/24h)")
    else:
        cr = str(rw.close_reason)
        kind = "TP" if "TP" in cr and "LATE" not in cr else ("SL" if "STOP" in cr else "OTHER")
        tp = {"EXH": np.nan, "DIST": np.nan, "FLATBASE": 5.0, "WILLY": 1.25, "HT": 5.0}[sl]
        if kind == "TP" and np.isnan(tp):
            tp = abs(float(rw.exit_price) / E - 1) * 100
        out.update(ref_net=float(rw.pnl_percentage), ref_idx=int((rw.close_ms - b0) // MIN), ref_kind=kind, ref_tp=tp, ref_src="as traded")
    # path stats
    n = min(S.NWIN, len(t))
    fav_hi = ((1 - l / E) if d < 0 else (h / E - 1)) * 100
    upto = min(out["ref_idx"] + 1, len(t))
    pk = int(np.argmax(fav_hi[:upto])) if upto > 0 else 0
    out.update(peak_fav=float(fav_hi[:upto].max()) if upto else np.nan, min_to_peak=float((t[pk] - rw.open_ms) / MIN + 0.5))
    out["cpath"] = (((1 - c[:360] / E) if d < 0 else (c[:360] / E - 1)) * 100).astype(np.float32)
    for skip0 in (1, 0):
        suf = "" if skip0 else "_b0"
        nets = []; fires = []
        for nm in S.CELLS:
            ridx = out["ref_idx"]; fire = -1; nv = out["ref_net"]
            if nm.startswith("F_"):
                X = float(nm.split("_")[1][1:]); N = int(nm.split("_")[2][1:])
                if d < 0:
                    lvl = E * (1 - X / 100); hit = l[:min(N + 1, n)] <= lvl - tick; lv_net = X - S.TAKER - S.MAKER
                else:
                    lvl = math.ceil(S.price_for_net(E, X, S.MAKER) / tick - 1e-9) * tick; hit = h[:min(N + 1, n)] >= lvl + tick
                    lv_net = (lvl / E - 1) * 100 - S.TAKER - S.MAKER * lvl / E
                # bars whose start lies within N min of the fill (bar 0 = the entry minute)
                okb = (t[:len(hit)] < rw.open_ms + N * MIN)
                hit = hit & okb
                if skip0:
                    hit[0] = False
                if hit.any():
                    j = int(np.argmax(hit))
                    if j < ridx or (j == ridx and out["ref_kind"] == "TP" and out["ref_tp"] > X + 1e-9):
                        fire, nv = j, lv_net
            else:
                Rr = float(nm.split("_")[1][1:])
                mins = (t[:n] + MIN - rw.open_ms) / MIN
                ex = ((1 - l[:n] / E) if d < 0 else (h[:n] / E - 1)) * 100
                if skip0:
                    ex = ex.copy(); ex[0] = -1e9
                mfe = np.maximum.accumulate(ex)
                ok = mfe / np.maximum(mins, 0.1) >= Rr
                if skip0:
                    ok[0] = False
                if ok.any():
                    j = int(np.argmax(ok))
                    if j < ridx:
                        cf = ((1 - c[j] / E) if d < 0 else (c[j] / E - 1)) * 100
                        fire, nv = j, cf - 2 * S.TAKER - 0.10
            nets.append(nv); fires.append(fire)
        out["nets" + suf] = np.array(nets, np.float32); out["fire" + suf] = np.array(fires, np.int16)
    return out


def main():
    X = fills()
    print("fills", len(X), X.groupby(["entry_strategy", "src"]).size().to_dict(), flush=True)
    import exh_exit2_sim  # noqa: F401
    res = []
    for rw in X.itertuples(index=False):
        try:
            r = one(rw)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print("ERR", rw.pair, rw.opened_at, repr(e), flush=True); r = None
        if r is None:
            print("no path", rw.pair, rw.opened_at, flush=True); continue
        res.append(r)
    Z = pd.DataFrame(res)
    Z.to_pickle(OUT + "_real.pkl")
    flat = Z.drop(columns=["cpath", "nets", "fire", "nets_b0", "fire_b0"]).copy()
    for i, c in enumerate(S.CELLS):
        flat["net_" + c] = [v[i] for v in Z.nets]; flat["fire_" + c] = [v[i] for v in Z.fire]
    flat.to_csv(OUT + "_real.csv", index=False)
    print("priced", len(Z), "fapi weight last", _W["used"], flush=True)


if __name__ == "__main__":
    main()
