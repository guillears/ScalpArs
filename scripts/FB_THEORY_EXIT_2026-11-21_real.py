#!/usr/bin/env python3
"""FB_THEORY_EXIT_2026-11-21 — REAL fills sim (PREREG reports/FB_THEORY_EXIT_2026-11-21_PREREG.md). Research only.
Cohort: every exports_archive/scalpars_orders_*.csv, FLATBASE_SHORT, CLOSED, no probes, operator-closed out (scout_gatekit), dedup
(opened_at[:19], pair, direction) keeping the newest export. Entry = the real fill price; walk from the first 1m bar opening after the fill
minute; costs taker 0.05 entry · maker 0.02 TP · taker 0.05 + 0.25 slip on stop / time / rule; real funding (settlement → first bar with
open ≥ it, mark-scaled; STATE_EXIT_2026-11-21_common.fund_cum). 1m klines: this study's own cache first (reports/FB_THEORY_EXIT_2026-11-21_data/
k1m — fetched here for days not complete in the STATE_EXIT cache or ≥ 2026-11-19), then the STATE_EXIT cache (read only).
Output: reports/FB_THEORY_EXIT_2026-11-21_data/real_fills.csv + real_cells.csv
Usage: venv/bin/python -I scripts/FB_THEORY_EXIT_2026-11-21_real.py [--nofetch]"""
import argparse, glob, importlib.util, json, os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "FB_THEORY_EXIT_2026-11-21_data")
MYK1, MYFUN = os.path.join(OUT, "k1m"), os.path.join(OUT, "fund")
MIN, HR, DAY, H4 = 60_000, 3_600_000, 86_400_000, 14_400_000
CUR_BATCH = "2026-11-17T17:00"
SLIP, TAKER, MAKER = 0.25, 0.05, 0.02


def _imp(name, fn):
    sp = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", fn))
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


SC = _imp("SEC", "STATE_EXIT_2026-11-21_common.py")
K = _imp("FBT", "FB_THEORY_EXIT_2026-11-21_common.py")
import scout_gatekit as GK  # noqa: E402
THEIR_K1, THEIR_FUN = SC.K1, SC.FUN
NOW = int(time.time() * 1000)


def cohort():
    fs = sorted(glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv")))
    parts = []
    want = ["opened_at", "closed_at", "pair", "direction", "entry_strategy", "entry_price", "quantity", "pnl", "close_reason", "status",
            "entry_funding_rate", "is_probe", "entry_frenzy_spike_at", "entry_frenzy_vwap", "funding_fee_usd", "notional_value"]
    for f in fs:
        try:
            hd = pd.read_csv(f, nrows=0).columns
            if "entry_strategy" not in hd:
                continue
            x = pd.read_csv(f, low_memory=False, usecols=[c for c in want if c in hd])
        except Exception as e:
            print("skip", os.path.basename(f), e); continue
        x = x[(x.entry_strategy == "FLATBASE_SHORT") & (x.status == "CLOSED")]
        if len(x):
            x["src"] = os.path.basename(f); x["mode"] = "live" if "_live_" in f else "paper"
            parts.append(x)
    X = pd.concat(parts, ignore_index=True)
    if "is_probe" in X:
        X = X[~X.is_probe.fillna(False).astype(str).str.lower().isin(["true", "1"])]
    X["k"] = X.opened_at.astype(str).str[:19]
    X = X.sort_values("src").drop_duplicates(["k", "pair", "direction"], keep="last")
    X, oc = GK.split_operator_closed(X)
    print(GK.operator_closed_line(oc))
    X["booked"] = X.pnl / (X.quantity * X.entry_price) * 100
    return X.sort_values("k").reset_index(drop=True)


def ms(s):
    return int(pd.Timestamp(str(s).replace("T", " ")[:19]).value // 10**6)


def their_complete(p, ds):
    f = os.path.join(THEIR_K1, p, ds + ".npz")
    if not os.path.exists(f):
        return False
    z = np.load(f)
    return len(z["t"]) >= 1440


def fetch(X):
    SC.K1 = MYK1; SC.FUN = MYFUN
    os.makedirs(MYK1, exist_ok=True); os.makedirs(MYFUN, exist_ok=True)
    need, months = set(), {}
    for r in X.itertuples():
        te = ms(r.opened_at); sp = ms(r.entry_frenzy_spike_at)
        a = min(te - 26 * HR, sp - HR); b = min(te + 72 * HR + 10 * MIN, NOW)
        for ds in SC.days_of(a, b):
            if ds >= "2026-11-19" or not their_complete(r.pair, ds):
                need.add((r.pair, ds))
        months.setdefault(r.pair, set()).update(pd.date_range(pd.Timestamp(a, unit="ms").normalize().replace(day=1), pd.Timestamp(b, unit="ms"),
                                                               freq="MS").strftime("%Y-%m"))
    need = sorted(x for x in need if not os.path.exists(os.path.join(MYK1, x[0], x[1] + ".npz")) or x[1] >= "2026-11-20")
    for p, ds in need:
        f = os.path.join(MYK1, p, ds + ".npz")
        if os.path.exists(f) and ds >= "2026-11-20":
            os.remove(f)                                    # own cache only: refresh the still-growing days
        SC.fetch_kday(p, ds)
    print("kline days fetched", len(need), "fapi weight seen", SC._W["used"], flush=True)
    for p, m in months.items():
        f = os.path.join(MYFUN, p + ".csv")
        if os.path.exists(f):
            x = pd.read_csv(f); x = x[~x.m.astype(str).str.startswith("cur:")]; x.to_csv(f, index=False)   # re-read the current month
        SC.fetch_funding(p, m)
    print("funding done", flush=True)


def kday(p, ds):
    for d in (MYK1, THEIR_K1):
        f = os.path.join(d, p, ds + ".npz")
        if os.path.exists(f):
            z = np.load(f)
            if len(z["t"]) and (d == MYK1 or len(z["t"]) >= 1440 or True):
                return {c: z[c] for c in ("t", "o", "h", "l", "c", "v", "qv", "tbq")}
    return None


def k1m(p, t0, t1):
    parts = [a for a in (kday(p, ds) for ds in SC.days_of(t0, t1)) if a is not None]
    if not parts:
        return None
    out = {c: np.concatenate([x[c] for x in parts]) for c in parts[0]}
    m = (out["t"] >= t0) & (out["t"] < t1)
    out = {c: v[m] for c, v in out.items()}
    _, ix = np.unique(out["t"], return_index=True)
    return {c: v[ix] for c, v in out.items()}


def funding(p):
    for d in (MYFUN, THEIR_FUN):
        f = os.path.join(d, p + ".csv")
        if os.path.exists(f):
            x = pd.read_csv(f).dropna(subset=["rate"])
            x = x[x.t > 0].drop_duplicates("t").sort_values("t")
            if len(x):
                return x.t.values.astype("int64"), x.rate.values.astype(float)
    return np.array([], "int64"), np.array([])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nofetch", action="store_true")
    A = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    X = cohort()
    print("real FB fills", len(X), flush=True)
    if not A.nofetch:
        fetch(X)
    kmed = json.load(open(os.path.join(OUT, "kmed.json")))
    CL = K.cells()
    fills, cells = [], []
    for fid, r in X.iterrows():
        p = r.pair; te = ms(r.opened_at); sp = ms(r.entry_frenzy_spike_at); E = float(r.entry_price)
        a = min(te - 26 * HR, sp - HR) // MIN * MIN
        raw = k1m(p, a, te + 72 * HR + 10 * MIN)
        if raw is None:
            print("no data", p, r.k); continue
        g = SC.grid(raw, a, min(te + 72 * HR + 10 * MIN, int(raw["t"][-1]) + MIN))
        T, o, h, l, c, v = g["t"], g["o"], g["h"], g["l"], g["c"], g["v"]
        i0 = int(np.searchsorted(T, (te // MIN) * MIN + MIN))
        ifill = int(np.searchsorted(T, (te // MIN) * MIN))
        on = int(np.searchsorted(T, sp - 5 * MIN)); bi = int(np.searchsorted(T, sp - 31 * MIN))
        tp = (h[on:ifill] + l[on:ifill] + c[on:ifill]) / 3; q = v[on:ifill] * tp
        B4 = te // H4 * H4; j4a = int(np.searchsorted(T, B4 - H4)); j4b = int(np.searchsorted(T, B4))
        an = dict(base=float(c[bi]), peak=float(h[on:ifill].max()), vwap=float((tp * q).sum() / q.sum()) if q.sum() > 0 else np.nan,
                  cfhi=float(h[j4a:j4b].max()) if j4b > j4a else np.nan)
        fs_t, fs_r = funding(p)
        F = SC.fund_cum(T, o, E, -1, fs_t, fs_r, te)
        end_t = te + 72 * HR
        cover = float(g["present"][i0:int(np.searchsorted(T, end_t))].mean())
        d = dict(fid=fid, pair=p, k=r.k, te=te, E=E, mode=r["mode"], booked=r.booked, reason=r.close_reason, entry_funding_rate=r.entry_funding_rate,
                 period="BATCH" if r.k >= CUR_BATCH else "MASTER", dist_base=(E / an["base"] - 1) * 100, peak_d=(an["peak"] / E - 1) * 100,
                 vwap_d=(an["vwap"] / E - 1) * 100, cfhi_d=(an["cfhi"] / E - 1) * 100, stamped_vwap_d=(float(r.entry_frenzy_vwap) / E - 1) * 100,
                 data_end=int(T[-1]), open_ended=bool(T[-1] + MIN < end_t), cover=cover, spike=sp)
        fills.append(d)
        for cl in CL:
            w = K.run_cell(cl, T, o, h, l, c, i0, E, end_t, F, an, SLIP, TAKER, TAKER, MAKER, 1, kmed, te)
            if w is None:
                continue
            cells.append(dict(fid=fid, cell=cl["name"], net=w["net"], why=w["why"], xt=int(T[w["j"]]) + MIN, fund=w["fund"]))
        # path for the event-study chart: gross % to the short, hourly marks for 96 h after the fill
        hh_ = np.arange(0, 97)
        ix = np.searchsorted(T, te + hh_ * HR)
        ok = ix < len(T)
        pth = np.full(len(hh_), np.nan); pth[ok] = (1 - c[np.minimum(ix[ok], len(T) - 1)] / E) * 100
        d["path"] = json.dumps([None if not np.isfinite(x) else round(float(x), 3) for x in pth])
    Fz = pd.DataFrame(fills); Cz = pd.DataFrame(cells)
    Fz.to_csv(os.path.join(OUT, "real_fills.csv"), index=False); Cz.to_csv(os.path.join(OUT, "real_cells.csv"), index=False)
    ref = Cz[Cz.cell == "REF"].merge(Fz[["fid", "pair", "k", "booked", "reason", "period"]], on="fid")
    post = ref[ref.k >= "2026-10-28"]
    print("REF vs booked (fills booked under −5 / +10, ≥ 10-28):", len(post), "· mean |gap|", round((post.net - post.booked).abs().mean(), 3),
          "· why match", int(((post.why == "TP") == post.reason.str.contains("TP")).sum()), flush=True)
    print(ref[["k", "pair", "period", "booked", "reason", "net", "why"]].round(2).to_string())
    print(Fz[["k", "pair", "dist_base", "peak_d", "vwap_d", "stamped_vwap_d", "cfhi_d", "open_ended", "cover"]].round(2).to_string())


if __name__ == "__main__":
    main()
