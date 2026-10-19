#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — Parts C / D / E / F: an event-level replica of the FRENZY_LONG + FRENZY_LITE entry logic on the
REAL-FILL ERA (2026-10-02 00:18 → 2026-10-19 14:54 UTC, batch gaps excluded), driven by the engine-exact 5m walk
(study_frenzy_realfills_walk.py) at frenzy_state_vol_mult T, with variants of the LITE window / green-bar / one-judgement rules.

Frozen before any variant was read:
  ENTRY     signal = the just-closed 5m bar; order at close + 8 s; price = the first print at / after that (ticks, else the next 1m open) ×
            (1 + 0.10 %) (live ruler, FRENZY latency study); refused when > 1 % from the signal close (frenzy_max_entry_dislocation_pct).
  LONG      frenzy_long_status READY on a fresh-ON bar (the walk's code at T) → bearish-day block (last closed UTC daily BTC candle < 0 ∧ BTC
            5m EMA13 < EMA50, the engine's stamps rebuilt) → WILLY global hold (refused, re-judged as a catch-up ≤ 6 bars after the hold ends
            if the price is within 1 % of the ON close) → pair held → LONG slots (2) → pair-day cap (3 FRENZY-family entries per UTC day).
  LITE      services.frenzy.frenzy_lite_status with T and the variant's frenzy_lite_max_hours; READY / GREEN_BAR judge the stretch (variant
            green_consumes=False: only READY judges; variant judge_once=False: only a FILL judges); WILLY hold refuses WITHOUT judging (≤ 6 bars,
            then the stretch is consumed); then bearish → pair held → LITE slots (2) → pair-day cap → dislocation, each consuming the stretch.
  WILLY     the REAL WILLY fills (master kept + live) are replayed as they happened: they hold their pair and the global WILLY hold.
  EXIT/SIZE today's: LONG +1.5 / −3 / 12 h maker TP at $100 notional · LITE C (+3 if vol ≥ 20× ∧ BTC 1h EMA20 slope > 0, else +2) / −3 / 12 h
            maker TP at $2,450 (8×) — study_frenzy_realfills_common fees; paths ticks first, else 1m (provisional flag).
  Not modelled: the ≤ 25-pair shortlist, manual / WIDE positions, momentum slots, restarts / pauses inside a batch.
Usage: venv/bin/python scripts/study_frenzy_realfills_engine.py cands|price|sim [workers]
"""
import os, sys, glob
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_frenzy_realfills_common as RC  # noqa: E402
import study_frenzy_realfills_exitgrid as EG  # noqa: E402
sys.path.insert(0, RC.ROOT)
import config  # noqa: E402
from services.frenzy import frenzy_lite_status, FRENZY_LITE_JUDGES, FRENZY_LITE_READY  # noqa: E402
RF = RC.RF
TH0 = config.trading_config.thresholds
MS5, MIN, HR = 300_000, 60_000, 3_600_000
ERAS = [("B16", "2026-10-02 00:18", "2026-10-03 21:45"), ("B17", "2026-10-03 23:25", "2026-10-06 12:27"),
        ("B18", "2026-10-06 13:55", "2026-10-08 18:30"), ("B19", "2026-10-08 18:30", "2026-10-12 03:33"),
        ("B20", "2026-10-12 03:33", "2026-10-13 00:00"), ("B21", "2026-10-13 00:00", "2026-10-14 18:36"),
        ("B22", "2026-10-14 18:36", "2026-10-16 20:00"), ("B23", "2026-10-16 20:00", "2026-10-19 14:54")]
ERA_MS = [(e, int(pd.Timestamp(a).value // 1e6), int(pd.Timestamp(b).value // 1e6)) for e, a, b in ERAS]
NOTL = RC.NOTIONAL


def era_of(ms):
    for e, a, b in ERA_MS:
        if a <= ms < b:
            return e
    return None


class THX:
    def __init__(self, base, **kw):
        self._b, self._kw = base, kw

    def __getattr__(self, k):
        if k in self._kw:
            return self._kw[k]
        return getattr(self._b, k)


# ───────────── BTC stamps (bearish block, LITE C slope) ─────────────
def btc_frame():
    fr = [pd.read_csv(f) for f in ("reports/backtest_cache/k5m_full/BTCUSDT.csv",)]
    fr += [pd.read_csv(x) for x in sorted(glob.glob(f"{RF}/dl/k5/BTCUSDT/*.csv"))]
    if os.path.exists(f"{RF}/dl/k5api/BTCUSDT.csv"):
        fr.append(pd.read_csv(f"{RF}/dl/k5api/BTCUSDT.csv"))
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    d = d[d.open_time >= int(pd.Timestamp("2026-08-15").value // 1e6)].reset_index(drop=True)
    c = d.c.values
    e13 = pd.Series(c).ewm(span=13, adjust=False).mean().values; e50 = pd.Series(c).ewm(span=50, adjust=False).mean().values
    d["gap"] = (e13 - e50) / e50 * 100
    d["tclose"] = d.open_time + MS5
    day = pd.to_datetime(d.open_time, unit="ms").dt.floor("D")
    dc = d.groupby(day).agg(o=("o", "first"), c=("c", "last"), n=("o", "size"))
    dc["ret"] = (dc.c / dc.o - 1) * 100
    # last CLOSED daily candle at the bar close time
    dd = pd.to_datetime(d.tclose, unit="ms").dt.floor("D") - pd.Timedelta(days=1)
    d["d1"] = dd.map(dc.ret).values
    h = d.set_index(pd.to_datetime(d.open_time, unit="ms")).c.resample("1h").last()
    hc = h.dropna(); e20 = hc.ewm(span=20, adjust=False).mean()
    sl = (e20 - e20.shift(1))                                   # slope of the 1h EMA20, last closed bar vs the one before
    hclose = pd.Series(sl.values, index=hc.index + pd.Timedelta(hours=1))
    d["h1slope"] = hclose.reindex(pd.to_datetime(d.tclose, unit="ms"), method="ffill").values
    return d[["tclose", "gap", "d1", "h1slope"]].set_index("tclose")


# ───────────── candidates ─────────────
def load_walk():
    W = pd.concat([pd.read_pickle(f) for f in glob.glob(f"{RF}/walk/w_*.pkl")], ignore_index=True)
    W["era"] = [era_of(x) for x in W.t_close.values]
    return W[W.era.notna()].reset_index(drop=True)


def lite_signal_mask(W, hmax=96.0):
    vm = W.vol_mult.values
    return ((~W.in_state.values) & (W.above_streak.values >= 12) & (vm < W["T"].values) & (W.hours.values >= 2.0) &
            (W.hours.values <= hmax) & (W.vol24.fillna(0).values >= float(TH0.frenzy_min_volume_usd)))


def cands():
    W = load_walk()
    W.to_pickle(f"{RF}/walk_all.pkl")
    L = W[W.fresh_on & W.long_ready]
    T = W[lite_signal_mask(W) & W.bar_red.values]
    C = pd.concat([L[["pair", "t_close", "c", "vol_mult"]], T[["pair", "t_close", "c", "vol_mult"]]]).drop_duplicates(["pair", "t_close"])
    C = C.sort_values(["pair", "t_close"]).reset_index(drop=True)
    C["key"] = C.pair + "|" + C.t_close.astype(str)
    C.to_pickle(f"{RF}/cands.pkl")
    # data needs: (pair, day) with no ticks and no 1m page covering t_close .. +12 h
    need = set()
    for r in C.itertuples():
        for d in pd.date_range(pd.Timestamp(r.t_close, unit="ms").normalize(), pd.Timestamp(r.t_close + 12 * HR, unit="ms").normalize()):
            ds = f"{d:%Y-%m-%d}"
            if RC.WT._tick_file(r.pair, ds) is None:
                need.add((r.pair, ds))
    N = pd.DataFrame(sorted(need), columns=["pair", "day"])
    N.to_csv(f"{RF}/need_1m_cands.csv", index=False)
    print("walk rows", len(W), "LONG ready", len(L), "LITE signal bars", len(T), "cands", len(C), "pair-days w/o ticks", len(N))


# ───────────── pricing ─────────────
def price_one(a):
    k, pair, tc, close = a
    te = int(tc) + 8_000
    try:
        P = RC.gross_path(pair, te, float(close), 720)
    except Exception as e:  # noqa: BLE001
        return k, None, repr(e)
    if P is None or not len(P["t"]):
        return k, None, "nopath"
    p0 = P["op"][0] if np.isfinite(P["op"][0]) else P["cl"][0]
    f = (1 + p0 / 100) * 1.001                                         # first print + 0.10 % live slip
    if abs(f - 1) > 0.01 + 0.001:
        return k, dict(disloc=True, E=float(close) * f), None
    for x in ("hi", "lo", "op", "cl"):
        P[x] = ((1 + P[x] / 100) / f - 1) * 100
    P["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(P["hi"])[:-1]]
    o = dict(disloc=False, E=float(close) * f, n_bar=P["n_bar"], path_end_h=(P["end_ms"] - te) / HR)
    lx = RC.walk(P, tp=1.5, sl=3.0, cap=720)
    o.update(long_pct=lx[0], long_why=lx[1], long_min=lx[2])
    for tp in (2.0, 3.0):
        x = RC.walk(P, tp=tp, sl=3.0, cap=720)
        o[f"lite{int(tp)}_pct"], o[f"lite{int(tp)}_why"], o[f"lite{int(tp)}_min"] = x
    m, st, tst = RC.mfe_before(P, 720, 3.0)
    o.update(mfe=m, run5=m >= 5, t_stop3=tst)
    # MAE path (for Part C): min low within 6 / 24 h relative to this entry, minutes to the first −3 / −5 / −8
    for s_ in (3.0, 5.0, 8.0):
        i = np.flatnonzero(P["lo"] <= -s_)
        o[f"t_hit_m{int(s_)}"] = (P["t"][i[0]] - te) / MIN if len(i) else np.nan
    # Part-A cells at cap 720 (FIX + TRAIL + BETR + SPLIT) for the Part-C / F comparisons
    o["grid"] = np.array([EG.run_cell(P, {"sleeve": "FRENZY_LONG"}, c[2])[0] for c in GRID], dtype=float)
    o["scale"] = {f"s{s1}_t{tp}": RC.walk(P, tp=tp, sl=s1, cap=720)[0] for s1 in (3.0, 5.0, 8.0, 12.0) for tp in (1.5, 3.0, 5.0)}
    return k, o, None


GRID = [c for c in EG.cells() if c[0] in ("FIX", "TRAIL", "BETR", "SPLIT") and (c[2].get("cap", 720) == 720 or c[0] == "SPLIT")]


def price(nw):
    C = pd.read_pickle(f"{RF}/cands.pkl")
    with Pool(nw) as p:
        res = p.map(price_one, [(r.key, r.pair, r.t_close, r.c) for r in C.itertuples()], chunksize=4)
    out = {k: (o if o is not None else {"err": e}) for k, o, e in res}
    pd.to_pickle(out, f"{RF}/cands_priced.pkl")
    print("priced", len(out), "errors", sum(1 for v in out.values() if "err" in v))


# ───────────── simulation ─────────────
def real_positions():
    """every REAL FRENZY-family fill (master kept non-probe + live) with open / close ms."""
    M = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    L = pd.read_csv(RC.LIVE, low_memory=False)
    keep = lambda X: X[X.entry_strategy.isin(RC.SL3) & (X.status == "CLOSED")]
    Mk = keep(M); Mk = Mk[(Mk.stack_keep.astype(str).str.lower() == "true") & ~Mk.is_probe.astype(str).str.lower().isin(["true", "1"])]
    R = pd.concat([Mk.assign(src="master"), keep(L).assign(src="live")], ignore_index=True)
    R["t_open"] = pd.to_datetime(R.opened_at.astype(str).str[:19].str.replace("T", " ")).dt.as_unit("ms").astype("int64")
    R["t_shut"] = pd.to_datetime(R.closed_at.astype(str).str[:19].str.replace("T", " ")).dt.as_unit("ms").astype("int64")
    R["sig_close"] = (R.t_open - 1000) // MS5 * MS5
    return R[["pair", "entry_strategy", "t_open", "t_shut", "sig_close", "entry_frenzy_vol_mult", "entry_frenzy_hours", "era", "src",
              "opened_at"]].reset_index(drop=True)


def ep_of(r):
    return dict(verified=r.verified, hours=r.hours, in_state=r.in_state, above_streak=r.above_streak, vol_mult=r.vol_mult,
                last_bar_ts=r.bar_open, bar_red=r.bar_red, bar_ret_pct=r.bar_ret)


def simulate(W, PR, BT, RP, T=100.0, hmax=16.75, green_consumes=True, judge_once=True, lite_on=True, long_on=True):
    th = THX(TH0, frenzy_state_vol_mult=T, frenzy_lite_max_hours=hmax, frenzy_lite_enabled=True)
    X = W[W["T"] == T].sort_values(["t_close", "pair"])
    willy = RP[RP.entry_strategy == "FRENZY_WILLY"][["pair", "t_open", "t_shut"]].values
    opens = []                         # (pair, sleeve, t_open, t_exit)
    judged, hold_first, pending_long, pd_count, fills, log = {}, {}, {}, {}, [], []
    bt_idx = BT.index.values

    def btc_at(tc):
        j = int(np.searchsorted(bt_idx, tc, side="right")) - 1
        return BT.iloc[j] if j >= 0 else None

    def busy(pair, t):
        if any(p == pair and a <= t < b for p, a, b in willy):
            return True
        return any(p == pair and a <= t < b for p, s, a, b in opens)

    def nslots(sleeve, t):
        return sum(1 for p, s, a, b in opens if s == sleeve and a <= t < b)

    def willy_hold(t):
        return any(a <= t < b for p, a, b in willy)

    def try_open(r, sleeve, why_prefix):
        te = int(r.t_close) + 8_000
        pr = PR.get(f"{r.pair}|{int(r.t_close)}")
        b = btc_at(r.t_close)
        if b is not None and np.isfinite(b.d1) and np.isfinite(b.gap) and b.d1 < 0 and b.gap < 0:
            return "BEARISH_DAY"
        if busy(r.pair, te):
            return "PAIR_HELD"
        if nslots(sleeve, te) >= 2:
            return "SLOTS"
        dkey = (r.pair, pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"))
        if pd_count.get(dkey, 0) >= 3:
            return "PAIR_DAY_CAP"
        if pr is None or "err" in pr:
            return "UNPRICED"
        if pr.get("disloc"):
            return "DISLOCATION"
        if sleeve == "FRENZY_LONG":
            pct, why, mn = pr["long_pct"], pr["long_why"], pr["long_min"]; tp = 1.5
        else:
            slope = b.h1slope if b is not None else np.nan
            tp = 3 if (np.isfinite(r.vol_mult) and r.vol_mult >= 20 and np.isfinite(slope) and slope > 0) else 2
            pct, why, mn = pr[f"lite{tp}_pct"], pr[f"lite{tp}_why"], pr[f"lite{tp}_min"]
        opens.append((r.pair, sleeve, te, te + int((mn if np.isfinite(mn) else 720) * MIN)))
        pd_count[dkey] = pd_count.get(dkey, 0) + 1
        fills.append(dict(pair=r.pair, sleeve=sleeve, t_close=int(r.t_close), te=te, era=r.era, spike_ts=int(r.spike_ts), hours=r.hours,
                          vol_mult=r.vol_mult, vs_vwap=r.vs_vwap, tp=tp, pct=pct, why=why, exit_min=mn, mfe=pr["mfe"], run5=pr["run5"],
                          n_bar=pr["n_bar"], usd=pct / 100 * NOTL[sleeve], how=why_prefix))
        return "OPEN"

    for r in X.itertuples():
        t = int(r.t_close)
        # LONG catch-up after a WILLY hold (≤ 6 bars, price within 1 %)
        if long_on and r.pair in pending_long and not willy_hold(t + 8_000):
            on_t, on_c = pending_long.pop(r.pair)
            if t - on_t <= 6 * MS5 and abs(r.c / on_c - 1) <= 0.01 and r.in_state:
                res = try_open(r, "FRENZY_LONG", "CATCHUP")
                log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LONG", code="CATCHUP_" + res, hours=r.hours, vol_mult=r.vol_mult))
        if long_on and r.fresh_on:
            if r.long_ready:
                if willy_hold(t + 8_000):
                    pending_long[r.pair] = (t, r.c); res = "WILLY_HOLD"
                else:
                    res = try_open(r, "FRENZY_LONG", "ON")
            else:
                res = r.long_code
            log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LONG", code=res, hours=r.hours, vol_mult=r.vol_mult, spike_ts=int(r.spike_ts)))
        if lite_on and not r.in_state:
            ep = ep_of(r)
            ready, code, _ = frenzy_lite_status(ep, th, r.vol24, judged.get(r.pair))
            if code not in FRENZY_LITE_JUDGES:
                continue
            sid = int(r.bar_open) - (int(r.above_streak) - 1) * MS5
            if ready and willy_hold(t + 8_000):
                hf = hold_first.get(r.pair)
                if hf is None or hf[0] != sid:
                    hold_first[r.pair] = (sid, t)
                elif t - hf[1] > 6 * MS5:
                    judged[r.pair] = sid; log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="HOLD_EXPIRED", hours=r.hours,
                                                          vol_mult=r.vol_mult, spike_ts=int(r.spike_ts)))
                    continue
                log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code="WILLY_HOLD", hours=r.hours, vol_mult=r.vol_mult,
                                spike_ts=int(r.spike_ts)))
                continue
            if code == FRENZY_LITE_READY:
                res = try_open(r, "FRENZY_LITE", "LITE")
                if judge_once or res == "OPEN":
                    judged[r.pair] = sid
            else:   # GREEN_BAR
                res = code
                if green_consumes and judge_once:
                    judged[r.pair] = sid
            log.append(dict(pair=r.pair, t_close=t, sleeve="FRENZY_LITE", code=res, hours=r.hours, vol_mult=r.vol_mult, spike_ts=int(r.spike_ts),
                            stretch=sid))
    return pd.DataFrame(fills), pd.DataFrame(log)


VARIANTS = [dict(name="T100 (today)", T=100.0), dict(name="T85", T=85.0), dict(name="T70", T=70.0), dict(name="T50", T=50.0)] + \
    [dict(name=f"T100 LITE window {h} h", T=100.0, hmax=float(h)) for h in (24, 36, 48, 72, 96)] + \
    [dict(name="T100 LITE green bar does not consume", T=100.0, green_consumes=False),
     dict(name="T100 LITE retry until filled", T=100.0, judge_once=False),
     dict(name="T70 LITE window 24 h", T=70.0, hmax=24.0)]


def sim_all():
    W = pd.read_pickle(f"{RF}/walk_all.pkl"); PR = pd.read_pickle(f"{RF}/cands_priced.pkl"); BT = btc_frame(); RP = real_positions()
    allf, alll = [], []
    for v in VARIANTS:
        kw = {k: v[k] for k in ("T", "hmax", "green_consumes", "judge_once") if k in v}
        F, Lg = simulate(W, PR, BT, RP, **kw)
        F["variant"] = v["name"]; Lg["variant"] = v["name"]
        allf.append(F); alll.append(Lg)
        print(v["name"], len(F), F.groupby("sleeve").pct.agg(["size", "mean"]).round(3).to_dict() if len(F) else {}, flush=True)
    pd.concat(allf, ignore_index=True).to_pickle(f"{RF}/sim_fills.pkl")
    pd.concat(alll, ignore_index=True).to_pickle(f"{RF}/sim_log.pkl")


if __name__ == "__main__":
    step = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    if step == "cands":
        cands()
    elif step == "price":
        price(nw)
    elif step == "sim":
        sim_all()
