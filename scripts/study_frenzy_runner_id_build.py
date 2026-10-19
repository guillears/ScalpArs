#!/usr/bin/env python3
"""FRENZY RUNNER-ID 2026-10-19 — cohort builder: path outcome (MFE before the stop) + exits + entry/HTF features.

Research only. Read-only on code / config / caches. PRE-REGISTERED here before any outcome was read:

OUTCOME (path, net % of notional, the bot's paper accounting gross − 0.045 − 0.045·r):
  path   = ticks while consecutive covered days last, then 1m bars (scout_frenzy_willy_exit.load_path; 2025 caches via
           study_tick_maker_lock_common); REAL fills whose cache stops early: 1m public klines fetched by study_frenzy_runner_id_fetch.py
  stop-first: the first event with low <= −3 ends the run; on a 1m bar touching both, the bar's high is NOT counted
  mfe_H  = max net high strictly before the stop, within H = 120 / 360 / 720 min (censored flag when the path ends first without a stop)
  runner = mfe_720 >= +5 (sensitivity +3 / +8, and the 120 / 360 horizons) ; t_mfe = minutes to the 720-min MFE
EXITS walked on the same path (first-touch; bar = stop-first, fill at the line or at the bar open when it opens through):
  X0  today's sleeve exit: LONG +1.5 / −3 / 720 · LITE C (+3 if vol_mult >= 20 ∧ BTC 1h slope > 0 else +2) / −3 / 720 ·
      WILLY +1.25 / −4 / 90 with the late TP +0.25 from minute 30 (cap close at the last price)
  R1  half / runner: half at +1.5; the other half −3 stop until +1.5 is hit, then line = max(+0.2, peak − 3); 720 cap at the last price
  R2  pure trail: −3 stop until the peak reaches +3, then line = max(+0.2, peak − 3); 720 cap
FEATURES (all known at the entry bar): every numeric entry_* stamp + HTF rebuilt from 5m klines closed before the entry (h_*) + BTC (b_*).
Usage: venv/bin/python scripts/study_frenzy_runner_id_build.py <scratch_dir> [workers]
"""
import os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"; sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
import study_tick_maker_lock_common as C  # noqa: E402  (patches the 2025 tick / 1m caches into the walkers)
FX, WT = C.FX, C.WT
S = sys.argv[1]
SL = ["FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"]
LIVE = "/Users/guillearslanian/Downloads/scalpars_orders_live_2026-10-19_14-54-01.csv"
MIN = 60_000
HZ = (120, 360, 720)


# ───────────────────────── cohorts ─────────────────────────
def cohorts():
    B = pd.read_csv("reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv", low_memory=False)
    B = B[(B.seed == 1) & B.sleeve.isin(SL)].copy()
    B["blk"] = "BT"; B["y_book"] = B.pct_new
    M = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    L = pd.read_csv(LIVE, low_memory=False)
    Mk = M[M.entry_strategy.isin(SL) & (M.stack_keep.astype(str).str.lower() == "true")
           & ~M.is_probe.astype(str).str.lower().isin(["true", "1"]) & (M.status == "CLOSED")].copy()
    Mk["blk"] = "MASTER"; Mk["y_book"] = pd.to_numeric(Mk.stack_pct, errors="coerce")
    Lk = L[L.entry_strategy.isin(SL) & (L.status == "CLOSED")].copy()
    Lk["blk"] = "LIVE"; Lk["y_book"] = pd.to_numeric(Lk.pnl_percentage, errors="coerce"); Lk["era"] = "B23"
    R = pd.concat([Mk, Lk], ignore_index=True); R["sleeve"] = R.entry_strategy
    D = pd.concat([B, R], ignore_index=True)
    D["ts"] = pd.to_datetime(D.opened_at.astype(str).str[:19].str.replace("T", " "))
    D["te"] = D.ts.dt.as_unit("ms").astype("int64")
    D["E"] = pd.to_numeric(D.entry_price, errors="coerce")
    key = D.blk.where(D.blk == "BT", "REAL") + D.opened_at.astype(str).str[:19] + D.pair + D.sleeve
    assert not key.duplicated().any()
    return D.reset_index(drop=True)


# ───────────────────────── path walkers ─────────────────────────
def mfe_before_stop(P, H, sl=3.0):
    te = P["te"]; n = int(np.searchsorted(P["t"], te + H * MIN, side="right"))
    hi, lo, bar = P["hi"][:n], P["lo"][:n], P["bar"][:n]
    st = np.flatnonzero(lo <= -sl)
    k = int(st[0]) if len(st) else n
    pre = hi[:k]
    m = float(pre.max()) if len(pre) else -0.09          # stopped on the first event: MFE = the entry (net of the round trip)
    tm = float((P["t"][int(np.argmax(pre))] - te) / MIN) if len(pre) else np.nan
    censored = (not len(st)) and (P["end_ms"] < te + H * MIN - MIN)
    return m, tm, bool(len(st)), bool(censored), (float((P["t"][k] - te) / MIN) if len(st) else np.nan)


def walk(P, mode, tp=None, sl=3.0, cap=720, late=None):
    """first-touch walker on the net path; returns the net % of the whole position (half/half averaged for R1)."""
    te = P["te"]; n = int(np.searchsorted(P["t"], te + cap * MIN, side="right"))
    t, hi, lo, cl, op, bar = P["t"][:n], P["hi"][:n], P["lo"][:n], P["cl"][:n], P["op"][:n], P["bar"][:n]
    if n == 0:
        return np.nan, "UNRES"
    if mode == "FIX":
        for i in range(n):
            tpi = tp if (late is None or (t[i] - te) < late[0] * MIN) else late[1]
            if lo[i] <= -sl:
                f = op[i] if (bar[i] and np.isfinite(op[i]) and op[i] < -sl) else -sl
                return float(f), "STOP"
            if hi[i] >= tpi:
                f = op[i] if (bar[i] and np.isfinite(op[i]) and op[i] > tpi) else tpi
                return float(f), "TP"
        why = "CAP" if P["end_ms"] >= te + cap * MIN - MIN else "UNRES"
        return float(cl[-1]), why
    if mode in ("R1", "R2"):
        arm = 1.5 if mode == "R1" else 3.0
        half = None; armed = False; pk = -1e9
        for i in range(n):
            line = max(0.2, pk - 3.0) if armed else -sl
            if lo[i] <= line:
                f = op[i] if (bar[i] and np.isfinite(op[i]) and op[i] < line) else line
                rest = float(f)
                if mode == "R1":
                    return (0.5 * (half if half is not None else rest) + 0.5 * rest), "LINE"
                return rest, "LINE"
            if mode == "R1" and half is None and hi[i] >= 1.5:
                half = 1.5
            pk = max(pk, hi[i])
            if pk >= arm:
                armed = True
        why = "CAP" if P["end_ms"] >= te + cap * MIN - MIN else "UNRES"
        last = float(cl[-1])
        if mode == "R1":
            return 0.5 * (half if half is not None else last) + 0.5 * last, why
        return last, why
    raise ValueError(mode)


def lite_tp(r):
    vm, bs = pd.to_numeric(r.get("entry_frenzy_vol_mult"), errors="coerce"), pd.to_numeric(r.get("entry_btc_1h_slope"), errors="coerce")
    return 3.0 if (np.isfinite(vm) and np.isfinite(bs) and vm >= 20 and bs > 0) else 2.0


_FK = {}


def fetched_1m(pair):
    if pair not in _FK:
        f = f"{S}/fetch/{pair}_1m.csv"
        _FK[pair] = pd.read_csv(f) if os.path.exists(f) else None
    return _FK[pair]


def one(args):
    i, r = args
    tc = {}
    try:
        k1 = None
        if r["blk"] != "BT":
            fk = fetched_1m(r["pair"])
            if fk is not None:
                base = WT.load_1m(r["pair"])
                k1 = pd.concat([base, fk]).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
        P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), 720, tc, k1m=k1)
    except Exception as e:  # noqa: BLE001
        return dict(i=i, err=repr(e))
    o = dict(i=i, n_tick=P["n_tick"], n_bar=P["n_bar"], path_end_min=(P["end_ms"] - P["te"]) / MIN)
    for H in HZ:
        m, tm, st, cen, tst = mfe_before_stop(P, H)
        o[f"mfe_{H}"] = m; o[f"tmfe_{H}"] = tm; o[f"stop_{H}"] = st; o[f"cens_{H}"] = cen
        if H == 720:
            o["t_stop"] = tst
    sv = r["sleeve"]
    if sv == "FRENZY_LONG":
        o["x0"], o["x0_why"] = walk(P, "FIX", tp=1.5, sl=3.0, cap=720)
    elif sv == "FRENZY_LITE":
        o["x0"], o["x0_why"] = walk(P, "FIX", tp=lite_tp(r), sl=3.0, cap=720)
    else:
        o["x0"], o["x0_why"] = walk(P, "FIX", tp=1.25, sl=4.0, cap=90, late=(30, 0.25))
    o["x15"], _ = walk(P, "FIX", tp=1.5, sl=3.0, cap=720)          # common +1.5/−3 ruler (parity vs pct_new for LONG)
    o["x3"], _ = walk(P, "FIX", tp=3.0, sl=3.0, cap=720)           # +3/−3 (parity vs pct_new for BT LITE)
    o["r1"], o["r1_why"] = walk(P, "R1")
    o["r2"], o["r2_why"] = walk(P, "R2")
    return o


# ───────────────────────── HTF features (5m klines closed before the entry) ─────────────────────────
_K5 = {}


def k5(pair):
    if pair in _K5:
        return _K5[pair]
    fr = []
    for c in ("reports/backtest_cache_2025/k5m_full", "reports/backtest_cache/k5m_full", f"{S}/fetch"):
        f = f"{c}/{pair}.csv" if not c.endswith("fetch") else f"{c}/{pair}_5m.csv"
        if os.path.exists(f):
            fr.append(pd.read_csv(f, usecols=["open_time", "o", "h", "l", "c", "qvol"]))
    d = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True) if fr else None
    if len(_K5) > 40:
        _K5.clear()
    _K5[pair] = d
    return d


def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(x, n=14):
    d = np.diff(x); up = np.where(d > 0, d, 0.0); dn = np.where(d < 0, -d, 0.0)
    au = pd.Series(up).ewm(alpha=1 / n, adjust=False).mean().values; ad = pd.Series(dn).ewm(alpha=1 / n, adjust=False).mean().values
    rs = au[-1] / ad[-1] if ad[-1] > 0 else np.inf
    return 100 - 100 / (1 + rs)


def htf(pair, te, pre="h_"):
    d = k5(pair)
    out = {}
    if d is None:
        return out
    j = int(np.searchsorted(d.open_time.values, te - 300_000, side="right"))   # bars with open + 5m <= te
    w = d.iloc[max(0, j - 2016 * 2):j]
    if len(w) < 2016 or int(w.open_time.values[-1]) + 300_000 < te - 15 * 60_000:   # stale cache (ends before the entry) → unscored
        return out
    c, h, l, ot = w.c.values, w.h.values, w.l.values, w.open_time.values
    now = c[-1]
    out[pre + "ret24"] = (now / c[-289] - 1) * 100
    out[pre + "ret72"] = (now / c[-865] - 1) * 100
    out[pre + "ret7d"] = (now / c[-2017] - 1) * 100 if len(c) > 2017 else np.nan
    out[pre + "off24h_high"] = (now / h[-288:].max() - 1) * 100
    lo7 = l[-2016:]; k = int(np.argmin(lo7))
    out[pre + "hrs_since_7d_low"] = (len(lo7) - 1 - k) * 5 / 60
    out[pre + "up_from_7d_low"] = (now / lo7[k] - 1) * 100
    out[pre + "range24"] = (h[-288:].max() / l[-288:].min() - 1) * 100
    s = pd.Series(c, index=pd.to_datetime(ot, unit="ms"))
    c1h = s.resample("1h").last().dropna().values
    c4h = s.resample("4h").last().dropna().values
    cd = s.resample("1D").last().dropna()
    hd = pd.Series(h, index=s.index).resample("1D").max(); ld = pd.Series(l, index=s.index).resample("1D").min()
    if len(c1h) >= 60:
        e20, e50 = ema(c1h, 20), ema(c1h, 50)
        out[pre + "1h_rsi"] = rsi(c1h[-200:])
        out[pre + "1h_gap20_50"] = (e20[-1] / e50[-1] - 1) * 100
        out[pre + "1h_align"] = float(now > e20[-1] > e50[-1])
        out[pre + "1h_px_vs_ema20"] = (now / e20[-1] - 1) * 100
        out[pre + "1h_ema20_slope3"] = (e20[-1] / e20[-4] - 1) * 100
    if len(c4h) >= 30:
        e20, e50 = ema(c4h, 20), ema(c4h, 50)
        out[pre + "4h_gap20_50"] = (e20[-1] / e50[-1] - 1) * 100
        out[pre + "4h_align"] = float(now > e20[-1] > e50[-1])
        out[pre + "4h_rsi"] = rsi(c4h[-120:])
    # previous complete UTC day (the current day is partial)
    if len(cd) >= 3:
        pc, ph, pl = cd.iloc[-2], hd.iloc[-2], ld.iloc[-2]
        out[pre + "prevday_closepos"] = (pc - pl) / (ph - pl) if ph > pl else np.nan
        out[pre + "prevday_ret"] = (pc / cd.iloc[-3] - 1) * 100
        ups = 0
        for z in range(2, min(9, len(cd))):
            if cd.iloc[-z] > cd.iloc[-z - 1]:
                ups += 1
            else:
                break
        out[pre + "up_days_streak"] = ups
    qv = w.qvol.values
    if len(qv) >= 2016:
        out[pre + "vol24_vs_7d"] = qv[-288:].sum() / (qv[-2016:].sum() / 7)
    return out


def feat_one(args):
    i, pair, te = args
    try:
        o = htf(pair, te)
        b = htf("BTCUSDT", te, pre="b_")
        o.update({k: v for k, v in b.items() if k in ("b_ret24", "b_ret72", "b_1h_gap20_50", "b_4h_gap20_50", "b_1h_rsi", "b_4h_rsi",
                                                      "b_1h_ema20_slope3", "b_off24h_high")})
    except Exception as e:  # noqa: BLE001
        o = {"feat_err": repr(e)}
    o["i"] = i
    return o


if __name__ == "__main__":
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    D = cohorts()
    D = D.sort_values(["pair", "te"]).reset_index(drop=True)
    if os.environ.get("SAMPLE"):
        D = pd.concat([D[D.blk != "BT"], D[D.blk == "BT"].sample(int(os.environ["SAMPLE"]), random_state=1)]).reset_index(drop=True)
    jobs = [(i, dict(pair=r.pair, te=r.te, E=r.E, blk=r.blk, sleeve=r.sleeve, entry_frenzy_vol_mult=r.entry_frenzy_vol_mult,
                     entry_btc_1h_slope=r.entry_btc_1h_slope)) for i, r in D.iterrows()]
    with Pool(nw) as p:
        res = p.map(one, jobs, chunksize=20)
    Pp = pd.DataFrame(res).set_index("i")
    with Pool(nw) as p:
        fe = p.map(feat_one, [(i, r.pair, int(r.te)) for i, r in D.iterrows()], chunksize=20)
    F = pd.DataFrame(fe).set_index("i")
    out = D.join(Pp).join(F)
    out.to_pickle(f"{S}/cohort{os.environ.get('SAMPLE', '')}.pkl")
    print("done", len(out), out.groupby(["blk", "sleeve"]).size().to_dict())
