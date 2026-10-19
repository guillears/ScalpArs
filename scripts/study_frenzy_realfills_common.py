#!/usr/bin/env python3
"""FRENZY REAL-FILLS study 2026-10-19 — shared code: real-fill cohort, gross price paths, one generic exit walker, fees.

Research only; read-only on code / config / caches. Pre-registered conventions (fixed before any grid cell was read):
  PATH   from the fill (opened_at, entry_price) to +12 h: ticks first (reports/backtest_cache ticks / ticks_q + aggTrades days fetched from
         data.binance.vision into the scratch dir), aggregated to 1-SECOND bars (hi / lo / first / last print); then 1m klines (caches +
         public klines fetched for this study) — flagged `n_bar` > 0 = provisional. All prices as GROSS % vs the fill price.
  WALK   one generic first-touch walker, bar semantics on both resolutions: the stop / trail line is judged BEFORE the take-profit on the
         same bar (conservative); the line uses the peak of the PREVIOUS bars only; a bar that opens through the line fills at its open.
         line_i = −SL; once the prior peak ≥ b → max(line, be); once the prior peak ≥ A → max(line, peak − G). TP may switch at a time.
  FEES   entry taker 0.045 % (the real fill price already carries the entry slip). Exit: a resting TP line = MAKER 0.018 % at the line;
         every other exit (stop, trail, time cap, BE) = TAKER 0.045 % + 0.03 % slippage. Net % of notional = f − 0.045 − fee·(1 + f/100) − slip.
"""
import os, sys, glob
import numpy as np, pandas as pd
ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
sys.path.insert(0, os.path.join(ROOT, "scripts")); os.chdir(ROOT)
import study_tick_maker_lock_common as C  # noqa: E402  (2025 caches patched in; WT._NET_BLOCKED)
WT, FX = C.WT, C.FX
SCR = "/private/tmp/claude-501/-Users-guillearslanian-Downloads-NOFA-AI/c066304e-3f8a-400a-8ab7-c1fe68ff169b/scratchpad"
RF = f"{SCR}/rf"
LIVE = "/Users/guillearslanian/Downloads/scalpars_orders_live_2026-10-19_14-54-01.csv"
SL3 = ["FRENZY_LONG", "FRENZY_LITE", "FRENZY_WILLY"]
MIN, SEC = 60_000, 1_000
TAKER, MAKER, SLIP, ENTRY = 0.045, 0.018, 0.03, 0.045
NOTIONAL = {"FRENZY_LONG": 100.0, "FRENZY_LITE": 2450.0, "FRENZY_WILLY": 2300.0}   # today's live notionals (B23 fills: AGT 1× $101,
#                                                                                   AVA 8× $2,457, ALT / DUSK 10× $2,244 / $2,324) at $1,225
ACCOUNT = 1225.0

# ───────────── data patches: our own tick days + 1m pages ─────────────
_orig_tf = WT._tick_file


def _tick_file(pair, day):
    f = _orig_tf(pair, day)
    if f is None:
        p = f"{RF}/dl/ticks/{pair}/{day}.npz"
        if os.path.exists(p):
            return p
    return f


WT._tick_file = _tick_file
_K1X = {}


def load_1m(pair):
    if pair in _K1X:
        return _K1X[pair]
    fr = [WT.load_1m(pair)]
    for f in [f"{SCR}/rid/fetch/{pair}_1m.csv", f"{RF}/dl/k1api/{pair}.csv"] + sorted(glob.glob(f"{RF}/dl/k1/{pair}/*.csv")):
        if os.path.exists(f):
            fr.append(WT._read_k1(f))
    fr = [x for x in fr if len(x)]
    k = pd.concat(fr).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    if len(_K1X) > 30:
        _K1X.clear()
    _K1X[pair] = k
    return k


# ───────────── cohort ─────────────
def real_fills():
    M = pd.read_csv("reports/MASTER_POOL_stacked.csv", low_memory=False)
    L = pd.read_csv(LIVE, low_memory=False)
    Mk = M[M.entry_strategy.isin(SL3) & (M.stack_keep.astype(str).str.lower() == "true")
           & ~M.is_probe.astype(str).str.lower().isin(["true", "1"]) & (M.status == "CLOSED")].copy()
    Mk["src"] = "master"; Mk["y_book"] = pd.to_numeric(Mk.stack_pct, errors="coerce")
    Lk = L[L.entry_strategy.isin(SL3) & (L.status == "CLOSED")].copy()
    Lk["src"] = "live"; Lk["era"] = "B23"; Lk["y_book"] = pd.to_numeric(Lk.pnl_percentage, errors="coerce")
    D = pd.concat([Mk, Lk], ignore_index=True).copy()
    D["key"] = D.opened_at.astype(str).str[:19].str.replace(" ", "T") + "|" + D.pair + "|" + D.direction
    assert not D.key.duplicated().any()
    D["sleeve"] = D.entry_strategy
    D["ts"] = pd.to_datetime(D.opened_at.astype(str).str[:19].str.replace("T", " "))
    D["te"] = D.ts.dt.as_unit("ms").astype("int64")
    D["E"] = pd.to_numeric(D.entry_price, errors="coerce")
    D["day"] = D.ts.dt.strftime("%Y-%m-%d")
    return D.sort_values("te").reset_index(drop=True)


def lite_tp(r):
    u = pd.to_numeric(r.get("frenzy_lite_tp_used"), errors="coerce")
    if np.isfinite(u) and u > 0:
        return float(u)
    vm, bs = pd.to_numeric(r.get("entry_frenzy_vol_mult"), errors="coerce"), pd.to_numeric(r.get("entry_btc_1h_slope"), errors="coerce")
    return 3.0 if (np.isfinite(vm) and np.isfinite(bs) and vm >= 20 and bs > 0) else 2.0


# ───────────── path ─────────────
def gross_path(pair, te, E, horizon=720):
    """→ dict(t: bar END ms, hi, lo, op, cl gross %, end_ms, n_tick_s (1 s bars), n_bar (1m bars), te)."""
    sav = WT.net_pct
    WT.net_pct = lambda p, e: (np.asarray(p, dtype=float) / float(e) - 1) * 100
    try:
        P = FX.load_path(pair, int(te), float(E), horizon, {}, k1m=load_1m(pair))
    finally:
        WT.net_pct = sav
    nt = P["n_tick"]
    t, x = P["t"][:nt], P["cl"][:nt]
    T, HI, LO, OP, CL = [], [], [], [], []
    if nt:
        s = (t - 1) // SEC                                   # second bucket (tick at exactly te excluded upstream)
        brk = np.flatnonzero(np.diff(s)) + 1
        st = np.r_[0, brk]; en = np.r_[brk, nt]
        T.append((s[st] + 1) * SEC); HI.append(np.maximum.reduceat(x, st)); LO.append(np.minimum.reduceat(x, st))
        OP.append(x[st]); CL.append(x[en - 1])
    if P["n_bar"]:
        T.append(P["t"][nt:]); HI.append(P["hi"][nt:]); LO.append(P["lo"][nt:]); OP.append(P["op"][nt:]); CL.append(P["cl"][nt:])
    if not T:
        return None
    o = dict(t=np.concatenate(T).astype("int64"), hi=np.concatenate(HI), lo=np.concatenate(LO), op=np.concatenate(OP), cl=np.concatenate(CL),
             end_ms=int(P["end_ms"]), n_tick_s=int(len(T[0]) if nt else 0), n_bar=int(P["n_bar"]), te=int(te), n_raw_tick=int(nt))
    o["op"] = np.where(np.isfinite(o["op"]), o["op"], o["cl"] * 0 + np.nan)
    o["pk_prev"] = np.r_[-np.inf, np.maximum.accumulate(o["hi"])[:-1]]
    return o


# ───────────── generic walker ─────────────
def walk(P, tp=np.inf, sl=3.0, b=np.inf, be=0.1, A=np.inf, G=np.inf, cap=720, tp_late=None, maker_tp=True):
    """→ (net %, why, exit minute). tp_late = (minute, tp2): from that order age the TP is tp2 (WILLY late TP)."""
    te = P["te"]; t = P["t"]
    n = int(np.searchsorted(t, te + cap * MIN, side="right"))
    if n == 0:
        return np.nan, "UNRES", np.nan
    hi, lo, op, pk = P["hi"][:n], P["lo"][:n], P["op"][:n], P["pk_prev"][:n]
    line = np.full(n, -float(sl))
    if np.isfinite(b):
        line = np.where(pk >= b, np.maximum(line, be), line)
    if np.isfinite(A):
        line = np.where(pk >= A, np.maximum(line, pk - G), line)
    if tp_late is not None:
        TPa = np.where(t[:n] - te < tp_late[0] * MIN, tp, tp_late[1])
    else:
        TPa = tp
    hl = lo <= line
    ht = hi >= TPa
    il = int(np.argmax(hl)) if hl.any() else n
    it = int(np.argmax(ht)) if ht.any() else n
    if il < n and il <= it:
        f = line[il]
        if np.isfinite(op[il]) and op[il] < f:
            f = op[il]
        why = "STOP" if line[il] <= -sl + 1e-9 else "LINE"
        return f - ENTRY - TAKER * (1 + f / 100) - SLIP, why, (t[il] - te) / MIN
    if it < n:
        f = TPa[it] if np.ndim(TPa) else TPa
        if maker_tp:
            return f - ENTRY - MAKER * (1 + f / 100), "TP", (t[it] - te) / MIN
        if np.isfinite(op[it]) and op[it] > f:
            f = op[it]
        return f - ENTRY - TAKER * (1 + f / 100), "TP", (t[it] - te) / MIN
    f = P["cl"][n - 1]
    why = "CAP" if P["end_ms"] >= te + cap * MIN - MIN else "UNRES"
    return f - ENTRY - TAKER * (1 + f / 100) - SLIP, why, (t[n - 1] - te) / MIN


def today(P, r, maker_tp=True):
    s = r["sleeve"]
    if s == "FRENZY_LONG":
        return walk(P, tp=1.5, sl=3.0, cap=720, maker_tp=maker_tp)
    if s == "FRENZY_LITE":
        return walk(P, tp=lite_tp(r), sl=3.0, cap=720, maker_tp=maker_tp)
    return walk(P, tp=1.25, sl=4.0, cap=90, tp_late=(30, 0.25), maker_tp=maker_tp)


def mfe_before(P, H=720, sl=3.0):
    """gross MFE strictly before the first bar whose low ≤ −sl (stop-first), within H minutes; plus stop flag + minutes."""
    te = P["te"]; n = int(np.searchsorted(P["t"], te + H * MIN, side="right"))
    lo, hi = P["lo"][:n], P["hi"][:n]
    s = np.flatnonzero(lo <= -sl)
    k = int(s[0]) if len(s) else n
    m = float(hi[:k].max()) if k else 0.0
    return m, bool(len(s)), ((P["t"][k] - te) / MIN if len(s) else np.nan)


def day_boot(vals_by_day, nb=4000, seed=7):
    """vals_by_day: list of per-day arrays → (mean, lo95, hi95, P(mean>0)) of the per-FILL mean under a day-block bootstrap."""
    rng = np.random.default_rng(seed)
    days = [np.asarray(v, float) for v in vals_by_day if len(v)]
    k = len(days)
    if k == 0:
        return np.nan, np.nan, np.nan, np.nan
    s = np.array([v.sum() for v in days]); c = np.array([len(v) for v in days])
    idx = rng.integers(0, k, size=(nb, k))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m > 0).mean())
