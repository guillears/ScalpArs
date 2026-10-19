#!/usr/bin/env python3
"""⚡ SURGE_LONG missed-opportunities study (operator 2026-10-19: "guarantee we are not missing many opportunities").

READ-ONLY research. Never touches the bot, its config or services/*; imports the bot's pure rules (services.surge) and re-uses the
validated tick walker (scripts/surge_bearrun_review.sec_path / run_exit) and today's SURGE exit line
(scripts/study_surge_trail_cap_walk.stop_fn(dist_today) == services.trading_engine._bullrun_exit_for(no_lock=True), asserted there).

Stages:
  feat <era>   per-pair features at every BTC bar with 30-min move ≥ 0.3 % ∧ bar volume ≥ 5× (high or not) and the 3 bars after:
               rolling 24 h quote volume (universe rank + market-volume gvol), Wilder ATR %, 30-min move, EMA13−EMA50 gap, pair 24 h high.
               → reports/backtest_cache/surge_missed/feat_<era>.pkl  (era = 2025 | 2026; cache = backtest_cache_2025 | backtest_cache)
  sim          sequential sleeve simulation (after-fill spacing, ≤ 4 per trigger, ≤ 4 open, one position per pair) for TODAY and every
               variant; walks every needed (pair, entry) once on aggTrade ticks (1m fallback, flagged) with TODAY's SURGE exit.
               → reports/SURGE_MISSED_OPPS_fills.csv (one row per variant fill), reports/SURGE_MISSED_OPPS_walks.csv,
                 reports/SURGE_MISSED_OPPS_universe.csv (per trigger × top-20 pair status)
Usage: venv/bin/python scripts/study_surge_missed_opps.py feat 2025 | feat 2026 | sim
"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "backtest_cache", "surge_missed"); os.makedirs(OUT, exist_ok=True)
BAR, MIN, DAY = 300_000, 60_000, 86_400_000
CFG = json.load(open(os.path.join(ROOT, "trading_config.json"), encoding="utf-8"))
TH = CFG.get("thresholds", CFG)
ERAS = {"2025": (os.path.join(REP, "backtest_cache_2025"), "2025-01-01", "2026-01-01"),
        "2026": (os.path.join(REP, "backtest_cache"), "2026-01-01", "2026-10-15")}
MOVE = float(TH["surge_btc_move_pct"]); VOLX = float(TH["surge_btc_vol_mult"]); GMIN = float(TH["surge_long_gvol_min"])
ATR_MIN = float(TH["surge_atr_min_pct"]); UNI = int(TH["surge_universe_size"]); SLOTS = int(TH["surge_max_slots"])
SPACING_H = float(TH["surge_trigger_spacing_hours"]); GAP_MIN = float(TH.get("surge_long_min_ema13_50_gap_pct") or 0.0)
HOLD = int(TH.get("surge_max_hold_minutes", 240))
LATENCY_S = 60          # the review / replay calibration (live fills opened 27–45 s after the bar close)
USD_PER_PCT = 22.1      # live SURGE notional ≈ $2,214 (AGT $2,124 · GTC $2,305 at 10× on ~$220 margin) → $ per 1 % of P&L
assert TH["surge_long_require_24h_high"] and TH["surge_spacing_after_fill"] and TH["surge_long_require_leader"]


def _set(s):
    return {x.strip().upper() for x in str(s or "").split(",") if x.strip()}


# ------------------------------------------------------------------------------------------------------------- features
def _btc(cache):
    d = pd.read_csv(os.path.join(cache, "k5m_full", "BTCUSDT.csv")).drop_duplicates("open_time").set_index("open_time").sort_index()
    d["qv"] = d.vol * d.c
    d["r30"] = (d.c / d.c.shift(6) - 1) * 100
    d["hi24"] = d.h.shift(1).rolling(288).max()
    d["volx"] = d.qv / d.qv.shift(1).rolling(288).median()
    d["at_hi"] = d.c >= d.hi24
    return d


def _pair_feat(args):
    path, T = args
    try:
        d = pd.read_csv(path, usecols=["open_time", "o", "h", "l", "c", "vol"]).drop_duplicates("open_time").set_index("open_time").sort_index()
    except Exception:
        return None
    if len(d) < 300:
        return None
    c, h, l = d.c, d.h, d.l
    qv = d.vol * c
    f = pd.DataFrame(index=d.index)
    f["v24"] = qv.rolling(288, min_periods=250).sum()
    f["vol"] = d.vol
    f["m48"] = d.vol.rolling(48).mean()
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    f["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100           # Wilder ATR(14) % (≈ services.surge.wilder_atr_pct)
    f["pmove"] = (c / c.shift(6) - 1) * 100
    e13 = c.ewm(span=13, adjust=False).mean(); e50 = c.ewm(span=50, adjust=False).mean()
    f["gap"] = (e13 - e50) / e50 * 100
    hi = h.shift(1).rolling(288).max()
    f["p_off_hi"] = (c / hi - 1) * 100
    f["p_r4h"] = (c / c.shift(48) - 1) * 100
    f = f[f.index.isin(T)].astype("float32")
    f["pair"] = os.path.basename(path)[:-4]
    return f.reset_index()


def feat(era):
    cache, a, b = ERAS[era]
    t0, t1 = pd.Timestamp(a).value // 10**6, pd.Timestamp(b).value // 10**6
    btc = _btc(cache)
    m = (btc.index >= t0) & (btc.index < t1) & (btc.r30 >= MOVE) & (btc.volx >= VOLX)
    cand = btc.index[m].values
    T = set()
    for t in cand:
        for k in range(0, 4):
            T.add(int(t) + k * BAR)
    T = np.array(sorted(T))
    files = [p for p in glob.glob(os.path.join(cache, "k5m_full", "*.csv")) if os.path.basename(p).endswith("USDT.csv")]
    print(f"{era}: {len(cand)} move+vol bars → {len(T)} feature bars · {len(files)} pair files", flush=True)
    parts = []
    with ProcessPoolExecutor(8) as ex:
        for n, r in enumerate(ex.map(_pair_feat, [(p, T) for p in files], chunksize=4)):
            if r is not None and len(r):
                parts.append(r)
            if n % 100 == 0:
                print(f"  {n}/{len(files)}", flush=True)
    F = pd.concat(parts, ignore_index=True)
    B = btc.loc[btc.index.isin(T), ["c", "r30", "volx", "at_hi", "hi24"]].reset_index()
    B["cand"] = B.open_time.isin(cand)
    pd.to_pickle(dict(F=F, B=B, cand=cand), os.path.join(OUT, f"feat_{era}.pkl"))
    print("written", len(F), "rows")


# ------------------------------------------------------------------------------------------------------------- universe
def universe(era):
    """per feature bar: gvol (bot's market-volume reading) + top-20 tradeable universe with every pair's SURGE status."""
    cache, _, _ = ERAS[era]
    z = pd.read_pickle(os.path.join(OUT, f"feat_{era}.pkl"))
    F, B = z["F"], z["B"]
    ex = json.load(open(os.path.join(cache, "exchange_info.json")))
    nd = float(CFG.get("new_listing_filter_days", 90) or 0)

    def coin(p):
        info = ex.get(p) or {}
        if info and info.get("underlyingType") != "COIN":
            return False
        return not any("alpha" in str(x).lower() for x in (info.get("underlyingSubType") or []))
    pairs = F.pair.unique()
    ok_coin = {p: coin(p) for p in pairs}
    onboard = {p: (ex.get(p) or {}).get("onboardDate") for p in pairs}
    skip = _set(CFG.get("pair_blacklist")) | _set(CFG.get("no_trade_pairs")) | _set(TH.get("surge_long_pair_blacklist"))
    F = F[F.pair.map(ok_coin)].copy()
    ob = F.pair.map(onboard).astype("float64")
    F["young"] = (nd > 0) & ob.notna() & (ob > F.open_time - nd * DAY)
    G = F[~F.young & np.isfinite(F.v24) & np.isfinite(F.m48) & (F.m48 > 0)]
    G = G.sort_values(["open_time", "v24"], ascending=[True, False]).groupby("open_time").head(50)
    gv = G.groupby("open_time").apply(lambda g: g.vol.sum() / g.m48.sum() if len(g) >= 30 else np.nan, include_groups=False)
    U = F[~F.young & ~F.pair.isin(skip) & np.isfinite(F.v24)]
    U = U.sort_values(["open_time", "v24"], ascending=[True, False]).groupby("open_time").head(UNI).copy()
    U["rank"] = U.groupby("open_time").cumcount() + 1
    B = B.set_index("open_time")
    B["gvol"] = gv.reindex(B.index)
    U = U.merge(B[["r30"]].rename(columns={"r30": "btc_move"}), left_on="open_time", right_index=True, how="left")
    U["mv_rank"] = U.groupby("open_time").pmove.rank(ascending=False, method="first")
    U["era"] = era
    return U, B.reset_index()


# ------------------------------------------------------------------------------------------------------------------ walk
_SR = _TW = None
def _walker():
    global _SR, _TW
    if _SR is None:
        import surge_bearrun_review as SR
        import study_surge_trail_cap_walk as TW
        TW.check_engine_parity()
        _SR, _TW = SR, TW
    return _SR, _TW


_K1 = {}
def _k1m(pair, cache):
    key = (cache, pair)
    if key not in _K1:
        if len(_K1) > 60:
            _K1.clear()
        fr = [pd.read_csv(os.path.join(cache, x, f"{pair}.csv"), usecols=["open_time", "o", "h", "l", "c"])
              for x in ("k1m", "k1m_ondemand", "k1m_spike") if os.path.exists(os.path.join(cache, x, f"{pair}.csv"))]
        _K1[key] = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time").to_numpy() if fr else None
    return _K1[key]


def walk_one(pair, t_entry, atr, era):
    """TODAY's SURGE_LONG exit (peak − 1×ATR after +1 %, initial stop min(−0.7, max(−1.5×ATR, −1.2)), no lock, 240 min) on ticks
    (every price-changing print), else on 1m candles (green O→L→H→C, red O→H→L→C). → dict(pnl, why, minutes, peak, src)."""
    SR, TW = _walker()
    cache = ERAS[era][0]
    SR.CACHE = cache
    t1 = t_entry + HOLD * MIN + 5000
    fn = TW.stop_fn(TW.dist_today, float(atr))
    sp = SR.sec_path(pair, t_entry, t1)
    if sp is not None:
        ts, px, e = sp; src = "tick"
    else:
        k = _k1m(pair, cache)
        if k is None:
            return dict(pnl=np.nan, src="none")
        k = k[(k[:, 0] >= t_entry) & (k[:, 0] <= t1)]
        if len(k) < 5 or k[0, 0] > t_entry + 2 * MIN:
            return dict(pnl=np.nan, src="none")
        e = float(k[0, 1]); ts, px = [], []
        for ot, o, h, l, c in k:
            seq = (o, l, h, c) if c >= o else (o, h, l, c)
            for j, p in enumerate(seq):
                ts.append(ot + j * 15_000); px.append(p)
        ts, px = np.array(ts[1:], dtype=np.int64), np.array(px[1:], dtype=float); src = "1m"
    pnl = (px / e - 1) * 100 - SR.FEE
    o = SR.run_exit(ts, pnl, t_entry, fn, None, HOLD)
    if o is None:
        return dict(pnl=np.nan, src="none")
    return dict(pnl=o[0], why=o[1], minutes=o[2], peak=o[3], src=src, entry_px=e)


# ------------------------------------------------------------------------------------------------------------------- sim
VARIANTS = {   # name: dict(spacing_h, need_high, extra_bars, topk, atr_min)
    "TODAY":        dict(sp=SPACING_H, high=True, wb=0, topk=0, atr=ATR_MIN),
    "SPACING_2H":   dict(sp=2.0, high=True, wb=0, topk=0, atr=ATR_MIN),
    "SPACING_1H":   dict(sp=1.0, high=True, wb=0, topk=0, atr=ATR_MIN),
    "SPACING_0":    dict(sp=0.0, high=True, wb=0, topk=0, atr=ATR_MIN),
    "TOP2_MOVERS":  dict(sp=SPACING_H, high=True, wb=0, topk=2, atr=ATR_MIN),
    "TOP3_MOVERS":  dict(sp=SPACING_H, high=True, wb=0, topk=3, atr=ATR_MIN),
    "NO_24H_HIGH":  dict(sp=SPACING_H, high=False, wb=0, topk=0, atr=ATR_MIN),
    "WINDOW_10M":   dict(sp=SPACING_H, high=True, wb=1, topk=0, atr=ATR_MIN),
    "WINDOW_15M":   dict(sp=SPACING_H, high=True, wb=2, topk=0, atr=ATR_MIN),
    "ATR_1.0":      dict(sp=SPACING_H, high=True, wb=0, topk=0, atr=1.0),   # info: the dominant refusal (not asked; priced only)
}


def _picks(rows, btc_move, v, taken):
    """rows = the bar's top-20 universe (rank order). LIVE rule: ATR ≥ min ∧ pair 30-min > BTC 30-min ∧ EMA13−50 gap ≥ 0.
    topk K: the leaders PLUS the next best positive movers (ATR / gap still required) until K picks."""
    ok = rows[(rows.atr >= v["atr"]) & (rows.gap >= GAP_MIN) & ~rows.pair.isin(taken)]
    lead = ok[ok.pmove > btc_move]
    out = [(r.pair, "LEADER") for r in lead.itertuples()]
    if v["topk"] and len(out) < v["topk"]:
        rest = ok[(ok.pmove <= btc_move) & (ok.pmove > 0)].sort_values("pmove", ascending=False)
        out += [(r.pair, "MOVER") for r in rest.head(v["topk"] - len(out)).itertuples()]
    return out


def sim():
    SR, TW = _walker()
    Us, Bs = [], []
    for era in ERAS:
        U, B = universe(era); B["era"] = era
        Us.append(U); Bs.append(B)
    U = pd.concat(Us, ignore_index=True); B = pd.concat(Bs, ignore_index=True)
    B = B.drop_duplicates("open_time").set_index("open_time").sort_index()
    byt = {t: g for t, g in U.groupby("open_time")}
    era_of = B.era.to_dict()
    wf = os.path.join(REP, "SURGE_MISSED_OPPS_walks.csv")
    W = {}
    if os.path.exists(wf):
        for r in pd.read_csv(wf).to_dict("records"):
            W[(r["pair"], int(r["t_entry"]))] = r

    def get_walk(pair, t_entry, atr, era):
        k = (pair, int(t_entry))
        if k not in W:
            W[k] = dict(pair=pair, t_entry=int(t_entry), atr=atr, era=era, **walk_one(pair, int(t_entry), atr, era))
        return W[k]

    cand = B.index[B.cand].values
    fills, uni_rows = [], []
    for name, v in VARIANTS.items():
        last_fill = -10**18
        open_pos = []   # (pair, exit_ms)
        n_trig = n_spaced = 0
        for t in cand:
            b = B.loc[t]
            if not (b.r30 >= MOVE and b.volx >= VOLX):
                continue
            if v["high"] and not bool(b.at_hi):
                continue
            g = b.gvol
            if not (np.isfinite(g) and g >= GMIN):       # refused trigger: recorded, never starts the spacing
                continue
            if t - last_fill < v["sp"] * 3600_000:
                n_spaced += 1
                if name == "TODAY":
                    uni_rows.append(dict(trig=t, era=era_of[t], status="SPACED", btc_move=b.r30, gvol=g, at_hi=bool(b.at_hi)))
                continue
            n_trig += 1
            open_pos = [(p, x) for p, x in open_pos if x > t + BAR]
            taken = {p for p, _ in open_pos}
            got = 0
            for k in range(0, v["wb"] + 1):
                bt = int(t) + k * BAR
                rows = byt.get(bt)
                if rows is None or bt not in B.index:
                    break
                bm = float(B.at[bt, "r30"])
                if name == "TODAY" and k == 0:
                    for r in rows.itertuples():
                        st = ("ATR_LOW" if not r.atr >= ATR_MIN else "NOT_LEADER" if not r.pmove > bm else
                              "EMA_GAP" if not r.gap >= GAP_MIN else "PASS")
                        uni_rows.append(dict(trig=t, era=era_of[t], status="TRIGGER", pair=r.pair, rank=r.rank, mv_rank=r.mv_rank,
                                             atr=r.atr, pmove=r.pmove, gap=r.gap, p_off_hi=r.p_off_hi, p_r4h=r.p_r4h,
                                             btc_move=bm, gvol=g, pair_status=st))
                for p, why in _picks(rows, bm, v, taken):
                    if got >= SLOTS or len(open_pos) >= SLOTS:
                        break
                    r = rows[rows.pair == p].iloc[0]
                    te = bt + BAR + LATENCY_S * 1000
                    w = get_walk(p, te, float(r.atr), era_of[t])
                    if not np.isfinite(w.get("pnl", np.nan)):
                        continue
                    got += 1; taken.add(p)
                    open_pos.append((p, te + float(w["minutes"]) * MIN))
                    fills.append(dict(variant=name, trig=int(t), era=era_of[t], entry_bar=bt, bar_k=k, pair=p, kind=why, rank=r.rank,
                                      mv_rank=r.mv_rank, atr=r.atr, pmove=r.pmove, btc_move=bm, gap=r.gap, p_off_hi=r.p_off_hi,
                                      gvol=g, at_hi=bool(b.at_hi), t_entry=te, pnl=w["pnl"], why=w["why"], minutes=w["minutes"],
                                      peak=w["peak"], src=w["src"]))
            if got:
                last_fill = t
        print(f"{name:12s} triggers {n_trig:4d} · spaced-out {n_spaced:4d} · fills {sum(1 for f in fills if f['variant'] == name)}", flush=True)
        pd.DataFrame(W.values()).to_csv(wf, index=False)
    Fl = pd.DataFrame(fills)
    Fl.to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_fills.csv"), index=False)
    pd.DataFrame(uni_rows).to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_universe.csv"), index=False)
    # the spaced-out triggers' would-be picks (TODAY rule, priced) — question 2
    sp = []
    for r in [x for x in uni_rows if x["status"] == "SPACED"]:
        t = r["trig"]; rows = byt.get(t)
        if rows is None:
            continue
        for p, why in _picks(rows, float(B.at[t, "r30"]), VARIANTS["TODAY"], set())[:SLOTS]:
            rr = rows[rows.pair == p].iloc[0]
            w = get_walk(p, t + BAR + LATENCY_S * 1000, float(rr.atr), era_of[t])
            sp.append(dict(trig=t, era=era_of[t], pair=p, atr=rr.atr, pmove=rr.pmove, gap=rr.gap, pnl=w.get("pnl"), why=w.get("why"), src=w.get("src")))
    pd.DataFrame(sp).to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_spaced.csv"), index=False)
    # every refused pair at the TODAY triggers, priced (question 3: ATR ≥ 1.0 refusals and every NOT_LEADER / EMA_GAP)
    rf = []
    for r in [x for x in uni_rows if x["status"] == "TRIGGER" and x["pair_status"] != "PASS"]:
        if r["pair_status"] == "ATR_LOW" and not r["atr"] >= 1.0:
            continue
        w = get_walk(r["pair"], r["trig"] + BAR + LATENCY_S * 1000, float(r["atr"]), r["era"])
        rf.append({**r, "pnl": w.get("pnl"), "why": w.get("why"), "src": w.get("src"), "peak": w.get("peak")})
    pd.DataFrame(rf).to_csv(os.path.join(REP, "SURGE_MISSED_OPPS_refused.csv"), index=False)
    pd.DataFrame(W.values()).to_csv(wf, index=False)
    print("done")


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "feat":
        feat(sys.argv[2])
    elif st == "sim":
        sim()
