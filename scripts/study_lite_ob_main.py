#!/usr/bin/env python3
"""LITE OB study (2026-10-15) — does the ORDER BOOK around the +1 % touch separate LITE trades that then die at −3 from the ones that
reach the C take-profit? Pre-registration: reports/LITE_OB_PREREG.txt (frozen before any feature × outcome table).
Research only: no engine / config / template / test edit, no bot API, no commit. Reads reports/lite_ob/{book.pkl,k1m,k5m}.

stages:  real  → reports/lite_ob/real_trades.csv, real_minutes.csv
         emul  → reports/lite_ob/emul_signals.csv (episodes via services.frenzy.frenzy_walk on the fetched 5m klines)
         test  → reports/lite_ob/tests_*.csv + summary
Usage: venv/bin/python -W ignore scripts/study_lite_ob_main.py [real|emul|test|all]
"""
import os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "lite_ob")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
assert os.path.exists(os.path.join(REP, "LITE_OB_PREREG.txt")), "pre-register first"
import study_lite_exitopt_common as O  # noqa: E402
WT, C = O.WT, O.C
MIN, BAR = 60_000, 300_000
SL, CAP, SLIP, TOUCH = 3.0, 720, -0.20, 1.0
LIVE_CSV = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-15_21-18-32.csv")
BOOK_T0 = int(pd.Timestamp("2026-10-03 16:00", tz="UTC").value // 1_000_000)
LEVELS = ["imb_05", "imb_1", "imb_2", "lg_ask_1", "lg_bid_1", "wall_ask_dist", "lg_wall_ask_usd", "wall_ask_share", "wall_bid_dist",
          "lg_wall_bid_usd", "wall_bid_share", "spread"]
SNAPS = {"PRE": -1, "AT": 0, "P1": 1, "P2": 2, "P3": 3}
MASK = os.environ.get("MASK", "0") == "1"      # addendum D4: drop snapshots at / after the C exit minute
SUF = "_masked" if MASK else ""


# ───────────────────────── data ─────────────────────────
_K1 = {}


def k1(pair):
    if pair not in _K1:
        f = os.path.join(OUT, "k1m", f"{pair}.csv")
        _K1[pair] = pd.read_csv(f).sort_values("open_time").reset_index(drop=True) if os.path.exists(f) else None
    return _K1[pair]


_BK = None


def book():
    """pair → DataFrame indexed by minute ms with the 12 level features."""
    global _BK
    if _BK is None:
        B = pd.read_pickle(os.path.join(OUT, "book.pkl"))
        B["tm"] = B.t.values.astype("datetime64[ms]").astype("int64")
        lg = lambda x: np.log10(np.maximum(x.astype(float), 1.0))  # noqa: E731
        F = pd.DataFrame(dict(pair=B.pair, tm=B.tm, imb_05=B.ob_imb_05, imb_1=B.ob_imb_1, imb_2=B.ob_imb_2, lg_ask_1=lg(B.ob_ask_usd_1),
                              lg_bid_1=lg(B.ob_bid_usd_1), wall_ask_dist=B.ob_wall_ask_dist_pct, lg_wall_ask_usd=lg(B.ob_wall_ask_usd),
                              wall_ask_share=B.ob_wall_ask_share, wall_bid_dist=B.ob_wall_bid_dist_pct, lg_wall_bid_usd=lg(B.ob_wall_bid_usd),
                              wall_bid_share=B.ob_wall_bid_share, spread=B.ob_spread_pct, mid=B.ob_mid))
        F.loc[B.ob_ask_usd_1.isna(), "lg_ask_1"] = np.nan; F.loc[B.ob_bid_usd_1.isna(), "lg_bid_1"] = np.nan
        _BK = {p: g.set_index("tm").sort_index() for p, g in F.groupby("pair")}
    return _BK


def snap(pair, tm):
    g = book().get(pair)
    if g is None or tm not in g.index:
        return None
    return g.loc[tm]


def path(pair, te, E):
    k = k1(pair)
    if k is None:
        return None
    first = -(-int(te) // MIN) * MIN
    k = k[(k.open_time >= first) & (k.open_time < first + (CAP + 1) * MIN)]
    if not len(k) or int(k.open_time.iloc[0]) != first:
        return None
    o = k.open_time.values.astype("int64")
    g = np.flatnonzero(np.diff(o) != MIN)
    if len(g):
        k = k.iloc[:g[0] + 1]; o = o[:g[0] + 1]
    tbr = (k.tbqv / k.qv.replace(0, np.nan)).values
    return dict(o=o, hi=WT.net_pct(k.h.values, E), lo=WT.net_pct(k.l.values, E), cl=WT.net_pct(k.c.values, E), op=WT.net_pct(k.o.values, E),
                qv=k.qv.values, tbr=tbr, te=int(te))


def price(P, E, tick, tp):
    """C walk on 1m: stop first on a shared bar (LITE_AT1 convention). → dict(C, why, j_exit, k_touch, peak, peak_j)."""
    _, th, bk = O.rm_levels(E, tick, np.array([tp]))
    th, bk = float(th[0]), float(bk[0])
    n = len(P["o"])
    s = np.flatnonzero(P["lo"] <= -SL); t = np.flatnonzero(P["hi"] >= th - 1e-12)
    a = s[0] if len(s) else n; b = t[0] if len(t) else n
    if a < n and a <= b:
        why, j, v = "STOP", int(a), min(-SL, float(P["op"][a])) + SLIP
    elif b < n:
        why, j, v = "TP", int(b), bk
    elif n >= CAP:
        why, j, v = "CAP", n - 1, float(P["cl"][n - 1]) + SLIP
    else:
        why, j, v = "UNRES", n - 1, np.nan
    hi = P["hi"][:j + 1]
    it = np.flatnonzero(hi >= TOUCH)
    k = int(it[0]) if len(it) else -1
    pj = int(np.argmax(hi)) if len(hi) else -1
    return dict(C=v, why=why, j_exit=j, k_touch=k, peak=float(hi.max()) if len(hi) else np.nan, peak_j=pj, th=th)


def feats(pair, te, anchor_ms):
    """the 180 touch features (36 × PRE/AT/P1..P3) + the 24 entry features. anchor_ms = open of the touch (or peak) minute."""
    d = {}
    ent_ms = (int(te) // MIN) * MIN
    E0 = snap(pair, ent_ms); Ep = snap(pair, ent_ms - MIN)
    for f in LEVELS:
        d[f"ENT_{f}"] = E0[f] if E0 is not None else np.nan
        d[f"ENT_dP_{f}"] = (E0[f] - Ep[f]) if (E0 is not None and Ep is not None) else np.nan
    if anchor_ms is None:
        return d
    for nm, off in SNAPS.items():
        S = snap(pair, anchor_ms + off * MIN); Sp = snap(pair, anchor_ms + (off - 1) * MIN)
        for f in LEVELS:
            v = S[f] if S is not None else np.nan
            d[f"{nm}_{f}"] = v
            d[f"{nm}_dE_{f}"] = v - d[f"ENT_{f}"] if S is not None else np.nan
            d[f"{nm}_dP_{f}"] = (v - Sp[f]) if (S is not None and Sp is not None) else np.nan
    return d


_BT = None


def macro(te, anchor):
    """D2 macro: BTC move entry→touch open (1m opens), BTC 1h EMA20 3-h slope at the touch, BTC 5-bar taker-buy share before the touch."""
    global _BT
    if _BT is None:
        _BT = btc_tables()
    _, h_t, h_e = _BT
    b = k1("BTCUSDT"); o = b.open_time.values
    d = {}
    i0 = int(np.searchsorted(o, -(-int(te) // MIN) * MIN)); i1 = int(np.searchsorted(o, anchor))
    d["btc_since_entry"] = (b.o.values[i1] / b.o.values[i0] - 1) * 100 if i1 < len(o) and i0 < len(o) and o[i1] == anchor else np.nan
    i = int(np.searchsorted(h_t, anchor, side="right")) - 1
    d["btc1h_touch"] = (h_e[i] / h_e[i - 3] - 1) * 100 if i >= 3 else np.nan
    d["btc_buy5"] = b.tbqv.values[max(0, i1 - 5):i1].sum() / b.qv.values[max(0, i1 - 5):i1].sum() if i1 >= 5 else np.nan
    return d


def tier3(P, E, tick):
    """G2/G3 split: does a resting +3 / −3 walk reach +3 (independent of C)?"""
    return price(P, E, tick, 3.0)["why"] == "TP"


def tick_of(pair, E):
    t = C.ex_tick(pair)
    return float(t) if t is not None and np.isfinite(t) else float(E) * 1e-4


# ───────────────────────── stage: real trades ─────────────────────────
def real_cohort():
    M = pd.read_csv(os.path.join(REP, "MASTER_POOL_stacked.csv"), low_memory=False)
    M = M[M.entry_strategy == "FRENZY_LITE"].assign(block="MASTER")
    L = pd.read_csv(LIVE_CSV, low_memory=False)
    L = L[(L.entry_strategy == "FRENZY_LITE") & (L.status == "CLOSED")].assign(block="LIVE")
    cols = ["block", "pair", "opened_at", "closed_at", "entry_price", "exit_price", "pnl_percentage", "close_reason", "entry_frenzy_vol_mult",
            "entry_btc_1h_slope", "entry_frenzy_hours"]
    R = pd.concat([M[cols], L[cols]], ignore_index=True)
    R["te"] = pd.to_datetime(R.opened_at, utc=True).values.astype("datetime64[ms]").astype("int64")
    R["tx"] = pd.to_datetime(R.closed_at, utc=True, format="mixed").values.astype("datetime64[ms]").astype("int64")
    R["czone"] = ((R.entry_frenzy_vol_mult >= 20) & (R.entry_btc_1h_slope > 0)).astype(int)
    R["tpC"] = np.where(R.czone == 1, 3.0, 2.0)
    return R.sort_values("te").reset_index(drop=True)


def stage_real():
    R = real_cohort()
    rows, mins = [], []
    for r in R.itertuples():
        E = float(r.entry_price); tick = tick_of(r.pair, E)
        P = path(r.pair, r.te, E)
        rec = dict(block=r.block, pair=r.pair, opened=pd.Timestamp(r.te, unit="ms").strftime("%m-%d %H:%M"), te=r.te, E=E, czone=r.czone,
                   tpC=r.tpC, booked=r.pnl_percentage, as_traded=r.close_reason, vol_mult=r.entry_frenzy_vol_mult,
                   btc1h=r.entry_btc_1h_slope, hours=r.entry_frenzy_hours, day=pd.Timestamp(r.te, unit="ms").strftime("%Y-%m-%d"))
        g = book().get(r.pair)
        hold = g.loc[(g.index >= (r.te // MIN) * MIN) & (g.index <= r.tx)] if g is not None else []
        rec["book_min_hold"] = len(hold)
        if P is None:
            rec["err"] = "no 1m"; rows.append(rec); continue
        pr = price(P, E, tick, r.tpC)
        rec.update(C=pr["C"], C_why=pr["why"], C_xm=pr["j_exit"] + 1, peak=pr["peak"], touch=pr["k_touch"] >= 0)
        # as-traded window: peak before the real close
        jx = int(np.searchsorted(P["o"], r.tx, side="right")) - 1
        rec["peak_astraded"] = float(P["hi"][:jx + 1].max()) if jx >= 0 else np.nan
        ia = np.flatnonzero(P["hi"][:jx + 1] >= TOUCH)
        rec["touch_astraded"] = bool(len(ia))
        if pr["k_touch"] >= 0:
            k = pr["k_touch"]; anchor = int(P["o"][k]); rec["anchor"] = "TOUCH"
        elif len(ia):
            k = int(ia[0]); anchor = int(P["o"][k]); rec["anchor"] = "TOUCH(as-traded)"
        else:
            k = int(np.argmax(P["hi"][:max(1, jx + 1)])); anchor = int(P["o"][k]); rec["anchor"] = "PEAK(<+1)"
        rec["anchor_utc"] = pd.Timestamp(anchor, unit="ms").strftime("%m-%d %H:%M"); rec["anchor_min"] = k + 1; rec["anchor_ms"] = anchor
        rec["acted"] = bool(pr["k_touch"] >= 0 and pr["j_exit"] > pr["k_touch"])
        rec["outcome"] = ("DIE" if pr["why"] == "STOP" else "GO" if pr["why"] == "TP" else pr["why"]) if pr["k_touch"] >= 0 else \
            ("DOA" if pr["why"] == "STOP" else pr["why"])
        rec["tbr_touch"] = P["tbr"][k]; rec["tbr_pre3"] = np.nanmean(P["tbr"][max(0, k - 3):k]) if k > 0 else np.nan
        rec.update(feats(r.pair, r.te, anchor))
        rec["reach3"] = tier3(P, E, tick)
        rec["group"] = {"DIE": "G1", "GO": "G3" if rec["reach3"] else "G2"}.get(rec["outcome"], rec["outcome"])
        rec.update(macro(r.te, anchor))
        rows.append(rec)
        # per-minute chart rows: entry − 5 … exit + 5
        lo_ms, hi_ms = (r.te // MIN) * MIN - 5 * MIN, max(r.tx, int(P["o"][min(pr["j_exit"], len(P["o"]) - 1)])) + 5 * MIN
        kk = k1(r.pair); kk = kk[(kk.open_time >= lo_ms) & (kk.open_time <= hi_ms)]
        for q in kk.itertuples():
            S = snap(r.pair, int(q.open_time))
            mins.append(dict(pair=r.pair, opened=rec["opened"], minute=pd.Timestamp(q.open_time, unit="ms").strftime("%m-%d %H:%M"),
                             rel_min=(q.open_time - (r.te // MIN) * MIN) // MIN, hi=float(WT.net_pct(q.h, E)), lo=float(WT.net_pct(q.l, E)),
                             cl=float(WT.net_pct(q.c, E)), qv=q.qv, taker_buy_ratio=q.tbqv / q.qv if q.qv else np.nan,
                             is_anchor=int(q.open_time) == anchor,
                             **({f: S[f] for f in LEVELS} if S is not None else {f: np.nan for f in LEVELS})))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "real_trades.csv"), index=False)
    pd.DataFrame(mins).to_csv(os.path.join(OUT, "real_minutes.csv"), index=False)
    print(D[["block", "pair", "opened", "as_traded", "booked", "tpC", "C", "C_why", "peak", "peak_astraded", "anchor", "anchor_utc", "outcome",
             "book_min_hold"]].to_string())


# ───────────────────────── stage: emulated signals ─────────────────────────
def _load5(pair):
    f = os.path.join(OUT, "k5m", f"{pair}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f).rename(columns={"v": "vol", "qv": "qvol"})
    return d[["open_time", "o", "h", "l", "c", "vol", "qvol"]].sort_values("open_time").reset_index(drop=True).astype({"open_time": "int64"})


def _ep_job(pair):
    import study_lite_theory_common as T
    import study_lite_theory_episodes as EP
    T.load_5m = _load5; T.START = int(pd.Timestamp("2026-10-03", tz="UTC").value // 1_000_000)
    T.END = int(pd.Timestamp("2026-10-17", tz="UTC").value // 1_000_000)
    try:
        return EP.build_pair(pair)
    except Exception as e:  # noqa: BLE001
        print("ep fail", pair, e, flush=True)
        return None


def btc_tables():
    b = _load5("BTCUSDT")
    c = b.c.astype(float)
    e13 = c.ewm(span=13, adjust=False).mean(); e50 = c.ewm(span=50, adjust=False).mean()
    gap = (e13 - e50) / e50 * 100
    day = b.open_time // 86_400_000
    dc = b.groupby(day).c.last(); dret = (dc / dc.shift(1) - 1) * 100
    scan_day = (b.open_time + BAR + 12_000) // 86_400_000
    bear = pd.DataFrame(dict(T=b.open_time.values, btc_1d=dret.reindex(scan_day - 1).values, btc_gap=gap.values))
    # 1h closes (close at each hour boundary) → EMA20 → 3 h slope %
    ce = b.open_time.values + BAR
    hh = pd.Series(c.values[ce % 3_600_000 == 0], index=ce[ce % 3_600_000 == 0])
    e = hh.ewm(span=20, adjust=False).mean()
    return bear, e.index.values.astype("int64"), e.values


def stage_emul():
    import study_lite_theory_signals as S
    from services.frenzy import frenzy_bearish_block
    B0 = pd.read_pickle(os.path.join(OUT, "book.pkl"))
    pairs = sorted(set(B0.pair))
    t0 = time.time()
    with Pool(int(os.environ.get("JOBS", "8"))) as P:
        res = [r for r in P.map(_ep_job, pairs, chunksize=1) if r is not None]
    B = pd.concat(res, ignore_index=True)
    B.to_pickle(os.path.join(OUT, "bars.pkl"))
    print(f"episode bars {len(B):,} · pairs {B.pair.nunique()} · {time.time() - t0:.0f}s", flush=True)
    bear, h_t, h_e = btc_tables()
    B = B.merge(bear, on="T", how="left")
    th = S.TH
    B["bear"] = [frenzy_bearish_block(a if np.isfinite(a) else None, g if np.isfinite(g) else None, th) == "BEARISH_DAY"
                 for a, g in zip(B.btc_1d.values, B.btc_gap.values)]
    Sg = S.signals(B, check=True, **S.TODAY)
    Sg = Sg[Sg.code == "READY"].merge(B[["pair", "T", "bear", "vol_mult", "hours", "spike_ts"]], on=["pair", "T"], how="left")
    Sg = Sg[(~Sg.bear) & (Sg["T"] + BAR >= BOOK_T0)].reset_index(drop=True)
    # real fills for correspondence
    R = real_cohort()
    rows = []
    for s in Sg.itertuples():
        te = int(s.T) + BAR + 12_000
        k = k1(s.pair)
        rec = dict(pair=s.pair, T=int(s.T), te=te, signal_utc=pd.Timestamp(s.T, unit="ms").strftime("%m-%d %H:%M"), vol_mult=s.vol_mult,
                   hours=s.hours, day=pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"))
        i = int(np.searchsorted(h_t, te, side="right")) - 1
        rec["btc1h"] = (h_e[i] / h_e[i - 3] - 1) * 100 if i >= 3 else np.nan
        rec["czone"] = int(s.vol_mult >= 20 and rec["btc1h"] > 0); rec["tpC"] = 3.0 if rec["czone"] else 2.0
        m = R[(R.pair == s.pair) & ((R.te - te).abs() <= 5 * MIN)]
        rec["real_match"] = f"{m.block.iloc[0]} {pd.Timestamp(int(m.te.iloc[0]), unit='ms').strftime('%m-%d %H:%M')}" if len(m) else ""
        if k is None:
            rec["err"] = "no 1m"; rows.append(rec); continue
        first = -(-te // MIN) * MIN
        j = int(np.searchsorted(k.open_time.values, first))
        if j >= len(k) or int(k.open_time.iloc[j]) != first:
            rec["err"] = "no 1m at entry"; rows.append(rec); continue
        E = float(k.o.iloc[j]); rec["E"] = E
        P = path(s.pair, te, E)
        pr = price(P, E, tick_of(s.pair, E), rec["tpC"])
        rec.update(C=pr["C"], C_why=pr["why"], C_xm=pr["j_exit"] + 1, peak=pr["peak"], touch=pr["k_touch"] >= 0,
                   acted=bool(pr["k_touch"] >= 0 and pr["j_exit"] > pr["k_touch"]))
        rec["outcome"] = ("DIE" if pr["why"] == "STOP" else "GO" if pr["why"] == "TP" else pr["why"]) if pr["k_touch"] >= 0 else \
            ("DOA" if pr["why"] == "STOP" else pr["why"])
        anchor = int(P["o"][pr["k_touch"]]) if pr["k_touch"] >= 0 else None
        if anchor is not None:
            rec["anchor_utc"] = pd.Timestamp(anchor, unit="ms").strftime("%m-%d %H:%M"); rec["anchor_min"] = pr["k_touch"] + 1
            rec["anchor_ms"] = anchor
            kk = pr["k_touch"]
            rec["tbr_touch"] = P["tbr"][kk]; rec["tbr_pre3"] = np.nanmean(P["tbr"][max(0, kk - 3):kk]) if kk > 0 else np.nan
        rec.update(feats(s.pair, te, anchor))
        rec["reach3"] = tier3(P, E, tick_of(s.pair, E))
        rec["group"] = {"DIE": "G1", "GO": "G3" if rec["reach3"] else "G2"}.get(rec["outcome"], rec["outcome"])
        if anchor is not None:
            rec.update(macro(te, anchor))
        rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(OUT, "emul_signals.csv"), index=False)
    print(f"emulated READY non-bearish signals since book start: {len(D)} · touches {int(D.touch.fillna(False).sum())} · "
          f"with PRE/AT snapshot {int((D.touch.fillna(False) & (D.PRE_imb_1.notna() | D.AT_imb_1.notna())).sum())}")
    print(D.outcome.value_counts(dropna=False).to_string())
    print("real matches:", D.real_match.astype(bool).sum(), sorted(D.real_match[D.real_match.astype(bool)]))


# ───────────────────────── stage: tests ─────────────────────────
def rank_z(x, y):
    """Mann–Whitney: (z, two-sided p normal approx with tie correction, AUC = P(x > y)). scipy is not installed."""
    import math
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    allv = np.concatenate([x, y]); r = pd.Series(allv).rank().values
    u = r[:n1].sum() - n1 * (n1 + 1) / 2
    N = n1 + n2; _, cnt = np.unique(allv, return_counts=True)
    sd = np.sqrt(n1 * n2 / 12 * ((N + 1) - (cnt ** 3 - cnt).sum() / (N * (N - 1))))
    z = (u - n1 * n2 / 2) / sd if sd > 0 else 0.0
    return float(z), float(math.erfc(abs(z) / math.sqrt(2))), float(u / (n1 * n2))


def dayblock_perm(te, day, R, rng):
    order = np.argsort(te, kind="stable")
    d = np.asarray(day)[order]
    starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    blocks = np.split(np.arange(len(d)), starts[1:])
    out = np.empty((len(d), R), np.int64)
    for r in range(R):
        perm = rng.permutation(len(blocks))
        seq = np.concatenate([blocks[k] for k in perm])
        out[order, r] = order[seq]
    return out


def zmat(X, y):
    """vectorised rank-sum z per column (NaNs dropped per column; tie-corrected) for one label vector y (1 = DIE)."""
    out = np.full(X.shape[1], np.nan)
    for c in range(X.shape[1]):
        v = X[:, c]; ok = np.isfinite(v)
        a = v[ok & (y == 1)]; b = v[ok & (y == 0)]
        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            continue
        r = pd.Series(np.concatenate([a, b])).rank().values
        u = r[:n1].sum() - n1 * (n1 + 1) / 2
        N = n1 + n2
        _, cnt = np.unique(np.concatenate([a, b]), return_counts=True)
        sd = np.sqrt(n1 * n2 / 12 * ((N + 1) - (cnt ** 3 - cnt).sum() / (N * (N - 1))))
        out[c] = (u - n1 * n2 / 2) / sd if sd > 0 else 0.0
    return out


def family_test(D, cols, ylab, name, R=2000, seed=11):
    D = D.reset_index(drop=True)
    y = (D[ylab].values == 1).astype(int)
    X = D[cols].values.astype(float)
    z = zmat(X, y)
    rng = np.random.default_rng(seed)
    PM = dayblock_perm(D.te.values, D.day.values, R, rng)
    mx = np.empty(R)
    for r in range(R):
        zz = zmat(X, y[PM[:, r]])
        mx[r] = np.nanmax(np.abs(zz)) if np.isfinite(zz).any() else 0.0
    rows = []
    for i, c in enumerate(cols):
        a = X[y == 1, i]; b = X[y == 0, i]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        _, p, auc = rank_z(a, b)
        rows.append(dict(sample=name, feature=c, n_die=len(a), n_go=len(b), med_die=np.median(a) if len(a) else np.nan,
                         med_go=np.median(b) if len(b) else np.nan, z=z[i], p_rank=p, auc_die_gt_go=auc,
                         p_fw=float((mx >= abs(z[i])).mean()) if np.isfinite(z[i]) else np.nan))
    T = pd.DataFrame(rows)
    if "group" in D:   # addendum D1: descriptive per-group medians
        for g in ("G1", "G2", "G3"):
            sub = D[D.group == g]
            T[f"n_{g}"] = [int(np.isfinite(sub[c].values.astype(float)).sum()) for c in cols]
            T[f"med_{g}"] = [np.nanmedian(sub[c].values.astype(float)) if np.isfinite(sub[c].values.astype(float)).any() else np.nan
                             for c in cols]
    T = T.sort_values("p_rank")
    obs = np.nanmax(np.abs(z)) if np.isfinite(z).any() else np.nan
    return T, dict(sample=name, N=len(D), n_die=int(y.sum()), n_go=int((1 - y).sum()), days=D.day.nunique(), max_abs_z=obs,
                   null95=float(np.percentile(mx, 95)), fw_p=float((mx >= obs).mean()))


def d2_cols(D):
    D = D.copy()
    D["B1_imb1"] = D.PRE_imb_1
    D["B2_dask1"] = D.PRE_dE_lg_ask_1
    D["B3_askwall"] = D.PRE_wall_ask_share / D.PRE_wall_ask_dist
    lb = D.P3_lg_bid_1.fillna(D.P2_lg_bid_1).fillna(D.P1_lg_bid_1)
    D["B4_dbid_next"] = lb - D.PRE_lg_bid_1
    return D


def _pz(y, m):
    """two-proportion z: die-rate in mask m vs the rest (rows already restricted to scored)."""
    a, b = y[m], y[~m]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    p = y.mean()
    se = np.sqrt(p * (1 - p) * (1 / len(a) + 1 / len(b)))
    return (a.mean() - b.mean()) / se if se > 0 else 0.0


def d2_test(Pt, Rt, Et, R=2000, seed=13):
    BF = ["B1_imb1", "B2_dask1", "B3_askwall", "B4_dbid_next"]
    P = d2_cols(Pt).reset_index(drop=True)
    P["y"] = (P.outcome == "DIE").astype(int)
    cut = {b: float(np.nanmedian(P[b])) for b in BF}
    bm = float(np.nanmedian(P.btc_buy5))
    MS = {"M1_btc_since_entry>0": lambda D: np.where(D.btc_since_entry.isna(), np.nan, (D.btc_since_entry > 0).astype(float)),
          "M2_btc1h_touch>0": lambda D: np.where(D.btc1h_touch.isna(), np.nan, (D.btc1h_touch > 0).astype(float)),
          "M3_Czone_in": lambda D: D.czone.astype(float).values,
          "M4_btc_buy5>med": lambda D: np.where(D.btc_buy5.isna(), np.nan, (D.btc_buy5 > bm).astype(float))}
    quads = []
    for b in BF:
        hb = np.where(P[b].isna(), np.nan, (P[b] > cut[b]).astype(float))
        for mn, mf in MS.items():
            mv = mf(P)
            ok = np.isfinite(hb) & np.isfinite(mv)
            for bh in (0, 1):
                for mh in (0, 1):
                    quads.append((b, mn, bh, mh, ok, ok & (hb == bh) & (mv == mh)))
    y = P.y.values

    def zs(yy):
        return np.array([_pz(yy[ok], q[ok]) for (_, _, _, _, ok, q) in quads])
    z = zs(y)
    PM = dayblock_perm(P.te.values, P.day.values, R, np.random.default_rng(seed))
    mx = np.array([np.nanmax(np.abs(zs(y[PM[:, r]]))) for r in range(R)])
    rows = []
    srcs = P.src.values
    for i, (b, mn, bh, mh, ok, q) in enumerate(quads):
        g = P.group.values[q]
        rows.append(dict(book=b, book_side=f"{'>' if bh else '<='} {cut[b]:.4g}", macro=mn, macro_side=int(mh), n_scored=int(ok.sum()),
                         N=int(q.sum()), n_real=int((q & (srcs == "REAL")).sum()), G1_died=int((g == "G1").sum()),
                         G2_plus2=int((g == "G2").sum()), G3_plus3=int((g == "G3").sum()),
                         die_rate=float(y[q].mean()) if q.sum() else np.nan, die_rate_rest=float(y[ok & ~q].mean()) if (ok & ~q).sum() else np.nan,
                         z=z[i], p_fw=float((mx >= abs(z[i])).mean()) if np.isfinite(z[i]) else np.nan, small=bool(q.sum() < 5)))
    Q = pd.DataFrame(rows)
    Q.to_csv(os.path.join(OUT, "tests_d2_book_macro" + SUF + ".csv"), index=False)
    obs = np.nanmax(np.abs(z))
    s = dict(N=len(P), n_die=int(y.sum()), days=P.day.nunique(), max_abs_z=obs, null95=float(np.percentile(mx, 95)), fw_p=float((mx >= obs).mean()),
             cuts=cut, btc_buy5_median=bm)
    pd.Series(s).to_json(os.path.join(OUT, "tests_d2_summary" + SUF + ".json"))
    print("D2:", s)
    print(Q.sort_values("z", key=abs, ascending=False).head(12).to_string())


# ───────────────────────── D3: market features + exploratory exhaustive screen ─────────────────────────
def _rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0); mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(pdm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    mdi = 100 * pd.Series(mdm, index=h.index).ewm(alpha=1 / n, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


class Mkt:
    def __init__(self):
        b = _load5("BTCUSDT"); b.index = b.open_time + BAR          # close time
        c = b.c.astype(float)
        f5 = pd.DataFrame(index=b.index)
        f5["btc_rsi5m"] = _rsi(c).values; f5["btc_adx5m"] = _adx(b.h, b.l, c).values
        f5["btc_gap13_50_5m"] = ((c.ewm(span=13, adjust=False).mean() / c.ewm(span=50, adjust=False).mean() - 1) * 100).values
        f5["btc_off24h"] = ((c / b.h.rolling(288).max() - 1) * 100).values
        f5["btc_r72"] = ((c / c.shift(864) - 1) * 100).values
        f5["btc_eff72"] = ((c - c.shift(864)).abs() / c.diff().abs().rolling(864).sum()).values
        self.f5 = f5
        hb = b.groupby((b.open_time // 3_600_000) * 3_600_000).agg(h=("h", "max"), l=("l", "min"), c=("c", "last"), n=("c", "size"))
        hb = hb[hb.n == 12]; hb.index = hb.index + 3_600_000
        hc = hb.c.astype(float)
        f1 = pd.DataFrame(index=hb.index)
        f1["btc_rsi1h"] = _rsi(hc).values; f1["btc_adx1h"] = _adx(hb.h, hb.l, hc).values
        f1["btc_gap50_200_1h"] = ((hc.ewm(span=50, adjust=False).mean() / hc.ewm(span=200, adjust=False).mean() - 1) * 100).values
        e20 = hc.ewm(span=20, adjust=False).mean(); f1["btc1h_slope"] = ((e20 / e20.shift(3) - 1) * 100).values
        self.f1 = f1
        day = b.open_time // 86_400_000; dc = b.groupby(day).c.last(); self.dret = (dc / dc.shift(1) - 1) * 100
        self.k = {s: k1(s) for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
        g = pd.read_csv(os.path.join(REP, "ML_BTCFWD_grid5m.csv.gz"), usecols=["t", "br_up5", "agg_tbr5"]).dropna()
        self.g_t = g.t.values.astype("int64"); self.g = g[["br_up5", "agg_tbr5"]].values

    @staticmethod
    def _at(df, t):
        i = int(np.searchsorted(df.index.values, t, side="right")) - 1
        return df.iloc[i] if i >= 0 else None

    def opn(self, s, t):
        k = self.k[s]
        if k is None:
            return np.nan
        i = int(np.searchsorted(k.open_time.values, t))
        return float(k.o.values[i]) if i < len(k) and k.open_time.values[i] == t else np.nan

    def qv(self, t0, t1):
        tot = 0.0
        for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            k = self.k[s]
            if k is None:
                return np.nan
            m = (k.open_time.values >= t0) & (k.open_time.values < t1)
            if m.sum() < (t1 - t0) // MIN:
                return np.nan
            tot += k.qv.values[m].sum()
        return tot

    def feats(self, te, anchor, czone, vol_mult, hours):
        d = dict(czone=czone, vol_mult=vol_mult, ep_hours=hours)
        e0 = -(-int(te) // MIN) * MIN
        for tag, t in (("_e", e0), ("_t", anchor)):
            r5 = self._at(self.f5, t); r1 = self._at(self.f1, t)
            for c in self.f5.columns:
                d[c + tag] = r5[c] if r5 is not None else np.nan
            for c in self.f1.columns:
                d[c + tag] = r1[c] if r1 is not None else np.nan
            d["btc_1d" + tag] = self.dret.get((t // 86_400_000) - 1, np.nan)
            j = int(np.searchsorted(self.g_t, t - BAR, side="right")) - 1
            ok = j >= 0 and t - self.g_t[j] < 30 * MIN
            d["br_up5" + tag], d["agg_tbr5" + tag] = (self.g[j] if ok else (np.nan, np.nan))
        for s, nm in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
            pa = self.opn(s, anchor)
            d[f"{nm}_since_entry"] = (pa / self.opn(s, e0) - 1) * 100
            d[f"{nm}_r5_t"] = (pa / self.opn(s, anchor - 5 * MIN) - 1) * 100
            d[f"{nm}_r15_t"] = (pa / self.opn(s, anchor - 15 * MIN) - 1) * 100
        q5 = self.qv(anchor - 5 * MIN, anchor)
        d["mvol5_vs_entry5"] = q5 / self.qv(e0 - 5 * MIN, e0)
        d["mvol5_vs_pre60"] = q5 / (self.qv(e0 - 60 * MIN, e0) / 12)
        k = self.k["BTCUSDT"]; m = (k.open_time.values >= anchor - 5 * MIN) & (k.open_time.values < anchor)
        d["btc_buy5_t"] = k.tbqv.values[m].sum() / k.qv.values[m].sum() if m.sum() == 5 else np.nan
        return d


def d3_screen(P, bcols, R=1000, seed=17):
    MK = Mkt()
    M = pd.DataFrame([MK.feats(r.te, r.anchor_ms, r.czone, r.vol_mult, r.hours if "hours" in P else np.nan) for r in P.itertuples()])
    mcols = list(M.columns)
    cov = M.notna().mean()
    P = pd.concat([P.reset_index(drop=True), M.add_prefix("mk_")], axis=1)
    y = P.y.values.astype(float); n = len(P)
    Xb = P[bcols].values.astype(float); Xm = M.values.astype(float)
    hb = np.where(np.isfinite(Xb), (Xb > np.nanmedian(Xb, 0)).astype(float), np.nan)
    hm = np.where(np.isfinite(Xm), (Xm > np.nanmedian(Xm, 0)).astype(float), np.nan)
    Fb, Fm = np.isfinite(hb).astype(float), np.isfinite(hm).astype(float)
    Qb = [(hb == h).astype(float) for h in (0, 1)]; Qm = [(hm == h).astype(float) for h in (0, 1)]
    Nok = Fb.T @ Fm
    Nq = {(a, b): Qb[a].T @ Qm[b] for a in (0, 1) for b in (0, 1)}

    def zall(yy):
        Dok = (Fb * yy[:, None]).T @ Fm
        out = {}
        for (a, b), nq in Nq.items():
            dq = (Qb[a] * yy[:, None]).T @ Qm[b]
            nr = Nok - nq; dr = Dok - dq
            p = np.where(Nok > 0, Dok / np.maximum(Nok, 1), np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                se = np.sqrt(p * (1 - p) * (1 / nq + 1 / nr))
                z = (dq / nq - dr / nr) / se
            z[(nq < 5) | (nr < 5) | ~np.isfinite(z)] = np.nan
            out[(a, b)] = z
        return out
    Z = zall(y)
    obs = max(np.nanmax(np.abs(z)) for z in Z.values())
    PM = dayblock_perm(P.te.values, P.day.values, R, np.random.default_rng(seed))
    mx = np.array([max(np.nanmax(np.abs(z)) for z in zall(y[PM[:, r]]).values()) for r in range(R)])
    rows = []
    for (a, b), z in Z.items():
        ii, jj = np.where(np.isfinite(z))
        for i, j in zip(ii, jj):
            rows.append((bcols[i], mcols[j], a, b, z[i, j]))
    S = pd.DataFrame(rows, columns=["book", "macro", "book_high", "macro_high", "z"])
    S["absz"] = S.z.abs(); S = S.sort_values("absz", ascending=False)
    S["p_fw"] = [(mx >= v).mean() for v in S.absz.values]
    top = S.head(15).copy()
    g = P.group.values
    ext = []
    for r in top.itertuples():
        xb = P[r.book].values.astype(float); xm = P["mk_" + r.macro].values.astype(float)
        cb, cm = np.nanmedian(xb), np.nanmedian(xm)
        q = np.isfinite(xb) & np.isfinite(xm) & ((xb > cb) == bool(r.book_high)) & ((xm > cm) == bool(r.macro_high))
        ok = np.isfinite(xb) & np.isfinite(xm)
        ext.append(dict(book_cut=f"{'>' if r.book_high else '<='} {cb:.4g}", macro_cut=f"{'>' if r.macro_high else '<='} {cm:.4g}", N=int(q.sum()),
                        n_real=int((q & (P.src.values == "REAL")).sum()), days=int(P.day[q].nunique()),
                        pairs=int(P.pair[q].nunique()), G1_died=int((g[q] == "G1").sum()), G2_plus2=int((g[q] == "G2").sum()),
                        G3_plus3=int((g[q] == "G3").sum()), die_rate=float(y[q].mean()), die_rate_rest=float(y[ok & ~q].mean())))
    top = pd.concat([top.reset_index(drop=True), pd.DataFrame(ext)], axis=1)
    S.to_csv(os.path.join(OUT, "tests_d3_screen_all" + SUF + ".csv.gz"), index=False)
    top.to_csv(os.path.join(OUT, "tests_d3_top15" + SUF + ".csv"), index=False)
    cov.rename("coverage").to_csv(os.path.join(OUT, "tests_d3_macro_coverage" + SUF + ".csv"))
    summ = dict(N=n, n_die=int(y.sum()), days=P.day.nunique(), book_vars=len(bcols), macro_vars=len(mcols), eligible_quadrants=len(S),
                max_abs_z=float(obs), null95=float(np.percentile(mx, 95)), fw_p=float((mx >= obs).mean()))
    pd.Series(summ).to_json(os.path.join(OUT, "tests_d3_summary" + SUF + ".json"))
    print("D3:", summ); print(top.to_string())
    P.to_csv(os.path.join(OUT, "pooled_touch_rows" + SUF + ".csv"), index=False)


def stage_test():
    Rl = pd.read_csv(os.path.join(OUT, "real_trades.csv"))
    Em = pd.read_csv(os.path.join(OUT, "emul_signals.csv"))
    Rl["src"] = "REAL"; Em["src"] = "EMUL"
    if MASK:
        for D in (Rl, Em):
            k = D.anchor_min - 1; j = D.C_xm - 1
            for nm, off in SNAPS.items():
                leak = (k + off >= j).fillna(False).values
                cs = [c for c in D.columns if c.startswith(nm + "_")]
                D.loc[leak, cs] = np.nan
                print(f"mask {nm}: {int(leak.sum())} rows")
    tcols = [f"{s}_{x}{f}" for s in SNAPS for x in ("", "dE_", "dP_") for f in LEVELS]
    ecols = [f"ENT_{f}" for f in LEVELS] + [f"ENT_dP_{f}" for f in LEVELS]
    has = lambda D: D[["PRE_imb_1", "AT_imb_1", "PRE_spread", "AT_spread"]].notna().any(axis=1)  # noqa: E731
    Rt = Rl[(Rl.outcome.isin(["DIE", "GO"])) & has(Rl)].copy()
    Et = Em[(Em.outcome.isin(["DIE", "GO"])) & has(Em)].copy()
    Pt = pd.concat([Rt, Et[Et.real_match.fillna("") == ""]], ignore_index=True)
    sums, tabs = [], []
    for nm, D in (("REAL", Rt), ("EMUL", Et), ("POOLED", Pt)):
        D = D.assign(y=(D.outcome == "DIE").astype(int))
        T, s = family_test(D, tcols, "y", nm); s["family"] = "TOUCH (180)"
        tabs.append(T.assign(family="TOUCH")); sums.append(s)
        # acted-only (C still open at the touch-minute close) — secondary
    # entry family: DOA vs NOT-DOA (any resolved; book ENT present)
    for nm, D in (("REAL", Rl), ("EMUL", Em), ("POOLED", pd.concat([Rl, Em[Em.real_match.fillna("") == ""]], ignore_index=True))):
        D = D[D.outcome.isin(["DIE", "GO", "DOA"]) & D.ENT_imb_1.notna()].copy()
        D["y"] = (D.outcome == "DOA").astype(int)
        T, s = family_test(D, ecols, "y", nm); s["family"] = "ENTRY (24) DOA"
        tabs.append(T.assign(family="ENTRY")); sums.append(s)
    d2_test(Pt, Rt, Et)
    PP = d2_cols(Pt).assign(y=lambda D: (D.outcome == "DIE").astype(int))
    d3_screen(PP, tcols + ["B1_imb1", "B2_dask1", "B3_askwall", "B4_dbid_next"])
    TT = pd.concat(tabs, ignore_index=True)
    TT.to_csv(os.path.join(OUT, "tests_features" + SUF + ".csv"), index=False)
    SS = pd.DataFrame(sums)
    SS.to_csv(os.path.join(OUT, "tests_family" + SUF + ".csv"), index=False)
    print(SS.to_string())
    # sign consistency REAL vs EMUL on the pooled top
    for fam in ("TOUCH", "ENTRY"):
        P = TT[(TT.family == fam) & (TT["sample"] == "POOLED")].sort_values("p_rank").head(15)
        Rr = TT[(TT.family == fam) & (TT["sample"] == "REAL")].set_index("feature")
        Ee = TT[(TT.family == fam) & (TT["sample"] == "EMUL")].set_index("feature")
        P = P.assign(z_real=P.feature.map(Rr.z), z_emul=P.feature.map(Ee.z))
        print(fam); print(P[["feature", "n_die", "n_go", "med_die", "med_go", "z", "p_rank", "p_fw", "z_real", "z_emul"]].to_string())


if __name__ == "__main__":
    st = sys.argv[1] if len(sys.argv) > 1 else "all"
    if st in ("real", "all"):
        stage_real()
    if st in ("emul", "all"):
        stage_emul()
    if st in ("test", "all"):
        stage_test()
