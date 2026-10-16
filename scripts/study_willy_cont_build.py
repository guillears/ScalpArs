#!/usr/bin/env python3
"""WILLY CONTINUATION — step 2: price every WILLY fill's path under the pre-registered exit arms and build the decision-moment features.
Pre-registration: reports/WILLY_CONT_PREREG.txt (asserted). Research only, cache + reports/willy_cont/k1m (fetched public klines).

Arms (coordinator correction 2026-10-16: the late TP closes AT MARKET at the print when already ≥ +0.25 at minute 30):
  TODAY  +1.25 resting (line) ≤ 30 min · minute 30: last print ≥ +0.25 → market at the first print after te+30 (1m tail: next bar open)
         else +0.25 resting (line) · stop −4 → min(print, −4) − 0.20 · cap 90 → last print
  HOLD2 / HOLD3   TODAY, but at the +1.25 touch the TP moves to +2 / +3 (late rule kept)
  KEEP   +1.25 / −4 / 90, no late rule (FX.evaluate)          CLOSE30   market at the first print after te+30 (if still open)
Output reports/willy_cont/rows.pkl (one row per fill: BT seeds 1–3, MASTER, LIVE).
Usage: venv/bin/python scripts/study_willy_cont_build.py [--jobs 8]
"""
import os, sys, glob, argparse
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REP = os.path.join(ROOT, "reports")
OUT = os.path.join(REP, "willy_cont")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
assert os.path.exists(os.path.join(REP, "WILLY_CONT_PREREG.txt")), "pre-registration missing"
import study_tick_maker_lock_common as C  # noqa: E402  (2025 loader patches, network blocked)
FX, WT = C.FX, C.WT
WT._NET_BLOCKED = True

MIN, BAR5, DAY = 60_000, 300_000, 86_400_000
TP, SL, CAP, LATE, LTP, SLIP = 1.25, 4.0, 90, 30, 0.25, 0.20
FILLS = os.path.join(REP, "ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv")
MASTER = os.path.join(REP, "MASTER_POOL_stacked.csv")
LIVE = os.path.expanduser("~/Downloads/scalpars_orders_live_2026-10-16_15-19-54 (1).csv")
K1M = os.path.join(OUT, "k1m")
CTX = os.path.join(REP, "willy_theory", "ctx")


def ms(s):
    return pd.to_datetime(pd.Series(s).astype(str).str[:23], format="mixed").values.astype("datetime64[ms]").astype("int64")


# ───────────────────────── exit arms on one path ─────────────────────────
def _stop_val(P, i):
    if P["bar"][i]:
        op = P["op"][i]
        v = min(-SL, float(op)) if np.isfinite(op) else -SL
    else:
        v = float(P["lo"][i])
    return min(v, -SL) - SLIP


def _mkt(P, i):
    """market close at event i: a tick print's net; a 1m bar's open (the first price after the decision)."""
    if P["bar"][i]:
        op = P["op"][i]
        return float(op) if np.isfinite(op) else float(P["cl"][i])
    return float(P["cl"][i])


def arm(P, hold=None, late=True, close30=False):
    """→ (booked %, why, exit minute). hold = TP after the +1.25 touch (None = bank at +1.25). late = DL 297 rule (market at the print if
    ≥ +0.25 at minute 30, else +0.25 resting). close30 = market at minute 30 regardless. Stop before TP on the same event."""
    te = P["te"]
    n = int(np.searchsorted(P["t"], te + CAP * MIN, side="right"))
    t, hi, lo, cl = P["t"][:n], P["hi"][:n], P["lo"][:n], P["cl"][:n]
    line = TP; touched = False; phase_late = False; latel = None
    for i in range(n):
        if not phase_late and (late or close30) and t[i] > te + LATE * MIN:
            phase_late = True
            st = float(cl[i - 1]) if i > 0 else np.nan                     # the last print ≤ te + 30
            if close30:
                return _mkt(P, i), "CLOSE30", (t[i] - te) / MIN
            if np.isfinite(st) and st >= LTP:
                return _mkt(P, i), "LATE_MKT", (t[i] - te) / MIN
            latel = LTP
        if lo[i] <= -SL:
            return _stop_val(P, i), "STOP", (t[i] - te) / MIN
        cur = latel if phase_late else line
        if hi[i] >= cur:
            if phase_late:
                return float(cur), "LATE_TP", (t[i] - te) / MIN
            if hold is None or touched:
                return float(cur), ("TP" if not touched else f"TP{hold:g}"), (t[i] - te) / MIN
            touched = True; line = hold
            if hi[i] >= line:
                return float(line), f"TP{hold:g}", (t[i] - te) / MIN
    if P["end_ms"] < te + CAP * MIN or n == 0:
        return np.nan, "UNRES", np.nan
    return float(cl[n - 1]), "CAP", float(CAP)


def keep_book(ev):
    w = ev["why"]
    if w == "TP":
        return float(ev["pct_line"])
    if w == "STOP":
        return min(float(ev["pct"]), -SL) - SLIP
    if w == "CAP":
        return float(ev["pct"])
    return np.nan


def events(P):
    te = P["te"]
    n = int(np.searchsorted(P["t"], te + CAP * MIN, side="right"))
    t, hi, lo, cl = P["t"][:n], P["hi"][:n], P["lo"][:n], P["cl"][:n]
    o = {}
    ist = np.flatnonzero(lo <= -SL); s = int(ist[0]) if len(ist) else n
    for lv in (1.25, 2.0, 3.0):
        k = np.flatnonzero(hi[:s] >= lv)
        o[f"t_{lv:g}"] = (t[k[0]] - te) / MIN if len(k) else np.nan     # first touch before any stop (bar both lines = stop)
        o[f"ta_{lv:g}"] = int(t[k[0]]) if len(k) else np.nan
    o["t_stop"] = (t[s] - te) / MIN if s < n else np.nan
    k = o["ta_1.25"]
    o["mae_pre"] = float(lo[:np.searchsorted(t, k, side="left") + 1].min()) if np.isfinite(k) and n else np.nan
    n30 = int(np.searchsorted(t, te + LATE * MIN, side="right"))
    o["cl30"] = float(cl[n30 - 1]) if n30 else np.nan
    o["mfe30"] = float(hi[:n30].max()) if n30 else np.nan
    o["mae30"] = float(lo[:n30].min()) if n30 else np.nan
    o["mfe90"] = float(hi.max()) if n else np.nan
    o["mae90"] = float(lo.min()) if n else np.nan
    # Q2 label: open at 30 (no 1.25 touch ≤ 30, no stop ≤ 30) → reaches +1.25 before −4 by 90
    if n30 and n30 < n:
        h2, l2 = hi[n30:], lo[n30:]
        a = np.flatnonzero(h2 >= TP); b = np.flatnonzero(l2 <= -SL)
        a = int(a[0]) if len(a) else None; b = int(b[0]) if len(b) else None
        o["r125_after30"] = float(a is not None and (b is None or a < b))
        o["stop_after30"] = float(b is not None and (a is None or b <= a))
    else:
        o["r125_after30"] = np.nan; o["stop_after30"] = np.nan
    return o


def walk(r, tc):
    k1 = r.get("k1m")
    P = FX.load_path(r["pair"], int(r["te"]), float(r["E"]), CAP, tc, k1m=k1)
    out = dict(key=r["key"])
    out["src"] = FX.path_src(P)
    out["tick90"] = bool(P["n_tick"] > 0 and P["tick_end"] >= int(r["te"]) + (CAP + 1) * MIN)
    pl = FX.evaluate(P, TP, SL, CAP)
    out.update(plain_why=pl["why"], plain_pct=pl["pct"], y_keep=keep_book(pl))
    el = WT.evaluate_late(P, TP, SL, CAP, LATE, LTP)
    out.update(evl_why=el["why"], evl_pct=el["pct"])
    for nm, kw in (("today", {}), ("hold2", dict(hold=2.0)), ("hold3", dict(hold=3.0)), ("close30", dict(close30=True))):
        y, w, x = arm(P, **kw)
        out.update({f"y_{nm}": y, f"why_{nm}": w, f"x_{nm}": x})
    out.update(events(P))
    return out


def job(rows):
    WT._NET_BLOCKED = True
    tc, out = {}, []
    for r in rows:
        try:
            out.append(walk(r, tc))
        except Exception as e:  # noqa: BLE001
            out.append(dict(key=r["key"], err=str(e)[:200]))
        if len(tc) > 400:
            tc.clear()
    return out


# ───────────────────────── coin 1m taker window ─────────────────────────
def coin_k(pair, te):
    st = int(te // MIN) * MIN - 9 * MIN
    f = os.path.join(K1M, f"{pair}_{st}.csv")
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    return d if len(d) else None


def coin_feats(k, te, D, pre):
    """closed 1m bars (open_time + 1 min ≤ D): last-5 taker share, volume vs the 5 min before entry, since-entry taker share, coin 5-min ret."""
    if k is None or not np.isfinite(D):
        return {}
    ot = k.open_time.values.astype("int64")
    qv = k.qvol.values.astype(float); tq = k.tbq.values.astype(float); c = k.c.values.astype(float)
    cl = ot + MIN <= D
    idx = np.flatnonzero(cl)
    o = {}
    if len(idx) >= 5:
        j = idx[-5:]
        o[f"{pre}_tbs5"] = tq[j].sum() / qv[j].sum() if qv[j].sum() > 0 else np.nan
        o[f"{pre}_qv5"] = qv[j].sum()
        if len(idx) >= 6:
            o[f"{pre}_cret5"] = (c[idx[-1]] / c[idx[-6]] - 1) * 100
    since = cl & (ot >= (te // MIN) * MIN)
    o[f"{pre}_tbs_since"] = tq[since].sum() / qv[since].sum() if since.any() and qv[since].sum() > 0 else np.nan
    pre_m = ot + MIN <= te
    jp = np.flatnonzero(pre_m)[-5:]
    if len(jp) == 5 and qv[jp].sum() > 0 and f"{pre}_qv5" in o:
        o[f"{pre}_vratio"] = o[f"{pre}_qv5"] / qv[jp].sum()
    o.pop(f"{pre}_qv5", None)
    return o


# ───────────────────────── market frames ─────────────────────────
def market_1m():
    fr = {}
    for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        parts = [pd.read_csv(os.path.join(REP, "ml_btcfwd_fetch", "k1m_taker", f"{s}.csv"))]
        parts += [pd.read_csv(f) for f in glob.glob(os.path.join(CTX, f"m1_{s}_*.csv"))]
        d = pd.concat(parts).drop_duplicates("open_time", keep="first").sort_values("open_time").set_index("open_time")
        fr[s] = d[["o", "c", "qvol", "tbq"]].astype(float)
    return fr


def market_feats_at(M, D, te, pre):
    """BTC / ETH / SOL: return since the entry minute's open, last 5 / 15 closed 1m bars; market taker share last 5 closed bars."""
    o = {}
    last = int(D // MIN) * MIN - MIN                       # open time of the last closed bar
    e0 = int(te // MIN) * MIN
    sq, tq = 0.0, 0.0; ok = True
    for s, nm in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
        d = M[s]
        try:
            cL = d.c.at[last]
        except KeyError:
            ok = False; continue
        if pre != "e":
            try:
                o[f"{pre}_{nm}_since"] = (cL / d.o.at[e0] - 1) * 100
            except KeyError:
                pass
        for k in (5, 15):
            try:
                o[f"{pre}_{nm}_r{k}"] = (cL / d.c.at[last - k * MIN] - 1) * 100
            except KeyError:
                pass
        w = d.loc[last - 4 * MIN:last]
        if len(w) == 5:
            sq += w.qvol.sum(); tq += w.tbq.sum()
        else:
            ok = False
    o[f"{pre}_mkt_tbs5"] = tq / sq if ok and sq > 0 else np.nan
    return o


def breadth_5m():
    g = pd.read_csv(os.path.join(REP, "ML_BTCFWD_grid5m.csv.gz"), usecols=["t", "br_up5", "agg_tbr5", "agg_tbr15", "br_n"])
    g = g[g.br_n >= 30].set_index("t")
    # master / live (grid ends 2026-10-15 15:55): top-50 5m bars of reports/willy_theory/ctx u5_*
    rows = {}
    for f in glob.glob(os.path.join(CTX, "u5_*.csv")):
        d = pd.read_csv(f)
        for r in d.itertuples():
            t = int(r.open_time) + BAR5
            rows.setdefault(t, {})[os.path.basename(f).split("_")[1]] = (r.o, r.c, r.qvol, r.tbq)
    ext = []
    for t, m in rows.items():
        if t in g.index or len(m) < 30:
            continue
        v = np.array(list(m.values()), dtype=float)
        ext.append(dict(t=t, br_up5=float((v[:, 1] > v[:, 0]).mean()), agg_tbr5=v[:, 3].sum() / v[:, 2].sum(), br_n=len(m)))
    if ext:
        E = pd.DataFrame(ext).set_index("t").sort_index()
        E["agg_tbr15"] = np.nan
        g = pd.concat([g, E]).sort_index()
        g = g[~g.index.duplicated()]
    return g


def breadth_at(G, D, pre):
    b = int(D // BAR5) * BAR5
    o = {}
    try:
        o[f"{pre}_br_up5"] = G.br_up5.at[b]; o[f"{pre}_agg_tbr5"] = G.agg_tbr5.at[b]; o[f"{pre}_agg_tbr15"] = G.agg_tbr15.at[b]
        w = G.br_up5.reindex([b - 2 * BAR5, b - BAR5, b])
        o[f"{pre}_br_up15"] = float(w.mean()) if w.notna().all() else np.nan
    except KeyError:
        pass
    return o


def intraday_tables(M):
    """BTC rv60 / prior-7d, market quote volume last 60 / prior-7d mean, BTC day return; per-day BTC 15- and 5-min autocorrelation (prior 3 days)."""
    b = M["BTCUSDT"]
    full = pd.RangeIndex(int(b.index.min()), int(b.index.max()) + MIN, MIN)
    bc = b.c.reindex(full)
    lr = np.log(bc).diff()
    rv60 = lr.rolling(60, min_periods=50).std()
    rv7d = lr.rolling(10080, min_periods=8000).std()
    qv = sum(M[s].qvol.reindex(full) for s in M)
    q60 = qv.rolling(60, min_periods=50).sum()
    q60_7d = q60.rolling(10080, min_periods=8000).mean()
    T = pd.DataFrame(dict(btc_rv60r=rv60 / rv7d, mkt_qv60r=q60 / q60_7d, bc=bc), index=full)
    # persistence per UTC day from the prior 3 full days
    days = pd.to_datetime(full, unit="ms").normalize()
    c15 = bc[(full % (15 * MIN)) == 0]; c5 = bc[(full % (5 * MIN)) == 0]
    r15 = np.log(c15).diff(); r5 = np.log(c5).diff()
    per = {}
    for d in pd.date_range(days.min() + pd.Timedelta(days=3), days.max(), freq="D"):
        d0 = int(d.value // 1_000_000)
        a = r15[(r15.index >= d0 - 3 * DAY) & (r15.index < d0)].dropna()
        a5 = r5[(r5.index >= d0 - 3 * DAY) & (r5.index < d0)].dropna()
        per[d.strftime("%Y-%m-%d")] = dict(ac15_3d=a.autocorr(1) if len(a) > 200 else np.nan, ac5_3d=a5.autocorr(1) if len(a5) > 600 else np.nan)
    return T, pd.DataFrame(per).T


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=8); a = ap.parse_args()
    D = pd.read_csv(FILLS, low_memory=False)
    D = D[(D.sleeve == "FRENZY_WILLY") & (D.status == "CLOSED") & (~D.is_probe.astype(bool))].copy()
    D["te"] = ms(D.opened_at); D["E"] = D.entry_price.astype(float); D["block"] = "BT"; D["batch"] = "BT"
    Ms = pd.read_csv(MASTER, low_memory=False); Ms = Ms[Ms.entry_strategy == "FRENZY_WILLY"].copy()
    Ms["te"] = ms(Ms.opened_at.astype(str).str[:19]); Ms["E"] = Ms.entry_price.astype(float); Ms["block"] = "MASTER"; Ms["seed"] = 0
    Lv = pd.read_csv(LIVE, low_memory=False); Lv = Lv[Lv.entry_strategy == "FRENZY_WILLY"].copy()
    Lv["te"] = ms(Lv.opened_at.astype(str).str[:19]); Lv["E"] = Lv.entry_price.astype(float); Lv["block"] = "LIVE"; Lv["seed"] = 0
    msk = pd.read_pickle(os.path.join(REP, "willy_theory", "ms.pkl"))
    bmap = {(b, p, str(o)[:19]): (bt, u) for b, p, o, bt, u in zip(msk.block, msk.pair, msk.opened_at, msk.batch, msk.usd_pt)}
    for X in (Ms, Lv):
        X["batch"] = [bmap.get((b, p, str(o)[:19]), (None, np.nan))[0] for b, p, o in zip(X.block, X.pair, X.opened_at)]
        X["usd_pt"] = [bmap.get((b, p, str(o)[:19]), (None, np.nan))[1] for b, p, o in zip(X.block, X.pair, X.opened_at)]
    A = pd.concat([D, Ms, Lv], ignore_index=True, sort=False)
    A["key"] = A.pair + "|" + A.te.astype(str) + "|" + A.E.map(lambda x: f"{x:.10g}")
    U = A.drop_duplicates("key")[["key", "pair", "te", "E", "block"]]
    items = U.to_dict("records")
    for r in items:                                   # master / live: the local 1m cache ends 2026-10-12 → add the fetched window
        if r["block"] != "BT":
            k = coin_k(r["pair"], r["te"])
            base = WT.load_1m(r["pair"])
            if k is not None:
                kk = k[["open_time", "o", "h", "l", "c"]].astype({"open_time": "int64"})
                r["k1m"] = pd.concat([base, kk]).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    print(f"walking {len(items)} unique paths", flush=True)
    from multiprocessing import Pool
    pairs = sorted({r["pair"] for r in items}); g = {p: i % (a.jobs * 4) for i, p in enumerate(pairs)}
    chunks = [sorted([r for r in items if g[r["pair"]] == i], key=lambda r: r["te"]) for i in range(a.jobs * 4)]
    with Pool(a.jobs) as Pp:
        R = pd.DataFrame(sum(Pp.map(job, [c for c in chunks if c], chunksize=1), []))
    if "err" in R.columns and R.err.notna().any():
        print("walk errors", R.err.notna().sum(), R[R.err.notna()].err.head().tolist())
    # parity vs the LOWVWAP rows (plain 1.25/4/90 walk, same walker)
    S = pd.read_csv(os.path.join(REP, "WILLY_LOWVWAP_SHORTRUN_rows.csv"), low_memory=False)
    S["key"] = S.pair + "|" + S.te.astype(str) + "|" + S.E.map(lambda x: f"{float(x):.10g}")
    J = S.drop_duplicates("key").merge(R, on="key", how="inner", suffixes=("_s", ""))
    ok = np.isclose(J.plain_pct_s.astype(float), J.plain_pct.astype(float), atol=1e-6, equal_nan=True) & (J.plain_why_s.astype(str) == J.plain_why.astype(str))
    ok2 = np.isclose(J.raw_pct.astype(float), J.evl_pct.astype(float), atol=1e-6, equal_nan=True) & (J.raw_why.astype(str) == J.evl_why.astype(str))
    par = f"PARITY vs WILLY_LOWVWAP_SHORTRUN_rows: joined {len(J)} · plain walk identical {ok.sum()}/{len(J)} ({ok.mean()*100:.2f}%) · evaluate_late identical {ok2.sum()}/{len(J)}"
    print(par, flush=True)
    assert ok.mean() >= 0.999, "walker parity broken"
    # coin + market features per unique key
    M = market_1m(); G = breadth_5m(); T, PER = intraday_tables(M)
    DF = pd.read_csv(os.path.join(REP, "lite_regime", "day_features.csv")).set_index("day")
    DF = DF[[c for c in DF.columns if c not in ("day_ms",) and not c.startswith(("lite_", "willy_"))]].add_prefix("d_")
    Rk = R.set_index("key")
    feats = []
    for r in U.itertuples():
        te = int(r.te); f = dict(key=r.key)
        k = coin_k(r.pair, te)
        f["k1m_ok"] = k is not None
        rr = Rk.loc[r.key] if r.key in Rk.index else None
        f.update(coin_feats(k, te, te, "e"))
        f.update(market_feats_at(M, te, te, "e")); f.update(breadth_at(G, te, "e"))
        if rr is not None:
            ta = rr.get("ta_1.25")
            if np.isfinite(ta) and rr["t_1.25"] <= LATE:
                f.update(coin_feats(k, te, ta, "a")); f.update(market_feats_at(M, ta, te, "a")); f.update(breadth_at(G, ta, "a"))
            f.update(coin_feats(k, te, te + LATE * MIN, "m")); f.update(market_feats_at(M, te + LATE * MIN, te, "m"))
            f.update(breadth_at(G, te + LATE * MIN, "m"))
        lastc = int(te // MIN) * MIN - MIN
        if lastc in T.index:
            f["i_btc_rv60r"] = T.btc_rv60r.at[lastc]; f["i_mkt_qv60r"] = T.mkt_qv60r.at[lastc]
            d0 = int(te // DAY) * DAY
            if d0 in T.index and np.isfinite(T.bc.at[lastc]):
                f["i_btc_day"] = (T.bc.at[lastc] / T.bc.at[d0] - 1) * 100
        feats.append(f)
    FE = pd.DataFrame(feats)
    out = A.merge(R, on="key", how="left").merge(FE, on="key", how="left")
    out["ts"] = pd.to_datetime(out.te, unit="ms")
    out["day"] = out.ts.dt.strftime("%Y-%m-%d"); out["month"] = out.ts.dt.strftime("%Y-%m")
    out["hour"] = out.ts.dt.hour.astype(float)
    out["era"] = np.where(out.ts < "2026-01-01", "2025", np.where(out.ts < "2026-07-01", "2026-H1", "2026-H2"))
    out = out.join(DF, on="day").join(PER, on="day")
    out.to_pickle(os.path.join(OUT, "rows.pkl"))
    open(os.path.join(OUT, "parity.txt"), "w").write(par + "\n")
    print("wrote rows", len(out), out.groupby("block").size().to_dict())


if __name__ == "__main__":
    main()
