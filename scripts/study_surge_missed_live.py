#!/usr/bin/env python3
"""⚡ SURGE missed-opps study — BATCH / LIVE block (2026-10-05 → now): the same rules and variants as study_surge_missed_opps.py,
on public 5m klines fetched by study_surge_missed_live_fetch.py, walked on 1m klines (O→L→H→C green / O→H→L→C red) with TODAY's
SURGE exit. Also reconciles the replica's per-trigger pair statuses with the bot's decision journals (BLOCK SURGE_* rows).
Usage: venv/bin/python scripts/study_surge_missed_live.py need   → writes the 1m need-list (then run the fetcher's 1m mode)
       venv/bin/python scripts/study_surge_missed_live.py run
Read-only."""
import glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import study_surge_missed_opps as M

LIVE = os.path.join(ROOT, "reports", "backtest_cache", "surge_missed", "live")
BAR, MIN, DAY = M.BAR, M.MIN, M.DAY
START = pd.Timestamp("2026-10-05").value // 10**6
SCR = sys.argv[2] if len(sys.argv) > 2 else None


def load():
    ex = json.load(open(os.path.join(LIVE, "exchange_info.json")))
    btc = pd.read_csv(os.path.join(LIVE, "k5m", "BTCUSDT.csv")).drop_duplicates("open_time").set_index("open_time").sort_index()
    btc = btc.iloc[:-1]   # forming bar
    btc["qv"] = btc.vol * btc.c
    btc["r30"] = (btc.c / btc.c.shift(6) - 1) * 100
    btc["hi24"] = btc.h.shift(1).rolling(288).max()
    btc["volx"] = btc.qv / btc.qv.shift(1).rolling(288).median()
    btc["at_hi"] = btc.c >= btc.hi24
    m = (btc.index >= START) & (btc.r30 >= M.MOVE) & (btc.volx >= M.VOLX)
    cand = btc.index[m].values
    T = sorted({int(t) + k * BAR for t in cand for k in range(4)} & set(btc.index.tolist()))
    parts = []
    for f in glob.glob(os.path.join(LIVE, "k5m", "*.csv")):
        d = pd.read_csv(f)
        if len(d) < 300:
            continue
        tmp = os.path.join(LIVE, "_tmp.csv")
        r = M._pair_feat((f, set(T)))
        if r is not None and len(r):
            parts.append(r)
    F = pd.concat(parts, ignore_index=True)
    B = btc.loc[btc.index.isin(T), ["c", "r30", "volx", "at_hi", "hi24"]].reset_index()
    B["cand"] = B.open_time.isin(cand)
    # universe + gvol (same code path as the backtest, with this cache's exchange info)
    M.ERAS["live"] = (LIVE, "2026-10-05", "2026-12-31")
    pd.to_pickle(dict(F=F, B=B, cand=cand), os.path.join(M.OUT, "feat_live.pkl"))
    U, B = M.universe("live")
    return U, B


def walk_1m(pair, t_entry, atr):
    SR, TW = M._walker()
    f = os.path.join(LIVE, "k1m", f"{pair}_{int(t_entry)}.csv")
    if not os.path.exists(f):
        return dict(pnl=np.nan, src="none")
    k = pd.read_csv(f)[["open_time", "o", "h", "l", "c"]].to_numpy()
    k = k[k[:, 0] >= t_entry]
    if len(k) < 2:
        return dict(pnl=np.nan, src="none")
    e = float(k[0, 1]); ts, px = [], []
    for ot, o, h, l, c in k:
        seq = (o, l, h, c) if c >= o else (o, h, l, c)
        for j, p in enumerate(seq):
            ts.append(ot + j * 15_000); px.append(p)
    ts, px = np.array(ts[1:], dtype=np.int64), np.array(px[1:], dtype=float)
    pnl = (px / e - 1) * 100 - SR.FEE
    o = SR.run_exit(ts, pnl, t_entry, TW.stop_fn(TW.dist_today, float(atr)), None, M.HOLD)
    done = ts[-1] >= t_entry + M.HOLD * MIN - 2 * MIN or (o is not None and o[1] != "TIME")
    return dict(pnl=o[0], why=o[1] if done else "OPEN", minutes=o[2], peak=o[3], src="1m", entry_px=e)


def main():
    U, B = load()
    B = B.set_index("open_time").sort_index()
    byt = {t: g for t, g in U.groupby("open_time")}
    cand = B.index[B.cand].values
    need, fills, uni = set(), [], []
    for name, v in M.VARIANTS.items():
        last_fill = -10**18; open_pos = []
        for t in cand:
            b = B.loc[t]
            if v["high"] and not bool(b.at_hi):
                continue
            g = b.gvol
            gv_ok = np.isfinite(g) and g >= M.GMIN
            if name == "TODAY":
                uni.append(dict(trig=t, status=("GVOL_LOW" if not gv_ok else ("SPACED" if t - last_fill < v["sp"] * 3600_000 else "TRIGGER")),
                                btc_move=b.r30, volx=b.volx, gvol=g))
            if not gv_ok or t - last_fill < v["sp"] * 3600_000:
                if name == "TODAY" and gv_ok:   # spaced-out: price its would-be picks
                    rows = byt.get(t)
                    if rows is not None:
                        for p, why in M._picks(rows, float(b.r30), v, set())[:M.SLOTS]:
                            rr = rows[rows.pair == p].iloc[0]
                            te = int(t) + BAR + M.LATENCY_S * 1000
                            need.add((p, te))
                            fills.append(dict(variant="SPACED_OUT_PICKS", trig=int(t), pair=p, kind=why, atr=rr.atr, pmove=rr.pmove,
                                              btc_move=b.r30, gap=rr.gap, t_entry=te))
                continue
            open_pos = [(p, x) for p, x in open_pos if x > t + BAR]
            taken = {p for p, _ in open_pos}; got = 0
            for k in range(v["wb"] + 1):
                bt = int(t) + k * BAR
                rows = byt.get(bt)
                if rows is None or bt not in B.index:
                    break
                bm = float(B.at[bt, "r30"])
                if name == "TODAY" and k == 0:
                    for r in rows.itertuples():
                        st = ("ATR_LOW" if not r.atr >= M.ATR_MIN else "NOT_LEADER" if not r.pmove > bm else
                              "EMA_GAP" if not r.gap >= M.GAP_MIN else "PASS")
                        uni.append(dict(trig=t, status="PAIR", pair=r.pair, rank=r.rank, atr=r.atr, pmove=r.pmove, gap=r.gap, btc_move=bm, pair_status=st))
                        if st in ("NOT_LEADER", "EMA_GAP") or (st == "ATR_LOW" and r.atr >= 1.0):
                            te = int(t) + BAR + M.LATENCY_S * 1000
                            need.add((r.pair, te))
                            fills.append(dict(variant="REFUSED_" + st, trig=int(t), pair=r.pair, kind=st, atr=r.atr, pmove=r.pmove, btc_move=bm,
                                              gap=r.gap, t_entry=te))
                for p, why in M._picks(rows, bm, v, taken):
                    if got >= M.SLOTS or len(open_pos) >= M.SLOTS:
                        break
                    rr = rows[rows.pair == p].iloc[0]
                    te = bt + BAR + M.LATENCY_S * 1000
                    need.add((p, te))
                    w = walk_1m(p, te, float(rr.atr))
                    got += 1; taken.add(p)
                    open_pos.append((p, te + float(w.get("minutes", M.HOLD) if np.isfinite(w.get("pnl", np.nan)) else M.HOLD) * MIN))
                    fills.append(dict(variant=name, trig=int(t), entry_bar=bt, bar_k=k, pair=p, kind=why, atr=rr.atr, pmove=rr.pmove, btc_move=bm,
                                      gap=rr.gap, t_entry=te))
            if got:
                last_fill = t
    now = int(pd.Timestamp.utcnow().value // 10**6)
    N = pd.DataFrame(sorted(need), columns=["pair", "t0"]); N["t1"] = (N.t0 + (M.HOLD + 2) * MIN).clip(upper=now)
    N.to_csv(os.path.join(LIVE, "need_1m.csv"), index=False)
    Fl = pd.DataFrame(fills)
    W = [walk_1m(r.pair, r.t_entry, r.atr) for r in Fl.itertuples()]
    Fl = pd.concat([Fl, pd.DataFrame(W)], axis=1)
    Fl["trig_utc"] = pd.to_datetime(Fl.trig, unit="ms") + pd.Timedelta(minutes=5)   # the bar CLOSE (journal / ledger convention)
    Fl.to_csv(os.path.join(M.REP, "SURGE_MISSED_OPPS_live_fills.csv"), index=False)
    Uu = pd.DataFrame(uni); Uu["trig_utc"] = pd.to_datetime(Uu.trig, unit="ms") + pd.Timedelta(minutes=5)
    Uu.to_csv(os.path.join(M.REP, "SURGE_MISSED_OPPS_live_universe.csv"), index=False)
    print("need 1m:", len(N), "· missing walks:", Fl.pnl.isna().sum())
    print(Uu[Uu.status != "PAIR"].to_string())
    print(Fl.groupby("variant").agg(N=("pnl", "size"), walked=("pnl", "count"), trig=("trig", "nunique"), avg=("pnl", "mean"),
                                     wr=("pnl", lambda x: (x > 0).mean())).round(3).to_string())


if __name__ == "__main__":
    main()
