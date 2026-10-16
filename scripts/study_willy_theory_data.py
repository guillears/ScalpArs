#!/usr/bin/env python3
"""WILLY THEORY — step 4: assemble the analysis frames (no new pricing).
  BT   one row per rebuilt WILLY entry: signal features + market (W6 / W6b / post) + taker (W2) + RH / R1 outcome + grid + MATCH flag
  ALT  every priced window bar (W0 / W1 / W7 refused cohort / W8 random-bar grid)
  MS   master 19 + live 8 WILLY fills with the same features (stamps + fetched context) and their as-traded / today outcomes, % and $
Output reports/willy_theory/{bt,alt,ms}.pkl
"""
import os, sys, glob
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_willy_theory_common as W  # noqa: E402

BAR, MIN = W.BAR_MS, W.MIN


def k1m_taker(pair):
    fs = glob.glob(os.path.join(W.OUT, "k1m_taker", f"{pair}_*.csv"))
    fr = [pd.read_csv(f) for f in fs]
    fr = [f for f in fr if len(f)]
    if not fr:
        return None
    return pd.concat(fr).drop_duplicates("open_time").set_index("open_time").sort_index()


def taker_feats(k, spike_open, T):
    """spike-bar and red-bar taker-buy share of quote volume (5 × 1m), last 1m of the red bar, red-bar 1m share trend."""
    out = {}
    if k is None:
        return out
    for nm, a in (("spk", spike_open), ("red", T)):
        idx = [a + i * MIN for i in range(5)]
        x = k.reindex(idx)
        if x.qvol.notna().sum() == 5 and x.qvol.sum() > 0:
            out[f"tbs_{nm}"] = x.tbq.sum() / x.qvol.sum() * 100
            out[f"q_{nm}"] = x.qvol.sum()
            if nm == "red":
                l1 = x.iloc[-1]
                out["tbs_red_last1m"] = l1.tbq / l1.qvol * 100 if l1.qvol > 0 else np.nan
                f2 = x.iloc[:2]; l2 = x.iloc[-2:]
                out["tbs_red_trend"] = (l2.tbq.sum() / l2.qvol.sum() - f2.tbq.sum() / f2.qvol.sum()) * 100 if l2.qvol.sum() > 0 and f2.qvol.sum() > 0 else np.nan
    return out


def bt():
    S = pd.read_pickle(W.SIG)
    S = S[S.T_ent.notna()].copy()
    S["T"] = S.T_ent.astype("int64")
    R = pd.read_pickle(os.path.join(W.OUT, "priced.pkl"))
    D = S.merge(R, on=["pair", "T"], how="left")
    M = pd.read_pickle(os.path.join(W.OUT, "market.pkl"))
    D = D.merge(M, on="T", how="left")
    tk = []
    for p, G in D.groupby("pair"):
        k = k1m_taker(p)
        for r in G.itertuples():
            tk.append(dict(pair=p, T=r.T, **taker_feats(k, int(r.spike_ts) - BAR, int(r.T))))
    D = D.merge(pd.DataFrame(tk), on=["pair", "T"], how="left")
    P = pd.read_csv(os.path.join(W.OUT, "parity_rows.csv"))
    P1 = P[(P.seed == 1) & P.hit][["pair", "T", "entry_price", "opened_at"]].rename(columns={"entry_price": "E_replay"})
    D = D.merge(P1, on=["pair", "T"], how="left")
    D["MATCH"] = D.E_replay.notna()
    D["ts"] = pd.to_datetime(D["T"], unit="ms")
    D["day"] = D.ts.dt.strftime("%Y-%m-%d"); D["month"] = D.ts.dt.strftime("%Y-%m")
    D["hour"] = D.ts.dt.hour
    D["mcap"] = D.vol24 / D.R_ent
    import study_yr6wl_frenzy_common as YC
    T5 = YC.top50_pairdays()
    D["top50"] = [(d, p) in T5 for d, p in zip(D.day, D.pair)]
    D["tick"] = D.tick90.fillna(False).astype(bool) & D.why.isin(["TP", "LATE_TP", "STOP", "CAP"])
    return D


def alt():
    A = pd.read_pickle(W.ALT)
    R = pd.read_pickle(os.path.join(W.OUT, "priced.pkl"))
    A = A.merge(R, on=["pair", "T"], how="left")
    A["era"] = [W.era_of(x) for x in A.T0]
    A["day"] = pd.to_datetime(A["T"], unit="ms").dt.strftime("%Y-%m-%d")
    A["tick"] = A.tick90.fillna(False).astype(bool) & A.why.isin(["TP", "LATE_TP", "STOP", "CAP"])
    S = pd.read_pickle(W.SIG)[["pair", "spike_ts", "R_red1", "end", "T_ent"]]
    A = A.merge(S, on=["pair", "spike_ts"], how="left", suffixes=("", "_sig"))
    sup = W.supply_table()
    A["R"] = [v / (sup[p] * c) if p in sup and c > 0 else np.nan for p, v, c in zip(A.pair, A.vol24, A.c)]
    return A


def ctx(kind, pair, a=None):
    fs = glob.glob(os.path.join(W.OUT, "ctx", f"{kind}_{pair}_{a if a is not None else '*'}.csv"))
    fr = [pd.read_csv(f) for f in fs]
    fr = [f for f in fr if len(f)]
    return pd.concat(fr).drop_duplicates("open_time").set_index("open_time").sort_index() if fr else None


def ms_block():
    """master 19 + live 8: stamps + fetched context → the same feature names as BT; outcome as traded and today."""
    rows = []
    LW = pd.read_csv(W.LOWVWAP_ROWS, low_memory=False)
    LW = LW[LW.block == "MASTER"]
    LV = pd.read_csv(W.LOWVWAP_LIVE)
    sup = W.supply_table()
    import study_willy_theory_market as MK
    tm = MK.top50_map(); days = sorted(tm)
    for f, blk in ((W.MASTER, "MASTER"), (W.LIVE, "LIVE")):
        d = pd.read_csv(f, low_memory=False)
        d = d[d.entry_strategy == "FRENZY_WILLY"]
        for r in d.itertuples():
            te = int(W.ms(pd.Series([str(r.opened_at)[:19]]))[0])
            sp = int(W.ms(pd.Series([str(r.entry_frenzy_spike_at)[:19]]))[0])
            T = (te // BAR) * BAR - BAR
            o = dict(block=blk, batch=(getattr(r, "era", None) if blk == "MASTER" else "LIVE"), pair=r.pair, opened_at=str(r.opened_at)[:19],
                     te=te, T=T, spike_ts=sp, pct_traded=float(r.pnl_percentage), usd_traded=float(r.pnl),
                     ent_bar_ret=r.entry_frenzy_bar_ret_pct, wait_bars=r.entry_frenzy_willy_wait_bars, vs_vwap=r.entry_frenzy_vs_vwap_pct,
                     run_pct=r.entry_frenzy_run_pct, vol_mult=r.entry_frenzy_vol_mult, hours=r.entry_frenzy_hours,
                     R_ent=r.entry_vol_mcap_ratio, mcap=r.entry_mcap_usd, btc_1h_slope=r.entry_btc_1h_slope,
                     close_reason=r.close_reason)
            o["usd_pt"] = o["usd_traded"] / o["pct_traded"] if abs(o["pct_traded"]) > 1e-9 else np.nan
            o["min_from_spike"] = (T + BAR - sp) / MIN
            o["day"] = pd.Timestamp(te, unit="ms").strftime("%Y-%m-%d"); o["hour"] = pd.Timestamp(te, unit="ms").hour
            lst = W.listing_ms()
            o["listing_days"] = (T - lst[r.pair]) / W.DAY_MS if r.pair in lst else np.nan
            # today's outcome
            if blk == "MASTER":
                m = LW[(LW.pair == r.pair) & (LW.opened_at.astype(str).str[:16] == o["opened_at"][:16])]
                o["y"] = float(m.y.iloc[0]) if len(m) else np.nan
                o["why"] = m.why.iloc[0] if len(m) else None
            else:
                m = LV[(LV.pair == r.pair) & (LV.opened.str[:11] == pd.Timestamp(te, unit="ms").strftime("%m-%dT%H:%M")[:11])]
                o["y"] = float(m.k1m_today_pct.iloc[0]) if len(m) else np.nan
                o["why"] = m.k1m_today_why.iloc[0] if len(m) else None
            o["usd_today"] = o["y"] * o["usd_pt"]
            # pair context (fetched 5m with taker)
            p5 = ctx("p5", r.pair)
            if p5 is not None:
                c, op, h, l, q = p5.c, p5.o, p5.h, p5.l, p5.qvol
                on = sp - BAR
                if on in p5.index and (on - 6 * BAR) in p5.index:
                    seq = [on - k * BAR for k in range(6, -1, -1)]
                    cc = c.reindex(seq).values
                    br = (cc[1:] / cc[:-1] - 1) * 100; pos = br[br > 0]
                    o["spike_ret6"] = (cc[-1] / cc[0] - 1) * 100
                    o["spike_conc"] = br.max() / pos.sum() if len(pos) and pos.sum() > 0 else np.nan
                    rg = h[on] - l[on]
                    o["spike_uwick"] = (h[on] - max(op[on], c[on])) / rg if rg > 0 else np.nan
                    base = l.reindex([on - k * BAR for k in range(18, 5, -1)]).min()
                    if T in p5.index:
                        hs = h.loc[on:T].max()
                        o["off_high"] = (c[T] / hs - 1) * 100
                        o["red_vs_spike_q"] = q[T] / q[on] if q[on] > 0 else np.nan
                        o["vs_base_pct"] = (c[T] / base - 1) * 100
                        o["retrace_frac"] = (hs - c[T]) / (hs - base) if hs > base else np.nan
                        o["tbs_red"] = p5.tbq[T] / q[T] * 100 if q[T] > 0 else np.nan
                        o["tbs_spk"] = p5.tbq[on] / q[on] * 100 if q[on] > 0 else np.nan
                        o["spike_volx"] = np.nan
            kt = None
            fs = glob.glob(os.path.join(W.OUT, "k1m_taker", f"{r.pair}_*.csv"))
            if fs:
                kt = pd.concat([pd.read_csv(x) for x in fs]).drop_duplicates("open_time").set_index("open_time").sort_index()
                tf = taker_feats(kt, sp - BAR, T)
                o.update({k: v for k, v in tf.items() if k in ("tbs_red_last1m", "tbs_red_trend") or k not in o})
            # market (fetched 1m BTC / ETH / SOL, top-50 5m)
            mk = {}
            for m_, nm in (("BTCUSDT", "btc"), ("ETHUSDT", "eth"), ("SOLUSDT", "sol")):
                k1 = ctx("m1", m_, T - 60 * MIN)
                if k1 is None:
                    continue
                mk[nm] = k1
                b = k1.reindex([T + i * MIN for i in range(5)])
                if b.c.notna().all():
                    o[f"{nm}_bar"] = (b.c.iloc[-1] / b.o.iloc[0] - 1) * 100
                if nm == "btc":
                    cl = k1.c
                    tcl = T + BAR - MIN           # the red bar's last minute (close = T + 5 min)
                    def cat(off):
                        return cl.get(tcl + off * MIN, np.nan)
                    o["btc_15m"] = (cat(0) / cat(-15) - 1) * 100
                    o["btc_1h"] = (cat(0) / cat(-60) - 1) * 100 if (tcl - 60 * MIN) in cl.index else np.nan
                    for kk, lab in ((5, "f5"), (15, "f15"), (30, "f30")):
                        o[f"btc_{lab}"] = (cat(kk) / cat(0) - 1) * 100
            if mk:
                tb = sum(v.reindex([T + i * MIN for i in range(5)])[["qvol", "tbq"]].sum() for v in mk.values())
                o["mkt_tbs"] = tb.tbq / tb.qvol * 100 if tb.qvol > 0 else np.nan
            dprev = (te // W.DAY_MS) * W.DAY_MS - W.DAY_MS
            kd = np.searchsorted(days, dprev, side="right") - 1
            red, up, f = [], [], {"f5": [], "f15": [], "f30": []}
            for p in (tm[days[kd]] if kd >= 0 else []):
                u = ctx("u5", p, T - 60 * MIN)
                if u is None or T not in u.index:
                    continue
                red.append(u.c[T] < u.o[T])
                if (T - 12 * BAR) in u.index:
                    up.append(u.c[T] > u.c[T - 12 * BAR])
                for kk, lab in ((1, "f5"), (3, "f15"), (6, "f30")):
                    if (T + kk * BAR) in u.index:
                        f[lab].append((u.c[T + kk * BAR] / u.c[T] - 1) * 100)
            o["brd_red"] = np.mean(red) * 100 if len(red) >= 20 else np.nan
            o["brd_1h"] = np.mean(up) * 100 if len(up) >= 20 else np.nan
            for lab in f:
                o[f"brd_{lab}"] = np.mean(f[lab]) if len(f[lab]) >= 20 else np.nan
            o["top50"] = bool(kd >= 0 and r.pair in tm[days[kd]])
            rows.append(o)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    B = bt(); B.to_pickle(os.path.join(W.OUT, "bt.pkl"))
    print(f"BT {len(B):,} · tick-verdict rows {B.tick.sum():,} · MATCH {B.MATCH.sum():,} · taker red scored {B.tbs_red.notna().sum():,}")
    A = alt(); A.to_pickle(os.path.join(W.OUT, "alt.pkl"))
    print(f"ALT {len(A):,} · tick {A.tick.sum():,}")
    MS = ms_block(); MS.to_pickle(os.path.join(W.OUT, "ms.pkl"))
    print(MS[["block", "batch", "pair", "opened_at", "pct_traded", "usd_traded", "y", "why", "usd_today", "ent_bar_ret", "tbs_red", "btc_bar", "brd_red"]].to_string())
