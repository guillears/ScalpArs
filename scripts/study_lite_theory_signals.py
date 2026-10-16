#!/usr/bin/env python3
"""LITE THEORY — step 2: the LITE signal emulator (today's rule + the pre-registered definition variants) on the rebuilt bars, and the
PARITY check against the replay's LITE fills (no outcome is read here).

emulate() mirrors services.frenzy.frenzy_lite_status' order (flag → not in state → streak ≥ need → vol < vmax → hours window → stretch
not judged → 24 h volume (not judging) → candle colour (judging)). For today's parameters every bar's code is ASSERTED equal to the
engine's own frenzy_lite_status (called on the same values). Bearish-day: services.frenzy.frenzy_bearish_block on BTC's last closed daily
return + BTC 5m EMA13−EMA50 gap (rebuilt; parity vs the fills' own stamps reported).
"""
import os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_lite_theory_common as T  # noqa: E402
from services.frenzy import frenzy_lite_status, frenzy_bearish_block, FRENZY_LITE_JUDGES  # noqa: E402

TH = T.th()
TODAY = dict(need=12, hmin=2.0, hmax=16.75, vmax=100.0, color="red", v24=20e6)


def btc_bear_table():
    """per 5m bar open T: btc_1d_ret (last closed UTC day vs the day before, at scan time T + 5 min) and btc trend gap (EMA13−EMA50 %)."""
    fr = [pd.read_csv(f) for f in (os.path.join(T.C26, "btc_5m.csv"), os.path.join(T.C25, "k5m_full", "BTCUSDT.csv"),
                                   os.path.join(T.C26, "k5m_full", "BTCUSDT.csv"), os.path.join(T.C26, "k5m", "BTCUSDT.csv"),
                                   os.path.join(T.C26, "k5m_ext", "BTCUSDT.csv")) if os.path.exists(f)]
    b = pd.concat(fr).drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    c = b.c.astype(float)
    e13 = c.ewm(span=13, adjust=False).mean(); e50 = c.ewm(span=50, adjust=False).mean()
    gap = (e13 - e50) / e50 * 100
    day = b.open_time // 86_400_000
    dc = b.groupby(day).c.last()
    dret = (dc / dc.shift(1) - 1) * 100            # return of day D (close D vs close D−1)
    scan_day = (b.open_time + T.BAR_MS + T.T_LAG_MS) // 86_400_000
    r1d = dret.reindex(scan_day - 1).values         # last CLOSED day at scan time
    return pd.DataFrame(dict(T=b.open_time.values.astype("int64"), btc_1d=r1d, btc_gap=gap.values))


def emulate(G, need=12, hmin=2.0, hmax=16.75, vmax=100.0, color="red", v24=20e6, check=False):
    """G = one pair's bars sorted by T. → list of dict(T, code, sid). code: READY (enter) / REFUSED_COLOR (judged, no entry) /
    VOL24 (not judged). Only judging / entering bars are returned. check=True asserts code parity with the engine (today's params)."""
    out = []
    judged = None
    Tm = G["T"].values; st = G.streak.values; vm = G.vol_mult.values; hr = G.hours.values; ins = G.in_state.values
    red = G.bar_red.values; v24a = G.vol24.values; br = G.bar_ret.values
    for i in range(len(G)):
        sid = int(Tm[i]) - (int(st[i]) - 1) * T.BAR_MS if st[i] >= 1 else None
        code = None
        if ins[i]:
            code = "IN_STATE"
        elif st[i] < need:
            code = "NOT_ABOVE"
        elif not (vm[i] < vmax):
            code = "VOL_HIGH"
        elif hr[i] < hmin:
            code = "TOO_EARLY"
        elif hr[i] > hmax:
            code = "TOO_LATE"
        elif judged is not None and judged >= sid:
            code = "DONE"
        elif not (np.isfinite(v24a[i]) and v24a[i] >= v24):
            code = "VOL24"
        else:
            ok = {"red": bool(red[i]), "green": not bool(red[i]), "any": True}[color]
            code = "READY" if ok else "COLOR"
        if check:
            ep = dict(verified=True, hours=float(hr[i]), in_state=bool(ins[i]), above_streak=int(st[i]), vol_mult=float(vm[i]),
                      last_bar_ts=int(Tm[i]), bar_red=bool(red[i]), bar_ret_pct=float(br[i]) if np.isfinite(br[i]) else None)
            _, ec, _ = frenzy_lite_status(ep, TH, float(v24a[i]) if np.isfinite(v24a[i]) else None, judged)
            m = {"FRENZY_LITE_IN_STATE": "IN_STATE", "FRENZY_LITE_NOT_ABOVE": "NOT_ABOVE", "FRENZY_LITE_VOL_HIGH": "VOL_HIGH",
                 "FRENZY_LITE_TOO_EARLY": "TOO_EARLY", "FRENZY_LITE_TOO_LATE": "TOO_LATE", "FRENZY_LITE_STRETCH_DONE": "DONE",
                 "FRENZY_LITE_VOL24_LOW": "VOL24", "FRENZY_LITE_GREEN_BAR": "COLOR", "FRENZY_LITE_READY": "READY"}[ec]
            assert m == code, (G.pair.iloc[0], int(Tm[i]), m, code)
        if code in ("READY", "COLOR"):
            judged = sid
            out.append(dict(T=int(Tm[i]), code=code, sid=sid))
    return out


def signals(B, check=False, **kw):
    res = []
    for p, G in B.groupby("pair", sort=False):
        G = G.sort_values("T")
        for r in emulate(G, check=check, **kw):
            r["pair"] = p
            res.append(r)
    return pd.DataFrame(res)


def load_bars():
    B = pd.read_pickle(T.BARS).sort_values(["pair", "T"]).reset_index(drop=True)
    bt = btc_bear_table()
    B = B.merge(bt, on="T", how="left")
    B["bear"] = [frenzy_bearish_block(a if np.isfinite(a) else None, g if np.isfinite(g) else None, TH) == "BEARISH_DAY"
                 for a, g in zip(B.btc_1d.values, B.btc_gap.values)]
    B["year"] = np.where(B["T"] < T.SPLIT26, "2025", "2026")
    B["era"] = [T.era_of(x) for x in B["T"].values]
    B["ep"] = B.pair + "|" + B.spike_ts.astype(str)
    B["day"] = pd.to_datetime(B["T"], unit="ms").dt.strftime("%Y-%m-%d")
    B["month"] = B.day.str[:7]
    return B


def replay_fills():
    F = pd.read_csv(T.FILLS, low_memory=False, usecols=lambda c: c in ("sleeve", "seed", "pair", "t", "entry_price", "entry_frenzy_spike_at",
                                                                       "entry_frenzy_hours", "entry_frenzy_above_streak",
                                                                       "entry_frenzy_vol_mult", "entry_frenzy_vs_vwap_pct",
                                                                       "entry_btc_1d_ret_pct", "entry_btc_trend_gap_pct"))
    F = F[(F.sleeve == "FRENZY_LITE") & (F.seed == 1)].copy()
    F["te"] = pd.to_datetime(F.t).values.astype("datetime64[ms]").astype("int64")
    F["T"] = (F.te // T.BAR_MS) * T.BAR_MS - T.BAR_MS
    return F


def parity(B, S):
    F = replay_fills()
    R = S[S.code == "READY"][["pair", "T"]].assign(sig=True)
    M = F.merge(R, on=["pair", "T"], how="left")
    M["sig"] = M.sig.fillna(False).astype(bool)
    Bk = B[["pair", "T", "streak", "hours", "vol_mult", "vs_vwap", "spike_ts", "btc_1d", "btc_gap", "bear"]]
    M = M.merge(Bk, on=["pair", "T"], how="left")
    return M


if __name__ == "__main__":
    B = load_bars()
    S = signals(B, check=True, **TODAY)
    S.to_pickle(os.path.join(T.OUT, "signals_today.pkl"))
    M = parity(B, S)
    M.to_csv(os.path.join(T.OUT, "parity_rows.csv"), index=False)
    n = len(M); k = int(M.sig.sum())
    print(f"engine-code parity asserted on {len(B):,} bars (emulator == services.frenzy.frenzy_lite_status)")
    print(f"replay LITE fills (seed 1): {n} · reproduced by a rebuilt READY signal on the same pair & bar: {k} = {k / n * 100:.1f}%")
    for y in ("2025", "2026"):
        mm = M[np.where(M["T"] < T.SPLIT26, "2025", "2026") == y]
        print(f"  {y}: {int(mm.sig.sum())}/{len(mm)} = {mm.sig.mean() * 100:.1f}%")
    mb = M[M.streak.notna()]
    print(f"fills whose bar exists in the rebuild: {len(mb)} · streak equal {np.mean(mb.streak == mb.entry_frenzy_above_streak) * 100:.1f}% · "
          f"spike equal {np.mean(pd.to_datetime(mb.spike_ts - 0, unit='ms') == pd.to_datetime(mb.entry_frenzy_spike_at)) * 100:.1f}% · "
          f"|Δvs_vwap| median {np.nanmedian(abs(mb.vs_vwap - mb.entry_frenzy_vs_vwap_pct)):.4f} · vol_mult ratio median "
          f"{np.nanmedian(mb.vol_mult / mb.entry_frenzy_vol_mult):.3f}")
    print(f"bearish flag on fills (should be False): {int(mb.bear.sum())} of {len(mb)} · btc_1d sign agree "
          f"{np.mean(np.sign(mb.btc_1d) == np.sign(mb.entry_btc_1d_ret_pct)) * 100:.1f}% · gap sign agree "
          f"{np.mean(np.sign(mb.btc_gap) == np.sign(mb.entry_btc_trend_gap_pct)) * 100:.1f}%")
    print(f"rebuilt READY signals: {int((S.code == 'READY').sum())} (of which not bearish: "
          f"{int((S[S.code == 'READY'].merge(B[['pair', 'T', 'bear']], on=['pair', 'T']).bear == False).sum())}) · COLOR-refused "
          f"{int((S.code == 'COLOR').sum())}")
    miss = M[~M.sig]
    print("misses: no bar in rebuild", int(miss.streak.isna().sum()), "· bar present but no READY", int(miss.streak.notna().sum()))
    print(miss[miss.streak.notna()][["pair", "t", "streak", "entry_frenzy_above_streak", "hours", "entry_frenzy_hours", "vol_mult",
                                     "entry_frenzy_vol_mult"]].head(15).to_string())
