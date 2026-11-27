#!/usr/bin/env python3
"""SHORT_BTC_OFF24LO_2026-11-27 — do the short sleeves lose when BTC is bouncing off its 24 h low? (operator 11-27: the batch's Flatbase
losers came with BTC +2.7 % above its 24 h low vs winners +1.1 %). Research only.
Variable = the bot's own stamp entry_btc_off24lo_pct (services/trading_engine.py: BTC price ÷ min low of the prior 288 5m bars − 1), rebuilt
on BTC 5m closes at the bar before each entry. Never tested before on FB / EXH (OOS_REAL_FLATBASE tested BTC 4 h / 24 h returns, not this).
Theory: a fresh BTC relief bounce lifts alts and squeezes crowded shorts on pumped coins → shorts entered while BTC is well off its low
lose more. Buckets < 1 · 1–2 · 2–3 · ≥ 3 %; DAY counts; DESIGN (< 2025-07) / HOLDOUT; half-years; FB = 545 rebuild at today's exit;
EXH = TP_GRID F7 kept (3,358); real FB = master kept + batch (34) from the stamp.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(HERE); sys.path.insert(0, HERE)
import study_frenzy_swing_short_walk as W  # noqa: E402
HOL = int(pd.Timestamp("2025-07-01").value // 10**6)
BINS = [-1, 1, 2, 3, 100]


def btc_off24lo():
    t, o, h, l_, c = W.load("BTCUSDT")[:5]
    lo = pd.Series(l_).rolling(288).min().values
    return t, (c / lo - 1) * 100          # at bar k's close: close / min low of bars k−287 … k


def at(entry_ms, t, v):
    k = np.searchsorted(t, np.asarray(entry_ms) // 300_000 * 300_000) - 1     # the last bar CLOSED before the entry
    out = np.where((k >= 0) & (k < len(t)), v[np.clip(k, 0, len(v) - 1)], np.nan)
    out[(np.asarray(entry_ms) - t[np.clip(k, 0, len(t) - 1)]) > 3_600_000] = np.nan      # BTC data gap → unscored
    return out


def tab(K, name):
    K = K.assign(b=pd.cut(K.v, BINS))
    g = K.groupby("b", observed=True).agg(N=("net", "size"), days=("day", "nunique"), WR=("net", lambda x: (x > 0).mean() * 100), avg=("net", "mean"))
    for blk, m in (("DESIGN", K.entry_ms < HOL), ("HOLDOUT", K.entry_ms >= HOL)):
        g[blk] = K[m].groupby("b", observed=True).net.mean()
    g["halves_pos"] = [f"{(K[K.b == b].groupby('half').net.mean() > 0).sum()}/{K[K.b == b].half.nunique()}" for b in g.index]
    print(f"\n{name} (scored {K.v.notna().sum()} of {len(K)})\n" + g.round(2).to_string())
    return g.assign(sleeve=name)


def main():
    t, v = btc_off24lo()
    print("BTC 5m data to", pd.to_datetime(t[-1], unit="ms"))
    out = []
    b = pd.read_csv(os.path.join(R, "reports", "FLATBASE_NOPROGRESS_STOP_2026-11-27_data", "base.csv"))
    b = b[b.src == "BT"].rename(columns={"te": "entry_ms"})
    b["day"] = b.entry_ms // 86_400_000
    tt = pd.to_datetime(b.entry_ms, unit="ms"); b["half"] = tt.dt.year.astype(str) + "-H" + np.where(tt.dt.month <= 6, "1", "2")
    b["v"] = at(b.entry_ms.values, t, v)
    out.append(tab(b.dropna(subset=["v"]), "FLATBASE backtest (today's exit)"))
    sp = importlib.util.spec_from_file_location("sa", os.path.join(HERE, "EXH_SPIKE_AGE_2026-11-26.py"))
    SA = importlib.util.module_from_spec(sp); sp.loader.exec_module(SA)
    D, _ = SA.load_bt(); K = SA.kept(D)
    K["v"] = at(K.entry_ms.values, t, v)
    out.append(tab(K.dropna(subset=["v"]), "EXH backtest (TP 7)"))
    pd.concat(out).to_csv(os.path.join(R, "reports", "BATCH28_REVIEW_2026-11-27_data", "short_btc_off24lo.csv"))


if __name__ == "__main__":
    main()
