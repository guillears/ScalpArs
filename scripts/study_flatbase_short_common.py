"""FRENZY flat-base short study (2026-10-22) — shared helpers (loader, constants). Pre-registration:
reports/FRENZY_FLATBASE_SHORT_2026-10-22_PREREG.md. READ-ONLY on the bot.

The engine episodes come from scripts/study_frenzy_swing_short_walk.py UNCHANGED (96.5 % bar parity with services.frenzy.frenzy_walk);
this module only appends the new data folder (klines to 2026-10-22 + delisted pairs) to its source list.
"""
import glob, os, sys
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "scripts"))
import study_frenzy_swing_short_walk as W  # noqa: E402

D = os.path.join(R, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_data")
OUTP = os.path.join(R, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22")
K5 = os.path.join(D, "k5m")
if K5 not in W.SRC:
    W.SRC.append(K5)
BAR, HR, DAY = 300_000, 3_600_000, 86_400_000
T0_FEAT = pd.Timestamp("2024-06-01").value // 10**6
SLIP, FEES = 0.0005, 0.10
FUND_MISSING = -0.10
HOLDS_D = [3, 7, 14]
TPS = ["none", "base"]
STOPS = [0.0, 25.0, 50.0]            # 0 = no stop
LEVS = [1, 2, 3]
ENTRIES = ["S1", "S2", "S3", "S4"]
SEARCH = 96 * 12                      # signals within spike → spike + 96 h
EXITS = [(h, tp, s, lv) for h in HOLDS_D for tp in TPS for s in STOPS for lv in LEVS]   # 54
PRIMARY = ("S2", 7, "base", 25.0, 1)
SECONDARY = ("S3", 7, "base", 25.0, 1)
B_LIVE0, B_MAST0, B_YEAR0, B_YEAR1 = pd.Timestamp("2026-10-14"), pd.Timestamp("2026-10-02"), pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-02")
FREEZE_CUT = pd.Timestamp("2025-07-01")


def cell_name(e, h, tp, s, lv):
    return f"{e}|{h}d|tp_{tp}|stop{int(s)}|{lv}x"


def load_full(pair):
    """the full 5m grid from 2024-06 (same sources / precedence / gap fill as the walk's loader, earlier start for the features)."""
    parts = []
    for i, d in enumerate(W.SRC):
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f):
            x = pd.read_csv(f)
            if i == 0:
                x = x[x.open_time < pd.Timestamp("2026-01-10").value // 10**6]
            parts.append(x)
    if not parts:
        return None
    d = pd.concat(parts).drop_duplicates("open_time", keep="last").sort_values("open_time")
    d = d[d.open_time >= T0_FEAT]
    # settled / delisted contracts keep publishing flat zero-volume bars after the last trade: cut the trailing zero-volume tail (≥ 1 h)
    nz = np.nonzero(d.vol.values > 0)[0]
    if len(nz) and len(d) - 1 - nz[-1] >= 12:
        d = d.iloc[: nz[-1] + 1]
    if len(d) < 3000:
        return None
    grid = np.arange(d.open_time.iloc[0], d.open_time.iloc[-1] + 1, BAR, dtype="int64")
    d = d.set_index("open_time").reindex(grid)
    real = d.c.notna().values
    c = d.c.ffill().values.astype(float)
    o = d.o.fillna(pd.Series(c, index=d.index)).values.astype(float)
    h = d.h.fillna(pd.Series(c, index=d.index)).values.astype(float)
    l = d.l.fillna(pd.Series(c, index=d.index)).values.astype(float)
    v = d.vol.fillna(0).values.astype(float); qv = d.qvol.fillna(0).values.astype(float)
    return grid, o, h, l, c, v, qv, real


def funding_table(pair):
    parts = []
    for d in (os.path.join(R, "reports", "backtest_cache", "funding"), os.path.join(R, "reports", "FRENZY_SWING_SHORT_2026-10-20_data", "funding"),
              os.path.join(D, "funding")):
        f = os.path.join(d, pair + ".csv")
        if os.path.exists(f):
            try:
                x = pd.read_csv(f)
                if len(x):
                    parts.append(x[["t", "rate"]])
            except Exception:
                pass
    if not parts:
        return None
    x = pd.concat(parts).drop_duplicates("t").sort_values("t")
    return x.t.values.astype("int64"), x.rate.values.astype(float)


def funding_sum(F, t0, t1):
    """% of notional the SHORT receives over settlements in (t0, t1]; vectorised; missing coverage → −0.10 %/8 h assumed."""
    t0 = np.asarray(t0, dtype="int64"); t1 = np.asarray(t1, dtype="int64")
    if F is None or not len(F[0]):
        return (t1 - t0) / (8 * HR) * FUND_MISSING, np.ones(len(t0), bool)
    ts, rt = F
    cs = np.r_[0.0, np.cumsum(rt)]
    a = np.searchsorted(ts, t0, side="right"); b = np.searchsorted(ts, t1, side="right")
    s = (cs[b] - cs[a]) * 100
    miss = (ts[0] > t0) | (ts[-1] < t1 - 9 * HR)
    s = np.where(miss, (t1 - t0) / (8 * HR) * FUND_MISSING, s)
    return s, miss
