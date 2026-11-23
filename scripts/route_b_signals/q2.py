"""Route B signal-level loader — the Q2 THEORY-LINE variables of the weekly-scan PREREG §4 (build item 7), built exactly as written there.
Pure on a state.Coin + the state dict at the signal (no look-ahead: every read ends at the state minute s; the coin "p90 / p99" and
"prior median" are over rows BEFORE s). Values: 1.0 = the condition holds · 0.0 = it does not · NaN = UNSCORED (an input missing / a
distribution too thin) — never "rest". STRICT rule for every AND / OR line (S1b, S2a, S2c, S5): a line is scored only when EVERY leg is
scored (no leg decides alone), so coverage never decides the value (review: the Kleene short-cut put non-hot coins only in one group).

Definitions (PREREG §4, frozen 2026-11-23 15:01 UTC):
  p90 / p99 = the coin's own distribution over the 7 days before the state minute, ≥ 20 non-zero values, else unscored. The sample =
      the coin's w-min liquidation sums (minutes in (m − w, m]) at every minute m < s that had a liquidation of that coin (the
      state.Coin.liq5_series convention, generalised to w), non-zero values only.
  prior median = over the 24 h before the full-book row the state read (rows in [row − 24 h, row), that row excluded — the same rule as
      the trigger depth condition, triggers.Depth), ≥ 60 recorded minutes, else unscored. The fb_*_usd_05_lb lower-bound flag is kept
      beside it (q2_fb_*_05_lb) for the audit.
  S1-a  EXH       5-min short liquidations ≥ p99 of non-zero 5-min sums (and > 0)                          dir −
  S1-b  EXH       oi_chg_15m > 0 AND mark return 15 min (rb_ret_15m_pct) > 0                                dir −  HOLDOUT-only
  S1-c  EXH       taker-buy share 15 min (rb_taker_buy_share_15) ≤ 0.45                                     dir +  HOLDOUT-only
  S2-a  DIST      rb_ret_15m_pct ≤ −10 AND 15-min long liquidations ≥ p99 of non-zero 15-min sums (and > 0) dir −  HOLDOUT-only
  S2-b  DIST      full-book ask $ within 0.5 % (fb_ask_usd_05) < its prior median                           dir −
  S2-c  DIST      funding ≤ the lower tercile of DIST DESIGN rows AND global_ls_acct ≤ its lower tercile on the same rows — the cut
                  needs the DESIGN cohort, so it is computed at the 12-10 scan with s2c_cuts() / s2c() below (NOT in the loader)
  S3-a  FLATBASE  oi_chg_240m > 0                                                                           dir −
  S3-b  FLATBASE  240-min short liquidations ≥ p90 of non-zero 240-min sums (and > 0)                       dir −
  S4-a  WILLY_HT  fb_bid_usd_05 ≥ its prior median                                                          dir +
  S4-b  WILLY_HT  rb_taker_buy_share_5 ≥ 0.55                                                               dir +  HOLDOUT-only
  S5    EXH_NO_SPOT 15-min short liquidations ≥ p90 of non-zero 15-min sums (and > 0) OR oi_chg_15m > 0     dir −
Every line is computed on every signal row (the scan selects the sleeve); the components are kept for the audit (q2_* inputs).
"""
import math

import numpy as np
import pandas as pd

MIN_MS = 60_000
DAY_MS = 1440 * MIN_MS
P_DAYS = 7
P_MIN_N = 20
MED_MIN_N = 60
LINES = ("S1a", "S1b", "S1c", "S2a", "S2b", "S3a", "S3b", "S4a", "S4b", "S5")
DIRECTION = {"S1a": -1, "S1b": -1, "S1c": +1, "S2a": -1, "S2b": -1, "S2c": -1, "S3a": -1, "S3b": -1, "S4a": +1, "S4b": +1, "S5": -1}
HOLDOUT_ONLY = ("S1b", "S1c", "S2a", "S4b")
SLEEVE = {"S1a": "EXHAUSTION_SHORT", "S1b": "EXHAUSTION_SHORT", "S1c": "EXHAUSTION_SHORT", "S2a": "DISTRIBUTION_SHORT",
          "S2b": "DISTRIBUTION_SHORT", "S2c": "DISTRIBUTION_SHORT", "S3a": "FLATBASE_SHORT", "S3b": "FLATBASE_SHORT",
          "S4a": "FRENZY_WILLY_HT", "S4b": "FRENZY_WILLY_HT", "S5": "EXHAUSTION_SHORT:EXH_NO_SPOT"}


def _ok(v):
    return v is not None and v == v


def liq_series(coin, side, w):
    """(minutes with a liquidation, the w-min sum of `side` ending at each) — cached on the coin."""
    cache = coin.__dict__.setdefault("_q2_liq", {})
    key = (side, w)
    if key not in cache:
        m = coin.lm
        cum = coin.cll if side == "long" else coin.cls
        j = np.searchsorted(m, m, "right")
        i = np.searchsorted(m, m - w * MIN_MS, "right")
        cache[key] = (m, cum[j] - cum[i])
    return cache[key]


def liq_now(coin, side, w, s):
    cum = coin.cll if side == "long" else coin.cls
    return coin._win(coin.lm, cum, s - w * MIN_MS, s)[0]


def liq_pct_before(coin, side, w, s, q):
    """the q-quantile of the coin's non-zero w-min liquidation sums at liquidation minutes in [s − 7 d, s); NaN with < 20 values."""
    m, v = liq_series(coin, side, w)
    a, b = np.searchsorted(m, s - P_DAYS * DAY_MS, "left"), np.searchsorted(m, s, "left")
    x = v[a:b]
    x = x[x > 0]
    return float(np.quantile(x, q)) if len(x) >= P_MIN_N else math.nan


def burst(coin, side, w, s, q, covered=None):
    """(1.0 / 0.0 / NaN, current w-min sum, threshold). covered(a, b) → False when the recorder was down in a minute of (a, b]: the
    liquidation stream only stores non-zero minutes, so an outage would otherwise read as "no burst" (unscored instead)."""
    if coin is None or (covered is not None and not covered(s - w * MIN_MS, s)):
        return math.nan, math.nan, math.nan
    p = liq_pct_before(coin, side, w, s, q)
    now = liq_now(coin, side, w, s)
    if p != p:
        return math.nan, now, p
    return float(now > 0 and now >= p), now, p


def fb_prior_median(coin, col, row_minute):
    """median of the coin's full-book `col` over rows with minute in [row − 24 h, row) (the read row excluded); NaN with < 60 non-NaN
    values."""
    if coin is None or row_minute is None or row_minute != row_minute or not len(getattr(coin, "fm", [])) or col not in coin.fb.columns:
        return math.nan
    s = int(row_minute)
    a, b = np.searchsorted(coin.fm, s - DAY_MS, "left"), np.searchsorted(coin.fm, s, "left")
    x = pd.to_numeric(coin.fb[col].iloc[a:b], errors="coerce").dropna().to_numpy(float)
    return float(np.median(x)) if len(x) >= MED_MIN_N else math.nan


def _cmp(v, thr, op):
    if not (_ok(v) and _ok(thr)):
        return math.nan
    return float(op(v, thr))


def _and(*legs):
    """strict AND: NaN unless every leg is scored."""
    return float(all(x == 1.0 for x in legs)) if all(_ok(x) for x in legs) else math.nan


def _or(*legs):
    """strict OR: NaN unless every leg is scored."""
    return float(any(x == 1.0 for x in legs)) if all(_ok(x) for x in legs) else math.nan


BURSTS = {"S1a": ("short", 5, 0.99), "S2a": ("long", 15, 0.99), "S3b": ("short", 240, 0.90), "S5": ("short", 15, 0.90)}


def lines(coin, s, r, covered=None):
    """{q2_<line>: 1.0 / 0.0 / NaN} + components, for one signal at state minute s with state dict r. covered = Market.covered (a
    liquidation window overlapping a recorder outage is unscored). Pure on its inputs."""
    o = {}
    sd, w, q = BURSTS["S1a"]
    s1a, o["q2_liq_short_5"], o["q2_liq_short_5_p99_7d"] = burst(coin, sd, w, s, q, covered)
    o["q2_S1a"] = s1a
    oi15, ret15 = r.get("oi_chg_15m"), r.get("rb_ret_15m_pct")
    oi_up = float(oi15 > 0) if _ok(oi15) else math.nan
    o["q2_S1b"] = _and(oi_up, float(ret15 > 0) if _ok(ret15) else math.nan)
    o["q2_S1c"] = _cmp(r.get("rb_taker_buy_share_15"), 0.45, lambda a, b: a <= b)
    sd, w, q = BURSTS["S2a"]
    b15l, o["q2_liq_long_15"], o["q2_liq_long_15_p99_7d"] = burst(coin, sd, w, s, q, covered)
    o["q2_S2a"] = _and(float(ret15 <= -10) if _ok(ret15) else math.nan, b15l)
    age = r.get("fb_age_min")
    row = (s - int(age) * MIN_MS) if (r.get("has_fb") is True and _ok(age)) else None
    med_ask = fb_prior_median(coin, "fb_ask_usd_05", row)
    o["q2_fb_ask_05_prior_median"] = med_ask
    o["q2_fb_ask_05_lb"] = r.get("fb_ask_usd_05_lb", math.nan)
    o["q2_S2b"] = _cmp(r.get("fb_ask_usd_05"), med_ask, lambda a, b: a < b)
    oi240 = r.get("oi_chg_240m")
    o["q2_S3a"] = float(oi240 > 0) if _ok(oi240) else math.nan
    sd, w, q = BURSTS["S3b"]
    o["q2_S3b"], o["q2_liq_short_240"], o["q2_liq_short_240_p90_7d"] = burst(coin, sd, w, s, q, covered)
    med_bid = fb_prior_median(coin, "fb_bid_usd_05", row)
    o["q2_fb_bid_05_prior_median"] = med_bid
    o["q2_fb_bid_05_lb"] = r.get("fb_bid_usd_05_lb", math.nan)
    o["q2_S4a"] = _cmp(r.get("fb_bid_usd_05"), med_bid, lambda a, b: a >= b)
    o["q2_S4b"] = _cmp(r.get("rb_taker_buy_share_5"), 0.55, lambda a, b: a >= b)
    sd, w, q = BURSTS["S5"]
    b15s, o["q2_liq_short_15"], o["q2_liq_short_15_p90_7d"] = burst(coin, sd, w, s, q, covered)
    o["q2_S5"] = _or(b15s, oi_up)
    return o


def _col(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def s2c_cuts(design_dist_rows):
    """S2-c's frozen cuts: the lower terciles (pandas linear quantile at 1/3) of rb_funding_pct and global_ls_acct over the DIST DESIGN
    rows where BOTH are scored ("on the same rows"). → dict(funding, global_ls_acct, n). Pure."""
    f, g = _col(design_dist_rows, "rb_funding_pct"), _col(design_dist_rows, "global_ls_acct")
    both = f.notna() & g.notna()
    return dict(funding=float(f[both].quantile(1 / 3)) if both.any() else math.nan,
                global_ls_acct=float(g[both].quantile(1 / 3)) if both.any() else math.nan, n=int(both.sum()))


def s2c(rows, cuts):
    """S2-c per row with the frozen cuts (≤, inclusive) → Series of 1.0 / 0.0 / NaN; strict: NaN unless both legs are scored. Pure."""
    f, g = _col(rows, "rb_funding_pct"), _col(rows, "global_ls_acct")
    fc, gc = cuts.get("funding"), cuts.get("global_ls_acct")
    out = pd.Series(np.nan, index=rows.index)
    if not (_ok(fc) and _ok(gc)):
        return out
    both = f.notna() & g.notna()
    out[both] = ((f[both] <= fc) & (g[both] <= gc)).astype(float)
    return out
