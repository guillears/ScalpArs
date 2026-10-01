#!/usr/bin/env python3
"""🏃 RUNAWAY-PAIR sleeve design (operator, 2026-09-30 — MOVR +90 % in 24 h while BTC was flat; every momentum scan refused it
on ADX_MAX / EMA_GAP_MAX / RSI_RANGE + the BTC macro gate, i.e. FILTER_FAR = new-sleeve territory, DECISION_LOG 151).

v2 (2026-10-01, after the caveman + deep reviews of v1 — the v1 grid is NOT valid): control uses its OWN ATR / EMA13 (v1 borrowed
the event's post-pump ATR) and is skipped when its bar is missing · a missing control FAILS "beats control" · the statistic is
PAIRED (event − its own control) and clustered by UTC DAY (v1's 60-min windows overlapped 4–8 h holds) · Bonferroni bound from a
t-interval on day means (v1's 0.04 % bootstrap tail was Monte-Carlo noise) · a minute that OPENS through the stop exits at that
open (v1 filled at the stop) · NEXT enters at the open of the minute AFTER the trigger close (scan latency) and every fill pays
SLIP_PCT per market side (entry for NEXT, exit always) · ≥ 30 days of history judged AT the event · the universe filter runs
BEFORE the 8 h cooldown. v3 (second review round): clean controls (no same-pair trigger within ± cooldown + hold; t − 24/48/72 h),
entry slippage on the entry PRICE before the walk, WR = r > +0.02 with a scratch-rate column, top-3 share of the GROSS gains, one-sided
Bonferroni bound, trade-weighted mean, drops counted. Survivorship caveat: only pairs present in today's cache (delisted pairs missing).

PRE-DECLARED (written before any result; MOVR 09-30 is the inspiration and lies AFTER the data end 09-28 → not in the evidence):
  universe   the bot's universe proxy: top-50 by 24 h quote volume at the event among the cached tradeable pairs (blacklist,
             no-trade, BTC/ETH, stables excluded), ≥ 30 days of history
  trigger    on a CLOSED 5m bar, pair:
               LONG   4 h return ≥ +TH  ∧  close ≥ prior 24 h high  ∧  bar quote volume ≥ 3× prior-288 median
               SHORT  4 h return ≤ −TH  ∧  close ≤ prior 24 h low   ∧  same volume rule
             ∧ BTC flat: |BTC 4 h return| < 1 %  (the move is the pair's own) · one event per pair per 8 h
             TH ∈ {8, 12} %
  entry      NEXT      next 5m bar open (= the trigger close + 5 min, the bot's scan cadence)
             PULLBACK  first touch, within 120 min, of the trigger bar's EMA13 (5m) — no touch = no trade
  exit (1m candle-path walk, OHLC path per minute; net of 0.09 % fees)
             MOMENTUM   the live momentum exit: −0.70 until +0.40, then max(peak − 1×ATR, +0.10)      (reference: the bot today)
             SURGE      the live SURGE_LONG / Bull-Run REARM exit (1×ATR trail), mirrored for shorts
             BR2ATR     the Bull-Run GREEN exit (2×ATR trail), mirrored
             WIDE       stop −2×ATR (capped −6 %), arm at +1×ATR, trail peak − 1.5×ATR, floor +0.10
             hold cap   240 / 480 min
  (fix 2026-09-30 before reading PULLBACK: the fill minute contributes only what can follow a limit fill — see walk())
  units      UTC DAYS (v1 used 60-min market windows — superseded, see the v2 note above)
  control    the same pair, same clock time, t − 24 / 48 / 72 h (the first CLEAN one), same entry/exit (no trigger)
  split      train < 2026-05-01 ≤ test
  PASS bar (v2) ≥ 20 days · day-mean > 0 in BOTH halves · PAIRED event − control > 0 in BOTH halves and its 95 % day CI > 0 ·
             Bonferroni t-bound (64 cells) of the day-means > 0 · top-3 days < 50 % of the total · no pair > 30 % of profit
Usage: venv/bin/python scripts/runaway_sleeve_design.py   → reports/RUNAWAY_SLEEVE_GRID_v3_2026-10-01.csv (+ the per-trade FILLS / CONTROL csvs, not in git) + console summary"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import btc_spike_follow_1m as M   # noqa: E402  (1m cache/fetch + V2 = 5m frames D, FEE, ci)

V2 = M.V2
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
sys.path.insert(0, V2.ROOT)
import services.trading_engine as TE  # noqa: E402

BAR, MIN, H = V2.BAR, M.MIN, 3_600_000
SPLIT = 1777593600000                                   # 2026-05-01
TOP_N, VOL_X, BTC_FLAT, COOL = 50, 3.0, 1.0, 8 * H
THS = (8.0, 12.0)
N_CELLS = 64
SLIP_PCT = 0.05                                          # % per market side (taker slippage on 3×-volume bars)
OUT = os.path.join(V2.ROOT, "reports", "RUNAWAY_SLEEVE_GRID_v3_2026-10-01.csv")

btc = V2.btc
btc_r4 = (btc.c / btc.c.shift(48) - 1) * 100


def detect():
    """All triggers (side, TH) → list of (t, pair, side, th, atr, ema13)."""
    out = []
    for p, d in V2.D.items():
        if len(d) < 30 * 288:
            continue
        born = d.index[0]
        r4 = (d.c / d.c.shift(48) - 1) * 100
        hi = d.h.shift(1).rolling(288, min_periods=280).max(); lo = d.l.shift(1).rolling(288, min_periods=280).min()
        med = d.qvol.shift(1).rolling(288, min_periods=280).median()
        e13 = d.c.ewm(span=13, adjust=False).mean()
        b4 = btc_r4.reindex(d.index)
        flat = b4.abs() < BTC_FLAT
        volok = d.qvol >= VOL_X * med
        for th in THS:
            for side, ok in (("LONG", (r4 >= th) & (d.c >= hi)), ("SHORT", (r4 <= -th) & (d.c <= lo))):
                m = (ok & volok & flat).fillna(False)
                last = -10**18
                for t in d.index[m.values]:
                    if t - born < 30 * 86_400_000 or t - last < COOL:
                        continue
                    if p not in set(V2.universe(int(t), TOP_N)):   # universe BEFORE the cooldown (v2)
                        continue
                    last = t
                    out.append((int(t), p, side, th, float(d.atrp.loc[t]), float(e13.loc[t])))
    return pd.DataFrame(out, columns=["t", "pair", "side", "th", "atr", "ema13"])


def stop_fns(atr, side):
    """exit name → fn(peak %) → stop level (% in the trade's favour, net)."""
    br = lambda door: (lambda pk: TE._bullrun_exit_for(float("inf"), pk, atr, door)[2])
    wide_sl = max(-2.0 * atr, -6.0)
    return {
        "MOMENTUM": lambda pk: -0.70 if pk < 0.40 else max(pk - atr, 0.10),
        "SURGE": br("REARM"),
        "BR2ATR": br("GREEN"),
        "WIDE": lambda pk: wide_sl if pk < atr else max(pk - 1.5 * atr, 0.10),
    }


def walk(w, entry_px, side, fn, limit_fill=False):
    """Generic 1m candle-path walker (LONG or SHORT). Returns (net %, peak %) or None. Last row = time-cap exit.
    limit_fill: the entry is a resting limit filled INSIDE the first minute → that minute only contributes what can follow the
    fill (LONG: the low, then the close; SHORT: the high, then the close) — never the open/extreme that came before it
    (deep self-check 2026-09-30: walking the fill minute from its open credited pre-fill prices to the peak)."""
    if w is None or len(w) < 5:
        return None
    sg = 1 if side == "LONG" else -1
    peak = 0.0
    for k, (o, h, l, c) in enumerate(zip(w.o.values, w.h.values, w.l.values, w.c.values)):
        if k == 0 and limit_fill:
            path = (entry_px, l, c) if side == "LONG" else (entry_px, h, c)
        else:
            path = (o, h, l, c) if c < o else (o, l, h, c)
        for j, px in enumerate(path):
            v = sg * (px / entry_px - 1) * 100 - V2.FEE
            stop = fn(peak)
            if v <= stop:
                return (v if (j == 0 and k > 0) else stop), peak   # a minute that OPENS through the stop fills at that open (v2)
            peak = max(peak, v)
    return sg * (w.c.iloc[-1] / entry_px - 1) * 100 - V2.FEE, peak


DROPS = {}


def simulate(ev, control=False, entries_wanted=("NEXT", "PULLBACK")):
    """v3 (2026-10-01, second review round): a control is drawn at t − 24 h, else t − 48 h, else t − 72 h — the first one with NO
    trigger of the same pair within ± (cooldown + 8 h hold); none → no control for that event · market-entry slippage moves the ENTRY
    PRICE before the walk (stops / floors then sit where a live fill would put them) and only the exit slip is subtracted after ·
    drops are counted per reason · a hold window shorter than 95 % of its label is skipped."""
    rows = []
    tag = "control" if control else "events"
    by_pair = {p: np.sort(g.t.values) for p, g in ev.groupby("pair")} if control else {}
    guard = COOL + 480 * MIN
    def drop(why):
        DROPS[(tag, why)] = DROPS.get((tag, why), 0) + 1
    for k, r in enumerate(ev.itertuples()):
        t = r.t
        if control:
            t = None
            for back in (288, 576, 864):
                c = r.t - back * BAR
                if not (np.abs(by_pair[r.pair] - c) <= guard).any():
                    t = c; break
            if t is None:
                drop("no clean control"); continue
        start = t + BAR                                  # the trigger bar's close
        w_all = M.bars_1m(r.pair, start, start + (120 + 480) * MIN)
        if w_all is None or len(w_all) < 60:
            drop("no 1m data"); continue
        w_all = w_all[["o", "h", "l", "c"]].astype(float)
        if int(w_all.index[0]) != start or len(w_all) < 2:
            drop("1m gap at start"); continue
        atr, lvl = r.atr, r.ema13
        if control:                                       # the control's OWN ATR and EMA13 at its clock time (v2)
            d = V2.D[r.pair]
            if t not in d.index:
                drop("control bar missing"); continue
            atr = float(d.atrp.loc[t]); lvl = float(d.c.loc[:t].ewm(span=13, adjust=False).mean().iloc[-1])
        if not np.isfinite(atr):
            drop("ATR not finite"); continue
        fns = stop_fns(atr, r.side)
        sg = 1 if r.side == "LONG" else -1
        entries = {}
        if "NEXT" in entries_wanted:                      # scan latency: the minute after the trigger close; taker entry pays the slip
            entries["NEXT"] = (int(w_all.index[1]), float(w_all.o.iloc[1]) * (1 + sg * SLIP_PCT / 100))
        if "PULLBACK" in entries_wanted:
            first = w_all.loc[start:start + 120 * MIN - MIN]
            hit = first[(first.l <= lvl)] if r.side == "LONG" else first[(first.h >= lvl)]
            if len(hit):
                te = int(hit.index[0]); entries["PULLBACK"] = (te, min(float(hit.o.iloc[0]), lvl) if r.side == "LONG" else max(float(hit.o.iloc[0]), lvl))
        for en, (te, px) in entries.items():
            for hold in (240, 480):
                w = w_all.loc[te: te + hold * MIN - MIN]
                if len(w) < 0.95 * hold:
                    drop(f"short window {en}/{hold}"); continue
                for ex, fn in fns.items():
                    res = walk(w, px, r.side, fn, limit_fill=(en == "PULLBACK"))
                    if res:
                        rows.append((r.t, r.pair, r.side, r.th, en, ex, hold, res[0] - SLIP_PCT, res[1]))   # market exit slip
        if k % 200 == 0:
            print(f"  {tag} {k}/{len(ev)}", file=sys.stderr)
    return pd.DataFrame(rows, columns=["t", "pair", "side", "th", "entry", "exit", "hold", "r", "peak"])


def tcrit(df, alpha):
    """Two-sided Student-t critical value (Cornish-Fisher expansion of the normal quantile; < 0.5 % error for df ≥ 5 — no scipy)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    g1 = (z ** 3 + z) / 4; g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96; g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384
    return z + g1 / df + g2 / df ** 2 + g3 / df ** 3


def day_means(g):
    return g.assign(day=pd.to_datetime(g.t, unit="ms").dt.date).groupby("day").r.mean()


def table(F, C):
    """v2: per cell — event day-means (one value per UTC day), the PAIRED event − own-control difference per event, also by day;
    95 % and Bonferroni (α = 0.05 / N_CELLS) t-intervals on the day means; concentration by day and pair."""
    out = []
    key = ["side", "th", "entry", "exit", "hold"]
    Cm = C.set_index(key + ["t", "pair"]).r
    for k, g in F.groupby(key):
        dm = day_means(g); nd = len(dm)
        a, b = dm[[x < pd.Timestamp(SPLIT, unit="ms").date() for x in dm.index]].mean(), dm[[x >= pd.Timestamp(SPLIT, unit="ms").date() for x in dm.index]].mean()
        cr = [Cm.get(tuple(k) + (t, p)) for t, p in zip(g.t, g.pair)]
        pg = g.assign(c=cr).dropna(subset=["c"]); pg = pg.assign(r=pg.r - pg.c)
        pdm = day_means(pg) if len(pg) else pd.Series(dtype=float)
        def ci(x, alpha):
            if len(x) < 5:
                return np.nan, np.nan
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)); q = tcrit(len(x) - 1, alpha)
            return m - q * se, m + q * se
        lo95, hi95 = ci(dm, 0.05); lob, _ = ci(dm, 2 * 0.05 / N_CELLS); plo, phi = ci(pdm, 0.05)   # Bonferroni bound is ONE-sided
        pa = pdm[[x < pd.Timestamp(SPLIT, unit="ms").date() for x in pdm.index]].mean() if len(pdm) else np.nan
        pb = pdm[[x >= pd.Timestamp(SPLIT, unit="ms").date() for x in pdm.index]].mean() if len(pdm) else np.nan
        pos = dm[dm > 0].sum(); top3 = dm.sort_values(ascending=False).head(3).sum() / pos if pos > 0 else np.nan   # share of the GROSS gains
        pp = g.groupby("pair").r.sum(); pshare = (pp.max() / pp[pp > 0].sum()) if (pp > 0).any() else np.nan
        ok = (nd >= 20 and a > 0 and b > 0 and (pa == pa and pa > 0) and (pb == pb and pb > 0) and lob > 0 and plo > 0
              and (top3 == top3 and top3 < 0.5) and (pshare == pshare and pshare <= 0.30))
        out.append(dict(side=k[0], th=k[1], entry=k[2], exit=k[3], hold=k[4], fills=len(g), days=nd, per_day=round(dm.mean(), 3),
                        per_trade=round(g.r.mean(), 3), wr=round((g.r > 0.02).mean() * 100), scratch=round((g.r.abs() <= 0.02).mean() * 100),
                        n_paired=len(pg), ci95=f"[{lo95:+.2f},{hi95:+.2f}]", bonf_lo=round(lob, 3),
                        train=round(a, 3), test=round(b, 3), paired_vs_ctrl=round(pdm.mean(), 3) if len(pdm) else None,
                        paired_ci95=f"[{plo:+.2f},{phi:+.2f}]", paired_train=round(pa, 3), paired_test=round(pb, 3),
                        top3_day_share=round(top3, 2) if top3 == top3 else None, max_pair_share=round(pshare, 2) if pshare == pshare else None,
                        median_peak=round(g.peak.median(), 2), PASS="✅" if ok else ""))
    return pd.DataFrame(out).sort_values(["side", "per_day"], ascending=[True, False])


if __name__ == "__main__":
    pd.set_option("display.width", 280); pd.set_option("display.max_rows", 200); pd.set_option("display.max_columns", 30)
    ev = detect()
    print(f"events (in universe, BTC flat): {len(ev)} · by side/th:\n{ev.groupby(['side', 'th']).size().to_string()}")
    print(f"span {pd.to_datetime(ev.t.min(), unit='ms')} → {pd.to_datetime(ev.t.max(), unit='ms')} · pairs {ev.pair.nunique()}")
    F = simulate(ev); C = simulate(ev, control=True)
    T = table(F, C)
    print("drops:", DROPS)
    print(f"cells with 95% day CI > 0: {(T.ci95.str.extract(r'\[([+-][0-9.]+)')[0].astype(float) > 0).sum()} · paired CI > 0: "
          f"{(T.paired_ci95.str.extract(r'\[([+-][0-9.]+)')[0].astype(float) > 0).sum()} · train&test > 0: {((T.train > 0) & (T.test > 0)).sum()} of {len(T)}")
    T.to_csv(OUT, index=False)
    F.to_csv(OUT.replace("GRID", "FILLS"), index=False); C.to_csv(OUT.replace("GRID", "CONTROL"), index=False)
    print(T.to_string(index=False))
    print(f"\nPASS: {(T.PASS == '✅').sum()} of {len(T)} cells · grid → {OUT}")
