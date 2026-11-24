#!/usr/bin/env python3
"""TP_RUNNER × Route B — at the TP touch, does the ORDER FLOW say "don't sell, trail"? (operator 2026-11-24 "What about the info in Route B?
there is no indicator that might trigger a 'dont stop' active trailing?" → "yes run it"; research only, nothing changes in the bot).

PRIOR (prior-study check): TP_HOLD 11-24 screened kline / taker-share (klines) / OI / top-trader at the touch → nothing beat the null; DL 626
tested Route B order flow at the PEAK (a sell signal) → no. NEW = Route B order flow at the TP touch as a CONTINUATION signal.
COHORT   reports/ROUTE_B_SIGNALS_data/signals.pkl.gz rows priced on today's exits (out_reason TP: every signal, taken or blocked, whose
         today's-exit walk reached the TP) — EXH / DIST / HT; touch = out_exit_ms (the TP bar), TP price = out_exit_px. De-duplicated to one
         touch per coin per 60 min (operator 11-20 "possibilities, not events"); the raw count is stated.
OUTCOME  the TP_RUNNER rule (scripts/scout_tp_runner_636.py — identical walk): 4 × 15m ATR14 at the touch, from the running best, HT floor +5
         net, EXH / DIST the signal's own stop kept, 24 h; Δ vs the maker TP incl. fee gap, 0.10 slip and funding (study cache). Secondary:
         REACH = the best further favourable move in the 4 h after the touch (pp of entry, rule-free).
FEATURES (minutes strictly BEFORE the touch minute; signed so + = in the trade's favour where a direction exists): taker share with us
         1 / 5 / 15 min, share accel (5 − 60), volume 5 vs 60, biggest print 5 vs 60; book imbalance ±0.5 %, depth ratio log(bid ÷ ask) at
         0.2 / 0.5 / 1 % signed, its 15-min change, spread, reach on the move side; liquidations with / against us 5 / 15 min (+ vs 60);
         premium (mark ÷ index − 1) signed and its 15-min change, funding (receives = +); OI change 5 / 15 min, top-trader position L/S,
         taker B/S ratio signed; price speed 5 / 15 min signed, minutes from entry to TP, ATR.
TESTS    (rev. after review) on HOUR units (window-units rule: one row per touch hour, means): median split per feature × outcome × block (all · EXH · DIST); ONE family-wise null = 2,000
         shuffles (max |t| over every block, outcome and feature); trade-level splits shown; follow-ups for the top cells (medians, ATR-scaled reach / adverse, first side hit, runner Δ + day CI, per day / sleeve / coin, second targets); exhaustive 2D median
         quadrants (cell N ≥ 8) vs the same null. Event-study chart. Outputs reports/TP_RUNNER_ROUTEB_2026-11-24{_results.txt,_features.csv,_screen.csv,_chart.png} (the .md is written from _results.txt)."""
import glob
import importlib
import os
import sys
import itertools
import functools

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts")); sys.path.insert(0, ROOT)
T = importlib.import_module("TP_HOLD_2026-11-24"); C = T.C
S = importlib.import_module("scout_tp_runner_636")
sys.path.insert(0, os.path.join(ROOT, "scripts", "route_b_signals"))
import sources as SRC   # noqa: E402  (the loader's own split-minute merge rules)
DV = importlib.import_module("TP_RUNNER_DIVE_2026-11-24")
print = functools.partial(print, flush=True)
MIN, HR = 60_000, 3_600_000
RB = os.path.join(ROOT, "reports", "ROUTE_B_SIGNALS_data")
OUT = os.path.join(ROOT, "reports", "TP_RUNNER_ROUTEB_2026-11-24")
SIDE = {"LONG": 1, "SHORT": -1}
rng = np.random.default_rng(1124)


def cohort():
    s = pd.read_pickle(os.path.join(RB, "signals.pkl.gz"))
    t = s[(s.out_reason == "TP") & s.sleeve.isin(["EXHAUSTION_SHORT", "DISTRIBUTION_SHORT", "FRENZY_WILLY_HT"])].copy()
    t["touch"] = t.out_exit_ms.astype("int64") // MIN * MIN
    t = t.sort_values(["touch", "sleeve", "pair"], kind="stable")       # (review) deterministic tie-break
    raw = len(t)
    keep, last = [], {}
    for i, r in t.iterrows():
        if r.pair in last and r.touch - last[r.pair] < 60 * MIN:
            continue
        last[r.pair] = r.touch; keep.append(i)
    t = t.loc[keep].reset_index(drop=True)
    t["taken"] = t.status.astype(str).str.startswith("OPENED")
    return t, raw


def minutes(pairs):
    fr = {k: [] for k in ("book", "mark", "liq", "pos")}
    for f in sorted(glob.glob(os.path.join(RB, "minutes", "*.v2.pkl.gz"))):
        m = pd.read_pickle(f)
        for k in fr:
            v = m.get(k)
            if isinstance(v, pd.DataFrame) and len(v):
                fr[k].append(v[v.sym.isin(pairs)])
    cat = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in fr.items()}
    for x in cat.values():
        if len(x):
            x["minute"] = x.minute.astype("int64")
    # (review) a coin-minute split across two hourly files is MERGED as route_b_signals/sources.load_minutes does
    b = cat["book"].sort_values(["sym", "minute", "n_book"])
    g = b.groupby(["sym", "minute"], sort=False)
    book = pd.concat([g[SRC.BOOK_COLS].last(), g[["n_trades", "buy_usd", "sell_usd"]].sum(min_count=1), g[["max_trade_usd"]].max()],
                     axis=1).reset_index().sort_values(["sym", "minute"]).reset_index(drop=True)
    mark = cat["mark"].groupby(["sym", "minute"], sort=True)[SRC.MARK_COLS].last().reset_index()
    liq = (cat["liq"].groupby(["sym", "minute"], sort=True)[["liq_long_usd", "liq_short_usd"]].sum().reset_index() if len(cat["liq"])
           else pd.DataFrame(columns=["sym", "minute", "liq_long_usd", "liq_short_usd"]))
    _, pos = SRC.merge_fb_pos(pd.DataFrame(), cat["pos"]) if len(cat["pos"]) else (None, pd.DataFrame())
    return dict(book=book, mark=mark, liq=liq, pos=pos)


def win(df, sym, a, b):
    if not len(df):
        return df
    x = df[df.sym == sym]
    return x[(x.minute >= a) & (x.minute < b)]


def feats(r, M):
    d, m0, p = SIDE[str(r.direction).upper()], int(r.touch), r.pair
    f = {}
    bk = lambda k: win(M["book"], p, m0 - k * MIN, m0)
    b60, b15, b5, b1 = bk(60), bk(15), bk(5), bk(1)

    def share(x):
        tot = (x.buy_usd + x.sell_usd).sum()
        w = x.buy_usd.sum() if d > 0 else x.sell_usd.sum()
        return w / tot if tot > 0 else np.nan
    if len(b5):
        f["tk_share_1"] = share(b1) if len(b1) else np.nan
        f["tk_share_5"], f["tk_share_15"] = share(b5), share(b15)
        f["tk_share_accel"] = share(b5) - share(b60) if len(b60) else np.nan
        v5, v60 = (b5.buy_usd + b5.sell_usd).mean(), (b60.buy_usd + b60.sell_usd).mean()
        f["vol5_vs_60"] = v5 / v60 if v60 > 0 else np.nan
        f["bigprint5_vs_60"] = b5.max_trade_usd.max() / b60.max_trade_usd.median() if b60.max_trade_usd.median() > 0 else np.nan
        last = b5.iloc[-1]
        f["book_imb05_signed"] = d * last.imb_05
        for bd in ("02", "05", "1"):
            bi, ai = last[f"bid_usd_{bd}"], last[f"ask_usd_{bd}"]
            f[f"depth_ratio_{bd}_signed"] = d * np.log(bi / ai) if bi > 0 and ai > 0 else np.nan
        if len(b15) >= 10:
            fst = b15.iloc[0]
            r0 = d * np.log(fst.bid_usd_05 / fst.ask_usd_05) if fst.bid_usd_05 > 0 and fst.ask_usd_05 > 0 else np.nan
            f["depth_ratio_05_chg15"] = f["depth_ratio_05_signed"] - r0
        f["spread_bps"] = last.spread_bps
        f["reach_moveside"] = last.reach_bid_pct if d < 0 else last.reach_ask_pct
        f["speed5_signed"] = d * (last.mid / b5.iloc[0].mid - 1) * 100 if b5.iloc[0].mid > 0 else np.nan
        f["speed15_signed"] = d * (last.mid / b15.iloc[0].mid - 1) * 100 if len(b15) and b15.iloc[0].mid > 0 else np.nan
    lq = M["liq"]
    for k in (5, 15, 60):
        if not len(bk(k)):                     # (review) the coin was not recorded → unknown, not "no liquidations"
            f[f"liq_with_{k}"] = f[f"liq_against_{k}"] = np.nan
            continue
        x = win(lq, p, m0 - k * MIN, m0)
        w = (x.liq_long_usd.sum() if d < 0 else x.liq_short_usd.sum()) if len(x) else 0.0   # short: longs liquidated = forced sells with us
        a = (x.liq_short_usd.sum() if d < 0 else x.liq_long_usd.sum()) if len(x) else 0.0
        f[f"liq_with_{k}"], f[f"liq_against_{k}"] = w, a
    mk = win(M["mark"], p, m0 - 16 * MIN, m0)
    if len(mk) and mk.iloc[-1]["index"] > 0:
        pr = (mk.mark / mk["index"] - 1) * 100
        f["premium_signed"] = d * pr.iloc[-1]
        f["premium_chg15_signed"] = d * (pr.iloc[-1] - pr.iloc[0])
        f["funding_receive"] = -d * mk.funding.iloc[-1] * 100
    po = win(M["pos"], p, m0 - 16 * MIN, m0)
    if len(po):
        l = po.iloc[-1]
        f["oi_chg_5m"] = l.oi_chg_5m
        f["oi_chg_15m"] = (l.oi / po.oi.iloc[0] - 1) * 100 if po.oi.iloc[0] > 0 else np.nan
        f["top_ls_pos_signed"] = d * np.log(l.top_ls_pos) if l.top_ls_pos > 0 else np.nan   # (review) + = top traders on our side
        f["taker_bs_signed"] = d * np.log(l.taker_bs_ratio) if l.taker_bs_ratio > 0 else np.nan
    f["min_entry_to_tp"] = (m0 - int(r.out_entry_ms)) / MIN if pd.notna(r.out_entry_ms) else np.nan
    return f


def outcome(r):
    d, p, m0 = SIDE[str(r.direction).upper()], r.pair, int(r.touch)
    a = C.k1m(p, m0 - 5 * HR, m0 + 24 * HR + 2 * MIN)
    if a is None or not len(a["t"]):
        return None
    bars = [[int(t), o, h, l, c] for t, o, h, l, c in zip(a["t"], a["o"], a["h"], a["l"], a["c"])]
    atr = S.atr15_pct(bars, m0)
    E, tp = float(r.out_entry_px), float(r.out_exit_px)
    if atr is None:
        return None
    guard = S.ht_floor_px(E) if d > 0 else E * (1 + float(r.out_sl_pct) / 100)
    why, px, t_exit = S.runner_walk(bars, m0, tp, d, S.K_ATR * atr, guard)
    if why == "PENDING":
        return dict(why="PENDING", atr=atr)
    fund = DV.fund_between(p, m0, t_exit - MIN, E, px, d)
    dl = S.delta_pp(d, E, tp, px, fund)
    post = [q for q in bars if m0 + MIN <= q[0] < m0 + MIN + 4 * HR]
    reach = max((d * ((q[2] if d > 0 else q[3]) - tp) / E * 100 for q in post), default=np.nan)
    adverse = max((-d * ((q[3] if d > 0 else q[2]) - tp) / E * 100 for q in post), default=np.nan)
    first = {}
    for x in (3, 5):                           # which side of ± x pp (from the TP, pp of entry) is hit first in 4 h
        first[f"first_ours_{x}"] = np.nan
        for q in post:
            up = d * ((q[2] if d > 0 else q[3]) - tp) / E * 100 >= x
            dn = -d * ((q[3] if d > 0 else q[2]) - tp) / E * 100 >= x
            if up or dn:
                first[f"first_ours_{x}"] = float(up and not dn); break
    ext = {}
    walk = [q for q in bars if m0 + MIN <= q[0] < m0 + MIN + 24 * HR]
    for x in (2, 3, 5, 8):                     # a second resting target x % past the TP, guard = the same floor / own stop, 24 h
        tgt, v = tp * (1 + d * x / 100), np.nan
        for q in walk:
            if (d < 0 and q[2] >= guard) or (d > 0 and q[3] <= guard):
                v = S.delta_pp(d, E, tp, (min if d > 0 else max)(guard, q[1]), 0.0); break
            if (d < 0 and q[3] <= tgt) or (d > 0 and q[2] >= tgt):
                v = d * (tgt - tp) / E * 100; break               # maker → maker: no fee gap, no slip
        else:
            v = S.delta_pp(d, E, tp, walk[-1][4], 0.0) if walk else np.nan
        ext[f"ext_{x}"] = v
    path = {}
    for q in post:
        k = (q[0] - m0) // MIN
        path[int(k)] = d * (q[4] - tp) / E * 100
    return dict(why=why, atr=atr, delta=dl, fund=fund, reach4h=reach, adverse4h=adverse, **first, **ext, path=path)


def tstat(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or len(y) < 3:
        return 0.0
    se = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
    return (x.mean() - y.mean()) / se if se > 0 else 0.0


def units(D, feats_, ycols):
    """(review, WINDOW-UNITS rule) one row per touch HOUR: the mean of every feature and outcome over that hour's touches."""
    g = D.assign(hour=pd.to_datetime(D.touch, unit="ms").dt.floor("h"))
    return g.groupby("hour")[list(feats_) + list(ycols)].mean().reset_index()


def split_t(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 12 or np.nanstd(x[ok]) == 0:
        return None
    med = np.median(x[ok])
    hi, lo = ok & (x > med), ok & (x <= med)
    return med, y[hi].mean(), y[lo].mean(), tstat(y[hi], y[lo]), int(ok.sum())


def screen_all(blocks, feats_, ycols, nperm=2000):
    """blocks = {name: hour-unit frame}. Every (block × outcome × feature) median split; ONE family-wise null = the max |t| over ALL
    blocks and outcomes per shuffle (each block's hour units shuffled independently, outcomes jointly). → (table, 95 % threshold, null)."""
    rows = []
    for bn, U in blocks.items():
        for yc in ycols:
            for f in feats_:
                r = split_t(U[f].values, U[yc].values)
                if r:
                    rows.append(dict(block=bn, outcome=yc, feature=f, N_hours=r[4], median=r[0], hi_mean=r[1], lo_mean=r[2],
                                     diff=r[1] - r[2], t=r[3]))
    R = pd.DataFrame(rows)
    null = []
    for _ in range(nperm):
        best = 0.0
        for bn, U in blocks.items():
            Y = U[list(ycols)].values[rng.permutation(len(U))]
            for k, yc in enumerate(ycols):
                for f in R[(R.block == bn) & (R.outcome == yc)].feature:
                    r = split_t(U[f].values, Y[:, k])
                    if r:
                        best = max(best, abs(r[3]))
        null.append(best)
    null = np.array(null)
    R["fw_p"] = [(null >= abs(t)).mean() for t in R.t]
    return R.sort_values("t", key=abs, ascending=False).reset_index(drop=True), np.quantile(null, 0.95)


def screen2d(D, feats_, ycol, nperm=500, nmin=8):
    y = D[ycol].values
    cols = [f for f in feats_ if np.isfinite(D[f].values).sum() >= 24]
    masks = {}
    for f in cols:
        x = D[f].values; ok = np.isfinite(x); med = np.median(x[ok])
        masks[f] = (ok & (x > med), ok & (x <= med))
    cells = []
    for a, b in itertools.combinations(cols, 2):
        both = (masks[a][0] | masks[a][1]) & (masks[b][0] | masks[b][1])     # (review) only rows scored on BOTH
        for ia, ib in itertools.product((0, 1), (0, 1)):
            m = masks[a][ia] & masks[b][ib]
            if m.sum() >= nmin:
                cells.append((a + ("↑" if ia == 0 else "↓"), b + ("↑" if ib == 0 else "↓"), m, both))
    def best(yy):
        out = (0.0, None)
        for a, b, m, both in cells:
            rest = ~m & both & np.isfinite(yy)
            t = tstat(yy[m], yy[rest])
            if abs(t) > abs(out[0]):
                out = (t, (a, b, int(m.sum()), yy[m].mean(), yy[rest].mean()))
        return out
    obs = best(y)
    null = [abs(best(rng.permutation(y))[0]) for _ in range(nperm)]
    return obs, np.quantile(null, 0.95), len(cells)


def dayboot(D, col, mask, n=2000):
    day = pd.to_datetime(D.touch, unit="ms").dt.floor("D")
    g = D.assign(day=day, m=mask)
    days = g.day.unique(); out = []
    for _ in range(n):
        x = pd.concat([g[g.day == dd] for dd in rng.choice(days, len(days))])
        a, b = x[x.m][col], x[~x.m][col]
        if len(a) and len(b):
            out.append(a.mean() - b.mean())
    return np.quantile(out, [0.025, 0.975])


def follow_up(D, f, P):
    """every number the write-up quotes about one variable's halves (rows scored on f only)."""
    X = D[np.isfinite(D[f])].copy()
    m = (X[f] > X[f].median()).values
    hi, lo = X[m], X[~m]
    day = pd.to_datetime(X.touch, unit="ms").dt.strftime("%m-%d")
    P(f"\n#### follow-up: {f} (scored {len(X)}/{len(D)} · halves {m.sum()} / {(~m).sum()} · 4 day clusters)")
    q = lambda c: f"{hi[c].mean():+.2f} vs {lo[c].mean():+.2f} (median {hi[c].median():+.2f} vs {lo[c].median():+.2f})"
    P(f"  reach 4 h: {q('reach4h')} · adverse 4 h: {q('adverse4h')} · ATR {hi.atr.median():.2f} vs {lo.atr.median():.2f}")
    P(f"  per ATR (medians): reach {(hi.reach4h / hi.atr).median():.2f} vs {(lo.reach4h / lo.atr).median():.2f} · adverse "
      f"{(hi.adverse4h / hi.atr).median():.2f} vs {(lo.adverse4h / lo.atr).median():.2f} · reach − adverse {(hi.reach4h - hi.adverse4h).median():+.2f} vs "
      f"{(lo.reach4h - lo.adverse4h).median():+.2f}")
    P(f"  our side hit first at ±3 pp {hi.first_ours_3.mean():.0%} vs {lo.first_ours_3.mean():.0%} · ±5 pp {hi.first_ours_5.mean():.0%} vs {lo.first_ours_5.mean():.0%}")
    ci = dayboot(X, "delta", m)
    P(f"  runner Δ: {q('delta')} · ran {(hi.delta > 0).mean():.0%} vs {(lo.delta > 0).mean():.0%} · day-block CI of the gap {ci[0]:+.2f} … {ci[1]:+.2f}")
    ci = dayboot(X, "reach4h", m)
    P(f"  reach gap day-block CI {ci[0]:+.2f} … {ci[1]:+.2f}")
    for c in ("reach4h", "adverse4h", "delta"):
        pdy = {d: f"{X[(day == d).values & m][c].mean() - X[(day == d).values & ~m][c].mean():+.1f} (n {int(((day == d).values & m).sum())}/{int(((day == d).values & ~m).sum())})"
               for d in sorted(day.unique())}
        psl = {s[:4]: f"{X[(X.sleeve == s).values & m][c].mean() - X[(X.sleeve == s).values & ~m][c].mean():+.1f}" for s in sorted(X.sleeve.unique())}
        P(f"  {c} gap per day {pdy} · per sleeve {psl}")
    sh = (hi.groupby("pair").reach4h.sum() / hi.reach4h.sum()).sort_values(ascending=False)
    P(f"  high half: {hi.pair.nunique()} coins · {pd.to_datetime(hi.touch, unit='ms').dt.floor('h').nunique()} hours · top coins' share of its reach "
      f"{sh.head(3).round(2).to_dict()}")
    P("  second resting target x % past the TP (guard = HT floor / own stop, 24 h; maker target, taker + 0.10 slip guard), Δ vs TP:")
    for x in (2, 3, 5, 8):
        c = f"ext_{x}"
        P(f"    +{x} %: high {hi[c].mean():+.2f} · low {lo[c].mean():+.2f} · all {X[c].mean():+.2f}")
    return m


def main():
    out = []
    P = lambda *a: (print(*a), out.append(" ".join(str(x) for x in a)))
    D0, raw = cohort()
    P(f"TP touches (today's exits): {raw} raw → {len(D0)} one per coin per 60 min · {D0.sleeve.value_counts().to_dict()}")
    T.fetch(pd.DataFrame(dict(pair=D0.pair, t_touch=D0.touch)))
    for p in D0.pair.unique():
        ms_ = sorted({pd.Timestamp(t, unit="ms").strftime("%Y-%m") for t in D0[D0.pair == p].touch})
        try:
            C.fetch_funding(p, ms_)
        except SystemExit:
            pass
    M = minutes(set(D0.pair))
    rows, paths = [], {}
    for i, r in D0.iterrows():
        o = outcome(r)
        if o is None or o["why"] == "PENDING":
            P("  no path / pending:", r.pair, pd.Timestamp(r.touch, unit="ms")); continue
        paths[i] = o.pop("path")
        rows.append(dict(i=i, sleeve=r.sleeve, pair=r.pair, touch=r.touch, taken=r.taken, **o, **feats(r, M)))
    D = pd.DataFrame(rows)
    D.to_csv(OUT + "_features.csv", index=False)
    NOTF = {"i", "sleeve", "pair", "touch", "taken", "why", "delta", "fund", "reach4h", "adverse4h"}
    F = [c for c in D.columns if c not in NOTF and not c.startswith(("first_", "ext_"))]
    hrs = pd.to_datetime(D.touch, unit="ms").dt.floor("h")
    P(f"\npriced {len(D)} ({int(D.taken.sum())} taken) · days {hrs.dt.floor('D').nunique()} · hours {hrs.nunique()}")
    P("outcome by sleeve:\n" + D.groupby("sleeve").agg(N=("delta", "size"), mean_delta=("delta", "mean"), median=("delta", "median"),
                                                     ran=("delta", lambda x: (x > 0).mean()), reach4h=("reach4h", "median"),
                                                     fund=("fund", "mean")).round(2).to_string())
    P("exit mix:", D.why.value_counts().to_dict())
    P(f"features ({len(F)}): coverage " + str({f: int(np.isfinite(D[f]).sum()) for f in F}))
    Y = ("delta", "reach4h")
    blocks = {"ALL": units(D, F, Y)}
    for s_, g in D.groupby("sleeve"):
        if len(g) >= 20:
            blocks[s_[:4]] = units(g, F, Y)
    P("hour units per block:", {k: len(v) for k, v in blocks.items()})
    R, thr = screen_all(blocks, F, Y)
    R.to_csv(OUT + "_screen.csv", index=False)
    P(f"\n== ONE family: {len(R)} tests ({len(blocks)} blocks × 2 outcomes × features) on HOUR units · 95 % |t| threshold {thr:.2f}")
    P(R.head(12).round(3).to_string(index=False))
    P("\n== trade-level reading of the top hour-unit cells (shown, not the test):")
    for r in R.head(5).itertuples():
        X = D if r.block == "ALL" else D[D.sleeve.str[:4] == r.block]
        z = split_t(X[r.feature].values, X[r.outcome].values)
        if z:
            P(f"  {r.block} · {r.outcome} · {r.feature}: {z[1]:+.2f} vs {z[2]:+.2f} (N {z[4]}, t {z[3]:+.2f})")
    obs, thr2, ncell = screen2d(D, F, "delta")
    P(f"\n2D (runner Δ, all touches, rows scored on both): {ncell} cells · best t {obs[0]:+.2f} {obs[1]} · null 95 % {thr2:.2f}")
    obs, thr2, ncell = screen2d(D, F, "reach4h")
    P(f"2D (reach 4 h): {ncell} cells · best t {obs[0]:+.2f} {obs[1]} · null 95 % {thr2:.2f}")
    top_d = R[R.outcome == "delta"].iloc[0]
    follow_up(D, top_d.feature, P)
    top_r = R[R.outcome == "reach4h"].iloc[0]
    m = follow_up(D, top_r.feature, P)
    if top_r.feature != "liq_with_5":
        follow_up(D, "liq_with_5", P)
    with open(OUT + "_results.txt", "w") as fh:
        fh.write("\n".join(out) + "\n")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    grid = np.arange(1, 241)

    def mean_path(idx):
        P_ = np.array([[paths[j].get(k, np.nan) for k in grid] for j in idx]) if len(idx) else np.full((1, len(grid)), np.nan)
        return np.nanmean(P_, axis=0)
    ran, gave = D[D.delta > 0].i.values, D[D.delta <= 0].i.values
    ax[0].plot(grid, mean_path(ran), color="#2a9d8f", label=f"runner beat TP (n {len(ran)})")
    ax[0].plot(grid, mean_path(gave), color="#e76f51", label=f"gave back (n {len(gave)})")
    ax[0].axhline(0, color="#888", lw=.8); ax[0].set_title("Price after the TP touch (pp of entry, + = our way)"); ax[0].set_xlabel("minutes after the touch"); ax[0].legend()
    X = D[np.isfinite(D[top_r.feature])]
    hi_i, lo_i = X[m].i.values, X[~m].i.values
    ax[1].plot(grid, mean_path(hi_i), color="#264653", label=f"{top_r.feature} above median (n {len(hi_i)})")
    ax[1].plot(grid, mean_path(lo_i), color="#e9c46a", label=f"below (n {len(lo_i)})")
    ax[1].axhline(0, color="#888", lw=.8); ax[1].set_title(f"Split by the top reach variable (hour-unit fw p {top_r.fw_p:.2f})"); ax[1].set_xlabel("minutes after the touch"); ax[1].legend()
    fig.tight_layout(); fig.savefig(OUT + "_chart.png", dpi=130)


if __name__ == "__main__":
    main()
