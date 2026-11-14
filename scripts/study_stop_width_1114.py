#!/usr/bin/env python3
"""STOP WIDTH (+ WILLY TP-cap) study — operator 2026-11-14: "for willy and willy ht … maybe they all hit the TP, so we could even not have SL
(or much wider). Do it for lite and lite retry too … show me before and after in B25 and in current batch" + "study too if we should change
the +8 of max tp to a lower level".

Sleeves (each apart): FRENZY_WILLY · FRENZY_WILLY_HT · FRENZY_LITE · FRENZY_LITE_RETRY.
Cohorts: B25 (reports/BASELINE25_batch1023-1111_orders.csv) · B26 (the export passed, opened ≥ 2026-11-11 15:11 UTC). CLOSED, probes and
operator closes (MANUAL / EXTERNAL) out.
TODAY's exits (only the stop / WILLY TP cap move):
  WILLY  TP T = the engine's spike-high target (frenzy_willy_high_target, max frenzy_willy_high_tp_max_pct 8), −4, late TP +0.25 from 30 min,
         cap 90; falling coins (1h EMA20 < EMA200) trail past T at max(peak − 2 × ATR, +0.10, −stop) (DL 541)
  HT     TP max(T, +2), −4, late +0.25 from 30, cap 90 (DL 537)
  LITE / RETRY  TP the stored LITE TP (else +3), −3, cap 720 min
Stop grid: WILLY / HT 4 (today) · 5 · 6 · 8 · 10 · LIQ · LITE / RETRY 3 (today) · 4 · 5 · 6 · 8 · LIQ — LIQ = "no stop": the position is
liquidated at the leverage it TRADED (100 / lev − 0.5 %; the margin is lost) — a real "no stop" cannot exist on futures.
TP-cap grid (WILLY / HT): 8 (today) · 6 · 5 · 4 · 3, crossed with every stop (a joint entry × exit read: every combo counts in the luck test).
RULER (CLAUDE.md exit/stop CF rule): BEFORE = the booked %. AFTER: a fill live did NOT stop keeps its booked % under a wider stop (a replica
stop there is a wick live rode through — counted as "replica-only stop", must be 0 or the replica is lying); a fill live STOPPED is re-walked
(saved = survives and ends above its booked % · deeper = stops wider anyway). A TP-cap change re-walks every fill (it moves winners too).
Path: study_b19_fzexit_common.load_path (aggTrades ticks, else 1m klines); missing 1m bars fetched once (public fapi klines, used-weight
header read, stop on 418 / 429) into reports/backtest_cache/study_stop_width_1114/1m. Slip 0.05 % on stop / trail / cap market exits.
One size (DL 503): % of the account per 1 % move = invest × ticket (0.24375) × leverage today — WILLY / HT 1.0 × 8 = 1.95 · LITE 0.85 × 8 =
1.66 · RETRY 0.95 × 10 = 2.32. DCR-cal over every calendar day of the batch. One stop's cost = stop × that size.
⚠ SUPERSEDED (operator 2026-11-14): this run reads every B25 fill incl. the ones today's filters block → the decision read is
scripts/study_stop_width_kept_1114.py (kept fills; it imports this file's walker / grid / paths). Known limits kept on record (review 546):
a live-stopped fill's AFTER here is the raw re-walk (the kept script uses BEFORE + (variant − today)); LIQ is booked at −(100 / lev − 0.5)
− slip, ≈ 0.4 pt short of a full margin loss (no conclusion moves); no shared Binance cooldown (a one-off study fetch, weight-paced).
Usage: venv/bin/python -I scripts/study_stop_width_1114.py <orders export.csv>"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if ROOT not in sys.path:
    sys.path.insert(1, ROOT)
import build_willy_exit_cf as CF            # noqa: E402
import study_b19_fzexit_common as FX        # noqa: E402

MIN = 60_000
SLIP = 0.05
TICKET = 9750 / 4 / 10000
SIZE = {"FRENZY_WILLY": 1.0 * TICKET * 8, "FRENZY_WILLY_HT": 1.0 * TICKET * 8, "FRENZY_LITE": 0.85 * TICKET * 8,
        "FRENZY_LITE_RETRY": 0.95 * TICKET * 10}
SLEEVES = list(SIZE)
STOPS = {"W": [4, 5, 6, 8, 10, "LIQ"], "L": [3, 4, 5, 6, 8, "LIQ"]}
TPCAPS = [8, 6, 5, 4, 3]
B25 = os.path.join(ROOT, "reports", "BASELINE25_batch1023-1111_orders.csv")
B26_FROM = "2026-11-11T15:11"
SPAN = {"B25": ("2026-10-23", "2026-11-11"), "B26": ("2026-11-11", "2026-11-14")}
K1 = os.path.join(ROOT, "reports", "backtest_cache", "study_stop_width_1114", "1m")
K1_OLD = os.path.join(ROOT, "reports", "backtest_cache", "study_willy_ht_1112", "1m")
OUT = os.path.join(ROOT, "reports", "STOP_WIDTH_STUDY_2026-11-14")

_orig = FX.WT.load_1m


def _load_1m(pair):
    k = _orig(pair)
    for d in (K1_OLD, K1):
        f = os.path.join(d, f"{pair}.csv")
        if os.path.exists(f):
            k = pd.concat([k, FX.WT._read_k1(f)]).drop_duplicates("open_time", keep="last").sort_values("open_time").reset_index(drop=True)
    FX.WT._K1[pair] = k
    return k


FX.WT.load_1m = _load_1m
_W = {"used": 0}


def fetch_1m(pair, start_ms, n):
    """one klines page into K1 (public endpoint, ≤ 1,200 weight/min, stop on 418/429)."""
    os.makedirs(K1, exist_ok=True)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={pair}&interval=1m&startTime={int(start_ms)}&limit={int(min(n, 1000))}"
    try:
        with urllib.request.urlopen(url, timeout=15) as r:
            rows = json.loads(r.read().decode()); used = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or 0)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"Binance {e.code} — stopped")
        return 0
    if used > 900:
        time.sleep(60)
    time.sleep(0.15)
    if not rows:
        return 0
    now = int(time.time() * 1000)
    d = pd.DataFrame([[int(x[0]), x[1], x[2], x[3], x[4]] for x in rows if int(x[0]) + MIN <= now],   # review 546: never cache a forming candle
                     columns=["open_time", "o", "h", "l", "c"])
    f = os.path.join(K1, f"{pair}.csv")
    if os.path.exists(f):
        d = pd.concat([pd.read_csv(f), d]).drop_duplicates("open_time", keep="last").sort_values("open_time")
    d.to_csv(f, index=False)
    FX.WT._K1.pop(pair, None)
    return len(rows)


def path(pair, te, E, cap):
    P = FX.load_path(pair, te, E, cap)
    if P["end_ms"] < te + cap * MIN:
        fetch_1m(pair, (te // MIN) * MIN, cap + 3)
        FX.WT._K1.pop(pair, None)
        P = FX.load_path(pair, te, E, cap)
    return P


def walk(P, tp, stop, cap, late=None, run=None):
    """one position on a net path → (pct, why). stop = positive % (the loss line −stop). late = (minute, +tp) or None. run = (k, atr, floor)
    → at the first touch of tp trail max(peak − k × atr, floor, −stop) instead of selling (WILLY runner, DL 541)."""
    te, t = int(P["te"]), P["t"]
    n = int(np.searchsorted(t, te + cap * MIN, side="right"))
    fl_i = None
    if late:
        idx = np.flatnonzero(t[:n] > te + late[0] * MIN)
        fl_i = int(idx[0]) if len(idx) else None
    armed, pk = False, float("nan")
    for i in range(n):
        bar = bool(P["bar"][i]); o = float(P["op"][i])
        hi, lo = float(P["hi"][i]), float(P["lo"][i])
        if armed:
            fl = max(pk - run[0] * run[1], run[2], -stop)
            if lo <= fl:
                v = (min(fl, o) if np.isfinite(o) else fl) if bar else lo
                return v - SLIP, "TRAIL"
            pk = max(pk, hi)
            continue
        if lo <= -stop:
            v = (min(-stop, o) if np.isfinite(o) else -stop) if bar else lo
            return v - SLIP, "STOP"
        lt = fl_i is not None and i >= fl_i
        line = late[1] if (lt and late[1] < tp) else tp
        if hi >= line:
            if run is not None and line == tp:
                armed, pk = True, float(tp)
                continue
            if lt and i == fl_i and line == late[1]:
                return float((o if (np.isfinite(o) and o >= line) else line) if bar else hi), "LATE_TP"
            return float(line), ("LATE_TP" if line < tp else "TP")
    if P["end_ms"] < te + cap * MIN or n == 0:
        return float("nan"), "UNRES"
    return float(P["cl"][n - 1]) - SLIP, "CAP"


def liq(lev):
    return max(0.5, 100.0 / float(lev) - 0.5)


def load(export):
    keep = lambda d: d[(d.status.astype(str) == "CLOSED") & d.entry_strategy.isin(SLEEVES)
                       & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")
                       & ~d.close_reason.astype(str).str.upper().str.startswith(("MANUAL", "EXTERNAL"))]
    a = keep(pd.read_csv(B25, low_memory=False)).assign(batch="B25")
    b = pd.read_csv(export, low_memory=False)
    b = keep(b[b.opened_at.astype(str) >= B26_FROM]).assign(batch="B26")
    d = pd.concat([a, b], ignore_index=True).drop_duplicates(["opened_at", "pair", "entry_strategy"], keep="last")
    return d.sort_values("opened_at").reset_index(drop=True)


def price(d, th):
    th_unc = dict(th, frenzy_willy_high_tp_max_pct=1000.0)
    rows, tcache = [], {}
    for r in d.to_dict("records"):
        sl = r["entry_strategy"]; w = sl.startswith("FRENZY_WILLY")
        te, E = CF._ms(r["opened_at"]), CF._num(r["entry_price"])
        cap = 90 if w else 720
        lev = CF._num(r.get("leverage")) or 8.0
        live_stop = str(r.get("close_reason")).upper() == "STOP_LOSS"
        base = dict(batch=r["batch"], sleeve=sl, opened_at=str(r["opened_at"])[:19], pair=r["pair"], lev=lev, booked=CF._num(r["pnl_percentage"]),
                    close=r.get("close_reason"), live_stop=live_stop, day=str(r.get("closed_at") or r["opened_at"])[:10], liq=liq(lev))
        P = path(r["pair"], te, E, cap)
        if not len(P["t"]):
            rows.append(dict(base, why_unpriced="no path")); continue
        if w:
            tg = CF.resolve_target(th, r, tcache)
            T8 = tg["T"] if tg["T"] is not None else 1.25
            H = tg["H"]
            Traw = CF.target_from_h(th_unc, r, H)[0] if H else None
            atr, gap = CF._num(r.get("entry_atr_pct")), CF._num(r.get("entry_pair_1h_ema20_200_gap_pct"))
            run = (2.0, atr, 0.10) if (sl == "FRENZY_WILLY" and th.get("frenzy_willy_runner_enabled") and atr and atr > 0
                                        and gap is not None and np.isfinite(gap) and gap < 0) else None
            late = (30, 0.25)
            for c in TPCAPS:
                Tc = T8 if c == 8 else (max(1.25, min(float(c), Traw)) if Traw is not None else min(T8, float(c)))
                tp = max(Tc, 2.0) if sl == "FRENZY_WILLY_HT" else Tc
                for s in STOPS["W"]:
                    sv = liq(lev) if s == "LIQ" else float(s)
                    pct, why = walk(P, tp, sv, cap, late, run)
                    base[f"c{c}_s{s}"], base[f"c{c}_s{s}_why"] = pct, why
            base["T8"], base["Traw"] = T8, Traw
        else:
            tp = CF._num(r.get("frenzy_lite_tp_used")) or float(th.get("frenzy_lite_tp_pct", 3.0))
            for s in STOPS["L"]:
                sv = liq(lev) if s == "LIQ" else float(s)
                pct, why = walk(P, tp, sv, cap)
                base[f"c0_s{s}"], base[f"c0_s{s}_why"] = pct, why
        rows.append(base)
    return pd.DataFrame(rows)


def after(g, col, today_col, stop_only):
    """AFTER % per fill: stop-only change → booked unless live stopped (then replica, shifted by the fill's own replica error is NOT
    applied — the replica value is used as is); TP-cap change → booked + (variant − today's replica)."""
    if stop_only:
        return np.where(g.live_stop, g[col], g.booked)
    return g.booked + (g[col] - g[today_col])


def dcr(pcts, days, size, span):
    D = pd.date_range(span[0], span[1]).strftime("%Y-%m-%d")
    r = pd.Series(np.asarray(pcts, float) * size, index=list(days)).groupby(level=0).sum()
    v = np.prod([1 + r.get(x, 0.0) / 100 for x in D]) ** (1 / len(D)) - 1
    return v * 100


def boot(delta, days, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    s = pd.Series(np.asarray(delta, float), index=list(days)).groupby(level=0).sum()
    if len(s) < 2:
        return float("nan"), float("nan")
    v = s.values
    b = [rng.choice(v, len(v)).sum() for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(export):
    th = CF.load_th()
    d = load(export)
    R = price(d, th)
    R.to_csv(f"{OUT}_rows.csv", index=False)
    L = ["# Stop width (+ WILLY TP cap) study — 2026-11-14", "", "BEFORE = booked %. AFTER (stop only) = booked unless the fill was stopped live (then the re-walk). AFTER (TP cap) = booked + (variant − today re-walk). LIQ = no stop: liquidated at the traded leverage.", ""]
    for sl in SLEEVES:
        w = sl.startswith("FRENZY_WILLY")
        kind = "W" if w else "L"; today_s = 4 if w else 3
        caps = TPCAPS if w else [0]
        today = f"c{caps[0]}_s{today_s}"
        sz = SIZE[sl]
        L += ["", f"## {sl} — one size {sz:.2f} % of the account per 1 % move · one stop today ≈ {today_s * sz:.1f} % of the account", ""]
        for bt in ("B25", "B26"):
            g = R[(R.sleeve == sl) & (R.batch == bt)]
            gp = g[g[today].notna()] if today in g else g.iloc[0:0]
            if not len(g):
                L.append(f"**{bt}:** no fills."); continue
            par = (gp[today] - gp.booked).abs() <= 0.5
            L += [f"**{bt}** — {len(g)} fills ({int(g.live_stop.sum())} stopped live) · priced {len(gp)} · replica of today's exit matches the booked % "
                  f"(±0.5) on {int(par.sum())} of {len(gp)}", "",
                  "| TP cap | Stop | N | WR | Avg % | Σ % | Δ Σ vs today | 95 % day CI of Δ | saved / deeper (live-stopped) | replica-only stops | "
                  "worst fill % | one stop = % of account | DCR %/day before → after |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            bef = gp.booked.values
            dcr0 = dcr(bef, gp.day, sz, SPAN[bt])
            for c in caps:
                for s in STOPS[kind]:
                    col = f"c{c}_s{s}"
                    stop_only = (c == caps[0])
                    if c == caps[0] and s == today_s:
                        a = bef
                    else:
                        a = after(gp, col, today, stop_only)
                    a = np.asarray(a, float)
                    dl = a - bef
                    ls = gp.live_stop.values
                    saved = int(((dl > 1e-9) & ls).sum()); deeper = int(((dl <= 1e-9) & ls & (gp[col + "_why"].isin(["STOP"]).values)).sum())
                    ro = int(((gp[col + "_why"] != "STOP").values & (gp[today + "_why"] == "STOP").values & ~ls).sum()) if stop_only else 0
                    lo, hi = boot(dl, gp.day)
                    sv = "liq" if s == "LIQ" else f"−{s}"
                    one = (f"{(gp.liq.mean() if s == 'LIQ' else s) * sz:.1f}")
                    L.append(f"| {'+' + str(c) if w else 'LITE TP'} | {sv} | {len(a)} | {(a > 0).mean():.0%} | {a.mean():+.2f} | {a.sum():+.1f} | "
                             f"{dl.sum():+.1f} | [{lo:+.1f}, {hi:+.1f}] | {saved} / {deeper} | {ro} | {a.min():+.1f} | {one} | "
                             f"{dcr0:+.2f} → {dcr(a, gp.day, sz, SPAN[bt]):+.2f} |")
            L.append("")
    L += ["", "## Every live-stopped fill (what each wider stop would have done)", ""]
    for sl in SLEEVES:
        w = sl.startswith("FRENZY_WILLY"); kind = "W" if w else "L"; c = 8 if w else 0
        g = R[(R.sleeve == sl) & R.live_stop]
        cols = [f"c{c}_s{s}" for s in STOPS[kind]]
        L += [f"**{sl}**", "", "| batch | opened | pair | lev | booked | " + " | ".join(("liq" if s == "LIQ" else f"−{s}") for s in STOPS[kind]) + " |",
              "|" + "---|" * (5 + len(cols))]
        for r in g.itertuples():
            L.append(f"| {r.batch} | {r.opened_at[5:16]} | {r.pair} | {r.lev:g} | {r.booked:+.2f} | "
                     + " | ".join("–" if pd.isna(getattr(r, x)) else f"{getattr(r, x):+.2f}" for x in cols) + " |")
        L.append("")
    open(f"{OUT}.md", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
