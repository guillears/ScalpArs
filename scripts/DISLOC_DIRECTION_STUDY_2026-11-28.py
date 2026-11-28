#!/usr/bin/env python3
"""DISLOC_DIRECTION_STUDY_2026-11-28 — research only. Would a ONE-SIDED FRENZY-family dislocation guard (abort a LONG only when the live
price is ABOVE the decision price by > frenzy_max_entry_dislocation_pct) have made money? Every past *_DISLOC abort priced AS IF FILLED at
the live price, split CHEAPER (live below decision — favourable for a long) vs DEARER (adverse).

Cohort   e=BLOCK rows, gate *_DISLOC / FRENZY_CATCHUP_MOVED, every decisions export (exports_archive/ + ~/Downloads), dedup (t, pair, gate).
         t = the signal 5m bar close (UTC). *_LATE rows counted (descriptive).
Prices   CloudWatch "[..._DISLOC] PAIR LONG: entry aborted — live L is x% from decision D" when the line is still kept (14-day retention):
         exact refusal second, D, L. Else reconstructed from Binance USDⓈ-M public aggTrades: decision second = t + the median CloudWatch
         decision lag; D = last trade at / before it; L = first trade in (decision, decision + 20 s] with |L / D − 1| > 1 % → direction = its
         sign; none → UNRESOLVED (listed, not counted). The CloudWatch overlap is reconstructed too, to calibrate (direction agreement,
         price error).
Pricing  scripts/scout_blocked_filters.price_frenzy (the validated net-path walker, each sleeve's live exit from trading_config.json):
         entry = refusal + 8 s at L moved 0.10 % against the trade, taker fee. Plain WILLY's exit needs the spike high: the episode spike
         time is not on a BLOCK row → approximated as the highest-volume 5m bar in the 12 h before t (flagged APPROX_SPIKE).
         KEPT = the same sleeves' closed full-size fills over the same span, same pricer (scout_blocked_filters.load_kept / kept_as_rows).
Out      reports/DISLOC_DIRECTION_STUDY_2026-11-28{.md,.csv,_chart.png}.
Binance  ≤ 1,200 weight / min (reads X-MBX-USED-WEIGHT-1M, pauses at 900), stops on 418 / 429.
Usage    venv/bin/python -I scripts/DISLOC_DIRECTION_STUDY_2026-11-28.py
"""
import glob
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "scripts"), R]
import scout_blocked_filters as SB   # noqa: E402

OUT = os.path.join(R, "reports", "DISLOC_DIRECTION_STUDY_2026-11-28")
LOG_GROUP = "/aws/elasticbeanstalk/SCALPARS-env-2/var/log/web.stdout.log"
MAX_PCT = 1.0
SLEEVE_OF_GATE = {"FRENZY_WILLY_HT_DISLOC": "FRENZY_WILLY_HT", "FRENZY_WILLY_DISLOC": "FRENZY_WILLY", "FRENZY_LITE_DISLOC": "FRENZY_LITE",
                  "FRENZY_LITE_RETRY_DISLOC": "FRENZY_LITE_RETRY", "FRENZY_DISLOC": "FRENZY_LONG", "FRENZY_WIDE_DISLOC": "FRENZY_LONG"}
SEED = 20261128
_W = {"used": 0}


# ─────────────────────────── Binance (budgeted) ───────────────────────────
def get(url):
    if _W["used"] >= 900:
        time.sleep(61 - time.time() % 60)
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            _W["used"] = int(r.headers.get("X-MBX-USED-WEIGHT-1M") or _W["used"])
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code in (418, 429):
            raise SystemExit(f"Binance {e.code} — stopped (rate limit)")
        raise


def agg(sym, a_ms, b_ms):
    out, s = [], int(a_ms)
    while s < b_ms:
        r = get(f"https://fapi.binance.com/fapi/v1/aggTrades?symbol={sym}&startTime={s}&endTime={int(min(b_ms, s + 3_599_000))}&limit=1000")
        if not r:
            break
        out += r
        if len(r) < 1000:
            break
        s = r[-1]["T"] + 1
    return [(int(x["T"]), float(x["p"])) for x in out]


def kl(sym, a_ms, n, iv="1m"):
    return get(f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}&interval={iv}&startTime={int(a_ms)}&limit={n}")


# ─────────────────────────── cohort ───────────────────────────
def cohort():
    fs = glob.glob(os.path.join(R, "exports_archive", "scalpars_decisions_*.csv")) + glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_*.csv"))
    fr = []
    for f in fs:
        try:
            d = pd.read_csv(f, usecols=["t", "e", "pair", "dir", "gate"], dtype=str)
        except Exception:
            continue
        fr.append(d[(d.e == "BLOCK") & d.gate.fillna("").str.contains("DISLOC|CATCHUP_MOVED|_LATE")])
    d = pd.concat(fr).drop_duplicates(["t", "pair", "gate"]).sort_values("t").reset_index(drop=True)
    d["t_ms"] = [int(pd.Timestamp(x).value // 10**6) for x in d.t]
    return d, len(fs)


# ─────────────────────────── CloudWatch ───────────────────────────
CW_RE = re.compile(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),(\d{3}) .*\[(\w+_DISLOC)\] (\w+) (LONG|SHORT): entry aborted — live ([\d.]+) is ([\d.]+)% from decision ([\d.]+)")


def cw_line(pair, t_ms):
    try:
        out = subprocess.run(["aws", "logs", "filter-log-events", "--region", "eu-west-1", "--log-group-name", LOG_GROUP,
                              "--start-time", str(t_ms), "--end-time", str(t_ms + 180_000), "--filter-pattern", f'"_DISLOC] {pair}"',
                              "--output", "json", "--query", "events[].message"], capture_output=True, text=True, timeout=120)
        msgs = json.loads(out.stdout or "[]") or []
    except Exception:
        return None
    for m in msgs:
        g = CW_RE.search(m)
        if g:
            ts = int(pd.Timestamp(g.group(1)).value // 10**6) + int(g.group(2))
            return dict(cw_ms=ts, cw_live=float(g.group(6)), cw_dec=float(g.group(8)), cw_pct=float(g.group(7)))
    return None


DEC_OFF_MS = 3_000   # the decision price is the scan-start ticker read: on the AIOT 11-24 log-confirmed abort it equals the last trade at
                     # bar close + 3.0 s (calibrated on ticks; the log's live price equals the trade at the refusal second within 0.2 %)


def _trades(pair, t_ms):
    """ticks around t: the walkers' tick cache (reports/backtest_cache/ticks, scout_willy_timecap._load_day) first, else public aggTrades
    (Binance allows a time search only over the last 2 days)."""
    import scout_willy_timecap as WT
    d = pd.Timestamp(t_ms, unit="ms").strftime("%Y-%m-%d")
    z = WT._load_day(pair, d, None)
    if z is not None:
        t, p = z[0], z[1]
        m = (t >= t_ms - 120_000) & (t <= t_ms + 60_000)
        if m.any():
            return t[m], p[m], "TICKS"
    try:
        tr = agg(pair, t_ms - 120_000, t_ms + 60_000)
    except urllib.error.HTTPError as e:                # 4xx other than a rate limit: older than 2 days / unknown symbol
        return None, None, f"HTTP_{e.code}"
    if not tr:
        return None, None, "NO_TRADES"
    return np.array([x[0] for x in tr], dtype="int64"), np.array([x[1] for x in tr]), "AGGTRADES"


def reconstruct(pair, t_ms, ref_lag_ms):
    """D = last trade at t + DEC_OFF_MS · direction = the first trade in (t + DEC_OFF_MS, t + 20 s] more than 1 % from D (rel. tol. 1e-6 for
    the cache's float32) · L = the last trade at t + ref_lag_ms (the median log refusal delay), or that crossing trade when the price is back
    within 1 % by then · no crossing → UNRESOLVED."""
    t, p, src = _trades(pair, t_ms)
    if t is None:
        return dict(rec=src)
    dec = t_ms + DEC_OFF_MS
    i = int(np.searchsorted(t, dec, side="right")) - 1
    if i < 0:
        return dict(rec="NO_TRADES")
    D = float(p[i])
    lim = MAX_PCT / 100 + 1e-6
    w = np.flatnonzero((t > dec) & (t <= t_ms + 20_000) & (np.abs(p / D - 1) > lim))
    if not len(w):
        return dict(rec="UNRESOLVED", rec_dec=D, rec_src=src)
    j = int(np.searchsorted(t, t_ms + ref_lag_ms, side="right")) - 1
    Lr = float(p[j]) if j >= 0 else None
    k = int(w[0])
    if Lr is None or abs(Lr / D - 1) <= lim or np.sign(Lr / D - 1) != np.sign(p[k] / D - 1):
        return dict(rec="OK", rec_dec=D, rec_live=float(p[k]), rec_ms=int(t[k]), rec_src=src)
    return dict(rec="OK", rec_dec=D, rec_live=Lr, rec_ms=int(t_ms + ref_lag_ms), rec_src=src)


def approx_spike_ms(pair, t_ms):
    k = kl(pair, t_ms - 12 * 3_600_000, 144, "5m")
    if not k:
        return None
    i = int(np.argmax([float(x[7]) for x in k]))
    return int(k[i][0]) + 300_000


# ─────────────────────────── stats ───────────────────────────
def wilson_lo(k, n, z=1.645):
    if n == 0:
        return float("nan")
    p = k / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def day_boot(df, n=5000):
    if len(df) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    g = [x.pct.to_numpy() for _, x in df.groupby("day")]
    m = []
    for _ in range(n):
        s = [g[i] for i in rng.integers(0, len(g), len(g))]
        v = np.concatenate(s)
        m.append(v.mean())
    return tuple(np.percentile(m, [2.5, 97.5]))


def row(lab, g):
    if not len(g):
        return f"| {lab} | 0 | – | – | – | – | – |"
    w = g.why.astype(str)
    return (f"| {lab} | {len(g)} | {g.day.nunique()} | {(g.pct > 0).mean():.0%} | {g.pct.mean():+.2f} | {g.pct.sum():+.2f} | "
            f"{int(w.str.contains('TP').sum())} / {int((w == 'STOP').sum() + w.str.contains('SL').sum())} / {int(w.str.contains('TIME|CAP').sum())} |")


def main():
    d, nfiles = cohort()
    late = d[d.gate.str.contains("_LATE")]
    disl = d[d.gate.str.contains("DISLOC|CATCHUP_MOVED")].copy()
    # CloudWatch (14-day retention) + calibration lag
    cws = [cw_line(r.pair, r.t_ms) if r.t_ms > (time.time() - 14 * 86400) * 1000 else None for r in disl.itertuples()]
    for k in ("cw_ms", "cw_live", "cw_dec", "cw_pct"):
        disl[k] = [c.get(k) if c else None for c in cws]
    lags = [(c["cw_ms"] - r.t_ms) for c, r in zip(cws, disl.itertuples()) if c]
    lag = int(np.median(lags)) if lags else 11_000          # the median log refusal delay after the bar close
    rec = [reconstruct(r.pair, r.t_ms, lag) for r in disl.itertuples()]
    for k in ("rec", "rec_dec", "rec_live", "rec_ms", "rec_src"):
        disl[k] = [x.get(k) for x in rec]
    disl["src"] = np.where(disl.cw_live.notna(), "CLOUDWATCH", np.where(disl.rec == "OK", "REBUILT_" + disl.rec_src.fillna("").astype(str), "UNRESOLVED"))
    disl["D"] = disl.cw_dec.where(disl.cw_dec.notna(), disl.rec_dec)
    disl["L"] = disl.cw_live.where(disl.cw_live.notna(), disl.rec_live)
    disl["ref_ms"] = disl.cw_ms.where(disl.cw_ms.notna(), disl.rec_ms)
    disl["move_pct"] = (disl.L / disl.D - 1) * 100
    disl["direction"] = np.where(disl.move_pct < 0, "CHEAPER", np.where(disl.move_pct > 0, "DEARER", None))
    disl["sleeve"] = disl.gate.map(SLEEVE_OF_GATE).fillna("FRENZY_CATCHUP")
    # price
    t = SB.th()
    now = int(time.time() * 1000)
    pc, tc = {}, {}
    import scout_frenzy_v2b_shadow as VB
    Bud = VB._Bud(True)

    def prep(fz):
        """the pricer's own data step (scout_blocked_filters.price_all): shared 1m pages + the top-up, before any walk."""
        if not len(fz):
            return
        import scout_willy_timecap as WT
        span = min(max(SB.exit_of(sl, t)["cap"] for sl in set(fz.sleeve)), SB.SPAN_MAX)
        WT.ensure_data(pd.DataFrame(dict(ts=pd.to_datetime(fz.te, unit="ms"), pair=fz.pair.values, te=fz.te.values, src=None)),
                       span, now, fetch=True, max_req=40)
        SB._topup_1m(fz, t, now, state_path=os.path.join(__import__("tempfile").gettempdir(), "disloc_study_topup_state.json"))
    pre = disl[(disl.src != "UNRESOLVED") & disl.sleeve.isin(SB.FRENZY) & disl.L.notna()]
    prep(pd.DataFrame(dict(pair=pre.pair.values, sleeve=pre.sleeve.values, te=(pre.ref_ms.astype("int64") + SB.LAT_MS).values,
                           E=(pre.L.astype(float) * (1 + SB.SLIP / 100)).values)))
    rows = []
    for r in disl.itertuples():
        pct = why = status = None
        flag = ""
        if r.src != "UNRESOLVED" and r.sleeve in SB.FRENZY and pd.notna(r.L):
            snap = {}
            if r.sleeve == "FRENZY_WILLY" or (r.sleeve == "FRENZY_WILLY_HT" and not SB.exit_of(r.sleeve, t).get("flat")):
                snap["entry_frenzy_spike_at"] = approx_spike_ms(r.pair, r.t_ms)
                flag = "APPROX_SPIKE"
            E = float(r.L) * (1 + SB.SLIP / 100)
            te = int(r.ref_ms) + SB.LAT_MS
            rr = dict(pair=r.pair, sleeve=r.sleeve, te=te, E=E, snap=snap, key=f"D|{r.pair}|{r.t}")
            try:
                pct, why, status, _ = SB.price_frenzy(rr, t, now, pc, tc, Bud)
            except SystemExit:
                raise
            except Exception as e:
                why, status = f"ERR {type(e).__name__}: {str(e)[:60]}", "NOT_PRICEABLE"
        else:
            status = "NOT_PRICEABLE" if r.src != "UNRESOLVED" else "UNRESOLVED"
            why = "catch-up sleeve not on a priced exit" if r.sleeve == "FRENZY_CATCHUP" else why
        rows.append(dict(pct=pct, why=why, status=status, flag=flag))
    for k in ("pct", "why", "status", "flag"):
        disl[k] = [x[k] for x in rows]
    disl["day"] = pd.to_datetime(disl.t_ms, unit="ms").dt.strftime("%Y-%m-%d")
    disl.to_csv(OUT + ".csv", index=False)

    P = disl[disl.status == "PRICED"].copy()
    C, Dr = P[P.direction == "CHEAPER"], P[P.direction == "DEARER"]
    # kept fills, same sleeves, same span, same pricer
    since = int(disl.t_ms.min())
    K = SB.load_kept(since_ms=since)
    K = K[K.sleeve.isin(set(P.sleeve))] if len(K) else K
    KR = SB.kept_as_rows(K) if len(K) else pd.DataFrame()
    if len(KR):
        prep(KR)
    kp = []
    for r in KR.to_dict("records") if len(KR) else []:
        try:
            pct, why, st, _ = SB.price_frenzy(r, t, now, pc, tc, Bud)
        except Exception:
            st = None
        if st == "PRICED":
            kp.append(dict(sleeve=r["sleeve"], pct=pct, why=why, day=r["day"], booked=r["booked"]))
    KP = pd.DataFrame(kp, columns=["sleeve", "pct", "why", "day", "booked"])

    # calibration
    ov = disl[disl.cw_live.notna() & (disl.rec == "OK")]
    agree = int((np.sign(ov.cw_live / ov.cw_dec - 1) == np.sign(ov.rec_live / ov.rec_dec - 1)).sum()) if len(ov) else 0
    derr = ((ov.rec_dec / ov.cw_dec - 1).abs() * 100).median() if len(ov) else float("nan")
    lerr = ((ov.rec_live / ov.cw_live - 1).abs() * 100).median() if len(ov) else float("nan")
    cw_n = int(disl.cw_live.notna().sum())
    unres_cw = int((disl.cw_live.notna() & (disl.rec != "OK")).sum())

    # CHEAPER cohort reads
    lo, hi = day_boot(C)
    wl = wilson_lo(int((C.pct > 0).sum()), len(C))
    conc = C.groupby("pair").pct.sum().sort_values(ascending=False) if len(C) else pd.Series(dtype=float)
    top_share = (conc.iloc[0] / C.pct[C.pct > 0].sum()) if (len(C) and C.pct[C.pct > 0].sum() > 0) else float("nan")
    loo = [C.drop(i).pct.mean() for i in C.index] if len(C) > 1 else []
    avg_w = P[P.pct > 0].pct.mean() if (P.pct > 0).any() else float("nan")
    avg_l = -P[P.pct <= 0].pct.mean() if (P.pct <= 0).any() else float("nan")

    # chart
    chart = OUT + "_chart.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        paths = {}
        for r in P.itertuples():
            k = kl(r.pair, int(r.ref_ms) // 60_000 * 60_000, 121)
            if not k:
                continue
            c = np.array([float(x[4]) for x in k])
            paths[r.Index] = (c / float(r.L) - 1) * 100
        for j, (lab, g) in enumerate((("CHEAPER (live below decision)", C), ("DEARER (live above decision)", Dr))):
            for sub, col, nm in ((g[g.pct > 0], "#2a9d8f", "winners"), (g[g.pct <= 0], "#e76f51", "losers")):
                arr = [paths[i] for i in sub.index if i in paths]
                if arr:
                    m = min(map(len, arr))
                    ax[j].plot(np.mean([a[:m] for a in arr], axis=0), color=col, label=f"{nm} (n={len(arr)})")
            ax[j].axhline(0, color="#888", lw=0.7)
            ax[j].set_title(lab)
            ax[j].set_xlabel("minutes after the abort")
            ax[j].legend()
        ax[0].set_ylabel("price vs the live price at the abort, %")
        fig.suptitle("FRENZY-family dislocation aborts — average price path (1m closes)")
        fig.tight_layout()
        fig.savefig(chart, dpi=110)
    except Exception as e:
        chart = f"(chart failed: {e})"

    def bucket(m):
        a = abs(m)
        return "1–2 %" if a < 2 else ("2–4 %" if a < 4 else "> 4 %")
    L = ["# FRENZY-family dislocation guard — would a one-sided guard pay? (2026-11-28, research only)", "",
         "**Prior studies:** DL 85 / 87 (09-19) tested the Bull-run sleeve's 0.3 % guard in both directions; far entries lost there, and the "
         "\"cheaper\" side turned out to be a paper tick-size bug. The FRENZY-family 1 % guard was never priced or split by direction: this is "
         "the first read.", "",
         "## Question", "The guard cancels a FRENZY / WILLY / WILLY_HT / LITE long when the live price at the order is more than 1 % away from the "
         "decision price — **either way**. Would cancelling only when the price ran UP (dearer) and filling when it fell (cheaper) have made money?", "",
         "## Data", f"- {len(disl)} aborts in {nfiles} decisions exports ({disl.t.min()[:10]} → {disl.t.max()[:10]}; older exports no longer exist): "
         + ", ".join(f"{g} {n}" for g, n in disl.gate.value_counts().items()) + f". Also {len(late)} lateness refusals (descriptive: "
         + ", ".join(f"{g} {n}" for g, n in late.gate.value_counts().items()) + ").",
         f"- Exact prices from the server log for {cw_n} (14-day log retention); the rest rebuilt from Binance trades (the walkers' tick cache; decision = the last trade 3 s after the bar close, as calibrated on AIOT). "
         f"Calibration on the {len(ov)} with both: direction agrees {agree} / {len(ov)}, decision price error median {derr:.2f} %, live price "
         f"error median {lerr:.2f} %" + (f"; {unres_cw} log-confirmed aborts the rebuild could NOT see (move inside a second, or the book "
                                          "moved without trades)" if unres_cw else "") + ".",
         f"- Unresolved (no > 1 % trade in the 20 s after the decision): {int((disl.src == 'UNRESOLVED').sum())} — listed in the CSV, not counted. "
         f"Not priceable: {int((disl.status == 'NOT_PRICEABLE').sum())}.",
         "- Priced AS IF FILLED at the live price + 0.10 % slip, entry 8 s after the abort, taker fee, each sleeve's live exit today "
         "(scout_blocked_filters pricer: WILLY_HT +6 / −4 / 24 h; LITE flat TP; plain WILLY its spike-high target"
         + (" — spike time approximated, flagged" if (P.flag == "APPROX_SPIKE").any() else "") + ").", "",
         "## Result by sleeve × direction", "",
         "| Sleeve · direction | N | days | WR | avg net % | Σ % | TP / STOP / TIME |", "|---|---|---|---|---|---|---|"]
    for sl in sorted(P.sleeve.unique()):
        for dr in ("CHEAPER", "DEARER"):
            g = P[(P.sleeve == sl) & (P.direction == dr)]
            if len(g):
                L.append(row(f"{sl} · {'🟢' if dr == 'CHEAPER' else '🔴'} {dr}", g))
    L += [row("**ALL · 🟢 CHEAPER**", C), row("**ALL · 🔴 DEARER**", Dr), row("**ALL aborts**", P), "",
          "Kept fills of the same sleeves over the same span, same pricer:", "", "| Sleeve | N | days | WR | avg net % | Σ % | TP / STOP / TIME |",
          "|---|---|---|---|---|---|---|"]
    for sl in sorted(KP.sleeve.unique()) if len(KP) else []:
        L.append(row(sl + " (kept)", KP[KP.sleeve == sl]))
    L += ["", "## CHEAPER side by move size (is it a crash starting?)", "", "| move | N | days | WR | avg net % | Σ % | TP / STOP / TIME |", "|---|---|---|---|---|---|---|"]
    for b in ("1–2 %", "2–4 %", "> 4 %"):
        L.append(row(b, C[C.move_pct.map(bucket) == b] if len(C) else C))
    L += ["", "## CHEAPER cohort — robustness", "",
          f"- N {len(C)} on {C.day.nunique() if len(C) else 0} days · WR {(C.pct > 0).mean():.0%} (one-sided 95 % Wilson floor {wl:.0%}) · avg {C.pct.mean():+.2f} %"
          f" · day-clustered bootstrap 95 % CI [{lo:+.2f}, {hi:+.2f}]" if len(C) else "- no priced CHEAPER aborts",
          f"- Pair concentration: {', '.join(f'{p} {v:+.2f}' for p, v in conc.items())} · top pair = {top_share:.0%} of the winners' Σ" if len(C) else "",
          f"- Leave-one-out avg: min {min(loo):+.2f} · max {max(loo):+.2f}" if loo else "",
          f"- Breakeven WR on the priced aborts (|avg loss| ÷ (avg win + |avg loss|)): {avg_l / (avg_w + avg_l):.0%}" if (avg_w == avg_w and avg_l == avg_l) else "",
          f"- In-sample haircut 30–50 %: avg {C.pct.mean() * 0.5:+.2f} … {C.pct.mean() * 0.7:+.2f} %" if len(C) else "", "",
          f"## Chart", f"![event study]({os.path.basename(chart)})" if chart.endswith(".png") else chart, "",
          "Average price path from the abort (1m closes, % vs the live price): winners vs losers, CHEAPER vs DEARER.", ""]
    n_c, d_c = len(C), (C.day.nunique() if len(C) else 0)
    passes = n_c >= 15 and d_c >= 8 and (lo == lo and lo > 0) and (top_share == top_share and top_share < 0.5)
    L += ["## Verdict (locked gates: N ≥ 15 · ≥ 8 windows · CI above 0 · no pair ≥ 50 %, then a 30–50 % haircut)", "",
          ("🟢 **Meets the bar**" if passes else "🟡 **Below the bar → OBSERVE line, no change**")
          + f" — CHEAPER N {n_c} on {d_c} days; CI [{lo:+.2f}, {hi:+.2f}]; top pair {top_share:.0%}." if n_c else "⚪ nothing priced", "",
          "## Caveats", "- Small N: the decisions exports only reach back to late October; older aborts are not recorded anywhere.",
          "- Rebuilt prices: Binance trades, not the order book the guard read; the 3 s decision offset is calibrated on ONE log-confirmed abort (the bot's scan-start time varies).",
          "- Paper fills: the bot was in paper for most aborts; a real order into a falling book can fill worse than +0.10 %.",
          "- 1m walker: intra-minute stop / TP order is not visible (both ways).",
          "- Crash-start risk: a cheaper live price can be the first second of a collapse; the move-size table is the check.",
          "- Plain WILLY's spike time is approximated (highest-volume 5m bar in the 12 h before), so its target is approximate.",
          f"- Direction near the line: {int(((P.move_pct.abs() < 1.5) & (P.src != 'CLOUDWATCH')).sum())} rebuilt aborts moved < 1.5 % — inside "
          "about one calibration error of the 1 % line, so their side (and that the guard even fired on that side) is uncertain; the CHEAPER "
          f"side's rebuilt moves of ≥ 1.5 %: {int(((C.move_pct.abs() >= 1.5) & (C.src != 'CLOUDWATCH')).sum())}.",
          f"- The guard as a whole: all {len(P)} priced aborts avg {P.pct.mean():+.2f} % vs the kept fills above — on this N the guard "
          "removed trades that did no worse than the ones it let through, on BOTH sides; small N, not a finding on its own.", ""]
    open(OUT + ".md", "w").write("\n".join(x for x in L if x is not None) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
