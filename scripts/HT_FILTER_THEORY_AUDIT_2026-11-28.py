#!/usr/bin/env python3
"""HT_FILTER_THEORY_AUDIT_2026-11-28 — theory-first audit of the FRENZY_WILLY_HT entry blockers (research only, never changes the bot).
(operator 2026-11-28: "re-audit all here, to make if they really have fundamentals behind to be a filter. Theory first!")

The theories and their PATH predictions were written before any new number was read (top of the report). Each filter is judged on whether
its OWN mechanism's predictions hold — not only on whether its blocked trades lose.

COHORTS
  BACKTEST  reports/HT_BLOCKERS_AUDIT_2026-11-27.csv — 1,142 unfiltered HT signals 2025-01 → 2026-11-17 priced at today's HT exit
            (TP +6 net / SL −4 / 24 h, 1m, funding) with FS / NH / BU flags (DL 655 machinery, re-used, not re-priced).
  REAL      every FRENZY_WILLY_HT fill in exports_archive orders exports (dedup opened_at + pair, probes out) + every priced live HT refusal
            (reports/HT_BLOCKERS_LIVE_2026-11-27.csv) — all re-walked here with the SAME function (HT_BLOCKERS_AUDIT one(): entry =
            1m open × 1.001, today's exit) so blocked and kept sit on one ruler.
PATH      per setup on the 1m grid: best move in the trade · best move in 24 h · cleared the spike high H in 24 h · came within 1 % of H ·
          WORKED = +1 % before the −4 % stop (scripts/path_labels.py definition, the stop at HT's) · move at +30 / +60 min · funding rate of
          the next settlement + minutes to it · funding paid in the trade.
STATS     day-clustered bootstrap 95 % CI (2,000) · label-shuffle null on the prediction statistic (2,000) · LOO on real cohorts · per
          half-year · BTC daily trend sign (1d EMA50 vs EMA200, the last closed day).
Out: reports/HT_FILTER_THEORY_AUDIT_2026-11-28.md · .csv (one row per setup, path features) · _chart.png"""
import glob
import importlib
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, "scripts")); sys.path.insert(0, R)
C = importlib.import_module("STATE_EXIT_2026-11-21_common")
AU = importlib.import_module("HT_BLOCKERS_AUDIT_2026-11-27")

MIN, HR, DAY = 60_000, 3_600_000, 86_400_000
OUT = os.path.join(R, "reports", "HT_FILTER_THEORY_AUDIT_2026-11-28")
RNG = np.random.default_rng(20261128)
TP_NET, SL_NET = 6.0, 4.0
SCR = os.environ.get("HT_AUDIT_SCRATCH", os.path.join(__import__("tempfile").gettempdir(), "ht_filter_theory_audit"))   # intermediate tables (not a deliverable)
os.makedirs(SCR, exist_ok=True)


# ─────────────────────────── data ───────────────────────────
def real_cohort():
    fs = sorted(glob.glob(os.path.join(R, "exports_archive", "scalpars_orders_*.csv")))
    cols = ("opened_at", "pair", "entry_strategy", "status", "is_probe", "entry_frenzy_willy_spike_high", "close_reason")
    d = pd.concat([pd.read_csv(f, low_memory=False, usecols=lambda c: c in cols) for f in fs], ignore_index=True)
    d = d[(d.entry_strategy.astype(str) == "FRENZY_WILLY_HT") & (d.status.astype(str) == "CLOSED")]
    d = d[~d.get("is_probe", pd.Series(False, index=d.index)).astype(str).str.lower().isin(["true", "1", "1.0"])]
    d = d.drop_duplicates(["opened_at", "pair"])
    F = pd.DataFrame(dict(pair=d.pair.values, te=[int(pd.Timestamp(x).value // 10**6) for x in d.opened_at],
                          H=pd.to_numeric(d.entry_frenzy_willy_spike_high, errors="coerce").values, kind="FILL", lines=""))
    b = pd.read_csv(os.path.join(R, "reports", "HT_BLOCKERS_LIVE_2026-11-27.csv"))
    b = b[b.pct.notna()]
    B = pd.DataFrame(dict(pair=b.pair.values, te=[int(pd.Timestamp(x).value // 10**6) for x in b.ts], H=np.nan, kind="REFUSED",
                          lines=b.lines.astype(str).values))
    return pd.concat([F, B], ignore_index=True).sort_values("te").reset_index(drop=True)


def ensure_klines(A):
    need = sorted({(p, ds) for p, te in zip(A.pair, A.te) for ds in C.days_of(int(te) - 6 * HR, int(te) + 25 * HR)})
    got = 0
    for i, (p, ds) in enumerate(need):
        got += C.fetch_kday(p, ds) or 0
    return len(need), got


def btc_trend():
    """last CLOSED day's sign of EMA50 − EMA200 on BTCUSDT 1d → dict day → +1 / −1."""
    u = "https://fapi.binance.com/fapi/v1/klines?symbol=BTCUSDT&interval=1d&limit=1000"
    k = json.load(urllib.request.urlopen(u, timeout=20))
    s = pd.Series([float(r[4]) for r in k], index=pd.to_datetime([r[0] for r in k], unit="ms"))
    g = (s.ewm(span=50, adjust=False).mean() - s.ewm(span=200, adjust=False).mean()).shift(1)   # the last CLOSED day
    return {d.strftime("%Y-%m-%d"): (1 if v > 0 else -1) for d, v in g.dropna().items()}


def path(r):
    """path features from the entry bar (same entry as the pricer: 1m open × 1.001)."""
    p, te = r["pair"], int(r["te"])
    a = C.k1m(p, te - 6 * HR, te + 24 * HR + MIN)
    if a is None or not len(a["t"]):
        return {}
    g = C.grid(a, te - 6 * HR, te + 24 * HR + MIN)
    i0 = int(np.searchsorted(g["t"], te))
    if i0 >= len(g["t"]):
        return {}
    E = float(g["o"][i0]) * 1.001
    end = min(i0 + 24 * 60, len(g["t"]))
    h, l, c = g["h"][i0:end], g["l"][i0:end], g["c"][i0:end]
    H = float(r["H_used"]) if r.get("H_used") == r.get("H_used") and r.get("H_used") else np.nan
    stop_px = E * (100 - SL_NET + C.TAKER) / (100 - C.TAKER)
    tp_px = E * (100 + TP_NET + C.TAKER) / (100 - C.MAKER)
    j = int(np.searchsorted(g["t"], int(r["xt"]))) - i0 if r.get("xt") == r.get("xt") else len(h) - 1
    j = max(0, min(j, len(h) - 1))
    up = np.nonzero(h >= E * 1.01)[0]; dn = np.nonzero(l <= stop_px)[0]
    worked = bool(len(up) and (not len(dn) or up[0] < dn[0]))
    fs_t, fs_r = C.funding(p)
    nx = np.nonzero(fs_t > te)[0]
    f_rate = float(fs_r[nx[0]]) * 100 if len(nx) else np.nan
    f_min = (int(fs_t[nx[0]]) - te) / MIN if len(nx) else np.nan
    f_int = (int(fs_t[nx[0] + 1]) - int(fs_t[nx[0]])) / HR if len(nx) and nx[0] + 1 < len(fs_t) else np.nan
    o = dict(E2=E, mfe_trade=(np.nanmax(h[:j + 1]) / E - 1) * 100, mfe24=(np.nanmax(h) / E - 1) * 100,
             mae24=(np.nanmin(l) / E - 1) * 100, worked=worked,
             r30=(c[min(30, len(c) - 1)] / E - 1) * 100, r60=(c[min(60, len(c) - 1)] / E - 1) * 100,
             tp_above_H=(tp_px > H) if H == H else np.nan, f_rate=f_rate, f_min=f_min, f_int_h=f_int)
    if H == H:
        mx24, mxt = np.nanmax(h), np.nanmax(h[:j + 1])
        o.update(clear_H=bool(mx24 >= H), clear_H_trade=bool(mxt >= H), near_H_trade=bool(mxt >= H * 0.99),
                 stall_H=bool(H * 0.98 <= mxt < H))
    return o


# ─────────────────────────── stats ───────────────────────────
def boot_ci(x, day, n=2000):
    x, day = np.asarray(x, float), np.asarray(day)
    u = np.unique(day)
    if len(x) < 3 or len(u) < 3:
        return (np.nan, np.nan)
    idx = {k: np.nonzero(day == k)[0] for k in u}
    m = [np.concatenate([x[idx[k]] for k in RNG.choice(u, len(u))]).mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def diff_ci(a, da, b, db, n=2000):
    """day-clustered bootstrap CI of mean(a) − mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or len(b) < 3:
        return (np.nan, np.nan)
    ua, ub = np.unique(da), np.unique(db)
    ia = {k: np.nonzero(np.asarray(da) == k)[0] for k in ua}; ib = {k: np.nonzero(np.asarray(db) == k)[0] for k in ub}
    m = []
    for _ in range(n):
        xa = np.concatenate([a[ia[k]] for k in RNG.choice(ua, len(ua))]); xb = np.concatenate([b[ib[k]] for k in RNG.choice(ub, len(ub))])
        m.append(xa.mean() - xb.mean())
    return tuple(np.percentile(m, [2.5, 97.5]))


def shuffle_p(y, flag, n=2000):
    """P(|shuffled diff| ≥ |observed diff|) — label shuffle of the flag (two-sided)."""
    y, flag = np.asarray(y, float), np.asarray(flag, bool)
    ok = ~np.isnan(y)
    y, flag = y[ok], flag[ok]
    if flag.sum() < 3 or (~flag).sum() < 3:
        return np.nan
    obs = abs(y[flag].mean() - y[~flag].mean())
    k = flag.sum()
    c = 0
    for _ in range(n):
        s = RNG.permutation(len(y))[:k]
        m = np.zeros(len(y), bool); m[s] = True
        c += abs(y[m].mean() - y[~m].mean()) >= obs
    return (c + 1) / (n + 1)


def loo_range(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return (np.nan, np.nan)
    s = x.sum()
    v = [(s - xi) / (len(x) - 1) for xi in x]
    return min(v), max(v)


def st(g, col="net"):
    if not len(g):
        return "0"
    return f"{len(g)} · {(g[col] > 0).mean():.0%} · {g[col].mean():+.2f}"


def pct(g, col):
    v = g[col].dropna().astype(float)
    return f"{v.mean():.0%}" if len(v) else "–"


def med(g, col):
    v = g[col].dropna().astype(float)
    return f"{v.median():+.2f}" if len(v) else "–"


def tbl(rows, head):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]


# ─────────────────────────── main ───────────────────────────
def main():
    t0 = time.time()
    BT = pd.read_csv(os.path.join(R, "reports", "HT_BLOCKERS_AUDIT_2026-11-27.csv"))
    BT["cohort"] = "BACKTEST"
    RL = real_cohort()
    n_need, n_got = ensure_klines(RL)
    rows = []
    for r in RL.to_dict("records"):
        o = AU.one(r)
        if o:
            rows.append({**r, **o})
    RL = pd.DataFrame(rows)
    RL["FS"] = RL.lines.astype(str).str.contains("FAILED_SPIKE")   # fills: FS unknown before its 11-12 arming → treated as kept (blind spot)
    RL["NH"] = RL.dip < 1.4
    RL["BU"] = RL.burst == "zone"
    RL["day"] = pd.to_datetime(RL.te, unit="ms").dt.strftime("%Y-%m-%d")
    RL["half"] = "2026-H2"
    RL["cohort"] = "REAL"
    ALL = pd.concat([BT, RL], ignore_index=True)
    feats = [path(r) for r in ALL.to_dict("records")]
    ALL = pd.concat([ALL.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    try:
        bt = btc_trend()
    except Exception as e:
        bt = {}; print("btc trend unread", e)
    ALL["btc_trend"] = ALL.day.map(bt)
    ALL.to_csv(OUT + ".csv", index=False)
    B = ALL[ALL.cohort == "BACKTEST"].copy(); RE = ALL[ALL.cohort == "REAL"].copy()
    print(f"cohorts: backtest {len(B)} · real {len(RE)} (fills {int((RE.kind == 'FILL').sum())} · refusals {int((RE.kind == 'REFUSED').sum())}) "
          f"· klines {n_need} pair-days ({n_got} fetched) · {time.time() - t0:.0f}s", flush=True)
    res = {}
    res["NH"] = audit_nh(B, RE)
    res["BU"] = audit_bu(B, RE)
    res["FU"] = audit_fu(B, RE)
    chart(ALL)
    write(res, B, RE)


def audit_nh(B, RE):
    L = []
    base = B[~B.FS & ~B.BU]      # what the other two armed blockers let through (DL 655 convention)
    bins = [-1e9, 0, 0.7, 1.4, 3, 6, 1e9]
    labs = ["above H (<0)", "0–0.7", "0.7–1.4", "1.4–3", "3–6", "> 6"]
    rows = []
    for nm, g in base.groupby(pd.cut(base.dip, bins, labels=labs), observed=False):
        rows.append([nm, st(g), pct(g, "worked"), med(g, "mfe_trade"), pct(g, "near_H_trade"), pct(g, "clear_H"),
                     (f"{(g.clear_H_trade.astype(float)[g.near_H_trade.astype(bool)]).mean():.0%}" if g.near_H_trade.astype(bool).any() else "–"),
                     pct(g, "stall_H"), pct(g, "tp_above_H")])
    L += ["**Dose-response by dip below the spike high (backtest, rows FAILED_SPIKE and BURST keep):**", ""]
    L += tbl(rows, ["dip %", "N · WR · avg", "WORKED", "best move in trade (med)", "came ≤ 1 % of H", "cleared H in 24 h",
                    "cleared H | came near", "stalled 0–2 % under H", "TP above H"])
    # P3: TP-vs-H mechanics
    q = base[base.tp_above_H.notna()].copy(); q["tpH"] = q.tp_above_H.astype(bool)
    rows = []
    for (a, b_), g in q.groupby(["tpH", "NH"]):
        rows.append([("TP above H" if a else "TP below H"), ("dip < 1.4" if b_ else "dip ≥ 1.4"), st(g), pct(g, "worked"), med(g, "mfe24")])
    L += ["", "**TP-vs-H mechanics (P3):** the +6 TP sits above the spike high whenever the dip is < ≈ 5.7 % — so NEAR_HIGH could be a "
          "mechanical 'needs a new high' effect rather than overhead supply. Inside TP-above-H setups only:", ""]
    L += tbl(rows, ["TP vs H", "dip", "N · WR · avg", "WORKED", "best move 24 h (med)"])
    ta = q[q.tpH]
    nh, rest = ta[ta.NH], ta[~ta.NH]
    d = (nh.net.mean() - rest.net.mean()) if len(nh) and len(rest) else np.nan
    ci = diff_ci(nh.net, nh.day, rest.net, rest.day)
    pnull = shuffle_p(ta.net, ta.NH)
    L += ["", f"Inside TP-above-H: dip < 1.4 vs 1.4–5.7 → Δ avg {d:+.2f} pts (day-CI {ci[0]:+.2f} … {ci[1]:+.2f}, shuffle p {pnull:.3f})."]
    # P1 stall: shuffle on stall share NH vs rest within near-H reachers
    sp = shuffle_p(base.stall_H.astype(float), base.NH) if base.stall_H.notna().any() else np.nan
    # per period
    rows = []
    for hf, g in base.groupby("half"):
        a_, b2 = g[g.NH], g[~g.NH]
        rows.append([hf, st(a_), st(b2), pct(a_, "worked"), pct(b2, "worked"), pct(a_, "clear_H"), pct(b2, "clear_H")])
    for s_, g in base.groupby("btc_trend"):
        a_, b2 = g[g.NH], g[~g.NH]
        rows.append([f"BTC daily {'up' if s_ > 0 else 'down'}", st(a_), st(b2), pct(a_, "worked"), pct(b2, "worked"), pct(a_, "clear_H"), pct(b2, "clear_H")])
    L += ["", "**Per period / BTC daily trend (backtest):**", ""]
    L += tbl(rows, ["period", "blocked N · WR · avg", "kept N · WR · avg", "blocked WORKED", "kept WORKED", "blocked cleared H", "kept cleared H"])
    # real
    rn = RE[RE.dip.notna()]
    a_, b2 = rn[rn.NH], rn[~rn.NH & ~rn.FS & ~rn.BU]
    lo = loo_range(a_.net)
    L += ["", f"**Real setups (fills + priced refusals, re-walked):** dip < 1.4 {st(a_)} (WORKED {pct(a_, 'worked')}, cleared H {pct(a_, 'clear_H')}, "
          f"LOO {lo[0]:+.2f} … {lo[1]:+.2f}) vs kept-by-all {st(b2)} (WORKED {pct(b2, 'worked')}, cleared H {pct(b2, 'clear_H')})."]
    bl = base[base.NH]; kp = base[~base.NH]
    return dict(L=L, n=len(bl), avg_b=bl.net.mean(), avg_k=kp.net.mean(), ci=diff_ci(bl.net, bl.day, kp.net, kp.day),
                p=shuffle_p(base.net, base.NH), worked=(bl.worked.mean(), kp.worked.mean()), stall=(bl.stall_H.mean(), kp.stall_H.mean()),
                stall_p=sp, tpH=(d, ci, pnull), real=(len(a_), a_.net.mean() if len(a_) else np.nan, len(b2), b2.net.mean() if len(b2) else np.nan),
                cost=bl.net.mean())


def audit_bu(B, RE):
    L = []
    base = B[~B.FS & ~B.NH & B.pre_vol.notna() & B.cloc.notna()].copy()
    base["burst"] = base.pre_vol > 1.6
    cb = [-1, 0.25, 0.5, 0.75, 1.01]; cl = ["< 0.25", "0.25–0.5", "0.5–0.75", "≥ 0.75"]
    rows = []
    for bflag, g0 in base.groupby("burst"):
        for nm, g in g0.groupby(pd.cut(g0.cloc, cb, labels=cl), observed=False):
            rows.append([("burst (pre_vol > 1.6)" if bflag else "no burst"), nm, st(g), pct(g, "worked"), med(g, "r30"), med(g, "r60"), med(g, "mfe_trade")])
    L += ["**Close location × burst (backtest, rows FAILED_SPIKE and NEAR_HIGH keep):** the theory needs the low-close penalty INSIDE bursts "
          "only (B2 / B3).", ""]
    L += tbl(rows, ["volume", "cloc15", "N · WR · avg", "WORKED", "move +30 min (med)", "+60 min (med)", "best move in trade (med)"])
    pb = [-1, 1.0, 1.6, 3.0, 1e9]; pl = ["≤ 1.0", "1.0–1.6", "1.6–3", "> 3"]
    lowc = base[base.cloc < 0.5]
    rows = [[nm, st(g), pct(g, "worked"), med(g, "r30")] for nm, g in lowc.groupby(pd.cut(lowc.pre_vol, pb, labels=pl), observed=False)]
    L += ["", "**Low close (cloc < 0.5) by burst size (B4):**", ""]
    L += tbl(rows, ["pre_vol15_vs60", "N · WR · avg", "WORKED", "move +30 min (med)"])
    z, r = base[base.BU], base[~base.BU]
    nb = base[~base.burst]
    nb_low, nb_hi = nb[nb.cloc < 0.5], nb[nb.cloc >= 0.5]
    rows = []
    for hf, g in base.groupby("half"):
        rows.append([hf, st(g[g.BU]), st(g[~g.BU]), pct(g[g.BU], "worked"), pct(g[~g.BU], "worked")])
    for s_, g in base.groupby("btc_trend"):
        rows.append([f"BTC daily {'up' if s_ > 0 else 'down'}", st(g[g.BU]), st(g[~g.BU]), pct(g[g.BU], "worked"), pct(g[~g.BU], "worked")])
    L += ["", "**Per period / BTC daily trend (backtest):**", ""]
    L += tbl(rows, ["period", "zone N · WR · avg", "rest N · WR · avg", "zone WORKED", "rest WORKED"])
    rz = RE[RE.BU]; rr = RE[~RE.BU & RE.pre_vol.notna() & ~RE.NH & ~RE.FS]
    L += ["", f"**Real setups (re-walked, burst read from klines at the entry):** zone {st(rz)} (WORKED {pct(rz, 'worked')}, +30 min {med(rz, 'r30')}) "
          f"vs rest kept by all {st(rr)} (WORKED {pct(rr, 'worked')}, +30 min {med(rr, 'r30')})."]
    return dict(L=L, n=len(z), avg_b=z.net.mean(), avg_k=r.net.mean(), ci=diff_ci(z.net, z.day, r.net, r.day), p=shuffle_p(base.net, base.BU),
                worked=(z.worked.mean(), r.worked.mean()), worked_p=shuffle_p(base.worked.astype(float), base.BU),
                r30=(z.r30.median(), r.r30.median()), nb=(st(nb_low), st(nb_hi), nb_low.net.mean() - nb_hi.net.mean(),
                                                          shuffle_p(nb.net, nb.cloc < 0.5)),
                real=(len(rz), rz.net.mean() if len(rz) else np.nan, len(rr), rr.net.mean() if len(rr) else np.nan), cost=z.net.mean())


def audit_fu(B, RE):
    L = []
    allr = pd.concat([B, RE], ignore_index=True)
    kept = B[~B.FS & ~B.NH & ~B.BU]
    ev = kept.net.mean(); ci = boot_ci(kept.net, kept.day)
    rk = RE[(RE.kind == "FILL") & ~RE.FS & ~RE.NH & ~RE.BU]
    L += [f"**Economics (F1):** HT's expected value per trade at today's exit — backtest kept-by-all {st(kept)} (day-CI {ci[0]:+.2f} … {ci[1]:+.2f}); "
          f"real kept fills re-walked {st(rk)}. One funding payment costs ≈ the rate (mark-scaled). The guard (≥ 0.05 % within 15 min) "
          f"skips a trade worth ≈ {ev:+.2f} % to save ≈ 0.05–0.2 %: it pays only when E[trade | high funding] < the payment."]
    g = allr[allr.f_rate.notna()].copy()
    g["trig"] = (g.f_rate >= 0.05) & (g.f_min <= 15)
    near = g[(g.f_rate >= 0.05)]
    L += ["", f"Setups with the next rate ≥ +0.05 % (any time to it): {len(near)} of {len(g)} ({len(near) / max(len(g), 1):.1%}) · inside 15 min "
          f"(the guard's trigger): {int(g.trig.sum())} — outcome {st(g[g.trig])}. Hourly-funding coins at entry: {int((g.f_int_h <= 1.01).sum())}."]
    fb = [-1e9, 0.0, 0.01, 0.03, 0.05, 1e9]; fl = ["≤ 0", "0–0.01", "0.01–0.03", "0.03–0.05", "≥ 0.05"]
    rows = []
    for nm, x in g.groupby(pd.cut(g.f_rate, fb, labels=fl), observed=False):
        rows.append([nm, st(x), pct(x, "worked"), (f"{x.fund.mean():+.3f}" if "fund" in x and len(x) else "–")])
    L += ["", "**Signal test (F2): outcome by the next funding rate at entry (all HT setups, backtest + real):** a crowded-longs signal would "
          "show a monotone fall in WORKED / avg as funding rises.", ""]
    L += tbl(rows, ["next rate %", "N · WR · avg", "WORKED", "funding paid in trade (avg %)"])
    hi = g[g.f_rate >= 0.03]; lo = g[g.f_rate < 0.03]
    d = hi.net.mean() - lo.net.mean() if len(hi) and len(lo) else np.nan
    return dict(L=L, ev=ev, ev_ci=ci, n_trig=int(g.trig.sum()), trig_avg=g[g.trig].net.mean() if g.trig.any() else np.nan,
                n_hi=len(hi), d_hi=d, p_hi=shuffle_p(g.net, g.f_rate >= 0.03), fund_kept=kept.fund.mean() if "fund" in kept else np.nan)


def chart(ALL):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    B = ALL[ALL.cohort == "BACKTEST"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    specs = [("NEAR_HIGH (dip < 1.4 %)", B[~B.FS & ~B.BU], "NH"), ("BURST (pre_vol > 1.6 ∧ cloc < 0.5)", B[~B.FS & ~B.NH], "BU"),
             ("High funding (next rate ≥ 0.03 %)", B, None)]
    for a, (title, g, flag) in zip(ax, specs):
        f = (g.f_rate >= 0.03) if flag is None else g[flag].astype(bool)
        for lab, m, col in (("blocked / flagged", f, "#c0392b"), ("kept", ~f, "#2471a3")):
            P = avg_path(g[m])
            if P is not None:
                a.plot(P.index, P.values, color=col, lw=2, label=f"{lab} (N {int(m.sum())})")
            for wl, ls in (("W", ":"), ("L", "--")):
                s = g[m & ((g.net > 0) if wl == "W" else (g.net <= 0))]
                P = avg_path(s)
                if P is not None:
                    a.plot(P.index, P.values, color=col, lw=1, ls=ls, alpha=0.7)
        a.axhline(0, color="grey", lw=0.6); a.set_title(title, fontsize=10); a.set_xlabel("minutes from entry"); a.legend(fontsize=8)
    ax[0].set_ylabel("avg close vs entry, % (solid = all · dotted = winners · dashed = losers)")
    fig.suptitle("WILLY_HT blockers — anchored price path after entry, 2-year backtest (held after the exit, no survivor bias)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT + "_chart.png", dpi=110); plt.close(fig)


def avg_path(g, n=240):
    if len(g) < 3:
        return None
    M = []
    for r in g.to_dict("records"):
        a = C.k1m(r["pair"], int(r["te"]), int(r["te"]) + (n + 1) * MIN)
        if a is None or not len(a["t"]):
            continue
        gg = C.grid(a, int(r["te"]), int(r["te"]) + (n + 1) * MIN)
        E = gg["o"][0] if gg["o"][0] == gg["o"][0] else np.nan
        if not E == E:
            continue
        M.append((gg["c"][:n + 1] / E - 1) * 100)
    if not M:
        return None
    m = np.nanmean(np.vstack([x for x in M if len(x) == n + 1]), axis=0)
    return pd.Series(m, index=np.arange(n + 1))


def write(res, B, RE):
    nh, bu, fu = res["NH"], res["BU"], res["FU"]
    with open(os.path.join(SCR, "body.json"), "w") as fh:
        json.dump({k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv)) for kk, vv in v.items() if kk != "L"}
                   for k, v in res.items()}, fh, default=str, indent=1)
    body = ["## NEAR_HIGH — tables", ""] + nh["L"] + ["", "## BURST — tables", ""] + bu["L"] + ["", "## FUNDING — tables", ""] + fu["L"]
    open(os.path.join(SCR, "tables.md"), "w").write("\n".join(body) + "\n")
    print("\n".join(body))
    print(json.dumps({k: {kk: str(vv) for kk, vv in v.items() if kk != "L"} for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
