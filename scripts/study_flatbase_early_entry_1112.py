#!/usr/bin/env python3
"""FLATBASE_SHORT — earlier-entry study (operator 2026-11-12: "seems like we are always entering LATE").
Pre-registration: reports/FLATBASE_EARLY_ENTRY_PREREG_2026-11-12.md (rules, exit, read and candidate bar frozen there BEFORE this ran).
Research only, no bot API, public 5m klines only (cached, budgeted, shared cooldown).

Blocks: ① live = the engine's own Flatbase ledger (decisions exports) · ② backtest = the 2026-10-22 study's FLAT episodes.
Usage: venv/bin/python -I scripts/study_flatbase_early_entry_1112.py [--no-backtest] [--procs 8]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
_ROOT = os.path.dirname(_D)
_ROOT in sys.path or sys.path.insert(0, _ROOT)
import scout_long_stop4 as LS      # noqa: E402  (budgeted, cached public fapi klines)

BAR, HOUR, H4 = 300_000, 3_600_000, 14_400_000
MIN_VOL = 20e6
MAX_BARS = 96 * 12
HOLD_BARS = 72 * 12
CONF_MS = 24 * HOUR
STOP_G, TP_G, TP_NET, FEE = 10.0, 5.0, 4.94, 0.09
X_SIZE = 0.38                      # one size: notional ≈ 0.38 × account (FLATBASE 0.2 × slot margin × 8× at $10k)
DL428 = pd.Timestamp("2026-11-02T17:00:00")   # FB_CONFIRM ship (≈ commit + 10 min; parity split only)
RULES = ["LIVE", "S4", "H1_4", "H1C_4", "H1C_2", "RED5", "OFF10C", "ROLL4", "ROLL4C", "PEAK4", "PEAK4C"]   # ROLL / PEAK: addendum (operator mid-run)
CHALLENGERS = RULES[1:]
SPEC = dict(LIVE=(H4, 48, H4), S4=(H4, 48, None), H1_4=(HOUR, 48, None), H1C_4=(HOUR, 48, HOUR), H1C_2=(HOUR, 24, HOUR),
            RED5=(BAR, 6, None), OFF10C=(HOUR, 24, HOUR),
            ROLL4=(BAR, 48, None), ROLL4C=(BAR, 48, "R"), PEAK4=(BAR, 0, None), PEAK4C=(BAR, 0, "R"))   # "R" = rolling confirm: closes 4, 8 … 24 h after the signal   # (signal grid, min bars after the spike bar, confirm grid)
BOOT_N, SEED = 4000, 7
OUT = os.path.join(_ROOT, "reports", "FLATBASE_EARLY_ENTRY_2026-11-12")
CACHE = os.path.join(_ROOT, "reports", "backtest_cache", "study_fb_early")


# ───────────────────────────── pure rule core ─────────────────────────────
def entries(t, o, h, l, c, v, spike_ts, first_seen_ms=None):
    """→ {rule: dict(i=entry bar index | None, why, sig=signal index)} on a contiguous 5m grid (missing bars filled flat, zero volume)."""
    n = len(t)
    on = int((int(spike_ts) - BAR - t[0]) // BAR)
    out = {r: dict(i=None, why="NO_SIGNAL", sig=None) for r in RULES}
    if on < 0 or on >= n:
        return {r: dict(i=None, why="NO_DATA", sig=None) for r in RULES}
    tp_ = (h + l + c) / 3.0
    q = v * tp_
    cq = np.r_[0.0, np.cumsum(q)]
    hi = min(n - 1, on + MAX_BARS)
    pv = qq = 0.0
    peak, pk = -1.0, on
    vw = np.full(n, np.nan); pkv = np.full(n, np.nan); pki = np.zeros(n, int)
    for w in range(on, hi + 1):
        pv += tp_[w] * q[w]; qq += q[w]
        vw[w] = pv / qq if qq > 0 else c[w]
        if h[w] > peak:
            peak, pk = h[w], w
        pkv[w], pki[w] = peak, pk
    fs = int(first_seen_ms) if first_seen_ms is not None and not pd.isna(first_seen_ms) else None
    for r in RULES:
        G, mb, CG = SPEC[r]
        sig = None
        for w in range(on + mb, hi + 1):
            if (t[w] + BAR) % G != 0 or not c[w] < vw[w]:
                continue
            if fs is not None and t[w] < fs:
                continue
            if w < 287:
                continue
            if cq[w + 1] - cq[w - 287] < MIN_VOL:
                continue
            if r == "RED5" and not (w > pki[w] and c[w] <= o[w] * 0.95):
                continue
            if r == "OFF10C" and not c[w] <= pkv[w] * 0.90:
                continue
            if r in ("PEAK4", "PEAK4C") and w - pki[w] < 48:
                continue
            sig = w
            break
        if sig is None:
            out[r] = dict(i=None, why=("NO_SIGNAL" if hi >= on + MAX_BARS else "PENDING"), sig=None)
            continue
        dec = sig
        if CG is not None:
            end = t[sig] + BAR + CONF_MS
            dec = None
            for k in range(sig + 1, n):
                ck = t[k] + BAR
                if ck > end:
                    break
                if (((k - sig) % 48 == 0) if CG == "R" else (ck % CG == 0)) and c[k] < c[sig]:
                    dec = k
                    break
            if dec is None:
                out[r] = dict(i=None, why=("SKIPPED" if t[n - 1] + BAR >= end else "PENDING"), sig=sig)
                continue
        out[r] = dict(i=(dec + 1 if dec + 1 < n else None), why=("ENTER" if dec + 1 < n else "PENDING"), sig=sig)
    return out


def walk_exit(t, o, h, l, c, i):
    """today's exit from entry bar i (fill at its open) → (why, net %, exit bar index)."""
    E = o[i]
    if not E > 0:
        return "BAD", np.nan, None
    stop, tp = E * (1 + STOP_G / 100), E * (1 - TP_G / 100)
    last = i + HOLD_BARS - 1
    for j in range(i, min(len(t), last + 1)):
        if h[j] >= stop:
            px = max(o[j], stop) if j > i else stop
            return "STOP", -(px / E - 1) * 100 - FEE, j
        if l[j] <= tp:
            return "TP", TP_NET, j
    if last < len(t):
        return "TIME", -(c[last] / E - 1) * 100 - FEE, last
    return "PENDING", np.nan, None


def score_episode(t, o, h, l, c, v, spike_ts, first_seen_ms=None):
    en = entries(t, o, h, l, c, v, spike_ts, first_seen_ms)
    on = int((int(spike_ts) - BAR - t[0]) // BAR)
    rows = {}
    for r, e in en.items():
        d = dict(why=e["why"], pct=0.0, te=None, tx=None, E=None, h_spike=None, h_peak=None, off_peak=None, sig=e["sig"])
        if e["i"] is not None:
            i = e["i"]
            why, pct, j = walk_exit(t, o, h, l, c, i)
            pk = int(np.argmax(h[on:i])) + on if i > on else on
            d.update(why=why, pct=(pct if why != "PENDING" else np.nan), te=int(t[i]), tx=(int(t[j]) + BAR if j is not None else None), E=float(o[i]),
                     h_spike=(t[i] - int(spike_ts)) / HOUR, h_peak=(t[i] - t[pk] - BAR) / HOUR, off_peak=(o[i] / h[pk] - 1) * 100)
        elif e["why"] == "PENDING":
            d["pct"] = np.nan
        rows[r] = d
    return rows


def grid_from_rows(rows):
    """[[open, o, h, l, c, v], …] → contiguous arrays (gaps filled flat, zero volume)."""
    d = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]).drop_duplicates("t").sort_values("t")
    g = np.arange(int(d.t.iloc[0]), int(d.t.iloc[-1]) + 1, BAR, dtype="int64")
    d = d.set_index("t").reindex(g)
    c = d.c.ffill().values.astype(float)
    f = lambda s: s.fillna(pd.Series(c, index=d.index)).values.astype(float)   # noqa: E731
    return g, f(d.o), f(d.h), f(d.l), c, d.v.fillna(0).values.astype(float)


# ───────────────────────────── ① live block ─────────────────────────────
def ledger():
    fs = glob.glob(os.path.expanduser("~/Downloads/scalpars_decisions_*.csv")) + glob.glob(os.path.join(_ROOT, "exports_archive", "**", "scalpars_decisions_*.csv"), recursive=True)
    fr = []
    for f in fs:
        try:
            d = pd.read_csv(f, low_memory=False, usecols=lambda x: x in ("t", "e", "pair", "gate", "price", "fb_spike_at", "fb_first_seen_at", "fb_turnover", "fb_reason"))
        except Exception:
            continue
        d = d[d.e.astype(str) == "FLATBASE"]
        if len(d):
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d["p"] = (d.gate != "TRACKING").astype(int)
    d = d.sort_values(["p", "t"]).drop_duplicates(["pair", "fb_spike_at"], keep="last")
    return d[~d.gate.isin(["NOT_FLAT", "UNREADABLE"])].reset_index(drop=True)


def ms(s):
    return int(pd.Timestamp(s).value // 10**6)


def live_block():
    L = ledger()
    bud = LS.Budget()
    now = int(time.time() * 1000)
    out = []
    for _, e in L.iterrows():
        S = ms(e.fb_spike_at)
        a = S - 25 * HOUR
        rows = []
        for k in range(3):   # 3 × 1,500 bars = 375 h ≥ 25 h before + 96 + 24 + 72 h after
            st = a + k * 1500 * BAR
            if st > now:
                break
            r = LS.fapi_klines(e.pair, "5m", st, 1500, bud, CACHE, now=now)
            if r is None:
                print("fetch stopped:", bud.why)
                break
            rows += r
            if len(r) < 1500:
                break
        if bud.stop:
            break
        if len(rows) < 400:
            out.append(dict(pair=e.pair, spike=e.fb_spike_at, status=e.gate, turnover=e.fb_turnover, ledger_px=e.price, nodata=True))
            continue
        t, o, h, l, c, v = grid_from_rows(rows)
        fsm = ms(e.fb_first_seen_at) if isinstance(e.fb_first_seen_at, str) else None
        res = score_episode(t, o, h, l, c, v, S, fsm)
        rec = dict(pair=e.pair, spike=e.fb_spike_at, status=e.gate, turnover=e.fb_turnover, ledger_px=e.price, nodata=False)
        for r, d in res.items():
            for kk, vv in d.items():
                rec[f"{r}_{kk}"] = vv
        # parity: the rule the engine ran → its decisive close (ledger OPENED price = S4 close before DL 428, confirm close after)
        ran = "LIVE" if pd.Timestamp(e.fb_spike_at) >= DL428 - pd.Timedelta(hours=96) and pd.Timestamp(e.t) >= DL428 else "S4"
        rec["ran"] = ran
        out.append(rec)
    print(f"live: {len(out)} episodes · {bud.requests} requests · weight {bud.used}")
    return pd.DataFrame(out)


# ───────────────────────────── ② backtest block ─────────────────────────────
def _bt_pair(args):
    pair, eps = args
    import study_flatbase_short_common as C
    full = C.load_full(pair)
    if full is None:
        return []
    t, o, h, l, c, v, qv, real = full
    F = C.funding_table(pair)
    res = []
    for _, e in eps.iterrows():
        S = int(e.spike_ts)
        on = int((S - BAR - t[0]) // BAR)
        if on < 300 or on >= len(t):
            continue
        lo_, hi_ = on - 300, min(len(t), on + MAX_BARS + 24 * 12 + HOLD_BARS + 12)
        sc = score_episode(t[lo_:hi_], o[lo_:hi_], h[lo_:hi_], l[lo_:hi_], c[lo_:hi_], v[lo_:hi_], S, e.first_seen_t)
        rec = dict(pair=pair, spike=pd.Timestamp(S, unit="ms"))
        for r, d in sc.items():
            for kk, vv in d.items():
                rec[f"{r}_{kk}"] = vv
            fund = np.nan
            if d["te"] is not None and d["tx"] is not None:
                fund = float(C.funding_sum(F, np.array([d["te"]]), np.array([d["tx"]]))[0][0])
            rec[f"{r}_fund"] = fund
        res.append(rec)
    return res


def backtest_block(procs):
    import multiprocessing as mp
    E = pd.read_csv(os.path.join(_ROOT, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_episodes.csv"))
    E = E[E.label == "flat"]
    jobs = [(p, g) for p, g in E.groupby("pair")]
    res = []
    with mp.Pool(procs) as pool:
        for k, r in enumerate(pool.imap_unordered(_bt_pair, jobs, chunksize=2)):
            res += r
    print(f"backtest: {len(res)} flat episodes")
    return pd.DataFrame(res)


# ───────────────────────────── read ─────────────────────────────
def boot_lb(delta, days, alpha):
    df = pd.DataFrame(dict(d=delta, day=days))
    g = df.groupby("day").d.agg(["sum", "count"])
    s, n = g["sum"].values, g["count"].values
    rng = np.random.default_rng(SEED)
    k = len(s)
    if k < 2:
        return np.nan, np.nan
    idx = rng.integers(0, k, (BOOT_N, k))
    m = s[idx].sum(1) / n[idx].sum(1)
    return float(np.quantile(m, alpha)), float((m > 0).mean())


def dcr(df, rule, start, end):
    x = df[df[f"{rule}_why"].isin(["TP", "STOP", "TIME"])]
    if not len(x):
        return 0.0
    day = pd.to_datetime(x[f"{rule}_tx"], unit="ms").dt.floor("D")
    r = (x[f"{rule}_pct"] * X_SIZE / 100).groupby(day).sum()
    days = pd.date_range(start, end, freq="D")
    rr = r.reindex(days, fill_value=0.0)
    return (float(np.prod(1 + rr.values)) ** (1 / len(days)) - 1) * 100


def table(df, label, fund=False):
    lines = [f"\n### {label}"]
    ok = df.copy()
    pend = np.zeros(len(ok), bool)
    for r in RULES:
        pend |= ok[f"{r}_pct"].isna().values
    ok = ok[~pend]
    lines.append(f"episodes {len(df)} · scored {len(ok)} (pending / still running {int(pend.sum())} out)")
    if not len(ok):
        return lines, ok
    days = pd.to_datetime(ok.spike).dt.floor("D")
    start, end = days.min(), days.max() + pd.Timedelta(days=7)
    lines.append("| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | Δ /episode [LB 1−.05/10] · P(Δ>0) | LOO Δ Σ | DCR one size %/day | hrs after spike | hrs after peak | % below peak |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in RULES:
        p = ok[f"{r}_pct"] + (ok[f"{r}_fund"].fillna(0) if fund and f"{r}_fund" in ok else 0)
        tr = ok[f"{r}_why"].isin(["TP", "STOP", "TIME"])
        n = int(tr.sum())
        wr = (p[tr] > 0).mean() * 100 if n else np.nan
        av = p[tr].mean() if n else np.nan
        S_ = p.sum()
        pl = ok["LIVE_pct"] + (ok["LIVE_fund"].fillna(0) if fund and "LIVE_fund" in ok else 0)
        dl = (p - pl).values
        if r == "LIVE":
            dtxt, loo, dsum = "—", "—", "—"
        else:
            lb, pp = boot_lb(dl, days.values, 0.05 / 10)
            dtxt = f"{dl.mean():+.2f} [{lb:+.2f}] · {pp:.2f}"
            loo = f"{dl.sum() - dl[np.argmax(np.abs(dl))]:+.1f}"
            dsum = f"{dl.sum():+.1f}"
        dd = ok.copy()
        dd[f"{r}_pct"] = p
        lines.append(f"| {r} | {n} | {wr:.0f}% | {av:+.2f} | {S_:+.1f} | {dsum} | {dtxt} | {loo} | {dcr(dd, r, start, end):+.2f} | "
                     f"{ok.loc[tr, f'{r}_h_spike'].median():.1f} | {ok.loc[tr, f'{r}_h_peak'].median():.1f} | {ok.loc[tr, f'{r}_off_peak'].median():+.1f} |")
    return lines, ok


def verdicts(bt, live_ok):
    out = ["\n## Pre-registered candidate bar"]
    bt = bt.copy()
    bt["half"] = pd.to_datetime(bt.spike).apply(lambda s: f"{max(s.year, 2025)}H{1 if (s.month <= 6 or s.year < 2025) else 2}")
    for r in CHALLENGERS:
        p = bt[f"{r}_pct"] - bt["LIVE_pct"]
        lb, pp = boot_lb(p.values, pd.to_datetime(bt.spike).dt.floor("D").values, 0.05 / 10)
        halves = bt.assign(d=p).groupby("half").d.mean()
        pos = int((halves > 0).sum())
        ld = (live_ok[f"{r}_pct"] - live_ok["LIVE_pct"]).sum() if len(live_ok) else np.nan
        ok = p.mean() > 0 and lb > 0 and pos >= 3 and ld >= 0
        out.append(f"- **{r}**: backtest Δ {p.mean():+.2f}/episode, LB {lb:+.2f}, halves + {pos}/4 ({', '.join(f'{k} {v:+.2f}' for k, v in halves.items())}) · live Δ Σ {ld:+.1f} → "
                   + ("**CANDIDATE (observe)**" if ok else "REJECTED"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    A = ap.parse_args()
    lines = ["# FLATBASE_SHORT — earlier-entry study (2026-11-12) · rules frozen in reports/FLATBASE_EARLY_ENTRY_PREREG_2026-11-12.md",
             f"Exit for every rule: today's +5 / −10 / 72 h. One size = notional {X_SIZE} × account. No funding in ① (paper)."]
    lv = live_block()
    lv.to_csv(OUT + "_live_rows.csv", index=False)
    trade = lv[(lv.status != "SHADOW") & ~lv.nodata]
    shadow = lv[(lv.status == "SHADOW") & ~lv.nodata]
    # parity: the engine's decisive close (ledger price on OPENED rows) vs the simulated rule the engine ran
    par = []
    for _, e in trade[trade.status == "OPENED"].iterrows():
        sig = e[f"{e.ran}_E"]
        par.append(dict(pair=e.pair, spike=e.spike, ran=e.ran, ledger_px=e.ledger_px, sim_entry_open=sig, why=e[f"{e.ran}_why"]))
    P = pd.DataFrame(par)
    if len(P):
        P["gap_pct"] = (P.sim_entry_open / P.ledger_px - 1) * 100
        lines.append(f"\n**Parity** (OPENED episodes, sim entry open vs the ledger's decisive close): {len(P)} · |gap| ≤ 0.5 % on {(P.gap_pct.abs() <= 0.5).sum()} · "
                     f"no sim entry on {P.sim_entry_open.isna().sum()}")
        P.to_csv(OUT + "_parity.csv", index=False)
    l1, live_ok = table(trade, "① LIVE ledger episodes (turnover > 1 / unknown) — master, 10-20 → 11-12")
    lines += l1
    l2, _ = table(shadow, "side line — SHADOW episodes (turnover ≤ 1, never traded)")
    lines += l2
    if not A.no_backtest:
        bt = backtest_block(A.procs)
        bt.to_pickle(OUT + "_bt_rows.pkl")
        l3, bt_ok = table(bt, "② BACKTEST — study FLAT episodes 2024-11 → 2026-10-22 (no funding)")
        lines += l3
        l4, _ = table(bt, "② BACKTEST — with funding (short receives / pays)", fund=True)
        lines += l4
        bt_ok = bt_ok.copy()
        for half, g in bt_ok.groupby(pd.to_datetime(bt_ok.spike).apply(lambda s: f"{max(s.year, 2025)}H{1 if (s.month <= 6 or s.year < 2025) else 2}")):
            lh, _ = table(g, f"② backtest {half}")
            lines += lh
        lines += verdicts(bt_ok, live_ok)
    open(OUT + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
