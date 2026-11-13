#!/usr/bin/env python3
"""FLATBASE_SHORT — confirmation-depth filter (operator 2026-11-13: "define me the filter … impact in current batch and master … backtest support").
The BANK line (DECISION_LOG 429, registered 11-02) froze the threshold at 1 % — never re-fit here.

Rules (same S4 signal, same +5 / −10 / 72 h exit, entry at the next 5m open after the decisive 4-h close):
  LIVE  today: the first 4-h close below the S4 close within 24 h → short, else skip
  M1    MARGIN (the DL 429 candidate): the first 4-h close ≥ 1 % below the S4 close within 24 h → short, else skip
        (a shallower close below S4 no longer enters; the bot keeps waiting on the same 24 h clock)
  SK1   SKIP-SHALLOW: as LIVE, but when that first close below is < 1 % below S4, the episode is skipped (no later entry)
Blocks: ① master = the engine's own Flatbase ledger episodes (turnover > 1 / unknown), today's stack re-simulated (parity: the study script),
        split B24 / B25 / B26 by spike time; ② backtest = the 2026-10-22 study's FLAT episodes (turnover unknown) — only a cross-check.
Usage: venv/bin/python -I scripts/study_flatbase_depth_filter_1113.py [--no-backtest] [--procs 8]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

_D = os.path.dirname(os.path.abspath(__file__))
_D in sys.path or sys.path.insert(0, _D)
import study_flatbase_early_entry_1112 as S   # noqa: E402  (signal core, exit walk, ledger, kline cache)
import scout_long_stop4 as LS                 # noqa: E402

BAR, H4, HOUR = S.BAR, S.H4, S.HOUR
MARGIN = 1.0
RULES = ["LIVE", "M1", "SK1"]
B26_START = pd.Timestamp("2026-11-11T20:49")
B25_START = pd.Timestamp("2026-10-23T12:00")    # B24 = the first day of the sleeve (10-22 → 10-23 morning)
OUT = os.path.join(S._ROOT, "reports", "FLATBASE_DEPTH_FILTER_2026-11-13")


def decide(t, o, h, l, c, v, spike_ts, fs=None):
    """→ {rule: dict(why, pct, te, tx, depth)} — depth = how far the FIRST 4-h close below S4 sits below it, %."""
    en = S.entries(t, o, h, l, c, v, spike_ts, fs)["LIVE"]
    base = dict(why=en["why"], pct=(np.nan if en["why"] == "PENDING" else 0.0), te=None, tx=None, depth=np.nan)
    out = {r: dict(base) for r in RULES}
    sig = en["sig"]
    if sig is None:
        return out
    n, end = len(t), t[sig] + BAR + S.CONF_MS
    first = m1 = None
    for k in range(sig + 1, n):
        ck = t[k] + BAR
        if ck > end:
            break
        if ck % H4 != 0:
            continue
        if first is None and c[k] < c[sig]:
            first = k
        if c[k] <= c[sig] * (1 - MARGIN / 100):
            m1 = k
            break
    done = t[n - 1] + BAR >= end
    depth = (c[sig] - c[first]) / c[sig] * 100 if first is not None else np.nan

    def run(k):
        if k is None:
            return dict(why=("SKIPPED" if done else "PENDING"), pct=(0.0 if done else np.nan), te=None, tx=None, depth=depth)
        i = k + 1
        if i >= n:
            return dict(why="PENDING", pct=np.nan, te=None, tx=None, depth=depth)
        why, pct, j = S.walk_exit(t, o, h, l, c, i)
        return dict(why=why, pct=(np.nan if why == "PENDING" else pct), te=int(t[i]), tx=(int(t[j]) + BAR if j is not None else None), depth=depth)

    out["LIVE"] = run(first)
    out["M1"] = run(m1)
    if first is not None and depth < MARGIN:
        out["SK1"] = dict(why="SKIPPED_SHALLOW", pct=0.0, te=None, tx=None, depth=depth)
    else:
        out["SK1"] = dict(out["LIVE"])
    if first is None and not done:   # still waiting for any close below
        for r in RULES:
            out[r] = dict(why="PENDING", pct=np.nan, te=None, tx=None, depth=depth)
    return out


def master():
    L = S.ledger()
    L = L[L.gate != "SHADOW"]
    bud, now = LS.Budget(), int(time.time() * 1000)
    rows = []
    for _, e in L.iterrows():
        Sms = S.ms(e.fb_spike_at)
        a, kl = Sms - 25 * HOUR, []
        for k in range(3):
            st = a + k * 1500 * BAR
            if st > now:
                break
            r = LS.fapi_klines(e.pair, "5m", st, 1500, bud, S.CACHE, now=now)
            if r is None:
                raise SystemExit("fetch stopped: " + bud.why)
            kl += r
            if len(r) < 1500:
                break
        if len(kl) < 400:
            continue
        t, o, h, l, c, v = S.grid_from_rows(kl)
        fsm = S.ms(e.fb_first_seen_at) if isinstance(e.fb_first_seen_at, str) else None
        d = decide(t, o, h, l, c, v, Sms, fsm)
        sp = pd.Timestamp(e.fb_spike_at)
        rec = dict(pair=e.pair, spike=e.fb_spike_at, status=e.gate, batch=("B26" if sp >= B26_START else "B25" if sp >= B25_START else "B24"))
        for r, x in d.items():
            for kk, vv in x.items():
                rec[f"{r}_{kk}"] = vv
        rows.append(rec)
    return pd.DataFrame(rows)


def _bt_pair(args):
    pair, eps = args
    import study_flatbase_short_common as C
    full = C.load_full(pair)
    if full is None:
        return []
    t, o, h, l, c, v, qv, real = full
    res = []
    for _, e in eps.iterrows():
        Sms = int(e.spike_ts)
        on = int((Sms - BAR - t[0]) // BAR)
        if on < 300 or on >= len(t):
            continue
        lo, hi = on - 300, min(len(t), on + S.MAX_BARS + 24 * 12 + S.HOLD_BARS + 12)
        d = decide(t[lo:hi], o[lo:hi], h[lo:hi], l[lo:hi], c[lo:hi], v[lo:hi], Sms, e.first_seen_t)
        rec = dict(pair=pair, spike=str(pd.Timestamp(Sms, unit="ms")))
        for r, x in d.items():
            for kk, vv in x.items():
                rec[f"{r}_{kk}"] = vv
        res.append(rec)
    return res


def backtest(procs):
    import multiprocessing as mp
    E = pd.read_csv(os.path.join(S._ROOT, "reports", "FRENZY_FLATBASE_SHORT_2026-10-22_episodes.csv"))
    E = E[E.label == "flat"]
    res = []
    with mp.Pool(procs) as pool:
        for r in pool.imap_unordered(_bt_pair, [(p, g) for p, g in E.groupby("pair")], chunksize=2):
            res += r
    return pd.DataFrame(res)


def dcr(df, rule):
    x = df[df[f"{rule}_why"].isin(["TP", "STOP", "TIME"])]
    days = pd.date_range(pd.to_datetime(df.spike).min().floor("D"), pd.to_datetime(df.spike).max().floor("D") + pd.Timedelta(days=3), freq="D")
    if not len(x):
        return 0.0
    r = (x[f"{rule}_pct"] * S.X_SIZE / 100).groupby(pd.to_datetime(x[f"{rule}_tx"], unit="ms").dt.floor("D")).sum().reindex(days, fill_value=0.0)
    return (float(np.prod(1 + r.values)) ** (1 / len(days)) - 1) * 100


def read(df, label, boot=True):
    ok = df[~df[[f"{r}_pct" for r in RULES]].isna().any(axis=1)].copy()
    L = [f"\n### {label} — {len(df)} episodes · {len(ok)} decided ({len(df) - len(ok)} still running, out)",
         "| rule | trades | WR | avg %/trade | Σ pts | Δ Σ vs LIVE | LOO Δ Σ | P(Δ > 0) day-boot | DCR one size %/day |", "|---|---|---|---|---|---|---|---|---|"]
    if not len(ok):
        return L, ok
    days = pd.to_datetime(ok.spike).dt.floor("D").values
    for r in RULES:
        tr = ok[f"{r}_why"].isin(["TP", "STOP", "TIME"])
        p = ok[f"{r}_pct"]
        d = (p - ok["LIVE_pct"]).values
        if r == "LIVE":
            ds = loo = pp = "—"
        else:
            ds = f"{d.sum():+.1f}"
            loo = f"{d.sum() - d[np.argmax(np.abs(d))]:+.1f}" if len(d) else "—"
            pp = f"{S.boot_lb(d, days, 0.05)[1]:.2f}" if boot and len(set(days)) > 1 else "—"
        L.append(f"| {r} | {int(tr.sum())} | {(p[tr] > 0).mean() * 100:.0f}% | {p[tr].mean():+.2f} | {p.sum():+.1f} | {ds} | {loo} | {pp} | {dcr(ok, r):+.2f} |")
    return L, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--procs", type=int, default=8)
    A = ap.parse_args()
    M = master()
    M.to_csv(OUT + "_master_rows.csv", index=False)
    lines = [f"# FLATBASE confirmation-depth filter (margin {MARGIN:g} %, frozen DL 429) · exit +5 / −10 / 72 h · one size {S.X_SIZE} × account"]
    for lab, g in [("① MASTER all batches (today's stack, re-simulated)", M), ("① B24", M[M.batch == "B24"]), ("① B25", M[M.batch == "B25"]),
                   ("① B26 — current batch", M[M.batch == "B26"])]:
        lines += read(g, lab)[0]
    aff = M[(M.LIVE_depth < MARGIN) | (M.M1_why != M.LIVE_why)]
    lines.append("\n**Episodes the filter changes (master):**")
    for _, e in aff.iterrows():
        lines.append(f"- {e.pair} {e.spike[:16]} ({e.batch}) depth {e.LIVE_depth:.2f} % · LIVE {e.LIVE_why} {e.LIVE_pct:+.2f} → M1 {e.M1_why} {e.M1_pct:+.2f} · SK1 {e.SK1_why} {e.SK1_pct:+.2f}")
    if not A.no_backtest:
        B = backtest(A.procs)
        B.to_pickle(OUT + "_bt_rows.pkl")
        lines += read(B, "② BACKTEST all (study FLAT episodes 2024-11 → 2026-10-22, turnover unknown)")[0]
        half = pd.to_datetime(B.spike).apply(lambda s: f"{max(s.year, 2025)}H{1 if (s.month <= 6 or s.year < 2025) else 2}")
        for hf, g in B.groupby(half):
            lines += read(g, f"② backtest {hf}", boot=False)[0]
        sh = B[B.LIVE_depth < MARGIN]
        sh = sh[~sh[[f"{r}_pct" for r in RULES]].isna().any(axis=1)]
        trs = sh.LIVE_why.isin(["TP", "STOP", "TIME"])
        lines.append(f"\n② backtest — the SHALLOW cohort under today's rule (depth < {MARGIN:g} %): N {int(trs.sum())} · WR {(sh.LIVE_pct[trs] > 0).mean() * 100:.0f}% · avg {sh.LIVE_pct[trs].mean():+.2f} · "
                     f"deep: N {int(B.LIVE_why.isin(['TP', 'STOP', 'TIME']).sum() - trs.sum())}")
        dd = B[B.LIVE_why.isin(["TP", "STOP", "TIME"])]
        for lo, hi in [(0, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 100)]:
            g = dd[(dd.LIVE_depth >= lo) & (dd.LIVE_depth < hi)]
            if len(g):
                lines.append(f"  depth {lo}–{hi} %: N {len(g)} · WR {(g.LIVE_pct > 0).mean() * 100:.0f}% · avg {g.LIVE_pct.mean():+.2f}")
    open(OUT + ".md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
