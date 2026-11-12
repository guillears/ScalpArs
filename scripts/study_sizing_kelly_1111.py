#!/usr/bin/env python3
"""SLEEVE SIZING STUDY — edge-based (cautious fractional Kelly) vs today's flat 5 %-per-stop C5 sizes (operator 2026-11-11: "re think that static
rule, it should be more related with number of trades and WR and avg win").

RULE (pre-declared): exposure X per sleeve (account-% per 1 % move of the position) = f × Kelly(cautious) where
  Kelly(cautious) = m_lo / E[r²] · r = a fill's P&L as a fraction · m_lo = the 10th percentile of a 4,000-resample bootstrap of the mean
  (fewer trades → wider → smaller) · a sleeve with NO loss on record gets one full stop-out added before the estimate · m_lo ≤ 0 → X = 0
  (paused) · N < N_MIN → the TRIAL size (1 % of the account per worst stop) · X capped so one worst stop ≤ CAP % of the account.
WALK-FORWARD (no look-ahead): every period's sizes come ONLY from the fills that closed before the period starts (expanding window).
  master = per batch (era), in batch order, kept non-probe fills, stack_pct (today's stack) · backtest = per calendar month on the 2-year
  today's-stack replay (ENGINE_REPLAY_YR6W_2Y_TODAY: ML_BULL90 applied, ML_MKT_HOT NOT — caveat; sleeves ML / LITE / WILLY only).
COMPARED: TODAY (C5 fixed) · K25_cap8 · K25_cap5 · K50_cap8 · K25_cap8 without the trial floor (N_MIN 0).
METRIC: account daily compound return over EVERY calendar day of each period (r_d = Σ fills closed that day of pct × X), worst day, max
  drawdown of the compounded path, per-period wins. Writes reports/SIZING_KELLY_2026-11-11.txt (+ _periods.csv)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "SIZING_KELLY_2026-11-11")
# sleeve: (entry_strategy, direction, worst stop %, today's X)  — today's X = C5 (DL 528): ≈ 4.9 % of the account per stop; ML 1 × 10× ≈ 2.43
SLEEVES = {"ML": ("MOMENTUM", "LONG", 1.2, 2.43), "LITE": ("FRENZY_LITE", "LONG", 3.0, 4.9 / 3), "RETRY": ("FRENZY_LITE_RETRY", "LONG", 3.0, 4.9 / 3),
           "WILLY": ("FRENZY_WILLY", "LONG", 4.0, 4.9 / 4), "HT": ("FRENZY_WILLY_HT", "LONG", 4.0, 4.9 / 4), "FLAT": ("FLATBASE_SHORT", "SHORT", 10.0, 4.9 / 10)}
RULES = {"TODAY": None, "K25_cap8": (0.25, 8.0, 15), "K25_cap5": (0.25, 5.0, 15), "K50_cap8": (0.50, 8.0, 15), "K25_cap8_noTrial": (0.25, 8.0, 0)}
TRIAL_STOP = 1.0
BOOT, SEED = 4000, 20261111


def kelly_x(r_pct, stop, frac, cap, n_min):
    r = np.asarray(r_pct, float) / 100.0
    if len(r) < max(n_min, 1):
        return TRIAL_STOP / stop
    if not (r < 0).any():
        r = np.r_[r, -stop / 100.0]
    rng = np.random.default_rng(SEED)
    lo = np.quantile(rng.choice(r, (BOOT, len(r))).mean(axis=1), 0.10)
    if lo <= 0:
        return 0.0
    return min(frac * lo / (r ** 2).mean(), cap / stop)


def path_stats(daily):
    r = np.asarray(daily, float)
    eq = np.cumprod(1 + r)
    dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return ((eq[-1]) ** (1 / len(r)) - 1) * 100, r.min() * 100, dd * 100


def simulate(F, periods, sleeves):
    """F: fills with sleeve, pct, o, c (UTC naive). periods: list of (name, start, end). → {rule: (daily list, per-period rows)}"""
    out = {}
    for rule, prm in RULES.items():
        daily, rows = [], []
        for name, t0, t1 in periods:
            hist = F[F.c < t0]
            cur = F[(F.o >= t0) & (F.o <= t1)]
            days = pd.date_range(t0.normalize(), t1.normalize(), freq="D")
            X = {}
            for s in sleeves:
                stop, today = SLEEVES[s][2], SLEEVES[s][3]
                X[s] = today if prm is None else kelly_x(hist[hist.sleeve == s].pct.values, stop, *prm)
            rd = pd.Series(0.0, index=days)
            for s in sleeves:
                g = cur[cur.sleeve == s]
                if len(g):
                    rd = rd.add((g.pct * X[s] / 100).groupby(g.c.dt.normalize()).sum().reindex(days).fillna(0.0), fill_value=0.0)
            daily += list(rd.values)
            dcr, worst, dd = path_stats(rd.values)
            rows.append(dict(rule=rule, period=name, days=len(days), fills=len(cur), dcr=dcr, worst_day=worst, maxdd=dd,
                             **{f"stop%_{s}": round(X[s] * SLEEVES[s][2], 2) for s in sleeves}))
        out[rule] = (daily, rows)
    return out


def report(title, res, L):
    L.append(f"\n## {title}")
    P = pd.DataFrame([r for _, (d, rows) in res.items() for r in rows])
    base = P[P.rule == "TODAY"].set_index("period").dcr
    L.append("| Rule | %/day (all days) | Worst day | Max drawdown | Periods better than TODAY |")
    L.append("|---|---|---|---|---|")
    for rule, (daily, rows) in res.items():
        dcr, worst, dd = path_stats(daily)
        pr = P[P.rule == rule].set_index("period").dcr
        better = int((pr > base + 1e-9).sum()) if rule != "TODAY" else "–"
        L.append(f"| {rule} | {dcr:+.3f} | {worst:+.2f} % | {dd:+.2f} % | {better if rule == 'TODAY' else f'{better} of {len(pr)}'} |")
    return P


def master():
    d = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    d["o"] = pd.to_datetime(d.opened_at, format="mixed", utc=True).dt.tz_convert(None)
    d["c"] = pd.to_datetime(d.closed_at, format="mixed", utc=True).dt.tz_convert(None)
    bounds = d.groupby("era").agg(t0=("o", "min"), t1=("c", "max")).sort_values("t0")
    k = d[(d.stack_keep == True) & ~d.cell_multiplier_source.astype(str).str.endswith("_PROBE")].copy()  # noqa: E712
    k["pct"] = pd.to_numeric(k.stack_pct, errors="coerce")
    k["sleeve"] = None
    for s, (es, di, _, _) in SLEEVES.items():
        k.loc[(k.entry_strategy == es) & (k.direction == di), "sleeve"] = s
    k = k[k.sleeve.notna() & k.pct.notna()]
    periods = [(e, r.t0, r.t1) for e, r in bounds.iterrows()]
    return k, periods


def backtest():
    d = pd.read_csv(os.path.join(ROOT, "reports", "ENGINE_REPLAY_YR6W_2Y_TODAY_fills.csv"), low_memory=False)
    d["o"] = pd.to_datetime(d.opened_at, format="mixed").dt.tz_localize(None) if True else None
    d["c"] = pd.to_datetime(d.closed_at, format="mixed").dt.tz_localize(None)
    d["pct"] = pd.to_numeric(d.pnl_percentage, errors="coerce")
    d["sleeve"] = None
    for s in ("ML", "LITE", "WILLY"):
        es, di = SLEEVES[s][:2]
        d.loc[(d.entry_strategy == es) & (d.direction == di), "sleeve"] = s
    d = d[d.sleeve.notna() & d.pct.notna()]
    ms = pd.date_range(d.o.min().normalize().replace(day=1), d.o.max(), freq="MS")
    periods = [(m.strftime("%Y-%m"), m, min(m + pd.offsets.MonthEnd(0), d.c.max())) for m in ms]
    return d, periods


def main():
    L = ["# SLEEVE SIZING — cautious fractional Kelly vs today's flat 5 %-per-stop (walk-forward, 2026-11-11)",
         "Sizes for each period use ONLY fills closed before it. TODAY = C5 (≈ 4.9 % per stop; Momentum Long 2.43 × the account per 1 %). "
         f"Trial size = {TRIAL_STOP} % per worst stop while a sleeve has < 15 fills; a sleeve whose cautious edge is ≤ 0 is paused (0)."]
    k, per = master()
    resM = simulate(k, per, list(SLEEVES))
    PM = report(f"MASTER — {len(k)} kept fills of the six C5 sleeves, {len(per)} batches (walk-forward by batch)", resM, L)
    rec = [p for p in per if p[0] in ("B19", "B20", "B21", "B22", "B23", "B24", "B25")]
    resR = simulate(k, rec, list(SLEEVES))
    report("MASTER, Frenzy era only (B19 → B25, the batches where every sleeve trades)", resR, L)
    L.append("\nPer batch (Frenzy era), %/day and loss per worst stop (% of account) the rule chose:")
    PR = pd.DataFrame([r for _, (d, rows) in resR.items() for r in rows])
    L.append(PR[PR.rule.isin(["TODAY", "K25_cap8"])].round(2).to_string(index=False))
    b, bper = backtest()
    resB = simulate(b, bper, ["ML", "LITE", "WILLY"])
    PB = report(f"BACKTEST — {len(b)} replay fills (ML / LITE / WILLY), {len(bper)} months, sizes re-estimated monthly", resB, L)
    yr = {}
    for rule, (daily, rows) in resB.items():
        P = pd.DataFrame(rows)
        for y in ("2025", "2026"):
            q = P[P.period.str.startswith(y)]
            yr[(rule, y)] = q
    L.append("\nBacktest by year (mean of monthly %/day):")
    for rule in RULES:
        L.append(f"  {rule}: 2025 {yr[(rule, '2025')].dcr.mean():+.3f} · 2026 {yr[(rule, '2026')].dcr.mean():+.3f}")
    L.append("\nBacktest final month sizes (loss per worst stop, % of account) under K25_cap8:")
    L.append(PB[PB.rule == "K25_cap8"].tail(1).round(2).to_string(index=False))
    pd.concat([PM.assign(scope="master"), PR.assign(scope="master_frenzy_era"), PB.assign(scope="backtest")]).to_csv(OUT + "_periods.csv", index=False)
    open(OUT + ".txt", "w").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
