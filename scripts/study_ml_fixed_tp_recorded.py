"""🎯 Momentum-long FIXED take-profits (+0.5 / 1 / 2 / 3 %) read from the RECORDED trade data — operator 2026-11-05: "do it with master
batch trades and current batch trades, before and after … we have the data of the trades and their post 45 minutes results".

No replica, no klines: every number comes from the fill's own recorded columns.
COHORT (full-size only — probes out by sweep_separators._is_probe, BOTH flags; the N per source is printed and checked):
  · MASTER = reports/MASTER_POOL_stacked.csv, MOMENTUM LONG CLOSED, stack_keep, non-probe — before = stack_pct (today's stack, 1×)
  · B25    = the latest orders export, MOMENTUM LONG CLOSED, non-probe, opened ≥ 2026-10-23 18:00, not in the master — before = as traded
RULE for a fixed TP at +x % (x = net % at 1×, the same unit as pnl_percentage / peak_pnl / post_exit_*, all measured from the ENTRY):
  ① peak_pnl ≥ x during the trade → the TP fills at +x (the peak came before the real exit) — unless the fill fell to today's −0.70 stop
     BEFORE its peak (older eras' wider stops): then compared as booked (Δ 0)
  ② else the real exit was a loss / time exit (stop, premise, hard stop, no-expansion, manual) → unchanged (the TP never fired first)
  ③ else the real exit was a PROFIT exit (runner trail, ladder, hard TP, trailing stop) → with a plain TP the trade stays open; read the
     45 min the bot recorded after the exit: TP reached (post_exit_peak_pnl ≥ x) before the stop (post_exit_trough_pnl ≤ STOP) → +x;
     stop first → STOP; both, order unknown → STOP (conservative); neither → still open at 45 min, marked at post_exit_final_pnl
  ④ ③ without post-exit data → unpriced (shown, never silently 0)
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "best_case"))
import review_math as RM  # noqa: E402
import sweep_separators as SW  # noqa: E402

MASTER = os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv")
OUT_MD = os.path.join(ROOT, "reports", "ML_FIXED_TP_RECORDED.md")
OUT_CSV = os.path.join(ROOT, "reports", "ML_FIXED_TP_RECORDED_fills.csv")
B25_OPEN = pd.Timestamp("2026-10-23 18:00")
B25_START, B25_FLOWS = 2900.0, [("2026-10-26T13:50:00", 7100.0)]          # DECISION_LOG 375: paper 2,900 → 10,000 top-up, no reset
TPS = (0.5, 1.0, 2.0, 3.0)
STOP = -0.70                                                               # today's momentum-long L1 stop, net % at 1× (B25 stops −0.69…−0.75)
PROFIT_EXITS = ("RUNNER_TRAIL", "HARD_TP_LADDER", "HARD_TP", "TRAILING_STOP")
KNOWN_FULL_MASTER = 158                                                    # memory 11-04: momentum-long master full-size kept


def _true(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _key(s):
    return pd.to_datetime(s, format="mixed").dt.strftime("%Y-%m-%dT%H:%M:%S")


def latest_export():
    c = glob.glob(os.path.expanduser("~/Downloads/scalpars_orders_*.csv")) + glob.glob(os.path.join(ROOT, "exports_archive", "scalpars_orders_*.csv"))
    return max(c, key=lambda p: os.path.basename(p)) if c else None


def _ml(d):
    return (d.entry_strategy.astype(str) == "MOMENTUM") & (d.direction.astype(str) == "LONG") & (d.status.astype(str) == "CLOSED")


COLS = ["block", "era", "opened_at", "closed_at", "pair", "close_reason", "peak_reached_at", "trough_reached_at", "before", "peak_pnl",
        "trough_pnl", "post_exit_peak_pnl", "post_exit_trough_pnl", "post_exit_peak_minutes", "post_exit_trough_minutes", "post_exit_final_pnl",
        "notional_value", "entry_mkt_up4h_pct",   # 🌡 (DL 489) the engine's V2 Market-hot stamp (NULL before the deploy → rebuilt)
        "entry_adx_delta"]   # 🔂 (DL 534) the second-chance zone — the watch lines re-base pre-ship fills on today's exit (scripts/ml_sc_history.py)


def load_master():
    """→ (master full-size momentum longs kept on today's stack, before = stack_pct, the number of probe fills dropped)."""
    M = pd.read_csv(MASTER, low_memory=False)
    mm = _ml(M) & _true(M.stack_keep)
    probes = int((mm & SW._is_probe(M)).sum())
    M = M[mm & ~SW._is_probe(M)].copy()
    M["block"], M["before"] = "MASTER", pd.to_numeric(M.stack_pct, errors="coerce")
    return M.reindex(columns=COLS), probes


def batch_rows(E, master_keys, block="B25", era="B25"):
    """→ (closed full-size momentum longs of an export opened ≥ the B25 start and not in the master — before = as traded, probes dropped)."""
    eb = _ml(E) & (pd.to_datetime(E.opened_at, format="mixed") >= B25_OPEN)
    probes = int((eb & SW._is_probe(E)).sum())
    E = E[eb & ~SW._is_probe(E)].copy()
    E = E[~(_key(E.opened_at) + E.pair).isin(master_keys)]
    E["block"], E["era"], E["before"] = block, era, pd.to_numeric(E.pnl_percentage, errors="coerce")
    return E.reindex(columns=COLS), probes


def finish(F):
    """numeric columns + the per-fill flags every read needs (stop before peak, day, exit kind, ML_MKT_HOT on the V2 stamp / rebuild — DL 489)."""
    F = F.reset_index(drop=True)
    for c in COLS[8:]:
        F[c] = pd.to_numeric(F[c], errors="coerce")
    F["stop_before_peak"] = stop_before_peak(F)
    F["day"] = pd.to_datetime(F.opened_at, format="mixed").dt.strftime("%Y-%m-%d")
    F["exit_kind"] = F.close_reason.astype(str).str.replace(r" L\d.*$", "", regex=True)
    F["mkt_hot"] = attach_mkt_hot(F)
    return F


def master_keys(M=None):
    """keys (opened_at, pair) of EVERY master momentum long — raw, before the keep / probe filter (second review 479): a rebuilt master that
    absorbs a batch marks some of its fills stack_keep=False; keyed on the kept rows only, those would come back from the exports."""
    if M is None:
        R = pd.read_csv(MASTER, low_memory=False, usecols=["opened_at", "pair", "direction", "entry_strategy", "status"])
        M = R[_ml(R)]
    return set(_key(M.opened_at) + M.pair)


def load(export):
    M, probes_m = load_master()
    E, probes_b = batch_rows(pd.read_csv(export, low_memory=False), master_keys())
    F = finish(pd.concat([M, E], ignore_index=True))
    return F, dict(master=int((F.block == "MASTER").sum()), b25=int((F.block == "B25").sum()), probes_master=probes_m, probes_b25=probes_b)


def attach_mkt_hot(F, now_ms=None):
    """the ML_MKT_HOT watch line per fill (True = in the hot zone = the fills the candidate filter would BLOCK; NaN = not scored) on the 🌡 V2
    rolling top-100 (DECISION_LOG 489; replaced the frozen 108 and the best-case zones.csv read on 2026-11-06): the fill's own engine stamp
    entry_mkt_up4h_pct when present, else rebuilt by scripts/scout_ml_mkt_hot_watch.py from the 11-06 study cache — NO network here (fills
    past the cache stay NaN; scout_ml_fixed_tp_watch.score_missing reads those through the budgeted public klines)."""
    import scout_ml_mkt_hot_watch as MH
    o = pd.to_datetime(F.opened_at.astype(str).str.replace(" ", "T").str[:19], format="ISO8601", errors="coerce")
    c = pd.DataFrame({"o": o.values, MH.STAMP: pd.to_numeric(F.get(MH.STAMP, pd.Series(np.nan, index=F.index)), errors="coerce").values},
                     index=F.index)
    miss = MH.needs_rebuild(c) & c.o.notna()
    br = pd.Series(dtype=float)
    if miss.any():
        now_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
        closes, _ok = MH.fetch_closes(int(c.o[miss].min().value // 10**6), now_ms, days=sorted(set(c.o[miss].dt.normalize())), fetch=False)
        br = MH.breadth_series(closes) if len(closes) else br
    z = MH.score(c[c.o.notna()], br).zone.reindex(F.index)
    return z.map(lambda x: np.nan if (pd.isna(x) or x == "unscored") else x == "zone").astype(object)


def stop_before_peak(F, stop=STOP):
    """True where the fill fell to today's stop BEFORE its peak (older eras ran wider stops — deep review 479): today's −0.70 would have
    closed it first, so no TP can be credited and the fill is compared as booked. PURE."""
    tr = pd.to_numeric(F.get("trough_pnl"), errors="coerce")
    tt = pd.to_datetime(F.get("trough_reached_at"), format="mixed", errors="coerce", utc=True)
    pt = pd.to_datetime(F.get("peak_reached_at"), format="mixed", errors="coerce", utc=True)
    return ((tr <= stop) & tt.notna() & (pt.isna() | (tt < pt))).to_numpy(bool)


def tp_outcome(r, x, stop=STOP):
    """→ (after %, how). PURE (one fill as a mapping)."""
    if r.get("stop_before_peak", False):
        return float(r["before"]), "same exit (today's stop first)"
    if pd.notna(r["peak_pnl"]) and r["peak_pnl"] >= x:
        return float(x), "TP in trade"
    if r["exit_kind"] not in PROFIT_EXITS:
        return float(r["before"]), "same exit"
    pk, tr, fin = r["post_exit_peak_pnl"], r["post_exit_trough_pnl"], r["post_exit_final_pnl"]
    if pd.isna(pk) or pd.isna(tr):
        return np.nan, "no post-exit data"
    hit_tp, hit_stop = pk >= x, tr <= stop
    if hit_tp and hit_stop:                     # extremes' times are not first touches → order unknown → the stop (deep review 479)
        return float(stop), "NEW STOP after exit"
    if hit_tp:
        return float(x), "TP after exit"
    if hit_stop:
        return float(stop), "NEW STOP after exit"
    return float(fin) if pd.notna(fin) else np.nan, "open at 45 min"


def price(F):
    for x in TPS:
        o = F.apply(lambda r: tp_outcome(r, x), axis=1)
        F[f"TP{x:g}"] = [a for a, _ in o]
        F[f"TP{x:g}_how"] = [b for _, b in o]
    return F


def _blk(F):
    return [("MASTER (all archived batches)", F.block == "MASTER"), ("B25 (current batch, every fill)", F.block == "B25"),
            ("ALL", pd.Series(True, index=F.index))]


def table(F):
    import luck_test as LT
    rows = []
    for name, m in _blk(F):
        S = F[m]
        D = pd.DataFrame({f"TP{x:g}": S[f"TP{x:g}"] - S.before for x in TPS})
        okall = D.notna().all(axis=1)                   # the luck test runs on fills priced under EVERY TP (deep review 479: same N as shown)
        for x in TPS:
            c = f"TP{x:g}"
            ok = okall
            g = LT.gain_line(D[ok], S.day.values[ok.to_numpy()], c) if ok.sum() >= LT.MIN_N else dict(p_single=np.nan, p_search=np.nan, verdict=LT.FEW)
            how = S[c + "_how"].value_counts()
            rows.append(dict(block=name, tp=x, n=int(ok.sum()), unpriced=int((~ok).sum()), before_avg=S.before[ok].mean(), after_avg=S[c][ok].mean(),
                             before_wr=(S.before[ok] > 0).mean() * 100, after_wr=(S[c][ok] > 0).mean() * 100, sum_before=S.before[ok].sum(),
                             sum_after=S[c][ok].sum(), better=int((D[c][ok] > 0.005).sum()), worse=int((D[c][ok] < -0.005).sum()),
                             tp_in=int(how.get("TP in trade", 0)), tp_after=int(how.get("TP after exit", 0)), new_stop=int(how.get("NEW STOP after exit", 0)),
                             open45=int(how.get("open at 45 min", 0)), p=g.get("p_single", np.nan), p_search=g.get("p_search", np.nan),
                             verdict=g.get("verdict", ""), loo=RM.loo_text(D[c][ok], S.pair[ok] + " " + S.day[ok].str[5:], unit="pts"),
                             loo_day=RM.loo_text(D[c][ok].groupby(S.day[ok]).sum(), D[c][ok].groupby(S.day[ok]).sum().index, unit="pts")))
    return pd.DataFrame(rows)


def hot_table(F):
    """ML_MKT_HOT filter view per block: today's exit on ALL fills → today's exit on the KEPT fills (not hot, scored) → kept + fixed TP."""
    import luck_test as LT
    rows = []
    for name, m in _blk(F):
        S = F[m]
        kept = S.mkt_hot.notna() & ~S.mkt_hot.fillna(False).astype(bool)
        K = S[kept]
        rows.append(dict(block=name, line="All fills · today's exit", n=len(S), avg=S.before.mean(), wr=(S.before > 0).mean() * 100,
                         sum=S.before.sum(), p=np.nan, verdict=""))
        hot = S[S.mkt_hot.fillna(False).astype(bool)]
        rows.append(dict(block=name, line="Blocked by MKT_HOT (hot) · today's exit", n=len(hot), avg=hot.before.mean(),
                         wr=(hot.before > 0).mean() * 100, sum=hot.before.sum(), p=np.nan, verdict=""))
        rows.append(dict(block=name, line="Kept by MKT_HOT · today's exit", n=len(K), avg=K.before.mean(), wr=(K.before > 0).mean() * 100,
                         sum=K.before.sum(), p=np.nan, verdict=""))
        D = pd.DataFrame({f"TP{x:g}": K[f"TP{x:g}"] - K.before for x in TPS})
        ok = D.notna().all(axis=1)
        for x in TPS:
            c = f"TP{x:g}"
            g = LT.gain_line(D[ok], K.day.values[ok.to_numpy()], c) if ok.sum() >= LT.MIN_N else dict(p_single=np.nan, verdict=LT.FEW)
            rows.append(dict(block=name, line=f"Kept by MKT_HOT · fixed TP +{x:g} %", n=int(ok.sum()), avg=K[c][ok].mean(),
                             wr=(K[c][ok] > 0).mean() * 100, sum=K[c][ok].sum(), p=g.get("p_single", np.nan), verdict=g.get("verdict", ""),
                             today_same=K.before[ok].mean()))
        rows[-len(TPS) - 3]["unscored"] = int(S.mkt_hot.isna().sum())
    return pd.DataFrame(rows)


def per_batch(F):
    out = []
    for era, S in F.groupby("era", sort=False):
        ok = S[[f"TP{x:g}" for x in TPS]].notna().all(axis=1)          # before and after on the SAME priced fills (deep review 479)
        row = dict(era=era, n=int(ok.sum()), before=S.before[ok].sum())
        for x in TPS:
            row[f"TP{x:g}"] = S[f"TP{x:g}"][ok].sum(min_count=1)
        out.append(row)
    return pd.DataFrame(out)


def b25_dcr(F, export, keep=None):
    """DCR-cal %/day of the momentum-long sleeve on the REAL B25 account path, before → after each TP (each fill's actual notional)."""
    acc = pd.read_csv(export, low_memory=False)
    acc = acc[(acc.status == "CLOSED") & (pd.to_datetime(acc.opened_at, format="mixed") >= B25_OPEN)].copy()
    acc["k"] = _key(acc.opened_at) + acc.pair + acc.direction.astype(str)
    acc["u0"] = pd.to_numeric(acc.pnl, errors="coerce")
    B = F[F.block == "B25"].assign(k=lambda d: _key(d.opened_at) + d.pair + "LONG")
    sleeve = acc.k.isin(set(B.k))
    assert int(sleeve.sum()) == len(B), f"account merge matched {int(sleeve.sum())} of {len(B)} B25 fills"
    t0, t1 = pd.to_datetime(acc.opened_at, format="mixed").min(), pd.to_datetime(acc.closed_at, format="mixed").max()
    out = {"before": RM.dcr_account(acc, sleeve, "u0", B25_START, t0, t1, B25_FLOWS)}
    d = B.set_index("k")
    blocked = set(d.index[~keep.loc[B.index].to_numpy(bool)]) if keep is not None else set()
    if keep is not None:                                                       # the filter: blocked fills never open → $0
        acc["uk"] = np.where(acc.k.isin(blocked), 0.0, acc.u0)
        out["kept"] = RM.dcr_account(acc, sleeve, "uk", B25_START, t0, t1, B25_FLOWS)
    for x in TPS:
        delta = (d[f"TP{x:g}"] - d.before).fillna(0.0) / 100 * d.notional_value
        acc[f"u{x:g}"] = acc.u0 + acc.k.map(delta).fillna(0.0)
        if keep is not None:
            acc[f"u{x:g}"] = np.where(acc.k.isin(blocked), 0.0, acc[f"u{x:g}"])
        out[f"TP{x:g}"] = RM.dcr_account(acc, sleeve, f"u{x:g}", B25_START, t0, t1, B25_FLOWS)
    return out


def f(v, d=2, sign=True):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:+.{d}f}" if sign else f"{v:.{d}f}")


def report(F, T, PB, N, dcr, export, H=None, dcr_hot=None):
    L = ["# Momentum longs — fixed take-profits from the RECORDED trade data (observe-only)", "",
         f"Full-size fills only: **master {N['master']}** (kept on today's stack; {N['probes_master']} probe fills excluded) · **B25 {N['b25']}** "
         f"(`{os.path.basename(export)}`, every momentum long since 2026-10-23 18:00; {N['probes_b25']} probe fills excluded). "
         f"Before = the fill's booked % at 1× (master: stack_pct). After = the same fill with a plain fixed TP, read from its own recorded "
         f"peak and the 45 minutes the bot recorded after its exit (rule in the script header; stop {STOP:+.2f} %). No replica, no klines.", ""]
    for name, _ in _blk(F):
        S = T[T.block == name]
        L += [f"## {name}", "", "| Exit | N | Avg % before → after | WR before → after | Σ % before → after | Better / worse | TP hit in trade · after exit | "
              "New stops | Open at 45 min | Unpriced | p (luck, this TP) | p after 4-TP search | Verdict |", "|" + "---|" * 13]
        for r in S.itertuples():
            L.append(f"| Fixed TP +{r.tp:g} % | {r.n} | {f(r.before_avg)} → **{f(r.after_avg)}** | {r.before_wr:.0f} % → {r.after_wr:.0f} % | "
                     f"{f(r.sum_before, 1)} → {f(r.sum_after, 1)} | {r.better} / {r.worse} | {r.tp_in} · {r.tp_after} | {r.new_stop} | {r.open45} | "
                     f"{r.unpriced} | {f(r.p, 2, False)} | {f(r.p_search, 2, False)} | {r.verdict} |")
        L.append("")
        for r in S.itertuples():
            L.append(f"- TP +{r.tp:g} %: by fill {r.loo} · by DAY {r.loo_day}")
        L.append("")
    L += ["## Where the change comes from — Σ Δ % by what happened (count)", "",
          "| Block | TP | TP in trade (cuts a bigger win) | TP after the real exit | New stop after the exit | Still open at 45 min (marked) |",
          "|---|---|---|---|---|---|"]
    for name, m in _blk(F)[:2]:
        S = F[m]
        for x in TPS:
            c = f"TP{x:g}"
            d = (S[c] - S.before).groupby(S[c + "_how"])
            g = lambda k: f"{f(d.sum().get(k, 0.0), 1)} ({int(d.size().get(k, 0))})"
            L.append(f"| {name.split(' ')[0]} | +{x:g} % | {g('TP in trade')} | {g('TP after exit')} | {g('NEW STOP after exit')} | {g('open at 45 min')} |")
    L += ["", "A gain in the last column is not a take-profit: it is the trade held 45 min longer than today's exit, marked where it stood.", ""]
    if H is not None:
        L += ["## Entry filter ML_MKT_HOT (watch line, DL 416) × fixed TP — master and B25, before → after", "",
              "Kept = fills NOT in the hot zone (≤ 50 % of the rolling top-100 above their 4 h-earlier close at entry — V2, DL 489) — the fills the candidate "
              "filter would let through. Blocked = the hot-zone fills it would refuse. Unscored fills (no market data) are in neither.", ""]
        for name, _ in _blk(F):
            S = H[H.block == name]
            us = S.unscored.dropna()
            L += [f"### {name}" + (f" · {int(us.iloc[0])} unscored" if len(us) else ""), "",
                  "| Cohort · exit | N | Avg % | WR | Σ % | p (luck vs today's exit, kept fills) | Verdict |", "|---|---|---|---|---|---|---|"]
            for r in S.itertuples():
                L.append(f"| {r.line} | {r.n} | {f(r.avg)}" + (f" (today {f(r.today_same)} on these)" if pd.notna(getattr(r, "today_same", np.nan)) else "")
                         + f" | {r.wr:.0f} % | {f(r.sum, 1)} | {f(r.p, 2, False)} | {r.verdict} |")
            L.append("")
        if dcr_hot:
            L += ["B25 daily compound with the filter (blocked fills never opened) — momentum-long sleeve, real account path:", "",
                  "| | All · today | Kept · today | Kept · TP +0.5 | Kept · TP +1 | Kept · TP +2 | Kept · TP +3 |", "|---|---|---|---|---|---|---|",
                  "| DCR-cal %/day | " + " | ".join(f(dcr_hot[k], 3) for k in ("before", "kept", "TP0.5", "TP1", "TP2", "TP3")) + " |", ""]
    L += ["## Per batch — Σ % at 1× before → after", "", "| Batch | N | Today | TP +0.5 | TP +1 | TP +2 | TP +3 |", "|---|---|---|---|---|---|---|"]
    for _, r in PB.iterrows():
        L.append(f"| {r['era']} | {r['n']} | {f(r['before'], 1)} | " + " | ".join(f(r[f'TP{x:g}'], 1) for x in TPS) + " |")
    L += ["", "## B25 daily compound (momentum-long sleeve, real account path)", "",
          "| | Today | TP +0.5 | TP +1 | TP +2 | TP +3 |", "|---|---|---|---|---|---|",
          "| DCR-cal %/day | " + " | ".join(f(dcr[k], 3) for k in ("before", "TP0.5", "TP1", "TP2", "TP3")) + " |", "",
          "Unpriced fills (no post-exit record) stay as booked in the DCR. ",
          "Master DCR is not shown: master momentum longs are a few fills a day spread over 128 days (the per-batch Σ above carries it).", "",
          "**Blind spots:** ① the 45-min post-exit window — a trade still open at 45 min is marked there, not followed to its real end · "
          "② post_exit_peak/trough are the extremes, not first-touch times: when both the TP and the stop were reached after the exit, the "
          "stop is booked (order unknown) · ③ in-trade peak_pnl ≥ TP books the TP — unless the fill fell to today's −0.70 stop BEFORE its "
          "peak (older, wider stops), then it is compared as booked · ④ master exits ran under older exit rules: their post-exit window "
          "starts at THAT exit, and 'same exit' losses keep their old-era stop (some below −0.70) while new stops use −0.70 · ⑤ the "
          "180-min no-expansion exit is kept as booked · ⑥ a gain in 'still open at 45 min' is holding longer, not a TP — read the by-DAY "
          "leave-one-out before trusting it."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export")
    a = ap.parse_args()
    export = a.export or latest_export()
    F, N = load(export)
    print(f"cohort: master {N['master']} full-size ({N['probes_master']} probes out) · B25 {N['b25']} ({N['probes_b25']} probes out)")
    if N["master"] > KNOWN_FULL_MASTER + 10:
        sys.exit(f"master {N['master']} ≫ known full-size {KNOWN_FULL_MASTER}: probes leaked — stop")
    F = price(F)
    T, PB = table(F), per_batch(F)
    dcr = b25_dcr(F, export)
    H = hot_table(F)
    keep = F.mkt_hot.notna() & ~F.mkt_hot.fillna(False).astype(bool)
    dcr_hot = b25_dcr(F, export, keep=keep)
    F.to_csv(OUT_CSV, index=False)
    open(OUT_MD, "w").write(report(F, T, PB, N, dcr, export, H, dcr_hot))
    print(f"→ {OUT_MD}")


if __name__ == "__main__":
    main()
