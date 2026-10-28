#!/usr/bin/env python3
"""NOLATE-on-runners before → after (operator 10-28: "effect in master batch? and in B25 before and after").

Every CLOSED FRENZY_WILLY / FRENZY_WILLY_HT fill — the archived master batches (reports/BASELINE*_orders*.csv, the set
build_willy_exit_cf prices) + the current batch export — priced on its own path with build_willy_exit_cf.price_row:
  BEFORE = today's live V2b exit (T at the spike high, −4, late +0.25 @ 30 for EVERY trade, cap 90, resting TP)
  AFTER  = the same, but NO late TP when the trade's target T > +1.25 (a runner); T = +1.25 trades keep the late TP.
Per batch: N · WR · avg % · Σ % · Δ · DCR-cal before → after at an INTENDED size (r_d = Σ fill % × X / 100 per open day, X = notional ÷
account: WILLY 1.91 (margin $1,950 at $10.2k × 10×), HT ¼ of it — labelled as intended sizing; the archived batches carry no account
path here) · B25 also on the REAL account path (review_math.dcr_account: actual notionals, $2,900 start, +$7,100 top-up, DL 406 ⑥).
Cache-first (study_b19_fzexit_common.load_path). Usage: venv/bin/python -I scripts/study_willy_nolate_runners.py [current_orders.csv]
"""
import glob, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import build_willy_exit_cf as CF  # noqa: E402
import review_math as RM  # noqa: E402

B25_START, B25_FLOWS = 2900.0, [("2026-10-26T13:50:00", 7100.0)]   # DECISION_LOG 375: paper 2,900 → 10,000 top-up, no reset

X = {"FRENZY_WILLY": 1950 * 10 / 10188, "FRENZY_WILLY_HT": 0.25 * 1950 * 10 / 10188}
OUT = os.path.join(ROOT, "reports", "WILLY_NOLATE_RUNNERS_BEFORE_AFTER.csv")


def batches(current):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "reports", "BASELINE[0-9]*_*orders*.csv"))):
        d = pd.read_csv(f, low_memory=False)
        if "entry_strategy" in d and d.entry_strategy.astype(str).isin(CF.STRATS).any():
            out.append((os.path.basename(f).split("_")[0], d))
    if current:
        d = pd.read_csv(current, low_memory=False)
        d = d[pd.to_datetime(d.opened_at, format="mixed") >= pd.Timestamp("2026-10-23 18:30")]
        out.append(("B25 (current)", d))
    return out


def dcr(rows, col, span_days):
    if not len(rows):
        return 0.0
    r = rows.assign(day=pd.to_datetime(rows.closed_at, format="mixed").dt.date, w=rows[col] * rows.entry_strategy.map(X) / 100.0)   # booked on the close
    g = (1 + r.groupby("day").w.sum()).prod()
    return (g ** (1.0 / span_days) - 1) * 100 if g > 0 else -100.0


def main():
    th = CF.load_th()
    th_nl = dict(th, frenzy_willy_late_tp_minutes=0)
    rule = CF.rule_v2b(th)
    tp = CF.pre356_params(th)[0]
    cache, tcache, rows, seen = {}, {}, [], set()
    for name, d in batches(sys.argv[1] if len(sys.argv) > 1 else None):
        t0, t1 = pd.to_datetime(d.opened_at, format="mixed").min(), pd.to_datetime(d.closed_at, format="mixed").max()
        span = (t1.normalize() - t0.normalize()).days + 1
        w = d[d.entry_strategy.astype(str).isin(CF.STRATS) & (d.status == "CLOSED")]
        w = w.drop_duplicates(["opened_at", "pair", "direction"])
        w = w[~(w.opened_at.astype(str).str.replace(" ", "T").str[:19] + w.pair).isin(seen)]   # cross-batch dedup (review 2)
        seen |= set(w.opened_at.astype(str).str.replace(" ", "T").str[:19] + w.pair)
        for r in w.to_dict("records"):
            a = CF.price_row(r, th, cache, tcache, rule)
            b = CF.price_row(r, th_nl, cache, tcache, rule)
            T = a["target_pct"]
            runner = T is not None and T > tp + 1e-9
            rows.append(dict(batch=name, span_days=span, opened_at=a["opened_at"], closed_at=str(r.get("closed_at")), pair=a["pair"],
                             entry_strategy=a["entry_strategy"],
                             live_pct=a["live_pct"], target=T, runner=runner, before=a["cf_pct"], before_why=a["cf_why"],
                             after=(b["cf_pct"] if runner else a["cf_pct"]), after_why=(b["cf_why"] if runner else a["cf_why"])))
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    un = R[R.before.isna() | R.after.isna()]
    P = R.dropna(subset=["before", "after"]).copy()
    P["delta"] = P.after - P.before
    print(f"{len(R)} fills · priced {len(P)} · unpriced {len(un)}" + (": " + ", ".join(f"{b} {p} {o}" for b, p, o in zip(un.batch, un.pair, un.opened_at)) if len(un) else ""))
    print("| Batch | Days | N | runners | WR before → after | avg % before → after | Σ % before → after | Δ Σ % | changed (better/worse) | DCR-cal %/day before → after (INTENDED sizing) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    groups = [(b, g) for b, g in P.groupby("batch", sort=False)]
    master = P[P.batch != "B25 (current)"]
    for label, g in groups + [("MASTER (all archived)", master), ("ALL incl. B25", P)]:
        if not len(g):
            continue
        if label.startswith(("MASTER", "ALL")):
            spans = g.groupby("batch").span_days.first()
            dB = np.prod([(1 + dcr(x, "before", spans[b]) / 100) ** spans[b] for b, x in g.groupby("batch")]) ** (1 / spans.sum()) - 1
            dA = np.prod([(1 + dcr(x, "after", spans[b]) / 100) ** spans[b] for b, x in g.groupby("batch")]) ** (1 / spans.sum()) - 1
            days = int(spans.sum()); dB *= 100; dA *= 100
        else:
            days = int(g.span_days.iloc[0]); dB = dcr(g, "before", days); dA = dcr(g, "after", days)
        ch = g[g.delta.abs() > 1e-6]
        print(f"| {label} | {days} | {len(g)} | {int(g.runner.sum())} | {(g.before > 0).mean():.0%} → {(g.after > 0).mean():.0%} | "
              f"{g.before.mean():+.3f} → {g.after.mean():+.3f} | {g.before.sum():+.2f} → {g.after.sum():+.2f} | {g.delta.sum():+.2f} | "
              f"{len(ch)} ({(ch.delta > 0).sum()}/{(ch.delta < 0).sum()}) | {dB:+.3f} → {dA:+.3f} |")
    print("\n🧮 Leave-one-out (DECISION_LOG 406):")
    for label, g in groups + [("MASTER (all archived)", master)]:
        print(f"  {label}: " + RM.loo_text(g.delta, g.pair + " " + g.opened_at.str[5:10]))
    cur = sys.argv[1] if len(sys.argv) > 1 else None
    b = P[P.batch == "B25 (current)"]
    if cur and len(b):
        acc = pd.read_csv(cur, low_memory=False)
        acc = acc[(acc.status == "CLOSED") & (pd.to_datetime(acc.opened_at, format="mixed") >= pd.Timestamp("2026-10-23 18:30"))].copy()
        acc["k"] = acc.opened_at.astype(str).str.replace(" ", "T").str[:19] + acc.pair + acc.direction.astype(str)
        acc["usd_before"] = pd.to_numeric(acc.pnl, errors="coerce")
        b = b.assign(k=b.opened_at.astype(str).str.replace(" ", "T").str[:19] + b.pair + "LONG")
        acc = acc.merge(b[["k", "before", "after"]], on="k", how="left")
        isw = acc.before.notna()
        assert int(isw.sum()) == len(b), f"account merge matched {int(isw.sum())} of {len(b)} WILLY fills"   # review: never a silent no-op
        acc.loc[~isw, "after"] = acc.loc[~isw, "before"]
        acc["usd_after"] = acc.usd_before
        acc.loc[isw, "usd_after"] = acc.loc[isw, "usd_before"] + (acc.loc[isw, "after"] - acc.loc[isw, "before"]) / 100 * pd.to_numeric(acc.loc[isw, "notional_value"], errors="coerce")
        t0 = pd.to_datetime(acc.opened_at, format="mixed").min(); t1 = pd.to_datetime(acc.closed_at, format="mixed").max()
        sleeve = acc.entry_strategy.isin(CF.STRATS)                   # review 2: the sleeve mask, priced or not (unpriced stay as booked)
        for nm, m in (("WILLY + HT", sleeve), ("WILLY only", acc.entry_strategy == "FRENZY_WILLY")):
            bf = RM.dcr_account(acc, m, "usd_before", B25_START, t0, t1, B25_FLOWS)
            af = RM.dcr_account(acc, m, "usd_after", B25_START, t0, t1, B25_FLOWS)
            print(f"  📈 B25 DCR on the REAL account path ({nm}; start ${B25_START:,.0f}, top-up +$7,100 10-26 13:50, each fill's actual "
                  f"notional): {bf:+.3f} → {af:+.3f} %/day · $ {acc.loc[m, 'usd_before'].sum():+,.0f} → {acc.loc[m, 'usd_after'].sum():+,.0f}")
    print("\nChanged fills:")
    for r in P[P.delta.abs() > 1e-6].itertuples():
        print(f"  {r.batch} {r.opened_at} {r.pair} {r.entry_strategy} T {r.target:+.2f} · before {r.before:+.2f} {r.before_why} → after {r.after:+.2f} {r.after_why} (Δ {r.delta:+.2f})")
    print(f"\n→ {OUT}")


if __name__ == "__main__":
    main()
