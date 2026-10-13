#!/usr/bin/env python3
"""YR6 WL — FRENZY headline numbers recomputed a SECOND, independent way (stdlib csv only — no pandas, no shared loader), plus the
confirmation P of the LONG top-50 split.

  venv/bin/python -I scripts/study_yr6wl_frenzy_verify.py

Recomputes (seed 1, CLOSED, non-probe FRENZY rows; y = pct_new; day = opened_at[:10]):
  per sleeve × regime N / WR / avg · top-50 split (own parse of both universe files: same-day OR prior-day top-50) ·
  LITE blocked-by-G1 (entry_btc_1h_slope ≤ 0) avg per regime · LONG non-top-50 avg per regime · WILLY 2025 fills < −5 %.
Then (pandas, day bootstrap): P(diff same sign) of LONG top-50 vs not in each regime.
"""
import csv, datetime as dt, os, random, sys
from collections import defaultdict

ROOT = "/Users/guillearslanian/Downloads/NOFA AI"
csv.field_size_limit(10 ** 9)
top = set()
for f in ("reports/backtest_cache/universe_daily_top50.csv", "reports/backtest_cache_2025/universe_daily_top50.csv"):
    for r in csv.DictReader(open(os.path.join(ROOT, f))):
        d = dt.datetime.fromtimestamp(int(r["date_ms"]) / 1000, dt.timezone.utc).date()
        top.add((d.isoformat(), r["pair"])); top.add(((d + dt.timedelta(days=1)).isoformat(), r["pair"]))
agg = defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, "reports/ENGINE_REPLAY_YR6_2Y_TODAY_fills.csv"))):
    if r["seed"] != "1" or not r["sleeve"].startswith("FRENZY") or r["status"] != "CLOSED" or r["is_probe"] == "True":
        continue
    y = float(r["pct_new"]); day = r["opened_at"][:10]; s, g = r["sleeve"], r["regime"]
    t50 = (day, r["pair"]) in top
    agg[(s, g)].append(y); agg[(s, g, "top50" if t50 else "non")].append(y)
    try:
        sl = float(r["entry_btc_1h_slope"])
        if s == "FRENZY_LITE" and sl <= 0:
            agg[(s, g, "G1blk")].append(y)
    except ValueError:
        pass
    if s == "FRENZY_WILLY" and g == "2025 BULL" and y < -5:
        agg[("WILLY<-5",)].append(y)
for k in sorted(agg, key=str):
    v = agg[k]
    print(f"{' · '.join(k):45s} N {len(v):4d}  WR {100 * sum(x > 0 for x in v) / len(v):5.1f}  avg {sum(v) / len(v):+.3f}  sum {sum(v):+.1f}")

sys.path.insert(0, os.path.join(ROOT, "scripts"))
import study_yr6wl_frenzy_common as C  # noqa: E402
D = C.load()
for s in ("FRENZY_LONG", "FRENZY_WIDE", "FRENZY_LITE", "FRENZY_WILLY"):
    for reg in C.REGIMES:
        S = D[(D.sleeve == s) & (D.regime == reg)]
        m = ~S.top50.values
        d = S.y[m].mean() - S.y[~m].mean()
        print(f"{s} {reg}: non-top50 − top50 = {d:+.3f}  P(diff < 0, day bootstrap) = {C.boot_diff_p(S.y.values, m, S.day.values, -1):.3f}")
