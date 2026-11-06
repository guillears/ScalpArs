#!/usr/bin/env python3
"""📊 Best-case view ③ — how much each best line can be trusted (operator 2026-11-03: "how many more trades we need … to really trust the
results? with high confidence level?").

Per best line (from best_dcr_noflip.csv, plus the FAN flips TP +2 exit):
  BUILT ON  the trades the line was found on — every master fill + current-batch fills opened before the line was frozen (in-sample).
  CLEAN     current-batch fills opened ≥ the freeze (the DL commit + 10 min, as its scout counts them): never used to pick the line.
  measure   block line → the BLOCKED fills' booked P&L % (the line claims they lose) · exit / entry-timing line → Δ = after − today's exit per
            fill (FAN flips: − booked, their exit never changed; FB_CONFIRM: − the +5 / −10 walk, DECISION_LOG 448).
  N for 95 %  trades for an 80 % chance of showing the effect at 95 % confidence: n = ((1.645 + 0.84) × sd ÷ |mean|)², sd and mean from
            BUILT ON, floored at the checkpoint N — still optimistic (in-sample means are inflated, fills cluster by day);
            "never" = BUILT ON does not even point the line's way. A block line also shows N per side to prove the blocked fills do
            WORSE than the kept ones (two-sample) — a line whose blocked fills average ≈ 0 can be clearly worse than the rest (it lifts the
            average trade and frees slots) yet never provably ADD money (ML_MKT_HOT; operator 11-03 caught the 3,300). FB_CONFIRM's BUILT ON is the current batch before the freeze + 1 B24 fill.
  deciding test (DECISION_LOG 484)  the count that decides THIS line, from its registered gate (YARD): "blocked < kept" per side at the
            gate's own one-sided 90 % (P(zone < rest) ≥ 0.90; its avg ≤ +0.05 / day-pair legs are not modelled) for the Market-hot lines (their blocked fills average ≈ 0, so "adds money" never proves — the 3,306), "adds money" for the rest.
  pace / ≈ dates  each line's trades per day over the WHOLE current batch (B25_START → the export's last fill), not since its freeze; the
            checkpoint date needs its N trades AND D trading days; a projection — zone fills come in bursts, so dates move by days.
  clean 95 % range  day-clustered bootstrap of the clean mean (4,000 resamples, seed 7); "–" below 3 clean fills.
  backtest  what the backtest said at registration (its scout docstring) — a cross-check, never a veto (MASTER OVER BACKTEST).
  call      line_calls() — the DECISION_LOG 447 ladder on CLEAN fills (PASS / STRADDLE / AGAINST / PRE); best_dcr.py switches on only PASS
            lines for the CLEAN-ONLY totals (operator 2026-11-04: "build the clean-only column", the number without the in-sample cut).
Reads reports/best_case/{zones,best_dcr_noflip,flip_exits,fb_combo}.csv (written by the other steps). Prints a markdown table.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SC = os.path.join(ROOT, "reports", "best_case")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import scout_gatekit as GK  # noqa: E402

# line → (freeze: DL number or fixed UTC time, checkpoint N, checkpoint days, backtest at registration)
LINES = {
    "WILLY_GAP": (414, 15, 8, "disagrees (3 / 3 seeds)"),
    "OBS-F1 late weak spike": ("2026-10-26T15:12:00", 15, 8, "disagrees (3 / 3 seeds)"),
    "FB_CONFIRM": (417, 15, 8, "agrees, small (+1.25 → +1.30 %/trade)"),
    "EXIT TP2": (411, 20, 8, "not run (history: crash window only)"),
    "LATE_BTCFALL": (353, 15, 8, "agrees 2025, fails 2026"),
    "MKT_HOT": (416, 15, 8, "none for HT (WILLY: reversed)"),
    "SURGE_BTC_HOT": (427, 15, 8, "not run"),
    "ML_MKT_HOT": (416, 15, 8, "agrees (3 / 3 seeds)"),
    "LONG_ADX35": (418, 15, 8, "disagrees (seed 1)"),
}
# (DECISION_LOG 484, operator 11-06 "why so long … maybe you are not considering trades you should") the test that DECIDES each line, from its own
# registered gate: "rest" = blocked fills must do WORSE THAN THE KEPT ONES (ML_MKT_HOT's operator bar: P(zone < rest); WILLY_HT's Market-hot is
# decided on the same bar, DL 416) · every other line = "money" (blocked fills must lose / the change must gain — the expectancy bar).
YARD = {"ML_MKT_HOT": "rest", "MKT_HOT": "rest"}
B25_START = pd.Timestamp("2026-10-23 18:30")   # the current batch's start (as best_dcr.B0) — the pace denominator; ⚠ update per batch
# 🎲 (DECISION_LOG 473) was the line's theory written BEFORE its data were read? first = pre-registered (a real test) · after = a story fitted to
# what the data showed (explains, never proves) · against = the theory frozen first predicted the opposite. Source: each line's DL entry.
THEORY = {"WILLY_GAP": "after (EMA cut from the data, DL 414)", "OBS-F1 late weak spike": "after (W/L screen, DL 384)",
          "FB_CONFIRM": "after (best of the waits tried, DL 417)", "EXIT TP2": "after (best of 3 exits, DL 411)",
          "LATE_BTCFALL": "first (cut pre-registered, DL 353)", "MKT_HOT": "after (borrowed from ML_MKT_HOT)",
          "SURGE_BTC_HOT": "after (cut seen in 6 fills, DL 427)", "ML_MKT_HOT": "against (frozen H5 said the opposite, DL 416)",
          "LONG_ADX35": "after (post-hoc cut, DL 418)"}
LUCK = {}   # (sleeve, line) → luck.csv row, filled in main() from best_dcr.py's luck test
TL = []     # (DECISION_LOG 485) one dict per row() call — the ④ checkpoint timeline is built from the SAME numbers as ③
REVIEW_AT = pd.Timestamp("2026-11-12 12:00")   # the next batch review (B25, ≈ Nov 12) — ④a projects to it; ⚠ update per batch (or pass a date)
TIMELINE_MARK = "<!-- ④ checkpoint timeline -->"
# how each line's registration search compares with the luck test's family (deep review 473: the family is not a bound either way)
LUCK_NOTE = {"FB_CONFIRM": "search priced: confirm +5/−10 and +1/−4, waits 1–24 h, selling 1–7 % higher (DL 474); 'a move against then a 1 h close back below' not priced",
             "WILLY_GAP": "registration searched wider: 28,545 cells", "ML_MKT_HOT": "registration searched 49 market conditions, not in this family",
             "LONG_ADX35": "registration kept one 1D cut — the 1D p is the closer read"}


def luck_cell(sleeve, line):
    r = LUCK.get((sleeve, line))
    if r is None:
        return "not run"
    if r["verdict"] == "TOO FEW":
        return f"too few to test ({int(r['n'])} trades before the freeze)"
    note = f" ({LUCK_NOTE[line]})" if line in LUCK_NOTE else ""
    dr = r.get("dropped", 0)
    if dr == dr and dr:
        note += f" · {int(dr)} trade(s) left out: an alternative unpriced"
    p1 = r.get("p_1d", float("nan"))
    p1s = f" · 1D search p {p1:.2f}" if p1 == p1 and p1 != r["p_search"] else ""
    return f"**{r['verdict']}** — alone p {r['p_single']:.3f}{p1s} · full search p {r['p_search']:.2f}{note}"


SHOW = {"FLIP:FAN_RATIO_GATE": "FAN flips", "FRENZY_WILLY": "WILLY", "FRENZY_WILLY_HT": "WILLY_HT", "SPIKE_FADE": "Spike Fade",
        "FLATBASE_SHORT": "Flatbase", "FRENZY_LITE_RETRY": "LITE RETRY", "SURGE_LONG": "SURGE Long", "MOMENTUM_LONG": "Momentum Long",
        "FRENZY_LONG": "FRENZY_LONG"}
LABEL = {"EXIT TP2": "Exit TP +2", "MKT_HOT": "Market-hot", "SURGE_BTC_HOT": "BTC RSI ≥ 78", "ML_MKT_HOT": "Market-hot"}


def freeze_ts(spec):
    if isinstance(spec, str):
        return pd.Timestamp(spec)
    ms = GK.dl_deploy_ms(spec, lag_min=10, label=f"DECISION_LOG {spec}")
    return None if ms is None else pd.Timestamp(int(ms), unit="ms")


def n95(x, sign):
    """sign −1: the line claims mean < 0 (blocked fills lose) · +1: claims mean > 0 (Δ of an exit / timing change)."""
    x = pd.Series(x, dtype=float).dropna()
    if len(x) < 3 or x.std() == 0:
        return "–"
    m = x.mean()
    if m * sign <= 0:
        return "never"
    return f"{int(np.ceil(((1.645 + 0.84) * x.std() / abs(m)) ** 2))}"


def n95_vs_rest(z, r, zc=1.645):
    """trades needed PER SIDE to show the blocked fills do worse than the kept ones (two-sample, 80 % power, one-sided 95 %)."""
    z, r = pd.Series(z, dtype=float).dropna(), pd.Series(r, dtype=float).dropna()
    if len(z) < 3 or len(r) < 3:
        return "–"
    gap = r.mean() - z.mean()
    if gap <= 0:
        return "never"
    sd = np.sqrt((z.var() + r.var()) / 2)
    return f"{int(np.ceil(2 * ((zc + 0.84) * sd / gap) ** 2))}"


def boot_range(x, days, seed=7, n=4000):
    """day-clustered bootstrap 95 % range of the mean → (lo, hi), or None below 3 fills / 3 days."""
    x, days = np.asarray(x, float), np.asarray(days)
    u = np.unique(days)
    if len(x) < 3 or len(u) < 3:                            # 1–2 days: a day bootstrap has no range to speak of
        return None
    by = [x[days == d] for d in u]
    rng = np.random.default_rng(seed)
    ms = [np.concatenate([by[i] for i in rng.integers(0, len(u), len(u))]).mean() for _ in range(n)]
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return float(lo), float(hi)


def boot(x, days, seed=7, n=4000):
    r = boot_range(x, days, seed, n)
    return "–" if r is None else f"{r[0]:+.2f} … {r[1]:+.2f}".replace("-", "−")


def pace(o_all, end):
    """the line's pace over the WHOLE current batch (DECISION_LOG 484 — not since its freeze, which can sit in a quiet stretch): (trades / day,
    trading days / day) from every current-batch fill in its zone (or every signal / flip for FB_CONFIRM / TP +2). PURE."""
    o = pd.to_datetime(pd.Series(o_all), format="mixed")
    span = (end - B25_START).total_seconds() / 86400
    if span <= 0 or not len(o):
        return 0.0, 0.0
    return len(o) / span, o.dt.strftime("%Y-%m-%d").nunique() / span


def eta_days(n_more, d_more, p):
    """days until n_more trades AND d_more trading days arrive at pace p = (trades / day, days / day); 0 = now, inf = no pace. PURE."""
    tp, dp = p
    need = []
    if n_more > 0:
        need.append(n_more / tp if tp > 0 else float("inf"))
    if d_more > 0:
        need.append(d_more / dp if dp > 0 else float("inf"))
    return max(need) if need else 0.0


def when(days, end):
    """days from the export → 'now' / 'Nov 14' / '> 3 months' / 'no pace yet'. PURE."""
    if days == 0:
        return "now"
    if days == float("inf"):
        return "no pace yet"
    return "> 3 months" if days > 90 else f"{end + pd.Timedelta(days=days):%b %d}"


def ladder(clean, cdays, sign, n_min, d_min):
    """DECISION_LOG 447 line call from CLEAN fills only → PASS (past checkpoint, 95 % range on the line's side: no haircut) · STRADDLE
    (past checkpoint, range crosses 0) · AGAINST (past checkpoint, range on the wrong side) · PRE (before the checkpoint: haircut stays).
    sign −1: the line claims the blocked fills lose · +1: claims a gain per trade."""
    c = pd.Series(clean, dtype=float)
    d = pd.Series(list(cdays), index=c.index)[c.notna()]
    c = c.dropna()
    if len(c) < n_min or d.nunique() < d_min:
        return "PRE"
    r = boot_range(c.values, d.values)
    if r is None:
        return "PRE"
    lo, hi = r
    if (sign < 0 and hi < 0) or (sign > 0 and lo > 0):
        return "PASS"
    if (sign < 0 and lo > 0) or (sign > 0 and hi < 0):
        return "AGAINST"
    return "STRADDLE"


def fmt(x):
    return f"{x:+.2f}".replace("-", "−")


def row(sleeve, line, built, clean, cdays, sign, fz, rest=None, p=(0.0, 0.0), end=None, clean_rest=0, p_rest=(0.0, 0.0), kept_avg=float("nan")):
    spec, n_min, d_min, bt = LINES[line]
    b = pd.Series(built, dtype=float).dropna()
    c = pd.Series(clean, dtype=float)
    cdays = pd.Series(list(cdays), index=c.index)[c.notna()]   # days stay aligned with the fills kept (a NaN result drops both)
    c = c.dropna()
    measure = "blocked trades" if sign < 0 else "gain per trade"
    bs = f"{len(b)} · {fmt(b.mean())}" if len(b) else "0"
    cs = f"{len(c)} on {len(set(cdays))} d · {fmt(c.mean())}" if len(c) else "0"
    nd = n95(b, sign)
    if nd not in ("–", "never"):
        nd = str(max(int(nd), n_min))                       # never below the line's own checkpoint (tiny in-sample spread ≠ proof)
    gap = "–" if nd in ("–", "never") else f"{max(0, int(nd) - len(c))} more"
    nr = n95_vs_rest(b, rest) if rest is not None else "n/a"
    fzs = "not in git" if fz is None else f"{fz:%m-%d}"
    # (484) the deciding test + when the checkpoint / that test arrive at the WHOLE-batch pace
    yard = YARD.get(line, "money")
    end = end if end is not None else pd.Timestamp.utcnow().tz_localize(None)   # naive UTC like the fills
    if yard == "rest":   # the line's own operator bar is P(zone < rest) ≥ 0.90 → one-sided 90 % (z 1.282), 80 % power (deep review 484)
        nr90 = n95_vs_rest(b, rest, zc=1.282) if rest is not None else "n/a"
        need = None if nr90 in ("–", "never", "n/a") else max(int(nr90), n_min)
        dec = ("blocked < kept: never" if nr90 == "never" else f"blocked < kept (90 %): {nr90}/side"
               + ("" if need is None else f" ({max(0, need - len(c))} more blocked · {max(0, need - clean_rest)} more kept)"))
        more = None if need is None else max(0, need - len(c))
    else:
        dec = ("adds money: can't be sized yet (no built-on spread)" if nd == "–" else f"adds money: {nd}" + ("" if gap == "–" else f" ({gap})"))
        more = None if gap == "–" else max(0, int(nd) - len(c))
    d_cp = eta_days(n_min - len(c), d_min - len(set(cdays)), p)
    if (nd == "never" and yard == "money") or (yard == "rest" and nr90 == "never"):
        when_dec = "never (it does not point the line's way)"
    elif more is None:
        when_dec = "–"
    else:
        d_dec = eta_days(more, 0, p)
        if yard == "rest":                                    # both sides must arrive: the kept side at its own pace
            d_dec = max(d_dec, eta_days(max(0, need - clean_rest), 0, p_rest))
        when_dec = when(max(d_dec, d_cp), end)                # a line is never decided before its checkpoint
    when_cp = when(d_cp, end)
    TL.append(dict(sleeve=sleeve, line=line, n=len(c), days=len(set(cdays)), avg=(float(c.mean()) if len(c) else float("nan")), sign=sign,
                   yard=yard, kept_avg=kept_avg,
                   p=p, n_min=n_min, d_min=d_min, cp=when_cp, dec=dec, decided=when_dec, end=end))
    return (f"| {SHOW.get(sleeve, sleeve)} | {LABEL.get(line, line)} | {measure} | {fzs} | {bs} | {cs} | {len(c)}/{n_min} trades, "
            f"{len(set(cdays))}/{d_min} days | {boot(c.values, cdays.values)} | **{dec}** | {p[0]:.2f} trades/day | {when_cp} | {when_dec} | "
            f"{nd} ({gap}) | {nr} | {bt} | {THEORY.get(line, '–')} | {luck_cell(sleeve, line)} |")


def inputs():
    """→ (Z per-fill zones, fb FB_CONFIRM Δ per signal, fx FAN flips TP +2 Δ per fill) from the pipeline's csv files."""
    Z = pd.read_csv(os.path.join(SC, "zones.csv"), low_memory=False)
    Z["o"] = pd.to_datetime(Z.o, format="mixed")
    fb = pd.read_csv(os.path.join(SC, "fb_combo.csv"))
    fb["o"] = pd.to_datetime(fb.k, format="mixed")
    fb["d"] = fb.cf510 - fb.now510                                 # vs TODAY's exit (+5 / −10), not the +8 / −15 some fills booked
    if "cf_why" in fb:   # (471 deep review) confirm-entered fills (after the DL 428 ship) have no S4 counterfactual — their Δ is replica
        fb.loc[fb.cf_why.astype(str).eq("CONFIRMED_LIVE"), "d"] = np.nan   # noise, never FB_CONFIRM evidence; the revert gate reads them
    fx = pd.read_csv(os.path.join(SC, "flip_exits.csv"))
    fx["o"] = pd.to_datetime(fx.k, format="mixed")
    fx["d"] = fx["TP2"] - np.where(fx.src == "B25", fx.booked, fx["stack"])
    return Z, fb, fx


def clean_cohort(line, sleeve, Z, fb, fx):
    """→ (clean values, their days, sign) — current-batch fills opened ≥ the line's freeze; nothing counts before the freeze is in git."""
    fz = freeze_ts(LINES[line][0])
    if line == "FB_CONFIRM":
        g, v, sign = fb, "d", +1
        cl = g.iloc[0:0] if fz is None else g[g.o >= fz]
    elif line == "EXIT TP2":
        g, v, sign = fx, "d", +1
        cl = g.iloc[0:0] if fz is None else g[(g.src == "B25") & (g.o >= fz)]
    else:
        g, v, sign = Z[(Z.sl == sleeve) & Z[line].fillna(False).astype(bool)], "pct", -1
        cl = g.iloc[0:0] if fz is None else g[(g.src == "B25") & (g.o >= fz)]
    return cl[v], cl.o.dt.strftime("%Y-%m-%d"), sign


def line_calls(pairs, Z=None, fb=None, fx=None):
    """pairs = [(sleeve, line)] → {(sleeve, line): DL 447 call} (keyed per sleeve: one line name read on two sleeves is two calls);
    a line without a registered freeze is PRE (never clean)."""
    if Z is None:
        Z, fb, fx = inputs()
    out = {}
    for sleeve, line in pairs:
        if line not in LINES:
            out[(sleeve, line)] = "PRE"
            continue
        c, d, sign = clean_cohort(line, sleeve, Z, fb, fx)
        out[(sleeve, line)] = ladder(c, d, sign, LINES[line][1], LINES[line][2])
    return out


def shares():
    """per sleeve: the best line's gain over today's rules, %/day — master and current batch POOLED by each sleeve's own live days
    (best_dcr.csv scenario B × best_hold.csv days; the "line's gain" column of the checkpoint timeline)."""
    t = pd.read_csv(os.path.join(SC, "best_dcr.csv"))
    t = t[t.scen == "B"].set_index("sleeve")
    hp = os.path.join(SC, "best_hold.csv")
    h = pd.read_csv(hp).set_index("sleeve") if os.path.exists(hp) else pd.DataFrame()
    out = {}
    for s_, r in t.iterrows():
        bd = float(h.loc[s_, "b25_days"]) if s_ in h.index else 1.0
        md = float(h.loc[s_, "m_days"]) if (s_ in h.index and h.loc[s_, "m_n0"] > 0 and r.m_before == r.m_before) else 0.0
        gm = (r.m_best - r.m_before) if md else 0.0
        out[s_] = (gm * md + (r.b25_best - r.b25_before) * bd) / (md + bd)
    return out


def timeline(tl, review_at, share):
    """→ markdown of the ④ checkpoint timeline (operator 11-06: "checkpoint timeline", DECISION_LOG 485) from the rows ③ printed. PURE."""
    def way(r):
        if not r["n"] or r["avg"] != r["avg"]:
            return ""
        if r.get("yard") == "rest":                     # its own test: clean blocked avg below the clean kept avg (deep review 485)
            k = r.get("kept_avg", float("nan"))
            return "" if k != k else (f" ✓ (kept {fmt(k)})" if r["avg"] < k else f" ✗ (kept {fmt(k)})")
        return " ✓" if (r["avg"] < 0 if r["sign"] < 0 else r["avg"] > 0) else " ✗"
    rows = sorted(tl, key=lambda r: -share.get(r["sleeve"], 0.0))
    end = max(r["end"] for r in tl) if tl else review_at
    dd = max(0.0, (review_at - end).total_seconds() / 86400)
    late = review_at <= end
    a = [f"### ④a Where the lines stand by the review ({review_at:%b %d}, {dd:.1f} days after this export)"
         + (" — ⚠ the review date is not after this export: pass a later date (run_best_case.py <export> <date>)" if late else ""), "",
         f"Clean trades / trading days; checkpoint = N trades on D days. Pace = the line's trades per day over the whole current batch. "
         "Line's gain = how much the line adds over today's rules, %/day, master and current batch pooled by each sleeve's own live days "
         "(a line that leaves master unchanged still counts master days, so it reads below its current-batch gain; the gains do not add up to "
         "table ②'s totals, which compound on the shared account).", "",
         "| Line (sleeve) | Now | Pace | Expected by the review | Checkpoint (N / days) | Reaches it ≈ | Line's gain (%/day) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        tp, dp = r["p"]
        ne, de = r["n"] + tp * dd, min(r["days"] + dp * dd, r["days"] + np.ceil(dd))
        a.append(f"| {LABEL.get(r['line'], r['line'])} ({SHOW.get(r['sleeve'], r['sleeve'])}) | {r['n']} / {r['days']} | {tp:.2f} / day | "
                 f"~{ne:.0f} / ~{de:.0f} | {r['n_min']} / {r['d_min']} | **{r['cp']}** | {fmt(share.get(r['sleeve'], 0.0))} |")
    b = ["", "### ④b What each line needs to be decided", "",
         "Deciding test = the one the line's own gate uses (blocked < kept at 90 % for the Market-hot lines, adds money for the rest). ✓ / ✗ = the "
         "clean trades so far point the line's way or not (Market-hot lines: blocked vs kept). Dates are projections — zone trades come in bursts.", "",
         "| Line (sleeve) | Line's gain | Clean so far | Deciding test: trades needed | Checkpoint ≈ | Decided ≈ |", "|---|---|---|---|---|---|"]
    for r in rows:
        cs = f"{r['n']} on {r['days']} d · {fmt(r['avg'])}{way(r)}" if r["n"] else "0"
        b.append(f"| {LABEL.get(r['line'], r['line'])} ({SHOW.get(r['sleeve'], r['sleeve'])}) | {fmt(share.get(r['sleeve'], 0.0))} | {cs} | "
                 f"{r['dec']} | {r['cp']} | **{r['decided']}** |")
    return "\n".join(a + b)


def main():
    TL.clear()                                          # one ④ per run, even if main() is called twice in-process (deep review 485)
    Z, fb, fx = inputs()
    lp = os.path.join(SC, "luck.csv")
    if os.path.exists(lp):
        LUCK.update({(r.sleeve, r.line): r._asdict() for r in pd.read_csv(lp).itertuples(index=False)})
    best = pd.read_csv(os.path.join(SC, "best_dcr_noflip.csv"))
    best = best[(best.scen == "B") & best.lines.notna() & (best.lines.astype(str) != "")]
    out = ["| Sleeve | Line | Measured on | Frozen | Built on: N · avg % | Clean since frozen: N · avg % | Checkpoint progress | Clean 95 % range "
           "| Deciding test: trades needed (still needed) | Pace (whole batch) | Checkpoint ≈ | Deciding test ≈ "
           "| N to prove it adds money (still needed) | N to prove blocked < kept (per side) | Backtest | Theory written | Luck test (built-on trades) |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    end = max(Z.o.max(), fb.o.max() if len(fb) else Z.o.max(), fx.o.max() if len(fx) else Z.o.max())   # the export's last fill
    rows = []
    for r in best.itertuples():
        for line in str(r.lines).split(" + "):
            if line == "FB_CONFIRM":
                continue                                    # entry-timing line, read below from fb_combo
            if line not in LINES:
                rows.append((0, f"| {SHOW.get(r.sleeve, r.sleeve)} | {line} | – | – | – | – | no freeze registered | – | – | – | – | – | – | – | – | – | {luck_cell(r.sleeve, line)} |"))
                continue
            fz = freeze_ts(LINES[line][0])
            g = Z[(Z.sl == r.sleeve) & Z[line].fillna(False).astype(bool)]
            if fz is None:                                  # freeze commit not found → nothing counts as clean
                pre, cl = g, g.iloc[0:0]
            else:
                pre, cl = g[(g.src == "MASTER") | (g.o < fz)], g[(g.src == "B25") & (g.o >= fz)]
            g_all = Z[Z.sl == r.sleeve]
            keep = g_all[~g_all[line].fillna(False).astype(bool)]
            keep = keep if fz is None else keep[(keep.src == "MASTER") | (keep.o < fz)]
            g25 = g[g.src == "B25"]
            kc = 0 if fz is None else int(((g_all.src == "B25") & (g_all.o >= fz) & ~g_all[line].fillna(False).astype(bool)).sum())
            k25 = g_all[(g_all.src == "B25") & ~g_all[line].fillna(False).astype(bool)]
            kav = float("nan") if fz is None else float(k25[k25.o >= fz].pct.mean()) if (k25.o >= fz).any() else float("nan")
            rows.append((r.b25_best - r.b25_before, row(r.sleeve, line, pre.pct, cl.pct, cl.o.dt.strftime("%Y-%m-%d"), -1, fz, keep.pct,
                                                        pace(g25.o, end), end, kc, pace(k25.o, end), kav)))
    # FB_CONFIRM: Δ = confirm-entry result − today's-exit result, per signal (a skip books 0)
    fz = freeze_ts(LINES["FB_CONFIRM"][0])
    pre, cl = (fb, fb.iloc[0:0]) if fz is None else (fb[fb.o < fz], fb[fb.o >= fz])
    rows.append((9, row("FLATBASE_SHORT", "FB_CONFIRM", pre.d, cl.d, cl.o.dt.strftime("%Y-%m-%d"), +1, fz, None,
                        pace(fb[(fb.o >= B25_START) & fb.d.notna()].o, end), end)))
    # FAN flips TP +2: Δ = TP2 − booked (current batch) / − stack (master)
    fz = freeze_ts(LINES["EXIT TP2"][0])
    if fz is None:
        pre, cl = fx, fx.iloc[0:0]
    else:
        pre, cl = fx[(fx.src == "MASTER") | (fx.o < fz)], fx[(fx.src == "B25") & (fx.o >= fz)]
    rows.append((8, row("FLIP:FAN_RATIO_GATE", "EXIT TP2", pre.d, cl.d, cl.o.dt.strftime("%Y-%m-%d"), +1, fz, None,
                        pace(fx[fx.src == "B25"].o, end), end)))
    out += [x for _, x in sorted(rows, key=lambda t: -t[0])]
    print("\n".join(out))
    review_at = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else REVIEW_AT
    print(TIMELINE_MARK)
    print(timeline(TL, review_at, shares()))


if __name__ == "__main__":
    main()
