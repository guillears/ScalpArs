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
SHOW = {"FLIP:FAN_RATIO_GATE": "FAN flips", "FRENZY_WILLY": "WILLY", "FRENZY_WILLY_HT": "WILLY_HT", "SPIKE_FADE": "Spike Fade",
        "FLATBASE_SHORT": "Flatbase", "FRENZY_LITE_RETRY": "LITE RETRY", "SURGE_LONG": "SURGE Long", "MOMENTUM_LONG": "Momentum Long",
        "FRENZY_LONG": "FRENZY_LONG"}
LABEL = {"EXIT TP2": "Exit TP +2", "MKT_HOT": "Market-hot", "SURGE_BTC_HOT": "BTC RSI ≥ 78"}


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


def n95_vs_rest(z, r):
    """trades needed PER SIDE to show the blocked fills do worse than the kept ones (two-sample, 80 % power, one-sided 95 %)."""
    z, r = pd.Series(z, dtype=float).dropna(), pd.Series(r, dtype=float).dropna()
    if len(z) < 3 or len(r) < 3:
        return "–"
    gap = r.mean() - z.mean()
    if gap <= 0:
        return "never"
    sd = np.sqrt((z.var() + r.var()) / 2)
    return f"{int(np.ceil(2 * ((1.645 + 0.84) * sd / gap) ** 2))}"


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


def row(sleeve, line, built, clean, cdays, sign, fz, rest=None):
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
    return (f"| {SHOW.get(sleeve, sleeve)} | {LABEL.get(line, line)} | {measure} | {fzs} | {bs} | {cs} | {len(c)}/{n_min} trades, "
            f"{len(set(cdays))}/{d_min} days | {boot(c.values, cdays.values)} | {nd} ({gap}) | {nr} | {bt} |")


def inputs():
    """→ (Z per-fill zones, fb FB_CONFIRM Δ per signal, fx FAN flips TP +2 Δ per fill) from the pipeline's csv files."""
    Z = pd.read_csv(os.path.join(SC, "zones.csv"), low_memory=False)
    Z["o"] = pd.to_datetime(Z.o, format="mixed")
    fb = pd.read_csv(os.path.join(SC, "fb_combo.csv"))
    fb["o"] = pd.to_datetime(fb.k, format="mixed")
    fb["d"] = fb.cf510 - fb.now510                                 # vs TODAY's exit (+5 / −10), not the +8 / −15 some fills booked
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


def main():
    Z, fb, fx = inputs()
    best = pd.read_csv(os.path.join(SC, "best_dcr_noflip.csv"))
    best = best[(best.scen == "B") & best.lines.notna() & (best.lines.astype(str) != "")]
    out = ["| Sleeve | Line | Measured on | Frozen | Built on: N · avg % | Clean since frozen: N · avg % | Checkpoint progress | Clean 95 % range "
           "| N to prove it adds money (still needed) | N to prove blocked < kept (per side) | Backtest |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for r in best.itertuples():
        for line in str(r.lines).split(" + "):
            if line == "FB_CONFIRM":
                continue                                    # entry-timing line, read below from fb_combo
            if line not in LINES:
                rows.append((0, f"| {SHOW.get(r.sleeve, r.sleeve)} | {line} | – | – | – | – | no freeze registered | – | – | – | – |"))
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
            rows.append((r.b25_best - r.b25_before, row(r.sleeve, line, pre.pct, cl.pct, cl.o.dt.strftime("%Y-%m-%d"), -1, fz, keep.pct)))
    # FB_CONFIRM: Δ = confirm-entry result − today's-exit result, per signal (a skip books 0)
    fz = freeze_ts(LINES["FB_CONFIRM"][0])
    pre, cl = (fb, fb.iloc[0:0]) if fz is None else (fb[fb.o < fz], fb[fb.o >= fz])
    rows.append((9, row("FLATBASE_SHORT", "FB_CONFIRM", pre.d, cl.d, cl.o.dt.strftime("%Y-%m-%d"), +1, fz)))
    # FAN flips TP +2: Δ = TP2 − booked (current batch) / − stack (master)
    fz = freeze_ts(LINES["EXIT TP2"][0])
    if fz is None:
        pre, cl = fx, fx.iloc[0:0]
    else:
        pre, cl = fx[(fx.src == "MASTER") | (fx.o < fz)], fx[(fx.src == "B25") & (fx.o >= fz)]
    rows.append((8, row("FLIP:FAN_RATIO_GATE", "EXIT TP2", pre.d, cl.d, cl.o.dt.strftime("%Y-%m-%d"), +1, fz)))
    out += [x for _, x in sorted(rows, key=lambda t: -t[0])]
    print("\n".join(out))


if __name__ == "__main__":
    main()
