#!/usr/bin/env python3
"""📊 Best-case view ③ — how much each best line can be trusted (operator 2026-11-03: "how many more trades we need … to really trust the
results? with high confidence level?").

Per best line (from best_dcr_noflip.csv, plus the FAN flips TP +2 exit):
  BUILT ON  the trades the line was found on — every master fill + current-batch fills opened before the line was frozen (in-sample).
  CLEAN     current-batch fills opened ≥ the freeze (the DL commit + 10 min, as its scout counts them): never used to pick the line.
  measure   block line → the BLOCKED fills' booked P&L % (the line claims they lose) · exit / entry-timing line → Δ = after − booked per fill.
  N for 95 %  trades for an 80 % chance of showing the effect at 95 % confidence: n = ((1.645 + 0.84) × sd ÷ |mean|)², sd and mean from
            BUILT ON, floored at the checkpoint N — still optimistic (in-sample means are inflated, fills cluster by day);
            "never" = BUILT ON does not even point the line's way. FB_CONFIRM's BUILT ON is the current batch before the freeze + 1 B24 fill.
  clean 95 % range  day-clustered bootstrap of the clean mean (4,000 resamples, seed 7); "–" below 3 clean fills.
  backtest  what the backtest said at registration (its scout docstring) — a cross-check, never a veto (MASTER OVER BACKTEST).
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


def boot(x, days, seed=7, n=4000):
    x, days = np.asarray(x, float), np.asarray(days)
    u = np.unique(days)
    if len(x) < 3 or len(u) < 3:                            # 1–2 days: a day bootstrap has no range to speak of
        return "–"
    by = [x[days == d] for d in u]
    rng = np.random.default_rng(seed)
    ms = [np.concatenate([by[i] for i in rng.integers(0, len(u), len(u))]).mean() for _ in range(n)]
    lo, hi = np.percentile(ms, [2.5, 97.5])
    return f"{lo:+.2f} … {hi:+.2f}".replace("-", "−")


def fmt(x):
    return f"{x:+.2f}".replace("-", "−")


def row(sleeve, line, built, clean, cdays, sign, fz):
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
    fzs = "not in git" if fz is None else f"{fz:%m-%d}"
    return (f"| {SHOW.get(sleeve, sleeve)} | {LABEL.get(line, line)} | {measure} | {fzs} | {bs} | {cs} | {len(c)}/{n_min} trades, "
            f"{len(set(cdays))}/{d_min} days | {boot(c.values, cdays.values)} | {nd} ({gap}) | {bt} |")


def main():
    Z = pd.read_csv(os.path.join(SC, "zones.csv"), low_memory=False)
    Z["o"] = pd.to_datetime(Z.o, format="mixed")
    best = pd.read_csv(os.path.join(SC, "best_dcr_noflip.csv"))
    best = best[(best.scen == "B") & best.lines.notna() & (best.lines.astype(str) != "")]
    out = ["| Sleeve | Line | Measured on | Frozen | Built on: N · avg % | Clean since frozen: N · avg % | Checkpoint progress | Clean 95 % range "
           "| N for 95 % (still needed) | Backtest |", "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for r in best.itertuples():
        for line in str(r.lines).split(" + "):
            if line == "FB_CONFIRM":
                continue                                    # entry-timing line, read below from fb_combo
            if line not in LINES:
                rows.append((0, f"| {SHOW.get(r.sleeve, r.sleeve)} | {line} | – | – | – | – | no freeze registered | – | – | – |"))
                continue
            fz = freeze_ts(LINES[line][0])
            g = Z[(Z.sl == r.sleeve) & Z[line].fillna(False).astype(bool)]
            if fz is None:                                  # freeze commit not found → nothing counts as clean
                pre, cl = g, g.iloc[0:0]
            else:
                pre, cl = g[(g.src == "MASTER") | (g.o < fz)], g[(g.src == "B25") & (g.o >= fz)]
            rows.append((r.b25_best - r.b25_before, row(r.sleeve, line, pre.pct, cl.pct, cl.o.dt.strftime("%Y-%m-%d"), -1, fz)))
    # FB_CONFIRM: Δ = confirm-entry result − booked, per signal (a skip books 0)
    fb = pd.read_csv(os.path.join(SC, "fb_combo.csv"))
    fz = freeze_ts(LINES["FB_CONFIRM"][0])
    fb["o"] = pd.to_datetime(fb.k, format="mixed")
    fb["d"] = fb.cf510 - fb.booked
    pre, cl = (fb, fb.iloc[0:0]) if fz is None else (fb[fb.o < fz], fb[fb.o >= fz])
    rows.append((9, row("FLATBASE_SHORT", "FB_CONFIRM", pre.d, cl.d, cl.o.dt.strftime("%Y-%m-%d"), +1, fz)))
    # FAN flips TP +2: Δ = TP2 − booked (current batch) / − stack (master)
    fx = pd.read_csv(os.path.join(SC, "flip_exits.csv"))
    fz = freeze_ts(LINES["EXIT TP2"][0])
    fx["o"] = pd.to_datetime(fx.k, format="mixed")
    fx["d"] = fx["TP2"] - np.where(fx.src == "B25", fx.booked, fx["stack"])
    if fz is None:
        pre, cl = fx, fx.iloc[0:0]
    else:
        pre, cl = fx[(fx.src == "MASTER") | (fx.o < fz)], fx[(fx.src == "B25") & (fx.o >= fz)]
    rows.append((8, row("FLIP:FAN_RATIO_GATE", "EXIT TP2", pre.d, cl.d, cl.o.dt.strftime("%Y-%m-%d"), +1, fz)))
    out += [x for _, x in sorted(rows, key=lambda t: -t[0])]
    print("\n".join(out))


if __name__ == "__main__":
    main()
