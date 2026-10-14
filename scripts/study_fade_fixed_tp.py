#!/usr/bin/env python3
"""SPIKE_FADE fixed take-profit vs today's exits (operator 2026-10-13): add a fixed TP at +X % net (0.40 / 0.45 / 0.50 / 0.55) on top of
the current fade exit stack. First touch on the fill's recorded net peak: peak ≥ X → the TP fills first at X (the peak is reached before
the actual exit); peak < X → the TP never prints, the fill keeps today's priced outcome (stack_pct). Sample: master pool kept SPIKE_FADE
fills (today's stack, $/pt = stack_pnl / stack_pct at today's ticket), per batch. Output reports/FADE_FIXED_TP.txt."""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XS = (0.40, 0.45, 0.50, 0.55)
rng = np.random.default_rng(7)


def load_master():
    M = pd.read_csv(os.path.join(ROOT, "reports", "MASTER_POOL_stacked.csv"), low_memory=False)
    F = M[(M.entry_strategy == "SPIKE_FADE") & (M.stack_keep == True)].copy()   # noqa: E712
    for c in ("stack_pct", "stack_pnl", "peak_pnl"):
        F[c] = pd.to_numeric(F[c], errors="coerce")
    F["upp"] = F.stack_pnl / F.stack_pct.replace(0, np.nan)
    F["day"] = F.opened_at.astype(str).str[:10]
    return F


def tp_pct(base, peak, x):
    return np.where(peak >= x, x, base)


def day_p(d, day):
    g = pd.DataFrame(dict(d=d, day=day)).groupby("day").d.agg(["sum", "size"])
    if len(g) < 3:
        return float("nan")
    idx = rng.integers(0, len(g), size=(4000, len(g)))
    s, n = g["sum"].values, g["size"].values
    return float(((s[idx].sum(1) / n[idx].sum(1)) > 0).mean())


def main():
    F = load_master()
    L = [__doc__.splitlines()[0], f"master kept SPIKE_FADE fills: {len(F)} · peak ≥ X share " + " · ".join(f"{x:.2f}: {(F.peak_pnl >= x).mean() * 100:.0f} %" for x in XS), ""]
    def s(p, u):
        return f"{len(p):3} · {(p > 0).mean() * 100:3.0f} % · {p.mean():+.3f} % · ${np.nansum(p * u):+,.0f}"
    L.append(f"{'batch':6} {'TODAY (actual exits)':>30} " + " ".join(f"{'TP +' + format(x, '.2f'):>30}" for x in XS))
    for era, g in list(F.groupby("era")) + [("TOTAL", F)]:
        row = f"{era:6} {s(g.stack_pct.values, g.upp.values):>30} "
        for x in XS:
            row += f"{s(tp_pct(g.stack_pct.values, g.peak_pnl.values, x), g.upp.values):>30} "
        L.append(row)
    L.append("")
    for x in XS:
        n = tp_pct(F.stack_pct.values, F.peak_pnl.values, x); d = n - F.stack_pct.values
        cut = d < -1e-9; gain = d > 1e-9
        L.append(f"TP +{x:.2f}: Δ avg {d.mean():+.3f} %/fill · Σ Δ {d.sum():+.2f} pts · Δ$ {np.nansum(d * F.upp.values):+,.0f} · runners cut {cut.sum()} "
                 f"({d[cut].sum():+.2f} pts) · fills lifted {gain.sum()} ({d[gain].sum():+.2f} pts) · day-bootstrap P(Δ>0) {day_p(d, F.day.values):.2f}")
    txt = "\n".join(L); open(os.path.join(ROOT, "reports", "FADE_FIXED_TP.txt"), "w").write(txt + "\n"); print(txt)


if __name__ == "__main__":
    main()
